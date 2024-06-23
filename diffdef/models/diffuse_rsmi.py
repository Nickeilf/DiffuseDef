from .diffuse import DiffuseModel2, DiffuseModel3, DiffuseBERT
from .base import BaseModel
from ..utils import get_activation_layer, get_loss_function

from ..utils.io import load_model

from .rsmi.model_noise_forward import roberta_noise_forward, bert_noise_forward
from .rsmi.model_adv import SeqClsWrapper

import math
import types
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
from transformers import (
    AutoModel,
    AutoConfig,
    BertLayer,
    AutoModelForSequenceClassification,
)
from diffusers import DDPMScheduler, UNet1DModel


class DiffuseModelRSMI(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_train_timesteps = args.num_train_timesteps
        self.num_test_timesteps = 10
        self.device = "cuda"

        # load model and freeze weight
        self.cls_config = AutoConfig.from_pretrained(
            args.pretrained_model,
            num_labels=args.output_dim,
            output_hidden_states=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_model, from_tf=False, config=self.cls_config
        ).to(self.device)

        if "bert-" in args.pretrained_model:
            model.bert.encoder.forward = types.MethodType(
                bert_noise_forward, model.bert.encoder
            )
        elif "roberta" in args.pretrained_model:
            model.roberta.encoder.forward = types.MethodType(
                roberta_noise_forward, model.roberta.encoder
            )
        else:
            raise Exception("Specify Base model correctly...")

        model.config.single_layer = args.single_layer
        model.config.num_labels = args.output_dim
        model.config.nth_layers = args.nth_layers
        model.config.noise_eps = args.noise_eps

        args.num_classes = args.output_dim
        args.device = self.device

        # load pretrained rsmi model
        self.classifier = SeqClsWrapper(model, args)
        checkpoint = torch.load(args.cls_path)
        self.classifier.load_state_dict(checkpoint["model"], strict=False)
        self.classifier.to(self.device)

        self.classifier.loss_fn = get_loss_function(args.loss_fn)

        for param in self.classifier.parameters():
            param.requires_grad = False

        # diffusion-related
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps
        )

        self.diffusion_config = deepcopy(model.config)
        self.diffusion_config.num_hidden_layers = args.get(
            "diffusion_layer", model.config.num_hidden_layers
        )
        self.diffusion_config.num_attention_heads = args.get(
            "diffusion_layer", model.config.num_hidden_layers
        )
        self.diffusion_layer = DiffuseBERT(self.diffusion_config)

    def forward(self, batch):
        inputs_dict = batch["text"]
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

        text_hiddens = outputs.last_hidden_state  # (B, S, H)

        # Step 1: sample noise
        noise = torch.randn(text_hiddens.shape).to(self.device)  # (B, S, H)
        bs = text_hiddens.shape[0]

        # Sample a random timestep for each text
        timesteps = torch.randint(
            0, self.num_train_timesteps, (bs,), device=self.device
        ).long()

        # Add noise to the clean text according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_hiddens = self.noise_scheduler.add_noise(text_hiddens, noise, timesteps)

        # Step 2: predict noise
        noise_pred = self.diffusion_layer(noisy_hiddens, timesteps)
        noise_loss = F.mse_loss(noise_pred, noise)

        loss = noise_loss

        results = {
            "noise_pred": noise_pred,
            "loss": loss,
        }
        return results

    def denoise(self, noisy_hiddens, timestep):
        bs = noisy_hiddens.shape[0]
        for t_cur in range(timestep):
            t = torch.ones((bs,), device=self.device).long() * t_cur
            noise_pred = self.diffusion_layer(noisy_hiddens, t)

            # 2. compute previous image: x_t -> x_t-1
            noisy_hiddens = self.noise_scheduler.step(
                noise_pred, t_cur, noisy_hiddens
            ).prev_sample

        return noisy_hiddens

    def base_model_classify(self, text_inputs):
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

        text_hiddens = outputs.last_hidden_state  # (B, S, H)

        text_pooled_hiddens = self.classifier.model.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.model.dropout(text_pooled_hiddens)

        logits = self.classifier.model.classifier(text_pooled_hiddens)

        return logits

    def classify(self, batch):
        labels = batch["label"]

        inputs_dict = batch["text"]
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

        text_hiddens = outputs.last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        text_pooled_hiddens = self.classifier.model.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.model.dropout(text_pooled_hiddens)

        logits = self.classifier.model.classifier(text_pooled_hiddens)
        loss = self.classifier.loss_fn(logits, labels)

        results = {
            "logits": logits,
            "loss": loss,
        }

        return results

    def predict(self, text_inputs):
        inputs_dict = text_inputs
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

        text_hiddens = outputs.last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        text_pooled_hiddens = self.classifier.model.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.model.dropout(text_pooled_hiddens)

        logits = self.classifier.model.classifier(text_pooled_hiddens)
        outputs = {
            "logits": logits,
        }

        return outputs

    def generate(self, input_ids, attention_mask=None, token_type_ids=None):
        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

        text_hiddens = outputs.last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        text_pooled_hiddens = self.classifier.model.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.model.dropout(text_pooled_hiddens)

        logits = self.classifier.model.classifier(text_pooled_hiddens)
        outputs = {
            "logits": logits,
        }

        return outputs


def input_masking_function(input_ids, indices, args):
    """Mask input tokens based on indices computed by grad_mask function"""

    masked_ids = input_ids.clone()

    for ids_, m_idx in zip(masked_ids, indices):  # for each sample in a batch
        for j in range(args.multi_mask):
            try:
                ids_[m_idx[j]] = args.mask_idx
            except:
                continue
    return masked_ids


class DiffuseModelRSMIEnsembleAvg(DiffuseModelRSMI):
    NUM_ENSEMBLE = 10

    def base_model_classify(self, text_inputs):
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

            text_hiddens = outputs.last_hidden_state  # (B, S, H)

        text_pooled_hiddens = self.classifier.model.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.model.dropout(text_pooled_hiddens)

        logits = self.classifier.model.classifier(text_pooled_hiddens)

        return logits

    def classify(self, batch):
        labels = batch["label"]

        inputs_dict = batch["text"]
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

            text_hiddens = outputs.last_hidden_state  # (B, S, H)
            bs, seq_len, hidden_size = text_hiddens.size()

            repeated_hiddens = text_hiddens.repeat(
                (self.NUM_ENSEMBLE, 1, 1)
            )  # (B x N, S, H)

            # randomly add noise
            noise = torch.randn(repeated_hiddens.shape).to(self.device)  # (B x N, S, H)

            # only 1 timestep
            timesteps = torch.ones((bs * self.NUM_ENSEMBLE,), device=self.device).long()

            # Add noise to the clean text according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_hiddens = self.noise_scheduler.add_noise(
                repeated_hiddens, noise, timesteps
            )

            # denoise and average
            text_hiddens = self.denoise(noisy_hiddens, self.num_test_timesteps)
            text_hiddens = text_hiddens.reshape(
                (bs, self.NUM_ENSEMBLE, seq_len, hidden_size)
            )
            text_hiddens = text_hiddens.mean(dim=1)

            text_pooled_hiddens = self.classifier.model.bert.pooler(
                text_hiddens
            )  # (B, H)
            text_pooled_hiddens = self.classifier.model.dropout(text_pooled_hiddens)

            logits = self.classifier.model.classifier(text_pooled_hiddens)
            loss = self.classifier.loss_fn(logits, labels)

        results = {
            "logits": logits,
            "loss": loss,
        }

        return results

    def predict(self, text_inputs):
        inputs_dict = text_inputs
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        self.classifier.zero_grad(set_to_none=True)
        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

            text_hiddens = outputs.last_hidden_state  # (B, S, H)

            bs, seq_len, hidden_size = text_hiddens.size()

            repeated_hiddens = text_hiddens.repeat(
                (self.NUM_ENSEMBLE, 1, 1)
            )  # (B x N, S, H)

            # randomly add noise
            noise = torch.randn(repeated_hiddens.shape).to(self.device)  # (B x N, S, H)

            # only 1 timestep
            timesteps = torch.ones((bs * self.NUM_ENSEMBLE,), device=self.device).long()

            # Add noise to the clean text according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_hiddens = self.noise_scheduler.add_noise(
                repeated_hiddens, noise, timesteps
            )

            # denoise and average
            text_hiddens = self.denoise(noisy_hiddens, self.num_test_timesteps)
            text_hiddens = text_hiddens.reshape(
                (bs, self.NUM_ENSEMBLE, seq_len, hidden_size)
            )
            text_hiddens = text_hiddens.mean(dim=1)

            text_pooled_hiddens = self.classifier.model.bert.pooler(
                text_hiddens
            )  # (B, H)
            text_pooled_hiddens = self.classifier.model.dropout(text_pooled_hiddens)

        logits = self.classifier.model.classifier(text_pooled_hiddens)
        outputs = {
            "logits": logits,
        }

        return outputs


class DiffuseModelRSMI2(DiffuseModelRSMI):
    # for roberta

    def classify(self, batch):
        labels = batch["label"]

        inputs_dict = batch["text"]
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

        text_hiddens = outputs.last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        logits = self.classifier.model.classifier(text_hiddens)
        loss = self.classifier.loss_fn(logits, labels)

        results = {
            "logits": logits,
            "loss": loss,
        }

        return results

    def predict(self, text_inputs):
        inputs_dict = text_inputs
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

        text_hiddens = outputs.last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        logits = self.classifier.model.classifier(text_hiddens)
        outputs = {
            "logits": logits,
        }

        return outputs


class DiffuseModelRSMI2EnsembleAvg(DiffuseModelRSMI2):
    NUM_ENSEMBLE = 10

    def classify(self, batch):
        labels = batch["label"]

        inputs_dict = batch["text"]
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

            text_hiddens = outputs.last_hidden_state  # (B, S, H)
            bs, seq_len, hidden_size = text_hiddens.size()

            repeated_hiddens = text_hiddens.repeat(
                (self.NUM_ENSEMBLE, 1, 1)
            )  # (B x N, S, H)

            # randomly add noise
            noise = torch.randn(repeated_hiddens.shape).to(self.device)  # (B x N, S, H)

            # only 1 timestep
            timesteps = torch.ones((bs * self.NUM_ENSEMBLE,), device=self.device).long()

            # Add noise to the clean text according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_hiddens = self.noise_scheduler.add_noise(
                repeated_hiddens, noise, timesteps
            )

            # denoise and average
            text_hiddens = self.denoise(noisy_hiddens, self.num_test_timesteps)
            text_hiddens = text_hiddens.reshape(
                (bs, self.NUM_ENSEMBLE, seq_len, hidden_size)
            )
            text_hiddens = text_hiddens.mean(dim=1)

            logits = self.classifier.model.classifier(text_hiddens)
            loss = self.classifier.loss_fn(logits, labels)

        results = {
            "logits": logits,
            "loss": loss,
        }

        return results

    def predict(self, text_inputs):
        inputs_dict = text_inputs
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        indices, _ = self.classifier.grad_mask(
            input_ids, attention_mask, pred=None, mask_filter=True
        )
        masked_ids = input_masking_function(input_ids, indices, self.args)

        self.classifier.zero_grad(set_to_none=True)
        with torch.no_grad():
            outputs = self.classifier.get_encoder_outputs(masked_ids, attention_mask)

            text_hiddens = outputs.last_hidden_state  # (B, S, H)

            bs, seq_len, hidden_size = text_hiddens.size()

            repeated_hiddens = text_hiddens.repeat(
                (self.NUM_ENSEMBLE, 1, 1)
            )  # (B x N, S, H)

            # randomly add noise
            noise = torch.randn(repeated_hiddens.shape).to(self.device)  # (B x N, S, H)

            # only 1 timestep
            timesteps = torch.ones((bs * self.NUM_ENSEMBLE,), device=self.device).long()

            # Add noise to the clean text according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_hiddens = self.noise_scheduler.add_noise(
                repeated_hiddens, noise, timesteps
            )

            # denoise and average
            text_hiddens = self.denoise(noisy_hiddens, self.num_test_timesteps)
            text_hiddens = text_hiddens.reshape(
                (bs, self.NUM_ENSEMBLE, seq_len, hidden_size)
            )
            text_hiddens = text_hiddens.mean(dim=1)

            logits = self.classifier.model.classifier(text_hiddens)

        outputs = {
            "logits": logits,
        }

        return outputs
