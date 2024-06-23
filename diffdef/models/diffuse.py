from .base import BaseModel, VanillaModel
from ..utils import get_activation_layer, get_loss_function

from ..utils.io import load_model

import math
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
from diffusers import DDPMScheduler


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffuseBERT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim = config.hidden_size

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, 2 * self.dim),
            nn.GELU(),
            nn.Linear(2 * self.dim, self.dim),
        )
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, hiddens, timestep):
        # TODO add attention mask here
        # Currently at inference time it only works with batch_size = 1 because attention mask is not used in this layer
        timestep_embedding = self.time_emb(timestep).unsqueeze(1)

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hiddens)

            hiddens = layer_outputs[0]
            hiddens += timestep_embedding

        return self.proj(hiddens)


class DiffuseModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.num_train_timesteps = args.num_train_timesteps
        self.num_test_timesteps = 10
        self.device = "cuda"

        # load model and freeze weight
        self.classifier = VanillaModel(args=args)
        self.classifier = load_model(
            self.classifier,
            os.path.join(args.cls_path, "best_checkpoint.pt"),
            self.device,
        )
        for param in self.classifier.parameters():
            param.requires_grad = False

        # diffusion-related
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps
        )

        self.diffusion_config = self.classifier.config.copy()
        self.diffusion_config.num_hidden_layers = args.diffusion_layer
        self.diffusion_layer = DiffuseBERT(self.diffusion_config)

    def forward(self, batch):
        text_hiddens = self.classifier.encode(batch)  # (B, S, H)

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

    def classify(self, batch):
        labels = batch["label"]
        text_hiddens = self.classifier.encode(batch)  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        text_pooled_hiddens = text_hiddens.mean(dim=1)  # (B, H)
        text_pooled_hiddens = self.classifier.proj(text_pooled_hiddens)

        logits = self.classifier.output(text_pooled_hiddens)
        loss = self.classifier.loss_fn(logits, labels)

        results = {
            "logits": logits,
            "loss": loss,
        }

        return results

    def predict(self, text_inputs):
        text_hiddens = self.classifier.backbone_model(
            **text_inputs
        ).last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)
        outputs = self.classifier.classify(text_hiddens)

        return outputs


class DiffuseModel2(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.num_train_timesteps = args.num_train_timesteps
        self.num_test_timesteps = 10
        self.device = "cuda"

        # load model and freeze weight
        self.cls_config = AutoConfig.from_pretrained(
            args.pretrained_model,
            num_labels=args.output_dim,
            finetuning_task="agnews",
            output_hidden_states=True,
        )
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            args.cls_path, from_tf=False, config=self.cls_config
        ).to(self.device)
        self.classifier.loss_fn = get_loss_function(args.loss_fn)

        for param in self.classifier.parameters():
            param.requires_grad = False

        # diffusion-related
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps
        )

        self.diffusion_config = deepcopy(self.classifier.config)
        self.diffusion_config.num_hidden_layers = args.get(
            "diffusion_layer", self.classifier.config.num_hidden_layers
        )
        self.diffusion_config.num_attention_heads = args.get(
            "diffusion_layer", self.classifier.config.num_hidden_layers
        )
        self.diffusion_layer = DiffuseBERT(self.diffusion_config)

    def base_model_classify(self, text_inputs):
        text_hiddens = self.classifier.bert(
            **text_inputs
        ).last_hidden_state  # (B, S, H)
        text_pooled_hiddens = self.classifier.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_pooled_hiddens)

        return logits

    def forward(self, batch):
        text_hiddens = self.classifier.bert(
            **batch["text"]
        ).last_hidden_state  # (B, S, H)

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

    def classify(self, batch):
        labels = batch["label"]
        text_hiddens = self.classifier.bert(
            **batch["text"]
        ).last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        text_pooled_hiddens = self.classifier.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_pooled_hiddens)
        loss = self.classifier.loss_fn(logits, labels)

        results = {
            "logits": logits,
            "loss": loss,
        }

        return results

    def predict(self, text_inputs):
        text_hiddens = self.classifier.bert(
            **text_inputs
        ).last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        text_pooled_hiddens = self.classifier.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_pooled_hiddens)
        outputs = {
            "logits": logits,
        }

        return outputs

    def generate(self, input_ids, attention_mask=None, token_type_ids=None):
        text_hiddens = self.classifier.bert(
            input_ids, attention_mask, token_type_ids
        ).last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        text_pooled_hiddens = self.classifier.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_pooled_hiddens)
        outputs = {
            "logits": logits,
        }

        return outputs


class DiffuseModel3(BaseModel):
    # for roberta

    def __init__(self, args):
        super().__init__(args)

        self.num_train_timesteps = args.num_train_timesteps
        self.num_test_timesteps = 10
        self.device = "cuda"

        # load model and freeze weight
        self.cls_config = AutoConfig.from_pretrained(
            args.pretrained_model,
            num_labels=args.output_dim,
            finetuning_task="agnews",
            output_hidden_states=True,
        )
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            args.cls_path, from_tf=False, config=self.cls_config
        ).to(self.device)
        self.classifier.loss_fn = get_loss_function(args.loss_fn)

        for param in self.classifier.parameters():
            param.requires_grad = False

        # diffusion-related
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps
        )

        self.diffusion_config = deepcopy(self.classifier.config)
        self.diffusion_config.num_hidden_layers = args.get(
            "diffusion_layer", self.classifier.config.num_hidden_layers
        )
        self.diffusion_config.num_attention_heads = args.get(
            "diffusion_layer", self.classifier.config.num_hidden_layers
        )
        self.diffusion_layer = DiffuseBERT(self.diffusion_config)

    def forward(self, batch):
        text_hiddens = self.classifier.roberta(
            **batch["text"]
        ).last_hidden_state  # (B, S, H)

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

    def classify(self, batch):
        labels = batch["label"]
        text_hiddens = self.classifier.roberta(
            **batch["text"]
        ).last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        # text_pooled_hiddens = self.classifier.roberta.pooler(text_hiddens)  # (B, H)
        # text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_hiddens)
        loss = self.classifier.loss_fn(logits, labels)

        results = {
            "logits": logits,
            "loss": loss,
        }

        return results

    def predict(self, text_inputs):
        text_hiddens = self.classifier.roberta(
            **text_inputs
        ).last_hidden_state  # (B, S, H)
        text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)

        # text_pooled_hiddens = self.classifier.roberta.pooler(text_hiddens)  # (B, H)
        # text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_hiddens)
        outputs = {
            "logits": logits,
        }

        return outputs
