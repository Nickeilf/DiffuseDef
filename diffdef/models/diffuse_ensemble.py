from .base import BaseModel, VanillaModel
from .diffuse import DiffuseModel2, DiffuseModel3
from ..utils import get_activation_layer, get_loss_function

from ..utils.io import load_model

import math
import os
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoModel,
    AutoConfig,
    BertLayer,
    AutoModelForSequenceClassification,
)
from diffusers import DDPMScheduler


class DiffuseModel2EnsembleAvg(DiffuseModel2):
    NUM_ENSEMBLE = 10

    def classify(self, batch):
        labels = batch["label"]
        text_hiddens = self.classifier.bert(
            **batch["text"]
        ).last_hidden_state  # (B, S, H)

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

        # text_hiddens = self.denoise(text_hiddens, self.num_test_timesteps)
        text_pooled_hiddens = self.classifier.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_pooled_hiddens)
        loss = self.classifier.loss_fn(logits, labels)

        results = {
            "logits": logits,
            "loss": loss,
        }

        return results

    def base_model_classify(self, text_inputs):
        text_hiddens = self.classifier.bert(
            **text_inputs
        ).last_hidden_state  # (B, S, H)
        text_pooled_hiddens = self.classifier.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_pooled_hiddens)

        return logits

    def predict(self, text_inputs):
        text_hiddens = self.classifier.bert(
            **text_inputs
        ).last_hidden_state  # (B, S, H)
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

        text_pooled_hiddens = self.classifier.bert.pooler(text_hiddens)  # (B, H)
        text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_pooled_hiddens)
        outputs = {
            "logits": logits,
        }

        return outputs


class DiffuseModel3EnsembleAvg(DiffuseModel3):
    # for roberta
    NUM_ENSEMBLE = 10

    def classify(self, batch):
        labels = batch["label"]
        text_hiddens = self.classifier.roberta(
            **batch["text"]
        ).last_hidden_state  # (B, S, H)
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

        # text_pooled_hiddens = self.classifier.bert.pooler(text_hiddens)  # (B, H)
        # text_pooled_hiddens = self.classifier.dropout(text_pooled_hiddens)

        logits = self.classifier.classifier(text_hiddens)
        outputs = {
            "logits": logits,
        }

        return outputs
