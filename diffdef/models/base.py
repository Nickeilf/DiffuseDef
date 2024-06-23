from ..utils import get_activation_layer, get_loss_function

from torch import nn
from transformers import AutoModel, AutoConfig


class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def _get_learnable_params(self):
        """Returns the list of parameters having `requires_grad` enabled."""
        return [(n, p) for (n, p) in self.named_parameters() if p.requires_grad]

    def _stop_gradients(self, module: nn.Module) -> None:
        """Disables gradients for the given `nn.Module` instance.

        Args:
            module: `nn.Module` instance against which the modification
                will be applied.
        """
        for name, param in module.named_parameters():
            param.requires_grad = False

    def edit_layers(self):
        # Earlier layer features requested, chop unused layers
        if self.extract_from != -1:
            # Get TF layers
            tf_layers = self._get_tf_layers()
            new_layers = nn.ModuleList([tf_layers[i] for i in range(self.extract_from)])
            self._set_tf_layers(new_layers)

        # Do nothing if == 'all'
        if isinstance(self.finetune_last_bert_layers, int):
            if self.finetune_last_bert_layers > 0:
                for layer in self._get_tf_layers()[: -self.finetune_last_bert_layers]:
                    self._stop_gradients(layer)
            elif self.finetune_last_bert_layers == 0:
                self._stop_gradients(self.model.base_model)

    def _get_tf_layers(self):
        """Returns the list of backend TF layers as the way of accessing them
        differs across HuggingFace models."""
        if hasattr(self.model, "transformer"):
            return self.model.transformer.layer
        elif hasattr(self.model, "encoder"):
            return self.model.encoder.layer
        elif hasattr(self.model, "layer"):
            return self.model.layer
        elif hasattr(self.model, "text_model"):
            return self.model.text_model.encoder.layers
        else:
            raise RuntimeError("This HuggingFace model is not supported.")

    def _set_tf_layers(self, layers: nn.ModuleList) -> None:
        if hasattr(self.model, "transformer"):
            self.model.transformer.layer = layers
        elif hasattr(self.model, "encoder"):
            self.model.encoder.layer = layers
        elif hasattr(self.model, "text_model"):
            self.model.text_model.encoder.layers = layers
        else:
            raise RuntimeError("This HuggingFace model is not supported.")

    def forward(self, batch):
        raise NotImplementedError(
            "You should implement forward() in the derived class!"
        )


class VanillaModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.hf_model_name = args.pretrained_model
        self.dropout = args.dropout
        self.activ_func = args.activ_func
        self.output_dim = args.output_dim
        self.loss_fn = get_loss_function(args.loss_fn)

        # loading PLM from huggingface
        self.config = AutoConfig.from_pretrained(
            self.hf_model_name,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
            output_hidden_states=True,
        )
        self.backbone_model = AutoModel.from_pretrained(
            self.hf_model_name, config=self.config
        )
        # self.edit_layers()

        self.proj = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size, eps=1e-12),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            get_activation_layer(self.activ_func)(),
            nn.Dropout(self.dropout),
        )

        self.output = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            get_activation_layer(self.activ_func)(),
            nn.Linear(self.config.hidden_size // 2, self.output_dim),
        )

    def forward(self, batch):
        text_inputs = batch["text"]
        labels = batch["label"]

        text_hiddens = self.backbone_model(**text_inputs).last_hidden_state  # (B, S, H)
        text_pooled_hiddens = text_hiddens.mean(dim=1)  # (B, H)
        text_pooled_hiddens = self.proj(text_pooled_hiddens)

        logits = self.output(text_pooled_hiddens)
        loss = self.loss_fn(logits, labels)

        results = {
            "logits": logits,
            "loss": loss,
        }
        return results

    def encode(self, batch):
        text_inputs = batch["text"]

        text_hiddens = self.backbone_model(**text_inputs).last_hidden_state  # (B, S, H)

        return text_hiddens

    def classify(self, text_hiddens):
        text_pooled_hiddens = text_hiddens.mean(dim=1)  # (B, H)
        text_pooled_hiddens = self.proj(text_pooled_hiddens)

        logits = self.output(text_pooled_hiddens)

        results = {
            "logits": logits,
        }
        return results

    def predict(self, text_inputs):
        text_hiddens = self.backbone_model(**text_inputs).pooler_output  # (B, H)
        text_hiddens = self.proj(text_hiddens)

        logits = self.output(text_hiddens)

        results = {
            "logits": logits,
        }
        return results
