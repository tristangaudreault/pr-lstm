import torch.nn as nn
from transformers import PreTrainedModel

from ..torch.model import PRLSTM
from .config import PRLMConfig


class PRLM(PreTrainedModel):
    config_class = PRLMConfig

    def __init__(self, config):
        super().__init__(config)

        self.model = PRLSTM(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        logits = self.model(input_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]

            loss_fn = nn.CrossEntropyLoss()

            loss = loss_fn(
                shift_logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
        }
