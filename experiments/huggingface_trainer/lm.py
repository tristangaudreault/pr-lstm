from transformers import GPT2Config, GPT2LMHeadModel
import torch.nn as nn

from pr_lstm.torch.model import ParallelRecursiveLM


def gpt2(**kwargs):
    config = GPT2Config(
        vocab_size=kwargs["vocab_size"],
        n_positions=512,
    )

    return GPT2LMHeadModel(config)


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, layers=1):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        inputs = self.embed(input_ids)
        logits, _ = self.lstm(inputs)
        logits = self.head(logits)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            loss = loss_fn(
                shift_logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1)
            )
        return {"logits": logits, "loss": loss}


def lstm(**kwargs):
    return LSTMLM(kwargs["vocab_size"], kwargs["hidden_size"])


def pr_lstm(**kwargs):
    return ParallelRecursiveLM(kwargs["vocab_size"], kwargs["hidden_size"])
