import torch
import torch.nn as nn
import torch.nn.functional as F
from s5.jax_compat import associative_scan


class InputStateEmbedding(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.gates = nn.Linear(input_size, 3 * input_size)

    def forward(self, inputs):
        gate_outputs = self.gates(inputs)
        i, g, o = torch.chunk(gate_outputs, 3, dim=-1)
        c = F.sigmoid(i) * F.tanh(g)
        h = F.sigmoid(o) * F.tanh(c)

        return (c, h)


class ParallelRecursiveLM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.ISE = InputStateEmbedding(hidden_size)
        self.cell = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        batch_size, seq_len = input_ids.size()

        inputs = self.embedding(input_ids)

        input_states = self.ISE(inputs)

        def combine_fn(left, right):
            c1, h1 = left
            c2, h2 = right

            c3, h3 = self.cell(
                h2.flatten(0, -2), (c1.flatten(0, -2), h1.flatten(0, -2))
            )

            return c3.reshape(*c1.size()), h3.reshape(*c1.size())

        cs, hs = associative_scan(combine_fn, input_states, axis=1)

        logits = self.output(hs)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            loss = loss_fn(shift_logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))

        return {"logits": logits, "loss": loss}
