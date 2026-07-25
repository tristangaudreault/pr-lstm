import torch
import torch.nn.functional as F
from s5.jax_compat import associative_scan
from torch import nn


class InputStateLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.gates = nn.Linear(input_size, 3 * input_size)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_outputs = self.gates(inputs)
        i, g, o = torch.chunk(gate_outputs, 3, dim=-1)
        c = F.sigmoid(i) * F.tanh(g)
        h = F.sigmoid(o) * F.tanh(c)

        return (c, h)


class PRLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.InputStateLayer = InputStateLayer(hidden_size)
        self.cell = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs = self.embedding(input_ids)

        input_states = self.InputStateLayer(inputs)

        def combine_fn(left, right):
            c1, h1 = left
            _, h2 = right

            c3, h3 = self.cell(
                h2.flatten(0, -2), (c1.flatten(0, -2), h1.flatten(0, -2))
            )

            return c3.reshape(*c1.size()), h3.reshape(*c1.size())

        _, hs = associative_scan(combine_fn, input_states, axis=1)

        logits = self.output(hs)

        return logits
