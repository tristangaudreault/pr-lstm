from abc import ABC, abstractmethod
import math
import logging

import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import haiku as hk

logger = logging.getLogger(__name__)


class ModelABC(ABC):
    def __init__(self, hidden_size, num_heads, num_branches):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_branches = num_branches


class ChorusRNN(ModelABC, hk.RNNCore):
    def __init__(self, hidden_size, num_heads, num_branches, name=None, **_):
        ModelABC.__init__(self, hidden_size, num_heads, num_branches)
        hk.RNNCore.__init__(self, name=name)

        self.rnn_cell_1 = hk.GRU(hidden_size=hidden_size)
        self.rnn_cell_2 = hk.GRU(hidden_size=hidden_size)

    def __call__(self, x, state):
        batch_size, seq_len, embed_dim = x.shape
        branch_len = math.ceil(seq_len / self.num_branches)

        # Transform the input
        required_len = self.num_branches * branch_len
        if seq_len % required_len != 0:
            x = jnp.pad(x, ((0, 0), (0, required_len - seq_len % required_len), (0, 0)))
        x = jnp.reshape(
            x,
            (
                batch_size * self.num_branches,
                branch_len,
                embed_dim,
            ),
        )
        x = jnp.transpose(x, (1, 0, 2))

        if batch_size == 1:
            jax.debug.print("Input: {}", x)

        # Process
        outputs = [[], []]

        hx = state[0]
        for t in range(branch_len):
            x_t = x[t]
            output, hx = self.rnn_cell_1(x_t, hx)
            outputs[0].append(output)

        x = jnp.reshape(hx, (batch_size, self.num_branches, self.hidden_size))
        hx = state[1]
        for i in range(self.num_branches):
            x_i = x[:, i, :]
            output, hx = self.rnn_cell_2(x_i, hx)
            outputs[1].append(output)

        # Reconstruct the output
        hx = jnp.expand_dims(hx, axis=1)

        if batch_size == 1:
            jax.debug.print("Output: {}", outputs)

        return hx

    def initial_state(self, batch_size):
        return jnp.repeat(
            self.rnn_cell_1.initial_state(batch_size), self.num_branches, axis=0
        ), self.rnn_cell_2.initial_state(batch_size)
