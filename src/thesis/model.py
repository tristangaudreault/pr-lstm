from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn

import jax
import jax.numpy as jnp
import haiku as hk


class ModelABC(ABC):
    def __init__(self, hidden_size, num_heads, num_branches):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_branches = num_branches


class HaikuModel(ModelABC, hk.RNNCore):
    def __init__(self, hidden_size, num_heads, num_branches, name=None, **_):
        ModelABC.__init__(self, hidden_size, num_heads, num_branches)
        hk.RNNCore.__init__(self, name=name)

        self.rnn_cell_1 = hk.GRU(hidden_size=hidden_size)
        self.rnn_cell_2 = hk.GRU(hidden_size=hidden_size)

    def __call__(self, x, hx):
        batch_size, seq_length, embed_dim = x.shape
        branch_length = math.ceil(seq_length / self.num_branches)

        # Transform the input
        required_len = self.num_branches * branch_length
        if seq_length % required_len != 0:
            x = jnp.pad(
                x, ((0, 0), (0, required_len - seq_length % required_len), (0, 0))
            )
        x = jnp.reshape(
            x,
            (
                batch_size * self.num_branches,
                branch_length,
                embed_dim,
            ),
        )
        x = jnp.transpose(x, (1, 0, 2))
        hx_1, hx_2 = hx

        # Process the input
        for t in range(branch_length):
            x_t = x[t]
            _, hx_1 = self.rnn_cell_1(x_t, hx_1)
        hx_1 = jnp.reshape(hx_1, (batch_size, self.num_branches, self.hidden_size))

        for i in range(self.num_branches):
            x_i = hx_1[:, i, :]
            _, hx_2 = self.rnn_cell_2(x_i, hx_2)

        # Reconstruct the output
        hx_2 = jnp.expand_dims(hx_2, axis=1)

        return hx_2

    def initial_state(self, batch_size):
        return self.rnn_cell_1.initial_state(
            batch_size * self.num_branches
        ), self.rnn_cell_2.initial_state(batch_size)
