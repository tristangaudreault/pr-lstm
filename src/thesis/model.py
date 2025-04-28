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

        self.rnn_cell = hk.GRU(hidden_size=hidden_size)

        self.multihead_attn = hk.MultiHeadAttention(
            num_heads=num_heads,
            key_size=hidden_size // num_heads,
            w_init=hk.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )

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

        # Process the input
        outputs = []
        for t in range(branch_length):
            x_t = x[t]
            _, hx = self.rnn_cell(x_t, hx)
            outputs.append(hx)

            hx = jnp.reshape(hx, (batch_size, self.num_branches, self.hidden_size))
            hx = self.multihead_attn(hx, hx, hx)
            hx = jnp.reshape(hx, (batch_size * self.num_branches, self.hidden_size))

        # Reconstruct the output
        outputs = jnp.stack(outputs, axis=2)
        outputs = jnp.reshape(
            outputs, (batch_size, self.num_branches * branch_length, self.hidden_size)
        )
        outputs = outputs[:, :seq_length, :]

        return outputs, hx

    def initial_state(self, batch_size):
        return self.rnn_cell.initial_state(batch_size * self.num_branches)
