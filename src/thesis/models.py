import math
import logging

import jax.numpy as jnp
import haiku as hk

from neural_networks_chomsky_hierarchy.models import transformer  # type: ignore

logger = logging.getLogger(__name__)


class Chorus(hk.RNNCore):
    def __init__(
        self, hidden_size: int, max_branches: int | None = None, name: str | None = None
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.max_branches = max_branches

        self.rnn_cell = hk.GRU(hidden_size=hidden_size)

    def initial_state(self, batch_size: int):
        return self.rnn_cell.initial_state(batch_size)

    def get_num_branches(self, seq_len: int):
        num_branches = math.ceil(math.sqrt(seq_len))
        if self.max_branches:
            num_branches = min(num_branches, self.max_branches)

        return num_branches

    @staticmethod
    def adapt_input(x: jnp.ndarray, num_branches: int) -> jnp.ndarray:
        """Adapt the input for use by the Chorus model. Returns an array of shape (batch size, number of branches, branch Length, embedding dimension)"""
        batch_size, seq_len, embed_dim = x.shape
        branch_len = -(-seq_len // num_branches)

        required_len = num_branches * branch_len
        pad_len = required_len - seq_len
        x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0))) if pad_len > 0 else x

        x = x.reshape(batch_size, num_branches, branch_len, embed_dim)
        return x.transpose(2, 0, 1, 3).reshape(
            branch_len, batch_size * num_branches, embed_dim
        )

    def process(
        self, x: jnp.ndarray, h0: jnp.ndarray, num_branches: int
    ) -> jnp.ndarray:
        h0 = jnp.repeat(h0, num_branches, axis=0)
        return hk.dynamic_unroll(self.rnn_cell, x, h0)[1]

    def __call__(self, x: jnp.ndarray, h0: jnp.ndarray) -> jnp.ndarray:
        num_branches = self.get_num_branches(x.shape[1])
        return self.process(
            self.adapt_input(x, num_branches),
            h0,
            num_branches,
        )


class ChorusRNN(Chorus):
    def __init__(
        self,
        hidden_size: int,
        outer_hidden_size: int,
        max_branches: int | None = None,
        name: str | None = None,
    ):
        super().__init__(hidden_size, max_branches, name=name)
        self.composer = hk.GRU(hidden_size=outer_hidden_size)

    def initial_state(self, batch_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return super().initial_state(batch_size), self.composer.initial_state(
            batch_size
        )

    def process(
        self, x: jnp.ndarray, h0: jnp.ndarray, num_branches: int
    ) -> jnp.ndarray:
        branch_hxs = super().process(x, h0[0], num_branches)
        branch_hxs = jnp.reshape(branch_hxs, (-1, num_branches, self.hidden_size))
        return hk.dynamic_unroll(self.composer, branch_hxs, h0[1], time_major=False)[1]
