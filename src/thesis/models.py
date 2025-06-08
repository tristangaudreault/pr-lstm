import math
import logging
from typing import Any

import jax.numpy as jnp
import haiku as hk

logger = logging.getLogger(__name__)

from thesis.utils import reshape_with_padding


class Chorus(hk.RNNCore):
    def __init__(
        self,
        hidden_size: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size

        self.rnn_cell = hk.GRU(hidden_size=hidden_size)

    def initial_state(self, batch_size: int) -> Any:
        return self.rnn_cell.initial_state(batch_size)

    def get_num_branches(self, seq_len: int) -> int:
        return math.ceil(math.sqrt(seq_len))

    def expand_initial_state(self, h0: Any, num_branches: int) -> Any:
        return jnp.repeat(h0, num_branches, axis=0)

    def process(
        self, x: jnp.ndarray, h0: jnp.ndarray, num_branches: int
    ) -> jnp.ndarray:
        """
        Applies branched processing to an input 3D tensor of shape (seq, batch, dim) and returns the final hidden state of shape (batch * num_branches, hidden_dim).

        Args:
            x (jnp.ndarray): Input of shape (seq, batch, dim).
            h0 (jnp.ndarray): Initial hidden state of shape (batch, hidden_dim).
            num_branches: Number of branches to split the sequence into.

        Returns:
            jnp.ndarray: Final hidden state of shape (batch * num_branches, hidden_dim).
        """
        assert x.ndim == 3, "Expected x of shape (seq, batch, dim)"
        assert h0.ndim == 2, "Expected h0 of shape (batch, hidden_dim)"

        x = reshape_with_padding(x, num_branches)
        seq_rows, seq_cols, batch, dim = x.shape

        x = x.transpose(1, 2, 0, 3)
        x = x.reshape(seq_cols, batch * seq_rows, dim)

        _, hx = hk.dynamic_unroll(self.rnn_cell, x, h0)
        return hx

    def __call__(self, x: jnp.ndarray, h0: jnp.ndarray) -> jnp.ndarray:
        x = x.swapaxes(0, 1)
        num_branches = self.get_num_branches(x.shape[0])
        h0 = self.expand_initial_state(h0, num_branches)
        return self.process(x, h0, num_branches)


class ChorusRNN(Chorus):
    def __init__(
        self,
        hidden_size: int,
        outer_hidden_size: int,
        name: str | None = None,
    ):
        super().__init__(hidden_size, name=name)
        self.outer_rnn_cell = hk.GRU(hidden_size=outer_hidden_size)

    def initial_state(self, batch_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        return super().initial_state(batch_size), self.outer_rnn_cell.initial_state(
            batch_size
        )

    def expand_initial_state(self, h0: Any, num_branches: int):
        inner_h0, outer_h0 = h0
        return super().expand_initial_state(inner_h0, num_branches), outer_h0

    def process(
        self, x: jnp.ndarray, h0: jnp.ndarray, num_branches: int
    ) -> jnp.ndarray:
        _, batch, _ = x.shape
        inner_h0, outer_h0 = h0
        inner_hx = super().process(x, inner_h0, num_branches)
        inner_hx = jnp.reshape(inner_hx, (batch, num_branches, self.hidden_size))
        _, outer_hx = hk.dynamic_unroll(
            self.outer_rnn_cell, inner_hx, outer_h0, time_major=False
        )
        return outer_hx
