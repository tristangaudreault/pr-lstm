import math
import logging
from typing import Any

import jax.numpy as jnp
import haiku as hk
from einops import rearrange

logger = logging.getLogger(__name__)

from thesis.utils import reshape_with_padding


class Chorus(hk.RNNCore):
    def __init__(
        self,
        hidden_size: int,
        outer_hidden_size: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.inner_hidden_size = hidden_size
        self.outer_hidden_size = outer_hidden_size

        self.inner_cell = hk.GRU(hidden_size=hidden_size)
        self.outer_cell = hk.GRU(hidden_size=outer_hidden_size)

    def initial_state(self, batch_size: int) -> Any:
        return self.inner_cell.initial_state(batch_size), self.outer_cell.initial_state(
            batch_size
        )

    def get_num_branches(self, seq_len: int) -> int:
        return math.ceil(math.sqrt(seq_len))

    @staticmethod
    def branch(
        x: jnp.ndarray, h0: jnp.ndarray, num_branches: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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

        x = reshape_with_padding(x, num_branches=num_branches)
        h0 = jnp.repeat(h0, num_branches, axis=0)

        return x, h0

    @staticmethod
    def process(
        cell: hk.RNNCore,
        x: jnp.ndarray,
        h0: jnp.ndarray,
        time_major: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return hk.dynamic_unroll(cell, x, h0, time_major)  # type: ignore

    def __call__(
        self, x: jnp.ndarray, h0: tuple[jnp.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        x = x.swapaxes(0, 1)
        inner_h0, outer_h0 = h0
        num_branches = self.get_num_branches(x.shape[0])

        branched_x, branched_h0 = self.branch(x, inner_h0, num_branches)
        _, branched_hx = self.process(self.inner_cell, branched_x, branched_h0)

        branched_hx = rearrange(
            branched_hx,
            "(batch num_branches) dim -> batch num_branches dim",
            num_branches=num_branches,
        )
        _, hx = self.process(self.outer_cell, branched_hx, outer_h0, time_major=False)
        return hx
