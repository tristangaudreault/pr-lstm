import logging
from typing import Callable, Sequence
import math

logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import haiku as hk
from einops import rearrange

from thesis.utils import reshape_with_padding, ceil_division


class Chorus(hk.Module):
    def __init__(
        self,
        hyperlayers: Sequence[Callable[[jnp.ndarray], jnp.ndarray]],
        rows: int | None = None,
        cols: int = 2,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hyperlayers = hyperlayers
        self.rows = rows
        self.cols = cols

    def infer_shape(self, sequence_length: int) -> tuple[int, int]:
        if self.rows is None and self.cols is None:
            side = math.ceil(math.sqrt(sequence_length))
            return side, side

        if self.rows is None:
            return ceil_division(sequence_length, self.cols), self.cols

        if self.cols is None:
            return self.rows, ceil_division(sequence_length, self.rows)

        return self.rows, self.cols

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.shape[0]
        hx = x
        for hyperlayer in self.hyperlayers[:-1]:
            rows, cols = self.infer_shape(sequence_length=x.shape[1])
            hx = reshape_with_padding(hx, rows=rows, cols=cols)
            hx = hyperlayer(hx)
            pattern = "(batch rows) dim -> batch rows dim"
            hx = rearrange(hx, pattern, batch=batch)
        hx = self.hyperlayers[-1](hx)
        return hx
