import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
import haiku as hk
from einops import rearrange

logger = logging.getLogger(__name__)

from thesis.utils import reshape_with_padding, prepend_cls


class Chorus(hk.Module):
    def __init__(
        self,
        transformer: Callable[[], Callable],
        rnn: Callable[[], hk.RNNCore],
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.transformer = transformer
        self.rnn = rnn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.shape[0]
        x = reshape_with_padding(x, cols=4)
        x = prepend_cls(x)
        hx = self.transformer()(x)
        hx = hx[:, 0, :]
        hx = rearrange(
            hx,
            "(batch rows) dim -> batch rows dim",
            batch=batch,
        )
        rnn = self.rnn()
        h0 = rnn.initial_state(batch)
        _, hx = hk.dynamic_unroll(rnn, hx, h0, time_major=False)
        return hx
