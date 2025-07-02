import logging
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

import jax.numpy as jnp
import haiku as hk
from einops import rearrange

from thesis.utils import reshape_with_padding


class Chorus(hk.Module):
    def __init__(
        self,
        hyperlayers: Sequence[Callable[[jnp.ndarray], jnp.ndarray]],
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hyperlayers = hyperlayers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.shape[0]
        hx = x
        for hyperlayer in self.hyperlayers[:-1]:
            hx = reshape_with_padding(hx, cols=4)
            logger.debug(hx.shape)
            hx = hyperlayer(hx)
            logger.debug(hx.shape)
            hx = rearrange(
                hx,
                "(batch rows) dim -> batch rows dim",
                batch=batch,
            )
            logger.debug(hx.shape)
        hx = hyperlayer(hx)
        logger.debug(hx.shape)
        return hx
