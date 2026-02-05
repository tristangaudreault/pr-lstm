from functools import partial, wraps
from itertools import accumulate
from typing import Any, Callable, Protocol
from operator import sub

import jax
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk
import logging

from .helpers import RNNCell
from .utils import concat_tree, get_split_array, print_grad


logger = logging.getLogger(__name__)


def normalize(arr, a, b, axis: int):
    # Calculate min/max along the last axis (axis=-1)
    # keepdims=True ensures the result is (shape..., 1) instead of (shape...)
    arr_min = jnp.min(arr, axis=axis, keepdims=True)
    arr_max = jnp.max(arr, axis=axis, keepdims=True)

    # Standard formula, now applied per-vector
    denom = arr_max - arr_min

    # Optional: Handle cases where all elements in a row are the same
    # denom[denom == 0] = 1.0

    return a + (arr - arr_min) * (b - a) / denom


class Speculative(hk.Module):
    """Speculative model.

    Attributes
    ----------
    hidden_size : int
        Hidden size of the inner RNN cells.
    K : int
        Number of speculations to perform.
    """

    def __init__(
        self,
        hidden_size: int,
        K: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.K = K

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Applies the speculative model.

        Parameters
        ----------
        xs : array_like
            Input sequence tensor of shape ``(B, T, I)``.

        Returns
        -------
        array_like
            Output sequence tensor of shape (B, T, H).
        """
        batch_size = xs.shape[0]

        h0, F, out_filter = self.get_cell(batch_size)
        S, Q = self.get_quantizer()

        _, phi_step = self.restrict_to_S(F, xs, jax.lax.stop_gradient(S))

        # Implementation detail: Q is injected in the composition function
        def compose(a, b):
            quantization = Q(a, is_training=True)
            indices = quantization["encoding_indices"][..., None]
            composed = jnp.take_along_axis(b, indices, axis=-2)

            return (
                a + composed
            )  # + quantization["loss"] - jax.lax.stop_gradient(quantization["loss"])

        hs = self.scan_phi_step(compose, phi_step, h0)

        return out_filter(hs)

    def get_cell(self, batch_size: int) -> tuple[Any, Callable, Callable]:
        F = hk.LSTM(self.hidden_size)
        h0 = F.initial_state(batch_size)

        # Implementation detail: tree utilities
        self.split_array, self.total_size = get_split_array(h0)

        return h0, F, lambda h: h.hidden

    def get_quantizer(self) -> tuple[Any, Callable]:
        Q: hk.nets.VectorQuantizer = hk.nets.VectorQuantizer(
            self.total_size, self.K, 0.25
        )
        S = Q.embeddings.T
        S = self.split_array(S)

        return S, Q

    def restrict_to_S(self, F: Callable, xs: Array, S: Any) -> Any:
        F = jax.vmap(F, in_axes=(None, 0))
        F = jax.vmap(F, in_axes=(0, None))
        F = jax.vmap(F, in_axes=(0, None))

        return F(xs, S)

    def scan_phi_step(self, compose: Callable, phi_step: Array, h0: Array) -> Array:
        # Implementation detail: the quantizer expects concatenated trees
        phi_step = concat_tree(phi_step)
        h0 = concat_tree(h0)

        phi_one = jax.lax.associative_scan(
            compose,
            phi_step,
            axis=1,
        )

        # Implementation detail: the trailing Q is applied here
        compose = jax.vmap(compose, in_axes=(None, 1), out_axes=1)
        hs = compose(h0[:, None, :], phi_one)[..., 0, :]

        # Implementation detail: reconstructing trees
        hs = self.split_array(hs)

        return hs
