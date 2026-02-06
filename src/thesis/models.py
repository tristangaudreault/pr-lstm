from typing import Any, Callable
import math
import os

import jax
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk
import logging

from .utils import concat_tree, get_split_array, print_grad, recursive_vmap


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
        K: int,
        hidden_size: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.K = K
        self.hidden_size = hidden_size

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
        batch_size, seq_len, embed_dim = xs.shape
        split_idx = self.get_split_idx(seq_len)
        if hk.running_init():
            logger.debug("%d / %d (unrolled / scanned)", split_idx, seq_len - split_idx)

        h0, F = self.get_core(batch_size)

        _, unrolled = self.unroll(F, xs[:, :split_idx, :], h0)
        if seq_len > split_idx:
            hs = self.speculative_scan(F, xs[:, split_idx:, :], unrolled)
        else:
            hs = unrolled

        return self.out_filter(hs)

    def get_split_idx(self, seq_len: int):
        flag_map = {
            "UNROLL_FIRST": 1,
            "UNROLL_ALL": seq_len,
            "SCAN_LAST": seq_len - 1,
        }
        for flag, value in flag_map.items():
            if int(os.environ.get(flag, 0)):
                return min(seq_len, max(1, value))
        return math.ceil(math.log2(seq_len))

    def get_core(self, batch_size: int) -> tuple[Any, hk.RNNCore]:
        F = hk.LSTM(self.hidden_size)
        h0 = F.initial_state(batch_size)

        self.split_array, self.total_size = get_split_array(h0)
        self.out_filter = lambda h: h.hidden

        return h0, F

    def unroll(self, F, xs, h0):
        return hk.dynamic_unroll(F, xs, h0, time_major=False, return_all_states=True)

    def get_quantizer(self) -> tuple[Any, Callable[[Array], Array]]:
        hk_Q = hk.nets.VectorQuantizer(self.total_size, self.K, 0.25)

        S = hk_Q.embeddings.T
        S = self.split_array(S)

        Q = lambda v: hk_Q(v, is_training=True)

        return S, Q

    def speculative_scan(self, F: hk.RNNCore, xs: Array, unrolled: Array) -> Array:
        '''Associatively scans the rnn step function F by speculating on previous values

        Parameters
        ----------
        F : hk.RNNCore
            RNN step function
        xs : array_like
            Input array of shape [B, T_2, E]
        unrolled : array_like
            [B, T_1, H]

        Returns
        -------
        array_like
            [B, T_1 + T_2, H]
        '''
        S, Q = self.get_quantizer()

        S = jax.lax.stop_gradient(S)
        _, steps = recursive_vmap(F, ((None, 0), (0, None), (0, None)))(xs, S)

        # Q injected in composition function
        def compose(a, b):
            q = Q(a)
            indices = q["encoding_indices"][..., None]
            composed = jnp.take_along_axis(b, indices, axis=-2)

            return composed + a - jax.lax.stop_gradient(a)

        steps = concat_tree(steps)
        scanned = jax.lax.associative_scan(
            compose,
            steps,
            axis=1,
        )

        # Applying trailing Q
        compose = jax.vmap(compose, in_axes=(None, 1), out_axes=1)
        unrolled = concat_tree(unrolled)
        hs = compose(unrolled[:, -1:, :], scanned)[..., 0, :]
        hs = jnp.concatenate((unrolled, hs), axis=1)
        hs = self.split_array(hs)

        return hs
