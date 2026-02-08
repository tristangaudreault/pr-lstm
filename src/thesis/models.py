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

        h0, F = self.get_core(batch_size)

        _, hs = hk.dynamic_unroll(
            F, xs[:, :split_idx, :], h0, time_major=False, return_all_states=True
        )

        if seq_len > split_idx:
            hs = self.speculative_scan(F, xs[:, split_idx:, :], hs)

        return self.out_filter(hs)

    def get_core(self, batch_size: int) -> tuple[Any, hk.RNNCore]:
        F = hk.LSTM(self.hidden_size)
        h0 = F.initial_state(batch_size)

        self.split_array, self.total_size = get_split_array(h0)
        self.out_filter = lambda h: h.hidden

        return h0, F

    def make_scannable(
        self, F: Callable, xs: Array
    ) -> tuple[Callable[[Array, Array], Array], Array]:
        hk_Q = hk.nets.VectorQuantizer(self.total_size, self.K, 0.25)

        S = hk_Q.embeddings.T
        S = self.split_array(S)

        Q = lambda v: hk_Q(v, is_training=True)

        S = jax.lax.stop_gradient(S)

        _, steps = recursive_vmap(F, ((None, 0), (0, None), (0, None)))(xs, S)
        steps = concat_tree(steps)

        # Q injected in composition function
        def compose(a, b):
            q = Q(a)
            indices = q["encoding_indices"][..., None]
            composed = jnp.take_along_axis(b, indices, axis=-2)

            return composed + a - jax.lax.stop_gradient(a)

        return compose, steps

    def speculative_scan(self, F: hk.RNNCore, xs: Array, unrolled: Array) -> Array:
        """Associatively scans the rnn step function F by speculating on previous values

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
        """
        unrolled = concat_tree(unrolled)

        fn, elems = self.make_scannable(F, xs)
        scanned = jax.lax.associative_scan(
            fn,
            elems,
            axis=1,
        )

        # Applying trailing Q
        compose = jax.vmap(fn, in_axes=(None, 1), out_axes=1)
        hs = compose(unrolled[:, -1:, :], scanned)[..., 0, :]
        hs = jnp.concatenate((unrolled, hs), axis=1)

        hs = self.split_array(hs)

        return hs

    def get_split_idx(self, seq_len: int) -> int:
        if int(os.environ.get("UNROLL_FIRST", 0)):
            split_idx = 1
        elif int(os.environ.get("UNROLL_ALL", 0)):
            split_idx = seq_len
        elif int(os.environ.get("SCAN_LAST", 0)):
            split_idx = seq_len - 1
        else:
            split_idx = math.ceil(math.log2(seq_len))

        split_idx = min(seq_len, max(1, split_idx))

        if hk.running_init():
            logger.debug("%d / %d (unrolled / scanned)", split_idx, seq_len - split_idx)

        return split_idx
