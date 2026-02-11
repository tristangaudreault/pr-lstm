from typing import Any, Callable, cast
import math
import os
from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk
import logging

from . import utils

logger = logging.getLogger(__name__)


class CrossTemporal(hk.Module):
    def __init__(self, hidden_size: int, name: str | None = None):
        super().__init__(name=name)
        self.hidden_size = hidden_size

    def __call__(self, xs: jax.Array):
        batch_size = xs.shape[0]

        R = hk.LSTM(self.hidden_size)
        i_0 = jnn.tanh(hk.Linear(self.hidden_size)(xs))
        j_0 = R.initial_state(batch_size)
        y_1, j_1 = jax.vmap(R, in_axes=(1, None), out_axes=1)(i_0, j_0)

        R = jax.vmap(R)

        def fn(inputs, prev_state):
            (inputs, _), (_, prev_state) = inputs, prev_state
            y, h = R(inputs, prev_state)

            return y, h

        ys, hs = jax.lax.associative_scan(fn, (y_1, j_1), axis=1)

        return ys


class Speculative(hk.Module):
    """Speculative model.

    Attributes
    ----------
    K : int
        Number of speculations points.
    hidden_size : int
        Hidden size of the inner RNN cells.
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

        F = self.get_core()

        h_0 = F.initial_state(batch_size)
        self.split_state, self.state_size = utils.get_split_array(h_0)

        max_unroll = self.get_max_unroll(seq_len)

        # Residual structure not yet implemented for unrolls larger than 1
        ys, handoff_state = hk.dynamic_unroll(
            F, xs[:, :max_unroll, :], h_0, time_major=False
        )
        ys = cast(Array, ys)
        handoff_elem = (ys[:, -1, :], handoff_state)

        if max_unroll < seq_len:
            ys_ext = self.speculative_scan(F, xs[:, max_unroll:, :], handoff_elem)
            ys = jnp.concatenate((ys, ys_ext), axis=1)

        return ys

    def get_core(self) -> hk.RNNCore:
        return hk.LSTM(self.hidden_size)

    def speculative_scan(
        self, F: hk.RNNCore, xs: Array, init: tuple[Array, Any]
    ) -> Array:
        """Associatively scans the rnn step function F by speculating on previous values

        Parameters
        ----------
        F : hk.RNNCore
            RNN step function
        xs : array_like
            Input array of shape (B, T, E)
        init_state : array_like
            Initial state array of shape (B, H). Collapses speculations

        Returns
        -------
        array_like
            Scanned outputs of shape (B, T, H)
        """
        fn, elems, init_elem = self.build_scan(F, xs, init)
        out_elems = jax.lax.associative_scan(
            fn,
            elems,
            axis=1,
        )

        fn_init = jax.vmap(fn, in_axes=(None, 1), out_axes=1)
        ys, _ = fn_init(init_elem, out_elems)
        ys = ys[..., 0, :]

        # print(ys[0])

        return ys

    def build_scan(
        self, F: Callable, xs: Array, init_elem: Any
    ) -> tuple[Callable, tuple[Array, Array], tuple[Array, Array]]:
        S = hk.get_parameter(
            "S",
            (self.K, self.state_size),
            init=hk.initializers.RandomUniform(minval=-1, maxval=1),
        )

        def fn(a, b):
            a_y, a_h = a
            b_y, b_h = b

            # a_h = hypermap(a_h)

            w = utils.soft_quantize(a_h, S)
            # d = utils.soft_quantize_dists(a_h, S)
            # w = jnn.softmax(
            #     -(d + hypermap(d)) / 1e-1
            # )
            # utils.soft_quantize((S + a_h) if residual else a_h, S, temperature=1e-1)
            return (w @ b_y, w @ b_h)

        k_ys, k_hs = utils.recursive_vmap(F, ((None, 0), (0, None), (0, None)))(
            xs, self.split_state(S)
        )
        k_hs = utils.concat_tree(k_hs)

        init_y, init_h = init_elem
        init_h = utils.concat_tree(init_h)
        init_y, init_h = init_y[:, None, :], init_h[:, None, :]
        return fn, (k_ys, k_hs), (init_y, init_h)

    def get_max_unroll(self, seq_len: int) -> int:
        if (N := os.environ.get("MAX_UNROLL")) is not None:
            max_unroll = int(N)
        elif (N := os.environ.get("MAX_SCAN")) is not None:
            max_unroll = seq_len - int(N)
        else:
            max_unroll = math.ceil(math.log2(seq_len))

        max_unroll = min(seq_len, max(1, max_unroll))

        if hk.running_init():
            logger.debug(
                "%d / %d (unrolled / scanned)", max_unroll, seq_len - max_unroll
            )

        return max_unroll
