import math

import jax
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

    def __call__(self, xs: jax.Array) -> jax.Array:
        seq_len = xs.shape[1]
        speculations = jax.vmap(hk.Linear(self.hidden_size), in_axes=1, out_axes=1)(xs)

        num_refinements = 1  # max(1, math.floor(math.log2(math.log2(seq_len))))
        for _ in range(num_refinements):
            speculations = self.refine(speculations)
            # new_ys, _ = self.single_pass(ys[:, seq_len // 2 :, :])
            # ys = jnp.concatenate((ys[:, : seq_len // 2, :], new_ys), axis=1)

        return speculations

    def refine(self, xs: jax.Array) -> jax.Array:
        batch_size = xs.shape[0]

        refiner = hk.LSTM(self.hidden_size)

        s_0 = refiner.initial_state(batch_size)
        y_1, s_1 = jax.vmap(refiner, in_axes=(1, None), out_axes=1)(xs, s_0)

        refiner = jax.vmap(refiner, in_axes=1, out_axes=1)

        def fn(a, b):
            (inputs, _), (_, prev_state) = a, b

            return refiner(inputs, prev_state)

        ys, hs = jax.lax.associative_scan(fn, (y_1, s_1), axis=1)

        return ys
