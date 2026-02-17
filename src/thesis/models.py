import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk
import logging

from . import utils

logger = logging.getLogger(__name__)
z = tuple[jax.Array, Any]


class CrossTemporal(hk.Module):
    def __init__(
        self,
        hidden_size: int,
        connection: Callable[[z, z], z] = lambda a, b: (a[0], b[1]),
        iterations: Callable[[int], int] = lambda seq_len: 1,
        time_axis: int = 1,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.connection = connection
        self.iterations = iterations
        self.time_axis = time_axis

    def vmap_time(self, fn):
        return jax.vmap(fn, in_axes=self.time_axis, out_axes=self.time_axis)

    def __call__(self, xs: jax.Array) -> jax.Array:
        speculations = self.speculate(xs)

        iterations = self.iterations(xs.shape[self.time_axis])
        for n in range(iterations):
            speculations = self.iterate(speculations)

        ys, hs = speculations

        return ys

    def speculate(self, xs: jax.Array) -> z:
        """
        Args:
            xs (array_like): Input array.
        """
        batch_size = xs.shape[0]

        speculator = hk.LSTM(self.hidden_size, name="F_theta")
        initial_state = speculator.initial_state(batch_size)
        speculator = jax.vmap(
            speculator, in_axes=(self.time_axis, None), out_axes=self.time_axis
        )

        return speculator(xs, initial_state)

    def iterate(self, speculations: z) -> z:
        operator = hk.LSTM(self.hidden_size, name="F_phi")
        operator = self.vmap_time(operator)

        return jax.lax.associative_scan(
            lambda a, b: operator(*self.connection(a, b)),
            speculations,
            axis=self.time_axis,
        )
