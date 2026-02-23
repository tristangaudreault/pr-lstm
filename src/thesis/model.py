import logging

import jax
import jax.numpy as jnp
import haiku as hk

from . import utils

logger = logging.getLogger(__name__)


class CrossTemporal(hk.Module):
    def __init__(
        self,
        hidden_size: int,
        time_axis: int = 1,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.time_axis = time_axis

    def __call__(self, xs: jax.Array) -> jax.Array:
        cell = hk.LSTM(self.hidden_size)

        initial_state = cell.initial_state(xs.shape[0])
        inputs = utils.map_with_index(
            lambda i, leaf: hk.Linear(leaf.shape[-1], name="FC" + str(i))(xs),
            initial_state,
        )

        cell = jax.vmap(cell, in_axes=self.time_axis, out_axes=self.time_axis)

        def operator(a, b):
            _, h = cell(jnp.concatenate(jax.tree_util.tree_leaves(a), axis=-1), b)
            return h

        ys = jax.lax.associative_scan(
            operator,
            inputs,
            axis=self.time_axis,
        )

        return jax.tree.leaves(ys)[0]
