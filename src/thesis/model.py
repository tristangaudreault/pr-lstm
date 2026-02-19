import logging

import jax
import jax.numpy as jnp
import haiku as hk


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
        cell_vmap_inputs = jax.vmap(
            cell, in_axes=(self.time_axis, None), out_axes=self.time_axis
        )
        cell_vmap = jax.vmap(cell, in_axes=self.time_axis, out_axes=self.time_axis)

        initial_state = cell.initial_state(xs.shape[0])
        initial_inputs = hk.Linear(
            self.hidden_size * (2 if isinstance(cell, hk.LSTM) else 1)
        )(xs)
        _, speculations = cell_vmap_inputs(initial_inputs, initial_state)

        def operator(a, b):
            _, h = cell_vmap(jnp.concatenate(jax.tree_util.tree_leaves(a), axis=-1), b)
            return h

        ys = jax.lax.associative_scan(
            operator,
            speculations,
            axis=self.time_axis,
        )

        if isinstance(cell, hk.LSTM):
            ys, _ = ys

        return ys
