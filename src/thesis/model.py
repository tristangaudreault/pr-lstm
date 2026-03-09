from typing import Any
import logging

import jax
import jax.numpy as jnp
import haiku as hk

from . import utils

logger = logging.getLogger(__name__)


class CRNN(hk.Module):
    def __init__(
        self,
        hidden_size: int,
        batch_first: bool = True,
        use_h0: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.batch_axis = int(not batch_first)
        self.time_axis = int(batch_first)
        self.use_h0 = use_h0

    def __call__(self, xs: jax.Array) -> jax.Array:
        cell = self.get_cell()

        batch_size = xs.shape[self.batch_axis]
        h0 = cell.initial_state(batch_size)
        gs = self.intermediate_map(xs, h0)

        ys = self.scan(cell, gs)

        if self.use_h0:
            ys = ys[:, 1:, :]

        return ys

    def get_cell(self, name: str | None = None) -> hk.RNNCore:
        return hk.GRU(self.hidden_size, name=name)

    def intermediate_map(self, xs: jax.Array, h0: Any) -> Any:
        def leaf_map(idx: int, leaf: jax.Array):
            name = "FC_" + str(idx)
            mapped = hk.Linear(leaf.shape[-1], name=name)(xs)

            if self.use_h0:
                mapped = jnp.concatenate(
                    (
                        leaf[:, None, :],
                        mapped,
                    ),
                    axis=self.time_axis,
                )

            return mapped

        return utils.map_with_index(leaf_map, h0)

    def scan(self, cell: hk.RNNCore, gs: Any):
        cell_vmap = jax.vmap(cell, in_axes=self.time_axis, out_axes=self.time_axis)

        def operator(a, b):
            a_concat = jnp.concatenate(jax.tree_util.tree_leaves(a), axis=-1)
            _, h = cell_vmap(a_concat, b)
            return h

        hs = jax.lax.associative_scan(
            operator,
            gs,
            axis=self.time_axis,
        )
        ys = jax.tree.leaves(hs)[0]

        return ys
