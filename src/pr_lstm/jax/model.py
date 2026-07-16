import logging
from typing import Callable

import jax
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk

from .layers import NaryLSTM, NaryRNN


logger = logging.getLogger(__name__)


class ParallelRecursive(hk.Module):
    CELL_MAP = {
        "gru": hk.GRU,
        "rnn": NaryRNN,
        "lstm": NaryLSTM,
    }

    def __init__(
        self,
        hidden_size: int,
        cell_name: str,
        batch_first: bool = True,
        num_layers: int = 2,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.cell_class = self.CELL_MAP[cell_name.lower()]
        self.batch_axis = int(not batch_first)
        self.time_axis = int(batch_first)
        assert num_layers >= 1
        assert not (
            num_layers > 1 and self.cell_class is hk.GRU
        ), "GRU cell does not support multi-layer approaches."
        self.num_layers = num_layers

    def __call__(self, xs: jax.Array) -> jax.Array:
        return self.scan(
            self.get_cell(),
            self.get_elems(xs),
            axis=self.time_axis,
        )

    def get_cell(self) -> hk.RNNCore:
        cells = [
            jax.vmap(
                self.cell_class(self.hidden_size),
                in_axes=self.time_axis,
                out_axes=self.time_axis,
            )
            for _ in range(self.num_layers)
        ]

        def cell(l, r):
            # Binary cell
            h, state = cells[0](l, r)

            # Unary cells
            for unary_layer in range(1, self.num_layers):
                h, state = cells[unary_layer](state)

            return state

        return cell

    def get_elems(self, xs):
        if self.cell_class is NaryLSTM:
            gates = hk.Linear(3 * self.hidden_size)(xs)
            i, g, o = jnp.split(gates, indices_or_sections=3, axis=-1)
            c = jnn.sigmoid(i) * jnp.tanh(g)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)

            return hk.LSTMState(h, c)

        return hk.Linear(self.hidden_size)(xs)

    def scan(self, cell, elems, axis):
        states = jax.lax.associative_scan(cell, elems, axis=axis)
        if isinstance(states, hk.LSTMState):
            return states.hidden
        return states
