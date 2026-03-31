import logging
from typing import Callable

import jax
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk


logger = logging.getLogger(__name__)


class SCM(hk.Module):
    INNER_CELLS = {"gru": hk.GRU, "rnn": hk.VanillaRNN}

    def __init__(
        self,
        hidden_size: int,
        inner_cell: str,
        substeps: int = 0,
        batch_first: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.inner_cell = self.INNER_CELLS[inner_cell.lower()]
        self.substeps = substeps
        self.batch_axis = int(not batch_first)
        self.time_axis = int(batch_first)

    def __call__(self, xs: jax.Array) -> jax.Array:
        return self.scan(
            self.get_op(),
            self.get_elems(xs),
            axis=self.time_axis,
        )

    def get_cell(self, name: str | None = None) -> hk.RNNCore:
        cell = self.inner_cell(self.hidden_size, name=name)
        cell_vmap = jax.vmap(cell, in_axes=self.time_axis, out_axes=self.time_axis)
        return cell_vmap

    def get_op(self):
        cell = self.get_cell("cell")

        def op(l, r):
            return cell(r, l)[1]

        return op

    def get_elems(self, xs):
        return hk.Linear(self.hidden_size)(xs)

    def scan(self, op, elems, axis):
        return jax.lax.associative_scan(op, elems, axis=axis)


class LSCM(SCM):
    def get_elems(self, xs):
        gates = hk.Linear(3 * self.hidden_size)(xs)
        i, g, o = jnp.split(gates, indices_or_sections=3, axis=-1)
        c = jnn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)

        return hk.LSTMState(h, c)

    def get_cell(
        self, name: str | None = None
    ) -> Callable[[hk.LSTMState, hk.LSTMState], tuple[jax.Array, hk.LSTMState]]:
        linear = hk.Linear(5 * self.hidden_size)
        linear_2 = hk.Linear(4 * self.hidden_size)

        def cell(l: hk.LSTMState, r: hk.LSTMState) -> tuple[jax.Array, hk.LSTMState]:
            gates = linear(jnp.concatenate([l.hidden, r.hidden], axis=-1))
            i, g, f_l, f_r, o = jnp.split(gates, indices_or_sections=5, axis=-1)
            f_l = jax.nn.sigmoid(f_l)  # Forget bias, as in sonnet.
            f_r = jax.nn.sigmoid(f_r)  # Forget bias, as in sonnet.
            c = f_l * l.cell + f_r * r.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            new_state = hk.LSTMState(h, c)

            for substep in range(self.substeps):
                gates = linear_2(new_state.hidden)
                i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)
                f = jax.nn.sigmoid(f)
                c = f * new_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
                h = jax.nn.sigmoid(o) * jnp.tanh(c)
                new_state = hk.LSTMState(h, c)

            return h, new_state

        return cell

    def scan(self, op, elems, axis):
        return super().scan(op, elems, axis).hidden
