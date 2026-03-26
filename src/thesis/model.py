import logging

import jax
import jax.numpy as jnp
import haiku as hk


logger = logging.getLogger(__name__)


class SCM(hk.Module):
    INNER_CELLS = {"gru": hk.GRU, "rnn": hk.VanillaRNN}

    def __init__(
        self,
        hidden_size: int,
        inner_cell: str,
        batch_first: bool = True,
        use_h0: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.inner_cell = self.INNER_CELLS[inner_cell.lower()]
        self.batch_axis = int(not batch_first)
        self.time_axis = int(batch_first)
        self.use_h0 = use_h0

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

        def op(a, b):
            return cell(a, b)[0]

        return op

    def get_elems(self, xs):
        return hk.Linear(self.hidden_size)(xs)

    def scan(self, op, elems, axis):
        return jax.lax.associative_scan(op, elems, axis=axis)


class LSCM(SCM):
    def __init__(
        self,
        hidden_size: int,
        inner_cell: str,
        batch_first: bool = True,
        use_h0: bool = False,
        name: str | None = None,
    ):
        super().__init__(hidden_size, inner_cell, batch_first, use_h0, name)
        self.inner_cell = hk.LSTM

    def get_elems(self, xs):
        hidden = hk.Linear(self.hidden_size)(xs)
        cell = hk.Linear(self.hidden_size)(xs)  # jnp.zeros_like(hidden)
        return hk.LSTMState(hidden, cell)

    def get_op(self):
        # cell = self.get_cell("cell")

        # def op(a: hk.LSTMState, b: hk.LSTMState):
        #     return cell(b.hidden, hk.LSTMState(a.hidden, a.cell + b.cell))[1]

        linear = hk.Linear(5 * self.hidden_size)

        def op(a: hk.LSTMState, b: hk.LSTMState):
            hh = jnp.concatenate([a.hidden, b.hidden], axis=-1)
            gated = linear(hh)
            i, g, f_a, f_b, o = jnp.split(gated, indices_or_sections=5, axis=-1)
            f_a = jax.nn.sigmoid(f_a + 1)  # Forget bias, as in sonnet.
            f_b = jax.nn.sigmoid(f_b + 1)  # Forget bias, as in sonnet.
            c = f_a * a.cell + f_b * b.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            return hk.LSTMState(h, c)

        return op

    def scan(self, op, elems, axis):
        return super().scan(op, elems, axis).hidden


class ASCM(SCM):
    def get_op(self):
        gg = self.get_cell("gg")
        pg = self.get_cell("pg")
        pp = self.get_cell("pp")

        def op(a, b):
            g_a, p_a = a
            g_b, p_b = b
            return gg(g_a, pg(p_a, g_b)[1])[1], pp(p_a, p_b)[1]

        return op

    def get_elems(self, xs):
        gs = hk.Linear(self.hidden_size)(xs)
        ps = hk.Linear(self.hidden_size)(xs)
        return (gs, ps)

    def scan(self, op, elems, axis):
        gs, ps = jax.lax.associative_scan(op, elems, axis=axis)
        return gs
