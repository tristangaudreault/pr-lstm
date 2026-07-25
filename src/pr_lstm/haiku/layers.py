import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp


class NaryCell(hk.Module):
    def __init__(self, hidden_size: int, name: str | None = None):
        super().__init__(name=name)
        self.hidden_size = hidden_size


class NaryLSTM(NaryCell):
    def __call__(self, *states: hk.LSTMState) -> tuple[jax.Array, hk.LSTMState]:
        hs, cs = zip(*states)
        N = len(states)
        num_gates = 3 + N
        linear = hk.Linear(num_gates * self.hidden_size)
        gates = linear(jnp.concatenate(hs, axis=-1))
        i, g, o, *fs = jnp.split(gates, indices_or_sections=num_gates, axis=-1)
        cell = jax.nn.sigmoid(i) * jnp.tanh(g) + sum(
            jax.nn.sigmoid(f) * c for f, c in zip(fs, cs)
        )
        hidden = jnn.sigmoid(o) * jnn.tanh(cell)

        return hidden, hk.LSTMState(hidden, cell)


class NaryRNN(NaryCell):
    def __call__(self, *states: jax.Array) -> tuple[jax.Array, jax.Array]:
        linear = hk.Linear(self.hidden_size)
        hidden = jnn.relu(linear(jnp.concatenate(states, axis=-1)))

        return hidden, hidden
