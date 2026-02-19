from typing import Any

import haiku as hk
import jax.numpy as jnp


class LinearRNN(hk.RNNCore):
    def __init__(self, hidden_size: int, name: str | None = None):
        super().__init__(name)
        self.hidden_size = hidden_size

    def initial_state(self, batch_size: int | None):
        return jnp.zeros([batch_size, self.hidden_size])

    def __call__(self, inputs, prev_state) -> tuple[Any, Any]:
        next_state = hk.Linear(self.hidden_size)(inputs) + hk.Linear(self.hidden_size)(
            prev_state
        )
        return next_state, next_state
