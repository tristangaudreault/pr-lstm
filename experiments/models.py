from functools import partial
from typing import Any, Callable

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
from neural_networks_chomsky_hierarchy.experiments import constants
from neural_networks_chomsky_hierarchy.models import rnn
from thesis.models import Speculative
import thesis.models


def make_model(
    output_size: int,
    inner_core: type[hk.Module],
    return_all_outputs: bool = False,
    input_window: int = 1,
    **model_kwargs: Any,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def model(x: jnp.ndarray, input_length: int = 1) -> jnp.ndarray:
        output = inner_core(**model_kwargs)(x)
        if not return_all_outputs:
            output = output[:, -1:, :]

        output = jnn.relu(output)
        output = hk.Linear(output_size)(output)

        return output

    return model


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


class ProjectedRNN(hk.RNNCore):
    def __init__(
        self,
        hidden_size: int,
        projection_size: int = 2,
        name: str | None = None,
        **lstm_kwargs,
    ):
        super().__init__(name=name)

        self.hidden_size = hidden_size
        self.projection_size = projection_size

        self.lstm = hk.LSTM(hidden_size, **lstm_kwargs)
        self.up_proj = hk.Linear(hidden_size)
        self.down_proj = hk.Linear(projection_size)

    def initial_state(self, batch_size: int | None):
        return self.lstm.initial_state(batch_size)

    def __call__(self, inputs, prev_state):
        # LSTM step
        output, next_state = self.lstm(inputs, prev_state)

        # Project output
        projected = self.down_proj(output)

        return projected, next_state


constants.MODEL_BUILDERS.update(
    {
        "speculative": partial(make_model, inner_core=thesis.models.Speculative),  # type: ignore
        "cross_temporal": partial(make_model, inner_core=thesis.models.CrossTemporal),
        "gru": partial(rnn.make_rnn, rnn_core=hk.GRU),
        "linear_rnn": partial(rnn.make_rnn, rnn_core=LinearRNN),
        "projected_rnn": partial(
            rnn.make_rnn,
            rnn_core=ProjectedRNN,
        ),
    }
)
