from functools import partial
from typing import Any, Callable

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
from neural_networks_chomsky_hierarchy.experiments import constants
from neural_networks_chomsky_hierarchy.models import rnn
from thesis.models import Speculative


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
        output = hk.Linear(output_size, name="output")(output)

        return output

    return model


constants.MODEL_BUILDERS.update(
    {
        "speculative": partial(make_model, inner_core=Speculative),  # type: ignore
        "gru": partial(rnn.make_rnn, rnn_core=hk.GRU),
    }
)
