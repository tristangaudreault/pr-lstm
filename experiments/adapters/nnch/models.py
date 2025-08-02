from functools import partial
from typing import Any, Callable

import haiku as hk
import jax.numpy as jnp
from neural_networks_chomsky_hierarchy.experiments import constants
from thesis.models import Speculative


def make_model(
    output_size: int,
    inner_core: type[hk.Module],
    return_all_outputs: bool = False,
    input_window: int = 1,
    **model_kwargs: Any,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def model(x: jnp.ndarray, input_length: int = 1) -> jnp.ndarray:
        output = inner_core()(x)
        if not return_all_outputs:
            output = output[:, -1, :]
        output = jnp.reshape(output, (x.shape[0], -1))
        output = jnp.expand_dims(output, axis=1)

        return output

    return model


constants.MODEL_BUILDERS.update(
    {
        "speculative": partial(make_model, inner_core=partial(Speculative, hidden_size=16, K=2, rows=None, cols=1)),  # type: ignore
    }
)
