from functools import partial
from typing import Any, Callable

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
from neural_networks_chomsky_hierarchy.experiments import constants
from neural_networks_chomsky_hierarchy.models import transformer
from thesis import hyperlayers
from thesis.models import Chorus


def make_model(
    output_size: int,
    inner_core: type[hk.Module],
    return_all_outputs: bool = False,
    input_window: int = 1,
    **model_kwargs: Any,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def model(x: jnp.ndarray, input_length: int = 1) -> jnp.ndarray:
        output = inner_core()(x)
        output = jnp.reshape(output, (x.shape[0], -1))
        output = jnn.relu(output)
        output = hk.Linear(output_size)(output)
        output = jnp.expand_dims(output, axis=1)

        return output

    return model


constants.MODEL_BUILDERS["chorus"] = partial(
    make_model,
    inner_core=partial(
        Chorus,
        hyperlayers=(
            # hyperlayers.cls_hyperlayer(
            #     transformer.TransformerEncoder,
            #     config=transformer.TransformerConfig(
            #         output_size=32,
            #         embedding_dim=1,
            #         use_embeddings=False,
            #         num_hiddens_per_head=32,
            #         num_layers=1,
            #         num_heads=1,
            #     ),
            # ),
            hyperlayers.rnn_hyperlayer(hk.GRU, hidden_size=64),
            hyperlayers.rnn_hyperlayer(hk.GRU, hidden_size=64),
            hyperlayers.rnn_hyperlayer(hk.GRU, hidden_size=64),
        ),
        cols=2,
    ),  # type: ignore
)
