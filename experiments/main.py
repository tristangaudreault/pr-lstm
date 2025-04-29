import functools
import sys
from typing import Any, Callable

sys.path.insert(1, "external")
from neural_networks_chomsky_hierarchy.experiments import example, constants
from neural_networks_chomsky_hierarchy.models import rnn, tape_rnn

from absl import app
from absl import logging
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp

import thesis


def make_chorus(
    output_size: int,
    rnn_core: type[hk.RNNCore],
    return_all_outputs: bool = False,
    input_window: int = 1,
    **rnn_kwargs: Any,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Minimally modified implementation of make_rnn. Returns an Chorus model, not haiku transformed.

    Only the last output in the sequence is returned. A linear layer is added to
    match the required output_size.

    Args:
      output_size: The output size of the model.
      rnn_core: The haiku RNN core to use. LSTM by default.
      return_all_outputs: Whether to return the whole sequence of outputs of the
        RNN, or just the last one.
      input_window: The number of tokens that are fed at once to the RNN.
      **rnn_kwargs: Kwargs to be passed to the RNN core.
    """

    def rnn_model(x: jnp.ndarray, input_length: int = 1) -> jnp.ndarray:
        core = rnn_core(**rnn_kwargs)
        if issubclass(rnn_core, tape_rnn.TapeRNNCore):
            initial_state = core.initial_state(
                x.shape[0], input_length
            )  # pytype: disable=wrong-arg-count
        else:
            initial_state = core.initial_state(x.shape[0])

        batch_size, seq_length, embed_size = x.shape
        if seq_length % input_window != 0:
            x = jnp.pad(
                x, ((0, 0), (0, input_window - seq_length % input_window), (0, 0))
            )
        new_seq_length = x.shape[1]
        x = jnp.reshape(
            x, (batch_size, new_seq_length // input_window, input_window, embed_size)
        )

        x = hk.Flatten(preserve_dims=2)(x)

        output = core(x, initial_state)

        if not return_all_outputs:
            output = output[:, -1, :]  # (batch, time, alphabet_dim)
        output = jnn.relu(output)
        output = hk.Linear(output_size)(output)
        return output

    return rnn_model


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    # Monkey patching
    constants.MODEL_BUILDERS["chorus"] = functools.partial(
        make_chorus,
        rnn_core=thesis.model.HaikuModel,
        num_heads=4,
        num_branches=4,
    )

    app.run(example.main)
