from typing import Callable

import jax
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk


RNNStep = Callable[[Array, Array], tuple[Array, Array]]


def make_rnn(hidden_size: int) -> RNNStep:
    """Builds a haiku RNN step function.

    Parameters
    ----------
    hidden_size : int
        Hidden size of the RNN model.

    Returns
    -------
    RNNStep
        Vanilla RNN step function.
    """

    def rnn(prev_state: Array, inputs: Array) -> tuple[Array, Array]:
        input_to_hidden = hk.Linear(hidden_size, with_bias=True, name="U")
        hidden_to_hidden = hk.Linear(hidden_size, with_bias=True, name="W")
        out = jnn.tanh(input_to_hidden(inputs) + hidden_to_hidden(prev_state))
        return out, out

    return rnn


def create_ensemble(f: RNNStep, ensemble_size: int) -> RNNStep:
    """Creates a model ensemble.

    Parameters
    ----------
    f : RNNStep
        Template RNN step function of the ensemble.
    ensemble_size : int
        Number of independant models to construct as part of the ensemble.

    Returns
    -------
    RNNStep
        Ensemble of ``ensemble_size`` RNN steps.

    Source
    ------
    Adapted from:
    "https://github.com/google-deepmind/dm-haiku/issues/533"
    """
    init_rng = hk.next_rng_keys(ensemble_size) if hk.running_init() else None
    model = hk.transform(f)
    # in_axes: rng is mapped, data is not.
    init_model = hk.lift(jax.vmap(model.init, in_axes=(0, 2, None)), name="ensemble")

    def ensemble(prev_state: Array, inputs: Array) -> tuple[Array, Array]:
        params = init_model(init_rng, prev_state, inputs)
        # in_axes: params are mapped, rng/data are not.
        return jax.vmap(model.apply, in_axes=(0, None, 2, None), out_axes=2)(
            params, None, prev_state, inputs
        )

    return ensemble
