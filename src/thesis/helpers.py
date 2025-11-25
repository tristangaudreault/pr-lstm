from typing import Callable, Any, Protocol

import haiku as hk
import jax
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn

from . import regressions

RNNStep = Callable[[Array, Array], tuple[Array, Array]]
ScanStep = Callable[[Array, Array], tuple[Array, tuple[Array, Array]]]


class Scan(Protocol):
    def __call__(
        self, f: ScanStep, init: Array, xs: Array, *args: Any, **kwargs: Any
    ) -> tuple[Array, Array]: ...


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

    def rnn(inputs: Array, prev_state: Array) -> tuple[Array, Array]:
        input_to_hidden = hk.Linear(hidden_size, name="W")
        hidden_to_hidden = hk.Linear(hidden_size, with_bias=False, name="U")
        out = jnn.tanh(input_to_hidden(inputs) + hidden_to_hidden(prev_state))
        return out, out

    return rnn


def speculative_scan(
    f: ScanStep,
    init: Array,
    xs: Array,
    reverse: bool = False,
) -> tuple[Array, Array]:
    """
    Parameters
    ----------
    f : Callable
        A python function to be scanned.
    init: array_like
        Input speculations of shape ``(..., K, B, H)``.
    xs: array_like
        Input sequence of shape ``(T , ..., K, B, H)``.
    reverse:
        Optional bool specifying direction of scan (forward is the default).

    Returns
    -------
    array_like:
        Last carry value of ``f``, with shape ``(..., B, H)``.
    array_like:
        Stacked carry values of ``f``, with shape ``(T, ..., B, H)``.
    """
    _, (_, H) = f(init, xs)
    H, init = map(lambda x: jnp.swapaxes(x, -2, -3), (H, init))

    regressor = regressions.build_regressor(init)
    h = jax.lax.associative_scan(
        regressor,
        H,
        reverse=bool(reverse),
    )
    h = h.take(0, -2)

    return h[-1], h


def unroll(
    rnn_step: RNNStep,
    input_sequence: Array,
    initial_state: Array,
    scan_fn: Scan = speculative_scan,
    time_major: bool = True,
) -> Array:
    """Unrolls an RNN step function.

    Parameters
    ----------
    rnn_step : RNNStep
        RNN step function to unroll.
    input_sequence : array_like
        Input elements tensor of shape ``(T, ...)`` if ``time_major = True`` or ``(B, T, ...)`` otherwise.
    initial_state : array_like
        Initial state tensor given to the rnn_step.
    scan_fn : Scan
        Scan function used to unroll the ``rnn_step``.
    time_major : bool
        Optional bool specifying direction of scan (forward is the default).

    Returns
    -------
    array_like
        Stacked carry values of ``rnn_step``, with shape ``(T, ...)`` if ``time_major = True`` and ``(B, T, ...)`` otherwise.

    Source
    ------
    Adapted from:
    https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py#L146#L217
    """

    def scan_step(
        prev_state: Array, inputs: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        outputs, next_state = rnn_step(inputs, prev_state)
        return next_state, (outputs, next_state)

    if not time_major:
        input_sequence = jnp.swapaxes(input_sequence, 0, -2)

    _, result_sequences = scan_fn(scan_step, initial_state, input_sequence)

    if not time_major:
        result_sequences = jnp.moveaxis(result_sequences, -2, 0)

    return result_sequences


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
    init_model = hk.lift(jax.vmap(model.init, in_axes=(0, None, 0)), name="ensemble")

    def ensemble(inputs: Array, prev_state: Array) -> tuple[Array, Array]:
        params = init_model(init_rng, inputs, prev_state)
        # in_axes: params are mapped, rng/data are not.
        return jax.vmap(model.apply, in_axes=(0, None, None, 0), out_axes=1)(
            params, None, inputs, prev_state
        )

    return ensemble
