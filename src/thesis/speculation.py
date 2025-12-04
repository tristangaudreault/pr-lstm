from typing import Sequence, cast
from functools import reduce
from operator import mul

from jax import Array
import jax.numpy as jnp
from scipy.stats.qmc import Sobol


def sobol(shape: Sequence[int], dtype: jnp.dtype) -> Array:
    *batch_ds, d = shape
    num = reduce(mul, batch_ds, 1)

    sampler = Sobol(shape[-1])
    X = sampler.random(shape[-2])
    X = 2 * X - 1
    X = jnp.broadcast_to(X[None, None, None, :, :], shape)

    return cast(Array, X)
