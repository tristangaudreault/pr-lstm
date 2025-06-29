import pytest
import jax.numpy as jnp
import numpy as np
import jax
from jax import random

from thesis import utils


def visual_test_reshape_with_padding():
    x = jnp.arange(1, 12).reshape(1, -1, 1)
    result = utils.reshape_with_padding(x, cols=4)
    jax.debug.print(
        "x: {x} (shape: {x_shape})\ny: {y} (shape: {y_shape})",
        x=x,
        x_shape=x.shape,
        y=result,
        y_shape=result.shape,
    )
    # expected = jnp.arange(1, 13).reshape(4, 3, 1, 1)
    # assert jnp.all(result == expected)


def visual_test_prepend_cls():
    x = jnp.arange(1, 13).reshape(12, 1, 1)
    y = utils.prepend_cls(x)
    jax.debug.print(
        "x: {x} (shape: {x_shape})\ny: {y} (shape: {y_shape})",
        x=x,
        x_shape=x.shape,
        y=y,
        y_shape=y.shape,
    )


if __name__ == "__main__":
    visual_test_prepend_cls()
