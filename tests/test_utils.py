from thesis import utils

import jax.numpy as jnp


def test_reshape_with_padding():
    x = jnp.arange(1, 12).reshape(-1, 1, 1)
    result = utils.reshape_with_padding(x, 4)
    expected = jnp.array([*range(1, 12), 0]).reshape(4, 3, 1, 1)
    assert jnp.all(result == expected)
