import numpy as np
import jax.numpy as jnp

from thesis import models


def test_adapt_input():
    x = jnp.array(
        [
            [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0], [5.0, 0.0]],
            [[0.0, 6.0], [7.0, 0.0], [0.0, 8.0], [9.0, 0.0], [0.0, 10.0]],
        ]
    )

    adapted_x = model.Chorus.adapt_input(x, 3)

    expected_adapted_x = jnp.array(
        [
            [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [0.0, 6.0], [0.0, 8.0], [0.0, 10.0]],
            [[0.0, 2.0], [0.0, 4.0], [0.0, 0.0], [7.0, 0.0], [9.0, 0.0], [0.0, 0.0]],
        ]
    )

    np.testing.assert_array_equal(adapted_x, expected_adapted_x)


def test_adapt_output():
    hx = jnp.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    )

    adapted_hx = model.Chorus.adapt_output(None, hx, 2, 3)[1]

    expected_adapted_hx = jnp.array(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]]
    )

    np.testing.assert_array_equal(adapted_hx, expected_adapted_hx)
