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
