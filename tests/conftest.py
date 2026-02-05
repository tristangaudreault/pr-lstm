from typing import Sequence

import pytest

import jax
import jax.numpy as jnp


@pytest.fixture
def make_tree():
    def _make_tree(shapes: Sequence[int | Sequence[int]]):
        shape_arr = jnp.stack(
            jnp.broadcast_arrays(*(jnp.array(shape, ndmin=1) for shape in shapes)),
            axis=1,
        )
        return [
            jax.random.uniform(jax.random.PRNGKey(i), shape)
            for i, shape in enumerate(shape_arr)
        ]

    yield _make_tree
