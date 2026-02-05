from functools import partial

from thesis import utils

import jax
import jax.numpy as jnp


def test_tree_concat_inversion(make_tree):
    tree = make_tree([5, [10, 6, 9], 4, 3])
    split_array, total_size = utils.get_split_array(tree, axis=1)

    arr = utils.concat_tree(tree, axis=1)
    tree_recovered = split_array(arr)

    assert jax.tree.all(jax.tree.map(jnp.array_equal, tree, tree_recovered))
