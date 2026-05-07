from typing import Any, Callable

import jax


def map_with_index(fn: Callable, tree: Any):
    leaves, treedef = jax.tree.flatten(tree)

    new_leaves = [fn(i, leaf) for i, leaf in enumerate(leaves)]

    return jax.tree.unflatten(treedef, new_leaves)
