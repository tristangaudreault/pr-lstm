from typing import Any, Callable, Sequence, Mapping
from itertools import accumulate

import jax
import jax.numpy as jnp


def concat_tree(tree: Any, axis: int | None = -1) -> jax.Array:
    """Concatenates the leaves of a pytree along a given axis.

    Args:
        tree (Any): Input tree with broadcastable leaf dimensions along the non-concatenated axes.
        axis (int | None, optional): Concatenation axis. Defaults to -1.

    Returns:
        jax.Array: The array with tree leaves concatenated along the input axis.
    """ """"""
    return jnp.concatenate(jax.tree.flatten(tree)[0], axis=axis)


def get_split_array(
    tree: Any, axis: int = -1
) -> tuple[Callable[[jax.Array], Any], int]:
    """Creates a function to reverse tree concatenation based on a template tree.

    Args:
        tree (Any): Template tree.
        axis (int, optional): Split axis. Defaults to -1.

    Returns:
        tuple[Callable[[jax.Array], Any], int]: Tuple containing the tree concatenation inversion function and the total size of the concatenated dimension.
    """
    leaves, treedef = jax.tree.flatten(tree)

    *split_indices, total_size = list(accumulate([leaf.shape[axis] for leaf in leaves]))

    def split_array(arr: jax.Array):
        return jax.tree.unflatten(treedef, jnp.split(arr, split_indices, axis=axis))

    return split_array, total_size


@jax.custom_vjp
def print_grad(x):
    return x


def print_grad_fwd(x):
    return x, None


def print_grad_bwd(_, x_grad):
    jax.debug.print("x_grad: {}", x_grad, ordered=True)
    return (x_grad,)


print_grad.defvjp(print_grad_fwd, print_grad_bwd)
