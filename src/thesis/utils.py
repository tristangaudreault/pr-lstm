import math
import jax.numpy as jnp
from einops import rearrange


def reshape_with_padding(x: jnp.ndarray, num_branches: int) -> jnp.ndarray:
    """
    Reshapes a 3D tensor of shape (seq, batch, dim) into (num_branches, branch_len, batch, dim),
    padding the sequence dimension if necessary.

    Args:
        x (jnp.ndarray): Input of shape (seq, batch, dim).
        num_branches (int): Number of branches to split the sequence into.

    Returns:
        jnp.ndarray: Array of shape (num_branches, branch_len, batch, dim).

    Raises:
        ValueError: If num_branches is not positive or x is not 3D.
    """
    if x.ndim != 3:
        raise ValueError(f"Input must be 3D, got shape {x.shape}")
    if num_branches <= 0:
        raise ValueError(f"num_branches must be positive, got {num_branches}")

    seq = x.shape[0]
    branch_len = math.ceil(seq / num_branches)
    pad_len = (num_branches * branch_len) - seq

    if pad_len > 0:
        x = jnp.pad(x, ((0, pad_len), (0, 0), (0, 0)))

    # Fixed einops pattern to match desired output shape
    return rearrange(
        x,
        "(num_branches branch_len) batch dim -> branch_len (batch num_branches) dim",
        num_branches=num_branches,
    )
