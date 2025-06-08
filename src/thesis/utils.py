import math
import jax.numpy as jnp


def reshape_with_padding(x: jnp.ndarray, seq_rows: int) -> jnp.ndarray:
    """
    Reshapes a 3D tensor of shape (seq, batch, dim) into (seq_rows, seq_cols, batch, dim),
    padding the sequence dimension if necessary.

    Args:
        x (jnp.ndarray): Input of shape (seq, batch, dim).
        seq_rows (int): Desired number of rows in the resulting sequence dimension.

    Returns:
        jnp.ndarray: Return array of shape (seq_rows, seq_cols, batch, dim).
    """
    seq, batch, dim = x.shape
    seq_cols = math.ceil(seq / seq_rows)
    required_len = seq_rows * seq_cols
    pad_len = required_len - seq

    if pad_len > 0:
        x = jnp.pad(x, ((0, pad_len), (0, 0), (0, 0)))

    x = x.reshape(seq_rows, seq_cols, batch, dim)
    return x
