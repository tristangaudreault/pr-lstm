import math
import jax.numpy as jnp
import jax.nn
from einops import rearrange


def reshape_with_padding(
    x: jnp.ndarray, rows: int | None = None, cols: int | None = None
) -> jnp.ndarray:
    seq = x.shape[1]

    if rows is None and cols is None:
        raise ValueError("At least one of either `rows` or `cols` must be specified.")
    elif rows is None:
        rows = math.ceil(seq / cols)  # type: ignore
    elif cols is None:
        cols = math.ceil(seq / rows)

    pad_len = (rows * cols) - seq  # type: ignore
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))

    pattern = "batch (rows cols) dim -> (batch rows) cols dim"
    return rearrange(x, pattern, rows=rows)


def prepend_cls(x: jnp.ndarray):
    x = jnp.pad(x, ((0, 0), (0, 0), (0, 1)))
    cls = jax.nn.one_hot(x.shape[2] - 1, x.shape[2])
    cls = jnp.broadcast_to(cls, (x.shape[0], 1, x.shape[2]))
    return jnp.concatenate((cls, x), axis=1)
