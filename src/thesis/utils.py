import jax.nn
import jax.numpy as jnp
from einops import rearrange


def ceil_division(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def reshape_with_padding(x: jnp.ndarray, rows: int, cols: int) -> jnp.ndarray:
    seq = x.shape[1]

    pad_len = (rows * cols) - seq
    x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))

    pattern = "batch (rows cols) dim -> batch rows cols dim"
    return rearrange(x, pattern, rows=rows)


def prepend_cls(x: jnp.ndarray):
    x = jnp.pad(x, ((0, 0), (0, 0), (0, 1)))
    cls = jax.nn.one_hot(x.shape[2] - 1, x.shape[2])
    cls = jnp.broadcast_to(cls, (x.shape[0], 1, x.shape[2]))
    return jnp.concatenate((cls, x), axis=1)
