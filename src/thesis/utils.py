import math

import jax.nn
import jax.numpy as jnp
from einops import rearrange


def ceil_division(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def infer_shape(rows, cols, sequence_length: int) -> tuple[int, int]:
    if rows is None and cols is None:
        side = math.ceil(math.sqrt(sequence_length))
        return side, side

    if rows is None:
        return ceil_division(sequence_length, cols), cols

    if cols is None:
        return rows, ceil_division(sequence_length, rows)

    return rows, cols


def reshape_with_padding(x: jnp.ndarray, rows: int, cols: int) -> jnp.ndarray:
    seq = x.shape[1]

    pad_len = (rows * cols) - seq
    x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))

    pattern = "batch (rows cols) dim -> batch rows cols dim"
    return rearrange(x, pattern, rows=rows)


def get_attn_weights(query, key):
    d_k = query.shape[-1]

    scores = jnp.einsum("...qd,...kd->...qk", query, key)
    # scores = scores / jnp.sqrt(d_k)
    # attn_weights = jax.nn.softmax(scores, axis=-1)
    return scores


def apply_attn_weights(attn_weights, value):
    return jnp.einsum("...qk,...kch->...qch", attn_weights, value)


def scaled_dot_product_attention(query, key, value):
    attn_weights = get_attn_weights(query, key)
    output = apply_attn_weights(attn_weights, value)
    return output, attn_weights


def add_batch(nest, batch_size: int | None):
    """
    Adds a batch dimension at axis 0 to the leaves of a nested structure.
    References:
        Copied from:
        https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py#L221
    """
    broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
    return jax.tree.map(broadcast, nest)
