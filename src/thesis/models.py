from typing import cast

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn
from einops import rearrange
import logging

from thesis.utils import (
    reshape_with_padding,
    scaled_dot_product_attention,
)

logger = logging.getLogger(__name__)


class Speculative(hk.Module):
    def __init__(
        self,
        hidden_size: int,
        M: int,
        K: int,
        temperature: float,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.M = M
        self.K = K
        self.temperature = temperature

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape
        N = -(-seq_len // self.M)

        rnn = hk.VanillaRNN(
            hidden_size=self.hidden_size,
        )

        X_cal = get_X_cal(x, N, self.M, self.K)
        S_cal = get_S_cal(batch_size, N, self.K, self.hidden_size)

        H = g(
            rnn,
            X_cal,
            S_cal,
        )
        ys = scan(H, temperature=self.temperature)

        ys = transform_outputs(ys, seq_len)

        return ys


def get_X_cal(x: jnp.ndarray, N: int, M: int, K: int) -> jnp.ndarray:
    batch_size, _, embed_dim = x.shape

    reshaped = reshape_with_padding(x, N, M)
    expanded = reshaped[:, :, jnp.newaxis, :, :]
    broadcasted = jnp.broadcast_to(
        expanded,
        (batch_size, N, K, M, embed_dim),
    )
    X_cal = broadcasted.transpose(3, 0, 1, 2, 4)

    return X_cal


def get_S_cal(batch_size: int, N: int, K: int, hidden_size: int):
    raw = hk.get_parameter(
        "S", shape=(K, hidden_size), init=hk.initializers.RandomUniform()
    )
    expanded = raw[jnp.newaxis, jnp.newaxis, :, :]
    S_cal = jnp.broadcast_to(expanded, (batch_size, N - 1, K, hidden_size))
    S_cal = jnp.pad(S_cal, ((0, 0), (1, 0), (0, 0), (0, 0)))

    return S_cal


def g(
    rnn: hk.RNNCore,
    X_cal: jnp.ndarray,
    S_cal: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    X_vbatch = X_cal.reshape(X_cal.shape[0], -1, X_cal.shape[-1])
    S_vbatch = S_cal.reshape(-1, S_cal.shape[-1])

    outputs_vbatch, _ = hk.dynamic_unroll(rnn, X_vbatch, S_vbatch)
    outputs_vbatch = cast(jnp.ndarray, outputs_vbatch)
    outputs = outputs_vbatch.reshape(X_cal.shape[0], *S_cal.shape)
    transposed_outputs = outputs.transpose(1, 2, 3, 0, 4)
    return (S_cal, transposed_outputs)


def scan(H: tuple[jnp.ndarray, jnp.ndarray], temperature: float):
    def compose(h_a: jnp.ndarray, h_b: jnp.ndarray):
        key_a, value_a = h_a
        query = value_a[:, :, :, -1, :]
        key, value = h_b
        new_value, attn_weights = scaled_dot_product_attention(
            query, key, value, temperature=temperature
        )
        return key_a, new_value

    _, raw_ys = jax.lax.associative_scan(
        compose,
        H,
        axis=1,
    )
    ys = jnp.take(raw_ys, 0, axis=2)

    return ys


def transform_outputs(ys: jnp.ndarray, seq_len: int):
    ys = rearrange(ys, "b r c h -> b (r c) h")
    ys = ys[:, :seq_len, :]

    return ys
