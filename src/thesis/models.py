import logging
from typing import cast

logger = logging.getLogger(__name__)

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn
from einops import rearrange

import cleartrace

from thesis.utils import (
    infer_shape,
    reshape_with_padding,
    scaled_dot_product_attention,
    add_batch,
)


class Speculative(hk.Module):
    def __init__(
        self,
        hidden_size: int,
        proj_size: int,
        rows: int | None = None,
        cols: int | None = None,
        temperature: int = 1,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.rows = rows
        self.cols = cols
        self.temperature = temperature

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape
        rows, cols = infer_shape(self.rows, self.cols, seq_len)

        rnn = PRNN(
            inner_core=hk.VanillaRNN,
            hidden_size=self.hidden_size,
            proj_size=self.proj_size,
        )
        s0 = rnn.initial_state(batch_size)

        partitions = partition(x, rows, cols)
        keys = generate_keys(batch_size, rows, self.proj_size, s0)
        values = unroll_keys(rnn, keys, partitions, self.proj_size)
        ys = refine(keys, values, self.temperature)
        ys = assemble_output(ys, seq_len)

        return ys


@cleartrace.traced
def partition(x: jnp.ndarray, rows: int, cols: int) -> jnp.ndarray:
    partitions = reshape_with_padding(x, rows, cols)
    return partitions


@cleartrace.traced
def generate_keys(
    batch_size: int, rows: int, proj_size: int, s0: jnp.ndarray
) -> jnp.ndarray:
    s0_expanded = s0[:, jnp.newaxis, jnp.newaxis, :]
    s0_broadcasted = jnp.broadcast_to(
        s0_expanded, (batch_size, 1, proj_size, proj_size)
    )

    eye = jnp.eye(proj_size)
    eye_broadcasted = jnp.broadcast_to(
        eye, (batch_size, rows - 1, proj_size, proj_size)
    )

    keys = jnp.concatenate((s0_broadcasted, eye_broadcasted), axis=1)
    return keys


@cleartrace.traced
def unroll_keys(
    rnn: hk.RNNCore,
    keys: jnp.ndarray,
    partitions: jnp.ndarray,
    K: int,
) -> jnp.ndarray:
    batch_size, rows, cols, dim = partitions.shape

    inputs = jnp.broadcast_to(
        partitions[:, :, jnp.newaxis, :, :], (batch_size, rows, K, cols, dim)
    ).transpose(3, 0, 1, 2, 4)
    states = keys

    outputs, _ = hk.dynamic_unroll(rnn, inputs, states)

    outputs = cast(jnp.ndarray, outputs)
    values = outputs.transpose(1, 2, 3, 0, 4)
    return values


@cleartrace.traced
def refine(keys: jnp.ndarray, values: jnp.ndarray, temperature: int):
    def attend(a: jnp.ndarray, b: jnp.ndarray):
        key_a, value_a = a
        query = value_a[:, :, :, -1, :]
        key, value = b
        new_value, _ = scaled_dot_product_attention(
            query, key, value, temperature=temperature
        )
        return key_a, new_value

    _, ys = jax.lax.associative_scan(
        attend,
        (keys, values),
        axis=1,
    )
    ys = ys[:, 1:, 0, :, :]
    return ys


@cleartrace.traced
def assemble_output(ys: jnp.ndarray, seq_len: int):
    ys = rearrange(ys, "b r c h -> b (r c) h")
    ys = ys[:, :seq_len, :]

    return ys


class PRNN(hk.RNNCore):
    def __init__(
        self,
        inner_core: type[hk.VanillaRNN],
        proj_size: int,
        hidden_size: int,
        name: str = "softmax_rnn",
    ):
        super().__init__(name=name)
        self._inner_core = inner_core
        self.hidden_size = hidden_size
        self.proj_size = proj_size

    def __call__(self, inputs, prev_state):
        hidden_state_proj, new_state = self._inner_core(self.hidden_size)(
            inputs, prev_state
        )
        logits = hk.Linear(self.proj_size)(hidden_state_proj)
        probabilites = jnn.softmax(logits, axis=-1)
        return probabilites, probabilites

    def initial_state(self, batch_size: int | None):
        state = jnp.zeros([self.proj_size])
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state
