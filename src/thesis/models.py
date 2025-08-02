import logging
from multiprocessing import Process, Pipe

logger = logging.getLogger(__name__)

import haiku as hk
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


class PRNN(hk.RNNCore):
    def __init__(
        self,
        inner_core: type[hk.RNNCore],
        proj_size: int,
        hidden_size: int,
        name: str = "softmax_rnn",
    ):
        super().__init__(name=name)
        self._inner_core = inner_core
        self.proj_size = proj_size
        self.hidden_size = hidden_size

    def __call__(self, inputs, prev_state):
        prev_state_proj = hk.Linear(self.proj_size)(prev_state)
        hidden_state_proj, new_state = self._inner_core(self.proj_size)(  # type: ignore
            inputs, prev_state_proj
        )
        logits = hk.Linear(self.hidden_size)(hidden_state_proj)
        probabilites = jnn.softmax(logits, axis=-1)
        return probabilites, probabilites

    def initial_state(self, batch_size: int | None):
        state = jnp.zeros([self.hidden_size])
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state


class Speculative(hk.Module):
    def __init__(
        self,
        hidden_size: int,
        K: int,
        rows: int | None = 2,
        cols: int | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = 2
        self.K = 2
        self.rows = rows
        self.cols = cols

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape
        rows, cols = infer_shape(self.rows, self.cols, seq_len)
        num_keys = rows - 1

        rnn = PRNN(inner_core=hk.VanillaRNN, proj_size=2, hidden_size=2)
        s0 = rnn.initial_state(batch_size)

        partitions = Speculative.partition(x, rows, cols)
        keys = Speculative.generate_keys(batch_size, num_keys, self.hidden_size)
        first_values, query_init, values = self.unroll_keys(
            rnn, s0, keys, partitions, self.K, self.hidden_size
        )
        ys = Speculative.refine(query_init, keys, values)
        ys = Speculative.assemble_output(first_values, ys, seq_len)

        return ys

    @staticmethod
    @cleartrace.traced
    def partition(x: jnp.ndarray, rows, cols) -> jnp.ndarray:
        partitions = reshape_with_padding(x, rows, cols)
        return partitions

    @staticmethod
    @cleartrace.traced
    def generate_keys(batch_size: int, num_keys: int, key_size: int) -> jnp.ndarray:
        eye = jnp.eye(key_size)
        broadcasted = jnp.broadcast_to(eye, (batch_size, num_keys, key_size, key_size))
        return broadcasted

    @staticmethod
    @cleartrace.traced
    def unroll_keys(
        rnn: hk.RNNCore,
        s0: jnp.ndarray,
        keys: jnp.ndarray,
        partitions: jnp.ndarray,
        K: int,
        hidden_size: int = 13,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        batch_size, rows, cols, dim = partitions.shape

        inputs_vbatch = jnp.concatenate(
            (partitions[:, :1, :], partitions[:, 1:, :].repeat(repeats=K, axis=1)),
            axis=1,
        ).reshape(-1, cols, dim)
        states_vbatch = jnp.concatenate(
            (
                s0[:, jnp.newaxis, :],
                keys.reshape(batch_size, -1, hidden_size),
            ),
            axis=1,
        ).reshape(-1, hidden_size)
        outputs_vbatch, _ = hk.dynamic_unroll(
            rnn, inputs_vbatch, states_vbatch, time_major=False
        )

        unrolled_keys = outputs_vbatch.reshape(batch_size, -1, cols, hidden_size)  # type: ignore
        first_values = unrolled_keys[:, :1, :, :]
        query_init = unrolled_keys[:, 0, -1:, :]
        values = unrolled_keys[:, 1:, :, :].reshape(
            batch_size, -1, K, cols, hidden_size
        )
        return first_values, query_init, values

    @staticmethod
    @cleartrace.traced
    def refine(query_init, keys, values):
        def attend(query: jnp.ndarray, xs: jnp.ndarray):
            key, value = xs
            states, _ = scaled_dot_product_attention(query, key, value)
            next_query = states[:, :, -1, :]
            y = states[:, 0, :, :]
            return next_query, y

        # Could potentially use the jax.lax.associative_scan
        final_query, ys = hk.scan(
            attend,
            query_init,
            (
                keys.swapaxes(0, 1),
                values.swapaxes(0, 1),
            ),
        )

        return ys

    @staticmethod
    @cleartrace.traced
    def assemble_output(first_values: jnp.ndarray, ys: jnp.ndarray, seq_len: int):
        ys = jnp.concatenate((first_values, ys.swapaxes(0, 1)), axis=1)
        ys = rearrange(ys, "b r c h -> b (r c) h")
        ys = ys[:, :seq_len, :]

        return ys
