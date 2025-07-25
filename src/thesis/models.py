import logging

logger = logging.getLogger(__name__)

import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn

from thesis.utils import infer_shape, reshape_with_padding, scaled_dot_product_attention


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
        self.hidden_size = hidden_size
        self.K = K
        self.rows = rows
        self.cols = cols

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, length, _ = x.shape
        partitions, rows, cols = self.partition(x)
        rnn = hk.VanillaRNN(self.hidden_size)

        s0 = rnn.initial_state(batch_size)
        keys = self.sample_keys(batch_size, rows - 1, std_dev=2)

        values = self.generate_values(rnn, s0, keys, partitions)
        first_values = values[:, :1, :, :]
        initial_query = first_values[:, 0, -1:, :]
        values = values[:, 1:, :, :].reshape(
            batch_size, rows - 1, self.K, cols, self.hidden_size
        )

        def marginalize(query: jnp.ndarray, xs: jnp.ndarray):
            key, value = xs
            states, _ = scaled_dot_product_attention(query, key, value)
            next_query = states[:, :, -1, :]
            return next_query, states[:, 0, :, :]

        # Could potentially use the jax.lax.associative_scan
        final_query, ys = hk.scan(
            marginalize,
            initial_query,
            (
                keys.swapaxes(0, 1),
                values.swapaxes(0, 1),
            ),
        )

        ys = jnp.concatenate((first_values, ys.swapaxes(0, 1)), axis=1)
        ys = ys.reshape(batch_size, rows * cols, self.hidden_size)
        ys = ys[:, :length, :]

        return ys

    def partition(self, x: jnp.ndarray) -> tuple[jnp.ndarray, int, int]:
        length = x.shape[1]
        rows, cols = infer_shape(self.rows, self.cols, length)
        return reshape_with_padding(x, rows, cols), rows, cols

    def sample_keys(
        self, batch_size: int, num_keys: int, std_dev: float = 1
    ) -> jnp.ndarray:
        key = hk.next_rng_key()
        epsilon = jax.random.normal(
            key,
            shape=(
                batch_size,
                num_keys,
                self.K,
                self.hidden_size,
            ),
            # minval=-1,
            # maxval=1,
        )
        return jnn.relu(epsilon * std_dev)

    def generate_values(
        self,
        rnn: hk.RNNCore,
        s0: jnp.ndarray,
        keys: jnp.ndarray,
        partitions: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_size, rows, cols, dim = partitions.shape
        num_keys = (rows - 1) * self.K

        inputs_vbatch = jnp.concatenate(
            (partitions[:, :1, :], partitions[:, 1:, :].repeat(repeats=self.K, axis=1)),
            axis=1,
        ).reshape(-1, cols, dim)
        states_vbatch = jnp.concatenate(
            (
                s0[:, jnp.newaxis, :],
                keys.reshape(batch_size, num_keys, self.hidden_size),
            ),
            axis=1,
        ).reshape(-1, self.hidden_size)
        outputs_vbatch, _ = hk.dynamic_unroll(
            rnn, inputs_vbatch, states_vbatch, time_major=False
        )

        values = outputs_vbatch.reshape(batch_size, (num_keys + 1), cols, self.hidden_size)  # type: ignore
        return values
