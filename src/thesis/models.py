import haiku as hk
import jax
from jax.core import Tracer
import jax.numpy as jnp
import jax.nn as jnn
import logging

logger = logging.getLogger(__name__)


class Speculative(hk.Module):
    def __init__(
        self,
        hidden_size: int,
        M: int,
        K: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.M = M
        self.K = K

        self.w_x = hk.Linear(hidden_size, with_bias=False, name="w_xh")
        self.transition = hk.Linear(K, with_bias=True, name="w_hz")

    def __call__(self, x: jax.Array) -> jax.Array:
        x_proj = self.w_x(x)
        S = hk.get_parameter(
            "S", shape=(self.K, self.hidden_size), init=hk.initializers.RandomUniform()
        )
        logits = (
            jnn.tanh(x_proj[:, :, jnp.newaxis, :] + S[jnp.newaxis, jnp.newaxis, :, :])
            - S
        )

        assignments = self.transition(logits)
        ys = scan(assignments)
        output = jnp.matmul(ys, S)

        if not isinstance(assignments, Tracer):
            assignments_mean = assignments.mean(axis=[0, 1, 2])
            logger.debug(f"assignments mean: {assignments_mean}")

        return output


def scan(assignments: jax.Array):
    raw_ys = jax.lax.associative_scan(
        jnp.matmul,
        assignments,
        axis=1,
    )
    ys = raw_ys[:, :, 0, :]

    return ys
