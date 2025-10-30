import haiku as hk
import jax
from jax.core import Tracer
import jax.numpy as jnp
import jax.nn as jnn
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def sigma_z(x, alpha=1.0):
    activated = jnn.relu(jnp.abs(x) - alpha)
    # normalized = activated / (jnp.sum(activated, axis=-1, keepdims=True) + 1e-6)
    signed = activated * jnp.sign(x)

    return signed


def get_S(hidden_size, K) -> jax.Array:
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (hidden_size, K))
    Q, R = jnp.linalg.qr(A)
    Q *= jnp.sign(jnp.diag(R))
    S = Q[jnp.newaxis, jnp.newaxis, :, :]
    return S


def pow_signed(x, p):
    return jnp.sign(x) * (jnp.abs(x) ** p)


class Speculative(hk.Module):
    def __init__(
        self,
        hidden_size: int,
        K: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.K = K

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        input_size = x.shape[-1]
        S = get_S(self.hidden_size, self.K)
        W_h = hk.get_parameter(
            "W_h",
            (self.hidden_size, self.hidden_size),
            init=hk.initializers.Orthogonal(),
        )
        W_x = hk.get_parameter(
            "W_x",
            (self.hidden_size, input_size),
            init=hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(input_size)),
        )
        b_h = hk.get_parameter("b_h", (self.hidden_size, 1), init=jnp.zeros)

        x = x[:, :, :, jnp.newaxis]
        logits = jnn.relu(W_h @ S + W_x @ x + b_h)
        Z = S.transpose(0, 1, 3, 2) @ logits
        output = jax.lax.associative_scan(
            compose,
            Z,
            axis=1,
        )
        output = output[:, :, :, 0]

        if not isinstance(Z, Tracer):
            print("Z:", jnp.mean(Z, axis=(0, 1, 2)))
            print("output:", jnp.max(output, (0, 1)))

        if x.shape[0] > 128:
            jax.debug.print(
                "Z: {}\noutput assigments: {}",
                Z[0, -1, :, 0],
                output[0, -1, :],
            )
        return output


def compose(a, b):
    mult = b @ a
    mult = 4 * (mult ** 3)
    return mult / jnp.linalg.norm(mult, axis=-2, keepdims=True)
