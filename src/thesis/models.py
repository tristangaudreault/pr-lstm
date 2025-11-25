import haiku as hk
import jax
from jax import Array
import jax.numpy as jnp
import logging

from .helpers import create_ensemble, make_rnn, unroll

logger = logging.getLogger(__name__)


def get_specs(shape: tuple[int, ...], learnable: bool) -> Array:
    """Creates speculation tensor

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the required speculation tensor.
    learnable : bool
        If set to ``True`` the speculations will be registered as parameters, and will therefore be learned.
        Otherwise they will be randomly chosen during every call.

    Returns
    -------
    array_like
        Speculation tensor of shape ``shape``, with values sampled from the uniform distribution between ``-1`` and ``1``.
    """
    init = hk.initializers.RandomUniform(minval=-1)
    if learnable:
        return hk.get_parameter(
            "S",
            shape=shape,
            init=init,
        )
    else:
        return init(shape, jnp.float32)


class Speculative(hk.Module):
    """Speculative model.

    Attributes
    ----------
    hidden_size : int
        Hidden size of the inner RNN cells.
    J : int
        Size of the ensemble of RNN cells.
    K : int
        Number of speculations to perform.
    learnable_specs : bool
        If set to ``True`` the speculations will be registered as parameters, and will therefore be learned.
        Otherwise they will be randomly chosen during every call.
    """

    def __init__(
        self,
        hidden_size: int,
        J: int,
        K: int,
        learnable_specs: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.J = J
        self.K = K
        self.learnable_specs = learnable_specs

    def __call__(self, xs: jax.Array) -> jax.Array:
        """Applies the speculative model.

        Parameters
        ----------
        xs : array_like
            Input sequence tensor of shape ``(B, T, I)``.

        Returns
        -------
        array_like
            Output sequence tensor of shape (B, T, H * J).
        """
        batch_size, seq_size, _ = xs.shape

        rnn_vmap = create_ensemble(make_rnn(self.hidden_size), self.J)
        xs = xs[:, None, ...]
        specs = get_specs((self.J, self.K, 1, self.hidden_size), self.learnable_specs)

        hs = unroll(rnn_vmap, xs, specs, time_major=False)
        hs = hs.reshape(batch_size, seq_size, -1)

        return hs
