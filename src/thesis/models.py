import jax
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk
import logging

from .rnn import create_ensemble, make_rnn
from .speculation import sobol
from .regressions import pairwise_euclidean, gumbel_softmax, hardmax

logger = logging.getLogger(__name__)


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
    learnable_spec_points : bool
        If set to ``True`` the speculations will be registered as parameters, and will therefore be learned.
        Otherwise they will be randomly chosen during every call.
    """

    def __init__(
        self,
        hidden_size: int,
        J: int,
        K: int,
        temperature: float,
        learnable_specs: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.J = J
        self.K = K
        self.temperature = temperature
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
            Output sequence tensor of shape (B, T, J * H).
        """
        batch_size, seq_size, _ = xs.shape

        rnn_ensemble = create_ensemble(make_rnn(self.hidden_size), self.J)
        speculations = hk.get_parameter(
            "S", (1, 1, self.J, self.K, self.hidden_size), init=sobol
        )
        xs = xs[:, :, None, :]
        _, responses = rnn_ensemble(speculations, xs)

        def predictor(test_points, response_points):
            similarity_matrix = -pairwise_euclidean(test_points, speculations)
            weights = (
                gumbel_softmax(
                    similarity_matrix / self.temperature,
                    tau=1.0 / self.temperature,
                    hard=False,
                )
                if self.temperature > 0.0
                else hardmax(similarity_matrix)
            )
            return weights @ response_points

        hs = jax.lax.associative_scan(predictor, responses, axis=1)

        hs = hs.take(0, -2)
        hs = hs.reshape(batch_size, seq_size, -1)
        size = self.J * self.hidden_size
        hs = hk.Linear(size, name="combine")(hs)

        return hs
