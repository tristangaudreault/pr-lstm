import jax
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk



def squared_norm(a: Array, /, axis: int = -1) -> Array:
    """Squared euclidean norm over a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, default -1
        Axis along which a squared euclidean norm is computed.

    Returns
    -------
    array_like
        An array with the same shape as a, with the specified axis removed.
    """
    return jnp.sum(a**2, axis=axis)


def pairwise_euclidean(a: Array, b: Array, /) -> Array:
    """Computes pairs of squared euclidean distances

    Parameters
    ----------
    a, b : array_like
        Input arrays of shape (..., M, N) and (..., N, P) respectively.

    Returns
    -------
    array_like
        An array of shape (..., M, N).
    """
    A2, B2 = map(squared_norm, (a, b))
    A2, B2 = A2[..., None], B2[..., None, :]
    AB = a @ b.swapaxes(-1, -2)
    return A2 + B2 - 2 * AB


def rbf_kernel_matrix(a: Array, b: Array, /, gamma: int | float | Array = 1) -> Array:
    """Radial basis function kernel matrix.

    Parameters
    ----------
    a, b : array_like
        Input arrays of shape (..., M, N) and (..., N, P) respectively.
    gamma : int or float or array_like, default 1
        Free parameter, equivalent to ``1 /  (2 * sigma ** 2)``.

    Returns
    -------
    array_like
        An array of shape (..., M, P).
    """

    return jnp.exp(-gamma * pairwise_euclidean(a, b))


def hardmax(x: Array, /):
    y = jnn.one_hot(jnp.argmax(x, axis=-1), x.shape[-1])
    
    return y


def gumbel_softmax(logits: Array, tau: float, hard: bool = False):
    g = jax.random.gumbel(hk.next_rng_key(), shape=logits.shape)
    y_soft = jnn.softmax((logits + g) / tau)
    if not hard:
        return y_soft
    y_hard = hardmax(y_soft)
    return jax.lax.stop_gradient(y_hard - y_soft) + y_soft
