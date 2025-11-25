from jax import Array
import jax.numpy as jnp


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
    A2, B2 = map(squared_norm, (a, b))
    A2, B2 = A2[..., None], B2[..., None, :]
    AB = a @ b.swapaxes(-1, -2)

    return jnp.exp(-gamma * (A2 + B2 - 2 * AB))


def build_regressor(specs: Array):
    def regressor(prev_state: Array, targets: Array):
        similarity_matrix = rbf_kernel_matrix(prev_state, specs, gamma=1)
        probability_matrix = similarity_matrix / jnp.sum(similarity_matrix, axis=-1, keepdims=True)
        regressions = targets
        expectation = probability_matrix @ regressions
        return expectation

    return regressor
