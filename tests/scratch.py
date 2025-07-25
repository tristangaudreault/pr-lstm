from typing import Callable

import jax
import jax.numpy as jnp


def rnn(params, inputs, s0, f: Callable):
    """
    Run the RNN over input sequence.

    Args:
        params: dict with keys 'W', 'U', 'b'
        inputs: (T, input_dim) array
        s0: (hidden_dim,) initial hidden state
        f: activation function

    Returns:
        s_seq: (T, hidden_dim) array of hidden states
    """
    W, U, b = params["W"], params["U"], params["b"]

    def step(s_prev, x_t):
        a_t = W @ x_t + b
        b_t = U @ s_prev
        s_t = f(a_t + b_t)
        jax.debug.print("s_t = {s_t}", s_t=s_t)
        return s_t, s_t  # new carry and output

    _, s_seq = jax.lax.scan(step, s0, inputs)
    return s_seq


def parallel_rnn(params, inputs, s0, f: Callable):
    """
    Run the RNN over input sequence.

    Args:
        params: dict with keys 'W', 'U', 'b'
        inputs: (T, input_dim) array
        s0: (hidden_dim,) initial hidden state
        f: activation function

    Returns:
        s_seq: (T, hidden_dim) array of hidden states
    """
    W, U, b = params["W"], params["U"], params["b"]
    a = inputs @ W.T + b

    e_a = f(a)
    U_e_a = e_a @ U.T
    i_a = f(U_e_a)
    b0 = f(U @ s0)
    carry = b0
    for i in i_a[:-1]:
        carry = jnp.pow(i, carry)
        jax.debug.print("carry = {carry}", carry=carry)
    carry = e_a[-1] * carry
    jax.debug.print("carry = {carry}", carry=carry)

    return carry


def test_exponential():
    for T in (4,):
        key = jax.random.PRNGKey(T)
        input_dim, hidden_dim = 4, 8

        # Random input
        x = jax.random.normal(key, (T, input_dim))

        # Random parameters
        params = {
            "W": jax.random.normal(key, (hidden_dim, input_dim)),
            "U": jax.random.normal(key, (hidden_dim, hidden_dim)),
            "b": jax.random.normal(key, (hidden_dim,)),
        }

        # Initial state
        s0 = jnp.zeros((hidden_dim,))

        # Activation function
        f = lambda x: jnp.pow(1.1, x)

        # Run RNN
        expected_s_out = rnn(params, x, s0, f)
        print("Expected output:", expected_s_out[-1, 0], end="\t")
        # print("-----")
        s_out = parallel_rnn(params, x, s0, f)
        print("Actual output:", s_out[0])
        # print("------")
        print("Average diff", jnp.mean(jnp.abs(expected_s_out - s_out)))


if __name__ == "__main__":
    test_exponential()
