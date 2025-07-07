import haiku as hk
import jax.numpy as jnp

from thesis.utils import prepend_cls


def rnn_hyperlayer(rnn_core: type[hk.RNNCore], *args, **kwargs):
    def hyperlayer(x: jnp.ndarray):
        core = rnn_core(*args, **kwargs)
        initial_state = core.initial_state(x.shape[0])
        output, final_state = hk.dynamic_unroll(
            core, x, initial_state, time_major=False
        )
        return final_state

    return hyperlayer


def cls_hyperlayer(layer: type[hk.Module], *args, **kwargs):
    def hyperlayer(x: jnp.ndarray):
        encoder = layer(*args, **kwargs)
        x = prepend_cls(x)
        output = encoder(x)
        cls_token_output = output[:, 0, :]
        return cls_token_output

    return hyperlayer
