import jax.numpy as jnp
import haiku as hk

from thesis.utils import prepend_cls


def rnn_hyperlayer(rnn_core: type[hk.RNNCore], **kwargs):
    def hyperlayer(x: jnp.ndarray):
        core = rnn_core(**kwargs)
        initial_state = core.initial_state(x.shape[0])
        output, final_state = hk.dynamic_unroll(
            core, x, initial_state, time_major=False
        )
        return final_state

    return hyperlayer


def transformer_encoder_hyperlayer(transformer_encoder: type[hk.Module], **kwargs):
    def hyperlayer(x: jnp.ndarray):
        encoder = transformer_encoder(**kwargs)
        x = prepend_cls(x)
        output = encoder(x)
        cls_token_output = output[:, 0, :]
        return cls_token_output

    return hyperlayer
