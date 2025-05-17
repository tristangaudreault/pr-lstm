import math
import logging

import jax.numpy as jnp
import haiku as hk

from neural_networks_chomsky_hierarchy.models import transformer

logger = logging.getLogger(__name__)


class Chorus(hk.RNNCore):
    def __init__(self, hidden_size, num_branches, name=None):
        super().__init__(name=name)

        self.hidden_size = hidden_size
        self.num_branches = num_branches

        self.rnn_cell = hk.GRU(hidden_size=hidden_size)

    @staticmethod
    def adapt_input(x, num_branches):
        """Adapt the input for use by the Chorus model. Returns an array of shape (batch size, number of branches, branch Length, embedding dimension)"""
        batch_size, seq_len, embed_dim = x.shape
        branch_len = math.ceil(seq_len / num_branches)

        required_len = num_branches * branch_len
        if seq_len != required_len:
            x = jnp.pad(x, ((0, 0), (0, required_len - seq_len % required_len), (0, 0)))

        x = jnp.reshape(
            x,
            (
                batch_size * num_branches,
                branch_len,
                embed_dim,
            ),
        )

        return x.swapaxes(0, 1)

    def process(self, x, h0):
        return hk.dynamic_unroll(self.rnn_cell, x, h0)

    @staticmethod
    def adapt_output(outputs, hx, hidden_size, num_branches):
        hx = jnp.reshape(hx, (-1, num_branches * hidden_size))
        hx = jnp.expand_dims(hx, axis=1)

        return outputs, hx

    def __call__(self, x, h0):
        adapted_x = self.adapt_input(x, self.num_branches)
        outputs, hx = self.process(adapted_x, h0)

        return self.adapt_output(outputs, hx, self.hidden_size, self.num_branches)

    def initial_state(self, batch_size):
        return jnp.repeat(
            self.rnn_cell.initial_state(batch_size), self.num_branches, axis=0
        )


class ChorusRNN(Chorus):
    def __init__(self, hidden_size, num_branches, name=None):
        super().__init__(hidden_size, num_branches, name=name)

        self.composer = hk.GRU(hidden_size=hidden_size)

    def process(self, x, h0):
        _, branch_hxs = super().process(x, h0[0])
        branch_hxs = jnp.reshape(branch_hxs, (-1, self.num_branches, self.hidden_size))

        return hk.dynamic_unroll(self.composer, branch_hxs, h0[1], time_major=False)

    @staticmethod
    def adapt_output(outputs, hx, hidden_size, num_branches):
        return outputs, jnp.expand_dims(hx, axis=1)

    def initial_state(self, batch_size):
        return super().initial_state(batch_size), self.composer.initial_state(
            batch_size
        )


class ChorusAttn(Chorus):
    def __init__(self, hidden_size, num_branches, num_heads, name=None):
        super().__init__(hidden_size, num_branches, name=name)

        self.multihead_attn = hk.MultiHeadAttention(
            num_heads=num_heads,
            key_size=hidden_size // num_heads,
            w_init=hk.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
        )

    def process(self, x, h0):
        def f(hx, xt):
            output, hx = self.rnn_cell(xt, hx)

            hx = self.multihead_attn(hx, hx, hx)

            return hx, output

        hx, outputs = hk.scan(f, h0, x)

        return outputs, hx


class ChorusTransformer(Chorus):
    def __init__(self, hidden_size, num_branches, name=None):
        super().__init__(hidden_size, num_branches, name=name)

        config = transformer.TransformerConfig(
            output_size=hidden_size,
            embedding_dim=hidden_size,
            num_layers=1,
            num_heads=4,
            num_hiddens_per_head=None,
            use_embeddings=False,
        )
        self.transformerEncoder = transformer.TransformerEncoder(config)

    def process(self, x, h0):
        def f(hx, xt):
            output, hx = self.rnn_cell(xt, hx)

            hx = jnp.reshape(hx, (-1, self.num_branches, self.hidden_size))
            hx = self.transformerEncoder(hx)
            hx = jnp.reshape(hx, (-1, self.hidden_size))

            return hx, output

        hx, outputs = hk.scan(f, h0, x)

        return outputs, hx
