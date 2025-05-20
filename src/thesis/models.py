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

    def initial_state(self, batch_size):
        return jnp.repeat(
            self.rnn_cell.initial_state(batch_size), self.num_branches, axis=0
        )

    @staticmethod
    def adapt_input(x, num_branches):
        """Adapt the input for use by the Chorus model. Returns an array of shape (batch size, number of branches, branch Length, embedding dimension)"""
        batch_size, seq_len, embed_dim = x.shape
        branch_len = -(-seq_len // num_branches)

        required_len = num_branches * branch_len
        pad_len = required_len - seq_len
        x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0))) if pad_len > 0 else x

        x = x.reshape(batch_size, num_branches, branch_len, embed_dim)
        return x.transpose(2, 0, 1, 3).reshape(
            branch_len, batch_size * num_branches, embed_dim
        )

    def process(self, x, h0):
        return hk.dynamic_unroll(self.rnn_cell, x, h0)[1]

    def __call__(self, x, h0):
        return self.process(self.adapt_input(x, self.num_branches), h0)


class ChorusRNN(Chorus):
    def __init__(self, hidden_size, num_branches, name=None):
        super().__init__(hidden_size, num_branches, name=name)
        self.composer = hk.GRU(hidden_size=hidden_size)

    def process(self, x, h0):
        branch_hxs = super().process(x, h0[0])
        branch_hxs = jnp.reshape(branch_hxs, (-1, self.num_branches, self.hidden_size))
        return hk.dynamic_unroll(self.composer, branch_hxs, h0[1], time_major=False)[1]

    def initial_state(self, batch_size):
        return super().initial_state(batch_size), self.composer.initial_state(
            batch_size
        )


class ChorusAttn(Chorus):
    def __init__(self, hidden_size, num_branches, num_heads=4, name=None):
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

        return hk.scan(f, h0, x)[0]


class ChorusTransformer(Chorus):
    def __init__(self, hidden_size, num_branches, name=None):
        super().__init__(hidden_size, num_branches, name=name)
        self.transformer = transformer.make_transformer_encoder(
            output_size=2,
            embedding_dim=hidden_size,
            num_layers=1,
            num_heads=4,
            num_hiddens_per_head=None,
            use_embeddings=False,
            return_all_outputs=True,
        )

    def process(self, x, h0):
        hx = super().process(x, h0)
        hx = jnp.reshape(hx, (-1, self.num_branches, self.hidden_size))
        hx = self.transformer(hx)
        return jnp.mean(hx, axis=1)
