from typing import Tuple
from jaxtyping import Array, Float

from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

from .common import Module


class Attention(Module):
    num_heads: int
    head_dim: int
    model_dim: int
    context_length: int
    init_range: float = 0.02

    def setup(self):
        self.mask = nn.make_causal_mask(
            jnp.ones((self.context_length), dtype="bool"),
            dtype="bool",
        )

        init_dense = partial(
            nn.DenseGeneral,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
        )

        self.c_attn = init_dense(features=3 * self.model_dim)
        self.c_proj = init_dense(features=self.model_dim)

    @nn.compact
    def __call__(self, x: Float[Array, "seq embed"]) -> Float[Array, "seq embed"]:
        """
        References:
        - `flax.linen.attention`.
        """
        # Apply a linear transformation to the input tensor.
        hidden_states = self.c_attn(x)

        # Split the hidden states into query, key, and value.
        query, key, value = self._split_outputs(hidden_states)
        query_length, key_length = query.shape[-3], key.shape[-3]
        self._qkv_intermediates((query, key, value))

        # Compute the attention weights.
        query = query / jnp.sqrt(query.shape[-1])
        scores = jnp.einsum("...qhd,...khd->...hqk", query, key)
        # self.intermediate("scores", scores)

        # Apply the causal mask to the attention weights.
        mask = self.mask[:, :query_length, :key_length]
        big_neg = jnp.finfo(jnp.float32).min
        scores = jnp.where(mask, scores, big_neg)

        # Normalize the attention weights
        pattern = jax.nn.softmax(scores)
        self.intermediate("pattern", pattern)

        # Apply the attention pattern to the value tensor.
        z = jnp.einsum("...hqk,...khd->...qhd", pattern, value)
        # self.intermediate("z", z)

        # Apply a linear transformation to the attention output.
        output = self.c_proj(self._merge_heads(z))
        return output

    def _qkv_intermediates(self, qkv: Tuple[Array, Array, Array]) -> bool:
        ret_vals = []
        for name, value in zip(("query", "key", "value"), qkv):
            ret_vals.append(self.intermediate(name, value))

        return all(ret_vals)

    def _split_outputs(self, states: Array):
        return map(self._split_heads, jnp.split(states, 3, axis=-1))

    def _split_heads(self, states: Array):
        return states.reshape((states.shape[0], self.num_heads, self.head_dim))

    def _merge_heads(self, states: Array):
        return states.reshape((states.shape[0], self.model_dim))
