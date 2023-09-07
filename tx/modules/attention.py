import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from jaxtyping import Array, Float

from .common import Module


class Attention(Module):
    num_heads: int
    head_dim: int
    model_dim: int
    context_length: int
    init_range: float = 0.02

    def setup(self):
        self.mask = nn.make_causal_mask(
            jnp.ones((1, self.context_length), dtype="bool"),
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
    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        """
        References:
        - `flax.linen.attention`.
        """
        batch_size = x.shape[0]

        # Apply a linear transformation to the input tensor.
        hidden_states = self.c_attn(x)

        # Split the hidden states into query, key, and value.
        query, key, value = self._split_outputs(hidden_states)
        query_length, key_length = query.shape[1], key.shape[1]
        self.intermediate("query", query)
        self.intermediate("key", key)
        self.intermediate("value", value)

        # Compute the attention weights.
        depth = query.shape[-1]
        query = query / jnp.sqrt(depth)
        scores = jnp.einsum("...qhd,...khd->...hqk", query, key)
        self.intermediate("scores", scores)

        # Apply the causal mask to the attention weights.
        mask = self.mask[:, :, :query_length, :key_length]
        mask = jnp.broadcast_to(
            mask,
            (batch_size, *mask.shape[1:]),
        )
        self.intermediate("mask", mask)

        big_neg = jnp.finfo(jnp.float32).min
        scores = jnp.where(mask, scores, big_neg)

        # Normalize the attention weights
        pattern = jax.nn.softmax(scores)
        self.intermediate("pattern", pattern)

        # Apply the attention pattern to the value tensor.
        z = jnp.einsum("...hqk,...khd->...qhd", pattern, value)
        self.intermediate("z", z)

        # Apply a linear transformation to the attention output.
        output = self.c_proj(self._merge_heads(z))
        return output

    def _split_outputs(self, hidden_states: Array):
        return map(self._split_heads, jnp.split(hidden_states, 3, axis=2))

    def _split_heads(self, hidden_states: Array):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states: Array):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.model_dim,))
