from typing import List, Tuple
from jaxtyping import Array, Float

from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.struct as struct


class Attention(nn.Module):
    num_heads: int
    head_dim: int
    model_dim: int
    init_range: float = 0.02
    use_bias: bool = True

    intermediates: List[str] = struct.field(default_factory=list)

    def intermediate(self, name: str, value: Array) -> bool:
        if name in self.intermediates:
            return self.sow("intermediates", name, value)
        return False

    @nn.compact
    def __call__(
        self, x: Float[Array, "... seq embed"]
    ) -> Float[Array, "... seq embed"]:
        init_dense = partial(
            nn.DenseGeneral,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
            use_bias=self.use_bias,
        )

        # Apply a linear transformation to the input tensor.
        hidden_states = init_dense(name="c_attn", features=3 * self.model_dim)(x)

        # Split the hidden states into query, key, and value.
        query, key, value = self._split_outputs(hidden_states)
        query_length, key_length = query.shape[-3], key.shape[-3]
        self._qkv_intermediates((query, key, value))

        # Compute the attention weights.
        query = query / jnp.sqrt(query.shape[-1])
        scores = jnp.einsum("...qhd,...khd->...hqk", query, key)
        self.intermediate("scores", scores)

        # Apply the causal mask to the attention weights.
        mask = nn.make_causal_mask(jnp.ones(x.shape[:-1]), dtype="bool")
        mask = jnp.broadcast_to(mask[..., :query_length, :key_length], scores.shape)
        big_neg = jnp.finfo(jnp.float32).min
        scores = jnp.where(mask, scores, big_neg)

        # Normalize the attention weights
        weights = jax.nn.softmax(scores)
        self.intermediate("weights", weights)

        # Apply the attention pattern to the value tensor.
        z = jnp.einsum("...hqk,...khd->...qhd", weights, value)
        self.intermediate("z", z)

        # Apply a linear transformation to the attention output.
        merged_z = self._merge_heads(z)
        output = init_dense(name="c_proj", features=self.model_dim)(merged_z)
        return output

    def _qkv_intermediates(self, qkv: Tuple[Array, Array, Array]) -> bool:
        ret_vals = []
        for name, value in zip(("query", "key", "value"), qkv):
            ret_vals.append(self.intermediate(name, value))

        return all(ret_vals)

    def _split_outputs(self, states: Array):
        return map(self._split_heads, jnp.split(states, 3, axis=-1))

    def _split_heads(self, states: Array):
        return states.reshape(states.shape[:-1] + (self.num_heads, self.head_dim))

    def _merge_heads(self, states: Array):
        return states.reshape(states.shape[:-2] + (self.model_dim,))
