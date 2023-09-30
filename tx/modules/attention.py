from typing import List, Tuple
from jaxtyping import Array, Float

from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.struct as struct


class MultiHeadAttention(nn.Module):
    """Multi-headed attention module.

    Attributes:
        num_heads: The number of attention heads.
        head_dim: The dimensionality of each attention head.
        features: The dimensionality of the linear models.
        init_range: The standard deviation of the normal distribution used to
            initialize the linear transformations.
        use_bias: Whether to include a bias term in the linear transformations.
        intermediates: A list of intermediate arrays to store in the module's
            state dictionary.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, seq_len, features).
    2. Output shape: (...batch_dimensions, sequence_length, model_dim).

    Transformation Steps:
    1. Applies a linear transformation to the input array.
    2. Splits the transformed array into `query`, `key`, and `value` arrays.
    3. Computes the attention weights using the `query` and `key` arrays.
    4. Applies the attention pattern to the `value` array.
    5. Applies a linear transformation to the attention output.

    Intermediate Arrays:
    1. `query`: The query array (end of step 2).
    2. `key`: The key array (end of step 2).
    3. `value`: The value array (end of step 2).
    4. `scores`: The attention weights (step 3, before normalisation).
    5. `weights`: The normalized attention weights (end of step 3).
    6. `z`: The attention output (end of step 4).

    Notes:
     - The `features` attribute defines the dimensions of all linear
     transformations (i.e.`query`, `key`, `value`, and `output`).

    References:
     - [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
     - [Flax Linen Attention Module](https://flax.readthedocs.io/en/v0.5.3/_modules/flax/linen/attention.html).
    """

    features: int
    """The dimensionality of the linear models."""
    num_heads: int
    """The number of attention heads."""
    head_dim: int
    """The dimensionality of each attention head."""
    init_range: float = 0.02
    """The standard deviation of the normal distribution used to initialize the
    linear transformations."""
    use_bias: bool = True
    """Whether to include a bias term in the linear transformations."""

    intermediates: List[str] = struct.field(default_factory=list)
    """A list of intermediate arrays to store in the module's state dictionary."""

    def intermediate(self, name: str, value: Array) -> bool:
        """Stores an intermediate array in the module's state dictionary.

        Args:
            name: The name of the intermediate array.
            value: The intermediate array.

        Returns:
            Whether the intermediate array was successfully stored.
        """
        if name in self.intermediates:
            return self.sow("intermediates", name, value)
        return False

    # TODO: Consider taking the mask as an argument.
    @nn.compact
    def __call__(self, x: Float[Array, "... S F"]) -> Float[Array, "... S F"]:
        """Applies the multi-headed attention module to the input array."""
        init_dense = partial(
            nn.DenseGeneral,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
            use_bias=self.use_bias,
        )

        # Apply a linear transformation to the input tensor.
        hidden_states = init_dense(name="c_attn", features=3 * self.features)(x)

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
        mask = mask[..., :query_length, :key_length]
        big_neg = jnp.finfo(jnp.float64).min
        scores = jnp.where(mask, scores, big_neg)

        # Normalise the attention weights
        weights: Array = jax.nn.softmax(scores)
        self.intermediate("weights", weights)

        # Apply the attention pattern to the value tensor.
        z = jnp.einsum("...hqk,...khd->...qhd", weights, value)
        self.intermediate("z", z)

        # Apply a linear transformation to the attention output.
        merged_z = self._merge_heads(z)
        output = init_dense(name="c_proj", features=self.features)(merged_z)
        return output

    def _qkv_intermediates(self, qkv: Tuple[Array, Array, Array]) -> bool:
        """Stores the query, key, and value arrays in the module's state dictionary."""
        ret_vals = []
        for name, value in zip(("query", "key", "value"), qkv):
            ret_vals.append(self.intermediate(name, value))

        return all(ret_vals)

    def _split_outputs(self, states: Array):
        """Splits the hidden states into query, key, and value."""
        return map(self._split_heads, jnp.split(states, 3, axis=-1))

    def _split_heads(self, states: Array):
        """Splits the hidden states into attention heads."""
        return states.reshape(states.shape[:-1] + (self.num_heads, self.head_dim))

    def _merge_heads(self, states: Array):
        """Merges the attention heads into hidden states."""
        return states.reshape(states.shape[:-2] + (self.features,))
