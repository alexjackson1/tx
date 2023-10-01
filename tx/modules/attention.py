from typing import Optional, Tuple
from jaxtyping import Array, Float, Bool

from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

from tx.modules.hooks import HookMap, HookPoint, apply_hooks


class MultiHeadAttention(nn.Module):
    """Multi-headed attention module.

    Attributes:
        num_heads: The number of attention heads.
        head_dim: The dimensionality of each attention head.
        features: The dimensionality of the linear models.
        init_range: The standard deviation of the normal distribution used to
            initialize the linear transformations.
        use_bias: Whether to include a bias term in the linear transformations.
        dtype: The dtype of the input array.
        param_dtype: The dtype of the parameters.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, seq_len, features).
    2. Output shape: (...batch_dimensions, sequence_length, model_dim).

    Transformation Steps:
    1. Applies a linear transformation to the input array.
    2. Splits the transformed array into `query`, `key`, and `value` arrays.
    3. Computes the attention weights using the `query` and `key` arrays.
    4. Applies the attention pattern to the `value` array.
    5. Applies a linear transformation to the attention output.

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
    dtype: Optional[jnp.dtype] = None
    """The dtype of the input array."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    # TODO: Consider taking the mask as an argument.
    @nn.compact
    def __call__(
        self,
        x: Float[Array, "... S F"],
        mask: Bool[Array, "... S"],
        hooks: Optional[HookMap] = None,
    ) -> Float[Array, "... S F"]:
        """Applies the multi-headed attention module to the input array."""
        dtype = self.dtype or jnp.result_type(x)
        x = jnp.asarray(x, dtype)

        init_dense = partial(
            nn.DenseGeneral,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
            use_bias=self.use_bias,
            dtype=dtype,
            param_dtype=self.param_dtype,
        )

        # Apply a linear transformation to the input tensor.
        hidden_states = init_dense(name="c_attn", features=3 * self.features)(x)

        # Split the hidden states into query, key, and value.
        qkv_states = self._split_outputs(hidden_states)
        query, key, value = self._apply_qkv_hooks(qkv_states, hooks)
        query_length, key_length = query.shape[-3], key.shape[-3]

        # Compute the attention weights.
        query = query / jnp.sqrt(query.shape[-1])
        scores = jnp.einsum("...qhd,...khd->...hqk", query, key)

        # Apply the causal mask to the attention weights.
        mask = mask[..., :query_length, :key_length]
        big_neg = jnp.finfo(dtype).min
        scores = jnp.where(mask, scores, big_neg)
        scores = apply_hooks(HookPoint.ATTN_SCORES, hooks, scores)

        # Normalise the attention weights
        weights: Array = jax.nn.softmax(scores)
        weights = apply_hooks(HookPoint.ATTN_WEIGHTS, hooks, weights)

        # Apply the attention pattern to the value tensor.
        z = jnp.einsum("...hqk,...khd->...qhd", weights, value)
        z = apply_hooks(HookPoint.ATTN_Z, hooks, z)

        # Apply a linear transformation to the attention output.
        merged_z = self._merge_heads(z)
        output = init_dense(name="c_proj", features=self.features)(merged_z)
        output = apply_hooks(HookPoint.ATTN_OUTPUT, hooks, output)

        return output

    def _apply_qkv_hooks(
        self, qkv: Tuple[Array, Array, Array], hooks: Optional[HookMap]
    ) -> Tuple[Array, Array, Array]:
        """Stores the query, key, and value arrays in the module's state dictionary."""
        ret_vals = []
        hook_points = (HookPoint.ATTN_QUERY, HookPoint.ATTN_KEY, HookPoint.ATTN_VALUE)
        for hook_point, array in zip(hook_points, qkv):
            ret_vals.append(apply_hooks(hook_point, hooks, array))

        return ret_vals

    def _split_outputs(self, states: Array):
        """Splits the hidden states into query, key, and value."""
        return map(self._split_heads, jnp.split(states, 3, axis=-1))

    def _split_heads(self, states: Array):
        """Splits the hidden states into attention heads."""
        return states.reshape(states.shape[:-1] + (self.num_heads, self.head_dim))

    def _merge_heads(self, states: Array):
        """Merges the attention heads into hidden states."""
        return states.reshape(states.shape[:-2] + (self.features,))
