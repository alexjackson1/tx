from jaxtyping import Array, Float, Bool, PyTree
from typing import NamedTuple, Optional, Sequence, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import flax.linen as nn

from ..hooks import HookPoint, apply_hooks, HookFn


def apply_mask(
    mask: Bool[Array, "... 1 QL KL"], scores: Float[Array, "... NH QL KL"]
) -> Float[Array, "... NH QL KL"]:
    """Applies a causal mask to the attention scores (pre-normalisation).

    Args:
        mask: A boolean array indicating which elements to mask.
        scores: The attention scores.
        offset: The offset of the mask.

    Returns:
        The masked attention scores.
    """
    return jnp.where(mask, scores, jnp.finfo(scores.dtype).min)


class KeyValueCache(NamedTuple):
    """A cache entry containing the previous key and value arrays.

    Attributes:
        key: The previous key array.
        value: The previous value array.
        index: The current index.
    """

    key: nn.Variable[Float[Array, "... S NH HD"]]
    """The previous key array."""
    value: nn.Variable[Float[Array, "... S NH HD"]]
    """The previous value array."""
    index: nn.Variable[int]
    """The current index."""


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
    context_length: int = 1024
    """The length of the context."""
    init_range: float = 0.02
    """The standard deviation of the normal distribution used to initialize the
    linear transformations."""
    use_bias: bool = True
    """Whether to include a bias term in the linear transformations."""
    decode: bool = False
    """Whether to run in autoregressive mode."""
    dtype: Optional[jnp.dtype] = None
    """The dtype of the input array."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    @nn.nowrap
    def qkv_dense(self, name: str) -> nn.DenseGeneral:
        return nn.DenseGeneral(
            name=name,
            features=(self.num_heads, self.head_dim),
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    @nn.nowrap
    def proj_dense(self, name: str) -> nn.DenseGeneral:
        return nn.DenseGeneral(
            name=name,
            features=self.features,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    @nn.compact
    def __call__(
        self, states: Float[Array, "... S F"], hooks: Optional[PyTree[HookFn]] = None
    ) -> Float[Array, "... S F"]:
        """Applies the multi-headed attention module to the input array.

        Args:
            states: The input array.
            hooks: A dictionary of hooks to apply to the attention outputs.

        Returns:
            The attention output.
        """

        dtype = self.dtype or jnp.result_type(states)
        q_inputs, kv_inputs = jnp.asarray(states, dtype), jnp.asarray(states, dtype)
        batch_dims = states.shape[:-2]

        # Initialise/load the cache.
        cache_initialized = self.has_variable("cache", "cache_index")
        if self.decode:
            cache = self.init_cache(states.shape[:-2])
            cache_empty = cache.index.value == 0

            if cache_initialized and not cache_empty:
                # Just take the last token from the key/value arrays.
                kv_inputs = kv_inputs.take(-1, axis=-2)
                kv_inputs = jnp.expand_dims(kv_inputs, axis=-2)

        # Create a causal mask for the attention weights.
        mask = nn.make_causal_mask(
            jnp.ones((self.context_length,), dtype="bool"),
            extra_batch_dims=len(batch_dims),
            dtype="bool",
        )

        # Apply a linear transformation to the input array(s).
        qkv_states = (
            self.qkv_dense(name="query")(q_inputs),
            self.qkv_dense(name="key")(kv_inputs),
            self.qkv_dense(name="value")(kv_inputs),
        )
        query, key, value = self._apply_qkv_hooks(qkv_states, hooks)
        query_length, key_length = query.shape[-3], key.shape[-3]

        # Apply the cache to the key and value arrays.
        if self.decode and cache_initialized:
            key, value, causal_mask = self.apply_cache(
                cache, kv_inputs, key, value, mask
            )
        else:
            causal_mask = mask[..., :query_length, :key_length]

        # Compute the attention weights.
        query = query / jnp.sqrt(query.shape[-1])
        scores = jnp.einsum("...qhd,...khd->...hqk", query, key)

        # Apply the causal mask to the attention weights.
        scores = apply_mask(causal_mask, scores)
        scores = self.apply_hooks(HookPoint.ATTN_SCORES, hooks, scores)

        # Normalise the attention weights
        weights: Array = jax.nn.softmax(scores)
        weights = self.apply_hooks(HookPoint.ATTN_WEIGHTS, hooks, weights)

        # Apply the attention pattern to the value tensor.
        z = jnp.einsum("...hqk,...khd->...qhd", weights, value)
        z = self.apply_hooks(HookPoint.ATTN_Z, hooks, z)

        # Apply a linear transformation to the attention output.
        merged_z = self._merge_heads(z)
        output = self.proj_dense(name="c_proj")(merged_z)
        output = self.apply_hooks(HookPoint.ATTN_OUTPUT, hooks, output)
        return output

    def init_cache(self, batch_dims: Sequence[int] = ()) -> KeyValueCache:
        """Initialises the cache."""
        shape = (*batch_dims, self.context_length, self.num_heads, self.head_dim)
        key = self.variable("cache", "cached_key", jnp.zeros, shape, self.dtype)
        value = self.variable("cache", "cached_value", jnp.zeros, shape, self.dtype)
        zero_init = lambda: jnp.array(0, dtype=jnp.int32)
        index = self.variable("cache", "cache_index", zero_init)
        return KeyValueCache(key, value, index)

    def apply_cache(
        self,
        cache: KeyValueCache,
        kv_inputs: Float[Array, "... S F"],
        key: Float[Array, "... 1 S NH HD"],
        value: Float[Array, "... 1 S NH HD"],
        mask: Bool[Array, "... 1 CL CL"],
    ) -> Tuple[
        Float[Array, "... CL NH HD"],
        Float[Array, "... CL NH HD"],
        Float[Array, "... 1 S CL"],
    ]:
        """Applies the cache to the key and value arrays."""
        batch_dims = kv_inputs.shape[:-2]
        cache_index = cache.index
        cached_key, cached_value = cache.key, cache.value

        # Create causal mask based on the cache index.
        batch_zeros = (0,) * len(batch_dims)
        batch_ones = (1,) * len(batch_dims)
        indices = (*batch_zeros, 0, cache_index.value, 0)
        sizes = (*batch_ones, 1, kv_inputs.shape[-2], self.context_length)

        causal_mask = lax.dynamic_slice(
            mask,
            jnp.array(indices, dtype=jnp.int32),
            jnp.array(sizes, dtype=jnp.int32),
        )

        # Update the cache.
        indices = jnp.array((*batch_zeros, cache_index.value, 0, 0), dtype=jnp.int32)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value, cached_value.value = key, value
        cache_index.value = cache_index.value + kv_inputs.shape[-2]

        return key, value, causal_mask

    def apply_hooks(
        self, hook_point: HookPoint, hooks: PyTree[HookFn], x: Array, **kwargs
    ) -> Array:
        return apply_hooks(hook_point, hooks, x, module=self, **kwargs)

    def _apply_qkv_hooks(
        self, qkv: Tuple[Array, Array, Array], hooks: Optional[PyTree[HookFn]]
    ) -> Tuple[Array, Array, Array]:
        """Stores the query, key, and value arrays in the module's state dictionary."""
        ret_vals = []
        hook_points = (HookPoint.ATTN_QUERY, HookPoint.ATTN_KEY, HookPoint.ATTN_VALUE)
        for hook_point, array in zip(hook_points, qkv):
            ret_vals.append(self.apply_hooks(hook_point, hooks, array))

        return ret_vals

    def _merge_heads(self, states: Array):
        """Merges the attention heads into hidden states."""
        return states.reshape(states.shape[:-2] + (self.features,))
