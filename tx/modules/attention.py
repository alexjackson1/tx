from functools import partial
from jaxtyping import Array, Float, Bool
from typing import NamedTuple, Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import flax.linen as nn

from .hooks import HookMap, HookPoint, apply_hooks


def apply_causal_mask(
    mask: Bool[Array, "S S"], scores: Float[Array, "... NH QL KL"], offset: int = 0
) -> Float[Array, "... NH QL KL"]:
    """Applies a causal mask to the attention scores (pre-normalisation).

    Args:
        mask: A boolean array indicating which elements to mask.
        scores: The attention scores.
        offset: The offset of the mask.

    Returns:
        The masked attention scores.
    """
    big_neg = jnp.finfo(scores.dtype).min
    return jnp.where(mask, scores, big_neg)


def combine_masks(*masks: Optional[Array], dtype: jnp.dtype = jnp.float32) -> Array:
    """Combine attention masks.

    Args:
      *masks: set of attention mask arguments to combine, some can be None.
      dtype: dtype for the returned mask.

    Returns:
      Combined mask, reduced by logical and, returns None if no masks given.
    """
    masks_list = [m for m in masks if m is not None]
    if not masks_list:
        return None

    if len(masks_list) == 1:
        return masks_list[0].astype(dtype)

    if not all(map(lambda x: x.shape == masks_list[0].shape, masks_list)):
        shapes = tuple(map(lambda x: x.shape, masks_list))
        raise ValueError(f"masks must have same shape: {shapes}")

    mask, *other_masks = masks_list
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)

    return mask.astype(dtype)


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
        decode: Whether to cache the key and value arrays.
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
    """Whether to cache the key and value arrays."""
    dtype: Optional[jnp.dtype] = None
    """The dtype of the input array."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    def init_cache(
        self,
        key: Float[Array, "... S NH HD"],
        value: Float[Array, "... S NH HD"],
    ) -> Tuple[bool, KeyValueCache]:
        names = lambda x: ("cache", f"cached_{x}")
        is_initialized = self.has_variable(*names("key"))
        max_length = self.context_length
        key_shape = key.shape[:-3] + (max_length, self.num_heads, self.head_dim)
        value_shape = value.shape[:-3] + (max_length, self.num_heads, self.head_dim)
        c_key = self.variable(*names("key"), jnp.zeros, key_shape, key.dtype)
        c_val = self.variable(*names("value"), jnp.zeros, value_shape, value.dtype)
        c_idx = self.variable(*names("idx"), lambda: jnp.array(0, dtype=jnp.int64))
        return is_initialized, KeyValueCache(c_key, c_val, c_idx)

    @nn.compact
    def __call__(
        self,
        states: Float[Array, "... S F"],
        mask: Bool[Array, "... S"],
        hooks: Optional[HookMap] = None,
    ) -> Float[Array, "... S F"]:
        """Applies the multi-headed attention module to the input array.

        Args:
            states: The input array.
            mask: A boolean array indicating which elements to mask.
            hooks: A dictionary of hooks to apply to the attention outputs.
            cache_entry: A cache entry containing the previous key and value
                arrays.

        Returns:
            The attention output.
        """

        # TODO: Make this implementation immutable.
        dtype = self.dtype or jnp.result_type(states)
        states = jnp.asarray(states, dtype)

        init_dense = partial(
            nn.DenseGeneral,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
            use_bias=self.use_bias,
            dtype=dtype,
            param_dtype=self.param_dtype,
        )

        # Apply a linear transformation to the input tensor.
        hidden_states = init_dense(name="c_attn", features=3 * self.features)(states)

        # Split the hidden states into query, key, and value.
        qkv_states = self._split_outputs(hidden_states)
        query, key, value = self._apply_qkv_hooks(qkv_states, hooks)

        # If using a cache, append the new keys and values to the cached values.
        if self.decode:
            is_initialized, cache = self.init_cache(key, value)
            if is_initialized:
                (*batch_dims, max_length, num_heads, head_dim) = cache.key.value.shape

                # shape check of cached keys against query input
                # expected_shape = tuple(batch_dims) + (1, num_heads, head_dim)
                # if expected_shape != query.shape:
                #     raise ValueError(
                #         "Autoregressive cache shape error, "
                #         "expected query shape %s instead got %s."
                #         % (expected_shape, query.shape)
                #     )

                # update key, value caches with our new 1d spatial slices
                cur_index = cache.index.value
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cache.key.value, key, indices)
                value = lax.dynamic_update_slice(cache.value.value, value, indices)
                cache.key.value = key
                cache.value.value = value
                cache.index.value = cache.index.value + 1

                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                mask = combine_masks(
                    jnp.tril(
                        jnp.ones((1, 1, max_length), dtype="bool"), k=int(cur_index)
                    ),
                    jnp.broadcast_to(
                        jnp.arange(max_length) <= cur_index,
                        tuple(batch_dims) + (1, 1, max_length),
                    ),
                )

        # Compute the attention weights.
        query = query / jnp.sqrt(query.shape[-1])
        scores = jnp.einsum("...qhd,...khd->...hqk", query, key)

        # Apply the causal mask to the attention weights.
        scores = apply_causal_mask(mask, scores, 0)
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
