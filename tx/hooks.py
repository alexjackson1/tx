from enum import Enum
from jaxtyping import Array
from typing import Callable, NamedTuple, Optional, TypedDict

import flax.linen as nn


HookFn = Callable[[nn.Module, Array], Array]
"""A function that applies a hook to an array."""


class Hook(NamedTuple):
    """A hook that can be applied to an activation."""

    apply: HookFn


def store_hook(
    x: Array,
    hook_point: "HookPoint" = None,
    module: nn.Module = None,
    **kwargs,
) -> Array:
    """Stores the given array in the given module."""
    assert module is not None, "Module must be defined"
    assert hook_point is not None, "Hook point must be defined"
    module.sow("intermediates", hook_point.value, x)
    return x


StoreHook = Hook(store_hook)


class HookPoint(Enum):
    """The points at which hooks can be applied."""

    # Transformer
    EMBED = "embed_hook"
    """The output of the embedding layer."""
    POS_EMBED = "pos_embed_hook"
    """The output of the positional embedding layer."""
    RESIDUAL = "residual_hook"
    """The residual connection."""
    FINAL_OUTPUT = "final_output_hook"
    """The final output of the model (prior to unembedding)."""

    # Layer Norm
    LN_STD = "std_hook"
    """The standard deviation of the layer norm input."""
    LN_NORMALIZED = "normalized_hook"
    """The normalized layer norm input."""

    # Multi-Head Attention
    ATTN_QUERY = "query_hook"
    """The attention query."""
    ATTN_KEY = "key_hook"
    """The attention key."""
    ATTN_VALUE = "value_hook"
    """The attention value."""
    ATTN_SCORES = "scores_hook"
    """The attention scores."""
    ATTN_WEIGHTS = "weights_hook"
    """The attention weights."""
    ATTN_Z = "z_hook"
    """The attention weights indexed with values"""
    ATTN_OUTPUT = "output_hook"
    """The output of the attention block at each layer."""

    # MLP
    MLP_PRE_ACTIVATION = "pre_activation_hook"
    """The output of the MLP before the activation function."""
    MLP_POST_ACTIVATION = "post_activation_hook"
    """The output of the MLP after the activation function."""


class HookMap(TypedDict):
    """A mapping of hook points to hooks."""

    embed: Optional[Hook]
    """A hook applied to the output of the embedding layer."""
    pos_embed: Optional[Hook]
    """A hook applied to the output of the positional embedding layer."""
    residual: Optional[Hook]
    """A hook applied to the residual connection."""
    final_output: Optional[Hook]
    """A hook applied to the final output of the model (prior to unembedding)."""
    ln_std: Optional[Hook]
    """A hook applied to the standard deviation of the layer norm input."""
    ln_normalized: Optional[Hook]
    """A hook applied to the normalized layer norm input."""
    attn_query: Optional[Hook]
    """A hook applied to the attention query."""
    attn_key: Optional[Hook]
    """A hook applied to the attention key."""
    attn_value: Optional[Hook]
    """A hook applied to the attention value."""
    attn_scores: Optional[Hook]
    """A hook applied to the attention scores."""
    attn_weights: Optional[Hook]
    """A hook applied to the attention weights."""
    attn_z: Optional[Hook]
    """A hook applied to the attention weights indexed with values."""
    attn_output: Optional[Hook]
    """A hook applied to the output of the attention block at each layer."""
    mlp_pre_activation: Optional[Hook]
    """A hook applied to the output of the MLP before the activation function."""
    mlp_post_activation: Optional[Hook]
    """A hook applied to the output of the MLP after the activation function."""


def apply_hooks(
    hook_point: HookPoint, hooks: Optional[HookMap], x: Array, **kwargs
) -> Array:
    """Applies a hook to the given array."""
    if hooks is not None and hook_point.value in hooks:
        x = hooks[hook_point.value].apply(x, hook_point=hook_point, **kwargs)
    return x
