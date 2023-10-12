from enum import Enum
from jaxtyping import Array, PyTree
from typing import Any, Callable, Dict

import flax.linen as nn


HookFn = Callable[[Array, Dict[str, Any]], Array]
"""A function that applies a hook to an array."""


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


def apply_hooks(
    hook_point: HookPoint, hooks: PyTree[HookFn], x: Array, **kwargs
) -> Array:
    """Applies a hook to the given array."""
    if hooks is not None and hook_point.value in hooks:
        x = hooks[hook_point.value](x, hook_point=hook_point, **kwargs)
    return x


def compose_hooks(*hooks: HookFn) -> HookFn:
    """Composes multiple hooks into a single hook."""

    def new_hook(
        x: Array, hook_point: HookPoint = None, module: nn.Module = None, **kwargs
    ) -> Array:
        for hook in hooks:
            x = hook(x, hook_point=hook_point, module=module, **kwargs)
        return x

    return new_hook
