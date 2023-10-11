from jaxtyping import Array
import flax.linen as nn
from .common import Hook, HookMap, HookPoint


def store_hook(
    x: Array, hook_point: HookPoint = None, module: nn.Module = None, **kwargs
) -> Array:
    """Stores the given array in the given module."""
    assert module is not None, "Module must be defined"
    assert hook_point is not None, "Hook point must be defined"
    module.sow("intermediates", hook_point.value, x)
    return x


StoreHook = Hook(store_hook)
"""A hook that stores the given array as an intermediate value."""

CacheAll: HookMap = {
    HookPoint.EMBED.value: StoreHook,
    HookPoint.POS_EMBED.value: StoreHook,
    HookPoint.RESIDUAL.value: StoreHook,
    HookPoint.FINAL_OUTPUT.value: StoreHook,
    HookPoint.LN_STD.value: StoreHook,
    HookPoint.LN_NORMALIZED.value: StoreHook,
    HookPoint.ATTN_QUERY.value: StoreHook,
    HookPoint.ATTN_KEY.value: StoreHook,
    HookPoint.ATTN_VALUE.value: StoreHook,
    HookPoint.ATTN_SCORES.value: StoreHook,
    HookPoint.ATTN_WEIGHTS.value: StoreHook,
    HookPoint.ATTN_Z.value: StoreHook,
    HookPoint.ATTN_OUTPUT.value: StoreHook,
    HookPoint.MLP_PRE_ACTIVATION.value: StoreHook,
    HookPoint.MLP_POST_ACTIVATION.value: StoreHook,
}
"""A hook map that stores all intermediate values."""
