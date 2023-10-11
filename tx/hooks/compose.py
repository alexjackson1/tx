from jaxtyping import Array
from typing import Union
import flax.linen as nn
from .common import Hook, HookFn, HookMap, HookPoint


def compose_hook_fns(*hooks: HookFn) -> HookFn:
    """Composes multiple hooks into a single hook."""

    def composed_fn(
        x: Array, hook_point: HookPoint = None, module: nn.Module = None, **kwargs
    ) -> Array:
        for hook in hooks:
            x = hook(x, hook_point=hook_point, module=module, **kwargs)
        return x

    return composed_fn


def compose_hooks(*hooks: Hook) -> Hook:
    """Composes multiple hooks into a single hook."""
    return Hook(compose_hook_fns(*[hook.apply for hook in hooks]))


def compose_hook_maps(*hook_maps: "HookMap") -> "HookMap":
    """Composes multiple hook maps into a single hook map."""
    return {
        key: compose_hooks(
            *[hook_map[key] for hook_map in hook_maps if key in hook_map]
        )
        for key in HookPoint
    }


def compose(*hooks: Union[HookFn, Hook, "HookMap"]) -> Union[HookFn, Hook, "HookMap"]:
    """Composes multiple hooks into a single hook."""
    if all(isinstance(hook, Hook) for hook in hooks):
        return compose_hooks(*hooks)
    elif all(isinstance(hook, dict) for hook in hooks):
        return compose_hook_maps(*hooks)
    elif all(callable(hook) for hook in hooks):
        return compose_hook_fns(*hooks)
    else:
        raise ValueError("Hooks must all be of the same type")
