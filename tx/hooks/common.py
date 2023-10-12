from jaxtyping import Array, PyTree
from typing import Any, Callable, Dict, Optional

import flax.linen as nn

from tx.tree_util import KeyPath, tree_contains_path

HookFn = Callable[[Array, Dict[str, Any]], Array]
"""A function that applies a hook to an array."""


def apply_hooks(
    path: KeyPath,
    hooks: Optional[PyTree[HookFn]],
    x: Array,
    **kwargs,
) -> Array:
    """Applies a hook to the given array."""
    if hooks is not None and tree_contains_path(hooks, path):
        sub_tree = hooks
        for key in path:
            sub_tree = sub_tree[key]

        assert callable(sub_tree), f"Hook at path {path} is not callable"
        x = sub_tree(x, path=path, **kwargs)
    return x


def compose_hooks(*hooks: HookFn) -> HookFn:
    """Composes multiple hooks into a single hook."""

    def new_hook(x: Array, **kwargs) -> Array:
        for hook in hooks:
            x = hook(x, **kwargs)
        return x

    return new_hook


def store_hook(
    x: Array, path: KeyPath = None, module: nn.Module = None, **kwargs
) -> Array:
    """Stores the given array in the given module."""
    assert module is not None, "Module must be defined"
    assert path is not None, "Key path must be defined"
    hook_name = path[-1]
    module.sow("intermediates", hook_name, x)
    return x
