from typing import Any, Callable
from jaxtyping import Array, PyTree

import jax

Params = PyTree[Array]


def compose(f: Callable[..., Any], *pytrees: PyTree[Any]) -> PyTree[Any]:
    """Composes a list of pytrees using a function.

    Args:
        f: A function that takes a sequence of leaves and returns a new leaf.
        *pytrees: A sequence of pytrees.

    Returns:
        A new pytree.
    """

    if len(pytrees) < 2:
        return pytrees[0]

    # Get the tree structure of the first pytree
    first_tree_structure = jax.tree_structure(pytrees[0])

    # Compare the tree structure of the first pytree with the rest
    for i, pytree in enumerate(pytrees[1:], start=2):
        if jax.tree_structure(pytree) != first_tree_structure:
            raise ValueError(
                f"The shape of pytree {i} does not match the shape of the first pytree."
            )

    # Combine the leaves using the specified combination function
    return jax.tree_util.tree_map(f, pytrees[0], *pytrees[1:])
