from jax.config import config


config.update("jax_enable_x64", True)


import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from jaxtyping import Array, PyTree
from typing import Sequence

import pytest
from pytest_mock import MockerFixture

import jax.random as jr
import jax.numpy as jnp

from tx.modules import MultiHeadAttention, LayerNorm
from tx.hooks import store_hook


@pytest.fixture
def rng():
    return jr.PRNGKey(0)


@pytest.fixture
def multi_head_attention() -> MultiHeadAttention:
    return MultiHeadAttention(
        features=768,
        num_heads=12,
        head_dim=64,
    )


@pytest.fixture
def layer_norm() -> LayerNorm:
    return LayerNorm(epsilon=1e-5)


@pytest.fixture
def multi_head_attention_params(
    rng, multi_head_attention: MultiHeadAttention
) -> PyTree[Array]:
    return multi_head_attention.init(rng, jnp.ones((1024, 768)))["params"]


@pytest.fixture
def layer_norm_params(rng, layer_norm: LayerNorm) -> PyTree[Array]:
    return layer_norm.init(rng, jnp.ones((256,)))["params"]


def make_hook_fn(stub):
    def hook_fn(x: Array, **kwargs):
        stub(x)
        return x

    return hook_fn


@pytest.mark.parametrize(
    "hook_name",
    [
        "query_hook",
        "key_hook",
        "value_hook",
        "scores_hook",
        "weights_hook",
        "z_hook",
        "output_hook",
    ],
)
def test_attention_hooks_called(
    rng: Array,
    mocker: MockerFixture,
    multi_head_attention: MultiHeadAttention,
    multi_head_attention_params: PyTree[Array],
    hook_name: str,
):
    stub = mocker.stub()
    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (1024, 768))
    hooks = {hook_name: make_hook_fn(stub)}
    multi_head_attention.apply(variables, inputs, hooks)

    stub.assert_called_once()


@pytest.mark.parametrize("hook_name", ["std_hook", "normalized_hook"])
def test_layer_norm_hooks_called(
    rng: Array,
    mocker: MockerFixture,
    layer_norm: LayerNorm,
    layer_norm_params: PyTree[Array],
    hook_name: str,
):
    stub = mocker.stub()
    variables = {"params": layer_norm_params}
    inputs = jr.uniform(rng, (256,))
    hooks = {hook_name: make_hook_fn(stub)}
    layer_norm.apply(variables, inputs, hooks)

    stub.assert_called_once()


SEQ_LENGTH = 256


def format_ids(val):
    if isinstance(val, str):
        return val
    elif isinstance(val, tuple):
        return val[0]
    else:
        return None


@pytest.mark.parametrize(
    "hook_name,expected",
    [
        ("query_hook", (SEQ_LENGTH, 12, 64)),
        ("key_hook", (SEQ_LENGTH, 12, 64)),
        ("value_hook", (SEQ_LENGTH, 12, 64)),
        ("scores_hook", (12, SEQ_LENGTH, SEQ_LENGTH)),
        ("weights_hook", (12, SEQ_LENGTH, SEQ_LENGTH)),
        ("z_hook", (SEQ_LENGTH, 12, 64)),
        ("output_hook", (SEQ_LENGTH, 768)),
    ],
    ids=format_ids,
)
def test_attention_hooks_can_store_values(
    rng: Array,
    multi_head_attention: MultiHeadAttention,
    multi_head_attention_params: PyTree[Array],
    hook_name: str,
    expected: Sequence[int],
):
    hooks = {hook_name: store_hook}

    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (SEQ_LENGTH, 768))
    _, state = multi_head_attention.apply(
        variables, inputs, hooks, mutable=["intermediates"]
    )

    assert state["intermediates"][hook_name][0].shape == expected


@pytest.mark.parametrize(
    "hook_name,expected",
    [
        ("std_hook", (SEQ_LENGTH, SEQ_LENGTH)),
        ("normalized_hook", (SEQ_LENGTH, SEQ_LENGTH)),
    ],
    ids=format_ids,
)
def test_layer_norm_hooks_can_store_values(
    rng: Array,
    layer_norm: LayerNorm,
    layer_norm_params: PyTree[Array],
    hook_name: str,
    expected: Sequence[int],
):
    hooks = {hook_name: store_hook}

    variables = {"params": layer_norm_params}
    inputs = jr.uniform(rng, (SEQ_LENGTH, SEQ_LENGTH))
    _, state = layer_norm.apply(variables, inputs, hooks, mutable=["intermediates"])

    assert state["intermediates"][hook_name][0].shape == expected


if __name__ == "__main__":
    pytest.main([__file__])
