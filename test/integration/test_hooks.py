from jax.config import config


config.update("jax_enable_x64", True)


import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from jaxtyping import Array
from typing import Sequence

import pytest
from pytest_mock import MockerFixture

import jax.random as jr
import jax.numpy as jnp

from tx.modules import MultiHeadAttention, LayerNorm
from tx.models.gpt2 import GPT2Config, GPT2Transformer, GPT2MLP
from tx.hooks import store_hook
from tx.tree_util import KeyPath, Params


@pytest.fixture
def rng():
    return jr.PRNGKey(0)


@pytest.fixture
def config():
    return GPT2Config()


@pytest.fixture
def transformer(config: GPT2Config) -> GPT2Transformer:
    return GPT2Transformer.from_config(config)


@pytest.fixture
def mlp(config: GPT2Config) -> GPT2MLP:
    return GPT2MLP(features=[config.mlp_dim, config.model_dim])


@pytest.fixture
def multi_head_attention(config: GPT2Config) -> MultiHeadAttention:
    return MultiHeadAttention(
        features=config.model_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
    )


@pytest.fixture
def layer_norm(config: GPT2Config) -> LayerNorm:
    return LayerNorm(epsilon=config.layer_norm_eps)


@pytest.fixture
def transformer_params(rng, transformer: GPT2Transformer) -> Params:
    return transformer.init(rng, jnp.ones((256,), jnp.int32))["params"]


@pytest.fixture
def mlp_params(rng, mlp: GPT2MLP) -> Params:
    return mlp.init(rng, jnp.ones((256,)))["params"]


@pytest.fixture
def multi_head_attention_params(
    rng, multi_head_attention: MultiHeadAttention
) -> Params:
    return multi_head_attention.init(rng, jnp.ones((1024, 768)))["params"]


@pytest.fixture
def layer_norm_params(rng, layer_norm: LayerNorm) -> Params:
    return layer_norm.init(rng, jnp.ones((256,)))["params"]


def make_hook_fn(stub):
    def hook_fn(x: Array, **kwargs):
        stub(x)
        return x

    return hook_fn


@pytest.mark.parametrize(
    "hook_name", ["embed_hook", "pos_embed_hook", "residual_hook", "output_hook"]
)
def test_transformer_hooks_called(
    mocker: MockerFixture,
    transformer: GPT2Transformer,
    transformer_params: Params,
    hook_name: str,
):
    stub = mocker.stub()
    variables = {"params": transformer_params}
    inputs = jnp.ones((256,), jnp.int32)
    hooks = {hook_name: make_hook_fn(stub)}

    transformer.apply(variables, inputs, hooks)

    if hook_name != "residual_hook":
        stub.assert_called_once()
    else:
        stub.assert_called()


@pytest.mark.parametrize("hook_name", ["pre_activation_hook", "post_activation_hook"])
def test_mlp_hooks_called(
    rng: Array,
    mocker: MockerFixture,
    mlp: GPT2MLP,
    mlp_params: Params,
    hook_name: str,
):
    stub = mocker.stub()
    variables = {"params": mlp_params}
    inputs = jr.uniform(rng, (256, 256))
    hooks = {hook_name: make_hook_fn(stub)}
    mlp.apply(variables, inputs, hooks)

    stub.assert_called_once()


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
    multi_head_attention_params: Params,
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
    layer_norm_params: Params,
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
        ("embed_hook", (SEQ_LENGTH, 768)),
        ("pos_embed_hook", (SEQ_LENGTH, 768)),
        ("residual_hook", (SEQ_LENGTH, 768)),
        ("output_hook", (SEQ_LENGTH, 768)),
    ],
    ids=format_ids,
)
def test_transformer_hooks_can_store_values(
    transformer: GPT2Transformer,
    transformer_params: Params,
    hook_name: str,
    expected: Sequence[int],
):
    hooks = {hook_name: store_hook}

    variables = {"params": transformer_params}
    inputs = jnp.ones((SEQ_LENGTH,), jnp.int32)
    _, state = transformer.apply(variables, inputs, hooks, mutable=["intermediates"])

    assert state["intermediates"][hook_name][0].shape == expected


@pytest.mark.parametrize(
    "hook_name,expected",
    [
        ("pre_activation_hook", (SEQ_LENGTH, 3072)),
        ("post_activation_hook", (SEQ_LENGTH, 3072)),
    ],
    ids=format_ids,
)
def test_mlp_hooks_can_store_values(
    rng: Array,
    mlp: GPT2MLP,
    mlp_params: Params,
    hook_name: str,
    expected: Sequence[int],
):
    hooks = {hook_name: store_hook}

    variables = {"params": mlp_params}
    inputs = jr.uniform(rng, (256, 256))
    _, state = mlp.apply(variables, inputs, hooks, mutable=["intermediates"])

    assert state["intermediates"][hook_name][0].shape == expected


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
    multi_head_attention_params: Params,
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
    layer_norm_params: Params,
    hook_name: str,
    expected: Sequence[int],
):
    hooks = {hook_name: store_hook}

    variables = {"params": layer_norm_params}
    inputs = jr.uniform(rng, (SEQ_LENGTH, SEQ_LENGTH))
    _, state = layer_norm.apply(variables, inputs, hooks, mutable=["intermediates"])

    assert state["intermediates"][hook_name][0].shape == expected


def format_nested_ids(val):
    if isinstance(val, tuple):
        if isinstance(val[0], str):
            return "/".join(val)

        if isinstance(val[0], int):
            return ",".join([str(v) for v in val])

    return val


@pytest.mark.parametrize(
    "hook_path,expected",
    [
        (("block_0", "mlp", "pre_activation_hook"), (SEQ_LENGTH, 3072)),
        (("block_4", "mlp", "post_activation_hook"), (SEQ_LENGTH, 3072)),
        (("block_1", "attn", "query_hook"), (SEQ_LENGTH, 12, 64)),
        (("block_2", "attn", "key_hook"), (SEQ_LENGTH, 12, 64)),
        (("block_3", "attn", "value_hook"), (SEQ_LENGTH, 12, 64)),
        (("block_4", "attn", "scores_hook"), (12, SEQ_LENGTH, SEQ_LENGTH)),
        (("block_5", "attn", "weights_hook"), (12, SEQ_LENGTH, SEQ_LENGTH)),
        (("block_6", "attn", "z_hook"), (SEQ_LENGTH, 12, 64)),
        (("block_7", "attn", "output_hook"), (SEQ_LENGTH, 768)),
        (("block_0", "ln_1", "std_hook"), (SEQ_LENGTH, 768)),
        (("block_1", "ln_2", "normalized_hook"), (SEQ_LENGTH, 768)),
        (("ln_f", "std_hook"), (SEQ_LENGTH, 768)),
        (("ln_f", "normalized_hook"), (SEQ_LENGTH, 768)),
    ],
    ids=format_nested_ids,
)
def test_nested_transformer_hooks_are_called(
    transformer: GPT2Transformer,
    transformer_params: Params,
    hook_path: KeyPath,
    expected: Sequence[int],
):
    hooks = temp_hooks = {}
    for key in hook_path[:-1]:
        temp_hooks[key] = {}
        temp_hooks = temp_hooks[key]

    temp_hooks[hook_path[-1]] = store_hook

    variables = {"params": transformer_params}
    inputs = jnp.ones((SEQ_LENGTH,), jnp.int32)
    _, state = transformer.apply(variables, inputs, hooks, mutable=["intermediates"])

    state_ = state["intermediates"]
    for key in hook_path:
        state_ = state_[key]

    assert state_[0].shape == expected


if __name__ == "__main__":
    pytest.main([__file__])
