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
from optax import Params

from tx import TransformerConfig, Transformer, MLP, MultiHeadAttention, LayerNorm
from tx.hooks import Hook, HookPoint, StoreHook


@pytest.fixture
def rng():
    return jr.PRNGKey(0)


@pytest.fixture
def config():
    return TransformerConfig()


@pytest.fixture
def transformer(config: TransformerConfig) -> Transformer:
    return Transformer.from_config(config)


@pytest.fixture
def mlp(config: TransformerConfig) -> MLP:
    return MLP(features=[config.mlp_dim, config.model_dim])


@pytest.fixture
def multi_head_attention(config: TransformerConfig) -> MultiHeadAttention:
    return MultiHeadAttention(
        features=config.model_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
    )


@pytest.fixture
def layer_norm(config: TransformerConfig) -> LayerNorm:
    return LayerNorm(epsilon=config.layer_norm_eps)


@pytest.fixture
def transformer_params(rng, transformer: Transformer) -> Params:
    return transformer.init(rng, jnp.ones((256,), jnp.int32))["params"]


@pytest.fixture
def mlp_params(rng, mlp: MLP) -> Params:
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
    "hook_point",
    [HookPoint.EMBED, HookPoint.POS_EMBED, HookPoint.RESIDUAL, HookPoint.FINAL_OUTPUT],
    ids=lambda hp: hp.value,
)
def test_transformer_hooks_called(
    mocker: MockerFixture,
    transformer: Transformer,
    transformer_params: Params,
    hook_point: HookPoint,
):
    stub = mocker.stub()
    variables = {"params": transformer_params}
    inputs = jnp.ones((256,), jnp.int32)
    hooks = {hook_point.value: Hook(make_hook_fn(stub))}
    transformer.apply(variables, inputs, hooks)

    if hook_point != HookPoint.RESIDUAL:
        stub.assert_called_once()
    else:
        stub.assert_called()


@pytest.mark.parametrize(
    "hook_point",
    [HookPoint.MLP_PRE_ACTIVATION, HookPoint.MLP_POST_ACTIVATION],
    ids=lambda hp: hp.value,
)
def test_mlp_hooks_called(
    rng: Array,
    mocker: MockerFixture,
    mlp: MLP,
    mlp_params: Params,
    hook_point: HookPoint,
):
    stub = mocker.stub()
    variables = {"params": mlp_params}
    inputs = jr.uniform(rng, (256, 256))
    hooks = {hook_point.value: Hook(make_hook_fn(stub))}
    mlp.apply(variables, inputs, hooks)

    stub.assert_called_once()


@pytest.mark.parametrize(
    "hook_point",
    [
        HookPoint.ATTN_QUERY,
        HookPoint.ATTN_KEY,
        HookPoint.ATTN_VALUE,
        HookPoint.ATTN_SCORES,
        HookPoint.ATTN_WEIGHTS,
        HookPoint.ATTN_Z,
        HookPoint.ATTN_OUTPUT,
    ],
    ids=lambda hp: hp.value,
)
def test_attention_hooks_called(
    rng: Array,
    mocker: MockerFixture,
    multi_head_attention: MultiHeadAttention,
    multi_head_attention_params: Params,
    hook_point: HookPoint,
):
    stub = mocker.stub()
    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (1024, 768))
    hooks = {hook_point.value: Hook(make_hook_fn(stub))}
    multi_head_attention.apply(variables, inputs, hooks)

    stub.assert_called_once()


@pytest.mark.parametrize(
    "hook_point", [HookPoint.LN_STD, HookPoint.LN_NORMALIZED], ids=lambda hp: hp.value
)
def test_layer_norm_hooks_called(
    rng: Array,
    mocker: MockerFixture,
    layer_norm: LayerNorm,
    layer_norm_params: Params,
    hook_point: HookPoint,
):
    stub = mocker.stub()
    variables = {"params": layer_norm_params}
    inputs = jr.uniform(rng, (256,))
    hooks = {hook_point.value: Hook(make_hook_fn(stub))}
    layer_norm.apply(variables, inputs, hooks)

    stub.assert_called_once()


SEQ_LENGTH = 256


def format_ids(val):
    if isinstance(val, HookPoint):
        return val.value
    elif isinstance(val, tuple):
        return val[0]
    else:
        return None


@pytest.mark.parametrize(
    "hook_point,expected",
    [
        (HookPoint.EMBED, (SEQ_LENGTH, 768)),
        (HookPoint.POS_EMBED, (SEQ_LENGTH, 768)),
        (HookPoint.RESIDUAL, (SEQ_LENGTH, 768)),
        (HookPoint.FINAL_OUTPUT, (SEQ_LENGTH, 768)),
    ],
    ids=format_ids,
)
def test_transformer_hooks_can_store_values(
    transformer: Transformer,
    transformer_params: Params,
    hook_point: HookPoint,
    expected: Sequence[int],
):
    hooks = {hook_point.value: StoreHook}

    variables = {"params": transformer_params}
    inputs = jnp.ones((SEQ_LENGTH,), jnp.int32)
    _, state = transformer.apply(variables, inputs, hooks, mutable=["intermediates"])

    assert state["intermediates"][hook_point.value][0].shape == expected


@pytest.mark.parametrize(
    "hook_point,expected",
    [
        (HookPoint.MLP_PRE_ACTIVATION, (SEQ_LENGTH, 3072)),
        (HookPoint.MLP_POST_ACTIVATION, (SEQ_LENGTH, 3072)),
    ],
    ids=format_ids,
)
def test_mlp_hooks_can_store_values(
    rng: Array,
    mlp: MLP,
    mlp_params: Params,
    hook_point: HookPoint,
    expected: Sequence[int],
):
    hooks = {hook_point.value: StoreHook}

    variables = {"params": mlp_params}
    inputs = jr.uniform(rng, (256, 256))
    _, state = mlp.apply(variables, inputs, hooks, mutable=["intermediates"])

    assert state["intermediates"][hook_point.value][0].shape == expected


@pytest.mark.parametrize(
    "hook_point,expected",
    [
        (HookPoint.ATTN_QUERY, (SEQ_LENGTH, 12, 64)),
        (HookPoint.ATTN_KEY, (SEQ_LENGTH, 12, 64)),
        (HookPoint.ATTN_VALUE, (SEQ_LENGTH, 12, 64)),
        (HookPoint.ATTN_SCORES, (12, SEQ_LENGTH, SEQ_LENGTH)),
        (HookPoint.ATTN_WEIGHTS, (12, SEQ_LENGTH, SEQ_LENGTH)),
        (HookPoint.ATTN_Z, (SEQ_LENGTH, 12, 64)),
        (HookPoint.ATTN_OUTPUT, (SEQ_LENGTH, 768)),
    ],
    ids=format_ids,
)
def test_attention_hooks_can_store_values(
    rng: Array,
    multi_head_attention: MultiHeadAttention,
    multi_head_attention_params: Params,
    hook_point: HookPoint,
    expected: Sequence[int],
):
    hooks = {hook_point.value: StoreHook}

    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (SEQ_LENGTH, 768))
    _, state = multi_head_attention.apply(
        variables, inputs, hooks, mutable=["intermediates"]
    )

    assert state["intermediates"][hook_point.value][0].shape == expected


@pytest.mark.parametrize(
    "hook_point,expected",
    [
        (HookPoint.LN_STD, (SEQ_LENGTH, SEQ_LENGTH)),
        (HookPoint.LN_NORMALIZED, (SEQ_LENGTH, SEQ_LENGTH)),
    ],
    ids=format_ids,
)
def test_layer_norm_hooks_can_store_values(
    rng: Array,
    layer_norm: LayerNorm,
    layer_norm_params: Params,
    hook_point: HookPoint,
    expected: Sequence[int],
):
    hooks = {hook_point.value: StoreHook}

    variables = {"params": layer_norm_params}
    inputs = jr.uniform(rng, (SEQ_LENGTH, SEQ_LENGTH))
    _, state = layer_norm.apply(variables, inputs, hooks, mutable=["intermediates"])

    assert state["intermediates"][hook_point.value][0].shape == expected


if __name__ == "__main__":
    pytest.main([__file__])
