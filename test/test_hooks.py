from jax.config import config
from pytest_mock import MockerFixture

from tx.modules.hooks import Hook

config.update("jax_enable_x64", True)


import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxtyping import Array

import pytest

import jax.random as jr
import jax.numpy as jnp
import flax.linen as nn
from optax import Params

from tx.modules import (
    TransformerConfig,
    Transformer,
    MLP,
    MultiHeadAttention,
    LayerNorm,
)


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
    return multi_head_attention.init(
        rng,
        jnp.ones((12, 64, 768)),
        nn.make_causal_mask(jnp.ones((256,))),
    )["params"]


@pytest.fixture
def layer_norm_params(rng, layer_norm: LayerNorm) -> Params:
    return layer_norm.init(rng, jnp.ones((256,)))["params"]


def make_hook_fn(stub):
    def hook_fn(x: Array):
        stub(x)
        return x

    return hook_fn


def test_transformer_embed_hook_called(
    mocker: MockerFixture, transformer, transformer_params
):
    stub = mocker.stub(name="embed_hook_stub")
    variables = {"params": transformer_params}
    inputs = jnp.ones((256,), jnp.int32)
    hooks = {"embed": Hook(make_hook_fn(stub))}
    output = transformer.apply(variables, inputs, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_transformer_pos_embed_hook_called(
    mocker: MockerFixture, transformer, transformer_params
):
    stub = mocker.stub(name="pos_embed_hook_stub")
    variables = {"params": transformer_params}
    inputs = jnp.ones((256,), jnp.int32)
    hooks = {"pos_embed": Hook(make_hook_fn(stub))}
    output = transformer.apply(variables, inputs, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_transformer_final_output_hook_called(
    mocker: MockerFixture, transformer, transformer_params
):
    stub = mocker.stub(name="final_output_hook_stub")
    variables = {"params": transformer_params}
    inputs = jnp.ones((256,), jnp.int32)
    hooks = {"final_output": Hook(make_hook_fn(stub))}
    output = transformer.apply(variables, inputs, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_mlp_pre_activation_hook_called(rng, mocker: MockerFixture, mlp, mlp_params):
    stub = mocker.stub(name="mlp_pre_activation_hook_stub")
    variables = {"params": mlp_params}
    inputs = jr.uniform(rng, (256, 256))
    hooks = {"pre_activation": Hook(make_hook_fn(stub))}
    output = mlp.apply(variables, inputs, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_mlp_post_activation_hook_called(rng, mocker: MockerFixture, mlp, mlp_params):
    stub = mocker.stub(name="mlp_post_activation_hook_stub")
    variables = {"params": mlp_params}
    inputs = jr.uniform(rng, (256, 256))
    hooks = {"post_activation": Hook(make_hook_fn(stub))}
    output = mlp.apply(variables, inputs, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_multi_head_attention_query_hook_called(
    rng, mocker: MockerFixture, multi_head_attention, multi_head_attention_params
):
    stub = mocker.stub(name="multi_head_attention_query_hook_stub")
    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (12, 64, 768))
    mask = nn.make_causal_mask(jnp.ones((256,)))
    hooks = {"attn_query": Hook(make_hook_fn(stub))}
    output = multi_head_attention.apply(variables, inputs, mask, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_multi_head_attention_key_hook_called(
    rng, mocker: MockerFixture, multi_head_attention, multi_head_attention_params
):
    stub = mocker.stub(name="multi_head_attention_key_hook_stub")
    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (12, 64, 768))
    mask = nn.make_causal_mask(jnp.ones((256,)))
    hooks = {"attn_key": Hook(make_hook_fn(stub))}
    output = multi_head_attention.apply(variables, inputs, mask, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_multi_head_attention_value_hook_called(
    rng, mocker: MockerFixture, multi_head_attention, multi_head_attention_params
):
    stub = mocker.stub(name="multi_head_attention_value_hook_stub")
    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (12, 64, 768))
    mask = nn.make_causal_mask(jnp.ones((256,)))
    hooks = {"attn_value": Hook(make_hook_fn(stub))}
    output = multi_head_attention.apply(variables, inputs, mask, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_multi_head_attention_scores_hook_called(
    rng, mocker: MockerFixture, multi_head_attention, multi_head_attention_params
):
    stub = mocker.stub(name="multi_head_attention_scores_hook_stub")
    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (12, 64, 768))
    mask = nn.make_causal_mask(jnp.ones((256,)))
    hooks = {"attn_scores": Hook(make_hook_fn(stub))}
    output = multi_head_attention.apply(variables, inputs, mask, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_multi_head_attention_weights_hook_called(
    rng, mocker: MockerFixture, multi_head_attention, multi_head_attention_params
):
    stub = mocker.stub(name="multi_head_attention_weights_hook_stub")
    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (12, 64, 768))
    mask = nn.make_causal_mask(jnp.ones((256,)))
    hooks = {"attn_weights": Hook(make_hook_fn(stub))}
    output = multi_head_attention.apply(variables, inputs, mask, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_multi_head_attention_z_hook_called(
    rng, mocker: MockerFixture, multi_head_attention, multi_head_attention_params
):
    stub = mocker.stub(name="multi_head_attention_z_hook_stub")
    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (12, 64, 768))
    mask = nn.make_causal_mask(jnp.ones((256,)))
    hooks = {"attn_z": Hook(make_hook_fn(stub))}
    output = multi_head_attention.apply(variables, inputs, mask, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_multi_head_attention_output_hook_called(
    rng, mocker: MockerFixture, multi_head_attention, multi_head_attention_params
):
    stub = mocker.stub(name="multi_head_attention_output_hook_stub")
    variables = {"params": multi_head_attention_params}
    inputs = jr.uniform(rng, (12, 64, 768))
    mask = nn.make_causal_mask(jnp.ones((256,)))
    hooks = {"attn_output": Hook(make_hook_fn(stub))}
    output = multi_head_attention.apply(variables, inputs, mask, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_layer_norm_std_hook_called(
    rng, mocker: MockerFixture, layer_norm, layer_norm_params
):
    stub = mocker.stub(name="layer_norm_std_hook_stub")
    variables = {"params": layer_norm_params}
    inputs = jr.uniform(rng, (256,))
    hooks = {"ln_std": Hook(make_hook_fn(stub))}
    output = layer_norm.apply(variables, inputs, hooks)

    output.block_until_ready()
    stub.assert_called_once()


def test_layer_norm_normalized_hook_called(
    rng, mocker: MockerFixture, layer_norm, layer_norm_params
):
    stub = mocker.stub(name="layer_norm_normalized_hook_stub")
    variables = {"params": layer_norm_params}
    inputs = jr.uniform(rng, (256,))
    hooks = {"ln_normalized": Hook(make_hook_fn(stub))}
    output = layer_norm.apply(variables, inputs, hooks)

    output.block_until_ready()
    stub.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
