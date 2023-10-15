"""Tests that converted GPT-2 params fit model."""

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from jaxtyping import Array

import pytest

import jax.numpy as jnp

from tx.modules import Embed, PosEmbed, MultiHeadAttention, LayerNorm, Unembed
from tx.models.gpt2.loader import GPT2Loader
from tx.models.gpt2.modules import (
    GPT2Config,
    GPT2TransformerBlock,
    GPT2Transformer,
    GPT2MLP,
)
from tx.tree_util import Params


@pytest.fixture
def gpt2_params(gpt2_config: GPT2Config):
    return GPT2Loader.load_params("gpt2", gpt2_config)


@pytest.fixture
def gpt2_config():
    return GPT2Config()


def test_embed_with_gpt2_params(gpt2_config: GPT2Config, gpt2_params: Params):
    model = Embed(
        features=gpt2_config.model_dim,
        num_embeddings=gpt2_config.vocab_dim,
        param_dtype=gpt2_config.param_dtype,
    )
    variables = {"params": gpt2_params["embed"]}
    input_data = jnp.ones((gpt2_config.context_length,), dtype=jnp.int32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (gpt2_config.context_length, gpt2_config.model_dim)


def test_pos_embed_with_gpt2_params(gpt2_config: GPT2Config, gpt2_params: Params):
    model = PosEmbed(
        features=gpt2_config.model_dim,
        num_embeddings=gpt2_config.context_length,
        param_dtype=gpt2_config.param_dtype,
    )
    variables = {"params": gpt2_params["pos_embed"]}
    input_data = jnp.ones((gpt2_config.context_length,), dtype=gpt2_config.dtype)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (gpt2_config.context_length, gpt2_config.model_dim)


def test_attention_with_gpt2_params(gpt2_config: GPT2Config, gpt2_params: Params):
    model = MultiHeadAttention(
        num_heads=gpt2_config.num_heads,
        head_dim=gpt2_config.head_dim,
        features=gpt2_config.model_dim,
        dtype=gpt2_config.dtype,
        param_dtype=gpt2_config.param_dtype,
    )
    variables = {"params": gpt2_params["block_0"]["attn"]}
    input_data = jnp.ones(
        (gpt2_config.context_length, gpt2_config.model_dim), dtype=gpt2_config.dtype
    )
    output: Array = model.apply(variables, input_data)
    assert output.shape == (gpt2_config.context_length, gpt2_config.model_dim)


def test_mlp_with_gpt2_params(gpt2_config: GPT2Config, gpt2_params: Params):
    model = GPT2MLP(
        features=[gpt2_config.mlp_dim, gpt2_config.model_dim],
        dtype=gpt2_config.dtype,
        param_dtype=gpt2_config.param_dtype,
    )
    variables = {"params": gpt2_params["block_0"]["mlp"]}
    input_data = jnp.ones(
        (gpt2_config.context_length, gpt2_config.model_dim), dtype=gpt2_config.dtype
    )
    output: Array = model.apply(variables, input_data)
    assert output.shape == (gpt2_config.context_length, gpt2_config.model_dim)


def test_layer_norm_with_gpt2_params(gpt2_config: GPT2Config, gpt2_params: Params):
    model = LayerNorm(
        epsilon=gpt2_config.layer_norm_eps,
        dtype=gpt2_config.dtype,
        param_dtype=gpt2_config.param_dtype,
    )
    variables = {"params": gpt2_params["ln_f"]}
    input_data = jnp.ones(
        (gpt2_config.context_length, gpt2_config.model_dim), dtype=gpt2_config.dtype
    )
    output: Array = model.apply(variables, input_data)
    assert output.shape == (gpt2_config.context_length, gpt2_config.model_dim)


def test_transformer_block_with_gpt2_params(
    gpt2_config: GPT2Config, gpt2_params: Params
):
    model = GPT2TransformerBlock(
        num_heads=gpt2_config.num_heads,
        head_dim=gpt2_config.head_dim,
        model_dim=gpt2_config.model_dim,
        mlp_dim=gpt2_config.mlp_dim,
        dtype=gpt2_config.dtype,
        param_dtype=gpt2_config.param_dtype,
    )
    variables = {"params": gpt2_params["block_0"]}
    input_data = jnp.ones(
        (gpt2_config.context_length, gpt2_config.model_dim), dtype=gpt2_config.dtype
    )
    output: Array = model.apply(variables, input_data)
    assert output.shape == (gpt2_config.context_length, gpt2_config.model_dim)


def test_unembed_with_gpt2_params(gpt2_config: GPT2Config, gpt2_params: Params):
    model = Unembed(
        features=gpt2_config.model_dim,
        num_embeddings=gpt2_config.vocab_dim,
        dtype=gpt2_config.dtype,
        param_dtype=gpt2_config.param_dtype,
    )
    variables = {"params": gpt2_params["unembed"]}
    input_data = jnp.ones(
        (gpt2_config.context_length, gpt2_config.model_dim), dtype=gpt2_config.dtype
    )
    output: Array = model.apply(variables, input_data)
    assert output.shape == (gpt2_config.context_length, gpt2_config.vocab_dim)


def test_transformer_with_gpt2_params(gpt2_config: GPT2Config, gpt2_params: Params):
    model = GPT2Transformer.from_config(gpt2_config)
    variables = {"params": gpt2_params}
    input_data = jnp.ones((gpt2_config.context_length,), dtype=jnp.int32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (gpt2_config.context_length, gpt2_config.vocab_dim)
