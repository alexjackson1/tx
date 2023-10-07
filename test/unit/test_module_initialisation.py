"""Tests neural network module initialisation shapes and values."""

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from typing import Sequence
from jaxtyping import Array, Float, Int

import pytest

import numpy as np
import jax.random as jr
import jax.numpy as jnp
import flax.linen as nn

import tx

RNG = jr.PRNGKey(0)
ERROR = 1e-4


@pytest.fixture
def config():
    return tx.TransformerConfig(dtype=jnp.float32, param_dtype=jnp.float32)


def format_params(batch_dims: Sequence[int]) -> str:
    return f"batch_dims={batch_dims}"


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_embed_init(config: tx.TransformerConfig, batch_dims: Sequence[int]):
    layer = tx.Embed(
        features=config.model_dim,
        num_embeddings=config.vocab_dim,
        init_range=config.init_range,
    )

    input: Int[Array, "S"] = jnp.ones((*batch_dims, 4), dtype=jnp.int32)
    variables = layer.init(RNG, input)

    embedding: Array = variables["params"]["embedding"]
    assert embedding.shape == (config.vocab_dim, config.model_dim)
    assert jnp.std(embedding) < config.init_range + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_pos_embed_init(config: tx.TransformerConfig, batch_dims: Sequence[int]):
    layer = tx.PosEmbed(
        features=config.model_dim,
        num_embeddings=config.context_length,
        init_range=config.init_range,
    )

    input: Int[Array, "S"] = jnp.broadcast_to(
        jnp.arange(0, 4, dtype=jnp.int32), (*batch_dims, 4)
    )
    variables = layer.init(RNG, input)

    embedding: Array = variables["params"]["embedding"]
    assert embedding.shape == (config.context_length, config.model_dim)
    assert jnp.std(embedding) < config.init_range + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_attention_init(config: tx.TransformerConfig, batch_dims: Sequence[int]):
    layer = tx.MultiHeadAttention(
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        features=config.model_dim,
        context_length=config.context_length,
        init_range=config.init_range,
    )

    input: Float[Array, "S F"] = jnp.ones(
        (*batch_dims, 4, config.model_dim), dtype=jnp.float32
    )
    mask = nn.make_causal_mask(jnp.ones(4), dtype="bool")
    variables = layer.init(RNG, input, mask)

    proj_kernel: Array = variables["params"]["c_proj"]["kernel"]
    assert proj_kernel.shape[-1] == config.model_dim
    assert np.prod(proj_kernel.shape[:-1]) == config.num_heads * config.head_dim
    assert jnp.std(proj_kernel) < config.init_range + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_mlp_init(config: tx.TransformerConfig, batch_dims: Sequence[int]):
    layer = tx.MLP(
        features=[config.mlp_dim, config.model_dim],
        init_range=config.init_range,
    )

    input: Float[Array, "S F"] = jnp.ones(
        (*batch_dims, 4, config.model_dim), dtype=jnp.float32
    )
    variables = layer.init(RNG, input)

    proj_kernel: Array = variables["params"]["proj"]["kernel"]
    assert proj_kernel.shape == (config.mlp_dim, config.model_dim)
    assert jnp.std(proj_kernel) < config.init_range + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_layer_norm_init(config: tx.TransformerConfig, batch_dims: Sequence[int]):
    layer = tx.LayerNorm(epsilon=config.layer_norm_eps)

    input: Float[Array, "S F"] = jnp.ones(
        (*batch_dims, 4, config.model_dim), dtype=jnp.float32
    )
    variables = layer.init(RNG, input)

    scale: Array = variables["params"]["scale"]
    assert scale.shape == (config.model_dim,)
    assert jnp.all(scale == 1.0)

    bias: Array = variables["params"]["bias"]
    assert bias.shape == (config.model_dim,)
    assert jnp.all(bias == 0.0)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_transformer_block_init(
    config: tx.TransformerConfig, batch_dims: Sequence[int]
):
    layer = tx.TransformerBlock(
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        model_dim=config.model_dim,
        mlp_dim=config.mlp_dim,
        epsilon=config.layer_norm_eps,
        init_range=config.init_range,
    )

    input: Float[Array, "S F"] = jnp.ones(
        (*batch_dims, 4, config.model_dim), dtype=jnp.float32
    )
    variables = layer.init(RNG, input)

    ln1_scale: Array = variables["params"]["ln_1"]["scale"]
    assert ln1_scale.shape == (config.model_dim,)
    assert jnp.std(ln1_scale) < config.init_range + ERROR

    ln2_scale: Array = variables["params"]["ln_2"]["scale"]
    assert ln2_scale.shape == (config.model_dim,)
    assert jnp.std(ln2_scale) < config.init_range + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_unembed_init(config: tx.TransformerConfig, batch_dims: Sequence[int]):
    layer = tx.Unembed(
        features=config.model_dim,
        num_embeddings=config.vocab_dim,
        init_range=config.init_range,
    )

    input: Float[Array, "S F"] = jnp.ones(
        (*batch_dims, 4, config.model_dim), dtype=jnp.float32
    )
    variables = layer.init(RNG, input)

    kernel: Array = variables["params"]["kernel"]
    assert kernel.shape == (config.model_dim, config.vocab_dim)
    assert jnp.std(kernel) < config.init_range + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_transformer_init(config: tx.TransformerConfig, batch_dims: Sequence[int]):
    model = tx.Transformer.from_config(config)

    input: Int[Array, "S"] = jnp.ones((*batch_dims, 4), dtype=jnp.int32)
    variables = model.init(RNG, input)

    assert len(variables["params"]) == 16
