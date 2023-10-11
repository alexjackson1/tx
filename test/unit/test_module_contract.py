"""Tests neural network module contracts (i.e. input and output shapes)."""

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from typing import Sequence
from jaxtyping import Array, Float, Int

import pytest

import jax.random as jr
import jax.numpy as jnp
import flax.linen as nn

import tx


@pytest.fixture
def rng():
    return jr.PRNGKey(0)


@pytest.fixture
def config():
    return tx.TransformerConfig(dtype=jnp.float32, param_dtype=jnp.float32)


def format_params(batch_dims: Sequence[int]) -> str:
    return f"batch_dims={batch_dims}"


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_embed_contract(
    rng: Array, config: tx.TransformerConfig, batch_dims: Sequence[int]
):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = tx.Embed(features=config.model_dim, num_embeddings=config.vocab_dim)
    init_input: Int[Array, "B S"] = jnp.ones((*batch_dims, INIT_LEN), dtype=jnp.int32)
    variables = layer.init(rng, init_input)

    apply_input: Int[Array, "B S"] = jr.randint(
        rng, (*batch_dims, APPLY_LEN), 0, config.vocab_dim
    )
    output: Float[Array, "S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, config.model_dim)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_pos_embed_contract(
    rng: Array, config: tx.TransformerConfig, batch_dims: Sequence[int]
):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = tx.PosEmbed(features=config.model_dim, num_embeddings=config.context_length)
    init_input: Int[Array, "... S"] = jnp.broadcast_to(
        jnp.arange(0, INIT_LEN, dtype=jnp.int32), (*batch_dims, INIT_LEN)
    )
    variables = layer.init(rng, init_input)

    apply_input: Int[Array, "... S"] = jnp.broadcast_to(
        jnp.arange(0, APPLY_LEN, dtype=jnp.int32), (*batch_dims, APPLY_LEN)
    )
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, config.model_dim)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_attention_contract(
    rng: Array, config: tx.TransformerConfig, batch_dims: Sequence[int]
):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = tx.MultiHeadAttention(
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        features=config.model_dim,
        context_length=config.context_length,
    )
    init_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, INIT_LEN, config.model_dim)
    )
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, APPLY_LEN, config.model_dim)
    )
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, config.model_dim)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_mlp_contract(
    rng: Array, config: tx.TransformerConfig, batch_dims: Sequence[int]
):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = tx.MLP(features=[config.mlp_dim, config.model_dim])
    init_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, INIT_LEN, config.model_dim)
    )
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, APPLY_LEN, config.model_dim)
    )
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, config.model_dim)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_layer_norm_contract(rng: Array, batch_dims: Sequence[int]):
    ARBITRARY_DIMS = (4, 7)

    layer = tx.LayerNorm(epsilon=1e-5)
    init_input: Float[Array, "... S F"] = jr.uniform(rng, batch_dims + ARBITRARY_DIMS)
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(rng, batch_dims + ARBITRARY_DIMS)
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == batch_dims + ARBITRARY_DIMS


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_transformer_block_contract(
    rng: Array, config: tx.TransformerConfig, batch_dims: Sequence[int]
):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = tx.TransformerBlock(
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        model_dim=config.model_dim,
        mlp_dim=config.mlp_dim,
        epsilon=config.layer_norm_eps,
    )

    init_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, INIT_LEN, config.model_dim)
    )
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, APPLY_LEN, config.model_dim)
    )
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, config.model_dim)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_unembed_contract(
    rng: Array, config: tx.TransformerConfig, batch_dims: Sequence[int]
):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = tx.Unembed(features=config.model_dim, num_embeddings=config.vocab_dim)
    init_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, INIT_LEN, config.model_dim)
    )
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, APPLY_LEN, config.model_dim)
    )
    output: Float[Array, "... S V"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, config.vocab_dim)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_transformer_contract(
    rng: Array, config: tx.TransformerConfig, batch_dims: Sequence[int]
):
    INIT_LEN = 4
    APPLY_LEN = 7

    model = tx.Transformer.from_config(config)
    init_input: Int[Array, "... S"] = jnp.ones((*batch_dims, INIT_LEN), dtype=jnp.int32)
    variables = model.init(rng, init_input)

    apply_input: Int[Array, "... S"] = jr.randint(
        rng, (*batch_dims, APPLY_LEN), 0, config.vocab_dim
    )
    output: Float[Array, "... S V"] = model.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, config.vocab_dim)
