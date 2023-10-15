"""Tests neural network module contracts (i.e. input and output shapes)."""

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from typing import Sequence
from jaxtyping import Array, Float, Int

import pytest

import jax.random as jr
import jax.numpy as jnp

from tx.modules import Embed, PosEmbed, MultiHeadAttention, LayerNorm, Unembed


@pytest.fixture
def rng():
    return jr.PRNGKey(0)


def format_params(batch_dims: Sequence[int]) -> str:
    return f"batch_dims={batch_dims}"


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_embed_contract(rng: Array, batch_dims: Sequence[int]):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = Embed(features=768, num_embeddings=50257)
    init_input: Int[Array, "B S"] = jnp.ones((*batch_dims, INIT_LEN), dtype=jnp.int32)
    variables = layer.init(rng, init_input)

    apply_input: Int[Array, "B S"] = jr.randint(rng, (*batch_dims, APPLY_LEN), 0, 50257)
    output: Float[Array, "S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, 768)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_pos_embed_contract(rng: Array, batch_dims: Sequence[int]):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = PosEmbed(features=768, num_embeddings=1024)
    init_input: Int[Array, "... S"] = jnp.broadcast_to(
        jnp.arange(0, INIT_LEN, dtype=jnp.int32), (*batch_dims, INIT_LEN)
    )
    variables = layer.init(rng, init_input)

    apply_input: Int[Array, "... S"] = jnp.broadcast_to(
        jnp.arange(0, APPLY_LEN, dtype=jnp.int32), (*batch_dims, APPLY_LEN)
    )
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, 768)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_attention_contract(rng: Array, batch_dims: Sequence[int]):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = MultiHeadAttention(
        num_heads=12, head_dim=64, features=768, context_length=1024
    )
    init_input: Float[Array, "... S F"] = jr.uniform(rng, (*batch_dims, INIT_LEN, 768))
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, APPLY_LEN, 768)
    )
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, 768)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_layer_norm_contract(rng: Array, batch_dims: Sequence[int]):
    ARBITRARY_DIMS = (4, 7)

    layer = LayerNorm(epsilon=1e-5)
    init_input: Float[Array, "... S F"] = jr.uniform(rng, batch_dims + ARBITRARY_DIMS)
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(rng, batch_dims + ARBITRARY_DIMS)
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == batch_dims + ARBITRARY_DIMS


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_unembed_contract(rng: Array, batch_dims: Sequence[int]):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = Unembed(features=768, num_embeddings=50257)
    init_input: Float[Array, "... S F"] = jr.uniform(rng, (*batch_dims, INIT_LEN, 768))
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, APPLY_LEN, 768)
    )
    output: Float[Array, "... S V"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, 50257)
