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

from tx.modules import (
    Embed,
    PosEmbed,
    MultiHeadAttention,
    LayerNorm,
    Unembed,
)


RNG = jr.PRNGKey(0)
ERROR = 1e-4


def format_params(batch_dims: Sequence[int]) -> str:
    return f"batch_dims={batch_dims}"


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_embed_init(batch_dims: Sequence[int]):
    layer = Embed(features=768, num_embeddings=50257, init_range=0.02)

    input: Int[Array, "S"] = jnp.ones((*batch_dims, 4), dtype=jnp.int32)
    variables = layer.init(RNG, input)

    embedding: Array = variables["params"]["embedding"]
    assert embedding.shape == (50257, 768)
    assert jnp.std(embedding) < 0.02 + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_pos_embed_init(batch_dims: Sequence[int]):
    layer = PosEmbed(features=768, num_embeddings=1024, init_range=0.02)

    input: Int[Array, "S"] = jnp.broadcast_to(
        jnp.arange(0, 4, dtype=jnp.int32), (*batch_dims, 4)
    )
    variables = layer.init(RNG, input)

    embedding: Array = variables["params"]["embedding"]
    assert embedding.shape == (1024, 768)
    assert jnp.std(embedding) < 0.02 + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_attention_init(batch_dims: Sequence[int]):
    layer = MultiHeadAttention(
        num_heads=12,
        head_dim=64,
        features=768,
        context_length=1024,
        init_range=0.02,
    )

    input: Float[Array, "S F"] = jnp.ones((*batch_dims, 4, 768), dtype=jnp.float32)
    mask = nn.make_causal_mask(jnp.ones(4), dtype="bool")
    variables = layer.init(RNG, input, mask)

    proj_kernel: Array = variables["params"]["c_proj"]["kernel"]
    assert proj_kernel.shape[-1] == 768
    assert np.prod(proj_kernel.shape[:-1]) == 12 * 64
    assert jnp.std(proj_kernel) < 0.02 + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_layer_norm_init(batch_dims: Sequence[int]):
    layer = LayerNorm(epsilon=1e-5)

    input: Float[Array, "S F"] = jnp.ones((*batch_dims, 4, 768), dtype=jnp.float32)
    variables = layer.init(RNG, input)

    scale: Array = variables["params"]["scale"]
    assert scale.shape == (768,)
    assert jnp.all(scale == 1.0)

    bias: Array = variables["params"]["bias"]
    assert bias.shape == (768,)
    assert jnp.all(bias == 0.0)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_unembed_init(batch_dims: Sequence[int]):
    layer = Unembed(features=768, num_embeddings=50257, init_range=0.02)

    input: Float[Array, "S F"] = jnp.ones((*batch_dims, 4, 768), dtype=jnp.float32)
    variables = layer.init(RNG, input)

    kernel: Array = variables["params"]["kernel"]
    assert kernel.shape == (768, 50257)
    assert jnp.std(kernel) < 0.02 + ERROR
