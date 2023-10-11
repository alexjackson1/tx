"""Tests neural network module contracts (i.e. input and output shapes)."""

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from typing import Sequence
from jaxtyping import Array, Float, Bool

import pytest

import jax.random as jr
import jax.numpy as jnp

import tx


@pytest.fixture
def rng():
    return jr.PRNGKey(0)


@pytest.fixture
def config():
    return tx.TransformerConfig(dtype=jnp.float32, param_dtype=jnp.float32)


@pytest.fixture
def decode_module(config: tx.TransformerConfig):
    return tx.MultiHeadAttention(
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        features=config.model_dim,
        init_range=config.init_range,
        decode=True,
    )


def f(name: str):
    return lambda x: f"{name}={x}"


def count_populated_entries(
    cache_array: Float[Array, "... S NH HD"]
) -> Bool[Array, "S"]:
    num_batch_dims = len(cache_array.shape) - 3
    batch_dims = tuple(range(num_batch_dims))
    return jnp.any(cache_array != 0.0, axis=(*batch_dims, -2, -1))


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=f("batch_dims"))
@pytest.mark.parametrize("seq_len", [1, 5, 1024], ids=f("seq_len"))
def test_initialises_cache_correctly(
    rng: Array,
    decode_module: tx.MultiHeadAttention,
    batch_dims: Sequence[int],
    seq_len: int,
):
    input = jnp.ones((*batch_dims, seq_len, decode_module.features))
    variables = decode_module.init(rng, input)
    assert "cache" in variables

    cache = variables["cache"]
    cache_index = cache["cache_index"]
    cached_key, cached_value = cache["cached_key"], cache["cached_value"]
    expected_shape = (
        *batch_dims,
        decode_module.context_length,
        decode_module.num_heads,
        decode_module.head_dim,
    )

    assert cache_index == 0
    assert cached_key.shape == expected_shape
    assert cached_value.shape == expected_shape

    assert sum(count_populated_entries(cached_key)) == 0


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=f("batch_dims"))
@pytest.mark.parametrize("seq_len", [1, 5], ids=f("seq_len"))
def test_populates_cache_correctly(
    rng: Array,
    decode_module: tx.MultiHeadAttention,
    batch_dims: Sequence[int],
    seq_len: int,
):
    input = jnp.ones((*batch_dims, 1, decode_module.features))
    variables = decode_module.init(rng, input)

    for i in range(seq_len):
        logits, state = decode_module.apply(
            variables, input[..., : i + 1, :], mutable=["cache"]
        )

        cache = state["cache"]
        cache_index = cache["cache_index"]
        cached_key, cached_value = cache["cached_key"], cache["cached_value"]
        expected_shape = (
            *batch_dims,
            decode_module.context_length,
            decode_module.num_heads,
            decode_module.head_dim,
        )

        assert cache_index == i + 1
        assert cached_key.shape == expected_shape
        assert cached_value.shape == expected_shape

        variables["cache"] = cache

        assert sum(count_populated_entries(cached_key)) == i + 1
