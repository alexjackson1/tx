"""Tests neural network module contracts (i.e. input and output shapes)."""

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from typing import List, Union
from jaxtyping import Array, Float, Bool

import pytest

import jax.random as jr
import jax.numpy as jnp

from transformers import GPT2TokenizerFast

import tx
import tx.network


@pytest.fixture
def rng():
    return jr.PRNGKey(0)


@pytest.fixture
def config():
    return tx.TransformerConfig(dtype=jnp.float32, param_dtype=jnp.float32)


@pytest.fixture
def blank_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.eos_token = None
    tokenizer.pad_token = None
    tokenizer.bos_token = None
    tokenizer.padding_side = None
    return tokenizer


@pytest.fixture
def tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return tx.network.configure_tokenizer(tokenizer)


def count_populated_entries(
    cache_array: Float[Array, "... S NH HD"]
) -> Bool[Array, "S"]:
    num_batch_dims = len(cache_array.shape) - 3
    batch_dims = tuple(range(num_batch_dims))
    return jnp.any(cache_array != 0.0, axis=(*batch_dims, -2, -1))


def test_configure_tokenizer(blank_tokenizer):
    tokenizer = tx.network.configure_tokenizer(blank_tokenizer)
    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.pad_token == "<|endoftext|>"
    assert tokenizer.bos_token == "<|endoftext|>"
    assert tokenizer.padding_side == "right"


def test_token_ops_raise_errors(config):
    model = tx.network.GenerativeModel(config)

    with pytest.raises(ValueError):
        model.to_tokens("Example input", prepend_bos=True)

    with pytest.raises(ValueError):
        model.to_str(1)

    with pytest.raises(ValueError):
        model.to_str([1, 2, 3])

    with pytest.raises(ValueError):
        model.to_str_list([1, 2, 3])


@pytest.mark.parametrize(
    "input,expected",
    [(1, '"'), ([1, 2, 3], '"#$'), ([1, 2, 3, 4], '"#$%')],
    ids=["input=1", "input=[1, 2, 3]", "input=[1, 2, 3, 4]"],
)
def test_token_ops_with_int_ids(
    input: Union[int, List[int]],
    expected: str,
    config: tx.TransformerConfig,
    tokenizer: GPT2TokenizerFast,
):
    model = tx.network.GenerativeModel(config, tokenizer=tokenizer)
    assert model.to_str(input) == expected
    assert model.to_str(jnp.array(input)) == expected
    assert model.to_str_list(input) == [c for c in expected]


@pytest.mark.parametrize(
    "input,str_array,tokens",
    [
        ("Autoregressive", ["Aut", "ore", "gressive"], jnp.array([16541, 382, 19741])),
        ("Unhappy", ["Un", "happy"], jnp.array([3118, 34191])),
        ("rather", ["rather"], jnp.array([34330])),
    ],
    ids=["input='Autoregressive'", "input='Unhappy'", "input='rather'"],
)
def test_token_ops_with_string(
    input: str,
    str_array: List[str],
    tokens: Array,
    config: tx.TransformerConfig,
    tokenizer: GPT2TokenizerFast,
):
    model = tx.network.GenerativeModel(config, tokenizer=tokenizer)
    assert model.to_str_list(input) == str_array
    assert jnp.all(model.to_tokens(input) == tokens)
