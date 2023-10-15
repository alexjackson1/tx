import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from typing import List, Union
from jaxtyping import Array

import pytest

import jax.numpy as jnp

from transformers import GPT2TokenizerFast

import tx.tokens as token_ops


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
    return token_ops.configure_tokenizer(tokenizer)


def test_configure_tokenizer(blank_tokenizer):
    tokenizer = token_ops.configure_tokenizer(blank_tokenizer)
    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.pad_token == "<|endoftext|>"
    assert tokenizer.bos_token == "<|endoftext|>"
    assert tokenizer.padding_side == "right"


@pytest.mark.parametrize(
    "input,expected",
    [(1, '"'), ([1, 2, 3], '"#$'), ([1, 2, 3, 4], '"#$%')],
    ids=["input=1", "input=[1, 2, 3]", "input=[1, 2, 3, 4]"],
)
def test_token_ops_with_int_ids(
    input: Union[int, List[int]], expected: str, tokenizer: GPT2TokenizerFast
):
    assert token_ops.to_str(tokenizer, input) == expected
    assert token_ops.to_str(tokenizer, jnp.array(input)) == expected
    assert token_ops.to_str_list(tokenizer, input) == [c for c in expected]


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
    input: str, str_array: List[str], tokens: Array, tokenizer: GPT2TokenizerFast
):
    assert token_ops.to_str_list(tokenizer, input) == str_array
    assert jnp.all(token_ops.to_tokens(tokenizer, input) == tokens)
