import os, sys


sys.path.append(os.path.abspath(os.path.join("../..")))

import jax

jax.config.update("jax_enable_x64", True)

import pytest

from jaxtyping import Array

import jax.numpy as jnp
import flax.linen as nn

from transformers import GPT2TokenizerFast

from tx import TransformerConfig
from tx.models.gpt2 import PretrainedGPT2Model
from tx.hooks import HookPoint, StoreHook
from tx.network import GenerativeModel


@pytest.fixture
def config():
    return PretrainedGPT2Model.from_pretrained("gpt2").make_config(
        decode=True, dtype=jnp.float32, param_dtype=jnp.float32
    )


@pytest.fixture
def model(config: TransformerConfig):
    return GenerativeModel(
        config=config,
        tokenizer=GPT2TokenizerFast.from_pretrained("gpt2"),
        params=PretrainedGPT2Model.from_pretrained("gpt2").to_params(),
        hooks={HookPoint.ATTN_OUTPUT.value: StoreHook},
        hook_collections=["intermediates"],
    )


def test_gpt2_model(model: GenerativeModel):
    # Define the prompt and number of tokens it contains
    prompt = "Hello, I am"
    num_tokens = 5

    # Check that the model is properly initialised for this test
    config = model.config
    assert config.decode
    assert model.tokenizer is not None
    assert model.params is not None
    assert model.hooks is not None

    # Use the model to convert the prompt to tokens, check output
    tokens = model.to_tokens(prompt, prepend_bos=True)
    assert tokens.shape == (num_tokens,)
    assert tokens[0] == model.tokenizer.bos_token_id
    assert tokens[1] == model.to_tokens("Hello")[0]

    # Run the model on the tokens, check output
    expected_outputs = ("\n", ",", " I", "'m", " a")
    cur_tokens = tokens[None, 0]
    for i, output in enumerate(expected_outputs):
        logits, state = model(cur_tokens)
        # assert logits.shape == (1, model.config.vocab_dim)

        probs = jax.nn.softmax(logits)
        next_token = jnp.argmax(probs, axis=-1)[-1]
        assert model.to_str(next_token) == output

        if i <= 3:
            cur_tokens = jnp.append(cur_tokens, tokens[i + 1])

    # Check and extract the keys of the model state
    assert "intermediates" in state
    assert "cache" in state
    intermediates, cache = state["intermediates"], state["cache"]

    # Extract and check the shapes of the cached attention keys and values
    cached_key = cache["block_0"]["attn"]["cached_key"]
    cached_value = cache["block_0"]["attn"]["cached_value"]
    assert cached_key.shape == (
        config.context_length,
        config.num_heads,
        config.head_dim,
    )
    assert cached_value.shape == (
        config.context_length,
        config.num_heads,
        config.head_dim,
    )

    # Check that the index is correct
    assert cache["block_0"]["attn"]["cached_idx"] == num_tokens

    # Count and check the number of filled keys and values
    all_heads_size = config.num_heads * config.head_dim
    filled_keys: Array = jnp.count_nonzero(cached_key, axis=(1, 2)) / all_heads_size
    filled_values: Array = jnp.count_nonzero(cached_value, axis=(1, 2)) / all_heads_size
    assert jnp.sum(filled_keys[:num_tokens]) == num_tokens
    assert jnp.sum(filled_keys[num_tokens:]) == 0.0
    assert jnp.sum(filled_values[:num_tokens]) == num_tokens
    assert jnp.sum(filled_values[num_tokens:]) == 0.0

    # Compute the next token logits
    probs: Array = jax.nn.softmax(logits)
    next_tokens = jnp.argmax(probs, axis=-1)
    next_token_strs = model.to_str_list(next_tokens)
    # assert next_token_strs[0] == "\n"
    # assert next_token_strs[1] == ","
    # assert next_token_strs[2] == " I"
    # assert next_token_strs[3] == "'m"
    assert next_token_strs[0] == " a"

    # Append the next token to the tokens, check output
    next_token = next_tokens[-1]
    tokens: Array = jnp.append(tokens, next_token)
    assert tokens.shape == (num_tokens + 1,)

    # Run the model on the tokens, check output
    logits, state = model(tokens)
    intermediates, cache = state["intermediates"], state["cache"]
    assert logits.shape == (num_tokens + 1, model.config.vocab_dim)

    # Check and extract the keys of the model state
    cached_key = cache["block_0"]["attn"]["cached_key"]
    cached_value = cache["block_0"]["attn"]["cached_value"]
    assert cached_key.shape == (
        config.context_length,
        config.num_heads,
        config.head_dim,
    )
    assert cached_value.shape == (
        config.context_length,
        config.num_heads,
        config.head_dim,
    )

    # Check that the index is correct
    assert cache["block_0"]["attn"]["cached_idx"] == num_tokens + 1

    # Count and check the number of filled keys and values
    filled_keys: Array = jnp.count_nonzero(cached_key, axis=(1, 2)) / all_heads_size
    filled_values: Array = jnp.count_nonzero(cached_value, axis=(1, 2)) / all_heads_size
    assert jnp.sum(filled_keys[: num_tokens + 1]) == num_tokens + 1

    # Check the next token logits
    next_tokens = jnp.argmax(logits, axis=-1)
    next_token_strs = model.to_str_list(next_tokens)
    assert next_token_strs[0] == "\n"
    assert next_token_strs[1] == ","
    assert next_token_strs[2] == " I"
    assert next_token_strs[3] == "'m"
    assert next_token_strs[4] == " a"
    assert next_token_strs[5] == " student"
