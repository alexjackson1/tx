import os, sys


sys.path.append(os.path.abspath(os.path.join("../..")))

import jax

jax.config.update("jax_enable_x64", True)

import pytest

from jaxtyping import Array, Int

import jax.numpy as jnp

from tx import TransformerWithHooks
from tx.models.gpt2 import GPT2Config


@pytest.fixture
def transformer():
    return TransformerWithHooks.from_pretrained("gpt2", decode=True, dtype=jnp.float32)


def check_cache_contents(cache: dict, config: GPT2Config, num_tokens: int):
    # Extract and check the shapes of the cached attention keys and values
    cached_key: Array = cache["block_0"]["attn"]["cached_key"]
    cached_value: Array = cache["block_0"]["attn"]["cached_value"]
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
    assert cache["block_0"]["attn"]["cache_index"] == num_tokens

    # Count and check the number of filled keys and values
    all_heads_size = config.num_heads * config.head_dim
    filled_keys: Array = jnp.count_nonzero(cached_key, axis=(1, 2)) / all_heads_size
    filled_values: Array = jnp.count_nonzero(cached_value, axis=(1, 2)) / all_heads_size
    assert jnp.sum(filled_keys[:num_tokens]) == num_tokens
    assert jnp.sum(filled_keys[num_tokens:]) == 0.0
    assert jnp.sum(filled_values[:num_tokens]) == num_tokens
    assert jnp.sum(filled_values[num_tokens:]) == 0.0


def test_gpt2_model(transformer: TransformerWithHooks):
    # Define the prompt and number of tokens it contains
    prompt = "Hello, I am"
    num_tokens = 5

    # Check that the model is properly initialised for this test
    config: GPT2Config = transformer.config
    assert config.decode
    assert transformer.tokenizer is not None
    assert transformer.params is not None

    # Use the model to convert the prompt to tokens, check output
    tokens: Int[Array, "S"] = transformer.to_tokens(prompt, prepend_bos=True)
    assert tokens.shape == (num_tokens,)
    assert tokens[0] == transformer.tokenizer.bos_token_id
    assert tokens[1] == transformer.to_tokens("Hello")[0]

    # Run the model on the tokens, check output
    expected_outputs = ("\n", ",", " I", "'m", " a")
    cur_tokens: Int[Array, "S"] = tokens[None, 0]
    for i, output in enumerate(expected_outputs):
        logits, _ = transformer(cur_tokens)
        assert logits.shape == (cur_tokens.shape[0], config.vocab_dim)

        probs = jax.nn.softmax(logits)
        next_token = jnp.argmax(probs, axis=-1)[-1]
        assert transformer.to_str(next_token) == output

        if i < num_tokens - 1:
            cur_tokens = jnp.append(cur_tokens, tokens[i + 1])

    # Check the cache contents
    check_cache_contents(transformer.cache, config, num_tokens)

    # Compute the next token logits
    probs: Array = jax.nn.softmax(logits)
    next_tokens = jnp.argmax(probs, axis=-1)
    next_token_strs = transformer.to_str_list(next_tokens)
    assert next_token_strs[-1] == " a"

    # Run the model on the tokens, check output
    cur_tokens = jnp.append(cur_tokens, next_tokens[-1])
    logits, _ = transformer(cur_tokens)
    assert logits.shape == (num_tokens + 1, config.vocab_dim)

    # Check the cache contents
    check_cache_contents(transformer.cache, config, num_tokens + 1)

    # Check the next token logits
    probs: Array = jax.nn.softmax(logits)
    next_tokens = jnp.argmax(probs, axis=-1)
    next_token_strs = transformer.to_str_list(next_tokens)
    assert next_token_strs[-1] == " student"
