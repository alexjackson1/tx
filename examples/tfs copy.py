import os, sys


# Update sys.path to include the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set jax_enable_x64=True to enable float64 precision
from jax.config import config

config.update("jax_enable_x64", True)

# Main script begins
from jaxtyping import Array, Float

import jax.numpy as jnp
import flax.linen as nn
from optax import Params

from transformers import GPT2TokenizerFast, PreTrainedTokenizer

from tx import TransformerConfig
from tx.hooks import HookPoint, StoreHook
from tx.models import PretrainedGPT2Model
from tx.network import GenerativeModel


def return_attn_key(
    config: TransformerConfig, tokenizer: PreTrainedTokenizer, params: Params
) -> Float[Array, "S NH HD"]:
    # Create a generative model storing the params and hooks
    hooks, collections = {HookPoint.ATTN_KEY.value: StoreHook}, ["intermediates"]
    gpt2 = GenerativeModel(config, tokenizer, params, hooks, collections)

    # Create an input prompt
    reference_text = "Hello, I am"
    tokens = gpt2.to_tokens(reference_text, prepend_bos=True)

    if config.decode:
        for token_idx in range(len(tokens)):
            token = tokens[None, token_idx]

            _, state = gpt2(token)
            key_cache_value = state["cache"]["block_0"]["attn"]["cached_key"]
            attn_key = key_cache_value
            assert attn_key.shape == (
                config.context_length,
                config.num_heads,
                config.head_dim,
            )
        return attn_key[: len(tokens), :, :]
    else:
        # Run the model on the input prompt
        _, state = gpt2(tokens)
        key_hook_value = state["intermediates"]["block_0"]["attn"]["key_hook"]
        attn_key = key_hook_value[0]
        assert attn_key.shape == (len(tokens), config.num_heads, config.head_dim)
        return attn_key


params = PretrainedGPT2Model.from_pretrained("gpt2").to_params()
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
std_config = PretrainedGPT2Model.make_config(decode=False)
decode_config = PretrainedGPT2Model.make_config(decode=True)
print("Loaded params and tokenizer")

print(jnp.sum(return_attn_key(std_config, tokenizer, params), axis=(1, 2)))
print(jnp.sum(return_attn_key(decode_config, tokenizer, params), axis=(1, 2)))
