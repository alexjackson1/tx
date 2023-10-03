import os, sys

from optax import Params

# Update sys.path to include the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set jax_enable_x64=True to enable float64 precision
from jax.config import config

config.update("jax_enable_x64", True)

# Main script begins
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array

from transformers import GPT2TokenizerFast

from tx.hooks import HookMap, HookPoint, StoreHook
from tx.models import PretrainedGPT2Model
from tx.network import GenerativeModel

# Load the model config, params, and tokenizer
model_class = PretrainedGPT2Model
config = model_class.make_config(decode=True)
params = model_class.from_pretrained("gpt2").to_params()

tokenizer_class = GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# Define a hook to store intermediate activations
def store_hook(x: Array, hook_point: HookPoint, module: nn.Module, **kwargs):
    if module is not None and hook_point is not None:
        module.sow("intermediates", hook_point.value, x)
    return x


# Create a generative model storing the params and hooks
hooks, collections = HookMap(residual=StoreHook), ["intermediates"]
gpt2 = GenerativeModel(config, tokenizer, params, hooks, collections)

# Create an input prompt
reference_text = "Hello, I am"
tokens = gpt2.to_tokens(reference_text, prepend_bos=True)

# Run the model on the input prompt
# logits, state = gpt2(tokens)
# probs: Array = jax.nn.softmax(logits, axis=-1)
# next_token = jnp.argmax(logits[-1], axis=-1, keepdims=True)
# next_char = gpt2.to_str(next_token)
# print(repr(next_char))


print("Full Prompt:", gpt2.to_str(tokens))


def log_cache(cache: Params):
    cached_key: Array = cache["block_0"]["attn"]["cached_key"]
    cached_idx: Array = cache["block_0"]["attn"]["cached_idx"]
    print(cached_key.shape)
    print("Non-empty keys", sum(jnp.count_nonzero(cached_key, axis=(1, 2)) > 0))
    print("Index value", cached_idx)
    print("Last key", sum(cached_key[cached_idx - 1]))


for token_idx in range(len(tokens)):
    token = tokens[None, token_idx]
    print(gpt2.to_str(token))

    logits, state = gpt2(token)
    log_cache(state["cache"])

# probs = jax.nn.softmax(logits, axis=-1)
# next_token = jnp.argmax(logits[-1], axis=-1, keepdims=True)
# print(gpt2.to_str(next_token))

# for i in range(10):
#     # Define new input sequence, by appending the previously generated token
#     # new_tokens = jnp.concatenate([tokens, next_token], axis=-1)
#     print(gpt2.to_str(next_token))
#     # Pass our new sequence through the model, to get new output
#     logits, state = gpt2(next_token)
#     log_cache(state["cache"])
#     # Get the predicted token at the end of our sequence
#     next_token = jnp.argmax(logits[-1], axis=-1, keepdims=True)


# test_string = """The Total Perspective Vortex derives its picture of the whole Universe on the principle of"""
# print(test_string, end="", flush=True)
# for i in range(10):
#     test_tokens = jnp.expand_dims(gpt2.to_tokens(test_string), axis=0)
#     demo_logits = demo_gpt2.apply(
#         {"params": tfs_transformer_params(demo_cfg, gpt2_params)}, test_tokens
#     )
#     next_string = gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())
#     print(next_string, end="", flush=True)
#     test_string += next_string
