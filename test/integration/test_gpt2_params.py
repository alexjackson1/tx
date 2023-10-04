import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import jax.config

jax.config.update("jax_enable_x64", True)

from jaxtyping import Array

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from transformers import GPT2TokenizerFast
from transformers.models.gpt2.modeling_flax_gpt2 import (
    FlaxGPT2Block,
    GPT2Config,
    FlaxGPT2Attention,
)

from tx.modules import (
    MultiHeadAttention,
    Embed,
    LayerNorm,
    PosEmbed,
    TransformerBlock,
    Unembed,
)
from tx.models import PretrainedGPT2Model


@pytest.fixture
def gpt2():
    return PretrainedGPT2Model.from_pretrained("gpt2")


@pytest.fixture
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


INIT_FN = jax.nn.initializers.normal(stddev=0.02)


def tx_apply(module: nn.Module, params, input: Array) -> Array:
    variables = {"params": params}
    return module.apply(variables, input)


def hf_apply(module: nn.Module, params, input: Array) -> Array:
    input = jnp.expand_dims(input, axis=0)
    variables = {"params": params}
    output = module.apply(variables, input)
    return jnp.squeeze(output, axis=0)


def tx_embed(params, model_dim: int, vocab_dim: int, tokens: Array) -> Array:
    module = Embed(
        features=model_dim,
        num_embeddings=vocab_dim,
        param_dtype=jnp.float64,
    )
    return tx_apply(module, params["embed"], tokens)


def hf_embed(params, model_dim: int, vocab_dim: int, tokens: Array) -> Array:
    module = nn.Embed(
        vocab_dim,
        model_dim,
        embedding_init=INIT_FN,
        param_dtype=jnp.float64,
    )
    return hf_apply(module, params["transformer"]["wte"], tokens)


def tx_pos_embed(params, model_dim: int, context_length: int, tokens: Array) -> Array:
    module = PosEmbed(
        features=model_dim,
        num_embeddings=context_length,
        param_dtype=jnp.float64,
    )
    return tx_apply(module, params["pos_embed"], tokens)


def hf_pos_embed(params, model_dim: int, context_length: int, tokens: Array) -> Array:
    module = nn.Embed(
        context_length,
        model_dim,
        embedding_init=INIT_FN,
        param_dtype=jnp.float64,
    )
    return hf_apply(module, params["transformer"]["wpe"], tokens)


def tx_transformer_block(
    params,
    name: str,
    num_heads: int,
    head_dim: int,
    model_dim: int,
    mlp_dim: int,
    input: Array,
) -> Array:
    module = TransformerBlock(
        num_heads=num_heads,
        head_dim=head_dim,
        model_dim=model_dim,
        mlp_dim=mlp_dim,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    return tx_apply(module, params[name], input)


def hf_transformer_block(params, name: str, input: Array) -> Array:
    module = FlaxGPT2Block(
        GPT2Config(resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False),
        dtype=jnp.float64,
    )
    input = jnp.expand_dims(input, axis=0)
    variables = {"params": params["transformer"]["h"][name]}
    output = module.apply(variables, input)[0]
    return jnp.squeeze(output, axis=0)


def tx_attention(params, name: str, input: Array) -> Array:
    module = MultiHeadAttention(
        features=768,
        num_heads=12,
        head_dim=64,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    variables = {"params": params[name]["attn"]}
    output = module.apply(
        variables, input, nn.make_causal_mask(jnp.ones(input.shape[:-1]), dtype="bool")
    )
    return output


def hf_attention(params, name: str, input: Array) -> Array:
    module = FlaxGPT2Attention(
        GPT2Config(resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False),
        dtype=jnp.float64,
    )
    input = jnp.expand_dims(input, axis=0)
    variables = {"params": params["transformer"]["h"][name]["attn"]}
    output = module.apply(variables, input)[0]
    return jnp.squeeze(output, axis=0)


def tx_ln_f(gpt2_params, input: Array) -> Array:
    module = LayerNorm(epsilon=1e-5, dtype=jnp.float64, param_dtype=jnp.float64)
    return tx_apply(module, gpt2_params["ln_f"], input)


def hf_ln_f(gpt2_params, input: Array) -> Array:
    module = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float64, param_dtype=jnp.float64)
    return hf_apply(module, gpt2_params["transformer"]["ln_f"], input)


def tx_unembed(gpt2_params, model_dim: int, vocab_dim: int, input: Array) -> Array:
    module = Unembed(
        features=model_dim,
        num_embeddings=vocab_dim,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
    )
    return tx_apply(module, gpt2_params["unembed"], input)


def hf_unembed(gpt2_params, model_dim: int, vocab_dim: int, input: Array) -> Array:
    module = nn.Dense(
        vocab_dim,
        model_dim,
        kernel_init=INIT_FN,
        bias_init=nn.initializers.zeros,
        dtype=jnp.float64,
        param_dtype=jnp.float64,
        precision=None,
    )
    params = {
        "kernel": jnp.transpose(gpt2_params["transformer"]["wte"]["embedding"]),
        "bias": jnp.zeros((vocab_dim,)),
    }
    return hf_apply(module, params, input)


def test_attn_with_gpt2_params(gpt2: PretrainedGPT2Model):
    gpt2_params = gpt2.to_params()

    input = jr.uniform(jax.random.PRNGKey(0), (1024, 768), dtype=jnp.float64)

    tx_output = tx_attention(gpt2_params, "block_0", input)
    hf_output = hf_attention(gpt2._params, "0", input)

    assert tx_output.shape == hf_output.shape
    assert jnp.allclose(tx_output, hf_output, atol=1e-6, rtol=1e-6)


def test_with_gpt2_params(gpt2: PretrainedGPT2Model, tokenizer: GPT2TokenizerFast):
    gpt2_params = gpt2.to_params()
    gpt2_config = gpt2.tx_config

    tokens: Array = tokenizer("Hello, my name is", return_tensors="jax")["input_ids"]
    tokens = jnp.squeeze(tokens, axis=0)
    (seq_length,) = tokens.shape

    vocab_dim, model_dim = gpt2_config.vocab_dim, gpt2_config.model_dim
    context_length = gpt2_config.context_length

    # Embedding
    tx_embed_out = tx_embed(gpt2_params, model_dim, vocab_dim, tokens)
    hf_embed_out = hf_embed(gpt2._params, model_dim, vocab_dim, tokens)
    assert tx_embed_out.shape == hf_embed_out.shape
    assert jnp.allclose(tx_embed_out, hf_embed_out, atol=1e-6, rtol=1e-6)

    # Positional embedding
    tx_pos_embed_out = tx_pos_embed(gpt2_params, model_dim, context_length, tokens)
    hf_pos_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4"), (seq_length,))
    hf_pos_embed_out = hf_pos_embed(gpt2._params, model_dim, context_length, hf_pos_ids)
    assert tx_pos_embed_out.shape == hf_pos_embed_out.shape
    assert jnp.allclose(tx_pos_embed_out, hf_pos_embed_out, atol=1e-6, rtol=1e-6)

    # Transformer blocks
    next_input = tx_embed_out + tx_pos_embed_out
    for i in range(gpt2_config.num_layers):
        ## Transformer block
        tx_block_out = tx_transformer_block(
            gpt2_params,
            f"block_{i}",
            num_heads=gpt2_config.num_heads,
            head_dim=gpt2_config.head_dim,
            model_dim=gpt2_config.model_dim,
            mlp_dim=gpt2_config.mlp_dim,
            input=next_input,
        )
        hf_block_out = hf_transformer_block(gpt2._params, f"{i}", next_input)
        assert tx_block_out.shape == hf_block_out.shape
        assert jnp.allclose(tx_block_out, hf_block_out, atol=1e-6, rtol=1e-6)

        ## Update inputs
        next_input = tx_block_out

    # Layer norm
    tx_ln_f_out = tx_ln_f(gpt2_params, next_input)
    hf_ln_f_out = hf_ln_f(gpt2._params, next_input)
    assert tx_ln_f_out.shape == hf_ln_f_out.shape
    assert jnp.allclose(tx_ln_f_out, hf_ln_f_out, atol=1e-6, rtol=1e-6)

    # Unembedding
    tx_unembed_out = tx_unembed(gpt2_params, model_dim, vocab_dim, tx_ln_f_out)
    hf_unembed_out = hf_unembed(gpt2._params, model_dim, vocab_dim, hf_ln_f_out)
    assert tx_unembed_out.shape == hf_unembed_out.shape
    assert jnp.allclose(tx_unembed_out, hf_unembed_out, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
