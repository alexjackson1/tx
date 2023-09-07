import sys, os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxtyping import Array

import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn

from transformers import GPT2TokenizerFast
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Block, GPT2Config

from tx.modules import (
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


def test_with_gpt2_params(gpt2: PretrainedGPT2Model, tokenizer: GPT2TokenizerFast):
    gpt2_params = gpt2.to_params()
    gpt2_config = gpt2.tx_config

    tokens: Array = tokenizer("Hello, my name is", return_tensors="jax")["input_ids"]
    batch_size, seq_length = tokens.shape

    init_fn = jax.nn.initializers.normal(stddev=0.02)
    vocab_dim, model_dim = gpt2_config.vocab_dim, gpt2_config.model_dim
    context_length = gpt2_config.context_length

    # Embedding
    ## Transformer embedding
    tx_embed_mod = Embed(features=model_dim, num_embeddings=vocab_dim)
    tx_embed_params = gpt2_params["embed"]
    tx_embed_vars = {"params": tx_embed_params}
    tx_embed_out: Array = tx_embed_mod.apply(tx_embed_vars, tokens)

    ## Control embedding
    hf_embed_mod = nn.Embed(
        vocab_dim, model_dim, embedding_init=init_fn, dtype=jnp.float32
    )
    hf_embed_params = {"params": gpt2._params["transformer"]["wte"]}
    hf_embed_out: Array = hf_embed_mod.apply(hf_embed_params, tokens)

    ## Compare embeddings
    assert tx_embed_out.shape == hf_embed_out.shape
    assert jnp.allclose(tx_embed_out, hf_embed_out, atol=1e-6, rtol=1e-6)

    # Positional embedding
    ## Transformer positional embedding
    tx_pos_embed_mod = PosEmbed(features=model_dim, num_embeddings=context_length)
    tx_pos_embed_params = gpt2_params["pos_embed"]
    tx_pos_embed_vars = {"params": tx_pos_embed_params}
    tx_pos_embed_out: Array = tx_pos_embed_mod.apply(tx_pos_embed_vars, tokens)

    ## Control positional embedding
    hf_pos_embed_mod = nn.Embed(
        context_length, model_dim, embedding_init=init_fn, dtype=jnp.float32
    )
    hf_pos_embed_params = {"params": gpt2._params["transformer"]["wpe"]}
    hf_pos_ids = jnp.broadcast_to(
        jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
    )
    hf_pos_embed_out: Array = hf_pos_embed_mod.apply(hf_pos_embed_params, hf_pos_ids)

    ## Compare positional embeddings
    assert tx_pos_embed_out.shape == hf_pos_embed_out.shape
    assert jnp.allclose(tx_pos_embed_out, hf_pos_embed_out, atol=1e-6, rtol=1e-6)

    # Transformer blocks
    tx_next_input = tx_embed_out + tx_pos_embed_out
    hf_next_input = hf_embed_out + hf_pos_embed_out
    for i in range(12):
        ## Transformer block
        tx_block_mod = TransformerBlock(
            num_heads=gpt2_config.num_heads,
            head_dim=gpt2_config.head_dim,
            model_dim=model_dim,
            mlp_dim=gpt2_config.mlp_dim,
            context_length=context_length,
        )
        tx_block_params = gpt2_params[f"block_{i}"]
        tx_block_vars = {"params": tx_block_params}
        tx_block_out: Array = tx_block_mod.apply(tx_block_vars, tx_next_input)

        ## Control block
        hf_block_mod = FlaxGPT2Block(GPT2Config.from_pretrained("gpt2"))
        hf_block_params = {"params": gpt2._params["transformer"]["h"][f"{i}"]}
        hf_block_out: Array = hf_block_mod.apply(hf_block_params, hf_next_input)[0]

        ## Compare blocks
        assert tx_block_out.shape == hf_block_out.shape
        assert jnp.allclose(tx_block_out, hf_block_out, atol=1e-2, rtol=1e-2)

        ## Update inputs
        tx_next_input = tx_block_out
        hf_next_input = hf_block_out

    # Layer norm
    ## Transformer layer norm
    tx_ln_f_mod = LayerNorm(epsilon=gpt2_config.layer_norm_eps)
    tx_ln_f_params = gpt2_params["ln_f"]
    tx_ln_f_vars = {"params": tx_ln_f_params}
    tx_ln_f_out: Array = tx_ln_f_mod.apply(tx_ln_f_vars, tx_next_input)

    ## Control layer norm
    hf_ln_f_mod = nn.LayerNorm(epsilon=gpt2_config.layer_norm_eps)
    hf_ln_f_params = {"params": gpt2._params["transformer"]["ln_f"]}
    # Note we use the tx input here to ensure a close comparison
    hf_ln_f_out: Array = hf_ln_f_mod.apply(hf_ln_f_params, tx_next_input)

    ## Compare layer norms
    assert tx_ln_f_out.shape == hf_ln_f_out.shape
    assert jnp.allclose(tx_ln_f_out, hf_ln_f_out, atol=1e-6, rtol=1e-6)

    # Unembedding
    ## Transformer unembedding
    tx_unembed_mod = Unembed(features=model_dim, num_embeddings=vocab_dim)
    tx_unembed_params = gpt2_params["unembed"]
    tx_unembed_vars = {"params": tx_unembed_params}
    tx_unembed_out: Array = tx_unembed_mod.apply(tx_unembed_vars, tx_ln_f_out)

    ## Control unembedding
    hf_unembed_mod = nn.Dense(
        vocab_dim,
        model_dim,
        kernel_init=init_fn,
        bias_init=nn.initializers.zeros,
        dtype=jnp.float32,
        precision=None,
    )
    hf_unembed_params = {
        "params": {
            "kernel": jnp.transpose(gpt2._params["transformer"]["wte"]["embedding"]),
            "bias": jnp.zeros((vocab_dim,)),
        }
    }
    hf_unembed_out: Array = hf_unembed_mod.apply(hf_unembed_params, hf_ln_f_out)

    ## Compare unembeddings
    assert tx_unembed_out.shape == hf_unembed_out.shape
    assert jnp.allclose(tx_unembed_out, hf_unembed_out, atol=1e-6, rtol=1e-6)
