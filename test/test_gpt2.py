import sys
import os


sys.path.append(os.getcwd())

import pytest

import jax.numpy as jnp
from jaxtyping import Array

import flax.linen as nn

from tx.modules import (
    LayerNorm,
    Embed,
    PosEmbed,
    MultiHeadAttention,
    MLP,
    TransformerBlock,
    Unembed,
    Transformer,
)

from tx.models import PretrainedGPT2Model


@pytest.fixture
def gpt2_params():
    return PretrainedGPT2Model.from_pretrained("gpt2").to_params()


def test_layer_norm(gpt2_params):
    model = LayerNorm()
    variables = {"params": gpt2_params["ln_f"]}
    input_data = jnp.ones((1024, 768), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1024, 768)


def test_embed(gpt2_params):
    model = Embed(features=768, num_embeddings=50257)
    variables = {"params": gpt2_params["embed"]}
    input_data = jnp.ones((1024,), dtype=jnp.int32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1024, 768)


def test_pos_embed(gpt2_params):
    model = PosEmbed(features=768, num_embeddings=1024)
    variables = {"params": gpt2_params["pos_embed"]}
    input_data = jnp.ones((1024,), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1024, 768)


def test_attention(gpt2_params):
    model = MultiHeadAttention(num_heads=12, head_dim=64, features=768)
    variables = {"params": gpt2_params["block_0"]["attn"]}
    input_data = jnp.ones((1024, 768), dtype=jnp.float32)
    output: Array = model.apply(
        variables, input_data, mask=nn.make_causal_mask(jnp.ones(1024), dtype="bool")
    )
    assert output.shape == (1024, 768)


def test_mlp(gpt2_params):
    model = MLP(features=[3072, 768])
    variables = {"params": gpt2_params["block_0"]["mlp"]}
    input_data = jnp.ones((1024, 768), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1024, 768)


def test_transformer_block(gpt2_params):
    model = TransformerBlock(num_heads=12, head_dim=64, model_dim=768, mlp_dim=3072)
    variables = {"params": gpt2_params["block_0"]}
    input_data = jnp.ones((1024, 768), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1024, 768)


def test_unembed(gpt2_params):
    model = Unembed(features=768, num_embeddings=50257)
    variables = {"params": gpt2_params["unembed"]}
    input_data = jnp.ones((1024, 768), dtype=jnp.float32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1024, 50257)


def test_transformer(gpt2_params):
    model = Transformer()
    variables = {"params": gpt2_params}
    input_data = jnp.ones((1024,), dtype=jnp.int32)
    output: Array = model.apply(variables, input_data)
    assert output.shape == (1024, 50257)


if __name__ == "__main__":
    pytest.main([__file__])
