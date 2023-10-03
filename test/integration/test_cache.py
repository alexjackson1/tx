from typing import Tuple
from jax.config import config

config.update("jax_enable_x64", True)


import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from jaxtyping import Array

import pytest

import jax.random as jr
import jax.numpy as jnp
from optax import Params

from tx import TransformerConfig, Transformer


@pytest.fixture
def rng():
    return jr.PRNGKey(0)


@pytest.fixture
def config():
    return TransformerConfig(decode=True)


@pytest.fixture
def transformer(config: TransformerConfig) -> Transformer:
    return Transformer.from_config(config)


@pytest.fixture
def transformer_variables(rng, transformer: Transformer) -> Params:
    return transformer.init(rng, jnp.ones(1024, dtype=jnp.int32))


def test_transformer_works_with_blank_cache(
    config: TransformerConfig, transformer: Transformer, transformer_variables: Params
):
    *batch_dims, seq_length = (1,)
    inputs = jnp.ones((*batch_dims, seq_length), dtype=jnp.int64)
    outputs = transformer.apply(transformer_variables, inputs, mutable=["cache"])
    output: Array = outputs[0]
    state: Params = outputs[1]
    assert output.shape == (*batch_dims, seq_length, config.vocab_dim)
    assert len(state["cache"]) == config.num_layers
    assert state["cache"]["block_0"]["attn"]["cached_key"].shape == (1024, 12, 64)


def test_transformer_works_with_blank_cache_many_tokens(
    rng,
    config: TransformerConfig,
    transformer: Transformer,
    transformer_variables: Params,
):
    ex_tokens = jr.randint(rng, (10,), 0, config.vocab_dim)
    *batch_dims, seq_length = ex_tokens.shape

    for i in range(seq_length):
        input = ex_tokens[i : i + 1]
        outputs = transformer.apply(transformer_variables, input, mutable=["cache"])
        output: Array = outputs[0]
        state: Params = outputs[1]
        assert output.shape == (*batch_dims, 1, config.vocab_dim)
        assert len(state["cache"]) == config.num_layers
        assert state["cache"]["block_0"]["attn"]["cached_key"].shape == (1024, 12, 64)


if __name__ == "__main__":
    pytest.main([__file__])
