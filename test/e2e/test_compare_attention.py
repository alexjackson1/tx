import os, sys
from typing import Sequence

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from jax.config import config

config.update("jax_enable_x64", True)

from jaxtyping import Array

import pytest

import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from tx.models import PretrainedGPT2Model
from tx.tree_utils import Params

from examples.params import tfs_attention_params

import param_conversion as convert_params

# Compare three implementations of MultiHeadAttention
from examples.tfs_attention import Attention as TFSAttention
from flax.linen import MultiHeadDotProductAttention as FlaxAttention
from tx import TransformerConfig, MultiHeadAttention as TxAttention

PRECISION = 1e-6


@pytest.fixture
def rng() -> Array:
    return jr.PRNGKey(0)


@pytest.fixture
def gpt2() -> PretrainedGPT2Model:
    return PretrainedGPT2Model.from_pretrained("gpt2")


@pytest.fixture
def config(gpt2: PretrainedGPT2Model) -> TransformerConfig:
    return gpt2.tx_config


@pytest.fixture
def tx_module(config: TransformerConfig) -> TxAttention:
    return TxAttention(
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        features=config.model_dim,
        init_range=config.init_range,
    )


@pytest.fixture
def tx_params(gpt2: PretrainedGPT2Model) -> Params:
    return gpt2.to_params()["block_0"]["attn"]


@pytest.fixture
def flax_module(config: TransformerConfig) -> nn.Module:
    return FlaxAttention(
        num_heads=config.num_heads,
        qkv_features=config.model_dim,
        out_features=config.model_dim,
        dropout_rate=0.0,
        deterministic=True,
    )


@pytest.fixture
def flax_params(config: TransformerConfig, tx_params: Params) -> Params:
    return convert_params.to_flax(config, tx_params)


@pytest.fixture
def tfs_module(config: TransformerConfig) -> nn.Module:
    return TFSAttention(cfg=config)


@pytest.fixture
def tfs_params(gpt2: PretrainedGPT2Model, tx_params: Params) -> Params:
    return tfs_attention_params(gpt2.tx_config, tx_params)


def format_ids(val):
    return f"batch_dims={val}"


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,)], ids=format_ids)
def test_compare_tfs_implementation(
    rng: Array,
    tfs_module: nn.Module,
    tx_module: TxAttention,
    tfs_params: Params,
    tx_params: Params,
    batch_dims: Sequence[int],
):
    SEQ_LEN = 4
    random_input = jr.uniform(rng, (*batch_dims, SEQ_LEN, 768))

    # Transformer from scratch implementation (must have batch)
    if len(batch_dims) == 0:
        batched_input = jnp.expand_dims(random_input, axis=0)
        tfs_output: Array = tfs_module.apply({"params": tfs_params}, batched_input)[0]
    else:
        tfs_output: Array = tfs_module.apply({"params": tfs_params}, random_input)

    # Implementation from tx
    tx_output: Array = tx_module.apply({"params": tx_params}, random_input)

    assert jnp.allclose(tfs_output, tx_output, atol=PRECISION, rtol=PRECISION)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_ids)
def test_compare_flax_implementation(
    rng: Array,
    flax_module: nn.Module,
    tx_module: TxAttention,
    flax_params: Params,
    tx_params: Params,
    batch_dims: Sequence[int],
):
    SEQ_LEN = 4
    random_input = jr.uniform(rng, (*batch_dims, SEQ_LEN, 768))
    query_length = random_input.shape[-2]
    mask = nn.make_causal_mask(jnp.ones((SEQ_LEN,)), dtype="bool")

    # Flax implementation
    flax_output: Array = flax_module.apply(
        {"params": flax_params},
        random_input,
        random_input,
        mask=mask[:, :query_length, :query_length],
    )

    # Implementation from tx
    tx_output: Array = tx_module.apply({"params": tx_params}, random_input)

    assert jnp.allclose(flax_output, tx_output, atol=PRECISION, rtol=PRECISION)


@pytest.mark.skip("Does not test tx implementation, sanity check")
@pytest.mark.parametrize("batch_dims", [(), (1,), (2,)], ids=format_ids)
def test_compare_flax_with_tfs(
    rng: Array,
    flax_module: nn.Module,
    tfs_module: TFSAttention,
    flax_params: Params,
    tfs_params: Params,
    batch_dims: Sequence[int],
):
    SEQ_LEN = 4
    random_input = jr.uniform(rng, (*batch_dims, SEQ_LEN, 768))
    query_length = random_input.shape[-2]
    mask = nn.make_causal_mask(jnp.ones((SEQ_LEN,)), dtype="bool")

    # Flax implementation
    flax_output: Array = flax_module.apply(
        {"params": flax_params},
        random_input,
        random_input,
        mask=mask[:, :query_length, :query_length],
    )

    # Transformer from scratch implementation (must have batch)
    if len(batch_dims) == 0:
        batched_input = jnp.expand_dims(random_input, axis=0)
        tfs_output: Array = tfs_module.apply({"params": tfs_params}, batched_input)[0]
    else:
        tfs_output: Array = tfs_module.apply({"params": tfs_params}, random_input)

    assert jnp.allclose(flax_output, tfs_output, atol=PRECISION, rtol=PRECISION)
