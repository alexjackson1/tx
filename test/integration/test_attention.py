import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax.config import config

config.update("jax_enable_x64", True)


from jaxtyping import Array
from optax import Params

import pytest

import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from tx.models import PretrainedGPT2Model
from examples.params import tfs_attention_params

from tx.modules import MultiHeadAttention as TxAttention, TransformerConfig
from tx.params import tx_to_flax
from examples.tfs_attention import Attention as TFSAttention
from flax.linen import MultiHeadDotProductAttention as FlaxAttention


PRECISION = 1e-6


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
def flax_params(tx_params: Params, config: TransformerConfig) -> Params:
    return tx_to_flax(config, tx_params)


@pytest.fixture
def tfs_module(config: TransformerConfig) -> nn.Module:
    return TFSAttention(cfg=config)


@pytest.fixture
def tfs_params(gpt2: PretrainedGPT2Model, tx_params: Params) -> Params:
    return tfs_attention_params(gpt2.tx_config, tx_params)


@pytest.fixture
def hidden_states() -> Array:
    return jr.uniform(jr.PRNGKey(0), (4, 768))


@pytest.fixture
def causal_mask(hidden_states: Array) -> Array:
    return nn.make_causal_mask(jnp.ones((hidden_states.shape[0],)), dtype="bool")


def test_tx_attention_works(
    tx_module: TxAttention, tx_params: Params, hidden_states: Array
):
    output: Array = tx_module.apply({"params": tx_params}, hidden_states)
    assert output.shape == hidden_states.shape


def test_tx_attention_works_with_batches(
    tx_module: TxAttention, tx_params: Params, hidden_states: Array
):
    batched_input = jnp.expand_dims(hidden_states, axis=0)
    output: Array = tx_module.apply({"params": tx_params}, batched_input)
    assert output.shape[1:] == hidden_states.shape


def test_flax_attention_works(
    flax_module: nn.Module,
    flax_params: Params,
    hidden_states: Array,
    causal_mask: Array,
):
    query_length = hidden_states.shape[0]
    mask = causal_mask[:, :query_length, :query_length]
    output: Array = flax_module.apply(
        {"params": flax_params},
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        mask=mask,
    )
    assert output.shape == hidden_states.shape


def test_flax_attention_works_with_batches(
    flax_module: nn.Module,
    flax_params: Params,
    hidden_states: Array,
    causal_mask: Array,
):
    query_length = hidden_states.shape[0]
    mask = causal_mask[:, :query_length, :query_length]
    batched_input = jnp.expand_dims(hidden_states, axis=0)
    output: Array = flax_module.apply(
        {"params": flax_params},
        inputs_q=batched_input,
        inputs_kv=batched_input,
        mask=mask,
    )
    assert output.shape[1:] == hidden_states.shape


def test_tfs_attention_works_with_batches(
    tfs_module: nn.Module, tfs_params: Params, hidden_states: Array
):
    batched_input = jnp.expand_dims(hidden_states, axis=0)
    output: Array = tfs_module.apply({"params": tfs_params}, batched_input)
    assert output.shape[1:] == hidden_states.shape


def test_tx_versus_tfs(
    tfs_module: nn.Module,
    tx_module: TxAttention,
    tfs_params: Params,
    tx_params: Params,
    hidden_states: Array,
):
    batched_input = jnp.expand_dims(hidden_states, axis=0)
    tfs_output: Array = tfs_module.apply({"params": tfs_params}, batched_input)
    tx_output: Array = tx_module.apply({"params": tx_params}, batched_input)
    assert jnp.allclose(tfs_output, tx_output, atol=PRECISION, rtol=PRECISION)


def test_tx_versus_flax(
    flax_module: nn.Module,
    tx_module: TxAttention,
    flax_params: Params,
    tx_params: Params,
    hidden_states: Array,
    causal_mask: Array,
):
    query_length = hidden_states.shape[0]
    flax_output: Array = flax_module.apply(
        {"params": flax_params},
        hidden_states,
        hidden_states,
        mask=causal_mask[:, :query_length, :query_length],
    )
    tx_output: Array = tx_module.apply({"params": tx_params}, hidden_states)
    assert jnp.allclose(flax_output, tx_output, atol=PRECISION, rtol=PRECISION)


def test_tfs_versus_flax(
    flax_module: nn.Module,
    tfs_module: TFSAttention,
    flax_params: Params,
    tfs_params: Params,
    hidden_states: Array,
    causal_mask: Array,
):
    batched_input = jnp.expand_dims(hidden_states, axis=0)
    query_length = hidden_states.shape[0]
    flax_output: Array = flax_module.apply(
        {"params": flax_params},
        inputs_q=batched_input,
        inputs_kv=batched_input,
        mask=causal_mask[:, :query_length, :query_length],
    )
    tfs_output: Array = tfs_module.apply({"params": tfs_params}, batched_input)
    assert jnp.allclose(flax_output, tfs_output, atol=PRECISION, rtol=PRECISION)


if __name__ == "__main__":
    pytest.main([__file__])
