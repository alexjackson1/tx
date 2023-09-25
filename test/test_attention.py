import sys, os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
from jaxtyping import Array, Float

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import einops

import tx.modules as tx
from tx.models import PretrainedGPT2Model
from examples.params import tfs_attention_params
from tx.modules.transformer import TransformerConfig


@pytest.fixture
def gpt2() -> PretrainedGPT2Model:
    return PretrainedGPT2Model.from_pretrained("gpt2")


@pytest.fixture
def tx_config(gpt2: PretrainedGPT2Model) -> tx.TransformerConfig:
    return gpt2.tx_config


@pytest.fixture
def tx_module(tx_config: tx.TransformerConfig) -> tx.Attention:
    return tx.Attention(
        num_heads=tx_config.num_heads,
        head_dim=tx_config.head_dim,
        model_dim=tx_config.model_dim,
        init_range=tx_config.init_range,
    )


@pytest.fixture
def tx_params(gpt2: PretrainedGPT2Model) -> Dict[str, Array]:
    return gpt2.to_params()["block_0"]["attn"]


@pytest.fixture
def flax_params(
    tx_params: Dict[str, Array], tx_config: tx.TransformerConfig
) -> Dict[str, Array]:
    c_attn, c_proj = tx_params["c_attn"], tx_params["c_proj"]
    num_heads, head_dim, model_dim = (
        tx_config.num_heads,
        tx_config.head_dim,
        tx_config.model_dim,
    )

    qkv_kernel = jnp.split(c_attn["kernel"], 3, axis=-1)
    reshape_kernel = lambda a: jnp.reshape(
        a, (qkv_kernel[0].shape[0], num_heads, head_dim)
    )
    q_kernel, k_kernel, v_kernel = tuple(map(reshape_kernel, qkv_kernel))
    o_kernel = jnp.reshape(c_proj["kernel"], (num_heads, head_dim, model_dim))

    qkv_bias = jnp.split(c_attn["bias"], 3, axis=-1)
    reshape_bias = lambda a: jnp.reshape(a, (num_heads, head_dim))
    q_bias, k_bias, v_bias = tuple(map(reshape_bias, qkv_bias))
    o_bias = c_proj["bias"]

    flax_params = {}
    flax_params["query"] = {"kernel": q_kernel, "bias": q_bias}
    flax_params["key"] = {"kernel": k_kernel, "bias": k_bias}
    flax_params["value"] = {"kernel": v_kernel, "bias": v_bias}
    flax_params["out"] = {"kernel": o_kernel, "bias": o_bias}

    return flax_params


@pytest.fixture
def flax_module(tx_config: tx.TransformerConfig) -> nn.Module:
    return nn.MultiHeadDotProductAttention(
        num_heads=tx_config.num_heads,
        qkv_features=tx_config.model_dim,
        out_features=tx_config.model_dim,
        dropout_rate=0.0,
        deterministic=True,
    )


@pytest.fixture
def tfs_module(tx_config) -> nn.Module:
    class TFSAttention(nn.Module):
        cfg: TransformerConfig

        def setup(self):
            init_fn = nn.initializers.normal(stddev=self.cfg.init_range)
            qkv_kernel_shape = (
                self.cfg.num_heads,
                self.cfg.model_dim,
                self.cfg.head_dim,
            )
            self.W_Q = self.param("W_Q", init_fn, qkv_kernel_shape)
            self.W_K = self.param("W_K", init_fn, qkv_kernel_shape)
            self.W_V = self.param("W_V", init_fn, qkv_kernel_shape)
            self.W_O = self.param(
                "W_O",
                init_fn,
                (qkv_kernel_shape[0], qkv_kernel_shape[2], qkv_kernel_shape[1]),
            )

            qkv_bias_shape = (self.cfg.num_heads, self.cfg.head_dim)
            self.b_Q = self.param("b_Q", nn.initializers.zeros, qkv_bias_shape)
            self.b_K = self.param("b_K", nn.initializers.zeros, qkv_bias_shape)
            self.b_V = self.param("b_V", nn.initializers.zeros, qkv_bias_shape)
            self.b_O = self.param("b_O", nn.initializers.zeros, (self.cfg.model_dim,))

            self.IGNORE = jnp.array(-1e5, dtype=jnp.float32)

        def __call__(
            self, normalized_resid_pre: Float[Array, "batch seq model"]
        ) -> Float[Array, "batch seq model"]:
            # Calculate query, key and value vectors
            q = (
                einops.einsum(
                    normalized_resid_pre,
                    self.W_Q,
                    "batch seq model, n_head model h_dim -> batch seq n_head h_dim",
                )
                + self.b_Q
            )
            k = (
                einops.einsum(
                    normalized_resid_pre,
                    self.W_K,
                    "batch seq model, n_head model h_dim -> batch seq n_head h_dim",
                )
                + self.b_K
            )
            v = (
                einops.einsum(
                    normalized_resid_pre,
                    self.W_V,
                    "batch seq model, n_head model h_dim -> batch seq n_head h_dim",
                )
                + self.b_V
            )

            # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
            attn_scores = einops.einsum(
                q,
                k,
                "batch seq_q n_head h_dim, batch seq_k n_head h_dim -> batch n_head seq_q seq_k",
            )
            attn_scores_masked = self.apply_causal_mask(
                attn_scores / self.cfg.head_dim**0.5
            )
            attn_pattern = jax.nn.softmax(attn_scores_masked, axis=-1)

            # Take weighted sum of value vectors, according to attention probabilities
            z = einops.einsum(
                v,
                attn_pattern,
                "batch seq_k n_head h_dim, batch n_head seq_q seq_k -> batch seq_q n_head h_dim",
            )

            # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
            attn_out = (
                einops.einsum(
                    z,
                    self.W_O,
                    "batch seq_q n_head h_dim, n_head h_dim model -> batch seq_q model",
                )
                + self.b_O
            )

            return attn_out

        def apply_causal_mask(
            self, attn_scores: Float[Array, "batch n_head seq_q seq_k"]
        ) -> Float[Array, "batch n_head seq_q seq_k"]:
            """
            Applies a causal mask to attention scores, and returns masked scores.
            """
            # Define a mask that is True for all positions we want to set probabilities to zero for
            all_ones = jnp.ones((attn_scores.shape[-2], attn_scores.shape[-1]))
            mask = jnp.triu(all_ones, k=1)
            # Apply the mask to attention scores, then return the masked scores
            # attn_scores.masked_fill_(mask, self.IGNORE)
            attn_scores = jnp.where(mask, self.IGNORE, attn_scores)
            return attn_scores

    return TFSAttention(tx_config)


@pytest.fixture
def tfs_params(gpt2: PretrainedGPT2Model, tx_params) -> Dict[str, Array]:
    return tfs_attention_params(gpt2.tx_config, tx_params)


@pytest.fixture
def residual() -> Array:
    return jr.uniform(jr.PRNGKey(0), (4, 768))


@pytest.fixture
def batch_residual(residual) -> Array:
    return jnp.expand_dims(residual, axis=0)


def test_tx_attention_works(
    tx_module: tx.Attention, tx_params: Dict[str, Array], residual: Array
):
    output: Array = tx_module.apply({"params": tx_params}, residual)
    assert output.shape == residual.shape


def test_tfs_attention_works(
    tfs_module: nn.Module, tfs_params: Dict[str, Array], batch_residual: Array
):
    output: Array = tfs_module.apply({"params": tfs_params}, batch_residual)
    assert output.shape == batch_residual.shape


def test_tx_versus_tfs(
    tfs_module: nn.Module,
    tx_module: tx.Attention,
    tfs_params: Dict[str, Array],
    tx_params: Dict[str, Array],
    residual: Array,
    batch_residual: Array,
):
    tfs_output: Array = tfs_module.apply({"params": tfs_params}, batch_residual)
    tx_output: Array = tx_module.apply({"params": tx_params}, residual)
    assert jnp.allclose(tfs_output, tx_output, atol=1e-2, rtol=1e-2)


def test_tx_versus_flax(
    flax_module: nn.Module,
    tx_module: tx.Attention,
    flax_params: Dict[str, Array],
    tx_params: Dict[str, Array],
    residual: Array,
):
    mask = nn.make_causal_mask(
        jnp.ones((residual.shape[0],), dtype="bool"),
        dtype="bool",
    )
    query_length = residual.shape[0]
    mask = mask[:, :query_length, :query_length]
    flax_output: Array = flax_module.apply(
        {"params": flax_params}, residual, residual, mask=mask
    )
    tx_output: Array = tx_module.apply({"params": tx_params}, residual)
    assert jnp.allclose(flax_output, tx_output, atol=1e-2, rtol=1e-2)
