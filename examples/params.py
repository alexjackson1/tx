from typing import Dict, Union
from jaxtyping import Array

import jax.numpy as jnp

from tx.modules import TransformerConfig


ArrayOrDict = Union[Array, Dict[str, "ArrayOrDict"]]
Params = Dict[str, ArrayOrDict]


def flax_attention_params(cfg: TransformerConfig, params: Params) -> Params:
    model_dim, num_heads, head_dim = (cfg.model_dim, cfg.num_heads, cfg.head_dim)

    def split_attn_params(c_attn: Params) -> Params:
        rs_kernel = lambda x: jnp.reshape(x, (model_dim, num_heads, head_dim))
        q, k, v = jnp.split(c_attn["kernel"], 3, axis=-1)
        q, k, v = rs_kernel(q), rs_kernel(k), rs_kernel(v)

        rs_bias = lambda x: jnp.reshape(x, (num_heads, head_dim))
        q_b, k_b, v_b = jnp.split(c_attn["bias"], 3, axis=-1)
        q_b, k_b, v_b = rs_bias(q_b), rs_bias(k_b), rs_bias(v_b)

        return {
            "query": {"kernel": q, "bias": q_b},
            "key": {"kernel": k, "bias": k_b},
            "value": {"kernel": v, "bias": v_b},
        }

    def reshape_proj_params(c_proj: Params) -> Params:
        out_k = jnp.reshape(c_proj["kernel"], (num_heads, head_dim, model_dim))
        out_b = jnp.reshape(c_proj["bias"], (model_dim,))
        return {"out": {"kernel": out_k, "bias": out_b}}

    attn = params["block_0"]["attn"]
    c_attn, c_proj = attn["c_attn"], attn["c_proj"]
    return {**split_attn_params(c_attn), **reshape_proj_params(c_proj)}


def tfs_attention_params(cfg: TransformerConfig, params: Params) -> Params:
    model_dim, num_heads, head_dim = (cfg.model_dim, cfg.num_heads, cfg.head_dim)

    def split_attn_params(c_attn: Params) -> Params:
        rs_kernel = lambda x: jnp.reshape(x, (model_dim, num_heads, head_dim))
        rs_kernel_1 = lambda x: jnp.transpose(rs_kernel(x), (1, 0, 2))
        q, k, v = jnp.split(c_attn["kernel"], 3, axis=-1)
        q, k, v = rs_kernel_1(q), rs_kernel_1(k), rs_kernel_1(v)

        rs_bias = lambda x: jnp.reshape(x, (num_heads, head_dim))
        q_b, k_b, v_b = jnp.split(c_attn["bias"], 3, axis=-1)
        q_b, k_b, v_b = rs_bias(q_b), rs_bias(k_b), rs_bias(v_b)

        return {"W_Q": q, "W_K": k, "W_V": v, "b_Q": q_b, "b_K": k_b, "b_V": v_b}

    def reshape_proj_params(c_proj: Params) -> Params:
        out_k = jnp.reshape(c_proj["kernel"], (num_heads, head_dim, model_dim))
        out_b = jnp.reshape(c_proj["bias"], (model_dim,))
        return {"W_O": out_k, "b_O": out_b}

    c_attn, c_proj = params["c_attn"], params["c_proj"]
    return {**split_attn_params(c_attn), **reshape_proj_params(c_proj)}


def tfs_layer_norm_params(_cfg, params: Params) -> Params:
    return {"w": params["scale"], "b": params["bias"]}


def tfs_embed_params(_cfg, params: Params) -> Params:
    return {"W_E": params["embedding"]}


def tfs_pos_embed_params(_cfg, params: Params) -> Params:
    return {"W_pos": params["embedding"]}


def tfs_mlp_params(_cfg, params: Params) -> Params:
    return {
        "W_in": params["fc_1"]["kernel"],
        "W_out": params["proj"]["kernel"],
        "b_in": params["fc_1"]["bias"],
        "b_out": params["proj"]["bias"],
    }


def tfs_block_params(cfg, params: Params) -> Params:
    return {
        "ln1": tfs_layer_norm_params(cfg, params["ln_1"]),
        "attn": tfs_attention_params(cfg, params["attn"]),
        "ln2": tfs_layer_norm_params(cfg, params["ln_2"]),
        "mlp": tfs_mlp_params(cfg, params["mlp"]),
    }


def tfs_unembed_params(_cfg, params: Params) -> Params:
    return {"W_U": params["kernel"], "b_U": params["bias"]}


def tfs_transformer_params(cfg: TransformerConfig, params: Params) -> Params:
    return {
        "embed": tfs_embed_params(cfg, params["embed"]),
        "pos_embed": tfs_pos_embed_params(cfg, params["pos_embed"]),
        "ln_final": tfs_layer_norm_params(cfg, params["ln_f"]),
        "unembed": tfs_unembed_params(cfg, params["unembed"]),
        **{
            f"block_{i}": tfs_block_params(cfg, params[f"block_{i}"])
            for i in range(cfg.num_layers)
        },
    }
