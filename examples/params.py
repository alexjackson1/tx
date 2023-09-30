import jax.numpy as jnp
from optax import Params

from tx.modules import TransformerConfig


def tfs_attention_params(cfg: TransformerConfig, params: Params) -> Params:
    """Convert `tx` attention parameters to `tfs` attention parameters."""
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
    """Convert `tx` layer norm parameters to `tfs` layer norm parameters."""
    return {"w": params["scale"], "b": params["bias"]}


def tfs_embed_params(_cfg, params: Params) -> Params:
    """Convert `tx` embedding parameters to `tfs` embedding parameters."""
    return {"W_E": params["embedding"]}


def tfs_pos_embed_params(_cfg, params: Params) -> Params:
    """Convert `tx` positional embedding parameters to `tfs` positional embedding parameters."""
    return {"W_pos": params["embedding"]}


def tfs_mlp_params(_cfg, params: Params) -> Params:
    """Convert `tx` mlp parameters to `tfs` mlp parameters."""
    return {
        "W_in": params["fc_1"]["kernel"],
        "W_out": params["proj"]["kernel"],
        "b_in": params["fc_1"]["bias"],
        "b_out": params["proj"]["bias"],
    }


def tfs_block_params(cfg, params: Params) -> Params:
    """Convert `tx` block parameters to `tfs` block parameters."""
    return {
        "ln1": tfs_layer_norm_params(cfg, params["ln_1"]),
        "attn": tfs_attention_params(cfg, params["attn"]),
        "ln2": tfs_layer_norm_params(cfg, params["ln_2"]),
        "mlp": tfs_mlp_params(cfg, params["mlp"]),
    }


def tfs_unembed_params(_cfg, params: Params) -> Params:
    """Convert `tx` unembed parameters to `tfs` unembed parameters."""
    return {"W_U": params["kernel"], "b_U": params["bias"]}


def tfs_transformer_params(cfg: TransformerConfig, params: Params) -> Params:
    """Convert `tx` transformer parameters to `tfs` transformer parameters.

    Args:
        cfg: The transformer configuration.
        params: The tx transformer parameters.

    Returns:
        Parameters for the 'Transformer from Scratch' model.
    """
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
