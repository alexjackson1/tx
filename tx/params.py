import jax.numpy as jnp
import optax

from .modules.transformer import TransformerConfig


def tx_to_flax(config: TransformerConfig, tx_params: optax.Params) -> optax.Params:
    c_attn, c_proj = tx_params["c_attn"], tx_params["c_proj"]
    num_heads, head_dim, model_dim = (
        config.num_heads,
        config.head_dim,
        config.model_dim,
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
