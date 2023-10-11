from jaxtyping import Array, PyTree
import jax.numpy as jnp

from tx.modules.transformer import TransformerConfig

Params = PyTree[Array]


def to_flax(config: TransformerConfig, tx_params: Params) -> Params:
    shape = (config.num_heads, config.head_dim, config.model_dim)
    keys = ["query", "key", "value"]

    q_kernel, k_kernel, v_kernel = map(lambda x: tx_params[x]["kernel"], keys)
    q_bias, k_bias, v_bias = map(lambda x: tx_params[x]["bias"], keys)

    o_kernel = jnp.reshape(tx_params["c_proj"]["kernel"], shape)
    o_bias = tx_params["c_proj"]["bias"]

    return {
        "query": {"kernel": q_kernel, "bias": q_bias},
        "key": {"kernel": k_kernel, "bias": k_bias},
        "value": {"kernel": v_kernel, "bias": v_bias},
        "out": {"kernel": o_kernel, "bias": o_bias},
    }
