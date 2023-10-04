from functools import partial
from jaxtyping import Array
from typing import Dict, Optional

import jax.numpy as jnp
from transformers import FlaxGPT2LMHeadModel

from tx.modules.transformer import TransformerConfig


class PretrainedGPT2Model(FlaxGPT2LMHeadModel):
    tx_config = TransformerConfig(
        vocab_dim=50257,
        context_length=1024,
        model_dim=768,
        num_layers=12,
        num_heads=12,
        head_dim=64,
        mlp_dim=3072,
        layer_norm_eps=1e-5,
        init_range=0.02,
    )

    @classmethod
    def make_config(
        cls,
        decode: bool = False,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.float64,
    ) -> TransformerConfig:
        return cls.tx_config.replace(
            decode=decode, dtype=dtype, param_dtype=param_dtype
        )

    @classmethod
    def attn_params(cls, attn):
        cfg = cls.tx_config

        kernel_shape = (cfg.num_heads, cfg.head_dim, cfg.model_dim)
        reshape_kernel = lambda k: jnp.reshape(k, kernel_shape)

        qkv_kernel = attn["c_attn"]["kernel"]
        qkv_kernels = jnp.split(qkv_kernel, 3, axis=0)
        qkv_kernels = map(reshape_kernel, qkv_kernels)
        qkv_kernels = tuple(map(partial(jnp.transpose, axes=(2, 0, 1)), qkv_kernels))

        qkv_bias = attn["c_attn"]["bias"]
        qkv_biases = jnp.split(qkv_bias, 3, axis=0)
        qkv_biases = tuple(map(lambda x: jnp.reshape(x, (12, 64)), qkv_biases))

        q_kernel, k_kernel, v_kernel = qkv_kernels
        q_bias, k_bias, v_bias = qkv_biases

        return {
            "query": {"kernel": q_kernel, "bias": q_bias},
            "key": {"kernel": k_kernel, "bias": k_bias},
            "value": {"kernel": v_kernel, "bias": v_bias},
            "c_proj": {
                "kernel": jnp.transpose(attn["c_proj"]["kernel"]),
                "bias": attn["c_proj"]["bias"],
            },
        }

    @classmethod
    def mlp_params(cls, mlp):
        return {
            "fc_1": {
                "kernel": jnp.transpose(mlp["c_fc"]["kernel"]),
                "bias": mlp["c_fc"]["bias"],
            },
            "proj": {
                "kernel": jnp.transpose(mlp["c_proj"]["kernel"]),
                "bias": mlp["c_proj"]["bias"],
            },
        }

    @classmethod
    def block_params(cls, block):
        return {
            "ln_1": block["ln_1"],
            "attn": cls.attn_params(block["attn"]),
            "ln_2": block["ln_2"],
            "mlp": cls.mlp_params(block["mlp"]),
        }

    def to_params(self) -> Dict[str, Array]:
        params = self._params["transformer"]
        blocks = params["h"]
        embedding: Array = params["wte"]["embedding"]

        return {
            "embed": params["wte"],
            "pos_embed": params["wpe"],
            **{
                f"block_{i}": PretrainedGPT2Model.block_params(blocks[f"{i}"])
                for i in range(len(blocks))
            },
            "ln_f": params["ln_f"],
            "unembed": {
                "kernel": jnp.transpose(embedding),
                "bias": jnp.zeros((embedding.shape[0],)),
            },
        }
