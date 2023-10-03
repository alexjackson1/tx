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
        return {
            "c_attn": {
                "kernel": jnp.transpose(attn["c_attn"]["kernel"]),
                "bias": attn["c_attn"]["bias"],
            },
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
