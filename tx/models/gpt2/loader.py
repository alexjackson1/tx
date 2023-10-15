from functools import partial
from jaxtyping import Array, PyTree

import jax.numpy as jnp

from transformers import FlaxGPT2LMHeadModel, GPT2TokenizerFast

from .modules import GPT2Config


class GPT2Loader(FlaxGPT2LMHeadModel):
    tokenizer_class = GPT2TokenizerFast

    @classmethod
    def load_params(cls, model_name: str, config: GPT2Config) -> PyTree[Array]:
        model = cls.from_pretrained(model_name)
        params = model._params["transformer"]
        blocks = params["h"]
        embedding: Array = params["wte"]["embedding"]

        return {
            "embed": params["wte"],
            "pos_embed": params["wpe"],
            **{
                f"block_{i}": GPT2Loader.block_params(blocks[f"{i}"], config)
                for i in range(len(blocks))
            },
            "ln_f": params["ln_f"],
            "unembed": {
                "kernel": jnp.transpose(embedding),
                "bias": jnp.zeros((embedding.shape[0],)),
            },
        }

    @classmethod
    def attn_params(cls, attn: PyTree[Array], config: GPT2Config):
        kernel_shape = (config.num_heads, config.head_dim, config.model_dim)
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
    def mlp_params(cls, mlp: PyTree[Array]):
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
    def block_params(cls, block: PyTree[Array], config: GPT2Config) -> PyTree[Array]:
        return {
            "ln_1": block["ln_1"],
            "attn": cls.attn_params(block["attn"], config),
            "ln_2": block["ln_2"],
            "mlp": cls.mlp_params(block["mlp"]),
        }

    def to_params(self, config: GPT2Config) -> PyTree[Array]:
        params = self._params["transformer"]
        blocks = params["h"]
        embedding: Array = params["wte"]["embedding"]

        return {
            "embed": params["wte"],
            "pos_embed": params["wpe"],
            **{
                f"block_{i}": GPT2Loader.block_params(blocks[f"{i}"], config)
                for i in range(len(blocks))
            },
            "ln_f": params["ln_f"],
            "unembed": {
                "kernel": jnp.transpose(embedding),
                "bias": jnp.zeros((embedding.shape[0],)),
            },
        }
