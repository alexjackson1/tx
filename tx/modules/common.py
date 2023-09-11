from enum import Enum
from jaxtyping import Array, Float, Int

import jax
import jax.numpy as jnp
import flax.linen as nn
import einops


class Intermediates(Enum):
    embedding = "embedding"
    positional_embedding = "positional_embedding"
    residual = "residual"
    attention_output = "attention_output"
    mlp_pre_activation = "pre_activation"
    mlp_post_activation = "post_activation"
    final_output = "final_output"
    attn_scores = "scores"
    attn_pattern = "pattern"
    attn_z = "z"
    attn_q = "query"
    attn_k = "key"
    attn_v = "value"
    block_ln_1_output = "ln_1_output"
    block_ln_2_output = "ln_2_output"


AllIntermediates = [i.value for i in Intermediates]


class LayerNorm(nn.Module):
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x_mean = jnp.mean(x, axis=-1, keepdims=True)
        x_var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - x_mean) / jnp.sqrt(x_var + self.epsilon)

        scale = self.param("scale", jax.nn.initializers.ones, (x.shape[-1],))
        bias = self.param("bias", jax.nn.initializers.zeros, (x.shape[-1],))
        x = x * scale + bias

        return x


class Embed(nn.Module):
    num_embeddings: int
    features: int
    init_range: float = 0.02

    def setup(self):
        shape = (self.num_embeddings, self.features)
        init_fn = nn.initializers.normal(self.init_range)
        self.embedding = self.param("embedding", init_fn, shape)

    def __call__(self, tokens: Int[Array, "seq"]) -> Float[Array, "seq embed"]:
        return jnp.take(self.embedding, tokens, axis=0)


class PosEmbed(nn.Module):
    num_embeddings: int
    features: int
    init_range: float = 0.02

    def setup(self):
        shape = (self.num_embeddings, self.features)
        init_fn = nn.initializers.normal(self.init_range)
        self.embedding = self.param("embedding", init_fn, shape)

    def __call__(self, tokens: Int[Array, "seq"]) -> Float[Array, "seq embed"]:
        return self.embedding[: tokens.shape[0]]


class Unembed(nn.Module):
    features: int
    num_embeddings: int
    init_range: float = 0.02

    def setup(self):
        init_fn = jax.nn.initializers.normal(stddev=self.init_range)
        shape = (self.features, self.num_embeddings)
        self.kernel = self.param("kernel", init_fn, shape)
        self.bias = self.param("bias", jax.nn.initializers.zeros, (shape[-1],))

    def __call__(self, x: Float[Array, "seq embed"]) -> Float[Array, "seq vocab"]:
        x = einops.einsum(x, self.kernel, "seq embed, embed vocab -> seq vocab")
        x = x + self.bias
        return x
