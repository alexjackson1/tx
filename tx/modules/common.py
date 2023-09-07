from typing import Any
from jaxtyping import Array, Float, Int

import jax
import jax.numpy as jnp
import flax.linen as nn
import einops


class Module(nn.Module):
    def intermediate(self, name: str, value: Any) -> bool:
        return self.sow("intermediates", name, value)


class LayerNorm(Module):
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p m"]:
        x_mean = jnp.mean(x, axis=-1, keepdims=True)
        x_var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - x_mean) / jnp.sqrt(x_var + self.epsilon)

        scale = self.param("scale", jax.nn.initializers.ones, (x.shape[-1],))
        bias = self.param("bias", jax.nn.initializers.zeros, (x.shape[-1],))
        x = x * scale + bias

        return x


class Embed(Module):
    num_embeddings: int
    features: int
    init_range: float = 0.02

    def setup(self):
        shape = (self.num_embeddings, self.features)
        init_fn = nn.initializers.normal(self.init_range)
        self.embedding = self.param("embedding", init_fn, shape)

    def __call__(self, tokens: Int[Array, "b p"]) -> Float[Array, "b p m"]:
        return jnp.take(self.embedding, tokens, axis=0)


class PosEmbed(Module):
    num_embeddings: int
    features: int
    init_range: float = 0.02

    def setup(self):
        shape = (self.num_embeddings, self.features)
        init_fn = nn.initializers.normal(self.init_range)
        self.embedding = self.param("embedding", init_fn, shape)

    def __call__(self, tokens: Int[Array, "b p"]) -> Float[Array, "b p m"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.embedding[:seq_len], "p m -> b p m", b=batch)


class Unembed(Module):
    features: int
    num_embeddings: int
    init_range: float = 0.02

    def setup(self):
        init_fn = jax.nn.initializers.normal(stddev=self.init_range)
        shape = (self.features, self.num_embeddings)
        self.kernel = self.param("kernel", init_fn, shape)
        self.bias = self.param("bias", jax.nn.initializers.zeros, (shape[-1],))

    def __call__(self, x: Float[Array, "b p m"]) -> Float[Array, "b p v"]:
        x = einops.einsum(x, self.kernel, "b p m, m v -> b p v")
        x = x + self.bias
        return x
