from typing import Iterable
from jaxtyping import Array, Float

import dataclasses
from functools import partial

import jax
import flax.linen as nn

from .common import LayerNorm, Embed, PosEmbed, Module, Unembed
from .attention import Attention


@dataclasses.dataclass
class TransformerConfig:
    vocab_dim: int = 50257
    """The size of the vocabulary."""
    context_length: int = 1024
    """The (maximum) length of the context (sequence of inputs)."""
    model_dim: int = 768
    """The size of the model."""
    num_layers: int = 12
    """The number of layers of transformer blocks in the model."""
    num_heads: int = 12
    """The number of attention heads."""
    head_dim: int = 64
    """The size of the attention heads."""
    mlp_dim: int = 3072
    """The size of the intermediate layer in the `MLP` module."""
    layer_norm_eps: float = 1e-5
    """The epsilon value to in `LayerNorm` layers."""
    init_range: float = 0.02
    """The range of the normal distribution used to initialize the weights."""


class MLP(Module):
    features: Iterable[int]
    init_range: float = 0.02

    def setup(self):
        k_init = jax.nn.initializers.normal(stddev=self.init_range)
        b_init = jax.nn.initializers.zeros
        d_init = partial(nn.DenseGeneral, axis=-1, kernel_init=k_init, bias_init=b_init)
        self.fc_1 = d_init(features=self.features[0])
        self.proj = d_init(features=self.features[1])

    def __call__(self, x: Float[Array, "seq embed"]) -> Float[Array, "seq embed"]:
        x = self.fc_1(x)
        self.intermediate("pre_activation", x)

        x = nn.gelu(x)
        self.intermediate("post_activation", x)

        x = self.proj(x)
        return x


class TransformerBlock(Module):
    num_heads: int
    head_dim: int
    model_dim: int
    mlp_dim: int
    context_length: int
    epsilon: float = 1e-5
    init_range: float = 0.02

    def setup(self):
        self.ln_1 = LayerNorm(epsilon=self.epsilon)
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            model_dim=self.model_dim,
            init_range=self.init_range,
            context_length=self.context_length,
        )
        self.ln_2 = LayerNorm(epsilon=self.epsilon)
        self.mlp = MLP(
            features=[self.mlp_dim, self.model_dim],
            init_range=self.init_range,
        )

    def __call__(self, x: Float[Array, "seq embed"]) -> Float[Array, "seq embed"]:
        x_norm = self.ln_1(x)
        x = self.attn(x_norm) + x
        # self.intermediate("attention_output", x)

        x_norm = self.ln_2(x)
        x = self.mlp(x_norm) + x
        return x


class Transformer(Module):
    model_dim: int = 768
    layer_norm_eps: Float = 1e-5
    vocab_dim: int = 50257
    context_length: int = 1024
    num_heads: int = 12
    head_dim: int = 64
    mlp_dim: int = 3072
    num_layers: int = 12
    init_range: float = 0.02

    @classmethod
    def from_config(cls, config: TransformerConfig):
        return cls(**config.__dict__)

    def setup(self):
        self.embed = Embed(
            features=self.model_dim,
            num_embeddings=self.vocab_dim,
            init_range=self.init_range,
        )
        self.pos_embed = PosEmbed(
            features=self.model_dim,
            num_embeddings=self.context_length,
            init_range=self.init_range,
        )
        self.blocks = [
            TransformerBlock(
                name=f"block_{i}",
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                model_dim=self.model_dim,
                mlp_dim=self.mlp_dim,
                epsilon=self.layer_norm_eps,
                context_length=self.context_length,
                init_range=self.init_range,
            )
            for i in range(self.num_layers)
        ]
        self.ln_f = LayerNorm(epsilon=self.layer_norm_eps)
        self.unembed = Unembed(
            features=self.model_dim,
            num_embeddings=self.vocab_dim,
            init_range=self.init_range,
        )

    def __call__(self, tokens: Float[Array, "seq"]) -> Float[Array, "seq vocab"]:
        embed = self.embed(tokens)  # text embedding
        self.intermediate("embedding", embed)

        pos_embed = self.pos_embed(tokens)  # positional embedding
        self.intermediate("positional_embedding", pos_embed)

        x = embed + pos_embed  # combine embeddings
        # self.intermediate("residual", x)

        for block in self.blocks:  # loop over layers/blocks
            x = block(x)  # apply attention and mlp
            # self.intermediate("residual", x)

        x = self.ln_f(x)  # apply final layer norm
        self.intermediate("final_output", x)

        logits = self.unembed(x)  # unembed to logits
        return logits
