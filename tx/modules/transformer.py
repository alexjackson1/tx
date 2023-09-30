from typing import Iterable, List
from jaxtyping import Array, Float

import dataclasses
from functools import partial

import jax
import flax.linen as nn
import flax.struct as struct

from .common import LayerNorm, Embed, PosEmbed, Unembed
from .attention import MultiHeadAttention


@dataclasses.dataclass
class TransformerConfig:
    """Configuration for the `Transformer` module.

    Attributes:
        vocab_dim: The size of the vocabulary.
        context_length: The (maximum) length of the context (sequence of inputs).
        model_dim: The size of the model.
        num_layers: The number of layers of transformer blocks in the model.
        num_heads: The number of attention heads.
        head_dim: The size of the attention heads.
        mlp_dim: The size of the intermediate layer in the `MLP` module.
        layer_norm_eps: The epsilon value to in `LayerNorm` layers.
        init_range: The range of the normal distribution used to initialize the
            weights.
    """

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
    """The epsilon value for `LayerNorm` layers."""
    init_range: float = 0.02
    """The range of the normal distribution used to initialize the weights."""


class MLP(nn.Module):
    """Multi-layer perceptron module.

    Attributes:
        features: A list of integers defining the number of features in each
            layer.
        init_range: The standard deviation of the normal distribution used to
            initialize the linear transformations.
        use_bias: Whether to include a bias term in the linear transformations.
        intermediates: A list of intermediate arrays to store in the module's
            state dictionary.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, seq_len, features[0]).
    2. Output shape: (...batch_dimensions, sequence_length, features[-1]).

    Transformation Steps:
    1. Applies a linear transformation to the input array.
    2. Applies the GELU activation function to the linear transformation.
    3. Repeats according to the number of layers in the module.
    4. Applies a linear transformation to the output of the GELU activation.

    Intermediate Arrays:
    1. `pre_activation`: The output of the first linear transformation.
    2. `post_activation`: The output of the GELU activation.
    """

    features: Iterable[int]
    """A list of integers defining the number of features in each layer."""
    init_range: float = 0.02
    """The standard deviation of the normal distribution used to initialize the
    linear transformations."""

    intermediates: List[str] = struct.field(default_factory=list)
    """A list of intermediate arrays to store in the module's state dictionary."""

    def intermediate(self, name: str, value: Array) -> bool:
        """Store an intermediate array in the module's state dictionary.

        Args:
            name: The name of the intermediate array.
            value: The intermediate array.

        Returns:
            Whether the intermediate array was stored.
        """
        if name in self.intermediates:
            return self.sow("intermediates", name, value)
        return False

    @nn.compact
    def __call__(self, x: Float[Array, "S F1"]) -> Float[Array, "S FN"]:
        """Applies the MLP module to the input array."""
        init_dense = partial(
            nn.DenseGeneral,
            axis=-1,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
        )

        for i, features in enumerate(self.features[:-1]):
            x = init_dense(name=f"fc_{i+1}", features=features)(x)
            self.intermediate("pre_activation", x)

            x = nn.gelu(x)
            self.intermediate("post_activation", x)

        x = init_dense(name="proj", features=self.features[1])(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block module.

    Attributes:
        num_heads: The number of attention heads.
        head_dim: The size of the attention heads.
        model_dim: The size of the model.
        mlp_dim: The size of the intermediate layer in the `MLP` module.
        epsilon: The epsilon value to in `LayerNorm` layers.
        init_range: The range of the normal distribution used to initialize the
            weights.
        intermediates: A list of intermediate arrays to store in the module's
            state dictionary.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, seq_len, model_dim).
    2. Output shape: (...batch_dims, seq_len, model_dim).

    Transformation Steps:
    1. Applies layer normalization to the input array.
    2. Applies multi-headed attention to the layer normalized input.
    3. Applies residual connection to the output of the multi-headed attention.
    4. Applies layer normalization to the output of the residual connection.
    5. Applies a multi-layer perceptron to the output of the layer normalization.
    6. Applies residual connection to the output of the multi-layer perceptron.
    7. Applies layer normalization to the output of the residual connection.

    Intermediate Arrays:
    1. `ln_1_output`: The output of the first layer normalization.
    2. `attention_output`: The output of the multi-headed attention.
    3. `ln_2_output`: The output of the second layer normalization.
    4. `mlp_output`: The output of the multi-layer perceptron.
    5. `final_residual`: The output of the second residual connection.
    6. `final_output`: The output of the final layer normalization.
    """

    num_heads: int
    """The number of attention heads."""
    head_dim: int
    """The size of the attention heads."""
    model_dim: int
    """The size of the model."""
    mlp_dim: int
    """The size of the intermediate layer in the `MLP` module."""
    epsilon: float = 1e-5
    """The epsilon value for `LayerNorm` layers."""
    init_range: float = 0.02
    """The range of the normal distribution used to initialize the weights."""

    intermediates: List[str] = struct.field(default_factory=list)

    def intermediate(self, name: str, value: Array) -> bool:
        """Store an intermediate array in the module's state dictionary."""
        if name in self.intermediates:
            return self.sow("intermediates", name, value)
        return False

    @nn.compact
    def __call__(self, x: Float[Array, "S F"]) -> Float[Array, "S F"]:
        """Applies the transformer block module to the input array."""
        x_norm = LayerNorm(name="ln_1", epsilon=self.epsilon)(x)
        self.intermediate("ln_1_output", x_norm)
        x = (
            MultiHeadAttention(
                name="attn",
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                features=self.model_dim,
                init_range=self.init_range,
                intermediates=self.intermediates,
            )(x_norm)
            + x
        )
        self.intermediate("attention_output", x)

        x_norm = LayerNorm(name="ln_2", epsilon=self.epsilon)(x)
        self.intermediate("ln_2_output", x_norm)
        x = (
            MLP(
                name="mlp",
                features=[self.mlp_dim, self.model_dim],
                init_range=self.init_range,
                intermediates=self.intermediates,
            )(x_norm)
            + x
        )
        return x


class Transformer(nn.Module):
    """Transformer module.

    Attributes:
        model_dim: The size of the model.
        vocab_dim: The size of the vocabulary.
        context_length: The (maximum) length of the context (sequence of inputs).
        num_heads: The number of attention heads.
        head_dim: The size of the attention heads.
        mlp_dim: The size of the intermediate layer in the `MLP` module.
        num_layers: The number of layers of transformer blocks in the model.
        layer_norm_eps: The epsilon value to in `LayerNorm` layers.
        init_range: The range of the normal distribution used to initialize the
            weights.
        intermediates: A list of intermediate arrays to store in the module's
            state dictionary.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, seq_len).
    2. Output shape: (...batch_dims, seq_len, vocab_dim).

    Transformation Steps:
    1. Applies a text embedding to the input array.
    2. Applies a positional embedding to the input array.
    3. Combines the text and positional embeddings.
    4. Applies the transformer block to the combined embeddings.
    5. Applies layer normalization to the output of the transformer block.
    6. Applies a linear transformation to the output of the layer normalization.
    7. Applies a softmax function to the output of the linear transformation.

    Intermediate Arrays:
    1. `embedding`: The output of the text embedding.
    2. `positional_embedding`: The output of the positional embedding.
    3. `residual`: The output of the residual connection.
    4. `final_residual`: The output of the final residual connection.
    5. `final_output`: The output of the final layer normalization.
    """

    model_dim: int = 768
    """The size of the model."""
    layer_norm_eps: Float = 1e-5
    """The epsilon value for `LayerNorm` layers."""
    vocab_dim: int = 50257
    """The size of the vocabulary."""
    context_length: int = 1024
    """The (maximum) length of the context (sequence of inputs)."""
    num_heads: int = 12
    """The number of attention heads."""
    head_dim: int = 64
    """The size of the attention heads."""
    mlp_dim: int = 3072
    """The size of the intermediate layer in the `MLP` module."""
    num_layers: int = 12
    """The number of layers of transformer blocks in the model."""
    init_range: float = 0.02
    """The range of the normal distribution used to initialize the weights."""

    intermediates: List[str] = struct.field(default_factory=list)
    """A list of intermediate arrays to store in the module's state dictionary."""

    def intermediate(self, name: str, value: Array) -> bool:
        """Store an intermediate array in the module's state dictionary."""
        if name in self.intermediates:
            return self.sow("intermediates", name, value)
        return False

    @classmethod
    def from_config(cls, config: TransformerConfig, **kwargs):
        """Creates a `Transformer` module from a `TransformerConfig` object."""
        return cls(**config.__dict__, **kwargs)

    @nn.compact
    def __call__(self, tokens: Float[Array, "... S"]) -> Float[Array, "... S V"]:
        """Applies the transformer module to the input array."""
        embed = Embed(
            name="embed",
            features=self.model_dim,
            num_embeddings=self.vocab_dim,
            init_range=self.init_range,
        )(tokens)
        self.intermediate("embedding", embed)

        pos_embed = PosEmbed(
            name="pos_embed",
            features=self.model_dim,
            num_embeddings=self.context_length,
            init_range=self.init_range,
        )(tokens)
        self.intermediate("positional_embedding", pos_embed)

        x = embed + pos_embed
        self.intermediate("residual", x)

        for i in range(self.num_layers):
            x = TransformerBlock(
                name=f"block_{i}",
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                model_dim=self.model_dim,
                mlp_dim=self.mlp_dim,
                epsilon=self.layer_norm_eps,
                init_range=self.init_range,
                intermediates=self.intermediates,
            )(x)

        self.intermediate("final_residual", x)

        x = LayerNorm(name="ln_f", epsilon=self.layer_norm_eps)(x)
        self.intermediate("final_output", x)

        logits = Unembed(
            name="unembed",
            features=self.model_dim,
            num_embeddings=self.vocab_dim,
            init_range=self.init_range,
        )(x)
        return logits
