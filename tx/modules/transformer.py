from dataclasses import dataclass
from functools import partial
from jaxtyping import Array, Float, Int
from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from .common import LayerNorm, Embed, PosEmbed, Unembed
from .attention import MultiHeadAttention
from ..hooks import HookMap, HookPoint, apply_hooks


@dataclass
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
        dtype: The dtype of the input array.
        param_dtype: The dtype of the parameters.
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
    decode: bool = False
    """Whether to use the transformer in decode mode."""
    init_range: float = 0.02
    """The range of the normal distribution used to initialize the weights."""
    dtype: Optional[jnp.dtype] = None
    """The dtype of the input array."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    def replace(self, **kwargs):
        """Returns a new `TransformerConfig` object with the specified
        attributes replaced."""
        return TransformerConfig(**{**self.__dict__, **kwargs})


class MLP(nn.Module):
    """Multi-layer perceptron module.

    Attributes:
        features: A list of integers defining the number of features in each
            layer.
        init_range: The standard deviation of the normal distribution used to
            initialize the linear transformations.
        use_bias: Whether to include a bias term in the linear transformations.
        dtype: The dtype of the input array.
        param_dtype: The dtype of the parameters.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, seq_len, features[0]).
    2. Output shape: (...batch_dimensions, sequence_length, features[-1]).

    Transformation Steps:
    1. Applies a linear transformation to the input array.
    2. Applies the GELU activation function to the linear transformation.
    3. Repeats according to the number of layers in the module.
    4. Applies a linear transformation to the output of the GELU activation.
    """

    features: Iterable[int]
    """A list of integers defining the number of features in each layer."""
    init_range: float = 0.02
    """The standard deviation of the normal distribution used to initialize the
    linear transformations."""
    dtype: Optional[jnp.dtype] = None
    """The dtype of the input array."""
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, x: Float[Array, "S F1"], hooks: Optional[HookMap] = None
    ) -> Float[Array, "S FN"]:
        """Applies the MLP module to the input array."""
        dtype = self.dtype or jnp.result_type(x)
        x = jnp.asarray(x, dtype)

        init_dense = partial(
            nn.DenseGeneral,
            axis=-1,
            kernel_init=jax.nn.initializers.normal(stddev=self.init_range),
            bias_init=jax.nn.initializers.zeros,
            dtype=dtype,
            param_dtype=self.param_dtype,
        )

        for i, features in enumerate(self.features[:-1]):
            x = init_dense(name=f"fc_{i+1}", features=features)(x)
            x = apply_hooks(HookPoint.MLP_PRE_ACTIVATION, hooks, x, module=self)

            x = nn.gelu(x)
            x = apply_hooks(HookPoint.MLP_POST_ACTIVATION, hooks, x, module=self)

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
        decode: Whether to use cache in the transformer blocks.
        init_range: The range of the normal distribution used to initialize the
            weights.
        dtype: The dtype of the input array.
        param_dtype: The dtype of the parameters.

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
    decode: bool = False
    """Whether to use cache in the transformer blocks."""
    init_range: float = 0.02
    """The range of the normal distribution used to initialize the weights."""
    dtype: Optional[jnp.dtype] = None
    """The dtype of the input array."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    @nn.compact
    def __call__(
        self, x: Float[Array, "S F"], hooks: Optional[HookMap] = None
    ) -> Float[Array, "S F"]:
        """Applies the transformer block module to the input array."""
        # Cast the input array to the correct dtype
        dtype = self.dtype or jnp.result_type(x)
        x = jnp.asarray(x, dtype)

        # Define function for initialising layer normalisation layers
        init_layer_norm = partial(
            LayerNorm,
            epsilon=self.epsilon,
            dtype=dtype,
            param_dtype=self.param_dtype,
        )

        # First layer normalisation
        x_norm = init_layer_norm(name="ln_1")(x, hooks)

        # Multi-headed attention
        mask = nn.make_causal_mask(jnp.ones(x.shape[:-1]), dtype="bool")
        attn_output = MultiHeadAttention(
            name="attn",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            features=self.model_dim,
            init_range=self.init_range,
            decode=self.decode,
            dtype=dtype,
            param_dtype=self.param_dtype,
        )(x_norm, mask, hooks)

        # First residual connection
        x = attn_output + x

        # Second layer normalisation
        x_norm = init_layer_norm(name="ln_2")(x, hooks)

        # MLP
        mlp_out = MLP(
            name="mlp",
            features=[self.mlp_dim, self.model_dim],
            init_range=self.init_range,
            dtype=dtype,
            param_dtype=self.param_dtype,
        )(x_norm, hooks)

        # Second residual connection
        x = mlp_out + x
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
        decode: Whether to use cache in the transformer blocks.
        init_range: The range of the normal distribution used to initialize the
            weights.
        dtype: The dtype of the input array.
        param_dtype: The dtype of the parameters.

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
    decode: bool = False
    """Whether to use the transformer in decode mode."""
    init_range: float = 0.02
    """The range of the normal distribution used to initialize the weights."""
    dtype: Optional[jnp.dtype] = None
    """The dtype of the input array."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    @classmethod
    def from_config(cls, config: TransformerConfig, **kwargs):
        """Creates a `Transformer` module from a `TransformerConfig` object."""
        return cls(**config.__dict__, **kwargs)

    @nn.compact
    def __call__(
        self, tokens: Int[Array, "... S"], hooks: Optional[HookMap] = None
    ) -> Float[Array, "... S V"]:
        """Applies the transformer module to the input array."""
        dtype = jnp.promote_types(self.dtype, jnp.float32)

        # Embed the input tokens
        embed = Embed(
            name="embed",
            features=self.model_dim,
            num_embeddings=self.vocab_dim,
            init_range=self.init_range,
            param_dtype=self.param_dtype,
        )(tokens)
        embed = apply_hooks(HookPoint.EMBED, hooks, embed, module=self)

        # Embed the positional information
        # offset = self.compute_offset(tokens, cache)
        pos_embed = PosEmbed(
            name="pos_embed",
            features=self.model_dim,
            num_embeddings=self.context_length,
            init_range=self.init_range,
            param_dtype=self.param_dtype,
        )(tokens)
        pos_embed = apply_hooks(HookPoint.POS_EMBED, hooks, pos_embed)

        # Combine the embeddings
        x = embed + pos_embed

        # Apply the transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                name=f"block_{i}",
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                model_dim=self.model_dim,
                mlp_dim=self.mlp_dim,
                epsilon=self.layer_norm_eps,
                decode=self.decode,
                init_range=self.init_range,
                dtype=dtype,
                param_dtype=self.param_dtype,
            )(x, hooks)
            x = apply_hooks(HookPoint.RESIDUAL, hooks, x, module=self)

        # Final layer normalisation
        x = LayerNorm(
            name="ln_f",
            epsilon=self.layer_norm_eps,
            dtype=dtype,
            param_dtype=self.param_dtype,
        )(x, hooks)
        x = apply_hooks(HookPoint.FINAL_OUTPUT, hooks, x, module=self)

        logits = Unembed(
            name="unembed",
            features=self.model_dim,
            num_embeddings=self.vocab_dim,
            init_range=self.init_range,
            dtype=dtype,
            param_dtype=self.param_dtype,
        )(x)

        return logits
