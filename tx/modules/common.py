from typing import Optional
from jaxtyping import Array, Float, Int

import jax
import jax.numpy as jnp
import flax.linen as nn


class LayerNorm(nn.Module):
    """Layer normalization module.

    Attributes:
        epsilon: A small value used to prevent division by zero.
        dtype: The dtype of the input array.
        param_dtype: The dtype of the parameters.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, features).
    2. Output shape: (...batch_dims, features).

    Transformation Steps:
    1. Computes the mean and variance of the input array.
    2. Normalizes the input array using the mean and variance.
    3. Applies a scale and bias to the normalized array.

    Notes:
    - The `features` length is inferred from the input array.
    - The scale and bias are initialized to `1` and `0` respectively.
    """

    epsilon: float = 1e-5
    """A small value used to prevent division by zero."""
    dtype: Optional[jnp.dtype] = None
    """The dtype of the input array."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Normalizes the input array using the mean and variance."""
        # Convert to JAX array
        dtype = self.dtype or jnp.result_type(x)
        x = jnp.asarray(x, dtype)

        # Compute statistics
        x_mean = jnp.mean(x, axis=-1, keepdims=True)
        x_var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - x_mean) / jnp.sqrt(x_var + self.epsilon)

        # Apply scale and bias
        scale = self.param(
            "scale",
            jax.nn.initializers.ones,
            (x.shape[-1],),
            self.param_dtype,
        )
        bias = self.param(
            "bias",
            jax.nn.initializers.zeros,
            (x.shape[-1],),
            self.param_dtype,
        )
        x = x * scale + bias

        return x


class Embed(nn.Module):
    """Embedding module.

    Attributes:
        num_embeddings: The number of embeddings.
        features: The dimensionality of the embeddings.
        init_range: The standard deviation of the normal distribution used to
            initialize the embeddings.
        dtype: The dtype of the input array.
        param_dtype: The dtype of the parameters.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, seq_len).
    2. Output shape: (...batch_dims, seq_len, features).

    Transformation Steps:
    1. Looks up the embeddings for each token in the input array.
    """

    num_embeddings: int
    """The number of embeddings."""
    features: int
    """The dimensionality of the embeddings."""
    init_range: float = 0.02
    """The standard deviation of the normal distribution used to initialize the
    embeddings."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    @nn.compact
    def __call__(self, tokens: Int[Array, "... S"]) -> Float[Array, "... S F"]:
        """Looks up the embeddings for each token in the input array."""
        # Lookup embeddings
        embedding = self.param(
            "embedding",
            nn.initializers.normal(self.init_range),
            (self.num_embeddings, self.features),
            self.param_dtype,
        )
        return jnp.take(embedding, tokens, axis=0)


class PosEmbed(nn.Module):
    """Positional embedding module.

    Attributes:
        seq_len: The maximum sequence length.
        features: The dimensionality of the embeddings.
        init_range: The standard deviation of the normal distribution used to
            initialize the embeddings.
        dtype: The dtype of the input array.
        param_dtype: The dtype of the parameters.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, seq_len).
    2. Output shape: (...batch_dims, seq_len, features).

    Transformation Steps:
    1. Computes the positional embeddings for each token in the input array.
    """

    num_embeddings: int
    """The number of embeddings."""
    features: int
    """The dimensionality of the embeddings."""
    init_range: float = 0.02
    """The standard deviation of the normal distribution used to initialize the
    embeddings."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    @nn.compact
    def __call__(self, tokens: Int[Array, "... S"]) -> Float[Array, "... S F"]:
        """Computes the positional embeddings for each token in the input array."""
        # Lookup embeddings
        embedding = self.param(
            "embedding",
            nn.initializers.normal(self.init_range),
            (self.num_embeddings, self.features),
            self.param_dtype,
        )
        return embedding[: tokens.shape[-1]]


class Unembed(nn.Module):
    """Unembedding module.

    Attributes:
        features: The dimensionality of the embeddings.
        num_embeddings: The number of embeddings.
        init_range: The standard deviation of the normal distribution used to
            initialize the embeddings.
        dtype: The dtype of the input array.
        param_dtype: The dtype of the parameters.

    Input/Output Dimensionality:
    1. Input shape: (...batch_dims, seq_len, features).
    2. Output shape: (...batch_dims, seq_len, num_embeddings).

    Transformation Steps:
    1. Computes the logits for each token in the input array.
    """

    features: int
    """The dimensionality of the embeddings."""
    num_embeddings: int
    """The number of embeddings."""
    init_range: float = 0.02
    """The standard deviation of the normal distribution used to initialize the
    embeddings."""
    dtype: jnp.dtype = jnp.float32
    """The dtype of the input array."""
    param_dtype: jnp.dtype = jnp.float32
    """The dtype of the parameters."""

    @nn.compact
    def __call__(self, x: Float[Array, "... S F"]) -> Float[Array, "... S V"]:
        """Computes the logits for each token in the input array."""
        # Convert to JAX array
        dtype = self.dtype or jnp.result_type(x)
        x = jnp.asarray(x, dtype)

        # Compute logits
        kernel = self.param(
            "kernel",
            nn.initializers.normal(self.init_range),
            (self.features, self.num_embeddings),
            self.param_dtype,
        )
        bias = self.param(
            "bias",
            jax.nn.initializers.zeros,
            (self.num_embeddings,),
            self.param_dtype,
        )
        return jnp.einsum("...sf,fv->...sv", x, kernel) + bias
