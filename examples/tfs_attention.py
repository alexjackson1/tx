from jaxtyping import Array, Float

import jax
import jax.numpy as jnp
import flax.linen as nn
import einops

from tx.modules import TransformerConfig


class Attention(nn.Module):
    cfg: TransformerConfig

    def setup(self):
        init_fn = nn.initializers.normal(stddev=self.cfg.init_range)
        qkv_kernel_shape = (
            self.cfg.num_heads,
            self.cfg.model_dim,
            self.cfg.head_dim,
        )
        self.W_Q = self.param("W_Q", init_fn, qkv_kernel_shape)
        self.W_K = self.param("W_K", init_fn, qkv_kernel_shape)
        self.W_V = self.param("W_V", init_fn, qkv_kernel_shape)
        self.W_O = self.param(
            "W_O",
            init_fn,
            (qkv_kernel_shape[0], qkv_kernel_shape[2], qkv_kernel_shape[1]),
        )

        qkv_bias_shape = (self.cfg.num_heads, self.cfg.head_dim)
        self.b_Q = self.param("b_Q", nn.initializers.zeros, qkv_bias_shape)
        self.b_K = self.param("b_K", nn.initializers.zeros, qkv_bias_shape)
        self.b_V = self.param("b_V", nn.initializers.zeros, qkv_bias_shape)
        self.b_O = self.param("b_O", nn.initializers.zeros, (self.cfg.model_dim,))

        self.IGNORE = jnp.array(-1e5, dtype=jnp.float32)

    def __call__(
        self, normalized_resid_pre: Float[Array, "batch seq model"]
    ) -> Float[Array, "batch seq model"]:
        # Calculate query, key and value vectors
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch seq model, n_head model h_dim -> batch seq n_head h_dim",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch seq model, n_head model h_dim -> batch seq n_head h_dim",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch seq model, n_head model h_dim -> batch seq n_head h_dim",
            )
            + self.b_V
        )

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch seq_q n_head h_dim, batch seq_k n_head h_dim -> batch n_head seq_q seq_k",
        )
        attn_scores_masked = self.apply_causal_mask(
            attn_scores / self.cfg.head_dim**0.5
        )
        attn_pattern = jax.nn.softmax(attn_scores_masked, axis=-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v,
            attn_pattern,
            "batch seq_k n_head h_dim, batch n_head seq_q seq_k -> batch seq_q n_head h_dim",
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch seq_q n_head h_dim, n_head h_dim model -> batch seq_q model",
            )
            + self.b_O
        )

        return attn_out

    def apply_causal_mask(
        self, attn_scores: Float[Array, "batch n_head seq_q seq_k"]
    ) -> Float[Array, "batch n_head seq_q seq_k"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        # Define a mask that is True for all positions we want to set probabilities to zero for
        all_ones = jnp.ones((attn_scores.shape[-2], attn_scores.shape[-1]))
        mask = jnp.triu(all_ones, k=1)
        # Apply the mask to attention scores, then return the masked scores
        # attn_scores.masked_fill_(mask, self.IGNORE)
        attn_scores = jnp.where(mask, self.IGNORE, attn_scores)
        return attn_scores
