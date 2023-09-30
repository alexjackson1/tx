from enum import Enum


class Intermediate(Enum):
    """Possible intermediate outputs of a `tx` Transformer model."""

    EMBEDDING = "embedding"
    """The output of the embedding layer."""
    POSITIONAL_EMBEDDING = "positional_embedding"
    """The output of the positional embedding layer."""
    RESIDUAL = "residual"
    """The residual states at the start of each layer."""
    LN_1_OUTPUT = "ln_1_output"
    """The output of the first layer norm in a block."""
    ATTN_QUERY = "query"
    """The attention query."""
    ATTN_KEY = "key"
    """The attention key."""
    ATTN_VALUE = "value"
    """The attention value."""
    ATTN_SCORES = "scores"
    """The attention scores."""
    ATTN_WEIGHTS = "weights"
    """The attention weights."""
    ATTN_Z = """The attention weights indexed with values"""
    """The unprojected final attention values."""
    ATTN_OUTPUT = "attention_output"
    """The output of the attention block at each layer."""
    BLOCK_LN_2_OUTPUT = "ln_2_output"
    """The output of the second layer norm in a block."""
    MLP_PRE_ACTIVATION = "pre_activation"
    """The output of the MLP before the activation function."""
    MLP_POST_ACTIVATION = "post_activation"
    """The output of the MLP after the activation function."""
    FINAL_OUTPUT = "final_output"
    """The final output of the model (prior to unembedding)."""


AllIntermediates = [i.value for i in Intermediate]
"""A list of all possible intermediate outputs."""
