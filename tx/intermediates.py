from enum import Enum


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
