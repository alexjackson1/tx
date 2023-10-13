from tx.models.gpt2.module import GPT2Config, GPT2Transformer


def check_ln_hook_points(hook_points):
    assert "std_hook" in hook_points
    assert "normalized_hook" in hook_points


def test_gpt2_transformer_hook_points():
    config = GPT2Config()
    model = GPT2Transformer.from_config(config)
    hook_points = model.hook_points()

    assert "embed_hook" in hook_points
    assert "pos_embed_hook" in hook_points
    assert "residual_hook" in hook_points
    assert "output_hook" in hook_points

    for i in range(config.num_layers):
        assert f"block_{i}" in hook_points
        block_hps = hook_points[f"block_{i}"]

        assert "ln_1" in block_hps
        check_ln_hook_points(block_hps["ln_1"])

        assert "attn" in block_hps
        assert "query_hook" in block_hps["attn"]
        assert "key_hook" in block_hps["attn"]
        assert "value_hook" in block_hps["attn"]
        assert "scores_hook" in block_hps["attn"]
        assert "weights_hook" in block_hps["attn"]
        assert "z_hook" in block_hps["attn"]
        assert "output_hook" in block_hps["attn"]

        assert "ln_2" in block_hps
        check_ln_hook_points(block_hps["ln_2"])

        assert "mlp" in block_hps
        assert "pre_activation_hook" in block_hps["mlp"]
        assert "post_activation_hook" in block_hps["mlp"]

    assert "ln_f" in hook_points
    check_ln_hook_points(hook_points["ln_f"])
