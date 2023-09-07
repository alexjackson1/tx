from tx.modules import TransformerConfig, Transformer


if __name__ == "__main__":
    import jax

    from debug_utils import tree_print

    config = TransformerConfig()

    rng = jax.random.PRNGKey(0)
    tokens = jax.random.randint(rng, (1, config.context_length), 0, config.vocab_dim)

    model = Transformer()
    variables = model.init(rng, tokens)
    logits, state = model.apply(variables, tokens, mutable=["intermediates"])
    tree_print(state["intermediates"])
