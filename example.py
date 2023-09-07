if __name__ == "__main__":
    import jax
    from tx.modules import Transformer
    from tx.models import PretrainedGPT2Model

    from debug_utils import tree_print

    # Create model and input
    model = Transformer.from_config(PretrainedGPT2Model.tx_config)
    rand_input = jax.random.randint(jax.random.PRNGKey(0), (1, 1024), 0, 50257)

    # Initialise model parameters (either randomly or from pretrained model)
    # variables = model.init(jax.random.PRNGKey(0), rand_input)
    variables = {"params": PretrainedGPT2Model.from_pretrained("gpt2").to_params()}

    # Apply model to input and print intermediate values (activations, etc.)
    logits, state = model.apply(variables, rand_input, mutable=["intermediates"])
    tree_print(state["intermediates"])
