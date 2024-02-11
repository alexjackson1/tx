from jax import config

config.update("jax_enable_x64", True)

import jax


def print_intermediates(intermediates: dict):
    leafs = jax.tree_util.tree_flatten_with_path(intermediates)[0]
    for leaf in leafs:
        key = []
        for p in leaf[0]:
            if isinstance(p, jax.tree_util.SequenceKey):
                key.append(str(p.idx))
            elif isinstance(p, jax.tree_util.DictKey):
                key.append(p.key)
            else:
                key.append(p)
        key = "/".join(key)
        print(f"{key}: {leaf[1].shape}")


if __name__ == "__main__":
    from tx.modules import Transformer
    from tx.models import PretrainedGPT2Model
    from tx.modules.intermediate import Intermediate

    config = PretrainedGPT2Model.tx_config
    intermediates = [
        i.value
        for i in [
            Intermediate.RESIDUAL,
            Intermediate.ATTN_QUERY,
            Intermediate.ATTN_KEY,
            Intermediate.ATTN_OUTPUT,
        ]
    ]

    # Create model and input
    model = Transformer.from_config(config, intermediates=intermediates)
    rand_input = jax.random.randint(jax.random.PRNGKey(0), (1, 1024), 0, 50257)

    # Initialise model parameters (either randomly or from pretrained model)
    variables = model.init(jax.random.PRNGKey(0), rand_input)
    # variables = {"params": PretrainedGPT2Model.from_pretrained("gpt2").to_params()}

    # Apply model to input and print intermediate values (activations, etc.)
    logits, state = model.apply(variables, rand_input, mutable=["intermediates"])
    print_intermediates(state["intermediates"])
