from jax.config import config

from tx.hooks import Hook, HookPoint

config.update("jax_enable_x64", True)

import jax


if __name__ == "__main__":
    from tx.modules import Transformer
    from tx.models import PretrainedGPT2Model

    config = PretrainedGPT2Model.tx_config

    def make_hook(name):
        def hook(x):
            print(f"{name}: {x.shape}")
            return x

        return Hook(hook)

    hooks = {
        HookPoint.ATTN_QUERY.value: make_hook("Query"),
        HookPoint.ATTN_KEY.value: make_hook("Key"),
        HookPoint.ATTN_OUTPUT.value: make_hook("Output"),
    }

    # Create model and input
    model = Transformer.from_config(config)
    rand_input = jax.random.randint(jax.random.PRNGKey(0), (1, 1024), 0, 50257)

    # Initialise model parameters (either randomly or from pretrained model)
    variables = model.init(jax.random.PRNGKey(0), rand_input)
    # variables = {"params": PretrainedGPT2Model.from_pretrained("gpt2").to_params()}

    # Apply model to input and print intermediate values (activations, etc.)
    model.apply(variables, rand_input, hooks)
