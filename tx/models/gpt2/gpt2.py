from typing import Optional
from jaxtyping import PyTree, Array
import jax.numpy as jnp
from .module import GPT2Transformer, GPT2Config


class GPT2TransformerModel:
    config: GPT2Config
    module: GPT2Transformer
    params: PyTree[Array]
    cache: Optional[PyTree[Array]]

    def __init__(self, config: GPT2Config, params: Optional[PyTree[Array]] = None):
        self.config = config
        self.params = params
        self.module = GPT2Transformer.from_config(self.config)

        if self.params is None or self.config.decode:
            variables = self.module.init(
                jnp.ones((1, self.config.context_length), jnp.int32)
            )

            if self.params is None:
                self.params = variables["params"]

            if self.config.decode:
                self.cache = variables["cache"]
