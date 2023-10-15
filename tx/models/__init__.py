from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Array, PyTree, Float, Int
from typing import List, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

from transformers import PreTrainedTokenizer, AutoConfig, PretrainedConfig

import tx.hooks as hook_module
from tx.tokens import configure_tokenizer


@dataclass
class TransformerConfig:
    decode: bool = False
    context_length: int = 1024


class TransformerModule(ABC, nn.Module):
    config: TransformerConfig

    @staticmethod
    @abstractmethod
    def hook_points() -> PyTree[str]:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: TransformerConfig) -> "TransformerModule":
        pass

    @abstractmethod
    def __call__(
        self,
        variables: PyTree[Array],
        tokens: Int[Array, "... pos"],
        hooks: Optional[PyTree[hook_module.HookFn]] = None,
        mutable: List[str] = [],
    ) -> Tuple[Float[Array, "... pos vocab"], PyTree[Array]]:
        pass


class TransformerModel:
    config: TransformerConfig
    module: TransformerModule
    tokenizer: PreTrainedTokenizer
    params: PyTree[Array]
    cache: Optional[PyTree[Array]] = None
    mutable: List[str] = []

    def __init__(
        self,
        config: TransformerConfig,
        module: TransformerModule,
        tokenizer: PreTrainedTokenizer,
        params: PyTree[Array],
        cache: Optional[PyTree[Array]] = None,
    ):
        self.config = config
        self.module = module
        self.tokenizer = configure_tokenizer(tokenizer)
        self.params = params

        if cache is not None:
            self.cache = cache
            self.mutable = ["cache"]
        elif config.decode:
            state = module.init(
                jr.PRNGKey(0), jnp.ones((config.context_length,), jnp.int32)
            )
            self.cache = state["cache"]
            self.mutable = ["cache"]

    def hook_points(self) -> PyTree[str]:
        return self.module.hook_points()

    def run(
        self,
        tokens: Int[Array, "... pos"],
        hooks: Optional[PyTree[hook_module.HookFn]] = None,
        mutable: List[str] = [],
    ) -> Tuple[Float[Array, "... pos vocab"], PyTree[Array]]:
        # Initialise the cache if using
        if self.config.decode and self.cache is None:
            input_ids = jnp.ones((self.module.config.context_length,), jnp.int32)
            variables = self.module.init(jr.PRNGKey(0), input_ids)
            self.cache = variables["cache"]

        # Prepare the model variables (params and optional cache)
        variables = {"params": self.params}
        if self.config.decode:
            variables["cache"] = self.cache

        # Join the provided mutable collections with existing
        mutable = list(set(mutable).union(set(self.mutable)))

        # Apply the model to the input token(s)
        logits, state = self.module.apply(variables, tokens, hooks, mutable=mutable)

        # Store the updated cache state
        if self.config.decode:
            self.cache = state["cache"]

        return logits, state


def load_hf_model(
    model_name: str, decode: bool = False, dtype: Optional[jnp.dtype] = None, **kwargs
) -> TransformerModel:
    hf_config: PretrainedConfig = AutoConfig.from_pretrained(model_name, **kwargs)
    if hf_config.architectures is None:
        raise ValueError(f"Model with name {model_name} had no architectures")

    architecture = hf_config.architectures[0]

    if architecture == "GPT2LMHeadModel":
        from transformers import GPT2Config as HfGPT2Config
        from tx.models.gpt2 import GPT2TransformerModel, GPT2Config, GPT2Loader

        hf_config: HfGPT2Config = hf_config
        config = GPT2Config(
            decode=decode,
            model_dim=hf_config.n_embd,
            num_heads=hf_config.n_head,
            num_layers=hf_config.n_layer,
            vocab_dim=hf_config.vocab_size,
            context_length=hf_config.n_positions,
            head_dim=hf_config.n_embd // hf_config.n_head,
            layer_norm_eps=hf_config.layer_norm_epsilon,
            mlp_dim=hf_config.n_embd * 4,
            dtype=dtype,
            param_dtype=dtype,
        )
        return GPT2TransformerModel(config, model_name, tokenizer_name=model_name)
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
