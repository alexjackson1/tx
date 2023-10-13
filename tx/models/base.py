from dataclasses import dataclass
from jaxtyping import Array, PyTree
from typing import Literal, NamedTuple, Type

import flax.linen as nn

from transformers import PreTrainedTokenizerBase


@dataclass
class TransformerConfig:
    decode: bool = False
    context_length: int = 1024


class BaseTransformer(nn.Module):
    config: TransformerConfig

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "BaseTransformer":
        raise NotImplementedError()


class IndexEntry(NamedTuple):
    model_name: str
    module_class: Type[BaseTransformer]
    config: TransformerConfig
    tokenizer: PreTrainedTokenizerBase
    params: PyTree[Array]


ModelKey = Literal["gpt2"]


def load_pretrained_model(model: ModelKey) -> IndexEntry:
    if model.startswith("gpt2"):
        from tx.models.gpt2 import GPT2Loader, GPT2Transformer

        config = GPT2Loader.tx_config
        tokenizer = GPT2Loader.load_tokenizer()
        params = GPT2Loader.load_params()
        module_class = GPT2Transformer
        return IndexEntry("gpt2", module_class, config, tokenizer, params)
    else:
        raise ValueError(f"Unknown model: {model}")
