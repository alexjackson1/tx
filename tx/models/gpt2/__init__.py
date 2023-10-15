from typing import Optional
from jaxtyping import PyTree, Array

from transformers import PreTrainedTokenizer, AutoTokenizer

from tx.models import TransformerModel

from .loader import GPT2Loader
from .modules import GPT2Config, GPT2Transformer


class GPT2TransformerModel(TransformerModel):
    config: GPT2Config
    module: GPT2Transformer
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        config: GPT2Config,
        model_name: str,
        tokenizer_name: str,
        cache: Optional[PyTree[Array]] = None,
    ):
        super().__init__(
            config,
            GPT2Transformer.from_config(config),
            AutoTokenizer.from_pretrained(tokenizer_name),
            GPT2Loader.load_params(model_name, config),
            cache=cache,
        )
