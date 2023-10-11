from typing import Literal, NamedTuple
from optax import Params

from transformers import PreTrainedTokenizerBase
from tx.modules.transformer import TransformerConfig


class IndexEntry(NamedTuple):
    model_name: str
    config: TransformerConfig
    tokenizer: PreTrainedTokenizerBase
    params: Params


ModelStr = Literal["gpt2"]


def load_pretrained_model(model: ModelStr) -> IndexEntry:
    if model == "gpt2":
        from tx.models.gpt2 import PretrainedGPT2Model

        config = PretrainedGPT2Model.tx_config
        tokenizer = PretrainedGPT2Model.load_tokenizer()
        print("Loading GPT2 params...")
        params = PretrainedGPT2Model.load_params()
        return IndexEntry("gpt2", config, tokenizer, params)
    else:
        raise ValueError(f"Unknown model: {model}")
