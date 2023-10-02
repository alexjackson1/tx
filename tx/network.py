from jaxtyping import Int, Array, Float
from typing import Dict, List, Optional, Tuple, Union
from optax import Params

from transformers import PreTrainedTokenizerBase

from .modules import Transformer, TransformerConfig, HookMap
from .models.gpt2 import PretrainedGPT2Model


DArray = Union[Dict[str, Array], Dict[str, "DArray"]]


def prepends_bos_token(tokenizer: PreTrainedTokenizerBase) -> bool:
    bos_token_id = tokenizer.bos_token_id
    blank_input_ids = tokenizer("")["input_ids"]
    return len(blank_input_ids) > 0 and blank_input_ids[0] == bos_token_id


def configure_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    token_map = tokenizer.special_tokens_map
    if "eos_token" not in token_map or token_map["eos_token"] is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if "pad_token" not in token_map or token_map["pad_token"] is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    if "bos_token" not in token_map or token_map["bos_token"] is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})

    tokenizer.padding_side = "right"
    return tokenizer


Tokens = Int[Array, "S"]
String = str
TokensOrString = Union[Tokens, String]
Logits = Float[Array, "S"]


class GenerativeModel:
    config: TransformerConfig
    decode: bool

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    params: Optional[Params] = None
    hooks: HookMap
    mutable: List[str]
    cache: Optional[Params] = None

    def __init__(
        self,
        config: TransformerConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        params: Optional[Params] = None,
        hooks: Optional[HookMap] = None,
        hook_collections: List[str] = [],
    ):
        self.config = config
        self.decode = config.decode

        if self.decode:
            self.reset_cache()

        self.mutable = ["cache"] if self.config.decode else []

        if tokenizer is not None:
            self.tokenizer = configure_tokenizer(tokenizer)

        if params is not None:
            self.params = params

        if hooks is not None:
            self.hooks = hooks
            self.mutable = list(set(self.mutable).union(set(hook_collections)))
        else:
            self.hooks = HookMap()

    def reset_cache(self) -> None:
        self.cache = {}

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = configure_tokenizer(tokenizer)

    def set_params(self, params: Params) -> None:
        self.params = params

    def set_hooks(self, hooks: HookMap, collections: List[str] = []) -> None:
        self.mutable = ["cache"] if self.config.decode else []
        self.mutable = list(set(self.mutable).union(set(collections)))
        self.hooks = hooks

    def to_tokens(
        self,
        input: String,
        prepend_bos: bool = False,
        truncate: bool = True,
        max_length: Union[int, None] = 1024,
    ) -> Tokens:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        text = input if not prepend_bos else self.tokenizer.bos_token + input
        max_length = max_length if truncate else None
        add_special_tokens = not prepends_bos_token(self.tokenizer)
        output = self.tokenizer(
            text,
            return_tensors="jax",
            padding=True,
            truncation=truncate,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return output["input_ids"][0]

    def to_str(self, tokens: Tokens, clean_spaces: bool = False) -> String:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=clean_spaces)

    def to_str_list(
        self,
        input: TokensOrString,
        prepend_bos: bool = False,
        truncate: bool = True,
        max_length: Union[int, None] = 1024,
    ) -> List[str]:
        if isinstance(input, String):
            tokens = self.to_tokens(input, prepend_bos, truncate, max_length)
        else:
            tokens = input

        return list(map(self.to_str, tokens))

    def model(self) -> Transformer:
        return Transformer.from_config(self.config)

    def __call__(self, tokens: Tokens) -> Tuple[Logits, Params]:
        if self.params is None:
            raise ValueError("Params not initialised")

        variables = {"params": self.params}
        if self.decode:
            variables["cache"] = self.cache

        model = self.model()
        logits, state = model.apply(variables, tokens, self.hooks, mutable=self.mutable)

        if self.decode:
            self.cache = state["cache"]

        return logits, state
