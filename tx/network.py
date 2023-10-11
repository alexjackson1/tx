import jax
from jaxtyping import Array, Float, PyTree
from typing import Iterable, List, Literal, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr

from transformers import PreTrainedTokenizerBase

from .modules import Transformer, TransformerConfig
from .models import index
from .hooks import CacheAll, HookMap

Params = PyTree[Array]


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


Tokens = Union[int, Iterable[int]]
String = str
TokensOrString = Union[Tokens, String]
Logits = Float[Array, "S"]

ReturnType = Literal["logits", "preds", "loss"]


def list_union(a: List[str], b: List[str]) -> List[str]:
    return list(set(a).union(set(b)))


class GenerativeModel:
    config: TransformerConfig
    decode: bool
    module: Transformer

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    params: Optional[Params] = None
    hooks: HookMap
    mutable: List[str]
    hook_collections: List[str]
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
        self.module = Transformer.from_config(config)

        if tokenizer is not None:
            self.tokenizer = configure_tokenizer(tokenizer)

        if params is not None:
            self.params = params

        if self.decode:
            self.mutable = ["cache"]
        else:
            self.mutable = []

        self.hook_collections = hook_collections
        self.mutable = list_union(self.mutable, hook_collections)
        if hooks is not None:
            self.hooks = hooks
        else:
            self.hooks = {}

    def reset(self):
        self.cache = None
        self.hooks = {}
        self.hook_collections = []
        self.mutable = ["cache"] if self.decode else []

    @classmethod
    def from_pretrained(
        cls,
        model_id: index.ModelStr,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        hooks: Optional[HookMap] = None,
        hook_collections: List[str] = [],
    ) -> "GenerativeModel":
        model_details = index.load_pretrained_model(model_id)
        return cls(
            model_details.config,
            tokenizer=model_details.tokenizer if tokenizer is None else tokenizer,
            params=model_details.params,
            hooks=hooks,
            hook_collections=hook_collections,
        )

    def to_tokens(
        self,
        input: String,
        prepend_bos: bool = False,
        truncate: bool = True,
        max_length: Union[int, None] = 1024,
        extra_batch_dims=0,
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
        batch_dims = (0,) * extra_batch_dims
        return jnp.reshape(output["input_ids"][0], (*batch_dims, -1))

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
        elif isinstance(input, int):
            tokens = [input]
        else:
            tokens = input

        return list(map(self.to_str, tokens))

    def run(self, tokens: Tokens) -> Tuple[Logits, Params]:
        if self.params is None:
            raise ValueError("Params not initialised")

        # Initialise the cache if using
        if self.decode and self.cache is None:
            input_ids = jnp.ones((self.config.context_length,), jnp.int32)
            variables = self.module.init(jr.PRNGKey(0), input_ids)
            self.cache = variables["cache"]

        # Prepare the model variables (params and optional cache)
        variables = {"params": self.params}
        if self.decode:
            variables["cache"] = self.cache

        # Apply the model to the input token(s)
        logits, state = self.module.apply(
            variables, tokens, self.hooks, mutable=self.mutable
        )

        # Store the updated cache state
        if self.decode:
            self.cache = state["cache"]

        output = {k: v for k, v in state.items() if k in self.hook_collections}
        return logits, output

    def run_with_intermediates(self, tokens: Tokens) -> Tuple[Logits, Params, Params]:
        # self.hooks = compose(self.hooks, CacheAll)
        self.hooks = CacheAll
        self.hook_collections = list_union(self.hook_collections, ["intermediates"])
        self.mutable = list_union(self.mutable, ["intermediates"])
        logits, output = self.run(tokens)
        return (logits, output)

    def __call__(
        self, input: TokensOrString, return_type: Optional[ReturnType] = None
    ) -> Tuple[Logits, Params]:
        if isinstance(input, String):
            tokens = self.to_tokens(input)
        elif isinstance(input, int):
            tokens = jnp.array([input], jnp.int32)
        else:
            tokens = input

        logits, state = self.run(tokens)
        if return_type is None or return_type == "logits":
            return logits, state

        preds = jnp.argmax(jax.nn.softmax(logits, axis=-1), axis=-1)
        if return_type == "preds":
            return preds, state

        if return_type == "loss":
            labels = jnp.expand_dims(tokens[..., 1:], axis=-1)
            log_probs = jax.nn.log_softmax(logits[..., :-1, :], axis=-1)
            log_probs = jnp.take_along_axis(log_probs, labels, axis=-1)
            return -log_probs.mean(), state
