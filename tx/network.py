from jaxtyping import Array, Float, PyTree
from typing import Iterable, List, Literal, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import jax.random as jr

from transformers import PreTrainedTokenizerBase

from .hooks import HookFn, compose_hook_trees, store_hook
from .models import (
    ModelKey,
    TransformerConfig,
    BaseTransformer,
    load_pretrained_model,
)

from .tree_util import Params


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
    """Transformer configuration object."""
    module_class: Type[BaseTransformer]
    """Transformer module class."""
    module: Optional[BaseTransformer] = None
    """Transformer module."""
    params: Optional[PyTree[Array]] = None
    """Model parameters."""
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    """Tokenizer to use for converting between strings and tokens."""
    cache: Optional[PyTree[Array]] = None
    """Cache to use for decoding."""
    mutable: List[str] = []
    """Mutable collections to use for decoding."""

    def __init__(
        self,
        module_cls: Type[BaseTransformer],
        config: TransformerConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        params: Optional[Params] = None,
    ):
        self.config = config
        self.module_class = module_cls

        # Configure the tokenizer if provided
        if tokenizer is not None:
            self.tokenizer = configure_tokenizer(tokenizer)

        # Set the parameters if provided
        if params is not None:
            self.params = params

        # If decoding, set the cache as mutable
        if self.config.decode:
            self.mutable = ["cache"]

        # Initialise the module
        self.init_module()

    @classmethod
    def from_pretrained(
        cls, model_id: ModelKey, tokenizer: Optional[PreTrainedTokenizerBase] = None
    ) -> "GenerativeModel":
        model_details = load_pretrained_model(model_id)
        return cls(
            model_details.module_class,
            model_details.config,
            tokenizer=model_details.tokenizer if tokenizer is None else tokenizer,
            params=model_details.params,
        )

    def init_module(self):
        """Initialise the module."""
        if self.module is None:
            self.module = self.module_class.from_config(self.config)

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerBase):
        """Set the tokenizer."""
        self.tokenizer = configure_tokenizer(tokenizer)

    def reset_cache(self):
        """Reset the cache."""
        self.cache = None
        self.mutable = ["cache"] if self.config.decode else []

    def reset_params(self):
        """Reset the parameters."""
        self.params = None

    def reset(self):
        """Reset the model."""
        self.reset_cache()
        self.reset_params()

    def to_tokens(
        self,
        input: String,
        prepend_bos: bool = False,
        truncate: bool = True,
        max_length: Union[int, None] = 1024,
        extra_batch_dims: int = 0,
    ) -> Tokens:
        """Convert a string to an array of token(s).

        Args:
            input: Input string.
            prepend_bos: Whether to prepend the BOS token.
            truncate: Whether to truncate the input.
            max_length: Maximum length of the input.
            extra_batch_dims: Number of extra batch dimensions to add.

        Returns:
            The token(s) corresponding to the input string.
        """
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
        """Convert a (array of) token(s) to a string.

        Args:
            tokens: Input token(s).
            clean_spaces: Whether to clean up the tokenisation spaces.

        Returns:
            The string corresponding to the input token(s).
        """

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
        """Convert a (array of) token(s) to a list of strings.

        Args:
            input: Input token(s) or string.
            prepend_bos: Whether to prepend the BOS token.
            truncate: Whether to truncate the input.
            max_length: Maximum length of the input.

        Returns:
            The list of strings corresponding to the input token(s).
        """
        if isinstance(input, String):
            tokens = self.to_tokens(input, prepend_bos, truncate, max_length)
        elif isinstance(input, int):
            tokens = [input]
        else:
            tokens = input

        return list(map(self.to_str, tokens))

    def run(
        self,
        tokens: Tokens,
        hooks: Optional[PyTree[HookFn]] = None,
        mutable: List[str] = [],
    ) -> Tuple[Logits, PyTree[Array]]:
        if self.module is None:
            raise ValueError("Module not initialised")

        if self.params is None:
            raise ValueError("Params not initialised")

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
        mutable = list_union(mutable, self.mutable)

        # Apply the model to the input token(s)
        logits, state = self.module.apply(variables, tokens, hooks, mutable=mutable)

        # Store the updated cache state
        if self.config.decode:
            self.cache = state["cache"]

        return logits, state

    def run_with_intermediates(
        self,
        tokens: Tokens,
        hooks: Optional[PyTree[HookFn]] = None,
        mutable: List[str] = [],
    ) -> Tuple[Logits, PyTree[Array]]:
        # TODO: Not working (need to replace self.params with blank hooks)
        store_hooks = jax.tree_util.tree_map(lambda _: store_hook, self.params)

        if hooks is not None:
            hooks = compose_hook_trees(store_hooks, hooks)

        if len(mutable) != 0:
            mutable = list_union(mutable, self.mutable)

        return self.run(tokens, hooks, mutable)

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
