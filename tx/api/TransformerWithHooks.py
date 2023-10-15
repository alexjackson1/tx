from jaxtyping import Array, Float, PyTree, Int
from typing import List, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from transformers import PreTrainedTokenizer

from tx.models import TransformerModel, load_hf_model
from tx.hooks import HookFn, compose_hook_trees, store_hook
import tx.tokens as token_ops


ReturnType = Literal["logits", "preds", "loss"]


def next_token_loss(
    logits: Float[Array, "... pos vocab"], tokens: Int[Array, "... pos"]
) -> Float[Array, "... pos"]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    offset = log_probs[..., :-1, :]
    probs = jnp.take_along_axis(offset, tokens[..., 1:, None], axis=-1)[..., 0]
    return -probs


class TransformerWithHooks:
    model: TransformerModel
    """Transformer model."""

    def __init__(self, model: TransformerModel):
        self.model = model
        self.config = model.config
        self.decode = model.config.decode

    @classmethod
    def from_pretrained(
        cls, model_id: str, decode: bool = False, dtype: Optional[jnp.dtype] = None
    ) -> "TransformerWithHooks":
        model = load_hf_model(model_id, decode=decode, dtype=dtype)
        return cls(model)

    @property
    def params(self) -> PyTree[Array]:
        """Model parameters."""
        return self.model.params

    @property
    def cache(self) -> Optional[PyTree[Array]]:
        """Model cache."""
        return self.model.cache

    @property
    def mutable(self) -> List[str]:
        """Mutable state."""
        return self.model.mutable

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Tokenizer."""
        return self.model.tokenizer

    def to_tokens(
        self,
        input: str,
        prepend_bos: bool = False,
        truncate: bool = True,
        max_length: Optional[int] = 1024,
        extra_batch_dims: int = 0,
    ) -> Int[Array, "... S"]:
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
        if self.model.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        return token_ops.to_tokens(
            self.model.tokenizer,
            input,
            prepend_bos,
            truncate,
            max_length,
            extra_batch_dims,
        )

    def to_single_token(self, input: str) -> int:
        """Convert a string to a single token.

        Args:
            input: Input string.

        Returns:
            The token corresponding to the input string.
        """
        if self.model.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        return token_ops.to_single_token(self.model.tokenizer, input)

    def to_str(self, tokens: Int[Array, "... S"], clean_spaces: bool = False) -> str:
        """Convert a (array of) token(s) to a string.

        Args:
            tokens: Input token(s).
            clean_spaces: Whether to clean up the tokenisation spaces.

        Returns:
            The string corresponding to the input token(s).
        """

        if self.model.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        return token_ops.to_str(self.model.tokenizer, tokens, clean_spaces)

    def to_str_list(
        self,
        input: Union[Int[Array, "... S"], str],
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
        if self.model.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        return token_ops.to_str_list(
            self.model.tokenizer, input, prepend_bos, truncate, max_length
        )

    def run_with_intermediates(
        self,
        tokens: Int[Array, "... pos"],
        hooks: Optional[PyTree[HookFn]] = None,
        mutable: List[str] = [],
    ) -> Tuple[Float[Array, "... pos vocab"], PyTree[Array]]:
        hook_points = self.model.hook_points()
        store_hooks = jax.tree_util.tree_map(lambda _: store_hook, hook_points)

        if hooks is not None:
            hooks = compose_hook_trees(store_hooks, hooks)
        else:
            hooks = store_hooks

        mutable = list(set(mutable).union(["intermediates"]))
        return self.model.run(tokens, hooks, mutable)

    def __call__(
        self,
        input: Union[Int[Array, "... S"], str],
        hooks: Optional[PyTree[HookFn]] = None,
        mutable: List[str] = [],
        return_type: Optional[ReturnType] = None,
    ) -> Tuple[Float[Array, "... S V"], PyTree[Array]]:
        if isinstance(input, str):
            tokens = self.to_tokens(input)
        elif isinstance(input, int):
            tokens = jnp.array([input], jnp.int32)
        else:
            tokens = input

        logits, state = self.model.run(tokens, hooks, mutable)
        if return_type is None or return_type == "logits":
            return logits, state

        preds = jnp.argmax(jax.nn.softmax(logits, axis=-1), axis=-1)
        if return_type == "preds":
            return preds, state

        if return_type == "loss":
            loss = next_token_loss(logits, tokens)
            return loss.mean(), state
