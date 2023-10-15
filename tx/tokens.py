from functools import partial
from typing import List, Optional, Union, Literal
from jaxtyping import Int, Array, Float

import jax.numpy as jnp

from transformers import PreTrainedTokenizer

TokensOrString = Union[Int[Array, "S"], str]
Logits = Float[Array, "S"]
ReturnType = Literal["logits", "preds", "loss"]


def prepends_bos_token(tokenizer: PreTrainedTokenizer) -> bool:
    bos_token_id = tokenizer.bos_token_id
    blank_input_ids = tokenizer("")["input_ids"]
    return len(blank_input_ids) > 0 and blank_input_ids[0] == bos_token_id


def configure_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    token_map = tokenizer.special_tokens_map
    if "eos_token" not in token_map or token_map["eos_token"] is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if "pad_token" not in token_map or token_map["pad_token"] is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    if "bos_token" not in token_map or token_map["bos_token"] is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})

    tokenizer.padding_side = "right"
    return tokenizer


def to_tokens(
    tokenizer: PreTrainedTokenizer,
    input: str,
    prepend_bos: bool = False,
    truncate: bool = True,
    max_length: Optional[int] = 1024,
    extra_batch_dims: int = 0,
) -> Int[Array, "S"]:
    """Convert a string to an array of token(s).

    Args:
        tokenizer: Tokenizer to use.
        input: Input string.
        prepend_bos: Whether to prepend the BOS token.
        truncate: Whether to truncate the input.
        max_length: Maximum length of the input.
        extra_batch_dims: Number of extra batch dimensions to add.

    Returns:
        The token(s) corresponding to the input string.
    """
    text = input if not prepend_bos else tokenizer.bos_token + input
    max_length = max_length if truncate else None
    add_special_tokens = not prepends_bos_token(tokenizer)
    output = tokenizer(
        text,
        return_tensors="jax",
        padding=True,
        truncation=truncate,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    batch_dims = (0,) * extra_batch_dims
    return jnp.reshape(output["input_ids"][0], (*batch_dims, -1))


def to_single_token(tokenizer: PreTrainedTokenizer, input: str) -> int:
    """Convert a string to a single token.

    Args:
        tokenizer: Tokenizer to use.
        input: Input string.

    Returns:
        The token corresponding to the input string.
    """
    tokens = tokenizer.encode(input)
    if len(tokens) != 1:
        raise ValueError("Input string must be a single token")
    return tokens[0]


def to_str(
    tokenizer: PreTrainedTokenizer, tokens: Int[Array, "S"], clean_spaces: bool = False
) -> str:
    """Convert a (array of) token(s) to a string.

    Args:
        tokenizer: Tokenizer to use.
        tokens: Input token(s).
        clean_spaces: Whether to clean up the tokenisation spaces.

    Returns:
        The string corresponding to the input token(s).
    """

    return tokenizer.decode(tokens, clean_up_tokenization_spaces=clean_spaces)


def to_str_list(
    tokenizer: PreTrainedTokenizer,
    input: Union[Int[Array, "S"], str],
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
    if isinstance(input, str):
        tokens = to_tokens(tokenizer, input, prepend_bos, truncate, max_length)
    elif isinstance(input, int):
        tokens = [input]
    else:
        tokens = input

    return list(map(lambda t: to_str(tokenizer, t), tokens))
