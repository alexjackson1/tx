from typing import List, Union
from jaxtyping import Array, Int
from transformers import PreTrainedTokenizerBase


def prepends_bos_token(tokenizer: PreTrainedTokenizerBase) -> bool:
    bos_token_id = tokenizer.bos_token_id
    blank_input_ids = tokenizer("")["input_ids"]
    return not (len(blank_input_ids) > 0 and blank_input_ids[0] == bos_token_id)


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


def to_tokens(
    input: str,
    tokenizer: PreTrainedTokenizerBase,
    prepend_bos: bool = False,
    truncate: bool = True,
    max_length: Union[int, None] = 1024,
) -> Int[Array, "seq"]:
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
    return output["input_ids"][0]


def tokens_to_str(
    tokenizer: PreTrainedTokenizerBase, tokens: Int[Array, "seq"]
) -> List[str]:
    return tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
