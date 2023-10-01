from jaxtyping import Int, Array, Float
from typing import Dict, List, Optional, Union

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


class GenerativeModel:
    config: TransformerConfig
    tokenizer: PreTrainedTokenizerBase
    variables: DArray

    def __init__(
        self,
        config: TransformerConfig,
        variables: Union[DArray, None] = None,
        tokenizer: Union[PreTrainedTokenizerBase, None] = None,
    ):
        self.config = config
        self.variables = variables

        if tokenizer is not None:
            self.configure_tokenizer(tokenizer)

    def configure_tokenizer(
        self, tokenizer: Union[PreTrainedTokenizerBase, None] = None
    ):
        if tokenizer is None and self.tokenizer is None:
            raise ValueError("Tokenizer not provided")
        elif tokenizer is not None:
            self.tokenizer = tokenizer

        self.tokenizer = configure_tokenizer(self.tokenizer)

    def to_tokens(
        self,
        input: str,
        prepend_bos: bool = False,
        truncate: bool = True,
        max_length: Union[int, None] = 1024,
    ) -> Int[Array, "seq"]:
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

    def to_str(self, tokens: Int[Array, "seq"], clean_spaces: bool = False) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=clean_spaces)

    def to_str_list(
        self,
        input: Union[str, Int[Array, "seq"]],
        prepend_bos: bool = False,
        truncate: bool = True,
        max_length: Union[int, None] = 1024,
    ) -> List[str]:
        if isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos, truncate, max_length)
        else:
            tokens = input

        return list(map(self.to_str, tokens))

    def run(
        self, inputs: Int[Array, "seq"], hooks: Optional[HookMap]
    ) -> Float[Array, "seq vocab"]:
        if self.variables is None:
            raise ValueError("Variables not provided")

        transformer = Transformer.from_config(self.config)
        return transformer.apply(self.variables, inputs, hooks)

    def __call__(self, inputs: Int[Array, "seq"]) -> Float[Array, "seq vocab"]:
        return self.run(inputs)


if __name__ == "__main__":
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from transformers import GPT2TokenizerFast

    gpt2 = PretrainedGPT2Model.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GenerativeModel(
        gpt2.tx_config, variables={"params": gpt2.to_params()}, tokenizer=tokenizer
    )

    input_text = "Hello, my name is"
    print(input_text, end="", flush=True)
    for _ in range(10):
        x = model.generate(input_text)
        print(x, end="", flush=True)
        input_text += x

    print()
