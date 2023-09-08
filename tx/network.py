from typing import Dict, List, Union
from jaxtyping import Int, Array

from transformers import PreTrainedTokenizerBase

from tx.models.gpt2 import PretrainedGPT2Model
from tx.modules import Transformer
from tx.modules.transformer import TransformerConfig


DArray = Union[Dict[str, Array], Dict[str, "DArray"]]


class GenerativeModel:
    module: Transformer
    tokenizer: PreTrainedTokenizerBase
    variables: DArray

    def __init__(
        self,
        config: TransformerConfig,
        variables: Union[DArray, None] = None,
        tokenizer: Union[PreTrainedTokenizerBase, None] = None,
    ):
        self.module = Transformer.from_config(config)
        self.variables = variables

        if tokenizer is not None:
            self.configure_tokenizer(tokenizer)

    @property
    def _add_special_tokens(self):
        return not (
            len(tokenizer("")["input_ids"]) > 0
            and tokenizer("")["input_ids"][0] == self.tokenizer.bos_token_id
        )

    def configure_tokenizer(
        self, tokenizer: Union[PreTrainedTokenizerBase, None] = None
    ):
        if tokenizer is None and self.tokenizer is None:
            raise ValueError("Tokenizer not provided")
        elif tokenizer is not None:
            self.tokenizer = tokenizer

        # Add special tokens if they are not already added
        token_map = self.tokenizer.special_tokens_map
        if "eos_token" not in token_map or token_map["eos_token"] is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if "pad_token" not in token_map or token_map["pad_token"] is None:
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
        if "bos_token" not in token_map or token_map["bos_token"] is None:
            self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.eos_token})

        tokenizer.padding_side = "right"

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: Union[bool, None] = False,
        truncate: bool = True,
        max_length: Union[int, None] = 1024,
    ) -> Int[Array, "batch pos"]:
        assert tokenizer is not None, "Cannot use to_tokens without a tokenizer"
        if prepend_bos is not None and prepend_bos:
            if isinstance(input, str):
                input = self.tokenizer.bos_token + input
            else:
                input = [self.tokenizer.bos_token + string for string in input]

        cfg = {
            "return_tensors": "jax",
            "padding": True,
            "truncation": truncate,
            "max_length": max_length if truncate else None,
            "add_special_tokens": self._add_special_tokens,
        }
        tokens = tokenizer(input, **cfg)["input_ids"]
        return tokens

    def generate(self, sequence: str, max_length: Int = 50) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")
        if self.variables is None:
            raise ValueError("Variables not provided")

        inputs = self.to_tokens(sequence, prepend_bos=True)
        inputs = self.tokenizer.encode(sequence, return_tensors="jax")
        inputs = inputs.reshape(-1)
        outputs = self.module.apply(self.variables, inputs)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        outputs = outputs.argmax(axis=-1)[-1]
        return self.tokenizer.decode(outputs, clean_up_tokenization_spaces=False)

    def __call__(self, inputs: Array) -> Array:
        return self.module.apply(self.variables, inputs)


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
