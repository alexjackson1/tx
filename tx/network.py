from typing import Dict, List, Union
from jaxtyping import Int, Array

from transformers import PreTrainedTokenizerBase

from tx.models.gpt2 import PretrainedGPT2Model
from tx.modules import Transformer
from tx.modules.transformer import TransformerConfig
import tx.tokens as token_utils


DArray = Union[Dict[str, Array], Dict[str, "DArray"]]


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

        self.tokenizer = token_utils.configure_tokenizer(self.tokenizer)

    def str_to_tokens(
        self,
        input: str,
        prepend_bos: bool = False,
        truncate: bool = True,
        max_length: Union[int, None] = 1024,
    ) -> Int[Array, "seq"]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")

        return token_utils.to_tokens(
            input, self.tokenizer, prepend_bos, truncate, max_length
        )

    def tokens_to_str(self, tokens: Int[Array, "seq"]) -> List[str]:
        return token_utils.to_str(self.tokenizer, tokens)

    def as_str_tokens(
        self,
        input: str,
        prepend_bos: bool = False,
        truncate: bool = True,
        max_length: Union[int, None] = 1024,
    ) -> List[str]:
        tokens = self.str_to_tokens(input, prepend_bos, truncate, max_length)
        return self.tokens_to_str(tokens)

    def generate(self, sequence: str, max_length: Int = 50) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")
        if self.variables is None:
            raise ValueError("Variables not provided")

        inputs = self.str_to_tokens(sequence, prepend_bos=True)
        inputs = self.tokenizer.encode(sequence, return_tensors="jax")
        inputs = inputs.reshape(-1)
        transformer = Transformer.from_config(self.config)
        outputs = transformer.apply(self.variables, inputs)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        outputs = outputs.argmax(axis=-1)[-1]
        return self.tokenizer.decode(outputs, clean_up_tokenization_spaces=False)

    def __call__(self, inputs: Array, intermediates: List[str] = []) -> Array:
        if self.variables is None:
            raise ValueError("Variables not provided")

        transformer = Transformer.from_config(self.config, intermediates=intermediates)
        return transformer.apply(self.variables, inputs, mutable=["intermediates"])


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
