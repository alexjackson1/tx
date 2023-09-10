import pytest
from unittest.mock import Mock
from transformers import PreTrainedTokenizerBase
from tx.tokens import (
    prepends_bos_token,
    configure_tokenizer,
    to_tokens,
    to_str,
)


# Mock the tokenizer
@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token = "[EOS]"
    tokenizer.special_tokens_map = {}
    return tokenizer


# Test prepends_bos_token
def test_prepends_bos_token_with_bos_token(mock_tokenizer):
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}  # Mock non-empty input_ids
    assert prepends_bos_token(mock_tokenizer) is True


def test_prepends_bos_token_without_bos_token(mock_tokenizer):
    mock_tokenizer.return_value = {
        "input_ids": [2, 3]
    }  # Mock input_ids without bos_token_id
    assert prepends_bos_token(mock_tokenizer) is False


# Test configure_tokenizer
def test_configure_tokenizer(mock_tokenizer):
    mock_tokenizer = configure_tokenizer(mock_tokenizer)
    assert mock_tokenizer.add_special_tokens.call_count == 3

    keys = []
    for call in mock_tokenizer.add_special_tokens.call_args_list:
        keys.extend(call[0][0].keys())

    assert "eos_token" in keys
    assert "pad_token" in keys
    assert "bos_token" in keys


# Test to_tokens
def test_to_tokens(mock_tokenizer):
    input_text = "Hello, World!"
    expected_input_ids = [[1, 2, 3]]  # Mocked input_ids
    mock_tokenizer.return_value = {"input_ids": expected_input_ids}
    result = to_tokens(input_text, mock_tokenizer)
    assert result == expected_input_ids[0]


# Test tokens_to_str
def test_tokens_to_str(mock_tokenizer):
    input_ids = [1, 2, 3]  # Mocked input_ids
    expected_text = "Hello, World!"  # Mocked decoded text
    mock_tokenizer.batch_decode.return_value = [expected_text]
    result = to_str(mock_tokenizer, input_ids)
    assert result == [expected_text]


# Run tests
if __name__ == "__main__":
    pytest.main()
