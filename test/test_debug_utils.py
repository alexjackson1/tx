import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from jax import numpy as jnp
from debug_utils import tree_print

test_data_array = jnp.array([1, 2, 3])
test_data_dict = {"a": 1, "b": [2, test_data_array], "c": {"x": 4, "y": [5, 6]}}
test_data_list = [1, 2, [3, 4], {"a": 5, "b": [6, 7]}]
test_data_scalar = 42


def test_tree_print_dict(capsys):
    tree_print(test_data_dict)
    captured = capsys.readouterr()
    expected_output = """a: 1.
b:
  0: 2.
  1: (3,).
c:
  x: 4.
  y:
    0: 5.
    1: 6.
"""
    assert captured.out == expected_output


def test_tree_print_list(capsys):
    tree_print(test_data_list)
    captured = capsys.readouterr()
    expected_output = """0: 1.
1: 2.
2:
  0: 3.
  1: 4.
3:
  a: 5.
  b:
    0: 6.
    1: 7.
"""
    assert captured.out == expected_output


def test_tree_print_array(capsys):
    tree_print(test_data_array)
    captured = capsys.readouterr()
    expected_output = "(3,)."
    assert captured.out.strip() == expected_output


def test_tree_print_scalar(capsys):
    tree_print(test_data_scalar)
    captured = capsys.readouterr()
    expected_output = "42."
    assert captured.out.strip() == expected_output


if __name__ == "__main__":
    pytest.main([__file__])
