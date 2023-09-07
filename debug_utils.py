from typing import Dict, Union
from jaxtyping import Array

Data = Union[Array, Dict[str, "Data"]]


def _as_str(d: Data, i: int = 0) -> str:
    indent = "  " * i
    space = " " if i != 0 else ""

    s = ""
    if isinstance(d, dict):
        for key, value in d.items():
            s += f"\n{indent}{key}:"
            s += _as_str(value, i + 1)  # recurse
    elif isinstance(d, (list, tuple)):
        for j, value in enumerate(d):
            s += f"\n{indent}{j}:"
            s += _as_str(value, i + 1)  # recurse
    elif "shape" in dir(d):
        s += f"{space}{d.shape}."
    else:
        s += f"{space}{d}."

    if i == 0 and len(s) > 0 and s[0] == "\n":
        s = s[1:]
    return s


def tree_print(data: Data):
    print(_as_str(data))
