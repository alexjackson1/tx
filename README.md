# tx

This library is intended to be a reimplementation of the 'TransformerLens'
library[^1] using JAX and the Flax module system.

[^1]: Formerly 'EasyTransformer', 'TransformerLens' is maintained by Joseph Bloom and was created by Neel Nanda.

## Installation

### 0. Prerequisites

The following prerequisites are required to use this library.

1. A Python 3.7+ installation.
2. A working installation of `jax` and `jaxlib` (either CPU or GPU).
3. Other module requirements (see `requirements.txt`).

### 1. Create Virtual Environment

Assuming you have a working Python 3.7+ installation, you first should clone this project into a new directory; you a can then create a virtual environment and upgrade pip to the latest version as follows.

```bash
git clone https://github.com/alexjackson1/tx.git <project_dir>
cd <project_dir>
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

### 2. Install a Compatible Version of JAX

To install a version of JAX that is compatible with your hardware, please refer to the [JAX installation instructions](https://github.com/google/jax#installation) on the project README.
Installation via `pip` wheels is highly recommended.

### 3. Install the Remaining Requirements

Once you have installed a compatible version of JAX, you can install the remaining requirements as follows.
This includes Flax, the module system used by this library for defining networks.

```bash
pip install -r requirements.txt
```

## Usage

This library is still in development and is not yet ready for use.

An example script is provided that prints the activations of a GPT2-style Transformer using pretrained weights.
To run it, simply execute the following command.

```bash
python example.py
```

## License

This project is licensed under the terms of the MIT license.
The full license text can be found in the [`LICENSE` file](LICENSE).

The original 'TransformerLens' library is also licensed under the terms of the MIT license.
The full license text can be found [here](https://github.com/neelnanda-io/TransformerLens/blob/main/LICENSE).
Additionally, the original library can be cited as shown below.

```bibtex
@misc{nandatransformerlens2022,
    title  = {TransformerLens},
    author = {Nanda, Neel and Bloom, Joseph},
    url    = {https://github.com/neelnanda-io/TransformerLens},
    year   = {2022}
}
```
