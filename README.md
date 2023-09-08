# tx

This library is intended to be a reimplementation of the 'TransformerLens'
library[^1] using JAX and the Flax module system.

[^1]: Formerly 'EasyTransformer', 'TransformerLens' is maintained by Joseph Bloom and was created by Neel Nanda.

## Installation

The following prerequisites are required to use the library.

1. A Python 3.7+ installation.
2. A working installation of `jax` and `jaxlib` (either CPU or GPU).
3. Other module requirements (see `requirements.txt`).

### 1. Create Virtual Environment

Assuming you have a working Python 3.7+ installation, you should first clone this project into a new directory `<project_dir>`, and then create and upgrade a virtual environment in `<env_dir>`.

```bash
git clone https://github.com/alexjackson1/tx.git <project_dir>
cd <project_dir>
python -m venv <env_dir>
source <env_dir>/bin/activate
pip install --upgrade pip
```

### 2. Install a Compatible Version of JAX

To install a version of JAX that is compatible with your hardware, please refer to the [JAX installation instructions](https://github.com/google/jax#installation) on the project README.
Installation via the `pip` wheel(s) is highly recommended.

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

## Relation to `TransformerLens`

The API of this library is intended to (eventually) expose the same functionality as the original 'TransformerLens' library, making some changes where appropriate.

### Similarities

1. The library seeks to model Transformer architectures and enable users to inspect intermediate activations and other hidden information (e.g. attention weights).
2. Modules are written 'from scratch', attempting to eliminate abstractions that obfuscate the underlying mathematics.
3. GPU acceleration is supported as a first-class feature.

### Differences

1. The transformer architecture and related algorithms use JAX, instead of PyTorch, for better performance and hardware acceleration.
2. In-keeping with the functional paradigm of JAX, the library and API are designed to be more functional in nature and embrace the [Flax philosophy](https://flax.readthedocs.io/en/latest/philosophy.html).
3. Module definitions use a 'single batch' style made possible by `jax.vmap` (reducing cognitive load and improving readability).

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
