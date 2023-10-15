import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from typing import Sequence, Tuple
from jaxtyping import Array, Float, Int, PyTree

import pytest
from pytest_mock import MockerFixture

import jax.numpy as jnp
import jax.random as jr

from tx.hooks import store_hook
from tx.models.gpt2.modules import (
    GPT2Transformer,
    GPT2Config,
    GPT2TransformerBlock,
    GPT2MLP,
)

RNG = jr.PRNGKey(0)
ERROR = 1e-4


def format_params(batch_dims: Sequence[int]) -> str:
    return f"batch_dims={batch_dims}"


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_transformer_init(config: GPT2Config, batch_dims: Sequence[int]):
    model = GPT2Transformer.from_config(config)

    input: Int[Array, "S"] = jnp.ones((*batch_dims, 4), dtype=jnp.int32)
    variables = model.init(RNG, input)

    assert len(variables["params"]) == 16


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_transformer_block_init(config: GPT2Config, batch_dims: Sequence[int]):
    layer = GPT2TransformerBlock(
        num_heads=12,
        head_dim=64,
        model_dim=768,
        mlp_dim=3072,
        epsilon=config.layer_norm_eps,
        init_range=0.02,
    )

    input: Float[Array, "S F"] = jnp.ones((*batch_dims, 4, 768), dtype=jnp.float32)
    variables = layer.init(RNG, input)

    ln1_scale: Array = variables["params"]["ln_1"]["scale"]
    assert ln1_scale.shape == (768,)
    assert jnp.std(ln1_scale) < 0.02 + ERROR

    ln2_scale: Array = variables["params"]["ln_2"]["scale"]
    assert ln2_scale.shape == (768,)
    assert jnp.std(ln2_scale) < 0.02 + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_mlp_init(batch_dims: Sequence[int]):
    layer = GPT2MLP(features=[3072, 768], init_range=0.02)

    input: Float[Array, "S F"] = jnp.ones((*batch_dims, 4, 768), dtype=jnp.float32)
    variables = layer.init(RNG, input)

    proj_kernel: Array = variables["params"]["proj"]["kernel"]
    assert proj_kernel.shape == (3072, 768)
    assert jnp.std(proj_kernel) < 0.02 + ERROR


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_mlp_contract(rng: Array, config: GPT2Config, batch_dims: Sequence[int]):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = GPT2MLP(features=[config.mlp_dim, 768])
    init_input: Float[Array, "... S F"] = jr.uniform(rng, (*batch_dims, INIT_LEN, 768))
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, APPLY_LEN, 768)
    )
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, 768)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_transformer_block_contract(
    rng: Array, config: GPT2Config, batch_dims: Sequence[int]
):
    INIT_LEN = 4
    APPLY_LEN = 7

    layer = GPT2TransformerBlock(
        num_heads=12,
        head_dim=64,
        model_dim=768,
        mlp_dim=config.mlp_dim,
        epsilon=config.layer_norm_eps,
    )

    init_input: Float[Array, "... S F"] = jr.uniform(rng, (*batch_dims, INIT_LEN, 768))
    variables = layer.init(rng, init_input)

    apply_input: Float[Array, "... S F"] = jr.uniform(
        rng, (*batch_dims, APPLY_LEN, 768)
    )
    output: Float[Array, "... S F"] = layer.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, 768)


@pytest.mark.parametrize("batch_dims", [(), (1,), (2,), (1, 2)], ids=format_params)
def test_transformer_contract(
    rng: Array, config: GPT2Config, batch_dims: Sequence[int]
):
    INIT_LEN = 4
    APPLY_LEN = 7

    model = GPT2Transformer.from_config(config)
    init_input: Int[Array, "... S"] = jnp.ones((*batch_dims, INIT_LEN), dtype=jnp.int32)
    variables = model.init(rng, init_input)

    apply_input: Int[Array, "... S"] = jr.randint(
        rng, (*batch_dims, APPLY_LEN), 0, 50257
    )
    output: Float[Array, "... S V"] = model.apply(variables, apply_input)
    assert output.shape == (*batch_dims, APPLY_LEN, 50257)


def make_hook_fn(stub):
    def hook_fn(x: Array, **kwargs):
        stub(x)
        return x

    return hook_fn


@pytest.fixture
def rng():
    return jr.PRNGKey(0)


@pytest.fixture
def config():
    return GPT2Config()


@pytest.fixture
def transformer(config: GPT2Config) -> GPT2Transformer:
    return GPT2Transformer.from_config(config)


@pytest.fixture
def mlp(config: GPT2Config) -> GPT2MLP:
    return GPT2MLP(features=[config.mlp_dim, config.model_dim])


@pytest.fixture
def transformer_params(rng, transformer: GPT2Transformer) -> PyTree[Array]:
    return transformer.init(rng, jnp.ones((256,), jnp.int32))["params"]


@pytest.fixture
def mlp_params(rng, mlp: GPT2MLP) -> PyTree[Array]:
    return mlp.init(rng, jnp.ones((256,)))["params"]


@pytest.mark.parametrize(
    "hook_name", ["embed_hook", "pos_embed_hook", "residual_hook", "output_hook"]
)
def test_transformer_hooks_called(
    mocker: MockerFixture,
    transformer: GPT2Transformer,
    transformer_params: PyTree[Array],
    hook_name: str,
):
    stub = mocker.stub()
    variables = {"params": transformer_params}
    inputs = jnp.ones((256,), jnp.int32)
    hooks = {hook_name: make_hook_fn(stub)}

    transformer.apply(variables, inputs, hooks)

    if hook_name != "residual_hook":
        stub.assert_called_once()
    else:
        stub.assert_called()


@pytest.mark.parametrize("hook_name", ["pre_activation_hook", "post_activation_hook"])
def test_mlp_hooks_called(
    rng: Array,
    mocker: MockerFixture,
    mlp: GPT2MLP,
    mlp_params: PyTree[Array],
    hook_name: str,
):
    stub = mocker.stub()
    variables = {"params": mlp_params}
    inputs = jr.uniform(rng, (256, 256))
    hooks = {hook_name: make_hook_fn(stub)}
    mlp.apply(variables, inputs, hooks)

    stub.assert_called_once()


SEQ_LENGTH = 256


def format_ids(val):
    if isinstance(val, str):
        return val
    elif isinstance(val, tuple):
        return val[0]
    else:
        return None


@pytest.mark.parametrize(
    "hook_name,expected",
    [
        ("embed_hook", (SEQ_LENGTH, 768)),
        ("pos_embed_hook", (SEQ_LENGTH, 768)),
        ("residual_hook", (SEQ_LENGTH, 768)),
        ("output_hook", (SEQ_LENGTH, 768)),
    ],
    ids=format_ids,
)
def test_transformer_hooks_can_store_values(
    transformer: GPT2Transformer,
    transformer_params: PyTree[Array],
    hook_name: str,
    expected: Sequence[int],
):
    hooks = {hook_name: store_hook}

    variables = {"params": transformer_params}
    inputs = jnp.ones((SEQ_LENGTH,), jnp.int32)
    _, state = transformer.apply(variables, inputs, hooks, mutable=["intermediates"])

    assert state["intermediates"][hook_name][0].shape == expected


@pytest.mark.parametrize(
    "hook_name,expected",
    [
        ("pre_activation_hook", (SEQ_LENGTH, 3072)),
        ("post_activation_hook", (SEQ_LENGTH, 3072)),
    ],
    ids=format_ids,
)
def test_mlp_hooks_can_store_values(
    rng: Array,
    mlp: GPT2MLP,
    mlp_params: PyTree[Array],
    hook_name: str,
    expected: Sequence[int],
):
    hooks = {hook_name: store_hook}

    variables = {"params": mlp_params}
    inputs = jr.uniform(rng, (256, 256))
    _, state = mlp.apply(variables, inputs, hooks, mutable=["intermediates"])

    assert state["intermediates"][hook_name][0].shape == expected


def format_nested_ids(val):
    if isinstance(val, tuple):
        if isinstance(val[0], str):
            return "/".join(val)

        if isinstance(val[0], int):
            return ",".join([str(v) for v in val])

    return val


@pytest.mark.parametrize(
    "hook_path,expected",
    [
        (("block_0", "mlp", "pre_activation_hook"), (SEQ_LENGTH, 3072)),
        (("block_4", "mlp", "post_activation_hook"), (SEQ_LENGTH, 3072)),
        (("block_1", "attn", "query_hook"), (SEQ_LENGTH, 12, 64)),
        (("block_2", "attn", "key_hook"), (SEQ_LENGTH, 12, 64)),
        (("block_3", "attn", "value_hook"), (SEQ_LENGTH, 12, 64)),
        (("block_4", "attn", "scores_hook"), (12, SEQ_LENGTH, SEQ_LENGTH)),
        (("block_5", "attn", "weights_hook"), (12, SEQ_LENGTH, SEQ_LENGTH)),
        (("block_6", "attn", "z_hook"), (SEQ_LENGTH, 12, 64)),
        (("block_7", "attn", "output_hook"), (SEQ_LENGTH, 768)),
        (("block_0", "ln_1", "std_hook"), (SEQ_LENGTH, 768)),
        (("block_1", "ln_2", "normalized_hook"), (SEQ_LENGTH, 768)),
        (("ln_f", "std_hook"), (SEQ_LENGTH, 768)),
        (("ln_f", "normalized_hook"), (SEQ_LENGTH, 768)),
    ],
    ids=format_nested_ids,
)
def test_nested_transformer_hooks_are_called(
    transformer: GPT2Transformer,
    transformer_params: PyTree[Array],
    hook_path: Tuple[str, ...],
    expected: Sequence[int],
):
    hooks = temp_hooks = {}
    for key in hook_path[:-1]:
        temp_hooks[key] = {}
        temp_hooks = temp_hooks[key]

    temp_hooks[hook_path[-1]] = store_hook

    variables = {"params": transformer_params}
    inputs = jnp.ones((SEQ_LENGTH,), jnp.int32)
    _, state = transformer.apply(variables, inputs, hooks, mutable=["intermediates"])

    state_ = state["intermediates"]
    for key in hook_path:
        state_ = state_[key]

    assert state_[0].shape == expected
