import torch

import nestynet
from nestynet.default_setup import dtype, initialize_environment
from nestynet.optimizer import ResidualsModule

from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor
from nestynet_sr.sr_core.bridges import AtomNode


def _build_dual(*, nxvars=2, num_segments=2):
    device = torch.device("cpu")
    nmid = nxvars + 2
    model1 = nestynet.nets.NestyNet_Model(
        "G_Model",
        nmid,
        nxvars,
        num_segments,
        0.1,
        dtype,
        device,
        seg_width=1,
    ).to(device).to(dtype)
    model2 = nestynet.nets.NestyNet_Model(
        "G_Model",
        1,
        nmid,
        num_segments,
        0.1,
        dtype,
        device,
    ).to(device).to(dtype)
    segments = torch.arange(num_segments, device=device)
    stage0 = nestynet.adaptors.SegmentedAdaptor(model1, segments=segments)
    stage1 = nestynet.adaptors.SegmentedAdaptor(model2, segments=segments)
    return nestynet.adaptors.DualSegmentedAdaptor(stage0, stage1)


def _assert_block_equal(actual, expected):
    assert actual["owner"] != expected["owner"]
    for key, expected_value in expected.items():
        if key == "owner":
            continue
        actual_value = actual[key]
        if torch.is_tensor(expected_value):
            assert torch.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_single_native_identity_atom_preserves_the_leaf_optimizer_contract():
    initialize_environment(seed=123)
    dual = _build_dual()
    model = ASTCompositeAdaptor(AtomNode("nn", (0, 1)), [dual])
    model.declare_global_input_dim(2)

    assert model._transparent_identity_leaf() is dual

    model_parameters = list(model.named_parameters())
    dual_parameters = list(dual.named_parameters())
    assert [name for name, _ in model_parameters] == [name for name, _ in dual_parameters]
    assert all(
        model_parameter is dual_parameter
        for (_, model_parameter), (_, dual_parameter) in zip(
            model_parameters,
            dual_parameters,
        )
    )

    dual_block = next(dual.blocks())
    model_block = next(model.blocks())
    assert model_block["owner"] == id(model)
    _assert_block_equal(model_block, dual_block)

    x = torch.linspace(-1.0, 1.0, 12, dtype=dtype).reshape(6, 2)
    y = (x[:, :1] - 0.25 * x[:, 1:2]).contiguous()
    model.pre_block(block=model_block, theta=None)
    dual_cache = dual.build_cache((x, y))
    model_cache = model.build_cache((x, y))

    assert model_cache.keys() == dual_cache.keys()
    assert "leaves" not in model_cache
    torch.testing.assert_close(model(x), dual(x), rtol=0.0, atol=0.0)
    torch.testing.assert_close(model_cache["f"], dual_cache["f"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        model.residuals(model_cache),
        dual.residuals(dual_cache),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.jacobian(model_cache),
        dual.jacobian(dual_cache),
        rtol=0.0,
        atol=0.0,
    )

    direction = torch.linspace(-0.5, 0.5, dual.num_parameters(), dtype=dtype)
    upstream = torch.linspace(-1.0, 1.0, x.shape[0], dtype=dtype)
    for actual, expected in (
        (model.jvp(model_cache, direction), dual.jvp(dual_cache, direction)),
        (model.vjp(model_cache, upstream), dual.vjp(dual_cache, upstream)),
        (model.diag(model_cache), dual.diag(dual_cache)),
        (model.dense(model_cache), dual.dense(dual_cache)),
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=len(x),
        shuffle=False,
    )
    residual_module = ResidualsModule([model], loader, device=torch.device("cpu"))
    model_idx, model_values = model.linear_refinement(
        [residual_module],
        torch.device("cpu"),
        lam_LM=1.0e-3,
    )
    dual_idx, dual_values = dual.linear_refinement(
        [residual_module],
        torch.device("cpu"),
        lam_LM=1.0e-3,
    )
    torch.testing.assert_close(model_idx, dual_idx, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model_values, dual_values, rtol=0.0, atol=0.0)


def test_fit_link_and_input_mapping_keep_the_composite_path():
    initialize_environment(seed=123)
    dual = _build_dual()
    linked = ASTCompositeAdaptor(AtomNode("nn", (0, 1)), [dual])
    linked.declare_global_input_dim(2)
    linked.fit_y_link = "asinh"
    assert linked._transparent_identity_leaf() is None
    assert all(name.startswith("leaf0.") for name, _ in linked.named_parameters())

    sliced = ASTCompositeAdaptor(AtomNode("nn", (1, 2)), [_build_dual()])
    sliced.declare_global_input_dim(3)
    assert sliced._transparent_identity_leaf() is None


def test_prefix_slice_cannot_activate_identity_transparency():
    initialize_environment(seed=123)
    x = torch.randn(5, 3, dtype=dtype)

    two_input_leaf = _build_dual(nxvars=2)
    prefix = ASTCompositeAdaptor(AtomNode("nn", (0, 1)), [two_input_leaf])
    prefix.declare_global_input_dim(3)
    assert prefix._transparent_identity_leaf() is None
    torch.testing.assert_close(prefix(x), two_input_leaf(x[:, :2]))

    three_input_leaf = _build_dual(nxvars=3)
    short_atom = ASTCompositeAdaptor(AtomNode("nn", (0, 1)), [three_input_leaf])
    short_atom.declare_global_input_dim(3)
    assert short_atom._transparent_identity_leaf() is None
