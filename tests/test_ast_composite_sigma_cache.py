# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch
import pytest

import nestynet
from nestynet.default_setup import initialize_environment
from nestynet.optimizer import EvidenceConfig, LMConfig, Predictive_LM_Optimizer, ResidualsModule
from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor
from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode
from nestynet_sr.sr_core.fit_links import fit_link_torch_d1

torch.set_default_dtype(torch.float64)


class LinearLeaf(torch.nn.Module):
    """Minimal scalar-output LAProvider leaf for ASTCompositeAdaptor tests."""

    def __init__(self, weight: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[float(weight)]], dtype=torch.float64))

    def num_parameters(self):
        return int(self.weight.numel())

    def forward(self, x):
        x = x if x.ndim == 2 else x.unsqueeze(-1)
        return x @ self.weight

    def build_cache(self, data, **kw):
        del kw
        if len(data) == 2:
            x, y = data
            y_sigma = None
        else:
            x, y, y_sigma = data[0], data[1], data[2]
        f = self.forward(x)
        return {
            "x": x,
            "y": y,
            "y_sigma": y_sigma,
            "f": f,
            "jac": x.unsqueeze(-1),
            "SegmentedModel": False,
            "S": 1,
            "Pseg": 1,
            "O": 1,
        }

    def jvp(self, cache, v, out_dim=None):
        coeff = torch.as_tensor(v, device=cache["x"].device, dtype=cache["x"].dtype).reshape(-1)[0]
        out = cache["x"][:, :1] * coeff
        return out[:, 0] if out_dim is not None else out

    def vjp(self, cache, v, out_dim=None):
        del out_dim
        vec = torch.as_tensor(v, device=cache["x"].device, dtype=cache["x"].dtype)
        if vec.ndim == 1:
            vec = vec.unsqueeze(-1)
        return (cache["x"][:, :1] * vec).sum(0, keepdim=True)

    def blocks(self, *args, **kwargs):
        del args, kwargs
        yield {
            "analytic_map": slice(0, 1),
            "global_map": slice(0, 1),
        }


class LossOnlyAwareLeaf(LinearLeaf):
    """Leaf that omits derivative-only tensors when requested."""

    def __init__(self, weight: float):
        super().__init__(weight)
        self.jacobian_calls = 0

    def build_cache(self, data, **kw):
        need_derivatives = bool(kw.get("need_derivatives", True))
        if len(data) == 2:
            x, y = data
            y_sigma = None
        else:
            x, y, y_sigma = data[0], data[1], data[2]
        f = self.forward(x)
        cache = {
            "x": x,
            "y": y,
            "y_sigma": y_sigma,
            "f": f,
            "SegmentedModel": False,
            "S": 1,
            "Pseg": 1,
            "O": 1,
        }
        if need_derivatives:
            cache["sigma"] = torch.ones_like(f)
        return cache

    def jacobian(self, cache):
        self.jacobian_calls += 1
        if "sigma" not in cache:
            raise KeyError("sigma")
        return cache["x"].unsqueeze(-1)


class _DummySegmentedBase(torch.nn.Module):
    def __init__(self, num_segments: int = 4):
        super().__init__()
        self.num_segments = int(num_segments)
        self.Nout_size = 1


class SegmentAwareLeaf(torch.nn.Module):
    """Minimal segmented-like leaf whose cache output depends on `segments`."""

    def __init__(self, *, bias: float = 0.25, segments=(1, 3)):
        super().__init__()
        self.base_model = _DummySegmentedBase()
        self.segments = torch.as_tensor(segments, dtype=torch.long)
        self.bias = torch.nn.Parameter(torch.tensor([[float(bias)]], dtype=torch.float64))
        self.last_segments = None

    def num_parameters(self):
        return int(self.bias.numel())

    def forward(self, x):
        x = x if x.ndim == 2 else x.unsqueeze(-1)
        return x[:, :1] + self.bias

    def build_cache(self, data, **kw):
        x, y = data[0], data[1]
        segs = kw.get("segments", self.segments)
        if segs is None:
            segs = torch.arange(self.base_model.num_segments, device=x.device, dtype=torch.long)
        else:
            segs = torch.as_tensor(segs, device=x.device, dtype=torch.long)
        self.last_segments = segs.detach().clone()
        seg_shift = segs.to(device=x.device, dtype=x.dtype).sum()
        f = x[:, :1] + self.bias + seg_shift
        return {
            "x": x,
            "y": y,
            "f": f,
            "jac": torch.ones(x.shape[0], 1, 1, dtype=x.dtype, device=x.device),
            "SegmentedModel": True,
            "S": int(segs.numel()),
            "Pseg": 1,
            "O": 1,
        }

    def jvp(self, cache, v, out_dim=None):
        coeff = torch.as_tensor(v, device=cache["x"].device, dtype=cache["x"].dtype).reshape(-1)[0]
        out = torch.ones_like(cache["f"]) * coeff
        return out[:, 0] if out_dim is not None else out

    def vjp(self, cache, v, out_dim=None):
        del out_dim
        vec = torch.as_tensor(v, device=cache["x"].device, dtype=cache["x"].dtype)
        if vec.ndim == 1:
            vec = vec.unsqueeze(-1)
        return vec.sum(0, keepdim=True)

    def blocks(self, *args, **kwargs):
        del args, kwargs
        yield {
            "analytic_map": slice(0, 1),
            "global_map": slice(0, 1),
        }


class ZeroParamIdentityLeaf(torch.nn.Module):
    """Zero-parameter structural leaf, matching a raw variable factor."""

    def num_parameters(self):
        return 0

    def forward(self, x):
        x = x if x.ndim == 2 else x.unsqueeze(-1)
        return x[:, :1]

    def build_cache(self, data, **kw):
        del kw
        x, y = data[0], data[1]
        f = self.forward(x)
        return {
            "x": x,
            "y": y,
            "f": f,
            "jac": x.new_zeros(x.shape[0], 1, 0),
            "SegmentedModel": False,
            "S": 1,
            "Pseg": 0,
            "O": 1,
        }

    def jvp(self, cache, v, out_dim=None):
        del v, out_dim
        return cache["f"].new_zeros(cache["f"].shape)

    def vjp(self, cache, v, out_dim=None):
        del v, out_dim
        return cache["f"].new_zeros(0)

    def blocks(self, *args, **kwargs):
        del args, kwargs
        if False:
            yield {}


def _make_single_leaf_model():
    return ASTCompositeAdaptor(AtomNode("nn", (0,)), [LinearLeaf(1.5)])


def _make_two_leaf_model():
    ast = AddNode(AtomNode("nn", (0,)), AtomNode("nn", (1,)))
    return ASTCompositeAdaptor(ast, [LinearLeaf(1.0), LinearLeaf(-0.5)])


def _make_segmented_leaf_model():
    leaf = SegmentAwareLeaf()
    return ASTCompositeAdaptor(AtomNode("nn", (0,)), [leaf]), leaf


def test_ast_composite_build_cache_emits_fitspace_sigma_and_weights_for_asinh():
    model = _make_single_leaf_model()
    model.fit_y_link = "asinh"
    model.fit_y_link_scale = 2.5

    x = torch.tensor([[1.0], [3.0], [-2.0]], dtype=torch.float64)
    y = torch.tensor([[0.0], [2.0], [-6.0]], dtype=torch.float64)
    y_sigma = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

    cache = model.build_cache((x, y, y_sigma))

    eps = torch.finfo(torch.float64).eps
    expected_raw_sigma = y_sigma.unsqueeze(-1).clamp_min(eps)
    expected_target_sigma = (
        fit_link_torch_d1(y, "asinh", 2.5).abs() * expected_raw_sigma
    ).clamp_min(eps)
    expected_weights = 1.0 / (expected_target_sigma * expected_target_sigma)

    assert "y_sigma" in cache
    assert "target_sigma" in cache
    assert "sample_weights" in cache
    torch.testing.assert_close(cache["y_sigma"], expected_raw_sigma)
    torch.testing.assert_close(cache["target_sigma"], expected_target_sigma)
    torch.testing.assert_close(cache["sample_weights"], expected_weights)


@pytest.mark.parametrize(
    ("fit_link", "fit_scale"),
    [
        (None, 1.0),
        ("asinh", 2.5),
    ],
)
def test_ast_composite_dense_and_diag_respect_sample_weights(fit_link, fit_scale):
    model = _make_two_leaf_model()
    model.fit_y_link = fit_link
    model.fit_y_link_scale = fit_scale

    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, -1.0],
            [-2.0, 0.5],
        ],
        dtype=torch.float64,
    )
    y = torch.tensor([[0.5], [1.0], [-1.5]], dtype=torch.float64)
    y_sigma = torch.tensor([0.5, 1.5, 2.0], dtype=torch.float64)

    cache = model.build_cache((x, y, y_sigma))
    J = model.jacobian(cache).reshape(-1, model.num_parameters())
    w = cache["sample_weights"].reshape(-1)

    expected_dense = J.t().matmul(w.unsqueeze(-1) * J)
    expected_diag = (w.unsqueeze(-1) * J.square()).sum(0)

    torch.testing.assert_close(cache["sample_weights"], 1.0 / cache["target_sigma"].square())
    torch.testing.assert_close(model.dense(cache), expected_dense)
    torch.testing.assert_close(model.diag(cache), expected_diag)


def test_ast_composite_single_segmented_leaf_exposes_base_model_for_evidence():
    model, leaf = _make_segmented_leaf_model()

    assert model.base_model is leaf.base_model
    assert model.base_models() == [leaf.base_model]
    torch.testing.assert_close(model.segments, leaf.segments)


def test_ast_composite_zero_param_prefactor_preserves_evidence_base_model():
    leaf = SegmentAwareLeaf()
    ast = MulNode(AtomNode("var", (0,)), AtomNode("nn", (1,)))
    model = ASTCompositeAdaptor(ast, [ZeroParamIdentityLeaf(), leaf])

    assert model.num_parameters() == leaf.num_parameters()
    assert model.base_model is leaf.base_model
    assert model.base_models() == [leaf.base_model]
    torch.testing.assert_close(model.segments, leaf.segments)


def test_ast_composite_single_segmented_leaf_uses_segment_aware_cache_for_top_level_f():
    model, leaf = _make_segmented_leaf_model()

    x = torch.tensor([[1.0], [3.0], [-2.0]], dtype=torch.float64)
    y = torch.zeros_like(x)

    cache = model.build_cache((x, y))

    torch.testing.assert_close(leaf.last_segments, model.segments)
    expected_f = x[:, :1] + leaf.bias.detach() + model.segments.to(dtype=x.dtype).sum()
    torch.testing.assert_close(cache["f"], expected_f)
    assert not torch.allclose(cache["f"], leaf.forward(x).detach())


def test_ast_composite_loss_only_cache_omits_leaf_jacobian_but_keeps_residuals():
    leaf = LossOnlyAwareLeaf(1.5)
    model = ASTCompositeAdaptor(AtomNode("nn", (0,)), [leaf])

    x = torch.tensor([[1.0], [3.0], [-2.0]], dtype=torch.float64)
    y = torch.tensor([[0.0], [2.0], [-6.0]], dtype=torch.float64)
    y_sigma = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

    full = model.build_cache((x, y, y_sigma), need_derivatives=True)
    lite = model.build_cache((x, y, y_sigma), need_derivatives=False)

    assert leaf.jacobian_calls == 1
    assert "jac" in full["leaves"][0]
    assert "jac" not in lite["leaves"][0]
    torch.testing.assert_close(full["f"], lite["f"])
    torch.testing.assert_close(
        model.residuals(full, (x, y)),
        model.residuals(lite, (x, y)),
    )


@pytest.mark.parametrize("track_grad", [False, True])
def test_ast_composite_residuals_reuse_segment_aware_leaf_cache(track_grad):
    model, _leaf = _make_segmented_leaf_model()

    x = torch.tensor([[2.0], [-1.0]], dtype=torch.float64)
    y = torch.tensor([[10.0], [4.0]], dtype=torch.float64)
    cache = model.build_cache((x, y))

    r = model.residuals(cache, (x, y), track_grad=track_grad)
    expected = y - cache["leaves"][0]["f"]

    if not track_grad:
        expected = expected.detach()

    torch.testing.assert_close(r, expected)
    if track_grad:
        assert r.requires_grad


def test_ast_composite_jacobian_accepts_out_dim():
    model = _make_two_leaf_model()

    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, -1.0],
        ],
        dtype=torch.float64,
    )
    y = torch.tensor([[0.5], [1.0]], dtype=torch.float64)

    cache = model.build_cache((x, y))
    full = model.jacobian(cache)
    single = model.jacobian(cache, out_dim=0)

    torch.testing.assert_close(single, full[:, 0:1, :])


def test_ast_composite_linear_refinement_passes_segment_prior_to_leaf_solve():
    initialize_environment(seed=123)
    device = torch.device("cpu")
    dtype = torch.float64
    base = nestynet.nets.NestyNet_Model(
        "G_Model", 1, 1, 1, 0.1, dtype, device, seg_width=1
    ).to(device).to(dtype)
    with torch.no_grad():
        base.base_model.a_fit.fill_(2.0)

    leaf = nestynet.adaptors.SegmentedAdaptor(
        base,
        segments=torch.arange(1, device=device),
    )
    model = ASTCompositeAdaptor(AtomNode("nn", (0,)), [leaf])

    x = torch.linspace(-1.0, 1.0, 8, device=device, dtype=dtype).unsqueeze(-1)
    y = torch.zeros(8, 1, device=device, dtype=dtype)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=8,
        shuffle=False,
        drop_last=False,
    )

    def factory(_opt):
        return ResidualsModule([model], loader, device=device)

    cfg = LMConfig(
        LM_strategy="direct_solve",
        spla_enable=False,
        max_iter=1,
        use_backtracking=False,
        use_quadratic_extrapolation=False,
        linear_refinement=True,
        auto_enable_linear_refinement=True,
        use_geodesic_acceleration=False,
    )
    evidence_cfg = EvidenceConfig(
        enabled=True,
        lambda_patch=0.0,
        lambda_mean=0.0,
        lambda_slope=0.0,
        lambda_quad=0.0,
        segment_alpha_init=100.0,
        prior_rel_scale=0.0,
        prior_abs_scale=1.0,
        prior_anchor_mode="live",
        update_alpha_every_accepted=0,
        dense_logdet_max=64,
        allow_linear_refinement=True,
    )
    opt = Predictive_LM_Optimizer(
        list(model.parameters()),
        [factory],
        cfg=cfg,
        evidence_cfg=evidence_cfg,
    )

    with torch.no_grad():
        base.base_model.a_fit.zero_()
    idx, prior_vals = model.linear_refinement(
        [opt.base_residual_modules[0]],
        device,
        lam_LM=0.0,
        ridge_val=0.0,
    )
    prior_terms = int(opt.state.get("linear_refinement_prior_terms", 0))

    opt.evidence_controller.state.segment_alphas[0] = 0.0
    idx_data, data_vals = model.linear_refinement(
        [opt.base_residual_modules[0]],
        device,
        lam_LM=0.0,
        ridge_val=0.0,
    )

    assert prior_terms == 1
    assert int(opt.state.get("linear_refinement_prior_terms", 0)) == 0
    torch.testing.assert_close(idx, idx_data)
    assert float(prior_vals.item()) > 1.0
    torch.testing.assert_close(data_vals, torch.zeros_like(data_vals), rtol=0.0, atol=1.0e-12)


def test_ast_composite_dual_linear_refinement_passes_stage1_segment_prior():
    initialize_environment(seed=123)
    device = torch.device("cpu")
    dtype = torch.float64
    stage0_model = nestynet.nets.NestyNet_Model(
        "G_Model", 1, 1, 1, 0.1, dtype, device, seg_width=1
    ).to(device).to(dtype)
    stage1_model = nestynet.nets.NestyNet_Model(
        "G_Model", 1, 1, 1, 0.1, dtype, device, seg_width=1
    ).to(device).to(dtype)
    with torch.no_grad():
        stage1_model.base_model.a_fit.fill_(2.0)

    stage0 = nestynet.adaptors.SegmentedAdaptor(
        stage0_model,
        segments=torch.arange(1, device=device),
    )
    stage1 = nestynet.adaptors.SegmentedAdaptor(
        stage1_model,
        segments=torch.arange(1, device=device),
    )
    dual = nestynet.adaptors.DualSegmentedAdaptor(stage0, stage1)
    model = ASTCompositeAdaptor(AtomNode("nn", (0,)), [dual])

    x = torch.linspace(-1.0, 1.0, 8, device=device, dtype=dtype).unsqueeze(-1)
    y = torch.zeros(8, 1, device=device, dtype=dtype)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=8,
        shuffle=False,
        drop_last=False,
    )

    def factory(_opt):
        return ResidualsModule([model], loader, device=device)

    cfg = LMConfig(
        LM_strategy="direct_solve",
        spla_enable=False,
        max_iter=1,
        use_backtracking=False,
        use_quadratic_extrapolation=False,
        linear_refinement=True,
        auto_enable_linear_refinement=True,
        use_geodesic_acceleration=False,
    )
    evidence_cfg = EvidenceConfig(
        enabled=True,
        lambda_patch=0.0,
        lambda_mean=0.0,
        lambda_slope=0.0,
        lambda_quad=0.0,
        segment_alpha_init=100.0,
        prior_rel_scale=0.0,
        prior_abs_scale=1.0,
        prior_anchor_mode="live",
        update_alpha_every_accepted=0,
        dense_logdet_max=64,
        allow_linear_refinement=True,
    )
    opt = Predictive_LM_Optimizer(
        list(model.parameters()),
        [factory],
        cfg=cfg,
        evidence_cfg=evidence_cfg,
    )

    with torch.no_grad():
        stage1_model.base_model.a_fit.zero_()
    idx, prior_vals = model.linear_refinement(
        [opt.base_residual_modules[0]],
        device,
        lam_LM=0.0,
        ridge_val=0.0,
    )
    prior_terms = int(opt.state.get("linear_refinement_prior_terms", 0))

    for seg_idx in list(opt.evidence_controller.state.segment_alphas):
        opt.evidence_controller.state.segment_alphas[int(seg_idx)] = 0.0
    idx_data, data_vals = model.linear_refinement(
        [opt.base_residual_modules[0]],
        device,
        lam_LM=0.0,
        ridge_val=0.0,
    )

    assert prior_terms == 1
    assert int(opt.state.get("linear_refinement_prior_terms", 0)) == 0
    torch.testing.assert_close(idx, idx_data)
    assert float(prior_vals.item()) > 1.0
    torch.testing.assert_close(data_vals, torch.zeros_like(data_vals), rtol=0.0, atol=1.0e-12)
