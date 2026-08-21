# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nestynet_sr.sr_core.atoms import RRationalPolyLeaf
from nestynet_sr.sr_core.bridges import AtomNode, PowNode
from nestynet_sr.sr_core.bridges import MulNode, build_composite_from_ast
from nestynet_sr.sr_core.coefficient_units import solve_rational_coefficient_gauge
from nestynet_sr.sr_search import candidate_builders as cb
from nestynet_sr.sr_search.stageB import fitting as stageb_fitting
from nestynet_sr.sr_search.stageB import engine as stageb_engine
from nestynet_sr.sr_search.candidate_builders import (
    _build_inv_poly_candidate,
    _build_inv_poly_candidates,
    _build_ratpoly_1d_candidate,
    _build_ratpoly_1d_candidates,
    _build_ratpoly_candidates,
    _build_sqrt_ratpoly_1d_candidates,
)
from nestynet_sr.sr_search.stageB.engine import (
    Candidate,
    StageBContext,
    StageBState,
    _build_rratpoly_degree_trim_candidate,
    _ratpoly_degree_bands,
    _ratpoly_den_pivot_degree,
    candidate_pattern_name,
)
from nestynet_sr.sr_search.representation import _ratpoly_leaf_repr


def _unwrap_core(leaf_mod):
    return getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))


def _make_ctx(*, disabled_patterns=None):
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    state = StageBState(
        root=root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.0,
    )
    return StageBContext(
        state=state,
        train_loader=[],
        val_loader=[],
        lm_hp=SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=0.0,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        disabled_patterns=set(disabled_patterns or ()),
        verbose=False,
    )


def test_rratpoly_authoritative_repr_preserves_dynamic_range_terms():
    leaf = RRationalPolyLeaf(
        indices=(0,),
        deg_num=7,
        deg_den=5,
        dtype=torch.float64,
    )

    with torch.no_grad():
        leaf.coeffs_num.zero_()
        for free_idx, full_idx in enumerate(leaf.free_pos_num.tolist()):
            power = int(leaf.exps_num_full[full_idx, 0])
            if power == 0:
                leaf.coeffs_num[free_idx] = 1.0e6

        leaf.coeffs_den.zero_()
        for idx, exponent in enumerate(leaf.exps_den[:, 0].tolist()):
            power = int(exponent)
            if power == 0:
                leaf.coeffs_den[idx] = 2.0
            elif power == 4:
                leaf.coeffs_den[idx] = 5.65e-4
            elif power == 5:
                leaf.coeffs_den[idx] = 1.506e-6

    _scale, expr = _ratpoly_leaf_repr(leaf, (0,), sig=6)
    compact = expr.replace(" ", "")
    assert "x0^7" in compact
    assert "x0^4" in compact
    assert "x0^5" in compact

    x = torch.linspace(1.0, 5.0, 41, dtype=torch.float64)
    expected = leaf(x[:, None]).squeeze(-1)
    actual = eval(
        expr.replace("^", "**"),
        {"__builtins__": {}},
        {"x0": x},
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-15, atol=2.0e-15)


def test_build_ratpoly_candidates_invalid_target_returns_empty_list():
    target = AtomNode(kind="poly", var_idxs=(0,), kwargs={}, tag="leaf0")
    res = _build_ratpoly_candidates(
        root=target,
        target=target,
        reuse={},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert res == []


def test_build_inv_poly_candidates_invalid_target_returns_empty_list():
    target = AtomNode(kind="poly", var_idxs=(0,), kwargs={}, tag="leaf0")
    res = _build_inv_poly_candidates(
        root=target,
        target=target,
        reuse={},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert res == []


def test_build_inv_poly_candidate_wrapper_returns_none_tuple_on_empty():
    target = AtomNode(kind="poly", var_idxs=(0,), kwargs={}, tag="leaf0")
    res = _build_inv_poly_candidate(
        root=target,
        target=target,
        reuse={},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert res == (None, None)


def test_build_ratpoly_1d_candidates_invalid_target_returns_empty_list():
    target = AtomNode(kind="poly", var_idxs=(0,), kwargs={}, tag="leaf0")
    res = _build_ratpoly_1d_candidates(
        root=target,
        target=target,
        reuse={},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert res == []


def test_build_ratpoly_1d_candidate_wrapper_returns_none_tuple_on_empty():
    target = AtomNode(kind="poly", var_idxs=(0,), kwargs={}, tag="leaf0")
    res = _build_ratpoly_1d_candidate(
        root=target,
        target=target,
        reuse={},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert res == (None, None, None)


def test_build_ratpoly_candidates_emit_sparse_scale_times_rratpoly_when_nd_pivot_clear(monkeypatch):
    def _fake_teacher_data(*args, **kwargs):
        gen = torch.Generator().manual_seed(7)
        x = torch.rand(700, 2, generator=gen, dtype=torch.float64)
        f = 1.0 + 0.5 * x[:, 0] - 0.25 * x[:, 1]
        return x, f

    def _fake_probe(X, F, deg_num, deg_den, **kwargs):
        if (deg_num, deg_den) == (3, 3):
            return 1.0e-6
        return float("inf")

    def _fake_fit_nd(X, F, exps_num, exps_den, **kwargs):
        assert kwargs.get("return_support_indices", False) is True
        a = torch.tensor([0.5, 2.0], dtype=torch.float64)
        b = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        support_num = torch.tensor([0, 5], dtype=torch.int64)
        support_den = torch.tensor([0, 4, 8], dtype=torch.int64)
        return a, b, support_num, support_den

    monkeypatch.setattr(cb, "_gather_atom_teacher_data", _fake_teacher_data)
    monkeypatch.setattr(cb, "_rational_probe_nd", _fake_probe)
    monkeypatch.setattr(cb, "_fit_rational_coeffs_nd", _fake_fit_nd)

    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nn0")
    res = _build_ratpoly_candidates(
        root=target,
        target=target,
        reuse={"nn0": nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_deg_num=3,
        max_deg_den=3,
        min_points=200,
        rel_rms_threshold=1e-3,
    )

    assert len(res) == 1
    cand_root, init_fn, meta = res[0]
    assert isinstance(cand_root, MulNode)
    assert str(cand_root.left.kind).lower() == "scale"
    assert str(cand_root.right.kind).lower() == "rratpoly"
    assert cand_root.right.kwargs["deg_num"] == 2
    assert cand_root.right.kwargs["deg_den"] == 3
    assert cand_root.right.kwargs["exps_num_override"] == [[0, 0], [2, 0]]
    assert cand_root.right.kwargs["exps_den_override"] == [[0, 0], [1, 1], [2, 1]]
    assert meta["leaf_kind"] == "rratpoly"
    assert meta["reduced"] is True
    assert meta["ratpoly_scale_tag"] == cand_root.left.tag
    assert meta["ratpoly_target_tag"] == "nn0"
    assert meta["ratpoly_var_idxs"] == (0, 1)
    assert isinstance(meta["ratpoly_target_sig"], int)
    assert meta["signature"][0] == meta["ratpoly_target_sig"]

    model, atom_to_leaf = build_composite_from_ast(
        cand_root,
        dtype=torch.float64,
        device=torch.device("cpu"),
        reuse={},
        return_atom_map=True,
    )
    init_fn(cand_root, model)

    scale_core = _unwrap_core(atom_to_leaf[id(cand_root.left)])
    rat_core = _unwrap_core(atom_to_leaf[id(cand_root.right)])
    assert isinstance(rat_core, RRationalPolyLeaf)
    assert torch.allclose(
        scale_core.value.detach(),
        torch.tensor(2.0, dtype=scale_core.value.dtype, device=scale_core.value.device),
    )
    assert torch.allclose(
        rat_core.coeffs_num.detach(),
        torch.tensor([0.25], dtype=rat_core.coeffs_num.dtype, device=rat_core.coeffs_num.device),
    )
    assert torch.allclose(
        rat_core.coeffs_den.detach(),
        torch.tensor([1.0, -2.0, 1.0], dtype=rat_core.coeffs_den.dtype, device=rat_core.coeffs_den.device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rratpoly_custom_init_accepts_cpu_fit_coefficients_on_cuda(monkeypatch):
    def _fake_teacher_data(*args, **kwargs):
        gen = torch.Generator().manual_seed(7)
        x = torch.rand(700, 2, generator=gen, dtype=torch.float64)
        f = 1.0 + 0.5 * x[:, 0] - 0.25 * x[:, 1]
        return x, f

    def _fake_probe(X, F, deg_num, deg_den, **kwargs):
        if (deg_num, deg_den) == (3, 3):
            return 1.0e-6
        return float("inf")

    def _fake_fit_nd(X, F, exps_num, exps_den, **kwargs):
        assert kwargs.get("return_support_indices", False) is True
        a = torch.tensor([0.5, 2.0], dtype=torch.float64)
        b = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        support_num = torch.tensor([0, 5], dtype=torch.int64)
        support_den = torch.tensor([0, 4, 8], dtype=torch.int64)
        return a, b, support_num, support_den

    monkeypatch.setattr(cb, "_gather_atom_teacher_data", _fake_teacher_data)
    monkeypatch.setattr(cb, "_rational_probe_nd", _fake_probe)
    monkeypatch.setattr(cb, "_fit_rational_coeffs_nd", _fake_fit_nd)

    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nn0")
    res = _build_ratpoly_candidates(
        root=target,
        target=target,
        reuse={"nn0": nn.Identity()},
        train_loader=[],
        device=torch.device("cuda"),
        dtype=torch.float64,
        max_deg_num=3,
        max_deg_den=3,
        min_points=200,
        rel_rms_threshold=1e-3,
    )

    assert len(res) == 1
    cand_root, init_fn, _meta = res[0]
    model, atom_to_leaf = build_composite_from_ast(
        cand_root,
        dtype=torch.float64,
        device=torch.device("cuda"),
        reuse={},
        return_atom_map=True,
    )
    init_fn(cand_root, model)

    scale_core = _unwrap_core(atom_to_leaf[id(cand_root.left)])
    rat_core = _unwrap_core(atom_to_leaf[id(cand_root.right)])
    assert rat_core.coeffs_num.device.type == "cuda"
    assert torch.allclose(
        scale_core.value.detach().cpu(),
        torch.tensor(2.0, dtype=torch.float64),
    )
    assert torch.allclose(
        rat_core.coeffs_num.detach().cpu(),
        torch.tensor([0.25], dtype=torch.float64),
    )


def test_build_ratpoly_candidates_fallback_to_sparse_ratpoly_when_nd_pivot_ambiguous(monkeypatch):
    def _fake_teacher_data(*args, **kwargs):
        gen = torch.Generator().manual_seed(7)
        x = torch.rand(700, 2, generator=gen, dtype=torch.float64)
        f = 1.0 + 0.5 * x[:, 0] - 0.25 * x[:, 1]
        return x, f

    def _fake_probe(X, F, deg_num, deg_den, **kwargs):
        if (deg_num, deg_den) == (3, 3):
            return 1.0e-6
        return float("inf")

    def _fake_fit_nd(X, F, exps_num, exps_den, **kwargs):
        assert kwargs.get("return_support_indices", False) is True
        a = torch.tensor([1.0, 0.8, -0.7], dtype=torch.float64)
        b = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        support_num = torch.tensor([0, 3, 5], dtype=torch.int64)
        support_den = torch.tensor([0, 4, 8], dtype=torch.int64)
        return a, b, support_num, support_den

    monkeypatch.setattr(cb, "_gather_atom_teacher_data", _fake_teacher_data)
    monkeypatch.setattr(cb, "_rational_probe_nd", _fake_probe)
    monkeypatch.setattr(cb, "_fit_rational_coeffs_nd", _fake_fit_nd)

    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nn0")
    res = _build_ratpoly_candidates(
        root=target,
        target=target,
        reuse={"nn0": nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_deg_num=3,
        max_deg_den=3,
        min_points=200,
        rel_rms_threshold=1e-3,
    )

    assert len(res) == 1
    cand_root, _init_fn, meta = res[0]
    assert isinstance(cand_root, AtomNode)
    assert cand_root.kind.lower() == "ratpoly"
    assert cand_root.tag == "nn0"
    assert cand_root.kwargs["deg_num"] == 2
    assert cand_root.kwargs["deg_den"] == 3
    assert cand_root.kwargs["exps_num_override"] == [[0, 0], [0, 2], [2, 0]]
    assert cand_root.kwargs["exps_den_override"] == [[0, 0], [1, 1], [2, 1]]
    assert meta["leaf_kind"] == "ratpoly"
    assert meta["ratpoly_target_tag"] == "nn0"
    assert meta["ratpoly_var_idxs"] == (0, 1)
    assert isinstance(meta["ratpoly_target_sig"], int)
    assert meta["signature"][0] == meta["ratpoly_target_sig"]


def test_build_ratpoly_1d_candidates_emit_sparse_scale_times_rratpoly(monkeypatch):
    def _fake_teacher_data(*args, **kwargs):
        x = torch.linspace(0.2, 0.8, 600, dtype=torch.float64)
        f = 2.0 * x / ((1.0 - x.square()).square())
        return x, f

    def _fake_fit(
        x,
        f,
        deg_num,
        deg_den,
        min_points,
        min_total_num=0,
        min_total_den=0,
        dtype=None,
        return_support=False,
        return_support_indices=False,
    ):
        if (deg_num, deg_den, min_total_num, min_total_den) != (4, 4, 0, 0):
            return None
        assert return_support is True
        assert return_support_indices is False
        return (
            torch.tensor([2.0], dtype=x.dtype, device=x.device),
            torch.tensor([1.0, -2.0, 1.0], dtype=x.dtype, device=x.device),
            torch.tensor([[1]], dtype=torch.int64, device=x.device),
            torch.tensor([[0], [2], [4]], dtype=torch.int64, device=x.device),
        )

    monkeypatch.setattr(cb, "_gather_teacher_data_1d", _fake_teacher_data)
    monkeypatch.setattr(cb, "_fit_rational_coeffs_1d", _fake_fit)

    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="nn0")
    res = _build_ratpoly_1d_candidates(
        root=target,
        target=target,
        reuse={"nn0": nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_deg_num=4,
        max_deg_den=4,
        min_points=200,
        rel_rms_threshold=1e-9,
    )

    assert len(res) == 1
    cand_root, init_fn, meta = res[0]
    assert isinstance(cand_root, MulNode)
    assert isinstance(cand_root.left, AtomNode)
    assert str(cand_root.left.kind).lower() == "scale"
    assert isinstance(cand_root.right, AtomNode)
    assert str(cand_root.right.kind).lower() == "rratpoly"
    assert cand_root.right.kwargs["deg_num"] == 1
    assert cand_root.right.kwargs["deg_den"] == 4
    assert cand_root.right.kwargs["exps_num_override"] == [[1]]
    assert cand_root.right.kwargs["exps_den_override"] == [[0], [2], [4]]
    assert cand_root.right.kwargs["_mul_scale_tag"] == cand_root.left.tag
    assert meta["pattern_family"] == "ratpoly_1d"
    assert meta["ratpoly_scale_tag"] == cand_root.left.tag
    assert meta["ratpoly_target_tag"] == "nn0"
    assert meta["ratpoly_var_idxs"] == (0,)
    assert meta["deg_num_screen"] == 4
    assert meta["deg_den_screen"] == 4
    assert meta["n_terms_num"] == 1
    assert meta["n_terms_den"] == 3
    assert meta["reuse_blacklist_tags"] == ["nn0", cand_root.left.tag]

    model, atom_to_leaf = build_composite_from_ast(
        cand_root,
        dtype=torch.float64,
        device=torch.device("cpu"),
        reuse={},
        return_atom_map=True,
    )
    init_fn(cand_root, model)

    scale_core = _unwrap_core(atom_to_leaf[id(cand_root.left)])
    rat_core = _unwrap_core(atom_to_leaf[id(cand_root.right)])
    assert isinstance(rat_core, RRationalPolyLeaf)
    assert torch.allclose(
        scale_core.value.detach(),
        torch.tensor(2.0, dtype=scale_core.value.dtype, device=scale_core.value.device),
    )
    assert rat_core.coeffs_num.numel() == 0
    assert torch.allclose(
        rat_core.coeffs_den.detach(),
        torch.tensor([1.0, -2.0, 1.0], dtype=rat_core.coeffs_den.dtype, device=rat_core.coeffs_den.device),
    )


def test_build_ratpoly_1d_candidates_skip_zero_reduced_lead(monkeypatch):
    def _fake_teacher_data(*args, **kwargs):
        x = torch.linspace(0.25, 2.0, 600, dtype=torch.float64)
        f = 3.0 / (1.0 + x)
        return x, f

    def _fake_fit(
        x,
        f,
        deg_num,
        deg_den,
        min_points,
        min_total_num=0,
        min_total_den=0,
        dtype=None,
        return_support=False,
        return_support_indices=False,
    ):
        if (deg_num, deg_den, min_total_num, min_total_den) != (1, 1, 0, 0):
            return None
        assert return_support is True
        assert return_support_indices is False
        return (
            torch.tensor([0.0], dtype=x.dtype, device=x.device),
            torch.tensor([1.0, 1.0], dtype=x.dtype, device=x.device),
            torch.tensor([[1]], dtype=torch.int64, device=x.device),
            torch.tensor([[0], [1]], dtype=torch.int64, device=x.device),
        )

    monkeypatch.setattr(cb, "_gather_teacher_data_1d", _fake_teacher_data)
    monkeypatch.setattr(cb, "_fit_rational_coeffs_1d", _fake_fit)

    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="nn0")
    res = _build_ratpoly_1d_candidates(
        root=target,
        target=target,
        reuse={"nn0": nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_deg_num=1,
        max_deg_den=1,
        min_points=200,
        rel_rms_threshold=1e-9,
    )

    assert res == []


def test_build_sqrt_ratpoly_1d_candidates_wrap_reduced_rational_leaf():
    class _SqrtRationalTeacher(nn.Module):
        def forward(self, x):
            z = x[:, :1].to(dtype=torch.float64)
            rad = (1.0 + z.square()) / ((1.0 - z) * (1.0 - z))
            return torch.sqrt(rad)

    x = torch.linspace(0.2, 0.8, 900, dtype=torch.float64).unsqueeze(1)
    y = torch.zeros_like(x)
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="nn0")

    res = _build_sqrt_ratpoly_1d_candidates(
        root=target,
        target=target,
        reuse={"nn0": _SqrtRationalTeacher()},
        train_loader=[(x, y)],
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_deg_num=4,
        max_deg_den=4,
        min_points=200,
        rel_rms_threshold=1.0e-6,
    )

    assert res
    cand_root, init_fn, meta = res[0]
    assert isinstance(cand_root, PowNode)
    assert float(cand_root.exponent) == 0.5
    assert isinstance(cand_root.base, MulNode)
    assert isinstance(cand_root.base.left, AtomNode)
    assert str(cand_root.base.left.kind).lower() == "scale"
    assert isinstance(cand_root.base.right, AtomNode)
    assert str(cand_root.base.right.kind).lower() == "rratpoly"
    assert meta["pattern_family"] == "sqrt_ratpoly_1d"
    assert meta["sqrt_ratpoly_kind"] == "sqrt"

    model, atom_to_leaf = build_composite_from_ast(
        cand_root,
        dtype=torch.float64,
        device=torch.device("cpu"),
        reuse={},
        return_atom_map=True,
    )
    init_fn(cand_root, model)
    rat_core = _unwrap_core(atom_to_leaf[id(cand_root.base.right)])
    assert isinstance(rat_core, RRationalPolyLeaf)


def test_build_ratpoly_1d_candidates_dedup_identical_support(monkeypatch):
    def _fake_teacher_data(*args, **kwargs):
        x = torch.linspace(0.2, 0.8, 600, dtype=torch.float64)
        f = 2.0 * x / ((1.0 - x.square()).square())
        return x, f

    def _fake_fit(
        x,
        f,
        deg_num,
        deg_den,
        min_points,
        min_total_num=0,
        min_total_den=0,
        dtype=None,
        return_support=False,
        return_support_indices=False,
    ):
        if (deg_num, deg_den, min_total_num, min_total_den) not in {
            (4, 4, 0, 0),
            (5, 5, 0, 0),
        }:
            return None
        assert return_support is True
        return (
            torch.tensor([2.0], dtype=x.dtype, device=x.device),
            torch.tensor([1.0, -2.0, 1.0], dtype=x.dtype, device=x.device),
            torch.tensor([[1]], dtype=torch.int64, device=x.device),
            torch.tensor([[0], [2], [4]], dtype=torch.int64, device=x.device),
        )

    monkeypatch.setattr(cb, "_gather_teacher_data_1d", _fake_teacher_data)
    monkeypatch.setattr(cb, "_fit_rational_coeffs_1d", _fake_fit)

    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="nn0")
    res = _build_ratpoly_1d_candidates(
        root=target,
        target=target,
        reuse={"nn0": nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_deg_num=5,
        max_deg_den=5,
        min_points=200,
        rel_rms_threshold=1e-9,
    )

    assert len(res) == 1
    _, _, meta = res[0]
    assert meta["deg_num_screen"] == 4
    assert meta["deg_den_screen"] == 4
    assert isinstance(meta["ratpoly_target_sig"], int)
    assert meta["signature"][0] == meta["ratpoly_target_sig"]
    assert meta["signature"][1:] == (2, 0, 1, -1, 5, 1, 0, 1, 0, 1)


def test_ratpoly_degree_bands_preserve_highest_and_denominator_pivot():
    exps_num = torch.tensor([[0], [1], [3], [5]], dtype=torch.int64)
    assert _ratpoly_degree_bands(exps_num) == [0, 1, 3]

    exps_den = torch.tensor([[0], [1], [2], [4]], dtype=torch.int64)
    coeffs_den = torch.tensor([1.0, 0.2, -0.5, 0.8], dtype=torch.float64)
    pivot_degree = _ratpoly_den_pivot_degree(exps_den, coeffs_den)
    assert pivot_degree == 0
    assert _ratpoly_degree_bands(exps_den, exclude_degree=pivot_degree) == [1, 2]


def test_build_rratpoly_degree_trim_candidate_finds_nested_product_target():
    scale_main = AtomNode(kind="scale", var_idxs=(), kwargs={}, tag="scale_main")
    scale_extra = AtomNode(kind="scale", var_idxs=(), kwargs={}, tag="scale_extra")
    rat_atom = AtomNode(
        kind="rratpoly",
        var_idxs=(0,),
        kwargs={
            "deg_num": 2,
            "deg_den": 2,
            "exps_num_override": [[0], [2]],
            "exps_den_override": [[0], [1], [2]],
            "_mul_scale_tag": "scale_main",
        },
        tag="rat0",
    )
    root = MulNode(left=MulNode(left=scale_main, right=scale_extra), right=rat_atom)
    model = build_composite_from_ast(
        root,
        dtype=torch.float64,
        device=torch.device("cpu"),
        reuse={},
    )
    state = StageBState(root=root, model=model, reuse={}, val_loss=1.0)
    cand = Candidate(
        label="ratpoly_1d[4]",
        root=root,
        meta={
            "pattern_family": "ratpoly_1d",
            "ratpoly_scale_tag": "scale_main",
            "ratpoly_target_tag": "rat0",
            "ratpoly_var_idxs": (0,),
        },
    )

    trim_cand = _build_rratpoly_degree_trim_candidate(
        state,
        cand,
        branch="num",
        degree=0,
    )

    assert trim_cand is not None
    assert trim_cand.root is not None
    trimmed_rat = trim_cand.root.right
    assert isinstance(trimmed_rat, AtomNode)
    assert str(trimmed_rat.kind).lower() == "rratpoly"
    assert trimmed_rat.kwargs["exps_num_override"] == [[2]]
    assert trimmed_rat.kwargs["exps_den_override"] == [[0], [1], [2]]


def test_build_rratpoly_degree_trim_candidate_supports_plain_multid_ratpoly():
    zero = (0,)
    certificate = solve_rational_coefficient_gauge(
        target_dim=zero,
        input_dims=(zero, zero),
        numerator_exponents=((0, 0), (1, 0), (0, 2)),
        denominator_exponents=((0, 0), (1, 0)),
        coefficient_policy="free_const_only",
    )
    assert certificate.ok
    rat_atom = AtomNode(
        kind="ratpoly",
        var_idxs=(0, 1),
        kwargs={
            "deg_num": 2,
            "deg_den": 1,
            "exps_num_override": [[0, 0], [1, 0], [0, 2]],
            "exps_den_override": [[0, 0], [1, 0]],
        },
        tag="rp0",
    )
    root = rat_atom
    model = build_composite_from_ast(
        root,
        dtype=torch.float64,
        device=torch.device("cpu"),
        reuse={},
    )
    state = StageBState(root=root, model=model, reuse={}, val_loss=1.0)
    cand = Candidate(
        label="ratpoly[4]",
        root=root,
        meta={
            "pattern_family": "ratpoly",
            "leaf_kind": "ratpoly",
            "ratpoly_target_sig": 222,
            "ratpoly_target_tag": "rp0",
            "ratpoly_var_idxs": (0, 1),
            "ratpoly_exps_num_key": ((0, 0), (1, 0), (0, 2)),
            "ratpoly_exps_den_key": ((0, 0), (1, 0)),
            "signature": (222, 0, 3, 2, 0, 0, 1, 0, 0, 2, -1, 2, 2, 0, 0, 1, 0),
            "unit_support_planned": True,
            "coefficient_unit_certificate": certificate.to_dict(),
        },
    )

    trim_cand = _build_rratpoly_degree_trim_candidate(
        state,
        cand,
        branch="num",
        degree=0,
    )

    assert trim_cand is not None
    assert trim_cand.root is not None
    assert isinstance(trim_cand.root, AtomNode)
    assert trim_cand.root.kind.lower() == "ratpoly"
    assert trim_cand.root.kwargs["exps_num_override"] == [[1, 0], [0, 2]]
    assert trim_cand.root.kwargs["exps_den_override"] == [[0, 0], [1, 0]]
    assert trim_cand.signature[0] == 222
    refreshed = trim_cand.meta["coefficient_unit_certificate"]
    assert refreshed["valid"] is True
    assert refreshed["input_dims"] == [["0"], ["0"]]
    assert [item["exponent"] for item in refreshed["numerator"]] == [[1, 0], [0, 2]]
    assert [item["exponent"] for item in refreshed["denominator"]] == [[0, 0], [1, 0]]
    assert trim_cand.meta["unit_support_certificate_refresh"]["status"] == "recomputed"


def test_degree_trim_removes_legacy_unit_certificate_that_cannot_be_refreshed():
    rat_atom = AtomNode(
        kind="ratpoly",
        var_idxs=(0,),
        kwargs={
            "deg_num": 2,
            "deg_den": 1,
            "exps_num_override": [[0], [1], [2]],
            "exps_den_override": [[0], [1]],
        },
        tag="legacy_rp",
    )
    model = build_composite_from_ast(
        rat_atom,
        dtype=torch.float64,
        device=torch.device("cpu"),
        reuse={},
    )
    state = StageBState(root=rat_atom, model=model, reuse={}, val_loss=1.0)
    cand = Candidate(
        label="ratpoly[legacy]",
        root=rat_atom,
        meta={
            "pattern_family": "ratpoly",
            "leaf_kind": "ratpoly",
            "ratpoly_target_tag": "legacy_rp",
            "ratpoly_var_idxs": (0,),
            "ratpoly_exps_num_key": ((0,), (1,), (2,)),
            "ratpoly_exps_den_key": ((0,), (1,)),
            "unit_support_planned": True,
            "coefficient_unit_certificate": {
                "valid": True,
                "target_dim": ["0"],
                # PR2-era certificates did not retain input_dims.
            },
        },
    )

    trimmed = _build_rratpoly_degree_trim_candidate(
        state,
        cand,
        branch="num",
        degree=0,
    )

    assert trimmed is not None
    assert "unit_support_planned" not in trimmed.meta
    assert "coefficient_unit_certificate" not in trimmed.meta
    refresh = trimmed.meta["unit_support_certificate_refresh"]
    assert refresh["status"] == "removed"
    assert refresh["code"] == "missing_certificate_dimensions"


def test_post_accept_ratpoly_trim_keeps_local_trim_without_global_recheck(monkeypatch):
    ctx = _make_ctx()
    ctx.state = StageBState(
        root=AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="base"),
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.309e-11,
    )

    accepted_state = StageBState(
        root=AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={}, tag="rat"),
        model=torch.nn.Identity(),
        reuse={},
        val_loss=2.4479e-11,
    )
    trimmed_state = StageBState(
        root=AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={"trimmed": True}, tag="rat"),
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.4395e-10,
    )
    cand = Candidate(
        label="ratpoly_1d[4]",
        root=accepted_state.root,
        meta={
            "pattern_family": "ratpoly_1d",
            "ratpoly_scale_tag": "scale_main",
            "ratpoly_var_idxs": (0,),
        },
    )
    trim_cand = Candidate(
        label="ratpoly_trim_num0",
        root=trimmed_state.root,
        meta={
            "pattern_family": "ratpoly_1d",
            "ratpoly_scale_tag": "scale_main",
            "ratpoly_var_idxs": (0,),
        },
    )

    class _DummyRatCore:
        def __init__(self, exps_num, exps_den, coeffs_den):
            self.exps_num_full = torch.tensor(exps_num, dtype=torch.int64)
            self.exps_den = torch.tensor(exps_den, dtype=torch.int64)
            self.coeffs_den = torch.tensor(coeffs_den, dtype=torch.float64)

    def _fake_lookup(state, scale_tag):
        if state is accepted_state:
            return (
                AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={}, tag="rat"),
                AtomNode(kind="scale", var_idxs=(), kwargs={}, tag="scale_main"),
                _DummyRatCore([[0], [1]], [[0], [1], [2]], [1.0, 0.25, -0.5]),
                None,
            )
        if state is trimmed_state:
            return (
                AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={}, tag="rat"),
                AtomNode(kind="scale", var_idxs=(), kwargs={}, tag="scale_main"),
                _DummyRatCore([[1]], [[0], [1], [2]], [1.0, 0.25, -0.5]),
                None,
            )
        return None

    def _fake_build(state, cand_local, *, branch, degree):
        if state is accepted_state and branch == "num" and degree == 0:
            return trim_cand
        return None

    def _fake_fit(self, cand_local, epochs_override=None):
        assert cand_local is trim_cand
        return trimmed_state

    def _fake_should_accept(self, cand_local, cand_state_local):
        if cand_state_local is trimmed_state and self.state is accepted_state:
            return True, "loss-below-floor-simpler"
        if cand_state_local is trimmed_state and self.state is trimmed_state:
            return True, "loss-below-floor-simpler"
        if cand_state_local is trimmed_state and self.state is ctx.state:
            return False, "loss-below-floor-too-much-regression(ratio=1.100e+01, cap=1.309e-10)"
        raise AssertionError("unexpected should_accept call")

    monkeypatch.setattr(stageb_engine, "_lookup_rratpoly_trim_target", _fake_lookup)
    monkeypatch.setattr(stageb_engine, "_build_rratpoly_degree_trim_candidate", _fake_build)
    monkeypatch.setattr(StageBContext, "fit_candidate", _fake_fit)
    monkeypatch.setattr(StageBContext, "should_accept", _fake_should_accept)

    out_cand, out_state = ctx._post_accept_ratpoly_trim(cand, accepted_state)

    assert out_cand is trim_cand
    assert out_state is trimmed_state
    assert out_cand.label == "ratpoly_1d[4]"
    assert out_cand.meta["ratpoly_trim_steps"] == [
        {
            "branch": "num",
            "degree": 0,
            "reason_local": "loss-below-floor-simpler",
            "reason_anchor": "loss-below-floor-simpler",
        }
    ]


def test_post_accept_ratpoly_trim_skips_duplicate_trim_support(monkeypatch):
    ctx = _make_ctx()
    ctx.state = StageBState(
        root=AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="base"),
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.0,
    )

    raw_cand = Candidate(
        label="ratpoly_1d[10]",
        root=AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={}, tag="raw"),
        meta={
            "pattern_family": "ratpoly_1d",
            "ratpoly_scale_tag": "scale_main",
            "ratpoly_target_sig": 12345,
            "ratpoly_var_idxs": (0,),
        },
    )
    accepted_state = StageBState(
        root=raw_cand.root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.0e-12,
    )
    trim_cand = Candidate(
        label="ratpoly_1d[10]/trim_num0",
        root=AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={"trimmed": True}, tag="trim"),
        meta={
            "pattern_family": "ratpoly_1d",
            "ratpoly_scale_tag": "scale_main",
            "ratpoly_target_sig": 12345,
            "ratpoly_var_idxs": (0,),
            "signature": (12345, 2, 0, 1, -1, 5, 1, 0, 1, 0, 1),
        },
        signature=(12345, 2, 0, 1, -1, 5, 1, 0, 1, 0, 1),
    )
    trimmed_state = StageBState(
        root=trim_cand.root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=5.0e-13,
    )

    class _DummyRatCore:
        def __init__(self):
            self.exps_num_full = torch.tensor([[0], [1]], dtype=torch.int64)
            self.exps_den = torch.tensor([[0], [2], [4]], dtype=torch.int64)
            self.coeffs_den = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)

    fit_calls = {"n": 0}

    def _fake_lookup(state, scale_tag):
        if scale_tag != "scale_main":
            return None
        return (
            AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={}, tag="rat"),
            AtomNode(kind="scale", var_idxs=(), kwargs={}, tag="scale_main"),
            _DummyRatCore(),
            None,
        )

    def _fake_build(state, cand_local, *, branch, degree):
        if branch == "num" and degree == 0:
            return trim_cand
        return None

    def _fake_fit(self, cand_local, epochs_override=None):
        assert cand_local is trim_cand
        fit_calls["n"] += 1
        return trimmed_state

    def _fake_should_accept(self, cand_local, cand_state_local):
        if cand_state_local is trimmed_state:
            return True, "loss-below-floor-simpler"
        raise AssertionError("unexpected should_accept call")

    monkeypatch.setattr(stageb_engine, "_lookup_rratpoly_trim_target", _fake_lookup)
    monkeypatch.setattr(stageb_engine, "_build_rratpoly_degree_trim_candidate", _fake_build)
    monkeypatch.setattr(StageBContext, "fit_candidate", _fake_fit)
    monkeypatch.setattr(StageBContext, "should_accept", _fake_should_accept)

    first_cand, first_state = ctx._post_accept_ratpoly_trim(raw_cand, accepted_state)
    second_cand, second_state = ctx._post_accept_ratpoly_trim(raw_cand, accepted_state)

    assert first_cand is trim_cand
    assert first_state is trimmed_state
    assert second_cand is raw_cand
    assert second_state is accepted_state
    assert fit_calls["n"] == 1


def test_post_accept_ratpoly_trim_skips_duplicate_multid_support(monkeypatch):
    ctx = _make_ctx()
    raw_cand = Candidate(
        label="ratpoly[6]",
        root=AtomNode(kind="ratpoly", var_idxs=(0, 1), kwargs={}, tag="rp0"),
        meta={
            "pattern_family": "ratpoly",
            "leaf_kind": "ratpoly",
            "ratpoly_target_sig": 67890,
            "ratpoly_target_tag": "rp0",
            "ratpoly_var_idxs": (0, 1),
            "ratpoly_exps_num_key": ((0, 0), (1, 0), (0, 2)),
            "ratpoly_exps_den_key": ((0, 0), (1, 0)),
        },
    )
    accepted_state = StageBState(
        root=raw_cand.root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.0e-12,
    )
    trim_cand = Candidate(
        label="ratpoly[6]/trim_num0",
        root=AtomNode(kind="ratpoly", var_idxs=(0, 1), kwargs={"trimmed": True}, tag="rp0"),
        meta={
            "pattern_family": "ratpoly",
            "leaf_kind": "ratpoly",
            "ratpoly_target_sig": 67890,
            "ratpoly_target_tag": "rp0",
            "ratpoly_var_idxs": (0, 1),
            "ratpoly_exps_num_key": ((1, 0), (0, 2)),
            "ratpoly_exps_den_key": ((0, 0), (1, 0)),
            "signature": (67890, 0, 2, 2, 1, 0, 0, 2, -1, 2, 2, 0, 0, 1, 0),
        },
        signature=(67890, 0, 2, 2, 1, 0, 0, 2, -1, 2, 2, 0, 0, 1, 0),
    )
    trimmed_state = StageBState(
        root=trim_cand.root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=5.0e-13,
    )

    class _DummyRatCore:
        def __init__(self):
            self.exps_num = torch.tensor([[0, 0], [1, 0], [0, 2]], dtype=torch.int64)
            self.exps_den = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
            self.coeffs_den = torch.tensor([1.0, -0.5], dtype=torch.float64)

    fit_calls = {"n": 0}

    def _fake_lookup(state, cand_local):
        return (
            AtomNode(kind="ratpoly", var_idxs=(0, 1), kwargs={}, tag="rp0"),
            None,
            _DummyRatCore(),
            None,
            "ratpoly",
        )

    def _fake_build(state, cand_local, *, branch, degree):
        if branch == "num" and degree == 0:
            return trim_cand
        return None

    def _fake_fit(self, cand_local, epochs_override=None):
        assert cand_local is trim_cand
        fit_calls["n"] += 1
        return trimmed_state

    def _fake_should_accept(self, cand_local, cand_state_local):
        if cand_state_local is trimmed_state:
            return True, "loss-below-floor-simpler"
        raise AssertionError("unexpected should_accept call")

    monkeypatch.setattr(stageb_engine, "_lookup_ratpoly_trim_target", _fake_lookup)
    monkeypatch.setattr(stageb_engine, "_build_rratpoly_degree_trim_candidate", _fake_build)
    monkeypatch.setattr(StageBContext, "fit_candidate", _fake_fit)
    monkeypatch.setattr(StageBContext, "should_accept", _fake_should_accept)

    first_cand, first_state = ctx._post_accept_ratpoly_trim(raw_cand, accepted_state)
    second_cand, second_state = ctx._post_accept_ratpoly_trim(raw_cand, accepted_state)

    assert first_cand is trim_cand
    assert first_state is trimmed_state
    assert second_cand is raw_cand
    assert second_state is accepted_state
    assert fit_calls["n"] == 1


def test_fit_candidate_filters_reuse_blacklist_tags(monkeypatch):
    ctx = _make_ctx()
    batch = torch.ones(8, 1, dtype=torch.float64)
    ctx.train_loader = batch
    ctx.val_loader = batch
    ctx.state = StageBState(
        root=AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="base"),
        model=torch.nn.Identity(),
        reuse={
            "keep": nn.Identity(),
            "target": nn.Identity(),
            "scale": nn.Identity(),
        },
        val_loss=1.0,
    )

    seen = {}

    def _fake_fit_candidate_root(
        *,
        root,
        reuse,
        train_loader,
        val_loader,
        lm_hp,
        device,
        dtype,
        epochs_stageB,
        loss_scale,
        trig_by_axis,
        custom_init_fn,
        fresh_nn_factory,
        atom_factory,
    ):
        seen["reuse_keys"] = sorted(reuse.keys())
        return StageBState(root=root, model=torch.nn.Identity(), reuse=dict(reuse), val_loss=0.5)

    monkeypatch.setattr(stageb_fitting, "_fit_candidate_root", _fake_fit_candidate_root)

    cand = Candidate(
        label="ratpoly_1d[10]",
        root=AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={}, tag="target"),
        meta={"reuse_blacklist_tags": ["target", "scale"]},
    )

    out = ctx.fit_candidate(cand)

    assert seen["reuse_keys"] == ["keep"]
    assert out.val_loss == 0.5


def test_select_ratpoly_candidate_prefers_trimmed_descendant(monkeypatch):
    ctx = _make_ctx()
    raw_state = StageBState(
        root=AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={}, tag="raw"),
        model=torch.nn.Identity(),
        reuse={},
        val_loss=2.0e-29,
    )
    trim_state = StageBState(
        root=AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={"trimmed": True}, tag="trim"),
        model=torch.nn.Identity(),
        reuse={},
        val_loss=8.0e-13,
    )
    raw_cand = Candidate(
        label="ratpoly_1d[10]",
        root=raw_state.root,
        meta={"pattern_family": "ratpoly_1d"},
    )
    trim_cand = Candidate(
        label="ratpoly_1d[10]",
        root=trim_state.root,
        meta={"pattern_family": "ratpoly_1d", "ratpoly_trim_steps": [{"branch": "num", "degree": 0}]},
    )

    def _fake_trim(self, cand, cand_state):
        assert cand is raw_cand
        assert cand_state is raw_state
        return trim_cand, trim_state

    def _fake_should_accept(self, cand, cand_state):
        if cand_state is raw_state:
            return False, "loss-below-floor-not-simpler"
        if cand_state is trim_state:
            return True, "loss-below-floor-simpler"
        raise AssertionError("unexpected should_accept call")

    monkeypatch.setattr(StageBContext, "_post_accept_ratpoly_trim", _fake_trim)
    monkeypatch.setattr(StageBContext, "should_accept", _fake_should_accept)

    ok, out_cand, out_state, reason = ctx._select_ratpoly_candidate(raw_cand, raw_state)

    assert ok
    assert out_cand is trim_cand
    assert out_state is trim_state
    assert reason == "loss-below-floor-simpler"


def test_select_ratpoly_candidate_falls_back_to_raw_when_trim_loses(monkeypatch):
    ctx = _make_ctx()
    raw_state = StageBState(
        root=AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={}, tag="raw"),
        model=torch.nn.Identity(),
        reuse={},
        val_loss=2.0e-29,
    )
    trim_state = StageBState(
        root=AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={"trimmed": True}, tag="trim"),
        model=torch.nn.Identity(),
        reuse={},
        val_loss=8.0e-13,
    )
    raw_cand = Candidate(
        label="ratpoly_1d[10]",
        root=raw_state.root,
        meta={"pattern_family": "ratpoly_1d"},
    )
    trim_cand = Candidate(
        label="ratpoly_1d[10]",
        root=trim_state.root,
        meta={"pattern_family": "ratpoly_1d", "ratpoly_trim_steps": [{"branch": "num", "degree": 0}]},
    )

    def _fake_trim(self, cand, cand_state):
        assert cand is raw_cand
        assert cand_state is raw_state
        return trim_cand, trim_state

    def _fake_should_accept(self, cand, cand_state):
        if cand_state is raw_state:
            return True, "loss-below-floor-better-loss"
        if cand_state is trim_state:
            return False, "loss-below-floor-not-simpler"
        raise AssertionError("unexpected should_accept call")

    monkeypatch.setattr(StageBContext, "_post_accept_ratpoly_trim", _fake_trim)
    monkeypatch.setattr(StageBContext, "should_accept", _fake_should_accept)

    ok, out_cand, out_state, reason = ctx._select_ratpoly_candidate(raw_cand, raw_state)

    assert ok
    assert out_cand is raw_cand
    assert out_state is raw_state
    assert reason == "loss-below-floor-better-loss"


def test_candidate_pattern_name_normalizes_indexed_labels():
    assert candidate_pattern_name("inv_poly[2]") == "inv_poly"
    assert candidate_pattern_name("ratpoly[2]") == "ratpoly"
    assert candidate_pattern_name("ratpoly_1d[2]") == "ratpoly_1d"
    assert candidate_pattern_name("factorized_search[0]") == "factorized_search"
    assert candidate_pattern_name("sqrt_ratpoly") == "sqrt_ratpoly"


def test_is_pattern_disabled_matches_family_and_exact_label():
    ctx_family = _make_ctx(disabled_patterns={"ratpoly"})
    assert ctx_family.is_pattern_disabled("ratpoly[2]")
    assert ctx_family.is_pattern_disabled(Candidate(label="ratpoly[2]"))

    ctx_exact = _make_ctx(disabled_patterns={"ratpoly[2]"})
    assert ctx_exact.is_pattern_disabled("ratpoly[2]")
    assert not ctx_exact.is_pattern_disabled("ratpoly[1]")


def test_precheck_rejects_indexed_candidate_when_family_disabled():
    ctx = _make_ctx(disabled_patterns={"ratpoly"})
    cand = Candidate(
        label="ratpoly[1]",
        root=AtomNode(kind="ratpoly", var_idxs=(0,), kwargs={"deg_num": 2, "deg_den": 2}, tag=None),
    )
    pre = ctx.precheck_candidate("rule_multi_dnn", cand)
    assert not pre.ok
    assert pre.reason == "disabled-pattern"


def test_precheck_rejects_indexed_inv_poly_when_family_disabled():
    ctx = _make_ctx(disabled_patterns={"inv_poly"})
    cand = Candidate(
        label="inv_poly[1]",
        root=AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 2}, tag=None),
    )
    pre = ctx.precheck_candidate("rule_uni_nn", cand)
    assert not pre.ok
    assert pre.reason == "disabled-pattern"


def test_precheck_rejects_indexed_ratpoly_1d_when_family_disabled():
    ctx = _make_ctx(disabled_patterns={"ratpoly_1d"})
    cand = Candidate(
        label="ratpoly_1d[1]",
        root=AtomNode(kind="ratpoly", var_idxs=(0,), kwargs={"deg_num": 2, "deg_den": 2}, tag=None),
    )
    pre = ctx.precheck_candidate("rule_uni_nn", cand)
    assert not pre.ok
    assert pre.reason == "disabled-pattern"
