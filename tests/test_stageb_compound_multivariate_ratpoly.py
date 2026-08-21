# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import pytest
import torch

from nestynet_sr.sr_core.atoms import PolyLeaf
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    FixedConst,
    FreeConst,
    LogNode,
    MulNode,
    PowNode,
    Var,
    ast_to_human_readable,
    compound_input_expr,
    effective_arity,
    extra_input_var_idxs,
    get_input_exprs,
    has_nontrivial_input,
    _is_module_compatible_with_atom,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search import candidate_builders
from nestynet_sr.sr_search.features import ScaleSpec
from nestynet_sr.sr_search.stageB import rules as stageb_rules


def _spec(us, x_dims, y_dim):
    return UnitsSpec(
        unit_system=us,
        x_dims=tuple(x_dims),
        y_dim=y_dim,
    )


def _compound_target():
    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf0",
        inputs=(z_expr, Var(2)),
    )


def _compound_univariate_target():
    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="leaf0",
        inputs=(z_expr,),
    )


def _assert_preserves_compound_inputs(atom):
    assert has_nontrivial_input(atom)
    assert effective_arity(atom) == 2
    assert compound_input_expr(atom) is not None
    assert tuple(extra_input_var_idxs(atom)) == (2,)


@pytest.mark.parametrize("trainable_constant", [False, True])
def test_rule_multidnn_uses_evaluable_effective_input_dims_for_compound_ratpoly_probe(
    monkeypatch,
    trainable_constant,
):
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    constant = (
        FreeConst("ell", init=2.0)
        if trainable_constant
        else FixedConst("ell", value=2.0)
    )
    target = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2),
        tag="leaf0",
        inputs=(
            MulNode(Var(0), PowNode(Var(1), -1.0)),
            MulNode(constant, Var(2)),
        ),
    )
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(L, L, dimless),
        y_dim=L,
        fixed_const_dims={} if trainable_constant else {"ell": L},
        fixed_const_values={} if trainable_constant else {"ell": 2.0},
        free_const_dims={"ell": L} if trainable_constant else {},
    )
    captured = {}

    monkeypatch.setattr(stageb_rules, "build_atom_to_leaf_map", lambda _root, _model: {})
    monkeypatch.setattr(stageb_rules, "_probe_genadd_for_nn_leaf", lambda **_kwargs: None)
    monkeypatch.setattr(stageb_rules, "_probe_trapped_for_nn_leaf", lambda **_kwargs: None)
    monkeypatch.setattr(
        stageb_rules,
        "macro_arg_wrapper_policy",
        lambda _ctx, _lm_hp, _target: SimpleNamespace(trig=True),
    )
    monkeypatch.setattr(stageb_rules, "propose_exp_of_quadratic", lambda *_a, **_k: [])
    monkeypatch.setattr(stageb_rules, "propose_exp_poly_from_log_hint", lambda *_a, **_k: [])
    monkeypatch.setattr(stageb_rules, "propose_rational_linear", lambda *_a, **_k: None)
    monkeypatch.setattr(stageb_rules, "propose_sin_cos_from_inverse_hint", lambda *_a, **_k: [])
    monkeypatch.setattr(stageb_rules, "propose_sinc_family", lambda *_a, **_k: [])
    monkeypatch.setattr(stageb_rules, "propose_symexp_denom_family", lambda *_a, **_k: [])
    monkeypatch.setattr(stageb_rules, "propose_tanh_family", lambda *_a, **_k: [])
    monkeypatch.setattr(stageb_rules, "_build_inv_poly_candidates", lambda **_kwargs: [])

    def _capture_ratpoly(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(stageb_rules, "_build_ratpoly_candidates", _capture_ratpoly)

    ctx = SimpleNamespace(
        state=SimpleNamespace(root=target, model=torch.nn.Identity(), reuse={"leaf0": object()}),
        enforce_units=True,
        units_spec=spec,
        infer_target_dim=lambda _target: L,
        train_loader_probe=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        trig_by_axis={},
        scale_specs=[],
        scaling_by_axis={},
        lm_hp=SimpleNamespace(),
        verbose=False,
        is_pattern_disabled=lambda name: name == "inv_poly",
        cached=lambda _key, fn: None,
        log=lambda *_a, **_k: None,
    )

    stageb_rules.RuleMultiDNN().propose(ctx, target)

    assert captured["target_dim"] == tuple(L)
    if trainable_constant:
        # Symbolic unit inference knows the declaration, but the current
        # input-expression bridge cannot evaluate a fitted FreeConst leaf.
        # The real builder receives no unit support payload and fails closed.
        assert captured["x_dims"] is None
        assert candidate_builders._build_ratpoly_candidates(
            root=target,
            target=target,
            reuse={"leaf0": object()},
            train_loader=[],
            device=torch.device("cpu"),
            dtype=torch.float64,
            enforce_units=True,
            target_dim=captured["target_dim"],
            x_dims=captured["x_dims"],
        ) == []
    else:
        assert captured["x_dims"] == [tuple(dimless), tuple(L)]
    # A mixed dimensionless/unitful coordinate set still uses the exact
    # support planner; it must not disable units merely because the old
    # homogeneous-basis shortcut was inapplicable.
    assert captured["enforce_units"] is True


def test_build_ratpoly_candidates_uses_effective_arity_and_preserves_inputs(monkeypatch):
    target = _compound_target()
    X = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]],
        dtype=torch.float64,
    )
    F = torch.tensor([1.0, 1.2, 1.4, 1.6], dtype=torch.float64)

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )
    monkeypatch.setattr(
        candidate_builders,
        "_rational_probe_nd",
        lambda *_a, **_k: 0.0,
    )

    results = candidate_builders._build_ratpoly_candidates(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=1,
        max_points=8,
        max_deg_num=1,
        max_deg_den=1,
        rel_rms_threshold=0.1,
    )

    assert results
    cand_root, _init, _meta = results[0]
    assert cand_root.kind == "ratpoly"
    _assert_preserves_compound_inputs(cand_root)


def test_build_sqrt_ratpoly_candidate_preserves_inputs_for_compound_target(monkeypatch):
    target = _compound_target()
    X = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]],
        dtype=torch.float64,
    )
    F = torch.tensor([1.0, 1.2, 1.4, 1.6], dtype=torch.float64)
    calls = {"n": 0}

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    def _fake_probe(*_args, **_kwargs):
        calls["n"] += 1
        coeffs = torch.ones(1, dtype=torch.float64)
        if calls["n"] == 1:
            return 0.0, coeffs, coeffs
        return 1.0, coeffs, coeffs

    monkeypatch.setattr(candidate_builders, "_rational_probe_nd", _fake_probe)

    cand_root, init_fn, _meta = candidate_builders._build_sqrt_ratpoly_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=1,
        max_points=8,
        rel_rms_threshold=0.1,
    )

    assert init_fn is not None
    assert isinstance(cand_root, PowNode)
    assert cand_root.base.kind == "ratpoly"
    assert cand_root.base.kwargs["deg_num"] == 4
    assert cand_root.base.kwargs["deg_den"] == 4
    _assert_preserves_compound_inputs(cand_root.base)


def test_build_sqrt_ratpoly_candidate_propagates_sparse_support(monkeypatch):
    target = _compound_target()
    X = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0], [3.5, 4.5]],
        dtype=torch.float64,
    )
    F = torch.tensor([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=torch.float64)
    calls = {"n": 0}

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    def _fake_probe(*_args, **_kwargs):
        calls["n"] += 1
        coeffs = torch.ones(1, dtype=torch.float64)
        if calls["n"] == 1:
            return 0.0, coeffs, coeffs
        return 1.0, coeffs, coeffs

    def _fake_fit_nd(*args, **kwargs):
        a = torch.tensor([1.0, 0.5], dtype=torch.float64)
        b = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        support_num = torch.tensor([0, 5], dtype=torch.int64)
        support_den = torch.tensor([0, 4, 8], dtype=torch.int64)
        return a, b, support_num, support_den

    monkeypatch.setattr(candidate_builders, "_rational_probe_nd", _fake_probe)
    monkeypatch.setattr(candidate_builders, "_fit_rational_coeffs_nd", _fake_fit_nd)

    cand_root, init_fn, _meta = candidate_builders._build_sqrt_ratpoly_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=1,
        max_points=8,
        rel_rms_threshold=0.1,
    )

    assert init_fn is not None
    assert isinstance(cand_root, PowNode)
    assert cand_root.base.kind == "ratpoly"
    assert cand_root.base.kwargs["deg_num"] == 2
    assert cand_root.base.kwargs["deg_den"] == 3
    assert cand_root.base.kwargs["exps_num_override"] == [[0, 0], [2, 0]]
    assert cand_root.base.kwargs["exps_den_override"] == [[0, 0], [1, 1], [2, 1]]
    _assert_preserves_compound_inputs(cand_root.base)


def test_build_quadratic_poly_candidate_preserves_inputs_for_compound_target(monkeypatch):
    target = _compound_target()
    X = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]],
        dtype=torch.float64,
    )
    F = X[:, 0] + 2.0 * X[:, 1]

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    cand_root, init_fn = candidate_builders._build_quadratic_poly_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        degree=2,
        min_points=1,
        max_points=8,
        rel_rms_threshold=0.1,
    )

    assert init_fn is not None
    assert cand_root.kind == "poly"
    _assert_preserves_compound_inputs(cand_root)


def test_quadratic_poly_custom_init_matches_by_inputs_not_raw_var_idxs(monkeypatch):
    target = _compound_target()
    sibling = AtomNode(
        kind="poly",
        var_idxs=(0, 1, 2),
        tag="sibling",
        inputs=(MulNode(Var(0), Var(1)), Var(2)),
        kwargs={"degree": 2, "min_total": 0},
    )
    root = AddNode(sibling, target)
    X = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]],
        dtype=torch.float64,
    )
    F = X[:, 0] + 2.0 * X[:, 1]

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    cand_root, init_fn = candidate_builders._build_quadratic_poly_candidate(
        root=root,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        degree=2,
        min_points=1,
        max_points=8,
        rel_rms_threshold=0.1,
    )

    assert init_fn is not None
    sibling_core = PolyLeaf(n_in=2, degree=2, min_total=0, dtype=torch.float64)
    candidate_core = PolyLeaf(n_in=2, degree=2, min_total=0, dtype=torch.float64)
    with torch.no_grad():
        sibling_core.coeffs.fill_(7.0)

    model = SimpleNamespace(
        leaf=[
            SimpleNamespace(core=sibling_core),
            SimpleNamespace(core=candidate_core),
        ]
    )

    init_fn(cand_root, model)

    assert torch.allclose(sibling_core.coeffs, torch.full_like(sibling_core.coeffs, 7.0))
    assert torch.linalg.norm(candidate_core.coeffs).item() > 0.0


def test_build_log_ratpoly_candidate_preserves_inputs_for_compound_target(monkeypatch):
    target = _compound_target()
    X = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]],
        dtype=torch.float64,
    )
    F = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )
    monkeypatch.setattr(
        candidate_builders,
        "_rational_probe_nd",
        lambda *_a, **_k: 0.0,
    )

    cand_root, init_fn = candidate_builders._build_log_ratpoly_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=1,
        max_points=8,
        rel_rms_threshold=0.1,
    )

    assert init_fn is not None
    assert isinstance(cand_root, LogNode)
    assert cand_root.arg.kind == "ratpoly"
    _assert_preserves_compound_inputs(cand_root.arg)


def test_build_log_ratpoly_candidate_propagates_sparse_support(monkeypatch):
    target = _compound_target()
    X = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0], [3.5, 4.5]],
        dtype=torch.float64,
    )
    F = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float64)

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )
    monkeypatch.setattr(
        candidate_builders,
        "_rational_probe_nd",
        lambda *_a, **_k: 0.0,
    )

    def _fake_fit_nd(*args, **kwargs):
        a = torch.tensor([1.0, 0.5], dtype=torch.float64)
        b = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        support_num = torch.tensor([0, 5], dtype=torch.int64)
        support_den = torch.tensor([0, 4, 8], dtype=torch.int64)
        return a, b, support_num, support_den

    monkeypatch.setattr(candidate_builders, "_fit_rational_coeffs_nd", _fake_fit_nd)

    cand_root, init_fn = candidate_builders._build_log_ratpoly_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=1,
        max_points=8,
        rel_rms_threshold=0.1,
    )

    assert init_fn is not None
    assert isinstance(cand_root, LogNode)
    assert cand_root.arg.kind == "ratpoly"
    assert cand_root.arg.kwargs["deg_num"] == 2
    assert cand_root.arg.kwargs["deg_den"] == 3
    assert cand_root.arg.kwargs["exps_num_override"] == [[0, 0], [2, 0]]
    assert cand_root.arg.kwargs["exps_den_override"] == [[0, 0], [1, 1], [2, 1]]
    _assert_preserves_compound_inputs(cand_root.arg)


def test_build_log_poly_candidate_uses_effective_arity_and_preserves_inputs(monkeypatch):
    target = _compound_target()
    X = torch.tensor(
        [
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
            [2.5, 3.5],
            [3.0, 4.0],
            [3.5, 4.5],
            [4.0, 5.0],
            [4.5, 5.5],
        ],
        dtype=torch.float64,
    )
    F = torch.log(3.0 + 0.5 * X[:, 0] + 0.25 * X[:, 1])

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    cand_root, init_fn = candidate_builders._build_log_poly_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        degree=1,
        min_points=1,
        max_points=16,
        rel_rms_threshold=0.1,
    )

    assert init_fn is not None
    assert isinstance(cand_root, LogNode)
    assert cand_root.arg.kind == "poly"
    _assert_preserves_compound_inputs(cand_root.arg)


def test_build_pure_exp_rat_candidate_preserves_inputs_for_compound_target(monkeypatch):
    target = _compound_target()
    X = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]],
        dtype=torch.float64,
    )
    F = torch.tensor([1.0, 1.1, 1.2, 1.3], dtype=torch.float64)
    calls = {"n": 0}

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    def _probe(*_args, **_kwargs):
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 1.0

    monkeypatch.setattr(candidate_builders, "_rational_probe_nd", _probe)

    cand_root, init_fn = candidate_builders._build_pure_exp_rat_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=1,
        max_points=8,
    )

    assert init_fn is not None
    assert cand_root.kind == "exp_ratpoly"
    _assert_preserves_compound_inputs(cand_root)


def test_make_power_exp_ratpoly_rewrite_preserves_inputs_for_compound_target():
    target = _compound_target()

    cand_root = candidate_builders._make_power_exp_ratpoly_rewrite(
        root=target,
        target=target,
        pivot_axis=2,
        exponent=1.0,
        deg_num=2,
        deg_den=2,
    )

    assert isinstance(cand_root, MulNode)
    assert cand_root.right.kind == "exp_ratpoly"
    _assert_preserves_compound_inputs(cand_root.right)


def test_build_power_exp_1d_candidate_uses_compound_scalar_input(monkeypatch):
    target = _compound_univariate_target()
    z_expr = compound_input_expr(target)
    X = torch.linspace(1.0, 4.0, 16, dtype=torch.float64).view(-1, 1)
    F = X.view(-1).clone()
    captured = {}

    def _gather(*_args, **kwargs):
        captured["input_expr"] = kwargs.get("input_expr")
        captured["axis"] = kwargs.get("axis")
        return X, F

    monkeypatch.setattr(candidate_builders, "_gather_teacher_data_1d", _gather)
    monkeypatch.setattr(
        candidate_builders,
        "_fit_power_coeffs_1d",
        lambda *_args, **_kwargs: (1.0, 1.0),
    )

    cand_root, init_fn = candidate_builders._build_power_exp_1d_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=1,
    )

    assert init_fn is not None
    assert isinstance(cand_root, MulNode)
    assert captured["axis"] is None
    assert ast_to_human_readable(captured["input_expr"]) == ast_to_human_readable(z_expr)
    assert cand_root.left.kind == "power"
    assert cand_root.right.kind == "exp_poly"
    assert ast_to_human_readable(get_input_exprs(cand_root.left)[0]) == ast_to_human_readable(z_expr)
    assert ast_to_human_readable(get_input_exprs(cand_root.right)[0]) == ast_to_human_readable(z_expr)


def test_build_power_exp_rat_candidate_uses_effective_extra_var_pivot(monkeypatch):
    target = _compound_target()
    X = torch.tensor(
        [
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
            [2.5, 3.5],
            [3.0, 4.0],
            [3.5, 4.5],
            [4.0, 5.0],
            [4.5, 5.5],
        ],
        dtype=torch.float64,
    )
    F = X[:, 1] * torch.exp(0.1 * X[:, 0])
    calls = {"n": 0}

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    def _probe(*_args, **_kwargs):
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 1.0

    monkeypatch.setattr(candidate_builders, "_rational_probe_nd", _probe)

    scale_specs = [
        ScaleSpec(
            name="scale_x2",
            indices=[2],
            k_hat=1.0,
            mean=1.0,
            std=0.0,
            rel_std=0.0,
            n_points=X.shape[0],
        )
    ]

    cand_root, init_fn = candidate_builders._build_power_exp_rat_candidate(
        root=target,
        target=target,
        scale_specs=scale_specs,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=1,
        max_points=16,
    )

    assert init_fn is not None
    assert isinstance(cand_root, MulNode)
    assert cand_root.left.kind == "power"
    assert cand_root.right.kind == "exp_ratpoly"
    assert effective_arity(cand_root.left) == 1
    assert ast_to_human_readable(get_input_exprs(cand_root.left)[0]) == "x2"
    _assert_preserves_compound_inputs(cand_root.right)


def test_build_power_exp_rat_candidate_can_use_compound_pivot_spec(monkeypatch):
    target = _compound_target()
    z_expr = compound_input_expr(target)
    X = torch.tensor(
        [
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
            [2.5, 3.5],
            [3.0, 4.0],
            [3.5, 4.5],
            [4.0, 5.0],
            [4.5, 5.5],
        ],
        dtype=torch.float64,
    )
    F = X[:, 0] * torch.exp(0.1 * X[:, 1])
    calls = {"n": 0}

    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    def _probe(*_args, **_kwargs):
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 1.0

    monkeypatch.setattr(candidate_builders, "_rational_probe_nd", _probe)

    scale_specs = [
        ScaleSpec(
            name="scale_z",
            indices=[0, 1],
            k_hat=1.0,
            mean=1.0,
            std=0.0,
            rel_std=0.0,
            n_points=X.shape[0],
            compound_name="x0/x1",
            compound_expr=z_expr,
        )
    ]

    cand_root, init_fn = candidate_builders._build_power_exp_rat_candidate(
        root=target,
        target=target,
        scale_specs=scale_specs,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        min_points=1,
        max_points=16,
    )

    assert init_fn is not None
    assert isinstance(cand_root, MulNode)
    assert cand_root.left.kind == "power"
    assert cand_root.right.kind == "exp_ratpoly"
    assert effective_arity(cand_root.left) == 1
    assert ast_to_human_readable(get_input_exprs(cand_root.left)[0]) == ast_to_human_readable(z_expr)
    _assert_preserves_compound_inputs(cand_root.right)


def test_power_product_rule_builds_factors_on_compound_effective_inputs(monkeypatch):
    target = _compound_target()
    z_expr = compound_input_expr(target)
    z = torch.linspace(1.0, 6.0, 256, dtype=torch.float64)
    x2 = torch.linspace(2.0, 7.0, 256, dtype=torch.float64)
    X = torch.stack((z, x2), dim=1)
    F = 3.0 * X[:, 0] * (X[:, 1] ** 2)

    monkeypatch.setattr(stageb_rules, "build_atom_to_leaf_map", lambda _root, _model: {id(target): object()})
    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    ctx = SimpleNamespace(
        state=SimpleNamespace(root=target, model=object(), reuse={"leaf0": object()}),
        train_loader_probe=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        units_spec=None,
        log=lambda *_args, **_kwargs: None,
    )

    cands = stageb_rules.RulePowerProduct().propose(ctx, target)

    assert cands
    atoms = []

    def _walk(node):
        if isinstance(node, AtomNode):
            atoms.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, PowNode):
            _walk(node.base)
        elif isinstance(node, LogNode):
            _walk(node.arg)

    _walk(cands[0].root)
    rpoly_inputs = [
        get_input_exprs(atom)[0]
        for atom in atoms
        if str(atom.kind).lower() == "rpoly"
    ]
    assert any(ast_to_human_readable(inp) == ast_to_human_readable(z_expr) for inp in rpoly_inputs)
    assert any(ast_to_human_readable(inp) == "x2" for inp in rpoly_inputs)


def test_power_product_partial_peel_uses_fresh_residual_nn_tag(monkeypatch):
    target = _compound_target()
    z = torch.linspace(1.0, 6.0, 256, dtype=torch.float64)
    x2 = torch.linspace(2.0, 7.0, 256, dtype=torch.float64)
    X = torch.stack((z, x2), dim=1)
    F = (X[:, 0] ** -1.0) * (X[:, 1] ** 1.5)

    monkeypatch.setattr(stageb_rules, "build_atom_to_leaf_map", lambda _root, _model: {id(target): object()})
    monkeypatch.setattr(
        candidate_builders,
        "_gather_atom_teacher_data",
        lambda **_kwargs: (X, F),
    )

    ctx = SimpleNamespace(
        state=SimpleNamespace(root=target, model=object(), reuse={"leaf0": object()}),
        train_loader_probe=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        units_spec=None,
        log=lambda *_args, **_kwargs: None,
    )

    cands = stageb_rules.RulePowerProduct().propose(ctx, target)
    partial = next(c for c in cands if c.label == "power_product_partial_peel")

    nn_atoms = []

    def _walk(node):
        if isinstance(node, AtomNode):
            if str(node.kind).lower() == "nn":
                nn_atoms.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, PowNode):
            _walk(node.base)

    _walk(partial.root)
    assert len(nn_atoms) == 1
    residual = nn_atoms[0]
    assert residual.tag != target.tag
    assert str(residual.tag).startswith("leaf0_pp_resid_")
    assert effective_arity(residual) == 1
    assert ast_to_human_readable(get_input_exprs(residual)[0]) == "x2"


def test_nn_reuse_rejects_wrapped_leaf_with_wrong_input_arity():
    class _Base(torch.nn.Module):
        Nx_size = 2

    class _Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = _Base()

    one_input = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0", inputs=(Var(0),))
    two_input = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf0", inputs=(Var(0), Var(1)))

    assert not _is_module_compatible_with_atom(_Wrapped(), one_input)
    assert _is_module_compatible_with_atom(_Wrapped(), two_input)
