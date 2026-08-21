# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    FixedConst,
    FreeConst,
    MulNode,
    PowNode,
    Var,
)
from nestynet_sr.sr_core.units import (
    UnitSystem,
    UnitsSpec,
    check_units_ast,
    eval_analytic_expr_dim,
)
from nestynet_sr.sr_search.stageB.engine import Candidate, StageBContext, _input_basis_dims_for_atom
from nestynet_sr.sr_search.stageB.rules import (
    _effective_input_dims_for_atom,
    _prepare_univariate_units_probe,
)


def _spec(us, x_dims, y_dim):
    return UnitsSpec(
        unit_system=us,
        x_dims=tuple(x_dims),
        y_dim=y_dim,
    )


def _stageb_ctx_for_units(spec):
    return StageBContext(
        state=SimpleNamespace(root=ConstNode(0.0)),
        train_loader=None,
        val_loader=None,
        lm_hp=SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-8,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        units_spec=spec,
        enforce_units=True,
        verbose=False,
    )


def test_stageb_precheck_rejects_candidate_with_unreachable_nn_leaf_units():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    L_over_T = us.dim({"L": 1, "T": -1})
    root = AddNode(
        AtomNode(kind="nn", var_idxs=(0,), tag="leaf_L_only"),
        AtomNode(kind="nn", var_idxs=(1, 2), tag="leaf_velocity_only"),
    )
    spec = _spec(us, [L, L_over_T, L_over_T], T)

    whole_ast = check_units_ast(root, spec)
    assert whole_ast.ok, whole_ast.reason

    pre = _stageb_ctx_for_units(spec).precheck_candidate(
        "nn_leaf_separability",
        Candidate("gauge_add_split", root),
    )

    assert not pre.ok
    assert "nn-output-unreachable" in pre.reason
    assert "leaf_L_only" in pre.reason


def test_stageb_precheck_allows_reachable_nn_leaf_units():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    root = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf_LT")
    spec = _spec(us, [L, T], us.dim({"L": 1, "T": -1}))

    pre = _stageb_ctx_for_units(spec).precheck_candidate(
        "nn_leaf_separability",
        Candidate("gauge_mul_split", root),
    )

    assert pre.ok


def test_effective_input_dims_for_compound_ratio_are_dimless():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()

    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="nn_ratio",
        inputs=(z_expr,),
    )
    spec = _spec(us, [L, L], dimless)

    assert _effective_input_dims_for_atom(atom, spec) == [tuple(dimless)]


def test_effective_input_dims_include_all_compound_inputs():
    us = UnitSystem(("L", "T", "M"))
    LT = us.dim({"L": 1, "T": 1})
    dimless = us.dimless()

    p_expr = MulNode(Var(0), Var(1))
    q_expr = MulNode(Var(2), Var(3))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="nn_two_compounds",
        inputs=(p_expr, q_expr),
    )
    spec = _spec(
        us,
        [
            us.dim({"L": 1}),
            us.dim({"T": 1}),
            us.dim({"L": 1}),
            us.dim({"T": 1}),
        ],
        dimless,
    )

    assert _effective_input_dims_for_atom(atom, spec) == [tuple(LT), tuple(LT)]


def test_effective_input_dims_use_the_same_declared_fixed_constant_path_for_all_units():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    length = us.dim({"L": 1})
    expr = MulNode(FixedConst("ell", value=2.0), Var(0))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0,),
        tag="nn_fixed_constant_input",
        inputs=(expr,),
    )

    for declared_dim in (dimless, length):
        spec = UnitsSpec(
            unit_system=us,
            x_dims=(dimless,),
            y_dim=declared_dim,
            fixed_const_dims={"ell": declared_dim},
            fixed_const_values={"ell": 2.0},
        )

        assert eval_analytic_expr_dim(
            expr,
            spec.x_dims,
            fixed_const_dims=spec.fixed_const_dims,
        ) == tuple(declared_dim)
        assert _effective_input_dims_for_atom(atom, spec) == [tuple(declared_dim)]

    undeclared = UnitsSpec(unit_system=us, x_dims=(dimless,), y_dim=dimless)
    assert _effective_input_dims_for_atom(atom, undeclared) == []


def test_effective_input_dims_do_not_advertise_uncompiled_free_constant_inputs():
    us = UnitSystem(("L", "T", "M"))
    dimless = us.dimless()
    length = us.dim({"L": 1})
    expr = MulNode(FreeConst("a", init=2.0), Var(0))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0,),
        tag="nn_free_constant_input",
        inputs=(expr,),
    )
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(dimless,),
        y_dim=length,
        free_const_dims={"a": length},
    )

    # Pure symbolic inference can type the declaration.
    assert eval_analytic_expr_dim(
        expr,
        spec.x_dims,
        free_const_dims=spec.free_const_dims,
    ) == tuple(length)
    # Rational rule preparation must still fail closed because eval_inputs()
    # cannot evaluate a fitted FreeConst inside AtomNode.inputs yet.
    assert _effective_input_dims_for_atom(atom, spec) == []


def test_nonsense_units_basis_uses_all_compound_inputs():
    us = UnitSystem(("L", "T", "M"))
    LT = us.dim({"L": 1, "T": 1})
    M2 = us.dim({"M": 2})
    dimless = us.dimless()

    p_expr = MulNode(Var(0), Var(1))
    q_expr = MulNode(Var(2), Var(3))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="nn_two_compounds",
        inputs=(p_expr, q_expr),
    )
    spec = _spec(
        us,
        [
            us.dim({"L": 1}),
            us.dim({"T": 1}),
            us.dim({"M": 1}),
            us.dim({"M": 1}),
        ],
        dimless,
    )

    basis = [tuple(dim) for dim in _input_basis_dims_for_atom(atom, spec)]

    assert tuple(LT) in basis
    assert tuple(M2) in basis


def test_prepare_univariate_units_probe_uses_compound_input_dim():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()

    z_expr = MulNode(Var(0), PowNode(Var(1), -1.0))
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="nn_ratio",
        inputs=(z_expr,),
    )
    spec = _spec(us, [L, L], dimless)
    ctx = SimpleNamespace(
        enforce_units=True,
        infer_target_dim=lambda _target: dimless,
    )

    inv_homo, target_dim, x_dims = _prepare_univariate_units_probe(ctx, atom, spec)

    assert inv_homo is False
    assert target_dim == tuple(dimless)
    assert x_dims == [tuple(dimless)]


def test_prepare_univariate_units_probe_keeps_simple_atom_dims():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()

    atom = AtomNode(kind="nn", var_idxs=(0,), tag="nn_simple")
    spec = _spec(us, [L], dimless)
    ctx = SimpleNamespace(
        enforce_units=True,
        infer_target_dim=lambda _target: dimless,
    )

    inv_homo, target_dim, x_dims = _prepare_univariate_units_probe(ctx, atom, spec)

    assert inv_homo is True
    assert target_dim == tuple(dimless)
    assert x_dims == [tuple(L)]


def test_prepare_univariate_units_probe_accepts_simple_reciprocal_coordinate_dim():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    inv_L = us.dim({"L": -1})

    atom = AtomNode(
        kind="nn",
        var_idxs=(0,),
        tag="nn_simple_recip",
        inputs=(PowNode(Var(0), -1.0),),
    )
    spec = _spec(us, [L], inv_L)
    ctx = SimpleNamespace(
        enforce_units=True,
        infer_target_dim=lambda _target: inv_L,
    )

    inv_homo, target_dim, x_dims = _prepare_univariate_units_probe(ctx, atom, spec)

    assert inv_homo is True
    assert target_dim == tuple(inv_L)
    assert x_dims == [tuple(inv_L)]


# ---------------------------------------------------------------------------
# Multi-input compound atoms: every effective input must be validated
# (regression tests for the first-input-only validation bug).
# ---------------------------------------------------------------------------


def _ratio(i, j):
    return MulNode(Var(i), PowNode(Var(j), -1.0))


def test_poly_compound_second_input_incommensurate_is_rejected():
    us = UnitSystem(("L", "T"))
    spec = _spec(
        us,
        x_dims=[us.dim([0, -1]), us.dim([0, -1]), us.dim([1, 0])],
        y_dim=us.dimless(),
    )
    # inputs[0] = x0/x1 (dimensionless), inputs[1] = x2 (L): a shared-monomial
    # poly over incommensurate inputs is dimensional nonsense.  The old code
    # validated only inputs[0] and passed this.
    atom = AtomNode(
        kind="poly",
        var_idxs=(0, 1, 2),
        kwargs={"degree": 2},
        inputs=(_ratio(0, 1), Var(2)),
    )
    result = check_units_ast(atom, spec)
    assert not result.ok
    assert "commensurate" in str(result.reason)


def test_poly_compound_all_dimless_inputs_pass():
    us = UnitSystem(("L", "T"))
    spec = _spec(
        us,
        x_dims=[us.dim([0, -1]), us.dim([0, -1]), us.dim([1, 0]), us.dim([1, 0])],
        y_dim=us.dimless(),
    )
    atom = AtomNode(
        kind="poly",
        var_idxs=(0, 1, 2, 3),
        kwargs={"degree": 2},
        inputs=(_ratio(0, 1), _ratio(2, 3)),
    )
    result = check_units_ast(atom, spec)
    assert result.ok


def test_dimless_kind_compound_second_input_unitful_is_rejected():
    us = UnitSystem(("L", "T"))
    spec = _spec(
        us,
        x_dims=[us.dim([0, -1]), us.dim([0, -1]), us.dim([1, 0])],
        y_dim=us.dimless(),
    )
    kind = "exp_poly"
    assert kind in spec.dimless_atom_kinds
    atom = AtomNode(
        kind=kind,
        var_idxs=(0, 1, 2),
        kwargs={},
        inputs=(_ratio(0, 1), Var(2)),
    )
    result = check_units_ast(atom, spec)
    assert not result.ok


def test_dimless_kind_compound_all_dimless_inputs_pass():
    us = UnitSystem(("L", "T"))
    spec = _spec(
        us,
        x_dims=[us.dim([0, -1]), us.dim([0, -1]), us.dim([1, 0]), us.dim([1, 0])],
        y_dim=us.dimless(),
    )
    kind = "exp_poly"
    assert kind in spec.dimless_atom_kinds
    atom = AtomNode(
        kind=kind,
        var_idxs=(0, 1, 2, 3),
        kwargs={},
        inputs=(_ratio(0, 1), _ratio(2, 3)),
    )
    result = check_units_ast(atom, spec)
    assert result.ok
