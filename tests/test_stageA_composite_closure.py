# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_core.bridges import AtomNode, MulNode, SinNode, Var, ast_to_human_readable, get_input_exprs
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast
from nestynet_sr.sr_search.search import (
    _build_compound_candidate_ast,
    _build_stageA_composite_closure_ast,
    _stageA_composite_closure_applicable,
)


def _mixed_z():
    return MulNode(MulNode(Var(2), Var(3)), SinNode(Var(4)))


def test_stageA_composite_closure_builds_visible_scale_times_mixed_z():
    atom = AtomNode(kind="nn", var_idxs=(2, 3, 4), tag="leaf")
    cand, reason = _build_stageA_composite_closure_ast(atom, atom, _mixed_z())

    assert reason is None
    assert cand is not None
    human = ast_to_human_readable(cand)
    assert "scale()" in human
    assert "sin(x4)" in human
    assert "NN" not in human


def test_stageA_composite_closure_accepts_dimensionally_valid_mixed_z():
    us = UnitSystem(("L", "T"))
    atom = AtomNode(kind="nn", var_idxs=(2, 3, 4), tag="leaf")
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(
            us.dim({}),
            us.dim({}),
            us.dim({"L": 1}),
            us.dim({"T": 1}),
            us.dim({}),
        ),
        y_dim=us.dim({"L": 1, "T": 1}),
    )

    cand, reason = _build_stageA_composite_closure_ast(
        atom,
        atom,
        _mixed_z(),
        units_spec=spec,
        enforce_units=True,
    )

    assert reason is None
    assert cand is not None
    assert check_units_ast(cand, spec).ok


def test_stageA_composite_closure_rejects_unitful_trig_argument():
    us = UnitSystem(("L", "T"))
    atom = AtomNode(kind="nn", var_idxs=(2, 3, 4), tag="leaf")
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(
            us.dim({}),
            us.dim({}),
            us.dim({"L": 1}),
            us.dim({"T": 1}),
            us.dim({"L": 1}),
        ),
        y_dim=us.dim({"L": 1, "T": 1}),
    )

    cand, reason = _build_stageA_composite_closure_ast(
        atom,
        atom,
        _mixed_z(),
        units_spec=spec,
        enforce_units=True,
    )

    assert cand is None
    assert reason is not None


def test_stageA_composite_closure_uses_declared_unitful_scalar_when_needed():
    us = UnitSystem(("L", "T", "M"))
    atom = AtomNode(kind="nn", var_idxs=(2, 3, 4), tag="leaf")
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(
            us.dim({}),
            us.dim({}),
            us.dim({"L": 1}),
            us.dim({"T": 1}),
            us.dim({}),
        ),
        y_dim=us.dim({"L": 1, "T": 1, "M": 1}),
        free_const_dims={"mass_scale": us.dim({"M": 1})},
    )

    cand, reason = _build_stageA_composite_closure_ast(
        atom,
        atom,
        _mixed_z(),
        units_spec=spec,
        enforce_units=True,
    )

    assert reason is None
    assert cand is not None
    assert "mass_scale" in ast_to_human_readable(cand)
    assert check_units_ast(cand, spec).ok


def test_mixed_pattern_fallback_keeps_trig_axis_out_of_extras():
    atom = AtomNode(kind="nn", var_idxs=(2, 3, 4), tag="leaf")
    cand = _build_compound_candidate_ast(atom, atom, _mixed_z(), exponents=(1, 1, "trig"))

    assert isinstance(cand, AtomNode)
    assert len(get_input_exprs(cand)) == 1
    assert "sin(x4)" in ast_to_human_readable(cand)


def test_stageA_composite_closure_applicability_is_full_mixed_only():
    assert _stageA_composite_closure_applicable(
        kind="mixed",
        extra_var_idxs=[],
        extra_input_asts=None,
        prefactor_exps=None,
    )
    assert not _stageA_composite_closure_applicable(
        kind="mixed",
        extra_var_idxs=[1],
        extra_input_asts=None,
        prefactor_exps=None,
    )
    assert not _stageA_composite_closure_applicable(
        kind="monomial",
        extra_var_idxs=[],
        extra_input_asts=None,
        prefactor_exps=None,
    )
