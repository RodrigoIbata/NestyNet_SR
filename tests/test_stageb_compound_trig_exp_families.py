# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compound-aware trig/sinc/exp_poly candidate builders (pb119-class).

These families used to construct candidate atoms over raw ``target.var_idxs``
with no ``inputs=``, so on compound leaves like ``nn(x0/x1, x6)`` the units
checker (correctly) rejected them against raw variable dims.  The builders now
carry the target's input expressions and reason in input-position space; a raw
axis buried inside a compound input makes the variant skip, never fall back.
"""

from types import SimpleNamespace

import nestynet_sr.sr_search.stageB.rules  # noqa: F401  (import-order: break template_library cycle)
from nestynet_sr.sr_core.bridges import (
    AtomNode,
    MulNode,
    PowNode,
    Var,
    collect_all_atoms,
    get_input_exprs,
    has_nontrivial_input,
    is_trivial_input,
    trivial_input_position,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast
from nestynet_sr.sr_search._candidate_builders_univariate import (
    _make_multid_trig_pair_rewrite,
    _make_multid_trig_rewrite,
)
from nestynet_sr.sr_search.template_library import propose_exp_poly_from_log_hint


def _ratio(i, j):
    return MulNode(Var(i), PowNode(Var(j), -1.0))


def _pb119_leaf(tag="leaf4"):
    """nn(z=x0/x1, x6) with raw var union (0, 1, 6)."""
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1, 6),
        kwargs={},
        tag=tag,
        inputs=(_ratio(0, 1), Var(6)),
    )


def _pb119_units():
    us = UnitSystem(("L", "T"))
    x_dims = [us.dimless()] * 7
    x_dims[0] = us.dim([0, -1])
    x_dims[1] = us.dim([0, -1])
    return UnitsSpec(unit_system=us, x_dims=tuple(x_dims), y_dim=us.dimless())


def test_trivial_input_position_mapping_and_skip():
    leaf = _pb119_leaf()
    assert trivial_input_position(leaf, 6) == 1
    assert trivial_input_position(leaf, 0) is None  # buried inside z
    assert trivial_input_position(leaf, 1) is None
    simple = AtomNode(kind="nn", var_idxs=(2, 3), kwargs={})
    assert trivial_input_position(simple, 3) == 1


def test_exp_poly_log_hint_preserves_compound_inputs_and_units():
    target = _pb119_leaf()
    ctx = SimpleNamespace(state=SimpleNamespace(root=target))
    hint = SimpleNamespace(ok=True, best_name="log")
    cands = propose_exp_poly_from_log_hint(ctx, target, hint, degrees=(1, 2))
    assert cands
    spec = _pb119_units()
    for cand in cands:
        exp_atoms = [
            a
            for a in collect_all_atoms(cand.root)
            if str(a.kind).lower() == "exp_poly"
        ]
        assert len(exp_atoms) == 1
        atom = exp_atoms[0]
        assert has_nontrivial_input(atom)
        assert atom.n_in == 2
        inputs = get_input_exprs(atom)
        assert not is_trivial_input(inputs[0])
        # Independent clone: not the same objects as the target's inputs.
        assert inputs[0] is not target.inputs[0]
        # The raw-var spelling was units-rejected; the compound one must pass.
        result = check_units_ast(cand.root, spec)
        assert result.ok, result.reason


def test_exp_poly_log_hint_simple_target_unchanged():
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leafS")
    ctx = SimpleNamespace(state=SimpleNamespace(root=target))
    hint = SimpleNamespace(ok=True, best_name="log")
    cands = propose_exp_poly_from_log_hint(ctx, target, hint, degrees=(1,))
    assert cands
    for cand in cands:
        for atom in collect_all_atoms(cand.root):
            if str(atom.kind).lower() == "exp_poly":
                assert not has_nontrivial_input(atom)


def test_multid_trig_rewrite_compound_amp_consumes_remaining_inputs():
    target = _pb119_leaf()
    spec_trig = SimpleNamespace(axis=6, omega=2.0)
    new_root = _make_multid_trig_rewrite(
        root=target, target=target, spec=spec_trig, trig_kind="sin"
    )
    assert new_root is not None
    amp_atoms = [
        a
        for a in collect_all_atoms(new_root)
        if str(a.kind).lower() == "poly" and has_nontrivial_input(a)
    ]
    assert len(amp_atoms) == 1
    amp = amp_atoms[0]
    assert amp.n_in == 1
    assert not is_trivial_input(get_input_exprs(amp)[0])
    assert tuple(amp.var_idxs) == (0, 1)
    result = check_units_ast(new_root, _pb119_units())
    assert result.ok, result.reason


def test_multid_trig_rewrite_skips_axis_inside_compound():
    target = _pb119_leaf()
    spec_trig = SimpleNamespace(axis=0, omega=1.0)  # x0 lives inside z
    assert (
        _make_multid_trig_rewrite(
            root=target, target=target, spec=spec_trig, trig_kind="sin"
        )
        is None
    )


def test_multid_trig_pair_rewrite_compound_and_skip():
    # Three-input compound: z = x0/x1, plus trivial x5, x6.
    target = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 5, 6),
        kwargs={},
        tag="leafP",
        inputs=(_ratio(0, 1), Var(5), Var(6)),
    )
    new_root = _make_multid_trig_pair_rewrite(
        root=target,
        target=target,
        axis=6,
        partner_axis=5,
        trig_kind="cos",
        init_amp_coeffs=[1.0],
    )
    assert new_root is not None
    amp_atoms = [
        a
        for a in collect_all_atoms(new_root)
        if str(a.kind).lower() == "poly" and str(a.tag or "").endswith("_amp")
    ]
    assert len(amp_atoms) == 1
    assert has_nontrivial_input(amp_atoms[0])
    assert amp_atoms[0].n_in == 1
    # Partner buried inside the compound -> skip.
    assert (
        _make_multid_trig_pair_rewrite(
            root=target, target=target, axis=6, partner_axis=0, trig_kind="cos"
        )
        is None
    )


def test_sinc_family_compound_multi_polys_carry_independent_input_clones():
    from nestynet_sr.sr_search.template_library import propose_sinc_family

    target = _pb119_leaf()
    ctx = SimpleNamespace(state=SimpleNamespace(root=target), lm_hp=SimpleNamespace())
    trig_spec = SimpleNamespace(axis=6, omega=2.0)
    cands = propose_sinc_family(ctx, target, trig_spec, degree_arg=2, p=2)
    assert cands, "compound-multi sinc produced no candidates"
    spec = _pb119_units()
    for cand in cands:
        polys = [
            a
            for a in collect_all_atoms(cand.root)
            if str(a.kind).lower() == "poly" and has_nontrivial_input(a)
        ]
        # sin-argument and denominator polynomials both consume the inputs.
        assert len(polys) == 2
        for atom in polys:
            assert atom.n_in == 2
        # Independent clones (AST must stay a tree, not a DAG).
        assert get_input_exprs(polys[0])[0] is not get_input_exprs(polys[1])[0]
        result = check_units_ast(cand.root, spec)
        assert result.ok, result.reason


def test_sinc_family_skips_when_axis_inside_compound():
    from nestynet_sr.sr_search.template_library import propose_sinc_family

    target = _pb119_leaf()
    ctx = SimpleNamespace(state=SimpleNamespace(root=target), lm_hp=SimpleNamespace())
    trig_spec = SimpleNamespace(axis=0, omega=1.0)  # buried inside z
    assert propose_sinc_family(ctx, target, trig_spec, degree_arg=2, p=2) == []
