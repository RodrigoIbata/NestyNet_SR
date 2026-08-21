# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

from nestynet_sr.sr_core.bridges import AtomNode, PowNode, Var
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast, eval_analytic_expr_dim
from nestynet_sr.sr_search.compound_proposals import (
    CompoundProposal,
    build_metric_distance_compound_proposals,
    proposal_signature,
    stageA_tuple_from_proposal,
    stageB_meta_from_proposal,
)
from nestynet_sr.sr_search.metric_distance import (
    build_metric_distance_proposals,
    metric_stageA_tuple,
)
from nestynet_sr.sr_search.search import (
    _compound_best_proposal_confidence,
    _compound_proposal_brief,
    _stageA_forced_monomial_expr_from_units,
    _stageA_forced_monomial_reason,
    _stageA_composite_closure_applicable,
    _stageA_composite_closure_skip_reason,
)
from nestynet_sr.sr_search.stageB.rules import RuleMetricDistance


def _spec(us, x_dims, y_dim):
    return UnitsSpec(unit_system=us, x_dims=tuple(x_dims), y_dim=y_dim)


def test_law_of_cosines_proposals_are_units_aware():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, dimless], us.dim({"L": -1}))
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")

    props = build_metric_distance_proposals(atom, units_spec=spec)

    inv = [p for p in props if p.family == "lawcos" and p.wrapper == "inv_sqrt_q"]
    assert inv
    assert inv[0].z_dim == tuple(us.dim({"L": -1}))


def test_shared_metric_builder_emits_compound_proposals_directly():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, dimless], L)
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")

    props = build_metric_distance_compound_proposals(atom, units_spec=spec, wrappers=("q",))
    lawcos = [p for p in props if p.family == "lawcos" and p.wrapper == "q"]

    assert lawcos
    assert isinstance(lawcos[0], CompoundProposal)
    assert lawcos[0].kind == "metric_distance"
    assert lawcos[0].meta["metric_family"] == "lawcos"
    assert lawcos[0].meta["metric_wrapper"] == "q"
    assert lawcos[0].base_dim == tuple(us.dim({"L": 2}))


def test_gram3_pairangles_metric_emits_unit_valid_proposals():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, L, dimless, dimless, dimless], us.dim({"L": 2}))
    inputs = (Var(0), Var(1), Var(2), Var(3), Var(4), Var(5))

    props = build_metric_distance_compound_proposals(
        inputs,
        units_spec=spec,
        wrappers=("q",),
        max_proposals=32,
    )
    gram = [p for p in props if p.family.startswith("gram3_pairangles_")]

    assert len(gram) == 4
    assert {p.meta["signs"] for p in gram} == {
        (1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, -1.0, -1.0),
    }
    assert all(p.wrapper == "q" for p in gram)
    assert all(p.base_dim == tuple(us.dim({"L": 2})) for p in gram)
    assert all(p.z_dim == tuple(us.dim({"L": 2})) for p in gram)
    assert all(p.consumed_pattern == (1, 1, 1, 1, 1, 1) for p in gram)
    assert all(p.meta["gram_rank"] == 3 for p in gram)
    assert all(p.meta["angle_mode"] == "pairwise" for p in gram)
    assert all(check_units_ast(p.z_ast, spec).ok for p in gram)


def test_gram3_pairangles_rejects_mismatched_radius_units():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    T = us.dim({"T": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, T, L, dimless, dimless, dimless], us.dim({"L": 2}))
    inputs = (Var(0), Var(1), Var(2), Var(3), Var(4), Var(5))

    props = build_metric_distance_compound_proposals(
        inputs,
        units_spec=spec,
        wrappers=("q",),
        max_proposals=32,
    )

    assert not [p for p in props if p.family.startswith("gram3_pairangles_")]


def test_gram3_pairangles_filters_wrapper_alias_duplicates():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, L, dimless, dimless, dimless], us.dim({"L": 2}))
    inputs = (Var(0), Var(1), Var(2), Var(3), Var(4), Var(5))

    props = build_metric_distance_compound_proposals(
        inputs,
        units_spec=spec,
        wrappers=("q", "identity"),
        max_proposals=32,
    )
    gram = [p for p in props if p.family.startswith("gram3_pairangles_")]
    signatures = [proposal_signature(p) for p in gram]

    assert gram
    assert len(signatures) == len(set(signatures))
    assert all(p.wrapper == "q" for p in gram)


def test_squared_radius_lawcos_matches_output_units():
    us = UnitSystem(("L", "T", "M"))
    E = us.dim({"M": 1, "T": -2})
    dimless = us.dimless()
    spec = _spec(us, [E, E, dimless], E)
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")

    props = build_metric_distance_proposals(atom, units_spec=spec)
    direct = [p for p in props if p.family == "lawcos_sq_plus" and p.wrapper == "q"]

    assert direct
    assert direct[0].z_dim == tuple(E)


def test_law_of_cosines_rejects_unitful_angle():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    spec = _spec(us, [L, L, L], us.dimless())
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")

    props = build_metric_distance_proposals(atom, units_spec=spec)

    assert not [p for p in props if p.family == "lawcos"]


def test_cartesian_difference_metric_builds_stageA_tuple():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    spec = _spec(us, [L, L, L, L], L)
    inputs = (Var(0), Var(1), Var(2), Var(3))

    props = build_metric_distance_proposals(inputs, units_spec=spec, include_polar=False)
    cart = [p for p in props if p.family == "cartdist" and p.wrapper == "sqrt_q"]

    assert cart
    stagea_tuple = metric_stageA_tuple(cart[0])
    assert stagea_tuple[0] == (1, 1, 1, 1)
    assert stagea_tuple[4]["kind"] == "metric_distance"
    assert stagea_tuple[4]["metric_wrapper"] == "sqrt_q"


def test_metric_proposal_converts_to_shared_compound_proposal():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, dimless], L)
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")

    prop = [
        p
        for p in build_metric_distance_proposals(atom, units_spec=spec, wrappers=("q",))
        if p.family == "lawcos"
    ][0]
    shared = prop.to_compound_proposal()

    assert shared.kind == "metric_distance"
    assert shared.family == prop.family
    assert shared.wrapper == prop.wrapper
    assert shared.consumed_pattern == prop.pattern
    assert shared.confidence == prop.confidence
    assert shared.base_dim == prop.q_dim
    assert shared.z_dim == prop.z_dim
    sig = proposal_signature(shared)
    assert sig == proposal_signature(shared)
    assert sig[0] == "metric_distance"
    assert sig[1] == prop.family
    assert sig[2] == "z"


def test_metric_stage_adapters_preserve_legacy_metadata():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, dimless], L)
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")
    prop = build_metric_distance_proposals(atom, units_spec=spec, wrappers=("q",))[0]
    shared = prop.to_compound_proposal()

    legacy_tuple = metric_stageA_tuple(prop)
    shared_tuple = stageA_tuple_from_proposal(shared)
    stageb_meta = stageB_meta_from_proposal(shared, pattern="metric_distance")

    assert legacy_tuple[0] == shared_tuple[0] == prop.pattern
    assert legacy_tuple[2] == shared_tuple[2] == prop.confidence
    assert legacy_tuple[4]["kind"] == "metric_distance"
    assert legacy_tuple[4]["metric_family"] == prop.family
    assert legacy_tuple[4]["metric_wrapper"] == prop.wrapper
    assert legacy_tuple[4]["q_dim"] == prop.q_dim
    assert legacy_tuple[4]["z_dim"] == prop.z_dim
    assert stageb_meta["pattern"] == "metric_distance"
    assert stageb_meta["pattern_family"] == "metric_distance"
    assert stageb_meta["metric_family"] == prop.family


def test_metric_compatibility_builder_matches_shared_builder_order():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, dimless], L)
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")

    legacy = build_metric_distance_proposals(atom, units_spec=spec, wrappers=("q",))
    shared = build_metric_distance_compound_proposals(atom, units_spec=spec, wrappers=("q",))

    assert [p.label for p in legacy] == [p.label for p in shared]
    assert [p.family for p in legacy] == [p.family for p in shared]
    assert [p.wrapper for p in legacy] == [p.wrapper for p in shared]
    assert [p.pattern for p in legacy] == [p.consumed_pattern for p in shared]
    assert [p.z_dim for p in legacy] == [p.z_dim for p in shared]


def test_shared_metric_builder_filters_wrapper_alias_duplicates():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, dimless], L)
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")

    props = build_metric_distance_compound_proposals(
        atom,
        units_spec=spec,
        wrappers=("q", "identity"),
    )
    signatures = [proposal_signature(p) for p in props]

    assert len(signatures) == len(set(signatures))
    assert all(p.wrapper == "q" for p in props)


def test_stageB_metric_distance_proposes_unit_valid_terminal_closure():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    inv_L = us.dim({"L": -1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, dimless], inv_L)
    root = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")
    ctx = SimpleNamespace(
        state=SimpleNamespace(root=root),
        enforce_units=True,
        units_spec=spec,
        verbose=False,
        infer_target_dim=lambda _target: inv_L,
    )

    cands = RuleMetricDistance().propose(ctx, root)
    inv_cands = [c for c in cands if c.meta.get("metric_wrapper") == "inv_sqrt_q"]

    assert inv_cands
    units = check_units_ast(inv_cands[0].root, spec)
    assert units.ok, units.reason
    assert inv_cands[0].meta["pattern_family"] == "metric_distance"
    assert inv_cands[0].meta["compound_proposal_signature"][0] == "metric_distance"
    assert inv_cands[0].meta["compound_proposal_signature"][2] == "inv_sqrt_z"


def test_stageA_compound_gate_uses_best_proposal_confidence():
    proposals = [
        ((1, 1, "trig"), Var(0), 0.65, None, {"kind": "mixed"}),
        ((1, 1, 1), Var(1), 0.98, None, {"kind": "metric_distance"}),
    ]

    assert _compound_best_proposal_confidence(proposals) == 0.98


def test_stageA_metric_distance_can_try_visible_analytic_closure():
    assert _stageA_composite_closure_applicable(
        kind="metric_distance",
        extra_var_idxs=[],
        extra_input_asts=[],
        prefactor_exps=None,
    )


def test_stageA_closure_skip_reason_reports_preserved_extras():
    reason = _stageA_composite_closure_skip_reason(
        kind="metric_distance",
        extra_var_idxs=[2],
        extra_input_asts=[],
        prefactor_exps=None,
    )

    assert "preserved extras" in reason
    assert "raw extras=[2]" in reason


def test_stageA_metric_proposal_brief_includes_family_and_wrapper():
    brief = _compound_proposal_brief(
        (1, 1, 1),
        Var(0),
        0.98,
        {
            "kind": "metric_distance",
            "metric_family": "lawcos_sq_plus",
            "metric_wrapper": "q",
        },
    )

    assert "kind=metric_distance" in brief
    assert "family=lawcos_sq_plus" in brief
    assert "wrapper=q" in brief
    assert "conf=0.980" in brief


def test_buckingham_forced_monomial_builds_sqrt_lawcos_closure():
    us = UnitSystem(("L", "T", "M"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L, dimless], L)
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="leaf")
    q_prop = [
        p
        for p in build_metric_distance_proposals(
            atom,
            units_spec=spec,
            wrappers=("q",),
        )
        if p.family == "lawcos" and p.wrapper == "q"
    ][0]

    forced, powers, reason = _stageA_forced_monomial_expr_from_units(
        current_ast=atom,
        atom=atom,
        z_expr=q_prop.z_ast,
        extra_var_idxs=[],
        extra_input_asts=[],
        units_spec=spec,
        enforce_units=True,
    )

    assert reason == ""
    assert powers == (1 / 2,)
    assert isinstance(forced, PowNode)
    assert forced.exponent == 0.5
    assert eval_analytic_expr_dim(forced, spec.x_dims) == spec.y_phi_dim


def test_buckingham_forced_monomial_reason_is_specific():
    assert _stageA_forced_monomial_reason(
        "compound destroys dimensionless freedom: 3 -> 0 (need >= 1); forces monomial"
    )
    assert not _stageA_forced_monomial_reason(
        "compound reduces dimensional rank: remaining variables cannot span output dimension"
    )
