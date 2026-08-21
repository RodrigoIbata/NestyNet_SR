# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from nestynet_sr.sr_core import build_radial_r2_ast
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    MulNode,
    PowNode,
    Var,
    collect_nn_atoms,
)
from nestynet_sr.sr_core.carrier_units import (
    CARRIER_INTERNAL_UNITS_INVALID,
    CARRIER_UNITS_DEFERRED,
    OUTER_MAP_UNITS_INVALID,
    OUTER_MAP_UNITS_VALID,
    STAGEA_BUCKINGHAM_DEFERRED,
    UnitDecision,
    context_from_metadata,
    mark_inner_coordinate_metadata,
    precheck_carrier_units,
    stagea_provisional_unit_metadata,
    validate_outer_map_units,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, check_units_ast
from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.stagea_bridge import (
    _mark_promoted_carrier_roles,
    stageA_generalized_symmetry_proposals,
)
from nestynet_sr.sr_search.search import (
    _build_compound_candidate_ast,
    _stageA_compound_buckingham_reason,
)
from nestynet_sr.sr_search.factorized_search.engine.proposal_execution import (
    ProposalScoringState,
    score_external_candidate_expr,
)
from nestynet_sr.sr_search.factorized_search.expr_ast import (
    dims_eq,
    node_dims,
    node_size,
)
from nestynet_sr.sr_search.wrapper_policy import build_compound_z_variants


def _ctx(carrier_dim=(2.0, 0.0), target_dim=(1.0, 0.0)):
    meta = mark_inner_coordinate_metadata({}, source="generalized_symmetry")
    return context_from_metadata(
        meta,
        carrier_dim=carrier_dim,
        target_dim=target_dim,
    )


def _length_units_spec(*, n_vars=3):
    unit_system = UnitSystem(("L", "T"))
    length = unit_system.dim([1, 0])
    return UnitsSpec(
        unit_system=unit_system,
        x_dims=tuple(length for _ in range(n_vars)),
        y_dim=length,
    )


def _radial_nn_candidate(*, n_vars=3):
    carrier = build_radial_r2_ast(tuple(range(n_vars)))
    candidate = AtomNode(
        kind="nn",
        var_idxs=tuple(range(n_vars)),
        tag="radial_carrier",
        inputs=(carrier,),
    )
    return carrier, candidate


def _feynman_003_nn_candidate():
    delta_01 = AddNode(Var(0), MulNode(ConstNode(-1.0), Var(1)))
    delta_23 = AddNode(Var(2), MulNode(ConstNode(-1.0), Var(3)))
    carrier = AddNode(PowNode(delta_01, 2.0), PowNode(delta_23, 2.0))
    candidate = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="feynman_003_carrier",
        inputs=(carrier,),
    )
    return carrier, candidate


def test_certified_carrier_defers_target_dimension_until_outer_map():
    result = precheck_carrier_units(_ctx())

    assert result.decision is UnitDecision.DEFERRED_UNTIL_OUTER_MAP
    assert result.diagnostic == CARRIER_UNITS_DEFERRED


def test_unknown_carrier_dimension_fails_closed_before_outer_map():
    result = precheck_carrier_units(_ctx(carrier_dim=None))

    assert result.decision is UnitDecision.INVALID
    assert result.diagnostic == CARRIER_INTERNAL_UNITS_INVALID


def test_stagea_full_units_allow_valid_provisional_carrier_but_reject_illegal_ast():
    spec = _length_units_spec()
    _carrier, candidate = _radial_nn_candidate()
    valid = check_units_ast(candidate, spec)

    assert valid.ok
    assert collect_nn_atoms(candidate)

    illegal_carrier = AddNode(Var(0), Var(1))
    mixed_spec = UnitsSpec(
        unit_system=spec.unit_system,
        x_dims=(spec.unit_system.dim([1, 0]), spec.unit_system.dim([0, 1])),
        y_dim=spec.y_dim,
    )
    illegal_candidate = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        tag="illegal_carrier",
        inputs=(illegal_carrier,),
    )

    invalid = check_units_ast(illegal_candidate, mixed_spec)
    assert not invalid.ok


def test_stagea_bridge_fails_closed_when_promoted_carrier_units_are_invalid():
    unit_system = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dim([1, 0]), unit_system.dim([0, 1])),
        y_dim=unit_system.dim([1, 0]),
    )
    proposal = (
        (1, 1),
        AddNode(Var(0), Var(1)),
        1.0,
        None,
        {
            "kind": "gs_promoted_reduction",
            "source": "generalized_symmetry",
        },
    )

    (marked,) = _mark_promoted_carrier_roles([proposal], units_spec=spec)
    metadata = marked[4]

    assert metadata["candidate_role"] == "inner_coordinate"
    assert metadata["carrier_certified"] is False
    assert metadata["carrier_unit_decision"] == UnitDecision.INVALID.value
    assert metadata["carrier_unit_diagnostic"] == CARRIER_INTERNAL_UNITS_INVALID


def test_stagea_buckingham_deferral_requires_explicit_certified_carrier_role():
    spec = _length_units_spec()
    carrier, candidate = _radial_nn_candidate()
    common = {
        "current_ast": candidate,
        "atom": candidate,
        "z_expr": carrier,
        "kind": "gs_promoted_reduction",
        "extra_var_idxs": [],
        "extra_input_asts": None,
        "units_spec": spec,
        "enforce_units": True,
    }

    ordinary_reason = _stageA_compound_buckingham_reason(**common)
    source_only_reason = _stageA_compound_buckingham_reason(
        **common,
        candidate_meta={
            "kind": "gs_promoted_reduction",
            "source": "generalized_symmetry",
        },
    )
    metadata = mark_inner_coordinate_metadata(
        {"kind": "gs_promoted_reduction"},
        source="generalized_symmetry",
    )
    carrier_reason = _stageA_compound_buckingham_reason(
        **common,
        candidate_meta=metadata,
    )

    assert ordinary_reason is not None
    assert "destroys dimensionless freedom" in ordinary_reason
    assert source_only_reason == ordinary_reason
    assert carrier_reason is None
    assert metadata["stageA_buckingham_decision"] == STAGEA_BUCKINGHAM_DEFERRED
    assert metadata[STAGEA_BUCKINGHAM_DEFERRED] is True


def test_stagea_bridge_marks_shared_bank_entries_as_legal_carriers():
    rng = np.random.default_rng(7)
    x_vals = rng.uniform(-2.0, 2.0, size=(500, 3))
    radius = np.sqrt(np.sum(x_vals**2, axis=1))
    gradients = x_vals / radius[:, None]
    spec = _length_units_spec()
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy="augment",
        general_affine=True,
        general_affine_charts=("identity",),
        general_affine_promotion_noise_calibrated=True,
        pairwise_composition=True,
    )

    proposals, _diagnostics = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=x_vals,
        y_vals=radius,
        dydx_vals=gradients,
        cols=(0, 1, 2),
        cfg=cfg,
        units_spec=spec,
    )

    promoted = [
        proposal
        for proposal in proposals
        if proposal[4].get("kind") == "gs_promoted_reduction"
        and proposal[4].get("gs_coordinate_kind") == "radial"
    ]
    assert promoted
    metadata = promoted[0][4]
    assert metadata["candidate_role"] == "inner_coordinate"
    assert metadata["candidate_source"] == "generalized_symmetry"
    assert metadata["carrier_certified"] is True
    assert metadata["carrier_dim"] == [2.0, 0.0]
    assert metadata["target_dim"] == [1.0, 0.0]
    assert all(
        proposal[4].get("candidate_role") == "inner_coordinate"
        and proposal[4].get("candidate_source") == "generalized_symmetry"
        and proposal[4].get("carrier_certified") is True
        for proposal in proposals
    )


def test_stagea_radial_sqrt_wrapper_survives_full_units_and_buckingham_gate():
    spec = _length_units_spec(n_vars=4)
    carrier, candidate = _feynman_003_nn_candidate()
    metadata = mark_inner_coordinate_metadata(
        {
            "kind": "gs_promoted_reduction",
            "gs_coordinate_kind": "radial",
            "form": "r2",
            "allow_sqrt": True,
        },
        source="generalized_symmetry",
    )
    variants = dict(
        build_compound_z_variants(
            carrier,
            kind="gs_promoted_reduction",
            pattern=(1, 1, 1, 1),
            meta=metadata,
            search_hp=SimpleNamespace(),
            atom_var_idxs=(0, 1, 2, 3),
        )
    )

    sqrt_carrier = variants["sqrt"]
    sqrt_candidate = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="radial_carrier",
        inputs=(sqrt_carrier,),
    )
    assert check_units_ast(sqrt_candidate, spec).ok
    assert (
        _stageA_compound_buckingham_reason(
            current_ast=candidate,
            atom=candidate,
            z_expr=sqrt_carrier,
            kind="gs_promoted_reduction",
            extra_var_idxs=[],
            extra_input_asts=None,
            units_spec=spec,
            enforce_units=True,
            candidate_meta=metadata,
        )
        is None
    )
    assert metadata["stageA_buckingham_decision"] == STAGEA_BUCKINGHAM_DEFERRED


def test_stagea_provisional_carrier_marker_survives_in_nn_ast():
    carrier, candidate = _feynman_003_nn_candidate()
    metadata = mark_inner_coordinate_metadata(
        {"kind": "gs_promoted_reduction"},
        source="generalized_symmetry",
    )
    marker = stagea_provisional_unit_metadata(
        metadata,
        carrier_dim=(2.0, 0.0),
        target_dim=(1.0, 0.0),
    )

    provisional = _build_compound_candidate_ast(
        candidate,
        candidate,
        carrier,
        exponents=(1, 1, 1, 1),
        unit_handoff_metadata=marker,
    )

    (atom,) = collect_nn_atoms(provisional)
    retained = atom.kwargs["_unit_handoff"]
    assert retained["decision"] == UnitDecision.DEFERRED_UNTIL_OUTER_MAP.value
    assert retained["diagnostic"] == CARRIER_UNITS_DEFERRED
    assert retained["outer_map_pending"] is True
    assert retained["carrier_dim"] == [2.0, 0.0]
    assert retained["target_dim"] == [1.0, 0.0]


def test_power_half_maps_squared_length_carrier_to_length_target():
    result = validate_outer_map_units(
        _ctx(),
        {
            "kind": "power",
            "b": 0.5,
            "log_a": 0.0,
            "sgn_f": 1.0,
            "sgn_y": 1.0,
        },
    )

    assert result.ok
    assert result.diagnostic == OUTER_MAP_UNITS_VALID
    assert result.assembled_dim == (1.0, 0.0)


def test_identity_cannot_map_squared_length_carrier_to_length_target():
    result = validate_outer_map_units(_ctx(), {"kind": "identity"})

    assert not result.ok
    assert result.diagnostic == OUTER_MAP_UNITS_INVALID
    assert result.assembled_dim == (2.0, 0.0)


def test_malformed_outer_map_metadata_fails_closed():
    for mapping in ("power", [("kind", "power"), ("b", 0.5), ("bad",)]):
        result = validate_outer_map_units(_ctx(), mapping)
        assert not result.ok
        assert result.diagnostic == OUTER_MAP_UNITS_INVALID
        assert "malformed" in result.reason


def test_transcendental_outer_maps_reject_dimensionful_carrier():
    for family in ("sine", "exp", "log"):
        result = validate_outer_map_units(_ctx(), {"kind": family})
        assert not result.ok
        assert result.diagnostic == OUTER_MAP_UNITS_INVALID
        if family != "log":
            assert "dimensionless carrier" in result.reason


def test_raw_polynomial_dimension_action_is_checked_term_by_term():
    square = validate_outer_map_units(
        _ctx(carrier_dim=(1.0, 0.0), target_dim=(2.0, 0.0)),
        {"kind": "poly", "coeffs": [0.0, 0.0, 1.0], "mu": 0.0, "std": 1.0},
    )
    mixed = validate_outer_map_units(
        _ctx(carrier_dim=(1.0, 0.0), target_dim=(2.0, 0.0)),
        {"kind": "poly", "coeffs": [0.0, 1.0, 1.0], "mu": 0.0, "std": 1.0},
    )

    assert square.ok
    assert square.assembled_dim == (2.0, 0.0)
    assert not mixed.ok
    assert "incompatible dimensions" in mixed.reason


def test_unresolved_auxiliary_linear_head_fails_closed():
    result = validate_outer_map_units(
        _ctx(),
        {
            "kind": "power",
            "b": 0.5,
            "_lin_head": {"terms": [("var", 0)], "coeffs": [0.0, 1.0]},
        },
    )

    assert not result.ok
    assert "term dimensions are unresolved" in result.reason


def test_reported_zero_head_energy_cannot_bypass_head_unit_validation():
    for head in (
        {"terms": [], "coeffs": [1.0]},
        {"terms": [("var", 0)], "coeffs": [0.0]},
        "malformed",
        [1.0],
        {"terms": [], "coeffs": [float("nan")]},
        {"terms": [], "coeffs": [float("inf")]},
    ):
        result = validate_outer_map_units(
            _ctx(),
            {
                "kind": "power",
                "b": 0.5,
                "_lin_head": head,
                "_score_decomp": {"head_energy_frac": 0.0},
            },
        )

        assert not result.ok
        assert result.diagnostic == OUTER_MAP_UNITS_INVALID


def test_numerically_zero_auxiliary_head_does_not_block_valid_outer_map():
    result = validate_outer_map_units(
        _ctx(),
        {
            "kind": "power",
            "b": 0.5,
            "_lin_head": {
                "terms": [("var", 0)],
                "coeffs": [1.0e-15, -2.0e-16],
            },
            "_score_decomp": {"head_energy_frac": 0.0},
        },
    )

    assert result.ok
    assert result.diagnostic == OUTER_MAP_UNITS_VALID


def test_unit_matching_auxiliary_head_is_valid_but_unitful_bias_is_not():
    mapping = {
        "kind": "power",
        "b": 0.5,
        "_lin_head": {"terms": [("var", 0)], "coeffs": [0.0, 0.25]},
        "_score_decomp": {"head_energy_frac": 0.0},
    }
    valid = validate_outer_map_units(
        _ctx(),
        mapping,
        linear_head_term_dims=[(1.0, 0.0)],
    )
    invalid = validate_outer_map_units(
        _ctx(),
        {
            **mapping,
            "_lin_head": {"terms": [("var", 0)], "coeffs": [0.5, 0.25]},
        },
        linear_head_term_dims=[(1.0, 0.0)],
    )
    invalid_large_scale = validate_outer_map_units(
        _ctx(),
        {
            **mapping,
            "_lin_head": {
                "terms": [("var", 0)],
                "coeffs": [1.0, 1.0e20],
            },
        },
        linear_head_term_dims=[(1.0, 0.0)],
    )

    assert valid.ok
    assert not invalid.ok
    assert "undeclared unitful parameter" in invalid.reason
    assert not invalid_large_scale.ok
    assert "undeclared unitful parameter" in invalid_large_scale.reason


class _Archive:
    def __init__(self):
        self.updates = []

    def update(self, key, mse, expr, z, mapping, raw_mse=None):
        self.updates.append(
            {
                "key": key,
                "mse": mse,
                "expr": expr,
                "z": z,
                "mapping": mapping,
                "raw_mse": raw_mse,
            }
        )
        return True


def _score_external(
    expr,
    mapping,
    *,
    candidate_meta=None,
    var_dims=((1.0, 0.0),),
    target_dim=(1.0, 0.0),
):
    calls = []
    stats = {}
    archive = _Archive()
    x = torch.tensor([[1.0] * len(var_dims), [2.0] * len(var_dims)], dtype=torch.float64)
    y = torch.ones((2, 1), dtype=torch.float64)

    def scorer(node, *_args, **_kwargs):
        calls.append(node)
        z = torch.ones((2, 1), dtype=torch.float64)
        return 0.0, ("candidate", len(calls)), z, dict(mapping), node

    result = score_external_candidate_expr(
        expr,
        parent_raw_mse=None,
        stats=stats,
        route_name="test_route",
        candidate_meta=candidate_meta,
        state=ProposalScoringState(
            n_evaluated=0,
            best_raw_mse=float("inf"),
            best_raw_mse_struct=float("inf"),
            best_mse=float("inf"),
        ),
        dm=True,
        var_dims=var_dims,
        y_dims=target_dim,
        refine_cfg={},
        score_prescreen_stats={},
        closure_search_anchor_head_compare_enable=False,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        proj=None,
        fp_mode=None,
        q_scale=None,
        q_clip=None,
        poly_degree=4,
        refine_enable=False,
        refine_state=None,
        early_stop_mse=0.0,
        complexity_penalty=0.0,
        score_expr_fn=scorer,
        simplify_fn=lambda node: node,
        is_valid_node_fn=lambda node: isinstance(node, tuple),
        node_str_fn=str,
        node_dims_fn=node_dims,
        dims_eq_fn=dims_eq,
        node_size_fn=node_size,
        mapping_cost_fn=lambda _mapping: 0.0,
        mapping_is_structural_fn=lambda _mapping: True,
        arch=archive,
    )
    return result, calls, archive, stats


def test_fss_target_shaped_candidate_still_rejects_dimension_mismatch_pre_score():
    result, calls, archive, _stats = _score_external(
        ("sqr", ("var", 0)),
        {"kind": "power", "b": 0.5},
    )

    assert result is None
    assert calls == []
    assert archive.updates == []


def test_fss_certified_carrier_reaches_scorer_and_validates_assembled_power():
    metadata = mark_inner_coordinate_metadata({}, source="generalized_symmetry")
    result, calls, archive, stats = _score_external(
        ("sqr", ("var", 0)),
        {"kind": "power", "b": 0.5},
        candidate_meta=metadata,
    )

    assert len(calls) == 1
    assert result is not None
    assert len(archive.updates) == 1
    assert stats[CARRIER_UNITS_DEFERRED] == 1
    assert stats[OUTER_MAP_UNITS_VALID] == 1
    handoff = archive.updates[0]["mapping"]["_unit_handoff"]
    assert handoff["candidate_role"] == "inner_coordinate"
    assert handoff["candidate_source"] == "generalized_symmetry"
    assert handoff["carrier_dim"] == [2.0, 0.0]
    assert handoff["target_dim"] == [1.0, 0.0]
    assert handoff["map_family"] == "power"
    assert handoff["assembled_dim"] == [1.0, 0.0]


def test_fss_rejects_carrier_after_invalid_identity_outer_map():
    metadata = mark_inner_coordinate_metadata({}, source="generalized_symmetry")
    result, calls, archive, stats = _score_external(
        ("sqr", ("var", 0)),
        {"kind": "identity"},
        candidate_meta=metadata,
    )

    assert len(calls) == 1
    assert result is None
    assert archive.updates == []
    assert stats[OUTER_MAP_UNITS_INVALID] == 1


def test_fss_rejects_internally_invalid_carrier_before_outer_map():
    metadata = mark_inner_coordinate_metadata({}, source="generalized_symmetry")
    result, calls, archive, stats = _score_external(
        ("add", ("var", 0), ("var", 1)),
        {"kind": "power", "b": 0.5},
        candidate_meta=metadata,
        var_dims=((1.0, 0.0), (0.0, 1.0)),
    )

    assert result is None
    assert calls == []
    assert archive.updates == []
    assert stats[CARRIER_INTERNAL_UNITS_INVALID] == 1


def test_fss_noncarrier_route_keeps_existing_postscore_behavior():
    result, calls, archive, stats = _score_external(
        ("var", 0),
        {"kind": "sine"},
    )

    assert len(calls) == 1
    assert result is not None
    assert len(archive.updates) == 1
    assert OUTER_MAP_UNITS_VALID not in stats
    assert OUTER_MAP_UNITS_INVALID not in stats


def test_production_gs_to_fss_recovers_feynman_003_with_units():
    import dataclasses

    from nestynet_sr.sr_search.factorized_search.aif_closure_benchmark import (
        parse_equations_txt,
    )
    from nestynet_sr.sr_search.factorized_search.oracle_lab import (
        default_oracle_hyperparams,
        equation_spec_from_dict,
        run_oracle_equation,
    )

    raw = next(
        row
        for row in parse_equations_txt("data/equations.txt")
        if row["id"] == "feynman_003"
    )
    hp = dataclasses.replace(
        default_oracle_hyperparams(),
        n_iter=1,
        n_fit=400,
        n_probe=400,
        n_seeds=1,
        brute_depth=0,
        periodic_seed_enable=False,
    )

    report = run_oracle_equation(
        equation_spec_from_dict(raw),
        factorized_search_hp=hp,
        dtype=torch.float64,
        enforce_dims=True,
        verbose=False,
        gs_carrier_seed=True,
    )

    best = report["best"]
    assert best is not None
    assert float(best["mse"]) < 1.0e-20
    assert best["mapping_kind"] == "power"
    handoff = best["mapping"]["_unit_handoff"]
    assert handoff["diagnostic"] == OUTER_MAP_UNITS_VALID
    assert handoff["map_family"] == "power"
    assert handoff["carrier_dim"] == [2.0, 0.0, 0.0, 0.0, 0.0]
    assert handoff["target_dim"] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert report["gs_carrier_unit_stats"][CARRIER_UNITS_DEFERRED] == 1
    assert report["gs_carrier_unit_stats"][OUTER_MAP_UNITS_VALID] == 1
    assert (
        report["gs_carrier_unit_stats_by_seed"][0]["unit_handoff_events"][-1][
            "diagnostic"
        ]
        == OUTER_MAP_UNITS_VALID
    )
