# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_search.factorized_search.subproblem_tests import (
    build_expanded_family_evidence_bundle,
    build_named_outer_family_evidence,
    build_square_family_evidence,
    default_outer_family_battery_specs,
    family_evidence_should_run,
    family_evidence_status,
    normalize_family_battery_mode,
)


def test_default_outer_family_battery_specs_cover_expected_families():
    specs = default_outer_family_battery_specs(
        periodic_min_improvement_ratio=0.5,
        periodic_precheck_max_seeds=6,
        default_min_improvement_ratio=0.8,
    )

    assert [spec.name for spec in specs] == ["periodic", "exp", "power", "rational"]
    assert specs[0].precheck_max_seeds == 6
    assert specs[1].min_improvement_ratio == 0.8


def test_named_outer_family_evidence_marks_triggered_runs():
    evidence = build_named_outer_family_evidence(
        "exp",
        recursive_enable=True,
        wrappers_left=1,
        flat_rows_present=True,
        target_dim_ok=True,
        best_flat_probe_mse=10.0,
        seed_nodes=[("var", 0)],
        min_improvement_ratio=0.5,
        best_probe_mse=3.0,
    )

    assert family_evidence_status(evidence, "exp") == "triggered"
    assert family_evidence_should_run(evidence, "exp") is True
    assert evidence.hard_constraints["candidate_count"] == 1
    assert evidence.hard_constraints["should_run"] is True
    assert evidence.metadata["improvement_ratio"] == 0.3
    assert evidence.family_scores["exp"] > 1.0


def test_named_outer_family_evidence_records_dimension_gate():
    evidence = build_named_outer_family_evidence(
        "rational",
        recursive_enable=True,
        wrappers_left=1,
        flat_rows_present=True,
        target_dim_ok=False,
        best_flat_probe_mse=4.0,
        seed_nodes=[("var", 0)],
        min_improvement_ratio=0.8,
        best_probe_mse=1.0,
    )

    assert family_evidence_status(evidence, "rational") == "nondimensionless_target"
    assert family_evidence_should_run(evidence, "rational") is False
    assert evidence.hard_constraints["target_dim_ok"] is False
    assert evidence.hard_constraints["dimensionless"]["required"] is True
    assert evidence.hard_constraints["dimensionless"]["target_dim_ok"] is False


def test_named_outer_family_evidence_exports_canonical_context_and_regime_sections():
    evidence = build_named_outer_family_evidence(
        "exp",
        recursive_enable=True,
        wrappers_left=2,
        flat_rows_present=True,
        target_dim_ok=True,
        best_flat_probe_mse=5.0,
        seed_nodes=[("var", 0)],
        min_improvement_ratio=0.8,
        best_probe_mse=2.0,
        target_dim=("L",),
        active_vars=(0, 2),
        recursion_level=1,
        direction="outside_in",
        target_mode="identity",
        target_mapping_kind="affine",
        regime_metadata={"dataset_ids": ["r0", "r1"]},
    )

    assert evidence.hard_constraints["context"]["target_dim"] == ("L",)
    assert evidence.hard_constraints["context"]["active_vars"] == (0, 2)
    assert evidence.hard_constraints["context"]["wrappers_left"] == 2
    assert evidence.hard_constraints["context"]["direction"] == "outside_in"
    assert evidence.hard_constraints["context"]["target_mode"] == "identity"
    assert evidence.hard_constraints["context"]["target_mapping_kind"] == "affine"
    assert evidence.hard_constraints["regime"]["dataset_ids"] == ["r0", "r1"]


def test_square_family_evidence_wraps_legacy_square_decision():
    evidence = build_square_family_evidence(
        proposal_name="square",
        proposal_score=0.02,
        proposal_improvement=18.0,
        proposal_details={"axis_stats": [{"axis": 0}]},
        prefer=True,
        diagnostics={"reason": "prefer_square_multi_axis", "num_good_axes": 2},
    )

    assert family_evidence_status(evidence, "square") == "triggered"
    assert family_evidence_should_run(evidence, "square") is False
    assert evidence.hard_constraints["prefer"] is True
    assert evidence.metadata["decision_diagnostics"]["reason"] == "prefer_square_multi_axis"
    assert evidence.family_scores["square"] == 18.0


def test_family_battery_mode_normalization_and_expanded_bundle_defaults():
    assert normalize_family_battery_mode(None) == "outer"
    assert normalize_family_battery_mode(" expanded ") == "expanded"
    assert normalize_family_battery_mode("unknown") == "outer"

    bundle = build_expanded_family_evidence_bundle(
        x_fit=torch.tensor([[0.0]], dtype=torch.float64),
        t_fit=torch.tensor([[1.0]], dtype=torch.float64),
    )

    assert set(bundle.keys()) == {
        "symmetry",
        "separability",
        "low_rank_dependence",
        "domain_hazard",
        "asymptotic_monomial",
        "branch_structure",
        "coordinate_invariant",
        "regime_lift",
    }
    for evidence in bundle.values():
        assert evidence.hard_constraints["should_run"] is False
        assert evidence.metadata["advisory_only"] is True
    assert bundle["symmetry"].hard_constraints["status"] == "insufficient_points"
    assert bundle["regime_lift"].hard_constraints["status"] == "missing_regime_metadata"
