# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

from nestynet_sr.sr_search.factorized_search.subproblem_tests import build_expanded_family_evidence_bundle


def test_expanded_family_battery_detects_even_symmetry():
    x_axis = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64).unsqueeze(-1)
    y = (x_axis[:, :1] ** 2)
    grad = 2.0 * x_axis[:, :1]
    d2 = torch.full_like(x_axis[:, :1], 2.0)

    bundle = build_expanded_family_evidence_bundle(
        x_fit=x_axis,
        t_fit=y,
        grad_fit=grad,
        d2_fit=d2,
        fit_jet_source="oracle",
        probe_jet_source="oracle",
    )

    symmetry = bundle["symmetry"]
    assert symmetry.hard_constraints["status"] == "even_like"
    assert symmetry.hard_constraints["zero_crossing_count"] == 0
    assert symmetry.hard_constraints["mirror_even_residual"] <= symmetry.hard_constraints["mirror_odd_residual"]
    assert symmetry.hard_constraints["jet_evidence_used"] is True
    assert symmetry.hard_constraints["gradient_mirror_even_residual"] <= symmetry.hard_constraints["gradient_mirror_odd_residual"]
    assert symmetry.hard_constraints["d2_mirror_even_residual"] <= symmetry.hard_constraints["d2_mirror_odd_residual"]
    assert symmetry.hard_constraints["jet_source"] == "oracle"
    assert symmetry.hard_constraints["exact_jet_used"] is True


def test_expanded_family_battery_detects_low_rank_dependence():
    x0 = torch.linspace(-1.0, 1.0, 19, dtype=torch.float64)
    x1 = (x0 ** 2) + 0.2
    x = torch.stack([x0, x1], dim=1)
    y = x[:, :1].clone()
    grad = torch.stack([torch.ones_like(x0), torch.zeros_like(x0)], dim=1)

    bundle = build_expanded_family_evidence_bundle(
        x_fit=x,
        t_fit=y,
        grad_fit=grad,
        fit_jet_source="numeric_local_quadratic",
        fit_jet_fallback_used=True,
        target_dim=("L",),
        active_vars=(0,),
        wrappers_left=2,
        recursion_level=1,
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        regime_metadata={"dataset_ids": ["fit0"]},
    )

    low_rank = bundle["low_rank_dependence"]
    assert low_rank.hard_constraints["top_var"] == 0
    assert low_rank.hard_constraints["dominant_var_frac"] >= 0.75
    assert low_rank.hard_constraints["active_var_estimate"] == 1
    assert low_rank.hard_constraints["jet_evidence_used"] is True
    assert low_rank.hard_constraints["jet_top_var"] == 0
    assert low_rank.hard_constraints["numeric_jet_fallback_used"] is True
    assert low_rank.hard_constraints["context"]["target_dim"] == ("L",)
    assert low_rank.hard_constraints["context"]["active_vars"] == (0,)
    assert low_rank.hard_constraints["context"]["wrappers_left"] == 2
    assert low_rank.hard_constraints["context"]["direction"] == "inside_out"
    assert low_rank.hard_constraints["regime"]["dataset_ids"] == ["fit0"]


def test_expanded_family_battery_distinguishes_additive_from_interaction_heavy():
    axis = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    x = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    additive_grad = torch.stack([2.0 * x[:, 0], 2.0 * x[:, 1]], dim=1)
    interaction_grad = torch.stack([x[:, 1], x[:, 0]], dim=1)

    additive = build_expanded_family_evidence_bundle(
        x_fit=x,
        t_fit=(x[:, :1] + x[:, 1:2]),
        grad_fit=additive_grad,
    )["separability"]
    interaction = build_expanded_family_evidence_bundle(
        x_fit=x,
        t_fit=(x[:, :1] * x[:, 1:2]),
        grad_fit=interaction_grad,
    )["separability"]

    assert additive.hard_constraints["status"] == "additive_like"
    assert additive.hard_constraints["interaction_gain"] <= 0.10
    assert additive.hard_constraints["jet_evidence_used"] is True
    assert additive.hard_constraints["jet_interaction_gain"] <= 0.10
    assert interaction.hard_constraints["status"] == "interaction_heavy"
    assert interaction.hard_constraints["jet_interaction_gain"] > additive.hard_constraints["jet_interaction_gain"]


def test_expanded_family_battery_flags_domain_hazard():
    x = torch.linspace(0.05, 1.0, 33, dtype=torch.float64).unsqueeze(-1)
    y = 1.0 / x
    grad = -1.0 / (x * x)
    d2 = 2.0 / (x * x * x)

    bundle = build_expanded_family_evidence_bundle(
        x_fit=x,
        t_fit=y,
        grad_fit=grad,
        d2_fit=d2,
    )

    hazard = bundle["domain_hazard"]
    assert hazard.hard_constraints["hazard_severe"] is True
    assert hazard.hard_constraints["status"] == "severe_hazard"
    assert hazard.hard_constraints["gradient_signal_source"] == "jet"
    assert hazard.hard_constraints["curvature_signal_source"] == "jet"
    assert hazard.hard_constraints["gradient_spike_ratio"] > 8.0


def test_expanded_family_battery_detects_asymptotic_monomial_behavior():
    x = torch.linspace(1.0, 6.0, 33, dtype=torch.float64).unsqueeze(-1)
    y = x ** 3

    evidence = build_expanded_family_evidence_bundle(
        x_fit=x,
        t_fit=y,
    )["asymptotic_monomial"]

    assert evidence.hard_constraints["status"] == "monomial_like"
    assert evidence.hard_constraints["log_fit_r2"] >= 0.95
    assert abs(float(evidence.hard_constraints["exponent_estimate"]) - 3.0) <= 0.15


def test_expanded_family_battery_detects_branch_structure():
    x = torch.linspace(0.02, 1.0, 41, dtype=torch.float64).unsqueeze(-1)
    y = torch.sqrt(x)
    grad = 0.5 / torch.sqrt(x)

    evidence = build_expanded_family_evidence_bundle(
        x_fit=x,
        t_fit=y,
        grad_fit=grad,
    )["branch_structure"]

    assert evidence.hard_constraints["status"] in {"branch_like", "branch_like_hazard"}
    assert evidence.hard_constraints["one_sided_support"] is True
    assert evidence.hard_constraints["branch_cut_risk"] >= 0.65


def test_expanded_family_battery_detects_regime_lift_from_constant_drift():
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64).unsqueeze(-1)
    y = x.clone()

    evidence = build_expanded_family_evidence_bundle(
        x_fit=x,
        t_fit=y,
        regime_metadata={
            "local_constants_by_experiment": {
                "r0": {"c": 1.0},
                "r1": {"c": 2.0},
                "r2": {"c": 4.0},
            },
            "regime_ids": ["r0", "r1", "r2"],
            "trigger_mean_cv": 0.25,
        },
    )["regime_lift"]

    assert evidence.hard_constraints["status"] == "drifting_constants"
    assert evidence.hard_constraints["top_constant_name"] == "c"
    assert evidence.family_scores["regime_lift"] > 0.0


def test_expanded_family_battery_detects_coordinate_invariant_signal():
    axis = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    x = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    y = (x[:, :1] + x[:, 1:2])
    grad = torch.ones_like(x)

    evidence = build_expanded_family_evidence_bundle(
        x_fit=x,
        t_fit=y,
        grad_fit=grad,
        active_vars=(0, 1),
    )["coordinate_invariant"]

    assert evidence.hard_constraints["status"] == "single_index_like"
    assert evidence.hard_constraints["gradient_direction_coherence"] >= 0.95
    assert evidence.seed_nodes
