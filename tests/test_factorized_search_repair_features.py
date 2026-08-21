# SPDX-License-Identifier: MPL-2.0

import pytest

from nestynet_sr.sr_search.factorized_search.repair_critic import extract_repair_critic_features
from nestynet_sr.sr_search.factorized_search.engine.signals import (
    InverseSteeringPotential,
    ModeStateFeatures,
    PathStateFeatures,
)
from nestynet_sr.sr_search.factorized_search.policy.features import RepairControllerFeatureRecord
from nestynet_sr.sr_search.factorized_search.repair_features import (
    InverseSteeringPotential as LegacyInverseSteeringPotential,
    PathStateFeatures as LegacyPathStateFeatures,
    RepairControllerFeatureRecord as LegacyRepairControllerFeatureRecord,
)


def test_repair_features_facade_reexports_split_modules():
    assert LegacyInverseSteeringPotential is InverseSteeringPotential
    assert LegacyPathStateFeatures is PathStateFeatures
    assert LegacyRepairControllerFeatureRecord is RepairControllerFeatureRecord


def test_repair_controller_feature_record_preserves_flat_critic_features():
    row = {
        "parent_expr": "parent_demo",
        "parent_best_eff_mse": 1.0e-3,
        "parent_best_raw_mse": 2.0e-3,
        "parent_visits": 7,
        "parent_visits_since_improve": 3,
        "parent_stagnation_score": 0.4,
        "parent_stagnation_ratio": 0.25,
        "gate_allowed": True,
        "gate_reason": "ok",
        "path_entropy": 0.2,
        "path_top_mass": 0.7,
        "path_second_mass": 0.2,
        "path_positive_count": 3,
        "identity_vs_full_log_mse_contrast": 1.8,
        "affine_vs_full_log_mse_contrast": 0.9,
        "identity_best_alt_probe_mse": 0.05,
        "affine_best_alt_probe_mse": 0.08,
        "full_best_alt_probe_mse": 0.01,
        "gate_best_path": [1, 0],
        "gate_best_weighted_rel_gain": 0.75,
        "gate_best_rel_gain": 0.60,
        "gate_best_valid_frac": 0.85,
        "gate_best_confidence": 0.90,
        "gate_best_transport_rel": 0.55,
        "gate_best_static_score": 0.80,
        "gate_best_branch_factor": 1.10,
        "gate_best_cut_factor": 0.95,
        "gate_best_profile_exact_monotone": True,
        "gate_best_profile_has_periodic": False,
        "gate_best_profile_has_muldiv": True,
        "gate_best_profile_has_explogsqrt": False,
        "selected_path": [1, 0],
        "selected_target_mode": "identity",
        "selected_path_gain": 0.72,
        "selected_path_gain_pre_cut": 0.80,
        "selected_rel_gain": 0.63,
        "selected_transport_rel": 0.58,
        "selected_lin_rel": 0.50,
        "selected_branch_factor": 1.05,
        "selected_cut_factor": 0.97,
        "selected_effective_n": 11.0,
        "local_candidate_count": 4,
        "estimated_child_raw_mse": 0.04,
        "estimated_child_eff_mse": 0.05,
        "estimated_parent_raw_mse": 0.12,
        "estimated_parent_eff_mse": 0.14,
        "estimated_one_hole_rel_improve_raw": 0.66,
        "estimated_one_hole_rel_improve_eff": 0.64,
        "proxy_one_hole_potential_eff": 0.61,
    }

    record = RepairControllerFeatureRecord.from_flat_row(row)
    typed_features = extract_repair_critic_features(record)
    flat_features = extract_repair_critic_features(row)

    assert record.parent.parent_expr == "parent_demo"
    assert record.path is not None
    assert record.path.path == (1, 0)
    assert record.candidate.selected_path == (1, 0)
    assert record.to_flat_dict()["gate_best_weighted_rel_gain"] == pytest.approx(0.75)
    assert typed_features == flat_features


def test_inverse_steering_potential_serializes_typed_path_rows():
    best_row = PathStateFeatures(
        path=(2, 1),
        target_mode="identity",
        weighted_rel_gain=0.82,
        rel_gain=0.67,
        valid_frac=0.91,
        confidence=0.88,
        mode_rows=(
            ModeStateFeatures(target_mode="identity", best_alt_probe_mse=0.06),
            ModeStateFeatures(target_mode="full", best_alt_probe_mse=0.01),
        ),
    )
    other_row = PathStateFeatures(path=(0,), target_mode="full", weighted_rel_gain=0.15)
    diag = InverseSteeringPotential(
        allowed=True,
        reason="ok",
        best_path=(2, 1),
        best_rel_gain=0.67,
        best_weighted_rel_gain=0.82,
        candidate_paths=((2, 1), (0,)),
        path_rows=(best_row, other_row),
    )

    assert diag.best_path_row == best_row
    assert diag.path_row_map()[(2, 1)] == best_row

    payload = diag.to_dict()
    assert payload["allowed"] is True
    assert payload["best_path"] == [2, 1]
    assert payload["candidate_paths"] == [[2, 1], [0]]
    assert payload["path_rows"][0]["mode_rows"][0]["target_mode"] == "identity"


def test_candidate_selected_path_falls_back_to_controller_action_path():
    row = {
        "parent_expr": "parent_demo",
        "controller_action_path": [2, 1],
        "path_summaries": [
            {
                "path": [2, 1],
                "target_mode": "identity",
                "weighted_rel_gain": 0.9,
                "rel_gain": 0.8,
                "valid_frac": 0.95,
                "confidence": 0.93,
            },
            {
                "path": [0],
                "target_mode": "full",
                "weighted_rel_gain": 0.2,
                "rel_gain": 0.1,
                "valid_frac": 0.6,
                "confidence": 0.5,
            },
        ],
    }

    record = RepairControllerFeatureRecord.from_flat_row(row)

    assert record.candidate.selected_path == (2, 1)


def test_path_state_features_accepts_inverse_action_beam_aliases():
    row = {
        "path": [1, 0],
        "target_mode": "identity",
        "path_gain": 0.75,
        "path_gain_pre_cut": 0.90,
        "path_gain_raw": 0.60,
        "path_cut_factor": 0.83,
        "best_alt_mse": 0.12,
        "valid_frac": 0.91,
        "confidence": 0.87,
    }

    path = PathStateFeatures.from_row(row)

    assert path.weighted_rel_gain == pytest.approx(0.75)
    assert path.weighted_rel_gain_pre_cut == pytest.approx(0.90)
    assert path.weighted_rel_gain_raw == pytest.approx(0.60)
    assert path.cut_factor == pytest.approx(0.83)
    assert path.best_alt_probe_mse == pytest.approx(0.12)
    assert path.valid_frac == pytest.approx(0.91)
    assert path.confidence == pytest.approx(0.87)
