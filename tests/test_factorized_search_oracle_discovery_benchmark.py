# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("sympy")

import nestynet_sr.sr_search.factorized_search.oracle_discovery_benchmark as discovery_mod


def _mk_spec(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _simple_payload(spec_id: str, *, expr: str = "x") -> dict:
    return {
        "id": spec_id,
        "basis": ["L"],
        "variables": [{"name": "x", "bounds": [0.0, 2.0], "dim": [1]}],
        "constants": [],
        "target": {"expr": expr, "dim": [1]},
    }


def _report(path: Path, *, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")


def test_run_oracle_discovery_benchmark_builds_committee_summary(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
            {
                "expr": "(2*x0)",
                "expr_ast": ["mul", ["const", 2.0], ["var", 0]],
                "mse": 0.02,
                "size": 3,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
        ],
    )
    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_benchmark(
        regression_payload,
        output_dir=tmp_path,
        committee_topk=2,
        max_members=4,
        dtype=torch.float64,
    )

    assert payload["mode"] == "oracle_discovery_benchmark"
    assert payload["aggregate"]["n_runs"] == 1
    run = payload["runs"][0]
    assert run["committee_summary"]["member_count"] == 1
    assert run["physics_summary"]
    assert run["committee_members"][0]["local_constants_by_experiment"]["toy"]


def test_run_oracle_discovery_benchmark_selects_high_disagreement_experiment(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
            {
                "expr": "(-x0)",
                "expr_ast": ["neg", ["var", 0]],
                "mse": 0.02,
                "size": 2,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
        ],
    )
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "flat", "type": "points", "points": [[0.0]]},
                    {"experiment_id": "spread", "type": "points", "points": [[1.0], [2.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_benchmark(
        regression_payload,
        output_dir=tmp_path,
        committee_topk=2,
        max_members=4,
        experiment_manifest_path=str(manifest_path),
        beta=0.0,
        gamma=0.0,
        dtype=torch.float64,
    )

    selection = payload["runs"][0]["experiment_selection"]
    assert selection["selected"]["experiment_id"] == "spread"


def test_run_oracle_discovery_benchmark_populates_witness_predictions(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "(x0*x0)",
                "expr_ast": ["sqr", ["var", 0]],
                "mse": 0.01,
                "size": 2,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            }
        ],
    )
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "spread", "type": "points", "points": [[1.0], [2.0], [3.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_benchmark(
        regression_payload,
        output_dir=tmp_path,
        committee_topk=1,
        max_members=2,
        experiment_manifest_path=str(manifest_path),
        witness_capture_enable=True,
        witness_hessian_diag_enable=True,
        diagnostic_set="extended",
        dtype=torch.float64,
    )

    candidate = payload["runs"][0]["experiment_candidates_full"][0]
    member_id = payload["runs"][0]["committee_members"][0]["member_id"]
    assert candidate["derivative_predictions"][member_id] == [[2.0], [4.0], [6.0]]
    diag = candidate["diagnostic_predictions"][member_id]
    assert diag["hdiag_abs_mean"] == 2.0
    assert payload["config"]["witness_capture_enable"] is True


def test_run_oracle_discovery_benchmark_applies_research_profile_overrides(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
            {
                "expr": "(2 - x0)",
                "expr_ast": ["sub", ["const", 2.0], ["var", 0]],
                "mse": 0.02,
                "size": 3,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
        ],
    )
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "profile_points",
                        "type": "points",
                        "points": [[0.8], [1.2]],
                        "bounds": {"x": [0.0, 2.0]},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_benchmark(
        regression_payload,
        output_dir=tmp_path,
        research_profile="teacher_witness_full",
        committee_topk=2,
        max_members=4,
        experiment_manifest_path=str(manifest_path),
        dtype=torch.float64,
    )

    config = payload["config"]
    assert config["research_profile"] == "teacher_witness_full"
    assert config["witness_capture_enable"] is True
    assert config["witness_hessian_diag_enable"] is True
    assert config["diagnostic_set"] == "physics"
    assert config["beta"] == 1.0
    assert config["gamma"] == 0.5
    assert config["disagreement_mode"] == "witness"
    assert config["experiment_optimize_enable"] is True
    assert config["theory_benchmark_enable"] is True
    assert config["discovery_constant_lift_enable"] is True
    assert config["discovery_constant_lift_apply_enable"] is True
    assert config["discovery_constant_lift_apply_topk"] == 2


def test_run_oracle_discovery_benchmark_witness_mode_sees_shape_disagreement(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
            {
                "expr": "(2 - x0)",
                "expr_ast": ["sub", ["const", 2.0], ["var", 0]],
                "mse": 0.02,
                "size": 3,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
        ],
    )
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "shape_agree", "type": "points", "points": [[1.0], [1.0]]},
                    {"experiment_id": "shape_split", "type": "points", "points": [[0.0], [2.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_benchmark(
        regression_payload,
        output_dir=tmp_path,
        committee_topk=2,
        max_members=4,
        experiment_manifest_path=str(manifest_path),
        beta=0.0,
        gamma=0.0,
        disagreement_mode="witness",
        dtype=torch.float64,
    )

    selection = payload["runs"][0]["experiment_selection"]
    assert payload["config"]["disagreement_mode"] == "witness"
    assert selection["disagreement_mode"] == "witness"
    assert selection["selected"]["experiment_id"] == "shape_split"


def test_run_oracle_discovery_benchmark_defaults_to_witness_mode(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
            {
                "expr": "(2 - x0)",
                "expr_ast": ["sub", ["const", 2.0], ["var", 0]],
                "mse": 0.02,
                "size": 3,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
        ],
    )
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "shape_agree", "type": "points", "points": [[1.0], [1.0]]},
                    {"experiment_id": "shape_split", "type": "points", "points": [[0.0], [2.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_benchmark(
        regression_payload,
        output_dir=tmp_path,
        committee_topk=2,
        max_members=4,
        experiment_manifest_path=str(manifest_path),
        beta=0.0,
        gamma=0.0,
        dtype=torch.float64,
    )

    selection = payload["runs"][0]["experiment_selection"]
    assert payload["config"]["research_profile"] == "default"
    assert payload["config"]["disagreement_mode"] == "witness"
    assert selection["disagreement_mode"] == "witness"
    assert selection["selected"]["experiment_id"] == "shape_split"


def test_run_oracle_discovery_benchmark_scalar_request_without_legacy_profile_uses_witness(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
            {
                "expr": "(2 - x0)",
                "expr_ast": ["sub", ["const", 2.0], ["var", 0]],
                "mse": 0.02,
                "size": 3,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
        ],
    )
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "shape_agree", "type": "points", "points": [[1.0], [1.0]]},
                    {"experiment_id": "shape_split", "type": "points", "points": [[0.0], [2.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_benchmark(
        regression_payload,
        output_dir=tmp_path,
        committee_topk=2,
        max_members=4,
        experiment_manifest_path=str(manifest_path),
        beta=0.0,
        gamma=0.0,
        disagreement_mode="scalar",
        dtype=torch.float64,
    )

    selection = payload["runs"][0]["experiment_selection"]
    assert payload["config"]["research_profile"] == "default"
    assert payload["config"]["disagreement_mode"] == "witness"
    assert selection["disagreement_mode"] == "witness"
    assert selection["selected"]["experiment_id"] == "shape_split"


def test_run_oracle_discovery_benchmark_can_optimize_and_emit_theory_metrics(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
            {
                "expr": "(2 - x0)",
                "expr_ast": ["sub", ["const", 2.0], ["var", 0]],
                "mse": 0.02,
                "size": 3,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
        ],
    )
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "opt_points",
                        "type": "points",
                        "points": [[0.8], [1.2]],
                        "bounds": {"x": [0.0, 2.0]},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_benchmark(
        regression_payload,
        output_dir=tmp_path,
        committee_topk=2,
        max_members=4,
        experiment_manifest_path=str(manifest_path),
        disagreement_mode="witness",
        experiment_optimize_enable=True,
        experiment_opt_steps=24,
        experiment_opt_lr=0.1,
        theory_benchmark_enable=True,
        dtype=torch.float64,
    )

    run = payload["runs"][0]
    assert payload["config"]["experiment_optimize_enable"] is True
    assert payload["config"]["theory_benchmark_enable"] is True
    assert run["experiment_selection"]["optimization"]["optimized_candidate_count"] == 1
    assert run["experiment_candidates_full"][0]["conditions"]["optimized"] is True
    assert run["theory_benchmark"]["enabled"] is True
    assert payload["aggregate"]["mean_next_experiment_quality"] is not None


def test_run_oracle_discovery_benchmark_applies_constant_lift_proposals(monkeypatch, tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            }
        ],
    )

    def fake_discover_constant_lifts(committee, **kwargs):
        member = list(committee)[0]
        regime_payload = dict(next(iter(member.local_constants_by_experiment.values())))
        constant_name = sorted(regime_payload.keys())[0]
        return {
            "enabled": True,
            "proposal_count": 1,
            "triggered_member_count": 1,
            "members": [
                {
                    "member_id": str(member.member_id),
                    "triggered": True,
                    "parameter_stability": {"passed": False, "score": 0.2, "details": {}},
                    "proposals": [
                        {
                            "constant_name": str(constant_name),
                            "mean_cv": 0.8,
                            "sample_count": 3,
                            "substitution_preview": {
                                "constant_name": str(constant_name),
                                "lift_expr": "x0",
                            },
                            "lifted_display_expr": "(x0 + x0)",
                            "expr": "x0",
                            "improvement_ratio": 10.0,
                            "feature_source": "dataset_metadata",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(discovery_mod, "discover_constant_lifts", fake_discover_constant_lifts)

    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_benchmark(
        regression_payload,
        output_dir=tmp_path,
        committee_topk=1,
        max_members=4,
        discovery_constant_lift_enable=True,
        discovery_constant_lift_apply_enable=True,
        discovery_constant_lift_apply_topk=1,
        discovery_constant_lift_min_rel_gain=2.0,
        dtype=torch.float64,
    )

    run = payload["runs"][0]
    assert payload["config"]["discovery_constant_lift_apply_enable"] is True
    assert run["constant_lift_summary"]["applied_member_count"] == 1
    assert run["constant_lift_summary"]["surviving_applied_member_count"] == 1
    lifted_member = next(
        row
        for row in run["committee_members"]
        if row["metadata"].get("constant_lift_applied", False)
    )
    assert lifted_member["display_expr"] == "(x0 + x0)"
    assert lifted_member["metadata"]["constant_lift_parent_member_id"]


def test_run_oracle_discovery_research_benchmark_reports_profile_activation(monkeypatch, tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "mse_eff": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
                "proposal_family": "local_recursive_sr",
                "generation_source": "local_recursive_sr",
                "witness_value_loss": 0.01,
                "witness_grad_loss": 0.30,
                "witness_energy_total": 0.31,
                "witness_fit_jet_source": "oracle",
                "witness_probe_jet_source": "oracle",
                "witness_exact_jet_used": True,
            },
            {
                "expr": "(2 - x0)",
                "expr_ast": ["sub", ["const", 2.0], ["var", 0]],
                "mse": 0.015,
                "mse_eff": 0.015,
                "size": 3,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
                "proposal_family": "coordinate_lift",
                "generation_source": "coordinate_lift",
                "witness_value_loss": 0.015,
                "witness_grad_loss": 0.01,
                "witness_energy_total": 0.025,
                "witness_fit_jet_source": "symbolic",
                "witness_probe_jet_source": "symbolic",
                "witness_exact_jet_used": True,
            },
            {
                "expr": "(-x0)",
                "expr_ast": ["neg", ["var", 0]],
                "mse": 0.03,
                "mse_eff": 0.03,
                "size": 2,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
                "proposal_family": "tangent_edit",
                "generation_source": "tangent_edit",
                "witness_value_loss": 0.03,
                "witness_grad_loss": 0.02,
                "witness_energy_total": 0.05,
                "witness_fit_jet_source": "numeric_local_quadratic",
                "witness_probe_jet_source": "numeric_local_quadratic",
                "witness_numeric_jet_fallback_used": True,
            },
            {
                "expr": "(x0*x0)",
                "expr_ast": ["sqr", ["var", 0]],
                "mse": 0.04,
                "mse_eff": 0.04,
                "size": 2,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
                "proposal_family": "soft_edit_search",
                "generation_source": "soft_edit_search",
                "witness_value_loss": 0.04,
                "witness_grad_loss": 0.01,
                "witness_energy_total": 0.05,
                "witness_fit_jet_source": "oracle",
                "witness_probe_jet_source": "numeric_local_quadratic",
                "witness_exact_jet_used": True,
                "witness_numeric_jet_fallback_used": True,
            },
        ],
    )
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "shape_agree", "type": "points", "points": [[1.0], [1.0]]},
                    {"experiment_id": "shape_split", "type": "points", "points": [[0.0], [2.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_discover_constant_lifts(committee, **kwargs):
        member = list(committee)[0]
        regime_payload = dict(next(iter(member.local_constants_by_experiment.values())))
        constant_name = sorted(regime_payload.keys())[0]
        return {
            "enabled": True,
            "proposal_count": 1,
            "triggered_member_count": 1,
            "members": [
                {
                    "member_id": str(member.member_id),
                    "triggered": True,
                    "parameter_stability": {"passed": False, "score": 0.2, "details": {}},
                    "proposals": [
                        {
                            "constant_name": str(constant_name),
                            "mean_cv": 0.8,
                            "sample_count": 3,
                            "substitution_preview": {
                                "constant_name": str(constant_name),
                                "lift_expr": "x0",
                            },
                            "lifted_display_expr": "(x0 + x0)",
                            "expr": "x0",
                            "improvement_ratio": 10.0,
                            "feature_source": "dataset_metadata",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(discovery_mod, "discover_constant_lifts", fake_discover_constant_lifts)

    regression_payload = {
        "suite_id": "toy_suite",
        "suite_manifest": "toy.json",
        "rows": [
            {
                "spec_id": "toy",
                "spec_path": str(spec_path),
                "profile": "current",
                "mode": "refine_off",
                "budget": 10,
                "repeat": 0,
                "seed": 0,
                "report_path": str(report_path),
            }
        ],
    }

    payload = discovery_mod.run_oracle_discovery_research_benchmark(
        regression_payload,
        output_dir=tmp_path,
        experiment_manifest_path=str(manifest_path),
        research_profiles=["legacy", "teacher_witness", "teacher_witness_full", "teacher_witness_exact"],
        committee_topk=4,
        max_members=4,
        dtype=torch.float64,
    )

    assert payload["mode"] == "oracle_discovery_research_benchmark"
    assert payload["profile_order"] == [
        "legacy",
        "teacher_witness",
        "teacher_witness_full",
        "teacher_witness_exact",
    ]
    profiles = {entry["research_profile"]: entry for entry in payload["profiles"]}
    legacy = profiles["legacy"]["research_activation_summary"]
    teacher = profiles["teacher_witness"]["research_activation_summary"]
    teacher_full = profiles["teacher_witness_full"]["research_activation_summary"]
    teacher_exact = profiles["teacher_witness_exact"]["research_activation_summary"]
    assert legacy["witness_mode_run_count"] == 1
    assert teacher["witness_mode_run_count"] == 1
    assert teacher["selected_experiment_counts"]["shape_split"] == 1
    assert teacher_full["experiment_optimization_run_count"] == 1
    assert teacher_full["constant_lift_applied_total"] == 1
    assert teacher_full["interesting_route_usage"]["local_recursive_sr"] == 1
    assert teacher_full["interesting_route_usage"]["coordinate_lift"] == 1
    assert teacher_full["interesting_route_usage"]["tangent_edit"] == 1
    assert teacher_full["interesting_route_usage"]["soft_edit_search"] == 1
    assert teacher_full["witness_weighted_ranking_changed_run_count"] == 1
    assert teacher_full["exact_jet_row_total"] == 3
    assert teacher_full["numeric_jet_fallback_row_total"] == 2
    assert teacher_full["jet_source_counts"] == {
        "numeric_local_quadratic": 2,
        "oracle": 2,
        "symbolic": 1,
    }
    assert teacher_exact["exact_jet_row_total"] == 3
    assert teacher_exact["numeric_jet_fallback_row_total"] == 2
    assert payload["comparison"]["profiles_with_witness_mode"] == [
        "legacy",
        "teacher_witness",
        "teacher_witness_full",
        "teacher_witness_exact",
    ]
    assert payload["comparison"]["profiles_with_experiment_optimization"] == [
        "teacher_witness_full",
        "teacher_witness_exact",
    ]
    assert payload["comparison"]["profiles_with_exact_jet_usage"] == [
        "legacy",
        "teacher_witness",
        "teacher_witness_full",
        "teacher_witness_exact",
    ]
    assert payload["comparison"]["profiles_with_numeric_jet_fallback"] == [
        "legacy",
        "teacher_witness",
        "teacher_witness_full",
        "teacher_witness_exact",
    ]
    assert payload["comparison"]["jet_source_usage_by_profile"]["teacher_witness_exact"] == {
        "numeric_local_quadratic": 2,
        "oracle": 2,
        "symbolic": 1,
    }


def test_oracle_discovery_benchmark_main_writes_output(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            }
        ],
    )
    results_path = tmp_path / "oracle_regression_results.json"
    results_path.write_text(
        json.dumps(
            {
                "suite_id": "toy_suite",
                "suite_manifest": "toy.json",
                "rows": [
                    {
                        "spec_id": "toy",
                        "spec_path": str(spec_path),
                        "profile": "current",
                        "mode": "refine_off",
                        "budget": 10,
                        "repeat": 0,
                        "seed": 0,
                        "report_path": str(report_path),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "oracle_discovery_results.json"

    rc = discovery_mod.main(
        [
            "--results",
            str(results_path),
            "--output",
            str(out_path),
        ]
    )

    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "oracle_discovery_benchmark"


def test_oracle_discovery_benchmark_main_can_write_profile_smoke_benchmark(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    _mk_spec(spec_path, _simple_payload("toy"))
    report_path = tmp_path / "individual_reports" / "toy.current.refine_off.n10.r0.json"
    _report(
        report_path,
        rows=[
            {
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mse": 0.01,
                "size": 1,
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            }
        ],
    )
    results_path = tmp_path / "oracle_regression_results.json"
    results_path.write_text(
        json.dumps(
            {
                "suite_id": "toy_suite",
                "suite_manifest": "toy.json",
                "rows": [
                    {
                        "spec_id": "toy",
                        "spec_path": str(spec_path),
                        "profile": "current",
                        "mode": "refine_off",
                        "budget": 10,
                        "repeat": 0,
                        "seed": 0,
                        "report_path": str(report_path),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "oracle_discovery_research_benchmark.json"

    rc = discovery_mod.main(
        [
            "--results",
            str(results_path),
            "--output",
            str(out_path),
            "--research_profile_benchmark_enable",
        ]
    )

    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "oracle_discovery_research_benchmark"
    assert payload["profile_order"] == [
        "legacy",
        "teacher_witness",
        "teacher_witness_full",
        "teacher_witness_exact",
    ]
