# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json

import nestynet_sr.sr_search.factorized_search.oracle_lab as oracle_lab_mod
import nestynet_sr.sr_search.factorized_search.oracle_regression as oracle_regression_mod
import nestynet_sr.sr_search.factorized_search.oracle_suite as oracle_suite_mod


def test_oracle_regression_quick12_manifest_resolves_specs():
    manifest_path, payload = oracle_regression_mod.load_regression_suite()
    specs = oracle_regression_mod.resolve_suite_spec_paths(payload, manifest_path=manifest_path)
    assert payload["suite_id"] == "quick12"
    assert float(payload["defaults"]["wall_time_limit_s"]) == 300.0
    assert len(specs) == 12
    assert all(p.is_file() for p in specs)


def test_oracle_regression_broad20_manifest_resolves_specs():
    manifest_path, payload = oracle_regression_mod.load_regression_suite(
        "examples/oracle_factorized_search/regression_suites/broad20.json"
    )
    specs = oracle_regression_mod.resolve_suite_spec_paths(payload, manifest_path=manifest_path)
    assert payload["suite_id"] == "broad20"
    assert len(specs) == 20
    assert all(p.is_file() for p in specs)


def test_oracle_regression_inverse_compare_manifest_resolves_specs():
    manifest_path, payload = oracle_regression_mod.load_regression_suite(
        "examples/oracle_factorized_search/regression_suites/quick12_inverse_compare.json"
    )
    specs = oracle_regression_mod.resolve_suite_spec_paths(payload, manifest_path=manifest_path)
    assert payload["suite_id"] == "quick12_inverse_compare"
    assert payload["defaults"]["profiles"] == ["current", "no_inverse"]
    assert payload["profile_overrides"]["current"]["inverse_steering_enable"] is True
    assert payload["profile_overrides"]["current"]["inverse_spec_enable"] is True
    assert payload["profile_overrides"]["current"]["hole_search_enable"] is False
    assert "no_inverse" in payload["profile_overrides"]
    assert len(specs) == 12
    assert all(p.is_file() for p in specs)


def test_oracle_regression_method_attribution_manifest_resolves_specs():
    manifest_path, payload = oracle_regression_mod.load_regression_suite(
        "examples/oracle_factorized_search/regression_suites/quick12_method_attribution.json"
    )
    specs = oracle_regression_mod.resolve_suite_spec_paths(payload, manifest_path=manifest_path)
    assert payload["suite_id"] == "quick12_method_attribution"
    assert payload["defaults"]["profiles"] == ["residual_basin_only", "inverse_spec", "hole_fix"]
    assert payload["defaults"]["modes"] == ["refine_off"]
    assert payload["profile_overrides"]["residual_basin_only"]["inverse_steering_enable"] is False
    assert payload["profile_overrides"]["inverse_spec"]["inverse_steering_enable"] is True
    assert payload["profile_overrides"]["inverse_spec"]["hole_search_enable"] is False
    assert payload["profile_overrides"]["hole_fix"]["hole_search_enable"] is True
    assert len(specs) == 12
    assert all(p.is_file() for p in specs)


def test_oracle_regression_frozen_compare_manifest_resolves_specs():
    manifest_path, payload = oracle_regression_mod.load_regression_suite(
        "examples/oracle_factorized_search/regression_suites/quick12_frozen_compare.json"
    )
    specs = oracle_regression_mod.resolve_suite_spec_paths(payload, manifest_path=manifest_path)
    assert payload["suite_id"] == "quick12_frozen_compare"
    assert payload["defaults"]["profiles"] == ["noop", "current_best"]
    assert "noop" in payload["profile_overrides"]
    assert "current_best" in payload["profile_overrides"]
    assert len(specs) == 12
    assert all(p.is_file() for p in specs)


def test_oracle_regression_compare_spec_summaries_flags_regressions():
    baseline = [
        {
            "spec_id": "toy",
            "profile": "current",
            "mode": "refine_off",
            "budget": 100,
            "solve_rate": 1.0,
            "best_mse_median": 1.0e-6,
            "wall_seconds_mean": 1.0,
        }
    ]
    current = [
        {
            "spec_id": "toy",
            "profile": "current",
            "mode": "refine_off",
            "budget": 100,
            "solve_rate": 0.0,
            "best_mse_median": 1.0e-3,
            "wall_seconds_mean": 3.5,
        }
    ]
    regressions = oracle_regression_mod.compare_spec_summaries(
        current,
        baseline,
        mse_factor=10.0,
        time_factor=2.0,
    )
    assert len(regressions) == 1
    reasons = " | ".join(regressions[0]["reasons"])
    assert "solve_rate" in reasons
    assert "best_mse_median" in reasons
    assert "wall_seconds_mean" in reasons


def test_oracle_suite_make_hp_applies_profile_inverse_overrides():
    args = oracle_lab_mod._parse_args(
        [
            "--spec",
            "dummy.json",
            "--wall_time_limit_s",
            "123.0",
            "--inverse_steering",
            "--inverse_spec",
            "--hole_search_enable",
            "--no_inverse_micro_search",
        ]
    )
    hp = oracle_suite_mod._make_hp(args, budget=17, refine_enable=False)
    assert hp.n_iter == 17
    assert hp.wall_time_limit_s == 123.0
    assert hp.inverse_steering_enable is True
    assert hp.inverse_spec_enable is True
    assert hp.hole_search_enable is True


def test_oracle_regression_main_writes_outputs(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_oracle_suite(
        spec_paths,
        *,
        budgets,
        modes,
        profiles,
        profile_overrides,
        n_repeats,
        seed,
        dtype,
        enforce_dims,
        success_mse_threshold,
        verbose,
        hp_overrides,
        output_dir,
        save_individual_reports,
        jobs,
    ):
        captured["spec_paths"] = [str(p) for p in spec_paths]
        captured["no_brute_force"] = bool(getattr(hp_overrides, "no_brute_force", False))
        captured["jobs"] = int(jobs)
        captured["wall_time_limit_s"] = getattr(hp_overrides, "wall_time_limit_s", None)
        captured["profiles"] = [str(v) for v in profiles]
        captured["profile_overrides"] = {str(k): dict(v) for k, v in dict(profile_overrides or {}).items()}
        out_dir = output_dir
        (out_dir / "oracle_suite_results.json").write_text("{}", encoding="utf-8")
        (out_dir / "oracle_suite_rows.csv").write_text("", encoding="utf-8")
        (out_dir / "oracle_suite_summary.csv").write_text("", encoding="utf-8")
        return {
            "rows": [
                {
                    "spec_id": "toy",
                    "spec_path": "toy.json",
                    "profile": "current",
                    "mode": "refine_off",
                    "budget": 100,
                    "repeat": 0,
                    "seed": 0,
                    "best_mse": 1.0e-6,
                    "success": 1,
                    "best_expr": "x0",
                    "mapping_kind": "poly1",
                    "wall_seconds": 0.1,
                }
            ]
        }

    monkeypatch.setattr(oracle_regression_mod, "run_oracle_suite", _fake_run_oracle_suite)
    output_dir = tmp_path / "regression"
    rc = oracle_regression_mod.main(
        [
            "--output_dir",
            str(output_dir),
            "--jobs",
            "6",
        ]
    )
    assert rc == 0
    assert captured["no_brute_force"] is True
    assert captured["jobs"] == 6
    assert captured["wall_time_limit_s"] == 300.0
    assert captured["profiles"] == ["current"]
    assert captured["profile_overrides"] == {}
    assert len(captured["spec_paths"]) == 12
    result_path = output_dir / "oracle_regression_results.json"
    summary_path = output_dir / "oracle_regression_spec_summary.csv"
    assert result_path.is_file()
    assert summary_path.is_file()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["suite_id"] == "quick12"
    assert payload["profiles"] == ["current"]
    assert payload["spec_summary"][0]["spec_id"] == "toy"
    assert payload["spec_summary"][0]["profile"] == "current"


def test_oracle_regression_discovery_writes_output(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_oracle_suite(
        spec_paths,
        *,
        budgets,
        modes,
        profiles,
        profile_overrides,
        n_repeats,
        seed,
        dtype,
        enforce_dims,
        success_mse_threshold,
        verbose,
        hp_overrides,
        output_dir,
        save_individual_reports,
        jobs,
    ):
        captured["save_individual_reports"] = bool(save_individual_reports)
        return {
            "rows": [
                {
                    "spec_id": "toy",
                    "spec_path": "toy.json",
                    "profile": "current",
                    "mode": "refine_off",
                    "budget": 100,
                    "repeat": 0,
                    "seed": 0,
                    "best_mse": 1.0e-6,
                    "success": 1,
                    "best_expr": "x0",
                    "mapping_kind": "poly1",
                    "wall_seconds": 0.1,
                    "report_path": "toy_report.json",
                }
            ]
        }

    def _fake_run_oracle_discovery_benchmark(
        regression_payload,
        *,
        output_dir,
        committee_topk,
        max_members,
        experiment_manifest_path,
        beta,
        gamma,
        disagreement_mode,
        lambda_cost,
        lambda_noise,
        lambda_feasibility,
        witness_capture_enable,
        witness_hessian_diag_enable,
        diagnostic_set,
        dtype,
    ):
        captured["committee_topk"] = int(committee_topk)
        captured["max_members"] = max_members
        captured["experiment_manifest_path"] = experiment_manifest_path
        captured["witness_capture_enable"] = bool(witness_capture_enable)
        captured["witness_hessian_diag_enable"] = bool(witness_hessian_diag_enable)
        captured["diagnostic_set"] = str(diagnostic_set)
        captured["disagreement_mode"] = str(disagreement_mode)
        return {
            "mode": "oracle_discovery_benchmark",
            "aggregate": {"n_runs": 1},
            "runs": [],
        }

    monkeypatch.setattr(oracle_regression_mod, "run_oracle_suite", _fake_run_oracle_suite)
    monkeypatch.setattr(oracle_regression_mod, "run_oracle_discovery_benchmark", _fake_run_oracle_discovery_benchmark)
    output_dir = tmp_path / "regression_discovery"
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(json.dumps({"experiments": []}), encoding="utf-8")

    rc = oracle_regression_mod.main(
        [
            "--output_dir",
            str(output_dir),
            "--discovery_enable",
            "--discovery_committee_topk",
            "5",
            "--discovery_experiment_manifest",
            str(manifest_path),
        ]
    )

    assert rc == 0
    assert captured["save_individual_reports"] is True
    assert captured["committee_topk"] == 5
    assert captured["experiment_manifest_path"] == str(manifest_path)
    assert captured["disagreement_mode"] == "witness"
    assert (output_dir / "oracle_discovery_results.json").is_file()
    payload = json.loads((output_dir / "oracle_regression_results.json").read_text(encoding="utf-8"))
    assert payload["discovery_enabled"] is True
    assert payload["discovery_results_path"].endswith("oracle_discovery_results.json")
