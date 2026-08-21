# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import nestynet_sr.sr_search.factorized_search.oracle_portfolio_compare as portfolio_mod


def test_resolve_worker_torch_threads_defaults_to_one_for_parallel_jobs():
    assert portfolio_mod._resolve_worker_torch_threads(
        jobs=6,
        torch_num_threads=None,
        torch_num_interop_threads=None,
    ) == (1, 1)
    assert portfolio_mod._resolve_worker_torch_threads(
        jobs=1,
        torch_num_threads=None,
        torch_num_interop_threads=None,
    ) == (None, None)
    assert portfolio_mod._resolve_worker_torch_threads(
        jobs=6,
        torch_num_threads=2,
        torch_num_interop_threads=3,
    ) == (2, 3)


@pytest.mark.parametrize(
    ("expert", "family"),
    [
        ("periodic_specialist", ["periodic"]),
        ("exp_specialist", ["exp"]),
        ("log_specialist", ["log"]),
        ("rational_specialist", ["rational"]),
    ],
)
def test_make_hp_specialist_applies_stripped_profile(expert, family):
    args = portfolio_mod._parse_oracle_lab_args(["--spec", "dummy.json"])
    hp = portfolio_mod._make_hp(
        args,
        budget=1000,
        refine_enable=True,
        expert=expert,
    )

    assert hp.closure_search_enable is True
    assert hp.closure_search_families == family
    assert hp.max_depth == 5
    assert hp.brute_depth == 0
    assert hp.refine_enable is False
    assert hp.inverse_steering_enable is False
    assert hp.hole_search_enable is False
    assert hp.repair_pass_enable is False
    assert hp.inverse_spec_enable is False
    assert hp.inverse_spec_recursive_enable is False
    assert hp.no_residual is True
    assert hp.score_prescreen_enable is False
    assert hp.score_mapping_family_mode == "cheap"
    assert hp.early_stop_mse == 0.0


def test_score_structure_match_distinguishes_wrong_family():
    truth = ("add", ("cos", ("var", 0)), ("var", 1))
    wrong = ("exp", ("var", 0))
    close = ("add", ("cos", ("var", 0)), ("var", 1))
    assoc_close = ("add", ("var", 1), ("cos", ("var", 0)))

    wrong_sc = portfolio_mod.score_structure_match(truth, wrong)
    assert wrong_sc["exact_canonical_match"] is False
    assert wrong_sc["structure_ops_hit"] is False
    assert wrong_sc["truth_var_coverage"] is False

    close_sc = portfolio_mod.score_structure_match(truth, close)
    assert close_sc["exact_canonical_match"] is True
    assert close_sc["structure_ops_hit"] is True
    assert close_sc["truth_var_coverage"] is True

    assoc_sc = portfolio_mod.score_structure_match(truth, assoc_close)
    assert assoc_sc["exact_canonical_match"] is True
    assert assoc_sc["exact_structure_signature_match"] is True


def test_classify_solution_outcome_distinguishes_solve_surrogate_and_miss():
    truth = ("add", ("cos", ("var", 0)), ("var", 1))
    exact = ("add", ("cos", ("var", 0)), ("var", 1))
    wrong = ("exp", ("var", 0))
    affine_surrogate = ("mul", ("mul", ("var", 0), ("var", 2)), ("var", 1))
    assoc_exact = ("add", ("var", 1), ("cos", ("var", 0)))

    exact_sc = portfolio_mod.score_structure_match(truth, exact)
    solve = portfolio_mod.classify_solution_outcome(
        best_mse=1.0e-12,
        truth_ast=truth,
        candidate_ast=exact,
        structure=exact_sc,
    )
    assert solve["solution_label"] == "solve"

    assoc_sc = portfolio_mod.score_structure_match(truth, assoc_exact)
    assoc_solve = portfolio_mod.classify_solution_outcome(
        best_mse=1.0e-12,
        truth_ast=truth,
        candidate_ast=assoc_exact,
        structure=assoc_sc,
    )
    assert assoc_solve["solution_label"] == "solve"

    wrong_sc = portfolio_mod.score_structure_match(truth, wrong)
    surrogate = portfolio_mod.classify_solution_outcome(
        best_mse=1.0e-12,
        truth_ast=truth,
        candidate_ast=wrong,
        structure=wrong_sc,
    )
    assert surrogate["solution_label"] == "surrogate"

    affine_truth = ("add", ("mul", ("mul", ("var", 2), ("var", 0)), ("var", 1)), ("var", 3))
    affine_sc = portfolio_mod.score_structure_match(affine_truth, affine_surrogate)
    affine_outcome = portfolio_mod.classify_solution_outcome(
        best_mse=1.0e-12,
        truth_ast=affine_truth,
        candidate_ast=affine_surrogate,
        structure=affine_sc,
    )
    assert affine_sc["top_level_op_match"] is False
    assert affine_outcome["solution_label"] == "surrogate"

    miss = portfolio_mod.classify_solution_outcome(
        best_mse=1.0e-3,
        truth_ast=truth,
        candidate_ast=exact,
        structure=exact_sc,
    )
    assert miss["solution_label"] == "miss"


def test_build_portfolio_rows_tracks_mse_and_structure_preferences():
    rows = [
        {
            "spec_id": "toy",
            "spec_path": "toy.json",
            "expert": "baseline",
            "mode": "refine_off",
            "budget": 100,
            "repeat": 0,
            "seed": 0,
            "best_mse": 1.0e-8,
            "success": 1,
            "best_expr": "exp(x0)",
            "numeric_solve": 1,
            "solution_label": "surrogate",
            "solution_label_reason": "numeric_only",
            "exact_canonical_match": 0,
            "structure_ops_hit": 0,
        },
        {
            "spec_id": "toy",
            "spec_path": "toy.json",
            "expert": "periodic_scaffold",
            "mode": "refine_off",
            "budget": 100,
            "repeat": 0,
            "seed": 0,
            "best_mse": 1.0e-4,
            "success": 0,
            "best_expr": "cos(x0)+x1",
            "numeric_solve": 0,
            "solution_label": "miss",
            "solution_label_reason": "mse_above_threshold",
            "exact_canonical_match": 1,
            "structure_ops_hit": 1,
        },
        {
            "spec_id": "toy",
            "spec_path": "toy.json",
            "expert": "log_scaffold",
            "mode": "refine_off",
            "budget": 100,
            "repeat": 0,
            "seed": 0,
            "best_mse": 1.0e-6,
            "success": 0,
            "best_expr": "log(x0)+x1",
            "numeric_solve": 0,
            "solution_label": "miss",
            "solution_label_reason": "mse_above_threshold",
            "exact_canonical_match": 0,
            "structure_ops_hit": 1,
        },
    ]
    portfolio_rows = portfolio_mod.build_portfolio_rows(
        rows,
        experts=["baseline", "periodic_scaffold", "log_scaffold"],
    )
    assert len(portfolio_rows) == 1
    row = portfolio_rows[0]
    assert row["best_of_two_expert"] == "baseline"
    assert row["best_of_two_solution_expert"] == "baseline"
    assert row["baseline_solution_label"] == "surrogate"
    assert row["periodic_scaffold_solution_label"] == "miss"
    assert row["log_scaffold_solution_label"] == "miss"
    assert row["any_numeric_solve"] == 1
    assert row["any_structural_solve"] == 0
    assert row["any_surrogate"] == 1
    assert row["all_miss"] == 0
    assert row["any_exact_canonical_match"] == 1
    assert row["structure_preferred_expert"] == "periodic_scaffold"


def test_oracle_portfolio_compare_main_writes_outputs(monkeypatch, tmp_path):
    manifest_path = tmp_path / "suite.json"
    spec_path = tmp_path / "toy_spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "suite_id": "toy_suite",
                "defaults": {
                    "budgets": [100],
                    "modes": ["refine_off"],
                    "n_repeats": 1,
                    "seed": 0,
                    "dtype": "float64",
                    "ignore_dims": False,
                    "success_mse": 1.0e-6,
                    "quiet": True,
                    "fast_benchmark": True,
                },
                "specs": [str(spec_path)],
            }
        ),
        encoding="utf-8",
    )

    def _fake_load_equation_spec(path):
        return SimpleNamespace(id="toy", target_expr="cos(x0)+x1")

    def _fake_compile_target_ast(spec):
        return ("add", ("cos", ("var", 0)), ("var", 1))

    def _fake_run_oracle_equation(spec, *, factorized_search_hp, seed, dtype, enforce_dims, verbose):
        if bool(getattr(factorized_search_hp, "closure_search_enable", False)):
            best = {
                "mse": 1.0e-4,
                "expr": "cos(x0)+x1",
                "expr_ast": ["add", ["cos", ["var", 0]], ["var", 1]],
                "mapping_kind": "identity",
            }
        else:
            best = {
                "mse": 1.0e-8,
                "expr": "exp(x0)",
                "expr_ast": ["exp", ["var", 0]],
                "mapping_kind": "identity",
            }
        return {"best": best, "wall_seconds": 0.25}

    monkeypatch.setattr(portfolio_mod, "load_equation_spec", _fake_load_equation_spec)
    monkeypatch.setattr(portfolio_mod, "compile_target_ast", _fake_compile_target_ast)
    monkeypatch.setattr(portfolio_mod, "run_oracle_equation", _fake_run_oracle_equation)

    output_dir = tmp_path / "portfolio_out"
    rc = portfolio_mod.main(
        [
            "--suite_manifest",
            str(manifest_path),
            "--output_dir",
            str(output_dir),
            "--jobs",
            "1",
        ]
    )
    assert rc == 0
    payload = json.loads((output_dir / "oracle_portfolio_results.json").read_text(encoding="utf-8"))
    assert payload["experts"] == ["baseline", "periodic_scaffold"]
    assert payload["overall_summary"]["best_of_two_beats_baseline_count"] == 0
    assert payload["overall_summary"]["best_of_portfolio_beats_baseline_count"] == 0
    assert payload["overall_summary"]["any_numeric_solve_count"] == 1
    assert payload["overall_summary"]["any_structural_solve_count"] == 0
    assert payload["overall_summary"]["any_surrogate_count"] == 1
    assert payload["overall_summary"]["all_miss_count"] == 0
    assert payload["overall_summary"]["any_exact_canonical_match_count"] == 1
    assert len(payload["portfolio_rows"]) == 1
    row = payload["portfolio_rows"][0]
    assert row["best_of_two_expert"] == "baseline"
    assert row["baseline_solution_label"] == "surrogate"
    assert row["periodic_scaffold_solution_label"] == "miss"
    assert row["structure_preferred_expert"] == "periodic_scaffold"


@pytest.mark.parametrize(
    ("expert", "family"),
    [
        ("periodic_specialist", ["periodic"]),
        ("exp_specialist", ["exp"]),
        ("log_specialist", ["log"]),
        ("rational_specialist", ["rational"]),
    ],
)
def test_specialist_expert_is_accepted_in_main(monkeypatch, tmp_path, expert, family):
    manifest_path = tmp_path / "suite.json"
    spec_path = tmp_path / "toy_spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "suite_id": "toy_suite",
                "defaults": {
                    "budgets": [100],
                    "modes": ["refine_off"],
                    "n_repeats": 1,
                    "seed": 0,
                    "dtype": "float64",
                    "ignore_dims": False,
                    "success_mse": 1.0e-6,
                    "quiet": True,
                    "fast_benchmark": True,
                },
                "specs": [str(spec_path)],
            }
        ),
        encoding="utf-8",
    )

    def _fake_load_equation_spec(path):
        return SimpleNamespace(id="toy", target_expr="cos(x0)+x1")

    def _fake_compile_target_ast(spec):
        return ("add", ("cos", ("var", 0)), ("var", 1))

    captured = {}

    def _fake_run_oracle_equation(spec, *, factorized_search_hp, seed, dtype, enforce_dims, verbose):
        captured["hp"] = factorized_search_hp
        best = {
            "mse": 1.0e-12,
            "expr": "cos(x0)+x1",
            "expr_ast": ["add", ["cos", ["var", 0]], ["var", 1]],
            "mapping_kind": "identity",
        }
        return {"best": best, "wall_seconds": 0.25}

    monkeypatch.setattr(portfolio_mod, "load_equation_spec", _fake_load_equation_spec)
    monkeypatch.setattr(portfolio_mod, "compile_target_ast", _fake_compile_target_ast)
    monkeypatch.setattr(portfolio_mod, "run_oracle_equation", _fake_run_oracle_equation)

    output_dir = tmp_path / "portfolio_out_specialist"
    rc = portfolio_mod.main(
        [
            "--suite_manifest",
            str(manifest_path),
            "--output_dir",
            str(output_dir),
            "--experts",
            f"baseline,{expert}",
            "--jobs",
            "1",
        ]
    )
    assert rc == 0
    payload = json.loads((output_dir / "oracle_portfolio_results.json").read_text(encoding="utf-8"))
    assert payload["experts"] == ["baseline", expert]
    hp = captured["hp"]
    assert hp.closure_search_enable is True
    assert hp.closure_search_families == family
    assert hp.refine_enable is False
    assert hp.inverse_steering_enable is False
    assert hp.hole_search_enable is False
