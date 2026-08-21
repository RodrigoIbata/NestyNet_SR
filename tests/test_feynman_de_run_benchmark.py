# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("scipy")

REPO_ROOT = Path(__file__).resolve().parent.parent
FEYNMAN_DE_DIR = REPO_ROOT / "examples" / "feynman_de"
if str(FEYNMAN_DE_DIR) in sys.path:
    sys.path.remove(str(FEYNMAN_DE_DIR))
sys.path.insert(0, str(FEYNMAN_DE_DIR))
_problem_defs_mod = sys.modules.get("problem_defs")
if _problem_defs_mod is not None:
    _problem_defs_path = Path(getattr(_problem_defs_mod, "__file__", "") or "").resolve()
    if _problem_defs_path.parent != FEYNMAN_DE_DIR:
        del sys.modules["problem_defs"]

import run_benchmark as rb  # noqa: E402
from problem_defs import ProblemDef  # noqa: E402


def _problem(order: int = 1) -> ProblemDef:
    return ProblemDef(
        id="999",
        order=int(order),
        indep_var="x",
        dep_var="u",
        equation="du/dx=-u" if int(order) == 1 else "d2u/dx2=-u",
        description="unit-test",
        feynman_ref="-",
        params=[],
        param_ranges=[],
        ic_type="value",
    )


def test_run_factorized_search_engine_sets_repo_context(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")

    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    seen: dict[str, str] = {}

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        seen["cwd"] = str(cwd)
        seen["py"] = "" if env is None else str(env.get("PYTHONPATH", ""))
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"best": {"score": 1.0e-4, "residual_ast": "(du + u)"}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)

    out = rb._run_factorized_search_engine(
        _problem(order=1),
        [run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        holdout_last_k=0,
        traj_metric="max",
        no_sim_validate=True,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert out["engine"] == "factorized_search_oracle"
    assert out["status"] == "UNVERIFIED"
    assert seen["cwd"] == str(rb.REPO_ROOT)
    assert seen["py"].split(os.pathsep)[0] == str(rb.REPO_ROOT)


def test_build_run_de_command_adds_hybrid_rescue_flags(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd_sparse = rb.build_run_de_command(_problem(order=1), [csv_path], tmp_path, fast=True, rescue=False)
    cmd_hybrid = rb.build_run_de_command(_problem(order=1), [csv_path], tmp_path, fast=True, rescue=True)

    assert "--factorized-search-rescue" not in cmd_sparse
    assert "--factorized-rescue" not in cmd_sparse
    assert "--factorized-rescue" in cmd_hybrid
    assert cmd_hybrid[cmd_hybrid.index("--factorized-rescue") + 1] == "auto"
    assert "--factorized-search-rescue" in cmd_hybrid
    assert cmd_hybrid[cmd_hybrid.index("--factorized-search-rescue") + 1] == "auto"
    assert cmd_hybrid[cmd_hybrid.index("--factorized-search-preset") + 1] == "fast"
    assert cmd_hybrid[cmd_hybrid.index("--order_candidates") + 1] == "1"
    assert cmd_hybrid[cmd_hybrid.index("--epochs") + 1] == "120"
    assert cmd_hybrid[cmd_hybrid.index("--num_segments") + 1] == "12"
    assert "--ignore_units" in cmd_hybrid


def test_run_hybrid_engine_error_attaches_sigkill_resource_report(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    def _fake_command(cmd, log_path, *, cwd=None, env=None):
        return rb._LoggedCommandResult(
            args=list(cmd),
            returncode=-9,
            resource_report={
                "monitor": "test",
                "returncode": -9,
                "killed_by_signal": True,
                "signal": "SIGKILL",
                "peak_tree_rss_mb": 4096.0,
                "last_tree_rss_mb": 3900.0,
                "sample_count": 3,
            },
        )

    monkeypatch.setattr(rb, "_run_command_to_log", _fake_command)

    out = rb._run_hybrid_engine(
        _problem(order=2),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        factorized_de=True,
    )

    assert out["status"] == "ERROR"
    assert out["resource_failure_suspected"] is True
    assert out["command_killed_by_signal"] is True
    assert out["command_signal"] == "SIGKILL"
    assert out["command_peak_tree_rss_mb"] == pytest.approx(4096.0)
    assert "peak_tree_rss=4096.0 MB" in out["message"]


def test_build_run_de_command_fast_profile_fits_1000_point_smoke_data(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd = rb.build_run_de_command(_problem(order=1), [csv_path], tmp_path, fast=True, rescue=False)

    n_train = int(cmd[cmd.index("--ndata_train") + 1])
    n_val = int(cmd[cmd.index("--ndata_val") + 1])
    batch_size = int(cmd[cmd.index("--batch_size") + 1])

    assert n_train + n_val <= 1000
    assert batch_size <= n_train
    assert batch_size <= n_val
    assert cmd[cmd.index("--data_split") + 1] == "interleaved"


def test_build_run_de_command_only_enables_inverse_x_terms_for_singular_rhs(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    nonsingular = rb.build_run_de_command(
        _problem(order=2),
        [csv_path],
        tmp_path,
        fast=True,
        rescue=False,
    )

    singular_problem = ProblemDef(
        id="010",
        order=1,
        indep_var="r",
        dep_var="v",
        equation="dv/dr=-v/r",
        description="singular",
        feynman_ref="-",
        params=[],
        param_ranges=[],
        ic_type="decay",
    )
    singular = rb.build_run_de_command(
        singular_problem,
        [csv_path],
        tmp_path,
        fast=True,
        rescue=False,
    )

    inverse_flags = {"--include_inv_xdu", "--include_inv_xu", "--include_inv_x2u"}
    assert inverse_flags.isdisjoint(nonsingular)
    assert inverse_flags.issubset(set(singular))


def test_build_run_de_command_supports_factorized_search_only(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd = rb.build_run_de_command(
        _problem(order=1),
        [csv_path],
        tmp_path,
        fast=True,
        rescue=False,
        factorized_search_only=True,
    )

    assert "--factorized-search-only" in cmd
    assert "--factorized-rescue" not in cmd
    assert "--factorized-search-rescue" not in cmd
    assert cmd[cmd.index("--factorized-search-preset") + 1] == "fast"


def test_build_run_de_command_forwards_two_block_factorized_flag(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd = rb.build_run_de_command(
        _problem(order=2),
        [csv_path],
        tmp_path,
        fast=True,
        rescue=True,
        factorized_two_block_shared_coord="always",
    )

    assert "--factorized-two-block-shared-coord" in cmd
    assert cmd[cmd.index("--factorized-two-block-shared-coord") + 1] == "always"


def test_build_run_de_command_forwards_de_coe_mode(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd = rb.build_run_de_command(
        _problem(order=1),
        [csv_path],
        tmp_path,
        fast=True,
        rescue=True,
        de_coe_mode="reservoir",
        de_coe_csr_on_ties=True,
        de_coe_reservoir_scouts=2,
    )

    assert "--de-coe-mode" in cmd
    assert cmd[cmd.index("--de-coe-mode") + 1] == "reservoir"
    assert "--de-coe-csr-on-ties" in cmd
    assert "--de-coe-reservoir-scouts" in cmd
    assert cmd[cmd.index("--de-coe-reservoir-scouts") + 1] == "2"


def test_build_run_de_command_forwards_factorized_de_whole_rhs_policy(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd = rb.build_run_de_command(
        _problem(order=1),
        [csv_path],
        tmp_path,
        fast=True,
        rescue=False,
        factorized_de=True,
        factorized_de_whole_rhs="never",
        factorized_de_typed_lanes="never",
        factorized_de_typed_lane_workers=4,
        factorized_search_de_refine_mode="off",
        factorized_search_max_attempts=1,
    )

    assert "--factorized-de-whole-rhs" in cmd
    assert cmd[cmd.index("--factorized-de-whole-rhs") + 1] == "never"
    assert "--factorized-de-typed-lanes" in cmd
    assert cmd[cmd.index("--factorized-de-typed-lanes") + 1] == "never"
    assert "--factorized-de-typed-lane-workers" in cmd
    assert cmd[cmd.index("--factorized-de-typed-lane-workers") + 1] == "4"
    assert "--factorized-search-de-refine-mode" in cmd
    assert cmd[cmd.index("--factorized-search-de-refine-mode") + 1] == "off"
    assert "--factorized-search-max-attempts" in cmd
    assert cmd[cmd.index("--factorized-search-max-attempts") + 1] == "1"


def test_build_run_de_command_forwards_factorized_search_integrate_topk(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd = rb.build_run_de_command(
        _problem(order=1),
        [csv_path],
        tmp_path,
        fast=True,
        rescue=False,
        factorized_search_only=True,
        factorized_search_integrate_topk=0,
    )

    assert "--factorized-search-integrate-topk" in cmd
    assert cmd[cmd.index("--factorized-search-integrate-topk") + 1] == "0"


def test_build_run_de_command_forwards_direct_generator_witness_topk(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd = rb.build_run_de_command(
        _problem(order=2),
        [csv_path],
        tmp_path,
        fast=True,
        rescue=False,
        factorized_de=True,
        factorized_search_direct_generator_witness_topk=4,
    )

    assert "--factorized-search-direct-generator-witness-topk" in cmd
    assert cmd[cmd.index("--factorized-search-direct-generator-witness-topk") + 1] == "4"


def test_committee_adjudication_uses_committee_choice_without_engine_bias():
    candidates = [
        (
            "stlsq",
            {
                "proposal_id": "stlsq_heavy",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "ic0", "nrmse": 0.01}],
                "complexity": 10.0,
                "discovered_order": 1,
                "canonical_equation": "u_x + u = 0",
            },
        ),
        (
            "factorized_search",
            {
                "proposal_id": "fss_simple",
                "engine": "factorized_search",
                "status": "PASS",
                "traj_scores": [{"traj_id": "ic0", "nrmse": 0.01}],
                "complexity": 1.0,
                "discovered_order": 1,
                "canonical_equation": "u_x + u = 0",
            },
        ),
    ]

    legacy_engine, _ = rb._choose_rollout_candidate(candidates, fallback_engine="stlsq")
    decision = rb.run_de_committee_audit(
        [],
        rollout_candidates=[row for _, row in candidates],
        selected_engine="stlsq",
        run_compile_domain=False,
    ).to_dict()
    committee_engine, _ = rb._committee_selected_rollout_candidate(decision, candidates)

    assert legacy_engine == "stlsq"
    assert committee_engine == "factorized_search"


def test_rollout_choice_prefers_clean_factorized_when_nrmse_is_comparable():
    candidates = [
        (
            "factorized",
            {
                "engine": "factorized",
                "status": "PASS",
                "traj_scores": [{"traj_id": "ic0", "nrmse": 0.020}],
                "discovered_order": 2,
            },
        ),
        (
            "factorized_search",
            {
                "engine": "factorized_search",
                "source_lane": "factorized_search",
                "status": "PASS",
                "traj_scores": [{"traj_id": "ic0", "nrmse": 0.012}],
                "discovered_order": 2,
            },
        ),
    ]

    engine, row = rb._choose_rollout_candidate(candidates)

    assert engine == "factorized"
    assert row["engine"] == "factorized"


def test_rollout_choice_allows_broad_fss_when_materially_better():
    candidates = [
        (
            "factorized",
            {
                "engine": "factorized",
                "status": "PASS",
                "traj_scores": [{"traj_id": "ic0", "nrmse": 0.020}],
                "discovered_order": 2,
            },
        ),
        (
            "factorized_search",
            {
                "engine": "factorized_search",
                "source_lane": "factorized_search",
                "status": "PASS",
                "traj_scores": [{"traj_id": "ic0", "nrmse": 0.004}],
                "discovered_order": 2,
            },
        ),
    ]

    engine, row = rb._choose_rollout_candidate(candidates)

    assert engine == "factorized_search"
    assert row["traj_scores"][0]["nrmse"] == pytest.approx(0.004)


def test_factorized_search_rollout_skips_domain_rejected_candidate(monkeypatch):
    def _unexpected_rhs(_candidate):
        raise AssertionError("domain-rejected candidates should not be materialized")

    monkeypatch.setattr(rb, "factorized_search_report_to_rhs_callable", _unexpected_rhs)

    out = rb._evaluate_factorized_search_candidate_rollout(
        {
            "engine": "factorized_search",
            "order": 1,
            "canonical_equation": "bad",
            "mapping": {"_domain_projection": {"enabled": True, "ok": False}},
        },
        probe_runs=[],
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_traj_time_budget_s=None,
        sim_validate_blowup_factor=100.0,
        sim_validate_blowup_abs=1.0e6,
    )

    assert out["status"] == "ERROR"
    assert "domain-rejected" in out["message"]
    assert out["traj_scores"] == []


def test_factorized_search_rollout_skips_structurally_rejected_candidate(monkeypatch):
    def _unexpected_rhs(_candidate):
        raise AssertionError("structurally rejected candidates should not be materialized")

    monkeypatch.setattr(rb, "factorized_search_report_to_rhs_callable", _unexpected_rhs)

    out = rb._evaluate_factorized_search_candidate_rollout(
        {
            "engine": "factorized_search",
            "order": 1,
            "canonical_equation": "log(0)",
            "structural_ok": False,
            "structural_hard_reject": True,
            "structural_reasons": ["log_nonpositive_constant"],
        },
        probe_runs=[],
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_traj_time_budget_s=None,
        sim_validate_blowup_factor=100.0,
        sim_validate_blowup_abs=1.0e6,
    )

    assert out["status"] == "ERROR"
    assert "structurally unsafe" in out["message"]
    assert out["traj_scores"] == []


def test_csr_on_ties_skips_clear_committee_winner(monkeypatch):
    candidates = [
        ("stlsq", {"engine": "stlsq", "status": "PASS", "traj_scores": [{"nrmse": 0.0}], "discovered_order": 1}),
        (
            "factorized_search",
            {
                "engine": "factorized_search",
                "status": "FAIL",
                "traj_scores": [{"nrmse": 1.0}],
                "candidate_rank": 0,
                "discovered_order": 1,
            },
        ),
    ]
    decision = rb.run_de_committee_audit(
        [],
        rollout_candidates=[row for _, row in candidates],
        selected_engine="stlsq",
        run_compile_domain=False,
    ).to_dict()

    def _unexpected_refine(*args, **kwargs):
        raise AssertionError("CSR should not run for a clear winner")

    monkeypatch.setattr(rb, "refine_factorized_search_candidate_from_runs", _unexpected_refine)
    out_candidates, diag = rb._maybe_run_csr_on_tied_factorized_search(
        committee_decision=decision,
        current_candidates=candidates,
        factorized_search_shortlist=[{"engine": "factorized_search", "candidate_rank": 0}],
        fit_runs=[],
        probe_runs=[],
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_traj_time_budget_s=1.0,
        sim_validate_blowup_factor=10.0,
        sim_validate_blowup_abs=1.0e3,
        enabled=True,
    )

    assert out_candidates == candidates
    assert diag["invoked"] is False
    assert diag["reason"] == "clear_committee_winner"


def test_csr_on_ties_refines_bounded_factorized_search_survivors(monkeypatch):
    candidates = [
        ("stlsq", {"engine": "stlsq", "status": "PASS", "traj_scores": [{"nrmse": 0.01}], "discovered_order": 1}),
        (
            "factorized_search",
            {
                "engine": "factorized_search",
                "status": "PASS",
                "traj_scores": [{"nrmse": 0.0104}],
                "candidate_rank": 0,
                "discovered_order": 1,
                "canonical_equation": "fss0",
            },
        ),
        (
            "factorized_search",
            {
                "engine": "factorized_search",
                "status": "PASS",
                "traj_scores": [{"nrmse": 0.0105}],
                "candidate_rank": 1,
                "discovered_order": 1,
                "canonical_equation": "fss1",
            },
        ),
        (
            "factorized_search",
            {
                "engine": "factorized_search",
                "status": "PASS",
                "traj_scores": [{"nrmse": 0.0106}],
                "candidate_rank": 2,
                "discovered_order": 1,
                "canonical_equation": "fss2",
            },
        ),
    ]
    decision = rb.run_de_committee_audit(
        [],
        rollout_candidates=[row for _, row in candidates],
        selected_engine="stlsq",
        config={"tolerance_nrmse": 1.0e-3},
        run_compile_domain=False,
    ).to_dict()
    shortlist = [
        {"engine": "factorized_search", "candidate_rank": 0, "expr_ast": ["var", 0], "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]}},
        {"engine": "factorized_search", "candidate_rank": 1, "expr_ast": ["var", 0], "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]}},
        {"engine": "factorized_search", "candidate_rank": 2, "expr_ast": ["var", 0], "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]}},
    ]
    seen_trials: list[int] = []

    def _fake_refine(candidate, **kwargs):
        seen_trials.append(int(kwargs["max_trials"]))
        return {
            "accepted": True,
            "candidate": {**candidate, "de_coe_csr_refined": True},
            "base_probe_mse": 1.0,
            "refined_probe_mse": 0.1,
            "trials_used": 3,
        }

    def _fake_rollout(candidate, **kwargs):
        assert candidate["de_coe_csr_refined"] is True
        return {
            "engine": "factorized_search",
            "status": "PASS",
            "traj_scores": [{"nrmse": 0.005}],
            "candidate_rank": candidate["candidate_rank"],
            "discovered_order": 1,
            "canonical_equation": "csr",
        }

    monkeypatch.setattr(rb, "refine_factorized_search_candidate_from_runs", _fake_refine)
    monkeypatch.setattr(rb, "_evaluate_factorized_search_candidate_rollout", _fake_rollout)

    out_candidates, diag = rb._maybe_run_csr_on_tied_factorized_search(
        committee_decision=decision,
        current_candidates=candidates,
        factorized_search_shortlist=shortlist,
        fit_runs=[object()],
        probe_runs=[object()],
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_traj_time_budget_s=1.0,
        sim_validate_blowup_factor=10.0,
        sim_validate_blowup_abs=1.0e3,
        enabled=True,
        max_candidates=2,
        max_trials=5,
    )

    assert len(seen_trials) == 2
    assert seen_trials == [5, 5]
    assert diag["invoked"] is True
    assert diag["attempted"] == 2
    assert diag["accepted"] == 2
    assert diag["validated"] == 2
    assert len(out_candidates) == len(candidates) + 2


def test_reservoir_scout_slates_merge_namespaced_support():
    main = [
        {
            "proposal_id": "main:first",
            "engine": "stlsq",
            "role_signature": "library",
            "canonical_key": "residual:du+u",
            "order": 1,
            "x_axis": 0,
            "rhs_payload": {"engine": "stlsq", "order": 1, "x_axis": 0, "canonical_equation": "u_x + u = 0"},
            "canonical_equation": "u_x + u = 0",
            "support": {"support_count": 1, "sources": ["first_line"], "engines": ["stlsq"]},
            "provenance": {"source_id": "first_line"},
        }
    ]
    scout = [
        {
            "proposal_id": "scout:first",
            "engine": "stlsq",
            "role_signature": "library",
            "canonical_key": "residual:du+u",
            "order": 1,
            "x_axis": 0,
            "rhs_payload": {"engine": "stlsq", "order": 1, "x_axis": 0, "canonical_equation": "-u_x - u = 0"},
            "canonical_equation": "-u_x - u = 0",
            "support": {"support_count": 1, "sources": ["first_line"], "engines": ["stlsq"]},
            "provenance": {"source_id": "first_line"},
        }
    ]

    merged, diag = rb._merge_reservoir_scout_slates(
        main,
        [{"namespace": "reservoir_scout0", "proposal_slate": scout}],
    )

    assert len(merged) == 1
    assert diag["merged_proposal_count"] == 1
    assert diag["max_support_count"] == 2
    assert merged[0]["support"]["sources"] == ["first_line", "reservoir_scout0:first_line"]


def test_reservoir_evaluates_bounded_new_scout_candidates(monkeypatch):
    proposal_slate = [
        {
            "proposal_id": "main",
            "engine": "stlsq",
            "role_signature": "library",
            "canonical_key": "residual:du+u",
            "order": 1,
            "x_axis": 0,
            "rhs_payload": {"engine": "stlsq", "order": 1, "x_axis": 0, "canonical_equation": "u_x + u = 0"},
            "canonical_equation": "u_x + u = 0",
            "support": {"support_count": 1, "sources": ["first_line"], "engines": ["stlsq"]},
        },
        {
            "proposal_id": "reservoir_scout0:typed",
            "engine": "factorized",
            "role_signature": "typed:x_coeff_on_u",
            "canonical_key": "residual:du+u*x",
            "order": 1,
            "x_axis": 0,
            "rhs_payload": {"engine": "factorized", "order": 1, "x_axis": 0, "canonical_equation": "u_x + x*u = 0"},
            "canonical_equation": "u_x + x*u = 0",
            "support": {
                "support_count": 2,
                "sources": ["reservoir_scout0:factorized_rescue", "reservoir_scout1:factorized_rescue"],
                "engines": ["factorized"],
            },
        },
        {
            "proposal_id": "reservoir_scout0:stlsq",
            "engine": "stlsq",
            "role_signature": "library",
            "canonical_key": "residual:du+x",
            "order": 1,
            "x_axis": 0,
            "rhs_payload": {"engine": "stlsq", "order": 1, "x_axis": 0, "canonical_equation": "u_x + x = 0"},
            "canonical_equation": "u_x + x = 0",
            "support": {"support_count": 1, "sources": ["reservoir_scout0:first_line"], "engines": ["stlsq"]},
        },
    ]
    current = [
        (
            "stlsq",
            {
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"nrmse": 0.01}],
                "canonical_equation": "u_x + u = 0",
                "discovered_order": 1,
            },
        )
    ]
    seen: list[str] = []

    def _fake_factorized(candidate, **kwargs):
        seen.append(str(candidate["canonical_equation"]))
        return {
            "engine": "factorized",
            "status": "PASS",
            "traj_scores": [{"nrmse": 0.005}],
            "canonical_equation": candidate["canonical_equation"],
            "discovered_order": 1,
        }

    def _unexpected_library(candidate, **kwargs):
        raise AssertionError("max_candidates=1 should keep only the higher-support scout proposal")

    monkeypatch.setattr(rb, "_evaluate_factorized_candidate_rollout", _fake_factorized)
    monkeypatch.setattr(rb, "_evaluate_library_candidate_rollout", _unexpected_library)

    out_candidates, diag = rb._evaluate_reservoir_scout_proposals(
        proposal_slate,
        current,
        probe_runs=[object()],
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_traj_time_budget_s=1.0,
        sim_validate_blowup_factor=10.0,
        sim_validate_blowup_abs=1.0e3,
        max_candidates=1,
    )

    assert seen == ["u_x + x*u = 0"]
    assert diag["validated"] == 1
    assert diag["candidate_ids"] == ["reservoir_scout0:typed"]
    assert len(out_candidates) == 2
    assert out_candidates[-1][1]["reservoir_scout"] is True


def test_build_run_de_command_honors_explicit_factorized_search_preset(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd = rb.build_run_de_command(
        _problem(order=1),
        [csv_path],
        tmp_path,
        fast=True,
        rescue=False,
        factorized_search_only=True,
        factorized_search_preset="compositional",
    )

    assert cmd[cmd.index("--factorized-search-preset") + 1] == "compositional"


def test_build_run_de_command_honors_explicit_compositional_fast_preset(tmp_path: Path):
    csv_path = tmp_path / "de999_ic0.csv"
    cmd = rb.build_run_de_command(
        _problem(order=1),
        [csv_path],
        tmp_path,
        fast=True,
        rescue=False,
        factorized_search_only=True,
        factorized_search_preset="compositional_fast",
    )

    assert cmd[cmd.index("--factorized-search-preset") + 1] == "compositional_fast"


def test_build_run_de_command_includes_units_when_dims_exist(tmp_path: Path):
    csv_path = tmp_path / "de000_ic0.csv"
    problem = ProblemDef(
        id="000",
        order=1,
        indep_var="t",
        dep_var="u",
        equation="du/dt=-lambda*u",
        description="dimmed",
        feynman_ref="-",
        params=["lambda"],
        param_ranges=["[0.1,2.0]"],
        ic_type="decay",
    )
    cmd = rb.build_run_de_command(problem, [csv_path], tmp_path, fast=True, rescue=True)

    assert "--ignore_units" not in cmd
    assert cmd[cmd.index("--y_units") + 1] == "[0]"
    assert cmd[cmd.index("--x_units") + 1] == "[[1]]"
    assert cmd[cmd.index("--units_basis") + 1] == "D"
    local_consts = json.loads(cmd[cmd.index("--local_consts") + 1])
    assert local_consts["lambda"] == [-1]
    assert local_consts["inv_lambda"] == [1]
    assert local_consts["lambda_sq"] == [-2]
    assert local_consts["inv_lambda_sq"] == [2]


def test_local_const_dims_include_bounded_composite_coefficients():
    canonical_dims = rb.get_canonical_problem_dims("131")
    assert canonical_dims is not None

    local_consts = rb._local_const_dims_for_units(canonical_dims)

    assert local_consts["mu_mul_omega_sq_over_T"] == [-2, 0, 0, 0]


def test_build_run_de_command_no_dims_omits_unit_payloads(tmp_path: Path):
    csv_path = tmp_path / "de000_ic0.csv"
    problem = ProblemDef(
        id="000",
        order=1,
        indep_var="t",
        dep_var="u",
        equation="du/dt=-lambda*u",
        description="dimmed",
        feynman_ref="-",
        params=["lambda"],
        param_ranges=["[0.1,2.0]"],
        ic_type="decay",
    )
    cmd = rb.build_run_de_command(
        problem,
        [csv_path],
        tmp_path,
        fast=True,
        rescue=False,
        use_dims=False,
    )

    assert "--ignore_units" in cmd
    assert "--y_units" not in cmd
    assert "--x_units" not in cmd
    assert "--units_basis" not in cmd
    assert "--local_consts" not in cmd


def test_de_candidate_eval_shortlist_from_meta_sorts_and_caps():
    rows = [
        {"canonical_equation": "rank2", "candidate_rank": 2, "pointwise_score": 0.2, "probe_rms": 0.0},
        {"canonical_equation": "rank1", "candidate_rank": 1, "pointwise_score": 0.1, "probe_rms": 0.2},
        {"canonical_equation": "rank0", "candidate_rank": 0, "pointwise_score": 0.1, "probe_rms": 0.1},
        {"canonical_equation": "rank3", "candidate_rank": 3, "pointwise_score": 0.3, "probe_rms": 0.0},
        {"canonical_equation": "rank4", "candidate_rank": 4, "pointwise_score": 0.4, "probe_rms": 0.0},
    ]
    meta = {"de_candidate_eval": {"status": "OK", "rollout_shortlist": rows}}

    default = rb._de_candidate_eval_shortlist_from_meta(meta)
    capped = rb._de_candidate_eval_shortlist_from_meta(meta, max_candidates=2)
    errored = rb._de_candidate_eval_shortlist_from_meta({"de_candidate_eval": {"status": "ERROR"}})

    assert [row["canonical_equation"] for row in default] == ["rank0", "rank1", "rank2", "rank3"]
    assert [row["canonical_equation"] for row in capped] == ["rank0", "rank1"]
    assert errored == []


def test_de_candidate_eval_shortlist_filters_unsafe_implicit_rationals():
    rows = [
        {
            "canonical_equation": "safe",
            "kind": "assembled_implicit_rational",
            "candidate_family": "implicit_rational",
            "candidate_rank": 0,
            "pointwise_score": 0.0,
            "probe_rms": 0.0,
            "denominator_safety": {"safe": True},
        },
        {
            "canonical_equation": "unsafe",
            "kind": "assembled_implicit_rational",
            "candidate_family": "implicit_rational",
            "candidate_rank": 1,
            "pointwise_score": -1.0,
            "probe_rms": 0.0,
            "denominator_safety": {"safe": False, "reason": "near_zero"},
        },
    ]

    out = rb._de_candidate_eval_shortlist_from_meta(
        {"de_candidate_eval": {"status": "OK", "rollout_shortlist": rows}}
    )

    assert [row["canonical_equation"] for row in out] == ["safe"]


def test_evaluate_de_candidate_eval_rollout_uses_materialized_payload(monkeypatch):
    seen: dict[str, Any] = {}

    def _fake_library_rollout(candidate, **kwargs):
        seen["candidate"] = dict(candidate)
        return {
            "engine": "stlsq",
            "status": "PASS",
            "message": "NRMSE mean=0 max=0",
            "traj_scores": [{"traj_id": "ic0", "nrmse": 0.0}],
            "discovered_order": int(candidate["order"]),
            "canonical_equation": str(candidate["canonical_equation"]),
        }

    monkeypatch.setattr(rb, "_evaluate_library_candidate_rollout", _fake_library_rollout)

    out = rb._evaluate_de_candidate_eval_rollout(
        {
            "candidate_rank": 3,
            "candidate_family": "implicit_rational",
            "source_rank": 1,
            "pointwise_score": -12.5,
            "denominator_safety": {"safe": True},
            "validation_candidate": {
                "kind": "assembled_implicit_rational",
                "order": 1,
                "x_axis": 0,
                "canonical_equation": "u_x + u/(1+x) = 0",
            },
        },
        probe_runs=[],
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_traj_time_budget_s=1.0,
        sim_validate_blowup_factor=10.0,
        sim_validate_blowup_abs=1.0e3,
    )

    assert seen["candidate"]["kind"] == "assembled_implicit_rational"
    assert out["engine"] == "de_candidate_eval"
    assert out["status"] == "PASS"
    assert out["candidate_rank"] == 3
    assert out["candidate_family"] == "implicit_rational"
    assert out["source_rank"] == 1
    assert out["pointwise_score"] == -12.5
    assert out["denominator_safety"] == {"safe": True}
    assert out["canonical_equation"] == "u_x + u/(1+x) = 0"


def test_evaluate_de_candidate_eval_rollout_skips_unsafe_implicit_rational(monkeypatch):
    def _unexpected_library_rollout(candidate, **kwargs):
        raise AssertionError("unsafe implicit rational should not be simulation-validated")

    monkeypatch.setattr(rb, "_evaluate_library_candidate_rollout", _unexpected_library_rollout)

    out = rb._evaluate_de_candidate_eval_rollout(
        {
            "candidate_rank": 4,
            "candidate_family": "implicit_rational",
            "source_rank": 2,
            "pointwise_score": -99.0,
            "denominator_safety": {"safe": False, "reason": "near_zero"},
            "kind": "assembled_implicit_rational",
            "order": 1,
            "x_axis": 0,
            "canonical_equation": "u_x + u/0 = 0",
        },
        probe_runs=[],
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_traj_time_budget_s=1.0,
        sim_validate_blowup_factor=10.0,
        sim_validate_blowup_abs=1.0e3,
    )

    assert out["engine"] == "de_candidate_eval"
    assert out["status"] == "ERROR"
    assert out["candidate_rank"] == 4
    assert out["denominator_safety"] == {"safe": False, "reason": "near_zero"}


def test_rollout_domain_safety_catches_typed_assembly_zero_crossing(tmp_path: Path):
    csv_path = tmp_path / "de300_ic5.csv"
    meta_path = tmp_path / "de300_ic5.meta.json"
    csv_path.write_text("y,x0\n0.10,0.00\n-0.02,0.05\n-0.10,0.10\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic5",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=0.1,
        v0=0.0,
    )
    u = {"type": "atom", "kind": "u"}
    du = {"type": "atom", "kind": "du"}
    singular = {
        "type": "mul",
        "left": {"type": "pow", "base": u, "exponent": -1.0},
        "right": du,
    }
    candidate = {
        "lane": "two_block_typed_assembly",
        "order": 2,
        "validation_candidate": {
            "order": 2,
            "x_axis": 0,
            "term_asts_json": [singular],
            "canonical_equation": "u_x0x0 + (u ** -1) * u_x0 = 0",
        },
    }

    report = rb._rollout_domain_safety_report(candidate, [run])

    assert report["safe"] is False
    assert report["reason"] == "rollout_domain_violation"
    assert report["violations"][0]["traj_id"] == "ic5"
    assert report["violations"][0]["kind"] == "negative_power"
    assert report["violations"][0]["reason"] == "denominator_crosses_zero"


def test_evaluate_factorized_rollout_skips_domain_unsafe_candidate(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "de300_ic5.csv"
    meta_path = tmp_path / "de300_ic5.meta.json"
    csv_path.write_text("y,x0\n0.10,0.00\n-0.02,0.05\n-0.10,0.10\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic5",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=0.1,
        v0=0.0,
    )

    def _unexpected_rhs(candidate):
        raise AssertionError("domain-unsafe factorized candidate should not reach RHS compilation")

    monkeypatch.setattr(rb, "library_candidate_to_rhs_callable", _unexpected_rhs)

    u = {"type": "atom", "kind": "u"}
    du = {"type": "atom", "kind": "du"}
    out = rb._evaluate_factorized_candidate_rollout(
        {
            "lane": "two_block_typed_assembly",
            "order": 2,
            "canonical_equation": "u_x0x0 + (u ** -1) * u_x0 = 0",
            "validation_candidate": {
                "order": 2,
                "x_axis": 0,
                "term_asts_json": [
                    {
                        "type": "mul",
                        "left": {"type": "pow", "base": u, "exponent": -1.0},
                        "right": du,
                    }
                ],
            },
        },
        probe_runs=[run],
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_traj_time_budget_s=1.0,
        sim_validate_blowup_factor=10.0,
        sim_validate_blowup_abs=1.0e3,
    )

    assert out["engine"] == "factorized"
    assert out["status"] == "ERROR"
    assert out["message"] == "Skipped rollout-domain-unsafe factorized candidate before integration"
    assert out["rollout_domain_safety"]["safe"] is False


def test_factorized_shortlist_adds_pruned_variant_for_tiny_singular_terms():
    u = {"type": "atom", "kind": "u"}
    du = {"type": "atom", "kind": "du"}

    def const(value: float) -> dict[str, Any]:
        return {"type": "const", "value": float(value)}

    def mul(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        return {"type": "mul", "left": left, "right": right}

    def add(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        return {"type": "add", "left": left, "right": right}

    pole_term = mul(const(-3.6228e-4), mul({"type": "pow", "base": u, "exponent": -1.0}, du))
    small_quadratic = mul(const(-9.35447e-3), {"type": "pow", "base": u, "exponent": 2.0})
    damping = mul(const(0.550991), du)
    stiffness = mul(const(3.00881), u)
    forcing = const(-1.25201)
    ast = add(add(add(add(pole_term, small_quadratic), damping), stiffness), forcing)
    candidate = {
        "lane": "two_block_typed_assembly",
        "candidate_rank": 0,
        "canonical_equation": "u_x0x0 + de300-noisy = 0",
        "validation_candidate": {
            "order": 2,
            "x_axis": 0,
            "term_asts_json": [ast],
            "canonical_equation": "u_x0x0 + de300-noisy = 0",
        },
    }

    shortlist = rb._factorized_shortlist_from_candidate(candidate)
    variants = [row for row in shortlist if row.get("parsimony_pruned")]

    assert len(variants) == 1
    pruned = variants[0]
    assert pruned["candidate_rank"] == 10_000
    assert len(pruned["parsimony_pruned_terms"]) == 2
    assert any(row["domain_sensitive"] for row in pruned["parsimony_pruned_terms"])
    assert "(u ** -1" not in pruned["canonical_equation"]
    assert "(u ** 2" not in pruned["canonical_equation"]
    assert "u_x0" in pruned["canonical_equation"]
    assert "+ u)" in pruned["canonical_equation"] or "* u)" in pruned["canonical_equation"]


def test_run_hybrid_engine_prefers_rollout_over_internal_factorized_search_selection(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    seen: dict[str, list[str]] = {}

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        seen["cmd"] = list(cmd)
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "factorized_search",
                "rescue_attempted": True,
                "rescue_triggered": True,
                "first_line": {
                    "engine": "stlsq",
                    "order": 1,
                    "x_axis": 0,
                    "terms": ["U()"],
                    "coefficients": [-1.0],
                    "validation_candidate": {
                        "order": 1,
                        "x_axis": 0,
                        "coefficients": [-1.0],
                        "term_asts_json": [{"type": "atom", "kind": "u", "var_idxs": [], "kwargs": {}}],
                    },
                    "canonical_equation": "u_x - u = 0",
                },
                "factorized_search_rescue": {
                    "engine": "factorized_search",
                    "order": 1,
                    "expr_ast": ["var", 1],
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                    "canonical_equation": "u_x - (u) = 0",
                    "diagnostics": {
                        "report": {
                            "include_x": True,
                            "constants": [],
                        }
                    },
                },
                "selected": {
                    "engine": "factorized_search",
                    "order": 1,
                    "expr_ast": ["var", 1],
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                    "canonical_equation": "u_x - (u) = 0",
                    "diagnostics": {
                        "report": {
                            "include_x": True,
                            "constants": [],
                        }
                    },
                },
                "canonical_equation": "u_x - (u) = 0",
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)
    monkeypatch.setattr(rb, "library_candidate_to_rhs_callable", lambda report: (1, "lib_rhs"))
    monkeypatch.setattr(rb, "factorized_search_report_to_rhs_callable", lambda report: (1, "residual_basin_rhs"))

    def _fake_validate(*args, **kwargs):
        rhs_fn = kwargs["rhs_fn"]
        if rhs_fn == "lib_rhs":
            return "PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic0", "nrmse": 0.0}]
        return "FAIL", "NRMSE mean=10 max=10", [{"traj_id": "ic0", "nrmse": 10.0}]

    monkeypatch.setattr(rb, "validate_by_simulation", _fake_validate)

    out = rb._run_hybrid_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        de_coe_mode="audit",
    )

    assert out["engine"] == "hybrid"
    assert out["selected_engine"] == "stlsq"
    assert out["internal_selected_engine"] == "factorized_search"
    assert out["internal_selected_engine_mismatch"] is True
    assert out["status"] == "PASS"
    assert out["rescued_additional"] is False
    assert out["first_line_status"] == "PASS"
    assert out["canonical_equation"] == "u_x - u = 0"
    assert out["committee_selected_engine"] == "stlsq"
    assert out["internal_selected_engine_committee_mismatch"] is True
    assert seen["cmd"][seen["cmd"].index("--factorized-search-integrate-topk") + 1] == "0"

    adjudicated = rb._run_hybrid_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        de_coe_mode="adjudicate",
    )

    assert adjudicated["selected_engine"] == "stlsq"
    assert adjudicated["selection_mode"] == "committee_adjudicate"
    assert adjudicated["committee_adjudicated"] is True
    assert adjudicated["committee_selected_engine"] == "stlsq"


def test_run_hybrid_engine_can_choose_factorized_search_by_rollout(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "stlsq",
                "rescue_attempted": True,
                "rescue_triggered": True,
                "first_line": {
                    "engine": "stlsq",
                    "order": 1,
                    "x_axis": 0,
                    "validation_candidate": {
                        "order": 1,
                        "x_axis": 0,
                        "coefficients": [-1.0],
                        "term_asts_json": [{"type": "atom", "kind": "u", "var_idxs": [], "kwargs": {}}],
                    },
                    "canonical_equation": "u_x - u = 0",
                },
                "factorized_search_rescue": {
                    "engine": "factorized_search",
                    "order": 1,
                    "expr_ast": ["var", 1],
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                    "canonical_equation": "u_x - (u) = 0",
                    "diagnostics": {"report": {"include_x": True, "constants": []}},
                },
                "selected": {
                    "engine": "stlsq",
                    "order": 1,
                    "canonical_equation": "u_x - u = 0",
                },
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)
    monkeypatch.setattr(rb, "library_candidate_to_rhs_callable", lambda report: (1, "lib_rhs"))
    monkeypatch.setattr(rb, "factorized_search_report_to_rhs_callable", lambda report: (1, "residual_basin_rhs"))

    def _fake_validate(*args, **kwargs):
        rhs_fn = kwargs["rhs_fn"]
        if rhs_fn == "lib_rhs":
            return "FAIL", "NRMSE mean=2 max=2", [{"traj_id": "ic0", "nrmse": 2.0}]
        return "PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic0", "nrmse": 0.0}]

    monkeypatch.setattr(rb, "validate_by_simulation", _fake_validate)

    out = rb._run_hybrid_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert out["selected_engine"] == "factorized_search"
    assert out["internal_selected_engine"] == "stlsq"
    assert out["internal_selected_engine_mismatch"] is True
    assert out["status"] == "PASS"
    assert out["rescued_additional"] is True


def test_run_factorized_de_validates_selected_direct_residual_candidate(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    selected_candidate = {
        "engine": "factorized_search",
        "kind": "factorized",
        "order": 1,
        "x_axis": 0,
        "feature_names": ["u"],
        "expr_ast": ["var", 0],
        "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
        "canonical_equation": "u_x0 - (-u) = 0",
    }

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "factorized_search",
                "internal_selected_engine": "factorized_search",
                "factorized_de": True,
                "factorized_de_diagnostics": {
                    "selected_lane": "direct_residual_fss",
                    "selected_engine": "factorized_search",
                    "direct_residual_attempted": True,
                    "direct_residual_probe_rms": 1.0e-6,
                    "typed_lanes_policy": "never",
                    "typed_lanes_attempted": False,
                    "coefficient_dim_mode": "inferred_outer",
                    "factorized_search_attempted": False,
                    "whole_rhs_policy": {"policy": "auto", "run": False, "reason": "typed_probe_rms_pass"},
                },
                "factorized_rescue": {},
                "factorized_search_rescue": {},
                "selected": {
                    **selected_candidate,
                    "shortlist": [{**selected_candidate, "shortlist_rank": 0}],
                },
                "canonical_equation": "u_x0 - (-u) = 0",
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)
    monkeypatch.setattr(rb, "factorized_search_report_to_rhs_callable", lambda report: (1, "direct_rhs"))
    monkeypatch.setattr(
        rb,
        "validate_by_simulation",
        lambda *args, **kwargs: ("PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic0", "nrmse": 0.0}]),
    )

    out = rb._run_hybrid_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        factorized_de=True,
    )

    assert out["selected_engine"] == "factorized_search"
    assert out["selected_lane"] == "direct_residual_fss"
    # The rollout pool contains the parent selected candidate plus its
    # serialized shortlist twin.
    assert out["direct_residual_shortlist_size"] == 2
    assert out["direct_residual_validated_candidates"] == 2
    assert out["direct_residual_status"] == "PASS"
    assert out["status"] == "PASS"


def test_run_factorized_de_validates_selected_regularized_implicit_candidate(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    selected_candidate = {
        "engine": "factorized_search",
        "kind": "factorized",
        "order": 2,
        "x_axis": 0,
        "candidate_rank": 0,
        "feature_names": ["x0", "u", "du"],
        "rhs_ast": "((-0.44 * u_x0) + (-2.4 * u))",
        "expr_ast": ["add", ["mul", ["const", -0.44], ["var", 2]], ["mul", ["const", -2.4], ["var", 1]]],
        "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
        "canonical_equation": "u_x0x0 + 0.44*u_x0 + 2.4*u = 0",
        "diagnostics": {
            "candidate_source": "regularized_implicit_residual",
            "domain_ok": True,
            "structural_ok": True,
        },
    }

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "factorized_search",
                "internal_selected_engine": "factorized_search",
                "factorized_de": True,
                "factorized_de_diagnostics": {
                    "selected_lane": "regularized_implicit_residual",
                    "selected_engine": "factorized_search",
                    "direct_residual_attempted": True,
                    "direct_residual_probe_rms": 1.0e-3,
                    "typed_lanes_policy": "always",
                    "typed_lanes_attempted": False,
                    "coefficient_dim_mode": "regularized_implicit",
                    "factorized_search_attempted": False,
                    "whole_rhs_policy": {"policy": "never", "run": False, "reason": "policy_never"},
                },
                "factorized_rescue": {},
                "factorized_search_rescue": {},
                "selected": selected_candidate,
                "canonical_equation": selected_candidate["canonical_equation"],
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)
    monkeypatch.setattr(rb, "factorized_search_report_to_rhs_callable", lambda report: (2, "implicit_rhs"))
    monkeypatch.setattr(
        rb,
        "validate_by_simulation",
        lambda *args, **kwargs: ("PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic0", "nrmse": 0.0}]),
    )

    out = rb._run_hybrid_engine(
        _problem(order=2),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        factorized_de=True,
        factorized_de_typed_lanes="always",
        factorized_de_whole_rhs="never",
    )

    assert out["engine"] == "factorized_de"
    assert out["selected_engine"] == "factorized_search"
    assert out["selected_lane"] == "regularized_implicit_residual"
    assert out["direct_residual_shortlist_size"] == 1
    assert out["direct_residual_validated_candidates"] == 1
    assert out["direct_residual_status"] == "PASS"
    assert out["selected_shortlist_rank"] == 0
    assert out["status"] == "PASS"


def test_run_hybrid_engine_can_choose_factorized_by_rollout(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        factorized_row = {
            "engine": "factorized",
            "kind": "factorized_blocks",
            "order": 1,
            "x_axis": 0,
            "canonical_equation": "u_x + U() = 0",
            "candidate_rank": 0,
            "validation_candidate": {
                "order": 1,
                "x_axis": 0,
                "coefficients": [1.0],
                "term_asts_json": [{"type": "atom", "kind": "u", "var_idxs": [], "kwargs": {}}],
            },
        }
        payload = {
            "de_discovery": {
                "selected_engine": "stlsq",
                "factorized_attempted": True,
                "factorized_triggered": True,
                "first_line": {
                    "engine": "stlsq",
                    "order": 1,
                    "x_axis": 0,
                    "validation_candidate": {
                        "order": 1,
                        "x_axis": 0,
                        "coefficients": [-1.0],
                        "term_asts_json": [{"type": "atom", "kind": "u", "var_idxs": [], "kwargs": {}}],
                    },
                    "canonical_equation": "u_x - u = 0",
                },
                "factorized_rescue": {
                    "engine": "factorized",
                    "order": 1,
                    "canonical_equation": "u_x + U() = 0",
                    "internal_selected_shortlist_rank": 0,
                    "validation_candidate": factorized_row["validation_candidate"],
                    "shortlist": [factorized_row],
                },
                "factorized_search_rescue": {},
                "selected": {
                    "engine": "stlsq",
                    "order": 1,
                    "canonical_equation": "u_x - u = 0",
                },
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        rb,
        "library_candidate_to_rhs_callable",
        lambda candidate: (1, "factorized_rhs" if candidate.get("engine") == "factorized" else "lib_rhs"),
    )

    def _fake_validate(*args, **kwargs):
        rhs_fn = kwargs["rhs_fn"]
        if rhs_fn == "factorized_rhs":
            return "PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic0", "nrmse": 0.0}]
        return "FAIL", "NRMSE mean=2 max=2", [{"traj_id": "ic0", "nrmse": 2.0}]

    monkeypatch.setattr(rb, "validate_by_simulation", _fake_validate)

    out = rb._run_hybrid_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert out["selected_engine"] == "factorized"
    assert out["internal_selected_engine"] == "stlsq"
    assert out["internal_selected_engine_mismatch"] is True
    assert out["status"] == "PASS"
    assert out["rescued_additional"] is True


def test_run_factorized_search_only_engine_validates_selected_candidate(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    seen: dict[str, list[str]] = {}

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        seen["cmd"] = list(cmd)
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "factorized_search",
                "selected": {
                    "engine": "factorized_search",
                    "order": 1,
                    "expr_ast": ["var", 1],
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                    "canonical_equation": "u_x - (u) = 0",
                    "diagnostics": {"report": {"include_x": True, "constants": []}},
                },
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)
    monkeypatch.setattr(rb, "factorized_search_report_to_rhs_callable", lambda report: (1, "residual_basin_rhs"))
    monkeypatch.setattr(
        rb,
        "validate_by_simulation",
        lambda *args, **kwargs: ("PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic0", "nrmse": 0.0}]),
    )

    out = rb._run_factorized_search_only_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert out["engine"] == "factorized_search_only"
    assert out["selected_engine"] == "factorized_search"
    assert out["status"] == "PASS"
    assert seen["cmd"][seen["cmd"].index("--factorized-search-integrate-topk") + 1] == "0"


def test_run_factorized_search_only_engine_handles_empty_shortlist(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "factorized_search",
                "selected": {},
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    def _unexpected_validate(*args, **kwargs):
        raise AssertionError("empty shortlist should not enter rollout validation")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)
    monkeypatch.setattr(rb, "validate_by_simulation", _unexpected_validate)

    out = rb._run_factorized_search_only_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert out["engine"] == "factorized_search_only"
    assert out["status"] == "ERROR"
    assert out["factorized_search_shortlist_size"] == 0
    assert out["factorized_search_validated_candidates"] == 0
    assert "No factorized symbolic search candidates" in out["message"]


def test_run_factorized_search_only_engine_can_select_de_candidate_eval(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    def _fake_run_command(cmd, log_path, *, cwd=None, env=None):
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "factorized_search",
                "selected": {},
                "de_candidate_eval": {
                    "status": "OK",
                    "rollout_shortlist": [
                        {
                            "engine": "de_candidate_eval",
                            "kind": "assembled_implicit_rational",
                            "order": 1,
                            "x_axis": 0,
                            "canonical_equation": "u_x + u/(1+x) = 0",
                            "candidate_family": "implicit_rational",
                            "candidate_rank": 0,
                            "source_rank": 0,
                            "pointwise_score": -10.0,
                            "probe_rms": 0.0,
                            "denominator_safety": {"safe": True},
                            "coefficients": [1.0],
                            "term_asts_json": [{"type": "atom", "kind": "u", "var_idxs": [], "kwargs": {}}],
                        }
                    ],
                },
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    seen: dict[str, Any] = {}

    def _fake_rhs(candidate):
        seen["candidate"] = dict(candidate)
        return 1, "de_candidate_eval_rhs"

    def _fake_validate(*args, **kwargs):
        assert kwargs["rhs_fn"] == "de_candidate_eval_rhs"
        return "PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic0", "nrmse": 0.0}]

    monkeypatch.setattr(rb, "_run_command_to_log", _fake_run_command)
    monkeypatch.setattr(rb, "library_candidate_to_rhs_callable", _fake_rhs)
    monkeypatch.setattr(rb, "validate_by_simulation", _fake_validate)

    out = rb._run_factorized_search_only_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert seen["candidate"]["kind"] == "assembled_implicit_rational"
    assert out["engine"] == "factorized_search_only"
    assert out["selected_engine"] == "de_candidate_eval"
    assert out["status"] == "PASS"
    assert out["factorized_search_shortlist_size"] == 0
    assert out["factorized_search_validated_candidates"] == 0
    assert out["de_candidate_eval_shortlist_size"] == 1
    assert out["de_candidate_eval_validated_candidates"] == 1
    assert out["selected_candidate_family"] == "implicit_rational"
    assert out["canonical_equation"] == "u_x + u/(1+x) = 0"


def test_run_hybrid_engine_can_choose_later_factorized_search_shortlist_candidate(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "factorized_search",
                "rescue_attempted": True,
                "rescue_triggered": True,
                "first_line": {
                    "engine": "stlsq",
                    "order": 1,
                    "x_axis": 0,
                    "validation_candidate": {
                        "order": 1,
                        "x_axis": 0,
                        "coefficients": [-1.0],
                        "term_asts_json": [{"type": "atom", "kind": "u", "var_idxs": [], "kwargs": {}}],
                    },
                    "canonical_equation": "u_x - u = 0",
                },
                "factorized_search_rescue": {
                    "engine": "factorized_search",
                    "order": 1,
                    "expr_ast": ["var", 1],
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                    "canonical_equation": "u_x - (u) = 0",
                    "diagnostics": {"report": {"include_x": True, "constants": []}},
                    "shortlist": [
                        {
                            "order": 1,
                            "expr_ast": ["var", 1],
                            "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                            "canonical_equation": "u_x - (u) = 0",
                            "shortlist_rank": 0,
                            "include_x": True,
                            "constants_ordered": [],
                        },
                        {
                            "order": 1,
                            "expr_ast": ["var", 0],
                            "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                            "canonical_equation": "u_x - (x) = 0",
                            "shortlist_rank": 1,
                            "include_x": True,
                            "constants_ordered": [],
                        },
                    ],
                },
                "selected": {
                    "engine": "factorized_search",
                    "order": 1,
                    "expr_ast": ["var", 1],
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                    "canonical_equation": "u_x - (u) = 0",
                    "diagnostics": {"report": {"include_x": True, "constants": []}},
                },
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)

    # Key the fake RHS off the equation so the parent candidate (now included
    # in the rollout pool) behaves identically to its serialized rank-0 twin.
    def _fake_rhs(candidate):
        return 1, f"rhs::{candidate.get('canonical_equation', '')}"

    monkeypatch.setattr(rb, "library_candidate_to_rhs_callable", lambda report: (1, "lib_rhs"))
    monkeypatch.setattr(rb, "factorized_search_report_to_rhs_callable", _fake_rhs)

    def _fake_validate(*args, **kwargs):
        rhs_fn = kwargs["rhs_fn"]
        if rhs_fn == "lib_rhs":
            return "FAIL", "NRMSE mean=2 max=2", [{"traj_id": "ic0", "nrmse": 2.0}]
        if rhs_fn == "rhs::u_x - (u) = 0":
            return "FAIL", "NRMSE mean=1 max=1", [{"traj_id": "ic0", "nrmse": 1.0}]
        return "PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic0", "nrmse": 0.0}]

    monkeypatch.setattr(rb, "validate_by_simulation", _fake_validate)

    out = rb._run_hybrid_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_max_candidates=3,
    )

    assert out["selected_engine"] == "factorized_search"
    assert out["status"] == "PASS"
    assert out["selected_shortlist_rank"] == 1
    assert out["internal_selected_shortlist_rank_mismatch"] is True
    assert out["canonical_equation"] == "u_x - (x) = 0"


def test_run_factorized_search_only_engine_can_choose_later_shortlist_candidate(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        out_path = results_dir / f"{rb._derive_run_de_base_filename([csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "factorized_search",
                "selected": {
                    "engine": "factorized_search",
                    "order": 1,
                    "expr_ast": ["var", 1],
                    "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                    "canonical_equation": "u_x - (u) = 0",
                    "shortlist": [
                        {
                            "order": 1,
                            "expr_ast": ["var", 1],
                            "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                            "canonical_equation": "u_x - (u) = 0",
                            "shortlist_rank": 0,
                            "include_x": True,
                            "constants_ordered": [],
                        },
                        {
                            "order": 1,
                            "expr_ast": ["var", 0],
                            "mapping": {"kind": "poly", "coeffs": [0.0, -1.0]},
                            "canonical_equation": "u_x - (x) = 0",
                            "shortlist_rank": 1,
                            "include_x": True,
                            "constants_ordered": [],
                        },
                    ],
                    "diagnostics": {"report": {"include_x": True, "constants": []}},
                },
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)

    # Key the fake RHS off the equation so the parent candidate (now included
    # in the rollout pool) behaves identically to its serialized rank-0 twin.
    def _fake_rhs(candidate):
        return 1, f"rhs::{candidate.get('canonical_equation', '')}"

    monkeypatch.setattr(rb, "factorized_search_report_to_rhs_callable", _fake_rhs)

    def _fake_validate(*args, **kwargs):
        rhs_fn = kwargs["rhs_fn"]
        if rhs_fn == "rhs::u_x - (u) = 0":
            return "FAIL", "NRMSE mean=1 max=1", [{"traj_id": "ic0", "nrmse": 1.0}]
        return "PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic0", "nrmse": 0.0}]

    monkeypatch.setattr(rb, "validate_by_simulation", _fake_validate)

    out = rb._run_factorized_search_only_engine(
        _problem(order=1),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
        sim_validate_max_candidates=3,
    )

    assert out["engine"] == "factorized_search_only"
    assert out["status"] == "PASS"
    assert out["selected_shortlist_rank"] == 1
    assert out["internal_selected_shortlist_rank_mismatch"] is True
    assert out["canonical_equation"] == "u_x - (x) = 0"


def test_factorized_search_shortlist_includes_parent_candidate_first():
    """Regression: the rollout pool must include the parent selected candidate.

    The parent carries the promotion-time mapping refit, which can be the only
    faithful copy of the selected law (de123 pendulum benchmark failure where
    rollout reranking over corrupted shortlist rows overrode a correct
    internal selection).
    """
    parent = {
        "engine": "factorized_search",
        "order": 1,
        "expr_ast": ["var", 0],
        "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
        "canonical_equation": "u_x - (u) = 0",
        "shortlist": [{"shortlist_rank": 0, "canonical_equation": "u_x - (2*u) = 0"}],
    }
    pool = rb._factorized_search_shortlist_from_candidate(parent)
    assert len(pool) == 2
    assert pool[0]["canonical_equation"] == "u_x - (u) = 0"
    # rank -1 marks the parent and wins rollout ties against serialized rows
    assert pool[0]["shortlist_rank"] == -1
    assert "shortlist" not in pool[0]
    # the original candidate payload is not mutated
    assert parent["shortlist"]
    assert "shortlist_rank" not in parent
    # the cap never evicts the parent
    capped = rb._factorized_search_shortlist_from_candidate(parent, max_candidates=1)
    assert len(capped) == 1
    assert capped[0]["shortlist_rank"] == -1


def test_run_hybrid_engine_validates_sparse_candidates_on_probe_runs(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    def _make_run(name: str) -> rb.TrajRun:
        csv_path = data_dir / f"de999_{name}.csv"
        meta_path = data_dir / f"de999_{name}.meta.json"
        csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
        meta_path.write_text("{}", encoding="utf-8")
        return rb.TrajRun(
            traj_id=name,
            csv_path=csv_path,
            meta_path=meta_path,
            x_min=0.0,
            x_max=0.1,
            u0=1.0,
            v0=0.0,
        )

    fit_run = _make_run("ic_fit")
    probe_run = _make_run("ic_probe")

    validation_candidate = {
        "order": 1,
        "x_axis": 0,
        "coefficients": [1.0],
        "term_asts_json": [{"type": "atom", "kind": "u", "var_idxs": [], "kwargs": {}}],
    }

    def _fake_run(cmd, text=True, capture_output=False, stdout=None, stderr=None, cwd=None, env=None):
        out_path = results_dir / f"{rb._derive_run_de_base_filename([fit_run.csv_path])}_de.json"
        payload = {
            "de_discovery": {
                "selected_engine": "stlsq",
                "rescue_attempted": False,
                "rescue_triggered": False,
                "first_line": {
                    "engine": "stlsq",
                    "order": 1,
                    "canonical_equation": "u_x + u = 0",
                    "terms": ["U()"],
                    "coefficients": [1.0],
                    "validation_candidate": validation_candidate,
                },
                "selected": {
                    "engine": "stlsq",
                    "order": 1,
                    "canonical_equation": "u_x + u = 0",
                    "terms": ["U()"],
                    "coefficients": [1.0],
                    "validation_candidate": validation_candidate,
                },
                "canonical_equation": "u_x + u = 0",
            }
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    calls: list[list[str]] = []

    def _fake_validate(runs, **kwargs):
        calls.append([str(r.traj_id) for r in runs])
        return "PASS", "NRMSE mean=0 max=0", [{"traj_id": "ic_probe", "nrmse": 0.0}]

    monkeypatch.setattr(rb.subprocess, "run", _fake_run)
    monkeypatch.setattr(rb, "validate_by_simulation", _fake_validate)

    out = rb._run_hybrid_engine(
        _problem(order=1),
        [fit_run],
        probe_runs=[probe_run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert out["status"] == "PASS"
    assert out["first_line_status"] == "PASS"
    assert out["rescued_additional"] is False
    assert calls == [["ic_probe"]]


def test_compute_hybrid_summary_counts_sparse_and_rescued_cases():
    rows = [
        {
            "id": "001",
            "engines": {
                "hybrid": {
                    "status": "PASS",
                    "first_line_status": "PASS",
                    "rescued_additional": False,
                }
            },
        },
        {
            "id": "002",
            "engines": {
                "hybrid": {
                    "status": "PASS",
                    "first_line_status": "FAIL",
                    "rescued_additional": True,
                }
            },
        },
        {
            "id": "003",
            "engines": {
                "hybrid": {
                    "status": "FAIL",
                    "first_line_status": "FAIL",
                    "rescued_additional": False,
                    "failure_kind": "timeout",
                }
            },
        },
    ]

    summary = rb._compute_hybrid_summary(rows)

    assert summary["sparse_only_pass"] == 1
    assert summary["rescued_additional"] == 1
    assert summary["final_pass"] == 2
    assert summary["rescued_problem_ids"] == ["002"]
    assert summary["failure_kind_counts"] == {"timeout": 1}


def test_factorized_de_metrics_from_meta_extracts_operator_diagnostics():
    metrics = rb._factorized_de_metrics_from_meta(
        {
            "selected_engine": "factorized",
            "factorized_de_diagnostics": {
                "selected_lane": "factorized",
                "direct_residual_probe_rms": 3.0e-4,
                "direct_residual_attempted": True,
                "typed_lanes_policy": "never",
                "typed_lanes_attempted": False,
                "coefficient_dim_mode": "inferred_outer",
                "factorized_search_attempted": True,
                "factorized_probe_rms": 1.0e-4,
                "factorized_search_probe_rms": 2.0e-4,
                "whole_rhs_policy": {
                    "policy": "auto",
                    "reason": "typed_candidate_ambiguous",
                    "run": True,
                    "budget_scope": "global",
                    "max_attempts": 1,
                },
            },
            "factorized_rescue": {
                "diagnostics": {
                    "lane": "second_order_state_nonlinearity",
                    "family": "poly2",
                    "factorized_de_diagnostics": {
                        "selected_lane": "two_block_typed_assembly",
                        "selected_family": "shared_refit",
                        "typed_explorer_launches": 2,
                        "generic_explorer_launches": 0,
                        "family_gate_evaluations": 4,
                        "family_gate_passes": 3,
                        "explorer_skipped": 3,
                        "scheduler_coord_candidates_skipped": 5,
                        "two_block_typed_candidates": 1,
                        "n_candidates": 7,
                        "shortlist_size": 4,
                    },
                }
            },
            "factorized_search_rescue": {
                "diagnostics": {
                    "rescue_attempts_run": 1,
                    "rescue_attempts_available": 3,
                    "rescue_attempts_capped": True,
                }
            },
        }
    )

    assert metrics["selected_lane"] == "factorized"
    assert metrics["direct_residual_probe_rms"] == pytest.approx(3.0e-4)
    assert metrics["direct_residual_attempted"] is True
    assert metrics["typed_lanes_policy"] == "never"
    assert metrics["typed_lanes_attempted"] is False
    assert metrics["coefficient_dim_mode"] == "inferred_outer"
    assert metrics["typed_selected_lane"] == "two_block_typed_assembly"
    assert metrics["whole_rhs_attempted"] is True
    assert metrics["whole_rhs_budget_scope"] == "global"
    assert metrics["whole_rhs_max_attempts"] == 1
    assert metrics["whole_rhs_attempts_run"] == 1
    assert metrics["whole_rhs_attempts_capped"] is True
    assert metrics["typed_explorer_launches"] == 2
    assert metrics["family_gate_skips"] == 3
    assert metrics["two_block_typed_candidates"] == 1


def test_compute_factorized_de_summary_reports_runtime_counters_and_nrmse():
    rows = [
        {
            "id": "900",
            "engines": {
                "factorized_de": {
                    "status": "PASS",
                    "selected_engine": "factorized",
                    "selected_lane": "factorized",
                    "direct_residual_attempted": True,
                    "typed_lanes_policy": "never",
                    "coefficient_dim_mode": "inferred_outer",
                    "typed_selected_lane": "x_coeff_on_u",
                    "whole_rhs_attempted": False,
                    "whole_rhs_attempts_run": 0,
                    "family_gate_skips": 1,
                    "typed_explorer_launches": 0,
                    "traj_scores": [{"nrmse": 0.01}, {"nrmse": 0.02}],
                }
            },
        },
        {
            "id": "901",
            "engines": {
                "factorized_de": {
                    "status": "FAIL",
                    "selected_engine": "factorized_search",
                    "selected_lane": "factorized_search",
                    "direct_residual_attempted": True,
                    "typed_lanes_policy": "auto",
                    "coefficient_dim_mode": "inferred_outer",
                    "typed_selected_lane": "state_nonlinearity",
                    "whole_rhs_attempted": True,
                    "whole_rhs_attempts_run": 1,
                    "family_gate_skips": 0,
                    "typed_explorer_launches": 2,
                    "traj_scores": [{"nrmse": 0.5}],
                }
            },
        },
    ]

    summary = rb._compute_factorized_de_summary(rows)

    assert summary["total"] == 2
    assert summary["final_pass"] == 1
    assert summary["selected_engine_counts"] == {"factorized": 1, "factorized_search": 1}
    assert summary["selected_lane_counts"] == {"factorized": 1, "factorized_search": 1}
    assert summary["direct_residual_attempted"] == 2
    assert summary["typed_lanes_policy_counts"] == {"auto": 1, "never": 1}
    assert summary["coefficient_dim_mode_counts"] == {"inferred_outer": 2}
    assert summary["typed_selected_lane_counts"] == {"x_coeff_on_u": 1, "state_nonlinearity": 1}
    assert summary["whole_rhs_attempted"] == 1
    assert summary["whole_rhs_skipped"] == 1
    assert summary["whole_rhs_attempts_run"] == 1
    assert summary["family_gate_skips"] == 1
    assert summary["typed_explorer_launches"] == 2
    assert summary["median_rollout_nrmse"] == pytest.approx(0.02)
    assert summary["worst_rollout_nrmse"] == pytest.approx(0.5)


def test_classify_failure_kind_distinguishes_timeout_nonfinite_and_high_nrmse():
    assert rb.classify_failure_kind(
        "FAIL",
        "Integration failed on ic1: timeout (budget 20s)",
        [{"traj_id": "ic1", "error": "timeout (budget 20s)"}],
    ) == "timeout"
    assert rb.classify_failure_kind(
        "FAIL",
        "Integration failed on ic1: Non-finite factorized symbolic search candidate evaluation",
        [{"traj_id": "ic1", "error": "Non-finite factorized symbolic search candidate evaluation"}],
    ) == "nonfinite_candidate"
    assert rb.classify_failure_kind(
        "FAIL",
        "NRMSE mean=0.225 max=0.225",
        [{"traj_id": "ic1", "nrmse": 0.225}],
    ) == "high_nrmse"


def test_compute_failure_kind_counts_aggregates_result_rows():
    rows = [
        {"status": "FAIL", "message": "Integration failed on ic1: timeout (budget 20s)", "traj_scores": []},
        {"status": "FAIL", "message": "Integration failed on ic1: Non-finite factorized symbolic search candidate evaluation", "traj_scores": []},
        {"status": "FAIL", "message": "NRMSE mean=1 max=1", "traj_scores": [{"traj_id": "ic1", "nrmse": 1.0}]},
        {"status": "PASS", "message": "NRMSE mean=0 max=0", "traj_scores": [{"traj_id": "ic1", "nrmse": 0.0}]},
    ]

    counts = rb.compute_failure_kind_counts(rows)

    assert counts == {
        "timeout": 1,
        "nonfinite_candidate": 1,
        "high_nrmse": 1,
    }


def test_clean_fallback_hybrid_keeps_partial_stlsq_when_fallback_fails(
    monkeypatch,
    tmp_path: Path,
):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    sparse_json = results_dir / "de999_ic0_de.json"
    sparse_json.write_text(json.dumps({"engine": "stlsq"}), encoding="utf-8")
    fallback_json = results_dir / "de999_ic0_de.factorized.json"

    def _fake_sparse(problem, fit_runs, **kwargs):
        assert problem.id == "999"
        assert [r.traj_id for r in fit_runs] == ["ic0"]
        return {
            "engine": "sparse",
            "selected_engine": "stlsq",
            "internal_selected_engine": "stlsq",
            "status": "PARTIAL",
            "message": "NRMSE mean=0.0139 max=0.0203",
            "json_path": str(sparse_json),
            "canonical_equation": "stlsq partial",
            "traj_scores": [{"traj_id": "ic0", "nrmse": 0.0203}],
            "coeff_map": {"u": -1.0},
        }

    def _fake_hybrid(problem, fit_runs, **kwargs):
        assert kwargs["factorized_de"] is True
        assert kwargs["probe_runs"] == [run]
        return {
            "engine": "factorized_de",
            "selected_engine": "factorized",
            "internal_selected_engine": "factorized",
            "status": "FAIL",
            "message": "NRMSE mean=2.29 max=2.86",
            "json_path": str(fallback_json),
            "canonical_equation": "fallback fail",
            "traj_scores": [{"traj_id": "ic0", "nrmse": 2.86}],
            "command_resource_report": {
                "monitor": "test",
                "peak_tree_rss_mb": 2048.0,
                "last_tree_rss_mb": 2000.0,
                "killed_by_signal": True,
                "signal": "SIGKILL",
            },
            "resource_failure_suspected": True,
            "command_killed_by_signal": True,
            "command_signal": "SIGKILL",
        }

    monkeypatch.setattr(rb, "_run_stlsq_engine", _fake_sparse)
    monkeypatch.setattr(rb, "_run_hybrid_engine", _fake_hybrid)

    out = rb._run_clean_fallback_hybrid_engine(
        _problem(order=2),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert out["engine"] == "hybrid"
    assert out["selected_engine"] == "stlsq"
    assert out["status"] == "PARTIAL"
    assert out["message"] == "NRMSE mean=0.0139 max=0.0203"
    assert out["canonical_equation"] == "stlsq partial"
    assert out["rescued_additional"] is False
    assert out["first_line_status"] == "PARTIAL"
    assert out["first_line_message"] == "NRMSE mean=0.0139 max=0.0203"
    assert out["clean_fallback_attempted"] is True
    assert out["clean_fallback_status"] == "FAIL"
    assert out["clean_fallback_message"] == "NRMSE mean=2.29 max=2.86"
    assert out["clean_fallback_resource_failure_suspected"] is True
    assert out["clean_fallback_killed_by_signal"] is True
    assert out["clean_fallback_signal"] == "SIGKILL"
    assert out["clean_fallback_peak_tree_rss_mb"] == pytest.approx(2048.0)
    assert out["hybrid_rollout_choice"] == "stlsq"
    assert out["clean_fallback_beats_first_line"] is False
    assert out["sparse_first_line_result"]["status"] == "PARTIAL"
    assert out["sparse_first_line_result"]["json_path"].endswith(".sparse_first_line.json")


def test_clean_fallback_hybrid_replaces_partial_stlsq_when_fallback_passes(
    monkeypatch,
    tmp_path: Path,
):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de999_ic0.csv"
    meta_path = data_dir / "de999_ic0.meta.json"
    csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
    meta_path.write_text("{}", encoding="utf-8")
    run = rb.TrajRun(
        traj_id="ic0",
        csv_path=csv_path,
        meta_path=meta_path,
        x_min=0.0,
        x_max=0.1,
        u0=1.0,
        v0=0.0,
    )

    sparse_json = results_dir / "de999_ic0_de.json"
    sparse_json.write_text(json.dumps({"engine": "stlsq"}), encoding="utf-8")
    fallback_json = results_dir / "de999_ic0_de.factorized.json"

    def _fake_sparse(problem, fit_runs, **kwargs):
        return {
            "engine": "sparse",
            "selected_engine": "stlsq",
            "internal_selected_engine": "stlsq",
            "status": "PARTIAL",
            "message": "NRMSE mean=0.0139 max=0.0203",
            "json_path": str(sparse_json),
            "canonical_equation": "stlsq partial",
            "traj_scores": [{"traj_id": "ic0", "nrmse": 0.0203}],
            "coeff_map": {},
        }

    def _fake_hybrid(problem, fit_runs, **kwargs):
        assert kwargs["factorized_de"] is True
        return {
            "engine": "factorized_de",
            "selected_engine": "factorized",
            "internal_selected_engine": "factorized",
            "status": "PASS",
            "message": "NRMSE mean=1e-06 max=1e-06",
            "json_path": str(fallback_json),
            "canonical_equation": "fallback pass",
            "traj_scores": [{"traj_id": "ic0", "nrmse": 1.0e-6}],
        }

    monkeypatch.setattr(rb, "_run_stlsq_engine", _fake_sparse)
    monkeypatch.setattr(rb, "_run_hybrid_engine", _fake_hybrid)

    out = rb._run_clean_fallback_hybrid_engine(
        _problem(order=2),
        [run],
        probe_runs=[run],
        results_dir=results_dir,
        fast=True,
        verbose=False,
        no_sim_validate=False,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert out["selected_engine"] == "factorized"
    assert out["status"] == "PASS"
    assert out["message"] == "NRMSE mean=1e-06 max=1e-06"
    assert out["canonical_equation"] == "fallback pass"
    assert out["hybrid_rollout_choice"] == "factorized"
    assert out["clean_fallback_beats_first_line"] is True
    assert out["rescued_additional"] is True


def test_run_problem_splits_fit_and_probe_trajectories(monkeypatch, tmp_path: Path):
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for name in ("ic2", "ic0", "ic1"):
        csv_path = data_dir / f"de999_{name}.csv"
        meta_path = data_dir / f"de999_{name}.meta.json"
        csv_path.write_text("y,x0\n1.0,0.0\n0.9,0.1\n", encoding="utf-8")
        meta_path.write_text("{}", encoding="utf-8")
        runs.append(
            rb.TrajRun(
                traj_id=name,
                csv_path=csv_path,
                meta_path=meta_path,
                x_min=0.0,
                x_max=0.1,
                u0=1.0,
                v0=0.0,
            )
        )

    captured: dict[str, Any] = {}

    monkeypatch.setattr(rb, "load_existing_runs", lambda *args, **kwargs: (runs, "existing_csv"))

    def _fake_sparse(problem, fit_runs, **kwargs):
        split = (
            [str(r.traj_id) for r in fit_runs],
            [str(r.traj_id) for r in kwargs["probe_runs"]],
        )
        captured["sparse"] = split
        captured.setdefault("sparse_calls", []).append(split)
        return {"engine": "sparse", "status": "PASS", "message": "ok"}

    def _fake_hybrid(problem, fit_runs, **kwargs):
        key = "factorized_de" if bool(kwargs.get("factorized_de", False)) else "hybrid"
        captured[key] = (
            [str(r.traj_id) for r in fit_runs],
            [str(r.traj_id) for r in kwargs["probe_runs"]],
        )
        return {
            "engine": key,
            "status": "PASS",
            "message": "ok",
            "first_line_status": "PASS",
            "rescued_additional": False,
        }

    def _fake_oracle(problem, runs_all, **kwargs):
        captured["factorized_search_oracle"] = (
            [str(r.traj_id) for r in runs_all],
            [str(r.traj_id) for r in kwargs["probe_runs"]],
        )
        return {"engine": "factorized_search_oracle", "status": "PASS", "message": "ok"}

    monkeypatch.setattr(rb, "_run_stlsq_engine", _fake_sparse)
    monkeypatch.setattr(rb, "_run_hybrid_engine", _fake_hybrid)
    monkeypatch.setattr(rb, "_run_factorized_search_engine", _fake_oracle)

    result = rb.run_problem(
        _problem(order=1),
        data_dir=data_dir,
        results_dir=results_dir,
        fast=True,
        skip_generate=True,
        verbose=False,
        engine="compare",
        n_traj=3,
        n_points=2,
        seed=0,
        split_mode="traj_holdout",
        holdout_last_k=1,
        traj_metric="max",
        no_sim_validate=True,
        pass_nrmse=1.0e-2,
        partial_nrmse=5.0e-2,
    )

    assert captured["sparse_calls"] == [
        (["ic0", "ic1"], ["ic2"]),
        (["ic0", "ic1"], ["ic2"]),
    ]
    assert "hybrid" not in captured
    assert captured["factorized_de"] == (["ic0", "ic1"], ["ic2"])
    assert captured["factorized_search_oracle"] == (["ic2", "ic0", "ic1"], ["ic2"])
    assert result["engines"]["hybrid"]["first_line_status"] == "PASS"
    assert result["engines"]["hybrid"]["clean_fallback_attempted"] is False
    assert "factorized_de" in result["engines"]
    assert "factorized_de=PASS" in result["message"]
    assert result["n_fit_traj"] == 2
    assert result["n_probe_traj"] == 1


def test_generate_data_multi_retries_ic_and_solver(monkeypatch, tmp_path: Path):
    calls = {"n": 0}

    def _rhs(x, s, params):
        return [-float(s[0])]

    def _fake_solve_ivp(fun, span, y0, t_eval, method, rtol, atol):
        calls["n"] += 1
        # First horizon attempt: fail all solver methods.
        if calls["n"] <= 3:
            return SimpleNamespace(
                status=-1,
                message=f"{method} failed",
                t=np.asarray(t_eval),
                y=np.asarray([np.asarray(t_eval) * np.nan]),
            )
        # Shortened horizon: only BDF succeeds.
        if str(method) != "BDF":
            return SimpleNamespace(
                status=-1,
                message=f"{method} failed",
                t=np.asarray(t_eval),
                y=np.asarray([np.asarray(t_eval)]),
            )
        t = np.asarray(t_eval, dtype=np.float64)
        u = float(y0[0]) * np.exp(-(t - float(t[0])))
        return SimpleNamespace(status=0, message="ok", t=t, y=np.asarray([u], dtype=np.float64))

    monkeypatch.setattr(rb, "resolve_rhs", lambda problem, prefer_manual=True: (_rhs, "unit_test_rhs"))
    monkeypatch.setattr(rb, "default_t_max", lambda problem, params: 1.0)
    monkeypatch.setattr(rb, "solve_ivp", _fake_solve_ivp)
    monkeypatch.setattr(
        rb,
        "_select_supported_x_min",
        lambda problem, rhs_fn, param_values, *, nominal_x_max, seed, n_traj: (
            0.0,
            {"policy": "unit_test_fixed"},
        ),
    )

    runs, rhs_source = rb.generate_data_multi(
        _problem(order=1),
        {},
        tmp_path / "data",
        n_traj=1,
        n_points=32,
        seed=0,
    )

    assert rhs_source == "unit_test_rhs"
    assert len(runs) == 1
    meta = json.loads(runs[0].meta_path.read_text(encoding="utf-8"))
    assert meta["solver"]["method"] == "BDF"
    assert int(meta["solver"]["ic_retry"]) == 0
    assert int(meta["solver"]["support_backoffs"]) == 1
    assert meta["horizon"]["selected_x_max"] < meta["horizon"]["nominal_x_max"]
    assert calls["n"] >= 6


def test_generate_data_multi_selects_x_min_from_black_box_pilot(monkeypatch, tmp_path: Path):
    def _rhs(x, s, params):
        return [-float(s[0])]

    def _fake_solve_ivp(fun, span, y0, t_eval, method, rtol, atol):
        x0 = float(span[0])
        t = np.asarray(t_eval, dtype=np.float64)
        if x0 < 0.5:
            return SimpleNamespace(
                status=-1,
                message="unsupported start",
                t=t,
                y=np.asarray([np.full_like(t, np.nan)], dtype=np.float64),
            )
        u = float(y0[0]) * np.exp(-(t - float(t[0])))
        return SimpleNamespace(status=0, message="ok", t=t, y=np.asarray([u], dtype=np.float64))

    opaque_problem = ProblemDef(
        id="999",
        order=1,
        indep_var="x",
        dep_var="u",
        equation="du/dx=opaque_rhs",
        description="opaque unit-test",
        feynman_ref="-",
        params=[],
        param_ranges=[],
        ic_type="value",
    )
    monkeypatch.setattr(rb, "resolve_rhs", lambda problem, prefer_manual=True: (_rhs, "unit_test_rhs"))
    monkeypatch.setattr(rb, "default_t_max", lambda problem, params: 2.0)
    monkeypatch.setattr(rb, "solve_ivp", _fake_solve_ivp)

    runs, rhs_source = rb.generate_data_multi(
        opaque_problem,
        {},
        tmp_path / "data",
        n_traj=1,
        n_points=32,
        seed=0,
    )

    assert rhs_source == "unit_test_rhs"
    meta = json.loads(runs[0].meta_path.read_text(encoding="utf-8"))
    assert float(meta["x_min"]) == pytest.approx(0.5)
    assert meta["x_start"]["policy"] == "black_box_support_pilot"
    assert meta["x_start"]["reason"] == "first_good_candidate"
    assert float(meta["x_start"]["selected_x_min"]) == pytest.approx(0.5)


def test_generate_data_multi_backs_off_explosive_growth(monkeypatch, tmp_path: Path):
    def _rhs(x, s, params):
        return [0.875 * float(s[0])]

    def _fake_solve_ivp(fun, span, y0, t_eval, method, rtol, atol):
        t = np.asarray(t_eval, dtype=np.float64)
        u = float(y0[0]) * np.exp(0.875 * (t - float(t[0])))
        return SimpleNamespace(status=0, message="ok", t=t, y=np.asarray([u], dtype=np.float64))

    monkeypatch.setattr(rb, "resolve_rhs", lambda problem, prefer_manual=True: (_rhs, "unit_test_rhs"))
    monkeypatch.setattr(rb, "default_t_max", lambda problem, params: 20.0)
    monkeypatch.setattr(rb, "solve_ivp", _fake_solve_ivp)

    runs, rhs_source = rb.generate_data_multi(
        _problem(order=1),
        {},
        tmp_path / "data",
        n_traj=1,
        n_points=64,
        seed=0,
    )

    assert rhs_source == "unit_test_rhs"
    meta = json.loads(runs[0].meta_path.read_text(encoding="utf-8"))
    assert int(meta["solver"]["support_backoffs"]) >= 1
    assert float(meta["nominal_x_max"]) == 20.0
    assert float(meta["x_max"]) <= 10.1
    assert meta["horizon"]["reason"] == "ok"
    assert float(meta["horizon"]["state_growth"]) <= rb.GEN_SUPPORT_MAX_STATE_GROWTH
