# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from nestynet_sr.run_de import write_de_json_report
from nestynet_sr.sr_de.de_committee import (
    run_de_committee_audit,
    selected_engine_from_decision,
    selected_summary_from_decision,
    tied_candidate_summaries_from_decision,
)
from nestynet_sr.sr_de.de_search import DESearchResult


def _proposal(
    proposal_id: str,
    *,
    engine: str = "stlsq",
    equation: str = "u_x + u = 0",
    complexity: float = 1.0,
    probe_mse: float = 1.0e-6,
    typed_confidence: str | None = None,
) -> dict:
    payload = {
        "proposal_id": proposal_id,
        "engine": engine,
        "role_signature": "library" if engine == "stlsq" else "whole_rhs_fss:poly",
        "canonical_key": "residual:du+u",
        "order": 1,
        "x_axis": 0,
        "rhs_payload": {
            "engine": engine,
            "order": 1,
            "x_axis": 0,
            "canonical_equation": equation,
            "validation_candidate": {
                "order": 1,
                "x_axis": 0,
                "coefficients": [1.0],
                "term_asts_json": [{"type": "atom", "kind": "u", "kwargs": {}}],
            },
        },
        "residual_payload": {"canonical_equation": equation},
        "canonical_equation": equation,
        "complexity": float(complexity),
        "pointwise_metrics": {"probe_mse": float(probe_mse), "probe_rms": float(probe_mse) ** 0.5},
        "diagnostics": {},
        "support": {"support_count": 1, "sources": [proposal_id], "engines": [engine]},
        "provenance": {"source_id": proposal_id},
        "cost": {},
    }
    if typed_confidence is not None:
        payload["rhs_payload"]["typed_metadata"] = {
            "collapse_confidence": str(typed_confidence),
            "collapse_reason": "ok",
        }
    return payload


def test_committee_prefers_rollout_stable_candidate_over_lower_residual():
    proposals = [
        _proposal("unstable", engine="stlsq", probe_mse=1.0e-12),
        _proposal("stable", engine="factorized_search", probe_mse=1.0e-4),
    ]
    decision = run_de_committee_audit(
        proposals,
        rollout_candidates=[
            {
                "proposal_id": "unstable",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.5}],
                "canonical_equation": "u_x + u = 0",
                "discovered_order": 1,
            },
            {
                "proposal_id": "stable",
                "engine": "factorized_search",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 1.0e-4}],
                "canonical_equation": "u_x + u = 0",
                "discovered_order": 1,
            },
        ],
        selected_engine="stlsq",
        run_compile_domain=False,
    )

    payload = decision.to_dict()
    assert payload["selected_id"] == "stable"
    assert selected_engine_from_decision(payload) == "factorized_search"
    assert selected_summary_from_decision(payload)["proposal_id"] == "stable"
    assert payload["selection_basis"] == "rollout_worst_nrmse"


def test_committee_uses_complexity_inside_noise_tie_bucket():
    proposals = [
        _proposal("complex", complexity=10.0, probe_mse=1.0e-6),
        _proposal("simple", complexity=2.0, probe_mse=1.0e-6),
    ]
    decision = run_de_committee_audit(
        proposals,
        rollout_candidates=[
            {
                "proposal_id": "complex",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.0104}],
                "discovered_order": 1,
            },
            {
                "proposal_id": "simple",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.0108}],
                "discovered_order": 1,
            },
        ],
        config={"tolerance_nrmse": 1.0e-3},
        run_compile_domain=False,
    )

    assert decision.selected_id == "simple"


def test_committee_support_is_only_tie_breaker_after_rollout_status_and_loss():
    weak = _proposal("weak_supported", engine="factorized", probe_mse=1.0e-6)
    weak["support"] = {
        "support_count": 5,
        "sources": ["a", "b", "c", "d", "e"],
        "engines": ["stlsq", "factorized"],
    }
    strong = _proposal("strong_rollout", engine="stlsq", probe_mse=1.0e-6)

    decision = run_de_committee_audit(
        [weak, strong],
        rollout_candidates=[
            {
                "proposal_id": "weak_supported",
                "engine": "factorized",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.2}],
                "discovered_order": 1,
            },
            {
                "proposal_id": "strong_rollout",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.01}],
                "discovered_order": 1,
            },
        ],
        config={"tolerance_nrmse": 1.0e-3},
        run_compile_domain=False,
    )

    assert decision.selected_id == "strong_rollout"


def test_committee_support_breaks_noise_tie_after_residual():
    low_support = _proposal("low_support", complexity=1.0, probe_mse=1.0e-6)
    high_support = _proposal("high_support", complexity=1.0, probe_mse=1.0e-6)
    high_support["support"] = {
        "support_count": 3,
        "sources": ["first_line", "scout0:first_line", "scout1:first_line"],
        "engines": ["stlsq"],
    }

    decision = run_de_committee_audit(
        [low_support, high_support],
        rollout_candidates=[
            {
                "proposal_id": "low_support",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.0102}],
                "discovered_order": 1,
            },
            {
                "proposal_id": "high_support",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.0103}],
                "discovered_order": 1,
            },
        ],
        config={"tolerance_nrmse": 1.0e-3},
        run_compile_domain=False,
    )

    assert decision.selected_id == "high_support"


def test_committee_typed_confidence_breaks_only_late_ties():
    typed = _proposal(
        "typed",
        engine="factorized",
        complexity=2.0,
        probe_mse=1.0e-6,
        typed_confidence="high",
    )
    untyped = _proposal("untyped", engine="stlsq", complexity=1.0, probe_mse=1.0e-6)

    tied = run_de_committee_audit(
        [untyped, typed],
        rollout_candidates=[
            {
                "proposal_id": "untyped",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.0102}],
                "discovered_order": 1,
            },
            {
                "proposal_id": "typed",
                "engine": "factorized",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.0103}],
                "discovered_order": 1,
            },
        ],
        config={"tolerance_nrmse": 1.0e-3},
        run_compile_domain=False,
    )

    assert tied.selected_id == "typed"
    selected = selected_summary_from_decision(tied)
    assert selected["typed_confidence"] == "high"

    clearly_better_rollout = run_de_committee_audit(
        [untyped, typed],
        rollout_candidates=[
            {
                "proposal_id": "untyped",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.0102}],
                "discovered_order": 1,
            },
            {
                "proposal_id": "typed",
                "engine": "factorized",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.20}],
                "discovered_order": 1,
            },
        ],
        config={"tolerance_nrmse": 1.0e-3},
        run_compile_domain=False,
    )

    assert clearly_better_rollout.selected_id == "untyped"


def test_committee_typed_confidence_uses_top_level_collapse_fallback():
    typed = _proposal("typed_fallback", engine="factorized", complexity=2.0, probe_mse=1.0e-6)
    typed["rhs_payload"]["collapse_confidence"] = "high"
    typed["rhs_payload"]["collapse_reason"] = "ok"
    untyped = _proposal("untyped", engine="stlsq", complexity=1.0, probe_mse=1.0e-6)

    decision = run_de_committee_audit(
        [untyped, typed],
        rollout_candidates=[
            {
                "proposal_id": "untyped",
                "engine": "stlsq",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.0102}],
                "discovered_order": 1,
            },
            {
                "proposal_id": "typed_fallback",
                "engine": "factorized",
                "status": "PASS",
                "traj_scores": [{"traj_id": "a", "nrmse": 0.0103}],
                "discovered_order": 1,
            },
        ],
        config={"tolerance_nrmse": 1.0e-3},
        run_compile_domain=False,
    )

    assert decision.selected_id == "typed_fallback"
    assert selected_summary_from_decision(decision)["typed_confidence"] == "high"


def test_tied_candidate_summaries_use_rollout_tolerance_and_role_cap():
    decision = {
        "selected_id": "a",
        "candidate_summary": [
            {
                "proposal_id": "a",
                "engine": "stlsq",
                "role_signature": "library",
                "status": "PASS",
                "fatal_failures": 0,
                "worst_rollout_nrmse": 0.0100,
                "median_rollout_nrmse": 0.0090,
            },
            {
                "proposal_id": "b",
                "engine": "factorized_search",
                "role_signature": "whole_rhs_fss",
                "status": "PASS",
                "fatal_failures": 0,
                "worst_rollout_nrmse": 0.0105,
                "median_rollout_nrmse": 0.0095,
            },
            {
                "proposal_id": "c",
                "engine": "factorized_search",
                "role_signature": "whole_rhs_fss",
                "status": "PASS",
                "fatal_failures": 0,
                "worst_rollout_nrmse": 0.0110,
                "median_rollout_nrmse": 0.0100,
            },
            {
                "proposal_id": "d",
                "engine": "factorized",
                "role_signature": "typed",
                "status": "FAIL",
                "fatal_failures": 0,
                "worst_rollout_nrmse": 0.0,
                "median_rollout_nrmse": 0.0,
            },
        ],
    }

    tied = tied_candidate_summaries_from_decision(
        decision,
        tolerance_nrmse=1.0e-3,
        max_candidates=4,
        max_per_role=1,
    )

    assert [row["proposal_id"] for row in tied] == ["a", "b"]


def test_committee_handles_all_candidates_failing_compile_domain():
    bad = _proposal("bad")
    bad["rhs_payload"]["validation_candidate"]["coefficients"] = [1.0, 2.0]

    decision = run_de_committee_audit([bad], run_compile_domain=True)

    assert decision.status == "no_valid_candidates"
    assert decision.selected_id is None
    assert decision.candidate_summary[0]["compile_status"] == "ERROR"


def test_run_de_report_audit_mode_does_not_change_selected_engine(tmp_path: Path):
    result = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[None],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-3,
        rms_val=2.0e-3,
        condition_number=1.0,
    )
    report_path = tmp_path / "report.json"
    args = SimpleNamespace(
        device=None,
        num_segments=8,
        epochs=100,
        loss_target=1.0e-8,
        order_candidates="1",
        max_x_power=1,
        max_u_power=1,
        max_xu_total_degree=0,
        include_xdu=False,
        include_inv_xdu=False,
        include_inv_xu=False,
        include_inv_x2u=False,
        include_du=False,
        include_d2u=False,
        include_udu=False,
        stlsq_lambda=1.0e-3,
        sparsity_penalty=1.0e-3,
        enforce_units=False,
        units_policy=None,
        nn_units_semantics=None,
        factorized_rescue="never",
        factorized_two_block_shared_coord="never",
        factorized_search_rescue="never",
        factorized_search_preset="default",
        factorized_search_trigger_val_rms=1.0e-3,
        factorized_search_trigger_cond=1.0e8,
        factorized_search_replace_rel_factor=0.98,
        factorized_search_n_iter=None,
        factorized_search_max_depth=None,
        factorized_search_n_fit=None,
        factorized_search_n_probe=None,
        factorized_search_return_topk=None,
        stageb_refine_residual=False,
        stageb_epochs=0,
        de_coe_mode="audit",
    )

    write_de_json_report(
        ["dummy.csv"],
        str(report_path),
        [1.0e-4],
        result,
        args,
        walltime=0.0,
    )

    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    de_payload = payload["de_discovery"]
    assert de_payload["selected_engine"] == "stlsq"
    assert de_payload["selected"]["engine"] == "stlsq"
    assert de_payload["committee_decision"]["status"] == "selected"
    assert payload["config"]["de_coe_mode"] == "audit"
