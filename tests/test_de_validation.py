# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nestynet_sr.sr_de.de_validation import (
    evaluate_compile_domain_witness,
    evaluate_library_candidate_rollout,
    library_candidate_to_rhs_callable,
    run_rollout_witnesses,
    validate_by_simulation,
    witness_specs_from_runs,
)
from nestynet_sr.sr_search.coe_witness import CoEWitnessExecutor


def _u_atom() -> dict:
    return {"type": "atom", "kind": "u", "kwargs": {}}


def _const_node(value: float) -> dict:
    return {"type": "const", "value": float(value)}


def _pow_node(base: dict, exponent: float) -> dict:
    return {"type": "pow", "base": base, "exponent": float(exponent)}


def _log_node(arg: dict) -> dict:
    return {"type": "log", "arg": arg}


def _write_exp_decay_run(tmp_path, *, traj_id: str = "ic0"):
    x = np.linspace(0.0, 1.0, 96, dtype=np.float64)
    u = np.exp(-x)
    path = tmp_path / f"{traj_id}.csv"
    np.savetxt(path, np.column_stack([u, x]), delimiter=",", header="u,x", comments="")
    return SimpleNamespace(csv_path=path, traj_id=traj_id, u0=float(u[0]), v0=0.0)


def test_library_compile_domain_witness_passes_finite_candidate():
    candidate = {
        "engine": "stlsq",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "u_x + u = 0",
        "validation_candidate": {
            "order": 1,
            "x_axis": 0,
            "coefficients": [1.0],
            "term_asts_json": [_u_atom()],
        },
    }

    result = evaluate_compile_domain_witness(
        candidate,
        engine="stlsq",
        domain_samples=[{"x": 0.0, "u": 1.0, "du": 0.0}],
    )

    assert result.status == "PASS"
    assert result.failure_kind is None
    assert result.metrics["order"] == 1


def test_library_compile_domain_witness_flags_nonfinite_candidate():
    candidate = {
        "engine": "stlsq",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "u_x - log(u) = 0",
        "validation_candidate": {
            "order": 1,
            "x_axis": 0,
            "coefficients": [-1.0],
            "term_asts_json": [_log_node(_u_atom())],
        },
    }

    result = evaluate_compile_domain_witness(
        candidate,
        engine="stlsq",
        domain_samples=[{"x": 0.0, "u": -1.0, "du": 0.0}],
    )

    assert result.status == "FAIL"
    assert result.failure_kind == "nonfinite_candidate"


def test_exact_first_order_rollout_passes(tmp_path):
    run = _write_exp_decay_run(tmp_path)
    candidate = {
        "engine": "stlsq",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "u_x + u = 0",
        "validation_candidate": {
            "order": 1,
            "x_axis": 0,
            "coefficients": [1.0],
            "term_asts_json": [_u_atom()],
        },
    }

    out = evaluate_library_candidate_rollout(
        candidate,
        probe_runs=[run],
        pass_nrmse=1.0e-4,
        partial_nrmse=1.0e-2,
        sim_validate_traj_time_budget_s=5.0,
        sim_validate_blowup_factor=100.0,
        sim_validate_blowup_abs=1.0e6,
    )

    assert out["status"] == "PASS"
    assert out["discovered_order"] == 1
    assert out["traj_scores"][0]["traj_id"] == "ic0"
    assert float(out["traj_scores"][0]["nrmse"]) < 1.0e-4


def test_rollout_witness_executor_runs_same_rhs(tmp_path):
    run = _write_exp_decay_run(tmp_path, traj_id="probe")
    candidate = {
        "engine": "stlsq",
        "order": 1,
        "x_axis": 0,
        "validation_candidate": {
            "order": 1,
            "x_axis": 0,
            "coefficients": [1.0],
            "term_asts_json": [_u_atom()],
        },
    }
    order, rhs_fn = library_candidate_to_rhs_callable(candidate)
    specs = witness_specs_from_runs([run], tier="short_rollout")

    results = run_rollout_witnesses(
        specs,
        rhs_fn=rhs_fn,
        order=order,
        proposal_id="p0",
        executor=CoEWitnessExecutor(parallelism=1),
    )

    assert len(results) == 1
    assert results[0].proposal_id == "p0"
    assert results[0].tier == "short_rollout"
    assert results[0].status == "PASS"
    assert float(results[0].rollout_nrmse) < 1.0e-4


def test_blowup_rollout_returns_failure_without_crashing(tmp_path):
    x = np.linspace(0.0, 1.0, 64, dtype=np.float64)
    u = np.full_like(x, 2.0)
    path = tmp_path / "blowup.csv"
    np.savetxt(path, np.column_stack([u, x]), delimiter=",", header="u,x", comments="")
    run = SimpleNamespace(csv_path=path, traj_id="blowup", u0=2.0, v0=0.0)
    candidate = {
        "engine": "stlsq",
        "order": 1,
        "x_axis": 0,
        "canonical_equation": "u_x - u^3 = 0",
        "validation_candidate": {
            "order": 1,
            "x_axis": 0,
            "coefficients": [-1.0],
            "term_asts_json": [_pow_node(_u_atom(), 3.0)],
        },
    }
    order, rhs_fn = library_candidate_to_rhs_callable(candidate)

    status, message, traj_scores = validate_by_simulation(
        [run],
        rhs_fn=rhs_fn,
        order=order,
        pass_nrmse=1.0e-4,
        partial_nrmse=1.0e-2,
        traj_time_budget_s=5.0,
        blowup_factor=2.0,
        blowup_abs=4.0,
    )

    assert status == "FAIL"
    assert "Integration failed on blowup" in message
    assert traj_scores
    assert traj_scores[0]["nrmse"] == float("inf")


def test_compile_domain_witness_reports_compile_errors():
    candidate = {
        "engine": "stlsq",
        "order": 1,
        "x_axis": 0,
        "validation_candidate": {
            "order": 1,
            "x_axis": 0,
            "coefficients": [1.0],
            "term_asts_json": [_const_node(1.0), _u_atom()],
        },
    }

    result = evaluate_compile_domain_witness(candidate, engine="stlsq")

    assert result.status == "ERROR"
    assert result.failure_kind == "compile_error"
