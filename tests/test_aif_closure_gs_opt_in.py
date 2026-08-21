# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.

from __future__ import annotations

from typing import Any

from nestynet_sr.sr_search.factorized_search import oracle_lab
from nestynet_sr.sr_search.factorized_search.aif_closure_benchmark import (
    run_benchmark,
)
from scripts.run_table5_fss_gs import (
    _arm_command,
    _arm_environment,
    _build_parser,
    _dense_audit,
)


def _spec() -> dict[str, Any]:
    return {
        "id": "feynman_000",
        "basis": ["L", "T"],
        "variables": [
            {"name": "x0", "bounds": [1.0, 3.0], "dim": [0.0, 0.0]},
        ],
        "constants": [],
        "target": {"expr": "x0", "dim": [0.0, 0.0]},
    }


def _install_fake_oracle(monkeypatch, calls: list[dict[str, Any]]) -> None:
    def fake_run_oracle_equation(*_args, **kwargs):
        calls.append(dict(kwargs))
        diagnostics = (
            [{"z_human": "x0", "gs_source_family": "test"}]
            if bool(kwargs.get("gs_carrier_seed", False))
            else []
        )
        return {
            "best": {
                "mse": 0.0,
                "raw_mse": 0.0,
                "expr": "x0",
                "expr_ast": ["var", 0],
                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                "mapping_kind": "poly",
            },
            "gs_carrier_seed_diagnostics": diagnostics,
        }

    monkeypatch.setattr(oracle_lab, "run_oracle_equation", fake_run_oracle_equation)


def test_aif_benchmark_default_does_not_forward_or_report_gs(monkeypatch):
    calls: list[dict[str, Any]] = []
    _install_fake_oracle(monkeypatch, calls)

    rows = run_benchmark([_spec()], n_iter=1)

    assert len(calls) == 1
    assert "gs_carrier_seed" not in calls[0]
    assert rows[0]["status"] == "solved"
    assert "gs_carrier_seed" not in rows[0]
    assert "gs_carrier_seed_diagnostics" not in rows[0]
    assert "candidate_payload" not in rows[0]


def test_aif_benchmark_forwards_and_reports_opt_in_gs(monkeypatch):
    calls: list[dict[str, Any]] = []
    _install_fake_oracle(monkeypatch, calls)

    rows = run_benchmark([_spec()], n_iter=1, gs_carrier_seed=True)

    assert len(calls) == 1
    assert calls[0]["gs_carrier_seed"] is True
    assert rows[0]["status"] == "solved"
    assert rows[0]["gs_carrier_seed"] is True
    assert rows[0]["gs_carrier_seed_count"] == 1
    assert rows[0]["gs_carrier_seed_diagnostics"][0]["z_human"] == "x0"


def test_aif_benchmark_retains_candidate_payload_only_when_requested(monkeypatch):
    calls: list[dict[str, Any]] = []
    _install_fake_oracle(monkeypatch, calls)

    rows = run_benchmark([_spec()], n_iter=1, retain_candidate_payload=True)

    assert len(calls) == 1
    assert "retain_candidate_payload" not in calls[0]
    assert "gs_carrier_seed" not in calls[0]
    assert rows[0]["candidate_payload"] == {
        "expr_ast": ["var", 0],
        "mapping": {
            "kind": "poly",
            "coeffs": [0.0, 1.0],
            "mu": 0.0,
            "std": 1.0,
        },
        "mapping_kind": "poly",
        "raw_mse": 0.0,
    }


def test_table5_dense_audit_evaluates_outer_mapping_and_linear_head():
    spec = {
        "id": "audit_mapping",
        "basis": ["L"],
        "variables": [
            {"name": "x0", "bounds": [-0.5, 0.5], "dim": [0.0]},
        ],
        "constants": [],
        "target": {"expr": "exp(x0) + 0.25*x0", "dim": [0.0]},
    }
    row = {
        "expr": "x0",
        "candidate_payload": {
            "expr_ast": ["var", 0],
            "mapping": {
                "kind": "exp",
                "a": 1.0,
                "b": 1.0,
                "c": 0.0,
                "mu": 0.0,
                "std": 1.0,
                "_lin_head": {
                    "terms": [["var", 0]],
                    "coeffs": [0.0, 0.25],
                },
            },
        },
    }

    audit = _dense_audit(row, spec, n_probe=256, seed=123)

    assert audit["status"] == "ok"
    assert audit["finite_fraction"] == 1.0
    assert audit["mse"] < 1.0e-28


def test_table5_arms_differ_only_by_gs_opt_ins():
    baseline = _arm_command("fss_only")
    with_gs = _arm_command("fss_gs")

    assert baseline == with_gs[:-1]
    assert with_gs[-1:] == ["--gs-carrier-seed"]


def test_table5_protocol_overrides_inherited_launcher_tuning(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_PROPOSALS", "999")
    monkeypatch.setenv("ANCHORS", "999")
    monkeypatch.setenv("BENCHMARK_MODULE", "unrelated.module")
    monkeypatch.setenv("PYTHON", "/bin/false")
    args = _build_parser().parse_args(["--dry-run"])

    env = _arm_environment(
        args,
        output_dir=tmp_path,
        equation_ids=["feynman_000"],
    )

    assert env["MAX_PROPOSALS"] == "48"
    assert env["ANCHORS"] == "8"
    assert env["BENCHMARK_MODULE"].endswith(".aif_closure_benchmark")
    assert env["PYTHON"] != "/bin/false"
