# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json

import nestynet_sr.sr_search.factorized_search.oracle_factorial_benchmark as factorial_mod


def test_factorial_manifest_resolves_specs():
    manifest_path, payload = factorial_mod.load_factorial_suite()
    specs = factorial_mod.resolve_suite_spec_paths(payload, manifest_path=manifest_path)
    assert payload["suite_id"] == "reduced_quick4"
    assert len(specs) == 4
    assert all(path.is_file() for path in specs)


def test_enumerate_factorial_arms_has_full_16_arm_grid():
    arms = factorial_mod.enumerate_factorial_arms()
    arm_ids = {arm["arm_id"] for arm in arms}
    assert len(arms) == 16
    assert "plus0_inv0_spec0_hole0" in arm_ids
    assert "plus1_inv1_spec1_hole1" in arm_ids


def test_factorial_main_writes_outputs(monkeypatch, tmp_path):
    def _fake_run_factorial_job(job):
        arm = dict(job["arm"])
        toggles = dict(arm["toggles"])
        row = {
            "status": "ok",
            "spec_id": "toy",
            "spec_path": str(job["spec_path"]),
            "budget": int(job["budget"]),
            "seed": int(job["seed"]),
            "arm_id": str(arm["arm_id"]),
            "refine_enable": bool(toggles["refine_enable"]),
            "inverse_steering_enable": bool(toggles["inverse_steering_enable"]),
            "inverse_spec_enable": bool(toggles["inverse_spec_enable"]),
            "hole_search_enable": bool(toggles["hole_search_enable"]),
            "effective_refine_enable": bool(toggles["refine_enable"]),
            "effective_inverse_steering_enable": bool(toggles["inverse_steering_enable"]),
            "effective_inverse_spec_enable": bool(toggles["inverse_spec_enable"] and toggles["inverse_steering_enable"]),
            "effective_hole_search_enable": bool(
                toggles["hole_search_enable"] and toggles["inverse_spec_enable"] and toggles["inverse_steering_enable"]
            ),
            "best_mse": 0.0 if toggles["inverse_steering_enable"] else 1.0,
            "success": 1 if toggles["inverse_steering_enable"] else 0,
            "best_expr": "x0",
            "mapping_kind": "poly1",
            "wall_seconds": 0.1,
        }
        return {"row": row, "report": {"best": {"expr": "x0", "mse": row["best_mse"]}}}

    monkeypatch.setattr(factorial_mod, "_run_factorial_job", _fake_run_factorial_job)

    manifest_path = tmp_path / "suite.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_id": "toy_factorial",
                "defaults": {"budgets": [50], "seeds": [0], "jobs": 1, "quiet": True, "dtype": "float64"},
                "specs": ["examples/oracle_factorized_search/specs/feynman_037.json"],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "factorial"
    rc = factorial_mod.main(
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
    results_path = output_dir / "oracle_factorial_results.json"
    assert results_path.is_file()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["suite_id"] == "toy_factorial"
    assert len(payload["rows"]) == 16
    assert len(payload["arm_summary"]) == 16


def test_factorial_write_csv_tolerates_error_rows(tmp_path):
    rows = [
        {"status": "ok", "spec_id": "toy_ok", "best_mse": 0.25},
        {"status": "error", "spec_id": "toy_err", "error": "boom"},
    ]
    csv_path = tmp_path / "rows.csv"

    factorial_mod._write_csv(rows, csv_path)

    text = csv_path.read_text(encoding="utf-8")
    assert "status" in text
    assert "error" in text
    assert "boom" in text
