# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json

import nestynet_sr.sr_search.factorized_search.oracle_method_attribution as attribution_mod
from nestynet_sr.sr_search.factorized_search.oracle_regression import aggregate_rows_by_spec


def _row(spec_id: str, profile: str, mode: str, *, success: int, best_mse: float, wall_seconds: float) -> dict:
    return {
        "spec_id": spec_id,
        "spec_path": f"{spec_id}.json",
        "profile": profile,
        "mode": mode,
        "budget": 100,
        "repeat": 0,
        "seed": 0,
        "best_mse": float(best_mse),
        "success": int(success),
        "best_expr": "x0",
        "mapping_kind": "poly1",
        "wall_seconds": float(wall_seconds),
    }


def _method_attribution_payload() -> dict:
    rows = [
        _row("s1", "residual_basin_only", "refine_off", success=0, best_mse=1.0e-2, wall_seconds=1.0),
        _row("s2", "residual_basin_only", "refine_off", success=0, best_mse=1.0e-2, wall_seconds=1.0),
        _row("s1", "inverse_spec", "refine_off", success=1, best_mse=1.0e-10, wall_seconds=1.4),
        _row("s2", "inverse_spec", "refine_off", success=0, best_mse=1.0e-3, wall_seconds=1.3),
        _row("s1", "hole_fix", "refine_off", success=1, best_mse=1.0e-12, wall_seconds=1.5),
        _row("s2", "hole_fix", "refine_off", success=1, best_mse=1.0e-12, wall_seconds=1.4),
    ]
    return {
        "suite_id": "quick12_method_attribution",
        "suite_manifest": "examples/oracle_factorized_search/regression_suites/quick12_method_attribution.json",
        "rows": rows,
        "spec_summary": aggregate_rows_by_spec(rows),
    }


def _inverse_compare_payload() -> dict:
    rows = [
        _row("s1", "current", "refine_off", success=1, best_mse=1.0e-10, wall_seconds=1.1),
        _row("s2", "current", "refine_off", success=1, best_mse=1.0e-10, wall_seconds=1.1),
        _row("s1", "no_inverse", "refine_off", success=0, best_mse=1.0e-2, wall_seconds=1.0),
        _row("s2", "no_inverse", "refine_off", success=1, best_mse=1.0e-8, wall_seconds=1.0),
    ]
    return {
        "suite_id": "quick12_inverse_compare",
        "suite_manifest": "examples/oracle_factorized_search/regression_suites/quick12_inverse_compare.json",
        "rows": rows,
        "spec_summary": aggregate_rows_by_spec(rows),
    }


def test_default_suite_comparisons_cover_method_attribution_suite():
    comps = attribution_mod.default_suite_comparisons("quick12_method_attribution")
    ids = [comp.comparison_id for comp in comps]
    assert ids == ["inverse_steering", "hole_fixing", "repair_stack"]


def test_summarize_method_attribution_recommends_promotion_for_clear_uplifts():
    report = attribution_mod.summarize_method_attribution(_method_attribution_payload())
    by_id = {row["comparison_id"]: row for row in report["comparisons"]}

    assert by_id["inverse_steering"]["recommendation"]["decision"] == "promote"
    assert by_id["hole_fixing"]["recommendation"]["decision"] == "promote"
    assert by_id["repair_stack"]["recommendation"]["decision"] == "promote"


def test_summarize_method_attribution_defaults_inverse_compare_suite():
    report = attribution_mod.summarize_method_attribution(_inverse_compare_payload())
    assert len(report["comparisons"]) == 1
    comparison = report["comparisons"][0]
    assert comparison["comparison_id"] == "inverse_steering"
    assert comparison["recommendation"]["decision"] == "promote"
    assert comparison["aggregate"]["success_loss_specs"] == 0


def test_oracle_method_attribution_main_writes_output(tmp_path):
    payload = _method_attribution_payload()
    results_path = tmp_path / "oracle_regression_results.json"
    out_path = tmp_path / "oracle_method_attribution.json"
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    rc = attribution_mod.main(
        [
            "--results",
            str(results_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["mode"] == "oracle_method_attribution"
    assert report["comparisons"]
