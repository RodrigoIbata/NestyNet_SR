# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json

import nestynet_sr.sr_search.factorized_search.capability_benchmark as capability_mod


def test_capability_suite_default_manifest_loads():
    manifest_path, payload = capability_mod.load_capability_suite()
    assert manifest_path.name == "planted_smoke.json"
    assert payload["suite_id"] == "planted_smoke"
    assert len(payload["cases"]) >= 4


def test_capability_suite_main_writes_outputs(tmp_path):
    output_dir = tmp_path / "capability"
    rc = capability_mod.main(
        [
            "--suite_manifest",
            "examples/oracle_factorized_search/capability_suites/planted_smoke.json",
            "--output_dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    results_path = output_dir / "capability_benchmark_results.json"
    summary_path = output_dir / "capability_benchmark_summary.json"
    assert results_path.is_file()
    assert summary_path.is_file()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["suite_id"] == "planted_smoke"
    rows = list(payload["rows"])
    assert any(row["case_type"] == "inverse_spec" for row in rows)
    assert any(row["case_type"] == "micro_search" for row in rows)
    inverse_rows = [row for row in rows if row["case_id"] == "inverse_simple_corrupt_hole"]
    assert {row["profile"] for row in inverse_rows} == {"flat", "recursive"}
    assert any(bool(row["truth_present"]) for row in inverse_rows)
    micro_rows = [row for row in rows if row["case_id"] == "micro_search_exp_mul"]
    assert {row["profile"] for row in micro_rows} == {"inverse", "residual"}
    assert any(row["profile"] == "inverse" and bool(row["success"]) for row in micro_rows)
