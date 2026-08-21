# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json

import nestynet_sr.sr_search.factorized_search.hole_search_archetype_benchmark as archetype_mod


def test_hole_search_archetype_suite_default_manifest_loads():
    manifest_path, payload = archetype_mod.load_hole_search_archetype_suite()
    assert manifest_path.name == "hole_search_archetypes_smoke.json"
    assert payload["suite_id"] == "hole_search_archetypes_smoke"
    assert len(payload["cases"]) == 3


def test_hole_search_archetype_suite_main_writes_outputs(tmp_path):
    output_dir = tmp_path / "hole_search_archetypes"
    rc = archetype_mod.main(
        [
            "--suite_manifest",
            "examples/oracle_factorized_search/capability_suites/hole_search_archetypes_smoke.json",
            "--output_dir",
            str(output_dir),
            "--save_individual_reports",
        ]
    )
    assert rc == 0

    results_path = output_dir / "hole_search_archetype_benchmark_results.json"
    summary_path = output_dir / "hole_search_archetype_benchmark_summary.json"
    assert results_path.is_file()
    assert summary_path.is_file()
    assert (output_dir / "cases" / "drifting_constant.json").is_file()
    assert (output_dir / "cases" / "single_index_coordinate.json").is_file()
    assert (output_dir / "cases" / "near_miss_tangent_edit.json").is_file()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "hole_search_archetype_benchmark"
    assert payload["suite_id"] == "hole_search_archetypes_smoke"
    rows = {row["case_id"]: row for row in payload["rows"]}
    assert set(rows) == {
        "drifting_constant",
        "single_index_coordinate",
        "near_miss_tangent_edit",
    }
    assert rows["drifting_constant"]["selected_route"] == "constant_lift_route"
    assert rows["single_index_coordinate"]["selected_route"] == "coordinate_lift"
    assert rows["near_miss_tangent_edit"]["selected_route"] in {"tangent_edit", "soft_edit_search"}
    assert all(bool(row["success"]) for row in rows.values())

    summary = payload["summary"]
    assert summary["n_rows"] == 3
    assert summary["success_rate"] == 1.0
