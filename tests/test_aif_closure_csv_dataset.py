# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import csv
from pathlib import Path

import pytest
import torch

from nestynet_sr.sr_search.factorized_search.aif_closure_benchmark import (
    _select_benchmark_best,
    build_csv_oracle_dataset,
    resolve_aif_csv_path,
)
from nestynet_sr.sr_search.factorized_search.oracle_lab import (
    compile_target_expression,
    equation_spec_from_dict,
)


def _simple_spec():
    return {
        "id": "feynman_000",
        "basis": ["L", "T"],
        "variables": [
            {"name": "x0", "bounds": [0.0, 10.0], "dim": [0, 0]},
            {"name": "x1", "bounds": [0.0, 10.0], "dim": [0, 0]},
        ],
        "constants": [],
        "target": {"expr": "x0 + 2*x1", "dim": [0, 0]},
    }


def _write_csv(path: Path, rows: list[tuple[float, float, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["y", "x0", "x1"])
        writer.writerows(rows)


def test_resolve_aif_csv_path(tmp_path: Path):
    expected = tmp_path / "pb037_I_37_4_data.csv"
    expected.write_text("y,x0\n1,1\n")

    assert resolve_aif_csv_path("feynman_037", tmp_path) == expected
    assert resolve_aif_csv_path("pb037", tmp_path) == expected
    assert resolve_aif_csv_path("37", tmp_path) == expected


def test_build_csv_oracle_dataset_uses_disjoint_slice_and_checks_y(tmp_path: Path):
    spec_dict = _simple_spec()
    spec = equation_spec_from_dict(spec_dict)
    target_fn = compile_target_expression(spec)
    rows = [
        (x0 + 2 * x1, x0, x1)
        for x0, x1 in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    ]
    csv_path = tmp_path / "pb000_demo_data.csv"
    _write_csv(csv_path, rows)

    ds = build_csv_oracle_dataset(
        spec_dict,
        csv_path=csv_path,
        target_fn=target_fn,
        n_fit=2,
        n_probe=2,
        data_slice=1,
        dtype=torch.float64,
    )

    assert ds["x_fit"].tolist() == [[4.0, 5.0], [5.0, 6.0]]
    assert ds["y_fit"].tolist() == [[14.0], [17.0]]
    assert ds["x_probe"].tolist() == [[6.0, 7.0], [7.0, 8.0]]
    assert ds["metadata"]["row_start"] == 4
    assert ds["metadata"]["row_stop"] == 8


def test_build_csv_oracle_dataset_rejects_y_mismatch(tmp_path: Path):
    spec_dict = _simple_spec()
    spec = equation_spec_from_dict(spec_dict)
    target_fn = compile_target_expression(spec)
    csv_path = tmp_path / "pb000_bad_data.csv"
    _write_csv(csv_path, [(3.0, 1.0, 1.0), (999.0, 2.0, 2.0)])

    with pytest.raises(ValueError, match="CSV y does not match oracle target"):
        build_csv_oracle_dataset(
            spec_dict,
            csv_path=csv_path,
            target_fn=target_fn,
            n_fit=1,
            n_probe=1,
            dtype=torch.float64,
        )


def test_build_csv_oracle_dataset_uses_oracle_y_when_check_disabled(tmp_path: Path):
    spec_dict = _simple_spec()
    spec = equation_spec_from_dict(spec_dict)
    target_fn = compile_target_expression(spec)
    csv_path = tmp_path / "pb000_bad_data.csv"
    _write_csv(csv_path, [(3.0, 1.0, 1.0), (999.0, 2.0, 2.0)])

    ds = build_csv_oracle_dataset(
        spec_dict,
        csv_path=csv_path,
        target_fn=target_fn,
        n_fit=1,
        n_probe=1,
        dtype=torch.float64,
        y_check=False,
    )

    assert ds["y_fit"].tolist() == [[3.0]]
    assert ds["y_probe"].tolist() == [[6.0]]
    assert ds["metadata"]["y_source"] == "oracle_expression"
    assert ds["metadata"]["y_check"]["enabled"] is False
    assert ds["metadata"]["y_check"]["max_abs"] == 993.0


def test_select_benchmark_best_promotes_solved_full_validation_audit():
    result = {
        "best": {
            "expr": "search-fit winner",
            "mse": 1.0e-12,
            "full_validation": {"probe_mse": 1.0e-1},
        },
        "best_full_audit": {
            "expr": "validated winner",
            "mse": 2.0e-12,
            "full_validation": {"probe_mse": 1.0e-11},
        },
    }

    best, source = _select_benchmark_best(
        result,
        final_validate_rerank=False,
        success_mse=1.0e-6,
    )

    assert source == "solved_full_audit"
    assert best["expr"] == "validated winner"


def test_select_benchmark_best_keeps_audit_only_for_unsolved_candidates():
    result = {
        "best": {"expr": "search winner", "mse": 1.0e-4},
        "best_full_audit": {
            "expr": "full-validation winner",
            "mse": 2.0e-4,
            "full_validation": {"probe_mse": 1.0e-5},
        },
    }

    best, source = _select_benchmark_best(
        result,
        final_validate_rerank=False,
        success_mse=1.0e-6,
    )

    assert source == "search"
    assert best["expr"] == "search winner"
