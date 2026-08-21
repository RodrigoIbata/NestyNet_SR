#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Summarize skeleton-refinement micro-benchmark JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


COUNT_FIELDS = (
    "score_calls",
    "refine_score_calls",
    "refinement_attempts",
    "hparam_optimizations",
    "grid_evals",
    "lbfgs_runs",
    "lbfgs_closures",
    "linear_solves",
    "linear_solves_multi",
    "materialized_rescores",
    "accepted_refinements",
    "attempt_cache_hits",
    "attempt_cache_misses",
    "attempt_cache_stores",
    "attempt_cache_size",
    "mapping_equiv_root_slots_pruned",
    "brute_refinement_attempts",
    "mutation_refinement_attempts",
    "slate_refinement_attempts",
    "controller_slate_refinement_attempts",
    "external_refinement_attempts",
)

CSV_FIELDS = (
    "source",
    "id",
    "case",
    "gate_kind",
    "base_mse",
    "refined_mse",
    "improvement_ratio",
    "trials_done",
    "elapsed_base_s",
    "elapsed_refine_s",
    *COUNT_FIELDS,
    "accepted_per_attempt",
    "attempt_cache_hit_rate",
    "grid_evals_per_attempt",
    "lbfgs_closures_per_attempt",
    "linear_solves_per_attempt",
)


def _number(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _merge_numeric(total: dict[str, Any], row: dict[str, Any]) -> None:
    for key, value in row.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            total[str(key)] = int(total.get(str(key), 0) or 0) + int(value)
            continue
        try:
            fv = float(value)
        except Exception:
            continue
        if math.isfinite(fv):
            total[str(key)] = float(total.get(str(key), 0.0) or 0.0) + fv


def _cost_summary(diag: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {key: int(round(_number(diag.get(key, 0)))) for key in COUNT_FIELDS if key in diag}
    for key in ("base_score_s", "hparam_optimization_s"):
        if key in diag:
            out[key] = _number(diag.get(key))
    attempts = max(_number(diag.get("refinement_attempts")), _number(diag.get("hparam_optimizations")))
    hits = _number(diag.get("attempt_cache_hits"))
    misses = _number(diag.get("attempt_cache_misses"))
    lookups = hits + misses
    accepted = _number(diag.get("accepted_refinements"))
    grid = _number(diag.get("grid_evals"))
    lbfgs = _number(diag.get("lbfgs_closures"))
    linear = _number(diag.get("linear_solves")) + _number(diag.get("linear_solves_multi"))
    out["accepted_per_attempt"] = None if attempts <= 0.0 else accepted / attempts
    out["attempt_cache_hit_rate"] = None if lookups <= 0.0 else hits / lookups
    out["grid_evals_per_attempt"] = None if attempts <= 0.0 else grid / attempts
    out["lbfgs_closures_per_attempt"] = None if attempts <= 0.0 else lbfgs / attempts
    out["linear_solves_per_attempt"] = None if attempts <= 0.0 else linear / attempts
    return out


def _row_from_case(source: Path, row: dict[str, Any]) -> dict[str, Any]:
    diag = dict(row.get("refine_diagnostics", {}) or {})
    summary = dict(row.get("refine_cost_summary", {}) or _cost_summary(diag))
    out = {
        "source": str(source),
        "id": str(row.get("id", "")),
        "case": str(row.get("case", "")),
        "gate_kind": str(row.get("gate_kind", "")),
        "base_mse": _number(row.get("base_mse")),
        "refined_mse": _number(row.get("refined_mse")),
        "improvement_ratio": _number(row.get("improvement_ratio")),
        "trials_done": int(_number(row.get("trials_done"))),
        "elapsed_base_s": _number(row.get("elapsed_base_s")),
        "elapsed_refine_s": _number(row.get("elapsed_refine_s")),
    }
    for key in COUNT_FIELDS:
        out[key] = int(round(_number(summary.get(key, diag.get(key, 0)))))
    for key in (
        "accepted_per_attempt",
        "attempt_cache_hit_rate",
        "grid_evals_per_attempt",
        "lbfgs_closures_per_attempt",
        "linear_solves_per_attempt",
    ):
        out[key] = summary.get(key, None)
    return out


def summarize(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    diagnostics_total: dict[str, Any] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload_diag = payload.get("refine_diagnostics", None)
        used_payload_diag = isinstance(payload_diag, dict)
        if used_payload_diag:
            _merge_numeric(diagnostics_total, payload_diag)
        for row in payload.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            rows.append(_row_from_case(path, row))
            diag = row.get("refine_diagnostics", None)
            if (not used_payload_diag) and isinstance(diag, dict):
                _merge_numeric(diagnostics_total, diag)
    return {
        "n_reports": int(len(paths)),
        "n_rows": int(len(rows)),
        "refine_diagnostics": diagnostics_total,
        "refine_cost_summary": _cost_summary(diagnostics_total),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = summarize([Path(p) for p in args.inputs])
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.csv_path is not None:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
            for row in summary["rows"]:
                writer.writerow({key: row.get(key, None) for key in CSV_FIELDS})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
