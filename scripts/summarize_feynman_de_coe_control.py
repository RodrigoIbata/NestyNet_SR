#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Summarize Feynman-DE CoE control-suite benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


CSV_FIELDS = (
    "source",
    "problem_id",
    "description",
    "status",
    "failure_kind",
    "selected_engine",
    "internal_selected_engine",
    "selected_lane",
    "typed_selected_lane",
    "whole_rhs_attempted",
    "whole_rhs_attempts_run",
    "family_gate_skips",
    "typed_explorer_launches",
    "rollout_override",
    "first_line_status",
    "rescued_additional",
    "n_traj",
    "n_fit_traj",
    "n_probe_traj",
    "holdout_last_k",
    "worst_traj_nrmse",
    "median_traj_nrmse",
    "mean_traj_nrmse",
    "stlsq_validated_candidates",
    "factorized_shortlist_size",
    "factorized_validated_candidates",
    "factorized_search_shortlist_size",
    "factorized_search_validated_candidates",
    "validated_candidates_total",
    "canonical_equation",
    "json_path",
    "message",
)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _status_count(counts: dict[str, int], key: Any) -> None:
    name = str(key or "")
    if not name:
        return
    counts[name] = int(counts.get(name, 0)) + 1


def _discover_summary_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(path.glob("de*/summary.json"))
            own = path / "summary.json"
            if own.exists():
                candidates.insert(0, own)
        else:
            candidates = [path]
        for cand in candidates:
            resolved = cand.resolve()
            if cand.exists() and resolved not in seen:
                seen.add(resolved)
                out.append(cand)
    return out


def _primary_engine_row(summary: dict[str, Any], problem: dict[str, Any]) -> dict[str, Any]:
    engines = problem.get("engines", {})
    if not isinstance(engines, dict) or not engines:
        return {}
    preferred = [
        str(summary.get("engine", "")),
        str(problem.get("engine", "")),
        str(problem.get("selected_engine", "")),
        "factorized_de",
        "hybrid",
        "factorized_search_only",
        "sparse",
        "factorized_search_oracle",
    ]
    for key in preferred:
        row = engines.get(key)
        if isinstance(row, dict):
            return row
    for row in engines.values():
        if isinstance(row, dict):
            return row
    return {}


def _first_present_number(
    keys: tuple[str, ...],
    *,
    problem: dict[str, Any],
    primary: dict[str, Any],
) -> int:
    for row in (problem, primary):
        for key in keys:
            if key in row and row.get(key) is not None:
                return _safe_int(row.get(key), 0)
    engines = problem.get("engines", {})
    if isinstance(engines, dict):
        for row in engines.values():
            if not isinstance(row, dict):
                continue
            for key in keys:
                if key in row and row.get(key) is not None:
                    return _safe_int(row.get(key), 0)
    return 0


def _traj_nrmse_summary(traj_scores: Any) -> dict[str, float | None]:
    vals: list[float] = []
    for row in list(traj_scores or []):
        if not isinstance(row, dict):
            continue
        val = _safe_float(row.get("nrmse"), float("nan"))
        if math.isfinite(val):
            vals.append(float(val))
    if not vals:
        return {
            "worst_traj_nrmse": None,
            "median_traj_nrmse": None,
            "mean_traj_nrmse": None,
        }
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    if len(vals_sorted) % 2:
        median = vals_sorted[mid]
    else:
        median = 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
    return {
        "worst_traj_nrmse": float(max(vals_sorted)),
        "median_traj_nrmse": float(median),
        "mean_traj_nrmse": float(sum(vals_sorted) / len(vals_sorted)),
    }


def _stlsq_validated(problem: dict[str, Any], primary: dict[str, Any]) -> int:
    for row in (problem, primary):
        scores = row.get("first_line_traj_scores")
        if isinstance(scores, list) and scores:
            return 1
    status = str(problem.get("first_line_status") or primary.get("first_line_status") or "")
    return 1 if status in {"PASS", "PARTIAL", "FAIL", "ERROR"} else 0


def _row_from_problem(source: Path, summary: dict[str, Any], problem: dict[str, Any]) -> dict[str, Any]:
    primary = _primary_engine_row(summary, problem)
    selected = str(problem.get("selected_engine") or primary.get("selected_engine") or "")
    internal = str(problem.get("internal_selected_engine") or primary.get("internal_selected_engine") or "")
    override = bool(selected and internal and selected != internal)
    if "internal_selected_engine_mismatch" in primary:
        override = bool(primary.get("internal_selected_engine_mismatch"))

    factorized_validated = _first_present_number(
        ("factorized_validated_candidates",),
        problem=problem,
        primary=primary,
    )
    fss_validated = _first_present_number(
        ("factorized_search_validated_candidates",),
        problem=problem,
        primary=primary,
    )
    stlsq_validated = _stlsq_validated(problem, primary)
    traj_summary = _traj_nrmse_summary(problem.get("traj_scores", primary.get("traj_scores", [])))
    return {
        "source": str(source),
        "problem_id": str(problem.get("id", "")),
        "description": str(problem.get("description", "")),
        "status": str(problem.get("status", "")),
        "failure_kind": problem.get("failure_kind"),
        "selected_engine": selected,
        "internal_selected_engine": internal,
        "selected_lane": str(problem.get("selected_lane", primary.get("selected_lane", "")) or ""),
        "typed_selected_lane": str(
            problem.get("typed_selected_lane", primary.get("typed_selected_lane", "")) or ""
        ),
        "whole_rhs_attempted": bool(
            problem.get("whole_rhs_attempted", primary.get("whole_rhs_attempted", False))
        ),
        "whole_rhs_attempts_run": _safe_int(
            problem.get("whole_rhs_attempts_run", primary.get("whole_rhs_attempts_run", 0)),
            0,
        ),
        "family_gate_skips": _safe_int(
            problem.get("family_gate_skips", primary.get("family_gate_skips", 0)),
            0,
        ),
        "typed_explorer_launches": _safe_int(
            problem.get("typed_explorer_launches", primary.get("typed_explorer_launches", 0)),
            0,
        ),
        "rollout_override": bool(override),
        "first_line_status": str(problem.get("first_line_status") or primary.get("first_line_status") or ""),
        "rescued_additional": bool(problem.get("rescued_additional", primary.get("rescued_additional", False))),
        "n_traj": _safe_int(problem.get("n_traj", 0), 0),
        "n_fit_traj": _safe_int(problem.get("n_fit_traj", 0), 0),
        "n_probe_traj": _safe_int(problem.get("n_probe_traj", 0), 0),
        "holdout_last_k": _safe_int(problem.get("holdout_last_k", 0), 0),
        **traj_summary,
        "stlsq_validated_candidates": int(stlsq_validated),
        "factorized_shortlist_size": _first_present_number(
            ("factorized_shortlist_size",),
            problem=problem,
            primary=primary,
        ),
        "factorized_validated_candidates": int(factorized_validated),
        "factorized_search_shortlist_size": _first_present_number(
            ("factorized_search_shortlist_size",),
            problem=problem,
            primary=primary,
        ),
        "factorized_search_validated_candidates": int(fss_validated),
        "validated_candidates_total": int(stlsq_validated + factorized_validated + fss_validated),
        "canonical_equation": str(problem.get("canonical_equation", "")),
        "json_path": str(problem.get("json_path", primary.get("json_path", ""))),
        "message": str(problem.get("message", "")),
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    summary_paths = _discover_summary_paths(paths)
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    internal_counts: dict[str, int] = {}
    selected_lane_counts: dict[str, int] = {}
    typed_lane_counts: dict[str, int] = {}
    failure_kind_counts: dict[str, int] = {}

    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for problem in list(payload.get("problems", []) or []):
            if not isinstance(problem, dict):
                continue
            row = _row_from_problem(path, payload, problem)
            rows.append(row)
            _status_count(status_counts, row.get("status"))
            _status_count(selected_counts, row.get("selected_engine"))
            _status_count(internal_counts, row.get("internal_selected_engine"))
            _status_count(selected_lane_counts, row.get("selected_lane"))
            _status_count(typed_lane_counts, row.get("typed_selected_lane"))
            _status_count(failure_kind_counts, row.get("failure_kind"))

    rollout_override_ids = [
        str(row.get("problem_id", ""))
        for row in rows
        if bool(row.get("rollout_override"))
    ]
    pass_rows = [row for row in rows if str(row.get("status")) == "PASS"]
    return {
        "n_reports": int(len(summary_paths)),
        "n_rows": int(len(rows)),
        "status_counts": status_counts,
        "selected_engine_counts": selected_counts,
        "internal_selected_engine_counts": internal_counts,
        "selected_lane_counts": selected_lane_counts,
        "typed_selected_lane_counts": typed_lane_counts,
        "failure_kind_counts": failure_kind_counts,
        "rollout_override_count": int(len(rollout_override_ids)),
        "rollout_override_ids": rollout_override_ids,
        "pass_count": int(len(pass_rows)),
        "validated_candidates_total": int(
            sum(_safe_int(row.get("validated_candidates_total", 0), 0) for row in rows)
        ),
        "whole_rhs_attempted": int(sum(1 for row in rows if bool(row.get("whole_rhs_attempted", False)))),
        "whole_rhs_attempts_run": int(sum(_safe_int(row.get("whole_rhs_attempts_run", 0), 0) for row in rows)),
        "family_gate_skips": int(sum(_safe_int(row.get("family_gate_skips", 0), 0) for row in rows)),
        "typed_explorer_launches": int(sum(_safe_int(row.get("typed_explorer_launches", 0), 0) for row in rows)),
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, None) for key in CSV_FIELDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="summary.json files or result directories")
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = summarize([Path(p) for p in args.inputs])
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.csv_path is not None:
        write_csv(args.csv_path, list(payload.get("rows", []) or []))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
