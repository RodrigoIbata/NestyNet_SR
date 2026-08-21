#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Run the AI Feynman oracle benchmark with the factorized symbolic search closure machine.

Usage:
  python -m nestynet_sr.sr_search.factorized_search.aif_closure_benchmark
  python -m nestynet_sr.sr_search.factorized_search.aif_closure_benchmark --only 037,090
  python -m nestynet_sr.sr_search.factorized_search.aif_closure_benchmark --n_iter 2000 --max_proposals 64
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch


def parse_equations_txt(path: str | Path) -> list[dict]:
    """Parse equations.txt into a list of spec dicts."""
    specs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split: id  vars  xmin  xmax  expr  y_units  x_units
            # The tricky part is that vars/xmin/xmax/units are bracket-delimited
            parts = line.split(None, 1)
            eq_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            # Extract bracketed fields from the end (x_units, then y_units)
            x_units_str, rest2 = _extract_last_bracketed(rest)
            y_units_str, rest3 = _extract_last_bracketed(rest2)

            # The remaining is: vars  xmin  xmax  expr
            # Extract vars (bracketed), xmin (bracketed), xmax (bracketed), then expr
            vars_str, after_vars = _extract_first_bracketed(rest3)
            xmin_str, after_xmin = _extract_first_bracketed(after_vars)
            xmax_str, after_xmax = _extract_first_bracketed(after_xmin)
            expr = after_xmax.strip()

            try:
                var_names = _parse_list(vars_str)
                xmin = _parse_float_list(xmin_str)
                xmax = _parse_float_list(xmax_str)
                y_dims = _parse_float_list(y_units_str)
                x_dims = _parse_nested_list(x_units_str)
            except Exception as e:
                print(f"WARNING: skipping {eq_id}: {e}")
                continue

            nvars = len(var_names)
            if len(xmin) != nvars or len(xmax) != nvars or len(x_dims) != nvars:
                print(f"WARNING: skipping {eq_id}: variable count mismatch")
                continue

            variables = []
            for i in range(nvars):
                variables.append({
                    "name": var_names[i],
                    "bounds": [float(xmin[i]), float(xmax[i])],
                    "dim": [float(d) for d in x_dims[i]],
                })

            specs.append({
                "id": f"feynman_{eq_id}",
                "basis": ["L", "T", "M", "I", "Theta"],
                "variables": variables,
                "constants": [],
                "target": {
                    "expr": expr,
                    "dim": [float(d) for d in y_dims],
                },
            })

    return specs


def _extract_last_bracketed(s: str) -> tuple[str, str]:
    """Extract the last [...] or [[...]] from a string."""
    s = s.rstrip()
    if not s.endswith("]"):
        return "", s
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] == "]":
            depth += 1
        elif s[i] == "[":
            depth -= 1
            if depth == 0:
                return s[i:], s[:i].rstrip()
    return "", s


def _extract_first_bracketed(s: str) -> tuple[str, str]:
    """Extract the first [...] from a string."""
    s = s.lstrip()
    if not s.startswith("["):
        return "", s
    depth = 0
    for i in range(len(s)):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                return s[: i + 1], s[i + 1 :].lstrip()
    return "", s


def _parse_list(s: str) -> list[str]:
    """Parse ['x0' 'x1'] or ['x0', 'x1']."""
    s = s.strip()
    if not s:
        return []
    # Handle numpy-style ['x0' 'x1'] (space-separated, no commas)
    s = s.replace("'", '"')
    s = re.sub(r'"\s+"', '", "', s)
    return json.loads(s)


def _parse_float_list(s: str) -> list[float]:
    """Parse [1. 3.] or [1.0, 3.0]."""
    s = s.strip()
    if not s:
        return []
    # Handle numpy-style [1. 3.] (space-separated)
    inner = s.strip("[]").strip()
    parts = inner.replace(",", " ").split()
    return [float(p) for p in parts]


def _parse_nested_list(s: str) -> list[list[float]]:
    """Parse [[0.0, 0.0], [1.0, 0.0]] or [[0. 0.] [1. 0.]]."""
    s = s.strip()
    if not s:
        return []
    # Normalise numpy-style to JSON
    s = re.sub(r"(\d)\s+([\d\-])", r"\1, \2", s)
    s = re.sub(r"\.(\s)", r".0\1", s)
    s = re.sub(r"\.\]", ".0]", s)
    s = re.sub(r"\]\s*\[", "], [", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def normalize_feynman_id(raw: str) -> str:
    """Return a three-digit AI Feynman ID from 37, 037, pb037, or feynman_037."""

    value = str(raw).strip()
    if value.startswith("feynman_"):
        value = value[len("feynman_") :]
    if value.startswith("pb"):
        value = value[2:]
    if not value.isdigit():
        raise ValueError(f"Invalid AI Feynman ID: {raw!r}")
    return f"{int(value):03d}"


def resolve_aif_csv_path(eq_id: str, data_dir: str | Path) -> Path:
    """Resolve ``pbNNN_*_data.csv`` for an AI Feynman ID inside a data directory."""

    root = Path(data_dir)
    norm = normalize_feynman_id(eq_id)
    matches = sorted(root.glob(f"pb{norm}_*_data.csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV found for feynman_{norm} under {root}")
    if len(matches) > 1:
        joined = ", ".join(str(p) for p in matches)
        raise ValueError(f"Multiple CSV files found for feynman_{norm}: {joined}")
    return matches[0]


def _read_csv_dataset(path: str | Path) -> tuple[list[str], list[str], list[dict[str, float]]]:
    csv_path = Path(path)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: CSV has no header")
        fields = [str(name) for name in reader.fieldnames]
        y_cols = [name for name in fields if name.startswith("y")]
        if len(y_cols) != 1:
            raise ValueError(f"{csv_path}: expected exactly one y* column, got {y_cols}")
        x_cols = [name for name in fields if name not in y_cols]
        rows: list[dict[str, float]] = []
        for row_idx, raw in enumerate(reader, start=2):
            parsed: dict[str, float] = {}
            for name in fields:
                try:
                    value = float(raw[name])
                except Exception as exc:
                    raise ValueError(f"{csv_path}:{row_idx}: invalid numeric value in {name!r}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{csv_path}:{row_idx}: non-finite value in {name!r}")
                parsed[name] = value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{csv_path}: CSV contains no data rows")
    return x_cols, y_cols, rows


def _slice_external_rows(
    rows: list[dict[str, float]],
    *,
    n_fit: int,
    n_probe: int,
    data_slice: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]], int, int]:
    if n_fit <= 0:
        raise ValueError(f"n_fit must be positive for CSV oracle data, got {n_fit}")
    if n_probe <= 0:
        raise ValueError(f"n_probe must be positive for CSV oracle data, got {n_probe}")
    if data_slice < 0:
        raise ValueError(f"data_slice must be >= 0, got {data_slice}")
    block = int(n_fit) + int(n_probe)
    start = int(data_slice) * block
    train_stop = start + int(n_fit)
    probe_stop = train_stop + int(n_probe)
    if probe_stop > len(rows):
        raise ValueError(
            f"CSV split exceeds row count: need rows [{start}:{probe_stop}) "
            f"for n_fit={n_fit}, n_probe={n_probe}, data_slice={data_slice}, "
            f"but CSV has {len(rows)} rows"
        )
    return rows[start:train_stop], rows[train_stop:probe_stop], start, probe_stop


def _rows_to_tensors(
    rows: list[dict[str, float]],
    *,
    variable_names: list[str],
    constants: list[dict[str, Any]],
    y_col: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    missing = [name for name in variable_names if name not in rows[0]]
    if missing:
        raise ValueError(f"CSV is missing required variable column(s): {missing}")
    x_values: list[list[float]] = []
    y_values: list[list[float]] = []
    for row in rows:
        x_row = [float(row[name]) for name in variable_names]
        for const in constants:
            x_row.append(float(const["value"]))
        x_values.append(x_row)
        y_values.append([float(row[y_col])])
    return (
        torch.tensor(x_values, dtype=dtype),
        torch.tensor(y_values, dtype=dtype),
    )


def _y_check_metrics(
    *,
    y_fit_csv: torch.Tensor,
    y_fit_oracle: torch.Tensor,
    y_probe_csv: torch.Tensor,
    y_probe_oracle: torch.Tensor,
) -> dict[str, float | int]:
    csv_y = torch.cat([y_fit_csv.reshape(-1, 1), y_probe_csv.reshape(-1, 1)], dim=0)
    oracle_y = torch.cat([y_fit_oracle.reshape(-1, 1), y_probe_oracle.reshape(-1, 1)], dim=0)
    diff = torch.abs(csv_y - oracle_y)
    scale = torch.clamp(torch.maximum(torch.abs(csv_y), torch.abs(oracle_y)), min=1.0)
    rel = diff / scale
    return {
        "n_checked": int(diff.numel()),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean((csv_y - oracle_y) ** 2)).item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
    }


def build_csv_oracle_dataset(
    spec_dict: dict,
    *,
    csv_path: str | Path,
    target_fn,
    n_fit: int,
    n_probe: int,
    data_slice: int = 0,
    dtype: torch.dtype = torch.float64,
    y_check: bool = True,
    y_check_abs_tol: float = 1.0e-8,
    y_check_rel_tol: float = 1.0e-8,
) -> dict[str, Any]:
    """Build an oracle dataset from an external AI Feynman CSV.

    Rows follow the normal SR convention: the first ``n_fit`` rows in the
    selected block are training rows and the next ``n_probe`` rows are probe
    / validation rows. CSV ``y`` is checked against the exact target expression
    before the search uses it.
    """

    x_cols, y_cols, rows = _read_csv_dataset(csv_path)
    train_rows, probe_rows, row_start, row_stop = _slice_external_rows(
        rows,
        n_fit=int(n_fit),
        n_probe=int(n_probe),
        data_slice=int(data_slice),
    )
    variable_names = [str(v["name"]) for v in spec_dict["variables"]]
    constants = list(spec_dict.get("constants", []) or [])
    x_fit, y_fit_csv = _rows_to_tensors(
        train_rows,
        variable_names=variable_names,
        constants=constants,
        y_col=y_cols[0],
        dtype=dtype,
    )
    x_probe, y_probe_csv = _rows_to_tensors(
        probe_rows,
        variable_names=variable_names,
        constants=constants,
        y_col=y_cols[0],
        dtype=dtype,
    )

    y_fit_oracle = target_fn(x_fit).to(dtype=dtype).reshape(-1, 1)
    y_probe_oracle = target_fn(x_probe).to(dtype=dtype).reshape(-1, 1)
    metrics = _y_check_metrics(
        y_fit_csv=y_fit_csv,
        y_fit_oracle=y_fit_oracle,
        y_probe_csv=y_probe_csv,
        y_probe_oracle=y_probe_oracle,
    )
    metrics["abs_tol"] = float(y_check_abs_tol)
    metrics["rel_tol"] = float(y_check_rel_tol)
    metrics["enabled"] = bool(y_check)
    if y_check:
        csv_y = torch.cat([y_fit_csv.reshape(-1, 1), y_probe_csv.reshape(-1, 1)], dim=0)
        oracle_y = torch.cat([y_fit_oracle.reshape(-1, 1), y_probe_oracle.reshape(-1, 1)], dim=0)
        diff = torch.abs(csv_y - oracle_y)
        scale = torch.clamp(torch.maximum(torch.abs(csv_y), torch.abs(oracle_y)), min=1.0)
        allowed = float(y_check_abs_tol) + float(y_check_rel_tol) * scale
        if bool((diff > allowed).any().item()):
            raise ValueError(
                f"{csv_path}: CSV y does not match oracle target for {spec_dict['id']} "
                f"(max_abs={metrics['max_abs']:.3e}, max_rel={metrics['max_rel']:.3e}, "
                f"abs_tol={float(y_check_abs_tol):.3e}, rel_tol={float(y_check_rel_tol):.3e})"
            )

    return {
        "x_fit": x_fit,
        "y_fit": y_fit_oracle,
        "x_probe": x_probe,
        "y_probe": y_probe_oracle,
        "var_dims": [tuple(float(d) for d in v["dim"]) for v in spec_dict["variables"]]
        + [tuple(float(d) for d in c["dim"]) for c in constants],
        "y_dims": tuple(float(d) for d in spec_dict["target"]["dim"]),
        "metadata": {
            "source": "csv",
            "path": str(Path(csv_path)),
            "x_columns": list(x_cols),
            "y_column": str(y_cols[0]),
            "y_source": "oracle_expression",
            "variable_columns": variable_names,
            "n_fit": int(n_fit),
            "n_probe": int(n_probe),
            "data_slice": int(data_slice),
            "row_start": int(row_start),
            "row_stop": int(row_stop),
            "y_check": metrics,
        },
    }


def _strided_indices(n_rows: int, n_take: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a deterministic subset spread across ``range(n_rows)``."""

    n_rows = int(n_rows)
    n_take = int(n_take)
    if n_take <= 0:
        raise ValueError(f"subset size must be positive, got {n_take}")
    if n_take > n_rows:
        raise ValueError(f"subset size {n_take} exceeds available rows {n_rows}")
    if n_take == n_rows:
        return torch.arange(n_rows, dtype=torch.long, device=device)
    return (torch.arange(n_take, dtype=torch.long, device=device) * n_rows) // n_take


def _subset_oracle_dataset(
    dataset: dict[str, Any],
    *,
    search_n_fit: int | None,
    search_n_probe: int | None,
) -> dict[str, Any]:
    """Create the search-time subset while preserving full split metadata."""

    n_fit_full = int(dataset["x_fit"].shape[0])
    n_probe_full = int(dataset["x_probe"].shape[0])
    n_fit_search = n_fit_full if search_n_fit is None else int(search_n_fit)
    n_probe_search = n_probe_full if search_n_probe is None else int(search_n_probe)
    fit_idx = _strided_indices(n_fit_full, n_fit_search, device=dataset["x_fit"].device)
    probe_idx = _strided_indices(n_probe_full, n_probe_search, device=dataset["x_probe"].device)

    metadata = dict(dataset.get("metadata", {}) or {})
    metadata["full_split"] = {
        "n_fit": int(n_fit_full),
        "n_probe": int(n_probe_full),
        "row_start": metadata.get("row_start", None),
        "row_stop": metadata.get("row_stop", None),
    }
    metadata["n_fit"] = int(n_fit_search)
    metadata["n_probe"] = int(n_probe_search)
    metadata["search_subset"] = {
        "enabled": bool(n_fit_search != n_fit_full or n_probe_search != n_probe_full),
        "strategy": "strided_floor",
        "source_n_fit": int(n_fit_full),
        "source_n_probe": int(n_probe_full),
        "n_fit": int(n_fit_search),
        "n_probe": int(n_probe_search),
    }

    return {
        "x_fit": dataset["x_fit"].index_select(0, fit_idx),
        "y_fit": dataset["y_fit"].index_select(0, fit_idx),
        "x_probe": dataset["x_probe"].index_select(0, probe_idx),
        "y_probe": dataset["y_probe"].index_select(0, probe_idx),
        "var_dims": list(dataset["var_dims"]),
        "y_dims": tuple(dataset["y_dims"]),
        "metadata": metadata,
    }


def _json_ast_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_json_ast_to_tuple(v) for v in value)
    return value


def _finite_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    if not torch.is_tensor(pred) or pred.shape != target.shape:
        return float("inf")
    if not torch.isfinite(pred).all():
        return float("inf")
    out = float(((target - pred) ** 2).mean().item())
    return out if math.isfinite(out) else float("inf")


def _validate_candidate_on_dataset(row: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one returned candidate on a validation dataset without refitting."""

    from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node, is_valid_node
    from nestynet_sr.sr_search.factorized_search.inverse_core import eval_mapping_total

    node = _json_ast_to_tuple(row.get("expr_ast"))
    mapping = dict(row.get("mapping") or {})
    if not is_valid_node(node):
        raise ValueError("candidate row has no valid expr_ast")

    with torch.no_grad():
        pred_fit = eval_node(node, dataset["x_fit"])
        pred_probe = eval_node(node, dataset["x_probe"])
        yhat_fit = eval_mapping_total(pred_fit, mapping, dataset["x_fit"])
        yhat_probe = eval_mapping_total(pred_probe, mapping, dataset["x_probe"])
        fit_mse = _finite_mse(yhat_fit, dataset["y_fit"])
        probe_mse = _finite_mse(yhat_probe, dataset["y_probe"])
        n_fit = int(dataset["x_fit"].shape[0])
        n_probe = int(dataset["x_probe"].shape[0])
        combined_num = fit_mse * n_fit + probe_mse * n_probe
        combined_den = max(1, n_fit + n_probe)
        combined_mse = float(combined_num / combined_den)

    return {
        "fit_mse": fit_mse,
        "probe_mse": probe_mse,
        "combined_mse": combined_mse,
        "n_fit": n_fit,
        "n_probe": n_probe,
    }


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(str(name))
    if raw is None or str(raw).strip() == "":
        return bool(default)
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _attach_full_validation(report: dict[str, Any], dataset: dict[str, Any], *, allow_rerank: bool = False) -> None:
    rows = list(report.get("results") or [])
    best_idx = None
    best_probe_mse = float("inf")
    failures = 0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        try:
            validation = _validate_candidate_on_dataset(row, dataset)
        except Exception as exc:
            validation = {
                "error": str(exc),
                "fit_mse": float("inf"),
                "probe_mse": float("inf"),
                "combined_mse": float("inf"),
                "n_fit": int(dataset["x_fit"].shape[0]),
                "n_probe": int(dataset["x_probe"].shape[0]),
            }
            failures += 1
        row["full_validation"] = validation
        ladder = row.get("score_ladder", None)
        if isinstance(ladder, dict):
            ladder = dict(ladder)
        else:
            ladder = {}
        ladder["final_validation"] = {
            "available": True,
            "source": "full_split_no_refit",
            "fit_mse": validation.get("fit_mse", None),
            "probe_mse": validation.get("probe_mse", None),
            "combined_mse": validation.get("combined_mse", None),
            "n_fit": validation.get("n_fit", None),
            "n_probe": validation.get("n_probe", None),
            "error": validation.get("error", None),
        }
        row["score_ladder"] = ladder
        row["final_validated_mse"] = validation.get("probe_mse", None)
        row["final_acceptance_basis"] = "full_validation_audit"
        probe_mse = float(validation.get("probe_mse", float("inf")))
        if math.isfinite(probe_mse) and probe_mse < best_probe_mse:
            best_probe_mse = probe_mse
            best_idx = idx

    metadata = dict(dataset.get("metadata", {}) or {})
    report["full_validation"] = {
        "enabled": True,
        "rerank_enabled": bool(allow_rerank),
        "dataset": metadata,
        "candidate_count": int(len(rows)),
        "failures": int(failures),
        "best_index": best_idx,
        "best_probe_mse": None if not math.isfinite(best_probe_mse) else float(best_probe_mse),
    }
    report["best_full_audit"] = rows[best_idx] if best_idx is not None else None
    if bool(allow_rerank):
        report["best_full"] = rows[best_idx] if best_idx is not None else None


def _full_validation_probe_mse(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return float("inf")
    validation = row.get("full_validation")
    if not isinstance(validation, dict):
        return float("inf")
    try:
        out = float(validation.get("probe_mse", float("inf")))
    except (TypeError, ValueError):
        return float("inf")
    return out if math.isfinite(out) else float("inf")


def _select_benchmark_best(
    result: dict[str, Any],
    *,
    final_validate_rerank: bool,
    success_mse: float,
) -> tuple[dict[str, Any], str]:
    """Choose the reported benchmark row without turning audit mode into broad reranking."""

    best_search = result.get("best", {}) or {}
    if not isinstance(best_search, dict):
        best_search = {}

    if bool(final_validate_rerank):
        best_full = result.get("best_full", None)
        if isinstance(best_full, dict):
            return best_full, "full_rerank"
        return best_search, "search"

    best_full_audit = result.get("best_full_audit", None)
    if (
        isinstance(best_full_audit, dict)
        and _full_validation_probe_mse(best_full_audit) < float(success_mse)
    ):
        return best_full_audit, "solved_full_audit"
    return best_search, "search"


def run_benchmark(
    specs: list[dict],
    *,
    n_iter: int = 1400,
    n_seeds: int = 1,
    max_proposals: int = 48,
    anchors: int = 8,
    preview_topk: int = 12,
    exact_topk: int = 8,
    beam_width: int = 6,
    seed_exact_topk: int = 6,
    seed_beam_width: int = 4,
    seed_scaffold_reserve: int = 8,
    pair_normal_enable: bool = False,
    pair_normal_topk: int = 3,
    pair_normal_max_pairs: int = 1,
    pair_rescue_topk: int = 4,
    pair_rescue_max_pairs: int = 6,
    closure_debug_topk: int = 0,
    emergent_basis_enable: bool | None = None,
    emergent_aux_atoms_enable: bool | None = None,
    seed: int = 42,
    success_mse: float = 1e-6,
    gs_carrier_seed: bool = False,
    retain_candidate_payload: bool = False,
    data_csv: str | None = None,
    data_dir: str | None = None,
    n_fit: int | None = None,
    n_probe: int | None = None,
    search_n_fit: int | None = None,
    search_n_probe: int | None = None,
    final_validate_full: bool = False,
    final_validate_rerank: bool = False,
    data_slice: int = 0,
    y_check: bool = True,
    y_check_abs_tol: float = 1.0e-8,
    y_check_rel_tol: float = 1.0e-8,
    refine_enable: bool | None = None,
    refine_profile: str | None = None,
    refine_mode: str | None = None,
    refine_optimizer: str | None = None,
    refine_lbfgs_steps: int | None = None,
    refine_fit_subset: int | None = None,
    refine_num_restarts: int | None = None,
    refine_max_variants: int | None = None,
    refine_max_params: int | None = None,
    refine_max_trials: int | None = None,
    refine_gate_best_factor: float | None = None,
    refine_linear_combo_enable: bool | None = None,
) -> list[dict]:
    from nestynet_sr.sr_search.factorized_search.oracle_lab import (
        build_oracle_dataset,
        compile_target_expression,
        load_equation_spec,
        run_oracle_equation,
    )
    from nestynet_sr.sr_search.factorized_search.config import (
        FactorizedSearchConfig,
        apply_refine_mode_placement_defaults,
        apply_refine_profile,
        factorized_config_report,
    )

    def _make_hp() -> FactorizedSearchConfig:
        hp = FactorizedSearchConfig()
        hp.search_profile = "paper2_feynman_closure"
        hp.n_iter = n_iter
        hp.n_seeds = n_seeds
        if n_fit is not None:
            hp.n_fit = int(n_fit)
        if n_probe is not None:
            hp.n_probe = int(n_probe)
        hp.closure_search_enable = True
        hp.closure_search_families = ["periodic", "exp", "log", "rational", "power", "quadratic", "affine"]
        hp.closure_search_max_proposals = max_proposals
        hp.closure_search_anchors_per_family = anchors
        hp.closure_search_preview_topk = preview_topk
        hp.closure_search_exact_topk = exact_topk
        hp.closure_search_beam_width = beam_width
        hp.closure_search_seed_exact_topk = seed_exact_topk
        hp.closure_search_seed_beam_width = seed_beam_width
        hp.closure_search_seed_scaffold_reserve = seed_scaffold_reserve
        hp.closure_search_pair_normal_enable = pair_normal_enable
        hp.closure_search_pair_normal_topk = pair_normal_topk
        hp.closure_search_pair_normal_max_pairs = pair_normal_max_pairs
        hp.closure_search_pair_rescue_topk = pair_rescue_topk
        hp.closure_search_pair_rescue_max_pairs = pair_rescue_max_pairs
        hp.closure_search_debug_topk = closure_debug_topk
        if emergent_basis_enable is not None:
            hp.closure_search_emergent_basis_enable = bool(emergent_basis_enable)
        if emergent_aux_atoms_enable is not None:
            hp.closure_search_emergent_aux_atoms_enable = bool(emergent_aux_atoms_enable)
        if refine_enable is not None:
            hp.refine_enable = bool(refine_enable)
        if refine_profile is not None:
            apply_refine_profile(hp, refine_profile)
        elif bool(hp.refine_enable):
            apply_refine_profile(hp, "rare_slate")
        if refine_mode is not None:
            apply_refine_mode_placement_defaults(hp, refine_mode)
        if refine_optimizer is not None:
            hp.refine_optimizer = str(refine_optimizer)
        if refine_lbfgs_steps is not None:
            hp.refine_lbfgs_steps = int(refine_lbfgs_steps)
        if refine_fit_subset is not None:
            hp.refine_fit_subset = int(refine_fit_subset)
        if refine_num_restarts is not None:
            hp.refine_num_restarts = int(refine_num_restarts)
        if refine_max_variants is not None:
            hp.refine_max_variants = int(refine_max_variants)
        if refine_max_params is not None:
            hp.refine_max_params = int(refine_max_params)
        if refine_max_trials is not None:
            hp.refine_max_trials = int(refine_max_trials)
        if refine_gate_best_factor is not None:
            hp.refine_gate_best_factor = float(refine_gate_best_factor)
        if refine_linear_combo_enable is not None:
            hp.refine_linear_combo_enable = bool(refine_linear_combo_enable)
        return hp

    results = []
    for spec_dict in specs:
        eq_id = spec_dict["id"]
        nvars = len(spec_dict["variables"])
        hp = _make_hp()

        # Skip high-variable-count problems (>6 vars) for now; this is the
        # eligibility filter used by the paper's oracle benchmark.
        if nvars > 6:
            print(f"{eq_id}: SKIP ({nvars} vars)")
            results.append({
                "id": eq_id,
                "nvars": nvars,
                "status": "skipped",
                "reason": "too_many_vars",
                "resolved_config": factorized_config_report(hp),
            })
            continue

        # Write temp spec file
        spec_path = Path(f"/tmp/aif_spec_{eq_id}.json")
        with open(spec_path, "w") as f:
            json.dump(spec_dict, f, indent=2)

        try:
            spec = load_equation_spec(str(spec_path))
        except Exception as e:
            print(f"{eq_id}: SPEC_ERROR ({e})")
            results.append({
                "id": eq_id,
                "nvars": nvars,
                "status": "spec_error",
                "error": str(e),
                "resolved_config": factorized_config_report(hp),
            })
            continue

        t0 = time.time()
        try:
            oracle_dataset = None
            full_validation_dataset = None
            resolved_data_csv = None
            if data_csv is not None:
                if len(specs) != 1:
                    raise ValueError("--data_csv can only be used when running one equation")
                resolved_data_csv = Path(data_csv)
            elif data_dir is not None:
                resolved_data_csv = resolve_aif_csv_path(eq_id, data_dir)
            if resolved_data_csv is not None:
                csv_n_fit = int(n_fit if n_fit is not None else 2000)
                csv_n_probe = int(n_probe if n_probe is not None else 2000)
                target_fn = compile_target_expression(spec)
                full_dataset = build_csv_oracle_dataset(
                    spec_dict,
                    csv_path=resolved_data_csv,
                    target_fn=target_fn,
                    n_fit=csv_n_fit,
                    n_probe=csv_n_probe,
                    data_slice=int(data_slice),
                    dtype=torch.float64,
                    y_check=bool(y_check),
                    y_check_abs_tol=float(y_check_abs_tol),
                    y_check_rel_tol=float(y_check_rel_tol),
                )
                oracle_dataset = _subset_oracle_dataset(
                    full_dataset,
                    search_n_fit=search_n_fit,
                    search_n_probe=search_n_probe,
                )
                hp.n_fit = int(oracle_dataset["x_fit"].shape[0])
                hp.n_probe = int(oracle_dataset["x_probe"].shape[0])
                if final_validate_full:
                    full_validation_dataset = full_dataset
                print(
                    f"{eq_id}: using CSV oracle data {resolved_data_csv} "
                    f"(full_n_fit={int(full_dataset['metadata']['n_fit'])}, "
                    f"full_n_probe={int(full_dataset['metadata']['n_probe'])}, "
                    f"search_n_fit={int(hp.n_fit)}, search_n_probe={int(hp.n_probe)}, "
                    f"data_slice={int(full_dataset['metadata']['data_slice'])})"
                )
            elif search_n_fit is not None or search_n_probe is not None or final_validate_full:
                full_n_fit = int(n_fit if n_fit is not None else hp.n_fit)
                full_n_probe = int(n_probe if n_probe is not None else hp.n_probe)
                hp.n_fit = full_n_fit
                hp.n_probe = full_n_probe
                target_fn = compile_target_expression(spec)
                full_dataset = build_oracle_dataset(
                    spec,
                    target_fn,
                    n_fit=full_n_fit,
                    n_probe=full_n_probe,
                    seed=int(seed),
                    dtype=torch.float64,
                )
                full_dataset["metadata"] = {
                    "source": "synthetic",
                    "seed": int(seed),
                    "n_fit": int(full_n_fit),
                    "n_probe": int(full_n_probe),
                }
                oracle_dataset = _subset_oracle_dataset(
                    full_dataset,
                    search_n_fit=search_n_fit,
                    search_n_probe=search_n_probe,
                )
                hp.n_fit = int(oracle_dataset["x_fit"].shape[0])
                hp.n_probe = int(oracle_dataset["x_probe"].shape[0])
                if final_validate_full:
                    full_validation_dataset = full_dataset
                print(
                    f"{eq_id}: using synthetic oracle data "
                    f"(full_n_fit={int(full_n_fit)}, full_n_probe={int(full_n_probe)}, "
                    f"search_n_fit={int(hp.n_fit)}, search_n_probe={int(hp.n_probe)})"
                )
            oracle_run_kwargs: dict[str, Any] = {}
            if bool(gs_carrier_seed):
                oracle_run_kwargs["gs_carrier_seed"] = True
            result = run_oracle_equation(
                spec,
                factorized_search_hp=hp,
                seed=seed,
                dtype=torch.float64,
                oracle_dataset=oracle_dataset,
                **oracle_run_kwargs,
            )
            if full_validation_dataset is not None:
                _attach_full_validation(
                    result,
                    full_validation_dataset,
                    allow_rerank=bool(final_validate_rerank),
                )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{eq_id}: ERROR ({elapsed:.1f}s) {e}")
            results.append({
                "id": eq_id,
                "nvars": nvars,
                "status": "error",
                "error": str(e),
                "wall_s": elapsed,
                "resolved_config": factorized_config_report(hp),
            })
            continue

        elapsed = time.time() - t0
        best_search = result.get("best", {}) or {}
        best, selection_source = _select_benchmark_best(
            result,
            final_validate_rerank=bool(final_validate_rerank),
            success_mse=float(success_mse),
        )
        best_full_validation = best.get("full_validation") if isinstance(best, dict) else None
        if isinstance(best_full_validation, dict):
            mse = float(best_full_validation.get("probe_mse", float("inf")))
        else:
            mse = float(best.get("mse", float("inf")))
        expr = str(best.get("expr", "?"))
        solved = mse < success_mse

        status = "SOLVED" if solved else f"mse={mse:.3e}"
        print(f"{eq_id}: {status:12s} ({elapsed:5.1f}s) {expr[:60]}")

        row = {
            "id": eq_id,
            "nvars": nvars,
            "status": "solved" if solved else "unsolved",
            "mse": mse,
            "search_mse": best_search.get("mse", None),
            "expr": expr,
            "wall_s": elapsed,
            "target": spec_dict["target"]["expr"],
        }
        if isinstance(best_full_validation, dict):
            row["full_validation"] = best_full_validation
        if isinstance(best, dict) and isinstance(best.get("score_ladder"), dict):
            row["score_ladder"] = best.get("score_ladder")
        if isinstance(best, dict) and best.get("acceptance_basis") is not None:
            row["acceptance_basis"] = best.get("acceptance_basis")
        if isinstance(best, dict) and best.get("final_acceptance_basis") is not None:
            row["final_acceptance_basis"] = best.get("final_acceptance_basis")
        if isinstance(best, dict) and best.get("final_validated_mse") is not None:
            row["final_validated_mse"] = best.get("final_validated_mse")
        if bool(retain_candidate_payload) and isinstance(best, dict):
            row["candidate_payload"] = {
                key: best.get(key)
                for key in (
                    "expr_ast",
                    "mapping",
                    "mapping_kind",
                    "raw_mse",
                    "embedding_roundtrip",
                )
                if key in best
            }
        if isinstance(result.get("closure_search_summary"), dict):
            row["closure_search_summary"] = result.get("closure_search_summary")
        if isinstance(result.get("closure_search_debug"), dict):
            row["closure_search_debug"] = result.get("closure_search_debug")
        if isinstance(result.get("dataset"), dict):
            row["dataset"] = result.get("dataset")
        if isinstance(result.get("resolved_config"), dict):
            row["resolved_config"] = result.get("resolved_config")
            row["search_profile"] = str(result["resolved_config"].get("search_profile", ""))
        if isinstance(result.get("archive_coherence"), dict):
            row["archive_coherence"] = result.get("archive_coherence")
        if isinstance(result.get("embedding_roundtrip_summary"), dict):
            row["embedding_roundtrip_summary"] = result.get("embedding_roundtrip_summary")
        if bool(gs_carrier_seed):
            gs_diagnostics = list(result.get("gs_carrier_seed_diagnostics", []) or [])
            row["gs_carrier_seed"] = True
            row["gs_carrier_seed_count"] = int(len(gs_diagnostics))
            row["gs_carrier_seed_diagnostics"] = gs_diagnostics
        if isinstance(result.get("full_validation"), dict):
            full_validation_summary = dict(result.get("full_validation") or {})
            full_validation_summary["selection_source"] = selection_source
            if selection_source == "solved_full_audit":
                full_validation_summary["solved_audit_promoted"] = True
            row["full_validation_summary"] = full_validation_summary
        results.append(row)

    return results


def main():
    from nestynet_sr.sr_search.factorized_search.config import REFINE_OPTIMIZER_NAMES, REFINE_PROFILE_NAMES

    parser = argparse.ArgumentParser(description="AI Feynman closure machine benchmark")
    parser.add_argument("--equations", default="data/equations.txt", help="Path to equations.txt")
    parser.add_argument("--only", default=None, help="Comma-separated equation IDs (e.g., 037,090)")
    parser.add_argument("--max_nvars", type=int, default=6, help="Skip problems with more variables")
    parser.add_argument("--n_iter", type=int, default=1400)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--max_proposals", type=int, default=48)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--preview_topk", type=int, default=12)
    parser.add_argument("--exact_topk", type=int, default=8)
    parser.add_argument("--beam_width", type=int, default=6)
    parser.add_argument("--seed_exact_topk", type=int, default=6)
    parser.add_argument("--seed_beam_width", type=int, default=4)
    parser.add_argument("--seed_scaffold_reserve", type=int, default=8)
    parser.add_argument("--pair_normal_enable", action="store_true")
    parser.add_argument("--pair_normal_topk", type=int, default=3)
    parser.add_argument("--pair_normal_max_pairs", type=int, default=1)
    parser.add_argument("--pair_rescue_topk", type=int, default=4)
    parser.add_argument("--pair_rescue_max_pairs", type=int, default=6)
    parser.add_argument("--closure_debug_topk", type=int, default=0)
    emergent_basis_g = parser.add_mutually_exclusive_group()
    emergent_basis_g.add_argument(
        "--emergent-basis",
        "--emergent_basis",
        dest="emergent_basis_enable",
        action="store_true",
        help="Enable legacy emergent basis-row promotion ablation.",
    )
    emergent_basis_g.add_argument(
        "--no-emergent-basis",
        "--no_emergent_basis",
        dest="emergent_basis_enable",
        action="store_false",
    )
    emergent_aux_g = parser.add_mutually_exclusive_group()
    emergent_aux_g.add_argument(
        "--emergent-aux-atoms",
        "--emergent_aux_atoms",
        dest="emergent_aux_atoms_enable",
        action="store_true",
        help="Enable emergent auxiliary atoms as next-round FSS SeedBlocks.",
    )
    emergent_aux_g.add_argument(
        "--no-emergent-aux-atoms",
        "--no_emergent_aux_atoms",
        dest="emergent_aux_atoms_enable",
        action="store_false",
    )
    parser.set_defaults(emergent_basis_enable=None, emergent_aux_atoms_enable=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success_mse", type=float, default=1e-6)
    gs_carrier_g = parser.add_mutually_exclusive_group()
    gs_carrier_g.add_argument(
        "--gs-carrier-seed",
        "--gs_carrier_seed",
        dest="gs_carrier_seed",
        action="store_true",
        help=(
            "Opt in to the existing generalized-symmetry carrier-seed bridge. "
            "The default is off, preserving the historical FSS-only benchmark."
        ),
    )
    gs_carrier_g.add_argument(
        "--no-gs-carrier-seed",
        "--no_gs_carrier_seed",
        dest="gs_carrier_seed",
        action="store_false",
    )
    parser.set_defaults(gs_carrier_seed=False)
    parser.add_argument(
        "--retain-candidate-payload",
        "--retain_candidate_payload",
        dest="retain_candidate_payload",
        action="store_true",
        help=(
            "Retain the selected carrier AST and fitted outer mapping in each "
            "benchmark row for reporting or independent validation. Default off."
        ),
    )
    parser.add_argument("--data_csv", default=None, help="External CSV for the selected equation")
    parser.add_argument("--data_dir", default=None, help="Directory containing pbNNN_*_data.csv files")
    parser.add_argument("--n_fit", type=int, default=None, help="External/synthetic fit-point count")
    parser.add_argument("--n_probe", type=int, default=None, help="External/synthetic probe-point count")
    parser.add_argument(
        "--search-n-fit",
        "--search_n_fit",
        dest="search_n_fit",
        type=int,
        default=None,
        help="Fit rows used during search; defaults to the full n_fit split",
    )
    parser.add_argument(
        "--search-n-probe",
        "--search_n_probe",
        dest="search_n_probe",
        type=int,
        default=None,
        help="Probe rows used during search; defaults to the full n_probe split",
    )
    final_validate_g = parser.add_mutually_exclusive_group()
    final_validate_g.add_argument(
        "--final-validate-full",
        "--final_validate_full",
        dest="final_validate_full",
        action="store_true",
        help="Evaluate returned candidates on the full n_fit/n_probe split after search",
    )
    final_validate_g.add_argument(
        "--no-final-validate-full",
        "--no_final_validate_full",
        dest="final_validate_full",
        action="store_false",
    )
    parser.set_defaults(final_validate_full=False)
    final_rerank_g = parser.add_mutually_exclusive_group()
    final_rerank_g.add_argument(
        "--final-validate-rerank",
        "--final_validate_rerank",
        dest="final_validate_rerank",
        action="store_true",
        help="Allow full-validation probe MSE to rerank returned candidates after search",
    )
    final_rerank_g.add_argument(
        "--no-final-validate-rerank",
        "--no_final_validate_rerank",
        dest="final_validate_rerank",
        action="store_false",
    )
    parser.set_defaults(final_validate_rerank=_env_bool("FINAL_VALIDATE_RERANK", False))
    parser.add_argument("--data_slice", type=int, default=0, help="Disjoint external data block index")
    parser.add_argument("--y_check_abs_tol", type=float, default=1.0e-8)
    parser.add_argument("--y_check_rel_tol", type=float, default=1.0e-8)
    parser.add_argument("--no_y_check", action="store_true", help="Disable CSV y-vs-oracle consistency check")
    parser.add_argument("--output", default=None, help="Output JSON path")
    refine_g = parser.add_mutually_exclusive_group()
    refine_g.add_argument("--refine-skeleton", "--plus", dest="refine_enable", action="store_true")
    refine_g.add_argument("--no-refine-skeleton", "--no-plus", dest="refine_enable", action="store_false")
    parser.set_defaults(refine_enable=None)
    parser.add_argument(
        "--refine-profile",
        "--refine_profile",
        dest="refine_profile",
        default=None,
        metavar="PROFILE",
        help=f"Named continuous skeleton refinement profile ({', '.join(REFINE_PROFILE_NAMES)}; aliases accepted)",
    )
    parser.add_argument(
        "--refine-mode",
        "--refine_mode",
        dest="refine_mode",
        choices=["off", "inline", "slate", "final_polish"],
        default=None,
        help="Continuous skeleton refinement placement mode",
    )
    parser.add_argument(
        "--refine-optimizer",
        "--refine_optimizer",
        dest="refine_optimizer",
        choices=list(REFINE_OPTIMIZER_NAMES),
        default=None,
    )
    parser.add_argument("--refine-lbfgs-steps", "--refine_lbfgs_steps", dest="refine_lbfgs_steps", type=int, default=None)
    parser.add_argument("--refine-fit-subset", "--refine_fit_subset", dest="refine_fit_subset", type=int, default=None)
    parser.add_argument("--refine-num-restarts", "--refine_num_restarts", dest="refine_num_restarts", type=int, default=None)
    parser.add_argument("--refine-max-variants", "--refine_max_variants", dest="refine_max_variants", type=int, default=None)
    parser.add_argument("--refine-max-params", "--refine_max_params", dest="refine_max_params", type=int, default=None)
    parser.add_argument("--refine-max-trials", "--refine_max_trials", dest="refine_max_trials", type=int, default=None)
    parser.add_argument(
        "--refine-gate-best-factor",
        "--refine_gate_best_factor",
        dest="refine_gate_best_factor",
        type=float,
        default=None,
    )
    linear_g = parser.add_mutually_exclusive_group()
    linear_g.add_argument("--refine-linear-combo", "--refine_linear_combo", dest="refine_linear_combo_enable", action="store_true")
    linear_g.add_argument("--no-refine-linear-combo", "--no_refine_linear_combo", dest="refine_linear_combo_enable", action="store_false")
    parser.set_defaults(refine_linear_combo_enable=None)
    args = parser.parse_args()

    specs = parse_equations_txt(args.equations)
    print(f"Parsed {len(specs)} equations from {args.equations}")

    if args.only:
        only_ids = {f"feynman_{eid.strip().zfill(3)}" for eid in args.only.split(",")}
        specs = [s for s in specs if s["id"] in only_ids]
        print(f"Filtered to {len(specs)}: {[s['id'] for s in specs]}")

    if args.max_nvars:
        before = len(specs)
        specs = [s for s in specs if len(s["variables"]) <= args.max_nvars]
        if len(specs) < before:
            print(f"Filtered to {len(specs)} specs (max {args.max_nvars} vars)")

    results = run_benchmark(
        specs,
        n_iter=args.n_iter,
        n_seeds=args.n_seeds,
        max_proposals=args.max_proposals,
        anchors=args.anchors,
        preview_topk=args.preview_topk,
        exact_topk=args.exact_topk,
        beam_width=args.beam_width,
        seed_exact_topk=args.seed_exact_topk,
        seed_beam_width=args.seed_beam_width,
        seed_scaffold_reserve=args.seed_scaffold_reserve,
        pair_normal_enable=args.pair_normal_enable,
        pair_normal_topk=args.pair_normal_topk,
        pair_normal_max_pairs=args.pair_normal_max_pairs,
        pair_rescue_topk=args.pair_rescue_topk,
        pair_rescue_max_pairs=args.pair_rescue_max_pairs,
        closure_debug_topk=args.closure_debug_topk,
        emergent_basis_enable=args.emergent_basis_enable,
        emergent_aux_atoms_enable=args.emergent_aux_atoms_enable,
        seed=args.seed,
        success_mse=args.success_mse,
        gs_carrier_seed=bool(args.gs_carrier_seed),
        retain_candidate_payload=bool(args.retain_candidate_payload),
        data_csv=args.data_csv,
        data_dir=args.data_dir,
        n_fit=args.n_fit,
        n_probe=args.n_probe,
        search_n_fit=args.search_n_fit,
        search_n_probe=args.search_n_probe,
        final_validate_full=bool(args.final_validate_full),
        final_validate_rerank=bool(args.final_validate_rerank),
        data_slice=args.data_slice,
        y_check=not bool(args.no_y_check),
        y_check_abs_tol=args.y_check_abs_tol,
        y_check_rel_tol=args.y_check_rel_tol,
        refine_enable=args.refine_enable,
        refine_profile=args.refine_profile,
        refine_mode=args.refine_mode,
        refine_optimizer=args.refine_optimizer,
        refine_lbfgs_steps=args.refine_lbfgs_steps,
        refine_fit_subset=args.refine_fit_subset,
        refine_num_restarts=args.refine_num_restarts,
        refine_max_variants=args.refine_max_variants,
        refine_max_params=args.refine_max_params,
        refine_max_trials=args.refine_max_trials,
        refine_gate_best_factor=args.refine_gate_best_factor,
        refine_linear_combo_enable=args.refine_linear_combo_enable,
    )

    solved = sum(1 for r in results if r["status"] == "solved")
    total = sum(1 for r in results if r["status"] in ("solved", "unsolved"))
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] in ("error", "spec_error"))

    print(f"\n{'='*60}")
    print(f"Results: {solved}/{total} solved, {skipped} skipped, {errors} errors")
    print(f"{'='*60}")

    if args.output:
        resolved_configs = [
            r.get("resolved_config")
            for r in results
            if isinstance(r, dict) and isinstance(r.get("resolved_config"), dict)
        ]
        unique_config_keys = []
        unique_configs = []
        for cfg in resolved_configs:
            key = json.dumps(cfg, sort_keys=True)
            if key in unique_config_keys:
                continue
            unique_config_keys.append(key)
            unique_configs.append(cfg)
        summary = {"solved": solved, "total": total, "skipped": skipped, "errors": errors}
        payload = {"summary": summary, "results": results}
        if bool(args.gs_carrier_seed):
            payload["gs_carrier_seed"] = True
        if len(unique_configs) == 1:
            payload["resolved_config"] = unique_configs[0]
        elif unique_configs:
            payload["resolved_config"] = {
                "common": False,
                "count": int(len(unique_configs)),
                "configs": unique_configs,
            }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
