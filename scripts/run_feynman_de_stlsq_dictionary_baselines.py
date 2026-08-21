#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Fast STLSQ dictionary baselines for compositional Feynman-DE cases.

This script isolates the representation question for paper 3.  It reads the
cached compositional trajectory CSVs, uses the known generated RHS as the
derivative target, fits several STLSQ dictionaries, and validates the selected
implicit law by rollout on held-out trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
FEYNMAN_DE_DIR = REPO_ROOT / "examples" / "feynman_de"
if str(FEYNMAN_DE_DIR) in sys.path:
    sys.path.remove(str(FEYNMAN_DE_DIR))
sys.path.insert(0, str(FEYNMAN_DE_DIR))
_problem_defs_mod = sys.modules.get("problem_defs")
if _problem_defs_mod is not None:
    _problem_defs_path = Path(getattr(_problem_defs_mod, "__file__", "") or "").resolve()
    if _problem_defs_path.parent != FEYNMAN_DE_DIR:
        del sys.modules["problem_defs"]

from nestynet_sr.sr_core.numerics import ridge_lstsq, stlsq  # noqa: E402
from problem_defs import default_param_values, load_problems, resolve_rhs  # noqa: E402
from run_benchmark import load_existing_runs, validate_by_simulation  # noqa: E402


BENCHMARK_FILE = REPO_ROOT / "data" / "feynman_de_compositional.txt"
DATA_DIR = REPO_ROOT / "data" / "feynman_de_compositional"

REQUIRED_ATOMS: dict[str, str] = {
    "900": "u/(1+x0)",
    "901": "u/(1+x0)^2",
    "902": "u*log(1+x0)",
    "903": "exp(u)",
}

STATUS_RANK = {"PASS": 0, "PARTIAL": 1, "FAIL": 2, "ERROR": 3}


@dataclass(frozen=True)
class LibraryTerm:
    name: str
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _as_array(value: Any, like: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        return np.full_like(like, float(arr), dtype=np.float64)
    return arr.astype(np.float64, copy=False)


def _term(name: str, fn: Callable[[np.ndarray, np.ndarray], Any]) -> LibraryTerm:
    def _wrapped(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return _as_array(fn(x, u), x)

    return LibraryTerm(name=name, fn=_wrapped)


def _safe_exp(z: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(z, -80.0, 80.0))


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.asarray(num, dtype=np.float64) / np.asarray(denom, dtype=np.float64)


def _base_standard_terms() -> list[LibraryTerm]:
    return [
        _term("1", lambda x, u: np.ones_like(x)),
        _term("x0", lambda x, u: x),
        _term("x0^2", lambda x, u: x * x),
        _term("u", lambda x, u: u),
        _term("u^2", lambda x, u: u * u),
        _term("u^3", lambda x, u: u * u * u),
        _term("x0*u", lambda x, u: x * u),
        _term("x0*u^2", lambda x, u: x * u * u),
        _term("x0^2*u", lambda x, u: x * x * u),
        _term("u/x0", lambda x, u: _safe_divide(u, x)),
        _term("u/x0^2", lambda x, u: _safe_divide(u, x * x)),
    ]


def _expanded_unary_terms() -> list[LibraryTerm]:
    return [
        _term("(1+x0)^-1", lambda x, u: 1.0 / (1.0 + x)),
        _term("(1+x0)^-2", lambda x, u: 1.0 / ((1.0 + x) * (1.0 + x))),
        _term("log(1+x0)", lambda x, u: np.log1p(x)),
        _term("exp(u)", lambda x, u: _safe_exp(u)),
    ]


def _expanded_closure_terms() -> list[LibraryTerm]:
    return [
        _term("u/(1+x0)", lambda x, u: u / (1.0 + x)),
        _term("u/(1+x0)^2", lambda x, u: u / ((1.0 + x) * (1.0 + x))),
        _term("u*log(1+x0)", lambda x, u: u * np.log1p(x)),
    ]


def build_library_terms(preset: str, pid: str) -> list[LibraryTerm]:
    preset_l = str(preset).strip().lower()
    if preset_l == "standard":
        terms = _base_standard_terms()
    elif preset_l == "expanded_unary":
        terms = _base_standard_terms() + _expanded_unary_terms()
    elif preset_l == "expanded_closure":
        terms = _base_standard_terms() + _expanded_unary_terms() + _expanded_closure_terms()
    elif preset_l == "oracle":
        terms = [_term("1", lambda x, u: np.ones_like(x))]
        if str(pid) == "900":
            terms += [
                _term("u", lambda x, u: u),
                _term("u/(1+x0)", lambda x, u: u / (1.0 + x)),
            ]
        elif str(pid) == "901":
            terms += [
                _term("u", lambda x, u: u),
                _term("u/(1+x0)^2", lambda x, u: u / ((1.0 + x) * (1.0 + x))),
            ]
        elif str(pid) == "902":
            terms += [
                _term("u", lambda x, u: u),
                _term("u*log(1+x0)", lambda x, u: u * np.log1p(x)),
            ]
        elif str(pid) == "903":
            terms += [_term("exp(u)", lambda x, u: _safe_exp(u))]
        else:
            raise ValueError(f"No oracle dictionary defined for de{pid}")
    else:
        raise ValueError(f"Unknown library preset: {preset!r}")

    unique: dict[str, LibraryTerm] = {}
    for term in terms:
        unique[term.name] = term
    return list(unique.values())


def _load_xy(run: Any) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(str(run.csv_path), delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid CSV shape for {run.csv_path}")
    u = np.asarray(data[:, 0], dtype=np.float64)
    x = np.asarray(data[:, 1], dtype=np.float64)
    return x, u


def _rhs_values(problem: Any, x: np.ndarray, u: np.ndarray, params: dict[str, float]) -> np.ndarray:
    rhs_fn, _source = resolve_rhs(problem)
    vals = np.empty_like(x, dtype=np.float64)
    for i, (xi, ui) in enumerate(zip(x, u)):
        vals[i] = float(rhs_fn(float(xi), [float(ui)], params)[0])
    return vals


def build_design(
    problem: Any,
    runs: Sequence[Any],
    terms: Sequence[LibraryTerm],
    *,
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    Phi_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    dropped = 0
    for run in runs:
        x, u = _load_xy(run)
        du = _rhs_values(problem, x, u, params)
        cols = [term.fn(x, u).reshape(-1) for term in terms]
        Phi_full = np.stack(cols, axis=1)
        y_full = -du.reshape(-1)
        mask = np.isfinite(y_full) & np.isfinite(Phi_full).all(axis=1)
        dropped += int(mask.size - int(mask.sum()))
        if int(mask.sum()) <= 0:
            continue
        Phi_parts.append(Phi_full[mask])
        y_parts.append(y_full[mask])
    if not Phi_parts:
        raise RuntimeError(f"No finite design rows for de{problem.id}")
    return np.concatenate(Phi_parts, axis=0), np.concatenate(y_parts, axis=0), int(dropped)


def fit_stlsq(
    Phi: np.ndarray,
    y: np.ndarray,
    *,
    lam: float,
    ridge: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    Phi_t = torch.as_tensor(Phi, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    coeffs_t, keep_t = stlsq(Phi_t, y_t, ridge=float(ridge), lam=float(lam), max_iter=int(max_iter))
    keep = keep_t.detach().cpu().numpy().astype(bool)
    coeffs = coeffs_t.detach().cpu().numpy().astype(np.float64)
    if int(keep.sum()) > 0:
        refit = ridge_lstsq(Phi_t[:, keep_t], y_t, ridge=0.0).detach().cpu().numpy().astype(np.float64)
        coeffs = np.zeros_like(coeffs)
        coeffs[keep] = refit
    return coeffs, keep


def _condition_number(Phi: np.ndarray, keep: np.ndarray) -> float | None:
    if int(keep.sum()) <= 0:
        return None
    try:
        return float(np.linalg.cond(Phi[:, keep]))
    except Exception:
        return None


def _residual_rms(Phi: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        resid = Phi @ coeffs - y
    resid = resid[np.isfinite(resid)]
    if int(resid.size) <= 0:
        return float("inf")
    return float(np.sqrt(np.mean(resid * resid)))


def _format_term(coeff: float, name: str) -> str:
    if name == "1":
        return f"{coeff:.6g}"
    return f"({coeff:.6g}*{name})"


def canonical_equation(terms: Sequence[LibraryTerm], coeffs: np.ndarray, keep: np.ndarray) -> str:
    pieces = [
        _format_term(float(coeffs[i]), terms[i].name)
        for i in range(len(terms))
        if bool(keep[i]) and abs(float(coeffs[i])) > 0.0
    ]
    rhs = " + ".join(pieces) if pieces else "0"
    return f"u_x0 + ({rhs}) = 0"


def make_rhs(terms: Sequence[LibraryTerm], coeffs: np.ndarray, keep: np.ndarray) -> Callable[[float, Sequence[float]], list[float]]:
    selected = [(terms[i], float(coeffs[i])) for i in range(len(terms)) if bool(keep[i])]

    def _rhs(x0: float, state: Sequence[float]) -> list[float]:
        x_arr = np.asarray([float(x0)], dtype=np.float64)
        u_arr = np.asarray([float(state[0])], dtype=np.float64)
        total = 0.0
        for term, coeff in selected:
            total += coeff * float(term.fn(x_arr, u_arr)[0])
        if not math.isfinite(total):
            raise FloatingPointError("non-finite STLSQ RHS")
        return [-float(total)]

    return _rhs


def _mean_max_nrmse(traj_scores: Sequence[dict[str, Any]]) -> tuple[float, float]:
    vals = [float(row.get("nrmse", float("inf"))) for row in traj_scores]
    if not vals:
        return float("inf"), float("inf")
    return float(sum(vals) / len(vals)), float(max(vals))


def _choice_key(row: dict[str, Any]) -> tuple[int, float, int, float]:
    return (
        int(STATUS_RANK.get(str(row.get("status", "ERROR")), 9)),
        float(row.get("max_nrmse", float("inf"))),
        int(row.get("selected_terms", 10**9)),
        float(row.get("fit_rms", float("inf"))),
    )


def evaluate_preset(
    problem: Any,
    fit_runs: Sequence[Any],
    probe_runs: Sequence[Any],
    *,
    preset: str,
    lambdas: Sequence[float],
    ridge: float,
    max_iter: int,
    pass_nrmse: float,
    partial_nrmse: float,
    traj_time_budget_s: float | None,
) -> dict[str, Any]:
    params = default_param_values(problem)
    terms = build_library_terms(preset, str(problem.id))
    Phi_fit, y_fit, dropped_fit = build_design(problem, fit_runs, terms, params=params)
    Phi_probe, y_probe, dropped_probe = build_design(problem, probe_runs, terms, params=params)
    required_atom = REQUIRED_ATOMS.get(str(problem.id), "")

    candidates: list[dict[str, Any]] = []
    for lam in lambdas:
        try:
            coeffs, keep = fit_stlsq(Phi_fit, y_fit, lam=float(lam), ridge=float(ridge), max_iter=int(max_iter))
            rhs_fn = make_rhs(terms, coeffs, keep)
            status, message, traj_scores = validate_by_simulation(
                probe_runs,
                rhs_fn=rhs_fn,
                order=1,
                pass_nrmse=float(pass_nrmse),
                partial_nrmse=float(partial_nrmse),
                traj_time_budget_s=traj_time_budget_s,
            )
            mean_nrmse, max_nrmse = _mean_max_nrmse(traj_scores)
            selected_names = [terms[i].name for i in range(len(terms)) if bool(keep[i])]
            cand = {
                "lambda": float(lam),
                "status": str(status),
                "message": str(message),
                "mean_nrmse": float(mean_nrmse),
                "max_nrmse": float(max_nrmse),
                "fit_rms": _residual_rms(Phi_fit, y_fit, coeffs),
                "probe_target_rms": _residual_rms(Phi_probe, y_probe, coeffs),
                "selected_terms": int(keep.sum()),
                "selected_term_names": selected_names,
                "selected_coefficients": [float(coeffs[i]) for i in range(len(terms)) if bool(keep[i])],
                "canonical_equation": canonical_equation(terms, coeffs, keep),
                "condition_number": _condition_number(Phi_fit, keep),
                "required_atom_selected": bool(required_atom and required_atom in selected_names),
                "traj_scores": traj_scores,
            }
        except Exception as exc:
            cand = {
                "lambda": float(lam),
                "status": "ERROR",
                "message": str(exc),
                "mean_nrmse": float("inf"),
                "max_nrmse": float("inf"),
                "fit_rms": float("inf"),
                "probe_target_rms": float("inf"),
                "selected_terms": 0,
                "selected_term_names": [],
                "selected_coefficients": [],
                "canonical_equation": "",
                "condition_number": None,
                "required_atom_selected": False,
                "traj_scores": [],
            }
        candidates.append(cand)

    best = min(candidates, key=_choice_key)
    return {
        "id": str(problem.id),
        "description": str(problem.description),
        "preset": str(preset),
        "required_atom": str(required_atom),
        "required_atom_present": bool(required_atom and any(term.name == required_atom for term in terms)),
        "library_size": int(len(terms)),
        "library_terms": [term.name for term in terms],
        "dropped_fit_rows": int(dropped_fit),
        "dropped_probe_rows": int(dropped_probe),
        "selected": best,
        "candidates": candidates,
    }


def _parse_csv_list(value: str, *, cast: Callable[[str], Any] = str) -> list[Any]:
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def _write_summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "preset",
        "status",
        "message",
        "mean_nrmse",
        "max_nrmse",
        "lambda",
        "library_size",
        "selected_terms",
        "required_atom",
        "required_atom_present",
        "required_atom_selected",
        "condition_number",
        "canonical_equation",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sel = dict(row.get("selected", {}) or {})
            writer.writerow(
                {
                    "id": row.get("id"),
                    "preset": row.get("preset"),
                    "status": sel.get("status"),
                    "message": sel.get("message"),
                    "mean_nrmse": sel.get("mean_nrmse"),
                    "max_nrmse": sel.get("max_nrmse"),
                    "lambda": sel.get("lambda"),
                    "library_size": row.get("library_size"),
                    "selected_terms": sel.get("selected_terms"),
                    "required_atom": row.get("required_atom"),
                    "required_atom_present": row.get("required_atom_present"),
                    "required_atom_selected": sel.get("required_atom_selected"),
                    "condition_number": sel.get("condition_number"),
                    "canonical_equation": sel.get("canonical_equation"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast STLSQ dictionary baselines for Feynman-DE compositional cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ids", default="900,901,902,903", help="Comma-separated problem IDs.")
    parser.add_argument(
        "--presets",
        default="standard,expanded_unary,expanded_closure,oracle",
        help="Comma-separated presets: standard, expanded_unary, expanded_closure, oracle.",
    )
    parser.add_argument("--benchmark_file", default=str(BENCHMARK_FILE))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "results" / "feynman_de_stlsq_dictionary_baselines"))
    parser.add_argument("--n_traj", type=int, default=6)
    parser.add_argument("--holdout_last_k", type=int, default=2)
    parser.add_argument("--lambdas", default="1e-8,1e-6,1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--ridge", type=float, default=1.0e-10)
    parser.add_argument("--stlsq_max_iter", type=int, default=10)
    parser.add_argument("--pass_nrmse", type=float, default=1.0e-2)
    parser.add_argument("--partial_nrmse", type=float, default=5.0e-2)
    parser.add_argument("--sim_validate_traj_time_budget_s", type=float, default=20.0)
    args = parser.parse_args()

    ids = [str(pid).replace("de", "") for pid in _parse_csv_list(args.ids)]
    presets = [str(p) for p in _parse_csv_list(args.presets)]
    lambdas = [float(v) for v in _parse_csv_list(args.lambdas, cast=float)]
    if not lambdas:
        raise ValueError("At least one STLSQ lambda is required")

    problems = load_problems(args.benchmark_file)
    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for pid in ids:
        if pid not in problems:
            raise KeyError(f"Unknown problem ID: {pid}")
        problem = problems[pid]
        if int(problem.order) != 1:
            raise ValueError(f"Only first-order cases are supported by this baseline script, got de{pid}")
        runs, _source = load_existing_runs(problem, data_dir, n_traj=int(args.n_traj))
        holdout = int(args.holdout_last_k)
        if holdout <= 0 or holdout >= len(runs):
            raise ValueError(f"holdout_last_k must be in [1, n_traj-1], got {holdout}")
        fit_runs = runs[:-holdout]
        probe_runs = runs[-holdout:]
        for preset in presets:
            row = evaluate_preset(
                problem,
                fit_runs,
                probe_runs,
                preset=preset,
                lambdas=lambdas,
                ridge=float(args.ridge),
                max_iter=int(args.stlsq_max_iter),
                pass_nrmse=float(args.pass_nrmse),
                partial_nrmse=float(args.partial_nrmse),
                traj_time_budget_s=(
                    None
                    if float(args.sim_validate_traj_time_budget_s) <= 0.0
                    else float(args.sim_validate_traj_time_budget_s)
                ),
            )
            rows.append(row)
            sel = row["selected"]
            print(
                "de{} {:>16s} {:>7s} mean={:.3g} max={:.3g} atom_present={} atom_selected={}".format(
                    pid,
                    str(preset),
                    str(sel.get("status", "?")),
                    float(sel.get("mean_nrmse", float("inf"))),
                    float(sel.get("max_nrmse", float("inf"))),
                    bool(row.get("required_atom_present")),
                    bool(sel.get("required_atom_selected")),
                )
            )

    payload = {
        "ids": ids,
        "presets": presets,
        "lambdas": lambdas,
        "target_source": "known_generated_rhs",
        "n_traj": int(args.n_traj),
        "holdout_last_k": int(args.holdout_last_k),
        "rows": rows,
    }
    json_path = results_dir / "stlsq_dictionary_baselines_summary.json"
    csv_path = results_dir / "stlsq_dictionary_baselines_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    _write_summary_csv(rows, csv_path)
    print(f"JSON summary: {json_path}")
    print(f"CSV summary:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
