#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Feynman DE Benchmark Runner.

Supports five discovery engines:
- ``sparse``: first-line ``nestynet_sr/run_de.py`` pipeline only
- ``hybrid``: first-line sparse/STLSQ; if rollout validation fails, run clean
  STLSQ-free ``factorized_de`` fallback
- ``factorized_search_only``: DE-facing factorized symbolic search on its own, without first-line sparse search
- ``factorized_de``: STLSQ-free direct DE FSS with optional typed and whole-RHS fallback lanes
- ``factorized_search_oracle``: legacy multi-trajectory oracle DE runner

It can generate multiple trajectories per problem (different ICs), and for
all engines evaluates discovered equations by rollout on held-out probe
trajectories when trajectory holdout is enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from nestynet_sr.sr_core.problem_dims import canonical_constant_payload, canonical_scalar_dims_payload
from nestynet_sr.sr_de.de_validation import (
    factorized_search_report_to_rhs_callable,
    library_candidate_to_rhs_callable,
    validate_by_simulation,
)
from nestynet_sr.sr_de.de_committee import (
    run_de_committee_audit,
    selected_engine_from_decision,
    selected_summary_from_decision,
    tied_candidate_summaries_from_decision,
)
from nestynet_sr.sr_de.de_csr import refine_factorized_search_candidate_from_runs
from nestynet_sr.sr_de.proposals import canonicalize_de_equation, merge_proposal_slates

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEYNMAN_DE_DIR = Path(__file__).resolve().parent
_ORIGINAL_SUBPROCESS_RUN = subprocess.run
if str(FEYNMAN_DE_DIR) in sys.path:
    sys.path.remove(str(FEYNMAN_DE_DIR))
sys.path.insert(0, str(FEYNMAN_DE_DIR))
_problem_defs_mod = sys.modules.get("problem_defs")
if _problem_defs_mod is not None:
    _problem_defs_path = Path(getattr(_problem_defs_mod, "__file__", "") or "").resolve()
    if _problem_defs_path.parent != FEYNMAN_DE_DIR:
        del sys.modules["problem_defs"]

from problem_defs import (
    IC_DEFAULTS,
    IC_OVERRIDE,
    GroundTruth,
    ProblemDef,
    default_param_values,
    default_t_max,
    get_canonical_problem_dims,
    load_problems,
    resolve_ground_truth,
    resolve_rhs,
)
from oracle_de_spec_writer import write_oracle_de_spec

RUN_DE_SCRIPT = REPO_ROOT / "nestynet_sr" / "run_de.py"
BENCHMARK_FILE = REPO_ROOT / "data" / "feynman_de_benchmark.txt"
ORACLE_DE_MOD = "nestynet_sr.sr_search.factorized_search.oracle_lab_de"
GEN_SOLVER_TRIALS: tuple[tuple[str, float, float], ...] = (
    ("RK45", 1.0e-8, 1.0e-10),
    ("Radau", 1.0e-8, 1.0e-10),
    ("BDF", 1.0e-8, 1.0e-10),
)
GEN_MAX_IC_RETRIES = 20
GEN_SUPPORT_MAX_BACKOFFS = 12
GEN_SUPPORT_BACKOFF_FACTOR = 0.5
GEN_SUPPORT_MAX_STATE_GROWTH = 1.0e3
GEN_SUPPORT_MAX_ABS_STATE = 1.0e5
GEN_SUPPORT_MIN_SPAN_FRACTION = 1.0 / 4096.0
GEN_X_MIN_PILOT_CANDIDATES: tuple[float, ...] = (0.0, 1.0e-3, 1.0e-2, 5.0e-2, 0.1, 0.25, 0.5, 1.0)
GEN_X_MIN_PILOT_TRAJ = 3
GEN_X_MIN_PILOT_POINTS = 128
GEN_X_MIN_PILOT_MIN_SPAN_FRACTION = 0.20
GEN_X_MIN_PILOT_GOOD_STATE_GROWTH = 5.0


@dataclass(frozen=True)
class TrajRun:
    traj_id: str
    csv_path: Path
    meta_path: Path
    x_min: float
    x_max: float
    u0: float
    v0: float


def _problem_seed(base_seed: int, pid: str) -> int:
    try:
        return int(base_seed) + int(pid) * 1_000_003
    except Exception:
        acc = 0
        for c in str(pid):
            acc = (acc * 131 + ord(c)) % 2_147_483_647
        return int(base_seed) + acc


def _declares_singular_origin(problem: ProblemDef) -> bool:
    """Declared class metadata gate for the inverse-coordinate atoms.

    Keyed on the benchmark file's ``flags`` column (``singular_origin``), never
    on the target equation, so the library stays answer-blind.
    """
    return "singular_origin" in problem.flags


def _default_x_min(problem: ProblemDef) -> float:
    return 0.0


def _sample_ic(
    problem: ProblemDef,
    param_values: dict[str, float],
    rng: np.random.Generator,
    *,
    x_min: float,
    ic_index: int,
) -> tuple[float, float]:
    if problem.id in IC_OVERRIDE:
        base = IC_OVERRIDE[problem.id](x_min if x_min > 0.0 else 1.0e-3, param_values)
        u0 = float(base[0])
        v0 = float(base[1]) if len(base) > 1 else 0.0
        if int(ic_index) > 0:
            u0 += float(rng.uniform(-0.03, 0.03))
            v0 += float(rng.uniform(-0.03, 0.03))
        return u0, v0

    ic = IC_DEFAULTS.get(problem.ic_type, IC_DEFAULTS["value"])
    base_u = float(ic.get("u0", 1.0))
    base_v = float(ic.get("v0", 0.0))

    if problem.ic_type == "bounded":
        u0 = float(rng.uniform(max(0.05, 0.5 * base_u), max(0.2, 1.5 * base_u)))
    else:
        u0 = float(rng.uniform(0.5 * base_u, 1.5 * base_u))

    if int(problem.order) == 1:
        return u0, 0.0

    if problem.ic_type == "oscillatory":
        v0 = float(rng.uniform(-1.5, 1.5))
    elif problem.ic_type == "decay":
        v0 = float(rng.uniform(-0.7, 0.7))
    elif problem.ic_type == "bounded":
        v0 = float(rng.uniform(-0.5, 0.5))
    else:
        v0 = float(rng.uniform(-1.0, 1.0))
    v0 += 0.15 * base_v
    return u0, v0


def _trajectory_support_stats(y_arr: np.ndarray, y0: Sequence[float]) -> dict[str, float | bool | str]:
    arr = np.asarray(y_arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    abs_arr = np.abs(arr)
    max_abs_state = float(np.nanmax(abs_arr)) if abs_arr.size else float("inf")
    y0_arr = np.asarray(list(y0), dtype=np.float64).reshape(-1)
    initial_scale = max(1.0, float(np.nanmax(np.abs(y0_arr))) if y0_arr.size else 1.0)
    state_growth = max_abs_state / max(initial_scale, 1.0e-300)
    ok = (
        math.isfinite(max_abs_state)
        and math.isfinite(state_growth)
        and max_abs_state <= float(GEN_SUPPORT_MAX_ABS_STATE)
        and state_growth <= float(GEN_SUPPORT_MAX_STATE_GROWTH)
    )
    reason = "ok"
    if not math.isfinite(max_abs_state) or not math.isfinite(state_growth):
        reason = "nonfinite_state_scale"
    elif max_abs_state > float(GEN_SUPPORT_MAX_ABS_STATE):
        reason = "max_abs_state"
    elif state_growth > float(GEN_SUPPORT_MAX_STATE_GROWTH):
        reason = "state_growth"
    return {
        "ok": bool(ok),
        "reason": str(reason),
        "max_abs_state": float(max_abs_state),
        "initial_scale": float(initial_scale),
        "state_growth": float(state_growth),
        "max_abs_state_limit": float(GEN_SUPPORT_MAX_ABS_STATE),
        "state_growth_limit": float(GEN_SUPPORT_MAX_STATE_GROWTH),
    }


def _solve_trajectory_supported(
    rhs_fn,
    param_values: dict[str, float],
    *,
    x_min: float,
    nominal_x_max: float,
    y0: Sequence[float],
    n_points: int,
) -> tuple[Any | None, dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    nominal_span = float(nominal_x_max) - float(x_min)
    if not math.isfinite(nominal_span) or nominal_span <= 0.0:
        return None, None, None, [f"invalid nominal span: x_min={x_min}, x_max={nominal_x_max}"]
    min_span = max(abs(nominal_span) * float(GEN_SUPPORT_MIN_SPAN_FRACTION), 1.0e-12)
    errors: list[str] = []

    for support_backoff in range(int(GEN_SUPPORT_MAX_BACKOFFS) + 1):
        span = max(min_span, nominal_span * (float(GEN_SUPPORT_BACKOFF_FACTOR) ** int(support_backoff)))
        x_max = float(x_min) + float(span)
        x_eval = np.linspace(float(x_min), float(x_max), int(n_points), dtype=np.float64)

        for method, rtol, atol in GEN_SOLVER_TRIALS:
            try:
                cand = solve_ivp(
                    lambda x, s: rhs_fn(float(x), s, param_values),
                    [float(x_min), float(x_max)],
                    list(y0),
                    t_eval=x_eval,
                    method=method,
                    rtol=float(rtol),
                    atol=float(atol),
                )
            except Exception as exc:
                errors.append(f"backoff={support_backoff} method={method}: {exc}")
                continue

            x = np.asarray(cand.t, dtype=np.float64)
            y_arr = np.asarray(cand.y, dtype=np.float64)
            u = np.asarray(y_arr[0], dtype=np.float64) if y_arr.ndim >= 2 and y_arr.shape[0] >= 1 else np.array([])
            if int(cand.status) != 0:
                errors.append(
                    f"backoff={support_backoff} method={method}: status={int(cand.status)} msg={cand.message}"
                )
                continue
            if x.size <= 4 or u.size <= 4:
                errors.append(f"backoff={support_backoff} method={method}: insufficient points ({int(x.size)})")
                continue
            if (not np.isfinite(x).all()) or (not np.isfinite(y_arr).all()):
                errors.append(f"backoff={support_backoff} method={method}: non-finite trajectory")
                continue

            support = _trajectory_support_stats(y_arr, y0)
            support.update(
                {
                    "policy": "state_growth_backoff",
                    "nominal_x_max": float(nominal_x_max),
                    "selected_x_max": float(x_max),
                    "x_min": float(x_min),
                    "backoffs": int(support_backoff),
                    "backoff_factor": float(GEN_SUPPORT_BACKOFF_FACTOR),
                    "min_span_fraction": float(GEN_SUPPORT_MIN_SPAN_FRACTION),
                }
            )
            if bool(support.get("ok", False)) or int(support_backoff) >= int(GEN_SUPPORT_MAX_BACKOFFS):
                solver = {
                    "method": str(method),
                    "rtol": float(rtol),
                    "atol": float(atol),
                    "support_backoffs": int(support_backoff),
                }
                return cand, solver, support, errors
            errors.append(
                "backoff={backoff} method={method}: support {reason} "
                "growth={growth:.3g} max_abs={max_abs:.3g}; shrinking horizon".format(
                    backoff=int(support_backoff),
                    method=str(method),
                    reason=str(support.get("reason", "unknown")),
                    growth=float(support.get("state_growth", float("inf"))),
                    max_abs=float(support.get("max_abs_state", float("inf"))),
                )
            )
            break

    return None, None, None, errors


def _candidate_x_min_grid(nominal_x_max: float) -> list[float]:
    """Candidate start points for black-box support probing."""
    x_max = float(nominal_x_max)
    out: list[float] = []
    for raw in GEN_X_MIN_PILOT_CANDIDATES:
        x0 = float(raw)
        if not math.isfinite(x0) or x0 < 0.0:
            continue
        if math.isfinite(x_max) and x0 >= x_max:
            continue
        if not any(abs(x0 - prev) <= 1.0e-15 for prev in out):
            out.append(x0)
    if not out:
        out.append(0.0)
    return out


def _select_supported_x_min(
    problem: ProblemDef,
    rhs_fn,
    param_values: dict[str, float],
    *,
    nominal_x_max: float,
    seed: int,
    n_traj: int,
) -> tuple[float, dict[str, Any]]:
    """Choose a start point from black-box pilot trajectory health.

    The benchmark generator knows the RHS oracle, but this policy does not
    inspect the symbolic equation. It tries a small grid of candidate start
    points and chooses the earliest one whose pilot trajectories cover a useful
    span with bounded state growth. This catches near-singular start points and
    preserves ordinary ``x_min=0`` data when it is numerically supported.
    """
    candidates = _candidate_x_min_grid(float(nominal_x_max))
    n_pilot = max(1, min(int(GEN_X_MIN_PILOT_TRAJ), int(n_traj)))
    records: list[dict[str, Any]] = []

    for x_min in candidates:
        cand_rng = np.random.default_rng(int(seed) + 100_003)
        pilot_records: list[dict[str, Any]] = []
        solved_all = True
        errors: list[str] = []
        for j in range(n_pilot):
            u0, v0 = _sample_ic(problem, param_values, cand_rng, x_min=float(x_min), ic_index=j)
            y0 = [u0] if int(problem.order) == 1 else [u0, v0]
            cand, solver_info, support_info, solve_errors = _solve_trajectory_supported(
                rhs_fn,
                param_values,
                x_min=float(x_min),
                nominal_x_max=float(nominal_x_max),
                y0=y0,
                n_points=int(GEN_X_MIN_PILOT_POINTS),
            )
            errors.extend(solve_errors[-3:])
            if cand is None or solver_info is None or support_info is None:
                solved_all = False
                break
            span = max(float(nominal_x_max) - float(x_min), 1.0e-300)
            selected_span = float(support_info.get("selected_x_max", x_min)) - float(x_min)
            pilot_records.append(
                {
                    "traj_index": int(j),
                    "solver": dict(solver_info),
                    "support": dict(support_info),
                    "selected_span_fraction": float(selected_span / span),
                }
            )

        if pilot_records:
            span_fracs = [float(r["selected_span_fraction"]) for r in pilot_records]
            growths = [float((r["support"] or {}).get("state_growth", float("inf"))) for r in pilot_records]
            abs_states = [float((r["support"] or {}).get("max_abs_state", float("inf"))) for r in pilot_records]
            backoffs = [int((r["solver"] or {}).get("support_backoffs", 0) or 0) for r in pilot_records]
            min_span_frac = float(min(span_fracs))
            max_growth = float(max(growths))
            max_abs_state = float(max(abs_states))
            max_backoffs = int(max(backoffs))
        else:
            min_span_frac = 0.0
            max_growth = float("inf")
            max_abs_state = float("inf")
            max_backoffs = int(GEN_SUPPORT_MAX_BACKOFFS) + 1

        ok = (
            bool(solved_all)
            and len(pilot_records) == n_pilot
            and min_span_frac >= float(GEN_X_MIN_PILOT_MIN_SPAN_FRACTION)
            and math.isfinite(max_growth)
            and math.isfinite(max_abs_state)
            and max_growth <= float(GEN_SUPPORT_MAX_STATE_GROWTH)
            and max_abs_state <= float(GEN_SUPPORT_MAX_ABS_STATE)
        )
        good = bool(ok) and max_growth <= float(GEN_X_MIN_PILOT_GOOD_STATE_GROWTH)
        record = {
            "x_min": float(x_min),
            "ok": bool(ok),
            "good": bool(good),
            "n_pilot": int(n_pilot),
            "n_solved": int(len(pilot_records)),
            "min_selected_span_fraction": float(min_span_frac),
            "max_state_growth": float(max_growth) if math.isfinite(max_growth) else None,
            "max_abs_state": float(max_abs_state) if math.isfinite(max_abs_state) else None,
            "max_support_backoffs": int(max_backoffs),
            "errors_tail": errors[-6:],
            "pilots": pilot_records,
        }
        records.append(record)
        if good:
            meta = {
                "policy": "black_box_support_pilot",
                "selected_x_min": float(x_min),
                "reason": "first_good_candidate",
                "candidates": records,
            }
            return float(x_min), meta

    for record in records:
        if bool(record.get("ok", False)):
            x_min = float(record["x_min"])
            meta = {
                "policy": "black_box_support_pilot",
                "selected_x_min": float(x_min),
                "reason": "first_supported_candidate",
                "candidates": records,
            }
            return float(x_min), meta

    best = max(
        records,
        key=lambda r: (
            float(r.get("min_selected_span_fraction", 0.0) or 0.0),
            -int(r.get("max_support_backoffs", int(GEN_SUPPORT_MAX_BACKOFFS) + 1) or 0),
            -float(r.get("max_state_growth", float("inf")) or float("inf")),
        ),
    ) if records else {"x_min": float(_default_x_min(problem))}
    x_min = float(best.get("x_min", _default_x_min(problem)))
    meta = {
        "policy": "black_box_support_pilot",
        "selected_x_min": float(x_min),
        "reason": "best_available_candidate",
        "candidates": records,
    }
    return float(x_min), meta


def _subprocess_env_with_repo_root() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = str(REPO_ROOT)
    py_path_prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{py_path_prev}" if py_path_prev else repo_root
    )
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


@dataclass
class _LoggedCommandResult:
    args: list[str]
    returncode: int
    resource_report: dict[str, Any]


def _signal_name_from_returncode(returncode: int) -> str:
    if int(returncode) >= 0:
        return ""
    try:
        return signal.Signals(-int(returncode)).name
    except Exception:
        return f"SIG{-int(returncode)}"


def _sample_process_tree_rss(root_pid: int) -> dict[str, Any]:
    """Return current RSS for root_pid and descendants using ps when available."""

    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss=,command="],
            text=True,
            capture_output=True,
            timeout=2.0,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    if int(getattr(proc, "returncode", 1)) != 0:
        return {"ok": False, "error": str(getattr(proc, "stderr", "") or "ps failed")}

    children: dict[int, list[int]] = {}
    rows: dict[int, tuple[int, int, str]] = {}
    for raw in str(proc.stdout or "").splitlines():
        parts = raw.strip().split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_kb = int(parts[2])
        except Exception:
            continue
        command = parts[3] if len(parts) >= 4 else ""
        rows[pid] = (ppid, rss_kb, command)
        children.setdefault(ppid, []).append(pid)

    root_pid_i = int(root_pid)
    stack = [root_pid_i]
    seen: set[int] = set()
    tree: list[int] = []
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if pid in rows:
            tree.append(pid)
        stack.extend(children.get(pid, []))

    if not tree:
        return {"ok": False, "root_pid": root_pid_i, "error": "process tree not found"}

    total_rss_kb = int(sum(rows[pid][1] for pid in tree))
    peak_pid = max(tree, key=lambda pid: rows[pid][1])
    peak_ppid, peak_rss_kb, peak_command = rows[peak_pid]
    return {
        "ok": True,
        "root_pid": root_pid_i,
        "process_count": int(len(tree)),
        "tree_rss_mb": float(total_rss_kb) / 1024.0,
        "tree_rss_kb": int(total_rss_kb),
        "largest_process_pid": int(peak_pid),
        "largest_process_ppid": int(peak_ppid),
        "largest_process_rss_mb": float(peak_rss_kb) / 1024.0,
        "largest_process_rss_kb": int(peak_rss_kb),
        "largest_process_command": str(peak_command)[:300],
    }


def _resource_report_from_samples(
    *,
    returncode: int,
    samples: Sequence[Mapping[str, Any]],
    started: float,
    ended: float,
    monitor_interval_s: float,
) -> dict[str, Any]:
    ok_samples = [sample for sample in samples if bool(sample.get("ok", False))]
    last_sample = ok_samples[-1] if ok_samples else {}
    peak_sample = (
        max(ok_samples, key=lambda sample: float(sample.get("tree_rss_mb", -1.0)))
        if ok_samples
        else {}
    )
    report: dict[str, Any] = {
        "monitor": "ps_tree_rss",
        "returncode": int(returncode),
        "wall_time_s": float(max(0.0, ended - started)),
        "sample_interval_s": float(monitor_interval_s),
        "sample_count": int(len(ok_samples)),
        "sample_error_count": int(len(samples) - len(ok_samples)),
        "killed_by_signal": bool(int(returncode) < 0),
        "signal": _signal_name_from_returncode(int(returncode)),
    }
    if ok_samples:
        report.update(
            {
                "last_tree_rss_mb": float(last_sample.get("tree_rss_mb", 0.0)),
                "last_tree_process_count": int(last_sample.get("process_count", 0) or 0),
                "peak_tree_rss_mb": float(peak_sample.get("tree_rss_mb", 0.0)),
                "peak_tree_process_count": int(peak_sample.get("process_count", 0) or 0),
                "peak_largest_process_pid": int(peak_sample.get("largest_process_pid", 0) or 0),
                "peak_largest_process_rss_mb": float(
                    peak_sample.get("largest_process_rss_mb", 0.0)
                ),
                "peak_largest_process_command": str(
                    peak_sample.get("largest_process_command", "") or ""
                ),
            }
        )
    else:
        last_error = samples[-1].get("error", "") if samples else "no samples"
        report["monitor_error"] = str(last_error)
    return report


def _format_command_failure_message(command_label: str, proc: _LoggedCommandResult) -> str:
    msg = f"{command_label} failed (rc={int(proc.returncode)})"
    report = _command_resource_report(proc)
    peak = report.get("peak_tree_rss_mb", None)
    if int(proc.returncode) == -9 and peak is not None:
        try:
            msg += f"; peak_tree_rss={float(peak):.1f} MB"
        except Exception:
            pass
    return msg


def _command_resource_report(proc: Any) -> dict[str, Any]:
    report = getattr(proc, "resource_report", None)
    return dict(report) if isinstance(report, dict) else {}


def _attach_command_resource_report(out: dict[str, Any], proc: Any) -> dict[str, Any]:
    report = _command_resource_report(proc)
    if report:
        out["command_resource_report"] = dict(report)
        if report.get("peak_tree_rss_mb", None) is not None:
            out["command_peak_tree_rss_mb"] = float(report["peak_tree_rss_mb"])
        if report.get("last_tree_rss_mb", None) is not None:
            out["command_last_tree_rss_mb"] = float(report["last_tree_rss_mb"])
    if int(proc.returncode) < 0:
        out["command_killed_by_signal"] = True
        out["command_signal"] = _signal_name_from_returncode(int(proc.returncode))
    if int(proc.returncode) == -9:
        out["resource_failure_suspected"] = True
    return out


def _run_command_to_log(
    cmd: Sequence[str],
    log_path: Path,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> _LoggedCommandResult:
    """Run a child command while streaming stdout/stderr to ``log_path``."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(str(part)) for part in cmd) + "\n\n")
        log.flush()

        # Tests monkeypatch subprocess.run heavily; preserve that seam while
        # using Popen monitoring in normal benchmark execution.
        if subprocess.run is not _ORIGINAL_SUBPROCESS_RUN:
            completed = subprocess.run(
                list(cmd),
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
            )
            return _LoggedCommandResult(
                args=list(cmd),
                returncode=int(completed.returncode),
                resource_report={
                    "monitor": "disabled_monkeypatched_subprocess_run",
                    "returncode": int(completed.returncode),
                    "sample_count": 0,
                    "sample_error_count": 0,
                    "killed_by_signal": bool(int(completed.returncode) < 0),
                    "signal": _signal_name_from_returncode(int(completed.returncode)),
                },
            )

        monitor_interval_s = max(
            0.25,
            float(os.environ.get("DE_BENCHMARK_MEMORY_SAMPLE_INTERVAL_S", "2.0") or 2.0),
        )
        started = time.perf_counter()
        proc = subprocess.Popen(
            list(cmd),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
        )
        samples: list[dict[str, Any]] = []
        next_sample = started
        while True:
            now = time.perf_counter()
            returncode = proc.poll()
            if now >= next_sample or returncode is not None:
                sample = _sample_process_tree_rss(int(proc.pid))
                sample["elapsed_s"] = float(max(0.0, now - started))
                samples.append(sample)
                next_sample = now + monitor_interval_s
            if returncode is not None:
                break
            time.sleep(min(0.25, monitor_interval_s))

        ended = time.perf_counter()
        report = _resource_report_from_samples(
            returncode=int(proc.returncode),
            samples=samples,
            started=started,
            ended=ended,
            monitor_interval_s=float(monitor_interval_s),
        )
        log.write("\n[resource-monitor] " + json.dumps(report, sort_keys=True) + "\n")
        log.flush()
        return _LoggedCommandResult(
            args=list(cmd),
            returncode=int(proc.returncode),
            resource_report=report,
        )


def generate_data_multi(
    problem: ProblemDef,
    param_values: dict[str, float],
    out_dir: Path,
    *,
    n_traj: int,
    n_points: int,
    seed: int,
) -> tuple[list[TrajRun], str]:
    rhs_fn, rhs_source = resolve_rhs(problem, prefer_manual=True)
    nominal_x_max = float(default_t_max(problem, param_values))
    x_min, x_start_meta = _select_supported_x_min(
        problem,
        rhs_fn,
        param_values,
        nominal_x_max=float(nominal_x_max),
        seed=int(seed),
        n_traj=int(n_traj),
    )
    rng = np.random.default_rng(int(seed))

    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[TrajRun] = []

    for k in range(int(n_traj)):
        sol = None
        u0 = float("nan")
        v0 = float("nan")
        solver_method = ""
        solver_rtol = float("nan")
        solver_atol = float("nan")
        solver_support_backoffs = 0
        ic_retry = -1
        support_meta: dict[str, Any] | None = None
        attempt_errors: list[str] = []

        for retry in range(int(GEN_MAX_IC_RETRIES)):
            u0, v0 = _sample_ic(problem, param_values, rng, x_min=x_min, ic_index=k)
            y0 = [u0] if int(problem.order) == 1 else [u0, v0]
            cand, solver_info, support_info, errors = _solve_trajectory_supported(
                rhs_fn,
                param_values,
                x_min=float(x_min),
                nominal_x_max=float(nominal_x_max),
                y0=y0,
                n_points=int(n_points),
            )
            attempt_errors.extend(f"retry={retry} {err}" for err in errors)
            if cand is not None and solver_info is not None and support_info is not None:
                sol = cand
                solver_method = str(solver_info.get("method", ""))
                solver_rtol = float(solver_info.get("rtol", float("nan")))
                solver_atol = float(solver_info.get("atol", float("nan")))
                solver_support_backoffs = int(solver_info.get("support_backoffs", 0) or 0)
                support_meta = dict(support_info)
                ic_retry = int(retry)
                break

        if sol is None:
            tail = "; ".join(attempt_errors[-4:]) if attempt_errors else "no solver attempts recorded"
            raise RuntimeError(
                f"Integration failed for de{problem.id} ic{k} after {int(GEN_MAX_IC_RETRIES)} IC retries: {tail}"
            )

        x = np.asarray(sol.t, dtype=np.float64)
        u = np.asarray(sol.y[0], dtype=np.float64)
        x_max = float(x[-1])

        csv_path = out_dir / f"de{problem.id}_ic{k}.csv"
        meta_path = out_dir / f"de{problem.id}_ic{k}.meta.json"
        np.savetxt(
            str(csv_path),
            np.column_stack([u, x]),
            delimiter=",",
            header="y,x0",
            comments="",
        )
        meta_payload = {
            "id": str(problem.id),
            "traj_id": f"ic{k}",
            "ic_index": int(k),
            "order": int(problem.order),
            "x_min": float(x_min),
            "x_max": float(x_max),
            "nominal_x_max": float(nominal_x_max),
            "u0": float(u0),
            "v0": float(v0),
            "params": {str(kn): float(vn) for kn, vn in param_values.items()},
            "rhs_source": str(rhs_source),
            "solver": {
                "method": str(solver_method),
                "rtol": float(solver_rtol),
                "atol": float(solver_atol),
                "ic_retry": int(ic_retry),
                "support_backoffs": int(solver_support_backoffs),
            },
            "x_start": x_start_meta,
            "horizon": support_meta or {},
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        runs.append(
            TrajRun(
                traj_id=f"ic{k}",
                csv_path=csv_path,
                meta_path=meta_path,
                x_min=float(x_min),
                x_max=float(x_max),
                u0=float(u0),
                v0=float(v0),
            )
        )

    return runs, str(rhs_source)


def _load_legacy_single_run(problem: ProblemDef, data_dir: Path) -> TrajRun | None:
    legacy_csv = data_dir / f"de{problem.id}.csv"
    if not legacy_csv.exists():
        return None
    data = np.loadtxt(str(legacy_csv), delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        return None
    u = np.asarray(data[:, 0], dtype=np.float64)
    x = np.asarray(data[:, 1], dtype=np.float64)
    if u.size < 2:
        return None
    du0 = 0.0
    if int(problem.order) >= 2 and x.size >= 2 and float(x[1] - x[0]) != 0.0:
        du0 = float((u[1] - u[0]) / (x[1] - x[0]))
    meta_path = data_dir / f"de{problem.id}_ic0.meta.json"
    if not meta_path.exists():
        payload = {
            "id": str(problem.id),
            "traj_id": "ic0",
            "ic_index": 0,
            "order": int(problem.order),
            "x_min": float(x[0]),
            "x_max": float(x[-1]),
            "u0": float(u[0]),
            "v0": float(du0),
            "params": {},
            "rhs_source": "legacy_csv",
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return TrajRun(
        traj_id="ic0",
        csv_path=legacy_csv,
        meta_path=meta_path,
        x_min=float(x[0]),
        x_max=float(x[-1]),
        u0=float(u[0]),
        v0=float(du0),
    )


def load_existing_runs(problem: ProblemDef, data_dir: Path, *, n_traj: int) -> tuple[list[TrajRun], str]:
    runs: list[TrajRun] = []
    for k in range(int(n_traj)):
        csv_path = data_dir / f"de{problem.id}_ic{k}.csv"
        meta_path = data_dir / f"de{problem.id}_ic{k}.meta.json"
        if not csv_path.exists() or not meta_path.exists():
            if int(n_traj) == 1 and int(k) == 0:
                legacy = _load_legacy_single_run(problem, data_dir)
                if legacy is not None:
                    return [legacy], "existing_csv_legacy"
            raise FileNotFoundError(
                f"Missing trajectory files for de{problem.id}: {csv_path.name} and/or {meta_path.name}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        runs.append(
            TrajRun(
                traj_id=str(meta.get("traj_id", f"ic{k}")),
                csv_path=csv_path,
                meta_path=meta_path,
                x_min=float(meta.get("x_min", 0.0)),
                x_max=float(meta.get("x_max", 1.0)),
                u0=float(meta.get("u0", 1.0)),
                v0=float(meta.get("v0", 0.0)),
            )
        )
    return runs, "existing_csv"


def _derive_run_de_base_filename(paths: Sequence[Path]) -> str:
    if len(paths) == 1:
        return paths[0].stem
    stems = [p.stem for p in paths]
    common = os.path.commonprefix(stems).rstrip("_-.")
    if common and len(common) >= 3:
        return f"{common}_multi{len(paths)}"
    return f"multi{len(paths)}_{stems[0]}"


def build_run_de_command(
    problem: ProblemDef,
    csv_paths: Sequence[Path],
    results_dir: Path,
    *,
    fast: bool,
    rescue: bool,
    factorized_search_only: bool = False,
    factorized_de: bool = False,
    factorized_search_preset: str | None = None,
    factorized_two_block_shared_coord: str | None = None,
    de_coe_mode: str = "off",
    de_coe_csr_on_ties: bool = False,
    de_coe_reservoir_scouts: int = 0,
    factorized_de_whole_rhs: str = "auto",
    factorized_de_typed_lanes: str = "never",
    factorized_de_typed_lane_workers: int = 1,
    factorized_search_de_refine_mode: str = "rare_final_polish",
    factorized_search_max_attempts: int | None = None,
    factorized_search_integrate_topk: int | None = None,
    factorized_search_direct_generator_witness_topk: int | None = None,
    factorized_rescue_mode: str | None = None,
    factorized_search_rescue_mode: str | None = None,
    use_dims: bool = True,
) -> list[str]:
    if sum(bool(v) for v in (rescue, factorized_search_only, factorized_de)) > 1:
        raise ValueError(
            "build_run_de_command() received incompatible rescue/factorized_search_only/factorized_de flags"
        )

    epochs = 120 if fast else 5000
    epochs_min = 20 if fast else 300
    nval_patience = 40 if fast else 250
    num_segments = 12 if fast else 48
    loss_target = "1e-6" if fast else "1e-8"
    batch_size = 256 if fast else 2000
    ndata_train = 480 if fast else 2000
    ndata_val = 480 if fast else 2000

    cmd = [sys.executable, str(RUN_DE_SCRIPT)]
    if len(csv_paths) == 1:
        cmd += ["--filepath", str(csv_paths[0])]
    else:
        cmd += ["--filepaths", *[str(p) for p in csv_paths]]

    cmd += [
        "--order_candidates",
        str(int(problem.order)),
        "--max_u_power",
        "3",
        "--max_x_power",
        "2",
        "--max_xu_total_degree",
        "3",
        "--include_du",
        "--include_xdu",
        "--include_udu",
        "--epochs",
        str(epochs),
        "--epochs_min",
        str(epochs_min),
        "--nval_patience",
        str(nval_patience),
        "--num_segments",
        str(num_segments),
        "--loss_target",
        str(loss_target),
        "--batch_size",
        str(batch_size),
        "--ndata_train",
        str(ndata_train),
        "--ndata_val",
        str(ndata_val),
        "--data_split",
        "interleaved",
        "--stlsq_lambda",
        "0.1",
        "--ridge",
        "1e-6",
        "--sparsity_penalty",
        "0.1",
        "--output_dir",
        str(results_dir),
        "--save_json",
    ]
    if _declares_singular_origin(problem):
        cmd += [
            "--include_inv_xdu",
            "--include_inv_xu",
            "--include_inv_x2u",
        ]
    canonical_dims = get_canonical_problem_dims(str(problem.id))
    if not bool(use_dims) or canonical_dims is None:
        cmd.append("--ignore_units")
    else:
        dims_payload = canonical_scalar_dims_payload(canonical_dims, x_axis=0, component_idx=0)
        cmd += [
            "--y_units",
            json.dumps(dims_payload["u"]),
            "--x_units",
            json.dumps([dims_payload["x"]]),
            "--units_basis",
            ",".join(str(b) for b in dims_payload["basis"]),
        ]
        local_consts = _local_const_dims_for_units(canonical_dims)
        if local_consts:
            cmd += ["--local_consts", json.dumps(local_consts)]
    if rescue or factorized_search_only or factorized_de:
        preset = str(factorized_search_preset) if factorized_search_preset is not None else ("fast" if fast else "paper")
        cmd += [
            "--factorized-search-preset",
            preset,
        ]
    if rescue:
        cmd += [
            "--factorized-rescue",
            str(factorized_rescue_mode or "auto"),
            "--factorized-search-rescue",
            str(factorized_search_rescue_mode or "auto"),
        ]
    if rescue or factorized_de:
        if factorized_two_block_shared_coord is not None:
            cmd += [
                "--factorized-two-block-shared-coord",
                str(factorized_two_block_shared_coord),
            ]
    if factorized_search_only:
        cmd.append("--factorized-search-only")
    if factorized_de:
        cmd.append("--factorized-de")
        cmd += ["--factorized-de-whole-rhs", str(factorized_de_whole_rhs)]
        cmd += ["--factorized-de-typed-lanes", str(factorized_de_typed_lanes)]
        cmd += ["--factorized-de-typed-lane-workers", str(max(1, int(factorized_de_typed_lane_workers)))]
    cmd += ["--factorized-search-de-refine-mode", str(factorized_search_de_refine_mode)]
    if (
        (rescue or factorized_search_only or factorized_de)
        and factorized_search_max_attempts is not None
    ):
        cmd += [
            "--factorized-search-max-attempts",
            str(max(0, int(factorized_search_max_attempts))),
        ]
    if (
        (rescue or factorized_search_only or factorized_de)
        and factorized_search_integrate_topk is not None
    ):
        cmd += [
            "--factorized-search-integrate-topk",
            str(max(0, int(factorized_search_integrate_topk))),
        ]
    if (
        (rescue or factorized_search_only or factorized_de)
        and factorized_search_direct_generator_witness_topk is not None
    ):
        cmd += [
            "--factorized-search-direct-generator-witness-topk",
            str(max(0, int(factorized_search_direct_generator_witness_topk))),
        ]
    if str(de_coe_mode or "off") != "off":
        cmd += ["--de-coe-mode", str(de_coe_mode)]
    if int(de_coe_reservoir_scouts) > 0:
        cmd += ["--de-coe-reservoir-scouts", str(int(de_coe_reservoir_scouts))]
    if bool(de_coe_csr_on_ties):
        cmd.append("--de-coe-csr-on-ties")
    return cmd


def build_oracle_de_command(
    spec_path: Path,
    out_json: Path,
    *,
    fast: bool,
    verbose: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        ORACLE_DE_MOD,
        "--spec",
        str(spec_path),
        "--output",
        str(out_json),
        "--plus",
        "--refine_linear_combo",
        # DE discovery across trajectories with shared physics should use a
        # *single* global mapping/coefficients. Allowing per-trajectory affine
        # maps (joint scoring/refinement) tends to hide structural errors and is
        # strongly associated with order confusion, especially for singular
        # coordinate equations.
        "--no_refine_joint_score",
        "--no_refine_joint_refine",
        "--no_refine_joint_terms",
        # Make continuous skeleton refinement subsets cover the full x-range (important for
        # stiff/singular problems where the informative region may be tiny).
        "--refine_fit_subset_mode",
        "stratified",
        "--poly_degree",
        "1",
        "--mapping_complexity_penalty",
        "0.01",
        "--brute_max_expressions",
        "50000",
    ]
    if fast:
        cmd += [
            "--n_iter",
            "15000",
            "--max_depth",
            "5",
            "--n_fit",
            "2000",
            "--n_probe",
            "2000",
            "--return_topk",
            "5",
            "--refine_fit_subset",
            "512",
        ]
    else:
        cmd += [
            "--n_iter",
            "60000",
            "--max_depth",
            "6",
            "--n_fit",
            "6000",
            "--n_probe",
            "6000",
            "--return_topk",
            "8",
            "--refine_fit_subset",
            "1024",
        ]
    if not verbose:
        cmd += ["--quiet"]
    return cmd


def extract_coeff_map(report_json: Path) -> tuple[dict[str, float], dict]:
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    de = payload.get("de_discovery", {})
    terms = list(de.get("terms", []))
    coeffs = list(de.get("coefficients", []))
    coeff_map: dict[str, float] = {}
    for term, c in zip(terms, coeffs):
        if isinstance(c, (list, tuple, dict)):
            return {}, de
        coeff_map[str(term)] = float(c)
    return coeff_map, de


def _normalize_engine_name(engine: str) -> str:
    engine_l = str(engine).strip().lower()
    aliases = {
        "stlsq": "sparse",
        "factorized": "factorized_search_only",
        "factorized_search_de": "factorized_search_only",
        "both": "compare",
    }
    return aliases.get(engine_l, engine_l)


def _coeff_map_from_candidate(candidate: dict[str, Any]) -> dict[str, float]:
    terms = list(candidate.get("terms", []) or [])
    coeffs = list(candidate.get("coefficients", []) or [])
    out: dict[str, float] = {}
    for term, c in zip(terms, coeffs):
        if isinstance(c, (list, tuple, dict)):
            return {}
        out[str(term)] = float(c)
    return out


def _dim_payload(dim: Sequence[float]) -> list[int | float]:
    out: list[int | float] = []
    for value in dim:
        fv = float(value)
        iv = int(round(fv))
        out.append(iv if abs(fv - float(iv)) <= 1.0e-12 else fv)
    return out


def _dim_tuple(dim: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in dim)


def _dim_add(lhs: Sequence[float], rhs: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a) + float(b) for a, b in zip(lhs, rhs))


def _dim_sub(lhs: Sequence[float], rhs: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a) - float(b) for a, b in zip(lhs, rhs))


def _dim_scale(dim: Sequence[float], scale: float) -> tuple[float, ...]:
    return tuple(float(v) * float(scale) for v in dim)


def _local_const_dims_for_units(canonical_dims: Any) -> dict[str, list[int | float]]:
    """Expose declared constants and simple derived coefficient dimensions.

    The sparse DE library estimates coefficients numerically; unit filtering
    only needs to know which coefficient dimensions are physically plausible.
    A declared length ``L`` should therefore make both ``L`` and ``1/L``
    available as coefficient dimensions, and pairs of parameters should allow
    simple products/ratios such as ``R/L`` or ``k/m``.  Some physical
    coefficients are slightly deeper monomials, e.g. ``omega**2*mu/T`` for a
    string spatial mode, so we also expose a small bounded monomial closure.
    """
    raw_dims = {
        str(name): _dim_tuple(dim)
        for name, dim in sorted(dict(getattr(canonical_dims, "constant_dims", {}) or {}).items())
    }
    out: dict[str, list[int | float]] = {}

    def add(name: str, dim: Sequence[float]) -> None:
        out.setdefault(str(name), _dim_payload(dim))

    for name, dim in raw_dims.items():
        add(name, dim)
        add(f"inv_{name}", _dim_scale(dim, -1.0))
        add(f"{name}_sq", _dim_scale(dim, 2.0))
        add(f"inv_{name}_sq", _dim_scale(dim, -2.0))

    names = list(raw_dims)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            dim_l = raw_dims[left]
            dim_r = raw_dims[right]
            add(f"{left}_mul_{right}", _dim_add(dim_l, dim_r))
            add(f"{left}_over_{right}", _dim_sub(dim_l, dim_r))
            add(f"{right}_over_{left}", _dim_sub(dim_r, dim_l))

    def token(name: str, power: int) -> str:
        abs_power = abs(int(power))
        if abs_power == 1:
            return name
        if abs_power == 2:
            return f"{name}_sq"
        return f"{name}_pow{abs_power}"

    def monomial_name(exponents: Sequence[int]) -> str:
        numer = [token(name, exp) for name, exp in zip(names, exponents) if int(exp) > 0]
        denom = [token(name, exp) for name, exp in zip(names, exponents) if int(exp) < 0]
        label = "_mul_".join(numer) if numer else "1"
        if denom:
            label = f"{label}_over_{'_mul_'.join(denom)}"
        return label

    max_abs_exp = 2
    max_total_abs_exp = 4
    zero_dim = tuple(0.0 for _ in range(len(next(iter(raw_dims.values()), ()))))
    for exponents_raw in product(range(-max_abs_exp, max_abs_exp + 1), repeat=len(names)):
        exponents = tuple(int(exp) for exp in exponents_raw)
        total_abs = sum(abs(exp) for exp in exponents)
        if total_abs <= 1 or total_abs > max_total_abs_exp:
            continue
        dim = zero_dim
        for name, exp in zip(names, exponents):
            if exp:
                dim = _dim_add(dim, _dim_scale(raw_dims[name], float(exp)))
        add(monomial_name(exponents), dim)

    return {name: out[name] for name in sorted(out)}


def _candidate_canonical_equation(candidate: Mapping[str, Any], fallback: str = "") -> str:
    raw = candidate.get("canonical_equation_simplified", None)
    if raw:
        return str(raw)
    raw = candidate.get("canonical_equation", None)
    if raw:
        return str(raw)
    validation_candidate = candidate.get("validation_candidate", None)
    if isinstance(validation_candidate, Mapping):
        raw = validation_candidate.get("canonical_equation_simplified", None)
        if raw:
            return str(raw)
        raw = validation_candidate.get("canonical_equation", None)
        if raw:
            return str(raw)
    return str(fallback or "")


def _sorted_runs_for_split(runs: Sequence[TrajRun]) -> list[TrajRun]:
    return sorted(runs, key=lambda run: str(run.traj_id))


def _split_runs_for_holdout(
    runs: Sequence[TrajRun],
    *,
    split_mode: str,
    holdout_last_k: int,
) -> tuple[list[TrajRun], list[TrajRun]]:
    ordered = _sorted_runs_for_split(runs)
    if str(split_mode).strip().lower() != "traj_holdout" or int(holdout_last_k) <= 0:
        return ordered, ordered
    k = min(int(holdout_last_k), max(0, len(ordered) - 1))
    if k <= 0:
        return ordered, ordered
    return ordered[:-k], ordered[-k:]


def _validate_library_candidate(
    candidate: dict[str, Any],
    gt: GroundTruth,
    param_values: dict[str, float],
) -> tuple[str, str, dict[str, float]]:
    coeff_map = _coeff_map_from_candidate(candidate)
    status, message = validate(coeff_map, candidate, gt, param_values)
    return str(status), str(message), coeff_map


def classify_failure_kind(
    status: str | None,
    message: str | None,
    traj_scores: Sequence[dict[str, Any]] | None = None,
) -> str | None:
    status_s = str(status or "").upper()
    if status_s not in ("FAIL", "ERROR"):
        return None

    texts: list[str] = []
    if message:
        texts.append(str(message))
    for row in list(traj_scores or []):
        err = row.get("error", None)
        if err:
            texts.append(str(err))
    text_blob = "\n".join(texts).lower()

    if "timeout (budget" in text_blob or "exceeded" in text_blob:
        return "timeout"
    if (
        "non-finite factorized symbolic search candidate evaluation" in text_blob
        or "non-finite discovered rhs value" in text_blob
        or "non-finite discovered rhs" in text_blob
    ):
        return "nonfinite_candidate"
    if "terminated early (blow-up" in text_blob or "blow-up" in text_blob:
        return "blowup"
    if "nrmse mean=" in text_blob:
        return "high_nrmse"
    if "integration failed on" in text_blob:
        return "integration_failure"
    if (
        "run_de.py failed" in text_blob
        or "missing json" in text_blob
        or "json parse error" in text_blob
        or "could not build" in text_blob
        or "no candidate could be integrated" in text_blob
        or "invalid csv" in text_blob
        or "unsupported engine" in text_blob
    ):
        return "pipeline_error"
    return "other_failure"


def compute_failure_kind_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        kind = row.get("failure_kind", None)
        if kind is None:
            kind = classify_failure_kind(
                row.get("status", None),
                row.get("message", None),
                row.get("traj_scores", None),
            )
        if kind:
            counts[str(kind)] = counts.get(str(kind), 0) + 1
    return counts


_ROLLOUT_STATUS_RANK: dict[str, int] = {
    "PASS": 0,
    "PARTIAL": 1,
    "FAIL": 2,
    "UNVERIFIED": 3,
    "ERROR": 4,
}


def _max_traj_nrmse(traj_scores: Sequence[dict[str, Any]] | None) -> float:
    vals: list[float] = []
    for row in list(traj_scores or []):
        try:
            val = float(row.get("nrmse", float("inf")))
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
    return float(max(vals)) if vals else float("inf")


def _rollout_choice_key(engine_name: str, row: dict[str, Any]) -> tuple[int, float, int, int]:
    status = str(row.get("status", "ERROR"))
    rank = int(_ROLLOUT_STATUS_RANK.get(status, len(_ROLLOUT_STATUS_RANK) + 1))
    max_nrmse = _max_traj_nrmse(row.get("traj_scores", []))
    try:
        order = int(row.get("discovered_order", 99))
    except Exception:
        order = 99
    engine_bias = 0 if str(engine_name) == "stlsq" else 1
    return rank, float(max_nrmse), int(order), int(engine_bias)


def _rollout_structural_rank(engine_name: str, row: dict[str, Any]) -> int:
    name = str(engine_name)
    source_lane = str(row.get("source_lane", "") or row.get("selected_lane", "") or "")
    if name == "stlsq":
        return 0
    if name == "factorized":
        return 1
    if source_lane in {"direct_residual_fss", "regularized_implicit_residual"}:
        return 2
    if name == "factorized_search":
        return 3
    return 4


def _rollout_symbolic_size(row: dict[str, Any]) -> int:
    for key in ("symbolic_size_simplified", "symbolic_size_raw", "size"):
        try:
            value = row.get(key, None)
            if value is not None:
                out = int(value)
                if out > 0:
                    return out
        except Exception:
            continue
    return 10**9


def _rollout_complexity_key(row: dict[str, Any]) -> tuple[int, float, int]:
    try:
        snap_cost = float(row.get("projection_snap_cost", 0.0) or 0.0)
    except Exception:
        snap_cost = 0.0
    if not math.isfinite(snap_cost):
        snap_cost = 0.0
    return (_rollout_symbolic_size(row), float(snap_cost), _candidate_rank(row))


def _rollout_candidate_preferred(
    lhs_name: str,
    lhs_row: dict[str, Any],
    rhs_name: str,
    rhs_row: dict[str, Any],
) -> bool:
    lhs_status = int(_ROLLOUT_STATUS_RANK.get(str(lhs_row.get("status", "ERROR")), len(_ROLLOUT_STATUS_RANK) + 1))
    rhs_status = int(_ROLLOUT_STATUS_RANK.get(str(rhs_row.get("status", "ERROR")), len(_ROLLOUT_STATUS_RANK) + 1))
    if lhs_status != rhs_status:
        return lhs_status < rhs_status
    lhs_nrmse = _max_traj_nrmse(lhs_row.get("traj_scores", []))
    rhs_nrmse = _max_traj_nrmse(rhs_row.get("traj_scores", []))
    lhs_rank = _rollout_structural_rank(lhs_name, lhs_row)
    rhs_rank = _rollout_structural_rank(rhs_name, rhs_row)
    if lhs_rank != rhs_rank:
        if lhs_rank < rhs_rank:
            return bool(lhs_nrmse <= 2.0 * rhs_nrmse)
        return bool(lhs_nrmse < 0.5 * rhs_nrmse)
    if math.isfinite(lhs_nrmse) and math.isfinite(rhs_nrmse):
        close_tol = max(1.0e-9, 0.01 * max(abs(lhs_nrmse), abs(rhs_nrmse), 1.0e-12))
        if abs(lhs_nrmse - rhs_nrmse) <= close_tol:
            lhs_complexity = _rollout_complexity_key(lhs_row)
            rhs_complexity = _rollout_complexity_key(rhs_row)
            if lhs_complexity != rhs_complexity:
                return lhs_complexity < rhs_complexity
    if lhs_nrmse != rhs_nrmse:
        return lhs_nrmse < rhs_nrmse
    try:
        lhs_order = int(lhs_row.get("discovered_order", 99))
    except Exception:
        lhs_order = 99
    try:
        rhs_order = int(rhs_row.get("discovered_order", 99))
    except Exception:
        rhs_order = 99
    if lhs_order != rhs_order:
        return lhs_order < rhs_order
    return lhs_rank < rhs_rank


def _choose_rollout_candidate(
    candidates: Sequence[tuple[str, dict[str, Any]]],
    *,
    fallback_engine: str | None = None,
) -> tuple[str, dict[str, Any]]:
    valid = [(name, row) for name, row in candidates if isinstance(row, dict)]
    if not valid:
        raise ValueError("No rollout candidates available")

    if fallback_engine is not None:
        for name, row in valid:
            if str(name) == str(fallback_engine):
                fallback = (name, row)
                break
        else:
            fallback = valid[0]
    else:
        fallback = valid[0]

    best_name, best_row = fallback
    for name, row in valid:
        if _rollout_candidate_preferred(name, row, best_name, best_row):
            best_name, best_row = name, row
    return best_name, best_row


def _candidate_rank(row: dict[str, Any]) -> int:
    for key in ("candidate_rank", "shortlist_rank", "rerank_rank", "original_rank"):
        try:
            if key in row and row.get(key) is not None:
                return int(row.get(key))
        except Exception:
            continue
    return 10**9


def _choose_best_same_engine_rollout(
    engine_name: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    valid = [row for row in rows if isinstance(row, dict)]
    if not valid:
        raise ValueError("No rollout rows available")
    best_row = valid[0]
    for row in valid[1:]:
        if _rollout_candidate_preferred(engine_name, row, engine_name, best_row):
            best_row = row
    return best_row


def _domain_projection_report_ok(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return True
    if value.get("ok", None) is False:
        return False
    for key in ("domain_projection", "domain_projection_eval", "_domain_projection"):
        child = value.get(key, None)
        if isinstance(child, Mapping) and child.get("ok", None) is False:
            return False
    return True


def _structural_report_ok(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return True
    if value.get("structural_ok", None) is False:
        return False
    if value.get("structural_hard_reject", None) is True:
        return False
    if value.get("hidden_score_head", None) is True:
        return False
    return True


def _factorized_search_candidate_domain_safe(candidate: Mapping[str, Any]) -> bool:
    if not _structural_report_ok(candidate):
        return False
    if candidate.get("domain_ok", None) is False:
        return False
    if not _domain_projection_report_ok(candidate):
        return False
    mapping = candidate.get("mapping", None)
    if isinstance(mapping, Mapping) and not _domain_projection_report_ok(mapping):
        return False
    diagnostics = candidate.get("diagnostics", None)
    if isinstance(diagnostics, Mapping):
        if not _structural_report_ok(diagnostics):
            return False
        if not _domain_projection_report_ok(diagnostics):
            return False
        report = diagnostics.get("report", None)
        if isinstance(report, Mapping):
            best = report.get("best", None)
            if isinstance(best, Mapping) and not _structural_report_ok(best):
                return False
    return True


def _attach_committee_audit_fields(
    out: dict[str, Any],
    decision: dict[str, Any] | None,
    *,
    internal_selected_engine: str | None,
) -> None:
    if not isinstance(decision, dict) or not decision:
        return
    committee_engine = selected_engine_from_decision(decision)
    out["committee_decision"] = decision
    out["committee_selected_engine"] = str(committee_engine or "")
    out["internal_selected_engine_committee_mismatch"] = bool(
        committee_engine is not None
        and internal_selected_engine is not None
        and str(committee_engine) != str(internal_selected_engine)
    )


def _committee_selected_rollout_candidate(
    decision: dict[str, Any] | None,
    candidates: Sequence[tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]] | None:
    summary = selected_summary_from_decision(decision)
    if not isinstance(summary, dict):
        return None
    engine = str(summary.get("engine", "") or "")
    selected_id = str(summary.get("proposal_id", "") or "")
    canonical = str(summary.get("canonical_equation", "") or "")
    rank_raw = summary.get("candidate_rank", None)
    try:
        selected_rank = None if rank_raw is None else int(rank_raw)
    except Exception:
        selected_rank = None

    valid = [(name, row) for name, row in candidates if isinstance(row, dict)]
    if selected_id:
        for name, row in valid:
            if str(row.get("proposal_id", "") or "") == selected_id:
                return str(name), row
    if selected_rank is not None:
        for name, row in valid:
            if str(name) == engine and _candidate_rank(row) == selected_rank:
                return str(name), row
    if canonical:
        for name, row in valid:
            if str(name) == engine and str(row.get("canonical_equation", "") or "") == canonical:
                return str(name), row
    for name, row in valid:
        if str(name) == engine:
            return str(name), row
    return None


def _shortlist_candidate_by_rank(
    shortlist: Sequence[dict[str, Any]],
    rank: Any,
) -> dict[str, Any] | None:
    try:
        rank_i = int(rank)
    except Exception:
        return None
    for row in list(shortlist or []):
        if not isinstance(row, dict):
            continue
        if _candidate_rank(row) == rank_i:
            return dict(row)
    return None


def _maybe_run_csr_on_tied_factorized_search(
    *,
    committee_decision: dict[str, Any] | None,
    current_candidates: Sequence[tuple[str, dict[str, Any]]],
    factorized_search_shortlist: Sequence[dict[str, Any]],
    fit_runs: Sequence[TrajRun],
    probe_runs: Sequence[TrajRun],
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None,
    sim_validate_blowup_factor: float,
    sim_validate_blowup_abs: float,
    enabled: bool,
    max_candidates: int = 2,
    max_trials: int = 8,
    tolerance_nrmse: float = 1.0e-3,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    out_candidates = [(name, row) for name, row in list(current_candidates or []) if isinstance(row, dict)]
    diag: dict[str, Any] = {
        "enabled": bool(enabled),
        "invoked": False,
        "reason": "",
        "selected_for_refine": 0,
        "attempted": 0,
        "accepted": 0,
        "validated": 0,
        "max_candidates": int(max_candidates),
        "max_trials": int(max_trials),
        "tolerance_nrmse": float(tolerance_nrmse),
        "attempts": [],
    }
    if not bool(enabled):
        diag["reason"] = "disabled"
        return out_candidates, diag
    if not isinstance(committee_decision, dict) or not committee_decision:
        diag["reason"] = "missing_committee_decision"
        return out_candidates, diag

    tied = tied_candidate_summaries_from_decision(
        committee_decision,
        tolerance_nrmse=float(tolerance_nrmse),
        max_candidates=max(2, int(max_candidates) + 1),
        max_per_role=2,
    )
    if len(tied) < 2:
        diag["reason"] = "clear_committee_winner"
        return out_candidates, diag

    fss_summaries = [
        row for row in tied
        if str(row.get("engine", "")) == "factorized_search"
    ]
    if not fss_summaries:
        diag["reason"] = "no_factorized_search_tied_candidate"
        return out_candidates, diag

    seen_ranks: set[int] = set()
    to_refine: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for summary in fss_summaries:
        rank = summary.get("candidate_rank", None)
        try:
            rank_i = int(rank)
        except Exception:
            continue
        if rank_i in seen_ranks:
            continue
        source = _shortlist_candidate_by_rank(factorized_search_shortlist, rank_i)
        if source is None:
            continue
        seen_ranks.add(rank_i)
        to_refine.append((summary, source))
        if len(to_refine) >= int(max_candidates):
            break

    diag["selected_for_refine"] = int(len(to_refine))
    if not to_refine:
        diag["reason"] = "no_materializable_factorized_search_candidate"
        return out_candidates, diag

    diag["invoked"] = True
    for idx, (summary, source) in enumerate(to_refine):
        attempt: dict[str, Any] = {
            "candidate_rank": summary.get("candidate_rank", None),
            "proposal_id": summary.get("proposal_id", None),
        }
        diag["attempted"] = int(diag["attempted"]) + 1
        refine_res = refine_factorized_search_candidate_from_runs(
            source,
            fit_runs=fit_runs,
            probe_runs=probe_runs,
            max_trials=int(max_trials),
            profile="rare_final_polish",
            seed=17_003 + int(idx),
        )
        attempt.update({
            "accepted": bool(refine_res.get("accepted", False)),
            "reason": refine_res.get("reason", None),
            "base_probe_mse": refine_res.get("base_probe_mse", None),
            "refined_probe_mse": refine_res.get("refined_probe_mse", None),
            "trials_used": refine_res.get("trials_used", None),
            "error": refine_res.get("error", None),
        })
        if not bool(refine_res.get("accepted", False)):
            diag["attempts"].append(attempt)
            continue
        refined_candidate = dict(refine_res.get("candidate", {}) or {})
        if not refined_candidate:
            attempt["accepted"] = False
            attempt["reason"] = "empty_refined_candidate"
            diag["attempts"].append(attempt)
            continue
        diag["accepted"] = int(diag["accepted"]) + 1
        refined_eval = _evaluate_factorized_search_candidate_rollout(
            refined_candidate,
            probe_runs=probe_runs,
            pass_nrmse=float(pass_nrmse),
            partial_nrmse=float(partial_nrmse),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
        )
        refined_eval["de_coe_csr_refined"] = True
        refined_eval["source_candidate_rank"] = source.get("candidate_rank", source.get("shortlist_rank", None))
        refined_eval["candidate_rank"] = int(900_000 + idx)
        refined_eval["proposal_id"] = f"de_coe_csr:{summary.get('proposal_id', idx)}"
        out_candidates.append(("factorized_search", refined_eval))
        diag["validated"] = int(diag["validated"]) + 1
        attempt["rollout_status"] = refined_eval.get("status", None)
        attempt["rollout_traj_scores"] = refined_eval.get("traj_scores", [])
        diag["attempts"].append(attempt)

    if int(diag["accepted"]) <= 0:
        diag["reason"] = "no_refinement_accepted"
    else:
        diag["reason"] = "refined_candidates_validated"
    return out_candidates, diag


def _rollout_row_key(row: dict[str, Any]) -> tuple[str, int, int] | None:
    if not isinstance(row, dict):
        return None
    canonical_key = str(row.get("canonical_key", "") or "")
    if not canonical_key:
        equation = row.get("canonical_equation", None)
        if equation not in (None, ""):
            canonical_key = canonicalize_de_equation(equation)
    if not canonical_key:
        return None
    try:
        order = int(row.get("discovered_order", row.get("order", 0)))
    except Exception:
        order = 0
    try:
        x_axis = int(row.get("x_axis", 0))
    except Exception:
        x_axis = 0
    return canonical_key, order, x_axis


def _proposal_key(row: dict[str, Any]) -> tuple[str, int, int] | None:
    if not isinstance(row, dict):
        return None
    canonical_key = str(row.get("canonical_key", "") or "")
    if not canonical_key:
        equation = row.get("canonical_equation", None)
        if equation not in (None, ""):
            canonical_key = canonicalize_de_equation(equation)
    if not canonical_key:
        return None
    try:
        order = int(row.get("order", 0))
    except Exception:
        order = 0
    try:
        x_axis = int(row.get("x_axis", 0))
    except Exception:
        x_axis = 0
    return canonical_key, order, x_axis


def _reservoir_scout_subsets(
    fit_runs: Sequence[TrajRun],
    *,
    max_scouts: int,
) -> list[list[TrajRun]]:
    runs = list(fit_runs or [])
    if int(max_scouts) <= 0 or len(runs) <= 1:
        return []
    count = min(int(max_scouts), len(runs))
    subsets: list[list[TrajRun]] = []
    seen: set[tuple[str, ...]] = set()
    for idx in range(count):
        if len(runs) == 2:
            subset = [runs[idx % 2]]
        else:
            drop = idx % len(runs)
            subset = [run for j, run in enumerate(runs) if j != drop]
        key = tuple(str(run.csv_path) for run in subset)
        if not subset or key in seen:
            continue
        seen.add(key)
        subsets.append(subset)
    return subsets


def _run_reservoir_scout_reports(
    problem: ProblemDef,
    fit_runs: Sequence[TrajRun],
    *,
    results_dir: Path,
    fast: bool,
    verbose: bool,
    max_scouts: int,
    factorized_search_preset: str | None,
    use_dims: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diag: dict[str, Any] = {
        "enabled": int(max_scouts) > 0,
        "requested": int(max_scouts),
        "scouts_run": 0,
        "proposal_slates": 0,
        "proposals_seen": 0,
        "reports": [],
    }
    subsets = _reservoir_scout_subsets(fit_runs, max_scouts=int(max_scouts))
    scout_slates: list[dict[str, Any]] = []
    if not subsets:
        diag["reason"] = "not_enough_fit_trajectories" if int(max_scouts) > 0 else "disabled"
        return scout_slates, diag

    for scout_idx, subset in enumerate(subsets):
        scout_dir = results_dir / "de_coe_reservoir_scouts" / f"de{problem.id}_scout{scout_idx}"
        scout_dir.mkdir(parents=True, exist_ok=True)
        csv_paths = [run.csv_path for run in subset]
        cmd = build_run_de_command(
            problem,
            csv_paths,
            scout_dir,
            fast=True,
            rescue=True,
            factorized_search_preset=factorized_search_preset or "fast",
            de_coe_mode="off",
            de_coe_reservoir_scouts=0,
            factorized_search_de_refine_mode="off",
            factorized_rescue_mode="always",
            factorized_search_rescue_mode="never",
            use_dims=bool(use_dims),
        )
        if verbose:
            print(f"  [reservoir scout {scout_idx}] Command: {' '.join(cmd)}")
        started = time.perf_counter()
        log_path = scout_dir / f"de{problem.id}_reservoir_scout{scout_idx}.log"
        proc = _run_command_to_log(cmd, log_path)
        elapsed = time.perf_counter() - started
        report_row: dict[str, Any] = {
            "scout_index": int(scout_idx),
            "fit_trajectories": [str(run.csv_path) for run in subset],
            "returncode": int(proc.returncode),
            "log_path": str(log_path),
            "wall_time_s": float(elapsed),
            "command_resource_report": _command_resource_report(proc),
        }
        diag["scouts_run"] = int(diag["scouts_run"]) + 1
        if proc.returncode != 0:
            report_row["status"] = "ERROR"
            report_row["message"] = _format_command_failure_message("run_de.py", proc)
            _attach_command_resource_report(report_row, proc)
            diag["reports"].append(report_row)
            continue
        report_path = scout_dir / f"{_derive_run_de_base_filename(csv_paths)}_de.json"
        report_row["json_path"] = str(report_path)
        if not report_path.exists():
            report_row["status"] = "ERROR"
            report_row["message"] = f"Missing JSON: {report_path}"
            diag["reports"].append(report_row)
            continue
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            slate = payload.get("de_discovery", {}).get("proposal_slate", []) or []
            slate_rows = [row for row in slate if isinstance(row, dict)]
        except Exception as exc:
            report_row["status"] = "ERROR"
            report_row["message"] = f"JSON parse error: {exc}"
            diag["reports"].append(report_row)
            continue
        report_row["status"] = "OK"
        report_row["proposal_count"] = int(len(slate_rows))
        diag["proposal_slates"] = int(diag["proposal_slates"]) + 1
        diag["proposals_seen"] = int(diag["proposals_seen"]) + int(len(slate_rows))
        diag["reports"].append(report_row)
        scout_slates.append(
            {
                "namespace": f"reservoir_scout{scout_idx}",
                "proposal_slate": slate_rows,
            }
        )

    diag["reason"] = "ok" if scout_slates else "no_valid_scout_slates"
    return scout_slates, diag


def _merge_reservoir_scout_slates(
    main_slate: Sequence[dict[str, Any]],
    scout_slates: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slates: list[Sequence[dict[str, Any]]] = [[row for row in list(main_slate or []) if isinstance(row, dict)]]
    namespaces: list[str | None] = [None]
    for row in list(scout_slates or []):
        if not isinstance(row, dict):
            continue
        slate = [p for p in list(row.get("proposal_slate", []) or []) if isinstance(p, dict)]
        if not slate:
            continue
        slates.append(slate)
        namespaces.append(str(row.get("namespace", f"reservoir_scout{len(namespaces) - 1}")))
    merged = merge_proposal_slates(slates, source_namespaces=namespaces)
    support_counts = [
        int((proposal.get("support", {}) or {}).get("support_count", 0) or 0)
        for proposal in merged
        if isinstance(proposal, dict)
    ]
    diag = {
        "main_proposal_count": int(len(slates[0])),
        "scout_slate_count": int(max(0, len(slates) - 1)),
        "merged_proposal_count": int(len(merged)),
        "max_support_count": int(max(support_counts, default=0)),
    }
    return merged, diag


def _evaluate_reservoir_scout_proposals(
    proposal_slate: Sequence[dict[str, Any]],
    current_candidates: Sequence[tuple[str, dict[str, Any]]],
    *,
    probe_runs: Sequence[TrajRun],
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None,
    sim_validate_blowup_factor: float,
    sim_validate_blowup_abs: float,
    max_candidates: int,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    out_candidates = [(name, row) for name, row in list(current_candidates or []) if isinstance(row, dict)]
    diag: dict[str, Any] = {
        "selected_for_validation": 0,
        "validated": 0,
        "max_candidates": int(max_candidates),
        "candidate_ids": [],
    }
    if int(max_candidates) <= 0:
        diag["reason"] = "disabled"
        return out_candidates, diag

    existing_keys = {
        key for _, row in out_candidates
        for key in [_rollout_row_key(row)]
        if key is not None
    }
    candidates: list[dict[str, Any]] = []
    for proposal in list(proposal_slate or []):
        if not isinstance(proposal, dict):
            continue
        support = proposal.get("support", {}) or {}
        sources = [str(src) for src in list(support.get("sources", []) or [])]
        if not any(src.startswith("reservoir_scout") for src in sources):
            continue
        key = _proposal_key(proposal)
        if key is None or key in existing_keys:
            continue
        payload = proposal.get("rhs_payload", None)
        if not isinstance(payload, dict):
            continue
        candidates.append(proposal)

    def _proposal_eval_key(proposal: dict[str, Any]) -> tuple[int, float, float, str]:
        support_count = int((proposal.get("support", {}) or {}).get("support_count", 0) or 0)
        metrics = proposal.get("pointwise_metrics", {}) or {}
        pointwise = float("inf")
        for key in ("probe_mse", "mse", "score"):
            try:
                val = float(metrics.get(key, float("inf")))
            except Exception:
                val = float("inf")
            if math.isfinite(val):
                pointwise = val
                break
        return (
            -support_count,
            pointwise,
            float(proposal.get("complexity", 0.0) or 0.0),
            str(proposal.get("proposal_id", "")),
        )

    candidates.sort(key=_proposal_eval_key)
    for proposal in candidates[: int(max_candidates)]:
        payload = dict(proposal.get("rhs_payload", {}) or {})
        engine = str(proposal.get("engine", payload.get("engine", "")) or "")
        if engine == "factorized_search":
            row = _evaluate_factorized_search_candidate_rollout(
                payload,
                probe_runs=probe_runs,
                pass_nrmse=float(pass_nrmse),
                partial_nrmse=float(partial_nrmse),
                sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
                sim_validate_blowup_factor=float(sim_validate_blowup_factor),
                sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            )
            engine_name = "factorized_search"
        elif engine == "factorized":
            row = _evaluate_factorized_candidate_rollout(
                payload,
                probe_runs=probe_runs,
                pass_nrmse=float(pass_nrmse),
                partial_nrmse=float(partial_nrmse),
                sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
                sim_validate_blowup_factor=float(sim_validate_blowup_factor),
                sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            )
            engine_name = "factorized"
        else:
            row = _evaluate_library_candidate_rollout(
                payload,
                probe_runs=probe_runs,
                pass_nrmse=float(pass_nrmse),
                partial_nrmse=float(partial_nrmse),
                sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
                sim_validate_blowup_factor=float(sim_validate_blowup_factor),
                sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            )
            engine_name = "stlsq"
        row["proposal_id"] = str(proposal.get("proposal_id", "") or "")
        row["canonical_key"] = str(proposal.get("canonical_key", "") or "")
        row["role_signature"] = str(proposal.get("role_signature", "") or "")
        row["reservoir_scout"] = True
        row["support_count"] = int((proposal.get("support", {}) or {}).get("support_count", 0) or 0)
        out_candidates.append((engine_name, row))
        existing_key = _rollout_row_key(row)
        if existing_key is not None:
            existing_keys.add(existing_key)
        diag["validated"] = int(diag["validated"]) + 1
        diag["candidate_ids"].append(row["proposal_id"])

    diag["selected_for_validation"] = int(min(len(candidates), int(max_candidates)))
    diag["reason"] = "ok" if int(diag["validated"]) > 0 else "no_new_scout_candidates"
    return out_candidates, diag


def _factorized_search_shortlist_from_candidate(
    candidate: dict[str, Any],
    *,
    max_candidates: int = 0,
) -> list[dict[str, Any]]:
    if not isinstance(candidate, dict) or not candidate:
        return []
    shortlist = list(candidate.get("shortlist", []) or [])
    # Always include the parent candidate itself: its mapping is refit at
    # promotion time, so it may be the only faithful copy of the selected law
    # even when a serialized shortlist exists. Rank -1 makes it win rollout
    # ties against its own serialized twin.
    parent = dict(candidate)
    parent.pop("shortlist", None)
    parent.setdefault("candidate_rank", -1)
    parent.setdefault("shortlist_rank", -1)
    candidates = [parent]
    candidates.extend(row for row in shortlist if isinstance(row, dict))
    if int(max_candidates) > 0:
        candidates = candidates[: int(max_candidates)]
    return candidates


def _validation_payload_from_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    payload = candidate.get("validation_candidate", None)
    return payload if isinstance(payload, Mapping) else None


def _validation_term_asts(candidate: Mapping[str, Any]) -> list[Any]:
    payload = _validation_payload_from_candidate(candidate)
    if not isinstance(payload, Mapping):
        return []
    return list(payload.get("term_asts_json", []) or [])


def _ast_json_children(node: Any) -> list[Any]:
    if not isinstance(node, Mapping):
        return []
    kind = str(node.get("type", "")).lower()
    if kind in {"add", "sub", "mul", "div"}:
        return [node.get("left"), node.get("right")]
    if kind == "pow":
        children = [node.get("base")]
        exponent = node.get("exponent", None)
        if isinstance(exponent, Mapping):
            children.append(exponent)
        return children
    if kind in {"log", "exp", "sin", "cos", "sqrt", "abs", "arg", "real", "imag", "conj", "neg"}:
        return [node.get("arg")]
    return []


def _ast_json_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _iter_domain_sensitive_ast_nodes(node: Any) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if not isinstance(node, Mapping):
        return out
    kind = str(node.get("type", "")).lower()
    if kind == "pow":
        exponent = node.get("exponent", 1.0)
        exp_f = _ast_json_float(exponent) if not isinstance(exponent, Mapping) else None
        if exp_f is not None:
            if exp_f < 0.0:
                out.append(("negative_power", node.get("base")))
            elif exp_f > 0.0 and abs(exp_f - round(exp_f)) > 1.0e-12:
                out.append(("fractional_power", node.get("base")))
    elif kind == "log":
        out.append(("log", node.get("arg")))
    elif kind == "sqrt":
        out.append(("sqrt", node.get("arg")))
    for child in _ast_json_children(node):
        out.extend(_iter_domain_sensitive_ast_nodes(child))
    return out


def _ast_json_has_domain_sensitive_node(node: Any) -> bool:
    return bool(_iter_domain_sensitive_ast_nodes(node))


def _load_rollout_domain_arrays(run: TrajRun) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        data = np.loadtxt(str(run.csv_path), delimiter=",", skiprows=1)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 2:
        return None
    u = np.asarray(data[:, 0], dtype=np.float64)
    x = np.asarray(data[:, 1], dtype=np.float64)
    if x.size < 2 or u.size != x.size:
        return None
    t0 = float(x[0])
    full_t1 = float(x[-1])
    full_span = abs(float(full_t1) - float(t0))
    span = min(full_span, full_span * 0.25, 5.0)
    if span > 0.0 and span < full_span:
        sign = 1.0 if full_t1 >= t0 else -1.0
        t1 = t0 + sign * span
        eps = max(1.0e-12, 1.0e-12 * abs(t1))
        mask = x <= t1 + eps if sign >= 0.0 else x >= t1 - eps
        if int(mask.sum()) >= 2:
            x = x[mask]
            u = u[mask]
    try:
        du = np.gradient(u, x)
    except Exception:
        du = np.zeros_like(u)
    return x, u, np.asarray(du, dtype=np.float64)


def _eval_ast_json_array(node: Any, *, x: np.ndarray, u: np.ndarray, du: np.ndarray, x_axis: int) -> np.ndarray:
    if node is None:
        return np.ones_like(u, dtype=np.float64)
    if not isinstance(node, Mapping):
        raise TypeError(f"Expected AST node dict, got {type(node).__name__}")
    kind = str(node.get("type", "")).lower()
    if kind == "const":
        return np.full_like(u, float(node.get("value", 0.0)), dtype=np.float64)
    if kind == "atom":
        atom_kind = str(node.get("kind", "")).lower()
        if atom_kind in {"var", "x", "input"}:
            var_idxs = list(node.get("var_idxs", []) or [])
            idx = int(var_idxs[0]) if var_idxs else int(x_axis)
            if idx != int(x_axis):
                raise ValueError(f"Unsupported x-axis index {idx}; expected {x_axis}")
            return np.asarray(x, dtype=np.float64)
        if atom_kind in {"u", "field", "state"}:
            return np.asarray(u, dtype=np.float64)
        if atom_kind in {"du", "d1u", "grad_u"}:
            return np.asarray(du, dtype=np.float64)
        if atom_kind in {"const", "constant"}:
            return np.ones_like(u, dtype=np.float64)
        raise ValueError(f"Unsupported atom kind in domain AST: {atom_kind!r}")
    if kind == "add":
        return _eval_ast_json_array(node.get("left"), x=x, u=u, du=du, x_axis=x_axis) + _eval_ast_json_array(
            node.get("right"), x=x, u=u, du=du, x_axis=x_axis
        )
    if kind == "sub":
        return _eval_ast_json_array(node.get("left"), x=x, u=u, du=du, x_axis=x_axis) - _eval_ast_json_array(
            node.get("right"), x=x, u=u, du=du, x_axis=x_axis
        )
    if kind == "mul":
        return _eval_ast_json_array(node.get("left"), x=x, u=u, du=du, x_axis=x_axis) * _eval_ast_json_array(
            node.get("right"), x=x, u=u, du=du, x_axis=x_axis
        )
    if kind == "div":
        with np.errstate(all="ignore"):
            return _eval_ast_json_array(node.get("left"), x=x, u=u, du=du, x_axis=x_axis) / _eval_ast_json_array(
                node.get("right"), x=x, u=u, du=du, x_axis=x_axis
            )
    if kind == "pow":
        exponent = node.get("exponent", 1.0)
        if isinstance(exponent, Mapping):
            exponent = _eval_ast_json_array(exponent, x=x, u=u, du=du, x_axis=x_axis)
        with np.errstate(all="ignore"):
            return np.power(_eval_ast_json_array(node.get("base"), x=x, u=u, du=du, x_axis=x_axis), exponent)
    if kind == "log":
        with np.errstate(all="ignore"):
            return np.log(_eval_ast_json_array(node.get("arg"), x=x, u=u, du=du, x_axis=x_axis))
    if kind == "sqrt":
        with np.errstate(all="ignore"):
            return np.sqrt(_eval_ast_json_array(node.get("arg"), x=x, u=u, du=du, x_axis=x_axis))
    if kind == "exp":
        with np.errstate(all="ignore"):
            return np.exp(_eval_ast_json_array(node.get("arg"), x=x, u=u, du=du, x_axis=x_axis))
    if kind == "sin":
        return np.sin(_eval_ast_json_array(node.get("arg"), x=x, u=u, du=du, x_axis=x_axis))
    if kind == "cos":
        return np.cos(_eval_ast_json_array(node.get("arg"), x=x, u=u, du=du, x_axis=x_axis))
    if kind == "abs":
        return np.abs(_eval_ast_json_array(node.get("arg"), x=x, u=u, du=du, x_axis=x_axis))
    if kind == "neg":
        return -_eval_ast_json_array(node.get("arg"), x=x, u=u, du=du, x_axis=x_axis)
    raise ValueError(f"Unsupported AST node type in domain AST: {kind!r}")


def _rollout_domain_safety_report(candidate: Mapping[str, Any], probe_runs: Sequence[TrajRun]) -> dict[str, Any]:
    terms = _validation_term_asts(candidate)
    sensitive: list[tuple[int, str, Any]] = []
    for idx, term in enumerate(terms):
        for kind, arg in _iter_domain_sensitive_ast_nodes(term):
            sensitive.append((int(idx), str(kind), arg))
    if not sensitive:
        return {"safe": True, "reason": "no_domain_sensitive_terms", "checks": []}

    payload = _validation_payload_from_candidate(candidate) or {}
    try:
        x_axis = int(payload.get("x_axis", candidate.get("x_axis", 0)))
    except Exception:
        x_axis = 0
    checks: list[dict[str, Any]] = []
    unsafe: list[dict[str, Any]] = []
    for run in list(probe_runs or []):
        arrays = _load_rollout_domain_arrays(run)
        if arrays is None:
            continue
        x, u, du = arrays
        for term_idx, kind, arg in sensitive:
            try:
                vals = np.asarray(_eval_ast_json_array(arg, x=x, u=u, du=du, x_axis=x_axis), dtype=np.float64)
            except Exception as exc:
                checks.append(
                    {
                        "traj_id": str(run.traj_id),
                        "term_index": int(term_idx),
                        "kind": str(kind),
                        "safe": False,
                        "reason": f"domain_eval_error:{exc}",
                    }
                )
                unsafe.append(checks[-1])
                continue
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                row = {
                    "traj_id": str(run.traj_id),
                    "term_index": int(term_idx),
                    "kind": str(kind),
                    "safe": False,
                    "reason": "nonfinite_domain",
                }
                checks.append(row)
                unsafe.append(row)
                continue
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            min_abs = float(np.min(np.abs(finite)))
            row = {
                "traj_id": str(run.traj_id),
                "term_index": int(term_idx),
                "kind": str(kind),
                "min": vmin,
                "max": vmax,
                "min_abs": min_abs,
                "safe": True,
                "reason": "ok",
            }
            if kind == "negative_power":
                crosses_zero = bool(vmin < 0.0 < vmax)
                near_zero = bool(min_abs <= 1.0e-10)
                if crosses_zero or near_zero:
                    row["safe"] = False
                    row["reason"] = "denominator_crosses_zero" if crosses_zero else "denominator_near_zero"
            elif kind == "log":
                if vmin <= 0.0:
                    row["safe"] = False
                    row["reason"] = "log_argument_nonpositive"
            elif kind in {"sqrt", "fractional_power"}:
                if vmin < -1.0e-12:
                    row["safe"] = False
                    row["reason"] = "root_argument_negative"
            checks.append(row)
            if not bool(row["safe"]):
                unsafe.append(row)

    return {
        "safe": not unsafe,
        "reason": "ok" if not unsafe else "rollout_domain_violation",
        "checks": checks,
        "violations": unsafe,
    }


def _ast_json_to_expr(node: Any) -> str:
    if node is None:
        return "1"
    if not isinstance(node, Mapping):
        return str(node)
    kind = str(node.get("type", "")).lower()
    if kind == "const":
        try:
            return f"{float(node.get('value', 0.0)):.6g}"
        except Exception:
            return str(node.get("value", 0.0))
    if kind == "atom":
        atom_kind = str(node.get("kind", "")).lower()
        if atom_kind in {"var", "x", "input"}:
            idxs = list(node.get("var_idxs", []) or [0])
            return f"x{int(idxs[0])}"
        if atom_kind in {"u", "field", "state"}:
            return "u"
        if atom_kind in {"du", "d1u", "grad_u"}:
            return "u_x0"
        return atom_kind or "?"
    if kind == "add":
        return f"({_ast_json_to_expr(node.get('left'))} + {_ast_json_to_expr(node.get('right'))})"
    if kind == "sub":
        return f"({_ast_json_to_expr(node.get('left'))} - {_ast_json_to_expr(node.get('right'))})"
    if kind == "mul":
        return f"({_ast_json_to_expr(node.get('left'))} * {_ast_json_to_expr(node.get('right'))})"
    if kind == "div":
        return f"({_ast_json_to_expr(node.get('left'))} / {_ast_json_to_expr(node.get('right'))})"
    if kind == "pow":
        return f"({_ast_json_to_expr(node.get('base'))} ** {node.get('exponent', 1.0)})"
    if kind in {"log", "exp", "sin", "cos", "sqrt", "abs", "neg"}:
        arg = _ast_json_to_expr(node.get("arg"))
        return f"(-{arg})" if kind == "neg" else f"{kind}({arg})"
    return str(node)


def _ast_json_add_terms(node: Any) -> list[Any]:
    if isinstance(node, Mapping) and str(node.get("type", "")).lower() == "add":
        return [*_ast_json_add_terms(node.get("left")), *_ast_json_add_terms(node.get("right"))]
    return [node]


def _split_numeric_factor(node: Any) -> tuple[float, Any | None]:
    if not isinstance(node, Mapping):
        return 1.0, node
    kind = str(node.get("type", "")).lower()
    if kind == "const":
        return float(node.get("value", 0.0)), None
    if kind == "mul":
        left = node.get("left")
        right = node.get("right")
        if isinstance(left, Mapping) and str(left.get("type", "")).lower() == "const":
            return float(left.get("value", 0.0)), right
        if isinstance(right, Mapping) and str(right.get("type", "")).lower() == "const":
            return float(right.get("value", 0.0)), left
    return 1.0, node


def _make_ast_json_term(coeff: float, expr: Any | None) -> dict[str, Any]:
    coeff_f = float(coeff)
    if expr is None:
        return {"type": "const", "value": coeff_f}
    if abs(coeff_f - 1.0) <= 1.0e-14:
        return dict(expr) if isinstance(expr, Mapping) else expr
    return {
        "type": "mul",
        "left": {"type": "const", "value": coeff_f},
        "right": dict(expr) if isinstance(expr, Mapping) else expr,
    }


def _join_ast_json_add_terms(terms: Sequence[Any]) -> Any | None:
    items = list(terms)
    if not items:
        return None
    out = items[0]
    for term in items[1:]:
        out = {"type": "add", "left": out, "right": term}
    return out


def _pruned_factorized_candidate_variants(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    lane = str(candidate.get("lane", "") or "")
    typed = candidate.get("typed_metadata", None)
    if not lane and isinstance(typed, Mapping):
        lane = str(typed.get("lane", "") or "")
    if lane != "two_block_typed_assembly":
        return []
    payload = _validation_payload_from_candidate(candidate)
    terms_json = _validation_term_asts(candidate)
    if not isinstance(payload, Mapping) or len(terms_json) != 1:
        return []
    terms = _ast_json_add_terms(terms_json[0])
    if len(terms) <= 2:
        return []
    split_terms = [_split_numeric_factor(term) for term in terms]
    coeff_mags = [abs(float(coeff)) for coeff, _expr in split_terms if math.isfinite(float(coeff))]
    scale = max(coeff_mags, default=0.0)
    if scale <= 0.0 or not math.isfinite(scale):
        return []
    keep_terms: list[Any] = []
    dropped: list[dict[str, Any]] = []
    rel_tol = 5.0e-3
    for idx, (coeff, expr) in enumerate(split_terms):
        is_tiny = abs(float(coeff)) <= float(rel_tol) * float(scale)
        if is_tiny:
            dropped.append(
                {
                    "term_index": int(idx),
                    "coefficient": float(coeff),
                    "relative_to_max": float(abs(float(coeff)) / float(scale)),
                    "domain_sensitive": bool(_ast_json_has_domain_sensitive_node(expr)),
                    "expr": _ast_json_to_expr(expr),
                }
            )
            continue
        keep_terms.append(_make_ast_json_term(float(coeff), expr))
    if not dropped or not keep_terms:
        return []
    pruned_ast = _join_ast_json_add_terms(keep_terms)
    if pruned_ast is None:
        return []
    out = dict(candidate)
    out["validation_candidate"] = dict(payload)
    out["validation_candidate"]["term_asts_json"] = [pruned_ast]
    out["canonical_equation"] = f"u_x0x0 + {_ast_json_to_expr(pruned_ast)} = 0"
    out["residual_ast"] = _ast_json_to_expr(pruned_ast)
    out["residual_ast_simplified"] = out["residual_ast"]
    out["assembly_pruned"] = True
    out["parsimony_pruned"] = True
    out["parsimony_pruned_from_rank"] = candidate.get("candidate_rank", candidate.get("shortlist_rank", None))
    out["parsimony_pruned_terms"] = dropped
    try:
        out["symbolic_size_raw"] = max(1, int(candidate.get("symbolic_size_raw", 10**9)) - 3 * len(dropped))
        out["symbolic_size_simplified"] = max(1, int(candidate.get("symbolic_size_simplified", 10**9)) - 3 * len(dropped))
    except Exception:
        pass
    out["candidate_rank"] = int(_candidate_rank(dict(candidate))) + 10_000
    out["shortlist_rank"] = out["candidate_rank"]
    if isinstance(typed, Mapping):
        out_typed = dict(typed)
        out_typed["assembly_pruned"] = True
        out_typed["parsimony_pruned"] = True
        out_typed["parsimony_pruned_terms"] = dropped
        out["typed_metadata"] = out_typed
    return [out]


def _factorized_shortlist_from_candidate(
    candidate: dict[str, Any],
    *,
    max_candidates: int = 0,
) -> list[dict[str, Any]]:
    if not isinstance(candidate, dict) or not candidate:
        return []
    shortlist = list(candidate.get("shortlist", []) or [])
    # Include the parent candidate (when evaluable) ahead of the serialized
    # shortlist rows — see _factorized_search_shortlist_from_candidate.
    candidates = []
    if isinstance(candidate.get("validation_candidate", None), dict):
        parent = dict(candidate)
        parent.pop("shortlist", None)
        parent.setdefault("candidate_rank", -1)
        parent.setdefault("shortlist_rank", -1)
        candidates.append(parent)
    candidates.extend(row for row in shortlist if isinstance(row, dict))
    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        rows = [row, *_pruned_factorized_candidate_variants(row)]
        for cand in rows:
            key = json.dumps(cand.get("validation_candidate", cand.get("canonical_equation", "")), sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(cand)
    candidates = expanded
    if int(max_candidates) > 0:
        candidates = candidates[: int(max_candidates)]
    return candidates


_ROLLOUT_CANDIDATE_METADATA_KEYS = (
    "lane",
    "family",
    "base_mode",
    "source_lane",
    "coeff_expr",
    "probe_rms",
    "probe_mse",
    "symbolic_size_raw",
    "symbolic_size_simplified",
    "projection_kind",
    "projection_support",
    "projection_coeffs",
    "projection_signature",
    "projection_snap_cost",
    "auxiliary_first_line",
    "first_line_certified",
    "assembly_pruned",
    "parsimony_pruned",
    "parsimony_pruned_from_rank",
    "parsimony_pruned_terms",
)


def _copy_rollout_candidate_metadata(out: dict[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    for key in _ROLLOUT_CANDIDATE_METADATA_KEYS:
        if key in candidate and key not in out:
            out[key] = candidate.get(key)
    typed_metadata = candidate.get("typed_metadata", None)
    if isinstance(typed_metadata, Mapping):
        for key in _ROLLOUT_CANDIDATE_METADATA_KEYS:
            if key in typed_metadata and key not in out:
                out[key] = typed_metadata.get(key)
    return out


def _auxiliary_factorized_search_shortlist_from_candidate(
    candidate: dict[str, Any],
    *,
    max_candidates: int = 0,
) -> list[dict[str, Any]]:
    if not isinstance(candidate, dict) or not candidate:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in list(candidate.get("auxiliary_rollout_candidates", []) or []):
        if not isinstance(entry, dict):
            continue
        payload = entry.get("candidate", None) if isinstance(entry.get("candidate", None), dict) else entry
        if not isinstance(payload, dict) or not payload:
            continue
        source_lane = str(entry.get("source_lane", payload.get("source_lane", "")) or "")
        rows = _factorized_search_shortlist_from_candidate(payload, max_candidates=0)
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_out = dict(row)
            if source_lane:
                row_out.setdefault("source_lane", source_lane)
            row_out["auxiliary_rollout_candidate"] = True
            row_out.setdefault("auxiliary_first_line", entry.get("auxiliary_first_line", payload.get("auxiliary_first_line", True)))
            row_out.setdefault("first_line_certified", entry.get("first_line_certified", payload.get("first_line_certified", False)))
            key = json.dumps(
                {
                    "source_lane": row_out.get("source_lane", ""),
                    "canonical_equation": row_out.get("canonical_equation", ""),
                    "candidate_key": row_out.get("candidate_key", ""),
                    "candidate_id": row_out.get("candidate_id", ""),
                    "expr_ast": row_out.get("expr_ast", None),
                    "mapping": row_out.get("mapping", None),
                },
                sort_keys=True,
                default=str,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row_out)
    if int(max_candidates) > 0:
        out = out[: int(max_candidates)]
    return out


def _de_candidate_eval_shortlist_from_meta(
    de_meta: dict[str, Any],
    *,
    max_candidates: int = 0,
) -> list[dict[str, Any]]:
    report = de_meta.get("de_candidate_eval", None) if isinstance(de_meta, dict) else None
    if not isinstance(report, dict) or str(report.get("status", "")) != "OK":
        return []
    rows = [row for row in list(report.get("rollout_shortlist", []) or []) if isinstance(row, dict)]
    usable = [
        row
        for row in rows
        if isinstance(row.get("validation_candidate", row), dict)
        and _de_candidate_eval_rollout_safe(row)
    ]
    usable.sort(
        key=lambda row: (
            float(row.get("pointwise_score", float("inf"))),
            float(row.get("probe_rms", float("inf"))),
            int(row.get("candidate_rank", 10**9)),
        )
    )
    cap = int(max_candidates) if int(max_candidates) > 0 else 4
    return usable[: max(0, cap)]


def _de_candidate_eval_rollout_safe(candidate: Mapping[str, Any]) -> bool:
    payload = candidate.get("validation_candidate", candidate)
    if not isinstance(payload, Mapping):
        return False
    kind = str(payload.get("kind", candidate.get("kind", "")) or "")
    family = str(candidate.get("candidate_family", payload.get("candidate_family", "")) or "")
    is_rational = kind == "assembled_implicit_rational" or family == "implicit_rational"
    if not is_rational:
        return True
    safety = candidate.get("denominator_safety", payload.get("denominator_safety", None))
    return isinstance(safety, Mapping) and bool(safety.get("safe", False))


def _evaluate_de_candidate_eval_rollout(
    candidate: dict[str, Any],
    *,
    probe_runs: Sequence[TrajRun],
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None,
    sim_validate_blowup_factor: float,
    sim_validate_blowup_abs: float,
) -> dict[str, Any]:
    if not _de_candidate_eval_rollout_safe(candidate):
        return {
            "engine": "de_candidate_eval",
            "status": "ERROR",
            "message": "Skipped unsafe implicit-rational DE candidate",
            "traj_scores": [],
            "discovered_order": int(candidate.get("order", -1)),
            "candidate_rank": int(candidate.get("candidate_rank", 10**9)),
            "candidate_family": str(candidate.get("candidate_family", "")),
            "source_rank": candidate.get("source_rank", None),
            "pointwise_score": candidate.get("pointwise_score", None),
            "denominator_safety": candidate.get("denominator_safety", None),
            "canonical_equation": _candidate_canonical_equation(candidate),
        }
    domain_safety = _rollout_domain_safety_report(candidate, probe_runs)
    if not bool(domain_safety.get("safe", False)):
        return {
            "engine": "de_candidate_eval",
            "status": "ERROR",
            "message": "Skipped rollout-domain-unsafe DE candidate before integration",
            "traj_scores": [],
            "discovered_order": int(candidate.get("order", -1)),
            "candidate_rank": int(candidate.get("candidate_rank", 10**9)),
            "candidate_family": str(candidate.get("candidate_family", "")),
            "source_rank": candidate.get("source_rank", None),
            "pointwise_score": candidate.get("pointwise_score", None),
            "denominator_safety": candidate.get("denominator_safety", None),
            "rollout_domain_safety": domain_safety,
            "canonical_equation": _candidate_canonical_equation(candidate),
        }
    payload = dict(candidate.get("validation_candidate", candidate) or {})
    payload.setdefault("canonical_equation", candidate.get("canonical_equation", ""))
    payload.setdefault("order", candidate.get("order", -1))
    payload.setdefault("x_axis", candidate.get("x_axis", 0))
    row = _evaluate_library_candidate_rollout(
        payload,
        probe_runs=probe_runs,
        pass_nrmse=float(pass_nrmse),
        partial_nrmse=float(partial_nrmse),
        sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
        sim_validate_blowup_factor=float(sim_validate_blowup_factor),
        sim_validate_blowup_abs=float(sim_validate_blowup_abs),
    )
    row["engine"] = "de_candidate_eval"
    row["candidate_rank"] = int(candidate.get("candidate_rank", 10**9))
    row["candidate_family"] = str(candidate.get("candidate_family", ""))
    row["source_rank"] = candidate.get("source_rank", None)
    row["pointwise_score"] = candidate.get("pointwise_score", None)
    row["denominator_safety"] = candidate.get("denominator_safety", None)
    row["rollout_domain_safety"] = domain_safety
    row["canonical_equation"] = str(payload.get("canonical_equation", row.get("canonical_equation", "")))
    return row


def validate(
    coeff_map: dict[str, float],
    de_meta: dict,
    gt: GroundTruth,
    param_values: dict[str, float],
) -> tuple[str, str]:
    messages = []
    discovered_order = int(de_meta.get("order", -1))
    if discovered_order != gt.order:
        return "FAIL", f"Wrong order: expected {gt.order}, got {discovered_order}"

    expected = resolve_ground_truth(gt, param_values)

    coeff_errors = {}
    missing_terms = []
    for term, expected_val in expected.items():
        if term not in coeff_map:
            missing_terms.append(term)
            continue
        discovered_val = coeff_map[term]
        abs_err = abs(discovered_val - expected_val)
        rel_err = abs_err / max(abs(expected_val), 1e-6)
        coeff_errors[term] = (expected_val, discovered_val, rel_err)
        if rel_err > gt.coeff_rtol and abs_err > gt.coeff_atol:
            messages.append(
                f"  {term}: expected {expected_val:.6g}, got {discovered_val:.6g} (rel={rel_err:.1%})"
            )

    if missing_terms:
        return "FAIL", f"Missing expected terms: {missing_terms}"

    expected_keys = set(expected.keys())
    decoy_terms = [k for k in coeff_map if k not in expected_keys]
    bad_decoys = [k for k in decoy_terms if abs(coeff_map[k]) > gt.decoy_atol]
    if bad_decoys:
        for k in bad_decoys:
            messages.append(f"  nuisance {k}: {coeff_map[k]:.6g} (tol={gt.decoy_atol})")

    if messages:
        detail = "\n".join(messages)
        return "PARTIAL", f"Coefficient issues:\n{detail}"

    max_rel = 0.0
    for _, (_, _, rel_e) in coeff_errors.items():
        max_rel = max(max_rel, rel_e)
    return "PASS", f"max relative error: {max_rel:.1%}"


def _run_stlsq_engine(
    problem: ProblemDef,
    runs: Sequence[TrajRun],
    *,
    probe_runs: Sequence[TrajRun] | None = None,
    results_dir: Path,
    fast: bool,
    verbose: bool,
    no_sim_validate: bool,
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None = 20.0,
    sim_validate_blowup_factor: float = 100.0,
    sim_validate_blowup_abs: float = 1.0e6,
    de_coe_mode: str = "off",
    use_dims: bool = True,
) -> dict[str, Any]:
    fit_runs = list(runs)
    probe_runs = list(probe_runs or fit_runs)
    csv_paths = [r.csv_path for r in fit_runs]
    cmd = build_run_de_command(
        problem,
        csv_paths,
        results_dir,
        fast=fast,
        rescue=False,
        de_coe_mode=de_coe_mode,
        use_dims=bool(use_dims),
    )
    if verbose:
        print(f"  [sparse] Command: {' '.join(cmd)}")

    log_path = results_dir / f"de{problem.id}_sparse.log"
    proc = _run_command_to_log(cmd, log_path)

    if proc.returncode != 0:
        msg = _format_command_failure_message("run_de.py", proc)
        if verbose:
            print(f"  [sparse] See log: {log_path}")
        return _attach_command_resource_report(
            {"status": "ERROR", "message": msg, "engine": "sparse"},
            proc,
        )

    report_path = results_dir / f"{_derive_run_de_base_filename(csv_paths)}_de.json"
    if not report_path.exists():
        return {
            "status": "ERROR",
            "message": f"Missing JSON: {report_path}",
            "engine": "sparse",
        }

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        de_meta = payload.get("de_discovery", {})
        selected = de_meta.get("selected", {}) or {}
    except Exception as exc:
        return {
            "status": "ERROR",
            "message": f"JSON parse error: {exc}",
            "engine": "sparse",
        }

    out: dict[str, Any] = {
        "engine": "sparse",
        "json_path": str(report_path),
        "command_resource_report": _command_resource_report(proc),
        "selected_engine": str(de_meta.get("selected_engine", "stlsq")),
        "internal_selected_engine": str(de_meta.get("internal_selected_engine", de_meta.get("selected_engine", "stlsq"))),
        "coeff_map": _coeff_map_from_candidate(selected),
        "canonical_equation": _candidate_canonical_equation(
            selected,
            fallback=str(de_meta.get("canonical_equation", "") or ""),
        ),
    }
    _attach_committee_audit_fields(
        out,
        de_meta.get("committee_decision", None),
        internal_selected_engine=str(out.get("internal_selected_engine", "stlsq")),
    )
    if bool(no_sim_validate):
        out["status"] = "UNVERIFIED"
        out["message"] = "Simulation validation disabled"
        return out

    try:
        order, rhs_fn = library_candidate_to_rhs_callable(selected)
    except Exception as exc:
        return {
            "engine": "sparse",
            "status": "ERROR",
            "message": f"Could not build sparse RHS: {exc}",
            "json_path": str(report_path),
            "canonical_equation": out["canonical_equation"],
            "coeff_map": out["coeff_map"],
        }

    status, message, traj_scores = validate_by_simulation(
        probe_runs,
        rhs_fn=rhs_fn,
        order=int(order),
        pass_nrmse=float(pass_nrmse),
        partial_nrmse=float(partial_nrmse),
        traj_time_budget_s=sim_validate_traj_time_budget_s,
        blowup_factor=float(sim_validate_blowup_factor),
        blowup_abs=float(sim_validate_blowup_abs),
    )
    out["status"] = str(status)
    out["message"] = str(message)
    out["traj_scores"] = traj_scores
    out["discovered_order"] = int(order)
    return out


def _evaluate_library_candidate_rollout(
    candidate: dict[str, Any],
    *,
    probe_runs: Sequence[TrajRun],
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None,
    sim_validate_blowup_factor: float,
    sim_validate_blowup_abs: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "engine": "stlsq",
        "canonical_equation": _candidate_canonical_equation(candidate),
        "coeff_map": _coeff_map_from_candidate(candidate),
    }
    try:
        order, rhs_fn = library_candidate_to_rhs_callable(candidate)
    except Exception as exc:
        out["status"] = "ERROR"
        out["message"] = f"Could not build sparse RHS: {exc}"
        out["traj_scores"] = []
        out["discovered_order"] = int(candidate.get("order", -1))
        return out

    status, message, traj_scores = validate_by_simulation(
        probe_runs,
        rhs_fn=rhs_fn,
        order=int(order),
        pass_nrmse=float(pass_nrmse),
        partial_nrmse=float(partial_nrmse),
        traj_time_budget_s=sim_validate_traj_time_budget_s,
        blowup_factor=float(sim_validate_blowup_factor),
        blowup_abs=float(sim_validate_blowup_abs),
    )
    out["status"] = str(status)
    out["message"] = str(message)
    out["traj_scores"] = traj_scores
    out["discovered_order"] = int(order)
    return out


def _evaluate_factorized_search_candidate_rollout(
    candidate: dict[str, Any],
    *,
    probe_runs: Sequence[TrajRun],
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None,
    sim_validate_blowup_factor: float,
    sim_validate_blowup_abs: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "engine": "factorized_search",
        "canonical_equation": str(candidate.get("canonical_equation", "")),
        "candidate_rank": _candidate_rank(candidate),
        "candidate_id": str(candidate.get("candidate_id", "") or ""),
        "candidate_key": str(candidate.get("candidate_key", "") or ""),
    }
    _copy_rollout_candidate_metadata(out, candidate)
    if not _factorized_search_candidate_domain_safe(candidate):
        out["status"] = "ERROR"
        out["message"] = (
            "Skipped structurally unsafe or domain-rejected factorized symbolic search "
            "candidate before rollout"
        )
        out["traj_scores"] = []
        out["discovered_order"] = int(candidate.get("order", -1))
        return out
    try:
        order, rhs_fn = factorized_search_report_to_rhs_callable(candidate)
    except Exception as exc:
        out["status"] = "ERROR"
        out["message"] = f"Could not build factorized symbolic search RHS: {exc}"
        out["traj_scores"] = []
        out["discovered_order"] = int(candidate.get("order", -1))
        return out

    status, message, traj_scores = validate_by_simulation(
        probe_runs,
        rhs_fn=rhs_fn,
        order=int(order),
        pass_nrmse=float(pass_nrmse),
        partial_nrmse=float(partial_nrmse),
        traj_time_budget_s=sim_validate_traj_time_budget_s,
        blowup_factor=float(sim_validate_blowup_factor),
        blowup_abs=float(sim_validate_blowup_abs),
    )
    out["status"] = str(status)
    out["message"] = str(message)
    out["traj_scores"] = traj_scores
    out["discovered_order"] = int(order)
    return out


def _evaluate_factorized_candidate_rollout(
    candidate: dict[str, Any],
    *,
    probe_runs: Sequence[TrajRun],
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_traj_time_budget_s: float | None,
    sim_validate_blowup_factor: float,
    sim_validate_blowup_abs: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "engine": "factorized",
        "canonical_equation": str(candidate.get("canonical_equation", "")),
        "candidate_rank": _candidate_rank(candidate),
        "candidate_id": str(candidate.get("candidate_id", "") or ""),
        "candidate_key": str(candidate.get("candidate_key", "") or ""),
    }
    _copy_rollout_candidate_metadata(out, candidate)
    domain_safety = _rollout_domain_safety_report(candidate, probe_runs)
    out["rollout_domain_safety"] = domain_safety
    if not bool(domain_safety.get("safe", False)):
        out["status"] = "ERROR"
        out["message"] = "Skipped rollout-domain-unsafe factorized candidate before integration"
        out["traj_scores"] = []
        out["discovered_order"] = int(candidate.get("order", -1))
        return out
    try:
        order, rhs_fn = library_candidate_to_rhs_callable(candidate)
    except Exception as exc:
        out["status"] = "ERROR"
        out["message"] = f"Could not build factorized RHS: {exc}"
        out["traj_scores"] = []
        out["discovered_order"] = int(candidate.get("order", -1))
        return out

    status, message, traj_scores = validate_by_simulation(
        probe_runs,
        rhs_fn=rhs_fn,
        order=int(order),
        pass_nrmse=float(pass_nrmse),
        partial_nrmse=float(partial_nrmse),
        traj_time_budget_s=sim_validate_traj_time_budget_s,
        blowup_factor=float(sim_validate_blowup_factor),
        blowup_abs=float(sim_validate_blowup_abs),
    )
    out["status"] = str(status)
    out["message"] = str(message)
    out["traj_scores"] = traj_scores
    out["discovered_order"] = int(order)
    return out


def _safe_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _factorized_de_metrics_from_meta(de_meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(de_meta, Mapping):
        return {}
    fdf_diag = de_meta.get("factorized_de_diagnostics", {})
    if not isinstance(fdf_diag, Mapping):
        fdf_diag = {}
    whole_rhs = fdf_diag.get("whole_rhs_policy", {})
    if not isinstance(whole_rhs, Mapping):
        whole_rhs = {}

    factorized = de_meta.get("factorized_rescue", {})
    factorized_diag = factorized.get("diagnostics", {}) if isinstance(factorized, Mapping) else {}
    if not isinstance(factorized_diag, Mapping):
        factorized_diag = {}
    typed_diag = factorized_diag.get("factorized_de_diagnostics", {})
    if not isinstance(typed_diag, Mapping):
        typed_diag = {}

    rescue = de_meta.get("factorized_search_rescue", {})
    rescue_diag = rescue.get("diagnostics", {}) if isinstance(rescue, Mapping) else {}
    if not isinstance(rescue_diag, Mapping):
        rescue_diag = {}
    attempts = rescue_diag.get("rescue_attempts", [])
    attempts_run_default = len(attempts) if isinstance(attempts, list) else 0

    return {
        "selected_lane": str(fdf_diag.get("selected_lane", de_meta.get("selected_engine", "")) or ""),
        "direct_residual_probe_rms": _safe_float_or_none(fdf_diag.get("direct_residual_probe_rms", None)),
        "direct_residual_attempted": bool(fdf_diag.get("direct_residual_attempted", False)),
        "typed_lanes_policy": str(fdf_diag.get("typed_lanes_policy", "") or ""),
        "typed_lanes_attempted": bool(fdf_diag.get("typed_lanes_attempted", False)),
        "coefficient_dim_mode": str(fdf_diag.get("coefficient_dim_mode", "") or ""),
        "typed_selected_lane": str(typed_diag.get("selected_lane", factorized_diag.get("lane", "")) or ""),
        "typed_selected_family": str(typed_diag.get("selected_family", factorized_diag.get("family", "")) or ""),
        "typed_selected_base_mode": str(
            typed_diag.get("selected_base_mode", factorized_diag.get("base_mode", "")) or ""
        ),
        "typed_probe_rms": _safe_float_or_none(fdf_diag.get("factorized_probe_rms", None)),
        "whole_rhs_probe_rms": _safe_float_or_none(fdf_diag.get("factorized_search_probe_rms", None)),
        "whole_rhs_policy": str(whole_rhs.get("policy", "") or ""),
        "whole_rhs_reason": str(whole_rhs.get("reason", "") or ""),
        "whole_rhs_run": bool(whole_rhs.get("run", False)),
        "whole_rhs_attempted": bool(fdf_diag.get("factorized_search_attempted", False)),
        "whole_rhs_budget_scope": str(whole_rhs.get("budget_scope", "") or ""),
        "whole_rhs_max_attempts": _safe_int_or_none(
            whole_rhs.get("max_attempts", rescue_diag.get("rescue_max_attempts", None))
        ),
        "whole_rhs_attempts_available": int(
            rescue_diag.get("rescue_attempts_available", attempts_run_default) or 0
        ),
        "whole_rhs_attempts_run": int(rescue_diag.get("rescue_attempts_run", attempts_run_default) or 0),
        "whole_rhs_attempts_capped": bool(rescue_diag.get("rescue_attempts_capped", False)),
        "typed_explorer_launches": int(typed_diag.get("typed_explorer_launches", 0) or 0),
        "generic_explorer_launches": int(typed_diag.get("generic_explorer_launches", 0) or 0),
        "family_gate_evaluations": int(typed_diag.get("family_gate_evaluations", 0) or 0),
        "family_gate_passes": int(typed_diag.get("family_gate_passes", 0) or 0),
        "family_gate_skips": int(typed_diag.get("explorer_skipped", 0) or 0),
        "scheduler_coord_candidates_skipped": int(
            typed_diag.get("scheduler_coord_candidates_skipped", 0) or 0
        ),
        "two_block_typed_candidates": int(typed_diag.get("two_block_typed_candidates", 0) or 0),
        "typed_candidate_count": int(typed_diag.get("n_candidates", 0) or 0),
        "typed_shortlist_size": int(typed_diag.get("shortlist_size", 0) or 0),
    }


def _run_hybrid_engine(
    problem: ProblemDef,
    runs: Sequence[TrajRun],
    *,
    probe_runs: Sequence[TrajRun] | None = None,
    results_dir: Path,
    fast: bool,
    verbose: bool,
    no_sim_validate: bool,
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_max_candidates: int = 0,
    sim_validate_traj_time_budget_s: float | None = 20.0,
    sim_validate_blowup_factor: float = 100.0,
    sim_validate_blowup_abs: float = 1.0e6,
    sim_validate_progress: bool = True,
    factorized_search_preset: str | None = None,
    factorized_two_block_shared_coord: str | None = None,
    factorized_de: bool = False,
    de_coe_mode: str = "off",
    de_coe_csr_on_ties: bool = False,
    de_coe_reservoir_scouts: int = 0,
    factorized_de_whole_rhs: str = "auto",
    factorized_de_typed_lanes: str = "never",
    factorized_de_typed_lane_workers: int = 1,
    factorized_search_de_refine_mode: str = "rare_final_polish",
    factorized_search_max_attempts: int | None = None,
    factorized_search_integrate_topk: int | None = None,
    factorized_search_direct_generator_witness_topk: int | None = None,
    use_dims: bool = True,
) -> dict[str, Any]:
    fit_runs = list(runs)
    probe_runs = list(probe_runs or fit_runs)
    csv_paths = [r.csv_path for r in fit_runs]
    engine_label = "factorized_de" if bool(factorized_de) else "hybrid"
    integrate_topk = factorized_search_integrate_topk
    if integrate_topk is None and not bool(no_sim_validate):
        integrate_topk = 0
    cmd = build_run_de_command(
        problem,
        csv_paths,
        results_dir,
        fast=fast,
        rescue=not bool(factorized_de),
        factorized_de=bool(factorized_de),
        factorized_search_preset=factorized_search_preset,
        factorized_two_block_shared_coord=factorized_two_block_shared_coord,
        de_coe_mode=de_coe_mode,
        de_coe_csr_on_ties=bool(de_coe_csr_on_ties),
        de_coe_reservoir_scouts=int(de_coe_reservoir_scouts),
        factorized_de_whole_rhs=factorized_de_whole_rhs,
        factorized_de_typed_lanes=factorized_de_typed_lanes,
        factorized_de_typed_lane_workers=int(factorized_de_typed_lane_workers),
        factorized_search_de_refine_mode=factorized_search_de_refine_mode,
        factorized_search_max_attempts=factorized_search_max_attempts,
        factorized_search_integrate_topk=integrate_topk,
        factorized_search_direct_generator_witness_topk=factorized_search_direct_generator_witness_topk,
        use_dims=bool(use_dims),
    )
    if verbose:
        print(f"  [{engine_label}] Command: {' '.join(cmd)}")

    log_path = results_dir / f"de{problem.id}_{engine_label}.log"
    proc = _run_command_to_log(cmd, log_path)

    if proc.returncode != 0:
        msg = _format_command_failure_message("run_de.py", proc)
        if verbose:
            print(f"  [{engine_label}] See log: {log_path}")
        return _attach_command_resource_report(
            {"status": "ERROR", "message": msg, "engine": engine_label},
            proc,
        )

    report_path = results_dir / f"{_derive_run_de_base_filename(csv_paths)}_de.json"
    if not report_path.exists():
        return {
            "status": "ERROR",
            "message": f"Missing JSON: {report_path}",
            "engine": engine_label,
        }

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        de_meta = payload.get("de_discovery", {})
    except Exception as exc:
        return {
            "status": "ERROR",
            "message": f"JSON parse error: {exc}",
            "engine": engine_label,
        }

    internal_selected_engine = str(
        de_meta.get(
            "internal_selected_engine",
            de_meta.get("selected_engine", "factorized_search" if bool(factorized_de) else "stlsq"),
        )
    )
    first_line = de_meta.get("first_line", {}) or {}
    factorized_candidate = de_meta.get("factorized_rescue", {}) or {}
    rescue_candidate = de_meta.get("factorized_search_rescue", {}) or {}
    selected = de_meta.get("selected", {}) or {}
    factorized_de_metrics = (
        _factorized_de_metrics_from_meta(de_meta) if bool(factorized_de) else {}
    )
    selected_canonical = _candidate_canonical_equation(
        selected,
        fallback=str(de_meta.get("canonical_equation", "") or ""),
    )

    first_line_status = "NONE" if bool(factorized_de) else "UNVERIFIED"
    first_line_message = (
        "STLSQ disabled by --factorized-de" if bool(factorized_de) else "Simulation validation disabled"
    )
    first_line_coeff_map: dict[str, float] = (
        {} if bool(factorized_de) else _coeff_map_from_candidate(first_line) if isinstance(first_line, dict) else {}
    )
    first_line_scores: list[dict[str, Any]] = []
    first_line_eval: dict[str, Any] | None = None
    factorized_eval: dict[str, Any] | None = None
    factorized_evals: list[dict[str, Any]] = []
    direct_search_eval: dict[str, Any] | None = None
    direct_search_evals: list[dict[str, Any]] = []
    rescue_eval: dict[str, Any] | None = None
    rescue_evals: list[dict[str, Any]] = []
    de_candidate_eval_rows: list[dict[str, Any]] = []
    de_candidate_eval_evals: list[dict[str, Any]] = []
    de_candidate_eval_best: dict[str, Any] | None = None
    factorized_shortlist = _factorized_shortlist_from_candidate(
        factorized_candidate,
        max_candidates=int(sim_validate_max_candidates),
    )
    direct_search_shortlist: list[dict[str, Any]] = _auxiliary_factorized_search_shortlist_from_candidate(
        factorized_candidate,
        max_candidates=int(sim_validate_max_candidates),
    )
    selected_factorized_de_lane = str(factorized_de_metrics.get("selected_lane", "") or "")
    selected_first_line_fss_lane = selected_factorized_de_lane in {
        "direct_residual_fss",
        "regularized_implicit_residual",
    }
    if bool(factorized_de) and selected_first_line_fss_lane:
        selected_direct_shortlist = _factorized_search_shortlist_from_candidate(
            selected,
            max_candidates=int(sim_validate_max_candidates),
        )
        direct_search_shortlist = [*selected_direct_shortlist, *direct_search_shortlist]
        if int(sim_validate_max_candidates) > 0:
            direct_search_shortlist = direct_search_shortlist[: int(sim_validate_max_candidates)]
    rescue_shortlist = _factorized_search_shortlist_from_candidate(
        rescue_candidate,
        max_candidates=int(sim_validate_max_candidates),
    )
    de_candidate_eval_rows = _de_candidate_eval_shortlist_from_meta(
        de_meta,
        max_candidates=int(sim_validate_max_candidates),
    )
    if (not bool(factorized_de)) and (not bool(no_sim_validate)) and isinstance(first_line, dict) and first_line:
        first_line_eval = _evaluate_library_candidate_rollout(
            first_line,
            probe_runs=probe_runs,
            pass_nrmse=float(pass_nrmse),
            partial_nrmse=float(partial_nrmse),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
        )
        first_line_status = str(first_line_eval.get("status", "UNVERIFIED"))
        first_line_message = str(first_line_eval.get("message", first_line_message))
        first_line_scores = list(first_line_eval.get("traj_scores", []) or [])
    if not bool(no_sim_validate) and factorized_shortlist:
        if bool(sim_validate_progress) and len(factorized_shortlist) > 1:
            print(f"  [{engine_label} sim-validate] Factorized shortlist candidates={len(factorized_shortlist)}")
        for candidate in factorized_shortlist:
            factorized_evals.append(
                _evaluate_factorized_candidate_rollout(
                    candidate,
                    probe_runs=probe_runs,
                    pass_nrmse=float(pass_nrmse),
                    partial_nrmse=float(partial_nrmse),
                    sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
                    sim_validate_blowup_factor=float(sim_validate_blowup_factor),
                    sim_validate_blowup_abs=float(sim_validate_blowup_abs),
                )
        )
        factorized_eval = _choose_best_same_engine_rollout("factorized", factorized_evals)
        factorized_eval["source_lane"] = "factorized"
    if not bool(no_sim_validate) and direct_search_shortlist:
        if bool(sim_validate_progress) and len(direct_search_shortlist) > 1:
            print(
                f"  [{engine_label} sim-validate] first-line FSS shortlist "
                f"candidates={len(direct_search_shortlist)}"
            )
        for candidate in direct_search_shortlist:
            eval_row = _evaluate_factorized_search_candidate_rollout(
                candidate,
                probe_runs=probe_runs,
                pass_nrmse=float(pass_nrmse),
                partial_nrmse=float(partial_nrmse),
                sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
                sim_validate_blowup_factor=float(sim_validate_blowup_factor),
                sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            )
            eval_row["source_lane"] = str(
                candidate.get("source_lane", selected_factorized_de_lane or "direct_residual_fss")
                or "direct_residual_fss"
            )
            direct_search_evals.append(eval_row)
        direct_search_eval = _choose_best_same_engine_rollout("factorized_search", direct_search_evals)
        direct_search_eval.setdefault("source_lane", selected_factorized_de_lane or "direct_residual_fss")
    if not bool(no_sim_validate) and rescue_shortlist:
        if bool(sim_validate_progress) and len(rescue_shortlist) > 1:
            print(f"  [{engine_label} sim-validate] factorized symbolic search shortlist candidates={len(rescue_shortlist)}")
        for candidate in rescue_shortlist:
            rescue_evals.append(
                _evaluate_factorized_search_candidate_rollout(
                    candidate,
                    probe_runs=probe_runs,
                    pass_nrmse=float(pass_nrmse),
                    partial_nrmse=float(partial_nrmse),
                    sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
                    sim_validate_blowup_factor=float(sim_validate_blowup_factor),
                    sim_validate_blowup_abs=float(sim_validate_blowup_abs),
                )
            )
        rescue_eval = _choose_best_same_engine_rollout("factorized_search", rescue_evals)
        rescue_eval["source_lane"] = "factorized_search"
    if not bool(no_sim_validate) and de_candidate_eval_rows:
        if bool(sim_validate_progress) and len(de_candidate_eval_rows) > 1:
            print(f"  [{engine_label} sim-validate] DE assembled candidates={len(de_candidate_eval_rows)}")
        for candidate in de_candidate_eval_rows:
            de_candidate_eval_evals.append(
                _evaluate_de_candidate_eval_rollout(
                    candidate,
                    probe_runs=probe_runs,
                    pass_nrmse=float(pass_nrmse),
                    partial_nrmse=float(partial_nrmse),
                    sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
                    sim_validate_blowup_factor=float(sim_validate_blowup_factor),
                    sim_validate_blowup_abs=float(sim_validate_blowup_abs),
                )
            )
        de_candidate_eval_best = _choose_best_same_engine_rollout("de_candidate_eval", de_candidate_eval_evals)

    out: dict[str, Any] = {
        "engine": engine_label,
        "stlsq_free": bool(factorized_de),
        "json_path": str(report_path),
        "command_resource_report": _command_resource_report(proc),
        "selected_engine": str(internal_selected_engine),
        "internal_selected_engine": str(internal_selected_engine),
        "canonical_equation": selected_canonical,
        "first_line_status": str(first_line_status),
        "first_line_message": str(first_line_message),
        "first_line_coeff_map": first_line_coeff_map,
        "first_line_traj_scores": first_line_scores,
        "rescue_triggered": bool(de_meta.get("rescue_triggered", False)),
        "rescue_attempted": bool(de_meta.get("rescue_attempted", False)),
        "factorized_shortlist_size": int(len(factorized_shortlist)),
        "factorized_validated_candidates": int(len(factorized_evals)),
        "direct_residual_shortlist_size": int(len(direct_search_shortlist)),
        "direct_residual_validated_candidates": int(len(direct_search_evals)),
        "factorized_search_shortlist_size": int(len(rescue_shortlist)),
        "factorized_search_validated_candidates": int(len(rescue_evals)),
        "de_candidate_eval_shortlist_size": int(len(de_candidate_eval_rows)),
        "de_candidate_eval_validated_candidates": int(len(de_candidate_eval_evals)),
        "factorized_de_metrics": factorized_de_metrics,
    }
    if factorized_de_metrics:
        out.update(
            {
                "selected_lane": factorized_de_metrics.get("selected_lane", ""),
                "direct_residual_probe_rms": factorized_de_metrics.get("direct_residual_probe_rms", None),
                "direct_residual_attempted": bool(
                    factorized_de_metrics.get("direct_residual_attempted", False)
                ),
                "typed_lanes_policy": factorized_de_metrics.get("typed_lanes_policy", ""),
                "typed_lanes_attempted": bool(factorized_de_metrics.get("typed_lanes_attempted", False)),
                "coefficient_dim_mode": factorized_de_metrics.get("coefficient_dim_mode", ""),
                "typed_selected_lane": factorized_de_metrics.get("typed_selected_lane", ""),
                "whole_rhs_attempted": bool(factorized_de_metrics.get("whole_rhs_attempted", False)),
                "whole_rhs_reason": factorized_de_metrics.get("whole_rhs_reason", ""),
                "whole_rhs_attempts_run": int(factorized_de_metrics.get("whole_rhs_attempts_run", 0) or 0),
                "family_gate_skips": int(factorized_de_metrics.get("family_gate_skips", 0) or 0),
                "typed_explorer_launches": int(
                    factorized_de_metrics.get("typed_explorer_launches", 0) or 0
                ),
            }
        )
    if factorized_eval is not None:
        out["factorized_status"] = str(factorized_eval.get("status", "ERROR"))
        out["factorized_message"] = str(factorized_eval.get("message", ""))
        out["factorized_traj_scores"] = list(factorized_eval.get("traj_scores", []) or [])
        out["factorized_selected_shortlist_rank"] = int(factorized_eval.get("candidate_rank", 10**9))
    if direct_search_eval is not None:
        out["direct_residual_status"] = str(direct_search_eval.get("status", "ERROR"))
        out["direct_residual_message"] = str(direct_search_eval.get("message", ""))
        out["direct_residual_traj_scores"] = list(direct_search_eval.get("traj_scores", []) or [])
        out["direct_residual_selected_shortlist_rank"] = int(direct_search_eval.get("candidate_rank", 10**9))
    if rescue_eval is not None:
        out["factorized_search_status"] = str(rescue_eval.get("status", "ERROR"))
        out["factorized_search_message"] = str(rescue_eval.get("message", ""))
        out["factorized_search_traj_scores"] = list(rescue_eval.get("traj_scores", []) or [])
        out["factorized_search_selected_shortlist_rank"] = int(rescue_eval.get("candidate_rank", 10**9))
    if de_candidate_eval_best is not None:
        out["de_candidate_eval_status"] = str(de_candidate_eval_best.get("status", "ERROR"))
        out["de_candidate_eval_message"] = str(de_candidate_eval_best.get("message", ""))
        out["de_candidate_eval_traj_scores"] = list(de_candidate_eval_best.get("traj_scores", []) or [])
        out["de_candidate_eval_selected_shortlist_rank"] = int(de_candidate_eval_best.get("candidate_rank", 10**9))

    if bool(no_sim_validate):
        out["selection_mode"] = "internal_probe_rms"
        out["status"] = "UNVERIFIED"
        out["message"] = "Simulation validation disabled"
        out["rescued_additional"] = False
        _attach_committee_audit_fields(
            out,
            de_meta.get("committee_decision", None),
            internal_selected_engine=str(internal_selected_engine),
        )
        if str(internal_selected_engine) == "stlsq":
            out["coeff_map"] = _coeff_map_from_candidate(selected)
        return out

    candidates: list[tuple[str, dict[str, Any]]] = []
    if first_line_eval is not None:
        candidates.append(("stlsq", first_line_eval))
    if factorized_eval is not None:
        candidates.append(("factorized", factorized_eval))
    if direct_search_eval is not None:
        candidates.append(("factorized_search", direct_search_eval))
    if rescue_eval is not None:
        candidates.append(("factorized_search", rescue_eval))
    if de_candidate_eval_best is not None:
        candidates.append(("de_candidate_eval", de_candidate_eval_best))
    coe_mode = str(de_coe_mode or "off").strip().lower()
    coe_active = coe_mode in {"audit", "adjudicate", "reservoir"}
    coe_selects = coe_mode in {"adjudicate", "reservoir"}
    committee_proposal_slate = [row for row in list(de_meta.get("proposal_slate", []) or []) if isinstance(row, dict)]
    if coe_active and (coe_mode == "reservoir" or int(de_coe_reservoir_scouts) > 0):
        scout_slates, scout_diag = _run_reservoir_scout_reports(
            problem,
            fit_runs,
            results_dir=results_dir,
            fast=fast,
            verbose=verbose,
            max_scouts=int(de_coe_reservoir_scouts),
            factorized_search_preset=factorized_search_preset,
            use_dims=bool(use_dims),
        )
        merged_slate, merge_diag = _merge_reservoir_scout_slates(
            committee_proposal_slate,
            scout_slates,
        )
        committee_proposal_slate = merged_slate
        max_scout_candidates = min(6, max(0, 2 * int(de_coe_reservoir_scouts)))
        candidates, eval_diag = _evaluate_reservoir_scout_proposals(
            committee_proposal_slate,
            candidates,
            probe_runs=probe_runs,
            pass_nrmse=float(pass_nrmse),
            partial_nrmse=float(partial_nrmse),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            max_candidates=max_scout_candidates,
        )
        out["committee_reservoir"] = {
            "enabled": True,
            "mode": coe_mode,
            "scouts": scout_diag,
            "merge": merge_diag,
            "validation": eval_diag,
        }
        out["reservoir_validated_candidates"] = int(eval_diag.get("validated", 0))
    committee_decision = None
    if coe_active and candidates:
        committee_decision = run_de_committee_audit(
            committee_proposal_slate,
            rollout_candidates=[row for _, row in candidates],
            selected_engine=str(internal_selected_engine),
            config={
                "mode": coe_mode,
                "source": "benchmark",
                "engine": engine_label,
                "reservoir_scouts_requested": int(de_coe_reservoir_scouts),
            },
            run_compile_domain=False,
        ).to_dict()
        _attach_committee_audit_fields(
            out,
            committee_decision,
            internal_selected_engine=str(internal_selected_engine),
        )
    if not candidates:
        out["status"] = "ERROR"
        out["message"] = (
            "No validated factorized DE candidates available"
            if bool(factorized_de)
            else "No validated hybrid candidates available"
        )
        out["rescued_additional"] = False
        return out

    selection_candidates = list(candidates)
    if bool(de_coe_csr_on_ties):
        csr_candidates, csr_diag = _maybe_run_csr_on_tied_factorized_search(
            committee_decision=committee_decision,
            current_candidates=candidates,
            factorized_search_shortlist=[*direct_search_shortlist, *rescue_shortlist],
            fit_runs=fit_runs,
            probe_runs=probe_runs,
            pass_nrmse=float(pass_nrmse),
            partial_nrmse=float(partial_nrmse),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            enabled=bool(coe_active),
        )
        out["committee_csr"] = csr_diag
        out["factorized_search_csr_validated_candidates"] = int(csr_diag.get("validated", 0))
        if int(csr_diag.get("validated", 0)) > 0:
            committee_decision = run_de_committee_audit(
                committee_proposal_slate,
                rollout_candidates=[row for _, row in csr_candidates],
                selected_engine=str(internal_selected_engine),
                config={
                    "mode": coe_mode,
                    "source": "benchmark",
                    "engine": engine_label,
                    "csr_after_tie": True,
                    "reservoir_scouts_requested": int(de_coe_reservoir_scouts),
                },
                run_compile_domain=False,
            ).to_dict()
            _attach_committee_audit_fields(
                out,
                committee_decision,
                internal_selected_engine=str(internal_selected_engine),
            )
            if coe_selects:
                selection_candidates = list(csr_candidates)

    committee_choice = None
    if coe_selects:
        committee_choice = _committee_selected_rollout_candidate(committee_decision, selection_candidates)
    if committee_choice is not None:
        chosen_engine, chosen_eval = committee_choice
        out["selection_mode"] = "committee_adjudicate" if coe_mode == "adjudicate" else "committee_reservoir"
        out["committee_adjudicated"] = True
    else:
        chosen_engine, chosen_eval = _choose_rollout_candidate(
            selection_candidates,
            fallback_engine=str(internal_selected_engine),
        )
        out["selection_mode"] = "rollout_nrmse"
        out["committee_adjudicated"] = False
        if coe_selects:
            out["committee_adjudication_fallback"] = True
    out["selected_engine"] = str(chosen_engine)
    out["internal_selected_engine_mismatch"] = bool(str(chosen_engine) != str(internal_selected_engine))
    out["status"] = str(chosen_eval.get("status", "ERROR"))
    out["message"] = str(chosen_eval.get("message", ""))
    out["traj_scores"] = list(chosen_eval.get("traj_scores", []) or [])
    out["discovered_order"] = int(chosen_eval.get("discovered_order", -1))
    out["canonical_equation"] = str(chosen_eval.get("canonical_equation", selected_canonical))
    out["selected_candidate_id"] = str(chosen_eval.get("candidate_id", "") or "")
    out["selected_candidate_key"] = str(chosen_eval.get("candidate_key", "") or "")
    out["internal_selected_candidate_id"] = ""
    out["internal_selected_candidate_key"] = ""
    out["internal_selected_candidate_id_mismatch"] = False
    if str(chosen_engine) == "stlsq":
        out["coeff_map"] = dict(chosen_eval.get("coeff_map", {}))
        out["internal_selected_candidate_id"] = ""
        out["internal_selected_candidate_key"] = ""
    elif str(chosen_engine) == "de_candidate_eval":
        out["selected_shortlist_rank"] = int(chosen_eval.get("candidate_rank", 10**9))
        out["selected_candidate_family"] = str(chosen_eval.get("candidate_family", ""))
        out["internal_selected_shortlist_rank"] = None
        out["internal_selected_shortlist_rank_mismatch"] = False
        out["internal_selected_candidate_id"] = ""
        out["internal_selected_candidate_key"] = ""
    else:
        out["selected_shortlist_rank"] = int(chosen_eval.get("candidate_rank", 10**9))
        if str(chosen_engine) == "factorized":
            internal_rank = factorized_candidate.get("internal_selected_shortlist_rank", None)
            if internal_rank is None:
                internal_rank = 0 if factorized_shortlist else None
            internal_shortlist = factorized_shortlist
        elif str(chosen_eval.get("source_lane", "")) in {"direct_residual_fss", "regularized_implicit_residual"}:
            internal_rank = selected.get("internal_selected_shortlist_rank", None)
            if internal_rank is None:
                internal_rank = 0 if direct_search_shortlist else None
            internal_shortlist = direct_search_shortlist
        else:
            internal_rank = selected.get("internal_selected_shortlist_rank", None)
            if internal_rank is None:
                internal_rank = 0 if rescue_shortlist else None
            internal_shortlist = rescue_shortlist
        out["internal_selected_shortlist_rank"] = None if internal_rank is None else int(internal_rank)
        internal_candidate = _shortlist_candidate_by_rank(internal_shortlist, internal_rank)
        out["internal_selected_candidate_id"] = (
            "" if internal_candidate is None else str(internal_candidate.get("candidate_id", "") or "")
        )
        out["internal_selected_candidate_key"] = (
            "" if internal_candidate is None else str(internal_candidate.get("candidate_key", "") or "")
        )
        out["internal_selected_shortlist_rank_mismatch"] = bool(
            internal_shortlist
            and out["internal_selected_shortlist_rank"] is not None
            and int(chosen_eval.get("candidate_rank", 10**9)) != int(out["internal_selected_shortlist_rank"])
        )
        out["internal_selected_candidate_id_mismatch"] = bool(
            out["selected_candidate_id"]
            and out["internal_selected_candidate_id"]
            and out["selected_candidate_id"] != out["internal_selected_candidate_id"]
        )
    out["rescued_additional"] = bool(
        (not bool(factorized_de))
        and str(chosen_engine) in ("factorized", "factorized_search", "de_candidate_eval")
        and str(out["status"]) == "PASS"
        and str(first_line_status) != "PASS"
    )

    return out


def _run_clean_fallback_hybrid_engine(
    problem: ProblemDef,
    runs: Sequence[TrajRun],
    *,
    probe_runs: Sequence[TrajRun] | None = None,
    results_dir: Path,
    fast: bool,
    verbose: bool,
    no_sim_validate: bool,
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_max_candidates: int = 0,
    sim_validate_traj_time_budget_s: float | None = 20.0,
    sim_validate_blowup_factor: float = 100.0,
    sim_validate_blowup_abs: float = 1.0e6,
    sim_validate_progress: bool = True,
    factorized_search_preset: str | None = None,
    factorized_two_block_shared_coord: str | None = None,
    de_coe_mode: str = "off",
    de_coe_csr_on_ties: bool = False,
    de_coe_reservoir_scouts: int = 0,
    factorized_de_whole_rhs: str = "auto",
    factorized_de_typed_lanes: str = "never",
    factorized_de_typed_lane_workers: int = 1,
    factorized_search_de_refine_mode: str = "rare_final_polish",
    factorized_search_max_attempts: int | None = None,
    factorized_search_integrate_topk: int | None = None,
    factorized_search_direct_generator_witness_topk: int | None = None,
    use_dims: bool = True,
) -> dict[str, Any]:
    """Run sparse first, then a clean factorized_de fallback only if sparse fails.

    The fallback is launched as ``--factorized-de``, so it has no sparse/STLSQ
    primary result to use as a residual base or order constraint.
    """

    fit_runs = list(runs)
    probe_runs = list(probe_runs or fit_runs)
    sparse = _run_stlsq_engine(
        problem,
        fit_runs,
        probe_runs=probe_runs,
        results_dir=results_dir,
        fast=fast,
        verbose=verbose,
        no_sim_validate=no_sim_validate,
        pass_nrmse=pass_nrmse,
        partial_nrmse=partial_nrmse,
        sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
        sim_validate_blowup_factor=float(sim_validate_blowup_factor),
        sim_validate_blowup_abs=float(sim_validate_blowup_abs),
        de_coe_mode=de_coe_mode,
        use_dims=bool(use_dims),
    )

    first_line_status = str(sparse.get("status", "ERROR"))
    first_line_message = str(sparse.get("message", ""))

    def _base_out(selected: dict[str, Any]) -> dict[str, Any]:
        out = dict(selected)
        out["engine"] = "hybrid"
        out["hybrid_policy"] = "stlsq_then_clean_factorized_de"
        out["stlsq_free"] = False
        out["first_line_status"] = first_line_status
        out["first_line_message"] = first_line_message
        out["first_line_coeff_map"] = dict(sparse.get("coeff_map", {}) or {})
        out["first_line_traj_scores"] = list(sparse.get("traj_scores", []) or [])
        out["first_line_engine"] = "stlsq"
        out["first_line_json_path"] = str(sparse.get("json_path", "") or "")
        out["clean_fallback_attempted"] = False
        out["clean_fallback_engine"] = "factorized_de"
        out["rescued_additional"] = False
        return out

    if first_line_status == "PASS" or bool(no_sim_validate):
        out = _base_out(sparse)
        out["clean_fallback_reason"] = "first_line_pass" if first_line_status == "PASS" else "simulation_validation_disabled"
        return out

    sparse_json_path = Path(str(sparse.get("json_path", "") or ""))
    if sparse_json_path.is_file():
        snapshot_path = sparse_json_path.with_name(f"{sparse_json_path.stem}.sparse_first_line{sparse_json_path.suffix}")
        shutil.copy2(sparse_json_path, snapshot_path)
        sparse = dict(sparse)
        sparse["json_path"] = str(snapshot_path)

    fallback = _run_hybrid_engine(
        problem,
        fit_runs,
        probe_runs=probe_runs,
        results_dir=results_dir,
        fast=fast,
        verbose=verbose,
        no_sim_validate=no_sim_validate,
        pass_nrmse=pass_nrmse,
        partial_nrmse=partial_nrmse,
        sim_validate_max_candidates=int(sim_validate_max_candidates),
        sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
        sim_validate_blowup_factor=float(sim_validate_blowup_factor),
        sim_validate_blowup_abs=float(sim_validate_blowup_abs),
        sim_validate_progress=bool(sim_validate_progress),
        factorized_search_preset=factorized_search_preset,
        factorized_two_block_shared_coord=factorized_two_block_shared_coord,
        factorized_de=True,
        de_coe_mode=de_coe_mode,
        de_coe_csr_on_ties=bool(de_coe_csr_on_ties),
        de_coe_reservoir_scouts=int(de_coe_reservoir_scouts),
        factorized_de_whole_rhs=factorized_de_whole_rhs,
        factorized_de_typed_lanes=factorized_de_typed_lanes,
        factorized_de_typed_lane_workers=int(factorized_de_typed_lane_workers),
        factorized_search_de_refine_mode=factorized_search_de_refine_mode,
        factorized_search_max_attempts=factorized_search_max_attempts,
        factorized_search_integrate_topk=factorized_search_integrate_topk,
        factorized_search_direct_generator_witness_topk=factorized_search_direct_generator_witness_topk,
        use_dims=bool(use_dims),
    )

    fallback_engine = str(fallback.get("selected_engine", "factorized_search") or "factorized_search")
    # A non-PASS STLSQ result should not short-circuit hybrid, but it is still
    # a validated rollout candidate. Keep it as the floor: the clean fallback
    # must beat the first-line rollout before it can replace it.
    chosen_name, chosen = _choose_rollout_candidate(
        [
            ("stlsq", sparse),
            (fallback_engine, fallback),
        ],
        fallback_engine=fallback_engine,
    )
    out = _base_out(chosen)
    out["clean_fallback_attempted"] = True
    out["clean_fallback_reason"] = f"first_line_{first_line_status.lower() or 'unknown'}"
    out["clean_fallback_status"] = str(fallback.get("status", "ERROR"))
    out["clean_fallback_message"] = str(fallback.get("message", ""))
    out["clean_fallback_json_path"] = str(fallback.get("json_path", "") or "")
    out["clean_fallback_selected_engine"] = fallback_engine
    out["clean_fallback_result"] = fallback
    fallback_resource = fallback.get("command_resource_report", None)
    if isinstance(fallback_resource, dict):
        out["clean_fallback_command_resource_report"] = dict(fallback_resource)
        if fallback_resource.get("peak_tree_rss_mb", None) is not None:
            out["clean_fallback_peak_tree_rss_mb"] = float(fallback_resource["peak_tree_rss_mb"])
        if fallback_resource.get("last_tree_rss_mb", None) is not None:
            out["clean_fallback_last_tree_rss_mb"] = float(fallback_resource["last_tree_rss_mb"])
    if bool(fallback.get("resource_failure_suspected", False)):
        out["clean_fallback_resource_failure_suspected"] = True
    if bool(fallback.get("command_killed_by_signal", False)):
        out["clean_fallback_killed_by_signal"] = True
        out["clean_fallback_signal"] = str(fallback.get("command_signal", "") or "")
    out["sparse_first_line_result"] = sparse
    out["hybrid_rollout_choice"] = str(chosen_name)
    out["clean_fallback_beats_first_line"] = bool(str(chosen_name) != "stlsq")
    out["selected_engine"] = "stlsq" if str(chosen_name) == "stlsq" else fallback_engine
    out["selection_mode"] = "stlsq_first_clean_factorized_de_fallback"
    out["rescued_additional"] = bool(
        str(chosen_name) != "stlsq"
        and str(out.get("status", "")) == "PASS"
        and first_line_status != "PASS"
    )
    return out


def _run_factorized_search_only_engine(
    problem: ProblemDef,
    runs: Sequence[TrajRun],
    *,
    probe_runs: Sequence[TrajRun] | None = None,
    results_dir: Path,
    fast: bool,
    verbose: bool,
    no_sim_validate: bool,
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_max_candidates: int = 0,
    sim_validate_traj_time_budget_s: float | None = 20.0,
    sim_validate_blowup_factor: float = 100.0,
    sim_validate_blowup_abs: float = 1.0e6,
    sim_validate_progress: bool = True,
    factorized_search_preset: str | None = None,
    de_coe_mode: str = "off",
    de_coe_csr_on_ties: bool = False,
    de_coe_reservoir_scouts: int = 0,
    factorized_search_de_refine_mode: str = "rare_final_polish",
    factorized_search_max_attempts: int | None = None,
    factorized_search_integrate_topk: int | None = None,
    factorized_search_direct_generator_witness_topk: int | None = None,
    use_dims: bool = True,
) -> dict[str, Any]:
    fit_runs = list(runs)
    probe_runs = list(probe_runs or fit_runs)
    csv_paths = [r.csv_path for r in fit_runs]
    integrate_topk = factorized_search_integrate_topk
    if integrate_topk is None and not bool(no_sim_validate):
        integrate_topk = 0
    cmd = build_run_de_command(
        problem,
        csv_paths,
        results_dir,
        fast=fast,
        rescue=False,
        factorized_search_only=True,
        factorized_search_preset=factorized_search_preset,
        de_coe_mode=de_coe_mode,
        de_coe_csr_on_ties=bool(de_coe_csr_on_ties),
        de_coe_reservoir_scouts=int(de_coe_reservoir_scouts),
        factorized_search_de_refine_mode=factorized_search_de_refine_mode,
        factorized_search_max_attempts=factorized_search_max_attempts,
        factorized_search_integrate_topk=integrate_topk,
        factorized_search_direct_generator_witness_topk=factorized_search_direct_generator_witness_topk,
        use_dims=bool(use_dims),
    )
    if verbose:
        print(f"  [factorized_search_only] Command: {' '.join(cmd)}")

    log_path = results_dir / f"de{problem.id}_factorized_search_only.log"
    proc = _run_command_to_log(cmd, log_path)

    if proc.returncode != 0:
        msg = _format_command_failure_message("run_de.py", proc)
        if verbose:
            print(f"  [factorized_search_only] See log: {log_path}")
        return _attach_command_resource_report(
            {"status": "ERROR", "message": msg, "engine": "factorized_search_only"},
            proc,
        )

    report_path = results_dir / f"{_derive_run_de_base_filename(csv_paths)}_de.json"
    if not report_path.exists():
        return {
            "status": "ERROR",
            "message": f"Missing JSON: {report_path}",
            "engine": "factorized_search_only",
        }

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        de_meta = payload.get("de_discovery", {})
        selected = de_meta.get("selected", {}) or {}
    except Exception as exc:
        return {
            "status": "ERROR",
            "message": f"JSON parse error: {exc}",
            "engine": "factorized_search_only",
        }

    out: dict[str, Any] = {
        "engine": "factorized_search_only",
        "json_path": str(report_path),
        "command_resource_report": _command_resource_report(proc),
        "selected_engine": "factorized_search",
        "internal_selected_engine": str(
            de_meta.get("internal_selected_engine", de_meta.get("selected_engine", "factorized_search"))
        ),
        "canonical_equation": str(selected.get("canonical_equation", de_meta.get("canonical_equation", ""))),
        "rescue_triggered": bool(de_meta.get("rescue_triggered", False)),
        "rescue_attempted": bool(de_meta.get("rescue_attempted", False)),
    }
    if bool(no_sim_validate):
        out["selection_mode"] = "internal_probe_rms"
        out["status"] = "UNVERIFIED"
        out["message"] = "Simulation validation disabled"
        _attach_committee_audit_fields(
            out,
            de_meta.get("committee_decision", None),
            internal_selected_engine=str(out.get("internal_selected_engine", "factorized_search")),
        )
        return out

    shortlist = _factorized_search_shortlist_from_candidate(
        selected,
        max_candidates=int(sim_validate_max_candidates),
    )
    de_candidate_eval_rows = _de_candidate_eval_shortlist_from_meta(
        de_meta,
        max_candidates=int(sim_validate_max_candidates),
    )
    out["factorized_search_shortlist_size"] = int(len(shortlist))
    out["de_candidate_eval_shortlist_size"] = int(len(de_candidate_eval_rows))
    if not shortlist:
        out["selection_mode"] = "rollout_nrmse"
        out["factorized_search_validated_candidates"] = 0
        if not de_candidate_eval_rows:
            out["de_candidate_eval_validated_candidates"] = 0
            out["status"] = "ERROR"
            out["message"] = "No factorized symbolic search candidates available for rollout validation"
            return out
    if bool(sim_validate_progress) and len(shortlist) > 1:
        print(f"  [factorized_search_only sim-validate] shortlist candidates={len(shortlist)}")
    eval_rows = [
        _evaluate_factorized_search_candidate_rollout(
            candidate,
            probe_runs=probe_runs,
            pass_nrmse=float(pass_nrmse),
            partial_nrmse=float(partial_nrmse),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
        )
        for candidate in shortlist
    ]
    de_candidate_eval_evals = [
        _evaluate_de_candidate_eval_rollout(
            candidate,
            probe_runs=probe_runs,
            pass_nrmse=float(pass_nrmse),
            partial_nrmse=float(partial_nrmse),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
        )
        for candidate in de_candidate_eval_rows
    ]
    coe_mode = str(de_coe_mode or "off").strip().lower()
    coe_active = coe_mode in {"audit", "adjudicate", "reservoir"}
    coe_selects = coe_mode in {"adjudicate", "reservoir"}
    committee_proposal_slate = [row for row in list(de_meta.get("proposal_slate", []) or []) if isinstance(row, dict)]
    candidate_pairs: list[tuple[str, dict[str, Any]]] = [("factorized_search", row) for row in eval_rows]
    candidate_pairs.extend(("de_candidate_eval", row) for row in de_candidate_eval_evals)
    if coe_active and (coe_mode == "reservoir" or int(de_coe_reservoir_scouts) > 0):
        scout_slates, scout_diag = _run_reservoir_scout_reports(
            problem,
            fit_runs,
            results_dir=results_dir,
            fast=fast,
            verbose=verbose,
            max_scouts=int(de_coe_reservoir_scouts),
            factorized_search_preset=factorized_search_preset,
            use_dims=bool(use_dims),
        )
        merged_slate, merge_diag = _merge_reservoir_scout_slates(
            committee_proposal_slate,
            scout_slates,
        )
        committee_proposal_slate = merged_slate
        max_scout_candidates = min(6, max(0, 2 * int(de_coe_reservoir_scouts)))
        candidate_pairs, eval_diag = _evaluate_reservoir_scout_proposals(
            committee_proposal_slate,
            candidate_pairs,
            probe_runs=probe_runs,
            pass_nrmse=float(pass_nrmse),
            partial_nrmse=float(partial_nrmse),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            max_candidates=max_scout_candidates,
        )
        out["committee_reservoir"] = {
            "enabled": True,
            "mode": coe_mode,
            "scouts": scout_diag,
            "merge": merge_diag,
            "validation": eval_diag,
        }
        out["reservoir_validated_candidates"] = int(eval_diag.get("validated", 0))
    committee_decision = None
    if coe_active:
        committee_decision = run_de_committee_audit(
            committee_proposal_slate,
            rollout_candidates=[row for _, row in candidate_pairs],
            selected_engine=str(out.get("internal_selected_engine", "factorized_search")),
            config={
                "mode": coe_mode,
                "source": "benchmark",
                "engine": "factorized_search_only",
                "reservoir_scouts_requested": int(de_coe_reservoir_scouts),
            },
            run_compile_domain=False,
        ).to_dict()
        _attach_committee_audit_fields(
            out,
            committee_decision,
            internal_selected_engine=str(out.get("internal_selected_engine", "factorized_search")),
        )
    selection_pairs = list(candidate_pairs) if coe_selects else [("factorized_search", row) for row in eval_rows]
    if not coe_selects:
        selection_pairs.extend(("de_candidate_eval", row) for row in de_candidate_eval_evals)
    selection_rows = [row for _, row in selection_pairs if isinstance(row, dict)]
    if bool(de_coe_csr_on_ties):
        csr_candidates, csr_diag = _maybe_run_csr_on_tied_factorized_search(
            committee_decision=committee_decision,
            current_candidates=candidate_pairs,
            factorized_search_shortlist=shortlist,
            fit_runs=fit_runs,
            probe_runs=probe_runs,
            pass_nrmse=float(pass_nrmse),
            partial_nrmse=float(partial_nrmse),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            enabled=bool(coe_active),
        )
        out["committee_csr"] = csr_diag
        out["factorized_search_csr_validated_candidates"] = int(csr_diag.get("validated", 0))
        if int(csr_diag.get("validated", 0)) > 0:
            committee_decision = run_de_committee_audit(
                committee_proposal_slate,
                rollout_candidates=[row for _, row in csr_candidates],
                selected_engine=str(out.get("internal_selected_engine", "factorized_search")),
                config={
                    "mode": coe_mode,
                    "source": "benchmark",
                    "engine": "factorized_search_only",
                    "csr_after_tie": True,
                    "reservoir_scouts_requested": int(de_coe_reservoir_scouts),
                },
                run_compile_domain=False,
            ).to_dict()
            _attach_committee_audit_fields(
                out,
                committee_decision,
                internal_selected_engine=str(out.get("internal_selected_engine", "factorized_search")),
            )
            if coe_selects:
                selection_pairs = list(csr_candidates)
                selection_rows = [row for _, row in selection_pairs if isinstance(row, dict)]
    committee_choice = None
    if coe_selects:
        committee_choice = _committee_selected_rollout_candidate(
            committee_decision,
            selection_pairs,
        )
    if committee_choice is not None:
        _, eval_out = committee_choice
        out["selection_mode"] = "committee_adjudicate" if coe_mode == "adjudicate" else "committee_reservoir"
        out["committee_adjudicated"] = True
    else:
        if coe_selects:
            _, eval_out = _choose_rollout_candidate(selection_pairs, fallback_engine="factorized_search")
        else:
            eval_out = _choose_best_same_engine_rollout("factorized_search", selection_rows)
        out["selection_mode"] = "rollout_nrmse"
        out["committee_adjudicated"] = False
        if coe_selects:
            out["committee_adjudication_fallback"] = True
    out["factorized_search_validated_candidates"] = int(len(eval_rows))
    out["de_candidate_eval_validated_candidates"] = int(len(de_candidate_eval_evals))
    out["status"] = str(eval_out.get("status", "ERROR"))
    out["message"] = str(eval_out.get("message", ""))
    out["traj_scores"] = list(eval_out.get("traj_scores", []) or [])
    out["discovered_order"] = int(eval_out.get("discovered_order", -1))
    out["canonical_equation"] = str(eval_out.get("canonical_equation", out["canonical_equation"]))
    out["selected_engine"] = str(
        "de_candidate_eval" if str(eval_out.get("engine", "")) == "de_candidate_eval" else "factorized_search"
    )
    out["selected_candidate_id"] = str(eval_out.get("candidate_id", "") or "")
    out["selected_candidate_key"] = str(eval_out.get("candidate_key", "") or "")
    out["internal_selected_candidate_id"] = ""
    out["internal_selected_candidate_key"] = ""
    out["internal_selected_candidate_id_mismatch"] = False
    out["selected_shortlist_rank"] = int(eval_out.get("candidate_rank", 10**9))
    if str(out["selected_engine"]) == "de_candidate_eval":
        out["selected_candidate_family"] = str(eval_out.get("candidate_family", ""))
        out["internal_selected_shortlist_rank"] = None
        out["internal_selected_shortlist_rank_mismatch"] = False
    else:
        internal_rank = selected.get("internal_selected_shortlist_rank", None)
        if internal_rank is None:
            internal_rank = 0 if shortlist else None
        out["internal_selected_shortlist_rank"] = None if internal_rank is None else int(internal_rank)
        internal_candidate = _shortlist_candidate_by_rank(shortlist, internal_rank)
        out["internal_selected_candidate_id"] = (
            "" if internal_candidate is None else str(internal_candidate.get("candidate_id", "") or "")
        )
        out["internal_selected_candidate_key"] = (
            "" if internal_candidate is None else str(internal_candidate.get("candidate_key", "") or "")
        )
        out["internal_selected_shortlist_rank_mismatch"] = bool(
            shortlist
            and out["internal_selected_shortlist_rank"] is not None
            and int(eval_out.get("candidate_rank", 10**9)) != int(out["internal_selected_shortlist_rank"])
        )
        out["internal_selected_candidate_id_mismatch"] = bool(
            out["selected_candidate_id"]
            and out["internal_selected_candidate_id"]
            and out["selected_candidate_id"] != out["internal_selected_candidate_id"]
        )
    return out


def _build_dims_and_constants(
    problem: ProblemDef,
    param_values: dict[str, float],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Build dims dict and constants list from the shared canonical metadata.

    Returns ``(dims_dict, constants_list)`` ready for the spec writer.
    ``dims_dict`` is None when the problem has no dimensional metadata.
    """
    canonical_dims = get_canonical_problem_dims(str(problem.id))
    dims_dict = None
    if canonical_dims is not None:
        dims_dict = canonical_scalar_dims_payload(canonical_dims, x_axis=0, component_idx=0)
    constants = canonical_constant_payload(
        canonical_dims,
        param_values,
        names=tuple(problem.params),
    )
    return dims_dict, constants


def _write_oracle_spec(
    problem: ProblemDef,
    runs: Sequence[TrajRun],
    *,
    holdout_last_k: int,
    traj_metric: str,
    path: Path,
    param_values: dict[str, float] | None = None,
    use_dims: bool = True,
) -> dict[str, Any]:
    dims_dict: dict[str, Any] | None = None
    constants: list[dict[str, Any]] = []
    if param_values is not None:
        dims_dict, constants = _build_dims_and_constants(problem, param_values)
        if not use_dims:
            dims_dict = None

    return write_oracle_de_spec(
        path,
        spec_id=f"de{problem.id}",
        trajectories=list(runs),
        holdout_last_k=int(holdout_last_k),
        x_axis=0,
        order_candidates=(1, 2),
        include_x=True,
        y_transform="identity",
        traj_metric=str(traj_metric),
        sort_trajectories=True,
        constants=constants or None,
        dims=dims_dict,
        extra={
            "problem_id": str(problem.id),
            "description": str(problem.description),
            "order_preference_factor": 1.0,
        },
    )


def _run_factorized_search_engine(
    problem: ProblemDef,
    runs: Sequence[TrajRun],
    *,
    probe_runs: Sequence[TrajRun] | None = None,
    results_dir: Path,
    fast: bool,
    verbose: bool,
    holdout_last_k: int,
    traj_metric: str,
    no_sim_validate: bool,
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_max_candidates: int = 0,
    sim_validate_traj_time_budget_s: float | None = 20.0,
    sim_validate_blowup_factor: float = 100.0,
    sim_validate_blowup_abs: float = 1.0e6,
    sim_validate_progress: bool = True,
    param_values: dict[str, float] | None = None,
    use_dims: bool = True,
) -> dict[str, Any]:
    probe_runs = list(probe_runs or runs)
    spec_path = results_dir / "oracle_specs" / f"de{problem.id}_oracle_spec.json"
    out_json = results_dir / f"de{problem.id}_oracle_de.json"
    spec_payload = _write_oracle_spec(
        problem,
        runs,
        holdout_last_k=int(holdout_last_k),
        traj_metric=traj_metric,
        path=spec_path,
        param_values=param_values,
        use_dims=use_dims,
    )

    cmd = build_oracle_de_command(spec_path, out_json, fast=fast, verbose=verbose)
    if verbose:
        print(
            f"  [factorized_search_oracle] Split: mode={spec_payload.get('split_mode')} "
            f"holdout_last_k={int(spec_payload.get('holdout_last_k', 0))} "
            f"fit={len(spec_payload.get('fit_trajectories', []))} "
            f"probe={len(spec_payload.get('probe_trajectories', []))}"
        )
        print(f"  [factorized_search_oracle] Command: {' '.join(cmd)}")

    log_path = results_dir / f"de{problem.id}_factorized_search_oracle.log"
    proc = _run_command_to_log(
        cmd,
        log_path,
        cwd=str(REPO_ROOT),
        env=_subprocess_env_with_repo_root(),
    )
    if proc.returncode != 0:
        msg = _format_command_failure_message("oracle_lab_de", proc)
        if verbose:
            print(f"  [factorized_search_oracle] See log: {log_path}")
        return _attach_command_resource_report(
            {"engine": "factorized_search_oracle", "status": "ERROR", "message": msg},
            proc,
        )

    if not out_json.exists():
        return {
            "engine": "factorized_search_oracle",
            "status": "ERROR",
            "message": f"Missing JSON: {out_json}",
        }

    report = json.loads(out_json.read_text(encoding="utf-8"))
    report["command_resource_report"] = _command_resource_report(proc)
    # Inject spec-ordered constants for deterministic RHS reconstruction.
    if isinstance(spec_payload.get("constants", None), list):
        report["constants"] = list(spec_payload["constants"])
    if "include_x" not in report:
        report["include_x"] = bool(spec_payload.get("include_x", True))
    best = report.get("best", None)
    if best is None:
        return {
            "engine": "factorized_search_oracle",
            "status": "FAIL",
            "message": "No best candidate in oracle report",
            "json_path": str(out_json),
            "canonical_equation": "",
        }

    canonical = str(best.get("residual_ast", "")) or str(best.get("expr", ""))
    out: dict[str, Any] = {
        "engine": "factorized_search_oracle",
        "json_path": str(out_json),
        "command_resource_report": _command_resource_report(proc),
        "canonical_equation": canonical,
    }

    if bool(no_sim_validate):
        out["status"] = "UNVERIFIED"
        out["message"] = "Simulation validation disabled"
        return out

    # --- Validate candidates by simulation -------------------------------
    # Collect candidates: (order, row) from per_order[i]["results"].
    cands: list[tuple[int, dict]] = []
    for po in report.get("per_order", []):
        po_order = int(po.get("order", -1))
        for row in po.get("results", []):
            cands.append((po_order, row))

    # Sort by raw score (best first), de-duplicate by expression + mapping.
    cands.sort(key=lambda c: float(c[1].get("score", float("inf"))))
    seen_exprs: set[str] = set()
    unique_cands: list[tuple[int, dict]] = []
    for cand_order, row in cands:
        key = json.dumps(row.get("expr_ast", ""), sort_keys=True) + "|" + json.dumps(row.get("mapping", ""), sort_keys=True)
        if key in seen_exprs:
            continue
        seen_exprs.add(key)
        unique_cands.append((cand_order, row))

    if int(sim_validate_max_candidates) > 0:
        unique_cands = unique_cands[: int(sim_validate_max_candidates)]

    if sim_validate_progress:
        print(f"  [sim-validate] candidates={len(unique_cands)} (deduped across orders)")

    # Validate each candidate by simulation and then choose the best one by:
    #   1) Prefer any PASS over PARTIAL over FAIL.
    #   2) Within a status bucket, minimize worst-case trajectory error (max NRMSE).
    #   3) Tie-break by expression size, then discovered order (Occam).
    #
    # This avoids "first PASS wins" when multiple candidates pass.
    best_pass_key: tuple[float, int, int] = (float("inf"), 10**9, 99)
    best_partial_key: tuple[float, int, int] = (float("inf"), 10**9, 99)
    best_fail_key: tuple[float, int, int] = (float("inf"), 10**9, 99)
    best_pass: tuple[str, str, list, int | None, dict] | None = None
    best_partial: tuple[str, str, list, int | None, dict] | None = None
    best_fail: tuple[str, str, list, int | None, dict] | None = None

    for i_cand, (cand_order, row) in enumerate(unique_cands):
        if sim_validate_progress:
            score_s = row.get("score", row.get("mse", None))
            try:
                score_f = float(score_s)
                score_str = f"{score_f:.3g}" if math.isfinite(score_f) else str(score_s)
            except Exception:
                score_str = str(score_s)
            expr_preview = str(row.get("residual_ast", "") or row.get("expr", ""))
            expr_preview = expr_preview.replace("\n", " ")
            if len(expr_preview) > 140:
                expr_preview = expr_preview[:137] + "..."
            print(
                f"    [cand {i_cand+1:02d}/{len(unique_cands):02d}] "
                f"order={int(cand_order)} score={score_str} size={int(row.get('size', -1))} | {expr_preview}"
            )
            sys.stdout.flush()

        fb_report = {**report, "best": row}
        try:
            fb_ord, fb_rhs = factorized_search_report_to_rhs_callable(fb_report)
        except Exception:
            if sim_validate_progress:
                print("      -> SKIP (could not build RHS)")
            continue
        fb_status, fb_msg, fb_scores = validate_by_simulation(
            probe_runs,
            rhs_fn=fb_rhs,
            order=int(fb_ord),
            pass_nrmse=float(pass_nrmse),
            partial_nrmse=float(partial_nrmse),
            traj_time_budget_s=sim_validate_traj_time_budget_s,
            blowup_factor=float(sim_validate_blowup_factor),
            blowup_abs=float(sim_validate_blowup_abs),
        )

        # Key: (max_nrmse, expression_size, order). Lower is better.
        max_nrmse = max((float(s.get("nrmse", float("inf"))) for s in fb_scores), default=float("inf"))
        if not math.isfinite(max_nrmse):
            max_nrmse = float("inf")
        expr_size = int(row.get("size", 10**9))
        ord_i = int(fb_ord) if fb_ord is not None else int(cand_order)
        cand_key = (max_nrmse, expr_size, ord_i)
        pack = (fb_status, fb_msg, fb_scores, fb_ord, row)

        if sim_validate_progress:
            print(f"      -> {fb_status}: {fb_msg} | key={cand_key}")
            sys.stdout.flush()

        if fb_status == "PASS":
            if cand_key < best_pass_key:
                best_pass_key = cand_key
                best_pass = pack
        elif fb_status == "PARTIAL":
            if cand_key < best_partial_key:
                best_partial_key = cand_key
                best_partial = pack
        else:
            if cand_key < best_fail_key:
                best_fail_key = cand_key
                best_fail = pack

    best_choice = best_pass if best_pass is not None else (best_partial if best_partial is not None else best_fail)
    best_key = best_pass_key if best_pass is not None else (best_partial_key if best_partial is not None else best_fail_key)

    if best_choice is not None:
        status, message, traj_scores, order, best = best_choice
        canonical = str(best.get("residual_ast", "")) or str(best.get("expr", ""))
        out["canonical_equation"] = canonical
        if verbose and order is not None:
            print(f"  [sim-select] order={order} status={status} "
                  f"key={best_key}")
    else:
        status, message, traj_scores = "FAIL", "No candidate could be integrated", []
        order = None

    out["status"] = str(status)
    out["message"] = str(message)
    out["traj_scores"] = traj_scores
    out["discovered_order"] = int(order) if order is not None else -1
    out["best_score"] = float(best.get("score", best.get("mse", float("inf")))) if isinstance(best, dict) else float("inf")
    return out


def _compute_hybrid_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    hybrid_rows = []
    for row in results:
        engines = row.get("engines", {}) or {}
        if "hybrid" in engines:
            hybrid_rows.append(engines["hybrid"])
        elif str(row.get("engine", "")) == "hybrid":
            hybrid_rows.append(row)

    if not hybrid_rows:
        return {}

    sparse_only_pass = 0
    rescued_additional = 0
    final_pass = 0
    rescued_ids: list[str] = []
    failure_kind_counts: dict[str, int] = {}
    for row_parent in results:
        engines = row_parent.get("engines", {}) or {}
        row = engines.get("hybrid", row_parent if str(row_parent.get("engine", "")) == "hybrid" else None)
        if row is None:
            continue
        if str(row.get("first_line_status", "")) == "PASS":
            sparse_only_pass += 1
        if str(row.get("status", "")) == "PASS":
            final_pass += 1
        if bool(row.get("rescued_additional", False)):
            rescued_additional += 1
            rescued_ids.append(str(row_parent.get("id", "")))
        kind = row.get("failure_kind", None)
        if kind is None:
            kind = classify_failure_kind(
                row.get("status", None),
                row.get("message", None),
                row.get("traj_scores", None),
            )
        if kind:
            failure_kind_counts[str(kind)] = failure_kind_counts.get(str(kind), 0) + 1

    return {
        "sparse_only_pass": int(sparse_only_pass),
        "rescued_additional": int(rescued_additional),
        "final_pass": int(final_pass),
        "rescued_problem_ids": rescued_ids,
        "failure_kind_counts": failure_kind_counts,
    }


def _factorized_de_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in results:
        engines = row.get("engines", {}) or {}
        if "factorized_de" in engines and isinstance(engines["factorized_de"], dict):
            rows.append(engines["factorized_de"])
        elif str(row.get("engine", "")) == "factorized_de":
            rows.append(row)
        elif isinstance(row.get("clean_fallback_result", None), dict):
            rows.append(row["clean_fallback_result"])
    return rows


def _count_strings(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _row_nrmse_values(row: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for score in list(row.get("traj_scores", []) or []):
        if not isinstance(score, Mapping):
            continue
        value = _safe_float_or_none(score.get("nrmse", None))
        if value is not None:
            values.append(float(value))
    return values


def _compute_factorized_de_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = _factorized_de_rows(results)
    if not rows:
        return {}

    nrmse_values: list[float] = []
    for row in rows:
        nrmse_values.extend(_row_nrmse_values(row))

    return {
        "total": int(len(rows)),
        "final_pass": int(sum(1 for row in rows if str(row.get("status", "")) == "PASS")),
        "selected_engine_counts": _count_strings([row.get("selected_engine", "") for row in rows]),
        "selected_lane_counts": _count_strings([row.get("selected_lane", "") for row in rows]),
        "typed_lanes_policy_counts": _count_strings([row.get("typed_lanes_policy", "") for row in rows]),
        "coefficient_dim_mode_counts": _count_strings([row.get("coefficient_dim_mode", "") for row in rows]),
        "direct_residual_attempted": int(sum(1 for row in rows if bool(row.get("direct_residual_attempted", False)))),
        "typed_selected_lane_counts": _count_strings([row.get("typed_selected_lane", "") for row in rows]),
        "whole_rhs_attempted": int(sum(1 for row in rows if bool(row.get("whole_rhs_attempted", False)))),
        "whole_rhs_skipped": int(sum(1 for row in rows if not bool(row.get("whole_rhs_attempted", False)))),
        "whole_rhs_attempts_run": int(sum(int(row.get("whole_rhs_attempts_run", 0) or 0) for row in rows)),
        "family_gate_skips": int(sum(int(row.get("family_gate_skips", 0) or 0) for row in rows)),
        "typed_explorer_launches": int(sum(int(row.get("typed_explorer_launches", 0) or 0) for row in rows)),
        "median_rollout_nrmse": None if not nrmse_values else float(np.median(np.asarray(nrmse_values))),
        "worst_rollout_nrmse": None if not nrmse_values else float(max(nrmse_values)),
    }


def run_problem(
    problem: ProblemDef,
    *,
    data_dir: Path,
    results_dir: Path,
    fast: bool,
    skip_generate: bool,
    verbose: bool,
    engine: str,
    n_traj: int,
    n_points: int,
    seed: int,
    split_mode: str,
    holdout_last_k: int | None,
    traj_metric: str,
    no_sim_validate: bool,
    pass_nrmse: float,
    partial_nrmse: float,
    sim_validate_max_candidates: int = 0,
    sim_validate_traj_time_budget_s: float | None = 20.0,
    sim_validate_blowup_factor: float = 100.0,
    sim_validate_blowup_abs: float = 1.0e6,
    sim_validate_progress: bool = True,
    factorized_search_preset: str | None = None,
    factorized_two_block_shared_coord: str | None = None,
    de_coe_mode: str = "off",
    de_coe_csr_on_ties: bool = False,
    de_coe_reservoir_scouts: int = 0,
    factorized_de_whole_rhs: str = "auto",
    factorized_de_typed_lanes: str = "never",
    factorized_de_typed_lane_workers: int = 1,
    factorized_search_de_refine_mode: str = "rare_final_polish",
    factorized_search_max_attempts: int | None = None,
    factorized_search_integrate_topk: int | None = None,
    factorized_search_direct_generator_witness_topk: int | None = None,
    use_dims: bool = True,
) -> dict[str, Any]:
    engine = _normalize_engine_name(engine)
    pid = str(problem.id)
    result: dict[str, Any] = {
        "id": pid,
        "description": problem.description,
        "status": "ERROR",
        "message": "",
        "engine": str(engine),
    }

    param_values = default_param_values(problem)
    result["param_values"] = dict(param_values)

    problem_seed = _problem_seed(seed, pid)
    if skip_generate:
        try:
            runs, rhs_source = load_existing_runs(problem, data_dir, n_traj=int(n_traj))
        except Exception as exc:
            result["message"] = f"Loading existing trajectories failed: {exc}"
            return result
    else:
        try:
            runs, rhs_source = generate_data_multi(
                problem,
                param_values,
                data_dir,
                n_traj=int(n_traj),
                n_points=int(n_points),
                seed=int(problem_seed),
            )
        except Exception as exc:
            result["message"] = f"Data generation failed: {exc}"
            return result

    result["rhs_source"] = str(rhs_source)
    result["n_traj"] = int(len(runs))
    result["trajectories"] = [str(r.csv_path) for r in runs]

    split_mode_l = str(split_mode).strip().lower()
    if split_mode_l not in ("per_traj_point", "traj_holdout"):
        result["message"] = f"Invalid split_mode: {split_mode!r}"
        return result
    if holdout_last_k is None:
        eff_holdout = max(1, int(len(runs)) // 3) if (split_mode_l == "traj_holdout" and len(runs) > 1) else 0
    else:
        eff_holdout = int(holdout_last_k)
    if eff_holdout < 0:
        result["message"] = f"holdout_last_k must be >= 0, got {eff_holdout}"
        return result
    if split_mode_l == "per_traj_point":
        if eff_holdout != 0:
            result["message"] = f"split_mode=per_traj_point requires holdout_last_k=0, got {eff_holdout}"
            return result
    else:
        if len(runs) <= 1:
            eff_holdout = 0
        elif eff_holdout <= 0:
            result["message"] = "split_mode=traj_holdout requires holdout_last_k>=1 when n_traj>1"
            return result
        elif eff_holdout >= len(runs):
            result["message"] = (
                f"holdout_last_k={eff_holdout} must be < n_traj={len(runs)} for traj_holdout"
            )
            return result
    result["holdout_last_k"] = int(eff_holdout)
    fit_runs, probe_runs = _split_runs_for_holdout(
        runs,
        split_mode=split_mode_l,
        holdout_last_k=int(eff_holdout),
    )
    result["fit_trajectories"] = [str(r.csv_path) for r in fit_runs]
    result["probe_trajectories"] = [str(r.csv_path) for r in probe_runs]
    result["n_fit_traj"] = int(len(fit_runs))
    result["n_probe_traj"] = int(len(probe_runs))

    if verbose:
        print(f"  Trajectories: {len(runs)}")
        print(f"  RHS source: {rhs_source}")
        print(f"  Split mode: {split_mode_l}, holdout_last_k={int(eff_holdout)}")
        print(f"  Fit trajectories:   {[r.traj_id for r in fit_runs]}")
        print(f"  Probe trajectories: {[r.traj_id for r in probe_runs]}")

    engines: dict[str, dict[str, Any]] = {}

    if engine in ("sparse", "compare"):
        engines["sparse"] = _run_stlsq_engine(
            problem,
            fit_runs,
            probe_runs=probe_runs,
            results_dir=results_dir,
            fast=fast,
            verbose=verbose,
            no_sim_validate=no_sim_validate,
            pass_nrmse=pass_nrmse,
            partial_nrmse=partial_nrmse,
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            de_coe_mode=de_coe_mode,
            use_dims=bool(use_dims),
        )

    if engine in ("hybrid", "compare"):
        engines["hybrid"] = _run_clean_fallback_hybrid_engine(
            problem,
            fit_runs,
            probe_runs=probe_runs,
            results_dir=results_dir,
            fast=fast,
            verbose=verbose,
            no_sim_validate=no_sim_validate,
            pass_nrmse=pass_nrmse,
            partial_nrmse=partial_nrmse,
            sim_validate_max_candidates=int(sim_validate_max_candidates),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            sim_validate_progress=bool(sim_validate_progress),
            factorized_search_preset=factorized_search_preset,
            factorized_two_block_shared_coord=factorized_two_block_shared_coord,
            de_coe_mode=de_coe_mode,
            de_coe_csr_on_ties=bool(de_coe_csr_on_ties),
            de_coe_reservoir_scouts=int(de_coe_reservoir_scouts),
            factorized_de_whole_rhs=factorized_de_whole_rhs,
            factorized_de_typed_lanes=factorized_de_typed_lanes,
            factorized_de_typed_lane_workers=int(factorized_de_typed_lane_workers),
            factorized_search_de_refine_mode=factorized_search_de_refine_mode,
            factorized_search_max_attempts=factorized_search_max_attempts,
            factorized_search_integrate_topk=factorized_search_integrate_topk,
            factorized_search_direct_generator_witness_topk=factorized_search_direct_generator_witness_topk,
            use_dims=bool(use_dims),
        )

    if engine in ("factorized_de", "compare"):
        engines["factorized_de"] = _run_hybrid_engine(
            problem,
            fit_runs,
            probe_runs=probe_runs,
            results_dir=results_dir,
            fast=fast,
            verbose=verbose,
            no_sim_validate=no_sim_validate,
            pass_nrmse=pass_nrmse,
            partial_nrmse=partial_nrmse,
            sim_validate_max_candidates=int(sim_validate_max_candidates),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            sim_validate_progress=bool(sim_validate_progress),
            factorized_search_preset=factorized_search_preset,
            factorized_two_block_shared_coord=factorized_two_block_shared_coord,
            factorized_de=True,
            de_coe_mode=de_coe_mode,
            de_coe_csr_on_ties=bool(de_coe_csr_on_ties),
            de_coe_reservoir_scouts=int(de_coe_reservoir_scouts),
            factorized_de_whole_rhs=factorized_de_whole_rhs,
            factorized_de_typed_lanes=factorized_de_typed_lanes,
            factorized_de_typed_lane_workers=int(factorized_de_typed_lane_workers),
            factorized_search_de_refine_mode=factorized_search_de_refine_mode,
            factorized_search_max_attempts=factorized_search_max_attempts,
            factorized_search_integrate_topk=factorized_search_integrate_topk,
            factorized_search_direct_generator_witness_topk=factorized_search_direct_generator_witness_topk,
            use_dims=bool(use_dims),
        )

    if engine == "factorized_search_only":
        engines["factorized_search_only"] = _run_factorized_search_only_engine(
            problem,
            fit_runs,
            probe_runs=probe_runs,
            results_dir=results_dir,
            fast=fast,
            verbose=verbose,
            no_sim_validate=no_sim_validate,
            pass_nrmse=pass_nrmse,
            partial_nrmse=partial_nrmse,
            sim_validate_max_candidates=int(sim_validate_max_candidates),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            sim_validate_progress=bool(sim_validate_progress),
            factorized_search_preset=factorized_search_preset,
            de_coe_mode=de_coe_mode,
            de_coe_csr_on_ties=bool(de_coe_csr_on_ties),
            de_coe_reservoir_scouts=int(de_coe_reservoir_scouts),
            factorized_search_de_refine_mode=factorized_search_de_refine_mode,
            factorized_search_max_attempts=factorized_search_max_attempts,
            factorized_search_integrate_topk=factorized_search_integrate_topk,
            factorized_search_direct_generator_witness_topk=factorized_search_direct_generator_witness_topk,
            use_dims=bool(use_dims),
        )

    if engine in ("factorized_search_oracle", "compare"):
        engines["factorized_search_oracle"] = _run_factorized_search_engine(
            problem,
            runs,
            probe_runs=probe_runs,
            results_dir=results_dir,
            fast=fast,
            verbose=verbose,
            holdout_last_k=int(eff_holdout),
            traj_metric=traj_metric,
            no_sim_validate=no_sim_validate,
            pass_nrmse=pass_nrmse,
            partial_nrmse=partial_nrmse,
            sim_validate_max_candidates=int(sim_validate_max_candidates),
            sim_validate_traj_time_budget_s=sim_validate_traj_time_budget_s,
            sim_validate_blowup_factor=float(sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(sim_validate_blowup_abs),
            sim_validate_progress=bool(sim_validate_progress),
            param_values=param_values,
            use_dims=use_dims,
        )

    if not engines:
        result["status"] = "ERROR"
        result["message"] = f"Unsupported engine: {engine}"
        result["failure_kind"] = classify_failure_kind(result["status"], result["message"])
        return result

    for row in engines.values():
        if isinstance(row, dict):
            row["failure_kind"] = classify_failure_kind(
                row.get("status", None),
                row.get("message", None),
                row.get("traj_scores", None),
            )
    result["engines"] = engines

    primary_key = {
        "sparse": "sparse",
        "hybrid": "hybrid",
        "factorized_search_only": "factorized_search_only",
        "factorized_de": "factorized_de",
        "factorized_search_oracle": "factorized_search_oracle",
        "compare": "hybrid",
    }.get(engine, "sparse")
    primary = engines.get(primary_key) or next(iter(engines.values()))
    result["status"] = str(primary.get("status", "ERROR"))
    result["message"] = str(primary.get("message", ""))
    result["canonical_equation"] = str(primary.get("canonical_equation", ""))
    result["failure_kind"] = primary.get(
        "failure_kind",
        classify_failure_kind(
            primary.get("status", None),
            primary.get("message", None),
            primary.get("traj_scores", None),
        ),
    )
    if "coeff_map" in primary:
        result["coeff_map"] = primary["coeff_map"]
    if "traj_scores" in primary:
        result["traj_scores"] = primary["traj_scores"]
    if "json_path" in primary:
        result["json_path"] = primary["json_path"]
    if "selected_engine" in primary:
        result["selected_engine"] = primary["selected_engine"]
    if "internal_selected_engine" in primary:
        result["internal_selected_engine"] = primary["internal_selected_engine"]
    if "first_line_status" in primary:
        result["first_line_status"] = primary["first_line_status"]
    if "rescued_additional" in primary:
        result["rescued_additional"] = bool(primary["rescued_additional"])
    if "committee_selected_engine" in primary:
        result["committee_selected_engine"] = primary["committee_selected_engine"]
    if "internal_selected_engine_committee_mismatch" in primary:
        result["internal_selected_engine_committee_mismatch"] = bool(
            primary["internal_selected_engine_committee_mismatch"]
        )
    if "committee_decision" in primary:
        result["committee_decision"] = primary["committee_decision"]
    if "committee_csr" in primary:
        result["committee_csr"] = primary["committee_csr"]
    if "committee_reservoir" in primary:
        result["committee_reservoir"] = primary["committee_reservoir"]

    if engine == "compare":
        sp = engines.get("sparse", {}).get("status", "?")
        hy = engines.get("hybrid", {}).get("status", "?")
        fd = engines.get("factorized_de", {}).get("status", "?")
        bo = engines.get("factorized_search_oracle", {}).get("status", "?")
        result["message"] = (
            f"sparse={sp}; hybrid={hy}; factorized_de={fd}; "
            f"factorized_search_oracle={bo}; primary={primary_key}: {result['message']}"
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Feynman DE benchmark runner with multi-trajectory engines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated problem IDs, e.g. 'de000,de001' or '000,001'",
    )
    parser.add_argument("--all", action="store_true", help="Run all problems")
    parser.add_argument(
        "--engine",
        type=str,
        choices=[
            "sparse",
            "hybrid",
            "factorized_search_only",
            "factorized_de",
            "factorized_search_oracle",
            "compare",
            "stlsq",
            "both",
            "factorized",
            "factorized_search_de",
        ],
        default="hybrid",
    )
    parser.add_argument("--n_traj", type=int, default=6, help="Number of trajectories (different ICs) per problem")
    parser.add_argument("--n_points", type=int, default=5000, help="Points per generated trajectory")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for IC sampling")
    parser.add_argument("--split_mode", type=str, choices=["per_traj_point", "traj_holdout"], default="traj_holdout")
    parser.add_argument(
        "--holdout_last_k",
        type=int,
        default=None,
        help=(
            "When using factorized_search specs, hold out the last K trajectories for probe. "
            "Default: 0 for per_traj_point, else max(1, n_traj//3)."
        ),
    )
    parser.add_argument("--traj_metric", type=str, choices=["mean", "max"], default="max")
    parser.add_argument("--no_sim_validate", action="store_true", help="Disable simulation-based validation")
    parser.add_argument(
        "--sim_validate_max_candidates",
        type=int,
        default=0,
        help=(
            "Max number of candidates to validate by simulation for factorized symbolic search shortlist/oracle paths (0=all). "
            "Oracle candidates are deduped across orders; hybrid/factorized_search_only/factorized_de use serialized shortlists."
        ),
    )
    parser.add_argument(
        "--sim_validate_traj_time_budget_s",
        type=float,
        default=20.0,
        help=(
            "Wall-time budget (seconds) per trajectory during simulation validation. "
            "Applied across solver-method fallbacks. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--sim_validate_blowup_factor",
        type=float,
        default=100.0,
        help="Early-terminate integration when |u| exceeds factor*max(|u_true|) (trajectory-relative).",
    )
    parser.add_argument(
        "--sim_validate_blowup_abs",
        type=float,
        default=1.0e6,
        help="Absolute early-terminate threshold for |u| during simulation validation.",
    )
    parser.add_argument(
        "--no_sim_validate_progress",
        action="store_true",
        help="Suppress per-candidate simulation-validation progress output",
    )
    parser.add_argument("--no_dims", action="store_true", help="Disable dimensional analysis in oracle specs")
    parser.add_argument("--pass_nrmse", type=float, default=1.0e-2)
    parser.add_argument("--partial_nrmse", type=float, default=5.0e-2)
    parser.add_argument("--fast", action="store_true", help="Reduced budgets")
    parser.add_argument(
        "--factorized-search-preset",
        type=str,
        choices=["fast", "default", "paper", "compositional", "compositional_fast"],
        default=None,
        help="Optional factorized symbolic search preset override passed through to run_de.py for hybrid/factorized_search_only/factorized_de.",
    )
    parser.add_argument(
        "--factorized-two-block-shared-coord",
        type=str,
        choices=["never", "auto", "always"],
        default=None,
        help="Optional shared-coordinate two-block factorized rescue mode passed through to run_de.py for hybrid.",
    )
    parser.add_argument(
        "--de-coe-mode",
        type=str,
        choices=["off", "audit", "adjudicate", "reservoir"],
        default="off",
        help=(
            "DE Committee-of-Experts mode. 'audit' reports recommendations; "
            "'adjudicate' uses committee rollout ranking for the selected benchmark result; "
            "'reservoir' adds bounded trajectory-subset scouts and uses the support-aware committee."
        ),
    )
    parser.add_argument(
        "--de-coe-reservoir-scouts",
        type=int,
        default=0,
        help="Number of bounded trajectory-subset scouts to run in --de-coe-mode reservoir.",
    )
    parser.add_argument(
        "--de-coe-csr-on-ties",
        action="store_true",
        help="Run bounded continuous skeleton refinement on tied whole-RHS FSS rollout survivors.",
    )
    parser.add_argument(
        "--factorized-de-whole-rhs",
        type=str,
        choices=["never", "auto", "always"],
        default="auto",
        help="Policy for launching broad whole-RHS FSS in factorized_de.",
    )
    parser.add_argument(
        "--factorized-de-typed-lanes",
        type=str,
        choices=["never", "auto", "always", "force"],
        default="never",
        help=(
            "Policy for curated typed operator-factorized lanes in factorized_de. "
            "'always' enables them as a challenger when direct FSS is not exact; "
            "'force' preserves unconditional launch behavior."
        ),
    )
    parser.add_argument(
        "--factorized-de-typed-lane-workers",
        type=int,
        default=1,
        help="Maximum worker threads for independent typed-lane explorer launches inside factorized_de.",
    )
    parser.add_argument(
        "--factorized-search-de-refine-mode",
        type=str,
        choices=["off", "rare_final_polish", "rare_slate"],
        default="rare_final_polish",
        help="Continuous-refinement profile passed to DE-facing factorized symbolic search.",
    )
    parser.add_argument(
        "--factorized-search-integrate-topk",
        type=int,
        default=None,
        help=(
            "Internal integration-validation top-k passed to run_de.py for hybrid/"
            "factorized_search_only/factorized_de. Default: 0 when benchmark "
            "simulation validation is enabled, else run_de.py's standalone default."
        ),
    )
    parser.add_argument(
        "--factorized-search-max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum broad whole-RHS FSS heuristic attempts passed to run_de.py for hybrid/"
            "factorized_search_only/factorized_de."
        ),
    )
    parser.add_argument(
        "--factorized-search-direct-generator-witness-topk",
        type=int,
        default=None,
        help=(
            "Direct residual FSS generator-witness top-k passed to run_de.py for "
            "hybrid/factorized_search_only/factorized_de."
        ),
    )
    parser.add_argument("--skip_generate", action="store_true", help="Reuse existing generated trajectories")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(REPO_ROOT / "data" / "feynman_de"),
        help="Directory for generated CSVs",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(REPO_ROOT / "results" / "feynman_de"),
        help="Directory for results",
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default=str(BENCHMARK_FILE),
        help="Path to benchmark definition file",
    )
    args = parser.parse_args()
    args.engine = _normalize_engine_name(args.engine)

    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_problems = load_problems(args.benchmark_file)
    if args.only:
        ids = [pid.strip().replace("de", "") for pid in str(args.only).split(",")]
        problems = {pid: all_problems[pid] for pid in ids if pid in all_problems}
        missing = [pid for pid in ids if pid not in all_problems]
        if missing:
            print(f"Warning: unknown problem IDs: {missing}")
    elif args.all:
        problems = all_problems
    else:
        print("Specify --only <ids> or --all")
        return 1

    if not problems:
        print("No problems to run.")
        return 1

    results = []
    for pid in sorted(problems.keys()):
        problem = problems[pid]
        print(f"\n{'=' * 70}")
        print(f"de{pid}: {problem.description}")
        print(f"  Equation: {problem.equation}")
        print(f"  Order: {problem.order}, IC type: {problem.ic_type}")
        print(f"  Engine: {args.engine}, n_traj={int(args.n_traj)}")
        print(f"{'=' * 70}")

        result = run_problem(
            problem,
            data_dir=data_dir,
            results_dir=results_dir,
            fast=bool(args.fast),
            skip_generate=bool(args.skip_generate),
            verbose=bool(args.verbose),
            engine=str(args.engine),
            n_traj=int(args.n_traj),
            n_points=int(args.n_points),
            seed=int(args.seed),
            split_mode=str(args.split_mode),
            holdout_last_k=args.holdout_last_k,
            traj_metric=str(args.traj_metric),
            no_sim_validate=bool(args.no_sim_validate),
            pass_nrmse=float(args.pass_nrmse),
            partial_nrmse=float(args.partial_nrmse),
            sim_validate_max_candidates=int(args.sim_validate_max_candidates),
            sim_validate_traj_time_budget_s=(None if float(args.sim_validate_traj_time_budget_s) <= 0.0 else float(args.sim_validate_traj_time_budget_s)),
            sim_validate_blowup_factor=float(args.sim_validate_blowup_factor),
            sim_validate_blowup_abs=float(args.sim_validate_blowup_abs),
            sim_validate_progress=not bool(args.no_sim_validate_progress),
            factorized_search_preset=args.factorized_search_preset,
            factorized_two_block_shared_coord=args.factorized_two_block_shared_coord,
            de_coe_mode=str(args.de_coe_mode),
            de_coe_csr_on_ties=bool(args.de_coe_csr_on_ties),
            de_coe_reservoir_scouts=int(args.de_coe_reservoir_scouts),
            factorized_de_whole_rhs=str(args.factorized_de_whole_rhs),
            factorized_de_typed_lanes=str(args.factorized_de_typed_lanes),
            factorized_de_typed_lane_workers=int(args.factorized_de_typed_lane_workers),
            factorized_search_de_refine_mode=str(args.factorized_search_de_refine_mode),
            factorized_search_max_attempts=args.factorized_search_max_attempts,
            factorized_search_integrate_topk=args.factorized_search_integrate_topk,
            factorized_search_direct_generator_witness_topk=args.factorized_search_direct_generator_witness_topk,
            use_dims=not bool(args.no_dims),
        )
        results.append(result)

        status = result["status"]
        marker = {
            "PASS": "OK",
            "PARTIAL": "~~",
            "FAIL": "XX",
            "SKIP": "--",
            "UNVERIFIED": "??",
            "ERROR": "!!",
        }
        print(f"  [{marker.get(status, '??')}] {status}: {result['message']}")
        if result.get("rhs_source"):
            print(f"  RHS source: {result['rhs_source']}")
        if result.get("canonical_equation"):
            print(f"  Discovered: {result['canonical_equation']}")
        if result.get("internal_selected_engine") and result.get("selected_engine") and result.get("internal_selected_engine") != result.get("selected_engine"):
            print(
                "  Rollout override: internal={} -> benchmark={}".format(
                    result.get("internal_selected_engine"),
                    result.get("selected_engine"),
                )
            )
        if (
            result.get("committee_selected_engine")
            and result.get("internal_selected_engine")
            and result.get("committee_selected_engine") != result.get("internal_selected_engine")
        ):
            print(
                "  Committee audit: internal={} -> committee={}".format(
                    result.get("internal_selected_engine"),
                    result.get("committee_selected_engine"),
                )
            )

    print("\n" + "=" * 70)
    print("FEYNMAN DE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'ID':<6} {'Description':<35} {'Status':<12} {'Details'}")
    print("-" * 70)
    for r in results:
        pid = r["id"]
        desc = r["description"][:33]
        status = r["status"]
        msg = r["message"].split("\n")[0][:40] if r["message"] else ""
        print(f"de{pid:<4} {desc:<35} {status:<12} {msg}")
    print("-" * 70)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    parts = [f"{s}: {n}" for s, n in sorted(counts.items())]
    print(f"Total: {len(results)} | {' | '.join(parts)}")
    failure_kind_counts = compute_failure_kind_counts(results)
    if failure_kind_counts:
        parts_fk = [f"{k}: {v}" for k, v in sorted(failure_kind_counts.items())]
        print(f"Failure breakdown: {' | '.join(parts_fk)}")
    hybrid_summary = _compute_hybrid_summary(results)
    if hybrid_summary:
        print(
            "Hybrid summary: sparse-only PASS={} | rescued additional={} | final PASS={}".format(
                int(hybrid_summary["sparse_only_pass"]),
                int(hybrid_summary["rescued_additional"]),
                int(hybrid_summary["final_pass"]),
            )
        )
        if hybrid_summary.get("failure_kind_counts"):
            parts_hf = [f"{k}: {v}" for k, v in sorted(hybrid_summary["failure_kind_counts"].items())]
            print(f"Hybrid failure breakdown: {' | '.join(parts_hf)}")
    factorized_de_summary = _compute_factorized_de_summary(results)
    if factorized_de_summary:
        lane_parts = [
            f"{k}={v}" for k, v in sorted((factorized_de_summary.get("selected_lane_counts", {}) or {}).items())
        ]
        print(
            "Factorized-DE summary: PASS={} / {} | direct attempted={} | whole-RHS attempted={} | "
            "typed explorer launches={} | family-gate skips={}".format(
                int(factorized_de_summary["final_pass"]),
                int(factorized_de_summary["total"]),
                int(factorized_de_summary["direct_residual_attempted"]),
                int(factorized_de_summary["whole_rhs_attempted"]),
                int(factorized_de_summary["typed_explorer_launches"]),
                int(factorized_de_summary["family_gate_skips"]),
            )
        )
        if lane_parts:
            print("Factorized-DE selected lanes: " + ", ".join(lane_parts))
        if factorized_de_summary.get("worst_rollout_nrmse") is not None:
            print(
                "Factorized-DE rollout NRMSE: median={:.3e} | worst={:.3e}".format(
                    float(factorized_de_summary["median_rollout_nrmse"]),
                    float(factorized_de_summary["worst_rollout_nrmse"]),
                )
            )
    print("=" * 70)

    summary_path = results_dir / "summary.json"
    summary = {
        "engine": str(args.engine),
        "problems": [
            {
                "id": r["id"],
                "description": r["description"],
                "status": r["status"],
                "message": r["message"],
                "failure_kind": r.get("failure_kind"),
                "selected_engine": r.get("selected_engine", ""),
                "internal_selected_engine": r.get("internal_selected_engine", ""),
                "committee_selected_engine": r.get("committee_selected_engine", ""),
                "internal_selected_engine_committee_mismatch": bool(
                    r.get("internal_selected_engine_committee_mismatch", False)
                ),
                "first_line_status": r.get("first_line_status", ""),
                "rescued_additional": bool(r.get("rescued_additional", False)),
                "rhs_source": r.get("rhs_source", ""),
                "n_traj": int(r.get("n_traj", 0)),
                "n_fit_traj": int(r.get("n_fit_traj", 0)),
                "n_probe_traj": int(r.get("n_probe_traj", 0)),
                "holdout_last_k": int(r.get("holdout_last_k", 0)),
                "fit_trajectories": r.get("fit_trajectories", []),
                "probe_trajectories": r.get("probe_trajectories", []),
                "canonical_equation": r.get("canonical_equation", ""),
                "canonical_equation_raw": r.get("canonical_equation_raw", ""),
                "canonical_equation_simplified": r.get("canonical_equation_simplified", ""),
                "param_values": r.get("param_values", {}),
                "coeff_map": {k: v for k, v in r.get("coeff_map", {}).items()},
                "traj_scores": r.get("traj_scores", []),
                "engines": r.get("engines", {}),
                "committee_decision": r.get("committee_decision", {}),
                "committee_csr": r.get("committee_csr", {}),
                "committee_reservoir": r.get("committee_reservoir", {}),
                "json_path": r.get("json_path", ""),
            }
            for r in results
        ],
        "counts": counts,
        "failure_kind_counts": failure_kind_counts,
    }
    if hybrid_summary:
        summary["hybrid_summary"] = hybrid_summary
    if factorized_de_summary:
        summary["factorized_de_summary"] = factorized_de_summary
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    if counts.get("FAIL", 0) > 0 or counts.get("ERROR", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
