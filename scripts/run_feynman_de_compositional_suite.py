#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Launch the compositional Feynman-DE mini-benchmark.

This suite is designed to probe equation forms that are outside the current
first-line sparse/template language but inside factorized symbolic search's compositional search
language. The default workflow runs each problem with the ``factorized_search_only``
engine, and the new ``factorized_de`` engine exercises the STLSQ-free
factorized-plus-factorized symbolic search path. It reports:

    - final PASS count
    - selected-engine counts
    - hybrid rescue counts when the hybrid engine is used
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "examples" / "feynman_de" / "run_benchmark.py"
BENCHMARK_FILE = REPO_ROOT / "data" / "feynman_de_compositional.txt"
DEFAULT_IDS = ("890", "891", "892", "893", "900", "901", "902", "903", "904", "905", "906")


@dataclass
class CaseOutcome:
    problem_id: str
    returncode: int
    case_dir: str
    launcher_log: str
    summary_path: str | None
    status: str | None
    message: str | None
    failure_kind: str | None
    selected_engine: str | None
    first_line_status: str | None
    rescued_additional: bool | None
    canonical_equation: str | None
    fit_trajectories: list[str]
    probe_trajectories: list[str]
    skipped_existing: bool = False


def _parse_ids(text: str) -> list[str]:
    raw_parts = [p.strip() for p in str(text).replace(" ", ",").split(",")]
    ids: list[str] = []
    for part in raw_parts:
        if not part:
            continue
        part = part.lower()
        if part.startswith("de"):
            part = part[2:]
        ids.append(part.zfill(3))
    if not ids:
        raise ValueError("No problem IDs were provided")
    return ids


def _default_results_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"feynman_de_compositional_{stamp}"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = str(REPO_ROOT)
    py_path_prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{py_path_prev}" if py_path_prev else repo_root
    return env


def _case_has_required_trajectories(args: argparse.Namespace, problem_id: str) -> bool:
    for k in range(int(args.n_traj)):
        stem = f"de{problem_id}_ic{k}"
        csv_path = Path(args.data_dir) / f"{stem}.csv"
        meta_path = Path(args.data_dir) / f"{stem}.meta.json"
        if not csv_path.exists() or not meta_path.exists():
            return False
    return True


def _build_case_command(args: argparse.Namespace, problem_id: str, case_dir: Path) -> list[str]:
    use_skip_generate = bool(args.skip_generate) and _case_has_required_trajectories(args, problem_id)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--benchmark_file",
        str(args.benchmark_file),
        "--only",
        str(problem_id),
        "--engine",
        str(args.engine),
        "--n_traj",
        str(args.n_traj),
        "--n_points",
        str(args.n_points),
        "--split_mode",
        "traj_holdout",
        "--holdout_last_k",
        str(args.holdout_last_k),
        "--results_dir",
        str(case_dir),
        "--data_dir",
        str(args.data_dir),
        "--sim_validate_traj_time_budget_s",
        str(args.sim_validate_traj_time_budget_s),
    ]
    if args.factorized_search_preset is not None:
        cmd += ["--factorized-search-preset", str(args.factorized_search_preset)]
    if args.factorized_two_block_shared_coord is not None:
        cmd += ["--factorized-two-block-shared-coord", str(args.factorized_two_block_shared_coord)]
    if args.fast:
        cmd.append("--fast")
    if use_skip_generate:
        cmd.append("--skip_generate")
    if args.verbose:
        cmd.append("--verbose")
    if not args.sim_validate_progress:
        cmd.append("--no_sim_validate_progress")
    return cmd


def _load_problem_result(summary_path: Path) -> tuple[str | None, str | None, str | None, str | None, str | None, bool | None, str | None, list[str], list[str]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    problems = list(data.get("problems", []))
    row = problems[0] if problems else {}
    return (
        row.get("status"),
        row.get("message"),
        row.get("failure_kind"),
        row.get("selected_engine"),
        row.get("first_line_status"),
        row.get("rescued_additional"),
        row.get("canonical_equation"),
        list(row.get("fit_trajectories", []) or []),
        list(row.get("probe_trajectories", []) or []),
    )


def _run_case(args: argparse.Namespace, problem_id: str) -> CaseOutcome:
    case_dir = Path(args.results_root) / f"de{problem_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    summary_path = case_dir / "summary.json"
    launcher_log = case_dir / f"de{problem_id}_launcher.log"

    if args.resume and summary_path.exists():
        status, message, failure_kind, selected_engine, first_line_status, rescued_additional, canonical_equation, fit_traj, probe_traj = _load_problem_result(summary_path)
        return CaseOutcome(
            problem_id=problem_id,
            returncode=0,
            case_dir=str(case_dir),
            launcher_log=str(launcher_log),
            summary_path=str(summary_path),
            status=status,
            message=message,
            failure_kind=failure_kind,
            selected_engine=selected_engine,
            first_line_status=first_line_status,
            rescued_additional=bool(rescued_additional) if rescued_additional is not None else None,
            canonical_equation=canonical_equation,
            fit_trajectories=fit_traj,
            probe_trajectories=probe_traj,
            skipped_existing=True,
        )

    cmd = _build_case_command(args, problem_id, case_dir)
    with launcher_log.open("a", encoding="utf-8") as log:
        if bool(args.skip_generate) and "--skip_generate" not in cmd:
            log.write(
                f"# Missing cached trajectories for de{problem_id} at n_traj={int(args.n_traj)}; regenerating.\n"
            )
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=_subprocess_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    status = None
    message = None
    failure_kind = None
    selected_engine = None
    first_line_status = None
    rescued_additional = None
    canonical_equation = None
    fit_traj: list[str] = []
    probe_traj: list[str] = []
    if summary_path.exists():
        (
            status,
            message,
            failure_kind,
            selected_engine,
            first_line_status,
            rescued_additional,
            canonical_equation,
            fit_traj,
            probe_traj,
        ) = _load_problem_result(summary_path)

    return CaseOutcome(
        problem_id=problem_id,
        returncode=int(proc.returncode),
        case_dir=str(case_dir),
        launcher_log=str(launcher_log),
        summary_path=str(summary_path) if summary_path.exists() else None,
        status=status,
        message=message,
        failure_kind=failure_kind,
        selected_engine=selected_engine,
        first_line_status=first_line_status,
        rescued_additional=bool(rescued_additional) if rescued_additional is not None else None,
        canonical_equation=canonical_equation,
        fit_trajectories=fit_traj,
        probe_trajectories=probe_traj,
    )


def _aggregate(outcomes: list[CaseOutcome], args: argparse.Namespace) -> dict[str, Any]:
    counts: dict[str, int] = {}
    engine_counts: dict[str, int] = {}
    sparse_only_pass = 0
    rescued_additional = 0
    final_pass = 0
    rescued_ids: list[str] = []
    failure_kind_counts: dict[str, int] = {}

    for row in outcomes:
        status = str(row.status or ("ERROR" if row.returncode else "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
        if row.failure_kind:
            failure_kind_counts[str(row.failure_kind)] = failure_kind_counts.get(str(row.failure_kind), 0) + 1
        engine = str(row.selected_engine or "")
        if engine:
            engine_counts[engine] = engine_counts.get(engine, 0) + 1
        if row.first_line_status == "PASS":
            sparse_only_pass += 1
        if row.status == "PASS":
            final_pass += 1
        if bool(row.rescued_additional):
            rescued_additional += 1
            rescued_ids.append(row.problem_id)

    suite_summary = {
        "sparse_only_pass": int(sparse_only_pass),
        "rescued_additional": int(rescued_additional),
        "final_pass": int(final_pass),
        "rescued_problem_ids": rescued_ids,
        "failure_kind_counts": failure_kind_counts,
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runner": str(RUNNER),
        "benchmark_file": str(args.benchmark_file),
        "engine": str(args.engine),
        "ids": list(args.ids),
        "jobs": int(args.jobs),
        "fast": bool(args.fast),
        "n_traj": int(args.n_traj),
        "n_points": int(args.n_points),
        "holdout_last_k": int(args.holdout_last_k),
        "sim_validate_traj_time_budget_s": float(args.sim_validate_traj_time_budget_s),
        "data_dir": str(args.data_dir),
        "results_root": str(args.results_root),
        "counts": counts,
        "failure_kind_counts": failure_kind_counts,
        "selected_engine_counts": engine_counts,
        "suite_summary": suite_summary,
        "hybrid_summary": suite_summary,
        "cases": [asdict(row) for row in outcomes],
    }


def _print_case_line(row: CaseOutcome) -> None:
    state = "resume" if row.skipped_existing else "done"
    status = row.status or f"rc={row.returncode}"
    engine = row.selected_engine or "?"
    first_line = row.first_line_status or "?"
    suffix = f" failure_kind={row.failure_kind}" if row.failure_kind else ""
    print(
        f"[{state}] de{row.problem_id} status={status} "
        f"first_line={first_line} selected={engine}{suffix}"
    )
    if row.message:
        print(f"        {row.message}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch the compositional Feynman-DE mini-benchmark.",
    )
    p.add_argument(
        "--ids",
        default=",".join(DEFAULT_IDS),
        help="Comma-separated problem IDs. Defaults to the full compositional suite.",
    )
    p.add_argument(
        "--engine",
        choices=["factorized_search_only", "factorized_de", "hybrid", "sparse", "factorized_search_oracle", "compare"],
        default="factorized_search_only",
        help="Benchmark engine to run for each compositional case.",
    )
    p.add_argument("--jobs", type=int, default=2, help="Parallel case processes to run at once.")
    p.add_argument(
        "--benchmark_file",
        type=Path,
        default=BENCHMARK_FILE,
        help="Benchmark definition file.",
    )
    p.add_argument(
        "--results_root",
        type=Path,
        default=_default_results_root(),
        help="Root directory for per-case results.",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        default=REPO_ROOT / "data" / "feynman_de_compositional",
        help="Directory for generated trajectory CSVs.",
    )
    p.add_argument("--n_traj", type=int, default=6, help="Trajectories per case.")
    p.add_argument("--n_points", type=int, default=5000, help="Points per generated trajectory.")
    p.add_argument("--holdout_last_k", type=int, default=2, help="Number of held-out probe trajectories.")
    p.add_argument("--sim_validate_traj_time_budget_s", type=float, default=120.0)
    p.add_argument(
        "--factorized-search-preset",
        choices=["fast", "default", "paper", "compositional", "compositional_fast"],
        default=None,
        help="Optional factorized symbolic search preset override forwarded to run_benchmark.py.",
    )
    p.add_argument(
        "--factorized-two-block-shared-coord",
        choices=["never", "auto", "always"],
        default=None,
        help="Optional shared-coordinate two-block factorized rescue mode forwarded to run_benchmark.py.",
    )
    p.add_argument("--fast", action="store_true", help="Use the reduced fast budget in run_benchmark.py.")
    p.add_argument("--skip_generate", action="store_true", help="Reuse trajectories in data_dir instead of generating.")
    p.add_argument("--resume", action="store_true", help="Skip cases whose summary.json already exists.")
    p.add_argument(
        "--sim_validate_progress",
        action="store_true",
        help="Show verbose per-candidate simulation validation inside child logs.",
    )
    p.add_argument("--verbose", action="store_true", help="Pass --verbose to run_benchmark.py.")
    p.add_argument("--dry_run", action="store_true", help="Print commands and exit without running.")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.ids = _parse_ids(str(args.ids))
    args.jobs = max(1, int(args.jobs))
    args.n_traj = max(2, int(args.n_traj))
    args.n_points = max(128, int(args.n_points))
    args.holdout_last_k = max(1, int(args.holdout_last_k))
    if args.holdout_last_k >= args.n_traj:
        parser.error("--holdout_last_k must be smaller than --n_traj")
    min_points = 2048 if bool(args.fast) else 4000
    if args.n_points < int(min_points):
        print(
            f"Requested n_points={args.n_points} is too small for the current "
            f"{'fast' if args.fast else 'default'} run_de split budget; "
            f"using n_points={min_points} instead."
        )
        args.n_points = int(min_points)

    args.results_root.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_file": str(args.benchmark_file),
        "engine": str(args.engine),
        "ids": list(args.ids),
        "jobs": int(args.jobs),
        "fast": bool(args.fast),
        "n_traj": int(args.n_traj),
        "n_points": int(args.n_points),
        "holdout_last_k": int(args.holdout_last_k),
        "sim_validate_traj_time_budget_s": float(args.sim_validate_traj_time_budget_s),
        "data_dir": str(args.data_dir),
        "results_root": str(args.results_root),
    }
    (args.results_root / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        for pid in args.ids:
            cmd = _build_case_command(args, pid, Path(args.results_root) / f"de{pid}")
            print("$", " ".join(cmd))
        return 0

    print(f"Results root: {args.results_root}")
    print(f"Benchmark file: {args.benchmark_file}")
    print(f"Data dir: {args.data_dir}")
    print(
        f"Running {len(args.ids)} compositional cases with engine={args.engine} "
        f"jobs={args.jobs} fast={args.fast} n_points={args.n_points}"
    )

    outcomes: list[CaseOutcome] = []
    with ThreadPoolExecutor(max_workers=int(args.jobs)) as pool:
        fut_to_id = {pool.submit(_run_case, args, pid): pid for pid in args.ids}
        for fut in as_completed(fut_to_id):
            row = fut.result()
            outcomes.append(row)
            _print_case_line(row)

    outcomes.sort(key=lambda row: row.problem_id)
    payload = _aggregate(outcomes, args)
    out_path = Path(args.results_root) / "compositional_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    hs = payload["suite_summary"]
    print("\nSuite summary")
    print(
        f"  sparse-only pass: {hs['sparse_only_pass']} | "
        f"rescued additional: {hs['rescued_additional']} | "
        f"final pass: {hs['final_pass']}"
    )
    if payload["failure_kind_counts"]:
        parts = [f"{k}: {v}" for k, v in sorted(payload["failure_kind_counts"].items())]
        print(f"  failure kinds: {' | '.join(parts)}")
    if payload["selected_engine_counts"]:
        print(f"  selected engines: {payload['selected_engine_counts']}")
    print(f"Summary JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
