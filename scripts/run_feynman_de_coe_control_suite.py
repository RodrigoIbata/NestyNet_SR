#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Launch the Feynman-DE CoE control suite.

This is a behavior-preserving PR0 harness for the planned DE Committee of
Experts upgrade. It runs a small fixed set of DE benchmark cases through the
existing ``examples/feynman_de/run_benchmark.py`` entry point and records the
current selected/internal engine choices, rollout overrides, candidate
validation counts, trajectory NRMSE summaries, and wall times.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "examples" / "feynman_de" / "run_benchmark.py"
BENCHMARK_FILE = REPO_ROOT / "data" / "feynman_de_benchmark.txt"
DEFAULT_IDS = ("002", "010", "100", "103", "114", "119", "121", "131")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    internal_selected_engine: str | None
    selected_lane: str | None
    typed_selected_lane: str | None
    whole_rhs_attempted: bool | None
    whole_rhs_attempts_run: int | None
    family_gate_skips: int | None
    typed_explorer_launches: int | None
    first_line_status: str | None
    rollout_override: bool | None
    rescued_additional: bool | None
    canonical_equation: str | None
    fit_trajectories: list[str]
    probe_trajectories: list[str]
    wall_time_s: float
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
    return REPO_ROOT / "results" / f"feynman_de_coe_control_{stamp}"


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
        "--sim_validate_max_candidates",
        str(args.sim_validate_max_candidates),
        "--sim_validate_traj_time_budget_s",
        str(args.sim_validate_traj_time_budget_s),
        "--pass_nrmse",
        str(args.pass_nrmse),
        "--partial_nrmse",
        str(args.partial_nrmse),
    ]
    if args.factorized_search_preset is not None:
        cmd += ["--factorized-search-preset", str(args.factorized_search_preset)]
    if args.factorized_two_block_shared_coord is not None:
        cmd += ["--factorized-two-block-shared-coord", str(args.factorized_two_block_shared_coord)]
    if str(args.de_coe_mode or "off") != "off":
        cmd += ["--de-coe-mode", str(args.de_coe_mode)]
    if int(args.de_coe_reservoir_scouts) > 0:
        cmd += ["--de-coe-reservoir-scouts", str(args.de_coe_reservoir_scouts)]
    if bool(args.de_coe_csr_on_ties):
        cmd.append("--de-coe-csr-on-ties")
    if args.factorized_de_whole_rhs is not None:
        cmd += ["--factorized-de-whole-rhs", str(args.factorized_de_whole_rhs)]
    if args.factorized_search_de_refine_mode is not None:
        cmd += ["--factorized-search-de-refine-mode", str(args.factorized_search_de_refine_mode)]
    if args.factorized_search_max_attempts is not None:
        cmd += ["--factorized-search-max-attempts", str(max(0, int(args.factorized_search_max_attempts)))]
    if bool(args.fast):
        cmd.append("--fast")
    if use_skip_generate:
        cmd.append("--skip_generate")
    if args.verbose:
        cmd.append("--verbose")
    if not args.sim_validate_progress:
        cmd.append("--no_sim_validate_progress")
    return cmd


def _load_problem_result(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    problems = list(data.get("problems", []) or [])
    return dict(problems[0]) if problems and isinstance(problems[0], dict) else {}


def _rollout_override(row: dict[str, Any]) -> bool | None:
    selected = str(row.get("selected_engine", "") or "")
    internal = str(row.get("internal_selected_engine", "") or "")
    if not selected or not internal:
        return None
    return bool(selected != internal)


def _primary_engine_row(row: dict[str, Any]) -> dict[str, Any]:
    engines = row.get("engines", {})
    if not isinstance(engines, dict):
        return {}
    for key in (str(row.get("engine", "")), "factorized_de", "hybrid", "factorized_search_only", "sparse"):
        payload = engines.get(key)
        if isinstance(payload, dict):
            return payload
    return {}


def _field_from_problem_or_primary(row: dict[str, Any], primary: dict[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    return primary.get(key)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _outcome_from_summary(
    *,
    problem_id: str,
    returncode: int,
    case_dir: Path,
    launcher_log: Path,
    summary_path: Path | None,
    wall_time_s: float,
    skipped_existing: bool = False,
) -> CaseOutcome:
    row: dict[str, Any] = {}
    if summary_path is not None and summary_path.exists():
        row = _load_problem_result(summary_path)
    primary = _primary_engine_row(row)
    return CaseOutcome(
        problem_id=str(problem_id),
        returncode=int(returncode),
        case_dir=str(case_dir),
        launcher_log=str(launcher_log),
        summary_path=str(summary_path) if summary_path is not None and summary_path.exists() else None,
        status=row.get("status"),
        message=row.get("message"),
        failure_kind=row.get("failure_kind"),
        selected_engine=row.get("selected_engine"),
        internal_selected_engine=row.get("internal_selected_engine"),
        selected_lane=_field_from_problem_or_primary(row, primary, "selected_lane"),
        typed_selected_lane=_field_from_problem_or_primary(row, primary, "typed_selected_lane"),
        whole_rhs_attempted=(
            None
            if _field_from_problem_or_primary(row, primary, "whole_rhs_attempted") is None
            else bool(_field_from_problem_or_primary(row, primary, "whole_rhs_attempted"))
        ),
        whole_rhs_attempts_run=_optional_int(
            _field_from_problem_or_primary(row, primary, "whole_rhs_attempts_run")
        ),
        family_gate_skips=_optional_int(_field_from_problem_or_primary(row, primary, "family_gate_skips")),
        typed_explorer_launches=_optional_int(
            _field_from_problem_or_primary(row, primary, "typed_explorer_launches")
        ),
        first_line_status=row.get("first_line_status"),
        rollout_override=_rollout_override(row),
        rescued_additional=bool(row.get("rescued_additional")) if "rescued_additional" in row else None,
        canonical_equation=row.get("canonical_equation"),
        fit_trajectories=list(row.get("fit_trajectories", []) or []),
        probe_trajectories=list(row.get("probe_trajectories", []) or []),
        wall_time_s=float(wall_time_s),
        skipped_existing=bool(skipped_existing),
    )


def _run_case(args: argparse.Namespace, problem_id: str) -> CaseOutcome:
    case_dir = Path(args.results_root) / f"de{problem_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    summary_path = case_dir / "summary.json"
    launcher_log = case_dir / f"de{problem_id}_launcher.log"

    if args.resume and summary_path.exists():
        return _outcome_from_summary(
            problem_id=problem_id,
            returncode=0,
            case_dir=case_dir,
            launcher_log=launcher_log,
            summary_path=summary_path,
            wall_time_s=0.0,
            skipped_existing=True,
        )

    cmd = _build_case_command(args, problem_id, case_dir)
    started = time.perf_counter()
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
    elapsed = float(time.perf_counter() - started)
    return _outcome_from_summary(
        problem_id=problem_id,
        returncode=int(proc.returncode),
        case_dir=case_dir,
        launcher_log=launcher_log,
        summary_path=summary_path if summary_path.exists() else None,
        wall_time_s=elapsed,
    )


def _load_control_summary(summary_paths: list[Path]) -> dict[str, Any]:
    from scripts.summarize_feynman_de_coe_control import summarize

    return summarize(summary_paths)


def _aggregate(outcomes: list[CaseOutcome], args: argparse.Namespace) -> dict[str, Any]:
    summary_paths = [
        Path(row.summary_path)
        for row in outcomes
        if row.summary_path is not None and Path(row.summary_path).exists()
    ]
    control_summary = _load_control_summary(summary_paths) if summary_paths else {
        "n_reports": 0,
        "n_rows": 0,
        "rows": [],
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
        "sim_validate_max_candidates": int(args.sim_validate_max_candidates),
        "sim_validate_traj_time_budget_s": float(args.sim_validate_traj_time_budget_s),
        "pass_nrmse": float(args.pass_nrmse),
        "partial_nrmse": float(args.partial_nrmse),
        "de_coe_mode": str(args.de_coe_mode),
        "de_coe_csr_on_ties": bool(args.de_coe_csr_on_ties),
        "de_coe_reservoir_scouts": int(args.de_coe_reservoir_scouts),
        "factorized_de_whole_rhs": str(args.factorized_de_whole_rhs),
        "factorized_search_de_refine_mode": str(args.factorized_search_de_refine_mode),
        "factorized_search_max_attempts": args.factorized_search_max_attempts,
        "data_dir": str(args.data_dir),
        "results_root": str(args.results_root),
        "control_summary": control_summary,
        "cases": [asdict(row) for row in outcomes],
    }


def _print_case_line(row: CaseOutcome) -> None:
    state = "resume" if row.skipped_existing else "done"
    status = row.status or f"rc={row.returncode}"
    selected = row.selected_engine or "?"
    internal = row.internal_selected_engine or "?"
    override = " override" if bool(row.rollout_override) else ""
    suffix = f" failure_kind={row.failure_kind}" if row.failure_kind else ""
    print(
        f"[{state}] de{row.problem_id} status={status} "
        f"internal={internal} selected={selected}{override}{suffix} "
        f"wall={row.wall_time_s:.2f}s"
    )
    if row.message:
        print(f"        {row.message}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        default=",".join(DEFAULT_IDS),
        help="Comma-separated problem IDs for the control suite.",
    )
    parser.add_argument(
        "--engine",
        choices=["factorized_de", "hybrid", "factorized_search_only", "sparse", "factorized_search_oracle", "compare"],
        default="factorized_de",
        help="Benchmark engine to run for each control case.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel case processes.")
    parser.add_argument(
        "--benchmark_file",
        type=Path,
        default=BENCHMARK_FILE,
        help="Benchmark definition file.",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=_default_results_root(),
        help="Root directory for per-case results.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=REPO_ROOT / "data" / "feynman_de_coe_control",
        help="Directory for generated trajectory CSVs.",
    )
    parser.add_argument("--n_traj", type=int, default=4, help="Trajectories per case.")
    parser.add_argument("--n_points", type=int, default=1500, help="Points per generated trajectory.")
    parser.add_argument("--holdout_last_k", type=int, default=1, help="Held-out probe trajectories.")
    parser.add_argument("--sim_validate_max_candidates", type=int, default=3)
    parser.add_argument("--sim_validate_traj_time_budget_s", type=float, default=20.0)
    parser.add_argument("--pass_nrmse", type=float, default=1.0e-2)
    parser.add_argument("--partial_nrmse", type=float, default=5.0e-2)
    parser.add_argument(
        "--factorized-search-preset",
        choices=["fast", "default", "paper", "compositional", "compositional_fast"],
        default=None,
        help="Optional factorized symbolic search preset override forwarded to run_benchmark.py.",
    )
    parser.add_argument(
        "--factorized-two-block-shared-coord",
        choices=["never", "auto", "always"],
        default=None,
        help="Optional shared-coordinate two-block factorized rescue mode forwarded to run_benchmark.py.",
    )
    parser.add_argument(
        "--de-coe-mode",
        choices=["off", "audit", "adjudicate", "reservoir"],
        default="off",
        help="Optional DE Committee-of-Experts mode forwarded to run_benchmark.py.",
    )
    parser.add_argument(
        "--de-coe-csr-on-ties",
        action="store_true",
        help="Forward --de-coe-csr-on-ties to run_benchmark.py.",
    )
    parser.add_argument(
        "--de-coe-reservoir-scouts",
        type=int,
        default=0,
        help="Forward bounded reservoir scout count to run_benchmark.py.",
    )
    parser.add_argument(
        "--factorized-de-whole-rhs",
        choices=["never", "auto", "always"],
        default=None,
        help="Optional whole-RHS FSS policy forwarded to run_benchmark.py.",
    )
    parser.add_argument(
        "--factorized-search-de-refine-mode",
        choices=["off", "rare_final_polish", "rare_slate"],
        default=None,
        help="Optional DE-facing FSS refinement mode forwarded to run_benchmark.py.",
    )
    parser.add_argument(
        "--factorized-search-max-attempts",
        type=int,
        default=None,
        help="Optional broad whole-RHS FSS max-attempt cap forwarded to run_benchmark.py.",
    )
    parser.set_defaults(fast=True)
    parser.add_argument("--fast", dest="fast", action="store_true", help="Use reduced run_benchmark.py budgets.")
    parser.add_argument("--full", dest="fast", action="store_false", help="Use non-fast run_benchmark.py budgets.")
    parser.add_argument("--skip_generate", action="store_true", help="Reuse trajectories in data_dir when complete.")
    parser.add_argument("--resume", action="store_true", help="Skip cases whose summary.json already exists.")
    parser.add_argument(
        "--sim_validate_progress",
        action="store_true",
        help="Show verbose per-candidate simulation validation inside child logs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Pass --verbose to run_benchmark.py.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and exit without running.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.ids = _parse_ids(str(args.ids))
    args.jobs = max(1, int(args.jobs))
    args.n_traj = max(2, int(args.n_traj))
    args.n_points = max(128, int(args.n_points))
    args.holdout_last_k = max(1, int(args.holdout_last_k))
    args.sim_validate_max_candidates = max(0, int(args.sim_validate_max_candidates))
    args.de_coe_reservoir_scouts = max(0, int(args.de_coe_reservoir_scouts))
    if args.holdout_last_k >= args.n_traj:
        parser.error("--holdout_last_k must be smaller than --n_traj")

    args.results_root.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
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
        "sim_validate_max_candidates": int(args.sim_validate_max_candidates),
        "sim_validate_traj_time_budget_s": float(args.sim_validate_traj_time_budget_s),
        "pass_nrmse": float(args.pass_nrmse),
        "partial_nrmse": float(args.partial_nrmse),
        "de_coe_mode": str(args.de_coe_mode),
        "de_coe_csr_on_ties": bool(args.de_coe_csr_on_ties),
        "de_coe_reservoir_scouts": int(args.de_coe_reservoir_scouts),
        "factorized_de_whole_rhs": str(args.factorized_de_whole_rhs),
        "factorized_search_de_refine_mode": str(args.factorized_search_de_refine_mode),
        "factorized_search_max_attempts": args.factorized_search_max_attempts,
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
        f"Running {len(args.ids)} DE-CoE control cases with engine={args.engine} "
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
    out_path = Path(args.results_root) / "de_coe_control_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    control = payload["control_summary"]
    print("\nDE-CoE control summary")
    print(
        f"  rows: {control.get('n_rows', 0)} | "
        f"PASS: {control.get('pass_count', 0)} | "
        f"rollout overrides: {control.get('rollout_override_count', 0)}"
    )
    if control.get("status_counts"):
        print(f"  status counts: {control['status_counts']}")
    if control.get("selected_engine_counts"):
        print(f"  selected engines: {control['selected_engine_counts']}")
    print(f"Summary JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
