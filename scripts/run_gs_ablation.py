#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Run baseline-vs-GS ablation pairs for NestyNet_SR_GS examples.

This is intentionally a thin orchestration layer.  It does not know how to score
all possible experiments.  It runs the same command twice, once unchanged and
once with generalized-symmetry arguments appended, captures logs, and writes a
manifest that downstream summarizers can inspect.

Typical usage from an example directory:

    ./run_gs_ablation.sh --dry-run
    ./run_gs_ablation.sh --fast --only 031,032

For custom experiments:

    python scripts/run_gs_ablation.py --suite custom \
      --cmd "python -m nestynet_sr.run_SR --filepath data.csv --fast" \
      --gs-args "--gs-auto"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_suite_command(suite: str) -> tuple[list[str] | None, str, str]:
    """Return (cmd, gs_args, result_arg_style)."""

    py = sys.executable
    suite = str(suite)
    # result_arg_style: results_dir | output_dir | none
    registry: dict[str, tuple[list[str], str, str]] = {
        "classSR": (
            [
                py, "-m", "nestynet_sr.run_SR",
                "--filepath", "examples/classSR/data/quad_1.csv",
                "--factorized-search",
                "--y_units", "[1,0]",
                "--x_units", "[[0,1]]",
                "--units_basis", "L,T",
                "--local_consts", '{"c0":[1,0], "c1":[1,-1]}',
                "--global_consts", '{"g0":[1,-2]}',
                "--batch_size", "256", "--ndata_train", "256", "--ndata_val", "256",
                "--log_level", "INFO",
            ],
            "--gs-auto",
            "results_dir",
        ),
        "feynman_de": (
            [py, "examples/feynman_de/run_benchmark.py", "--engine", "sparse", "--fast", "--only", "031,032,033,034,038,043"],
            "--gs-enable --de-hard-tail-templates",
            "results_dir",
        ),
        "lane_emden": (
            [py, "examples/lane_emden/smoke_lane_emden_discovery.py", "--generate", "--epochs", "80"],
            "--gs-auto",
            "output_dir",
        ),
        "logistic_growth": (
            [py, "examples/logistic_growth/smoke_logistic_discovery.py", "--generate", "--epochs", "80"],
            "--gs-auto",
            "output_dir",
        ),
        "dho": (
            [py, "examples/dho/smoke_dho_discovery_sr.py", "--generate"],
            "--gs-auto",
            "results_dir",
        ),
        "special_relativity": (
            [py, "examples/special_relativity/run_class_sr_discovery.py", "--fast", "--generate"],
            "--gs-auto",
            "results_dir",
        ),
        "feynman_complex": (
            [py, "examples/feynman_complex/run_benchmark.py", "--fast", "--only", "free_schrodinger,complex_oscillator"],
            "",
            "results_dir",
        ),
        "Maxwell": (
            [py, "examples/Maxwell/run_benchmark.py", "--fast"],
            "",
            "results_dir",
        ),
        "MOND": (
            [py, "examples/MOND/run_benchmark.py", "--fast"],
            "",
            "results_dir",
        ),
        "hamiltonian": (
            [py, "examples/hamiltonian/anharmonic_oscillator.py"],
            "",
            "results_dir",
        ),
        "kepler_ephemeris_real": (
            [py, "examples/kepler_ephemeris_real/smoke_kepler_discovery.py", "--profile", "clean", "--generate"],
            "",
            "results_dir",
        ),
        "multi_dataset": (
            [py, "examples/multi_dataset/smoke_multi_logistic.py"],
            "--gs-auto",
            "results_dir",
        ),
        "oracle_factorized_search": (
            [py, "examples/oracle_factorized_search/run_aif_closure_benchmark.py", "--fast"],
            "",
            "results_dir",
        ),
    }
    return registry.get(suite, (None, "--gs-auto", "results_dir"))


def _default_pre_command(suite: str) -> list[str] | None:
    py = sys.executable
    registry = {
        "classSR": [py, "examples/classSR/generate_quadratic.py"],
    }
    return registry.get(str(suite))


def _has_flag(cmd: Sequence[str], *flags: str) -> bool:
    return any(tok in set(flags) for tok in cmd)


def _inject_result_dir(cmd: list[str], style: str, out_dir: Path) -> list[str]:
    if str(style) == "none":
        return list(cmd)
    flag = "--output_dir" if str(style) == "output_dir" else "--results_dir"
    if _has_flag(cmd, flag):
        return list(cmd)
    return list(cmd) + [flag, str(out_dir)]


def _run_one(label: str, cmd: list[str], *, cwd: Path, dry_run: bool, env: dict[str, str], log_path: Path) -> dict:
    record = {"label": label, "cmd": cmd, "log_path": str(log_path), "returncode": None, "seconds": 0.0}
    print("\n" + "=" * 80)
    print(f"[{label}] {' '.join(shlex.quote(x) for x in cmd)}")
    print("=" * 80)
    if dry_run:
        record["returncode"] = 0
        return record
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="")
        proc.wait()
    record["seconds"] = time.time() - t0
    record["returncode"] = int(proc.returncode)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--suite", default="custom", help="Experiment suite name; used for default command lookup and output paths")
    ap.add_argument("--cmd", default=None, help="Base command to run twice. Use quotes. If omitted, a small suite-specific default is used when available.")
    ap.add_argument("--pre-cmd", default=None, help="Optional preparation command run before both branches")
    ap.add_argument("--gs-args", default=None, help="GS arguments appended to the GS branch. Empty string means no GS args for this suite.")
    ap.add_argument("--result-arg-style", choices=["auto", "results_dir", "output_dir", "none"], default="auto", help="How to inject per-branch output directory")
    ap.add_argument("--results-root", default=None, help="Root directory for ablation outputs")
    ap.add_argument("--label-baseline", default="baseline")
    ap.add_argument("--label-gs", default="gs_auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--", dest="ignored", nargs="*")
    args, unknown = ap.parse_known_args(argv)

    default_cmd, default_gs_args, default_style = _default_suite_command(args.suite)
    if args.cmd is not None:
        base_cmd = shlex.split(args.cmd)
    elif default_cmd is not None:
        base_cmd = list(default_cmd)
    else:
        print(
            f"No default command is registered for suite {args.suite!r}. "
            "Pass --cmd 'python ...' and optionally --gs-args '--gs-auto'.",
            file=sys.stderr,
        )
        return 2

    # Unknown args are appended to both branches so wrappers can forward suite-specific knobs.
    if unknown:
        base_cmd += list(unknown)

    gs_args = default_gs_args if args.gs_args is None else str(args.gs_args)
    gs_argv = shlex.split(gs_args) if gs_args.strip() else []
    style = default_style if args.result_arg_style == "auto" else args.result_arg_style

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_root = Path(args.results_root or (REPO_ROOT / "results" / "gs_ablations" / str(args.suite) / timestamp)).resolve()
    baseline_dir = results_root / args.label_baseline
    gs_dir = results_root / args.label_gs

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONPATH", str(REPO_ROOT))

    pre = shlex.split(args.pre_cmd) if args.pre_cmd else _default_pre_command(args.suite)
    if pre:
        print("[pre]", " ".join(shlex.quote(x) for x in pre))
        if not args.dry_run:
            subprocess.check_call(pre, cwd=str(REPO_ROOT), env=env)

    cmd_baseline = _inject_result_dir(base_cmd, style, baseline_dir)
    cmd_gs = _inject_result_dir(base_cmd + gs_argv, style, gs_dir)

    manifest = {
        "suite": args.suite,
        "created": timestamp,
        "results_root": str(results_root),
        "baseline_dir": str(baseline_dir),
        "gs_dir": str(gs_dir),
        "gs_args": gs_argv,
        "commands": {},
        "records": [],
    }
    results_root.mkdir(parents=True, exist_ok=True)
    manifest["commands"]["baseline"] = cmd_baseline
    manifest["commands"]["gs"] = cmd_gs

    rec_base = _run_one(args.label_baseline, cmd_baseline, cwd=REPO_ROOT, dry_run=args.dry_run, env=env, log_path=results_root / f"{args.label_baseline}.log")
    manifest["records"].append(rec_base)
    if int(rec_base["returncode"] or 0) != 0 and not args.continue_on_error:
        (results_root / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return int(rec_base["returncode"])

    rec_gs = _run_one(args.label_gs, cmd_gs, cwd=REPO_ROOT, dry_run=args.dry_run, env=env, log_path=results_root / f"{args.label_gs}.log")
    manifest["records"].append(rec_gs)
    manifest["gs_reports"] = [str(p) for p in gs_dir.rglob("*.gs_report.json")]
    (results_root / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nAblation manifest: {results_root / 'ablation_manifest.json'}")
    if manifest["gs_reports"]:
        print("GS reports:")
        for p in manifest["gs_reports"]:
            print(f"  {p}")
    if int(rec_gs["returncode"] or 0) != 0:
        return int(rec_gs["returncode"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
