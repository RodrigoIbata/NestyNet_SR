#!/usr/bin/env python3
"""Run cheap cached DE127 GS ablations and summarize them.

This script is intended to be executed inside the repository Apptainer image.
It reuses an existing surrogate_cache, runs STLSQ-only DE discovery for de127,
and writes Markdown/CSV tables with the validation outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "results" / "feynman_de" / "de127_gs_ablation_cached"
DEFAULT_CACHE = (
    REPO
    / "results"
    / "feynman_de"
    / "de127_stlsq_oldgs_qdag_cached"
    / "surrogate_cache"
)

COMMON = [
    "--engine",
    "stlsq",
    "--only",
    "127",
    "--skip_generate",
    "--expr-ir",
    "qdag",
    "--expr-canonicalize",
    "safe",
    "--expr-report",
    "--expr-deep-enable",
    "--expr-deep-max-depth",
    "7",
    "--verbose",
]

OLD_GS = [
    "--gs-enable",
    "--gs-mode",
    "auto",
    "--gs-policy",
    "replace-shadowed",
    "--gs-known-generators",
    "--gs-general-affine",
    "--gs-lorentz-boosts",
    "--de-hard-tail-templates",
    "--de-hard-tail-velocity-templates",
    "--gs-de-lie-prolongation",
    "--gs-de-lie-prolongation-weight",
    "1.0",
    "--gs-dim-policy",
    "audit",
]


@dataclass(frozen=True)
class Case:
    case_id: str
    hypothesis: str
    gs_args: tuple[str, ...]


CASES = [
    Case("00_no_gs", "Plain STLSQ/QDAG baseline on current cached priors.", ("--gs-mode", "off", "--gs-no-known-generators")),
    Case(
        "01_gs_audit_only",
        "Diagnostics only; should not change sparse library.",
        (
            "--gs-enable",
            "--gs-mode",
            "audit",
            "--gs-policy",
            "augment",
            "--gs-known-generators",
        ),
    ),
    Case(
        "02_known_only",
        "Known generators enabled, but no DE templates.",
        (
            "--gs-enable",
            "--gs-mode",
            "auto",
            "--gs-policy",
            "augment",
            "--gs-known-generators",
        ),
    ),
    Case(
        "03_radial_augment",
        "Add neutral radial/singular hard-tail priors by augmentation.",
        (
            "--gs-enable",
            "--gs-mode",
            "auto",
            "--gs-policy",
            "augment",
            "--gs-known-generators",
            "--de-hard-tail-templates",
        ),
    ),
    Case(
        "04_radial_replace",
        "Replace shadowed baseline radial motifs while adding neutral hard-tail priors.",
        (
            "--gs-enable",
            "--gs-mode",
            "auto",
            "--gs-policy",
            "replace-shadowed",
            "--gs-known-generators",
            "--de-hard-tail-templates",
        ),
    ),
    Case(
        "05_radial_velocity_replace",
        "Old policy shape, but without affine/Lorentz/Lie scoring.",
        (
            "--gs-enable",
            "--gs-mode",
            "auto",
            "--gs-policy",
            "replace-shadowed",
            "--gs-known-generators",
            "--de-hard-tail-templates",
            "--de-hard-tail-velocity-templates",
        ),
    ),
    Case(
        "06_old_no_general_affine",
        "Old GS minus learned general-affine.",
        tuple(a for a in OLD_GS if a != "--gs-general-affine"),
    ),
    Case(
        "07_old_no_lorentz",
        "Old GS minus Lorentz/hyperbolic generators.",
        tuple(a for a in OLD_GS if a != "--gs-lorentz-boosts"),
    ),
    Case(
        "08_old_no_lie_score",
        "Old GS minus Lie-prolongation candidate scoring.",
        tuple(
            a
            for i, a in enumerate(OLD_GS)
            if a not in {"--gs-de-lie-prolongation", "--gs-de-lie-prolongation-weight"}
            and not (
                i > 0
                and OLD_GS[i - 1] == "--gs-de-lie-prolongation-weight"
            )
        ),
    ),
    Case("09_old_profile", "Prior V4 GS full-fix profile plus QDAG.", tuple(OLD_GS)),
    Case(
        "10_old_plus_unit_audit",
        "Old GS plus unit-torus audit only.",
        tuple(OLD_GS + ["--gs-unit-torus", "--gs-dim-policy", "audit"]),
    ),
    Case(
        "11_old_plus_dim_augment",
        "Old GS plus unit-torus proposals.",
        tuple(OLD_GS + ["--gs-unit-torus", "--gs-dim-policy", "augment"]),
    ),
    Case(
        "12_old_plus_pi_augment",
        "Old GS plus Buckingham-pi invariant proposals.",
        tuple(
            OLD_GS
            + [
                "--gs-unit-torus",
                "--gs-pi-invariants",
                "--gs-dim-policy",
                "augment",
            ]
        ),
    ),
    Case(
        "13_old_plus_contact",
        "Old GS plus contact/jet-space velocity templates.",
        tuple(OLD_GS + ["--gs-de-contact-templates"]),
    ),
    Case(
        "14_old_plus_noether",
        "Old GS plus Noether/variational templates.",
        tuple(OLD_GS + ["--gs-de-noether-templates"]),
    ),
    Case(
        "15_old_plus_discrete",
        "Old GS plus parity/time-reversal templates.",
        tuple(OLD_GS + ["--gs-de-discrete-symmetry-templates"]),
    ),
    Case(
        "16_old_plus_weighted",
        "Old GS plus quasi-homogeneous weighted-scaling templates.",
        tuple(OLD_GS + ["--gs-de-weighted-scaling-templates"]),
    ),
    Case(
        "17_old_plus_radial_reduction",
        "Old GS plus systematic radial-reduction templates.",
        tuple(OLD_GS + ["--gs-de-radial-reduction-templates"]),
    ),
    Case(
        "18_old_plus_determining",
        "Old GS plus bounded determining-equation diagnostics.",
        tuple(OLD_GS + ["--gs-de-determining-equations"]),
    ),
    Case(
        "19_old_plus_all_upgrades",
        "Old GS plus all bounded upgrade families.",
        tuple(OLD_GS + ["--gs-de-all-upgrades"]),
    ),
]


def _copy_cache(cache_src: Path, out_dir: Path) -> None:
    dst = out_dir / "surrogate_cache"
    if dst.exists():
        return
    if not cache_src.exists():
        raise FileNotFoundError(f"cache source not found: {cache_src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_src, dst)


def _summary_value(summary: dict, *keys: str):
    cur = summary
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _load_case(case: Case, out_dir: Path, skipped: bool) -> dict:
    summary_path = out_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
    selected = _summary_value(summary, "equations", "de127", "selected") or {}
    scores = selected.get("scores") if isinstance(selected, dict) else None
    scores = scores if isinstance(scores, list) else []
    finite = [
        float(row.get("nrmse"))
        for row in scores
        if isinstance(row, dict)
        and row.get("nrmse") not in (None, "inf")
        and row.get("nrmse") != float("inf")
    ]
    mean_nrmse = summary.get("nrmse_mean")
    max_nrmse = summary.get("nrmse_max")
    if mean_nrmse is None and finite:
        mean_nrmse = sum(finite) / len(finite)
    if max_nrmse is None and finite:
        max_nrmse = max(finite)
    return {
        "case": case.case_id,
        "status": summary.get("status", "MISSING"),
        "failure_kind": summary.get("failure_kind", ""),
        "nrmse_mean": "" if mean_nrmse is None else f"{float(mean_nrmse):.6g}",
        "nrmse_max": "" if max_nrmse is None else f"{float(max_nrmse):.6g}",
        "message": summary.get("message", ""),
        "equation": selected.get("canonical_equation") or selected.get("equation") or "",
        "skipped_existing": str(bool(skipped)),
        "summary": str(summary_path),
        "hypothesis": case.hypothesis,
        "flags": " ".join(case.gs_args),
    }


def _run_case(case: Case, root: Path, cache_src: Path, force: bool) -> dict:
    out_dir = root / case.case_id
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        return _load_case(case, out_dir, skipped=True)
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    _copy_cache(cache_src, out_dir)
    cmd = [
        sys.executable,
        "-u",
        str(REPO / "examples" / "feynman_de" / "run_benchmark.py"),
        *COMMON,
        "--results_dir",
        str(out_dir),
        *case.gs_args,
    ]
    log_path = out_dir / "ablation_runner.log"
    print(f"[ablation] {case.case_id}: {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    py_paths = [str(REPO), str(REPO.parent / "NestyNet")]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    row = _load_case(case, out_dir, skipped=False)
    row["returncode"] = str(proc.returncode)
    row["log"] = str(log_path)
    return row


def _write_outputs(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "ablation_results.csv"
    fields = [
        "case",
        "status",
        "failure_kind",
        "nrmse_mean",
        "nrmse_max",
        "message",
        "equation",
        "summary",
        "hypothesis",
        "flags",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    md_path = root / "ablation_table.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# DE127 Cached GS Ablation\n\n")
        fh.write("| Case | Status | Failure | Mean NRMSE | Max NRMSE | Equation |\n")
        fh.write("|---|---:|---|---:|---:|---|\n")
        for row in rows:
            eqn = str(row.get("equation", "")).replace("|", "\\|")
            fh.write(
                f"| {row['case']} | {row['status']} | {row['failure_kind']} | "
                f"{row['nrmse_mean']} | {row['nrmse_max']} | `{eqn}` |\n"
            )
        fh.write("\n## Switch Matrix\n\n")
        fh.write("| Case | Hypothesis | Flags |\n")
        fh.write("|---|---|---|\n")
        for row in rows:
            flags = str(row.get("flags", "")).replace("|", "\\|")
            hypo = str(row.get("hypothesis", "")).replace("|", "\\|")
            fh.write(f"| {row['case']} | {hypo} | `{flags}` |\n")
    print(f"[ablation] wrote {csv_path}")
    print(f"[ablation] wrote {md_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--cache-src", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()
    selected = set(args.cases or [])
    cases = [case for case in CASES if not selected or case.case_id in selected]
    rows = [_run_case(case, args.root, args.cache_src, bool(args.force)) for case in cases]
    _write_outputs(args.root, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
