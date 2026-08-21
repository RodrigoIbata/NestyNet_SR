#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Uniform baseline-vs-GS ablation runner for the examples tree.

The runner intentionally works as a thin orchestration layer.  It does not try to
reinterpret each example's success criterion; it executes the example twice with
controlled GS environment/flags, records commands, return codes, runtime, and any
GS reports produced by run_SR.py/run_de.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    command: tuple[str, ...]
    description: str
    gs_args: tuple[str, ...] = ()
    output_arg: str | None = None
    default_extra: tuple[str, ...] = ()
    note: str = ""


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "classSR": ExperimentSpec(
        "classSR",
        (PYTHON, "examples/classSR/smoke_quadratic_single.py"),
        "Single-dataset Class-SR smoke run through run_SR.py.",
    ),
    "core_acceptance_suites": ExperimentSpec(
        "core_acceptance_suites",
        (PYTHON, "nestynet_sr/run_core_acceptance_suite.py"),
        "Frozen core acceptance suite; child run_SR/run_de calls inherit GS env.",
        output_arg="--output_dir",
    ),
    "dho": ExperimentSpec(
        "dho",
        (PYTHON, "examples/dho/smoke_dho_discovery_sr.py"),
        "Damped harmonic oscillator first-class DE through run_SR.py.",
        output_arg="--results_dir",
    ),
    "feynman_de": ExperimentSpec(
        "feynman_de",
        (PYTHON, "examples/feynman_de/run_benchmark.py", "--engine", "sparse", "--only", "010,113,114,115,117,121,127,206", "--fast", "--skip_generate"),
        "Scalar DE hard-tail FAIL/PARTIAL subset from NestyNet III.",
        # No gs_args: run_benchmark.py registers no GS flags. The GS variant
        # reaches the child run_de.py processes through the inherited
        # NESTYNET_GS_ENABLE / NESTYNET_DE_HARD_TAIL_[VELOCITY_]TEMPLATES
        # environment set by _variant_env().
        output_arg="--results_dir",
    ),
    "feynman_de_coe": ExperimentSpec(
        "feynman_de_coe",
        (PYTHON, "examples/feynman_de/run_benchmark.py", "--engine", "compare", "--only", "010,113,114,115,117,121,127,206", "--fast", "--skip_generate"),
        "Committee-style comparison proxy for the DE hard-tail FAIL/PARTIAL subset.",
        # No gs_args: see feynman_de above; GS reaches run_de.py via env.
        output_arg="--results_dir",
        note="Uses the feynman_de benchmark compare engine as a compact COE proxy.",
    ),
    "feynman_complex": ExperimentSpec(
        "feynman_complex",
        (PYTHON, "examples/feynman_complex/run_benchmark.py", "--only", "000,001", "--fast"),
        "Complex-valued DE smoke subset.  Current GS effect is report/env plumbing unless the child path uses run_de.py.",
        output_arg="--results_dir",
    ),
    "hamiltonian": ExperimentSpec(
        "hamiltonian",
        (PYTHON, "examples/hamiltonian/anharmonic_oscillator.py"),
        "Hamiltonian discovery smoke example.  This path does not yet consume affine GS proposals.",
    ),
    "kepler_ephemeris_real": ExperimentSpec(
        "kepler_ephemeris_real",
        (PYTHON, "examples/kepler_ephemeris_real/run_class_sr_discovery.py", "--fast"),
        "Kepler Class-SR reduced hierarchy example; child run_SR calls inherit GS env.",
        output_arg="--results_dir",
    ),
    "lane_emden": ExperimentSpec(
        "lane_emden",
        (PYTHON, "examples/lane_emden/smoke_lane_emden_discovery.py", "--skip_lm"),
        "Lane-Emden DE hard-tail smoke run; run_de child inherits GS-DE env.",
        output_arg="--output_dir",
    ),
    "logistic_growth": ExperimentSpec(
        "logistic_growth",
        (PYTHON, "examples/logistic_growth/smoke_logistic_discovery.py", "--skip_lm"),
        "Logistic-growth DE smoke run; run_de child inherits GS-DE env.",
        output_arg="--output_dir",
    ),
    "Maxwell": ExperimentSpec(
        "Maxwell",
        (PYTHON, "examples/Maxwell/run_benchmark.py", "--only", "mw000", "--engine", "stlsq", "--fast"),
        "Vector Maxwell smoke run.",
        output_arg="--results_dir",
    ),
    "MOND": ExperimentSpec(
        "MOND",
        (PYTHON, "examples/MOND/run_benchmark.py", "--only", "mond000", "--fast"),
        "MOND PDE smoke run.",
        output_arg="--results_dir",
    ),
    "multi_dataset": ExperimentSpec(
        "multi_dataset",
        (PYTHON, "examples/multi_dataset/smoke_multi_logistic.py"),
        "Multi-dataset logistic SR smoke run through run_SR.py.",
    ),
    "oracle_factorized_search": ExperimentSpec(
        "oracle_factorized_search",
        (PYTHON, "examples/oracle_factorized_search/run_aif_closure_benchmark.py", "--only", "000", "--n_iter", "1", "--n_seeds", "1", "--max_proposals", "10"),
        "Oracle factorized-search smoke run.  No Stage-A affine GS effect expected.",
    ),
    "special_relativity": ExperimentSpec(
        "special_relativity",
        (PYTHON, "examples/special_relativity/run_class_sr_discovery.py", "--fast"),
        "Special-relativity Class-SR vignette; child run_SR calls inherit GS env.",
        output_arg="--results_dir",
    ),
}


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _variant_env(variant: str, mode: str, outdir: Path, policy: str = "augment", general_affine: bool = False, known_generators: bool = True, jet_enable: bool = True, de_templates: bool = True, de_velocity_templates: bool = True, lorentz_boosts: bool = False, de_lie_prolongation: bool = False, de_lie_prolongation_weight: float | None = None, de_lie_prolongation_tol: float | None = None, de_lie_prolongation_max_samples: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["NESTYNET_RESULTS_DIR"] = str(outdir)
    env["NESTYNET_SR_RESULTS_DIR"] = str(outdir)
    env["NESTYNET_DE_RESULTS_DIR"] = str(outdir)
    if variant == "gs":
        if str(policy) == "gs-only-affine":
            known_generators = False
        env["NESTYNET_GS_ENABLE"] = "1"
        env["NESTYNET_GS_POLICY"] = str(policy)
        env["NESTYNET_SR_GS_POLICY"] = str(policy)
        env["NESTYNET_SR_GS_ENABLE"] = "1"
        env["NESTYNET_GS_MODE"] = str(mode)
        env["NESTYNET_SR_GS_MODE"] = str(mode)
        if mode == "auto":
            env["NESTYNET_GS_AUTO"] = "1"
            env["NESTYNET_SR_GS_AUTO"] = "1"
        if bool(known_generators):
            env["NESTYNET_GS_KNOWN_GENERATORS"] = "1"
            env["NESTYNET_SR_GS_KNOWN_GENERATORS"] = "1"
        else:
            env["NESTYNET_GS_NO_KNOWN_GENERATORS"] = "1"
            env["NESTYNET_SR_GS_NO_KNOWN_GENERATORS"] = "1"
        if bool(general_affine) or str(policy) == "gs-only-affine":
            env["NESTYNET_GS_GENERAL_AFFINE"] = "1"
            env["NESTYNET_SR_GS_GENERAL_AFFINE"] = "1"
        if not bool(jet_enable):
            env["NESTYNET_GS_NO_JET"] = "1"
            env["NESTYNET_SR_GS_NO_JET"] = "1"
        if bool(lorentz_boosts):
            env["NESTYNET_GS_LORENTZ_BOOSTS"] = "1"
        # Neutral DE priors are harmless when not consumed, and useful for
        # run_de paths. They are deliberately not GS evidence.
        if bool(de_templates):
            env["NESTYNET_DE_HARD_TAIL_TEMPLATES"] = "1"
            env["NESTYNET_SR_DE_HARD_TAIL_TEMPLATES"] = "1"
        if bool(de_velocity_templates):
            env["NESTYNET_DE_HARD_TAIL_VELOCITY_TEMPLATES"] = "1"
            env["NESTYNET_SR_DE_HARD_TAIL_VELOCITY_TEMPLATES"] = "1"
        if bool(de_lie_prolongation):
            env["NESTYNET_GS_DE_LIE_PROLONGATION"] = "1"
            env["NESTYNET_SR_GS_DE_LIE_PROLONGATION"] = "1"
        if de_lie_prolongation_weight is not None:
            env["NESTYNET_GS_DE_LIE_PROLONGATION_WEIGHT"] = str(de_lie_prolongation_weight)
            env["NESTYNET_SR_GS_DE_LIE_PROLONGATION_WEIGHT"] = str(de_lie_prolongation_weight)
        if de_lie_prolongation_tol is not None:
            env["NESTYNET_GS_DE_LIE_PROLONGATION_TOL"] = str(de_lie_prolongation_tol)
            env["NESTYNET_SR_GS_DE_LIE_PROLONGATION_TOL"] = str(de_lie_prolongation_tol)
        if de_lie_prolongation_max_samples is not None:
            env["NESTYNET_GS_DE_LIE_PROLONGATION_MAX_SAMPLES"] = str(de_lie_prolongation_max_samples)
            env["NESTYNET_SR_GS_DE_LIE_PROLONGATION_MAX_SAMPLES"] = str(de_lie_prolongation_max_samples)
    else:
        env["NESTYNET_GS_ENABLE"] = "0"
        env["NESTYNET_SR_GS_ENABLE"] = "0"
        env["NESTYNET_GS_MODE"] = "off"
    return env


def _format_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _command_for(spec: ExperimentSpec, *, variant: str, mode: str, outdir: Path, extra: list[str], policy: str = "augment", general_affine: bool = False, known_generators: bool = True, jet_enable: bool = True, lorentz_boosts: bool = False, de_lie_prolongation: bool = False, de_lie_prolongation_weight: float | None = None, de_lie_prolongation_tol: float | None = None, de_lie_prolongation_max_samples: int | None = None) -> list[str]:
    if str(policy) == "gs-only-affine":
        known_generators = False
    cmd = [str(x) for x in spec.command]
    if spec.output_arg:
        cmd += [spec.output_arg, str(outdir)]
    if variant == "gs":
        cmd += list(spec.gs_args)
        if "run_benchmark.py" in " ".join(cmd):
            if "--gs-mode" not in cmd:
                cmd += ["--gs-mode", str(mode)]
            if "--gs-policy" not in cmd:
                cmd += ["--gs-policy", str(policy)]
            if bool(general_affine) and "--gs-general-affine" not in cmd:
                cmd.append("--gs-general-affine")
            if not bool(known_generators) and "--gs-no-known-generators" not in cmd:
                cmd.append("--gs-no-known-generators")
            elif bool(known_generators) and "--gs-known-generators" not in cmd:
                cmd.append("--gs-known-generators")
            if not bool(jet_enable) and "--gs-no-jet" not in cmd:
                cmd.append("--gs-no-jet")
            if bool(lorentz_boosts) and "--gs-lorentz-boosts" not in cmd:
                cmd.append("--gs-lorentz-boosts")
            if bool(de_lie_prolongation) and "--gs-de-lie-prolongation" not in cmd:
                cmd.append("--gs-de-lie-prolongation")
            if de_lie_prolongation_weight is not None and "--gs-de-lie-prolongation-weight" not in cmd:
                cmd += ["--gs-de-lie-prolongation-weight", str(de_lie_prolongation_weight)]
            if de_lie_prolongation_tol is not None and "--gs-de-lie-prolongation-tol" not in cmd:
                cmd += ["--gs-de-lie-prolongation-tol", str(de_lie_prolongation_tol)]
            if de_lie_prolongation_max_samples is not None and "--gs-de-lie-prolongation-max-samples" not in cmd:
                cmd += ["--gs-de-lie-prolongation-max-samples", str(de_lie_prolongation_max_samples)]
    cmd += list(spec.default_extra)
    cmd += list(extra)
    return cmd


def run_one(spec: ExperimentSpec, *, variant: str, mode: str, root: Path, dry_run: bool, extra: list[str], policy: str = "augment", general_affine: bool = False, known_generators: bool = True, jet_enable: bool = True, de_templates: bool = True, de_velocity_templates: bool = True, lorentz_boosts: bool = False, de_lie_prolongation: bool = False, de_lie_prolongation_weight: float | None = None, de_lie_prolongation_tol: float | None = None, de_lie_prolongation_max_samples: int | None = None) -> dict:
    variant_dir = variant if variant != "gs" or str(policy) == "augment" else "gs_" + str(policy).replace("-", "_")
    outdir = root / spec.name / variant_dir
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = _command_for(spec, variant=variant, mode=mode, outdir=outdir, extra=extra, policy=policy, general_affine=general_affine, known_generators=known_generators, jet_enable=jet_enable, lorentz_boosts=lorentz_boosts, de_lie_prolongation=de_lie_prolongation, de_lie_prolongation_weight=de_lie_prolongation_weight, de_lie_prolongation_tol=de_lie_prolongation_tol, de_lie_prolongation_max_samples=de_lie_prolongation_max_samples)
    env = _variant_env(variant, mode, outdir, policy=policy, general_affine=general_affine, known_generators=known_generators, jet_enable=jet_enable, de_templates=de_templates, de_velocity_templates=de_velocity_templates, lorentz_boosts=lorentz_boosts, de_lie_prolongation=de_lie_prolongation, de_lie_prolongation_weight=de_lie_prolongation_weight, de_lie_prolongation_tol=de_lie_prolongation_tol, de_lie_prolongation_max_samples=de_lie_prolongation_max_samples)
    log_path = outdir / "ablation_stdout.log"
    started = time.time()
    row = {
        "experiment": spec.name,
        "variant": variant,
        "mode": mode if variant == "gs" else "off",
        "policy": policy if variant == "gs" else "off",
        "command": cmd,
        "command_string": _format_cmd(cmd),
        "output_dir": str(outdir),
        "log_path": str(log_path),
        "description": spec.description,
        "note": spec.note,
        "dry_run": bool(dry_run),
    }
    print(f"\n[{spec.name}:{variant}] {_format_cmd(cmd)}")
    if dry_run:
        row.update({"returncode": None, "runtime_sec": 0.0, "gs_reports": []})
        return row
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    runtime = time.time() - started
    gs_reports = sorted(str(p) for p in outdir.rglob("*.gs_report.json"))
    row.update({
        "returncode": int(proc.returncode),
        "runtime_sec": float(runtime),
        "gs_reports": gs_reports,
    })
    return row


def _write_summary(rows: list[dict], root: Path, selected: list[str]) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "nestynet_sr_gs_ablation_v1",
        "selected_experiments": selected,
        "rows": rows,
    }
    json_path = root / "gs_ablation_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = ["# GS ablation summary", "", "| experiment | variant | policy | rc | runtime s | GS reports |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        rc = "dry" if r.get("dry_run") else r.get("returncode")
        md.append(
            f"| `{r.get('experiment')}` | `{r.get('variant')}` | `{r.get('policy')}` | {rc} | "
            f"{float(r.get('runtime_sec') or 0.0):.1f} | {len(r.get('gs_reports') or [])} |"
        )
    md.append("")
    md.append("Commands and full logs are recorded in the JSON file and per-variant output directories.")
    md_path = root / "gs_ablation_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path


def run_cli(argv: list[str] | None = None, *, default_experiment: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run baseline-vs-GS ablations for NestyNet_SR_GS examples")
    parser.add_argument("--experiment", default=default_experiment, choices=sorted(EXPERIMENTS) if default_experiment is None else None)
    parser.add_argument("--all", action="store_true", help="Run every registered experiment")
    parser.add_argument("--variant", choices=["baseline", "gs", "both", "matrix"], default="both", help="matrix runs baseline plus GS augment, replace-shadowed, and gs-only-affine")
    parser.add_argument("--mode", choices=["audit", "propose", "auto"], default="auto")
    parser.add_argument("--policy", choices=["augment", "replace-shadowed", "gs-only-affine"], default="augment", help="GS proposal policy for the GS variant")
    parser.add_argument("--general-affine", action="store_true", help="Enable the learned sparse general-affine probe for GS variants")
    parser.add_argument("--no-known-generators", action="store_true", help="Disable named GS generators in GS variants")
    parser.add_argument("--no-jet", action="store_true", help="Disable GS jet diagnostics in GS variants")
    parser.add_argument("--lorentz-boosts", action="store_true", help="Enable Lorentz/hyperbolic named GS generators in GS variants")
    parser.add_argument("--no-de-templates", action="store_true", help="Do not enable GS-DE template env defaults in GS variants")
    parser.add_argument("--no-de-velocity-templates", action="store_true", help="Do not enable GS-DE velocity-template env defaults in GS variants")
    parser.add_argument("--de-lie-prolongation", action="store_true", help="Enable finite point-Lie prolongation scoring in GS DE variants")
    parser.add_argument("--de-lie-prolongation-weight", type=float, default=None, help="Lie-prolongation score penalty weight forwarded to DE variants")
    parser.add_argument("--de-lie-prolongation-tol", type=float, default=None, help="Lie-prolongation acceptance tolerance forwarded to DE variants")
    parser.add_argument("--de-lie-prolongation-max-samples", type=int, default=None, help="Maximum jet samples per prolongation score forwarded to DE variants")
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results" / "gs_ablation"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra args appended to every underlying command after --")
    args = parser.parse_args(argv)

    if args.all:
        names = sorted(EXPERIMENTS)
    else:
        name = args.experiment or default_experiment
        if not name:
            parser.error("provide --experiment NAME or use --all")
        names = [str(name)]
    root = Path(args.results_root).resolve()
    rows: list[dict] = []
    for name in names:
        spec = EXPERIMENTS[name]
        if args.variant == "matrix":
            rows.append(run_one(spec, variant="baseline", mode=args.mode, root=root, dry_run=bool(args.dry_run), extra=list(args.extra or []), policy="off", general_affine=False, known_generators=not bool(args.no_known_generators), jet_enable=not bool(args.no_jet), de_templates=not bool(args.no_de_templates), de_velocity_templates=not bool(args.no_de_velocity_templates), lorentz_boosts=bool(args.lorentz_boosts), de_lie_prolongation=bool(args.de_lie_prolongation), de_lie_prolongation_weight=args.de_lie_prolongation_weight, de_lie_prolongation_tol=args.de_lie_prolongation_tol, de_lie_prolongation_max_samples=args.de_lie_prolongation_max_samples))
            for policy in ("augment", "replace-shadowed", "gs-only-affine"):
                rows.append(run_one(spec, variant="gs", mode=args.mode, root=root, dry_run=bool(args.dry_run), extra=list(args.extra or []), policy=policy, general_affine=(bool(args.general_affine) or policy == "gs-only-affine"), known_generators=not bool(args.no_known_generators), jet_enable=not bool(args.no_jet), de_templates=not bool(args.no_de_templates), de_velocity_templates=not bool(args.no_de_velocity_templates), lorentz_boosts=bool(args.lorentz_boosts), de_lie_prolongation=bool(args.de_lie_prolongation), de_lie_prolongation_weight=args.de_lie_prolongation_weight, de_lie_prolongation_tol=args.de_lie_prolongation_tol, de_lie_prolongation_max_samples=args.de_lie_prolongation_max_samples))
            continue
        variants = ["baseline", "gs"] if args.variant == "both" else [args.variant]
        for variant in variants:
            rows.append(run_one(spec, variant=variant, mode=args.mode, root=root, dry_run=bool(args.dry_run), extra=list(args.extra or []), policy=str(args.policy), general_affine=bool(args.general_affine), known_generators=not bool(args.no_known_generators), jet_enable=not bool(args.no_jet), de_templates=not bool(args.no_de_templates), de_velocity_templates=not bool(args.no_de_velocity_templates), lorentz_boosts=bool(args.lorentz_boosts), de_lie_prolongation=bool(args.de_lie_prolongation), de_lie_prolongation_weight=args.de_lie_prolongation_weight, de_lie_prolongation_tol=args.de_lie_prolongation_tol, de_lie_prolongation_max_samples=args.de_lie_prolongation_max_samples))
    json_path, md_path = _write_summary(rows, root, names)
    print(f"\nWrote ablation summary:\n  {json_path}\n  {md_path}")
    return 0 if all((r.get("returncode") in (0, None)) for r in rows) else 1


def cli_for_current_dir(argv: list[str] | None = None) -> int:
    # Direct execution does not assume an experiment; callers may pass one in argv.
    return run_cli(argv, default_experiment=None)


if __name__ == "__main__":
    raise SystemExit(run_cli())
