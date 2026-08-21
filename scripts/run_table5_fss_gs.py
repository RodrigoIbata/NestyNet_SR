#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Run and summarize the isolated Paper III Table 5 FSS/GS comparison.

The production search path remains the existing ``run_factorized_search_all.sh``
launcher.  This script fixes the Table 5 protocol, runs two sequential arms,
and gives only the GS arm the opt-in ``--gs-carrier-seed`` flag.  It neither
changes nor enables generalized symmetry for any other command.

GS carrier seeds are discovered by differentiating the exact analytic oracle
target, so every result this launcher produces is an oracle-gradient ablation
rather than a data-only symbolic-regression result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EQUATIONS = REPO_ROOT / "data" / "equations.txt"
PARALLEL_LAUNCHER = REPO_ROOT / "scripts" / "run_factorized_search_all.sh"
ARM_NAMES = ("fss_only", "fss_gs")
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _source_metadata() -> dict[str, Any]:
    status = _run_git("status", "--porcelain", "--untracked-files=all")
    return {
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_branch": _run_git("branch", "--show-current"),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _normalise_requested_ids(raw: str | None) -> set[str] | None:
    if raw is None or not str(raw).strip():
        return None
    out: set[str] = set()
    for token in str(raw).replace(",", " ").split():
        value = token.strip()
        if value.startswith("feynman_"):
            value = value[len("feynman_") :]
        if not value.isdigit():
            raise ValueError(f"invalid equation ID: {token!r}")
        out.add(f"feynman_{int(value):03d}")
    return out


def _load_eligible_specs(
    equations: Path,
    *,
    max_nvars: int,
    requested_ids: set[str] | None,
) -> list[dict[str, Any]]:
    from nestynet_sr.sr_search.factorized_search.aif_closure_benchmark import (
        parse_equations_txt,
    )

    specs = [
        row
        for row in parse_equations_txt(equations)
        if len(row.get("variables", [])) <= int(max_nvars)
    ]
    if requested_ids is not None:
        known = {str(row["id"]) for row in specs}
        missing = sorted(requested_ids - known)
        if missing:
            raise ValueError(f"requested IDs are absent or exceed max_nvars: {missing}")
        specs = [row for row in specs if str(row["id"]) in requested_ids]
    return sorted(specs, key=lambda row: str(row["id"]))


def _row_from_result_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return None
    return dict(rows[0])


def _completed_ids(output_dir: Path, expected_ids: Sequence[str]) -> set[str]:
    completed: set[str] = set()
    for eq_id in expected_ids:
        row = _row_from_result_file(output_dir / f"{eq_id}.json")
        if row is not None and str(row.get("id", "")) == eq_id:
            completed.add(eq_id)
    return completed


def _clean_child_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("NESTY_") or key.startswith("NESTYNET_GS") or key.startswith(
            "NESTYNET_SR_GS"
        ):
            env.pop(key, None)
    for key in THREAD_ENV_KEYS:
        env[key] = "1"
    existing_pythonpath = str(env.get("PYTHONPATH", "") or "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else str(REPO_ROOT) + os.pathsep + existing_pythonpath
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _arm_environment(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    equation_ids: Sequence[str],
) -> dict[str, str]:
    env = _clean_child_environment()
    env.update(
        {
            "JOBS": str(int(args.jobs)),
            "EQUATIONS": str(Path(args.equations).resolve()),
            "ONLY_IDS": " ".join(eq_id.removeprefix("feynman_") for eq_id in equation_ids),
            "SKIP_IDS": "",
            "START_ID": "",
            "END_ID": "",
            "OUTPUT_DIR": str(output_dir),
            "LOG_DIR": str(output_dir / "logs"),
            "N_ITER": str(int(args.n_iter)),
            "N_SEEDS": "1",
            "MAX_PROPOSALS": "48",
            "ANCHORS": "8",
            "PREVIEW_TOPK": "12",
            "EXACT_TOPK": "8",
            "BEAM_WIDTH": "6",
            "SEED_EXACT_TOPK": "6",
            "SEED_BEAM_WIDTH": "4",
            "SEED_SCAFFOLD_RESERVE": "8",
            "PAIR_RESCUE_TOPK": "4",
            "PAIR_RESCUE_MAX_PAIRS": "6",
            "PAIR_NORMAL_TOPK": "3",
            "PAIR_NORMAL_MAX_PAIRS": "1",
            "CLOSURE_DEBUG_TOPK": "0",
            "MAX_NVARS": str(int(args.max_nvars)),
            "ORACLE_DATA_SOURCE": "synthetic",
            "DATA_DIR": str(REPO_ROOT / "data"),
            "N_FIT": str(int(args.n_fit)),
            "N_PROBE": str(int(args.n_probe)),
            "SEARCH_N_FIT": "",
            "SEARCH_N_PROBE": "",
            "FINAL_VALIDATE_FULL": "0",
            "FINAL_VALIDATE_RERANK": "0",
            "DATA_SLICE": "0",
            "Y_CHECK": "1",
            "Y_CHECK_ABS_TOL": "1e-8",
            "Y_CHECK_REL_TOL": "1e-8",
            "SEED": str(int(args.seed)),
            "SUCCESS_MSE": repr(float(args.success_mse)),
            "REFINE_SKELETON": "0",
            "REFINE_PROFILE": "",
            "REFINE_MODE": "",
            "REFINE_OPTIMIZER": "",
            "REFINE_LBFGS_STEPS": "",
            "REFINE_FIT_SUBSET": "",
            "REFINE_NUM_RESTARTS": "",
            "REFINE_MAX_VARIANTS": "",
            "REFINE_MAX_PARAMS": "",
            "REFINE_MAX_TRIALS": "",
            "REFINE_GATE_BEST_FACTOR": "",
            "REFINE_LINEAR_COMBO": "",
            "EMERGENT_BASIS": "0",
            "EMERGENT_AUX_ATOMS": "0",
            "PAIR_NORMAL_ENABLE": "0",
            "BENCHMARK_MODULE": (
                "nestynet_sr.sr_search.factorized_search.aif_closure_benchmark"
            ),
            "PYTHON": sys.executable,
            "DRY_RUN": "1" if bool(args.dry_run) else "0",
        }
    )
    return env


def _arm_command(arm: str) -> list[str]:
    command = [str(PARALLEL_LAUNCHER), "--retain-candidate-payload"]
    if arm == "fss_gs":
        command.append("--gs-carrier-seed")
    return command


def _manifest_environment(env: Mapping[str, str]) -> dict[str, str]:
    keys = (
        "JOBS",
        "EQUATIONS",
        "ONLY_IDS",
        "OUTPUT_DIR",
        "LOG_DIR",
        "N_ITER",
        "N_SEEDS",
        "MAX_PROPOSALS",
        "ANCHORS",
        "PREVIEW_TOPK",
        "EXACT_TOPK",
        "BEAM_WIDTH",
        "SEED_EXACT_TOPK",
        "SEED_BEAM_WIDTH",
        "SEED_SCAFFOLD_RESERVE",
        "PAIR_RESCUE_TOPK",
        "PAIR_RESCUE_MAX_PAIRS",
        "PAIR_NORMAL_TOPK",
        "PAIR_NORMAL_MAX_PAIRS",
        "CLOSURE_DEBUG_TOPK",
        "MAX_NVARS",
        "ORACLE_DATA_SOURCE",
        "N_FIT",
        "N_PROBE",
        "FINAL_VALIDATE_FULL",
        "FINAL_VALIDATE_RERANK",
        "SEED",
        "SUCCESS_MSE",
        "REFINE_SKELETON",
        "EMERGENT_BASIS",
        "EMERGENT_AUX_ATOMS",
        "PAIR_NORMAL_ENABLE",
        "BENCHMARK_MODULE",
        "PYTHON",
        *THREAD_ENV_KEYS,
    )
    return {key: str(env[key]) for key in keys if key in env}


def _selected_arms(name: str) -> tuple[str, ...]:
    if name == "both":
        return ARM_NAMES
    return (str(name),)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _ast_from_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_ast_from_json(item) for item in value)
    return value


def _mapping_for_eval(raw_mapping: Any) -> dict[str, Any]:
    if not isinstance(raw_mapping, dict):
        raise ValueError("candidate payload has no fitted outer mapping")
    mapping = dict(raw_mapping)
    raw_head = mapping.get("_lin_head")
    if isinstance(raw_head, dict):
        head = dict(raw_head)
        raw_terms = head.get("terms")
        if isinstance(raw_terms, (list, tuple)):
            head["terms"] = [_ast_from_json(term) for term in raw_terms]
        mapping["_lin_head"] = head
    return mapping


def _dense_audit(
    row: Mapping[str, Any],
    spec_dict: Mapping[str, Any],
    *,
    n_probe: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node
    from nestynet_sr.sr_search.factorized_search.inverse_core import eval_mapping_total
    from nestynet_sr.sr_search.factorized_search.oracle_lab import (
        build_oracle_dataset,
        compile_target_expression,
        equation_spec_from_dict,
    )

    payload = row.get("candidate_payload")
    if not isinstance(payload, dict):
        return {
            "status": "error",
            "error": "missing retained candidate payload",
        }

    try:
        spec = equation_spec_from_dict(dict(spec_dict))
        target_fn = compile_target_expression(spec)
        dataset = build_oracle_dataset(
            spec,
            target_fn,
            n_fit=1,
            n_probe=int(n_probe),
            seed=int(seed),
            dtype=torch.float64,
        )
        candidate = _ast_from_json(payload.get("expr_ast"))
        if not isinstance(candidate, tuple):
            raise ValueError("candidate payload has no carrier AST")
        mapping = _mapping_for_eval(payload.get("mapping"))
        with torch.no_grad():
            carrier = eval_node(candidate, dataset["x_probe"]).reshape(-1, 1)
            prediction = eval_mapping_total(
                carrier,
                mapping,
                dataset["x_probe"],
            ).reshape(-1, 1)
            target = dataset["y_probe"].reshape(-1, 1)
        finite = torch.isfinite(prediction) & torch.isfinite(target)
        finite_fraction = float(finite.to(dtype=torch.float64).mean().item())
        if not bool(finite.all()):
            return {
                "status": "nonfinite",
                "finite_fraction": finite_fraction,
                "n_probe": int(n_probe),
                "seed": int(seed),
            }
        error = prediction - target
        mse = float(torch.mean(error * error).item())
        rmse = float(math.sqrt(max(0.0, mse)))
        target_rms = float(torch.sqrt(torch.mean(target * target)).item())
        normalizer = max(target_rms, 1.0e-15)
        return {
            "status": "ok",
            "mse": mse,
            "rmse": rmse,
            "normalized_rmse": float(rmse / normalizer),
            "max_abs_error": float(torch.max(torch.abs(error)).item()),
            "target_rms": target_rms,
            "finite_fraction": finite_fraction,
            "n_probe": int(n_probe),
            "seed": int(seed),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "n_probe": int(n_probe),
            "seed": int(seed),
        }


def _load_arm_rows(output_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not output_dir.is_dir():
        return rows
    for path in sorted(output_dir.glob("feynman_*.json")):
        row = _row_from_result_file(path)
        if row is None:
            continue
        eq_id = str(row.get("id", "") or "")
        if eq_id:
            rows[eq_id] = row
    return rows


def _classification(
    row: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, bool]:
    search_mse = _finite_float(row.get("mse"))
    audit_mse = _finite_float(audit.get("mse"))
    search_solved = bool(search_mse is not None and search_mse < float(threshold))
    audit_solved = bool(
        audit.get("status") == "ok"
        and audit_mse is not None
        and audit_mse < float(threshold)
    )
    return {
        "search_solved": search_solved,
        "dense_audit_solved": audit_solved,
        "validated_solved": bool(search_solved and audit_solved),
    }


def _case_arm_record(
    row: Mapping[str, Any] | None,
    spec: Mapping[str, Any],
    *,
    arm: str,
    primary_mse: float,
    strict_mse: float,
    audit_n_probe: int,
    audit_seed: int,
) -> dict[str, Any]:
    if row is None:
        return {"status": "missing"}
    eq_num = int(str(spec["id"]).removeprefix("feynman_"))
    audit = _dense_audit(
        row,
        spec,
        n_probe=int(audit_n_probe),
        seed=int(audit_seed) + eq_num,
    )
    record = {
        "status": str(row.get("status", "")),
        "search_mse": _finite_float(row.get("mse")),
        "expression": str(row.get("expr", "") or ""),
        "wall_seconds": _finite_float(row.get("wall_s")),
        "primary": _classification(row, audit, threshold=float(primary_mse)),
        "strict": _classification(row, audit, threshold=float(strict_mse)),
        "dense_audit": audit,
    }
    if arm == "fss_gs":
        diagnostics = list(row.get("gs_carrier_seed_diagnostics", []) or [])
        record["gs"] = {
            "requested": bool(row.get("gs_carrier_seed", False)),
            "seed_count": int(row.get("gs_carrier_seed_count", len(diagnostics)) or 0),
            "diagnostics": diagnostics,
        }
    return record


def _timing_stats(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "median_seconds": None, "mean_seconds": None, "total_hours": 0.0}
    return {
        "n": int(len(clean)),
        "median_seconds": float(statistics.median(clean)),
        "mean_seconds": float(statistics.fmean(clean)),
        "total_hours": float(sum(clean) / 3600.0),
    }


def _arm_summary(
    cases: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    classification_key: str,
) -> dict[str, Any]:
    records = [case["arms"][arm] for case in cases]
    present = [record for record in records if record.get("status") != "missing"]
    wall_all = [
        value
        for record in present
        if (value := _finite_float(record.get("wall_seconds"))) is not None
    ]
    validated = [
        record
        for record in present
        if bool(record.get(classification_key, {}).get("validated_solved", False))
    ]
    unvalidated = [
        record
        for record in present
        if not bool(record.get(classification_key, {}).get("validated_solved", False))
    ]
    wall_validated = [
        value
        for record in validated
        if (value := _finite_float(record.get("wall_seconds"))) is not None
    ]
    wall_unvalidated = [
        value
        for record in unvalidated
        if (value := _finite_float(record.get("wall_seconds"))) is not None
    ]
    out = {
        "expected": int(len(records)),
        "present": int(len(present)),
        "missing": int(len(records) - len(present)),
        "search_solved": int(
            sum(bool(record.get(classification_key, {}).get("search_solved", False)) for record in present)
        ),
        "dense_audit_solved": int(
            sum(
                bool(record.get(classification_key, {}).get("dense_audit_solved", False))
                for record in present
            )
        ),
        "validated_solved": int(len(validated)),
        "search_pass_audit_fail": int(
            sum(
                bool(record.get(classification_key, {}).get("search_solved", False))
                and not bool(
                    record.get(classification_key, {}).get("dense_audit_solved", False)
                )
                for record in present
            )
        ),
        "timing_all": _timing_stats(wall_all),
        "timing_validated_solved": _timing_stats(wall_validated),
        "timing_not_validated": _timing_stats(wall_unvalidated),
    }
    if arm == "fss_gs":
        out["gs_seed_emission"] = {
            "emitted": int(
                sum(int(record.get("gs", {}).get("seed_count", 0)) > 0 for record in present)
            ),
            "empty_or_failed": int(
                sum(int(record.get("gs", {}).get("seed_count", 0)) == 0 for record in present)
            ),
        }
    return out


def _by_nvar(
    cases: Sequence[Mapping[str, Any]],
    *,
    available_arms: Sequence[str],
    classification_key: str,
) -> list[dict[str, Any]]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for case in cases:
        groups.setdefault(int(case["nvars"]), []).append(case)
    rows: list[dict[str, Any]] = []
    for nvars in sorted(groups):
        group = groups[nvars]
        out: dict[str, Any] = {"nvars": int(nvars), "total": int(len(group))}
        for arm in available_arms:
            records = [case["arms"][arm] for case in group]
            out[arm] = {
                "search_solved": int(
                    sum(
                        bool(record.get(classification_key, {}).get("search_solved", False))
                        for record in records
                    )
                ),
                "validated_solved": int(
                    sum(
                        bool(
                            record.get(classification_key, {}).get(
                                "validated_solved",
                                False,
                            )
                        )
                        for record in records
                    )
                ),
            }
        if all(arm in available_arms for arm in ARM_NAMES):
            out["paired"] = {
                "rescues": int(
                    sum(
                        not bool(
                            case["arms"]["fss_only"]
                            .get(classification_key, {})
                            .get("validated_solved", False)
                        )
                        and bool(
                            case["arms"]["fss_gs"]
                            .get(classification_key, {})
                            .get("validated_solved", False)
                        )
                        for case in group
                    )
                ),
                "regressions": int(
                    sum(
                        bool(
                            case["arms"]["fss_only"]
                            .get(classification_key, {})
                            .get("validated_solved", False)
                        )
                        and not bool(
                            case["arms"]["fss_gs"]
                            .get(classification_key, {})
                            .get("validated_solved", False)
                        )
                        for case in group
                    )
                ),
            }
        rows.append(out)
    return rows


def _paired_summary(
    cases: Sequence[Mapping[str, Any]],
    *,
    classification_key: str,
) -> dict[str, Any]:
    outcomes = {"both": 0, "rescue": 0, "regression": 0, "neither": 0}
    wall_deltas: list[float] = []
    for case in cases:
        baseline = case["arms"]["fss_only"]
        with_gs = case["arms"]["fss_gs"]
        base_ok = bool(
            baseline.get(classification_key, {}).get("validated_solved", False)
        )
        gs_ok = bool(with_gs.get(classification_key, {}).get("validated_solved", False))
        if base_ok and gs_ok:
            outcomes["both"] += 1
        elif gs_ok:
            outcomes["rescue"] += 1
        elif base_ok:
            outcomes["regression"] += 1
        else:
            outcomes["neither"] += 1
        base_wall = _finite_float(baseline.get("wall_seconds"))
        gs_wall = _finite_float(with_gs.get("wall_seconds"))
        if base_wall is not None and gs_wall is not None:
            wall_deltas.append(float(gs_wall - base_wall))
    return {
        "outcomes": outcomes,
        "gs_minus_fss_wall_seconds": _timing_stats(wall_deltas),
    }


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    primary = summary["thresholds"]["primary_mse"]
    strict = summary["thresholds"]["strict_mse"]
    lines = [
        "# Paper III Table 5: FSS and FSS+GS",
        "",
        f"Primary numerical threshold: MSE `< {primary:.0e}`. "
        "A solve is marked validated only when it also passes the independent dense holdout.",
        "",
        "Paper III declares `<1e-8`, while the historical count appears to have "
        "threshold-provenance ambiguity. This report therefore keeps the chosen "
        f"primary `<{primary:.0e}` result and a strict `<{strict:.0e}` "
        "reclassification from the same candidates; the published 90/115 is "
        "reference-only.",
        "",
    ]
    available = list(summary["available_arms"])
    if "fss_gs" in available:
        lines.extend(
            [
                "GS carrier seeds are discovered by differentiating the exact "
                "analytic oracle target, not by learning from data alone. The "
                "GS arm is therefore an oracle-gradient ablation and must be "
                "labeled as such wherever these numbers are quoted.",
                "",
            ]
        )
    header = ["nvar", "total"]
    for arm in available:
        header.extend([f"{arm} search", f"{arm} validated"])
    if all(arm in available for arm in ARM_NAMES):
        header.extend(["GS rescues", "GS regressions"])

    def append_classification_table(title: str, key: str, threshold: float) -> None:
        lines.extend(
            [
                f"## {title} (`<{threshold:.0e}`)",
                "",
                "| " + " | ".join(header) + " |",
                "|" + "|".join("---:" for _ in header) + "|",
            ]
        )
        for row in summary[key]["by_nvar"]:
            values = [str(row["nvars"]), str(row["total"])]
            for arm in available:
                values.extend(
                    [
                        str(row[arm]["search_solved"]),
                        str(row[arm]["validated_solved"]),
                    ]
                )
            if all(arm in available for arm in ARM_NAMES):
                values.extend(
                    [
                        str(row["paired"]["rescues"]),
                        str(row["paired"]["regressions"]),
                    ]
                )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    append_classification_table("Primary classification", "primary", float(primary))
    append_classification_table("Strict classification", "strict", float(strict))

    lines.extend(["## Timing", ""])
    lines.append(
        "The published 9 s and 245 s values are respectively the median and mean "
        "over all eligible runs; the large gap is a heavy-tail statistic, not two "
        "definitions of the same quantity."
    )
    lines.extend(
        [
            "",
            "| arm | runs | median all (s) | mean all (s) | total serial (h) | "
            "median validated (s) | mean validated (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in available:
        arm_summary = summary["primary"]["arms"][arm]
        all_t = arm_summary["timing_all"]
        solved_t = arm_summary["timing_validated_solved"]

        def fmt(value: Any) -> str:
            number = _finite_float(value)
            return "—" if number is None else f"{number:.3g}"

        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    str(all_t["n"]),
                    fmt(all_t["median_seconds"]),
                    fmt(all_t["mean_seconds"]),
                    fmt(all_t["total_hours"]),
                    fmt(solved_t["median_seconds"]),
                    fmt(solved_t["mean_seconds"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The machine-readable report also includes per-case dense-audit "
            "metrics, missing/error counts, and paired end-to-end "
            "GS-minus-FSS wall-time deltas.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_results(
    output_root: Path,
    specs: Sequence[Mapping[str, Any]],
    *,
    primary_mse: float,
    strict_mse: float,
    audit_n_probe: int,
    audit_seed: int,
) -> dict[str, Any]:
    rows_by_arm = {
        arm: _load_arm_rows(output_root / arm)
        for arm in ARM_NAMES
    }
    available_arms = [
        arm for arm in ARM_NAMES if rows_by_arm[arm]
    ]
    if not available_arms:
        raise ValueError(f"no per-equation results found under {output_root}")

    cases: list[dict[str, Any]] = []
    for spec in specs:
        eq_id = str(spec["id"])
        arms = {
            arm: _case_arm_record(
                rows_by_arm[arm].get(eq_id),
                spec,
                arm=arm,
                primary_mse=float(primary_mse),
                strict_mse=float(strict_mse),
                audit_n_probe=int(audit_n_probe),
                audit_seed=int(audit_seed),
            )
            for arm in available_arms
        }
        cases.append(
            {
                "id": eq_id,
                "nvars": int(len(spec.get("variables", []))),
                "target": str(spec.get("target", {}).get("expr", "")),
                "arms": arms,
            }
        )

    def classification_summary(key: str) -> dict[str, Any]:
        block: dict[str, Any] = {
            "arms": {
                arm: _arm_summary(cases, arm=arm, classification_key=key)
                for arm in available_arms
            },
            "by_nvar": _by_nvar(
                cases,
                available_arms=available_arms,
                classification_key=key,
            ),
        }
        if all(arm in available_arms for arm in ARM_NAMES):
            block["paired"] = _paired_summary(cases, classification_key=key)
        return block

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root.resolve()),
        "available_arms": available_arms,
        "expected_equations": int(len(specs)),
        "gs_seed_source": "oracle_gradient",
        "gs_seed_source_note": (
            "GS carrier seeds are discovered by differentiating the exact "
            "analytic oracle target, so the fss_gs arm is an oracle-gradient "
            "ablation rather than a data-only result"
        ),
        "thresholds": {
            "primary_mse": float(primary_mse),
            "strict_mse": float(strict_mse),
            "dense_audit_n_probe": int(audit_n_probe),
            "dense_audit_seed": int(audit_seed),
            "validation_rule": (
                "search MSE and independent dense-holdout MSE must both be below threshold"
            ),
        },
        "published_reference": {
            "fss_solved": 90,
            "eligible": 115,
            "paper_declared_success_mse": 1.0e-8,
            "median_wall_seconds_all": 9.0,
            "mean_wall_seconds_all": 245.0,
            "historical_count_threshold_verified": False,
            "provenance_note": (
                "Historical reference only. The caption declares 1e-8, but the "
                "raw run provenance has not been located; do not use 90/115 as "
                "the paired baseline for current GS."
            ),
        },
        "primary": classification_summary("primary"),
        "strict": classification_summary("strict"),
        "cases": cases,
    }
    _write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(
        _markdown_summary(summary),
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--equations", default=str(DEFAULT_EQUATIONS))
    parser.add_argument("--arm", choices=("both", *ARM_NAMES), default="both")
    parser.add_argument("--only", default=None, help="Comma/space-separated pilot IDs")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--n-iter", type=int, default=1400)
    parser.add_argument("--n-fit", type=int, default=512)
    parser.add_argument("--n-probe", type=int, default=2048)
    parser.add_argument("--max-nvars", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-mse", type=float, default=1.0e-6)
    parser.add_argument("--strict-mse", type=float, default=1.0e-8)
    parser.add_argument("--audit-n-probe", type=int, default=16384)
    parser.add_argument("--audit-seed", type=int, default=20260723)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a non-dry production launch from a dirty source tree.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if int(args.jobs) <= 0:
        parser.error("--jobs must be positive")
    for name in ("n_iter", "n_fit", "n_probe", "audit_n_probe"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    thresholds = (float(args.success_mse), float(args.strict_mse))
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        parser.error("MSE thresholds must be finite and positive")
    if float(args.strict_mse) > float(args.success_mse):
        parser.error("--strict-mse must be no greater than --success-mse")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (REPO_ROOT / "results" / "table5_fss_gs" / _utc_stamp()).resolve()
    )
    requested_ids = _normalise_requested_ids(args.only)
    specs = _load_eligible_specs(
        Path(args.equations).resolve(),
        max_nvars=int(args.max_nvars),
        requested_ids=requested_ids,
    )
    if not specs:
        parser.error("no eligible equations selected")
    equation_ids = [str(spec["id"]) for spec in specs]

    if bool(args.summarize_only):
        summarize_results(
            output_root,
            specs,
            primary_mse=float(args.success_mse),
            strict_mse=float(args.strict_mse),
            audit_n_probe=int(args.audit_n_probe),
            audit_seed=int(args.audit_seed),
        )
        print(f"Summary: {output_root / 'summary.md'}")
        return 0

    source = _source_metadata()
    if source["git_dirty"] and not bool(args.dry_run) and not bool(args.allow_dirty):
        parser.error(
            "source tree is dirty; commit/stash the benchmark harness or pass "
            "--allow-dirty explicitly"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "protocol.json"
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "output_root": str(output_root),
        "equations": str(Path(args.equations).resolve()),
        "eligible_equation_ids": equation_ids,
        "protocol": {
            "n_iter": int(args.n_iter),
            "n_seeds": 1,
            "n_fit": int(args.n_fit),
            "n_probe": int(args.n_probe),
            "max_nvars": int(args.max_nvars),
            "seed": int(args.seed),
            "primary_success_mse": float(args.success_mse),
            "strict_audit_mse": float(args.strict_mse),
            "oracle_data_source": "synthetic",
            "dtype": "torch.float64",
            "dimensions_enabled": True,
            "continuous_refinement": False,
            "emergent_basis": False,
            "emergent_aux_atoms": False,
            "pair_normal": False,
            "candidate_payload_retained_for_reporting_only": True,
            "gs_information_interface": "exact_oracle_target_gradients",
            "gs_carrier_dimension_policy": (
                "benchmark_local_pre_map_dimension_bypass"
            ),
            "shared_engine_defaults_changed": False,
            "arms_run_sequentially": True,
            "thread_count_per_worker": 1,
        },
        "arms": {},
        "dry_run": bool(args.dry_run),
    }
    _write_json(manifest_path, protocol)

    for arm in _selected_arms(args.arm):
        output_dir = output_root / arm
        existing = _completed_ids(output_dir, equation_ids)
        if existing and not bool(args.resume):
            parser.error(
                f"{output_dir} already has {len(existing)} completed result(s); "
                "use --resume or choose a fresh --output-root"
            )
        pending = [eq_id for eq_id in equation_ids if eq_id not in existing]
        if not pending:
            print(f"[{arm}] all {len(equation_ids)} selected equations are complete")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        env = _arm_environment(args, output_dir=output_dir, equation_ids=pending)
        command = _arm_command(arm)
        record = {
            "command": command,
            "environment": _manifest_environment(env),
            "gs_carrier_seed": bool(arm == "fss_gs"),
            "completed_before": int(len(existing)),
            "pending": pending,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": None,
            "wall_seconds": None,
        }
        protocol["arms"][arm] = record
        _write_json(manifest_path, protocol)

        print(f"[{arm}] {' '.join(command)}")
        started = time.perf_counter()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        record["wall_seconds"] = float(time.perf_counter() - started)
        record["returncode"] = int(proc.returncode)
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, protocol)
        if proc.returncode != 0:
            return int(proc.returncode)

    if not bool(args.dry_run) and not bool(args.no_summarize):
        summarize_results(
            output_root,
            specs,
            primary_mse=float(args.success_mse),
            strict_mse=float(args.strict_mse),
            audit_n_probe=int(args.audit_n_probe),
            audit_seed=int(args.audit_seed),
        )
        print(f"Summary: {output_root / 'summary.md'}")
    else:
        print(f"Protocol: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
