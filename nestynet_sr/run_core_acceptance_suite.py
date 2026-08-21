#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Manifest-driven frozen acceptance suite for core SR/DE workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import pickle
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence

from nestynet_sr.run_sr_reports import _report_final_selection_eligibility

try:
    import sympy as sp

    _HAVE_SYMPY = True
except Exception:  # pragma: no cover
    _HAVE_SYMPY = False


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_SR_SCRIPT = REPO_ROOT / "nestynet_sr" / "run_SR.py"
RUN_DE_SCRIPT = REPO_ROOT / "nestynet_sr" / "run_de.py"
DEFAULT_SUITE_MANIFEST = REPO_ROOT / "examples" / "core_acceptance_suites" / "frozen_core_fast.json"


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_output_json_path(path: str | pathlib.Path, *, default_name: str) -> pathlib.Path:
    out = pathlib.Path(path)
    if out.suffix.lower() == ".json":
        return out
    return out / default_name


def _write_csv(rows: Sequence[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "case_id",
        "kind",
        "success",
        "returncode",
        "wall_seconds",
        "report_path",
        "log_path",
        "truth_rmse_rel",
        "truth_frac_valid",
        "stageA_val_loss",
        "stageB_val_loss",
        "de_order",
        "de_rms_val_mean",
        "de_stageb_val_loss",
        "reasons",
    ]
    keys: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_acceptance_suite(path: str | pathlib.Path | None = None) -> tuple[pathlib.Path, dict[str, Any]]:
    manifest_path = pathlib.Path(path) if path is not None else DEFAULT_SUITE_MANIFEST
    payload = _load_json(manifest_path)
    cases = list(payload.get("cases") or [])
    if not cases:
        raise ValueError(f"No cases declared in suite manifest: {manifest_path}")
    return manifest_path, payload


def _resolve_repo_path(raw: str, *, manifest_path: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(str(raw))
    candidates: list[pathlib.Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(manifest_path.parent / p)
        candidates.append(REPO_ROOT / p)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"Could not resolve path: {raw}")


def resolve_suite_cases(payload: dict[str, Any], *, manifest_path: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(payload.get("cases") or []):
        if not isinstance(item, dict):
            raise TypeError(f"Expected case object in manifest: {item!r}")
        case = dict(item)
        case_id = str(case.get("case_id", "") or "")
        if not case_id:
            raise ValueError(f"Case missing case_id in {manifest_path}")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id in {manifest_path}: {case_id}")
        seen.add(case_id)

        kind = str(case.get("kind", "sr") or "sr").lower()
        if kind not in {"sr", "sr_firstclass_de", "de", "pytest", "python"}:
            raise ValueError(f"Unsupported case kind for {case_id}: {kind}")
        case["kind"] = kind

        if "filepath" in case and case.get("filepath"):
            case["filepath"] = str(_resolve_repo_path(str(case["filepath"]), manifest_path=manifest_path))
        if "filepaths" in case and case.get("filepaths"):
            case["filepaths"] = [
                str(_resolve_repo_path(str(raw), manifest_path=manifest_path))
                for raw in list(case.get("filepaths") or [])
            ]
        if "script" in case and case.get("script"):
            case["script"] = str(_resolve_repo_path(str(case["script"]), manifest_path=manifest_path))
        if "paths" in case and case.get("paths"):
            case["paths"] = [
                str(_resolve_repo_path(str(raw), manifest_path=manifest_path))
                for raw in list(case.get("paths") or [])
            ]

        out.append(case)

    if not out:
        raise ValueError(f"Suite manifest resolved zero cases: {manifest_path}")
    return out


def _parse_case_filter(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    vals = {tok.strip() for tok in str(raw).split(",") if tok.strip()}
    return vals or None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: Any) -> float | None:
    if values is None:
        return None
    if isinstance(values, (int, float)):
        return _coerce_float(values)
    try:
        xs = [_coerce_float(v) for v in list(values)]
    except Exception:
        return None
    xs = [v for v in xs if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _normalize_expr(expr: str) -> str:
    out = str(expr or "").replace("^", "**")
    out = out.replace("arcsin", "asin").replace("arccos", "acos")
    return out


def expressions_equivalent(lhs: str | None, rhs: str | None) -> bool:
    if lhs is None or rhs is None:
        return False
    left = _normalize_expr(lhs)
    right = _normalize_expr(rhs)
    if "".join(left.split()) == "".join(right.split()):
        return True
    if not _HAVE_SYMPY:
        return False
    try:
        local_dict = {
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "asin": sp.asin,
            "acos": sp.acos,
            "pi": sp.pi,
            "E": sp.E,
        }
        lhs_expr = sp.sympify(left, locals=local_dict)
        rhs_expr = sp.sympify(right, locals=local_dict)
        diff = sp.simplify(sp.together(lhs_expr - rhs_expr))
        return bool(diff == 0)
    except Exception:
        return False


def _metric_key_from_term(term: str) -> str:
    out = []
    for ch in str(term):
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "term"


def _extract_coeff_row(coeffs: Any) -> list[float]:
    if coeffs is None:
        raise ValueError("Missing coeffs payload")
    if hasattr(coeffs, "detach"):
        coeffs = coeffs.detach().cpu()
    ndim = getattr(coeffs, "ndim", None)
    if ndim == 2:
        if int(getattr(coeffs, "shape", [0])[0]) < 1:
            raise ValueError("Empty 2D coeff tensor")
        coeffs = coeffs[0]
    if hasattr(coeffs, "tolist"):
        coeffs = coeffs.tolist()
    if isinstance(coeffs, list) and coeffs and isinstance(coeffs[0], (list, tuple)):
        coeffs = coeffs[0]
    return [float(v) for v in list(coeffs)]


def extract_sr_de_coeff_map(report: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    de_block = dict(report.get("de") or {})
    artifacts = dict(de_block.get("artifacts") or {})
    pkl_path = pathlib.Path(str(artifacts.get("pkl", "") or ""))
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Missing DE artifact pkl: {pkl_path}")
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    terms = list(payload.get("term_asts") or [])
    coeff_row = _extract_coeff_row(payload.get("coeffs"))
    coeff_map: dict[str, float] = {}
    for term, coeff in zip(terms, coeff_row):
        key = str(term) if isinstance(term, str) else repr(term)
        coeff_map[key] = float(coeff)
    return coeff_map, payload


def extract_run_de_coeff_map(report: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    de = dict(report.get("de_discovery") or {})
    coeffs = de.get("coefficients")
    if coeffs is None:
        raise ValueError("Missing de_discovery.coefficients")
    coeff_row = _extract_coeff_row(coeffs)
    terms = [str(t) for t in list(de.get("terms") or [])]
    coeff_map: dict[str, float] = {}
    for term, coeff in zip(terms, coeff_row):
        coeff_map[str(term)] = float(coeff)
    return coeff_map, de


def evaluate_sr_report(report: dict[str, Any], expect: dict[str, Any] | None = None) -> dict[str, Any]:
    expect = dict(expect or {})
    reasons: list[str] = []

    stageA = report.get("stageA")
    stageB = report.get("stageB")
    stageC = report.get("stageC")
    truth_eval = report.get("truth_eval")
    selection_eligible, selection_reason = _report_final_selection_eligibility(report)
    raw_truth_success = (
        bool((truth_eval or {}).get("success"))
        if isinstance(truth_eval, dict)
        else False
    )
    truth_success = bool(raw_truth_success and selection_eligible)

    stageA_val = _coerce_float((stageA or {}).get("val_loss") if isinstance(stageA, dict) else None)
    stageB_val = _coerce_float((stageB or {}).get("val_loss") if isinstance(stageB, dict) else None)
    truth_rmse_rel = _coerce_float((truth_eval or {}).get("rmse_rel") if isinstance(truth_eval, dict) else None)
    truth_frac_valid = _coerce_float((truth_eval or {}).get("frac_valid") if isinstance(truth_eval, dict) else None)
    stageC_expr = None
    if isinstance(stageC, dict):
        stageC_expr = stageC.get("y_expr_str") or stageC.get("phi_expr_str")

    if expect.get("stageA_required") and not isinstance(stageA, dict):
        reasons.append("missing stageA block")
    if expect.get("stageB_required") and not isinstance(stageB, dict):
        reasons.append("missing stageB block")
    if expect.get("stageC_required") and not isinstance(stageC, dict):
        reasons.append("missing stageC block")
    if expect.get("truth_eval_required") and not isinstance(truth_eval, dict):
        reasons.append("missing truth_eval block")
    if not selection_eligible:
        reasons.append(
            "final selection is not eligible for success"
            + (f": {selection_reason}" if selection_reason else "")
        )

    if "stageA_val_loss_max" in expect and stageA_val is not None:
        if stageA_val > float(expect["stageA_val_loss_max"]):
            reasons.append(
                f"stageA.val_loss {stageA_val:.3e} > {float(expect['stageA_val_loss_max']):.3e}"
            )
    elif "stageA_val_loss_max" in expect:
        reasons.append("stageA.val_loss missing")
    if "stageB_val_loss_max" in expect and stageB_val is not None:
        if stageB_val > float(expect["stageB_val_loss_max"]):
            reasons.append(
                f"stageB.val_loss {stageB_val:.3e} > {float(expect['stageB_val_loss_max']):.3e}"
            )
    elif "stageB_val_loss_max" in expect:
        reasons.append("stageB.val_loss missing")

    if "truth_success" in expect:
        if truth_success != bool(expect["truth_success"]):
            reasons.append(f"truth_eval.success={truth_success} != expected {bool(expect['truth_success'])}")
    if "truth_rmse_rel_max" in expect:
        if truth_rmse_rel is None:
            reasons.append("truth_eval.rmse_rel missing")
        elif truth_rmse_rel > float(expect["truth_rmse_rel_max"]):
            reasons.append(
                f"truth_eval.rmse_rel {truth_rmse_rel:.3e} > {float(expect['truth_rmse_rel_max']):.3e}"
            )
    if "truth_frac_valid_min" in expect:
        if truth_frac_valid is None:
            reasons.append("truth_eval.frac_valid missing")
        elif truth_frac_valid < float(expect["truth_frac_valid_min"]):
            reasons.append(
                f"truth_eval.frac_valid {truth_frac_valid:.3f} < {float(expect['truth_frac_valid_min']):.3f}"
            )

    expr_target = expect.get("stageC_equivalent_expr")
    expr_match = None
    if expr_target is not None:
        expr_match = expressions_equivalent(stageC_expr, str(expr_target))
        if not expr_match:
            reasons.append(f"stageC expression not equivalent to expected {expr_target!r}")

    return {
        "success": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {
            "stageA_val_loss": stageA_val,
            "stageB_val_loss": stageB_val,
            "truth_rmse_rel": truth_rmse_rel,
            "truth_frac_valid": truth_frac_valid,
            "truth_success": truth_success,
            "raw_truth_success": raw_truth_success,
            "final_selection_eligible": selection_eligible,
            "final_selection_ineligible_reason": selection_reason,
            "stageC_expr": stageC_expr,
            "stageC_expr_equivalent": expr_match,
        },
    }


def evaluate_de_report(
    kind: str,
    report: dict[str, Any],
    expect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expect = dict(expect or {})
    reasons: list[str] = []

    if kind == "sr_firstclass_de":
        de_block = dict(report.get("de") or {})
        coeff_map, coeff_source = extract_sr_de_coeff_map(report)
        enabled = bool(de_block.get("enabled", False))
        order = int(de_block.get("order", -1) or -1)
        rms_val_mean = _mean(de_block.get("rms_val"))
        stageb_val = _coerce_float((de_block.get("stageB") or {}).get("val_loss"))
    else:
        de_block = dict(report.get("de_discovery") or {})
        coeff_map, coeff_source = extract_run_de_coeff_map(report)
        enabled = True
        order = int(de_block.get("order", -1) or -1)
        rms_val_mean = _coerce_float(de_block.get("rms_val"))
        stageb_val = _coerce_float((de_block.get("stageb_residual") or {}).get("val_loss"))

    if "de_enabled" in expect and enabled != bool(expect["de_enabled"]):
        reasons.append(f"de.enabled={enabled} != expected {bool(expect['de_enabled'])}")
    if "de_order" in expect and order != int(expect["de_order"]):
        reasons.append(f"de.order={order} != expected {int(expect['de_order'])}")
    if "de_rms_val_max" in expect:
        if rms_val_mean is None:
            reasons.append("de.rms_val missing")
        elif rms_val_mean > float(expect["de_rms_val_max"]):
            reasons.append(f"de.rms_val_mean {rms_val_mean:.3e} > {float(expect['de_rms_val_max']):.3e}")
    if "de_stageb_val_loss_max" in expect:
        if stageb_val is None:
            reasons.append("de.stageB.val_loss missing")
        elif stageb_val > float(expect["de_stageb_val_loss_max"]):
            reasons.append(
                f"de.stageB.val_loss {stageb_val:.3e} > {float(expect['de_stageb_val_loss_max']):.3e}"
            )

    coeff_expect = dict(expect.get("de_expected_coefficients") or {})
    for term, spec in coeff_expect.items():
        spec_dict = dict(spec or {})
        if term not in coeff_map:
            reasons.append(f"missing coefficient term {term!r}")
            continue
        actual = float(coeff_map[term])
        target = float(spec_dict.get("value", 0.0))
        abs_tol = float(spec_dict.get("abs_tol", 0.0))
        if abs(actual - target) > abs_tol:
            reasons.append(f"coeff[{term}]={actual:.6g} differs from {target:.6g} by > {abs_tol:.3g}")

    if "de_other_terms_abs_max" in expect:
        limit = float(expect["de_other_terms_abs_max"])
        bad_terms = [
            (term, float(coeff))
            for term, coeff in coeff_map.items()
            if term not in coeff_expect and abs(float(coeff)) > limit
        ]
        if bad_terms:
            reasons.append(
                "unexpected terms exceed tolerance: "
                + ", ".join(f"{term}={coeff:.3g}" for term, coeff in bad_terms)
            )

    metrics: dict[str, Any] = {
        "de_enabled": enabled,
        "de_order": order,
        "de_rms_val_mean": rms_val_mean,
        "de_stageb_val_loss": stageb_val,
        "de_coeff_map": coeff_map,
        "de_coeff_source_keys": sorted(str(k) for k in coeff_source.keys()),
    }
    for term, coeff in coeff_map.items():
        metrics[f"de_coeff_{_metric_key_from_term(term)}"] = float(coeff)

    return {
        "success": len(reasons) == 0,
        "reasons": reasons,
        "metrics": metrics,
    }


def compare_case_summaries(
    current_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    *,
    wall_time_factor: float,
    metric_factor: float,
) -> list[dict[str, Any]]:
    baseline_map = {str(row.get("case_id", "")): row for row in list(baseline_rows or [])}
    regressions: list[dict[str, Any]] = []
    lower_is_better = [
        "truth_rmse_rel",
        "stageA_val_loss",
        "stageB_val_loss",
        "de_rms_val_mean",
        "de_stageb_val_loss",
    ]

    for row in list(current_rows or []):
        case_id = str(row.get("case_id", "") or "")
        if not case_id:
            continue
        base = baseline_map.get(case_id)
        if base is None:
            continue

        reasons: list[str] = []
        cur_success = bool(row.get("success", False))
        base_success = bool(base.get("success", False))
        if base_success and not cur_success:
            reasons.append("case now fails but baseline passed")

        cur_wall = _coerce_float(row.get("wall_seconds"))
        base_wall = _coerce_float(base.get("wall_seconds"))
        if (
            cur_success
            and base_success
            and cur_wall is not None
            and base_wall is not None
            and cur_wall > max(1.0e-12, base_wall) * float(wall_time_factor)
        ):
            reasons.append(
                f"wall_seconds {cur_wall:.3f}s > {float(wall_time_factor):.2f}x baseline {base_wall:.3f}s"
            )

        for key in lower_is_better:
            cur_val = _coerce_float(row.get(key))
            base_val = _coerce_float(base.get(key))
            if cur_val is None or base_val is None:
                continue
            if cur_val > max(1.0e-12, base_val) * float(metric_factor):
                reasons.append(
                    f"{key} {cur_val:.3e} > {float(metric_factor):.2f}x baseline {base_val:.3e}"
                )

        if reasons:
            regressions.append(
                {
                    "case_id": case_id,
                    "reasons": reasons,
                    "current": row,
                    "baseline": base,
                }
            )

    return regressions


def evaluate_command_case(
    *,
    case: dict[str, Any],
    returncode: int,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    expect = dict(case.get("expect") or {})
    reasons: list[str] = []
    combined = f"{stdout}\n{stderr}"

    expected_returncode = int(expect.get("returncode", 0))
    if int(returncode) != expected_returncode:
        reasons.append(f"returncode {int(returncode)} != expected {expected_returncode}")

    for token in list(expect.get("stdout_must_contain") or []):
        tok = str(token)
        if tok not in combined:
            reasons.append(f"missing required output token {tok!r}")

    return {
        "success": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {},
    }


def _case_filepaths(case: dict[str, Any]) -> list[str]:
    if case.get("filepaths"):
        return [str(p) for p in list(case.get("filepaths") or [])]
    if case.get("filepath"):
        return [str(case.get("filepath"))]
    raise ValueError(f"Case {case.get('case_id', '<unknown>')} missing filepath/filepaths")


def _derive_base_filename(paths: Sequence[str]) -> str:
    if len(paths) == 1:
        return pathlib.Path(paths[0]).stem
    stems = [pathlib.Path(p).stem for p in paths]
    common = os.path.commonprefix(stems).rstrip("_-.")
    if common and len(common) >= 3:
        return f"{common}_multi{len(paths)}"
    return f"multi{len(paths)}_{stems[0]}"


def _case_timeout(case: dict[str, Any], defaults: dict[str, Any]) -> float | None:
    raw = case.get("timeout_s", defaults.get("timeout_s"))
    return float(raw) if raw is not None else None


def build_case_command(
    case: dict[str, Any],
    *,
    case_dir: pathlib.Path,
    python_executable: str,
) -> tuple[list[str], pathlib.Path | None]:
    kind = str(case.get("kind", "sr"))
    extra_args = [str(v) for v in list(case.get("args") or [])]

    if kind in {"sr", "sr_firstclass_de"}:
        filepaths = _case_filepaths(case)
        report_path = case_dir / f"{str(case['case_id'])}.report.json"
        cmd = [python_executable, str(RUN_SR_SCRIPT)]
        if len(filepaths) == 1:
            cmd.extend(["--filepath", filepaths[0]])
        else:
            cmd.extend(["--filepaths", *filepaths])
        cmd.extend(extra_args)
        cmd.extend(["--report_json", str(report_path)])
        return cmd, report_path

    if kind == "de":
        filepaths = _case_filepaths(case)
        report_path = case_dir / f"{_derive_base_filename(filepaths)}_de.json"
        cmd = [python_executable, str(RUN_DE_SCRIPT)]
        if len(filepaths) == 1:
            cmd.extend(["--filepath", filepaths[0]])
        else:
            cmd.extend(["--filepaths", *filepaths])
        cmd.extend(extra_args)
        cmd.extend(["--output_dir", str(case_dir), "--save_json"])
        return cmd, report_path

    if kind == "python":
        script = str(case.get("script", "") or "")
        if not script:
            raise ValueError(f"Python case {case.get('case_id', '<unknown>')} missing script")
        cmd = [python_executable, script, *extra_args]
        return cmd, None

    if kind == "pytest":
        paths = [str(p) for p in list(case.get("paths") or [])]
        if not paths:
            raise ValueError(f"Pytest case {case.get('case_id', '<unknown>')} missing paths")
        cmd = [python_executable, "-m", "pytest", "-q", *paths, *extra_args]
        return cmd, None

    raise ValueError(f"Unsupported case kind: {kind}")


def run_case(
    case: dict[str, Any],
    *,
    output_dir: pathlib.Path,
    defaults: dict[str, Any] | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    defaults = dict(defaults or {})
    case_dir = output_dir / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "case.log"

    py_exec = str(python_executable or defaults.get("python_executable") or sys.executable)
    cmd, report_path = build_case_command(case, case_dir=case_dir, python_executable=py_exec)
    timeout_s = _case_timeout(case, defaults)

    started = time.perf_counter()
    returncode = 0
    stdout = ""
    stderr = ""
    reasons: list[str] = []
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        returncode = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        reasons.append(f"timed out after {float(timeout_s):.1f}s")
    wall_seconds = float(time.perf_counter() - started)

    log_lines = [
        f"case_id: {case['case_id']}",
        f"kind: {case['kind']}",
        f"command: {shlex.join(cmd)}",
        f"returncode: {returncode}",
        f"wall_seconds: {wall_seconds:.6f}",
    ]
    if timeout_s is not None:
        log_lines.append(f"timeout_s: {float(timeout_s):.1f}")
    log_lines.extend(["", "=== STDOUT ===", stdout, "", "=== STDERR ===", stderr])
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    if returncode != 0:
        reasons.append(f"command exited with returncode {returncode}")
    if report_path is not None and not report_path.exists():
        reasons.append(f"missing report JSON: {report_path}")

    metrics: dict[str, Any] = {}
    report: dict[str, Any] | None = None
    if report_path is not None and report_path.exists():
        try:
            report = _load_json(report_path)
        except Exception as e:
            reasons.append(f"failed to parse report JSON: {e}")

    try:
        if report is not None:
            expect = dict(case.get("expect") or {})
            if case["kind"] == "sr":
                evaluation = evaluate_sr_report(report, expect)
            else:
                evaluation = evaluate_de_report(str(case["kind"]), report, expect)
        else:
            evaluation = evaluate_command_case(
                case=case,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )
        reasons.extend(list(evaluation.get("reasons") or []))
        metrics.update(dict(evaluation.get("metrics") or {}))
    except Exception as e:
        reasons.append(f"evaluation failed: {e}")

    row = {
        "case_id": str(case["case_id"]),
        "kind": str(case["kind"]),
        "success": len(reasons) == 0,
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "wall_seconds": wall_seconds,
        "report_path": str(report_path) if report_path is not None else None,
        "log_path": str(log_path),
        "command": cmd,
        "reasons": reasons,
    }
    row.update(metrics)
    return row


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen acceptance suite for core SR/DE workflows")
    p.add_argument("--suite_manifest", type=str, default=str(DEFAULT_SUITE_MANIFEST))
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--python_executable", type=str, default=None)
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--bless_baseline", type=str, default=None)
    p.add_argument("--regression_wall_time_factor", type=float, default=2.0)
    p.add_argument("--regression_metric_factor", type=float, default=10.0)
    p.add_argument("--fail_on_regression", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path, manifest = load_acceptance_suite(args.suite_manifest)
    defaults = dict(manifest.get("defaults") or {})
    cases = resolve_suite_cases(manifest, manifest_path=manifest_path)

    case_filter = _parse_case_filter(args.only)
    if case_filter is not None:
        cases = [case for case in cases if str(case.get("case_id")) in case_filter]
        if not cases:
            raise ValueError(f"No suite cases matched --only={args.only}")

    suite_id = str(manifest.get("suite_id", "core_acceptance") or "core_acceptance")
    output_dir = pathlib.Path(args.output_dir or (REPO_ROOT / "results" / f"core_acceptance_{suite_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        run_case(
            case,
            output_dir=output_dir,
            defaults=defaults,
            python_executable=args.python_executable,
        )
        for case in cases
    ]

    payload = {
        "suite_id": suite_id,
        "suite_manifest": str(manifest_path),
        "description": str(manifest.get("description", "") or ""),
        "defaults": defaults,
        "n_cases": int(len(rows)),
        "n_pass": int(sum(1 for row in rows if bool(row.get("success", False)))),
        "n_fail": int(sum(1 for row in rows if not bool(row.get("success", False)))),
        "case_summary": rows,
    }
    results_json_path = output_dir / "core_acceptance_results.json"
    _write_json(payload, results_json_path)
    _write_csv(rows, output_dir / "core_acceptance_case_summary.csv")

    regressions: list[dict[str, Any]] = []
    if args.baseline:
        baseline_payload = _load_json(pathlib.Path(args.baseline))
        regressions = compare_case_summaries(
            rows,
            list(baseline_payload.get("case_summary") or []),
            wall_time_factor=float(args.regression_wall_time_factor),
            metric_factor=float(args.regression_metric_factor),
        )
        compare_payload = {
            "baseline": str(args.baseline),
            "regression_wall_time_factor": float(args.regression_wall_time_factor),
            "regression_metric_factor": float(args.regression_metric_factor),
            "n_regressions": int(len(regressions)),
            "regressions": regressions,
        }
        _write_json(compare_payload, output_dir / "core_acceptance_compare.json")

    blessed_baseline_path: pathlib.Path | None = None
    if args.bless_baseline:
        blessed_baseline_path = _resolve_output_json_path(
            args.bless_baseline,
            default_name=f"{suite_id}.baseline.json",
        )
        blessed_payload = dict(payload)
        blessed_payload["blessed_baseline"] = {
            "source_results_json": str(results_json_path),
            "source_output_dir": str(output_dir),
        }
        _write_json(blessed_payload, blessed_baseline_path)

    failures = [row for row in rows if not bool(row.get("success", False))]

    print(f"[acceptance] suite={suite_id} cases={len(rows)} pass={payload['n_pass']} fail={payload['n_fail']}")
    print(f"[acceptance] outputs written to {output_dir}")
    if blessed_baseline_path is not None:
        print(f"[acceptance] blessed baseline written to {blessed_baseline_path}")
    if failures:
        for row in failures:
            reason_str = "; ".join(str(v) for v in list(row.get("reasons") or []))
            print(f"[acceptance] FAIL {row['case_id']} :: {reason_str}")
    else:
        print("[acceptance] all cases passed")

    if regressions:
        print(f"[acceptance] regressions={len(regressions)}")
        for row in regressions:
            print(f"[acceptance] REGRESSION {row['case_id']} :: {'; '.join(row['reasons'])}")
    elif args.baseline:
        print("[acceptance] no regressions flagged")

    if failures:
        return 1
    if regressions and args.fail_on_regression:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
