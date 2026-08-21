#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Lane-Emden discovery runner (modernized for current DE library).

This script runs three stages:
1. Baseline sparse DE discovery
2. Power-template search (heuristic ψ init)
3. Power-template search with LM-over-ψ

Compared to the historical script, this version uses the explicit singular term
`x^-1 * u_x` in the library (`--include_inv_xdu`) and avoids the ξ=0 singular
point by generating data on [xi_min, xi_max] with xi_min > 0.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], title: str) -> tuple[bool, str]:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Command: {' '.join(cmd)}\n")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print("STDERR:", proc.stderr, file=sys.stderr)
    return proc.returncode == 0, proc.stdout + "\n" + proc.stderr


def _safe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_coeff_map(report_json: Path) -> tuple[dict[str, float], dict]:
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    de = payload.get("de_discovery", {})
    terms = list(de.get("terms", []))
    coeffs = list(de.get("coefficients", []))
    coeff_map: dict[str, float] = {}
    for term, coeff in zip(terms, coeffs):
        coeff_map[str(term)] = float(coeff)
    return coeff_map, de


def find_inverse_du_term(coeff_map: dict[str, float]) -> str | None:
    for term in coeff_map:
        if "u_x0" in term and "** -1" in term:
            return term
    return None


def copy_result_artifacts(output_dir: Path, stem: str, tag: str) -> None:
    src_json = output_dir / f"{stem}_de.json"
    src_human = output_dir / f"{stem}_de.human"
    if src_json.exists():
        shutil.copy2(src_json, output_dir / f"{stem}_{tag}_de.json")
    if src_human.exists():
        shutil.copy2(src_human, output_dir / f"{stem}_{tag}_de.human")


def summarize_stage(
    stage_name: str,
    coeff_map: dict[str, float],
    de_meta: dict,
    n_true: float,
    baseline_rms_val: float | None = None,
    expect_linear_u: bool | None = None,
) -> tuple[str, str]:
    inv_term = find_inverse_du_term(coeff_map)
    inv_coeff = float(coeff_map.get(inv_term, 0.0)) if inv_term is not None else 0.0
    u_coeff = float(coeff_map.get("u", 0.0))
    rms_val = de_meta.get("rms_val", None)
    rms_val_f = float(rms_val) if rms_val is not None else None
    varpro_meta = de_meta.get("varpro_metadata", {}) or {}
    template_params = varpro_meta.get("template_params", {}) or {}

    issues: list[str] = []
    critical = False

    if inv_term is None:
        critical = True
        issues.append("missing x^-1*u_x term")
    else:
        rel_inv = abs(inv_coeff - 2.0) / 2.0
        if rel_inv > 0.35:
            issues.append(f"inv coeff off: expected ~2, got {inv_coeff:.4f}")

    expect_linear = bool(abs(n_true - 1.0) < 1e-9) if expect_linear_u is None else bool(expect_linear_u)
    if expect_linear:
        rel_u = abs(u_coeff - 1.0)
        if rel_u > 0.35:
            issues.append(f"u coeff off: expected ~1, got {u_coeff:.4f}")
    else:
        p_val = None
        for key, value in template_params.items():
            if str(key).startswith("p"):
                try:
                    p_val = float(value)
                    break
                except Exception:
                    continue
        if p_val is None:
            issues.append("no recovered power exponent in varpro metadata")
        else:
            if abs(p_val - n_true) > 0.35:
                issues.append(f"power exponent off: expected ~{n_true:.3f}, got {p_val:.4f}")

    if baseline_rms_val is not None and rms_val_f is not None and "template" in stage_name.lower():
        if rms_val_f > baseline_rms_val:
            issues.append(
                f"validation RMS did not improve over baseline ({rms_val_f:.3e} > {baseline_rms_val:.3e})"
            )

    if critical:
        return "FAIL", "; ".join(issues)
    if issues:
        return "PARTIAL", "; ".join(issues)
    return "PASS", "coefficients and residual checks look good"


def extract_power_param_from_meta(de_meta: dict) -> float | None:
    varpro_meta = de_meta.get("varpro_metadata", {}) or {}
    template_params = varpro_meta.get("template_params", {}) or {}
    if "p" in template_params:
        try:
            return float(template_params["p"])
        except Exception:
            return None
    for key, value in template_params.items():
        if str(key).startswith("p"):
            try:
                return float(value)
            except Exception:
                continue
    return None


def count_free_parameters(de_meta: dict) -> int:
    num_terms = int(de_meta.get("num_terms", len(de_meta.get("coefficients", [])) or 0))
    varpro_meta = de_meta.get("varpro_metadata", {}) or {}
    template_params = varpro_meta.get("template_params", {}) or {}
    n_template_params = 0
    for _, value in template_params.items():
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            n_template_params += 1
    return num_terms + n_template_params


def compute_bic_from_rms(rms_val: float | None, n_obs: int, k_params: int) -> float | None:
    if rms_val is None:
        return None
    if n_obs <= 1:
        return None
    if not math.isfinite(float(rms_val)):
        return None
    mse = max(float(rms_val) ** 2, 1e-300)
    return float(n_obs * math.log(mse) + k_params * math.log(float(n_obs)))


def identifiability_screen(
    de_meta: dict,
    *,
    cond_max: float,
    near_linear_tol: float,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    varpro_meta = de_meta.get("varpro_metadata", {}) or {}
    condition_number = varpro_meta.get("condition_number", None)
    if condition_number is not None:
        try:
            cond_val = float(condition_number)
        except Exception:
            cond_val = float("inf")
        if not math.isfinite(cond_val):
            issues.append("non-finite template condition number")
        elif cond_val > cond_max:
            issues.append(f"condition number too high ({cond_val:.2f} > {cond_max:.2f})")

    p_val = extract_power_param_from_meta(de_meta)
    if p_val is not None and math.isfinite(p_val):
        if abs(p_val - 1.0) < near_linear_tol:
            issues.append(
                f"power exponent nearly linear ({p_val:.4f}); nonlinear branch not identifiable"
            )
    return len(issues) == 0, issues


def _choose_model_branch(
    *,
    linear_candidate: dict | None,
    nonlinear_candidate: dict | None,
    n_obs: int,
    bic_delta: float,
    cond_max: float,
    near_linear_tol: float,
) -> tuple[str, str, dict]:
    details: dict[str, object] = {
        "n_obs": int(n_obs),
        "bic_delta_required": float(bic_delta),
        "cond_max": float(cond_max),
        "near_linear_tol": float(near_linear_tol),
        "candidates": {},
        "selected": None,
        "reason": "",
    }

    def _candidate_payload(c: dict | None) -> dict | None:
        if c is None:
            return None
        de_meta = c["de_meta"]
        rms_val = de_meta.get("rms_val", None)
        rms_val_f = float(rms_val) if rms_val is not None else None
        k_params = count_free_parameters(de_meta)
        bic = compute_bic_from_rms(rms_val_f, n_obs=n_obs, k_params=k_params)
        out = {
            "label": c["label"],
            "tag": c["tag"],
            "rms_val": rms_val_f,
            "k_params": int(k_params),
            "bic": bic,
        }
        if c.get("is_nonlinear", False):
            ok_ident, ident_issues = identifiability_screen(
                de_meta,
                cond_max=cond_max,
                near_linear_tol=near_linear_tol,
            )
            out["identifiability_ok"] = bool(ok_ident)
            out["identifiability_issues"] = ident_issues
        return out

    lin_payload = _candidate_payload(linear_candidate)
    nonlin_payload = _candidate_payload(nonlinear_candidate)
    if lin_payload is not None:
        details["candidates"]["linear"] = lin_payload
    if nonlin_payload is not None:
        details["candidates"]["nonlinear"] = nonlin_payload

    if linear_candidate is None and nonlinear_candidate is None:
        details["reason"] = "no successful candidate models"
        return "FAIL", details["reason"], details
    if linear_candidate is None:
        details["selected"] = "nonlinear"
        details["reason"] = "only nonlinear branch available"
        return "PASS", str(details["reason"]), details
    if nonlinear_candidate is None:
        details["selected"] = "linear"
        details["reason"] = "only linear branch available"
        return "PASS", str(details["reason"]), details

    bic_lin = lin_payload.get("bic", None) if lin_payload else None
    bic_non = nonlin_payload.get("bic", None) if nonlin_payload else None
    ident_ok = bool(nonlin_payload.get("identifiability_ok", True)) if nonlin_payload else True

    if bic_lin is None and bic_non is None:
        if ident_ok:
            details["selected"] = "nonlinear"
            details["reason"] = "BIC unavailable for both; selected nonlinear (identifiable)"
            return "PARTIAL", str(details["reason"]), details
        details["selected"] = "linear"
        details["reason"] = "BIC unavailable and nonlinear branch failed identifiability"
        return "PARTIAL", str(details["reason"]), details

    if bic_lin is None and bic_non is not None:
        if ident_ok:
            details["selected"] = "nonlinear"
            details["reason"] = "linear BIC unavailable; selected nonlinear"
            return "PARTIAL", str(details["reason"]), details
        details["selected"] = "linear"
        details["reason"] = "linear BIC unavailable and nonlinear branch failed identifiability"
        return "PARTIAL", str(details["reason"]), details

    if bic_non is None:
        details["selected"] = "linear"
        details["reason"] = "nonlinear BIC unavailable; selected linear"
        return "PARTIAL", str(details["reason"]), details

    assert bic_lin is not None and bic_non is not None
    if ident_ok and (bic_non + bic_delta < bic_lin):
        details["selected"] = "nonlinear"
        details["reason"] = (
            f"nonlinear selected by BIC ({bic_non:.3f} + {bic_delta:.2f} < {bic_lin:.3f})"
        )
        return "PASS", str(details["reason"]), details

    details["selected"] = "linear"
    if not ident_ok:
        ident_issues = nonlin_payload.get("identifiability_issues", []) if nonlin_payload else []
        details["reason"] = "nonlinear branch rejected by identifiability: " + "; ".join(ident_issues)
    else:
        details["reason"] = (
            f"linear selected by BIC penalty ({bic_non:.3f} + {bic_delta:.2f} >= {bic_lin:.3f})"
        )
    return "PASS", str(details["reason"]), details


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Run Lane-Emden discovery tests with modern DE-library settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datafile", type=Path, default=repo_root / "data" / "lane_emden.csv")
    parser.add_argument("--output_dir", type=Path, default=repo_root / "results" / "lane_emden")
    parser.add_argument("--n", type=float, default=1.5, help="Polytropic index for generation")
    parser.add_argument("--xi_min", type=float, default=0.2, help="Minimum ξ used in data generation")
    parser.add_argument("--xi_max", type=float, default=2.0, help="Maximum ξ used in data generation")
    parser.add_argument("--noise", type=float, default=0.0, help="Relative noise level for y")
    parser.add_argument("--generate", action="store_true", help="Generate data before running")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline STLSQ run")
    parser.add_argument("--skip_heuristic", action="store_true", help="Skip template heuristic run")
    parser.add_argument("--skip_lm", action="store_true", help="Skip template LM run")
    parser.add_argument("--epochs", type=int, default=1200, help="Surrogate training epochs")
    parser.add_argument("--template_lm_epochs", type=int, default=120, help="LM epochs for ψ")
    parser.add_argument(
        "--selection_bic_delta",
        type=float,
        default=2.0,
        help="Minimum BIC advantage required to choose nonlinear branch",
    )
    parser.add_argument(
        "--selection_cond_max",
        type=float,
        default=500.0,
        help="Reject nonlinear branch if template condition number exceeds this value",
    )
    parser.add_argument(
        "--selection_near_linear_tol",
        type=float,
        default=0.1,
        help="Reject nonlinear branch if |p-1| is smaller than this tolerance",
    )
    parser.add_argument(
        "--selection_n_obs",
        type=int,
        default=200,
        help="Effective validation sample size for BIC scoring",
    )
    args = parser.parse_args()

    run_de_script = repo_root / "nestynet_sr" / "run_de.py"
    generate_script = script_dir / "generate_lane_emden.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate:
        ok, _ = run_command(
            [
                sys.executable,
                str(generate_script),
                "--n",
                str(args.n),
                "--xi_min",
                str(args.xi_min),
                "--xi_max",
                str(args.xi_max),
                "--noise",
                str(args.noise),
                "--numerical",
                "--output",
                str(args.datafile),
            ],
            "GENERATING SYNTHETIC LANE-EMDEN DATA",
        )
        if not ok:
            return 1

    if not args.datafile.exists():
        print(f"Data file not found: {args.datafile}")
        print("Run with --generate to create it.")
        return 1

    stem = args.datafile.stem
    result_json = args.output_dir / f"{stem}_de.json"

    common = [
        sys.executable,
        str(run_de_script),
        "--filepath",
        str(args.datafile),
        "--order_candidates",
        "2",
        "--include_inv_xdu",
        "--no_const",
        "--no_x",
        "--no_xu",
        "--epochs",
        str(args.epochs),
        "--epochs_min",
        "200",
        "--nval_patience",
        "200",
        "--loss_target",
        "1e-10",
        "--batch_size",
        "200",
        "--ndata_train",
        "1600",
        "--ndata_val",
        "200",
        "--stlsq_lambda",
        "5e-3",
        "--output_dir",
        str(args.output_dir),
        "--save_json",
    ]

    stage_results: dict[str, tuple[str, str]] = {}
    baseline_rms_val: float | None = None
    linear_candidate: dict | None = None
    nonlinear_heuristic_candidate: dict | None = None
    nonlinear_lm_candidate: dict | None = None

    if not args.skip_baseline:
        ok, _ = run_command(common, "TEST 1: BASELINE STLSQ")
        if not ok:
            stage_results["baseline"] = ("FAIL", "run_de failed")
        elif result_json.exists():
            coeff_map, de_meta = load_coeff_map(result_json)
            baseline_rms_val = (
                float(de_meta["rms_val"]) if de_meta.get("rms_val", None) is not None else None
            )
            stage_results["baseline"] = summarize_stage(
                "baseline", coeff_map, de_meta, n_true=args.n, expect_linear_u=True
            )
            copy_result_artifacts(args.output_dir, stem, "baseline")
            linear_candidate = {
                "label": "baseline_linear",
                "tag": "baseline",
                "de_meta": de_meta,
                "json_path": args.output_dir / f"{stem}_baseline_de.json",
                "human_path": args.output_dir / f"{stem}_baseline_de.human",
                "is_nonlinear": False,
            }
        else:
            stage_results["baseline"] = ("FAIL", f"missing {result_json}")

    if not args.skip_heuristic:
        ok, _ = run_command(
            common
            + [
                "--no_u",
                "--varpro",
                "--varpro_templates",
                "power",
                "--max_templates",
                "4",
                "--prefer_autonomous",
            ],
            "TEST 2: NONLINEAR BRANCH (POWER TEMPLATE, HEURISTIC PSI)",
        )
        if not ok:
            stage_results["heuristic"] = ("FAIL", "run_de failed")
        elif result_json.exists():
            coeff_map, de_meta = load_coeff_map(result_json)
            stage_results["heuristic"] = summarize_stage(
                "template_heuristic",
                coeff_map,
                de_meta,
                n_true=args.n,
                baseline_rms_val=baseline_rms_val,
                expect_linear_u=False,
            )
            copy_result_artifacts(args.output_dir, stem, "heuristic")
            nonlinear_heuristic_candidate = {
                "label": "nonlinear_heuristic",
                "tag": "heuristic",
                "de_meta": de_meta,
                "json_path": args.output_dir / f"{stem}_heuristic_de.json",
                "human_path": args.output_dir / f"{stem}_heuristic_de.human",
                "is_nonlinear": True,
            }
        else:
            stage_results["heuristic"] = ("FAIL", f"missing {result_json}")

    if not args.skip_lm:
        ok, _ = run_command(
            common
            + [
                "--no_u",
                "--varpro",
                "--varpro_templates",
                "power",
                "--max_templates",
                "4",
                "--prefer_autonomous",
                "--template_lm",
                "--template_lm_epochs",
                str(args.template_lm_epochs),
                "--template_lm_epochs_min",
                "20",
                "--template_lm_nval_patience",
                "30",
            ],
            "TEST 3: POWER TEMPLATE + LM-OVER-PSI",
        )
        if not ok:
            stage_results["lm"] = ("FAIL", "run_de failed")
        elif result_json.exists():
            coeff_map, de_meta = load_coeff_map(result_json)
            stage_results["lm"] = summarize_stage(
                "template_lm",
                coeff_map,
                de_meta,
                n_true=args.n,
                baseline_rms_val=baseline_rms_val,
                expect_linear_u=False,
            )
            copy_result_artifacts(args.output_dir, stem, "lm")
            nonlinear_lm_candidate = {
                "label": "nonlinear_lm",
                "tag": "lm",
                "de_meta": de_meta,
                "json_path": args.output_dir / f"{stem}_lm_de.json",
                "human_path": args.output_dir / f"{stem}_lm_de.human",
                "is_nonlinear": True,
            }
        else:
            stage_results["lm"] = ("FAIL", f"missing {result_json}")

    nonlinear_candidate = nonlinear_lm_candidate or nonlinear_heuristic_candidate
    selection_status, selection_detail, selection_payload = _choose_model_branch(
        linear_candidate=linear_candidate,
        nonlinear_candidate=nonlinear_candidate,
        n_obs=max(int(args.selection_n_obs), 2),
        bic_delta=float(args.selection_bic_delta),
        cond_max=float(args.selection_cond_max),
        near_linear_tol=float(args.selection_near_linear_tol),
    )
    stage_results["selection"] = (selection_status, selection_detail)

    selected_label = str(selection_payload.get("selected", "") or "")
    selected_candidate = None
    if selected_label == "linear":
        selected_candidate = linear_candidate
    elif selected_label == "nonlinear":
        selected_candidate = nonlinear_candidate

    if selected_candidate is not None:
        src_json = Path(selected_candidate["json_path"])
        src_human = Path(selected_candidate["human_path"])
        _safe_copy(src_json, args.output_dir / f"{stem}_selected_de.json")
        _safe_copy(src_human, args.output_dir / f"{stem}_selected_de.human")
        _safe_copy(src_json, args.output_dir / f"{stem}_de.json")
        _safe_copy(src_human, args.output_dir / f"{stem}_de.human")
        selection_payload["selected_tag"] = selected_candidate["tag"]

    selection_report = args.output_dir / f"{stem}_model_selection.json"
    selection_report.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("LANE-EMDEN SUMMARY")
    print("=" * 72)
    print(f"{'Stage':<14} {'Status':<8} Details")
    print("-" * 72)
    for stage in ("baseline", "heuristic", "lm", "selection"):
        if stage not in stage_results:
            continue
        status, detail = stage_results[stage]
        print(f"{stage:<14} {status:<8} {detail}")
    print("-" * 72)
    print(f"Artifacts directory: {args.output_dir}")
    print(f"Model selection report: {selection_report}")
    print("=" * 72)

    if stage_results.get("selection", ("FAIL", ""))[0] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
