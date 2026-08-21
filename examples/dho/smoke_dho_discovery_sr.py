#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
End-to-end DHO discovery through run_SR.py first-class DE mode:
1) Ensure/generate DHO trajectory CSV from raw dynamics
2) Run run_SR.py with --discover_de
3) Verify recovered DE coefficients from the DE artifact payload
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], description: str) -> bool:
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    return result.returncode == 0


def _extract_coeff_map_from_de_pkl(de_pkl: Path) -> tuple[dict[str, float], dict]:
    with open(de_pkl, "rb") as f:
        payload = pickle.load(f)
    term_asts = list(payload.get("term_asts", []))
    coeffs = payload.get("coeffs", None)
    if coeffs is None:
        raise ValueError(f"Missing coeffs in DE payload: {de_pkl}")
    if hasattr(coeffs, "detach"):
        coeffs = coeffs.detach().cpu()
    # Single-dataset shape (1,K) or (K,)
    if getattr(coeffs, "ndim", 0) == 2:
        coeff_row = coeffs[0].tolist()
    else:
        coeff_row = list(coeffs)
    coeff_map: dict[str, float] = {}
    for term, c in zip(term_asts, coeff_row):
        coeff_map[repr(term)] = float(c)
    return coeff_map, payload


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_datafile = repo_root / "data" / "dho_sr.csv"
    run_sr_script = repo_root / "nestynet_sr" / "run_SR.py"
    gen_script = script_dir / "generate_dho.py"
    default_results_dir = repo_root / "results"

    parser = argparse.ArgumentParser(
        description="Run end-to-end DHO discovery via run_SR.py --discover_de",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datafile", type=str, default=str(default_datafile), help="Input CSV path")
    parser.add_argument("--results_dir", type=str, default=str(default_results_dir), help="Output directory")
    parser.add_argument("--generate", action="store_true", help="Generate dataset before discovery")
    parser.add_argument("--gamma", type=float, default=1.6, help="Ground-truth damping for checks")
    parser.add_argument("--omega", type=float, default=1.0, help="Ground-truth frequency for checks")
    parser.add_argument("--t_max", type=float, default=6.0, help="Max x0 when generating data")
    parser.add_argument("--coeff_tol_u", type=float, default=0.15, help="Tolerance for u coefficient")
    parser.add_argument("--coeff_tol_du", type=float, default=0.15, help="Tolerance for u_x0 coefficient")
    parser.add_argument("--decoy_tol", type=float, default=0.05, help="Tolerance for nuisance coefficients")
    args = parser.parse_args()

    data_path = Path(args.datafile).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.generate or (not data_path.exists()):
        gen_cmd = [
            sys.executable,
            str(gen_script),
            "--output",
            str(data_path),
            "--gamma",
            str(float(args.gamma)),
            "--omega",
            str(float(args.omega)),
            "--t_max",
            str(float(args.t_max)),
            "--n_points",
            "5000",
        ]
        ok = _run(gen_cmd, "Generating DHO data for run_SR")
        if not ok:
            print("[DHO/SR] Data generation failed.")
            return 1

    if not data_path.exists():
        print(f"[DHO/SR] Missing dataset: {data_path}")
        return 1

    # Keep SR Stage B off for speed; run DE Stage B with a small cap for stability.
    run_cmd = [
        sys.executable,
        str(run_sr_script),
        "--filepath",
        str(data_path),
        "--fast",
        "--no_stageB",
        "--discover_de",
        "--de_order_candidates",
        "2",
        "--de_include_du",
        "--de_no_x",
        "--de_no_xu",
        "--de_no_xdu",
        "--de_stlsq_lambda",
        "1e-3",
        "--de_ridge",
        "1e-10",
        "--de_stageB_max_outer_iters",
        "1",
        "--de_stageB_epochs",
        "200",
        "--force_y_ops",
        "identity",
        "--batch_size",
        "2000",
        "--ndata_train",
        "2000",
        "--ndata_val",
        "2000",
        "--report_json",
        str(results_dir / f"{data_path.stem}.report.json"),
    ]
    ok = _run(run_cmd, "Running DHO via run_SR.py --discover_de")
    if not ok:
        print("[DHO/SR] run_SR.py failed.")
        return 1

    report_json = results_dir / f"{data_path.stem}.report.json"
    if not report_json.exists():
        print(f"[DHO/SR] Missing report JSON: {report_json}")
        return 1

    report = json.loads(report_json.read_text(encoding="utf-8"))
    de_block = report.get("de", {})
    artifacts = de_block.get("artifacts", {}) if isinstance(de_block, dict) else {}
    de_pkl = Path(str(artifacts.get("pkl", "")))
    de_human = Path(str(artifacts.get("human", "")))
    if not de_pkl.exists():
        print(f"[DHO/SR] Missing DE artifact pkl: {de_pkl}")
        return 1
    if not de_human.exists():
        print(f"[DHO/SR] Missing DE artifact human file: {de_human}")
        return 1

    coeff_map, de_payload = _extract_coeff_map_from_de_pkl(de_pkl)
    order = int(de_payload.get("order", -1))
    if order != 2:
        print(f"[DHO/SR] FAIL: expected order=2, got {order}")
        return 1

    c_u = coeff_map.get("u", None)
    c_du = coeff_map.get("u_x0", None)
    if c_u is None:
        print("[DHO/SR] FAIL: missing 'u' term in discovered DE payload")
        return 1
    if c_du is None:
        print("[DHO/SR] FAIL: missing 'u_x0' term in discovered DE payload")
        return 1

    expected_u = float(args.omega) ** 2
    expected_du = float(args.gamma)
    if abs(float(c_u) - expected_u) > float(args.coeff_tol_u):
        print(
            f"[DHO/SR] FAIL: u coefficient mismatch. expected~{expected_u:.6g}, got {float(c_u):.6g}, "
            f"tol={float(args.coeff_tol_u):.3g}"
        )
        return 1
    if abs(float(c_du) - expected_du) > float(args.coeff_tol_du):
        print(
            f"[DHO/SR] FAIL: u_x0 coefficient mismatch. expected~{expected_du:.6g}, got {float(c_du):.6g}, "
            f"tol={float(args.coeff_tol_du):.3g}"
        )
        return 1

    decoy_terms = [k for k in coeff_map.keys() if k not in {"u", "u_x0"}]
    bad_decoys = [k for k in decoy_terms if abs(float(coeff_map[k])) > float(args.decoy_tol)]
    if bad_decoys:
        print("[DHO/SR] FAIL: nuisance terms too large:")
        for k in bad_decoys:
            print(f"  {k}: {float(coeff_map[k]):.6g} (tol={float(args.decoy_tol):.3g})")
        return 1

    eqn = ""
    eqn_raw = de_block.get("eqn_raw", []) if isinstance(de_block, dict) else []
    if isinstance(eqn_raw, list) and eqn_raw:
        eqn = str(eqn_raw[0])
    print("\n" + "=" * 70)
    print("DHO run_SR + first-class DE summary")
    print("=" * 70)
    print(f"Recovered equation: {eqn or '<missing>'}")
    print(f"Expected: u_x0x0 + {expected_u:.6g}*u + {expected_du:.6g}*u_x0 = 0")
    print(f"Recovered coefficients: u={float(c_u):.6g}, u_x0={float(c_du):.6g}")
    if decoy_terms:
        print("Nuisance coefficients:")
        for k in decoy_terms:
            print(f"  {k}: {float(coeff_map[k]):.6g}")
    print("\nPASS: DHO DE discovered via run_SR.py --discover_de")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
