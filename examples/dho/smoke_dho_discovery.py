#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
End-to-end DHO DE discovery:
1) Ensure/generate DHO trajectory CSV from raw dynamics
2) Run run_de.py (surrogate + derivative library + STLSQ)
3) Verify recovered second-order DE coefficients
"""

from __future__ import annotations

import argparse
import json
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


def _extract_coeff_map(report_json: Path) -> tuple[dict[str, float], dict]:
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    de = payload.get("de_discovery", {})
    terms = list(de.get("terms", []))
    coeffs = list(de.get("coefficients", []))
    coeff_map: dict[str, float] = {}
    for term, c in zip(terms, coeffs):
        coeff_map[str(term)] = float(c)
    return coeff_map, de


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_datafile = repo_root / "data" / "dho.csv"
    run_de_script = repo_root / "nestynet_sr" / "run_de.py"
    gen_script = script_dir / "generate_dho.py"
    default_results_dir = repo_root / "results"

    parser = argparse.ArgumentParser(
        description="Run end-to-end DHO DE discovery using run_de.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datafile", type=str, default=str(default_datafile), help="Input CSV path")
    parser.add_argument("--results_dir", type=str, default=str(default_results_dir), help="Output directory")
    parser.add_argument("--generate", action="store_true", help="Generate dataset before discovery")
    parser.add_argument("--gamma", type=float, default=0.4, help="Ground-truth damping for checks")
    parser.add_argument("--omega", type=float, default=2.0, help="Ground-truth frequency for checks")
    parser.add_argument("--coeff_tol_u", type=float, default=0.08, help="Tolerance for u coefficient")
    parser.add_argument("--coeff_tol_du", type=float, default=0.05, help="Tolerance for u_x0 coefficient")
    parser.add_argument("--decoy_tol", type=float, default=0.05, help="Tolerance for nuisance coefficients")
    parser.add_argument("--epochs", type=int, default=3000, help="Surrogate training epochs")
    parser.add_argument("--num_segments", type=int, default=32, help="Surrogate segments")
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
            "--n_points",
            "5000",
        ]
        ok = _run(gen_cmd, "Generating DHO data")
        if not ok:
            print("[DHO] Data generation failed.")
            return 1

    if not data_path.exists():
        print(f"[DHO] Missing dataset: {data_path}")
        return 1

    run_cmd = [
        sys.executable,
        str(run_de_script),
        "--filepath",
        str(data_path),
        "--order_candidates",
        "2",
        "--include_du",
        "--epochs",
        str(int(args.epochs)),
        "--epochs_min",
        "300",
        "--nval_patience",
        "250",
        "--num_segments",
        str(int(args.num_segments)),
        "--loss_target",
        "1e-8",
        "--batch_size",
        "2000",
        "--ndata_train",
        "2000",
        "--ndata_val",
        "2000",
        "--stlsq_lambda",
        "1e-3",
        "--ridge",
        "1e-10",
        "--output_dir",
        str(results_dir),
        "--save_json",
    ]
    ok = _run(run_cmd, "Running DHO DE discovery (run_de.py)")
    if not ok:
        print("[DHO] run_de.py failed.")
        return 1

    base = data_path.stem
    report_json = results_dir / f"{base}_de.json"
    human_txt = results_dir / f"{base}_de.human"
    if not report_json.exists():
        print(f"[DHO] Missing report JSON: {report_json}")
        return 1
    if not human_txt.exists():
        print(f"[DHO] Missing human output: {human_txt}")
        return 1

    coeff_map, de_meta = _extract_coeff_map(report_json)
    expected_u = float(args.omega) ** 2
    expected_du = float(args.gamma)

    order = int(de_meta.get("order", -1))
    if order != 2:
        print(f"[DHO] FAIL: expected order=2, got {order}")
        return 1

    if "u" not in coeff_map:
        print("[DHO] FAIL: missing 'u' term in discovered DE")
        return 1
    if "u_x0" not in coeff_map:
        print("[DHO] FAIL: missing 'u_x0' term in discovered DE")
        return 1

    c_u = float(coeff_map["u"])
    c_du = float(coeff_map["u_x0"])
    if abs(c_u - expected_u) > float(args.coeff_tol_u):
        print(
            f"[DHO] FAIL: u coefficient mismatch. expected~{expected_u:.6g}, got {c_u:.6g}, "
            f"tol={float(args.coeff_tol_u):.3g}"
        )
        return 1
    if abs(c_du - expected_du) > float(args.coeff_tol_du):
        print(
            f"[DHO] FAIL: u_x0 coefficient mismatch. expected~{expected_du:.6g}, got {c_du:.6g}, "
            f"tol={float(args.coeff_tol_du):.3g}"
        )
        return 1

    decoy_terms = [t for t in coeff_map.keys() if t not in {"u", "u_x0"}]
    bad_decoys = [t for t in decoy_terms if abs(coeff_map[t]) > float(args.decoy_tol)]
    if bad_decoys:
        print("[DHO] FAIL: nuisance terms too large:")
        for t in bad_decoys:
            print(f"  {t}: {coeff_map[t]:.6g} (tol={float(args.decoy_tol):.3g})")
        return 1

    print("\n" + "=" * 70)
    print("DHO discovery summary")
    print("=" * 70)
    print(f"Recovered equation: {de_meta.get('canonical_equation', '<missing>')}")
    print(f"Expected: u_x0x0 + {expected_u:.6g}*u + {expected_du:.6g}*u_x0 = 0")
    print(f"Recovered coefficients: u={c_u:.6g}, u_x0={c_du:.6g}")
    if decoy_terms:
        print("Nuisance coefficients:")
        for t in decoy_terms:
            print(f"  {t}: {coeff_map[t]:.6g}")
    print("\nPASS: DHO DE discovered from raw data using run_de.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
