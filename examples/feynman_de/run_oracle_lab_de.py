#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Convenience runner for factorized symbolic search/continuous skeleton refinement DE oracle experiments on Feynman CSVs.

Examples
--------
Simple starter case:
    python examples/feynman_de/run_oracle_lab_de.py

Run one explicit spec:
    python examples/feynman_de/run_oracle_lab_de.py \\
      --spec examples/feynman_de/oracle_specs/de000_simple.json \\
      --n_iter 2000

Run folder sweep:
    python examples/feynman_de/run_oracle_lab_de.py --folder_mode --only de000,de001 --fast
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from dataclasses import replace
from typing import Any

import torch

# ------------------------------------------------------------------
# Ensure local checkout precedence when running this file as a script.
# ------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from nestynet_sr.sr_search.factorized_search.oracle_lab_de import (  # noqa: E402
    DELabSpec,
    default_oracle_de_hyperparams,
    equation_de_spec_from_dict,
    load_de_equation_spec,
    run_oracle_de_equation,
    save_oracle_de_report,
)


REPO_ROOT = pathlib.Path(_project_root)
DEFAULT_SPEC = REPO_ROOT / "examples" / "feynman_de" / "oracle_specs" / "de000_simple.json"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "feynman_de"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "feynman_de_oracle_lab"


def _parse_order_candidates(raw: str) -> tuple[int, ...]:
    vals = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t == "":
            continue
        vals.append(int(t))
    out = tuple(sorted(set(v for v in vals if v in (1, 2))))
    if not out:
        raise ValueError("Expected order candidates from {1,2}")
    return out


def _resolve_csvs(data_dir: pathlib.Path, pattern: str, only: str | None) -> list[pathlib.Path]:
    csvs = sorted(data_dir.glob(pattern))
    csvs = [p for p in csvs if p.is_file()]

    if only is None or str(only).strip() == "":
        return csvs

    wanted = []
    for tok in str(only).split(","):
        t = tok.strip().lower()
        if t == "":
            continue
        if not t.startswith("de"):
            t = f"de{t}"
        wanted.append(t)
    want_set = set(wanted)

    out = [p for p in csvs if p.stem.lower() in want_set]
    missing = sorted(w for w in want_set if w not in {p.stem.lower() for p in out})
    if missing:
        print(f"Warning: missing CSVs for {missing}")
    return out


def _make_spec_from_csv(csv_path: pathlib.Path, args: argparse.Namespace) -> DELabSpec:
    order_raw = args.order_candidates if args.order_candidates is not None else "1,2"
    payload: dict[str, Any] = {
        "id": csv_path.stem,
        "csv_paths": [str(csv_path)],
        "order_candidates": list(_parse_order_candidates(order_raw)),
        "x_axis": int(args.x_axis),
        "include_x": bool(args.include_x),
        "include_u": bool(args.include_u),
        "include_du": bool(args.include_du),
        "x_col": str(args.x_col),
        "u_col": str(args.u_col),
        "split_mode": str(args.split_mode),
        "traj_metric": str(args.traj_metric),
        "deriv": {
            "method": str(args.deriv_method),
            "s": float(args.spline_s),
            "k": int(args.spline_k),
            "du_col": args.du_col,
            "d2u_col": args.d2u_col,
        },
        "validate_integrate_topk": int(args.validate_integrate_topk),
    }
    return equation_de_spec_from_dict(payload, source=str(csv_path))


def _make_hp(args: argparse.Namespace):
    hp = default_oracle_de_hyperparams()

    hp.n_iter = int(args.n_iter)
    hp.max_depth = int(args.max_depth)
    hp.poly_degree = int(args.poly_degree)
    hp.return_topk = int(args.return_topk)
    hp.n_fit = int(args.n_fit)
    hp.n_probe = int(args.n_probe)
    hp.n_seeds = int(args.n_seeds)
    hp.split_iter_across_seeds = bool(args.split_iter_across_seeds)
    hp.brute_depth = int(args.brute_depth) if args.brute_depth is not None else None
    hp.early_stop_mse = float(args.early_stop_mse)
    hp.refine_enable = bool(args.refine_enable)

    if args.refine_lbfgs_steps is not None:
        hp.refine_lbfgs_steps = int(args.refine_lbfgs_steps)
    if args.refine_num_restarts is not None:
        hp.refine_num_restarts = int(args.refine_num_restarts)
    if args.refine_max_variants is not None:
        hp.refine_max_variants = int(args.refine_max_variants)
    if args.refine_max_params is not None:
        hp.refine_max_params = int(args.refine_max_params)

    return hp


def _row_from_report(spec: DELabSpec, report: dict[str, Any], json_path: pathlib.Path) -> dict[str, Any]:
    best = report.get("best", None)
    if best is None:
        return {
            "spec_id": spec.id,
            "best_order": None,
            "best_mse": float("inf"),
            "best_expr": "",
            "mapping_kind": "",
            "json_path": str(json_path),
        }
    return {
        "spec_id": spec.id,
        "best_order": int(best.get("order", -1)),
        "best_mse": float(best.get("mse", float("inf"))),
        "best_expr": str(best.get("expr", "")),
        "mapping_kind": str(best.get("mapping_kind", "")),
        "json_path": str(json_path),
    }


def _write_summary(rows: list[dict[str, Any]], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"

    summary_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    if rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        summary_csv.write_text("", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run oracle_lab_de on one starter spec or a full Feynman-DE CSV folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec", type=str, default=str(DEFAULT_SPEC), help="Spec path for single-run mode")
    p.add_argument("--output", type=str, default=None, help="Optional single-run output JSON path")
    p.add_argument("--folder_mode", action="store_true", help="Run folder sweep instead of single spec")
    p.add_argument("--only", type=str, default=None, help="Comma-separated IDs for folder mode (e.g. de000,de001)")
    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="CSV directory for folder mode")
    p.add_argument("--pattern", type=str, default="de*.csv", help="CSV glob pattern for folder mode")
    p.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Results output directory")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    p.add_argument("--ignore_dims", action="store_true")
    p.add_argument("--quiet", action="store_true")

    # DE-table controls (used in folder mode; for single spec these can still override some fields)
    p.add_argument("--order_candidates", type=str, default=None)
    p.add_argument("--x_axis", type=int, default=0)
    p.add_argument("--x_col", type=str, default="x0")
    p.add_argument("--u_col", type=str, default="y")
    p.add_argument("--split_mode", type=str, choices=["per_traj_point", "traj_holdout"], default="per_traj_point")
    p.add_argument("--traj_metric", type=str, choices=["mean", "max"], default="mean")

    x_g = p.add_mutually_exclusive_group()
    x_g.add_argument("--include_x", dest="include_x", action="store_true")
    x_g.add_argument("--no_x", dest="include_x", action="store_false")
    p.set_defaults(include_x=True)

    u_g = p.add_mutually_exclusive_group()
    u_g.add_argument("--include_u", dest="include_u", action="store_true")
    u_g.add_argument("--no_u", dest="include_u", action="store_false")
    p.set_defaults(include_u=True)

    du_g = p.add_mutually_exclusive_group()
    du_g.add_argument("--include_du", dest="include_du", action="store_true")
    du_g.add_argument("--no_du", dest="include_du", action="store_false")
    p.set_defaults(include_du=True)
    p.add_argument("--deriv_method", type=str, choices=["spline", "finite_diff", "precomputed"], default="spline")
    p.add_argument("--spline_s", type=float, default=0.0)
    p.add_argument("--spline_k", type=int, default=3)
    p.add_argument("--du_col", type=str, default=None)
    p.add_argument("--d2u_col", type=str, default=None)
    p.add_argument("--validate_integrate_topk", type=int, default=1)

    # factorized symbolic search controls
    p.add_argument("--n_iter", type=int, default=1500)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--poly_degree", type=int, default=4)
    p.add_argument("--return_topk", type=int, default=5)
    p.add_argument("--n_fit", type=int, default=1500)
    p.add_argument("--n_probe", type=int, default=2000)
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--split_iter_across_seeds", action="store_true", default=True)
    p.add_argument("--brute_depth", type=int, default=2)
    p.add_argument("--early_stop_mse", type=float, default=1.0e-8)
    p.add_argument("--refine_enable", action="store_true", default=False)
    p.add_argument("--refine_lbfgs_steps", type=int, default=None)
    p.add_argument("--refine_num_restarts", type=int, default=None)
    p.add_argument("--refine_max_variants", type=int, default=None)
    p.add_argument("--refine_max_params", type=int, default=None)

    p.add_argument("--fast", action="store_true", help="Small quick-test budget")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fast:
        args.n_iter = min(int(args.n_iter), 500)
        args.n_fit = min(int(args.n_fit), 800)
        args.n_probe = min(int(args.n_probe), 1200)
        args.return_topk = min(int(args.return_topk), 3)
    if args.only and not args.folder_mode:
        print("--only requires --folder_mode.")
        return 2

    hp = _make_hp(args)
    dtype = torch.float64 if str(args.dtype).lower() == "float64" else torch.float32
    out_dir = pathlib.Path(args.results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    if not args.folder_mode:
        spec = load_de_equation_spec(args.spec)
        # Allow order override for quick ablations from this wrapper.
        if args.order_candidates is not None:
            spec = replace(spec, order_candidates=_parse_order_candidates(args.order_candidates))
        report = run_oracle_de_equation(
            spec,
            factorized_search_hp=hp,
            seed=int(args.seed),
            dtype=dtype,
            enforce_dims=not bool(args.ignore_dims),
            verbose=not bool(args.quiet),
        )
        out_json = pathlib.Path(args.output).resolve() if args.output else (out_dir / f"{spec.id}_oracle_de.json")
        save_oracle_de_report(report, out_json)
        row = _row_from_report(spec, report, out_json)
        rows.append(row)
        print(
            f"[oracle-lab-de] {spec.id}: best_mse={row['best_mse']:.6g} "
            f"order={row['best_order']} expr={row['best_expr']}"
        )
    else:
        data_dir = pathlib.Path(args.data_dir).resolve()
        csvs = _resolve_csvs(data_dir, args.pattern, args.only)
        if not csvs:
            print("No CSV files matched. Check --data_dir/--pattern/--only.")
            return 1

        for i, csv_path in enumerate(csvs):
            spec = _make_spec_from_csv(csv_path, args)
            run_seed = int(args.seed) + i * 1_000_003
            report = run_oracle_de_equation(
                spec,
                factorized_search_hp=hp,
                seed=run_seed,
                dtype=dtype,
                enforce_dims=not bool(args.ignore_dims),
                verbose=not bool(args.quiet),
            )
            out_json = out_dir / f"{spec.id}_oracle_de.json"
            save_oracle_de_report(report, out_json)
            row = _row_from_report(spec, report, out_json)
            rows.append(row)
            print(
                f"[oracle-lab-de] {spec.id}: best_mse={row['best_mse']:.6g} "
                f"order={row['best_order']} expr={row['best_expr']}"
            )

    _write_summary(rows, out_dir)
    print(f"[oracle-lab-de] wrote {len(rows)} report(s) to {out_dir}")
    print(f"[oracle-lab-de] summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
