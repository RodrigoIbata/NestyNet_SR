#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kepler_demo_utils import (
    _jsonable,
    build_default_kepler_datasets,
    evaluate_radial_family_with_fixed_mu,
    fit_areal_law,
)


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(arr))))


def _find_best_exponent(dataset, exponents: np.ndarray) -> tuple[float, float, float]:
    rows = []
    for exponent in list(exponents):
        fit = evaluate_radial_family_with_fixed_mu([dataset], mu=float(dataset.mu), exponent=float(exponent))
        rows.append((float(exponent), float(fit["mean_rmse"])))
    best_exponent, best_rmse = min(rows, key=lambda item: (item[1], abs(item[0] - 2.0)))
    exact_rmse = min(rows, key=lambda item: abs(item[0] - 2.0))[1]
    return float(best_exponent), float(best_rmse), float(exact_rmse)


def _energy_cleanliness(dataset, ell_fit: float) -> tuple[float, float]:
    energy_series = (
        0.5 * np.square(dataset.rdot)
        + 0.5 * ((float(ell_fit) * float(ell_fit)) / np.square(dataset.r))
        - float(dataset.mu) / dataset.r
    )
    centered = energy_series - float(np.mean(energy_series))
    scale = max(_rms(energy_series), 1.0e-15)
    return float(np.max(np.abs(centered))), float(np.max(np.abs(centered)) / scale)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a candidate raw-manifest pool by per-object weathered Kepler residuals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_manifest", type=str, required=True)
    parser.add_argument(
        "--candidate_summary",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "selection_jpl_ssodnet_mass_gt_1e17_arc15000_summary.json"),
    )
    parser.add_argument(
        "--selected_manifest",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "raw_states_manifest_main_belt_pallas_holdout.json"),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path("results") / "kepler_ephemeris_real_candidate_residuals"),
    )
    parser.add_argument("--power_exp_min", type=float, default=1.6)
    parser.add_argument("--power_exp_max", type=float, default=2.4)
    parser.add_argument("--power_exp_count", type=int, default=81)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    candidate_summary = json.loads(Path(args.candidate_summary).read_text(encoding="utf-8"))
    selected_rows = json.loads(Path(args.selected_manifest).read_text(encoding="utf-8"))
    selected_numbers = {int(str(row["horizons_command"]).rstrip(";")) for row in selected_rows}
    candidate_by_number = {int(row["sso_number"]): row for row in list(candidate_summary["rows"])}

    datasets = build_default_kepler_datasets(
        provider="raw_csv",
        profile="weathered",
        raw_manifest=args.raw_manifest,
    )
    exponents = np.linspace(
        float(args.power_exp_min),
        float(args.power_exp_max),
        int(args.power_exp_count),
        dtype=np.float64,
    )

    per_object = []
    for dataset in datasets:
        number = None
        orbit_id = str(dataset.orbit_id)
        if orbit_id.startswith("mp_"):
            try:
                number = int(orbit_id.split("_", 2)[1])
            except Exception:
                number = None
        meta = None if number is None else candidate_by_number.get(int(number), None)

        areal = fit_areal_law([dataset])
        ell_fit = float(areal["per_dataset"][0]["ell_fit"])
        omega_rmse = float(areal["per_dataset"][0]["rmse"])
        omega_rel_rmse = float(omega_rmse / max(_rms(dataset.omega), 1.0e-15))

        radial = evaluate_radial_family_with_fixed_mu([dataset], mu=float(dataset.mu), exponent=2.0)
        radial_row = radial["per_dataset"][0]
        radial_rmse = float(radial_row["rmse"])
        radial_rel_rmse = float(radial_rmse / max(_rms(dataset.rddot), 1.0e-15))

        best_exp, best_rmse, exact_rmse = _find_best_exponent(dataset, exponents)
        exp_margin = float(exact_rmse - best_rmse)
        energy_abs_resid, energy_rel_resid = _energy_cleanliness(dataset, ell_fit)
        ptheta_series = dataset.x * dataset.vy - dataset.y * dataset.vx
        ptheta_rel_std = float(np.std(ptheta_series) / max(abs(np.mean(ptheta_series)), 1.0e-15))

        cleanliness_score = float(
            np.sqrt(
                omega_rel_rmse ** 2
                + radial_rel_rmse ** 2
                + energy_rel_resid ** 2
            )
        )

        row = {
            "orbit_id": orbit_id,
            "sso_number": None if number is None else int(number),
            "body_name": str(meta["body_name"]) if isinstance(meta, dict) else orbit_id,
            "mass_kg": None if not isinstance(meta, dict) else float(meta["mass_kg"]),
            "semi_major_axis_au": None if not isinstance(meta, dict) else float(meta["semi_major_axis_au"]),
            "periapsis_distance_au": None if not isinstance(meta, dict) else float(meta["periapsis_distance_au"]),
            "apoapsis_distance_au": None if not isinstance(meta, dict) else float(meta["apoapsis_distance_au"]),
            "selection_source": None if not isinstance(meta, dict) else str(meta["selection_source"]),
            "is_current_selected7": bool(number in selected_numbers if number is not None else False),
            "eccentricity": float(dataset.e),
            "dynamic_range": float(dataset.dynamic_range),
            "omega_rmse": omega_rmse,
            "omega_rel_rmse": omega_rel_rmse,
            "radial_rmse": radial_rmse,
            "radial_rel_rmse": radial_rel_rmse,
            "best_exponent": best_exp,
            "best_exponent_abs_error": float(abs(best_exp - 2.0)),
            "exact_exponent_rmse": exact_rmse,
            "best_exponent_rmse": best_rmse,
            "exponent_margin": exp_margin,
            "energy_abs_residual": energy_abs_resid,
            "energy_rel_residual": energy_rel_resid,
            "ptheta_rel_std": ptheta_rel_std,
            "cleanliness_score": cleanliness_score,
        }
        per_object.append(row)

    per_object.sort(key=lambda row: (row["cleanliness_score"], row["radial_rel_rmse"], row["body_name"]))

    score_summary = {
        "raw_manifest": str(Path(args.raw_manifest).resolve()),
        "n_objects": int(len(per_object)),
        "screen_cadence_days": None,
        "cleanest_top20": per_object[:20],
        "noisiest_top20": list(reversed(per_object[-20:])),
        "current_selected7": [row for row in per_object if bool(row["is_current_selected7"])],
        "all_rows": per_object,
    }
    summary_path = results_dir / "candidate_kepler_residual_summary.json"
    summary_path.write_text(json.dumps(_jsonable(score_summary), indent=2), encoding="utf-8")

    radial = np.asarray([float(row["radial_rel_rmse"]) for row in per_object], dtype=np.float64)
    leverage = np.asarray([float(row["dynamic_range"]) for row in per_object], dtype=np.float64)
    is_sel = np.asarray([bool(row["is_current_selected7"]) for row in per_object], dtype=bool)

    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    ax.scatter(leverage[~is_sel], radial[~is_sel], s=18, c="#c9ced6", alpha=0.65, label="Other candidates")
    ax.scatter(leverage[is_sel], radial[is_sel], s=52, c="#1b5e20", label="Current selected 7")
    for row in per_object:
        if bool(row["is_current_selected7"]):
            ax.annotate(
                str(row["body_name"]),
                (float(row["dynamic_range"]), float(row["radial_rel_rmse"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#1b5e20",
            )
    ax.set_xlabel("Radial leverage Q/q")
    ax.set_ylabel("Relative radial Kepler residual")
    ax.set_title("Per-object Weathered Kepler Residual")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    fig_path_png = results_dir / "candidate_kepler_residuals_vs_leverage.png"
    fig_path_pdf = results_dir / "candidate_kepler_residuals_vs_leverage.pdf"
    fig.savefig(fig_path_png, dpi=180)
    fig.savefig(fig_path_pdf)
    plt.close(fig)

    print(f"Summary : {summary_path}")
    print(f"Figure  : {fig_path_png}")
    print(f"Figure  : {fig_path_pdf}")
    print("Cleanest top 10:")
    for row in per_object[:10]:
        print(
            f"  {row['body_name']}: score={row['cleanliness_score']:.3e} "
            f"radial_rel={row['radial_rel_rmse']:.3e} "
            f"best_p={row['best_exponent']:.3f}"
        )
    print("Noisiest top 10:")
    for row in list(reversed(per_object[-10:])):
        print(
            f"  {row['body_name']}: score={row['cleanliness_score']:.3e} "
            f"radial_rel={row['radial_rel_rmse']:.3e} "
            f"best_p={row['best_exponent']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
