#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_selected_numbers(manifest_path: Path) -> tuple[dict[int, str], dict[int, str]]:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    number_to_orbit = {}
    number_to_split = {}
    for row in list(rows):
        number = int(str(row["horizons_command"]).rstrip(";"))
        number_to_orbit[number] = str(row["orbit_id"])
        number_to_split[number] = str(row["split"])
    return number_to_orbit, number_to_split


def _build_frame(
    *,
    parquet_path: Path,
    candidate_summary_path: Path,
    selected_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    candidate_rows = list(candidate_summary["rows"])
    candidate_numbers = {int(row["sso_number"]) for row in candidate_rows}
    selected_orbit_by_number, selected_split_by_number = _load_selected_numbers(selected_manifest_path)

    cols = [
        "sso_number",
        "sso_name",
        "sso_class",
        "mass.value",
        "orbital_elements.eccentricity.value",
        "orbital_elements.semi_major_axis.value",
        "orbital_elements.periapsis_distance.value",
        "orbital_elements.apoapsis_distance.value",
        "moid.Mars.value",
        "moid.Jupiter.value",
        "moid.EMB.value",
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    df = df[df["sso_number"].notna()].copy()
    df["sso_number"] = df["sso_number"].astype(int)
    df = df[df["sso_number"].isin(candidate_numbers | set(selected_orbit_by_number))].copy()
    df["a"] = df["orbital_elements.semi_major_axis.value"].astype(float)
    df["e"] = df["orbital_elements.eccentricity.value"].astype(float)
    df["q"] = df["orbital_elements.periapsis_distance.value"].astype(float)
    df["Q"] = df["orbital_elements.apoapsis_distance.value"].astype(float)
    df["mass_kg"] = df["mass.value"].astype(float)
    df["radial_ratio"] = df["Q"] / df["q"]
    df["log_radial_ratio"] = np.log(df["radial_ratio"])
    df["planet_moid_min"] = df[["moid.Mars.value", "moid.Jupiter.value"]].min(axis=1).astype(float)
    df["earth_moid"] = df["moid.EMB.value"].astype(float)
    df["is_candidate_pool"] = df["sso_number"].isin(candidate_numbers)
    df["is_selected7"] = df["sso_number"].isin(selected_orbit_by_number)
    df["selected_orbit_id"] = df["sso_number"].map(selected_orbit_by_number)
    df["selected_split"] = df["sso_number"].map(selected_split_by_number)

    candidate_df = df[df["is_candidate_pool"]].copy()
    selected_df = df[df["is_selected7"]].copy()
    selected_in_pool = selected_df[selected_df["is_candidate_pool"]].copy()
    selected_missing = selected_df[~selected_df["is_candidate_pool"]].copy()

    selected_max_leverage = float(selected_in_pool["radial_ratio"].max()) if not selected_in_pool.empty else float("nan")
    selected_min_clean = float(selected_in_pool["planet_moid_min"].min()) if not selected_in_pool.empty else float("nan")

    promising = candidate_df[
        (~candidate_df["is_selected7"])
        & (candidate_df["radial_ratio"] > selected_max_leverage)
        & (candidate_df["planet_moid_min"] >= selected_min_clean)
    ].copy()

    frontier_candidates = candidate_df[~candidate_df["is_selected7"]].copy()
    frontier_candidates = frontier_candidates.sort_values(
        ["radial_ratio", "planet_moid_min"],
        ascending=[False, False],
    )
    frontier_rows = []
    best_clean = -np.inf
    for _, row in frontier_candidates.iterrows():
        clean = float(row["planet_moid_min"])
        if clean > best_clean:
            frontier_rows.append(row)
            best_clean = clean
    frontier_df = pd.DataFrame(frontier_rows)

    malformed_selected = []
    if not selected_missing.empty:
        for _, row in selected_missing.iterrows():
            malformed_selected.append(
                {
                    "sso_number": int(row["sso_number"]),
                    "sso_name": str(row["sso_name"]),
                    "mass_kg": float(row["mass_kg"]),
                    "note": "current selected body missing from mass-selected candidate pool",
                }
            )

    summary = {
        "candidate_count": int(len(candidate_df)),
        "selected7_count": int(len(selected_df)),
        "selected7_in_candidate_pool": int(len(selected_in_pool)),
        "selected7_missing_from_candidate_pool": int(len(selected_missing)),
        "selected7_missing_rows": malformed_selected,
        "selected_max_radial_ratio": selected_max_leverage,
        "selected_min_planet_moid": selected_min_clean,
        "promising_additions_count": int(len(promising)),
        "pareto_frontier_count": int(len(frontier_df)),
        "promising_additions_top20": (
            promising.sort_values(["radial_ratio", "planet_moid_min"], ascending=[False, False])
            .loc[:, ["sso_number", "sso_name", "mass_kg", "e", "radial_ratio", "planet_moid_min"]]
            .head(20)
            .to_dict(orient="records")
        ),
        "pareto_frontier_top20": (
            frontier_df.loc[:, ["sso_number", "sso_name", "mass_kg", "e", "radial_ratio", "planet_moid_min"]]
            .head(20)
            .to_dict(orient="records")
        ),
    }
    return df, summary


def _annotate(ax, df: pd.DataFrame, *, xcol: str, ycol: str, color: str, size: int = 8) -> None:
    for _, row in df.iterrows():
        ax.annotate(
            str(row["sso_name"]),
            (float(row[xcol]), float(row[ycol])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=size,
            color=color,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the new JPL∩SsODNet candidate pool against the current 7-body weathered subset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ssodnet_parquet", type=str, required=True)
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
        "--output_dir",
        type=str,
        default=str(Path("results") / "kepler_ephemeris_real_candidate_overlay"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, summary = _build_frame(
        parquet_path=Path(args.ssodnet_parquet),
        candidate_summary_path=Path(args.candidate_summary),
        selected_manifest_path=Path(args.selected_manifest),
    )

    candidate_df = df[df["is_candidate_pool"]].copy()
    selected_in_pool = df[df["is_selected7"] & df["is_candidate_pool"]].copy()
    selected_missing = df[df["is_selected7"] & ~df["is_candidate_pool"]].copy()
    promising = candidate_df[
        (~candidate_df["is_selected7"])
        & (candidate_df["radial_ratio"] > float(summary["selected_max_radial_ratio"]))
        & (candidate_df["planet_moid_min"] >= float(summary["selected_min_planet_moid"]))
    ].copy()
    frontier_numbers = {int(row["sso_number"]) for row in list(summary["pareto_frontier_top20"])}
    frontier_df = candidate_df[candidate_df["sso_number"].isin(frontier_numbers)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    ax.scatter(candidate_df["a"], candidate_df["e"], s=16, c="#c9ced6", alpha=0.55, label="307 candidates")
    ax.scatter(selected_in_pool["a"], selected_in_pool["e"], s=48, c="#1b5e20", label="Current selected in 307")
    if not selected_missing.empty:
        ax.scatter(
            selected_missing["a"],
            selected_missing["e"],
            s=64,
            facecolors="none",
            edgecolors="#c62828",
            linewidths=1.5,
            label="Current selected missing from 307",
        )
    ax.scatter(frontier_df["a"], frontier_df["e"], s=38, c="#ef6c00", marker="D", label="Pareto frontier (top shown)")
    _annotate(ax, selected_in_pool, xcol="a", ycol="e", color="#1b5e20")
    _annotate(ax, selected_missing, xcol="a", ycol="e", color="#c62828")
    ax.set_xlabel("Semi-major axis a [AU]")
    ax.set_ylabel("Eccentricity e")
    ax.set_title("Orbital Geometry")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    ax.scatter(
        candidate_df["radial_ratio"],
        candidate_df["planet_moid_min"],
        s=16,
        c="#c9ced6",
        alpha=0.55,
        label="307 candidates",
    )
    ax.scatter(
        selected_in_pool["radial_ratio"],
        selected_in_pool["planet_moid_min"],
        s=48,
        c="#1b5e20",
        label="Current selected in 307",
    )
    if not selected_missing.empty:
        ax.scatter(
            selected_missing["radial_ratio"],
            selected_missing["planet_moid_min"],
            s=64,
            facecolors="none",
            edgecolors="#c62828",
            linewidths=1.5,
            label="Current selected missing from 307",
        )
    ax.scatter(
        promising["radial_ratio"],
        promising["planet_moid_min"],
        s=34,
        c="#1565c0",
        alpha=0.85,
        label="More leverage, no dirtier than current set",
    )
    ax.axvline(float(summary["selected_max_radial_ratio"]), color="#1b5e20", linestyle="--", linewidth=1.0)
    ax.axhline(float(summary["selected_min_planet_moid"]), color="#1b5e20", linestyle="--", linewidth=1.0)
    _annotate(
        ax,
        promising.sort_values(["radial_ratio", "planet_moid_min"], ascending=[False, False]).head(12),
        xcol="radial_ratio",
        ycol="planet_moid_min",
        color="#1565c0",
    )
    _annotate(ax, selected_in_pool, xcol="radial_ratio", ycol="planet_moid_min", color="#1b5e20")
    _annotate(ax, selected_missing, xcol="radial_ratio", ycol="planet_moid_min", color="#c62828")
    ax.set_xlabel("Radial leverage Q/q")
    ax.set_ylabel("min(MOID Mars, MOID Jupiter) [AU]")
    ax.set_title("Leverage vs Perturbation Proxy")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("JPL∩SsODNet Massive Main-Belt Pool vs Current 7-Body Weathered Subset", fontsize=12)
    png_path = output_dir / "candidate_pool_vs_selected7.png"
    pdf_path = output_dir / "candidate_pool_vs_selected7.pdf"
    summary_path = output_dir / "candidate_pool_vs_selected7_summary.json"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Figure  : {png_path}")
    print(f"Figure  : {pdf_path}")
    print(f"Summary : {summary_path}")
    print(f"Promising additions count: {summary['promising_additions_count']}")
    if summary["selected7_missing_rows"]:
        print("Selected bodies missing from 307 pool:")
        for row in list(summary["selected7_missing_rows"]):
            print(f"  {row['sso_number']} {row['sso_name']} mass={row['mass_kg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
