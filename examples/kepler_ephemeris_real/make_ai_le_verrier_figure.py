#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Paper figures for the AI Le Verrier analysis.

Figure 1 (ai_le_verrier_ladder): the blind planet ladder on the 308-asteroid
belt ensemble — held-out residual per stage, recovered-parameter accuracy, and
the spectral peeling of the body-averaged (solar-reflex) residual.

Figure 2 (ai_le_verrier_neptune): the trans-Uranian discovery — geometry,
assumed-distance profile (the mu-a ridge), and sky-longitude error against the
true Neptune (validation only).

All content is data-driven from the ladder / profile result JSONs.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import discover_planet_ladder as dpl

# Okabe-Ito, fixed categorical order (validated CVD-safe)
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERMILLION = "#D55E00"
INK = "#1f2937"
MUTED = "#6b7280"

MU_SUN_BELT = 2.959110e-4

PLANET_LINES_DAY = {
    "Mercury": 87.97,
    "Venus": 224.70,
    "Earth": 365.25,
    "Mars": 686.98,
}


def base_results(root: Path) -> Path:
    return root / "NestyNet_SR" / "examples" / "kepler_ephemeris_real" / "results"


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "NestyNet_SR").is_dir() and (parent / "NestyNet_papers").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


def _setup_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
        }
    )


def _bodies_from_stage(stage: dict, mu_sun: float) -> list[dpl.KeplerianPerturber]:
    out = []
    for p in stage["perturbers"]:
        out.append(
            dpl.KeplerianPerturber(
                a_au=float(p["a_au"]),
                eccentricity=float(p["eccentricity"]),
                inclination_rad=math.radians(float(p["inclination_deg"])),
                node_rad=float(p["node_rad"]),
                arg_peri_rad=float(p["arg_peri_rad"]),
                mean_anomaly0_rad=float(p["mean_anomaly0_rad"]),
                mu_au3_per_d2=float(p["mu_au3_per_d2"]),
                train_sse=0.0,
            )
        )
    return out


def _body_mean_residual_spectrum(
    obs: dpl.ObservationSet, residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum of the body-averaged residual (x,y quadrature)."""
    t = np.asarray(obs.t_day)
    times = np.unique(t)
    dt = float(np.median(np.diff(times)))
    pos = {float(tv): i for i, tv in enumerate(times)}
    body_ids = np.asarray(obs.body_index)
    n_bodies = int(body_ids.max()) + 1
    mat = np.full((n_bodies, times.size, 2), np.nan)
    for row in range(t.size):
        mat[body_ids[row], pos[float(t[row])], :] = residual[row, :2]
    mean_series = np.nan_to_num(np.nanmean(mat, axis=0))
    spec = None
    for comp in range(2):
        series = mean_series[:, comp] - mean_series[:, comp].mean()
        amp = np.abs(np.fft.rfft(series)) / times.size
        spec = amp**2 if spec is None else spec + amp**2
    freqs = np.fft.rfftfreq(times.size, d=dt)
    return freqs[1:], np.sqrt(spec[1:])


def figure_ladder(root: Path) -> None:
    summary = json.loads(
        (base_results(root) / "kepler_planet_ladder_full" / "planet_ladder_summary.json")
        .read_text(encoding="utf-8")
    )
    models = summary["models"][:7]  # delta-mu stage + six planets
    stage_labels = [
        "Sun+$\\delta\\mu_\\odot$",
        "+Jupiter",
        "+Venus",
        "+Earth",
        "+Saturn",
        "+Mercury",
        "+Mars",
    ]
    rel = np.asarray([float(m["test"]["rel_rmse"]) for m in models])

    fig, axes = plt.subplots(
        1, 3, figsize=(7.3, 2.65), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 0.95, 1.15]},
    )
    ax_lad, ax_err, ax_spec = axes

    # (a) ladder of held-out residuals
    x = np.arange(len(models))
    ax_lad.bar(x, rel, color=C_BLUE, edgecolor=INK, linewidth=0.5, width=0.72)
    ax_lad.set_yscale("log")
    ax_lad.set_xticks(x)
    ax_lad.set_xticklabels(stage_labels, fontsize=6.0, rotation=38, ha="right")
    ax_lad.set_ylabel("held-out residual RMSE / Sun-only")
    ax_lad.set_ylim(3e-3, 1.3)
    for xi, yi in zip(x, rel):
        ax_lad.text(xi, yi * 1.15, f"{yi:.3f}" if yi > 0.01 else f"{yi:.4f}",
                    ha="center", va="bottom", fontsize=5.8, color=INK)
    ax_lad.grid(axis="y", alpha=0.15, which="major")
    ax_lad.set_title("(a) Blind point-mass ladder, 308 asteroids", fontsize=8)

    # (b) recovered-parameter accuracy at the six-planet stage
    stage6 = models[6]
    names = []
    err_a, err_p, err_mu = [], [], []
    for p in stage6["perturbers"]:
        c = p["closest_known_by_a"]
        names.append(c["name"].replace("earth+moon", "Earth+Moon").capitalize()
                     if c["name"] != "earth+moon" else "Earth+Moon")
        err_a.append(100 * float(c["a_rel_error"]))
        err_p.append(100 * float(c["period_rel_error"]))
        err_mu.append(100 * float(c["mu_ratio_rel_error"]))
    xb = np.arange(len(names))
    ax_err.scatter(xb - 0.18, err_a, s=16, color=C_BLUE, marker="o", label="$a$", zorder=3)
    ax_err.scatter(xb, err_p, s=16, color=C_ORANGE, marker="s", label="$P$", zorder=3)
    ax_err.scatter(xb + 0.18, err_mu, s=18, color=C_GREEN, marker="^", label="$GM$", zorder=3)
    ax_err.set_yscale("log")
    ax_err.set_xticks(xb)
    ax_err.set_xticklabels([n[:2] if False else n for n in names], fontsize=6.0, rotation=28, ha="right")
    ax_err.set_ylabel("relative error [%]")
    ax_err.set_ylim(5e-4, 30)
    ax_err.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=":")
    ax_err.axhline(0.1, color=MUTED, linewidth=0.6, linestyle=":")
    ax_err.grid(axis="y", alpha=0.15, which="major")
    ax_err.legend(frameon=False, loc="upper left", ncol=3, columnspacing=0.9, handletextpad=0.2)
    ax_err.set_title("(b) Recovered vs reference", fontsize=8)

    # (c) spectral peeling of the body-averaged residual
    series = dpl.load_state_series_from_manifest(str(dpl.DEFAULT_BULK_RAW_MANIFEST))
    blocks = dpl.build_residual_observation_blocks(series, mu_sun=MU_SUN_BELT, stride=15, edge_trim=4)
    train_blocks, _ = dpl.split_observation_blocks(blocks, holdout_fraction=0.25)
    obs = dpl.stack_observations(train_blocks)
    y = obs.residual_accel_au_per_d2
    mu_col = dpl.mu_correction_template(obs)
    ramp = plt.cm.Blues(np.linspace(0.35, 0.95, 6))
    for si, stage in enumerate(models[1:7]):
        bodies = _bodies_from_stage(stage, MU_SUN_BELT)
        pred = float(stage["delta_mu_au3_per_d2"]) * mu_col
        for b in bodies:
            pred = pred + b.mu_au3_per_d2 * dpl.body_template(obs, b, mu_sun=MU_SUN_BELT)
        freqs, spec = _body_mean_residual_spectrum(obs, y - pred)
        periods = 1.0 / freqs
        keep = (periods > 25) & (periods < 3000)
        label = stage_labels[si + 1].replace("+", "after ")
        ax_spec.plot(periods[keep], spec[keep], color=ramp[si], linewidth=0.9, label=label)
    for name, pday in PLANET_LINES_DAY.items():
        ax_spec.axvline(pday, color=C_VERMILLION, linewidth=0.6, linestyle="--", alpha=0.6)
        ax_spec.text(pday, 2.6e-10, name, rotation=90, fontsize=5.5, ha="right",
                     va="top", color=C_VERMILLION)
    ax_spec.set_xscale("log")
    ax_spec.set_yscale("log")
    ax_spec.set_xlabel("period [day]")
    ax_spec.set_ylabel("body-averaged residual amplitude [AU d$^{-2}$]")
    ax_spec.grid(alpha=0.12, which="major")
    ax_spec.legend(frameon=False, loc="lower right", fontsize=5.4, handlelength=1.4)
    ax_spec.set_title("(c) Solar-reflex residual spectrum", fontsize=8)

    out_pdf = root / "NestyNet_papers" / "figures_paper3" / "ai_le_verrier_ladder.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(out_pdf)


def _candidate_positions(row: dict, t_day: np.ndarray, mu_sun: float) -> np.ndarray:
    return dpl.keplerian_source_positions(
        t_day,
        a_au=float(row["a_au"]),
        eccentricity=float(row["eccentricity"]),
        inclination_rad=math.radians(float(row.get("inclination_deg", 0.0))),
        node_rad=float(row.get("node_rad", 0.0)),
        arg_peri_rad=float(row.get("arg_peri_rad", 0.0)),
        mean_anomaly0_rad=float(row.get("mean_anomaly0_rad", 0.0)),
        mu_sun=mu_sun,
    )


def _lon_deg(pos: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(pos[:, 1], pos[:, 0])) % 360.0


def figure_neptune(root: Path) -> None:
    base = root / "NestyNet_SR"
    prof = json.loads(
        (base_results(root) / "neptune_distance_profile" / "neptune_distance_profile.json")
        .read_text(encoding="utf-8")
    )
    rows = prof["profile"]
    floor_test = float(prof["floor"]["test_rel_rmse"])

    example_dir = base / "examples" / "kepler_ephemeris_real"
    ura_series = dpl.load_state_series_from_manifest(
        example_dir / "data" / "uranus_states_manifest.json"
    )[0]
    ura = {"x_au": ura_series.position_au[:, 0], "y_au": ura_series.position_au[:, 1]}
    nep = np.genfromtxt(example_dir / "data" / "raw_planets" / "neptune.csv",
                        delimiter=",", names=True)

    fig, axes = plt.subplots(
        1, 3, figsize=(7.3, 2.65), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
    )
    ax_geo, ax_prof, ax_lon = axes

    # rows used across panels
    def row_at(a_val: float) -> dict:
        return min(rows, key=lambda r: abs(float(r["a_au"]) - a_val))

    r30 = row_at(30.0)
    r25 = row_at(25.0)
    r36 = row_at(36.0)

    # (a) geometry in the ecliptic plane
    ax_geo.plot(ura["x_au"], ura["y_au"], color=C_BLUE, linewidth=1.6)
    ax_geo.text(-24.0, -13.0, "Uranus\n1980--2010", color=C_BLUE, fontsize=6.2, ha="center")
    ax_geo.plot(nep["x_au"], nep["y_au"], color=INK, linewidth=1.3, linestyle="--", zorder=6)
    ax_geo.annotate("Neptune 1980--2010\n(truth, never fit)", xy=(-4.9, -29.9),
                    xytext=(-30.0, -33.0), fontsize=6.2, color=INK, ha="center",
                    arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.5})
    period_day = float(2 * math.pi * math.sqrt(30.0**3 / MU_SUN_BELT))
    t_orb = np.linspace(0.0, period_day, 720)
    orb30 = _candidate_positions(r30, t_orb, MU_SUN_BELT)
    ax_geo.plot(orb30[:, 0], orb30[:, 1], color=C_VERMILLION, linewidth=1.3)
    ax_geo.text(0.0, 32.5, "recovered body ($a{=}30$)", color=C_VERMILLION, fontsize=6.2, ha="center")
    # window arcs of the recovered body
    t_win = np.linspace(0.0, 10957.0, 200)
    win30 = _candidate_positions(r30, t_win, MU_SUN_BELT)
    ax_geo.plot(win30[:, 0], win30[:, 1], color=C_VERMILLION, linewidth=3.0, alpha=0.35)
    ax_geo.scatter([0], [0], s=30, color="#f2b705", edgecolor="#6b4e00", linewidth=0.5, zorder=5)
    j = int(np.argmin(np.abs(nep["t_day"] - 4748.0)))
    ax_geo.annotate("1993\nconjunction", xy=(nep["x_au"][j], nep["y_au"][j]),
                    xytext=(30.0, -18.0), fontsize=5.6, color=INK, ha="center",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})
    ax_geo.set_aspect("equal", adjustable="box")
    ax_geo.set_xlim(-39, 39)
    ax_geo.set_ylim(-39, 39)
    ax_geo.set_xlabel("AU")
    ax_geo.set_ylabel("AU")
    ax_geo.grid(alpha=0.12)
    ax_geo.set_title("(a) Trans-Uranian candidate", fontsize=8)

    # (b) assumed-distance profile: held-out error (top curve) + fitted mass
    a_vals = np.asarray([float(r["a_au"]) for r in rows])
    test = np.asarray([float(r["test_rel_rmse"]) for r in rows])
    mus = np.asarray([float(r["mu_over_sun"]) for r in rows])
    good = a_vals >= 21.0
    ax_prof.plot(a_vals[good], test[good] / floor_test, color=C_BLUE, linewidth=1.4,
                 marker="o", markersize=2.4)
    ax_prof.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="--")
    ax_prof.text(44.5, 1.005, "no candidate", color=MUTED, fontsize=5.8, va="bottom", ha="center")
    ax_prof.axvline(30.07, color=C_VERMILLION, linewidth=0.8, linestyle=":")
    ax_prof.text(30.07, 0.735, "Neptune", color=C_VERMILLION, fontsize=6.0, rotation=90,
                 va="bottom", ha="right")
    ax_prof.axvline(36.15, color=MUTED, linewidth=0.8, linestyle=":")
    ax_prof.text(36.15, 0.735, "Le Verrier 1846", color=MUTED, fontsize=6.0, rotation=90,
                 va="bottom", ha="right")
    ax_prof.set_xlabel("assumed semi-major axis $a$ [AU]")
    ax_prof.set_ylabel("held-out residual / no-candidate floor")
    ax_prof.set_ylim(0.72, 1.02)
    ax_prof.grid(alpha=0.15)
    ax_prof.set_title("(b) Assumed-distance profile", fontsize=8)

    # (c) fitted mass along the ridge + sky-longitude error vs epoch (inset-style split)
    ax_mu = ax_lon
    ax_mu.plot(a_vals[good], mus[good], color=C_GREEN, linewidth=1.4, marker="o", markersize=2.4)
    ax_mu.scatter([30.07], [5.1514e-5], s=55, marker="*", color=C_VERMILLION, zorder=5)
    ax_mu.annotate("true Neptune", xy=(30.07, 5.1514e-5), xytext=(-6, 10),
                   textcoords="offset points", fontsize=6.0, color=C_VERMILLION, ha="right")
    ax_mu.set_yscale("log")
    ax_mu.set_xlabel("assumed semi-major axis $a$ [AU]")
    ax_mu.set_ylabel("fitted $GM/GM_\\odot$")
    ax_mu.grid(alpha=0.15, which="major")
    ax_mu.set_title("(c) Mass--distance ridge", fontsize=8)

    out_pdf = root / "NestyNet_papers" / "figures_paper3" / "ai_le_verrier_neptune.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(out_pdf)

    # companion panel: sky-longitude error vs epoch for ridge members
    fig2, ax = plt.subplots(figsize=(3.5, 2.3), constrained_layout=True)
    t_grid = np.asarray(nep["t_day"], dtype=np.float64)
    lon_nep = _lon_deg(np.column_stack([nep["x_au"], nep["y_au"], nep["z_au"]]))
    for row, color, lw, label in [
        (r25, "#9ecae1", 1.0, "$a{=}25$"),
        (r30, C_VERMILLION, 1.5, "$a{=}30$"),
        (r36, "#4292c6", 1.0, "$a{=}36$"),
    ]:
        lon_c = _lon_deg(_candidate_positions(row, t_grid, MU_SUN_BELT))
        err = (lon_c - lon_nep + 180.0) % 360.0 - 180.0
        ax.plot(1980.0 + t_grid / 365.25, err, color=color, linewidth=lw, label=label)
    ax.axhline(0.0, color=MUTED, linewidth=0.7)
    ax.axvline(1993.1, color=MUTED, linewidth=0.6, linestyle=":")
    ax.text(1993.1, 3.6, "conjunction", fontsize=5.8, color=MUTED, rotation=90, va="top", ha="right")
    ax.set_xlabel("epoch [yr]")
    ax.set_ylabel("candidate $-$ Neptune\necliptic longitude [deg]")
    ax.set_ylim(-5, 5)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, loc="lower left", ncol=3, columnspacing=1.0)
    out2 = root / "NestyNet_papers" / "figures_paper3" / "ai_le_verrier_longitude.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    fig2.savefig(out2.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig2)
    print(out2)


def main() -> int:
    root = _workspace_root()
    _setup_rcparams()
    figure_ladder(root)
    figure_neptune(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
