#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Assumed-distance profile for the trans-Uranian body.

For each assumed semi-major axis a of a candidate 7th body, the remaining
five Keplerian elements are optimized inside the joint model (six known
planets with frozen elements, all GMs and the candidate's GM profiled out),
and the train / held-out-extrapolation errors are recorded.  This makes the
(mu, a) ridge explicit instead of reporting a single point on it, and shows
which combination the data actually constrain: the candidate's sky longitude.

True Neptune ephemerides are used for post-hoc validation only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from kepler_demo_utils import _jsonable
from discover_third_body_residuals import KeplerianPerturber
from discover_planet_ladder import (
    ObservationSet,
    _log,
    body_template,
    build_residual_observation_blocks,
    evaluate_model,
    fit_linear_coeffs,
    keplerian_source_positions,
    load_known_bodies,
    load_state_series_from_manifest,
    split_observation_blocks_time,
    stack_observations,
)


def candidate_from_params(a_au: float, params: np.ndarray, *, e_max: float, i_max_rad: float) -> KeplerianPerturber:
    return KeplerianPerturber(
        a_au=float(a_au),
        eccentricity=float(np.clip(params[0], 0.0, e_max)),
        inclination_rad=float(np.clip(params[1], 0.0, i_max_rad)),
        node_rad=float(np.mod(params[2], 2.0 * math.pi)),
        arg_peri_rad=float(np.mod(params[3], 2.0 * math.pi)),
        mean_anomaly0_rad=float(np.mod(params[4], 2.0 * math.pi)),
        mu_au3_per_d2=0.0,
        train_sse=float("inf"),
    )


def fit_candidate_at_a(
    train_obs: ObservationSet,
    known: list[KeplerianPerturber],
    a_au: float,
    *,
    mu_sun: float,
    e_max: float,
    i_max_rad: float,
    m0_grid: int = 48,
    maxfev: int = 4000,
) -> KeplerianPerturber:
    from scipy.optimize import minimize

    y = train_obs.residual_accel_au_per_d2
    fixed = [body_template(train_obs, b, mu_sun=mu_sun) for b in known]
    scale = max(float(np.sum(np.square(y))), 1.0e-300)

    def sse_of(params: np.ndarray) -> float:
        e, inc = float(params[0]), float(params[1])
        if not (0.0 <= e <= e_max) or not (0.0 <= inc <= i_max_rad):
            return float("inf")
        cand = candidate_from_params(a_au, params, e_max=e_max, i_max_rad=i_max_rad)
        tmpl = body_template(train_obs, cand, mu_sun=mu_sun)
        if not np.all(np.isfinite(tmpl)):
            return float("inf")
        _c, _p, m = fit_linear_coeffs(y, fixed + [tmpl])
        return float(m["sse"]) / scale

    # coarse phase scan with a circular co-planar candidate
    best_m0, best_val = 0.0, float("inf")
    for m0 in np.linspace(0.0, 2.0 * math.pi, int(m0_grid), endpoint=False):
        val = sse_of(np.asarray([0.0, 0.0, 0.0, 0.0, m0]))
        if val < best_val:
            best_val, best_m0 = val, float(m0)
    x0 = np.asarray([0.01, math.radians(1.0), 0.0, 0.0, best_m0])
    result = minimize(
        sse_of, x0=x0, method="Powell",
        options={"maxiter": 200, "maxfev": int(maxfev), "xtol": 1.0e-7, "ftol": 1.0e-10},
    )
    best = result.x if (np.isfinite(float(result.fun)) and float(result.fun) <= sse_of(x0)) else x0
    return candidate_from_params(a_au, np.asarray(best, dtype=np.float64), e_max=e_max, i_max_rad=i_max_rad)


def ecliptic_longitude_deg(pos: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(pos[:, 1], pos[:, 0])) % 360.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Assumed-distance profile for the trans-Uranian candidate")
    parser.add_argument("--raw_manifest", type=str, default="data/uranus_states_manifest.json")
    parser.add_argument("--known_bodies_json", type=str,
                        default="results/uranus_neptune_final2/planet_ladder_summary.json")
    parser.add_argument("--known_bodies_stage", type=str, default="stage8_8bodies")
    parser.add_argument("--n_known", type=int, default=6,
                        help="use only the first n bodies from the stage as knowns")
    parser.add_argument("--mu_sun", type=float, default=2.959110e-4)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--edge_trim", type=int, default=4)
    parser.add_argument("--holdout_fraction", type=float, default=0.25)
    parser.add_argument("--a_grid_min", type=float, default=16.0)
    parser.add_argument("--a_grid_max", type=float, default=50.0)
    parser.add_argument("--a_grid_count", type=int, default=35)
    parser.add_argument("--e_max", type=float, default=0.12)
    parser.add_argument("--i_max_deg", type=float, default=5.0)
    parser.add_argument("--neptune_csv", type=str, default="data/raw_planets/neptune.csv",
                        help="true Neptune states, post-hoc validation only")
    parser.add_argument("--results_dir", type=str, default="results/neptune_distance_profile")
    args = parser.parse_args(argv)

    mu_sun = float(args.mu_sun)
    series = load_state_series_from_manifest(args.raw_manifest)
    blocks = build_residual_observation_blocks(
        series, mu_sun=mu_sun, stride=int(args.stride), edge_trim=int(args.edge_trim)
    )
    train_blocks, test_blocks = split_observation_blocks_time(
        blocks, holdout_fraction=float(args.holdout_fraction)
    )
    train_obs = stack_observations(train_blocks)
    test_obs = stack_observations(test_blocks)

    known = load_known_bodies(args.known_bodies_json, stage=args.known_bodies_stage or None)
    known = known[: int(args.n_known)]
    _log(f"{len(known)} frozen known bodies; train {train_obs.n_vectors} vectors, test {test_obs.n_vectors}")

    # reference floor: knowns only
    _c0, _p0, m_train0 = evaluate_model(train_obs, known, mu_sun=mu_sun, fit_mu_correction=False)
    coeffs0, _pp, _mm = evaluate_model(train_obs, known, mu_sun=mu_sun, fit_mu_correction=False)
    _c0t, _p0t, m_test0 = evaluate_model(
        test_obs, known, mu_sun=mu_sun, fit_mu_correction=False, refit=False, fixed_coeffs=coeffs0
    )
    _log(f"floor (no candidate): train_rel_rmse={m_train0['rel_rmse']:.4e} test_rel_rmse={m_test0['rel_rmse']:.4e}")

    # true Neptune for post-hoc validation
    nep = None
    nep_path = Path(args.neptune_csv)
    if nep_path.exists():
        arr = np.genfromtxt(nep_path, delimiter=",", names=True)
        nep = {
            "t": np.asarray(arr["t_day"], dtype=np.float64),
            "pos": np.column_stack([arr["x_au"], arr["y_au"], arr["z_au"]]).astype(np.float64),
        }

    e_max = float(args.e_max)
    i_max_rad = math.radians(float(args.i_max_deg))
    a_grid = np.linspace(float(args.a_grid_min), float(args.a_grid_max), int(args.a_grid_count))
    rows: list[dict[str, Any]] = []
    probe_days = {"1980.0": 0.0, "1990.0": 3652.0, "1993.0": 4748.0, "2000.0": 7305.0, "2009.9": 10950.0}
    for a_au in a_grid:
        t0 = time.time()
        cand = fit_candidate_at_a(
            train_obs, known, float(a_au), mu_sun=mu_sun, e_max=e_max, i_max_rad=i_max_rad
        )
        model = known + [cand]
        coeffs, _pred, m_train = evaluate_model(train_obs, model, mu_sun=mu_sun, fit_mu_correction=False)
        _ct, _pt, m_test = evaluate_model(
            test_obs, model, mu_sun=mu_sun, fit_mu_correction=False, refit=False, fixed_coeffs=coeffs
        )
        mu_fit = float(coeffs[-1])
        row: dict[str, Any] = {
            "a_au": float(a_au),
            "mu_over_sun": mu_fit / mu_sun,
            "period_year": float(2.0 * math.pi * math.sqrt(a_au**3 / mu_sun) / 365.25),
            "eccentricity": float(cand.eccentricity),
            "inclination_deg": float(math.degrees(cand.inclination_rad)),
            "node_rad": float(cand.node_wrapped),
            "arg_peri_rad": float(cand.arg_peri_wrapped),
            "mean_anomaly0_rad": float(cand.mean_anomaly0_wrapped),
            "train_rel_rmse": float(m_train["rel_rmse"]),
            "test_rel_rmse": float(m_test["rel_rmse"]),
            "test_bic": float(m_test["bic"]),
        }
        if nep is not None:
            t_probe = np.asarray(list(probe_days.values()))
            pos = keplerian_source_positions(
                t_probe,
                a_au=float(a_au),
                eccentricity=float(cand.eccentricity),
                inclination_rad=float(cand.inclination_rad),
                node_rad=float(cand.node_wrapped),
                arg_peri_rad=float(cand.arg_peri_wrapped),
                mean_anomaly0_rad=float(cand.mean_anomaly0_wrapped),
                mu_sun=mu_sun,
            )
            lon_cand = ecliptic_longitude_deg(pos)
            lon_err = {}
            for (label, td), lc in zip(probe_days.items(), lon_cand):
                j = int(np.argmin(np.abs(nep["t"] - td)))
                lt = math.degrees(math.atan2(nep["pos"][j, 1], nep["pos"][j, 0])) % 360.0
                lon_err[label] = float((lc - lt + 180.0) % 360.0 - 180.0)
            row["longitude_error_deg"] = lon_err
        rows.append(row)
        _log(
            f"a={a_au:6.2f} AU: mu/musun={row['mu_over_sun']:.3e} "
            f"train={row['train_rel_rmse']:.4e} test={row['test_rel_rmse']:.4e} "
            f"lon_err(1993)={row.get('longitude_error_deg', {}).get('1993.0', float('nan')):+.2f} deg "
            f"({time.time() - t0:.1f}s)"
        )

    best_test = min(rows, key=lambda r: r["test_rel_rmse"])
    best_train = min(rows, key=lambda r: r["train_rel_rmse"])
    summary = {
        "config": vars(args),
        "floor": {"train_rel_rmse": float(m_train0["rel_rmse"]), "test_rel_rmse": float(m_test0["rel_rmse"])},
        "known_bodies": [
            {"a_au": b.a_au, "mu_au3_per_d2": b.mu_au3_per_d2} for b in known
        ],
        "profile": rows,
        "best_by_test": best_test,
        "best_by_train": best_train,
        "neptune_reference": {"a_au": 30.069, "mu_over_sun": 5.1514e-5, "period_year": 164.79},
    }
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "neptune_distance_profile.json"
    out.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    _log(f"best by held-out extrapolation: a={best_test['a_au']:.2f} AU, mu/musun={best_test['mu_over_sun']:.3e}")
    _log(f"summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
