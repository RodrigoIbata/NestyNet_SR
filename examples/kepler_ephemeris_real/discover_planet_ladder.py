#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Greedy multi-body point-mass ladder on Kepler acceleration residuals.

Generalizes discover_third_body_residuals.py:

* optional free-sign solar-mass correction column (delta-mu times r/|r|^3), so
  the pass can subtract the *recovered* mu_sun from the earlier rungs and
  remain fully self-contained;
* an arbitrary number of blind perturbers, scanned on a mean-motion-uniform
  grid that reaches the inner Solar System (the inner planets act on the belt
  almost purely through the heliocentric indirect term, i.e. the Sun's reflex
  acceleration);
* block-coordinate joint refinement of every body's Keplerian elements after
  each ladder stage, with all linear coefficients (GMs, delta-mu) profiled out;
* body-holdout or time-holdout validation, per-stage BIC model selection;
* a shared-residual diagnostic (cross-body correlation and periodogram of the
  body-averaged residual) recorded at every stage;
* optional known-body input (e.g. the belt-discovered planets) so the same
  ladder can be pointed at Uranus to search for a trans-Uranian body.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kepler_demo_utils import DEFAULT_SOLAR_MU_AU_DAY, _jsonable
from discover_third_body_residuals import (
    CircularPerturber,
    KeplerianPerturber,
    ObservationBlock,
    ObservationSet,
    _fit_metrics,
    _period_days,
    build_residual_observation_blocks,
    circular_source_positions,
    keplerian_from_circular,
    keplerian_source_positions,
    load_state_series_from_manifest,
    split_observation_blocks,
    stack_observations,
    third_body_template,
)

DEFAULT_BULK_RAW_MANIFEST = (
    Path(__file__).resolve().parent
    / "data"
    / "raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.json"
)

# Post-hoc reference values only; never seen by the fit.
KNOWN_PLANETS = {
    "mercury": {"mu_over_sun": 1.6601e-7, "a_au": 0.3871, "period_year": 0.2408, "e": 0.2056, "i_deg": 7.005},
    "venus": {"mu_over_sun": 2.4478e-6, "a_au": 0.7233, "period_year": 0.6152, "e": 0.0068, "i_deg": 3.395},
    "earth+moon": {"mu_over_sun": 3.0404e-6, "a_au": 1.0000, "period_year": 1.0000, "e": 0.0167, "i_deg": 0.0},
    "mars": {"mu_over_sun": 3.2271e-7, "a_au": 1.5237, "period_year": 1.8808, "e": 0.0934, "i_deg": 1.850},
    "jupiter": {"mu_over_sun": 9.5479e-4, "a_au": 5.2044, "period_year": 11.862, "e": 0.0489, "i_deg": 1.303},
    "saturn": {"mu_over_sun": 2.8588e-4, "a_au": 9.5826, "period_year": 29.457, "e": 0.0565, "i_deg": 2.485},
    "uranus": {"mu_over_sun": 4.3662e-5, "a_au": 19.191, "period_year": 84.017, "e": 0.0472, "i_deg": 0.773},
    "neptune": {"mu_over_sun": 5.1514e-5, "a_au": 30.069, "period_year": 164.79, "e": 0.0086, "i_deg": 1.770},
}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Linear solve: non-negative GMs plus optional free-sign columns
# ---------------------------------------------------------------------------

def fit_linear_coeffs(
    target: np.ndarray,
    templates: Sequence[np.ndarray],
    *,
    free_cols: frozenset[int] | set[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Least squares with non-negativity on all columns except ``free_cols``.

    Active-set scheme: free columns are never deactivated; constrained columns
    with negative solutions are dropped one at a time (most negative first).
    """
    y = np.asarray(target, dtype=np.float64)
    cols = [np.asarray(c, dtype=np.float64) for c in templates]
    if not cols:
        pred = np.zeros_like(y)
        return np.zeros(0), pred, _fit_metrics(y, pred, k_params=0)
    y_vec = y.reshape(-1)
    matrix = np.column_stack([c.reshape(-1) for c in cols])
    with np.errstate(invalid="ignore", over="ignore"):
        norms = np.linalg.norm(matrix, axis=0)
        finite_cols = np.all(np.isfinite(matrix), axis=0)
    valid = finite_cols & np.isfinite(norms) & (norms > 0.0)
    matrix = np.where(valid[None, :], matrix, 0.0)
    scaled = np.where(valid[None, :], matrix / np.where(valid, norms, 1.0)[None, :], 0.0)
    n_col = matrix.shape[1]
    active = valid.copy()
    coeffs_scaled = np.zeros(n_col, dtype=np.float64)
    for _ in range(n_col + 1):
        if not np.any(active):
            break
        sol, *_ = np.linalg.lstsq(scaled[:, active], y_vec, rcond=None)
        trial = np.zeros(n_col, dtype=np.float64)
        trial[active] = sol
        constrained = np.asarray(
            [active[j] and (j not in free_cols) for j in range(n_col)], dtype=bool
        )
        bad = constrained & (trial < -1.0e-20)
        if not np.any(bad):
            coeffs_scaled = np.where(constrained, np.maximum(trial, 0.0), trial)
            break
        worst = np.flatnonzero(bad)[int(np.argmin(trial[bad]))]
        active[worst] = False
    coeffs = np.where(valid, coeffs_scaled / np.where(valid, norms, 1.0), 0.0)
    pred = matrix @ coeffs
    pred = pred.reshape(y.shape)
    if not np.all(np.isfinite(pred)):
        pred = np.zeros_like(y)
        coeffs = np.zeros(n_col, dtype=np.float64)
    return coeffs, pred, _fit_metrics(y, pred, k_params=int(n_col))


# ---------------------------------------------------------------------------
# Model = optional delta-mu column + list of Keplerian bodies
# ---------------------------------------------------------------------------

def mu_correction_template(obs: ObservationSet) -> np.ndarray:
    r = obs.position_au
    radius = np.linalg.norm(r, axis=1)
    return r / (np.maximum(radius, 1.0e-15) ** 3)[:, None]


def body_template(obs: ObservationSet, body: KeplerianPerturber, *, mu_sun: float) -> np.ndarray:
    source = keplerian_source_positions(
        obs.t_day,
        a_au=float(body.a_au),
        eccentricity=float(body.eccentricity),
        inclination_rad=float(body.inclination_rad),
        node_rad=float(body.node_wrapped),
        arg_peri_rad=float(body.arg_peri_wrapped),
        mean_anomaly0_rad=float(body.mean_anomaly0_wrapped),
        mu_sun=float(mu_sun),
    )
    return third_body_template(obs.position_au, source)


def evaluate_model(
    obs: ObservationSet,
    bodies: Sequence[KeplerianPerturber],
    *,
    mu_sun: float,
    fit_mu_correction: bool,
    refit: bool = True,
    fixed_coeffs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return (coeffs, pred, metrics); coeffs = [delta_mu?] + GMs."""
    templates: list[np.ndarray] = []
    free: set[int] = set()
    if fit_mu_correction:
        templates.append(mu_correction_template(obs))
        free.add(0)
    templates.extend(body_template(obs, b, mu_sun=mu_sun) for b in bodies)
    k_params = (1 if fit_mu_correction else 0) + 7 * len(bodies)
    if not templates:
        pred = np.zeros_like(obs.residual_accel_au_per_d2)
        return np.zeros(0), pred, _fit_metrics(obs.residual_accel_au_per_d2, pred, k_params=0)
    if refit:
        coeffs, pred, _ = fit_linear_coeffs(
            obs.residual_accel_au_per_d2, templates, free_cols=frozenset(free)
        )
    else:
        if fixed_coeffs is None:
            raise ValueError("fixed_coeffs required when refit=False")
        coeffs = np.asarray(fixed_coeffs, dtype=np.float64)
        pred = np.zeros_like(obs.residual_accel_au_per_d2)
        for c, tmpl in zip(coeffs, templates):
            pred = pred + float(c) * tmpl
    metrics = _fit_metrics(obs.residual_accel_au_per_d2, pred, k_params=k_params)
    return coeffs, pred, metrics


# ---------------------------------------------------------------------------
# Blind circular scan, mean-motion-uniform grid
# ---------------------------------------------------------------------------

def build_scan_a_grid(
    *,
    a_min: float,
    a_max: float,
    mu_sun: float,
    baseline_day: float,
    phase_drift_rad: float = 0.5 * math.pi,
    max_count: int = 1200,
) -> np.ndarray:
    """Semi-major-axis grid uniform in mean motion.

    Adjacent grid points differ in accumulated phase over the observation
    baseline by at most ``phase_drift_rad``, so every periodic signal falls
    inside the basin of attraction of the local refiner.
    """
    n_hi = math.sqrt(mu_sun / float(a_min) ** 3)
    n_lo = math.sqrt(mu_sun / float(a_max) ** 3)
    count = int(math.ceil((n_hi - n_lo) * float(baseline_day) / float(phase_drift_rad))) + 1
    count = int(min(max(count, 8), max_count))
    n_grid = np.linspace(n_hi, n_lo, count, dtype=np.float64)
    return (float(mu_sun) / n_grid**2) ** (1.0 / 3.0)


def scan_circular_body(
    obs: ObservationSet,
    target: np.ndarray,
    *,
    a_grid: np.ndarray,
    phase_count: int,
    mu_sun: float,
    existing_a: Sequence[float] = (),
    min_rel_a_separation: float = 0.1,
    template_abs_cap: float = 1.0e6,
    top_k: int = 1,
) -> list[CircularPerturber]:
    """Return the best circular candidates, deduplicated in semi-major axis."""
    y_vec = np.asarray(target, dtype=np.float64).reshape(-1)
    yy = float(y_vec @ y_vec)
    t_day = obs.t_day
    r = obs.position_au
    phases = np.linspace(0.0, 2.0 * math.pi, int(phase_count), endpoint=False)
    candidates: list[CircularPerturber] = []
    for a_val in np.asarray(a_grid, dtype=np.float64):
        if any(abs(float(a_val) - a0) < min_rel_a_separation * a0 for a0 in existing_a):
            continue
        mean_motion = math.sqrt(float(mu_sun) / float(a_val) ** 3)
        nt = mean_motion * t_day
        cos_nt = np.cos(nt)
        sin_nt = np.sin(nt)
        for phase in phases:
            c_p, s_p = math.cos(float(phase)), math.sin(float(phase))
            source = np.column_stack(
                [
                    float(a_val) * (cos_nt * c_p - sin_nt * s_p),
                    float(a_val) * (sin_nt * c_p + cos_nt * s_p),
                    np.zeros_like(nt),
                ]
            )
            tmpl = third_body_template(r, source)
            x_vec = tmpl.reshape(-1)
            with np.errstate(invalid="ignore", over="ignore"):
                max_abs = float(np.max(np.abs(x_vec)))
            if not math.isfinite(max_abs) or max_abs > template_abs_cap:
                continue
            xx = float(x_vec @ x_vec)
            if not math.isfinite(xx) or xx <= 0.0:
                continue
            xy = float(x_vec @ y_vec)
            coeff = max(xy / xx, 0.0)
            sse = yy - 2.0 * coeff * xy + coeff * coeff * xx
            if math.isfinite(sse):
                candidates.append(
                    CircularPerturber(
                        a_au=float(a_val), phase_rad=float(phase), mu_au3_per_d2=coeff, train_sse=sse
                    )
                )
    if not candidates:
        raise ValueError("empty circular scan after exclusions")
    candidates.sort(key=lambda c: c.train_sse)
    picked: list[CircularPerturber] = []
    for cand in candidates:
        if any(abs(cand.a_au - p.a_au) < min_rel_a_separation * p.a_au for p in picked):
            continue
        picked.append(cand)
        if len(picked) >= max(int(top_k), 1):
            break
    return picked


def refine_circular_body(
    obs: ObservationSet,
    target: np.ndarray,
    initial: CircularPerturber,
    *,
    mu_sun: float,
    a_bounds: tuple[float, float],
    template_abs_cap: float = 1.0e6,
) -> CircularPerturber:
    from scipy.optimize import minimize

    y = np.asarray(target, dtype=np.float64)

    def objective(params: np.ndarray) -> float:
        a_val, phase = float(params[0]), float(params[1])
        if a_val < a_bounds[0] or a_val > a_bounds[1]:
            return float("inf")
        source = circular_source_positions(obs.t_day, a_au=a_val, phase_rad=phase, mu_sun=mu_sun)
        tmpl = third_body_template(obs.position_au, source)
        with np.errstate(invalid="ignore", over="ignore"):
            max_abs = float(np.max(np.abs(tmpl)))
        if not math.isfinite(max_abs) or max_abs > template_abs_cap:
            return float("inf")
        coeffs, pred, metrics = fit_linear_coeffs(y, [tmpl])
        return float(metrics["sse"])

    result = minimize(
        objective,
        x0=np.asarray([initial.a_au, initial.phase_wrapped], dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1.0e-7, "fatol": 1.0e-26, "adaptive": True},
    )
    if np.isfinite(float(result.fun)) and float(result.fun) <= float(initial.train_sse):
        a_val = float(np.clip(result.x[0], a_bounds[0], a_bounds[1]))
        phase = float(np.mod(result.x[1], 2.0 * math.pi))
        source = circular_source_positions(obs.t_day, a_au=a_val, phase_rad=phase, mu_sun=mu_sun)
        tmpl = third_body_template(obs.position_au, source)
        coeffs, _pred, metrics = fit_linear_coeffs(y, [tmpl])
        return CircularPerturber(
            a_au=a_val, phase_rad=phase, mu_au3_per_d2=float(coeffs[0]), train_sse=float(metrics["sse"])
        )
    return initial


# ---------------------------------------------------------------------------
# Joint block-coordinate refinement of Keplerian elements
# ---------------------------------------------------------------------------

def refine_body_in_joint_model(
    obs: ObservationSet,
    bodies: list[KeplerianPerturber],
    idx: int,
    *,
    mu_sun: float,
    fit_mu_correction: bool,
    a_bounds: tuple[float, float],
    e_max: float,
    i_max_rad: float,
    maxiter: int,
    maxfev: int,
    template_abs_cap: float = 1.0e6,
) -> KeplerianPerturber:
    """Refine the 6 elements of bodies[idx] inside the full joint model.

    All other bodies' templates (and the delta-mu column) are cached; the
    linear coefficients are re-profiled at every objective evaluation.
    """
    from scipy.optimize import minimize

    y = obs.residual_accel_au_per_d2
    fixed_templates: list[np.ndarray] = []
    free: set[int] = set()
    if fit_mu_correction:
        fixed_templates.append(mu_correction_template(obs))
        free.add(0)
    for j, b in enumerate(bodies):
        if j != idx:
            fixed_templates.append(body_template(obs, b, mu_sun=mu_sun))
    scale = max(float(np.sum(np.square(y))), 1.0e-300)
    body0 = bodies[idx]

    def objective(params: np.ndarray) -> float:
        a_val = float(params[0])
        ecc = float(params[1])
        inc = float(params[2])
        if not (a_bounds[0] <= a_val <= a_bounds[1]) or not (0.0 <= ecc <= e_max) or not (
            0.0 <= inc <= i_max_rad
        ):
            return float("inf")
        cand = KeplerianPerturber(
            a_au=a_val,
            eccentricity=ecc,
            inclination_rad=inc,
            node_rad=float(np.mod(params[3], 2.0 * math.pi)),
            arg_peri_rad=float(np.mod(params[4], 2.0 * math.pi)),
            mean_anomaly0_rad=float(np.mod(params[5], 2.0 * math.pi)),
            mu_au3_per_d2=0.0,
            train_sse=float("inf"),
        )
        tmpl = body_template(obs, cand, mu_sun=mu_sun)
        with np.errstate(invalid="ignore", over="ignore"):
            max_abs = float(np.max(np.abs(tmpl)))
        if not math.isfinite(max_abs) or max_abs > template_abs_cap:
            return float("inf")
        _c, _p, metrics = fit_linear_coeffs(y, fixed_templates + [tmpl], free_cols=frozenset(free))
        sse = float(metrics["sse"])
        return sse / scale if math.isfinite(sse) else float("inf")

    x0 = np.asarray(
        [
            body0.a_au,
            body0.eccentricity,
            body0.inclination_rad,
            body0.node_wrapped,
            body0.arg_peri_wrapped,
            body0.mean_anomaly0_wrapped,
        ],
        dtype=np.float64,
    )
    x0[0] = float(np.clip(x0[0], a_bounds[0], a_bounds[1]))
    x0[1] = float(np.clip(x0[1], 0.0, e_max))
    x0[2] = float(np.clip(x0[2], 0.0, i_max_rad))
    f0 = objective(x0)
    # NOTE: scipy's Powell with box bounds mishandles this objective (it can
    # return a point far worse than x0); we run unbounded Powell instead and
    # keep the constraints as infinite walls inside the objective.
    result = minimize(
        objective,
        x0=x0,
        method="Powell",
        options={"maxiter": int(maxiter), "maxfev": int(maxfev), "xtol": 1.0e-7, "ftol": 1.0e-10},
    )
    if np.isfinite(float(result.fun)) and float(result.fun) < f0:
        best = np.asarray(result.x, dtype=np.float64)
    else:
        best = x0
    refined = KeplerianPerturber(
        a_au=float(best[0]),
        eccentricity=float(np.clip(best[1], 0.0, e_max)),
        inclination_rad=float(np.clip(best[2], 0.0, i_max_rad)),
        node_rad=float(np.mod(best[3], 2.0 * math.pi)),
        arg_peri_rad=float(np.mod(best[4], 2.0 * math.pi)),
        mean_anomaly0_rad=float(np.mod(best[5], 2.0 * math.pi)),
        mu_au3_per_d2=float(body0.mu_au3_per_d2),
        train_sse=float("inf"),
    )
    return refined


# ---------------------------------------------------------------------------
# Time-mode holdout
# ---------------------------------------------------------------------------

def split_observation_blocks_time(
    blocks: Sequence[ObservationBlock],
    *,
    holdout_fraction: float,
) -> tuple[list[ObservationBlock], list[ObservationBlock]]:
    train_blocks: list[ObservationBlock] = []
    test_blocks: list[ObservationBlock] = []
    frac = float(np.clip(holdout_fraction, 0.0, 0.9))
    for block in blocks:
        n = int(block.t_day.size)
        cut = max(int(round(n * (1.0 - frac))), 2)
        cut = min(cut, n - 1) if frac > 0 else n
        train_blocks.append(
            ObservationBlock(
                orbit_id=block.orbit_id,
                body_name=block.body_name,
                split="train",
                t_day=block.t_day[:cut],
                position_au=block.position_au[:cut],
                residual_accel_au_per_d2=block.residual_accel_au_per_d2[:cut],
            )
        )
        test_blocks.append(
            ObservationBlock(
                orbit_id=block.orbit_id,
                body_name=block.body_name,
                split="holdout",
                t_day=block.t_day[cut:],
                position_au=block.position_au[cut:],
                residual_accel_au_per_d2=block.residual_accel_au_per_d2[cut:],
            )
        )
    return train_blocks, test_blocks


# ---------------------------------------------------------------------------
# Shared-residual diagnostic
# ---------------------------------------------------------------------------

def shared_residual_diagnostic(
    obs: ObservationSet,
    residual: np.ndarray,
    *,
    top_k: int = 6,
) -> dict[str, Any]:
    """Cross-body correlation and periodogram of the body-averaged residual.

    The heliocentric indirect term of an unmodeled planet is identical for
    every asteroid, so it survives body-averaging and shows up as a line at
    the planet's orbital period.
    """
    t = np.asarray(obs.t_day, dtype=np.float64)
    times = np.unique(t)
    if times.size < 16:
        return {"note": "too few epochs for diagnostic"}
    dt = float(np.median(np.diff(times)))
    body_ids = np.asarray(obs.body_index)
    n_bodies = int(body_ids.max()) + 1
    time_pos = {float(tv): i for i, tv in enumerate(times)}
    mat = np.full((n_bodies, times.size, 3), np.nan, dtype=np.float64)
    for row in range(t.size):
        mat[body_ids[row], time_pos[float(t[row])], :] = residual[row]
    counts = np.sum(~np.isnan(mat[:, :, 0]), axis=0)
    mean_series = np.nanmean(mat, axis=0)
    mean_series[counts == 0] = 0.0
    mean_series = np.nan_to_num(mean_series)

    # periodogram of the body-averaged residual (x, y components)
    peaks: list[dict[str, float]] = []
    spec_total = None
    for comp in range(2):
        series = mean_series[:, comp] - float(np.mean(mean_series[:, comp]))
        amp = np.abs(np.fft.rfft(series))
        spec_total = amp**2 if spec_total is None else spec_total + amp**2
    freqs = np.fft.rfftfreq(times.size, d=dt)
    spec = np.sqrt(spec_total)
    order = np.argsort(spec[1:])[::-1] + 1
    used: list[float] = []
    for idx in order:
        f = float(freqs[idx])
        if f <= 0:
            continue
        period = 1.0 / f
        if any(abs(period - p) < 0.08 * p for p in used):
            continue
        used.append(period)
        peaks.append({"period_day": period, "amplitude": float(spec[idx])})
        if len(peaks) >= top_k:
            break

    # mean pairwise cross-body correlation of the x-component residual
    full_rows = np.flatnonzero(np.all(~np.isnan(mat[:, :, 0]), axis=1))
    mean_corr = float("nan")
    if full_rows.size >= 3:
        z = mat[full_rows][:, :, 0]
        z = z - z.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(z, axis=1)
        good = norms > 0
        z = z[good] / norms[good][:, None]
        b_count = z.shape[0]
        if b_count >= 3:
            s = z.sum(axis=0)
            mean_corr = (float(s @ s) - b_count) / (b_count * (b_count - 1))
    return {
        "mean_pairwise_body_correlation_x": mean_corr,
        "n_bodies_in_diagnostic": int(n_bodies),
        "periodogram_top_peaks": peaks,
    }


# ---------------------------------------------------------------------------
# Post-hoc naming (reference values, never seen by the fit)
# ---------------------------------------------------------------------------

def summarize_body(body: KeplerianPerturber, *, mu_sun: float) -> dict[str, Any]:
    period_year = _period_days(body.a_au, mu_sun) / 365.25
    mu_ratio = float(body.mu_au3_per_d2) / float(mu_sun)
    rows = []
    for name, ref in KNOWN_PLANETS.items():
        rows.append(
            {
                "name": name,
                "a_rel_error": abs(body.a_au - ref["a_au"]) / ref["a_au"],
                "period_rel_error": abs(period_year - ref["period_year"]) / ref["period_year"],
                "mu_ratio_rel_error": abs(mu_ratio - ref["mu_over_sun"]) / ref["mu_over_sun"]
                if mu_ratio > 0
                else float("inf"),
            }
        )
    closest = min(rows, key=lambda r: r["a_rel_error"])
    return {
        "a_au": float(body.a_au),
        "period_year": float(period_year),
        "mu_au3_per_d2": float(body.mu_au3_per_d2),
        "mu_over_sun": mu_ratio,
        "eccentricity": float(body.eccentricity),
        "inclination_deg": float(math.degrees(body.inclination_rad)),
        "node_rad": float(body.node_wrapped),
        "arg_peri_rad": float(body.arg_peri_wrapped),
        "mean_anomaly0_rad": float(body.mean_anomaly0_wrapped),
        "closest_known_by_a": closest,
    }


def load_known_bodies(path: str | Path, *, stage: str | None = None) -> list[KeplerianPerturber]:
    """Load bodies from a hand-written list or from a ladder/third-body summary JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "models" in payload:
        models = list(payload["models"])
        if stage:
            chosen = next(m for m in models if m["name"] == stage)
        else:
            best_name = payload.get("best_by_test_bic")
            chosen = next(
                (m for m in models if m["name"] == best_name),
                max(models, key=lambda m: len(m.get("perturbers", []))),
            )
        rows = list(chosen["perturbers"])
    else:
        rows = list(payload)
    out: list[KeplerianPerturber] = []
    for row in rows:
        out.append(
            KeplerianPerturber(
                a_au=float(row["a_au"]),
                eccentricity=float(row.get("eccentricity", 0.0)),
                inclination_rad=float(
                    row.get("inclination_rad", math.radians(float(row.get("inclination_deg", 0.0))))
                ),
                node_rad=float(row.get("node_rad", 0.0)),
                arg_peri_rad=float(row.get("arg_peri_rad", 0.0)),
                mean_anomaly0_rad=float(row.get("mean_anomaly0_rad", 0.0)),
                mu_au3_per_d2=float(row.get("mu_au3_per_d2", 0.0)),
                train_sse=float("inf"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Ladder driver
# ---------------------------------------------------------------------------

def run_ladder(args: argparse.Namespace) -> dict[str, Any]:
    mu_sun = float(args.mu_sun)
    body_ids = None
    if args.body_id:
        body_ids = [s.strip() for s in str(args.body_id).split(",") if s.strip()]
    series = load_state_series_from_manifest(
        args.raw_manifest,
        max_bodies=None if int(args.max_bodies) <= 0 else int(args.max_bodies),
        body_ids=body_ids,
    )
    _log(f"loaded {len(series)} bodies from {args.raw_manifest}")
    blocks = build_residual_observation_blocks(
        series, mu_sun=mu_sun, stride=int(args.stride), edge_trim=int(args.edge_trim)
    )
    if str(args.holdout_mode) == "time":
        train_blocks, test_blocks = split_observation_blocks_time(
            blocks, holdout_fraction=float(args.holdout_fraction)
        )
    else:
        train_blocks, test_blocks = split_observation_blocks(
            blocks, holdout_fraction=float(args.holdout_fraction)
        )
    train_obs = stack_observations(train_blocks)
    test_obs = stack_observations(test_blocks)
    _log(
        f"train {len(train_blocks)} blocks / {train_obs.n_vectors} vectors; "
        f"test {len(test_blocks)} blocks / {test_obs.n_vectors} vectors; "
        f"holdout_mode={args.holdout_mode}"
    )

    # scan/refine observation sets: full temporal cadence, decimated bodies
    scan_stride = max(int(args.scan_body_stride), 1)
    scan_blocks = train_blocks[::scan_stride]
    scan_obs = stack_observations(scan_blocks)
    refine_stride = max(int(args.refine_body_stride), 1)
    refine_obs = stack_observations(train_blocks[::refine_stride])
    baseline_day = float(np.max(train_obs.t_day) - np.min(train_obs.t_day))
    a_grid = build_scan_a_grid(
        a_min=float(args.a_min),
        a_max=float(args.a_max),
        mu_sun=mu_sun,
        baseline_day=baseline_day,
        phase_drift_rad=float(args.scan_phase_drift_rad),
        max_count=int(args.scan_max_a_count),
    )
    _log(
        f"scan grid: {a_grid.size} semi-major axes in [{args.a_min}, {args.a_max}] AU "
        f"x {args.phase_count} phases on {len(scan_blocks)} bodies"
    )

    fit_mu = bool(args.fit_mu_correction)
    e_max = float(args.keplerian_e_max)
    i_max_rad = math.radians(float(args.keplerian_i_max_deg))
    a_bounds = (float(args.a_hard_min), float(args.a_hard_max))

    bodies: list[KeplerianPerturber] = []
    n_known = 0
    if args.known_bodies_json:
        bodies = load_known_bodies(args.known_bodies_json, stage=args.known_bodies_stage or None)
        n_known = len(bodies)
        _log(f"loaded {n_known} known bodies from {args.known_bodies_json}")

    models: list[dict[str, Any]] = []

    def record_stage(name: str, n_scanned: int) -> tuple[np.ndarray, np.ndarray]:
        coeffs, pred, train_metrics = evaluate_model(
            train_obs, bodies, mu_sun=mu_sun, fit_mu_correction=fit_mu
        )
        # write fitted GMs back into the body records
        offset = 1 if fit_mu else 0
        for j in range(len(bodies)):
            bodies[j] = replace(bodies[j], mu_au3_per_d2=float(coeffs[offset + j]), train_sse=float(train_metrics["sse"]))
        _c, _p, test_metrics = evaluate_model(
            test_obs, bodies, mu_sun=mu_sun, fit_mu_correction=fit_mu, refit=False, fixed_coeffs=coeffs
        )
        delta_mu = float(coeffs[0]) if fit_mu else 0.0
        residual = train_obs.residual_accel_au_per_d2 - pred
        diag = shared_residual_diagnostic(train_obs, residual)
        entry = {
            "name": name,
            "n_bodies": len(bodies),
            "n_scanned_bodies": n_scanned,
            "n_known_bodies": n_known,
            "train": train_metrics,
            "test": test_metrics,
            "delta_mu_au3_per_d2": delta_mu,
            "mu_corrected_au3_per_d2": mu_sun - delta_mu,
            "perturbers": [summarize_body(b, mu_sun=mu_sun) for b in bodies],
            "shared_residual_diagnostic": diag,
        }
        models.append(entry)
        _log(
            f"{name}: train_rel_rmse={train_metrics['rel_rmse']:.4e} "
            f"test_rel_rmse={test_metrics['rel_rmse']:.4e} test_bic={test_metrics['bic']:.6e} "
            f"mu_corrected={mu_sun - delta_mu:.6e}"
        )
        for j, b in enumerate(bodies):
            s = summarize_body(b, mu_sun=mu_sun)
            c = s["closest_known_by_a"]
            _log(
                f"   body{j + 1}: a={s['a_au']:.4f} AU P={s['period_year']:.4f} yr "
                f"mu/mu_sun={s['mu_over_sun']:.4e} e={s['eccentricity']:.4f} "
                f"i={s['inclination_deg']:.3f} deg | closest={c['name']} "
                f"a_err={c['a_rel_error']:.3%} mu_err={c['mu_ratio_rel_error']:.3%}"
            )
        return pred, coeffs

    new_e_max = float(args.new_body_e_max) if args.new_body_e_max is not None else e_max
    new_i_max_rad = (
        math.radians(float(args.new_body_i_max_deg))
        if args.new_body_i_max_deg is not None
        else i_max_rad
    )

    def caps_for(j: int) -> tuple[float, float]:
        if j >= n_known:
            return new_e_max, new_i_max_rad
        return e_max, i_max_rad

    def joint_refine_sweep(tag: str) -> None:
        refinable = [
            j
            for j in range(len(bodies))
            if not (bool(args.freeze_known_elements) and j < n_known)
        ]
        for sweep in range(int(args.joint_sweeps)):
            # newest bodies first: their elements are the least converged
            for j in sorted(refinable, reverse=True):
                t0 = time.time()
                e_cap, i_cap = caps_for(j)
                bodies[j] = refine_body_in_joint_model(
                    refine_obs,
                    bodies,
                    j,
                    mu_sun=mu_sun,
                    fit_mu_correction=fit_mu,
                    a_bounds=a_bounds,
                    e_max=e_cap,
                    i_max_rad=i_cap,
                    maxiter=int(args.keplerian_maxiter),
                    maxfev=int(args.keplerian_maxfev),
                    template_abs_cap=float(args.template_abs_cap),
                )
                _log(
                    f"   [{tag}] sweep {sweep + 1}: refined body {j + 1} "
                    f"(a={bodies[j].a_au:.4f}) in {time.time() - t0:.1f}s"
                )

    # stage 0: delta-mu column only (or known bodies).  Known-body elements are
    # NOT refined here by default: retuning the known-planet theory before any
    # candidate body exists lets it absorb the very signal being sought.
    if bodies and bool(args.initial_known_sweep) and not bool(args.freeze_known_elements):
        joint_refine_sweep("known")
    pred, coeffs = record_stage(f"stage0_{len(bodies)}bodies", 0)

    for k in range(len(bodies) + 1, int(args.max_perturbers) + 1):
        if fit_mu or bodies:
            _c, pred_scan, _m = evaluate_model(
                scan_obs, bodies, mu_sun=mu_sun, fit_mu_correction=fit_mu,
                refit=False, fixed_coeffs=coeffs,
            )
        else:
            pred_scan = np.zeros_like(scan_obs.residual_accel_au_per_d2)
        target_scan = scan_obs.residual_accel_au_per_d2 - pred_scan
        t0 = time.time()
        seeds = scan_circular_body(
            scan_obs,
            target_scan,
            a_grid=a_grid,
            phase_count=int(args.phase_count),
            mu_sun=mu_sun,
            existing_a=[b.a_au for b in bodies],
            min_rel_a_separation=float(args.min_rel_a_separation),
            template_abs_cap=float(args.template_abs_cap),
            top_k=int(args.scan_seed_top_k),
        )
        _log(
            f"stage {k}: circular scan -> seeds "
            + ", ".join(f"a={c.a_au:.3f}" for c in seeds)
            + f" ({time.time() - t0:.1f}s)"
        )
        target_train = train_obs.residual_accel_au_per_d2 - pred
        # try each scan seed as a Keplerian body inside the joint model,
        # keep the seed with the lowest joint train SSE
        best_body: KeplerianPerturber | None = None
        best_sse = float("inf")
        for seed in seeds:
            seed = refine_circular_body(
                train_obs, target_train, seed, mu_sun=mu_sun, a_bounds=a_bounds,
                template_abs_cap=float(args.template_abs_cap),
            )
            trial = keplerian_from_circular(
                seed,
                eccentricity=float(args.keplerian_e_init),
                inclination_rad=math.radians(float(args.keplerian_i_init_deg)),
            )
            bodies.append(trial)
            bodies[-1] = refine_body_in_joint_model(
                refine_obs,
                bodies,
                len(bodies) - 1,
                mu_sun=mu_sun,
                fit_mu_correction=fit_mu,
                a_bounds=a_bounds,
                e_max=new_e_max,
                i_max_rad=new_i_max_rad,
                maxiter=int(args.keplerian_maxiter),
                maxfev=int(args.keplerian_maxfev),
                template_abs_cap=float(args.template_abs_cap),
            )
            _c, _p, m = evaluate_model(
                train_obs, bodies, mu_sun=mu_sun, fit_mu_correction=fit_mu
            )
            _log(
                f"stage {k}: seed a={seed.a_au:.3f} -> refined a={bodies[-1].a_au:.4f}, "
                f"train_sse={m['sse']:.6e}"
            )
            if float(m["sse"]) < best_sse:
                best_sse = float(m["sse"])
                best_body = bodies[-1]
            bodies.pop()
        assert best_body is not None
        bodies.append(best_body)
        joint_refine_sweep(f"stage{k}")
        pred, coeffs = record_stage(f"stage{k}_{len(bodies)}bodies", k - n_known)

    best = min(models, key=lambda m: float(m["test"]["bic"]))
    summary = {
        "raw_manifest": str(Path(args.raw_manifest).resolve()),
        "mu_sun_au3_per_d2": mu_sun,
        "mu_sun_reference_au3_per_d2": float(DEFAULT_SOLAR_MU_AU_DAY),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "data": {
            "n_bodies_total": len(blocks),
            "n_bodies_train": len(train_blocks),
            "n_bodies_test": len(test_blocks),
            "n_vectors_train": train_obs.n_vectors,
            "n_vectors_test": test_obs.n_vectors,
        },
        "known_planets_reference_only": KNOWN_PLANETS,
        "models": models,
        "best_by_test_bic": best["name"],
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blind multi-body point-mass ladder on Kepler acceleration residuals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_manifest", type=str, default=str(DEFAULT_BULK_RAW_MANIFEST))
    parser.add_argument("--results_dir", type=str, default=str(Path("results") / "kepler_planet_ladder"))
    parser.add_argument("--summary_name", type=str, default="planet_ladder_summary.json")
    parser.add_argument("--mu_sun", type=float, default=float(DEFAULT_SOLAR_MU_AU_DAY),
                        help="mu_sun subtracted to form residuals; pass the recovered value "
                        "together with --fit_mu_correction for a self-contained pass")
    parser.add_argument("--fit_mu_correction", action="store_true",
                        help="add a free-sign delta-mu * r/|r|^3 column to every model")
    parser.add_argument("--max_bodies", type=int, default=0)
    parser.add_argument("--body_id", type=str, default="")
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--edge_trim", type=int, default=4)
    parser.add_argument("--holdout_mode", type=str, choices=("bodies", "time"), default="bodies")
    parser.add_argument("--holdout_fraction", type=float, default=0.25)
    parser.add_argument("--max_perturbers", type=int, default=6)
    parser.add_argument("--a_min", type=float, default=0.35, help="scan window lower edge")
    parser.add_argument("--a_max", type=float, default=12.0, help="scan window upper edge")
    parser.add_argument("--a_hard_min", type=float, default=0.2, help="refinement lower bound")
    parser.add_argument("--a_hard_max", type=float, default=60.0, help="refinement upper bound")
    parser.add_argument("--phase_count", type=int, default=48)
    parser.add_argument("--scan_phase_drift_rad", type=float, default=0.5 * math.pi)
    parser.add_argument("--scan_max_a_count", type=int, default=1200)
    parser.add_argument("--scan_body_stride", type=int, default=4,
                        help="use every k-th training body in the scan (full cadence kept)")
    parser.add_argument("--refine_body_stride", type=int, default=2,
                        help="use every k-th training body in joint element refinement")
    parser.add_argument("--min_rel_a_separation", type=float, default=0.1)
    parser.add_argument("--template_abs_cap", type=float, default=1.0e6)
    parser.add_argument("--joint_sweeps", type=int, default=2)
    parser.add_argument("--keplerian_e_init", type=float, default=0.05)
    parser.add_argument("--keplerian_e_max", type=float, default=0.25)
    parser.add_argument("--keplerian_i_init_deg", type=float, default=1.0)
    parser.add_argument("--keplerian_i_max_deg", type=float, default=10.0)
    parser.add_argument("--keplerian_maxiter", type=int, default=300)
    parser.add_argument("--keplerian_maxfev", type=int, default=6000)
    parser.add_argument("--known_bodies_json", type=str, default="",
                        help="JSON with known-body elements (hand list or a ladder summary)")
    parser.add_argument("--known_bodies_stage", type=str, default="",
                        help="model name inside the summary to take known bodies from")
    parser.add_argument("--freeze_known_elements", action="store_true",
                        help="do not refine known-body elements (GMs are always refit)")
    parser.add_argument("--initial_known_sweep", action="store_true",
                        help="refine known-body elements before scanning any new body "
                        "(off by default: it can absorb the sought signal)")
    parser.add_argument("--scan_seed_top_k", type=int, default=3,
                        help="number of deduplicated scan candidates tried as Keplerian seeds")
    parser.add_argument("--new_body_e_max", type=float, default=None,
                        help="eccentricity cap for newly scanned bodies (default: keplerian_e_max)")
    parser.add_argument("--new_body_i_max_deg", type=float, default=None,
                        help="inclination cap [deg] for newly scanned bodies (default: keplerian_i_max_deg)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_ladder(args)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / str(args.summary_name)
    out_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    _log(f"best by test BIC: {summary['best_by_test_bic']}")
    _log(f"summary: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
