#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kepler_demo_utils import DEFAULT_SOLAR_MU_AU_DAY, _jsonable


DEFAULT_BULK_RAW_MANIFEST = (
    Path(__file__).resolve().parent
    / "data"
    / "raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.json"
)
DEFAULT_CURATED_RAW_MANIFEST = Path(__file__).resolve().parent / "data" / "raw_states_manifest.json"

KNOWN_OUTER_PLANETS = {
    "jupiter": {
        "mu_over_sun": 9.5479e-4,
        "a_au": 5.2044,
        "period_year": 11.862,
    },
    "saturn": {
        "mu_over_sun": 2.8588e-4,
        "a_au": 9.5826,
        "period_year": 29.457,
    },
}


@dataclass(frozen=True)
class StateSeries:
    orbit_id: str
    body_name: str
    split: str
    t_day: np.ndarray
    position_au: np.ndarray
    velocity_au_per_d: np.ndarray


@dataclass(frozen=True)
class ObservationBlock:
    orbit_id: str
    body_name: str
    split: str
    t_day: np.ndarray
    position_au: np.ndarray
    residual_accel_au_per_d2: np.ndarray


@dataclass(frozen=True)
class ObservationSet:
    t_day: np.ndarray
    position_au: np.ndarray
    residual_accel_au_per_d2: np.ndarray
    body_index: np.ndarray
    body_names: tuple[str, ...]

    @property
    def n_vectors(self) -> int:
        return int(self.t_day.size)

    @property
    def n_scalars(self) -> int:
        return int(3 * self.t_day.size)


@dataclass(frozen=True)
class CircularPerturber:
    a_au: float
    phase_rad: float
    mu_au3_per_d2: float
    train_sse: float

    @property
    def phase_wrapped(self) -> float:
        return float(np.mod(float(self.phase_rad), 2.0 * math.pi))


@dataclass(frozen=True)
class KeplerianPerturber:
    a_au: float
    eccentricity: float
    inclination_rad: float
    node_rad: float
    arg_peri_rad: float
    mean_anomaly0_rad: float
    mu_au3_per_d2: float
    train_sse: float

    @property
    def node_wrapped(self) -> float:
        return float(np.mod(float(self.node_rad), 2.0 * math.pi))

    @property
    def arg_peri_wrapped(self) -> float:
        return float(np.mod(float(self.arg_peri_rad), 2.0 * math.pi))

    @property
    def mean_anomaly0_wrapped(self) -> float:
        return float(np.mod(float(self.mean_anomaly0_rad), 2.0 * math.pi))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    repo_root = _repo_root()
    example_data_dir = Path(__file__).resolve().parent / "data"
    candidates = [
        repo_root / path,
        example_data_dir / path.name,
    ]
    parts = path.parts
    if "data" in parts:
        data_idx = parts.index("data")
        if data_idx + 1 < len(parts):
            candidates.append(example_data_dir / Path(*parts[data_idx + 1 :]))
    if len(parts) >= 2:
        candidates.append(example_data_dir / parts[-2] / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_normalized_state_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(csv_path)
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if arr.shape == ():
        arr = np.asarray([tuple(arr.tolist())], dtype=arr.dtype)
    names = set(arr.dtype.names or ())
    required = {"x_au", "y_au", "z_au", "vx_au_per_d", "vy_au_per_d", "vz_au_per_d"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} missing required normalized-state columns: {missing}")
    if "t_day" in names:
        t_day = np.asarray(arr["t_day"], dtype=np.float64)
    elif "jd" in names:
        jd = np.asarray(arr["jd"], dtype=np.float64)
        t_day = jd - float(jd[0])
    elif "mjd" in names:
        mjd = np.asarray(arr["mjd"], dtype=np.float64)
        t_day = mjd - float(mjd[0])
    else:
        raise ValueError(f"{path} must contain one of t_day, jd, or mjd")
    position = np.column_stack([arr["x_au"], arr["y_au"], arr["z_au"]]).astype(np.float64, copy=False)
    velocity = np.column_stack([arr["vx_au_per_d"], arr["vy_au_per_d"], arr["vz_au_per_d"]]).astype(
        np.float64,
        copy=False,
    )
    return np.asarray(t_day, dtype=np.float64), position, velocity


def load_state_series_from_manifest(
    manifest_path: str | Path,
    *,
    max_bodies: int | None = None,
    body_ids: Sequence[str] | None = None,
) -> list[StateSeries]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    wanted = None if body_ids is None else {str(item) for item in body_ids}
    rows = []
    for row in list(manifest):
        if not isinstance(row, dict):
            continue
        orbit_id = str(row.get("orbit_id", ""))
        body_name = str(row.get("body_name", orbit_id))
        if wanted is not None and orbit_id not in wanted and body_name not in wanted:
            continue
        rows.append(row)
    if max_bodies is not None and int(max_bodies) > 0:
        rows = rows[: int(max_bodies)]
    out: list[StateSeries] = []
    for row in rows:
        csv_path = _resolve_repo_path(str(row["csv_path"]))
        t_day, position, velocity = _load_normalized_state_csv(csv_path)
        out.append(
            StateSeries(
                orbit_id=str(row.get("orbit_id", csv_path.stem)),
                body_name=str(row.get("body_name", row.get("orbit_id", csv_path.stem))),
                split=str(row.get("split", "candidate")),
                t_day=t_day,
                position_au=position,
                velocity_au_per_d=velocity,
            )
        )
    return out


def estimate_acceleration_from_velocity(t_day: np.ndarray, velocity_au_per_d: np.ndarray) -> np.ndarray:
    t = np.asarray(t_day, dtype=np.float64)
    v = np.asarray(velocity_au_per_d, dtype=np.float64)
    if t.ndim != 1 or v.ndim != 2 or v.shape[1] != 3 or v.shape[0] != t.size:
        raise ValueError("expected t shape (N,) and velocity shape (N, 3)")
    if t.size < 5:
        raise ValueError("need at least five samples to estimate acceleration")
    accel = np.empty_like(v, dtype=np.float64)
    for axis in range(3):
        accel[:, axis] = np.gradient(v[:, axis], t, edge_order=2)
    return accel


def solar_acceleration(position_au: np.ndarray, mu_sun: float) -> np.ndarray:
    r = np.asarray(position_au, dtype=np.float64)
    radius = np.linalg.norm(r, axis=1)
    denom = np.maximum(radius, 1.0e-15) ** 3
    return -float(mu_sun) * r / denom[:, None]


def build_residual_observation_blocks(
    series: Sequence[StateSeries],
    *,
    mu_sun: float,
    stride: int = 10,
    edge_trim: int = 4,
) -> list[ObservationBlock]:
    if int(stride) < 1:
        raise ValueError("stride must be positive")
    if int(edge_trim) < 0:
        raise ValueError("edge_trim must be non-negative")
    blocks: list[ObservationBlock] = []
    for item in list(series):
        observed_accel = estimate_acceleration_from_velocity(item.t_day, item.velocity_au_per_d)
        residual = observed_accel - solar_acceleration(item.position_au, float(mu_sun))
        start = int(edge_trim)
        stop = residual.shape[0] - int(edge_trim) if int(edge_trim) > 0 else residual.shape[0]
        if stop <= start:
            raise ValueError(f"edge_trim removes all samples for {item.orbit_id}")
        sample = slice(start, stop, int(stride))
        blocks.append(
            ObservationBlock(
                orbit_id=item.orbit_id,
                body_name=item.body_name,
                split=item.split,
                t_day=np.asarray(item.t_day[sample], dtype=np.float64),
                position_au=np.asarray(item.position_au[sample], dtype=np.float64),
                residual_accel_au_per_d2=np.asarray(residual[sample], dtype=np.float64),
            )
        )
    return blocks


def split_observation_blocks(
    blocks: Sequence[ObservationBlock],
    *,
    holdout_fraction: float = 0.25,
) -> tuple[list[ObservationBlock], list[ObservationBlock]]:
    block_list = list(blocks)
    if not block_list:
        raise ValueError("cannot split an empty observation block list")
    explicit_holdout = [block for block in block_list if str(block.split) == "holdout"]
    explicit_train = [block for block in block_list if str(block.split) != "holdout"]
    if explicit_holdout and explicit_train:
        return explicit_train, explicit_holdout

    if len(block_list) == 1 or float(holdout_fraction) <= 0.0:
        return block_list, block_list
    period = max(2, int(round(1.0 / min(max(float(holdout_fraction), 1.0e-6), 0.95))))
    test = [block for idx, block in enumerate(block_list) if idx % period == period - 1]
    train = [block for idx, block in enumerate(block_list) if idx % period != period - 1]
    if not test:
        test = [block_list[-1]]
        train = block_list[:-1] or block_list
    if not train:
        train = block_list
    return train, test


def stack_observations(blocks: Sequence[ObservationBlock]) -> ObservationSet:
    block_list = list(blocks)
    if not block_list:
        raise ValueError("cannot stack an empty observation block list")
    t_parts = []
    r_parts = []
    y_parts = []
    body_index_parts = []
    body_names = []
    for idx, block in enumerate(block_list):
        n = int(block.t_day.size)
        if block.position_au.shape != (n, 3) or block.residual_accel_au_per_d2.shape != (n, 3):
            raise ValueError(f"invalid observation shapes for {block.orbit_id}")
        t_parts.append(np.asarray(block.t_day, dtype=np.float64))
        r_parts.append(np.asarray(block.position_au, dtype=np.float64))
        y_parts.append(np.asarray(block.residual_accel_au_per_d2, dtype=np.float64))
        body_index_parts.append(np.full(n, idx, dtype=np.int64))
        body_names.append(str(block.body_name))
    return ObservationSet(
        t_day=np.concatenate(t_parts),
        position_au=np.vstack(r_parts),
        residual_accel_au_per_d2=np.vstack(y_parts),
        body_index=np.concatenate(body_index_parts),
        body_names=tuple(body_names),
    )


def circular_source_positions(
    t_day: np.ndarray,
    *,
    a_au: float,
    phase_rad: float,
    mu_sun: float,
) -> np.ndarray:
    a = float(a_au)
    if a <= 0.0:
        raise ValueError("circular source semi-major axis must be positive")
    mean_motion = math.sqrt(float(mu_sun) / (a ** 3))
    theta = mean_motion * np.asarray(t_day, dtype=np.float64) + float(phase_rad)
    return np.column_stack(
        [
            a * np.cos(theta),
            a * np.sin(theta),
            np.zeros_like(theta, dtype=np.float64),
        ]
    )


def solve_kepler_equation(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    e = float(np.clip(float(eccentricity), 0.0, 0.95))
    m = np.asarray(mean_anomaly, dtype=np.float64)
    two_pi = 2.0 * math.pi
    wrapped = np.mod(m, two_pi)
    ecc = wrapped.copy() if e < 0.8 else np.full_like(wrapped, math.pi, dtype=np.float64)
    for _ in range(50):
        f = ecc - e * np.sin(ecc) - wrapped
        fp = 1.0 - e * np.cos(ecc)
        delta = f / np.maximum(fp, 1.0e-15)
        ecc = ecc - delta
        if float(np.max(np.abs(delta))) < 1.0e-13:
            break
    return np.asarray(ecc + (m - wrapped), dtype=np.float64)


def keplerian_source_positions(
    t_day: np.ndarray,
    *,
    a_au: float,
    eccentricity: float,
    inclination_rad: float,
    node_rad: float,
    arg_peri_rad: float,
    mean_anomaly0_rad: float,
    mu_sun: float,
) -> np.ndarray:
    a = float(a_au)
    if a <= 0.0:
        raise ValueError("Keplerian source semi-major axis must be positive")
    e = float(np.clip(float(eccentricity), 0.0, 0.95))
    inc = float(inclination_rad)
    node = float(node_rad)
    arg = float(arg_peri_rad)
    mean0 = float(mean_anomaly0_rad)
    mean_motion = math.sqrt(float(mu_sun) / (a ** 3))
    mean_anomaly = mean_motion * np.asarray(t_day, dtype=np.float64) + mean0
    ecc_anomaly = solve_kepler_equation(mean_anomaly, e)
    x_orb = a * (np.cos(ecc_anomaly) - e)
    y_orb = a * math.sqrt(max(1.0 - e * e, 0.0)) * np.sin(ecc_anomaly)

    c_o = math.cos(node)
    s_o = math.sin(node)
    c_i = math.cos(inc)
    s_i = math.sin(inc)
    c_w = math.cos(arg)
    s_w = math.sin(arg)

    r11 = c_o * c_w - s_o * s_w * c_i
    r12 = -c_o * s_w - s_o * c_w * c_i
    r21 = s_o * c_w + c_o * s_w * c_i
    r22 = -s_o * s_w + c_o * c_w * c_i
    r31 = s_w * s_i
    r32 = c_w * s_i
    return np.column_stack(
        [
            r11 * x_orb + r12 * y_orb,
            r21 * x_orb + r22 * y_orb,
            r31 * x_orb + r32 * y_orb,
        ]
    )


def third_body_template(position_au: np.ndarray, source_position_au: np.ndarray) -> np.ndarray:
    r = np.asarray(position_au, dtype=np.float64)
    rp = np.asarray(source_position_au, dtype=np.float64)
    if r.shape != rp.shape or r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("expected matching (N, 3) position and source arrays")
    dr = rp - r
    dr_norm = np.linalg.norm(dr, axis=1)
    rp_norm = np.linalg.norm(rp, axis=1)
    direct = dr / (np.maximum(dr_norm, 1.0e-15) ** 3)[:, None]
    indirect = rp / (np.maximum(rp_norm, 1.0e-15) ** 3)[:, None]
    return direct - indirect


def circular_third_body_template(obs: ObservationSet, *, a_au: float, phase_rad: float, mu_sun: float) -> np.ndarray:
    source = circular_source_positions(obs.t_day, a_au=float(a_au), phase_rad=float(phase_rad), mu_sun=float(mu_sun))
    return third_body_template(obs.position_au, source)


def keplerian_third_body_template(obs: ObservationSet, perturber: KeplerianPerturber, *, mu_sun: float) -> np.ndarray:
    source = keplerian_source_positions(
        obs.t_day,
        a_au=float(perturber.a_au),
        eccentricity=float(perturber.eccentricity),
        inclination_rad=float(perturber.inclination_rad),
        node_rad=float(perturber.node_wrapped),
        arg_peri_rad=float(perturber.arg_peri_wrapped),
        mean_anomaly0_rad=float(perturber.mean_anomaly0_wrapped),
        mu_sun=float(mu_sun),
    )
    return third_body_template(obs.position_au, source)


def fit_template_coefficients(
    target: np.ndarray,
    templates: Sequence[np.ndarray],
    *,
    nonnegative: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    y = np.asarray(target, dtype=np.float64)
    template_list = [np.asarray(col, dtype=np.float64) for col in list(templates)]
    if not template_list:
        pred = np.zeros_like(y)
        return np.zeros(0, dtype=np.float64), pred, _fit_metrics(y, pred, k_params=0)
    for col in template_list:
        if col.shape != y.shape:
            raise ValueError("all templates must have the same shape as the target")
    if len(template_list) == 1:
        x_vec = template_list[0].reshape(-1)
        y_vec = y.reshape(-1)
        denom = float(np.dot(x_vec, x_vec))
        if not math.isfinite(denom) or denom <= 0.0:
            coeff = 0.0
        else:
            coeff = float(np.dot(x_vec, y_vec) / denom)
            if nonnegative:
                coeff = max(coeff, 0.0)
        pred = coeff * template_list[0]
        if not np.all(np.isfinite(pred)):
            coeff = 0.0
            pred = np.zeros_like(y)
        return np.asarray([coeff], dtype=np.float64), pred, _fit_metrics(y, pred, k_params=1)
    full_matrix = np.column_stack([col.reshape(-1) for col in template_list])
    matrix = full_matrix
    y_vec = y.reshape(-1)
    norms = np.linalg.norm(matrix, axis=0)
    valid = np.isfinite(norms) & (norms > 0.0)
    if not np.all(valid):
        matrix = matrix[:, valid]
        norms = norms[valid]
    if matrix.shape[1] == 0:
        pred = np.zeros_like(y)
        return np.zeros(len(template_list), dtype=np.float64), pred, _fit_metrics(y, pred, k_params=len(template_list))
    scaled_matrix = matrix / norms[None, :]
    if nonnegative:
        active = np.ones(scaled_matrix.shape[1], dtype=bool)
        scaled_coeffs = np.zeros(scaled_matrix.shape[1], dtype=np.float64)
        while np.any(active):
            active_coeffs, *_ = np.linalg.lstsq(scaled_matrix[:, active], y_vec, rcond=None)
            trial = np.zeros(scaled_matrix.shape[1], dtype=np.float64)
            trial[active] = active_coeffs
            if np.all(trial[active] >= -1.0e-20):
                scaled_coeffs = np.maximum(trial, 0.0)
                break
            active_indices = np.flatnonzero(active)
            active[active_indices[int(np.argmin(trial[active]))]] = False
        else:
            scaled_coeffs = np.zeros(scaled_matrix.shape[1], dtype=np.float64)
    else:
        scaled_coeffs, *_ = np.linalg.lstsq(scaled_matrix, y_vec, rcond=None)
    coeffs_valid = scaled_coeffs / norms
    coeffs = np.zeros(len(template_list), dtype=np.float64)
    coeffs[np.flatnonzero(valid)] = coeffs_valid
    pred = np.zeros_like(y)
    if np.all(np.isfinite(coeffs)):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for coeff, col in zip(coeffs, template_list):
                pred = pred + float(coeff) * col
    if not np.all(np.isfinite(pred)):
        pred = np.zeros_like(y)
        coeffs = np.zeros(len(template_list), dtype=np.float64)
    return coeffs, pred, _fit_metrics(y, pred, k_params=int(coeffs.size))


def _fit_metrics(target: np.ndarray, pred: np.ndarray, *, k_params: int) -> dict[str, float]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    resid = y - p
    sse = float(np.sum(np.square(resid)))
    n = int(y.size)
    rmse = float(math.sqrt(sse / max(n, 1)))
    target_rmse = float(math.sqrt(float(np.mean(np.square(y))))) if n else 0.0
    rel_rmse = float(rmse / max(target_rmse, 1.0e-30))
    bic = float(n * math.log(max(sse / max(n, 1), 1.0e-300)) + int(k_params) * math.log(max(n, 2)))
    return {
        "sse": sse,
        "rmse": rmse,
        "target_rmse": target_rmse,
        "rel_rmse": rel_rmse,
        "n_scalars": int(n),
        "k_params": int(k_params),
        "bic": bic,
    }


def scan_circular_perturber(
    obs: ObservationSet,
    *,
    target: np.ndarray | None = None,
    a_grid: np.ndarray,
    phase_grid: np.ndarray,
    mu_sun: float,
    template_abs_cap: float = 1.0e6,
    min_a_separation: float | None = None,
    existing_a: Sequence[float] = (),
) -> CircularPerturber:
    y = obs.residual_accel_au_per_d2 if target is None else np.asarray(target, dtype=np.float64)
    best: CircularPerturber | None = None
    best_sse = float("inf")
    existing = [float(item) for item in existing_a]
    for a_val in np.asarray(a_grid, dtype=np.float64):
        if min_a_separation is not None and any(abs(float(a_val) - other) < float(min_a_separation) for other in existing):
            continue
        for phase_val in np.asarray(phase_grid, dtype=np.float64):
            template = circular_third_body_template(obs, a_au=float(a_val), phase_rad=float(phase_val), mu_sun=float(mu_sun))
            if not np.all(np.isfinite(template)) or float(np.max(np.abs(template))) > float(template_abs_cap):
                continue
            coeffs, _pred, metrics = fit_template_coefficients(y, [template], nonnegative=True)
            sse = float(metrics["sse"])
            if not math.isfinite(sse):
                continue
            if sse < best_sse:
                best_sse = sse
                best = CircularPerturber(
                    a_au=float(a_val),
                    phase_rad=float(phase_val),
                    mu_au3_per_d2=float(coeffs[0]),
                    train_sse=sse,
                )
    if best is None:
        raise ValueError("empty circular perturber scan after exclusions")
    return best


def refine_circular_perturber(
    obs: ObservationSet,
    initial: CircularPerturber,
    *,
    target: np.ndarray | None = None,
    mu_sun: float,
    a_bounds: tuple[float, float],
    template_abs_cap: float = 1.0e6,
) -> CircularPerturber:
    try:
        from scipy.optimize import minimize
    except Exception:
        return initial

    y = obs.residual_accel_au_per_d2 if target is None else np.asarray(target, dtype=np.float64)

    def objective(params: np.ndarray) -> float:
        a_val = float(params[0])
        phase_val = float(params[1])
        template = circular_third_body_template(obs, a_au=a_val, phase_rad=phase_val, mu_sun=float(mu_sun))
        if not np.all(np.isfinite(template)) or float(np.max(np.abs(template))) > float(template_abs_cap):
            return float("inf")
        _coeffs, _pred, metrics = fit_template_coefficients(y, [template], nonnegative=True)
        return float(metrics["sse"])

    result = minimize(
        objective,
        x0=np.asarray([float(initial.a_au), float(initial.phase_wrapped)], dtype=np.float64),
        bounds=[(float(a_bounds[0]), float(a_bounds[1])), (0.0, 2.0 * math.pi)],
        method="Nelder-Mead",
        options={"maxiter": 120, "xatol": 1.0e-5, "fatol": 1.0e-24},
    )
    if not bool(getattr(result, "success", False)) and not np.isfinite(float(result.fun)):
        return initial
    a_val = float(result.x[0])
    phase_val = float(np.mod(float(result.x[1]), 2.0 * math.pi))
    template = circular_third_body_template(obs, a_au=a_val, phase_rad=phase_val, mu_sun=float(mu_sun))
    coeffs, _pred, metrics = fit_template_coefficients(y, [template], nonnegative=True)
    refined = CircularPerturber(
        a_au=a_val,
        phase_rad=phase_val,
        mu_au3_per_d2=float(coeffs[0]),
        train_sse=float(metrics["sse"]),
    )
    if float(refined.train_sse) <= float(initial.train_sse):
        return refined
    return initial


def evaluate_circular_model(
    obs: ObservationSet,
    perturbers: Sequence[CircularPerturber],
    *,
    mu_sun: float,
    refit_coefficients: bool = True,
) -> tuple[list[CircularPerturber], np.ndarray, dict[str, float]]:
    specs = list(perturbers)
    templates = [
        circular_third_body_template(obs, a_au=item.a_au, phase_rad=item.phase_wrapped, mu_sun=float(mu_sun))
        for item in specs
    ]
    if refit_coefficients and templates:
        coeffs, pred, metrics = fit_template_coefficients(obs.residual_accel_au_per_d2, templates, nonnegative=True)
        metrics = _fit_metrics(obs.residual_accel_au_per_d2, pred, k_params=3 * len(specs))
        updated = [
            CircularPerturber(
                a_au=spec.a_au,
                phase_rad=spec.phase_wrapped,
                mu_au3_per_d2=float(coeffs[idx]),
                train_sse=float(metrics["sse"]),
            )
            for idx, spec in enumerate(specs)
        ]
        return updated, pred, metrics
    coeff_templates = [float(item.mu_au3_per_d2) * template for item, template in zip(specs, templates)]
    pred = np.sum(np.stack(coeff_templates, axis=0), axis=0) if coeff_templates else np.zeros_like(obs.residual_accel_au_per_d2)
    metrics = _fit_metrics(obs.residual_accel_au_per_d2, pred, k_params=3 * len(specs))
    return specs, pred, metrics


def keplerian_from_circular(
    item: CircularPerturber,
    *,
    eccentricity: float = 0.05,
    inclination_rad: float = 0.02,
) -> KeplerianPerturber:
    return KeplerianPerturber(
        a_au=float(item.a_au),
        eccentricity=float(np.clip(float(eccentricity), 0.0, 0.95)),
        inclination_rad=max(float(inclination_rad), 0.0),
        node_rad=0.0,
        arg_peri_rad=0.0,
        mean_anomaly0_rad=float(item.phase_wrapped),
        mu_au3_per_d2=float(item.mu_au3_per_d2),
        train_sse=float(item.train_sse),
    )


def _keplerian_from_param_vector(
    params: np.ndarray,
    *,
    mu_au3_per_d2: float,
    train_sse: float,
    e_max: float,
    i_max: float,
) -> KeplerianPerturber:
    p = np.asarray(params, dtype=np.float64)
    return KeplerianPerturber(
        a_au=float(p[0]),
        eccentricity=float(np.clip(float(p[1]), 0.0, float(e_max))),
        inclination_rad=float(np.clip(float(p[2]), 0.0, float(i_max))),
        node_rad=float(np.mod(float(p[3]), 2.0 * math.pi)),
        arg_peri_rad=float(np.mod(float(p[4]), 2.0 * math.pi)),
        mean_anomaly0_rad=float(np.mod(float(p[5]), 2.0 * math.pi)),
        mu_au3_per_d2=float(mu_au3_per_d2),
        train_sse=float(train_sse),
    )


def _keplerian_param_vector(item: KeplerianPerturber) -> np.ndarray:
    return np.asarray(
        [
            float(item.a_au),
            float(item.eccentricity),
            float(item.inclination_rad),
            float(item.node_wrapped),
            float(item.arg_peri_wrapped),
            float(item.mean_anomaly0_wrapped),
        ],
        dtype=np.float64,
    )


def refine_keplerian_perturber(
    obs: ObservationSet,
    initial: KeplerianPerturber,
    *,
    target: np.ndarray | None = None,
    mu_sun: float,
    a_bounds: tuple[float, float],
    e_bounds: tuple[float, float],
    inclination_bounds: tuple[float, float],
    template_abs_cap: float = 1.0e6,
    maxiter: int = 100,
    maxfev: int = 1200,
) -> KeplerianPerturber:
    try:
        from scipy.optimize import minimize
    except Exception:
        return initial

    y = obs.residual_accel_au_per_d2 if target is None else np.asarray(target, dtype=np.float64)
    objective_scale = max(float(np.sum(np.square(y))), 1.0e-300)
    e_min, e_max = float(e_bounds[0]), float(e_bounds[1])
    i_min, i_max = float(inclination_bounds[0]), float(inclination_bounds[1])
    bounds = [
        (float(a_bounds[0]), float(a_bounds[1])),
        (e_min, e_max),
        (i_min, i_max),
        (0.0, 2.0 * math.pi),
        (0.0, 2.0 * math.pi),
        (0.0, 2.0 * math.pi),
    ]

    def objective(params: np.ndarray) -> float:
        raw = np.asarray(params, dtype=np.float64)
        if (
            float(raw[0]) < bounds[0][0]
            or float(raw[0]) > bounds[0][1]
            or float(raw[1]) < bounds[1][0]
            or float(raw[1]) > bounds[1][1]
            or float(raw[2]) < bounds[2][0]
            or float(raw[2]) > bounds[2][1]
        ):
            return float("inf")
        candidate = _keplerian_from_param_vector(
            raw,
            mu_au3_per_d2=0.0,
            train_sse=float("inf"),
            e_max=e_max,
            i_max=i_max,
        )
        template = keplerian_third_body_template(obs, candidate, mu_sun=float(mu_sun))
        if not np.all(np.isfinite(template)) or float(np.max(np.abs(template))) > float(template_abs_cap):
            return float("inf")
        _coeffs, _pred, metrics = fit_template_coefficients(y, [template], nonnegative=True)
        sse = float(metrics["sse"])
        return sse / objective_scale if math.isfinite(sse) else float("inf")

    x0 = _keplerian_param_vector(initial)
    x0[0] = float(np.clip(x0[0], bounds[0][0], bounds[0][1]))
    x0[1] = float(np.clip(x0[1], bounds[1][0], bounds[1][1]))
    x0[2] = float(np.clip(x0[2], bounds[2][0], bounds[2][1]))
    x0[3:] = np.mod(x0[3:], 2.0 * math.pi)
    initial_obj = objective(x0)
    result = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        method="Powell",
        options={
            "maxiter": int(maxiter),
            "maxfev": int(maxfev),
            "xtol": 1.0e-5,
            "ftol": 1.0e-8,
            "disp": False,
        },
    )
    result_fun = float(getattr(result, "fun", float("inf")))
    if (not np.isfinite(result_fun)) or result_fun > float(initial_obj):
        simplex_steps = np.asarray(
            [
                max(0.03 * float(x0[0]), 0.03),
                max(0.5 * max(float(x0[1]), 0.02), 0.01),
                max(0.5 * max(float(x0[2]), 0.01), math.radians(0.25)),
                0.10,
                0.10,
                0.10,
            ],
            dtype=np.float64,
        )
        simplex = np.vstack([x0, *[x0 + simplex_steps[idx] * np.eye(6)[idx] for idx in range(6)]])
        result_nm = minimize(
            objective,
            x0=x0,
            method="Nelder-Mead",
            options={
                "initial_simplex": simplex,
                "maxiter": int(maxiter),
                "maxfev": int(maxfev),
                "xatol": 1.0e-5,
                "fatol": 1.0e-8,
                "adaptive": True,
                "disp": False,
            },
        )
        if np.isfinite(float(getattr(result_nm, "fun", float("inf")))) and float(result_nm.fun) < result_fun:
            result = result_nm
            result_fun = float(result_nm.fun)

    if not np.isfinite(float(getattr(result, "fun", float("inf")))):
        best_params = x0
        best_obj = initial_obj
    elif float(result.fun) <= float(initial_obj):
        best_params = np.asarray(result.x, dtype=np.float64)
        best_obj = float(result.fun)
    else:
        best_params = x0
        best_obj = initial_obj

    candidate = _keplerian_from_param_vector(
        best_params,
        mu_au3_per_d2=0.0,
        train_sse=float(best_obj),
        e_max=e_max,
        i_max=i_max,
    )
    template = keplerian_third_body_template(obs, candidate, mu_sun=float(mu_sun))
    coeffs, _pred, metrics = fit_template_coefficients(y, [template], nonnegative=True)
    return _keplerian_from_param_vector(
        best_params,
        mu_au3_per_d2=float(coeffs[0]),
        train_sse=float(metrics["sse"]),
        e_max=e_max,
        i_max=i_max,
    )


def evaluate_keplerian_model(
    obs: ObservationSet,
    perturbers: Sequence[KeplerianPerturber],
    *,
    mu_sun: float,
    refit_coefficients: bool = True,
) -> tuple[list[KeplerianPerturber], np.ndarray, dict[str, float]]:
    specs = list(perturbers)
    templates = [keplerian_third_body_template(obs, item, mu_sun=float(mu_sun)) for item in specs]
    if refit_coefficients and templates:
        coeffs, pred, metrics = fit_template_coefficients(obs.residual_accel_au_per_d2, templates, nonnegative=True)
        metrics = _fit_metrics(obs.residual_accel_au_per_d2, pred, k_params=7 * len(specs))
        updated = [
            KeplerianPerturber(
                a_au=spec.a_au,
                eccentricity=spec.eccentricity,
                inclination_rad=spec.inclination_rad,
                node_rad=spec.node_wrapped,
                arg_peri_rad=spec.arg_peri_wrapped,
                mean_anomaly0_rad=spec.mean_anomaly0_wrapped,
                mu_au3_per_d2=float(coeffs[idx]),
                train_sse=float(metrics["sse"]),
            )
            for idx, spec in enumerate(specs)
        ]
        return updated, pred, metrics
    coeff_templates = [float(item.mu_au3_per_d2) * template for item, template in zip(specs, templates)]
    pred = np.sum(np.stack(coeff_templates, axis=0), axis=0) if coeff_templates else np.zeros_like(obs.residual_accel_au_per_d2)
    metrics = _fit_metrics(obs.residual_accel_au_per_d2, pred, k_params=7 * len(specs))
    return specs, pred, metrics


def _phase_distance(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % (2.0 * math.pi)
    return float(min(delta, 2.0 * math.pi - delta))


def _period_days(a_au: float, mu_sun: float) -> float:
    return float(2.0 * math.pi * math.sqrt(float(a_au) ** 3 / float(mu_sun)))


def summarize_perturber(item: CircularPerturber | KeplerianPerturber, *, mu_sun: float) -> dict[str, Any]:
    period_year = _period_days(item.a_au, float(mu_sun)) / 365.25
    mu_ratio = float(item.mu_au3_per_d2) / float(mu_sun)
    known_rows = []
    for name, known in KNOWN_OUTER_PLANETS.items():
        known_rows.append(
            {
                "name": name,
                "a_abs_error_au": float(abs(float(item.a_au) - float(known["a_au"]))),
                "a_rel_error": float(abs(float(item.a_au) - float(known["a_au"])) / float(known["a_au"])),
                "period_abs_error_year": float(abs(period_year - float(known["period_year"]))),
                "mu_ratio_rel_error": float(abs(mu_ratio - float(known["mu_over_sun"])) / float(known["mu_over_sun"])),
            }
        )
    closest = min(known_rows, key=lambda row: row["a_rel_error"])
    out = {
        "a_au": float(item.a_au),
        "mu_au3_per_d2": float(item.mu_au3_per_d2),
        "mu_over_sun": float(mu_ratio),
        "period_day": float(period_year * 365.25),
        "period_year": float(period_year),
        "closest_known_by_a": closest,
        "known_comparisons": known_rows,
    }
    if isinstance(item, KeplerianPerturber):
        out.update(
            {
                "eccentricity": float(item.eccentricity),
                "inclination_rad": float(item.inclination_rad),
                "inclination_deg": float(math.degrees(float(item.inclination_rad))),
                "node_rad": float(item.node_wrapped),
                "arg_peri_rad": float(item.arg_peri_wrapped),
                "mean_anomaly0_rad": float(item.mean_anomaly0_wrapped),
            }
        )
    else:
        out["phase_rad"] = float(item.phase_wrapped)
    return out


def run_discovery(args: argparse.Namespace) -> dict[str, Any]:
    mu_sun = float(args.mu_sun)
    body_ids = None
    if args.body_id:
        body_ids = [str(item).strip() for item in str(args.body_id).split(",") if str(item).strip()]
    series = load_state_series_from_manifest(
        args.raw_manifest,
        max_bodies=None if int(args.max_bodies) <= 0 else int(args.max_bodies),
        body_ids=body_ids,
    )
    blocks = build_residual_observation_blocks(
        series,
        mu_sun=mu_sun,
        stride=int(args.stride),
        edge_trim=int(args.edge_trim),
    )
    train_blocks, test_blocks = split_observation_blocks(blocks, holdout_fraction=float(args.holdout_fraction))
    train_obs = stack_observations(train_blocks)
    test_obs = stack_observations(test_blocks)
    a_grid = np.linspace(float(args.a_min), float(args.a_max), int(args.a_count), dtype=np.float64)
    phase_grid = np.linspace(0.0, 2.0 * math.pi, int(args.phase_count), endpoint=False, dtype=np.float64)

    null_train = _fit_metrics(train_obs.residual_accel_au_per_d2, np.zeros_like(train_obs.residual_accel_au_per_d2), k_params=0)
    null_test = _fit_metrics(test_obs.residual_accel_au_per_d2, np.zeros_like(test_obs.residual_accel_au_per_d2), k_params=0)

    first = scan_circular_perturber(
        train_obs,
        a_grid=a_grid,
        phase_grid=phase_grid,
        mu_sun=mu_sun,
        template_abs_cap=float(args.template_abs_cap),
    )
    if not bool(args.no_refine):
        first = refine_circular_perturber(
            train_obs,
            first,
            mu_sun=mu_sun,
            a_bounds=(float(args.a_min), float(args.a_max)),
            template_abs_cap=float(args.template_abs_cap),
        )
    one_train_specs, one_train_pred, one_train = evaluate_circular_model(train_obs, [first], mu_sun=mu_sun)
    one_test_specs, _one_test_pred, one_test = evaluate_circular_model(
        test_obs,
        one_train_specs,
        mu_sun=mu_sun,
        refit_coefficients=False,
    )
    one_kepler_train_specs: list[KeplerianPerturber] | None = None
    one_kepler_train_pred: np.ndarray | None = None

    models: list[dict[str, Any]] = [
        {
            "name": "null",
            "n_perturbers": 0,
            "train": null_train,
            "test": null_test,
            "perturbers": [],
        },
        {
            "name": "one_circular_perturber",
            "n_perturbers": 1,
            "train": one_train,
            "test": one_test,
            "perturbers": [summarize_perturber(item, mu_sun=mu_sun) for item in one_train_specs],
        },
    ]

    if not bool(args.no_keplerian_refine):
        initial_kepler = keplerian_from_circular(
            one_train_specs[0],
            eccentricity=float(args.keplerian_e_init),
            inclination_rad=math.radians(float(args.keplerian_i_init_deg)),
        )
        first_kepler = refine_keplerian_perturber(
            train_obs,
            initial_kepler,
            mu_sun=mu_sun,
            a_bounds=(float(args.a_min), float(args.a_max)),
            e_bounds=(0.0, float(args.keplerian_e_max)),
            inclination_bounds=(0.0, math.radians(float(args.keplerian_i_max_deg))),
            template_abs_cap=float(args.template_abs_cap),
            maxiter=int(args.keplerian_maxiter),
            maxfev=int(args.keplerian_maxfev),
        )
        one_kepler_train_specs, one_kepler_train_pred, one_kepler_train = evaluate_keplerian_model(
            train_obs,
            [first_kepler],
            mu_sun=mu_sun,
        )
        _one_kepler_test_specs, _one_kepler_test_pred, one_kepler_test = evaluate_keplerian_model(
            test_obs,
            one_kepler_train_specs,
            mu_sun=mu_sun,
            refit_coefficients=False,
        )
        models.append(
            {
                "name": "one_keplerian_perturber",
                "n_perturbers": 1,
                "train": one_kepler_train,
                "test": one_kepler_test,
                "perturbers": [summarize_perturber(item, mu_sun=mu_sun) for item in one_kepler_train_specs],
            }
        )

    if int(args.max_perturbers) >= 2:
        residual_after_first = train_obs.residual_accel_au_per_d2 - one_train_pred
        second = scan_circular_perturber(
            train_obs,
            target=residual_after_first,
            a_grid=a_grid,
            phase_grid=phase_grid,
            mu_sun=mu_sun,
            template_abs_cap=float(args.template_abs_cap),
            min_a_separation=float(args.min_a_separation),
            existing_a=[float(one_train_specs[0].a_au)],
        )
        if not bool(args.no_refine):
            second = refine_circular_perturber(
                train_obs,
                second,
                target=residual_after_first,
                mu_sun=mu_sun,
                a_bounds=(float(args.a_min), float(args.a_max)),
                template_abs_cap=float(args.template_abs_cap),
            )
        two_train_specs, _two_train_pred, two_train = evaluate_circular_model(
            train_obs,
            [one_train_specs[0], second],
            mu_sun=mu_sun,
        )
        _two_test_specs, _two_test_pred, two_test = evaluate_circular_model(
            test_obs,
            two_train_specs,
            mu_sun=mu_sun,
            refit_coefficients=False,
        )
        models.append(
            {
                "name": "two_circular_perturbers_sequential",
                "n_perturbers": 2,
                "train": two_train,
                "test": two_test,
                "perturbers": [summarize_perturber(item, mu_sun=mu_sun) for item in two_train_specs],
            }
        )

        if one_kepler_train_specs is not None and one_kepler_train_pred is not None:
            residual_after_kepler_first = train_obs.residual_accel_au_per_d2 - one_kepler_train_pred
            second_kepler_seed_circular = scan_circular_perturber(
                train_obs,
                target=residual_after_kepler_first,
                a_grid=a_grid,
                phase_grid=phase_grid,
                mu_sun=mu_sun,
                template_abs_cap=float(args.template_abs_cap),
                min_a_separation=float(args.min_a_separation),
                existing_a=[float(one_kepler_train_specs[0].a_au)],
            )
            second_kepler_initial = keplerian_from_circular(
                second_kepler_seed_circular,
                eccentricity=float(args.keplerian_e_init),
                inclination_rad=math.radians(float(args.keplerian_i_init_deg)),
            )
            second_kepler = refine_keplerian_perturber(
                train_obs,
                second_kepler_initial,
                target=residual_after_kepler_first,
                mu_sun=mu_sun,
                a_bounds=(float(args.a_min), float(args.a_max)),
                e_bounds=(0.0, float(args.keplerian_e_max)),
                inclination_bounds=(0.0, math.radians(float(args.keplerian_i_max_deg))),
                template_abs_cap=float(args.template_abs_cap),
                maxiter=int(args.keplerian_maxiter),
                maxfev=int(args.keplerian_maxfev),
            )
            two_kepler_specs, _two_kepler_train_pred, two_kepler_train = evaluate_keplerian_model(
                train_obs,
                [one_kepler_train_specs[0], second_kepler],
                mu_sun=mu_sun,
            )
            _two_kepler_test_specs, _two_kepler_test_pred, two_kepler_test = evaluate_keplerian_model(
                test_obs,
                two_kepler_specs,
                mu_sun=mu_sun,
                refit_coefficients=False,
            )
            models.append(
                {
                    "name": "two_keplerian_perturbers_sequential",
                    "n_perturbers": 2,
                    "train": two_kepler_train,
                    "test": two_kepler_test,
                    "perturbers": [summarize_perturber(item, mu_sun=mu_sun) for item in two_kepler_specs],
                }
            )

    best_by_test_bic = min(models, key=lambda row: float(row["test"]["bic"]))
    summary = {
        "raw_manifest": str(Path(args.raw_manifest).resolve()),
        "mu_sun_au3_per_d2": mu_sun,
        "config": {
            "max_bodies": int(args.max_bodies),
            "stride": int(args.stride),
            "edge_trim": int(args.edge_trim),
            "holdout_fraction": float(args.holdout_fraction),
            "a_min": float(args.a_min),
            "a_max": float(args.a_max),
            "a_count": int(args.a_count),
            "phase_count": int(args.phase_count),
            "max_perturbers": int(args.max_perturbers),
            "min_a_separation": float(args.min_a_separation),
            "template_abs_cap": float(args.template_abs_cap),
            "refine": not bool(args.no_refine),
            "keplerian_refine": not bool(args.no_keplerian_refine),
            "keplerian_e_init": float(args.keplerian_e_init),
            "keplerian_e_max": float(args.keplerian_e_max),
            "keplerian_i_init_deg": float(args.keplerian_i_init_deg),
            "keplerian_i_max_deg": float(args.keplerian_i_max_deg),
            "keplerian_maxiter": int(args.keplerian_maxiter),
            "keplerian_maxfev": int(args.keplerian_maxfev),
        },
        "data": {
            "n_bodies_total": int(len(blocks)),
            "n_bodies_train": int(len(train_blocks)),
            "n_bodies_test": int(len(test_blocks)),
            "n_vectors_train": int(train_obs.n_vectors),
            "n_vectors_test": int(test_obs.n_vectors),
            "train_bodies": list(train_obs.body_names),
            "test_bodies": list(test_obs.body_names),
        },
        "known_outer_planets_reference_only": KNOWN_OUTER_PLANETS,
        "models": models,
        "best_by_test_bic": best_by_test_bic["name"],
    }
    return summary


def _default_manifest() -> Path:
    if DEFAULT_BULK_RAW_MANIFEST.exists():
        return DEFAULT_BULK_RAW_MANIFEST
    return DEFAULT_CURATED_RAW_MANIFEST


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit blind circular and Keplerian third-body perturbation templates to the "
            "weathered Kepler residuals from heliocentric HORIZONS vectors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_manifest", type=str, default=str(_default_manifest()))
    parser.add_argument("--results_dir", type=str, default=str(Path("results") / "kepler_ephemeris_real_third_body"))
    parser.add_argument("--summary_name", type=str, default="third_body_residual_summary.json")
    parser.add_argument("--mu_sun", type=float, default=float(DEFAULT_SOLAR_MU_AU_DAY))
    parser.add_argument("--max_bodies", type=int, default=48, help="Use 0 or a negative value to load every body")
    parser.add_argument("--body_id", type=str, default="", help="Comma-separated orbit_id/body_name filter")
    parser.add_argument("--stride", type=int, default=10, help="Subsample residual observations after differentiating daily velocities")
    parser.add_argument("--edge_trim", type=int, default=4, help="Drop this many acceleration samples from each edge")
    parser.add_argument("--holdout_fraction", type=float, default=0.25)
    parser.add_argument("--a_min", type=float, default=4.0)
    parser.add_argument("--a_max", type=float, default=12.0)
    parser.add_argument("--a_count", type=int, default=73)
    parser.add_argument("--phase_count", type=int, default=96)
    parser.add_argument("--max_perturbers", type=int, default=2)
    parser.add_argument(
        "--min_a_separation",
        type=float,
        default=1.5,
        help="Minimum AU separation between sequential circular perturber scans",
    )
    parser.add_argument("--template_abs_cap", type=float, default=1.0e6)
    parser.add_argument("--no_refine", action="store_true", help="Disable local scipy refinement after grid search")
    parser.add_argument("--no_keplerian_refine", action="store_true", help="Disable eccentric/inclined Keplerian refinement")
    parser.add_argument("--keplerian_e_init", type=float, default=0.05)
    parser.add_argument("--keplerian_e_max", type=float, default=0.25)
    parser.add_argument("--keplerian_i_init_deg", type=float, default=1.0)
    parser.add_argument("--keplerian_i_max_deg", type=float, default=8.0)
    parser.add_argument("--keplerian_maxiter", type=int, default=80)
    parser.add_argument("--keplerian_maxfev", type=int, default=900)
    return parser


def _print_summary(summary: dict[str, Any], summary_path: Path) -> None:
    data = summary["data"]
    print(
        f"Loaded {data['n_bodies_total']} bodies "
        f"({data['n_vectors_train']} train vectors, {data['n_vectors_test']} test vectors)"
    )
    for model in summary["models"]:
        test = model["test"]
        improvement = 1.0 - float(test["sse"]) / max(float(summary["models"][0]["test"]["sse"]), 1.0e-300)
        print(
            f"{model['name']}: test_rel_rmse={float(test['rel_rmse']):.4e} "
            f"test_sse_improvement={improvement:.4%} test_bic={float(test['bic']):.3e}"
        )
        for idx, pert in enumerate(list(model["perturbers"])):
            closest = pert["closest_known_by_a"]
            shape = ""
            if "eccentricity" in pert:
                shape = f" e={float(pert['eccentricity']):.4f} i={float(pert['inclination_deg']):.3f}deg"
            print(
                f"  p{idx + 1}: a={float(pert['a_au']):.4f} AU "
                f"P={float(pert['period_year']):.3f} yr "
                f"mu/mu_sun={float(pert['mu_over_sun']):.4e}{shape} "
                f"closest={closest['name']} "
                f"a_rel_err={float(closest['a_rel_error']):.3%} "
                f"mu_rel_err={float(closest['mu_ratio_rel_error']):.3%}"
            )
    print(f"Best by test BIC: {summary['best_by_test_bic']}")
    print(f"Summary: {summary_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_discovery(args)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / str(args.summary_name)
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    _print_summary(summary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
