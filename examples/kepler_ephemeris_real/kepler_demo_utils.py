# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence
import warnings

from astropy import units as u
from astropy.constants import GM_sun
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from astropy.utils import iers
from astropy.utils.data import CacheMissingWarning
import numpy as np

_BASE_PATH = Path(__file__).resolve().parent / "_base_kepler_utils.py"
_BASE_SPEC = importlib.util.spec_from_file_location("_kepler_reduced_base_utils", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"unable to load base Kepler utilities from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = _BASE
_BASE_SPEC.loader.exec_module(_BASE)

# Avoid stale IERS / cache-cleanup noise when using Astropy's builtin ephemerides offline.
iers.conf.auto_download = False
iers.conf.auto_max_age = None
warnings.filterwarnings("ignore", category=iers.IERSStaleWarning)
warnings.filterwarnings("ignore", category=CacheMissingWarning)

KeplerReducedDataset = _BASE.KeplerReducedDataset

DEFAULT_SOLAR_MU_AU_DAY = float(GM_sun.to_value(u.au**3 / u.day**2))
DEFAULT_START_DATE = "1980-01-01"
DEFAULT_YEARS = 30.0
DEFAULT_CADENCE_DAYS = 1.0
DEFAULT_PROVIDER = "raw_csv"
DEFAULT_PROFILE = "weathered"
DEFAULT_RAW_MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "raw_states_manifest.json"
DATASET_FAMILY = "kepler_ephemeris_real"


@dataclass(frozen=True)
class EphemerisBodySpec:
    orbit_id: str
    body_name: str
    split: str
    csv_path: str | None = None


def _reexport_base_symbols() -> None:
    skip = {"build_default_kepler_datasets", "build_default_orbit_specs", "generate_kepler_dataset"}
    for name in getattr(_BASE, "__all__", []):
        if name in skip:
            continue
        globals()[name] = getattr(_BASE, name)


_reexport_base_symbols()


def _resolved_mu(mu: float | None) -> float:
    if mu is None:
        return float(DEFAULT_SOLAR_MU_AU_DAY)
    mu_f = float(mu)
    if math.isclose(mu_f, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        return float(DEFAULT_SOLAR_MU_AU_DAY)
    return mu_f


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_repo_paths(raw_path: str | Path) -> list[Path]:
    raw = str(raw_path)
    path = Path(raw)
    repo_root = _repo_root()
    example_data_dir = Path(__file__).resolve().parent / "data"

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if raw.startswith("examples/kepler_ephemeris/"):
            candidates.append(
                repo_root / raw.replace("examples/kepler_ephemeris/", "examples/kepler_ephemeris_real/", 1)
            )
        candidates.append(repo_root / path)

    if not path.is_absolute():
        parts = path.parts
        if "data" in parts:
            idx = parts.index("data")
            if idx + 1 < len(parts):
                candidates.append(example_data_dir / Path(*parts[idx + 1 :]))
        candidates.append(example_data_dir / path.name)
        if len(parts) >= 2:
            candidates.append(example_data_dir / parts[-2] / path.name)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _resolve_repo_path(raw_path: str | Path) -> Path:
    candidates = _candidate_repo_paths(raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if not candidates:
        raise ValueError(f"unable to derive a path candidate from {raw_path!r}")
    return candidates[0].resolve()


def _canonicalize_manifest_dict(manifest: dict[str, Any]) -> dict[str, Any]:
    out = dict(manifest)
    rows = []
    for row in list(manifest.get("orbits", []) or []):
        if not isinstance(row, dict):
            continue
        row_copy = dict(row)
        for key in ("combined_csv", "omega_csv", "rddot_csv"):
            raw_path = row_copy.get(key, None)
            if raw_path is not None:
                row_copy[key] = str(_resolve_repo_path(str(raw_path)))
        rows.append(row_copy)
    if rows:
        out["orbits"] = rows

    ephemeris_generation = out.get("ephemeris_generation", None)
    if isinstance(ephemeris_generation, dict):
        ephem = dict(ephemeris_generation)
        raw_manifest_path = ephem.get("raw_manifest_path", None)
        if raw_manifest_path is not None:
            ephem["raw_manifest_path"] = str(_resolve_repo_path(str(raw_manifest_path)))
        raw_rows = []
        for row in list(ephem.get("raw_manifest_rows", []) or []):
            if not isinstance(row, dict):
                continue
            row_copy = dict(row)
            csv_path = row_copy.get("csv_path", None)
            if csv_path is not None:
                row_copy["csv_path"] = str(_resolve_repo_path(str(csv_path)))
            raw_rows.append(row_copy)
        if raw_rows:
            ephem["raw_manifest_rows"] = raw_rows
        out["ephemeris_generation"] = ephem
    return out


def build_generation_provenance(
    *,
    provider: str,
    profile: str,
    mu: float | None,
    start_date: str,
    years: float,
    cadence_days: float,
    raw_manifest: str | Path | None = None,
    accel_source: str = "gradient",
) -> dict[str, Any]:
    provider_name = str(provider)
    provenance: dict[str, Any] = {
        "dataset_family": DATASET_FAMILY,
        "provider": provider_name,
        "profile": str(profile),
        "mu": float(_resolved_mu(mu)),
        "start_date": str(start_date),
        "years": float(years),
        "cadence_days": float(cadence_days),
        "accel_source": str(accel_source),
    }
    if raw_manifest is None:
        return provenance

    raw_manifest_path = _resolve_repo_path(raw_manifest)
    rows = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    source_rows = []
    for row in list(rows):
        if not isinstance(row, dict):
            continue
        csv_path = row.get("csv_path", None)
        source_rows.append(
            {
                "orbit_id": str(row.get("orbit_id", "")),
                "body_name": str(row.get("body_name", row.get("orbit_id", ""))),
                "split": str(row.get("split", "")),
                "csv_path": None if csv_path is None else str(_resolve_repo_path(str(csv_path))),
                "horizons_command": row.get("horizons_command", None),
                "target_name": row.get("target_name", None),
                "center_name": row.get("center_name", None),
                "start_date": row.get("start_date", None),
                "stop_date": row.get("stop_date", None),
                "cadence_days": row.get("cadence_days", None),
                "n_rows": row.get("n_rows", None),
            }
        )
    provenance["raw_manifest_path"] = str(raw_manifest_path.resolve())
    provenance["raw_manifest_rows"] = source_rows
    return provenance


def write_generated_artifacts(
    output_root: str | Path,
    datasets: Sequence[KeplerReducedDataset],
    *,
    generation_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = _BASE.write_generated_artifacts(output_root, datasets)
    if generation_provenance is None:
        return result

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ephemeris_generation"] = _BASE._jsonable(generation_provenance)
    manifest_path.write_text(json.dumps(_BASE._jsonable(manifest), indent=2), encoding="utf-8")
    return result


def load_generation_provenance(data_dir: str | Path) -> dict[str, Any] | None:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _canonicalize_manifest_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    raw = manifest.get("ephemeris_generation", None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("manifest ephemeris_generation field must be a JSON object")
    return dict(raw)


def _sample_time_grid(
    *,
    start_date: str,
    years: float,
    cadence_days: float,
) -> tuple[Time, np.ndarray]:
    cadence = float(cadence_days)
    if cadence <= 0.0:
        raise ValueError("cadence_days must be positive")
    total_days = 365.25 * float(years)
    if total_days <= 0.0:
        raise ValueError("years must be positive")
    n_steps = max(32, int(math.floor(total_days / cadence)))
    t_days = cadence * np.arange(n_steps, dtype=np.float64)
    times = Time(str(start_date), scale="tdb") + t_days * u.day
    return times, t_days


def build_default_orbit_specs(
    *,
    provider: str = DEFAULT_PROVIDER,
    raw_manifest: str | Path | None = None,
    seed: int = 123,
    train_samples: int = 1024,
    validation_samples: int = 1024,
    holdout_samples: int = 2048,
) -> list[EphemerisBodySpec]:
    _ = (seed, train_samples, validation_samples, holdout_samples)
    if str(provider) == "astropy_builtin":
        return [
            EphemerisBodySpec(orbit_id="venus", body_name="venus", split="train"),
            EphemerisBodySpec(orbit_id="earth", body_name="earth", split="train"),
            EphemerisBodySpec(orbit_id="jupiter", body_name="jupiter", split="train"),
            EphemerisBodySpec(orbit_id="saturn", body_name="saturn", split="train"),
            EphemerisBodySpec(orbit_id="mars", body_name="mars", split="validation"),
            EphemerisBodySpec(orbit_id="mercury", body_name="mercury", split="holdout"),
        ]
    if str(provider) == "raw_csv":
        manifest_path = DEFAULT_RAW_MANIFEST_PATH if raw_manifest is None else _resolve_repo_path(raw_manifest)
        rows = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        out = []
        for row in list(rows):
            if not isinstance(row, dict):
                continue
            csv_path = row.get("csv_path", None)
            if csv_path is None:
                raise ValueError(f"raw manifest row missing csv_path: {row!r}")
            body_name = str(row.get("body_name", row.get("orbit_id", "")) or "").strip()
            orbit_id = str(row.get("orbit_id", body_name) or "").strip()
            split = str(row.get("split", "")).strip()
            if not orbit_id or not body_name or not split:
                raise ValueError(f"raw manifest row missing orbit_id/body_name/split: {row!r}")
            out.append(
                EphemerisBodySpec(
                    orbit_id=orbit_id,
                    body_name=body_name,
                    split=split,
                    csv_path=str(_resolve_repo_path(str(csv_path))),
                )
            )
        if not out:
            raise ValueError(f"raw manifest {manifest_path} did not define any usable bodies")
        out.sort(key=lambda spec: (_BASE._split_sort_key(spec.split), spec.orbit_id))
        return out
    raise ValueError(f"unsupported provider {provider!r}")


def _normalized(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return np.asarray(v, dtype=np.float64) / norm


def _projected_state_basis(position_xyz: np.ndarray, h_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_seed = np.asarray(position_xyz, dtype=np.float64) - np.dot(position_xyz, h_hat) * h_hat
    if float(np.linalg.norm(x_seed)) <= 1.0e-12:
        trial = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(trial, h_hat))) > 0.95:
            trial = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        x_seed = trial - np.dot(trial, h_hat) * h_hat
    x_hat = _normalized(x_seed)
    y_hat = _normalized(np.cross(h_hat, x_hat))
    return x_hat, y_hat


def _project_series_to_plane(
    positions_xyz: np.ndarray,
    velocities_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_vecs = np.cross(positions_xyz, velocities_xyz)
    h_hat = _normalized(np.mean(h_vecs, axis=0))
    x_hat, y_hat = _projected_state_basis(positions_xyz[0], h_hat)
    x = np.sum(positions_xyz * x_hat[None, :], axis=1)
    y = np.sum(positions_xyz * y_hat[None, :], axis=1)
    vx = np.sum(velocities_xyz * x_hat[None, :], axis=1)
    vy = np.sum(velocities_xyz * y_hat[None, :], axis=1)
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(vx, dtype=np.float64),
        np.asarray(vy, dtype=np.float64),
    )


def _initial_orbital_elements_from_state(
    *,
    x: float,
    y: float,
    vx: float,
    vy: float,
    mu: float,
) -> dict[str, float]:
    r_vec = np.asarray([float(x), float(y)], dtype=np.float64)
    v_vec = np.asarray([float(vx), float(vy)], dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    v_sq = float(np.dot(v_vec, v_vec))
    rv = float(np.dot(r_vec, v_vec))
    h = float(x * vy - y * vx)
    energy = 0.5 * v_sq - float(mu) / r
    if energy >= 0.0:
        raise ValueError("expected a bound heliocentric orbit with negative specific energy")
    a = -float(mu) / (2.0 * energy)
    e_vec = ((v_sq - float(mu) / r) * r_vec - rv * v_vec) / float(mu)
    e = float(np.linalg.norm(e_vec))
    if e >= 1.0:
        raise ValueError("expected eccentricity < 1 for a bound Kepler orbit")
    if e > 1.0e-12:
        cos_E = (1.0 - r / a) / e
        sin_E = rv / (e * math.sqrt(float(mu) * a))
        E0 = math.atan2(sin_E, cos_E)
        mean_anomaly0 = E0 - e * math.sin(E0)
    else:
        mean_anomaly0 = math.atan2(y, x)
    return {
        "a": float(a),
        "e": float(max(e, 0.0)),
        "h": float(h),
        "energy": float(energy),
        "mean_anomaly0": float(mean_anomaly0),
    }


def _propagate_clean_dataset(
    *,
    orbit_id: str,
    split: str,
    mu: float,
    t_days: np.ndarray,
    positions_xyz: np.ndarray,
    velocities_xyz: np.ndarray,
) -> KeplerReducedDataset:
    x0, y0, vx0, vy0 = _project_series_to_plane(positions_xyz[:1], velocities_xyz[:1])
    elements = _initial_orbital_elements_from_state(
        x=float(x0[0]),
        y=float(y0[0]),
        vx=float(vx0[0]),
        vy=float(vy0[0]),
        mu=float(mu),
    )
    a = float(elements["a"])
    e = float(elements["e"])
    mean_motion = math.sqrt(float(mu) / (a ** 3))
    mean_anomaly = float(elements["mean_anomaly0"]) + mean_motion * np.asarray(t_days, dtype=np.float64)
    eccentric_anomaly = np.asarray(_BASE._solve_kepler_equation(mean_anomaly, e), dtype=np.float64)

    cos_e = np.cos(eccentric_anomaly)
    sin_e = np.sin(eccentric_anomaly)
    sqrt_one_minus_e2 = math.sqrt(max(1.0 - e * e, 0.0))
    radius = a * (1.0 - e * cos_e)
    x = a * (cos_e - e)
    y = a * sqrt_one_minus_e2 * sin_e
    theta = np.unwrap(np.arctan2(y, x))
    edot = mean_motion / (1.0 - e * cos_e)
    vx = -a * sin_e * edot
    vy = a * sqrt_one_minus_e2 * cos_e * edot
    h = math.sqrt(float(mu) * a * (1.0 - e * e))
    omega = h / np.square(radius)
    rdot = a * e * sin_e * edot
    rddot = (h * h) / np.power(radius, 3) - float(mu) / np.square(radius)
    ax = -float(mu) * x / np.power(radius, 3)
    ay = -float(mu) * y / np.power(radius, 3)
    period = 2.0 * math.pi / mean_motion
    energy = -float(mu) / (2.0 * a)
    dynamic_range = float(np.max(radius) / max(float(np.min(radius)), 1.0e-12))

    return KeplerReducedDataset(
        orbit_id=str(orbit_id),
        split=str(split),
        mu=float(mu),
        a=float(a),
        e=float(e),
        period=float(period),
        mean_motion=float(mean_motion),
        h=float(h),
        energy=float(energy),
        dynamic_range=float(dynamic_range),
        t=np.asarray(t_days, dtype=np.float64),
        mean_anomaly=np.asarray(mean_anomaly, dtype=np.float64),
        eccentric_anomaly=np.asarray(eccentric_anomaly, dtype=np.float64),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        vx=np.asarray(vx, dtype=np.float64),
        vy=np.asarray(vy, dtype=np.float64),
        ax=np.asarray(ax, dtype=np.float64),
        ay=np.asarray(ay, dtype=np.float64),
        r=np.asarray(radius, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        rdot=np.asarray(rdot, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        rddot=np.asarray(rddot, dtype=np.float64),
    )


_ACCEL_EDGE_TRIM = 32  # samples excluded from each end in surrogate diagnostics


def _series_rel_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(pred - ref))) / np.sqrt(np.mean(np.square(ref))))


def _chart_channel_fit(
    t_days: np.ndarray,
    series: np.ndarray,
    *,
    harmonic: bool = False,
    cheap_fit_cfg: Any | None = None,
    deep_fit_cfg: Any | None = None,
    gn_iters: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cylinder-chart surrogate for one 1D channel (nestynet.charts).

    The chart is a cylinder: a circle lift (cos M, sin M) at the dominant
    periodicity read off the data (windowed matched filter, then Gauss-Newton
    refinement THROUGH the fit) times a slow linear axis that absorbs element
    drift and perturbation content a fixed circle cannot carry.  With
    ``harmonic=True`` a second circle at twice the refined frequency is added
    (the eccentricity harmonic).  No Kepler prior enters; the construction
    only uses "the data have a dominant periodicity".

    Returns (analytic first derivative, fitted values, diagnostics).
    Derivatives are exact chain-rule derivatives of the fit, never finite
    differences.
    """
    import torch

    from nestynet.charts import (
        CircleChart,
        FitConfig,
        IdentityChart,
        ProductChart,
        fit_chart,
        gn_refine_params,
        matched_filter_frequency,
    )

    t = np.asarray(t_days, dtype=np.float64)
    values = np.asarray(series, dtype=np.float64)
    cheap = cheap_fit_cfg or FitConfig(segments=16, epochs=300, restarts=2)
    deep = deep_fit_cfg or FitConfig(segments=48, epochs=2000, restarts=2)

    omega0 = float(matched_filter_frequency(values, t))
    x0 = float(0.5 * (t[0] + t[-1]))
    subcharts = [CircleChart(omega0, x0=x0)]
    if harmonic:
        subcharts.append(CircleChart(2.0 * omega0, x0=x0))
    subcharts.append(IdentityChart.from_data(torch.tensor(t.reshape(-1, 1), dtype=torch.float64)))
    chart = ProductChart(subcharts)

    refine_names = ["0.omega"] + (["1.omega"] if harmonic else [])
    chart = gn_refine_params(t, values, chart, refine_names, cheap, iters=int(gn_iters))
    omega = float(chart.get_param("0.omega"))
    if harmonic:
        chart.set_param("1.omega", 2.0 * omega)

    res = fit_chart(t, values, chart, deep)
    X = torch.tensor(t.reshape(-1, 1), dtype=torch.float64)
    with torch.no_grad():
        d1 = res.charted_model.grad(X)[:, 0, 0].cpu().numpy()
        fit = res.charted_model(X)[:, 0].cpu().numpy()
    diagnostics = {
        "chart": chart.describe(),
        "val_rel_rmse": float(res.val_rel_rmse),
        "omegas": [float(omega)] + ([2.0 * omega] if harmonic else []),
        "matched_filter_omega": omega0,
    }
    return np.asarray(d1, dtype=np.float64), np.asarray(fit, dtype=np.float64), diagnostics


def _surrogate_accelerations(
    *,
    t_days: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    certificate: bool = False,
    harmonic: bool = False,
    cheap_fit_cfg: Any | None = None,
    deep_fit_cfg: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Analytic accelerations from chart surrogates of the velocity channels.

    HORIZONS supplies velocities as data, so the acceleration is ONE
    unsupervised derivative away (fit v, differentiate once).  With
    ``certificate=True`` the position channels are fitted as well and their
    analytic first derivatives are scored against the velocity DATA; that
    measured value-to-derivative gap certifies the unsupervised acceleration
    channel without ground-truth accelerations (paper-1 derivative-gap law).
    """
    kwargs = {"harmonic": bool(harmonic), "cheap_fit_cfg": cheap_fit_cfg, "deep_fit_cfg": deep_fit_cfg}
    ax, _fit_vx, diag_vx = _chart_channel_fit(t_days, vx, **kwargs)
    ay, _fit_vy, diag_vy = _chart_channel_fit(t_days, vy, **kwargs)

    trim = slice(_ACCEL_EDGE_TRIM, max(len(t_days) - _ACCEL_EDGE_TRIM, _ACCEL_EDGE_TRIM + 1))
    ax_fd = np.gradient(vx, t_days, edge_order=2)
    ay_fd = np.gradient(vy, t_days, edge_order=2)
    provenance: dict[str, Any] = {
        "accel_source": "chart_surrogate",
        "channels": {"vx": diag_vx, "vy": diag_vy},
        "fd_rel_diff": {
            "ax": _series_rel_rmse(ax[trim], ax_fd[trim]),
            "ay": _series_rel_rmse(ay[trim], ay_fd[trim]),
        },
    }

    if certificate:
        dxdt, _fit_x, diag_x = _chart_channel_fit(t_days, x, **kwargs)
        dydt, _fit_y, diag_y = _chart_channel_fit(t_days, y, **kwargs)
        gap_x = _series_rel_rmse(dxdt[trim], vx[trim])
        gap_y = _series_rel_rmse(dydt[trim], vy[trim])
        provenance["certificate"] = {
            "channels": {"x": diag_x, "y": diag_y},
            "measured_derivative_rel_rmse": {"x": gap_x, "y": gap_y},
            "gap_ratio": {
                "x": gap_x / max(diag_x["val_rel_rmse"], 1.0e-300),
                "y": gap_y / max(diag_y["val_rel_rmse"], 1.0e-300),
            },
        }
    return ax, ay, provenance


def _surrogate_accel_input_sha(
    t_days: np.ndarray,
    positions_xyz: np.ndarray,
    velocities_xyz: np.ndarray,
    *,
    certificate: bool,
    harmonic: bool,
) -> str:
    """Portable cache key: hash the pre-projection 3D state series.

    The planar series the fits consume pass through a per-body orbital-plane
    eigenbasis whose last bits are platform-dependent (LAPACK), which made
    array-level keys non-portable across machines.  The raw parsed states are
    bit-identical wherever the CSV bytes are, so the key is taken upstream of
    the projection; the schema tag keeps pre-v2 entries from ever aliasing.
    """
    import hashlib

    digest = hashlib.sha256()
    for arr in (t_days, positions_xyz, velocities_xyz):
        digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    digest.update(
        f"key_schema=xyz_v2;certificate={bool(certificate)};harmonic={bool(harmonic)}".encode()
    )
    return digest.hexdigest()


def _cached_surrogate_accelerations(
    cache_dir: str | Path,
    orbit_id: str,
    *,
    t_days: np.ndarray,
    positions_xyz: np.ndarray,
    velocities_xyz: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    certificate: bool,
    harmonic: bool,
    cheap_fit_cfg: Any | None,
    deep_fit_cfg: Any | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Content-addressed per-body cache for the surrogate accelerations.

    The cache key is a SHA-256 over the exact planar series the fits consume
    plus the configuration flags, so a stale or foreign cache entry can never
    be silently reused.
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    input_sha = _surrogate_accel_input_sha(
        t_days, positions_xyz, velocities_xyz, certificate=certificate, harmonic=harmonic
    )
    cache_path = cache_root / f"{orbit_id}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            if str(payload["input_sha"]) == input_sha:
                provenance = json.loads(str(payload["provenance_json"]))
                provenance["accel_cache"] = {"hit": True, "path": str(cache_path)}
                return (
                    np.asarray(payload["ax"], dtype=np.float64),
                    np.asarray(payload["ay"], dtype=np.float64),
                    provenance,
                )
    ax, ay, provenance = _surrogate_accelerations(
        t_days=t_days,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        certificate=certificate,
        harmonic=harmonic,
        cheap_fit_cfg=cheap_fit_cfg,
        deep_fit_cfg=deep_fit_cfg,
    )
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        ax=ax,
        ay=ay,
        input_sha=np.str_(input_sha),
        provenance_json=np.str_(json.dumps(provenance)),
    )
    temporary.replace(cache_path)
    provenance = dict(provenance)
    provenance["accel_cache"] = {"hit": False, "path": str(cache_path)}
    return ax, ay, provenance


def load_cached_surrogate_accels(
    cache_dir: str | Path,
    orbit_id: str,
    *,
    t_days: np.ndarray,
    positions_xyz: np.ndarray,
    velocities_xyz: np.ndarray,
    certificate: bool = True,
    harmonic: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Read-only, SHA-verified load of a precomputed surrogate-accel entry.

    Returns ``(ax, ay, provenance)`` in the projected orbital-plane basis, or
    ``None`` when the entry is absent or was computed from different inputs
    or flags.  Never triggers a fit.
    """
    cache_path = Path(cache_dir) / f"{orbit_id}.npz"
    if not cache_path.exists():
        return None
    input_sha = _surrogate_accel_input_sha(
        t_days, positions_xyz, velocities_xyz, certificate=certificate, harmonic=harmonic
    )
    with np.load(cache_path, allow_pickle=False) as payload:
        if str(payload["input_sha"]) != input_sha:
            return None
        return (
            np.asarray(payload["ax"], dtype=np.float64),
            np.asarray(payload["ay"], dtype=np.float64),
            json.loads(str(payload["provenance_json"])),
        )


def cached_surrogate_rddot(
    cache_dir: str | Path,
    orbit_id: str,
    *,
    t_days: np.ndarray,
    positions_xyz: np.ndarray,
    velocities_xyz: np.ndarray,
    certificate: bool = True,
    harmonic: bool = False,
) -> np.ndarray | None:
    """Radial acceleration on the full time grid from cached surrogate accels.

    ``rddot = (x*ax + y*ay)/r + r*omega^2`` is exact algebra given the analytic
    ``(ax, ay)``; no differentiation happens here.  Returns ``None`` on a cache
    miss (see :func:`load_cached_surrogate_accels`).
    """
    loaded = load_cached_surrogate_accels(
        cache_dir,
        orbit_id,
        t_days=t_days,
        positions_xyz=positions_xyz,
        velocities_xyz=velocities_xyz,
        certificate=certificate,
        harmonic=harmonic,
    )
    if loaded is None:
        return None
    ax, ay, _provenance = loaded
    x, y, vx, vy = _project_series_to_plane(positions_xyz, velocities_xyz)
    r = np.sqrt(np.square(x) + np.square(y))
    omega = (x * vy - y * vx) / np.maximum(np.square(r), 1.0e-12)
    return (x * ax + y * ay) / np.maximum(r, 1.0e-12) + r * np.square(omega)


def _state_series_to_weathered_dataset(
    *,
    orbit_id: str,
    split: str,
    mu: float,
    t_days: np.ndarray,
    positions_xyz: np.ndarray,
    velocities_xyz: np.ndarray,
    accel_source: str = "gradient",
    accel_certificate: bool = False,
    accel_harmonic: bool = False,
    accel_cache_dir: str | Path | None = None,
    cheap_fit_cfg: Any | None = None,
    deep_fit_cfg: Any | None = None,
) -> KeplerReducedDataset:
    x, y, vx, vy = _project_series_to_plane(positions_xyz, velocities_xyz)
    accel_provenance: dict[str, Any] | None = None
    if str(accel_source) == "gradient":
        ax = np.gradient(vx, t_days, edge_order=2)
        ay = np.gradient(vy, t_days, edge_order=2)
    elif str(accel_source) == "surrogate":
        surrogate_kwargs: dict[str, Any] = {
            "t_days": t_days,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "certificate": bool(accel_certificate),
            "harmonic": bool(accel_harmonic),
            "cheap_fit_cfg": cheap_fit_cfg,
            "deep_fit_cfg": deep_fit_cfg,
        }
        if accel_cache_dir is not None:
            ax, ay, accel_provenance = _cached_surrogate_accelerations(
                accel_cache_dir,
                str(orbit_id),
                positions_xyz=positions_xyz,
                velocities_xyz=velocities_xyz,
                **surrogate_kwargs,
            )
        else:
            ax, ay, accel_provenance = _surrogate_accelerations(**surrogate_kwargs)
    else:
        raise ValueError(f"unsupported accel_source {accel_source!r}")
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.unwrap(np.arctan2(y, x))
    rdot = (x * vx + y * vy) / np.maximum(r, 1.0e-12)
    omega = (x * vy - y * vx) / np.maximum(np.square(r), 1.0e-12)
    radial_component = (x * ax + y * ay) / np.maximum(r, 1.0e-12)
    rddot = radial_component + r * np.square(omega)

    h_series = x * vy - y * vx
    v_sq = np.square(vx) + np.square(vy)
    energy_series = 0.5 * v_sq - float(mu) / np.maximum(r, 1.0e-12)
    e_vec = (
        (v_sq - float(mu) / np.maximum(r, 1.0e-12))[:, None] * np.column_stack([x, y])
        - (x * vx + y * vy)[:, None] * np.column_stack([vx, vy])
    ) / float(mu)
    e_series = np.linalg.norm(e_vec, axis=1)

    h = float(np.mean(h_series))
    energy = float(np.mean(energy_series))
    a = float(-float(mu) / (2.0 * energy)) if energy < 0.0 else float("nan")
    e = float(np.mean(e_series))
    mean_motion = float(math.sqrt(float(mu) / (a ** 3))) if math.isfinite(a) and a > 0.0 else float("nan")
    period = float(2.0 * math.pi / mean_motion) if math.isfinite(mean_motion) and mean_motion > 0.0 else float("nan")
    dynamic_range = float(np.max(r) / max(float(np.min(r)), 1.0e-12))

    mean_anomaly = np.full_like(t_days, np.nan, dtype=np.float64)
    eccentric_anomaly = np.full_like(t_days, np.nan, dtype=np.float64)
    return KeplerReducedDataset(
        accel_provenance=accel_provenance,
        orbit_id=str(orbit_id),
        split=str(split),
        mu=float(mu),
        a=float(a),
        e=float(e),
        period=float(period),
        mean_motion=float(mean_motion),
        h=float(h),
        energy=float(energy),
        dynamic_range=float(dynamic_range),
        t=np.asarray(t_days, dtype=np.float64),
        mean_anomaly=mean_anomaly,
        eccentric_anomaly=eccentric_anomaly,
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        vx=np.asarray(vx, dtype=np.float64),
        vy=np.asarray(vy, dtype=np.float64),
        ax=np.asarray(ax, dtype=np.float64),
        ay=np.asarray(ay, dtype=np.float64),
        r=np.asarray(r, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        rdot=np.asarray(rdot, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        rddot=np.asarray(rddot, dtype=np.float64),
    )


def _heliocentric_states_from_astropy(body_name: str, times: Time) -> tuple[np.ndarray, np.ndarray]:
    pos_body, vel_body = get_body_barycentric_posvel(str(body_name), times)
    pos_sun, vel_sun = get_body_barycentric_posvel("sun", times)
    position_xyz = (pos_body.xyz - pos_sun.xyz).to_value(u.au).T
    velocity_xyz = (vel_body.xyz - vel_sun.xyz).to_value(u.au / u.day).T
    return (
        np.asarray(position_xyz, dtype=np.float64),
        np.asarray(velocity_xyz, dtype=np.float64),
    )


def _load_normalized_state_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.genfromtxt(Path(csv_path), delimiter=",", names=True, dtype=np.float64)
    if arr.shape == ():
        arr = np.asarray([tuple(arr.tolist())], dtype=arr.dtype)
    names = set(arr.dtype.names or ())
    if {"x_au", "y_au", "z_au", "vx_au_per_d", "vy_au_per_d", "vz_au_per_d"} - names:
        missing = sorted({"x_au", "y_au", "z_au", "vx_au_per_d", "vy_au_per_d", "vz_au_per_d"} - names)
        raise ValueError(f"{csv_path} missing required normalized-state columns: {missing}")
    if "t_day" in names:
        t_days = np.asarray(arr["t_day"], dtype=np.float64)
    elif "jd" in names:
        jd = np.asarray(arr["jd"], dtype=np.float64)
        t_days = jd - float(jd[0])
    elif "mjd" in names:
        mjd = np.asarray(arr["mjd"], dtype=np.float64)
        t_days = mjd - float(mjd[0])
    else:
        raise ValueError(f"{csv_path} must contain one of t_day, jd, or mjd")
    positions_xyz = np.column_stack([arr["x_au"], arr["y_au"], arr["z_au"]]).astype(np.float64, copy=False)
    velocities_xyz = np.column_stack([arr["vx_au_per_d"], arr["vy_au_per_d"], arr["vz_au_per_d"]]).astype(
        np.float64,
        copy=False,
    )
    return np.asarray(t_days, dtype=np.float64), positions_xyz, velocities_xyz


def generate_kepler_dataset(
    spec: EphemerisBodySpec,
    *,
    mu: float | None = None,
    profile: str = "clean",
    start_date: str = DEFAULT_START_DATE,
    years: float = DEFAULT_YEARS,
    cadence_days: float = DEFAULT_CADENCE_DAYS,
    accel_source: str = "gradient",
    accel_certificate: bool = False,
    accel_harmonic: bool = False,
    accel_cache_dir: str | Path | None = None,
    cheap_fit_cfg: Any | None = None,
    deep_fit_cfg: Any | None = None,
) -> KeplerReducedDataset:
    mu_f = _resolved_mu(mu)
    mode = str(profile)
    if mode not in {"clean", "weathered"}:
        raise ValueError(f"unsupported ephemeris profile {profile!r}")
    if mode == "clean" and str(accel_source) != "gradient":
        raise ValueError("accel_source applies to the weathered profile only; the clean propagation has exact accelerations")

    if spec.csv_path is None:
        times, t_days = _sample_time_grid(
            start_date=str(start_date),
            years=float(years),
            cadence_days=float(cadence_days),
        )
        positions_xyz, velocities_xyz = _heliocentric_states_from_astropy(spec.body_name, times)
    else:
        t_days, positions_xyz, velocities_xyz = _load_normalized_state_csv(spec.csv_path)

    if mode == "clean":
        return _propagate_clean_dataset(
            orbit_id=spec.orbit_id,
            split=spec.split,
            mu=mu_f,
            t_days=t_days,
            positions_xyz=positions_xyz,
            velocities_xyz=velocities_xyz,
        )
    return _state_series_to_weathered_dataset(
        orbit_id=spec.orbit_id,
        split=spec.split,
        mu=mu_f,
        t_days=t_days,
        positions_xyz=positions_xyz,
        velocities_xyz=velocities_xyz,
        accel_source=str(accel_source),
        accel_certificate=bool(accel_certificate),
        accel_harmonic=bool(accel_harmonic),
        accel_cache_dir=accel_cache_dir,
        cheap_fit_cfg=cheap_fit_cfg,
        deep_fit_cfg=deep_fit_cfg,
    )


def build_default_kepler_datasets(
    *,
    mu: float | None = None,
    seed: int = 123,
    train_samples: int = 1024,
    validation_samples: int = 1024,
    holdout_samples: int = 2048,
    provider: str = DEFAULT_PROVIDER,
    profile: str = DEFAULT_PROFILE,
    start_date: str = DEFAULT_START_DATE,
    years: float = DEFAULT_YEARS,
    cadence_days: float = DEFAULT_CADENCE_DAYS,
    raw_manifest: str | Path | None = None,
    accel_source: str = "gradient",
    accel_certificate: bool = False,
    accel_harmonic: bool = False,
    accel_cache_dir: str | Path | None = None,
    cheap_fit_cfg: Any | None = None,
    deep_fit_cfg: Any | None = None,
) -> list[KeplerReducedDataset]:
    specs = build_default_orbit_specs(
        provider=str(provider),
        raw_manifest=raw_manifest,
        seed=int(seed),
        train_samples=int(train_samples),
        validation_samples=int(validation_samples),
        holdout_samples=int(holdout_samples),
    )
    datasets = [
        generate_kepler_dataset(
            spec,
            mu=mu,
            profile=str(profile),
            start_date=str(start_date),
            years=float(years),
            cadence_days=float(cadence_days),
            accel_source=str(accel_source),
            accel_certificate=bool(accel_certificate),
            accel_harmonic=bool(accel_harmonic),
            accel_cache_dir=accel_cache_dir,
            cheap_fit_cfg=cheap_fit_cfg,
            deep_fit_cfg=deep_fit_cfg,
        )
        for spec in specs
    ]
    datasets.sort(key=lambda ds: (_BASE._split_sort_key(ds.split), ds.orbit_id))
    return datasets


def load_kepler_manifest(data_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(data_dir) / "manifest.json"
    return _canonicalize_manifest_dict(json.loads(manifest_path.read_text(encoding="utf-8")))


def load_kepler_datasets_from_manifest(data_dir: str | Path) -> list[KeplerReducedDataset]:
    data_root = Path(data_dir)
    manifest = load_kepler_manifest(data_root)
    rows = list(manifest.get("orbits", []) or [])
    datasets: list[KeplerReducedDataset] = []
    for row in rows:
        combined_path = Path(str(row["combined_csv"]))
        arr = np.genfromtxt(combined_path, delimiter=",", names=True, dtype=np.float64)
        if arr.shape == ():
            arr = np.asarray([tuple(arr.tolist())], dtype=arr.dtype)
        datasets.append(
            KeplerReducedDataset(
                orbit_id=str(row["orbit_id"]),
                split=str(row["split"]),
                mu=float(row["mu"]),
                a=float(row["a"]),
                e=float(row["e"]),
                period=float(row["period"]),
                mean_motion=float(2.0 * math.pi / float(row["period"])),
                h=float(row["h"]),
                energy=float(row["energy"]),
                dynamic_range=float(row["dynamic_range"]),
                t=np.asarray(arr["t"], dtype=np.float64),
                mean_anomaly=np.asarray(arr["mean_anomaly"], dtype=np.float64),
                eccentric_anomaly=np.asarray(arr["eccentric_anomaly"], dtype=np.float64),
                x=np.asarray(arr["x"], dtype=np.float64),
                y=np.asarray(arr["y"], dtype=np.float64),
                vx=np.asarray(arr["vx"], dtype=np.float64),
                vy=np.asarray(arr["vy"], dtype=np.float64),
                ax=np.asarray(arr["ax"], dtype=np.float64),
                ay=np.asarray(arr["ay"], dtype=np.float64),
                r=np.asarray(arr["r"], dtype=np.float64),
                theta=np.asarray(arr["theta"], dtype=np.float64),
                rdot=np.asarray(arr["rdot"], dtype=np.float64),
                omega=np.asarray(arr["omega"], dtype=np.float64),
                rddot=np.asarray(arr["rddot"], dtype=np.float64),
            )
        )
    datasets.sort(key=lambda ds: (_BASE._split_sort_key(ds.split), ds.orbit_id))
    return datasets


def target_filepaths(
    data_dir: str | Path,
    target: str,
    *,
    splits: Sequence[str] | None = None,
) -> list[Path]:
    target_name = str(target)
    data_root = Path(data_dir)
    target_dir = data_root / target_name
    if not target_dir.exists():
        raise FileNotFoundError(f"missing generated target directory: {target_dir}")
    split_filter = None if splits is None else {str(item) for item in splits}
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = load_kepler_manifest(data_root)
        selected_paths = []
        key = f"{target_name}_csv"
        for row in list(manifest.get("orbits", []) or []):
            if not isinstance(row, dict):
                continue
            if split_filter is not None and str(row.get("split", "")) not in split_filter:
                continue
            csv_path = row.get(key, None)
            if csv_path is None:
                continue
            selected_paths.append(Path(str(csv_path)))
        if split_filter is not None or selected_paths:
            return selected_paths
    return sorted(target_dir.glob("*.csv"))


__all__ = list(getattr(_BASE, "__all__", []))
for _name in [
    "build_default_kepler_datasets",
    "build_default_orbit_specs",
    "build_generation_provenance",
    "cached_surrogate_rddot",
    "generate_kepler_dataset",
    "load_cached_surrogate_accels",
    "load_kepler_manifest",
    "load_kepler_datasets_from_manifest",
    "load_generation_provenance",
    "target_filepaths",
    "write_generated_artifacts",
]:
    if _name not in __all__:
        __all__.append(_name)
for _name in [
    "EphemerisBodySpec",
    "DEFAULT_SOLAR_MU_AU_DAY",
    "DEFAULT_START_DATE",
    "DEFAULT_YEARS",
    "DEFAULT_CADENCE_DAYS",
    "DEFAULT_PROVIDER",
    "DEFAULT_PROFILE",
    "DEFAULT_RAW_MANIFEST_PATH",
]:
    if _name not in __all__:
        __all__.append(_name)
