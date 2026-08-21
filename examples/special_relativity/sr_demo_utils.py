# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import (
    AbsNode,
    AddNode,
    ArgNode,
    AtomNode,
    ConjNode,
    ConstNode,
    CosNode,
    ExpNode,
    ImagNode,
    LogNode,
    MulNode,
    PowNode,
    RealNode,
    SinNode,
    ast_from_composite,
    build_composite_from_ast,
    collect_all_atoms,
    make_reuse_only_nn_factory,
    make_stage_a_nn_factory,
    sync_ast_num_segments_from_state_dict,
)
from nestynet_sr.sr_search.config import ModelHyperparams
from nestynet_sr.sr_search.model_builders import LeafBuilder, build_composite_ast
from nestynet_sr.sr_search.xcoord import XCoordSystem


def beta_to_regime_id(beta: float) -> str:
    value = int(round(abs(float(beta)) * 1000.0))
    sign = "p" if float(beta) >= 0.0 else "m"
    return f"beta_{sign}{value:04d}"


def lorentz_gamma(beta: float) -> float:
    beta_f = float(beta)
    if abs(beta_f) >= 1.0:
        raise ValueError(f"Lorentz boost requires |beta| < 1, got {beta_f!r}")
    return 1.0 / math.sqrt(1.0 - beta_f * beta_f)


def lorentz_boost_matrix(beta: float) -> np.ndarray:
    gamma = lorentz_gamma(beta)
    beta_f = float(beta)
    return np.asarray(
        [
            [gamma, -gamma * beta_f],
            [-gamma * beta_f, gamma],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class OperationalIntervalDataset:
    regime_id: str
    beta: float
    u: np.ndarray
    x: np.ndarray
    u_prime: np.ndarray
    x_prime: np.ndarray
    populations: tuple[str, ...]
    metadata: dict[str, Any]


def apply_lorentz_boost(
    u: Sequence[float] | np.ndarray,
    x: Sequence[float] | np.ndarray,
    *,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(u, dtype=np.float64)
    xv = np.asarray(x, dtype=np.float64)
    if uv.shape != xv.shape:
        raise ValueError("u and x must have the same shape")
    mat = lorentz_boost_matrix(beta)
    u_prime = float(mat[0, 0]) * uv + float(mat[0, 1]) * xv
    x_prime = float(mat[1, 0]) * uv + float(mat[1, 1]) * xv
    return u_prime, x_prime


def _sample_signs(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=int(n))


def _validate_population_fractions(
    timelike_fraction: float,
    spacelike_fraction: float,
    near_null_fraction: float,
) -> None:
    vals = [float(timelike_fraction), float(spacelike_fraction), float(near_null_fraction)]
    if any(v < 0.0 for v in vals):
        raise ValueError("population fractions must be non-negative")
    total = float(sum(vals))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(
            "population fractions must sum to 1.0; "
            f"got timelike={vals[0]:.6f}, spacelike={vals[1]:.6f}, near_null={vals[2]:.6f}"
        )


def generate_operational_interval_dataset(
    beta: float,
    *,
    n_samples: int = 2048,
    seed: int | None = None,
    u_max: float = 10.0,
    x_max: float | None = None,
    timelike_fraction: float = 0.35,
    spacelike_fraction: float = 0.35,
    near_null_fraction: float = 0.30,
    near_null_width: float = 0.03,
    noise_std: float = 0.0,
) -> OperationalIntervalDataset:
    _validate_population_fractions(timelike_fraction, spacelike_fraction, near_null_fraction)
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive")
    if float(u_max) <= 0.0:
        raise ValueError("u_max must be positive")
    if x_max is None:
        x_max = float(u_max)
    if float(x_max) <= 0.0:
        raise ValueError("x_max must be positive")
    if float(near_null_width) < 0.0:
        raise ValueError("near_null_width must be non-negative")
    if float(noise_std) < 0.0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    n_total = int(n_samples)
    n_timelike = int(round(float(timelike_fraction) * n_total))
    n_spacelike = int(round(float(spacelike_fraction) * n_total))
    n_near_null = n_total - n_timelike - n_spacelike

    parts_u: list[np.ndarray] = []
    parts_x: list[np.ndarray] = []
    labels: list[str] = []

    if n_timelike > 0:
        base = rng.uniform(0.25, float(u_max), size=n_timelike)
        u = _sample_signs(rng, n_timelike) * base
        x = rng.uniform(-0.85, 0.85, size=n_timelike) * base
        parts_u.append(u)
        parts_x.append(x)
        labels.extend(["timelike"] * n_timelike)

    if n_spacelike > 0:
        base = rng.uniform(0.25, float(x_max), size=n_spacelike)
        x = _sample_signs(rng, n_spacelike) * base
        u = rng.uniform(-0.85, 0.85, size=n_spacelike) * base
        parts_u.append(u)
        parts_x.append(x)
        labels.extend(["spacelike"] * n_spacelike)

    if n_near_null > 0:
        base = rng.uniform(0.25, max(float(u_max), float(x_max)), size=n_near_null)
        u = _sample_signs(rng, n_near_null) * base
        cone_side = _sample_signs(rng, n_near_null)
        eps = rng.uniform(-float(near_null_width), float(near_null_width), size=n_near_null)
        x = cone_side * base * (1.0 + eps)
        parts_u.append(u)
        parts_x.append(x)
        labels.extend(["near_null"] * n_near_null)

    u_all = np.concatenate(parts_u, axis=0).astype(np.float64, copy=False)
    x_all = np.concatenate(parts_x, axis=0).astype(np.float64, copy=False)
    perm = rng.permutation(n_total)
    u_all = u_all[perm]
    x_all = x_all[perm]
    populations = tuple(labels[idx] for idx in perm.tolist())

    u_prime, x_prime = apply_lorentz_boost(u_all, x_all, beta=beta)
    if float(noise_std) > 0.0:
        u_prime = u_prime + rng.normal(0.0, float(noise_std), size=u_prime.shape)
        x_prime = x_prime + rng.normal(0.0, float(noise_std), size=x_prime.shape)

    regime_id = beta_to_regime_id(beta)
    metadata = {
        "beta": float(beta),
        "gamma": float(lorentz_gamma(beta)),
        "n_samples": int(n_total),
        "u_max": float(u_max),
        "x_max": float(x_max),
        "timelike_fraction": float(timelike_fraction),
        "spacelike_fraction": float(spacelike_fraction),
        "near_null_fraction": float(near_null_fraction),
        "near_null_width": float(near_null_width),
        "noise_std": float(noise_std),
    }
    return OperationalIntervalDataset(
        regime_id=regime_id,
        beta=float(beta),
        u=u_all,
        x=x_all,
        u_prime=np.asarray(u_prime, dtype=np.float64),
        x_prime=np.asarray(x_prime, dtype=np.float64),
        populations=populations,
        metadata=metadata,
    )


def fit_regime_affine_map(dataset: OperationalIntervalDataset) -> dict[str, Any]:
    design = np.column_stack([dataset.u, dataset.x]).astype(np.float64, copy=False)
    coeff_u, _, _, _ = np.linalg.lstsq(design, dataset.u_prime, rcond=None)
    coeff_x, _, _, _ = np.linalg.lstsq(design, dataset.x_prime, rcond=None)
    matrix = np.vstack([coeff_u, coeff_x]).astype(np.float64, copy=False)

    u_pred = float(coeff_u[0]) * dataset.u + float(coeff_u[1]) * dataset.x
    x_pred = float(coeff_x[0]) * dataset.u + float(coeff_x[1]) * dataset.x
    u_rmse = float(np.sqrt(np.mean((u_pred - dataset.u_prime) ** 2)))
    x_rmse = float(np.sqrt(np.mean((x_pred - dataset.x_prime) ** 2)))

    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    expected = lorentz_boost_matrix(dataset.beta)
    return {
        "regime_id": str(dataset.regime_id),
        "beta": float(dataset.beta),
        "matrix": matrix,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "u_rmse": u_rmse,
        "x_rmse": x_rmse,
        "symmetry_gap": float(max(abs(a - d), abs(b - c))),
        "determinant": float(np.linalg.det(matrix)),
        "expected_matrix": expected,
        "expected_max_abs_error": float(np.max(np.abs(matrix - expected))),
    }


def fit_regime_affine_maps(
    datasets: Sequence[OperationalIntervalDataset],
) -> list[dict[str, Any]]:
    return [fit_regime_affine_map(ds) for ds in list(datasets)]


def lift_lorentz_coefficient_laws(
    regime_fits: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fits = list(regime_fits)
    if not fits:
        raise ValueError("lift_lorentz_coefficient_laws requires at least one regime fit")

    betas = np.asarray([float(row["beta"]) for row in fits], dtype=np.float64)
    a = np.asarray([float(row["a"]) for row in fits], dtype=np.float64)
    b = np.asarray([float(row["b"]) for row in fits], dtype=np.float64)
    c = np.asarray([float(row["c"]) for row in fits], dtype=np.float64)
    d = np.asarray([float(row["d"]) for row in fits], dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_from_u = -b / a
        r_from_x = -c / d
        z_from_u = 1.0 / np.square(a)
        z_from_x = 1.0 / np.square(d)

    gamma_expected = np.asarray([lorentz_gamma(beta) for beta in betas], dtype=np.float64)
    z_expected = 1.0 - np.square(betas)
    beta_linear_design = np.column_stack([np.ones_like(betas), betas])
    z_even_design = np.column_stack([np.ones_like(betas), np.square(betas)])

    r_linear_coeffs, _, _, _ = np.linalg.lstsq(beta_linear_design, r_from_u, rcond=None)
    z_even_coeffs, _, _, _ = np.linalg.lstsq(z_even_design, z_from_u, rcond=None)

    return {
        "betas": betas,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "r_from_u": r_from_u,
        "r_from_x": r_from_x,
        "z_from_u": z_from_u,
        "z_from_x": z_from_x,
        "gamma_expected": gamma_expected,
        "z_expected": z_expected,
        "max_beta_residual": float(np.max(np.abs(r_from_u - betas))),
        "max_z_residual": float(np.max(np.abs(z_from_u - z_expected))),
        "symmetry_max_abs_error": float(max(np.max(np.abs(a - d)), np.max(np.abs(b - c)))),
        "gamma_max_abs_error": float(np.max(np.abs(a - gamma_expected))),
        "r_linear_coeffs": np.asarray(r_linear_coeffs, dtype=np.float64),
        "z_even_coeffs": np.asarray(z_even_coeffs, dtype=np.float64),
    }


def recover_interval_metric(
    regime_fits: Sequence[dict[str, Any]] | Sequence[np.ndarray],
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    basis = (
        np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    )

    for item in list(regime_fits):
        matrix = np.asarray(item["matrix"] if isinstance(item, dict) else item, dtype=np.float64)
        if matrix.shape != (2, 2):
            raise ValueError(f"expected 2x2 regime matrix, got shape {matrix.shape!r}")
        matrices.append(matrix)
        operator_cols = []
        for basis_matrix in basis:
            delta = matrix.T @ basis_matrix @ matrix - basis_matrix
            operator_cols.append(np.asarray([delta[0, 0], delta[0, 1], delta[1, 1]], dtype=np.float64))
        operator = np.column_stack(operator_cols)
        rows.append(operator)

    stacked = np.vstack(rows)
    _, singular_values, vh = np.linalg.svd(stacked, full_matrices=False)
    null_vec = np.asarray(vh[-1], dtype=np.float64)
    if abs(float(null_vec[0])) > 1.0e-12:
        null_vec = null_vec / float(null_vec[0])
    else:
        scale = float(np.max(np.abs(null_vec)))
        if scale <= 1.0e-12:
            raise ValueError("failed to recover a non-trivial invariant metric")
        null_vec = null_vec / scale
    metric = np.asarray(
        [
            [null_vec[0], null_vec[1]],
            [null_vec[1], null_vec[2]],
        ],
        dtype=np.float64,
    )
    if float(metric[0, 0]) < 0.0:
        metric = -metric
        null_vec = -null_vec

    per_matrix_errors = []
    for matrix in matrices:
        delta = matrix.T @ metric @ matrix - metric
        per_matrix_errors.append(float(np.linalg.norm(delta, ord="fro")))

    eigvals = np.linalg.eigvalsh(metric)
    quadratic_coeffs = {
        "u2": float(metric[0, 0]),
        "ux": float(2.0 * metric[0, 1]),
        "x2": float(metric[1, 1]),
    }
    return {
        "metric": metric,
        "quadratic_coeffs": quadratic_coeffs,
        "singular_values": np.asarray(singular_values, dtype=np.float64),
        "null_vector": np.asarray(null_vec, dtype=np.float64),
        "per_matrix_errors": np.asarray(per_matrix_errors, dtype=np.float64),
        "max_preservation_error": float(max(per_matrix_errors) if per_matrix_errors else 0.0),
        "determinant": float(np.linalg.det(metric)),
        "eigenvalues": np.asarray(eigvals, dtype=np.float64),
        "is_indefinite": bool(np.prod(eigvals) < 0.0),
    }


def analyze_operational_boost_family(
    datasets: Sequence[OperationalIntervalDataset],
) -> dict[str, Any]:
    regime_fits = fit_regime_affine_maps(datasets)
    coefficient_laws = lift_lorentz_coefficient_laws(regime_fits)
    metric = recover_interval_metric(regime_fits)
    return {
        "regime_fits": regime_fits,
        "coefficient_laws": coefficient_laws,
        "metric": metric,
    }


def load_classsr_payload(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stageb_payload(path: str | Path) -> dict[str, Any]:
    return pickle.loads(Path(path).read_bytes())


def _scalar_param_value(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            return None
        raw = raw[0]
    try:
        out = float(raw)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def build_classsr_param_map(
    class_sr_payload: dict[str, Any],
    *,
    dataset_index: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for tag, raw in dict(class_sr_payload.get("class_params", {}) or {}).items():
        value = _scalar_param_value(raw)
        if value is not None:
            out[str(tag)] = float(value)
    experiment_params = list(class_sr_payload.get("experiment_params", []) or [])
    if int(dataset_index) < len(experiment_params):
        for tag, raw in dict(experiment_params[int(dataset_index)] or {}).items():
            value = _scalar_param_value(raw)
            if value is not None:
                out[str(tag)] = float(value)
    return out


def evaluate_ast_numeric(
    node: Any,
    *,
    x_values: Sequence[float],
    param_values: dict[str, float] | None = None,
) -> float:
    params = dict(param_values or {})

    if isinstance(node, ConstNode):
        return float(node.value)
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "") or "").lower()
        tag = getattr(node, "tag", None)
        kwargs = dict(getattr(node, "kwargs", {}) or {})
        if kind == "var":
            var_idxs = tuple(getattr(node, "var_idxs", ()) or ())
            if len(var_idxs) != 1:
                raise ValueError(f"expected scalar var leaf, got var_idxs={var_idxs!r}")
            idx = int(var_idxs[0])
            return float(x_values[idx])
        if kind in {"scale", "free_const"}:
            if tag is not None and str(tag) in params:
                return float(params[str(tag)])
            fallback = kwargs.get("value", kwargs.get("init", 1.0))
            return float(fallback)
        if kind == "fixed_const":
            if tag is not None and str(tag) in params:
                return float(params[str(tag)])
            return float(kwargs.get("value", 1.0))
        raise ValueError(f"unsupported AtomNode kind {kind!r} in SR demo evaluator")
    if isinstance(node, AddNode):
        return evaluate_ast_numeric(node.left, x_values=x_values, param_values=params) + evaluate_ast_numeric(
            node.right,
            x_values=x_values,
            param_values=params,
        )
    if isinstance(node, MulNode):
        return evaluate_ast_numeric(node.left, x_values=x_values, param_values=params) * evaluate_ast_numeric(
            node.right,
            x_values=x_values,
            param_values=params,
        )
    if isinstance(node, PowNode):
        base = evaluate_ast_numeric(node.base, x_values=x_values, param_values=params)
        return float(base ** float(node.exponent))
    if isinstance(node, SinNode):
        return float(math.sin(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, CosNode):
        return float(math.cos(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, ExpNode):
        return float(math.exp(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, LogNode):
        return float(math.log(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, AbsNode):
        return float(abs(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, RealNode):
        return float(np.real(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, ImagNode):
        return float(np.imag(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, ArgNode):
        return float(np.angle(evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)))
    if isinstance(node, ConjNode):
        value = evaluate_ast_numeric(node.arg, x_values=x_values, param_values=params)
        return float(np.conj(value).real)
    raise ValueError(f"unsupported AST node type {type(node)!r} in SR demo evaluator")


def extract_affine_coefficients_from_predictor(
    predictor,
    *,
    num_vars: int = 2,
    probe_points: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    n = int(num_vars)
    zero = np.zeros((n,), dtype=np.float64)
    intercept = float(predictor(zero))
    coeffs = []
    for axis in range(n):
        point = np.zeros((n,), dtype=np.float64)
        point[axis] = 1.0
        coeffs.append(float(predictor(point)) - intercept)
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)

    if probe_points is None:
        probe_points = (
            (-1.5, -0.75),
            (-0.5, 0.25),
            (0.25, -1.25),
            (1.0, 1.0),
            (2.0, -0.5),
        )

    residuals = []
    for raw in probe_points:
        point = np.asarray(raw, dtype=np.float64)
        if point.shape != (n,):
            raise ValueError(f"probe point has shape {point.shape!r}, expected {(n,)!r}")
        truth = float(predictor(point))
        pred = float(intercept + np.dot(coeffs_arr, point))
        residuals.append(truth - pred)
    residuals_arr = np.asarray(residuals, dtype=np.float64)

    return {
        "intercept": float(intercept),
        "coeffs": coeffs_arr,
        "probe_points": np.asarray(probe_points, dtype=np.float64),
        "probe_residuals": residuals_arr,
        "probe_rmse": float(np.sqrt(np.mean(np.square(residuals_arr)))),
        "max_probe_abs_residual": float(np.max(np.abs(residuals_arr))) if residuals_arr.size else 0.0,
    }


def extract_affine_coefficients_from_ast(
    root: Any,
    *,
    param_values: dict[str, float] | None = None,
    num_vars: int = 2,
    probe_points: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    return extract_affine_coefficients_from_predictor(
        lambda point: evaluate_ast_numeric(root, x_values=point, param_values=param_values),
        num_vars=int(num_vars),
        probe_points=probe_points,
    )


def _tag_to_leafidx(root: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, atom in enumerate(collect_all_atoms(root)):
        tag = str(getattr(atom, "tag", None) or f"leaf{idx}")
        out.setdefault(tag, idx)
    return out


def _set_leaf_params_from_flat(leaf: torch.nn.Module, raw_values: Sequence[float]) -> None:
    flat = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    offset = 0
    with torch.no_grad():
        for param in leaf.parameters():
            n = int(param.numel())
            if offset + n > int(flat.size):
                raise ValueError(
                    f"parameter vector too short for leaf {type(leaf).__name__}: "
                    f"needed at least {offset + n}, got {flat.size}"
                )
            view = flat[offset:offset + n].reshape(tuple(param.shape))
            tensor = torch.as_tensor(view, dtype=param.dtype, device=param.device)
            param.copy_(tensor)
            offset += n
    if offset != int(flat.size):
        raise ValueError(
            f"parameter vector size mismatch for leaf {type(leaf).__name__}: "
            f"consumed {offset}, got {flat.size}"
        )


def _candidate_model_ckpt_paths(class_sr_json_path: str | Path) -> list[Path]:
    path = Path(class_sr_json_path).resolve()
    stem = path.name
    if stem.endswith("_classSR.json"):
        stem = stem[: -len("_classSR.json")]
    candidates = [
        path.parent / f"{stem}.identity.mod",
        path.parent / f"{stem}.mod",
    ]
    try:
        repo_root = path.parents[3]
        candidates.extend(
            [
                repo_root / "models" / f"{stem}.identity.mod",
                repo_root / "models" / f"{stem}.mod",
            ]
        )
    except IndexError:
        pass
    out = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def find_model_ckpt_path(class_sr_json_path: str | Path) -> Path | None:
    for candidate in _candidate_model_ckpt_paths(class_sr_json_path):
        if candidate.exists():
            return candidate
    return None


def load_stagea_model_template(
    model_ckpt_path: str | Path,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    device_obj = torch.device("cpu") if device is None else torch.device(device)
    payload = torch.load(Path(model_ckpt_path), map_location=device_obj, weights_only=False)
    ast_mod = payload.get("ast", None)
    state_dict = payload.get("model_state_dict", None)
    if ast_mod is None or state_dict is None:
        raise ValueError(f"model checkpoint missing ast/model_state_dict: {model_ckpt_path}")

    sync_ast_num_segments_from_state_dict(ast_mod, state_dict)
    model_hp = ModelHyperparams()
    leaf_builder = LeafBuilder(model_hp, device_obj, dtype)
    model, _, _ = build_composite_ast(
        ast_mod,
        num_segments=None,
        dual_layer=None,
        leaf_builder=leaf_builder,
        device=device_obj,
        dtype=dtype,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    tagged_ast, reuse = ast_from_composite(model)
    return {
        "model": model,
        "tagged_ast": tagged_ast,
        "reuse": reuse,
        "leaf_builder": leaf_builder,
        "fresh_nn_factory": make_stage_a_nn_factory(leaf_builder),
        "device": device_obj,
        "dtype": dtype,
    }


def build_stageb_model_from_template(
    stageb_root: Any,
    *,
    template: dict[str, Any],
) -> torch.nn.Module:
    device_obj = template["device"]
    dtype = template["dtype"]
    reuse_build = {
        str(tag): copy.deepcopy(leaf).to(device=device_obj, dtype=dtype)
        for tag, leaf in dict(template["reuse"]).items()
    }
    nn_factory = make_reuse_only_nn_factory(
        device=device_obj,
        dtype=dtype,
        fresh_nn_factory=template["fresh_nn_factory"],
    )
    model = build_composite_from_ast(
        stageb_root,
        dtype=dtype,
        device=device_obj,
        nn_factory=nn_factory,
        reuse=reuse_build,
    )
    model.eval()
    return model


def apply_classsr_params_to_model(
    model: torch.nn.Module,
    *,
    root: Any,
    class_sr_payload: dict[str, Any],
    dataset_index: int,
) -> None:
    tag_to_leafidx = _tag_to_leafidx(root)
    class_params = dict(class_sr_payload.get("class_params", {}) or {})
    experiment_params = list(class_sr_payload.get("experiment_params", []) or [])
    params_by_tag: dict[str, Sequence[float]] = {}
    for tag, values in class_params.items():
        params_by_tag[str(tag)] = values
    if int(dataset_index) < len(experiment_params):
        for tag, values in dict(experiment_params[int(dataset_index)] or {}).items():
            params_by_tag[str(tag)] = values
    for tag, values in params_by_tag.items():
        leaf_idx = tag_to_leafidx.get(str(tag), None)
        if leaf_idx is None or leaf_idx >= len(model.leaf):
            continue
        _set_leaf_params_from_flat(model.leaf[leaf_idx], values)


def extract_affine_coefficients_from_model(
    model: torch.nn.Module,
    *,
    xcoord_system: XCoordSystem | None = None,
    num_vars: int = 2,
    probe_points: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    xcoords = xcoord_system

    def _predict(point: np.ndarray) -> float:
        x_arr = np.asarray(point, dtype=np.float64).reshape(1, int(num_vars))
        x_tensor = torch.as_tensor(x_arr, dtype=torch.float64)
        if xcoords is not None and not xcoords.is_identity():
            x_tensor = xcoords.apply_torch(x_tensor)
        with torch.no_grad():
            y = model(x_tensor)
        y_arr = np.asarray(y.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        if y_arr.size != 1:
            raise ValueError(f"expected scalar model output, got shape {tuple(y.shape)!r}")
        return float(y_arr[0])

    return extract_affine_coefficients_from_predictor(
        _predict,
        num_vars=int(num_vars),
        probe_points=probe_points,
    )


def extract_classsr_affine_rows(
    *,
    class_sr_json_path: str | Path,
    stageb_pkl_path: str | Path,
    model_ckpt_path: str | Path | None = None,
    num_vars: int = 2,
    probe_points: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    class_payload = load_classsr_payload(class_sr_json_path)
    stageb_payload = load_stageb_payload(stageb_pkl_path)
    root = stageb_payload["stageB_ast"]
    dataset_ids = list(stageb_payload.get("stageB_dataset_ids", []) or [])
    experiment_params = list(class_payload.get("experiment_params", []) or [])
    if not dataset_ids:
        dataset_ids = [f"dataset_{i}" for i in range(len(experiment_params))]

    xcoord_system = XCoordSystem.from_map(
        stageb_payload.get("x_transform_map", None),
        Nx_raw=int(num_vars),
    )
    use_model_bridge = False
    template = None
    resolved_model_ckpt_path = None if model_ckpt_path is None else Path(model_ckpt_path)
    if resolved_model_ckpt_path is None:
        resolved_model_ckpt_path = find_model_ckpt_path(class_sr_json_path)
    if resolved_model_ckpt_path is not None:
        try:
            template = load_stagea_model_template(resolved_model_ckpt_path)
            use_model_bridge = True
        except Exception:
            template = None
            use_model_bridge = False

    rows = []
    for idx, dataset_id in enumerate(dataset_ids):
        param_values = build_classsr_param_map(class_payload, dataset_index=idx)
        if use_model_bridge:
            model = build_stageb_model_from_template(root, template=template)
            apply_classsr_params_to_model(
                model,
                root=root,
                class_sr_payload=class_payload,
                dataset_index=idx,
            )
            affine = extract_affine_coefficients_from_model(
                model,
                xcoord_system=xcoord_system,
                num_vars=int(num_vars),
                probe_points=probe_points,
            )
            extraction_mode = "model"
        else:
            affine = extract_affine_coefficients_from_ast(
                root,
                param_values=param_values,
                num_vars=int(num_vars),
                probe_points=probe_points,
            )
            extraction_mode = "ast"
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "param_values": dict(param_values),
                "extraction_mode": extraction_mode,
                "intercept": float(affine["intercept"]),
                "coeffs": np.asarray(affine["coeffs"], dtype=np.float64),
                "probe_rmse": float(affine["probe_rmse"]),
                "max_probe_abs_residual": float(affine["max_probe_abs_residual"]),
            }
        )

    return {
        "class_sr_json_path": str(class_sr_json_path),
        "stageb_pkl_path": str(stageb_pkl_path),
        "model_ckpt_path": None if resolved_model_ckpt_path is None else str(resolved_model_ckpt_path),
        "dataset_ids": [str(item) for item in dataset_ids],
        "y_expr_str": stageb_payload.get("y_expr_str", None),
        "phi_expr_strs": stageb_payload.get("phi_expr_strs", None),
        "derived_invariants": list(class_payload.get("derived_invariants", []) or []),
        "rows": rows,
    }


def _canonical_symbolic_dataset_id(dataset_id: str) -> str:
    text = str(dataset_id)
    for prefix in ("uprime_", "xprime_"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def merge_symbolic_affine_tables(
    uprime_table: dict[str, Any],
    xprime_table: dict[str, Any],
    *,
    beta_by_dataset: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    x_rows_by_id = {
        _canonical_symbolic_dataset_id(str(row["dataset_id"])): row
        for row in list(xprime_table.get("rows", []) or [])
    }
    merged = []
    for u_row in list(uprime_table.get("rows", []) or []):
        source_dataset_id = str(u_row["dataset_id"])
        dataset_id = _canonical_symbolic_dataset_id(source_dataset_id)
        x_row = x_rows_by_id.get(dataset_id, None)
        if x_row is None:
            continue
        u_coeffs = np.asarray(u_row["coeffs"], dtype=np.float64)
        x_coeffs = np.asarray(x_row["coeffs"], dtype=np.float64)
        row = {
            "dataset_id": dataset_id,
            "u_dataset_id": source_dataset_id,
            "x_dataset_id": str(x_row["dataset_id"]),
            "beta": (
                None
                if beta_by_dataset is None or dataset_id not in beta_by_dataset
                else float(beta_by_dataset[dataset_id])
            ),
            "a": float(u_coeffs[0]),
            "b": float(u_coeffs[1]),
            "c": float(x_coeffs[0]),
            "d": float(x_coeffs[1]),
            "u_intercept": float(u_row["intercept"]),
            "x_intercept": float(x_row["intercept"]),
            "u_probe_rmse": float(u_row["probe_rmse"]),
            "x_probe_rmse": float(x_row["probe_rmse"]),
            "matrix": np.asarray(
                [
                    [float(u_coeffs[0]), float(u_coeffs[1])],
                    [float(x_coeffs[0]), float(x_coeffs[1])],
                ],
                dtype=np.float64,
            ),
        }
        merged.append(row)
    return merged


def analyze_symbolic_boost_family(
    merged_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(merged_rows)
    regime_fits = []
    for row in rows:
        if row.get("beta", None) is None:
            raise ValueError("analyze_symbolic_boost_family requires beta metadata per dataset")
        regime_fits.append(
            {
                "regime_id": str(row["dataset_id"]),
                "beta": float(row["beta"]),
                "matrix": np.asarray(row["matrix"], dtype=np.float64),
                "a": float(row["a"]),
                "b": float(row["b"]),
                "c": float(row["c"]),
                "d": float(row["d"]),
                "u_rmse": 0.0,
                "x_rmse": 0.0,
                "symmetry_gap": float(max(abs(float(row["a"]) - float(row["d"])), abs(float(row["b"]) - float(row["c"])))),
                "determinant": float(np.linalg.det(np.asarray(row["matrix"], dtype=np.float64))),
                "expected_matrix": lorentz_boost_matrix(float(row["beta"])),
                "expected_max_abs_error": float(
                    np.max(
                        np.abs(
                            np.asarray(row["matrix"], dtype=np.float64)
                            - lorentz_boost_matrix(float(row["beta"]))
                        )
                    )
                ),
            }
        )
    coefficient_laws = lift_lorentz_coefficient_laws(regime_fits)
    metric = recover_interval_metric(regime_fits)
    return {
        "regime_fits": regime_fits,
        "coefficient_laws": coefficient_laws,
        "metric": metric,
    }
