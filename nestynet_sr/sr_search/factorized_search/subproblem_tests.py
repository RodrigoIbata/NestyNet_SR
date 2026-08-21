# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .subproblem_spec import FamilyEvidence, canonicalize_family_hard_constraints


@dataclass(frozen=True)
class OuterFamilyBatterySpec:
    name: str
    min_improvement_ratio: float
    precheck_max_seeds: int
    requires_dimensionless_target: bool = True


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return float(out)


def _coerce_seed_nodes(seed_nodes: Sequence[Any] | None) -> tuple[Any, ...]:
    out: list[Any] = []
    for node in list(seed_nodes or ()):
        if node is None:
            continue
        out.append(node)
    return tuple(out)


def normalize_family_battery_mode(mode: str | None) -> str:
    token = str(mode or "outer").strip().lower()
    if token == "expanded":
        return "expanded"
    return "outer"


def _ensure_matrix(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        out = value.to(dtype=torch.float64)
    else:
        try:
            out = torch.as_tensor(value, dtype=torch.float64)
        except Exception:
            return None
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2 or int(out.shape[0]) <= 0:
        return None
    if not torch.isfinite(out).all():
        return None
    return out


def _ensure_col(value: Any) -> torch.Tensor | None:
    out = _ensure_matrix(value)
    if out is None:
        return None
    if int(out.shape[1]) == 1:
        return out
    return out.mean(dim=1, keepdim=True)


def _stack_xy(
    *,
    x_fit: Any,
    t_fit: Any,
    x_probe: Any | None = None,
    t_probe: Any | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    fit_x = _ensure_matrix(x_fit)
    fit_y = _ensure_col(t_fit)
    if fit_x is not None and fit_y is not None and int(fit_x.shape[0]) == int(fit_y.shape[0]):
        xs.append(fit_x)
        ys.append(fit_y)
    probe_x = _ensure_matrix(x_probe)
    probe_y = _ensure_col(t_probe)
    if probe_x is not None and probe_y is not None and int(probe_x.shape[0]) == int(probe_y.shape[0]):
        xs.append(probe_x)
        ys.append(probe_y)
    if not xs or not ys:
        return None, None
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _stack_x_feature(
    *,
    x_fit: Any,
    feature_fit: Any,
    x_probe: Any | None = None,
    feature_probe: Any | None = None,
    expected_cols: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    xs: list[torch.Tensor] = []
    features: list[torch.Tensor] = []

    def _append(x_raw: Any, feature_raw: Any) -> None:
        xx = _ensure_matrix(x_raw)
        ff = _ensure_matrix(feature_raw)
        if xx is None or ff is None:
            return
        if int(xx.shape[0]) != int(ff.shape[0]):
            return
        if expected_cols is not None and int(ff.shape[1]) != int(expected_cols):
            return
        xs.append(xx)
        features.append(ff)

    _append(x_fit, feature_fit)
    _append(x_probe, feature_probe)
    if not xs or not features:
        return None, None
    return torch.cat(xs, dim=0), torch.cat(features, dim=0)


def _pearson_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    xx = x.reshape(-1).to(dtype=torch.float64)
    yy = y.reshape(-1).to(dtype=torch.float64)
    if int(xx.numel()) != int(yy.numel()) or int(xx.numel()) < 2:
        return 0.0
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    xx_var = float((xx * xx).mean().item())
    yy_var = float((yy * yy).mean().item())
    if xx_var <= 1.0e-20 or yy_var <= 1.0e-20:
        return 0.0
    cov = float((xx * yy).mean().item())
    return float(abs(cov) / math.sqrt(xx_var * yy_var))


def _family_evidence(
    family_name: str,
    *,
    family_scores: Mapping[str, float],
    status: str,
    hard_constraints: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    seed_nodes: Sequence[Any] | None = None,
    target_dim: Any = None,
    active_vars: Sequence[int] | None = None,
    wrappers_left: int | None = None,
    recursion_level: int | None = None,
    direction: str | None = None,
    target_mode: str | None = None,
    target_mapping_kind: str | None = None,
    dimensionless_target_required: bool | None = None,
    target_dim_ok: bool | None = None,
    domain_masks: Mapping[str, Any] | None = None,
    regime_metadata: Mapping[str, Any] | None = None,
) -> FamilyEvidence:
    hard = {
        "status": str(status),
        "should_run": False,
        "advisory_only": True,
    }
    hard.update(dict(hard_constraints or {}))
    meta = {
        "status": str(status),
        "should_run": False,
        "advisory_only": True,
    }
    meta.update(dict(metadata or {}))
    return FamilyEvidence(
        family_scores={
            str(key): float(value)
            for key, value in dict(family_scores or {}).items()
        },
        hard_constraints=canonicalize_family_hard_constraints(
            family_name,
            hard,
            status=str(status),
            should_run=False,
            advisory_only=True,
            target_dim=target_dim,
            active_vars=active_vars,
            wrappers_left=wrappers_left,
            recursion_level=recursion_level,
            direction=direction,
            target_mode=target_mode,
            target_mapping_kind=target_mapping_kind,
            dimensionless_target_required=dimensionless_target_required,
            target_dim_ok=target_dim_ok,
            domain_masks=domain_masks,
            regime_metadata=regime_metadata,
        ),
        seed_nodes=_coerce_seed_nodes(seed_nodes),
        metadata=meta,
    )


def _primary_axis(x: torch.Tensor) -> int:
    if int(x.shape[1]) <= 1:
        return 0
    spread = x.std(dim=0, unbiased=False)
    if spread.numel() <= 0 or not torch.isfinite(spread).any():
        return 0
    return int(torch.argmax(torch.nan_to_num(spread, nan=0.0)).item())


def _sorted_axis_data(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    axis = _primary_axis(x)
    coords = x[:, axis]
    order = torch.argsort(coords)
    return axis, coords[order], y.reshape(-1)[order]


def _sort_by_axis(
    x: torch.Tensor,
    values: torch.Tensor,
    *,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords = x[:, int(axis)]
    order = torch.argsort(coords)
    return coords[order], values[order]


def _finite_slopes(coords: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if int(coords.numel()) < 2 or int(values.numel()) < 2:
        return torch.zeros((0,), dtype=values.dtype)
    dx = coords[1:] - coords[:-1]
    dy = values[1:] - values[:-1]
    mask = dx.abs() > 1.0e-12
    if not bool(mask.any()):
        return torch.zeros((0,), dtype=values.dtype)
    return dy[mask] / dx[mask]


def _sign_changes(values: torch.Tensor) -> int:
    raw = values.detach().cpu().reshape(-1).tolist()
    out = 0
    last = 0
    for item in raw:
        if not math.isfinite(float(item)) or abs(float(item)) <= 1.0e-12:
            continue
        sign = 1 if float(item) > 0.0 else -1
        if last != 0 and sign != last:
            out += 1
        last = sign
    return int(out)


def _crossing_count(values: torch.Tensor) -> int:
    if int(values.numel()) < 2:
        return 0
    a = values[:-1]
    b = values[1:]
    return int(((a * b) < 0.0).to(dtype=torch.int64).sum().item())


def _lstsq_mse(design: torch.Tensor, target: torch.Tensor) -> float | None:
    try:
        solution = torch.linalg.lstsq(design, target).solution
        pred = design @ solution
        mse = float(torch.mean((pred - target) ** 2).item())
    except Exception:
        return None
    return mse if math.isfinite(mse) else None


def _mirror_residual(values: torch.Tensor, *, sign: float) -> float:
    flat = values.reshape(-1).to(dtype=torch.float64)
    if int(flat.numel()) <= 0:
        return 1.0
    rev = torch.flip(flat, dims=[0])
    scale = max(1.0e-12, float(flat.abs().mean().item()))
    resid = torch.mean(torch.abs(flat - (float(sign) * rev)))
    out = float((resid / scale).item())
    return out if math.isfinite(out) else 1.0


def _mean_square_by_col(values: torch.Tensor) -> torch.Tensor:
    vv = values.to(dtype=torch.float64)
    if int(vv.numel()) <= 0:
        return torch.zeros((0,), dtype=torch.float64)
    return torch.mean(vv * vv, dim=0)


def _spike_ratio(values: torch.Tensor) -> float:
    flat = values.reshape(-1).abs().to(dtype=torch.float64)
    if int(flat.numel()) <= 0:
        return 1.0
    mean_abs = float(flat.mean().item())
    if mean_abs <= 1.0e-12:
        return 1.0
    return float(flat.max().item() / mean_abs)


def _quantile_tail_mask(values: torch.Tensor, *, q: float, min_points: int) -> torch.Tensor:
    flat = values.reshape(-1).to(dtype=torch.float64)
    if int(flat.numel()) <= 0:
        return torch.zeros((0,), dtype=torch.bool)
    if int(flat.numel()) <= int(min_points):
        return torch.ones_like(flat, dtype=torch.bool)
    try:
        threshold = torch.quantile(flat, float(q))
    except Exception:
        threshold = flat.median()
    mask = flat >= threshold
    if int(mask.to(dtype=torch.int64).sum().item()) >= int(min_points):
        return mask
    topk = min(int(min_points), int(flat.numel()))
    ranked = torch.argsort(flat, descending=True)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[ranked[:topk]] = True
    return mask


def _linear_fit_stats(x: torch.Tensor, y: torch.Tensor) -> dict[str, float] | None:
    xx = x.reshape(-1, 1).to(dtype=torch.float64)
    yy = y.reshape(-1, 1).to(dtype=torch.float64)
    if int(xx.shape[0]) != int(yy.shape[0]) or int(xx.shape[0]) < 2:
        return None
    design = torch.cat([torch.ones_like(xx), xx], dim=1)
    try:
        solution = torch.linalg.lstsq(design, yy).solution
        pred = design @ solution
    except Exception:
        return None
    resid = yy - pred
    mse = float(torch.mean(resid * resid).item())
    var = float(torch.mean((yy - yy.mean()) ** 2).item())
    r2 = 0.0 if var <= 1.0e-20 else float(max(0.0, 1.0 - (mse / max(var, 1.0e-20))))
    return {
        "bias": float(solution[0, 0].item()),
        "slope": float(solution[1, 0].item()),
        "mse": float(mse),
        "r2": float(r2),
    }


def _combine_add_terms(terms: Sequence[tuple]) -> tuple | None:
    nodes = [node for node in list(terms or ()) if isinstance(node, tuple) and node]
    if not nodes:
        return None
    out = nodes[0]
    for node in nodes[1:]:
        out = ("add", out, node)
    return out


def _scale_var_node(coeff: float, var_idx: int) -> tuple | None:
    if not math.isfinite(float(coeff)) or abs(float(coeff)) <= 1.0e-12:
        return None
    if abs(float(coeff) - 1.0) <= 1.0e-12:
        return ("var", int(var_idx))
    if abs(float(coeff) + 1.0) <= 1.0e-12:
        return ("neg", ("var", int(var_idx)))
    return ("mul", ("const", float(coeff)), ("var", int(var_idx)))


def _quantize_direction_coeff(value: float) -> float:
    vv = float(value)
    av = abs(vv)
    if av >= 0.75:
        base = 1.0
    elif av >= 0.35:
        base = 0.5
    else:
        return 0.0
    return float(math.copysign(base, vv))


def _coordinate_seed_nodes(
    *,
    active_vars: Sequence[int],
    direction: torch.Tensor | None,
) -> tuple[Any, ...]:
    if direction is None or int(direction.numel()) < 2:
        return tuple()
    idx = [int(v) for v in tuple(active_vars or ())]
    if len(idx) < 2:
        return tuple()
    abs_dir = torch.abs(direction)
    ranked = torch.argsort(abs_dir, descending=True).tolist()
    keep = [int(pos) for pos in ranked[: min(3, len(ranked))]]
    if len(keep) < 2:
        return tuple()
    max_abs = float(max(abs(float(direction[pos].item())) for pos in keep))
    if max_abs <= 1.0e-12:
        return tuple()
    quantized_terms: list[tuple] = []
    for pos in keep:
        coeff = _quantize_direction_coeff(float(direction[pos].item()) / max_abs)
        node = _scale_var_node(coeff, idx[pos])
        if node is not None:
            quantized_terms.append(node)
    out: list[Any] = []
    if len(quantized_terms) >= 2:
        combo = _combine_add_terms(quantized_terms)
        if combo is not None:
            out.append(combo)
    top_i = idx[int(keep[0])]
    top_j = idx[int(keep[1])]
    sign_i = 1.0 if float(direction[int(keep[0])].item()) >= 0.0 else -1.0
    sign_j = 1.0 if float(direction[int(keep[1])].item()) >= 0.0 else -1.0
    if sign_i * sign_j >= 0.0:
        out.append(("add", ("var", int(top_i)), ("var", int(top_j))))
    else:
        out.append(("sub", ("var", int(top_i)), ("var", int(top_j))))
    if len(keep) >= 3:
        top_k = idx[int(keep[2])]
        out.append(("add", ("add", ("var", int(top_i)), ("var", int(top_j))), ("var", int(top_k))))
    dedup: list[Any] = []
    seen: set[str] = set()
    for node in out:
        token = str(node)
        if token in seen:
            continue
        seen.add(token)
        dedup.append(node)
    return tuple(dedup)


def _normalize_regime_constant_rows(value: Any) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for regime_id, payload in dict(value or {}).items():
        if not isinstance(payload, Mapping):
            continue
        numeric: dict[str, float] = {}
        for name, raw in dict(payload).items():
            scalar = _safe_float(raw, None)
            if scalar is not None:
                numeric[str(name)] = float(scalar)
        if numeric:
            rows[str(regime_id)] = numeric
    return rows


def _parameter_stability_from_constants(
    local_constants_by_experiment: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    rows = _normalize_regime_constant_rows(local_constants_by_experiment)
    if len(rows) < 2:
        return None
    grouped: dict[str, list[float]] = {}
    for payload in rows.values():
        for name, value in payload.items():
            grouped.setdefault(str(name), []).append(float(value))
    if not grouped:
        return None
    parameter_cvs: dict[str, float] = {}
    parameter_sample_counts: dict[str, int] = {}
    cvs: list[float] = []
    for name, values in grouped.items():
        if len(values) <= 0:
            continue
        mean = float(sum(values) / len(values))
        centered = [float(value) - mean for value in values]
        var = float(sum(value * value for value in centered) / len(centered))
        std = math.sqrt(max(0.0, var))
        cv = float(std / max(1.0e-12, abs(mean)))
        parameter_cvs[str(name)] = float(cv)
        parameter_sample_counts[str(name)] = int(len(values))
        cvs.append(float(cv))
    if not parameter_cvs:
        return None
    mean_cv = float(sum(cvs) / len(cvs))
    score = float(1.0 / (1.0 + mean_cv))
    return {
        "passed": None,
        "score": float(score),
        "details": {
            "mean_cv": float(mean_cv),
            "parameter_cvs": parameter_cvs,
            "parameter_sample_counts": parameter_sample_counts,
            "n_parameters": int(len(parameter_cvs)),
            "regime_count": int(len(rows)),
        },
    }


def _build_regime_lift_evidence(
    *,
    regime_metadata: Mapping[str, Any] | None,
    target_dim: Any,
    active_vars: Sequence[int] | None,
    wrappers_left: int | None,
    recursion_level: int | None,
    direction: str | None,
    target_mode: str | None,
    target_mapping_kind: str | None,
) -> FamilyEvidence:
    regime = dict(regime_metadata or {})
    if not regime:
        return _family_evidence(
            "regime_lift",
            family_scores={"regime_lift": 0.0},
            status="missing_regime_metadata",
            target_dim=target_dim,
            active_vars=active_vars,
            wrappers_left=wrappers_left,
            recursion_level=recursion_level,
            direction=direction,
            target_mode=target_mode,
            target_mapping_kind=target_mapping_kind,
            regime_metadata=regime_metadata,
        )
    regime_ids = [
        str(item)
        for item in list(regime.get("regime_ids", regime.get("dataset_ids", ())) or ())
        if str(item)
    ]
    trigger_mean_cv = _safe_float(regime.get("trigger_mean_cv", 0.5), 0.5) or 0.5
    stability = dict(regime.get("parameter_stability", {}) or {})
    if not stability:
        inferred = _parameter_stability_from_constants(
            regime.get("local_constants_by_experiment", None)
        )
        if inferred is not None:
            stability = inferred
    if not stability:
        status = "insufficient_regime_signal" if len(regime_ids) >= 2 else "insufficient_regimes"
        return _family_evidence(
            "regime_lift",
            family_scores={"regime_lift": 0.0},
            status=status,
            hard_constraints={
                "regime_count": int(len(regime_ids)),
                "trigger_mean_cv": float(trigger_mean_cv),
            },
            target_dim=target_dim,
            active_vars=active_vars,
            wrappers_left=wrappers_left,
            recursion_level=recursion_level,
            direction=direction,
            target_mode=target_mode,
            target_mapping_kind=target_mapping_kind,
            regime_metadata=regime_metadata,
        )
    details = dict(stability.get("details", {}) or {})
    parameter_cvs = {
        str(key): float(value)
        for key, value in dict(details.get("parameter_cvs", {}) or {}).items()
        if _safe_float(value, None) is not None
    }
    top_constant_name = ""
    top_constant_cv = 0.0
    if parameter_cvs:
        top_constant_name = max(
            parameter_cvs.keys(),
            key=lambda key: (float(parameter_cvs[key]), str(key)),
        )
        top_constant_cv = float(parameter_cvs[top_constant_name])
    mean_cv = _safe_float(details.get("mean_cv", None), None)
    if mean_cv is None and parameter_cvs:
        mean_cv = float(sum(parameter_cvs.values()) / len(parameter_cvs))
    score = _safe_float(stability.get("score", None), None)
    drift_score = (
        float(1.0 - score)
        if score is not None
        else (0.0 if mean_cv is None else float(mean_cv / (1.0 + mean_cv)))
    )
    passed = stability.get("passed", None)
    if passed is False or (mean_cv is not None and float(mean_cv) > float(trigger_mean_cv)):
        status = "drifting_constants"
    else:
        status = "stable_constants"
    return _family_evidence(
        "regime_lift",
        family_scores={"regime_lift": float(max(0.0, drift_score))},
        status=status,
        hard_constraints={
            "regime_count": int(details.get("regime_count", len(regime_ids))),
            "trigger_mean_cv": float(trigger_mean_cv),
            "parameter_stability_passed": passed,
            "parameter_stability_score": score,
            "mean_cv": mean_cv,
            "top_constant_name": str(top_constant_name),
            "top_constant_cv": float(top_constant_cv),
            "parameter_cvs": parameter_cvs,
        },
        metadata={
            "parameter_stability": stability,
            "top_constant_name": str(top_constant_name),
            "top_constant_cv": float(top_constant_cv),
        },
        target_dim=target_dim,
        active_vars=active_vars,
        wrappers_left=wrappers_left,
        recursion_level=recursion_level,
        direction=direction,
        target_mode=target_mode,
        target_mapping_kind=target_mapping_kind,
        regime_metadata=regime_metadata,
    )


def build_expanded_family_evidence_bundle(
    *,
    x_fit: Any,
    t_fit: Any,
    x_probe: Any | None = None,
    t_probe: Any | None = None,
    grad_fit: Any | None = None,
    grad_probe: Any | None = None,
    d2_fit: Any | None = None,
    d2_probe: Any | None = None,
    fit_jet_source: str | None = None,
    probe_jet_source: str | None = None,
    fit_jet_fallback_used: bool = False,
    probe_jet_fallback_used: bool = False,
    target_dim: Any = None,
    active_vars: Sequence[int] | None = None,
    wrappers_left: int | None = None,
    recursion_level: int | None = None,
    direction: str | None = None,
    target_mode: str | None = None,
    target_mapping_kind: str | None = None,
    regime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, FamilyEvidence]:
    regime_lift = _build_regime_lift_evidence(
        regime_metadata=regime_metadata,
        target_dim=target_dim,
        active_vars=active_vars,
        wrappers_left=wrappers_left,
        recursion_level=recursion_level,
        direction=direction,
        target_mode=target_mode,
        target_mapping_kind=target_mapping_kind,
    )
    x, y = _stack_xy(x_fit=x_fit, t_fit=t_fit, x_probe=x_probe, t_probe=t_probe)
    if x is None or y is None or int(x.shape[0]) < 4:
        missing = _family_evidence(
            "symmetry",
            family_scores={"symmetry": 0.0},
            status="insufficient_points",
            target_dim=target_dim,
            active_vars=active_vars,
            wrappers_left=wrappers_left,
            recursion_level=recursion_level,
            direction=direction,
            target_mode=target_mode,
            target_mapping_kind=target_mapping_kind,
            regime_metadata=regime_metadata,
        )
        return {
            "symmetry": missing,
            "separability": _family_evidence(
                "separability",
                family_scores={"separability": 0.0},
                status="insufficient_points",
                target_dim=target_dim,
                active_vars=active_vars,
                wrappers_left=wrappers_left,
                recursion_level=recursion_level,
                direction=direction,
                target_mode=target_mode,
                target_mapping_kind=target_mapping_kind,
                regime_metadata=regime_metadata,
            ),
            "low_rank_dependence": _family_evidence(
                "low_rank_dependence",
                family_scores={"low_rank_dependence": 0.0},
                status="insufficient_points",
                target_dim=target_dim,
                active_vars=active_vars,
                wrappers_left=wrappers_left,
                recursion_level=recursion_level,
                direction=direction,
                target_mode=target_mode,
                target_mapping_kind=target_mapping_kind,
                regime_metadata=regime_metadata,
            ),
            "domain_hazard": _family_evidence(
                "domain_hazard",
                family_scores={"domain_hazard": 0.0},
                status="insufficient_points",
                target_dim=target_dim,
                active_vars=active_vars,
                wrappers_left=wrappers_left,
                recursion_level=recursion_level,
                direction=direction,
                target_mode=target_mode,
                target_mapping_kind=target_mapping_kind,
                regime_metadata=regime_metadata,
            ),
            "asymptotic_monomial": _family_evidence(
                "asymptotic_monomial",
                family_scores={"asymptotic_monomial": 0.0},
                status="insufficient_points",
                target_dim=target_dim,
                active_vars=active_vars,
                wrappers_left=wrappers_left,
                recursion_level=recursion_level,
                direction=direction,
                target_mode=target_mode,
                target_mapping_kind=target_mapping_kind,
                regime_metadata=regime_metadata,
            ),
            "branch_structure": _family_evidence(
                "branch_structure",
                family_scores={"branch_structure": 0.0},
                status="insufficient_points",
                target_dim=target_dim,
                active_vars=active_vars,
                wrappers_left=wrappers_left,
                recursion_level=recursion_level,
                direction=direction,
                target_mode=target_mode,
                target_mapping_kind=target_mapping_kind,
                regime_metadata=regime_metadata,
            ),
            "coordinate_invariant": _family_evidence(
                "coordinate_invariant",
                family_scores={"coordinate_invariant": 0.0},
                status="insufficient_points",
                target_dim=target_dim,
                active_vars=active_vars,
                wrappers_left=wrappers_left,
                recursion_level=recursion_level,
                direction=direction,
                target_mode=target_mode,
                target_mapping_kind=target_mapping_kind,
                regime_metadata=regime_metadata,
            ),
            "regime_lift": regime_lift,
        }

    x_grad, grad = _stack_x_feature(
        x_fit=x_fit,
        feature_fit=grad_fit,
        x_probe=x_probe,
        feature_probe=grad_probe,
        expected_cols=int(x.shape[1]),
    )
    x_d2, d2 = _stack_x_feature(
        x_fit=x_fit,
        feature_fit=d2_fit,
        x_probe=x_probe,
        feature_probe=d2_probe,
        expected_cols=int(x.shape[1]),
    )
    fit_source = str(fit_jet_source or "")
    probe_source = str(probe_jet_source or "")
    jet_source = fit_source if fit_source and fit_source == probe_source else ("mixed" if (fit_source or probe_source) else "")
    exact_jet_used = fit_source in {"oracle", "symbolic", "runtime_teacher"} or probe_source in {"oracle", "symbolic", "runtime_teacher"}
    numeric_jet_fallback_used = bool(fit_jet_fallback_used or probe_jet_fallback_used)

    axis, coords_sorted, values_sorted = _sorted_axis_data(x, y)
    values_rev = torch.flip(values_sorted, dims=[0])
    scale = max(1.0e-12, float(values_sorted.abs().mean().item()))
    even_resid = float((values_sorted - values_rev).abs().mean().item() / scale)
    odd_resid = float((values_sorted + values_rev).abs().mean().item() / scale)
    grad_even_resid = None
    grad_odd_resid = None
    d2_even_resid = None
    d2_odd_resid = None
    if grad is not None and x_grad is not None and int(grad.shape[0]) >= 4:
        _, grad_sorted = _sort_by_axis(x_grad, grad[:, axis : axis + 1], axis=axis)
        grad_even_resid = _mirror_residual(grad_sorted, sign=-1.0)
        grad_odd_resid = _mirror_residual(grad_sorted, sign=1.0)
    if d2 is not None and x_d2 is not None and int(d2.shape[0]) >= 4:
        _, d2_sorted = _sort_by_axis(x_d2, d2[:, axis : axis + 1], axis=axis)
        d2_even_resid = _mirror_residual(d2_sorted, sign=1.0)
        d2_odd_resid = _mirror_residual(d2_sorted, sign=-1.0)
    even_components = [float(even_resid)]
    odd_components = [float(odd_resid)]
    if grad_even_resid is not None and grad_odd_resid is not None:
        even_components.append(float(grad_even_resid))
        odd_components.append(float(grad_odd_resid))
    if d2_even_resid is not None and d2_odd_resid is not None:
        even_components.append(float(d2_even_resid))
        odd_components.append(float(d2_odd_resid))
    symmetry_even_score = float(sum(even_components) / max(1, len(even_components)))
    symmetry_odd_score = float(sum(odd_components) / max(1, len(odd_components)))
    symmetry_status = "even_like" if symmetry_even_score <= symmetry_odd_score and symmetry_even_score < 0.5 else ("odd_like" if symmetry_odd_score < 0.5 else "asymmetric")
    symmetry = _family_evidence(
        "symmetry",
        family_scores={
            "symmetry_even": float(1.0 / (1.0 + symmetry_even_score)),
            "symmetry_odd": float(1.0 / (1.0 + symmetry_odd_score)),
        },
        status=symmetry_status,
        hard_constraints={
            "primary_axis": int(axis),
            "mirror_even_residual": float(even_resid),
            "mirror_odd_residual": float(odd_resid),
            "gradient_mirror_even_residual": None if grad_even_resid is None else float(grad_even_resid),
            "gradient_mirror_odd_residual": None if grad_odd_resid is None else float(grad_odd_resid),
            "d2_mirror_even_residual": None if d2_even_resid is None else float(d2_even_resid),
            "d2_mirror_odd_residual": None if d2_odd_resid is None else float(d2_odd_resid),
            "jet_evidence_used": bool(grad_even_resid is not None or d2_even_resid is not None),
            "fit_jet_source": str(fit_source),
            "probe_jet_source": str(probe_source),
            "jet_source": str(jet_source),
            "exact_jet_used": bool(exact_jet_used),
            "numeric_jet_fallback_used": bool(numeric_jet_fallback_used),
            "zero_crossing_count": int(_crossing_count(values_sorted)),
        },
        metadata={
            "primary_axis": int(axis),
            "mirror_even_residual": float(even_resid),
            "mirror_odd_residual": float(odd_resid),
            "gradient_mirror_even_residual": None if grad_even_resid is None else float(grad_even_resid),
            "gradient_mirror_odd_residual": None if grad_odd_resid is None else float(grad_odd_resid),
            "d2_mirror_even_residual": None if d2_even_resid is None else float(d2_even_resid),
            "d2_mirror_odd_residual": None if d2_odd_resid is None else float(d2_odd_resid),
            "jet_evidence_used": bool(grad_even_resid is not None or d2_even_resid is not None),
            "fit_jet_source": str(fit_source),
            "probe_jet_source": str(probe_source),
            "jet_source": str(jet_source),
            "exact_jet_used": bool(exact_jet_used),
            "numeric_jet_fallback_used": bool(numeric_jet_fallback_used),
            "zero_crossing_count": int(_crossing_count(values_sorted)),
        },
        target_dim=target_dim,
        active_vars=active_vars,
        wrappers_left=wrappers_left,
        recursion_level=recursion_level,
        direction=direction,
        target_mode=target_mode,
        target_mapping_kind=target_mapping_kind,
        regime_metadata=regime_metadata,
    )

    corrs = [float(_pearson_abs(x[:, idx], y[:, 0])) for idx in range(int(x.shape[1]))]
    corr_sum = max(1.0e-12, float(sum(corrs)))
    corr_top_var = int(max(range(len(corrs)), key=lambda idx: corrs[idx])) if corrs else 0
    corr_dominant_frac = float(corrs[corr_top_var] / corr_sum) if corrs else 0.0
    corr_active_var_estimate = int(sum(1 for value in corrs if value >= (0.35 * max(corrs or [0.0])) and value > 0.05))
    jet_var_energy = None
    jet_top_var = None
    jet_dominant_frac = None
    jet_active_var_estimate = None
    if grad is not None and int(grad.shape[0]) >= 4:
        energy = _mean_square_by_col(grad)
        if int(energy.numel()) == int(x.shape[1]) and float(torch.sum(energy).item()) > 1.0e-20:
            jet_var_energy = [float(v) for v in energy.detach().cpu().tolist()]
            jet_top_var = int(torch.argmax(energy).item())
            jet_dominant_frac = float(energy[jet_top_var].item() / max(1.0e-12, float(torch.sum(energy).item())))
            jet_active_var_estimate = int(
                sum(
                    1
                    for value in jet_var_energy
                    if value >= (0.35 * max(jet_var_energy or [0.0])) and value > 1.0e-8
                )
            )
    top_var = int(jet_top_var if jet_top_var is not None else corr_top_var)
    dominant_frac = float(jet_dominant_frac if jet_dominant_frac is not None else corr_dominant_frac)
    active_var_estimate = int(jet_active_var_estimate if jet_active_var_estimate is not None else corr_active_var_estimate)
    low_rank_status = "strong_single_index" if dominant_frac >= 0.75 else "distributed"
    low_rank = _family_evidence(
        "low_rank_dependence",
        family_scores={"low_rank_dependence": float(dominant_frac)},
        status=low_rank_status,
        hard_constraints={
            "top_var": int(top_var),
            "dominant_var_frac": float(dominant_frac),
            "active_var_estimate": int(active_var_estimate),
            "jet_evidence_used": bool(jet_var_energy is not None),
            "jet_top_var": None if jet_top_var is None else int(jet_top_var),
            "jet_dominant_var_frac": None if jet_dominant_frac is None else float(jet_dominant_frac),
            "jet_active_var_estimate": None if jet_active_var_estimate is None else int(jet_active_var_estimate),
            "jet_var_energy": jet_var_energy,
            "var_abs_corrs": [float(v) for v in corrs],
            "fit_jet_source": str(fit_source),
            "probe_jet_source": str(probe_source),
            "jet_source": str(jet_source),
            "exact_jet_used": bool(exact_jet_used),
            "numeric_jet_fallback_used": bool(numeric_jet_fallback_used),
        },
        metadata={
            "top_var": int(top_var),
            "dominant_var_frac": float(dominant_frac),
            "active_var_estimate": int(active_var_estimate),
            "jet_evidence_used": bool(jet_var_energy is not None),
            "jet_top_var": None if jet_top_var is None else int(jet_top_var),
            "jet_dominant_var_frac": None if jet_dominant_frac is None else float(jet_dominant_frac),
            "jet_active_var_estimate": None if jet_active_var_estimate is None else int(jet_active_var_estimate),
            "jet_var_energy": jet_var_energy,
            "var_abs_corrs": [float(v) for v in corrs],
            "fit_jet_source": str(fit_source),
            "probe_jet_source": str(probe_source),
            "jet_source": str(jet_source),
            "exact_jet_used": bool(exact_jet_used),
            "numeric_jet_fallback_used": bool(numeric_jet_fallback_used),
        },
        target_dim=target_dim,
        active_vars=active_vars,
        wrappers_left=wrappers_left,
        recursion_level=recursion_level,
        direction=direction,
        target_mode=target_mode,
        target_mapping_kind=target_mapping_kind,
        regime_metadata=regime_metadata,
    )

    ones = torch.ones((int(x.shape[0]), 1), dtype=x.dtype, device=x.device)
    add_design = torch.cat([ones, x], dim=1)
    mse_add = _lstsq_mse(add_design, y)
    pair_terms = []
    for i in range(int(x.shape[1])):
        for j in range(i + 1, int(x.shape[1])):
            pair_terms.append((x[:, i] * x[:, j]).reshape(-1, 1))
    if pair_terms:
        full_design = torch.cat([add_design] + pair_terms, dim=1)
        mse_full = _lstsq_mse(full_design, y)
    else:
        mse_full = mse_add
    interaction_gain = 0.0
    if mse_add is not None and mse_full is not None and mse_add > 1.0e-12:
        interaction_gain = float(max(0.0, (float(mse_add) - float(mse_full)) / float(mse_add)))
    jet_interaction_gain = None
    jet_component_improvements = None
    if grad is not None and x_grad is not None and int(grad.shape[0]) >= max(4, int(x.shape[1]) + 2):
        grad_energy = _mean_square_by_col(grad)
        full_grad_design = torch.cat(
            [torch.ones((int(x_grad.shape[0]), 1), dtype=x_grad.dtype, device=x_grad.device), x_grad],
            dim=1,
        )
        active_grad_dims = [
            int(idx)
            for idx, value in enumerate(grad_energy.detach().cpu().tolist())
            if value >= (0.35 * max(grad_energy.detach().cpu().tolist() or [0.0])) and value > 1.0e-8
        ]
        improvements: list[float] = []
        for idx in active_grad_dims:
            target_grad = grad[:, idx : idx + 1]
            self_design = torch.cat(
                [
                    torch.ones((int(x_grad.shape[0]), 1), dtype=x_grad.dtype, device=x_grad.device),
                    x_grad[:, idx : idx + 1],
                ],
                dim=1,
            )
            mse_self = _lstsq_mse(self_design, target_grad)
            mse_full_grad = _lstsq_mse(full_grad_design, target_grad)
            if mse_self is None or mse_full_grad is None or mse_self <= 1.0e-12:
                improvements.append(0.0)
                continue
            improvements.append(float(max(0.0, (float(mse_self) - float(mse_full_grad)) / float(mse_self))))
        if improvements:
            jet_component_improvements = [float(v) for v in improvements]
            jet_interaction_gain = float(sum(improvements) / len(improvements))
    separability_gain = float(jet_interaction_gain if jet_interaction_gain is not None else interaction_gain)
    separability_score = float(max(0.0, 1.0 - separability_gain))
    separability_status = "additive_like" if active_var_estimate >= 2 and separability_gain <= 0.10 else "interaction_heavy"
    separability = _family_evidence(
        "separability",
        family_scores={"separability": float(separability_score)},
        status=separability_status,
        hard_constraints={
            "active_var_estimate": int(active_var_estimate),
            "interaction_gain": float(interaction_gain),
            "jet_evidence_used": bool(jet_interaction_gain is not None),
            "jet_interaction_gain": None if jet_interaction_gain is None else float(jet_interaction_gain),
            "jet_component_improvements": jet_component_improvements,
            "mse_additive_linear": mse_add,
            "mse_pairwise_linear": mse_full,
            "fit_jet_source": str(fit_source),
            "probe_jet_source": str(probe_source),
            "jet_source": str(jet_source),
            "exact_jet_used": bool(exact_jet_used),
            "numeric_jet_fallback_used": bool(numeric_jet_fallback_used),
        },
        metadata={
            "active_var_estimate": int(active_var_estimate),
            "interaction_gain": float(interaction_gain),
            "jet_evidence_used": bool(jet_interaction_gain is not None),
            "jet_interaction_gain": None if jet_interaction_gain is None else float(jet_interaction_gain),
            "jet_component_improvements": jet_component_improvements,
            "mse_additive_linear": mse_add,
            "mse_pairwise_linear": mse_full,
            "fit_jet_source": str(fit_source),
            "probe_jet_source": str(probe_source),
            "jet_source": str(jet_source),
            "exact_jet_used": bool(exact_jet_used),
            "numeric_jet_fallback_used": bool(numeric_jet_fallback_used),
        },
        target_dim=target_dim,
        active_vars=active_vars,
        wrappers_left=wrappers_left,
        recursion_level=recursion_level,
        direction=direction,
        target_mode=target_mode,
        target_mapping_kind=target_mapping_kind,
        regime_metadata=regime_metadata,
    )

    slopes = _finite_slopes(coords_sorted, values_sorted)
    second = slopes[1:] - slopes[:-1] if int(slopes.numel()) > 1 else torch.zeros((0,), dtype=values_sorted.dtype)
    grad_signal = slopes
    grad_source = "finite_difference"
    if grad is not None and x_grad is not None and int(grad.shape[0]) >= 4:
        _, grad_sorted = _sort_by_axis(x_grad, grad[:, axis : axis + 1], axis=axis)
        grad_signal = grad_sorted.reshape(-1)
        grad_source = "jet"
    curvature_signal = second
    curvature_source = "finite_difference"
    if d2 is not None and x_d2 is not None and int(d2.shape[0]) >= 4:
        _, d2_sorted = _sort_by_axis(x_d2, d2[:, axis : axis + 1], axis=axis)
        curvature_signal = d2_sorted.reshape(-1)
        curvature_source = "jet"
    grad_spike = _spike_ratio(grad_signal)
    curvature_spike = _spike_ratio(curvature_signal)
    singularity_margin = float(1.0 / (1.0 + grad_spike + curvature_spike))
    hazard_severe = bool(singularity_margin < 0.20 or grad_spike > 8.0 or curvature_spike > 8.0)
    domain_status = "severe_hazard" if hazard_severe else ("moderate_hazard" if singularity_margin < 0.35 else "stable")
    domain_hazard = _family_evidence(
        "domain_hazard",
        family_scores={"domain_hazard": float(singularity_margin)},
        status=domain_status,
        hard_constraints={
            "gradient_spike_ratio": float(grad_spike),
            "curvature_spike_ratio": float(curvature_spike),
            "singularity_margin_proxy": float(singularity_margin),
            "hazard_severe": bool(hazard_severe),
            "slope_sign_change_count": int(_sign_changes(grad_signal)),
            "curvature_sign_change_count": int(_sign_changes(curvature_signal)),
            "gradient_signal_source": str(grad_source),
            "curvature_signal_source": str(curvature_source),
            "jet_evidence_used": bool(grad_source == "jet" or curvature_source == "jet"),
            "fit_jet_source": str(fit_source),
            "probe_jet_source": str(probe_source),
            "jet_source": str(jet_source),
            "exact_jet_used": bool(exact_jet_used),
            "numeric_jet_fallback_used": bool(numeric_jet_fallback_used),
        },
        metadata={
            "gradient_spike_ratio": float(grad_spike),
            "curvature_spike_ratio": float(curvature_spike),
            "singularity_margin_proxy": float(singularity_margin),
            "hazard_severe": bool(hazard_severe),
            "slope_sign_change_count": int(_sign_changes(grad_signal)),
            "curvature_sign_change_count": int(_sign_changes(curvature_signal)),
            "gradient_signal_source": str(grad_source),
            "curvature_signal_source": str(curvature_source),
            "jet_evidence_used": bool(grad_source == "jet" or curvature_source == "jet"),
            "fit_jet_source": str(fit_source),
            "probe_jet_source": str(probe_source),
            "jet_source": str(jet_source),
            "exact_jet_used": bool(exact_jet_used),
            "numeric_jet_fallback_used": bool(numeric_jet_fallback_used),
        },
        target_dim=target_dim,
        active_vars=active_vars,
        wrappers_left=wrappers_left,
        recursion_level=recursion_level,
        direction=direction,
        target_mode=target_mode,
        target_mapping_kind=target_mapping_kind,
        regime_metadata=regime_metadata,
        domain_masks={
            "gradient_signal_source": str(grad_source),
            "curvature_signal_source": str(curvature_source),
        },
    )

    valid_tail = (coords_sorted.abs() > 1.0e-8) & (values_sorted.abs() > 1.0e-12)
    asymptotic = _family_evidence(
        "asymptotic_monomial",
        family_scores={"asymptotic_monomial": 0.0},
        status="insufficient_tail",
        target_dim=target_dim,
        active_vars=active_vars,
        wrappers_left=wrappers_left,
        recursion_level=recursion_level,
        direction=direction,
        target_mode=target_mode,
        target_mapping_kind=target_mapping_kind,
        regime_metadata=regime_metadata,
    )
    if int(valid_tail.to(dtype=torch.int64).sum().item()) >= 4:
        coords_tail_all = coords_sorted.abs()[valid_tail]
        values_tail_all = values_sorted.abs()[valid_tail]
        tail_mask = _quantile_tail_mask(coords_tail_all, q=0.65, min_points=4)
        coords_tail = coords_tail_all[tail_mask]
        values_tail = values_tail_all[tail_mask]
        fit_stats = _linear_fit_stats(torch.log(coords_tail), torch.log(values_tail))
        if fit_stats is not None:
            asym_status = "monomial_like" if float(fit_stats["r2"]) >= 0.92 else "non_monomial"
            asymptotic = _family_evidence(
                "asymptotic_monomial",
                family_scores={"asymptotic_monomial": float(fit_stats["r2"])},
                status=asym_status,
                hard_constraints={
                    "primary_axis": int(axis),
                    "tail_count": int(coords_tail.shape[0]),
                    "log_fit_r2": float(fit_stats["r2"]),
                    "log_fit_mse": float(fit_stats["mse"]),
                    "exponent_estimate": float(fit_stats["slope"]),
                    "log_amplitude_estimate": float(fit_stats["bias"]),
                },
                metadata={
                    "primary_axis": int(axis),
                    "tail_count": int(coords_tail.shape[0]),
                    "log_fit_r2": float(fit_stats["r2"]),
                    "log_fit_mse": float(fit_stats["mse"]),
                    "exponent_estimate": float(fit_stats["slope"]),
                    "log_amplitude_estimate": float(fit_stats["bias"]),
                },
                target_dim=target_dim,
                active_vars=active_vars,
                wrappers_left=wrappers_left,
                recursion_level=recursion_level,
                direction=direction,
                target_mode=target_mode,
                target_mapping_kind=target_mapping_kind,
                regime_metadata=regime_metadata,
            )

    pos_frac = float(torch.mean((coords_sorted > 1.0e-8).to(dtype=torch.float64)).item())
    neg_frac = float(torch.mean((coords_sorted < -1.0e-8).to(dtype=torch.float64)).item())
    zero_band_frac = float(torch.mean((coords_sorted.abs() <= 1.0e-8).to(dtype=torch.float64)).item())
    value_pos_frac = float(torch.mean((values_sorted > 1.0e-10).to(dtype=torch.float64)).item())
    value_neg_frac = float(torch.mean((values_sorted < -1.0e-10).to(dtype=torch.float64)).item())
    value_zero_frac = float(torch.mean((values_sorted.abs() <= 1.0e-10).to(dtype=torch.float64)).item())
    one_sided_support = bool(min(pos_frac, neg_frac) <= 0.10 and max(pos_frac, neg_frac) >= 0.60)
    hazard_risk = float(max(0.0, 1.0 - singularity_margin))
    branch_cut_risk = float(
        min(
            1.0,
            0.45 * (1.0 if one_sided_support else 0.0)
            + 0.25 * max(value_pos_frac, value_neg_frac)
            + 0.30 * hazard_risk,
        )
    )
    if bool(one_sided_support) and bool(hazard_severe):
        branch_status = "branch_like_hazard"
    elif float(branch_cut_risk) >= 0.65:
        branch_status = "branch_like"
    else:
        branch_status = "two_sided_or_smooth"
    branch_structure = _family_evidence(
        "branch_structure",
        family_scores={"branch_structure": float(branch_cut_risk)},
        status=branch_status,
        hard_constraints={
            "primary_axis": int(axis),
            "positive_coord_frac": float(pos_frac),
            "negative_coord_frac": float(neg_frac),
            "axis_zero_band_frac": float(zero_band_frac),
            "target_positive_frac": float(value_pos_frac),
            "target_negative_frac": float(value_neg_frac),
            "target_zero_frac": float(value_zero_frac),
            "one_sided_support": bool(one_sided_support),
            "branch_cut_risk": float(branch_cut_risk),
            "hazard_severe": bool(hazard_severe),
            "singularity_margin_proxy": float(singularity_margin),
        },
        metadata={
            "primary_axis": int(axis),
            "branch_cut_risk": float(branch_cut_risk),
            "one_sided_support": bool(one_sided_support),
            "hazard_severe": bool(hazard_severe),
        },
        target_dim=target_dim,
        active_vars=active_vars,
        wrappers_left=wrappers_left,
        recursion_level=recursion_level,
        direction=direction,
        target_mode=target_mode,
        target_mapping_kind=target_mapping_kind,
        regime_metadata=regime_metadata,
        domain_masks={
            "one_sided_support": bool(one_sided_support),
            "branch_cut_risk": float(branch_cut_risk),
        },
    )

    coordinate_invariant = _family_evidence(
        "coordinate_invariant",
        family_scores={"coordinate_invariant": 0.0},
        status="insufficient_gradient_evidence",
        target_dim=target_dim,
        active_vars=active_vars,
        wrappers_left=wrappers_left,
        recursion_level=recursion_level,
        direction=direction,
        target_mode=target_mode,
        target_mapping_kind=target_mapping_kind,
        regime_metadata=regime_metadata,
    )
    coordinate_vars = [int(v) for v in tuple(active_vars or ()) if 0 <= int(v) < int(x.shape[1])]
    if len(coordinate_vars) < 2 and grad is not None and int(grad.shape[1]) >= 2:
        grad_energy = _mean_square_by_col(grad)
        ranked = torch.argsort(grad_energy, descending=True).tolist()
        coordinate_vars = [int(idx) for idx in ranked[: min(3, len(ranked))] if float(grad_energy[int(idx)].item()) > 1.0e-8]
    if len(coordinate_vars) >= 2 and grad is not None and int(grad.shape[0]) >= 4:
        grad_slice = grad[:, coordinate_vars].to(dtype=torch.float64)
        finite_mask = torch.isfinite(grad_slice).all(dim=1)
        norm = torch.linalg.vector_norm(grad_slice, dim=1)
        usable = finite_mask & (norm > 1.0e-10)
        if int(usable.to(dtype=torch.int64).sum().item()) >= 4:
            grad_rows = grad_slice[usable]
            row_norm = torch.linalg.vector_norm(grad_rows, dim=1, keepdim=True).clamp_min(1.0e-12)
            unit_rows = grad_rows / row_norm
            try:
                _u, singular_values, vh = torch.linalg.svd(grad_rows, full_matrices=False)
            except Exception:
                singular_values = torch.zeros((0,), dtype=torch.float64)
                vh = None
            rank1_frac = 0.0
            direction_vec = None
            if vh is not None and int(singular_values.numel()) > 0:
                total_sv = float(torch.sum(singular_values).item())
                if total_sv > 1.0e-12:
                    rank1_frac = float(singular_values[0].item() / total_sv)
                    direction_vec = vh[0]
            if direction_vec is None:
                direction_vec = unit_rows.mean(dim=0)
            anchor = int(torch.argmax(torch.abs(direction_vec)).item())
            if float(direction_vec[anchor].item()) < 0.0:
                direction_vec = -direction_vec
            unit_dir = direction_vec / torch.linalg.vector_norm(direction_vec).clamp_min(1.0e-12)
            coherence = float(torch.mean(torch.abs(unit_rows @ unit_dir)).item())
            invariant_score = float(max(rank1_frac, coherence))
            invariant_status = "single_index_like" if float(invariant_score) >= 0.90 else "mixed_coordinate"
            coordinate_invariant = _family_evidence(
                "coordinate_invariant",
                family_scores={"coordinate_invariant": float(invariant_score)},
                status=invariant_status,
                hard_constraints={
                    "coordinate_vars": [int(v) for v in coordinate_vars],
                    "gradient_rank1_fraction": float(rank1_frac),
                    "gradient_direction_coherence": float(coherence),
                    "mean_direction": [float(v) for v in unit_dir.detach().cpu().tolist()],
                },
                metadata={
                    "coordinate_vars": [int(v) for v in coordinate_vars],
                    "gradient_rank1_fraction": float(rank1_frac),
                    "gradient_direction_coherence": float(coherence),
                },
                seed_nodes=_coordinate_seed_nodes(
                    active_vars=tuple(coordinate_vars),
                    direction=direction_vec,
                ),
                target_dim=target_dim,
                active_vars=active_vars,
                wrappers_left=wrappers_left,
                recursion_level=recursion_level,
                direction=direction,
                target_mode=target_mode,
                target_mapping_kind=target_mapping_kind,
                regime_metadata=regime_metadata,
            )
    return {
        "symmetry": symmetry,
        "separability": separability,
        "low_rank_dependence": low_rank,
        "domain_hazard": domain_hazard,
        "asymptotic_monomial": asymptotic,
        "branch_structure": branch_structure,
        "coordinate_invariant": coordinate_invariant,
        "regime_lift": regime_lift,
    }


def default_outer_family_battery_specs(
    *,
    periodic_min_improvement_ratio: float,
    periodic_precheck_max_seeds: int,
    default_min_improvement_ratio: float,
    default_precheck_max_seeds: int = 4,
) -> tuple[OuterFamilyBatterySpec, ...]:
    return (
        OuterFamilyBatterySpec(
            name="periodic",
            min_improvement_ratio=float(periodic_min_improvement_ratio),
            precheck_max_seeds=max(1, int(periodic_precheck_max_seeds)),
        ),
        OuterFamilyBatterySpec(
            name="exp",
            min_improvement_ratio=float(default_min_improvement_ratio),
            precheck_max_seeds=max(1, int(default_precheck_max_seeds)),
        ),
        OuterFamilyBatterySpec(
            name="power",
            min_improvement_ratio=float(default_min_improvement_ratio),
            precheck_max_seeds=max(1, int(default_precheck_max_seeds)),
        ),
        OuterFamilyBatterySpec(
            name="rational",
            min_improvement_ratio=float(default_min_improvement_ratio),
            precheck_max_seeds=max(1, int(default_precheck_max_seeds)),
        ),
    )


def build_named_outer_family_evidence(
    family_name: str,
    *,
    recursive_enable: bool,
    wrappers_left: int,
    flat_rows_present: bool,
    target_dim_ok: bool,
    best_flat_probe_mse: float | None,
    seed_nodes: Sequence[Any] | None,
    min_improvement_ratio: float,
    best_probe_mse: float | None = None,
    status_override: str | None = None,
    extra_hard_constraints: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    target_dim: Any = None,
    active_vars: Sequence[int] | None = None,
    recursion_level: int | None = None,
    direction: str | None = None,
    target_mode: str | None = None,
    target_mapping_kind: str | None = None,
    dimensionless_target_required: bool = True,
    domain_masks: Mapping[str, Any] | None = None,
    regime_metadata: Mapping[str, Any] | None = None,
) -> FamilyEvidence:
    family_key = str(family_name or "")
    best_flat = _safe_float(best_flat_probe_mse, None)
    best_probe = _safe_float(best_probe_mse, None)
    candidate_count = int(len(list(seed_nodes or ())))
    improvement_ratio = None
    if best_flat is not None and best_probe is not None:
        improvement_ratio = float(best_probe / max(best_flat, 1.0e-12))
    if status_override is not None:
        status = str(status_override)
    elif not bool(recursive_enable):
        status = "disabled"
    elif int(wrappers_left) <= 0:
        status = "no_wrappers_left"
    elif not bool(flat_rows_present):
        status = "no_flat_rows"
    elif not bool(target_dim_ok):
        status = "nondimensionless_target"
    elif candidate_count <= 0:
        status = "no_candidate_nodes"
    elif best_flat is None:
        status = "nonfinite_flat_probe"
    elif best_probe is None:
        status = "no_finite_fit"
    elif float(improvement_ratio) <= float(min_improvement_ratio):
        status = "triggered"
    else:
        status = "insufficient_improvement"
    should_run = bool(status == "triggered")
    score = 0.0
    if best_flat is not None and best_probe is not None:
        score = float(best_flat / max(best_probe, 1.0e-12))
    elif should_run:
        score = 1.0
    hard_constraints = {
        "status": str(status),
        "recursive_enable": bool(recursive_enable),
        "wrappers_left": int(max(0, int(wrappers_left))),
        "flat_rows_present": bool(flat_rows_present),
        "target_dim_ok": bool(target_dim_ok),
        "candidate_count": int(candidate_count),
        "best_flat_probe_mse": best_flat,
        "best_probe_mse": best_probe,
        "improvement_ratio": improvement_ratio,
        "min_improvement_ratio": float(min_improvement_ratio),
        "should_run": bool(should_run),
    }
    hard_constraints.update(dict(extra_hard_constraints or {}))
    metadata = {
        "status": str(status),
        "should_run": bool(should_run),
        "best_probe_mse": best_probe,
        "improvement_ratio": improvement_ratio,
        "candidate_count": int(candidate_count),
    }
    metadata.update(dict(extra_metadata or {}))
    return FamilyEvidence(
        family_scores={family_key: float(score)},
        hard_constraints=canonicalize_family_hard_constraints(
            family_key,
            hard_constraints,
            status=str(status),
            should_run=bool(should_run),
            advisory_only=True,
            target_dim=target_dim,
            active_vars=active_vars,
            wrappers_left=int(max(0, int(wrappers_left))),
            recursion_level=recursion_level,
            direction=direction,
            target_mode=target_mode,
            target_mapping_kind=target_mapping_kind,
            dimensionless_target_required=bool(dimensionless_target_required),
            target_dim_ok=bool(target_dim_ok),
            domain_masks=domain_masks,
            regime_metadata=regime_metadata,
        ),
        seed_nodes=_coerce_seed_nodes(seed_nodes),
        metadata=metadata,
    )


def family_evidence_status(evidence: FamilyEvidence | None, family_name: str) -> str:
    if evidence is None:
        return ""
    status = dict(evidence.metadata or {}).get("status", "")
    if status:
        return str(status)
    return str(dict(evidence.hard_constraints or {}).get("status", "") or "")


def family_evidence_should_run(evidence: FamilyEvidence | None, family_name: str) -> bool:
    if evidence is None:
        return False
    if "should_run" in dict(evidence.metadata or {}):
        return bool(dict(evidence.metadata or {}).get("should_run", False))
    return bool(dict(evidence.hard_constraints or {}).get("should_run", False))


def build_square_family_evidence(
    *,
    proposal_name: str,
    proposal_score: float,
    proposal_improvement: float,
    proposal_details: Mapping[str, Any] | None,
    prefer: bool,
    diagnostics: Mapping[str, Any] | None,
) -> FamilyEvidence:
    family_key = str(proposal_name or "square")
    diag = dict(diagnostics or {})
    status = "triggered" if bool(prefer) else str(diag.get("reason", "rejected") or "rejected")
    score = max(0.0, float(_safe_float(proposal_improvement, 0.0) or 0.0))
    hard_constraints = {
        "status": str(status),
        "prefer": bool(prefer),
        "proposal_score": _safe_float(proposal_score, None),
        "proposal_improvement": _safe_float(proposal_improvement, None),
        **dict(diag),
    }
    metadata = {
        "status": str(status),
        "prefer": bool(prefer),
        "proposal_score": _safe_float(proposal_score, None),
        "proposal_improvement": _safe_float(proposal_improvement, None),
        "proposal_details": dict(proposal_details or {}),
        "decision_diagnostics": dict(diag),
    }
    return _family_evidence(
        family_key,
        family_scores={family_key: float(score)},
        status=str(status),
        hard_constraints=hard_constraints,
        metadata=metadata,
    )


__all__ = [
    "OuterFamilyBatterySpec",
    "build_expanded_family_evidence_bundle",
    "build_named_outer_family_evidence",
    "build_square_family_evidence",
    "default_outer_family_battery_specs",
    "family_evidence_should_run",
    "family_evidence_status",
    "normalize_family_battery_mode",
]
