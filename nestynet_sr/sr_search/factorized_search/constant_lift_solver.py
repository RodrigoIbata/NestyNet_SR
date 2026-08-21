# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from .bridge import run_explorer
from .expr_ast import BINARY_OPS, UNARY_OPS, is_valid_node, node_str
from .inverse_core import _normalize_inverse_local_score_mode
from .inverse_spec_solver import (
    _apply_continuation_frames,
    _candidate_to_preview_row,
    _dedup_scored_candidates,
    _score_node_against_problem,
)
from .lift_route_evidence import build_local_lift_route_context
from .subproblem_spec import SubproblemSpec
from .tangent_edit import _build_solver_context, _local_problem_from_payload


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    scalar = _safe_float(value)
    return None if scalar is None else float(scalar)


def _normalize_dataset_metadata(
    regime_ids: Sequence[str],
    dataset_metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, dict[str, float]]:
    ids = [str(item) for item in list(regime_ids or [])]
    if not ids or dataset_metadata is None:
        return {}
    rows: dict[str, dict[str, float]] = {}
    if isinstance(dataset_metadata, Mapping) and all(
        isinstance(value, Mapping) for value in dataset_metadata.values()
    ):
        raw_rows = {
            str(key): dict(value)
            for key, value in dict(dataset_metadata).items()
            if isinstance(value, Mapping)
        }
    elif isinstance(dataset_metadata, Sequence) and not isinstance(dataset_metadata, (str, bytes)):
        raw_rows = {
            str(ids[idx]): dict(value)
            for idx, value in enumerate(list(dataset_metadata))
            if idx < len(ids) and isinstance(value, Mapping)
        }
    else:
        raw_rows = {}
    for regime_id in ids:
        row = dict(raw_rows.get(str(regime_id), {}) or {})
        numeric: dict[str, float] = {}
        for key, value in row.items():
            scalar = _safe_float(value)
            if scalar is not None:
                numeric[str(key)] = float(scalar)
        if numeric:
            rows[str(regime_id)] = numeric
    return rows


def build_regime_feature_matrix(
    regime_ids: Sequence[str],
    *,
    dataset_metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, list[str], str]:
    ids = [str(item) for item in list(regime_ids or [])]
    if not ids:
        raise ValueError("regime_ids must be non-empty")
    normalized = _normalize_dataset_metadata(ids, dataset_metadata)
    feature_names = sorted(
        key
        for key in {
            str(name)
            for row in normalized.values()
            for name in row.keys()
        }
        if all(key in normalized.get(regime_id, {}) for regime_id in ids)
    )
    if feature_names:
        rows = [
            [float(normalized[regime_id][name]) for name in feature_names]
            for regime_id in ids
        ]
        return torch.tensor(rows, dtype=dtype), list(feature_names), "dataset_metadata"
    values = [[float(idx)] for idx in range(len(ids))]
    return torch.tensor(values, dtype=dtype), ["regime_index"], "regime_index"


def _merged_subproblem_context(subproblem_spec: SubproblemSpec | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    witness = getattr(subproblem_spec, "witness", None)
    diagnostics = getattr(witness, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        merged.update(dict(diagnostics))
    merged.update(dict(getattr(subproblem_spec, "metadata", {}) or {}))
    return merged


def _normalize_local_constants_by_experiment(
    local_constants_by_experiment: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for regime_id, payload in dict(local_constants_by_experiment or {}).items():
        if not isinstance(payload, Mapping):
            continue
        numeric: dict[str, float] = {}
        for key, value in dict(payload).items():
            scalar = _safe_float(value)
            if scalar is not None:
                numeric[str(key)] = float(scalar)
        if numeric:
            rows[str(regime_id)] = numeric
    return rows


def _parameter_stability_summary(
    parameter_samples: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    max_mean_cv: float = 0.5,
) -> dict[str, Any]:
    if parameter_samples is None:
        return {"passed": None, "score": None, "details": {"reason": "parameter samples missing"}}
    grouped: dict[str, list[float]] = {}
    if isinstance(parameter_samples, Mapping):
        iterable = [parameter_samples]
    else:
        iterable = list(parameter_samples)
    for row_idx, sample in enumerate(iterable):
        if isinstance(sample, Mapping):
            items = sample.items()
        else:
            items = [(f"param_{int(row_idx)}", sample)]
        for key, value in items:
            scalar = _safe_float(value)
            if scalar is not None:
                grouped.setdefault(str(key), []).append(float(scalar))
    if not grouped:
        return {"passed": None, "score": None, "details": {"reason": "no finite parameter samples"}}
    cvs: list[float] = []
    parameter_cvs: dict[str, float] = {}
    parameter_sample_counts: dict[str, int] = {}
    for key, values in grouped.items():
        if not values:
            continue
        mean = float(sum(values) / float(len(values)))
        centered = [float(value) - mean for value in values]
        var = float(sum(value * value for value in centered) / float(len(centered)))
        std = math.sqrt(max(0.0, var))
        cv = float(std) / max(1.0e-12, abs(mean))
        cvs.append(float(cv))
        parameter_cvs[str(key)] = float(cv)
        parameter_sample_counts[str(key)] = int(len(values))
    mean_cv = 0.0 if not cvs else float(sum(cvs) / len(cvs))
    score = 1.0 / (1.0 + float(mean_cv))
    return {
        "passed": bool(float(mean_cv) <= float(max_mean_cv)),
        "score": float(score),
        "details": {
            "mean_cv": float(mean_cv),
            "n_parameters": int(len(grouped)),
            "parameter_cvs": parameter_cvs,
            "parameter_sample_counts": parameter_sample_counts,
        },
    }


def _task_thresholds(metadata: Mapping[str, Any]) -> tuple[int, float]:
    min_regimes = 3
    for key in ("constant_lift_min_regimes", "min_regimes"):
        try:
            candidate = int(metadata.get(key, min_regimes) or min_regimes)
        except Exception:
            continue
        min_regimes = max(2, int(candidate))
        break
    trigger_mean_cv = 0.5
    for key in ("constant_lift_trigger_mean_cv", "trigger_mean_cv", "max_mean_cv"):
        candidate = _safe_float(metadata.get(key, None))
        if candidate is None:
            continue
        trigger_mean_cv = max(0.0, float(candidate))
        break
    return int(min_regimes), float(trigger_mean_cv)


def _synthesize_constant_lift_task(
    metadata: Mapping[str, Any],
    *,
    route_signal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    local_constants = _normalize_local_constants_by_experiment(
        metadata.get("local_constants_by_experiment", None)
    )
    if not local_constants:
        return {}
    requested_regime_ids = [str(item) for item in list(metadata.get("dataset_ids", ()) or ()) if str(item)]
    regime_ids = [
        str(regime_id)
        for regime_id in (requested_regime_ids if requested_regime_ids else sorted(local_constants.keys()))
        if str(regime_id) in local_constants
    ]
    min_regimes, trigger_mean_cv = _task_thresholds(metadata)
    if len(regime_ids) < int(min_regimes):
        return {}
    parameter_samples = [dict(local_constants.get(regime_id, {}) or {}) for regime_id in regime_ids]
    stability = _parameter_stability_summary(
        parameter_samples,
        max_mean_cv=float(trigger_mean_cv),
    )
    details = dict(stability.get("details", {}) or {})
    parameter_cvs = dict(details.get("parameter_cvs", {}) or {})
    parameter_counts = dict(details.get("parameter_sample_counts", {}) or {})
    ranked_constants: list[tuple[str, float, int]] = []
    preferred_constant = str(dict(route_signal or {}).get("top_constant_name", "") or "")
    preferred_cv = _safe_float(dict(route_signal or {}).get("top_constant_cv", None))
    for constant_name, value in parameter_cvs.items():
        mean_cv = _safe_float(value)
        if mean_cv is None or float(mean_cv) <= float(trigger_mean_cv):
            continue
        try:
            sample_count = int(parameter_counts.get(constant_name, 0) or 0)
        except Exception:
            sample_count = 0
        if sample_count < int(min_regimes):
            continue
        ranked_constants.append((str(constant_name), float(mean_cv), int(sample_count)))
    if preferred_constant and preferred_constant in parameter_cvs:
        preferred_count = int(parameter_counts.get(preferred_constant, 0) or 0)
        if preferred_count >= int(min_regimes):
            preferred_mean_cv = _safe_float(parameter_cvs.get(preferred_constant, preferred_cv))
            if preferred_mean_cv is not None and float(preferred_mean_cv) > float(trigger_mean_cv):
                ranked_constants.insert(0, (str(preferred_constant), float(preferred_mean_cv), int(preferred_count)))
    ranked_constants.sort(key=lambda row: (-float(row[1]), -int(row[2]), str(row[0])))
    if not ranked_constants:
        return {}
    constant_name, mean_cv, sample_count = ranked_constants[0]
    values_by_regime = {
        str(regime_id): float(local_constants[regime_id][constant_name])
        for regime_id in regime_ids
        if constant_name in local_constants.get(str(regime_id), {})
    }
    kept_regime_ids = [str(regime_id) for regime_id in regime_ids if str(regime_id) in values_by_regime]
    if len(kept_regime_ids) < int(min_regimes):
        return {}
    return {
        "constant_name": str(constant_name),
        "regime_ids": list(kept_regime_ids),
        "values_by_regime": dict(values_by_regime),
        "dataset_metadata": metadata.get("dataset_metadata", None),
        "feature_nodes": list(metadata.get("constant_lift_feature_nodes", ()) or ()),
        "task_source": "parameter_stability_auto",
        "parameter_stability": dict(stability),
        "mean_cv": float(mean_cv),
        "sample_count": int(sample_count),
        "trigger_mean_cv": float(trigger_mean_cv),
    }


def _weighted_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    resid = y_pred.reshape_as(y_true) - y_true
    return float(torch.mean(resid * resid).item())


def _const_baseline_mse(y: torch.Tensor) -> float:
    baseline = torch.full_like(y, float(torch.mean(y).item()))
    return _weighted_mse(y, baseline)


def _mul_const(node, coeff: float):
    cc = float(coeff)
    if not math.isfinite(cc) or abs(cc) < 1.0e-12:
        return None
    if abs(cc - 1.0) < 1.0e-12:
        return node
    if abs(cc + 1.0) < 1.0e-12:
        return ("neg", node)
    return ("mul", ("const", cc), node)


def _affine_ast(coeffs: Sequence[float]) -> tuple:
    coeff_list = [float(value) for value in list(coeffs or [])]
    bias = coeff_list[0] if coeff_list else 0.0
    terms: list[tuple] = []
    if math.isfinite(bias) and abs(bias) >= 1.0e-12:
        terms.append(("const", float(bias)))
    for idx, coeff in enumerate(coeff_list[1:]):
        term = _mul_const(("var", int(idx)), float(coeff))
        if term is not None:
            terms.append(term)
    if not terms:
        return ("const", 0.0)
    out = terms[0]
    for term in terms[1:]:
        out = ("add", out, term)
    return out


def _ast_from_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_ast_from_jsonable(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_ast_from_jsonable(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _ast_from_jsonable(item) for key, item in dict(value).items()}
    return value


def _const_node(value: Any) -> tuple:
    scalar = _safe_float(value)
    return ("const", 0.0 if scalar is None else float(scalar))


def _mul_const_ast(node, coeff: Any):
    scalar = _safe_float(coeff)
    if scalar is None:
        return None
    if abs(float(scalar)) <= 1.0e-12:
        return ("const", 0.0)
    if abs(float(scalar) - 1.0) <= 1.0e-12:
        return node
    if abs(float(scalar) + 1.0) <= 1.0e-12:
        return ("neg", node)
    return ("mul", ("const", float(scalar)), node)


def _add_terms(nodes: Sequence[Any]) -> tuple | None:
    terms = [node for node in list(nodes or ()) if isinstance(node, tuple) and node]
    if not terms:
        return None
    out = terms[0]
    for term in terms[1:]:
        out = ("add", out, term)
    return out


def _normalize_node(node):
    candidate = _ast_from_jsonable(node)
    return candidate if is_valid_node(candidate) else None


def _compose_feature_expr(node, feature_nodes: Sequence[tuple]):
    if not isinstance(node, tuple) or not node:
        return node
    op = str(node[0])
    if op == "var":
        idx = int(node[1])
        if 0 <= idx < len(feature_nodes):
            return feature_nodes[idx]
        raise ValueError(f"feature var index {idx} out of range")
    if op in ("const", "hparam"):
        return node
    if op in UNARY_OPS and len(node) >= 2:
        return (op, _compose_feature_expr(node[1], feature_nodes))
    if op in BINARY_OPS and len(node) >= 3:
        return (
            op,
            _compose_feature_expr(node[1], feature_nodes),
            _compose_feature_expr(node[2], feature_nodes),
        )
    raise ValueError(f"unsupported node {node!r}")


def _normalized_feature_expr(base_node, *, mu: Any, std: Any):
    mu_v = _safe_float(mu)
    std_v = _safe_float(std)
    out = base_node
    if mu_v is not None and abs(float(mu_v)) > 1.0e-12:
        out = ("sub", out, ("const", float(mu_v)))
    if std_v is not None and abs(float(std_v) - 1.0) > 1.0e-12:
        out = ("div", out, ("const", float(std_v)))
    return out


def _poly_ast(base_node, coeffs: Sequence[Any], *, mu: Any, std: Any):
    coeff_list = [_safe_float(value) for value in list(coeffs or ())]
    coeff_list = [0.0 if value is None else float(value) for value in coeff_list]
    if not coeff_list:
        return ("const", 0.0)
    z = _normalized_feature_expr(base_node, mu=mu, std=std)
    out: tuple | None = None
    for coeff in reversed(coeff_list):
        coeff_node = ("const", float(coeff))
        if out is None:
            out = coeff_node
        else:
            out = ("add", coeff_node, ("mul", z, out))
    return out if out is not None else ("const", 0.0)


def _pade_poly_ast(base_node, coeffs: Sequence[Any], *, mu: Any, std: Any):
    coeff_list = [_safe_float(value) for value in list(coeffs or ())]
    coeff_list = [0.0 if value is None else float(value) for value in coeff_list]
    z = _normalized_feature_expr(base_node, mu=mu, std=std)
    if not coeff_list:
        return ("const", 0.0)
    out: tuple | None = None
    for coeff in reversed(coeff_list):
        coeff_node = ("const", float(coeff))
        if out is None:
            out = coeff_node
        else:
            out = ("add", coeff_node, ("mul", z, out))
    return out if out is not None else ("const", 0.0)


def _mapping_to_ast(base_node, mapping: Mapping[str, Any]) -> tuple | None:
    kind = str(dict(mapping or {}).get("kind", "") or "").strip().lower()
    if kind in ("", "identity"):
        return base_node
    if kind == "poly":
        return _poly_ast(
            base_node,
            dict(mapping or {}).get("coeffs", (0.0, 1.0)),
            mu=dict(mapping or {}).get("mu", 0.0),
            std=dict(mapping or {}).get("std", 1.0),
        )
    if kind == "pade":
        numer = _pade_poly_ast(
            base_node,
            dict(mapping or {}).get("numer", (0.0, 1.0)),
            mu=dict(mapping or {}).get("mu", 0.0),
            std=dict(mapping or {}).get("std", 1.0),
        )
        denom = _pade_poly_ast(
            base_node,
            dict(mapping or {}).get("denom", (1.0,)),
            mu=dict(mapping or {}).get("mu", 0.0),
            std=dict(mapping or {}).get("std", 1.0),
        )
        return ("div", numer, denom)
    if kind == "exp":
        z = _normalized_feature_expr(base_node, mu=dict(mapping or {}).get("mu", 0.0), std=dict(mapping or {}).get("std", 1.0))
        exp_term = ("exp", ("mul", _const_node(dict(mapping or {}).get("b", 1.0)), z))
        return _add_terms(
            [
                _mul_const_ast(exp_term, dict(mapping or {}).get("a", 1.0)),
                _const_node(dict(mapping or {}).get("c", 0.0)),
            ]
        )
    if kind == "sine":
        z = _normalized_feature_expr(base_node, mu=dict(mapping or {}).get("mu", 0.0), std=dict(mapping or {}).get("std", 1.0))
        wz = ("mul", _const_node(dict(mapping or {}).get("omega", 1.0)), z)
        return _add_terms(
            [
                _mul_const_ast(("sin", wz), dict(mapping or {}).get("A", 0.0)),
                _mul_const_ast(("cos", wz), dict(mapping or {}).get("B", 0.0)),
                _const_node(dict(mapping or {}).get("c", 0.0)),
            ]
        )
    if kind == "power":
        signed = _mul_const_ast(base_node, dict(mapping or {}).get("sgn_f", 1.0))
        if signed is None:
            return None
        exponent = (
            "add",
            _const_node(dict(mapping or {}).get("log_a", 0.0)),
            ("mul", _const_node(dict(mapping or {}).get("b", 1.0)), ("log", signed)),
        )
        out = ("exp", exponent)
        return _mul_const_ast(out, dict(mapping or {}).get("sgn_y", 1.0))
    return None


def _feature_nodes_from_task(
    task: Mapping[str, Any],
    *,
    subproblem_spec: SubproblemSpec | None,
    nvars: int,
    required_count: int,
) -> list[tuple] | None:
    raw_nodes = list(dict(task or {}).get("feature_nodes", ()) or ())
    out = [_normalize_node(node) for node in raw_nodes]
    out = [node for node in out if isinstance(node, tuple)]
    if len(out) >= int(required_count):
        return out[: int(required_count)]
    active_vars = tuple(int(v) for v in tuple(getattr(subproblem_spec, "active_vars", ()) or ()))
    fallback_vars = [idx for idx in active_vars if 0 <= int(idx) < int(nvars)]
    if len(fallback_vars) < int(required_count):
        fallback_vars = list(range(max(0, int(required_count))))
    if len(fallback_vars) < int(required_count):
        return None
    return [("var", int(idx)) for idx in fallback_vars[: int(required_count)]]


def _task_metadata(
    subproblem_spec: SubproblemSpec | None,
    *,
    lift_route_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _merged_subproblem_context(subproblem_spec)
    task = dict(metadata.get("constant_lift_task", {}) or {})
    if not task:
        values = metadata.get("constant_lift_values_by_regime", None)
        if isinstance(values, Mapping):
            task = {
                "values_by_regime": dict(values),
                "regime_ids": list(metadata.get("constant_lift_regime_ids", ()) or ()),
                "dataset_metadata": metadata.get("constant_lift_dataset_metadata", None),
                "feature_nodes": list(metadata.get("constant_lift_feature_nodes", ()) or ()),
                "constant_name": str(metadata.get("constant_lift_constant_name", "") or ""),
                "task_source": "metadata",
            }
    if not task:
        route_context = dict(lift_route_context or {})
        task = _synthesize_constant_lift_task(
            metadata,
            route_signal=dict(route_context.get("constant_lift", {}) or {}),
        )
    elif "task_source" not in task:
        task["task_source"] = "metadata"
    return task


def _affine_fallback_candidate(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    ones = torch.ones((int(x.shape[0]), 1), dtype=x.dtype, device=x.device)
    design = torch.cat([ones, x], dim=1)
    coeffs = torch.linalg.lstsq(design, y).solution.squeeze(-1)
    y_hat = design @ coeffs.reshape(-1, 1)
    expr_ast = _affine_ast(coeffs.detach().cpu().tolist())
    return {
        "solver": "affine_fallback",
        "expr": str(node_str(expr_ast)),
        "expr_ast": expr_ast,
        "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
        "mse_raw": float(_weighted_mse(y, y_hat)),
        "mse_eff": float(_weighted_mse(y, y_hat)),
    }


def solve_constant_lift_task(
    *,
    regime_ids: Sequence[str],
    values_by_regime: Mapping[str, Any],
    dataset_metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    dtype: torch.dtype = torch.float64,
    n_iter: int = 1500,
    max_depth: int = 3,
    poly_degree: int = 2,
    return_topk: int = 1,
) -> dict[str, Any] | None:
    ids = [str(item) for item in list(regime_ids or []) if str(item) in dict(values_by_regime or {})]
    if len(ids) < 2:
        return None
    values: list[float] = []
    kept_ids: list[str] = []
    for regime_id in ids:
        scalar = _safe_float(dict(values_by_regime or {}).get(str(regime_id), None))
        if scalar is None:
            continue
        kept_ids.append(str(regime_id))
        values.append(float(scalar))
    if len(kept_ids) < 2:
        return None
    x = build_regime_feature_matrix(
        kept_ids,
        dataset_metadata=dataset_metadata,
        dtype=dtype,
    )
    x_tensor, feature_names, feature_source = x
    y_tensor = torch.tensor(values, dtype=dtype).reshape(-1, 1)
    baseline_mse = _const_baseline_mse(y_tensor)

    seed_digest = hashlib.sha1(
        "|".join(kept_ids).encode("utf-8", errors="ignore")
    ).hexdigest()
    search_seed = int(seed_digest[:8], 16)
    results: list[dict[str, Any]] = []
    try:
        results = list(
            run_explorer(
                nvars=int(x_tensor.shape[1]),
                n_iter=max(128, int(n_iter)),
                max_depth=max(1, int(max_depth)),
                poly_degree=max(1, int(poly_degree)),
                seed=int(search_seed),
                dtype=dtype,
                x_fit_data=x_tensor,
                y_fit_data=y_tensor,
                x_probe_data=x_tensor,
                y_probe_data=y_tensor,
                return_topk=max(1, int(return_topk)),
                simplify_skeletons=False,
                no_residual=True,
                no_crossover=True,
                print_every=0,
                verbose=False,
            )
            or []
        )
    except Exception:
        results = []

    candidates: list[dict[str, Any]] = []
    if results:
        best = dict(results[0])
        candidates.append(
            {
                "solver": "factorized_search",
                "expr": str(best.get("expr", "") or ""),
                "expr_ast": best.get("toy_ast", None),
                "mapping": _jsonable(dict(best.get("mapping", {}) or {})),
                "mse_raw": float(best.get("mse_raw", best.get("mse", float("inf")))),
                "mse_eff": float(best.get("mse_eff", best.get("mse_raw", best.get("mse", float("inf"))))),
            }
        )
    try:
        candidates.append(_affine_fallback_candidate(x_tensor, y_tensor))
    except Exception:
        pass
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda row: float(row.get("mse_eff", row.get("mse_raw", float("inf")))),
    )
    best_mse = float(best.get("mse_eff", best.get("mse_raw", float("inf"))))
    return {
        "solver": str(best.get("solver", "")),
        "expr": str(best.get("expr", "") or ""),
        "expr_ast": _jsonable(best.get("expr_ast", None)),
        "mapping": _jsonable(dict(best.get("mapping", {}) or {})),
        "fit_mse": float(best_mse),
        "probe_mse": float(best_mse),
        "baseline_mse": float(baseline_mse),
        "improvement_ratio": (
            None
            if not math.isfinite(float(best_mse))
            else float(baseline_mse / max(1.0e-12, float(best_mse)))
        ),
        "regime_ids": list(kept_ids),
        "regime_points": _jsonable(x_tensor),
        "regime_values": [float(v) for v in values],
        "feature_names": [str(name) for name in feature_names],
        "feature_source": str(feature_source),
        "search_seed": int(search_seed),
    }


@torch.no_grad()
def solve_local_constant_lift_preview_rows(
    *,
    parent_node,
    spec_payload: Mapping[str, Any],
    path: Sequence[int],
    target_mode: str,
    target_mapping_kind: str,
    beam_rank: int,
    slate_id: str,
    path_gain: float,
    max_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims=None,
    local_score_mode: str = "affine",
    preview_topk: int = 2,
    max_subtree_depth: int | None = None,
    constant_lift_topk: int = 2,
    lift_route_context: Mapping[str, Any] | None = None,
    witness_loss_enable: bool = False,
    witness_grad_weight: float = 0.0,
    witness_d2_weight: float = 0.0,
    witness_diag_weight: float = 0.0,
    witness_physics_weight: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    route_preview_topk = max(1, int(preview_topk))
    hole_path = tuple(int(v) for v in tuple(path or ()))
    mode_name = _normalize_inverse_local_score_mode(local_score_mode)
    solver_meta: dict[str, Any] = {
        "status": "started",
        "requested_preview_topk": int(route_preview_topk),
        "constant_lift_topk": int(max(1, int(constant_lift_topk))),
        "preview_count": 0,
        "candidate_count_scored": 0,
    }
    problem, continuation_frames, hole_sub, subproblem_spec = _local_problem_from_payload(spec_payload)
    route_context = dict(lift_route_context or {})
    if not route_context:
        route_context = build_local_lift_route_context(spec_payload)
    route_signal = dict(route_context.get("constant_lift", {}) or {})
    solver_meta["route_trigger_status"] = str(route_signal.get("status", "") or "")
    solver_meta["route_trigger_score"] = _safe_float(route_signal.get("score", None))
    solver_meta["route_trigger_preferred"] = bool(route_signal.get("preferred", False))
    solver_meta["route_reason_family"] = str(route_signal.get("reason_family", "") or "")
    solver_meta["expanded_family_evidence"] = dict(route_context.get("expanded_family_evidence", {}) or {})
    if problem is None:
        solver_meta["status"] = "invalid_local_problem"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    task = _task_metadata(subproblem_spec, lift_route_context=route_context)
    values_by_regime = dict(task.get("values_by_regime", {}) or {})
    regime_ids = [str(item) for item in list(task.get("regime_ids", ()) or values_by_regime.keys()) if str(item) in values_by_regime]
    if len(regime_ids) < 2 or not values_by_regime:
        solver_meta["status"] = "no_constant_lift_task"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    solver_meta["task_source"] = str(task.get("task_source", "") or "")
    if _safe_float(task.get("mean_cv", None)) is not None:
        solver_meta["mean_cv"] = float(task["mean_cv"])
    if _safe_float(task.get("trigger_mean_cv", None)) is not None:
        solver_meta["trigger_mean_cv"] = float(task["trigger_mean_cv"])
    if task.get("parameter_stability", None) is not None:
        solver_meta["parameter_stability"] = _jsonable(task.get("parameter_stability", None))
    lift = solve_constant_lift_task(
        regime_ids=regime_ids,
        values_by_regime=values_by_regime,
        dataset_metadata=task.get("dataset_metadata", None),
        dtype=problem.xf.dtype,
        poly_degree=int(poly_degree),
        return_topk=max(1, int(constant_lift_topk)),
    )
    if not isinstance(lift, Mapping):
        solver_meta["status"] = "constant_lift_failed"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    raw_expr = _normalize_node(lift.get("expr_ast", None))
    if raw_expr is None:
        solver_meta["status"] = "invalid_lift_expr_ast"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    max_var_idx = -1
    stack = [raw_expr]
    while stack:
        node = stack.pop()
        if not isinstance(node, tuple) or not node:
            continue
        if str(node[0]) == "var":
            max_var_idx = max(max_var_idx, int(node[1]))
        elif str(node[0]) in UNARY_OPS and len(node) >= 2:
            stack.append(node[1])
        elif str(node[0]) in BINARY_OPS and len(node) >= 3:
            stack.append(node[1])
            stack.append(node[2])
    feature_count = max(0, int(max_var_idx) + 1)
    feature_nodes = _feature_nodes_from_task(task, subproblem_spec=subproblem_spec, nvars=int(nvars), required_count=int(feature_count))
    if feature_nodes is None:
        solver_meta["status"] = "missing_feature_nodes"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    try:
        base_expr = _compose_feature_expr(raw_expr, feature_nodes)
    except Exception:
        solver_meta["status"] = "feature_compose_failed"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    mapped_expr = _mapping_to_ast(base_expr, dict(lift.get("mapping", {}) or {}))
    if mapped_expr is None or not is_valid_node(mapped_expr):
        solver_meta["status"] = "unsupported_mapping_kind"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}

    base_node = hole_sub
    if not isinstance(base_node, tuple) or not base_node:
        base_node = None
    ctx = _build_solver_context(
        parent_node=parent_node,
        hole_path=hole_path,
        hole_sub=base_node if isinstance(base_node, tuple) else mapped_expr,
        max_depth=int(max_depth),
        max_subtree_depth=int(max_subtree_depth if max_subtree_depth is not None else max_depth),
        nvars=int(nvars),
        poly_degree=int(poly_degree),
        var_dims=var_dims,
        pool_nodes=[],
        pool_dims=[],
        local_score_mode=str(mode_name),
        target_mode=str(target_mode or ""),
        target_mapping_kind=str(target_mapping_kind or ""),
        witness_loss_enable=bool(witness_loss_enable),
        witness_grad_weight=float(witness_grad_weight),
        witness_d2_weight=float(witness_d2_weight),
        witness_diag_weight=float(witness_diag_weight),
        witness_physics_weight=float(witness_physics_weight),
    )
    scored = _score_node_against_problem(
        mapped_expr,
        problem=problem,
        ctx=ctx,
        source="constant_lift_route",
        generation_kind="constant_lift_route",
    )
    if (
        scored is None
        and str(mode_name) in ("affine", "strict")
        and min(int(problem.xf.shape[0]), int(problem.xp.shape[0])) < 4
    ):
        fallback_ctx = replace(ctx, local_score_mode="fitbest")
        scored = _score_node_against_problem(
            mapped_expr,
            problem=problem,
            ctx=fallback_ctx,
            source="constant_lift_route",
            generation_kind="constant_lift_route",
        )
    if scored is None:
        solver_meta["status"] = "constant_lift_unscored"
        solver_meta["wall_seconds"] = float(time.perf_counter() - started)
        return {"rows": [], "solver_meta": solver_meta}
    payload = dict(scored.payload or {})
    payload.update(
        {
            "constant_name": str(task.get("constant_name", "") or ""),
            "feature_names": [str(item) for item in list(lift.get("feature_names", []) or [])],
            "feature_source": str(lift.get("feature_source", "") or ""),
            "improvement_ratio": _safe_float(lift.get("improvement_ratio", None)),
            "solver": str(lift.get("solver", "") or ""),
            "regime_ids": [str(item) for item in list(lift.get("regime_ids", []) or [])],
        }
    )
    scored = replace(scored, family="constant_lift_route", payload=payload)
    deduped = _dedup_scored_candidates([scored], complexity_penalty=0.0)

    preview_rows: list[dict[str, Any]] = []
    for cand in deduped[:route_preview_topk]:
        try:
            wrapped_node = _apply_continuation_frames(cand.node, continuation_frames)
        except Exception:
            continue
        wrapped_cand = replace(cand, node=wrapped_node, generation_kind=f"followup:{str(cand.generation_kind)}")
        row = _candidate_to_preview_row(
            wrapped_cand,
            parent_node=parent_node,
            beam_state={
                "sub": hole_sub,
                "target_mode": str(target_mode or ""),
                "target_mapping_kind": str(target_mapping_kind or ""),
                "path_gain": float(path_gain),
                "poly_degree": int(poly_degree),
            },
            beam_rank=int(beam_rank),
            slate_id=str(slate_id),
            path=hole_path,
            xf=problem.xf,
            tf=problem.tf,
            xp=problem.xp,
            tp=problem.tp,
            max_depth=int(max_depth),
            var_dims=var_dims,
            local_score_mode=str(mode_name),
        )
        if row is None:
            continue
        row["proposal_family"] = "constant_lift_route"
        row["generation_source"] = "constant_lift_route"
        row["tuple_provenance"] = "constant_lift_route"
        row["constant_lift_constant_name"] = str(task.get("constant_name", "") or "")
        row["constant_lift_feature_names"] = [str(item) for item in list(lift.get("feature_names", []) or [])]
        row["constant_lift_feature_source"] = str(lift.get("feature_source", "") or "")
        row["constant_lift_improvement_ratio"] = _safe_float(lift.get("improvement_ratio", None))
        row["constant_lift_expr_ast"] = _ast_from_jsonable(lift.get("expr_ast", None))
        row["constant_lift_solver"] = str(lift.get("solver", "") or "")
        row["constant_lift_route_status"] = str(route_signal.get("status", "") or "")
        row["constant_lift_route_score"] = _safe_float(route_signal.get("score", None))
        row["constant_lift_route_reason_family"] = str(route_signal.get("reason_family", "") or "")
        preview_rows.append(row)

    def _row_loss(row: Mapping[str, Any], key: str, fallback_key: str | None = None) -> float:
        value = _safe_float(row.get(key, None))
        if value is not None:
            return float(value)
        if fallback_key is not None:
            fallback = _safe_float(row.get(fallback_key, None))
            if fallback is not None:
                return float(fallback)
        return float("inf")

    preview_rows.sort(
        key=lambda row: (
            _row_loss(row, "witness_energy_total", "local_probe_mse"),
            _row_loss(row, "local_fit_mse"),
        )
    )
    preview_rows = preview_rows[:route_preview_topk]
    for local_rank, row in enumerate(preview_rows):
        row["local_rank"] = int(local_rank)
        row["local_candidate_count"] = int(len(preview_rows))
    solver_meta["candidate_count_scored"] = int(len(deduped))
    solver_meta["preview_count"] = int(len(preview_rows))
    solver_meta["constant_name"] = str(task.get("constant_name", "") or "")
    solver_meta["feature_source"] = str(lift.get("feature_source", "") or "")
    solver_meta["status"] = "ok" if preview_rows else "no_constant_lift_candidates"
    solver_meta["wall_seconds"] = float(time.perf_counter() - started)
    return {"rows": preview_rows, "solver_meta": solver_meta}


__all__ = [
    "build_regime_feature_matrix",
    "solve_local_constant_lift_preview_rows",
    "solve_constant_lift_task",
]
