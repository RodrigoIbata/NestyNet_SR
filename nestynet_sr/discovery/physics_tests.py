# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from nestynet_sr.sr_core.bridges import AddNode, ConstNode, CosNode, ExpNode, LogNode, MulNode, PowNode, SinNode, Var
from nestynet_sr.sr_core.units import UnitsSpec, check_units_ast
from nestynet_sr.sr_search.factorized_search.expr_ast import is_valid_node as is_valid_tuple_ast


@dataclass(frozen=True)
class PhysicsCheckResult:
    name: str
    passed: bool | None
    score: float | None
    details: Mapping[str, Any] = field(default_factory=dict)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        try:
            values = values.detach().cpu().reshape(-1).tolist()
        except Exception:
            values = [float(values)]
    if isinstance(values, Mapping):
        return [_safe_float(value) for value in values.values() if math.isfinite(_safe_float(value))]
    if isinstance(values, (str, bytes)):
        return []
    if isinstance(values, Sequence):
        out: list[float] = []
        for value in values:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                out.extend(_to_float_list(value))
                continue
            scalar = _safe_float(value)
            if math.isfinite(scalar):
                out.append(float(scalar))
        return out
    scalar = _safe_float(values)
    return [] if not math.isfinite(scalar) else [float(scalar)]


def _mean(values: Sequence[float]) -> float | None:
    xs = [float(value) for value in values if math.isfinite(float(value))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _variance(values: Sequence[float]) -> float | None:
    xs = [float(value) for value in values if math.isfinite(float(value))]
    if len(xs) < 2:
        return 0.0 if xs else None
    mean = float(sum(xs) / len(xs))
    return float(sum((value - mean) ** 2 for value in xs) / len(xs))


def _pearson_abs(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    if x_mean is None or y_mean is None:
        return None
    x_centered = [float(x) - float(x_mean) for x in xs]
    y_centered = [float(y) - float(y_mean) for y in ys]
    x_var = sum(value * value for value in x_centered)
    y_var = sum(value * value for value in y_centered)
    if x_var <= 1.0e-30 or y_var <= 1.0e-30:
        return 0.0
    cov = sum(xv * yv for xv, yv in zip(x_centered, y_centered))
    return abs(float(cov) / math.sqrt(x_var * y_var))


def _tuple_ast_to_bridge_node(node: Any) -> Any:
    op = str(node[0])
    if op == "var":
        return Var(int(node[1]))
    if op == "const":
        return ConstNode(float(node[1]))
    if op == "sin":
        return SinNode(_tuple_ast_to_bridge_node(node[1]))
    if op == "cos":
        return CosNode(_tuple_ast_to_bridge_node(node[1]))
    if op == "exp":
        return ExpNode(_tuple_ast_to_bridge_node(node[1]))
    if op == "log":
        return LogNode(_tuple_ast_to_bridge_node(node[1]))
    if op == "sqrt":
        return PowNode(_tuple_ast_to_bridge_node(node[1]), 0.5)
    if op == "sqr":
        return PowNode(_tuple_ast_to_bridge_node(node[1]), 2.0)
    if op == "neg":
        return MulNode(ConstNode(-1.0), _tuple_ast_to_bridge_node(node[1]))
    if op == "add":
        return AddNode(_tuple_ast_to_bridge_node(node[1]), _tuple_ast_to_bridge_node(node[2]))
    if op == "sub":
        return AddNode(_tuple_ast_to_bridge_node(node[1]), MulNode(ConstNode(-1.0), _tuple_ast_to_bridge_node(node[2])))
    if op == "mul":
        return MulNode(_tuple_ast_to_bridge_node(node[1]), _tuple_ast_to_bridge_node(node[2]))
    if op == "div":
        return MulNode(_tuple_ast_to_bridge_node(node[1]), PowNode(_tuple_ast_to_bridge_node(node[2]), -1.0))
    raise ValueError(f"unsupported tuple AST op {op!r}")


def _coerce_expr(candidate: Any) -> Any:
    if isinstance(candidate, Mapping):
        expr = candidate.get("symbolic_structure", candidate.get("expr", candidate.get("law", None)))
    else:
        expr = candidate
    if expr is None:
        return None
    if is_valid_tuple_ast(expr):
        return _tuple_ast_to_bridge_node(expr)
    return expr


def check_dimensional_consistency(candidate: Any, units_spec: UnitsSpec | None) -> PhysicsCheckResult:
    if units_spec is None:
        return PhysicsCheckResult(
            name="dimensional_consistency",
            passed=None,
            score=None,
            details={"reason": "units_spec missing"},
        )
    expr = _coerce_expr(candidate)
    if expr is None:
        return PhysicsCheckResult(
            name="dimensional_consistency",
            passed=False,
            score=0.0,
            details={"reason": "candidate law missing"},
        )
    result = check_units_ast(expr, units_spec)
    passed = bool(getattr(result, "ok", False))
    return PhysicsCheckResult(
        name="dimensional_consistency",
        passed=passed,
        score=1.0 if passed else 0.0,
        details={"reason": str(getattr(result, "reason", ""))},
    )


def check_residual_structure(
    residuals: Any,
    *,
    regressors: Sequence[Any] | Mapping[str, Any] | None = None,
    max_abs_correlation: float = 0.2,
) -> PhysicsCheckResult:
    residual_vec = _to_float_list(residuals)
    if not residual_vec:
        return PhysicsCheckResult(
            name="residual_structure",
            passed=None,
            score=None,
            details={"reason": "residuals missing"},
        )
    corr_values: list[float] = []
    if isinstance(regressors, Mapping):
        reg_iter = list(regressors.values())
    else:
        reg_iter = list(regressors or [])
    for regressor in reg_iter:
        reg_vec = _to_float_list(regressor)
        corr = _pearson_abs(residual_vec, reg_vec)
        if corr is not None and math.isfinite(corr):
            corr_values.append(float(corr))
    lag_corr = _pearson_abs(residual_vec[:-1], residual_vec[1:]) if len(residual_vec) > 2 else None
    if lag_corr is not None and math.isfinite(lag_corr):
        corr_values.append(float(lag_corr))
    max_corr = 0.0 if not corr_values else max(corr_values)
    score = max(0.0, 1.0 - float(max_corr))
    passed = bool(float(max_corr) <= float(max_abs_correlation))
    return PhysicsCheckResult(
        name="residual_structure",
        passed=passed,
        score=float(score),
        details={"max_abs_correlation": float(max_corr)},
    )


def check_parameter_stability(
    parameter_samples: Sequence[Any] | Mapping[str, Any] | None,
    *,
    max_mean_cv: float = 0.5,
) -> PhysicsCheckResult:
    if parameter_samples is None:
        return PhysicsCheckResult(
            name="parameter_stability",
            passed=None,
            score=None,
            details={"reason": "parameter samples missing"},
        )
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
            if math.isfinite(scalar):
                grouped.setdefault(str(key), []).append(float(scalar))
    if not grouped:
        return PhysicsCheckResult(
            name="parameter_stability",
            passed=None,
            score=None,
            details={"reason": "no finite parameter samples"},
        )
    cvs: list[float] = []
    parameter_cvs: dict[str, float] = {}
    parameter_sample_counts: dict[str, int] = {}
    for key, values in grouped.items():
        mean = _mean(values)
        var = _variance(values)
        if mean is None or var is None:
            continue
        std = math.sqrt(max(0.0, float(var)))
        cv = float(std) / max(1.0e-12, abs(float(mean)))
        cvs.append(float(cv))
        parameter_cvs[str(key)] = float(cv)
        parameter_sample_counts[str(key)] = int(len(values))
    mean_cv = 0.0 if not cvs else float(sum(cvs) / len(cvs))
    score = 1.0 / (1.0 + float(mean_cv))
    return PhysicsCheckResult(
        name="parameter_stability",
        passed=bool(float(mean_cv) <= float(max_mean_cv)),
        score=float(score),
        details={
            "mean_cv": float(mean_cv),
            "n_parameters": int(len(grouped)),
            "parameter_cvs": parameter_cvs,
            "parameter_sample_counts": parameter_sample_counts,
        },
    )


def check_regime_generalization(
    *,
    train_error: Any,
    validation_error: Any = None,
    regime_errors: Sequence[Any] | None = None,
    max_error_ratio: float = 3.0,
) -> PhysicsCheckResult:
    train = _safe_float(train_error)
    val = _safe_float(validation_error)
    regimes = [_safe_float(value) for value in list(regime_errors or []) if math.isfinite(_safe_float(value))]
    if not math.isfinite(train):
        return PhysicsCheckResult(
            name="regime_generalization",
            passed=None,
            score=None,
            details={"reason": "train_error missing"},
        )
    comparators = [float(value) for value in regimes]
    if math.isfinite(val):
        comparators.append(float(val))
    if not comparators:
        return PhysicsCheckResult(
            name="regime_generalization",
            passed=None,
            score=None,
            details={"reason": "validation/regime errors missing"},
        )
    worst_ratio = max(float(value) / max(1.0e-12, float(train)) for value in comparators)
    score = 1.0 / (1.0 + max(0.0, float(worst_ratio) - 1.0))
    return PhysicsCheckResult(
        name="regime_generalization",
        passed=bool(float(worst_ratio) <= float(max_error_ratio)),
        score=float(score),
        details={"worst_error_ratio": float(worst_ratio)},
    )


def _run_symmetry_tests(
    candidate: Any,
    symmetry_tests: Sequence[Callable[[Any], Any] | Mapping[str, Any]] | None,
) -> list[PhysicsCheckResult]:
    out: list[PhysicsCheckResult] = []
    for idx, test in enumerate(list(symmetry_tests or [])):
        if callable(test):
            value = test(candidate)
            if isinstance(value, PhysicsCheckResult):
                out.append(value)
                continue
            if isinstance(value, Mapping):
                out.append(
                    PhysicsCheckResult(
                        name=str(value.get("name", f"symmetry_{int(idx)}")),
                        passed=value.get("passed", None),
                        score=value.get("score", None),
                        details=dict(value.get("details", {}) or {}),
                    )
                )
                continue
            passed = bool(value)
            out.append(
                PhysicsCheckResult(
                    name=f"symmetry_{int(idx)}",
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    details={},
                )
            )
            continue
        payload = dict(test)
        out.append(
            PhysicsCheckResult(
                name=str(payload.get("name", f"symmetry_{int(idx)}")),
                passed=payload.get("passed", None),
                score=payload.get("score", None),
                details=dict(payload.get("details", {}) or {}),
            )
        )
    return out


def score_physics_consistency(
    candidate: Any,
    *,
    units_spec: UnitsSpec | None = None,
    residuals: Any = None,
    regressors: Sequence[Any] | Mapping[str, Any] | None = None,
    parameter_samples: Sequence[Any] | Mapping[str, Any] | None = None,
    train_error: Any = None,
    validation_error: Any = None,
    regime_errors: Sequence[Any] | None = None,
    symmetry_tests: Sequence[Callable[[Any], Any] | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata = dict(candidate.get("metadata", {}) or {}) if isinstance(candidate, Mapping) else {}
    resolved_units_spec = units_spec if units_spec is not None else metadata.get("units_spec", None)
    resolved_residuals = residuals if residuals is not None else metadata.get("residuals", None)
    resolved_regressors = regressors if regressors is not None else metadata.get("residual_regressors", None)
    resolved_parameter_samples = (
        parameter_samples
        if parameter_samples is not None
        else metadata.get("parameter_samples", metadata.get("bootstrap_constants", None))
    )
    resolved_train_error = train_error if train_error is not None else (
        candidate.get("train_error", metadata.get("train_error", None)) if isinstance(candidate, Mapping) else None
    )
    resolved_validation_error = validation_error if validation_error is not None else (
        candidate.get("validation_error", metadata.get("validation_error", None)) if isinstance(candidate, Mapping) else None
    )
    resolved_regime_errors = regime_errors if regime_errors is not None else metadata.get("regime_errors", None)
    checks = [
        check_dimensional_consistency(candidate, resolved_units_spec),
        check_residual_structure(resolved_residuals, regressors=resolved_regressors),
        check_parameter_stability(resolved_parameter_samples),
        check_regime_generalization(
            train_error=resolved_train_error,
            validation_error=resolved_validation_error,
            regime_errors=resolved_regime_errors,
        ),
    ]
    checks.extend(_run_symmetry_tests(candidate, symmetry_tests if symmetry_tests is not None else metadata.get("symmetry_tests", None)))
    available_scores = [float(check.score) for check in checks if check.score is not None and math.isfinite(float(check.score))]
    overall_score = 1.0 if not available_scores else float(sum(available_scores) / len(available_scores))
    overall_passed = not any(check.passed is False for check in checks)
    return {
        "overall_score": float(overall_score),
        "passed": bool(overall_passed),
        "checks": {check.name: asdict(check) for check in checks},
    }


__all__ = [
    "PhysicsCheckResult",
    "check_dimensional_consistency",
    "check_parameter_stability",
    "check_regime_generalization",
    "check_residual_structure",
    "score_physics_consistency",
]
