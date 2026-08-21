# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Mapping, Sequence

import torch

from nestynet_sr.discovery import (
    CommitteeMember,
    ExperimentCandidate,
    apply_constant_lift_proposals,
    build_committee_state,
    discover_constant_lifts,
    parameter_samples_from_local_constants,
    resolve_surface_disagreement_mode,
    score_physics_consistency,
    select_next_experiment,
)
from nestynet_sr.discovery.experiment_opt import optimize_continuous_experiment_candidates
from nestynet_sr.discovery.integration import (
    serialize_committee_member,
    serialize_experiment_candidate,
)
from nestynet_sr.discovery.witness import capture_symbolic_witness

from .bridge import dims_to_units_spec
from .expr_ast import eval_node, is_valid_node
from .research_profiles import (
    RESEARCH_PROFILE_NAMES,
    apply_research_profile_overrides,
    resolve_discovery_research_profile,
)
from .expr_mapping import eval_mapping
from .oracle_lab import EquationSpec, load_equation_spec


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _dtype_from_name(name: str | None) -> torch.dtype:
    token = str(name or "float64").strip().lower()
    if token in ("float32", "fp32", "f32"):
        return torch.float32
    if token in ("float64", "fp64", "f64", "double"):
        return torch.float64
    raise ValueError(f"unknown dtype: {name!r}")


def _load_json(path: str | pathlib.Path) -> dict[str, Any]:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(payload: dict[str, Any], path: str | pathlib.Path) -> None:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    return str(value)


_VALUE_SCORE_KEYS: tuple[str, ...] = (
    "local_probe_mse",
    "mse_eff",
    "mse",
    "preview_loss",
    "global_probe_mse",
    "validation_error",
)
_ROUTE_NAME_KEYS: tuple[str, ...] = (
    "solver_market_route",
    "proposal_family",
    "generation_source",
    "tuple_provenance",
)
_ROUTE_NAME_ALIASES: dict[str, str] = {
    "recursive_local_sr": "local_recursive_sr",
}
_INTERESTING_ROUTES: tuple[str, ...] = (
    "local_recursive_sr",
    "coordinate_lift",
    "tangent_edit",
    "soft_edit_search",
)


def _finite_or_none(value: Any) -> float | None:
    scalar = _safe_float(value, float("nan"))
    return None if not math.isfinite(scalar) else float(scalar)


def _canonical_route_name(row: Mapping[str, Any]) -> str:
    for key in _ROUTE_NAME_KEYS:
        token = str(row.get(key, "") or "").strip().lower()
        if not token:
            continue
        return str(_ROUTE_NAME_ALIASES.get(token, token))
    return ""


def _value_score_from_row(row: Mapping[str, Any]) -> float | None:
    for key in _VALUE_SCORE_KEYS:
        score = _finite_or_none(row.get(key, None))
        if score is not None:
            return float(score)
    return None


def _report_activation_from_results(rows: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    route_usage: dict[str, int] = {}
    fit_jet_source_counts: dict[str, int] = {}
    probe_jet_source_counts: dict[str, int] = {}
    jet_source_counts: dict[str, int] = {}
    witness_teacher_row_count = 0
    witness_component_row_count = 0
    exact_jet_row_count = 0
    numeric_jet_fallback_row_count = 0
    rows_with_jet_provenance_count = 0
    value_ranked: list[tuple[int, float]] = []
    witness_ranked: list[tuple[int, float]] = []
    for idx, raw in enumerate(list(rows or [])):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        fit_jet_source = str(row.get("witness_fit_jet_source", "") or "").strip().lower()
        probe_jet_source = str(row.get("witness_probe_jet_source", "") or "").strip().lower()
        row_jet_sources = {src for src in (fit_jet_source, probe_jet_source) if src}
        if fit_jet_source:
            fit_jet_source_counts[fit_jet_source] = int(fit_jet_source_counts.get(fit_jet_source, 0)) + 1
        if probe_jet_source:
            probe_jet_source_counts[probe_jet_source] = int(probe_jet_source_counts.get(probe_jet_source, 0)) + 1
        if row_jet_sources:
            rows_with_jet_provenance_count += 1
            for source in row_jet_sources:
                jet_source_counts[source] = int(jet_source_counts.get(source, 0)) + 1
        if bool(row.get("witness_exact_jet_used", False)):
            exact_jet_row_count += 1
        if bool(row.get("witness_numeric_jet_fallback_used", False)):
            numeric_jet_fallback_row_count += 1
        route_name = _canonical_route_name(row)
        if route_name:
            route_usage[route_name] = int(route_usage.get(route_name, 0)) + 1
        value_score = _value_score_from_row(row)
        witness_total = _finite_or_none(row.get("witness_energy_total", None))
        if value_score is not None:
            value_ranked.append((int(idx), float(value_score)))
        if witness_total is not None:
            witness_teacher_row_count += 1
            witness_ranked.append((int(idx), float(witness_total)))
        if any(
            _finite_or_none(row.get(key, None)) is not None
            for key in (
                "witness_grad_loss",
                "witness_d2_loss",
                "witness_diag_loss",
                "witness_physics_loss",
            )
        ):
            witness_component_row_count += 1
    interesting_route_usage = {
        str(name): int(route_usage.get(name, 0))
        for name in _INTERESTING_ROUTES
    }
    witness_ranking_available = bool(len(value_ranked) >= 2 and len(witness_ranked) >= 2)
    witness_ranking_changed = False
    if witness_ranking_available:
        best_value_idx = min(value_ranked, key=lambda item: (float(item[1]), int(item[0])))[0]
        best_witness_idx = min(witness_ranked, key=lambda item: (float(item[1]), int(item[0])))[0]
        witness_ranking_changed = bool(int(best_value_idx) != int(best_witness_idx))
    return {
        "route_usage": {str(k): int(v) for k, v in sorted(route_usage.items())},
        "interesting_route_usage": interesting_route_usage,
        "fit_jet_source_counts": {str(k): int(v) for k, v in sorted(fit_jet_source_counts.items())},
        "probe_jet_source_counts": {str(k): int(v) for k, v in sorted(probe_jet_source_counts.items())},
        "jet_source_counts": {str(k): int(v) for k, v in sorted(jet_source_counts.items())},
        "rows_with_jet_provenance_count": int(rows_with_jet_provenance_count),
        "exact_jet_row_count": int(exact_jet_row_count),
        "numeric_jet_fallback_row_count": int(numeric_jet_fallback_row_count),
        "witness_teacher_row_count": int(witness_teacher_row_count),
        "witness_component_row_count": int(witness_component_row_count),
        "witness_weighted_ranking_available": bool(witness_ranking_available),
        "witness_weighted_ranking_changed": bool(witness_ranking_changed),
    }


def _run_research_activation_summary(
    *,
    report_results: Sequence[Mapping[str, Any]] | None,
    selection_payload: Mapping[str, Any] | None,
    experiment_candidates: Sequence[ExperimentCandidate] | None,
    constant_lift_summary: Mapping[str, Any] | None,
    research_profile: str,
) -> dict[str, Any]:
    report_activation = _report_activation_from_results(report_results)
    derivative_prediction_members: set[str] = set()
    diagnostic_prediction_members: set[str] = set()
    witness_candidate_count = 0
    for candidate in list(experiment_candidates or []):
        derivative_prediction_members.update(str(k) for k in dict(candidate.derivative_predictions or {}).keys())
        diagnostic_prediction_members.update(str(k) for k in dict(candidate.diagnostic_predictions or {}).keys())
        if candidate.derivative_predictions or candidate.diagnostic_predictions:
            witness_candidate_count += 1
    selected = (
        selection_payload.get("selected", None)
        if isinstance(selection_payload, Mapping)
        else None
    )
    optimization = (
        dict(selection_payload.get("optimization", {}) or {})
        if isinstance(selection_payload, Mapping)
        else {}
    )
    proposal_count = int(dict(constant_lift_summary or {}).get("proposal_count", 0) or 0)
    applied_count = int(
        dict(constant_lift_summary or {}).get(
            "surviving_applied_member_count",
            dict(constant_lift_summary or {}).get("applied_member_count", 0),
        )
        or 0
    )
    route_usage = dict(report_activation.get("route_usage", {}) or {})
    interesting_route_usage = dict(report_activation.get("interesting_route_usage", {}) or {})
    return {
        "research_profile": str(research_profile or "legacy"),
        "selected_experiment_id": None if not isinstance(selected, Mapping) else str(selected.get("experiment_id", "") or ""),
        "witness_mode_selected": bool(
            isinstance(selection_payload, Mapping)
            and str(selection_payload.get("disagreement_mode", "witness") or "witness") == "witness"
        ),
        "witness_capture_active": bool(witness_candidate_count > 0),
        "derivative_prediction_member_count": int(len(derivative_prediction_members)),
        "diagnostic_prediction_member_count": int(len(diagnostic_prediction_members)),
        "witness_candidate_count": int(witness_candidate_count),
        "experiment_optimization_used": bool(int(optimization.get("optimized_candidate_count", 0) or 0) > 0),
        "constant_lift_proposal_count": int(proposal_count),
        "constant_lift_applied_count": int(applied_count),
        "route_usage": {str(k): int(v) for k, v in sorted(route_usage.items())},
        "interesting_route_usage": {str(k): int(v) for k, v in sorted(interesting_route_usage.items())},
        "fit_jet_source_counts": {
            str(k): int(v)
            for k, v in sorted(dict(report_activation.get("fit_jet_source_counts", {}) or {}).items())
        },
        "probe_jet_source_counts": {
            str(k): int(v)
            for k, v in sorted(dict(report_activation.get("probe_jet_source_counts", {}) or {}).items())
        },
        "jet_source_counts": {
            str(k): int(v)
            for k, v in sorted(dict(report_activation.get("jet_source_counts", {}) or {}).items())
        },
        "rows_with_jet_provenance_count": int(report_activation.get("rows_with_jet_provenance_count", 0) or 0),
        "exact_jet_row_count": int(report_activation.get("exact_jet_row_count", 0) or 0),
        "numeric_jet_fallback_row_count": int(report_activation.get("numeric_jet_fallback_row_count", 0) or 0),
        "witness_teacher_row_count": int(report_activation.get("witness_teacher_row_count", 0) or 0),
        "witness_component_row_count": int(report_activation.get("witness_component_row_count", 0) or 0),
        "witness_weighted_ranking_available": bool(report_activation.get("witness_weighted_ranking_available", False)),
        "witness_weighted_ranking_changed": bool(report_activation.get("witness_weighted_ranking_changed", False)),
        "new_stack_active": bool(
            str(research_profile or "legacy") != "legacy"
            or witness_candidate_count > 0
            or int(applied_count) > 0
            or bool(report_activation.get("witness_weighted_ranking_changed", False))
            or any(int(v) > 0 for v in interesting_route_usage.values())
        ),
    }


def _aggregate_research_activation(runs: Sequence[Mapping[str, Any]] | None, *, research_profile: str) -> dict[str, Any]:
    route_usage: dict[str, int] = {}
    interesting_route_usage: dict[str, int] = {str(name): 0 for name in _INTERESTING_ROUTES}
    fit_jet_source_counts: dict[str, int] = {}
    probe_jet_source_counts: dict[str, int] = {}
    jet_source_counts: dict[str, int] = {}
    selected_experiment_counts: dict[str, int] = {}
    witness_mode_run_count = 0
    witness_capture_run_count = 0
    experiment_optimization_run_count = 0
    constant_lift_applied_total = 0
    constant_lift_proposal_total = 0
    witness_teacher_row_total = 0
    witness_component_row_total = 0
    rows_with_jet_provenance_total = 0
    exact_jet_row_total = 0
    numeric_jet_fallback_row_total = 0
    witness_ranking_changed_run_count = 0
    for run in list(runs or []):
        activation = dict(run.get("research_activation", {}) or {})
        if bool(activation.get("witness_mode_selected", False)):
            witness_mode_run_count += 1
        if bool(activation.get("witness_capture_active", False)):
            witness_capture_run_count += 1
        if bool(activation.get("experiment_optimization_used", False)):
            experiment_optimization_run_count += 1
        constant_lift_applied_total += int(activation.get("constant_lift_applied_count", 0) or 0)
        constant_lift_proposal_total += int(activation.get("constant_lift_proposal_count", 0) or 0)
        witness_teacher_row_total += int(activation.get("witness_teacher_row_count", 0) or 0)
        witness_component_row_total += int(activation.get("witness_component_row_count", 0) or 0)
        rows_with_jet_provenance_total += int(activation.get("rows_with_jet_provenance_count", 0) or 0)
        exact_jet_row_total += int(activation.get("exact_jet_row_count", 0) or 0)
        numeric_jet_fallback_row_total += int(activation.get("numeric_jet_fallback_row_count", 0) or 0)
        if bool(activation.get("witness_weighted_ranking_changed", False)):
            witness_ranking_changed_run_count += 1
        selected_experiment_id = str(activation.get("selected_experiment_id", "") or "")
        if selected_experiment_id:
            selected_experiment_counts[selected_experiment_id] = int(
                selected_experiment_counts.get(selected_experiment_id, 0)
            ) + 1
        for key, value in dict(activation.get("route_usage", {}) or {}).items():
            route_usage[str(key)] = int(route_usage.get(str(key), 0)) + int(value or 0)
        for key, value in dict(activation.get("interesting_route_usage", {}) or {}).items():
            interesting_route_usage[str(key)] = int(interesting_route_usage.get(str(key), 0)) + int(value or 0)
        for key, value in dict(activation.get("fit_jet_source_counts", {}) or {}).items():
            fit_jet_source_counts[str(key)] = int(fit_jet_source_counts.get(str(key), 0)) + int(value or 0)
        for key, value in dict(activation.get("probe_jet_source_counts", {}) or {}).items():
            probe_jet_source_counts[str(key)] = int(probe_jet_source_counts.get(str(key), 0)) + int(value or 0)
        for key, value in dict(activation.get("jet_source_counts", {}) or {}).items():
            jet_source_counts[str(key)] = int(jet_source_counts.get(str(key), 0)) + int(value or 0)
    return {
        "research_profile": str(research_profile or "legacy"),
        "n_runs": int(len(list(runs or []))),
        "witness_mode_run_count": int(witness_mode_run_count),
        "witness_capture_run_count": int(witness_capture_run_count),
        "experiment_optimization_run_count": int(experiment_optimization_run_count),
        "constant_lift_proposal_total": int(constant_lift_proposal_total),
        "constant_lift_applied_total": int(constant_lift_applied_total),
        "witness_teacher_row_total": int(witness_teacher_row_total),
        "witness_component_row_total": int(witness_component_row_total),
        "rows_with_jet_provenance_total": int(rows_with_jet_provenance_total),
        "exact_jet_row_total": int(exact_jet_row_total),
        "numeric_jet_fallback_row_total": int(numeric_jet_fallback_row_total),
        "witness_weighted_ranking_changed_run_count": int(witness_ranking_changed_run_count),
        "selected_experiment_counts": {str(k): int(v) for k, v in sorted(selected_experiment_counts.items())},
        "route_usage": {str(k): int(v) for k, v in sorted(route_usage.items())},
        "interesting_route_usage": {str(k): int(v) for k, v in sorted(interesting_route_usage.items())},
        "fit_jet_source_counts": {str(k): int(v) for k, v in sorted(fit_jet_source_counts.items())},
        "probe_jet_source_counts": {str(k): int(v) for k, v in sorted(probe_jet_source_counts.items())},
        "jet_source_counts": {str(k): int(v) for k, v in sorted(jet_source_counts.items())},
    }


def _default_report_path(row: Mapping[str, Any], output_dir: pathlib.Path) -> pathlib.Path:
    spec_id = str(row.get("spec_id", "") or "")
    profile = str(row.get("profile", "current") or "current")
    mode = str(row.get("mode", "") or "")
    budget = int(row.get("budget", 0) or 0)
    repeat = int(row.get("repeat", 0) or 0)
    return output_dir / "individual_reports" / f"{spec_id}.{profile}.{mode}.n{budget}.r{repeat}.json"


def _resolve_report_path(row: Mapping[str, Any], output_dir: pathlib.Path) -> pathlib.Path:
    raw = str(row.get("report_path", "") or "").strip()
    if raw:
        p = pathlib.Path(raw)
        if p.is_file():
            return p
    fallback = _default_report_path(row, output_dir)
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Could not resolve oracle individual report for row={dict(row)}")


def _flatten_mapping_constants(mapping: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in dict(mapping or {}).items():
        name = str(key)
        if name.startswith("_"):
            continue
        scalar = _safe_float(value)
        if math.isfinite(scalar):
            out[name] = float(scalar)
            continue
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                item_f = _safe_float(item)
                if math.isfinite(item_f):
                    out[f"{name}_{int(idx)}"] = float(item_f)
    return out


def _eval_linear_head(head: Mapping[str, Any] | None, x: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(head, Mapping):
        return None
    terms = head.get("terms", None)
    coeffs = head.get("coeffs", None)
    if not isinstance(terms, (list, tuple)) or not isinstance(coeffs, (list, tuple)):
        return None
    if len(coeffs) != len(terms) + 1:
        return None
    try:
        out = torch.full((x.shape[0], 1), float(coeffs[0]), dtype=x.dtype, device=x.device)
    except Exception:
        return None
    for coeff, term in zip(coeffs[1:], terms):
        try:
            term_val = eval_node(term, x)
        except Exception:
            return None
        if not torch.isfinite(term_val).all():
            return None
        out = out + float(coeff) * term_val
    return out


def _eval_mapping_total(pred: torch.Tensor, mapping: Mapping[str, Any], x: torch.Tensor) -> torch.Tensor:
    y_hat = eval_mapping(pred, dict(mapping))
    head_pred = _eval_linear_head(dict(mapping).get("_lin_head", None), x)
    if head_pred is not None and torch.isfinite(head_pred).all():
        y_hat = y_hat + head_pred
    return y_hat


def _member_from_oracle_result(
    result: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    spec: EquationSpec,
    rank: int,
    physics_score: float = 1.0,
) -> CommitteeMember:
    expr_ast = result.get("expr_ast", None)
    expr_text = str(result.get("expr", "") or "")
    structure = expr_ast if expr_ast is not None else expr_text
    size = max(1, int(result.get("size", 1) or 1))
    mapping = dict(result.get("mapping", {}) or {})
    fitted_constants = _flatten_mapping_constants(mapping)
    return CommitteeMember(
        member_id=(
            f"{spec.id}:{str(row.get('profile', 'current'))}:{str(row.get('mode', ''))}:"
            f"b{int(row.get('budget', 0) or 0)}:r{int(row.get('repeat', 0) or 0)}:k{int(rank)}"
        ),
        symbolic_structure=structure,
        fitted_constants=fitted_constants,
        shared_constants={},
        local_constants_by_experiment={str(spec.id): dict(fitted_constants)} if fitted_constants else {},
        train_error=float(result.get("mse", float("nan"))),
        validation_error=float(result.get("mse", float("nan"))),
        simplicity_score=1.0 / float(size),
        physics_consistency_score=float(physics_score),
        metadata={
            "expr": expr_text,
            "expr_ast": _jsonable(expr_ast),
            "mapping": _jsonable(mapping),
            "mapping_kind": str(result.get("mapping_kind", "") or ""),
            "spec_path": str(row.get("spec_path", "") or ""),
            "spec_id": str(spec.id),
        },
    )


def _committee_members_from_report(
    row: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    committee_topk: int,
) -> tuple[EquationSpec, list[CommitteeMember], dict[str, Any]]:
    spec = load_equation_spec(str(row.get("spec_path", "")))
    report_rows = [
        dict(result)
        for result in list(report.get("results", []) or [])[: max(1, int(committee_topk))]
        if isinstance(result, Mapping)
    ]
    units_spec = dims_to_units_spec(
        [tuple(v.dim) for v in spec.variables] + [tuple(c.dim) for c in spec.constants],
        tuple(spec.target_dim),
    )
    members: list[CommitteeMember] = []
    physics_reports: dict[str, Any] = {}
    for rank, result in enumerate(report_rows):
        provisional = _member_from_oracle_result(result, row=row, spec=spec, rank=int(rank), physics_score=1.0)
        physics = score_physics_consistency(
            {
                "symbolic_structure": provisional.symbolic_structure,
                "train_error": provisional.train_error,
                "validation_error": provisional.validation_error,
                "metadata": dict(provisional.metadata or {}),
            },
            units_spec=units_spec,
            parameter_samples=parameter_samples_from_local_constants(
                provisional.local_constants_by_experiment,
                regime_ids=[str(spec.id)],
            ),
        )
        member = _member_from_oracle_result(
            result,
            row=row,
            spec=spec,
            rank=int(rank),
            physics_score=float(physics.get("overall_score", 1.0) or 1.0),
        )
        members.append(member)
        physics_reports[str(member.member_id)] = physics
    return spec, members, physics_reports


def _row_points_tensor(points: Sequence[Sequence[Any]], *, spec: EquationSpec, dtype: torch.dtype) -> torch.Tensor:
    rows = [[float(v) for v in list(row)] for row in list(points or [])]
    if not rows:
        raise ValueError("points experiment requires at least one row")
    n_var = len(spec.variables)
    n_total = n_var + len(spec.constants)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("points experiment rows must have equal width")
    if width not in (n_var, n_total):
        raise ValueError(f"points experiment width must be {n_var} or {n_total}, got {width}")
    x = torch.tensor(rows, dtype=dtype)
    if width == n_var and spec.constants:
        cols = [
            torch.full((x.shape[0], 1), float(const.value), dtype=dtype)
            for const in spec.constants
        ]
        x = torch.cat([x, *cols], dim=1)
    return x


def _lookup_bounds(bounds: Mapping[str, Any], variable_name: str, index: int, default: tuple[float, float]) -> tuple[float, float]:
    for key in (variable_name, str(variable_name), str(index), f"x{int(index)}"):
        if key in bounds:
            raw = list(bounds[key])
            if len(raw) != 2:
                break
            lo = float(raw[0])
            hi = float(raw[1])
            if not lo < hi:
                raise ValueError(f"invalid bounds for {key!r}: {raw!r}")
            return lo, hi
    return default


def _sample_box_tensor(
    entry: Mapping[str, Any],
    *,
    spec: EquationSpec,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_points = max(1, int(entry.get("n_points", 32) or 32))
    seed = int(entry.get("seed", 0) or 0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo_vals: list[float] = []
    hi_vals: list[float] = []
    bounds = dict(entry.get("bounds", {}) or {})
    for idx, variable in enumerate(spec.variables):
        lo, hi = _lookup_bounds(bounds, variable.name, idx, variable.bounds)
        lo_vals.append(float(lo))
        hi_vals.append(float(hi))
    lo_t = torch.tensor(lo_vals, dtype=dtype).reshape(1, len(lo_vals))
    hi_t = torch.tensor(hi_vals, dtype=dtype).reshape(1, len(hi_vals))
    u = torch.rand((n_points, len(lo_vals)), generator=g, dtype=dtype)
    x = lo_t + (hi_t - lo_t) * u
    if spec.constants:
        const_cols = [
            torch.full((n_points, 1), float(const.value), dtype=dtype)
            for const in spec.constants
        ]
        x = torch.cat([x, *const_cols], dim=1)
    return x


def _spec_bounds_list(
    entry: Mapping[str, Any],
    *,
    spec: EquationSpec,
    points_tensor: torch.Tensor | None = None,
) -> list[list[float]]:
    bounds = dict(entry.get("bounds", {}) or {})
    out: list[list[float]] = []
    for idx, variable in enumerate(spec.variables):
        lo, hi = _lookup_bounds(bounds, variable.name, idx, variable.bounds)
        out.append([float(lo), float(hi)])
    for const in list(spec.constants or []):
        value = float(const.value)
        out.append([value, value])
    if not out and points_tensor is not None:
        for idx in range(int(points_tensor.shape[1])):
            lo_v = float(points_tensor[:, idx].min().item())
            hi_v = float(points_tensor[:, idx].max().item())
            if not hi_v > lo_v:
                hi_v = lo_v + 1.0
            out.append([lo_v, hi_v])
    return out


def _experiment_applies(entry: Mapping[str, Any], row: Mapping[str, Any], spec: EquationSpec) -> bool:
    spec_ids = [str(v) for v in list(entry.get("spec_ids", []) or []) if str(v)]
    if spec_ids and str(spec.id) not in spec_ids:
        return False
    profiles = [str(v) for v in list(entry.get("profiles", []) or []) if str(v)]
    if profiles and str(row.get("profile", "")) not in profiles:
        return False
    modes = [str(v) for v in list(entry.get("modes", []) or []) if str(v)]
    if modes and str(row.get("mode", "")) not in modes:
        return False
    budgets = [int(v) for v in list(entry.get("budgets", []) or [])]
    if budgets and int(row.get("budget", 0) or 0) not in budgets:
        return False
    return True


def build_oracle_experiment_candidates(
    spec: EquationSpec,
    row: Mapping[str, Any],
    committee: Sequence[CommitteeMember],
    *,
    experiment_manifest: Mapping[str, Any],
    dtype: torch.dtype,
    witness_capture_enable: bool = False,
    witness_hessian_diag_enable: bool = False,
    diagnostic_set: str = "basic",
) -> list[ExperimentCandidate]:
    witness_enabled = bool(witness_capture_enable)
    candidates: list[ExperimentCandidate] = []
    entries = list(experiment_manifest.get("experiments", experiment_manifest.get("candidates", [])) or [])
    for idx, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        if not _experiment_applies(entry, row, spec):
            continue
        kind = str(entry.get("type", "box") or "box").strip().lower()
        if kind == "points":
            x = _row_points_tensor(entry.get("points", []), spec=spec, dtype=dtype)
        elif kind == "box":
            x = _sample_box_tensor(entry, spec=spec, dtype=dtype)
        else:
            raise ValueError(f"unsupported oracle experiment type {kind!r}")
        observable_predictions: dict[str, Any] = {}
        derivative_predictions: dict[str, Any] = {}
        diagnostic_predictions: dict[str, Any] = {}
        for member in committee:
            expr_ast = dict(member.metadata or {}).get("expr_ast", None)
            mapping = dict(dict(member.metadata or {}).get("mapping", {}) or {})
            if expr_ast is None or not is_valid_node(expr_ast):
                continue
            pred = None
            try:
                pred = _eval_mapping_total(eval_node(expr_ast, x), mapping, x).squeeze(-1)
            except Exception:
                pred = None
            if witness_enabled:
                witness = capture_symbolic_witness(
                    expr_ast=expr_ast,
                    x=x,
                    forward_value_fn=lambda node, xx: _eval_mapping_total(eval_node(node, xx), mapping, xx),
                    capture_gradients=True,
                    capture_hessian_diag=bool(witness_hessian_diag_enable),
                    diagnostic_set=str(diagnostic_set or "basic"),
                )
                pred = witness.get("observable", pred)
            else:
                witness = {}
            if pred is None:
                continue
            observable_predictions[str(member.member_id)] = _jsonable(pred)
            deriv = witness.get("derivative", None) if witness_enabled else None
            if deriv is not None:
                derivative_predictions[str(member.member_id)] = _jsonable(deriv)
            diag = dict(witness.get("diagnostic", {}) or {}) if witness_enabled else {}
            if diag:
                diagnostic_predictions[str(member.member_id)] = _jsonable(diag)
        candidates.append(
            ExperimentCandidate(
                experiment_id=str(entry.get("experiment_id", "") or entry.get("id", "") or f"experiment_{int(idx)}"),
                conditions={
                    "type": kind,
                    "n_points": int(x.shape[0]),
                    "shape": [int(v) for v in x.shape],
                },
                observable_predictions=observable_predictions,
                derivative_predictions=derivative_predictions,
                diagnostic_predictions=diagnostic_predictions,
                cost=float(entry.get("cost", 0.0) or 0.0),
                noise_risk=float(entry.get("noise_risk", 0.0) or 0.0),
                feasibility_penalty=float(entry.get("feasibility_penalty", 0.0) or 0.0),
                metadata={
                    "points_preview": _jsonable(x[: min(4, int(x.shape[0]))]),
                    "continuous_optimizer": {
                        "enabled": True,
                        "source_type": str(kind),
                        "points": _jsonable(x),
                        "bounds": _spec_bounds_list(entry, spec=spec, points_tensor=x),
                    },
                    "witness_capture": {
                        "enabled": bool(witness_enabled),
                        "hessian_diag_enabled": bool(witness_enabled and witness_hessian_diag_enable),
                        "diagnostic_set": str(diagnostic_set or "basic"),
                    },
                },
            )
        )
    return candidates


def run_oracle_discovery_benchmark(
    regression_payload: Mapping[str, Any],
    *,
    output_dir: str | pathlib.Path,
    committee_topk: int = 8,
    max_members: int | None = None,
    experiment_manifest_path: str | pathlib.Path | None = None,
    beta: float = 0.0,
    gamma: float = 0.0,
    disagreement_mode: str | None = None,
    lambda_cost: float = 1.0,
    lambda_noise: float = 1.0,
    lambda_feasibility: float = 1.0,
    dtype: torch.dtype = torch.float64,
    discovery_constant_lift_enable: bool = False,
    discovery_constant_lift_min_regimes: int = 3,
    discovery_constant_lift_trigger_mean_cv: float = 0.5,
    discovery_constant_lift_apply_enable: bool = False,
    discovery_constant_lift_apply_topk: int = 1,
    discovery_constant_lift_min_rel_gain: float = 1.01,
    witness_capture_enable: bool = False,
    witness_hessian_diag_enable: bool = False,
    diagnostic_set: str = "basic",
    experiment_optimize_enable: bool = False,
    experiment_opt_steps: int = 32,
    experiment_opt_lr: float = 0.05,
    experiment_project_mode: str = "nearest_box",
    theory_benchmark_enable: bool = False,
    research_profile: str | None = None,
) -> dict[str, Any]:
    profile_requested = research_profile is not None and str(research_profile).strip() != ""
    if profile_requested:
        resolved_profile, profile_overrides = resolve_discovery_research_profile(research_profile)
    else:
        resolved_profile, profile_overrides = "default", {}
    profile_values = apply_research_profile_overrides(
        {
            "beta": float(beta),
            "gamma": float(gamma),
            "disagreement_mode": disagreement_mode,
            "discovery_constant_lift_enable": bool(discovery_constant_lift_enable),
            "discovery_constant_lift_apply_enable": bool(discovery_constant_lift_apply_enable),
            "discovery_constant_lift_apply_topk": int(max(0, int(discovery_constant_lift_apply_topk))),
            "witness_capture_enable": bool(witness_capture_enable),
            "witness_hessian_diag_enable": bool(witness_hessian_diag_enable),
            "diagnostic_set": str(diagnostic_set or "basic"),
            "experiment_optimize_enable": bool(experiment_optimize_enable),
            "theory_benchmark_enable": bool(theory_benchmark_enable),
        },
        overrides=profile_overrides,
    )
    beta = float(profile_values["beta"])
    gamma = float(profile_values["gamma"])
    disagreement_mode = resolve_surface_disagreement_mode(
        profile_values.get("disagreement_mode", None),
        default_mode="witness",
    )
    discovery_constant_lift_enable = bool(profile_values["discovery_constant_lift_enable"])
    discovery_constant_lift_apply_enable = bool(profile_values["discovery_constant_lift_apply_enable"])
    discovery_constant_lift_apply_topk = int(max(0, int(profile_values["discovery_constant_lift_apply_topk"])))
    witness_capture_enable = bool(profile_values["witness_capture_enable"])
    witness_hessian_diag_enable = bool(profile_values["witness_hessian_diag_enable"])
    diagnostic_set = str(profile_values["diagnostic_set"] or "basic")
    experiment_optimize_enable = bool(profile_values["experiment_optimize_enable"])
    theory_benchmark_enable = bool(profile_values["theory_benchmark_enable"])

    out_dir = pathlib.Path(output_dir)
    manifest_payload = None
    if experiment_manifest_path:
        manifest_payload = _load_json(experiment_manifest_path)
    runs: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    committee_sizes: list[float] = []
    physics_scores: list[float] = []
    theory_metrics: list[dict[str, Any]] = []
    for row in [dict(item) for item in list(regression_payload.get("rows", []) or []) if isinstance(item, Mapping)]:
        report_path = _resolve_report_path(row, out_dir)
        report = _load_json(report_path)
        report_rows = [
            dict(result)
            for result in list(report.get("results", []) or [])
            if isinstance(result, Mapping)
        ]
        spec, members, physics_reports = _committee_members_from_report(
            row,
            report,
            committee_topk=max(1, int(committee_topk)),
        )
        units_spec = dims_to_units_spec(
            [tuple(v.dim) for v in spec.variables] + [tuple(c.dim) for c in spec.constants],
            tuple(spec.target_dim),
        )
        committee = build_committee_state(
            members,
            max_members=max_members,
            deduplicate=True,
        )
        experiment_candidates: list[ExperimentCandidate] = []
        selection_payload = None
        constant_lift_summary = None
        if bool(discovery_constant_lift_enable):
            constant_lift_summary = discover_constant_lifts(
                list(committee.members),
                dataset_ids=[str(spec.id)],
                dataset_metadata=None,
                min_regimes=max(2, int(discovery_constant_lift_min_regimes)),
                trigger_mean_cv=float(discovery_constant_lift_trigger_mean_cv),
                dtype=dtype,
            )
            if bool(discovery_constant_lift_apply_enable):
                applied = apply_constant_lift_proposals(
                    list(committee.members),
                    constant_lift_summary,
                    apply_topk=max(0, int(discovery_constant_lift_apply_topk)),
                    min_rel_gain=float(discovery_constant_lift_min_rel_gain),
                )
                constant_lift_summary = dict(applied.get("summary", {}) or {})
                applied_members: list[CommitteeMember] = []
                for member in list(applied.get("applied_members", []) or []):
                    if not isinstance(member, CommitteeMember):
                        continue
                    physics = score_physics_consistency(
                        {
                            "symbolic_structure": member.symbolic_structure,
                            "train_error": member.train_error,
                            "validation_error": member.validation_error,
                            "metadata": dict(member.metadata or {}),
                        },
                        units_spec=units_spec,
                        parameter_samples=parameter_samples_from_local_constants(
                            member.local_constants_by_experiment,
                            regime_ids=[str(spec.id)],
                        ),
                    )
                    physics_reports[str(member.member_id)] = physics
                    applied_members.append(
                        CommitteeMember(
                            member_id=member.member_id,
                            symbolic_structure=member.symbolic_structure,
                            fitted_constants=member.fitted_constants,
                            shared_constants=member.shared_constants,
                            local_constants_by_experiment=member.local_constants_by_experiment,
                            train_error=member.train_error,
                            validation_error=member.validation_error,
                            regime_holdout_error=member.regime_holdout_error,
                            simplicity_score=member.simplicity_score,
                            physics_consistency_score=float(physics.get("overall_score", 1.0) or 1.0),
                            committee_weight=member.committee_weight,
                            canonical_key=member.canonical_key,
                            display_expr=member.display_expr,
                            metadata=member.metadata,
                        )
                    )
                if applied_members:
                    committee = build_committee_state(
                        list(committee.members) + applied_members,
                        max_members=max_members,
                        deduplicate=True,
                    )
                surviving_applied_ids = [
                    str(member.member_id)
                    for member in committee.members
                    if bool(dict(member.metadata or {}).get("constant_lift_applied", False))
                ]
                constant_lift_summary["surviving_applied_member_count"] = int(len(surviving_applied_ids))
                constant_lift_summary["surviving_applied_member_ids"] = list(surviving_applied_ids)
        committee_sizes.append(float(len(committee.members)))
        physics_scores.extend(
            float(dict(physics_reports.get(str(member.member_id), {}) or {}).get("overall_score", 0.0) or 0.0)
            for member in committee.members
            if math.isfinite(float(dict(physics_reports.get(str(member.member_id), {}) or {}).get("overall_score", 0.0) or 0.0))
        )
        if manifest_payload is not None and committee.members:
            experiment_candidates = build_oracle_experiment_candidates(
                spec,
                row,
                list(committee.members),
                experiment_manifest=manifest_payload,
                dtype=dtype,
                witness_capture_enable=bool(witness_capture_enable),
                witness_hessian_diag_enable=bool(witness_hessian_diag_enable),
                diagnostic_set=str(diagnostic_set or "basic"),
            )
            experiment_optimizer = None
            optimization_result_holder: dict[str, Any] = {}
            if bool(experiment_optimize_enable):
                forward_fns_by_member_id: dict[str, Any] = {}
                for member in committee.members:
                    expr_ast = dict(member.metadata or {}).get("expr_ast", None)
                    mapping = dict(dict(member.metadata or {}).get("mapping", {}) or {})
                    if expr_ast is None or not is_valid_node(expr_ast):
                        continue
                    forward_fns_by_member_id[str(member.member_id)] = (
                        lambda xx, expr_ast=expr_ast, mapping=mapping: _eval_mapping_total(eval_node(expr_ast, xx), mapping, xx)
                    )

                def _experiment_optimizer(current_state, current_candidates, **kwargs):
                    result = optimize_continuous_experiment_candidates(
                        current_state,
                        current_candidates,
                        forward_fns_by_member_id=forward_fns_by_member_id,
                        beta=float(kwargs.get("beta", beta)),
                        gamma=float(kwargs.get("gamma", gamma)),
                        disagreement_mode=resolve_surface_disagreement_mode(
                            kwargs.get("disagreement_mode", disagreement_mode),
                            default_mode=disagreement_mode,
                        ),
                        lambda_cost=float(kwargs.get("lambda_cost", lambda_cost)),
                        lambda_noise=float(kwargs.get("lambda_noise", lambda_noise)),
                        lambda_feasibility=float(kwargs.get("lambda_feasibility", lambda_feasibility)),
                        opt_steps=int(max(1, int(experiment_opt_steps))),
                        opt_lr=float(experiment_opt_lr),
                        project_mode=str(experiment_project_mode or "nearest_box"),
                        include_gradients=bool(witness_capture_enable or float(beta) > 0.0 or str(disagreement_mode) == "witness"),
                        include_diagnostics=bool(float(gamma) > 0.0),
                    )
                    optimization_result_holder["result"] = result
                    return result

                experiment_optimizer = _experiment_optimizer
            selection_payload = select_next_experiment(
                committee,
                experiment_candidates,
                beta=float(beta),
                gamma=float(gamma),
                disagreement_mode=disagreement_mode,
                lambda_cost=float(lambda_cost),
                lambda_noise=float(lambda_noise),
                lambda_feasibility=float(lambda_feasibility),
                optimize_continuous=bool(experiment_optimize_enable),
                experiment_optimizer=experiment_optimizer,
            )
            optimized_candidates = dict(optimization_result_holder.get("result", {}) or {}).get("candidates", None)
            if isinstance(optimized_candidates, Sequence):
                experiment_candidates = list(optimized_candidates)
            selected = selection_payload.get("selected", None)
            if isinstance(selected, Mapping):
                experiment_id = str(selected.get("experiment_id", "") or "")
                if experiment_id:
                    selected_counts[experiment_id] = int(selected_counts.get(experiment_id, 0)) + 1
        theory_benchmark = None
        if bool(theory_benchmark_enable):
            best_member = None if not committee.members else min(
                committee.members,
                key=lambda member: (
                    _safe_float(member.validation_error),
                    -_safe_float(member.physics_consistency_score, 0.0),
                    str(member.member_id),
                ),
            )
            selected = selection_payload.get("selected", None) if isinstance(selection_payload, Mapping) else None
            ranking = list(selection_payload.get("ranking", []) or []) if isinstance(selection_payload, Mapping) else []
            selection_margin = None
            if len(ranking) >= 2:
                selection_margin = float(ranking[0]["score"]) - float(ranking[1]["score"])
            regime_score = None
            if best_member is not None:
                regime_score = _safe_float(
                    dict(physics_reports.get(str(best_member.member_id), {}) or {})
                    .get("checks", {})
                    .get("regime_generalization", {})
                    .get("score", None)
                )
            proposal_count = int(dict(constant_lift_summary or {}).get("proposal_count", 0) or 0)
            applied_count = int(dict(constant_lift_summary or {}).get("surviving_applied_member_count", 0) or 0)
            constant_lift_success = bool(applied_count > 0) if bool(discovery_constant_lift_apply_enable) else bool(proposal_count > 0)
            theory_benchmark = {
                "enabled": True,
                "best_member_id": None if best_member is None else str(best_member.member_id),
                "best_member_validation_error": None if best_member is None else _safe_float(best_member.validation_error),
                "best_member_physics_score": None if best_member is None else _safe_float(best_member.physics_consistency_score),
                "next_experiment_quality": None if not isinstance(selected, Mapping) else _safe_float(selected.get("score", None)),
                "selection_margin": None if selection_margin is None else float(selection_margin),
                "ood_survival_score": None if regime_score is None or not math.isfinite(float(regime_score)) else float(regime_score),
                "constant_lift_success": bool(constant_lift_success),
                "constant_lift_proposal_count": int(proposal_count),
                "constant_lift_applied_count": int(applied_count),
            }
            theory_metrics.append(theory_benchmark)
        research_activation = _run_research_activation_summary(
            report_results=report_rows,
            selection_payload=selection_payload,
            experiment_candidates=experiment_candidates,
            constant_lift_summary=constant_lift_summary,
            research_profile=str(resolved_profile),
        )
        runs.append(
            {
                "spec_id": str(row.get("spec_id", "")),
                "spec_path": str(row.get("spec_path", "")),
                "profile": str(row.get("profile", "current")),
                "mode": str(row.get("mode", "")),
                "budget": int(row.get("budget", 0) or 0),
                "repeat": int(row.get("repeat", 0) or 0),
                "seed": int(row.get("seed", 0) or 0),
                "report_path": str(report_path),
                "committee_members": [serialize_committee_member(member) for member in committee.members],
                "committee_summary": {
                    "member_count": int(len(committee.members)),
                    "canonical_member_count": int(committee.canonical_member_count),
                    "discarded_member_ids": list(committee.discarded_member_ids),
                    "members": [
                        {
                            "member_id": member.member_id,
                            "display_expr": member.display_expr,
                            "validation_error": member.validation_error,
                            "committee_weight": member.committee_weight,
                            "physics_consistency_score": member.physics_consistency_score,
                            "canonical_key": member.canonical_key,
                        }
                        for member in committee.members
                    ],
                },
                "physics_summary": physics_reports,
                "constant_lift_summary": constant_lift_summary,
                "theory_benchmark": theory_benchmark,
                "research_activation": research_activation,
                "experiment_selection": selection_payload,
                "experiment_candidates_full": [
                    serialize_experiment_candidate(candidate)
                    for candidate in experiment_candidates
                ],
                "experiment_candidates": [
                    {
                        "experiment_id": candidate.experiment_id,
                        "conditions": _jsonable(candidate.conditions),
                        "cost": float(candidate.cost),
                        "noise_risk": float(candidate.noise_risk),
                        "feasibility_penalty": float(candidate.feasibility_penalty),
                        "observable_prediction_members": sorted(str(k) for k in candidate.observable_predictions.keys()),
                        "derivative_prediction_members": sorted(str(k) for k in candidate.derivative_predictions.keys()),
                        "diagnostic_prediction_members": sorted(str(k) for k in candidate.diagnostic_predictions.keys()),
                    }
                    for candidate in experiment_candidates
                ],
            }
        )
    aggregate = {
        "n_runs": int(len(runs)),
        "mean_committee_size": None
        if not committee_sizes
        else float(sum(committee_sizes) / len(committee_sizes)),
        "mean_physics_score": None
        if not physics_scores
        else float(sum(physics_scores) / len(physics_scores)),
        "selected_experiment_counts": {str(k): int(v) for k, v in sorted(selected_counts.items())},
        "mean_next_experiment_quality": None
        if not theory_metrics
        else float(
            sum(
                float(item["next_experiment_quality"])
                for item in theory_metrics
                if item.get("next_experiment_quality", None) is not None and math.isfinite(float(item["next_experiment_quality"]))
            )
            / max(
                1,
                sum(
                    1
                    for item in theory_metrics
                    if item.get("next_experiment_quality", None) is not None and math.isfinite(float(item["next_experiment_quality"]))
                ),
            )
        ),
        "mean_ood_survival_score": None
        if not theory_metrics
        else float(
            sum(
                float(item["ood_survival_score"])
                for item in theory_metrics
                if item.get("ood_survival_score", None) is not None and math.isfinite(float(item["ood_survival_score"]))
            )
            / max(
                1,
                sum(
                    1
                    for item in theory_metrics
                    if item.get("ood_survival_score", None) is not None and math.isfinite(float(item["ood_survival_score"]))
                ),
            )
        ),
        "constant_lift_success_rate": None
        if not theory_metrics
        else float(
            sum(1.0 for item in theory_metrics if bool(item.get("constant_lift_success", False)))
            / max(1, len(theory_metrics))
        ),
    }
    research_activation_summary = _aggregate_research_activation(
        runs,
        research_profile=str(resolved_profile),
    )
    return {
        "mode": "oracle_discovery_benchmark",
        "suite_id": str(regression_payload.get("suite_id", "") or ""),
        "suite_manifest": str(regression_payload.get("suite_manifest", "") or ""),
        "config": {
            "research_profile": str(resolved_profile),
            "committee_topk": int(committee_topk),
            "max_members": None if max_members is None else int(max_members),
            "experiment_manifest_path": None if experiment_manifest_path is None else str(experiment_manifest_path),
            "beta": float(beta),
            "gamma": float(gamma),
            "disagreement_mode": str(disagreement_mode),
            "lambda_cost": float(lambda_cost),
            "lambda_noise": float(lambda_noise),
            "lambda_feasibility": float(lambda_feasibility),
            "dtype": str(dtype),
            "discovery_constant_lift_enable": bool(discovery_constant_lift_enable),
            "discovery_constant_lift_min_regimes": int(max(2, int(discovery_constant_lift_min_regimes))),
            "discovery_constant_lift_trigger_mean_cv": float(discovery_constant_lift_trigger_mean_cv),
            "discovery_constant_lift_apply_enable": bool(discovery_constant_lift_apply_enable),
            "discovery_constant_lift_apply_topk": int(max(0, int(discovery_constant_lift_apply_topk))),
            "discovery_constant_lift_min_rel_gain": float(discovery_constant_lift_min_rel_gain),
            "witness_capture_enable": bool(witness_capture_enable),
            "witness_hessian_diag_enable": bool(witness_hessian_diag_enable),
            "diagnostic_set": str(diagnostic_set or "basic"),
            "experiment_optimize_enable": bool(experiment_optimize_enable),
            "experiment_opt_steps": int(max(1, int(experiment_opt_steps))),
            "experiment_opt_lr": float(experiment_opt_lr),
            "experiment_project_mode": str(experiment_project_mode or "nearest_box"),
            "theory_benchmark_enable": bool(theory_benchmark_enable),
        },
        "aggregate": aggregate,
        "research_activation_summary": research_activation_summary,
        "runs": runs,
    }


def run_oracle_discovery_research_benchmark(
    regression_payload: Mapping[str, Any],
    *,
    output_dir: str | pathlib.Path,
    research_profiles: Sequence[str] | None = None,
    committee_topk: int = 8,
    max_members: int | None = None,
    experiment_manifest_path: str | pathlib.Path | None = None,
    beta: float = 0.0,
    gamma: float = 0.0,
    disagreement_mode: str | None = None,
    lambda_cost: float = 1.0,
    lambda_noise: float = 1.0,
    lambda_feasibility: float = 1.0,
    dtype: torch.dtype = torch.float64,
    discovery_constant_lift_enable: bool = False,
    discovery_constant_lift_min_regimes: int = 3,
    discovery_constant_lift_trigger_mean_cv: float = 0.5,
    discovery_constant_lift_apply_enable: bool = False,
    discovery_constant_lift_apply_topk: int = 1,
    discovery_constant_lift_min_rel_gain: float = 1.01,
    witness_capture_enable: bool = False,
    witness_hessian_diag_enable: bool = False,
    diagnostic_set: str = "basic",
    experiment_optimize_enable: bool = False,
    experiment_opt_steps: int = 32,
    experiment_opt_lr: float = 0.05,
    experiment_project_mode: str = "nearest_box",
    theory_benchmark_enable: bool = False,
) -> dict[str, Any]:
    profile_names = [
        normalize_name
        for normalize_name in (
            resolve_discovery_research_profile(name)[0]
            for name in list(research_profiles or RESEARCH_PROFILE_NAMES)
        )
    ]
    seen_profiles: set[str] = set()
    profile_order: list[str] = []
    for name in profile_names:
        if str(name) in seen_profiles:
            continue
        seen_profiles.add(str(name))
        profile_order.append(str(name))
    profile_reports: list[dict[str, Any]] = []
    profiles_with_witness_mode: list[str] = []
    profiles_with_experiment_optimization: list[str] = []
    profiles_with_constant_lift_apply: list[str] = []
    profiles_with_witness_ranking_change: list[str] = []
    profiles_with_exact_jet_usage: list[str] = []
    profiles_with_numeric_jet_fallback: list[str] = []
    route_usage_by_profile: dict[str, dict[str, int]] = {}
    jet_source_usage_by_profile: dict[str, dict[str, int]] = {}
    selected_experiment_counts_by_profile: dict[str, dict[str, int]] = {}
    for profile_name in profile_order:
        report = run_oracle_discovery_benchmark(
            regression_payload,
            output_dir=output_dir,
            committee_topk=int(committee_topk),
            max_members=max_members,
            experiment_manifest_path=experiment_manifest_path,
            beta=float(beta),
            gamma=float(gamma),
            disagreement_mode=disagreement_mode,
            lambda_cost=float(lambda_cost),
            lambda_noise=float(lambda_noise),
            lambda_feasibility=float(lambda_feasibility),
            dtype=dtype,
            discovery_constant_lift_enable=bool(discovery_constant_lift_enable),
            discovery_constant_lift_min_regimes=int(discovery_constant_lift_min_regimes),
            discovery_constant_lift_trigger_mean_cv=float(discovery_constant_lift_trigger_mean_cv),
            discovery_constant_lift_apply_enable=bool(discovery_constant_lift_apply_enable),
            discovery_constant_lift_apply_topk=int(discovery_constant_lift_apply_topk),
            discovery_constant_lift_min_rel_gain=float(discovery_constant_lift_min_rel_gain),
            witness_capture_enable=bool(witness_capture_enable),
            witness_hessian_diag_enable=bool(witness_hessian_diag_enable),
            diagnostic_set=str(diagnostic_set or "basic"),
            experiment_optimize_enable=bool(experiment_optimize_enable),
            experiment_opt_steps=int(experiment_opt_steps),
            experiment_opt_lr=float(experiment_opt_lr),
            experiment_project_mode=str(experiment_project_mode or "nearest_box"),
            theory_benchmark_enable=bool(theory_benchmark_enable),
            research_profile=str(profile_name),
        )
        activation_summary = dict(report.get("research_activation_summary", {}) or {})
        if int(activation_summary.get("witness_mode_run_count", 0) or 0) > 0:
            profiles_with_witness_mode.append(str(profile_name))
        if int(activation_summary.get("experiment_optimization_run_count", 0) or 0) > 0:
            profiles_with_experiment_optimization.append(str(profile_name))
        if int(activation_summary.get("constant_lift_applied_total", 0) or 0) > 0:
            profiles_with_constant_lift_apply.append(str(profile_name))
        if int(activation_summary.get("witness_weighted_ranking_changed_run_count", 0) or 0) > 0:
            profiles_with_witness_ranking_change.append(str(profile_name))
        if int(activation_summary.get("exact_jet_row_total", 0) or 0) > 0:
            profiles_with_exact_jet_usage.append(str(profile_name))
        if int(activation_summary.get("numeric_jet_fallback_row_total", 0) or 0) > 0:
            profiles_with_numeric_jet_fallback.append(str(profile_name))
        route_usage_by_profile[str(profile_name)] = {
            str(k): int(v)
            for k, v in sorted(dict(activation_summary.get("interesting_route_usage", {}) or {}).items())
        }
        jet_source_usage_by_profile[str(profile_name)] = {
            str(k): int(v)
            for k, v in sorted(dict(activation_summary.get("jet_source_counts", {}) or {}).items())
        }
        selected_experiment_counts_by_profile[str(profile_name)] = {
            str(k): int(v)
            for k, v in sorted(dict(activation_summary.get("selected_experiment_counts", {}) or {}).items())
        }
        profile_reports.append(
            {
                "research_profile": str(profile_name),
                "config": dict(report.get("config", {}) or {}),
                "aggregate": dict(report.get("aggregate", {}) or {}),
                "research_activation_summary": activation_summary,
                "runs": list(report.get("runs", []) or []),
            }
        )
    return {
        "mode": "oracle_discovery_research_benchmark",
        "suite_id": str(regression_payload.get("suite_id", "") or ""),
        "suite_manifest": str(regression_payload.get("suite_manifest", "") or ""),
        "profile_order": list(profile_order),
        "profiles": profile_reports,
        "comparison": {
            "profiles_with_witness_mode": list(profiles_with_witness_mode),
            "profiles_with_experiment_optimization": list(profiles_with_experiment_optimization),
            "profiles_with_constant_lift_apply": list(profiles_with_constant_lift_apply),
            "profiles_with_witness_weighted_ranking_change": list(profiles_with_witness_ranking_change),
            "profiles_with_exact_jet_usage": list(profiles_with_exact_jet_usage),
            "profiles_with_numeric_jet_fallback": list(profiles_with_numeric_jet_fallback),
            "route_usage_by_profile": route_usage_by_profile,
            "jet_source_usage_by_profile": jet_source_usage_by_profile,
            "selected_experiment_counts_by_profile": selected_experiment_counts_by_profile,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process oracle regression results with discovery committee scoring")
    parser.add_argument("--results", required=True, help="oracle_regression_results.json path")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument(
        "--research_profile_benchmark_enable",
        action="store_true",
        help="Run a smoke ablation benchmark over multiple named research profiles.",
    )
    parser.add_argument(
        "--research_profile_benchmark_profiles",
        type=str,
        default="legacy,teacher_witness,teacher_witness_full,teacher_witness_exact",
        help="Comma-separated research profiles for the smoke ablation benchmark.",
    )
    parser.add_argument(
        "--research_profile",
        type=str,
        default=None,
        choices=list(RESEARCH_PROFILE_NAMES),
        help="Optional named discovery research profile preset; omit to use the default witness-mode scheduler.",
    )
    parser.add_argument("--committee_topk", type=int, default=8)
    parser.add_argument("--max_members", type=int, default=None)
    parser.add_argument("--experiment_manifest", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--disagreement_mode", type=str, default="auto", choices=["auto", "witness"])
    parser.add_argument("--lambda_cost", type=float, default=1.0)
    parser.add_argument("--lambda_noise", type=float, default=1.0)
    parser.add_argument("--lambda_feasibility", type=float, default=1.0)
    parser.add_argument("--discovery_constant_lift_enable", action="store_true")
    parser.add_argument("--discovery_constant_lift_min_regimes", type=int, default=3)
    parser.add_argument("--discovery_constant_lift_trigger_mean_cv", type=float, default=0.5)
    parser.add_argument("--discovery_constant_lift_apply_enable", action="store_true")
    parser.add_argument("--discovery_constant_lift_apply_topk", type=int, default=1)
    parser.add_argument("--discovery_constant_lift_min_rel_gain", type=float, default=1.01)
    parser.add_argument("--witness_capture_enable", action="store_true")
    parser.add_argument("--witness_hessian_diag_enable", action="store_true")
    parser.add_argument("--diagnostic_set", type=str, default="basic", choices=["basic", "extended", "physics"])
    parser.add_argument("--experiment_optimize_enable", action="store_true")
    parser.add_argument("--experiment_opt_steps", type=int, default=32)
    parser.add_argument("--experiment_opt_lr", type=float, default=0.05)
    parser.add_argument("--experiment_project_mode", type=str, default="nearest_box")
    parser.add_argument("--theory_benchmark_enable", action="store_true")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_profile_csv(raw: str | None) -> list[str]:
    return [
        str(item).strip()
        for item in str(raw or "").split(",")
        if str(item).strip()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    results_path = pathlib.Path(str(args.results))
    payload = _load_json(results_path)
    out_dir = results_path.parent
    common_kwargs = dict(
        output_dir=out_dir,
        committee_topk=int(args.committee_topk),
        max_members=None if args.max_members is None else int(args.max_members),
        experiment_manifest_path=args.experiment_manifest,
        beta=float(args.beta),
        gamma=float(args.gamma),
        disagreement_mode=None if args.disagreement_mode is None else str(args.disagreement_mode),
        lambda_cost=float(args.lambda_cost),
        lambda_noise=float(args.lambda_noise),
        lambda_feasibility=float(args.lambda_feasibility),
        discovery_constant_lift_enable=bool(args.discovery_constant_lift_enable),
        discovery_constant_lift_min_regimes=max(2, int(args.discovery_constant_lift_min_regimes)),
        discovery_constant_lift_trigger_mean_cv=float(args.discovery_constant_lift_trigger_mean_cv),
        discovery_constant_lift_apply_enable=bool(args.discovery_constant_lift_apply_enable),
        discovery_constant_lift_apply_topk=max(0, int(args.discovery_constant_lift_apply_topk)),
        discovery_constant_lift_min_rel_gain=float(args.discovery_constant_lift_min_rel_gain),
        witness_capture_enable=bool(args.witness_capture_enable),
        witness_hessian_diag_enable=bool(args.witness_hessian_diag_enable),
        diagnostic_set=str(args.diagnostic_set or "basic"),
        experiment_optimize_enable=bool(args.experiment_optimize_enable),
        experiment_opt_steps=max(1, int(args.experiment_opt_steps)),
        experiment_opt_lr=float(args.experiment_opt_lr),
        experiment_project_mode=str(args.experiment_project_mode or "nearest_box"),
        theory_benchmark_enable=bool(args.theory_benchmark_enable),
        dtype=_dtype_from_name(args.dtype),
    )
    if bool(args.research_profile_benchmark_enable):
        report = run_oracle_discovery_research_benchmark(
            payload,
            research_profiles=_parse_profile_csv(args.research_profile_benchmark_profiles),
            **common_kwargs,
        )
    else:
        report = run_oracle_discovery_benchmark(
            payload,
            research_profile=None if args.research_profile is None else str(args.research_profile),
            **common_kwargs,
        )
    out_path = pathlib.Path(str(args.output)) if args.output else out_dir / (
        "oracle_discovery_research_benchmark.json"
        if bool(args.research_profile_benchmark_enable)
        else "oracle_discovery_results.json"
    )
    _write_json(report, out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
