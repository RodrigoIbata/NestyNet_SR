# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Fail-closed automatic routing into polynomial Poisson discovery.

The automatic branch is deliberately a thin policy layer over
``discover_poisson_structure_multi``.  It runs only for adequately sampled
autonomous vector fields, caps the tensor dictionary and candidate budget,
and always returns a structured skip/rejection reason instead of perturbing
the baseline system-DE result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AsinNode,
    AcosNode,
    AtanNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    PowNode,
    SinNode,
)
from nestynet_sr.sr_de.poisson_core import StableNullspaceConfig, VectorField
from nestynet_sr.sr_de.poisson_invariants import HamiltonianFitConfig
from nestynet_sr.sr_de.poisson_search import (
    PoissonSearchConfig,
    PoissonSearchResult,
    discover_poisson_structure_multi,
)


@dataclass(frozen=True)
class AutoPoissonConfig:
    """Coverage and runtime gates for automatic Poisson escalation."""

    enabled: bool = True
    min_state_dim: int = 2
    max_state_dim: int = 6
    min_samples_per_dataset: int = 96
    max_samples_per_dataset: int = 2048
    max_datasets: int = 8
    min_coordinate_relative_span: float = 1.0e-4
    min_coordinate_absolute_span: float = 1.0e-8
    max_standardized_condition: float = 1.0e8
    max_tensor_coefficients: int = 420
    lanes: tuple[str, ...] = ("constant", "linear", "affine", "quadratic")
    max_representatives: int = 48
    sparse_rotation_steps: int = 16
    bootstrap: int = 2
    max_hamiltonian_stlsq_iterations: int = 64
    noise_calibrated: bool = True
    noise_calibrated_relative_tolerance: float | None = None
    noise_calibration_factor: float = 3.0
    max_noise_relative_tolerance: float = 1.0e-3
    noise_near_null_max_vectors: int = 2
    noise_near_null_min_spectral_gap: float = 10.0
    nullspace_max_principal_angle: float = 0.25
    hamiltonian_library_degree: int = 2
    casimir_library_degree: int = 3
    random_seed: int = 0


@dataclass
class AutoPoissonReport:
    """Outcome of an automatic branch, including fail-closed skip reasons."""

    status: str
    reason: str
    enabled: bool
    state_dim: int | None = None
    dataset_count: int = 0
    sample_counts: tuple[int, ...] = ()
    coverage: tuple[Mapping[str, Any], ...] = ()
    search: PoissonSearchResult | None = None
    selected_lane: str | None = None
    promotion_tier: str | None = None
    noise_calibration: Mapping[str, Any] | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.search is not None

    def to_report(self) -> dict[str, Any]:
        best = None if self.search is None else self.search.best
        return {
            "status": self.status,
            "reason": self.reason,
            "enabled": self.enabled,
            "state_dim": self.state_dim,
            "dataset_count": self.dataset_count,
            "sample_counts": [int(v) for v in self.sample_counts],
            "coverage": [dict(row) for row in self.coverage],
            "selected_lane": self.selected_lane,
            "promotion_tier": self.promotion_tier,
            "noise_calibration": dict(self.noise_calibration or {}),
            "accepted": self.accepted,
            "generic_rank": None if best is None else int(best.rank.generic_rank),
            "jacobi_relative": None if best is None else float(best.jacobi.relative),
            "polynomial_jacobi_max_abs": (
                None if best is None else float(best.polynomial_jacobi.max_abs)
            ),
            "hamiltonian_validation_relative": (
                []
                if best is None
                else [float(v) for v in best.hamiltonian_validation_relative]
            ),
            "hamiltonian": (
                None
                if best is None or best.hamiltonian is None
                else best.hamiltonian.to_report()
            ),
            "failure_reasons": (
                [] if best is None else [str(v) for v in best.failure_reasons]
            ),
            "casimirs": (
                None
                if best is None or best.casimirs is None
                else best.casimirs.to_report()
            ),
        }


def _sample_points(z: torch.Tensor, limit: int) -> torch.Tensor:
    if z.shape[0] <= limit:
        return z
    indices = torch.linspace(
        0, z.shape[0] - 1, steps=limit, device=z.device
    ).round().long()
    return z.index_select(0, torch.unique_consecutive(indices))


def _coverage_report(z: torch.Tensor, cfg: AutoPoissonConfig) -> dict[str, Any]:
    spans = torch.max(z, dim=0).values - torch.min(z, dim=0).values
    scales = torch.sqrt(torch.mean(z.square(), dim=0)).clamp_min(1.0)
    thresholds = (
        float(cfg.min_coordinate_absolute_span)
        + float(cfg.min_coordinate_relative_span) * scales
    )
    span_ok = bool(torch.all(spans > thresholds).item())
    standardized = (z - torch.mean(z, dim=0)) / spans.clamp_min(
        float(cfg.min_coordinate_absolute_span)
    )
    singular_values = torch.linalg.svdvals(standardized)
    if singular_values.numel() == 0 or float(singular_values[-1].item()) <= 0.0:
        condition = math.inf
    else:
        condition = float((singular_values[0] / singular_values[-1]).item())
    return {
        "coordinate_spans": [float(v) for v in spans.detach().cpu()],
        "span_thresholds": [float(v) for v in thresholds.detach().cpu()],
        "span_ok": span_ok,
        "standardized_condition": condition,
        "condition_ok": bool(
            math.isfinite(condition)
            and condition <= float(cfg.max_standardized_condition)
        ),
    }


def _tensor_unknown_count(state_dim: int, lane: str) -> int:
    pair_count = state_dim * (state_dim - 1) // 2
    if lane == "constant":
        scalar_count = 1
    elif lane == "linear":
        scalar_count = state_dim
    elif lane == "affine":
        scalar_count = state_dim + 1
    elif lane == "quadratic":
        scalar_count = (state_dim + 1) * (state_dim + 2) // 2
    else:
        raise ValueError(f"unknown Poisson lane {lane!r}")
    return pair_count * scalar_count


def _default_search_config(cfg: AutoPoissonConfig) -> PoissonSearchConfig:
    noise_tolerance = (
        min(
            max(0.0, float(cfg.noise_calibrated_relative_tolerance)),
            max(0.0, float(cfg.max_noise_relative_tolerance)),
        )
        if bool(cfg.noise_calibrated)
        and cfg.noise_calibrated_relative_tolerance is not None
        else None
    )
    noise_active = bool(noise_tolerance is not None and noise_tolerance > 0.0)
    invariance_tolerance = max(1.0e-8, noise_tolerance or 0.0)
    hamiltonian_tolerance = max(1.0e-7, noise_tolerance or 0.0)
    return PoissonSearchConfig(
        lanes=tuple(cfg.lanes),
        validation_fraction=0.25,
        random_seed=int(cfg.random_seed),
        stop_at_first_accepted_lane=True,
        max_representatives=max(1, int(cfg.max_representatives)),
        sparse_rotation_steps=max(0, int(cfg.sparse_rotation_steps)),
        nullspace=StableNullspaceConfig(
            rank_rtol=1.0e-9,
            rank_atol=1.0e-11,
            bootstrap=max(0, int(cfg.bootstrap)),
            random_seed=int(cfg.random_seed) + 1,
            near_null_max_vectors=(
                max(1, int(cfg.noise_near_null_max_vectors))
                if noise_active
                else 0
            ),
            near_null_min_spectral_gap=float(
                cfg.noise_near_null_min_spectral_gap
            ),
        ),
        hamiltonian=HamiltonianFitConfig(
            solver="stlsq",
            ridge=1.0e-12,
            stlsq_lambda=1.0e-7,
            stlsq_max_iter=min(
                12, max(1, int(cfg.max_hamiltonian_stlsq_iterations))
            ),
            relative_residual_tolerance=hamiltonian_tolerance,
            absolute_residual_tolerance=1.0e-10,
        ),
        hamiltonian_mode="independent",
        hamiltonian_library_degree=max(1, int(cfg.hamiltonian_library_degree)),
        casimir_library_degree=max(1, int(cfg.casimir_library_degree)),
        require_complete_casimirs=True,
        invariance_relative_tolerance=invariance_tolerance,
        jacobi_relative_tolerance=1.0e-8,
        polynomial_jacobi_tolerance=1.0e-9,
        require_nullspace_stability=True,
        nullspace_max_principal_angle=float(cfg.nullspace_max_principal_angle),
    )


def _bounded_search_config(
    cfg: AutoPoissonConfig,
    requested: PoissonSearchConfig | None,
) -> tuple[PoissonSearchConfig | None, str | None]:
    """Build a trusted automatic policy with restrictive-only public overrides.

    Constructing a fresh config is deliberate: copying ``requested`` and
    replacing a short list would silently expose every future search field as
    a new automatic-policy escape.
    """

    if requested is None:
        return _default_search_config(cfg), None
    allowed_lanes = set(cfg.lanes)
    if not requested.lanes or any(lane not in allowed_lanes for lane in requested.lanes):
        return None, "search_config_lane_outside_auto_policy"
    bounded_lanes = tuple(
        lane for lane in cfg.lanes if lane in set(requested.lanes)
    )
    trusted = _default_search_config(cfg)
    bounded_nullspace = replace(
        trusted.nullspace,
        rank_rtol=min(
            float(requested.nullspace.rank_rtol),
            float(trusted.nullspace.rank_rtol),
        ),
        rank_atol=min(
            float(requested.nullspace.rank_atol),
            float(trusted.nullspace.rank_atol),
        ),
        # Bootstrap count is a certificate requirement as well as a compute
        # budget. Tune it through AutoPoissonConfig, not the nested override.
        bootstrap=max(0, int(cfg.bootstrap)),
        random_seed=int(requested.nullspace.random_seed),
    )
    bounded_hamiltonian = replace(
        trusted.hamiltonian,
        stlsq_max_iter=min(
            max(1, int(requested.hamiltonian.stlsq_max_iter)),
            max(1, int(cfg.max_hamiltonian_stlsq_iterations)),
        ),
        relative_residual_tolerance=min(
            float(requested.hamiltonian.relative_residual_tolerance),
            float(trusted.hamiltonian.relative_residual_tolerance),
        ),
        absolute_residual_tolerance=min(
            float(requested.hamiltonian.absolute_residual_tolerance),
            float(trusted.hamiltonian.absolute_residual_tolerance),
        ),
    )
    return (
        PoissonSearchConfig(
            lanes=bounded_lanes,
            validation_fraction=max(
                float(trusted.validation_fraction),
                min(float(requested.validation_fraction), 0.5),
            ),
            random_seed=int(requested.random_seed),
            stop_at_first_accepted_lane=True,
            normalize_dataset_blocks=True,
            nullspace=bounded_nullspace,
            max_representatives=min(
                int(requested.max_representatives),
                max(1, int(cfg.max_representatives)),
            ),
            sparse_rotation_steps=min(
                int(requested.sparse_rotation_steps),
                max(0, int(cfg.sparse_rotation_steps)),
            ),
            coefficient_tolerance=float(trusted.coefficient_tolerance),
            invariance_relative_tolerance=min(
                float(requested.invariance_relative_tolerance),
                float(trusted.invariance_relative_tolerance),
            ),
            invariance_absolute_tolerance=min(
                float(requested.invariance_absolute_tolerance),
                float(trusted.invariance_absolute_tolerance),
            ),
            jacobi_relative_tolerance=min(
                float(requested.jacobi_relative_tolerance),
                float(trusted.jacobi_relative_tolerance),
            ),
            jacobi_absolute_tolerance=min(
                float(requested.jacobi_absolute_tolerance),
                float(trusted.jacobi_absolute_tolerance),
            ),
            polynomial_jacobi_tolerance=min(
                float(requested.polynomial_jacobi_tolerance),
                float(trusted.polynomial_jacobi_tolerance),
            ),
            rank_relative_tolerance=min(
                float(requested.rank_relative_tolerance),
                float(trusted.rank_relative_tolerance),
            ),
            rank_absolute_tolerance=min(
                float(requested.rank_absolute_tolerance),
                float(trusted.rank_absolute_tolerance),
            ),
            minimum_rank_stable_fraction=max(
                float(requested.minimum_rank_stable_fraction),
                float(trusted.minimum_rank_stable_fraction),
            ),
            require_nonzero_rank=True,
            require_nullspace_stability=True,
            nullspace_max_principal_angle=float(
                trusted.nullspace_max_principal_angle
            ),
            hamiltonian=bounded_hamiltonian,
            hamiltonian_mode=str(trusted.hamiltonian_mode),
            require_hamiltonian=True,
            hamiltonian_library_degree=min(
                int(requested.hamiltonian_library_degree),
                max(1, int(cfg.hamiltonian_library_degree)),
            ),
            casimir_library_degree=min(
                int(requested.casimir_library_degree),
                max(1, int(cfg.casimir_library_degree)),
            ),
            # Completeness is an automatic promotion certificate, so a nested
            # manual override cannot disable the discovery that supplies it.
            discover_casimirs=True,
            require_complete_casimirs=True,
            casimir_incompleteness_weight=max(
                0.0,
                float(trusted.casimir_incompleteness_weight),
            ),
            complexity_weight=max(0.0, float(requested.complexity_weight)),
        ),
        None,
    )


def auto_discover_poisson_structure_multi(
    rhs_list: Sequence[Any],
    state_points_list: Sequence[torch.Tensor],
    config: AutoPoissonConfig | None = None,
    *,
    search_config: PoissonSearchConfig | None = None,
) -> AutoPoissonReport:
    """Guard and run the shared-geometry Poisson lane ladder."""

    cfg = config or AutoPoissonConfig()
    if not cfg.enabled:
        return AutoPoissonReport(
            status="skipped", reason="disabled", enabled=False
        )
    if not rhs_list or len(rhs_list) != len(state_points_list):
        return AutoPoissonReport(
            status="skipped",
            reason="rhs_and_state_dataset_count_mismatch",
            enabled=True,
        )
    if len(rhs_list) > int(cfg.max_datasets):
        return AutoPoissonReport(
            status="skipped",
            reason="dataset_budget_exceeded",
            enabled=True,
            dataset_count=len(rhs_list),
        )
    try:
        points = tuple(torch.as_tensor(z) for z in state_points_list)
        if any(z.ndim != 2 for z in points):
            raise ValueError("state points must have shape (N,d)")
        state_dim = int(points[0].shape[1])
        if any(int(z.shape[1]) != state_dim for z in points):
            raise ValueError("state dimensions do not match")
        counts = tuple(int(z.shape[0]) for z in points)
        if not int(cfg.min_state_dim) <= state_dim <= int(cfg.max_state_dim):
            return AutoPoissonReport(
                status="skipped",
                reason="unsupported_state_dimension",
                enabled=True,
                state_dim=state_dim,
                dataset_count=len(points),
                sample_counts=counts,
            )
        if min(counts) < int(cfg.min_samples_per_dataset):
            return AutoPoissonReport(
                status="skipped",
                reason="insufficient_state_samples",
                enabled=True,
                state_dim=state_dim,
                dataset_count=len(points),
                sample_counts=counts,
            )
        bounded_search, search_policy_error = _bounded_search_config(cfg, search_config)
        if search_policy_error is not None or bounded_search is None:
            return AutoPoissonReport(
                status="skipped",
                reason=search_policy_error or "invalid_bounded_search_config",
                enabled=True,
                state_dim=state_dim,
                dataset_count=len(points),
                sample_counts=counts,
            )
        max_unknowns = max(
            _tensor_unknown_count(state_dim, lane) for lane in bounded_search.lanes
        )
        if max_unknowns > int(cfg.max_tensor_coefficients):
            return AutoPoissonReport(
                status="skipped",
                reason="tensor_dictionary_budget_exceeded",
                enabled=True,
                state_dim=state_dim,
                dataset_count=len(points),
                sample_counts=counts,
            )
        sampled = tuple(
            _sample_points(z, int(cfg.max_samples_per_dataset)) for z in points
        )
        coverage = tuple(_coverage_report(z, cfg) for z in sampled)
        if not all(row["span_ok"] and row["condition_ok"] for row in coverage):
            return AutoPoissonReport(
                status="skipped",
                reason="insufficient_state_space_coverage",
                enabled=True,
                state_dim=state_dim,
                dataset_count=len(points),
                sample_counts=counts,
                coverage=coverage,
            )
        search = discover_poisson_structure_multi(
            rhs_list,
            sampled,
            bounded_search,
        )
        return AutoPoissonReport(
            status="accepted" if search.accepted else "rejected",
            reason=(
                "accepted_certified_poisson_structure"
                if search.accepted
                else "no_lane_passed_all_certificates"
            ),
            enabled=True,
            state_dim=state_dim,
            dataset_count=len(points),
            sample_counts=counts,
            coverage=coverage,
            search=search,
            selected_lane=None if search.best is None else search.best.lane,
            promotion_tier=(
                None if search.best is None else str(search.best.nullspace.tier)
            ),
            noise_calibration=(
                None
                if cfg.noise_calibrated_relative_tolerance is None
                else {
                    "source": "explicit_auto_config",
                    "relative_tolerance": min(
                        max(
                            0.0,
                            float(cfg.noise_calibrated_relative_tolerance),
                        ),
                        max(0.0, float(cfg.max_noise_relative_tolerance)),
                    ),
                }
            ),
        )
    except Exception as exc:
        return AutoPoissonReport(
            status="failed",
            reason=f"{type(exc).__name__}:{str(exc)[:240]}",
            enabled=True,
            dataset_count=len(rhs_list),
        )


def auto_discover_poisson_structure(
    rhs: Any,
    state_points: torch.Tensor,
    config: AutoPoissonConfig | None = None,
    *,
    search_config: PoissonSearchConfig | None = None,
) -> AutoPoissonReport:
    """Single-vector-field convenience wrapper."""

    return auto_discover_poisson_structure_multi(
        [rhs], [state_points], config, search_config=search_config
    )


def _state_ast_value(
    node: Any,
    z: torch.Tensor,
    output_to_axis: Mapping[int, int],
) -> torch.Tensor:
    if node is None:
        return torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
    if isinstance(node, ConstNode):
        return torch.full_like(z[:, 0], float(node.value))
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind not in {"u", "field", "state"}:
            raise ValueError(f"nonautonomous_or_derivative_atom:{kind}")
        kwargs = getattr(node, "kwargs", None) or {}
        output = int(kwargs.get("out_idx", kwargs.get("out", 0)))
        if output not in output_to_axis:
            raise ValueError(f"state component {output} is outside the system")
        return z[:, output_to_axis[output]]
    if isinstance(node, AddNode):
        return _state_ast_value(node.left, z, output_to_axis) + _state_ast_value(
            node.right, z, output_to_axis
        )
    if isinstance(node, MulNode):
        return _state_ast_value(node.left, z, output_to_axis) * _state_ast_value(
            node.right, z, output_to_axis
        )
    if isinstance(node, PowNode):
        if not isinstance(node.exponent, (int, float)):
            raise ValueError("state AST exponent must be numeric")
        return _state_ast_value(node.base, z, output_to_axis).pow(node.exponent)
    unary = {
        LogNode: torch.log,
        ExpNode: torch.exp,
        SinNode: torch.sin,
        CosNode: torch.cos,
        AsinNode: torch.asin,
        AcosNode: torch.acos,
        AtanNode: torch.atan,
    }
    for cls, operation in unary.items():
        if isinstance(node, cls):
            return operation(_state_ast_value(node.arg, z, output_to_axis))
    raise ValueError(f"unsupported_state_ast:{type(node).__name__}")


def vector_field_from_system_result(result: Any) -> VectorField:
    """Compile an autonomous first-order system result into a vector field."""

    if int(getattr(result, "order", -1)) != 1:
        raise ValueError("Poisson auto-routing requires a first-order system")
    outputs = tuple(int(v) for v in getattr(result, "out_idxs", ()))
    coefficients = torch.as_tensor(getattr(result, "coeffs"))
    terms = tuple(getattr(result, "term_asts", ()))
    if coefficients.ndim != 2 or coefficients.shape != (len(outputs), len(terms)):
        raise ValueError("system coefficient shape is inconsistent")
    output_to_axis = {output: axis for axis, output in enumerate(outputs)}

    # Validate autonomy once with a symbolic-sized probe before returning the
    # callable; actual values/Jacobians retain the caller's dtype and device.
    probe = torch.ones((2, len(outputs)), dtype=torch.float64)
    for term in terms:
        _state_ast_value(term, probe, output_to_axis)

    def value(z: torch.Tensor) -> torch.Tensor:
        coeff = coefficients.to(device=z.device, dtype=z.dtype)
        columns = torch.stack(
            [_state_ast_value(term, z, output_to_axis) for term in terms], dim=1
        )
        return -(columns.unsqueeze(1) * coeff.unsqueeze(0)).sum(dim=2)

    return VectorField(value, state_dim=len(outputs))


def auto_discover_poisson_from_system_result(
    result: Any,
    state_points: torch.Tensor,
    config: AutoPoissonConfig | None = None,
) -> AutoPoissonReport:
    """Guard an already recovered system law and attach Poisson geometry."""

    cfg = config or AutoPoissonConfig()
    if not cfg.enabled:
        return AutoPoissonReport(status="skipped", reason="disabled", enabled=False)
    try:
        field = vector_field_from_system_result(result)
    except Exception as exc:
        return AutoPoissonReport(
            status="skipped",
            reason=f"nonautonomous_or_unsupported_system:{type(exc).__name__}:{str(exc)[:160]}",
            enabled=True,
            state_dim=len(tuple(getattr(result, "out_idxs", ()))) or None,
            dataset_count=1,
            sample_counts=(int(state_points.shape[0]),),
        )
    calibration: dict[str, Any] = {
        "enabled": bool(cfg.noise_calibrated),
        "source": "none",
    }
    calibrated_cfg = cfg
    if bool(cfg.noise_calibrated):
        residual_values = getattr(result, "rms_val", None)
        if residual_values is None or not all(
            math.isfinite(float(value)) for value in residual_values
        ):
            residual_values = getattr(result, "rms_train", None)
        measured_proposal: float | None = None
        if residual_values and all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in residual_values
        ):
            residual_absolute = math.sqrt(
                sum(float(value) ** 2 for value in residual_values)
                / len(residual_values)
            )
            sampled_values = field.value(torch.as_tensor(state_points))
            field_scale = float(sampled_values.square().mean().sqrt().item())
            measured_relative = residual_absolute / max(
                field_scale, torch.finfo(sampled_values.dtype).eps
            )
            measured_proposal = (
                float(cfg.noise_calibration_factor) * measured_relative
            )
            calibration.update(
                residual_absolute=residual_absolute,
                field_rms=field_scale,
                measured_relative=measured_relative,
                factor=float(cfg.noise_calibration_factor),
                proposed_relative_tolerance=measured_proposal,
            )
            if measured_proposal > float(cfg.max_noise_relative_tolerance):
                calibration.update(
                    source="system_de_residual",
                    accepted=False,
                    reason="measured_noise_exceeds_automatic_limit",
                )
                return AutoPoissonReport(
                    status="skipped",
                    reason="system_residual_exceeds_noise_calibration_limit",
                    enabled=True,
                    state_dim=len(tuple(getattr(result, "out_idxs", ()))) or None,
                    dataset_count=1,
                    sample_counts=(int(state_points.shape[0]),),
                    noise_calibration=calibration,
                )

        explicit = cfg.noise_calibrated_relative_tolerance
        if explicit is not None:
            tolerance = min(
                max(0.0, float(explicit)),
                max(0.0, float(cfg.max_noise_relative_tolerance)),
            )
            calibrated_cfg = replace(
                cfg, noise_calibrated_relative_tolerance=tolerance
            )
            calibration.update(
                source="explicit_auto_config",
                relative_tolerance=tolerance,
                accepted=bool(tolerance > 0.0),
            )
        elif measured_proposal is not None:
            calibration["source"] = "system_de_residual"
            if measured_proposal > 0.0:
                calibrated_cfg = replace(
                    cfg, noise_calibrated_relative_tolerance=measured_proposal
                )
                calibration["relative_tolerance"] = measured_proposal
                calibration["accepted"] = True
            else:
                calibration["accepted"] = False
                calibration["reason"] = "zero_measured_residual_uses_exact_tier"
        else:
            calibration["accepted"] = False
            calibration["reason"] = "no_finite_system_residual_calibration"
    report = auto_discover_poisson_structure(field, state_points, calibrated_cfg)
    report.noise_calibration = calibration
    return report


__all__ = [
    "AutoPoissonConfig",
    "AutoPoissonReport",
    "auto_discover_poisson_from_system_result",
    "auto_discover_poisson_structure",
    "auto_discover_poisson_structure_multi",
    "vector_field_from_system_result",
]
