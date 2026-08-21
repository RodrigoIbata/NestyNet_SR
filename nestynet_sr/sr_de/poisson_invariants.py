# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.

r"""Hamiltonian and Casimir recovery for a fixed Poisson bivector.

Fixing :math:`\Pi` makes both searches linear nullspace/regression problems:

* ``f = Pi grad(sum a_k phi_k)`` is linear in the Hamiltonian coefficients.
* ``Pi grad(sum c_k chi_k) = 0`` is linear in the Casimir coefficients.

Scalar terms can be NestyNet AST nodes, monomial exponent tuples, callables,
or small objects exposing ``value_and_gradient(Z)``.  This keeps the geometry
layer independent of a particular symbolic-search frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, List, Optional, Sequence, Tuple

import torch

from nestynet_sr.sr_core.numerics import ridge_lstsq, stlsq
from nestynet_sr.sr_de.de_search import group_stlsq
from nestynet_sr.sr_de.poisson_core import (
    StableNullspaceConfig,
    StableNullspaceResult,
    sparse_nullspace_representatives,
    stable_nullspace,
)


ScalarTerm = Any


def _ast_node_types() -> tuple[type, ...]:
    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        AcosNode,
        AddNode,
        ArgNode,
        AsinNode,
        AtanNode,
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
    )

    return (
        AtomNode,
        AddNode,
        MulNode,
        PowNode,
        LogNode,
        ExpNode,
        SinNode,
        CosNode,
        AsinNode,
        AcosNode,
        AtanNode,
        ConjNode,
        RealNode,
        ImagNode,
        AbsNode,
        ArgNode,
        ConstNode,
    )


@dataclass(frozen=True)
class ScalarLibraryEvaluation:
    values: torch.Tensor  # (N,K)
    gradients: torch.Tensor  # (N,K,d)


@dataclass(frozen=True)
class HamiltonianFitConfig:
    solver: str = "stlsq"  # "stlsq" or "least_squares"
    ridge: float = 1.0e-10
    stlsq_lambda: float = 1.0e-6
    stlsq_max_iter: int = 12
    relative_residual_tolerance: float = 1.0e-6
    absolute_residual_tolerance: float = 1.0e-10
    gauge_mode: str = "sparsest"  # "sparsest", "minimum_norm", or "raw"
    gauge_rank_rtol: float = 1.0e-10
    gauge_rank_atol: float = 1.0e-12
    gauge_zero_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.gauge_mode not in {"sparsest", "minimum_norm", "raw"}:
            raise ValueError("gauge_mode must be 'sparsest', 'minimum_norm', or 'raw'")
        if self.gauge_rank_rtol < 0.0 or self.gauge_rank_atol < 0.0:
            raise ValueError("gauge rank tolerances must be non-negative")
        if self.gauge_zero_tolerance < 0.0:
            raise ValueError("gauge_zero_tolerance must be non-negative")


@dataclass(frozen=True)
class HamiltonianGaugeReport:
    """Coefficient-space certificate for ``H modulo Cas(Pi)`` in one library."""

    nullspace_basis: torch.Tensor
    nullspace_projector: torch.Tensor
    rank: int
    nullity: int
    rank_tolerance: float
    representative_mode: str
    raw_coeffs: torch.Tensor
    representative_coeffs: torch.Tensor
    flow_equivalence_rms: float
    flow_equivalence_relative: float
    eliminated_terms: Tuple[int, ...]

    def to_report(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "nullity": int(self.nullity),
            "rank_tolerance": float(self.rank_tolerance),
            "representative_mode": self.representative_mode,
            "nullspace_basis": self.nullspace_basis.detach().cpu().tolist(),
            "raw_coefficients": [
                float(value) for value in self.raw_coeffs.detach().cpu()
            ],
            "representative_coefficients": [
                float(value)
                for value in self.representative_coeffs.detach().cpu()
            ],
            "flow_equivalence_rms": float(self.flow_equivalence_rms),
            "flow_equivalence_relative": float(self.flow_equivalence_relative),
            "eliminated_terms": [int(v) for v in self.eliminated_terms],
        }


@dataclass
class HamiltonianFitResult:
    terms: Tuple[ScalarTerm, ...]
    coeffs: torch.Tensor
    support: torch.Tensor
    prediction: torch.Tensor
    rms: float
    relative_rms: float
    accepted: bool
    identifiable_terms: torch.Tensor
    complexity: int
    gauge: HamiltonianGaugeReport | None = None

    def evaluate(self, Z: torch.Tensor) -> torch.Tensor:
        library = evaluate_scalar_library(Z, self.terms)
        return library.values @ self.coeffs

    def to_report(self) -> dict[str, Any]:
        return {
            "coefficients": [float(value) for value in self.coeffs.detach().cpu()],
            "support": [bool(value) for value in self.support.detach().cpu()],
            "rms": float(self.rms),
            "relative_rms": float(self.relative_rms),
            "accepted": bool(self.accepted),
            "complexity": int(self.complexity),
            "gauge": None if self.gauge is None else self.gauge.to_report(),
        }


@dataclass
class MultiHamiltonianFitResult:
    mode: str
    terms: Tuple[ScalarTerm, ...]
    coeffs: torch.Tensor  # (D,K), repeated rows in fully_shared mode
    support: torch.Tensor  # (K,) union/shared support
    fits: Tuple[HamiltonianFitResult, ...]
    accepted: bool

    def to_report(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "support": [bool(value) for value in self.support.detach().cpu()],
            "accepted": bool(self.accepted),
            "fits": [fit.to_report() for fit in self.fits],
        }


@dataclass
class CasimirCandidate:
    coeffs: torch.Tensor
    poisson_residual_rms: float
    poisson_residual_relative: float
    flow_residual_rms: Optional[float]
    flow_residual_relative: Optional[float]
    gradient_rms: float
    complexity: int
    accepted: bool
    failure_reasons: Tuple[str, ...]
    heldout_poisson_residual_rms: Optional[float] = None
    heldout_poisson_residual_relative: Optional[float] = None
    heldout_flow_residual_rms: Optional[float] = None
    heldout_flow_residual_relative: Optional[float] = None
    expression: str = ""
    ast: Any | None = None
    coordinate: Any | None = None
    selected: bool = False
    gradient_rank_gain: int = 0
    independence_fraction: float = 0.0
    heldout_independence_fraction: float = 0.0


@dataclass
class CasimirDiscoveryResult:
    terms: Tuple[ScalarTerm, ...]
    nullspace: StableNullspaceResult
    identifiable_terms: torch.Tensor
    candidates: Tuple[CasimirCandidate, ...]
    rejected_candidates: Tuple[CasimirCandidate, ...]
    expected_corank: int = 0
    discovered_corank: int = 0
    complete: bool = False
    status: str = "unassessed"
    generic_poisson_rank: int = 0
    gradient_projector: torch.Tensor | None = None
    coordinates: Tuple[Any, ...] = ()

    def to_report(self) -> dict[str, Any]:
        def candidate_report(candidate: CasimirCandidate) -> dict[str, Any]:
            return {
                "expression": candidate.expression,
                "complexity": int(candidate.complexity),
                "poisson_residual_relative": float(candidate.poisson_residual_relative),
                "heldout_poisson_residual_relative": candidate.heldout_poisson_residual_relative,
                "flow_residual_relative": candidate.flow_residual_relative,
                "heldout_flow_residual_relative": candidate.heldout_flow_residual_relative,
                "gradient_rms": float(candidate.gradient_rms),
                "gradient_rank_gain": int(candidate.gradient_rank_gain),
                "independence_fraction": float(candidate.independence_fraction),
                "heldout_independence_fraction": float(
                    candidate.heldout_independence_fraction
                ),
                "accepted": bool(candidate.accepted),
                "selected": bool(candidate.selected),
                "failure_reasons": list(candidate.failure_reasons),
            }

        return {
            "status": self.status,
            "complete": bool(self.complete),
            "expected_corank": int(self.expected_corank),
            "discovered_corank": int(self.discovered_corank),
            "generic_poisson_rank": int(self.generic_poisson_rank),
            "coefficient_nullity": int(self.nullspace.nullity),
            "expressions": [candidate.expression for candidate in self.candidates],
            "candidates": [candidate_report(row) for row in self.candidates],
            "rejected_candidates": [
                candidate_report(row) for row in self.rejected_candidates
            ],
        }


@dataclass(frozen=True)
class CasimirCarrierScore:
    """FSS-compatible direct score for a scalar Casimir carrier."""

    poisson_residual_rms: float
    poisson_residual_relative: float
    variation: float
    gradient_rms: float
    heldout_poisson_residual_rms: float
    heldout_poisson_residual_relative: float
    heldout_variation: float
    heldout_gradient_rms: float
    finite: bool
    accepted: bool
    failure_reasons: Tuple[str, ...]

    def to_report(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _coerce_value_gradient(
    result: Any, Z: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    value = getattr(result, "values", getattr(result, "value", None))
    gradient = getattr(result, "gradients", getattr(result, "gradient", None))
    if value is not None and gradient is not None:
        return value, gradient
    return None


def _normalise_term_shapes(
    value: torch.Tensor, gradient: torch.Tensor, Z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    value = torch.as_tensor(value, device=Z.device, dtype=Z.dtype)
    gradient = torch.as_tensor(gradient, device=Z.device, dtype=Z.dtype)
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if gradient.ndim == 3 and gradient.shape[1] == 1:
        gradient = gradient[:, 0, :]
    if tuple(value.shape) != (Z.shape[0],):
        raise ValueError(f"scalar term value must have shape (N,), got {tuple(value.shape)}")
    if tuple(gradient.shape) != tuple(Z.shape):
        raise ValueError(
            f"scalar term gradient must have shape {tuple(Z.shape)}, "
            f"got {tuple(gradient.shape)}"
        )
    return value, gradient


def _evaluate_monomial(
    Z: torch.Tensor, exponents: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = tuple(int(v) for v in exponents)
    if len(alpha) != Z.shape[1] or any(v < 0 for v in alpha):
        raise ValueError("monomial exponent tuple is incompatible with Z")
    value = torch.ones(Z.shape[0], device=Z.device, dtype=Z.dtype)
    for axis, power in enumerate(alpha):
        if power:
            value = value * Z[:, axis].pow(power)
    gradient = Z.new_zeros(Z.shape)
    for axis, power in enumerate(alpha):
        if power == 0:
            continue
        derivative = torch.full_like(value, float(power))
        for other_axis, other_power in enumerate(alpha):
            exponent = other_power - (1 if other_axis == axis else 0)
            if exponent:
                derivative = derivative * Z[:, other_axis].pow(exponent)
        gradient[:, axis] = derivative
    return value, gradient


def evaluate_scalar_term(
    Z: torch.Tensor, term: ScalarTerm
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one arbitrary scalar term and its analytic/autograd gradient."""

    if Z.ndim != 2:
        raise ValueError("Z must have shape (N,d)")

    # Exponent tuples are the lightweight public representation of monomials.
    if isinstance(term, (tuple, list)) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in term
    ):
        return _evaluate_monomial(Z, term)

    method = getattr(term, "value_and_gradient", None)
    if callable(method):
        pair = _coerce_value_gradient(method(Z), Z)
        if pair is None:
            raise TypeError("value_and_gradient must return (value, gradient)")
        return _normalise_term_shapes(pair[0], pair[1], Z)

    evaluate = getattr(term, "evaluate", None)
    gradient_method = getattr(term, "gradient", None)
    if callable(evaluate) and callable(gradient_method):
        return _normalise_term_shapes(evaluate(Z), gradient_method(Z), Z)

    # NestyNet AST nodes have a tested analytic chain-rule evaluator.  It is
    # private for historical reasons, so keep the dependency local and fall
    # through cleanly for non-AST terms.
    try:
        from nestynet_sr.sr_core.bridges import _eval_single_input

        if isinstance(term, _ast_node_types()):
            value, gradient, _ = _eval_single_input(
                term, Z, need_grad=True, need_hess=False
            )
            assert gradient is not None
            return _normalise_term_shapes(value, gradient, Z)
    except (ImportError, TypeError):
        pass

    if not callable(term):
        raise TypeError(
            "scalar terms must be monomial tuples, AST nodes, callables, or "
            "objects exposing value_and_gradient"
        )

    # Generic symbolic callables get an autograd fallback.  The detached clone
    # prevents discovery from mutating the caller's graph.
    Z_req = Z.detach().clone().requires_grad_(True)
    result = term(Z_req)
    pair = _coerce_value_gradient(result, Z_req)
    if pair is not None:
        return _normalise_term_shapes(pair[0], pair[1], Z)
    value = torch.as_tensor(result, device=Z.device, dtype=Z.dtype)
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if tuple(value.shape) != (Z.shape[0],):
        raise ValueError("callable scalar term must return shape (N,) or (N,1)")
    if not value.requires_grad:
        gradient = Z_req.new_zeros(Z_req.shape)
    else:
        gradient = torch.autograd.grad(value.sum(), Z_req, create_graph=False)[0]
    return value.detach(), gradient.detach()


def evaluate_scalar_library(
    Z: torch.Tensor, terms: Sequence[ScalarTerm]
) -> ScalarLibraryEvaluation:
    """Evaluate values and gradients of a scalar term library."""

    terms_t = tuple(terms)
    if not terms_t:
        return ScalarLibraryEvaluation(
            values=Z.new_zeros((Z.shape[0], 0)),
            gradients=Z.new_zeros((Z.shape[0], 0, Z.shape[1])),
        )
    evaluated = [evaluate_scalar_term(Z, term) for term in terms_t]
    values = torch.stack([item[0] for item in evaluated], dim=1)
    gradients = torch.stack([item[1] for item in evaluated], dim=1)
    return ScalarLibraryEvaluation(values=values, gradients=gradients)


def evaluate_poisson_tensor(Pi: Any, Z: torch.Tensor) -> torch.Tensor:
    """Evaluate a tensor/callable/bivector object as ``(N,d,d)``."""

    value = Pi
    if not isinstance(Pi, torch.Tensor):
        if callable(Pi):
            value = Pi(Z)
        elif callable(getattr(Pi, "evaluate", None)):
            value = Pi.evaluate(Z)
        else:
            value = getattr(Pi, "tensor", Pi)
    # Do not use a generic ``.values`` fallback here: dense torch tensors have
    # a ``values`` method for sparse layouts, which would unwrap to a builtin
    # method rather than to numerical data.
    if not isinstance(value, torch.Tensor):
        value = getattr(value, "tensor", value)
    value = torch.as_tensor(value, device=Z.device, dtype=Z.dtype)
    d = Z.shape[1]
    if tuple(value.shape) == (d, d):
        value = value.unsqueeze(0).expand(Z.shape[0], -1, -1)
    if tuple(value.shape) != (Z.shape[0], d, d):
        raise ValueError(
            f"Poisson tensor must evaluate to {(Z.shape[0], d, d)}, "
            f"got {tuple(value.shape)}"
        )
    return value


def build_hamiltonian_design_matrix(
    Pi: Any, Z: torch.Tensor, terms: Sequence[ScalarTerm]
) -> Tuple[torch.Tensor, ScalarLibraryEvaluation, torch.Tensor]:
    """Build columns ``Pi(z) grad(phi_k)(z)`` for arbitrary state dimension."""

    tensor = evaluate_poisson_tensor(Pi, Z)
    library = evaluate_scalar_library(Z, terms)
    flows = torch.einsum("nij,nkj->nki", tensor, library.gradients)
    design = flows.permute(0, 2, 1).reshape(Z.shape[0] * Z.shape[1], len(terms))
    identifiable = flows.square().mean(dim=(0, 2)).sqrt() > 1.0e-14
    return design, library, identifiable


def _solve_linear_coefficients(
    design: torch.Tensor,
    target: torch.Tensor,
    config: HamiltonianFitConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    solver = str(config.solver).lower()
    if solver == "stlsq":
        return stlsq(
            design,
            target,
            ridge=config.ridge,
            lam=config.stlsq_lambda,
            max_iter=config.stlsq_max_iter,
        )
    if solver in {"least_squares", "lstsq", "ridge"}:
        if config.ridge > 0:
            coeffs = ridge_lstsq(design, target, config.ridge)
        else:
            coeffs = torch.linalg.lstsq(design, target).solution
        support = coeffs.abs() >= config.stlsq_lambda
        return coeffs, support
    raise ValueError("Hamiltonian solver must be 'stlsq' or 'least_squares'")


def _coefficient_nullspace(
    design: torch.Tensor,
    config: HamiltonianFitConfig,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """Return the full coefficient nullspace of the Hamiltonian design."""

    full_matrices = design.shape[0] < design.shape[1]
    _u, singular_values, vh = torch.linalg.svd(
        design, full_matrices=full_matrices
    )
    leading = float(singular_values[0].item()) if singular_values.numel() else 0.0
    tolerance = max(
        float(config.gauge_rank_atol),
        float(config.gauge_rank_rtol) * max(1.0, leading),
    )
    rank = int(torch.count_nonzero(singular_values > tolerance).item())
    rank = min(rank, int(design.shape[1]))
    basis = vh[rank:].mT.contiguous()
    projector = basis @ basis.mT
    return basis, projector, rank, tolerance


def _sparsest_gauge_representative(
    base: torch.Tensor,
    nullspace_basis: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Greedily zero coefficients while remaining on ``base + ker(A)``."""

    if nullspace_basis.shape[1] == 0:
        return base.clone()
    zero_indices: list[int] = [
        int(index)
        for index in torch.nonzero(base.abs() <= tolerance, as_tuple=False).reshape(-1)
    ]
    representative = base.clone()
    remaining = [index for index in range(base.numel()) if index not in zero_indices]
    remaining.sort(key=lambda index: (float(base[index].abs().item()), index))
    scale = max(float(torch.linalg.vector_norm(base).item()), 1.0)

    for index in remaining:
        trial_indices = zero_indices + [int(index)]
        rows = nullspace_basis[trial_indices]
        target = -base[trial_indices]
        eta = torch.linalg.lstsq(rows, target).solution
        constraint_error = rows @ eta - target
        if float(torch.linalg.vector_norm(constraint_error).item()) > tolerance * scale:
            continue
        trial = base + nullspace_basis @ eta
        trial[trial_indices] = 0.0
        representative = trial
        zero_indices = trial_indices
    return representative


def _apply_hamiltonian_gauge(
    design: torch.Tensor,
    coeffs: torch.Tensor,
    config: HamiltonianFitConfig,
) -> tuple[torch.Tensor, HamiltonianGaugeReport]:
    basis, projector, rank, rank_tolerance = _coefficient_nullspace(design, config)
    raw = coeffs.clone()
    minimum_norm = raw - projector @ raw
    if config.gauge_mode == "raw":
        representative = raw.clone()
    elif config.gauge_mode == "minimum_norm":
        representative = minimum_norm
    else:
        representative = _sparsest_gauge_representative(
            minimum_norm,
            basis,
            float(config.gauge_zero_tolerance),
        )
    representative = torch.where(
        representative.abs() > float(config.gauge_zero_tolerance),
        representative,
        torch.zeros_like(representative),
    )
    flow_difference = design @ (representative - raw)
    flow_equivalence_rms = float(flow_difference.square().mean().sqrt().item())
    raw_flow_scale = float((design @ raw).square().mean().sqrt().item())
    flow_equivalence_relative = flow_equivalence_rms / max(
        raw_flow_scale, torch.finfo(design.dtype).eps
    )
    eliminated = tuple(
        int(index)
        for index in range(raw.numel())
        if float(raw[index].abs().item()) > float(config.gauge_zero_tolerance)
        and float(representative[index].abs().item())
        <= float(config.gauge_zero_tolerance)
    )
    return representative, HamiltonianGaugeReport(
        nullspace_basis=basis,
        nullspace_projector=projector,
        rank=rank,
        nullity=int(basis.shape[1]),
        rank_tolerance=rank_tolerance,
        representative_mode=str(config.gauge_mode),
        raw_coeffs=raw,
        representative_coeffs=representative.clone(),
        flow_equivalence_rms=flow_equivalence_rms,
        flow_equivalence_relative=flow_equivalence_relative,
        eliminated_terms=eliminated,
    )


def _apply_shared_support_hamiltonian_gauge(
    designs: Sequence[torch.Tensor],
    coeff_rows: Sequence[torch.Tensor],
    config: HamiltonianFitConfig,
) -> tuple[list[torch.Tensor], list[HamiltonianGaugeReport], torch.Tensor]:
    """Choose gauge representatives with one support shared by every head."""

    geometries = [
        _coefficient_nullspace(design, config) for design in designs
    ]
    raw_rows = [coeff.clone() for coeff in coeff_rows]
    if config.gauge_mode == "raw":
        bases = [raw.clone() for raw in raw_rows]
    else:
        bases = [
            raw - projector @ raw
            for raw, (_basis, projector, _rank, _tol) in zip(
                raw_rows, geometries
            )
        ]
    representatives = [base.clone() for base in bases]

    if config.gauge_mode == "sparsest" and representatives:
        tolerance = float(config.gauge_zero_tolerance)
        zero_indices = [
            index
            for index in range(representatives[0].numel())
            if all(float(base[index].abs().item()) <= tolerance for base in bases)
        ]
        remaining = [
            index
            for index in range(representatives[0].numel())
            if index not in zero_indices
        ]
        remaining.sort(
            key=lambda index: (
                sum(float(base[index].square().item()) for base in bases),
                index,
            )
        )
        scales = [max(float(torch.linalg.vector_norm(base).item()), 1.0) for base in bases]
        for index in remaining:
            trial_indices = zero_indices + [int(index)]
            trial_rows: list[torch.Tensor] = []
            feasible = True
            for base, geometry, scale in zip(bases, geometries, scales):
                basis = geometry[0]
                rows = basis[trial_indices]
                target = -base[trial_indices]
                if basis.shape[1] == 0:
                    error = target
                    trial = base.clone()
                else:
                    eta = torch.linalg.lstsq(rows, target).solution
                    error = rows @ eta - target
                    trial = base + basis @ eta
                if float(torch.linalg.vector_norm(error).item()) > tolerance * scale:
                    feasible = False
                    break
                trial[trial_indices] = 0.0
                trial_rows.append(trial)
            if feasible:
                representatives = trial_rows
                zero_indices = trial_indices

    representatives = [
        torch.where(
            representative.abs() > float(config.gauge_zero_tolerance),
            representative,
            torch.zeros_like(representative),
        )
        for representative in representatives
    ]
    shared_support = torch.stack(
        [
            representative.abs() > float(config.gauge_zero_tolerance)
            for representative in representatives
        ],
        dim=0,
    ).any(dim=0)

    reports: list[HamiltonianGaugeReport] = []
    for design, raw, representative, geometry in zip(
        designs, raw_rows, representatives, geometries
    ):
        basis, projector, rank, rank_tolerance = geometry
        difference = design @ (representative - raw)
        rms = float(difference.square().mean().sqrt().item())
        scale = float((design @ raw).square().mean().sqrt().item())
        relative = rms / max(scale, torch.finfo(design.dtype).eps)
        eliminated = tuple(
            int(index)
            for index in range(raw.numel())
            if float(raw[index].abs().item()) > float(config.gauge_zero_tolerance)
            and not bool(shared_support[index].item())
        )
        reports.append(
            HamiltonianGaugeReport(
                nullspace_basis=basis,
                nullspace_projector=projector,
                rank=rank,
                nullity=int(basis.shape[1]),
                rank_tolerance=rank_tolerance,
                representative_mode=f"{config.gauge_mode}_shared_support",
                raw_coeffs=raw,
                representative_coeffs=representative.clone(),
                flow_equivalence_rms=rms,
                flow_equivalence_relative=relative,
                eliminated_terms=eliminated,
            )
        )
    return representatives, reports, shared_support


def _make_hamiltonian_result(
    terms: Tuple[ScalarTerm, ...],
    coeffs: torch.Tensor,
    support: torch.Tensor,
    design: torch.Tensor,
    target: torch.Tensor,
    identifiable: torch.Tensor,
    config: HamiltonianFitConfig,
    gauge: HamiltonianGaugeReport | None = None,
) -> HamiltonianFitResult:
    prediction_flat = design @ coeffs
    error = prediction_flat - target
    rms = float(error.square().mean().sqrt().item())
    target_scale = float(target.square().mean().sqrt().item())
    relative = 0.0 if rms == 0.0 else rms / max(target_scale, 1.0e-30)
    accepted = rms <= config.absolute_residual_tolerance or (
        target_scale > 1.0e-30
        and relative <= config.relative_residual_tolerance
    )
    return HamiltonianFitResult(
        terms=terms,
        coeffs=coeffs,
        support=support,
        prediction=prediction_flat,
        rms=rms,
        relative_rms=relative,
        accepted=accepted,
        identifiable_terms=identifiable,
        complexity=int(support.sum().item()),
        gauge=gauge,
    )


def fit_hamiltonian_given_poisson(
    Pi: Any,
    field_values: torch.Tensor,
    Z: torch.Tensor,
    terms: Sequence[ScalarTerm],
    config: Optional[HamiltonianFitConfig] = None,
) -> HamiltonianFitResult:
    """Fit a sparse Hamiltonian after the Poisson tensor is fixed."""

    config = config or HamiltonianFitConfig()
    if tuple(field_values.shape) != tuple(Z.shape):
        raise ValueError("field_values and Z must both have shape (N,d)")
    terms_t = tuple(terms)
    design, _, identifiable = build_hamiltonian_design_matrix(Pi, Z, terms_t)
    target = field_values.reshape(-1)
    coeffs, support = _solve_linear_coefficients(design, target, config)
    # Pure gauge columns (notably the constant H term) are never scientific
    # support even if a regularized solver returns numerical crumbs.
    support = support & identifiable
    coeffs = torch.where(support, coeffs, torch.zeros_like(coeffs))
    if bool(support.any()):
        coeffs_active = torch.linalg.lstsq(
            design[:, support], target
        ).solution
        coeffs = coeffs.clone()
        coeffs[support] = coeffs_active
    coeffs, gauge = _apply_hamiltonian_gauge(design, coeffs, config)
    support = coeffs.abs() > float(config.gauge_zero_tolerance)
    coeffs = torch.where(support, coeffs, torch.zeros_like(coeffs))
    return _make_hamiltonian_result(
        terms_t,
        coeffs,
        support,
        design,
        target,
        identifiable,
        config,
        gauge,
    )


def fit_hamiltonians_given_poisson(
    Pi: Any,
    field_values_list: Sequence[torch.Tensor],
    state_points_list: Sequence[torch.Tensor],
    terms: Sequence[ScalarTerm],
    *,
    mode: str = "independent",
    config: Optional[HamiltonianFitConfig] = None,
) -> MultiHamiltonianFitResult:
    """Fit per-system Hamiltonians with independent/shared/full coefficients."""

    config = config or HamiltonianFitConfig()
    if len(field_values_list) != len(state_points_list) or not field_values_list:
        raise ValueError("field_values_list and state_points_list must match and be nonempty")
    mode = str(mode).lower()
    if mode not in {"independent", "shared_support", "fully_shared"}:
        raise ValueError("mode must be independent, shared_support, or fully_shared")
    terms_t = tuple(terms)
    designs = []
    targets = []
    identifiables = []
    for F, Z in zip(field_values_list, state_points_list):
        if tuple(F.shape) != tuple(Z.shape):
            raise ValueError("each field/state pair must have shape (N_d,d)")
        design, _, identifiable = build_hamiltonian_design_matrix(Pi, Z, terms_t)
        designs.append(design)
        targets.append(F.reshape(-1))
        identifiables.append(identifiable)

    coeff_rows: List[torch.Tensor]
    supports: List[torch.Tensor]
    if mode == "independent":
        solved = [_solve_linear_coefficients(A, y, config) for A, y in zip(designs, targets)]
        coeff_rows = [item[0] for item in solved]
        supports = [item[1] & identifiable for item, identifiable in zip(solved, identifiables)]
    elif mode == "shared_support":
        # Group-STLSQ uses normal equations for support selection.  A
        # degenerate Poisson tensor makes the Hamiltonian design singular by
        # construction (constants and Casimir combinations), so use a tiny
        # selection-only ridge even when the requested unbiased fit has
        # ridge=0.  Every selected head is refit by rank-aware lstsq below.
        selection_ridge = max(float(config.ridge), 1.0e-12)
        coeff_matrix, shared = group_stlsq(
            designs,
            targets,
            ridge=selection_ridge,
            lam=config.stlsq_lambda,
            max_iter=config.stlsq_max_iter,
            scale_columns=True,
        )
        shared = shared & torch.stack(identifiables).any(dim=0)
        coeff_rows = [coeff_matrix[d] for d in range(coeff_matrix.shape[0])]
        supports = [shared.clone() for _ in coeff_rows]
    else:
        stacked_design = torch.cat(designs, dim=0)
        stacked_target = torch.cat(targets, dim=0)
        coeff, shared = _solve_linear_coefficients(stacked_design, stacked_target, config)
        shared = shared & torch.stack(identifiables).any(dim=0)
        coeff_rows = [coeff.clone() for _ in designs]
        supports = [shared.clone() for _ in designs]

    # Unbiased dataset-specific refits on selected support.  Fully shared mode
    # gets one joint refit and then repeats that same coefficient row.
    if mode == "fully_shared":
        support = supports[0]
        coeff = torch.zeros_like(coeff_rows[0])
        if bool(support.any()):
            coeff[support] = torch.linalg.lstsq(
                torch.cat([A[:, support] for A in designs], dim=0),
                torch.cat(targets, dim=0),
            ).solution
        coeff_rows = [coeff.clone() for _ in designs]
    else:
        for d, (A, y, support) in enumerate(zip(designs, targets, supports)):
            coeff = torch.zeros_like(coeff_rows[d])
            if bool(support.any()):
                coeff[support] = torch.linalg.lstsq(A[:, support], y).solution
            coeff_rows[d] = coeff

    gauge_reports: list[HamiltonianGaugeReport]
    if mode == "fully_shared":
        representative, gauge = _apply_hamiltonian_gauge(
            torch.cat(designs, dim=0), coeff_rows[0], config
        )
        coeff_rows = [representative.clone() for _ in designs]
        supports = [
            representative.abs() > float(config.gauge_zero_tolerance)
            for _ in designs
        ]
        gauge_reports = [gauge for _ in designs]
    elif mode == "shared_support":
        coeff_rows, gauge_reports, shared_support = (
            _apply_shared_support_hamiltonian_gauge(
                designs, coeff_rows, config
            )
        )
        supports = [shared_support.clone() for _ in designs]
    else:
        gauged = [
            _apply_hamiltonian_gauge(A, coeff, config)
            for A, coeff in zip(designs, coeff_rows)
        ]
        coeff_rows = [item[0] for item in gauged]
        supports = [
            coeff.abs() > float(config.gauge_zero_tolerance)
            for coeff in coeff_rows
        ]
        gauge_reports = [item[1] for item in gauged]

    fits = tuple(
        _make_hamiltonian_result(
            terms_t,
            coeff,
            support,
            design,
            target,
            identifiable,
            config,
            gauge,
        )
        for coeff, support, design, target, identifiable, gauge in zip(
            coeff_rows,
            supports,
            designs,
            targets,
            identifiables,
            gauge_reports,
        )
    )
    coeffs = torch.stack(coeff_rows, dim=0)
    union_support = torch.stack(supports, dim=0).any(dim=0)
    return MultiHamiltonianFitResult(
        mode=mode,
        terms=terms_t,
        coeffs=coeffs,
        support=union_support,
        fits=fits,
        accepted=all(fit.accepted for fit in fits),
    )


def _monomial_ast(exponents: Sequence[int]) -> Any:
    from nestynet_sr.sr_core.bridges import ConstNode, MulNode, PowNode, Var

    factors = []
    for axis, power in enumerate(exponents):
        if int(power) == 0:
            continue
        variable = Var(int(axis))
        factors.append(variable if int(power) == 1 else PowNode(variable, float(power)))
    if not factors:
        return ConstNode(1.0)
    out = factors[0]
    for factor in factors[1:]:
        out = MulNode(out, factor)
    return out


def scalar_combination_ast(
    terms: Sequence[ScalarTerm],
    coeffs: torch.Tensor,
    *,
    coefficient_tolerance: float = 1.0e-10,
) -> Any | None:
    """Render a supported scalar-library combination as a NestyNet AST."""

    from nestynet_sr.sr_core.bridges import (
        AbsNode,
        AcosNode,
        AddNode,
        ArgNode,
        AsinNode,
        AtanNode,
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
    )
    from nestynet_sr.sr_core.ast_simplify import simplify_ast

    nodes = []
    for coefficient, term in zip(coeffs.detach().cpu().tolist(), terms):
        value = float(coefficient)
        if abs(value) <= float(coefficient_tolerance):
            continue
        if isinstance(term, (tuple, list)) and all(
            isinstance(power, int) and not isinstance(power, bool) for power in term
        ):
            node = _monomial_ast(term)
        elif isinstance(
            term,
            (
                AtomNode,
                AddNode,
                MulNode,
                PowNode,
                LogNode,
                ExpNode,
                SinNode,
                CosNode,
                AsinNode,
                AcosNode,
                AtanNode,
                ConjNode,
                RealNode,
                ImagNode,
                AbsNode,
                ArgNode,
                ConstNode,
            ),
        ):
            node = term
        else:
            return None
        if abs(value - 1.0) > float(coefficient_tolerance):
            node = MulNode(ConstNode(value), node)
        nodes.append(node)
    if not nodes:
        return ConstNode(0.0)
    out = nodes[0]
    for node in nodes[1:]:
        out = AddNode(out, node)
    return simplify_ast(out)


def _render_scalar_combination(
    terms: Sequence[ScalarTerm], coeffs: torch.Tensor
) -> tuple[str, Any | None]:
    ast = scalar_combination_ast(terms, coeffs)
    if ast is not None:
        from nestynet_sr.sr_core.bridges import ast_to_human_readable

        return ast_to_human_readable(ast), ast
    pieces = []
    for index, (coefficient, term) in enumerate(
        zip(coeffs.detach().cpu().tolist(), terms)
    ):
        if abs(float(coefficient)) <= 1.0e-10:
            continue
        name = getattr(term, "name", None) or getattr(term, "__name__", None)
        pieces.append(f"{float(coefficient):+.8g}*{name or f'phi_{index}'}")
    return " ".join(pieces).lstrip("+"), None


def _scaled_casimir_residual(
    tensor: torch.Tensor,
    gradient: torch.Tensor,
    field_values: torch.Tensor | None,
) -> tuple[float, float, float | None, float | None]:
    poisson_flow = torch.einsum("nij,nj->ni", tensor, gradient)
    poisson_rms = float(poisson_flow.square().mean().sqrt().item())
    gradient_norm = torch.linalg.vector_norm(gradient, dim=1)
    poisson_scale = float(
        (
            torch.linalg.matrix_norm(tensor, dim=(-2, -1)) * gradient_norm
        ).square().mean().sqrt().item()
    )
    poisson_relative = poisson_rms / max(
        poisson_scale, torch.finfo(tensor.dtype).eps
    )
    if field_values is None:
        return poisson_rms, poisson_relative, None, None
    drift = torch.einsum("ni,ni->n", field_values, gradient)
    flow_rms = float(drift.square().mean().sqrt().item())
    flow_scale = float(
        (
            torch.linalg.vector_norm(field_values, dim=1) * gradient_norm
        ).square().mean().sqrt().item()
    )
    flow_relative = flow_rms / max(flow_scale, torch.finfo(tensor.dtype).eps)
    return poisson_rms, poisson_relative, flow_rms, flow_relative


def _generic_tensor_rank(
    tensor: torch.Tensor,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> int:
    singular_values = torch.linalg.svdvals(tensor)
    leading = singular_values[:, :1].clamp_min(1.0)
    thresholds = torch.maximum(
        singular_values.new_full((tensor.shape[0], 1), float(absolute_tolerance)),
        float(relative_tolerance) * leading,
    )
    ranks = torch.count_nonzero(singular_values > thresholds, dim=1)
    return int(ranks.max().item()) if ranks.numel() else 0


def _candidate_independence_fraction(
    gradient: torch.Tensor,
    selected_gradients: Sequence[torch.Tensor],
    *,
    tolerance: float,
) -> float:
    norm = torch.linalg.vector_norm(gradient, dim=1)
    noncollapsed = norm > float(tolerance)
    if not selected_gradients:
        return float(noncollapsed.to(gradient.dtype).mean().item())
    basis = torch.stack(tuple(selected_gradients), dim=2)  # (N,d,m)
    coefficients = torch.linalg.lstsq(basis, gradient.unsqueeze(2)).solution
    projected = torch.matmul(basis, coefficients).squeeze(2)
    residual = torch.linalg.vector_norm(gradient - projected, dim=1)
    independent = noncollapsed & (
        residual / norm.clamp_min(torch.finfo(gradient.dtype).eps)
        > float(tolerance)
    )
    return float(independent.to(gradient.dtype).mean().item())


def _gradient_span_projector(
    gradients: Sequence[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if not gradients:
        return reference.new_zeros(
            (reference.shape[0], reference.shape[1], reference.shape[1])
        )
    matrix = torch.stack(tuple(gradients), dim=2)
    gram = matrix.transpose(1, 2) @ matrix
    return matrix @ torch.linalg.pinv(gram) @ matrix.transpose(1, 2)


def _casimir_coordinate(
    candidate: CasimirCandidate,
    terms: Tuple[ScalarTerm, ...],
    index: int,
) -> Any:
    from nestynet_sr.sr_gs.quotient import CoordinateSpec

    coeffs = candidate.coeffs.detach().cpu().clone()

    def evaluator(values: Any) -> Any:
        import numpy as np

        array = np.asarray(values, dtype=np.float64)
        points = torch.as_tensor(array, dtype=torch.float64)
        evaluated = evaluate_scalar_library(points, terms)
        return (evaluated.values @ coeffs.to(points)).detach().cpu().numpy()

    return CoordinateSpec(
        name=f"casimir_{index}",
        kind="casimir",
        ast=candidate.ast,
        coordinate_map=None,
        gauge="univariate_reparameterization",
        evaluator=None if candidate.ast is not None else evaluator,
        provenance={
            "source": "poisson_conormal_distribution",
            "expression": candidate.expression,
            "poisson_residual_relative": candidate.poisson_residual_relative,
            "heldout_poisson_residual_relative": candidate.heldout_poisson_residual_relative,
            "gradient_rank_gain": candidate.gradient_rank_gain,
            "independence_fraction": candidate.independence_fraction,
            "heldout_independence_fraction": candidate.heldout_independence_fraction,
        },
    )


def score_casimir_carrier(
    Pi: Any,
    Z: torch.Tensor,
    term: ScalarTerm,
    *,
    validation_points: torch.Tensor | None = None,
    validation_poisson: Any | None = None,
    relative_residual_tolerance: float = 1.0e-7,
    absolute_residual_tolerance: float = 1.0e-10,
    gradient_floor: float = 1.0e-10,
    variation_floor: float = 1.0e-12,
) -> CasimirCarrierScore:
    """Score one FSS carrier directly by ``Pi grad(s)``; no outer map is fit."""

    value, gradient = evaluate_scalar_term(Z, term)
    tensor = evaluate_poisson_tensor(Pi, Z)
    poisson_rms, poisson_relative, _flow_rms, _flow_relative = (
        _scaled_casimir_residual(tensor, gradient, None)
    )
    variation = float(torch.var(value, unbiased=False).item())
    gradient_rms = float(gradient.square().mean().sqrt().item())
    heldout_points = Z if validation_points is None else validation_points
    heldout_value, heldout_gradient = evaluate_scalar_term(heldout_points, term)
    if (
        validation_points is not None
        and validation_poisson is None
        and isinstance(Pi, torch.Tensor)
        and tuple(Pi.shape) == (Z.shape[0], Z.shape[1], Z.shape[1])
    ):
        raise ValueError(
            "validation_poisson is required when Pi is an N-sample tensor "
            "and explicit validation_points are supplied"
        )
    heldout_tensor = evaluate_poisson_tensor(
        Pi if validation_poisson is None else validation_poisson,
        heldout_points,
    )
    (
        heldout_poisson_rms,
        heldout_poisson_relative,
        _heldout_flow_rms,
        _heldout_flow_relative,
    ) = _scaled_casimir_residual(heldout_tensor, heldout_gradient, None)
    heldout_variation = float(torch.var(heldout_value, unbiased=False).item())
    heldout_gradient_rms = float(
        heldout_gradient.square().mean().sqrt().item()
    )
    finite = bool(
        torch.isfinite(value).all().item()
        and torch.isfinite(gradient).all().item()
        and torch.isfinite(heldout_value).all().item()
        and torch.isfinite(heldout_gradient).all().item()
        and math.isfinite(poisson_relative)
        and math.isfinite(heldout_poisson_relative)
    )
    failures = []
    if not finite:
        failures.append("nonfinite")
    if variation <= float(variation_floor):
        failures.append("collapsed_variation")
    if heldout_variation <= float(variation_floor):
        failures.append("heldout_collapsed_variation")
    if gradient_rms <= float(gradient_floor):
        failures.append("collapsed_gradient")
    if heldout_gradient_rms <= float(gradient_floor):
        failures.append("heldout_collapsed_gradient")
    if not (
        poisson_rms <= float(absolute_residual_tolerance)
        or poisson_relative <= float(relative_residual_tolerance)
    ):
        failures.append("poisson_residual")
    if not (
        heldout_poisson_rms <= float(absolute_residual_tolerance)
        or heldout_poisson_relative <= float(relative_residual_tolerance)
    ):
        failures.append("heldout_poisson_residual")
    return CasimirCarrierScore(
        poisson_residual_rms=poisson_rms,
        poisson_residual_relative=poisson_relative,
        variation=variation,
        gradient_rms=gradient_rms,
        heldout_poisson_residual_rms=heldout_poisson_rms,
        heldout_poisson_residual_relative=heldout_poisson_relative,
        heldout_variation=heldout_variation,
        heldout_gradient_rms=heldout_gradient_rms,
        finite=finite,
        accepted=not failures,
        failure_reasons=tuple(failures),
    )


def discover_casimirs(
    Pi: Any,
    Z: torch.Tensor,
    terms: Sequence[ScalarTerm],
    *,
    field_values: Optional[torch.Tensor] = None,
    validation_points: Optional[torch.Tensor] = None,
    validation_poisson: Any | None = None,
    validation_field_values: Optional[torch.Tensor] = None,
    validation_fraction: float = 0.2,
    random_seed: int = 0,
    nullspace_config: Optional[StableNullspaceConfig] = None,
    coefficient_threshold: float = 1.0e-8,
    relative_residual_tolerance: float = 1.0e-7,
    absolute_residual_tolerance: float = 1.0e-10,
    rank_relative_tolerance: float = 1.0e-8,
    rank_absolute_tolerance: float = 1.0e-10,
    max_representatives: int = 64,
    sparse_rotation_steps: int = 16,
    independence_tolerance: float = 1.0e-6,
    minimum_independence_fraction: float = 0.8,
    require_nullspace_stability: bool = True,
    nullspace_max_principal_angle: float = 0.35,
) -> CasimirDiscoveryResult:
    """Discover a basis-independent, functionally complete Casimir foliation.

    Coefficient-nullspace vectors are only candidate representations.  The
    promoted result greedily selects carriers whose gradients add a generic
    conormal direction, up to ``dim(M)-rank(Pi)``.  Held-out Poisson and flow
    residuals are hard acceptance gates.
    """

    if not 0.0 <= float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie in [0,1)")
    if not 0.0 <= float(minimum_independence_fraction) <= 1.0:
        raise ValueError("minimum_independence_fraction must lie in [0,1]")
    if Z.ndim != 2 or Z.shape[0] < 1:
        raise ValueError("Z must have shape (N,d)")
    terms_t = tuple(terms)
    if not terms_t:
        raise ValueError("terms must contain at least one scalar carrier")
    if field_values is not None and tuple(field_values.shape) != tuple(Z.shape):
        raise ValueError("field_values must have shape (N,d)")

    train_indices: torch.Tensor | None = None
    heldout_indices: torch.Tensor | None = None
    if validation_points is not None:
        if validation_points.ndim != 2 or validation_points.shape[1] != Z.shape[1]:
            raise ValueError("validation_points must have shape (N_validation,d)")
        train_points = Z
        heldout_points = validation_points
        train_field = field_values
        heldout_field = validation_field_values
        if heldout_field is not None and tuple(heldout_field.shape) != tuple(
            heldout_points.shape
        ):
            raise ValueError(
                "validation_field_values must match validation_points"
            )
    elif float(validation_fraction) > 0.0 and Z.shape[0] >= 5:
        heldout_count = max(
            1,
            min(
                Z.shape[0] - 2,
                int(round(float(validation_fraction) * Z.shape[0])),
            ),
        )
        generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
        permutation = torch.randperm(Z.shape[0], generator=generator).to(Z.device)
        heldout_indices = permutation[:heldout_count]
        train_indices = permutation[heldout_count:]
        train_points = Z[train_indices]
        heldout_points = Z[heldout_indices]
        train_field = None if field_values is None else field_values[train_indices]
        heldout_field = (
            None if field_values is None else field_values[heldout_indices]
        )
    else:
        train_points = Z
        heldout_points = Z
        train_field = field_values
        heldout_field = field_values

    sample_tensor = (
        Pi
        if isinstance(Pi, torch.Tensor)
        and tuple(Pi.shape) == (Z.shape[0], Z.shape[1], Z.shape[1])
        else None
    )
    if sample_tensor is not None and train_indices is not None:
        train_tensor = sample_tensor[train_indices]
        assert heldout_indices is not None
        heldout_tensor = sample_tensor[heldout_indices]
    elif sample_tensor is not None and validation_points is not None:
        if validation_poisson is None:
            raise ValueError(
                "validation_poisson is required when Pi is an N-sample tensor "
                "and explicit validation_points are supplied"
            )
        train_tensor = sample_tensor
        heldout_tensor = evaluate_poisson_tensor(
            validation_poisson, heldout_points
        )
    elif sample_tensor is not None and validation_points is None:
        train_tensor = sample_tensor
        heldout_tensor = sample_tensor
    else:
        train_tensor = evaluate_poisson_tensor(Pi, train_points)
        heldout_tensor = evaluate_poisson_tensor(
            Pi if validation_poisson is None else validation_poisson,
            heldout_points,
        )
    train_library = evaluate_scalar_library(train_points, terms_t)
    heldout_library = evaluate_scalar_library(heldout_points, terms_t)
    train_flows = torch.einsum(
        "nij,nkj->nki", train_tensor, train_library.gradients
    )
    heldout_flows = torch.einsum(
        "nij,nkj->nki", heldout_tensor, heldout_library.gradients
    )
    full_design = train_flows.permute(0, 2, 1).reshape(
        train_points.shape[0] * train_points.shape[1], len(terms_t)
    )
    heldout_design = heldout_flows.permute(0, 2, 1).reshape(
        heldout_points.shape[0] * heldout_points.shape[1], len(terms_t)
    )
    identifiable = (
        train_library.gradients.square().mean(dim=(0, 2)).sqrt() > 1.0e-14
    )
    if not bool(identifiable.any()):
        raise ValueError("scalar library contains no nonconstant carrier")
    reduced = full_design[:, identifiable]
    heldout_reduced = heldout_design[:, identifiable]
    policy = nullspace_config or StableNullspaceConfig(bootstrap=2)
    policy = replace(policy, bootstrap_block_size=int(Z.shape[1]))
    nullspace = stable_nullspace(
        reduced,
        heldout_reduced,
        config=policy,
    )

    combined_tensor = torch.cat((train_tensor, heldout_tensor), dim=0)
    generic_rank = _generic_tensor_rank(
        combined_tensor,
        relative_tolerance=rank_relative_tolerance,
        absolute_tolerance=rank_absolute_tolerance,
    )
    expected_corank = max(0, int(Z.shape[1]) - int(generic_rank))

    representatives = sparse_nullspace_representatives(
        nullspace,
        max_representatives=max_representatives,
        sparse_rotation_steps=sparse_rotation_steps,
        random_seed=random_seed,
    )
    evaluated: List[
        tuple[CasimirCandidate, torch.Tensor, torch.Tensor]
    ] = []
    canonical_coefficients: List[torch.Tensor] = []
    for representative in representatives:
        coeff_reduced = representative
        coeff_reduced = torch.where(
            coeff_reduced.abs() >= coefficient_threshold,
            coeff_reduced,
            torch.zeros_like(coeff_reduced),
        )
        norm = torch.linalg.vector_norm(coeff_reduced)
        if float(norm.item()) <= 0.0:
            continue
        support_reduced = coeff_reduced.abs() >= coefficient_threshold
        support_design = reduced[:, support_reduced]
        if support_design.shape[1] == 0:
            continue
        _u, _s, vh = torch.linalg.svd(support_design, full_matrices=False)
        sparse_direction = vh[-1]
        sparse_direction = sparse_direction / torch.linalg.vector_norm(
            sparse_direction
        ).clamp_min(torch.finfo(sparse_direction.dtype).eps)
        coeff_reduced = torch.zeros_like(coeff_reduced)
        coeff_reduced[support_reduced] = sparse_direction
        coeff = Z.new_zeros(len(terms_t))
        coeff[identifiable] = coeff_reduced
        coeff_norm = torch.linalg.vector_norm(coeff)
        if float(coeff_norm.item()) <= 0.0:
            continue
        coeff = coeff / coeff_norm
        pivot = int(torch.argmax(coeff.abs()).item())
        if float(coeff[pivot].item()) < 0.0:
            coeff = -coeff
        if any(
            float(torch.abs(torch.dot(coeff, old)).item()) > 1.0 - 1.0e-7
            for old in canonical_coefficients
        ):
            continue
        canonical_coefficients.append(coeff)
        train_gradient = torch.einsum(
            "k,nkd->nd", coeff, train_library.gradients
        )
        heldout_gradient = torch.einsum(
            "k,nkd->nd", coeff, heldout_library.gradients
        )
        poisson_rms, poisson_relative, flow_rms, flow_relative = (
            _scaled_casimir_residual(train_tensor, train_gradient, train_field)
        )
        (
            heldout_poisson_rms,
            heldout_poisson_relative,
            heldout_flow_rms,
            heldout_flow_relative,
        ) = _scaled_casimir_residual(
            heldout_tensor, heldout_gradient, heldout_field
        )
        gradient_rms = float(train_gradient.square().mean().sqrt().item())
        failures: List[str] = []
        if gradient_rms <= 1.0e-14:
            failures.append("collapsed_gradient")
        if not (
            poisson_rms <= float(absolute_residual_tolerance)
            or poisson_relative <= float(relative_residual_tolerance)
        ):
            failures.append("poisson_residual")
        if flow_rms is not None and flow_relative is not None and not (
            flow_rms <= float(absolute_residual_tolerance)
            or flow_relative <= float(relative_residual_tolerance)
        ):
            failures.append("flow_residual")
        if not (
            heldout_poisson_rms <= float(absolute_residual_tolerance)
            or heldout_poisson_relative <= float(relative_residual_tolerance)
        ):
            failures.append("heldout_poisson_residual")
        if (
            heldout_flow_rms is not None
            and heldout_flow_relative is not None
            and not (
                heldout_flow_rms <= float(absolute_residual_tolerance)
                or heldout_flow_relative <= float(relative_residual_tolerance)
            )
        ):
            failures.append("heldout_flow_residual")
        expression, ast = _render_scalar_combination(terms_t, coeff)
        candidate = CasimirCandidate(
            coeffs=coeff,
            poisson_residual_rms=poisson_rms,
            poisson_residual_relative=poisson_relative,
            flow_residual_rms=flow_rms,
            flow_residual_relative=flow_relative,
            gradient_rms=gradient_rms,
            complexity=int((coeff.abs() >= coefficient_threshold).sum().item()),
            accepted=not failures,
            failure_reasons=tuple(failures),
            heldout_poisson_residual_rms=heldout_poisson_rms,
            heldout_poisson_residual_relative=heldout_poisson_relative,
            heldout_flow_residual_rms=heldout_flow_rms,
            heldout_flow_residual_relative=heldout_flow_relative,
            expression=expression,
            ast=ast,
        )
        evaluated.append((candidate, train_gradient, heldout_gradient))

    evaluated.sort(
        key=lambda item: (
            0 if item[0].accepted else 1,
            item[0].complexity,
            item[0].heldout_poisson_residual_relative
            if item[0].heldout_poisson_residual_relative is not None
            else math.inf,
            item[0].poisson_residual_relative,
        )
    )
    selected: List[CasimirCandidate] = []
    selected_train_gradients: List[torch.Tensor] = []
    selected_heldout_gradients: List[torch.Tensor] = []
    rejected: List[CasimirCandidate] = []
    for candidate, train_gradient, heldout_gradient in evaluated:
        if not candidate.accepted:
            rejected.append(candidate)
            continue
        train_fraction = _candidate_independence_fraction(
            train_gradient,
            selected_train_gradients,
            tolerance=independence_tolerance,
        )
        heldout_fraction = _candidate_independence_fraction(
            heldout_gradient,
            selected_heldout_gradients,
            tolerance=independence_tolerance,
        )
        candidate.independence_fraction = train_fraction
        candidate.heldout_independence_fraction = heldout_fraction
        if (
            len(selected) >= expected_corank
            or train_fraction < float(minimum_independence_fraction)
            or heldout_fraction < float(minimum_independence_fraction)
        ):
            candidate.accepted = False
            candidate.failure_reasons = tuple(
                (*candidate.failure_reasons, "functionally_dependent")
            )
            rejected.append(candidate)
            continue
        candidate.selected = True
        candidate.gradient_rank_gain = 1
        candidate.coordinate = _casimir_coordinate(
            candidate, terms_t, len(selected)
        )
        selected.append(candidate)
        selected_train_gradients.append(train_gradient)
        selected_heldout_gradients.append(heldout_gradient)

    stability_angles = [
        angle
        for angle in (
            nullspace.heldout_principal_angle,
            *nullspace.bootstrap_principal_angles,
        )
        if angle is not None
    ]
    nullspace_stable = bool(
        not require_nullspace_stability
        or (
            stability_angles
            and max(stability_angles) <= float(nullspace_max_principal_angle)
        )
    )
    if not nullspace_stable:
        for candidate in selected:
            candidate.accepted = False
            candidate.selected = False
            candidate.gradient_rank_gain = 0
            candidate.coordinate = None
            candidate.failure_reasons = tuple(
                (*candidate.failure_reasons, "nullspace_unstable")
            )
            rejected.append(candidate)
        selected = []
        selected_train_gradients = []
        selected_heldout_gradients = []

    full_library = evaluate_scalar_library(Z, terms_t)
    full_gradients = [
        torch.einsum("k,nkd->nd", candidate.coeffs, full_library.gradients)
        for candidate in selected
    ]
    gradient_projector = _gradient_span_projector(full_gradients, Z)
    discovered_corank = len(selected)
    complete = bool(discovered_corank == expected_corank)
    if not nullspace_stable:
        status = "unstable_nullspace"
    elif expected_corank == 0:
        status = "full_rank_no_nonconstant_casimir"
    elif complete:
        status = "complete"
    else:
        status = "incomplete_library"
    return CasimirDiscoveryResult(
        terms=terms_t,
        nullspace=nullspace,
        identifiable_terms=identifiable,
        candidates=tuple(selected),
        rejected_candidates=tuple(rejected),
        expected_corank=expected_corank,
        discovered_corank=discovered_corank,
        complete=complete,
        status=status,
        generic_poisson_rank=generic_rank,
        gradient_projector=gradient_projector,
        coordinates=tuple(candidate.coordinate for candidate in selected),
    )
