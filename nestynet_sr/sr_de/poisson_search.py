# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.

"""Polynomial Poisson discovery through invariant-bivector nullspaces.

The public funnel is deliberately staged:

``L_f Pi = 0`` (linear determining solve)
    -> ``[Pi,Pi] = 0`` (sampled and polynomial Jacobi certificates)
    -> ``f = Pi grad H`` (linear fixed-Pi Hamiltonian fit).

Constant, homogeneous-linear, affine, and quadratic tensor lanes are tried in
that order by default.  Single- and multi-vector-field entry points share the
same implementation; in the multi-system case the determining matrices are
stacked while each system can have an independent, shared-support, or fully
shared Hamiltonian head.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, List, Optional, Sequence, Tuple

import torch

from nestynet_sr.sr_de.poisson_basis import BivectorBasis, PolynomialScalarBasis
from nestynet_sr.sr_de.poisson_certificates import (
    PolynomialJacobiCertificate,
    RankProfile,
    ScaledResidual,
    invariance_residual,
    jacobi_residual,
    polynomial_jacobi_certificate,
    rank_profile,
    skew_residual,
)
from nestynet_sr.sr_de.poisson_core import (
    StableNullspaceConfig,
    StableNullspaceResult,
    VectorField,
    sparse_nullspace_representatives,
    stable_nullspace,
    validate_state_points,
)
from nestynet_sr.sr_de.poisson_invariants import (
    CasimirDiscoveryResult,
    HamiltonianFitConfig,
    HamiltonianFitResult,
    MultiHamiltonianFitResult,
    ScalarTerm,
    build_hamiltonian_design_matrix,
    discover_casimirs,
    fit_hamiltonian_given_poisson,
    fit_hamiltonians_given_poisson,
)


_LANE_SPEC = {
    "constant": (0, True),
    "linear": (1, False),
    "affine": (1, True),
    "quadratic": (2, True),
}


@dataclass(frozen=True)
class PoissonSearchConfig:
    lanes: Tuple[str, ...] = ("constant", "linear", "affine", "quadratic")
    validation_fraction: float = 0.2
    random_seed: int = 0
    stop_at_first_accepted_lane: bool = True
    normalize_dataset_blocks: bool = True

    nullspace: StableNullspaceConfig = field(
        default_factory=lambda: StableNullspaceConfig(
            rank_rtol=1.0e-9,
            rank_atol=1.0e-11,
        )
    )
    max_representatives: int = 48
    sparse_rotation_steps: int = 16
    coefficient_tolerance: float = 1.0e-8

    invariance_relative_tolerance: float = 1.0e-6
    invariance_absolute_tolerance: float = 1.0e-9
    jacobi_relative_tolerance: float = 1.0e-6
    jacobi_absolute_tolerance: float = 1.0e-9
    polynomial_jacobi_tolerance: float = 1.0e-8
    rank_relative_tolerance: float = 1.0e-8
    rank_absolute_tolerance: float = 1.0e-10
    minimum_rank_stable_fraction: float = 0.9
    require_nonzero_rank: bool = True
    require_nullspace_stability: bool = True
    nullspace_max_principal_angle: float = 0.35

    hamiltonian: HamiltonianFitConfig = field(default_factory=HamiltonianFitConfig)
    hamiltonian_mode: str = "independent"
    require_hamiltonian: bool = True
    hamiltonian_library_degree: int = 2
    discover_casimirs: bool = True
    casimir_library_degree: int = 2
    require_complete_casimirs: bool = True
    casimir_incompleteness_weight: float = 1.0

    complexity_weight: float = 1.0e-4

    def __post_init__(self) -> None:
        unknown = [lane for lane in self.lanes if lane not in _LANE_SPEC]
        if unknown:
            raise ValueError(f"unknown Poisson lanes: {unknown}")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0,1)")
        if self.max_representatives < 1:
            raise ValueError("max_representatives must be positive")
        if self.sparse_rotation_steps < 0:
            raise ValueError("sparse_rotation_steps must be non-negative")
        if self.casimir_incompleteness_weight < 0.0:
            raise ValueError("casimir_incompleteness_weight must be non-negative")
        if not 0.0 <= self.nullspace_max_principal_angle <= 0.5 * math.pi:
            raise ValueError("nullspace_max_principal_angle must lie in [0,pi/2]")
        if self.hamiltonian_mode not in {
            "independent",
            "shared_support",
            "fully_shared",
        }:
            raise ValueError("invalid hamiltonian_mode")


@dataclass
class PoissonCandidate:
    lane: str
    degree: int
    include_constant: bool
    coefficients: torch.Tensor
    exponents: Tuple[Tuple[int, ...], ...]
    basis: BivectorBasis
    nullspace: StableNullspaceResult
    skew: ScaledResidual
    invariance: Tuple[ScaledResidual, ...]
    jacobi: ScaledResidual
    polynomial_jacobi: PolynomialJacobiCertificate
    rank: RankProfile
    hamiltonian: HamiltonianFitResult | MultiHamiltonianFitResult | None
    hamiltonian_validation_relative: Tuple[float, ...]
    casimirs: CasimirDiscoveryResult | None
    complexity: int
    score: float
    accepted: bool
    failure_reasons: Tuple[str, ...]

    def evaluate(self, Z: torch.Tensor):
        """Evaluate the represented tensor and coordinate derivatives."""

        return self.basis.assemble(self.coefficients, Z)

    @property
    def coefficient_table(self) -> torch.Tensor:
        return self.coefficients.reshape(len(self.basis.pairs), self.basis.scalar_size)


@dataclass(frozen=True)
class PoissonLaneResult:
    lane: str
    degree: int
    include_constant: bool
    nullspace: StableNullspaceResult
    candidates: Tuple[PoissonCandidate, ...]


@dataclass
class PoissonSearchResult:
    state_dim: int
    dataset_count: int
    lanes: Tuple[PoissonLaneResult, ...]
    candidates: Tuple[PoissonCandidate, ...]
    pareto_candidates: Tuple[PoissonCandidate, ...]
    best: PoissonCandidate | None
    accepted: bool


def _coerce_vector_field(field_like: Any, state_dim: int) -> VectorField:
    if isinstance(field_like, VectorField):
        if field_like.state_dim is not None and field_like.state_dim != state_dim:
            raise ValueError("vector field state dimension does not match state points")
        return field_like
    if callable(field_like):
        return VectorField.from_callable(field_like, state_dim=state_dim)
    value = getattr(field_like, "value", None)
    jacobian = getattr(field_like, "jacobian", None)
    if callable(value):
        return VectorField.from_callable(
            value,
            jacobian if callable(jacobian) else None,
            state_dim=state_dim,
        )
    raise TypeError("field must be a VectorField, callable, or value/jacobian object")


def _train_validation_split(
    Z: torch.Tensor,
    validation: torch.Tensor | None,
    *,
    fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_state_points(Z)
    if validation is not None:
        validate_state_points(validation, Z.shape[1])
        return Z, validation
    if fraction <= 0.0 or Z.shape[0] < 5:
        return Z, Z
    count = max(1, min(Z.shape[0] - 2, int(round(fraction * Z.shape[0]))))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(Z.shape[0], generator=generator).to(Z.device)
    return Z[permutation[count:]], Z[permutation[:count]]


def _normalise_determining_block(matrix: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.matrix_norm(matrix)
    if float(norm.item()) == 0.0:
        return matrix
    return matrix * (math.sqrt(float(matrix.shape[0])) / norm)


def _representatives(
    nullspace: StableNullspaceResult,
    config: PoissonSearchConfig,
) -> Tuple[torch.Tensor, ...]:
    """Generate deterministic sparse representatives inside a nullspace."""

    return sparse_nullspace_representatives(
        nullspace,
        max_representatives=config.max_representatives,
        sparse_rotation_steps=config.sparse_rotation_steps,
        random_seed=config.random_seed,
    )


def _default_scalar_terms(state_dim: int, degree: int) -> Tuple[Tuple[int, ...], ...]:
    return PolynomialScalarBasis(
        state_dim,
        max_degree=degree,
        include_constant=True,
    ).exponents


def _hamiltonian_validation_error(
    candidate: PoissonCandidate | None,
    basis: BivectorBasis,
    coefficients: torch.Tensor,
    fit: HamiltonianFitResult,
    F: torch.Tensor,
    Z: torch.Tensor,
) -> float:
    del candidate
    represented = basis.assemble(coefficients, Z)
    design, _, _ = build_hamiltonian_design_matrix(
        represented.tensor, Z, fit.terms
    )
    error = design @ fit.coeffs - F.reshape(-1)
    rms = float(error.square().mean().sqrt().item())
    scale = float(F.square().mean().sqrt().item())
    return 0.0 if rms == 0.0 else rms / max(scale, 1.0e-30)


def _pareto_front(candidates: Sequence[PoissonCandidate]) -> Tuple[PoissonCandidate, ...]:
    accepted = [candidate for candidate in candidates if candidate.accepted]
    front: List[PoissonCandidate] = []
    for candidate in accepted:
        objectives = (
            max((item.relative for item in candidate.invariance), default=math.inf),
            candidate.jacobi.relative,
            max(candidate.hamiltonian_validation_relative, default=math.inf),
            float(candidate.complexity),
        )
        dominated = False
        for other in accepted:
            if other is candidate:
                continue
            other_objectives = (
                max((item.relative for item in other.invariance), default=math.inf),
                other.jacobi.relative,
                max(other.hamiltonian_validation_relative, default=math.inf),
                float(other.complexity),
            )
            if all(a <= b for a, b in zip(other_objectives, objectives)) and any(
                a < b for a, b in zip(other_objectives, objectives)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return tuple(sorted(front, key=lambda item: item.score))


def discover_poisson_structure(
    rhs: Any,
    state_points: torch.Tensor,
    config: Optional[PoissonSearchConfig] = None,
    *,
    hamiltonian_terms: Optional[Sequence[ScalarTerm]] = None,
    casimir_terms: Optional[Sequence[ScalarTerm]] = None,
    validation_points: Optional[torch.Tensor] = None,
) -> PoissonSearchResult:
    """Discover a polynomial Poisson structure for one autonomous vector field."""

    return discover_poisson_structure_multi(
        [rhs],
        [state_points],
        config,
        hamiltonian_terms=hamiltonian_terms,
        casimir_terms=casimir_terms,
        validation_points_list=None if validation_points is None else [validation_points],
    )


def discover_poisson_structure_multi(
    rhs_list: Sequence[Any],
    state_points_list: Sequence[torch.Tensor],
    config: Optional[PoissonSearchConfig] = None,
    *,
    hamiltonian_terms: Optional[Sequence[ScalarTerm]] = None,
    casimir_terms: Optional[Sequence[ScalarTerm]] = None,
    validation_points_list: Optional[Sequence[torch.Tensor]] = None,
) -> PoissonSearchResult:
    """Discover one shared Poisson tensor from several autonomous systems."""

    config = config or PoissonSearchConfig()
    if len(rhs_list) != len(state_points_list) or not rhs_list:
        raise ValueError("rhs_list and state_points_list must match and be nonempty")
    if validation_points_list is not None and len(validation_points_list) != len(rhs_list):
        raise ValueError("validation_points_list must match rhs_list")
    state_dim = int(state_points_list[0].shape[1])
    for Z in state_points_list:
        validate_state_points(Z, state_dim)
    fields = tuple(_coerce_vector_field(rhs, state_dim) for rhs in rhs_list)

    train_points: List[torch.Tensor] = []
    validation_points: List[torch.Tensor] = []
    train_values: List[torch.Tensor] = []
    train_jacobians: List[torch.Tensor] = []
    validation_values: List[torch.Tensor] = []
    validation_jacobians: List[torch.Tensor] = []
    for dataset, (field_obj, Z) in enumerate(zip(fields, state_points_list)):
        explicit_validation = (
            None if validation_points_list is None else validation_points_list[dataset]
        )
        Z_train, Z_validation = _train_validation_split(
            Z,
            explicit_validation,
            fraction=config.validation_fraction,
            seed=config.random_seed + dataset,
        )
        F_train, DF_train = field_obj.value_and_jacobian(Z_train)
        F_validation, DF_validation = field_obj.value_and_jacobian(Z_validation)
        train_points.append(Z_train)
        validation_points.append(Z_validation)
        train_values.append(F_train)
        train_jacobians.append(DF_train)
        validation_values.append(F_validation)
        validation_jacobians.append(DF_validation)

    h_terms = tuple(hamiltonian_terms) if hamiltonian_terms is not None else _default_scalar_terms(
        state_dim, config.hamiltonian_library_degree
    )
    c_terms = tuple(casimir_terms) if casimir_terms is not None else _default_scalar_terms(
        state_dim, config.casimir_library_degree
    )

    lane_results: List[PoissonLaneResult] = []
    all_candidates: List[PoissonCandidate] = []
    pair_count = state_dim * (state_dim - 1) // 2

    for lane in config.lanes:
        degree, include_constant = _LANE_SPEC[lane]
        scalar_basis = PolynomialScalarBasis(
            state_dim,
            max_degree=degree,
            include_constant=include_constant,
        )
        bivector_basis = BivectorBasis(state_dim, scalar_basis)
        train_matrices = [
            bivector_basis.determining_matrix(Z, F, DF)
            for Z, F, DF in zip(train_points, train_values, train_jacobians)
        ]
        validation_matrices = [
            bivector_basis.determining_matrix(Z, F, DF)
            for Z, F, DF in zip(
                validation_points, validation_values, validation_jacobians
            )
        ]
        if config.normalize_dataset_blocks and len(train_matrices) > 1:
            train_matrices = [_normalise_determining_block(D) for D in train_matrices]
            validation_matrices = [
                _normalise_determining_block(D) for D in validation_matrices
            ]
        train_matrix = torch.cat(train_matrices, dim=0)
        validation_matrix = torch.cat(validation_matrices, dim=0)
        nullspace_policy = config.nullspace
        if nullspace_policy.bootstrap > 0:
            nullspace_policy = replace(
                nullspace_policy,
                bootstrap_block_size=pair_count,
            )
        nullspace = stable_nullspace(
            train_matrix,
            validation_matrix,
            config=nullspace_policy,
        )

        lane_candidates: List[PoissonCandidate] = []
        for coefficients in _representatives(nullspace, config):
            validation_evaluations = [
                bivector_basis.assemble(coefficients, Z) for Z in validation_points
            ]
            skew_certificate = skew_residual(validation_evaluations[0].tensor)
            invariance_certificates = tuple(
                invariance_residual(
                    represented.tensor,
                    represented.derivatives,
                    F,
                    DF,
                    relative_tolerance=config.invariance_relative_tolerance,
                    absolute_tolerance=config.invariance_absolute_tolerance,
                )
                for represented, F, DF in zip(
                    validation_evaluations,
                    validation_values,
                    validation_jacobians,
                )
            )
            combined_pi = torch.cat(
                [item.tensor for item in validation_evaluations], dim=0
            )
            combined_dpi = torch.cat(
                [item.derivatives for item in validation_evaluations], dim=0
            )
            jacobi_certificate = jacobi_residual(
                combined_pi,
                combined_dpi,
                relative_tolerance=config.jacobi_relative_tolerance,
                absolute_tolerance=config.jacobi_absolute_tolerance,
            )
            coefficient_table = coefficients.reshape(
                len(bivector_basis.pairs), scalar_basis.size
            )
            polynomial_certificate = polynomial_jacobi_certificate(
                coefficient_table,
                scalar_basis.exponents,
                tolerance=config.polynomial_jacobi_tolerance,
            )
            rank_certificate = rank_profile(
                combined_pi,
                relative_tolerance=config.rank_relative_tolerance,
                absolute_tolerance=config.rank_absolute_tolerance,
                minimum_stable_fraction=config.minimum_rank_stable_fraction,
            )

            poisson_callable = lambda Z, b=bivector_basis, c=coefficients: b.assemble(c, Z).tensor
            if len(fields) == 1:
                hamiltonian: HamiltonianFitResult | MultiHamiltonianFitResult = (
                    fit_hamiltonian_given_poisson(
                        poisson_callable,
                        train_values[0],
                        train_points[0],
                        h_terms,
                        config.hamiltonian,
                    )
                )
                h_fits = (hamiltonian,)
            else:
                hamiltonian = fit_hamiltonians_given_poisson(
                    poisson_callable,
                    train_values,
                    train_points,
                    h_terms,
                    mode=config.hamiltonian_mode,
                    config=config.hamiltonian,
                )
                h_fits = hamiltonian.fits
            hamiltonian_validation = tuple(
                _hamiltonian_validation_error(
                    None,
                    bivector_basis,
                    coefficients,
                    fit,
                    F,
                    Z,
                )
                for fit, F, Z in zip(
                    h_fits,
                    validation_values,
                    validation_points,
                )
            )
            h_validation_passed = all(
                value <= config.hamiltonian.relative_residual_tolerance
                for value in hamiltonian_validation
            )

            casimir_result = None
            if config.discover_casimirs and rank_certificate.generic_rank < state_dim:
                casimir_Z = torch.cat(validation_points, dim=0)
                casimir_F = torch.cat(validation_values, dim=0)
                casimir_result = discover_casimirs(
                    poisson_callable,
                    casimir_Z,
                    c_terms,
                    field_values=casimir_F,
                    nullspace_config=replace(config.nullspace, bootstrap=0),
                    coefficient_threshold=config.coefficient_tolerance,
                    validation_fraction=0.25,
                    random_seed=config.random_seed + 101,
                    max_representatives=config.max_representatives,
                    sparse_rotation_steps=config.sparse_rotation_steps,
                )

            failures: List[str] = []
            nullspace_angles = [
                angle
                for angle in (
                    nullspace.heldout_principal_angle,
                    *nullspace.bootstrap_principal_angles,
                )
                if angle is not None
            ]
            if config.require_nullspace_stability and (
                not nullspace_angles
                or max(nullspace_angles) > config.nullspace_max_principal_angle
            ):
                failures.append("nullspace_unstable")
            if not skew_certificate.passed:
                failures.append("skew")
            if not all(item.passed for item in invariance_certificates):
                failures.append("invariance")
            if not jacobi_certificate.passed or not polynomial_certificate.passed:
                failures.append("jacobi")
            if not rank_certificate.accepted:
                failures.append("rank_unstable")
            if config.require_nonzero_rank and rank_certificate.generic_rank == 0:
                failures.append("zero_rank")
            if config.require_hamiltonian and (
                not hamiltonian.accepted or not h_validation_passed
            ):
                failures.append("hamiltonian")
            if (
                config.discover_casimirs
                and config.require_complete_casimirs
                and rank_certificate.generic_rank < state_dim
                and (casimir_result is None or not casimir_result.complete)
            ):
                failures.append("casimir_incomplete")

            pi_complexity = int(
                (coefficients.abs() >= config.coefficient_tolerance).sum().item()
            )
            h_complexity = (
                hamiltonian.complexity
                if isinstance(hamiltonian, HamiltonianFitResult)
                else int(hamiltonian.support.sum().item())
            )
            complexity = pi_complexity + h_complexity
            if casimir_result is not None:
                complexity += sum(
                    candidate.complexity
                    for candidate in casimir_result.candidates
                )
            max_invariance = max(item.relative for item in invariance_certificates)
            max_hamiltonian = max(hamiltonian_validation, default=0.0)
            casimir_deficit = (
                0
                if casimir_result is None
                else max(
                    0,
                    casimir_result.expected_corank
                    - casimir_result.discovered_corank,
                )
            )
            score = (
                max_invariance
                + jacobi_certificate.relative
                + max_hamiltonian
                + config.casimir_incompleteness_weight * casimir_deficit
                + config.complexity_weight * complexity
            )
            candidate = PoissonCandidate(
                lane=lane,
                degree=degree,
                include_constant=include_constant,
                coefficients=coefficients,
                exponents=scalar_basis.exponents,
                basis=bivector_basis,
                nullspace=nullspace,
                skew=skew_certificate,
                invariance=invariance_certificates,
                jacobi=jacobi_certificate,
                polynomial_jacobi=polynomial_certificate,
                rank=rank_certificate,
                hamiltonian=hamiltonian,
                hamiltonian_validation_relative=hamiltonian_validation,
                casimirs=casimir_result,
                complexity=complexity,
                score=score,
                accepted=not failures,
                failure_reasons=tuple(failures),
            )
            lane_candidates.append(candidate)

        lane_candidates.sort(key=lambda candidate: candidate.score)
        lane_result = PoissonLaneResult(
            lane=lane,
            degree=degree,
            include_constant=include_constant,
            nullspace=nullspace,
            candidates=tuple(lane_candidates),
        )
        lane_results.append(lane_result)
        all_candidates.extend(lane_candidates)
        if config.stop_at_first_accepted_lane and any(
            candidate.accepted for candidate in lane_candidates
        ):
            break

    all_candidates.sort(key=lambda candidate: (not candidate.accepted, candidate.score))
    pareto = _pareto_front(all_candidates)
    best = next((candidate for candidate in all_candidates if candidate.accepted), None)
    return PoissonSearchResult(
        state_dim=state_dim,
        dataset_count=len(fields),
        lanes=tuple(lane_results),
        candidates=tuple(all_candidates),
        pareto_candidates=pareto,
        best=best,
        accepted=best is not None,
    )


__all__ = [
    "PoissonCandidate",
    "PoissonLaneResult",
    "PoissonSearchConfig",
    "PoissonSearchResult",
    "discover_poisson_structure",
    "discover_poisson_structure_multi",
]
