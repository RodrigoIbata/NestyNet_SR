# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

r"""Distinguish Poisson Casimirs from symmetry-algebra Casimirs.

This compact example exercises both new Casimir lanes:

1. On ``so(3)*``, the Euler-top Poisson tensor is degenerate and the physical
   state-space Casimir ``M_1^2 + M_2^2 + M_3^2`` is recovered.  Hamiltonian
   fitting then reports the expected gauge ``H ~ H + Phi(C)`` and selects a
   sparse representative without changing the vector field.
2. Canonical six-dimensional phase space is full rank and correctly has no
   nonconstant Poisson Casimir.  Its recovered rotation algebra nevertheless
   has the quadratic algebra Casimir ``K(J) = J_1^2 + J_2^2 + J_3^2``.  Pulling
   it back through the rotation charges recovers ``|q x p|^2``.

Run from the repository root with

    python -m examples.poisson_geometry.casimir_taxonomy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nestynet_sr.sr_de import (
    CasimirDiscoveryResult,
    HamiltonianFitConfig,
    HamiltonianFitResult,
    PolynomialScalarBasis,
    discover_casimirs,
    fit_hamiltonian_given_poisson,
)
from nestynet_sr.sr_gs import (
    AlgebraCasimirResult,
    ChargeBracketCertificate,
    certify_charge_brackets,
    discover_algebra_casimirs,
    extract_phase_structure_constants,
    normalized_positive_quadratic_casimir,
)
from nestynet_sr.sr_gs.noether_reduction import (
    canonical_generators,
    momentum_map,
    momentum_map_gradient,
    symplectic_matrix,
)


DTYPE = torch.float64


@dataclass(frozen=True)
class CasimirTaxonomyShowcase:
    """Certified outputs for the state-space and algebra-Casimir lanes."""

    poisson_casimirs: CasimirDiscoveryResult
    hamiltonian: HamiltonianFitResult
    canonical_casimirs: CasimirDiscoveryResult
    algebra_casimir: AlgebraCasimirResult
    charge_brackets: ChargeBracketCertificate
    poisson_casimir_alignment: float
    algebra_pullback_relative_rms: float

    def summary(self) -> dict[str, object]:
        poisson_candidate = self.poisson_casimirs.candidates[0]
        algebra_candidate = self.algebra_casimir.casimirs.candidates[0]
        gauge = self.hamiltonian.gauge
        return {
            "poisson_casimir_complete": self.poisson_casimirs.complete,
            "poisson_casimir_expression": poisson_candidate.expression,
            "poisson_casimir_coordinate_kind": poisson_candidate.coordinate.kind,
            "poisson_casimir_alignment": self.poisson_casimir_alignment,
            "hamiltonian_gauge_nullity": None if gauge is None else gauge.nullity,
            "hamiltonian_gauge_mode": (
                None if gauge is None else gauge.representative_mode
            ),
            "hamiltonian_flow_equivalence_relative": (
                None if gauge is None else gauge.flow_equivalence_relative
            ),
            "canonical_poisson_casimir_status": self.canonical_casimirs.status,
            "algebra_casimir_complete": self.algebra_casimir.complete,
            "algebra_casimir_expression": algebra_candidate.expression,
            "charge_bracket_relative_residual": (
                self.charge_brackets.relative_residual
            ),
            "algebra_pullback_relative_rms": self.algebra_pullback_relative_rms,
            "global_momentum_map_proven": (
                self.charge_brackets.global_equivariance_proven
            ),
        }


def _so3_poisson(momentum: torch.Tensor) -> torch.Tensor:
    """The rigid-body Lie--Poisson tensor ``Pi v = M x v``."""

    tensor = momentum.new_zeros((momentum.shape[0], 3, 3))
    tensor[:, 0, 1] = -momentum[:, 2]
    tensor[:, 0, 2] = momentum[:, 1]
    tensor[:, 1, 2] = -momentum[:, 0]
    return tensor - tensor.transpose(1, 2)


def _canonical_poisson(phase_points: torch.Tensor) -> torch.Tensor:
    matrix = torch.as_tensor(symplectic_matrix(3), dtype=phase_points.dtype)
    return matrix.to(phase_points).expand(phase_points.shape[0], -1, -1)


def _monomial_values(
    points: np.ndarray, terms: tuple[tuple[int, ...], ...]
) -> np.ndarray:
    values = np.ones((points.shape[0], len(terms)), dtype=np.float64)
    for column, exponent in enumerate(terms):
        for axis, power in enumerate(exponent):
            if power:
                values[:, column] *= points[:, axis] ** int(power)
    return values


def _quadratic_alignment(
    coefficients: torch.Tensor, terms: tuple[tuple[int, ...], ...]
) -> float:
    expected = torch.zeros_like(coefficients)
    for axis in range(3):
        exponent = tuple(2 if index == axis else 0 for index in range(3))
        expected[terms.index(exponent)] = 1.0
    denominator = torch.linalg.vector_norm(coefficients) * torch.linalg.vector_norm(
        expected
    )
    return float(torch.abs(torch.dot(coefficients, expected) / denominator).item())


def run_showcase(
    *, sample_count: int = 384, seed: int = 607
) -> CasimirTaxonomyShowcase:
    """Recover both kinds of Casimir and certify their different roles."""

    if sample_count < 96:
        raise ValueError("sample_count must be at least 96")
    generator = torch.Generator().manual_seed(seed)

    momentum = torch.randn(sample_count, 3, generator=generator, dtype=DTYPE)
    euler_terms = PolynomialScalarBasis(
        3, max_degree=2, include_constant=True
    ).exponents
    poisson_casimirs = discover_casimirs(
        _so3_poisson,
        momentum,
        euler_terms,
        validation_fraction=0.25,
        random_seed=seed + 1,
        max_representatives=48,
    )

    inverse_inertia = momentum.new_tensor([1.0, 0.6, 0.25])
    field_values = torch.linalg.cross(momentum, momentum * inverse_inertia)
    hamiltonian = fit_hamiltonian_given_poisson(
        _so3_poisson,
        field_values,
        momentum,
        euler_terms,
        HamiltonianFitConfig(
            solver="least_squares",
            ridge=0.0,
            gauge_mode="sparsest",
            relative_residual_tolerance=1.0e-10,
        ),
    )

    phase_points = torch.randn(sample_count, 6, generator=generator, dtype=DTYPE)
    canonical_terms = PolynomialScalarBasis(
        6, max_degree=2, include_constant=True
    ).exponents
    canonical_casimirs = discover_casimirs(
        _canonical_poisson,
        phase_points,
        canonical_terms,
        validation_fraction=0.25,
        random_seed=seed + 2,
    )

    rotations = tuple(
        generator
        for generator in canonical_generators(3)
        if generator.family == "rotation"
    )
    structure = extract_phase_structure_constants(rotations)
    algebra_casimir = discover_algebra_casimirs(
        structure,
        max_degree=2,
        sample_count=sample_count,
        random_seed=seed + 3,
    )

    phase_numpy = phase_points.detach().cpu().numpy()
    charges = np.stack(
        [momentum_map(rotation, phase_numpy, n=3) for rotation in rotations],
        axis=1,
    )
    charge_gradients = np.stack(
        [
            momentum_map_gradient(rotation, phase_numpy, n=3)
            for rotation in rotations
        ],
        axis=1,
    )
    poisson_values = np.broadcast_to(
        symplectic_matrix(3), (sample_count, 6, 6)
    )
    charge_brackets = certify_charge_brackets(
        charges,
        charge_gradients,
        poisson_values,
        structure,
        allow_central_cocycle=True,
        residual_tol=1.0e-11,
    )

    if not algebra_casimir.casimirs.candidates:
        raise RuntimeError("rotation algebra produced no Casimir candidate")
    algebra_candidate = algebra_casimir.casimirs.candidates[0]
    normalized_coefficients = normalized_positive_quadratic_casimir(
        algebra_candidate,
        algebra_casimir.casimirs.terms,
        dimension=3,
    )
    recovered_l_squared = np.einsum(
        "nk,k->n",
        _monomial_values(charges, algebra_casimir.casimirs.terms),
        normalized_coefficients,
    )
    direct_l_squared = np.sum(
        np.cross(phase_numpy[:, :3], phase_numpy[:, 3:]) ** 2,
        axis=1,
    )
    algebra_pullback_relative_rms = float(
        np.linalg.norm(recovered_l_squared - direct_l_squared)
        / np.linalg.norm(direct_l_squared)
    )

    if not poisson_casimirs.candidates:
        raise RuntimeError("Euler tensor produced no Casimir candidate")
    poisson_casimir_alignment = _quadratic_alignment(
        poisson_casimirs.candidates[0].coeffs, poisson_casimirs.terms
    )
    report = CasimirTaxonomyShowcase(
        poisson_casimirs=poisson_casimirs,
        hamiltonian=hamiltonian,
        canonical_casimirs=canonical_casimirs,
        algebra_casimir=algebra_casimir,
        charge_brackets=charge_brackets,
        poisson_casimir_alignment=poisson_casimir_alignment,
        algebra_pullback_relative_rms=algebra_pullback_relative_rms,
    )
    _require_certified(report)
    return report


def _require_certified(report: CasimirTaxonomyShowcase) -> None:
    failures: list[str] = []
    if not report.poisson_casimirs.complete:
        failures.append("Euler Poisson Casimir was not complete")
    if report.poisson_casimir_alignment < 1.0 - 1.0e-8:
        failures.append("Euler Casimir is not proportional to |M|^2")
    if not report.hamiltonian.accepted or report.hamiltonian.gauge is None:
        failures.append("gauge-aware Euler Hamiltonian fit failed")
    elif report.hamiltonian.gauge.flow_equivalence_relative > 1.0e-10:
        failures.append("Hamiltonian gauge changed the reconstructed flow")
    if report.canonical_casimirs.status != "full_rank_no_nonconstant_casimir":
        failures.append("canonical phase space reported a Poisson Casimir")
    if not report.algebra_casimir.accepted:
        failures.append("rotation-algebra Casimir was not certified")
    if not report.charge_brackets.accepted:
        failures.append("rotation charge brackets were not certified")
    if report.algebra_pullback_relative_rms > 1.0e-10:
        failures.append("algebra Casimir did not pull back to |q x p|^2")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    result = run_showcase()
    print("Casimir taxonomy")
    for key, value in result.summary().items():
        print(f"  {key}: {value}")
