# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Certified structure constants and Casimirs of recovered symmetry algebras.

The structure constants use the vector-field convention already employed by
the GS closure certificates,

``[V_a,V_b] = D V_b V_a - D V_a V_b = c[a,b,c] V_c``.

For the canonical momentum-map convention in :mod:`noether_reduction`, the
charge brackets are the anti-homomorphic form
``{J_a,J_b} = -c[a,b,c] J_c + kappa[a,b]``.  The sign is explicit in every
charge-bracket report; Casimir discovery itself is insensitive to this global
sign because it leaves the coadjoint distribution unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from nestynet_sr.sr_de.poisson_basis import PolynomialScalarBasis
from nestynet_sr.sr_de.poisson_core import StableNullspaceConfig
from nestynet_sr.sr_de.poisson_invariants import (
    CasimirCandidate,
    CasimirDiscoveryResult,
    discover_casimirs,
)
from nestynet_sr.sr_gs.algebra_certificates import affine_graph_bracket_coeffs


_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class StructureConstantsCertificate:
    """Projection and Jacobi certificate for a finite-dimensional algebra."""

    generator_names: tuple[str, ...]
    structure_constants: np.ndarray
    pair_relative_residuals: np.ndarray
    max_closure_residual: float
    antisymmetry_residual: float
    jacobi_residual: float
    accepted: bool
    convention: str = "[V_a,V_b]=c_ab^c V_c"

    @property
    def dimension(self) -> int:
        return int(self.structure_constants.shape[0])

    def to_report(self) -> dict[str, Any]:
        return {
            "generator_names": list(self.generator_names),
            "dimension": self.dimension,
            "structure_constants": self.structure_constants.tolist(),
            "max_closure_residual": float(self.max_closure_residual),
            "antisymmetry_residual": float(self.antisymmetry_residual),
            "jacobi_residual": float(self.jacobi_residual),
            "accepted": bool(self.accepted),
            "convention": self.convention,
        }


@dataclass(frozen=True)
class ChargeBracketCertificate:
    """Sampled equivariance/cocycle check for recovered local charges."""

    brackets: np.ndarray
    reconstructed_brackets: np.ndarray
    central_cocycle: np.ndarray
    relative_residual: float
    cocycle_residual: float
    accepted: bool
    convention: str
    global_equivariance_proven: bool = False

    def to_report(self) -> dict[str, Any]:
        return {
            "relative_residual": float(self.relative_residual),
            "cocycle_residual": float(self.cocycle_residual),
            "central_cocycle": self.central_cocycle.tolist(),
            "accepted": bool(self.accepted),
            "convention": self.convention,
            "global_equivariance_proven": bool(self.global_equivariance_proven),
        }


@dataclass(frozen=True)
class AlgebraCasimirResult:
    """Casimirs of the Lie--Poisson tensor induced on ``g*``."""

    structure: StructureConstantsCertificate
    casimirs: CasimirDiscoveryResult
    accepted: bool

    @property
    def expected_corank(self) -> int:
        return int(self.casimirs.expected_corank)

    @property
    def complete(self) -> bool:
        return bool(self.casimirs.complete)

    def to_report(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "complete": self.complete,
            "expected_corank": self.expected_corank,
            "structure": self.structure.to_report(),
            "casimirs": self.casimirs.to_report(),
        }


def _jacobi_residual(structure_constants: np.ndarray) -> float:
    c = np.asarray(structure_constants, dtype=np.float64)
    scale = max(float(np.linalg.norm(c) ** 2), _EPS)
    largest = 0.0
    for a in range(c.shape[0]):
        for b in range(c.shape[0]):
            for d in range(c.shape[0]):
                row = (
                    np.einsum("e,ef->f", c[a, b], c[:, d, :])
                    + np.einsum("e,ef->f", c[b, d], c[:, a, :])
                    + np.einsum("e,ef->f", c[d, a], c[:, b, :])
                )
                largest = max(largest, float(np.linalg.norm(row)) / scale)
    return largest


def _structure_certificate(
    basis: np.ndarray,
    brackets: Sequence[Sequence[np.ndarray]],
    names: Sequence[str],
    *,
    closure_tol: float,
    jacobi_tol: float,
) -> StructureConstantsCertificate:
    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("basis must have shape (coefficient_count, generator_count)")
    dimension = int(matrix.shape[1])
    if len(names) != dimension:
        raise ValueError("names must match the generator count")
    if np.linalg.matrix_rank(matrix) != dimension:
        raise ValueError("generator basis is linearly dependent")
    constants = np.zeros((dimension, dimension, dimension), dtype=np.float64)
    residuals = np.zeros((dimension, dimension), dtype=np.float64)
    for a in range(dimension):
        for b in range(dimension):
            bracket = np.asarray(brackets[a][b], dtype=np.float64).reshape(-1)
            coefficients, *_ = np.linalg.lstsq(matrix, bracket, rcond=None)
            reconstruction = matrix @ coefficients
            scale = max(float(np.linalg.norm(bracket)), _EPS)
            residuals[a, b] = float(np.linalg.norm(bracket - reconstruction) / scale)
            constants[a, b] = coefficients
    antisymmetry = float(
        np.linalg.norm(constants + constants.swapaxes(0, 1))
        / max(np.linalg.norm(constants), _EPS)
    )
    jacobi = _jacobi_residual(constants)
    closure = float(np.max(residuals))
    accepted = bool(
        closure <= float(closure_tol)
        and antisymmetry <= float(closure_tol)
        and jacobi <= float(jacobi_tol)
    )
    return StructureConstantsCertificate(
        generator_names=tuple(str(name) for name in names),
        structure_constants=constants,
        pair_relative_residuals=residuals,
        max_closure_residual=closure,
        antisymmetry_residual=antisymmetry,
        jacobi_residual=jacobi,
        accepted=accepted,
    )


def extract_phase_structure_constants(
    generators: Sequence[Any],
    *,
    closure_tol: float = 1.0e-10,
    jacobi_tol: float = 1.0e-10,
) -> StructureConstantsCertificate:
    """Extract structure constants from affine phase generators ``Mz+c``."""

    rows = tuple(generators)
    if not rows:
        raise ValueError("at least one phase generator is required")
    state_dim = int(np.asarray(rows[0].c).size)
    encoded = []
    for generator in rows:
        matrix = np.asarray(generator.M, dtype=np.float64)
        translation = np.asarray(generator.c, dtype=np.float64)
        if matrix.shape != (state_dim, state_dim) or translation.shape != (
            state_dim,
        ):
            raise ValueError("phase generators must share one affine state dimension")
        encoded.append(np.concatenate((matrix.reshape(-1), translation)))
    basis = np.stack(encoded, axis=1)
    brackets: list[list[np.ndarray]] = []
    for first in rows:
        bracket_row = []
        for second in rows:
            matrix = second.M @ first.M - first.M @ second.M
            translation = second.M @ first.c - first.M @ second.c
            bracket_row.append(np.concatenate((matrix.reshape(-1), translation)))
        brackets.append(bracket_row)
    return _structure_certificate(
        basis,
        brackets,
        [getattr(generator, "name", f"generator_{index}") for index, generator in enumerate(rows)],
        closure_tol=closure_tol,
        jacobi_tol=jacobi_tol,
    )


def extract_affine_structure_constants(
    algebra: Any,
    *,
    closure_tol: float = 1.0e-7,
    jacobi_tol: float = 1.0e-7,
) -> StructureConstantsCertificate:
    """Express every certified affine-GS bracket in the recovered basis."""

    basis = np.asarray(algebra.nullspace_basis, dtype=np.float64)
    dimension = int(basis.shape[1])
    brackets = [
        [
            affine_graph_bracket_coeffs(
                basis[:, first], basis[:, second], int(algebra.input_dim)
            )
            for second in range(dimension)
        ]
        for first in range(dimension)
    ]
    result = _structure_certificate(
        basis,
        brackets,
        [f"generator_{index}" for index in range(dimension)],
        closure_tol=closure_tol,
        jacobi_tol=jacobi_tol,
    )
    if not bool(getattr(algebra, "promotable", False)):
        return StructureConstantsCertificate(
            generator_names=result.generator_names,
            structure_constants=result.structure_constants,
            pair_relative_residuals=result.pair_relative_residuals,
            max_closure_residual=result.max_closure_residual,
            antisymmetry_residual=result.antisymmetry_residual,
            jacobi_residual=result.jacobi_residual,
            accepted=False,
            convention=result.convention,
        )
    return result


def _lie_poisson_tensor(
    points: torch.Tensor, structure_constants: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("abc,nc->nab", structure_constants, points)


def discover_algebra_casimirs(
    structure: StructureConstantsCertificate,
    *,
    max_degree: int = 2,
    sample_count: int = 512,
    random_seed: int = 0,
    mu_points: np.ndarray | torch.Tensor | None = None,
    max_representatives: int = 64,
) -> AlgebraCasimirResult:
    """Discover polynomial invariants of the coadjoint distribution on ``g*``."""

    if int(max_degree) < 1:
        raise ValueError("max_degree must be positive")
    dimension = structure.dimension
    if mu_points is None:
        generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
        points = torch.randn(
            int(sample_count), dimension, generator=generator, dtype=torch.float64
        )
    else:
        points = torch.as_tensor(mu_points, dtype=torch.float64)
        if points.ndim != 2 or points.shape[1] != dimension:
            raise ValueError("mu_points must have shape (N, algebra_dimension)")
    constants = torch.as_tensor(structure.structure_constants, dtype=points.dtype)

    def poisson(mu: torch.Tensor) -> torch.Tensor:
        return _lie_poisson_tensor(mu, constants.to(mu))

    terms = PolynomialScalarBasis(
        dimension, max_degree=int(max_degree), include_constant=True
    ).exponents
    casimirs = discover_casimirs(
        poisson,
        points,
        terms,
        validation_fraction=0.25,
        random_seed=int(random_seed) + 1,
        nullspace_config=StableNullspaceConfig(
            rank_rtol=1.0e-10,
            rank_atol=1.0e-12,
            bootstrap=2,
            random_seed=int(random_seed) + 2,
        ),
        max_representatives=max_representatives,
        sparse_rotation_steps=20,
        relative_residual_tolerance=1.0e-9,
        absolute_residual_tolerance=1.0e-11,
    )
    return AlgebraCasimirResult(
        structure=structure,
        casimirs=casimirs,
        accepted=bool(structure.accepted and casimirs.complete),
    )


def certify_charge_brackets(
    charge_values: np.ndarray,
    charge_gradients: np.ndarray,
    poisson_values: np.ndarray,
    structure: StructureConstantsCertificate,
    *,
    allow_central_cocycle: bool = True,
    residual_tol: float = 1.0e-8,
    cocycle_tol: float = 1.0e-8,
) -> ChargeBracketCertificate:
    """Verify ``{J_a,J_b}=-c_ab^c J_c+kappa_ab`` on held state samples."""

    charges = np.asarray(charge_values, dtype=np.float64)
    gradients = np.asarray(charge_gradients, dtype=np.float64)
    poisson = np.asarray(poisson_values, dtype=np.float64)
    if charges.ndim != 2:
        raise ValueError("charge_values must have shape (N,s)")
    sample_count, dimension = charges.shape
    state_dim = int(poisson.shape[-1]) if poisson.ndim == 3 else -1
    if dimension != structure.dimension:
        raise ValueError("charge count must equal the algebra dimension")
    if gradients.shape != (sample_count, dimension, state_dim):
        raise ValueError("charge_gradients must have shape (N,s,state_dim)")
    if poisson.shape != (sample_count, state_dim, state_dim):
        raise ValueError("poisson_values must have shape (N,state_dim,state_dim)")
    brackets = np.einsum("nai,nij,nbj->nab", gradients, poisson, gradients)
    constants = structure.structure_constants
    linear = -np.einsum("abc,nc->nab", constants, charges)
    discrepancy = brackets - linear
    if allow_central_cocycle:
        cocycle = np.mean(discrepancy, axis=0)
        cocycle = 0.5 * (cocycle - cocycle.T)
    else:
        cocycle = np.zeros((dimension, dimension), dtype=np.float64)
    reconstructed = linear + cocycle[None, :, :]
    residual = brackets - reconstructed
    scale = max(float(np.linalg.norm(brackets)), float(np.linalg.norm(reconstructed)), _EPS)
    relative = float(np.linalg.norm(residual) / scale)

    cocycle_defect = 0.0
    cocycle_scale = max(float(np.linalg.norm(constants) * np.linalg.norm(cocycle)), _EPS)
    for a in range(dimension):
        for b in range(dimension):
            for d in range(dimension):
                value = (
                    np.dot(constants[a, b], cocycle[:, d])
                    + np.dot(constants[b, d], cocycle[:, a])
                    + np.dot(constants[d, a], cocycle[:, b])
                )
                cocycle_defect = max(cocycle_defect, abs(float(value)) / cocycle_scale)
    accepted = bool(
        structure.accepted
        and relative <= float(residual_tol)
        and cocycle_defect <= float(cocycle_tol)
    )
    return ChargeBracketCertificate(
        brackets=brackets,
        reconstructed_brackets=reconstructed,
        central_cocycle=cocycle,
        relative_residual=relative,
        cocycle_residual=cocycle_defect,
        accepted=accepted,
        convention="{J_a,J_b}=-c_ab^c J_c+kappa_ab",
        global_equivariance_proven=False,
    )


def normalized_positive_quadratic_casimir(
    candidate: CasimirCandidate,
    terms: Sequence[Sequence[int]],
    *,
    dimension: int,
    tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Fix the scalar gauge of a positive quadratic Casimir to mean eigenvalue one."""

    coefficients = candidate.coeffs.detach().cpu().numpy().astype(np.float64)
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    for coefficient, exponent in zip(coefficients, terms):
        degree = int(sum(exponent))
        if abs(float(coefficient)) <= float(tolerance):
            continue
        if degree != 2:
            raise ValueError("candidate is not a homogeneous quadratic Casimir")
        active = [axis for axis, power in enumerate(exponent) if int(power)]
        if len(active) == 1 and int(exponent[active[0]]) == 2:
            matrix[active[0], active[0]] += float(coefficient)
        elif len(active) == 2 and all(int(exponent[axis]) == 1 for axis in active):
            i, j = active
            matrix[i, j] += 0.5 * float(coefficient)
            matrix[j, i] += 0.5 * float(coefficient)
        else:
            raise ValueError("unsupported quadratic exponent")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.max(eigenvalues) < 0.0:
        matrix = -matrix
        coefficients = -coefficients
        eigenvalues = -eigenvalues[::-1]
    if np.min(eigenvalues) <= float(tolerance) * max(np.max(eigenvalues), 1.0):
        raise ValueError("quadratic Casimir is not positive definite")
    mean_eigenvalue = float(np.mean(eigenvalues))
    if mean_eigenvalue <= 0.0:
        raise ValueError("quadratic Casimir has no positive normalization")
    return coefficients / mean_eigenvalue


__all__ = [
    "AlgebraCasimirResult",
    "ChargeBracketCertificate",
    "StructureConstantsCertificate",
    "certify_charge_brackets",
    "discover_algebra_casimirs",
    "extract_affine_structure_constants",
    "extract_phase_structure_constants",
    "normalized_positive_quadratic_casimir",
]
