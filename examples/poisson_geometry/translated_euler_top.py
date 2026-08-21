# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

r"""Discover the affine Lie--Poisson bracket of a translated Euler top.

Writing ``M = z-c`` hides the usual rigid-body bracket behind a translation:

    {z_i, z_j} = -epsilon_ijk (z_k-c_k).

The homogeneous-linear lane can no longer represent the geometry.  Shared
experiments with different inertia tensors therefore provide a crisp ladder
test: constant and linear lanes have trivial determining nullspaces, while
the affine lane recovers the common bracket, quadratic Hamiltonians, the
translated quadratic Casimir, and the rank-zero singular point ``z=c``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nestynet_sr.sr_de import (
    AutoPoissonConfig,
    AutoPoissonReport,
    CasimirCandidate,
    MultiHamiltonianFitResult,
    PoissonCandidate,
    PoissonSearchResult,
    VectorField,
    auto_discover_poisson_structure_multi,
)


DTYPE = torch.float64


@dataclass(frozen=True)
class TranslatedEulerTopShowcase:
    """Discovery result and certificates for affine-bracket recovery."""

    search: PoissonSearchResult
    automatic: AutoPoissonReport
    best: PoissonCandidate
    shift: torch.Tensor
    lower_lane_nullities: tuple[tuple[str, int], ...]
    tensor_alignment: float
    quadratic_hamiltonian_terms: int
    translated_casimir: CasimirCandidate
    casimir_alignment: float
    singular_point_tensor_norm: float

    def summary(self) -> dict[str, object]:
        return {
            "accepted": self.search.accepted,
            "lanes": tuple(lane.lane for lane in self.search.lanes),
            "lower_lane_nullities": self.lower_lane_nullities,
            "accepted_lane": self.best.lane,
            "affine_nullity": self.best.nullspace.nullity,
            "tensor_alignment": self.tensor_alignment,
            "generic_rank": self.best.rank.generic_rank,
            "singular_point_tensor_norm": self.singular_point_tensor_norm,
            "polynomial_jacobi_max_abs": self.best.polynomial_jacobi.max_abs,
            "hamiltonian_validation_relative": (
                self.best.hamiltonian_validation_relative
            ),
            "quadratic_hamiltonian_terms": self.quadratic_hamiltonian_terms,
            "casimir_alignment": self.casimir_alignment,
            "casimir_poisson_rms": self.translated_casimir.poisson_residual_rms,
        }


def _euler_field(shift: torch.Tensor, inverse_inertia: torch.Tensor) -> VectorField:
    def value(z: torch.Tensor) -> torch.Tensor:
        momentum = z - shift
        return torch.linalg.cross(momentum, momentum * inverse_inertia)

    return VectorField(value, state_dim=3)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.abs(torch.dot(left, right) / denominator).item())


def run_showcase(
    *, sample_count: int = 400, seed: int = 91
) -> TranslatedEulerTopShowcase:
    """Run the constant-to-affine lane ladder on three shared experiments."""

    if sample_count < 64:
        raise ValueError("sample_count must be at least 64")
    shift = torch.tensor([0.35, -0.45, 0.25], dtype=DTYPE)
    inverse_inertias = (
        torch.tensor([1.0, 0.55, 0.25], dtype=DTYPE),
        torch.tensor([0.8, 0.45, 0.18], dtype=DTYPE),
        torch.tensor([1.3, 0.7, 0.32], dtype=DTYPE),
    )
    generator = torch.Generator().manual_seed(seed)
    points = [
        shift
        + 1.8
        * torch.rand((sample_count, 3), generator=generator, dtype=DTYPE)
        - 0.9
        for _ in inverse_inertias
    ]
    automatic = auto_discover_poisson_structure_multi(
        [_euler_field(shift, inertia) for inertia in inverse_inertias],
        points,
        AutoPoissonConfig(random_seed=seed + 10),
    )
    search = automatic.search
    if search is None or search.best is None:
        raise RuntimeError("affine Poisson discovery returned no accepted candidate")
    best = search.best

    expected_table = torch.zeros_like(best.coefficient_table)
    exponent_index = {exponent: i for i, exponent in enumerate(best.exponents)}
    # {z0,z1}=-(z2-c2), {z0,z2}=z1-c1, {z1,z2}=-(z0-c0)
    expected_table[0, exponent_index[(0, 0, 0)]] = shift[2]
    expected_table[0, exponent_index[(0, 0, 1)]] = -1.0
    expected_table[1, exponent_index[(0, 0, 0)]] = -shift[1]
    expected_table[1, exponent_index[(0, 1, 0)]] = 1.0
    expected_table[2, exponent_index[(0, 0, 0)]] = shift[0]
    expected_table[2, exponent_index[(1, 0, 0)]] = -1.0
    tensor_alignment = _cosine(best.coefficients, expected_table.reshape(-1))

    if not isinstance(best.hamiltonian, MultiHamiltonianFitResult):
        raise RuntimeError("multi-system search did not return Hamiltonian heads")
    quadratic_indices = [
        index
        for index, exponent in enumerate(best.hamiltonian.terms)
        if sum(exponent) == 2
    ]
    quadratic_hamiltonian_terms = int(
        best.hamiltonian.support[quadratic_indices].sum().item()
    )

    if best.casimirs is None or not best.casimirs.candidates:
        raise RuntimeError("affine bracket did not produce a Casimir candidate")
    expected_casimir = torch.zeros(len(best.casimirs.terms), dtype=DTYPE)
    for axis in range(3):
        linear = tuple(1 if i == axis else 0 for i in range(3))
        square = tuple(2 if i == axis else 0 for i in range(3))
        expected_casimir[best.casimirs.terms.index(linear)] = -shift[axis]
        expected_casimir[best.casimirs.terms.index(square)] = 0.5
    translated_casimir = max(
        best.casimirs.candidates,
        key=lambda candidate: _cosine(candidate.coeffs, expected_casimir),
    )
    casimir_alignment = _cosine(translated_casimir.coeffs, expected_casimir)
    singular_point_tensor_norm = float(
        torch.linalg.matrix_norm(best.evaluate(shift.unsqueeze(0)).tensor).item()
    )
    lower_lane_nullities = tuple(
        (lane.lane, lane.nullspace.nullity)
        for lane in search.lanes
        if lane.lane != "affine"
    )

    report = TranslatedEulerTopShowcase(
        search=search,
        automatic=automatic,
        best=best,
        shift=shift,
        lower_lane_nullities=lower_lane_nullities,
        tensor_alignment=tensor_alignment,
        quadratic_hamiltonian_terms=quadratic_hamiltonian_terms,
        translated_casimir=translated_casimir,
        casimir_alignment=casimir_alignment,
        singular_point_tensor_norm=singular_point_tensor_norm,
    )
    _require_certified(report)
    return report


def _require_certified(report: TranslatedEulerTopShowcase) -> None:
    failures: list[str] = []
    if report.lower_lane_nullities != (("constant", 0), ("linear", 0)):
        failures.append("constant or homogeneous-linear lane was not rejected")
    if report.best.lane != "affine" or report.best.nullspace.nullity != 1:
        failures.append("expected a unique affine invariant-bivector direction")
    if report.tensor_alignment < 1.0 - 1.0e-7:
        failures.append("recovered tensor is not the translated Euler bracket")
    if report.best.rank.generic_rank != 2 or not report.best.rank.accepted:
        failures.append("generic Poisson rank is not stably two")
    if report.singular_point_tensor_norm > 1.0e-8:
        failures.append("tensor did not lose rank at the translated origin")
    if not report.best.polynomial_jacobi.passed:
        failures.append("exact affine Jacobi/cocycle certificate failed")
    if max(report.best.hamiltonian_validation_relative) > 1.0e-7:
        failures.append("held-out quadratic Hamiltonian reconstruction failed")
    # A degenerate bracket identifies H only modulo its quadratic Casimir.
    # The gauge-aware fitter deliberately chooses a two-term representative
    # instead of requiring all three diagonal quadratic terms.
    if report.quadratic_hamiltonian_terms < 2:
        failures.append("Hamiltonian gauge representative lost quadratic structure")
    if report.casimir_alignment < 1.0 - 1.0e-7:
        failures.append("translated quadratic Casimir was not recovered")
    if report.translated_casimir.poisson_residual_rms > 1.0e-8:
        failures.append("translated Casimir failed the Poisson-null certificate")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    result = run_showcase()
    print("Translated Euler top")
    for key, value in result.summary().items():
        print(f"  {key}: {value}")
