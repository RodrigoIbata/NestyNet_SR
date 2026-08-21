# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

r"""Discover a quadratic log-canonical bracket for cyclic Lotka--Volterra.

The shared geometry is

    {x, y} = x*y,  {x, z} = -x*z,  {y, z} = y*z,

while each experiment has its own linear Hamiltonian.  The cubic monomial
``x*y*z`` is a Casimir.  Multiple Hamiltonian heads are important here: they
identify the shared tensor much more sharply than a single trajectory family.
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
class CyclicLotkaVolterraShowcase:
    """Discovery result plus target-independent geometric diagnostics."""

    search: PoissonSearchResult
    automatic: AutoPoissonReport
    best: PoissonCandidate
    tensor_alignment: float
    hamiltonian_alignments: tuple[float, ...]
    cubic_casimir: CasimirCandidate
    cubic_casimir_alignment: float

    def summary(self) -> dict[str, object]:
        return {
            "accepted": self.search.accepted,
            "lane": self.best.lane,
            "determining_nullity": self.best.nullspace.nullity,
            "tensor_alignment": self.tensor_alignment,
            "hamiltonian_alignments": self.hamiltonian_alignments,
            "generic_rank": self.best.rank.generic_rank,
            "rank_stable_fraction": self.best.rank.stable_fraction,
            "sampled_jacobi_relative": self.best.jacobi.relative,
            "polynomial_jacobi_max_abs": self.best.polynomial_jacobi.max_abs,
            "hamiltonian_validation_relative": (
                self.best.hamiltonian_validation_relative
            ),
            "cubic_casimir_alignment": self.cubic_casimir_alignment,
            "casimir_poisson_rms": self.cubic_casimir.poisson_residual_rms,
            "casimir_flow_rms": self.cubic_casimir.flow_residual_rms,
        }


def _log_canonical_tensor(z: torch.Tensor) -> torch.Tensor:
    interaction = z.new_tensor(
        [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]
    )
    return interaction.unsqueeze(0) * z[:, :, None] * z[:, None, :]


def _field(hamiltonian_gradient: torch.Tensor) -> VectorField:
    def value(z: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "nij,j->ni", _log_canonical_tensor(z), hamiltonian_gradient
        )

    return VectorField(value, state_dim=3)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.abs(torch.dot(left, right) / denominator).item())


def run_showcase(
    *, sample_count: int = 400, seed: int = 23
) -> CyclicLotkaVolterraShowcase:
    """Discover the shared bracket, three Hamiltonians, and cubic Casimir."""

    if sample_count < 64:
        raise ValueError("sample_count must be at least 64")
    heads = (
        torch.tensor([1.0, 1.0, 1.0], dtype=DTYPE),
        torch.tensor([1.4, 0.6, 1.8], dtype=DTYPE),
        torch.tensor([0.7, 1.5, 1.1], dtype=DTYPE),
    )
    generator = torch.Generator().manual_seed(seed)
    points = [
        0.5
        + 1.5
        * torch.rand((sample_count, 3), generator=generator, dtype=DTYPE)
        for _ in heads
    ]
    automatic = auto_discover_poisson_structure_multi(
        [_field(head) for head in heads],
        points,
        AutoPoissonConfig(random_seed=seed + 10),
    )
    search = automatic.search
    if search is None or search.best is None:
        raise RuntimeError("quadratic Poisson discovery returned no accepted candidate")
    best = search.best

    expected_table = torch.zeros_like(best.coefficient_table)
    exponent_index = {exponent: i for i, exponent in enumerate(best.exponents)}
    expected_table[0, exponent_index[(1, 1, 0)]] = 1.0
    expected_table[1, exponent_index[(1, 0, 1)]] = -1.0
    expected_table[2, exponent_index[(0, 1, 1)]] = 1.0
    tensor_alignment = _cosine(best.coefficients, expected_table.reshape(-1))

    if not isinstance(best.hamiltonian, MultiHamiltonianFitResult):
        raise RuntimeError("multi-system search did not return Hamiltonian heads")
    linear_indices = [
        best.hamiltonian.terms.index(exponent)
        for exponent in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ]
    hamiltonian_alignments = tuple(
        _cosine(coefficients[linear_indices], head)
        for coefficients, head in zip(best.hamiltonian.coeffs, heads)
    )

    if best.casimirs is None or not best.casimirs.candidates:
        raise RuntimeError("quadratic bracket did not produce a Casimir candidate")
    cubic_index = best.casimirs.terms.index((1, 1, 1))
    cubic_casimir = max(
        best.casimirs.candidates,
        key=lambda candidate: abs(float(candidate.coeffs[cubic_index].item())),
    )
    cubic_casimir_alignment = abs(float(cubic_casimir.coeffs[cubic_index].item()))

    report = CyclicLotkaVolterraShowcase(
        search=search,
        automatic=automatic,
        best=best,
        tensor_alignment=tensor_alignment,
        hamiltonian_alignments=hamiltonian_alignments,
        cubic_casimir=cubic_casimir,
        cubic_casimir_alignment=cubic_casimir_alignment,
    )
    _require_certified(report)
    return report


def _require_certified(report: CyclicLotkaVolterraShowcase) -> None:
    failures: list[str] = []
    if report.best.lane != "quadratic" or report.best.nullspace.nullity != 1:
        failures.append("expected a unique quadratic invariant-bivector direction")
    if report.tensor_alignment < 1.0 - 1.0e-7:
        failures.append("recovered tensor is not the log-canonical bracket")
    if min(report.hamiltonian_alignments) < 1.0 - 1.0e-7:
        failures.append("one or more linear Hamiltonians were not recovered")
    if report.best.rank.generic_rank != 2 or not report.best.rank.accepted:
        failures.append("generic Poisson rank is not stably two")
    if not report.best.polynomial_jacobi.passed:
        failures.append("exact polynomial Jacobi certificate failed")
    if max(report.best.hamiltonian_validation_relative) > 1.0e-7:
        failures.append("held-out Hamiltonian reconstruction failed")
    if report.cubic_casimir_alignment < 1.0 - 1.0e-7:
        failures.append("cubic xyz Casimir was not recovered")
    if report.cubic_casimir.poisson_residual_rms > 1.0e-8:
        failures.append("cubic Casimir failed the Poisson-null certificate")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    result = run_showcase()
    print("Cyclic three-species Lotka--Volterra")
    for key, value in result.summary().items():
        print(f"  {key}: {value}")
