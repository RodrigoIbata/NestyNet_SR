# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

r"""Recover the conformal algebra of inverse-square mechanics.

The equation

    u_xx = g / u**3

has the point-symmetry triple

    P = d_x,
    D = x d_x + (u/2) d_u,
    K = x**2 d_x + x*u d_u.

``K`` is genuinely quadratic in point coordinates.  This example discovers
the complete three-dimensional determining nullspace, identifies ``K``, fits
its functional relative-invariance multiplier, and compiles the point
invariant ``u/x`` and a rectifying orbit coordinate proportional to ``1/x``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import Mul, Pow, Var
from nestynet_sr.sr_gs import (
    InvariantCompilationResult,
    InvariantCompilerConfig,
    JetSpaceSpec,
    OrbitCoordinateResult,
    PolynomialDESymmetryCandidate,
    PolynomialDESymmetryConfig,
    PolynomialDESymmetryResult,
    compile_orbit_coordinate,
    compile_point_invariants,
    project_generator_direction,
    recover_polynomial_de_symmetries,
)


@dataclass(frozen=True)
class ConformalInverseSquareShowcase:
    """Discovery objects and compact certificates for the showcase."""

    symmetry: PolynomialDESymmetryResult
    special_conformal: PolynomialDESymmetryCandidate
    special_conformal_projection_residual: float
    special_conformal_alignment: float
    multiplier_x_coefficient: float
    multiplier_x_expected: float
    invariant: InvariantCompilationResult
    orbit_coordinate: OrbitCoordinateResult

    def summary(self) -> dict[str, float | int | str | bool]:
        return {
            "status": self.symmetry.status,
            "certified_nullity": self.symmetry.certified_nullity,
            "special_conformal_projection_residual": (
                self.special_conformal_projection_residual
            ),
            "special_conformal_alignment": self.special_conformal_alignment,
            "multiplier_x_coefficient": self.multiplier_x_coefficient,
            "multiplier_x_expected": self.multiplier_x_expected,
            "invariant_recovered": bool(self.invariant.invariants),
            "invariant_validation_relative": (
                self.invariant.invariants[0].validation_action_relative
                if self.invariant.invariants
                else math.inf
            ),
            "orbit_coordinate_recovered": self.orbit_coordinate.accepted,
            "orbit_validation_relative": (
                self.orbit_coordinate.validation_residual_relative
            ),
            "bracket_closure_residual": self.symmetry.bracket_closure_residual,
        }


def _direction(result: PolynomialDESymmetryResult, terms: dict[str, float]) -> np.ndarray:
    vector = np.zeros(len(result.coefficient_labels), dtype=float)
    for label, coefficient in terms.items():
        vector[result.coefficient_labels.index(label)] = float(coefficient)
    return vector


def run_showcase(
    *,
    coupling: float = 1.7,
    sample_count: int = 600,
    seed: int = 17,
) -> ConformalInverseSquareShowcase:
    """Run discovery and return its independently inspectable certificates."""

    if coupling == 0.0:
        raise ValueError("coupling must be nonzero for inverse-square mechanics")
    if sample_count < 64:
        raise ValueError("sample_count must be at least 64")

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 2.0, sample_count)
    u = rng.uniform(0.7, 2.2, sample_count)
    u_x = rng.uniform(-1.5, 1.5, sample_count)
    on_shell = {
        "x": x,
        "u": u,
        "u_x": u_x,
        "u_xx": coupling / u**3,
    }
    off_shell = {
        "x": rng.uniform(0.45, 2.1, sample_count),
        "u": rng.uniform(0.65, 2.3, sample_count),
        "u_x": rng.uniform(-1.7, 1.7, sample_count),
        "u_xx": rng.uniform(-2.0, 2.0, sample_count),
    }
    jet = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=2)
    residual = f"u_xx-({coupling!r})/(u*u*u)"
    symmetry = recover_polynomial_de_symmetries(
        jet_space=jet,
        residual=residual,
        on_shell_samples=on_shell,
        off_shell_samples=off_shell,
        config=PolynomialDESymmetryConfig(
            generator_degree=2,
            multiplier_degree=2,
            heldout_fraction=0.25,
            bootstrap=4,
            random_seed=seed + 1,
            rank_rtol=1.0e-9,
            rank_atol=1.0e-11,
            on_shell_tol=1.0e-8,
            off_shell_tol=1.0e-8,
            bracket_closure_tol=1.0e-7,
            max_candidates=32,
        ),
    )

    expected_k = _direction(symmetry, {"xi:x^2": 1.0, "eta:x*u": 1.0})
    expected_k_unit = expected_k / np.linalg.norm(expected_k)
    special_conformal = max(
        symmetry.candidates,
        key=lambda row: abs(np.dot(np.asarray(row.coefficients), expected_k_unit)),
    )
    _projection, projection_residual = project_generator_direction(symmetry, expected_k)
    alignment = float(
        np.dot(np.asarray(special_conformal.coefficients), expected_k_unit)
    )
    x_multiplier_index = symmetry.multiplier_monomials.index((1, 0, 0, 0))
    multiplier_x = float(
        special_conformal.multiplier_coefficients[x_multiplier_index]
    )
    multiplier_x_expected = -3.0 * alignment / math.sqrt(2.0)

    points = torch.as_tensor(np.column_stack((x, u)), dtype=torch.float64)
    split = 3 * sample_count // 4
    compiler_config = InvariantCompilerConfig(
        action_rtol=1.0e-8,
        orbit_rtol=1.0e-8,
        max_invariants=1,
    )
    invariant = compile_point_invariants(
        (special_conformal.generator,),
        points[:split],
        points[split:],
        candidate_asts=(Mul(Var(1), Pow(Var(0), -1)),),
        config=compiler_config,
    )
    orbit_coordinate = compile_orbit_coordinate(
        special_conformal.generator,
        points[:split],
        points[split:],
        candidate_asts=(Pow(Var(0), -1),),
        config=compiler_config,
    )

    report = ConformalInverseSquareShowcase(
        symmetry=symmetry,
        special_conformal=special_conformal,
        special_conformal_projection_residual=float(projection_residual),
        special_conformal_alignment=alignment,
        multiplier_x_coefficient=multiplier_x,
        multiplier_x_expected=multiplier_x_expected,
        invariant=invariant,
        orbit_coordinate=orbit_coordinate,
    )
    _require_certified(report)
    return report


def _require_certified(report: ConformalInverseSquareShowcase) -> None:
    """Fail loudly when the executable example no longer demonstrates its claim."""

    failures: list[str] = []
    if report.symmetry.status != "recovered" or report.symmetry.certified_nullity != 3:
        failures.append("expected the three-dimensional conformal symmetry algebra")
    if report.special_conformal_projection_residual > 1.0e-8:
        failures.append("quadratic special-conformal direction was not recovered")
    if abs(abs(report.special_conformal_alignment) - 1.0) > 1.0e-7:
        failures.append("special-conformal representative is not aligned with K")
    if abs(report.multiplier_x_coefficient - report.multiplier_x_expected) > 1.0e-7:
        failures.append("functional relative-invariance multiplier is incorrect")
    if not report.invariant.invariants:
        failures.append("u/x invariant was not compiled")
    if not report.orbit_coordinate.accepted:
        failures.append("1/x orbit coordinate was not compiled")
    if report.symmetry.bracket_closure_residual > 1.0e-7:
        failures.append("recovered generators failed bracket closure")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    result = run_showcase()
    print("Conformal inverse-square mechanics")
    for key, value in result.summary().items():
        print(f"  {key}: {value}")
