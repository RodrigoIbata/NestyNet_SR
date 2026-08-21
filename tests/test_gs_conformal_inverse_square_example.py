import math

import pytest

from examples.quadratic_symmetry.conformal_inverse_square import run_showcase


def test_conformal_inverse_square_showcase_recovers_quadratic_geometry():
    report = run_showcase(sample_count=420, seed=31)

    assert report.symmetry.status == "recovered"
    assert report.symmetry.certified_nullity == 3
    assert report.special_conformal_projection_residual < 1.0e-9
    assert abs(report.special_conformal_alignment) == pytest.approx(1.0, abs=1.0e-9)
    assert report.multiplier_x_coefficient == pytest.approx(
        report.multiplier_x_expected,
        abs=1.0e-8,
    )
    assert report.invariant.invariants[0].validation_action_relative < 1.0e-10
    assert report.orbit_coordinate.validation_residual_relative < 1.0e-10
    expected_orbit_coefficient = (
        -math.sqrt(2.0) / report.special_conformal_alignment
    )
    assert report.orbit_coordinate.coefficients[0] == pytest.approx(
        expected_orbit_coefficient,
        abs=1.0e-8,
    )
    assert report.symmetry.bracket_closure_residual < 1.0e-9
