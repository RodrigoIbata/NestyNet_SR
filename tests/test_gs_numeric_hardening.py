import warnings

import numpy as np
import pytest

from nestynet_sr.sr_core.separability_math import (
    _stable_weighted_orthogonal_mean,
)
from nestynet_sr.sr_gs.pairwise_composition import (
    _finite_rowwise_linear_combination,
    _joint_ray_residual_value,
)


def test_joint_ray_residual_is_scale_invariant_without_runtime_warnings():
    gradient = np.asarray(
        [
            [1.0, 2.0, -0.5],
            [-0.7, 0.4, 1.2],
            [0.3, -1.1, 0.8],
            [1.4, -0.2, 0.6],
        ],
        dtype=float,
    )
    ray = (1, -1, 2)
    expected = _joint_ray_residual_value(gradient, ray)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = _joint_ray_residual_value(gradient * 1.0e300, ray)

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=1.0e-13, abs=1.0e-15)


def test_joint_ray_residual_rejects_nonfinite_input_without_warning():
    gradient = np.eye(3, dtype=float)
    gradient[0, 0] = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        residual = _joint_ray_residual_value(gradient, (1, 1, 1))

    assert residual == float("inf")


def test_rowwise_linear_combination_handles_large_finite_values_and_rejects_overflow():
    ordinary = np.asarray(
        [
            [0.3, -0.4, 1.2],
            [2.0, 0.5, -0.7],
        ],
        dtype=float,
    )
    ordinary_coefficients = (2.0, -1.0, 3.0)
    representable = np.asarray(
        [
            [8.0e307, 8.0e307],
            [1.0e308, -1.0e308],
        ],
        dtype=float,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ordinary_combined = _finite_rowwise_linear_combination(
            ordinary,
            ordinary_coefficients,
        )
        combined = _finite_rowwise_linear_combination(representable, (1.0, 1.0))
        rejected = _finite_rowwise_linear_combination(
            np.asarray([[1.0e308, 1.0e308]], dtype=float),
            (1.0, 1.0),
        )

    assert ordinary_combined is not None
    np.testing.assert_array_equal(
        ordinary_combined,
        ordinary @ np.asarray(ordinary_coefficients),
    )
    assert combined is not None
    assert np.all(np.isfinite(combined))
    assert combined[0] == pytest.approx(1.6e308)
    assert combined[1] == pytest.approx(0.0, abs=0.0)
    assert rejected is None


def test_weighted_orthogonal_mean_is_scale_invariant_without_runtime_warnings():
    values = np.asarray(
        [
            [0.8, -0.2, 0.5],
            [-0.4, 0.7, 1.1],
            [0.3, 0.9, -0.6],
        ],
        dtype=float,
    )
    direction = np.asarray([1.0, -2.0, 1.0], dtype=float)
    weights = np.asarray([1.0, 2.0, 4.0], dtype=float)
    denominator = float(np.dot(direction, direction) + 1.0e-12)
    coefficients = (values @ direction) / denominator
    expected = np.average(
        values - coefficients[:, None] * direction[None, :],
        axis=0,
        weights=weights,
    )
    ordinary = _stable_weighted_orthogonal_mean(values, direction, weights)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = _stable_weighted_orthogonal_mean(
            values * 1.0e300,
            direction,
            weights * 1.0e300,
        )

    assert ordinary is not None
    assert actual is not None
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(ordinary, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        actual / 1.0e300,
        expected,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
