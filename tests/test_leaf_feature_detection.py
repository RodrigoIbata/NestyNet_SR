# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Tests for leaf-level feature detection on compound variables."""

import torch
import math

from nestynet_sr.sr_search.features import (
    discover_trig_from_data,
    discover_compound_features_from_data,
    LeafFeatures,
    TrigScaleSpec,
)
from nestynet_sr.sr_search.wrapper_policy import compound_z_wrapper_policy


class MockSearchHP:
    """Mock search hyperparameters."""
    compound_try_trig_wrappers = True
    compound_try_rational_wrappers = True
    compound_try_square_wrappers = True
    compound_try_abs_wrappers = True
    compound_rational_only_if_ratio_like = True
    compound_wrapper_prefer_factor = 0.1
    compound_leaf_trig_strength_threshold = 5.0


def test_trig_detection_sine():
    """Detect trig structure in y = sin(z)."""
    N = 1000
    z = torch.linspace(0, 6 * math.pi, N)
    y = torch.sin(z)

    spec = discover_trig_from_data(z, y, strength_threshold=5.0)

    assert spec is not None, "Should detect trig in sin(z)"
    assert spec.axis == 0, "Compound variable should be axis 0"
    # omega should be close to 1.0
    assert abs(spec.omega - 1.0) < 0.2, f"Expected omega~1.0, got {spec.omega}"
    assert spec.strength > 50, f"Expected strong signal, got {spec.strength}"


def test_trig_detection_high_frequency():
    """Detect trig structure in y = sin(5*z)."""
    N = 1000
    z = torch.linspace(0, 4 * math.pi, N)
    y = torch.sin(5 * z)

    spec = discover_trig_from_data(z, y, strength_threshold=5.0, max_omega=20.0)

    assert spec is not None, "Should detect trig in sin(5z)"
    # omega should be close to 5.0
    assert abs(spec.omega - 5.0) < 1.0, f"Expected omega~5.0, got {spec.omega}"


def test_compound_features_from_data():
    """Test the compound features wrapper function."""
    N = 1000
    z = torch.linspace(0, 4 * math.pi, N)
    y = torch.cos(2 * z)

    features = discover_compound_features_from_data(z, y, trig_strength_threshold=5.0)

    assert isinstance(features, LeafFeatures)
    assert 0 in features.trig_by_axis, "Should detect trig on axis 0"
    spec = features.trig_by_axis[0]
    assert abs(spec.omega - 2.0) < 0.5, f"Expected omega~2.0, got {spec.omega}"


def test_wrapper_policy_uses_oracle_trig_specs():
    """Verify wrapper policy uses oracle trig specs (axis 0) for leaf-level trig."""
    hp = MockSearchHP()

    # Without oracle specs, no trig
    policy1 = compound_z_wrapper_policy(
        kind="monomial",
        pattern=(1, 1),
        meta={},
        search_hp=hp,
        trig_spec=None,
        atom_var_idxs=[0, 1],
        oracle_trig_specs=None,
    )
    assert not policy1.trig, "Should not have trig without oracle specs"

    # With oracle trig spec on axis 0 (compound variable z)
    oracle_specs = [
        TrigScaleSpec(
            axis=0, omega=3.14, trig_fn="sin", k_hat=1.0,
            rel_std=0.01, n_points=1000,
        )
    ]
    policy2 = compound_z_wrapper_policy(
        kind="monomial",
        pattern=(1, 1),
        meta={},
        search_hp=hp,
        trig_spec=None,  # No axis-level trig
        atom_var_idxs=[0, 1],
        oracle_trig_specs=oracle_specs,
    )
    assert policy2.trig, "Should have trig with oracle axis-0 spec"
    assert abs(policy2.trig_omega - 3.14) < 0.1, f"Should use oracle omega, got {policy2.trig_omega}"


def test_axis_level_takes_precedence():
    """Verify axis-level trig_spec takes precedence over oracle specs."""
    hp = MockSearchHP()

    # Axis-level spec
    class MockTrigSpec:
        axis = 0
        omega = 2.0
        strength = 50.0

    # Oracle specs with different omega on axis 0
    oracle_specs = [
        TrigScaleSpec(
            axis=0, omega=5.0, trig_fn="cos", k_hat=1.0,
            rel_std=0.01, n_points=1000,
        )
    ]

    policy = compound_z_wrapper_policy(
        kind="monomial",
        pattern=(1, 0),  # Only axis 0 participates
        meta={},
        search_hp=hp,
        trig_spec=MockTrigSpec(),
        atom_var_idxs=[0, 1],
        oracle_trig_specs=oracle_specs,
    )

    assert policy.trig, "Should have trig"
    # Should use axis-level omega since it matches first
    assert abs(policy.trig_omega - 2.0) < 0.1, f"Should use axis omega, got {policy.trig_omega}"


def test_product_compound_trig():
    """Test trig detection on f(x,y) = sin(x*y) pattern."""
    N = 2000
    # Generate data for x, y in [-2, 2]
    torch.manual_seed(42)
    x = torch.rand(N, 2) * 4 - 2
    z = x[:, 0] * x[:, 1]  # z = x*y
    y = torch.sin(z)

    features = discover_compound_features_from_data(z, y, trig_strength_threshold=5.0)

    # Should detect trig on z = x*y
    assert 0 in features.trig_by_axis, "Should detect trig on compound z=x*y"
    spec = features.trig_by_axis[0]
    # omega should be close to 1.0 since y = sin(1*z)
    assert abs(spec.omega - 1.0) < 0.5, f"Expected omega~1.0 for sin(z), got {spec.omega}"


if __name__ == "__main__":
    print("Running test_trig_detection_sine...")
    test_trig_detection_sine()
    print("PASSED\n")

    print("Running test_trig_detection_high_frequency...")
    test_trig_detection_high_frequency()
    print("PASSED\n")

    print("Running test_compound_features_from_data...")
    test_compound_features_from_data()
    print("PASSED\n")

    print("Running test_wrapper_policy_uses_oracle_trig_specs...")
    test_wrapper_policy_uses_oracle_trig_specs()
    print("PASSED\n")

    print("Running test_axis_level_takes_precedence...")
    test_axis_level_takes_precedence()
    print("PASSED\n")

    print("Running test_product_compound_trig...")
    test_product_compound_trig()
    print("PASSED\n")

    print("\n=== All tests passed! ===")
