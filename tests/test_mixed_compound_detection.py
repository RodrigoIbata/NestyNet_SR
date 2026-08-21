# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""
Tests for mixed compound variable detection (monomial * trig product).

Tests cases like:
  - f = x0 * x1 * cos(x2)
  - f = x0 * cos(x1) * sin(x2)
  - f = (x0 / x1) * cos(x2)
"""

import numpy as np


def test_mixed_compound_basic():
    """Test detection of f = x0 * x1 * cos(omega * x2)."""
    from nestynet_sr.sr_core.separability_math import check_mixed_compound
    from nestynet_sr.sr_search.features import TrigAxisSpec

    np.random.seed(42)
    N = 2000
    omega = 3.0

    # Generate data for f = x0 * x1 * cos(omega * x2)
    x0 = np.random.uniform(0.5, 2.0, N)
    x1 = np.random.uniform(0.5, 2.0, N)
    x2 = np.random.uniform(-np.pi, np.pi, N)

    # Compute gradients
    # df/dx0 = x1 * cos(omega * x2)
    # df/dx1 = x0 * cos(omega * x2)
    # df/dx2 = -omega * x0 * x1 * sin(omega * x2)
    dydx = np.zeros((N, 3))
    dydx[:, 0] = x1 * np.cos(omega * x2)
    dydx[:, 1] = x0 * np.cos(omega * x2)
    dydx[:, 2] = -omega * x0 * x1 * np.sin(omega * x2)

    x_vals = np.column_stack([x0, x1, x2])

    # Create trig axis spec for x2 (axis index 2)
    trig_specs = [
        TrigAxisSpec(
            axis=2,
            omega=omega,
            strength=50.0,  # Strong trig signature
            n_points=N,
            tmin=-np.pi,
            tmax=np.pi,
        )
    ]

    # Run mixed compound detection
    proposals = check_mixed_compound(
        var_idxs=(0, 1, 2),
        x_vals=x_vals,
        dydx_vals=dydx,
        trig_axis_specs=trig_specs,
        max_exponent=2,
        precision=0.1,
        min_overall_confidence=0.3,
    )

    # Should detect a mixed compound
    assert len(proposals) > 0, "Should detect at least one mixed compound proposal"

    best = proposals[0]
    print(f"Best proposal: linear={best.linear_var_idxs}, exp={best.linear_exponents}, "
          f"trig={best.trig_var_idxs}, omega={best.trig_omegas}, conf={best.overall_confidence:.3f}")

    # Verify structure
    assert set(best.linear_var_idxs) == {0, 1}, f"Linear vars should be (0, 1), got {best.linear_var_idxs}"
    assert best.trig_var_idxs == (2,), f"Trig vars should be (2,), got {best.trig_var_idxs}"
    assert best.overall_confidence > 0.3, f"Confidence should be > 0.3, got {best.overall_confidence}"

    # Check that omega is approximately correct
    assert len(best.trig_omegas) == 1
    # Note: omega is passed through from trig_axis_specs, so it should match
    assert abs(best.trig_omegas[0] - omega) < 0.1, f"Omega should be ~{omega}, got {best.trig_omegas[0]}"


def test_mixed_compound_ratio_times_cos():
    """Test detection of f = (x0 / x1) * cos(omega * x2)."""
    from nestynet_sr.sr_core.separability_math import check_mixed_compound
    from nestynet_sr.sr_search.features import TrigAxisSpec

    np.random.seed(123)
    N = 2000
    omega = 2.5

    # Generate data for f = (x0 / x1) * cos(omega * x2)
    x0 = np.random.uniform(0.5, 2.0, N)
    x1 = np.random.uniform(0.5, 2.0, N)
    x2 = np.random.uniform(-np.pi, np.pi, N)

    # Compute gradients
    # f = (x0 / x1) * cos(omega * x2)
    # df/dx0 = (1 / x1) * cos(omega * x2)
    # df/dx1 = -(x0 / x1^2) * cos(omega * x2)
    # df/dx2 = -omega * (x0 / x1) * sin(omega * x2)
    dydx = np.zeros((N, 3))
    dydx[:, 0] = (1 / x1) * np.cos(omega * x2)
    dydx[:, 1] = -(x0 / x1**2) * np.cos(omega * x2)
    dydx[:, 2] = -omega * (x0 / x1) * np.sin(omega * x2)

    x_vals = np.column_stack([x0, x1, x2])

    # Create trig axis spec for x2
    trig_specs = [
        TrigAxisSpec(
            axis=2,
            omega=omega,
            strength=50.0,
            n_points=N,
            tmin=-np.pi,
            tmax=np.pi,
        )
    ]

    # Run detection
    proposals = check_mixed_compound(
        var_idxs=(0, 1, 2),
        x_vals=x_vals,
        dydx_vals=dydx,
        trig_axis_specs=trig_specs,
        max_exponent=2,
        precision=0.1,
    )

    assert len(proposals) > 0, "Should detect ratio * cos compound"

    best = proposals[0]
    print(f"Best proposal: linear={best.linear_var_idxs}, exp={best.linear_exponents}, "
          f"trig={best.trig_var_idxs}, conf={best.overall_confidence:.3f}")

    # Linear part should be x0 and x1
    assert set(best.linear_var_idxs) == {0, 1}
    # Exponents should be (1, -1) or equivalent
    assert best.linear_exponents[0] * best.linear_exponents[1] < 0, \
        f"Exponents should have opposite signs (ratio), got {best.linear_exponents}"


def test_mixed_compound_multiple_trig():
    """Test detection of f = x0 * cos(x1) * sin(x2)."""
    from nestynet_sr.sr_core.separability_math import check_mixed_compound
    from nestynet_sr.sr_search.features import TrigAxisSpec

    np.random.seed(456)
    N = 2000
    omega1 = 1.0
    omega2 = 1.0

    # Generate data for f = x0 * cos(omega1 * x1) * sin(omega2 * x2)
    x0 = np.random.uniform(0.5, 2.0, N)
    x1 = np.random.uniform(-np.pi, np.pi, N)
    x2 = np.random.uniform(-np.pi, np.pi, N)

    # Compute gradients
    # df/dx0 = cos(omega1 * x1) * sin(omega2 * x2)
    # df/dx1 = -omega1 * x0 * sin(omega1 * x1) * sin(omega2 * x2)
    # df/dx2 = omega2 * x0 * cos(omega1 * x1) * cos(omega2 * x2)
    dydx = np.zeros((N, 3))
    dydx[:, 0] = np.cos(omega1 * x1) * np.sin(omega2 * x2)
    dydx[:, 1] = -omega1 * x0 * np.sin(omega1 * x1) * np.sin(omega2 * x2)
    dydx[:, 2] = omega2 * x0 * np.cos(omega1 * x1) * np.cos(omega2 * x2)

    x_vals = np.column_stack([x0, x1, x2])

    # Create trig axis specs for x1 and x2
    trig_specs = [
        TrigAxisSpec(axis=1, omega=omega1, strength=50.0, n_points=N, tmin=-np.pi, tmax=np.pi),
        TrigAxisSpec(axis=2, omega=omega2, strength=50.0, n_points=N, tmin=-np.pi, tmax=np.pi),
    ]

    # Run detection
    proposals = check_mixed_compound(
        var_idxs=(0, 1, 2),
        x_vals=x_vals,
        dydx_vals=dydx,
        trig_axis_specs=trig_specs,
        max_exponent=2,
        precision=0.2,  # Relax precision for multiple trig case
    )

    assert len(proposals) > 0, "Should detect x0 * cos(x1) * sin(x2) compound"

    best = proposals[0]
    print(f"Best proposal: linear={best.linear_var_idxs}, exp={best.linear_exponents}, "
          f"trig={best.trig_var_idxs}, kinds={best.trig_kinds}, conf={best.overall_confidence:.3f}")

    # Linear part should be x0 only
    assert best.linear_var_idxs == (0,), f"Expected linear var (0,), got {best.linear_var_idxs}"
    # Trig part should be x1 and x2
    assert set(best.trig_var_idxs) == {1, 2}, f"Expected trig vars {{1, 2}}, got {best.trig_var_idxs}"


def test_build_mixed_compound_ast():
    """Test AST construction for mixed compounds."""
    from nestynet_sr.sr_core.separability_math import build_mixed_compound_ast
    from nestynet_sr.sr_core.bridges import ast_to_human_readable

    # Build z = x0 * x1 * cos(2.0 * x2)
    z_ast = build_mixed_compound_ast(
        linear_var_idxs=(0, 1),
        linear_exponents=(1, 1),
        trig_var_idxs=(2,),
        trig_omegas=(2.0,),
        trig_kinds=("cos",),
        trig_phases=(0.0,),
    )

    readable = ast_to_human_readable(z_ast)
    print(f"AST readable: {readable}")

    # Should contain x0, x1, cos, and 2.0
    assert "x0" in readable.lower() or "var(0)" in readable.lower()
    assert "cos" in readable.lower()


def test_no_trig_axes_returns_empty():
    """Test that no proposals are returned when there are no trig axes."""
    from nestynet_sr.sr_core.separability_math import check_mixed_compound

    np.random.seed(789)
    N = 1000

    # Pure monomial: f = x0 * x1
    x0 = np.random.uniform(0.5, 2.0, N)
    x1 = np.random.uniform(0.5, 2.0, N)

    dydx = np.zeros((N, 2))
    dydx[:, 0] = x1  # df/dx0 = x1
    dydx[:, 1] = x0  # df/dx1 = x0

    x_vals = np.column_stack([x0, x1])

    # No trig specs - should return empty (not a "mixed" compound)
    proposals = check_mixed_compound(
        var_idxs=(0, 1),
        x_vals=x_vals,
        dydx_vals=dydx,
        trig_axis_specs=[],  # No trig axes
    )

    assert len(proposals) == 0, "Should return empty for pure monomial (no trig)"


def test_all_trig_returns_empty():
    """Test that no proposals are returned when all axes are trig."""
    from nestynet_sr.sr_core.separability_math import check_mixed_compound
    from nestynet_sr.sr_search.features import TrigAxisSpec

    np.random.seed(101)
    N = 1000

    x0 = np.random.uniform(-np.pi, np.pi, N)
    x1 = np.random.uniform(-np.pi, np.pi, N)

    # Dummy gradients
    dydx = np.random.randn(N, 2)
    x_vals = np.column_stack([x0, x1])

    # All axes are trig
    trig_specs = [
        TrigAxisSpec(axis=0, omega=1.0, strength=50.0, n_points=N, tmin=-np.pi, tmax=np.pi),
        TrigAxisSpec(axis=1, omega=1.0, strength=50.0, n_points=N, tmin=-np.pi, tmax=np.pi),
    ]

    proposals = check_mixed_compound(
        var_idxs=(0, 1),
        x_vals=x_vals,
        dydx_vals=dydx,
        trig_axis_specs=trig_specs,
    )

    # Should return empty - all trig is not a "mixed" compound
    assert len(proposals) == 0, "Should return empty for all-trig (not mixed)"


if __name__ == "__main__":
    print("Running test_mixed_compound_basic...")
    test_mixed_compound_basic()
    print("PASSED\n")

    print("Running test_mixed_compound_ratio_times_cos...")
    test_mixed_compound_ratio_times_cos()
    print("PASSED\n")

    print("Running test_mixed_compound_multiple_trig...")
    test_mixed_compound_multiple_trig()
    print("PASSED\n")

    print("Running test_build_mixed_compound_ast...")
    test_build_mixed_compound_ast()
    print("PASSED\n")

    print("Running test_no_trig_axes_returns_empty...")
    test_no_trig_axes_returns_empty()
    print("PASSED\n")

    print("Running test_all_trig_returns_empty...")
    test_all_trig_returns_empty()
    print("PASSED\n")

    print("All tests passed!")
