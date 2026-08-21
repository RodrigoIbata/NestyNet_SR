# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""
Test the pure difference compound detection (z = xi - xj).

This tests the _detect_pure_difference_compounds function which detects
when a function depends on the difference of two variables (e.g., cos(x4-x5)).
"""

import numpy as np
import torch


def test_pure_difference_detection_basic():
    """Test that _detect_pure_difference_compounds finds z = x0 - x1."""
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds
    from dataclasses import dataclass

    @dataclass
    class MockFeatureSpec:
        kind: str
        coeffs: torch.Tensor
        name: str = "mock_feat"

    # Function: f(x0, x1) = cos(x0 - x1)
    # Gradient: df/dx0 = -sin(x0 - x1), df/dx1 = sin(x0 - x1)
    # So df/dx0 + df/dx1 = 0 (pure difference signature)

    np.random.seed(42)
    N = 500
    x0 = np.random.uniform(0, 6, N)
    x1 = np.random.uniform(0, 6, N)
    x_vals = np.column_stack([x0, x1])

    # Compute gradients: df/dx0 = -sin(x0-x1), df/dx1 = sin(x0-x1)
    diff = x0 - x1
    grad_x0 = -np.sin(diff)
    grad_x1 = np.sin(diff)
    dydx_vals = np.column_stack([grad_x0, grad_x1])

    # Create invariance feature indicating same-sign coefficients [+1, +1]
    # This signals x0 + x1 invariance, implying dependence on x0 - x1
    invariance_feats = [
        MockFeatureSpec(kind="integer_linear", coeffs=torch.tensor([1.0, 1.0]))
    ]

    proposals = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1),
        invariance_feats=invariance_feats,
        precision=0.1,
    )

    print(f"Found {len(proposals)} proposals")
    for p in proposals:
        print(f"  Proposal: {p}")

    assert len(proposals) >= 1, "Should find at least one pure difference proposal"

    # Check the first proposal
    coeffs, z_ast, conf, extra, meta = proposals[0]
    print(f"Coefficients: {coeffs}")
    print(f"Confidence: {conf}")
    print(f"Meta: {meta}")

    assert meta.get("kind") in ("pure_difference", "power_difference"), "Should be a pure_difference proposal"
    assert conf > 0.9, f"Confidence should be high, got {conf}"

    # The linear_coeffs should indicate z = x0 - x1 (or x1 - x0)
    # Either (1, -1) or (-1, 1) are valid
    assert coeffs in [(1, -1), (-1, 1)], f"Coefficients should be (1,-1) or (-1,1), got {coeffs}"

    print("PASSED: Basic pure difference detection")


def test_pure_difference_detection_6var():
    """Test pure difference detection in a 6-variable context (like pb101)."""
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds
    from dataclasses import dataclass

    @dataclass
    class MockFeatureSpec:
        kind: str
        coeffs: torch.Tensor
        name: str = "mock_feat"

    # Simulating pb101: f(x0,...,x5) depends on cos(x4 - x5)
    # The invariance feature would be coeffs=[0,0,0,0,1,1] (x4+x5 invariance)

    np.random.seed(42)
    N = 500
    x = np.random.uniform(1, 3, (N, 6))
    x[:, 4] = np.random.uniform(0, 6, N)  # x4
    x[:, 5] = np.random.uniform(0, 6, N)  # x5

    # Simplified model: f depends on cos(x4-x5) primarily
    # Gradient: most components small, but df/dx4 = -sin(x4-x5), df/dx5 = sin(x4-x5)
    diff = x[:, 4] - x[:, 5]
    dydx = np.zeros((N, 6))
    dydx[:, 0] = np.random.randn(N) * 0.1  # Some noise in other components
    dydx[:, 1] = np.random.randn(N) * 0.1
    dydx[:, 2] = np.random.randn(N) * 0.1
    dydx[:, 3] = np.random.randn(N) * 0.1
    dydx[:, 4] = -np.sin(diff)
    dydx[:, 5] = np.sin(diff)

    # Invariance feature: x4 + x5 invariance (same-sign coeffs for x4, x5)
    invariance_feats = [
        MockFeatureSpec(
            kind="integer_linear",
            coeffs=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        )
    ]

    proposals = _detect_pure_difference_compounds(
        x_vals=x,
        dydx_vals=dydx,
        var_idxs=(0, 1, 2, 3, 4, 5),
        invariance_feats=invariance_feats,
        precision=0.1,
    )

    print(f"Found {len(proposals)} proposals for 6-var case")
    for p in proposals:
        print(f"  Proposal: {p}")

    assert len(proposals) >= 1, "Should find at least one pure difference proposal"

    # Check the proposal
    coeffs, z_ast, conf, extra, meta = proposals[0]
    print(f"Meta: {meta}")
    assert meta.get("kind") in ("pure_difference", "power_difference")
    assert meta.get("indices") in [(4, 5), (5, 4)], f"Should identify x4-x5 pair, got {meta.get('indices')}"
    assert conf > 0.9, f"Confidence should be high, got {conf}"

    print("PASSED: 6-variable pure difference detection")


def test_no_false_positives():
    """Test that pure difference detection doesn't fire on non-difference functions."""
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds
    from dataclasses import dataclass

    @dataclass
    class MockFeatureSpec:
        kind: str
        coeffs: torch.Tensor
        name: str = "mock_feat"

    # Function: f(x0, x1) = x0 * x1 (product, not difference)
    # Gradient: df/dx0 = x1, df/dx1 = x0
    # df/dx0 + df/dx1 = x0 + x1 ≠ 0 (not a pure difference)

    np.random.seed(42)
    N = 500
    x0 = np.random.uniform(1, 5, N)
    x1 = np.random.uniform(1, 5, N)
    x_vals = np.column_stack([x0, x1])

    # Gradients for product: df/dx0 = x1, df/dx1 = x0
    dydx_vals = np.column_stack([x1, x0])

    # Even if we mistakenly provide an invariance feature suggesting same-sign,
    # the gradient test should reject it
    invariance_feats = [
        MockFeatureSpec(kind="integer_linear", coeffs=torch.tensor([1.0, 1.0]))
    ]

    proposals = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1),
        invariance_feats=invariance_feats,
        precision=0.1,  # 10% tolerance
    )

    print(f"Found {len(proposals)} proposals for non-difference function")

    # Should not find proposals because df/dx0 + df/dx1 ≠ 0
    assert len(proposals) == 0, f"Should not find proposals for product function, found {len(proposals)}"

    print("PASSED: No false positives for non-difference function")


def test_antigradient_detection_without_invariance_feature():
    """The direct gi+gj check should find a difference without feature hints."""
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds

    rng = np.random.default_rng(123)
    N = 200
    x_vals = rng.uniform(0.5, 3.0, size=(N, 4))

    # Only 40 nonzero pair-gradient points: this is below the ratio scanner's
    # valid-point floor, but the anti-gradient certificate is still exact.
    dydx_vals = np.zeros((N, 4))
    g = np.linspace(0.2, 1.0, 40)
    dydx_vals[:40, 2] = g
    dydx_vals[:40, 3] = -g

    proposals = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1, 2, 3),
        invariance_feats=None,
        precision=0.05,
    )

    hits = [
        p for p in proposals
        if p[4].get("indices") in ((2, 3), (3, 2))
    ]
    assert hits, "Should detect x2-x3 from anti-gradient evidence alone"
    assert hits[0][4].get("source") == "anti_gradient"
    assert hits[0][2] > 0.95


def test_antigradient_detection_pb102_like_without_invariance_feature():
    """pb102-like denominator depends on x2,x3 only through x2-x3."""
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds

    rng = np.random.default_rng(456)
    N = 512
    x0 = rng.uniform(0.5, 2.0, N)
    x1 = rng.uniform(0.1, 0.8, N)
    x2 = rng.uniform(0.0, 3.0, N)
    x3 = rng.uniform(0.0, 3.0, N)
    x_vals = np.column_stack([x0, x1, x2, x3])

    d = x2 - x3
    numer = x0 * (1.0 - x1 ** 2)
    denom = 1.0 + x1 * np.cos(d)

    dydx_vals = np.zeros_like(x_vals)
    dydx_vals[:, 0] = (1.0 - x1 ** 2) / denom
    dydx_vals[:, 1] = (-2.0 * x0 * x1) / denom - numer * np.cos(d) / (denom ** 2)
    dydx_vals[:, 2] = numer * x1 * np.sin(d) / (denom ** 2)
    dydx_vals[:, 3] = -dydx_vals[:, 2]

    proposals = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1, 2, 3),
        invariance_feats=None,
        precision=0.05,
    )

    assert any(
        p[4].get("indices") in ((2, 3), (3, 2)) and p[2] > 0.95
        for p in proposals
    )


def test_disjoint_difference_pairs_are_bundled_first():
    """Two independent differences should be tried as one coordinate lift first."""
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds

    rng = np.random.default_rng(654)
    N = 256
    x_vals = rng.uniform(0.5, 3.0, size=(N, 4))
    g01 = 0.2 + 0.1 * x_vals[:, 0]
    g23 = 0.3 + 0.2 * x_vals[:, 2]
    dydx_vals = np.column_stack([g01, -g01, g23, -g23])

    proposals = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1, 2, 3),
        invariance_feats=None,
        precision=0.05,
    )

    assert proposals
    coeffs, _z_ast, conf, extra, meta = proposals[0]
    assert meta.get("kind") == "power_difference_bundle"
    assert int(meta.get("bundle_size")) == 2
    assert tuple(coeffs) == (1, -1, 1, -1)
    assert conf > 0.95
    assert extra == []
    assert len(meta.get("extra_input_asts", ())) == 1


def test_antigradient_preserves_existing_compound_for_extra_difference():
    """On NN[z,x2,x3], finding x2-x3 should keep the old z as an input."""
    from nestynet_sr.sr_core.separability_math import build_linear_ast
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds

    rng = np.random.default_rng(789)
    N = 200
    x_vals = rng.uniform(0.5, 3.0, size=(N, 3))  # effective inputs: z, x2, x3
    dydx_vals = np.zeros((N, 3))
    dydx_vals[:, 0] = 0.3 + 0.1 * x_vals[:, 0]
    g = np.linspace(0.2, 1.0, 40)
    dydx_vals[:40, 1] = g
    dydx_vals[:40, 2] = -g

    z_old = build_linear_ast((1, 0), (1, -1))
    proposals = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=("z", 2, 3),
        invariance_feats=None,
        precision=0.05,
        z_ast_existing=z_old,
    )

    hits = [
        p for p in proposals
        if p[4].get("indices") in ((2, 3), (3, 2))
    ]
    assert hits
    coeffs, _z_ast, conf, extra, meta = hits[0]
    assert coeffs[0] == 0
    assert conf > 0.95
    assert extra == []
    assert meta.get("source") == "anti_gradient"
    assert "preserve_z_ast" in meta


def test_antigradient_difference_respects_units_when_requested():
    """Unitful differences are only proposed when dimensions match."""
    from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds

    us = UnitSystem(("L", "T"))
    units_spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([1, 0]), us.dim([0, 1])),
        y_dim=us.dim([0, 0]),
    )

    rng = np.random.default_rng(321)
    N = 200
    x_vals = rng.uniform(0.5, 3.0, size=(N, 2))
    g = np.sin(x_vals[:, 0])
    dydx_vals = np.column_stack([g, -g])

    blocked = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1),
        invariance_feats=None,
        precision=0.05,
        units_spec=units_spec,
        enforce_units=True,
    )
    assert blocked == []

    allowed_without_units = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1),
        invariance_feats=None,
        precision=0.05,
        units_spec=units_spec,
        enforce_units=False,
    )
    assert allowed_without_units


def test_reciprocal_difference_detection_from_gradient_colinearity():
    """Detect z = 1/x0 - 1/x1 as the same kind of pair coordinate."""
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds

    rng = np.random.default_rng(987)
    N = 300
    x0 = rng.uniform(0.7, 4.0, N)
    x1 = rng.uniform(0.7, 4.0, N)
    x_vals = np.column_stack([x0, x1])

    # f = 1/x1 - 1/x0.  The detector may propose the opposite sign
    # coordinate 1/x0 - 1/x1; a downstream NN/scale can absorb that sign.
    dydx_vals = np.column_stack([
        1.0 / (x0 ** 2),
        -1.0 / (x1 ** 2),
    ])

    proposals = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1),
        invariance_feats=None,
        precision=0.05,
    )

    hits = [
        p for p in proposals
        if (p[4] or {}).get("kind") == "power_pair_sumdiff"
        and int((p[4] or {}).get("power", 0)) == 1
        and (p[4] or {}).get("op") == "minus"
        and bool((p[4] or {}).get("left_inverse"))
        and bool((p[4] or {}).get("right_inverse"))
    ]
    assert hits, "Should detect reciprocal difference from gradient colinearity"
    assert hits[0][2] > 0.95


def test_reciprocal_difference_respects_units_when_requested():
    """1/x0 +/- 1/x1 is proposed only when the term dimensions match."""
    from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
    from nestynet_sr.sr_search.search import _detect_pure_difference_compounds

    us = UnitSystem(("L", "T"))
    units_spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([1, 0]), us.dim([0, 1])),
        y_dim=us.dim([0, 0]),
    )

    rng = np.random.default_rng(6543)
    N = 300
    x0 = rng.uniform(0.7, 4.0, N)
    x1 = rng.uniform(0.7, 4.0, N)
    x_vals = np.column_stack([x0, x1])
    dydx_vals = np.column_stack([
        1.0 / (x0 ** 2),
        -1.0 / (x1 ** 2),
    ])

    blocked = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1),
        invariance_feats=None,
        precision=0.05,
        units_spec=units_spec,
        enforce_units=True,
    )
    assert not any((p[4] or {}).get("kind") == "power_pair_sumdiff" for p in blocked)

    allowed_without_units = _detect_pure_difference_compounds(
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        var_idxs=(0, 1),
        invariance_feats=None,
        precision=0.05,
        units_spec=units_spec,
        enforce_units=False,
    )
    assert any(
        (p[4] or {}).get("kind") == "power_pair_sumdiff"
        for p in allowed_without_units
    )


if __name__ == "__main__":
    test_pure_difference_detection_basic()
    print()
    test_pure_difference_detection_6var()
    print()
    test_no_false_positives()
    print()
    test_antigradient_detection_without_invariance_feature()
    print()
    test_antigradient_detection_pb102_like_without_invariance_feature()
    print()
    test_disjoint_difference_pairs_are_bundled_first()
    print()
    test_antigradient_preserves_existing_compound_for_extra_difference()
    print()
    test_antigradient_difference_respects_units_when_requested()
    print()
    test_reciprocal_difference_detection_from_gradient_colinearity()
    print()
    test_reciprocal_difference_respects_units_when_requested()
    print()
    print("All tests passed!")
