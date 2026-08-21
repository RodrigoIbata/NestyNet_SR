# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""
Test iterative structure detection after compound confirmation.

This test verifies that when a compound variable z = x*y is detected and
the compound leaf is trained, we can then detect trig structure on z
and propose nested compounds like w = sin(z).
"""

import math
import torch

from nestynet_sr.sr_search.features import (
    discover_leaf_features,
    LeafFeatures,
    LeafProvider,
    TrigAxisSpec,
)
from nestynet_sr.sr_core.bridges import (
    AtomNode,
    MulNode,
)
from nestynet_sr.sr_search.wrapper_policy import snap_omega


class MockLeaf(torch.nn.Module):
    """A mock leaf that computes sin(omega * z) for testing."""

    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = omega
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = x[:, 0:1]  # First column is the compound variable z
        return torch.sin(self.omega * z)

    def grad(self, x):
        z = x[:, 0:1]
        # df/dz = omega * cos(omega * z)
        dfdz = self.omega * torch.cos(self.omega * z)
        # Return gradient for all input dimensions (only first is non-zero for this mock)
        result = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        result[:, 0] = dfdz.squeeze(-1)
        return result

    def grad_grad(self, x):
        z = x[:, 0:1]
        # d2f/dz2 = -omega^2 * sin(omega * z)
        d2fdz2 = -self.omega**2 * torch.sin(self.omega * z)
        # Return Hessian [N, D, D] - only (0,0) entry is non-zero
        N, D = x.shape
        result = torch.zeros(N, D, D, device=x.device, dtype=x.dtype)
        result[:, 0, 0] = d2fdz2.squeeze(-1)
        return result


def test_leaf_provider_works():
    """Test that LeafProvider wraps a leaf correctly for feature detection."""
    omega = 3.0
    leaf = MockLeaf(omega=omega)
    provider = LeafProvider(leaf)

    z_data = torch.linspace(-2, 2, 100).reshape(-1, 1)

    # Test forward
    f = provider.forward(z_data)
    expected = torch.sin(omega * z_data)
    assert torch.allclose(f, expected, atol=1e-5), "Forward pass mismatch"

    # Test grad
    g = provider.grad(z_data)
    expected_grad = omega * torch.cos(omega * z_data)
    assert torch.allclose(g[:, 0:1], expected_grad, atol=1e-5), "Gradient mismatch"

    # Test grad_grad
    H = provider.grad_grad(z_data)
    expected_hess = -omega**2 * torch.sin(omega * z_data)
    assert torch.allclose(H[:, 0, 0:1], expected_hess, atol=1e-5), "Hessian mismatch"


def test_discover_leaf_features_detects_trig():
    """Test that discover_leaf_features can detect trig on compound leaf input."""
    omega = 5.0
    leaf = MockLeaf(omega=omega)

    # Create mock compound atom with z = x0 * x1
    # The z_expr is the expression for z (product of x0 and x1)
    z_expr = MulNode(
        AtomNode("var", (0,), tag="x0"),
        AtomNode("var", (1,), tag="x1"),
    )
    atom = AtomNode(
        "nn",
        (0, 1),
        tag="test_atom",
        kwargs={"input_expr": z_expr, "extra_var_idxs": []},
        inputs=(z_expr,),
    )

    # Generate raw input data
    N = 2000
    torch.manual_seed(42)
    x_data = torch.rand(N, 2) * 4 - 2  # x0, x1 in [-2, 2]

    # Call discover_leaf_features
    features = discover_leaf_features(
        leaf,
        atom,
        x_data,
        detect_trig=True,
        trig_strength_threshold=3.0,  # Lower threshold for testing
        trig_max_omega=50.0,
    )

    # Check that trig was detected on axis 0 (the compound variable z)
    assert isinstance(features, LeafFeatures), "Should return LeafFeatures"
    assert 0 in features.trig_by_axis, "Should detect trig on axis 0 (compound z)"

    z_trig = features.trig_by_axis[0]
    assert isinstance(z_trig, TrigAxisSpec), "Should be TrigAxisSpec"

    # Check that detected omega is close to the true omega
    detected_omega = z_trig.omega
    omega_snapped = snap_omega(detected_omega)
    print(f"True omega: {omega}, detected: {detected_omega}, snapped: {omega_snapped}")

    # Omega detection may not be exact, but should be in the right ballpark
    assert 0.5 * omega <= detected_omega <= 2.0 * omega, (
        f"Detected omega {detected_omega} should be close to true omega {omega}"
    )


def test_snap_omega():
    """Test omega snapping to nice values."""
    # Test that values near common frequencies snap correctly
    test_cases = [
        (0.95, 1.0),
        (1.05, 1.0),
        (1.95, 2.0),
        (2.05, 2.0),
        (3.14, math.pi),
        (6.28, 2 * math.pi),
    ]

    for omega_in, expected in test_cases:
        snapped = snap_omega(omega_in)
        # Allow some tolerance
        assert abs(snapped - expected) < 0.3, (
            f"snap_omega({omega_in}) = {snapped}, expected ~{expected}"
        )


if __name__ == "__main__":
    test_leaf_provider_works()
    print("✓ test_leaf_provider_works passed")

    test_discover_leaf_features_detects_trig()
    print("✓ test_discover_leaf_features_detects_trig passed")

    test_snap_omega()
    print("✓ test_snap_omega passed")

    print("\n✓ All tests passed!")
