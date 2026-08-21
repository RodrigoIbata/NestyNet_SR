# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Tests for compound variable detection and AST support.

Tests:
1. Detection algorithm (check_monomial_compound)
2. AST helpers (build_monomial_ast, Div)
3. Chain rule with compound variables (ASTCompositeAdaptor)
"""

import numpy as np
import torch
import pytest

torch.set_default_dtype(torch.float64)

from nestynet_sr.sr_core import (
    build_monomial_ast,
    check_monomial_compound,
    Div,
    Var,
    Mul,
    ast_to_human_readable,
)
from nestynet_sr.sr_core.bridges import Pow, AtomNode, MulNode


# ──────────────────────────────────────────────────────────────
# Test check_monomial_compound detection
# ──────────────────────────────────────────────────────────────


def test_product_detection():
    """Test detection of f(x0 * x1) with exponents (1, 1)."""
    np.random.seed(42)
    N = 200
    x0 = np.random.uniform(0.5, 2.0, N)
    x1 = np.random.uniform(0.5, 2.0, N)
    x_vals = np.stack([x0, x1], axis=1)

    # f = sin(x0 * x1), so df/dx0 = x1 * cos(x0*x1), df/dx1 = x0 * cos(x0*x1)
    z = x0 * x1
    cos_z = np.cos(z)
    dydx_vals = np.stack([x1 * cos_z, x0 * cos_z], axis=1)

    proposals, _ = check_monomial_compound(
        var_idxs=(0, 1),
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        max_exponent=2,
        precision=0.1,
    )

    assert len(proposals) > 0, "Should detect at least one compound proposal"
    exponents, confidence = proposals[0]
    assert exponents == (1, 1), f"Expected (1, 1), got {exponents}"
    assert confidence > 0.8, f"Expected high confidence, got {confidence}"


def test_ratio_detection():
    """Test detection of f(x0 / x1) with exponents (1, -1)."""
    np.random.seed(42)
    N = 200
    x0 = np.random.uniform(0.5, 2.0, N)
    x1 = np.random.uniform(0.5, 2.0, N)
    x_vals = np.stack([x0, x1], axis=1)

    # f = exp(x0 / x1), so df/dx0 = (1/x1) * exp(x0/x1), df/dx1 = -x0/x1^2 * exp(x0/x1)
    z = x0 / x1
    exp_z = np.exp(z)
    dydx_vals = np.stack([exp_z / x1, -x0 * exp_z / (x1 ** 2)], axis=1)

    proposals, _ = check_monomial_compound(
        var_idxs=(0, 1),
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        max_exponent=2,
        precision=0.1,
    )

    assert len(proposals) > 0, "Should detect at least one compound proposal"
    exponents, confidence = proposals[0]
    assert exponents == (1, -1), f"Expected (1, -1), got {exponents}"
    assert confidence > 0.8, f"Expected high confidence, got {confidence}"


def test_no_compound_separable():
    """Test that separable functions don't trigger compound detection."""
    np.random.seed(42)
    N = 200
    x0 = np.random.uniform(0.5, 2.0, N)
    x1 = np.random.uniform(0.5, 2.0, N)
    x_vals = np.stack([x0, x1], axis=1)

    # f = sin(x0) + cos(x1) (separable), df/dx0 = cos(x0), df/dx1 = -sin(x1)
    dydx_vals = np.stack([np.cos(x0), -np.sin(x1)], axis=1)

    proposals, _ = check_monomial_compound(
        var_idxs=(0, 1),
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        max_exponent=2,
        precision=0.1,
    )

    # Should either return no proposals or low confidence
    if len(proposals) > 0:
        _, confidence = proposals[0]
        assert confidence < 0.5, f"Separable function should have low confidence, got {confidence}"


# ──────────────────────────────────────────────────────────────
# Test build_monomial_ast
# ──────────────────────────────────────────────────────────────


def test_build_product_ast():
    """Test building AST for z = x0 * x1."""
    z_ast = build_monomial_ast((0, 1), (1, 1))
    expr = ast_to_human_readable(z_ast)
    # Should contain references to both variables
    assert "x_0" in expr or "x0" in expr, f"Expected x_0 or x0 in {expr}"
    assert "x_1" in expr or "x1" in expr, f"Expected x_1 or x1 in {expr}"


def test_build_ratio_ast():
    """Test building AST for z = x0 / x1."""
    z_ast = build_monomial_ast((0, 1), (1, -1))
    # Should be Mul(Var(0), Pow(Var(1), -1))
    assert isinstance(z_ast, MulNode), f"Expected MulNode, got {type(z_ast)}"


def test_human_readable_inverse_of_ratio():
    """Display canonicalization: (x0/x1)^-1 -> (x1/x0)."""
    z = Mul(Var(0), Pow(Var(1), -1))
    z_inv = Pow(z, -1)
    expr = ast_to_human_readable(z_inv)
    assert expr == "(x1 / x0)", f"Expected '(x1 / x0)', got {expr}"


def test_human_readable_double_inverse():
    """Display canonicalization: (u^-1)^-1 -> u."""
    z = Pow(Var(0), -1)
    z_inv = Pow(z, -1)
    expr = ast_to_human_readable(z_inv)
    assert expr == "x0", f"Expected 'x0', got {expr}"


def test_build_power_ast():
    """Test building AST for z = x0^2 / x1."""
    z_ast = build_monomial_ast((0, 1), (2, -1))
    expr = ast_to_human_readable(z_ast)
    assert "x_0" in expr or "x0" in expr, f"Expected x_0 or x0 in {expr}"


# ──────────────────────────────────────────────────────────────
# Test Div helper
# ──────────────────────────────────────────────────────────────


def test_div_helper():
    """Test Div(a, b) produces Mul(a, Pow(b, -1))."""
    a = Var(0)
    b = Var(1)
    result = Div(a, b)

    assert isinstance(result, MulNode), f"Div should return MulNode, got {type(result)}"


# ──────────────────────────────────────────────────────────────
# Test compound variable chain rule (basic evaluation)
# ──────────────────────────────────────────────────────────────


def test_compound_variable_evaluation():
    """Test that compound variable AST evaluates correctly."""
    try:
        import nestynet
        from nestynet.adaptors.adaptors import SegmentedAdaptor
    except ImportError:
        pytest.skip("nestynet not available")

    from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor

    # Build z = x0 * x1
    z_ast = build_monomial_ast((0, 1), (1, 1))

    # Create atom with compound inputs
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        kwargs={"num_segments": 4, "dual_layer": False},
        inputs=(z_ast,),
    )

    # Build leaf (simple linear model for testing)
    def make_leaf():
        kw = dict(
            model_base_name="G_Model",
            model_scale=0.1,
            dtype=torch.float64,
            device="cpu",
            num_segments=4,
            seg_width=2,
        )
        net = nestynet.nets.NestyNet_Model(Nout_size=1, Nx_size=1, **kw)  # 1D input (z)
        seg = torch.arange(net.base_model.num_segments, device="cpu")
        return SegmentedAdaptor(net, segments=seg)

    leaf = make_leaf()
    adaptor = ASTCompositeAdaptor(atom, [leaf])

    # Test forward pass
    B = 10
    x = torch.rand(B, 2, dtype=torch.float64) + 0.5
    y = adaptor.forward(x)

    assert y.shape == (B, 1), f"Expected (B, 1) output, got {y.shape}"


def test_compound_variable_gradient():
    """Test that compound variable gradients are computed correctly via chain rule."""
    try:
        import nestynet
        from nestynet.adaptors.adaptors import SegmentedAdaptor
    except ImportError:
        pytest.skip("nestynet not available")

    from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor

    # Build z = x0 * x1
    z_ast = build_monomial_ast((0, 1), (1, 1))

    # Create atom with compound inputs
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        kwargs={"num_segments": 4, "dual_layer": False},
        inputs=(z_ast,),
    )

    # Build leaf
    def make_leaf():
        kw = dict(
            model_base_name="G_Model",
            model_scale=0.1,
            dtype=torch.float64,
            device="cpu",
            num_segments=4,
            seg_width=2,
        )
        net = nestynet.nets.NestyNet_Model(Nout_size=1, Nx_size=1, **kw)  # 1D input (z)
        seg = torch.arange(net.base_model.num_segments, device="cpu")
        return SegmentedAdaptor(net, segments=seg)

    leaf = make_leaf()
    adaptor = ASTCompositeAdaptor(atom, [leaf])

    # Test gradient computation
    B = 10
    x = torch.rand(B, 2, dtype=torch.float64) + 0.5

    g = adaptor.grad(x)
    assert g.shape == (B, 1, 2), f"Expected (B, 1, 2) gradient, got {g.shape}"

    # Verify chain rule: d/dx0[L(x0*x1)] = L'(z) * x1
    # Numerical gradient check
    eps = 1e-6
    x0_plus = x.clone()
    x0_plus[:, 0] += eps
    x0_minus = x.clone()
    x0_minus[:, 0] -= eps
    grad_x0_numerical = (adaptor.forward(x0_plus) - adaptor.forward(x0_minus)) / (2 * eps)

    grad_x0_analytical = g[:, 0, 0]
    err = (grad_x0_numerical.squeeze() - grad_x0_analytical).abs().max()
    assert err < 1e-4, f"Gradient mismatch: max error {err:.2e}"


if __name__ == "__main__":
    test_product_detection()
    print("test_product_detection PASSED")

    test_ratio_detection()
    print("test_ratio_detection PASSED")

    test_no_compound_separable()
    print("test_no_compound_separable PASSED")

    test_build_product_ast()
    print("test_build_product_ast PASSED")

    test_build_ratio_ast()
    print("test_build_ratio_ast PASSED")

    test_build_power_ast()
    print("test_build_power_ast PASSED")

    test_div_helper()
    print("test_div_helper PASSED")

    test_compound_variable_evaluation()
    print("test_compound_variable_evaluation PASSED")

    test_compound_variable_gradient()
    print("test_compound_variable_gradient PASSED")

    print("\nAll tests PASSED!")
