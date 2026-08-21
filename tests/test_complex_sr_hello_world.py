# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Hello-world tests for complex number symbolic regression.

These tests verify that NestyNet can handle simple complex equations represented
as 2-component real vector outputs [Re, Im]. This is analogous to easy AI Feynman
problems but for complex-valued functions.

Test approach:
1. Create synthetic data for simple complex equations
2. Train 2-output surrogate (for complex output) or 1-output (for scalar)
3. Verify forward evaluation matches expected complex output
4. Run order-0 system DE discovery to find polynomial structure

Benchmarks from data/feynman_complex_simple.txt:
- CX000: identity z -> [x0, x1]
- CX006: z^2 -> [x0^2-x1^2, 2*x0*x1]
- CX012: z*w -> [x0*x2-x1*x3, x0*x3+x1*x2]
- CX022: |z|^2*z -> [(x0^2+x1^2)*x0, (x0^2+x1^2)*x1]
"""

import numpy as np
import torch
import pytest

torch.set_default_dtype(torch.float64)


# =============================================================================
# Mock Surrogates (exact polynomial implementations)
# =============================================================================

class IdentitySurrogate(torch.nn.Module):
    """CX000: identity z = x0 + i*x1 -> [x0, x1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()  # [x0, x1] -> [x0, x1]

    def parameters(self):
        return iter([torch.zeros(1)])


class ComplexSquareSurrogate(torch.nn.Module):
    """CX006: z^2 = (x0+i*x1)^2 -> [x0^2-x1^2, 2*x0*x1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1 = x[:, 0], x[:, 1]
        re = x0**2 - x1**2
        im = 2.0 * x0 * x1
        return torch.stack([re, im], dim=1)

    def parameters(self):
        return iter([torch.zeros(1)])


class ComplexProductSurrogate(torch.nn.Module):
    """CX012: z*w -> [x0*x2-x1*x3, x0*x3+x1*x2]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        re = x0 * x2 - x1 * x3
        im = x0 * x3 + x1 * x2
        return torch.stack([re, im], dim=1)

    def parameters(self):
        return iter([torch.zeros(1)])


class NLSCubicSurrogate(torch.nn.Module):
    """CX022: |z|^2*z -> [(x0^2+x1^2)*x0, (x0^2+x1^2)*x1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1 = x[:, 0], x[:, 1]
        mod2 = x0**2 + x1**2
        re = mod2 * x0
        im = mod2 * x1
        return torch.stack([re, im], dim=1)

    def parameters(self):
        return iter([torch.zeros(1)])


class MagnitudeSquaredSurrogate(torch.nn.Module):
    """CX005: |z|^2 = x0^2 + x1^2 (scalar output)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1 = x[:, 0], x[:, 1]
        return (x0**2 + x1**2).unsqueeze(-1)

    def parameters(self):
        return iter([torch.zeros(1)])


# =============================================================================
# Dataloader utilities
# =============================================================================

def make_dataloader(nvars: int, N: int = 500, xmin: float = -2.0, xmax: float = 2.0, seed: int = 42):
    """Create a simple dataloader with uniform random samples."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(xmin, xmax, (N, nvars))
    X = torch.tensor(x, dtype=torch.float64)
    return [(X,)]  # single-batch iterable


# =============================================================================
# Test: CX006 - Complex Square z^2
# =============================================================================

def test_cx006_complex_square():
    """Test discovery of z^2 = (x0+i*x1)^2 -> [x0^2-x1^2, 2*x0*x1]."""
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import Var, Mul, Pow

    surrogate = ComplexSquareSurrogate()
    dl = make_dataloader(nvars=2)

    # Library: x0, x1, x0^2, x1^2, x0*x1
    x0, x1 = Var(0), Var(1)
    lib = [
        x0,
        x1,
        Pow(x0, 2),
        Pow(x1, 2),
        Mul(x0, x1),
    ]

    cfg = SystemDESearchConfig(
        order_candidates=(0,),
        include_const=True,
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate=surrogate,
        train_dataloader=dl,
        cfg=cfg,
        library_terms=lib,
    )

    assert result.order == 0, f"Expected order=0, got {result.order}"
    assert len(result.out_idxs) == 2, f"Expected 2 outputs, got {len(result.out_idxs)}"

    print("\n=== CX006: z^2 ===")
    for i in range(len(result.out_idxs)):
        print(result.format_equation(i))

    # Check RMS is very small
    for i, rms in enumerate(result.rms_train):
        print(f"  eq{i} RMS = {rms:.2e}")
        assert rms < 1e-10, f"RMS too large for eq{i}: {rms}"

    print("PASSED: CX006 complex square z^2\n")


# =============================================================================
# Test: CX012 - Complex Product z*w
# =============================================================================

def test_cx012_complex_product():
    """Test discovery of z*w -> [x0*x2-x1*x3, x0*x3+x1*x2]."""
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import Var, Mul

    surrogate = ComplexProductSurrogate()
    dl = make_dataloader(nvars=4)

    # Library: all pairwise products x_i*x_j
    x0, x1, x2, x3 = Var(0), Var(1), Var(2), Var(3)
    lib = [
        x0, x1, x2, x3,
        Mul(x0, x2),  # Re term: +
        Mul(x1, x3),  # Re term: -
        Mul(x0, x3),  # Im term: +
        Mul(x1, x2),  # Im term: +
    ]

    cfg = SystemDESearchConfig(
        order_candidates=(0,),
        include_const=True,
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate=surrogate,
        train_dataloader=dl,
        cfg=cfg,
        library_terms=lib,
    )

    assert result.order == 0, f"Expected order=0, got {result.order}"
    assert len(result.out_idxs) == 2, f"Expected 2 outputs, got {len(result.out_idxs)}"

    print("\n=== CX012: z*w ===")
    for i in range(len(result.out_idxs)):
        print(result.format_equation(i))

    for i, rms in enumerate(result.rms_train):
        print(f"  eq{i} RMS = {rms:.2e}")
        assert rms < 1e-10, f"RMS too large for eq{i}: {rms}"

    print("PASSED: CX012 complex product z*w\n")


# =============================================================================
# Test: CX022 - NLS Cubic |z|^2*z
# =============================================================================

def test_cx022_nls_cubic():
    """Test discovery of |z|^2*z -> [(x0^2+x1^2)*x0, (x0^2+x1^2)*x1]."""
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import Var, Mul, Pow

    surrogate = NLSCubicSurrogate()
    dl = make_dataloader(nvars=2)

    # Library includes cubic terms
    x0, x1 = Var(0), Var(1)
    lib = [
        x0,
        x1,
        Pow(x0, 2),
        Pow(x1, 2),
        Mul(x0, x1),
        Pow(x0, 3),                         # x0^3
        Pow(x1, 3),                         # x1^3
        Mul(x0, Pow(x1, 2)),                # x0*x1^2
        Mul(Pow(x0, 2), x1),                # x0^2*x1
    ]

    cfg = SystemDESearchConfig(
        order_candidates=(0,),
        include_const=True,
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate=surrogate,
        train_dataloader=dl,
        cfg=cfg,
        library_terms=lib,
    )

    assert result.order == 0, f"Expected order=0, got {result.order}"
    assert len(result.out_idxs) == 2, f"Expected 2 outputs, got {len(result.out_idxs)}"

    print("\n=== CX022: |z|^2*z ===")
    for i in range(len(result.out_idxs)):
        print(result.format_equation(i))

    for i, rms in enumerate(result.rms_train):
        print(f"  eq{i} RMS = {rms:.2e}")
        assert rms < 1e-10, f"RMS too large for eq{i}: {rms}"

    print("PASSED: CX022 NLS cubic |z|^2*z\n")


# =============================================================================
# Test: CX005 - Scalar output |z|^2
# =============================================================================

def test_cx005_magnitude_squared():
    """Test discovery of |z|^2 = x0^2 + x1^2 (scalar output)."""
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import Var, Mul, Pow

    surrogate = MagnitudeSquaredSurrogate()
    dl = make_dataloader(nvars=2)

    x0, x1 = Var(0), Var(1)
    lib = [
        x0,
        x1,
        Pow(x0, 2),
        Pow(x1, 2),
        Mul(x0, x1),
    ]

    cfg = SystemDESearchConfig(
        order_candidates=(0,),
        include_const=True,
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate=surrogate,
        train_dataloader=dl,
        cfg=cfg,
        library_terms=lib,
    )

    assert result.order == 0, f"Expected order=0, got {result.order}"
    assert len(result.out_idxs) == 1, f"Expected 1 output, got {len(result.out_idxs)}"

    print("\n=== CX005: |z|^2 ===")
    print(result.format_equation(0))

    rms = result.rms_train[0]
    print(f"  eq0 RMS = {rms:.2e}")
    assert rms < 1e-10, f"RMS too large: {rms}"

    print("PASSED: CX005 magnitude squared |z|^2\n")


# =============================================================================
# Test: Verify complex AST node forward evaluation
# =============================================================================

def test_complex_ast_forward_evaluation():
    """Test that complex AST nodes (RealNode, ImagNode, etc.) evaluate correctly."""
    try:
        from nestynet_sr.sr_core.bridges import (
            ConstNode, AddNode, MulNode, RealNode, ImagNode, AbsNode, ConjNode,
        )
        from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor
    except ImportError:
        pytest.skip("Complex AST nodes not available")

    print("\n=== Complex AST Forward Evaluation ===")

    # Test 1: real(const) where const = 3+4j
    const = ConstNode(3 + 4j)
    real_node = RealNode(const)

    # Need a dummy leaf for ASTCompositeAdaptor
    import nestynet
    from nestynet.adaptors.adaptors import SegmentedAdaptor

    kw = dict(
        model_base_name="G_Model",
        model_scale=0.1,
        dtype=torch.float64,
        device="cpu",
        num_segments=2,
        seg_width=2,
    )
    net = nestynet.nets.NestyNet_Model(Nout_size=1, Nx_size=1, **kw)
    seg = torch.arange(net.base_model.num_segments)
    leaf = SegmentedAdaptor(net, segments=seg)

    # Build AST: real(3+4j) + 0*atom (to have a leaf)
    from nestynet_sr.sr_core.bridges import AtomNode
    a0 = AtomNode("nn", (0,))
    zero_term = MulNode(ConstNode(0.0), a0)
    ast = AddNode(real_node, zero_term)

    composite = ASTCompositeAdaptor(ast, [leaf])
    x = torch.randn(5, 1)
    y = composite.forward(x)

    # Expected: real(3+4j) = 3.0
    expected = torch.full((5, 1), 3.0, dtype=torch.float64)
    assert torch.allclose(y, expected, atol=1e-10), f"real(3+4j) failed: got {y.flatten()}"
    print("  [PASS] real(3+4j) = 3.0")

    # Test 2: imag(const) where const = 3+4j
    imag_node = ImagNode(const)
    ast2 = AddNode(imag_node, zero_term)
    composite2 = ASTCompositeAdaptor(ast2, [leaf])
    y2 = composite2.forward(x)

    expected2 = torch.full((5, 1), 4.0, dtype=torch.float64)
    assert torch.allclose(y2, expected2, atol=1e-10), f"imag(3+4j) failed: got {y2.flatten()}"
    print("  [PASS] imag(3+4j) = 4.0")

    # Test 3: abs(const) where const = 3+4j -> 5.0
    abs_node = AbsNode(const)
    ast3 = AddNode(abs_node, zero_term)
    composite3 = ASTCompositeAdaptor(ast3, [leaf])
    y3 = composite3.forward(x)

    expected3 = torch.full((5, 1), 5.0, dtype=torch.float64)
    assert torch.allclose(y3, expected3, atol=1e-10), f"abs(3+4j) failed: got {y3.flatten()}"
    print("  [PASS] abs(3+4j) = 5.0")

    # Test 4: conj(const) where const = 3+4j -> 3-4j
    conj_node = ConjNode(const)
    # Check real part of conjugate
    real_conj = RealNode(conj_node)
    ast4 = AddNode(real_conj, zero_term)
    composite4 = ASTCompositeAdaptor(ast4, [leaf])
    y4 = composite4.forward(x)

    expected4 = torch.full((5, 1), 3.0, dtype=torch.float64)
    assert torch.allclose(y4, expected4, atol=1e-10), f"real(conj(3+4j)) failed: got {y4.flatten()}"
    print("  [PASS] real(conj(3+4j)) = 3.0")

    # Check imag part of conjugate
    imag_conj = ImagNode(conj_node)
    ast5 = AddNode(imag_conj, zero_term)
    composite5 = ASTCompositeAdaptor(ast5, [leaf])
    y5 = composite5.forward(x)

    expected5 = torch.full((5, 1), -4.0, dtype=torch.float64)
    assert torch.allclose(y5, expected5, atol=1e-10), f"imag(conj(3+4j)) failed: got {y5.flatten()}"
    print("  [PASS] imag(conj(3+4j)) = -4.0")

    print("PASSED: Complex AST forward evaluation\n")


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Complex SR Hello World Tests")
    print("=" * 70)

    # Run complex AST forward evaluation test
    test_complex_ast_forward_evaluation()

    # Run system DE discovery tests
    test_cx005_magnitude_squared()
    test_cx006_complex_square()
    test_cx012_complex_product()
    test_cx022_nls_cubic()

    print("=" * 70)
    print("All complex SR hello-world tests passed!")
    print("=" * 70)
