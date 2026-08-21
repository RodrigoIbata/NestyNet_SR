# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Tests for complex number AST nodes: Conj, Real, Imag, Abs, Arg.

Sections:
1. AST Structure Tests (no neural nets)
2. Forward Evaluation Tests (using ConstNode with complex values)
3. JVP/VJP Adjoint Symmetry Tests (bilinear identity)
"""

import torch
import pytest

torch.set_default_dtype(torch.float64)

try:
    import nestynet
    from nestynet.adaptors.adaptors import SegmentedAdaptor
except Exception:
    pytest.skip("nestynet not importable", allow_module_level=True)

try:
    from symbolic_regression_DE.adaptors.ast_composite import ASTCompositeAdaptor
    from symbolic_regression_DE.sr_core.bridges import (
        AtomNode, AddNode, MulNode, ConstNode,
        ConjNode, RealNode, ImagNode, AbsNode, ArgNode,
        ast_to_human_readable,
    )
except Exception:
    from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor
    from nestynet_sr.sr_core.bridges import (
        AtomNode, AddNode, MulNode, ConstNode,
        ConjNode, RealNode, ImagNode, AbsNode, ArgNode,
        ast_to_human_readable,
    )


# ═══════════════════════════════════════════════════════════════════
# Section 1: AST Structure Tests
# ═══════════════════════════════════════════════════════════════════

class TestASTStructure:
    """Tests for node construction and string representation."""

    def test_conj_node_repr(self):
        """ConjNode.__repr__ produces expected string."""
        a0 = AtomNode("nn", (0,))
        node = ConjNode(a0)
        assert repr(node) == "conj(nn(x0))"

    def test_real_node_repr(self):
        """RealNode.__repr__ produces expected string."""
        a0 = AtomNode("nn", (0,))
        node = RealNode(a0)
        assert repr(node) == "real(nn(x0))"

    def test_imag_node_repr(self):
        """ImagNode.__repr__ produces expected string."""
        a0 = AtomNode("nn", (0,))
        node = ImagNode(a0)
        assert repr(node) == "imag(nn(x0))"

    def test_abs_node_repr(self):
        """AbsNode.__repr__ produces expected string."""
        a0 = AtomNode("nn", (0,))
        node = AbsNode(a0)
        assert repr(node) == "abs(nn(x0))"

    def test_arg_node_repr(self):
        """ArgNode.__repr__ produces expected string."""
        a0 = AtomNode("nn", (0,))
        node = ArgNode(a0)
        assert repr(node) == "arg(nn(x0))"

    def test_ast_to_human_readable_conj(self):
        """ast_to_human_readable for ConjNode."""
        a0 = AtomNode("nn", (0,))
        node = ConjNode(a0)
        assert ast_to_human_readable(node) == "conj(NN[x0])"

    def test_ast_to_human_readable_real(self):
        """ast_to_human_readable for RealNode."""
        a0 = AtomNode("nn", (0,))
        node = RealNode(a0)
        assert ast_to_human_readable(node) == "real(NN[x0])"

    def test_ast_to_human_readable_imag(self):
        """ast_to_human_readable for ImagNode."""
        a0 = AtomNode("nn", (0,))
        node = ImagNode(a0)
        assert ast_to_human_readable(node) == "imag(NN[x0])"

    def test_ast_to_human_readable_abs(self):
        """ast_to_human_readable for AbsNode."""
        a0 = AtomNode("nn", (0,))
        node = AbsNode(a0)
        assert ast_to_human_readable(node) == "abs(NN[x0])"

    def test_ast_to_human_readable_arg(self):
        """ast_to_human_readable for ArgNode."""
        a0 = AtomNode("nn", (0,))
        node = ArgNode(a0)
        assert ast_to_human_readable(node) == "arg(NN[x0])"

    def test_ast_to_human_readable_nested(self):
        """ast_to_human_readable for nested complex nodes."""
        a0 = AtomNode("nn", (0,))
        inner = ConjNode(a0)
        outer = RealNode(inner)
        assert ast_to_human_readable(outer) == "real(conj(NN[x0]))"


# ═══════════════════════════════════════════════════════════════════
# Section 2: Forward Evaluation Tests
# ═══════════════════════════════════════════════════════════════════

def _mk_leaf(Nx_leaf, S, *, seg_width=2, device="cpu"):
    """Create a SegmentedAdaptor leaf for testing."""
    kw = dict(
        model_base_name="G_Model",
        model_scale=0.1,
        dtype=torch.float64,
        device=device,
        num_segments=S,
        seg_width=seg_width,
    )
    net = nestynet.nets.NestyNet_Model(Nout_size=1, Nx_size=Nx_leaf, **kw)
    seg = torch.arange(net.base_model.num_segments, device=device)
    return SegmentedAdaptor(net, segments=seg)


class TestForwardEvaluation:
    """Tests for forward evaluation of complex nodes with neural network leaves."""

    def test_conj_with_const(self):
        """conj(NN + const) where const=1+2j produces expected conjugate."""
        # Build AST: conj(a0 + (1+2j))
        a0 = AtomNode("nn", (0,))
        const = ConstNode(1 + 2j)
        sum_node = AddNode(a0, const)
        node = ConjNode(sum_node)

        leaf0 = _mk_leaf(1, S=2)
        composite = ASTCompositeAdaptor(node, [leaf0])

        x = 0.3 * torch.randn(5, 1)
        y = composite.forward(x)

        # Expected: conj(leaf0(x) + (1+2j))
        with torch.no_grad():
            y0 = leaf0.forward(x)
        expected = torch.conj(y0 + (1 + 2j))
        assert torch.allclose(y, expected, atol=1e-10)

    def test_real_with_const(self):
        """real(NN + const) where const=3+4j extracts real part."""
        # Build AST: real(a0 + (3+4j))
        a0 = AtomNode("nn", (0,))
        const = ConstNode(3 + 4j)
        sum_node = AddNode(a0, const)
        node = RealNode(sum_node)

        leaf0 = _mk_leaf(1, S=2)
        composite = ASTCompositeAdaptor(node, [leaf0])

        x = 0.3 * torch.randn(5, 1)
        y = composite.forward(x)

        # Expected: real(leaf0(x) + (3+4j))
        with torch.no_grad():
            y0 = leaf0.forward(x)
        expected = torch.real(y0 + (3 + 4j))
        assert torch.allclose(y, expected, atol=1e-10)

    def test_imag_with_const(self):
        """imag(NN + const) where const=3+4j extracts imaginary part."""
        # Build AST: imag(a0 + (3+4j))
        a0 = AtomNode("nn", (0,))
        const = ConstNode(3 + 4j)
        sum_node = AddNode(a0, const)
        node = ImagNode(sum_node)

        leaf0 = _mk_leaf(1, S=2)
        composite = ASTCompositeAdaptor(node, [leaf0])

        x = 0.3 * torch.randn(5, 1)
        y = composite.forward(x)

        # Expected: imag(leaf0(x) + (3+4j))
        with torch.no_grad():
            y0 = leaf0.forward(x)
        expected = torch.imag(y0 + (3 + 4j))
        assert torch.allclose(y, expected, atol=1e-10)

    def test_abs_with_const(self):
        """abs(NN + const) where const=3+4j computes modulus."""
        # Build AST: abs(a0 + (3+4j))
        a0 = AtomNode("nn", (0,))
        const = ConstNode(3 + 4j)
        sum_node = AddNode(a0, const)
        node = AbsNode(sum_node)

        leaf0 = _mk_leaf(1, S=2)
        composite = ASTCompositeAdaptor(node, [leaf0])

        x = 0.3 * torch.randn(5, 1)
        y = composite.forward(x)

        # Expected: abs(leaf0(x) + (3+4j))
        with torch.no_grad():
            y0 = leaf0.forward(x)
        expected = torch.abs(y0 + (3 + 4j))
        assert torch.allclose(y, expected, atol=1e-10)

    def test_arg_with_const(self):
        """arg(NN + const) where const=1+1j computes phase angle."""
        # Build AST: arg(a0 + (1+1j))
        a0 = AtomNode("nn", (0,))
        const = ConstNode(1 + 1j)
        sum_node = AddNode(a0, const)
        node = ArgNode(sum_node)

        leaf0 = _mk_leaf(1, S=2)
        composite = ASTCompositeAdaptor(node, [leaf0])

        # Use small x so NN output doesn't dominate the 1+1j constant
        x = 0.01 * torch.randn(5, 1)
        y = composite.forward(x)

        # Expected: arg(leaf0(x) + (1+1j))
        with torch.no_grad():
            y0 = leaf0.forward(x)
        expected = torch.angle(y0 + (1 + 1j))
        assert torch.allclose(y, expected, atol=1e-10)

    def test_real_with_two_nn_leaves(self):
        """real(NN[x0] + 1j*NN[x1]) extracts real part correctly."""
        # Build AST: real(a0 + 1j*a1)
        a0 = AtomNode("nn", (0,))
        a1 = AtomNode("nn", (1,))
        imag_part = MulNode(ConstNode(1j), a1)
        complex_sum = AddNode(a0, imag_part)
        node = RealNode(complex_sum)

        # Create leaf networks
        leaf0 = _mk_leaf(1, S=2)
        leaf1 = _mk_leaf(1, S=2)
        composite = ASTCompositeAdaptor(node, [leaf0, leaf1])

        # Test forward pass
        x = 0.3 * torch.randn(5, 2)
        y = composite.forward(x)

        # Compute expected: real(leaf0(x[:,0]) + 1j*leaf1(x[:,1]))
        with torch.no_grad():
            y0 = leaf0.forward(x[:, 0:1])
            y1 = leaf1.forward(x[:, 1:2])
        expected = torch.real(y0 + 1j * y1)
        assert torch.allclose(y, expected, atol=1e-10)

    def test_imag_with_two_nn_leaves(self):
        """imag(NN[x0] + 1j*NN[x1]) extracts imaginary part correctly."""
        # Build AST: imag(a0 + 1j*a1)
        a0 = AtomNode("nn", (0,))
        a1 = AtomNode("nn", (1,))
        imag_part = MulNode(ConstNode(1j), a1)
        complex_sum = AddNode(a0, imag_part)
        node = ImagNode(complex_sum)

        # Create leaf networks
        leaf0 = _mk_leaf(1, S=2)
        leaf1 = _mk_leaf(1, S=2)
        composite = ASTCompositeAdaptor(node, [leaf0, leaf1])

        # Test forward pass
        x = 0.3 * torch.randn(5, 2)
        y = composite.forward(x)

        # Compute expected: imag(leaf0(x[:,0]) + 1j*leaf1(x[:,1]))
        with torch.no_grad():
            y0 = leaf0.forward(x[:, 0:1])
            y1 = leaf1.forward(x[:, 1:2])
        expected = torch.imag(y0 + 1j * y1)
        assert torch.allclose(y, expected, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════
# Section 3: JVP/VJP Adjoint Symmetry Tests
# ═══════════════════════════════════════════════════════════════════

def _orthonormal_columns(n, m, dtype, device):
    """Generate orthonormal column vectors."""
    X = torch.randn(n, m, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q[:, :m]


def _canon_Y(Y, B):
    """Canonicalize output to shape (B,)."""
    if Y.ndim == 2 and Y.size(1) == 1:
        return Y.reshape(B)
    if Y.ndim == 1:
        return Y
    raise ValueError(f"Unexpected Y shape {tuple(Y.shape)}")


def _canon_A(A, B):
    """Canonicalize adjoint to shape (B,)."""
    if A.ndim == 2 and A.size(1) == 1:
        return A.reshape(B)
    if A.ndim == 1:
        return A
    raise ValueError(f"Unexpected A shape {tuple(A.shape)}")


def _canon_g(g, P):
    """Canonicalize gradient to shape (P,)."""
    if g.ndim == 2 and g.size(0) == 1 and g.size(1) == P:
        return g.reshape(P)
    if g.ndim == 1 and g.numel() == P:
        return g
    return g.reshape(-1)


def _test_adjoint_identity(model, B, Nx_global, seed=0):
    """
    Test the bilinear adjoint identity: <A, J v> == <J^T A, v>.

    Parameters
    ----------
    model : ASTCompositeAdaptor
        The composite model to test.
    B : int
        Batch size.
    Nx_global : int
        Number of input dimensions.
    seed : int
        Random seed.
    """
    torch.manual_seed(seed)
    device = "cpu"

    x = 0.3 * torch.randn(B, Nx_global, device=device)
    with torch.no_grad():
        y = model.forward(x).detach()
    cache = model.build_cache((x, y))

    P = model.num_parameters()
    if P == 0:
        return  # No parameters to test

    mA = min(B, 8)
    mV = min(P, 8)

    # Orthonormal A_i in output space (B,)
    QA = _orthonormal_columns(B, mA, x.dtype, device)
    A_list = [QA[:, i].clone().reshape(B) for i in range(mA)]

    # Orthonormal v_j in parameter space (P,)
    QP = _orthonormal_columns(P, mV, x.dtype, device)
    v_list = [QP[:, j].clone().reshape(P) for j in range(mV)]

    # Precompute J v_j
    Y_list = [_canon_Y(model.jvp(cache, vj, out_dim=None), B) for vj in v_list]

    L = torch.zeros(mA, mV, dtype=x.dtype)
    R = torch.zeros_like(L)

    for i, A in enumerate(A_list):
        A = _canon_A(A, B)
        g = _canon_g(model.vjp(cache, A, out_dim=None), P)
        for j, (vj, Yj) in enumerate(zip(v_list, Y_list)):
            L[i, j] = (A * Yj).sum()
            R[i, j] = (g * vj).sum()

    diff = L - R
    tol = 1e-12 * (1.0 + float(L.norm()))
    assert diff.norm() <= tol, f"Adjoint mismatch: ||L-R||={float(diff.norm())}, ||L||={float(L.norm())}"


class TestJVPVJPAdjoint:
    """JVP/VJP adjoint symmetry tests for complex nodes.

    Note: Complex operations like real(a0 + 1j*a1) propagate complex adjoints
    through the imaginary branch (1j*a1), which real-valued NestyNet leaves
    cannot handle. Therefore, we test complex nodes with purely real inputs,
    which is still a valid test of the adjoint symmetry for those code paths.
    """

    @pytest.mark.parametrize("S", [1, 2])
    @pytest.mark.parametrize("B", [1, 3])
    def test_real_node_adjoint(self, S, B):
        """Adjoint identity for RealNode with real input: F = real(a0) + a1.

        For real input, real(x) = x, so the adjoint passes through unchanged.
        """
        a0 = AtomNode("nn", (0,))
        a1 = AtomNode("nn", (1,))
        real_node = RealNode(a0)
        ast = AddNode(real_node, a1)

        leaves = [
            _mk_leaf(1, S),  # x0
            _mk_leaf(1, S),  # x1
        ]
        model = ASTCompositeAdaptor(ast, leaves)
        _test_adjoint_identity(model, B, Nx_global=2)

    @pytest.mark.parametrize("S", [1, 2])
    @pytest.mark.parametrize("B", [1, 3])
    def test_abs_node_adjoint_real_input(self, S, B):
        """Adjoint identity for AbsNode with real input: F = abs(a0) + a1.

        Note: When input is real, abs(x) = |x| and d|x|/dx = sign(x).
        The adjoint stays real, so this works with real-valued leaves.
        """
        a0 = AtomNode("nn", (0,))
        a1 = AtomNode("nn", (1,))
        abs_node = AbsNode(a0)
        ast = AddNode(abs_node, a1)

        leaves = [
            _mk_leaf(1, S),  # x0
            _mk_leaf(1, S),  # x1
        ]
        model = ASTCompositeAdaptor(ast, leaves)
        _test_adjoint_identity(model, B, Nx_global=2)

    @pytest.mark.parametrize("S", [1, 2])
    @pytest.mark.parametrize("B", [1, 3])
    def test_conj_real_input_adjoint(self, S, B):
        """Adjoint identity for ConjNode with real input: F = conj(a0) + a1.

        Note: conj(x) = x for real x, so this tests the identity case.
        The adjoint is just passed through (with conjugation, but x is real).
        """
        a0 = AtomNode("nn", (0,))
        a1 = AtomNode("nn", (1,))
        conj_node = ConjNode(a0)
        ast = AddNode(conj_node, a1)

        leaves = [
            _mk_leaf(1, S),  # x0
            _mk_leaf(1, S),  # x1
        ]
        model = ASTCompositeAdaptor(ast, leaves)
        _test_adjoint_identity(model, B, Nx_global=2)



# ═══════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running test_complex_nodes.py...")

    # Section 1: AST Structure Tests
    print("\n=== Section 1: AST Structure Tests ===")
    ts = TestASTStructure()
    ts.test_conj_node_repr()
    print("  [PASS] test_conj_node_repr")
    ts.test_real_node_repr()
    print("  [PASS] test_real_node_repr")
    ts.test_imag_node_repr()
    print("  [PASS] test_imag_node_repr")
    ts.test_abs_node_repr()
    print("  [PASS] test_abs_node_repr")
    ts.test_arg_node_repr()
    print("  [PASS] test_arg_node_repr")
    ts.test_ast_to_human_readable_conj()
    print("  [PASS] test_ast_to_human_readable_conj")
    ts.test_ast_to_human_readable_real()
    print("  [PASS] test_ast_to_human_readable_real")
    ts.test_ast_to_human_readable_imag()
    print("  [PASS] test_ast_to_human_readable_imag")
    ts.test_ast_to_human_readable_abs()
    print("  [PASS] test_ast_to_human_readable_abs")
    ts.test_ast_to_human_readable_arg()
    print("  [PASS] test_ast_to_human_readable_arg")
    ts.test_ast_to_human_readable_nested()
    print("  [PASS] test_ast_to_human_readable_nested")

    # Section 2: Forward Evaluation Tests
    print("\n=== Section 2: Forward Evaluation Tests ===")
    tf = TestForwardEvaluation()
    tf.test_conj_with_const()
    print("  [PASS] test_conj_with_const")
    tf.test_real_with_const()
    print("  [PASS] test_real_with_const")
    tf.test_imag_with_const()
    print("  [PASS] test_imag_with_const")
    tf.test_abs_with_const()
    print("  [PASS] test_abs_with_const")
    tf.test_arg_with_const()
    print("  [PASS] test_arg_with_const")
    tf.test_real_with_two_nn_leaves()
    print("  [PASS] test_real_with_two_nn_leaves")
    tf.test_imag_with_two_nn_leaves()
    print("  [PASS] test_imag_with_two_nn_leaves")

    # Section 3: JVP/VJP Adjoint Symmetry Tests
    print("\n=== Section 3: JVP/VJP Adjoint Symmetry Tests ===")
    ta = TestJVPVJPAdjoint()
    for S in [1, 2]:
        for B in [1, 3]:
            ta.test_real_node_adjoint(S, B)
            print(f"  [PASS] test_real_node_adjoint(S={S}, B={B})")
            ta.test_abs_node_adjoint_real_input(S, B)
            print(f"  [PASS] test_abs_node_adjoint_real_input(S={S}, B={B})")
            ta.test_conj_real_input_adjoint(S, B)
            print(f"  [PASS] test_conj_real_input_adjoint(S={S}, B={B})")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
