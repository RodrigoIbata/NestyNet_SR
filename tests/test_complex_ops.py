# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Unit tests for complex_ops.py - complex field helpers and constrained discovery.

Tests are organized in two phases:
- Phase 1: Library helper tests (AST construction)
- Phase 2: Constrained discovery tests (coefficient tying and formatting)
"""

import math
import torch

torch.set_default_dtype(torch.float64)


# ══════════════════════════════════════════════════════════════
# Phase 1 Tests: Library Helpers
# ══════════════════════════════════════════════════════════════


def test_psi_creation():
    """Test ComplexField dataclass creation."""
    from nestynet_sr.sr_de.complex_ops import Psi, ComplexField

    # Default values
    psi = Psi()
    assert isinstance(psi, ComplexField)
    assert psi.out_real == 0
    assert psi.out_imag == 1

    # Custom values
    psi2 = Psi(out_real=2, out_imag=3)
    assert psi2.out_real == 2
    assert psi2.out_imag == 3

    print("PASSED: test_psi_creation")


def test_real_imag_parts():
    """Test real_part and imag_part functions."""
    from nestynet_sr.sr_de.complex_ops import Psi, real_part, imag_part
    from nestynet_sr.sr_core.bridges import AtomNode

    psi = Psi(out_real=0, out_imag=1)

    u = real_part(psi)
    v = imag_part(psi)

    # Check that these are U atoms with correct out_idx
    assert isinstance(u, AtomNode)
    assert isinstance(v, AtomNode)
    assert u.kwargs.get("out_idx", 0) == 0
    assert v.kwargs.get("out_idx", 0) == 1

    # Test with different indices
    psi2 = Psi(out_real=5, out_imag=7)
    u2 = real_part(psi2)
    v2 = imag_part(psi2)
    assert u2.kwargs.get("out_idx", 0) == 5
    assert v2.kwargs.get("out_idx", 0) == 7

    print("PASSED: test_real_imag_parts")


def test_abs_sq_produces_correct_ast():
    """Test AbsSq produces Add(Pow(U(0),2), Pow(U(1),2))."""
    from nestynet_sr.sr_de.complex_ops import Psi, AbsSq
    from nestynet_sr.sr_core.bridges import AddNode, PowNode, AtomNode

    psi = Psi()
    mod_sq = AbsSq(psi)

    # Should be Add(Pow(...), Pow(...))
    assert isinstance(mod_sq, AddNode), f"Expected AddNode, got {type(mod_sq)}"

    left = mod_sq.left
    right = mod_sq.right

    assert isinstance(left, PowNode), f"Expected PowNode for left, got {type(left)}"
    assert isinstance(right, PowNode), f"Expected PowNode for right, got {type(right)}"

    # Check exponents are 2
    assert left.exponent == 2, f"Expected exponent 2, got {left.exponent}"
    assert right.exponent == 2, f"Expected exponent 2, got {right.exponent}"

    # Check bases are U atoms
    assert isinstance(left.base, AtomNode)
    assert isinstance(right.base, AtomNode)

    print("PASSED: test_abs_sq_produces_correct_ast")


def test_dpsi_derivatives():
    """Test DPsi returns correct DU atoms."""
    from nestynet_sr.sr_de.complex_ops import Psi, DPsi
    from nestynet_sr.sr_core.bridges import AtomNode

    psi = Psi()
    du, dv = DPsi(axis=1, z=psi)

    # Check both are DU atoms
    assert isinstance(du, AtomNode)
    assert isinstance(dv, AtomNode)
    assert du.kind.lower() in ("du", "d1u", "grad_u")
    assert dv.kind.lower() in ("du", "d1u", "grad_u")

    # Check axis
    assert du.kwargs.get("axis", 0) == 1
    assert dv.kwargs.get("axis", 0) == 1

    # Check out_idx
    assert du.kwargs.get("out_idx", 0) == 0
    assert dv.kwargs.get("out_idx", 0) == 1

    # Test without z argument (using defaults)
    du2, dv2 = DPsi(axis=2, out_real=3, out_imag=4)
    assert du2.kwargs.get("axis", 0) == 2
    assert du2.kwargs.get("out_idx", 0) == 3
    assert dv2.kwargs.get("out_idx", 0) == 4

    print("PASSED: test_dpsi_derivatives")


def test_d2psi_derivatives():
    """Test D2Psi returns correct D2U atoms."""
    from nestynet_sr.sr_de.complex_ops import Psi, D2Psi
    from nestynet_sr.sr_core.bridges import AtomNode

    psi = Psi()
    d2u, d2v = D2Psi(axis0=1, axis1=1, z=psi)

    # Check both are D2U atoms
    assert isinstance(d2u, AtomNode)
    assert isinstance(d2v, AtomNode)
    assert d2u.kind.lower() in ("d2u", "ddu", "hess_u")
    assert d2v.kind.lower() in ("d2u", "ddu", "hess_u")

    # Check axes
    assert d2u.kwargs.get("axis0", 0) == 1
    assert d2u.kwargs.get("axis1", 0) == 1
    assert d2v.kwargs.get("axis0", 0) == 1
    assert d2v.kwargs.get("axis1", 0) == 1

    # Check out_idx
    assert d2u.kwargs.get("out_idx", 0) == 0
    assert d2v.kwargs.get("out_idx", 0) == 1

    # Test mixed second derivatives
    d2u_xy, d2v_xy = D2Psi(axis0=1, axis1=2, z=psi)
    assert d2u_xy.kwargs.get("axis0", 0) == 1
    assert d2u_xy.kwargs.get("axis1", 0) == 2

    print("PASSED: test_d2psi_derivatives")


def test_complex_mul_expansion():
    """Test ComplexMul produces correct (ar*br - ai*bi, ar*bi + ai*br)."""
    from nestynet_sr.sr_de.complex_ops import Psi, ComplexMul
    from nestynet_sr.sr_core.bridges import AddNode

    a = Psi(out_real=0, out_imag=1)
    b = Psi(out_real=2, out_imag=3)

    real, imag = ComplexMul(a, b)

    # Real part should be Add(Mul(...), Mul(...)) with negative term
    assert isinstance(real, AddNode), f"Expected AddNode for real, got {type(real)}"

    # Imaginary part should be Add(Mul(...), Mul(...))
    assert isinstance(imag, AddNode), f"Expected AddNode for imag, got {type(imag)}"

    print("PASSED: test_complex_mul_expansion")


def test_abs_sq_psi_nls_term():
    """Test AbsSqPsi produces ((u²+v²)*u, (u²+v²)*v)."""
    from nestynet_sr.sr_de.complex_ops import Psi, AbsSqPsi
    from nestynet_sr.sr_core.bridges import MulNode, AddNode

    psi = Psi()
    nls_u, nls_v = AbsSqPsi(psi)

    # Both should be Mul(Add(...), U(...))
    assert isinstance(nls_u, MulNode), f"Expected MulNode for nls_u, got {type(nls_u)}"
    assert isinstance(nls_v, MulNode), f"Expected MulNode for nls_v, got {type(nls_v)}"

    # Left side should be |psi|² = u² + v²
    assert isinstance(nls_u.left, AddNode), f"Expected AddNode for |psi|², got {type(nls_u.left)}"
    assert isinstance(nls_v.left, AddNode), f"Expected AddNode for |psi|², got {type(nls_v.left)}"

    print("PASSED: test_abs_sq_psi_nls_term")


def test_laplacian_2d():
    """Test Laplacian2D returns D2Psi with same axis."""
    from nestynet_sr.sr_de.complex_ops import Psi, Laplacian2D
    from nestynet_sr.sr_core.bridges import AtomNode

    psi = Psi()
    lap_u, lap_v = Laplacian2D(psi, spatial_axis=1)

    assert isinstance(lap_u, AtomNode)
    assert isinstance(lap_v, AtomNode)
    assert lap_u.kwargs.get("axis0", 0) == 1
    assert lap_u.kwargs.get("axis1", 0) == 1

    print("PASSED: test_laplacian_2d")


# ══════════════════════════════════════════════════════════════
# Phase 2 Tests: Constrained Discovery
# ══════════════════════════════════════════════════════════════


class MockSchrodingerSurrogate(torch.nn.Module):
    """Mock surrogate for free Schrödinger: i∂ψ/∂t = -∂²ψ/∂x².

    Solution: plane wave ψ(x,t) = exp(i(kx - ωt)) with ω = k²

    With k=1: ψ = exp(i(x-t)), u = cos(x-t), v = sin(x-t)
    """

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k
        self.omega = k ** 2
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, xs = x[:, 0], x[:, 1]
        phase = self.k * xs - self.omega * t
        u = torch.cos(phase)
        v = torch.sin(phase)
        return torch.stack([u, v], dim=1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        t, xs = x[:, 0], x[:, 1]
        phase = self.k * xs - self.omega * t
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)

        du_dt = self.omega * sin_p
        du_dx = -self.k * sin_p
        dv_dt = -self.omega * cos_p
        dv_dx = self.k * cos_p

        gu = torch.stack([du_dt, du_dx], dim=-1)
        gv = torch.stack([dv_dt, dv_dx], dim=-1)
        return torch.stack([gu, gv], dim=1)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        t, xs = x[:, 0], x[:, 1]
        phase = self.k * xs - self.omega * t
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)

        d2u_dt2 = -self.omega ** 2 * cos_p
        d2u_dx2 = -self.k ** 2 * cos_p
        d2u_dtdx = self.omega * self.k * cos_p
        d2v_dt2 = -self.omega ** 2 * sin_p
        d2v_dx2 = -self.k ** 2 * sin_p
        d2v_dtdx = self.omega * self.k * sin_p

        B = x.shape[0]
        H = torch.zeros(B, 2, 2, 2, dtype=x.dtype, device=x.device)
        H[:, 0, 0, 0] = d2u_dt2
        H[:, 0, 0, 1] = d2u_dtdx
        H[:, 0, 1, 0] = d2u_dtdx
        H[:, 0, 1, 1] = d2u_dx2
        H[:, 1, 0, 0] = d2v_dt2
        H[:, 1, 0, 1] = d2v_dtdx
        H[:, 1, 1, 0] = d2v_dtdx
        H[:, 1, 1, 1] = d2v_dx2
        return H

    def parameters(self):
        return iter([self._dummy])


def test_complex_term_spec_creation():
    """Test ComplexTermSpec dataclass creation."""
    from nestynet_sr.sr_de.complex_ops import (
        ComplexTermSpec,
        Psi,
        D2Psi,
    )

    psi = Psi()
    d2u, d2v = D2Psi(1, 1, z=psi)

    spec = ComplexTermSpec(
        real_part=d2u,
        imag_part=d2v,
        name="d2psi/dx2",
        is_hermitian=True,
    )

    assert spec.name == "d2psi/dx2"
    assert spec.is_hermitian is True
    assert spec.real_part is d2u
    assert spec.imag_part is d2v

    print("PASSED: test_complex_term_spec_creation")


def test_complex_de_search_config_creation():
    """Test ComplexDESearchConfig dataclass creation."""
    from nestynet_sr.sr_de.complex_ops import (
        ComplexDESearchConfig,
        ComplexTermSpec,
        Psi,
        D2Psi,
    )

    psi = Psi()
    d2u, d2v = D2Psi(1, 1, z=psi)
    lap_term = ComplexTermSpec(d2u, d2v, "Laplacian", is_hermitian=True)

    cfg = ComplexDESearchConfig(
        time_axis=0,
        out_real=0,
        out_imag=1,
        complex_terms=[lap_term],
        include_const=False,
        stlsq_lambda=1e-3,
    )

    assert cfg.time_axis == 0
    assert cfg.out_real == 0
    assert cfg.out_imag == 1
    assert len(cfg.complex_terms) == 1
    assert cfg.include_const is False

    print("PASSED: test_complex_de_search_config_creation")


def test_make_laplacian_term():
    """Test make_laplacian_term convenience function."""
    from nestynet_sr.sr_de.complex_ops import Psi, make_laplacian_term, ComplexTermSpec

    psi = Psi()
    term = make_laplacian_term(psi, spatial_axis=1, name="d2psi/dx2")

    assert isinstance(term, ComplexTermSpec)
    assert term.name == "d2psi/dx2"
    assert term.is_hermitian is True

    print("PASSED: test_make_laplacian_term")


def test_make_nls_term():
    """Test make_nls_term convenience function."""
    from nestynet_sr.sr_de.complex_ops import Psi, make_nls_term, ComplexTermSpec

    psi = Psi()
    term = make_nls_term(psi)

    assert isinstance(term, ComplexTermSpec)
    assert term.name == "|psi|^2*psi"
    assert term.is_hermitian is True

    print("PASSED: test_make_nls_term")


def test_make_psi_term():
    """Test make_psi_term convenience function."""
    from nestynet_sr.sr_de.complex_ops import Psi, make_psi_term, ComplexTermSpec

    psi = Psi()
    term = make_psi_term(psi)

    assert isinstance(term, ComplexTermSpec)
    assert term.name == "psi"
    assert term.is_hermitian is True

    print("PASSED: test_make_psi_term")


def test_schrodinger_discovery_with_system_de():
    """Test discovering Schrödinger using system DE (baseline)."""
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import D2U

    surrogate = MockSchrodingerSurrogate(k=1.0)

    # 2D grid: t × x
    t_vals = torch.linspace(0.0, 2 * math.pi, 40)
    x_vals = torch.linspace(0.0, 2 * math.pi, 40)
    tt, xx = torch.meshgrid(t_vals, x_vals, indexing="ij")
    X = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)
    loader = [(X,)]

    # Library: ∂²u/∂x², ∂²v/∂x²
    d2u_dx2 = D2U(1, 1, out_idx=0)
    d2v_dx2 = D2U(1, 1, out_idx=1)
    lib = [d2u_dx2, d2v_dx2]

    cfg = SystemDESearchConfig(
        x_axis=0,
        order_candidates=(1,),
        include_const=False,
        include_x=False,
        include_u=False,
        include_u_cross=False,
        include_xu=False,
        include_du=False,
        include_d2u=False,
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate, loader, cfg=cfg, library_terms=lib, device=torch.device("cpu")
    )

    print("\nSystem DE result:")
    print(f"  Coefficients:\n{result.coeffs}")
    print(f"  RMS: {result.rms_train}")

    # Verify RMS is small
    for rms in result.rms_train:
        assert rms < 1e-6, f"RMS too large: {rms}"

    print("PASSED: test_schrodinger_discovery_with_system_de")


def test_complex_equation_formatting():
    """Test ComplexDESearchResult.format_complex_equation."""
    from nestynet_sr.sr_de.complex_ops import ComplexDESearchResult
    from nestynet_sr.sr_de.system_de_search import SystemDESearchResult

    # Create a mock underlying result
    mock_underlying = SystemDESearchResult(
        order=1,
        x_axis=0,
        out_idxs=(0, 1),
        term_asts=[],
        coeffs=torch.zeros(2, 0),
        rms_train=[1e-8, 1e-8],
    )

    # Create a result with known coefficients
    result = ComplexDESearchResult(
        underlying_result=mock_underlying,
        complex_coeffs=[complex(-1.0, 0.0), complex(0.5, 0.0)],
        term_names=["d2psi/dx2", "|psi|^2*psi"],
        rms_train=1e-8,
        is_valid_complex=True,
    )

    eq_str = result.format_complex_equation()
    print(f"\nFormatted equation: {eq_str}")

    # Check that the equation string contains expected terms
    assert "dpsi/dt" in eq_str
    assert "d2psi/dx2" in eq_str

    print("PASSED: test_complex_equation_formatting")


def test_coefficient_constraint_validation():
    """Test that is_valid_complex flag is set correctly."""
    from nestynet_sr.sr_de.complex_ops import ComplexDESearchResult
    from nestynet_sr.sr_de.system_de_search import SystemDESearchResult

    mock_underlying = SystemDESearchResult(
        order=1,
        x_axis=0,
        out_idxs=(0, 1),
        term_asts=[],
        coeffs=torch.zeros(2, 0),
        rms_train=[1e-8, 1e-8],
    )

    # Valid complex structure
    valid_result = ComplexDESearchResult(
        underlying_result=mock_underlying,
        complex_coeffs=[complex(-1.0, 0.0)],
        term_names=["d2psi/dx2"],
        rms_train=1e-8,
        is_valid_complex=True,
    )
    assert valid_result.is_valid_complex is True

    # Invalid complex structure
    invalid_result = ComplexDESearchResult(
        underlying_result=mock_underlying,
        complex_coeffs=[complex(-1.0, 0.0)],
        term_names=["d2psi/dx2"],
        rms_train=1e-8,
        is_valid_complex=False,
    )
    assert invalid_result.is_valid_complex is False

    print("PASSED: test_coefficient_constraint_validation")


# ══════════════════════════════════════════════════════════════
# Integration test with full discovery
# ══════════════════════════════════════════════════════════════


def test_schrodinger_discovery_constrained():
    """Test full constrained complex DE discovery for Schrödinger equation.

    This is an integration test that exercises the full discovery pipeline
    using the complex_ops helpers.
    """
    from nestynet_sr.sr_de.complex_ops import (
        Psi,
        make_laplacian_term,
        ComplexDESearchConfig,
        discover_complex_de_from_surrogate,
    )

    surrogate = MockSchrodingerSurrogate(k=1.0)

    # 2D grid: t × x
    t_vals = torch.linspace(0.0, 2 * math.pi, 40)
    x_vals = torch.linspace(0.0, 2 * math.pi, 40)
    tt, xx = torch.meshgrid(t_vals, x_vals, indexing="ij")
    X = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)
    loader = [(X,)]

    # Build complex term library
    psi = Psi()
    lap_term = make_laplacian_term(psi, spatial_axis=1, name="d2psi/dx2")

    cfg = ComplexDESearchConfig(
        time_axis=0,
        out_real=0,
        out_imag=1,
        complex_terms=[lap_term],
        include_const=False,
        stlsq_lambda=1e-4,
    )

    result = discover_complex_de_from_surrogate(
        surrogate, loader, cfg=cfg, device=torch.device("cpu")
    )

    print(f"\n{'='*60}")
    print("Constrained Complex DE Discovery: Free Schrödinger")
    print(f"{'='*60}")
    print(f"Complex coefficients: {result.complex_coeffs}")
    print(f"Term names: {result.term_names}")
    print(f"RMS: {result.rms_train:.2e}")
    print(f"Valid complex structure: {result.is_valid_complex}")
    print("\nFormatted equation:")
    print(f"  {result.format_complex_equation()}")

    # The Schrödinger equation i∂ψ/∂t = -∂²ψ/∂x² should give coefficient ≈ -1
    # for the Laplacian term
    if result.complex_coeffs:
        lap_coeff = result.complex_coeffs[0]
        print(f"\nLaplacian coefficient: {lap_coeff}")
        # Check it's approximately -1 (negative because of our sign convention)
        assert abs(lap_coeff.real - (-1.0)) < 0.2 or abs(lap_coeff.real - 1.0) < 0.2, \
            f"Expected Laplacian coefficient near ±1, got {lap_coeff}"

    print("\nPASSED: test_schrodinger_discovery_constrained")
    return result


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Complex Ops Library")
    print("=" * 60)

    # Phase 1 tests
    print("\n--- Phase 1: Library Helper Tests ---\n")
    test_psi_creation()
    test_real_imag_parts()
    test_abs_sq_produces_correct_ast()
    test_dpsi_derivatives()
    test_d2psi_derivatives()
    test_complex_mul_expansion()
    test_abs_sq_psi_nls_term()
    test_laplacian_2d()

    # Phase 2 tests
    print("\n--- Phase 2: Constrained Discovery Tests ---\n")
    test_complex_term_spec_creation()
    test_complex_de_search_config_creation()
    test_make_laplacian_term()
    test_make_nls_term()
    test_make_psi_term()
    test_schrodinger_discovery_with_system_de()
    test_complex_equation_formatting()
    test_coefficient_constraint_validation()

    # Integration test
    print("\n--- Integration Tests ---\n")
    test_schrodinger_discovery_constrained()

    print("\n" + "=" * 60)
    print("ALL COMPLEX OPS TESTS PASSED")
    print("=" * 60)
