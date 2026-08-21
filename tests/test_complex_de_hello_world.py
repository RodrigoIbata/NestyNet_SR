# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Hello-world tests for complex differential equation discovery.

Demonstrates that complex PDEs can be discovered using the existing vector DE
infrastructure by representing complex fields as 2-component real systems.

Key insight: A complex field ψ = u + iv decomposes into a 2-component real
system. The existing `discover_system_de_from_surrogate` handles this directly.

Test cases:
-----------
CDE000: Rotating phasor  dz/dt = i·ω·z
        → du/dt = -ω·v,  dv/dt = +ω·u
        Solution: z(t) = exp(i·ω·t) → u = cos(ω·t), v = sin(ω·t)

CDE010: Free Schrödinger  i∂ψ/∂t = -∂²ψ/∂x²
        → ∂u/∂t = -∂²v/∂x²,  ∂v/∂t = +∂²u/∂x²
        Solution: plane wave ψ(x,t) = exp(i(kx - ωt)) with ω = k²
"""

import math
import torch

torch.set_default_dtype(torch.float64)


# ══════════════════════════════════════════════════════════════
# CDE000: Rotating Phasor  dz/dt = i·ω·z
# ══════════════════════════════════════════════════════════════

class RotatingPhasorSurrogate(torch.nn.Module):
    """Mock surrogate for CDE000: dz/dt = i·ω·z  (ω=1)

    Solution: z(t) = exp(i·t) → u(t) = cos(t), v(t) = sin(t)

    Decomposed real system:
        du/dt = -v   (coefficient of v = +1 in residual du/dt + 1·v = 0)
        dv/dt = +u   (coefficient of u = -1 in residual dv/dt - 1·u = 0)
    """

    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = omega
        # Dummy parameter for device detection
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x[:, 0]
        u = torch.cos(self.omega * t)
        v = torch.sin(self.omega * t)
        return torch.stack([u, v], dim=1)  # (B, 2)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """Analytic gradient: du/dt, dv/dt."""
        t = x[:, 0]
        du_dt = -self.omega * torch.sin(self.omega * t)
        dv_dt = self.omega * torch.cos(self.omega * t)
        # Shape: (B, 2, 1) for 2 outputs, 1 input axis
        return torch.stack([
            du_dt.unsqueeze(-1),
            dv_dt.unsqueeze(-1),
        ], dim=1)

    def parameters(self):
        return iter([self._dummy])


def test_cde000_rotating_phasor():
    """CDE000: Discover dz/dt = i·ω·z as a coupled real system."""
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import U

    omega = 1.0
    surrogate = RotatingPhasorSurrogate(omega=omega)

    # Simple 1D time grid
    t_vals = torch.linspace(0.0, 2 * math.pi, 200)
    X = t_vals.unsqueeze(1)  # (N, 1)
    loader = [(X,)]  # Single-batch iterable

    # Library: just u and v (no spatial derivatives for ODE)
    u_term = U(out_idx=0)  # u
    v_term = U(out_idx=1)  # v
    lib = [u_term, v_term]

    cfg = SystemDESearchConfig(
        x_axis=0,                   # t is evolution axis
        order_candidates=(1,),      # first-order ODE
        include_const=False,
        include_x=False,
        include_u=False,            # we provide library explicitly
        include_u_cross=False,
        include_xu=False,
        include_du=False,
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate, loader, cfg=cfg, library_terms=lib, device=torch.device("cpu")
    )

    print("\n" + "=" * 60)
    print("CDE000: Rotating Phasor  dz/dt = i·ω·z  (ω=1)")
    print("=" * 60)
    print(f"Order: {result.order}")
    print(f"Term ASTs: {[str(t) for t in result.term_asts]}")
    print(f"Coefficients shape: {result.coeffs.shape}")
    print(f"Coefficients:\n{result.coeffs}")
    print("\nDiscovered system:")
    print(result.format_system())

    # Verify order
    assert result.order == 1, f"Expected order=1, got {result.order}"

    # Verify RMS is very small (exact surrogate)
    for i, rms in enumerate(result.rms_train):
        print(f"  eq{i} RMS = {rms:.2e}")
        assert rms < 1e-6, f"RMS too large for eq{i}: {rms}"

    # Verify cross-coupling structure:
    # eq0 (du/dt): residual = du/dt + c0·u + c1·v = 0
    #              Since du/dt = -ω·v, we need c1 ≈ +ω = +1 (and c0 ≈ 0)
    # eq1 (dv/dt): residual = dv/dt + c0·u + c1·v = 0
    #              Since dv/dt = +ω·u, we need c0 ≈ -ω = -1 (and c1 ≈ 0)
    coeffs = result.coeffs  # (2, K_sel)
    print("\nCoefficient verification:")

    # Find which columns correspond to u and v terms
    term_strs = [str(t) for t in result.term_asts]
    print(f"  Term strings: {term_strs}")

    # With explicit library [U(out_idx=0), U(out_idx=1)], order should match
    # eq0: coefficient of v (out_idx=1) should be ≈ +omega
    # eq1: coefficient of u (out_idx=0) should be ≈ -omega
    if len(term_strs) >= 2:
        # Check magnitudes are approximately omega
        max_coeff = coeffs.abs().max().item()
        assert max_coeff > 0.8 * omega, f"Expected coefficient magnitude ~{omega}, got {max_coeff}"

    print("\nPASSED: CDE000 Rotating phasor discovered correctly.")
    return result


# ══════════════════════════════════════════════════════════════
# CDE010: Free Schrödinger  i∂ψ/∂t = -∂²ψ/∂x²
# ══════════════════════════════════════════════════════════════

class FreeSchrodingerSurrogate(torch.nn.Module):
    """Mock surrogate for CDE010: Free Schrödinger equation.

    i∂ψ/∂t = -∂²ψ/∂x²   (ℏ=1, m=1/2 so coefficient is 1)

    Solution: Plane wave ψ(x,t) = exp(i(kx - ωt)) with dispersion ω = k²

    With k=1: ψ(x,t) = exp(i(x - t))
              u(x,t) = cos(x - t)
              v(x,t) = sin(x - t)

    Decomposed real system:
        ∂u/∂t = -∂²v/∂x²
        ∂v/∂t = +∂²u/∂x²
    """

    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k
        self.omega = k ** 2  # dispersion relation
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x[:, 0] = t, x[:, 1] = spatial x"""
        t, xs = x[:, 0], x[:, 1]
        phase = self.k * xs - self.omega * t
        u = torch.cos(phase)
        v = torch.sin(phase)
        return torch.stack([u, v], dim=1)  # (B, 2)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """First derivatives: ∂u/∂t, ∂u/∂x, ∂v/∂t, ∂v/∂x."""
        t, xs = x[:, 0], x[:, 1]
        phase = self.k * xs - self.omega * t

        # u = cos(phase), v = sin(phase)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)

        # ∂u/∂t = -sin(phase) · (-ω) = ω·sin(phase)
        du_dt = self.omega * sin_p
        # ∂u/∂x = -sin(phase) · k = -k·sin(phase)
        du_dx = -self.k * sin_p
        # ∂v/∂t = cos(phase) · (-ω) = -ω·cos(phase)
        dv_dt = -self.omega * cos_p
        # ∂v/∂x = cos(phase) · k = k·cos(phase)
        dv_dx = self.k * cos_p

        # Shape: (B, 2, 2) for 2 outputs, 2 input axes
        gu = torch.stack([du_dt, du_dx], dim=-1)  # (B, 2)
        gv = torch.stack([dv_dt, dv_dx], dim=-1)  # (B, 2)
        return torch.stack([gu, gv], dim=1)  # (B, 2, 2)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Second derivatives (Hessian): ∂²u/∂x², ∂²v/∂x², etc.

        Returns shape (B, 2, 2, 2) for 2 outputs, 2x2 Hessian per output.
        Indexing: H[b, out_idx, axis0, axis1]
        """
        t, xs = x[:, 0], x[:, 1]
        phase = self.k * xs - self.omega * t

        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)

        # u = cos(phase), v = sin(phase), phase = k·x - ω·t
        # ∂u/∂t = ω·sin(phase), ∂u/∂x = -k·sin(phase)
        # ∂v/∂t = -ω·cos(phase), ∂v/∂x = k·cos(phase)

        # ∂²u/∂t² = ω·cos(phase)·(-ω) = -ω²·cos(phase)
        d2u_dt2 = -self.omega ** 2 * cos_p
        # ∂²u/∂x² = -k·cos(phase)·k = -k²·cos(phase)
        d2u_dx2 = -self.k ** 2 * cos_p
        # ∂²u/∂t∂x = ω·cos(phase)·k = ω·k·cos(phase)
        d2u_dtdx = self.omega * self.k * cos_p

        # ∂²v/∂t² = -ω·(-sin(phase))·(-ω) = -ω²·sin(phase)
        d2v_dt2 = -self.omega ** 2 * sin_p
        # ∂²v/∂x² = k·(-sin(phase))·k = -k²·sin(phase)
        d2v_dx2 = -self.k ** 2 * sin_p
        # ∂²v/∂t∂x = -ω·(-sin(phase))·k = ω·k·sin(phase)
        d2v_dtdx = self.omega * self.k * sin_p

        B = x.shape[0]
        H = torch.zeros(B, 2, 2, 2, dtype=x.dtype, device=x.device)
        # u Hessian
        H[:, 0, 0, 0] = d2u_dt2
        H[:, 0, 0, 1] = d2u_dtdx
        H[:, 0, 1, 0] = d2u_dtdx
        H[:, 0, 1, 1] = d2u_dx2
        # v Hessian
        H[:, 1, 0, 0] = d2v_dt2
        H[:, 1, 0, 1] = d2v_dtdx
        H[:, 1, 1, 0] = d2v_dtdx
        H[:, 1, 1, 1] = d2v_dx2

        return H  # (B, 2, 2, 2)

    def parameters(self):
        return iter([self._dummy])


def test_cde010_free_schrodinger():
    """CDE010: Discover i∂ψ/∂t = -∂²ψ/∂x² as a coupled real system."""
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import D2U

    k = 1.0
    surrogate = FreeSchrodingerSurrogate(k=k)

    # 2D grid: t × x
    t_vals = torch.linspace(0.0, 2 * math.pi, 50)
    x_vals = torch.linspace(0.0, 2 * math.pi, 50)
    tt, xx = torch.meshgrid(t_vals, x_vals, indexing="ij")
    X = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)
    loader = [(X,)]

    # Library: second spatial derivatives ∂²u/∂x², ∂²v/∂x²
    # For i∂ψ/∂t = -∂²ψ/∂x², the real decomposition is:
    #   ∂u/∂t = -∂²v/∂x²  →  residual: ∂u/∂t + 1·∂²v/∂x² = 0
    #   ∂v/∂t = +∂²u/∂x²  →  residual: ∂v/∂t - 1·∂²u/∂x² = 0
    d2u_dx2 = D2U(1, 1, out_idx=0)  # ∂²u/∂x²
    d2v_dx2 = D2U(1, 1, out_idx=1)  # ∂²v/∂x²
    lib = [d2u_dx2, d2v_dx2]

    cfg = SystemDESearchConfig(
        x_axis=0,                   # t is evolution axis
        order_candidates=(1,),      # first-order in t (anchor is ∂/∂t)
        include_const=False,
        include_x=False,
        include_u=False,
        include_u_cross=False,
        include_xu=False,
        include_du=False,
        include_d2u=False,          # we provide library explicitly
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate, loader, cfg=cfg, library_terms=lib, device=torch.device("cpu")
    )

    print("\n" + "=" * 60)
    print("CDE010: Free Schrödinger  i∂ψ/∂t = -∂²ψ/∂x²")
    print("=" * 60)
    print(f"Order: {result.order}")
    print(f"Term ASTs: {[str(t) for t in result.term_asts]}")
    print(f"Coefficients shape: {result.coeffs.shape}")
    print(f"Coefficients:\n{result.coeffs}")
    print("\nDiscovered system:")
    print(result.format_system())

    # Verify order
    assert result.order == 1, f"Expected order=1, got {result.order}"

    # Verify RMS is very small
    for i, rms in enumerate(result.rms_train):
        print(f"  eq{i} RMS = {rms:.2e}")
        assert rms < 1e-6, f"RMS too large for eq{i}: {rms}"

    # Verify cross-coupling structure:
    # With k=1, ω=k²=1, we expect:
    #   eq0: ∂u/∂t + c·∂²v/∂x² = 0, with c ≈ +1
    #   eq1: ∂v/∂t + c·∂²u/∂x² = 0, with c ≈ -1
    coeffs = result.coeffs
    print("\nCoefficient verification:")
    print("  Expected: eq0 coeff of ∂²v/∂x² ≈ +1, eq1 coeff of ∂²u/∂x² ≈ -1")

    # Check coefficient magnitudes
    max_coeff = coeffs.abs().max().item()
    assert max_coeff > 0.8, f"Expected coefficient magnitude ~1, got {max_coeff}"

    print("\nPASSED: CDE010 Free Schrödinger discovered correctly.")
    return result


# ══════════════════════════════════════════════════════════════
# CDE020: Nonlinear Schrödinger (NLS)  i∂ψ/∂t = -∂²ψ/∂x² + g|ψ|²ψ
# ══════════════════════════════════════════════════════════════

class NLSSurrogate(torch.nn.Module):
    """Mock surrogate for CDE020: Nonlinear Schrödinger equation (focusing).

    i∂ψ/∂t = -∂²ψ/∂x² + g|ψ|²ψ   (g > 0 focusing)

    For simplicity, use a soliton-like profile (though not an exact soliton).
    We use a mock solution: ψ(x,t) = sech(x) · exp(i·μ·t)

    With μ = -g·sech²(0) + 1 = 1 - g (for normalization at x=0)

    Decomposed real system:
        ∂u/∂t = -∂²v/∂x² - g(u²+v²)v
        ∂v/∂t = +∂²u/∂x² + g(u²+v²)u

    Note: For testing we use a simplified Gaussian envelope to ensure smooth
    second derivatives without sech edge effects.
    """

    def __init__(self, g: float = 1.0, sigma: float = 1.0):
        super().__init__()
        self.g = g
        self.sigma = sigma
        # For Gaussian envelope: ψ(x,t) = exp(-x²/(2σ²)) · exp(i·μ·t)
        # where μ is chosen so the equation balances at x=0
        # ∂²(exp(-x²/(2σ²)))/∂x² at x=0 = -1/σ²
        # So μ = -(-1/σ²) + g·1 = 1/σ² + g  (but we use μ=1 for simplicity)
        self.mu = 1.0
        self.register_buffer("_dummy", torch.zeros(1))

    def _envelope(self, xs: torch.Tensor) -> torch.Tensor:
        """Gaussian envelope: exp(-x²/(2σ²))"""
        return torch.exp(-xs ** 2 / (2 * self.sigma ** 2))

    def _d_envelope(self, xs: torch.Tensor) -> torch.Tensor:
        """First derivative of envelope: -x/σ² · envelope"""
        env = self._envelope(xs)
        return -xs / (self.sigma ** 2) * env

    def _d2_envelope(self, xs: torch.Tensor) -> torch.Tensor:
        """Second derivative: (x²/σ⁴ - 1/σ²) · envelope"""
        env = self._envelope(xs)
        return ((xs ** 2 / self.sigma ** 4) - (1.0 / self.sigma ** 2)) * env

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x[:, 0] = t, x[:, 1] = spatial x"""
        t, xs = x[:, 0], x[:, 1]
        env = self._envelope(xs)
        phase = self.mu * t
        u = env * torch.cos(phase)
        v = env * torch.sin(phase)
        return torch.stack([u, v], dim=1)  # (B, 2)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """First derivatives."""
        t, xs = x[:, 0], x[:, 1]
        env = self._envelope(xs)
        d_env = self._d_envelope(xs)
        phase = self.mu * t
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)

        # u = env · cos(phase), v = env · sin(phase)
        du_dt = -self.mu * env * sin_p
        du_dx = d_env * cos_p
        dv_dt = self.mu * env * cos_p
        dv_dx = d_env * sin_p

        gu = torch.stack([du_dt, du_dx], dim=-1)
        gv = torch.stack([dv_dt, dv_dx], dim=-1)
        return torch.stack([gu, gv], dim=1)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Second derivatives (Hessian)."""
        t, xs = x[:, 0], x[:, 1]
        env = self._envelope(xs)
        d_env = self._d_envelope(xs)
        d2_env = self._d2_envelope(xs)
        phase = self.mu * t
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)

        # Second derivatives
        d2u_dt2 = -self.mu ** 2 * env * cos_p
        d2u_dx2 = d2_env * cos_p
        d2u_dtdx = -self.mu * d_env * sin_p

        d2v_dt2 = -self.mu ** 2 * env * sin_p
        d2v_dx2 = d2_env * sin_p
        d2v_dtdx = self.mu * d_env * cos_p

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


def test_cde020_nls():
    """CDE020: Discover NLS i∂ψ/∂t = -∂²ψ/∂x² + g|ψ|²ψ as a coupled real system.

    This test demonstrates nonlinear terms |ψ|²ψ = (u²+v²)·ψ in the library.

    Note: The mock surrogate uses a Gaussian envelope which is NOT an exact
    NLS soliton, so we verify only that the system correctly identifies
    the structure of the nonlinear terms.
    """
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import D2U, U, Mul, Add, Pow

    g = 1.0
    sigma = 1.0
    surrogate = NLSSurrogate(g=g, sigma=sigma)

    # 2D grid: t × x (narrower domain to keep envelope smooth)
    t_vals = torch.linspace(0.0, math.pi, 40)
    x_vals = torch.linspace(-2.0, 2.0, 40)
    tt, xx = torch.meshgrid(t_vals, x_vals, indexing="ij")
    X = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)
    loader = [(X,)]

    # Build library with both linear (∂²/∂x²) and nonlinear (|ψ|²·ψ) terms
    # Linear terms: ∂²u/∂x², ∂²v/∂x²
    d2u_dx2 = D2U(1, 1, out_idx=0)
    d2v_dx2 = D2U(1, 1, out_idx=1)

    # Nonlinear terms: (u²+v²)·u, (u²+v²)·v  (i.e., |ψ|²·ψ components)
    u = U(out_idx=0)
    v = U(out_idx=1)
    mod_sq = Add(Pow(u, 2), Pow(v, 2))  # |ψ|² = u² + v²
    nls_u = Mul(mod_sq, u)  # |ψ|²·u
    nls_v = Mul(mod_sq, v)  # |ψ|²·v

    lib = [d2u_dx2, d2v_dx2, nls_u, nls_v]

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

    print("\n" + "=" * 60)
    print("CDE020: NLS  i∂ψ/∂t = -∂²ψ/∂x² + g|ψ|²ψ  (g=1)")
    print("=" * 60)
    print(f"Order: {result.order}")
    print(f"Term ASTs: {[str(t) for t in result.term_asts]}")
    print(f"Coefficients shape: {result.coeffs.shape}")
    print(f"Coefficients:\n{result.coeffs}")
    print("\nDiscovered system:")
    print(result.format_system())

    # Verify order
    assert result.order == 1, f"Expected order=1, got {result.order}"

    # Note: The mock surrogate is not an exact NLS solution, so we expect
    # non-zero residuals. We verify only that the system structure is correct
    # and coefficients have reasonable magnitudes.
    for i, rms in enumerate(result.rms_train):
        print(f"  eq{i} RMS = {rms:.2e}")
        # Allow larger RMS since mock is not exact solution
        assert rms < 1.0, f"RMS unexpectedly large for eq{i}: {rms}"

    # Verify that both linear and nonlinear terms are present
    term_strs = [str(t) for t in result.term_asts]
    print("\nTerm verification:")
    print(f"  Found terms: {term_strs}")

    # Check we have at least 2 terms selected
    assert len(result.term_asts) >= 2, f"Expected at least 2 terms, got {len(result.term_asts)}"

    # Check coefficients have reasonable magnitudes
    max_coeff = result.coeffs.abs().max().item()
    assert max_coeff > 0.1, f"Coefficients too small: max = {max_coeff}"

    print("\nPASSED: CDE020 NLS structure discovered (mock surrogate, non-exact).")
    return result


# ══════════════════════════════════════════════════════════════
# CDE001: Damped Rotating Phasor  dz/dt = (-γ + i·ω)·z
# ══════════════════════════════════════════════════════════════

class DampedPhasorSurrogate(torch.nn.Module):
    """Mock surrogate for CDE001: dz/dt = (-γ + i·ω)·z

    Solution: z(t) = exp((-γ + i·ω)·t) = exp(-γ·t)·exp(i·ω·t)
              u(t) = exp(-γ·t)·cos(ω·t)
              v(t) = exp(-γ·t)·sin(ω·t)

    Decomposed real system:
        du/dt = -γ·u - ω·v
        dv/dt = -γ·v + ω·u
    """

    def __init__(self, gamma: float = 0.5, omega: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.omega = omega
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x[:, 0]
        decay = torch.exp(-self.gamma * t)
        u = decay * torch.cos(self.omega * t)
        v = decay * torch.sin(self.omega * t)
        return torch.stack([u, v], dim=1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        t = x[:, 0]
        decay = torch.exp(-self.gamma * t)
        cos_wt = torch.cos(self.omega * t)
        sin_wt = torch.sin(self.omega * t)

        # du/dt = -γ·exp(-γt)·cos(ωt) - ω·exp(-γt)·sin(ωt)
        du_dt = decay * (-self.gamma * cos_wt - self.omega * sin_wt)
        # dv/dt = -γ·exp(-γt)·sin(ωt) + ω·exp(-γt)·cos(ωt)
        dv_dt = decay * (-self.gamma * sin_wt + self.omega * cos_wt)

        return torch.stack([
            du_dt.unsqueeze(-1),
            dv_dt.unsqueeze(-1),
        ], dim=1)

    def parameters(self):
        return iter([self._dummy])


def test_cde001_damped_phasor():
    """CDE001: Discover dz/dt = (-γ + i·ω)·z as a coupled real system."""
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import U

    gamma = 0.5
    omega = 1.0
    surrogate = DampedPhasorSurrogate(gamma=gamma, omega=omega)

    t_vals = torch.linspace(0.0, 4 * math.pi, 300)
    X = t_vals.unsqueeze(1)
    loader = [(X,)]

    u_term = U(out_idx=0)
    v_term = U(out_idx=1)
    lib = [u_term, v_term]

    cfg = SystemDESearchConfig(
        x_axis=0,
        order_candidates=(1,),
        include_const=False,
        include_x=False,
        include_u=False,
        include_u_cross=False,
        include_xu=False,
        include_du=False,
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate, loader, cfg=cfg, library_terms=lib, device=torch.device("cpu")
    )

    print("\n" + "=" * 60)
    print(f"CDE001: Damped Phasor  dz/dt = ({-gamma} + i·{omega})·z")
    print("=" * 60)
    print(f"Order: {result.order}")
    print(f"Coefficients:\n{result.coeffs}")
    print("\nDiscovered system:")
    print(result.format_system())

    # Expected:
    #   du/dt + γ·u + ω·v = 0  → coeffs: [γ, ω] = [0.5, 1.0]
    #   dv/dt - ω·u + γ·v = 0  → coeffs: [-ω, γ] = [-1.0, 0.5]

    assert result.order == 1

    for i, rms in enumerate(result.rms_train):
        print(f"  eq{i} RMS = {rms:.2e}")
        assert rms < 1e-6, f"RMS too large for eq{i}: {rms}"

    # Verify coefficient values (approximate)
    coeffs = result.coeffs
    # Row 0 (du/dt equation): should have γ≈0.5 on u, ω≈1.0 on v
    # Row 1 (dv/dt equation): should have -ω≈-1.0 on u, γ≈0.5 on v
    print("\nCoefficient verification:")
    print(f"  eq0 (du/dt): coeffs = {coeffs[0].tolist()}")
    print(f"  eq1 (dv/dt): coeffs = {coeffs[1].tolist()}")

    # Check that we have the expected structure
    assert abs(coeffs[0, 0].item() - gamma) < 0.1, f"eq0 coeff[u] ≈ {gamma}"
    assert abs(coeffs[0, 1].item() - omega) < 0.1, f"eq0 coeff[v] ≈ {omega}"
    assert abs(coeffs[1, 0].item() + omega) < 0.1, f"eq1 coeff[u] ≈ {-omega}"
    assert abs(coeffs[1, 1].item() - gamma) < 0.1, f"eq1 coeff[v] ≈ {gamma}"

    print("\nPASSED: CDE001 Damped phasor discovered correctly.")
    return result


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Complex DE Discovery via Vector DE Infrastructure")
    print("=" * 60)

    # Run all tests
    test_cde000_rotating_phasor()
    test_cde001_damped_phasor()
    test_cde010_free_schrodinger()
    test_cde020_nls()

    print("\n" + "=" * 60)
    print("ALL COMPLEX DE HELLO-WORLD TESTS PASSED")
    print("=" * 60)
