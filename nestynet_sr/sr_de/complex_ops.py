# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Complex-field helper utilities for DE discovery.

These utilities simplify construction of complex-valued library terms for
differential equation discovery. A complex field ψ = u + iv is represented
as a 2-component real surrogate output.

The module provides two tiers of functionality:

**Phase 1 - Library Helpers**: Functions for constructing complex-valued
library terms decomposed into real/imaginary parts.

**Phase 2 - Constrained Discovery**: Coefficient-tied discovery with
validation and physics-style output formatting.

Example usage (NLS: i∂ψ/∂t = -∂²ψ/∂x² + g|ψ|²ψ)::

    from nestynet_sr.sr_de.complex_ops import Psi, AbsSqPsi, D2Psi

    psi = Psi()  # defaults: out_real=0, out_imag=1

    # Build library components
    d2u, d2v = D2Psi(1, 1)       # Laplacian components
    nls_u, nls_v = AbsSqPsi(psi)  # NLS nonlinearity |ψ|²ψ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import (
    D2U,
    DU,
    Add,
    Mul,
    Node,
    Pow,
    U,
)

# ──────────────────────────────────────────────────────────────
# Phase 1: Library Helpers
# ──────────────────────────────────────────────────────────────


@dataclass
class ComplexField:
    """Wrapper for 2-component real surrogate representing ψ = u + iv.

    Attributes
    ----------
    out_real : int
        Surrogate output index for the real part u.
    out_imag : int
        Surrogate output index for the imaginary part v.
    """

    out_real: int = 0
    out_imag: int = 1


def Psi(out_real: int = 0, out_imag: int = 1) -> ComplexField:
    """Create a complex field wrapper.

    Parameters
    ----------
    out_real : int
        Surrogate output index for the real part (default 0).
    out_imag : int
        Surrogate output index for the imaginary part (default 1).

    Returns
    -------
    ComplexField
        Wrapper object for the complex field.
    """
    return ComplexField(out_real=int(out_real), out_imag=int(out_imag))


def real_part(z: ComplexField) -> Node:
    """Return the real part u as an AST node U(out_idx=z.out_real).

    Parameters
    ----------
    z : ComplexField
        Complex field wrapper.

    Returns
    -------
    Node
        AST node representing the real part.
    """
    return U(out_idx=int(z.out_real))


def imag_part(z: ComplexField) -> Node:
    """Return the imaginary part v as an AST node U(out_idx=z.out_imag).

    Parameters
    ----------
    z : ComplexField
        Complex field wrapper.

    Returns
    -------
    Node
        AST node representing the imaginary part.
    """
    return U(out_idx=int(z.out_imag))


def AbsSq(z: ComplexField) -> Node:
    """Return |ψ|² = u² + v² as a real-valued AST node.

    Parameters
    ----------
    z : ComplexField
        Complex field wrapper.

    Returns
    -------
    Node
        AST node for |ψ|² = u² + v².
    """
    u = U(out_idx=int(z.out_real))
    v = U(out_idx=int(z.out_imag))
    return Add(Pow(u, 2), Pow(v, 2))


def DPsi(axis: int, z: Optional[ComplexField] = None, out_real: int = 0, out_imag: int = 1) -> Tuple[Node, Node]:
    """Return (∂u/∂x, ∂v/∂x) tuple of first derivative AST nodes.

    Parameters
    ----------
    axis : int
        Derivative axis.
    z : ComplexField, optional
        Complex field wrapper. If provided, uses its out_real/out_imag.
    out_real : int
        Output index for real part (used if z is None).
    out_imag : int
        Output index for imaginary part (used if z is None).

    Returns
    -------
    Tuple[Node, Node]
        (du/dx, dv/dx) AST nodes.
    """
    if z is not None:
        out_real = int(z.out_real)
        out_imag = int(z.out_imag)
    du = DU(int(axis), out_idx=int(out_real))
    dv = DU(int(axis), out_idx=int(out_imag))
    return (du, dv)


def D2Psi(axis0: int, axis1: int, z: Optional[ComplexField] = None, out_real: int = 0, out_imag: int = 1) -> Tuple[Node, Node]:
    """Return (∂²u/∂x∂y, ∂²v/∂x∂y) tuple of second derivative AST nodes.

    Parameters
    ----------
    axis0 : int
        First derivative axis.
    axis1 : int
        Second derivative axis.
    z : ComplexField, optional
        Complex field wrapper. If provided, uses its out_real/out_imag.
    out_real : int
        Output index for real part (used if z is None).
    out_imag : int
        Output index for imaginary part (used if z is None).

    Returns
    -------
    Tuple[Node, Node]
        (d2u/dxdy, d2v/dxdy) AST nodes.
    """
    if z is not None:
        out_real = int(z.out_real)
        out_imag = int(z.out_imag)
    d2u = D2U(int(axis0), int(axis1), out_idx=int(out_real))
    d2v = D2U(int(axis0), int(axis1), out_idx=int(out_imag))
    return (d2u, d2v)


def ComplexMul(a: ComplexField, b: ComplexField) -> Tuple[Node, Node]:
    """Return (real, imag) parts of complex multiplication a * b.

    Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i)
        = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)

    Parameters
    ----------
    a : ComplexField
        First complex field.
    b : ComplexField
        Second complex field.

    Returns
    -------
    Tuple[Node, Node]
        (real_part, imag_part) AST nodes.
    """
    ar = U(out_idx=int(a.out_real))
    ai = U(out_idx=int(a.out_imag))
    br = U(out_idx=int(b.out_real))
    bi = U(out_idx=int(b.out_imag))

    # Real part: ar*br - ai*bi
    real = Add(Mul(ar, br), Mul(Pow(ai, 1), Mul(Pow(bi, 1), Pow(U(out_idx=0), 0))))  # This is wrong, fix below
    # Actually use proper subtraction via Add with negation
    from nestynet_sr.sr_core.bridges import ConstNode
    neg_one = ConstNode(-1.0)
    real = Add(Mul(ar, br), Mul(neg_one, Mul(ai, bi)))

    # Imag part: ar*bi + ai*br
    imag = Add(Mul(ar, bi), Mul(ai, br))

    return (real, imag)


def AbsSqPsi(z: ComplexField) -> Tuple[Node, Node]:
    """Return |ψ|²·ψ decomposed to ((u²+v²)·u, (u²+v²)·v).

    This is the nonlinear term in the NLS equation.

    Parameters
    ----------
    z : ComplexField
        Complex field wrapper.

    Returns
    -------
    Tuple[Node, Node]
        (|ψ|²·u, |ψ|²·v) AST nodes for real and imaginary parts.
    """
    u = U(out_idx=int(z.out_real))
    v = U(out_idx=int(z.out_imag))
    mod_sq = Add(Pow(u, 2), Pow(v, 2))  # |ψ|² = u² + v²

    # Need fresh U nodes since AST nodes may be evaluated multiple times
    u2 = U(out_idx=int(z.out_real))
    v2 = U(out_idx=int(z.out_imag))

    nls_real = Mul(mod_sq, u2)  # |ψ|²·u
    nls_imag = Mul(mod_sq, v2)  # |ψ|²·v

    return (nls_real, nls_imag)


def Laplacian2D(z: ComplexField, spatial_axis: int = 1) -> Tuple[Node, Node]:
    """Return Laplacian (∇²ψ) components for a 1D spatial system.

    Parameters
    ----------
    z : ComplexField
        Complex field wrapper.
    spatial_axis : int
        The spatial coordinate axis (default 1, assuming axis 0 is time).

    Returns
    -------
    Tuple[Node, Node]
        (∂²u/∂x², ∂²v/∂x²) AST nodes.
    """
    return D2Psi(spatial_axis, spatial_axis, z=z)


# ──────────────────────────────────────────────────────────────
# Phase 2: Constrained Complex Discovery
# ──────────────────────────────────────────────────────────────


@dataclass
class ComplexTermSpec:
    """A complex-valued library term with both components.

    Attributes
    ----------
    real_part : Node
        AST node for the real part of the term.
    imag_part : Node
        AST node for the imaginary part of the term.
    name : str
        Human-readable name for the term (e.g., "∂²ψ/∂x²").
    is_hermitian : bool
        If True, the operator is Hermitian (coefficients have opposite signs
        in the two real equations). If False, anti-Hermitian (same signs).
    """

    real_part: Node
    imag_part: Node
    name: str
    is_hermitian: bool = True


@dataclass
class ComplexDESearchConfig:
    """Configuration for complex DE discovery with coefficient constraints.

    For complex equations of the form i·∂ψ/∂t = L[ψ], the real decomposition is:
        ∂u/∂t = -Im(L[ψ])
        ∂v/∂t = +Re(L[ψ])

    The coefficient tying enforces that if a term T has coefficient c in the
    complex equation, then:
    - Hermitian operators: coeff_eq0 = -coeff_eq1
    - Anti-Hermitian operators: coeff_eq0 = +coeff_eq1

    Attributes
    ----------
    time_axis : int
        Axis used for the time derivative (anchor).
    out_real : int
        Surrogate output index for real part.
    out_imag : int
        Surrogate output index for imaginary part.
    complex_terms : Sequence[ComplexTermSpec]
        Complex-valued library terms to include.
    include_const : bool
        Whether to include a constant term.
    stlsq_lambda : float
        STLSQ sparsification threshold.
    stlsq_max_iter : int
        Maximum STLSQ iterations.
    ridge : float
        Ridge regularization parameter.
    max_batches : int
        Maximum number of batches to use from dataloader.
    max_points : int
        Maximum number of data points to use.
    """

    time_axis: int = 0
    out_real: int = 0
    out_imag: int = 1
    complex_terms: Sequence[ComplexTermSpec] = ()
    include_const: bool = False
    stlsq_lambda: float = 1e-3
    stlsq_max_iter: int = 10
    ridge: float = 1e-10
    max_batches: int = 32
    max_points: int = 20000


@dataclass
class ComplexDESearchResult:
    """Result of complex DE discovery.

    Attributes
    ----------
    underlying_result : SystemDESearchResult
        The underlying 2-component system DE result.
    complex_coeffs : List[complex]
        Discovered complex coefficients for each term.
    term_names : List[str]
        Names of the selected terms.
    rms_train : float
        Combined RMS training residual.
    is_valid_complex : bool
        True if the discovered coefficients satisfy the complex structure
        constraints (cross-coupling pattern is consistent).
    """

    underlying_result: "SystemDESearchResult"
    complex_coeffs: List[complex]
    term_names: List[str]
    rms_train: float
    is_valid_complex: bool

    def format_complex_equation(self, var_name: str = "x", psi_name: str = "psi") -> str:
        """Format the discovered equation in physics notation.

        Returns a string like: i d{psi}/dt = -1.0*d2{psi}/dx2 + 0.5*|{psi}|^2*{psi}

        Parameters
        ----------
        var_name : str
            Name of the spatial variable (default "x").
        psi_name : str
            Name of the complex field (default "psi").

        Returns
        -------
        str
            Human-readable complex equation string.
        """
        terms_str = []
        for c, name in zip(self.complex_coeffs, self.term_names):
            if abs(c) < 1e-12:
                continue

            # Format coefficient
            if abs(c.imag) < 1e-12:
                # Pure real coefficient
                c_str = f"{c.real:g}"
            elif abs(c.real) < 1e-12:
                # Pure imaginary coefficient
                c_str = f"{c.imag:g}i"
            else:
                # General complex coefficient
                c_str = f"({c.real:g}+{c.imag:g}i)"

            # Handle common cases
            if abs(c.real - 1.0) < 1e-6 and abs(c.imag) < 1e-12:
                terms_str.append(name)
            elif abs(c.real + 1.0) < 1e-6 and abs(c.imag) < 1e-12:
                terms_str.append(f"-{name}")
            else:
                terms_str.append(f"{c_str}*{name}")

        rhs = " + ".join(terms_str).replace("+ -", "- ")
        if rhs.strip() == "":
            rhs = "0"

        return f"i d{psi_name}/dt = {rhs}"


# Import here to avoid circular import
from nestynet_sr.sr_de.system_de_search import (
    SystemDESearchConfig,
    SystemDESearchResult,
    discover_system_de_from_surrogate,
)


def discover_complex_de_from_surrogate(
    surrogate,
    train_dataloader,
    val_dataloader=None,
    *,
    cfg: Optional[ComplexDESearchConfig] = None,
    device=None,
) -> ComplexDESearchResult:
    """Discover a complex DE with coefficient constraints and validation.

    This function wraps the system DE discovery to handle complex equations
    of the form i·∂ψ/∂t = L[ψ] where L is a (typically Hermitian) operator.

    The complex equation decomposes as:
        ∂u/∂t = -Im(L[ψ])
        ∂v/∂t = +Re(L[ψ])

    Parameters
    ----------
    surrogate : nn.Module
        Trained surrogate network with 2 outputs (real, imag).
    train_dataloader : iterable
        Training data loader.
    val_dataloader : iterable, optional
        Validation data loader.
    cfg : ComplexDESearchConfig, optional
        Configuration for complex DE discovery.
    device : torch.device, optional
        Device to use for computation.

    Returns
    -------
    ComplexDESearchResult
        Result containing complex coefficients, validation flag, and formatted output.
    """
    if cfg is None:
        cfg = ComplexDESearchConfig()

    # Build the 2-component library from complex terms
    # For i·∂ψ/∂t = Σ c_k T_k, decomposition is:
    #   ∂u/∂t = Σ c_k · (-Im(T_k))  where -Im(T_k) = -T_k.imag_part
    #   ∂v/∂t = Σ c_k · (+Re(T_k))  where +Re(T_k) = +T_k.real_part
    #
    # So library for eq0 (u) contains imaginary parts with sign flip
    # and library for eq1 (v) contains real parts

    lib_terms: List[Node] = []
    term_info: List[Tuple[str, bool]] = []  # (name, is_hermitian)

    for spec in cfg.complex_terms:
        # For each complex term, we add both parts to the shared library
        # The coefficient tying will be handled in post-processing
        lib_terms.append(spec.imag_part)  # contributes to eq0 with -coeff
        lib_terms.append(spec.real_part)  # contributes to eq1 with +coeff
        term_info.append((spec.name, spec.is_hermitian))

    # Create system DE config
    sys_cfg = SystemDESearchConfig(
        x_axis=int(cfg.time_axis),
        order_candidates=(1,),  # Complex DEs are first-order in time
        out_idxs=(int(cfg.out_real), int(cfg.out_imag)),
        include_const=bool(cfg.include_const),
        include_x=False,
        include_u=False,
        include_u_cross=False,
        include_xu=False,
        include_du=False,
        include_d2u=False,
        ridge=float(cfg.ridge),
        stlsq_lambda=float(cfg.stlsq_lambda),
        stlsq_max_iter=int(cfg.stlsq_max_iter),
        max_batches=int(cfg.max_batches),
        max_points=int(cfg.max_points),
        share_support_across_equations=False,  # Allow different supports initially
    )

    # Run system DE discovery
    result = discover_system_de_from_surrogate(
        surrogate,
        train_dataloader,
        val_dataloader,
        cfg=sys_cfg,
        device=device,
        library_terms=lib_terms if lib_terms else None,
    )

    # Post-process to extract complex coefficients and validate structure
    complex_coeffs: List[complex] = []
    term_names: List[str] = []
    is_valid = True

    # Map from term index to complex coefficient
    # Library structure: [imag_0, real_0, imag_1, real_1, ...]
    # eq0 coeff of imag_k = -c_k (from ∂u/∂t = -Im(L))
    # eq1 coeff of real_k = +c_k (from ∂v/∂t = +Re(L))

    len(cfg.complex_terms)
    coeffs = result.coeffs  # (2, K_selected)

    # Build mapping from selected terms back to original complex terms
    [repr(t) for t in result.term_asts]

    for i, spec in enumerate(cfg.complex_terms):
        imag_str = repr(spec.imag_part)
        real_str = repr(spec.real_part)

        c_from_imag = 0.0
        c_from_real = 0.0

        # Find coefficient in eq0 for imaginary part
        for k, t in enumerate(result.term_asts):
            if repr(t) == imag_str:
                # eq0 coeff of imag = -c, so c = -coeff
                c_from_imag = -float(coeffs[0, k].item())
                break

        # Find coefficient in eq1 for real part
        for k, t in enumerate(result.term_asts):
            if repr(t) == real_str:
                # eq1 coeff of real = +c, so c = coeff
                c_from_real = float(coeffs[1, k].item())
                break

        # For Hermitian operators, c should be real and consistent
        # For anti-Hermitian, c should be purely imaginary
        if spec.is_hermitian:
            # Both should give same real coefficient
            c_avg = (c_from_imag + c_from_real) / 2.0
            if abs(c_from_imag) > 1e-6 or abs(c_from_real) > 1e-6:
                rel_diff = abs(c_from_imag - c_from_real) / max(abs(c_avg), 1e-12)
                if rel_diff > 0.1:  # Allow 10% tolerance
                    is_valid = False
            complex_coeffs.append(complex(c_avg, 0.0))
        else:
            # Anti-Hermitian: coefficient is purely imaginary
            c_avg = (c_from_imag + c_from_real) / 2.0
            complex_coeffs.append(complex(0.0, c_avg))

        term_names.append(spec.name)

    # Compute combined RMS
    rms_combined = sum(result.rms_train) / max(1, len(result.rms_train))

    return ComplexDESearchResult(
        underlying_result=result,
        complex_coeffs=complex_coeffs,
        term_names=term_names,
        rms_train=float(rms_combined),
        is_valid_complex=is_valid,
    )


# ──────────────────────────────────────────────────────────────
# Convenience builders for common complex terms
# ──────────────────────────────────────────────────────────────


def make_laplacian_term(z: ComplexField, spatial_axis: int = 1, name: str = "d2psi/dx2") -> ComplexTermSpec:
    """Create a ComplexTermSpec for the Laplacian ∂²ψ/∂x².

    Parameters
    ----------
    z : ComplexField
        Complex field wrapper.
    spatial_axis : int
        Spatial coordinate axis.
    name : str
        Human-readable name for the term.

    Returns
    -------
    ComplexTermSpec
        Term specification for the Laplacian.
    """
    d2u, d2v = D2Psi(spatial_axis, spatial_axis, z=z)
    return ComplexTermSpec(
        real_part=d2u,
        imag_part=d2v,
        name=name,
        is_hermitian=True,
    )


def make_nls_term(z: ComplexField, name: str = "|psi|^2*psi") -> ComplexTermSpec:
    """Create a ComplexTermSpec for the NLS nonlinearity |ψ|²ψ.

    Parameters
    ----------
    z : ComplexField
        Complex field wrapper.
    name : str
        Human-readable name for the term.

    Returns
    -------
    ComplexTermSpec
        Term specification for |ψ|²ψ.
    """
    nls_u, nls_v = AbsSqPsi(z)
    return ComplexTermSpec(
        real_part=nls_u,
        imag_part=nls_v,
        name=name,
        is_hermitian=True,  # |ψ|²ψ is a real-coefficient nonlinearity
    )


def make_psi_term(z: ComplexField, name: str = "psi") -> ComplexTermSpec:
    """Create a ComplexTermSpec for ψ itself (useful for linear potentials).

    Parameters
    ----------
    z : ComplexField
        Complex field wrapper.
    name : str
        Human-readable name for the term.

    Returns
    -------
    ComplexTermSpec
        Term specification for ψ.
    """
    u = real_part(z)
    v = imag_part(z)
    return ComplexTermSpec(
        real_part=u,
        imag_part=v,
        name=name,
        is_hermitian=True,
    )
