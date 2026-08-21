# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Generalized Noether classification for sampled Poisson systems.

The canonical helper in :mod:`nestynet_sr.sr_gs.noether_reduction` constructs
charges in a fixed ``(q, p)`` chart.  This module separates the three conditions
which must not be conflated for a general Poisson tensor:

1. ``L_Y Pi = 0`` -- ``Y`` is a Poisson symmetry;
2. ``Y(H) = 0`` -- the candidate symmetry preserves the Hamiltonian;
3. ``Y = Pi grad(G)`` -- ``Y`` is Hamiltonian and admits a sampled/local charge.

The last condition is fit linearly in a caller-provided scalar charge library.
A sampled fit is never advertised as a global momentum map: topology and
Poisson-cohomology obstructions require a separate proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ChargeFitReport:
    """Linear fit of a charge ``G=sum_k coefficients[k] * phi_k``."""

    coefficients: Array
    term_names: tuple[str, ...]
    reconstructed_generator: Array
    relative_residual: float
    condition_number: float
    accepted: bool
    local_charge_found: bool
    global_charge_proven: bool
    charge_status: str


@dataclass(frozen=True)
class NoetherClassification:
    """Separate Poisson, Hamiltonian-invariance, and charge conclusions."""

    lie_derivative: Array
    hamiltonian_derivative: Array
    lie_relative_residual: float
    hamiltonian_relative_residual: float
    poisson_symmetry: bool
    preserves_hamiltonian: bool
    charge_fit: ChargeFitReport | None
    hamiltonian_generator: bool
    hamiltonian_symmetry: bool
    classification: str
    global_charge_proven: bool
    caveat: str


def _finite_array(value: Array, *, name: str, ndim: int | None = None) -> Array:
    out = np.asarray(value, dtype=np.float64)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _rms(value: Array) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def poisson_lie_derivative(
    generator_values: Array,
    generator_jacobians: Array,
    poisson_values: Array,
    poisson_derivatives: Array,
) -> Array:
    """Evaluate ``L_Y Pi`` at sampled state-space points.

    Shape convention is

    - ``Y[n, i]``;
    - ``DY[n, i, k] = partial_k Y^i``;
    - ``Pi[n, i, j]``;
    - ``DPi[n, i, j, k] = partial_k Pi^{ij}``.
    """

    y = _finite_array(generator_values, name="generator_values", ndim=2)
    dy = _finite_array(generator_jacobians, name="generator_jacobians", ndim=3)
    pi = _finite_array(poisson_values, name="poisson_values", ndim=3)
    dpi = _finite_array(poisson_derivatives, name="poisson_derivatives", ndim=4)
    n, d = y.shape
    if dy.shape != (n, d, d):
        raise ValueError(f"generator_jacobians must have shape {(n, d, d)}, got {dy.shape}")
    if pi.shape != (n, d, d):
        raise ValueError(f"poisson_values must have shape {(n, d, d)}, got {pi.shape}")
    if dpi.shape != (n, d, d, d):
        raise ValueError(
            f"poisson_derivatives must have shape {(n, d, d, d)}, got {dpi.shape}"
        )
    transport = np.einsum("nk,nijk->nij", y, dpi)
    first_index = np.einsum("nik,nkj->nij", dy, pi)
    second_index = np.einsum("njk,nik->nij", dy, pi)
    return transport - first_index - second_index


def fit_charge_from_gradient_library(
    generator_values: Array,
    poisson_values: Array,
    charge_gradients: Array,
    *,
    term_names: Sequence[str] | None = None,
    mask: Array | None = None,
    residual_tol: float = 1.0e-8,
    rcond: float | None = None,
) -> ChargeFitReport:
    """Fit ``Y = Pi grad(G)`` for a linear scalar charge library.

    ``charge_gradients[n, k, j]`` is ``partial_j phi_k``.  This convention
    matches a scalar AST library evaluated together with its analytic gradient.
    """

    y = _finite_array(generator_values, name="generator_values", ndim=2)
    pi = _finite_array(poisson_values, name="poisson_values", ndim=3)
    gradients = _finite_array(charge_gradients, name="charge_gradients", ndim=3)
    n, d = y.shape
    if pi.shape != (n, d, d):
        raise ValueError(f"poisson_values must have shape {(n, d, d)}, got {pi.shape}")
    if gradients.shape[0] != n or gradients.shape[2] != d:
        raise ValueError(
            "charge_gradients must have shape (N, K, d) matching generator_values"
        )
    k_terms = gradients.shape[1]
    if k_terms == 0:
        raise ValueError("charge_gradients must contain at least one term")
    if term_names is None:
        names = tuple(f"phi_{k}" for k in range(k_terms))
    else:
        names = tuple(str(name) for name in term_names)
        if len(names) != k_terms:
            raise ValueError("term_names length must equal the charge library size")
    if mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = np.asarray(mask, dtype=bool)
        if active.shape != (n,):
            raise ValueError(f"mask must have shape ({n},), got {active.shape}")
    if not np.any(active):
        raise ValueError("mask selects no samples")

    flows = np.einsum("nij,nkj->nki", pi, gradients)
    design = flows[active].transpose(0, 2, 1).reshape(-1, k_terms)
    target = y[active].reshape(-1)
    coeffs, _, _, singular_values = np.linalg.lstsq(design, target, rcond=rcond)
    reconstructed = np.einsum("nki,k->ni", flows, coeffs)
    residual = np.linalg.norm(y[active] - reconstructed[active])
    scale = max(np.linalg.norm(y[active]), np.finfo(np.float64).tiny)
    relative = float(residual / scale)
    if singular_values.size == 0 or singular_values[-1] <= 0.0:
        condition = float("inf")
    else:
        condition = float(singular_values[0] / singular_values[-1])
    accepted = bool(relative <= float(residual_tol))
    return ChargeFitReport(
        coefficients=coeffs,
        term_names=names,
        reconstructed_generator=reconstructed,
        relative_residual=relative,
        condition_number=condition,
        accepted=accepted,
        local_charge_found=accepted,
        global_charge_proven=False,
        charge_status="sampled_local_charge" if accepted else "charge_library_fit_failed",
    )


def classify_noether_symmetry(
    generator_values: Array,
    generator_jacobians: Array,
    poisson_values: Array,
    poisson_derivatives: Array,
    hamiltonian_gradients: Array,
    *,
    charge_gradients: Array | None = None,
    charge_term_names: Sequence[str] | None = None,
    mask: Array | None = None,
    poisson_tol: float = 1.0e-8,
    hamiltonian_tol: float = 1.0e-8,
    charge_tol: float = 1.0e-8,
    scale_floor: float = 1.0e-14,
) -> NoetherClassification:
    """Classify a candidate symmetry without manufacturing a momentum map.

    The returned ``hamiltonian_symmetry`` is true only when all three sampled
    tests pass.  Even then, ``global_charge_proven`` remains false: a finite
    state-space sample and a scalar library establish only a local/candidate
    charge, not global exactness.
    """

    y = _finite_array(generator_values, name="generator_values", ndim=2)
    dy = _finite_array(generator_jacobians, name="generator_jacobians", ndim=3)
    pi = _finite_array(poisson_values, name="poisson_values", ndim=3)
    dpi = _finite_array(poisson_derivatives, name="poisson_derivatives", ndim=4)
    grad_h = _finite_array(hamiltonian_gradients, name="hamiltonian_gradients", ndim=2)
    n, d = y.shape
    if grad_h.shape != (n, d):
        raise ValueError(f"hamiltonian_gradients must have shape {(n, d)}")
    if mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = np.asarray(mask, dtype=bool)
        if active.shape != (n,):
            raise ValueError(f"mask must have shape ({n},)")
    if not np.any(active):
        raise ValueError("mask selects no samples")

    lie = poisson_lie_derivative(y, dy, pi, dpi)
    h_derivative = np.einsum("ni,ni->n", y, grad_h)
    lie_scale = _rms(y[active]) * _rms(dpi[active]) + _rms(dy[active]) * _rms(pi[active])
    lie_relative = _rms(lie[active]) / max(lie_scale, float(scale_floor))
    h_scale = _rms(y[active]) * _rms(grad_h[active])
    h_relative = _rms(h_derivative[active]) / max(h_scale, float(scale_floor))
    poisson_symmetry = bool(lie_relative <= float(poisson_tol))
    preserves_hamiltonian = bool(h_relative <= float(hamiltonian_tol))

    charge_fit: ChargeFitReport | None = None
    if charge_gradients is not None:
        charge_fit = fit_charge_from_gradient_library(
            y,
            pi,
            charge_gradients,
            term_names=charge_term_names,
            mask=active,
            residual_tol=charge_tol,
        )
    hamiltonian_generator = bool(charge_fit is not None and charge_fit.accepted)
    hamiltonian_symmetry = bool(
        poisson_symmetry and preserves_hamiltonian and hamiltonian_generator
    )
    if hamiltonian_symmetry:
        classification = "hamiltonian_symmetry_with_sampled_local_charge"
    elif poisson_symmetry and preserves_hamiltonian:
        classification = "poisson_symmetry_preserving_H_charge_not_found"
    elif poisson_symmetry:
        classification = "poisson_symmetry_not_preserving_H"
    else:
        classification = "not_a_poisson_symmetry"
    return NoetherClassification(
        lie_derivative=lie,
        hamiltonian_derivative=h_derivative,
        lie_relative_residual=float(lie_relative),
        hamiltonian_relative_residual=float(h_relative),
        poisson_symmetry=poisson_symmetry,
        preserves_hamiltonian=preserves_hamiltonian,
        charge_fit=charge_fit,
        hamiltonian_generator=hamiltonian_generator,
        hamiltonian_symmetry=hamiltonian_symmetry,
        classification=classification,
        global_charge_proven=False,
        caveat=(
            "A sampled fit Y=Pi grad(G) supplies a local charge candidate only; "
            "global exactness and topology/Poisson-cohomology obstructions are not proven."
        ),
    )


def canonical_symplectic_matrix(n_dof: int) -> Array:
    """Canonical Poisson matrix in coordinate order ``(q, p)``."""

    n = int(n_dof)
    if n <= 0:
        raise ValueError("n_dof must be positive")
    out = np.zeros((2 * n, 2 * n), dtype=np.float64)
    out[:n, n:] = np.eye(n)
    out[n:, :n] = -np.eye(n)
    return out


def canonical_affine_momentum_map(
    linear_part: Array,
    translation: Array,
    state_points: Array,
    *,
    symplectic_tol: float = 1.0e-10,
) -> Array:
    """Canonical affine momentum map with a strict symplectic assertion.

    For ``Y(z)=M z+c`` and canonical ``J``, a quadratic charge exists only if
    ``M J + J M.T = 0`` (equivalently ``-J M`` is symmetric).  Unlike the old
    fast path, this function never silently symmetrizes a non-Hamiltonian
    generator.
    """

    m = _finite_array(linear_part, name="linear_part", ndim=2)
    c = _finite_array(translation, name="translation", ndim=1)
    z = _finite_array(state_points, name="state_points", ndim=2)
    if m.shape[0] != m.shape[1] or m.shape[0] % 2:
        raise ValueError("linear_part must be square with even dimension")
    d = m.shape[0]
    if c.shape != (d,) or z.shape[1] != d:
        raise ValueError("translation and state_points must match linear_part dimension")
    j = canonical_symplectic_matrix(d // 2)
    defect = m @ j + j @ m.T
    scale = max(np.linalg.norm(m) * np.linalg.norm(j), 1.0)
    if np.linalg.norm(defect) > float(symplectic_tol) * scale:
        raise ValueError(
            "affine generator is not canonical symplectic: "
            "M J + J M.T does not vanish"
        )
    quadratic = -j @ m
    symmetry_defect = quadratic - quadratic.T
    if np.linalg.norm(symmetry_defect) > float(symplectic_tol) * max(
        np.linalg.norm(quadratic), 1.0
    ):
        raise ValueError("-J M is not symmetric within tolerance")
    linear = -j @ c
    return 0.5 * np.einsum("ni,ij,nj->n", z, quadratic, z) + z @ linear


__all__ = [
    "ChargeFitReport",
    "NoetherClassification",
    "canonical_affine_momentum_map",
    "canonical_symplectic_matrix",
    "classify_noether_symmetry",
    "fit_charge_from_gradient_library",
    "poisson_lie_derivative",
]
