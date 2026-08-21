# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Exact rank-two Poisson/Nambu reconstruction in three dimensions.

This module is deliberately independent of the symbolic-regression backend.
It consumes sampled scalar integrals (their values and gradients) and sampled
vector-field values.  A future ``poisson_core.VectorField`` or AST expression
can provide those arrays without changing the geometry API here.

On a regular three-dimensional patch, a Poisson tensor can be identified with
a vector ``J`` through ``Pi @ v = J x v``.  The representation

``J = mu * grad(C)``

satisfies Jacobi by construction.  Consequently

``f = Pi grad(H) = mu * grad(C) x grad(H)``.

The reconstruction below fits ``mu`` either pointwise or in a caller-provided
linear feature library.  It explicitly masks points where the two integral
gradients cease to be independent; no division by a vanishing cross product is
silently performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ScalarIntegralSamples:
    """Sampled scalar integral and its state gradient.

    Parameters
    ----------
    values:
        Integral values with shape ``(n_samples,)``.
    gradients:
        Gradients with shape ``(n_samples, 3)``.
    name:
        Human-readable expression or label used in reports.
    complexity:
        Optional symbolic-complexity score.  It is metadata only and does not
        affect reconstruction.
    """

    values: Array
    gradients: Array
    name: str = "I"
    complexity: float = 0.0


@dataclass(frozen=True)
class MultiplierFit:
    """Linear or pointwise fit of the Nambu multiplier ``mu``."""

    mode: str
    pointwise: Array
    fitted: Array
    coefficients: Array | None
    feature_names: tuple[str, ...]
    relative_residual: float
    condition_number: float


@dataclass(frozen=True)
class Nambu3DReport:
    """Certificate and sampled tensor returned by :func:`reconstruct_nambu_3d`."""

    casimir_name: str
    hamiltonian_name: str
    regular_mask: Array
    singular_mask: Array
    gradient_cross_norm: Array
    multiplier: MultiplierFit
    poisson_vector: Array
    poisson_tensor: Array
    reconstructed_field: Array
    first_integral_relative_residuals: tuple[float, float]
    reconstruction_relative_residual: float
    reconstruction_relative_residual_regular: float
    regular_fraction: float
    generic_rank: int
    rank_two_fraction: float
    jacobi_by_construction: bool
    jacobi_certification: str
    jacobi_statement: str
    accepted: bool


def _as_float_array(value: Array, *, name: str) -> Array:
    out = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _validate_integral(integral: ScalarIntegralSamples, n_samples: int) -> tuple[Array, Array]:
    values = _as_float_array(integral.values, name=f"{integral.name}.values")
    gradients = _as_float_array(integral.gradients, name=f"{integral.name}.gradients")
    if values.shape != (n_samples,):
        raise ValueError(
            f"{integral.name}.values must have shape ({n_samples},), got {values.shape}"
        )
    if gradients.shape != (n_samples, 3):
        raise ValueError(
            f"{integral.name}.gradients must have shape ({n_samples}, 3), "
            f"got {gradients.shape}"
        )
    return values, gradients


def cross_product_matrices(vectors: Array) -> Array:
    """Return skew matrices ``P`` satisfying ``P @ u = vectors x u``.

    ``vectors`` may have shape ``(3,)`` or ``(..., 3)``.  The returned array
    has shape ``(3, 3)`` or ``(..., 3, 3)`` respectively.
    """

    # NaNs are preserved deliberately: reconstruction reports use them to mark
    # singular samples whose tensor is unresolved.  Callers can use the
    # accompanying regular/singular mask rather than mistaking a zero fill for
    # a rank-zero tensor.
    v = np.asarray(vectors, dtype=np.float64)
    if np.any(np.isinf(v)):
        raise ValueError("vectors contains infinite values")
    if v.shape[-1:] != (3,):
        raise ValueError(f"vectors must end in dimension 3, got {v.shape}")
    out = np.zeros(v.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def fit_nambu_multiplier(
    vector_field: Array,
    carrier: Array,
    regular_mask: Array,
    *,
    features: Array | None = None,
    feature_names: Sequence[str] | None = None,
    rcond: float | None = None,
) -> MultiplierFit:
    """Fit ``f = mu * carrier`` on the regular patch.

    With ``features=None``, the multiplier is estimated independently at every
    regular sample.  This is the exact sampled reconstruction and is useful as
    a pseudo-target for symbolic regression.  If ``features`` has shape
    ``(N, K)``, a global linear model ``mu = features @ coefficients`` is fit
    directly against all vector components, rather than first fitting noisy
    pointwise ratios.
    """

    f = _as_float_array(vector_field, name="vector_field")
    w = _as_float_array(carrier, name="carrier")
    mask = np.asarray(regular_mask, dtype=bool)
    if f.shape != w.shape or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("vector_field and carrier must both have shape (N, 3)")
    if mask.shape != (f.shape[0],):
        raise ValueError(f"regular_mask must have shape ({f.shape[0]},)")
    if not np.any(mask):
        raise ValueError("no regular samples remain after the independence mask")

    denom = np.einsum("ni,ni->n", w, w)
    pointwise = np.full(f.shape[0], np.nan, dtype=np.float64)
    pointwise[mask] = np.einsum("ni,ni->n", f[mask], w[mask]) / denom[mask]

    if features is None:
        fitted = pointwise.copy()
        reconstruction = fitted[mask, None] * w[mask]
        residual = np.linalg.norm(f[mask] - reconstruction)
        scale = max(np.linalg.norm(f[mask]), np.finfo(np.float64).tiny)
        return MultiplierFit(
            mode="pointwise",
            pointwise=pointwise,
            fitted=fitted,
            coefficients=None,
            feature_names=(),
            relative_residual=float(residual / scale),
            condition_number=1.0,
        )

    phi = _as_float_array(features, name="features")
    if phi.ndim == 1:
        phi = phi[:, None]
    if phi.shape[0] != f.shape[0]:
        raise ValueError(f"features must have {f.shape[0]} rows, got {phi.shape}")
    if phi.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    if feature_names is None:
        names = tuple(f"psi_{k}" for k in range(phi.shape[1]))
    else:
        names = tuple(str(name) for name in feature_names)
        if len(names) != phi.shape[1]:
            raise ValueError("feature_names length must equal the feature column count")

    # A[n, i, k] = carrier[n, i] * phi[n, k].  Flattening creates the
    # least-squares system for all three field components simultaneously.
    design = np.einsum("ni,nk->nik", w[mask], phi[mask]).reshape(-1, phi.shape[1])
    target = f[mask].reshape(-1)
    coeffs, _, _, singular_values = np.linalg.lstsq(design, target, rcond=rcond)
    fitted = phi @ coeffs
    reconstruction = fitted[mask, None] * w[mask]
    residual = np.linalg.norm(f[mask] - reconstruction)
    scale = max(np.linalg.norm(f[mask]), np.finfo(np.float64).tiny)
    if singular_values.size == 0 or singular_values[-1] <= 0.0:
        condition = float("inf")
    else:
        condition = float(singular_values[0] / singular_values[-1])
    return MultiplierFit(
        mode="linear_features",
        pointwise=pointwise,
        fitted=fitted,
        coefficients=coeffs,
        feature_names=names,
        relative_residual=float(residual / scale),
        condition_number=condition,
    )


def _relative_scalar_residual(values: Array, scale_factors: Array) -> float:
    numerator = float(np.sqrt(np.mean(np.square(values))))
    denominator = float(np.sqrt(np.mean(np.square(scale_factors))))
    return numerator / max(denominator, np.finfo(np.float64).tiny)


def reconstruct_nambu_3d(
    vector_field: Array,
    casimir: ScalarIntegralSamples,
    hamiltonian: ScalarIntegralSamples,
    *,
    multiplier_features: Array | None = None,
    multiplier_feature_names: Sequence[str] | None = None,
    multiplier_features_are_differentiable: bool = False,
    independence_rtol: float = 1.0e-10,
    independence_atol: float = 1.0e-12,
    field_zero_atol: float = 1.0e-14,
    reconstruction_tol: float = 1.0e-8,
    integral_tol: float = 1.0e-8,
    minimum_regular_fraction: float = 0.5,
    minimum_rank_two_fraction: float = 0.5,
) -> Nambu3DReport:
    """Reconstruct ``f = mu grad(C) x grad(H)`` and its Poisson tensor.

    The caller chooses which integral is interpreted as the Casimir and which
    as the Hamiltonian.  Swapping them is a valid alternative candidate and
    reverses the fitted multiplier.  Assignment should normally use units,
    cross-experiment variation, and symbolic complexity outside this routine.

    The Jacobi statement is structural: every differentiable field
    ``J=mu*grad(C)`` obeys ``J dot curl(J)=0``.  Pointwise pseudo-targets still
    require a differentiable symbolic fit of ``mu`` before this becomes a
    between-sample tensor certificate. Sampled feature matrices carry the same
    requirement, expressed by ``multiplier_features_are_differentiable``.
    """

    f = _as_float_array(vector_field, name="vector_field")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"vector_field must have shape (N, 3), got {f.shape}")
    for name, value in (
        ("minimum_regular_fraction", minimum_regular_fraction),
        ("minimum_rank_two_fraction", minimum_rank_two_fraction),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    n_samples = f.shape[0]
    _, grad_c = _validate_integral(casimir, n_samples)
    _, grad_h = _validate_integral(hamiltonian, n_samples)

    carrier = np.cross(grad_c, grad_h)
    carrier_norm = np.linalg.norm(carrier, axis=1)
    independence_scale = np.linalg.norm(grad_c, axis=1) * np.linalg.norm(grad_h, axis=1)
    threshold = float(independence_atol) + float(independence_rtol) * independence_scale
    finite = np.isfinite(carrier_norm) & np.isfinite(independence_scale)
    regular_mask = finite & (carrier_norm > threshold)
    singular_mask = ~regular_mask

    multiplier = fit_nambu_multiplier(
        f,
        carrier,
        regular_mask,
        features=multiplier_features,
        feature_names=multiplier_feature_names,
    )
    reconstructed = np.full_like(f, np.nan)
    reconstructed[regular_mask] = (
        multiplier.fitted[regular_mask, None] * carrier[regular_mask]
    )
    # A zero field is represented without relying on an undefined multiplier at
    # a singular point.  Nonzero singular samples remain NaN and are reported.
    singular_zero_field = singular_mask & (np.linalg.norm(f, axis=1) <= field_zero_atol)
    reconstructed[singular_zero_field] = 0.0

    poisson_vector = np.full_like(f, np.nan)
    poisson_vector[regular_mask] = (
        multiplier.fitted[regular_mask, None] * grad_c[regular_mask]
    )
    poisson_vector[singular_zero_field] = 0.0
    poisson_tensor = cross_product_matrices(poisson_vector)

    valid_reconstruction = np.all(np.isfinite(reconstructed), axis=1)
    if np.any(valid_reconstruction):
        total_residual = np.linalg.norm(f[valid_reconstruction] - reconstructed[valid_reconstruction])
        total_scale = max(
            np.linalg.norm(f[valid_reconstruction]), np.finfo(np.float64).tiny
        )
        reconstruction_relative = float(total_residual / total_scale)
    else:
        reconstruction_relative = float("inf")
    regular_residual = np.linalg.norm(f[regular_mask] - reconstructed[regular_mask])
    regular_scale = max(np.linalg.norm(f[regular_mask]), np.finfo(np.float64).tiny)
    regular_reconstruction_relative = float(regular_residual / regular_scale)

    f_norm = np.linalg.norm(f, axis=1)
    c_drift = np.einsum("ni,ni->n", f, grad_c)
    h_drift = np.einsum("ni,ni->n", f, grad_h)
    c_scale = f_norm * np.linalg.norm(grad_c, axis=1)
    h_scale = f_norm * np.linalg.norm(grad_h, axis=1)
    integral_residuals = (
        _relative_scalar_residual(c_drift, c_scale),
        _relative_scalar_residual(h_drift, h_scale),
    )

    regular_fraction = float(np.mean(regular_mask))
    nonzero_j = regular_mask & (np.linalg.norm(poisson_vector, axis=1) > independence_atol)
    regular_count = max(int(np.count_nonzero(regular_mask)), 1)
    rank_two_fraction = float(np.count_nonzero(nonzero_j) / regular_count)
    generic_rank = 2 if np.any(nonzero_j) else 0
    jacobi_by_construction = bool(
        multiplier.mode == "linear_features"
        and multiplier_features_are_differentiable
    )
    accepted = bool(
        regular_fraction >= float(minimum_regular_fraction)
        and generic_rank == 2
        and rank_two_fraction >= float(minimum_rank_two_fraction)
        and jacobi_by_construction
        and regular_reconstruction_relative <= float(reconstruction_tol)
        and max(integral_residuals) <= float(integral_tol)
    )
    if jacobi_by_construction:
        jacobi_certification = "differentiable_multiplier_form"
        jacobi_statement = (
            "J=mu*grad(C) implies J·curl(J)=0 on every differentiable regular patch."
        )
    else:
        jacobi_certification = (
            "pointwise_multiplier_pseudotarget_only"
            if multiplier.mode == "pointwise"
            else "sampled_feature_multiplier_without_differentiability_contract"
        )
        jacobi_statement = (
            "The sampled multiplier reconstructs the observed vectors but does not "
            "carry a differentiable feature contract; Jacobi between samples is not "
            "yet certified."
        )
    return Nambu3DReport(
        casimir_name=str(casimir.name),
        hamiltonian_name=str(hamiltonian.name),
        regular_mask=regular_mask,
        singular_mask=singular_mask,
        gradient_cross_norm=carrier_norm,
        multiplier=multiplier,
        poisson_vector=poisson_vector,
        poisson_tensor=poisson_tensor,
        reconstructed_field=reconstructed,
        first_integral_relative_residuals=integral_residuals,
        reconstruction_relative_residual=reconstruction_relative,
        reconstruction_relative_residual_regular=regular_reconstruction_relative,
        regular_fraction=regular_fraction,
        generic_rank=generic_rank,
        rank_two_fraction=rank_two_fraction,
        jacobi_by_construction=jacobi_by_construction,
        jacobi_certification=jacobi_certification,
        jacobi_statement=jacobi_statement,
        accepted=accepted,
    )


__all__ = [
    "MultiplierFit",
    "Nambu3DReport",
    "ScalarIntegralSamples",
    "cross_product_matrices",
    "fit_nambu_multiplier",
    "reconstruct_nambu_3d",
]
