# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.

r"""Numerical and polynomial certificates for Poisson bivectors.

The routines in this module deliberately do not know how a bivector was
discovered.  They operate on evaluated tensors, or on polynomial coefficient
arrays, so the same gates can be used by the finite-dimensional polynomial
lanes and by later symbolic/AST lanes.

Conventions
-----------
``dPi[n, i, j, k]`` means :math:`\partial_k \Pi^{ij}` and Hamiltonian
vector fields use :math:`f=\Pi\nabla H`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isqrt
from typing import Dict, Mapping, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ScaledResidual:
    """Absolute and scale-free residual for one geometric identity."""

    rms: float
    scale: float
    relative: float
    passed: bool


@dataclass(frozen=True)
class RankProfile:
    """Numerical rank information on a collection of state-space points."""

    ranks: torch.Tensor
    singular_values: torch.Tensor
    generic_rank: int
    stable_fraction: float
    even: bool
    stable: bool
    accepted: bool
    threshold_relative: float
    threshold_absolute: float


@dataclass(frozen=True)
class PolynomialJacobiCertificate:
    """Global coefficient-space Jacobi certificate for a polynomial tensor.

    ``coefficients`` maps ``(i, j, k, exponent_tuple)`` to the corresponding
    coefficient of the Jacobiator.  The state polynomial is identically zero
    exactly when all of these values vanish.  With floating learned bivector
    coefficients this remains a numerical coefficient certificate; exact
    arithmetic requires a subsequent rational/algebraic snap.
    """

    coefficients: Mapping[Tuple[int, int, int, Tuple[int, ...]], torch.Tensor]
    rms: float
    max_abs: float
    tolerance: float
    passed: bool


def _validate_bivector_evaluation(
    Pi: torch.Tensor, dPi: torch.Tensor | None = None
) -> Tuple[int, int]:
    if Pi.ndim != 3 or Pi.shape[1] != Pi.shape[2]:
        raise ValueError(f"Pi must have shape (N,d,d), got {tuple(Pi.shape)}")
    n, d, _ = Pi.shape
    if dPi is not None and tuple(dPi.shape) != (n, d, d, d):
        raise ValueError(
            "dPi must have shape (N,d,d,d) with derivative index last; "
            f"got {tuple(dPi.shape)}"
        )
    return n, d


def skew_residual(
    Pi: torch.Tensor,
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
    eps: float = 1.0e-30,
) -> ScaledResidual:
    r"""Certify :math:`\Pi+\Pi^T=0`."""

    _validate_bivector_evaluation(Pi)
    error = Pi + Pi.transpose(-1, -2)
    rms = float(error.square().mean().sqrt().item())
    scale = float(Pi.square().mean().sqrt().item())
    relative = rms / max(scale, eps)
    passed = rms <= absolute_tolerance or relative <= relative_tolerance
    return ScaledResidual(rms=rms, scale=scale, relative=relative, passed=passed)


def jacobiator(Pi: torch.Tensor, dPi: torch.Tensor) -> torch.Tensor:
    """Evaluate independent components of the Schouten bracket ``[Pi, Pi]``.

    Returns a tensor with shape ``(N, binom(d, 3))`` in lexicographic
    ``(i,j,k)`` order.  In dimensions below three there are no independent
    Jacobi equations and the second dimension is zero.
    """

    n, d = _validate_bivector_evaluation(Pi, dPi)
    triples = tuple(combinations(range(d), 3))
    if not triples:
        return Pi.new_zeros((n, 0))

    out = []
    for i, j, k in triples:
        # Sum_l Pi^{i l} d_l Pi^{j k} plus cyclic permutations.
        term = torch.einsum("nl,nl->n", Pi[:, i, :], dPi[:, j, k, :])
        term = term + torch.einsum("nl,nl->n", Pi[:, j, :], dPi[:, k, i, :])
        term = term + torch.einsum("nl,nl->n", Pi[:, k, :], dPi[:, i, j, :])
        out.append(term)
    return torch.stack(out, dim=1)


def jacobi_residual(
    Pi: torch.Tensor,
    dPi: torch.Tensor,
    *,
    relative_tolerance: float = 1.0e-7,
    absolute_tolerance: float = 1.0e-10,
    eps: float = 1.0e-30,
) -> ScaledResidual:
    """Return an absolute and dimensionally scaled Jacobi residual.

    The natural scale of the Jacobiator is ``rms(Pi) * rms(dPi)``.  Constant
    tensors have zero derivative scale and are accepted through the absolute
    gate rather than producing an ill-defined relative score.
    """

    J = jacobiator(Pi, dPi)
    if J.numel() == 0:
        return ScaledResidual(rms=0.0, scale=0.0, relative=0.0, passed=True)
    rms = float(J.square().mean().sqrt().item())
    pi_scale = float(Pi.square().mean().sqrt().item())
    dpi_scale = float(dPi.square().mean().sqrt().item())
    scale = pi_scale * dpi_scale
    relative = 0.0 if rms == 0.0 else rms / max(scale, eps)
    passed = rms <= absolute_tolerance or (
        scale > eps and relative <= relative_tolerance
    )
    return ScaledResidual(rms=rms, scale=scale, relative=relative, passed=passed)


def invariance_residual(
    Pi: torch.Tensor,
    dPi: torch.Tensor,
    field_values: torch.Tensor,
    field_jacobians: torch.Tensor,
    *,
    relative_tolerance: float = 1.0e-7,
    absolute_tolerance: float = 1.0e-10,
    eps: float = 1.0e-30,
) -> ScaledResidual:
    """Certify ``L_f Pi = 0`` using all independent upper-triangle entries."""

    n, d = _validate_bivector_evaluation(Pi, dPi)
    if tuple(field_values.shape) != (n, d):
        raise ValueError("field_values must have shape (N,d)")
    if tuple(field_jacobians.shape) != (n, d, d):
        raise ValueError("field_jacobians must have shape (N,d,d)")

    # f^k d_k Pi^{ij} - (d_k f^i) Pi^{kj}
    #                         - (d_k f^j) Pi^{ik}
    advected = torch.einsum("nk,nijk->nij", field_values, dPi)
    left = torch.einsum("nik,nkj->nij", field_jacobians, Pi)
    right = torch.einsum("nik,njk->nij", Pi, field_jacobians)
    lie = advected - left - right
    pairs = torch.triu_indices(d, d, offset=1, device=Pi.device)
    independent = lie[:, pairs[0], pairs[1]]
    rms = float(independent.square().mean().sqrt().item())

    f_scale = float(field_values.square().mean().sqrt().item())
    df_scale = float(field_jacobians.square().mean().sqrt().item())
    pi_scale = float(Pi.square().mean().sqrt().item())
    dpi_scale = float(dPi.square().mean().sqrt().item())
    scale = f_scale * dpi_scale + df_scale * pi_scale
    relative = 0.0 if rms == 0.0 else rms / max(scale, eps)
    passed = rms <= absolute_tolerance or (
        scale > eps and relative <= relative_tolerance
    )
    return ScaledResidual(rms=rms, scale=scale, relative=relative, passed=passed)


def rank_profile(
    Pi: torch.Tensor,
    *,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
    minimum_stable_fraction: float = 0.9,
) -> RankProfile:
    """Estimate the generic rank and flag unstable/rank-changing samples.

    The modal numerical rank is reported as the generic rank.  This is robust
    to a small number of singular probes while still exposing the full rank
    vector so callers can split genuinely rank-changing domains into patches.
    """

    n, _ = _validate_bivector_evaluation(Pi)
    if n == 0:
        raise ValueError("rank_profile needs at least one sample")
    singular_values = torch.linalg.svdvals(Pi)
    thresholds = absolute_tolerance + relative_tolerance * singular_values[:, :1]
    ranks = (singular_values > thresholds).sum(dim=1)
    unique, counts = torch.unique(ranks, return_counts=True)
    best = int(torch.argmax(counts).item())
    generic_rank = int(unique[best].item())
    stable_fraction = float(counts[best].item()) / float(n)
    even = bool(torch.all(torch.remainder(ranks, 2) == 0).item())
    stable = stable_fraction >= minimum_stable_fraction
    return RankProfile(
        ranks=ranks,
        singular_values=singular_values,
        generic_rank=generic_rank,
        stable_fraction=stable_fraction,
        even=even,
        stable=stable,
        accepted=even and stable,
        threshold_relative=relative_tolerance,
        threshold_absolute=absolute_tolerance,
    )


def _infer_state_dim_from_pair_count(pair_count: int) -> int:
    # pair_count = d(d-1)/2 => d = (1 + sqrt(1+8p))/2
    disc = 1 + 8 * pair_count
    root = isqrt(disc)
    if root * root != disc or (1 + root) % 2:
        raise ValueError(f"{pair_count} is not a triangular pair count")
    return (1 + root) // 2


def _ordered_polynomial_entry(
    coefficients: torch.Tensor, i: int, j: int
) -> Tuple[torch.Tensor, int]:
    """Return coefficients of Pi^{ij} and its pair-orientation sign."""

    if i == j:
        return coefficients.new_zeros(coefficients.shape[1]), 0
    d = _infer_state_dim_from_pair_count(int(coefficients.shape[0]))
    pairs = tuple(combinations(range(d), 2))
    lookup = {pair: p for p, pair in enumerate(pairs)}
    if i < j:
        return coefficients[lookup[(i, j)]], 1
    return coefficients[lookup[(j, i)]], -1


def polynomial_jacobi_coefficients(
    coefficients: torch.Tensor,
    exponents: Sequence[Sequence[int]],
) -> Dict[Tuple[int, int, int, Tuple[int, ...]], torch.Tensor]:
    """Assemble all polynomial Jacobiator coefficients by exact convolution.

    Parameters
    ----------
    coefficients:
        Pair-major coefficient matrix ``(binom(d,2), K)``.  A flattened
        pair-major vector is also accepted.
    exponents:
        Length-``K`` monomial exponent tuples.  Constant, affine, linear and
        quadratic total-degree bases are all handled by the same routine.

    Notes
    -----
    No state-space collocation is used.  Integer monomial differentiation and
    exponent convolution are exact; only the supplied learned coefficients
    retain their floating-point uncertainty.
    """

    exponents_t = tuple(tuple(int(v) for v in alpha) for alpha in exponents)
    if not exponents_t:
        raise ValueError("exponents must not be empty")
    d = len(exponents_t[0])
    if d < 1 or any(len(alpha) != d for alpha in exponents_t):
        raise ValueError("all exponent tuples must have the same state dimension")
    k_terms = len(exponents_t)
    pair_count = d * (d - 1) // 2
    if coefficients.ndim == 1:
        if coefficients.numel() != pair_count * k_terms:
            raise ValueError("flattened coefficients have incompatible size")
        coefficients = coefficients.reshape(pair_count, k_terms)
    if tuple(coefficients.shape) != (pair_count, k_terms):
        raise ValueError(
            f"coefficients must have shape {(pair_count, k_terms)}, "
            f"got {tuple(coefficients.shape)}"
        )

    output: Dict[Tuple[int, int, int, Tuple[int, ...]], torch.Tensor] = {}

    def accumulate_product(
        triple: Tuple[int, int, int],
        first: Tuple[int, int],
        second: Tuple[int, int],
        derivative_axis: int,
    ) -> None:
        first_coeff, first_sign = _ordered_polynomial_entry(
            coefficients, first[0], first[1]
        )
        second_coeff, second_sign = _ordered_polynomial_entry(
            coefficients, second[0], second[1]
        )
        if first_sign == 0 or second_sign == 0:
            return
        sign = first_sign * second_sign
        for a, alpha in enumerate(exponents_t):
            ca = first_coeff[a]
            for b, beta in enumerate(exponents_t):
                derivative_power = beta[derivative_axis]
                if derivative_power == 0:
                    continue
                gamma = list(alpha)
                for axis in range(d):
                    gamma[axis] += beta[axis]
                gamma[derivative_axis] -= 1
                key = (*triple, tuple(gamma))
                value = sign * derivative_power * ca * second_coeff[b]
                output[key] = output.get(key, coefficients.new_zeros(())) + value

    for triple in combinations(range(d), 3):
        i, j, k = triple
        for ell in range(d):
            accumulate_product(triple, (i, ell), (j, k), ell)
            accumulate_product(triple, (j, ell), (k, i), ell)
            accumulate_product(triple, (k, ell), (i, j), ell)

    # Retain structurally generated zero coefficients: they are useful for a
    # deterministic certificate/report and remain differentiable in b.
    return output


def polynomial_jacobi_certificate(
    coefficients: torch.Tensor,
    exponents: Sequence[Sequence[int]],
    *,
    tolerance: float = 1.0e-10,
) -> PolynomialJacobiCertificate:
    """Certify polynomial Jacobi globally in coefficient space."""

    values = polynomial_jacobi_coefficients(coefficients, exponents)
    if not values:
        return PolynomialJacobiCertificate(
            coefficients=values,
            rms=0.0,
            max_abs=0.0,
            tolerance=tolerance,
            passed=True,
        )
    stacked = torch.stack(tuple(values.values()))
    rms = float(stacked.square().mean().sqrt().item())
    max_abs = float(stacked.abs().max().item())
    return PolynomialJacobiCertificate(
        coefficients=values,
        rms=rms,
        max_abs=max_abs,
        tolerance=tolerance,
        passed=max_abs <= tolerance,
    )
