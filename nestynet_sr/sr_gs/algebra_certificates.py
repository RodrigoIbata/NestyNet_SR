# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Certificates for affine graph-symmetry algebras.

The determining operator recovers a linear subspace.  This module decides what
that subspace is allowed to do next: remain audit-only, or become eligible for
quotient/reduction construction.  Descriptive labels are reported for humans but
do not control acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

_EPS = 1.0e-12


@dataclass(frozen=True)
class BracketCertificate:
    """One Lie-bracket closure check for two basis vectors."""

    i: int
    j: int
    bracket_norm: float
    outside_span_norm: float
    residual_rel: float
    output_alpha: float
    output_beta: float
    bracket_coeffs: tuple[float, ...]

    def to_report(self) -> dict[str, Any]:
        return {
            "i": int(self.i),
            "j": int(self.j),
            "bracket_norm": float(self.bracket_norm),
            "outside_span_norm": float(self.outside_span_norm),
            "residual_rel": float(self.residual_rel),
            "output_alpha": float(self.output_alpha),
            "output_beta": float(self.output_beta),
            "bracket_coeffs": [float(v) for v in self.bracket_coeffs],
        }


@dataclass(frozen=True)
class AffineAlgebraCertificate:
    """Audit and promotion certificate for an affine graph-symmetry subspace."""

    is_closed: bool
    quotient_ready: bool
    quotient_policy: str
    bracket_closure_residual: float
    bracket_records: tuple[BracketCertificate, ...]
    distribution_rank: int
    orbit_dimension: int
    quotient_codimension: int
    pointwise_rank_histogram: dict[int, int]
    train_residual_rel: float
    heldout_residual_rel: float
    heldout_verified: bool
    spectral_gap: float
    condition_number: float
    bootstrap_max_principal_angle: float
    subspace_stable: bool
    dimensionally_consistent: bool
    dimensional_warnings: tuple[str, ...] = ()
    classifications: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "is_closed": bool(self.is_closed),
            "quotient_ready": bool(self.quotient_ready),
            "quotient_policy": self.quotient_policy,
            "bracket_closure_residual": float(self.bracket_closure_residual),
            "brackets": [r.to_report() for r in self.bracket_records],
            "distribution_rank": int(self.distribution_rank),
            "orbit_dimension": int(self.orbit_dimension),
            "quotient_codimension": int(self.quotient_codimension),
            "pointwise_rank_histogram": {str(k): int(v) for k, v in self.pointwise_rank_histogram.items()},
            "train_residual_rel": float(self.train_residual_rel),
            "heldout_residual_rel": float(self.heldout_residual_rel),
            "heldout_verified": bool(self.heldout_verified),
            "spectral_gap": float(self.spectral_gap),
            "condition_number": float(self.condition_number),
            "bootstrap_max_principal_angle": float(self.bootstrap_max_principal_angle),
            "subspace_stable": bool(self.subspace_stable),
            "dimensionally_consistent": bool(self.dimensionally_consistent),
            "dimensional_warnings": list(self.dimensional_warnings),
            "classifications": list(self.classifications),
            "evidence": dict(self.evidence),
        }


def certify_affine_algebra(
    *,
    basis: np.ndarray,
    projector: np.ndarray,
    input_dim: int,
    singular_values: Sequence[float],
    independent_row_count: int,
    unknown_count: int,
    train_residual_rel: float,
    heldout_residual_rel: float,
    acceptance_residual_tol: float,
    structurally_underdetermined: bool,
    distribution_rank: int,
    pointwise_distribution_ranks: Sequence[int],
    bootstrap_principal_angles: Sequence[float] = (),
    dimensional_warnings: Sequence[str] = (),
    closure_tol: float = 1.0e-6,
    bootstrap_angle_tol: float = 5.0e-2,
    heldout_consistency_factor: float | None = None,
) -> AffineAlgebraCertificate:
    """Build closure, stability, and quotient-readiness diagnostics.

    ``heldout_consistency_factor`` (noise-calibrated mode) additionally
    accepts held-out verification when the held-out residual is consistent
    with the train residual (``heldout <= factor * train``), independent of
    the absolute ``acceptance_residual_tol`` that only oracle-exact gradients
    can meet.  ``None`` keeps the absolute-only behavior.
    """

    B = np.asarray(basis, dtype=float)
    P = np.asarray(projector, dtype=float)
    n = int(input_dim)
    nullity = int(B.shape[1]) if B.ndim == 2 else 0
    bracket_records = _bracket_records(B, P, n)
    closure_residual = max((r.residual_rel for r in bracket_records), default=0.0)
    is_closed = bool(closure_residual <= float(closure_tol))
    train_ok = float(train_residual_rel) <= float(acceptance_residual_tol)
    heldout_ok = float(heldout_residual_rel) <= max(10.0 * float(acceptance_residual_tol), float(acceptance_residual_tol))
    heldout_verified = bool(train_ok and heldout_ok)
    if not heldout_verified and heldout_consistency_factor is not None:
        train = float(train_residual_rel)
        heldout = float(heldout_residual_rel)
        heldout_verified = bool(
            math.isfinite(train)
            and math.isfinite(heldout)
            and heldout <= max(float(heldout_consistency_factor) * train, float(acceptance_residual_tol))
        )
    bootstrap_max = max((float(v) for v in bootstrap_principal_angles), default=0.0)
    subspace_stable = bool(bootstrap_max <= float(bootstrap_angle_tol))
    dimensionally_consistent = not tuple(dimensional_warnings)
    rank_hist = _rank_histogram(pointwise_distribution_ranks)
    orbit_dimension = int(distribution_rank)
    quotient_codimension = max(0, int(input_dim) - orbit_dimension)
    spectral_gap, condition_number = _spectrum_diagnostics(singular_values, independent_row_count, unknown_count)
    classifications = tuple(classify_affine_generator(B[:, i], n) for i in range(nullity)) if nullity else ()

    quotient_ready = bool(
        nullity > 0
        and not structurally_underdetermined
        and is_closed
        and heldout_verified
        and subspace_stable
        and dimensionally_consistent
        and orbit_dimension > 0
    )
    if quotient_ready:
        policy = "quotient_ready"
    elif nullity <= 0:
        policy = "audit_only_no_nullspace"
    elif structurally_underdetermined:
        policy = "audit_only_structurally_underdetermined"
    elif not is_closed:
        policy = "reject_for_quotient_nonclosed"
    elif not heldout_verified:
        policy = "audit_only_residual_not_verified"
    elif not subspace_stable:
        policy = "audit_only_unstable_subspace"
    elif not dimensionally_consistent:
        policy = "audit_only_dimensionally_questionable"
    elif orbit_dimension <= 0:
        policy = "audit_only_zero_distribution"
    else:
        policy = "audit_only"

    return AffineAlgebraCertificate(
        is_closed=is_closed,
        quotient_ready=quotient_ready,
        quotient_policy=policy,
        bracket_closure_residual=float(closure_residual),
        bracket_records=tuple(bracket_records),
        distribution_rank=orbit_dimension,
        orbit_dimension=orbit_dimension,
        quotient_codimension=quotient_codimension,
        pointwise_rank_histogram=rank_hist,
        train_residual_rel=float(train_residual_rel),
        heldout_residual_rel=float(heldout_residual_rel),
        heldout_verified=heldout_verified,
        spectral_gap=float(spectral_gap),
        condition_number=float(condition_number),
        bootstrap_max_principal_angle=float(bootstrap_max),
        subspace_stable=subspace_stable,
        dimensionally_consistent=dimensionally_consistent,
        dimensional_warnings=tuple(str(w) for w in dimensional_warnings),
        classifications=classifications,
        evidence={
            "nullity": int(nullity),
            "unknown_count": int(unknown_count),
            "independent_row_count": int(independent_row_count),
            "closure_tol": float(closure_tol),
            "bootstrap_angle_tol": float(bootstrap_angle_tol),
            "acceptance_residual_tol": float(acceptance_residual_tol),
        },
    )


def affine_graph_bracket_coeffs(c1: np.ndarray, c2: np.ndarray, input_dim: int) -> np.ndarray:
    """Lie bracket coefficients for affine graph vector fields.

    Coefficient ordering is ``A.reshape(-1), b, alpha, beta`` for
    ``(A x + b).d_x + (alpha + beta*y).d_y``.
    """

    n = int(input_dim)
    A1, b1, alpha1, beta1 = unpack_affine_graph_coeffs(c1, n)
    A2, b2, alpha2, beta2 = unpack_affine_graph_coeffs(c2, n)
    A = A2 @ A1 - A1 @ A2
    b = A2 @ b1 - A1 @ b2
    alpha = beta2 * alpha1 - beta1 * alpha2
    beta = 0.0
    return np.concatenate([A.reshape(-1), b, np.asarray([alpha, beta], dtype=float)])


def classify_affine_generator(coeffs: np.ndarray, input_dim: int, *, tol: float = 1.0e-8) -> str:
    """Return a descriptive label for one affine graph generator."""

    A, b, alpha, beta = unpack_affine_graph_coeffs(coeffs, int(input_dim))
    has_A = np.linalg.norm(A) > tol
    has_b = np.linalg.norm(b) > tol
    has_output = abs(float(alpha)) > tol or abs(float(beta)) > tol
    if not has_A and not has_b and not has_output:
        return "zero"
    if not has_A and not has_b:
        if abs(float(alpha)) > tol and abs(float(beta)) <= tol:
            return "output_translation"
        if abs(float(beta)) > tol and abs(float(alpha)) <= tol:
            return "output_scaling"
        return "output_affine"
    input_label = _classify_input_affine_part(A, b, tol=tol)
    if has_output:
        return f"{input_label}+output_action"
    return input_label


def unpack_affine_graph_coeffs(coeffs: np.ndarray, input_dim: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    n = int(input_dim)
    expected = n * n + n + 2
    if c.size != expected:
        raise ValueError(f"coefficient vector length {c.size} != expected {expected}")
    A = c[: n * n].reshape(n, n)
    b = c[n * n : n * n + n]
    return A, b, float(c[-2]), float(c[-1])


def _bracket_records(B: np.ndarray, projector: np.ndarray, input_dim: int) -> list[BracketCertificate]:
    if B.ndim != 2 or B.shape[1] <= 1:
        return []
    records = []
    for i in range(B.shape[1]):
        for j in range(i + 1, B.shape[1]):
            bracket = affine_graph_bracket_coeffs(B[:, i], B[:, j], input_dim)
            norm = float(np.linalg.norm(bracket))
            if norm <= _EPS:
                outside = 0.0
                residual = 0.0
            else:
                outside_vec = bracket - projector @ bracket
                outside = float(np.linalg.norm(outside_vec))
                residual = float(outside / norm)
            _A, _b, alpha, beta = unpack_affine_graph_coeffs(bracket, input_dim)
            records.append(
                BracketCertificate(
                    i=int(i),
                    j=int(j),
                    bracket_norm=norm,
                    outside_span_norm=outside,
                    residual_rel=residual,
                    output_alpha=float(alpha),
                    output_beta=float(beta),
                    bracket_coeffs=tuple(float(v) for v in bracket),
                )
            )
    return records


def _classify_input_affine_part(A: np.ndarray, b: np.ndarray, *, tol: float) -> str:
    has_A = np.linalg.norm(A) > tol
    has_b = np.linalg.norm(b) > tol
    if not has_A and has_b:
        return "translation"
    if has_A and has_b:
        return "mixed_affine"
    diag = np.diag(np.diag(A))
    offdiag = A - diag
    if np.linalg.norm(offdiag) <= tol:
        return "diagonal_scaling"
    skew_resid = np.linalg.norm(A + A.T)
    sym_resid = np.linalg.norm(A - A.T)
    if skew_resid <= max(tol, 1.0e-6 * max(1.0, float(np.linalg.norm(A)))):
        return "rotation"
    if sym_resid <= max(tol, 1.0e-6 * max(1.0, float(np.linalg.norm(A)))):
        return "symmetric_linear"
    if np.count_nonzero(np.abs(offdiag) > tol) and np.linalg.norm(diag) <= tol:
        return "shear_or_offdiagonal"
    return "mixed_linear"


def _rank_histogram(pointwise_ranks: Sequence[int]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for rank in pointwise_ranks:
        r = int(rank)
        hist[r] = hist.get(r, 0) + 1
    return hist


def _spectrum_diagnostics(singular_values: Sequence[float], independent_row_count: int, unknown_count: int) -> tuple[float, float]:
    s = np.asarray(tuple(float(v) for v in singular_values), dtype=float)
    if s.size == 0:
        return math.inf, math.inf
    rank = int(min(max(0, independent_row_count), s.size))
    if rank <= 0:
        condition = math.inf
    else:
        smallest_nonzero = max(float(s[rank - 1]), _EPS)
        condition = float(max(float(s[0]), _EPS) / smallest_nonzero)
    if 0 < rank < min(int(unknown_count), s.size):
        gap = float(max(float(s[rank - 1]), _EPS) / max(float(s[rank]), _EPS))
    else:
        gap = math.inf
    return gap, condition
