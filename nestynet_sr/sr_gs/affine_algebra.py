# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Global affine graph-symmetry determining operator.

This module is the PR3 replacement substrate for the older pairwise affine
generator probe.  It solves the scalar graph-tangency equation jointly for input
and output affine actions:

    V = (A x + b) . d_x + (alpha + beta y) d_y

on the graph ``y=f(x)``:

    grad(f)(x) . (A x + b) - alpha - beta f(x) = 0.

The returned object stores a basis-independent nullspace projector first, and
renders generator views only as diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from .algebra_certificates import (
    AffineAlgebraCertificate,
    affine_graph_bracket_coeffs,
    certify_affine_algebra,
)

_EPS = 1.0e-12


@dataclass(frozen=True)
class AffineNormalization:
    """Center/scale metadata for the affine determining solve."""

    x_center: tuple[float, ...]
    x_scale: tuple[float, ...]
    y_center: float
    y_scale: float

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - np.asarray(self.x_center, dtype=float)) / np.asarray(self.x_scale, dtype=float)

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float).reshape(-1) - float(self.y_center)) / float(self.y_scale)

    def normalize_grad(self, grad: np.ndarray) -> np.ndarray:
        return np.asarray(grad, dtype=float) * (np.asarray(self.x_scale, dtype=float) / float(self.y_scale))


@dataclass(frozen=True)
class AffineGeneratorView:
    """Rendered affine graph vector field."""

    coeffs_normalized: tuple[float, ...]
    A_normalized: tuple[tuple[float, ...], ...]
    b_normalized: tuple[float, ...]
    alpha_normalized: float
    beta_normalized: float
    A_physical: tuple[tuple[float, ...], ...]
    b_physical: tuple[float, ...]
    alpha_physical: float
    beta_physical: float


@dataclass(frozen=True)
class SymmetryAlgebraSpec:
    """Basis-independent affine graph-symmetry result."""

    input_dim: int
    coefficient_labels: tuple[str, ...]
    normalization: AffineNormalization
    singular_values: tuple[float, ...]
    rank_tolerance: float
    unknown_count: int
    independent_row_count: int
    nullity: int
    discovered_nullity: int
    nullspace_basis: np.ndarray
    nullspace_projector: np.ndarray
    best_candidate: np.ndarray
    train_residual_rel: float
    heldout_residual_rel: float
    acceptance_residual_tol: float
    structurally_underdetermined: bool
    promotable: bool
    distribution_rank: int
    pointwise_distribution_ranks: tuple[int, ...]
    distribution_basis: np.ndarray
    linear_invariant_covectors: np.ndarray
    bootstrap_principal_angles: tuple[float, ...] = ()
    bracket_closure_residual: float = math.inf
    certificate: AffineAlgebraCertificate | None = None
    basis_generators: tuple[AffineGeneratorView, ...] = ()
    source: str = "data"
    dimensional_warnings: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    chart: str = "identity"

    def input_fields(self, x: np.ndarray, *, basis: np.ndarray | None = None, physical: bool = True) -> np.ndarray:
        """Evaluate input vector fields for a coefficient basis.

        Returns an array with shape ``(n_generators, n_samples, input_dim)``.
        """

        coeffs = self.nullspace_basis if basis is None else np.asarray(basis, dtype=float)
        if coeffs.size == 0:
            return np.zeros((0, len(np.asarray(x)), self.input_dim), dtype=float)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(-1, 1)
        x_arr = np.asarray(x, dtype=float)
        if physical:
            return np.stack([_physical_generator_fields(c, x_arr, self.normalization)[0] for c in coeffs.T], axis=0)
        u = self.normalization.normalize_x(x_arr)
        return np.stack([_unpack_coeffs(c, self.input_dim)[0] @ u.T + _unpack_coeffs(c, self.input_dim)[1][:, None] for c in coeffs.T], axis=0).transpose(0, 2, 1)

    def to_report(self) -> dict[str, Any]:
        return {
            "type": "affine_graph_symmetry_algebra",
            "source": self.source,
            "input_dim": int(self.input_dim),
            "coefficient_labels": list(self.coefficient_labels),
            "singular_values": [float(v) for v in self.singular_values],
            "rank_tolerance": float(self.rank_tolerance),
            "unknown_count": int(self.unknown_count),
            "independent_row_count": int(self.independent_row_count),
            "nullity": int(self.nullity),
            "train_residual_rel": float(self.train_residual_rel),
            "heldout_residual_rel": float(self.heldout_residual_rel),
            "structurally_underdetermined": bool(self.structurally_underdetermined),
            "promotable": bool(self.promotable),
            "distribution_rank": int(self.distribution_rank),
            "bootstrap_principal_angles": [float(v) for v in self.bootstrap_principal_angles],
            "bracket_closure_residual": float(self.bracket_closure_residual),
            "certificate": self.certificate.to_report() if self.certificate is not None else None,
            "dimensional_warnings": list(self.dimensional_warnings),
            "evidence": dict(self.evidence),
            "chart": str(self.chart),
        }


def discover_affine_algebra(
    x: Any,
    y: Any,
    grad: Any,
    *,
    heldout_fraction: float = 0.0,
    bootstrap: int = 0,
    random_state: int = 0,
    rank_rtol: float = 1.0e-9,
    rank_atol: float = 1.0e-11,
    acceptance_residual_tol: float = 1.0e-8,
    units: Sequence[Any] | None = None,
    nullity_strategy: str = "rank_tol",
    min_spectral_gap: float = 10.0,
    max_gap_nullity: int | None = None,
    closure_tol: float | None = None,
    bootstrap_angle_tol: float | None = None,
    heldout_consistency_factor: float | None = None,
) -> SymmetryAlgebraSpec:
    """Solve the global affine graph-symmetry determining operator.

    Parameters
    ----------
    x, y, grad:
        Samples, scalar targets, and physical gradients ``dy/dx``.
    heldout_fraction:
        Fraction of finite rows reserved for held-out residual diagnostics.
    bootstrap:
        Number of bootstrap subspace-stability replicates.
    nullity_strategy:
        ``"rank_tol"`` (default, absolute rank tolerance) or
        ``"spectral_gap"`` (noise-calibrated: nullity at the largest
        singular-value tail gap of at least ``min_spectral_gap``).
    max_gap_nullity:
        Search window for the spectral-gap split.  ``None`` resolves to
        ``input_dim**2``: the graph-invariance algebra of a single covector
        already has dimension ``n**2 - 1`` (e.g. 8 for three inputs), so the
        window must scale with the input dimension, not a fixed constant.
    closure_tol, bootstrap_angle_tol:
        Optional certificate-tolerance overrides; ``None`` keeps the
        certificate defaults.
    """

    x_arr, y_arr, grad_arr = _finite_arrays(x, y, grad)
    n_samples, input_dim = x_arr.shape
    if max_gap_nullity is None:
        max_gap_nullity = int(input_dim) * int(input_dim)
    normalization = _fit_normalization(x_arr, y_arr)
    u = normalization.normalize_x(x_arr)
    y_norm = normalization.normalize_y(y_arr)
    grad_norm = normalization.normalize_grad(grad_arr)
    D = build_affine_determining_matrix(u, y_norm, grad_norm)
    unknown_count = D.shape[1]
    labels = affine_coefficient_labels(input_dim)

    rng = np.random.default_rng(int(random_state))
    train_idx, heldout_idx = _split_indices(n_samples, float(heldout_fraction), rng)
    D_train = D[train_idx]
    D_heldout = D[heldout_idx] if heldout_idx.size else D_train

    basis, projector, singular_values, rank_tol, row_rank, best = _solve_nullspace(
        D_train,
        rank_rtol=float(rank_rtol),
        rank_atol=float(rank_atol),
        nullity_strategy=str(nullity_strategy),
        min_spectral_gap=float(min_spectral_gap),
        max_gap_nullity=int(max_gap_nullity),
    )
    nullity = int(basis.shape[1])
    spectral_gap = _spectral_gap_at_rank(singular_values, int(len(singular_values)) - nullity)
    structurally_underdetermined = bool(D_train.shape[0] < unknown_count)

    train_resid = _relative_residual(D_train, basis if nullity else best.reshape(-1, 1))
    heldout_resid = _relative_residual(D_heldout, basis if nullity else best.reshape(-1, 1))
    fields = _basis_input_fields(basis, x_arr, normalization)
    pointwise_ranks = _pointwise_ranks(fields, rank_strategy=str(nullity_strategy), min_spectral_gap=float(min_spectral_gap))
    distribution_rank = int(max(pointwise_ranks) if pointwise_ranks else 0)
    distribution_basis = _distribution_basis(fields, rank_strategy=str(nullity_strategy), min_spectral_gap=float(min_spectral_gap))
    invariant_covectors = _orthogonal_complement(distribution_basis, input_dim)

    # Covector shrink rescue (spectral-gap mode only): the gap split can
    # overshoot by absorbing an *approximate* near-symmetry direction — the
    # worst-satisfied member of the null block — whose generator field
    # destroys the distribution annihilator. Drop the worst directions
    # (largest per-direction determining residual) one at a time until the
    # retained sub-span admits an invariant covector. The retained algebra is
    # re-certified below, its bootstrap stability is measured like-for-like,
    # and any snapped covector still faces its own determining-residual test.
    covector_shrink_steps = 0
    pre_shrink_nullity = int(nullity)
    if str(nullity_strategy) == "spectral_gap" and nullity > 1 and invariant_covectors.size == 0:
        col_residuals = np.linalg.norm(D_train @ basis, axis=0)
        order = np.argsort(col_residuals)  # best-satisfied first
        sorted_basis = basis[:, order]
        for keep in range(nullity - 1, 0, -1):
            reduced = sorted_basis[:, :keep]
            fields_r = _basis_input_fields(reduced, x_arr, normalization)
            distribution_basis_r = _distribution_basis(fields_r, rank_strategy=str(nullity_strategy), min_spectral_gap=float(min_spectral_gap))
            covectors_r = _orthogonal_complement(distribution_basis_r, input_dim)
            if covectors_r.size:
                covector_shrink_steps = int(nullity) - keep
                basis = reduced
                projector = basis @ basis.T
                nullity = int(keep)
                train_resid = _relative_residual(D_train, basis)
                heldout_resid = _relative_residual(D_heldout, basis)
                fields = fields_r
                pointwise_ranks = _pointwise_ranks(fields, rank_strategy=str(nullity_strategy), min_spectral_gap=float(min_spectral_gap))
                distribution_rank = int(max(pointwise_ranks) if pointwise_ranks else 0)
                distribution_basis = distribution_basis_r
                invariant_covectors = covectors_r
                break
    bootstrap_angles = _bootstrap_principal_angles(
        D_train,
        projector,
        nullity,
        bootstrap=int(bootstrap),
        rng=rng,
        rank_rtol=float(rank_rtol),
        rank_atol=float(rank_atol),
        nullity_strategy=str(nullity_strategy),
        min_spectral_gap=float(min_spectral_gap),
        max_gap_nullity=int(max_gap_nullity),
    )
    generators = tuple(_generator_view(c, input_dim, normalization) for c in basis.T)
    warnings = _dimensional_warnings(units, input_dim)
    certify_overrides: dict[str, float] = {}
    if closure_tol is not None:
        certify_overrides["closure_tol"] = float(closure_tol)
    if bootstrap_angle_tol is not None:
        certify_overrides["bootstrap_angle_tol"] = float(bootstrap_angle_tol)
    if heldout_consistency_factor is not None:
        certify_overrides["heldout_consistency_factor"] = float(heldout_consistency_factor)
    certificate = certify_affine_algebra(
        basis=basis,
        projector=projector,
        input_dim=int(input_dim),
        singular_values=singular_values,
        independent_row_count=int(row_rank),
        unknown_count=int(unknown_count),
        train_residual_rel=float(train_resid),
        heldout_residual_rel=float(heldout_resid),
        acceptance_residual_tol=float(acceptance_residual_tol),
        structurally_underdetermined=structurally_underdetermined,
        distribution_rank=distribution_rank,
        pointwise_distribution_ranks=pointwise_ranks,
        bootstrap_principal_angles=bootstrap_angles,
        dimensional_warnings=warnings,
        **certify_overrides,
    )
    bracket_resid = float(certificate.bracket_closure_residual)
    promotable = bool(certificate.quotient_ready)

    return SymmetryAlgebraSpec(
        input_dim=int(input_dim),
        coefficient_labels=labels,
        normalization=normalization,
        singular_values=tuple(float(v) for v in singular_values),
        rank_tolerance=float(rank_tol),
        unknown_count=int(unknown_count),
        independent_row_count=int(row_rank),
        nullity=int(nullity),
        discovered_nullity=int(pre_shrink_nullity),
        nullspace_basis=basis,
        nullspace_projector=projector,
        best_candidate=best,
        train_residual_rel=float(train_resid),
        heldout_residual_rel=float(heldout_resid),
        acceptance_residual_tol=float(acceptance_residual_tol),
        structurally_underdetermined=structurally_underdetermined,
        promotable=promotable,
        distribution_rank=distribution_rank,
        pointwise_distribution_ranks=tuple(int(v) for v in pointwise_ranks),
        distribution_basis=distribution_basis,
        linear_invariant_covectors=invariant_covectors,
        bootstrap_principal_angles=tuple(float(v) for v in bootstrap_angles),
        bracket_closure_residual=float(bracket_resid),
        certificate=certificate,
        basis_generators=generators,
        dimensional_warnings=warnings,
        evidence={
            "n_samples": int(n_samples),
            "n_train": int(train_idx.size),
            "n_heldout": int(D_heldout.shape[0]),
            "rank_rtol": float(rank_rtol),
            "rank_atol": float(rank_atol),
            "heldout_fraction": float(heldout_fraction),
            "bootstrap": int(bootstrap),
            "nullity_strategy": str(nullity_strategy),
            "min_spectral_gap": float(min_spectral_gap),
            "spectral_gap": float(min(spectral_gap, 1.0e15)),
            "covector_shrink_steps": int(covector_shrink_steps),
            "covector_shrink_boundary_gap": float(
                min(_spectral_gap_at_rank(singular_values, int(len(singular_values)) - int(nullity)), 1.0e15)
            ),
        },
    )


def build_affine_determining_matrix(x_norm: Any, y_norm: Any, grad_norm: Any) -> np.ndarray:
    """Build rows ``[g_i*x_j, g_i, -1, -y]`` for normalized samples."""

    x_arr = np.asarray(x_norm, dtype=float)
    y_arr = np.asarray(y_norm, dtype=float).reshape(-1)
    grad_arr = np.asarray(grad_norm, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_norm must have shape (N,n)")
    if grad_arr.shape != x_arr.shape:
        raise ValueError(f"grad_norm shape {grad_arr.shape} must match x_norm shape {x_arr.shape}")
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(f"y_norm length {y_arr.shape[0]} != x rows {x_arr.shape[0]}")
    gx = np.einsum("ni,nj->nij", grad_arr, x_arr).reshape(x_arr.shape[0], -1)
    return np.concatenate([gx, grad_arr, -np.ones((x_arr.shape[0], 1)), -y_arr[:, None]], axis=1)


def affine_coefficient_labels(input_dim: int) -> tuple[str, ...]:
    labels = []
    for i in range(int(input_dim)):
        for j in range(int(input_dim)):
            labels.append(f"A[{i},{j}]")
    labels.extend(f"b[{i}]" for i in range(int(input_dim)))
    labels.extend(["alpha", "beta"])
    return tuple(labels)


def _finite_arrays(x: Any, y: Any, grad: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = _to_numpy(x)
    y_arr = _to_numpy(y).reshape(-1)
    grad_arr = _to_numpy(grad)
    if x_arr.ndim != 2:
        raise ValueError(f"x must have shape (N,n); got {x_arr.shape}")
    if grad_arr.ndim == 3 and grad_arr.shape[1] == 1:
        grad_arr = grad_arr[:, 0, :]
    if grad_arr.shape != x_arr.shape:
        raise ValueError(f"grad shape {grad_arr.shape} must match x shape {x_arr.shape}")
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(f"y length {y_arr.shape[0]} != x rows {x_arr.shape[0]}")
    mask = np.isfinite(x_arr).all(axis=1) & np.isfinite(y_arr) & np.isfinite(grad_arr).all(axis=1)
    if int(mask.sum()) < 2:
        raise ValueError("at least two finite samples are required")
    return x_arr[mask].astype(float), y_arr[mask].astype(float), grad_arr[mask].astype(float)


def _to_numpy(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


def _fit_normalization(x: np.ndarray, y: np.ndarray) -> AffineNormalization:
    x_center = np.median(x, axis=0)
    x_scale = np.asarray([_robust_scale(x[:, i]) for i in range(x.shape[1])], dtype=float)
    y_center = float(np.median(y))
    y_scale = float(_robust_scale(y))
    return AffineNormalization(
        x_center=tuple(float(v) for v in x_center),
        x_scale=tuple(float(v) for v in x_scale),
        y_center=float(y_center),
        y_scale=float(y_scale),
    )


def _robust_scale(v: np.ndarray) -> float:
    arr = np.asarray(v, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    centered = arr - float(np.median(arr))
    mad = 1.4826 * float(np.median(np.abs(centered)))
    std = float(np.std(arr))
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    iqr = 0.7413 * float(abs(q75 - q25))
    return max(1.0e-12, mad, std, iqr)


def _split_indices(n: int, heldout_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(int(n))
    if heldout_fraction <= 0.0 or n < 4:
        return idx, np.asarray([], dtype=int)
    n_held = int(round(float(heldout_fraction) * n))
    n_held = max(1, min(n - 2, n_held))
    perm = rng.permutation(idx)
    heldout = np.sort(perm[:n_held])
    train = np.sort(perm[n_held:])
    return train, heldout


def _spectral_gap_rank(s: np.ndarray, *, min_spectral_gap: float, max_gap_nullity: int) -> int:
    """Rank cut at the largest relative gap in the singular-value tail.

    Noise-calibrated alternative to the absolute rank tolerance: a real
    symmetry separates the noise-floor block from the signal block by a large
    multiplicative gap whatever the noise level, whereas generic data shows
    only O(1) consecutive ratios.  Returns full rank (no nullspace) when no
    tail gap reaches ``min_spectral_gap``.
    """

    values = np.asarray(s, dtype=float)
    n_vals = int(values.size)
    if n_vals < 2:
        return n_vals
    best_rank, best_gap = n_vals, 0.0
    lo = max(1, n_vals - max(1, int(max_gap_nullity)))
    for k in range(lo, n_vals):
        gap = float(values[k - 1] / max(float(values[k]), 1.0e-300))
        if gap > best_gap:
            best_gap, best_rank = gap, k
    if best_gap < float(min_spectral_gap):
        return n_vals
    return int(best_rank)


def _spectral_gap_at_rank(s: Sequence[float], rank: int) -> float:
    """Ratio of the smallest signal singular value to the largest null one."""

    values = np.asarray(tuple(s), dtype=float)
    if values.size == 0 or rank <= 0:
        return 0.0
    if rank >= values.size:
        return float("inf")
    return float(values[rank - 1] / max(float(values[rank]), 1.0e-300))


def _solve_nullspace(
    D: np.ndarray,
    *,
    rank_rtol: float,
    rank_atol: float,
    nullity_strategy: str = "rank_tol",
    min_spectral_gap: float = 10.0,
    max_gap_nullity: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, np.ndarray]:
    if D.ndim != 2:
        raise ValueError("determining matrix must be 2D")
    p = D.shape[1]
    if D.shape[0] == 0:
        basis = np.eye(p)
        return basis, np.eye(p), np.asarray([], dtype=float), float(rank_atol), 0, basis[:, 0]
    U, s, Vt = np.linalg.svd(D, full_matrices=True)
    del U
    s0 = float(s[0]) if s.size else 0.0
    rank_tol = max(float(rank_atol), float(rank_rtol) * max(1.0, s0))
    if str(nullity_strategy) == "spectral_gap":
        rank = _spectral_gap_rank(s, min_spectral_gap=float(min_spectral_gap), max_gap_nullity=int(max_gap_nullity))
    else:
        rank = int(np.sum(s > rank_tol))
    nullity = max(0, p - rank)
    raw_basis = Vt[rank:].T if nullity else np.zeros((p, 0), dtype=float)
    basis = _orthonormalize_columns(raw_basis)
    projector = basis @ basis.T if basis.size else np.zeros((p, p), dtype=float)
    best = Vt[-1].copy() if Vt.shape[0] else np.eye(p)[:, 0]
    best_norm = float(np.linalg.norm(best))
    if best_norm > 0.0:
        best = best / best_norm
    return basis, projector, s, rank_tol, rank, best


def _orthonormalize_columns(M: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(M, dtype=float)
    if arr.size == 0:
        return np.zeros((arr.shape[0], 0), dtype=float)
    Q, R = np.linalg.qr(arr)
    diag = np.abs(np.diag(R))
    keep = diag > tol
    return Q[:, keep]


def _relative_residual(D: np.ndarray, basis: np.ndarray) -> float:
    if basis.size == 0:
        return math.inf
    if not np.isfinite(D).all() or not np.isfinite(basis).all():
        return math.inf
    resid = np.einsum("ij,jk->ik", D, basis)
    denom = max(float(np.linalg.norm(D, ord="fro")), _EPS) * max(1.0, math.sqrt(float(basis.shape[1])))
    return float(np.linalg.norm(resid, ord="fro") / denom)


def _basis_input_fields(basis: np.ndarray, x: np.ndarray, normalization: AffineNormalization) -> np.ndarray:
    if basis.size == 0:
        return np.zeros((0, x.shape[0], x.shape[1]), dtype=float)
    return np.stack([_physical_generator_fields(c, x, normalization)[0] for c in basis.T], axis=0)


def _physical_generator_fields(coeffs: np.ndarray, x: np.ndarray, normalization: AffineNormalization) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    n = x.shape[1]
    A_norm, b_norm, alpha_norm, beta = _unpack_coeffs(coeffs, n)
    sx = np.asarray(normalization.x_scale, dtype=float)
    cx = np.asarray(normalization.x_center, dtype=float)
    S = np.diag(sx)
    Sinv = np.diag(1.0 / sx)
    A_phys = S @ A_norm @ Sinv
    b_phys = sx * b_norm - A_phys @ cx
    alpha_phys = float(normalization.y_scale) * float(alpha_norm) - float(beta) * float(normalization.y_center)
    fields = x @ A_phys.T + b_phys
    return fields, A_phys, b_phys, alpha_phys, float(beta)


def _unpack_coeffs(coeffs: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    expected = n * n + n + 2
    if c.size != expected:
        raise ValueError(f"coefficient vector length {c.size} != expected {expected}")
    A = c[: n * n].reshape(n, n)
    b = c[n * n : n * n + n]
    alpha = float(c[-2])
    beta = float(c[-1])
    return A, b, alpha, beta


def _pointwise_ranks(fields: np.ndarray, *, rank_strategy: str = "rank_tol", min_spectral_gap: float = 10.0) -> list[int]:
    if fields.size == 0:
        return []
    ranks = []
    for i in range(fields.shape[1]):
        M = fields[:, i, :]
        if str(rank_strategy) == "spectral_gap":
            s = np.linalg.svd(M, compute_uv=False)
            ranks.append(int(_spectral_gap_rank(s, min_spectral_gap=float(min_spectral_gap), max_gap_nullity=int(M.shape[1]))))
        else:
            scale = max(1.0, float(np.linalg.norm(M)))
            ranks.append(int(np.linalg.matrix_rank(M, tol=1.0e-9 * scale)))
    return ranks


def _distribution_basis(fields: np.ndarray, *, rank_strategy: str = "rank_tol", min_spectral_gap: float = 10.0) -> np.ndarray:
    if fields.size == 0:
        return np.zeros((0, 0), dtype=float)
    M = fields.reshape(-1, fields.shape[-1])
    if M.size == 0:
        return np.zeros((0, fields.shape[-1]), dtype=float)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    del U
    if str(rank_strategy) == "spectral_gap":
        rank = int(_spectral_gap_rank(s, min_spectral_gap=float(min_spectral_gap), max_gap_nullity=int(M.shape[-1])))
    else:
        tol = max(1.0e-11, 1.0e-9 * max(1.0, float(s[0]) if s.size else 0.0))
        rank = int(np.sum(s > tol))
    return Vt[:rank]


def _orthogonal_complement(row_basis: np.ndarray, n: int) -> np.ndarray:
    if row_basis.size == 0:
        return np.eye(int(n))
    U, s, Vt = np.linalg.svd(row_basis, full_matrices=True)
    del U
    tol = max(1.0e-11, 1.0e-9 * max(1.0, float(s[0]) if s.size else 0.0))
    rank = int(np.sum(s > tol))
    return Vt[rank:]


def _bootstrap_principal_angles(
    D: np.ndarray,
    reference_projector: np.ndarray,
    reference_nullity: int,
    *,
    bootstrap: int,
    rng: np.random.Generator,
    rank_rtol: float,
    rank_atol: float,
    nullity_strategy: str = "rank_tol",
    min_spectral_gap: float = 10.0,
    max_gap_nullity: int = 6,
) -> list[float]:
    if bootstrap <= 0 or reference_nullity <= 0 or D.shape[0] < 2:
        return []
    out = []
    for _ in range(int(bootstrap)):
        idx = rng.integers(0, D.shape[0], size=D.shape[0])
        basis_b, projector_b, _s, _tol, _rank, _best = _solve_nullspace(
            D[idx],
            rank_rtol=rank_rtol,
            rank_atol=rank_atol,
            nullity_strategy=nullity_strategy,
            min_spectral_gap=min_spectral_gap,
            max_gap_nullity=max_gap_nullity,
        )
        if basis_b.shape[1] < reference_nullity:
            out.append(float(math.pi / 2.0))
            continue
        if basis_b.shape[1] > reference_nullity:
            # Like-for-like comparison after a covector shrink: keep the
            # replicate's best-satisfied ``reference_nullity`` directions,
            # mirroring the deterministic worst-first shrink of the reference.
            col_residuals = np.linalg.norm(D[idx] @ basis_b, axis=0)
            order = np.argsort(col_residuals)
            basis_b = basis_b[:, order[: int(reference_nullity)]]
            projector_b = basis_b @ basis_b.T
        diff = reference_projector - projector_b
        norm = float(np.linalg.norm(diff, ord=2))
        norm = max(0.0, min(1.0, norm))
        out.append(float(math.asin(norm)))
    return out


def _bracket_closure_residual(basis: np.ndarray, projector: np.ndarray, n: int) -> float:
    if basis.shape[1] <= 1:
        return 0.0
    max_resid = 0.0
    for i in range(basis.shape[1]):
        for j in range(i + 1, basis.shape[1]):
            bracket = _affine_bracket_coeffs(basis[:, i], basis[:, j], n)
            norm = float(np.linalg.norm(bracket))
            if norm <= _EPS:
                continue
            resid = float(np.linalg.norm(bracket - projector @ bracket) / norm)
            max_resid = max(max_resid, resid)
    return float(max_resid)


def _affine_bracket_coeffs(c1: np.ndarray, c2: np.ndarray, n: int) -> np.ndarray:
    return affine_graph_bracket_coeffs(c1, c2, n)


def _generator_view(coeffs: np.ndarray, n: int, normalization: AffineNormalization) -> AffineGeneratorView:
    A_norm, b_norm, alpha_norm, beta = _unpack_coeffs(coeffs, n)
    dummy = np.zeros((1, n), dtype=float)
    _fields, A_phys, b_phys, alpha_phys, beta_phys = _physical_generator_fields(coeffs, dummy, normalization)
    return AffineGeneratorView(
        coeffs_normalized=tuple(float(v) for v in coeffs.reshape(-1)),
        A_normalized=tuple(tuple(float(v) for v in row) for row in A_norm),
        b_normalized=tuple(float(v) for v in b_norm),
        alpha_normalized=float(alpha_norm),
        beta_normalized=float(beta),
        A_physical=tuple(tuple(float(v) for v in row) for row in A_phys),
        b_physical=tuple(float(v) for v in b_phys),
        alpha_physical=float(alpha_phys),
        beta_physical=float(beta_phys),
    )


def _dimensional_warnings(units: Sequence[Any] | None, n: int) -> tuple[str, ...]:
    if units is None:
        return ()
    if len(units) != n:
        return (f"units length {len(units)} does not match input_dim {n}",)
    warnings = []
    for i in range(n):
        for j in range(n):
            if i != j and str(units[i]) != str(units[j]):
                warnings.append(f"A[{i},{j}] mixes {units[j]} into {units[i]}")
    return tuple(warnings)
