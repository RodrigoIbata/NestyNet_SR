# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Rational polynomial degree probing with dimensional analysis.

Given a target dimension ``dim_f`` and input variable dimensions ``x_dims``,
determine which rational polynomial degree pairs ``(deg_num, deg_den)`` are
dimensionally valid and report minimum/maximum useful degrees and monomial
counts.

A rational polynomial P(x)/Q(x) requires:
* Every monomial in P shares one dimension ``dim_P``.
* Every monomial in Q shares one dimension ``dim_Q``.
* ``dim_P - dim_Q = dim_f``.

For polynomial monomials (non-negative integer exponents), the dimension of
``x^alpha`` is the linear map ``sum(alpha_i * x_dims[i])``.  Monomials are
grouped into *dim-classes* — sets of monomials sharing the same dimension
vector.  Valid rational polynomials pick one dim-class for P and one for Q
such that their difference equals the target.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nestynet_sr.sr_core.coefficient_units import monomial_dimension

# Re-use the project's Dim type (tuple of Fraction).
Dim = Tuple[Fraction, ...]


# ─────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────


@dataclass
class DimClassPair:
    """One dimensionally-valid (numerator, denominator) pairing."""

    dim_num: Dim  # dimension shared by all numerator monomials
    dim_den: Dim  # dimension shared by all denominator monomials
    # degree → list of exponent tuples at that total degree
    monomials_num: Dict[int, List[Tuple[int, ...]]] = field(default_factory=dict)
    monomials_den: Dict[int, List[Tuple[int, ...]]] = field(default_factory=dict)

    @property
    def min_deg_num(self) -> int:
        return min(self.monomials_num) if self.monomials_num else 0

    @property
    def min_deg_den(self) -> int:
        return min(self.monomials_den) if self.monomials_den else 0

    @property
    def max_deg_num(self) -> int:
        return max(self.monomials_num) if self.monomials_num else 0

    @property
    def max_deg_den(self) -> int:
        return max(self.monomials_den) if self.monomials_den else 0

    @property
    def total_monomials_num(self) -> int:
        return sum(len(v) for v in self.monomials_num.values())

    @property
    def total_monomials_den(self) -> int:
        return sum(len(v) for v in self.monomials_den.values())

    def n_monomials_num_up_to(self, deg: int) -> int:
        """Count numerator monomials with total degree <= deg."""
        return sum(len(v) for k, v in self.monomials_num.items() if k <= deg)

    def n_monomials_den_up_to(self, deg: int) -> int:
        """Count denominator monomials with total degree <= deg."""
        return sum(len(v) for k, v in self.monomials_den.items() if k <= deg)

    def exponents_num_up_to(self, deg: int) -> List[Tuple[int, ...]]:
        """All numerator exponent tuples with total degree <= deg, sorted."""
        out = []
        for k in sorted(self.monomials_num):
            if k > deg:
                break
            out.extend(self.monomials_num[k])
        return out

    def exponents_den_up_to(self, deg: int) -> List[Tuple[int, ...]]:
        """All denominator exponent tuples with total degree <= deg, sorted."""
        out = []
        for k in sorted(self.monomials_den):
            if k > deg:
                break
            out.extend(self.monomials_den[k])
        return out


@dataclass
class RatPolyDegreeInfo:
    """Result of a rational polynomial degree probe."""

    # Valid (numerator, denominator) dim-class pairings, sorted by
    # (min_deg_num + min_deg_den, min_deg_num, min_deg_den).
    valid_pairs: List[DimClassPair] = field(default_factory=list)

    # Dimension of the null-space of the dimension map restricted to
    # non-negative exponents — i.e. how many independent dimensionless
    # products can be formed from the variables (Buckingham Pi count).
    dimensionless_rank: int = 0

    # Convenience flags.
    all_dimensionless: bool = False  # target + all vars are dimensionless
    same_units: bool = False  # all vars share the same dimension

    @property
    def global_min_deg_num(self) -> int:
        if not self.valid_pairs:
            return 0
        return min(p.min_deg_num for p in self.valid_pairs)

    @property
    def global_min_deg_den(self) -> int:
        if not self.valid_pairs:
            return 0
        return min(p.min_deg_den for p in self.valid_pairs)

    @property
    def has_polynomial_solution(self) -> bool:
        """True if any valid pair has min_deg_den == 0 (pure polynomial)."""
        return any(p.min_deg_den == 0 for p in self.valid_pairs)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _to_dim(d: Sequence) -> Dim:
    """Normalise a dimension vector to a tuple of Fraction."""
    return tuple(Fraction(x) for x in d)


def _is_dimless(d: Dim) -> bool:
    return all(e == 0 for e in d)


def _dim_sub(a: Dim, b: Dim) -> Dim:
    return tuple(ai - bi for ai, bi in zip(a, b))


def _dim_add(a: Dim, b: Dim) -> Dim:
    return tuple(ai + bi for ai, bi in zip(a, b))


def _monomial_dim(alpha: Tuple[int, ...], x_dims: List[Dim]) -> Dim:
    """Dimension of monomial x^alpha = prod(x_i^alpha_i)."""
    return monomial_dimension(alpha, x_dims)


def _enumerate_monomials(n_in: int, total_deg: int) -> List[Tuple[int, ...]]:
    """All non-negative integer exponent tuples summing to exactly total_deg."""
    result: List[Tuple[int, ...]] = []

    def rec(pos: int, remaining: int, cur: List[int]):
        if pos == n_in:
            if remaining == 0:
                result.append(tuple(cur))
            return
        for p in range(remaining + 1):
            cur.append(p)
            rec(pos + 1, remaining - p, cur)
            cur.pop()

    rec(0, total_deg, [])
    return result


def _dimension_matrix_rank(x_dims: List[Dim]) -> int:
    """Rank of the d×B dimension matrix (variables as rows)."""
    if not x_dims:
        return 0
    d = len(x_dims)
    B = len(x_dims[0])
    mat = np.array([[float(x_dims[i][k]) for k in range(B)] for i in range(d)])
    return int(np.linalg.matrix_rank(mat, tol=1e-12))


# ─────────────────────────────────────────────────────────────────────
# Main probe function
# ─────────────────────────────────────────────────────────────────────


def probe_rational_degrees(
    target_dim: Sequence,
    x_dims: Sequence[Sequence],
    max_total_degree: int = 8,
) -> RatPolyDegreeInfo:
    """Determine dimensionally plausible rational polynomial degrees.

    Parameters
    ----------
    target_dim : sequence of numbers
        Dimension of the target function f (exponent vector over SI base).
    x_dims : sequence of sequences
        Dimension of each input variable.
    max_total_degree : int
        Maximum total degree to enumerate monomials up to (for both
        numerator and denominator).

    Returns
    -------
    RatPolyDegreeInfo
        Dataclass with valid dim-class pairs, min/max degrees, monomial
        counts, and convenience flags.
    """
    target = _to_dim(target_dim)
    xd = [_to_dim(d) for d in x_dims]
    n_in = len(xd)

    if n_in == 0:
        return RatPolyDegreeInfo()

    # ── Convenience flags ──
    all_dimless = _is_dimless(target) and all(_is_dimless(d) for d in xd)
    same_units = len(set(xd)) == 1 if n_in > 0 else True
    dim_rank = _dimension_matrix_rank(xd)
    dimless_rank = n_in - dim_rank  # Buckingham Pi count

    # ── Enumerate monomials and group by dimension ──
    # dim_classes: dim → {total_deg → [alpha, ...]}
    dim_classes: Dict[Dim, Dict[int, List[Tuple[int, ...]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for total_deg in range(0, max_total_degree + 1):
        for alpha in _enumerate_monomials(n_in, total_deg):
            md = _monomial_dim(alpha, xd)
            dim_classes[md][total_deg].append(alpha)

    # ── Find valid (P-class, Q-class) pairs ──
    # We need dim_P - dim_Q = target.
    # For each candidate dim_P, the required dim_Q is dim_P - target.
    #
    # Special case: Q = 1 (constant) has dim_Q = dimensionless.
    # This corresponds to a pure polynomial numerator.
    valid_pairs: List[DimClassPair] = []
    seen: set = set()

    for dim_P, degs_P in dim_classes.items():
        dim_Q_needed = _dim_sub(dim_P, target)
        if dim_Q_needed in dim_classes:
            key = (dim_P, dim_Q_needed)
            if key in seen:
                continue
            seen.add(key)
            pair = DimClassPair(
                dim_num=dim_P,
                dim_den=dim_Q_needed,
                monomials_num=dict(degs_P),
                monomials_den=dict(dim_classes[dim_Q_needed]),
            )
            valid_pairs.append(pair)

    # Sort by minimum total degree (num + den), then num, then den.
    valid_pairs.sort(
        key=lambda p: (p.min_deg_num + p.min_deg_den, p.min_deg_num, p.min_deg_den)
    )

    info = RatPolyDegreeInfo(
        valid_pairs=valid_pairs,
        dimensionless_rank=dimless_rank,
        all_dimensionless=all_dimless,
        same_units=same_units,
    )


    return info


# ─────────────────────────────────────────────────────────────────────
# Pure-polynomial convenience probe
# ─────────────────────────────────────────────────────────────────────


def probe_poly_exponents(
    target_dim: Sequence,
    x_dims: Sequence[Sequence],
    max_degree: int = 8,
) -> Optional[Dict[int, List[Tuple[int, ...]]]]:
    """Return dimensionally valid monomial exponents for a polynomial P(x).

    A polynomial P(x) with ``dim(P) = target_dim`` can only contain
    monomials whose dimension equals ``target_dim``.  This function
    enumerates such monomials up to ``max_degree`` and returns them
    grouped by total degree.

    Parameters
    ----------
    target_dim : sequence of numbers
        Required dimension of the polynomial (e.g. dim of the atom output,
        or negated dim for ``1/P``, or doubled dim for ``sqrt(P)``).
    x_dims : sequence of sequences
        Dimension of each input variable.
    max_degree : int
        Maximum total degree to enumerate.

    Returns
    -------
    dict or None
        ``{total_degree: [exponent_tuples]}`` for valid monomials, or
        ``None`` if no valid monomials exist up to ``max_degree``.
    """
    target = _to_dim(target_dim)
    xd = [_to_dim(d) for d in x_dims]
    n_in = len(xd)
    if n_in == 0:
        return None

    result: Dict[int, List[Tuple[int, ...]]] = defaultdict(list)
    for total_deg in range(0, max_degree + 1):
        for alpha in _enumerate_monomials(n_in, total_deg):
            if _monomial_dim(alpha, xd) == target:
                result[total_deg].append(alpha)


    if not result:
        return None
    return dict(result)


# ─────────────────────────────────────────────────────────────────────
# Cheap data probe (linear algebra)
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ProbeFitResult:
    """Result of a cheap rational probe fit for one degree pair."""

    pair: DimClassPair
    deg_num: int
    deg_den: int
    n_terms_num: int
    n_terms_den: int
    rel_rms: float  # relative RMS error on hold-out set
    coeffs_num: Optional[np.ndarray] = None
    coeffs_den: Optional[np.ndarray] = None


def probe_rational_fit(
    X: np.ndarray,
    F: np.ndarray,
    degree_info: RatPolyDegreeInfo,
    max_pairs: int = 20,
    max_deg_per_pair: int = 3,
    fit_frac: float = 0.7,
    eps_Q: float = 1e-10,
    seed: int = 0,
) -> List[ProbeFitResult]:
    """For each valid dim-class pair, do a cheap linear-algebra fit.

    Tries increasing degree levels for each pair (from min up to
    min + max_deg_per_pair) and returns results sorted by rel_rms.

    Parameters
    ----------
    X : (N, d) array
        Input data.
    F : (N,) array
        Target values.
    degree_info : RatPolyDegreeInfo
        Output of :func:`probe_rational_degrees`.
    max_pairs : int
        Maximum number of dim-class pairs to try.
    max_deg_per_pair : int
        How many degree levels above the minimum to try per pair.
    fit_frac : float
        Fraction of data used for fitting (rest for validation).
    eps_Q : float
        Minimum absolute denominator value.
    seed : int
        Random seed for train/val split.

    Returns
    -------
    list of ProbeFitResult
        Sorted by rel_rms (ascending).
    """
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_fit = max(10, int(N * fit_frac))
    idx_fit, idx_val = idx[:n_fit], idx[n_fit:]
    if len(idx_val) < 5:
        idx_val = idx_fit  # fallback: validate on training set

    X_fit, F_fit = X[idx_fit], F[idx_fit]
    X_val, F_val = X[idx_val], F[idx_val]

    results: List[ProbeFitResult] = []

    for pair in degree_info.valid_pairs[:max_pairs]:
        # Try increasing degrees for this pair.
        for offset in range(max_deg_per_pair + 1):
            deg_n = pair.min_deg_num + offset
            deg_d = pair.min_deg_den + offset
            if deg_n > (pair.max_deg_num if pair.monomials_num else 0):
                continue
            if deg_d > (pair.max_deg_den if pair.monomials_den else 0):
                continue

            exps_num = pair.exponents_num_up_to(deg_n)
            exps_den = pair.exponents_den_up_to(deg_d)
            if not exps_num or not exps_den:
                continue

            res = _fit_rational_svd(
                X_fit, F_fit, X_val, F_val,
                exps_num, exps_den, eps_Q,
            )
            if res is None:
                continue

            rel_rms, a, b = res
            results.append(ProbeFitResult(
                pair=pair,
                deg_num=deg_n,
                deg_den=deg_d,
                n_terms_num=len(exps_num),
                n_terms_den=len(exps_den),
                rel_rms=rel_rms,
                coeffs_num=a,
                coeffs_den=b,
            ))

    # Sort by (rel_rms binned to 1e-6, total parameter count) so that among
    # near-identical fits the simplest one (fewest terms) wins.
    results.sort(key=lambda r: (
        round(r.rel_rms / 1e-6) if r.rel_rms > 1e-6 else 0,
        r.n_terms_num + r.n_terms_den,
        r.deg_num + r.deg_den,
    ))
    return results


def _fit_rational_svd(
    X_fit: np.ndarray,
    F_fit: np.ndarray,
    X_val: np.ndarray,
    F_val: np.ndarray,
    exps_num: List[Tuple[int, ...]],
    exps_den: List[Tuple[int, ...]],
    eps_Q: float = 1e-10,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """SVD-based rational fit: f*Q - P = 0 as homogeneous least-squares.

    Returns (rel_rms_val, coeffs_num, coeffs_den) or None on failure.
    """
    N_fit = X_fit.shape[0]
    n_num = len(exps_num)
    n_den = len(exps_den)

    # Build monomial matrices.
    Phi_num_fit = np.ones((N_fit, n_num), dtype=np.float64)
    for j, alpha in enumerate(exps_num):
        for k, a in enumerate(alpha):
            if a != 0:
                Phi_num_fit[:, j] *= X_fit[:, k] ** a

    Phi_den_fit = np.ones((N_fit, n_den), dtype=np.float64)
    for j, alpha in enumerate(exps_den):
        for k, a in enumerate(alpha):
            if a != 0:
                Phi_den_fit[:, j] *= X_fit[:, k] ** a

    # Build the homogeneous system: [f*Phi_den | -Phi_num] @ [b; a] = 0.
    f_col = F_fit.reshape(-1, 1)
    A = np.hstack([f_col * Phi_den_fit, -Phi_num_fit])

    if not np.all(np.isfinite(A)):
        return None

    try:
        _, s, Vt = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError:
        return None

    # Solution is the last row of V^T (smallest singular value).
    sol = Vt[-1, :]
    b = sol[:n_den]
    a = sol[n_den:]

    # Evaluate on validation set.
    N_val = X_val.shape[0]
    Phi_num_val = np.ones((N_val, n_num), dtype=np.float64)
    for j, alpha in enumerate(exps_num):
        for k, ak in enumerate(alpha):
            if ak != 0:
                Phi_num_val[:, j] *= X_val[:, k] ** ak

    Phi_den_val = np.ones((N_val, n_den), dtype=np.float64)
    for j, alpha in enumerate(exps_den):
        for k, ak in enumerate(alpha):
            if ak != 0:
                Phi_den_val[:, j] *= X_val[:, k] ** ak

    P_val = Phi_num_val @ a
    Q_val = Phi_den_val @ b

    # Mask out near-zero denominator points.
    mask = np.abs(Q_val) > eps_Q
    if mask.sum() < 5:
        return None

    pred = P_val[mask] / Q_val[mask]
    truth = F_val[mask]

    scale = np.median(np.abs(truth))
    if scale < 1e-30:
        scale = 1.0
    rel_rms = float(np.sqrt(np.median((pred - truth) ** 2)) / scale)

    if not np.isfinite(rel_rms):
        return None

    return rel_rms, a, b


# ─────────────────────────────────────────────────────────────────────
# Convenience: summary string
# ─────────────────────────────────────────────────────────────────────


def summarise_degree_info(info: RatPolyDegreeInfo) -> str:
    """Human-readable summary of degree probe results."""
    lines = []
    lines.append(f"[RatPoly Degree Probe] {len(info.valid_pairs)} valid dim-class pair(s)")
    lines.append(f"  dimensionless_rank={info.dimensionless_rank}, "
                 f"all_dimless={info.all_dimensionless}, same_units={info.same_units}")
    if info.valid_pairs:
        lines.append(f"  global_min_deg_num={info.global_min_deg_num}, "
                     f"global_min_deg_den={info.global_min_deg_den}, "
                     f"has_polynomial={info.has_polynomial_solution}")
    for i, p in enumerate(info.valid_pairs[:10]):
        n_num = p.total_monomials_num
        n_den = p.total_monomials_den
        lines.append(
            f"  pair {i}: dim_num={_dim_str(p.dim_num)}, dim_den={_dim_str(p.dim_den)}, "
            f"deg_num=[{p.min_deg_num}..{p.max_deg_num}] ({n_num} terms), "
            f"deg_den=[{p.min_deg_den}..{p.max_deg_den}] ({n_den} terms)"
        )
    if len(info.valid_pairs) > 10:
        lines.append(f"  ... and {len(info.valid_pairs) - 10} more pair(s)")
    return "\n".join(lines)


def _dim_str(d: Dim) -> str:
    """Compact dimension string like '[1,-2,1,0,0]'."""
    return "[" + ",".join(str(float(x)) if x != int(x) else str(int(x)) for x in d) + "]"
