# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Coefficient fitting utilities for Stage B.
Functions for fitting analytical forms (polynomials, rational functions,
trigonometric, power laws, exponentials) to data.
"""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from nestynet_sr.sr_core.atoms import (
    Expm1Leaf,
    ExpPolyLeaf,
    ExpRationalPolyLeaf,
    PlanckFullLeaf,
    PolyLogLeaf,
    PlanckLeaf,
    PolyLeaf,
    PowerLeaf,
    RationalPolyLeaf,
    RExpPolyLeaf,
    RPolyLogLeaf,
    RPolyLeaf,
    RRationalPolyLeaf,
    SinLinearLeaf,
    TanhLinearLeaf,
    _enumerate_exponents,
    _eval_monomials,
)
from nestynet_sr.sr_core.bridges import Node, effective_arity

from .features import TrigAxisSpec
from .rational_sparsify import (
    DEFAULT_POLY_STLSQ_CFG,
    DEFAULT_RAT_STLSQ_CFG,
    _log_sparsify_result,
    stlsq_sparsify_poly_coeffs,
    stlsq_sparsify_rational_coeffs,
)

import logging as _logging

_log = _logging.getLogger(__name__)


PLANCK_STRUCTURAL_POWERS: Tuple[float, ...] = (0.0, 1.0, 2.0)


def _filter_outliers(
    X: torch.Tensor,
    f: torch.Tensor,
    method: str = "mad",
    threshold: float = 20.0,
    percentile: float = 99.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter outlier samples based on target magnitude.

    Parameters
    ----------
    X : Tensor [N, d]
        Input features.
    f : Tensor [N] or [N, 1]
        Target values.
    method : str
        'mad' - keep samples where |f| < median(|f|) + threshold * MAD(|f|)
        'percentile' - keep samples below the given percentile of |f|
    threshold : float
        For 'mad': number of MADs above median to keep (default 20)
    percentile : float
        For 'percentile': percentile cutoff (default 99.0 = drop top 1%)

    Returns
    -------
    X_filtered, f_filtered
    """
    f_flat = f.view(-1)
    f_abs = f_flat.abs()

    if method == "mad":
        med = f_abs.median()
        mad = (f_abs - med).abs().median()
        # MAD can be 0 for constant data; fall back to percentile in that case
        if mad < 1e-12:
            cutoff = f_abs.quantile(min(percentile / 100.0, 0.999))
        else:
            cutoff = med + threshold * mad
    elif method == "percentile":
        cutoff = f_abs.quantile(min(percentile / 100.0, 0.999))
    else:
        raise ValueError(f"Unknown outlier filter method: {method}")

    mask = f_abs <= cutoff

    # Ensure we keep enough samples
    if mask.sum() < 50:
        # If filtering removes too much, keep top 95% by magnitude
        cutoff = f_abs.quantile(0.95)
        mask = f_abs <= cutoff

    return X[mask], f_flat[mask]


def _fit_poly_coeffs_1d(
    x: torch.Tensor,
    f: torch.Tensor,
    degree: int,
    min_points: int = 200,
    dtype: torch.dtype = torch.float64,
) -> Optional[torch.Tensor]:
    """
    Fit f(x) ≈ P(x) as a 1D polynomial of given degree, using the
    *same* monomial enumeration as PolyLeaf / RationalPolyLeaf.
    """
    x = x.view(-1, 1).to(dtype=dtype)
    f = f.view(-1).to(dtype=dtype)
    N = x.shape[0]
    if N < min_points:
        return None

    # 1D exponents: [(0,), (1,), ..., (degree,)]
    exps = _enumerate_exponents(1, degree)
    exps_t = torch.tensor(exps, dtype=torch.int64, device=x.device)
    Phi = _eval_monomials(x, exps_t)  # [N, M]
    M = Phi.shape[1]
    if N < (M + 5):
        return None

    G = Phi.T @ Phi
    G = G + 1e-10 * torch.eye(M, dtype=G.dtype, device=G.device)
    rhs = Phi.T @ f
    coeffs = torch.linalg.solve(G, rhs)
    try:
        coeffs_sparse, meta = stlsq_sparsify_poly_coeffs(
            Phi=Phi,
            y=f,
            coeffs=coeffs,
            cfg=DEFAULT_POLY_STLSQ_CFG,
        )
        _log_sparsify_result(
            "_fit_poly_coeffs_1d", coeffs, None, coeffs_sparse, None, meta,
        )
        coeffs = coeffs_sparse
    except Exception as exc:
        _log.info("[_fit_poly_coeffs_1d] poly sparsify failed: %s", exc)
    return coeffs


def _fit_rational_coeffs_1d(
    x: torch.Tensor,
    f: torch.Tensor,
    deg_num: int,
    deg_den: int,
    min_points: int = 200,
    dtype: torch.dtype = torch.float64,
    min_total_num: int = 0,
    min_total_den: int = 0,
    return_support: bool = False,
    return_support_indices: bool = False,
    exps_num_override: Optional[Sequence[Sequence[int]]] = None,
    exps_den_override: Optional[Sequence[Sequence[int]]] = None,
) -> Optional[
    Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
]:
    """
    Fit f(x) ≈ P(x)/Q(x) with 1D polynomials P,Q of given degrees,
    using the same monomial enumeration as RationalPolyLeaf.

    When *min_total_num* / *min_total_den* equal the respective degrees,
    only the homogeneous (highest-degree) monomial is kept, ensuring
    dimensional consistency when the input variable carries physical units.
    """
    if bool(return_support) and bool(return_support_indices):
        raise ValueError("return_support and return_support_indices are mutually exclusive")

    x = x.view(-1, 1).to(dtype=dtype)
    f = f.view(-1).to(dtype=dtype)
    N = x.shape[0]
    if N < min_points:
        return None

    def _validated_override(
        rows: Sequence[Sequence[int]],
        *,
        degree: int,
        min_total: int,
        label: str,
    ) -> List[Tuple[int, ...]]:
        out: List[Tuple[int, ...]] = []
        seen: set[Tuple[int, ...]] = set()
        for row_index, raw_row in enumerate(rows):
            try:
                values = tuple(raw_row)
            except TypeError as exc:
                raise ValueError(f"{label}[{row_index}] must be an exponent row") from exc
            if len(values) != 1:
                raise ValueError(
                    f"{label}[{row_index}] must contain exactly one exponent"
                )
            raw_value = values[0]
            if hasattr(raw_value, "item"):
                try:
                    raw_value = raw_value.item()
                except Exception:
                    pass
            if isinstance(raw_value, bool) or not isinstance(raw_value, Integral):
                raise ValueError(
                    f"{label}[{row_index}][0] must be an exact integer, "
                    f"got {raw_value!r}"
                )
            value = int(raw_value)
            if value < 0:
                raise ValueError(
                    f"{label}[{row_index}][0] must be nonnegative, got {value}"
                )
            if value < int(min_total) or value > int(degree):
                raise ValueError(
                    f"{label}[{row_index}] total degree {value} is outside "
                    f"[{int(min_total)}, {int(degree)}]"
                )
            exponent = (value,)
            if exponent in seen:
                raise ValueError(f"{label} repeats exponent {exponent}")
            seen.add(exponent)
            out.append(exponent)
        if not out:
            raise ValueError(f"{label} must contain at least one exponent row")
        return out

    # 1D monomial bases matching RationalPolyLeaf.  Exact overrides let
    # unit-aware producers fit only coefficient-admissible dimension classes.
    exps_num = (
        _enumerate_exponents(1, deg_num, min_total=min_total_num)
        if exps_num_override is None
        else _validated_override(
            exps_num_override,
            degree=deg_num,
            min_total=min_total_num,
            label="exps_num_override",
        )
    )
    exps_den = (
        _enumerate_exponents(1, deg_den, min_total=min_total_den)
        if exps_den_override is None
        else _validated_override(
            exps_den_override,
            degree=deg_den,
            min_total=min_total_den,
            label="exps_den_override",
        )
    )
    exps_num_t = torch.tensor(exps_num, dtype=torch.int64, device=x.device)
    exps_den_t = torch.tensor(exps_den, dtype=torch.int64, device=x.device)
    if (
        exps_num_t.ndim != 2
        or int(exps_num_t.shape[1]) != 1
        or int(exps_num_t.shape[0]) <= 0
        or exps_den_t.ndim != 2
        or int(exps_den_t.shape[1]) != 1
        or int(exps_den_t.shape[0]) <= 0
    ):
        return None

    Phi_num = _eval_monomials(x, exps_num_t)  # [N, M_num]
    Phi_den = _eval_monomials(x, exps_den_t)  # [N, M_den]
    M_num = Phi_num.shape[1]
    M_den = Phi_den.shape[1]

    if N < (M_num + M_den + 5):
        return None

    # Linear system A c ≈ 0 where c = [a; b]
    F_col = f.unsqueeze(1)  # [N, 1]
    A_left = Phi_num  # [N, M_num]
    A_right = -F_col * Phi_den  # [N, M_den]
    A = torch.cat([A_left, A_right], dim=1)  # [N, M_num + M_den]

    Gram = (A.T @ A) / float(N)
    Gram = Gram.to(dtype=dtype)
    evals, vecs = torch.linalg.eigh(Gram)
    evals = evals.clamp_min(0.0)
    c = vecs[:, 0]  # smallest eigenvalue

    a = c[:M_num]
    b = c[M_num:]

    # Normalise so that b[0] ≈ 1 if that is safe
    if abs(b[0]) > 1e-12:
        a = a / b[0]
        b = b / b[0]

    # Ensure denominator Q(x) is mostly positive (required by RationalPolyLeaf's clamp_min)
    with torch.no_grad():
        Q_vals = Phi_den @ b
        frac_positive = float((Q_vals > 0).sum()) / max(1, Q_vals.numel())
        if frac_positive < 0.5:
            # Flip signs so Q is mostly positive
            a = -a
            b = -b

    # De-Padeify / de-rationalize tiny terms using shared STLSQ machinery.
    try:
        a_sparse, b_sparse, meta = stlsq_sparsify_rational_coeffs(
            Phi_num=Phi_num,
            Phi_den=Phi_den,
            y=f,
            coeffs_num=a,
            coeffs_den=b,
            cfg=DEFAULT_RAT_STLSQ_CFG,
        )
        _log_sparsify_result("_fit_rational_coeffs_1d", a, b, a_sparse, b_sparse, meta)
        a, b = a_sparse, b_sparse
    except Exception as exc:
        _log.info("[_fit_rational_coeffs_1d] rational sparsify failed: %s", exc)

    if (not bool(return_support)) and (not bool(return_support_indices)):
        return a, b

    support_eps = 1e-14
    mask_num = (a.abs() > support_eps)
    if int(mask_num.numel()) > 0 and int(mask_num.sum().item()) == 0:
        mask_num = torch.zeros_like(mask_num, dtype=torch.bool)
        mask_num[int(a.abs().argmax().item())] = True

    mask_den = (b.abs() > support_eps)
    if int(mask_den.numel()) > 0 and int(mask_den.sum().item()) == 0:
        mask_den = torch.zeros_like(mask_den, dtype=torch.bool)
        mask_den[int(b.abs().argmax().item())] = True

    idx_num = torch.nonzero(mask_num, as_tuple=False).view(-1).to(dtype=torch.int64)
    idx_den = torch.nonzero(mask_den, as_tuple=False).view(-1).to(dtype=torch.int64)
    a_sparse = a[idx_num].clone()
    b_sparse = b[idx_den].clone()

    if bool(return_support_indices):
        return a_sparse, b_sparse, idx_num, idx_den

    exps_num_sparse = exps_num_t[idx_num].clone()
    exps_den_sparse = exps_den_t[idx_den].clone()
    return a_sparse, b_sparse, exps_num_sparse, exps_den_sparse


def _fit_rational_coeffs_nd(
    X: torch.Tensor,
    f: torch.Tensor,
    exps_num: torch.Tensor,
    exps_den: torch.Tensor,
    min_points: int = 200,
    dtype: torch.dtype = torch.float64,
    subsample_frac: float = 1.0,
    eps_Q: float = 1e-10,
    seed: int = 0,
    return_support: bool = False,
    return_support_indices: bool = False,
) -> Optional[
    Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
]:
    """
    ND analogue of _fit_rational_coeffs_1d, but using *given* exponent
    tables exps_num / exps_den (typically from an ExpRationalPolyLeaf).
    Solves for P/Q with P,Q polynomials sharing those monomial supports.

    Args:
        subsample_frac: Fraction of data to subsample for fitting (1.0 = all data).
            Note: This is different from fit_frac in _rational_probe_nd which
            controls the train/validation split ratio.
        eps_Q: Unused here but accepted for API compatibility.
        seed: Random seed for reproducible subsampling when subsample_frac < 1.
        return_support: When True, also return reduced exponent tables for
            active monomials after de-Padeification.
        return_support_indices: When True, return support indices into the
            dense exponent tables instead of sliced exponent tables.
    """
    if bool(return_support) and bool(return_support_indices):
        raise ValueError("return_support and return_support_indices are mutually exclusive")

    X = X.to(dtype=dtype)
    f = f.view(-1).to(dtype=dtype)
    N, dim = X.shape

    # Subsample if subsample_frac < 1
    if subsample_frac < 1.0 and N > min_points:
        gen = torch.Generator(device=X.device)
        gen.manual_seed(seed)
        n_use = max(min_points, int(N * subsample_frac))
        perm = torch.randperm(N, generator=gen, device=X.device)[:n_use]
        X = X[perm]
        f = f[perm]
        N = n_use

    if N < min_points:
        return None

    exps_num_t = exps_num.to(device=X.device, dtype=torch.int64)
    exps_den_t = exps_den.to(device=X.device, dtype=torch.int64)

    Phi_num = _eval_monomials(X, exps_num_t)  # [N, M_num]
    Phi_den = _eval_monomials(X, exps_den_t)  # [N, M_den]
    M_num = Phi_num.shape[1]
    M_den = Phi_den.shape[1]

    if N < (M_num + M_den + 5):
        return None

    F_col = f.unsqueeze(1)
    A_left = Phi_num
    A_right = -F_col * Phi_den
    A = torch.cat([A_left, A_right], dim=1)  # [N, M_num+M_den]

    Gram = (A.T @ A) / float(N)
    Gram = Gram.to(dtype=dtype)
    evals, vecs = torch.linalg.eigh(Gram)
    evals = evals.clamp_min(0.0)
    c = vecs[:, 0]

    a = c[:M_num]
    b = c[M_num:]

    if b.numel() > 0 and abs(float(b[0])) > 1e-12:
        a = a / b[0]
        b = b / b[0]

    # Ensure denominator Q(x) is mostly positive (required by RationalPolyLeaf's clamp_min)
    with torch.no_grad():
        Q_vals = Phi_den @ b
        frac_positive = float((Q_vals > 0).sum()) / max(1, Q_vals.numel())
        if frac_positive < 0.5:
            # Flip signs so Q is mostly positive
            a = -a
            b = -b

    # De-Padeify / de-rationalize tiny terms using shared STLSQ machinery.
    try:
        a_sparse, b_sparse, meta = stlsq_sparsify_rational_coeffs(
            Phi_num=Phi_num,
            Phi_den=Phi_den,
            y=f,
            coeffs_num=a,
            coeffs_den=b,
            cfg=DEFAULT_RAT_STLSQ_CFG,
        )
        _log_sparsify_result("_fit_rational_coeffs_nd", a, b, a_sparse, b_sparse, meta)
        a, b = a_sparse, b_sparse
    except Exception as exc:
        _log.info("[_fit_rational_coeffs_nd] rational sparsify failed: %s", exc)

    if (not bool(return_support)) and (not bool(return_support_indices)):
        return a, b

    # Build explicit active support so callers can instantiate reduced leaves
    # whose trainable parameter vectors only include surviving monomials.
    support_eps = 1e-14
    mask_num = (a.abs() > support_eps)
    if int(mask_num.numel()) > 0 and int(mask_num.sum().item()) == 0:
        mask_num = torch.zeros_like(mask_num, dtype=torch.bool)
        mask_num[int(a.abs().argmax().item())] = True

    mask_den = (b.abs() > support_eps)
    if int(mask_den.numel()) > 0 and int(mask_den.sum().item()) == 0:
        mask_den = torch.zeros_like(mask_den, dtype=torch.bool)
        mask_den[int(b.abs().argmax().item())] = True

    idx_num = torch.nonzero(mask_num, as_tuple=False).view(-1).to(dtype=torch.int64)
    idx_den = torch.nonzero(mask_den, as_tuple=False).view(-1).to(dtype=torch.int64)
    a_sparse = a[idx_num].clone()
    b_sparse = b[idx_den].clone()

    if bool(return_support_indices):
        return a_sparse, b_sparse, idx_num, idx_den

    exps_num_sparse = exps_num_t[idx_num].clone()
    exps_den_sparse = exps_den_t[idx_den].clone()
    return a_sparse, b_sparse, exps_num_sparse, exps_den_sparse


def _rational_probe_nd(
    X: torch.Tensor,
    f: torch.Tensor,
    deg_num: int = 2,
    deg_den: int = 2,
    min_points: int = 200,
    max_points: int = 1000,
    dtype: torch.dtype = torch.float64,
    fit_frac: float = 0.7,
    eps_Q: float = 1e-8,
    seed: int = 0,
    return_coeffs: bool = False,
    filter_outliers: bool = False,
    outlier_method: str = "mad",
    outlier_threshold: float = 20.0,
    error_metric: str = "rms_rel",
    min_total_num: int = 0,
    min_total_den: int = 0,
    exps_num_override: Optional[Sequence[Sequence[int]]] = None,
    exps_den_override: Optional[Sequence[Sequence[int]]] = None,
) -> Union[float, Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """
    Cheap multi-dimensional rational fit quality probe:

        f(x) ≈ P(x)/Q(x)

    with P,Q polynomials of total degree ≤ deg_num / deg_den.

    Returns a relative error metric. On any failure, returns +inf.

    Args:
        fit_frac: Fraction of data to use for fitting (rest used for validation).
        eps_Q: Minimum absolute value of Q for valid evaluation points.
        seed: Random seed for reproducible train/val split.
        return_coeffs: If True, also return the fitted coefficients (a, b) for
            P(x) = sum_i a_i * m_i(x) and Q(x) = sum_j b_j * n_j(x).
        filter_outliers: If True, pre-filter samples with extreme |f| values
            to handle near-singular expressions (default False).
        outlier_method: Method for outlier detection: 'mad' (median absolute
            deviation) or 'percentile' (default 'mad').
        outlier_threshold: For 'mad' method, number of MADs above median to
            keep (default 20.0).
        error_metric: Error metric to return: 'rms_rel' (original, default)
            or 'median_rel' (robust to outliers).

    Returns:
        If return_coeffs is False: relative error (float).
        If return_coeffs is True: (error, coeffs_num, coeffs_den) where
            coeffs are None on failure.
    """
    def _fail():
        return (float("inf"), None, None) if return_coeffs else float("inf")

    try:
        X = X.to(dtype=dtype)
        f = f.view(-1).to(dtype=dtype)
        N, dim = X.shape
        if N < min_points:
            return _fail()

        # Pre-filter outliers to handle near-singular samples
        if filter_outliers:
            X, f = _filter_outliers(X, f, method=outlier_method, threshold=outlier_threshold)
            N = X.size(0)
            if N < min_points:
                return _fail()

        Np = min(N, max_points)
        # Shuffle with seed for reproducible split (sample from ALL N points, not just first Np)
        gen = torch.Generator(device=X.device)
        gen.manual_seed(seed)
        perm = torch.randperm(N, generator=gen, device=X.device)[:Np]
        Xp = X[perm]
        fp = f[perm]

        # Split into fit and validation sets
        n_fit = max(min_points // 2, int(Np * fit_frac))
        X_fit, X_val = Xp[:n_fit], Xp[n_fit:]
        f_fit, f_val = fp[:n_fit], fp[n_fit:]

        if exps_num_override is None:
            exps_num_t = torch.tensor(
                _enumerate_exponents(dim, deg_num, min_total=min_total_num),
                dtype=torch.int64,
                device=Xp.device,
            )
        else:
            exps_num_t = torch.tensor(exps_num_override, dtype=torch.int64, device=Xp.device)

        if exps_den_override is None:
            exps_den_t = torch.tensor(
                _enumerate_exponents(dim, deg_den, min_total=min_total_den),
                dtype=torch.int64,
                device=Xp.device,
            )
        else:
            exps_den_t = torch.tensor(exps_den_override, dtype=torch.int64, device=Xp.device)

        if exps_num_t.ndim != 2 or exps_num_t.shape[1] != dim:
            return _fail()
        if exps_den_t.ndim != 2 or exps_den_t.shape[1] != dim:
            return _fail()

        M_num = int(exps_num_t.shape[0])
        M_den = int(exps_den_t.shape[0])
        if M_num <= 0 or M_den <= 0:
            return _fail()
        if n_fit < (M_num + M_den + 5):
            return _fail()

        # Fit on training set
        Phi_num_fit = _eval_monomials(X_fit, exps_num_t)  # [n_fit, M_num]
        Phi_den_fit = _eval_monomials(X_fit, exps_den_t)  # [n_fit, M_den]

        F_col = f_fit.unsqueeze(1)
        A_left = Phi_num_fit
        A_right = -F_col * Phi_den_fit
        A = torch.cat([A_left, A_right], dim=1)

        Gram = (A.T @ A) / float(n_fit)
        Gram = Gram.to(dtype=dtype)
        evals, vecs = torch.linalg.eigh(Gram)
        evals = evals.clamp_min(0.0)
        c = vecs[:, 0]
        a = c[:M_num]
        b = c[M_num:]
        if b.numel() > 0 and abs(float(b[0])) > 1e-12:
            a = a / b[0]
            b = b / b[0]

        # Ensure Q is mostly positive (matching RationalPolyLeaf's clamp_min semantics)
        # Evaluate Q on fit data to decide sign
        with torch.no_grad():
            Q_fit = Phi_den_fit @ b
            frac_positive = float((Q_fit > 0).sum()) / max(1, Q_fit.numel())
            if frac_positive < 0.5:
                # Flip signs so Q is mostly positive
                a = -a
                b = -b
                frac_positive = 1.0 - frac_positive
            # Reject if Q is still negative too often (even after sign flip)
            # RationalPolyLeaf uses clamp_min, so negative Q becomes eps and explodes P/Q
            if frac_positive < 0.9:
                return _fail()

        # NOTE: no de-Padeification here — _rational_probe_nd is a cheap
        # screening heuristic.  Sparsifying the coefficients can change the
        # returned error unpredictably (especially for ill-conditioned raw-space
        # baselines), which cascades into acceptance filters in the NLS screen.
        # De-Padeification is applied later when the coefficients are actually
        # used (in _fit_rational_coeffs_nd / _fit_rational_coeffs_1d).

        # Evaluate on validation set (or all data if validation set too small)
        with torch.no_grad():
            if len(f_val) >= max(10, M_num + M_den):
                X_eval, f_eval = X_val, f_val
            else:
                X_eval, f_eval = Xp, fp

            Phi_num_eval = _eval_monomials(X_eval, exps_num_t)
            Phi_den_eval = _eval_monomials(X_eval, exps_den_t)
            Q = Phi_den_eval @ b
            # Use clamp_min to match RationalPolyLeaf semantics (not abs())
            Q_clamped = Q.clamp_min(eps_Q)
            # Still mask out points where Q was originally <= 0 (clamping distorts these)
            mask = Q > eps_Q
            if mask.sum().item() < max(10, M_num + M_den):
                return _fail()
            Pm = Phi_num_eval[mask] @ a
            Qm = Q_clamped[mask]
            fm = f_eval[mask]
            y_pred = Pm / Qm
            resid = y_pred - fm

            # Compute error metric
            if error_metric == "rms_rel":
                # Original RMS-based relative error
                rms_abs = float(torch.sqrt(torch.mean(resid * resid)))
                std_target = float(fm.std(unbiased=False))
                if std_target < 1e-12:
                    err = 0.0 if rms_abs < 1e-12 else float("inf")
                else:
                    err = rms_abs / std_target
            elif error_metric == "median_rel":
                # Robust median-based relative error
                scale = fm.abs().median().clamp_min(1e-12)
                err = float((resid.abs().median() / scale).item())
            else:
                raise ValueError(f"Unknown error_metric: {error_metric}")

            if return_coeffs:
                return err, a.detach().clone(), b.detach().clone()
            return err
    except Exception:
        return _fail()


def _nonlinear_substitution_screen(
    X: torch.Tensor,
    F: torch.Tensor,
    teacher: Optional[nn.Module] = None,
    max_deg_num: int = 3,
    max_deg_den: int = 3,
    extra_degree_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    threshold: float = 0.02,
    max_points: int = 2000,
    min_points: int = 200,
    trig_hints: Optional[Dict[int, str]] = None,
    outer_transforms: Optional[List[str]] = None,
    square_sign_consistency: float = 0.98,
    target_dim: Optional[Sequence[Any]] = None,
    input_dims: Optional[Sequence[Sequence[Any]]] = None,
    coefficient_policy: str = "free_const_only",
    max_attempts: int = 1024,
    max_support_attempts: int = 2048,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """Screen nonlinear variable substitutions that make a leaf rational.

    For each column of *X*, try replacing it with T(col) for T in
    {cos, sin, exp, log}.  If the result is well-approximated by a
    low-degree rational P/Q in the transformed coordinates, return a
    hit describing the transform, degrees, and fit error.

    A **parity pre-screen** (one extra forward pass through *teacher*)
    narrows the candidate transforms cheaply:
    - even symmetry in a variable → only try cos
    - odd symmetry → only try sin
    - neither → try all four

    Optionally, **outer transforms** (``"square"``, ``"reciprocal"``)
    probe whether a power of the output is rational even when the raw
    output is not.  For instance if ``F = sqrt(P/Q)`` then ``F²`` is
    rational.  Only accepted when the outer-transformed fit is
    significantly better than the best identity-output fit.

    Parameters
    ----------
    X : Tensor [N, dim]
        Leaf input coordinates.
    F : Tensor [N]
        Leaf output values.
    teacher : nn.Module, optional
        The NN leaf module, used for the parity check. If ``None``,
        parity is skipped and all transforms are tried.
    max_deg_num, max_deg_den : int
        Maximum polynomial degrees for the ordinary numerator / denominator
        search rectangle.
    extra_degree_pairs : sequence of (int, int), optional
        Additional numerator/denominator degree pairs to probe without opening
        the intervening rectangular search space.
    threshold : float
        Relative-error acceptance threshold.
    max_points, min_points : int
        Data sub-sampling bounds passed to ``_rational_probe_nd``.
    trig_hints : dict, optional
        Mapping ``{col_idx: "cos"|"sin"}`` from Stage A trig detection.
        When present for a column, overrides the teacher-based parity
        pre-screen (more reliable than the reflection test).
    outer_transforms : list of str, optional
        Extra output transforms to probe, e.g. ``["square", "reciprocal"]``.
        Accepted only when ``err_outer < threshold`` **and**
        ``err_outer < 0.5 * best_identity_err`` for this (col, T) combo.
    square_sign_consistency : float, optional
        Minimum dominant-sign fraction required for ``outer_transform="square"``.
        This avoids building ``sqrt(ratpoly)`` wrappers for sign-ambiguous targets.
    target_dim, input_dims : sequences, optional
        Exact dimensions of the leaf output and its effective input coordinates.
        Supplying either requires supplying both.  The screen then constructs
        anonymous-coefficient supports with the shared coefficient gauge solver
        before every numerical probe.  Dimensionless data follows this same path.
    coefficient_policy : str, optional
        Coefficient policy forwarded to support planning.  The current rational
        leaves contain anonymous coefficients, so only ``free_const_only`` is
        admissible.
    max_attempts, max_support_attempts : int, optional
        Independent bounds for numerical proposal probes and exact structural
        support checks.
    diagnostics : dict, optional
        Mutable mapping populated with attempt, rejection, emission, and
        exhaustion diagnostics.

    Returns
    -------
    List[Dict]
        Hits sorted by ascending error.  Each dict has keys:
        ``col_idx``, ``transform``, ``parity``, ``deg_num``,
        ``deg_den``, ``error``, and optionally ``outer_transform``.
        For ``outer_transform="square"``, ``sign_hint`` and
        ``sign_consistency`` are also returned.
    """

    extra_pairs = tuple(
        sorted({(int(pair[0]), int(pair[1])) for pair in (extra_degree_pairs or ())})
    )
    if any(deg_num < 0 or deg_den < 0 for deg_num, deg_den in extra_pairs):
        raise ValueError("nonlinear-substitution degree pairs must be non-negative")

    stats: Dict[str, Any] = {
        "unit_aware": False,
        "raw_attempted": 0,
        "baseline_attempted": 0,
        "support_raw_attempted": 0,
        "unit_rejected": 0,
        "numeric_rejected": 0,
        "deduplicated": 0,
        "emitted": 0,
        "exhausted": True,
        "exhaustion_reason": "candidate_space_exhausted",
        "truncated_by_attempt_budget": False,
        "numeric_attempt_budget_exhausted": False,
        "support_attempt_budget_exhausted": False,
        "max_attempts": max(0, int(max_attempts)),
        "max_support_attempts": max(0, int(max_support_attempts)),
        "extra_degree_pairs": [list(pair) for pair in extra_pairs],
        "reason_counts": {},
    }

    def _record_reason(code: str, count: int = 1) -> None:
        reasons = stats["reason_counts"]
        reasons[str(code)] = int(reasons.get(str(code), 0)) + int(count)

    def _finish(rows: List[Dict]) -> List[Dict]:
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row.get("unit_support_rank", 0)),
                float(row.get("error", float("inf"))),
                int(row.get("unit_support_complexity", 10**9)),
                int(row.get("deg_num", 0)) + int(row.get("deg_den", 0)),
                int(row.get("col_idx", 0)),
                str(row.get("transform", "")),
                str(row.get("outer_transform", "identity")),
            ),
        )
        stats["emitted"] = int(len(ordered))
        if bool(stats["truncated_by_attempt_budget"]):
            stats["exhaustion_reason"] = "attempt_budget_exhausted"
        else:
            stats["exhaustion_reason"] = "candidate_space_exhausted"
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(stats)
        return ordered

    X = X.to(dtype=torch.float64)
    F = F.view(-1).to(dtype=torch.float64)
    N, dim = X.shape
    if N < min_points:
        _record_reason("insufficient_points")
        return _finish([])

    unit_requested = target_dim is not None or input_dims is not None
    unit_target = None
    unit_inputs = None
    if unit_requested:
        if target_dim is None or input_dims is None:
            stats["unit_rejected"] += 1
            _record_reason("incomplete_unit_dimensions")
            return _finish([])
        try:
            from nestynet_sr.sr_core.coefficient_units import normalize_dimension

            unit_target = normalize_dimension(target_dim, label="target_dim")
            unit_inputs = tuple(
                normalize_dimension(
                    item,
                    rank=len(unit_target),
                    label=f"input_dims[{index}]",
                )
                for index, item in enumerate(input_dims)
            )
            if len(unit_inputs) != dim:
                raise ValueError(
                    f"input_dims has arity {len(unit_inputs)}; expected {dim}"
                )
            if not unit_target:
                raise ValueError("target_dim must have positive rank")
            policy = str(coefficient_policy or "free_const_only").strip().lower().replace("-", "_")
            if policy not in {"free_const_only", "dimensionless", "strict"}:
                raise ValueError(
                    "anonymous rational coefficients require free_const_only"
                )
            coefficient_policy = "free_const_only"
            stats["unit_aware"] = True
        except Exception:
            stats["unit_rejected"] += 1
            _record_reason("invalid_unit_dimensions")
            return _finish([])

    legacy_trials = []
    legacy_seen_degrees = set()
    for dn in range(1, int(max_deg_num) + 1):
        for dd in range(0, int(max_deg_den) + 1):
            actual_dd = int(max(dd, 1) if dd > 0 else 1)
            degree_key = (int(dn), actual_dd)
            if degree_key in legacy_seen_degrees:
                continue
            legacy_seen_degrees.add(degree_key)
            legacy_trials.append(
                {
                    "deg_num": int(dn),
                    "deg_den": actual_dd,
                    "parsimony_deg_den": int(dd),
                    "complexity": int(dn + dd),
                    "exps_num_override": None,
                    "exps_den_override": None,
                }
            )
    for dn, dd in extra_pairs:
        actual_dd = int(max(dd, 1) if dd > 0 else 1)
        degree_key = (int(dn), actual_dd)
        if degree_key in legacy_seen_degrees:
            continue
        legacy_seen_degrees.add(degree_key)
        legacy_trials.append(
            {
                "deg_num": int(dn),
                "deg_den": actual_dd,
                "parsimony_deg_den": int(dd),
                "complexity": int(dn + dd),
                "exps_num_override": None,
                "exps_den_override": None,
            }
        )

    support_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    structural_attempt_limit = max(0, int(max_support_attempts))

    def _support_trials(
        rational_target_dim,
        rational_input_dims,
    ) -> List[Dict[str, Any]]:
        if not bool(stats["unit_aware"]):
            return legacy_trials
        key = (
            tuple(rational_target_dim),
            tuple(tuple(item) for item in rational_input_dims),
            int(max_deg_num),
            int(max_deg_den),
            extra_pairs,
            str(coefficient_policy),
        )
        if key in support_cache:
            return support_cache[key]

        structural_attempts_remaining = max(
            0,
            structural_attempt_limit - int(stats["support_raw_attempted"]),
        )
        if structural_attempts_remaining <= 0:
            stats["truncated_by_attempt_budget"] = True
            if not bool(stats["support_attempt_budget_exhausted"]):
                stats["support_attempt_budget_exhausted"] = True
                _record_reason("support_attempt_budget_exhausted")
            support_cache[key] = []
            return support_cache[key]

        from .rational_supports import plan_unit_consistent_rational_supports

        trials = []
        seen_supports = set()
        plan_bounds = [(int(max_deg_num), int(max_deg_den), None)] + [
            (int(deg_num), int(deg_den), (int(deg_num), max(1, int(deg_den))))
            for deg_num, deg_den in extra_pairs
        ]
        for plan_deg_num, plan_deg_den, required_pair in plan_bounds:
            structural_attempts_remaining = max(
                0,
                structural_attempt_limit - int(stats["support_raw_attempted"]),
            )
            if structural_attempts_remaining <= 0:
                stats["truncated_by_attempt_budget"] = True
                stats["support_attempt_budget_exhausted"] = True
                _record_reason("support_attempt_budget_exhausted")
                break
            plan = plan_unit_consistent_rational_supports(
                target_dim=rational_target_dim,
                input_dims=rational_input_dims,
                max_deg_num=plan_deg_num,
                max_deg_den=plan_deg_den,
                coefficient_policy=coefficient_policy,
                max_attempts=structural_attempts_remaining,
            )
            plan_stats = plan.diagnostics()
            stats["support_raw_attempted"] += int(plan_stats["raw_attempted"])
            stats["unit_rejected"] += int(plan_stats["unit_rejected"])
            stats["deduplicated"] += int(plan_stats["deduplicated"])
            for code, count in dict(plan_stats.get("reason_counts") or {}).items():
                _record_reason(str(code), int(count))
            if bool(plan_stats["truncated_by_attempt_budget"]):
                stats["truncated_by_attempt_budget"] = True
                if not bool(stats["support_attempt_budget_exhausted"]):
                    stats["support_attempt_budget_exhausted"] = True
                    _record_reason("support_attempt_budget_exhausted")
            for support in plan.supports:
                payload = support.to_dict()
                public_pair = (int(payload["deg_num"]), int(payload["deg_den"]))
                if required_pair is not None and public_pair != required_pair:
                    continue
                support_key = (
                    tuple(tuple(row) for row in payload["exps_num_override"]),
                    tuple(tuple(row) for row in payload["exps_den_override"]),
                )
                if support_key in seen_supports:
                    continue
                seen_supports.add(support_key)
                payload["parsimony_deg_den"] = int(payload["deg_den"])
                trials.append(payload)
        support_cache[key] = trials
        return trials

    numerical_attempt_limit = max(0, int(max_attempts))

    def _probe_trials(
        X_probe: torch.Tensor,
        F_probe: torch.Tensor,
        trials: Sequence[Dict[str, Any]],
        *,
        baseline: bool,
    ) -> List[Tuple[Dict[str, Any], float]]:
        fits: List[Tuple[Dict[str, Any], float]] = []
        for trial in trials:
            if not baseline and int(stats["raw_attempted"]) >= numerical_attempt_limit:
                stats["truncated_by_attempt_budget"] = True
                stats["numeric_attempt_budget_exhausted"] = True
                _record_reason("numeric_attempt_budget_exhausted")
                break
            if baseline:
                stats["baseline_attempted"] += 1
            else:
                stats["raw_attempted"] += 1
            try:
                err = _rational_probe_nd(
                    X_probe,
                    F_probe,
                    deg_num=int(trial["deg_num"]),
                    deg_den=int(trial["deg_den"]),
                    max_points=max_points,
                    min_points=min_points,
                    filter_outliers=True,
                    exps_num_override=trial.get("exps_num_override"),
                    exps_den_override=trial.get("exps_den_override"),
                )
            except Exception:
                err = float("inf")
            fits.append((dict(trial), float(err)))
        return fits

    def _trial_order(item: Tuple[Dict[str, Any], float]) -> Tuple[Any, ...]:
        trial, err = item
        return (
            int(trial.get("complexity", 10**9)),
            int(trial["deg_num"]) + int(trial.get("parsimony_deg_den", trial["deg_den"])),
            int(trial["deg_num"]),
            int(trial["deg_den"]),
            float(err),
        )

    def _hit_from_fit(
        *,
        col_idx: int,
        transform: str,
        parity: str,
        item: Tuple[Dict[str, Any], float],
        rational_target_dim=None,
        rational_input_dims=None,
    ) -> Dict[str, Any]:
        trial, err = item
        hit: Dict[str, Any] = {
            "col_idx": int(col_idx),
            "transform": str(transform),
            "parity": str(parity),
            "deg_num": int(trial["deg_num"]),
            "deg_den": int(trial["deg_den"]),
            "error": float(err),
        }
        if bool(stats["unit_aware"]):
            hit.update(
                {
                    "unit_support_planned": True,
                    "unit_support_complexity": int(trial["complexity"]),
                    "exps_num_override": [
                        list(row) for row in trial["exps_num_override"]
                    ],
                    "exps_den_override": [
                        list(row) for row in trial["exps_den_override"]
                    ],
                    "coefficient_policy": str(coefficient_policy),
                    "rational_target_dim": tuple(rational_target_dim),
                    "transformed_input_dims": tuple(
                        tuple(item) for item in rational_input_dims
                    ),
                    "coefficient_unit_certificate": dict(
                        trial["coefficient_unit_certificate"]
                    ),
                }
            )
        return hit

    # Multivariate mode is prone to accidental fits; require stronger evidence
    # than the caller threshold and demand clear improvement over raw-space fit.
    strong_threshold = float(threshold)
    baseline_best_err = float("inf")
    if dim > 1:
        strong_threshold = min(float(threshold), 0.02)
        baseline_trials = _support_trials(unit_target, unit_inputs)
        baseline_fits = _probe_trials(
            X,
            F,
            baseline_trials,
            baseline=True,
        )
        finite_baseline = [err for _trial, err in baseline_fits if math.isfinite(err)]
        if finite_baseline:
            baseline_best_err = min(finite_baseline)

    _TRANSFORMS = [
        ("cos", torch.cos),
        ("sin", torch.sin),
        ("exp", torch.exp),
        ("log", torch.log),
    ]

    results: List[Dict] = []

    for col_idx in range(dim):
        v = X[:, col_idx]

        # ---- parity pre-screen ----
        parity = "none"
        # trig hint from Stage A (most reliable) overrides reflection test
        if trig_hints and col_idx in trig_hints:
            parity = "even" if trig_hints[col_idx] == "cos" else "odd"
        elif teacher is not None:
            try:
                med = v.median()
                X_ref = X.clone()
                X_ref[:, col_idx] = 2.0 * med - v  # reflect around median
                with torch.no_grad():
                    # Determine device/dtype from teacher parameters (if any)
                    _p = next(teacher.parameters(), None)
                    _dt = _p.dtype if _p is not None else X_ref.dtype
                    _dv = _p.device if _p is not None else X_ref.device
                    F_ref = teacher(X_ref.to(dtype=_dt, device=_dv))
                    F_ref = F_ref.view(-1).to(dtype=torch.float64, device=F.device)
                std_F = F.std()
                if std_F > 1e-12:
                    even_rms = (F - F_ref).pow(2).mean().sqrt() / std_F
                    odd_rms = (F + F_ref).pow(2).mean().sqrt() / std_F
                    if float(even_rms) < 0.1:
                        parity = "even"
                    elif float(odd_rms) < 0.1:
                        parity = "odd"
            except Exception:
                parity = "none"

        for tname, tfn in _TRANSFORMS:
            # Parity filter: when trig symmetry is detected, skip non-trig
            # transforms but always try BOTH cos and sin — phase detection
            # can be unreliable for compound variables.
            if parity in ("even", "odd") and tname not in ("cos", "sin"):
                continue

            transformed_dims = unit_inputs
            if bool(stats["unit_aware"]):
                from nestynet_sr.sr_core.units import is_dimless

                if not is_dimless(unit_inputs[col_idx]):
                    stats["unit_rejected"] += 1
                    _record_reason("non_dimensionless_transform_argument")
                    continue
                zero = tuple(Fraction(0) for _ in unit_target)
                transformed_dims_list = list(unit_inputs)
                transformed_dims_list[col_idx] = zero
                transformed_dims = tuple(transformed_dims_list)

            # Domain checks
            if tname == "log" and float(v.min()) <= 0:
                continue
            if tname == "exp" and float(v.abs().max()) > 20:
                continue

            # Apply substitution
            X_sub = X.clone()
            X_sub[:, col_idx] = tfn(v)

            # Skip degenerate transformed columns (constant / near-constant /
            # pathological range compression).
            v_sub = X_sub[:, col_idx]
            if not torch.isfinite(v_sub).all():
                continue
            if float(v_sub.std()) < 1e-10:
                continue
            q05 = torch.quantile(v_sub, 0.05)
            q95 = torch.quantile(v_sub, 0.95)
            if float((q95 - q05).abs()) < 1e-8:
                continue

            # Grid search over rational supports.  In unit-aware mode these are
            # exact coefficient-solver emissions, never dense mixed-unit bases.
            identity_trials = _support_trials(unit_target, transformed_dims)
            all_fits = _probe_trials(
                X_sub,
                F,
                identity_trials,
                baseline=False,
            )
            finite_identity = [err for _trial, err in all_fits if math.isfinite(err)]
            best_err = min(finite_identity) if finite_identity else float("inf")

            # --- accept / reject identity-output hit ---
            def _identity_passes(err: float) -> bool:
                if dim > 1:
                    return bool(
                        math.isfinite(err)
                        and err < strong_threshold
                        and (
                            not math.isfinite(baseline_best_err)
                            or err < 0.6 * baseline_best_err
                        )
                    )
                return bool(math.isfinite(err) and err < float(threshold))

            identity_passes = [item for item in all_fits if _identity_passes(item[1])]
            stats["numeric_rejected"] += int(len(all_fits) - len(identity_passes))
            if identity_passes:
                # Legacy mode retains one hit per transform. Unit-aware mode
                # keeps every admissible support so the caller can continue
                # after a later build/dedup rejection until its requested
                # emission count is met.
                chosen_identity = sorted(identity_passes, key=_trial_order)
                if not bool(stats["unit_aware"]):
                    chosen_identity = chosen_identity[:1]
                for support_rank, item in enumerate(chosen_identity):
                    hit = _hit_from_fit(
                            col_idx=col_idx,
                            transform=tname,
                            parity=parity,
                            item=item,
                            rational_target_dim=unit_target,
                            rational_input_dims=transformed_dims,
                        )
                    if bool(stats["unit_aware"]):
                        hit["unit_support_rank"] = int(support_rank)
                    results.append(hit)

            # --- outer-transform probing (square, reciprocal) ---
            if outer_transforms:
                _OUTER_FNS = {
                    "square": lambda f: f * f,
                    "reciprocal": lambda f: 1.0 / f,
                }
                finite_sub = torch.isfinite(F) & torch.isfinite(X_sub).all(dim=1)
                for ot_name in outer_transforms:
                    ot_fn = _OUTER_FNS.get(ot_name)
                    if ot_fn is None:
                        continue
                    sign_hint = 1.0
                    sign_cons = 1.0
                    X_ot = X_sub
                    # Domain checks
                    if ot_name == "square":
                        if int(finite_sub.sum().item()) < int(min_points):
                            continue
                        F_use = F[finite_sub]
                        X_ot = X_sub[finite_sub]
                        F_t = F_use * F_use
                        if not torch.isfinite(F_t).all():
                            continue
                        eps = 1e-8
                        m_pos = F_use > eps
                        m_neg = F_use < -eps
                        n_pos = int(m_pos.sum().item())
                        n_neg = int(m_neg.sum().item())
                        n_sig = n_pos + n_neg
                        if n_sig <= 0:
                            continue
                        sign_hint = 1.0 if n_pos >= n_neg else -1.0
                        sign_cons = float(max(n_pos, n_neg) / max(1, n_sig))
                        if sign_cons < float(square_sign_consistency):
                            _log.info(
                                "[NLS] outer=square col=%d tfm=%s SKIP: "
                                "sign_cons=%.3f < %.3f (n_pos=%d, n_neg=%d)",
                                col_idx, tname, sign_cons,
                                float(square_sign_consistency), n_pos, n_neg,
                            )
                            continue
                    elif ot_name == "reciprocal":
                        eps = 1e-8
                        rec_mask = finite_sub & (F.abs() > eps)
                        if int(rec_mask.sum().item()) < int(min_points):
                            continue
                        X_ot = X_sub[rec_mask]
                        F_t = 1.0 / F[rec_mask]
                        if not torch.isfinite(F_t).all():
                            continue
                    else:
                        continue

                    if bool(stats["unit_aware"]):
                        from nestynet_sr.sr_core.units import scale_dim

                        factor = Fraction(2, 1) if ot_name == "square" else Fraction(-1, 1)
                        outer_rational_target = scale_dim(unit_target, factor)
                    else:
                        outer_rational_target = None
                    outer_trials = _support_trials(
                        outer_rational_target,
                        transformed_dims,
                    )
                    ot_all_fits = _probe_trials(
                        X_ot,
                        F_t,
                        outer_trials,
                        baseline=False,
                    )
                    finite_outer = [
                        err for _trial, err in ot_all_fits if math.isfinite(err)
                    ]
                    ot_best_err = min(finite_outer) if finite_outer else float("inf")
                    if finite_outer:
                        ot_best_trial = min(
                            (item for item in ot_all_fits if math.isfinite(item[1])),
                            key=lambda item: item[1],
                        )[0]
                        ot_best_dn = int(ot_best_trial["deg_num"])
                        ot_best_dd = int(ot_best_trial["deg_den"])
                    else:
                        ot_best_dn, ot_best_dd = 0, 0

                    _log.info(
                        "[NLS] outer=%s col=%d tfm=%s: ot_best_err=%.3e "
                        "deg=(%d,%d), identity_best=%.3e, thr=%.3e, "
                        "0.5*identity=%.3e",
                        ot_name, col_idx, tname, ot_best_err,
                        ot_best_dn, ot_best_dd, best_err, threshold,
                        0.5 * best_err if math.isfinite(best_err) else float("inf"),
                    )

                    # Accept if the outer-transform probe is below threshold.
                    # When the identity fit is also good, require the outer-
                    # transform to be no worse (but NOT strictly 2x better —
                    # the identity hit for an irrational function is a Padé
                    # approximation whose error is comparable to the true
                    # rational under noise, so a 0.5x factor rejects valid
                    # square/reciprocal detections).
                    if not math.isfinite(ot_best_err):
                        stats["numeric_rejected"] += int(len(ot_all_fits))
                        _log.info("[NLS] outer=%s SKIP: non-finite", ot_name)
                        continue
                    if ot_best_err >= float(threshold):
                        stats["numeric_rejected"] += int(len(ot_all_fits))
                        _log.info("[NLS] outer=%s SKIP: err %.3e >= thr %.3e",
                                  ot_name, ot_best_err, threshold)
                        continue
                    if math.isfinite(best_err) and ot_best_err >= best_err:
                        stats["numeric_rejected"] += int(len(ot_all_fits))
                        _log.info("[NLS] outer=%s SKIP: err %.3e >= identity %.3e",
                                  ot_name, ot_best_err, best_err)
                        continue

                    outer_passes = [
                        item
                        for item in ot_all_fits
                        if (
                            math.isfinite(item[1])
                            and item[1] < float(threshold)
                            and (not math.isfinite(best_err) or item[1] < best_err)
                        )
                    ]
                    stats["numeric_rejected"] += int(
                        len(ot_all_fits) - len(outer_passes)
                    )
                    chosen_outer = sorted(outer_passes, key=_trial_order)
                    if not bool(stats["unit_aware"]):
                        chosen_outer = chosen_outer[:1]
                    for support_rank, item in enumerate(chosen_outer):
                        hit = _hit_from_fit(
                            col_idx=col_idx,
                            transform=tname,
                            parity=parity,
                            item=item,
                            rational_target_dim=outer_rational_target,
                            rational_input_dims=transformed_dims,
                        )
                        if bool(stats["unit_aware"]):
                            hit["unit_support_rank"] = int(support_rank)
                        hit["outer_transform"] = ot_name
                        if ot_name == "square":
                            hit["sign_hint"] = float(sign_hint)
                            hit["sign_consistency"] = float(sign_cons)
                        results.append(hit)

            if bool(stats["numeric_attempt_budget_exhausted"]):
                break
        if bool(stats["numeric_attempt_budget_exhausted"]):
            break

    return _finish(results)


def _affine_decomposition_screen(
    X: torch.Tensor,
    F: torch.Tensor,
    trig_hints: Optional[Dict[int, "TrigAxisSpec"]] = None,
    min_points: int = 200,
    n_bins: int = 20,
    min_per_bin: int = 30,
    r2_threshold: float = 0.999,
) -> List[Dict]:
    """Screen whether g(f(z, w)) = a(z) + b(z) * h(w) for some transforms g, h.

    For a 2D atom f(z, w), check if an output transform g(f) is affine in
    some function h(w):
        g(f(z, w)) = a(z) + b(z) * h(w)
    If so, the 2D problem reduces to two 1D problems: finding a(z) and b(z).

    Parameters
    ----------
    X : Tensor [N, 2]
        Atom inputs: column 0 is z, column 1 is w.
    F : Tensor [N]
        Atom output values.
    trig_hints : dict, optional
        Mapping ``{col_idx: TrigAxisSpec}`` from Stage A/Stage B trig detection.
        When present for the affine column, the omega from the TrigAxisSpec is
        used for cos/sin/one-minus-cos transforms on w.
    min_points : int
        Minimum number of data points required.
    n_bins : int
        Number of bins to group by z values.
    min_per_bin : int
        Minimum number of points per bin for a valid fit.
    r2_threshold : float
        Minimum median R² across bins to accept a hit.

    Returns
    -------
    List[Dict]
        Hits sorted by descending median R². Each dict has keys:
        ``g_name``, ``h_name``, ``omega``, ``col_w``, ``median_r2``,
        ``a_values``, ``b_values``, ``z_centers``.
    """
    X = X.to(dtype=torch.float64)
    F = F.view(-1).to(dtype=torch.float64)
    N = X.shape[0]

    if X.shape[1] != 2:
        return []

    # Stage B often probes compact training slices (for example Class-SR smoke
    # runs with ~64 points). Keep the screen alive on those cases by adapting
    # the minimum sample/bucket requirements, while still refusing genuinely
    # under-resolved clouds.
    min_points_eff = max(24, min(int(min_points), int(N)))
    if N < min_points_eff:
        return []

    # Guard: skip if either column or F is constant
    if float(X[:, 0].std()) < 1e-10 or float(X[:, 1].std()) < 1e-10 or float(F.std()) < 1e-10:
        return []

    from nestynet_sr.sr_search.feature_grammar import OMEGA_SNAP_CANDS, snap_to_scales

    # Output transforms g and their inverses (independent of column ordering)
    g_transforms = [("identity", lambda f: f)]
    if float(F.abs().min()) > 1e-12:
        g_transforms.append(("reciprocal", lambda f: 1.0 / f))

    results: List[Dict] = []

    # Try both column orderings: bin by z, test affine in w
    for z_col, w_col in [(0, 1), (1, 0)]:
        z = X[:, z_col]
        w = X[:, w_col]

        # Determine omega candidates for trig transforms on w
        omega_candidates = {1.0}
        trig_hint_spec = trig_hints.get(w_col) if trig_hints else None
        hint_basis = ""
        if trig_hint_spec is not None:
            spec = trig_hint_spec
            hint_basis = str(
                getattr(spec, "basis_fn", "") or getattr(spec, "trig_fn", "")
            )
            om = float(getattr(spec, "omega", 1.0))
            if math.isfinite(om) and om > 0:
                snapped = snap_to_scales(om, OMEGA_SNAP_CANDS, rel_tol=0.25, abs_tol=0.25)
                omega_candidates.add(snapped)

        # Variable transforms h for w.  Keep the historical default cos/sin
        # omega=1 probes, but only add 1-cos when a trig hint confirms that
        # this column is genuinely periodic.
        h_transforms: List[Tuple[str, float, bool]] = [("identity", 1.0, False)]
        for om in sorted(omega_candidates):
            h_transforms.append((
                "cos",
                om,
                trig_hint_spec is not None and hint_basis in {"", "cos"},
            ))
            h_transforms.append(("sin", om, trig_hint_spec is not None and hint_basis == "sin"))
            if trig_hint_spec is not None:
                h_transforms.append(("one_minus_cos", om, hint_basis == "one_minus_cos"))

        # Sort by z, create equal-count bins
        z_order = torch.argsort(z)

        for g_name, g_fn in g_transforms:
            # Apply g transform
            try:
                G = g_fn(F)
            except Exception:
                continue
            if not torch.isfinite(G).all():
                continue

            for h_name, h_omega, h_basis_match in h_transforms:
                try:
                    if h_name == "cos":
                        W = torch.cos(h_omega * w)
                    elif h_name == "sin":
                        W = torch.sin(h_omega * w)
                    elif h_name == "one_minus_cos":
                        W = 1.0 - torch.cos(h_omega * w)
                    else:
                        W = w
                except Exception:
                    continue
                if not torch.isfinite(W).all():
                    continue
                if float(W.std()) < 1e-10:
                    continue

                # Bin by z and do per-bin OLS: G = a + b*W
                G_sorted = G[z_order]
                W_sorted = W[z_order]
                z_sorted = z[z_order]

                # Determine bin edges for equal-count bins
                target_bins = min(int(n_bins), max(3, int(round(math.sqrt(N)))))
                min_per_bin_eff = min(int(min_per_bin), max(6, N // max(1, target_bins)))
                actual_bins = min(target_bins, N // min_per_bin_eff)
                if actual_bins < 3:
                    continue

                bin_size = N // actual_bins
                z_bins = []
                for bi in range(actual_bins):
                    lo = bi * bin_size
                    hi = (bi + 1) * bin_size if bi < actual_bins - 1 else N
                    if hi - lo < min_per_bin_eff:
                        z_bins = []
                        break
                    z_bins.append((lo, hi))
                if len(z_bins) < 3:
                    continue

                z_cents_global = [float(z_sorted[lo:hi].mean()) for lo, hi in z_bins]

                # Fast path for genuinely affine leaves: G ≈ α + β*z + γ*W.
                # This is exactly the SR case and is much more stable than
                # estimating a(z), b(z) independently from tiny random bins.
                ones = torch.ones(N, dtype=torch.float64, device=G.device)
                Phi_global = torch.stack([ones, z, W], dim=1)
                try:
                    beta_global = torch.linalg.lstsq(Phi_global, G.unsqueeze(1)).solution.squeeze(1)
                    alpha_fit = float(beta_global[0])
                    z_slope_fit = float(beta_global[1])
                    w_slope_fit = float(beta_global[2])
                    pred_global = alpha_fit + z_slope_fit * z + w_slope_fit * W
                    ss_res_global = float(((G - pred_global) ** 2).sum())
                    ss_tot_global = float(((G - G.mean()) ** 2).sum())
                    global_r2 = 1.0 if ss_tot_global < 1e-20 else 1.0 - ss_res_global / ss_tot_global
                except Exception:
                    global_r2 = float("-inf")
                if global_r2 > r2_threshold:
                    results.append({
                        "g_name": g_name,
                        "h_name": h_name,
                        "omega": h_omega,
                        "col_w": w_col,
                        "median_r2": float(global_r2),
                        "a_values": [alpha_fit + z_slope_fit * zc for zc in z_cents_global],
                        "b_values": [w_slope_fit for _ in z_cents_global],
                        "z_centers": z_cents_global,
                        "global_affine": True,
                        "global_alpha": float(alpha_fit),
                        "global_z_slope": float(z_slope_fit),
                        "global_w_slope": float(w_slope_fit),
                        "basis_match": bool(h_basis_match),
                    })
                    continue

                r2_list = []
                a_vals = []
                b_vals = []
                z_cents = []
                skip = False

                for lo, hi in z_bins:
                    W_bin = W_sorted[lo:hi]
                    G_bin = G_sorted[lo:hi]
                    z_bin = z_sorted[lo:hi]

                    # OLS: G_bin = a + b * W_bin
                    n_b = W_bin.shape[0]
                    ones = torch.ones(n_b, dtype=torch.float64, device=W_bin.device)
                    Phi = torch.stack([ones, W_bin], dim=1)  # [n_b, 2]
                    try:
                        beta = torch.linalg.lstsq(Phi, G_bin.unsqueeze(1)).solution.squeeze(1)
                    except Exception:
                        skip = True
                        break

                    a_fit = float(beta[0])
                    b_fit = float(beta[1])
                    pred = a_fit + b_fit * W_bin
                    ss_res = float(((G_bin - pred) ** 2).sum())
                    ss_tot = float(((G_bin - G_bin.mean()) ** 2).sum())

                    if ss_tot < 1e-20:
                        # G is constant in this bin — trivially affine
                        r2_list.append(1.0)
                    else:
                        r2_list.append(1.0 - ss_res / ss_tot)

                    a_vals.append(a_fit)
                    b_vals.append(b_fit)
                    z_cents.append(float(z_bin.mean()))

                if skip or len(r2_list) < 3:
                    continue

                median_r2 = float(sorted(r2_list)[len(r2_list) // 2])

                if median_r2 > r2_threshold:
                    results.append({
                        "g_name": g_name,
                        "h_name": h_name,
                        "omega": h_omega,
                        "col_w": w_col,
                        "median_r2": median_r2,
                        "a_values": a_vals,
                        "b_values": b_vals,
                        "z_centers": z_cents,
                        "basis_match": bool(h_basis_match),
                    })

    return sorted(
        results,
        key=lambda r: (
            -float(r["median_r2"]),
            -int(bool(r.get("basis_match", False))),
            -int(bool(r.get("global_affine", False))),
        ),
    )


def _fit_sin_linear_coeffs_1d(
    x: torch.Tensor,
    f: torch.Tensor,
    omega0: float,
    min_points: int = 200,
    dtype: torch.dtype = torch.float64,
) -> Optional[Tuple[float, float, float]]:
    """
    Fit f(x) ≈ A * sin(omega0 * x + phi) in 1D, with fixed frequency omega0
    (typically from discover_trig_axes). Returns (A, omega, phi) or None.
    """
    x = x.view(-1).to(dtype=dtype)
    f = f.view(-1).to(dtype=dtype)
    N = x.numel()
    if N < min_points:
        return None

    omega = float(omega0)
    if abs(omega) < 1e-12:
        return None

    t = omega * x
    Phi = torch.stack([torch.sin(t), torch.cos(t)], dim=1)  # [N,2]

    Gram = Phi.T @ Phi
    Gram = Gram + 1e-10 * torch.eye(2, dtype=Gram.dtype, device=Gram.device)
    rhs = Phi.T @ f
    beta = torch.linalg.solve(Gram, rhs)  # [2]

    A_sin = float(beta[0].item())
    A_cos = float(beta[1].item())
    amp = math.hypot(A_sin, A_cos)
    if amp < 1e-8:
        return None

    # A_sin*sin + A_cos*cos = amp * sin( . + phi )
    phi = math.atan2(A_cos, A_sin)

    return amp, omega, phi


def _fit_tanh_linear_coeffs_1d(
    x: torch.Tensor,
    f: torch.Tensor,
    min_points: int = 200,
    dtype: torch.dtype = torch.float64,
    clip: float = 0.999,
) -> Optional[Tuple[float, float, float]]:
    """Fit f(x) ≈ A * tanh(omega * x + b) in 1D.

    Returns (A, omega, b) or None.

    We use a robust linearization on the non-saturated region:
        atanh(f/A) ≈ omega * x + b
    """
    x = x.view(-1).to(dtype=dtype)
    f = f.view(-1).to(dtype=dtype)
    N = x.numel()
    if N < min_points:
        return None

    # Robust amplitude estimate (avoid outliers)
    try:
        A = float(torch.quantile(f.abs(), 0.99).item())
    except Exception:
        A = float(f.abs().max().item())
    if (not math.isfinite(A)) or (A < 1e-8):
        return None

    u = (f / A).clamp(min=-clip, max=clip)

    # Avoid saturated points where atanh blows up.
    # If we end up with too few points, fall back to all (still clipped).
    mask = u.abs() < (clip - 0.02)
    if mask.sum().item() < max(20, min_points // 4):
        mask = torch.ones_like(u, dtype=torch.bool)

    y = torch.atanh(u[mask])
    xv = x[mask]
    if y.numel() < 2:
        return None

    Phi = torch.stack([torch.ones_like(xv), xv], dim=1)  # [n,2]
    beta = torch.linalg.lstsq(Phi, y.unsqueeze(1)).solution.squeeze(1)
    b = float(beta[0].item())
    omega = float(beta[1].item())
    if (not math.isfinite(omega)) or (abs(omega) < 1e-12):
        return None

    return A, omega, b


def _fit_power_coeffs_1d(
    x: torch.Tensor,
    f: torch.Tensor,
    min_points: int = 200,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-8,
) -> Optional[Tuple[float, float]]:
    """
    Fit f(x) ≈ A * x^p on positive x,f via log–log regression.
    Returns (A, p) or None if we don't have enough usable points.
    """
    x = x.view(-1).to(dtype=dtype)
    f = f.view(-1).to(dtype=dtype)

    mask = (x > eps) & (f > eps)
    if mask.sum().item() < min_points:
        return None

    z = torch.log(x[mask])
    y = torch.log(f[mask])
    N = z.numel()
    if N < 2:
        return None

    Phi = torch.stack([torch.ones(N, dtype=dtype, device=z.device), z], dim=1)  # [N,2]
    beta = torch.linalg.lstsq(Phi, y.unsqueeze(1)).solution.squeeze(1)  # [2]
    logA = beta[0]
    p = beta[1]
    A = torch.exp(logA)

    return float(A.item()), float(p.item())


def _fit_exp_poly_coeffs_1d(
    x: torch.Tensor,
    f: torch.Tensor,
    degree: int,
    min_points: int = 200,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-8,
) -> Optional[torch.Tensor]:
    """
    Fit log f(x) ≈ P(x) with 1D polynomial P of given degree.

    Used to initialise ExpPolyLeaf so that f(x) ≈ exp(P(x)).
    Only uses points with f > eps.
    """
    x = x.view(-1, 1)
    f = f.view(-1)

    mask = f > eps
    if mask.sum().item() < min_points:
        return None

    x_pos = x[mask]
    logf = torch.log(f[mask])

    return _fit_poly_coeffs_1d(
        x_pos,
        logf,
        degree=degree,
        min_points=min_points,
        dtype=dtype,
    )


def _fit_exp_ratpoly_coeffs_1d(
    x, f, deg_num, deg_den, min_points=200, dtype=torch.float64, eps=1e-8
):
    f = f.view(-1)
    m = f > eps
    if m.sum().item() < min_points:
        return None
    x_pos = x[m]
    logf = torch.log(f[m])
    coeffs = _fit_rational_coeffs_1d(
        x_pos, logf, deg_num=deg_num, deg_den=deg_den, min_points=min_points, dtype=dtype
    )
    return coeffs


def _fit_planck_tail(
    X: torch.Tensor,
    F: torch.Tensor,
    *,
    min_points: int = 400,
    eps: float = 1e-8,
    tail_fraction: float = 0.5,
    max_abs_p: float = 10.0,
    rel_rms_threshold: float = 0.05,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Fit Planck/Bose-Einstein tail in log-space.

    For large x, the Planck function f(x) = A * x^p / (exp(a*x) - 1)
    behaves as f(x) ≈ A * x^p * exp(-a*x), so:
        log(f) ≈ log(A) + p*log(x) - a*x

    This function performs linear regression on the high-x tail to estimate
    the Planck parameters.

    Parameters
    ----------
    X : Tensor
        1D input values (must be positive for valid fit).
    F : Tensor
        1D target values (must be positive for valid fit).
    min_points : int
        Minimum number of valid data points required.
    eps : float
        Small value for positivity filtering.
    tail_fraction : float
        Fraction of data (high-x region) to use for tail fit.
    max_abs_p : float
        Maximum absolute value of power exponent p.
    rel_rms_threshold : float
        Maximum relative RMS error in log-space fit.

    Returns
    -------
    Optional[Tuple[float, float, float, float]]
        (p_est, a_est, b0, rms_rel) where:
        - p_est: estimated power exponent
        - a_est: estimated exponential coefficient (α in exp(αx))
        - b0: log(A) intercept
        - rms_rel: relative RMS error of the fit
        Returns None if fit fails or doesn't meet quality threshold.
    """
    X = X.view(-1).to(dtype=torch.float64)
    F = F.view(-1).to(dtype=torch.float64)

    # Filter to positive, finite values (required for log-space fit)
    m = (X > eps) & (F > eps) & torch.isfinite(X) & torch.isfinite(F)
    if m.sum().item() < min_points:
        return None
    X = X[m]
    F = F[m]
    N = X.numel()
    if N < min_points:
        return None

    # Select high-x tail
    order = torch.argsort(X)
    k_tail = int(max(0, min(N - 1, int(tail_fraction * N))))
    thr = X[order[k_tail]]
    mt = X >= thr
    if mt.sum().item() < min_points:
        return None

    Xt = X[mt]
    Ft = F[mt]
    logF = torch.log(Ft)

    # Design matrix: [1, log(x), x] for regression log(f) = β0 + β1*log(x) + β2*x
    Phi = torch.stack([torch.ones_like(Xt), torch.log(Xt), Xt], dim=1)

    try:
        beta = torch.linalg.lstsq(Phi, logF.unsqueeze(1)).solution.squeeze(1)
    except RuntimeError:
        return None

    b0 = float(beta[0])       # log(A)
    p_est = float(beta[1])    # power of x
    c_est = float(beta[2])    # -α (coefficient of x)
    a_est = -c_est            # α in exp(α*x)

    # Validate parameters
    if not (math.isfinite(p_est) and math.isfinite(a_est)):
        return None
    if a_est <= 0.0 or abs(p_est) > max_abs_p:
        return None

    # Compute fit quality
    logF_fit = (Phi @ beta).view(-1)
    resid = logF - logF_fit
    rms_abs = float(torch.sqrt(torch.mean(resid * resid)))
    std = float(torch.std(logF, unbiased=False))
    if std < 1e-12:
        return None
    rms_rel = rms_abs / std

    if (not math.isfinite(rms_rel)) or rms_rel > rel_rms_threshold:
        return None

    return p_est, a_est, b0, rms_rel


def _fit_planck_tail_fixed_power(
    X: torch.Tensor,
    F: torch.Tensor,
    *,
    p_fixed: float,
    min_points: int = 400,
    eps: float = 1e-8,
    tail_fraction: float = 0.5,
    rel_rms_threshold: float = 0.05,
) -> Optional[Tuple[float, float, float, float]]:
    """Fit ``A * x**p / (exp(a*x) - 1)`` with fixed structural ``p``.

    Only ``A`` and ``a`` are fitted.  The returned tuple matches
    :func:`_fit_planck_tail`: ``(p_fixed, a_est, logA, rms_rel)``.
    """
    X = X.view(-1).to(dtype=torch.float64)
    F = F.view(-1).to(dtype=torch.float64)

    m = (X > eps) & (F > eps) & torch.isfinite(X) & torch.isfinite(F)
    if m.sum().item() < min_points:
        return None
    X = X[m]
    F = F[m]
    N = X.numel()
    if N < min_points:
        return None

    order = torch.argsort(X)
    k_tail = int(max(0, min(N - 1, int(tail_fraction * N))))
    thr = X[order[k_tail]]
    mt = X >= thr
    if mt.sum().item() < min_points:
        return None

    Xt = X[mt]
    Ft = F[mt]
    logF = torch.log(Ft)
    p = float(p_fixed)
    rhs = logF - p * torch.log(Xt)
    Phi = torch.stack([torch.ones_like(Xt), Xt], dim=1)

    try:
        beta = torch.linalg.lstsq(Phi, rhs.unsqueeze(1)).solution.squeeze(1)
    except RuntimeError:
        return None

    b0 = float(beta[0])
    a_est = -float(beta[1])
    if not (math.isfinite(a_est) and math.isfinite(b0)) or a_est <= 0.0:
        return None

    rhs_fit = (Phi @ beta).view(-1)
    resid = rhs - rhs_fit
    rms_abs = float(torch.sqrt(torch.mean(resid * resid)))
    std = float(torch.std(logF, unbiased=False))
    if std < 1e-12:
        return None
    rms_rel = rms_abs / std
    if (not math.isfinite(rms_rel)) or rms_rel > rel_rms_threshold:
        return None

    return p, a_est, b0, rms_rel


def _fit_planck_tail_discrete_power(
    X: torch.Tensor,
    F: torch.Tensor,
    *,
    powers: Sequence[float] = PLANCK_STRUCTURAL_POWERS,
    min_points: int = 400,
    eps: float = 1e-8,
    tail_fraction: float = 0.5,
    rel_rms_threshold: float = 0.05,
) -> Optional[Tuple[float, float, float, float]]:
    """Scan the normal reduced Planck structural powers and keep the best fit."""
    best: Optional[Tuple[float, float, float, float]] = None
    for p in powers:
        fit = _fit_planck_tail_fixed_power(
            X,
            F,
            p_fixed=float(p),
            min_points=min_points,
            eps=eps,
            tail_fraction=tail_fraction,
            rel_rms_threshold=rel_rms_threshold,
        )
        if fit is None:
            continue
        if best is None or float(fit[3]) < float(best[3]):
            best = fit
    return best


def _gather_teacher_data_1d(
    train_loader,
    teacher: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    *,
    axis: Optional[int] = None,
    input_expr: Optional[Node] = None,
    max_points: int = 5000,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Unified data gathering for 1D fitting (univariate or compound atoms).

    Gathers (x_1d, f(x_1d)) pairs where x_1d is either:
    - A single axis column: x[:, axis:axis+1] (univariate case)
    - The result of input_expr(x) (compound case)

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    teacher : torch.nn.Module
        The NN leaf module that maps input -> f(input).
    device : torch.device
        Device for computation.
    dtype : torch.dtype
        Data type for tensors.
    axis : int, optional
        For univariate atoms: the variable axis (column index).
    input_expr : Node, optional
        For compound atoms: AST representing the compound variable expression.
    max_points : int
        Maximum number of points to gather.

    Returns
    -------
    Optional[Tuple[torch.Tensor, torch.Tensor]]
        (X_1d, F) where X_1d is the 1D input values and F is teacher outputs,
        or None if no data could be gathered.

    Notes
    -----
    Exactly one of `axis` or `input_expr` should be provided.
    """
    from nestynet_sr.sr_core.bridges import eval_input_expr

    xs: List[torch.Tensor] = []
    fs: List[torch.Tensor] = []
    n_collected = 0

    teacher.eval()

    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch
        x = x.to(device=device, dtype=dtype)

        # Unified: always evaluate via input_expr (works for trivial Var(i) too).
        # Fallback to axis-based selection if input_expr is not provided (legacy).
        if input_expr is not None:
            x_1d = eval_input_expr(input_expr, x)  # [B, 1]
        else:
            x_1d = x[:, axis : axis + 1]  # [B, 1]

        with torch.no_grad():
            f = teacher(x_1d)
            if f.dim() == 2:
                f = f[:, 0]
            else:
                f = f.view(-1)

        xs.append(x_1d.detach().cpu())
        fs.append(f.detach().cpu())
        n_collected += x_1d.size(0)
        if n_collected >= max_points:
            break

    if not xs:
        return None

    X = torch.cat(xs, dim=0)[:max_points]
    F = torch.cat(fs, dim=0)[:max_points]
    return X, F


def _initialise_analytic_leaves_from_reuse(
    root: Node,
    model: torch.nn.Module,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int = 5000,
    trig_by_axis: Optional[Dict[int, TrigAxisSpec]] = None,
) -> None:
    """
    For each PolyLeaf / RationalPolyLeaf in the compiled model, use its
    tag to look up the corresponding Stage-A NN leaf in `reuse`, and fit
    an initial analytic approximation to that NN leaf on the training
    inputs.

    This gives LM a very good starting point for x, x^2, 1/x, ...-like structures.
    """
    # Import locally to avoid circular dependency
    from .stageB import _collect_all_atoms

    atoms = _collect_all_atoms(root)
    leaves = list(model.leaf)

    tag_to_core = {}
    for _a, _l in zip(atoms, leaves):
        _t = getattr(_a, "tag", None)
        if _t is None:
            continue
        _c = getattr(_l, "core", getattr(_l, "model", _l))
        tag_to_core[_t] = _c

    def _copy_params_if_compatible(core_new: nn.Module, core_old: nn.Module) -> bool:
        if type(core_new) is not type(core_old):
            return False

        if isinstance(core_new, PolyLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True

        if isinstance(core_new, RPolyLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True

        if isinstance(core_new, PolyLogLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True

        if isinstance(core_new, RPolyLogLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True

        if isinstance(core_new, RationalPolyLeaf):
            if (core_new.coeffs_num.shape != core_old.coeffs_num.shape) or (
                core_new.coeffs_den.shape != core_old.coeffs_den.shape
            ):
                return False
            if hasattr(core_new, "exps_num") and hasattr(core_old, "exps_num"):
                if core_new.exps_num.shape != core_old.exps_num.shape:
                    return False
                if not torch.equal(
                    core_new.exps_num.detach().cpu(), core_old.exps_num.detach().cpu()
                ):
                    return False
            if hasattr(core_new, "exps_den") and hasattr(core_old, "exps_den"):
                if core_new.exps_den.shape != core_old.exps_den.shape:
                    return False
                if not torch.equal(
                    core_new.exps_den.detach().cpu(), core_old.exps_den.detach().cpu()
                ):
                    return False
            with torch.no_grad():
                core_new.coeffs_num.copy_(
                    core_old.coeffs_num.to(
                        device=core_new.coeffs_num.device, dtype=core_new.coeffs_num.dtype
                    )
                )
                core_new.coeffs_den.copy_(
                    core_old.coeffs_den.to(
                        device=core_new.coeffs_den.device, dtype=core_new.coeffs_den.dtype
                    )
                )
            return True

        if isinstance(core_new, RRationalPolyLeaf):
            if (core_new.coeffs_num.shape != core_old.coeffs_num.shape) or (
                core_new.coeffs_den.shape != core_old.coeffs_den.shape
            ):
                return False
            if hasattr(core_new, "exps_num") and hasattr(core_old, "exps_num"):
                if core_new.exps_num.shape != core_old.exps_num.shape:
                    return False
                if not torch.equal(
                    core_new.exps_num.detach().cpu(), core_old.exps_num.detach().cpu()
                ):
                    return False
            if hasattr(core_new, "exps_den") and hasattr(core_old, "exps_den"):
                if core_new.exps_den.shape != core_old.exps_den.shape:
                    return False
                if not torch.equal(
                    core_new.exps_den.detach().cpu(), core_old.exps_den.detach().cpu()
                ):
                    return False
            with torch.no_grad():
                core_new.coeffs_num.copy_(
                    core_old.coeffs_num.to(
                        device=core_new.coeffs_num.device, dtype=core_new.coeffs_num.dtype
                    )
                )
                core_new.coeffs_den.copy_(
                    core_old.coeffs_den.to(
                        device=core_new.coeffs_den.device, dtype=core_new.coeffs_den.dtype
                    )
                )
            return True

        if isinstance(core_new, ExpPolyLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True

        if isinstance(core_new, RExpPolyLeaf):
            if core_new.coeffs.shape != core_old.coeffs.shape:
                return False
            if hasattr(core_new, "exps") and hasattr(core_old, "exps"):
                if core_new.exps.shape != core_old.exps.shape:
                    return False
                if not torch.equal(core_new.exps.detach().cpu(), core_old.exps.detach().cpu()):
                    return False
            with torch.no_grad():
                core_new.coeffs.copy_(
                    core_old.coeffs.to(device=core_new.coeffs.device, dtype=core_new.coeffs.dtype)
                )
            return True

        if isinstance(core_new, ExpRationalPolyLeaf):
            if (core_new.coeffs_num.shape != core_old.coeffs_num.shape) or (
                core_new.coeffs_den.shape != core_old.coeffs_den.shape
            ):
                return False
            if hasattr(core_new, "exps_num") and hasattr(core_old, "exps_num"):
                if core_new.exps_num.shape != core_old.exps_num.shape:
                    return False
                if not torch.equal(
                    core_new.exps_num.detach().cpu(), core_old.exps_num.detach().cpu()
                ):
                    return False
            if hasattr(core_new, "exps_den") and hasattr(core_old, "exps_den"):
                if core_new.exps_den.shape != core_old.exps_den.shape:
                    return False
                if not torch.equal(
                    core_new.exps_den.detach().cpu(), core_old.exps_den.detach().cpu()
                ):
                    return False
            with torch.no_grad():
                core_new.coeffs_num.copy_(
                    core_old.coeffs_num.to(
                        device=core_new.coeffs_num.device, dtype=core_new.coeffs_num.dtype
                    )
                )
                core_new.coeffs_den.copy_(
                    core_old.coeffs_den.to(
                        device=core_new.coeffs_den.device, dtype=core_new.coeffs_den.dtype
                    )
                )
            return True

        if isinstance(core_new, SinLinearLeaf):
            if core_new.weight.shape != core_old.weight.shape:
                return False
            with torch.no_grad():
                core_new.weight.copy_(
                    core_old.weight.to(device=core_new.weight.device, dtype=core_new.weight.dtype)
                )
                core_new.bias.copy_(
                    core_old.bias.to(device=core_new.bias.device, dtype=core_new.bias.dtype)
                )
                core_new.amp.copy_(
                    core_old.amp.to(device=core_new.amp.device, dtype=core_new.amp.dtype)
                )
            return True

        if isinstance(core_new, TanhLinearLeaf):
            if core_new.weight.shape != core_old.weight.shape:
                return False
            with torch.no_grad():
                core_new.weight.copy_(
                    core_old.weight.to(device=core_new.weight.device, dtype=core_new.weight.dtype)
                )
                core_new.bias.copy_(
                    core_old.bias.to(device=core_new.bias.device, dtype=core_new.bias.dtype)
                )
                core_new.amp.copy_(
                    core_old.amp.to(device=core_new.amp.device, dtype=core_new.amp.dtype)
                )
            return True

        if isinstance(core_new, PowerLeaf):
            with torch.no_grad():
                core_new.exponent.copy_(
                    core_old.exponent.to(
                        device=core_new.exponent.device, dtype=core_new.exponent.dtype
                    )
                )
                core_new.amp.copy_(
                    core_old.amp.to(device=core_new.amp.device, dtype=core_new.amp.dtype)
                )
            return True

        if isinstance(core_new, PlanckLeaf):
            with torch.no_grad():
                core_new.log_amp.copy_(
                    core_old.log_amp.to(
                        device=core_new.log_amp.device, dtype=core_new.log_amp.dtype
                    )
                )
                core_new.p.copy_(core_old.p.to(device=core_new.p.device, dtype=core_new.p.dtype))
                core_new.log_a.copy_(
                    core_old.log_a.to(device=core_new.log_a.device, dtype=core_new.log_a.dtype)
                )
                if hasattr(core_new, "b") and hasattr(core_old, "b"):
                    core_new.b.copy_(core_old.b.to(device=core_new.b.device, dtype=core_new.b.dtype))
            return True

        if isinstance(core_new, PlanckFullLeaf):
            with torch.no_grad():
                core_new.log_amp.copy_(
                    core_old.log_amp.to(
                        device=core_new.log_amp.device, dtype=core_new.log_amp.dtype
                    )
                )
                core_new.p.copy_(core_old.p.to(device=core_new.p.device, dtype=core_new.p.dtype))
                core_new.log_a.copy_(
                    core_old.log_a.to(device=core_new.log_a.device, dtype=core_new.log_a.dtype)
                )
                if hasattr(core_old, "b"):
                    core_new.b.copy_(core_old.b.to(device=core_new.b.device, dtype=core_new.b.dtype))
                else:
                    core_new.b.zero_()
            return True

        if isinstance(core_new, Expm1Leaf):
            with torch.no_grad():
                core_new.log_amp.copy_(
                    core_old.log_amp.to(
                        device=core_new.log_amp.device, dtype=core_new.log_amp.dtype
                    )
                )
                core_new.log_a.copy_(
                    core_old.log_a.to(device=core_new.log_a.device, dtype=core_new.log_a.dtype)
                )
                if hasattr(core_old, "b"):
                    core_new.b.copy_(core_old.b.to(device=core_new.b.device, dtype=core_new.b.dtype))
                else:
                    core_new.b.zero_()
            return True

        return False

    for atom, leaf_mod in zip(atoms, leaves):
        core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))

        is_poly = isinstance(core, PolyLeaf)
        is_rpoly = isinstance(core, RPolyLeaf)
        is_rat = isinstance(core, RationalPolyLeaf)
        is_rrat = isinstance(core, RRationalPolyLeaf)
        is_rpolylog = isinstance(core, RPolyLogLeaf)
        is_sin = isinstance(core, SinLinearLeaf)
        is_tanh = isinstance(core, TanhLinearLeaf)
        is_power = isinstance(core, PowerLeaf)
        is_exp = isinstance(core, (ExpPolyLeaf, RExpPolyLeaf))
        is_exp_rat = isinstance(core, ExpRationalPolyLeaf)

        if not (is_poly or is_rpoly or is_rat or is_rrat or is_rpolylog or is_sin or is_tanh or is_power or is_exp or is_exp_rat):
            continue

        tag = atom.tag
        if tag is None or tag not in reuse:
            continue

        teacher = reuse[tag]
        teacher_core = getattr(teacher, "core", getattr(teacher, "model", teacher))

        # Preserve already-solved analytic leaves (including multi-D) across candidates.
        # Special case: exp_poly -> rexp_poly conversion (constant term is shifted into a scale leaf
        # during model build via the _mul_scale_tag mechanism). Avoid re-fitting here to prevent
        # double-applying that shift.
        if isinstance(core, RExpPolyLeaf) and isinstance(teacher_core, ExpPolyLeaf) and (atom.kwargs or {}).get("_mul_scale_tag") is not None:
            continue
        # Same skip for poly -> rpoly, polylog -> rpolylog, ratpoly -> rratpoly:
        # bridges.py warm-start already normalised coefficients and pushed lead to scale.
        if isinstance(core, RPolyLeaf) and isinstance(teacher_core, PolyLeaf) and (atom.kwargs or {}).get("_mul_scale_tag") is not None:
            continue
        if isinstance(core, RPolyLogLeaf) and isinstance(teacher_core, PolyLogLeaf) and (atom.kwargs or {}).get("_mul_scale_tag") is not None:
            continue
        if isinstance(core, RRationalPolyLeaf) and isinstance(teacher_core, RationalPolyLeaf) and (atom.kwargs or {}).get("_mul_scale_tag") is not None:
            continue

        if _copy_params_if_compatible(core, teacher_core):
            continue

        # Otherwise only do data-fit initialisation for univariate leaves
        # (including compound atoms with effective_arity=1).
        if effective_arity(atom) != 1:
            continue

        # Get the first input expression (always defined for all atoms).
        from nestynet_sr.sr_core.bridges import compound_input_expr
        input_expr = compound_input_expr(atom)

        print(
            f"[Stage B] Initialising {core.__class__.__name__} for vars {atom.var_idxs}, tag={atom.tag}"
        )

        # Gather teacher data (unified: always via input_expr)
        data = _gather_teacher_data_1d(
            train_loader,
            teacher,
            device,
            dtype,
            input_expr=input_expr,
            max_points=max_points,
        )
        if data is None:
            continue
        X, F = data

        if is_poly:
            coeffs = _fit_poly_coeffs_1d(
                X,
                F,
                degree=core.degree,
                dtype=core.coeffs.dtype,
            )
            if coeffs is None or coeffs.numel() != core.coeffs.numel():
                continue
            with torch.no_grad():
                core.coeffs.copy_(coeffs.to(device=core.coeffs.device, dtype=core.coeffs.dtype))

        elif is_rpoly:
            # Fit using the RPolyLeaf's own monomial basis (exps_full) so that
            # coeffs_full[lead_pos] is always the correct leading coefficient,
            # regardless of min_total.  Then normalise free coeffs by the
            # leading coeff (which RPolyLeaf fixes to 1.0) and push the
            # extracted leading coeff into the associated scale atom.
            x_fit = X.view(-1, 1).to(dtype=core.coeffs.dtype)
            f_fit = F.view(-1).to(dtype=core.coeffs.dtype)
            if x_fit.shape[0] < 200:
                continue
            Phi = _eval_monomials(x_fit, core.exps_full.to(device=x_fit.device))
            M = Phi.shape[1]
            G = Phi.T @ Phi + 1e-10 * torch.eye(M, dtype=Phi.dtype, device=Phi.device)
            coeffs_full = torch.linalg.solve(G, Phi.T @ f_fit)

            lead_val = coeffs_full[core.lead_pos]
            if lead_val.abs() < 1e-30:
                continue          # degenerate fit — skip
            if core.free_pos.numel() > 0:
                idx = core.free_pos.to(device=coeffs_full.device)
                free_coeffs = coeffs_full[idx] / lead_val
                if free_coeffs.numel() == core.coeffs.numel():
                    with torch.no_grad():
                        core.coeffs.copy_(free_coeffs.to(
                            device=core.coeffs.device, dtype=core.coeffs.dtype))
            # Push the leading coefficient into the associated scale atom
            scale_tag = (atom.kwargs or {}).get("_mul_scale_tag")
            scale_core = tag_to_core.get(scale_tag) if scale_tag is not None else None
            if scale_core is not None and hasattr(scale_core, "value"):
                fac = float(lead_val.item())
                if fac == fac and abs(fac) < 1e30:      # finite guard
                    with torch.no_grad():
                        scale_core.value.mul_(torch.as_tensor(
                            fac,
                            dtype=scale_core.value.dtype,
                            device=scale_core.value.device))

        elif is_rat:
            coeffs = _fit_rational_coeffs_1d(
                X,
                F,
                deg_num=core.deg_num,
                deg_den=core.deg_den,
                dtype=core.coeffs_num.dtype,
            )
            if coeffs is None:
                continue
            a, b = coeffs
            if a.numel() != core.coeffs_num.numel() or b.numel() != core.coeffs_den.numel():
                continue
            with torch.no_grad():
                core.coeffs_num.copy_(
                    a.to(device=core.coeffs_num.device, dtype=core.coeffs_num.dtype)
                )
                core.coeffs_den.copy_(
                    b.to(device=core.coeffs_den.device, dtype=core.coeffs_den.dtype)
                )

        elif is_rrat:
            # Fit full rational, then normalise free numerator coefficients by
            # the leading numerator coeff (which RRationalPolyLeaf fixes to
            # 1.0) and push the extracted leading coeff into the scale atom.
            coeffs = _fit_rational_coeffs_nd(
                X.view(-1, 1).to(dtype=torch.float64),
                F,
                exps_num=core.exps_num_full.detach().to(dtype=torch.int64),
                exps_den=core.exps_den.detach().to(dtype=torch.int64),
                dtype=core.coeffs_num.dtype,
            )
            if coeffs is None:
                continue
            a_full, b = coeffs
            lead_val = a_full[core.lead_pos_num] if a_full.numel() > core.lead_pos_num else None
            if lead_val is None or lead_val.abs() < 1e-30:
                continue
            if core.free_pos_num.numel() > 0:
                idx = core.free_pos_num.to(device=a_full.device)
                free_a = a_full[idx] / lead_val
                if free_a.numel() == core.coeffs_num.numel():
                    with torch.no_grad():
                        core.coeffs_num.copy_(
                            free_a.to(device=core.coeffs_num.device, dtype=core.coeffs_num.dtype)
                        )
            if b.numel() == core.coeffs_den.numel():
                with torch.no_grad():
                    core.coeffs_den.copy_(
                        b.to(device=core.coeffs_den.device, dtype=core.coeffs_den.dtype)
                    )
            # Push leading numerator coeff into scale atom
            scale_tag = (atom.kwargs or {}).get("_mul_scale_tag")
            scale_core = tag_to_core.get(scale_tag) if scale_tag is not None else None
            if scale_core is not None and hasattr(scale_core, "value"):
                fac = float(lead_val.item())
                if fac == fac and abs(fac) < 1e30:
                    with torch.no_grad():
                        scale_core.value.mul_(torch.as_tensor(
                            fac,
                            dtype=scale_core.value.dtype,
                            device=scale_core.value.device))

        elif is_rpolylog:
            # Fit polynomial in log-space using the RPolyLogLeaf's own monomial
            # basis.  Normalise free coeffs by the leading coeff and push the
            # extracted leading coeff into the associated scale atom.
            X_log = X.clamp_min(core.eps).log()
            x_fit = X_log.view(-1, 1).to(dtype=core.coeffs.dtype)
            f_fit = F.view(-1).to(dtype=core.coeffs.dtype)
            if x_fit.shape[0] < 200:
                continue
            Phi = _eval_monomials(x_fit, core.exps_full.to(device=x_fit.device))
            M = Phi.shape[1]
            G = Phi.T @ Phi + 1e-10 * torch.eye(M, dtype=Phi.dtype, device=Phi.device)
            coeffs_full = torch.linalg.solve(G, Phi.T @ f_fit)

            lead_val = coeffs_full[core.lead_pos]
            if lead_val.abs() < 1e-30:
                continue
            if core.free_pos.numel() > 0:
                idx = core.free_pos.to(device=coeffs_full.device)
                free_coeffs = coeffs_full[idx] / lead_val
                if free_coeffs.numel() == core.coeffs.numel():
                    with torch.no_grad():
                        core.coeffs.copy_(free_coeffs.to(
                            device=core.coeffs.device, dtype=core.coeffs.dtype))
            # Push the leading coefficient into the associated scale atom
            scale_tag = (atom.kwargs or {}).get("_mul_scale_tag")
            scale_core = tag_to_core.get(scale_tag) if scale_tag is not None else None
            if scale_core is not None and hasattr(scale_core, "value"):
                fac = float(lead_val.item())
                if fac == fac and abs(fac) < 1e30:
                    with torch.no_grad():
                        scale_core.value.mul_(torch.as_tensor(
                            fac,
                            dtype=scale_core.value.dtype,
                            device=scale_core.value.device))

        elif is_sin:
            if trig_by_axis is None:
                continue
            axis = int(atom.var_idxs[0])
            spec = trig_by_axis.get(axis)
            if spec is None:
                continue

            coeffs = _fit_sin_linear_coeffs_1d(
                X,
                F,
                omega0=float(spec.omega),
                dtype=core.weight.dtype,
            )
            if coeffs is None:
                continue

            amp, omega, phi = coeffs
            with torch.no_grad():
                core.weight.zero_()
                core.weight[0] = torch.as_tensor(
                    omega, dtype=core.weight.dtype, device=core.weight.device
                )
                core.bias.copy_(
                    torch.as_tensor(phi, dtype=core.bias.dtype, device=core.bias.device)
                )
                core.amp.copy_(torch.as_tensor(amp, dtype=core.amp.dtype, device=core.amp.device))

        elif is_tanh:
            coeffs = _fit_tanh_linear_coeffs_1d(
                X,
                F,
                dtype=core.weight.dtype,
            )
            if coeffs is None:
                continue
            amp, omega, b = coeffs
            with torch.no_grad():
                core.weight.zero_()
                core.weight[0] = torch.as_tensor(
                    omega, dtype=core.weight.dtype, device=core.weight.device
                )
                core.bias.copy_(
                    torch.as_tensor(b, dtype=core.bias.dtype, device=core.bias.device)
                )
                core.amp.copy_(torch.as_tensor(amp, dtype=core.amp.dtype, device=core.amp.device))
        elif is_power:
            coeffs = _fit_power_coeffs_1d(
                X,
                F,
                dtype=core.exponent.dtype,
            )
            if coeffs is None:
                continue
            A, p = coeffs
            with torch.no_grad():
                core.amp.copy_(torch.as_tensor(A, dtype=core.amp.dtype, device=core.amp.device))
                core.exponent.copy_(
                    torch.as_tensor(p, dtype=core.exponent.dtype, device=core.exponent.device)
                )
        elif is_exp:
            coeffs_full = _fit_exp_poly_coeffs_1d(
                X,
                F,
                degree=getattr(core, "degree", 2),
                dtype=core.coeffs.dtype,
            )
            if coeffs_full is None:
                continue

            # exp_poly: direct copy (includes constant term in exponent)
            if isinstance(core, ExpPolyLeaf):
                if coeffs_full.numel() != core.coeffs.numel():
                    continue
                with torch.no_grad():
                    core.coeffs.copy_(
                        coeffs_full.to(device=core.coeffs.device, dtype=core.coeffs.dtype)
                    )
                continue

            # rexp_poly: pin exponent constant term and shift it into the associated scale leaf
            if isinstance(core, RExpPolyLeaf):
                if not hasattr(core, "exps_full") or not hasattr(core, "free_pos"):
                    continue
                if coeffs_full.numel() != core.exps_full.shape[0]:
                    continue

                const_pos = int(getattr(core, "const_pos", 0))
                free_pos = core.free_pos.detach().cpu().to(torch.int64)
                c0 = coeffs_full[const_pos]
                coeffs_red = coeffs_full[free_pos]

                with torch.no_grad():
                    if core.coeffs.numel() != coeffs_red.numel():
                        continue
                    core.coeffs.copy_(
                        coeffs_red.to(device=core.coeffs.device, dtype=core.coeffs.dtype)
                    )

                scale_tag = (atom.kwargs or {}).get("_mul_scale_tag")
                scale_core = tag_to_core.get(scale_tag) if scale_tag is not None else None
                if scale_core is not None and hasattr(scale_core, "value"):
                    c0_f = float(c0.item())
                    clamp = float(getattr(core, "clamp", 60.0))
                    c0_clip = max(min(c0_f, clamp), -clamp)
                    fac = float(torch.exp(torch.tensor(c0_clip)).item())
                    if fac == fac and fac != float("inf"):
                        with torch.no_grad():
                            scale_core.value.mul_(
                                torch.as_tensor(
                                    fac,
                                    dtype=scale_core.value.dtype,
                                    device=scale_core.value.device,
                                )
                            )
                continue
        elif is_exp_rat:
            coeffs = _fit_exp_ratpoly_coeffs_1d(
                X,
                F,
                deg_num=getattr(core, "deg_num", 2),
                deg_den=getattr(core, "deg_den", 2),
                dtype=core.coeffs_num.dtype,
            )
            if coeffs is None:
                continue
            a, b = coeffs
            if a.numel() != core.coeffs_num.numel() or b.numel() != core.coeffs_den.numel():
                continue
            with torch.no_grad():
                core.coeffs_num.copy_(
                    a.to(device=core.coeffs_num.device, dtype=core.coeffs_num.dtype)
                )
                core.coeffs_den.copy_(
                    b.to(device=core.coeffs_den.device, dtype=core.coeffs_den.dtype)
                )
        else:
            continue
