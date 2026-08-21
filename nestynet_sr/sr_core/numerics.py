# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Lightweight linear-algebra helpers shared across SR and DE modules.

This module intentionally has minimal imports (only torch) so that it can be
used by ``sr_search`` without pulling in the heavy ``sr_de`` package.
"""

from __future__ import annotations

from typing import Tuple

import torch


def rank_aware_lstsq(
    Phi: torch.Tensor, y: torch.Tensor, ridge: float
) -> Tuple[torch.Tensor, dict]:
    """Solve argmin ||Phi c - y||^2 + ridge||c||^2 without crashing on a singular design.

    For ``ridge > 0`` the (SPD) normal equations are solved directly, as before.
    For ``ridge == 0`` the normal-equations solve is attempted first (fast,
    bit-identical to the legacy path on full-rank designs); if the design is
    rank-deficient the solve is singular, and we fall back to the *minimum-norm*
    least-squares solution via SVD (``torch.linalg.lstsq`` driver ``gelsd``).

    Note: the min-norm fallback prevents a crash and returns a well-defined
    representative, but a rank-deficient design is genuinely non-identifiable —
    among collinear columns it picks a split by convention, not from data.  The
    returned ``info`` flags this so callers can report it rather than treat the
    support as uniquely determined.

    Returns ``(coeffs, info)`` with ``info = {rank_deficient, rank, n_cols}``.
    """
    N, K = Phi.shape
    if K == 0:
        return Phi.new_zeros(0), {"rank_deficient": False, "rank": 0, "n_cols": 0}

    b = Phi.T @ y
    A = Phi.T @ Phi
    if ridge > 0:
        A = A + ridge * torch.eye(K, device=Phi.device, dtype=Phi.dtype)
        return torch.linalg.solve(A, b), {"rank_deficient": False, "rank": K, "n_cols": K}

    # ridge == 0.  Solve directly on Phi by SVD (driver "gelsd"): this both
    # returns the minimum-norm least-squares solution for a rank-deficient
    # design (never crashing) and reports the numerical rank from Phi's singular
    # values.  Rank detection must use Phi, not the normal equations Phi^T Phi,
    # whose squared condition number hides near-collinearity.  For a full-rank
    # design this matches the legacy normal-equations solve to ~1e-12.
    res = torch.linalg.lstsq(Phi, y, driver="gelsd")
    rank = None
    if res.rank is not None and res.rank.numel() > 0:
        rank = int(res.rank.reshape(-1)[0].item())
    return res.solution, {
        "rank_deficient": (rank is not None and rank < int(K)),
        "rank": rank,
        "n_cols": int(K),
    }


def ridge_lstsq(Phi: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    """Solve argmin ||Phi c - y||^2 + ridge||c||^2.

    Uses the (fast) normal equations.  This raises on an exactly-singular design;
    callers that may face a rank-deficient support should use
    :func:`rank_aware_lstsq` instead, which never crashes and reports the rank.
    """
    N, K = Phi.shape
    A = Phi.T @ Phi
    if ridge > 0:
        A = A + ridge * torch.eye(K, device=Phi.device, dtype=Phi.dtype)
    b = Phi.T @ y
    return torch.linalg.solve(A, b)


def stlsq(
    Phi: torch.Tensor, y: torch.Tensor, *, ridge: float, lam: float, max_iter: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential Thresholded Least Squares (STLSQ).

    Returns (coeffs, keep_mask).
    """
    N, K = Phi.shape
    if K == 0:
        return Phi.new_zeros(0), Phi.new_zeros(0, dtype=torch.bool)

    # Column scaling for numerical stability.
    col_scale = Phi.square().mean(0).sqrt().clamp_min(1e-12)
    Phi_n = Phi / col_scale

    keep = torch.ones(K, dtype=torch.bool, device=Phi.device)
    c = Phi.new_zeros(K)

    for _ in range(max_iter):
        if int(keep.sum()) == 0:
            break
        c_keep = ridge_lstsq(Phi_n[:, keep], y, ridge)
        c_new = c.clone()
        c_new[keep] = c_keep

        # Threshold in the *original* coefficient scale.
        c_orig = c_new / col_scale
        keep_new = c_orig.abs() >= lam
        if torch.equal(keep_new, keep):
            c = c_new
            keep = keep_new
            break
        c = c_new
        keep = keep_new

    c_orig = c / col_scale
    return c_orig, keep
