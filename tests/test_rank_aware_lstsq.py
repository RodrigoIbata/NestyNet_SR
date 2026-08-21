# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Tests for the rank-aware linear solve (numerics.rank_aware_lstsq).

A rank-deficient design (collinear feature columns) must NOT raise a
singular-matrix error: the solver falls back to the minimum-norm
least-squares solution and flags ``rank_deficient``.  Full-rank solves are
unchanged (bit-identical to the legacy normal-equations path).
"""

from __future__ import annotations

import torch

from nestynet_sr.sr_core.numerics import rank_aware_lstsq, ridge_lstsq

torch.manual_seed(0)


def _ls_residual(Phi, c, y):
    return float((Phi @ c - y).square().mean().sqrt())


def test_full_rank_matches_normal_equations():
    Phi = torch.randn(200, 4, dtype=torch.float64)
    y = torch.randn(200, dtype=torch.float64)

    c, info = rank_aware_lstsq(Phi, y, ridge=0.0)
    expected = torch.linalg.solve(Phi.T @ Phi, Phi.T @ y)

    assert info["rank_deficient"] is False
    assert info["rank"] == 4
    assert torch.allclose(c, expected, atol=1e-10)


def test_rank_deficient_does_not_crash_and_flags():
    # Column 1 is exactly 2x column 0 -> rank deficient (rank 2 of 3 columns).
    base = torch.randn(150, 1, dtype=torch.float64)
    w = torch.randn(150, 1, dtype=torch.float64)
    Phi = torch.cat([base, 2.0 * base, w], dim=1)
    # A target that lies in the column space (so a perfect fit exists).
    c_true = torch.tensor([1.0, 0.0, -0.5], dtype=torch.float64)
    y = Phi @ c_true

    c, info = rank_aware_lstsq(Phi, y, ridge=0.0)

    assert info["rank_deficient"] is True
    assert info["rank"] == 2
    assert torch.isfinite(c).all()
    # min-norm solution must still fit the data (residual ~ 0).
    assert _ls_residual(Phi, c, y) < 1e-8


def test_rank_aware_handles_exact_alias():
    # Exact alias (col1 = -col0): rank 1 of 2.  rank_aware_lstsq must not raise
    # and must flag the deficiency; plain ridge_lstsq (normal equations) raises
    # on this singular design -- that division of responsibility is intentional.
    base = torch.randn(80, 1, dtype=torch.float64)
    Phi = torch.cat([base, -base], dim=1)
    y = base.squeeze(1) * 3.0

    c, info = rank_aware_lstsq(Phi, y, ridge=0.0)
    assert info["rank_deficient"] is True
    assert info["rank"] == 1
    assert torch.isfinite(c).all()
    assert _ls_residual(Phi, c, y) < 1e-8

    raised = False
    try:
        ridge_lstsq(Phi, y, ridge=0.0)
    except torch.linalg.LinAlgError:
        raised = True
    assert raised, "plain ridge_lstsq is expected to raise on an exactly-singular design"


def test_ridge_positive_unchanged():
    Phi = torch.randn(60, 3, dtype=torch.float64)
    y = torch.randn(60, dtype=torch.float64)
    c, info = rank_aware_lstsq(Phi, y, ridge=1e-6)
    K = 3
    expected = torch.linalg.solve(
        Phi.T @ Phi + 1e-6 * torch.eye(K, dtype=torch.float64), Phi.T @ y
    )
    assert info["rank_deficient"] is False
    assert torch.allclose(c, expected, atol=1e-12)


if __name__ == "__main__":
    test_full_rank_matches_normal_equations()
    test_rank_deficient_does_not_crash_and_flags()
    test_rank_aware_handles_exact_alias()
    test_ridge_positive_unchanged()
    print("OK: rank_aware_lstsq tests passed")
