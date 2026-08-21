# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""End-to-end de-Padeification test for the NLS-cos pathway.

Target function:
    f(x0, x1) = sqrt(1 - x0**2) / (1 + x0*x1)

f itself is irrational, but f**2 = (1 - x0**2) / (1 + x0*x1)**2 is a
rational function.  The numerator ``1 - x0**2`` has total degree 2;
the denominator ``(1 + x0*x1)**2 = 1 + 2*x0*x1 + x0**2*x1**2`` has
total degree 4 (the cross-term x0^2*x1^2).  We therefore probe with
deg_num=2, deg_den=4, giving a dense basis with many zero terms --
exactly the scenario where de-Padeification should shine.

In the full SR pipeline this is the kind of candidate the
nonlinear-substitution rule (nls_cos) produces after an
outer-transform peel (square).

The test exercises:
  1. _rational_probe_nd  -- cheap screening probe (builds monomial basis,
     fits via null-space, de-Padeifies, scores on held-out data).
  2. _fit_rational_coeffs_nd -- dense rational fit followed by
     STLSQ sparsification.
  3. Direct call to stlsq_sparsify_rational_coeffs with hand-built
     design matrices to verify coefficient pruning.

All three paths print the de-Padeification log at DEBUG level so you can
inspect which terms survive.
"""

import logging
import math

import torch

from nestynet_sr.sr_core.atoms import RationalPolyLeaf, _enumerate_exponents, _eval_monomials
from nestynet_sr.sr_search.fitting_utils import (
    _fit_rational_coeffs_nd,
    _rational_probe_nd,
)
from nestynet_sr.sr_search.rational_sparsify import (
    DEFAULT_RAT_STLSQ_CFG,
    RationalSparsifyConfig,
    _coeffs_summary,
    _log_sparsify_result,
    stlsq_sparsify_rational_coeffs,
)


# ---------------------------------------------------------------------------
# Enable DEBUG logging so the de-Padeification output is visible.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s  %(message)s",
)


def _make_data(N: int = 3000, seed: int = 42):
    """Generate clean data for f(x0, x1) = sqrt(1 - x0**2) / (1 + x0*x1).

    Returns X (N, 2), f (N,), f_sq (N,) where f_sq = f**2 is rational.
    """
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.rand(N, generator=gen, dtype=torch.float64) * 1.8 - 0.9  # |x0| < 0.9
    x1 = torch.rand(N, generator=gen, dtype=torch.float64) * 1.6 - 0.8

    denom = 1.0 + x0 * x1
    mask = denom.abs() > 0.15
    x0, x1, denom = x0[mask], x1[mask], denom[mask]

    num = torch.sqrt(1.0 - x0 ** 2)
    f = num / denom
    f_sq = (1.0 - x0 ** 2) / denom ** 2

    X = torch.stack([x0, x1], dim=1)
    return X, f, f_sq


# ---------------------------------------------------------------------------
# Test 1: _rational_probe_nd on f**2
# ---------------------------------------------------------------------------
def test_rational_probe_nd():
    """_rational_probe_nd should get a low validation error on f**2.

    Note: the probe does NOT apply de-Padeification (it's a cheap screening
    tool).  Coefficients will be dense with near-zero noise terms.
    De-Padeification happens later in _fit_rational_coeffs_nd (test 2).
    """
    X, _, f_sq = _make_data()
    print("\n=== Test 1: _rational_probe_nd on f**2 (no sparsification) ===")

    # deg_den=4 because (1+x0*x1)^2 has cross-term x0^2*x1^2 (total deg 4).
    err, a, b = _rational_probe_nd(
        X, f_sq,
        deg_num=2, deg_den=4,
        return_coeffs=True,
        min_points=200,
    )
    print(f"Validation error: {err:.6e}")
    assert math.isfinite(err) and err < 0.01, (
        f"Rational probe error too high: {err:.4e}"
    )

    if a is not None:
        print(_coeffs_summary(a, "num"))
    if b is not None:
        print(_coeffs_summary(b, "den"))

    print("PASSED: _rational_probe_nd achieves low error on f**2")


# ---------------------------------------------------------------------------
# Test 2: _fit_rational_coeffs_nd on f**2
# ---------------------------------------------------------------------------
def test_fit_rational_coeffs_nd():
    """_fit_rational_coeffs_nd should return sparsified coefficients."""
    X, _, f_sq = _make_data()
    print("\n=== Test 2: _fit_rational_coeffs_nd on f**2 ===")

    dim = 2
    deg_num, deg_den = 2, 4
    exps_num = torch.tensor(
        _enumerate_exponents(dim, deg_num), dtype=torch.int64,
    )
    exps_den = torch.tensor(
        _enumerate_exponents(dim, deg_den), dtype=torch.int64,
    )

    result = _fit_rational_coeffs_nd(X, f_sq, exps_num, exps_den)
    assert result is not None, "Fit returned None"

    a, b = result
    print(_coeffs_summary(a, "num"))
    print(_coeffs_summary(b, "den"))

    # True numerator (deg 2):  1 - x0**2  => nonzero on {1, x0^2}  (2 of 6 terms)
    # True denominator (deg 4): (1+x0*x1)^2 = 1 + 2*x0*x1 + x0^2*x1^2
    #   => nonzero on {1, x0*x1, x0^2*x1^2}  (3 of 15 terms)
    # After sparsification the many unused monomials should be pruned.
    nnz_num = int((a.abs() > 1e-6).sum().item())
    nnz_den = int((b.abs() > 1e-6).sum().item())
    print(f"Active numerator terms: {nnz_num}/{a.numel()}")
    print(f"Active denominator terms: {nnz_den}/{b.numel()}")

    # Sanity: the sparsifier should drop at least some terms
    total_terms = a.numel() + b.numel()
    active_terms = nnz_num + nnz_den
    print(f"Total terms: {total_terms}, active after sparsify: {active_terms}")
    assert active_terms < total_terms, (
        "Expected sparsification to prune at least one term"
    )

    print("PASSED: _fit_rational_coeffs_nd returns sparsified coefficients")


def test_fit_rational_coeffs_nd_support_builds_sparse_leaf():
    """return_support=True should expose active monomials for sparse leaf construction."""
    X, _, f_sq = _make_data()
    dim = 2
    deg_num, deg_den = 2, 4
    exps_num = torch.tensor(_enumerate_exponents(dim, deg_num), dtype=torch.int64)
    exps_den = torch.tensor(_enumerate_exponents(dim, deg_den), dtype=torch.int64)

    result = _fit_rational_coeffs_nd(
        X,
        f_sq,
        exps_num,
        exps_den,
        return_support=True,
    )
    assert result is not None
    a_sparse, b_sparse, exps_num_sparse, exps_den_sparse = result

    assert int(a_sparse.numel()) == int(exps_num_sparse.shape[0])
    assert int(b_sparse.numel()) == int(exps_den_sparse.shape[0])
    assert int(exps_num_sparse.shape[1]) == dim
    assert int(exps_den_sparse.shape[1]) == dim

    sparse_terms = int(a_sparse.numel() + b_sparse.numel())
    dense_terms = int(exps_num.shape[0] + exps_den.shape[0])
    assert sparse_terms < dense_terms

    leaf = RationalPolyLeaf(
        indices=(0, 1),
        deg_num=deg_num,
        deg_den=deg_den,
        exps_num_override=exps_num_sparse.tolist(),
        exps_den_override=exps_den_sparse.tolist(),
        dtype=torch.float64,
    )
    with torch.no_grad():
        leaf.coeffs_num.copy_(a_sparse.to(dtype=leaf.coeffs_num.dtype, device=leaf.coeffs_num.device))
        leaf.coeffs_den.copy_(b_sparse.to(dtype=leaf.coeffs_den.dtype, device=leaf.coeffs_den.device))

    with torch.no_grad():
        pred = leaf(X).view(-1)
        mse = float((pred - f_sq).square().mean().item())
    assert mse < 1e-3


def test_fit_rational_coeffs_nd_support_indices_builds_sparse_leaf():
    """return_support_indices=True should allow compact index-based ratpoly kwargs."""
    X, _, f_sq = _make_data()
    dim = 2
    deg_num, deg_den = 2, 4
    exps_num = torch.tensor(_enumerate_exponents(dim, deg_num), dtype=torch.int64)
    exps_den = torch.tensor(_enumerate_exponents(dim, deg_den), dtype=torch.int64)

    result = _fit_rational_coeffs_nd(
        X,
        f_sq,
        exps_num,
        exps_den,
        return_support_indices=True,
    )
    assert result is not None
    a_sparse, b_sparse, idx_num, idx_den = result

    assert int(a_sparse.numel()) == int(idx_num.numel())
    assert int(b_sparse.numel()) == int(idx_den.numel())
    assert int(a_sparse.numel() + b_sparse.numel()) < int(exps_num.shape[0] + exps_den.shape[0])

    leaf = RationalPolyLeaf(
        indices=(0, 1),
        deg_num=deg_num,
        deg_den=deg_den,
        support_num_override=[int(i) for i in idx_num.tolist()],
        support_den_override=[int(i) for i in idx_den.tolist()],
        dtype=torch.float64,
    )
    with torch.no_grad():
        leaf.coeffs_num.copy_(a_sparse.to(dtype=leaf.coeffs_num.dtype, device=leaf.coeffs_num.device))
        leaf.coeffs_den.copy_(b_sparse.to(dtype=leaf.coeffs_den.dtype, device=leaf.coeffs_den.device))
        pred = leaf(X).view(-1)
        mse = float((pred - f_sq).square().mean().item())
    assert mse < 1e-3


# ---------------------------------------------------------------------------
# Test 3: Direct stlsq_sparsify_rational_coeffs
# ---------------------------------------------------------------------------
def test_direct_sparsify():
    """Call stlsq_sparsify_rational_coeffs directly with known-good coefficients."""
    X, _, f_sq = _make_data()
    print("\n=== Test 3: direct stlsq_sparsify_rational_coeffs ===")

    dim = 2
    deg_num, deg_den = 2, 4
    exps_num = torch.tensor(
        _enumerate_exponents(dim, deg_num), dtype=torch.int64,
    )
    exps_den = torch.tensor(
        _enumerate_exponents(dim, deg_den), dtype=torch.int64,
    )

    Phi_num = _eval_monomials(X, exps_num)
    Phi_den = _eval_monomials(X, exps_den)

    # Monomial ordering for dim=2, deg=2 (numerator, 6 terms):
    # (0,0)=1, (0,1)=x1, (0,2)=x1^2, (1,0)=x0, (1,1)=x0*x1, (2,0)=x0^2
    # True P = 1 - x0^2  => only {1, x0^2} nonzero
    #
    # Monomial ordering for dim=2, deg=4 (denominator, 15 terms):
    # True Q = (1+x0*x1)^2 = 1 + 2*x0*x1 + x0^2*x1^2
    #   => only {1, x0*x1, x0^2*x1^2} nonzero (3 of 15)
    # The de-Padeifier should prune the other 12 denominator terms.

    # Fit dense coefficients via null-space method first
    M_num = Phi_num.shape[1]
    F_col = f_sq.unsqueeze(1)
    A = torch.cat([Phi_num, -F_col * Phi_den], dim=1)
    Gram = (A.T @ A) / float(X.shape[0])
    evals, vecs = torch.linalg.eigh(Gram)
    c = vecs[:, 0]
    a_dense = c[:M_num]
    b_dense = c[M_num:]
    if b_dense.numel() > 0 and abs(float(b_dense[0])) > 1e-12:
        a_dense = a_dense / b_dense[0]
        b_dense = b_dense / b_dense[0]

    # Ensure Q > 0
    with torch.no_grad():
        Q_vals = Phi_den @ b_dense
        if float((Q_vals > 0).sum()) / max(1, Q_vals.numel()) < 0.5:
            a_dense = -a_dense
            b_dense = -b_dense

    print("Dense coefficients:")
    print(f"  {_coeffs_summary(a_dense, 'num')}")
    print(f"  {_coeffs_summary(b_dense, 'den')}")

    # Run sparsification
    a_sparse, b_sparse, meta = stlsq_sparsify_rational_coeffs(
        Phi_num=Phi_num,
        Phi_den=Phi_den,
        y=f_sq,
        coeffs_num=a_dense,
        coeffs_den=b_dense,
        cfg=DEFAULT_RAT_STLSQ_CFG,
    )
    _log_sparsify_result(
        "test_direct", a_dense, b_dense, a_sparse, b_sparse, meta,
    )

    print("\nSparse coefficients:")
    print(f"  {_coeffs_summary(a_sparse, 'num')}")
    print(f"  {_coeffs_summary(b_sparse, 'den')}")
    print(f"  accepted={meta['accepted']}, nnz {meta['nnz_seed']} -> {meta['nnz_sparse']}")
    print(f"  MSE {meta['mse_seed']:.3e} -> {meta['mse_sparse']:.3e}")

    assert meta["accepted"] > 0.5, "Expected sparsification to be accepted"

    # Verify the sparse prediction is close to truth
    with torch.no_grad():
        Q = (Phi_den @ b_sparse).clamp(min=1e-8)
        pred = (Phi_num @ a_sparse) / Q
        mse = float((pred - f_sq).square().mean().item())
    print(f"  Sparse prediction MSE: {mse:.3e}")
    assert mse < 1e-3, f"Sparse prediction MSE too high: {mse:.3e}"

    print("PASSED: direct sparsification accepted with good MSE")


# ---------------------------------------------------------------------------
# Test 4: Effect-size threshold vs raw threshold comparison
# ---------------------------------------------------------------------------
def test_effect_size_vs_raw():
    """Compare effect_size_threshold=True vs False to verify both work."""
    X, _, f_sq = _make_data()
    print("\n=== Test 4: effect-size vs raw threshold ===")

    dim = 2
    exps_num = torch.tensor(_enumerate_exponents(dim, 2), dtype=torch.int64)
    exps_den = torch.tensor(_enumerate_exponents(dim, 4), dtype=torch.int64)
    Phi_num = _eval_monomials(X, exps_num)
    Phi_den = _eval_monomials(X, exps_den)

    # Fit dense
    M_num = Phi_num.shape[1]
    F_col = f_sq.unsqueeze(1)
    A = torch.cat([Phi_num, -F_col * Phi_den], dim=1)
    Gram = (A.T @ A) / float(X.shape[0])
    _, vecs = torch.linalg.eigh(Gram)
    c = vecs[:, 0]
    a = c[:M_num]
    b = c[M_num:]
    if b.numel() > 0 and abs(float(b[0])) > 1e-12:
        a = a / b[0]
        b = b / b[0]
    with torch.no_grad():
        if float((Phi_den @ b > 0).sum()) / max(1, Phi_den.shape[0]) < 0.5:
            a, b = -a, -b

    for mode_name, cfg in [
        ("effect_size=True", RationalSparsifyConfig(effect_size_threshold=True)),
        ("effect_size=False", RationalSparsifyConfig(effect_size_threshold=False)),
    ]:
        a_s, b_s, meta = stlsq_sparsify_rational_coeffs(
            Phi_num, Phi_den, f_sq, a.clone(), b.clone(), cfg=cfg,
        )
        nnz = int(meta["nnz_sparse"])
        mse = meta["mse_sparse"]
        print(f"  {mode_name}: nnz={nnz}, MSE={mse:.3e}, accepted={meta['accepted']}")

    print("PASSED: both effect-size modes run without error")


if __name__ == "__main__":
    test_rational_probe_nd()
    test_fit_rational_coeffs_nd()
    test_direct_sparsify()
    test_effect_size_vs_raw()
    print("\n" + "=" * 60)
    print("All de-Padeification tests passed!")
