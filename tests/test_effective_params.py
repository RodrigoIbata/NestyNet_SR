#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Unit tests for _count_effective_params and _effective_ratpoly_params.

Verifies that the effective-parameter counting correctly identifies
active (non-zero, normalisation-significant) polynomial and rational-
polynomial coefficients, and falls back to nominal counts for non-poly
leaves.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from nestynet_sr.sr_core.atoms import (
    PolyLeaf,
    RRationalPolyLeaf,
    _eval_monomials,
)
from nestynet_sr.sr_core.bridges import AddNode, AtomNode
from nestynet_sr.sr_search.stageB.engine import (
    _count_effective_params,
    _effective_ratpoly_params,
)


# ── helpers ──────────────────────────────────────────────────────────
def _ok(label: str, expected, actual):
    status = "PASS" if expected == actual else "FAIL"
    print(f"  [{status}] {label}: expected={expected}, got={actual}")
    return expected == actual


# ── Test 1: RRationalPolyLeaf  z^4 / (z^4 - 2z^2 + 1) ─────────────
def test_ratpoly_narrow_range():
    """deg_num=4, deg_den=4, free num coeffs all zero → 3 active den params."""
    leaf = RRationalPolyLeaf(indices=(0,), deg_num=4, deg_den=4)

    # Numerator: all free coefficients zero.  Lead monomial (z^4) fixed to 1.
    with torch.no_grad():
        leaf.coeffs_num.zero_()
        # Denominator: 1 - 2z^2 + z^4  →  coeffs [1, 0, -2, 0, 1]
        leaf.coeffs_den.copy_(torch.tensor([1.0, 0.0, -2.0, 0.0, 1.0]))

    atom = AtomNode(kind="ratpoly", var_idxs=(0,))
    z_data = torch.empty(200, 1).uniform_(0.2, 0.7)

    n = _effective_ratpoly_params(leaf, atom, z_data, 1e-6, _eval_monomials)
    return _ok("ratpoly narrow-range z∈[0.2,0.7]", 3, n)


# ── Test 2: large variable range – tiny raw coeff, big normalised ───
def test_ratpoly_wide_range():
    """z∈[1,1000]: raw coeff 1e-12 on z^4 is significant after normalisation."""
    # Use deg_num=0 (constant numerator) so the fixed lead is 1 (not z^4),
    # keeping the peak at O(1) and letting the denominator z^4 term shine.
    leaf = RRationalPolyLeaf(indices=(0,), deg_num=0, deg_den=4)

    with torch.no_grad():
        # Denominator: 1 + 0·z + 0·z^2 + 0·z^3 + 1e-12·z^4
        leaf.coeffs_den.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1e-12]))

    atom = AtomNode(kind="ratpoly", var_idxs=(0,))
    z_data = torch.empty(200, 1).uniform_(1.0, 1000.0)

    n = _effective_ratpoly_params(leaf, atom, z_data, 1e-6, _eval_monomials)
    # Constant term: contrib = 1.  z^4 term: 1e-12 * median(z^4) >> threshold.
    return _ok("ratpoly wide-range z∈[1,1000]", 2, n)


# ── Test 3: PolyLeaf with some zero coefficients ────────────────────
def test_poly_sparse():
    """PolyLeaf(n_in=1, degree=3, min_total=0): 2 of 4 coeffs non-zero → 2."""
    leaf = PolyLeaf(n_in=1, degree=3, min_total=0)

    with torch.no_grad():
        leaf.coeffs.copy_(torch.tensor([2.0, 0.0, -1.5, 0.0]))

    atom = AtomNode(kind="poly", var_idxs=(0,))
    z_data = torch.empty(200, 1).uniform_(0.5, 2.0)

    # Use _count_effective_params through a tiny wrapper model.
    model = nn.Module()
    model.leaf = nn.ModuleList([leaf])

    n = _count_effective_params(model, atom, z_data)
    return _ok("poly sparse coeffs", 2, n)


# ── Test 4: full model mock (non-poly + ratpoly) ────────────────────
def test_full_model():
    """Two nn.Linear leaves (2+3 params) + one ratpoly (3 effective) → 8."""
    # Leaf 0: nn.Linear(1, 1) → 2 params (weight + bias)
    lin1 = nn.Linear(1, 1, dtype=torch.float64)
    # Leaf 1: nn.Linear(2, 1) → 3 params
    lin2 = nn.Linear(2, 1, dtype=torch.float64)
    # Leaf 2: ratpoly with 3 effective params (same as Test 1)
    rpoly = RRationalPolyLeaf(indices=(0,), deg_num=4, deg_den=4)
    with torch.no_grad():
        rpoly.coeffs_num.zero_()
        rpoly.coeffs_den.copy_(torch.tensor([1.0, 0.0, -2.0, 0.0, 1.0]))

    model = nn.Module()
    model.leaf = nn.ModuleList([lin1, lin2, rpoly])

    # AST: Add(Add(atom0, atom1), atom2)  →  DFS yields [atom0, atom1, atom2]
    atom0 = AtomNode(kind="nn", var_idxs=(0,))
    atom1 = AtomNode(kind="nn", var_idxs=(0, 1))
    atom2 = AtomNode(kind="ratpoly", var_idxs=(0,))
    root = AddNode(AddNode(atom0, atom1), atom2)

    x_data = torch.empty(200, 2).uniform_(0.2, 0.7)

    n = _count_effective_params(model, root, x_data)
    return _ok("full model (5 non-poly + 3 ratpoly)", 8, n)


# ── Test 5: non-polynomial leaf → nominal param count ───────────────
def test_non_poly_leaf():
    """nn.Linear(1,1) leaf → 2 trainable params returned as-is."""
    lin = nn.Linear(1, 1, dtype=torch.float64)
    model = nn.Module()
    model.leaf = nn.ModuleList([lin])

    atom = AtomNode(kind="nn", var_idxs=(0,))
    x_data = torch.empty(50, 1).uniform_(-1, 1)

    n = _count_effective_params(model, atom, x_data)
    return _ok("non-poly leaf (nn.Linear)", 2, n)


# ── runner ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== _count_effective_params / _effective_ratpoly_params tests ===\n")
    results = [
        test_ratpoly_narrow_range(),
        test_ratpoly_wide_range(),
        test_poly_sparse(),
        test_full_model(),
        test_non_poly_leaf(),
    ]
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} passed")
    sys.exit(0 if all(results) else 1)
