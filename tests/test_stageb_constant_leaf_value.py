# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Tests for Stage-B constant leaf scalar initialisation helper."""

import pytest
import torch

from nestynet_sr.sr_core.atoms import FreeConstLeaf, PolyLeaf
from nestynet_sr.sr_search.stageB.leaf_utils import _set_constant_leaf_value


def test_set_constant_leaf_value_on_freeconst_leaf():
    leaf = FreeConstLeaf(init=1.0, dtype=torch.float64)
    assert _set_constant_leaf_value(leaf, 2.5)
    assert float(leaf.value.detach().cpu()) == pytest.approx(2.5)


def test_set_constant_leaf_value_on_constant_poly_leaf():
    leaf = PolyLeaf(n_in=0, degree=0, dtype=torch.float64)
    assert _set_constant_leaf_value(leaf, -3.0)
    coeffs = leaf.coeffs.detach().view(-1).cpu()
    assert coeffs.numel() == 1
    assert float(coeffs[0]) == pytest.approx(-3.0)


def test_set_constant_leaf_value_rejects_nonconstant_poly_leaf():
    leaf = PolyLeaf(n_in=1, degree=1, dtype=torch.float64)
    coeffs_before = leaf.coeffs.detach().clone()
    assert not _set_constant_leaf_value(leaf, 7.0)
    assert torch.allclose(leaf.coeffs.detach(), coeffs_before)
