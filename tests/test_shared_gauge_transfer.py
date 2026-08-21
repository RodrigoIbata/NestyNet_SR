# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""pb115 hardening: functional shared-coordinate gauge transfer in the
overlapping-square additive split. Exercises the real
_shared_gauge_transfer against the pb115 structure and a negative control."""

import torch

from nestynet_sr.sr_search.stageB.transforms import _shared_gauge_transfer

torch.set_default_dtype(torch.float64)


def _pb115_domain(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = 1.0 + 4.0 * torch.rand(n, 5, generator=g)  # x0..x4 in [1,5]
    x0, x1, x2, x3, x4 = (X[:, i] for i in range(5))
    A2 = x0**2 * x1**4                    # child F on g1=(0,1): clean monomial
    B2 = x1**2 * (x2 - x3 * x4) ** 2      # child G on g2=(1,2,3,4)
    return X, A2, B2


V1, V2 = [0, 1], [1, 2, 3, 4]


def test_transfers_legal_x1_gauge_and_cleans_children():
    X, A2, B2 = _pb115_domain()
    x1 = X[:, 1]
    h = 6.0 * x1**4                       # legal gauge: shared coord (x1) only
    F, G = A2 + h, B2 - h                 # what the marginal projection lands
    F2, G2, diag = _shared_gauge_transfer(X, F, G, V1, V2)
    assert diag["applied"] == 1.0, diag
    assert diag["cert_rel"] < 1e-4
    # F cleaned back to the pure monomial, sum preserved
    assert torch.allclose(F2, A2, rtol=1e-3, atol=1e-3 * float(A2.abs().mean()))
    assert torch.allclose(F2 + G2, F + G, rtol=1e-8)
    # cleaned child is now a clean power law
    lx = torch.stack([torch.log(X[:, 0]), torch.log(x1), torch.ones_like(x1)], 1)
    c = torch.linalg.lstsq(lx, torch.log(F2.clamp_min(1e-30)).unsqueeze(1)).solution.squeeze(1)
    pred = lx @ c
    r2 = 1 - float(((torch.log(F2.clamp_min(1e-30)) - pred) ** 2).sum()
                   / ((torch.log(F2.clamp_min(1e-30)) - torch.log(F2.clamp_min(1e-30)).mean()) ** 2).sum())
    assert r2 > 0.999, f"cleaned child should be clean power law, R2={r2}"


def test_no_transfer_on_already_clean_children():
    X, A2, B2 = _pb115_domain(seed=2)
    F2, G2, diag = _shared_gauge_transfer(X, A2, B2, V1, V2)
    # already-clean F: either not applied, or applied with ~zero gauge (F unchanged)
    assert torch.allclose(F2 + G2, A2 + B2, rtol=1e-8)
    assert torch.allclose(F2, A2, rtol=1e-2, atol=1e-2 * float(A2.abs().mean()))


def test_rejects_cross_variable_leak():
    # An illegal leak depending on a NON-shared var of the other child must not
    # be absorbed as an x1-only gauge -> certificate residual stays large.
    X, A2, B2 = _pb115_domain(seed=3)
    x1, x2, x3, x4 = X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    leak = 5.0 * x1**2 * (x2 - x3 * x4)   # contains x2,x3,x4 (G's vars)
    F, G = A2 + leak, B2 - leak
    F2, G2, diag = _shared_gauge_transfer(X, F, G, V1, V2)
    assert diag["applied"] == 0.0, f"must reject cross-leak, got {diag}"
    assert torch.equal(F2, F) and torch.equal(G2, G)


def test_no_shared_coords_is_noop():
    X, A2, B2 = _pb115_domain(seed=4)
    F2, G2, diag = _shared_gauge_transfer(X, A2, B2, [0], [2, 3])  # disjoint
    assert diag["applied"] == 0.0
    assert torch.equal(F2, A2) and torch.equal(G2, B2)
