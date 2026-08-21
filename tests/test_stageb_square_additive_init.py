# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_search.stageB.transforms import _square_additive_leaf_targets


def test_square_additive_init_targets_have_original_output_scale():
    torch.manual_seed(0)
    n = 2048
    x = 1.0 + 4.0 * torch.rand(n, 6, dtype=torch.float64)

    # pb115-like hard atom:
    # u = sqrt((x0*x1^2)^2 + (x1*(x2 - x3*x4))^2)
    left = x[:, 0] * x[:, 1].square()
    right = x[:, 1] * (x[:, 2] - x[:, 3] * x[:, 4])
    u = torch.sqrt(left.square() + right.square())
    v = u.square()

    L_target, R_target, diag = _square_additive_leaf_targets(
        x,
        u,
        v,
        [0, 1],
        [1, 2, 3, 4],
    )

    pred = torch.sqrt(L_target.square() + R_target.square())
    ratio = torch.median(pred) / torch.median(u)

    assert torch.isfinite(L_target).all()
    assert torch.isfinite(R_target).all()
    assert torch.all(L_target >= 0)
    assert torch.all(R_target >= 0)
    assert 0.5 < float(ratio) < 2.0
    assert diag["median_ratio"] < 2.0


def test_square_additive_init_targets_are_not_square_space_components():
    x = torch.linspace(1.0, 5.0, 512, dtype=torch.float64).reshape(-1, 1)
    x = torch.cat([x, 1.0 + 0.1 * x], dim=1)
    u = 3.0 * x[:, 0] + 2.0 * x[:, 1]
    v = u.square()

    L_target, R_target, _diag = _square_additive_leaf_targets(
        x,
        u,
        v,
        [0],
        [1],
    )

    # The old bug effectively initialized leaves in u^2-space.  The new
    # targets must live on the original u scale.
    assert float(torch.median(L_target)) < float(torch.median(v))
    assert float(torch.median(R_target)) < float(torch.median(v))
    assert float(torch.median(L_target)) < 10.0 * float(torch.median(u))
    assert float(torch.median(R_target)) < 10.0 * float(torch.median(u))
