# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_search.factorized_search.subproblem_witness import estimate_pointwise_target_jets


def test_estimate_pointwise_target_jets_recovers_smooth_1d_gradients():
    x = torch.linspace(-1.2, 1.2, 41, dtype=torch.float64).unsqueeze(-1)
    target = torch.sin(x)

    jets = estimate_pointwise_target_jets(
        x,
        target,
        include_d2=True,
        max_rows=24,
    )

    grad = jets["grad"]
    d2 = jets["d2"]
    assert jets["source"] == "numeric_local_quadratic"
    assert jets["status"] in {"ok", "ok_with_global_fallback"}
    assert grad is not None
    assert d2 is not None
    assert tuple(grad.shape) == tuple(x.shape)
    assert tuple(d2.shape) == tuple(x.shape)

    true_grad = torch.cos(x)
    true_d2 = -torch.sin(x)
    grad_mae = torch.mean(torch.abs(grad - true_grad)).item()
    d2_mae = torch.mean(torch.abs(d2 - true_d2)).item()
    assert grad_mae < 0.20
    assert d2_mae < 0.45
