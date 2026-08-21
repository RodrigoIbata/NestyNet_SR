# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

from nestynet_sr.discovery.witness import capture_runtime_witness, capture_symbolic_witness
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node


class _OddRuntimeModel:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((int(x.shape[0]), 1, int(x.shape[1])), dtype=x.dtype)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((int(x.shape[0]), 1, int(x.shape[1]), int(x.shape[1])), dtype=x.dtype)


class _RuntimeCandidate:
    def __init__(self, model):
        self.model = model
        self.y_inverse = None


def test_capture_runtime_witness_emits_physics_diagnostics_for_odd_profile():
    x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    witness = capture_runtime_witness(
        _RuntimeCandidate(_OddRuntimeModel()),
        x,
        predict_value_fn=lambda candidate, xx: candidate.model(xx)[:, 0],
        capture_gradients=True,
        capture_hessian_diag=True,
        diagnostic_set="physics",
    )

    diag = witness["diagnostic"]
    assert diag["zero_crossing_count"] == 1.0
    assert diag["mirror_odd_residual"] < 1.0e-12
    assert diag["monotonicity_switch_count"] == 0.0
    assert "singularity_margin_proxy" in diag
    assert "tail_slope_gap_abs" in diag


def test_capture_symbolic_witness_emits_physics_diagnostics_for_quadratic():
    x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    witness = capture_symbolic_witness(
        expr_ast=("sqr", ("var", 0)),
        x=x,
        forward_value_fn=lambda node, xx: eval_node(node, xx),
        capture_gradients=True,
        capture_hessian_diag=True,
        diagnostic_set="physics",
    )

    diag = witness["diagnostic"]
    assert diag["mirror_even_residual"] < 1.0e-12
    assert diag["zero_crossing_count"] == 1.0
    assert diag["convexity_pos_frac"] == 1.0
    assert diag["monotonicity_switch_count"] == 1.0
    assert "curvature_spike_ratio" in diag
