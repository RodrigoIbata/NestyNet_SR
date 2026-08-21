# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

from nestynet_sr.discovery.witness import capture_runtime_witness, capture_symbolic_witness
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node


class _QuadraticRuntimeModel:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :1] * x[:, :1]) + 3.0 * x[:, 1:2]

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        grad = torch.stack((2.0 * x[:, 0], 3.0 * torch.ones_like(x[:, 1])), dim=1)
        return grad.unsqueeze(1)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((int(x.shape[0]), 1, int(x.shape[1]), int(x.shape[1])), dtype=x.dtype)
        out[:, 0, 0, 0] = 2.0
        return out


class _RuntimeCandidate:
    def __init__(self):
        self.model = _QuadraticRuntimeModel()
        self.y_inverse = None


def test_capture_runtime_witness_uses_model_grad_and_hessian():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    witness = capture_runtime_witness(
        _RuntimeCandidate(),
        x,
        predict_value_fn=lambda candidate, xx: candidate.model(xx)[:, 0],
        capture_gradients=True,
        capture_hessian_diag=True,
        diagnostic_set="extended",
    )

    assert torch.allclose(witness["observable"], torch.tensor([7.0, 21.0], dtype=torch.float64))
    assert torch.allclose(
        witness["derivative"],
        torch.tensor([[2.0, 3.0], [6.0, 3.0]], dtype=torch.float64),
    )
    assert witness["diagnostic"]["hdiag_abs_mean"] == 1.0
    assert "value_q50" in witness["diagnostic"]


def test_capture_symbolic_witness_uses_autograd():
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    witness = capture_symbolic_witness(
        expr_ast=("sqr", ("var", 0)),
        x=x,
        forward_value_fn=lambda node, xx: eval_node(node, xx),
        capture_gradients=True,
        capture_hessian_diag=True,
    )

    assert torch.allclose(witness["observable"], torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64))
    assert torch.allclose(
        witness["derivative"],
        torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float64),
    )
    assert witness["diagnostic"]["hdiag_abs_mean"] == 2.0
