# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet.adaptors import AutogradAdaptor
from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor
from nestynet_sr.sr_core.bridges import AtomNode


class _TinyThreeParamLeaf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(
            torch.tensor([0.4, -0.2, 1.3], dtype=torch.float64)
        )

    def forward(self, x):
        z = x[:, 0]
        a, b, c = self.theta
        return (a * z + b * z * z + c).unsqueeze(-1)


def test_ast_composite_autograd_param_jacobian_uses_forward_columns_for_many_rows(monkeypatch):
    leaf = AutogradAdaptor(_TinyThreeParamLeaf())
    model = ASTCompositeAdaptor(AtomNode("nn", (0,)), [leaf])
    x = torch.linspace(-2.0, 2.0, 64, dtype=torch.float64).unsqueeze(-1)
    y = torch.zeros_like(x)

    def _fail_jacrev(*_args, **_kwargs):
        raise AssertionError("jacrev should not be used for R >> P leaf Jacobian fallback")

    monkeypatch.setattr(torch.func, "jacrev", _fail_jacrev)

    cache = model.build_cache((x, y))
    J = cache["leaves"][0]["jac"]

    z = x[:, 0]
    expected = -torch.stack([z, z * z, torch.ones_like(z)], dim=1).reshape(64, 1, 3)

    assert torch.allclose(J, expected, rtol=1e-10, atol=1e-10)
