# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_search.chainrule_wrapper import ChainRuleYModel, VirtualYModel
from nestynet_sr.sr_search.y_transforms import get_y_transform_registry


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x[:, :1]

    def grad(self, x, out_dim=None):
        b, nx = x.shape
        g = torch.zeros(b, 1, nx, dtype=x.dtype, device=x.device)
        g[:, 0, 0] = 1.0
        if out_dim is not None:
            return g[:, out_dim]
        return g

    def grad_grad(self, x, out_dim=None):
        b, nx = x.shape
        h = torch.zeros(b, 1, nx, nx, dtype=x.dtype, device=x.device)
        if out_dim is not None:
            return h[:, out_dim]
        return h

    def parameters(self, recurse=True):
        return iter([self._p])


def _find_transform(name: str):
    reg = get_y_transform_registry()
    for yt in reg:
        if yt.name == name:
            return yt
    raise AssertionError(f"Transform '{name}' not found")


def test_virtual_y_model_clips_extreme_derivatives():
    model = _ToyModel()
    yt = _find_transform("reciprocal")
    wrapper = VirtualYModel(model, yt, max_abs_deriv=100.0)

    x = torch.tensor(
        [[1.0e-30, 0.0], [-1.0e-30, 0.0], [1.0e-20, 0.0]],
        dtype=torch.float64,
    )
    g = wrapper.grad(x)
    h = wrapper.grad_grad(x)

    assert torch.isfinite(g).all()
    assert torch.isfinite(h).all()
    assert torch.max(torch.abs(g)).item() <= 100.0 + 1e-9
    assert torch.max(torch.abs(h)).item() <= 100.0 + 1e-9


def test_chainrule_alias_is_virtual_model():
    model = _ToyModel()
    yt = _find_transform("square")
    wrapper = ChainRuleYModel(model, yt)
    assert isinstance(wrapper, VirtualYModel)
