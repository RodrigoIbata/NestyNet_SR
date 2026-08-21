# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import numpy as np
import torch

from nestynet_sr.sr_search.y_transforms import (
    compose_y_stack_ops,
    encode_y_stack_name,
    get_separability_y_ops,
    resolve_y_transform_name,
)


def test_compose_stack_ops_roundtrip_square_log():
    names = ("square", "log")
    y_op, y_inv, stack_name = compose_y_stack_ops(names)
    assert stack_name == encode_y_stack_name(names)

    y = np.array([1.5, 2.0, 3.0], dtype=np.float64)
    t = y_op(y)
    y_back = y_inv(torch.as_tensor(t, dtype=torch.float64)).cpu().numpy()
    assert np.allclose(y_back, y, atol=1e-8, rtol=1e-8)


def test_resolve_stack_name_returns_composed_ops():
    stack_name = encode_y_stack_name(("square", "log"))
    y_op, y_inv, resolved_name = resolve_y_transform_name(stack_name)
    assert resolved_name == stack_name
    y = np.array([2.5, 4.0], dtype=np.float64)
    t = y_op(y)
    y_back = y_inv(torch.as_tensor(t, dtype=torch.float64)).cpu().numpy()
    assert np.allclose(y_back, y, atol=1e-8, rtol=1e-8)


def test_get_separability_ops_accepts_stack_name():
    stack_name = encode_y_stack_name(("square", "log"))
    specs, y_ops, d1_ops, d2_ops = get_separability_y_ops(names=[stack_name])
    assert len(specs) == 1
    assert specs[0].name == stack_name
    x = torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64)
    y = y_ops[0](x)
    d1 = d1_ops[0](x)
    d2 = d2_ops[0](x)
    assert torch.isfinite(y).all()
    assert torch.isfinite(d1).all()
    assert torch.isfinite(d2).all()
