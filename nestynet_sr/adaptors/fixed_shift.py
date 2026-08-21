# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch


class FixedOutputShiftAdaptor(torch.nn.Module):
    """
    Wrap an existing adaptor/module and add a *fixed* output shift (buffer).
    This does not change any x-derivatives or parameter Jacobians of the core.
    """

    def __init__(self, core: torch.nn.Module, shift: torch.Tensor):
        super().__init__()
        self.core = core
        if not torch.is_tensor(shift):
            shift = torch.tensor(shift, dtype=torch.get_default_dtype())
        self.register_buffer("shift", shift.detach().clone())

    def forward(self, x):
        y = self.core(x)
        return y + self.shift

    # Delegate everything else to core. This works well because the shift is constant:
    # Jacobians/Hessians w.r.t x and params are unchanged.
    def __getattr__(self, name):
        if name in ("core", "shift", "forward", "__getattr__", "__class__"):
            return super().__getattr__(name)
        return getattr(self.core, name)
