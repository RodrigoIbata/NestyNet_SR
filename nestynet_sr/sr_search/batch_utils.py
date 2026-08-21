# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Small dataloader helpers shared across proposal mechanisms."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def first_batch_xy(
    loader: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the first (x, y) batch from a DataLoader-like iterator.

    Many parts of the codebase only need a single cached batch for cheap
    screening.  Different training scripts sometimes yield:
      - (x, y)
      - (x, y, ...)
      - x

    This helper normalises those cases.
    """
    b0 = next(iter(loader))
    if isinstance(b0, (list, tuple)):
        xb = b0[0]
        yb = b0[1] if len(b0) > 1 else None
    else:
        xb = b0
        yb = None
    xb = xb.to(device=device, dtype=dtype)
    if yb is not None and isinstance(yb, torch.Tensor):
        yb = yb.to(device=device, dtype=dtype)
    return xb, yb
