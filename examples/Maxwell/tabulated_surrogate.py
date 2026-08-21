#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Shared lookup surrogate for tabulated vector-field data."""

from __future__ import annotations

import torch


class TabulatedVectorSurrogate(torch.nn.Module):
    """Lookup surrogate built from tabulated coordinates/fields/gradients.

    This class expects queries exactly on tabulated points (same grid).
    """

    def __init__(
        self,
        x_table: torch.Tensor,
        y_table: torch.Tensor,
        g_table: torch.Tensor,
        h_table: torch.Tensor | None = None,
        *,
        coord_decimals: int = 12,
    ) -> None:
        super().__init__()
        if x_table.ndim != 2:
            raise ValueError(f"x_table must be 2D, got shape {tuple(x_table.shape)}")
        if y_table.ndim != 2:
            raise ValueError(f"y_table must be 2D, got shape {tuple(y_table.shape)}")
        if g_table.ndim != 3:
            raise ValueError(f"g_table must be 3D, got shape {tuple(g_table.shape)}")
        if x_table.shape[0] != y_table.shape[0] or x_table.shape[0] != g_table.shape[0]:
            raise ValueError("x/y/g tables must share the first dimension")
        if g_table.shape[1] != y_table.shape[1]:
            raise ValueError("g_table second dimension must match y_table outputs")
        if g_table.shape[2] != x_table.shape[1]:
            raise ValueError("g_table third dimension must match x_table inputs")
        if h_table is not None:
            if h_table.ndim != 4:
                raise ValueError(f"h_table must be 4D, got shape {tuple(h_table.shape)}")
            if h_table.shape[0] != x_table.shape[0]:
                raise ValueError("h_table first dimension must match x_table rows")
            if h_table.shape[1] != y_table.shape[1]:
                raise ValueError("h_table second dimension must match y_table outputs")
            if h_table.shape[2] != x_table.shape[1] or h_table.shape[3] != x_table.shape[1]:
                raise ValueError("h_table last two dimensions must match x_table inputs")

        self.register_buffer("x_table", x_table.contiguous())
        self.register_buffer("y_table", y_table.contiguous())
        self.register_buffer("g_table", g_table.contiguous())
        self.has_hessian = h_table is not None
        self.register_buffer(
            "h_table",
            h_table.contiguous() if h_table is not None else None,
        )
        self._dummy_param = torch.nn.Parameter(
            torch.zeros(1, dtype=x_table.dtype), requires_grad=False
        )

        self.coord_decimals = int(coord_decimals)
        self.coord_scale = float(10 ** self.coord_decimals)
        self._index = {
            self._coord_key(self.x_table[i]): int(i)
            for i in range(int(self.x_table.shape[0]))
        }

    def _coord_key(self, row: torch.Tensor) -> tuple[int, ...]:
        vals = row.detach().cpu().tolist()
        return tuple(int(round(float(v) * self.coord_scale)) for v in vals)

    def _lookup_indices(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.x_table.shape[1]:
            raise ValueError(
                f"Expected x shape (N, {self.x_table.shape[1]}), got {tuple(x.shape)}"
            )
        idx = []
        for r in x:
            key = self._coord_key(r)
            j = self._index.get(key, None)
            if j is None:
                raise KeyError(
                    "Query point not found in lookup table. "
                    "Use dataloader coordinates that match the generated grid exactly."
                )
            idx.append(j)
        return torch.tensor(idx, dtype=torch.long, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self._lookup_indices(x)
        return self.y_table[idx]

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        idx = self._lookup_indices(x)
        return self.g_table[idx]

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_hessian:
            raise RuntimeError(
                "TabulatedVectorSurrogate was built without a Hessian table; "
                "pass h_table to enable second-derivative (Laplacian) terms."
            )
        idx = self._lookup_indices(x)
        return self.h_table[idx]
