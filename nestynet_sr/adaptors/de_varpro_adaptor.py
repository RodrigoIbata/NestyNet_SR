# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""VarPro adaptor for DE residuals.

Builds (y, X) for an anchored DE residual and eliminates the linear
coefficients beta by LSQ.

Note
----
forward(..., beta_override=...) evaluates residuals with a *fixed* coefficient
vector. This is crucial for correct validation scoring (do not re-fit beta on
validation).
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from nestynet_sr.adaptors.u_feature_leaf import UFeatureCache
from nestynet_sr.sr_de.de_search import _as_N, _eval_ast


class DEVarProAdaptor(nn.Module):
    """Variable Projection adaptor for ODE residuals."""

    def __init__(
        self,
        composite_model: Optional[nn.Module],
        order: int,
        x_axis: int,
        cache: UFeatureCache,
        term_asts: Optional[List] = None,
        lambda_reg: float = 1e-10,
    ):
        super().__init__()
        self.composite_model = composite_model
        self.order = int(order)
        self.x_axis = int(x_axis)
        self.cache = cache
        self.term_asts = term_asts if term_asts is not None else []
        self.lambda_reg = float(lambda_reg)

        if composite_model is not None:
            self.add_module("_composite", composite_model)

    def design_and_target(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (X, y, anchor) for the current term library.

        y is the *LSQ target* (=-anchor), so residuals are r = y - X beta.
        """
        if self.order == 1:
            self.cache.ensure(x, need_grad=True, need_hess=False)
            anchor = self.cache.g[:, 0, self.x_axis]
        elif self.order == 2:
            self.cache.ensure(x, need_grad=False, need_hess=True)
            anchor = self.cache.H[:, 0, self.x_axis, self.x_axis]
        else:
            raise ValueError(f"Unsupported DE order: {self.order}")

        y = -anchor
        X = self._evaluate_library_terms(x)
        return X, y, anchor

    def solve_beta(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Solve argmin_beta ||y - X beta||^2 with optional ridge."""
        B, K = X.shape
        if K == 0:
            return X.new_zeros((0,))

        XtX = X.T @ X
        Xty = X.T @ y

        if self.lambda_reg > 0.0:
            XtX = XtX + self.lambda_reg * torch.eye(K, device=X.device, dtype=X.dtype)

        try:
            beta = torch.linalg.solve(XtX, Xty)
        except RuntimeError:
            beta = torch.linalg.lstsq(X, y).solution

        return beta

    def forward(
        self, x: torch.Tensor, *, beta_override: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute residuals, optionally using a fixed beta.

        Parameters
        ----------
        x : torch.Tensor
            (B, Nx)
        beta_override : torch.Tensor, optional
            If provided, must have shape (K,) and will be used instead of solving.

        Returns
        -------
        dict with keys: residuals, beta, y, X, anchor
        """
        X, y, anchor = self.design_and_target(x)

        if beta_override is None:
            beta = self.solve_beta(X, y)
        else:
            beta = beta_override.to(device=X.device, dtype=X.dtype).view(-1)
            if beta.numel() != X.shape[1]:
                raise ValueError(
                    f"beta_override has numel={beta.numel()} but X has K={X.shape[1]} columns"
                )

        residuals = y - X @ beta

        return {
            "residuals": residuals,
            "beta": beta,
            "y": y,
            "X": X,
            "anchor": anchor,
        }

    def _evaluate_library_terms(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate library terms from discovered term ASTs.

        Returns
        -------
        X : torch.Tensor
            Design matrix (B, K). If K==0 (empty library), returns a (B, 0) tensor.
        """
        B = x.shape[0]

        # Empty library is allowed: residual reduces to just the anchor term.
        if self.term_asts is None or len(self.term_asts) == 0:
            return x.new_zeros((B, 0))

        cols = []
        for term_ast in self.term_asts:
            if term_ast is None:
                cols.append(torch.ones(B, device=x.device, dtype=x.dtype))
            else:
                v = _as_N(_eval_ast(term_ast, x, self.cache))
                cols.append(v)

        # Defensive: if cols is empty for any reason, return empty design.
        if len(cols) == 0:
            return x.new_zeros((B, 0))

        return torch.stack(cols, dim=1)

    def build_cache(self, batch):
        """Build cache for optimizer interfaces."""
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        result = self.forward(x)
        return {
            "beta": result["beta"],
            "residuals": result["residuals"],
            "y": result["y"],
            "X": result["X"],
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
