# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Test: discover coupled 1D Maxwell equations from a mock surrogate.

Ground truth (vacuum, c=1):
    ∂E/∂t = -∂B/∂x
    ∂B/∂t = -∂E/∂x

Analytic solution (two counter-propagating modes so E ≠ B):
    E(t,x) = sin(x-t) + sin(x+t)
    B(t,x) = sin(x-t) - sin(x+t)

The system anchor is order=1 w.r.t. t (axis 0).  The library contains
spatial derivatives ∂E/∂x and ∂B/∂x.  STLSQ should discover:
    ∂E/∂t + 1·∂B/∂x = 0
    ∂B/∂t + 1·∂E/∂x = 0
"""

import math
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_dtype(torch.float64)

# ── Constants ──────────────────────────────────────────────
C = 1.0
K = 1.0
OMEGA = C * K  # = 1.0


# ── Mock surrogate ────────────────────────────────────────
class Maxwell1DSurrogate(torch.nn.Module):
    """Mock surrogate: Ny=2, two counter-propagating modes.

    E(t,x) = sin(x-t) + sin(x+t)
    B(t,x) = sin(x-t) - sin(x+t)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, xs = x[:, 0], x[:, 1]
        s_m = torch.sin(xs - t)  # right-moving
        s_p = torch.sin(xs + t)  # left-moving
        E = s_m + s_p
        B = s_m - s_p
        return torch.stack([E, B], dim=1)  # (B, 2)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        t, xs = x[:, 0], x[:, 1]
        c_m = torch.cos(xs - t)
        c_p = torch.cos(xs + t)
        # E = sin(x-t)+sin(x+t): ∂E/∂t = -c_m+c_p,  ∂E/∂x = c_m+c_p
        # B = sin(x-t)-sin(x+t): ∂B/∂t = -c_m-c_p,  ∂B/∂x = c_m-c_p
        gE = torch.stack([-c_m + c_p, c_m + c_p], dim=-1)
        gB = torch.stack([-c_m - c_p, c_m - c_p], dim=-1)
        return torch.stack([gE, gB], dim=1)  # (B, 2, 2)

    def parameters(self):
        return iter([torch.zeros(1)])


# ── Test ──────────────────────────────────────────────────
def test_coupled_1d_maxwell():
    from nestynet_sr.sr_de.system_de_search import (
        discover_system_de_from_surrogate,
        SystemDESearchConfig,
    )

    # Grid data
    t_vals = torch.linspace(0.0, 2 * math.pi, 60)
    x_vals = torch.linspace(0.0, 2 * math.pi, 60)
    tt, xx = torch.meshgrid(t_vals, x_vals, indexing="ij")
    X = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)
    loader = DataLoader(TensorDataset(X), batch_size=len(X), shuffle=False)

    surrogate = Maxwell1DSurrogate()

    cfg = SystemDESearchConfig(
        x_axis=0,                   # t is evolution axis
        order_candidates=(1,),      # first-order system
        include_const=False,
        include_x=False,
        include_u=False,
        include_u_cross=False,
        include_xu=False,
        include_du=True,
        du_axes=(1,),               # spatial derivatives ∂/∂x
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(
        surrogate, loader, cfg=cfg, device=torch.device("cpu")
    )

    print("Order:", result.order)
    print("RMS train:", result.rms_train)
    print("Coefficients:\n", result.coeffs)
    print("System:\n", result.format_system())

    assert result.order == 1, f"Expected order 1, got {result.order}"

    # Both equations should have very low residual
    for i, rms in enumerate(result.rms_train):
        print(f"  eq{i} RMS = {rms:.2e}")
        assert rms < 1e-6, f"RMS too large for eq{i}: {rms}"

    # Verify cross-coupling structure:
    # eq0 (∂E/∂t): coefficient of ∂B/∂x ≈ +1  (since ∂E/∂t + 1·∂B/∂x = 0)
    # eq1 (∂B/∂t): coefficient of ∂E/∂x ≈ +1
    coeffs = result.coeffs  # shape (2, K_sel)
    print(f"  coeffs shape: {coeffs.shape}")
    print(f"  term_asts: {[str(t) for t in result.term_asts]}")

    # At least one non-zero coefficient per equation
    for i in range(2):
        row = coeffs[i].abs()
        assert row.max() > 0.5, f"eq{i}: no significant coefficient found"

    print("\nPASSED: Coupled 1D Maxwell equations discovered correctly.")


if __name__ == "__main__":
    test_coupled_1d_maxwell()
