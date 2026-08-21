# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Test: discover the 1D wave equation  u_tt = c² u_xx  from an analytic mock surrogate."""

import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_de.system_de_search import (
    discover_system_de_from_surrogate,
    SystemDESearchConfig,
)

# ── Analytic solution: u(t,x) = sin(k*x - ω*t),  c=2, k=1, ω=2 ──

C = 2.0
K = 1.0
OMEGA = C * K  # 2.0


class WaveSurrogate(torch.nn.Module):
    """Mock surrogate with analytic u, grad, grad_grad for a travelling sine wave."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,2)  columns: [t, x_spatial]
        t, xs = x[:, 0], x[:, 1]
        phase = K * xs - OMEGA * t
        return torch.sin(phase).unsqueeze(1)  # (B,1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        t, xs = x[:, 0], x[:, 1]
        phase = K * xs - OMEGA * t
        cos_ph = torch.cos(phase)
        du_dt = -OMEGA * cos_ph
        du_dx = K * cos_ph
        return torch.stack([du_dt, du_dx], dim=-1).unsqueeze(1)  # (B,1,2)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        t, xs = x[:, 0], x[:, 1]
        phase = K * xs - OMEGA * t
        sin_ph = torch.sin(phase)
        d2u_dtdt = -(OMEGA ** 2) * (-sin_ph)  # +ω² sin
        d2u_dxdx = -(K ** 2) * (-sin_ph)      # +k² sin  (wait, let's be careful)
        # u = sin(phase), phase = k*x - ω*t
        # d²u/dt² = -ω² sin(phase)
        # d²u/dx² = -k² sin(phase)
        # d²u/dtdx = ωk sin(phase)
        d2u_dtdt = -(OMEGA ** 2) * sin_ph
        d2u_dxdx = -(K ** 2) * sin_ph
        d2u_dtdx = OMEGA * K * sin_ph
        B = x.shape[0]
        H = torch.zeros(B, 1, 2, 2)
        H[:, 0, 0, 0] = d2u_dtdt
        H[:, 0, 0, 1] = d2u_dtdx
        H[:, 0, 1, 0] = d2u_dtdx
        H[:, 0, 1, 1] = d2u_dxdx
        return H  # (B,1,2,2)


def test_wave_equation():
    # Generate grid data
    t_vals = torch.linspace(0.0, 2 * math.pi, 50)
    x_vals = torch.linspace(0.0, 2 * math.pi, 50)
    tt, xx = torch.meshgrid(t_vals, x_vals, indexing="ij")
    X = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)  # (2500, 2)

    loader = DataLoader(TensorDataset(X), batch_size=2500, shuffle=False)
    surrogate = WaveSurrogate()

    cfg = SystemDESearchConfig(
        x_axis=0,                    # t is the evolution variable
        order_candidates=(2,),       # anchor = d²u/dt²
        include_d2u=True,
        d2u_axes=((1, 1),),          # u_xx in library
        include_const=True,
        include_x=False,
        include_u=False,
        include_u_cross=False,
        include_xu=False,
        stlsq_lambda=1e-4,
    )

    result = discover_system_de_from_surrogate(surrogate, loader, cfg=cfg, device=torch.device("cpu"))

    print("Order:", result.order)
    print("RMS train:", result.rms_train)
    print("Coefficients:", result.coeffs)
    print("Term ASTs:", result.term_asts)
    print("Equation:", result.format_system())

    assert result.order == 2, f"Expected order 2, got {result.order}"
    assert result.rms_train[0] < 1e-6, f"RMS too large: {result.rms_train[0]}"

    # The equation is: u_tt + c_0 * u_xx = 0
    # So c_0 should be -c² = -4.0
    coeffs = result.coeffs[0].tolist()
    # Find the u_xx coefficient (should be the dominant non-zero one)
    found_c2 = False
    for i, c in enumerate(coeffs):
        if abs(c) > 0.1:
            assert abs(c - (-C ** 2)) < 1e-4, f"Expected coeff ≈ -4.0, got {c}"
            found_c2 = True
    assert found_c2, "Did not find u_xx coefficient"

    print("\nPASSED: Wave equation u_tt = c² u_xx discovered correctly.")


if __name__ == "__main__":
    test_wave_equation()
