# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Test: discover a vector curl equation  ∂E/∂t = curl(B)  using VectorDESearchConfig.

Physics (Faraday-like, c=1):
    ∂E/∂t + (-1)·curl(B) = 0

Analytic plane-wave solution (propagating in z):
    E = (sin(z-t), 0, 0)
    B = (0, sin(z-t), 0)

Verification:
    curl(B) = (∂Bz/∂y - ∂By/∂z, ∂Bx/∂z - ∂Bz/∂x, ∂By/∂x - ∂Bx/∂y)
            = (0 - (-cos(z-t)), 0, 0) = (cos(z-t), 0, 0)
    ∂Ex/∂t = -cos(z-t)
    So ∂E/∂t + (-1)·curl(B) = 0  ✓

Surrogate has Ny=6: (Ex,Ey,Ez,Bx,By,Bz) with axes (t, x, y, z) = (0,1,2,3).
"""

import math
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_dtype(torch.float64)


# ── Mock surrogate ────────────────────────────────────────
class CurlSurrogate(torch.nn.Module):
    """Ny=6 surrogate for E=(sin(z-t),0,0), B=(0,sin(z-t),0).

    Input x: (B, 4) = [t, x, y, z].
    Output: (B, 6) = [Ex, Ey, Ez, Bx, By, Bz].
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, z = x[:, 0], x[:, 3]
        phase = z - t
        s = torch.sin(phase)
        zero = torch.zeros_like(s)
        # (Ex, Ey, Ez, Bx, By, Bz)
        return torch.stack([s, zero, zero, zero, s, zero], dim=1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 6, 4) gradient tensor."""
        B = x.shape[0]
        t, z = x[:, 0], x[:, 3]
        phase = z - t
        c = torch.cos(phase)

        G = torch.zeros(B, 6, 4, dtype=x.dtype, device=x.device)
        # Ex = sin(z-t): ∂/∂t = -cos, ∂/∂z = cos
        G[:, 0, 0] = -c
        G[:, 0, 3] = c
        # By = sin(z-t): same derivatives
        G[:, 4, 0] = -c
        G[:, 4, 3] = c
        return G

    def parameters(self):
        return iter([torch.zeros(1)])


# ── Test ──────────────────────────────────────────────────
def test_vector_curl_equation():
    from nestynet_sr.sr_de.system_de_search import (
        discover_vector_de_from_surrogate,
        VectorDESearchConfig,
    )
    from nestynet_sr.sr_core.bridges import VField
    from nestynet_sr.sr_de.vector_ops import curl

    # Define magnetic field B = outputs 3,4,5.
    B = VField("B", base_out_idx=3, n_comp=3, comp_names=("x", "y", "z"))

    # curl(B) with spatial axes (x=1, y=2, z=3)
    curl_B = curl(B, spatial_axes=(1, 2, 3))  # Vec of 3 scalar Nodes

    # Grid data: (t, x, y, z) — only t and z matter
    N = 40
    t_vals = torch.linspace(0.0, 2 * math.pi, N)
    z_vals = torch.linspace(0.0, 2 * math.pi, N)
    tt, zz = torch.meshgrid(t_vals, z_vals, indexing="ij")
    X = torch.zeros(N * N, 4)
    X[:, 0] = tt.reshape(-1)
    X[:, 3] = zz.reshape(-1)

    loader = DataLoader(TensorDataset(X), batch_size=len(X), shuffle=False)
    surrogate = CurlSurrogate()

    cfg = VectorDESearchConfig(
        x_axis=0,                    # t is evolution axis
        order_candidates=(1,),       # first-order
        out_idxs=(0, 1, 2),         # E components are the LHS
        include_const=False,
        stlsq_lambda=1e-4,
    )

    result = discover_vector_de_from_surrogate(
        surrogate, loader, cfg=cfg,
        vector_terms=[tuple(curl_B)],  # single vector term: curl(B)
        device=torch.device("cpu"),
    )

    print("Order:", result.order)
    print("RMS train:", result.rms_train)
    print("Coefficients:", result.coeffs)
    print("System:\n", result.format_system())

    assert result.order == 1, f"Expected order 1, got {result.order}"

    for i, rms in enumerate(result.rms_train):
        print(f"  comp{i} RMS = {rms:.2e}")
        assert rms < 1e-6, f"RMS too large for comp{i}: {rms}"

    # Coefficient should be ≈ -1 (since ∂E/∂t + coeff·curl(B) = 0 → coeff = -1)
    c = result.coeffs.tolist()
    print(f"  coeffs: {c}")
    found = False
    for v in (c if isinstance(c, list) else [c]):
        if abs(abs(v) - 1.0) < 0.1:
            found = True
    assert found, f"Expected coefficient ≈ ±1, got {c}"

    print("\nPASSED: Vector curl equation ∂E/∂t = curl(B) discovered correctly.")


if __name__ == "__main__":
    test_vector_curl_equation()
