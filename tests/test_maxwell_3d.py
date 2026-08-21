# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Test: discover full 3D Maxwell system (both Faraday + Ampère) using VectorSystemDESearchConfig.

Physics (vacuum, c=1):
    ∂E/∂t = +curl(B)    →  ∂E/∂t + (-1)·curl(B) = 0    (Ampère)
    ∂B/∂t = -curl(E)    →  ∂B/∂t + (+1)·curl(E) = 0    (Faraday)

Same plane-wave solution as test_vector_curl_equation:
    E = (sin(z-t), 0, 0),  B = (0, sin(z-t), 0)

Surrogate: Ny=6, axes (t,x,y,z) = (0,1,2,3).
"""

import math
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_dtype(torch.float64)


# ── Mock surrogate (same as curl test) ────────────────────
class MaxwellSurrogate(torch.nn.Module):
    """Ny=6: (Ex,Ey,Ez,Bx,By,Bz), plane wave E=(sin(z-t),0,0), B=(0,sin(z-t),0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, z = x[:, 0], x[:, 3]
        phase = z - t
        s = torch.sin(phase)
        zero = torch.zeros_like(s)
        return torch.stack([s, zero, zero, zero, s, zero], dim=1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t, z = x[:, 0], x[:, 3]
        phase = z - t
        c = torch.cos(phase)
        G = torch.zeros(B, 6, 4, dtype=x.dtype, device=x.device)
        G[:, 0, 0] = -c   # ∂Ex/∂t
        G[:, 0, 3] = c    # ∂Ex/∂z
        G[:, 4, 0] = -c   # ∂By/∂t
        G[:, 4, 3] = c    # ∂By/∂z
        return G

    def parameters(self):
        return iter([torch.zeros(1)])


# ── Test ──────────────────────────────────────────────────
def test_maxwell_3d_system():
    from nestynet_sr.sr_de.system_de_search import (
        discover_vector_system_de_from_surrogate,
        VectorSystemDESearchConfig,
        VectorEquationSpec,
    )
    from nestynet_sr.sr_core.bridges import VField
    from nestynet_sr.sr_de.vector_ops import curl

    # Define fields
    E = VField("E", base_out_idx=0, n_comp=3, comp_names=("x", "y", "z"))
    B = VField("B", base_out_idx=3, n_comp=3, comp_names=("x", "y", "z"))

    spatial = (1, 2, 3)
    curl_B = curl(B, spatial_axes=spatial)  # Vec of 3 Nodes
    curl_E = curl(E, spatial_axes=spatial)  # Vec of 3 Nodes

    # Two vector equations:
    #   eq0 (Ampère):  ∂E/∂t + c0·curl(B) = 0   → c0 ≈ -1
    #   eq1 (Faraday): ∂B/∂t + c1·curl(E) = 0   → c1 ≈ +1
    equations = [
        VectorEquationSpec(out_idxs=(0, 1, 2), name="Ampere"),
        VectorEquationSpec(out_idxs=(3, 4, 5), name="Faraday"),
    ]

    # Each equation gets both curl terms in the library
    vector_terms = [tuple(curl_B), tuple(curl_E)]

    # Grid data
    N = 40
    t_vals = torch.linspace(0.0, 2 * math.pi, N)
    z_vals = torch.linspace(0.0, 2 * math.pi, N)
    tt, zz = torch.meshgrid(t_vals, z_vals, indexing="ij")
    X = torch.zeros(N * N, 4)
    X[:, 0] = tt.reshape(-1)
    X[:, 3] = zz.reshape(-1)

    loader = DataLoader(TensorDataset(X), batch_size=len(X), shuffle=False)
    surrogate = MaxwellSurrogate()

    cfg = VectorSystemDESearchConfig(
        x_axis=0,
        order_candidates=(1,),
        include_const=False,
        stlsq_lambda=1e-4,
    )

    result = discover_vector_system_de_from_surrogate(
        surrogate, loader, cfg=cfg,
        equations=equations,
        vector_terms=vector_terms,
        device=torch.device("cpu"),
    )

    print("Order:", result.order)
    print("Coefficients:\n", result.coeffs)
    print("System:\n", result.format_system())

    assert result.order == 1, f"Expected order 1, got {result.order}"

    # Check RMS per equation
    for q, eq_rms in enumerate(result.rms_train):
        for i, rms in enumerate(eq_rms):
            print(f"  eq{q} comp{i} RMS = {rms:.2e}")
            assert rms < 1e-5, f"RMS too large for eq{q} comp{i}: {rms}"

    # coeffs shape: (2, K_sel) — one row per equation
    print(f"  coeffs shape: {result.coeffs.shape}")

    # Verify exact term identity and coefficient signs:
    # eq0 (Ampere):  dE/dt - curl(B) = 0  -> coeff(curl(B)) ≈ -1, coeff(curl(E)) ≈ 0
    # eq1 (Faraday): dB/dt + curl(E) = 0  -> coeff(curl(E)) ≈ +1, coeff(curl(B)) ≈ 0
    def vec_key(vec):
        return "|".join(repr(c) for c in vec)

    key_curl_b = vec_key(tuple(curl_B))
    key_curl_e = vec_key(tuple(curl_E))
    selected = {vec_key(t): j for j, t in enumerate(result.term_vecs)}
    print("  selected terms:")
    for j, t in enumerate(result.term_vecs):
        label = "curl(B)" if vec_key(t) == key_curl_b else "curl(E)"
        print(f"    [{j}] {label}")

    assert key_curl_b in selected, "curl(B) not selected"
    assert key_curl_e in selected, "curl(E) not selected"

    j_b = selected[key_curl_b]
    j_e = selected[key_curl_e]
    c_amp_b = float(result.coeffs[0, j_b].item())
    c_amp_e = float(result.coeffs[0, j_e].item())
    c_far_b = float(result.coeffs[1, j_b].item())
    c_far_e = float(result.coeffs[1, j_e].item())

    print(f"  Ampere:  coeff(curl(B))={c_amp_b:+.6f}, coeff(curl(E))={c_amp_e:+.6f}")
    print(f"  Faraday: coeff(curl(B))={c_far_b:+.6f}, coeff(curl(E))={c_far_e:+.6f}")

    tol = 5e-2
    assert abs(c_amp_b + 1.0) < tol, f"Ampere coeff(curl(B)) expected -1, got {c_amp_b}"
    assert abs(c_far_e - 1.0) < tol, f"Faraday coeff(curl(E)) expected +1, got {c_far_e}"
    assert abs(c_amp_e) < tol, f"Ampere coeff(curl(E)) should be near 0, got {c_amp_e}"
    assert abs(c_far_b) < tol, f"Faraday coeff(curl(B)) should be near 0, got {c_far_b}"

    print("\nPASSED: Full 3D Maxwell system discovered correctly.")


if __name__ == "__main__":
    test_maxwell_3d_system()
