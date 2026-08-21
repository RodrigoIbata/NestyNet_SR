#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Plot synthetic Maxwell fields and recovered-equation residuals.

This script:
1) loads fake tabulated fields/gradients from generate_fake_maxwell_data.py,
2) reruns vector-system discovery to obtain Maxwell coefficients,
3) plots field maps and residual maps from the recovered equations.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Sequence

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import Node, VField
from nestynet_sr.sr_de.system_de_search import (
    VectorEquationSpec,
    VectorSystemDESearchConfig,
    discover_vector_system_de_from_surrogate,
)
from nestynet_sr.sr_de.vector_ops import curl

torch.set_default_dtype(torch.float64)


def _vec_key(vec: Sequence[Node]) -> str:
    return "|".join(repr(c) for c in vec)


def _sym_vmax(a: np.ndarray, floor: float = 1e-12) -> float:
    return float(max(np.max(np.abs(a)), floor))


class TabulatedVectorSurrogate(torch.nn.Module):
    """Lookup surrogate built from tabulated coordinates/fields/gradients."""

    def __init__(
        self,
        x_table: torch.Tensor,
        y_table: torch.Tensor,
        g_table: torch.Tensor,
        *,
        coord_decimals: int = 12,
    ) -> None:
        super().__init__()
        self.register_buffer("x_table", x_table.contiguous())
        self.register_buffer("y_table", y_table.contiguous())
        self.register_buffer("g_table", g_table.contiguous())
        self._dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=x_table.dtype), requires_grad=False)
        self.coord_decimals = int(coord_decimals)
        self.coord_scale = float(10 ** self.coord_decimals)
        self._index = {
            self._coord_key(self.x_table[i]): int(i) for i in range(int(self.x_table.shape[0]))
        }

    def _coord_key(self, row: torch.Tensor) -> tuple[int, ...]:
        vals = row.detach().cpu().tolist()
        return tuple(int(round(float(v) * self.coord_scale)) for v in vals)

    def _lookup_indices(self, x: torch.Tensor) -> torch.Tensor:
        idx = []
        for r in x:
            key = self._coord_key(r)
            j = self._index.get(key, None)
            if j is None:
                raise KeyError("Query point not found in lookup table.")
            idx.append(j)
        return torch.tensor(idx, dtype=torch.long, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self._lookup_indices(x)
        return self.y_table[idx]

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        idx = self._lookup_indices(x)
        return self.g_table[idx]


def _curl_from_grad(g: np.ndarray, base_idx: int) -> np.ndarray:
    """Compute curl of a 3-component field from gradient table G[N,6,4]."""
    ix = base_idx + 0
    iy = base_idx + 1
    iz = base_idx + 2
    out = np.zeros((g.shape[0], 3), dtype=np.float64)
    # (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
    out[:, 0] = g[:, iz, 2] - g[:, iy, 3]
    out[:, 1] = g[:, ix, 3] - g[:, iz, 1]
    out[:, 2] = g[:, iy, 1] - g[:, ix, 2]
    return out


def _run_discovery_and_residuals(
    x: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    *,
    stlsq_lambda: float,
) -> tuple[object, np.ndarray, np.ndarray, dict[str, float]]:
    """Run Maxwell discovery and return residual vectors for both equations."""
    x_t = torch.from_numpy(x).to(dtype=torch.float64)
    y_t = torch.from_numpy(y).to(dtype=torch.float64)
    g_t = torch.from_numpy(g).to(dtype=torch.float64)

    loader = DataLoader(TensorDataset(x_t), batch_size=int(x_t.shape[0]), shuffle=False)
    surrogate = TabulatedVectorSurrogate(x_t, y_t, g_t)

    E = VField("E", base_out_idx=0, n_comp=3, comp_names=("x", "y", "z"))
    B = VField("B", base_out_idx=3, n_comp=3, comp_names=("x", "y", "z"))
    spatial = (1, 2, 3)

    curl_B_term = tuple(curl(B, spatial_axes=spatial))
    curl_E_term = tuple(curl(E, spatial_axes=spatial))
    E_term = (E("x"), E("y"), E("z"))
    B_term = (B("x"), B("y"), B("z"))

    vector_terms = [curl_B_term, curl_E_term, E_term, B_term]
    equations = [
        VectorEquationSpec(out_idxs=(0, 1, 2), name="Ampere"),
        VectorEquationSpec(out_idxs=(3, 4, 5), name="Faraday"),
    ]
    cfg = VectorSystemDESearchConfig(
        x_axis=0,
        order_candidates=(1,),
        include_const=False,
        stlsq_lambda=float(stlsq_lambda),
        stlsq_max_iter=20,
        sparsity_penalty=1e-6,
        share_support_across_equations=False,
    )

    result = discover_vector_system_de_from_surrogate(
        surrogate,
        loader,
        cfg=cfg,
        equations=equations,
        vector_terms=vector_terms,
        device=torch.device("cpu"),
    )

    # Numerical term arrays.
    dE_dt = g[:, 0:3, 0]
    dB_dt = g[:, 3:6, 0]
    curl_B = _curl_from_grad(g, base_idx=3)
    curl_E = _curl_from_grad(g, base_idx=0)
    E_vec = y[:, 0:3]
    B_vec = y[:, 3:6]

    key_curl_b = _vec_key(curl_B_term)
    key_curl_e = _vec_key(curl_E_term)
    key_e = _vec_key(E_term)
    key_b = _vec_key(B_term)

    term_arrays = {
        key_curl_b: curl_B,
        key_curl_e: curl_E,
        key_e: E_vec,
        key_b: B_vec,
    }

    res_amp = dE_dt.copy()
    res_far = dB_dt.copy()
    coeff_report = {
        "ampere_curl_b": 0.0,
        "ampere_curl_e": 0.0,
        "faraday_curl_b": 0.0,
        "faraday_curl_e": 0.0,
    }

    for j, tvec in enumerate(result.term_vecs):
        key = _vec_key(tvec)
        arr = term_arrays.get(key, None)
        if arr is None:
            continue
        c0 = float(result.coeffs[0, j].item())
        c1 = float(result.coeffs[1, j].item())
        res_amp += c0 * arr
        res_far += c1 * arr

        if key == key_curl_b:
            coeff_report["ampere_curl_b"] = c0
            coeff_report["faraday_curl_b"] = c1
        elif key == key_curl_e:
            coeff_report["ampere_curl_e"] = c0
            coeff_report["faraday_curl_e"] = c1

    return result, res_amp, res_far, coeff_report


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_data = repo_root / "data" / "maxwell" / "fake_maxwell_plane_wave.npz"
    default_output = script_dir / "maxwell_fields_and_residuals.png"

    parser = argparse.ArgumentParser(
        description="Plot Maxwell fake fields and discovered residual maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=default_data, help="Input .npz data file")
    parser.add_argument("--output", type=Path, default=default_output, help="Output figure path")
    parser.add_argument("--stlsq_lambda", type=float, default=1e-4, help="STLSQ threshold")
    parser.add_argument("--show", action="store_true", help="Show figure window")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            f"Generate it first with: python {script_dir / 'generate_fake_maxwell_data.py'}"
        )

    blob = np.load(args.data)
    X = np.asarray(blob["X"], dtype=np.float64)
    Y = np.asarray(blob["Y"], dtype=np.float64)
    G = np.asarray(blob["G"], dtype=np.float64)

    t_vals = np.asarray(blob["t_vals"], dtype=np.float64) if "t_vals" in blob else np.unique(X[:, 0])
    z_vals = np.asarray(blob["z_vals"], dtype=np.float64) if "z_vals" in blob else np.unique(X[:, 3])
    nt, nz = int(len(t_vals)), int(len(z_vals))
    if X.shape[0] != nt * nz:
        raise ValueError("Grid shape mismatch: expected nt*nz rows.")

    Yg = Y.reshape(nt, nz, 6)
    Gg = G.reshape(nt, nz, 6, 4)

    result, res_amp, res_far, c_report = _run_discovery_and_residuals(
        X, Y, G, stlsq_lambda=float(args.stlsq_lambda)
    )
    res_amp_g = res_amp.reshape(nt, nz, 3)
    res_far_g = res_far.reshape(nt, nz, 3)

    # Fields and derivatives used for visualization.
    Ex = Yg[:, :, 0]
    By = Yg[:, :, 4]
    dEx_dt = Gg[:, :, 0, 0]
    dBy_dt = Gg[:, :, 4, 0]
    curl_Bx = Gg[:, :, 5, 2] - Gg[:, :, 4, 3]
    curl_Ey = Gg[:, :, 0, 3] - Gg[:, :, 2, 1]

    extent = [float(z_vals[0]), float(z_vals[-1]), float(t_vals[0]), float(t_vals[-1])]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    cmap_field = "viridis"
    cmap_bal = "RdBu_r"

    im = axes[0, 0].imshow(Ex, origin="lower", aspect="auto", extent=extent, cmap=cmap_field)
    axes[0, 0].set_title("Field: Ex(t,z)")
    axes[0, 0].set_xlabel("z")
    axes[0, 0].set_ylabel("t")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.9)

    im = axes[0, 1].imshow(By, origin="lower", aspect="auto", extent=extent, cmap=cmap_field)
    axes[0, 1].set_title("Field: By(t,z)")
    axes[0, 1].set_xlabel("z")
    axes[0, 1].set_ylabel("t")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.9)

    tidx = nt // 4
    axes[0, 2].plot(z_vals, Ex[tidx], label="Ex(t_slice, z)", lw=2)
    axes[0, 2].plot(z_vals, By[tidx], label="By(t_slice, z)", lw=2, ls="--")
    axes[0, 2].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0, 2].set_title(f"Wave Slice (t={t_vals[tidx]:.3f})")
    axes[0, 2].set_xlabel("z")
    axes[0, 2].set_ylabel("amplitude")
    axes[0, 2].grid(True, alpha=0.25)
    axes[0, 2].legend(loc="upper right", fontsize=9)

    vmax = _sym_vmax(dEx_dt)
    im = axes[1, 0].imshow(
        dEx_dt, origin="lower", aspect="auto", extent=extent, cmap=cmap_bal, vmin=-vmax, vmax=vmax
    )
    axes[1, 0].set_title("Anchor: dEx/dt")
    axes[1, 0].set_xlabel("z")
    axes[1, 0].set_ylabel("t")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.9)

    vmax = _sym_vmax(curl_Bx)
    im = axes[1, 1].imshow(
        curl_Bx, origin="lower", aspect="auto", extent=extent, cmap=cmap_bal, vmin=-vmax, vmax=vmax
    )
    axes[1, 1].set_title("Term: curl(B)_x")
    axes[1, 1].set_xlabel("z")
    axes[1, 1].set_ylabel("t")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.9)

    amp_x_res = res_amp_g[:, :, 0]
    vmax = _sym_vmax(amp_x_res)
    im = axes[1, 2].imshow(
        amp_x_res, origin="lower", aspect="auto", extent=extent, cmap=cmap_bal, vmin=-vmax, vmax=vmax
    )
    axes[1, 2].set_title("Residual: Ampere (x-comp)")
    axes[1, 2].set_xlabel("z")
    axes[1, 2].set_ylabel("t")
    fig.colorbar(im, ax=axes[1, 2], shrink=0.9)

    vmax = _sym_vmax(dBy_dt)
    im = axes[2, 0].imshow(
        dBy_dt, origin="lower", aspect="auto", extent=extent, cmap=cmap_bal, vmin=-vmax, vmax=vmax
    )
    axes[2, 0].set_title("Anchor: dBy/dt")
    axes[2, 0].set_xlabel("z")
    axes[2, 0].set_ylabel("t")
    fig.colorbar(im, ax=axes[2, 0], shrink=0.9)

    neg_curl_Ey = -curl_Ey
    vmax = _sym_vmax(neg_curl_Ey)
    im = axes[2, 1].imshow(
        neg_curl_Ey, origin="lower", aspect="auto", extent=extent, cmap=cmap_bal, vmin=-vmax, vmax=vmax
    )
    axes[2, 1].set_title("Target match: -curl(E)_y")
    axes[2, 1].set_xlabel("z")
    axes[2, 1].set_ylabel("t")
    fig.colorbar(im, ax=axes[2, 1], shrink=0.9)

    far_y_res = res_far_g[:, :, 1]
    vmax = _sym_vmax(far_y_res)
    im = axes[2, 2].imshow(
        far_y_res, origin="lower", aspect="auto", extent=extent, cmap=cmap_bal, vmin=-vmax, vmax=vmax
    )
    axes[2, 2].set_title("Residual: Faraday (y-comp)")
    axes[2, 2].set_xlabel("z")
    axes[2, 2].set_ylabel("t")
    fig.colorbar(im, ax=axes[2, 2], shrink=0.9)

    rms_amp = np.sqrt(np.mean(res_amp ** 2, axis=0))
    rms_far = np.sqrt(np.mean(res_far ** 2, axis=0))
    coeff_line = (
        f"Ampere: coeff(curl(B))={c_report['ampere_curl_b']:+.4f}, "
        f"coeff(curl(E))={c_report['ampere_curl_e']:+.4f} | "
        f"Faraday: coeff(curl(B))={c_report['faraday_curl_b']:+.4f}, "
        f"coeff(curl(E))={c_report['faraday_curl_e']:+.4f}"
    )
    fig.suptitle(
        "Maxwell Synthetic Fields and Discovery Residuals\n"
        + coeff_line
        + f"\nRMS Ampere={rms_amp.tolist()}  RMS Faraday={rms_far.tolist()}",
        fontsize=12,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved figure: {args.output}")
    print("Recovered system:\n" + result.format_system())
    print(f"Ampere RMS by component: {rms_amp}")
    print(f"Faraday RMS by component: {rms_far}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
