#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Discover source-driven Maxwell equations in a wire-inspired toy setting.

Dataset convention:
    outputs [Ex,Ey,Ez,Bx,By,Bz,Jx,Jy,Jz]

Target equations:
    dE/dt - curl(B) + J = 0
    dB/dt + curl(E)     = 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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
from tabulated_surrogate import TabulatedVectorSurrogate

torch.set_default_dtype(torch.float64)


def _vec_key(vec: Sequence[Node]) -> str:
    return "|".join(repr(c) for c in vec)


def _print_selected_terms(result, term_name_by_key: dict[str, str]) -> dict[str, int]:
    print("\nSelected term columns:")
    selected: dict[str, int] = {}
    for j, tvec in enumerate(result.term_vecs):
        key = _vec_key(tvec)
        label = term_name_by_key.get(key, f"term_{j}")
        selected[key] = j
        print(f"  [{j}] {label}")
    return selected


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_data = repo_root / "data" / "maxwell" / "fake_maxwell_wire_source.npz"

    parser = argparse.ArgumentParser(
        description="Discover source-driven Maxwell equations from wire toy data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=default_data, help="Input .npz path")
    parser.add_argument("--batch_size", type=int, default=0, help="0 means full batch")
    parser.add_argument("--max_points", type=int, default=20000, help="Discovery max_points")
    parser.add_argument("--stlsq_lambda", type=float, default=5e-4, help="STLSQ threshold")
    parser.add_argument("--rms_tol", type=float, default=2e-6, help="RMS assertion tolerance")
    parser.add_argument("--gauss_tol", type=float, default=1e-10, help="Gauss-law RMS tolerance")
    parser.add_argument("--coeff_tol", type=float, default=1e-1, help="Tolerance for expected coefficients")
    parser.add_argument("--decoy_tol", type=float, default=1e-1, help="Tolerance for decoy coefficients")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            f"Generate it first with: python {script_dir / 'generate_fake_maxwell_wire_data.py'}"
        )

    blob = np.load(args.data)
    X_np = np.asarray(blob["X"], dtype=np.float64)
    Y_np = np.asarray(blob["Y"], dtype=np.float64)
    G_np = np.asarray(blob["G"], dtype=np.float64)

    div_e = G_np[:, 0, 1] + G_np[:, 1, 2] + G_np[:, 2, 3]
    div_b = G_np[:, 3, 1] + G_np[:, 4, 2] + G_np[:, 5, 3]
    div_e_rms = float(np.sqrt(np.mean(div_e * div_e)))
    div_b_rms = float(np.sqrt(np.mean(div_b * div_b)))

    X = torch.from_numpy(X_np).to(dtype=torch.float64)
    Y = torch.from_numpy(Y_np).to(dtype=torch.float64)
    G = torch.from_numpy(G_np).to(dtype=torch.float64)

    batch_size = int(X.shape[0]) if int(args.batch_size) <= 0 else int(args.batch_size)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    surrogate = TabulatedVectorSurrogate(X, Y, G)

    E = VField("E", base_out_idx=0, n_comp=3, comp_names=("x", "y", "z"))
    B = VField("B", base_out_idx=3, n_comp=3, comp_names=("x", "y", "z"))
    J = VField("J", base_out_idx=6, n_comp=3, comp_names=("x", "y", "z"))
    spatial = (1, 2, 3)

    curl_B = tuple(curl(B, spatial_axes=spatial))
    curl_E = tuple(curl(E, spatial_axes=spatial))
    J_vec = (J("x"), J("y"), J("z"))
    E_vec = (E("x"), E("y"), E("z"))
    B_vec = (B("x"), B("y"), B("z"))
    vector_terms = [curl_B, curl_E, J_vec, E_vec, B_vec]

    equations = [
        VectorEquationSpec(out_idxs=(0, 1, 2), name="AmpereSource"),
        VectorEquationSpec(out_idxs=(3, 4, 5), name="Faraday"),
    ]
    cfg = VectorSystemDESearchConfig(
        x_axis=0,
        order_candidates=(1,),
        include_const=False,
        stlsq_lambda=float(args.stlsq_lambda),
        stlsq_max_iter=20,
        sparsity_penalty=1e-6,
        share_support_across_equations=False,
        max_points=int(args.max_points),
    )

    result = discover_vector_system_de_from_surrogate(
        surrogate,
        loader,
        cfg=cfg,
        equations=equations,
        vector_terms=vector_terms,
        device=torch.device("cpu"),
    )

    print("=" * 86)
    print("Maxwell discovery in wire-source toy setting")
    print("=" * 86)
    print(f"Data file:      {args.data}")
    print(f"N points:       {X.shape[0]}")
    print(f"Gauss RMS:      div(E)={div_e_rms:.3e}, div(B)={div_b_rms:.3e}")
    print(f"Order:          {result.order}")
    print("RMS train:")
    for q, eq_rms in enumerate(result.rms_train):
        print(f"  eq{q}: {', '.join(f'{v:.3e}' for v in eq_rms)}")
    print("\nRecovered system:")
    print(result.format_system())

    term_name_by_key = {
        _vec_key(curl_B): "curl(B)",
        _vec_key(curl_E): "curl(E)",
        _vec_key(J_vec): "J",
        _vec_key(E_vec): "E",
        _vec_key(B_vec): "B",
    }
    selected = _print_selected_terms(result, term_name_by_key)

    # Assertions.
    assert result.order == 1, f"Expected order=1, got {result.order}"
    assert div_e_rms < float(args.gauss_tol), f"div(E) RMS too large: {div_e_rms}"
    assert div_b_rms < float(args.gauss_tol), f"div(B) RMS too large: {div_b_rms}"
    for q, eq_rms in enumerate(result.rms_train):
        for ci, rms in enumerate(eq_rms):
            assert rms < float(args.rms_tol), f"eq{q} comp{ci}: RMS too large ({rms})"

    key_curl_b = _vec_key(curl_B)
    key_curl_e = _vec_key(curl_E)
    key_j = _vec_key(J_vec)
    assert key_curl_b in selected, "curl(B) term was not selected"
    assert key_curl_e in selected, "curl(E) term was not selected"
    assert key_j in selected, "J term was not selected"

    j_b = selected[key_curl_b]
    j_e = selected[key_curl_e]
    j_j = selected[key_j]
    c = result.coeffs

    c_amp_b = float(c[0, j_b].item())
    c_amp_e = float(c[0, j_e].item())
    c_amp_j = float(c[0, j_j].item())
    c_far_b = float(c[1, j_b].item())
    c_far_e = float(c[1, j_e].item())
    c_far_j = float(c[1, j_j].item())

    print("\nKey coefficients:")
    print(f"  AmpereSource coeff(curl(B)) = {c_amp_b:+.6f}  (expected -1)")
    print(f"  AmpereSource coeff(curl(E)) = {c_amp_e:+.6f}  (expected ~0)")
    print(f"  AmpereSource coeff(J)       = {c_amp_j:+.6f}  (expected +1)")
    print(f"  Faraday      coeff(curl(B)) = {c_far_b:+.6f}  (expected ~0)")
    print(f"  Faraday      coeff(curl(E)) = {c_far_e:+.6f}  (expected +1)")
    print(f"  Faraday      coeff(J)       = {c_far_j:+.6f}  (expected ~0)")

    coeff_tol = float(args.coeff_tol)
    assert abs(c_amp_b + 1.0) < coeff_tol, f"Ampere coeff(curl(B)) expected -1, got {c_amp_b}"
    assert abs(c_amp_j - 1.0) < coeff_tol, f"Ampere coeff(J) expected +1, got {c_amp_j}"
    assert abs(c_far_e - 1.0) < coeff_tol, f"Faraday coeff(curl(E)) expected +1, got {c_far_e}"
    assert abs(c_amp_e) < coeff_tol, f"Ampere coeff(curl(E)) should be near 0, got {c_amp_e}"
    assert abs(c_far_b) < coeff_tol, f"Faraday coeff(curl(B)) should be near 0, got {c_far_b}"
    assert abs(c_far_j) < coeff_tol, f"Faraday coeff(J) should be near 0, got {c_far_j}"

    for decoy_name, decoy_key in (
        ("E", _vec_key(E_vec)),
        ("B", _vec_key(B_vec)),
    ):
        if decoy_key not in selected:
            print(f"  Decoy {decoy_name}: not selected")
            continue
        j = selected[decoy_key]
        c0 = float(c[0, j].item())
        c1 = float(c[1, j].item())
        print(f"  Decoy {decoy_name}: eq0={c0:+.3e}, eq1={c1:+.3e}")
        assert abs(c0) < float(args.decoy_tol) and abs(c1) < float(args.decoy_tol), (
            f"Decoy {decoy_name} should be near zero"
        )

    print("\nPASSED: Recovered source-driven Maxwell toy equations.")
    print("=" * 86)


if __name__ == "__main__":
    main()
