#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Discover Maxwell equations from tabulated synthetic field data.

Workflow:
1. Load fake field/gradient tables from generate_fake_maxwell_data.py.
2. Build a lookup surrogate that exposes forward() and grad().
3. Run coupled vector-system discovery for:
      dE/dt + c0*curl(B) + ... = 0
      dB/dt + c1*curl(E) + ... = 0
4. Verify c0 ~= -1 and c1 ~= +1.
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
        if tvec is None:
            label = "const"
            key = "const"
        else:
            key = _vec_key(tvec)
            label = term_name_by_key.get(key, f"term_{j}")
        selected[key] = j
        print(f"  [{j}] {label}")
    return selected


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_data = repo_root / "data" / "maxwell" / "fake_maxwell_plane_wave.npz"

    parser = argparse.ArgumentParser(
        description="Discover coupled Maxwell equations from synthetic tabulated data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=default_data, help="Input .npz path")
    parser.add_argument("--batch_size", type=int, default=0, help="0 means full batch")
    parser.add_argument("--stlsq_lambda", type=float, default=1e-4, help="STLSQ threshold")
    parser.add_argument("--rms_tol", type=float, default=1e-6, help="RMS assertion tolerance")
    parser.add_argument(
        "--coeff_tol",
        type=float,
        default=5e-2,
        help="Tolerance for expected +-1 curl coefficients",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            f"Generate it first with: python {script_dir / 'generate_fake_maxwell_data.py'}"
        )

    blob = np.load(args.data)
    X = torch.from_numpy(blob["X"]).to(dtype=torch.float64)
    Y = torch.from_numpy(blob["Y"]).to(dtype=torch.float64)
    G = torch.from_numpy(blob["G"]).to(dtype=torch.float64)

    batch_size = int(X.shape[0]) if int(args.batch_size) <= 0 else int(args.batch_size)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    surrogate = TabulatedVectorSurrogate(X, Y, G)

    E = VField("E", base_out_idx=0, n_comp=3, comp_names=("x", "y", "z"))
    B = VField("B", base_out_idx=3, n_comp=3, comp_names=("x", "y", "z"))
    spatial = (1, 2, 3)

    curl_B = tuple(curl(B, spatial_axes=spatial))
    curl_E = tuple(curl(E, spatial_axes=spatial))
    E_vec = (E("x"), E("y"), E("z"))  # decoy term
    B_vec = (B("x"), B("y"), B("z"))  # decoy term
    vector_terms = [curl_B, curl_E, E_vec, B_vec]

    equations = [
        VectorEquationSpec(out_idxs=(0, 1, 2), name="Ampere"),
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
    )

    result = discover_vector_system_de_from_surrogate(
        surrogate,
        loader,
        cfg=cfg,
        equations=equations,
        vector_terms=vector_terms,
        device=torch.device("cpu"),
    )

    print("=" * 72)
    print("Maxwell discovery from fake data")
    print("=" * 72)
    print(f"Data file:    {args.data}")
    print(f"N points:     {X.shape[0]}")
    print(f"Order:        {result.order}")
    print("RMS train:")
    for q, eq_rms in enumerate(result.rms_train):
        print(f"  eq{q}: {', '.join(f'{v:.3e}' for v in eq_rms)}")
    print("\nRecovered system:")
    print(result.format_system())

    term_name_by_key = {
        _vec_key(curl_B): "curl(B)",
        _vec_key(curl_E): "curl(E)",
        _vec_key(E_vec): "E",
        _vec_key(B_vec): "B",
    }
    selected = _print_selected_terms(result, term_name_by_key)

    # Hard checks.
    assert result.order == 1, f"Expected order=1, got {result.order}"
    for q, eq_rms in enumerate(result.rms_train):
        for ci, rms in enumerate(eq_rms):
            assert rms < float(args.rms_tol), f"eq{q} comp{ci}: RMS too large ({rms})"

    key_curl_b = _vec_key(curl_B)
    key_curl_e = _vec_key(curl_E)
    assert key_curl_b in selected, "curl(B) term was not selected"
    assert key_curl_e in selected, "curl(E) term was not selected"

    j_curl_b = selected[key_curl_b]
    j_curl_e = selected[key_curl_e]

    coeffs = result.coeffs
    c_ampere_curl_b = float(coeffs[0, j_curl_b].item())
    c_ampere_curl_e = float(coeffs[0, j_curl_e].item())
    c_faraday_curl_b = float(coeffs[1, j_curl_b].item())
    c_faraday_curl_e = float(coeffs[1, j_curl_e].item())

    print("\nKey coefficients:")
    print(f"  Ampere  coeff(curl(B)) = {c_ampere_curl_b:+.6f}  (expected -1)")
    print(f"  Ampere  coeff(curl(E)) = {c_ampere_curl_e:+.6f}  (expected ~0)")
    print(f"  Faraday coeff(curl(B)) = {c_faraday_curl_b:+.6f}  (expected ~0)")
    print(f"  Faraday coeff(curl(E)) = {c_faraday_curl_e:+.6f}  (expected +1)")

    tol = float(args.coeff_tol)
    assert abs(c_ampere_curl_b + 1.0) < tol, (
        f"Ampere curl(B) coefficient expected -1, got {c_ampere_curl_b}"
    )
    assert abs(c_faraday_curl_e - 1.0) < tol, (
        f"Faraday curl(E) coefficient expected +1, got {c_faraday_curl_e}"
    )
    assert abs(c_ampere_curl_e) < tol, f"Ampere curl(E) should be near 0, got {c_ampere_curl_e}"
    assert abs(c_faraday_curl_b) < tol, f"Faraday curl(B) should be near 0, got {c_faraday_curl_b}"

    for decoy_name, decoy_key in (("E", _vec_key(E_vec)), ("B", _vec_key(B_vec))):
        if decoy_key not in selected:
            continue
        j = selected[decoy_key]
        c0 = float(coeffs[0, j].item())
        c1 = float(coeffs[1, j].item())
        print(f"  Decoy {decoy_name}: eq0={c0:+.3e}, eq1={c1:+.3e}")
        assert abs(c0) < tol and abs(c1) < tol, f"Decoy {decoy_name} should be near zero"

    print("\nPASSED: Recovered Maxwell curl equations from fake field data.")
    print("=" * 72)


if __name__ == "__main__":
    main()
