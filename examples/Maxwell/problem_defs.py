#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Maxwell benchmark problem definitions and ground truth.

Defines the 3 initial Maxwell problems (vacuum, wire, conductive) with
structured metadata, expected coefficients, data generators, and term builders.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import Node, VField
from nestynet_sr.sr_de.system_de_search import VectorEquationSpec
from nestynet_sr.sr_de.vector_ops import curl, laplacian

# ---------------------------------------------------------------------------
# Ensure the examples/Maxwell directory is importable for the generate scripts
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VectorProblemDef:
    """One Maxwell vector-PDE benchmark problem."""

    id: str  # "mw000", "mw001", "mw002"
    description: str
    field_names: list[str]
    n_fields: int
    equations: list[VectorEquationSpec]
    spatial_axes: tuple[int, ...]
    params: list[str] = field(default_factory=list)
    param_defaults: dict[str, float] = field(default_factory=dict)


@dataclass
class VectorGroundTruth:
    """Expected coefficients for one problem.

    ``expected_coeffs`` maps term_name -> {eq_idx: expected_value}.
    ``decoy_terms`` lists term names whose coefficients should be ~0.
    """

    order: int
    expected_coeffs: dict[str, dict[int, float]]
    decoy_terms: list[str]
    coeff_tol: float = 0.12
    decoy_tol: float = 0.12
    rms_tol: float = 3e-6
    gauss_tol: float = 1e-10


@dataclass
class ComponentGroundTruth:
    """Expected per-component ground truth for factorized symbolic search validation.

    ``expected_terms`` maps feature_name -> expected_coefficient.
    """

    target_name: str  # e.g., "dEx/dt"
    equation_name: str  # "Ampere" or "Faraday"
    component: str  # "x", "y", or "z"
    expected_terms: dict[str, float]
    mse_tol: float = 1e-6


# ---------------------------------------------------------------------------
# Helpers to make vec keys (matches discover scripts)
# ---------------------------------------------------------------------------


def _vec_key(vec: Sequence[Node]) -> str:
    return "|".join(repr(c) for c in vec)


# ---------------------------------------------------------------------------
# Term builders
# ---------------------------------------------------------------------------


def _add_laplacian_decoys(
    vector_terms: list[tuple[Node, ...]],
    name_by_key: dict[str, str],
    named_vecs: dict[str, tuple[Node, ...]],
    fields: dict[str, VField],
    spatial: tuple[int, ...],
) -> None:
    """Append vector Laplacian decoys (∇²E, ∇²B) to a term library, in place.

    These are second-order operators that compete with curl on the broadened
    menu — the discovery must reject them (coeff ~0) rather than echo back the
    operators it was handed.  Evaluating them requires the surrogate's analytic
    Hessian (``grad_grad``), so they are only included on the perfect-information
    (tabulated) path for now.
    """
    for label, fld in fields.items():
        lap = tuple(laplacian(fld, spatial_axes=spatial))
        name = f"laplacian({label})"
        vector_terms.append(lap)
        name_by_key[_vec_key(lap)] = name
        named_vecs[name] = lap


def build_vector_terms_mw000(*, include_laplacian: bool = False) -> tuple[
    list[tuple[Node, ...]], dict[str, str], dict[str, tuple[Node, ...]]
]:
    """Build terms for vacuum plane wave (E, B only)."""
    E = VField("E", base_out_idx=0, n_comp=3, comp_names=("x", "y", "z"))
    B = VField("B", base_out_idx=3, n_comp=3, comp_names=("x", "y", "z"))
    spatial = (1, 2, 3)

    curl_B = tuple(curl(B, spatial_axes=spatial))
    curl_E = tuple(curl(E, spatial_axes=spatial))
    E_vec = (E("x"), E("y"), E("z"))
    B_vec = (B("x"), B("y"), B("z"))
    vector_terms = [curl_B, curl_E, E_vec, B_vec]

    name_by_key = {
        _vec_key(curl_B): "curl(B)",
        _vec_key(curl_E): "curl(E)",
        _vec_key(E_vec): "E",
        _vec_key(B_vec): "B",
    }
    named_vecs = {
        "curl(B)": curl_B,
        "curl(E)": curl_E,
        "E": E_vec,
        "B": B_vec,
    }
    if include_laplacian:
        _add_laplacian_decoys(vector_terms, name_by_key, named_vecs, {"E": E, "B": B}, spatial)
    return vector_terms, name_by_key, named_vecs


def build_vector_terms_mw001(*, include_laplacian: bool = False) -> tuple[
    list[tuple[Node, ...]], dict[str, str], dict[str, tuple[Node, ...]]
]:
    """Build terms for wire source (E, B, J)."""
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

    name_by_key = {
        _vec_key(curl_B): "curl(B)",
        _vec_key(curl_E): "curl(E)",
        _vec_key(J_vec): "J",
        _vec_key(E_vec): "E",
        _vec_key(B_vec): "B",
    }
    named_vecs = {
        "curl(B)": curl_B,
        "curl(E)": curl_E,
        "J": J_vec,
        "E": E_vec,
        "B": B_vec,
    }
    if include_laplacian:
        _add_laplacian_decoys(vector_terms, name_by_key, named_vecs, {"E": E, "B": B}, spatial)
    return vector_terms, name_by_key, named_vecs


def build_vector_terms_mw002(*, include_laplacian: bool = False) -> tuple[
    list[tuple[Node, ...]], dict[str, str], dict[str, tuple[Node, ...]]
]:
    """Build terms for conductive medium (E, B with sigma*E)."""
    E = VField("E", base_out_idx=0, n_comp=3, comp_names=("x", "y", "z"))
    B = VField("B", base_out_idx=3, n_comp=3, comp_names=("x", "y", "z"))
    spatial = (1, 2, 3)

    curl_B = tuple(curl(B, spatial_axes=spatial))
    curl_E = tuple(curl(E, spatial_axes=spatial))
    E_vec = (E("x"), E("y"), E("z"))
    B_vec = (B("x"), B("y"), B("z"))
    vector_terms = [curl_B, curl_E, E_vec, B_vec]

    name_by_key = {
        _vec_key(curl_B): "curl(B)",
        _vec_key(curl_E): "curl(E)",
        _vec_key(E_vec): "E",
        _vec_key(B_vec): "B",
    }
    named_vecs = {
        "curl(B)": curl_B,
        "curl(E)": curl_E,
        "E": E_vec,
        "B": B_vec,
    }
    if include_laplacian:
        _add_laplacian_decoys(vector_terms, name_by_key, named_vecs, {"E": E, "B": B}, spatial)
    return vector_terms, name_by_key, named_vecs


_TERM_BUILDERS = {
    "mw000": build_vector_terms_mw000,
    "mw001": build_vector_terms_mw001,
    "mw002": build_vector_terms_mw002,
    # mw003 shares mw000's E/B field layout and candidate menu; only the data
    # (multi-mode vacuum) differs.
    "mw003": build_vector_terms_mw000,
}


def build_vector_terms(
    problem: VectorProblemDef,
    *,
    include_laplacian: bool = False,
) -> tuple[list[tuple[Node, ...]], dict[str, str], dict[str, tuple[Node, ...]]]:
    """Build candidate vector terms for a problem.

    Parameters
    ----------
    include_laplacian : bool
        If True, append the ∇²E / ∇²B second-order decoys to the menu (requires
        a surrogate that exposes ``grad_grad``).

    Returns:
        vector_terms: list of vector term tuples for the discovery engine
        name_by_key:  dict mapping vec_key -> human-readable name
        named_vecs:   dict mapping human-readable name -> vec tuple
    """
    return _TERM_BUILDERS[problem.id](include_laplacian=include_laplacian)


def maxwell_units_spec(problem: VectorProblemDef):
    """Declared-coefficient-basis UnitsSpec for a Maxwell problem (or None).

    Bookkeeping dimensions over base ``(L, T, Phi)`` with ``Phi`` the B-field
    dimension and the Faraday relation ``[E]=Phi*L/T``.  The declared free
    constants are the Maxwell coefficient scales only -- ``kappa_curl`` (the
    curl coefficient, ``c^2`` = ``L^2/T^2``) and, for the conductive case,
    ``sigma`` (``1/T``).  Crucially NO diffusivity (``L^2/T``) is declared, so
    under the engine's atomic (no product/ratio closure) dimension matching the
    second-order Laplacian decoys -- whose required coefficient is a diffusivity
    -- are pruned before regression.  Returns ``None`` for problems without a
    declared basis here (e.g. the wire case ``mw001`` with a source term).
    """
    from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec

    if problem.id not in ("mw000", "mw002", "mw003"):
        return None
    us = UnitSystem(base=("L", "T", "Phi"))
    Edim, Bdim = (1, -1, 1), (0, 0, 1)              # [E]=Phi*L/T, [B]=Phi
    x_dims = ((0, 1, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0))  # axes t, x, y, z
    output_dims = (Edim, Edim, Edim, Bdim, Bdim, Bdim)
    free_const_dims = {"kappa_curl": (2, -2, 0)}    # c^2  (curl scale)
    if problem.id == "mw002":
        free_const_dims["sigma"] = (0, -1, 0)        # conductivity (1/T)
    return UnitsSpec(
        unit_system=us, x_dims=x_dims, y_dim=Edim,
        output_dims=output_dims, free_const_dims=free_const_dims,
    )


# ---------------------------------------------------------------------------
# Data generators — delegate to existing build_*_dataset functions
# ---------------------------------------------------------------------------


def _import_plane_wave_builder():
    from generate_fake_maxwell_data import build_maxwell_plane_wave_dataset

    return build_maxwell_plane_wave_dataset


def _import_wire_builder():
    from generate_fake_maxwell_wire_data import build_wire_dataset

    return build_wire_dataset


def _import_conductive_builder():
    from generate_fake_maxwell_conductive_data import build_conductive_dataset

    return build_conductive_dataset


def _import_multimode_builder():
    from generate_fake_maxwell_multimode_data import build_multimode_vacuum_dataset

    return build_multimode_vacuum_dataset


def build_problem_data(
    problem: VectorProblemDef, *, fast: bool = False, **overrides: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Generate (X, Y, G, meta) for a problem.

    Parameters
    ----------
    problem : VectorProblemDef
    fast : bool
        If True, use reduced grid sizes.
    **overrides
        Override individual grid/physics parameters.

    Returns
    -------
    X : Tensor (N, D)
    Y : Tensor (N, Ny)
    G : Tensor (N, Ny, D)
    meta : dict
    """
    if problem.id == "mw000":
        defaults = dict(
            c=1.0,
            k=1.0,
            nt=64 if not fast else 32,
            nz=64 if not fast else 32,
            t_max=2.0 * math.pi,
            z_max=2.0 * math.pi,
            x0=0.0,
            y0=0.0,
            noise_std=0.0,
            seed=0,
        )
        defaults.update(overrides)
        builder = _import_plane_wave_builder()
        arrays, meta = builder(**defaults)

    elif problem.id == "mw001":
        defaults = dict(
            a0=1.0,
            r0=0.7,
            omega=1.2,
            phase=0.3,
            nt=20 if not fast else 12,
            nx=31 if not fast else 15,
            ny=31 if not fast else 15,
            nz=1,
            t_max=2.0 * math.pi,
            x_max=2.5,
            y_max=2.5,
            z_max=0.0,
        )
        defaults.update(overrides)
        builder = _import_wire_builder()
        arrays, meta = builder(**defaults)

    elif problem.id == "mw002":
        defaults = dict(
            sigma=0.6,
            a1=1.0,
            a2=0.8,
            a3=0.7,
            k1=1.0,
            k2=2.0,
            k3=2.0,
            phi1=0.1,
            phi2=0.7,
            phi3=-0.5,
            nt=16 if not fast else 10,
            nx=10 if not fast else 6,
            ny=10 if not fast else 6,
            nz=10 if not fast else 6,
            t_max=8.0,
            x_max=2.0 * math.pi,
            y_max=2.0 * math.pi,
            z_max=2.0 * math.pi,
        )
        defaults.update(overrides)
        builder = _import_conductive_builder()
        arrays, meta = builder(**defaults)

    elif problem.id == "mw003":
        defaults = dict(
            nt=16 if not fast else 10,
            nx=8 if not fast else 6,
            ny=8 if not fast else 6,
            nz=8 if not fast else 6,
            t_max=2.0 * math.pi,
            box=2.0 * math.pi,
            noise_std=0.0,
            seed=0,
        )
        defaults.update(overrides)
        builder = _import_multimode_builder()
        arrays, meta = builder(**defaults)

    else:
        raise ValueError(f"Unknown problem id: {problem.id}")

    X = torch.from_numpy(np.asarray(arrays["X"], dtype=np.float64))
    Y = torch.from_numpy(np.asarray(arrays["Y"], dtype=np.float64))
    G = torch.from_numpy(np.asarray(arrays["G"], dtype=np.float64))
    return X, Y, G, meta


# ---------------------------------------------------------------------------
# Problem registry
# ---------------------------------------------------------------------------

PROBLEM_REGISTRY: dict[str, VectorProblemDef] = {
    "mw000": VectorProblemDef(
        id="mw000",
        description="Vacuum plane wave",
        field_names=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
        n_fields=6,
        equations=[
            VectorEquationSpec(out_idxs=(0, 1, 2), name="Ampere"),
            VectorEquationSpec(out_idxs=(3, 4, 5), name="Faraday"),
        ],
        spatial_axes=(1, 2, 3),
    ),
    "mw001": VectorProblemDef(
        id="mw001",
        description="Wire source (AC current)",
        field_names=["Ex", "Ey", "Ez", "Bx", "By", "Bz", "Jx", "Jy", "Jz"],
        n_fields=9,
        equations=[
            VectorEquationSpec(out_idxs=(0, 1, 2), name="AmpereSource"),
            VectorEquationSpec(out_idxs=(3, 4, 5), name="Faraday"),
        ],
        spatial_axes=(1, 2, 3),
    ),
    "mw002": VectorProblemDef(
        id="mw002",
        description="Conductive medium",
        field_names=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
        n_fields=6,
        equations=[
            VectorEquationSpec(out_idxs=(0, 1, 2), name="AmpereConductive"),
            VectorEquationSpec(out_idxs=(3, 4, 5), name="Faraday"),
        ],
        spatial_axes=(1, 2, 3),
        params=["sigma"],
        param_defaults={"sigma": 0.6},
    ),
    "mw003": VectorProblemDef(
        id="mw003",
        description="Vacuum multi-mode (identifiable)",
        field_names=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
        n_fields=6,
        equations=[
            VectorEquationSpec(out_idxs=(0, 1, 2), name="Ampere"),
            VectorEquationSpec(out_idxs=(3, 4, 5), name="Faraday"),
        ],
        spatial_axes=(1, 2, 3),
    ),
}

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

GROUND_TRUTH: dict[str, VectorGroundTruth] = {
    # mw000: dE/dt = +curl(B), dB/dt = -curl(E)
    # In residual form: dE/dt + c*curl(B) + ... = 0 => c_curl_B = -1
    "mw000": VectorGroundTruth(
        order=1,
        expected_coeffs={
            "curl(B)": {0: -1.0},  # Ampere: coeff(curl(B)) = -1
            "curl(E)": {1: +1.0},  # Faraday: coeff(curl(E)) = +1
        },
        decoy_terms=["E", "B", "laplacian(E)", "laplacian(B)"],
        coeff_tol=0.05,
        decoy_tol=0.05,
        rms_tol=1e-6,
    ),
    # mw001: dE/dt - curl(B) + J = 0, dB/dt + curl(E) = 0
    "mw001": VectorGroundTruth(
        order=1,
        expected_coeffs={
            "curl(B)": {0: -1.0},  # Ampere: coeff(curl(B)) = -1
            "curl(E)": {1: +1.0},  # Faraday: coeff(curl(E)) = +1
            "J": {0: +1.0},  # Ampere: coeff(J) = +1
        },
        decoy_terms=["E", "B", "laplacian(E)", "laplacian(B)"],
        coeff_tol=0.10,
        decoy_tol=0.10,
        rms_tol=2e-6,
    ),
    # mw002: dE/dt - curl(B) + sigma*E = 0, dB/dt + curl(E) = 0
    "mw002": VectorGroundTruth(
        order=1,
        expected_coeffs={
            "curl(B)": {0: -1.0},  # Ampere: coeff(curl(B)) = -1
            "curl(E)": {1: +1.0},  # Faraday: coeff(curl(E)) = +1
            "E": {0: 0.6},  # Ampere: coeff(E) = sigma = 0.6
        },
        decoy_terms=["B", "laplacian(E)", "laplacian(B)"],
        coeff_tol=0.12,
        decoy_tol=0.12,
        rms_tol=3e-6,
        gauss_tol=1e-10,
    ),
    # mw003: identifiable vacuum.  dE/dt = +curl(B), dB/dt = -curl(E), same as
    # mw000 but a multi-mode superposition so lap(E) is NOT proportional to E
    # (the broadened library is full rank: the ∇² decoys are genuine impostors).
    "mw003": VectorGroundTruth(
        order=1,
        expected_coeffs={
            "curl(B)": {0: -1.0},
            "curl(E)": {1: +1.0},
        },
        decoy_terms=["E", "B", "laplacian(E)", "laplacian(B)"],
        coeff_tol=0.05,
        decoy_tol=0.05,
        rms_tol=1e-6,
    ),
}

# ---------------------------------------------------------------------------
# Per-component ground truth for factorized symbolic search validation
# ---------------------------------------------------------------------------
# Feature column layout (shared across all problems):
#   0:  dBx_dy   1:  dBx_dz   2:  dBy_dx   3:  dBy_dz   4:  dBz_dx   5:  dBz_dy
#   6:  dEx_dy   7:  dEx_dz   8:  dEy_dx   9:  dEy_dz  10:  dEz_dx  11:  dEz_dy
#  12:  Ex      13:  Ey      14:  Ez      15:  Bx      16:  By      17:  Bz
#  18:  Jx      19:  Jy      20:  Jz   (only when n_fields == 9)
#
# Ampere:  dE/dt = curl(B) [- sigma*E] [- J]
#   curl(B)_x = dBz/dy - dBy/dz   (features 5, 3)
#   curl(B)_y = dBx/dz - dBz/dx   (features 1, 4)
#   curl(B)_z = dBy/dx - dBx/dy   (features 2, 0)
# Faraday: dB/dt = -curl(E)
#   -curl(E)_x = dEy/dz - dEz/dy  (features 9, 11)
#   -curl(E)_y = dEz/dx - dEx/dz  (features 10, 7)
#   -curl(E)_z = dEx/dy - dEy/dx  (features 6, 8)

COMPONENT_GROUND_TRUTH: dict[str, list[ComponentGroundTruth]] = {
    # mw000: vacuum  dE/dt = curl(B),  dB/dt = -curl(E)
    "mw000": [
        ComponentGroundTruth("dEx/dt", "Ampere", "x", {"dBz_dy": 1.0, "dBy_dz": -1.0}, 1e-6),
        ComponentGroundTruth("dEy/dt", "Ampere", "y", {"dBx_dz": 1.0, "dBz_dx": -1.0}, 1e-6),
        ComponentGroundTruth("dEz/dt", "Ampere", "z", {"dBy_dx": 1.0, "dBx_dy": -1.0}, 1e-6),
        ComponentGroundTruth("dBx/dt", "Faraday", "x", {"dEy_dz": 1.0, "dEz_dy": -1.0}, 1e-6),
        ComponentGroundTruth("dBy/dt", "Faraday", "y", {"dEz_dx": 1.0, "dEx_dz": -1.0}, 1e-6),
        ComponentGroundTruth("dBz/dt", "Faraday", "z", {"dEx_dy": 1.0, "dEy_dx": -1.0}, 1e-6),
    ],
    # mw001: wire source  dE/dt = curl(B) - J,  dB/dt = -curl(E)
    "mw001": [
        ComponentGroundTruth("dEx/dt", "Ampere", "x", {"dBz_dy": 1.0, "dBy_dz": -1.0, "Jx": -1.0}, 2e-6),
        ComponentGroundTruth("dEy/dt", "Ampere", "y", {"dBx_dz": 1.0, "dBz_dx": -1.0, "Jy": -1.0}, 2e-6),
        ComponentGroundTruth("dEz/dt", "Ampere", "z", {"dBy_dx": 1.0, "dBx_dy": -1.0, "Jz": -1.0}, 2e-6),
        ComponentGroundTruth("dBx/dt", "Faraday", "x", {"dEy_dz": 1.0, "dEz_dy": -1.0}, 2e-6),
        ComponentGroundTruth("dBy/dt", "Faraday", "y", {"dEz_dx": 1.0, "dEx_dz": -1.0}, 2e-6),
        ComponentGroundTruth("dBz/dt", "Faraday", "z", {"dEx_dy": 1.0, "dEy_dx": -1.0}, 2e-6),
    ],
    # mw002: conductive  dE/dt = curl(B) - sigma*E,  dB/dt = -curl(E)
    "mw002": [
        ComponentGroundTruth("dEx/dt", "Ampere", "x", {"dBz_dy": 1.0, "dBy_dz": -1.0, "Ex": -0.6}, 3e-6),
        ComponentGroundTruth("dEy/dt", "Ampere", "y", {"dBx_dz": 1.0, "dBz_dx": -1.0, "Ey": -0.6}, 3e-6),
        ComponentGroundTruth("dEz/dt", "Ampere", "z", {"dBy_dx": 1.0, "dBx_dy": -1.0, "Ez": -0.6}, 3e-6),
        ComponentGroundTruth("dBx/dt", "Faraday", "x", {"dEy_dz": 1.0, "dEz_dy": -1.0}, 3e-6),
        ComponentGroundTruth("dBy/dt", "Faraday", "y", {"dEz_dx": 1.0, "dEx_dz": -1.0}, 3e-6),
        ComponentGroundTruth("dBz/dt", "Faraday", "z", {"dEx_dy": 1.0, "dEy_dx": -1.0}, 3e-6),
    ],
}
