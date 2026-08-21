# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Problem definitions for the Feynman Complex-Valued DE benchmark.

This module still carries the benchmark registry, RHS/data generators, and
ground-truth builders used by the active factorized symbolic search runner. It also keeps the
older AST-library metadata for the legacy system-DE harnesses and reference
validation paths. Dimensional metadata is exposed both in the legacy
``ComplexProblemDims`` form and through the shared canonical
``CanonicalProblemDims`` adapter layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from nestynet_sr.sr_core.problem_dims import CanonicalProblemDims

from nestynet_sr.sr_core.bridges import (
    Add,
    Cos,
    D2U,
    DU,
    Mul,
    Node,
    Pow,
    Sin,
    U,
    Var,
)

# ---------------------------------------------------------------------------
# Problem definition dataclass
# ---------------------------------------------------------------------------


@dataclass
class ComplexProblemDef:
    """Definition of a complex-valued DE problem."""

    id: str
    type: str  # complex_ode, complex_pde, complex_system
    order: int
    axes: str
    x_axis: int
    fields: str
    equation: str
    description: str
    ref: str
    complex_ops: str
    params: List[str]
    param_ranges: List[Tuple[float, float]]


DimVec = Tuple[float, ...]


@dataclass(frozen=True)
class ComplexProblemDims:
    """Dimensional metadata for the active coupled-real complex benchmark path."""

    basis: Tuple[str, ...]
    axis_dims: Tuple[DimVec, ...]
    component_dims: Tuple[DimVec, ...]
    constant_dims: Dict[str, DimVec] = field(default_factory=dict)
    complex_pairs: Tuple[Tuple[int, int], ...] = ()
    target_dims: Optional[Tuple[DimVec, ...]] = None


def _dims(*vals: float) -> DimVec:
    return tuple(float(v) for v in vals)


def _repeat_dim(dim: DimVec, n: int) -> Tuple[DimVec, ...]:
    return tuple(tuple(float(v) for v in dim) for _ in range(int(n)))


def load_complex_problems(benchmark_path: str | Path) -> Dict[str, ComplexProblemDef]:
    """Parse the tab-separated benchmark file."""
    problems: Dict[str, ComplexProblemDef] = {}
    with open(benchmark_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 12:
                continue

            pid = parts[0].strip()
            ptype = parts[1].strip()
            order = int(parts[2].strip())
            axes = parts[3].strip()
            x_axis = int(parts[4].strip())
            fields = parts[5].strip()
            equation = parts[6].strip()
            desc = parts[7].strip().strip('"')
            ref = parts[8].strip()
            complex_ops = parts[9].strip()

            params_str = parts[10].strip()
            params = (
                [] if params_str == "-" else [p.strip() for p in params_str.split(",")]
            )

            ranges_str = parts[11].strip()
            if ranges_str == "-":
                param_ranges: List[Tuple[float, float]] = []
            else:
                param_ranges = []
                for m in re.finditer(r"\[([^\]]+)\]", ranges_str):
                    lo, hi = m.group(1).split(",")
                    param_ranges.append((float(lo), float(hi)))

            problems[pid] = ComplexProblemDef(
                id=pid,
                type=ptype,
                order=order,
                axes=axes,
                x_axis=x_axis,
                fields=fields,
                equation=equation,
                description=desc,
                ref=ref,
                complex_ops=complex_ops,
                params=params,
                param_ranges=param_ranges,
            )

    return problems


# ---------------------------------------------------------------------------
# Default parameter values, initial conditions, and integration time
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: Dict[str, Dict[str, float]] = {
    "C200": {"gamma": 0.3, "omega0": 2.0},
    "C203": {"mu": 1.0, "omega": 2.0},
    "C303": {"mu": 1.0, "omega": 2.0, "beta": 0.5},
    "C004": {"E1": 1.0, "E2": 1.5, "V12": 0.3, "V21": 0.3},
    "C005": {"m": 1.0, "p": 1.5, "Bx": 0.4, "Bz": 0.3},
    "C202": {"omega1": 1.5, "omega2": 2.5, "kappa": 0.3},
    "C104": {"kappa": 0.5},
    "C000": {"hbar": 1.0, "m": 1.0, "k": 2.0},
    "C002": {"hbar": 1.0, "m": 1.0, "V": 1.5, "k": 1.0},
    "C304": {"c": 2.0, "m": 1.0, "k": 0.4},
    "C003": {"g": 1.0},
    "C100": {},
    "C103": {},
    "C301": {"a": -1.0, "b": 1.0, "kappa": 0.5},
    "C008": {"m": 1.0, "k": 1.0},
    "C101": {"a": 1.0, "b": 0.5},
    "C305": {"g1": 1.0, "g2": 1.5, "kappa": 0.3},
    "C302": {"alpha": 0.3, "I_ext": 0.5},
    "C105": {"kappa": 0.5, "Delta": 1.0},
    "C201": {"gamma": 0.5, "omega0": 4.0, "F": 1.0, "Omega": 1.0},
    "C204": {"L": 1.0, "R": 1.0, "C_cap": 0.5, "V": 1.5},
    "C102": {},  # all coefficients are 1
    "C001": {"hbar": 1.0, "m": 1.0, "omega": 2.0},
    "C006": {"hbar": 1.0, "m": 1.0, "V": 1.0, "E": 3.0},
    "C007": {"hbar": 1.0, "m": 1.0, "V_eff": 1.0, "E": 3.0},
    "C300": {"g": 0.5, "N0": 1.0},
}

DEFAULT_ICS: Dict[str, Dict[str, float]] = {
    # C200: u and v must be linearly independent to avoid rank-deficient
    # regression.  Giving v a nonzero initial velocity and zero displacement
    # makes v(t) ~ sin while u(t) ~ cos, breaking collinearity.
    "C200": {"u0": 1.0, "v0": 0.0, "du0": 0.0, "dv0": 2.0},
    "C203": {"u0": 0.1, "v0": 0.1},
    "C303": {"u0": 0.1, "v0": 0.1},
    "C004": {"u1": 1.0, "v1": 0.0, "u2": 0.0, "v2": 0.0},
    "C005": {"u1": 1.0, "v1": 0.0, "u2": 0.0, "v2": 0.0},
    "C202": {"u1": 1.0, "v1": 0.0, "u2": 0.0, "v2": 0.5},
    "C104": {"u1": 1.0, "v1": 0.5, "u2": 0.1, "v2": 0.0},
    "C302": {"u0": 0.5, "v0": 0.0},
    "C105": {"u0": 1.0, "v0": 0.0},
    "C201": {"u0": 2.0, "v0": -1.0, "du0": 1.0, "dv0": 3.0},
    "C300": {"u0": 0.3, "v0": 0.1},
}

DEFAULT_TMAX: Dict[str, float] = {
    "C200": 10.0,
    "C203": 15.0,
    "C303": 15.0,
    "C004": 15.0,
    "C005": 15.0,
    "C202": 15.0,
    "C104": 5.0,
    "C000": np.pi,
    "C002": np.pi,
    "C304": np.pi,
    "C003": 2 * np.pi,
    "C100": 2 * np.pi,
    "C103": 2 * np.pi,
    "C301": 2.0,
    "C008": np.pi,
    "C101": 2.0,
    "C305": 1.0,
    "C302": 15.0,
    "C105": 10.0,
    "C201": 5.0,
    "C204": 5.0,  # omega_max (frequency range upper bound)
    "C102": 1.5,
    "C001": np.pi,
    "C006": np.pi,  # x_max: one full period for k=2
    "C007": 4.0,    # r_max (domain starts at r_min=1.0 inside generator)
    "C300": 15.0,
}

# Number of real components per problem (2 = one complex field, 4 = two complex fields).
NCOMPONENTS: Dict[str, int] = {
    "C200": 2,
    "C203": 2,
    "C303": 2,
    "C004": 4,
    "C005": 4,
    "C202": 4,
    "C104": 4,
    "C000": 2,
    "C002": 2,
    "C304": 2,
    "C003": 2,
    "C100": 2,
    "C103": 2,
    "C301": 2,
    "C008": 4,
    "C101": 2,
    "C305": 4,
    "C302": 2,
    "C105": 2,
    "C201": 2,
    "C204": 2,
    "C102": 4,
    "C001": 2,
    "C006": 2,
    "C007": 2,
    "C300": 2,
}

# Number of spatial + temporal input variables per problem.
# ODEs have 1 (time only); PDEs have 2+ (e.g. t, x).
NXVARS: Dict[str, int] = {
    "C200": 1,
    "C203": 1,
    "C303": 1,
    "C004": 1,
    "C005": 1,
    "C202": 1,
    "C104": 1,
    "C000": 2,
    "C002": 2,
    "C304": 2,
    "C003": 2,
    "C100": 2,
    "C103": 2,
    "C301": 2,
    "C008": 2,
    "C101": 2,
    "C305": 2,
    "C302": 1,
    "C105": 1,
    "C201": 1,
    "C204": 1,  # ω is the single independent variable
    "C102": 2,  # z, t
    "C001": 2,  # t, x
    "C006": 1,  # x only (time-independent)
    "C007": 1,  # r only (radial, time-independent)
    "C300": 1,  # T only (temperature)
}


COMPLEX_DIMS_REGISTRY: Dict[str, ComplexProblemDims] = {
    "C200": ComplexProblemDims(
        basis=("T", "Z"),
        axis_dims=(_dims(1, 0),),
        component_dims=_repeat_dim(_dims(0, 1), 2),
        constant_dims={"gamma": _dims(-1, 0), "omega0": _dims(-1, 0)},
        complex_pairs=((0, 1),),
    ),
    "C203": ComplexProblemDims(
        basis=("T",),
        axis_dims=(_dims(1),),
        component_dims=_repeat_dim(_dims(-0.5), 2),
        constant_dims={"mu": _dims(-1), "omega": _dims(-1)},
        complex_pairs=((0, 1),),
    ),
    "C303": ComplexProblemDims(
        basis=("T",),
        axis_dims=(_dims(1),),
        component_dims=_repeat_dim(_dims(-0.5), 2),
        constant_dims={"mu": _dims(-1), "omega": _dims(-1), "beta": _dims(0)},
        complex_pairs=((0, 1),),
    ),
    "C004": ComplexProblemDims(
        basis=("T",),
        axis_dims=(_dims(1),),
        component_dims=_repeat_dim(_dims(0), 4),
        constant_dims={"E1": _dims(-1), "E2": _dims(-1), "V12": _dims(-1), "V21": _dims(-1)},
        complex_pairs=((0, 1), (2, 3)),
    ),
    "C005": ComplexProblemDims(
        basis=("T", "M"),
        axis_dims=(_dims(1, 0),),
        component_dims=_repeat_dim(_dims(0, 0), 4),
        constant_dims={
            "m": _dims(0, 1),
            "p": _dims(-0.5, 0.5),
            "Bx": _dims(-1, 0),
            "Bz": _dims(-1, 0),
        },
        complex_pairs=((0, 1), (2, 3)),
    ),
    "C202": ComplexProblemDims(
        basis=("T",),
        axis_dims=(_dims(1),),
        component_dims=_repeat_dim(_dims(0), 4),
        constant_dims={"omega1": _dims(-1), "omega2": _dims(-1), "kappa": _dims(-1)},
        complex_pairs=((0, 1), (2, 3)),
    ),
    "C104": ComplexProblemDims(
        basis=("Z",),
        axis_dims=(_dims(1),),
        component_dims=_repeat_dim(_dims(-1), 4),
        constant_dims={"kappa": _dims(0)},
        complex_pairs=((0, 1), (2, 3)),
    ),
    "C000": ComplexProblemDims(
        basis=("X", "T", "M", "Psi"),
        axis_dims=(_dims(0, 1, 0, 0), _dims(1, 0, 0, 0)),
        component_dims=_repeat_dim(_dims(0, 0, 0, 1), 2),
        constant_dims={"hbar": _dims(2, -1, 1, 0), "m": _dims(0, 0, 1, 0), "k": _dims(-1, 0, 0, 0)},
        complex_pairs=((0, 1),),
    ),
    "C001": ComplexProblemDims(
        basis=("X", "T", "M", "Psi"),
        axis_dims=(_dims(0, 1, 0, 0), _dims(1, 0, 0, 0)),
        component_dims=_repeat_dim(_dims(0, 0, 0, 1), 2),
        constant_dims={
            "hbar": _dims(2, -1, 1, 0),
            "m": _dims(0, 0, 1, 0),
            "omega": _dims(0, -1, 0, 0),
        },
        complex_pairs=((0, 1),),
    ),
    "C002": ComplexProblemDims(
        basis=("X", "T", "M", "Psi"),
        axis_dims=(_dims(0, 1, 0, 0), _dims(1, 0, 0, 0)),
        component_dims=_repeat_dim(_dims(0, 0, 0, 1), 2),
        constant_dims={
            "hbar": _dims(2, -1, 1, 0),
            "m": _dims(0, 0, 1, 0),
            "V": _dims(2, -2, 1, 0),
            "k": _dims(-1, 0, 0, 0),
        },
        complex_pairs=((0, 1),),
    ),
    "C003": ComplexProblemDims(
        basis=("X", "Psi"),
        axis_dims=(_dims(2, 0), _dims(1, 0)),
        component_dims=_repeat_dim(_dims(0, 1), 2),
        constant_dims={"g": _dims(-2, -2)},
        complex_pairs=((0, 1),),
    ),
    "C100": ComplexProblemDims(
        basis=("X",),
        axis_dims=(_dims(2), _dims(1)),
        component_dims=_repeat_dim(_dims(-1), 2),
        complex_pairs=((0, 1),),
    ),
    "C101": ComplexProblemDims(
        basis=("D",),
        axis_dims=(_dims(0), _dims(0)),
        component_dims=_repeat_dim(_dims(0), 2),
        constant_dims={"a": _dims(0), "b": _dims(0)},
        complex_pairs=((0, 1),),
    ),
    "C102": ComplexProblemDims(
        basis=("X",),
        axis_dims=(_dims(2), _dims(1)),
        component_dims=_repeat_dim(_dims(-1), 4),
        complex_pairs=((0, 1), (2, 3)),
    ),
    "C103": ComplexProblemDims(
        basis=("X",),
        axis_dims=(_dims(2), _dims(1)),
        component_dims=_repeat_dim(_dims(-1), 2),
        complex_pairs=((0, 1),),
    ),
    "C105": ComplexProblemDims(
        basis=("Z", "A"),
        axis_dims=(_dims(1, 0),),
        component_dims=_repeat_dim(_dims(0, 1), 2),
        constant_dims={"kappa": _dims(-1, 0), "Delta": _dims(-1, 0)},
        complex_pairs=((0, 1),),
    ),
    "C201": ComplexProblemDims(
        basis=("T", "Z"),
        axis_dims=(_dims(1, 0),),
        component_dims=_repeat_dim(_dims(0, 1), 2),
        constant_dims={
            "gamma": _dims(-1, 0),
            "omega0": _dims(-1, 0),
            "F": _dims(-2, 1),
            "Omega": _dims(-1, 0),
        },
        complex_pairs=((0, 1),),
    ),
    "C204": ComplexProblemDims(
        basis=("T", "I", "V"),
        axis_dims=(_dims(-1, 0, 0),),
        component_dims=_repeat_dim(_dims(0, 1, 0), 2),
        constant_dims={
            "L": _dims(1, -1, 1),
            "R": _dims(0, -1, 1),
            "C_cap": _dims(1, 1, -1),
            "V": _dims(0, 0, 1),
        },
        complex_pairs=((0, 1),),
        target_dims=_repeat_dim(_dims(0, 0, 1), 2),
    ),
    "C300": ComplexProblemDims(
        basis=("Theta",),
        axis_dims=(_dims(1),),
        component_dims=_repeat_dim(_dims(0), 2),
        constant_dims={"g": _dims(-1), "N0": _dims(0)},
        complex_pairs=((0, 1),),
    ),
    "C301": ComplexProblemDims(
        basis=("X",),
        axis_dims=(_dims(2), _dims(1)),
        component_dims=_repeat_dim(_dims(0), 2),
        constant_dims={"a": _dims(-2), "b": _dims(-2), "kappa": _dims(0)},
        complex_pairs=((0, 1),),
    ),
    "C302": ComplexProblemDims(
        basis=("D",),
        axis_dims=(_dims(0),),
        component_dims=(_dims(0), _dims(0)),
        constant_dims={"alpha": _dims(0), "I_ext": _dims(0)},
        complex_pairs=(),
    ),
    "C304": ComplexProblemDims(
        basis=("X", "T"),
        axis_dims=(_dims(0, 1), _dims(1, 0)),
        component_dims=_repeat_dim(_dims(0, 0), 2),
        constant_dims={"c": _dims(1, -1), "m": _dims(0, -1), "k": _dims(-1, 0)},
        complex_pairs=((0, 1),),
    ),
    "C305": ComplexProblemDims(
        basis=("X",),
        axis_dims=(_dims(2), _dims(1)),
        component_dims=_repeat_dim(_dims(0), 4),
        constant_dims={"g1": _dims(-2), "g2": _dims(-2), "kappa": _dims(-2)},
        complex_pairs=((0, 1), (2, 3)),
    ),
    "C006": ComplexProblemDims(
        basis=("X", "T", "M", "Psi"),
        axis_dims=(_dims(1, 0, 0, 0),),
        component_dims=_repeat_dim(_dims(0, 0, 0, 1), 2),
        constant_dims={
            "hbar": _dims(2, -1, 1, 0),
            "m": _dims(0, 0, 1, 0),
            "V": _dims(2, -2, 1, 0),
            "E": _dims(2, -2, 1, 0),
        },
        complex_pairs=((0, 1),),
    ),
    "C007": ComplexProblemDims(
        basis=("R", "T", "M", "Psi"),
        axis_dims=(_dims(1, 0, 0, 0),),
        component_dims=_repeat_dim(_dims(0, 0, 0, 1), 2),
        constant_dims={
            "hbar": _dims(2, -1, 1, 0),
            "m": _dims(0, 0, 1, 0),
            "V_eff": _dims(2, -2, 1, 0),
            "E": _dims(2, -2, 1, 0),
        },
        complex_pairs=((0, 1),),
    ),
    "C008": ComplexProblemDims(
        basis=("X",),
        axis_dims=(_dims(1), _dims(1)),
        component_dims=_repeat_dim(_dims(0), 4),
        constant_dims={"m": _dims(-1), "k": _dims(-1)},
        complex_pairs=((0, 1), (2, 3)),
    ),
}


def get_complex_problem_dims(pid: str) -> Optional[ComplexProblemDims]:
    dims = COMPLEX_DIMS_REGISTRY.get(str(pid))
    if dims is None:
        return None

    ncomp = int(NCOMPONENTS.get(str(pid), 0))
    nxvars = int(NXVARS.get(str(pid), 0))
    if len(dims.axis_dims) != nxvars:
        raise ValueError(f"{pid}: axis_dims has len={len(dims.axis_dims)} but nxvars={nxvars}")
    if len(dims.component_dims) != ncomp:
        raise ValueError(f"{pid}: component_dims has len={len(dims.component_dims)} but ncomp={ncomp}")

    ndim = len(dims.basis)
    for axis_dim in dims.axis_dims:
        if len(axis_dim) != ndim:
            raise ValueError(f"{pid}: axis dim {axis_dim!r} does not match basis rank {ndim}")
    for comp_dim in dims.component_dims:
        if len(comp_dim) != ndim:
            raise ValueError(f"{pid}: component dim {comp_dim!r} does not match basis rank {ndim}")
    for name, const_dim in dims.constant_dims.items():
        if len(const_dim) != ndim:
            raise ValueError(
                f"{pid}: constant {name!r} dim {const_dim!r} does not match basis rank {ndim}"
            )

    for re_idx, im_idx in dims.complex_pairs:
        if re_idx < 0 or re_idx >= ncomp or im_idx < 0 or im_idx >= ncomp:
            raise ValueError(f"{pid}: complex pair {(re_idx, im_idx)!r} out of range for ncomp={ncomp}")
        if tuple(dims.component_dims[re_idx]) != tuple(dims.component_dims[im_idx]):
            raise ValueError(
                f"{pid}: complex pair {(re_idx, im_idx)!r} must share units, got "
                f"{dims.component_dims[re_idx]!r} vs {dims.component_dims[im_idx]!r}"
            )

    if dims.target_dims is not None:
        if len(dims.target_dims) != ncomp:
            raise ValueError(f"{pid}: target_dims has len={len(dims.target_dims)} but ncomp={ncomp}")
        for tgt in dims.target_dims:
            if len(tgt) != ndim:
                raise ValueError(f"{pid}: target dim {tgt!r} does not match basis rank {ndim}")

    return dims


def to_canonical_problem_dims(dims: ComplexProblemDims) -> CanonicalProblemDims:
    """Convert complex benchmark dims into the shared canonical form."""
    return CanonicalProblemDims(
        basis=tuple(str(v) for v in dims.basis),
        axis_dims=tuple(tuple(float(v) for v in dim) for dim in dims.axis_dims),
        component_dims=tuple(tuple(float(v) for v in dim) for dim in dims.component_dims),
        constant_dims={
            str(name): tuple(float(v) for v in dim)
            for name, dim in dict(dims.constant_dims).items()
        },
        complex_pairs=tuple((int(a), int(b)) for a, b in dims.complex_pairs),
        target_dims=None
        if dims.target_dims is None
        else tuple(tuple(float(v) for v in dim) for dim in dims.target_dims),
    )


def get_canonical_problem_dims(pid: str) -> Optional[CanonicalProblemDims]:
    dims = get_complex_problem_dims(str(pid))
    if dims is None:
        return None
    return to_canonical_problem_dims(dims)


# ---------------------------------------------------------------------------
# RHS functions (real 2-component decomposition for solve_ivp)
# ---------------------------------------------------------------------------


def _rhs_c200(t, state, params):
    """C200: d2z/dt2 + 2*gamma*dz/dt + omega0^2*z = 0.

    State = [u, v, du/dt, dv/dt] (2nd order, u and v decouple).
    """
    gamma = params["gamma"]
    omega0 = params["omega0"]
    u, v, du, dv = state
    return [
        du,
        dv,
        -2 * gamma * du - omega0**2 * u,
        -2 * gamma * dv - omega0**2 * v,
    ]


def _rhs_c203(t, state, params):
    """C203: dA/dt = (mu - |A|^2)*A + i*omega*A.

    State = [u, v] where A = u + iv.
      du/dt = (mu - u^2 - v^2)*u - omega*v
      dv/dt = (mu - u^2 - v^2)*v + omega*u
    """
    mu = params["mu"]
    omega = params["omega"]
    u, v = state
    mod_sq = u**2 + v**2
    return [
        (mu - mod_sq) * u - omega * v,
        (mu - mod_sq) * v + omega * u,
    ]


def _rhs_c303(t, state, params):
    """C303: dA/dt = (mu+i*omega)*A - (1+i*beta)*|A|^2*A.

    State = [u, v] where A = u + iv.
      du/dt = mu*u - omega*v - (u^2+v^2)*u + beta*(u^2+v^2)*v
      dv/dt = omega*u + mu*v - (u^2+v^2)*v - beta*(u^2+v^2)*u
    """
    mu = params["mu"]
    omega = params["omega"]
    beta = params["beta"]
    u, v = state
    mod_sq = u**2 + v**2
    return [
        mu * u - omega * v - mod_sq * u + beta * mod_sq * v,
        omega * u + mu * v - mod_sq * v - beta * mod_sq * u,
    ]


def _rhs_c004(t, state, params):
    """C004: Two-level quantum system i·dc₁/dt = E₁c₁ + V₁₂c₂.

    State = [u1, v1, u2, v2] where c₁ = u1+iv1, c₂ = u2+iv2.
    """
    E1 = params["E1"]
    E2 = params["E2"]
    V12 = params["V12"]
    V21 = params["V21"]
    u1, v1, u2, v2 = state
    return [
        E1 * v1 + V12 * v2,       # du1/dt
        -E1 * u1 - V12 * u2,      # dv1/dt
        V21 * v1 + E2 * v2,       # du2/dt
        -V21 * u1 - E2 * u2,      # dv2/dt
    ]


def _rhs_c005(t, state, params):
    """C005: Pauli spin-1/2 in magnetic field.

    i·dψ↑/dt = E_up·ψ↑ + Bx·ψ↓; i·dψ↓/dt = Bx·ψ↑ + E_dn·ψ↓
    where E_up = p²/(2m)+Bz, E_dn = p²/(2m)-Bz.
    State = [u1, v1, u2, v2].
    """
    m = params["m"]
    p = params["p"]
    Bx = params["Bx"]
    Bz = params["Bz"]
    E_up = p**2 / (2 * m) + Bz
    E_dn = p**2 / (2 * m) - Bz
    u1, v1, u2, v2 = state
    return [
        E_up * v1 + Bx * v2,      # du1/dt
        -E_up * u1 - Bx * u2,     # dv1/dt
        Bx * v1 + E_dn * v2,      # du2/dt
        -Bx * u1 - E_dn * u2,     # dv2/dt
    ]


def _rhs_c202(t, state, params):
    """C202: Coupled complex modes dz₁/dt = iω₁z₁ + κz₂.

    State = [u1, v1, u2, v2] where z₁ = u1+iv1, z₂ = u2+iv2.
    """
    omega1 = params["omega1"]
    omega2 = params["omega2"]
    kappa = params["kappa"]
    u1, v1, u2, v2 = state
    return [
        -omega1 * v1 + kappa * u2,     # du1/dt
        omega1 * u1 + kappa * v2,      # dv1/dt
        kappa * u1 - omega2 * v2,      # du2/dt
        kappa * v1 + omega2 * u2,      # dv2/dt
    ]


def _rhs_c104(t, state, params):
    """C104: Second harmonic generation.

    i·dA₁/dz = κ·conj(A₁)·A₂; i·dA₂/dz = κ·A₁²
    State = [u1, v1, u2, v2] where A₁ = u1+iv1, A₂ = u2+iv2.
    """
    kappa = params["kappa"]
    u1, v1, u2, v2 = state
    return [
        kappa * (u1 * v2 - v1 * u2),               # du1/dz
        -kappa * (u1 * u2 + v1 * v2),              # dv1/dz
        2 * kappa * u1 * v1,                        # du2/dz
        -kappa * (u1**2 - v1**2),                   # dv2/dz
    ]


def _rhs_c201(t, state, params):
    """C201: d2z/dt2 + 2*gamma*dz/dt + omega0^2*z = F*exp(i*Omega*t).

    State = [u, v, du/dt, dv/dt] (2nd order, u and v decouple).
    With Omega=1: driving = F*cos(t) for u, F*sin(t) for v.
    """
    gamma = params["gamma"]
    omega0 = params["omega0"]
    F = params["F"]
    Omega = params["Omega"]
    u, v, du, dv = state
    return [
        du,
        dv,
        -2 * gamma * du - omega0**2 * u + F * np.cos(Omega * t),
        -2 * gamma * dv - omega0**2 * v + F * np.sin(Omega * t),
    ]


def _rhs_c105(t, state, params):
    """C105: Parametric amplification i·dA/dz = κ·conj(A)·exp(iΔz).

    State = [u, v] where A = u + iv.
      du/dz = κ·u·sin(Δz) − κ·v·cos(Δz)
      dv/dz = −κ·u·cos(Δz) − κ·v·sin(Δz)
    """
    kappa = params["kappa"]
    Delta = params["Delta"]
    u, v = state
    s, c = np.sin(Delta * t), np.cos(Delta * t)
    return [
        kappa * u * s - kappa * v * c,
        -kappa * u * c - kappa * v * s,
    ]


def _rhs_c302(t, state, params):
    """C302: Josephson junction dphi/dt = V, dV/dt = -sin(phi) - alpha*V + I_ext.

    State = [phi, V].  No complex structure — real-valued 2-component system.
    """
    alpha = params["alpha"]
    I_ext = params["I_ext"]
    phi, V = state
    return [V, -np.sin(phi) - alpha * V + I_ext]


def _rhs_c300(t, state, params):
    """C300: BCS gap equation (Ginzburg-Landau relaxation).

    dΔ/dT = g·(N₀ − |Δ|²)·Δ

    State = [u, v] where Δ = u + iv.
      du/dT = g·(N₀ − u² − v²)·u
      dv/dT = g·(N₀ − u² − v²)·v
    """
    g = params["g"]
    N0 = params["N0"]
    u, v = state
    mod_sq = u**2 + v**2
    return [
        g * (N0 - mod_sq) * u,
        g * (N0 - mod_sq) * v,
    ]


RHS_REGISTRY: Dict[str, Callable] = {
    "C200": _rhs_c200,
    "C203": _rhs_c203,
    "C303": _rhs_c303,
    "C004": _rhs_c004,
    "C005": _rhs_c005,
    "C202": _rhs_c202,
    "C104": _rhs_c104,
    "C302": _rhs_c302,
    "C105": _rhs_c105,
    "C201": _rhs_c201,
    "C300": _rhs_c300,
}


# ---------------------------------------------------------------------------
# PDE data generators (analytic solutions on meshgrids)
# ---------------------------------------------------------------------------


def _generate_c000_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate plane-wave data for the free-particle Schrödinger equation.

    Returns ``(grid, [u_flat, v_flat])`` where *grid* has shape ``(N, 2)``
    with columns ``[t, x]`` and *u_flat*, *v_flat* are 1-D arrays of the
    same length.
    """
    hbar = params["hbar"]
    m = params["m"]
    k = params["k"]
    omega = hbar * k**2 / (2 * m)

    n_side = int(np.sqrt(n_points))
    t_1d = np.linspace(0, t_max, n_side, dtype=np.float64)
    x_1d = np.linspace(0, np.pi, n_side, dtype=np.float64)
    T, X = np.meshgrid(t_1d, x_1d, indexing="ij")

    phase = k * X - omega * T
    u_flat = np.cos(phase).ravel()
    v_flat = np.sin(phase).ravel()
    grid = np.column_stack([T.ravel(), X.ravel()])

    return grid, [u_flat, v_flat]


def _generate_c002_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate data for 1D Schrödinger with constant potential V.

    Uses a superposition of two plane waves (k1, k2) so that the spatial
    Laplacian is not collinear with the field.  A single plane wave has
    ∂²ψ/∂x² = -k²ψ, making the D2U and U library columns rank-deficient.
    """
    hbar = params["hbar"]
    m = params["m"]
    V = params["V"]
    k1 = params["k"]
    k2 = 1.5 * k1          # second wavenumber, moderate separation for stable FD
    omega1 = hbar * k1**2 / (2 * m) + V / hbar
    omega2 = hbar * k2**2 / (2 * m) + V / hbar

    n_side = int(np.sqrt(n_points))
    t_1d = np.linspace(0, t_max, n_side, dtype=np.float64)
    x_1d = np.linspace(0, 2 * np.pi, n_side, dtype=np.float64)
    T, X = np.meshgrid(t_1d, x_1d, indexing="ij")

    phase1 = k1 * X - omega1 * T
    phase2 = k2 * X - omega2 * T
    u_flat = (np.cos(phase1) + 0.5 * np.cos(phase2)).ravel()
    v_flat = (np.sin(phase1) + 0.5 * np.sin(phase2)).ravel()
    grid = np.column_stack([T.ravel(), X.ravel()])

    return grid, [u_flat, v_flat]


def _generate_c304_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate data for the complex Klein-Gordon equation.

    ∂²φ/∂t² − c²·∂²φ/∂x² + m²·φ = 0

    Dispersion: ω² = c²k² + m².  Two-wave superposition breaks the
    collinearity between ∂²φ/∂x² and φ.
    """
    c = params["c"]
    m = params["m"]
    k1 = params["k"]
    k2 = 1.2 * k1
    omega1 = np.sqrt(c**2 * k1**2 + m**2)
    omega2 = np.sqrt(c**2 * k2**2 + m**2)

    n_side = int(np.sqrt(n_points))
    t_1d = np.linspace(0, t_max, n_side, dtype=np.float64)
    x_1d = np.linspace(0, 2 * np.pi, n_side, dtype=np.float64)
    T, X = np.meshgrid(t_1d, x_1d, indexing="ij")

    phase1 = k1 * X - omega1 * T
    phase2 = k2 * X - omega2 * T
    u_flat = (np.cos(phase1) + 0.5 * np.cos(phase2)).ravel()
    v_flat = (np.sin(phase1) + 0.5 * np.sin(phase2)).ravel()
    grid = np.column_stack([T.ravel(), X.ravel()])

    return grid, [u_flat, v_flat]


def _generate_c003_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate data for Gross-Pitaevskii / defocusing NLS.

    i·∂ψ/∂t = −∂²ψ/∂x² + g·|ψ|²·ψ

    Uses the black soliton (dark soliton at rest):
      ψ(x,t) = A·tanh(A·√(g/2)·x)·exp(−i·g·A²·t)

    This breaks the D2U vs |ψ|²·ψ collinearity because ∂²tanh/∂x² ∝
    sech²·tanh is not proportional to tanh³.
    """
    g = params["g"]
    A = 1.0
    xi_scale = A * np.sqrt(g / 2)
    omega = g * A**2

    n_side = int(np.sqrt(n_points))
    t_1d = np.linspace(0, t_max, n_side, dtype=np.float64)
    x_1d = np.linspace(-5.0, 5.0, n_side, dtype=np.float64)
    T, X = np.meshgrid(t_1d, x_1d, indexing="ij")

    f = A * np.tanh(xi_scale * X)
    u_flat = (f * np.cos(omega * T)).ravel()
    v_flat = (-f * np.sin(omega * T)).ravel()
    grid = np.column_stack([T.ravel(), X.ravel()])

    return grid, [u_flat, v_flat]


def _generate_nls_bright_soliton(
    g: float, t_max: float, n_points: int, *, swap_axes: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate bright-soliton data for focusing NLS: i·u_z + u_tt + g|u|²u = 0.

    Soliton: u = A·sech(A√(g/2)·t)·exp(i·g·A²·z/2) with A=√2.

    Parameters
    ----------
    g : float
        Nonlinear coefficient.
    t_max : float
        Upper limit for axis-0 (evolution axis).
    n_points : int
        Approximate total grid points.
    swap_axes : bool
        If True, swap roles: axis-0 = t (evolution), axis-1 = x (transverse).
        Default False gives axis-0 = z, axis-1 = t.
    """
    A = np.sqrt(2.0)
    B = A * np.sqrt(g / 2)
    Omega = g * A**2 / 2

    n_side = int(np.sqrt(n_points))
    evol_1d = np.linspace(0, t_max, n_side, dtype=np.float64)
    trans_1d = np.linspace(-5.0, 5.0, n_side, dtype=np.float64)
    Z, T = np.meshgrid(evol_1d, trans_1d, indexing="ij")

    f = A * np.cosh(B * T) ** (-1)  # sech
    u_flat = (f * np.cos(Omega * Z)).ravel()
    v_flat = (f * np.sin(Omega * Z)).ravel()

    if swap_axes:
        # For C103: axis-0 = t (evolution), axis-1 = x (transverse)
        grid = np.column_stack([Z.ravel(), T.ravel()])
    else:
        # For C100: axis-0 = z (evolution), axis-1 = t (transverse)
        grid = np.column_stack([Z.ravel(), T.ravel()])

    return grid, [u_flat, v_flat]


def _generate_c100_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """C100: i·du/dz + d²u/dt² + |u|²u = 0. Bright soliton, g=1."""
    return _generate_nls_bright_soliton(g=1.0, t_max=t_max, n_points=n_points)


def _generate_c103_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """C103: i·dA/dt + d²A/dx² + 2|A|²A = 0. Bright soliton, g=2."""
    return _generate_nls_bright_soliton(g=2.0, t_max=t_max, n_points=n_points)


def _generate_c301_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate data for TDGL: ∂Φ/∂t = −a·Φ − b·|Φ|²·Φ + κ·∂²Φ/∂x².

    Uses spectral method of lines (FFT for ∂²/∂x², RK45 for time).
    With a = −1 the uniform state |Φ| = √(|a|/b) is stable, giving rich
    transient dynamics from a multi-mode initial condition.
    """
    from scipy.fft import fft, ifft
    from scipy.integrate import solve_ivp as _solve_ivp

    a = params["a"]
    b = params["b"]
    kappa = params["kappa"]

    Nx = 64
    L = 2 * np.pi
    x = np.linspace(0, L, Nx, endpoint=False, dtype=np.float64)
    k_freq = np.fft.fftfreq(Nx, d=L / Nx) * 2 * np.pi
    k2 = k_freq**2

    # Multi-mode initial condition
    u0 = 1.0 + 0.3 * np.cos(x) + 0.1 * np.cos(2 * x)
    v0 = 0.3 * np.sin(x) + 0.1 * np.sin(3 * x)
    y0 = np.concatenate([u0, v0])

    def rhs(t, y):
        u = y[:Nx]
        v = y[Nx:]
        mod_sq = u**2 + v**2
        u_xx = np.real(ifft(-k2 * fft(u)))
        v_xx = np.real(ifft(-k2 * fft(v)))
        du = -a * u - b * mod_sq * u + kappa * u_xx
        dv = -a * v - b * mod_sq * v + kappa * v_xx
        return np.concatenate([du, dv])

    Nt = int(np.sqrt(n_points))
    t_eval = np.linspace(0, t_max, Nt, dtype=np.float64)

    sol = _solve_ivp(rhs, [0, t_max], y0, t_eval=t_eval,
                     method="RK45", rtol=1e-10, atol=1e-12)
    if sol.status != 0:
        raise RuntimeError(f"C301 integration failed: {sol.message}")

    # Build (t, x) grid from Nt time steps × Nx spatial points
    T_grid, X_grid = np.meshgrid(sol.t, x, indexing="ij")  # (Nt, Nx)
    u_vals = sol.y[:Nx, :].T    # (Nt, Nx)
    v_vals = sol.y[Nx:, :].T    # (Nt, Nx)

    grid = np.column_stack([T_grid.ravel(), X_grid.ravel()])
    u_flat = u_vals.ravel()
    v_flat = v_vals.ravel()

    return grid, [u_flat, v_flat]


def _generate_c008_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate data for 1+1D Dirac equation (two-component complex spinor).

    i·∂ψ₁/∂t + i·∂ψ₁/∂x = m·ψ₂
    i·∂ψ₂/∂t − i·∂ψ₂/∂x = m·ψ₁

    Dispersion: ω² = k² + m².  Spinor ratio for mode j: rⱼ = (ωⱼ − kⱼ)/m.

    Uses a 4-wave superposition.  The library has 8 terms (4 sin-type,
    4 cos-type), each a linear combination of the mode basis functions.
    With only 2 waves the column matrix is rank-2 per group (rank-deficient);
    4 waves give full rank-4 and a unique STLSQ solution.
    """
    m_val = params["m"]
    k_base = params["k"]

    # 4 wavenumbers with both signs so spinor ratios r span both r<1 and r>1.
    # Positive k → r<1, negative k → r>1.  This gives balanced ψ₁/ψ₂
    # amplitudes and well-conditioned STLSQ columns.
    ks = [k_base, -k_base, 2.0 * k_base, -2.0 * k_base]
    amps = [1.0, 0.3, 0.5, 0.15]
    omegas = [np.sqrt(k**2 + m_val**2) for k in ks]
    rs = [(om - k) / m_val for om, k in zip(omegas, ks)]

    n_side = int(np.sqrt(n_points))
    t_1d = np.linspace(0, t_max, n_side, dtype=np.float64)
    x_1d = np.linspace(0, 2 * np.pi, n_side, dtype=np.float64)
    T, X = np.meshgrid(t_1d, x_1d, indexing="ij")

    u1 = np.zeros_like(T)
    v1 = np.zeros_like(T)
    u2 = np.zeros_like(T)
    v2 = np.zeros_like(T)
    for k, om, r, a in zip(ks, omegas, rs, amps):
        phase = k * X - om * T
        u1 += a * np.cos(phase)
        v1 += a * np.sin(phase)
        u2 += a * r * np.cos(phase)
        v2 += a * r * np.sin(phase)

    grid = np.column_stack([T.ravel(), X.ravel()])
    return grid, [u1.ravel(), v1.ravel(), u2.ravel(), v2.ravel()]


def _generate_c101_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate data for complex Ginzburg-Landau equation.

    du/dt = (1+ia)·d²u/dx² + u − (1+ib)·|u|²·u

    Uses spectral method of lines (FFT for ∂²/∂x², RK45 for time).
    With ab < 1 the uniform solution |u|=1 is stable, giving smooth
    transient dynamics from a multi-mode initial condition.
    """
    from scipy.fft import fft, ifft
    from scipy.integrate import solve_ivp as _solve_ivp

    a = params["a"]
    b = params["b"]

    Nx = 64
    L = 2 * np.pi
    x = np.linspace(0, L, Nx, endpoint=False, dtype=np.float64)
    k_freq = np.fft.fftfreq(Nx, d=L / Nx) * 2 * np.pi
    k2 = k_freq**2

    # Smooth initial condition with |u₀|² varying from ~0.25 to ~2.25.
    # This breaks the U vs |u|²·U collinearity while keeping the dynamics
    # smooth enough for accurate surrogate training.
    u0 = 1.0 + 0.5 * np.cos(x)
    v0 = 0.5 * np.sin(x)
    y0 = np.concatenate([u0, v0])

    def rhs(t, y):
        u_re = y[:Nx]
        u_im = y[Nx:]
        mod_sq = u_re**2 + u_im**2
        u_re_xx = np.real(ifft(-k2 * fft(u_re)))
        u_im_xx = np.real(ifft(-k2 * fft(u_im)))
        du_re = (u_re_xx - a * u_im_xx
                 + u_re - mod_sq * u_re + b * mod_sq * u_im)
        du_im = (a * u_re_xx + u_im_xx
                 + u_im - b * mod_sq * u_re - mod_sq * u_im)
        return np.concatenate([du_re, du_im])

    Nt = int(np.sqrt(n_points))
    t_eval = np.linspace(0, t_max, Nt, dtype=np.float64)

    sol = _solve_ivp(rhs, [0, t_max], y0, t_eval=t_eval,
                     method="RK45", rtol=1e-10, atol=1e-12)
    if sol.status != 0:
        raise RuntimeError(f"C101 integration failed: {sol.message}")

    T_grid, X_grid = np.meshgrid(sol.t, x, indexing="ij")
    u_vals = sol.y[:Nx, :].T
    v_vals = sol.y[Nx:, :].T

    grid = np.column_stack([T_grid.ravel(), X_grid.ravel()])
    return grid, [u_vals.ravel(), v_vals.ravel()]


def _generate_c305_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate data for coupled Bose-Einstein condensates.

    i·∂ψ₁/∂t = −∂²ψ₁/∂x² + g₁|ψ₁|²ψ₁ + κψ₂
    i·∂ψ₂/∂t = −∂²ψ₂/∂x² + g₂|ψ₂|²ψ₂ + κψ₁

    Uses spectral method of lines (FFT for ∂²/∂x², RK45 for time).
    Asymmetric initial conditions for ψ₁ and ψ₂ break degeneracies.
    """
    from scipy.fft import fft, ifft
    from scipy.integrate import solve_ivp as _solve_ivp

    g1 = params["g1"]
    g2 = params["g2"]
    kappa = params["kappa"]

    Nx = 64
    L = 2 * np.pi
    x = np.linspace(0, L, Nx, endpoint=False, dtype=np.float64)
    k_freq = np.fft.fftfreq(Nx, d=L / Nx) * 2 * np.pi
    k2 = k_freq**2

    # Asymmetric initial conditions with equal amplitudes but different
    # spatial modes.  Both |ψ₁|² and |ψ₂|² ∈ [0.49, 1.69], ensuring
    # balanced surrogate training while breaking ψ₁/ψ₂ symmetry.
    u1_0 = 1.0 + 0.3 * np.cos(x)
    v1_0 = 0.3 * np.sin(x)
    u2_0 = 1.0 + 0.3 * np.cos(2 * x)
    v2_0 = 0.3 * np.sin(2 * x)
    y0 = np.concatenate([u1_0, v1_0, u2_0, v2_0])

    def rhs(t, y):
        u1 = y[0 * Nx:1 * Nx]
        v1 = y[1 * Nx:2 * Nx]
        u2 = y[2 * Nx:3 * Nx]
        v2 = y[3 * Nx:4 * Nx]
        mod1_sq = u1**2 + v1**2
        mod2_sq = u2**2 + v2**2
        # Spatial second derivatives via FFT
        u1_xx = np.real(ifft(-k2 * fft(u1)))
        v1_xx = np.real(ifft(-k2 * fft(v1)))
        u2_xx = np.real(ifft(-k2 * fft(u2)))
        v2_xx = np.real(ifft(-k2 * fft(v2)))
        # Real decomposition of i·∂ψ/∂t = −∂²ψ/∂x² + g|ψ|²ψ + κψ_other
        du1 = -v1_xx + g1 * mod1_sq * v1 + kappa * v2
        dv1 = u1_xx - g1 * mod1_sq * u1 - kappa * u2
        du2 = -v2_xx + g2 * mod2_sq * v2 + kappa * v1
        dv2 = u2_xx - g2 * mod2_sq * u2 - kappa * u1
        return np.concatenate([du1, dv1, du2, dv2])

    Nt = int(np.sqrt(n_points))
    t_eval = np.linspace(0, t_max, Nt, dtype=np.float64)

    sol = _solve_ivp(rhs, [0, t_max], y0, t_eval=t_eval,
                     method="RK45", rtol=1e-10, atol=1e-12)
    if sol.status != 0:
        raise RuntimeError(f"C305 integration failed: {sol.message}")

    T_grid, X_grid = np.meshgrid(sol.t, x, indexing="ij")
    u1_vals = sol.y[0 * Nx:1 * Nx, :].T
    v1_vals = sol.y[1 * Nx:2 * Nx, :].T
    u2_vals = sol.y[2 * Nx:3 * Nx, :].T
    v2_vals = sol.y[3 * Nx:4 * Nx, :].T

    grid = np.column_stack([T_grid.ravel(), X_grid.ravel()])
    return grid, [u1_vals.ravel(), v1_vals.ravel(),
                  u2_vals.ravel(), v2_vals.ravel()]


def _generate_c204_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """C204: RLC phasor — algebraic I(ω) = V / Z(ω).

    Here t_max is repurposed as omega_max.
    """
    L = params["L"]
    R = params["R"]
    C_cap = params["C_cap"]
    V = params["V"]

    omega = np.linspace(0.5, t_max, n_points, dtype=np.float64)
    X = omega * L - 1.0 / (omega * C_cap)  # reactance
    denom = R**2 + X**2
    u = V * R / denom       # Re(I)
    v = -V * X / denom      # Im(I)

    grid = omega.reshape(-1, 1)  # (N, 1)
    return grid, [u, v]


def _generate_c102_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """C102: Manakov system (coupled NLS).

    i∂u/∂z + ∂²u/∂t² + (|u|²+|v|²)u = 0
    i∂v/∂z + ∂²v/∂t² + (|u|²+|v|²)v = 0

    Real decomposition (u=u₁+iu₂, v=v₁+iv₂, M=|u|²+|v|²):
      ∂u₁/∂z = −∂²u₂/∂t² − M·u₂
      ∂u₂/∂z =  ∂²u₁/∂t² + M·u₁
      ∂v₁/∂z = −∂²v₂/∂t² − M·v₂
      ∂v₂/∂z =  ∂²v₁/∂t² + M·v₁

    Spectral method of lines (FFT for ∂²/∂t², RK45 for z-stepping).
    """
    from scipy.fft import fft, ifft
    from scipy.integrate import solve_ivp as _solve_ivp

    Nx = 64
    L = 2 * np.pi
    t_arr = np.linspace(0, L, Nx, endpoint=False, dtype=np.float64)
    k_freq = np.fft.fftfreq(Nx, d=L / Nx) * 2 * np.pi
    k2 = k_freq**2

    # Smaller amplitudes for smoother z-evolution, different spatial modes
    # to break |u|²/|v|² collinearity.
    u1_0 = 0.5 + 0.2 * np.cos(t_arr)
    u2_0 = 0.2 * np.sin(t_arr)
    v1_0 = 0.4 + 0.2 * np.cos(2 * t_arr)
    v2_0 = 0.2 * np.sin(2 * t_arr)
    y0 = np.concatenate([u1_0, u2_0, v1_0, v2_0])

    def rhs(z, y):
        u1 = y[0 * Nx:1 * Nx]
        u2 = y[1 * Nx:2 * Nx]
        v1 = y[2 * Nx:3 * Nx]
        v2 = y[3 * Nx:4 * Nx]
        M = u1**2 + u2**2 + v1**2 + v2**2  # |u|² + |v|²
        # ∂²/∂t² via FFT
        u1_tt = np.real(ifft(-k2 * fft(u1)))
        u2_tt = np.real(ifft(-k2 * fft(u2)))
        v1_tt = np.real(ifft(-k2 * fft(v1)))
        v2_tt = np.real(ifft(-k2 * fft(v2)))
        return np.concatenate([
            -u2_tt - M * u2,   # du1/dz
             u1_tt + M * u1,   # du2/dz
            -v2_tt - M * v2,   # dv1/dz
             v1_tt + M * v1,   # dv2/dz
        ])

    Nz = int(np.sqrt(n_points))
    z_eval = np.linspace(0, t_max, Nz, dtype=np.float64)

    sol = _solve_ivp(rhs, [0, t_max], y0, t_eval=z_eval,
                     method="RK45", rtol=1e-10, atol=1e-12)
    if sol.status != 0:
        raise RuntimeError(f"C102 integration failed: {sol.message}")

    Z_grid, T_grid = np.meshgrid(sol.t, t_arr, indexing="ij")  # (Nz, Nx)
    u1_vals = sol.y[0 * Nx:1 * Nx, :].T
    u2_vals = sol.y[1 * Nx:2 * Nx, :].T
    v1_vals = sol.y[2 * Nx:3 * Nx, :].T
    v2_vals = sol.y[3 * Nx:4 * Nx, :].T

    grid = np.column_stack([Z_grid.ravel(), T_grid.ravel()])
    return grid, [u1_vals.ravel(), u2_vals.ravel(),
                  v1_vals.ravel(), v2_vals.ravel()]


def _generate_c001_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate harmonic oscillator Schrödinger data from eigenstates.

    Uses a superposition of the first 4 eigenstates (n=0..3) so that
    D2U and x²·U columns are not collinear.  Analytic solution:
    ψ(x,t) = Σₙ cₙ·φₙ(x)·exp(−i·Eₙ·t/ℏ).
    """
    from scipy.special import hermite
    from math import factorial

    hbar = params["hbar"]
    m = params["m"]
    omega = params["omega"]

    xi_scale = np.sqrt(m * omega / hbar)  # √(mω/ℏ)
    prefactor = (m * omega / (np.pi * hbar)) ** 0.25

    n_side = int(np.sqrt(n_points))
    t_1d = np.linspace(0, t_max, n_side, dtype=np.float64)
    # Domain wide enough for several eigenstates to be non-negligible
    x_1d = np.linspace(-3.0, 3.0, n_side, dtype=np.float64)
    T, X = np.meshgrid(t_1d, x_1d, indexing="ij")

    xi = xi_scale * X
    gauss = np.exp(-0.5 * xi**2)

    # Superposition coefficients (normalised doesn't matter for PDE form)
    c_n = [1.0, 0.6, 0.3, 0.15]

    psi_re = np.zeros_like(T)
    psi_im = np.zeros_like(T)
    for n, cn in enumerate(c_n):
        Hn = hermite(n)(xi)
        norm_n = prefactor / np.sqrt(2**n * factorial(n))
        phi_n = norm_n * Hn * gauss
        E_n = hbar * omega * (n + 0.5)
        phase = -E_n * T / hbar  # −Eₙt/ℏ
        psi_re += cn * phi_n * np.cos(phase)
        psi_im += cn * phi_n * np.sin(phase)

    grid = np.column_stack([T.ravel(), X.ravel()])
    return grid, [psi_re.ravel(), psi_im.ravel()]


def _generate_c007_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate hydrogen radial Schrödinger data.

    Solutions are spherical Bessel functions:
      u(r) = j₀(kr) = sin(kr)/(kr)   (regular at origin)
      v(r) = y₀(kr) = −cos(kr)/(kr)  (singular at origin)

    Domain starts at r_min=1.0 to avoid the strongest 1/r singularity.
    """
    hbar = params["hbar"]
    m = params["m"]
    V_eff = params["V_eff"]
    E = params["E"]
    k = np.sqrt(2 * m * (E - V_eff) / hbar**2)

    r_min = 1.0
    r = np.linspace(r_min, t_max, n_points, dtype=np.float64)
    u = np.sin(k * r) / (k * r)     # j₀(kr)
    v = -np.cos(k * r) / (k * r)    # y₀(kr)
    grid = r.reshape(-1, 1)
    return grid, [u, v]


def _generate_c006_data(
    params: Dict[str, float], t_max: float, n_points: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate time-independent Schrödinger eigenstate data.

    Solution: ψ(x) = exp(ikx) with k = sqrt(2m(E−V)/ℏ²).
    Real/imag: u(x) = cos(kx), v(x) = sin(kx).
    """
    hbar = params["hbar"]
    m = params["m"]
    V = params["V"]
    E = params["E"]
    k = np.sqrt(2 * m * (E - V) / hbar**2)

    x = np.linspace(0, t_max, n_points, dtype=np.float64)
    u = np.cos(k * x)
    v = np.sin(k * x)
    grid = x.reshape(-1, 1)
    return grid, [u, v]


PDE_DATA_GENERATORS: Dict[
    str,
    Callable[[Dict[str, float], float, int], Tuple[np.ndarray, List[np.ndarray]]],
] = {
    "C000": _generate_c000_data,
    "C002": _generate_c002_data,
    "C003": _generate_c003_data,
    "C008": _generate_c008_data,
    "C100": _generate_c100_data,
    "C103": _generate_c103_data,
    "C301": _generate_c301_data,
    "C304": _generate_c304_data,
    "C101": _generate_c101_data,
    "C305": _generate_c305_data,
    "C204": _generate_c204_data,
    "C102": _generate_c102_data,
    "C001": _generate_c001_data,
    "C006": _generate_c006_data,
    "C007": _generate_c007_data,
}


# ---------------------------------------------------------------------------
# Ground-truth term lists.  Each ``_gt_terms_cXXX`` enumerates the exact AST terms
# of the known governing equation for problem CXXX; the
# ``_build_cXXX_ground_truth`` validators below pair these terms with their
# true coefficients to score recovery.  These are ground-truth definitions,
# not a discovery library.
# ---------------------------------------------------------------------------


def _mod_sq_times_u(out_idx: int, re_idx: int = 0, im_idx: int = 1) -> Node:
    """|ψ|² * u[out_idx] where |ψ|² = u[re_idx]² + u[im_idx]²."""
    return Mul(
        Add(Pow(U(out_idx=re_idx), 2), Pow(U(out_idx=im_idx), 2)),
        U(out_idx=out_idx),
    )


def _gt_terms_c203() -> List[Node]:
    """Library for C203: u, v, |A|^2*u, |A|^2*v."""
    return [
        U(out_idx=0),           # u
        U(out_idx=1),           # v
        _mod_sq_times_u(0),     # |A|^2 * u
        _mod_sq_times_u(1),     # |A|^2 * v
    ]


def _gt_terms_c303() -> List[Node]:
    """Library for C303: u, v, |A|^2*u, |A|^2*v (same structure as C203)."""
    return [
        U(out_idx=0),           # u
        U(out_idx=1),           # v
        _mod_sq_times_u(0),     # |A|^2 * u
        _mod_sq_times_u(1),     # |A|^2 * v
    ]


def _gt_terms_c004() -> List[Node]:
    """Library for C004: u1, v1, u2, v2 (linear terms only)."""
    return [U(out_idx=0), U(out_idx=1), U(out_idx=2), U(out_idx=3)]


def _gt_terms_c005() -> List[Node]:
    """Library for C005: u1, v1, u2, v2 (linear terms only)."""
    return [U(out_idx=0), U(out_idx=1), U(out_idx=2), U(out_idx=3)]


def _gt_terms_c202() -> List[Node]:
    """Library for C202: u1, v1, u2, v2 (linear terms only)."""
    return [U(out_idx=0), U(out_idx=1), U(out_idx=2), U(out_idx=3)]


def _gt_terms_c104() -> List[Node]:
    """Library for C104: bilinear products of u1, v1, u2, v2."""
    return [
        Mul(U(out_idx=0), U(out_idx=3)),   # u1*v2
        Mul(U(out_idx=1), U(out_idx=2)),   # v1*u2
        Mul(U(out_idx=0), U(out_idx=2)),   # u1*u2
        Mul(U(out_idx=1), U(out_idx=3)),   # v1*v2
        Mul(U(out_idx=0), U(out_idx=1)),   # u1*v1
        Pow(U(out_idx=0), 2),              # u1^2
        Pow(U(out_idx=1), 2),              # v1^2
    ]


def _gt_terms_c000() -> List[Node]:
    """Library for C000: second spatial derivatives u_xx and v_xx."""
    return [
        D2U(1, 1, out_idx=0),   # ∂²u/∂x²
        D2U(1, 1, out_idx=1),   # ∂²v/∂x²
    ]


def _gt_terms_c001() -> List[Node]:
    """Library for C001: D2U + x²·field for harmonic oscillator Schrödinger."""
    x_sq = Pow(Var(1), 2)  # x²  (Var(1) = spatial coordinate)
    return [
        D2U(1, 1, out_idx=0),       # ∂²u/∂x²
        D2U(1, 1, out_idx=1),       # ∂²v/∂x²
        Mul(x_sq, U(out_idx=0)),    # x²·u
        Mul(x_sq, U(out_idx=1)),    # x²·v
    ]


def _gt_terms_c300_eq0() -> List[Node]:
    """Per-equation library for C300 eq0 (u-equation)."""
    return [U(out_idx=0), _mod_sq_times_u(0)]


def _gt_terms_c300_eq1() -> List[Node]:
    """Per-equation library for C300 eq1 (v-equation)."""
    return [U(out_idx=1), _mod_sq_times_u(1)]


def _gt_terms_c006() -> List[Node]:
    """Library for C006: field values only (eigenvalue equation)."""
    return [
        U(out_idx=0),   # u
        U(out_idx=1),   # v
    ]


def _gt_terms_c007() -> List[Node]:
    """Library for C007 (shared, for skip-check): all 4 terms."""
    inv_r = Pow(Var(0), -1)  # 1/r
    return [
        Mul(inv_r, DU(0, out_idx=0)),   # (1/r)·du/dr
        Mul(inv_r, DU(0, out_idx=1)),   # (1/r)·dv/dr
        U(out_idx=0),                    # u
        U(out_idx=1),                    # v
    ]


def _gt_terms_c002() -> List[Node]:
    """Library for C002: spatial second derivatives + field values."""
    return [
        D2U(1, 1, out_idx=0),   # ∂²u/∂x²
        D2U(1, 1, out_idx=1),   # ∂²v/∂x²
        U(out_idx=0),            # u
        U(out_idx=1),            # v
    ]


def _gt_terms_c003() -> List[Node]:
    """Library for C003: spatial second derivatives + |ψ|²·u, |ψ|²·v."""
    return [
        D2U(1, 1, out_idx=0),   # ∂²u/∂x²
        D2U(1, 1, out_idx=1),   # ∂²v/∂x²
        _mod_sq_times_u(0),     # |ψ|²·u
        _mod_sq_times_u(1),     # |ψ|²·v
    ]


def _gt_terms_c301() -> List[Node]:
    """Library for C301: D2U + U + |Φ|²·Φ (all three term types)."""
    return [
        D2U(1, 1, out_idx=0),   # ∂²u/∂x²
        D2U(1, 1, out_idx=1),   # ∂²v/∂x²
        U(out_idx=0),            # u
        U(out_idx=1),            # v
        _mod_sq_times_u(0),     # |Φ|²·u
        _mod_sq_times_u(1),     # |Φ|²·v
    ]


def _gt_terms_c304() -> List[Node]:
    """Library for C304: spatial second derivatives + field values."""
    return [
        D2U(1, 1, out_idx=0),   # ∂²u/∂x²
        D2U(1, 1, out_idx=1),   # ∂²v/∂x²
        U(out_idx=0),            # u
        U(out_idx=1),            # v
    ]


def _gt_terms_c008() -> List[Node]:
    """Library for C008: spatial derivatives + field values (8 terms)."""
    return [
        DU(1, out_idx=0),   # ∂u₁/∂x
        DU(1, out_idx=1),   # ∂v₁/∂x
        DU(1, out_idx=2),   # ∂u₂/∂x
        DU(1, out_idx=3),   # ∂v₂/∂x
        U(out_idx=0),       # u₁
        U(out_idx=1),       # v₁
        U(out_idx=2),       # u₂
        U(out_idx=3),       # v₂
    ]


def _gt_terms_c201() -> List[Node]:
    """Library for C201 (full, kept for reference): all 6 terms."""
    return [
        U(out_idx=0),           # u
        U(out_idx=1),           # v
        DU(0, out_idx=0),       # du/dt
        DU(0, out_idx=1),       # dv/dt
        Cos(Var(0)),            # cos(t)
        Sin(Var(0)),            # sin(t)
    ]


def _gt_terms_c204() -> List[Node]:
    """Library for C204: ω·u, ω·v, u/ω, v/ω.

    From iωLI + RI + I/(iωC) = V, the real system with anchor order=0 gives
    constant-coefficient equations in these four terms (plus constant).
    """
    inv_omega = Pow(Var(0), -1)  # 1/ω
    return [
        Mul(Var(0), U(out_idx=0)),      # ω·u
        Mul(Var(0), U(out_idx=1)),      # ω·v
        Mul(inv_omega, U(out_idx=0)),   # u/ω
        Mul(inv_omega, U(out_idx=1)),   # v/ω
    ]


def _gt_terms_c105() -> List[Node]:
    """Library for C105: field × trig(z) products.

    Uses Delta=1 so sin(Δz) = sin(z) = sin(Var(0)).
    """
    return [
        Mul(U(out_idx=0), Sin(Var(0))),   # u·sin(z)
        Mul(U(out_idx=0), Cos(Var(0))),   # u·cos(z)
        Mul(U(out_idx=1), Sin(Var(0))),   # v·sin(z)
        Mul(U(out_idx=1), Cos(Var(0))),   # v·cos(z)
    ]


def _gt_terms_c302() -> List[Node]:
    """Library for C302: Josephson junction — phi, V, sin(phi)."""
    return [
        U(out_idx=0),           # phi
        U(out_idx=1),           # V
        Sin(U(out_idx=0)),      # sin(phi)
    ]


def _gt_terms_c102() -> List[Node]:
    """Library for C102: Manakov system (12 terms).

    4 spatial Laplacians ∂²/∂t² + 4 self-modulus |u|²·comp + 4 cross-modulus |v|²·comp.
    """
    return [
        # Spatial Laplacians ∂²/∂t²
        D2U(1, 1, out_idx=0),  # ∂²u₁/∂t²
        D2U(1, 1, out_idx=1),  # ∂²u₂/∂t²
        D2U(1, 1, out_idx=2),  # ∂²v₁/∂t²
        D2U(1, 1, out_idx=3),  # ∂²v₂/∂t²
        # |u|²·comp_i (self-modulus for u-field)
        _mod_sq_times_u(0, re_idx=0, im_idx=1),  # |u|²·u₁
        _mod_sq_times_u(1, re_idx=0, im_idx=1),  # |u|²·u₂
        _mod_sq_times_u(2, re_idx=0, im_idx=1),  # |u|²·v₁
        _mod_sq_times_u(3, re_idx=0, im_idx=1),  # |u|²·v₂
        # |v|²·comp_i (cross-modulus for v-field)
        _mod_sq_times_u(0, re_idx=2, im_idx=3),  # |v|²·u₁
        _mod_sq_times_u(1, re_idx=2, im_idx=3),  # |v|²·u₂
        _mod_sq_times_u(2, re_idx=2, im_idx=3),  # |v|²·v₁
        _mod_sq_times_u(3, re_idx=2, im_idx=3),  # |v|²·v₂
    ]


def _gt_terms_c305() -> List[Node]:
    """Library for C305: coupled BEC (12 terms).

    4 spatial D2U + 4 self-interaction |ψₖ|²·uₖ + 4 coupling U.
    """
    return [
        D2U(1, 1, out_idx=0),              # ∂²u₁/∂x²
        D2U(1, 1, out_idx=1),              # ∂²v₁/∂x²
        D2U(1, 1, out_idx=2),              # ∂²u₂/∂x²
        D2U(1, 1, out_idx=3),              # ∂²v₂/∂x²
        _mod_sq_times_u(0, re_idx=0, im_idx=1),  # |ψ₁|²·u₁
        _mod_sq_times_u(1, re_idx=0, im_idx=1),  # |ψ₁|²·v₁
        _mod_sq_times_u(2, re_idx=2, im_idx=3),  # |ψ₂|²·u₂
        _mod_sq_times_u(3, re_idx=2, im_idx=3),  # |ψ₂|²·v₂
        U(out_idx=0),                       # u₁ (coupling)
        U(out_idx=1),                       # v₁ (coupling)
        U(out_idx=2),                       # u₂ (coupling)
        U(out_idx=3),                       # v₂ (coupling)
    ]


# Override for the anchor derivative order along x_axis (time).
# For ODEs this equals problem.order.  For PDEs the spatial order may
# differ from the temporal order (e.g. Schrödinger: 1st in time, 2nd in space).
ANCHOR_ORDER: Dict[str, int] = {
    "C000": 1,
    "C002": 1,
    "C003": 1,
    "C100": 1,
    "C103": 1,
    "C301": 1,
    "C008": 1,
    "C101": 1,
    "C305": 1,
    "C204": 0,  # algebraic (no derivatives)
    "C102": 1,  # 1st order in z (propagation), 2nd order in t (spatial)
    "C001": 1,  # 1st order in time, 2nd order in space
    "C006": 2,  # 2nd order eigenvalue equation in x
    "C007": 2,  # 2nd order radial equation in r
}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass
class ComplexGroundTruth:
    """Ground truth for system DE validation.

    ``system_coeffs`` maps ``(eq_idx, term_repr)`` to expected coefficient
    in the convention ``anchor + sum c_k * term_k = 0``.
    """

    order: int
    system_coeffs: Dict[Tuple[int, str], float]
    coeff_rtol: float = 0.10
    coeff_atol: float = 0.05
    decoy_atol: float = 0.05


def _build_c200_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C200: d2u/dt2 + 2*gamma*du/dt + omega0^2*u = 0 (decoupled u/v).

    System discovery convention: U(out_idx=0) -> "u", U(out_idx=1) -> "u1",
    DU(0, out_idx=0) -> "u_x0", DU(0, out_idx=1) -> "u1_x0".
    """
    gamma = params["gamma"]
    omega0 = params["omega0"]
    omega0_sq = omega0**2

    return ComplexGroundTruth(
        order=2,
        system_coeffs={
            (0, "u"): omega0_sq,         # +omega0^2 * u
            (0, "u_x0"): 2 * gamma,     # +2*gamma * du/dt
            (1, "u1"): omega0_sq,        # +omega0^2 * v
            (1, "u1_x0"): 2 * gamma,    # +2*gamma * dv/dt
        },
    )


def _build_c203_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C203: dA/dt = (mu - |A|^2)*A + i*omega*A.

    System form (anchor = du/dt):
      eq0: du/dt + (-mu)*u + omega*v + |A|^2*u = 0
      eq1: dv/dt + (-omega)*u + (-mu)*v + |A|^2*v = 0
    """
    mu = params["mu"]
    omega = params["omega"]

    lib = _gt_terms_c203()
    u0_r, u1_r, msqu_r, msqv_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, u0_r): -mu,
            (0, u1_r): omega,
            (0, msqu_r): 1.0,
            (1, u0_r): -omega,
            (1, u1_r): -mu,
            (1, msqv_r): 1.0,
        },
    )


def _build_c303_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C303: dA/dt = (mu+i*omega)*A - (1+i*beta)*|A|^2*A.

    System form (anchor = du/dt):
      eq0: du/dt + (-mu)*u + omega*v + |A|^2*u + (-beta)*|A|^2*v = 0
      eq1: dv/dt + (-omega)*u + (-mu)*v + beta*|A|^2*u + |A|^2*v = 0
    """
    mu = params["mu"]
    omega = params["omega"]
    beta = params["beta"]

    lib = _gt_terms_c303()
    u0_r, u1_r, msqu_r, msqv_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, u0_r): -mu,
            (0, u1_r): omega,
            (0, msqu_r): 1.0,
            (0, msqv_r): -beta,
            (1, u0_r): -omega,
            (1, u1_r): -mu,
            (1, msqu_r): beta,
            (1, msqv_r): 1.0,
        },
    )


def _build_c004_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C004: Two-level quantum system.

    anchor = du_k/dt, convention: anchor + Σ c_k·term_k = 0.
    eq0: du1/dt - E1·v1 - V12·v2 = 0   →  {U(1): -E1, U(3): -V12}
    eq1: dv1/dt + E1·u1 + V12·u2 = 0   →  {U(0): E1, U(2): V12}
    eq2: du2/dt - V21·v1 - E2·v2 = 0   →  {U(1): -V21, U(3): -E2}
    eq3: dv2/dt + V21·u1 + E2·u2 = 0   →  {U(0): V21, U(2): E2}
    """
    E1 = params["E1"]
    E2 = params["E2"]
    V12 = params["V12"]
    V21 = params["V21"]

    lib = _gt_terms_c004()
    u0_r, u1_r, u2_r, u3_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, u1_r): -E1,
            (0, u3_r): -V12,
            (1, u0_r): E1,
            (1, u2_r): V12,
            (2, u1_r): -V21,
            (2, u3_r): -E2,
            (3, u0_r): V21,
            (3, u2_r): E2,
        },
    )


def _build_c005_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C005: Pauli spin-1/2.

    Same structure as C004 with E_up = p²/(2m)+Bz, E_dn = p²/(2m)-Bz, coupling = Bx.
    """
    m = params["m"]
    p = params["p"]
    Bx = params["Bx"]
    Bz = params["Bz"]
    E_up = p**2 / (2 * m) + Bz
    E_dn = p**2 / (2 * m) - Bz

    lib = _gt_terms_c005()
    u0_r, u1_r, u2_r, u3_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, u1_r): -E_up,
            (0, u3_r): -Bx,
            (1, u0_r): E_up,
            (1, u2_r): Bx,
            (2, u1_r): -Bx,
            (2, u3_r): -E_dn,
            (3, u0_r): Bx,
            (3, u2_r): E_dn,
        },
    )


def _build_c202_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C202: Coupled complex modes dz₁/dt = iω₁z₁ + κz₂.

    anchor = du_k/dt:
    eq0: du1/dt + ω₁v₁ - κu₂ = 0   →  {U(1): ω₁, U(2): -κ}
    eq1: dv1/dt - ω₁u₁ - κv₂ = 0   →  {U(0): -ω₁, U(3): -κ}
    eq2: du2/dt - κu₁ + ω₂v₂ = 0   →  {U(0): -κ, U(3): ω₂}
    eq3: dv2/dt - κv₁ - ω₂u₂ = 0   →  {U(1): -κ, U(2): -ω₂}
    """
    omega1 = params["omega1"]
    omega2 = params["omega2"]
    kappa = params["kappa"]

    lib = _gt_terms_c202()
    u0_r, u1_r, u2_r, u3_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, u1_r): omega1,
            (0, u2_r): -kappa,
            (1, u0_r): -omega1,
            (1, u3_r): -kappa,
            (2, u0_r): -kappa,
            (2, u3_r): omega2,
            (3, u1_r): -kappa,
            (3, u2_r): -omega2,
        },
    )


def _build_c104_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C104: Second harmonic generation.

    i·dA₁/dz = κ·conj(A₁)·A₂; i·dA₂/dz = κ·A₁²
    anchor = du_k/dz:
    eq0: du1/dz - κ(u1v2 - v1u2) = 0   →  {u1v2: -κ, v1u2: κ}
    eq1: dv1/dz + κ(u1u2 + v1v2) = 0   →  {u1u2: κ, v1v2: κ}
    eq2: du2/dz - 2κ·u1v1 = 0          →  {u1v1: -2κ}
    eq3: dv2/dz + κ(u1² - v1²) = 0     →  {u1²: κ, v1²: -κ}
    """
    kappa = params["kappa"]

    lib = _gt_terms_c104()
    # lib = [u1*v2, v1*u2, u1*u2, v1*v2, u1*v1, u1^2, v1^2]
    reprs = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, reprs[0]): -kappa,      # u1*v2
            (0, reprs[1]): kappa,       # v1*u2
            (1, reprs[2]): kappa,       # u1*u2
            (1, reprs[3]): kappa,       # v1*v2
            (2, reprs[4]): -2 * kappa,  # u1*v1
            (3, reprs[5]): kappa,       # u1^2
            (3, reprs[6]): -kappa,      # v1^2
        },
    )


def _build_c000_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C000: free-particle Schrödinger (plane wave).

    Real decomposition (ψ = u + iv, α = ℏ/(2m)):
      ∂u/∂t = −α·∂²v/∂x²
      ∂v/∂t =  α·∂²u/∂x²

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u/∂t + α·∂²v/∂x² = 0
      eq1: ∂v/∂t − α·∂²u/∂x² = 0
    """
    hbar = params["hbar"]
    m = params["m"]
    alpha = hbar / (2 * m)

    lib = _gt_terms_c000()
    u_xx_r, v_xx_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, v_xx_r): alpha,      # +α·∂²v/∂x²
            (1, u_xx_r): -alpha,     # −α·∂²u/∂x²
        },
    )


def _build_c300_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C300: BCS gap equation (Ginzburg-Landau relaxation).

    dΔ/dT = g·(N₀ − |Δ|²)·Δ

    Real decomposition:
      du/dT = g·N₀·u − g·|Δ|²·u
      dv/dT = g·N₀·v − g·|Δ|²·v

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: du/dT + (−g·N₀)·u + g·|Δ|²·u = 0
      eq1: dv/dT + (−g·N₀)·v + g·|Δ|²·v = 0
    """
    g = params["g"]
    N0 = params["N0"]

    # Use per-equation library terms for repr strings
    lib_eq0 = _gt_terms_c300_eq0()
    u_r, msqu_r = [repr(t) for t in lib_eq0]
    lib_eq1 = _gt_terms_c300_eq1()
    v_r, msqv_r = [repr(t) for t in lib_eq1]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, u_r): -g * N0,      # −g·N₀
            (0, msqu_r): g,          # +g
            (1, v_r): -g * N0,      # −g·N₀
            (1, msqv_r): g,          # +g
        },
    )


def _build_c007_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C007: hydrogen radial Schrödinger.

    −(ℏ²/(2m))·(R'' + (2/r)·R') + V_eff·R = E·R
    →  R'' + (2/r)·R' + k²·R = 0  with k² = 2m(E−V_eff)/ℏ²

    u and v decouple:
      eq0: d²u/dr² + 2·(1/r)·du/dr + k²·u = 0
      eq1: d²v/dr² + 2·(1/r)·dv/dr + k²·v = 0
    """
    hbar = params["hbar"]
    m = params["m"]
    V_eff = params["V_eff"]
    E = params["E"]
    k_sq = 2 * m * (E - V_eff) / hbar**2

    lib = _gt_terms_c007()
    inv_r_du_r, inv_r_dv_r, u_r, v_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=2,
        system_coeffs={
            (0, inv_r_du_r): 2.0,    # +(2/r)·du/dr
            (0, u_r): k_sq,          # +k²·u
            (1, inv_r_dv_r): 2.0,    # +(2/r)·dv/dr
            (1, v_r): k_sq,          # +k²·v
        },
    )


def _build_c006_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C006: time-independent Schrödinger eigenvalue.

    −(ℏ²/(2m))·d²ψ/dx² + V·ψ = E·ψ  →  d²ψ/dx² + k²·ψ = 0

    u and v decouple (same real equation):
      eq0: d²u/dx² + k²·u = 0
      eq1: d²v/dx² + k²·v = 0

    where k² = 2m(E−V)/ℏ².
    """
    hbar = params["hbar"]
    m = params["m"]
    V = params["V"]
    E = params["E"]
    k_sq = 2 * m * (E - V) / hbar**2

    lib = _gt_terms_c006()
    u_r, v_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=2,
        system_coeffs={
            (0, u_r): k_sq,    # +k²·u
            (1, v_r): k_sq,    # +k²·v
        },
    )


def _build_c001_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C001: harmonic oscillator Schrödinger.

    iℏ·∂ψ/∂t = −(ℏ²/(2m))·∂²ψ/∂x² + ½mω²x²ψ

    Real decomposition (α = ℏ/(2m), β = mω²/(2ℏ)):
      ∂u/∂t = −α·∂²v/∂x² + β·x²·v
      ∂v/∂t =  α·∂²u/∂x² − β·x²·u

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u/∂t + α·∂²v/∂x² − β·x²·v = 0
      eq1: ∂v/∂t − α·∂²u/∂x² + β·x²·u = 0
    """
    hbar = params["hbar"]
    m = params["m"]
    omega = params["omega"]
    alpha = hbar / (2 * m)
    beta = m * omega**2 / (2 * hbar)

    lib = _gt_terms_c001()
    u_xx_r, v_xx_r, x2u_r, x2v_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, v_xx_r): alpha,      # +α·∂²v/∂x²
            (0, x2v_r): -beta,       # −β·x²·v
            (1, u_xx_r): -alpha,     # −α·∂²u/∂x²
            (1, x2u_r): beta,        # +β·x²·u
        },
    )


def _build_c003_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C003: Gross-Pitaevskii i·∂ψ/∂t = −∂²ψ/∂x² + g·|ψ|²·ψ.

    Real decomposition (ψ = u + iv):
      ∂u/∂t = −∂²v/∂x² + g·|ψ|²·v
      ∂v/∂t =  ∂²u/∂x² − g·|ψ|²·u

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u/∂t + ∂²v/∂x² − g·|ψ|²·v = 0
      eq1: ∂v/∂t − ∂²u/∂x² + g·|ψ|²·u = 0
    """
    g = params["g"]

    lib = _gt_terms_c003()
    u_xx_r, v_xx_r, msqu_r, msqv_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, v_xx_r): 1.0,       # +∂²v/∂x²
            (0, msqv_r): -g,        # −g·|ψ|²·v
            (1, u_xx_r): -1.0,      # −∂²u/∂x²
            (1, msqu_r): g,         # +g·|ψ|²·u
        },
    )


def _build_c304_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C304: complex Klein-Gordon ∂²φ/∂t² − c²·∂²φ/∂x² + m²·φ = 0.

    u and v decouple (same real equation independently).
    STLSQ convention (anchor = ∂²u/∂t², anchor + Σcₖ·termₖ = 0):
      eq0: ∂²u/∂t² − c²·∂²u/∂x² + m²·u = 0
      eq1: ∂²v/∂t² − c²·∂²v/∂x² + m²·v = 0
    """
    c_val = params["c"]
    m = params["m"]

    lib = _gt_terms_c304()
    u_xx_r, v_xx_r, u_r, v_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=2,
        system_coeffs={
            (0, u_xx_r): -(c_val**2),  # −c²·∂²u/∂x²
            (0, u_r): m**2,             # +m²·u
            (1, v_xx_r): -(c_val**2),  # −c²·∂²v/∂x²
            (1, v_r): m**2,             # +m²·v
        },
    )


def _build_c002_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C002: 1D Schrödinger with constant potential V.

    Real decomposition (ψ = u + iv, α = ℏ/(2m), β = V/ℏ):
      ∂u/∂t = −α·∂²v/∂x² − β·v
      ∂v/∂t =  α·∂²u/∂x² + β·u

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u/∂t + α·∂²v/∂x² − β·v = 0
      eq1: ∂v/∂t − α·∂²u/∂x² + β·u = 0
    """
    hbar = params["hbar"]
    m = params["m"]
    V = params["V"]
    alpha = hbar / (2 * m)
    beta = V / hbar

    lib = _gt_terms_c002()
    u_xx_r, v_xx_r, u_r, v_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, v_xx_r): alpha,     # +α·∂²v/∂x²
            (0, v_r): -beta,        # −β·v
            (1, u_xx_r): -alpha,    # −α·∂²u/∂x²
            (1, u_r): beta,         # +β·u
        },
    )


def _build_c301_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C301: TDGL ∂Φ/∂t = −a·Φ − b·|Φ|²·Φ + κ·∂²Φ/∂x².

    No i factor, so u and v have the same structure (coupled through |Φ|²).

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u/∂t + a·u + b·|Φ|²·u − κ·∂²u/∂x² = 0
      eq1: ∂v/∂t + a·v + b·|Φ|²·v − κ·∂²v/∂x² = 0
    """
    a = params["a"]
    b_val = params["b"]
    kappa = params["kappa"]

    lib = _gt_terms_c301()
    u_xx_r, v_xx_r, u_r, v_r, msqu_r, msqv_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, u_xx_r): -kappa,    # −κ·∂²u/∂x²
            (0, u_r): a,            # +a·u
            (0, msqu_r): b_val,     # +b·|Φ|²·u
            (1, v_xx_r): -kappa,    # −κ·∂²v/∂x²
            (1, v_r): a,            # +a·v
            (1, msqv_r): b_val,     # +b·|Φ|²·v
        },
    )


def _build_c100_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C100: focusing NLS (fiber optics) i·du/dz + d²u/dt² + |u|²u = 0.

    Axes: (z, t), x_axis=0 (z).  Anchor = ∂u/∂z (1st order).

    Real decomposition (u = u_re + i·u_im):
      ∂u_re/∂z = −∂²u_im/∂t² − |u|²·u_im
      ∂u_im/∂z =  ∂²u_re/∂t² + |u|²·u_re

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u_re/∂z + ∂²u_im/∂t² + |u|²·u_im = 0
      eq1: ∂u_im/∂z − ∂²u_re/∂t² − |u|²·u_re = 0
    """
    lib = _gt_terms_c003()
    u_xx_r, v_xx_r, msqu_r, msqv_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, v_xx_r): 1.0,       # +∂²u_im/∂t²
            (0, msqv_r): 1.0,       # +|u|²·u_im
            (1, u_xx_r): -1.0,      # −∂²u_re/∂t²
            (1, msqu_r): -1.0,      # −|u|²·u_re
        },
    )


def _build_c103_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C103: NLS soliton envelope i·dA/dt + d²A/dx² + 2|A|²A = 0.

    Axes: (t, x), x_axis=0 (t).  Anchor = ∂A/∂t (1st order).

    Real decomposition (A = u + iv):
      ∂u/∂t = −∂²v/∂x² − 2|A|²·v
      ∂v/∂t =  ∂²u/∂x² + 2|A|²·u

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u/∂t + ∂²v/∂x² + 2|A|²·v = 0
      eq1: ∂v/∂t − ∂²u/∂x² − 2|A|²·u = 0
    """
    lib = _gt_terms_c003()
    u_xx_r, v_xx_r, msqu_r, msqv_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, v_xx_r): 1.0,       # +∂²v/∂x²
            (0, msqv_r): 2.0,       # +2|A|²·v
            (1, u_xx_r): -1.0,      # −∂²u/∂x²
            (1, msqu_r): -2.0,      # −2|A|²·u
        },
    )


def _build_c008_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C008: 1+1D Dirac equation (two-component complex spinor).

    i·∂ψ₁/∂t + i·∂ψ₁/∂x = m·ψ₂;  i·∂ψ₂/∂t − i·∂ψ₂/∂x = m·ψ₁

    Real decomposition (ψ₁ = u₁+iv₁, ψ₂ = u₂+iv₂):
      ∂u₁/∂t = −∂u₁/∂x + m·v₂
      ∂v₁/∂t = −∂v₁/∂x − m·u₂
      ∂u₂/∂t =  ∂u₂/∂x + m·v₁
      ∂v₂/∂t =  ∂v₂/∂x − m·u₁

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u₁/∂t + ∂u₁/∂x − m·v₂ = 0
      eq1: ∂v₁/∂t + ∂v₁/∂x + m·u₂ = 0
      eq2: ∂u₂/∂t − ∂u₂/∂x − m·v₁ = 0
      eq3: ∂v₂/∂t − ∂v₂/∂x + m·u₁ = 0
    """
    m_val = params["m"]

    lib = _gt_terms_c008()
    reprs = [repr(t) for t in lib]
    # reprs: [du1_x, dv1_x, du2_x, dv2_x, u1, v1, u2, v2]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, reprs[0]): 1.0,       # +∂u₁/∂x
            (0, reprs[7]): -m_val,    # −m·v₂
            (1, reprs[1]): 1.0,       # +∂v₁/∂x
            (1, reprs[6]): m_val,     # +m·u₂
            (2, reprs[2]): -1.0,      # −∂u₂/∂x
            (2, reprs[5]): -m_val,    # −m·v₁
            (3, reprs[3]): -1.0,      # −∂v₂/∂x
            (3, reprs[4]): m_val,     # +m·u₁
        },
    )


def _build_c101_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C101: Complex Ginzburg-Landau du/dt = (1+ia)d²u/dx² + u − (1+ib)|u|²u.

    Real decomposition (u = u_re + i·u_im):
      ∂u_re/∂t = ∂²u_re/∂x² − a·∂²u_im/∂x² + u_re − |u|²·u_re + b·|u|²·u_im
      ∂u_im/∂t = a·∂²u_re/∂x² + ∂²u_im/∂x² + u_im − b·|u|²·u_re − |u|²·u_im

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u_re/∂t − ∂²u_re/∂x² + a·∂²u_im/∂x² − u_re + |u|²·u_re − b·|u|²·u_im = 0
      eq1: ∂u_im/∂t − a·∂²u_re/∂x² − ∂²u_im/∂x² − u_im + b·|u|²·u_re + |u|²·u_im = 0
    """
    a = params["a"]
    b = params["b"]

    lib = _gt_terms_c301()
    u_xx_r, v_xx_r, u_r, v_r, msqu_r, msqv_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, u_xx_r): -1.0,      # −∂²u_re/∂x²
            (0, v_xx_r): a,         # +a·∂²u_im/∂x²
            (0, u_r): -1.0,         # −u_re
            (0, msqu_r): 1.0,       # +|u|²·u_re
            (0, msqv_r): -b,        # −b·|u|²·u_im
            (1, u_xx_r): -a,        # −a·∂²u_re/∂x²
            (1, v_xx_r): -1.0,      # −∂²u_im/∂x²
            (1, v_r): -1.0,         # −u_im
            (1, msqu_r): b,         # +b·|u|²·u_re
            (1, msqv_r): 1.0,       # +|u|²·u_im
        },
    )


def _build_c305_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C305: Coupled BEC.

    i·∂ψ₁/∂t = −∂²ψ₁/∂x² + g₁|ψ₁|²ψ₁ + κψ₂
    i·∂ψ₂/∂t = −∂²ψ₂/∂x² + g₂|ψ₂|²ψ₂ + κψ₁

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: ∂u₁/∂t + ∂²v₁/∂x² − g₁|ψ₁|²v₁ − κv₂ = 0
      eq1: ∂v₁/∂t − ∂²u₁/∂x² + g₁|ψ₁|²u₁ + κu₂ = 0
      eq2: ∂u₂/∂t + ∂²v₂/∂x² − g₂|ψ₂|²v₂ − κv₁ = 0
      eq3: ∂v₂/∂t − ∂²u₂/∂x² + g₂|ψ₂|²u₂ + κu₁ = 0
    """
    g1 = params["g1"]
    g2 = params["g2"]
    kappa = params["kappa"]

    lib = _gt_terms_c305()
    reprs = [repr(t) for t in lib]
    # reprs: [d2u1_xx, d2v1_xx, d2u2_xx, d2v2_xx,
    #         |ψ₁|²u₁, |ψ₁|²v₁, |ψ₂|²u₂, |ψ₂|²v₂,
    #         u₁, v₁, u₂, v₂]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, reprs[1]): 1.0,       # +∂²v₁/∂x²
            (0, reprs[5]): -g1,       # −g₁|ψ₁|²v₁
            (0, reprs[11]): -kappa,   # −κv₂
            (1, reprs[0]): -1.0,      # −∂²u₁/∂x²
            (1, reprs[4]): g1,        # +g₁|ψ₁|²u₁
            (1, reprs[10]): kappa,    # +κu₂
            (2, reprs[3]): 1.0,       # +∂²v₂/∂x²
            (2, reprs[7]): -g2,       # −g₂|ψ₂|²v₂
            (2, reprs[9]): -kappa,    # −κv₁
            (3, reprs[2]): -1.0,      # −∂²u₂/∂x²
            (3, reprs[6]): g2,        # +g₂|ψ₂|²u₂
            (3, reprs[8]): kappa,     # +κu₁
        },
    )


def _build_c201_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C201: Driven complex oscillator d²z/dt² + 2γ·dz/dt + ω₀²·z = F·exp(iΩt).

    Real decomposition (z = u + iv, Ω=1):
      d²u/dt² = −2γ·du/dt − ω₀²·u + F·cos(t)
      d²v/dt² = −2γ·dv/dt − ω₀²·v + F·sin(t)

    STLSQ convention (anchor = d²u/dt²):
      eq0: d²u/dt² + 2γ·du/dt + ω₀²·u − F·cos(t) = 0
      eq1: d²v/dt² + 2γ·dv/dt + ω₀²·v − F·sin(t) = 0
    """
    gamma = params["gamma"]
    omega0 = params["omega0"]
    F = params["F"]

    lib = _gt_terms_c201()
    reprs = [repr(t) for t in lib]
    # reprs: [u, v, du/dt, dv/dt, cos(t), sin(t)]

    return ComplexGroundTruth(
        order=2,
        system_coeffs={
            (0, reprs[0]): omega0**2,    # +ω₀²·u
            (0, reprs[2]): 2 * gamma,    # +2γ·du/dt
            (0, reprs[4]): -F,           # −F·cos(t)
            (1, reprs[1]): omega0**2,    # +ω₀²·v
            (1, reprs[3]): 2 * gamma,    # +2γ·dv/dt
            (1, reprs[5]): -F,           # −F·sin(t)
        },
    )


def _build_c105_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C105: Parametric amplification i·dA/dz = κ·conj(A)·exp(iΔz).

    Real decomposition (A = u + iv, Δ=1):
      du/dz = κ·u·sin(z) − κ·v·cos(z)
      dv/dz = −κ·u·cos(z) − κ·v·sin(z)

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: du/dz − κ·u·sin(z) + κ·v·cos(z) = 0
      eq1: dv/dz + κ·u·cos(z) + κ·v·sin(z) = 0
    """
    kappa = params["kappa"]

    lib = _gt_terms_c105()
    reprs = [repr(t) for t in lib]
    # reprs: [u·sin(z), u·cos(z), v·sin(z), v·cos(z)]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, reprs[0]): -kappa,     # −κ·u·sin(z)
            (0, reprs[3]): kappa,      # +κ·v·cos(z)
            (1, reprs[1]): kappa,      # +κ·u·cos(z)
            (1, reprs[2]): kappa,      # +κ·v·sin(z)
        },
    )


def _build_c302_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C302: Josephson junction dphi/dt = V, dV/dt = -sin(phi) - alpha*V + I_ext.

    STLSQ convention (anchor + Σcₖ·termₖ = 0):
      eq0: dphi/dt − V = 0
      eq1: dV/dt − I_ext + sin(phi) + alpha*V = 0
    """
    alpha = params["alpha"]
    I_ext = params["I_ext"]

    lib = _gt_terms_c302()
    phi_r, V_r, sinphi_r = [repr(t) for t in lib]

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            (0, V_r): -1.0,            # −V
            (1, "const"): -I_ext,      # −I_ext
            (1, sinphi_r): 1.0,        # +sin(phi)
            (1, V_r): alpha,           # +alpha*V
        },
    )


def _build_c204_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C204: RLC phasor iωLI + RI + I/(iωC) = V.

    Real decomposition I = u + iv:
      R·u − X·v = V   with X = ωL − 1/(ωC)
      X·u + R·v = 0

    STLSQ convention (anchor = u for eq0, v for eq1, order=0):
      eq0: u − (L/R)·(ω·v) + (1/(RC))·(v/ω) − V/R = 0
      eq1: v + (L/R)·(ω·u) − (1/(RC))·(u/ω) = 0
    """
    L = params["L"]
    R = params["R"]
    C_cap = params["C_cap"]
    V = params["V"]

    lib = _gt_terms_c204()
    reprs = [repr(t) for t in lib]
    # reprs: [ω·u, ω·v, u/ω, v/ω]

    return ComplexGroundTruth(
        order=0,
        system_coeffs={
            (0, reprs[1]): -L / R,        # −(L/R)·(ω·v)
            (0, reprs[3]): 1.0 / (R * C_cap),  # +(1/(RC))·(v/ω)
            (0, "const"): -V / R,          # −V/R
            (1, reprs[0]): L / R,          # +(L/R)·(ω·u)
            (1, reprs[2]): -1.0 / (R * C_cap),  # −(1/(RC))·(u/ω)
        },
    )


def _build_c102_ground_truth(params: Dict[str, float]) -> ComplexGroundTruth:
    """C102: Manakov system i∂u/∂z + ∂²u/∂t² + (|u|²+|v|²)u = 0 (same for v).

    Real decomposition (u=u₁+iu₂, v=v₁+iv₂, M=|u|²+|v|²):
      ∂u₁/∂z + ∂²u₂/∂t² + M·u₂ = 0
      ∂u₂/∂z − ∂²u₁/∂t² − M·u₁ = 0
      ∂v₁/∂z + ∂²v₂/∂t² + M·v₂ = 0
      ∂v₂/∂z − ∂²v₁/∂t² − M·v₁ = 0

    Split M·comp = |u|²·comp + |v|²·comp (both coefficient 1).
    """
    lib = _gt_terms_c102()
    reprs = [repr(t) for t in lib]
    # 0-3: D2U(1,1,i) for i=0..3
    # 4-7: |u|²·comp_i for i=0..3
    # 8-11: |v|²·comp_i for i=0..3

    return ComplexGroundTruth(
        order=1,
        system_coeffs={
            # eq0: ∂u₁/∂z + ∂²u₂/∂t² + |u|²·u₂ + |v|²·u₂ = 0
            (0, reprs[1]): 1.0,   # +∂²u₂/∂t²
            (0, reprs[5]): 1.0,   # +|u|²·u₂
            (0, reprs[9]): 1.0,   # +|v|²·u₂
            # eq1: ∂u₂/∂z − ∂²u₁/∂t² − |u|²·u₁ − |v|²·u₁ = 0
            (1, reprs[0]): -1.0,  # −∂²u₁/∂t²
            (1, reprs[4]): -1.0,  # −|u|²·u₁
            (1, reprs[8]): -1.0,  # −|v|²·u₁
            # eq2: ∂v₁/∂z + ∂²v₂/∂t² + |u|²·v₂ + |v|²·v₂ = 0
            (2, reprs[3]): 1.0,   # +∂²v₂/∂t²
            (2, reprs[7]): 1.0,   # +|u|²·v₂
            (2, reprs[11]): 1.0,  # +|v|²·v₂
            # eq3: ∂v₂/∂z − ∂²v₁/∂t² − |u|²·v₁ − |v|²·v₁ = 0
            (3, reprs[2]): -1.0,  # −∂²v₁/∂t²
            (3, reprs[6]): -1.0,  # −|u|²·v₁
            (3, reprs[10]): -1.0, # −|v|²·v₁
        },
    )


GROUND_TRUTH_BUILDERS: Dict[str, Callable[[Dict[str, float]], ComplexGroundTruth]] = {
    "C200": _build_c200_ground_truth,
    "C203": _build_c203_ground_truth,
    "C303": _build_c303_ground_truth,
    "C004": _build_c004_ground_truth,
    "C005": _build_c005_ground_truth,
    "C202": _build_c202_ground_truth,
    "C104": _build_c104_ground_truth,
    "C000": _build_c000_ground_truth,
    "C002": _build_c002_ground_truth,
    "C304": _build_c304_ground_truth,
    "C003": _build_c003_ground_truth,
    "C100": _build_c100_ground_truth,
    "C103": _build_c103_ground_truth,
    "C301": _build_c301_ground_truth,
    "C008": _build_c008_ground_truth,
    "C101": _build_c101_ground_truth,
    "C305": _build_c305_ground_truth,
    "C302": _build_c302_ground_truth,
    "C105": _build_c105_ground_truth,
    "C201": _build_c201_ground_truth,
    "C204": _build_c204_ground_truth,
    "C102": _build_c102_ground_truth,
    "C001": _build_c001_ground_truth,
    "C006": _build_c006_ground_truth,
    "C007": _build_c007_ground_truth,
    "C300": _build_c300_ground_truth,
}
