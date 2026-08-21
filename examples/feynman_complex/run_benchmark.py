#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Feynman Complex-Valued DE Benchmark Runner.

Discovers complex-valued DEs by decomposing them into real N-component
systems.  Two discovery engines are exposed:

- ``sparse``: train one NestyNet surrogate per real component and run the
  shared system-DE STLSQ solver on the combined surrogate.
- ``factorized_search``: build explicit feature tables and run factorized
  symbolic search independently for each equation.

The factorized path uses the column-agnostic ``run_explorer_core`` machinery:
it sees an ``(N, nvars)`` feature matrix and discovers expression trees over
``Var(0)...Var(nvars-1)``.  For coupled multi-component systems we build a
feature table with ALL component values (and derivatives) as columns, then run
the explorer once per equation with different targets.

Supports:
- ODEs (nxvars=1): multi-trajectory data via ``solve_ivp``
- PDEs (nxvars=2): analytic data on regular grids
- 2-component and 4-component systems
- 1st and 2nd order anchor derivatives
- Non-autonomous problems (C105, C201)
- Algebraic (C204) and eigenvalue (C006, C007) special cases

Usage::

    python examples/feynman_complex/run_benchmark.py --engine sparse --only C000 --fast
    python examples/feynman_complex/run_benchmark.py --engine factorized_search --only C200 --fast --verbose
    python examples/feynman_complex/run_benchmark.py --engine sparse --all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, NamedTuple, Sequence, Tuple

import numpy as np
import torch
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from problem_defs import (  # noqa: E402
    ANCHOR_ORDER,
    ComplexGroundTruth,
    ComplexProblemDef,
    DEFAULT_ICS,
    DEFAULT_PARAMS,
    DEFAULT_TMAX,
    GROUND_TRUTH_BUILDERS,
    NCOMPONENTS,
    NXVARS,
    PDE_DATA_GENERATORS,
    RHS_REGISTRY,
    get_canonical_problem_dims,
    get_complex_problem_dims,
    load_complex_problems,
)

from nestynet_sr.sr_core import build_initial_ast  # noqa: E402
from nestynet_sr.sr_core.bridges import (  # noqa: E402
    Add,
    AddNode,
    AtomNode,
    Cos,
    CosNode,
    D2U,
    DU,
    Mul,
    MulNode,
    Node,
    Pow,
    PowNode,
    Sin,
    SinNode,
    U,
    Var,
)
from nestynet_sr.sr_core.problem_dims import (  # noqa: E402
    CanonicalProblemDims,
    canonical_component_target_dims,
    canonical_constant_dims,
    complex_pairs_for_problem_dims,
    default_complex_pairs,
    dim_add,
    dim_eq,
    dim_scale,
    dim_sub,
    dimless_dim,
    is_dimless,
    parameterized_trig_specs as _shared_parameterized_trig_specs,
)
from nestynet_sr.sr_de import (  # noqa: E402
    factorized_search_candidate_to_feature_predictor,
    evaluate_factorized_search_candidate,
    normalized_rmse,
)
from nestynet_sr.sr_de.de_search import ridge_lstsq, stlsq  # noqa: E402
from nestynet_sr.sr_de.system_de_search import (  # noqa: E402
    SystemDESearchConfig,
    SystemDESearchResult,
    build_system_residual_asts,
    discover_system_de_from_surrogate,
)
from nestynet_sr.sr_search.factorized_search.explorer import (  # noqa: E402
    node_size,
    node_str,
    run_explorer_core,
)
from nestynet_sr.sr_search.config import DataHyperparams, ModelHyperparams  # noqa: E402
from nestynet_sr.sr_search.data_utils import build_datasets  # noqa: E402
from nestynet_sr.sr_search.model_builders import LeafBuilder, build_composite_ast  # noqa: E402
from nestynet_sr.sr_search.training import train_initial_model  # noqa: E402

BENCHMARK_FILE = REPO_ROOT / "data" / "feynman_complex_benchmark.txt"

# Problems whose RHS depends explicitly on the independent variable.
NON_AUTONOMOUS = {"C105", "C201"}

# Solver retry strategies for ODE integration.
_SOLVER_TRIALS: tuple[tuple[str, float, float], ...] = (
    ("RK45", 1e-10, 1e-12),
    ("Radau", 1e-8, 1e-10),
    ("BDF", 1e-8, 1e-10),
)
_MAX_IC_RETRIES = 20

# Component label names used for sparse-engine CSV filenames.
_COMP_LABELS = ["u", "v", "u2", "v2", "u3", "v3", "u4", "v4"]

DimVec = Tuple[float, ...]

_dim_add = dim_add
_dim_sub = dim_sub
_dim_scale = dim_scale
_dim_eq = dim_eq
_dimless = dimless_dim
_is_dimless = is_dimless
_default_complex_pairs = default_complex_pairs


def _normalize_engine_name(engine: str) -> str:
    key = str(engine).strip().lower().replace("-", "_")
    aliases = {
        "stlsq": "sparse",
        "sparse_stlsq": "sparse",
        "system_stlsq": "sparse",
        "factorized": "factorized_search",
        "fss": "factorized_search",
        "factorized_symbolic_search": "factorized_search",
    }
    return aliases.get(key, key)


def _summary_name_for_engine(engine: str) -> str:
    return "summary.json" if _normalize_engine_name(engine) == "sparse" else "summary_factorized_search.json"


def _complex_pairs_for_problem(problem_dims: CanonicalProblemDims | None, ncomp: int) -> tuple[tuple[int, int], ...]:
    return complex_pairs_for_problem_dims(problem_dims, n_components=int(ncomp))


def _constant_feature_dims(
    problem_dims: CanonicalProblemDims | None,
    params: dict[str, float],
) -> list[tuple[str, DimVec]]:
    return canonical_constant_dims(problem_dims, tuple(str(name) for name in params), default_dimless=True)


def _constant_feature_arrays(
    n_rows: int,
    problem_dims: CanonicalProblemDims | None,
    params: dict[str, float],
) -> tuple[list[np.ndarray], list[str], list[DimVec] | None]:
    if problem_dims is None:
        return [], [], None
    cols: list[np.ndarray] = []
    names: list[str] = []
    dims: list[DimVec] = []
    for name, dim in _constant_feature_dims(problem_dims, params):
        cols.append(np.full(int(n_rows), float(params[name]), dtype=np.float64))
        names.append(str(name))
        dims.append(tuple(dim))
    return cols, names, dims


def _constant_feature_values(
    problem_dims: CanonicalProblemDims | None,
    params: dict[str, float],
) -> list[float]:
    if problem_dims is None:
        return []
    return [float(params[name]) for name, _ in _constant_feature_dims(problem_dims, params)]


def _parameterized_trig_specs(
    axis_dim: DimVec | None,
    problem_dims: CanonicalProblemDims | None,
    params: dict[str, float],
) -> list[tuple[str | None, float, str, str]]:
    return _shared_parameterized_trig_specs(
        axis_dim,
        constant_values={str(name): float(val) for name, val in dict(params).items()},
        constant_dims=None if problem_dims is None else problem_dims.constant_dims,
        x_label="x",
    )


def _target_dims_for_problem(
    problem_dims: CanonicalProblemDims | None,
    *,
    anchor_order: int,
    anchor_axis: int = 0,
) -> list[DimVec] | None:
    return canonical_component_target_dims(
        problem_dims,
        anchor_order=int(anchor_order),
        anchor_axis=int(anchor_axis),
    )


class _OdeFeatureContext(NamedTuple):
    problem_dims: CanonicalProblemDims | None
    component_dims: tuple[DimVec, ...] | None
    axis_dim: DimVec | None
    complex_pairs: tuple[tuple[int, int], ...]
    trig_specs: tuple[tuple[str | None, float, str, str], ...]
    target_dims: tuple[DimVec, ...] | None


def _make_ode_feature_context(
    pid: str,
    params: dict[str, float],
    ncomp: int,
    anchor_order: int,
) -> _OdeFeatureContext:
    problem_dims = get_canonical_problem_dims(pid)
    component_dims = (
        None
        if problem_dims is None
        else tuple(tuple(d) for d in problem_dims.component_dims[: int(ncomp)])
    )
    axis_dim = None if problem_dims is None else tuple(problem_dims.axis_dims[0])
    complex_pairs = _complex_pairs_for_problem(problem_dims, ncomp)
    trig_specs = tuple(_parameterized_trig_specs(axis_dim, problem_dims, params))
    target_dims_raw = _target_dims_for_problem(
        problem_dims,
        anchor_order=int(anchor_order),
        anchor_axis=0,
    )
    target_dims = None if target_dims_raw is None else tuple(tuple(d) for d in target_dims_raw)
    return _OdeFeatureContext(
        problem_dims=problem_dims,
        component_dims=component_dims,
        axis_dim=axis_dim,
        complex_pairs=complex_pairs,
        trig_specs=trig_specs,
        target_dims=target_dims,
    )


def _build_ode_feature_columns(
    pid: str,
    params: dict[str, float],
    t: np.ndarray,
    state: np.ndarray,
    ncomp: int,
    order: int,
    anchor_order: int,
    *,
    ctx: _OdeFeatureContext,
) -> tuple[list[np.ndarray], list[str], list[DimVec] | None]:
    is_non_autonomous = pid in NON_AUTONOMOUS
    cols: list[np.ndarray] = []
    names: list[str] = []
    dims: list[DimVec] | None = [] if ctx.problem_dims is not None else None

    if is_non_autonomous:
        cols.append(t)
        names.append("t")
        if dims is not None and ctx.axis_dim is not None:
            dims.append(tuple(ctx.axis_dim))
        for _const_name, scale, trig_kind, trig_name in ctx.trig_specs:
            cols.append(np.sin(scale * t) if trig_kind == "sin" else np.cos(scale * t))
            names.append(str(trig_name))
            if dims is not None and ctx.problem_dims is not None:
                dims.append(_dimless(len(ctx.problem_dims.basis)))

    for ci in range(ncomp):
        cols.append(state[ci])
        names.append(f"u{ci}")
        if dims is not None and ctx.component_dims is not None:
            dims.append(tuple(ctx.component_dims[ci]))

    if order == 2 and anchor_order == 2:
        for ci in range(ncomp):
            cols.append(state[ncomp + ci])
            names.append(f"du{ci}/dt")
            if dims is not None and ctx.component_dims is not None and ctx.axis_dim is not None:
                dims.append(_dim_sub(tuple(ctx.component_dims[ci]), tuple(ctx.axis_dim)))

    if order == 1 and ncomp >= 2:
        trig_cols, trig_names, trig_dims = _component_trig_feature_arrays(
            state,
            ncomp,
            component_dims=ctx.component_dims,
        )
        cols.extend(trig_cols)
        names.extend(trig_names)
        if dims is not None and trig_dims is not None:
            dims.extend(trig_dims)

        prod_cols, prod_names, prod_dims = _component_pairwise_product_arrays(
            [state[ci] for ci in range(int(ncomp))],
            ncomp,
            component_dims=ctx.component_dims,
        )
        cols.extend(prod_cols)
        names.extend(prod_names)
        if dims is not None and prod_dims is not None:
            dims.extend(prod_dims)

        inv_cols, inv_names, inv_dims = _complex_invariant_feature_arrays(
            state,
            ncomp,
            pairs=ctx.complex_pairs,
            component_dims=ctx.component_dims,
        )
        cols.extend(inv_cols)
        names.extend(inv_names)
        if dims is not None and inv_dims is not None:
            dims.extend(inv_dims)

        if is_non_autonomous:
            trig_cols, trig_names, trig_dims = _nonautonomous_trig_feature_arrays(
                t,
                state,
                ncomp,
                problem_dims=ctx.problem_dims,
                params=params,
            )
            cols.extend(trig_cols)
            names.extend(trig_names)
            if dims is not None and trig_dims is not None:
                dims.extend(trig_dims)

    const_cols, const_names, const_dims = _constant_feature_arrays(
        t.shape[0],
        ctx.problem_dims,
        params,
    )
    cols.extend(const_cols)
    names.extend(const_names)
    if dims is not None and const_dims is not None:
        dims.extend(const_dims)

    return cols, names, dims


def _build_ode_feature_values(
    pid: str,
    params: dict[str, float],
    t: float,
    state: Sequence[float],
    ncomp: int,
    order: int,
    anchor_order: int,
    *,
    ctx: _OdeFeatureContext,
) -> list[float]:
    is_non_autonomous = pid in NON_AUTONOMOUS
    feats: list[float] = []

    if is_non_autonomous:
        feats.append(float(t))
        for _const_name, scale, trig_kind, _trig_name in ctx.trig_specs:
            feats.append(math.sin(scale * float(t)) if trig_kind == "sin" else math.cos(scale * float(t)))

    for ci in range(ncomp):
        feats.append(float(state[ci]))

    if order == 2 and anchor_order == 2:
        for ci in range(ncomp):
            feats.append(float(state[ncomp + ci]))

    if order == 1 and ncomp >= 2:
        feats.extend(
            _component_trig_feature_values(
                state,
                ncomp,
                component_dims=ctx.component_dims,
            )
        )
        feats.extend(_complex_pairwise_product_values(state, ncomp))
        feats.extend(
            _complex_invariant_feature_values(
                state,
                ncomp,
                pairs=ctx.complex_pairs,
            )
        )
        if is_non_autonomous:
            feats.extend(
                _nonautonomous_trig_feature_values(
                    t,
                    state,
                    ncomp,
                    problem_dims=ctx.problem_dims,
                    params=params,
                )
            )

    feats.extend(_constant_feature_values(ctx.problem_dims, params))
    return feats


# ---------------------------------------------------------------------------
# Data generation – ODEs
# ---------------------------------------------------------------------------


def _problem_seed(base_seed: int, pid: str) -> int:
    acc = 0
    for c in str(pid):
        acc = (acc * 131 + ord(c)) % 2_147_483_647
    return int(base_seed) + acc


def _sample_ics(
    pid: str,
    ncomp: int,
    order: int,
    base_ics: dict[str, float],
    rng: np.random.Generator,
    ic_index: int,
) -> list[float]:
    """Return a list of initial-condition values for trajectory *ic_index*.

    For ic_index==0 the nominal ICs are returned.  For ic_index>0 each
    component is independently drawn from a range centred on the base
    value.  Large perturbations are essential for coupled/decoupled
    systems to break spurious cross-component correlations.
    """
    if order == 2:
        y0 = [
            base_ics.get("u0", 1.0),
            base_ics.get("v0", 0.0),
            base_ics.get("du0", 0.0),
            base_ics.get("dv0", 0.0),
        ]
    elif ncomp == 2:
        y0 = [base_ics.get("u0", 1.0), base_ics.get("v0", 0.0)]
    else:
        y0 = [
            base_ics.get("u1", 1.0),
            base_ics.get("v1", 0.0),
            base_ics.get("u2", 0.0),
            base_ics.get("v2", 0.0),
        ]
    if ic_index > 0:
        # Each component drawn independently to maximise linear independence.
        scale = max(abs(v) for v in y0) if any(y0) else 1.0
        scale = max(scale, 0.5)  # ensure minimum scale even when base is near zero
        y0 = [float(rng.uniform(-scale, scale)) for _ in y0]
    return y0


def generate_ode_multi_traj(
    pid: str,
    params: dict[str, float],
    ncomp: int,
    order: int,
    t_max: float,
    base_ics: dict[str, float],
    *,
    n_traj: int = 6,
    n_points: int = 5000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Integrate the ODE for *n_traj* different ICs.

    Returns a list of dicts, each with keys ``t``, ``state``, ``y0``.
    ``state`` has shape ``(n_state, n_points)`` matching ``solve_ivp.y``.
    """
    rhs_fn = RHS_REGISTRY[pid]
    rng = np.random.default_rng(seed)
    trajs: list[dict[str, Any]] = []

    for k in range(n_traj):
        t_eval = np.linspace(0.0, t_max, n_points, dtype=np.float64)

        for retry in range(_MAX_IC_RETRIES):
            y0 = _sample_ics(pid, ncomp, order, base_ics, rng, ic_index=k if retry == 0 else k + 100 + retry)
            solved = False

            for method, rtol, atol in _SOLVER_TRIALS:
                try:
                    sol = solve_ivp(
                        lambda t, s: rhs_fn(t, s, params),
                        [0.0, t_max],
                        y0,
                        t_eval=t_eval,
                        method=method,
                        rtol=rtol,
                        atol=atol,
                    )
                except Exception:
                    continue
                if sol.status != 0:
                    continue
                if not np.isfinite(sol.y).all():
                    continue
                solved = True
                break

            if solved:
                break
        else:
            raise RuntimeError(f"Integration failed for {pid} traj {k} after {_MAX_IC_RETRIES} retries")

        trajs.append({
            "t": np.asarray(sol.t, dtype=np.float64),
            "state": np.asarray(sol.y, dtype=np.float64),
            "y0": list(y0),
        })
    return trajs


# ---------------------------------------------------------------------------
# Data generation – PDEs
# ---------------------------------------------------------------------------


def generate_pde_data(
    pid: str,
    params: dict[str, float],
    t_max: float,
    n_points: int = 5000,
) -> dict[str, Any]:
    """Generate PDE data on a regular grid using the generator.

    The generators in :mod:`problem_defs` return flattened meshgrid data, but
    not all problems use *square* grids (some fix Nx and vary Nt).  Downstream
    derivative estimation must therefore use the *true* grid shape inferred
    from the returned coordinates.

    Returns dict with keys:
    - ``grid``: (N, nxvars) coordinate array
    - ``components``: list of ncomp arrays, each (N,)
    - ``shape``: inferred regular-grid shape, e.g. (Nt, Nx) for nxvars==2
    """
    gen_fn = PDE_DATA_GENERATORS[pid]
    grid, comp_arrays = gen_fn(params, t_max, n_points)
    grid = np.asarray(grid, dtype=np.float64)
    comp_arrays = [np.asarray(c, dtype=np.float64) for c in comp_arrays]

    if grid.ndim != 2:
        raise ValueError(f"{pid}: expected grid with shape (N, nxvars), got {grid.shape}")

    N = int(grid.shape[0])
    nxvars = int(grid.shape[1])
    for i, c in enumerate(comp_arrays):
        if c.shape[0] != N:
            raise ValueError(f"{pid}: component[{i}] has {c.shape[0]} points, expected {N}")

    # Infer the mesh shape from unique coordinates along each axis.
    # This is robust to rectangular grids (Nt != Nx) which occur in the
    # spectral-method generators.
    if nxvars == 1:
        shape = (N,)
    else:
        dims = [int(np.unique(grid[:, ax]).shape[0]) for ax in range(nxvars)]
        if int(np.prod(dims)) != N:
            raise ValueError(
                f"{pid}: expected a regular grid, but unique axis sizes {dims} give {int(np.prod(dims))} != N={N}"
            )
        shape = tuple(dims)
    return {
        "grid": grid,
        "components": comp_arrays,
        "shape": shape,
    }


# ---------------------------------------------------------------------------
# Feature table builders
# ---------------------------------------------------------------------------


def _compute_ode_derivatives(
    pid: str,
    params: dict[str, float],
    traj: dict[str, Any],
    ncomp: int,
    order: int,
) -> np.ndarray:
    """Evaluate the known RHS at each data point to get exact derivatives.

    Returns ``derivs`` of shape ``(n_points, n_state)`` where n_state
    matches the solve_ivp state size.  For 1st-order systems the
    derivatives are ``[du0/dt, dv0/dt, ...]``; for 2nd-order systems the
    full state derivative ``[du, dv, d2u, d2v]`` is returned.
    """
    rhs_fn = RHS_REGISTRY[pid]
    t = traj["t"]
    state = traj["state"]  # (n_state, N)
    N = t.shape[0]
    n_state = state.shape[0]
    derivs = np.empty((N, n_state), dtype=np.float64)
    for i in range(N):
        s_i = state[:, i].tolist()
        d_i = rhs_fn(float(t[i]), s_i, params)
        derivs[i, :] = d_i
    return derivs


def _compute_pde_spatial_derivs(
    grid: np.ndarray,
    components: list[np.ndarray],
    shape: tuple[int, ...],
    spatial_axis: int = 1,
    deriv_order: int = 2,
    periodic: bool = False,
) -> list[np.ndarray]:
    """Compute spatial derivatives on a regular grid.

    Supports first (``deriv_order=1``) and second (``deriv_order=2``)
    derivatives along the selected spatial axis.

    Returns a list of ncomp arrays, each of shape (N,).
    """
    ncomp = len(components)
    d2_list: list[np.ndarray] = []

    # Determine spatial coordinate spacing.
    # Grid columns: [t, x] for nxvars==2. spatial_axis==1 → column 1.
    grid_nd = grid[:, spatial_axis].reshape(*shape)
    n_along = int(shape[spatial_axis]) if spatial_axis < len(shape) else 0
    idx0 = (0,) * len(shape)
    idx1 = (0,) * spatial_axis + (1,) + (0,) * (len(shape) - spatial_axis - 1)
    dx = float(grid_nd[idx1] - grid_nd[idx0]) if n_along > 1 else 1.0
    if abs(dx) < 1e-30:
        dx = 1.0  # safety

    use_periodic = bool(periodic) and n_along >= 4
    if use_periodic:
        k = 2.0 * np.pi * np.fft.fftfreq(n_along, d=dx)
        k_shape = [1] * len(shape)
        k_shape[spatial_axis] = n_along
        K = k.reshape(k_shape)
        for ci in range(ncomp):
            u_grid = components[ci].reshape(*shape)
            U = np.fft.fft(u_grid, axis=spatial_axis)
            if int(deriv_order) == 1:
                du = np.fft.ifft(1j * K * U, axis=spatial_axis).real
                d2_list.append(du.ravel())
            else:
                d2u = np.fft.ifft(-(K**2) * U, axis=spatial_axis).real
                d2_list.append(d2u.ravel())
    else:
        edge_order = 2 if n_along >= 3 else 1
        for ci in range(ncomp):
            u_grid = components[ci].reshape(*shape)
            du_dx = np.gradient(u_grid, dx, axis=spatial_axis, edge_order=edge_order)
            if int(deriv_order) == 1:
                d2_list.append(du_dx.ravel())
            else:
                d2u_dx2 = np.gradient(du_dx, dx, axis=spatial_axis, edge_order=edge_order)
                d2_list.append(d2u_dx2.ravel())

    return d2_list


def _compute_pde_temporal_derivs(
    grid: np.ndarray,
    components: list[np.ndarray],
    shape: tuple[int, ...],
    temporal_axis: int = 0,
) -> list[np.ndarray]:
    """Compute first temporal derivatives ∂u_k/∂t on the regular grid.

    Returns a list of ncomp arrays, each of shape (N,).
    """
    ncomp = len(components)
    dt_list: list[np.ndarray] = []

    grid_nd = grid[:, temporal_axis].reshape(*shape)
    n_along = int(shape[temporal_axis]) if temporal_axis < len(shape) else 0
    idx0 = (0,) * len(shape)
    idx1 = (0,) * temporal_axis + (1,) + (0,) * (len(shape) - temporal_axis - 1)
    dt = float(grid_nd[idx1] - grid_nd[idx0]) if n_along > 1 else 1.0
    if abs(dt) < 1e-30:
        dt = 1.0

    edge_order = 2 if n_along >= 3 else 1

    for ci in range(ncomp):
        u_grid = components[ci].reshape(*shape)
        du_dt = np.gradient(u_grid, dt, axis=temporal_axis, edge_order=edge_order)
        dt_list.append(du_dt.ravel())

    return dt_list


def _compute_pde_second_temporal_derivs(
    grid: np.ndarray,
    components: list[np.ndarray],
    shape: tuple[int, ...],
    temporal_axis: int = 0,
) -> list[np.ndarray]:
    """Compute second temporal derivatives ∂²u_k/∂t² on the regular grid."""
    ncomp = len(components)
    d2t_list: list[np.ndarray] = []

    grid_nd = grid[:, temporal_axis].reshape(*shape)
    n_along = int(shape[temporal_axis]) if temporal_axis < len(shape) else 0
    idx0 = (0,) * len(shape)
    idx1 = (0,) * temporal_axis + (1,) + (0,) * (len(shape) - temporal_axis - 1)
    dt = float(grid_nd[idx1] - grid_nd[idx0]) if n_along > 1 else 1.0
    if abs(dt) < 1e-30:
        dt = 1.0

    edge_order = 2 if n_along >= 3 else 1

    for ci in range(ncomp):
        u_grid = components[ci].reshape(*shape)
        du = np.gradient(u_grid, dt, axis=temporal_axis, edge_order=edge_order)
        d2u = np.gradient(du, dt, axis=temporal_axis, edge_order=edge_order)
        d2t_list.append(d2u.ravel())

    return d2t_list


def _detect_periodic_spatial_axis(
    components: list[np.ndarray],
    shape: tuple[int, ...],
    spatial_axis: int,
) -> bool:
    """Heuristic periodic-boundary detector from endpoint continuity."""
    n_along = int(shape[spatial_axis]) if spatial_axis < len(shape) else 0
    if n_along < 16:
        return False

    rel_diffs: list[float] = []
    boundary_energy: list[float] = []
    for c in components:
        u = np.asarray(c, dtype=np.float64).reshape(*shape)
        start = np.take(u, 0, axis=spatial_axis)
        end = np.take(u, -1, axis=spatial_axis)
        num = float(np.linalg.norm(start - end))
        den = float(np.linalg.norm(start) + np.linalg.norm(end) + 1e-12)
        rel_diffs.append(num / den)
        boundary_energy.append(float((np.linalg.norm(start) + np.linalg.norm(end)) / (np.linalg.norm(u) + 1e-12)))

    # Need both continuity at boundaries and non-negligible boundary amplitude
    # (to avoid falsely classifying near-zero tails as periodic).
    rel_med = float(np.median(rel_diffs)) if rel_diffs else float("inf")
    bnd_med = float(np.median(boundary_energy)) if boundary_energy else 0.0
    return rel_med < 0.08 and bnd_med > 0.08


def _complex_pair_indices(
    ncomp: int,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Return (real, imag) component index pairs for complex fields."""
    if pairs is None:
        return list(_default_complex_pairs(int(ncomp)))
    return [(int(i), int(j)) for i, j in pairs if 0 <= int(i) < int(ncomp) and 0 <= int(j) < int(ncomp)]


def _complex_pairwise_product_arrays(
    state: np.ndarray,
    ncomp: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Return all component pairwise products ``u_i * u_j`` (i<=j)."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    for i in range(int(ncomp)):
        for j in range(i, int(ncomp)):
            cols.append(state[i] * state[j])
            names.append(f"u{i}*u{j}")
    return cols, names


def _complex_pairwise_product_values(state: Sequence[float], ncomp: int) -> list[float]:
    """Pointwise variant of :func:`_complex_pairwise_product_arrays`."""
    vals: list[float] = []
    for i in range(int(ncomp)):
        for j in range(i, int(ncomp)):
            vals.append(float(state[i]) * float(state[j]))
    return vals


def _complex_invariant_feature_arrays(
    state: np.ndarray,
    ncomp: int,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    component_dims: Sequence[DimVec] | None = None,
) -> tuple[list[np.ndarray], list[str], list[DimVec] | None]:
    """Build invariant features for first-order complex ODEs.

    For each complex pair (u, v), adds:
    - |A|^2
    - |A|^2 * u_j for all components j
    """
    cols: list[np.ndarray] = []
    names: list[str] = []
    dims: list[DimVec] | None = [] if component_dims is not None else None
    for pair_idx, (re_idx, im_idx) in enumerate(_complex_pair_indices(ncomp, pairs)):
        re = state[re_idx]
        im = state[im_idx]
        mod_sq = re * re + im * im
        cols.append(mod_sq)
        names.append(f"abs{pair_idx}2")
        if dims is not None:
            field_dim = tuple(component_dims[re_idx])
            dims.append(_dim_scale(field_dim, 2.0))
        for cj in range(int(ncomp)):
            cols.append(mod_sq * state[cj])
            names.append(f"abs{pair_idx}2*u{cj}")
            if dims is not None:
                dims.append(_dim_add(_dim_scale(field_dim, 2.0), tuple(component_dims[cj])))
    return cols, names, dims


def _complex_invariant_feature_values(
    state: Sequence[float],
    ncomp: int,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> list[float]:
    """Pointwise variant of :func:`_complex_invariant_feature_arrays`."""
    vals: list[float] = []
    for re_idx, im_idx in _complex_pair_indices(ncomp, pairs):
        re = float(state[re_idx])
        im = float(state[im_idx])
        mod_sq = re * re + im * im
        vals.append(mod_sq)
        for cj in range(int(ncomp)):
            vals.append(mod_sq * float(state[cj]))
    return vals


def _nonautonomous_trig_feature_arrays(
    t: np.ndarray,
    state: np.ndarray,
    ncomp: int,
    *,
    problem_dims: CanonicalProblemDims | None = None,
    params: dict[str, float] | None = None,
) -> tuple[list[np.ndarray], list[str], list[DimVec] | None]:
    """State-modulated trig features for non-autonomous ODEs."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    dims: list[DimVec] | None = [] if problem_dims is not None else None
    axis_dim = None if problem_dims is None else tuple(problem_dims.axis_dims[0])
    pvals = {} if params is None else dict(params)
    trig_specs = _parameterized_trig_specs(axis_dim, problem_dims, pvals)
    comp_dims = None if problem_dims is None else list(problem_dims.component_dims[: int(ncomp)])
    for const_name, scale, trig_kind, trig_name in trig_specs:
        carrier = np.sin(scale * t) if trig_kind == "sin" else np.cos(scale * t)
        for ci in range(int(ncomp)):
            cols.append(state[ci] * carrier)
            names.append(f"u{ci}*{trig_name}")
            if dims is not None and comp_dims is not None:
                dims.append(tuple(comp_dims[ci]))
    return cols, names, dims


def _nonautonomous_trig_feature_values(
    t: float,
    state: Sequence[float],
    ncomp: int,
    *,
    problem_dims: CanonicalProblemDims | None = None,
    params: dict[str, float] | None = None,
) -> list[float]:
    """Pointwise variant of :func:`_nonautonomous_trig_feature_arrays`."""
    vals: list[float] = []
    axis_dim = None if problem_dims is None else tuple(problem_dims.axis_dims[0])
    pvals = {} if params is None else dict(params)
    trig_specs = _parameterized_trig_specs(axis_dim, problem_dims, pvals)
    for _const_name, scale, trig_kind, _trig_name in trig_specs:
        trig_val = math.sin(scale * float(t)) if trig_kind == "sin" else math.cos(scale * float(t))
        for ci in range(int(ncomp)):
            vals.append(float(state[ci]) * trig_val)
    return vals


def _component_trig_feature_arrays(
    state: np.ndarray,
    ncomp: int,
    *,
    component_dims: Sequence[DimVec] | None = None,
) -> tuple[list[np.ndarray], list[str], list[DimVec] | None]:
    """Unary trig transforms of component values."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    dims: list[DimVec] | None = [] if component_dims is not None else None
    dim0 = None if component_dims is None else _dimless(len(component_dims[0]))
    for ci in range(int(ncomp)):
        if component_dims is not None and dim0 is not None and not _dim_eq(tuple(component_dims[ci]), dim0):
            continue
        cols.append(np.sin(state[ci]))
        names.append(f"sin(u{ci})")
        cols.append(np.cos(state[ci]))
        names.append(f"cos(u{ci})")
        if dims is not None and dim0 is not None:
            dims.extend([dim0, dim0])
    return cols, names, dims


def _component_trig_feature_values(
    state: Sequence[float],
    ncomp: int,
    *,
    component_dims: Sequence[DimVec] | None = None,
) -> list[float]:
    """Pointwise variant of :func:`_component_trig_feature_arrays`."""
    vals: list[float] = []
    dim0 = None if component_dims is None else _dimless(len(component_dims[0]))
    for ci in range(int(ncomp)):
        if component_dims is not None and dim0 is not None and not _dim_eq(tuple(component_dims[ci]), dim0):
            continue
        s = float(state[ci])
        vals.extend([math.sin(s), math.cos(s)])
    return vals


def _component_pairwise_product_arrays(
    components: list[np.ndarray],
    ncomp: int,
    *,
    component_dims: Sequence[DimVec] | None = None,
) -> tuple[list[np.ndarray], list[str], list[DimVec] | None]:
    """Pairwise products over component arrays (for PDE feature tables)."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    dims: list[DimVec] | None = [] if component_dims is not None else None
    for i in range(int(ncomp)):
        for j in range(i, int(ncomp)):
            cols.append(components[i] * components[j])
            names.append(f"u{i}*u{j}")
            if dims is not None:
                dims.append(_dim_add(tuple(component_dims[i]), tuple(component_dims[j])))
    return cols, names, dims


def _component_invariant_feature_arrays(
    components: list[np.ndarray],
    ncomp: int,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    component_dims: Sequence[DimVec] | None = None,
) -> tuple[list[np.ndarray], list[str], list[DimVec] | None]:
    """Complex-pair modulus features over component arrays (for PDE tables)."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    dims: list[DimVec] | None = [] if component_dims is not None else None
    for pair_idx, (re_idx, im_idx) in enumerate(_complex_pair_indices(ncomp, pairs)):
        re = components[re_idx]
        im = components[im_idx]
        mod_sq = re * re + im * im
        cols.append(mod_sq)
        names.append(f"abs{pair_idx}2")
        if dims is not None:
            field_dim = tuple(component_dims[re_idx])
            dims.append(_dim_scale(field_dim, 2.0))
        for cj in range(int(ncomp)):
            cols.append(mod_sq * components[cj])
            names.append(f"abs{pair_idx}2*u{cj}")
            if dims is not None:
                dims.append(_dim_add(_dim_scale(field_dim, 2.0), tuple(component_dims[cj])))
    return cols, names, dims


def _build_linear_expr_from_coeffs(
    coeffs: np.ndarray,
    intercept: float,
    *,
    coef_tol: float = 0.0,
) -> Any:
    """Build a node tree for ``intercept + Σ coeffs[i]*x_i``."""
    terms: list[Any] = []
    if abs(float(intercept)) > float(coef_tol):
        terms.append(("const", float(intercept)))
    for i, c in enumerate(np.asarray(coeffs, dtype=np.float64).tolist()):
        c = float(c)
        if abs(c) <= float(coef_tol):
            continue
        if abs(c - 1.0) < 1e-14:
            term = ("var", int(i))
        elif abs(c + 1.0) < 1e-14:
            term = ("mul", ("const", -1.0), ("var", int(i)))
        else:
            term = ("mul", ("const", c), ("var", int(i)))
        terms.append(term)
    if not terms:
        return ("const", 0.0)
    node = terms[0]
    for term in terms[1:]:
        node = ("add", node, term)
    return node


def _try_linear_prefit(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_probe: np.ndarray,
    y_probe: np.ndarray,
    *,
    max_terms: int = 24,
    rel_prune: float = 1e-6,
    var_dims: Sequence[DimVec] | None = None,
    y_dims: DimVec | None = None,
) -> dict[str, Any] | None:
    """Fit a sparse linear model in raw features and return an expression dict."""
    x_fit = np.asarray(x_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64).reshape(-1)
    x_probe = np.asarray(x_probe, dtype=np.float64)
    y_probe = np.asarray(y_probe, dtype=np.float64).reshape(-1)

    if x_fit.ndim != 2 or x_probe.ndim != 2:
        return None
    if x_fit.shape[1] == 0 or x_fit.shape[0] < 2:
        return None
    if y_fit.shape[0] != x_fit.shape[0] or y_probe.shape[0] != x_probe.shape[0]:
        return None

    nvars_total = int(x_fit.shape[1])
    eligible = np.arange(nvars_total, dtype=np.int64)
    allow_intercept = True
    y_dims_eff = None if y_dims is None else tuple(float(v) for v in y_dims)
    if var_dims is not None and y_dims_eff is not None:
        eligible = np.asarray(
            [
                i
                for i, dim in enumerate(var_dims)
                if dim is not None and _dim_eq(tuple(dim), y_dims_eff)
            ],
            dtype=np.int64,
        )
        allow_intercept = _is_dimless(y_dims_eff)
        if eligible.size == 0 and not allow_intercept:
            return None

    x_fit_eff = x_fit[:, eligible] if eligible.size > 0 else np.zeros((x_fit.shape[0], 0), dtype=np.float64)
    x_probe_eff = x_probe[:, eligible] if eligible.size > 0 else np.zeros((x_probe.shape[0], 0), dtype=np.float64)

    try:
        if allow_intercept:
            A_fit = np.column_stack([np.ones(x_fit_eff.shape[0], dtype=np.float64), x_fit_eff])
            sol = np.linalg.lstsq(A_fit, y_fit, rcond=None)[0]
            intercept = float(sol[0])
            coeffs_eff = np.asarray(sol[1:], dtype=np.float64)
        else:
            coeffs_eff = np.asarray(np.linalg.lstsq(x_fit_eff, y_fit, rcond=None)[0], dtype=np.float64)
            intercept = 0.0
    except Exception:
        return None
    if not np.isfinite(coeffs_eff).all() or not math.isfinite(float(intercept)):
        return None

    scale = float(np.max(np.abs(coeffs_eff))) if coeffs_eff.size > 0 else 0.0
    scale = max(scale, 1.0)
    tol = max(1e-12, float(rel_prune) * scale)
    keep = np.where(np.abs(coeffs_eff) > tol)[0]

    if keep.size > int(max_terms):
        order = np.argsort(np.abs(coeffs_eff[keep]))[::-1]
        keep = np.sort(keep[order[: int(max_terms)]])

    if keep.size > 0:
        try:
            if allow_intercept:
                A_sel = np.column_stack([np.ones(x_fit_eff.shape[0], dtype=np.float64), x_fit_eff[:, keep]])
                sol_sel = np.linalg.lstsq(A_sel, y_fit, rcond=None)[0]
                intercept = float(sol_sel[0])
                coeffs_sparse = np.zeros_like(coeffs_eff)
                coeffs_sparse[keep] = sol_sel[1:]
            else:
                sol_sel = np.linalg.lstsq(x_fit_eff[:, keep], y_fit, rcond=None)[0]
                intercept = 0.0
                coeffs_sparse = np.zeros_like(coeffs_eff)
                coeffs_sparse[keep] = sol_sel
        except Exception:
            return None
        if not np.isfinite(sol_sel).all():
            return None
        coeffs_eff = coeffs_sparse
    else:
        coeffs_eff = np.zeros_like(coeffs_eff)
        intercept = float(np.mean(y_fit)) if allow_intercept else 0.0

    coeffs = np.zeros(nvars_total, dtype=np.float64)
    if eligible.size > 0:
        coeffs[eligible] = coeffs_eff
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        yhat_probe = intercept + x_probe_eff @ coeffs_eff
    if not np.isfinite(yhat_probe).all():
        return None
    resid = y_probe - yhat_probe
    mse = float(np.mean(resid ** 2))
    if not math.isfinite(mse):
        return None
    nrmse = float(np.linalg.norm(resid) / (np.linalg.norm(y_probe) + 1e-12))

    expr = _build_linear_expr_from_coeffs(coeffs, intercept, coef_tol=max(1e-14, tol * 0.1))
    mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
    return {
        "expr": expr,
        "mapping": mapping,
        "mse": mse,
        "nrmse": nrmse,
        "expr_str": node_str(expr),
        "size": int(node_size(expr)),
    }


def _estimate_linear_prefit_nrmse(
    features: np.ndarray,
    targets: list[np.ndarray],
    *,
    seed: int,
    n_fit: int,
    n_probe: int,
    feature_dims: Sequence[DimVec] | None = None,
    target_dims: Sequence[DimVec] | None = None,
) -> float:
    """Estimate max per-equation probe NRMSE from sparse linear prefit."""
    N = int(features.shape[0])
    if N <= 1 or not targets:
        return float("inf")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_fit_eff = min(int(n_fit), max(1, int(0.7 * N)))
    n_probe_eff = min(int(n_probe), max(1, N - n_fit_eff))
    fi = idx[:n_fit_eff]
    pi = idx[n_fit_eff : n_fit_eff + n_probe_eff]
    if fi.size == 0 or pi.size == 0:
        return float("inf")

    max_nrmse = 0.0
    for eq_idx, y in enumerate(targets):
        eq_y_dims = None
        if target_dims is not None and eq_idx < len(target_dims):
            eq_y_dims = tuple(target_dims[eq_idx])
        r = _try_linear_prefit(
            features[fi],
            y[fi],
            features[pi],
            y[pi],
            var_dims=feature_dims,
            y_dims=eq_y_dims,
        )
        if r is None:
            return float("inf")
        max_nrmse = max(max_nrmse, float(r.get("nrmse", float("inf"))))
    return max_nrmse


def build_ode_feature_table(
    pid: str,
    params: dict[str, float],
    trajs: list[dict[str, Any]],
    ncomp: int,
    order: int,
    anchor_order: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[str], list[DimVec] | None, list[DimVec] | None]:
    """Build feature matrix and per-equation target vectors from ODE data.

    Returns ``(features, targets, feature_names, feature_dims, target_dims)`` where:
    - ``features``: shape ``(N_total, nvars)``
    - ``targets``:  list of ncomp arrays, each ``(N_total,)``
    - ``feature_names``: list of human-readable column names
    """
    if anchor_order is None:
        anchor_order = order

    ctx = _make_ode_feature_context(pid, params, ncomp, int(anchor_order))

    all_features: list[np.ndarray] = []
    all_targets: list[list[np.ndarray]] = [[] for _ in range(ncomp)]
    feature_dims: list[DimVec] | None = None
    feature_names: list[str] = []

    for traj in trajs:
        t = traj["t"]  # (N,)
        state = traj["state"]  # (n_state, N)
        derivs = _compute_ode_derivatives(pid, params, traj, ncomp, order)  # (N, n_state)
        cols, names, dims = _build_ode_feature_columns(
            pid,
            params,
            t,
            state,
            ncomp,
            order,
            int(anchor_order),
            ctx=ctx,
        )

        features_traj = np.column_stack(cols)
        all_features.append(features_traj)
        if not feature_names:
            feature_names = list(names)
            feature_dims = None if dims is None else list(dims)

        # Targets: the anchor derivatives for each equation.
        for ci in range(ncomp):
            if order == 2 and anchor_order == 2:
                # 2nd order: target is d2u_ci/dt2 = derivs[:, ncomp+ci]
                all_targets[ci].append(derivs[:, ncomp + ci])
            else:
                # 1st order: target is du_ci/dt = derivs[:, ci]
                all_targets[ci].append(derivs[:, ci])

    features = np.concatenate(all_features, axis=0)
    targets = [np.concatenate(t_list, axis=0) for t_list in all_targets]
    target_dims = None if ctx.target_dims is None else [tuple(d) for d in ctx.target_dims]
    return features, targets, feature_names, feature_dims, target_dims


def build_pde_feature_table(
    pid: str,
    params: dict[str, float],
    pde_data: dict[str, Any],
    ncomp: int,
    nxvars: int,
    anchor_order: int,
    *,
    include_nonlinear_terms: bool = True,
    include_first_spatial: bool = True,
    include_second_spatial: bool = True,
) -> tuple[np.ndarray, list[np.ndarray], list[str], list[DimVec] | None, list[DimVec] | None]:
    """Build feature matrix and per-equation target vectors from PDE data.

    For temporal-anchor PDEs, features include component values and spatial
    derivative features; targets are temporal derivatives.

    For anchor_order=2 and nxvars=1, this is treated as a spatial ODE
    (eigenvalue-style setting).
    """
    grid = pde_data["grid"]          # (N, nxvars)
    components = pde_data["components"]  # list of ncomp arrays
    shape = tuple(int(v) for v in pde_data["shape"])
    problem_dims = get_canonical_problem_dims(pid)
    component_dims = None if problem_dims is None else list(problem_dims.component_dims[: int(ncomp)])
    complex_pairs = _complex_pairs_for_problem(problem_dims, ncomp)
    target_dims = _target_dims_for_problem(problem_dims, anchor_order=int(anchor_order), anchor_axis=0)

    if anchor_order == 0:
        # Algebraic: no derivatives.  Target is zero (residual).
        targets = [np.zeros(grid.shape[0], dtype=np.float64) for _ in range(ncomp)]
        cols_alg: list[np.ndarray] = [grid[:, 0]]
        names_alg: list[str] = ["omega"]
        dims_alg: list[DimVec] | None = None
        if problem_dims is not None:
            dims_alg = [tuple(problem_dims.axis_dims[0])]
        for ci in range(ncomp):
            cols_alg.append(components[ci])
            names_alg.append(f"u{ci}")
            if dims_alg is not None and component_dims is not None:
                dims_alg.append(tuple(component_dims[ci]))
        const_cols, const_names, const_dims = _constant_feature_arrays(grid.shape[0], problem_dims, params)
        cols_alg.extend(const_cols)
        names_alg.extend(const_names)
        if dims_alg is not None and const_dims is not None:
            dims_alg.extend(const_dims)
        features = np.column_stack(cols_alg)
        return features, targets, names_alg, dims_alg, target_dims

    if anchor_order == 2 and nxvars == 1:
        # Spatial ODE/eigenvalue case: target is d2u/dx2.
        cols_ev: list[np.ndarray] = []
        names_ev: list[str] = []
        dims_ev: list[DimVec] | None = [] if problem_dims is not None else None
        for ci in range(ncomp):
            cols_ev.append(components[ci])
            names_ev.append(f"u{ci}")
            if dims_ev is not None and component_dims is not None:
                dims_ev.append(tuple(component_dims[ci]))
        dx = float(grid[1, 0] - grid[0, 0]) if grid.shape[0] > 1 else 1.0
        if abs(dx) < 1e-30:
            dx = 1.0
        edge_order = 2 if grid.shape[0] >= 3 else 1
        for ci in range(ncomp):
            du = np.gradient(components[ci], dx, edge_order=edge_order)
            cols_ev.append(du)
            names_ev.append(f"du{ci}/dx")
            if dims_ev is not None and component_dims is not None:
                dims_ev.append(_dim_sub(tuple(component_dims[ci]), tuple(problem_dims.axis_dims[0])))
        cols_ev.append(grid[:, 0])
        names_ev.append("x")
        if dims_ev is not None:
            dims_ev.append(tuple(problem_dims.axis_dims[0]))
        const_cols, const_names, const_dims = _constant_feature_arrays(grid.shape[0], problem_dims, params)
        cols_ev.extend(const_cols)
        names_ev.extend(const_names)
        if dims_ev is not None and const_dims is not None:
            dims_ev.extend(const_dims)
        features = np.column_stack(cols_ev)
        targets = []
        for ci in range(ncomp):
            du = np.gradient(components[ci], dx, edge_order=edge_order)
            d2u = np.gradient(du, dx, edge_order=edge_order)
            targets.append(d2u)
        return features, targets, names_ev, dims_ev, target_dims

    cols: list[np.ndarray] = []
    names: list[str] = []
    dims: list[DimVec] | None = [] if problem_dims is not None else None

    # Component values.
    for ci in range(ncomp):
        cols.append(components[ci])
        names.append(f"u{ci}")
        if dims is not None and component_dims is not None:
            dims.append(tuple(component_dims[ci]))

    if include_nonlinear_terms and ncomp >= 2:
        prod_cols, prod_names, prod_dims = _component_pairwise_product_arrays(
            components,
            ncomp,
            component_dims=component_dims,
        )
        cols.extend(prod_cols)
        names.extend(prod_names)
        if dims is not None and prod_dims is not None:
            dims.extend(prod_dims)

        inv_cols, inv_names, inv_dims = _component_invariant_feature_arrays(
            components,
            ncomp,
            pairs=complex_pairs,
            component_dims=component_dims,
        )
        cols.extend(inv_cols)
        names.extend(inv_names)
        if dims is not None and inv_dims is not None:
            dims.extend(inv_dims)

    # Spatial derivatives as features (for temporal-anchor PDEs).
    if nxvars >= 2:
        spatial_axis = 1
        periodic_spatial = _detect_periodic_spatial_axis(components, shape, spatial_axis)
        if include_first_spatial:
            d_spatial_1 = _compute_pde_spatial_derivs(
                grid,
                components,
                shape,
                spatial_axis,
                deriv_order=1,
                periodic=periodic_spatial,
            )
            for ci in range(ncomp):
                cols.append(d_spatial_1[ci])
                names.append(f"du{ci}/dx")
                if dims is not None and component_dims is not None:
                    dims.append(_dim_sub(tuple(component_dims[ci]), tuple(problem_dims.axis_dims[spatial_axis])))
        if include_second_spatial:
            d_spatial_2 = _compute_pde_spatial_derivs(
                grid,
                components,
                shape,
                spatial_axis,
                deriv_order=2,
                periodic=periodic_spatial,
            )
            for ci in range(ncomp):
                cols.append(d_spatial_2[ci])
                names.append(f"d2u{ci}/dx2")
                if dims is not None and component_dims is not None:
                    dims.append(_dim_sub(tuple(component_dims[ci]), _dim_scale(tuple(problem_dims.axis_dims[spatial_axis]), 2.0)))

        # Include coordinate-modulated component terms to capture broad classes
        # of spatially varying linear operators (e.g. x^2·u in harmonic
        # Schrödinger) without per-problem term templates.
        x = grid[:, spatial_axis]
        x2 = x * x
        cols.append(x)
        names.append("x")
        cols.append(x2)
        names.append("x2")
        if dims is not None:
            x_dim = tuple(problem_dims.axis_dims[spatial_axis])
            dims.append(x_dim)
            dims.append(_dim_scale(x_dim, 2.0))
        for ci in range(ncomp):
            cols.append(x * components[ci])
            names.append(f"x*u{ci}")
            cols.append(x2 * components[ci])
            names.append(f"x2*u{ci}")
            if dims is not None and component_dims is not None:
                x_dim = tuple(problem_dims.axis_dims[spatial_axis])
                dims.append(_dim_add(x_dim, tuple(component_dims[ci])))
                dims.append(_dim_add(_dim_scale(x_dim, 2.0), tuple(component_dims[ci])))

    const_cols, const_names, const_dims = _constant_feature_arrays(grid.shape[0], problem_dims, params)
    cols.extend(const_cols)
    names.extend(const_names)
    if dims is not None and const_dims is not None:
        dims.extend(const_dims)

    features = np.column_stack(cols)

    if anchor_order == 2 and nxvars >= 2:
        # 2nd-order temporal: target is ∂²u_k/∂t².
        temporal_axis = 0
        d2t = _compute_pde_second_temporal_derivs(grid, components, shape, temporal_axis)
        targets = d2t
    else:
        # 1st-order temporal: target is ∂u_k/∂t.
        temporal_axis = 0
        dt = _compute_pde_temporal_derivs(grid, components, shape, temporal_axis)
        targets = dt

    return features, targets, names, dims, target_dims


# ---------------------------------------------------------------------------
# Sparse/STLSQ engine
# ---------------------------------------------------------------------------


class CombinedSurrogate(torch.nn.Module):
    """Wrap single-output component surrogates as one N-output system."""

    def __init__(self, *surrogates: torch.nn.Module):
        super().__init__()
        self.surrogates = torch.nn.ModuleList(surrogates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parts = []
            for surr in self.surrogates:
                y = surr(x)
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                parts.append(y)
        return torch.cat(parts, dim=1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parts = []
            for surr in self.surrogates:
                g = surr.grad(x)
                if g.ndim == 2:
                    g = g.unsqueeze(1)
                parts.append(g)
        return torch.cat(parts, dim=1)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parts = []
            for surr in self.surrogates:
                h = surr.grad_grad(x)
                if h.ndim == 3:
                    h = h.unsqueeze(1)
                parts.append(h)
        return torch.cat(parts, dim=1)


def generate_sparse_component_data(
    problem: ComplexProblemDef,
    params: dict[str, float],
    ics: dict[str, float],
    t_max: float,
    data_dir: Path,
    *,
    n_traj: int = 6,
    n_points: int = 5000,
    seed: int = 0,
) -> list[Path]:
    """Generate one CSV per real component for the sparse/STLSQ engine."""
    pid = problem.id
    ncomp = NCOMPONENTS.get(pid, 2)
    nxvars = NXVARS.get(pid, 1)
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(_problem_seed(seed, pid))
    if pid in PDE_DATA_GENERATORS:
        grid, comp_arrays = PDE_DATA_GENERATORS[pid](params, t_max, n_points)
        grid = np.asarray(grid, dtype=np.float64)
        comp_arrays = [np.asarray(c, dtype=np.float64) for c in comp_arrays]
        idx = rng.permutation(grid.shape[0])
        grid = grid[idx]

        header = "y," + ",".join(f"x{i}" for i in range(nxvars))
        csv_paths: list[Path] = []
        for ci in range(ncomp):
            label = _COMP_LABELS[ci] if ci < len(_COMP_LABELS) else f"c{ci}"
            path = data_dir / f"{pid}_{label}.csv"
            np.savetxt(
                str(path),
                np.column_stack([comp_arrays[ci][idx], grid]),
                delimiter=",",
                header=header,
                comments="",
            )
            csv_paths.append(path)
        return csv_paths

    if pid not in RHS_REGISTRY:
        raise RuntimeError(f"No RHS function registered for {pid}")

    trajs = generate_ode_multi_traj(
        pid,
        params,
        ncomp,
        problem.order,
        t_max,
        ics,
        n_traj=1,
        n_points=n_points,
        seed=_problem_seed(seed, pid),
    )
    t_all = np.asarray(trajs[0]["t"], dtype=np.float64)
    idx = rng.permutation(len(t_all))
    t = t_all[idx]
    csv_paths = []
    for ci in range(ncomp):
        label = _COMP_LABELS[ci] if ci < len(_COMP_LABELS) else f"c{ci}"
        path = data_dir / f"{pid}_{label}.csv"
        comp = np.asarray(trajs[0]["state"][ci], dtype=np.float64)[idx]
        np.savetxt(
            str(path),
            np.column_stack([comp, t]),
            delimiter=",",
            header="y,x0",
            comments="",
        )
        csv_paths.append(path)
    return csv_paths


def _ode_trajectory_component_csv_paths(
    data_dir: Path,
    pid: str,
    *,
    n_traj: int,
    ncomp: int,
) -> list[list[Path]]:
    paths: list[list[Path]] = []
    for traj_idx in range(int(n_traj)):
        row: list[Path] = []
        for ci in range(int(ncomp)):
            label = _COMP_LABELS[ci] if ci < len(_COMP_LABELS) else f"c{ci}"
            row.append(data_dir / f"{pid}_traj{traj_idx}_{label}.csv")
        paths.append(row)
    return paths


def generate_sparse_ode_trajectory_component_data(
    problem: ComplexProblemDef,
    params: dict[str, float],
    ics: dict[str, float],
    t_max: float,
    data_dir: Path,
    *,
    n_traj: int = 6,
    n_points: int = 5000,
    seed: int = 0,
) -> list[list[Path]]:
    """Generate one component CSV per ODE trajectory for surrogate STLSQ.

    A single multi-IC ``u(t)`` surrogate is ill-posed because the same time
    value has several component values.  This mirrors the real-valued
    multi-dataset DE path: train one surrogate per trajectory, then discover
    one shared sparse equation from the concatenated surrogate derivatives.
    """
    pid = problem.id
    if pid not in RHS_REGISTRY:
        raise RuntimeError(f"No RHS function registered for {pid}")

    ncomp = NCOMPONENTS.get(pid, 2)
    data_dir.mkdir(parents=True, exist_ok=True)
    trajs = generate_ode_multi_traj(
        pid,
        params,
        ncomp,
        problem.order,
        t_max,
        ics,
        n_traj=max(1, int(n_traj)),
        n_points=int(n_points),
        seed=_problem_seed(seed, pid),
    )
    paths = _ode_trajectory_component_csv_paths(
        data_dir,
        pid,
        n_traj=len(trajs),
        ncomp=int(ncomp),
    )
    rng = np.random.default_rng(_problem_seed(seed, pid))
    for traj_idx, traj in enumerate(trajs):
        t_all = np.asarray(traj["t"], dtype=np.float64)
        idx = rng.permutation(len(t_all))
        t = t_all[idx]
        state = np.asarray(traj["state"], dtype=np.float64)
        for ci, path in enumerate(paths[traj_idx]):
            comp = state[ci][idx]
            np.savetxt(
                str(path),
                np.column_stack([comp, t]),
                delimiter=",",
                header="y,x0",
                comments="",
            )
    return paths


def train_component_surrogate(
    csv_path: Path,
    num_segments: int,
    epochs: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    *,
    loss_target: float = 1e-8,
    batch_size: int = 2000,
    ndata_train: int = 2000,
    ndata_val: int = 2000,
    verbose: bool = False,
    nxvars: int = 1,
) -> tuple[torch.nn.Module, float]:
    """Train a single-output NestyNet surrogate on one component CSV."""
    data_hp = DataHyperparams(
        batch_size=batch_size,
        ndata_select=ndata_train,
        ndata_select_val=ndata_val,
    )
    model_hp = ModelHyperparams(
        double_precision=True,
        repeatable_runs=True,
        model_base_name="G_Model",
        num_segments_min=num_segments,
        num_segments_max=num_segments,
    )

    _ds_tr, _ds_va, dl_tr, dl_va = build_datasets(
        str(csv_path),
        nxvars,
        np.float64,
        data_hp,
        None,
    )

    ast0 = build_initial_ast(Nxvars=nxvars, num_segments=num_segments, dual_layer=True)
    leaf_builder = LeafBuilder(model_hp, device, dtype)
    surrogate, nparam, _ = build_composite_ast(
        ast0,
        num_segments,
        dual_layer=True,
        leaf_builder=leaf_builder,
        device=device,
        dtype=dtype,
    )

    if verbose:
        print(f"    Model has {nparam} parameters")

    best_val, _best_train, best_p, lm_opt = train_initial_model(
        surrogate,
        dl_tr,
        dl_va,
        epochs=epochs,
        LM_strategy="direct_solve",
        nval_patience=250,
        loss_target=loss_target,
        epochs_min=300,
        chisq_tol=1e-10,
        device=device,
    )

    lm_opt._update_param_groups(best_p)
    surrogate.eval()
    return surrogate, float(best_val)


def _load_training_x_from_csv(
    csv_path: Path,
    *,
    nxvars: int,
    dtype: torch.dtype,
    batch_size: int = 2000,
    ndata_train: int = 2000,
    ndata_val: int = 2000,
) -> torch.Tensor:
    data_hp = DataHyperparams(
        batch_size=batch_size,
        ndata_select=ndata_train,
        ndata_select_val=ndata_val,
    )
    _ds_tr, _ds_va, dl_tr, _dl_va = build_datasets(
        str(csv_path),
        nxvars,
        np.float64,
        data_hp,
        None,
    )
    return torch.cat([batch[0] for batch in dl_tr], dim=0).to(dtype=dtype)


def train_ode_trajectory_surrogates(
    csv_paths_by_traj: Sequence[Sequence[Path]],
    *,
    nxvars: int,
    num_segments: int,
    epochs: int,
    device: torch.device,
    dtype: torch.dtype,
    loss_target: float,
    verbose: bool = False,
) -> tuple[list[CombinedSurrogate], list[torch.Tensor], list[list[float]]]:
    """Train component surrogates for each ODE trajectory."""
    combined_by_traj: list[CombinedSurrogate] = []
    x_by_traj: list[torch.Tensor] = []
    val_losses: list[list[float]] = []
    for traj_idx, row in enumerate(csv_paths_by_traj):
        comp_surrogates: list[torch.nn.Module] = []
        comp_losses: list[float] = []
        for ci, csv_path in enumerate(row):
            label = _COMP_LABELS[ci] if ci < len(_COMP_LABELS) else f"c{ci}"
            print(f"  Training traj{traj_idx} {label}-component surrogate...")
            surr, val_loss = train_component_surrogate(
                Path(csv_path),
                num_segments,
                epochs,
                device,
                dtype,
                verbose=verbose,
                loss_target=loss_target,
                nxvars=nxvars,
            )
            print(f"    val_loss = {val_loss:.6e}")
            comp_surrogates.append(surr)
            comp_losses.append(float(val_loss))
        combined_by_traj.append(CombinedSurrogate(*comp_surrogates))
        x_by_traj.append(
            _load_training_x_from_csv(
                Path(row[0]),
                nxvars=nxvars,
                dtype=dtype,
            )
        )
        val_losses.append(comp_losses)
    return combined_by_traj, x_by_traj, val_losses


def _normalized_problem_x_axis(problem: ComplexProblemDef, nxvars: int) -> int:
    x_axis = int(problem.x_axis)
    return x_axis if x_axis >= 0 else int(nxvars) + x_axis


def _normalize_sparse_library_policy(policy: str) -> str:
    key = str(policy).strip().lower().replace("-", "_")
    aliases = {
        "default": "class",
        "fair": "class",
        "metadata": "class",
        "metadata_class": "class",
    }
    return aliases.get(key, key)


def _class_sparse_default_lambda(n_terms: int, *, anchor_order: int, nxvars: int) -> float:
    """Default threshold for the metadata-driven class libraries.

    Class libraries enumerate the candidate terms characteristic of a problem
    class, so they are larger than a hand-picked per-equation list and the
    default threshold is scaled to the candidate count.
    """
    n_terms = int(n_terms)
    if int(anchor_order) == 2 and int(nxvars) >= 2:
        return 2.0e-1 if n_terms <= 8 else 1.0e-1
    if n_terms <= 8:
        return 2.0e-2
    if n_terms <= 18:
        return 5.0e-2
    return 1.0e-1 if int(nxvars) >= 2 or int(anchor_order) == 0 else 7.5e-2


def _class_sparse_accept_nrmse(problem: ComplexProblemDef) -> float:
    """Residual threshold for accepting a simpler class-library variant."""
    nxvars = NXVARS.get(problem.id, 1)
    anchor_order = ANCHOR_ORDER.get(problem.id, problem.order)
    if int(nxvars) >= 2:
        return 5.0e-3
    if int(anchor_order) == 0:
        return 2.0e-3
    return 1.0e-3


def _class_sparse_anchor_scale(
    combined: CombinedSurrogate,
    X: torch.Tensor,
    problem: ComplexProblemDef,
) -> float:
    """Return an RMS scale for the anchored derivative used as STLSQ target."""
    pid = problem.id
    nxvars = NXVARS.get(pid, 1)
    anchor_order = ANCHOR_ORDER.get(pid, problem.order)
    x_axis = _normalized_problem_x_axis(problem, nxvars)

    with torch.no_grad():
        if int(anchor_order) == 0:
            target = combined.forward(X)
        elif int(anchor_order) == 1:
            target = combined.grad(X)[:, :, int(x_axis)]
        elif int(anchor_order) == 2:
            target = combined.grad_grad(X)[:, :, int(x_axis), int(x_axis)]
        else:
            return 1.0

        target = target.detach().to(dtype=torch.float64)
        rms = torch.sqrt(torch.mean(target.square(), dim=0))
        scale = float(torch.mean(rms).detach().cpu())
    return max(scale, 1.0e-12)


def _add_unique_term(terms: list[Node], seen: set[str], node: Node) -> None:
    key = repr(node)
    if key not in seen:
        seen.add(key)
        terms.append(node)


def _dim_close(a: Sequence[float], b: Sequence[float], *, tol: float = 1.0e-12) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def _dim_zero(rank: int) -> DimVec:
    return tuple(0.0 for _ in range(int(rank)))


def _node_out_idx(node: AtomNode) -> int:
    return int(getattr(node, "kwargs", {}).get("out_idx", 0))


def _class_term_dim(node: Node | None, dims: Any) -> DimVec | None:
    if node is None:
        return _dim_zero(len(dims.basis))
    if isinstance(node, AtomNode):
        kind = str(node.kind)
        if kind == "u":
            return tuple(dims.component_dims[_node_out_idx(node)])
        if kind == "du":
            out = _node_out_idx(node)
            axis = int(node.kwargs.get("axis", 0))
            return _dim_sub(tuple(dims.component_dims[out]), tuple(dims.axis_dims[axis]))
        if kind == "d2u":
            out = _node_out_idx(node)
            axis0 = int(node.kwargs.get("axis0", 0))
            axis1 = int(node.kwargs.get("axis1", axis0))
            return _dim_sub(
                _dim_sub(tuple(dims.component_dims[out]), tuple(dims.axis_dims[axis0])),
                tuple(dims.axis_dims[axis1]),
            )
        if kind == "var":
            axis = int(node.var_idxs[0]) if node.var_idxs else 0
            return tuple(dims.axis_dims[axis])
        return None
    if isinstance(node, PowNode):
        base_dim = _class_term_dim(node.base, dims)
        if base_dim is None or not isinstance(node.exponent, (int, float)):
            return None
        return _dim_scale(tuple(base_dim), float(node.exponent))
    if isinstance(node, MulNode):
        left_dim = _class_term_dim(node.left, dims)
        right_dim = _class_term_dim(node.right, dims)
        if left_dim is None or right_dim is None:
            return None
        return _dim_add(tuple(left_dim), tuple(right_dim))
    if isinstance(node, AddNode):
        left_dim = _class_term_dim(node.left, dims)
        right_dim = _class_term_dim(node.right, dims)
        if left_dim is None or right_dim is None or not _dim_close(left_dim, right_dim):
            return None
        return tuple(left_dim)
    if isinstance(node, (SinNode, CosNode)):
        arg_dim = _class_term_dim(node.arg, dims)
        if arg_dim is None or not _dim_close(arg_dim, _dim_zero(len(dims.basis))):
            return None
        return _dim_zero(len(dims.basis))
    return None


def _class_anchor_dims(problem: ComplexProblemDef, dims: Any) -> list[DimVec]:
    pid = problem.id
    nxvars = NXVARS.get(pid, 1)
    x_axis = _normalized_problem_x_axis(problem, nxvars)
    anchor_order = ANCHOR_ORDER.get(pid, problem.order)
    if int(anchor_order) == 0:
        return [tuple(d) for d in dims.component_dims]
    out = []
    for comp_dim in dims.component_dims:
        dim = tuple(comp_dim)
        for _ in range(int(anchor_order)):
            dim = _dim_sub(dim, tuple(dims.axis_dims[int(x_axis)]))
        out.append(tuple(dim))
    return out


def _problem_supports_modulus_nonlinearity(problem: ComplexProblemDef) -> bool:
    ops = {op.strip().lower() for op in str(getattr(problem, "complex_ops", "")).split(",")}
    if "mod2" not in ops:
        return False

    dims = get_complex_problem_dims(problem.id)
    if dims is None or not getattr(dims, "complex_pairs", ()):
        return False

    anchors = _class_anchor_dims(problem, dims)
    rank = len(dims.basis)
    coeff_dims = [_dim_zero(rank)]
    coeff_dims.extend(tuple(dim) for dim in getattr(dims, "constant_dims", {}).values())

    ncomp = NCOMPONENTS.get(problem.id, 2)
    for re_idx, im_idx in _complex_component_pairs(int(ncomp)):
        for ci in range(int(ncomp)):
            term = _mod_sq_times_u_node(ci, re_idx=re_idx, im_idx=im_idx)
            term_dim = _class_term_dim(term, dims)
            if term_dim is None:
                continue
            for anchor_dim in anchors:
                required_coeff_dim = _dim_sub(tuple(anchor_dim), tuple(term_dim))
                if any(_dim_close(required_coeff_dim, coeff_dim) for coeff_dim in coeff_dims):
                    return True
    return False


def _class_coeff_dims_for_problem(dims: Any) -> list[DimVec]:
    rank = len(dims.basis)
    out: list[DimVec] = [_dim_zero(rank)]
    constants = [tuple(dim) for dim in getattr(dims, "constant_dims", {}).values()]
    out.extend(constants)
    # Include simple constant products such as omega0^2 and g*N0.  This keeps
    # the filter class-level without needing symbolic coefficient templates.
    for a in constants:
        for b in constants:
            out.append(_dim_add(tuple(a), tuple(b)))

    uniq: list[DimVec] = []
    for dim in out:
        if not any(_dim_close(dim, prev) for prev in uniq):
            uniq.append(tuple(dim))
    return uniq


def _trig_arg_can_be_dimensionless_with_constant(arg_dim: DimVec, dims: Any) -> bool:
    rank = len(dims.basis)
    if _dim_close(arg_dim, _dim_zero(rank)):
        return True
    for const_dim in getattr(dims, "constant_dims", {}).values():
        if _dim_close(_dim_add(tuple(arg_dim), tuple(const_dim)), _dim_zero(rank)):
            return True
    return False


def _class_term_dim_for_variant_filter(node: Node | None, dims: Any) -> DimVec | None:
    """Term dimension used only to reject impossible class variants.

    This is slightly more permissive than :func:`_class_term_dim` for
    ``sin(x)``/``cos(x)`` carriers: a benchmark may have a frequency constant
    such that ``Omega*x`` is dimensionless even though the class-level AST only
    writes ``sin(x)``.
    """
    if node is None:
        return _dim_zero(len(dims.basis))
    if isinstance(node, (SinNode, CosNode)):
        arg_dim = _class_term_dim_for_variant_filter(node.arg, dims)
        if arg_dim is None or not _trig_arg_can_be_dimensionless_with_constant(tuple(arg_dim), dims):
            return None
        return _dim_zero(len(dims.basis))
    if isinstance(node, PowNode):
        base_dim = _class_term_dim_for_variant_filter(node.base, dims)
        if base_dim is None or not isinstance(node.exponent, (int, float)):
            return None
        return _dim_scale(tuple(base_dim), float(node.exponent))
    if isinstance(node, MulNode):
        left_dim = _class_term_dim_for_variant_filter(node.left, dims)
        right_dim = _class_term_dim_for_variant_filter(node.right, dims)
        if left_dim is None or right_dim is None:
            return None
        return _dim_add(tuple(left_dim), tuple(right_dim))
    if isinstance(node, AddNode):
        left_dim = _class_term_dim_for_variant_filter(node.left, dims)
        right_dim = _class_term_dim_for_variant_filter(node.right, dims)
        if left_dim is None or right_dim is None or not _dim_close(left_dim, right_dim):
            return None
        return tuple(left_dim)
    return _class_term_dim(node, dims)


def _class_variant_dimensionally_supported(problem: ComplexProblemDef, terms: Sequence[Node]) -> bool:
    dims = get_complex_problem_dims(problem.id)
    if dims is None:
        return True

    anchors = _class_anchor_dims(problem, dims)
    coeff_dims = _class_coeff_dims_for_problem(dims)
    for term in terms:
        term_dim = _class_term_dim_for_variant_filter(term, dims)
        if term_dim is None:
            continue
        for anchor_dim in anchors:
            required_coeff_dim = _dim_sub(tuple(anchor_dim), tuple(term_dim))
            if any(_dim_close(required_coeff_dim, coeff_dim) for coeff_dim in coeff_dims):
                return True
    return False


def _is_modulus_class_variant(name: str) -> bool:
    key = str(name).lower()
    return "modulus" in key or "nonlinear" in key


def _complex_component_pairs(ncomp: int) -> list[tuple[int, int]]:
    if int(ncomp) == 2:
        return [(0, 1)]
    if int(ncomp) == 4:
        return [(0, 1), (2, 3)]
    return [(i, i + 1) for i in range(0, int(ncomp) - 1, 2)]


def _mod_sq_node(re_idx: int, im_idx: int) -> Node:
    return Add(Pow(U(out_idx=int(re_idx)), 2), Pow(U(out_idx=int(im_idx)), 2))


def _mod_sq_times_u_node(target_idx: int, *, re_idx: int, im_idx: int) -> Node:
    return Mul(_mod_sq_node(re_idx, im_idx), U(out_idx=int(target_idx)))


def _coordinate_is_inverse_safe(coord_mins: np.ndarray | None, axis: int) -> bool:
    if coord_mins is None:
        return False
    arr = np.asarray(coord_mins, dtype=np.float64).reshape(-1)
    if int(axis) < 0 or int(axis) >= arr.size:
        return False
    return bool(abs(float(arr[int(axis)])) >= 0.1)


def build_class_sparse_library(
    problem: ComplexProblemDef,
    *,
    coord_mins: np.ndarray | None = None,
) -> list[Node]:
    """Build a metadata-driven sparse library without per-problem term lists."""
    pid = problem.id
    ncomp = NCOMPONENTS.get(pid, 2)
    nxvars = NXVARS.get(pid, 1)
    x_axis = _normalized_problem_x_axis(problem, nxvars)
    anchor_order = ANCHOR_ORDER.get(pid, problem.order)
    non_anchor_axes = [ax for ax in range(int(nxvars)) if ax != int(x_axis)]

    terms: list[Node] = []
    seen: set[str] = set()

    def add(node: Node) -> None:
        _add_unique_term(terms, seen, node)

    def add_fields() -> None:
        for ci in range(int(ncomp)):
            add(U(out_idx=ci))

    def add_modulus_terms() -> None:
        pairs = _complex_component_pairs(int(ncomp))
        if not pairs:
            return
        for re_idx, im_idx in pairs:
            for ci in range(int(ncomp)):
                add(_mod_sq_times_u_node(ci, re_idx=re_idx, im_idx=im_idx))

    def add_pairwise_products() -> None:
        for i in range(int(ncomp)):
            for j in range(i, int(ncomp)):
                add(Mul(U(out_idx=i), U(out_idx=j)) if i != j else Pow(U(out_idx=i), 2))

    def add_coord_field_terms(axis: int, *, include_inv: bool = False) -> None:
        x = Var(int(axis))
        x2 = Pow(x, 2)
        inv_safe = include_inv and _coordinate_is_inverse_safe(coord_mins, int(axis))
        inv_x = Pow(x, -1) if inv_safe else None
        inv_x2 = Pow(x, -2) if inv_safe else None
        for ci in range(int(ncomp)):
            add(Mul(x, U(out_idx=ci)))
            add(Mul(x2, U(out_idx=ci)))
            if inv_x is not None:
                add(Mul(inv_x, U(out_idx=ci)))
            if inv_x2 is not None:
                add(Mul(inv_x2, U(out_idx=ci)))

    if int(anchor_order) == 0:
        axis = int(x_axis)
        x = Var(axis)
        inv_x = Pow(x, -1) if _coordinate_is_inverse_safe(coord_mins, axis) else None
        for ci in range(int(ncomp)):
            add(Mul(x, U(out_idx=ci)))
            if inv_x is not None:
                add(Mul(inv_x, U(out_idx=ci)))
        return terms

    if int(nxvars) >= 2:
        add_fields()
        for ax in non_anchor_axes:
            for ci in range(int(ncomp)):
                add(DU(int(ax), out_idx=ci))
                add(D2U(int(ax), int(ax), out_idx=ci))
            add_coord_field_terms(int(ax))
        add_modulus_terms()
        return terms

    # 1D ODE/eigenvalue systems.
    add_fields()
    if int(anchor_order) == 2:
        for ci in range(int(ncomp)):
            add(DU(int(x_axis), out_idx=ci))
        if _coordinate_is_inverse_safe(coord_mins, int(x_axis)):
            inv_x = Pow(Var(int(x_axis)), -1)
            for ci in range(int(ncomp)):
                add(Mul(inv_x, DU(int(x_axis), out_idx=ci)))
    else:
        add_pairwise_products()
        add_modulus_terms()
        carrier_axis = int(x_axis)
        sin_x = Sin(Var(carrier_axis))
        cos_x = Cos(Var(carrier_axis))
        add(sin_x)
        add(cos_x)
        for ci in range(int(ncomp)):
            add(Mul(U(out_idx=ci), sin_x))
            add(Mul(U(out_idx=ci), cos_x))
            add(Sin(U(out_idx=ci)))

    return terms


def build_class_sparse_library_variants(
    problem: ComplexProblemDef,
    *,
    coord_mins: np.ndarray | None = None,
) -> list[tuple[str, list[Node]]]:
    """Build small class-level library variants for model selection."""
    pid = problem.id
    ncomp = NCOMPONENTS.get(pid, 2)
    nxvars = NXVARS.get(pid, 1)
    x_axis = _normalized_problem_x_axis(problem, nxvars)
    anchor_order = ANCHOR_ORDER.get(pid, problem.order)
    non_anchor_axes = [ax for ax in range(int(nxvars)) if ax != int(x_axis)]

    def dedupe(nodes: Sequence[Node]) -> list[Node]:
        out: list[Node] = []
        seen: set[str] = set()
        for node in nodes:
            _add_unique_term(out, seen, node)
        return out

    fields = [U(out_idx=ci) for ci in range(int(ncomp))]
    pair_products = [
        Mul(U(out_idx=i), U(out_idx=j)) if i != j else Pow(U(out_idx=i), 2)
        for i in range(int(ncomp))
        for j in range(i, int(ncomp))
    ]
    modulus_terms = [
        _mod_sq_times_u_node(ci, re_idx=re_idx, im_idx=im_idx)
        for re_idx, im_idx in _complex_component_pairs(int(ncomp))
        for ci in range(int(ncomp))
    ]

    variants: list[tuple[str, list[Node]]] = []

    if int(anchor_order) == 0:
        x = Var(int(x_axis))
        inv_x = Pow(x, -1) if _coordinate_is_inverse_safe(coord_mins, int(x_axis)) else None
        freq_terms: list[Node] = []
        for ci in range(int(ncomp)):
            freq_terms.append(Mul(x, U(out_idx=ci)))
            if inv_x is not None:
                freq_terms.append(Mul(inv_x, U(out_idx=ci)))
        return [("algebraic_frequency", dedupe(freq_terms))]

    if int(nxvars) >= 2:
        first_derivs: list[Node] = []
        second_derivs: list[Node] = []
        coord_fields: list[Node] = []
        for ax in non_anchor_axes:
            x = Var(int(ax))
            x2 = Pow(x, 2)
            for ci in range(int(ncomp)):
                first_derivs.append(DU(int(ax), out_idx=ci))
                second_derivs.append(D2U(int(ax), int(ax), out_idx=ci))
                coord_fields.append(Mul(x, U(out_idx=ci)))
                coord_fields.append(Mul(x2, U(out_idx=ci)))

        nonlinear_supported = _problem_supports_modulus_nonlinearity(problem)
        declared_order = int(problem.order)
        if declared_order <= 1:
            nonlinear_variants = []
            if nonlinear_supported:
                nonlinear_variants = [
                    ("first_order_nonlinear", dedupe(first_derivs + fields + modulus_terms)),
                ]
            linear_variants = [
                ("spatial_first_system", dedupe(first_derivs + fields)),
                ("linear_system", dedupe(fields)),
            ]
        else:
            nonlinear_variants = [
                ("nonlinear_modulus", dedupe(second_derivs + modulus_terms)),
                ("reaction_diffusion_modulus", dedupe(second_derivs + fields + modulus_terms)),
            ]
            if int(ncomp) >= 4:
                nonlinear_variants = [
                    ("multifield_modulus", dedupe(second_derivs + modulus_terms)),
                    ("multifield_nonlinear", dedupe(second_derivs + fields + modulus_terms)),
                ]
            linear_variants = [
                ("spatial_second", dedupe(second_derivs)),
                ("linear_potential", dedupe(second_derivs + fields)),
                ("coordinate_potential", dedupe(second_derivs + fields + coord_fields)),
                ("spatial_first_system", dedupe(first_derivs + fields)),
            ]
        variants.extend(nonlinear_variants + linear_variants if nonlinear_supported else linear_variants + nonlinear_variants)
        return [(name, terms) for name, terms in variants if terms]

    if int(anchor_order) == 2:
        first_derivs = [DU(int(x_axis), out_idx=ci) for ci in range(int(ncomp))]
        sin_x = Sin(Var(int(x_axis)))
        cos_x = Cos(Var(int(x_axis)))
        forcing_terms = [sin_x, cos_x]
        variants.extend([
            ("second_order_linear", dedupe(fields)),
            ("damped_second_order", dedupe(fields + first_derivs)),
        ])
        if _coordinate_is_inverse_safe(coord_mins, int(x_axis)):
            inv_x = Pow(Var(int(x_axis)), -1)
            singular = []
            for ci in range(int(ncomp)):
                singular.append(Mul(inv_x, DU(int(x_axis), out_idx=ci)))
            variants.append(("radial_singular", dedupe(fields + singular)))
        if problem.id in NON_AUTONOMOUS:
            variants.append(("forced_damped_second_order", dedupe(fields + first_derivs + forcing_terms)))
        return [(name, terms) for name, terms in variants if terms]

    sin_x = Sin(Var(int(x_axis)))
    cos_x = Cos(Var(int(x_axis)))
    forcing_terms = [sin_x, cos_x]
    field_trig_terms = []
    field_sin_terms = []
    for ci in range(int(ncomp)):
        field_trig_terms.append(Mul(U(out_idx=ci), sin_x))
        field_trig_terms.append(Mul(U(out_idx=ci), cos_x))
        field_sin_terms.append(Sin(U(out_idx=ci)))

    variants.extend([
        ("linear_system", dedupe(fields)),
        ("polynomial_coupling", dedupe(fields + pair_products)),
        ("modulus_nonlinear", dedupe(fields + modulus_terms)),
        ("forced_linear", dedupe(fields + forcing_terms)),
        ("field_trig_forcing", dedupe(fields + field_trig_terms)),
        ("field_sine", dedupe(fields + field_sin_terms)),
    ])
    return [(name, terms) for name, terms in variants if terms]


def discover_sparse_class_system(
    combined: CombinedSurrogate,
    X: torch.Tensor,
    problem: ComplexProblemDef,
    *,
    stlsq_lambda: float | None = None,
) -> tuple[Any, str, int, float]:
    variants = build_class_sparse_library_variants(
        problem,
        coord_mins=X.min(dim=0).values.detach().cpu().numpy(),
    )
    if not variants:
        raise RuntimeError(f"No class sparse library variants generated for {problem.id}")
    supported_variants = [
        (name, terms)
        for name, terms in variants
        if _class_variant_dimensionally_supported(problem, terms)
    ]
    if supported_variants:
        variants = supported_variants

    nxvars = NXVARS.get(problem.id, 1)
    anchor_order = ANCHOR_ORDER.get(problem.id, problem.order)
    anchor_scale = _class_sparse_anchor_scale(combined, X, problem)
    accept_nrmse = _class_sparse_accept_nrmse(problem)
    candidates: list[tuple[float, float, Any, str, int, float, int]] = []
    errors: list[str] = []

    for variant_name, terms in variants:
        lam = (
            _class_sparse_default_lambda(len(terms), anchor_order=anchor_order, nxvars=nxvars)
            if stlsq_lambda is None
            else float(stlsq_lambda)
        )
        try:
            result = discover_sparse_coupled_system(
                combined,
                X,
                problem,
                stlsq_lambda=lam,
                library_terms=terms,
            )
        except Exception as exc:
            errors.append(f"{variant_name}: {exc}")
            continue
        mean_rms = float(sum(result.rms_train) / max(1, len(result.rms_train)))
        nrmse = mean_rms / anchor_scale
        n_active = int((result.coeffs.abs() > 1e-12).sum())
        score = nrmse + 5.0e-3 * float(n_active) + 1.0e-3 * float(len(terms))
        candidates.append((score, nrmse, result, variant_name, len(terms), float(lam), n_active))

    if not candidates:
        detail = "; ".join(errors[:5])
        raise RuntimeError(f"No class sparse variant succeeded for {problem.id}: {detail}")

    selection_pool = candidates
    if _problem_supports_modulus_nonlinearity(problem):
        nonlinear_candidates = [cand for cand in candidates if _is_modulus_class_variant(cand[3])]
        if nonlinear_candidates:
            selection_pool = nonlinear_candidates

    for _score, nrmse, result, variant_name, n_terms, lam, _n_active in selection_pool:
        if nrmse <= accept_nrmse:
            return result, variant_name, int(n_terms), float(lam)

    _score, _nrmse, result, variant_name, n_terms, lam, _n_active = min(selection_pool, key=lambda item: item[0])
    return result, variant_name, int(n_terms), float(lam)


def discover_sparse_coupled_system(
    combined: CombinedSurrogate,
    X: torch.Tensor,
    problem: ComplexProblemDef,
    *,
    stlsq_lambda: float = 1e-3,
    library_terms: Sequence[Node],
):
    """Discover a coupled system using an explicit sparse library.

    ``library_terms`` is required: the candidate terms are supplied by the
    metadata-driven class library (see :func:`discover_sparse_class_system`).
    """
    pid = problem.id
    ncomp = NCOMPONENTS.get(pid, 2)
    nxvars = NXVARS.get(pid, 1)
    anchor_order = ANCHOR_ORDER.get(pid, problem.order)
    lib = list(library_terms)

    cfg = SystemDESearchConfig(
        x_axis=_normalized_problem_x_axis(problem, nxvars),
        order_candidates=(anchor_order,),
        out_idxs=tuple(range(ncomp)),
        include_const=True,
        include_x=False,
        include_u=False,
        include_xu=False,
        include_u_cross=False,
        stlsq_lambda=float(stlsq_lambda),
        share_support_across_equations=False,
    )
    return discover_system_de_from_surrogate(
        combined,
        [(X,)],
        cfg=cfg,
        library_terms=lib,
    )


def _node_contains_atom_kind(node: Node | None, kind: str) -> bool:
    if node is None:
        return False
    if isinstance(node, AtomNode):
        return str(node.kind) == str(kind)
    if isinstance(node, PowNode):
        return _node_contains_atom_kind(node.base, kind) or (
            not isinstance(node.exponent, (int, float)) and _node_contains_atom_kind(node.exponent, kind)
        )
    if isinstance(node, MulNode):
        return _node_contains_atom_kind(node.left, kind) or _node_contains_atom_kind(node.right, kind)
    if isinstance(node, AddNode):
        return _node_contains_atom_kind(node.left, kind) or _node_contains_atom_kind(node.right, kind)
    if isinstance(node, (SinNode, CosNode)):
        return _node_contains_atom_kind(node.arg, kind)
    return False


def _eval_ode_node_surrogate_table(
    node: Node | None,
    X: torch.Tensor,
    values: torch.Tensor,
    grads: torch.Tensor | None,
    hess: torch.Tensor | None,
    *,
    x_axis: int,
) -> torch.Tensor:
    if node is None:
        return torch.ones(X.shape[0], dtype=X.dtype, device=X.device)
    if isinstance(node, AtomNode):
        kind = str(node.kind)
        if kind == "u":
            return values[:, _node_out_idx(node)]
        if kind == "du":
            out = _node_out_idx(node)
            axis = int(node.kwargs.get("axis", 0))
            if axis != int(x_axis):
                raise ValueError(f"ODE surrogate table only supports DU(axis={x_axis}), got axis={axis}")
            if grads is None:
                raise RuntimeError("ODE surrogate gradients unavailable")
            return grads[:, out, axis]
        if kind == "d2u":
            out = _node_out_idx(node)
            axis0 = int(node.kwargs.get("axis0", 0))
            axis1 = int(node.kwargs.get("axis1", axis0))
            if axis0 != int(x_axis) or axis1 != int(x_axis):
                raise ValueError(f"ODE surrogate table only supports D2U({x_axis},{x_axis})")
            if hess is None:
                raise RuntimeError("ODE surrogate Hessians unavailable")
            return hess[:, out, axis0, axis1]
        if kind == "var":
            axis = int(node.var_idxs[0]) if node.var_idxs else 0
            if axis != int(x_axis):
                raise ValueError(f"ODE surrogate table only supports Var({x_axis}), got Var({axis})")
            return X[:, axis]
        raise ValueError(f"Unsupported ODE table atom: {kind}")
    if isinstance(node, PowNode):
        base = _eval_ode_node_surrogate_table(node.base, X, values, grads, hess, x_axis=x_axis)
        if isinstance(node.exponent, (int, float)):
            return torch.pow(base, float(node.exponent))
        exp = _eval_ode_node_surrogate_table(node.exponent, X, values, grads, hess, x_axis=x_axis)
        return torch.pow(base, exp)
    if isinstance(node, MulNode):
        left = _eval_ode_node_surrogate_table(node.left, X, values, grads, hess, x_axis=x_axis)
        right = _eval_ode_node_surrogate_table(node.right, X, values, grads, hess, x_axis=x_axis)
        return left * right
    if isinstance(node, AddNode):
        left = _eval_ode_node_surrogate_table(node.left, X, values, grads, hess, x_axis=x_axis)
        right = _eval_ode_node_surrogate_table(node.right, X, values, grads, hess, x_axis=x_axis)
        return left + right
    if isinstance(node, SinNode):
        arg = _eval_ode_node_surrogate_table(node.arg, X, values, grads, hess, x_axis=x_axis)
        return torch.sin(arg)
    if isinstance(node, CosNode):
        arg = _eval_ode_node_surrogate_table(node.arg, X, values, grads, hess, x_axis=x_axis)
        return torch.cos(arg)
    raise ValueError(f"Unsupported ODE table node: {node!r}")


def _ode_surrogate_table_design(
    trajectory_surrogates: Sequence[CombinedSurrogate],
    x_by_traj: Sequence[torch.Tensor],
    problem: ComplexProblemDef,
    *,
    terms: Sequence[Node],
) -> tuple[torch.Tensor, list[torch.Tensor], list[Node | None]]:
    pid = problem.id
    ncomp = NCOMPONENTS.get(pid, 2)
    nxvars = NXVARS.get(pid, 1)
    x_axis = _normalized_problem_x_axis(problem, nxvars)
    anchor_order = ANCHOR_ORDER.get(pid, problem.order)
    term_asts: list[Node | None] = [None] + list(terms)
    need_hess = int(anchor_order) == 2 or any(_node_contains_atom_kind(term, "d2u") for term in term_asts)
    cols_by_term: list[list[torch.Tensor]] = [[] for _ in term_asts]
    targets_by_eq: list[list[torch.Tensor]] = [[] for _ in range(int(ncomp))]

    if len(trajectory_surrogates) != len(x_by_traj):
        raise ValueError("trajectory_surrogates and x_by_traj must have the same length")

    for combined, X_raw in zip(trajectory_surrogates, x_by_traj):
        X = X_raw.to(dtype=torch.float64)
        values = combined.forward(X).detach().to(dtype=torch.float64)
        grads = combined.grad(X).detach().to(dtype=torch.float64)
        hess = combined.grad_grad(X).detach().to(dtype=torch.float64) if need_hess else None
        for k, term in enumerate(term_asts):
            cols_by_term[k].append(
                _eval_ode_node_surrogate_table(
                    term,
                    X,
                    values,
                    grads,
                    hess,
                    x_axis=int(x_axis),
                )
            )
        for ci in range(int(ncomp)):
            if int(anchor_order) == 2:
                if hess is None:
                    raise RuntimeError("ODE surrogate Hessians unavailable")
                targets_by_eq[ci].append(hess[:, ci, int(x_axis), int(x_axis)])
            elif int(anchor_order) == 1:
                targets_by_eq[ci].append(grads[:, ci, int(x_axis)])
            elif int(anchor_order) == 0:
                targets_by_eq[ci].append(values[:, ci])
            else:
                raise ValueError(f"Unsupported ODE table anchor_order={anchor_order}")

    Phi = torch.stack([torch.cat(parts, dim=0) for parts in cols_by_term], dim=1)
    targets = [torch.cat(parts, dim=0) for parts in targets_by_eq]
    finite = torch.isfinite(Phi).all(dim=1)
    for target in targets:
        finite &= torch.isfinite(target)
    Phi = Phi[finite]
    targets = [target[finite] for target in targets]
    return Phi, targets, term_asts


def discover_sparse_ode_surrogate_system(
    problem: ComplexProblemDef,
    trajectory_surrogates: Sequence[CombinedSurrogate],
    x_by_traj: Sequence[torch.Tensor],
    *,
    stlsq_lambda: float | None = None,
) -> tuple[SystemDESearchResult, str, int, float]:
    pid = problem.id
    ncomp = NCOMPONENTS.get(pid, 2)
    nxvars = NXVARS.get(pid, 1)
    anchor_order = ANCHOR_ORDER.get(pid, problem.order)
    if not trajectory_surrogates:
        raise RuntimeError("No ODE trajectory surrogates supplied")
    all_t = torch.cat([X[:, 0].detach().cpu().to(dtype=torch.float64) for X in x_by_traj], dim=0).numpy()
    variants = build_class_sparse_library_variants(problem, coord_mins=np.array([float(np.min(all_t))]))
    if not variants:
        raise RuntimeError(f"No class sparse library variants generated for {pid}")
    supported_variants = [
        (name, terms)
        for name, terms in variants
        if _class_variant_dimensionally_supported(problem, terms)
    ]
    if supported_variants:
        variants = supported_variants

    candidates: list[tuple[float, float, SystemDESearchResult, str, int, float, int]] = []
    errors: list[str] = []
    accept_nrmse = _class_sparse_accept_nrmse(problem)

    for variant_name, terms in variants:
        lam = (
            _class_sparse_default_lambda(len(terms), anchor_order=anchor_order, nxvars=nxvars)
            if stlsq_lambda is None
            else float(stlsq_lambda)
        )
        try:
            Phi, targets, term_asts_all = _ode_surrogate_table_design(
                trajectory_surrogates,
                x_by_traj,
                problem,
                terms=terms,
            )
            if Phi.shape[0] < 10:
                raise RuntimeError(f"Too few finite surrogate rows: {Phi.shape[0]}")
            ys = [-target for target in targets]
            keeps = []
            for y in ys:
                _c, keep_i = stlsq(Phi, y, ridge=1.0e-10, lam=float(lam), max_iter=10)
                keeps.append(keep_i)
            keep = torch.stack(keeps, dim=0).any(dim=0)
            if int(keep.sum()) == 0:
                continue
            Csel = torch.zeros((int(ncomp), int(keep.sum())), dtype=torch.float64)
            rms_train: list[float] = []
            for eq_idx, target in enumerate(targets):
                y = -target
                Csel[eq_idx] = ridge_lstsq(Phi[:, keep], y, ridge=0.0)
                residual = target + Phi[:, keep] @ Csel[eq_idx]
                rms_train.append(float(residual.square().mean().sqrt().detach().cpu()))
            term_sel = [term for term, k in zip(term_asts_all, keep.tolist()) if k]
            result = SystemDESearchResult(
                order=int(anchor_order),
                x_axis=_normalized_problem_x_axis(problem, nxvars),
                out_idxs=tuple(range(int(ncomp))),
                term_asts=term_sel,
                coeffs=Csel.detach().cpu(),
                rms_train=rms_train,
                rms_val=None,
            )
            result.residual_asts = build_system_residual_asts(result)
        except Exception as exc:
            errors.append(f"{variant_name}: {exc}")
            continue

        anchor_scale = max(
            float(np.mean([float(target.square().mean().sqrt().detach().cpu()) for target in targets])),
            1.0e-12,
        )
        mean_rms = float(sum(result.rms_train) / max(1, len(result.rms_train)))
        nrmse = mean_rms / anchor_scale
        n_active = int((result.coeffs.abs() > 1.0e-12).sum())
        score = nrmse + 5.0e-3 * float(n_active) + 1.0e-3 * float(len(terms))
        candidates.append((score, nrmse, result, variant_name, len(terms), float(lam), n_active))

    if not candidates:
        detail = "; ".join(errors[:5])
        raise RuntimeError(f"No ODE surrogate sparse variant succeeded for {pid}: {detail}")

    selection_pool = candidates
    if _problem_supports_modulus_nonlinearity(problem):
        nonlinear_candidates = [cand for cand in candidates if _is_modulus_class_variant(cand[3])]
        if nonlinear_candidates:
            selection_pool = nonlinear_candidates

    for _score, nrmse, result, variant_name, n_terms, lam, _n_active in selection_pool:
        if nrmse <= accept_nrmse:
            return result, variant_name, int(n_terms), float(lam)

    _score, _nrmse, result, variant_name, n_terms, lam, _n_active = min(selection_pool, key=lambda item: item[0])
    return result, variant_name, int(n_terms), float(lam)


def _build_discovered_map_system(result: Any) -> dict[tuple[int, str], float]:
    discovered: dict[tuple[int, str], float] = {}
    for eq_idx in range(len(result.out_idxs)):
        for term, coeff in zip(result.term_asts, result.coeffs[eq_idx].tolist()):
            term_key = "const" if term is None else repr(term)
            discovered[(eq_idx, term_key)] = float(coeff)
    return discovered


def _build_sparse_rows_from_map(
    discovered: dict[tuple[int, str], float],
    *,
    coef_tol: float = 1e-12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (eq_idx, term_key), coeff in sorted(discovered.items(), key=lambda item: (item[0][0], item[0][1])):
        coeff_f = float(coeff)
        if abs(coeff_f) <= float(coef_tol):
            continue
        rows.append({
            "eq": int(eq_idx),
            "term": str(term_key),
            "coeff": coeff_f,
        })
    return rows


def _build_sparse_discovered_rows(result: Any, *, coef_tol: float = 1e-12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eq_idx in range(len(result.out_idxs)):
        for term, coeff in zip(result.term_asts, result.coeffs[eq_idx].tolist()):
            coeff_f = float(coeff)
            if abs(coeff_f) <= float(coef_tol):
                continue
            term_key = "const" if term is None else repr(term)
            rows.append({
                "eq": int(eq_idx),
                "term": term_key,
                "coeff": coeff_f,
            })
    return rows


def validate_sparse_result(
    discovered: dict[tuple[int, str], float],
    discovered_order: int,
    gt: ComplexGroundTruth,
) -> tuple[str, str]:
    """Validate sparse/STLSQ coefficients against the registered ground truth."""
    if discovered_order != gt.order:
        return "FAIL", f"Wrong order: expected {gt.order}, got {discovered_order}"

    messages: list[str] = []
    missing: list[tuple[int, str]] = []
    for key, expected_val_raw in gt.system_coeffs.items():
        expected_val = float(expected_val_raw)
        if key not in discovered:
            missing.append(key)
            continue
        disc_val = float(discovered[key])
        abs_err = abs(disc_val - expected_val)
        rel_err = abs_err / max(abs(expected_val), 1e-6)
        if rel_err > gt.coeff_rtol and abs_err > gt.coeff_atol:
            eq_idx, term_label = key
            messages.append(
                f"eq{eq_idx} {term_label}: expected {expected_val:.4g}, "
                f"got {disc_val:.4g} (rel={rel_err:.1%})"
            )

    if missing:
        return "FAIL", f"Missing expected terms: {missing}"

    expected_keys = set(gt.system_coeffs.keys())
    for key, val_raw in discovered.items():
        val = float(val_raw)
        if key not in expected_keys and abs(val) > gt.decoy_atol:
            eq_idx, term_label = key
            messages.append(f"eq{eq_idx} spurious {term_label}: {val:.4g}")

    if messages:
        return "PARTIAL", "Coefficient issues: " + "; ".join(messages)

    max_rel = 0.0
    for key, expected_val_raw in gt.system_coeffs.items():
        expected_val = float(expected_val_raw)
        disc_val = float(discovered.get(key, 0.0))
        abs_err = abs(disc_val - expected_val)
        rel_err = abs_err / max(abs(expected_val), 1e-6)
        max_rel = max(max_rel, rel_err)

    return "PASS", f"max relative error: {max_rel:.1%}"


def run_problem_sparse(
    problem: ComplexProblemDef,
    *,
    data_dir: Path,
    results_dir: Path,
    fast: bool = False,
    verbose: bool = False,
    n_traj: int = 6,
    n_points: int = 5000,
    seed: int = 0,
    skip_generate: bool = False,
    stlsq_lambda: float | None = None,
    sparse_library: str = "class",
) -> dict[str, Any]:
    """Run one problem with the component-surrogate STLSQ engine."""
    pid = problem.id
    sparse_library = _normalize_sparse_library_policy(sparse_library)
    result: dict[str, Any] = {
        "id": pid,
        "description": problem.description,
        "engine": "sparse",
        "sparse_library": sparse_library,
        "status": "ERROR",
        "message": "",
    }

    if pid not in RHS_REGISTRY and pid not in PDE_DATA_GENERATORS:
        result["status"] = "SKIP"
        result["message"] = "No RHS function or PDE data generator registered"
        return result
    if pid not in GROUND_TRUTH_BUILDERS:
        result["status"] = "SKIP"
        result["message"] = "No ground truth defined"
        return result
    if sparse_library != "class":
        result["status"] = "ERROR"
        result["message"] = f"Unsupported sparse library policy: {sparse_library}"
        return result

    params = DEFAULT_PARAMS.get(pid, {})
    ics = DEFAULT_ICS.get(pid, {})
    t_max = DEFAULT_TMAX.get(pid, 10.0)
    ncomp = NCOMPONENTS.get(pid, 2)
    nxvars = NXVARS.get(pid, 1)
    result["param_values"] = params
    result["n_traj"] = int(n_traj)
    result["n_points"] = int(n_points)

    device = torch.device("cpu")
    dtype = torch.float64
    epochs = 1500 if fast else 5000
    num_segments = 48
    surrogate_loss_target = 1.0e-8 if fast else 1.0e-10

    if pid in RHS_REGISTRY:
        try:
            print("  Sparse ODE path: multi-trajectory surrogate STLSQ")
            csv_paths_by_traj = _ode_trajectory_component_csv_paths(
                data_dir,
                pid,
                n_traj=max(1, int(n_traj)),
                ncomp=int(ncomp),
            )
            all_exist = all(path.exists() for row in csv_paths_by_traj for path in row)
            if not skip_generate or not all_exist:
                csv_paths_by_traj = generate_sparse_ode_trajectory_component_data(
                    problem,
                    params,
                    ics,
                    t_max,
                    data_dir,
                    n_traj=n_traj,
                    n_points=n_points,
                    seed=seed,
                )
                if verbose:
                    flat_paths = [str(path) for row in csv_paths_by_traj for path in row]
                    print(f"  Generated {flat_paths}")

            trajectory_surrogates, x_by_traj, val_losses = train_ode_trajectory_surrogates(
                csv_paths_by_traj,
                nxvars=nxvars,
                num_segments=num_segments,
                epochs=epochs,
                device=device,
                dtype=dtype,
                loss_target=surrogate_loss_target,
                verbose=verbose,
            )
            for traj_idx, losses in enumerate(val_losses):
                for ci, val_loss in enumerate(losses):
                    label = _COMP_LABELS[ci] if ci < len(_COMP_LABELS) else f"c{ci}"
                    result[f"val_loss_traj{traj_idx}_{label}"] = float(val_loss)
            result["n_traj"] = int(len(trajectory_surrogates))
            result["ode_sparse_path"] = "surrogate_multi_trajectory"

            disc, class_variant, candidate_count, lam = discover_sparse_ode_surrogate_system(
                problem,
                trajectory_surrogates,
                x_by_traj,
                stlsq_lambda=stlsq_lambda,
            )
            result["class_library_variant"] = class_variant
            result["discovered_equations"] = disc.format_system(var_name="x0")
            result["discovered_order"] = int(disc.order)
            result["rms_train"] = [float(v) for v in disc.rms_train]
            result["rms_val"] = []
            result["library_terms"] = int(candidate_count)
            result["n_features"] = int(candidate_count)
            discovered_map = _build_discovered_map_system(disc)
            result["discovered"] = _build_sparse_rows_from_map(discovered_map)
            print(
                f"  Class library variant: {class_variant} ({candidate_count} candidate terms, "
                f"ncomp={ncomp}, nxvars={nxvars}, x_axis={problem.x_axis}, "
                f"anchor_order={ANCHOR_ORDER.get(pid, problem.order)}, stlsq_lambda={lam:g})"
            )
            print(
                "  Discovered (class_ode_surrogate):\n    "
                + str(result["discovered_equations"]).replace("\n", "\n    ")
            )
        except Exception as exc:
            result["message"] = f"ODE surrogate discovery failed: {exc}"
            traceback.print_exc()
            return result

        gt = GROUND_TRUTH_BUILDERS[pid](params)
        status, message = validate_sparse_result(discovered_map, int(disc.order), gt)
        result["status"] = status
        result["message"] = message
        return result

    comp_labels = _COMP_LABELS[:ncomp]
    csv_paths = [data_dir / f"{pid}_{label}.csv" for label in comp_labels]
    all_exist = all(path.exists() for path in csv_paths)

    if not skip_generate or not all_exist:
        try:
            csv_paths = generate_sparse_component_data(
                problem,
                params,
                ics,
                t_max,
                data_dir,
                n_traj=n_traj,
                n_points=n_points,
                seed=seed,
            )
            if verbose:
                print(f"  Generated {[str(path) for path in csv_paths]}")
        except Exception as exc:
            result["message"] = f"Data generation failed: {exc}"
            traceback.print_exc()
            return result

    surrogates: list[torch.nn.Module] = []
    try:
        for ci, csv_path in enumerate(csv_paths):
            label = comp_labels[ci]
            print(f"  Training {label}-component surrogate...")
            surr, val_loss = train_component_surrogate(
                csv_path,
                num_segments,
                epochs,
                device,
                dtype,
                verbose=verbose,
                loss_target=surrogate_loss_target,
                nxvars=nxvars,
            )
            print(f"    val_loss = {val_loss:.6e}")
            surrogates.append(surr)
            result[f"val_loss_{label}"] = float(val_loss)
    except Exception as exc:
        result["message"] = f"Surrogate training failed: {exc}"
        traceback.print_exc()
        return result

    try:
        data_hp = DataHyperparams(batch_size=2000, ndata_select=2000, ndata_select_val=2000)
        _ds_tr, _ds_va, dl_tr, _dl_va = build_datasets(str(csv_paths[0]), nxvars, np.float64, data_hp, None)
        X = torch.cat([batch[0] for batch in dl_tr], dim=0).to(dtype=dtype)

        combined = CombinedSurrogate(*surrogates)
        print(f"  Sparse library policy: {sparse_library}")

        if sparse_library == "class":
            disc, class_variant, candidate_count, lam = discover_sparse_class_system(
                combined,
                X,
                problem,
                stlsq_lambda=stlsq_lambda,
            )
            print(
                f"  Class library variant: {class_variant} ({candidate_count} candidate terms, "
                f"ncomp={ncomp}, nxvars={nxvars}, x_axis={problem.x_axis}, "
                f"anchor_order={ANCHOR_ORDER.get(pid, problem.order)}, stlsq_lambda={lam:g})"
            )
            mode = "class_system"
            disc_order = int(disc.order)
            discovered_map = _build_discovered_map_system(disc)
            eqs_str = disc.format_system(var_name="x0")
            rms_train = [float(v) for v in disc.rms_train]
            rms_val = [] if disc.rms_val is None else [float(v) for v in disc.rms_val]
            library_terms = int(candidate_count)
            result["class_library_variant"] = class_variant

        result["discovered_equations"] = eqs_str
        result["discovered_order"] = int(disc_order)
        result["rms_train"] = rms_train
        result["rms_val"] = rms_val
        result["library_terms"] = int(library_terms)
        result["n_features"] = int(library_terms)
        result["discovered"] = _build_sparse_rows_from_map(discovered_map)
        print(f"  Discovered ({mode}):\n    " + eqs_str.replace("\n", "\n    "))
    except Exception as exc:
        result["message"] = f"Discovery failed: {exc}"
        traceback.print_exc()
        return result

    gt = GROUND_TRUTH_BUILDERS[pid](params)
    status, message = validate_sparse_result(discovered_map, int(disc_order), gt)
    result["status"] = status
    result["message"] = message
    return result


# ---------------------------------------------------------------------------
# factorized symbolic search discovery
# ---------------------------------------------------------------------------


def run_discovery_single_eq(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_probe: np.ndarray,
    y_probe: np.ndarray,
    *,
    n_iter: int = 50000,
    max_depth: int = 6,
    poly_degree: int = 4,
    seed: int = 0,
    verbose: bool = False,
    early_stop_mse: float = 1e-10,
    n_seeds: int = 10,
    fast: bool = False,
    linear_accept_nrmse: float | None = None,
    var_dims: Sequence[DimVec] | None = None,
    y_dims: DimVec | None = None,
) -> dict[str, Any]:
    """Run continuous skeleton refinement on a single equation.

    Parameters are the feature/target arrays for ONE equation.
    Returns dict with ``expr``, ``mapping``, ``mse``, ``expr_str``, ``size``.
    """
    nvars = x_fit.shape[1]
    x_fit_t = torch.as_tensor(x_fit, dtype=torch.float64)
    y_fit_t = torch.as_tensor(y_fit.reshape(-1, 1), dtype=torch.float64)
    x_probe_t = torch.as_tensor(x_probe, dtype=torch.float64)
    y_probe_t = torch.as_tensor(y_probe.reshape(-1, 1), dtype=torch.float64)

    best_result: dict[str, Any] | None = None
    best_score = float("inf")

    linear_prefit = _try_linear_prefit(
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        var_dims=var_dims,
        y_dims=y_dims,
    )
    if linear_prefit is not None:
        best_result = linear_prefit
        best_score = float(linear_prefit["mse"])
        if verbose:
            print(
                f"      linear prefit: mse={best_score:.3e} "
                f"nrmse={float(linear_prefit['nrmse']):.3e} "
                f"expr={linear_prefit['expr_str']}"
            )
        if best_score < early_stop_mse:
            return best_result
        if linear_accept_nrmse is not None and float(linear_prefit["nrmse"]) <= float(linear_accept_nrmse):
            return best_result

    n_iter_each = max(1, n_iter // n_seeds)

    # Tune plus-refinement budget to the search budget.
    if fast:
        if nvars >= 20:
            refine_max_trials = 20
            refine_trials_per_brute_depth = 3
            refine_trials_per_mutation_window = 3
            refine_mutation_window = 180
            brute_max_expressions = 120
        elif nvars >= 14:
            refine_max_trials = 80
            refine_trials_per_brute_depth = 6
            refine_trials_per_mutation_window = 6
            refine_mutation_window = 140
            brute_max_expressions = 300
        else:
            refine_max_trials = 120
            refine_trials_per_brute_depth = 8
            refine_trials_per_mutation_window = 8
            refine_mutation_window = 120
            brute_max_expressions = 400
    else:
        refine_max_trials = 1500
        refine_trials_per_brute_depth = 64
        refine_trials_per_mutation_window = 64
        refine_mutation_window = 500
        brute_max_expressions = 1000

    for si in range(n_seeds):
        seed_search = seed + si
        arch = run_explorer_core(
            target_fn=lambda _x: _x[:, :1] * float("nan"),
            nvars=nvars,
            var_dims=var_dims,
            y_dims=y_dims,
            n_iter=n_iter_each,
            max_depth=max_depth,
            poly_degree=poly_degree,
            lo=0.0,
            hi=1.0,
            seed=seed,
            seed_search=seed_search,
            dtype=torch.float64,
            x_fit_data=x_fit_t,
            y_fit_data=y_fit_t,
            x_probe_data=x_probe_t,
            y_probe_data=y_probe_t,
            early_stop_mse=early_stop_mse,
            brute_max_expressions=brute_max_expressions,
            refine_enable=True,
            refine_lbfgs_steps=15,
            refine_fit_subset=256,
            refine_num_restarts=2,
            refine_max_variants=4,
            refine_max_params=2,
            refine_linear_combo_enable=True,
            refine_linear_terms_max=6,
            refine_linear_prune_rel=1e-10,
            refine_gate_best_factor=10.0,
            refine_max_trials=refine_max_trials,
            refine_trials_per_brute_depth=refine_trials_per_brute_depth,
            refine_trials_per_mutation_window=refine_trials_per_mutation_window,
            refine_mutation_window=refine_mutation_window,
            refine_safe_eps=1e-6,
            refine_safe_penalty_weight=1e-2,
            refine_safe_exp_clip=30.0,
            refine_theta_l2=1e-4,
            refine_init_log_min=-1.5,
            refine_init_log_max=1.5,
            stall_window=500,
            stall_patience=3,
            stall_delta=1e-4,
            verbose=verbose,
        )

        for rec in arch.best(5):
            score = float(rec.best_mse)
            if score < best_score:
                best_score = score
                best_result = {
                    "expr": rec.best_expr,
                    "mapping": rec.mapping,
                    "mse": score,
                    "expr_str": node_str(rec.best_expr),
                    "size": int(node_size(rec.best_expr)),
                }

        if best_score < early_stop_mse:
            if verbose:
                print(f"      Early stop: mse={best_score:.3e}")
            break

    if best_result is None:
        return {"expr": None, "mapping": None, "mse": float("inf"), "expr_str": "", "size": 0}
    return best_result


def run_discovery(
    features: np.ndarray,
    targets: list[np.ndarray],
    ncomp: int,
    *,
    n_iter: int = 50000,
    max_depth: int = 6,
    seed: int = 0,
    verbose: bool = False,
    n_fit: int = 6000,
    n_probe: int = 6000,
    n_seeds: int = 10,
    fast: bool = False,
    linear_accept_nrmse: float | None = None,
    feature_dims: Sequence[DimVec] | None = None,
    target_dims: Sequence[DimVec] | None = None,
) -> list[dict[str, Any]]:
    """Run factorized symbolic search per equation.  Returns a list of result dicts."""
    N = features.shape[0]

    # Train/probe split: first n_fit for fitting, rest for probing.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_fit_eff = min(n_fit, int(0.7 * N))
    n_probe_eff = min(n_probe, N - n_fit_eff)

    fit_idx = idx[:n_fit_eff]
    probe_idx = idx[n_fit_eff : n_fit_eff + n_probe_eff]

    x_fit = features[fit_idx]
    x_probe = features[probe_idx]

    results: list[dict[str, Any]] = []
    for eq_idx in range(ncomp):
        if verbose:
            print(f"    Equation {eq_idx}/{ncomp}: discovering RHS...")
        y_fit = targets[eq_idx][fit_idx]
        y_probe = targets[eq_idx][probe_idx]
        eq_y_dims = None
        if target_dims is not None and eq_idx < len(target_dims):
            eq_y_dims = tuple(target_dims[eq_idx])

        res = run_discovery_single_eq(
            x_fit, y_fit, x_probe, y_probe,
            n_iter=n_iter,
            max_depth=max_depth,
            seed=seed + eq_idx * 1000,
            verbose=verbose,
            n_seeds=n_seeds,
            fast=fast,
            linear_accept_nrmse=linear_accept_nrmse,
            var_dims=feature_dims,
            y_dims=eq_y_dims,
        )
        results.append(res)
        if verbose:
            print(f"      eq{eq_idx}: mse={res['mse']:.3e}  expr={res['expr_str']}")

    return results


# ---------------------------------------------------------------------------
# Validation – ODE simulation
# ---------------------------------------------------------------------------


def _build_ode_system_rhs(
    discovered: list[dict[str, Any]],
    pid: str,
    params: dict[str, float],
    ncomp: int,
    order: int,
    anchor_order: int,
) -> Any:
    """Build a callable system RHS from per-equation discovered expressions.

    Returns a function ``rhs(t, state) -> list[float]`` compatible with
    ``solve_ivp``.
    """
    ctx = _make_ode_feature_context(pid, params, ncomp, int(anchor_order))
    predictors = []
    for disc in discovered:
        if disc.get("expr") is None and disc.get("expr_ast") is None:
            predictors.append(None)
        else:
            predictors.append(factorized_search_candidate_to_feature_predictor(disc))

    def system_rhs(t: float, state: Sequence[float]) -> list[float]:
        feats = _build_ode_feature_values(
            pid,
            params,
            t,
            state,
            ncomp,
            order,
            int(anchor_order),
            ctx=ctx,
        )

        # Evaluate each equation.
        derivs: list[float] = []
        for ci in range(ncomp):
            predictor = predictors[ci]
            if predictor is None:
                derivs.append(0.0)
            else:
                derivs.append(float(predictor(feats)))

        if order == 2:
            # For 2nd order: state = [u0, v0, du0, dv0]
            # Return [du0, dv0, d2u0, d2v0]
            velocities = [float(state[ncomp + ci]) for ci in range(ncomp)]
            return velocities + derivs
        return derivs

    return system_rhs


def validate_ode_simulation(
    discovered: list[dict[str, Any]],
    trajs: list[dict[str, Any]],
    pid: str,
    params: dict[str, float],
    ncomp: int,
    order: int,
    anchor_order: int,
    *,
    pass_nrmse: float = 0.01,
    partial_nrmse: float = 0.05,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Validate ODE discovery by simulating and comparing trajectories."""
    try:
        rhs_fn = _build_ode_system_rhs(discovered, pid, params, ncomp, order, anchor_order)
    except Exception as exc:
        return "ERROR", f"Failed to build system RHS: {exc}", []

    traj_scores: list[dict[str, Any]] = []
    for k, traj in enumerate(trajs):
        t = traj["t"]
        state_true = traj["state"]  # (n_state, N)
        y0 = traj["y0"]

        try:
            sol = solve_ivp(
                rhs_fn,
                [float(t[0]), float(t[-1])],
                y0,
                t_eval=t,
                method="RK45",
                rtol=1e-9,
                atol=1e-11,
            )
        except Exception as exc:
            return "FAIL", f"Integration error on traj {k}: {exc}", traj_scores

        if sol.status != 0:
            return "FAIL", f"Integration failed on traj {k}: {sol.message}", traj_scores

        # Compare all components.
        comp_nrmses: list[float] = []
        for ci in range(ncomp):
            nrmse = normalized_rmse(state_true[ci], np.asarray(sol.y[ci], dtype=np.float64))
            comp_nrmses.append(nrmse)
        max_nrmse = max(comp_nrmses)
        traj_scores.append({
            "traj_id": k,
            "nrmse": max_nrmse,
            "comp_nrmses": comp_nrmses,
        })

    if not traj_scores:
        return "FAIL", "No trajectories for validation", traj_scores

    mean_e = float(sum(t["nrmse"] for t in traj_scores) / len(traj_scores))
    max_e = float(max(t["nrmse"] for t in traj_scores))
    msg = f"NRMSE mean={mean_e:.3g} max={max_e:.3g}"

    if max_e < pass_nrmse:
        return "PASS", msg, traj_scores
    if max_e < partial_nrmse:
        return "PARTIAL", msg, traj_scores
    return "FAIL", msg, traj_scores


# ---------------------------------------------------------------------------
# Validation – PDE residual
# ---------------------------------------------------------------------------


def validate_pde_residual(
    discovered: list[dict[str, Any]],
    features: np.ndarray,
    targets: list[np.ndarray],
    ncomp: int,
    *,
    pass_nrmse: float = 0.01,
    partial_nrmse: float = 0.05,
    n_test: int = 2000,
    seed: int = 42,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Validate PDE discovery by evaluating residual on held-out points."""
    N = features.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)[:min(n_test, N)]

    eq_scores: list[dict[str, Any]] = []
    for ci in range(ncomp):
        expr = discovered[ci].get("expr")
        if expr is None:
            eq_scores.append({"eq": ci, "nrmse": float("inf")})
            continue

        y_true = targets[ci][idx]
        yhat = evaluate_factorized_search_candidate(discovered[ci], features[idx], dtype=torch.float64)
        nrmse = normalized_rmse(y_true, yhat)
        eq_scores.append({"eq": ci, "nrmse": float(nrmse)})

    max_e = max(s["nrmse"] for s in eq_scores)
    mean_e = float(sum(s["nrmse"] for s in eq_scores) / len(eq_scores))
    msg = f"Residual NRMSE mean={mean_e:.3g} max={max_e:.3g}"

    if max_e < pass_nrmse:
        return "PASS", msg, eq_scores
    if max_e < partial_nrmse:
        return "PARTIAL", msg, eq_scores
    return "FAIL", msg, eq_scores


# ---------------------------------------------------------------------------
# Per-problem runner
# ---------------------------------------------------------------------------


def run_problem_factorized_search(
    problem: ComplexProblemDef,
    *,
    data_dir: Path,
    results_dir: Path,
    fast: bool = False,
    verbose: bool = False,
    n_traj: int = 6,
    n_points: int = 5000,
    seed: int = 0,
    no_sim_validate: bool = False,
    pass_nrmse: float = 0.01,
    partial_nrmse: float = 0.05,
) -> dict[str, Any]:
    pid = problem.id
    result: dict[str, Any] = {
        "id": pid,
        "description": problem.description,
        "engine": "factorized_search",
        "status": "ERROR",
        "message": "",
    }

    ncomp = NCOMPONENTS.get(pid, 2)
    nxvars = NXVARS.get(pid, 1)
    order = problem.order
    anchor_order = ANCHOR_ORDER.get(pid, order)
    params = DEFAULT_PARAMS.get(pid, {})
    ics = DEFAULT_ICS.get(pid, {})
    t_max = DEFAULT_TMAX.get(pid, 10.0)
    result["param_values"] = params

    is_ode = pid in RHS_REGISTRY
    is_pde = pid in PDE_DATA_GENERATORS

    if not is_ode and not is_pde:
        result["status"] = "SKIP"
        result["message"] = "No RHS function or PDE data generator registered"
        return result

    problem_seed = _problem_seed(seed, pid)

    # factorized symbolic search budgets.
    if fast:
        n_iter = 4000
        max_depth = 4
        n_fit = 3000
        n_probe = 3000
        n_seeds = 1
    else:
        n_iter = 60000
        max_depth = 6
        n_fit = 6000
        n_probe = 6000
        n_seeds = 10

    # ------------------------------------------------------------------
    # 1. Generate data and build feature tables
    # ------------------------------------------------------------------
    try:
        if is_ode:
            trajs = generate_ode_multi_traj(
                pid, params, ncomp, order, t_max, ics,
                n_traj=n_traj, n_points=n_points, seed=problem_seed,
            )
            features, targets, feat_names, feature_dims, target_dims = build_ode_feature_table(
                pid, params, trajs, ncomp, order, anchor_order,
            )
            if verbose:
                print(f"  ODE: {len(trajs)} trajectories, {features.shape[0]} points")
                print(f"  Features ({features.shape[1]}): {feat_names}")
        else:
            pde_data = generate_pde_data(pid, params, t_max, n_points)
            derivative_candidates: list[tuple[str, bool, bool]] = [("default", True, True)]
            if nxvars >= 2:
                derivative_candidates = [
                    ("first+second", True, True),
                    ("first", True, False),
                    ("second", False, True),
                ]

            best_base: tuple[
                float,
                int,
                str,
                np.ndarray,
                list[np.ndarray],
                list[str],
                list[DimVec] | None,
                list[DimVec] | None,
                bool,
                bool,
            ] | None = None
            for mode_name, use_first, use_second in derivative_candidates:
                f_cand, t_cand, n_cand, d_cand, y_d_cand = build_pde_feature_table(
                    pid,
                    params,
                    pde_data,
                    ncomp,
                    nxvars,
                    anchor_order,
                    include_nonlinear_terms=False,
                    include_first_spatial=use_first,
                    include_second_spatial=use_second,
                )
                prefit = _estimate_linear_prefit_nrmse(
                    f_cand,
                    t_cand,
                    seed=problem_seed + 17 + len(n_cand),
                    n_fit=n_fit,
                    n_probe=n_probe,
                    feature_dims=d_cand,
                    target_dims=y_d_cand,
                )
                # Small complexity penalty prefers simpler feature sets unless
                # there is a clear prefit advantage.
                score = float(prefit) + 2e-4 * float(f_cand.shape[1])
                key = (score, int(f_cand.shape[1]))
                if best_base is None or key < (best_base[0], best_base[1]):
                    best_base = (
                        float(score),
                        int(f_cand.shape[1]),
                        mode_name,
                        f_cand,
                        t_cand,
                        n_cand,
                        d_cand,
                        y_d_cand,
                        use_first,
                        use_second,
                    )

            if best_base is None:
                raise RuntimeError("No PDE feature candidate generated")

            derivative_mode = best_base[2]
            features = best_base[3]
            targets = best_base[4]
            feat_names = best_base[5]
            feature_dims = best_base[6]
            target_dims = best_base[7]
            use_first = best_base[8]
            use_second = best_base[9]
            base_prefit_nrmse = _estimate_linear_prefit_nrmse(
                features,
                targets,
                seed=problem_seed + 29,
                n_fit=n_fit,
                n_probe=n_probe,
                feature_dims=feature_dims,
                target_dims=target_dims,
            )

            use_nonlinear = False
            nonlinear_prefit_nrmse = float("inf")
            if ncomp >= 2:
                features_nl, targets_nl, feat_names_nl, feature_dims_nl, target_dims_nl = build_pde_feature_table(
                    pid,
                    params,
                    pde_data,
                    ncomp,
                    nxvars,
                    anchor_order,
                    include_nonlinear_terms=True,
                    include_first_spatial=use_first,
                    include_second_spatial=use_second,
                )
                nonlinear_prefit_nrmse = _estimate_linear_prefit_nrmse(
                    features_nl,
                    targets_nl,
                    seed=problem_seed + 31,
                    n_fit=n_fit,
                    n_probe=n_probe,
                    feature_dims=feature_dims_nl,
                    target_dims=target_dims_nl,
                )
                base_score = float(base_prefit_nrmse) + 2e-4 * float(features.shape[1])
                nonlinear_score = float(nonlinear_prefit_nrmse) + 2e-4 * float(features_nl.shape[1])
                if np.isfinite(nonlinear_prefit_nrmse) and (
                    (not np.isfinite(base_prefit_nrmse))
                    or nonlinear_score + 5e-4 < base_score
                ):
                    features = features_nl
                    targets = targets_nl
                    feat_names = feat_names_nl
                    feature_dims = feature_dims_nl
                    target_dims = target_dims_nl
                    use_nonlinear = True
            result["pde_feature_policy"] = {
                "derivative_mode": derivative_mode,
                "n_features": int(features.shape[1]),
                "use_nonlinear": bool(use_nonlinear),
                "base_linear_prefit_nrmse": float(base_prefit_nrmse),
                "nonlinear_linear_prefit_nrmse": float(nonlinear_prefit_nrmse),
            }
            trajs = []  # no trajectories for PDEs
            if verbose:
                print(f"  PDE: {features.shape[0]} grid points")
                print(f"  Features ({features.shape[1]}): {feat_names}")
                if "pde_feature_policy" in result:
                    print(f"  Feature policy: {result['pde_feature_policy']}")
    except Exception as exc:
        result["message"] = f"Data generation failed: {exc}"
        traceback.print_exc()
        return result

    result["n_features"] = features.shape[1]
    result["n_points"] = features.shape[0]
    result["feature_names"] = feat_names

    # ------------------------------------------------------------------
    # 2. Run factorized symbolic search discovery
    # ------------------------------------------------------------------
    try:
        n_iter_eff = n_iter
        max_depth_eff = max_depth
        n_fit_eff = n_fit
        n_probe_eff = n_probe
        if fast and is_pde:
            if features.shape[1] >= 20:
                n_iter_eff = min(n_iter_eff, 800)
                max_depth_eff = min(max_depth_eff, 3)
                n_fit_eff = min(n_fit_eff, 1800)
                n_probe_eff = min(n_probe_eff, 1800)
            elif features.shape[1] >= 14:
                n_iter_eff = min(n_iter_eff, 2000)
                max_depth_eff = min(max_depth_eff, 4)
                n_fit_eff = min(n_fit_eff, 2200)
                n_probe_eff = min(n_probe_eff, 2200)

        linear_accept_nrmse = 5e-5 if is_ode else (2e-2 if features.shape[1] >= 18 else 1e-2)
        discovered = run_discovery(
            features, targets, ncomp,
            n_iter=n_iter_eff, max_depth=max_depth_eff, seed=problem_seed,
            verbose=verbose, n_fit=n_fit_eff, n_probe=n_probe_eff, n_seeds=n_seeds,
            fast=fast,
            linear_accept_nrmse=linear_accept_nrmse,
            feature_dims=feature_dims,
            target_dims=target_dims,
        )
    except Exception as exc:
        result["message"] = f"Discovery failed: {exc}"
        traceback.print_exc()
        return result

    # Record discovered expressions.
    result["discovered"] = []
    for ci, d in enumerate(discovered):
        result["discovered"].append({
            "eq": ci,
            "expr_str": d.get("expr_str", ""),
            "mse": float(d.get("mse", float("inf"))),
            "size": int(d.get("size", 0)),
        })

    # ------------------------------------------------------------------
    # 3. Validate
    # ------------------------------------------------------------------
    if no_sim_validate:
        result["status"] = "UNVERIFIED"
        result["message"] = "Validation disabled"
        return result

    try:
        if is_ode:
            status, message, scores = validate_ode_simulation(
                discovered, trajs, pid, params, ncomp, order, anchor_order,
                pass_nrmse=pass_nrmse, partial_nrmse=partial_nrmse,
            )
        else:
            status, message, scores = validate_pde_residual(
                discovered, features, targets, ncomp,
                pass_nrmse=pass_nrmse, partial_nrmse=partial_nrmse,
            )
        result["status"] = status
        result["message"] = message
        result["validation_scores"] = scores
    except Exception as exc:
        result["status"] = "ERROR"
        result["message"] = f"Validation failed: {exc}"
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Feynman Complex-Valued DE Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--only", type=str, default=None, help="Comma-separated problem IDs")
    parser.add_argument("--all", action="store_true", help="Run all problems")
    parser.add_argument(
        "--engine",
        type=str,
        default="factorized_search",
        choices=["sparse", "stlsq", "factorized_search", "factorized", "fss"],
        help="Discovery engine",
    )
    parser.add_argument("--fast", action="store_true", help="Reduced budgets")
    parser.add_argument("--n_traj", type=int, default=6, help="Trajectories per ODE problem")
    parser.add_argument("--n_points", type=int, default=5000, help="Points per trajectory/grid")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--pass_nrmse", type=float, default=0.01, help="NRMSE threshold for PASS")
    parser.add_argument("--partial_nrmse", type=float, default=0.05, help="NRMSE threshold for PARTIAL")
    parser.add_argument("--no_sim_validate", action="store_true", help="Skip factorized simulation/residual validation")
    parser.add_argument("--skip_generate", action="store_true", help="Reuse existing sparse-engine component CSVs")
    parser.add_argument(
        "--stlsq_lambda",
        type=float,
        default=None,
        help="Sparse/STLSQ sparsification threshold (default: adaptive by library size)",
    )
    parser.add_argument(
        "--sparse_library",
        type=str,
        default="class",
        choices=["class", "fair", "metadata"],
        help="Sparse/STLSQ candidate library policy (metadata-driven class library)",
    )
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    parser.add_argument(
        "--data_dir", type=str,
        default=str(REPO_ROOT / "data" / "feynman_complex"),
    )
    parser.add_argument(
        "--results_dir", type=str,
        default=str(REPO_ROOT / "results" / "feynman_complex"),
    )
    parser.add_argument(
        "--benchmark_file", type=str,
        default=str(BENCHMARK_FILE),
    )
    args = parser.parse_args(argv)
    engine = _normalize_engine_name(args.engine)
    if engine not in {"sparse", "factorized_search"}:
        parser.error(f"unsupported engine: {args.engine!r}")

    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_problems = load_complex_problems(args.benchmark_file)
    if args.only:
        ids = [pid.strip() for pid in args.only.split(",")]
        problems = {pid: all_problems[pid] for pid in ids if pid in all_problems}
        missing = [pid for pid in ids if pid not in all_problems]
        if missing:
            print(f"Warning: unknown problem IDs: {missing}")
    elif args.all:
        problems = all_problems
    else:
        print("Specify --only <ids> or --all")
        return 1

    if not problems:
        print("No problems to run.")
        return 1

    results = []
    for pid in sorted(problems.keys()):
        problem = problems[pid]
        print(f"\n{'=' * 70}")
        print(f"{pid}: {problem.description}")
        print(f"  Equation: {problem.equation}")
        print(f"  Order: {problem.order}, Type: {problem.type}")
        ncomp = NCOMPONENTS.get(pid, 2)
        nxvars = NXVARS.get(pid, 1)
        anchor = ANCHOR_ORDER.get(pid, problem.order)
        print(f"  ncomp={ncomp}, nxvars={nxvars}, anchor_order={anchor}")
        is_ode = pid in RHS_REGISTRY
        print(f"  Mode: {'ODE (solve_ivp)' if is_ode else 'PDE (analytic grid)'}")
        print(f"  Engine: {engine}")
        print(f"{'=' * 70}")

        if engine == "sparse":
            r = run_problem_sparse(
                problem,
                data_dir=data_dir,
                results_dir=results_dir,
                fast=args.fast,
                verbose=args.verbose,
                n_traj=args.n_traj,
                n_points=args.n_points,
                seed=args.seed,
                skip_generate=args.skip_generate,
                stlsq_lambda=args.stlsq_lambda,
                sparse_library=args.sparse_library,
            )
        else:
            r = run_problem_factorized_search(
                problem,
                data_dir=data_dir,
                results_dir=results_dir,
                fast=args.fast,
                verbose=args.verbose,
                n_traj=args.n_traj,
                n_points=args.n_points,
                seed=args.seed,
                no_sim_validate=args.no_sim_validate,
                pass_nrmse=args.pass_nrmse,
                partial_nrmse=args.partial_nrmse,
            )
        results.append(r)

        marker = {
            "PASS": "OK", "PARTIAL": "~~", "FAIL": "XX",
            "SKIP": "--", "UNVERIFIED": "??", "ERROR": "!!",
        }
        status = r["status"]
        print(f"  [{marker.get(status, '??')}] {status}: {r['message']}")
        if engine == "sparse" and r.get("discovered_equations"):
            print("    " + str(r["discovered_equations"]).replace("\n", "\n    "))
        elif r.get("discovered"):
            for d in r["discovered"]:
                print(f"    eq{d['eq']}: {d['expr_str']}  (mse={d['mse']:.3e})")

    # Summary table.
    print(f"\n{'=' * 70}")
    print(f"FEYNMAN COMPLEX DE BENCHMARK SUMMARY ({engine})")
    print("=" * 70)
    print(f"{'ID':<6} {'Description':<35} {'Status':<12} {'Details'}")
    print("-" * 70)
    for r in results:
        pid = r["id"]
        desc = r["description"][:33]
        status = r["status"]
        msg = r["message"].split("\n")[0][:40] if r["message"] else ""
        print(f"{pid:<6} {desc:<35} {status:<12} {msg}")
    print("-" * 70)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    parts = [f"{s}: {n}" for s, n in sorted(counts.items())]
    print(f"Total: {len(results)} | {' | '.join(parts)}")
    print("=" * 70)

    # Save JSON summary.
    summary_path = results_dir / _summary_name_for_engine(engine)
    problems_summary: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "id": r["id"],
            "description": r["description"],
            "engine": r.get("engine", engine),
            "status": r["status"],
            "message": r["message"],
            "param_values": r.get("param_values", {}),
            "n_features": r.get("n_features", 0),
            "discovered": r.get("discovered", []),
        }
        if engine == "sparse":
            row.update({
                "discovered_equations": r.get("discovered_equations", ""),
                "val_losses": {
                    k.replace("val_loss_", ""): v
                    for k, v in r.items()
                    if k.startswith("val_loss_")
                },
                "rms_train": r.get("rms_train", []),
                "rms_val": r.get("rms_val", []),
                "discovered_order": r.get("discovered_order", None),
                "library_terms": r.get("library_terms", 0),
                "sparse_library": r.get("sparse_library", ""),
                "class_library_variant": r.get("class_library_variant", ""),
            })
        else:
            row.update({
                "feature_names": r.get("feature_names", []),
                "validation_scores": r.get("validation_scores", []),
                "pde_feature_policy": r.get("pde_feature_policy", {}),
            })
        problems_summary.append(row)
    summary = {
        "engine": engine,
        "sparse_library": _normalize_sparse_library_policy(args.sparse_library) if engine == "sparse" else "",
        "problems": problems_summary,
        "counts": counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    if counts.get("FAIL", 0) > 0 or counts.get("ERROR", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
