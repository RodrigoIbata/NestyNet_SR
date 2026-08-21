# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Lightweight oracle-driven factorized symbolic search/continuous skeleton refinement lab runner for DE discovery.

This module treats DE discovery as supervised symbolic regression:

1. Load one or more trajectories (CSV).
2. Estimate derivatives with a pluggable derivative provider.
3. Build feature tables for each candidate order.
4. Run factorized symbolic search/continuous skeleton refinement on ``features -> highest derivative``.
5. Reconstruct RHS and residual ASTs for downstream DE tooling.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from nestynet_sr.adaptors.u_feature_leaf import UFeatureCache
from nestynet_sr.sr_core.ast_simplify import ast_node_count, simplify_ast
from nestynet_sr.sr_core.bridges import Add, ConstNode, D2U, DU, Mul, U, Var
from nestynet_sr.sr_core.problem_dims import CanonicalProblemDims, canonical_to_factorized_search_dims, dimless_dim
from nestynet_sr.sr_search.y_transforms import YTransform, build_default_y_transforms

from .config import (
    FactorizedSearchConfig,
    REFINE_OPTIMIZER_NAMES,
    REFINE_PROFILE_NAMES,
    apply_refine_mode_placement_defaults,
    apply_refine_profile,
    factorized_config_report,
)
from .bridge import factorized_search_to_nestynet, embed_mapping_in_ast
from .domain_projection import (
    domain_projection_is_acceptable,
    eval_node_with_domain_projection,
    merge_domain_projection_diagnostics,
)
from .engine.search import run_explorer_core
from .explorer import (
    _compile_linear_combo,
    _fit_best_with_cfg,
    _solve_linear_coeffs,
    eval_mapping,
    eval_node,
    make_engine_refinement_hooks,
    make_engine_runtime_hooks,
    node_size,
    node_str,
)


@dataclass(frozen=True)
class ConstantSpec:
    """Fixed scalar constant exposed as an optional feature."""

    name: str
    value: float
    dim: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DerivativeSpec:
    """Derivative estimation configuration."""

    method: str = "spline"  # spline | finite_diff | precomputed
    spline_s: float = 0.0
    spline_k: int = 3
    du_col: str | None = None
    d2u_col: str | None = None


@dataclass(frozen=True)
class DimensionSpec:
    """Optional dimensional-analysis metadata."""

    basis: tuple[str, ...]
    x_dim: tuple[float, ...]
    u_dim: tuple[float, ...]


@dataclass(frozen=True)
class TrajectoryRef:
    """Reference to one trajectory source used in multi-trajectory DE fitting."""

    id: str
    csv: str


@dataclass(frozen=True)
class DELabSpec:
    """DE oracle-lab specification."""

    id: str
    csv_paths: tuple[str, ...]
    order_candidates: tuple[int, ...] = (1, 2)
    x_axis: int = 0

    include_x: bool = True
    include_u: bool = True
    include_du: bool = True

    x_col: str = "x0"
    u_col: str = "y"
    out_idx: int = 0
    y_transform: str = "identity"
    split_mode: str = "per_traj_point"  # per_traj_point | traj_holdout
    traj_metric: str = "mean"  # mean | max
    trajectories: tuple[TrajectoryRef, ...] = ()
    fit_trajectories: tuple[TrajectoryRef, ...] = ()
    probe_trajectories: tuple[TrajectoryRef, ...] = ()

    constants: tuple[ConstantSpec, ...] = ()
    derivative: DerivativeSpec = DerivativeSpec()
    dims: DimensionSpec | None = None
    extra: dict[str, Any] | None = None

    validate_integrate_topk: int = 0

    @property
    def filepaths(self) -> tuple[str, ...]:
        """Alias for compatibility with external runner code."""
        return self.csv_paths


@dataclass(frozen=True)
class _Trajectory:
    traj_id: str
    path: str
    x: np.ndarray
    u: np.ndarray
    du: np.ndarray
    d2u: np.ndarray


@dataclass(frozen=True)
class TrajFeatures:
    """Per-trajectory tensors after derivative extraction."""

    traj_id: str
    x: torch.Tensor
    u: torch.Tensor
    du: torch.Tensor
    d2u: torch.Tensor


@dataclass(frozen=True)
class DEFeatureTensors:
    """Fit/probe tensors for DE regression in `x -> (u, du, d2u)` form."""

    x_fit: torch.Tensor
    u_fit: torch.Tensor
    du_fit: torch.Tensor
    d2u_fit: torch.Tensor
    x_probe: torch.Tensor
    u_probe: torch.Tensor
    du_probe: torch.Tensor
    d2u_probe: torch.Tensor


def _as_finite_float(v: Any, *, where: str) -> float:
    try:
        f = float(v)
    except Exception as exc:  # pragma: no cover - defensive conversion
        raise ValueError(f"{where}: expected numeric value, got {v!r} ({exc})") from exc
    if not math.isfinite(f):
        raise ValueError(f"{where}: expected finite value, got {v!r}")
    return f


def _as_positive_float(v: Any, *, where: str) -> float:
    f = _as_finite_float(v, where=where)
    if f <= 0.0:
        raise ValueError(f"{where}: expected > 0, got {f}")
    return f


def _as_nonnegative_int(v: Any, *, where: str) -> int:
    try:
        n = int(v)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"{where}: expected integer value, got {v!r} ({exc})") from exc
    if n < 0:
        raise ValueError(f"{where}: expected >= 0, got {n}")
    return n


def _as_positive_int(v: Any, *, where: str) -> int:
    n = _as_nonnegative_int(v, where=where)
    if n <= 0:
        raise ValueError(f"{where}: expected > 0, got {n}")
    return n


def _parse_order_candidates(raw: Any, *, where: str) -> tuple[int, ...]:
    vals: list[int] = []
    if isinstance(raw, str):
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        vals = [_as_positive_int(t, where=where) for t in toks]
    elif isinstance(raw, (list, tuple)):
        vals = [_as_positive_int(v, where=f"{where}[{i}]") for i, v in enumerate(raw)]
    else:
        raise ValueError(f"{where}: expected string or list of orders")

    uniq = tuple(sorted(set(int(v) for v in vals if int(v) in (1, 2))))
    if not uniq:
        raise ValueError(f"{where}: expected at least one order from {{1,2}}")
    return uniq


def _parse_dim_vector(raw: Any, *, n_base: int, where: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{where}: expected a list/tuple of exponents")
    if len(raw) != int(n_base):
        raise ValueError(f"{where}: expected {n_base} exponents, got {len(raw)}")
    return tuple(_as_finite_float(v, where=f"{where}[{i}]") for i, v in enumerate(raw))


def _dim_sub(d1: Sequence[float], d2: Sequence[float]) -> tuple[float, ...]:
    if len(d1) != len(d2):
        raise ValueError("dimension vector lengths do not match")
    return tuple(float(a) - float(b) for a, b in zip(d1, d2))


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    return str(value)


_REFINE_DIAG_COUNT_KEYS = (
    "score_calls",
    "refine_score_calls",
    "variants_generated",
    "gate_triggered_score_calls",
    "variants_after_gate",
    "gate_potential_checks",
    "gate_potential_evals",
    "grid_evals",
    "grid_only_returns",
    "grid_then_lbfgs_skips",
    "grid_then_lbfgs_escalations",
    "lbfgs_runs",
    "lbfgs_closures",
    "linear_solves",
    "linear_solves_multi",
    "hparam_optimizations",
    "refinement_attempts",
    "materialized_rescores",
    "accepted_refinements",
    "attempt_cache_hits",
    "attempt_cache_misses",
    "attempt_cache_stores",
    "attempt_cache_skipped_full",
    "attempt_cache_size",
    "mapping_equiv_root_slots_pruned",
    "brute_refine_score_calls",
    "brute_refinement_attempts",
    "brute_materialized_rescores",
    "brute_accepted_refinements",
    "mutation_refine_score_calls",
    "mutation_refinement_attempts",
    "mutation_materialized_rescores",
    "mutation_accepted_refinements",
    "slate_refine_score_calls",
    "slate_refinement_attempts",
    "slate_materialized_rescores",
    "slate_accepted_refinements",
    "controller_slate_refine_score_calls",
    "controller_slate_refinement_attempts",
    "controller_slate_materialized_rescores",
    "controller_slate_accepted_refinements",
    "external_refine_score_calls",
    "external_refinement_attempts",
    "external_materialized_rescores",
    "external_accepted_refinements",
)

_REFINE_DIAG_TIME_KEYS = (
    "base_score_s",
    "hparam_optimization_s",
)


def _merge_refine_diagnostics(total: dict[str, Any], row: Mapping[str, Any] | None) -> None:
    if not isinstance(row, Mapping):
        return
    for key, value in row.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            if isinstance(value, int):
                total[str(key)] = int(total.get(str(key), 0) or 0) + int(value)
            else:
                fv = float(value)
                if not math.isfinite(fv):
                    continue
                total[str(key)] = float(total.get(str(key), 0.0) or 0.0) + fv
        except Exception:
            continue


def _search_diagnostics_summary(per_order: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compact, report-level summary of full per-order search diagnostics."""

    out: dict[str, Any] = {
        "orders": [],
        "n_orders": 0,
        "additive_fss_orders": 0,
        "pre_mutation_context_rows": 0,
        "post_mutation_context_rows": 0,
        "pre_mutation_combo_rows": 0,
        "post_mutation_combo_rows": 0,
        "pre_mutation_early_return_orders": 0,
    }
    for order_row in per_order:
        if not isinstance(order_row, Mapping):
            continue
        search_diag = order_row.get("search_diagnostics", {})
        if not isinstance(search_diag, Mapping):
            continue
        order_summary: dict[str, Any] = {"order": order_row.get("order", None)}
        additive = search_diag.get("additive_fss", {})
        if isinstance(additive, Mapping):
            additive_enabled = bool(additive.get("enabled", False))
            if additive_enabled:
                out["additive_fss_orders"] = int(out["additive_fss_orders"]) + 1
            order_summary["additive_fss_enabled"] = additive_enabled
            for key in (
                "pre_mutation_context_rows",
                "post_mutation_context_rows",
                "pre_mutation_combo_rows",
                "post_mutation_combo_rows",
            ):
                try:
                    value = int(additive.get(key, 0) or 0)
                except Exception:
                    value = 0
                out[key] = int(out[key]) + value
                order_summary[key] = value
            early_return = bool(additive.get("pre_mutation_early_return", False))
            if early_return:
                out["pre_mutation_early_return_orders"] = int(out["pre_mutation_early_return_orders"]) + 1
            order_summary["pre_mutation_early_return"] = early_return
            for diag_key in (
                "pre_mutation_contextual_atom_diagnostics",
                "post_mutation_contextual_atom_diagnostics",
            ):
                diag_value = additive.get(diag_key, {})
                if isinstance(diag_value, Mapping):
                    order_summary[diag_key] = {
                        "enabled": bool(diag_value.get("enabled", False)),
                        "atoms_considered": int(
                            diag_value.get("atoms_considered", diag_value.get("input_atoms", 0)) or 0
                        ),
                        "residual_periodic_seed_rows": int(
                            diag_value.get(
                                "residual_periodic_seed_rows",
                                diag_value.get("periodic_seed_rows", 0),
                            )
                            or 0
                        ),
                        "contexts_tested": int(
                            diag_value.get("contexts_tested", diag_value.get("trial_count", 0)) or 0
                        ),
                        "promoted_rows": int(diag_value.get("promoted_rows", 0) or 0),
                    }
        out["orders"].append(order_summary)
    out["n_orders"] = int(len(out["orders"]))
    return out


def _refine_diag_number(diag: Mapping[str, Any], key: str) -> float:
    try:
        value = float(diag.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _refine_diagnostics_summary(diag: Mapping[str, Any] | None) -> dict[str, Any]:
    diag = diag if isinstance(diag, Mapping) else {}
    out: dict[str, Any] = {
        key: int(round(_refine_diag_number(diag, key)))
        for key in _REFINE_DIAG_COUNT_KEYS
        if key in diag
    }
    for key in _REFINE_DIAG_TIME_KEYS:
        if key in diag:
            out[key] = float(_refine_diag_number(diag, key))
    attempts = _refine_diag_number(diag, "refinement_attempts")
    hopt = _refine_diag_number(diag, "hparam_optimizations")
    accepted = _refine_diag_number(diag, "accepted_refinements")
    cache_hits = _refine_diag_number(diag, "attempt_cache_hits")
    cache_misses = _refine_diag_number(diag, "attempt_cache_misses")
    grid_evals = _refine_diag_number(diag, "grid_evals")
    lbfgs_closures = _refine_diag_number(diag, "lbfgs_closures")
    linear_solves = _refine_diag_number(diag, "linear_solves") + _refine_diag_number(
        diag,
        "linear_solves_multi",
    )
    denominator = max(attempts, hopt)
    out["accepted_per_attempt"] = None if denominator <= 0.0 else float(accepted / denominator)
    cache_lookups = cache_hits + cache_misses
    out["attempt_cache_hit_rate"] = None if cache_lookups <= 0.0 else float(cache_hits / cache_lookups)
    out["grid_evals_per_attempt"] = None if denominator <= 0.0 else float(grid_evals / denominator)
    out["lbfgs_closures_per_attempt"] = None if denominator <= 0.0 else float(lbfgs_closures / denominator)
    out["linear_solves_per_attempt"] = None if denominator <= 0.0 else float(linear_solves / denominator)
    return out


def _resolve_csv_ref(path_raw: str, *, base_dir: pathlib.Path | None) -> str:
    p = pathlib.Path(str(path_raw).strip())
    if base_dir is not None and not p.is_absolute():
        p = (base_dir / p).resolve()
    elif p.is_absolute():
        p = p.resolve()
    else:
        # Keep compatibility for callers that construct specs in-memory.
        p = pathlib.Path(str(path_raw).strip())
    return str(p)


def _parse_traj_list(
    raw: Any,
    *,
    where: str,
    base_dir: pathlib.Path | None,
) -> list[TrajectoryRef]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{where}: expected list")
    out: list[TrajectoryRef] = []
    for i, tr in enumerate(raw):
        row_where = f"{where}[{i}]"
        if not isinstance(tr, dict):
            raise ValueError(f"{row_where}: expected dict with keys 'id' and 'csv'")
        tid = str(tr.get("id", "")).strip()
        csv = str(tr.get("csv", "")).strip()
        if tid == "":
            raise ValueError(f"{row_where}.id: expected non-empty string")
        if csv == "":
            raise ValueError(f"{row_where}.csv: expected non-empty string")
        out.append(TrajectoryRef(id=tid, csv=_resolve_csv_ref(csv, base_dir=base_dir)))
    return out


def _merge_trajectory_refs(refs: Sequence[TrajectoryRef], *, where: str) -> list[TrajectoryRef]:
    out: list[TrajectoryRef] = []
    by_id: dict[str, str] = {}
    for i, tr in enumerate(refs):
        tid = str(tr.id)
        csv = str(tr.csv)
        prev = by_id.get(tid, None)
        if prev is not None and prev != csv:
            raise ValueError(
                f"{where}[{i}]: duplicate trajectory id {tid!r} has conflicting csv paths "
                f"({prev!r} vs {csv!r})"
            )
        if prev is None:
            by_id[tid] = csv
            out.append(TrajectoryRef(id=tid, csv=csv))
    return out


def equation_de_spec_from_dict(
    payload: dict[str, Any],
    *,
    source: str = "<dict>",
    base_dir: str | pathlib.Path | None = None,
) -> DELabSpec:
    """Validate and normalize a DE-lab payload into :class:`DELabSpec`."""

    if not isinstance(payload, dict):
        raise ValueError(f"{source}: spec root must be a dict")

    spec_id = str(payload.get("id", "")).strip()
    if spec_id == "":
        raise ValueError(f"{source}: missing non-empty 'id'")

    base_dir_p = None if base_dir is None else pathlib.Path(base_dir).resolve()

    fit_refs = _parse_traj_list(
        payload.get("fit_trajectories", None),
        where=f"{source}.fit_trajectories",
        base_dir=base_dir_p,
    )
    probe_refs = _parse_traj_list(
        payload.get("probe_trajectories", None),
        where=f"{source}.probe_trajectories",
        base_dir=base_dir_p,
    )
    fit_refs = _merge_trajectory_refs(fit_refs, where=f"{source}.fit_trajectories")
    probe_refs = _merge_trajectory_refs(probe_refs, where=f"{source}.probe_trajectories")
    has_explicit_split = ("fit_trajectories" in payload) or ("probe_trajectories" in payload)
    if has_explicit_split and len(fit_refs) == 0:
        raise ValueError(f"{source}: explicit fit/probe spec requires non-empty fit_trajectories")
    if has_explicit_split and len(probe_refs) > 0:
        fit_ids = {str(t.id) for t in fit_refs}
        probe_ids = {str(t.id) for t in probe_refs}
        overlap = sorted(fit_ids.intersection(probe_ids))
        if overlap:
            raise ValueError(f"{source}: fit/probe trajectories overlap: {overlap}")

    traj_refs = _parse_traj_list(
        payload.get("trajectories", None),
        where=f"{source}.trajectories",
        base_dir=base_dir_p,
    )

    csv_raw = payload.get("csv_paths", payload.get("filepaths", None))
    csv_paths_in: list[str] = []
    if csv_raw is not None:
        if not isinstance(csv_raw, (list, tuple)) or len(csv_raw) == 0:
            raise ValueError(f"{source}: 'csv_paths' (or alias 'filepaths') must be a non-empty list")
        csv_paths_in = [
            _resolve_csv_ref(str(p).strip(), base_dir=base_dir_p)
            for p in csv_raw
            if str(p).strip()
        ]
        if len(csv_paths_in) == 0:
            raise ValueError(f"{source}: no valid paths in 'csv_paths'")

    if not traj_refs and csv_paths_in:
        traj_refs = [
            TrajectoryRef(id=pathlib.Path(p).stem or f"traj{i:03d}", csv=str(p))
            for i, p in enumerate(csv_paths_in)
        ]

    if has_explicit_split:
        traj_refs = _merge_trajectory_refs(
            [*fit_refs, *probe_refs, *traj_refs],
            where=f"{source}.fit/probe/trajectories",
        )
    elif traj_refs:
        traj_refs = _merge_trajectory_refs(traj_refs, where=f"{source}.trajectories")

    if not traj_refs and not csv_paths_in:
        raise ValueError(
            f"{source}: provide either 'csv_paths'/'filepaths', non-empty 'trajectories', "
            "or explicit fit/probe trajectory lists"
        )

    csv_paths = tuple(tr.csv for tr in traj_refs) if traj_refs else tuple(csv_paths_in)

    order_candidates = _parse_order_candidates(payload.get("order_candidates", [1, 2]), where=f"{source}.order_candidates")
    x_axis = _as_nonnegative_int(payload.get("x_axis", 0), where=f"{source}.x_axis")

    include_x = bool(payload.get("include_x", True))
    include_u = bool(payload.get("include_u", True))
    include_du = bool(payload.get("include_du", True))

    x_col = str(payload.get("x_col", "x0")).strip() or "x0"
    u_col = str(payload.get("u_col", "y")).strip() or "y"
    out_idx = _as_nonnegative_int(payload.get("out_idx", 0), where=f"{source}.out_idx")
    y_transform = str(payload.get("y_transform", "identity")).strip() or "identity"
    _select_y_transform(y_transform)
    split_mode_in = str(payload.get("split_mode", "per_traj_point")).strip().lower() or "per_traj_point"
    if split_mode_in not in ("per_traj_point", "traj_holdout"):
        raise ValueError(f"{source}.split_mode: unsupported value {split_mode_in!r}")
    split_mode = ("traj_holdout" if len(probe_refs) > 0 else "per_traj_point") if has_explicit_split else split_mode_in
    traj_metric = str(payload.get("traj_metric", "mean")).strip().lower() or "mean"
    if traj_metric not in ("mean", "max"):
        raise ValueError(f"{source}.traj_metric: unsupported value {traj_metric!r}")

    deriv_raw = payload.get("deriv", {})
    if deriv_raw is None:
        deriv_raw = {}
    if not isinstance(deriv_raw, dict):
        raise ValueError(f"{source}: 'deriv' must be a dict when provided")
    method = str(deriv_raw.get("method", "spline")).strip().lower() or "spline"
    if method not in ("spline", "finite_diff", "precomputed"):
        raise ValueError(f"{source}.deriv.method: unsupported method {method!r}")
    derivative = DerivativeSpec(
        method=method,
        spline_s=float(_as_finite_float(deriv_raw.get("s", 0.0), where=f"{source}.deriv.s")),
        spline_k=_as_positive_int(deriv_raw.get("k", 3), where=f"{source}.deriv.k"),
        du_col=(None if deriv_raw.get("du_col", None) in (None, "") else str(deriv_raw.get("du_col"))),
        d2u_col=(None if deriv_raw.get("d2u_col", None) in (None, "") else str(deriv_raw.get("d2u_col"))),
    )

    constants_raw = payload.get("constants", [])
    if constants_raw is None:
        constants_raw = []
    if not isinstance(constants_raw, (list, tuple)):
        raise ValueError(f"{source}: 'constants' must be a list when provided")
    constants: list[ConstantSpec] = []
    for i, c in enumerate(constants_raw):
        where = f"{source}.constants[{i}]"
        if not isinstance(c, dict):
            raise ValueError(f"{where}: expected dict")
        name = str(c.get("name", "")).strip()
        if name == "":
            raise ValueError(f"{where}: missing non-empty 'name'")
        value = _as_finite_float(c.get("value", None), where=f"{where}.value")
        constants.append(ConstantSpec(name=name, value=float(value), dim=None))

    dims = None
    dims_raw = payload.get("dims", None)
    if dims_raw is not None:
        if not isinstance(dims_raw, dict):
            raise ValueError(f"{source}: 'dims' must be a dict when provided")
        basis_raw = dims_raw.get("basis", None)
        if not isinstance(basis_raw, (list, tuple)) or len(basis_raw) == 0:
            raise ValueError(f"{source}.dims.basis: expected non-empty list")
        basis = tuple(str(x) for x in basis_raw)
        n_base = len(basis)
        x_dim = _parse_dim_vector(dims_raw.get("x", None), n_base=n_base, where=f"{source}.dims.x")
        u_dim = _parse_dim_vector(dims_raw.get("u", None), n_base=n_base, where=f"{source}.dims.u")
        dims = DimensionSpec(basis=basis, x_dim=x_dim, u_dim=u_dim)

        # Reparse constant dims if provided in this mode.
        constants_dim: list[ConstantSpec] = []
        for i, c in enumerate(constants_raw):
            where = f"{source}.constants[{i}]"
            dim_raw = c.get("dim", None) if isinstance(c, dict) else None
            if dim_raw is None:
                c_dim = tuple(0.0 for _ in range(n_base))
            else:
                c_dim = _parse_dim_vector(dim_raw, n_base=n_base, where=f"{where}.dim")
            constants_dim.append(ConstantSpec(name=constants[i].name, value=constants[i].value, dim=c_dim))
        constants = constants_dim

    validate_integrate_topk = _as_nonnegative_int(
        payload.get("validate_integrate_topk", 0), where=f"{source}.validate_integrate_topk"
    )
    extra_raw = payload.get("extra", None)
    extra = None
    if extra_raw is not None:
        if not isinstance(extra_raw, dict):
            raise ValueError(f"{source}: 'extra' must be a dict when provided")
        extra = {str(k): _to_jsonable(v) for k, v in extra_raw.items()}

    if not include_x and not include_u and not include_du and len(constants) == 0:
        raise ValueError(f"{source}: feature set is empty")

    for o in order_candidates:
        n_feat = 0
        if include_x:
            n_feat += 1
        if include_u:
            n_feat += 1
        if int(o) == 2 and include_du:
            n_feat += 1
        n_feat += len(constants)
        if n_feat <= 0:
            raise ValueError(
                f"{source}: order_candidates includes order={int(o)} but no usable features remain"
            )

    return DELabSpec(
        id=spec_id,
        csv_paths=csv_paths,
        order_candidates=order_candidates,
        x_axis=x_axis,
        include_x=include_x,
        include_u=include_u,
        include_du=include_du,
        x_col=x_col,
        u_col=u_col,
        out_idx=out_idx,
        y_transform=y_transform,
        split_mode=split_mode,
        traj_metric=traj_metric,
        trajectories=tuple(traj_refs),
        fit_trajectories=tuple(fit_refs),
        probe_trajectories=tuple(probe_refs),
        constants=tuple(constants),
        derivative=derivative,
        dims=dims,
        extra=extra,
        validate_integrate_topk=validate_integrate_topk,
    )


def load_de_equation_spec(path: str | pathlib.Path) -> DELabSpec:
    """Load a DE lab spec from JSON or YAML."""

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()

    payload: dict[str, Any]
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YAML spec requested but PyYAML is not installed. "
                "Use JSON or install PyYAML."
            ) from exc
        payload = yaml.safe_load(text)
    else:
        try:
            payload = json.loads(text)
        except Exception:
            try:
                import yaml  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Unsupported spec extension (expected .json/.yaml/.yml) and payload is not valid JSON."
                ) from exc
            payload = yaml.safe_load(text)

    if not isinstance(payload, dict):
        raise ValueError(f"{p}: spec root must be a dict")

    if str(payload.get("id", "")).strip() == "":
        payload = dict(payload)
        payload["id"] = p.stem

    return equation_de_spec_from_dict(payload, source=str(p), base_dir=p.parent)


def _read_csv_columns(path: str | pathlib.Path, *, cols: Sequence[str]) -> dict[str, np.ndarray]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    arr = np.genfromtxt(str(p), delimiter=",", names=True, dtype=np.float64)
    if getattr(arr, "dtype", None) is None or arr.dtype.names is None:
        raise ValueError(f"{p}: expected CSV with header row")

    out: dict[str, np.ndarray] = {}
    names = set(arr.dtype.names)
    for c in cols:
        if c not in names:
            raise ValueError(f"{p}: missing required column {c!r}; available={sorted(names)}")
        out[c] = np.asarray(arr[c], dtype=np.float64).reshape(-1)
    return out


def _sort_unique(x: np.ndarray, *ys: np.ndarray) -> tuple[np.ndarray, ...]:
    if any(len(y) != len(x) for y in ys):
        raise ValueError("_sort_unique: length mismatch")

    cols = [x, *ys]
    finite = np.isfinite(x)
    for y in ys:
        finite &= np.isfinite(y)
    if not finite.any():
        raise ValueError("no finite rows")

    cols_f = [c[finite] for c in cols]
    idx = np.argsort(cols_f[0])
    cols_s = [c[idx] for c in cols_f]

    x_s = cols_s[0]
    uniq, inv, counts = np.unique(x_s, return_inverse=True, return_counts=True)
    if uniq.size == x_s.size:
        return tuple(cols_s)

    merged = [uniq]
    for c in cols_s[1:]:
        sums = np.bincount(inv, weights=c)
        merged.append(sums / counts)
    return tuple(merged)


def _select_y_transform(name: str) -> YTransform:
    chosen = str(name).strip() or "identity"
    transforms = build_default_y_transforms([chosen])
    if len(transforms) != 1:
        raise ValueError(f"expected one y-transform for {chosen!r}, got {len(transforms)}")
    return transforms[0]


def _inverse_chain_rule_1d(
    yt: YTransform,
    *,
    t: torch.Tensor,
    dt_dx: torch.Tensor,
    d2t_dx2: torch.Tensor,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover `(u, u_x, u_xx)` from transformed outputs `t = phi(u)`."""

    y = yt.torch_inv(t) if getattr(yt, "torch_inv", None) is not None else t
    a = yt.d1(y) if getattr(yt, "d1", None) is not None else torch.ones_like(y)
    b = yt.d2(y) if getattr(yt, "d2", None) is not None else torch.zeros_like(y)

    a_abs = torch.abs(a)
    a_sign = torch.sign(a)
    a_sign = torch.where(a_sign == 0, torch.ones_like(a_sign), a_sign)
    a_safe = torch.where(a_abs < float(eps), a_sign * float(eps), a)

    y_x = dt_dx / a_safe
    y_xx = d2t_dx2 / a_safe - b * (dt_dx * dt_dx) / (a_safe * a_safe * a_safe)
    return y, y_x, y_xx


def _flatten_x_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch

    if x is None:
        raise ValueError("loader returned x=None")

    x_t = x if torch.is_tensor(x) else torch.as_tensor(x)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(1)
    elif x_t.ndim > 2:
        x_t = x_t.reshape(x_t.shape[0], -1)
    return x_t


def _gather_x(
    loader: Any,
    *,
    max_batches: int,
    max_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if max_batches <= 0:
        raise ValueError(f"max_batches must be > 0, got {max_batches}")
    if max_points <= 0:
        raise ValueError(f"max_points must be > 0, got {max_points}")

    parts: list[torch.Tensor] = []
    total = 0
    for i, batch in enumerate(loader):
        if i >= int(max_batches):
            break
        xb = _flatten_x_batch(batch).to(device=device, dtype=dtype)
        if int(xb.shape[0]) <= 0:
            continue
        parts.append(xb)
        total += int(xb.shape[0])
        if total >= int(max_points):
            break

    if not parts:
        raise ValueError("failed to gather any x points from loader")

    x_all = torch.cat(parts, dim=0)
    if int(x_all.shape[0]) > int(max_points):
        x_all = x_all[: int(max_points)]
    return x_all


def _subsample_rows(x: torch.Tensor, n_take: int, *, seed: int) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected rank-2 tensor, got shape={tuple(x.shape)}")
    if n_take <= 0:
        raise ValueError(f"n_take must be > 0, got {n_take}")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = _sample_indices(int(x.shape[0]), int(n_take), generator=g)
    return x[idx]


def _feature_tensors_from_trajectories(
    trajectories: Sequence[_Trajectory],
    *,
    n_fit: int,
    n_probe: int,
    seed: int,
    dtype: torch.dtype,
) -> DEFeatureTensors:
    if not trajectories:
        raise ValueError("no trajectories provided")

    x_all = torch.as_tensor(
        np.concatenate([tr.x.reshape(-1, 1) for tr in trajectories], axis=0),
        dtype=dtype,
    )
    u_all = torch.as_tensor(
        np.concatenate([tr.u.reshape(-1, 1) for tr in trajectories], axis=0),
        dtype=dtype,
    )
    du_all = torch.as_tensor(
        np.concatenate([tr.du.reshape(-1, 1) for tr in trajectories], axis=0),
        dtype=dtype,
    )
    d2u_all = torch.as_tensor(
        np.concatenate([tr.d2u.reshape(-1, 1) for tr in trajectories], axis=0),
        dtype=dtype,
    )

    fit_idx = _sample_indices(
        int(x_all.shape[0]),
        int(n_fit),
        generator=torch.Generator(device="cpu").manual_seed(int(seed) + 11_011),
    )
    probe_idx = _sample_indices(
        int(x_all.shape[0]),
        int(n_probe),
        generator=torch.Generator(device="cpu").manual_seed(int(seed) + 22_033),
    )

    return DEFeatureTensors(
        x_fit=x_all[fit_idx],
        u_fit=u_all[fit_idx],
        du_fit=du_all[fit_idx],
        d2u_fit=d2u_all[fit_idx],
        x_probe=x_all[probe_idx],
        u_probe=u_all[probe_idx],
        du_probe=du_all[probe_idx],
        d2u_probe=d2u_all[probe_idx],
    )


class SplineDerivProvider:
    """Phase-1 provider using per-trajectory spline/finite-diff derivatives."""

    def __init__(
        self,
        derivative_provider_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> None:
        self.derivative_provider_fn = derivative_provider if derivative_provider_fn is None else derivative_provider_fn

    def load_trajectories(self, spec: DELabSpec) -> list[_Trajectory]:
        return _load_trajectories(spec, derivative_provider_fn=self.derivative_provider_fn)

    def build_features_from_spec(
        self,
        spec: DELabSpec,
        *,
        n_fit: int,
        n_probe: int,
        seed: int,
        dtype: torch.dtype,
    ) -> tuple[DEFeatureTensors, list[_Trajectory]]:
        trajectories = self.load_trajectories(spec)
        features = _feature_tensors_from_trajectories(
            trajectories,
            n_fit=int(n_fit),
            n_probe=int(n_probe),
            seed=int(seed),
            dtype=dtype,
        )
        return features, trajectories


class SurrogateDerivProvider:
    """Phase-2 provider backed by surrogate caches (`u`, `grad`, `hess`)."""

    def __init__(
        self,
        surrogate: Any,
        *,
        y_transform: str = "identity",
        out_idx: int = 0,
        cache: UFeatureCache | None = None,
        eval_batch: int = 8192,
    ) -> None:
        self.surrogate = surrogate
        self.out_idx = int(out_idx)
        if self.out_idx < 0:
            raise ValueError("out_idx must be >= 0")
        self.eval_batch = max(1, int(eval_batch))
        self.yt = _select_y_transform(str(y_transform))
        self.cache = cache if cache is not None else UFeatureCache(surrogate)

    def _eval_features(self, x: torch.Tensor, *, x_axis: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"expected rank-2 x, got shape={tuple(x.shape)}")
        if int(x_axis) < 0 or int(x_axis) >= int(x.shape[1]):
            raise ValueError(f"x_axis={x_axis} is out of bounds for x shape={tuple(x.shape)}")

        u_parts: list[torch.Tensor] = []
        du_parts: list[torch.Tensor] = []
        d2u_parts: list[torch.Tensor] = []

        for i0 in range(0, int(x.shape[0]), int(self.eval_batch)):
            xb = x[i0 : i0 + int(self.eval_batch)]
            self.cache.ensure(xb, need_grad=True, need_hess=True)

            if self.cache.u is None or self.cache.g is None or self.cache.H is None:
                raise RuntimeError("UFeatureCache did not populate u/g/H")

            if int(self.out_idx) >= int(self.cache.u.shape[1]):
                raise ValueError(
                    f"out_idx={self.out_idx} out of bounds for Ny={int(self.cache.u.shape[1])}"
                )

            t = self.cache.u[:, self.out_idx : self.out_idx + 1]
            t_x = self.cache.g[:, self.out_idx : self.out_idx + 1, int(x_axis)]
            t_xx = self.cache.H[:, self.out_idx : self.out_idx + 1, int(x_axis), int(x_axis)]

            u, du, d2u = _inverse_chain_rule_1d(
                self.yt,
                t=t,
                dt_dx=t_x,
                d2t_dx2=t_xx,
            )
            u_parts.append(u)
            du_parts.append(du)
            d2u_parts.append(d2u)

        return torch.cat(u_parts, dim=0), torch.cat(du_parts, dim=0), torch.cat(d2u_parts, dim=0)

    def build_features_from_loaders(
        self,
        train_loader: Any,
        val_loader: Any | None,
        *,
        spec: DELabSpec,
        seed: int,
        dtype: torch.dtype,
        device: torch.device,
        n_fit: int,
        n_probe: int,
        max_batches: int = 64,
        max_points_factor: int = 4,
    ) -> DEFeatureTensors:
        max_points_fit = max(int(n_fit), int(n_fit) * max(1, int(max_points_factor)))
        x_fit_pool = _gather_x(
            train_loader,
            max_batches=int(max_batches),
            max_points=max_points_fit,
            device=device,
            dtype=dtype,
        )
        x_fit = _subsample_rows(x_fit_pool, int(n_fit), seed=int(seed))

        if val_loader is None:
            x_probe_pool = x_fit_pool
        else:
            max_points_probe = max(int(n_probe), int(n_probe) * max(1, int(max_points_factor)))
            x_probe_pool = _gather_x(
                val_loader,
                max_batches=int(max_batches),
                max_points=max_points_probe,
                device=device,
                dtype=dtype,
            )
        x_probe = _subsample_rows(x_probe_pool, int(n_probe), seed=int(seed) + 1_000_003)

        u_fit, du_fit, d2u_fit = self._eval_features(x_fit, x_axis=int(spec.x_axis))
        u_probe, du_probe, d2u_probe = self._eval_features(x_probe, x_axis=int(spec.x_axis))

        return DEFeatureTensors(
            x_fit=x_fit,
            u_fit=u_fit,
            du_fit=du_fit,
            d2u_fit=d2u_fit,
            x_probe=x_probe,
            u_probe=u_probe,
            du_probe=du_probe,
            d2u_probe=d2u_probe,
        )


def derivative_provider(
    x: np.ndarray,
    u: np.ndarray,
    *,
    method: str = "spline",
    spline_s: float = 0.0,
    spline_k: int = 3,
    du_pre: np.ndarray | None = None,
    d2u_pre: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted/unique `(x, u, du, d2u)` using the requested derivative method."""

    method_l = str(method).lower().strip()

    if method_l == "precomputed":
        if du_pre is None:
            raise ValueError("precomputed derivative mode requires du_pre")
        if d2u_pre is None:
            # Fallback if only du is precomputed.
            x_s, u_s, du_s = _sort_unique(np.asarray(x), np.asarray(u), np.asarray(du_pre))
            if x_s.size < 5:
                raise ValueError("need at least 5 points after sorting/uniquing")
            d2u_s = np.gradient(du_s, x_s, edge_order=2)
            return x_s, u_s, du_s, np.asarray(d2u_s, dtype=np.float64)

        x_s, u_s, du_s, d2u_s = _sort_unique(
            np.asarray(x), np.asarray(u), np.asarray(du_pre), np.asarray(d2u_pre)
        )
        if x_s.size < 5:
            raise ValueError("need at least 5 points after sorting/uniquing")
        return x_s, u_s, du_s, d2u_s

    x_s, u_s = _sort_unique(np.asarray(x), np.asarray(u))
    if x_s.size < 5:
        raise ValueError("need at least 5 points after sorting/uniquing")

    if method_l == "finite_diff":
        du_s = np.gradient(u_s, x_s, edge_order=2)
        d2u_s = np.gradient(du_s, x_s, edge_order=2)
        return x_s, u_s, np.asarray(du_s, dtype=np.float64), np.asarray(d2u_s, dtype=np.float64)

    if method_l == "spline":
        try:
            from scipy.interpolate import UnivariateSpline
        except Exception as exc:
            raise RuntimeError("spline derivative mode requires scipy") from exc

        k = max(1, min(int(spline_k), int(x_s.size) - 1))
        spl = UnivariateSpline(x_s, u_s, s=float(spline_s), k=int(k))
        du_s = spl.derivative(1)(x_s)
        d2u_s = spl.derivative(2)(x_s)
        return x_s, u_s, np.asarray(du_s, dtype=np.float64), np.asarray(d2u_s, dtype=np.float64)

    raise ValueError(f"unsupported derivative method {method!r}")


def _trajectory_refs(spec: DELabSpec) -> tuple[TrajectoryRef, ...]:
    if len(spec.fit_trajectories) > 0 or len(spec.probe_trajectories) > 0:
        merged = _merge_trajectory_refs(
            [*spec.fit_trajectories, *spec.probe_trajectories],
            where="spec.fit_trajectories/spec.probe_trajectories",
        )
        if merged:
            return tuple(merged)
    if len(spec.trajectories) > 0:
        return tuple(spec.trajectories)
    return tuple(
        TrajectoryRef(id=pathlib.Path(p).stem or f"traj{i:03d}", csv=str(p))
        for i, p in enumerate(spec.csv_paths)
    )


def _load_trajectories(
    spec: DELabSpec,
    *,
    derivative_provider_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = derivative_provider,
) -> list[_Trajectory]:
    out: list[_Trajectory] = []

    for tr_ref in _trajectory_refs(spec):
        p = tr_ref.csv
        cols = [spec.x_col, spec.u_col]
        if spec.derivative.method == "precomputed" and spec.derivative.du_col is not None:
            cols.append(spec.derivative.du_col)
        if spec.derivative.method == "precomputed" and spec.derivative.d2u_col is not None:
            cols.append(spec.derivative.d2u_col)

        data = _read_csv_columns(p, cols=cols)
        x = data[spec.x_col]
        u = data[spec.u_col]

        du_pre = None if spec.derivative.du_col is None else data.get(spec.derivative.du_col, None)
        d2u_pre = None if spec.derivative.d2u_col is None else data.get(spec.derivative.d2u_col, None)

        x_s, u_s, du_s, d2u_s = derivative_provider_fn(
            x,
            u,
            method=spec.derivative.method,
            spline_s=spec.derivative.spline_s,
            spline_k=spec.derivative.spline_k,
            du_pre=du_pre,
            d2u_pre=d2u_pre,
        )

        out.append(
            _Trajectory(
                traj_id=str(tr_ref.id),
                path=str(p),
                x=x_s,
                u=u_s,
                du=du_s,
                d2u=d2u_s,
            )
        )

    if not out:
        raise ValueError("no trajectories loaded")
    return out


def _feature_names(spec: DELabSpec, order: int) -> list[str]:
    names: list[str] = []
    if spec.include_x:
        names.append(spec.x_col)
    if spec.include_u:
        names.append(spec.u_col)
    if int(order) == 2 and spec.include_du:
        names.append(spec.derivative.du_col or "du")
    for c in spec.constants:
        names.append(c.name)
    return names


def _target_name(spec: DELabSpec, order: int) -> str:
    return "du" if int(order) == 1 else "d2u"


def _build_table_for_order(
    spec: DELabSpec,
    traj: _Trajectory,
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feats: list[np.ndarray] = []
    if spec.include_x:
        feats.append(traj.x)
    if spec.include_u:
        feats.append(traj.u)
    # Fairness guard: never feed du into order-1 target discovery.
    if int(order) == 2 and spec.include_du:
        feats.append(traj.du)

    for c in spec.constants:
        feats.append(np.full_like(traj.x, float(c.value), dtype=np.float64))

    if not feats:
        raise ValueError(f"order={order}: empty feature set")

    y = traj.du if int(order) == 1 else traj.d2u
    Z = np.column_stack(feats)
    m = np.isfinite(y)
    m &= np.isfinite(Z).all(axis=1)
    Zf = Z[m]
    yf = y[m]

    if Zf.shape[0] < 8:
        raise ValueError(f"order={order}: too few finite rows ({Zf.shape[0]})")

    return Zf, yf, _feature_names(spec, order)


@dataclass(frozen=True)
class _OrderTables:
    x_fit: torch.Tensor
    y_fit: torch.Tensor
    x_probe: torch.Tensor
    y_probe: torch.Tensor
    feature_names: list[str]
    fit_traj_ids: list[str]
    probe_traj_ids: list[str]
    fit_meta: list[tuple[str, torch.Tensor, torch.Tensor]]
    probe_meta: list[tuple[str, torch.Tensor, torch.Tensor]]


def _subsample_table(
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    n_take: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if z.ndim != 2:
        raise ValueError(f"expected z rank-2, got {tuple(z.shape)}")
    if y.ndim != 2 or int(y.shape[1]) != 1:
        raise ValueError(f"expected y shape (N,1), got {tuple(y.shape)}")
    n_total = int(z.shape[0])
    n_take = max(1, int(n_take))
    if n_take >= n_total:
        return z, y

    # Stratified subsampling is significantly more stable for DE discovery,
    # especially for stiff / singular-coordinate problems where the informative
    # region may occupy a tiny fraction of the x-range.
    #
    # We use a 50/50 mixture of:
    #   - stratified samples along the highest-variance feature axis
    #   - uniform random samples
    #
    # The mixture keeps some stochasticity while ensuring coverage.
    g = torch.Generator(device="cpu").manual_seed(int(seed % (2**63 - 1)))

    n_strat = max(1, n_take // 2)
    n_rand = int(n_take - n_strat)

    with torch.no_grad():
        col_var = ((z - z.mean(dim=0, keepdim=True)) ** 2).mean(dim=0)
        axis = int(torch.argmax(col_var).item())
        order = torch.argsort(z[:, axis])
        edges = torch.linspace(0, n_total, steps=n_strat + 1, dtype=torch.float64, device="cpu")
        lo = edges[:-1].floor()
        hi = edges[1:].floor()
        width = torch.clamp(hi - lo, min=1.0)
        u = torch.rand(int(n_strat), generator=g, dtype=torch.float64, device="cpu")
        pos = (lo + (u * width).floor()).to(torch.long)
        pos = torch.clamp(pos, min=0, max=n_total - 1)
        if order.device.type != "cpu":
            pos = pos.to(device=order.device)
        idx_strat = order.index_select(0, pos)

    if n_rand > 0:
        idx_rand = _sample_indices(n_total, n_rand, generator=g)
        if idx_strat.device.type != "cpu":
            idx_rand = idx_rand.to(device=idx_strat.device)
        idx = torch.cat([idx_strat, idx_rand], dim=0)
        # De-duplicate; refill if needed.
        idx = torch.unique(idx)
        if int(idx.numel()) < n_take:
            need = int(n_take - idx.numel())
            extra = _sample_indices(n_total, need, generator=g)
            if idx.device.type != "cpu":
                extra = extra.to(device=idx.device)
            idx = torch.cat([idx, extra], dim=0)
        idx = idx[:n_take]
    else:
        idx = idx_strat

    return z.index_select(0, idx), y.index_select(0, idx)


def _split_table_disjoint(
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    n_fit_take: int,
    n_probe_take: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one trajectory table into disjoint fit/probe subsets."""
    if z.ndim != 2:
        raise ValueError(f"expected z rank-2, got {tuple(z.shape)}")
    if y.ndim != 2 or int(y.shape[1]) != 1:
        raise ValueError(f"expected y shape (N,1), got {tuple(y.shape)}")

    n_rows = int(z.shape[0])
    if n_rows < 2:
        raise ValueError(f"need at least 2 rows for disjoint fit/probe split, got {n_rows}")

    n_fit_req = max(1, int(n_fit_take))
    n_probe_req = max(1, int(n_probe_take))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(n_rows, generator=g)

    max_fit = max(1, n_rows - 1)
    fit_take = min(n_fit_req, max_fit)
    probe_room = n_rows - fit_take
    probe_take = min(n_probe_req, probe_room)
    if probe_take <= 0:
        fit_take = max(1, min(n_fit_req, n_rows - 1))
        probe_room = n_rows - fit_take
        probe_take = min(n_probe_req, probe_room)
    if probe_take <= 0:
        raise ValueError(
            f"failed to create disjoint fit/probe split with n_rows={n_rows}, "
            f"n_fit_take={n_fit_req}, n_probe_take={n_probe_req}"
        )

    fit_idx = perm[:fit_take]
    probe_idx = perm[fit_take : fit_take + probe_take]
    return z[fit_idx], y[fit_idx], z[probe_idx], y[probe_idx]


def _aggregate_score(values: Sequence[float], *, metric: str) -> float:
    vals_raw = [float(v) for v in values]
    if not vals_raw:
        return float("inf")
    if any(not math.isfinite(v) for v in vals_raw):
        return float("inf")
    metric_l = str(metric).lower().strip()
    if metric_l == "max":
        return float(max(vals_raw))
    if metric_l != "mean":
        raise ValueError(f"unsupported traj_metric {metric!r}")
    return float(sum(vals_raw) / len(vals_raw))


def _tuple_node(obj: Any) -> Any:
    if isinstance(obj, list):
        return tuple(_tuple_node(v) for v in obj)
    if isinstance(obj, tuple):
        return tuple(_tuple_node(v) for v in obj)
    return obj


def _score_head_complexity(mapping: Mapping[str, Any]) -> int:
    head = mapping.get("_lin_head", None)
    if not isinstance(head, dict):
        return 0
    terms = head.get("terms", [])
    coeffs = head.get("coeffs", [])
    n_terms = len(terms) if isinstance(terms, (list, tuple)) else 0
    n_coeffs = len(coeffs) if isinstance(coeffs, (list, tuple)) else (n_terms + 1 if n_terms else 1)
    cost = int(max(1, n_coeffs))
    if isinstance(terms, (list, tuple)):
        for term in terms:
            try:
                cost += int(node_size(_tuple_node(term)))
            except Exception:
                cost += 1
    return int(cost)


def _has_hidden_score_head(mapping: Any) -> bool:
    if not isinstance(mapping, dict):
        return False
    if not isinstance(mapping.get("_lin_head", None), dict):
        return False
    return mapping.get("_basis_transition", None) is None


def _mapping_complexity(mapping: Any) -> int:
    """Return the effective number of mapping parameters used for scoring."""
    if not isinstance(mapping, dict):
        return 0
    kind = str(mapping.get("kind", "")).strip().lower()
    base = 0
    if kind == "affine":
        base = 0
    elif kind == "poly":
        coeffs = mapping.get("coeffs", [])
        base = len(coeffs) if isinstance(coeffs, (list, tuple)) else 0
    elif kind == "power":
        base = 2
    elif kind == "exp":
        base = 3
    elif kind == "sine":
        base = 4
    elif kind == "pade":
        numer = mapping.get("numer", [])
        denom = mapping.get("denom", [])
        n_n = len(numer) if isinstance(numer, (list, tuple)) else 0
        n_d = len(denom) if isinstance(denom, (list, tuple)) else 0
        base = max(0, n_n + n_d - 1)
    else:
        base = 1 if kind else 0
    return int(base + _score_head_complexity(mapping))


def _periodic_seed_atom_rows(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    hp,
    fit_meta: Sequence[tuple[str, torch.Tensor, torch.Tensor]] | None = None,
) -> list[dict[str, Any]]:
    """Periodogram-hinted trig atoms pinned into the additive-combo pool.

    Frequency atoms can be evicted from the archive by complexity tie-breaks
    before the composer runs (a decorrelated ``sin(x0)`` is smaller than
    ``cos(w*x0)`` at the same useless score), so the combo pool receives the
    hinted atoms directly; the composer's joint linear solve then recovers
    forcing terms (de301-class) in closed form.

    With multiple trajectories, the pooled y(x) is multivalued (one branch
    per trajectory) and its resampling is jump-dominated, so the scan runs
    per trajectory via ``fit_meta`` and the hints are merged.
    """
    if not bool(getattr(hp, "periodic_seed_enable", True)):
        return []
    from .engine.search import _periodogram_frequency_hints

    max_hints = int(getattr(hp, "periodic_seed_max_hints", 2))
    min_prominence = float(getattr(hp, "periodic_seed_min_prominence", 8.0))
    tables: list[tuple[str, torch.Tensor, torch.Tensor]] = [("pooled", x_fit, y_fit)]
    if fit_meta:
        tables.extend((str(tid), z, y) for tid, z, y in fit_meta)

    hints: list[tuple[int, float, str]] = []
    for source, z_tbl, y_tbl in tables:
        try:
            table_hints = _periodogram_frequency_hints(
                z_tbl,
                y_tbl,
                max_hints=max_hints,
                min_prominence=min_prominence,
            )
        except Exception:
            continue
        hints.extend((int(var_idx), float(omega), str(source)) for var_idx, omega in table_hints)

    merged: list[tuple[int, float, str]] = []
    max_merged = max(0, max(2 * max_hints, 4 * max_hints))
    for var_idx, omega, source in hints:
        if all(
            var_idx != prev_var or abs(omega - prev_w) > 0.05 * max(omega, prev_w)
            for prev_var, prev_w, _prev_source in merged
        ):
            merged.append((int(var_idx), float(omega), str(source)))
    merged = merged[:max_merged]

    rows: list[dict[str, Any]] = []
    for var_idx, omega, source in merged:
        for fn_name in ("sin", "cos"):
            expr_obj = (fn_name, ("mul", ("const", float(omega)), ("var", int(var_idx))))
            rows.append(
                {
                    "expr": node_str(expr_obj),
                    "_expr_obj": expr_obj,
                    # Pinned score: keeps the atoms at the head of the pool
                    # ranking without entering the result rows themselves.
                    "score": -1.0,
                    "mapping_kind": "periodic_seed",
                    "periodic_hint_source": str(source),
                }
            )
    return rows


def _substitute_jet_atoms_for_features(node: Any, idxs: Mapping[str, int | None], x_axis: int) -> Any:
    """Rebuild a NestyNet_SR jet AST with jet atoms remapped to feature vars.

    ``Var(x_axis)`` / ``U()`` / ``DU(x_axis)`` become ``Var(k)`` for the
    feature-column index ``k`` of the DE regression table, so the result can
    be compiled by ``bridge.nestynet_to_factorized_search``.
    """

    from nestynet_sr.sr_core.bridges import (
        Add,
        AddNode,
        AtomNode,
        ConstNode,
        Cos,
        CosNode,
        Exp,
        ExpNode,
        Log,
        LogNode,
        Mul,
        MulNode,
        Pow,
        PowNode,
        Sin,
        SinNode,
        Var,
    )

    def _rec(cur: Any) -> Any:
        if isinstance(cur, AtomNode):
            kind = str(getattr(cur, "kind", "")).lower()
            if kind in ("var", "x", "input"):
                axis = int(cur.var_idxs[0])
                if axis != int(x_axis):
                    raise ValueError(f"seed uses x{axis}, expected x{x_axis}")
                if idxs.get("x") is None:
                    raise ValueError("x is not a feature column in this spec")
                return Var(int(idxs["x"]))
            if kind in ("u", "field", "state"):
                if idxs.get("u") is None:
                    raise ValueError("u is not a feature column in this spec")
                return Var(int(idxs["u"]))
            if kind in ("du", "d1u", "grad_u"):
                if idxs.get("du") is None:
                    raise ValueError("du is not a feature column at this order")
                return Var(int(idxs["du"]))
            raise ValueError(f"unsupported jet atom kind {kind!r} in symmetry seed")
        if isinstance(cur, ConstNode):
            return cur
        if isinstance(cur, AddNode):
            return Add(_rec(cur.left), _rec(cur.right))
        if isinstance(cur, MulNode):
            return Mul(_rec(cur.left), _rec(cur.right))
        if isinstance(cur, PowNode):
            return Pow(_rec(cur.base), cur.exponent)
        if isinstance(cur, ExpNode):
            return Exp(_rec(cur.arg))
        if isinstance(cur, LogNode):
            return Log(_rec(cur.arg))
        if isinstance(cur, SinNode):
            return Sin(_rec(cur.arg))
        if isinstance(cur, CosNode):
            return Cos(_rec(cur.arg))
        raise ValueError(f"unsupported node type {type(cur).__name__} in symmetry seed")

    return _rec(node)


def _gs_symmetry_seed_rows(spec: DELabSpec, order: int, *, max_seeds: int = 16) -> list[dict[str, Any]]:
    """Pinned combo-pool atoms compiled from symmetry-reduction seed ASTs.

    Whole-law proposals and pulled-back rows from ``sr_gs.de_reduction`` ride
    in ``spec.extra['gs_symmetry_seed_asts']`` as ``{"node": Node, "label":
    str}`` entries.  Like the periodogram-hinted trig atoms, they are pinned
    at the head of the additive-combo pool (score -1) so the composer's joint
    linear solve can assemble and coefficient-fit the law in closed form; the
    engine's mutation/refinement phases then compete against (and can refine)
    the result.  Empty payload leaves the search unchanged.
    """

    payload = list(((spec.extra or {}).get("gs_symmetry_seed_asts", ())) or ())
    if not payload:
        return []
    from .bridge import nestynet_to_factorized_search
    from .expr_ast import is_valid_node, node_str

    idxs = _context_feature_indices(spec, int(order))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload:
        if len(rows) >= int(max_seeds):
            break
        node = item.get("node") if isinstance(item, Mapping) else item
        label = str(item.get("label", "")) if isinstance(item, Mapping) else ""
        if node is None:
            continue
        try:
            remapped = _substitute_jet_atoms_for_features(node, idxs, int(spec.x_axis))
            expr_obj = nestynet_to_factorized_search(remapped)
        except Exception:
            continue
        if not (isinstance(expr_obj, tuple) and is_valid_node(expr_obj)):
            continue
        key = node_str(expr_obj)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "expr": key,
                "_expr_obj": expr_obj,
                # pinned strictly ahead of the periodogram-hinted trig atoms
                # (which use score -1.0): a decaying/aperiodic signal can spawn
                # enough spurious periodic hints to fill the carrier pool and
                # evict a symmetry seed, so these get first claim on the pool.
                "score": -2.0,
                "mapping_kind": "gs_symmetry_seed",
                "gs_seed_label": label,
            }
        )
    return rows


def _context_feature_indices(spec: DELabSpec, order: int) -> dict[str, int | None]:
    idx = 0
    x_idx = None
    u_idx = None
    du_idx = None
    if spec.include_x:
        x_idx = idx
        idx += 1
    if spec.include_u:
        u_idx = idx
        idx += 1
    if int(order) == 2 and spec.include_du:
        du_idx = idx
    return {"x": x_idx, "u": u_idx, "du": du_idx}


def _context_base_supports(spec: DELabSpec, order: int) -> list[tuple[Any, ...]]:
    idxs = _context_feature_indices(spec, order)
    const = ("const", 1.0)
    supports: list[tuple[Any, ...]] = [tuple(), (const,)]
    u_idx = idxs.get("u")
    du_idx = idxs.get("du")
    if u_idx is not None:
        u = ("var", int(u_idx))
        supports.extend([(u,), (const, u)])
        if du_idx is not None:
            du = ("var", int(du_idx))
            supports.extend([(u, du), (const, u, du)])
    elif du_idx is not None:
        du = ("var", int(du_idx))
        supports.extend([(du,), (const, du)])

    out: list[tuple[Any, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for support in supports:
        key = tuple(node_str(t) for t in support)
        if key in seen:
            continue
        seen.add(key)
        out.append(tuple(support))
    return out


def _periodic_atom_descriptor(expr_obj: Any) -> tuple[str, int, float] | None:
    if not isinstance(expr_obj, (tuple, list)) or len(expr_obj) < 2:
        return None
    fn = str(expr_obj[0])
    if fn not in {"sin", "cos"}:
        return None
    arg = expr_obj[1]
    if isinstance(arg, (tuple, list)) and len(arg) >= 2 and str(arg[0]) == "var":
        try:
            return fn, int(arg[1]), 1.0
        except Exception:
            return None
    if not (isinstance(arg, (tuple, list)) and len(arg) == 3 and str(arg[0]) == "mul"):
        return None

    def _const_var(lhs: Any, rhs: Any) -> tuple[int, float] | None:
        if (
            isinstance(lhs, (tuple, list))
            and len(lhs) >= 2
            and str(lhs[0]) == "const"
            and isinstance(rhs, (tuple, list))
            and len(rhs) >= 2
            and str(rhs[0]) == "var"
        ):
            try:
                return int(rhs[1]), float(lhs[1])
            except Exception:
                return None
        return None

    parsed = _const_var(arg[1], arg[2]) or _const_var(arg[2], arg[1])
    if parsed is None:
        return None
    var_idx, omega = parsed
    if not math.isfinite(float(omega)) or float(omega) <= 0.0:
        return None
    return fn, int(var_idx), float(omega)


def _periodic_pair_terms(var_idx: int, omega: float) -> tuple[Any, Any]:
    arg = ("mul", ("const", float(omega)), ("var", int(var_idx)))
    return ("sin", arg), ("cos", arg)


def _eval_term_matrix(terms: Sequence[Any], x: torch.Tensor) -> torch.Tensor | None:
    cols: list[torch.Tensor] = []
    for term in terms:
        try:
            val = eval_node(term, x).reshape(-1)
        except Exception:
            return None
        if not torch.isfinite(val).all():
            return None
        cols.append(val)
    if not cols:
        return torch.zeros((int(x.shape[0]), 0), dtype=x.dtype, device=x.device)
    return torch.stack(cols, dim=1)


def _contextual_periodic_seed_rows(
    *,
    base_terms: Sequence[Any],
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    hp: FactorizedSearchConfig,
    fit_meta: Sequence[tuple[str, torch.Tensor, torch.Tensor]] | None,
) -> list[dict[str, Any]]:
    if not bool(getattr(hp, "periodic_seed_enable", True)):
        return []
    from .engine.search import _periodogram_frequency_hints

    max_hints = int(getattr(hp, "periodic_seed_max_hints", 2))
    min_prominence = float(getattr(hp, "periodic_seed_min_prominence", 8.0))
    tables: list[tuple[str, torch.Tensor, torch.Tensor]] = [("pooled", x_fit, y_fit)]
    if fit_meta:
        tables.extend((str(tid), z_tbl, y_tbl) for tid, z_tbl, y_tbl in fit_meta)
    merged: list[tuple[int, float, str]] = []
    max_merged = max(0, max(2 * max_hints, 4 * max_hints))
    for source, z_tbl, y_tbl in tables:
        resid = y_tbl.reshape(-1, 1)
        if base_terms:
            Phi = _eval_term_matrix(base_terms, z_tbl)
            if Phi is None or int(Phi.shape[1]) <= 0:
                continue
            sol = _solve_linear_coeffs(Phi, y_tbl.reshape(-1, 1), ridge=1.0e-8)
            if sol is None or not torch.isfinite(sol).all():
                continue
            resid = y_tbl.reshape(-1, 1) - Phi @ sol
        try:
            hints = _periodogram_frequency_hints(
                z_tbl,
                resid,
                max_hints=max_hints,
                min_prominence=min_prominence,
            )
        except Exception:
            hints = []
        for var_idx, omega in hints:
            if all(
                int(var_idx) != prev_var or abs(float(omega) - prev_w) > 0.05 * max(float(omega), prev_w)
                for prev_var, prev_w, _prev_source in merged
            ):
                merged.append((int(var_idx), float(omega), str(source)))
    merged = merged[:max_merged]
    rows: list[dict[str, Any]] = []
    base_label = "+".join(node_str(t) for t in base_terms) if base_terms else "zero"
    for var_idx, omega, source in merged:
        for fn_name in ("sin", "cos"):
            expr_obj = (fn_name, ("mul", ("const", float(omega)), ("var", int(var_idx))))
            rows.append(
                {
                    "expr": node_str(expr_obj),
                    "_expr_obj": expr_obj,
                    "score": -2.0,
                    "mapping_kind": "contextual_residual_periodic_seed",
                    "contextual_base": base_label,
                    "periodic_hint_source": str(source),
                }
            )
    return rows


def _build_contextual_atom_rows(
    *,
    spec: DELabSpec,
    order: int,
    base_rows: Sequence[dict[str, Any]],
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    hp: FactorizedSearchConfig,
    fit_meta: Sequence[tuple[str, torch.Tensor, torch.Tensor]] | None = None,
    probe_meta: Sequence[tuple[str, torch.Tensor, torch.Tensor]] | None = None,
    traj_metric: str = "mean",
    diagnostics_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Promote discovered atoms by residualized additive-context value.

    This is a small variable-projection layer around FSS atoms: an atom is not
    scored only as a singleton, but as a possible additive term after refitting
    cheap base supports such as [1], [u], or [1,u,du].  Periodic atoms also get
    a sin/cos phase companion and a local frequency grid in that context.
    """

    if not bool(getattr(hp, "de_sparse_combo_enable", False)):
        return []

    try:
        pool_topk = int(getattr(hp, "de_sparse_combo_pool_topk", max(2, int(hp.return_topk))))
    except Exception:
        pool_topk = max(2, int(getattr(hp, "return_topk", 8)))
    atom_cap = max(4, min(max(pool_topk, int(getattr(hp, "return_topk", 8))), 48))
    trace_cap = 48
    ridge = float(getattr(hp, "de_sparse_combo_ridge", 1.0e-8))
    prune_rel = float(getattr(hp, "de_sparse_combo_prune_rel", 1.0e-5))
    max_depth = int(getattr(hp, "max_depth", 5)) + 3
    complexity_penalty = float(getattr(hp, "complexity_penalty", 0.0))
    mapping_penalty = float(getattr(hp, "mapping_complexity_penalty", 0.0))
    min_rel_improve = max(1.0e-8, float(getattr(hp, "score_head_min_rel_improve", 0.0) or 0.0))

    diag: dict[str, Any] = {
        "enabled": True,
        "atoms_considered": 0,
        "contexts_tested": 0,
        "promoted_rows": 0,
        "rejected_no_improvement": 0,
        "rejected_bad_fit": 0,
        "trace": [],
    }

    def _finish(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        diag["promoted_rows"] = int(len(rows))
        if diagnostics_out is not None:
            diagnostics_out.update(_to_jsonable(diag))
        return rows

    base_supports = _context_base_supports(spec, order)
    if not base_supports:
        return _finish([])

    by_expr: dict[str, dict[str, Any]] = {}
    for row in sorted(base_rows, key=lambda r: float(r.get("score", float("inf")))):
        expr_obj = row.get("_expr_obj", None)
        if expr_obj is None:
            continue
        key = str(node_str(expr_obj))
        if key not in by_expr:
            by_expr[key] = row

    residual_seed_rows: list[dict[str, Any]] = []
    for support in base_supports:
        residual_seed_rows.extend(
            _contextual_periodic_seed_rows(
                base_terms=support,
                x_fit=x_fit,
                y_fit=y_fit,
                hp=hp,
                fit_meta=fit_meta,
            )
        )
    for row in residual_seed_rows:
        expr_obj = row.get("_expr_obj", None)
        if expr_obj is None:
            continue
        key = str(node_str(expr_obj))
        prev = by_expr.get(key)
        if prev is None or float(row.get("score", float("inf"))) < float(prev.get("score", float("inf"))):
            by_expr[key] = row

    atoms = list(by_expr.values())[:atom_cap]
    diag["atoms_considered"] = int(len(atoms))
    diag["residual_periodic_seed_rows"] = int(len(residual_seed_rows))

    def _score_terms(terms: Sequence[Any]) -> tuple[float, float, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
        Phi_fit = _eval_term_matrix(terms, x_fit)
        Phi_probe = _eval_term_matrix(terms, x_probe)
        if Phi_fit is None or Phi_probe is None:
            return float("inf"), float("inf"), None, None, {"reason": "eval_failed"}
        if int(Phi_fit.shape[1]) <= 0:
            pred_probe = torch.zeros_like(y_probe.reshape(-1, 1))
            score = float(torch.mean((pred_probe - y_probe.reshape(-1, 1)) ** 2).detach().cpu().item())
            return float(score), float(score), torch.zeros((0, 1), dtype=x_fit.dtype, device=x_fit.device), Phi_fit, {"pooled_probe_mse": float(score)}
        sol = _solve_linear_coeffs(Phi_fit, y_fit.reshape(-1, 1), ridge)
        if sol is None or not torch.isfinite(sol).all():
            return float("inf"), float("inf"), None, Phi_fit, {"reason": "solve_failed"}
        pred_probe = Phi_probe @ sol
        pooled_probe_mse = float(torch.mean((pred_probe - y_probe.reshape(-1, 1)) ** 2).detach().cpu().item())
        if probe_meta is None or len(probe_meta) <= 0:
            return float(pooled_probe_mse), float(pooled_probe_mse), sol, Phi_fit, {
                "pooled_probe_mse": float(pooled_probe_mse)
            }
        traj_rows: list[dict[str, Any]] = []
        for tid, zp, yp in probe_meta:
            Phi_i = _eval_term_matrix(terms, zp)
            if Phi_i is None:
                return float("inf"), float("inf"), sol, Phi_fit, {"reason": "probe_meta_eval_failed"}
            pred_i = Phi_i @ sol
            mse_i = float(torch.mean((pred_i - yp.reshape(-1, 1)) ** 2).detach().cpu().item())
            if not math.isfinite(mse_i):
                return float("inf"), float("inf"), sol, Phi_fit, {"reason": "nonfinite_probe_meta"}
            traj_rows.append({"traj_id": str(tid), "mse": float(mse_i), "n_probe": int(yp.shape[0])})
        score = _aggregate_score([r["mse"] for r in traj_rows], metric=traj_metric)
        return float(score), float(pooled_probe_mse), sol, Phi_fit, {
            "pooled_probe_mse": float(pooled_probe_mse),
            "mse_traj": traj_rows,
        }

    def _omega_trials(desc: tuple[str, int, float] | None) -> list[tuple[float | None, tuple[Any, ...]]]:
        if desc is None:
            return [(None, tuple())]
        _fn, var_idx, omega0 = desc
        span = max(0.10 * abs(float(omega0)), 0.05)
        lo = max(1.0e-8, float(omega0) - span)
        hi = float(omega0) + span
        raw_ws = torch.linspace(lo, hi, 17, dtype=torch.float64).tolist()
        raw_ws.append(float(omega0))
        trials: list[tuple[float | None, tuple[Any, ...]]] = []
        seen: set[float] = set()
        for w in raw_ws:
            ww = float(w)
            key = round(ww, 12)
            if key in seen:
                continue
            seen.add(key)
            trials.append((ww, _periodic_pair_terms(int(var_idx), ww)))
        return trials

    input_exprs = _input_exprs_for_order(spec, order)
    anchor = _anchor_for_order(spec, order)
    mapping_obj = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
    mapping_json = _to_jsonable(mapping_obj)
    mapping_cplx = _mapping_complexity(mapping_json)
    rows_by_key: dict[str, dict[str, Any]] = {}

    for atom_row in atoms:
        atom = atom_row.get("_expr_obj", None)
        if atom is None:
            continue
        atom_desc = _periodic_atom_descriptor(atom)
        atom_expr = str(atom_row.get("expr", node_str(atom)))
        for support in base_supports:
            support_key = tuple(node_str(t) for t in support)
            base_score, base_pooled, _base_sol, _base_phi, _base_extra = _score_terms(support)
            if not math.isfinite(base_score):
                continue
            best_trial: dict[str, Any] | None = None
            for omega, periodic_terms in _omega_trials(atom_desc):
                atom_terms = tuple(periodic_terms) if periodic_terms else (atom,)
                terms = tuple(support) + tuple(atom_terms)
                if len({node_str(t) for t in terms}) != len(terms):
                    continue
                diag["contexts_tested"] = int(diag["contexts_tested"]) + 1
                score, pooled_probe_mse, sol, Phi_fit, extra = _score_terms(terms)
                if sol is None or Phi_fit is None or not math.isfinite(score):
                    diag["rejected_bad_fit"] = int(diag["rejected_bad_fit"]) + 1
                    continue
                if score >= base_score * (1.0 - min_rel_improve):
                    diag["rejected_no_improvement"] = int(diag["rejected_no_improvement"]) + 1
                    if len(diag["trace"]) < trace_cap:
                        diag["trace"].append(
                            {
                                "atom": atom_expr,
                                "base": list(support_key),
                                "omega": None if omega is None else float(omega),
                                "base_probe_mse": float(base_score),
                                "probe_mse": float(score),
                                "delta_vs_base": float(base_score - score),
                                "promoted": False,
                                "reason": "no_contextual_improvement",
                            }
                        )
                    continue

                try:
                    coeffs_1d = sol.reshape(-1)
                    atom_slice = coeffs_1d[len(support) :]
                    contrib = []
                    for jj in range(len(support), int(Phi_fit.shape[1])):
                        col = Phi_fit[:, jj]
                        contrib.append(abs(float(coeffs_1d[jj])) * float(torch.sqrt(torch.mean(col * col)).detach().cpu().item()))
                    atom_contrib = max(contrib) if contrib else 0.0
                    all_contrib = []
                    for jj in range(int(Phi_fit.shape[1])):
                        col = Phi_fit[:, jj]
                        all_contrib.append(abs(float(coeffs_1d[jj])) * float(torch.sqrt(torch.mean(col * col)).detach().cpu().item()))
                    max_contrib = max(all_contrib) if all_contrib else 0.0
                except Exception:
                    atom_contrib = 0.0
                    max_contrib = 0.0
                    atom_slice = torch.zeros((0,), dtype=x_fit.dtype, device=x_fit.device)
                if atom_contrib <= max(1.0e-12, float(prune_rel) * max(1.0, max_contrib)):
                    diag["rejected_no_improvement"] = int(diag["rejected_no_improvement"]) + 1
                    continue

                trial = {
                    "terms": terms,
                    "omega": omega,
                    "score": float(score),
                    "pooled_probe_mse": float(pooled_probe_mse),
                    "sol": sol,
                    "Phi_fit": Phi_fit,
                    "extra": extra,
                    "atom_coeffs": [float(v) for v in atom_slice.detach().cpu().reshape(-1).tolist()],
                    "atom_contrib": float(atom_contrib),
                    "base_score": float(base_score),
                    "base_pooled": float(base_pooled),
                }
                if best_trial is None or float(trial["score"]) < float(best_trial["score"]):
                    best_trial = trial

            if best_trial is None:
                continue
            terms = best_trial["terms"]
            sol = best_trial["sol"]
            Phi_fit = best_trial["Phi_fit"]
            expr_combo = _compile_linear_combo(
                list(terms),
                sol.reshape(-1),
                Phi_fit,
                prune_rel,
                max_depth,
            )
            if expr_combo is None:
                diag["rejected_bad_fit"] = int(diag["rejected_bad_fit"]) + 1
                continue
            try:
                compiled_fit = eval_node(expr_combo, x_fit).reshape(-1)
                fit_pred = (Phi_fit @ sol).reshape(-1)
                compile_err = float(torch.linalg.vector_norm(compiled_fit - fit_pred).detach().cpu().item())
                compile_ref = float(torch.linalg.vector_norm(fit_pred).detach().cpu().item()) + 1.0e-30
                if compile_err > 1.0e-6 * compile_ref:
                    diag["rejected_bad_fit"] = int(diag["rejected_bad_fit"]) + 1
                    continue
            except Exception:
                diag["rejected_bad_fit"] = int(diag["rejected_bad_fit"]) + 1
                continue

            score = float(best_trial["score"])
            size = int(node_size(expr_combo))
            score_eff = float(score + complexity_penalty * float(size) + mapping_penalty * float(mapping_cplx))
            rhs_ast = None
            residual_ast = None
            try:
                inner_nn = factorized_search_to_nestynet(expr_combo)
                rhs_ast = embed_mapping_in_ast(inner_nn, mapping_obj, input_exprs, units_mode="raw")
                if rhs_ast is not None:
                    residual_ast = Add(anchor, Mul(ConstNode(-1.0), rhs_ast))
            except Exception:
                rhs_ast = None
                residual_ast = None
            compiled_ast_meta = _compiled_ast_metadata(rhs_ast, residual_ast)
            expr_str = node_str(expr_combo)
            source_exprs = [node_str(t) for t in terms]
            row = {
                "order": int(order),
                "seed_search": -1,
                "expr": expr_str,
                "expr_ast": _to_jsonable(expr_combo),
                "mse": float(score),
                "score_raw": float(score),
                "score": float(score_eff),
                "mse_pooled": float(best_trial["pooled_probe_mse"]),
                "raw_mse": float(best_trial["pooled_probe_mse"]),
                "final_validated_mse": float(score),
                "mse_traj": list((best_trial.get("extra") or {}).get("mse_traj", [])),
                "size": int(size),
                "mapping_complexity": int(mapping_cplx),
                "mapping": mapping_json,
                "mapping_kind": "poly",
                "score_ladder": {
                    "schema_version": 1,
                    "contextual_atom": {
                        "atom": atom_expr,
                        "base": list(support_key),
                        "source": str(atom_row.get("mapping_kind", "")),
                        "omega": best_trial.get("omega", None),
                        "base_probe_mse": float(best_trial["base_score"]),
                        "probe_mse": float(score),
                        "delta_vs_base": float(best_trial["base_score"] - score),
                    },
                    "final_validation": {
                        "available": True,
                        "source": "de_probe_trajectories",
                        "metric": str(traj_metric),
                        "probe_mse": float(score),
                        "score_with_penalties": float(score_eff),
                    },
                },
                "acceptance_basis": "contextual_atom_delta",
                "final_acceptance_basis": "de_probe_validation",
                **compiled_ast_meta,
                "construction": "contextual_atom_promotion",
                "source_lane": "contextual_atom_promotion",
                "contextual_atom": atom_expr,
                "contextual_atom_source": str(atom_row.get("mapping_kind", "")),
                "contextual_base": list(support_key),
                "contextual_source_exprs": source_exprs,
                "contextual_coeffs": [float(v) for v in sol.reshape(-1).detach().cpu().tolist()],
                "contextual_atom_coeffs": list(best_trial["atom_coeffs"]),
                "contextual_delta_vs_base": float(best_trial["base_score"] - score),
                "contextual_base_probe_mse": float(best_trial["base_score"]),
                "contextual_refined_omega": best_trial.get("omega", None),
                "_expr_obj": expr_combo,
                "_mapping_obj": mapping_obj,
            }
            key = f"{expr_str}\n{json.dumps(mapping_json, sort_keys=True)}"
            prev = rows_by_key.get(key)
            if prev is None or float(row["score"]) < float(prev["score"]):
                rows_by_key[key] = row
            if len(diag["trace"]) < trace_cap:
                diag["trace"].append(
                    {
                        "atom": atom_expr,
                        "source": str(atom_row.get("mapping_kind", "")),
                        "base": list(support_key),
                        "omega": best_trial.get("omega", None),
                        "coeffs": row["contextual_coeffs"],
                        "base_probe_mse": float(best_trial["base_score"]),
                        "probe_mse": float(score),
                        "delta_vs_base": float(best_trial["base_score"] - score),
                        "expr": expr_str,
                        "promoted": True,
                        "promotion_reason": "contextual_delta",
                    }
                )

    rows = list(rows_by_key.values())
    rows.sort(
        key=lambda row: (
            float(row.get("score", float("inf"))),
            int(row.get("size", 10**9)),
            str(row.get("expr", "")),
        )
    )
    return _finish(rows)


def _build_sparse_combo_rows(
    *,
    spec: DELabSpec,
    order: int,
    base_rows: Sequence[dict[str, Any]],
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    hp: FactorizedSearchConfig,
    probe_meta: Sequence[tuple[str, torch.Tensor, torch.Tensor]] | None = None,
    traj_metric: str = "mean",
    diagnostics_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build additive basis-state candidates from factorized symbolic search-discovered atoms.

    The previous sparse-combo lane was pairwise. This version treats the top
    factorized symbolic search rows as a small discovered atom pool and performs a beam search over
    basis states. Coefficients are nuisance fits; the selected structure is the
    active set of symbolic atoms, which makes this a operator-factorized DE proposal lane
    rather than an STLSQ-style fixed-library threshold loop.
    """

    if not bool(getattr(hp, "de_sparse_combo_enable", False)):
        return []

    try:
        max_terms = int(getattr(hp, "de_sparse_combo_max_terms", 2))
    except Exception:
        max_terms = 2
    max_terms = max(2, max_terms)

    try:
        pool_topk = int(getattr(hp, "de_sparse_combo_pool_topk", max(2, int(hp.return_topk))))
    except Exception:
        pool_topk = max(2, int(getattr(hp, "return_topk", 8)))
    pool_topk = max(2, pool_topk)

    try:
        beam = int(getattr(hp, "de_sparse_combo_beam", 16))
    except Exception:
        beam = 16
    beam = max(1, beam)

    try:
        ridge = float(getattr(hp, "de_sparse_combo_ridge", 1.0e-8))
    except Exception:
        ridge = 1.0e-8
    try:
        prune_rel = float(getattr(hp, "de_sparse_combo_prune_rel", 1.0e-5))
    except Exception:
        prune_rel = 1.0e-10

    backward_prune = bool(getattr(hp, "de_sparse_combo_backward_prune", True))
    combo_mapping_mode = str(
        getattr(hp, "de_sparse_combo_mapping_mode", "affine_only") or "affine_only"
    ).strip().lower()
    mapping_mode = "affine_only"
    if combo_mapping_mode not in {"affine_only", "linear_only", "poly_only", "poly"}:
        if diagnostics_out is not None:
            diagnostics_out.update(
                {
                    "status": "disabled_by_mapping_guard",
                    "requested_mapping_mode": combo_mapping_mode,
                    "reason": "additive_composer_requires_affine_only_outer_mapping",
                }
            )
        return []
    mapping_penalty = float(getattr(hp, "mapping_complexity_penalty", 0.0))
    complexity_penalty = float(getattr(hp, "complexity_penalty", 0.0))
    poly_degree = 1
    # A faithful K-term scaled linear combo of depth-bounded atoms legitimately
    # exceeds the per-atom search depth (scaling muls + the add tree); without
    # headroom the compile truncates and the faithfulness gate rejects it.
    max_depth = int(getattr(hp, "max_depth", 5)) + 3
    try:
        corr_eps = float(getattr(hp, "de_sparse_combo_corr_eps", 1.0e-8))
    except Exception:
        corr_eps = 1.0e-8
    try:
        rank_eps = float(getattr(hp, "de_sparse_combo_rank_eps", 1.0e-10))
    except Exception:
        rank_eps = 1.0e-10
    try:
        max_condition = float(getattr(hp, "de_sparse_combo_max_condition", 1.0e10))
    except Exception:
        max_condition = 1.0e10
    try:
        cond_penalty = float(getattr(hp, "de_sparse_combo_cond_penalty", 0.0))
    except Exception:
        cond_penalty = 0.0
    try:
        coeff_stability_penalty = float(getattr(hp, "de_sparse_combo_coeff_stability_penalty", 0.0))
    except Exception:
        coeff_stability_penalty = 0.0
    try:
        coeff_spread_warn = float(getattr(hp, "de_sparse_combo_coeff_spread_warn", 2.0))
    except Exception:
        coeff_spread_warn = 2.0
    combo_diag: dict[str, Any] = {
        "requested_mapping_mode": combo_mapping_mode,
        "effective_mapping_mode": mapping_mode,
        "effective_poly_degree": int(poly_degree),
        "symbolic_duplicates": 0,
        "functional_duplicates": 0,
        "rank_rejected": 0,
        "condition_rejected": 0,
        "nonlinear_mapping_rejected": 0,
        "compile_unfaithful_rejected": 0,
        "states_fit": 0,
        "periodic_base_rows_count": 0,
        "periodic_base_rows": [],
        "periodic_carrier_rows_count": 0,
        "periodic_carrier_rows": [],
        "periodic_terms_count": 0,
        "periodic_terms": [],
        "periodic_combo_rows_count": 0,
        "periodic_combo_rows": [],
    }
    periodic_diag_cap = 16

    def _safe_diag_float(value: Any) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    def _parse_periodic_arg(arg: Any) -> tuple[float, int] | None:
        if not isinstance(arg, (tuple, list)) or len(arg) < 2:
            return None
        op = str(arg[0])
        if op == "var":
            try:
                return 1.0, int(arg[1])
            except Exception:
                return None
        if op != "mul" or len(arg) != 3:
            return None

        def _const_var(lhs: Any, rhs: Any) -> tuple[float, int] | None:
            if (
                isinstance(lhs, (tuple, list))
                and len(lhs) >= 2
                and str(lhs[0]) == "const"
                and isinstance(rhs, (tuple, list))
                and len(rhs) >= 2
                and str(rhs[0]) == "var"
            ):
                try:
                    return float(lhs[1]), int(rhs[1])
                except Exception:
                    return None
            return None

        return _const_var(arg[1], arg[2]) or _const_var(arg[2], arg[1])

    def _dedupe_periodic_matches(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, int, float]] = set()
        for match in matches:
            try:
                fn = str(match.get("fn", ""))
                var_idx = int(match.get("var_idx", -1))
                omega = float(match.get("omega", float("nan")))
            except Exception:
                continue
            if fn not in {"sin", "cos"} or var_idx < 0 or not math.isfinite(omega):
                continue
            key = (fn, var_idx, round(omega, 12))
            if key in seen:
                continue
            seen.add(key)
            out.append({"fn": fn, "var": f"x{var_idx}", "var_idx": int(var_idx), "omega": float(omega)})
        return out

    def _periodic_matches_from_ast(expr_obj: Any) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []

        def _walk(node: Any) -> None:
            if not isinstance(node, (tuple, list)) or len(node) == 0:
                return
            op = str(node[0])
            if op in {"sin", "cos"} and len(node) >= 2:
                parsed = _parse_periodic_arg(node[1])
                if parsed is not None:
                    omega, var_idx = parsed
                    matches.append(
                        {"fn": op, "var": f"x{int(var_idx)}", "var_idx": int(var_idx), "omega": float(omega)}
                    )
            for child in node[1:]:
                _walk(child)

        _walk(expr_obj)
        return _dedupe_periodic_matches(matches)

    _num_re = r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?"
    _scaled_trig_re = re.compile(
        rf"\b(sin|cos)\s*\(\s*\(?\s*({_num_re})\s*\*\s*x(\d+)\s*\)?\s*\)"
    )
    _unit_trig_re = re.compile(r"\b(sin|cos)\s*\(\s*x(\d+)\s*\)")

    def _periodic_matches_from_text(text: Any) -> list[dict[str, Any]]:
        if text is None:
            return []
        s = str(text)
        matches: list[dict[str, Any]] = []
        for m in _scaled_trig_re.finditer(s):
            try:
                matches.append(
                    {
                        "fn": str(m.group(1)),
                        "var": f"x{int(m.group(3))}",
                        "var_idx": int(m.group(3)),
                        "omega": float(m.group(2)),
                    }
                )
            except Exception:
                continue
        for m in _unit_trig_re.finditer(s):
            try:
                matches.append(
                    {
                        "fn": str(m.group(1)),
                        "var": f"x{int(m.group(2))}",
                        "var_idx": int(m.group(2)),
                        "omega": 1.0,
                    }
                )
            except Exception:
                continue
        return _dedupe_periodic_matches(matches)

    def _periodic_matches(exprish: Any) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        if isinstance(exprish, (tuple, list)):
            matches.extend(_periodic_matches_from_ast(exprish))
            try:
                matches.extend(_periodic_matches_from_text(node_str(exprish)))
            except Exception:
                pass
        else:
            matches.extend(_periodic_matches_from_text(exprish))
        return _dedupe_periodic_matches(matches)

    def _source_row_summary(row: Mapping[str, Any], rank: int) -> dict[str, Any]:
        expr_obj = row.get("_expr_obj", None)
        expr = str(row.get("expr", node_str(expr_obj) if expr_obj is not None else ""))
        matches = _periodic_matches(expr_obj if expr_obj is not None else expr)
        return {
            "rank": int(rank),
            "expr": expr,
            "score": _safe_diag_float(row.get("score", None)),
            "size": int(node_size(expr_obj)) if expr_obj is not None else None,
            "mapping_kind": str(row.get("mapping_kind", "")),
            "periodic_matches": matches,
        }

    def _term_summary(term: Mapping[str, Any], rank: int) -> dict[str, Any]:
        row = term.get("row", {})
        summary = _source_row_summary(row if isinstance(row, Mapping) else {}, rank)
        summary["pool_index"] = int(term.get("pool_index", -1))
        summary["expr"] = str(term.get("expr", summary.get("expr", "")))
        return summary

    def _combo_row_summary(row: Mapping[str, Any], rank: int) -> dict[str, Any]:
        source_exprs = [str(v) for v in row.get("combo_source_exprs", []) or []]
        matches: list[dict[str, Any]] = []
        matches.extend(_periodic_matches(row.get("_expr_obj", None)))
        matches.extend(_periodic_matches(row.get("expr", "")))
        for expr in source_exprs:
            matches.extend(_periodic_matches(expr))
        matches = _dedupe_periodic_matches(matches)
        return {
            "rank": int(rank),
            "expr": str(row.get("expr", "")),
            "score": _safe_diag_float(row.get("score", None)),
            "mse": _safe_diag_float(row.get("mse", None)),
            "raw_mse": _safe_diag_float(row.get("raw_mse", None)),
            "size": int(row.get("size", -1)) if row.get("size", None) is not None else None,
            "combo_n_terms": int(row.get("combo_n_terms", 0)),
            "combo_source_exprs": source_exprs,
            "combo_coeffs": [
                float(v)
                for v in (row.get("combo_coeffs", []) or [])
                if _safe_diag_float(v) is not None
            ],
            "mapping_kind": str(row.get("mapping_kind", "")),
            "periodic_matches": matches,
        }

    def _finish(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        combo_diag["rows"] = int(len(rows))
        periodic_rows: list[dict[str, Any]] = []
        for rank, row in enumerate(rows):
            summary = _combo_row_summary(row, rank)
            if len(summary.get("periodic_matches", [])) > 0:
                periodic_rows.append(summary)
        combo_diag["periodic_combo_rows_count"] = int(len(periodic_rows))
        combo_diag["periodic_combo_rows"] = periodic_rows[:periodic_diag_cap]
        if periodic_rows:
            combo_diag["periodic_best_combo_rank"] = int(periodic_rows[0]["rank"])
            combo_diag["periodic_best_combo_score"] = periodic_rows[0].get("score", None)
        if diagnostics_out is not None:
            diagnostics_out.update(_to_jsonable(combo_diag))
        return rows

    def _column_scaled_matrix(Phi: torch.Tensor) -> torch.Tensor | None:
        try:
            M = torch.as_tensor(Phi, dtype=torch.float64)
            if M.ndim != 2 or int(M.shape[0]) <= 0 or int(M.shape[1]) <= 0:
                return None
            if not torch.isfinite(M).all():
                return None
            scale = torch.sqrt(torch.mean(M * M, dim=0, keepdim=True))
            scale = torch.clamp(scale, min=1.0e-12)
            return M / scale
        except Exception:
            return None

    def _design_diagnostics(Phi: torch.Tensor) -> dict[str, Any]:
        M = _column_scaled_matrix(Phi)
        if M is None:
            return {"condition": None, "rank_ratio": None, "rank_ok": False, "condition_ok": False}
        try:
            s = torch.linalg.svdvals(M)
            if int(s.numel()) == 0:
                return {"condition": None, "rank_ratio": None, "rank_ok": False, "condition_ok": False}
            s_max = float(torch.max(s).detach().cpu().item())
            s_min = float(torch.min(s).detach().cpu().item())
        except Exception:
            return {"condition": None, "rank_ratio": None, "rank_ok": False, "condition_ok": False}
        if not math.isfinite(s_max) or not math.isfinite(s_min) or s_max <= 0.0:
            return {"condition": None, "rank_ratio": None, "rank_ok": False, "condition_ok": False}
        rank_ratio = float(s_min / max(s_max, 1.0e-300))
        condition = float(s_max / max(s_min, 1.0e-300))
        return {
            "condition": float(condition),
            "rank_ratio": float(rank_ratio),
            "rank_ok": bool(rank_ratio >= max(0.0, float(rank_eps))),
            "condition_ok": bool(
                (not math.isfinite(max_condition)) or max_condition <= 0.0 or condition <= max_condition
            ),
        }

    def _abs_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        try:
            av = torch.as_tensor(a, dtype=torch.float64).reshape(-1)
            bv = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
            mask = torch.isfinite(av) & torch.isfinite(bv)
            av = av[mask]
            bv = bv[mask]
            if int(av.numel()) < 2:
                return 0.0
            av = av - torch.mean(av)
            bv = bv - torch.mean(bv)
            na = torch.sqrt(torch.mean(av * av))
            nb = torch.sqrt(torch.mean(bv * bv))
            if float(na) <= 1.0e-14 or float(nb) <= 1.0e-14:
                raw = torch.sqrt(torch.mean((torch.as_tensor(a, dtype=torch.float64).reshape(-1) - torch.as_tensor(b, dtype=torch.float64).reshape(-1)) ** 2))
                return 1.0 if float(raw) <= 1.0e-12 else 0.0
            val = torch.mean(av * bv) / (na * nb)
            return float(abs(float(torch.clamp(val, min=-1.0, max=1.0).detach().cpu().item())))
        except Exception:
            return 0.0

    def _mapping_is_affine_only(mapping_json: Any) -> bool:
        if not isinstance(mapping_json, Mapping):
            return False
        kind = str(mapping_json.get("kind", "") or "").strip().lower()
        if kind in {"", "identity", "affine"}:
            return True
        if kind != "poly":
            return False
        coeffs = mapping_json.get("coeffs", [])
        if not isinstance(coeffs, Sequence):
            return False
        return len(list(coeffs)) <= 2

    def _mapping_effective_affine(mapping_json: Mapping[str, Any]) -> tuple[float, float] | None:
        if not _mapping_is_affine_only(mapping_json):
            return None
        kind = str(mapping_json.get("kind", "") or "").strip().lower()
        if kind in {"", "identity"}:
            return 1.0, 0.0
        if kind == "affine":
            try:
                return float(mapping_json.get("a", 1.0)), float(mapping_json.get("b", 0.0))
            except Exception:
                return None
        coeffs = list(mapping_json.get("coeffs", []) or [])
        try:
            if len(coeffs) <= 0:
                return 0.0, 0.0
            if len(coeffs) == 1:
                return 0.0, float(coeffs[0])
            mu = float(mapping_json.get("mu", 0.0))
            std = float(mapping_json.get("std", 1.0))
            if not math.isfinite(std) or abs(std) <= 1.0e-300:
                return None
            slope = float(coeffs[1]) / std
            intercept = float(coeffs[0]) - slope * mu
            return float(slope), float(intercept)
        except Exception:
            return None

    def _solve_augmented_coeffs(Phi: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
        A = None
        try:
            ones = torch.ones((int(Phi.shape[0]), 1), dtype=Phi.dtype, device=Phi.device)
            A = torch.cat([Phi, ones], dim=1)
            AtA = A.T @ A
            if ridge > 0.0:
                eye = torch.eye(int(AtA.shape[0]), dtype=AtA.dtype, device=AtA.device)
                eye[-1, -1] = 0.0
                AtA = AtA + float(ridge) * eye
            rhs = A.T @ y.reshape(-1, 1)
            sol_aug = torch.linalg.solve(AtA, rhs).reshape(-1)
            if torch.isfinite(sol_aug).all():
                return sol_aug
        except Exception:
            if A is None:
                return None
            try:
                sol_aug = torch.linalg.lstsq(A, y.reshape(-1, 1)).solution.reshape(-1)
                if torch.isfinite(sol_aug).all():
                    return sol_aug
            except Exception:
                return None
        return None

    def _coeff_stability_diag(
        *,
        state: Sequence[int],
        sol: torch.Tensor,
        mapping_json: Mapping[str, Any],
    ) -> dict[str, Any]:
        if probe_meta is None or len(probe_meta) <= 1:
            return {"available": False, "reason": "insufficient_probe_trajectories"}
        eff = _mapping_effective_affine(mapping_json)
        if eff is None:
            return {"available": False, "reason": "non_affine_mapping"}
        slope, intercept = eff
        global_eff = torch.cat(
            [
                torch.as_tensor(sol.reshape(-1), dtype=torch.float64) * float(slope),
                torch.tensor([float(intercept)], dtype=torch.float64, device=sol.device),
            ]
        )
        rows: list[dict[str, Any]] = []
        spreads: list[float] = []
        for tid, _zp, yp in probe_meta:
            vals = []
            for i in state:
                vv = terms[i]["traj_vals"].get(str(tid), None)
                if vv is None:
                    return {"available": False, "reason": "missing_traj_values"}
                vals.append(vv)
            try:
                Phi_i = torch.stack(vals, dim=1)
            except Exception:
                return {"available": False, "reason": "trajectory_stack_failed"}
            sol_i = _solve_augmented_coeffs(Phi_i, yp)
            if sol_i is None:
                rows.append({"traj_id": str(tid), "status": "ERROR"})
                continue
            local = torch.as_tensor(sol_i, dtype=torch.float64)
            denom = torch.clamp(torch.abs(global_eff), min=1.0e-12)
            spread_i = float(torch.max(torch.abs(local - global_eff) / denom).detach().cpu().item())
            if math.isfinite(spread_i):
                spreads.append(spread_i)
            rows.append(
                {
                    "traj_id": str(tid),
                    "status": "OK",
                    "spread_rel": None if not math.isfinite(spread_i) else float(spread_i),
                    "coeffs": [float(v) for v in local.detach().cpu().tolist()],
                }
            )
        max_spread = max(spreads) if spreads else float("inf")
        return {
            "available": bool(spreads),
            "max_spread_rel": None if not math.isfinite(max_spread) else float(max_spread),
            "warn": bool(math.isfinite(max_spread) and max_spread > max(0.0, float(coeff_spread_warn))),
            "rows": rows,
        }

    by_expr: dict[str, dict[str, Any]] = {}
    for row in sorted(base_rows, key=lambda r: float(r.get("score", float("inf")))):
        expr_obj = row.get("_expr_obj", None)
        if expr_obj is None:
            continue
        key = str(node_str(expr_obj))
        prev = by_expr.get(key)
        if prev is None or float(row.get("score", float("inf"))) < float(prev.get("score", float("inf"))):
            by_expr[key] = row
        else:
            combo_diag["symbolic_duplicates"] = int(combo_diag["symbolic_duplicates"]) + 1

    carrier_rows = list(by_expr.values())[:pool_topk]
    periodic_base_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(list(by_expr.values())):
        summary = _source_row_summary(row, rank)
        if summary.get("periodic_matches"):
            periodic_base_rows.append(summary)
    combo_diag["periodic_base_rows_count"] = int(len(periodic_base_rows))
    combo_diag["periodic_base_rows"] = periodic_base_rows[:periodic_diag_cap]
    periodic_carrier_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(carrier_rows):
        summary = _source_row_summary(row, rank)
        if summary.get("periodic_matches"):
            periodic_carrier_rows.append(summary)
    combo_diag["periodic_carrier_rows_count"] = int(len(periodic_carrier_rows))
    combo_diag["periodic_carrier_rows"] = periodic_carrier_rows[:periodic_diag_cap]
    if len(carrier_rows) < 2:
        return _finish([])

    terms: list[dict[str, Any]] = []
    for pool_i, row in enumerate(carrier_rows):
        expr_obj = row.get("_expr_obj", None)
        if expr_obj is None:
            continue
        try:
            phi_fit = eval_node(expr_obj, x_fit).reshape(-1)
            phi_probe = eval_node(expr_obj, x_probe).reshape(-1)
        except Exception:
            continue
        if (not torch.isfinite(phi_fit).all()) or (not torch.isfinite(phi_probe).all()):
            continue

        traj_vals: dict[str, torch.Tensor] = {}
        if probe_meta is not None:
            ok = True
            for tid, zp, _ in probe_meta:
                try:
                    vv = eval_node(expr_obj, zp).reshape(-1)
                except Exception:
                    ok = False
                    break
                if not torch.isfinite(vv).all():
                    ok = False
                    break
                traj_vals[str(tid)] = vv
            if not ok:
                continue

        term = {
            "pool_index": int(pool_i),
            "row": row,
            "expr_obj": expr_obj,
            "expr": str(row.get("expr", node_str(expr_obj))),
            "phi_fit": phi_fit,
            "phi_probe": phi_probe,
            "traj_vals": traj_vals,
        }
        duplicate_index = None
        if math.isfinite(corr_eps) and corr_eps > 0.0:
            corr_cutoff = max(0.0, min(1.0, 1.0 - float(corr_eps)))
            for j, prev_term in enumerate(terms):
                if _abs_corr(phi_fit, prev_term["phi_fit"]) >= corr_cutoff:
                    duplicate_index = j
                    break
        if duplicate_index is not None:
            combo_diag["functional_duplicates"] = int(combo_diag["functional_duplicates"]) + 1
            prev_term = terms[duplicate_index]
            try:
                prev_key = (
                    float(prev_term["row"].get("score", float("inf"))),
                    int(node_size(prev_term["expr_obj"])),
                    str(prev_term["expr"]),
                )
                new_key = (
                    float(row.get("score", float("inf"))),
                    int(node_size(expr_obj)),
                    str(term["expr"]),
                )
            except Exception:
                prev_key = (0.0, 0, "")
                new_key = (1.0, 1, "")
            if new_key < prev_key:
                terms[duplicate_index] = term
            continue
        terms.append(term)

    periodic_terms: list[dict[str, Any]] = []
    for rank, term in enumerate(terms):
        summary = _term_summary(term, rank)
        if summary.get("periodic_matches"):
            periodic_terms.append(summary)
    combo_diag["periodic_terms_count"] = int(len(periodic_terms))
    combo_diag["periodic_terms"] = periodic_terms[:periodic_diag_cap]
    if len(terms) < 2:
        return _finish([])

    input_exprs = _input_exprs_for_order(spec, order)
    anchor = _anchor_for_order(spec, order)
    score_cfg = {
        "score_mapping_family_mode": mapping_mode,
        "score_pade_structural_enable": bool(getattr(hp, "score_pade_structural_enable", False)),
        "score_pade_structural_max_degree": int(getattr(hp, "score_pade_structural_max_degree", 2)),
        "score_pade_structural_max_total_degree": int(
            getattr(hp, "score_pade_structural_max_total_degree", 3)
        ),
        "score_pade_structural_max_depth": int(getattr(hp, "score_pade_structural_max_depth", 8)),
        "score_pade_structural_max_size": int(getattr(hp, "score_pade_structural_max_size", 64)),
        "score_pade_structural_coeff_tol": float(
            getattr(hp, "score_pade_structural_coeff_tol", 1.0e-10)
        ),
        "score_pade_structural_mse_rel_tol": float(
            getattr(hp, "score_pade_structural_mse_rel_tol", 1.0e-6)
        ),
    }

    def _state_key(indices: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted({int(i) for i in indices}))

    def _fit_state(indices: Sequence[int]) -> dict[str, Any] | None:
        state = _state_key(indices)
        if len(state) < 2:
            return None
        if len(state) > max_terms:
            return None

        try:
            Phi_fit = torch.stack([terms[i]["phi_fit"] for i in state], dim=1)
            Phi_probe = torch.stack([terms[i]["phi_probe"] for i in state], dim=1)
        except Exception:
            return None
        design_diag = _design_diagnostics(Phi_fit)
        if not bool(design_diag.get("rank_ok", False)):
            combo_diag["rank_rejected"] = int(combo_diag["rank_rejected"]) + 1
            return None
        if not bool(design_diag.get("condition_ok", False)):
            combo_diag["condition_rejected"] = int(combo_diag["condition_rejected"]) + 1
            return None

        sol = _solve_linear_coeffs(Phi_fit, y_fit, ridge)
        if sol is None or (not torch.isfinite(sol).all()):
            return None

        pred_fit = Phi_fit @ sol
        fb = _fit_best_with_cfg(pred_fit, y_fit, poly_degree, score_cfg)
        if fb is None:
            return None
        _, mapping_obj = fb
        mapping_json = _to_jsonable(mapping_obj)
        if not _mapping_is_affine_only(mapping_json):
            combo_diag["nonlinear_mapping_rejected"] = int(combo_diag["nonlinear_mapping_rejected"]) + 1
            return None

        pred_probe = Phi_probe @ sol
        yhat_probe = eval_mapping(pred_probe, mapping_obj).reshape(-1, 1)
        if not torch.isfinite(yhat_probe).all():
            return None
        pooled_probe_mse = float(torch.mean((yhat_probe - y_probe) ** 2).detach().cpu().item())
        if not math.isfinite(pooled_probe_mse):
            return None

        exprs = [terms[i]["expr_obj"] for i in state]
        expr_combo = _compile_linear_combo(
            exprs,
            sol.squeeze(-1),
            Phi_fit,
            prune_rel,
            max_depth,
        )
        if expr_combo is None:
            return None
        # _compile_linear_combo may drop terms to satisfy max_depth, which
        # would desync the emitted AST from the scored linear span (the row
        # would keep the full-combo score/mapping but a truncated expr_ast).
        # Reject any state whose compiled AST is not faithful to Phi @ sol.
        try:
            pred_compiled = eval_node(expr_combo, x_fit).reshape(-1)
        except Exception:
            return None
        if not torch.isfinite(pred_compiled).all():
            return None
        compile_err = float(torch.linalg.vector_norm(pred_compiled - pred_fit.reshape(-1)))
        compile_ref = float(torch.linalg.vector_norm(pred_fit)) + 1.0e-30
        if compile_err > 1.0e-6 * compile_ref:
            combo_diag["compile_unfaithful_rejected"] = int(combo_diag["compile_unfaithful_rejected"]) + 1
            return None
        expr_str = str(node_str(expr_combo))

        traj_rows: list[dict[str, Any]] = []
        if probe_meta is not None and len(probe_meta) > 0:
            traj_ok = True
            for tid, _zp, yp in probe_meta:
                vals = []
                for i in state:
                    vv = terms[i]["traj_vals"].get(str(tid), None)
                    if vv is None:
                        traj_ok = False
                        break
                    vals.append(vv)
                if not traj_ok:
                    break
                Phi_i = torch.stack(vals, dim=1)
                pred_i = Phi_i @ sol
                yhat_i = eval_mapping(pred_i, mapping_obj).reshape(-1, 1)
                if not torch.isfinite(yhat_i).all():
                    traj_ok = False
                    break
                mse_i = float(torch.mean((yhat_i - yp) ** 2).detach().cpu().item())
                if not math.isfinite(mse_i):
                    traj_ok = False
                    break
                traj_rows.append({"traj_id": str(tid), "mse": float(mse_i), "n_probe": int(yp.shape[0])})
            if not traj_ok:
                return None
            score = _aggregate_score([r["mse"] for r in traj_rows], metric=traj_metric)
        else:
            score = float(pooled_probe_mse)
        if not math.isfinite(float(score)):
            return None

        mapping_cplx = _mapping_complexity(mapping_json)
        size = int(node_size(expr_combo))
        condition_penalty = 0.0
        condition = design_diag.get("condition", None)
        if condition is not None and math.isfinite(float(condition)) and math.isfinite(cond_penalty) and cond_penalty > 0.0:
            condition_penalty = float(cond_penalty) * math.log1p(float(condition))
        coeff_stability = _coeff_stability_diag(state=state, sol=sol.squeeze(-1), mapping_json=mapping_json)
        coeff_stability_extra = 0.0
        coeff_spread = coeff_stability.get("max_spread_rel", None) if isinstance(coeff_stability, Mapping) else None
        if (
            coeff_spread is not None
            and math.isfinite(float(coeff_spread))
            and math.isfinite(coeff_stability_penalty)
            and coeff_stability_penalty > 0.0
        ):
            coeff_stability_extra = float(coeff_stability_penalty) * float(coeff_spread)
        combo_diag["states_fit"] = int(combo_diag["states_fit"]) + 1
        score_eff = float(
            score
            + mapping_penalty * float(mapping_cplx)
            + complexity_penalty * float(size)
            + condition_penalty
            + coeff_stability_extra
        )
        carrier_probe_mse = float("inf")
        compiled_mapped_probe_mse = float("inf")
        try:
            carrier_probe_mse = float(torch.mean((pred_probe.reshape(-1, 1) - y_probe) ** 2).detach().cpu().item())
        except Exception:
            carrier_probe_mse = float("inf")
        try:
            pred_probe_compiled = eval_node(expr_combo, x_probe)
            yhat_probe_compiled = eval_mapping(pred_probe_compiled, mapping_obj).reshape(-1, 1)
            if torch.isfinite(yhat_probe_compiled).all():
                compiled_mapped_probe_mse = float(torch.mean((yhat_probe_compiled - y_probe) ** 2).detach().cpu().item())
        except Exception:
            compiled_mapped_probe_mse = float("inf")
        if not math.isfinite(carrier_probe_mse):
            carrier_probe_mse = None
        if not math.isfinite(compiled_mapped_probe_mse):
            compiled_mapped_probe_mse = None
        score_ladder = {
            "schema_version": 1,
            "carrier": {
                "expr": "linear_span",
                "probe_mse_identity": carrier_probe_mse,
            },
            "mapped": {
                "available": True,
                "mapping_kind": str(mapping_json.get("kind", "")) if isinstance(mapping_json, dict) else "",
                "mapping_structural": str(mapping_json.get("kind", "")).lower() in {"", "identity", "poly", "affine"} if isinstance(mapping_json, dict) else False,
                "probe_mse": float(pooled_probe_mse),
                "source": "basis_state_linear_span",
            },
            "head_augmented": {
                "available": False,
                "accepted": False,
                "probe_mse": None,
                "term_count": 0,
            },
            "compiled_structural": {
                "available": True,
                "accepted": True,
                "probe_mse": compiled_mapped_probe_mse,
                "expr": expr_str,
                "source": "compiled_basis_state_ast",
            },
            "refined": {
                "enabled": False,
                "attempted": False,
                "accepted": False,
                "probe_mse": None,
                "expr": None,
            },
            "final_validation": {
                "available": True,
                "source": "de_probe_trajectories",
                "metric": str(traj_metric),
                "probe_mse": float(score),
                "score_with_penalties": float(score_eff),
            },
        }

        rhs_ast = None
        residual_ast = None
        try:
            inner_nn = factorized_search_to_nestynet(expr_combo)
            rhs_ast = embed_mapping_in_ast(inner_nn, mapping_obj, input_exprs, units_mode="raw")
            if rhs_ast is not None:
                residual_ast = Add(anchor, Mul(ConstNode(-1.0), rhs_ast))
        except Exception:
            rhs_ast = None
            residual_ast = None

        compiled_ast_meta = _compiled_ast_metadata(rhs_ast, residual_ast)
        coeffs = [float(v) for v in sol.squeeze(-1).detach().cpu().tolist()]
        source_exprs = [str(terms[i]["expr"]) for i in state]
        periodic_matches: list[dict[str, Any]] = []
        periodic_matches.extend(_periodic_matches(expr_combo))
        for source_expr in source_exprs:
            periodic_matches.extend(_periodic_matches(source_expr))
        periodic_matches = _dedupe_periodic_matches(periodic_matches)
        row_out = {
            "order": int(order),
            "seed_search": -1,
            "expr": expr_str,
            "expr_ast": _to_jsonable(expr_combo),
            "mse": float(score),
            "score_raw": float(score),
            "score": float(score_eff),
            "mse_pooled": float(pooled_probe_mse),
            "raw_mse": float(pooled_probe_mse),
            "final_validated_mse": float(score),
            "compiled_structural_mse": compiled_mapped_probe_mse,
            "mse_traj": traj_rows,
            "size": size,
            "mapping_complexity": int(mapping_cplx),
            "mapping": mapping_json,
            "mapping_kind": str(mapping_json.get("kind", "")) if isinstance(mapping_json, dict) else "",
            "score_ladder": score_ladder,
            "acceptance_basis": "basis_state_combo",
            "final_acceptance_basis": "de_probe_validation",
            **compiled_ast_meta,
            "construction": "basis_state_combo",
            "basis_state_search": True,
            "combo_source_exprs": source_exprs,
            "combo_coeffs": coeffs,
            "combo_basis_indices": [int(terms[i]["pool_index"]) for i in state],
            "combo_state_indices": [int(i) for i in state],
            "combo_n_terms": int(len(state)),
            "combo_beam": int(beam),
            "combo_backward_prune": bool(backward_prune),
            "combo_max_terms": int(max_terms),
            "combo_mapping_mode": "affine_only",
            "combo_design_condition": design_diag.get("condition", None),
            "combo_rank_ratio": design_diag.get("rank_ratio", None),
            "combo_condition_penalty": float(condition_penalty),
            "combo_coeff_stability": _to_jsonable(coeff_stability),
            "combo_coeff_stability_penalty": float(coeff_stability_extra),
            "_expr_obj": expr_combo,
            "_mapping_obj": mapping_obj,
        }
        if periodic_matches:
            row_out["combo_contains_periodic"] = True
            row_out["combo_periodic_matches"] = periodic_matches
            row_out["combo_periodic_source_exprs"] = [
                expr for expr in source_exprs if _periodic_matches(expr)
            ]
        return row_out

    def _prune_state(indices: Sequence[int]) -> tuple[tuple[int, ...], dict[str, Any] | None]:
        state = _state_key(indices)
        row = _fit_state(state)
        if row is None:
            return state, None
        if not backward_prune:
            return state, row

        original_n = len(state)
        best_state = state
        best_row = row
        improved = True
        while improved and len(best_state) > 2:
            improved = False
            for drop_i in best_state:
                trial = tuple(i for i in best_state if i != drop_i)
                trial_row = _fit_state(trial)
                if trial_row is None:
                    continue
                s_trial = float(trial_row.get("score", float("inf")))
                s_best = float(best_row.get("score", float("inf")))
                if (s_trial < s_best - 1.0e-15) or (s_trial <= s_best * (1.0 + 1.0e-9)):
                    best_state = trial
                    best_row = trial_row
                    improved = True
                    break
        if len(best_state) != original_n:
            best_row["combo_pruned_from_n_terms"] = int(original_n)
        return best_state, best_row

    best_by_expr: dict[str, dict[str, Any]] = {}
    frontier = [(i,) for i in range(len(terms))]
    seen_states: set[tuple[int, ...]] = set(frontier)

    for _depth in range(2, max_terms + 1):
        candidates: list[tuple[tuple[int, ...], dict[str, Any]]] = []
        for state in frontier:
            if len(state) >= max_terms:
                continue
            for nxt in range(len(terms)):
                if nxt in state:
                    continue
                new_state = _state_key((*state, nxt))
                if new_state in seen_states or len(new_state) > max_terms:
                    continue
                seen_states.add(new_state)
                pruned_state, row = _prune_state(new_state)
                if row is None:
                    continue
                candidates.append((pruned_state, row))

        if not candidates:
            break

        for _state, row in candidates:
            mapping_json = row.get("mapping", {})
            try:
                mapping_key = json.dumps(mapping_json, sort_keys=True, default=str)
            except Exception:
                mapping_key = str(mapping_json)
            key = f"{row.get('expr', '')}\n{mapping_key}"
            prev = best_by_expr.get(key)
            if prev is None or float(row.get("score", float("inf"))) < float(prev.get("score", float("inf"))):
                best_by_expr[key] = row

        candidates.sort(
            key=lambda item: (
                float(item[1].get("score", float("inf"))),
                int(item[1].get("size", 10**9)),
                int(len(item[0])),
                str(item[1].get("expr", "")),
            )
        )
        next_frontier: list[tuple[int, ...]] = []
        for state, row in candidates:
            if len(next_frontier) >= beam:
                break
            if len(state) < max_terms and state not in next_frontier:
                next_frontier.append(state)
        if not next_frontier:
            break
        frontier = next_frontier

    rows = list(best_by_expr.values())
    rows.sort(
        key=lambda row: (
            float(row.get("score", float("inf"))),
            int(row.get("size", 10**9)),
            int(row.get("combo_n_terms", 10**9)),
            str(row.get("expr", "")),
        )
    )
    return _finish(rows)

def _build_multi_tables_for_order(
    spec: DELabSpec,
    trajectories: Sequence[_Trajectory],
    *,
    order: int,
    hp: FactorizedSearchConfig,
    seed: int,
    dtype: torch.dtype,
) -> _OrderTables:
    if not trajectories:
        raise ValueError("no trajectories provided")

    m_total = len(trajectories)
    mode = str(spec.split_mode).strip().lower() or "per_traj_point"
    if mode not in ("per_traj_point", "traj_holdout"):
        raise ValueError(f"unsupported split_mode {spec.split_mode!r}")

    fit_trajs: list[_Trajectory]
    probe_trajs: list[_Trajectory]
    if len(spec.fit_trajectories) > 0 or len(spec.probe_trajectories) > 0:
        by_id = {str(tr.traj_id): tr for tr in trajectories}
        by_path = {str(pathlib.Path(tr.path).resolve()): tr for tr in trajectories}

        def _pick(ref: TrajectoryRef, *, where: str) -> _Trajectory:
            tr = by_id.get(str(ref.id), None)
            ref_path = str(pathlib.Path(ref.csv).resolve())
            if tr is None:
                tr = by_path.get(ref_path, None)
            if tr is None:
                raise ValueError(
                    f"{where}: unknown trajectory id={ref.id!r} csv={ref.csv!r} "
                    f"(available ids={sorted(by_id.keys())})"
                )
            tr_path = str(pathlib.Path(tr.path).resolve())
            if tr_path != ref_path:
                raise ValueError(
                    f"{where}: trajectory id {ref.id!r} path mismatch "
                    f"(spec={ref_path!r}, loaded={tr_path!r})"
                )
            return tr

        fit_trajs = [_pick(ref, where="fit_trajectories") for ref in spec.fit_trajectories]
        if len(spec.probe_trajectories) > 0:
            probe_trajs = [_pick(ref, where="probe_trajectories") for ref in spec.probe_trajectories]
        else:
            # Explicit fit-only specs should behave like per-trajectory point splits.
            probe_trajs = list(fit_trajs)
        mode = "traj_holdout" if len(spec.probe_trajectories) > 0 else "per_traj_point"
    elif mode == "traj_holdout" and m_total >= 2:
        perm = torch.randperm(
            m_total,
            generator=torch.Generator(device="cpu").manual_seed(int(seed) + int(order) * 17_003),
        ).tolist()
        n_probe_traj = max(1, int(m_total) // 3)
        n_fit_traj = max(1, int(m_total) - int(n_probe_traj))
        fit_idx = set(perm[:n_fit_traj])
        probe_idx = set(perm[n_fit_traj:])
        if not probe_idx:
            probe_idx = {perm[-1]}
            fit_idx = set(perm[:-1]) if m_total > 1 else {perm[0]}
        fit_trajs = [trajectories[i] for i in range(m_total) if i in fit_idx]
        probe_trajs = [trajectories[i] for i in range(m_total) if i in probe_idx]
    else:
        fit_trajs = list(trajectories)
        probe_trajs = list(trajectories)

    fit_n_traj = max(1, len(fit_trajs))
    probe_n_traj = max(1, len(probe_trajs))
    n_fit_per = max(16, int(hp.n_fit) // fit_n_traj)
    n_probe_per = max(16, int(hp.n_probe) // probe_n_traj)

    x_fit_parts: list[torch.Tensor] = []
    y_fit_parts: list[torch.Tensor] = []
    x_probe_parts: list[torch.Tensor] = []
    y_probe_parts: list[torch.Tensor] = []
    fit_meta: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    probe_meta: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    feat_names: list[str] | None = None

    if mode == "per_traj_point":
        for i, tr in enumerate(fit_trajs):
            z_np, y_np, names = _build_table_for_order(spec, tr, order=int(order))
            if feat_names is None:
                feat_names = names
            z_t = torch.as_tensor(z_np, dtype=dtype)
            y_t = torch.as_tensor(y_np, dtype=dtype).reshape(-1, 1)
            z_fit, y_fit, z_probe, y_probe = _split_table_disjoint(
                z_t,
                y_t,
                n_fit_take=n_fit_per,
                n_probe_take=n_probe_per,
                seed=int(seed) + int(order) * 151_217 + i * 43_891,
            )
            x_fit_parts.append(z_fit)
            y_fit_parts.append(y_fit)
            fit_meta.append((str(tr.traj_id), z_fit, y_fit))
            x_probe_parts.append(z_probe)
            y_probe_parts.append(y_probe)
            probe_meta.append((str(tr.traj_id), z_probe, y_probe))
    else:
        for i, tr in enumerate(fit_trajs):
            z_np, y_np, names = _build_table_for_order(spec, tr, order=int(order))
            if feat_names is None:
                feat_names = names
            z_t = torch.as_tensor(z_np, dtype=dtype)
            y_t = torch.as_tensor(y_np, dtype=dtype).reshape(-1, 1)
            z_s, y_s = _subsample_table(
                z_t,
                y_t,
                n_take=n_fit_per,
                seed=int(seed) + int(order) * 100_003 + i * 31_337,
            )
            x_fit_parts.append(z_s)
            y_fit_parts.append(y_s)
            fit_meta.append((str(tr.traj_id), z_s, y_s))

        for i, tr in enumerate(probe_trajs):
            z_np, y_np, names = _build_table_for_order(spec, tr, order=int(order))
            if feat_names is None:
                feat_names = names
            z_t = torch.as_tensor(z_np, dtype=dtype)
            y_t = torch.as_tensor(y_np, dtype=dtype).reshape(-1, 1)
            z_s, y_s = _subsample_table(
                z_t,
                y_t,
                n_take=n_probe_per,
                seed=int(seed) + int(order) * 200_003 + i * 41_321 + 1,
            )
            x_probe_parts.append(z_s)
            y_probe_parts.append(y_s)
            probe_meta.append((str(tr.traj_id), z_s, y_s))

    if not x_fit_parts or not x_probe_parts:
        raise ValueError(f"order={order}: failed to build fit/probe tables from trajectories")

    return _OrderTables(
        x_fit=torch.cat(x_fit_parts, dim=0),
        y_fit=torch.cat(y_fit_parts, dim=0),
        x_probe=torch.cat(x_probe_parts, dim=0),
        y_probe=torch.cat(y_probe_parts, dim=0),
        feature_names=list(feat_names or []),
        fit_traj_ids=[str(tr.traj_id) for tr in fit_trajs],
        probe_traj_ids=[str(tr.traj_id) for tr in probe_trajs],
        fit_meta=fit_meta,
        probe_meta=probe_meta,
    )


def _ensure_col(v: torch.Tensor, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(v):
        raise TypeError(f"{name} must be a torch.Tensor")
    t = v.to(dtype=dtype)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    if t.ndim != 2 or int(t.shape[1]) != 1:
        raise ValueError(f"{name} must have shape (N,1) or (N,), got {tuple(t.shape)}")
    return t


def _validate_feature_tensors(features: DEFeatureTensors, *, dtype: torch.dtype) -> DEFeatureTensors:
    x_fit = features.x_fit.to(dtype=dtype)
    x_probe = features.x_probe.to(dtype=dtype)
    if x_fit.ndim != 2:
        raise ValueError(f"x_fit must be rank-2, got shape={tuple(x_fit.shape)}")
    if x_probe.ndim != 2:
        raise ValueError(f"x_probe must be rank-2, got shape={tuple(x_probe.shape)}")

    u_fit = _ensure_col(features.u_fit, name="u_fit", dtype=dtype)
    du_fit = _ensure_col(features.du_fit, name="du_fit", dtype=dtype)
    d2u_fit = _ensure_col(features.d2u_fit, name="d2u_fit", dtype=dtype)
    u_probe = _ensure_col(features.u_probe, name="u_probe", dtype=dtype)
    du_probe = _ensure_col(features.du_probe, name="du_probe", dtype=dtype)
    d2u_probe = _ensure_col(features.d2u_probe, name="d2u_probe", dtype=dtype)

    n_fit = int(x_fit.shape[0])
    n_probe = int(x_probe.shape[0])
    if n_fit != int(u_fit.shape[0]) or n_fit != int(du_fit.shape[0]) or n_fit != int(d2u_fit.shape[0]):
        raise ValueError("fit tensor length mismatch")
    if n_probe != int(u_probe.shape[0]) or n_probe != int(du_probe.shape[0]) or n_probe != int(d2u_probe.shape[0]):
        raise ValueError("probe tensor length mismatch")

    return DEFeatureTensors(
        x_fit=x_fit,
        u_fit=u_fit,
        du_fit=du_fit,
        d2u_fit=d2u_fit,
        x_probe=x_probe,
        u_probe=u_probe,
        du_probe=du_probe,
        d2u_probe=d2u_probe,
    )


def _build_table_from_features(
    spec: DELabSpec,
    features: DEFeatureTensors,
    *,
    order: int,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    split_l = str(split).lower().strip()
    if split_l not in ("fit", "probe"):
        raise ValueError(f"split must be 'fit' or 'probe', got {split!r}")

    if split_l == "fit":
        x_raw, u_raw, du_raw, d2u_raw = (
            features.x_fit,
            features.u_fit,
            features.du_fit,
            features.d2u_fit,
        )
    else:
        x_raw, u_raw, du_raw, d2u_raw = (
            features.x_probe,
            features.u_probe,
            features.du_probe,
            features.d2u_probe,
        )

    if int(spec.x_axis) < 0 or int(spec.x_axis) >= int(x_raw.shape[1]):
        raise ValueError(
            f"x_axis={int(spec.x_axis)} out of bounds for {split_l} x shape={tuple(x_raw.shape)}"
        )

    feat_cols: list[torch.Tensor] = []
    if spec.include_x:
        feat_cols.append(x_raw[:, int(spec.x_axis) : int(spec.x_axis) + 1])
    if spec.include_u:
        feat_cols.append(u_raw)
    # Fairness guard: never feed du into order-1 target discovery.
    if int(order) == 2 and spec.include_du:
        feat_cols.append(du_raw)
    for c in spec.constants:
        feat_cols.append(
            torch.full(
                (int(x_raw.shape[0]), 1),
                float(c.value),
                dtype=x_raw.dtype,
                device=x_raw.device,
            )
        )

    if not feat_cols:
        raise ValueError(f"order={order}: empty feature set")

    y = du_raw if int(order) == 1 else d2u_raw
    x_tbl = torch.cat(feat_cols, dim=1)
    m = torch.isfinite(y.reshape(-1)) & torch.isfinite(x_tbl).all(dim=1)
    if int(m.sum()) < 8:
        raise ValueError(f"order={order}/{split_l}: too few finite rows ({int(m.sum())})")
    return x_tbl[m], y[m], _feature_names(spec, order)


def _sample_indices(n_total: int, n_take: int, *, generator: torch.Generator) -> torch.Tensor:
    if n_total <= 0:
        raise ValueError(f"expected n_total > 0, got {n_total}")
    if n_take <= 0:
        raise ValueError(f"expected n_take > 0, got {n_take}")
    if n_take <= n_total:
        return torch.randperm(n_total, generator=generator)[:n_take]
    return torch.randint(low=0, high=n_total, size=(n_take,), generator=generator)


def _dims_for_order(spec: DELabSpec, order: int) -> tuple[list[tuple[float, ...]] | None, tuple[float, ...] | None]:
    if spec.dims is None:
        return None, None

    dim0 = dimless_dim(len(spec.dims.basis))
    canonical = CanonicalProblemDims.scalar(
        basis=tuple(spec.dims.basis),
        x_dim=tuple(spec.dims.x_dim),
        u_dim=tuple(spec.dims.u_dim),
        constant_dims={
            str(c.name): (dim0 if c.dim is None else tuple(float(v) for v in c.dim))
            for c in spec.constants
        },
    )
    return canonical_to_factorized_search_dims(
        canonical,
        order=int(order),
        x_axis=0,
        component_idx=0,
        include_x=bool(spec.include_x),
        include_u=bool(spec.include_u),
        include_du=bool(spec.include_du),
        constant_names=[str(c.name) for c in spec.constants],
    )


def _input_exprs_for_order(spec: DELabSpec, order: int) -> list[Any]:
    xs: list[Any] = []
    if spec.include_x:
        xs.append(Var(int(spec.x_axis)))
    if spec.include_u:
        xs.append(U())
    if int(order) == 2 and spec.include_du:
        xs.append(DU(int(spec.x_axis)))
    for c in spec.constants:
        xs.append(ConstNode(float(c.value)))
    return xs


def _anchor_for_order(spec: DELabSpec, order: int):
    xa = int(spec.x_axis)
    if int(order) == 1:
        return DU(xa)
    return D2U(xa, xa)


def _domain_projection_cfg_from_hp(hp: Any) -> dict[str, Any]:
    return {
        "score_domain_projection_enable": bool(getattr(hp, "score_domain_projection_enable", False)),
        "score_domain_projection_abs_tol": float(getattr(hp, "score_domain_projection_abs_tol", 1.0e-8)),
        "score_domain_projection_rel_tol": float(getattr(hp, "score_domain_projection_rel_tol", 1.0e-8)),
        "score_domain_projection_max_frac": float(getattr(hp, "score_domain_projection_max_frac", 1.0)),
        "score_domain_projection_positive_floor": float(
            getattr(hp, "score_domain_projection_positive_floor", 1.0e-12)
        ),
    }


def _arch_best_records(arch: Any, limit: int, *, strategy: str = "mse") -> list[Any]:
    try:
        return list(arch.best(int(limit), strategy=str(strategy)))
    except TypeError:
        return list(arch.best(int(limit)))


def _score_decade(score: Any) -> int:
    try:
        score_f = float(score)
    except Exception:
        return 10**9
    if not math.isfinite(score_f):
        return 10**9
    return int(math.floor(math.log10(max(float(score_f), 1.0e-300))))


def _row_validation_decade(row: Mapping[str, Any]) -> int:
    for key in ("mse", "probe_mse", "score_raw", "final_validated_mse", "raw_mse", "score"):
        if key not in row:
            continue
        decade = _score_decade(row.get(key))
        if decade < 10**9:
            return int(decade)
    return 10**9


def _de_complexity_order_key(row: Mapping[str, Any]) -> tuple[int, int, int, float, int]:
    try:
        score = float(row.get("score", float("inf")))
    except Exception:
        score = float("inf")
    try:
        size = int(row.get("symbolic_size_simplified", row.get("size", 10**9)))
    except Exception:
        size = 10**9
    try:
        original_rank = int(row.get("original_rank", row.get("score_rank", 10**9)))
    except Exception:
        original_rank = 10**9
    return (_row_validation_decade(row), _score_decade(score), int(size), float(score), int(original_rank))


def _materialize_order_arch_rows(
    *,
    arch: Any,
    spec: DELabSpec,
    order: int,
    seed_search: int,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    hp: FactorizedSearchConfig,
    input_exprs: Sequence[Any],
    anchor: Any,
    probe_meta: Sequence[tuple[str, torch.Tensor, torch.Tensor]] | None,
    traj_metric: str,
    mapping_penalty: float,
    finite_mask_enabled: bool,
    finite_mask_min_dataset_frac: float,
    finite_mask_min_points: int,
    domain_projection_cfg: Mapping[str, Any],
    limit: int,
    best_strategy: str = "mse",
) -> tuple[list[dict[str, Any]], float, int]:
    rows: list[dict[str, Any]] = []
    seed_best_score = float("inf")
    hidden_score_head_skipped = 0

    for rec in _arch_best_records(arch, max(1, int(limit)), strategy=best_strategy):
        mapping_obj = rec.mapping
        hidden_score_head = _has_hidden_score_head(mapping_obj)
        if hidden_score_head and not bool(getattr(hp, "de_accept_hidden_score_head", False)):
            hidden_score_head_skipped += 1
            continue
        mapping_json = _to_jsonable(mapping_obj)
        mapping_cplx = _mapping_complexity(mapping_json)

        traj_rows: list[dict[str, Any]] = []
        if probe_meta is not None and len(probe_meta) > 0:
            for tid, zp, yp in probe_meta:
                mse_i = float("inf")
                valid_frac_i = 0.0
                n_valid_i = 0
                n_probe_i = int(yp.shape[0])
                domain_diag_i = None
                try:
                    yhat_i = None
                    joint_terms = mapping_obj.get("_joint_linear_terms") if isinstance(mapping_obj, dict) else None
                    if isinstance(joint_terms, dict):
                        terms = joint_terms.get("terms", None)
                        coeffs = None
                        for ds in (joint_terms.get("datasets", []) or []):
                            if str(ds.get("id")) == str(tid):
                                coeffs = ds.get("coeffs")
                                break
                        if isinstance(terms, (list, tuple)) and isinstance(coeffs, (list, tuple)):
                            if len(terms) >= 1 and len(coeffs) == 1 + len(terms):
                                cols = []
                                term_diags = []
                                ok = True
                                for term in terms:
                                    v, dom_t = eval_node_with_domain_projection(term, zp, domain_projection_cfg)
                                    dom_t = dict(dom_t) if isinstance(dom_t, dict) else dom_t
                                    if not domain_projection_is_acceptable(dom_t):
                                        ok = False
                                        break
                                    if (v is None) or (not torch.isfinite(v).all()):
                                        ok = False
                                        break
                                    term_diags.append(dom_t)
                                    cols.append(v.reshape(-1, 1))
                                if ok and cols:
                                    Phi = torch.cat([torch.ones_like(cols[0]), *cols], dim=1)
                                    sol = torch.as_tensor(
                                        [float(c) for c in coeffs],
                                        dtype=Phi.dtype,
                                        device=Phi.device,
                                    ).reshape(-1, 1)
                                    yhat_i = Phi @ sol
                                    domain_diag_i = merge_domain_projection_diagnostics(*term_diags)
                    if yhat_i is None:
                        pred_i, domain_diag_i = eval_node_with_domain_projection(
                            rec.best_expr,
                            zp,
                            domain_projection_cfg,
                        )
                        if not domain_projection_is_acceptable(domain_diag_i):
                            raise ValueError("domain_projection_rejected")
                        mapping_i = mapping_obj
                        joint_aff = mapping_obj.get("_joint_affine") if isinstance(mapping_obj, dict) else None
                        if isinstance(joint_aff, dict):
                            for ds in (joint_aff.get("datasets", []) or []):
                                if str(ds.get("id")) == str(tid):
                                    mi = ds.get("mapping")
                                    if isinstance(mi, dict):
                                        mapping_i = mi
                                    break
                        yhat_i = eval_mapping(pred_i, mapping_i).reshape(-1, 1)
                    yhat_i = yhat_i.reshape_as(yp)
                    valid_mask_i = torch.isfinite(yhat_i).reshape(-1) & torch.isfinite(yp).reshape(-1)
                    n_valid_i = int(valid_mask_i.sum().detach().cpu().item())
                    n_total_i = int(valid_mask_i.numel())
                    valid_frac_i = 0.0 if n_total_i <= 0 else float(n_valid_i) / float(n_total_i)
                    if bool(finite_mask_enabled):
                        if n_valid_i >= int(finite_mask_min_points) and valid_frac_i >= float(finite_mask_min_dataset_frac):
                            r_i = yhat_i.reshape(-1)[valid_mask_i] - yp.reshape(-1)[valid_mask_i]
                            mse_i = float(torch.mean(r_i * r_i).detach().cpu().item())
                        else:
                            mse_i = float("inf")
                    elif bool(torch.all(valid_mask_i).detach().cpu().item()):
                        mse_i = float(torch.mean((yhat_i - yp) ** 2).detach().cpu().item())
                    if not math.isfinite(mse_i):
                        mse_i = float("inf")
                except Exception:
                    mse_i = float("inf")
                traj_rows.append(
                    {
                        "traj_id": str(tid),
                        "mse": float(mse_i),
                        "n_probe": int(n_probe_i),
                        "n_valid": int(n_valid_i),
                        "valid_frac": float(valid_frac_i),
                        "finite_mask_enabled": bool(finite_mask_enabled),
                        "domain_projection": (
                            dict(domain_diag_i)
                            if isinstance(domain_diag_i, Mapping)
                            and bool(domain_diag_i.get("enabled", False))
                            else None
                        ),
                    }
                )
            score = _aggregate_score([r["mse"] for r in traj_rows], metric=traj_metric)
        else:
            score = float(rec.best_mse)

        score_eff = float(score + mapping_penalty * float(mapping_cplx))
        seed_best_score = min(seed_best_score, float(score_eff))
        score_ladder = None
        acceptance_basis = ""
        finite_mask_diag = None
        domain_projection_diag = None
        if isinstance(mapping_json, dict):
            acceptance_basis = str(mapping_json.get("_acceptance_basis", ""))
            raw_finite_mask = mapping_json.get("_finite_mask", None)
            if isinstance(raw_finite_mask, dict):
                finite_mask_diag = dict(raw_finite_mask)
            raw_domain_projection = mapping_json.get("_domain_projection", None)
            if isinstance(raw_domain_projection, dict):
                domain_projection_diag = dict(raw_domain_projection)
            raw_ladder = mapping_json.get("_score_ladder", None)
            if isinstance(raw_ladder, dict):
                score_ladder = dict(raw_ladder)
                score_ladder["final_validation"] = {
                    "available": True,
                    "source": "de_probe_trajectories",
                    "metric": str(traj_metric),
                    "probe_mse": float(score),
                    "score_with_mapping_penalty": float(score_eff),
                }

        rhs_ast = None
        residual_ast = None
        try:
            inner_nn = factorized_search_to_nestynet(rec.best_expr)
            rhs_ast = embed_mapping_in_ast(inner_nn, mapping_obj, list(input_exprs), units_mode="raw")
            if rhs_ast is not None:
                residual_ast = Add(anchor, Mul(ConstNode(-1.0), rhs_ast))
        except Exception:
            rhs_ast = None
            residual_ast = None
        compiled_ast_meta = _compiled_ast_metadata(rhs_ast, residual_ast)

        rows.append(
            {
                "order": int(order),
                "seed_search": int(seed_search),
                "expr": node_str(rec.best_expr),
                "expr_ast": _to_jsonable(rec.best_expr),
                "mse": float(score),
                "score_raw": float(score),
                "score": float(score_eff),
                "mse_pooled": float(rec.best_mse),
                "raw_mse": float(getattr(rec, "best_raw_mse", rec.best_mse)),
                "final_validated_mse": float(score),
                "mse_traj": traj_rows,
                "size": int(node_size(rec.best_expr)),
                "mapping_complexity": int(mapping_cplx),
                "mapping": mapping_json,
                "mapping_kind": str(mapping_json.get("kind", "")) if isinstance(mapping_json, dict) else "",
                "finite_mask": finite_mask_diag,
                "domain_projection": domain_projection_diag,
                "score_ladder": score_ladder,
                "hidden_score_head": bool(hidden_score_head),
                "acceptance_basis": acceptance_basis,
                "final_acceptance_basis": "de_probe_validation",
                **compiled_ast_meta,
                "_expr_obj": rec.best_expr,
                "_mapping_obj": mapping_obj,
            }
        )

    return rows, float(seed_best_score), int(hidden_score_head_skipped)


def _pool_order_rows(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    pooled: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["expr"]), str(row.get("mapping_kind", "")))
        prev = pooled.get(key)
        if prev is None or float(row["score"]) < float(prev["score"]):
            pooled[key] = row
    return pooled


def _compiled_ast_metadata(rhs_ast: Any | None, residual_ast: Any | None) -> dict[str, Any]:
    rhs_simplified = None
    residual_simplified = None
    try:
        rhs_simplified = simplify_ast(rhs_ast) if rhs_ast is not None else None
    except Exception:
        rhs_simplified = rhs_ast
    try:
        residual_simplified = simplify_ast(residual_ast) if residual_ast is not None else None
    except Exception:
        residual_simplified = residual_ast
    return {
        "rhs_mapped_ast_raw": None if rhs_ast is None else repr(rhs_ast),
        "residual_ast_raw": None if residual_ast is None else repr(residual_ast),
        "rhs_mapped_ast_simplified": None if rhs_simplified is None else repr(rhs_simplified),
        "residual_ast_simplified": None if residual_simplified is None else repr(residual_simplified),
        "symbolic_size_raw": None if residual_ast is None else int(ast_node_count(residual_ast)),
        "symbolic_size_simplified": None
        if residual_simplified is None
        else int(ast_node_count(residual_simplified)),
        "rhs_mapped_ast": None if rhs_simplified is None else repr(rhs_simplified),
        "residual_ast": None if residual_simplified is None else repr(residual_simplified),
    }


def _trajectory_projection_reference_scale(spec: DELabSpec, order: int, tr: _Trajectory) -> float:
    vals: list[np.ndarray] = []
    if spec.include_u:
        vals.append(np.asarray(tr.u, dtype=np.float64).reshape(-1))
    if int(order) == 2 and spec.include_du:
        vals.append(np.asarray(tr.du, dtype=np.float64).reshape(-1))
    if not vals and spec.include_x:
        vals.append(np.asarray(tr.x, dtype=np.float64).reshape(-1))
    if not vals:
        const_vals = [float(c.value) for c in spec.constants if math.isfinite(float(c.value))]
        if const_vals:
            vals.append(np.asarray(const_vals, dtype=np.float64).reshape(-1))
    if not vals:
        return 1.0
    arr = np.concatenate(vals)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return 1.0
    abs_arr = np.abs(arr)
    return float(max(1.0e-12, float(np.mean(abs_arr)), 0.1 * float(np.max(abs_arr)), float(np.std(arr))))


def _domain_projection_cfg_for_trajectory(
    domain_projection_cfg: Mapping[str, Any] | None,
    *,
    spec: DELabSpec,
    order: int,
    tr: _Trajectory,
) -> Mapping[str, Any] | None:
    if not (
        isinstance(domain_projection_cfg, Mapping)
        and bool(domain_projection_cfg.get("score_domain_projection_enable", False))
    ):
        return domain_projection_cfg
    out = dict(domain_projection_cfg)
    out["score_domain_projection_reference_scale"] = _trajectory_projection_reference_scale(spec, order, tr)
    return out


def _candidate_rhs_scalar(
    expr_ast: Any,
    mapping: dict[str, Any],
    features: list[float],
    *,
    dtype: torch.dtype,
    domain_projection_cfg: Mapping[str, Any] | None = None,
) -> float:
    x = torch.tensor([features], dtype=dtype)
    if isinstance(domain_projection_cfg, Mapping) and bool(
        domain_projection_cfg.get("score_domain_projection_enable", False)
    ):
        pred, domain_diag = eval_node_with_domain_projection(expr_ast, x, domain_projection_cfg)
        if not domain_projection_is_acceptable(domain_diag):
            raise ValueError("domain_projection_rejected")
    else:
        pred = eval_node(expr_ast, x)
    y_hat = eval_mapping(pred, mapping)
    val = float(y_hat.reshape(-1)[0].detach().cpu().item())
    if not math.isfinite(val):
        raise ValueError("non-finite candidate RHS value")
    return val


def _validate_candidate_by_integration(
    *,
    spec: DELabSpec,
    order: int,
    expr_ast: Any,
    mapping: dict[str, Any],
    trajectories: Sequence[_Trajectory],
    dtype: torch.dtype,
    domain_projection_cfg: Mapping[str, Any] | None = None,
) -> float:
    try:
        from scipy.integrate import solve_ivp
    except Exception:
        return float("inf")

    mses: list[float] = []

    for tr in trajectories:
        x_obs = np.asarray(tr.x, dtype=np.float64)
        u_obs = np.asarray(tr.u, dtype=np.float64)
        if x_obs.size < 5:
            continue
        domain_projection_cfg_tr = _domain_projection_cfg_for_trajectory(
            domain_projection_cfg,
            spec=spec,
            order=order,
            tr=tr,
        )

        def _features(t: float, u: float, du: float | None) -> list[float]:
            vals: list[float] = []
            if spec.include_x:
                vals.append(float(t))
            if spec.include_u:
                vals.append(float(u))
            if int(order) == 2 and spec.include_du:
                vals.append(0.0 if du is None else float(du))
            for c in spec.constants:
                vals.append(float(c.value))
            return vals

        if int(order) == 1:
            u0 = float(u_obs[0])

            def _rhs_sys(t: float, s: np.ndarray) -> list[float]:
                rhs = _candidate_rhs_scalar(
                    expr_ast,
                    mapping,
                    _features(float(t), float(s[0]), None),
                    dtype=dtype,
                    domain_projection_cfg=domain_projection_cfg_tr,
                )
                return [rhs]

            y0 = [u0]
        else:
            u0 = float(u_obs[0])
            du0 = float(tr.du[0])

            def _rhs_sys(t: float, s: np.ndarray) -> list[float]:
                u_val = float(s[0])
                du_val = float(s[1])
                rhs = _candidate_rhs_scalar(
                    expr_ast,
                    mapping,
                    _features(float(t), u_val, du_val),
                    dtype=dtype,
                    domain_projection_cfg=domain_projection_cfg_tr,
                )
                return [du_val, rhs]

            y0 = [u0, du0]

        try:
            sol = solve_ivp(
                _rhs_sys,
                [float(x_obs[0]), float(x_obs[-1])],
                y0,
                t_eval=x_obs,
                method="RK45",
                rtol=1.0e-7,
                atol=1.0e-9,
            )
            if int(sol.status) != 0:
                mses.append(float("inf"))
                continue
            u_hat = np.asarray(sol.y[0], dtype=np.float64)
            mse = float(np.mean((u_hat - u_obs) ** 2))
            mses.append(mse if math.isfinite(mse) else float("inf"))
        except Exception:
            mses.append(float("inf"))

    if not mses:
        return float("inf")
    return float(sum(mses) / len(mses))


def default_oracle_de_hyperparams() -> FactorizedSearchConfig:
    """Return factorized symbolic search hyperparameters tuned for lightweight DE-oracle experiments."""

    hp = FactorizedSearchConfig()
    hp.n_seeds = 1
    hp.split_iter_across_seeds = True
    hp.score_head_enable = False
    hp.de_score_head_policy = "proposal_only"
    hp.de_accept_hidden_score_head = False
    hp.de_score_head_untyped_enable = False
    hp.score_pade_structural_enable = True
    return hp


def _run_order_search(
    *,
    spec: DELabSpec,
    order: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    hp: FactorizedSearchConfig,
    dtype: torch.dtype,
    run_seed: int,
    var_dims: list[tuple[float, ...]] | None,
    y_dims: tuple[float, ...] | None,
    verbose: bool,
    fit_meta: Sequence[tuple[str, torch.Tensor, torch.Tensor]] | None = None,
    probe_meta: Sequence[tuple[str, torch.Tensor, torch.Tensor]] | None = None,
    traj_metric: str = "mean",
    stop_event: threading.Event | None = None,
    diagnostics_out: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int, int, float]:
    nvars = int(x_fit.shape[1])
    n_seeds = max(1, int(hp.n_seeds))
    if bool(hp.split_iter_across_seeds) and n_seeds > 1:
        n_iter_each = max(1, int(hp.n_iter) // n_seeds)
    else:
        n_iter_each = int(hp.n_iter)

    early_stop_mse = float(hp.early_stop_mse)
    mapping_penalty = float(getattr(hp, "mapping_complexity_penalty", 0.0))
    raw_rows: list[dict[str, Any]] = []
    n_seeds_ran = 0
    refine_diagnostics_total: dict[str, Any] = {}
    refine_diagnostics_by_seed: list[dict[str, Any]] = []
    refine_slate_stats_by_seed: list[dict[str, Any]] = []
    hidden_score_head_skipped = 0
    input_exprs = _input_exprs_for_order(spec, order)
    anchor = _anchor_for_order(spec, order)
    finite_mask_enabled = bool(getattr(hp, "score_finite_mask_enable", False))
    finite_mask_min_dataset_frac = float(getattr(hp, "score_finite_mask_min_dataset_frac", 0.95))
    if not math.isfinite(finite_mask_min_dataset_frac):
        finite_mask_min_dataset_frac = 0.95
    finite_mask_min_dataset_frac = min(1.0, max(0.0, finite_mask_min_dataset_frac))
    finite_mask_min_points = max(1, int(getattr(hp, "score_finite_mask_min_points", 8)))
    domain_projection_cfg = _domain_projection_cfg_from_hp(hp)
    additive_diag: dict[str, Any] = {
        "enabled": bool(getattr(hp, "de_sparse_combo_enable", False)),
        "pre_mutation_enabled": bool(getattr(hp, "de_sparse_combo_pre_mutation_enable", True)),
        "pre_mutation_attempted": False,
        "pre_mutation_early_return": False,
        "pre_mutation_seed_rows": 0,
        "pre_mutation_combo_rows": 0,
        "pre_mutation_context_rows": 0,
        "pre_mutation_best_score": None,
        "post_mutation_combo_rows": 0,
        "post_mutation_context_rows": 0,
    }

    def _finalize_diagnostics() -> None:
        if diagnostics_out is not None:
            diagnostics_out.update(
                {
                    "refine_diagnostics": _to_jsonable(refine_diagnostics_total),
                    "refine_cost_summary": _to_jsonable(
                        _refine_diagnostics_summary(refine_diagnostics_total)
                    ),
                    "refine_diagnostics_by_seed": _to_jsonable(refine_diagnostics_by_seed),
                    "refine_slate_stats_by_seed": _to_jsonable(refine_slate_stats_by_seed),
                    "hidden_score_head_skipped": int(hidden_score_head_skipped),
                    "de_accept_hidden_score_head": bool(getattr(hp, "de_accept_hidden_score_head", False)),
                    "de_score_head_policy": str(getattr(hp, "de_score_head_policy", "proposal_only")),
                    "additive_fss": _to_jsonable(additive_diag),
                }
            )

    for i in range(n_seeds):
        if stop_event is not None and stop_event.is_set():
            break
        n_seeds_ran = int(i) + 1
        seed_search = run_seed + i
        core_kwargs = dict(
            target_fn=lambda _x: (_x[:, :1] * float("nan")),
            nvars=nvars,
            n_iter=n_iter_each,
            max_depth=int(hp.max_depth),
            poly_degree=int(hp.poly_degree),
            lo=0.0,
            hi=1.0,
            seed=run_seed,
            seed_search=seed_search,
            var_dims=var_dims,
            y_dims=y_dims,
            dtype=dtype,
            x_fit_data=x_fit,
            y_fit_data=y_fit,
            x_probe_data=x_probe,
            y_probe_data=y_probe,
            brute_depth=hp.brute_depth,
            early_stop_mse=early_stop_mse,
            brute_max_expressions=int(hp.brute_max_expressions),
            boost_enable=bool(getattr(hp, "boost_enable", False)),
            boost_max_terms=int(getattr(hp, "boost_max_terms", 6)),
            boost_topk_try=int(getattr(hp, "boost_topk_try", 15)),
            boost_min_rel_improve=float(getattr(hp, "boost_min_rel_improve", 1.0e-3)),
            boost_selection_split=str(getattr(hp, "boost_selection_split", "fit")),
            boost_ridge=getattr(hp, "boost_ridge", None),
            boost_include_parent=bool(getattr(hp, "boost_include_parent", True)),
            boost_from_scratch_prob=float(getattr(hp, "boost_from_scratch_prob", 0.25)),
            boost_prune_rel=float(getattr(hp, "boost_prune_rel", 1.0e-10)),
            boost_safe_eval=bool(getattr(hp, "boost_safe_eval", True)),
            boost_harvest_enable=bool(getattr(hp, "boost_harvest_enable", False)),
            boost_harvest_every=int(getattr(hp, "boost_harvest_every", 500)),
            boost_harvest_topk_residual_basins=int(getattr(hp, "boost_harvest_topk_residual_basins", 50)),
            boost_harvest_elites_per_residual_basin=int(getattr(hp, "boost_harvest_elites_per_residual_basin", 2)),
            boost_pool_extra_max=int(getattr(hp, "boost_pool_extra_max", 256)),
            boost_subtree_depth_max=int(getattr(hp, "boost_subtree_depth_max", 3)),
            boost_subtree_size_max=int(getattr(hp, "boost_subtree_size_max", 12)),
            boost_gate_enable=bool(getattr(hp, "boost_gate_enable", True)),
            boost_gate_warmup=int(getattr(hp, "boost_gate_warmup", 200)),
            boost_gate_best_factor=float(getattr(hp, "boost_gate_best_factor", 30.0)),
            boost_gate_gain_frac=float(getattr(hp, "boost_gate_gain_frac", 1.0e-2)),
            boost_gate_peak_ratio=float(getattr(hp, "boost_gate_peak_ratio", 5.0)),
            boost_gate_min_valid=int(getattr(hp, "boost_gate_min_valid", 8)),
            boost_gate_min_residual_basins=int(getattr(hp, "boost_gate_min_residual_basins", 10)),
            boost_gate_adaptive=bool(getattr(hp, "boost_gate_adaptive", True)),
            boost_gate_adapt_quantile=float(getattr(hp, "boost_gate_adapt_quantile", 0.75)),
            boost_gate_adapt_window=int(getattr(hp, "boost_gate_adapt_window", 256)),
            boost_gate_adapt_min_samples=int(getattr(hp, "boost_gate_adapt_min_samples", 32)),
            boost_gate_adapt_mix=float(getattr(hp, "boost_gate_adapt_mix", 1.0)),
            boost_gate_gain_frac_floor=float(getattr(hp, "boost_gate_gain_frac_floor", 1.0e-4)),
            boost_gate_gain_frac_cap=float(getattr(hp, "boost_gate_gain_frac_cap", 0.25)),
            refine_enable=bool(hp.refine_enable),
            refine_mode=str(getattr(hp, "refine_mode", "slate")),
            refine_during_brute=bool(getattr(hp, "refine_during_brute", False)),
            refine_during_mutation=bool(getattr(hp, "refine_during_mutation", False)),
            refine_during_controller_slate=bool(getattr(hp, "refine_during_controller_slate", False)),
            refine_during_slate=bool(getattr(hp, "refine_during_slate", True)),
            refine_slate_after_brute=bool(getattr(hp, "refine_slate_after_brute", True)),
            refine_slate_period=int(getattr(hp, "refine_slate_period", 0)),
            refine_final_polish=bool(getattr(hp, "refine_final_polish", True)),
            refine_slate_k=int(getattr(hp, "refine_slate_k", 16)),
            refine_slate_diverse_k=int(getattr(hp, "refine_slate_diverse_k", 8)),
            periodic_seed_enable=bool(getattr(hp, "periodic_seed_enable", True)),
            periodic_seed_max_hints=int(getattr(hp, "periodic_seed_max_hints", 2)),
            periodic_seed_min_prominence=float(getattr(hp, "periodic_seed_min_prominence", 8.0)),
            refine_slate_budget=int(getattr(hp, "refine_slate_budget", 32)),
            refine_optimizer=str(getattr(hp, "refine_optimizer", "lbfgs")),
            refine_lbfgs_escalate_improve_factor=float(
                getattr(hp, "refine_lbfgs_escalate_improve_factor", 2.0)
            ),
            refine_lbfgs_steps=int(hp.refine_lbfgs_steps),
            refine_fit_subset=int(hp.refine_fit_subset),
            refine_joint_fit_data=list(fit_meta) if fit_meta is not None else None,
            refine_joint_probe_data=list(probe_meta) if probe_meta is not None else None,
            refine_joint_score_enable=bool(getattr(hp, "refine_joint_score_enable", True)),
            refine_joint_terms_enable=bool(getattr(hp, "refine_joint_terms_enable", False)),
            refine_joint_weight_mode=str(getattr(hp, "refine_joint_weight_mode", "points")),
            refine_joint_enable=bool(getattr(hp, "refine_joint_enable", True)),
            refine_num_restarts=int(hp.refine_num_restarts),
            refine_max_variants=int(hp.refine_max_variants),
            refine_max_params=int(hp.refine_max_params),
            refine_linear_combo_enable=bool(hp.refine_linear_combo_enable),
            refine_linear_terms_max=int(hp.refine_linear_terms_max),
            refine_linear_prune_rel=float(hp.refine_linear_prune_rel),
            refine_gate_best_factor=float(hp.refine_gate_best_factor),
            refine_max_trials=int(hp.refine_max_trials),
            refine_trials_per_brute_depth=int(hp.refine_trials_per_brute_depth),
            refine_trials_per_mutation_window=int(hp.refine_trials_per_mutation_window),
            refine_mutation_window=int(hp.refine_mutation_window),
            refine_safe_eps=float(hp.refine_safe_eps),
            refine_safe_penalty_weight=float(hp.refine_safe_penalty_weight),
            refine_safe_exp_clip=float(hp.refine_safe_exp_clip),
            refine_theta_l2=float(hp.refine_theta_l2),
            refine_init_log_min=float(hp.refine_init_log_min),
            refine_init_log_max=float(hp.refine_init_log_max),
            score_head_enable=bool(getattr(hp, "score_head_enable", True)),
            score_head_vars_enable=bool(getattr(hp, "score_head_vars_enable", True)),
            score_head_omp_enable=(
                bool(getattr(hp, "score_head_omp_enable", False))
                or bool(getattr(hp, "de_score_head_untyped_enable", False))
            ),
            score_head_omp_max_terms=int(
                getattr(
                    hp,
                    "de_score_head_max_terms",
                    getattr(hp, "score_head_omp_max_terms", 2),
                )
            ),
            score_head_omp_topk_try=int(getattr(hp, "score_head_omp_topk_try", 15)),
            score_head_ridge=getattr(hp, "score_head_ridge", None),
            score_head_min_rel_improve=float(getattr(hp, "score_head_min_rel_improve", 0.0)),
            score_head_untyped_enable=bool(getattr(hp, "de_score_head_untyped_enable", False)),
            score_mapping_family_mode=str(getattr(hp, "score_mapping_family_mode", "full")),
            brute_score_mapping_family_mode=str(
                getattr(hp, "brute_score_mapping_family_mode", "gated")
            ),
            score_pade_structural_enable=bool(
                getattr(hp, "score_pade_structural_enable", False)
            ),
            score_pade_structural_max_degree=int(
                getattr(hp, "score_pade_structural_max_degree", 2)
            ),
            score_pade_structural_max_total_degree=int(
                getattr(hp, "score_pade_structural_max_total_degree", 3)
            ),
            score_pade_structural_max_depth=int(
                getattr(hp, "score_pade_structural_max_depth", 8)
            ),
            score_pade_structural_max_size=int(
                getattr(hp, "score_pade_structural_max_size", 64)
            ),
            score_pade_structural_coeff_tol=float(
                getattr(hp, "score_pade_structural_coeff_tol", 1.0e-10)
            ),
            score_pade_structural_mse_rel_tol=float(
                getattr(hp, "score_pade_structural_mse_rel_tol", 1.0e-6)
            ),
            score_mapping_expensive_gate_best_factor=float(
                getattr(hp, "score_mapping_expensive_gate_best_factor", 5.0)
            ),
            score_mapping_expensive_rel_y=float(
                getattr(hp, "score_mapping_expensive_rel_y", 0.10)
            ),
            score_prescreen_enable=bool(getattr(hp, "score_prescreen_enable", True)),
            score_prescreen_family_mode=str(
                getattr(hp, "score_prescreen_family_mode", "cheap")
            ),
            score_prescreen_residual_family_mode=str(
                getattr(hp, "score_prescreen_residual_family_mode", "gated")
            ),
            score_prescreen_residual_allow_hint=bool(
                getattr(hp, "score_prescreen_residual_allow_hint", False)
            ),
            score_prescreen_residual_use_global_best=bool(
                getattr(hp, "score_prescreen_residual_use_global_best", False)
            ),
            score_prescreen_parent_best_factor=float(
                getattr(hp, "score_prescreen_parent_best_factor", 1.5)
            ),
            score_prescreen_global_best_factor=float(
                getattr(hp, "score_prescreen_global_best_factor", 3.0)
            ),
            score_prescreen_residual_parent_best_factor=float(
                getattr(hp, "score_prescreen_residual_parent_best_factor", 1.1)
            ),
            score_prescreen_residual_global_best_factor=float(
                getattr(hp, "score_prescreen_residual_global_best_factor", 1.5)
            ),
            score_finite_mask_enable=bool(getattr(hp, "score_finite_mask_enable", False)),
            score_finite_mask_min_fit_frac=float(getattr(hp, "score_finite_mask_min_fit_frac", 0.98)),
            score_finite_mask_min_probe_frac=float(getattr(hp, "score_finite_mask_min_probe_frac", 0.98)),
            score_finite_mask_min_dataset_frac=float(getattr(hp, "score_finite_mask_min_dataset_frac", 0.95)),
            score_finite_mask_min_points=int(getattr(hp, "score_finite_mask_min_points", 8)),
            score_domain_projection_enable=bool(getattr(hp, "score_domain_projection_enable", False)),
            score_domain_projection_abs_tol=float(getattr(hp, "score_domain_projection_abs_tol", 1.0e-8)),
            score_domain_projection_rel_tol=float(getattr(hp, "score_domain_projection_rel_tol", 1.0e-8)),
            score_domain_projection_max_frac=float(getattr(hp, "score_domain_projection_max_frac", 1.0)),
            score_domain_projection_positive_floor=float(getattr(hp, "score_domain_projection_positive_floor", 1.0e-12)),
            complexity_penalty=float(getattr(hp, "complexity_penalty", 0.0)),
            stall_window=int(hp.stall_window),
            stall_patience=int(hp.stall_patience),
            stall_delta=float(hp.stall_delta),
            plateau_stop_enable=bool(getattr(hp, "plateau_stop_enable", False)),
            plateau_stop_max_soft_restarts=int(getattr(hp, "plateau_stop_max_soft_restarts", 0)),
            plateau_stop_min_evals=int(getattr(hp, "plateau_stop_min_evals", 0)),
            degenerate_abort_enable=bool(getattr(hp, "degenerate_abort_enable", True)),
            degenerate_abort_min_evals=int(getattr(hp, "degenerate_abort_min_evals", 1000)),
            degenerate_abort_max_accepted=int(getattr(hp, "degenerate_abort_max_accepted", 8)),
            verbose=bool(verbose),
            stop_event=stop_event,
        )
        core_kwargs.setdefault(
            "_runtime_hooks",
            {**make_engine_runtime_hooks(), **make_engine_refinement_hooks()},
        )

        if bool(additive_diag["enabled"]) and bool(additive_diag["pre_mutation_enabled"]):
            additive_diag["pre_mutation_attempted"] = True
            try:
                pool_topk = int(getattr(hp, "de_sparse_combo_pool_topk", max(2, int(hp.return_topk))))
            except Exception:
                pool_topk = max(2, int(getattr(hp, "return_topk", 8)))
            pool_topk = max(2, pool_topk)
            scout_kwargs = dict(core_kwargs)
            scout_kwargs.update(
                {
                    "n_iter": 0,
                    "refine_during_mutation": False,
                    "refine_slate_period": 0,
                    "refine_final_polish": False,
                    "verbose": False,
                }
            )
            shallow_arch = run_explorer_core(**scout_kwargs)
            shallow_rows, shallow_best_score, shallow_hidden_skipped = _materialize_order_arch_rows(
                arch=shallow_arch,
                spec=spec,
                order=order,
                seed_search=seed_search,
                x_probe=x_probe,
                y_probe=y_probe,
                hp=hp,
                input_exprs=input_exprs,
                anchor=anchor,
                probe_meta=probe_meta,
                traj_metric=traj_metric,
                mapping_penalty=mapping_penalty,
                finite_mask_enabled=finite_mask_enabled,
                finite_mask_min_dataset_frac=finite_mask_min_dataset_frac,
                finite_mask_min_points=finite_mask_min_points,
                domain_projection_cfg=domain_projection_cfg,
                limit=max(pool_topk, int(hp.return_topk)),
                best_strategy="mse_decade_size",
            )
            hidden_score_head_skipped += int(shallow_hidden_skipped)
            raw_rows.extend(shallow_rows)
            additive_diag["pre_mutation_seed_rows"] = int(additive_diag["pre_mutation_seed_rows"]) + len(shallow_rows)
            shallow_pooled_rows = sorted(_pool_order_rows(shallow_rows).values(), key=lambda r: float(r["score"]))
            pre_context_diag: dict[str, Any] = {}
            shallow_context_rows = _build_contextual_atom_rows(
                spec=spec,
                order=order,
                base_rows=[
                    *_periodic_seed_atom_rows(x_fit, y_fit, hp, fit_meta),
                    *_gs_symmetry_seed_rows(spec, order),
                    *shallow_pooled_rows,
                ],
                x_fit=x_fit,
                y_fit=y_fit,
                x_probe=x_probe,
                y_probe=y_probe,
                hp=hp,
                fit_meta=fit_meta,
                probe_meta=probe_meta,
                traj_metric=traj_metric,
                diagnostics_out=pre_context_diag,
            )
            additive_diag["pre_mutation_contextual_atom_diagnostics"] = pre_context_diag
            for row in shallow_context_rows:
                row["context_phase"] = "pre_mutation"
                row["source_lane"] = row.get("source_lane", "contextual_atom_promotion")
            raw_rows.extend(shallow_context_rows)
            additive_diag["pre_mutation_context_rows"] = (
                int(additive_diag["pre_mutation_context_rows"]) + len(shallow_context_rows)
            )
            pre_combo_diag: dict[str, Any] = {}
            shallow_combo_rows = _build_sparse_combo_rows(
                spec=spec,
                order=order,
                base_rows=[
                    *_periodic_seed_atom_rows(x_fit, y_fit, hp, fit_meta),
                    *_gs_symmetry_seed_rows(spec, order),
                    *shallow_pooled_rows,
                    *shallow_context_rows,
                ],
                x_fit=x_fit,
                y_fit=y_fit,
                x_probe=x_probe,
                y_probe=y_probe,
                hp=hp,
                probe_meta=probe_meta,
                traj_metric=traj_metric,
                diagnostics_out=pre_combo_diag,
            )
            additive_diag["pre_mutation_combo_diagnostics"] = pre_combo_diag
            for row in shallow_combo_rows:
                row["combo_phase"] = "pre_mutation"
                row["source_lane"] = "additive_fss"
            raw_rows.extend(shallow_combo_rows)
            additive_diag["pre_mutation_combo_rows"] = (
                int(additive_diag["pre_mutation_combo_rows"]) + len(shallow_combo_rows)
            )
            if shallow_combo_rows:
                best_combo = min(shallow_combo_rows, key=lambda r: float(r.get("score", float("inf"))))
                best_combo_score = float(best_combo.get("score", float("inf")))
                additive_diag["pre_mutation_best_score"] = (
                    None if not math.isfinite(best_combo_score) else float(best_combo_score)
                )
                if math.isfinite(best_combo_score) and best_combo_score < early_stop_mse:
                    additive_diag["pre_mutation_early_return"] = True
                    pooled_early = _pool_order_rows(raw_rows)
                    rows = sorted(pooled_early.values(), key=_de_complexity_order_key)[: int(hp.return_topk)]
                    _finalize_diagnostics()
                    if verbose:
                        print(
                            f"[oracle-de] order={order} early-stop after pre-mutation additive FSS: "
                            f"best_score={best_combo_score:.3e} < early_stop_mse={early_stop_mse:.3e}",
                            flush=True,
                    )
                    return rows, n_seeds, n_seeds_ran, float(n_iter_each)
            elif math.isfinite(shallow_best_score):
                additive_diag["pre_mutation_best_score"] = float(shallow_best_score)
            if shallow_context_rows:
                best_context = min(shallow_context_rows, key=lambda r: float(r.get("score", float("inf"))))
                best_context_score = float(best_context.get("score", float("inf")))
                prev_best = additive_diag.get("pre_mutation_best_score", None)
                if prev_best is None or (
                    math.isfinite(best_context_score) and best_context_score < float(prev_best)
                ):
                    additive_diag["pre_mutation_best_score"] = (
                        None if not math.isfinite(best_context_score) else float(best_context_score)
                    )
                if math.isfinite(best_context_score) and best_context_score < early_stop_mse:
                    additive_diag["pre_mutation_early_return"] = True
                    pooled_early = _pool_order_rows(raw_rows)
                    rows = sorted(pooled_early.values(), key=_de_complexity_order_key)[: int(hp.return_topk)]
                    _finalize_diagnostics()
                    if verbose:
                        print(
                            f"[oracle-de] order={order} early-stop after pre-mutation contextual atom: "
                            f"best_score={best_context_score:.3e} < early_stop_mse={early_stop_mse:.3e}",
                            flush=True,
                        )
                    return rows, n_seeds, n_seeds_ran, float(n_iter_each)

        arch = run_explorer_core(**core_kwargs)
        refine_diag = getattr(arch, "refine_diagnostics", None)
        if isinstance(refine_diag, dict):
            _merge_refine_diagnostics(refine_diagnostics_total, refine_diag)
            refine_diagnostics_by_seed.append(
                {
                    "seed_search": int(seed_search),
                    **dict(_to_jsonable(refine_diag)),
                }
            )
        refine_slate_stats = getattr(arch, "refine_slate_stats", None)
        if isinstance(refine_slate_stats, dict):
            refine_slate_stats_by_seed.append(
                {
                    "seed_search": int(seed_search),
                    **dict(_to_jsonable(refine_slate_stats)),
                }
            )

        full_rows, seed_best_score, full_hidden_skipped = _materialize_order_arch_rows(
            arch=arch,
            spec=spec,
            order=order,
            seed_search=seed_search,
            x_probe=x_probe,
            y_probe=y_probe,
            hp=hp,
            input_exprs=input_exprs,
            anchor=anchor,
            probe_meta=probe_meta,
            traj_metric=traj_metric,
            mapping_penalty=mapping_penalty,
            finite_mask_enabled=finite_mask_enabled,
            finite_mask_min_dataset_frac=finite_mask_min_dataset_frac,
            finite_mask_min_points=finite_mask_min_points,
            domain_projection_cfg=domain_projection_cfg,
            limit=int(hp.return_topk),
            best_strategy="mse_decade_size",
        )
        hidden_score_head_skipped += int(full_hidden_skipped)
        raw_rows.extend(full_rows)

        if seed_best_score < early_stop_mse:
            if verbose:
                print(
                    f"[oracle-de] order={order} early-stop after seed_search={int(seed_search)}: "
                    f"best_score={seed_best_score:.3e} < early_stop_mse={early_stop_mse:.3e}"
                )
            break

    pooled = _pool_order_rows(raw_rows)

    pooled_rows = sorted(
        [row for row in pooled.values() if not bool(row.get("basis_state_search", False))],
        key=lambda r: float(r["score"]),
    )
    context_rows = _build_contextual_atom_rows(
        spec=spec,
        order=order,
        base_rows=[
            *_periodic_seed_atom_rows(x_fit, y_fit, hp, fit_meta),
            *_gs_symmetry_seed_rows(spec, order),
            *pooled_rows,
        ],
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        fit_meta=fit_meta,
        probe_meta=probe_meta,
        traj_metric=traj_metric,
        diagnostics_out=additive_diag.setdefault("post_mutation_contextual_atom_diagnostics", {}),
    )
    additive_diag["post_mutation_context_rows"] = len(context_rows)
    for row in context_rows:
        row["context_phase"] = row.get("context_phase", "post_mutation")
        row["source_lane"] = row.get("source_lane", "contextual_atom_promotion")
        key = (str(row["expr"]), str(row.get("mapping_kind", "")))
        prev = pooled.get(key)
        if prev is None or float(row["score"]) < float(prev["score"]):
            pooled[key] = row
    pooled_rows = sorted(
        [row for row in pooled.values() if not bool(row.get("basis_state_search", False))],
        key=lambda r: float(r["score"]),
    )
    combo_rows = _build_sparse_combo_rows(
        spec=spec,
        order=order,
        base_rows=[
            *_periodic_seed_atom_rows(x_fit, y_fit, hp, fit_meta),
            *_gs_symmetry_seed_rows(spec, order),
            *pooled_rows,
        ],
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        probe_meta=probe_meta,
        traj_metric=traj_metric,
        diagnostics_out=additive_diag.setdefault("post_mutation_combo_diagnostics", {}),
    )
    additive_diag["post_mutation_combo_rows"] = len(combo_rows)
    for row in combo_rows:
        row["combo_phase"] = row.get("combo_phase", "post_mutation")
        row["source_lane"] = row.get("source_lane", "additive_fss")
        key = (str(row["expr"]), str(row.get("mapping_kind", "")))
        prev = pooled.get(key)
        if prev is None or float(row["score"]) < float(prev["score"]):
            pooled[key] = row

    rows = sorted(pooled.values(), key=_de_complexity_order_key)[: int(hp.return_topk)]
    _finalize_diagnostics()
    return rows, n_seeds, n_seeds_ran, float(n_iter_each)


def _dispatch_orders_parallel(
    order_candidates: Sequence[int],
    order_fn: Callable[[int, threading.Event | None], dict[str, Any]],
    *,
    parallel: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run *order_fn* for each candidate order, optionally in parallel.

    Parameters
    ----------
    order_candidates : sequence of int
        Orders to search (e.g. ``(1, 2)``).
    order_fn : callable ``(order, stop_event) -> dict``
        Per-order work that returns one element for ``per_order``.
    parallel : bool
        When *True* and more than one order is given, each order runs in its
        own thread sharing a ``threading.Event`` so the first to hit
        ``early_stop_mse`` cancels the others.
    verbose : bool
        Passed through for logging.
    """
    orders = [int(o) for o in order_candidates]

    if len(orders) <= 1 or not parallel:
        # Serial path — no threading overhead.
        return [order_fn(o, None) for o in orders]

    # Parallel path — shared stop event.
    stop_event = threading.Event()
    results: dict[int, dict[str, Any]] = {}
    errors: dict[int, BaseException] = {}

    def _worker(order: int) -> None:
        try:
            results[order] = order_fn(order, stop_event)
        except BaseException as exc:
            errors[order] = exc

    threads = [threading.Thread(target=_worker, args=(o,), daemon=True) for o in orders]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Propagate first error if any thread failed.
    if errors:
        first_order = min(errors)
        raise errors[first_order]

    # Return in original order.
    return [results[o] for o in orders]


def run_oracle_de_from_features(
    spec: DELabSpec,
    features: DEFeatureTensors,
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    verbose: bool = True,
    trajectories: Sequence[_Trajectory] | None = None,
    parallel_orders: bool = True,
) -> dict[str, Any]:
    """Run factorized symbolic search/continuous skeleton refinement directly from precomputed DE feature tensors."""

    hp = factorized_search_hp if factorized_search_hp is not None else default_oracle_de_hyperparams()
    run_seed = int(hp.seed if seed is None else seed)
    feats = _validate_feature_tensors(features, dtype=dtype)

    started = time.perf_counter()

    def _order_fn(order: int, stop_event: threading.Event | None) -> dict[str, Any]:
        order_i = int(order)

        x_fit, y_fit, feat_names = _build_table_from_features(spec, feats, order=order_i, split="fit")
        x_probe, y_probe, _ = _build_table_from_features(spec, feats, order=order_i, split="probe")

        var_dims, y_dims = _dims_for_order(spec, order_i)
        if not bool(enforce_dims):
            var_dims = None
            y_dims = None

        order_diagnostics: dict[str, Any] = {}
        rows, n_seeds, n_seeds_ran, n_iter_each = _run_order_search(
            spec=spec,
            order=order_i,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            hp=hp,
            dtype=dtype,
            run_seed=run_seed,
            var_dims=var_dims,
            y_dims=y_dims,
            verbose=verbose,
            probe_meta=[("pooled", x_probe, y_probe)],
            traj_metric=str(spec.traj_metric),
            stop_event=stop_event,
            diagnostics_out=order_diagnostics,
        )

        val_topk = max(0, int(spec.validate_integrate_topk))
        if val_topk > 0:
            domain_projection_cfg = _domain_projection_cfg_from_hp(hp)
            for i, row in enumerate(rows):
                if i < val_topk and trajectories is not None:
                    mse_int = _validate_candidate_by_integration(
                        spec=spec,
                        order=order_i,
                        expr_ast=row.get("_expr_obj"),
                        mapping=row.get("_mapping_obj"),
                        trajectories=trajectories,
                        dtype=dtype,
                        domain_projection_cfg=domain_projection_cfg,
                    )
                    row["integrate_mse"] = float(mse_int)
                    row["integrate_ok"] = bool(math.isfinite(mse_int))
                else:
                    row["integrate_mse"] = None
                    row["integrate_ok"] = None

        for row in rows:
            row.pop("_expr_obj", None)
            row.pop("_mapping_obj", None)

        best_row = rows[0] if rows else None
        return {
            "order": int(order_i),
            "nvars": int(x_fit.shape[1]),
            "n_points_fit": int(x_fit.shape[0]),
            "n_points_probe": int(x_probe.shape[0]),
            "n_points_total": int(x_fit.shape[0]) + int(x_probe.shape[0]),
            "feature_names": list(feat_names or []),
            "target_name": _target_name(spec, order_i),
            "split_mode": "prebuilt_features",
            "traj_metric": str(spec.traj_metric),
            "n_traj_total": 1,
            "n_traj_fit": 1,
            "n_traj_probe": 1,
            "fit_traj_ids": ["pooled"],
            "probe_traj_ids": ["pooled"],
            "var_dims": None if var_dims is None else [list(d) for d in var_dims],
            "y_dims": None if y_dims is None else list(y_dims),
            "n_seeds": int(n_seeds),
            "n_seeds_ran": int(n_seeds_ran),
            "n_iter_each": int(n_iter_each),
            "search_diagnostics": _to_jsonable(order_diagnostics),
            "additive_fss": _to_jsonable(order_diagnostics.get("additive_fss", {})),
            "refine_diagnostics": _to_jsonable(order_diagnostics.get("refine_diagnostics", {})),
            "refine_cost_summary": _to_jsonable(order_diagnostics.get("refine_cost_summary", {})),
            "refine_diagnostics_by_seed": _to_jsonable(
                order_diagnostics.get("refine_diagnostics_by_seed", [])
            ),
            "refine_slate_stats_by_seed": _to_jsonable(
                order_diagnostics.get("refine_slate_stats_by_seed", [])
            ),
            "results": rows,
            "best": best_row,
        }

    per_order = _dispatch_orders_parallel(
        spec.order_candidates, _order_fn, parallel=parallel_orders, verbose=verbose,
    )

    # Prefer lower order: higher-order models overfit more easily.
    # A factor of 10 per extra order is conservative (genuine order-2
    # problems show >1e8 advantage over order-1).
    _opf = 10.0
    if hasattr(spec, "extra") and isinstance(getattr(spec, "extra", None), dict):
        _opf = float(spec.extra.get("order_preference_factor", _opf))
    _min_ord = min((int(po.get("order", 99)) for po in per_order if po.get("best") is not None), default=1)

    best_global = None
    best_adj = float("inf")
    for po in per_order:
        b = po.get("best", None)
        if b is None:
            continue
        raw = float(b["score"])
        adjusted = raw * (_opf ** (int(po.get("order", 1)) - _min_ord))
        if adjusted < best_adj:
            best_adj = adjusted
            best_global = b

    elapsed = float(time.perf_counter() - started)
    refine_diagnostics_total: dict[str, Any] = {}
    for order_row in per_order:
        _merge_refine_diagnostics(
            refine_diagnostics_total,
            order_row.get("refine_diagnostics", {}),
        )

    report = {
        "spec_id": spec.id,
        "csv_paths": list(spec.csv_paths),
        "order_candidates": [int(o) for o in spec.order_candidates],
        "x_axis": int(spec.x_axis),
        "include_x": bool(spec.include_x),
        "include_u": bool(spec.include_u),
        "include_du": bool(spec.include_du),
        "x_col": str(spec.x_col),
        "u_col": str(spec.u_col),
        "out_idx": int(spec.out_idx),
        "y_transform": str(spec.y_transform),
        "split_mode": str(spec.split_mode),
        "traj_metric": str(spec.traj_metric),
        "trajectories": [{"id": "pooled", "csv": None}],
        "fit_trajectories": [{"id": "pooled", "csv": None}],
        "probe_trajectories": [{"id": "pooled", "csv": None}],
        "constants": {c.name: float(c.value) for c in spec.constants},
        "deriv": {
            "method": str(spec.derivative.method),
            "s": float(spec.derivative.spline_s),
            "k": int(spec.derivative.spline_k),
            "du_col": spec.derivative.du_col,
            "d2u_col": spec.derivative.d2u_col,
        },
        "dims": None
        if spec.dims is None
        else {
            "basis": list(spec.dims.basis),
            "x": list(spec.dims.x_dim),
            "u": list(spec.dims.u_dim),
        },
        "extra": None if spec.extra is None else _to_jsonable(spec.extra),
        "dtype": str(dtype),
        "enforce_dims": bool(enforce_dims),
        "seed": int(run_seed),
        "wall_seconds": elapsed,
        "resolved_config": _to_jsonable(factorized_config_report(hp)),
        "refine_diagnostics": _to_jsonable(refine_diagnostics_total),
        "refine_cost_summary": _to_jsonable(
            _refine_diagnostics_summary(refine_diagnostics_total)
        ),
        "search_diagnostics_summary": _to_jsonable(_search_diagnostics_summary(per_order)),
        "hp": {
            "n_iter": int(hp.n_iter),
            "max_depth": int(hp.max_depth),
            "poly_degree": int(hp.poly_degree),
            "return_topk": int(hp.return_topk),
            "n_fit": int(hp.n_fit),
            "n_probe": int(hp.n_probe),
            "n_seeds": int(hp.n_seeds),
            "split_iter_across_seeds": bool(hp.split_iter_across_seeds),
            "brute_depth": None if hp.brute_depth is None else int(hp.brute_depth),
            "early_stop_mse": float(hp.early_stop_mse),
            "brute_max_expressions": int(hp.brute_max_expressions),
            "refine_enable": bool(hp.refine_enable),
            "refine_profile": str(getattr(hp, "refine_profile", "default")),
            "refine_mode": str(getattr(hp, "refine_mode", "slate")),
            "refine_during_brute": bool(getattr(hp, "refine_during_brute", False)),
            "refine_during_mutation": bool(getattr(hp, "refine_during_mutation", False)),
            "refine_during_controller_slate": bool(getattr(hp, "refine_during_controller_slate", False)),
            "refine_during_slate": bool(getattr(hp, "refine_during_slate", True)),
            "refine_slate_after_brute": bool(getattr(hp, "refine_slate_after_brute", True)),
            "refine_slate_period": int(getattr(hp, "refine_slate_period", 0)),
            "refine_final_polish": bool(getattr(hp, "refine_final_polish", True)),
            "refine_slate_k": int(getattr(hp, "refine_slate_k", 16)),
            "refine_slate_diverse_k": int(getattr(hp, "refine_slate_diverse_k", 8)),
            "refine_slate_budget": int(getattr(hp, "refine_slate_budget", 32)),
            "refine_optimizer": str(getattr(hp, "refine_optimizer", "lbfgs")),
            "refine_lbfgs_escalate_improve_factor": float(
                getattr(hp, "refine_lbfgs_escalate_improve_factor", 2.0)
            ),
            "refine_lbfgs_steps": int(hp.refine_lbfgs_steps),
            "refine_num_restarts": int(hp.refine_num_restarts),
            "refine_max_variants": int(hp.refine_max_variants),
            "refine_max_params": int(hp.refine_max_params),
            "refine_linear_combo_enable": bool(hp.refine_linear_combo_enable),
            "refine_gate_best_factor": float(hp.refine_gate_best_factor),
            "refine_max_trials": int(hp.refine_max_trials),
            "complexity_penalty": float(getattr(hp, "complexity_penalty", 0.0)),
            "mapping_complexity_penalty": float(getattr(hp, "mapping_complexity_penalty", 0.0)),
            "de_sparse_combo_enable": bool(getattr(hp, "de_sparse_combo_enable", False)),
            "de_sparse_combo_pool_topk": int(getattr(hp, "de_sparse_combo_pool_topk", 8)),
            "de_sparse_combo_max_terms": int(getattr(hp, "de_sparse_combo_max_terms", 2)),
            "de_sparse_combo_beam": int(getattr(hp, "de_sparse_combo_beam", 16)),
            "de_sparse_combo_backward_prune": bool(getattr(hp, "de_sparse_combo_backward_prune", True)),
            "de_sparse_combo_mapping_mode": str(getattr(hp, "de_sparse_combo_mapping_mode", "affine_only")),
            "de_sparse_combo_corr_eps": float(getattr(hp, "de_sparse_combo_corr_eps", 1.0e-8)),
            "de_sparse_combo_rank_eps": float(getattr(hp, "de_sparse_combo_rank_eps", 1.0e-10)),
            "de_sparse_combo_max_condition": float(getattr(hp, "de_sparse_combo_max_condition", 1.0e10)),
            "de_sparse_combo_cond_penalty": float(getattr(hp, "de_sparse_combo_cond_penalty", 0.0)),
            "de_sparse_combo_coeff_stability_penalty": float(
                getattr(hp, "de_sparse_combo_coeff_stability_penalty", 0.0)
            ),
            "de_sparse_combo_coeff_spread_warn": float(getattr(hp, "de_sparse_combo_coeff_spread_warn", 2.0)),
            "de_score_head_untyped_enable": bool(getattr(hp, "de_score_head_untyped_enable", False)),
            "de_score_head_max_terms": int(getattr(hp, "de_score_head_max_terms", 2)),
        },
        "per_order": per_order,
        "best": best_global,
    }
    return report


def run_oracle_de_from_trajectories(
    spec: DELabSpec,
    trajectories: Sequence[_Trajectory],
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    verbose: bool = True,
    parallel_orders: bool = True,
) -> dict[str, Any]:
    """Run factorized symbolic search/continuous skeleton refinement using multiple trajectory tables and shared RHS scoring."""

    hp = factorized_search_hp if factorized_search_hp is not None else default_oracle_de_hyperparams()
    run_seed = int(hp.seed if seed is None else seed)
    if not trajectories:
        raise ValueError("no trajectories provided")

    started = time.perf_counter()

    def _order_fn(order: int, stop_event: threading.Event | None) -> dict[str, Any]:
        order_i = int(order)
        tables = _build_multi_tables_for_order(
            spec,
            trajectories,
            order=order_i,
            hp=hp,
            seed=run_seed,
            dtype=dtype,
        )

        var_dims, y_dims = _dims_for_order(spec, order_i)
        if not bool(enforce_dims):
            var_dims = None
            y_dims = None

        order_diagnostics: dict[str, Any] = {}
        rows, n_seeds, n_seeds_ran, n_iter_each = _run_order_search(
            spec=spec,
            order=order_i,
            x_fit=tables.x_fit,
            y_fit=tables.y_fit,
            x_probe=tables.x_probe,
            y_probe=tables.y_probe,
            hp=hp,
            dtype=dtype,
            run_seed=run_seed,
            var_dims=var_dims,
            y_dims=y_dims,
            verbose=verbose,
            fit_meta=tables.fit_meta,
            probe_meta=tables.probe_meta,
            traj_metric=str(spec.traj_metric),
            stop_event=stop_event,
            diagnostics_out=order_diagnostics,
        )

        val_topk = max(0, int(spec.validate_integrate_topk))
        if val_topk > 0:
            domain_projection_cfg = _domain_projection_cfg_from_hp(hp)
            for i, row in enumerate(rows):
                if i < val_topk:
                    mse_int = _validate_candidate_by_integration(
                        spec=spec,
                        order=order_i,
                        expr_ast=row.get("_expr_obj"),
                        mapping=row.get("_mapping_obj"),
                        trajectories=trajectories,
                        dtype=dtype,
                        domain_projection_cfg=domain_projection_cfg,
                    )
                    row["integrate_mse"] = float(mse_int)
                    row["integrate_ok"] = bool(math.isfinite(mse_int))
                else:
                    row["integrate_mse"] = None
                    row["integrate_ok"] = None

        for row in rows:
            row.pop("_expr_obj", None)
            row.pop("_mapping_obj", None)

        best_row = rows[0] if rows else None
        return {
            "order": int(order_i),
            "nvars": int(tables.x_fit.shape[1]),
            "n_points_fit": int(tables.x_fit.shape[0]),
            "n_points_probe": int(tables.x_probe.shape[0]),
            "n_points_total": int(tables.x_fit.shape[0]) + int(tables.x_probe.shape[0]),
            "feature_names": list(tables.feature_names or []),
            "target_name": _target_name(spec, order_i),
            "split_mode": str(spec.split_mode),
            "traj_metric": str(spec.traj_metric),
            "n_traj_total": int(len(trajectories)),
            "n_traj_fit": int(len(tables.fit_traj_ids)),
            "n_traj_probe": int(len(tables.probe_traj_ids)),
            "fit_traj_ids": list(tables.fit_traj_ids),
            "probe_traj_ids": list(tables.probe_traj_ids),
            "var_dims": None if var_dims is None else [list(d) for d in var_dims],
            "y_dims": None if y_dims is None else list(y_dims),
            "n_seeds": int(n_seeds),
            "n_seeds_ran": int(n_seeds_ran),
            "n_iter_each": int(n_iter_each),
            "search_diagnostics": _to_jsonable(order_diagnostics),
            "additive_fss": _to_jsonable(order_diagnostics.get("additive_fss", {})),
            "refine_diagnostics": _to_jsonable(order_diagnostics.get("refine_diagnostics", {})),
            "refine_cost_summary": _to_jsonable(order_diagnostics.get("refine_cost_summary", {})),
            "refine_diagnostics_by_seed": _to_jsonable(
                order_diagnostics.get("refine_diagnostics_by_seed", [])
            ),
            "refine_slate_stats_by_seed": _to_jsonable(
                order_diagnostics.get("refine_slate_stats_by_seed", [])
            ),
            "results": rows,
            "best": best_row,
        }

    per_order = _dispatch_orders_parallel(
        spec.order_candidates, _order_fn, parallel=parallel_orders, verbose=verbose,
    )

    # Prefer lower order: higher-order models overfit more easily.
    # A factor of 10 per extra order is conservative (genuine order-2
    # problems show >1e8 advantage over order-1).
    _opf = 10.0
    if hasattr(spec, "extra") and isinstance(getattr(spec, "extra", None), dict):
        _opf = float(spec.extra.get("order_preference_factor", _opf))
    _min_ord = min((int(po.get("order", 99)) for po in per_order if po.get("best") is not None), default=1)

    best_global = None
    best_adj = float("inf")
    for po in per_order:
        b = po.get("best", None)
        if b is None:
            continue
        raw = float(b["score"])
        adjusted = raw * (_opf ** (int(po.get("order", 1)) - _min_ord))
        if adjusted < best_adj:
            best_adj = adjusted
            best_global = b

    elapsed = float(time.perf_counter() - started)
    refine_diagnostics_total: dict[str, Any] = {}
    for order_row in per_order:
        _merge_refine_diagnostics(
            refine_diagnostics_total,
            order_row.get("refine_diagnostics", {}),
        )
    report = {
        "spec_id": spec.id,
        "csv_paths": list(spec.csv_paths),
        "order_candidates": [int(o) for o in spec.order_candidates],
        "x_axis": int(spec.x_axis),
        "include_x": bool(spec.include_x),
        "include_u": bool(spec.include_u),
        "include_du": bool(spec.include_du),
        "x_col": str(spec.x_col),
        "u_col": str(spec.u_col),
        "out_idx": int(spec.out_idx),
        "y_transform": str(spec.y_transform),
        "split_mode": str(spec.split_mode),
        "traj_metric": str(spec.traj_metric),
        "trajectories": [{"id": str(t.traj_id), "csv": str(t.path)} for t in trajectories],
        "fit_trajectories": [{"id": str(t.id), "csv": str(t.csv)} for t in spec.fit_trajectories],
        "probe_trajectories": [{"id": str(t.id), "csv": str(t.csv)} for t in spec.probe_trajectories],
        "constants": {c.name: float(c.value) for c in spec.constants},
        "deriv": {
            "method": str(spec.derivative.method),
            "s": float(spec.derivative.spline_s),
            "k": int(spec.derivative.spline_k),
            "du_col": spec.derivative.du_col,
            "d2u_col": spec.derivative.d2u_col,
        },
        "dims": None
        if spec.dims is None
        else {
            "basis": list(spec.dims.basis),
            "x": list(spec.dims.x_dim),
            "u": list(spec.dims.u_dim),
        },
        "extra": None if spec.extra is None else _to_jsonable(spec.extra),
        "dtype": str(dtype),
        "enforce_dims": bool(enforce_dims),
        "seed": int(run_seed),
        "wall_seconds": elapsed,
        "resolved_config": _to_jsonable(factorized_config_report(hp)),
        "refine_diagnostics": _to_jsonable(refine_diagnostics_total),
        "refine_cost_summary": _to_jsonable(
            _refine_diagnostics_summary(refine_diagnostics_total)
        ),
        "search_diagnostics_summary": _to_jsonable(_search_diagnostics_summary(per_order)),
        "hp": {
            "n_iter": int(hp.n_iter),
            "max_depth": int(hp.max_depth),
            "poly_degree": int(hp.poly_degree),
            "return_topk": int(hp.return_topk),
            "n_fit": int(hp.n_fit),
            "n_probe": int(hp.n_probe),
            "n_seeds": int(hp.n_seeds),
            "split_iter_across_seeds": bool(hp.split_iter_across_seeds),
            "brute_depth": None if hp.brute_depth is None else int(hp.brute_depth),
            "early_stop_mse": float(hp.early_stop_mse),
            "brute_max_expressions": int(hp.brute_max_expressions),
            "refine_enable": bool(hp.refine_enable),
            "refine_profile": str(getattr(hp, "refine_profile", "default")),
            "refine_mode": str(getattr(hp, "refine_mode", "slate")),
            "refine_during_brute": bool(getattr(hp, "refine_during_brute", False)),
            "refine_during_mutation": bool(getattr(hp, "refine_during_mutation", False)),
            "refine_during_controller_slate": bool(getattr(hp, "refine_during_controller_slate", False)),
            "refine_during_slate": bool(getattr(hp, "refine_during_slate", True)),
            "refine_slate_after_brute": bool(getattr(hp, "refine_slate_after_brute", True)),
            "refine_slate_period": int(getattr(hp, "refine_slate_period", 0)),
            "refine_final_polish": bool(getattr(hp, "refine_final_polish", True)),
            "refine_slate_k": int(getattr(hp, "refine_slate_k", 16)),
            "refine_slate_diverse_k": int(getattr(hp, "refine_slate_diverse_k", 8)),
            "refine_slate_budget": int(getattr(hp, "refine_slate_budget", 32)),
            "refine_optimizer": str(getattr(hp, "refine_optimizer", "lbfgs")),
            "refine_lbfgs_escalate_improve_factor": float(
                getattr(hp, "refine_lbfgs_escalate_improve_factor", 2.0)
            ),
            "refine_lbfgs_steps": int(hp.refine_lbfgs_steps),
            "refine_num_restarts": int(hp.refine_num_restarts),
            "refine_max_variants": int(hp.refine_max_variants),
            "refine_max_params": int(hp.refine_max_params),
            "refine_linear_combo_enable": bool(hp.refine_linear_combo_enable),
            "refine_gate_best_factor": float(hp.refine_gate_best_factor),
            "refine_max_trials": int(hp.refine_max_trials),
            "complexity_penalty": float(getattr(hp, "complexity_penalty", 0.0)),
            "mapping_complexity_penalty": float(getattr(hp, "mapping_complexity_penalty", 0.0)),
            "de_sparse_combo_enable": bool(getattr(hp, "de_sparse_combo_enable", False)),
            "de_sparse_combo_pool_topk": int(getattr(hp, "de_sparse_combo_pool_topk", 8)),
            "de_sparse_combo_max_terms": int(getattr(hp, "de_sparse_combo_max_terms", 2)),
            "de_sparse_combo_beam": int(getattr(hp, "de_sparse_combo_beam", 16)),
            "de_sparse_combo_backward_prune": bool(getattr(hp, "de_sparse_combo_backward_prune", True)),
            "de_score_head_untyped_enable": bool(getattr(hp, "de_score_head_untyped_enable", False)),
            "de_score_head_max_terms": int(getattr(hp, "de_score_head_max_terms", 2)),
        },
        "per_order": per_order,
        "best": best_global,
    }
    return report


def run_oracle_de_equation(
    spec: DELabSpec,
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    verbose: bool = True,
    parallel_orders: bool = True,
    derivative_provider_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = derivative_provider,
) -> dict[str, Any]:
    """Run factorized symbolic search/continuous skeleton refinement on DE feature tables built from trajectory files."""

    hp = factorized_search_hp if factorized_search_hp is not None else default_oracle_de_hyperparams()
    run_seed = int(hp.seed if seed is None else seed)

    trajectories = _load_trajectories(spec, derivative_provider_fn=derivative_provider_fn)
    return run_oracle_de_from_trajectories(
        spec,
        trajectories,
        factorized_search_hp=hp,
        seed=run_seed,
        dtype=dtype,
        enforce_dims=enforce_dims,
        verbose=verbose,
        parallel_orders=parallel_orders,
    )


def save_oracle_de_report(report: dict[str, Any], path: str | pathlib.Path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")


def _apply_cli_overrides(hp: FactorizedSearchConfig, args: argparse.Namespace) -> FactorizedSearchConfig:
    if args.n_iter is not None:
        hp.n_iter = int(args.n_iter)
    if args.max_depth is not None:
        hp.max_depth = int(args.max_depth)
    if args.poly_degree is not None:
        hp.poly_degree = int(args.poly_degree)
    if args.return_topk is not None:
        hp.return_topk = int(args.return_topk)
    if args.n_fit is not None:
        hp.n_fit = int(args.n_fit)
    if args.n_probe is not None:
        hp.n_probe = int(args.n_probe)
    if args.n_seeds is not None:
        hp.n_seeds = int(args.n_seeds)
    if args.split_iter_across_seeds is not None:
        hp.split_iter_across_seeds = bool(args.split_iter_across_seeds)
    if args.brute_depth is not None:
        hp.brute_depth = int(args.brute_depth)
    if args.no_brute_force:
        hp.brute_depth = 0
    if getattr(args, "brute_max_expressions", None) is not None:
        hp.brute_max_expressions = int(args.brute_max_expressions)
    if args.early_stop_mse is not None:
        hp.early_stop_mse = float(args.early_stop_mse)
    if args.complexity_penalty is not None:
        hp.complexity_penalty = float(args.complexity_penalty)
    if args.mapping_complexity_penalty is not None:
        hp.mapping_complexity_penalty = float(args.mapping_complexity_penalty)

    if args.refine_enable is not None:
        hp.refine_enable = bool(args.refine_enable)
    if getattr(args, "refine_profile", None) is not None:
        hp = apply_refine_profile(hp, args.refine_profile)
    if getattr(args, "refine_mode", None) is not None:
        hp = apply_refine_mode_placement_defaults(hp, args.refine_mode)
    for attr in (
        "refine_during_brute",
        "refine_during_mutation",
        "refine_during_controller_slate",
        "refine_during_slate",
        "refine_slate_after_brute",
        "refine_final_polish",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(hp, attr, bool(value))
    for attr in (
        "refine_slate_period",
        "refine_slate_k",
        "refine_slate_diverse_k",
        "refine_slate_budget",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(hp, attr, int(value))
    if getattr(args, "refine_optimizer", None) is not None:
        hp.refine_optimizer = str(args.refine_optimizer)
    if getattr(args, "refine_lbfgs_escalate_improve_factor", None) is not None:
        hp.refine_lbfgs_escalate_improve_factor = float(args.refine_lbfgs_escalate_improve_factor)
    if args.refine_lbfgs_steps is not None:
        hp.refine_lbfgs_steps = int(args.refine_lbfgs_steps)
    if args.refine_num_restarts is not None:
        hp.refine_num_restarts = int(args.refine_num_restarts)
    if args.refine_max_variants is not None:
        hp.refine_max_variants = int(args.refine_max_variants)
    if args.refine_max_params is not None:
        hp.refine_max_params = int(args.refine_max_params)
    if args.refine_linear_combo_enable is not None:
        hp.refine_linear_combo_enable = bool(args.refine_linear_combo_enable)
    if args.refine_gate_best_factor is not None:
        hp.refine_gate_best_factor = float(args.refine_gate_best_factor)
    if args.refine_max_trials is not None:
        hp.refine_max_trials = int(args.refine_max_trials)

    # Multi-dataset (multi-trajectory) joint-score / joint-refine controls.
    #
    # For DE discovery across trajectories that share the same governing law
    # (different ICs, same parameters), it is often important to keep a *single*
    # global mapping/linear coefficients across datasets. Allowing per-dataset
    # affine maps can hide structural errors and correlates strongly with order
    # misidentification.
    if getattr(args, "refine_joint_score_enable", None) is not None:
        hp.refine_joint_score_enable = bool(args.refine_joint_score_enable)
    if getattr(args, "refine_joint_enable", None) is not None:
        hp.refine_joint_enable = bool(args.refine_joint_enable)
    if getattr(args, "refine_joint_terms_enable", None) is not None:
        hp.refine_joint_terms_enable = bool(args.refine_joint_terms_enable)

    # continuous skeleton refinement subset controls.
    if getattr(args, "refine_fit_subset", None) is not None:
        hp.refine_fit_subset = int(args.refine_fit_subset)
    if getattr(args, "refine_fit_subset_mode", None) is not None:
        hp.refine_fit_subset_mode = str(args.refine_fit_subset_mode)

    return hp


def _apply_spec_overrides(spec: DELabSpec, args: argparse.Namespace) -> DELabSpec:
    out = spec

    if args.order_candidates is not None:
        out = replace(out, order_candidates=_parse_order_candidates(args.order_candidates, where="cli.order_candidates"))

    if args.include_x is not None:
        out = replace(out, include_x=bool(args.include_x))
    if args.include_u is not None:
        out = replace(out, include_u=bool(args.include_u))
    if args.include_du is not None:
        out = replace(out, include_du=bool(args.include_du))

    if args.x_col is not None:
        out = replace(out, x_col=str(args.x_col))
    if args.u_col is not None:
        out = replace(out, u_col=str(args.u_col))
    if args.out_idx is not None:
        oi = int(args.out_idx)
        if oi < 0:
            raise ValueError("out_idx must be >= 0")
        out = replace(out, out_idx=oi)
    if args.y_transform is not None:
        yt_name = str(args.y_transform).strip() or "identity"
        _select_y_transform(yt_name)  # validate name
        out = replace(out, y_transform=yt_name)
    if args.split_mode is not None:
        sm = str(args.split_mode).strip().lower()
        if sm not in ("per_traj_point", "traj_holdout"):
            raise ValueError("split_mode must be one of: per_traj_point, traj_holdout")
        out = replace(out, split_mode=sm)
    if args.traj_metric is not None:
        tm = str(args.traj_metric).strip().lower()
        if tm not in ("mean", "max"):
            raise ValueError("traj_metric must be one of: mean, max")
        out = replace(out, traj_metric=tm)

    deriv = out.derivative
    if args.deriv_method is not None:
        deriv = replace(deriv, method=str(args.deriv_method))
    if args.spline_s is not None:
        deriv = replace(deriv, spline_s=float(args.spline_s))
    if args.spline_k is not None:
        deriv = replace(deriv, spline_k=int(args.spline_k))
    if args.du_col is not None:
        deriv = replace(deriv, du_col=str(args.du_col))
    if args.d2u_col is not None:
        deriv = replace(deriv, d2u_col=str(args.d2u_col))
    out = replace(out, derivative=deriv)

    if args.validate_integrate_topk is not None:
        out = replace(out, validate_integrate_topk=int(args.validate_integrate_topk))

    if len(out.fit_trajectories) > 0 or len(out.probe_trajectories) > 0:
        out = replace(
            out,
            split_mode=("traj_holdout" if len(out.probe_trajectories) > 0 else "per_traj_point"),
        )

    if not out.include_x and not out.include_u and not out.include_du and len(out.constants) == 0:
        raise ValueError("feature set is empty after CLI overrides")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle factorized symbolic search/continuous skeleton refinement DE lab runner")
    p.add_argument("--spec", type=str, required=True, help="DE lab spec file (.json/.yaml)")
    p.add_argument("--output", type=str, default=None, help="Optional JSON report path")
    p.add_argument("--seed", type=int, default=None, help="Base random seed")
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    p.add_argument("--ignore_dims", action="store_true", help="Disable dimensional filtering")
    p.add_argument("--quiet", action="store_true", help="Reduce explorer logging")

    par_g = p.add_mutually_exclusive_group()
    par_g.add_argument(
        "--parallel_orders",
        dest="parallel_orders",
        action="store_true",
        help="Search candidate orders in parallel threads (default)",
    )
    par_g.add_argument(
        "--no_parallel_orders",
        dest="parallel_orders",
        action="store_false",
        help="Search candidate orders sequentially",
    )
    p.set_defaults(parallel_orders=True)

    p.add_argument("--order_candidates", type=str, default=None, help="Comma-separated orders (subset of 1,2)")
    p.add_argument("--x_col", type=str, default=None)
    p.add_argument("--u_col", type=str, default=None)
    p.add_argument("--out_idx", type=int, default=None)
    p.add_argument("--y_transform", type=str, default=None)
    p.add_argument("--split_mode", type=str, choices=["per_traj_point", "traj_holdout"], default=None)
    p.add_argument("--traj_metric", type=str, choices=["mean", "max"], default=None)

    feature_x = p.add_mutually_exclusive_group()
    feature_x.add_argument("--include_x", dest="include_x", action="store_true")
    feature_x.add_argument("--no_x", dest="include_x", action="store_false")
    p.set_defaults(include_x=None)

    feature_u = p.add_mutually_exclusive_group()
    feature_u.add_argument("--include_u", dest="include_u", action="store_true")
    feature_u.add_argument("--no_u", dest="include_u", action="store_false")
    p.set_defaults(include_u=None)

    feature_du = p.add_mutually_exclusive_group()
    feature_du.add_argument("--include_du", dest="include_du", action="store_true")
    feature_du.add_argument("--no_du", dest="include_du", action="store_false")
    p.set_defaults(include_du=None)

    p.add_argument("--deriv_method", type=str, choices=["spline", "finite_diff", "precomputed"], default=None)
    p.add_argument("--spline_s", type=float, default=None)
    p.add_argument("--spline_k", type=int, default=None)
    p.add_argument("--du_col", type=str, default=None)
    p.add_argument("--d2u_col", type=str, default=None)
    p.add_argument("--validate_integrate_topk", type=int, default=None)

    # Core search overrides
    p.add_argument("--n_iter", type=int, default=None)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--poly_degree", type=int, default=None)
    p.add_argument("--return_topk", type=int, default=None)
    p.add_argument("--n_fit", type=int, default=None)
    p.add_argument("--n_probe", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=None)

    split_g = p.add_mutually_exclusive_group()
    split_g.add_argument(
        "--split_iter_across_seeds",
        dest="split_iter_across_seeds",
        action="store_true",
        help="Split n_iter across seed sweep",
    )
    split_g.add_argument(
        "--no_split_iter_across_seeds",
        dest="split_iter_across_seeds",
        action="store_false",
        help="Use full n_iter budget per seed",
    )
    p.set_defaults(split_iter_across_seeds=None)

    p.add_argument("--brute_depth", type=int, default=None)
    p.add_argument(
        "--early_stop_mse",
        type=float,
        default=None,
        help="Global solved threshold for brute-force and mutation phases",
    )
    p.add_argument(
        "--complexity_penalty",
        type=float,
        default=None,
        help="Expression-size penalty used during explorer scoring",
    )
    p.add_argument(
        "--mapping_complexity_penalty",
        type=float,
        default=None,
        help="Additional ranking penalty per mapping-complexity unit",
    )
    p.add_argument("--no_brute_force", action="store_true")
    p.add_argument(
        "--brute_max_expressions",
        type=int,
        default=None,
        help="Max expressions scored during brute enumeration phase",
    )

    # continuous skeleton refinement toggles
    refine_g = p.add_mutually_exclusive_group()
    refine_g.add_argument("--plus", dest="refine_enable", action="store_true")
    refine_g.add_argument("--no-plus", dest="refine_enable", action="store_false")
    p.set_defaults(refine_enable=None)

    p.add_argument(
        "--refine_profile",
        default=None,
        metavar="PROFILE",
        help=f"Named continuous-refinement runtime profile ({', '.join(REFINE_PROFILE_NAMES)}; aliases accepted)",
    )
    p.add_argument(
        "--refine_mode",
        choices=["off", "inline", "slate", "final_polish"],
        default=None,
        help="Continuous refinement placement mode",
    )
    for attr, enabled_flag, disabled_flag in (
        ("refine_during_brute", "--refine_during_brute", "--no_refine_during_brute"),
        ("refine_during_mutation", "--refine_during_mutation", "--no_refine_during_mutation"),
        (
            "refine_during_controller_slate",
            "--refine_during_controller_slate",
            "--no_refine_during_controller_slate",
        ),
        ("refine_during_slate", "--refine_during_slate", "--no_refine_during_slate"),
        ("refine_slate_after_brute", "--refine_slate_after_brute", "--no_refine_slate_after_brute"),
        ("refine_final_polish", "--refine_final_polish", "--no_refine_final_polish"),
    ):
        placement_g = p.add_mutually_exclusive_group()
        placement_g.add_argument(enabled_flag, dest=attr, action="store_true")
        placement_g.add_argument(disabled_flag, dest=attr, action="store_false")
        p.set_defaults(**{attr: None})

    p.add_argument("--refine_slate_period", type=int, default=None)
    p.add_argument("--refine_slate_k", type=int, default=None)
    p.add_argument("--refine_slate_diverse_k", type=int, default=None)
    p.add_argument("--refine_slate_budget", type=int, default=None)
    p.add_argument("--refine_optimizer", choices=list(REFINE_OPTIMIZER_NAMES), default=None)
    p.add_argument("--refine_lbfgs_escalate_improve_factor", type=float, default=None)
    p.add_argument("--refine_lbfgs_steps", type=int, default=None)
    p.add_argument("--refine_num_restarts", type=int, default=None)
    p.add_argument("--refine_max_variants", type=int, default=None)
    p.add_argument("--refine_max_params", type=int, default=None)

    linear_g = p.add_mutually_exclusive_group()
    linear_g.add_argument(
        "--refine_linear_combo",
        dest="refine_linear_combo_enable",
        action="store_true",
    )
    linear_g.add_argument(
        "--no_refine_linear_combo",
        dest="refine_linear_combo_enable",
        action="store_false",
    )
    p.set_defaults(refine_linear_combo_enable=None)

    p.add_argument("--refine_gate_best_factor", type=float, default=None)
    p.add_argument("--refine_max_trials", type=int, default=None)

    # continuous skeleton refinement multi-dataset controls.
    # These matter when solving from multiple trajectories/ICs.
    joint_score_g = p.add_mutually_exclusive_group()
    joint_score_g.add_argument(
        "--refine_joint_score",
        dest="refine_joint_score_enable",
        action="store_true",
        help="Enable joint scoring with per-dataset affine maps (multi-dataset mode)",
    )
    joint_score_g.add_argument(
        "--no_refine_joint_score",
        dest="refine_joint_score_enable",
        action="store_false",
        help="Disable per-dataset joint scoring; use pooled (single-map) scoring",
    )
    p.set_defaults(refine_joint_score_enable=None)

    joint_refine_g = p.add_mutually_exclusive_group()
    joint_refine_g.add_argument(
        "--refine_joint_refine",
        dest="refine_joint_enable",
        action="store_true",
        help="Enable joint refinement (multi-dataset mode)",
    )
    joint_refine_g.add_argument(
        "--no_refine_joint_refine",
        dest="refine_joint_enable",
        action="store_false",
        help="Disable joint refinement; keep a single global set of coefficients",
    )
    p.set_defaults(refine_joint_enable=None)

    joint_terms_g = p.add_mutually_exclusive_group()
    joint_terms_g.add_argument(
        "--refine_joint_terms",
        dest="refine_joint_terms_enable",
        action="store_true",
        help="Enable joint scoring via per-dataset linear terms (advanced)",
    )
    joint_terms_g.add_argument(
        "--no_refine_joint_terms",
        dest="refine_joint_terms_enable",
        action="store_false",
        help="Disable per-dataset linear-terms joint scoring (default)",
    )
    p.set_defaults(refine_joint_terms_enable=None)

    # continuous skeleton refinement subset sampling.
    p.add_argument(
        "--refine_fit_subset",
        type=int,
        default=None,
        help="Subset size used in continuous skeleton refinement (0 disables subsampling)",
    )
    p.add_argument(
        "--refine_fit_subset_mode",
        type=str,
        default=None,
        choices=["hash_random", "stride", "stratified"],
        help="Subset selection mode used in continuous skeleton refinement",
    )

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    spec = _apply_spec_overrides(load_de_equation_spec(args.spec), args)
    hp = _apply_cli_overrides(default_oracle_de_hyperparams(), args)

    dtype = torch.float64 if str(args.dtype).lower() == "float64" else torch.float32
    report = run_oracle_de_equation(
        spec,
        factorized_search_hp=hp,
        seed=args.seed,
        dtype=dtype,
        enforce_dims=not bool(args.ignore_dims),
        verbose=not bool(args.quiet),
        parallel_orders=bool(args.parallel_orders),
    )

    best = report.get("best")
    if best is None:
        print(f"[oracle-de] {spec.id}: no candidate found")
    else:
        print(
            f"[oracle-de] {spec.id}: best_mse={float(best['mse']):.6g} "
            f"order={int(best.get('order', -1))} expr={best['expr']} mapping={best.get('mapping_kind', '')}"
        )

    if args.output is not None:
        save_oracle_de_report(report, args.output)
        print(f"[oracle-de] report written to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
