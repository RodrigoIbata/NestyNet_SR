# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Lightweight oracle-driven factorized symbolic search/continuous skeleton refinement lab runner.

This module bypasses Stage A/Stage B neural-surrogate plumbing and drives the
ResidualBasin explorer directly from a ground-truth equation specification.

Typical usage:

    python -m nestynet_sr.sr_search.factorized_search.oracle_lab \
        --spec examples/oracle_factorized_search/specs/feynman_090.json \
        --plus --n_iter 20000 --output results/feynman_090.oracle.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from nestynet_sr.sr_core.bridges import Var

from .config import (
    FactorizedSearchConfig,
    REFINE_OPTIMIZER_NAMES,
    REFINE_PROFILE_NAMES,
    apply_refine_mode_placement_defaults,
    apply_refine_profile,
    factorized_config_report,
)
from .engine.search import run_explorer_core
from .engine.signals import (
    InverseSteeringPotential,
    PathStateFeatures,
    path_distribution_metrics,
    path_summary_stats,
)
from .bridge import mapping_embedding_roundtrip
from .explorer import (
    ACTION_NAME,
    A_REPLACE,
    A_RESIDUAL,
    A_WRAP_UNARY,
    _fit_affine_mapping_from_pair,
    _inverse_collect_local_repair_candidates,
    _inverse_pool_shortlist,
    _invert_binary_context as _explorer_invert_binary_context,
    _invert_unary_context as _explorer_invert_unary_context,
    _normalize_inverse_local_score_mode,
    apply_action,
    apply_residual_action,
    build_pool,
    collect_paths,
    dims_eq,
    estimate_inverse_steering_potential,
    eval_mapping,
    eval_node,
    fit_best,
    get_at,
    invert_context_target as _explorer_invert_context_target,
    invert_mapping_target as _explorer_invert_mapping_target,
    make_engine_refinement_hooks,
    make_engine_runtime_hooks,
    node_depth,
    node_dims,
    node_size,
    node_str,
    replace_at,
    simplify,
)
from .inverse_action import (
    _build_inverse_action_child_expr,
    _estimate_inverse_action_transport,
    _group_inverse_action_preview_rows,
    _inverse_action_path_mode_beam_states,
    _inverse_local_mapping_preview,
    _select_inverse_exact_budget_rows,
    _sort_inverse_action_candidate_rows_by_preview,
    _transport_aligned_local_rows,
)
from .inverse_search import _inverse_path_profile, _inverse_rank_local_repair_candidates
from .inverse_spec_solver import solve_inverse_spec_preview_rows
from .shared_candidate import shared_candidate_row_dict


@dataclass(frozen=True)
class VariableSpec:
    """Input variable metadata for an oracle equation."""

    name: str
    bounds: tuple[float, float]
    dim: tuple[float, ...]


@dataclass(frozen=True)
class ConstantSpec:
    """Fixed constant exposed as a virtual input variable."""

    name: str
    value: float
    dim: tuple[float, ...]


@dataclass(frozen=True)
class EquationSpec:
    """Ground-truth equation specification consumed by the oracle lab."""

    id: str
    basis: tuple[str, ...]
    variables: tuple[VariableSpec, ...]
    constants: tuple[ConstantSpec, ...]
    target_expr: str
    target_dim: tuple[float, ...]


def _as_finite_float(v: Any, *, where: str) -> float:
    try:
        f = float(v)
    except Exception as exc:  # pragma: no cover - defensive conversion
        raise ValueError(f"{where}: expected numeric value, got {v!r} ({exc})") from exc
    if not math.isfinite(f):
        raise ValueError(f"{where}: expected finite value, got {v!r}")
    return f


def _parse_dim_vector(raw: Any, *, n_base: int, where: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{where}: expected a list/tuple of exponents")
    if len(raw) != int(n_base):
        raise ValueError(f"{where}: expected {n_base} exponents, got {len(raw)}")
    return tuple(_as_finite_float(x, where=f"{where}[{i}]") for i, x in enumerate(raw))


def _parse_bounds(raw: Any, *, where: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{where}: expected [lo, hi]")
    lo = _as_finite_float(raw[0], where=f"{where}[0]")
    hi = _as_finite_float(raw[1], where=f"{where}[1]")
    if not lo < hi:
        raise ValueError(f"{where}: expected lo < hi, got {lo} >= {hi}")
    return lo, hi


def equation_spec_from_dict(payload: dict[str, Any], *, source: str = "<dict>") -> EquationSpec:
    """Validate and normalize a raw equation manifest into :class:`EquationSpec`."""

    if not isinstance(payload, dict):
        raise ValueError(f"{source}: spec root must be a dict")

    basis_raw = payload.get("basis", None)
    if not isinstance(basis_raw, (list, tuple)) or len(basis_raw) == 0:
        raise ValueError(f"{source}: 'basis' must be a non-empty list")
    basis = tuple(str(x) for x in basis_raw)
    n_base = len(basis)

    spec_id = str(payload.get("id", "")).strip()
    if spec_id == "":
        raise ValueError(f"{source}: missing non-empty 'id'")

    vars_raw = payload.get("variables", None)
    if not isinstance(vars_raw, (list, tuple)) or len(vars_raw) == 0:
        raise ValueError(f"{source}: 'variables' must be a non-empty list")

    variables: list[VariableSpec] = []
    for i, row in enumerate(vars_raw):
        where = f"{source}.variables[{i}]"
        if not isinstance(row, dict):
            raise ValueError(f"{where}: expected dict")
        name = str(row.get("name", "")).strip()
        if name == "":
            raise ValueError(f"{where}: missing non-empty 'name'")
        bounds = _parse_bounds(row.get("bounds", None), where=f"{where}.bounds")
        dim = _parse_dim_vector(row.get("dim", None), n_base=n_base, where=f"{where}.dim")
        variables.append(VariableSpec(name=name, bounds=bounds, dim=dim))

    constants_raw = payload.get("constants", [])
    if constants_raw is None:
        constants_raw = []
    if not isinstance(constants_raw, (list, tuple)):
        raise ValueError(f"{source}: 'constants' must be a list when provided")

    constants: list[ConstantSpec] = []
    for i, row in enumerate(constants_raw):
        where = f"{source}.constants[{i}]"
        if not isinstance(row, dict):
            raise ValueError(f"{where}: expected dict")
        name = str(row.get("name", "")).strip()
        if name == "":
            raise ValueError(f"{where}: missing non-empty 'name'")
        value = _as_finite_float(row.get("value", None), where=f"{where}.value")
        dim = _parse_dim_vector(row.get("dim", None), n_base=n_base, where=f"{where}.dim")
        constants.append(ConstantSpec(name=name, value=value, dim=dim))

    target_raw = payload.get("target", None)
    if not isinstance(target_raw, dict):
        raise ValueError(f"{source}: 'target' must be a dict")
    target_expr = str(target_raw.get("expr", "")).strip()
    if target_expr == "":
        raise ValueError(f"{source}: target.expr must be non-empty")
    target_dim = _parse_dim_vector(target_raw.get("dim", None), n_base=n_base, where=f"{source}.target.dim")

    all_names = [v.name for v in variables] + [c.name for c in constants]
    if len(all_names) != len(set(all_names)):
        raise ValueError(f"{source}: variable/constant names must be unique")

    return EquationSpec(
        id=spec_id,
        basis=basis,
        variables=tuple(variables),
        constants=tuple(constants),
        target_expr=target_expr,
        target_dim=target_dim,
    )


def load_equation_spec(path: str | pathlib.Path) -> EquationSpec:
    """Load an equation spec from JSON or YAML file."""

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

    return equation_spec_from_dict(payload, source=str(p))


def _all_symbol_names(spec: EquationSpec) -> list[str]:
    return [v.name for v in spec.variables] + [c.name for c in spec.constants]


def compile_target_expression(spec: EquationSpec) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compile ``spec.target_expr`` into a torch callable ``f(X)->[N,1]``.

    The callable expects columns in this exact order:
    ``variables`` first, then ``constants``.
    """

    try:
        import sympy as sp
    except Exception as exc:
        raise RuntimeError(
            "compile_target_expression requires sympy. Install sympy>=1.12 to use oracle specs."
        ) from exc

    names = _all_symbol_names(spec)
    sym_map = {nm: sp.Symbol(nm, real=True) for nm in names}

    local_scope = {
        **sym_map,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }

    expr = sp.sympify(spec.target_expr, locals=local_scope)
    unknown = sorted(str(s) for s in expr.free_symbols if str(s) not in sym_map)
    if unknown:
        raise ValueError(
            f"Equation '{spec.id}' uses unknown symbol(s): {unknown}. "
            f"Declared names: {names}"
        )

    sym_order = [sym_map[nm] for nm in names]
    # Custom torch module with missing functions and constant-safe wrappers.
    # Use "numpy" printer to avoid sympy's TorchPrinter rejecting arcsin etc.
    import math as _math

    def _safe_tensor(x):
        return x if torch.is_tensor(x) else torch.tensor(float(x), dtype=torch.float64)

    _torch_extras = {
        "pi": torch.tensor(_math.pi, dtype=torch.float64),
        "Abs": torch.abs,
        # sin/cos/tan resolve to numpy by default (fine for value evaluation but
        # not autograd-differentiable); map them to torch so the compiled target
        # supports gradients/Hessians (used by the GS carrier-seed bridge).
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "asin": torch.asin,
        "acos": torch.acos,
        "atan": torch.atan,
        "atan2": torch.atan2,
        "arcsin": torch.asin,
        "arccos": torch.acos,
        "arctan": torch.atan,
        "tanh": torch.tanh,
        "cosh": torch.cosh,
        "sinh": torch.sinh,
        "sign": torch.sign,
        "sqrt": lambda x: torch.sqrt(_safe_tensor(x)),
        "log": lambda x: torch.log(_safe_tensor(x)),
        "exp": lambda x: torch.exp(_safe_tensor(x)),
        "pow": lambda base, exp: torch.pow(_safe_tensor(base), _safe_tensor(exp)),
        "Pow": lambda base, exp: torch.pow(_safe_tensor(base), _safe_tensor(exp)),
    }
    fn = sp.lambdify(sym_order, expr, modules=[_torch_extras, "numpy"])

    def _target_fn(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"target_fn expects rank-2 tensor [N,D], got shape={tuple(x.shape)}")
        if x.shape[1] != len(sym_order):
            raise ValueError(
                f"target_fn expects D={len(sym_order)} columns, got {x.shape[1]}"
            )

        cols = [x[:, i] for i in range(x.shape[1])]
        y_raw = fn(*cols)

        if torch.is_tensor(y_raw):
            y = y_raw.to(dtype=x.dtype, device=x.device)
        else:
            y = torch.as_tensor(y_raw, dtype=x.dtype, device=x.device)

        if y.ndim == 0:
            y = y.expand(x.shape[0])
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        elif y.ndim == 2 and y.shape[1] == 1:
            pass
        else:
            raise ValueError(f"target_fn produced unexpected shape {tuple(y.shape)}")

        return y

    return _target_fn


def sample_oracle_inputs(
    spec: EquationSpec,
    n_points: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample oracle inputs and append constant columns as fixed virtual variables."""

    n_var = len(spec.variables)
    lo = torch.tensor([v.bounds[0] for v in spec.variables], dtype=dtype).reshape(1, n_var)
    hi = torch.tensor([v.bounds[1] for v in spec.variables], dtype=dtype).reshape(1, n_var)

    u = torch.rand((int(n_points), n_var), generator=generator, dtype=dtype)
    x = lo + (hi - lo) * u

    if spec.constants:
        cols = [
            torch.full((int(n_points), 1), float(c.value), dtype=dtype)
            for c in spec.constants
        ]
        x = torch.cat([x, *cols], dim=1)

    return x


def build_oracle_dataset(
    spec: EquationSpec,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_fit: int,
    n_probe: int,
    seed: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Build fit/probe datasets and unit vectors for the oracle search."""

    g_fit = torch.Generator(device="cpu").manual_seed(int(seed))
    g_probe = torch.Generator(device="cpu").manual_seed(int(seed) + 1_000_003)

    x_fit = sample_oracle_inputs(spec, n_fit, generator=g_fit, dtype=dtype)
    x_probe = sample_oracle_inputs(spec, n_probe, generator=g_probe, dtype=dtype)

    y_fit = target_fn(x_fit)
    y_probe = target_fn(x_probe)

    if y_fit.ndim != 2 or y_fit.shape[1] != 1:
        raise ValueError(f"target_fn returned invalid y_fit shape {tuple(y_fit.shape)}")
    if y_probe.ndim != 2 or y_probe.shape[1] != 1:
        raise ValueError(f"target_fn returned invalid y_probe shape {tuple(y_probe.shape)}")
    if not torch.isfinite(y_fit).all() or not torch.isfinite(y_probe).all():
        raise ValueError("target_fn produced non-finite values on sampled domain")

    var_dims = [tuple(v.dim) for v in spec.variables] + [tuple(c.dim) for c in spec.constants]
    y_dims = tuple(spec.target_dim)

    return {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_probe": x_probe,
        "y_probe": y_probe,
        "var_dims": var_dims,
        "y_dims": y_dims,
    }


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


def default_oracle_hyperparams() -> FactorizedSearchConfig:
    """Return factorized symbolic search hyperparameters tuned for lightweight oracle experiments."""

    hp = FactorizedSearchConfig()
    hp.search_profile = "oracle_default"
    hp.n_seeds = 1
    hp.split_iter_across_seeds = True
    return hp


def run_oracle_equation(
    spec: EquationSpec,
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    print_every: int | None = None,
    verbose: bool = True,
    oracle_dataset: Mapping[str, Any] | None = None,
    gs_carrier_seed: bool = False,
    gs_carrier_seed_cfg: Any | None = None,
) -> dict[str, Any]:
    """Run factorized symbolic search/continuous skeleton refinement directly on a ground-truth equation spec.

    When ``gs_carrier_seed`` is set, the generalized-symmetry layer first
    discovers the internal coordinate(s) ``z(x)`` of the target and hands them
    to the engine's carrier-seed phase, so the outer-map battery fits ``g(z)``
    directly.  Default off => baseline behaviour unchanged.
    """

    hp = factorized_search_hp if factorized_search_hp is not None else default_oracle_hyperparams()
    run_seed = int(hp.seed if seed is None else seed)

    target_fn = compile_target_expression(spec)
    if oracle_dataset is None:
        ds = build_oracle_dataset(
            spec,
            target_fn,
            n_fit=int(hp.n_fit),
            n_probe=int(hp.n_probe),
            seed=run_seed,
            dtype=dtype,
        )
        dataset_metadata: dict[str, Any] = {
            "source": "synthetic",
            "seed": int(run_seed),
            "n_fit": int(hp.n_fit),
            "n_probe": int(hp.n_probe),
        }
    else:
        ds = dict(oracle_dataset)
        required = ("x_fit", "y_fit", "x_probe", "y_probe", "var_dims", "y_dims")
        missing = [key for key in required if key not in ds]
        if missing:
            raise ValueError(f"oracle_dataset missing required key(s): {missing}")
        for key in ("x_fit", "y_fit", "x_probe", "y_probe"):
            tensor = ds[key]
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor, dtype=dtype)
            ds[key] = tensor.to(dtype=dtype)
        if ds["x_fit"].ndim != 2 or ds["x_probe"].ndim != 2:
            raise ValueError("oracle_dataset x_fit/x_probe must be rank-2 tensors")
        if ds["y_fit"].ndim == 1:
            ds["y_fit"] = ds["y_fit"].reshape(-1, 1)
        if ds["y_probe"].ndim == 1:
            ds["y_probe"] = ds["y_probe"].reshape(-1, 1)
        if ds["y_fit"].ndim != 2 or ds["y_fit"].shape[1] != 1:
            raise ValueError("oracle_dataset y_fit must have shape [N,1]")
        if ds["y_probe"].ndim != 2 or ds["y_probe"].shape[1] != 1:
            raise ValueError("oracle_dataset y_probe must have shape [N,1]")
        if ds["x_fit"].shape[0] != ds["y_fit"].shape[0]:
            raise ValueError("oracle_dataset x_fit and y_fit row counts differ")
        if ds["x_probe"].shape[0] != ds["y_probe"].shape[0]:
            raise ValueError("oracle_dataset x_probe and y_probe row counts differ")
        if ds["x_fit"].shape[1] != ds["x_probe"].shape[1]:
            raise ValueError("oracle_dataset x_fit and x_probe column counts differ")
        for key in ("x_fit", "y_fit", "x_probe", "y_probe"):
            if not torch.isfinite(ds[key]).all():
                raise ValueError(f"oracle_dataset {key} contains non-finite values")
        dataset_metadata = dict(ds.get("metadata", {}) or {})
        dataset_metadata.setdefault("source", "external")
        dataset_metadata.setdefault("n_fit", int(ds["x_fit"].shape[0]))
        dataset_metadata.setdefault("n_probe", int(ds["x_probe"].shape[0]))

    var_dims = ds["var_dims"] if enforce_dims else None
    y_dims = ds["y_dims"] if enforce_dims else None
    nvars = int(ds["x_fit"].shape[1])

    # Generalized-symmetry carrier seeds (opt-in): discover the internal
    # coordinate(s) z(x) once and hand them to the engine's carrier-seed phase
    # so the outer-map battery fits g(z) directly. Empty => no-op.
    gs_carrier_seed_exprs: tuple = ()
    gs_carrier_seed_diag: list[dict[str, Any]] = []
    if gs_carrier_seed:
        try:
            from .gs_carrier_seed import discover_gs_carrier_seeds

            seeds, gs_carrier_seed_diag = discover_gs_carrier_seeds(
                target_fn, ds["x_fit"], n_var=len(spec.variables), cfg=gs_carrier_seed_cfg,
            )
            gs_carrier_seed_exprs = tuple(
                {
                    "expr": seed_expr,
                    "metadata": dict(diag.get("carrier_metadata") or {}),
                }
                for seed_expr, diag in zip(seeds, gs_carrier_seed_diag)
            )
            if verbose:
                print(
                    f"[gs-carrier-seed] discovered {len(gs_carrier_seed_exprs)} GS coordinate(s): "
                    + ", ".join(str(d.get("z_human", "")) for d in gs_carrier_seed_diag)
                )
        except Exception as exc:
            if verbose:
                print(f"[gs-carrier-seed] discovery failed: {type(exc).__name__}: {exc}")
            gs_carrier_seed_exprs, gs_carrier_seed_diag = (), []

    n_seeds = max(1, int(hp.n_seeds))
    if bool(hp.split_iter_across_seeds) and n_seeds > 1:
        n_iter_each = max(1, int(hp.n_iter) // n_seeds)
    else:
        n_iter_each = int(hp.n_iter)
    early_stop_mse = float(hp.early_stop_mse)
    try:
        wall_time_limit_s = None if getattr(hp, "wall_time_limit_s", None) is None else float(hp.wall_time_limit_s)
        if wall_time_limit_s is not None and (not math.isfinite(wall_time_limit_s) or wall_time_limit_s <= 0.0):
            wall_time_limit_s = None
    except Exception:
        wall_time_limit_s = None

    started = time.perf_counter()
    wall_time_deadline = None if wall_time_limit_s is None else (started + float(wall_time_limit_s))
    raw_rows: list[dict[str, Any]] = []
    action_counts_total: dict[str, int] = {}
    hole_search_stats_total: dict[str, Any] = {
        "prepare_calls": 0,
        "prepared_executable_checks": 0,
        "prepared_resolution_live_archive": 0,
        "prepared_resolution_snapshot": 0,
        "prepared_resolution_missing": 0,
        "prepare_prune_wall_seconds": 0.0,
        "prepare_mine_wall_seconds": 0.0,
        "prepare_select_wall_seconds": 0.0,
        "prepare_wall_seconds": 0.0,
        "selected": 0,
        "fired": 0,
        "ingested": 0,
        "mined": 0,
        "selected_with_any_frontier_entries": 0,
        "selected_with_nonempty_frontier": 0,
        "selected_with_opportunity": 0,
        "selected_with_resolved_parent": 0,
        "invalidated_parent": 0,
        "run_hole_search_action_called": 0,
        "child_expr_none": 0,
        "frontier_size": 0,
        "last_mined_iter": None,
        "best_eff_mse": None,
    }
    score_prescreen_stats_total: dict[str, Any] = {
        "prescore_calls": 0,
        "prescore_promoted": 0,
        "prescore_dropped": 0,
        "prescore_promoted_by_hint": 0,
        "prescore_promoted_by_parent_threshold": 0,
        "prescore_promoted_by_global_best_threshold": 0,
        "full_score_calls": 0,
        "full_score_calls_by_action": {},
    }
    route_scheduler_stats_total: dict[str, Any] = {
        "considered": 0,
        "opportunity_available": 0,
        "selected_expression_expand": 0,
        "selected_opportunity_expand": 0,
        "selection_forced": 0,
        "selection_epsilon": 0,
        "selection_ucb": 0,
        "model_scored": 0,
        "model_trained": 0,
        "reward_count": 0,
        "reward_sum": 0.0,
        "reward_sum_raw": 0.0,
        "reward_sum_adjusted": 0.0,
        "wall_seconds_sum": 0.0,
        "reward_mode": "",
        "time_penalty": 0.0,
        "time_floor": 0.0,
        "route_summary": {},
    }
    refine_diagnostics_total: dict[str, Any] = {}
    refine_diagnostics_by_seed: list[dict[str, Any]] = []
    refine_slate_stats_by_seed: list[dict[str, Any]] = []
    gs_carrier_unit_stats_total: dict[str, int] = {}
    gs_carrier_unit_stats_by_seed: list[dict[str, Any]] = []
    n_seeds_ran = 0
    search_stop_reasons: list[str] = []
    wall_time_limit_hit = False
    arch = None

    for i in range(n_seeds):
        if wall_time_deadline is not None:
            remaining_wall_time_s = float(wall_time_deadline - time.perf_counter())
            if remaining_wall_time_s <= 0.0:
                wall_time_limit_hit = True
                search_stop_reasons.append("wall_time_limit_before_seed")
                break
        else:
            remaining_wall_time_s = None
        n_seeds_ran = int(i) + 1
        seed_search = run_seed + i
        if verbose:
            print(
                f"[oracle] seed {int(i)+1}/{int(n_seeds)} "
                f"seed_search={int(seed_search)} n_iter={int(n_iter_each)}"
            )
        core_kwargs = dict(
            target_fn=target_fn,
            nvars=nvars,
            n_iter=n_iter_each,
            wall_time_limit_s=remaining_wall_time_s,
            max_depth=int(hp.max_depth),
            poly_degree=int(hp.poly_degree),
            lo=0.0,
            hi=1.0,
            seed=run_seed,
            seed_search=seed_search,
            var_dims=var_dims,
            y_dims=y_dims,
            dtype=dtype,
            x_fit_data=ds["x_fit"],
            y_fit_data=ds["y_fit"],
            x_probe_data=ds["x_probe"],
            y_probe_data=ds["y_probe"],
            carrier_seed_exprs=gs_carrier_seed_exprs,
            brute_depth=hp.brute_depth,
            early_stop_mse=early_stop_mse,
            brute_max_expressions=int(hp.brute_max_expressions),
            score_mapping_family_mode=str(getattr(hp, "score_mapping_family_mode", "full")),
            brute_score_mapping_family_mode=str(getattr(hp, "brute_score_mapping_family_mode", "gated")),
            score_mapping_expensive_gate_best_factor=float(getattr(hp, "score_mapping_expensive_gate_best_factor", 5.0)),
            score_mapping_expensive_rel_y=float(getattr(hp, "score_mapping_expensive_rel_y", 0.10)),
            score_prescreen_enable=bool(getattr(hp, "score_prescreen_enable", True)),
            score_prescreen_family_mode=str(getattr(hp, "score_prescreen_family_mode", "cheap")),
            score_prescreen_residual_family_mode=str(getattr(hp, "score_prescreen_residual_family_mode", "gated")),
            score_prescreen_residual_allow_hint=bool(getattr(hp, "score_prescreen_residual_allow_hint", False)),
            score_prescreen_residual_use_global_best=bool(getattr(hp, "score_prescreen_residual_use_global_best", False)),
            score_prescreen_parent_best_factor=float(getattr(hp, "score_prescreen_parent_best_factor", 1.5)),
            score_prescreen_global_best_factor=float(getattr(hp, "score_prescreen_global_best_factor", 3.0)),
            score_prescreen_residual_parent_best_factor=float(getattr(hp, "score_prescreen_residual_parent_best_factor", 1.1)),
            score_prescreen_residual_global_best_factor=float(getattr(hp, "score_prescreen_residual_global_best_factor", 1.5)),
            no_residual=bool(getattr(hp, "no_residual", False)),
            inverse_steering_enable=bool(getattr(hp, "inverse_steering_enable", False)),
            inverse_max_paths=int(getattr(hp, "inverse_max_paths", 12)),
            inverse_topk_terms=int(getattr(hp, "inverse_topk_terms", 6)),
            inverse_shortlist_mult=int(getattr(hp, "inverse_shortlist_mult", 4)),
            inverse_min_valid_frac=float(getattr(hp, "inverse_min_valid_frac", 0.25)),
            inverse_min_confidence=float(getattr(hp, "inverse_min_confidence", 0.10)),
            inverse_safe_eps=getattr(hp, "inverse_safe_eps", None),
            inverse_confidence_mode=str(getattr(hp, "inverse_confidence_mode", "conditioning")),
            inverse_confidence_target_gain=float(getattr(hp, "inverse_confidence_target_gain", 4.0)),
            inverse_confidence_floor=float(getattr(hp, "inverse_confidence_floor", 0.05)),
            inverse_branch_beam_width=int(getattr(hp, "inverse_branch_beam_width", 1)),
            inverse_micro_search_enable=bool(getattr(hp, "inverse_micro_search_enable", False)),
            inverse_micro_search_max_depth=int(getattr(hp, "inverse_micro_search_max_depth", 3)),
            inverse_micro_search_beam_width=int(getattr(hp, "inverse_micro_search_beam_width", 24)),
            inverse_micro_search_topk=int(getattr(hp, "inverse_micro_search_topk", 16)),
            inverse_micro_search_seed_terms=int(getattr(hp, "inverse_micro_search_seed_terms", 8)),
            inverse_local_score_mode=str(getattr(hp, "inverse_local_score_mode", "affine")),
            inverse_spec_enable=bool(getattr(hp, "inverse_spec_enable", False)),
            inverse_spec_enum_max_depth=int(getattr(hp, "inverse_spec_enum_max_depth", 4)),
            inverse_spec_enum_max_trees=int(getattr(hp, "inverse_spec_enum_max_trees", 5000)),
            inverse_spec_preview_topk=int(getattr(hp, "inverse_spec_preview_topk", 16)),
            inverse_spec_local_score_mode=str(getattr(hp, "inverse_spec_local_score_mode", "affine")),
            inverse_spec_include_legacy_seed=bool(getattr(hp, "inverse_spec_include_legacy_seed", True)),
            inverse_spec_complexity_penalty=float(getattr(hp, "inverse_spec_complexity_penalty", 0.0)),
            inverse_spec_repair_quota=float(getattr(hp, "inverse_spec_repair_quota", 0.0)),
            repair_pass_enable=bool(getattr(hp, "repair_pass_enable", False)),
            repair_pass_elite_k=int(getattr(hp, "repair_pass_elite_k", 8)),
            repair_pass_paths_per_elite=int(getattr(hp, "repair_pass_paths_per_elite", 2)),
            repair_pass_rounds=int(getattr(hp, "repair_pass_rounds", 2)),
            closure_search_enable=bool(getattr(hp, "closure_search_enable", False)),
            closure_search_families=list(getattr(hp, "closure_search_families", ["periodic", "exp", "log", "rational", "power", "quadratic"])),
            closure_search_max_proposals=int(getattr(hp, "closure_search_max_proposals", 16)),
            closure_search_anchors_per_family=int(getattr(hp, "closure_search_anchors_per_family", 4)),
            closure_search_preview_topk=int(getattr(hp, "closure_search_preview_topk", 4)),
            closure_search_exact_topk=int(getattr(hp, "closure_search_exact_topk", 2)),
            closure_search_beam_width=int(getattr(hp, "closure_search_beam_width", 4)),
            closure_search_seed_exact_topk=int(getattr(hp, "closure_search_seed_exact_topk", 6)),
            closure_search_seed_beam_width=int(getattr(hp, "closure_search_seed_beam_width", 4)),
            closure_search_seed_scaffold_reserve=int(
                getattr(hp, "closure_search_seed_scaffold_reserve", 8)
            ),
            closure_search_seed_family_cap=int(getattr(hp, "closure_search_seed_family_cap", 2)),
            closure_search_seed_exact_bound_bonus=float(
                getattr(hp, "closure_search_seed_exact_bound_bonus", 0.25)
            ),
            closure_search_pair_normal_enable=bool(
                getattr(hp, "closure_search_pair_normal_enable", False)
            ),
            closure_search_pair_normal_topk=int(
                getattr(hp, "closure_search_pair_normal_topk", 3)
            ),
            closure_search_pair_normal_max_pairs=int(
                getattr(hp, "closure_search_pair_normal_max_pairs", 1)
            ),
            closure_search_pair_rescue_enable=bool(
                getattr(hp, "closure_search_pair_rescue_enable", True)
            ),
            closure_search_pair_rescue_topk=int(
                getattr(hp, "closure_search_pair_rescue_topk", 4)
            ),
            closure_search_pair_rescue_max_pairs=int(
                getattr(hp, "closure_search_pair_rescue_max_pairs", 6)
            ),
            closure_search_emergent_basis_enable=bool(
                getattr(hp, "closure_search_emergent_basis_enable", False)
            ),
            closure_search_emergent_basis_max_source_rows=int(
                getattr(hp, "closure_search_emergent_basis_max_source_rows", 32)
            ),
            closure_search_emergent_basis_score_topk=int(
                getattr(hp, "closure_search_emergent_basis_score_topk", 8)
            ),
            closure_search_emergent_basis_max_per_round=int(
                getattr(hp, "closure_search_emergent_basis_max_per_round", 1)
            ),
            closure_search_emergent_basis_max_total=int(
                getattr(hp, "closure_search_emergent_basis_max_total", 4)
            ),
            closure_search_emergent_basis_min_probe_gain_rel=float(
                getattr(hp, "closure_search_emergent_basis_min_probe_gain_rel", 5.0e-3)
            ),
            closure_search_emergent_aux_atoms_enable=bool(
                getattr(hp, "closure_search_emergent_aux_atoms_enable", False)
            ),
            closure_search_emergent_aux_atoms_max_source_rows=int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_source_rows", 48)
            ),
            closure_search_emergent_aux_atoms_max_new_per_round=int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_new_per_round", 5)
            ),
            closure_search_emergent_aux_atoms_max_total=int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_total", 8)
            ),
            closure_search_emergent_aux_atoms_max_target=int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_target", 4)
            ),
            closure_search_emergent_aux_atoms_max_dimensionless=int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_dimensionless", 3)
            ),
            closure_search_emergent_aux_atoms_max_rational_derived=int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_rational_derived", 2)
            ),
            closure_search_emergent_aux_atoms_max_seed_blocks=int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_seed_blocks", 8)
            ),
            closure_search_debug_topk=int(
                getattr(hp, "closure_search_debug_topk", 0)
            ),
            closure_search_min_valid_frac=float(getattr(hp, "closure_search_min_valid_frac", 0.05)),
            closure_search_min_confidence=float(getattr(hp, "closure_search_min_confidence", 0.02)),
            closure_search_periodic_min_valid_scale=float(
                getattr(hp, "closure_search_periodic_min_valid_scale", 1.0)
            ),
            closure_search_periodic_min_confidence_scale=float(
                getattr(hp, "closure_search_periodic_min_confidence_scale", 1.0)
            ),
            closure_search_transport_min_lin_rel=float(
                getattr(hp, "closure_search_transport_min_lin_rel", 0.0)
            ),
            closure_search_anchor_head_compare_enable=bool(
                getattr(hp, "closure_search_anchor_head_compare_enable", False)
            ),
            hole_search_enable=bool(getattr(hp, "hole_search_enable", False)),
            hole_search_quota=float(getattr(hp, "hole_search_quota", 0.10)),
            hole_search_exact_budget=int(getattr(hp, "hole_search_exact_budget", 2)),
            hole_search_cooldown_iters=int(getattr(hp, "hole_search_cooldown_iters", 32)),
            hole_search_mine_cooldown_iters=int(getattr(hp, "hole_search_mine_cooldown_iters", 50)),
            hole_search_max_frontier=int(getattr(hp, "hole_search_max_frontier", 128)),
            hole_search_first_class_scheduler_enable=bool(getattr(hp, "hole_search_first_class_scheduler_enable", True)),
            hole_search_route_scheduler_enable=bool(getattr(hp, "hole_search_route_scheduler_enable", True)),
            hole_search_route_ucb_c=float(getattr(hp, "hole_search_route_ucb_c", 0.25)),
            hole_search_route_eps=float(getattr(hp, "hole_search_route_eps", 0.05)),
            hole_search_route_acquisition_weight=float(getattr(hp, "hole_search_route_acquisition_weight", 0.25)),
            hole_search_route_reward_mode=str(getattr(hp, "hole_search_route_reward_mode", "penalized")),
            hole_search_route_time_penalty=float(getattr(hp, "hole_search_route_time_penalty", 0.01)),
            hole_search_route_time_floor=float(getattr(hp, "hole_search_route_time_floor", 1.0)),
            hole_search_abstraction_enable=bool(getattr(hp, "hole_search_abstraction_enable", True)),
            hole_search_abstraction_on_improve=bool(getattr(hp, "hole_search_abstraction_on_improve", True)),
            hole_search_abstraction_on_stall=bool(getattr(hp, "hole_search_abstraction_on_stall", True)),
            hole_search_abstraction_cooldown_iters=int(getattr(hp, "hole_search_abstraction_cooldown_iters", 25)),
            hole_search_abstraction_max_parents=int(getattr(hp, "hole_search_abstraction_max_parents", 2)),
            hole_search_abstraction_max_paths_per_parent=int(getattr(hp, "hole_search_abstraction_max_paths_per_parent", 3)),
            hole_search_abstraction_improve_min_delta_log_mse=float(getattr(hp, "hole_search_abstraction_improve_min_delta_log_mse", 0.15)),
            hole_search_abstraction_stage_enable=bool(getattr(hp, "hole_search_abstraction_stage_enable", True)),
            hole_search_abstraction_stage_max_entries=int(getattr(hp, "hole_search_abstraction_stage_max_entries", 64)),
            hole_search_abstraction_promote_topk=int(getattr(hp, "hole_search_abstraction_promote_topk", 2)),
            hole_search_abstraction_promote_frontier_floor=int(getattr(hp, "hole_search_abstraction_promote_frontier_floor", 3)),
            hole_search_enum_max_depth=int(getattr(hp, "hole_search_enum_max_depth", 4)),
            hole_search_enum_max_trees=int(getattr(hp, "hole_search_enum_max_trees", 3000)),
            hole_search_preview_topk=int(getattr(hp, "hole_search_preview_topk", 8)),
            hole_search_tournament_enable=bool(getattr(hp, "hole_search_tournament_enable", True)),
            hole_search_tournament_n=int(getattr(hp, "hole_search_tournament_n", 8)),
            hole_search_tournament_elite_k=int(getattr(hp, "hole_search_tournament_elite_k", 2)),
            hole_search_tournament_preview_trees=int(getattr(hp, "hole_search_tournament_preview_trees", 64)),
            inverse_spec_recursive_enable=bool(getattr(hp, "inverse_spec_recursive_enable", True)),
            inverse_spec_recursive_max_depth=int(getattr(hp, "inverse_spec_recursive_max_depth", 2)),
            inverse_spec_recursive_trigger_rel_mse=float(getattr(hp, "inverse_spec_recursive_trigger_rel_mse", 0.25)),
            inverse_spec_recursive_seed_cap=int(getattr(hp, "inverse_spec_recursive_seed_cap", 6)),
            inverse_spec_recursive_branch_topk=int(getattr(hp, "inverse_spec_recursive_branch_topk", 4)),
            inverse_spec_recursive_child_topk=int(getattr(hp, "inverse_spec_recursive_child_topk", 2)),
            inverse_spec_max_subtree_depth=getattr(hp, "inverse_spec_max_subtree_depth", None),
            inverse_spec_fit_cap=int(getattr(hp, "inverse_spec_fit_cap", 96)),
            inverse_spec_probe_cap=int(getattr(hp, "inverse_spec_probe_cap", 192)),
            inverse_spec_exact_budget=int(getattr(hp, "inverse_spec_exact_budget", 4)),
            inverse_target_mode=str(getattr(hp, "inverse_target_mode", "robust")),
            inverse_full_mapping_penalty=float(getattr(hp, "inverse_full_mapping_penalty", 0.75)),
            inverse_exact_simple_target_bonus=float(getattr(hp, "inverse_exact_simple_target_bonus", 0.10)),
            inverse_additive_descend_penalty=float(getattr(hp, "inverse_additive_descend_penalty", 0.15)),
            inverse_nonadditive_leaf_penalty=float(getattr(hp, "inverse_nonadditive_leaf_penalty", 0.20)),
            inverse_exact_path_eta=float(getattr(hp, "inverse_exact_path_eta", 0.98)),
            inverse_exact_transport_min_lin_rel=float(getattr(hp, "inverse_exact_transport_min_lin_rel", 0.0)),
            inverse_gate_enable=bool(getattr(hp, "inverse_gate_enable", True)),
            inverse_gate_warmup=int(getattr(hp, "inverse_gate_warmup", 0)),
            inverse_gate_best_factor=float(getattr(hp, "inverse_gate_best_factor", 20.0)),
            inverse_gate_min_residual_basins=int(getattr(hp, "inverse_gate_min_residual_basins", 0)),
            inverse_gate_min_depth=int(getattr(hp, "inverse_gate_min_depth", 4)),
            inverse_gate_min_size=int(getattr(hp, "inverse_gate_min_size", 6)),
            inverse_gate_max_paths=int(getattr(hp, "inverse_gate_max_paths", 6)),
            inverse_gate_min_structural_score=float(getattr(hp, "inverse_gate_min_structural_score", 0.75)),
            inverse_gate_min_weighted_rel_gain=float(getattr(hp, "inverse_gate_min_weighted_rel_gain", 0.05)),
            inverse_gate_structural_bias=float(getattr(hp, "inverse_gate_structural_bias", 0.20)),
            inverse_periodic_min_valid_scale=float(getattr(hp, "inverse_periodic_min_valid_scale", 1.25)),
            inverse_periodic_min_confidence_scale=float(getattr(hp, "inverse_periodic_min_confidence_scale", 1.35)),
            inverse_periodic_path_penalty=float(getattr(hp, "inverse_periodic_path_penalty", 0.65)),
            inverse_nonperiodic_muldiv_bonus=float(getattr(hp, "inverse_nonperiodic_muldiv_bonus", 0.10)),
            inverse_nonperiodic_explogsqrt_bonus=float(getattr(hp, "inverse_nonperiodic_explogsqrt_bonus", 0.05)),
            inverse_branch_ambiguity_penalty=float(getattr(hp, "inverse_branch_ambiguity_penalty", 0.50)),
            inverse_transport_min_lin_rel=float(getattr(hp, "inverse_transport_min_lin_rel", 0.02)),
            inverse_transport_min_effective_n=float(getattr(hp, "inverse_transport_min_effective_n", 8.0)),
            inverse_experiment_log_enable=bool(getattr(hp, "inverse_experiment_log_enable", False)),
            repair_controller_enable=bool(getattr(hp, "repair_controller_enable", False)),
            repair_controller_min_score=float(getattr(hp, "repair_controller_min_score", 0.15)),
            repair_controller_steps=int(getattr(hp, "repair_controller_steps", 3)),
            repair_controller_ancestor_hops=int(getattr(hp, "repair_controller_ancestor_hops", 1)),
            repair_controller_min_step_rel_improve=float(getattr(hp, "repair_controller_min_step_rel_improve", 1.0e-3)),
            repair_controller_adaptive=bool(getattr(hp, "repair_controller_adaptive", True)),
            repair_controller_adapt_quantile=float(getattr(hp, "repair_controller_adapt_quantile", 0.75)),
            repair_controller_adapt_window=int(getattr(hp, "repair_controller_adapt_window", 128)),
            repair_controller_adapt_min_samples=int(getattr(hp, "repair_controller_adapt_min_samples", 16)),
            repair_controller_min_concentration=float(getattr(hp, "repair_controller_min_concentration", 0.30)),
            repair_controller_potential_weight=float(getattr(hp, "repair_controller_potential_weight", 1.00)),
            repair_controller_concentration_weight=float(getattr(hp, "repair_controller_concentration_weight", 0.35)),
            repair_controller_contrast_weight=float(getattr(hp, "repair_controller_contrast_weight", 0.20)),
            repair_controller_cost_weight=float(getattr(hp, "repair_controller_cost_weight", 0.10)),
            repair_controller_stagnation_weight=float(getattr(hp, "repair_controller_stagnation_weight", 0.15)),
            repair_controller_frontier_topk=int(getattr(hp, "repair_controller_frontier_topk", 24)),
            repair_controller_stagnation_visits=int(getattr(hp, "repair_controller_stagnation_visits", 8)),
            repair_controller_focus_prob=float(getattr(hp, "repair_controller_focus_prob", 0.50)),
            repair_controller_parent_max_repeats=int(getattr(hp, "repair_controller_parent_max_repeats", 2)),
            repair_controller_parent_min_eval_gap=int(getattr(hp, "repair_controller_parent_min_eval_gap", 32)),
            repair_controller_parent_reset_rel_improve=float(getattr(hp, "repair_controller_parent_reset_rel_improve", 0.05)),
            repair_controller_critic_enable=bool(getattr(hp, "repair_controller_critic_enable", False)),
            repair_controller_critic_path=str(getattr(hp, "repair_controller_critic_path", "")),
            repair_controller_critic_blend=float(getattr(hp, "repair_controller_critic_blend", 1.0)),
            repair_controller_critic_mode=str(getattr(hp, "repair_controller_critic_mode", "priority")),
            repair_opportunity_controller_enable=bool(getattr(hp, "repair_opportunity_controller_enable", False)),
            repair_opportunity_controller_path=str(getattr(hp, "repair_opportunity_controller_path", "")),
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
            refine_slate_budget=int(getattr(hp, "refine_slate_budget", 32)),
            refine_optimizer=str(getattr(hp, "refine_optimizer", "lbfgs")),
            refine_lbfgs_escalate_improve_factor=float(
                getattr(hp, "refine_lbfgs_escalate_improve_factor", 2.0)
            ),
            refine_lbfgs_steps=int(hp.refine_lbfgs_steps),
            refine_fit_subset=int(hp.refine_fit_subset),
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
            stall_window=int(hp.stall_window),
            stall_patience=int(hp.stall_patience),
            stall_delta=float(hp.stall_delta),
            degenerate_abort_enable=bool(getattr(hp, "degenerate_abort_enable", True)),
            degenerate_abort_min_evals=int(getattr(hp, "degenerate_abort_min_evals", 1000)),
            degenerate_abort_max_accepted=int(getattr(hp, "degenerate_abort_max_accepted", 8)),
            verbose=bool(verbose),
        )
        if print_every is not None:
            core_kwargs["print_every"] = int(print_every)
        core_kwargs.setdefault(
            "_runtime_hooks",
            {**make_engine_runtime_hooks(), **make_engine_refinement_hooks()},
        )

        if verbose:
            arch = run_explorer_core(**core_kwargs)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                arch = run_explorer_core(**core_kwargs)
        seed_stop_reason = str(getattr(arch, "search_stop_reason", "") or "")
        if seed_stop_reason:
            search_stop_reasons.append(seed_stop_reason)
        if bool(getattr(arch, "search_wall_time_limit_hit", False)):
            wall_time_limit_hit = True
        carrier_unit_stats = getattr(arch, "gs_carrier_unit_stats", None)
        if isinstance(carrier_unit_stats, dict):
            gs_carrier_unit_stats_by_seed.append(
                {
                    "seed_search": int(seed_search),
                    **dict(_to_jsonable(carrier_unit_stats)),
                }
            )
            for key, value in carrier_unit_stats.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    continue
                gs_carrier_unit_stats_total[str(key)] = int(
                    gs_carrier_unit_stats_total.get(str(key), 0)
                ) + max(0, int(value))
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

        action_dist = getattr(arch, "action_distribution", None)
        if isinstance(action_dist, dict):
            counts = action_dist.get("counts", {})
            if isinstance(counts, dict):
                for name, value in counts.items():
                    try:
                        iv = int(value)
                    except Exception:
                        continue
                    action_counts_total[str(name)] = int(action_counts_total.get(str(name), 0)) + max(iv, 0)
        hole_stats = getattr(arch, "hole_search_stats", None)
        if isinstance(hole_stats, dict):
            for key, value in hole_stats.items():
                if key == "best_eff_mse":
                    try:
                        if value is not None and math.isfinite(float(value)):
                            cur_best = hole_search_stats_total.get("best_eff_mse", None)
                            if cur_best is None or float(value) < float(cur_best):
                                hole_search_stats_total["best_eff_mse"] = float(value)
                    except Exception:
                        pass
                    continue
                if str(key).endswith("_wall_seconds"):
                    try:
                        if value is None:
                            continue
                        fv = float(value)
                        if not math.isfinite(fv) or fv < 0.0:
                            continue
                        hole_search_stats_total[key] = float(hole_search_stats_total.get(key, 0.0)) + fv
                    except Exception:
                        pass
                    continue
                if key in ("frontier_size", "last_mined_iter"):
                    try:
                        if value is None:
                            continue
                        if key == "frontier_size":
                            hole_search_stats_total[key] = max(
                                int(hole_search_stats_total.get(key, 0)),
                                max(0, int(value)),
                            )
                        else:
                            cur_val = hole_search_stats_total.get(key, None)
                            if cur_val is None:
                                cur_val = -1
                            hole_search_stats_total[key] = max(int(cur_val), int(value))
                    except Exception:
                        pass
                    continue
                try:
                    hole_search_stats_total[key] = int(hole_search_stats_total.get(key, 0)) + max(
                        0, int(value)
                    )
                except Exception:
                    continue
        prescreen_stats = getattr(arch, "score_prescreen_stats", None)
        if isinstance(prescreen_stats, dict):
            for key, value in prescreen_stats.items():
                if key == "full_score_calls_by_action":
                    if isinstance(value, dict):
                        bucket = score_prescreen_stats_total.setdefault("full_score_calls_by_action", {})
                        if not isinstance(bucket, dict):
                            bucket = {}
                            score_prescreen_stats_total["full_score_calls_by_action"] = bucket
                        for action_name, action_count in value.items():
                            try:
                                bucket[str(action_name)] = int(bucket.get(str(action_name), 0)) + max(0, int(action_count))
                            except Exception:
                                continue
                    continue
                try:
                    score_prescreen_stats_total[key] = int(score_prescreen_stats_total.get(key, 0)) + max(
                        0, int(value)
                    )
                except Exception:
                    continue
        route_stats = getattr(arch, "route_scheduler_stats", None)
        if isinstance(route_stats, dict):
            for key, value in route_stats.items():
                if key == "route_summary":
                    if isinstance(value, dict):
                        bucket = route_scheduler_stats_total.setdefault("route_summary", {})
                        if not isinstance(bucket, dict):
                            bucket = {}
                            route_scheduler_stats_total["route_summary"] = bucket
                        for route_name, route_row in value.items():
                            if not isinstance(route_row, dict):
                                continue
                            route_bucket = bucket.setdefault(str(route_name), {"count": 0, "q": 0.0})
                            for route_key, route_val in route_row.items():
                                if route_key == "q":
                                    try:
                                        route_bucket["q"] = float(route_val)
                                    except Exception:
                                        pass
                                    continue
                                if isinstance(route_val, bool):
                                    route_bucket[route_key] = bool(route_bucket.get(route_key, False) or bool(route_val))
                                    continue
                                try:
                                    if isinstance(route_val, int) and not isinstance(route_val, bool):
                                        route_bucket[route_key] = int(route_bucket.get(route_key, 0)) + max(0, int(route_val))
                                    else:
                                        route_bucket[route_key] = float(route_bucket.get(route_key, 0.0)) + float(route_val)
                                except Exception:
                                    continue
                    continue
                if key in {"reward_sum", "reward_sum_raw", "reward_sum_adjusted", "wall_seconds_sum"}:
                    try:
                        route_scheduler_stats_total[key] = float(route_scheduler_stats_total.get(key, 0.0)) + float(value)
                    except Exception:
                        pass
                    continue
                if key in {"time_penalty", "time_floor"}:
                    try:
                        if not route_scheduler_stats_total.get(key, 0.0):
                            route_scheduler_stats_total[key] = float(value)
                    except Exception:
                        pass
                    continue
                if key == "reward_mode":
                    if not route_scheduler_stats_total.get(key):
                        route_scheduler_stats_total[key] = str(value or "")
                    continue
                if key == "enabled":
                    route_scheduler_stats_total[key] = bool(route_scheduler_stats_total.get(key, False) or bool(value))
                    continue
                try:
                    route_scheduler_stats_total[key] = int(route_scheduler_stats_total.get(key, 0)) + max(0, int(value))
                except Exception:
                    continue

        seed_best_mse = float("inf")
        input_exprs = None
        for rec in arch.best(int(hp.return_topk)):
            seed_best_mse = min(seed_best_mse, float(rec.best_mse))
            if input_exprs is None:
                input_exprs = [Var(i) for i in range(nvars)]
            embedding_roundtrip = mapping_embedding_roundtrip(
                rec.best_expr,
                rec.mapping,
                input_exprs,
                ds["x_probe"],
                units_mode="raw",
            )
            m = _to_jsonable(rec.mapping)
            score_ladder = m.get("_score_ladder", None) if isinstance(m, dict) else None
            acceptance_basis = str(m.get("_acceptance_basis", "")) if isinstance(m, dict) else ""
            raw_rows.append(
                {
                    "seed_search": int(seed_search),
                    "expr": node_str(rec.best_expr),
                    "expr_ast": _to_jsonable(rec.best_expr),
                    "mse": float(rec.best_mse),
                    "raw_mse": float(getattr(rec, "best_raw_mse", rec.best_mse)),
                    "size": int(len(str(rec.best_expr))),
                    "mapping": m,
                    "mapping_kind": str(m.get("kind", "")) if isinstance(m, dict) else "",
                    "score_ladder": score_ladder,
                    "acceptance_basis": acceptance_basis,
                    "embedding_roundtrip": _to_jsonable(embedding_roundtrip),
                }
            )

        if seed_best_mse < early_stop_mse:
            if verbose:
                print(
                    f"[oracle] early-stop after seed_search={int(seed_search)}: "
                    f"best_mse={seed_best_mse:.3e} < early_stop_mse={early_stop_mse:.3e}"
                )
            break
        if wall_time_limit_hit:
            break

    # Deduplicate by expression + mapping family, keep best MSE.
    pooled: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (str(row["expr"]), str(row.get("mapping_kind", "")))
        prev = pooled.get(key)
        if prev is None or float(row["mse"]) < float(prev["mse"]):
            pooled[key] = row

    rows = sorted(pooled.values(), key=lambda r: float(r["mse"]))[: int(hp.return_topk)]
    elapsed = float(time.perf_counter() - started)
    archive_coherence = {}
    if arch is not None and hasattr(arch, "audit_coherence"):
        try:
            archive_coherence = arch.audit_coherence()
        except Exception as exc:
            archive_coherence = {"ok": False, "error": str(exc)}
    embedding_roundtrip_summary = {
        "checked": int(sum(1 for row in rows if isinstance(row.get("embedding_roundtrip"), dict))),
        "failed": int(
            sum(
                1
                for row in rows
                if isinstance(row.get("embedding_roundtrip"), dict)
                and not bool(row["embedding_roundtrip"].get("ok", False))
            )
        ),
        "max_abs_err": None,
        "max_rel_err": None,
    }
    roundtrip_diags = [
        row.get("embedding_roundtrip")
        for row in rows
        if isinstance(row.get("embedding_roundtrip"), dict)
    ]
    finite_abs = [
        float(diag["max_abs_err"])
        for diag in roundtrip_diags
        if diag.get("max_abs_err") is not None and math.isfinite(float(diag["max_abs_err"]))
    ]
    finite_rel = [
        float(diag["max_rel_err"])
        for diag in roundtrip_diags
        if diag.get("max_rel_err") is not None and math.isfinite(float(diag["max_rel_err"]))
    ]
    if finite_abs:
        embedding_roundtrip_summary["max_abs_err"] = float(max(finite_abs))
    if finite_rel:
        embedding_roundtrip_summary["max_rel_err"] = float(max(finite_rel))

    closure_search_stats_report = _to_jsonable(getattr(arch, "closure_search_stats", None))
    closure_search_summary = None
    closure_search_debug = None
    if isinstance(closure_search_stats_report, dict):
        try:
            atomized_best_probe = float(
                closure_search_stats_report.get("atomized_linear_span_best_probe", float("inf"))
            )
        except Exception:
            atomized_best_probe = float("inf")
        closure_search_summary = {
            "scaffolds_considered": int(closure_search_stats_report.get("scaffolds_considered", 0) or 0),
            "preview_candidates": int(closure_search_stats_report.get("preview_candidates", 0) or 0),
            "scored": int(closure_search_stats_report.get("scored", 0) or 0),
            "new_residual_basins": int(closure_search_stats_report.get("new_residual_basins", 0) or 0),
            "global_best_updates": int(closure_search_stats_report.get("global_best_updates", 0) or 0),
            "closure_search_rounds": int(closure_search_stats_report.get("closure_search_rounds", 0) or 0),
            "controller_stop_reason": str(closure_search_stats_report.get("basis_state_controller_stop_reason", "") or ""),
            "round_commit_scored": int(closure_search_stats_report.get("basis_state_round_commit_scored", 0) or 0),
            "round_commit_accepted": int(closure_search_stats_report.get("basis_state_round_commit_accepted", 0) or 0),
            "round_commit_selected": int(closure_search_stats_report.get("basis_state_round_commit_selected", 0) or 0),
            "round_commit_selected_singleton": int(
                closure_search_stats_report.get("basis_state_round_commit_selected_singleton", 0) or 0
            ),
            "round_commit_selected_pair": int(
                closure_search_stats_report.get("basis_state_round_commit_selected_pair", 0) or 0
            ),
            "pair_precommit_scored": int(closure_search_stats_report.get("basis_state_pair_precommit_scored", 0) or 0),
            "pair_precommit_accepted": int(closure_search_stats_report.get("basis_state_pair_precommit_accepted", 0) or 0),
            "pair_normal_scored": int(closure_search_stats_report.get("basis_state_pair_normal_scored", 0) or 0),
            "pair_normal_accepted": int(closure_search_stats_report.get("basis_state_pair_normal_accepted", 0) or 0),
            "pair_rescue_scored": int(closure_search_stats_report.get("basis_state_pair_rescue_scored", 0) or 0),
            "pair_rescue_accepted": int(closure_search_stats_report.get("basis_state_pair_rescue_accepted", 0) or 0),
            "emergent_basis_rows": int(closure_search_stats_report.get("emergent_basis_rows", 0) or 0),
            "emergent_basis_scored": int(closure_search_stats_report.get("emergent_basis_scored", 0) or 0),
            "emergent_basis_unique": int(closure_search_stats_report.get("emergent_basis_unique", 0) or 0),
            "emergent_basis_reject_counts": dict(
                closure_search_stats_report.get("emergent_basis_reject_counts", {}) or {}
            ),
            "emergent_aux_atoms_enable": bool(
                closure_search_stats_report.get("emergent_aux_atoms_enable", False)
            ),
            "emergent_aux_atoms_accepted": int(
                closure_search_stats_report.get("emergent_aux_atom_accepted", 0) or 0
            ),
            "emergent_aux_atoms_unique": int(
                closure_search_stats_report.get("emergent_aux_atom_unique", 0) or 0
            ),
            "emergent_aux_atom_registry_size": int(
                closure_search_stats_report.get("emergent_aux_atom_registry_size", 0) or 0
            ),
            "emergent_aux_atom_observation_pool_size": int(
                closure_search_stats_report.get("emergent_aux_atom_observation_pool_size", 0) or 0
            ),
            "emergent_aux_atom_followup_reserved": int(
                closure_search_stats_report.get("emergent_aux_atom_followup_reserved", 0) or 0
            ),
            "emergent_aux_atom_seed_blocks": int(
                closure_search_stats_report.get("emergent_aux_atom_seed_blocks", 0) or 0
            ),
            "emergent_aux_atom_seed_exprs": list(
                closure_search_stats_report.get("emergent_aux_atom_seed_exprs", []) or []
            ),
            "emergent_aux_atom_by_kind": dict(
                closure_search_stats_report.get("emergent_aux_atom_by_kind", {}) or {}
            ),
            "emergent_aux_atom_reject_counts": dict(
                closure_search_stats_report.get("emergent_aux_atom_reject_counts", {}) or {}
            ),
            "emergent_aux_atom_observed_bucket_counts": dict(
                closure_search_stats_report.get("emergent_aux_atom_observed_bucket_counts", {}) or {}
            ),
            "emergent_aux_atom_registry_by_bucket": dict(
                closure_search_stats_report.get("emergent_aux_atom_registry_by_bucket", {}) or {}
            ),
            "atom_policy_library_records": int(
                closure_search_stats_report.get("atom_policy_library_records", 0) or 0
            ),
            "atom_policy_library_relations": int(
                closure_search_stats_report.get("atom_policy_library_relations", 0) or 0
            ),
            "atom_policy_source_atoms": int(
                closure_search_stats_report.get("atom_policy_source_atoms", 0) or 0
            ),
            "atom_policy_use_obs_pool": bool(
                closure_search_stats_report.get("atom_policy_use_obs_pool", False)
            ),
            "atomized_linear_span_enable": bool(
                closure_search_stats_report.get("atomized_linear_span_enable", False)
            ),
            "atomized_linear_span_budget": int(
                closure_search_stats_report.get("atomized_linear_span_budget", 0) or 0
            ),
            "atomized_linear_span_use_obs_pool": bool(
                closure_search_stats_report.get("atomized_linear_span_use_obs_pool", False)
            ),
            "atomized_linear_span_same_round": bool(
                closure_search_stats_report.get("atomized_linear_span_same_round", False)
            ),
            "atomized_linear_span_include_intercept": bool(
                closure_search_stats_report.get("atomized_linear_span_include_intercept", False)
            ),
            "atomized_linear_span_exact_quota": int(
                closure_search_stats_report.get("atomized_linear_span_exact_quota", 0) or 0
            ),
            "atomized_linear_span_exact_scored": int(
                closure_search_stats_report.get("atomized_linear_span_exact_scored", 0) or 0
            ),
            "atomized_linear_span_exact_quota_skipped": int(
                closure_search_stats_report.get("atomized_linear_span_exact_quota_skipped", 0) or 0
            ),
            "atomized_linear_span_rows": int(
                closure_search_stats_report.get("atomized_linear_span_rows", 0) or 0
            ),
            "atomized_linear_span_scored": int(
                closure_search_stats_report.get("atomized_linear_span_scored", 0) or 0
            ),
            "atomized_linear_span_structural_seed_mode": str(
                closure_search_stats_report.get("atomized_linear_span_structural_seed_mode", "") or ""
            ),
            "atomized_linear_span_structural_seed_enabled": bool(
                closure_search_stats_report.get("atomized_linear_span_structural_seed_enabled", False)
            ),
            "atomized_linear_span_structural_seed_atoms": int(
                closure_search_stats_report.get("atomized_linear_span_structural_seed_atoms", 0) or 0
            ),
            "atomized_linear_span_structural_seed_target_atoms": int(
                closure_search_stats_report.get("atomized_linear_span_structural_seed_target_atoms", 0) or 0
            ),
            "atomized_linear_span_structural_seed_dimless_atoms": int(
                closure_search_stats_report.get("atomized_linear_span_structural_seed_dimless_atoms", 0) or 0
            ),
            "atomized_linear_span_structural_seed_budget": int(
                closure_search_stats_report.get("atomized_linear_span_structural_seed_budget", 0) or 0
            ),
            "atomized_linear_span_coverage_candidates": int(
                closure_search_stats_report.get("atomized_linear_span_coverage_candidates", 0) or 0
            ),
            "atomized_linear_span_best_probe": (
                float(atomized_best_probe) if math.isfinite(atomized_best_probe) else None
            ),
            "family_priority_scores": dict(
                closure_search_stats_report.get("family_priority_scores", {}) or {}
            ),
            "family_priority_decomposition": dict(
                closure_search_stats_report.get("family_priority_decomposition", {}) or {}
            ),
            "aux_scaffolds_enumerated": int(
                closure_search_stats_report.get("aux_scaffolds_enumerated", 0) or 0
            ),
            "protected_aux_scaffolds_enumerated": int(
                closure_search_stats_report.get("protected_aux_scaffolds_enumerated", 0) or 0
            ),
            "accepted_pair_events": list(closure_search_stats_report.get("accepted_pair_events", []) or []),
            "selected_pair_events": list(closure_search_stats_report.get("selected_pair_events", []) or []),
        }
        if int(getattr(hp, "closure_search_debug_topk", 0) or 0) > 0:
            closure_search_debug = {
                "summary": dict(closure_search_summary),
                "round_summaries": list(closure_search_stats_report.get("debug_round_summaries", []) or []),
                "preview_rows": list(closure_search_stats_report.get("debug_preview_rows", []) or []),
                "exact_rows": list(closure_search_stats_report.get("debug_exact_rows", []) or []),
                "pair_pool": list(closure_search_stats_report.get("debug_pair_pool", []) or []),
                "pair_attempts": list(closure_search_stats_report.get("debug_pair_attempts", []) or []),
                "emergent_basis": list(closure_search_stats_report.get("debug_emergent_basis", []) or []),
                "emergent_aux_atoms": list(
                    closure_search_stats_report.get("debug_emergent_aux_atoms", []) or []
                ),
                "emergent_aux_atom_registry": list(
                    closure_search_stats_report.get("emergent_aux_atom_registry", []) or []
                ),
                "emergent_aux_atom_observed_top": list(
                    closure_search_stats_report.get("emergent_aux_atom_observed_top", []) or []
                ),
                "emergent_aux_atom_seen_not_retained": list(
                    closure_search_stats_report.get("emergent_aux_atom_seen_not_retained", []) or []
                ),
                "emergent_aux_atom_registry_not_retained": list(
                    closure_search_stats_report.get("emergent_aux_atom_registry_not_retained", []) or []
                ),
                "emergent_aux_atom_observation_pool": list(
                    closure_search_stats_report.get("emergent_aux_atom_observation_pool", []) or []
                ),
                "atom_policy_library": dict(
                    closure_search_stats_report.get("atom_policy_library", {}) or {}
                ),
                "atomized_linear_span_rows": list(
                    closure_search_stats_report.get("debug_atomized_linear_span_rows", []) or []
                ),
                "atomized_linear_span_atoms": dict(
                    closure_search_stats_report.get("debug_atomized_linear_span_atoms", {}) or {}
                ),
                "accepted_pair_events": list(closure_search_stats_report.get("accepted_pair_events", []) or []),
                "selected_pair_events": list(closure_search_stats_report.get("selected_pair_events", []) or []),
            }

    action_counts_sorted = dict(
        sorted(action_counts_total.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    )
    total_action_selected = int(sum(int(v) for v in action_counts_sorted.values()))
    if total_action_selected > 0:
        action_fracs = {
            k: (float(v) / float(total_action_selected))
            for k, v in action_counts_sorted.items()
        }
    else:
        action_fracs = {k: 0.0 for k in action_counts_sorted}

    report = {
        "spec_id": spec.id,
        "basis": list(spec.basis),
        "target_expr": spec.target_expr,
        "target_dim": list(spec.target_dim),
        "variable_names": [v.name for v in spec.variables],
        "constant_names": [c.name for c in spec.constants],
        "constant_values": {c.name: float(c.value) for c in spec.constants},
        "nvars_total": nvars,
        "enforce_dims": bool(enforce_dims),
        "gs_carrier_seed": bool(gs_carrier_seed),
        "gs_carrier_seed_diagnostics": _to_jsonable(gs_carrier_seed_diag),
        "gs_carrier_unit_stats": _to_jsonable(gs_carrier_unit_stats_total),
        "gs_carrier_unit_stats_by_seed": _to_jsonable(gs_carrier_unit_stats_by_seed),
        "dtype": str(dtype),
        "seed": int(run_seed),
        "n_seeds": int(n_seeds),
        "n_seeds_ran": int(n_seeds_ran),
        "n_iter_each": int(n_iter_each),
        "wall_seconds": elapsed,
        "wall_time_limit_s": None if wall_time_limit_s is None else float(wall_time_limit_s),
        "wall_time_limit_hit": bool(wall_time_limit_hit),
        "search_stop_reasons": list(search_stop_reasons),
        "dataset": _to_jsonable(dataset_metadata),
        "resolved_config": _to_jsonable(factorized_config_report(hp)),
        "archive_coherence": _to_jsonable(archive_coherence),
        "results": rows,
        "best": rows[0] if rows else None,
        "embedding_roundtrip_summary": _to_jsonable(embedding_roundtrip_summary),
        "action_distribution": {
            "counts": action_counts_sorted,
            "fractions": action_fracs,
            "total_selected": total_action_selected,
        },
        "inverse_experiment_log": list(getattr(arch, "inverse_experiment_log", []) or []),
        "hole_search_stats": _to_jsonable(hole_search_stats_total),
        "score_prescreen_stats": _to_jsonable(score_prescreen_stats_total),
        "route_scheduler_stats": _to_jsonable(route_scheduler_stats_total),
        "repair_controller_stats": _to_jsonable(getattr(arch, "repair_controller_stats", None)),
        "refine_diagnostics": _to_jsonable(refine_diagnostics_total),
        "refine_cost_summary": _to_jsonable(_refine_diagnostics_summary(refine_diagnostics_total)),
        "refine_diagnostics_by_seed": _to_jsonable(refine_diagnostics_by_seed),
        "refine_slate_stats_by_seed": _to_jsonable(refine_slate_stats_by_seed),
        "closure_search_summary": _to_jsonable(closure_search_summary),
        "closure_search_debug": _to_jsonable(closure_search_debug),
        "hp": {
            "n_iter": int(hp.n_iter),
            "wall_time_limit_s": None if wall_time_limit_s is None else float(wall_time_limit_s),
            "max_depth": int(hp.max_depth),
            "poly_degree": int(hp.poly_degree),
            "return_topk": int(hp.return_topk),
            "n_fit": int(hp.n_fit),
            "n_probe": int(hp.n_probe),
            "brute_depth": None if hp.brute_depth is None else int(hp.brute_depth),
            "early_stop_mse": float(hp.early_stop_mse),
            "brute_max_expressions": int(hp.brute_max_expressions),
            "score_mapping_family_mode": str(getattr(hp, "score_mapping_family_mode", "full")),
            "brute_score_mapping_family_mode": str(getattr(hp, "brute_score_mapping_family_mode", "gated")),
            "score_mapping_expensive_gate_best_factor": float(getattr(hp, "score_mapping_expensive_gate_best_factor", 5.0)),
            "score_mapping_expensive_rel_y": float(getattr(hp, "score_mapping_expensive_rel_y", 0.10)),
            "score_prescreen_enable": bool(getattr(hp, "score_prescreen_enable", True)),
            "score_prescreen_family_mode": str(getattr(hp, "score_prescreen_family_mode", "cheap")),
            "score_prescreen_residual_family_mode": str(getattr(hp, "score_prescreen_residual_family_mode", "gated")),
            "score_prescreen_residual_allow_hint": bool(getattr(hp, "score_prescreen_residual_allow_hint", False)),
            "score_prescreen_residual_use_global_best": bool(getattr(hp, "score_prescreen_residual_use_global_best", False)),
            "score_prescreen_parent_best_factor": float(getattr(hp, "score_prescreen_parent_best_factor", 1.5)),
            "score_prescreen_global_best_factor": float(getattr(hp, "score_prescreen_global_best_factor", 3.0)),
            "score_prescreen_residual_parent_best_factor": float(getattr(hp, "score_prescreen_residual_parent_best_factor", 1.1)),
            "score_prescreen_residual_global_best_factor": float(getattr(hp, "score_prescreen_residual_global_best_factor", 1.5)),
            "no_residual": bool(getattr(hp, "no_residual", False)),
            "inverse_steering_enable": bool(getattr(hp, "inverse_steering_enable", False)),
            "inverse_confidence_mode": str(getattr(hp, "inverse_confidence_mode", "conditioning")),
            "inverse_confidence_target_gain": float(getattr(hp, "inverse_confidence_target_gain", 4.0)),
            "inverse_confidence_floor": float(getattr(hp, "inverse_confidence_floor", 0.05)),
            "inverse_branch_beam_width": int(getattr(hp, "inverse_branch_beam_width", 1)),
            "inverse_micro_search_enable": bool(getattr(hp, "inverse_micro_search_enable", False)),
            "inverse_micro_search_max_depth": int(getattr(hp, "inverse_micro_search_max_depth", 3)),
            "inverse_micro_search_beam_width": int(getattr(hp, "inverse_micro_search_beam_width", 24)),
            "inverse_micro_search_topk": int(getattr(hp, "inverse_micro_search_topk", 16)),
            "inverse_micro_search_seed_terms": int(getattr(hp, "inverse_micro_search_seed_terms", 8)),
            "inverse_local_score_mode": str(getattr(hp, "inverse_local_score_mode", "affine")),
            "inverse_spec_enable": bool(getattr(hp, "inverse_spec_enable", False)),
            "inverse_spec_enum_max_depth": int(getattr(hp, "inverse_spec_enum_max_depth", 4)),
            "inverse_spec_enum_max_trees": int(getattr(hp, "inverse_spec_enum_max_trees", 5000)),
            "inverse_spec_preview_topk": int(getattr(hp, "inverse_spec_preview_topk", 16)),
            "inverse_spec_local_score_mode": str(getattr(hp, "inverse_spec_local_score_mode", "affine")),
            "inverse_spec_include_legacy_seed": bool(getattr(hp, "inverse_spec_include_legacy_seed", True)),
            "inverse_spec_complexity_penalty": float(getattr(hp, "inverse_spec_complexity_penalty", 0.0)),
            "inverse_spec_repair_quota": float(getattr(hp, "inverse_spec_repair_quota", 0.0)),
            "repair_pass_enable": bool(getattr(hp, "repair_pass_enable", False)),
            "repair_pass_elite_k": int(getattr(hp, "repair_pass_elite_k", 8)),
            "repair_pass_paths_per_elite": int(getattr(hp, "repair_pass_paths_per_elite", 2)),
            "repair_pass_rounds": int(getattr(hp, "repair_pass_rounds", 2)),
            "closure_search_enable": bool(getattr(hp, "closure_search_enable", False)),
            "closure_search_families": list(getattr(hp, "closure_search_families", ["periodic", "exp", "log", "rational", "power", "quadratic"])),
            "closure_search_max_proposals": int(getattr(hp, "closure_search_max_proposals", 16)),
            "closure_search_anchors_per_family": int(getattr(hp, "closure_search_anchors_per_family", 4)),
            "closure_search_preview_topk": int(getattr(hp, "closure_search_preview_topk", 4)),
            "closure_search_exact_topk": int(getattr(hp, "closure_search_exact_topk", 2)),
            "closure_search_beam_width": int(getattr(hp, "closure_search_beam_width", 4)),
            "closure_search_seed_exact_topk": int(getattr(hp, "closure_search_seed_exact_topk", 6)),
            "closure_search_seed_beam_width": int(getattr(hp, "closure_search_seed_beam_width", 4)),
            "closure_search_seed_scaffold_reserve": int(
                getattr(hp, "closure_search_seed_scaffold_reserve", 8)
            ),
            "closure_search_seed_family_cap": int(getattr(hp, "closure_search_seed_family_cap", 2)),
            "closure_search_seed_exact_bound_bonus": float(
                getattr(hp, "closure_search_seed_exact_bound_bonus", 0.25)
            ),
            "closure_search_pair_normal_enable": bool(
                getattr(hp, "closure_search_pair_normal_enable", False)
            ),
            "closure_search_pair_normal_topk": int(
                getattr(hp, "closure_search_pair_normal_topk", 3)
            ),
            "closure_search_pair_normal_max_pairs": int(
                getattr(hp, "closure_search_pair_normal_max_pairs", 1)
            ),
            "closure_search_pair_rescue_enable": bool(
                getattr(hp, "closure_search_pair_rescue_enable", True)
            ),
            "closure_search_pair_rescue_topk": int(
                getattr(hp, "closure_search_pair_rescue_topk", 4)
            ),
            "closure_search_pair_rescue_max_pairs": int(
                getattr(hp, "closure_search_pair_rescue_max_pairs", 6)
            ),
            "closure_search_emergent_basis_enable": bool(
                getattr(hp, "closure_search_emergent_basis_enable", False)
            ),
            "closure_search_emergent_basis_max_source_rows": int(
                getattr(hp, "closure_search_emergent_basis_max_source_rows", 32)
            ),
            "closure_search_emergent_basis_score_topk": int(
                getattr(hp, "closure_search_emergent_basis_score_topk", 8)
            ),
            "closure_search_emergent_basis_max_per_round": int(
                getattr(hp, "closure_search_emergent_basis_max_per_round", 1)
            ),
            "closure_search_emergent_basis_max_total": int(
                getattr(hp, "closure_search_emergent_basis_max_total", 4)
            ),
            "closure_search_emergent_basis_min_probe_gain_rel": float(
                getattr(hp, "closure_search_emergent_basis_min_probe_gain_rel", 5.0e-3)
            ),
            "closure_search_emergent_aux_atoms_enable": bool(
                getattr(hp, "closure_search_emergent_aux_atoms_enable", False)
            ),
            "closure_search_emergent_aux_atoms_max_source_rows": int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_source_rows", 48)
            ),
            "closure_search_emergent_aux_atoms_max_new_per_round": int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_new_per_round", 5)
            ),
            "closure_search_emergent_aux_atoms_max_total": int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_total", 8)
            ),
            "closure_search_emergent_aux_atoms_max_target": int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_target", 4)
            ),
            "closure_search_emergent_aux_atoms_max_dimensionless": int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_dimensionless", 3)
            ),
            "closure_search_emergent_aux_atoms_max_rational_derived": int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_rational_derived", 2)
            ),
            "closure_search_emergent_aux_atoms_max_seed_blocks": int(
                getattr(hp, "closure_search_emergent_aux_atoms_max_seed_blocks", 8)
            ),
            "closure_search_debug_topk": int(
                getattr(hp, "closure_search_debug_topk", 0)
            ),
            "closure_search_min_valid_frac": float(getattr(hp, "closure_search_min_valid_frac", 0.05)),
            "closure_search_min_confidence": float(getattr(hp, "closure_search_min_confidence", 0.02)),
            "closure_search_periodic_min_valid_scale": float(
                getattr(hp, "closure_search_periodic_min_valid_scale", 1.0)
            ),
            "closure_search_periodic_min_confidence_scale": float(
                getattr(hp, "closure_search_periodic_min_confidence_scale", 1.0)
            ),
            "closure_search_transport_min_lin_rel": float(
                getattr(hp, "closure_search_transport_min_lin_rel", 0.0)
            ),
            "closure_search_anchor_head_compare_enable": bool(
                getattr(hp, "closure_search_anchor_head_compare_enable", False)
            ),
            "hole_search_enable": bool(getattr(hp, "hole_search_enable", False)),
            "hole_search_quota": float(getattr(hp, "hole_search_quota", 0.10)),
            "hole_search_exact_budget": int(getattr(hp, "hole_search_exact_budget", 2)),
            "hole_search_cooldown_iters": int(getattr(hp, "hole_search_cooldown_iters", 32)),
            "hole_search_mine_cooldown_iters": int(getattr(hp, "hole_search_mine_cooldown_iters", 50)),
            "hole_search_max_frontier": int(getattr(hp, "hole_search_max_frontier", 128)),
            "hole_search_first_class_scheduler_enable": bool(getattr(hp, "hole_search_first_class_scheduler_enable", True)),
            "hole_search_route_scheduler_enable": bool(getattr(hp, "hole_search_route_scheduler_enable", True)),
            "hole_search_route_ucb_c": float(getattr(hp, "hole_search_route_ucb_c", 0.25)),
            "hole_search_route_eps": float(getattr(hp, "hole_search_route_eps", 0.05)),
            "hole_search_route_acquisition_weight": float(getattr(hp, "hole_search_route_acquisition_weight", 0.25)),
            "hole_search_route_reward_mode": str(getattr(hp, "hole_search_route_reward_mode", "penalized")),
            "hole_search_route_time_penalty": float(getattr(hp, "hole_search_route_time_penalty", 0.01)),
            "hole_search_route_time_floor": float(getattr(hp, "hole_search_route_time_floor", 1.0)),
            "hole_search_abstraction_enable": bool(getattr(hp, "hole_search_abstraction_enable", True)),
            "hole_search_abstraction_on_improve": bool(getattr(hp, "hole_search_abstraction_on_improve", True)),
            "hole_search_abstraction_on_stall": bool(getattr(hp, "hole_search_abstraction_on_stall", True)),
            "hole_search_abstraction_cooldown_iters": int(getattr(hp, "hole_search_abstraction_cooldown_iters", 25)),
            "hole_search_abstraction_max_parents": int(getattr(hp, "hole_search_abstraction_max_parents", 2)),
            "hole_search_abstraction_max_paths_per_parent": int(getattr(hp, "hole_search_abstraction_max_paths_per_parent", 3)),
            "hole_search_abstraction_improve_min_delta_log_mse": float(getattr(hp, "hole_search_abstraction_improve_min_delta_log_mse", 0.15)),
            "hole_search_abstraction_stage_enable": bool(getattr(hp, "hole_search_abstraction_stage_enable", True)),
            "hole_search_abstraction_stage_max_entries": int(getattr(hp, "hole_search_abstraction_stage_max_entries", 64)),
            "hole_search_abstraction_promote_topk": int(getattr(hp, "hole_search_abstraction_promote_topk", 2)),
            "hole_search_abstraction_promote_frontier_floor": int(getattr(hp, "hole_search_abstraction_promote_frontier_floor", 3)),
            "hole_search_enum_max_depth": int(getattr(hp, "hole_search_enum_max_depth", 4)),
            "hole_search_enum_max_trees": int(getattr(hp, "hole_search_enum_max_trees", 3000)),
            "hole_search_preview_topk": int(getattr(hp, "hole_search_preview_topk", 8)),
            "hole_search_tournament_enable": bool(getattr(hp, "hole_search_tournament_enable", True)),
            "hole_search_tournament_n": int(getattr(hp, "hole_search_tournament_n", 8)),
            "hole_search_tournament_elite_k": int(getattr(hp, "hole_search_tournament_elite_k", 2)),
            "hole_search_tournament_preview_trees": int(getattr(hp, "hole_search_tournament_preview_trees", 64)),
            "stall_window": int(getattr(hp, "stall_window", 500)),
            "stall_patience": int(getattr(hp, "stall_patience", 3)),
            "stall_delta": float(getattr(hp, "stall_delta", 1.0e-4)),
            "inverse_spec_recursive_enable": bool(getattr(hp, "inverse_spec_recursive_enable", True)),
            "inverse_spec_recursive_max_depth": int(getattr(hp, "inverse_spec_recursive_max_depth", 2)),
            "inverse_spec_recursive_trigger_rel_mse": float(getattr(hp, "inverse_spec_recursive_trigger_rel_mse", 0.25)),
            "inverse_spec_recursive_seed_cap": int(getattr(hp, "inverse_spec_recursive_seed_cap", 6)),
            "inverse_spec_recursive_branch_topk": int(getattr(hp, "inverse_spec_recursive_branch_topk", 4)),
            "inverse_spec_recursive_child_topk": int(getattr(hp, "inverse_spec_recursive_child_topk", 2)),
            "inverse_spec_max_subtree_depth": getattr(hp, "inverse_spec_max_subtree_depth", None),
            "inverse_spec_fit_cap": int(getattr(hp, "inverse_spec_fit_cap", 96)),
            "inverse_spec_probe_cap": int(getattr(hp, "inverse_spec_probe_cap", 192)),
            "inverse_spec_exact_budget": int(getattr(hp, "inverse_spec_exact_budget", 4)),
            "inverse_target_mode": str(getattr(hp, "inverse_target_mode", "robust")),
            "inverse_full_mapping_penalty": float(getattr(hp, "inverse_full_mapping_penalty", 0.75)),
            "inverse_exact_simple_target_bonus": float(getattr(hp, "inverse_exact_simple_target_bonus", 0.10)),
            "inverse_additive_descend_penalty": float(getattr(hp, "inverse_additive_descend_penalty", 0.15)),
            "inverse_nonadditive_leaf_penalty": float(getattr(hp, "inverse_nonadditive_leaf_penalty", 0.20)),
            "inverse_exact_path_eta": float(getattr(hp, "inverse_exact_path_eta", 0.98)),
            "inverse_exact_transport_min_lin_rel": float(getattr(hp, "inverse_exact_transport_min_lin_rel", 0.0)),
            "inverse_periodic_min_valid_scale": float(getattr(hp, "inverse_periodic_min_valid_scale", 1.25)),
            "inverse_periodic_min_confidence_scale": float(getattr(hp, "inverse_periodic_min_confidence_scale", 1.35)),
            "inverse_periodic_path_penalty": float(getattr(hp, "inverse_periodic_path_penalty", 0.65)),
            "inverse_nonperiodic_muldiv_bonus": float(getattr(hp, "inverse_nonperiodic_muldiv_bonus", 0.10)),
            "inverse_nonperiodic_explogsqrt_bonus": float(getattr(hp, "inverse_nonperiodic_explogsqrt_bonus", 0.05)),
            "inverse_branch_ambiguity_penalty": float(getattr(hp, "inverse_branch_ambiguity_penalty", 0.50)),
            "inverse_transport_min_lin_rel": float(getattr(hp, "inverse_transport_min_lin_rel", 0.02)),
            "inverse_transport_min_effective_n": float(getattr(hp, "inverse_transport_min_effective_n", 8.0)),
            "inverse_experiment_log_enable": bool(getattr(hp, "inverse_experiment_log_enable", False)),
            "repair_controller_enable": bool(getattr(hp, "repair_controller_enable", False)),
            "repair_controller_min_score": float(getattr(hp, "repair_controller_min_score", 0.15)),
            "repair_controller_steps": int(getattr(hp, "repair_controller_steps", 3)),
            "repair_controller_ancestor_hops": int(getattr(hp, "repair_controller_ancestor_hops", 1)),
            "repair_controller_min_step_rel_improve": float(getattr(hp, "repair_controller_min_step_rel_improve", 1.0e-3)),
            "repair_controller_adaptive": bool(getattr(hp, "repair_controller_adaptive", True)),
            "repair_controller_adapt_quantile": float(getattr(hp, "repair_controller_adapt_quantile", 0.75)),
            "repair_controller_adapt_window": int(getattr(hp, "repair_controller_adapt_window", 128)),
            "repair_controller_adapt_min_samples": int(getattr(hp, "repair_controller_adapt_min_samples", 16)),
            "repair_controller_min_concentration": float(getattr(hp, "repair_controller_min_concentration", 0.30)),
            "repair_controller_potential_weight": float(getattr(hp, "repair_controller_potential_weight", 1.00)),
            "repair_controller_concentration_weight": float(getattr(hp, "repair_controller_concentration_weight", 0.35)),
            "repair_controller_contrast_weight": float(getattr(hp, "repair_controller_contrast_weight", 0.20)),
            "repair_controller_cost_weight": float(getattr(hp, "repair_controller_cost_weight", 0.10)),
            "repair_controller_stagnation_weight": float(getattr(hp, "repair_controller_stagnation_weight", 0.15)),
            "repair_controller_frontier_topk": int(getattr(hp, "repair_controller_frontier_topk", 24)),
            "repair_controller_stagnation_visits": int(getattr(hp, "repair_controller_stagnation_visits", 8)),
            "repair_controller_focus_prob": float(getattr(hp, "repair_controller_focus_prob", 0.50)),
            "repair_controller_parent_max_repeats": int(getattr(hp, "repair_controller_parent_max_repeats", 2)),
            "repair_controller_parent_min_eval_gap": int(getattr(hp, "repair_controller_parent_min_eval_gap", 32)),
            "repair_controller_parent_reset_rel_improve": float(getattr(hp, "repair_controller_parent_reset_rel_improve", 0.05)),
            "repair_controller_critic_enable": bool(getattr(hp, "repair_controller_critic_enable", False)),
            "repair_controller_critic_path": str(getattr(hp, "repair_controller_critic_path", "")),
            "repair_controller_critic_blend": float(getattr(hp, "repair_controller_critic_blend", 1.0)),
            "repair_controller_critic_mode": str(getattr(hp, "repair_controller_critic_mode", "priority")),
            "repair_opportunity_controller_enable": bool(getattr(hp, "repair_opportunity_controller_enable", False)),
            "repair_opportunity_controller_path": str(getattr(hp, "repair_opportunity_controller_path", "")),
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
        },
    }
    return report


def save_oracle_report(report: dict[str, Any], path: str | pathlib.Path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")




@dataclass(frozen=True)
class InverseStep:
    """One local inversion step while peeling context toward a subtree."""

    parent_path: tuple[int, ...]
    op: str
    child_slot: int
    valid_fraction: float
    confidence: float
    note: str = ""


@dataclass(frozen=True)
class InverseTarget:
    """Pseudo-target for a chosen subtree obtained by inverting surrounding context."""

    path: tuple[int, ...]
    target: torch.Tensor
    valid_mask: torch.Tensor
    confidence: float
    mapping_inverted: bool
    mapping_kind: str
    steps: tuple[InverseStep, ...]


def _ensure_col(v: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(v):
        raise TypeError(f"expected torch.Tensor, got {type(v).__name__}")
    if v.ndim == 1:
        return v.unsqueeze(-1)
    if v.ndim == 2 and v.shape[1] == 1:
        return v
    raise ValueError(f"expected [N] or [N,1] tensor, got shape={tuple(v.shape)}")


def _bool_col(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(-1)
    if mask.ndim != 2 or mask.shape[1] != 1:
        raise ValueError(f"expected mask shape [N,1], got {tuple(mask.shape)}")
    return mask.to(dtype=torch.bool)


def _mask_fraction(mask: torch.Tensor) -> float:
    m = _bool_col(mask)
    if m.numel() == 0:
        return 0.0
    return float(m.float().mean().item())


def _finite_mask(*xs: torch.Tensor) -> torch.Tensor:
    if not xs:
        raise ValueError("_finite_mask requires at least one tensor")
    mask = torch.ones_like(_ensure_col(xs[0]), dtype=torch.bool)
    for x in xs:
        mask = mask & torch.isfinite(_ensure_col(x))
    return mask

def _corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = _ensure_col(a).squeeze(-1)
    bb = _ensure_col(b).squeeze(-1)
    mask = torch.isfinite(aa) & torch.isfinite(bb)
    if int(mask.sum().item()) < 2:
        return 0.0
    aa = aa[mask]
    bb = bb[mask]
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    na = float(torch.linalg.norm(aa).item())
    nb = float(torch.linalg.norm(bb).item())
    if na < 1.0e-12 or nb < 1.0e-12:
        return 0.0
    return float((aa @ bb).item() / (na * nb))


def _slice_by_mask(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = _bool_col(mask).squeeze(-1)
    return x[m], _ensure_col(y)[m]


def _parse_path(path: str | Sequence[int] | None) -> tuple[int, ...]:
    if path is None:
        return ()
    if isinstance(path, (list, tuple)):
        out = []
        for i, v in enumerate(path):
            iv = int(v)
            if iv not in (1, 2):
                raise ValueError(f"path[{i}] must be 1 or 2, got {iv}")
            out.append(iv)
        return tuple(out)
    s = str(path).strip()
    if s in ("", "/", "root", "ROOT", "()"):
        return ()
    toks = [t for t in s.replace('.', '/').split('/') if t.strip()]
    out: list[int] = []
    for i, tok in enumerate(toks):
        try:
            iv = int(tok)
        except Exception as exc:
            raise ValueError(f"invalid path component {tok!r} in {path!r}") from exc
        if iv not in (1, 2):
            raise ValueError(f"path component {i} must be 1 or 2, got {iv}")
        out.append(iv)
    return tuple(out)


def _format_path(path: Sequence[int]) -> str:
    pp = tuple(int(v) for v in path)
    if not pp:
        return "root"
    return "/".join(str(v) for v in pp)


def _default_inverse_path(node: tuple) -> tuple[int, ...]:
    paths = collect_paths(node)
    if len(paths) <= 1:
        return ()
    return max(paths[1:], key=lambda p: (len(p), p))


def _path_relation(path: Sequence[int] | None, ref_path: Sequence[int] | None) -> str:
    if path is None or ref_path is None:
        return "unknown"
    pp = tuple(int(v) for v in path)
    rr = tuple(int(v) for v in ref_path)
    if pp == rr:
        return "same"
    if len(pp) < len(rr) and rr[: len(pp)] == pp:
        return "ancestor"
    if len(rr) < len(pp) and pp[: len(rr)] == rr:
        return "descendant"
    return "disjoint"


def _dedupe_paths(paths: Sequence[Sequence[int] | None]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for path in paths:
        if path is None:
            continue
        pp = tuple(int(v) for v in path)
        if pp in seen:
            continue
        seen.add(pp)
        out.append(pp)
    return out


def _minimal_mismatch_paths(a: tuple, b: tuple, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    """Return shallow repair-cut paths where ``a`` and ``b`` first diverge.

    This is intentionally *not* a leaf-level tree diff.  The goal is to identify
    the coarsest subtree cut that could plausibly repair the candidate in one
    symbolic edit.  If both children of a binary node differ, the current node is
    treated as the repair site instead of descending further.
    """

    if a == b:
        return []
    op_a = str(a[0]) if isinstance(a, tuple) and len(a) > 0 else ""
    op_b = str(b[0]) if isinstance(b, tuple) and len(b) > 0 else ""
    if op_a != op_b:
        return [tuple(prefix)]
    if op_a in ("var", "const", "hparam"):
        return [tuple(prefix)]
    if op_a in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg"):
        child = _minimal_mismatch_paths(a[1], b[1], prefix + (1,))
        return child if child else [tuple(prefix)]
    if op_a in ("add", "sub", "mul", "div"):
        diff1 = a[1] != b[1]
        diff2 = a[2] != b[2]
        if diff1 and diff2:
            return [tuple(prefix)]
        if diff1:
            child = _minimal_mismatch_paths(a[1], b[1], prefix + (1,))
            return child if child else [tuple(prefix)]
        if diff2:
            child = _minimal_mismatch_paths(a[2], b[2], prefix + (2,))
            return child if child else [tuple(prefix)]
        return [tuple(prefix)]
    return [tuple(prefix)]


def _best_relation_to_reference(path: Sequence[int] | None, ref_paths: Sequence[Sequence[int]]) -> str:
    if path is None:
        return "unknown"
    if not ref_paths:
        return "unknown"
    ranks = {"same": 3, "ancestor": 2, "descendant": 1, "disjoint": 0, "unknown": -1}
    best = "unknown"
    best_rank = -1
    for ref in ref_paths:
        rel = _path_relation(path, ref)
        rr = int(ranks.get(rel, -1))
        if rr > best_rank:
            best = rel
            best_rank = rr
    return best


def _json_ast_root_op(node: Any) -> str:
    if isinstance(node, (list, tuple)) and len(node) >= 1:
        return str(node[0])
    return ""


def _json_ast_depth(node: Any) -> int:
    if not isinstance(node, (list, tuple)) or len(node) == 0:
        return 0
    op = str(node[0])
    if op in ("var", "const", "hparam"):
        return 0
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg"):
        return 1 + _json_ast_depth(node[1] if len(node) > 1 else None)
    if op in ("add", "sub", "mul", "div"):
        left = _json_ast_depth(node[1] if len(node) > 1 else None)
        right = _json_ast_depth(node[2] if len(node) > 2 else None)
        return 1 + max(left, right)
    return 0


def _json_ast_size(node: Any) -> int:
    if not isinstance(node, (list, tuple)) or len(node) == 0:
        return 0
    op = str(node[0])
    if op in ("var", "const", "hparam"):
        return 1
    if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg"):
        return 1 + _json_ast_size(node[1] if len(node) > 1 else None)
    if op in ("add", "sub", "mul", "div"):
        return 1 + _json_ast_size(node[1] if len(node) > 1 else None) + _json_ast_size(node[2] if len(node) > 2 else None)
    return 1


def _oracle_rel_improvement(base_probe_mse: Any, candidate_probe_mse: Any) -> float:
    base = max(1.0e-30, _as_finite_float(base_probe_mse, where="base_probe_mse"))
    cand = max(1.0e-30, _as_finite_float(candidate_probe_mse, where="candidate_probe_mse"))
    return float(max(0.0, min(1.0, (base - cand) / base)))


def _oracle_mode_choice(mode_reports: dict[str, Any], *, base_probe_mse: float) -> dict[str, Any]:
    best_payload = None
    best_key = None
    for mode_name, payload in (mode_reports or {}).items():
        if not isinstance(payload, dict):
            continue
        truth_rank_raw = payload.get("truth_rank", None)
        truth_rank = int(truth_rank_raw) if truth_rank_raw is not None else 10**6
        truth_fit = payload.get("truth_fit_to_inverse", {}) or {}
        probe_mse = truth_fit.get("probe_mse", None)
        probe_mse = float(probe_mse) if probe_mse is not None else float("inf")
        corr_probe = truth_fit.get("corr_probe", None)
        corr_probe = float(corr_probe) if corr_probe is not None else float("-inf")
        top_rows = list(payload.get("top_replacements", []) or [])
        top_full_probe_mse = top_rows[0].get("full_probe_mse", None) if top_rows else None
        top_full_probe_mse = float(top_full_probe_mse) if top_full_probe_mse is not None else float("inf")
        improvement = _oracle_rel_improvement(base_probe_mse, top_full_probe_mse if math.isfinite(top_full_probe_mse) else base_probe_mse)
        key = (
            truth_rank,
            probe_mse,
            top_full_probe_mse,
            -corr_probe,
            -improvement,
            str(mode_name),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_payload = {
                "mode": str(mode_name),
                "truth_rank": None if truth_rank_raw is None else int(truth_rank_raw),
                "truth_probe_mse": None if not math.isfinite(probe_mse) else float(probe_mse),
                "top_full_probe_mse": None if not math.isfinite(top_full_probe_mse) else float(top_full_probe_mse),
                "corr_probe": None if not math.isfinite(corr_probe) else float(corr_probe),
                "improvement_estimate": float(improvement),
            }
    if best_payload is None:
        return {
            "mode": None,
            "truth_rank": None,
            "truth_probe_mse": None,
            "top_full_probe_mse": None,
            "corr_probe": None,
            "improvement_estimate": 0.0,
        }
    return best_payload


def _oracle_controller_row_from_path_sweep(report: dict[str, Any]) -> dict[str, Any]:
    gate_diag = dict(report.get("gate_diagnostic", {}) or {})
    path_rows = list(gate_diag.get("path_rows", []) or [])
    dist = path_distribution_metrics(path_rows)
    summary = path_summary_stats(path_rows)
    selected_report = dict(report.get("selected_path_report", {}) or {})
    selected_gate_row = dict(selected_report.get("gate_row", {}) or {})
    candidate_score = dict(report.get("candidate_score", {}) or {})
    candidate_ast = report.get("candidate_expr_ast", None)
    first_gate_row = dict(path_rows[0] if path_rows else {})
    row = {
        "parent_expr": str(report.get("candidate_expr", "")),
        "parent_root_op": _json_ast_root_op(candidate_ast),
        "parent_depth": int(_json_ast_depth(candidate_ast)),
        "parent_size": int(_json_ast_size(candidate_ast)),
        "parent_best_eff_mse": float(candidate_score.get("probe_mse", float("inf"))),
        "parent_best_raw_mse": float(candidate_score.get("fit_mse", float("inf"))),
        "gate_allowed": bool(gate_diag.get("allowed", False)),
        "gate_reason": str(gate_diag.get("reason", "")),
        "path_summaries": path_rows,
        "path_entropy": float(dist.get("path_entropy", 0.0)),
        "path_top_mass": float(dist.get("path_top_mass", 0.0)),
        "path_second_mass": float(dist.get("path_second_mass", 0.0)),
        "path_positive_count": float(dist.get("path_positive_count", 0.0)),
        "path_summary_count": int(summary.get("path_summary_count", len(path_rows))),
        "path_summary_gain_mass": float(summary.get("path_summary_gain_mass", 0.0)),
        "path_summary_gap": float(summary.get("path_summary_gap", 0.0)),
        "path_summary_support": float(summary.get("path_summary_support", 0.0)),
        "path_summary_mode_diversity": float(summary.get("path_summary_mode_diversity", 0.0)),
        "selected_path": list(selected_report.get("path", []) or []),
        "selected_target_mode": selected_gate_row.get("target_mode", None),
        "selected_path_gain": selected_gate_row.get("weighted_rel_gain", None),
        "selected_path_gain_pre_cut": selected_gate_row.get("weighted_rel_gain_pre_cut", selected_gate_row.get("weighted_rel_gain", None)),
        "selected_rel_gain": selected_gate_row.get("rel_gain", None),
        "selected_transport_rel": selected_gate_row.get("transport_rel", None),
        "selected_branch_factor": selected_gate_row.get("branch_factor", None),
        "selected_cut_factor": selected_gate_row.get("cut_factor", None),
        "local_candidate_count": int(max(0, selected_report.get("modes", {}).get("identity", {}).get("shared_proposal_count", 0) if isinstance(selected_report.get("modes", {}), dict) else 0)),
    }
    if first_gate_row:
        row.update({
            "gate_best_path": list(first_gate_row.get("path", []) or []),
            "gate_best_weighted_rel_gain": first_gate_row.get("weighted_rel_gain", None),
            "gate_best_rel_gain": first_gate_row.get("rel_gain", None),
            "gate_best_valid_frac": first_gate_row.get("valid_frac", None),
            "gate_best_confidence": first_gate_row.get("confidence", None),
            "gate_best_transport_rel": first_gate_row.get("transport_rel", None),
            "gate_best_static_score": first_gate_row.get("static_score", None),
            "gate_best_branch_factor": first_gate_row.get("branch_factor", None),
            "gate_best_cut_factor": first_gate_row.get("cut_factor", None),
            "gate_best_profile_exact_monotone": bool(first_gate_row.get("profile_exact_monotone", False)),
            "gate_best_profile_has_periodic": bool(first_gate_row.get("profile_has_periodic", False)),
            "gate_best_profile_has_muldiv": bool(first_gate_row.get("profile_has_muldiv", False)),
            "gate_best_profile_has_explogsqrt": bool(first_gate_row.get("profile_has_explogsqrt", False)),
        })
    return row


def _oracle_target_path_label(path_labels: Sequence[dict[str, Any]]) -> tuple[tuple[int, ...] | None, str]:
    relation_rank = {"same": 4, "ancestor": 3, "descendant": 2, "disjoint": 1, "unknown": 0}
    best_key = None
    best_path = None
    best_relation = "unknown"
    for row in path_labels:
        if not isinstance(row, dict):
            continue
        relation = str(row.get("relation", "unknown") or "unknown")
        path = tuple(int(v) for v in (row.get("path", []) or ()))
        truth_rank = row.get("best_mode_truth_rank", None)
        truth_rank = int(truth_rank) if truth_rank is not None else 10**6
        improvement = float(row.get("improvement_estimate", 0.0) or 0.0)
        key = (-int(relation_rank.get(relation, 0)), truth_rank, -improvement, len(path), tuple(path))
        if best_key is None or key < best_key:
            best_key = key
            best_path = path
            best_relation = relation
    return best_path, best_relation


def _oracle_mapping_nparams(mapping: Any) -> int:
    if not isinstance(mapping, dict):
        return 0
    for key in ("coeffs", "num", "den"):
        value = mapping.get(key, None)
        if isinstance(value, (list, tuple)):
            return int(len(value))
    params = mapping.get("params", None)
    if isinstance(params, (list, tuple)):
        return int(len(params))
    if isinstance(params, dict):
        return int(len(params))
    return 0


def _oracle_tuple_root_op(node: Any) -> str:
    if isinstance(node, tuple) and node:
        return str(node[0])
    return ""


def _oracle_deterministic_rng(*parts: Any) -> random.Random:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")
    return random.Random(seed)


def _oracle_mismatch_count(a: tuple, b: tuple) -> int:
    return int(len(_minimal_mismatch_paths(a, b)))


@torch.no_grad()
def _oracle_collect_build_candidate_rows(
    *,
    spec: EquationSpec,
    hp: FactorizedSearchConfig,
    run_seed: int,
    candidate_ast: tuple,
    truth_ast: tuple,
    ds: dict[str, Any],
    reference_paths: Sequence[Sequence[int]],
    selected_path: Sequence[int] | None,
    base_probe_mse: float,
) -> list[dict[str, Any]]:
    var_dims = ds.get("var_dims", None)
    nvars = int(ds["x_fit"].shape[1])
    max_depth_eff = max(int(node_depth(candidate_ast)), int(getattr(hp, "max_depth", node_depth(candidate_ast))))
    reference_eval_paths = [tuple(int(v) for v in p) for p in reference_paths if tuple(int(v) for v in p) in set(collect_paths(candidate_ast))]
    candidate_paths = _dedupe_paths([*reference_eval_paths, selected_path])
    path_action_ids = (A_REPLACE, A_WRAP_UNARY)
    build_rows: list[dict[str, Any]] = []
    parent_size = float(node_size(candidate_ast))
    parent_depth = float(node_depth(candidate_ast))
    pool_cache = _oracle_build_pool_cache(ds["x_fit"], ds["x_probe"], var_dims=var_dims)
    pool_phi_probe = pool_cache["pool_phi_probe"]
    pool_norms_probe = (pool_phi_probe ** 2).sum(0)
    residual_action_id = A_RESIDUAL
    action_ordinal = 0

    def _finalize_expr(expr: Any) -> Any:
        if expr is None:
            return None
        expr = simplify(expr)
        while isinstance(expr, tuple) and expr and expr[0] == "neg":
            expr = expr[1]
        if isinstance(expr, tuple) and expr and expr[0] == "sub" and node_str(expr[1]) > node_str(expr[2]):
            expr = ("sub", expr[2], expr[1])
        return expr

    def _build_row(expr: tuple, *, action_id: int, action_path: Sequence[int] | None, path_source: str) -> dict[str, Any] | None:
        scored = _score_expr_against_target(
            expr,
            x_fit=ds["x_fit"],
            y_fit=ds["y_fit"],
            x_probe=ds["x_probe"],
            y_probe=ds["y_probe"],
            poly_degree=int(hp.poly_degree),
        )
        if scored is None:
            return None
        child_expr = node_str(expr)
        mismatch_count = _oracle_mismatch_count(expr, truth_ast)
        relation = _best_relation_to_reference(action_path, reference_paths) if action_path else "unknown"
        row_payload = {
            "tuple_provenance": "oracle_build_slate",
            "proposal_family": "oracle_build_slate",
            "generation_source": "oracle_build_slate",
            "action": str(ACTION_NAME.get(int(action_id), f"action_{int(action_id)}")),
            "action_id": int(action_id),
            "path": [] if action_path is None else [int(v) for v in action_path],
            "path_source": str(path_source or ""),
            "target_mode": "",
            "child_key": child_expr,
            "child_expr": child_expr,
            "child_eff_mse": float(scored["probe_mse"]),
            "child_raw_mse": float(scored["fit_mse"]),
            "exact_child_score_observed": True,
            "accepted": bool(float(scored["probe_mse"]) < float(base_probe_mse)),
            "candidate_child_size": float(node_size(expr)),
            "candidate_child_depth": float(node_depth(expr)),
            "candidate_child_size_delta": float(node_size(expr) - parent_size),
            "candidate_child_depth_delta": float(node_depth(expr) - parent_depth),
            "candidate_root_op": _oracle_tuple_root_op(expr),
            "path_length": int(len(action_path or ())),
            "oracle_relation_to_reference": relation,
            "oracle_path_is_correct": bool(relation == "same") if action_path else False,
            "oracle_path_is_near": bool(relation in {"same", "ancestor", "descendant"}) if action_path else False,
            "oracle_truth_present_under_path": bool(relation in {"same", "ancestor", "descendant"}) if action_path else False,
            "oracle_is_truth_candidate": bool(mismatch_count == 0),
            "oracle_structural_mismatch_count": int(mismatch_count),
        }
        return shared_candidate_row_dict(row_payload, route_source="build")

    for path in candidate_paths:
        for action_id in path_action_ids:
            rng = _oracle_deterministic_rng(spec.id, run_seed, "oracle_build", list(path), int(action_id), action_ordinal)
            expr = _finalize_expr(
                apply_action(
                    candidate_ast,
                    int(action_id),
                    rng,
                    max_depth_eff,
                    nvars,
                    var_dims=var_dims,
                    reach=None,
                    path=path,
                )
            )
            action_ordinal += 1
            if expr is None:
                continue
            row = _build_row(expr, action_id=int(action_id), action_path=path, path_source="oracle_reference")
            if row is not None:
                build_rows.append(row)

    rng = _oracle_deterministic_rng(spec.id, run_seed, "oracle_build", "residual")
    expr = _finalize_expr(
        apply_residual_action(
            candidate_ast,
            _score_expr_against_target(
                candidate_ast,
                x_fit=ds["x_fit"],
                y_fit=ds["y_fit"],
                x_probe=ds["x_probe"],
                y_probe=ds["y_probe"],
                poly_degree=int(hp.poly_degree),
            )["mapping"],
            ds["x_fit"],
            ds["y_fit"],
            ds["x_probe"],
            ds["y_probe"],
            list(pool_cache["pool_nodes"]),
            pool_phi_probe,
            pool_norms_probe,
            list(pool_cache["pool_dims"]),
            rng,
            max_depth_eff,
            nvars,
            int(hp.poly_degree),
            var_dims=var_dims,
            topk=3,
        )
    )
    if expr is not None:
        row = _build_row(expr, action_id=int(residual_action_id), action_path=None, path_source="")
        if row is not None:
            build_rows.append(row)

    if not build_rows:
        return build_rows

    ranked = sorted(
        enumerate(build_rows),
        key=lambda item: (
            int(item[1].get("oracle_structural_mismatch_count", 10**6)),
            float(item[1].get("child_eff_mse", float("inf"))),
            str(item[1].get("child_expr", "")),
        ),
    )
    for rank, (_idx, row) in enumerate(ranked):
        row["oracle_truth_rank"] = int(rank)
        row["oracle_truth_rank_score"] = float(1.0 / (1.0 + float(rank)))
    return build_rows


def _oracle_best_mode_for_path(path_report: Mapping[str, Any]) -> str | None:
    mode_payloads = path_report.get("modes", {}) if isinstance(path_report, Mapping) else {}
    if not isinstance(mode_payloads, Mapping):
        return None
    best_key = None
    best_mode = None
    for mode_name, payload in mode_payloads.items():
        if not isinstance(payload, Mapping):
            continue
        truth_rank_raw = payload.get("truth_rank", None)
        truth_rank = int(truth_rank_raw) if truth_rank_raw is not None else 10**6
        top_rows = list(payload.get("top_replacements", []) or [])
        top_probe = top_rows[0].get("full_probe_mse", None) if top_rows else None
        top_probe_f = float(top_probe) if top_probe is not None else float("inf")
        key = (truth_rank, top_probe_f, str(mode_name))
        if best_key is None or key < best_key:
            best_key = key
            best_mode = str(mode_name)
    return best_mode


def generate_oracle_policy_pretrain_dataset(
    specs: Sequence[EquationSpec | str | pathlib.Path],
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seeds: Sequence[int] = (0,),
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    depth_min: int = 3,
    depth_max: int = 8,
    compare_modes: Sequence[str] | None = None,
    topk: int = 8,
    max_corrupt_paths_per_spec: int | None = None,
    sweep_all_paths: bool = False,
    sweep_max_paths: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build oracle-supervised controller pretraining rows from corrupted truths."""

    spec_objs: list[EquationSpec] = []
    for item in specs:
        if isinstance(item, EquationSpec):
            spec_objs.append(item)
        else:
            spec_objs.append(load_equation_spec(item))

    rows: list[dict[str, Any]] = []
    for spec in spec_objs:
        truth_ast = compile_target_ast(spec)
        truth_depth = int(node_depth(truth_ast))
        if truth_depth < int(depth_min) or truth_depth > int(depth_max):
            continue
        corrupt_paths = [tuple(int(v) for v in p) for p in collect_paths(truth_ast) if tuple(int(v) for v in p)]
        corrupt_paths.sort(key=lambda p: (len(p), tuple(p)))
        if max_corrupt_paths_per_spec is not None:
            corrupt_paths = corrupt_paths[: max(1, int(max_corrupt_paths_per_spec))]
        for seed in seeds:
            for corrupt_path in corrupt_paths:
                report = run_inverse_path_sweep_lab(
                    spec,
                    factorized_search_hp=factorized_search_hp,
                    seed=int(seed),
                    dtype=dtype,
                    enforce_dims=bool(enforce_dims),
                    corrupt_path=corrupt_path,
                    topk=int(topk),
                    compare_modes=compare_modes,
                    sweep_all_paths=bool(sweep_all_paths),
                    sweep_max_paths=sweep_max_paths,
                    verbose=False,
                )
                base_probe_mse = float(report.get("candidate_score", {}).get("probe_mse", float("inf")))
                path_labels: list[dict[str, Any]] = []
                for path_report in list(report.get("path_reports", []) or []):
                    if not isinstance(path_report, dict):
                        continue
                    mode_choice = _oracle_mode_choice(
                        dict(path_report.get("modes", {}) or {}),
                        base_probe_mse=base_probe_mse,
                    )
                    path_labels.append({
                        "path": list(path_report.get("path", []) or []),
                        "path_str": str(path_report.get("path_str", "")),
                        "relation": str(path_report.get("relation_to_reference", "unknown") or "unknown"),
                        "relation_to_selected": str(path_report.get("relation_to_selected", "unknown") or "unknown"),
                        "best_mode": mode_choice.get("mode", None),
                        "best_mode_truth_rank": mode_choice.get("truth_rank", None),
                        "best_mode_truth_probe_mse": mode_choice.get("truth_probe_mse", None),
                        "best_mode_top_full_probe_mse": mode_choice.get("top_full_probe_mse", None),
                        "improvement_estimate": float(mode_choice.get("improvement_estimate", 0.0) or 0.0),
                    })
                target_path, target_relation = _oracle_target_path_label(path_labels)
                controller_row = _oracle_controller_row_from_path_sweep(report)
                path_summaries = list(controller_row.get("path_summaries", []) or [])
                target_path_index = None
                if target_path is not None:
                    for idx, row in enumerate(path_summaries):
                        if tuple(int(v) for v in (row.get("path", []) or ())) == tuple(target_path):
                            target_path_index = int(idx)
                            break
                rows.append({
                    "spec_id": str(spec.id),
                    "seed": int(seed),
                    "truth_depth": int(truth_depth),
                    "corrupt_path": [int(v) for v in corrupt_path],
                    "corrupt_path_str": _format_path(corrupt_path),
                    "candidate_probe_mse": base_probe_mse,
                    "selected_path": None if report.get("selected_path", None) is None else list(report["selected_path"].get("path", []) or []),
                    "target_path": None if target_path is None else [int(v) for v in target_path],
                    "target_relation": str(target_relation),
                    "target_path_index": None if target_path_index is None else int(target_path_index),
                    "controller_row": controller_row,
                    "path_labels": path_labels,
                })
                if verbose:
                    print(
                        f"[oracle-pretrain] {spec.id} seed={int(seed)} corrupt={_format_path(corrupt_path)} "
                        f"target={None if target_path is None else _format_path(target_path)} relation={target_relation}"
                    )
    return _to_jsonable({
        "mode": "oracle_policy_pretrain_dataset",
        "n_rows": int(len(rows)),
        "config": {
            "seeds": [int(s) for s in seeds],
            "depth_min": int(depth_min),
            "depth_max": int(depth_max),
            "topk": int(topk),
            "compare_modes": [str(m) for m in (compare_modes or ("identity", "full", "affine"))],
            "max_corrupt_paths_per_spec": None if max_corrupt_paths_per_spec is None else int(max_corrupt_paths_per_spec),
            "sweep_all_paths": bool(sweep_all_paths),
            "sweep_max_paths": None if sweep_max_paths is None else int(sweep_max_paths),
        },
        "rows": rows,
    })


def generate_oracle_shared_candidate_pretrain_dataset(
    specs: Sequence[EquationSpec | str | pathlib.Path],
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seeds: Sequence[int] = (0,),
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    depth_min: int = 3,
    depth_max: int = 8,
    compare_modes: Sequence[str] | None = None,
    topk: int = 8,
    max_corrupt_paths_per_spec: int | None = None,
    sweep_all_paths: bool = False,
    sweep_max_paths: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build oracle repair-candidate rows aligned to the shared-candidate trainer."""

    spec_objs: list[EquationSpec] = []
    for item in specs:
        if isinstance(item, EquationSpec):
            spec_objs.append(item)
        else:
            spec_objs.append(load_equation_spec(item))

    dataset_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    skipped_specs: list[dict[str, Any]] = []
    for spec in spec_objs:
        try:
            truth_ast = compile_target_ast(spec)
            target_fn = compile_target_expression(spec)
        except Exception as exc:
            skipped_specs.append({
                "spec_id": str(spec.id),
                "reason": str(exc),
            })
            continue
        truth_depth = int(node_depth(truth_ast))
        if truth_depth < int(depth_min) or truth_depth > int(depth_max):
            continue
        corrupt_paths = [tuple(int(v) for v in p) for p in collect_paths(truth_ast) if tuple(int(v) for v in p)]
        corrupt_paths.sort(key=lambda p: (len(p), tuple(p)))
        if max_corrupt_paths_per_spec is not None:
            corrupt_paths = corrupt_paths[: max(1, int(max_corrupt_paths_per_spec))]
        for seed in seeds:
            ds = build_oracle_dataset(
                spec,
                target_fn,
                n_fit=int((factorized_search_hp if factorized_search_hp is not None else default_oracle_hyperparams()).n_fit),
                n_probe=int((factorized_search_hp if factorized_search_hp is not None else default_oracle_hyperparams()).n_probe),
                seed=int(seed),
                dtype=dtype,
            )
            for corrupt_path in corrupt_paths:
                report = run_inverse_path_sweep_lab(
                    spec,
                    factorized_search_hp=factorized_search_hp,
                    seed=int(seed),
                    dtype=dtype,
                    enforce_dims=bool(enforce_dims),
                    corrupt_path=corrupt_path,
                    topk=int(topk),
                    compare_modes=compare_modes,
                    sweep_all_paths=bool(sweep_all_paths),
                    sweep_max_paths=sweep_max_paths,
                    verbose=False,
                )
                controller_row = _oracle_controller_row_from_path_sweep(report)
                candidate_ast = report.get("candidate_expr_ast", None)
                if candidate_ast is None:
                    continue
                base_probe_mse = float(report.get("candidate_score", {}).get("probe_mse", float("inf")))
                oracle_row = dict(controller_row)
                oracle_row["spec_id"] = str(spec.id)
                oracle_row["seed"] = int(seed)
                oracle_row["truth_depth"] = int(truth_depth)
                oracle_row["oracle_corrupt_path"] = [int(v) for v in corrupt_path]
                oracle_row["oracle_corrupt_path_str"] = _format_path(corrupt_path)
                oracle_row["estimated_parent_eff_mse"] = float(base_probe_mse)
                oracle_row["estimated_parent_raw_mse"] = float(report.get("candidate_score", {}).get("fit_mse", base_probe_mse))
                oracle_row["inverse_repair_slate_id"] = f"oracle_shared_{spec.id}_{int(seed)}_{_format_path(corrupt_path)}"
                oracle_row["inverse_repair_slate"] = []
                oracle_row["controller_build_slate_id"] = f"oracle_build_{spec.id}_{int(seed)}_{_format_path(corrupt_path)}"
                oracle_row["controller_build_slate"] = []
                oracle_row["controller_build_slate_count"] = 0
                oracle_row["controller_build_slate_exact_observed_count"] = 0
                oracle_row["oracle_truth_in_slate"] = False
                oracle_row["oracle_best_truth_rank"] = None
                oracle_row["oracle_truth_rank_conditional"] = None
                oracle_row["oracle_truth_path"] = []
                oracle_row["oracle_truth_path_index"] = None

                path_meta: list[dict[str, Any]] = []
                for beam_rank, path_report in enumerate(list(report.get("path_reports", []) or [])):
                    if not isinstance(path_report, Mapping):
                        continue
                    path = tuple(int(v) for v in (path_report.get("path", []) or ()))
                    gate_row = path_report.get("gate_row", {}) if isinstance(path_report.get("gate_row", {}), Mapping) else {}
                    relation = str(path_report.get("relation_to_reference", "unknown") or "unknown")
                    truth_node = path_report.get("truth_subexpr_ast", None)
                    best_mode = _oracle_best_mode_for_path(path_report)
                    mode_reports = path_report.get("modes", {}) if isinstance(path_report.get("modes", {}), Mapping) else {}
                    mode_truth_ranks = [
                        int(payload.get("truth_rank"))
                        for payload in mode_reports.values()
                        if isinstance(payload, Mapping) and payload.get("truth_rank", None) is not None
                    ]
                    best_truth_rank = min(mode_truth_ranks) if mode_truth_ranks else None
                    second_truth_rank = sorted(mode_truth_ranks)[1] if len(mode_truth_ranks) > 1 else None
                    if beam_rank < len(list(oracle_row.get("path_summaries", []) or [])):
                        path_summary = oracle_row["path_summaries"][beam_rank]
                        if isinstance(path_summary, Mapping):
                            path_summary = dict(path_summary)
                            path_summary["oracle_relation_to_reference"] = relation
                            path_summary["oracle_is_reference_path"] = bool(relation == "same")
                            path_summary["oracle_best_mode"] = best_mode
                            path_summary["oracle_truth_present_under_path"] = bool(best_truth_rank is not None)
                            path_summary["oracle_best_truth_rank"] = None if best_truth_rank is None else int(best_truth_rank)
                            path_summary["oracle_second_truth_rank"] = None if second_truth_rank is None else int(second_truth_rank)
                            path_summary["oracle_truth_rank_margin"] = (
                                None
                                if best_truth_rank is None or second_truth_rank is None
                                else float(int(second_truth_rank) - int(best_truth_rank))
                            )
                            oracle_row["path_summaries"][beam_rank] = path_summary
                    if relation == "same" and oracle_row.get("oracle_truth_path_index", None) is None:
                        oracle_row["oracle_truth_path"] = [int(v) for v in path]
                        oracle_row["oracle_truth_path_index"] = int(beam_rank)
                    if best_truth_rank is not None:
                        prev_best_truth_rank = oracle_row.get("oracle_best_truth_rank", None)
                        if prev_best_truth_rank is None or int(best_truth_rank) < int(prev_best_truth_rank):
                            oracle_row["oracle_best_truth_rank"] = int(best_truth_rank)
                            oracle_row["oracle_truth_rank_conditional"] = int(best_truth_rank)
                    for mode_name, mode_payload in mode_reports.items():
                        if not isinstance(mode_payload, Mapping):
                            continue
                        inv_summary = mode_payload.get("inverse_target", {}) if isinstance(mode_payload.get("inverse_target", {}), Mapping) else {}
                        top_rows = list(mode_payload.get("top_replacements", []) or [])
                        truth_rank = mode_payload.get("truth_rank", None)
                        for local_rank, cand in enumerate(top_rows[: max(1, int(topk))]):
                            if not isinstance(cand, Mapping):
                                continue
                            expr_ast = cand.get("expr_ast", None)
                            if expr_ast is None:
                                continue
                            child_ast = replace_at(candidate_ast, path, expr_ast)
                            child_expr = node_str(child_ast)
                            child_eff = float(cand.get("full_probe_mse", float("inf")))
                            child_raw = float(cand.get("full_fit_mse", child_eff))
                            oracle_is_truth_candidate = bool(truth_node is not None and expr_ast == truth_node)
                            oracle_truth_rank_score = None if truth_rank is None else float(1.0 / (1.0 + max(0, int(truth_rank))))
                            oracle_mode_is_best = bool(best_mode == str(mode_name))
                            row_payload = {
                                "tuple_provenance": "oracle_path_sweep",
                                "proposal_family": "oracle_path_sweep",
                                "generation_source": "oracle_path_sweep",
                                "path": list(path),
                                "path_source": "oracle_path_sweep",
                                "target_mode": str(mode_name),
                                "action": "inv_steer",
                                "beam_rank": int(beam_rank),
                                "local_rank": int(local_rank),
                                "weighted_rel_gain": gate_row.get("weighted_rel_gain", None),
                                "weighted_rel_gain_pre_cut": gate_row.get("weighted_rel_gain_pre_cut", gate_row.get("weighted_rel_gain", None)),
                                "rel_gain": gate_row.get("rel_gain", None),
                                "valid_frac": gate_row.get("valid_frac", inv_summary.get("valid_fraction_fit", None)),
                                "confidence": gate_row.get("confidence", inv_summary.get("confidence", None)),
                                "static_score": gate_row.get("static_score", None),
                                "transport_rel": gate_row.get("transport_rel", None),
                                "branch_factor": gate_row.get("branch_factor", None),
                                "cut_factor": gate_row.get("cut_factor", None),
                                "branch_support": gate_row.get("branch_support", None),
                                "branch_positive_count": gate_row.get("branch_positive_count", None),
                                "family_scale": gate_row.get("family_scale", None),
                                "profile_exact_monotone": gate_row.get("profile_exact_monotone", False),
                                "profile_has_periodic": gate_row.get("profile_has_periodic", False),
                                "profile_has_muldiv": gate_row.get("profile_has_muldiv", False),
                                "profile_has_explogsqrt": gate_row.get("profile_has_explogsqrt", False),
                                "target_mapping_kind": inv_summary.get("effective_mapping_kind", inv_summary.get("mapping_kind", "")),
                                "local_probe_mse": float(cand.get("local_probe_mse", child_eff)),
                                "local_fit_mse": float(cand.get("local_fit_mse", child_raw)),
                                "local_corr_probe": float(cand.get("local_corr_probe", 0.0) or 0.0),
                                "local_mapping_kind": str(cand.get("local_mapping_kind", "")),
                                "local_mapping_nparams": int(_oracle_mapping_nparams(cand.get("local_mapping", None))),
                                "candidate_subtree_size": int(node_size(expr_ast)),
                                "candidate_subtree_depth": int(node_depth(expr_ast)),
                                "candidate_subtree_size_delta": int(node_size(expr_ast) - node_size(get_at(candidate_ast, path))),
                                "candidate_subtree_depth_delta": int(node_depth(expr_ast) - node_depth(get_at(candidate_ast, path))),
                                "candidate_child_size": int(node_size(child_ast)),
                                "candidate_child_depth": int(node_depth(child_ast)),
                                "candidate_child_size_delta": int(node_size(child_ast) - node_size(candidate_ast)),
                                "candidate_child_depth_delta": int(node_depth(child_ast) - node_depth(candidate_ast)),
                                "candidate_root_op": _oracle_tuple_root_op(child_ast),
                                "local_candidate_count": int(len(top_rows)),
                                "child_key": child_expr,
                                "child_expr": child_expr,
                                "child_eff_mse": float(child_eff),
                                "child_raw_mse": float(child_raw),
                                "dedup_kept": True,
                                "exact_child_score_observed": True,
                                "oracle_relation_to_reference": relation,
                                "oracle_path_is_correct": bool(relation == "same"),
                                "oracle_path_is_near": bool(relation in {"same", "ancestor", "descendant"}),
                                "oracle_truth_present_under_path": bool(truth_rank is not None),
                                "oracle_mode_is_best": oracle_mode_is_best,
                                "oracle_truth_rank": None if truth_rank is None else int(truth_rank),
                                "oracle_truth_rank_score": oracle_truth_rank_score,
                                "oracle_is_truth_candidate": oracle_is_truth_candidate,
                                "oracle_mapping_stable": bool(oracle_mode_is_best and truth_rank is not None),
                                "oracle_mapping_fragile": bool((not oracle_mode_is_best) and truth_rank is not None),
                            }
                            oracle_row["inverse_repair_slate"].append(shared_candidate_row_dict(row_payload, route_source="repair"))
                            if oracle_is_truth_candidate:
                                oracle_row["oracle_truth_in_slate"] = True
                    path_meta.append({
                        "path": list(path),
                        "relation_to_reference": relation,
                        "best_mode": best_mode,
                        "best_truth_rank": None if best_truth_rank is None else int(best_truth_rank),
                    })
                reference_paths = _dedupe_paths([
                    tuple(int(v) for v in corrupt_path),
                    tuple(int(v) for v in (oracle_row.get("oracle_truth_path", []) or ())),
                    tuple(int(v) for v in (report.get("selected_path", {}) or {}).get("path", []) or ()),
                    *_minimal_mismatch_paths(candidate_ast, truth_ast),
                ])
                build_rows = _oracle_collect_build_candidate_rows(
                    spec=spec,
                    hp=factorized_search_hp if factorized_search_hp is not None else default_oracle_hyperparams(),
                    run_seed=int(seed),
                    candidate_ast=candidate_ast,
                    truth_ast=truth_ast,
                    ds=ds,
                    reference_paths=reference_paths,
                    selected_path=(report.get("selected_path", {}) or {}).get("path", []),
                    base_probe_mse=base_probe_mse,
                )
                oracle_row["controller_build_slate"] = build_rows
                oracle_row["controller_build_slate_count"] = int(len(build_rows))
                oracle_row["controller_build_slate_exact_observed_count"] = int(
                    sum(1 for row in build_rows if bool(row.get("exact_child_score_observed", False)))
                )
                if any(bool(row.get("oracle_is_truth_candidate", False)) for row in build_rows):
                    oracle_row["oracle_truth_in_slate"] = True
                if len(list(oracle_row.get("inverse_repair_slate", []) or [])) < 2:
                    continue
                dataset_rows.append(oracle_row)
                meta_rows.append({
                    "spec_id": str(spec.id),
                    "seed": int(seed),
                    "truth_depth": int(truth_depth),
                    "corrupt_path": [int(v) for v in corrupt_path],
                    "candidate_probe_mse": float(base_probe_mse),
                    "n_candidates": int(len(oracle_row["inverse_repair_slate"])),
                    "n_build_candidates": int(len(oracle_row["controller_build_slate"])),
                    "path_meta": path_meta,
                })
                if verbose:
                    print(
                        f"[oracle-shared-candidate] {spec.id} seed={int(seed)} corrupt={_format_path(corrupt_path)} "
                        f"repair={int(len(oracle_row['inverse_repair_slate']))} "
                        f"build={int(len(oracle_row['controller_build_slate']))}"
                    )
    return _to_jsonable({
        "mode": "oracle_shared_candidate_pretrain_dataset",
        "n_rows": int(len(dataset_rows)),
        "n_skipped_specs": int(len(skipped_specs)),
        "config": {
            "seeds": [int(s) for s in seeds],
            "depth_min": int(depth_min),
            "depth_max": int(depth_max),
            "topk": int(topk),
            "compare_modes": [str(m) for m in (compare_modes or ("identity", "full", "affine"))],
            "max_corrupt_paths_per_spec": None if max_corrupt_paths_per_spec is None else int(max_corrupt_paths_per_spec),
            "sweep_all_paths": bool(sweep_all_paths),
            "sweep_max_paths": None if sweep_max_paths is None else int(sweep_max_paths),
        },
        "rows": dataset_rows,
        "meta_rows": meta_rows,
        "skipped_specs": skipped_specs,
    })


def generate_oracle_opportunity_shadow_dataset(
    specs: Sequence[EquationSpec | str | pathlib.Path],
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seeds: Sequence[int] = (0,),
    dtype: torch.dtype = torch.float64,
    depth_min: int = 3,
    depth_max: int = 8,
    topk: int = 8,
    max_corrupt_paths_per_spec: int | None = None,
    sweep_all_paths: bool = False,
    sweep_max_paths: int | None = None,
    verbose: bool = True,
    shadow_config=None,
) -> dict[str, Any]:
    from .opportunity_shadow_eval import generate_oracle_opportunity_shadow_dataset as _impl

    return _impl(
        specs,
        factorized_search_hp=factorized_search_hp,
        seeds=seeds,
        dtype=dtype,
        depth_min=depth_min,
        depth_max=depth_max,
        topk=topk,
        max_corrupt_paths_per_spec=max_corrupt_paths_per_spec,
        sweep_all_paths=sweep_all_paths,
        sweep_max_paths=sweep_max_paths,
        verbose=verbose,
        shadow_config=shadow_config,
    )


def _sympy_scope_for_spec(spec: EquationSpec):
    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover - dependency checked by caller
        raise RuntimeError("inverse oracle lab requires sympy") from exc

    nvars = len(spec.variables) + len(spec.constants)
    syms = [sp.Symbol(f"x{i}", real=True) for i in range(nvars)]
    scope: dict[str, Any] = {
        "sin": sp.sin,
        "cos": sp.cos,
        "asin": sp.asin,
        "acos": sp.acos,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "sqr": lambda z: z ** 2,
        "pi": sp.pi,
        "E": sp.E,
    }
    for i, sym in enumerate(syms):
        scope[f"x{i}"] = sym
    for i, v in enumerate(spec.variables):
        scope[v.name] = syms[i]
    for j, c in enumerate(spec.constants, start=len(spec.variables)):
        scope[c.name] = syms[j]
    return syms, scope


def _fold_tuple_ast(op: str, args: Sequence[tuple]) -> tuple:
    rows = list(args or ())
    if not rows:
        raise ValueError(f"cannot fold empty {op} expression")
    cur = rows[0]
    for arg in rows[1:]:
        cur = (str(op), cur, arg)
    return cur


def _sympy_to_tuple_ast(expr: Any) -> tuple:
    import sympy as sp

    if getattr(expr, "is_number", False):
        return ("const", float(expr))

    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name.startswith("x") and name[1:].isdigit():
            return ("var", int(name[1:]))
        raise ValueError(f"Unknown symbol: {name}")

    if isinstance(expr, sp.Add):
        return simplify(_fold_tuple_ast("add", [_sympy_to_tuple_ast(arg) for arg in list(expr.args)]))

    if isinstance(expr, sp.Mul):
        return simplify(_fold_tuple_ast("mul", [_sympy_to_tuple_ast(arg) for arg in list(expr.args)]))

    if isinstance(expr, sp.Pow):
        base = _sympy_to_tuple_ast(expr.args[0])
        exp_val = expr.args[1]
        if not getattr(exp_val, "is_number", False):
            raise ValueError(f"Unsupported symbolic exponent: {expr}")
        exp_f = float(exp_val)
        if math.isclose(exp_f, 0.5):
            return ("sqrt", base)
        if math.isclose(exp_f, 2.0):
            return ("sqr", base)
        if math.isclose(exp_f, -1.0):
            return ("div", ("const", 1.0), base)
        if math.isclose(exp_f, -0.5):
            return ("div", ("const", 1.0), ("sqrt", base))
        if math.isclose(exp_f, -2.0):
            return ("div", ("const", 1.0), ("sqr", base))
        if math.isclose(exp_f, round(exp_f)) and abs(exp_f) <= 6.0:
            exp_i = int(round(exp_f))
            if exp_i == 0:
                return ("const", 1.0)
            atoms = [base] * abs(exp_i)
            built = _fold_tuple_ast("mul", atoms)
            if exp_i > 0:
                return simplify(built)
            return simplify(("div", ("const", 1.0), built))
        raise ValueError(f"Unsupported numeric exponent in tuple-AST parser: {expr}")

    func = getattr(expr, "func", None)
    if func in (sp.sin, sp.cos, sp.asin, sp.acos, sp.exp, sp.log, sp.Abs):
        child = _sympy_to_tuple_ast(expr.args[0])
        op_map = {
            sp.sin: "sin",
            sp.cos: "cos",
            sp.asin: "asin",
            sp.acos: "acos",
            sp.exp: "exp",
            sp.log: "log",
            sp.Abs: "abs",
        }
        op = str(op_map[func])
        if op == "abs":
            raise ValueError(f"Unsupported absolute-value tuple AST: {expr}")
        return simplify((op, child))

    raise ValueError(f"Unsupported sympy type for tuple-AST conversion: {type(expr).__name__}: {expr}")


def parse_tuple_ast_expression(expr_text: str, spec: EquationSpec) -> tuple:
    """Parse a symbolic expression into a factorized_search tuple-AST.

    The parser accepts either declared oracle variable/constant names or the
    canonical explorer aliases ``x0``, ``x1``, ... for the concatenated input
    columns ``variables + constants``.
    """

    if str(expr_text).strip() == "":
        raise ValueError("expression must be non-empty")

    try:
        import sympy as sp
    except Exception as exc:
        raise RuntimeError("parse_tuple_ast_expression requires sympy") from exc

    _, scope = _sympy_scope_for_spec(spec)
    expr = sp.sympify(str(expr_text), locals=scope)
    nvars = len(spec.variables) + len(spec.constants)
    try:
        from nestynet_sr.sr_core.sympy_bridge import sympy_to_nestynet
        from .bridge import nestynet_to_factorized_search

        return nestynet_to_factorized_search(sympy_to_nestynet(expr, nvars))
    except Exception:
        return _sympy_to_tuple_ast(expr)


def compile_target_ast(spec: EquationSpec) -> tuple:
    """Compile ``spec.target_expr`` into the explorer's tuple-AST format."""
    return parse_tuple_ast_expression(spec.target_expr, spec)


def _default_corruption_replacement(
    truth_ast: tuple,
    path: Sequence[int],
    *,
    nvars: int,
    var_dims: Sequence[Sequence[float]] | None,
) -> tuple:
    truth_sub = get_at(truth_ast, tuple(path))
    truth_dim = node_dims(truth_sub, var_dims) if var_dims is not None else None
    for node in build_pool(nvars):
        if node == truth_sub:
            continue
        if var_dims is not None and truth_dim is not None:
            nd = node_dims(node, var_dims)
            if nd is None or not dims_eq(nd, truth_dim):
                continue
        return node
    return ("var", 0)


def build_candidate_ast_for_inverse_lab(
    spec: EquationSpec,
    *,
    candidate_expr: str | None = None,
    corrupt_path: str | Sequence[int] | None = None,
    replacement_expr: str | None = None,
    var_dims: Sequence[Sequence[float]] | None = None,
) -> tuple[tuple, tuple | None, tuple | None]:
    """Build a candidate AST for inverse-steering experiments.

    Returns ``(candidate_ast, truth_ast, used_corrupt_path)``.
    """

    truth_ast = compile_target_ast(spec)
    nvars = len(spec.variables) + len(spec.constants)

    if candidate_expr is not None:
        cand = parse_tuple_ast_expression(candidate_expr, spec)
        return cand, truth_ast, None

    if corrupt_path is None:
        return truth_ast, truth_ast, None

    path = _parse_path(corrupt_path)
    if path not in set(collect_paths(truth_ast)):
        raise ValueError(
            f"corrupt_path={_format_path(path)} is not present in truth AST {node_str(truth_ast)}"
        )

    if replacement_expr is not None:
        repl = parse_tuple_ast_expression(replacement_expr, spec)
    else:
        repl = _default_corruption_replacement(truth_ast, path, nvars=nvars, var_dims=var_dims)

    cand = replace_at(truth_ast, path, repl)
    return cand, truth_ast, path

def invert_mapping_target(
    y_target: torch.Tensor,
    mapping: dict[str, Any] | None,
    *,
    pred_ref: torch.Tensor | None = None,
    safe_eps: float = 1.0e-12,
    allow_identity_fallback: bool = True,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, float, bool, str]:
    """Delegate to the production explorer inverse helper to avoid lab drift."""
    return _explorer_invert_mapping_target(
        y_target,
        mapping,
        pred_ref=pred_ref,
        safe_eps=float(safe_eps),
        allow_identity_fallback=bool(allow_identity_fallback),
        confidence_mode=str(confidence_mode),
        confidence_target_gain=float(confidence_target_gain),
        confidence_floor=float(confidence_floor),
    )


def _invert_unary_context(
    op: str,
    parent_target: torch.Tensor,
    *,
    child_pred_ref: torch.Tensor | None,
    safe_eps: float,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, float, str]:
    """Delegate to the production explorer inverse helper to avoid lab drift."""
    child_t, out_mask, conf, _pw, note = _explorer_invert_unary_context(
        op,
        parent_target,
        child_pred_ref=child_pred_ref,
        safe_eps=float(safe_eps),
        confidence_mode=str(confidence_mode),
        confidence_target_gain=float(confidence_target_gain),
        confidence_floor=float(confidence_floor),
    )
    return child_t, out_mask, conf, note


def _invert_binary_context(
    op: str,
    parent_target: torch.Tensor,
    *,
    child_slot: int,
    other_pred: torch.Tensor,
    safe_eps: float,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, float, str]:
    """Delegate to the production explorer inverse helper to avoid lab drift."""
    child_t, out_mask, conf, _pw, note = _explorer_invert_binary_context(
        op,
        parent_target,
        child_slot=int(child_slot),
        other_pred=other_pred,
        safe_eps=float(safe_eps),
        confidence_mode=str(confidence_mode),
        confidence_target_gain=float(confidence_target_gain),
        confidence_floor=float(confidence_floor),
    )
    return child_t, out_mask, conf, note


@torch.no_grad()
def invert_context_target(
    candidate_ast: tuple,
    path: str | Sequence[int] | None,
    x: torch.Tensor,
    y_target: torch.Tensor,
    *,
    mapping: dict[str, Any] | None = None,
    safe_eps: float = 1.0e-12,
    allow_identity_fallback: bool = True,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    branch_beam_width: int = 1,
) -> InverseTarget:
    """Compute a pseudo-target via the production explorer inverse core."""

    pp = _parse_path(path)
    inv = _explorer_invert_context_target(
        candidate_ast,
        pp,
        x,
        y_target,
        mapping=mapping,
        safe_eps=float(safe_eps),
        allow_identity_fallback=bool(allow_identity_fallback),
        confidence_mode=str(confidence_mode),
        confidence_target_gain=float(confidence_target_gain),
        confidence_floor=float(confidence_floor),
        branch_beam_width=max(1, int(branch_beam_width)),
    )
    steps = tuple(
        InverseStep(
            parent_path=tuple(int(v) for v in st.parent_path),
            op=str(st.op),
            child_slot=int(st.child_slot),
            valid_fraction=float(st.valid_fraction),
            confidence=float(st.confidence),
            note=str(st.note),
        )
        for st in tuple(inv.steps)
    )
    return InverseTarget(
        path=tuple(int(v) for v in inv.path),
        target=_ensure_col(inv.target),
        valid_mask=_bool_col(inv.valid_mask),
        confidence=float(inv.confidence),
        mapping_inverted=bool(inv.mapping_inverted),
        mapping_kind=str(inv.mapping_kind),
        steps=steps,
    )


@torch.no_grad()
def _score_expr_against_target(
    expr_ast: tuple,
    *,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    poly_degree: int,
) -> dict[str, Any] | None:
    try:
        pred_fit = eval_node(expr_ast, x_fit)
    except Exception:
        return None
    if not torch.isfinite(pred_fit).all():
        return None

    fb = fit_best(pred_fit, y_fit, int(poly_degree))
    if fb is None:
        return None
    fit_mse, mapping = fb

    try:
        pred_probe = eval_node(expr_ast, x_probe)
    except Exception:
        return None
    if not torch.isfinite(pred_probe).all():
        return None

    y_hat_probe = eval_mapping(pred_probe, mapping)
    if not torch.isfinite(y_hat_probe).all():
        return None

    probe_mse = float(((y_probe - y_hat_probe) ** 2).mean().item())
    return {
        "expr": node_str(expr_ast),
        "expr_ast": expr_ast,
        "mapping": mapping,
        "mapping_kind": str(mapping.get("kind", "")),
        "fit_mse": float(fit_mse),
        "probe_mse": float(probe_mse),
        "corr_probe": _corrcoef(y_hat_probe, y_probe),
    }


@torch.no_grad()
def _score_node_on_local_target(
    node: tuple,
    *,
    x_fit: torch.Tensor,
    t_fit: torch.Tensor,
    x_probe: torch.Tensor,
    t_probe: torch.Tensor,
    poly_degree: int,
    local_score_mode: str = "affine",
) -> dict[str, Any] | None:
    if int(x_fit.shape[0]) < 4 or int(x_probe.shape[0]) < 4:
        return None
    try:
        pred_fit = eval_node(node, x_fit)
    except Exception:
        return None
    if not torch.isfinite(pred_fit).all():
        return None

    try:
        pred_probe = eval_node(node, x_probe)
    except Exception:
        return None
    if not torch.isfinite(pred_probe).all():
        return None

    mode = _normalize_inverse_local_score_mode(local_score_mode, default="affine")

    mapping: dict[str, Any]
    if mode == "strict":
        t_hat_probe = pred_probe
        fit_mse = float(((t_fit - pred_fit) ** 2).mean().item())
        probe_mse = float(((t_probe - t_hat_probe) ** 2).mean().item())
        mapping = {"kind": "identity"}
    elif mode == "affine":
        f = pred_fit.squeeze(-1)
        A = torch.stack([torch.ones_like(f), f], dim=1)
        try:
            sol = torch.linalg.lstsq(A, t_fit.squeeze(-1)).solution
        except Exception:
            return None
        if not torch.isfinite(sol).all():
            return None
        fp = pred_probe.squeeze(-1)
        fit_hat = (sol[0] + sol[1] * f).unsqueeze(-1)
        t_hat_probe = (sol[0] + sol[1] * fp).unsqueeze(-1)
        if (not torch.isfinite(fit_hat).all()) or (not torch.isfinite(t_hat_probe).all()):
            return None
        fit_mse = float(((t_fit - fit_hat) ** 2).mean().item())
        probe_mse = float(((t_probe - t_hat_probe) ** 2).mean().item())
        mapping = {"kind": "affine", "a": float(sol[1].item()), "b": float(sol[0].item())}
    else:
        fb = fit_best(pred_fit, t_fit, int(poly_degree))
        if fb is None:
            return None
        fit_mse, mapping = fb
        t_hat_probe = eval_mapping(pred_probe, mapping)
        if not torch.isfinite(t_hat_probe).all():
            return None
        probe_mse = float(((t_probe - t_hat_probe) ** 2).mean().item())

    if (not math.isfinite(float(fit_mse))) or (not math.isfinite(float(probe_mse))):
        return None
    return {
        "expr": node_str(node),
        "expr_ast": node,
        "local_mapping": mapping,
        "local_mapping_kind": str(mapping.get("kind", "")),
        "local_fit_mse": float(fit_mse),
        "local_probe_mse": float(probe_mse),
        "local_corr_probe": _corrcoef(t_hat_probe, t_probe),
    }


def _prepare_local_target_slices(
    x_fit: torch.Tensor,
    t_fit: torch.Tensor,
    mask_fit: torch.Tensor,
    x_probe: torch.Tensor,
    t_probe: torch.Tensor,
    mask_probe: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xf, tf = _slice_by_mask(x_fit, t_fit, mask_fit)
    xp, tp = _slice_by_mask(x_probe, t_probe, mask_probe)
    return xf, tf, xp, tp


@torch.no_grad()
def _cheap_affine_probe_stats(
    node: tuple,
    *,
    x_fit: torch.Tensor,
    t_fit: torch.Tensor,
    x_probe: torch.Tensor,
    t_probe: torch.Tensor,
) -> tuple[float, float] | None:
    if int(x_fit.shape[0]) < 4 or int(x_probe.shape[0]) < 4:
        return None
    try:
        pf = eval_node(node, x_fit)
        pp = eval_node(node, x_probe)
    except Exception:
        return None
    if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
        return None
    f = pf.squeeze(-1)
    A = torch.stack([torch.ones_like(f), f], dim=1)
    try:
        sol = torch.linalg.lstsq(A, t_fit.squeeze(-1)).solution
    except Exception:
        return None
    if not torch.isfinite(sol).all():
        return None
    fp = pp.squeeze(-1)
    yhat = (sol[0] + sol[1] * fp).unsqueeze(-1)
    if not torch.isfinite(yhat).all():
        return None
    mse = float(((t_probe - yhat) ** 2).mean().item())
    corr = _corrcoef(yhat, t_probe)
    return mse, corr


@torch.no_grad()
def _oracle_build_pool_cache(
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    *,
    var_dims: Sequence[Sequence[float]] | None,
) -> dict[str, Any]:
    nvars = int(x_fit.shape[1])
    pool_nodes = build_pool(nvars)
    pool_dims = [node_dims(n, var_dims) for n in pool_nodes] if var_dims is not None else [None] * len(pool_nodes)

    def _safe_eval(node: tuple, x: torch.Tensor) -> torch.Tensor:
        try:
            vals = eval_node(node, x).squeeze(-1)
        except Exception:
            vals = torch.zeros((int(x.shape[0]),), dtype=x.dtype, device=x.device)
        if not torch.isfinite(vals).all():
            vals = torch.zeros_like(vals)
        return vals

    pool_phi_fit = torch.stack([_safe_eval(node, x_fit) for node in pool_nodes], dim=1)
    pool_phi_probe = torch.stack([_safe_eval(node, x_probe) for node in pool_nodes], dim=1)
    return {
        "pool_nodes": pool_nodes,
        "pool_dims": pool_dims,
        "pool_phi_fit": pool_phi_fit,
        "pool_phi_probe": pool_phi_probe,
    }


@torch.no_grad()
def _collect_shared_explorer_local_candidates_for_path(
    candidate_ast: tuple,
    path: Sequence[int],
    *,
    target_specs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ds: dict[str, Any],
    hp: FactorizedSearchConfig,
    pool_cache: dict[str, Any],
    var_dims: Sequence[Sequence[float]] | None,
    local_score_mode: str,
) -> list[tuple]:
    path_t = tuple(int(v) for v in path)
    current_node = get_at(candidate_ast, path_t)
    target_dim = node_dims(current_node, var_dims) if var_dims is not None else None
    pool_nodes = list(pool_cache.get("pool_nodes", []) or [])
    pool_dims = list(pool_cache.get("pool_dims", []) or [])
    pool_phi_fit = pool_cache.get("pool_phi_fit")
    pool_phi_probe = pool_cache.get("pool_phi_probe")
    if pool_phi_fit is None or pool_phi_probe is None:
        raise ValueError("pool_cache is missing pool feature tensors")

    max_depth_eff = max(int(node_depth(candidate_ast)), int(getattr(hp, "max_depth", node_depth(candidate_ast))))
    topk_terms = max(2, int(getattr(hp, "inverse_topk_terms", 6)))
    shortlist_mult = max(1, int(getattr(hp, "inverse_shortlist_mult", 4)))
    local_mode = _normalize_inverse_local_score_mode(local_score_mode, default="affine")
    safe_eps_eff = getattr(hp, "inverse_safe_eps", None)
    if safe_eps_eff is None:
        safe_eps_eff = getattr(hp, "refine_safe_eps", 1.0e-12)
    safe_eps_eff = float(safe_eps_eff or 1.0e-12)

    seen_nodes: set[str] = set()
    shared_nodes: list[tuple] = []

    def _add_node(node: tuple) -> None:
        key = node_str(node)
        if key in seen_nodes:
            return
        seen_nodes.add(key)
        shared_nodes.append(node)

    _add_node(current_node)

    for t_fit_raw, mask_fit_raw, t_probe_raw, mask_probe_raw in target_specs:
        mfit = _bool_col(mask_fit_raw).squeeze(-1)
        mprobe = _bool_col(mask_probe_raw).squeeze(-1)
        if int(mfit.sum().item()) < 4 or int(mprobe.sum().item()) < 4:
            continue
        xf, tf = _slice_by_mask(ds["x_fit"], t_fit_raw, mask_fit_raw)
        xp, tp = _slice_by_mask(ds["x_probe"], t_probe_raw, mask_probe_raw)
        if int(xf.shape[0]) < 4 or int(xp.shape[0]) < 4:
            continue
        idxs = _inverse_pool_shortlist(
            pool_phi_fit,
            t_fit_raw,
            mask_fit_raw,
            pool_dims=pool_dims if var_dims is not None else None,
            target_dim=target_dim,
            shortlist_k=max(topk_terms, topk_terms * shortlist_mult),
        )
        if not idxs:
            continue
        cand_nodes = _inverse_collect_local_repair_candidates(
            parent_node=candidate_ast,
            path=path_t,
            sub=current_node,
            target_dim=target_dim,
            xf=xf,
            tf=tf,
            xp=xp,
            tp=tp,
            wf=None,
            wp=None,
            mfit=mfit,
            mprobe=mprobe,
            pool_nodes=pool_nodes,
            pool_dims=pool_dims,
            pool_phi_fit=pool_phi_fit,
            pool_phi_probe=pool_phi_probe,
            idxs=idxs,
            poly_degree=int(hp.poly_degree),
            local_mode=local_mode,
            topk_terms=topk_terms,
            shortlist_mult=shortlist_mult,
            safe_eps=safe_eps_eff,
            var_dims=var_dims,
            max_depth=max_depth_eff,
            micro_search_enable=bool(getattr(hp, "inverse_micro_search_enable", False)),
            micro_search_max_depth=int(getattr(hp, "inverse_micro_search_max_depth", 3)),
            micro_search_beam_width=int(getattr(hp, "inverse_micro_search_beam_width", 24)),
            micro_search_topk=int(getattr(hp, "inverse_micro_search_topk", 16)),
            micro_search_seed_terms=int(getattr(hp, "inverse_micro_search_seed_terms", 8)),
        )
        for node in cand_nodes:
            _add_node(node)
    return shared_nodes


@torch.no_grad()
def _rank_local_replacements(
    candidate_ast: tuple,
    path: Sequence[int],
    *,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    t_fit: torch.Tensor,
    mask_fit: torch.Tensor,
    t_probe: torch.Tensor,
    mask_probe: torch.Tensor,
    poly_degree: int,
    topk: int | None,
    var_dims: Sequence[Sequence[float]] | None,
    truth_node: tuple | None,
    local_score_mode: str = "affine",
    include_truth_candidate: bool = False,
    proposal_nodes: Sequence[tuple] | None = None,
    pinned_nodes: Sequence[tuple] | None = None,
    preselect_limit: int | None = None,
) -> list[dict[str, Any]]:
    path_t = tuple(int(v) for v in path)
    nvars = int(x_fit.shape[1])
    target_dim = node_dims(get_at(candidate_ast, path_t), var_dims) if var_dims is not None else None
    xf, tf, xp, tp = _prepare_local_target_slices(x_fit, t_fit, mask_fit, x_probe, t_probe, mask_probe)
    if int(xf.shape[0]) < 4 or int(xp.shape[0]) < 4:
        return []

    pinned_exprs = {node_str(node) for node in (pinned_nodes or [])}
    if include_truth_candidate and truth_node is not None:
        pinned_exprs.add(node_str(truth_node))

    seen: set[str] = set()
    nodes: list[tuple[str, tuple]] = []
    current_node = get_at(candidate_ast, path_t)
    nodes.append(("current", current_node))
    if include_truth_candidate and truth_node is not None:
        nodes.append(("truth", truth_node))
    for node in (pinned_nodes or []):
        nodes.append(("pinned", node))
    if proposal_nodes is not None and len(proposal_nodes) > 0:
        for node in proposal_nodes:
            nodes.append(("proposal", node))
    else:
        for node in build_pool(nvars):
            nodes.append(("pool", node))

    prelim: list[tuple[str, tuple, float, float, bool]] = []
    for source, node in nodes:
        expr_key = node_str(node)
        if expr_key in seen:
            continue
        seen.add(expr_key)

        if var_dims is not None and target_dim is not None:
            nd = node_dims(node, var_dims)
            if nd is None or not dims_eq(nd, target_dim):
                continue

        cheap = _cheap_affine_probe_stats(
            node,
            x_fit=xf,
            t_fit=tf,
            x_probe=xp,
            t_probe=tp,
        )
        if cheap is None:
            continue
        cheap_mse, cheap_corr = cheap
        expr_key = node_str(node)
        prelim.append((source, node, float(cheap_mse), float(cheap_corr), expr_key in pinned_exprs))

    if not prelim:
        return []

    prelim.sort(key=lambda row: (row[2], -abs(row[3]), node_size(row[1]), node_str(row[1])))
    if preselect_limit is None:
        if topk is None:
            preselect = len(prelim)
        else:
            preselect = max(8, min(24, int(topk) * 4))
    else:
        preselect = max(1, int(preselect_limit))
    shortlist = list(prelim[: min(len(prelim), preselect)])
    pinned_keys = {node_str(row[1]) for row in shortlist}
    for row in prelim:
        if bool(row[4]) and node_str(row[1]) not in pinned_keys:
            shortlist.append(row)
            pinned_keys.add(node_str(row[1]))

    rows: list[dict[str, Any]] = []
    for source, node, cheap_mse, cheap_corr, is_pinned in shortlist:
        local = _score_node_on_local_target(
            node,
            x_fit=xf,
            t_fit=tf,
            x_probe=xp,
            t_probe=tp,
            poly_degree=poly_degree,
            local_score_mode=local_score_mode,
        )
        if local is None:
            continue

        repaired_ast = replace_at(candidate_ast, path_t, node)
        full = _score_expr_against_target(
            repaired_ast,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            poly_degree=poly_degree,
        )
        if full is None:
            continue

        row = {
            "source": source,
            "expr": node_str(node),
            "expr_ast": node,
            "is_pinned": bool(is_pinned),
            "cheap_affine_probe_mse": float(cheap_mse),
            "cheap_affine_corr_probe": float(cheap_corr),
            "local_probe_mse": float(local["local_probe_mse"]),
            "local_fit_mse": float(local["local_fit_mse"]),
            "local_corr_probe": float(local["local_corr_probe"]),
            "local_mapping": local["local_mapping"],
            "local_mapping_kind": local["local_mapping_kind"],
            "full_probe_mse": float(full["probe_mse"]),
            "full_fit_mse": float(full["fit_mse"]),
            "full_mapping": full["mapping"],
            "full_mapping_kind": full["mapping_kind"],
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            float(r["local_probe_mse"]),
            float(r["full_probe_mse"]),
            node_size(r["expr_ast"]),
            str(r["expr"]),
        )
    )
    if topk is None:
        return rows
    return rows[: max(1, int(topk))]


def _inverse_mode_mapping(
    mode_name: str,
    *,
    pred_fit: torch.Tensor,
    y_fit: torch.Tensor,
    base_mapping: dict[str, Any] | None,
    safe_eps: float,
) -> tuple[dict[str, Any], str]:
    mode = str(mode_name).strip().lower()
    if mode in ("", "full", "fitted", "fitbest", "legacy"):
        mapping = base_mapping if isinstance(base_mapping, dict) else {"kind": "identity"}
        return mapping, str(mapping.get("kind", "identity"))
    if mode in ("identity", "id"):
        return {"kind": "identity"}, "identity"
    if mode == "affine":
        aff = _fit_affine_mapping_from_pair(pred_fit, y_fit, safe_eps=float(safe_eps))
        if aff is None:
            return {"kind": "identity"}, "identity"
        return aff, str(aff.get("kind", "identity"))
    raise ValueError(f"unsupported inverse comparison mode: {mode_name!r}")


def _inverse_target_summary(
    inv_fit: InverseTarget,
    inv_probe: InverseTarget,
    *,
    requested_mode: str,
    effective_mapping_kind: str,
) -> dict[str, Any]:
    return {
        "requested_mode": str(requested_mode),
        "effective_mapping_kind": str(effective_mapping_kind),
        "confidence": float(inv_fit.confidence),
        "mapping_inverted": bool(inv_fit.mapping_inverted),
        "mapping_kind": str(inv_fit.mapping_kind),
        "valid_fraction_fit": _mask_fraction(inv_fit.valid_mask),
        "valid_fraction_probe": _mask_fraction(inv_probe.valid_mask),
        "steps": [
            {
                "parent_path": list(st.parent_path),
                "parent_path_str": _format_path(st.parent_path),
                "op": st.op,
                "child_slot": int(st.child_slot),
                "valid_fraction": float(st.valid_fraction),
                "confidence": float(st.confidence),
                "note": st.note,
            }
            for st in inv_fit.steps
        ],
    }


def _find_ranked_node(rows: Sequence[dict[str, Any]], node: tuple | None) -> tuple[int | None, dict[str, Any] | None]:
    if node is None:
        return None, None
    for idx, row in enumerate(rows, start=1):
        if row.get("expr_ast") == node:
            return idx, row
    return None, None


@torch.no_grad()
def run_inverse_steering_lab(
    spec: EquationSpec,
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    candidate_expr: str | None = None,
    corrupt_path: str | Sequence[int] | None = None,
    replacement_expr: str | None = None,
    inverse_path: str | Sequence[int] | None = None,
    topk: int = 10,
    verbose: bool = True,
) -> dict[str, Any]:
    """Oracle sandbox for context-sensitive inverse steering.

    This intentionally does *not* run the full factorized symbolic search. Instead it:

    1. Builds an oracle dataset from the ground-truth equation.
    2. Builds a candidate AST (optionally by corrupting the truth AST locally).
    3. Fits the candidate's outer mapping to the oracle data.
    4. Inverts the candidate's surrounding context at one path.
    5. Compares that pseudo-target against the plain global residual.
    6. Ranks local replacement proposals under both signals.
    """

    hp = factorized_search_hp if factorized_search_hp is not None else default_oracle_hyperparams()
    run_seed = int(hp.seed if seed is None else seed)
    target_fn = compile_target_expression(spec)
    ds = build_oracle_dataset(
        spec,
        target_fn,
        n_fit=int(hp.n_fit),
        n_probe=int(hp.n_probe),
        seed=run_seed,
        dtype=dtype,
    )
    var_dims = ds["var_dims"] if enforce_dims else None

    candidate_ast, truth_ast, used_corrupt_path = build_candidate_ast_for_inverse_lab(
        spec,
        candidate_expr=candidate_expr,
        corrupt_path=corrupt_path,
        replacement_expr=replacement_expr,
        var_dims=var_dims,
    )

    path = _parse_path(inverse_path)
    if path == () and inverse_path is None:
        path = used_corrupt_path if used_corrupt_path is not None else _default_inverse_path(candidate_ast)

    def _collect_shared_explorer_local_candidates(
        target_specs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> list[tuple]:
        path_t = tuple(int(v) for v in path)
        current_node = get_at(candidate_ast, path_t)
        target_dim = node_dims(current_node, var_dims) if var_dims is not None else None
        nvars = int(ds["x_fit"].shape[1])
        pool_nodes = build_pool(nvars)
        pool_dims = [node_dims(n, var_dims) for n in pool_nodes] if var_dims is not None else [None] * len(pool_nodes)

        pool_phi_fit_list = []
        for node in pool_nodes:
            try:
                vals = eval_node(node, ds["x_fit"]).squeeze(-1)
            except Exception:
                vals = torch.zeros((int(ds["x_fit"].shape[0]),), dtype=ds["x_fit"].dtype, device=ds["x_fit"].device)
            pool_phi_fit_list.append(vals if torch.isfinite(vals).all() else torch.zeros_like(vals))
        pool_phi_fit = torch.stack(pool_phi_fit_list, dim=1)

        pool_phi_probe_list = []
        for node in pool_nodes:
            try:
                vals = eval_node(node, ds["x_probe"]).squeeze(-1)
            except Exception:
                vals = torch.zeros((int(ds["x_probe"].shape[0]),), dtype=ds["x_probe"].dtype, device=ds["x_probe"].device)
            pool_phi_probe_list.append(vals if torch.isfinite(vals).all() else torch.zeros_like(vals))
        pool_phi_probe = torch.stack(pool_phi_probe_list, dim=1)

        max_depth_eff = max(int(node_depth(candidate_ast)), int(getattr(hp, "max_depth", node_depth(candidate_ast))))
        topk_terms = max(2, int(getattr(hp, "inverse_topk_terms", 6)))
        shortlist_mult = max(1, int(getattr(hp, "inverse_shortlist_mult", 4)))
        local_mode = _normalize_inverse_local_score_mode(inv_local_score_mode, default="affine")
        safe_eps_eff = float(inv_safe_eps or 1.0e-12)

        seen_nodes: set[str] = set()
        shared_nodes: list[tuple] = []

        def _add_node(node: tuple) -> None:
            key = node_str(node)
            if key in seen_nodes:
                return
            seen_nodes.add(key)
            shared_nodes.append(node)

        _add_node(current_node)

        for t_fit_raw, mask_fit_raw, t_probe_raw, mask_probe_raw in target_specs:
            mfit = _bool_col(mask_fit_raw).squeeze(-1)
            mprobe = _bool_col(mask_probe_raw).squeeze(-1)
            if int(mfit.sum().item()) < 4 or int(mprobe.sum().item()) < 4:
                continue
            xf, tf = _slice_by_mask(ds["x_fit"], t_fit_raw, mask_fit_raw)
            xp, tp = _slice_by_mask(ds["x_probe"], t_probe_raw, mask_probe_raw)
            if int(xf.shape[0]) < 4 or int(xp.shape[0]) < 4:
                continue
            idxs = _inverse_pool_shortlist(
                pool_phi_fit,
                t_fit_raw,
                mask_fit_raw,
                pool_dims=pool_dims if var_dims is not None else None,
                target_dim=target_dim,
                shortlist_k=max(topk_terms, topk_terms * shortlist_mult),
            )
            if not idxs:
                continue
            cand_nodes = _inverse_collect_local_repair_candidates(
                parent_node=candidate_ast,
                path=path_t,
                sub=current_node,
                target_dim=target_dim,
                xf=xf,
                tf=tf,
                xp=xp,
                tp=tp,
                wf=None,
                wp=None,
                mfit=mfit,
                mprobe=mprobe,
                pool_nodes=pool_nodes,
                pool_dims=pool_dims,
                pool_phi_fit=pool_phi_fit,
                pool_phi_probe=pool_phi_probe,
                idxs=idxs,
                poly_degree=int(hp.poly_degree),
                local_mode=local_mode,
                topk_terms=topk_terms,
                shortlist_mult=shortlist_mult,
                safe_eps=safe_eps_eff,
                var_dims=var_dims,
                max_depth=max_depth_eff,
                micro_search_enable=bool(getattr(hp, "inverse_micro_search_enable", False)),
                micro_search_max_depth=int(getattr(hp, "inverse_micro_search_max_depth", 3)),
                micro_search_beam_width=int(getattr(hp, "inverse_micro_search_beam_width", 24)),
                micro_search_topk=int(getattr(hp, "inverse_micro_search_topk", 16)),
                micro_search_seed_terms=int(getattr(hp, "inverse_micro_search_seed_terms", 8)),
            )
            for node in cand_nodes:
                _add_node(node)
        return shared_nodes

    base = _score_expr_against_target(
        candidate_ast,
        x_fit=ds["x_fit"],
        y_fit=ds["y_fit"],
        x_probe=ds["x_probe"],
        y_probe=ds["y_probe"],
        poly_degree=int(hp.poly_degree),
    )
    if base is None:
        raise RuntimeError(f"candidate AST could not be scored: {node_str(candidate_ast)}")

    mapping = base["mapping"]
    inv_conf_mode = str(getattr(hp, "inverse_confidence_mode", "conditioning")).strip().lower()
    if inv_conf_mode == "":
        inv_conf_mode = "conditioning"
    inv_conf_target_gain = float(getattr(hp, "inverse_confidence_target_gain", 4.0))
    inv_conf_floor = float(getattr(hp, "inverse_confidence_floor", 0.05))
    inv_local_score_mode = _normalize_inverse_local_score_mode(
        getattr(hp, "inverse_local_score_mode", "affine"),
        default="affine",
    )
    inv_branch_beam_width = max(1, int(getattr(hp, "inverse_branch_beam_width", 1)))
    inv_safe_eps = getattr(hp, "inverse_safe_eps", None)
    if inv_safe_eps is None:
        inv_safe_eps = getattr(hp, "refine_safe_eps", 1.0e-12)
    inv_fit = invert_context_target(
        candidate_ast,
        path,
        ds["x_fit"],
        ds["y_fit"],
        mapping=mapping,
        safe_eps=float(inv_safe_eps or 1.0e-12),
        allow_identity_fallback=True,
        confidence_mode=inv_conf_mode,
        confidence_target_gain=inv_conf_target_gain,
        confidence_floor=inv_conf_floor,
        branch_beam_width=inv_branch_beam_width,
    )
    inv_probe = invert_context_target(
        candidate_ast,
        path,
        ds["x_probe"],
        ds["y_probe"],
        mapping=mapping,
        safe_eps=float(inv_safe_eps or 1.0e-12),
        allow_identity_fallback=True,
        confidence_mode=inv_conf_mode,
        confidence_target_gain=inv_conf_target_gain,
        confidence_floor=inv_conf_floor,
        branch_beam_width=inv_branch_beam_width,
    )

    pred_fit = eval_node(candidate_ast, ds["x_fit"])
    pred_probe = eval_node(candidate_ast, ds["x_probe"])
    y_hat_fit = eval_mapping(pred_fit, mapping)
    y_hat_probe = eval_mapping(pred_probe, mapping)
    resid_fit = ds["y_fit"] - y_hat_fit
    resid_probe = ds["y_probe"] - y_hat_probe
    resid_mask_fit = _finite_mask(resid_fit)
    resid_mask_probe = _finite_mask(resid_probe)

    truth_node = None
    truth_paths = set(collect_paths(truth_ast))
    if path in truth_paths:
        truth_node = get_at(truth_ast, path)

    shared_proposal_nodes = _collect_shared_explorer_local_candidates(
        (
            (inv_fit.target, inv_fit.valid_mask, inv_probe.target, inv_probe.valid_mask),
            (resid_fit, resid_mask_fit, resid_probe, resid_mask_probe),
        )
    )

    oracle_compare: dict[str, Any] = {}
    if truth_node is not None:
        truth_inv = _score_node_on_local_target(
            truth_node,
            x_fit=_slice_by_mask(ds["x_fit"], inv_fit.target, inv_fit.valid_mask)[0],
            t_fit=_slice_by_mask(ds["x_fit"], inv_fit.target, inv_fit.valid_mask)[1],
            x_probe=_slice_by_mask(ds["x_probe"], inv_probe.target, inv_probe.valid_mask)[0],
            t_probe=_slice_by_mask(ds["x_probe"], inv_probe.target, inv_probe.valid_mask)[1],
            poly_degree=int(hp.poly_degree),
            local_score_mode=inv_local_score_mode,
        )
        truth_res = _score_node_on_local_target(
            truth_node,
            x_fit=ds["x_fit"],
            t_fit=resid_fit,
            x_probe=ds["x_probe"],
            t_probe=resid_probe,
            poly_degree=int(hp.poly_degree),
            local_score_mode=inv_local_score_mode,
        )
        truth_fit_vals = eval_node(truth_node, ds["x_fit"])
        truth_probe_vals = eval_node(truth_node, ds["x_probe"])
        fit_true_fit = get_at(truth_ast, path)
        _ = fit_true_fit  # quiet linters when running outside tests
        oracle_compare = {
            "truth_subexpr": node_str(truth_node),
            "truth_subexpr_ast": truth_node,
            "truth_vs_inverse_direct_rmse_fit": float(
                torch.sqrt(
                    ((_slice_by_mask(ds["x_fit"], inv_fit.target, inv_fit.valid_mask)[1] - truth_fit_vals[_bool_col(inv_fit.valid_mask).squeeze(-1)]) ** 2).mean()
                ).item()
            ) if int(_bool_col(inv_fit.valid_mask).sum().item()) >= 1 else None,
            "truth_vs_inverse_corr_probe": _corrcoef(
                truth_probe_vals[_bool_col(inv_probe.valid_mask).squeeze(-1)],
                _slice_by_mask(ds["x_probe"], inv_probe.target, inv_probe.valid_mask)[1],
            ) if int(_bool_col(inv_probe.valid_mask).sum().item()) >= 2 else None,
            "truth_fit_to_inverse": None if truth_inv is None else {
                "probe_mse": float(truth_inv["local_probe_mse"]),
                "fit_mse": float(truth_inv["local_fit_mse"]),
                "corr_probe": float(truth_inv["local_corr_probe"]),
                "mapping_kind": truth_inv["local_mapping_kind"],
            },
            "truth_fit_to_residual": None if truth_res is None else {
                "probe_mse": float(truth_res["local_probe_mse"]),
                "fit_mse": float(truth_res["local_fit_mse"]),
                "corr_probe": float(truth_res["local_corr_probe"]),
                "mapping_kind": truth_res["local_mapping_kind"],
            },
            "truth_repair_full_expr": _score_expr_against_target(
                replace_at(candidate_ast, path, truth_node),
                x_fit=ds["x_fit"],
                y_fit=ds["y_fit"],
                x_probe=ds["x_probe"],
                y_probe=ds["y_probe"],
                poly_degree=int(hp.poly_degree),
            ),
        }

    top_inverse = _rank_local_replacements(
        candidate_ast,
        path,
        x_fit=ds["x_fit"],
        y_fit=ds["y_fit"],
        x_probe=ds["x_probe"],
        y_probe=ds["y_probe"],
        t_fit=inv_fit.target,
        mask_fit=inv_fit.valid_mask,
        t_probe=inv_probe.target,
        mask_probe=inv_probe.valid_mask,
        poly_degree=int(hp.poly_degree),
        topk=int(topk),
        var_dims=var_dims,
        truth_node=truth_node,
        local_score_mode=inv_local_score_mode,
        include_truth_candidate=False,
        proposal_nodes=shared_proposal_nodes,
    )
    top_residual = _rank_local_replacements(
        candidate_ast,
        path,
        x_fit=ds["x_fit"],
        y_fit=ds["y_fit"],
        x_probe=ds["x_probe"],
        y_probe=ds["y_probe"],
        t_fit=resid_fit,
        mask_fit=resid_mask_fit,
        t_probe=resid_probe,
        mask_probe=resid_mask_probe,
        poly_degree=int(hp.poly_degree),
        topk=int(topk),
        var_dims=var_dims,
        truth_node=truth_node,
        local_score_mode=inv_local_score_mode,
        include_truth_candidate=False,
        proposal_nodes=shared_proposal_nodes,
    )

    report = {
        "mode": "inverse_steering_lab",
        "spec_id": spec.id,
        "target_expr": spec.target_expr,
        "truth_expr": node_str(truth_ast),
        "truth_expr_ast": truth_ast,
        "candidate_expr": node_str(candidate_ast),
        "candidate_expr_ast": candidate_ast,
        "path": list(path),
        "path_str": _format_path(path),
        "current_subexpr": node_str(get_at(candidate_ast, path)),
        "current_subexpr_ast": get_at(candidate_ast, path),
        "truth_subexpr": None if truth_node is None else node_str(truth_node),
        "truth_subexpr_ast": truth_node,
        "candidate_score": {
            "fit_mse": float(base["fit_mse"]),
            "probe_mse": float(base["probe_mse"]),
            "mapping": base["mapping"],
            "mapping_kind": base["mapping_kind"],
        },
        "inverse_target": {
            "confidence": float(inv_fit.confidence),
            "mapping_inverted": bool(inv_fit.mapping_inverted),
            "mapping_kind": str(inv_fit.mapping_kind),
            "valid_fraction_fit": _mask_fraction(inv_fit.valid_mask),
            "valid_fraction_probe": _mask_fraction(inv_probe.valid_mask),
            "steps": [
                {
                    "parent_path": list(st.parent_path),
                    "parent_path_str": _format_path(st.parent_path),
                    "op": st.op,
                    "child_slot": int(st.child_slot),
                    "valid_fraction": float(st.valid_fraction),
                    "confidence": float(st.confidence),
                    "note": st.note,
                }
                for st in inv_fit.steps
            ],
        },
        "oracle_compare": oracle_compare,
        "shared_proposal_count": int(len(shared_proposal_nodes)),
        "top_inverse_replacements": top_inverse,
        "top_residual_replacements": top_residual,
        "hp": {
            "n_fit": int(hp.n_fit),
            "n_probe": int(hp.n_probe),
            "poly_degree": int(hp.poly_degree),
            "seed": int(run_seed),
            "enforce_dims": bool(enforce_dims),
            "inverse_confidence_mode": str(inv_conf_mode),
            "inverse_confidence_target_gain": float(inv_conf_target_gain),
            "inverse_confidence_floor": float(inv_conf_floor),
            "inverse_local_score_mode": str(inv_local_score_mode),
            "inverse_target_mode": str(getattr(hp, "inverse_target_mode", "robust")),
            "inverse_full_mapping_penalty": float(getattr(hp, "inverse_full_mapping_penalty", 0.75)),
            "inverse_exact_simple_target_bonus": float(getattr(hp, "inverse_exact_simple_target_bonus", 0.10)),
            "inverse_additive_descend_penalty": float(getattr(hp, "inverse_additive_descend_penalty", 0.15)),
            "inverse_nonadditive_leaf_penalty": float(getattr(hp, "inverse_nonadditive_leaf_penalty", 0.20)),
            "inverse_exact_path_eta": float(getattr(hp, "inverse_exact_path_eta", 0.98)),
            "inverse_exact_transport_min_lin_rel": float(getattr(hp, "inverse_exact_transport_min_lin_rel", 0.0)),
            "inverse_branch_beam_width": int(inv_branch_beam_width),
        },
    }

    if verbose:
        print(
            f"[inverse] {spec.id}: candidate_probe_mse={float(base['probe_mse']):.6g} "
            f"mapping={str(base['mapping_kind'])} path={_format_path(path)}"
        )
        print(
            f"[inverse] current={node_str(get_at(candidate_ast, path))} "
            f"truth={None if truth_node is None else node_str(truth_node)}"
        )
        print(
            f"[inverse] pseudo-target confidence={float(inv_fit.confidence):.3g} "
            f"valid_fit={_mask_fraction(inv_fit.valid_mask):.3f} "
            f"valid_probe={_mask_fraction(inv_probe.valid_mask):.3f}"
        )
        print(f"[inverse] shared explorer-style proposal pool: {int(len(shared_proposal_nodes))} candidates")
        if oracle_compare:
            tinv = oracle_compare.get("truth_fit_to_inverse")
            tres = oracle_compare.get("truth_fit_to_residual")
            if tinv is not None and tres is not None:
                print(
                    f"[inverse] truth-subtree local probe MSE: inverse={float(tinv['probe_mse']):.6g} "
                    f"residual={float(tres['probe_mse']):.6g}"
                )
        if top_inverse:
            row = top_inverse[0]
            print(
                f"[inverse] best inverse proposal: {row['expr']} "
                f"local_probe_mse={float(row['local_probe_mse']):.6g} "
                f"full_probe_mse={float(row['full_probe_mse']):.6g} source={row['source']}"
            )
        if top_residual:
            row = top_residual[0]
            print(
                f"[inverse] best residual proposal: {row['expr']} "
                f"local_probe_mse={float(row['local_probe_mse']):.6g} "
                f"full_probe_mse={float(row['full_probe_mse']):.6g} source={row['source']}"
            )

    return _to_jsonable(report)


@torch.no_grad()
def run_inverse_path_sweep_lab(
    spec: EquationSpec,
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    candidate_expr: str | None = None,
    corrupt_path: str | Sequence[int] | None = None,
    replacement_expr: str | None = None,
    inverse_path: str | Sequence[int] | None = None,
    topk: int = 8,
    compare_modes: Sequence[str] | None = None,
    sweep_all_paths: bool = False,
    sweep_max_paths: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Oracle sandbox that sweeps inverse diagnostics across multiple tree cuts.

    The report is meant to answer three questions for a corrupted candidate:

    1. Which path would the production inverse gate select?
    2. At the reference truth-repair path, where does the true subtree rank under
       different inverse target modes (identity/full/affine)?
    3. Is the selected path the same as, an ancestor of, or a descendant of the
       structural repair site?
    """

    hp = factorized_search_hp if factorized_search_hp is not None else default_oracle_hyperparams()
    run_seed = int(hp.seed if seed is None else seed)
    target_fn = compile_target_expression(spec)
    ds = build_oracle_dataset(
        spec,
        target_fn,
        n_fit=int(hp.n_fit),
        n_probe=int(hp.n_probe),
        seed=run_seed,
        dtype=dtype,
    )
    var_dims = ds["var_dims"] if enforce_dims else None

    candidate_ast, truth_ast, used_corrupt_path = build_candidate_ast_for_inverse_lab(
        spec,
        candidate_expr=candidate_expr,
        corrupt_path=corrupt_path,
        replacement_expr=replacement_expr,
        var_dims=var_dims,
    )

    base = _score_expr_against_target(
        candidate_ast,
        x_fit=ds["x_fit"],
        y_fit=ds["y_fit"],
        x_probe=ds["x_probe"],
        y_probe=ds["y_probe"],
        poly_degree=int(hp.poly_degree),
    )
    if base is None:
        raise RuntimeError(f"candidate AST could not be scored: {node_str(candidate_ast)}")

    mapping = base["mapping"]
    pred_fit = eval_node(candidate_ast, ds["x_fit"])
    pred_probe = eval_node(candidate_ast, ds["x_probe"])
    y_hat_fit = eval_mapping(pred_fit, mapping)
    y_hat_probe = eval_mapping(pred_probe, mapping)
    resid_fit = ds["y_fit"] - y_hat_fit
    resid_probe = ds["y_probe"] - y_hat_probe
    resid_mask_fit = _finite_mask(resid_fit)
    resid_mask_probe = _finite_mask(resid_probe)

    inv_conf_mode = str(getattr(hp, "inverse_confidence_mode", "conditioning")).strip().lower()
    if inv_conf_mode == "":
        inv_conf_mode = "conditioning"
    inv_conf_target_gain = float(getattr(hp, "inverse_confidence_target_gain", 4.0))
    inv_conf_floor = float(getattr(hp, "inverse_confidence_floor", 0.05))
    inv_local_score_mode = _normalize_inverse_local_score_mode(
        getattr(hp, "inverse_local_score_mode", "affine"),
        default="affine",
    )
    inv_branch_beam_width = max(1, int(getattr(hp, "inverse_branch_beam_width", 1)))
    inv_safe_eps = getattr(hp, "inverse_safe_eps", None)
    if inv_safe_eps is None:
        inv_safe_eps = getattr(hp, "refine_safe_eps", 1.0e-12)
    inv_safe_eps = float(inv_safe_eps or 1.0e-12)

    mode_list = [str(m).strip().lower() for m in (compare_modes or ("identity", "full", "affine")) if str(m).strip()]
    if not mode_list:
        mode_list = ["identity", "full", "affine"]
    mode_list = list(dict.fromkeys(mode_list))

    pool_cache = _oracle_build_pool_cache(ds["x_fit"], ds["x_probe"], var_dims=var_dims)

    gate_diag = estimate_inverse_steering_potential(
        candidate_ast,
        mapping,
        ds["x_fit"],
        ds["y_fit"],
        ds["x_probe"],
        ds["y_probe"],
        pool_cache["pool_phi_fit"],
        pool_cache["pool_phi_probe"],
        pool_cache["pool_dims"],
        pool_nodes=pool_cache["pool_nodes"],
        var_dims=var_dims,
        max_paths=int(getattr(hp, "inverse_gate_max_paths", 6)),
        topk_terms=max(1, min(int(getattr(hp, "inverse_topk_terms", 6)), 4)),
        shortlist_mult=max(1, min(int(getattr(hp, "inverse_shortlist_mult", 4)), 2)),
        min_valid_frac=float(getattr(hp, "inverse_min_valid_frac", 0.25)),
        min_confidence=float(getattr(hp, "inverse_min_confidence", 0.10)),
        min_structural_score=float(getattr(hp, "inverse_gate_min_structural_score", 0.75)),
        min_weighted_rel_gain=float(getattr(hp, "inverse_gate_min_weighted_rel_gain", 0.05)),
        structural_bias=float(getattr(hp, "inverse_gate_structural_bias", 0.20)),
        safe_eps=float(inv_safe_eps),
        confidence_mode=str(inv_conf_mode),
        confidence_target_gain=float(inv_conf_target_gain),
        confidence_floor=float(inv_conf_floor),
        branch_beam_width=int(inv_branch_beam_width),
        local_score_mode=str(inv_local_score_mode),
        target_mode=str(getattr(hp, "inverse_target_mode", "robust")),
        full_mapping_penalty=float(getattr(hp, "inverse_full_mapping_penalty", 0.75)),
        exact_simple_target_bonus=float(getattr(hp, "inverse_exact_simple_target_bonus", 0.10)),
        additive_descend_penalty=float(getattr(hp, "inverse_additive_descend_penalty", 0.15)),
        nonadditive_leaf_penalty=float(getattr(hp, "inverse_nonadditive_leaf_penalty", 0.20)),
        periodic_min_valid_scale=float(getattr(hp, "inverse_periodic_min_valid_scale", 1.25)),
        periodic_min_confidence_scale=float(getattr(hp, "inverse_periodic_min_confidence_scale", 1.35)),
        periodic_path_penalty=float(getattr(hp, "inverse_periodic_path_penalty", 0.65)),
        nonperiodic_muldiv_bonus=float(getattr(hp, "inverse_nonperiodic_muldiv_bonus", 0.10)),
        nonperiodic_explogsqrt_bonus=float(getattr(hp, "inverse_nonperiodic_explogsqrt_bonus", 0.05)),
        branch_ambiguity_penalty=float(getattr(hp, "inverse_branch_ambiguity_penalty", 0.50)),
    )

    if isinstance(gate_diag, InverseSteeringPotential):
        gate_path_rows = gate_diag.path_row_map()
        selected_path = None if gate_diag.best_path is None else tuple(int(v) for v in gate_diag.best_path)
        gate_candidate_paths = [tuple(int(v) for v in p) for p in gate_diag.candidate_paths]
        gate_diagnostic = gate_diag.to_dict()
        gate_allowed = bool(gate_diag.allowed)
        gate_reason = str(gate_diag.reason)
    else:
        gate_path_rows = {
            tuple(int(v) for v in row.get("path", ())): PathStateFeatures.from_row(row)
            for row in list(gate_diag.get("path_rows", []) or [])
            if row.get("path", None) is not None
        }
        selected_path = gate_diag.get("best_path", None)
        selected_path = None if selected_path is None else tuple(int(v) for v in selected_path)
        gate_candidate_paths = [tuple(int(v) for v in p) for p in list(gate_diag.get("candidate_paths", []) or [])]
        gate_diagnostic = dict(gate_diag)
        gate_allowed = bool(gate_diag.get("allowed", False))
        gate_reason = str(gate_diag.get("reason", ""))

    truth_paths = set(tuple(int(v) for v in p) for p in collect_paths(truth_ast))
    candidate_paths = [tuple(int(v) for v in p) for p in collect_paths(candidate_ast)]
    candidate_path_set = set(candidate_paths)

    user_path = None if inverse_path is None else _parse_path(inverse_path)
    mismatch_paths = _minimal_mismatch_paths(candidate_ast, truth_ast)
    reference_paths = _dedupe_paths([
        used_corrupt_path,
        user_path,
        *mismatch_paths,
    ])
    reference_paths = [p for p in reference_paths if p in candidate_path_set or p in truth_paths]
    reference_eval_paths = [p for p in reference_paths if p in candidate_path_set]

    essential_paths = _dedupe_paths([
        *(reference_eval_paths or []),
        user_path if user_path in candidate_path_set else None,
        selected_path,
    ])

    if bool(sweep_all_paths):
        extras = [p for p in candidate_paths if p not in set(essential_paths)]
        extras.sort(
            key=lambda p: (
                -float(gate_path_rows.get(p, PathStateFeatures()).weighted_rel_gain),
                -len(p),
                tuple(p),
            )
        )
    else:
        extras = [p for p in gate_candidate_paths if p not in set(essential_paths)]

    if sweep_max_paths is None:
        sweep_paths = _dedupe_paths([*essential_paths, *extras])
    else:
        limit = max(1, int(sweep_max_paths))
        base_paths = _dedupe_paths(essential_paths)
        extra_cap = max(0, limit - len(base_paths))
        sweep_paths = _dedupe_paths([*base_paths, *extras[:extra_cap]])
    if not sweep_paths:
        sweep_paths = [()]

    path_reports: list[dict[str, Any]] = []
    path_report_map: dict[tuple[int, ...], dict[str, Any]] = {}
    for path in sweep_paths:
        current_node = get_at(candidate_ast, path)
        truth_node = get_at(truth_ast, path) if path in truth_paths else None
        per_mode_targets: dict[str, tuple[InverseTarget, InverseTarget, str]] = {}
        shared_target_specs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = [
            (resid_fit, resid_mask_fit, resid_probe, resid_mask_probe)
        ]

        for mode_name in mode_list:
            mode_mapping, eff_mapping_kind = _inverse_mode_mapping(
                mode_name,
                pred_fit=pred_fit,
                y_fit=ds["y_fit"],
                base_mapping=mapping,
                safe_eps=float(inv_safe_eps),
            )
            inv_fit = invert_context_target(
                candidate_ast,
                path,
                ds["x_fit"],
                ds["y_fit"],
                mapping=mode_mapping,
                safe_eps=float(inv_safe_eps),
                allow_identity_fallback=True,
                confidence_mode=inv_conf_mode,
                confidence_target_gain=inv_conf_target_gain,
                confidence_floor=inv_conf_floor,
                branch_beam_width=inv_branch_beam_width,
            )
            inv_probe = invert_context_target(
                candidate_ast,
                path,
                ds["x_probe"],
                ds["y_probe"],
                mapping=mode_mapping,
                safe_eps=float(inv_safe_eps),
                allow_identity_fallback=True,
                confidence_mode=inv_conf_mode,
                confidence_target_gain=inv_conf_target_gain,
                confidence_floor=inv_conf_floor,
                branch_beam_width=inv_branch_beam_width,
            )
            per_mode_targets[mode_name] = (inv_fit, inv_probe, eff_mapping_kind)
            shared_target_specs.append((inv_fit.target, inv_fit.valid_mask, inv_probe.target, inv_probe.valid_mask))

        shared_proposals = _collect_shared_explorer_local_candidates_for_path(
            candidate_ast,
            path,
            target_specs=tuple(shared_target_specs),
            ds=ds,
            hp=hp,
            pool_cache=pool_cache,
            var_dims=var_dims,
            local_score_mode=inv_local_score_mode,
        )

        mode_reports: dict[str, Any] = {}
        for mode_name in mode_list:
            inv_fit, inv_probe, eff_mapping_kind = per_mode_targets[mode_name]
            rows = _rank_local_replacements(
                candidate_ast,
                path,
                x_fit=ds["x_fit"],
                y_fit=ds["y_fit"],
                x_probe=ds["x_probe"],
                y_probe=ds["y_probe"],
                t_fit=inv_fit.target,
                mask_fit=inv_fit.valid_mask,
                t_probe=inv_probe.target,
                mask_probe=inv_probe.valid_mask,
                poly_degree=int(hp.poly_degree),
                topk=None,
                var_dims=var_dims,
                truth_node=truth_node,
                local_score_mode=inv_local_score_mode,
                include_truth_candidate=bool(truth_node is not None),
                proposal_nodes=shared_proposals,
                pinned_nodes=[truth_node] if truth_node is not None else None,
                preselect_limit=max(8, int(len(shared_proposals) + 4)),
            )
            truth_rank, truth_row = _find_ranked_node(rows, truth_node)
            truth_fit_local = None
            truth_direct_rmse_fit = None
            truth_corr_probe = None
            if truth_node is not None:
                try:
                    truth_fit_vals = eval_node(truth_node, ds["x_fit"])
                    truth_probe_vals = eval_node(truth_node, ds["x_probe"])
                except Exception:
                    truth_fit_vals = None
                    truth_probe_vals = None
                if truth_fit_vals is not None and truth_probe_vals is not None:
                    fit_mask = _bool_col(inv_fit.valid_mask).squeeze(-1)
                    probe_mask = _bool_col(inv_probe.valid_mask).squeeze(-1)
                    if int(fit_mask.sum().item()) >= 1:
                        truth_direct_rmse_fit = float(
                            torch.sqrt(((inv_fit.target[fit_mask] - truth_fit_vals[fit_mask]) ** 2).mean()).item()
                        )
                    if int(probe_mask.sum().item()) >= 2:
                        truth_corr_probe = _corrcoef(truth_probe_vals[probe_mask], inv_probe.target[probe_mask])
                    xf_t, tf_t = _slice_by_mask(ds["x_fit"], inv_fit.target, inv_fit.valid_mask)
                    xp_t, tp_t = _slice_by_mask(ds["x_probe"], inv_probe.target, inv_probe.valid_mask)
                    truth_fit_local = _score_node_on_local_target(
                        truth_node,
                        x_fit=xf_t,
                        t_fit=tf_t,
                        x_probe=xp_t,
                        t_probe=tp_t,
                        poly_degree=int(hp.poly_degree),
                        local_score_mode=inv_local_score_mode,
                    )

            mode_reports[mode_name] = {
                "inverse_target": _inverse_target_summary(
                    inv_fit,
                    inv_probe,
                    requested_mode=mode_name,
                    effective_mapping_kind=eff_mapping_kind,
                ),
                "shared_proposal_count": int(len(shared_proposals)),
                "scored_candidate_count": int(len(rows)),
                "truth_rank": None if truth_rank is None else int(truth_rank),
                "truth_row": truth_row,
                "truth_fit_to_inverse": None if truth_fit_local is None else {
                    "probe_mse": float(truth_fit_local["local_probe_mse"]),
                    "fit_mse": float(truth_fit_local["local_fit_mse"]),
                    "corr_probe": float(truth_fit_local["local_corr_probe"]),
                    "mapping_kind": truth_fit_local["local_mapping_kind"],
                },
                "truth_vs_inverse_direct_rmse_fit": truth_direct_rmse_fit,
                "truth_vs_inverse_corr_probe": truth_corr_probe,
                "top_replacements": rows[: max(1, int(topk))],
            }

        path_row = {
            "path": list(path),
            "path_str": _format_path(path),
            "current_subexpr": node_str(current_node),
            "current_subexpr_ast": current_node,
            "truth_subexpr": None if truth_node is None else node_str(truth_node),
            "truth_subexpr_ast": truth_node,
            "is_reference_path": bool(path in reference_paths),
            "is_reference_eval_path": bool(path in reference_eval_paths),
            "is_selected_path": bool(selected_path is not None and tuple(path) == tuple(selected_path)),
            "relation_to_reference": _best_relation_to_reference(path, reference_paths),
            "relation_to_selected": _path_relation(path, selected_path),
            "reference_relations": [
                {
                    "reference_path": list(ref),
                    "reference_path_str": _format_path(ref),
                    "relation": _path_relation(path, ref),
                }
                for ref in reference_paths
            ],
            "gate_row": None if gate_path_rows.get(tuple(path), None) is None else gate_path_rows[tuple(path)].to_dict(),
            "modes": mode_reports,
        }
        path_reports.append(path_row)
        path_report_map[tuple(path)] = path_row

    reference_summary = []
    for ref in reference_paths:
        row = path_report_map.get(tuple(ref), None)
        modes_summary = {}
        if row is not None:
            for mode_name in mode_list:
                mode_row = row["modes"].get(mode_name, {})
                top_rows = list(mode_row.get("top_replacements", []) or [])
                modes_summary[mode_name] = {
                    "truth_rank": mode_row.get("truth_rank", None),
                    "top_expr": None if not top_rows else top_rows[0].get("expr", None),
                    "top_full_probe_mse": None if not top_rows else top_rows[0].get("full_probe_mse", None),
                    "top_local_probe_mse": None if not top_rows else top_rows[0].get("local_probe_mse", None),
                }
        reference_summary.append(
            {
                "path": list(ref),
                "path_str": _format_path(ref),
                "path_in_candidate": bool(tuple(ref) in candidate_path_set),
                "selected_path_relation": _path_relation(selected_path, ref),
                "gate_row": None if gate_path_rows.get(tuple(ref), None) is None else gate_path_rows[tuple(ref)].to_dict(),
                "modes": modes_summary,
            }
        )

    selected_path_report = None if selected_path is None else path_report_map.get(tuple(selected_path), None)

    report = {
        "mode": "inverse_path_sweep_lab",
        "spec_id": spec.id,
        "target_expr": spec.target_expr,
        "truth_expr": node_str(truth_ast),
        "truth_expr_ast": truth_ast,
        "candidate_expr": node_str(candidate_ast),
        "candidate_expr_ast": candidate_ast,
        "candidate_score": {
            "fit_mse": float(base["fit_mse"]),
            "probe_mse": float(base["probe_mse"]),
            "mapping": base["mapping"],
            "mapping_kind": base["mapping_kind"],
        },
        "reference_paths": [
            {"path": list(p), "path_str": _format_path(p), "in_candidate": bool(p in candidate_path_set)}
            for p in reference_paths
        ],
        "selected_path": None if selected_path is None else {
            "path": list(selected_path),
            "path_str": _format_path(selected_path),
            "relation_to_reference": _best_relation_to_reference(selected_path, reference_paths),
        },
        "gate_diagnostic": gate_diagnostic,
        "path_reports": path_reports,
        "reference_summary": reference_summary,
        "selected_path_report": selected_path_report,
        "hp": {
            "n_fit": int(hp.n_fit),
            "n_probe": int(hp.n_probe),
            "poly_degree": int(hp.poly_degree),
            "seed": int(run_seed),
            "enforce_dims": bool(enforce_dims),
            "compare_modes": list(mode_list),
            "sweep_all_paths": bool(sweep_all_paths),
            "sweep_max_paths": None if sweep_max_paths is None else int(sweep_max_paths),
            "inverse_confidence_mode": str(inv_conf_mode),
            "inverse_confidence_target_gain": float(inv_conf_target_gain),
            "inverse_confidence_floor": float(inv_conf_floor),
            "inverse_local_score_mode": str(inv_local_score_mode),
            "inverse_target_mode": str(getattr(hp, "inverse_target_mode", "robust")),
            "inverse_branch_beam_width": int(inv_branch_beam_width),
        },
    }

    if verbose:
        print(
            f"[path-sweep] {spec.id}: candidate_probe_mse={float(base['probe_mse']):.6g} "
            f"mapping={str(base['mapping_kind'])} gate_allowed={gate_allowed}"
        )
        if selected_path is not None:
            print(
                f"[path-sweep] selected_path={_format_path(selected_path)} "
                f"relation_to_reference={_best_relation_to_reference(selected_path, reference_paths)} "
                f"reason={gate_reason}"
            )
        for ref_row in reference_summary:
            mode_bits = []
            for mode_name in mode_list:
                mm = ref_row.get("modes", {}).get(mode_name, {})
                if mm:
                    mode_bits.append(
                        f"{mode_name}:rank={mm.get('truth_rank', None)} top={mm.get('top_expr', None)}"
                    )
            print(
                f"[path-sweep] reference_path={ref_row['path_str']} "
                f"selected_relation={ref_row['selected_path_relation']} "
                + " | ".join(mode_bits)
            )

    return _to_jsonable(report)


def _oracle_truth_rank_from_grouped_rows(
    grouped_rows: Sequence[Mapping[str, Any]],
    *,
    path: Sequence[int],
    truth_node: tuple | None,
) -> tuple[int | None, bool]:
    if truth_node is None:
        return None, False
    dedup_rows = [row for row in grouped_rows if bool(row.get("dedup_kept", False))]
    dedup_rows.sort(key=lambda row: int(row.get("post_dedup_rank", 1 << 30) or (1 << 30)))
    path_t = tuple(int(v) for v in path)
    for idx, row in enumerate(dedup_rows, start=1):
        expr_ast = row.get("expr", None)
        if not isinstance(expr_ast, tuple):
            continue
        try:
            cand_sub = get_at(expr_ast, path_t)
        except Exception:
            continue
        if cand_sub == truth_node:
            return int(idx), True
    return None, False


def _oracle_exact_score_preview_rows(
    selected_rows: Sequence[dict[str, Any]],
    *,
    ds: Mapping[str, Any],
    hp: FactorizedSearchConfig,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in selected_rows:
        expr_ast = row.get("expr", None)
        if not isinstance(expr_ast, tuple):
            continue
        scored = _score_expr_against_target(
            expr_ast,
            x_fit=ds["x_fit"],
            y_fit=ds["y_fit"],
            x_probe=ds["x_probe"],
            y_probe=ds["y_probe"],
            poly_degree=int(hp.poly_degree),
        )
        if scored is None:
            continue
        row["exact_child_score_observed"] = True
        row["full_fit_mse"] = float(scored["fit_mse"])
        row["full_probe_mse"] = float(scored["probe_mse"])
        row["full_mapping_kind"] = str(scored["mapping_kind"])
        out.append({
            "child_key": str(row.get("child_key", "") or ""),
            "expr": expr_ast,
            "expr_str": str(node_str(expr_ast)),
            "beam_rank": int(row.get("beam_rank", 0) or 0),
            "local_rank": int(row.get("local_rank", 0) or 0),
            "proposal_family": str(row.get("proposal_family", row.get("tuple_provenance", "")) or ""),
            "tuple_provenance": str(row.get("tuple_provenance", "") or ""),
            "full_fit_mse": float(scored["fit_mse"]),
            "full_probe_mse": float(scored["probe_mse"]),
            "full_mapping_kind": str(scored["mapping_kind"]),
        })
    out.sort(
        key=lambda row: (
            float(row.get("full_probe_mse", float("inf"))),
            float(row.get("full_fit_mse", float("inf"))),
            int(row.get("beam_rank", 0) or 0),
            int(row.get("local_rank", 0) or 0),
            str(row.get("expr_str", "")),
        )
    )
    return out


def _build_oracle_inverse_legacy_preview_rows_for_beam(
    *,
    parent_node,
    beam_state: Mapping[str, Any],
    beam_rank: int,
    slate_id: str,
    max_depth: int,
    poly_degree: int,
    topk_terms: int,
    shortlist_mult: int,
    safe_eps: float,
    exact_transport_min_lin_rel: float,
    var_dims,
    pool_nodes,
    pool_dims,
    pool_phi_fit,
    pool_phi_probe,
    micro_search_enable: bool,
    micro_search_max_depth: int,
    micro_search_beam_width: int,
    micro_search_topk: int,
    micro_search_seed_terms: int,
    local_score_mode: str,
    transport_ctx: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple]]:
    path = tuple(int(v) for v in (beam_state.get("path", ()) or ()))
    sub = beam_state.get("sub", None)
    if sub is None:
        return [], []
    dm = var_dims is not None
    local_limit = max(2, min(4, int(topk_terms)))
    cand_subtrees = _inverse_collect_local_repair_candidates(
        parent_node=parent_node,
        path=path,
        sub=sub,
        target_dim=beam_state.get("target_dim", None),
        xf=beam_state["xf"],
        tf=beam_state["tf"],
        xp=beam_state["xp"],
        tp=beam_state["tp"],
        wf=beam_state["wf"],
        wp=beam_state["wp"],
        mfit=beam_state["mfit"],
        mprobe=beam_state["mprobe"],
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        pool_phi_fit=pool_phi_fit,
        pool_phi_probe=pool_phi_probe,
        idxs=beam_state["pool_idx"],
        poly_degree=int(poly_degree),
        local_mode=str(local_score_mode),
        topk_terms=max(2, int(topk_terms)),
        shortlist_mult=max(1, int(shortlist_mult)),
        safe_eps=float(safe_eps),
        var_dims=var_dims if dm else None,
        max_depth=int(max_depth),
        micro_search_enable=bool(micro_search_enable),
        micro_search_max_depth=int(micro_search_max_depth),
        micro_search_beam_width=int(micro_search_beam_width),
        micro_search_topk=int(micro_search_topk),
        micro_search_seed_terms=int(micro_search_seed_terms),
    )
    if not cand_subtrees:
        return [], []
    local_rows = _inverse_rank_local_repair_candidates(
        cand_subtrees,
        xf=beam_state["xf"],
        tf=beam_state["tf"],
        xp=beam_state["xp"],
        tp=beam_state["tp"],
        wf=beam_state["wf"],
        wp=beam_state["wp"],
        poly_degree=int(poly_degree),
        local_mode=str(local_score_mode),
    )
    if not local_rows:
        return [], list(cand_subtrees)
    local_rows.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
    local_rows = local_rows[: int(local_limit)]
    local_rows = _transport_aligned_local_rows(
        local_rows,
        best_path=path,
        best_state=beam_state,
        transport_ctx=transport_ctx,
        safe_eps=float(safe_eps),
        exact_transport_min_lin_rel=float(exact_transport_min_lin_rel),
    )
    parent_sub_size = int(node_size(sub))
    parent_sub_depth = int(node_depth(sub))
    parent_size = int(node_size(parent_node))
    parent_depth = int(node_depth(parent_node))
    rows: list[dict[str, Any]] = []
    for local_rank, (local_probe_mse, local_fit_mse, cand_sub) in enumerate(local_rows):
        child_expr = _build_inverse_action_child_expr(
            cand_sub,
            parent_node=parent_node,
            best_path=path,
            max_depth=int(max_depth),
            var_dims=var_dims if dm else None,
        )
        if child_expr is None:
            continue
        child_key = str(node_str(child_expr))
        local_mapping_preview = _inverse_local_mapping_preview(
            cand_sub,
            xf=beam_state["xf"],
            tf=beam_state["tf"],
            xp=beam_state["xp"],
            tp=beam_state["tp"],
            poly_degree=int(poly_degree),
            local_mode=str(local_score_mode),
        )
        cand_sub_size = int(node_size(cand_sub))
        cand_sub_depth = int(node_depth(cand_sub))
        child_size = int(node_size(child_expr))
        child_depth = int(node_depth(child_expr))
        rows.append({
            "slate_id": str(slate_id),
            "expr": child_expr,
            "child_expr": str(child_key),
            "child_key": str(child_key),
            "path": path,
            "target_mode": str(beam_state.get("target_mode", "") or ""),
            "target_mapping_kind": str(beam_state.get("target_mapping_kind", "") or ""),
            "beam_rank": int(beam_rank),
            "local_rank": int(local_rank),
            "path_gain": float(beam_state.get("path_gain", 0.0) or 0.0),
            "route": "repair",
            "action": "inv_steer",
            "tuple_provenance": "beam_local_repair",
            "proposal_family": "beam_local_repair",
            "generation_source": "legacy_inverse_local",
            "beam_state": beam_state,
            "local_candidate_count": int(len(local_rows)),
            "local_probe_mse": float(local_probe_mse),
            "local_fit_mse": float(local_fit_mse),
            "local_fit_probe_gap": float(max(0.0, float(local_probe_mse) - float(local_fit_mse))),
            "local_mapping_kind": str(local_mapping_preview.get("local_mapping_kind", "") or ""),
            "local_mapping_nparams": int(local_mapping_preview.get("local_mapping_nparams", 0) or 0),
            "candidate_subtree_size": int(cand_sub_size),
            "candidate_subtree_depth": int(cand_sub_depth),
            "candidate_subtree_size_delta": int(cand_sub_size - parent_sub_size),
            "candidate_subtree_depth_delta": int(cand_sub_depth - parent_sub_depth),
            "candidate_child_size": int(child_size),
            "candidate_child_depth": int(child_depth),
            "candidate_child_size_delta": int(child_size - parent_size),
            "candidate_child_depth_delta": int(child_depth - parent_depth),
            "candidate_root_op": str(cand_sub[0]),
            "exact_child_score_observed": False,
            "dedup_kept": False,
            "pre_dedup_rank": 0,
            "post_dedup_rank": 0,
            "raw_mse": None,
            "eff_mse": None,
        })
    return rows, list(cand_subtrees)


@torch.no_grad()
def run_inverse_proposal_family_compare_lab(
    spec: EquationSpec,
    *,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    candidate_expr: str | None = None,
    corrupt_path: str | Sequence[int] | None = None,
    replacement_expr: str | None = None,
    inverse_path: str | Sequence[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compare legacy vs direct-spec inverse proposals on a chosen repair hole."""

    hp = factorized_search_hp if factorized_search_hp is not None else default_oracle_hyperparams()
    run_seed = int(hp.seed if seed is None else seed)
    target_fn = compile_target_expression(spec)
    ds = build_oracle_dataset(
        spec,
        target_fn,
        n_fit=int(hp.n_fit),
        n_probe=int(hp.n_probe),
        seed=run_seed,
        dtype=dtype,
    )
    var_dims = ds["var_dims"] if enforce_dims else None

    candidate_ast, truth_ast, used_corrupt_path = build_candidate_ast_for_inverse_lab(
        spec,
        candidate_expr=candidate_expr,
        corrupt_path=corrupt_path,
        replacement_expr=replacement_expr,
        var_dims=var_dims,
    )
    candidate_paths = [tuple(int(v) for v in p) for p in collect_paths(candidate_ast) if tuple(int(v) for v in p)]
    user_path = _parse_path(inverse_path)
    chosen_path = user_path if user_path else used_corrupt_path
    chosen_path_source = "user" if user_path else ("corrupt_path" if used_corrupt_path else "default")
    if chosen_path is None:
        chosen_path = _default_inverse_path(candidate_ast)
    if tuple(chosen_path) not in set(candidate_paths):
        raise ValueError(
            f"chosen inverse path {_format_path(chosen_path)} is not present in candidate AST {node_str(candidate_ast)}"
        )

    base = _score_expr_against_target(
        candidate_ast,
        x_fit=ds["x_fit"],
        y_fit=ds["y_fit"],
        x_probe=ds["x_probe"],
        y_probe=ds["y_probe"],
        poly_degree=int(hp.poly_degree),
    )
    if base is None:
        raise RuntimeError(f"candidate AST could not be scored: {node_str(candidate_ast)}")
    mapping = base["mapping"]

    inv_safe_eps = getattr(hp, "inverse_safe_eps", None)
    if inv_safe_eps is None:
        inv_safe_eps = getattr(hp, "refine_safe_eps", 1.0e-12)
    inv_safe_eps = float(inv_safe_eps or 1.0e-12)
    inv_conf_mode = str(getattr(hp, "inverse_confidence_mode", "conditioning")).strip().lower() or "conditioning"
    inv_local_score_mode = _normalize_inverse_local_score_mode(
        getattr(hp, "inverse_local_score_mode", "affine"),
        default="affine",
    )
    inv_spec_local_score_mode = _normalize_inverse_local_score_mode(
        getattr(hp, "inverse_spec_local_score_mode", "affine"),
        default="affine",
    )
    topk_terms = max(2, int(getattr(hp, "inverse_topk_terms", 6)))
    shortlist_mult = max(1, int(getattr(hp, "inverse_shortlist_mult", 4)))
    local_limit = max(2, min(4, int(topk_terms)))
    support_floor_beams = max(1, (int(local_limit) + 1) // 2)
    transport_ctx = _estimate_inverse_action_transport(
        candidate_ast,
        mapping,
        ds["x_fit"],
        ds["y_fit"],
        ds["x_probe"],
        ds["y_probe"],
        [tuple(chosen_path)],
        safe_eps=float(inv_safe_eps),
    )
    cfg = {
        "max_paths": 1,
        "dm": bool(var_dims is not None),
        "var_dims": var_dims,
        "max_depth": int(getattr(hp, "max_depth", node_depth(candidate_ast))),
        "poly_degree": int(hp.poly_degree),
        "topk_terms": int(topk_terms),
        "shortlist_mult": int(shortlist_mult),
        "local_mode": str(inv_local_score_mode),
        "min_valid_frac": float(getattr(hp, "inverse_min_valid_frac", 0.25)),
        "min_confidence": float(getattr(hp, "inverse_min_confidence", 0.10)),
        "safe_eps": float(inv_safe_eps),
        "confidence_mode": str(inv_conf_mode),
        "confidence_target_gain": float(getattr(hp, "inverse_confidence_target_gain", 4.0)),
        "confidence_floor": float(getattr(hp, "inverse_confidence_floor", 0.05)),
        "branch_beam_width": int(getattr(hp, "inverse_branch_beam_width", 1)),
        "micro_search_enable": bool(getattr(hp, "inverse_micro_search_enable", False)),
        "micro_search_max_depth": int(getattr(hp, "inverse_micro_search_max_depth", 3)),
        "micro_search_beam_width": int(getattr(hp, "inverse_micro_search_beam_width", 24)),
        "micro_search_topk": int(getattr(hp, "inverse_micro_search_topk", 16)),
        "micro_search_seed_terms": int(getattr(hp, "inverse_micro_search_seed_terms", 8)),
        "target_mode": str(getattr(hp, "inverse_target_mode", "robust")),
        "full_mapping_penalty": float(getattr(hp, "inverse_full_mapping_penalty", 0.75)),
        "exact_simple_target_bonus": float(getattr(hp, "inverse_exact_simple_target_bonus", 0.10)),
        "additive_descend_penalty": float(getattr(hp, "inverse_additive_descend_penalty", 0.15)),
        "nonadditive_leaf_penalty": float(getattr(hp, "inverse_nonadditive_leaf_penalty", 0.20)),
        "exact_path_eta": float(getattr(hp, "inverse_exact_path_eta", 0.98)),
        "exact_transport_min_lin_rel": float(getattr(hp, "inverse_exact_transport_min_lin_rel", 0.0)),
        "periodic_min_valid_scale": float(getattr(hp, "inverse_periodic_min_valid_scale", 1.25)),
        "periodic_min_confidence_scale": float(getattr(hp, "inverse_periodic_min_confidence_scale", 1.35)),
        "periodic_path_penalty": float(getattr(hp, "inverse_periodic_path_penalty", 0.65)),
        "nonperiodic_muldiv_bonus": float(getattr(hp, "inverse_nonperiodic_muldiv_bonus", 0.10)),
        "nonperiodic_explogsqrt_bonus": float(getattr(hp, "inverse_nonperiodic_explogsqrt_bonus", 0.05)),
        "branch_ambiguity_penalty": float(getattr(hp, "inverse_branch_ambiguity_penalty", 0.50)),
        "transport_min_lin_rel": float(getattr(hp, "inverse_transport_min_lin_rel", 0.02)),
        "transport_min_effective_n": float(getattr(hp, "inverse_transport_min_effective_n", 8.0)),
    }
    pool_cache = _oracle_build_pool_cache(ds["x_fit"], ds["x_probe"], var_dims=var_dims)
    beam_states = _inverse_action_path_mode_beam_states(
        parent_node=candidate_ast,
        parent_mapping=mapping,
        x_fit=ds["x_fit"],
        y_fit=ds["y_fit"],
        x_probe=ds["x_probe"],
        y_probe=ds["y_probe"],
        pool_nodes=pool_cache["pool_nodes"],
        pool_phi_fit=pool_cache["pool_phi_fit"],
        pool_phi_probe=pool_cache["pool_phi_probe"],
        pool_dims=pool_cache["pool_dims"],
        all_paths=[tuple(chosen_path)],
        path_target_modes=None,
        transport_ctx=transport_ctx,
        cfg=cfg,
        beam_width=max(1, min(int(getattr(hp, "inverse_max_paths", 1)), 4)),
    )
    if not beam_states:
        raise RuntimeError(f"no inverse beam state available at path {_format_path(chosen_path)}")

    truth_paths = set(tuple(int(v) for v in p) for p in collect_paths(truth_ast))
    truth_node = get_at(truth_ast, tuple(chosen_path)) if tuple(chosen_path) in truth_paths else None
    path_profile = _inverse_path_profile(candidate_ast, tuple(chosen_path), mapping)
    slate_id = f"oracle_inverse_compare_{spec.id}_{run_seed}_{_format_path(chosen_path)}"

    family_reports: dict[str, Any] = {}
    for family_name in ("legacy_only", "direct_spec_only", "union"):
        t0 = time.perf_counter()
        candidate_rows_by_beam: dict[int, list[dict[str, Any]]] = {}
        all_preview_rows: list[dict[str, Any]] = []
        solver_meta_rows: list[dict[str, Any]] = []
        for beam_rank, beam_state in enumerate(beam_states):
            beam_rows: list[dict[str, Any]] = []
            legacy_rows: list[dict[str, Any]] = []
            legacy_seed_nodes: list[tuple] = []
            if family_name in ("legacy_only", "union"):
                legacy_rows, legacy_seed_nodes = _build_oracle_inverse_legacy_preview_rows_for_beam(
                    parent_node=candidate_ast,
                    beam_state=beam_state,
                    beam_rank=int(beam_rank),
                    slate_id=str(slate_id),
                    max_depth=int(getattr(hp, "max_depth", node_depth(candidate_ast))),
                    poly_degree=int(hp.poly_degree),
                    topk_terms=int(topk_terms),
                    shortlist_mult=int(shortlist_mult),
                    safe_eps=float(inv_safe_eps),
                    exact_transport_min_lin_rel=float(getattr(hp, "inverse_exact_transport_min_lin_rel", 0.0)),
                    var_dims=var_dims,
                    pool_nodes=pool_cache["pool_nodes"],
                    pool_dims=pool_cache["pool_dims"],
                    pool_phi_fit=pool_cache["pool_phi_fit"],
                    pool_phi_probe=pool_cache["pool_phi_probe"],
                    micro_search_enable=bool(getattr(hp, "inverse_micro_search_enable", False)),
                    micro_search_max_depth=int(getattr(hp, "inverse_micro_search_max_depth", 3)),
                    micro_search_beam_width=int(getattr(hp, "inverse_micro_search_beam_width", 24)),
                    micro_search_topk=int(getattr(hp, "inverse_micro_search_topk", 16)),
                    micro_search_seed_terms=int(getattr(hp, "inverse_micro_search_seed_terms", 8)),
                    local_score_mode=str(inv_local_score_mode),
                    transport_ctx=transport_ctx,
                )
                beam_rows.extend(legacy_rows)
            if family_name in ("direct_spec_only", "union"):
                spec_result = solve_inverse_spec_preview_rows(
                    parent_node=candidate_ast,
                    beam_state=beam_state,
                    beam_rank=int(beam_rank),
                    slate_id=str(slate_id),
                    max_depth=int(getattr(hp, "max_depth", node_depth(candidate_ast))),
                    nvars=int(ds["x_fit"].shape[1]),
                    poly_degree=int(hp.poly_degree),
                    var_dims=var_dims,
                    pool_nodes=pool_cache["pool_nodes"],
                    pool_dims=pool_cache["pool_dims"],
                    include_legacy_seed_nodes=(
                        legacy_seed_nodes if bool(getattr(hp, "inverse_spec_include_legacy_seed", True)) else None
                    ),
                    local_score_mode=str(inv_spec_local_score_mode),
                    enum_max_depth=int(getattr(hp, "inverse_spec_enum_max_depth", 4)),
                    enum_max_trees=int(getattr(hp, "inverse_spec_enum_max_trees", 5000)),
                    preview_topk=int(getattr(hp, "inverse_spec_preview_topk", 16)),
                    complexity_penalty=float(getattr(hp, "inverse_spec_complexity_penalty", 0.0)),
                    recursive_enable=bool(getattr(hp, "inverse_spec_recursive_enable", True)),
                    recursive_max_depth=int(getattr(hp, "inverse_spec_recursive_max_depth", 2)),
                    recursive_trigger_rel_mse=float(getattr(hp, "inverse_spec_recursive_trigger_rel_mse", 0.25)),
                    recursive_seed_cap=int(getattr(hp, "inverse_spec_recursive_seed_cap", 6)),
                    recursive_branch_topk=int(getattr(hp, "inverse_spec_recursive_branch_topk", 4)),
                    recursive_child_topk=int(getattr(hp, "inverse_spec_recursive_child_topk", 2)),
                    max_subtree_depth=getattr(hp, "inverse_spec_max_subtree_depth", None),
                    safe_eps=float(inv_safe_eps),
                    confidence_mode=str(inv_conf_mode),
                    confidence_target_gain=float(getattr(hp, "inverse_confidence_target_gain", 4.0)),
                    confidence_floor=float(getattr(hp, "inverse_confidence_floor", 0.05)),
                    branch_beam_width=int(getattr(hp, "inverse_branch_beam_width", 1)),
                )
                solver_meta = dict(spec_result.get("solver_meta", {}) or {})
                solver_meta["beam_rank"] = int(beam_rank)
                solver_meta_rows.append(solver_meta)
                beam_rows.extend([row for row in list(spec_result.get("rows", []) or []) if isinstance(row, dict)])
            candidate_rows_by_beam[int(beam_rank)] = beam_rows
            all_preview_rows.extend(beam_rows)

        for idx, row in enumerate(all_preview_rows):
            row["pre_dedup_rank"] = int(idx)

        grouped_rows: list[dict[str, Any]] = []
        exact_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        allocation_meta = {
            "support_floor_beams": 0,
            "support_floor_selected": 0,
            "global_allocated": 0,
        }
        truth_rank = None
        truth_present = False
        if all_preview_rows:
            grouped_rows, _duplicate_rows_by_key = _group_inverse_action_preview_rows(all_preview_rows)
            for beam_rows in candidate_rows_by_beam.values():
                _sort_inverse_action_candidate_rows_by_preview(beam_rows)
            selected_rows, allocation_meta = _select_inverse_exact_budget_rows(
                candidate_rows_by_beam=candidate_rows_by_beam,
                global_exact_score_budget=int(local_limit),
                support_floor_beams=int(min(len(beam_states), support_floor_beams)),
            )
            exact_rows = _oracle_exact_score_preview_rows(
                selected_rows,
                ds=ds,
                hp=hp,
            )
            truth_rank, truth_present = _oracle_truth_rank_from_grouped_rows(
                grouped_rows,
                path=tuple(chosen_path),
                truth_node=truth_node,
            )

        best_exact = exact_rows[0] if exact_rows else None
        truth_exact = None
        if truth_node is not None:
            truth_exact_rows = []
            for row in exact_rows:
                try:
                    cand_sub = get_at(row["expr"], tuple(chosen_path))
                except Exception:
                    continue
                if cand_sub == truth_node:
                    truth_exact_rows.append(row)
            if truth_exact_rows:
                truth_exact = min(
                    truth_exact_rows,
                    key=lambda row: (
                        float(row.get("full_probe_mse", float("inf"))),
                        float(row.get("full_fit_mse", float("inf"))),
                    ),
                )

        preview_top = []
        for row in grouped_rows[: min(12, len(grouped_rows))]:
            expr_ast = row.get("expr", None)
            is_truth = False
            if truth_node is not None and isinstance(expr_ast, tuple):
                try:
                    is_truth = bool(get_at(expr_ast, tuple(chosen_path)) == truth_node)
                except Exception:
                    is_truth = False
            preview_top.append({
                "expr": str(node_str(expr_ast)) if isinstance(expr_ast, tuple) else str(row.get("child_key", "")),
                "child_key": str(row.get("child_key", "") or ""),
                "proposal_family": str(row.get("proposal_family", row.get("tuple_provenance", "")) or ""),
                "tuple_provenance": str(row.get("tuple_provenance", "") or ""),
                "beam_rank": int(row.get("beam_rank", 0) or 0),
                "local_rank": int(row.get("local_rank", 0) or 0),
                "local_probe_mse": None if row.get("local_probe_mse", None) is None else float(row.get("local_probe_mse")),
                "local_fit_mse": None if row.get("local_fit_mse", None) is None else float(row.get("local_fit_mse")),
                "provenance_count": int(row.get("provenance_count", 1) or 1),
                "is_truth_candidate": bool(is_truth),
            })

        family_reports[family_name] = {
            "preview_candidate_count": int(len(all_preview_rows)),
            "preview_candidate_count_unique": int(len(grouped_rows)),
            "truth_rank": None if truth_rank is None else int(truth_rank),
            "truth_present": bool(truth_present),
            "exact_slice_budget": int(local_limit),
            "exact_scored_count": int(len(exact_rows)),
            "best_full_probe_mse": None if best_exact is None else float(best_exact.get("full_probe_mse", float("inf"))),
            "best_full_fit_mse": None if best_exact is None else float(best_exact.get("full_fit_mse", float("inf"))),
            "best_full_expr": None if best_exact is None else str(best_exact.get("expr_str", "")),
            "truth_exact_probe_mse": None if truth_exact is None else float(truth_exact.get("full_probe_mse", float("inf"))),
            "truth_exact_expr": None if truth_exact is None else str(truth_exact.get("expr_str", "")),
            "enumerated_tree_count": int(sum(int(row.get("enum_tree_count", 0) or 0) for row in solver_meta_rows)),
            "enum_depth_reached": int(max([int(row.get("enum_depth_reached", 0) or 0) for row in solver_meta_rows] or [0])),
            "solver_meta": [dict(row) for row in solver_meta_rows],
            "allocation_meta": dict(allocation_meta),
            "wall_seconds": float(time.perf_counter() - t0),
            "preview_top": preview_top,
        }

    report = {
        "mode": "inverse_proposal_family_compare_lab",
        "spec_id": spec.id,
        "seed": int(run_seed),
        "target_expr": spec.target_expr,
        "truth_expr": node_str(truth_ast),
        "candidate_expr": node_str(candidate_ast),
        "candidate_score": {
            "fit_mse": float(base["fit_mse"]),
            "probe_mse": float(base["probe_mse"]),
            "mapping_kind": str(base["mapping_kind"]),
        },
        "chosen_path": {
            "path": [int(v) for v in tuple(chosen_path)],
            "path_str": _format_path(tuple(chosen_path)),
            "source": str(chosen_path_source),
            "depth": int(len(tuple(chosen_path))),
            "current_subexpr": str(node_str(get_at(candidate_ast, tuple(chosen_path)))),
            "truth_subexpr": None if truth_node is None else str(node_str(truth_node)),
            "profile": {
                "has_periodic": bool(path_profile.get("has_periodic", False)),
                "has_muldiv": bool(path_profile.get("has_muldiv", False)),
                "has_explogsqrt": bool(path_profile.get("has_explogsqrt", False)),
                "exact_monotone": bool(path_profile.get("exact_monotone", False)),
            },
        },
        "beam_count": int(len(beam_states)),
        "beam_states": [
            {
                "beam_rank": int(i),
                "target_mode": str(state.get("target_mode", "") or ""),
                "target_mapping_kind": str(state.get("target_mapping_kind", "") or ""),
                "path_gain": float(state.get("path_gain", 0.0) or 0.0),
                "rel_gain": float(state.get("rel_gain", 0.0) or 0.0),
                "valid_frac": float(state.get("valid_frac", 0.0) or 0.0),
                "confidence": float(state.get("confidence", 0.0) or 0.0),
            }
            for i, state in enumerate(beam_states)
        ],
        "families": family_reports,
        "hp": {
            "inverse_topk_terms": int(topk_terms),
            "inverse_shortlist_mult": int(shortlist_mult),
            "inverse_local_score_mode": str(inv_local_score_mode),
            "inverse_spec_enable": bool(getattr(hp, "inverse_spec_enable", False)),
            "inverse_spec_enum_max_depth": int(getattr(hp, "inverse_spec_enum_max_depth", 4)),
            "inverse_spec_enum_max_trees": int(getattr(hp, "inverse_spec_enum_max_trees", 5000)),
            "inverse_spec_preview_topk": int(getattr(hp, "inverse_spec_preview_topk", 16)),
            "inverse_spec_local_score_mode": str(inv_spec_local_score_mode),
            "inverse_spec_include_legacy_seed": bool(getattr(hp, "inverse_spec_include_legacy_seed", True)),
            "inverse_spec_complexity_penalty": float(getattr(hp, "inverse_spec_complexity_penalty", 0.0)),
            "inverse_spec_repair_quota": float(getattr(hp, "inverse_spec_repair_quota", 0.0)),
            "repair_pass_enable": bool(getattr(hp, "repair_pass_enable", False)),
            "repair_pass_elite_k": int(getattr(hp, "repair_pass_elite_k", 8)),
            "repair_pass_paths_per_elite": int(getattr(hp, "repair_pass_paths_per_elite", 2)),
            "repair_pass_rounds": int(getattr(hp, "repair_pass_rounds", 2)),
            "closure_search_enable": bool(getattr(hp, "closure_search_enable", False)),
            "closure_search_families": list(getattr(hp, "closure_search_families", ["periodic", "exp", "log", "rational", "power", "quadratic"])),
            "closure_search_max_proposals": int(getattr(hp, "closure_search_max_proposals", 16)),
            "closure_search_anchors_per_family": int(getattr(hp, "closure_search_anchors_per_family", 4)),
            "closure_search_preview_topk": int(getattr(hp, "closure_search_preview_topk", 4)),
            "closure_search_exact_topk": int(getattr(hp, "closure_search_exact_topk", 2)),
            "closure_search_min_valid_frac": float(getattr(hp, "closure_search_min_valid_frac", 0.05)),
            "closure_search_min_confidence": float(getattr(hp, "closure_search_min_confidence", 0.02)),
            "closure_search_periodic_min_valid_scale": float(
                getattr(hp, "closure_search_periodic_min_valid_scale", 1.0)
            ),
            "closure_search_periodic_min_confidence_scale": float(
                getattr(hp, "closure_search_periodic_min_confidence_scale", 1.0)
            ),
            "closure_search_transport_min_lin_rel": float(
                getattr(hp, "closure_search_transport_min_lin_rel", 0.0)
            ),
            "closure_search_anchor_head_compare_enable": bool(
                getattr(hp, "closure_search_anchor_head_compare_enable", False)
            ),
            "hole_search_enable": bool(getattr(hp, "hole_search_enable", False)),
            "hole_search_quota": float(getattr(hp, "hole_search_quota", 0.10)),
            "hole_search_exact_budget": int(getattr(hp, "hole_search_exact_budget", 2)),
            "hole_search_cooldown_iters": int(getattr(hp, "hole_search_cooldown_iters", 32)),
            "hole_search_mine_cooldown_iters": int(getattr(hp, "hole_search_mine_cooldown_iters", 50)),
            "hole_search_max_frontier": int(getattr(hp, "hole_search_max_frontier", 128)),
            "hole_search_enum_max_depth": int(getattr(hp, "hole_search_enum_max_depth", 4)),
            "hole_search_enum_max_trees": int(getattr(hp, "hole_search_enum_max_trees", 3000)),
            "hole_search_preview_topk": int(getattr(hp, "hole_search_preview_topk", 8)),
            "hole_search_tournament_enable": bool(getattr(hp, "hole_search_tournament_enable", True)),
            "hole_search_tournament_n": int(getattr(hp, "hole_search_tournament_n", 8)),
            "hole_search_tournament_elite_k": int(getattr(hp, "hole_search_tournament_elite_k", 2)),
            "hole_search_tournament_preview_trees": int(getattr(hp, "hole_search_tournament_preview_trees", 64)),
            "inverse_spec_recursive_enable": bool(getattr(hp, "inverse_spec_recursive_enable", True)),
            "inverse_spec_recursive_max_depth": int(getattr(hp, "inverse_spec_recursive_max_depth", 2)),
            "inverse_spec_recursive_trigger_rel_mse": float(getattr(hp, "inverse_spec_recursive_trigger_rel_mse", 0.25)),
            "inverse_spec_recursive_seed_cap": int(getattr(hp, "inverse_spec_recursive_seed_cap", 6)),
            "inverse_spec_recursive_branch_topk": int(getattr(hp, "inverse_spec_recursive_branch_topk", 4)),
            "inverse_spec_recursive_child_topk": int(getattr(hp, "inverse_spec_recursive_child_topk", 2)),
            "inverse_spec_max_subtree_depth": getattr(hp, "inverse_spec_max_subtree_depth", None),
            "inverse_spec_fit_cap": int(getattr(hp, "inverse_spec_fit_cap", 96)),
            "inverse_spec_probe_cap": int(getattr(hp, "inverse_spec_probe_cap", 192)),
            "inverse_spec_exact_budget": int(getattr(hp, "inverse_spec_exact_budget", 4)),
        },
    }
    if verbose:
        print(
            f"[inverse-compare] {spec.id} path={_format_path(tuple(chosen_path))} "
            f"legacy_rank={family_reports['legacy_only']['truth_rank']} "
            f"spec_rank={family_reports['direct_spec_only']['truth_rank']} "
            f"union_rank={family_reports['union']['truth_rank']}"
        )
    return _to_jsonable(report)



def _apply_cli_overrides(hp: FactorizedSearchConfig, args: argparse.Namespace) -> FactorizedSearchConfig:
    # Core
    if args.n_iter is not None:
        hp.n_iter = int(args.n_iter)
    if getattr(args, "wall_time_limit_s", None) is not None:
        hp.wall_time_limit_s = float(args.wall_time_limit_s)
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
    no_residual = getattr(args, "no_residual", None)
    if no_residual is not None:
        hp.no_residual = bool(no_residual)
    inverse_steering_enable = getattr(args, "inverse_steering_enable", None)
    if inverse_steering_enable is not None:
        hp.inverse_steering_enable = bool(inverse_steering_enable)
    if args.inverse_max_paths is not None:
        hp.inverse_max_paths = int(args.inverse_max_paths)
    if args.inverse_topk_terms is not None:
        hp.inverse_topk_terms = int(args.inverse_topk_terms)
    if args.inverse_shortlist_mult is not None:
        hp.inverse_shortlist_mult = int(args.inverse_shortlist_mult)
    if args.inverse_min_valid_frac is not None:
        hp.inverse_min_valid_frac = float(args.inverse_min_valid_frac)
    if args.inverse_min_confidence is not None:
        hp.inverse_min_confidence = float(args.inverse_min_confidence)
    if args.inverse_safe_eps is not None:
        hp.inverse_safe_eps = float(args.inverse_safe_eps)
    if args.inverse_confidence_mode is not None:
        hp.inverse_confidence_mode = str(args.inverse_confidence_mode)
    if args.inverse_confidence_target_gain is not None:
        hp.inverse_confidence_target_gain = float(args.inverse_confidence_target_gain)
    if args.inverse_confidence_floor is not None:
        hp.inverse_confidence_floor = float(args.inverse_confidence_floor)
    if args.inverse_branch_beam_width is not None:
        hp.inverse_branch_beam_width = int(args.inverse_branch_beam_width)
    inverse_micro_search_enable = getattr(args, "inverse_micro_search_enable", None)
    if inverse_micro_search_enable is not None:
        hp.inverse_micro_search_enable = bool(inverse_micro_search_enable)
    if args.inverse_micro_search_max_depth is not None:
        hp.inverse_micro_search_max_depth = int(args.inverse_micro_search_max_depth)
    if args.inverse_micro_search_beam_width is not None:
        hp.inverse_micro_search_beam_width = int(args.inverse_micro_search_beam_width)
    if args.inverse_micro_search_topk is not None:
        hp.inverse_micro_search_topk = int(args.inverse_micro_search_topk)
    if args.inverse_micro_search_seed_terms is not None:
        hp.inverse_micro_search_seed_terms = int(args.inverse_micro_search_seed_terms)
    if args.inverse_local_score_mode is not None:
        hp.inverse_local_score_mode = str(args.inverse_local_score_mode)
    inverse_spec_enable = getattr(args, "inverse_spec_enable", None)
    if inverse_spec_enable is not None:
        hp.inverse_spec_enable = bool(inverse_spec_enable)
    if getattr(args, "inverse_spec_enum_max_depth", None) is not None:
        hp.inverse_spec_enum_max_depth = int(args.inverse_spec_enum_max_depth)
    if getattr(args, "inverse_spec_enum_max_trees", None) is not None:
        hp.inverse_spec_enum_max_trees = int(args.inverse_spec_enum_max_trees)
    if getattr(args, "inverse_spec_preview_topk", None) is not None:
        hp.inverse_spec_preview_topk = int(args.inverse_spec_preview_topk)
    if getattr(args, "inverse_spec_local_score_mode", None) is not None:
        hp.inverse_spec_local_score_mode = str(args.inverse_spec_local_score_mode)
    inverse_spec_include_legacy_seed = getattr(args, "inverse_spec_include_legacy_seed", None)
    if inverse_spec_include_legacy_seed is not None:
        hp.inverse_spec_include_legacy_seed = bool(inverse_spec_include_legacy_seed)
    if getattr(args, "inverse_spec_complexity_penalty", None) is not None:
        hp.inverse_spec_complexity_penalty = float(args.inverse_spec_complexity_penalty)
    if getattr(args, "inverse_spec_repair_quota", None) is not None:
        hp.inverse_spec_repair_quota = float(args.inverse_spec_repair_quota)
    if getattr(args, "hole_search_enable", None) is not None:
        hp.hole_search_enable = bool(args.hole_search_enable)
    if getattr(args, "hole_search_quota", None) is not None:
        hp.hole_search_quota = float(args.hole_search_quota)
    if getattr(args, "hole_search_exact_budget", None) is not None:
        hp.hole_search_exact_budget = int(args.hole_search_exact_budget)
    if getattr(args, "hole_search_cooldown_iters", None) is not None:
        hp.hole_search_cooldown_iters = int(args.hole_search_cooldown_iters)
    if getattr(args, "hole_search_mine_cooldown_iters", None) is not None:
        hp.hole_search_mine_cooldown_iters = int(args.hole_search_mine_cooldown_iters)
    if getattr(args, "hole_search_max_frontier", None) is not None:
        hp.hole_search_max_frontier = int(args.hole_search_max_frontier)
    hole_search_first_class_scheduler_enable = getattr(args, "hole_search_first_class_scheduler_enable", None)
    if hole_search_first_class_scheduler_enable is not None:
        hp.hole_search_first_class_scheduler_enable = bool(hole_search_first_class_scheduler_enable)
    hole_search_route_scheduler_enable = getattr(args, "hole_search_route_scheduler_enable", None)
    if hole_search_route_scheduler_enable is not None:
        hp.hole_search_route_scheduler_enable = bool(hole_search_route_scheduler_enable)
    if getattr(args, "hole_search_route_ucb_c", None) is not None:
        hp.hole_search_route_ucb_c = float(args.hole_search_route_ucb_c)
    if getattr(args, "hole_search_route_eps", None) is not None:
        hp.hole_search_route_eps = float(args.hole_search_route_eps)
    if getattr(args, "hole_search_route_acquisition_weight", None) is not None:
        hp.hole_search_route_acquisition_weight = float(args.hole_search_route_acquisition_weight)
    if getattr(args, "hole_search_route_reward_mode", None) is not None:
        hp.hole_search_route_reward_mode = str(args.hole_search_route_reward_mode)
    if getattr(args, "hole_search_route_time_penalty", None) is not None:
        hp.hole_search_route_time_penalty = float(args.hole_search_route_time_penalty)
    if getattr(args, "hole_search_route_time_floor", None) is not None:
        hp.hole_search_route_time_floor = float(args.hole_search_route_time_floor)
    hole_search_abstraction_enable = getattr(args, "hole_search_abstraction_enable", None)
    if hole_search_abstraction_enable is not None:
        hp.hole_search_abstraction_enable = bool(hole_search_abstraction_enable)
    hole_search_abstraction_on_improve = getattr(args, "hole_search_abstraction_on_improve", None)
    if hole_search_abstraction_on_improve is not None:
        hp.hole_search_abstraction_on_improve = bool(hole_search_abstraction_on_improve)
    hole_search_abstraction_on_stall = getattr(args, "hole_search_abstraction_on_stall", None)
    if hole_search_abstraction_on_stall is not None:
        hp.hole_search_abstraction_on_stall = bool(hole_search_abstraction_on_stall)
    if getattr(args, "hole_search_abstraction_cooldown_iters", None) is not None:
        hp.hole_search_abstraction_cooldown_iters = int(args.hole_search_abstraction_cooldown_iters)
    if getattr(args, "hole_search_abstraction_max_parents", None) is not None:
        hp.hole_search_abstraction_max_parents = int(args.hole_search_abstraction_max_parents)
    if getattr(args, "hole_search_abstraction_max_paths_per_parent", None) is not None:
        hp.hole_search_abstraction_max_paths_per_parent = int(args.hole_search_abstraction_max_paths_per_parent)
    if getattr(args, "hole_search_abstraction_improve_min_delta_log_mse", None) is not None:
        hp.hole_search_abstraction_improve_min_delta_log_mse = float(
            args.hole_search_abstraction_improve_min_delta_log_mse
        )
    if getattr(args, "hole_search_abstraction_stage_enable", None) is not None:
        hp.hole_search_abstraction_stage_enable = bool(args.hole_search_abstraction_stage_enable)
    if getattr(args, "hole_search_abstraction_stage_max_entries", None) is not None:
        hp.hole_search_abstraction_stage_max_entries = int(args.hole_search_abstraction_stage_max_entries)
    if getattr(args, "hole_search_abstraction_promote_topk", None) is not None:
        hp.hole_search_abstraction_promote_topk = int(args.hole_search_abstraction_promote_topk)
    if getattr(args, "hole_search_abstraction_promote_frontier_floor", None) is not None:
        hp.hole_search_abstraction_promote_frontier_floor = int(args.hole_search_abstraction_promote_frontier_floor)
    if getattr(args, "hole_search_enum_max_depth", None) is not None:
        hp.hole_search_enum_max_depth = int(args.hole_search_enum_max_depth)
    if getattr(args, "hole_search_enum_max_trees", None) is not None:
        hp.hole_search_enum_max_trees = int(args.hole_search_enum_max_trees)
    if getattr(args, "hole_search_preview_topk", None) is not None:
        hp.hole_search_preview_topk = int(args.hole_search_preview_topk)
    if getattr(args, "hole_search_tournament_enable", None) is not None:
        hp.hole_search_tournament_enable = bool(args.hole_search_tournament_enable)
    if getattr(args, "hole_search_tournament_n", None) is not None:
        hp.hole_search_tournament_n = int(args.hole_search_tournament_n)
    if getattr(args, "hole_search_tournament_elite_k", None) is not None:
        hp.hole_search_tournament_elite_k = int(args.hole_search_tournament_elite_k)
    if getattr(args, "hole_search_tournament_preview_trees", None) is not None:
        hp.hole_search_tournament_preview_trees = int(args.hole_search_tournament_preview_trees)
    if getattr(args, "stall_window", None) is not None:
        hp.stall_window = int(args.stall_window)
    if getattr(args, "stall_patience", None) is not None:
        hp.stall_patience = int(args.stall_patience)
    if getattr(args, "stall_delta", None) is not None:
        hp.stall_delta = float(args.stall_delta)
    inverse_spec_recursive_enable = getattr(args, "inverse_spec_recursive_enable", None)
    if inverse_spec_recursive_enable is not None:
        hp.inverse_spec_recursive_enable = bool(inverse_spec_recursive_enable)
    if getattr(args, "inverse_spec_recursive_max_depth", None) is not None:
        hp.inverse_spec_recursive_max_depth = int(args.inverse_spec_recursive_max_depth)
    if getattr(args, "inverse_spec_recursive_trigger_rel_mse", None) is not None:
        hp.inverse_spec_recursive_trigger_rel_mse = float(args.inverse_spec_recursive_trigger_rel_mse)
    if getattr(args, "inverse_spec_recursive_seed_cap", None) is not None:
        hp.inverse_spec_recursive_seed_cap = int(args.inverse_spec_recursive_seed_cap)
    if getattr(args, "inverse_spec_recursive_branch_topk", None) is not None:
        hp.inverse_spec_recursive_branch_topk = int(args.inverse_spec_recursive_branch_topk)
    if getattr(args, "inverse_spec_recursive_child_topk", None) is not None:
        hp.inverse_spec_recursive_child_topk = int(args.inverse_spec_recursive_child_topk)
    if getattr(args, "inverse_spec_max_subtree_depth", None) is not None:
        hp.inverse_spec_max_subtree_depth = int(args.inverse_spec_max_subtree_depth)
    if getattr(args, "inverse_spec_fit_cap", None) is not None:
        hp.inverse_spec_fit_cap = int(args.inverse_spec_fit_cap)
    if getattr(args, "inverse_spec_probe_cap", None) is not None:
        hp.inverse_spec_probe_cap = int(args.inverse_spec_probe_cap)
    if getattr(args, "inverse_spec_exact_budget", None) is not None:
        hp.inverse_spec_exact_budget = int(args.inverse_spec_exact_budget)
    if args.inverse_target_mode is not None:
        hp.inverse_target_mode = str(args.inverse_target_mode)
    if args.inverse_full_mapping_penalty is not None:
        hp.inverse_full_mapping_penalty = float(args.inverse_full_mapping_penalty)
    if args.inverse_exact_simple_target_bonus is not None:
        hp.inverse_exact_simple_target_bonus = float(args.inverse_exact_simple_target_bonus)
    if args.inverse_additive_descend_penalty is not None:
        hp.inverse_additive_descend_penalty = float(args.inverse_additive_descend_penalty)
    if args.inverse_nonadditive_leaf_penalty is not None:
        hp.inverse_nonadditive_leaf_penalty = float(args.inverse_nonadditive_leaf_penalty)
    if args.inverse_exact_path_eta is not None:
        hp.inverse_exact_path_eta = float(args.inverse_exact_path_eta)
    if args.inverse_exact_transport_min_lin_rel is not None:
        hp.inverse_exact_transport_min_lin_rel = float(args.inverse_exact_transport_min_lin_rel)
    inverse_gate_enable = getattr(args, "inverse_gate_enable", None)
    if inverse_gate_enable is not None:
        hp.inverse_gate_enable = bool(inverse_gate_enable)
    if args.inverse_gate_warmup is not None:
        hp.inverse_gate_warmup = int(args.inverse_gate_warmup)
    if args.inverse_gate_best_factor is not None:
        hp.inverse_gate_best_factor = float(args.inverse_gate_best_factor)
    if args.inverse_gate_min_depth is not None:
        hp.inverse_gate_min_depth = int(args.inverse_gate_min_depth)
    if args.inverse_gate_min_size is not None:
        hp.inverse_gate_min_size = int(args.inverse_gate_min_size)
    if args.inverse_gate_max_paths is not None:
        hp.inverse_gate_max_paths = int(args.inverse_gate_max_paths)
    if args.inverse_gate_min_structural_score is not None:
        hp.inverse_gate_min_structural_score = float(args.inverse_gate_min_structural_score)
    if args.inverse_gate_min_weighted_rel_gain is not None:
        hp.inverse_gate_min_weighted_rel_gain = float(args.inverse_gate_min_weighted_rel_gain)
    if args.inverse_gate_structural_bias is not None:
        hp.inverse_gate_structural_bias = float(args.inverse_gate_structural_bias)
    if args.inverse_periodic_min_valid_scale is not None:
        hp.inverse_periodic_min_valid_scale = float(args.inverse_periodic_min_valid_scale)
    if args.inverse_periodic_min_confidence_scale is not None:
        hp.inverse_periodic_min_confidence_scale = float(args.inverse_periodic_min_confidence_scale)
    if args.inverse_periodic_path_penalty is not None:
        hp.inverse_periodic_path_penalty = float(args.inverse_periodic_path_penalty)
    if args.inverse_nonperiodic_muldiv_bonus is not None:
        hp.inverse_nonperiodic_muldiv_bonus = float(args.inverse_nonperiodic_muldiv_bonus)
    if args.inverse_nonperiodic_explogsqrt_bonus is not None:
        hp.inverse_nonperiodic_explogsqrt_bonus = float(args.inverse_nonperiodic_explogsqrt_bonus)
    if args.inverse_branch_ambiguity_penalty is not None:
        hp.inverse_branch_ambiguity_penalty = float(args.inverse_branch_ambiguity_penalty)
    if args.inverse_transport_min_lin_rel is not None:
        hp.inverse_transport_min_lin_rel = float(args.inverse_transport_min_lin_rel)
    if args.inverse_transport_min_effective_n is not None:
        hp.inverse_transport_min_effective_n = float(args.inverse_transport_min_effective_n)
    inverse_experiment_log_enable = getattr(args, "inverse_experiment_log_enable", None)
    if inverse_experiment_log_enable is not None:
        hp.inverse_experiment_log_enable = bool(inverse_experiment_log_enable)
    repair_controller_enable = getattr(args, "repair_controller_enable", None)
    if repair_controller_enable is not None:
        hp.repair_controller_enable = bool(repair_controller_enable)
    if getattr(args, "repair_controller_min_score", None) is not None:
        hp.repair_controller_min_score = float(args.repair_controller_min_score)
    if getattr(args, "repair_controller_steps", None) is not None:
        hp.repair_controller_steps = int(args.repair_controller_steps)
    if getattr(args, "repair_controller_ancestor_hops", None) is not None:
        hp.repair_controller_ancestor_hops = int(args.repair_controller_ancestor_hops)
    if getattr(args, "repair_controller_min_step_rel_improve", None) is not None:
        hp.repair_controller_min_step_rel_improve = float(args.repair_controller_min_step_rel_improve)
    repair_controller_adaptive = getattr(args, "repair_controller_adaptive", None)
    if repair_controller_adaptive is not None:
        hp.repair_controller_adaptive = bool(repair_controller_adaptive)
    if getattr(args, "repair_controller_adapt_quantile", None) is not None:
        hp.repair_controller_adapt_quantile = float(args.repair_controller_adapt_quantile)
    if getattr(args, "repair_controller_adapt_window", None) is not None:
        hp.repair_controller_adapt_window = int(args.repair_controller_adapt_window)
    if getattr(args, "repair_controller_adapt_min_samples", None) is not None:
        hp.repair_controller_adapt_min_samples = int(args.repair_controller_adapt_min_samples)
    if getattr(args, "repair_controller_min_concentration", None) is not None:
        hp.repair_controller_min_concentration = float(args.repair_controller_min_concentration)
    if getattr(args, "repair_controller_potential_weight", None) is not None:
        hp.repair_controller_potential_weight = float(args.repair_controller_potential_weight)
    if getattr(args, "repair_controller_concentration_weight", None) is not None:
        hp.repair_controller_concentration_weight = float(args.repair_controller_concentration_weight)
    if getattr(args, "repair_controller_contrast_weight", None) is not None:
        hp.repair_controller_contrast_weight = float(args.repair_controller_contrast_weight)
    if getattr(args, "repair_controller_cost_weight", None) is not None:
        hp.repair_controller_cost_weight = float(args.repair_controller_cost_weight)
    if getattr(args, "repair_controller_stagnation_weight", None) is not None:
        hp.repair_controller_stagnation_weight = float(args.repair_controller_stagnation_weight)
    if getattr(args, "repair_controller_frontier_topk", None) is not None:
        hp.repair_controller_frontier_topk = int(args.repair_controller_frontier_topk)
    if getattr(args, "repair_controller_stagnation_visits", None) is not None:
        hp.repair_controller_stagnation_visits = int(args.repair_controller_stagnation_visits)
    if getattr(args, "repair_controller_focus_prob", None) is not None:
        hp.repair_controller_focus_prob = float(args.repair_controller_focus_prob)
    if getattr(args, "repair_controller_parent_max_repeats", None) is not None:
        hp.repair_controller_parent_max_repeats = int(args.repair_controller_parent_max_repeats)
    if getattr(args, "repair_controller_parent_min_eval_gap", None) is not None:
        hp.repair_controller_parent_min_eval_gap = int(args.repair_controller_parent_min_eval_gap)
    if getattr(args, "repair_controller_parent_reset_rel_improve", None) is not None:
        hp.repair_controller_parent_reset_rel_improve = float(args.repair_controller_parent_reset_rel_improve)
    if getattr(args, "repair_controller_critic_enable", None) is not None:
        hp.repair_controller_critic_enable = bool(args.repair_controller_critic_enable)
    if getattr(args, "repair_controller_critic_path", None) is not None:
        hp.repair_controller_critic_path = str(args.repair_controller_critic_path)
    if getattr(args, "repair_controller_critic_blend", None) is not None:
        hp.repair_controller_critic_blend = float(args.repair_controller_critic_blend)
    if getattr(args, "repair_controller_critic_mode", None) is not None:
        hp.repair_controller_critic_mode = str(args.repair_controller_critic_mode)
    if args.brute_depth is not None:
        hp.brute_depth = int(args.brute_depth)
    if args.no_brute_force:
        hp.brute_depth = 0
    if getattr(args, "score_mapping_family_mode", None) is not None:
        hp.score_mapping_family_mode = str(args.score_mapping_family_mode)
    if getattr(args, "brute_score_mapping_family_mode", None) is not None:
        hp.brute_score_mapping_family_mode = str(args.brute_score_mapping_family_mode)
    if getattr(args, "score_mapping_expensive_gate_best_factor", None) is not None:
        hp.score_mapping_expensive_gate_best_factor = float(args.score_mapping_expensive_gate_best_factor)
    if getattr(args, "score_mapping_expensive_rel_y", None) is not None:
        hp.score_mapping_expensive_rel_y = float(args.score_mapping_expensive_rel_y)
    score_prescreen_enable = getattr(args, "score_prescreen_enable", None)
    if score_prescreen_enable is not None:
        hp.score_prescreen_enable = bool(score_prescreen_enable)
    if getattr(args, "score_prescreen_family_mode", None) is not None:
        hp.score_prescreen_family_mode = str(args.score_prescreen_family_mode)
    if getattr(args, "score_prescreen_residual_family_mode", None) is not None:
        hp.score_prescreen_residual_family_mode = str(args.score_prescreen_residual_family_mode)
    score_prescreen_residual_allow_hint = getattr(args, "score_prescreen_residual_allow_hint", None)
    if score_prescreen_residual_allow_hint is not None:
        hp.score_prescreen_residual_allow_hint = bool(score_prescreen_residual_allow_hint)
    score_prescreen_residual_use_global_best = getattr(args, "score_prescreen_residual_use_global_best", None)
    if score_prescreen_residual_use_global_best is not None:
        hp.score_prescreen_residual_use_global_best = bool(score_prescreen_residual_use_global_best)
    if getattr(args, "score_prescreen_parent_best_factor", None) is not None:
        hp.score_prescreen_parent_best_factor = float(args.score_prescreen_parent_best_factor)
    if getattr(args, "score_prescreen_global_best_factor", None) is not None:
        hp.score_prescreen_global_best_factor = float(args.score_prescreen_global_best_factor)
    if getattr(args, "score_prescreen_residual_parent_best_factor", None) is not None:
        hp.score_prescreen_residual_parent_best_factor = float(args.score_prescreen_residual_parent_best_factor)
    if getattr(args, "score_prescreen_residual_global_best_factor", None) is not None:
        hp.score_prescreen_residual_global_best_factor = float(args.score_prescreen_residual_global_best_factor)
    if getattr(args, "fast_benchmark", False):
        hp.brute_depth = 0
    if args.early_stop_mse is not None:
        hp.early_stop_mse = float(args.early_stop_mse)

    # continuous skeleton refinement
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

    return hp


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle factorized symbolic search/continuous skeleton refinement lab runner")
    p.add_argument("--spec", type=str, required=True, help="Equation spec file (.json/.yaml)")
    p.add_argument("--output", type=str, default=None, help="Optional JSON report path")
    p.add_argument("--seed", type=int, default=None, help="Base random seed")
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    p.add_argument("--ignore_dims", action="store_true", help="Disable dimensional filtering")
    p.add_argument(
        "--gs-carrier-seed", dest="gs_carrier_seed", action="store_true",
        help="SR GS->FSS bridge: discover internal coordinate(s) z(x) with the generalized-symmetry layer (charts/composition/warp) and seed them as FSS carriers so the outer-map battery fits g(z) directly. Default off.",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce explorer logging")

    # Core search overrides
    p.add_argument("--n_iter", type=int, default=None)
    p.add_argument("--wall_time_limit_s", type=float, default=None)
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
    residual_g = p.add_mutually_exclusive_group()
    residual_g.add_argument(
        "--no_residual",
        dest="no_residual",
        action="store_true",
        help="Disable residual action",
    )
    residual_g.add_argument(
        "--residual",
        dest="no_residual",
        action="store_false",
        help="Enable residual action",
    )
    p.set_defaults(no_residual=None)

    inverse_g = p.add_mutually_exclusive_group()
    inverse_g.add_argument(
        "--inverse_steering",
        dest="inverse_steering_enable",
        action="store_true",
        help="Enable context-sensitive inverse steering action",
    )
    inverse_g.add_argument(
        "--no_inverse_steering",
        dest="inverse_steering_enable",
        action="store_false",
        help="Disable context-sensitive inverse steering action",
    )
    p.set_defaults(inverse_steering_enable=None)
    p.add_argument("--inverse_max_paths", type=int, default=None)
    p.add_argument("--inverse_topk_terms", type=int, default=None)
    p.add_argument("--inverse_shortlist_mult", type=int, default=None)
    p.add_argument("--inverse_min_valid_frac", type=float, default=None)
    p.add_argument("--inverse_min_confidence", type=float, default=None)
    p.add_argument("--inverse_safe_eps", type=float, default=None)
    p.add_argument(
        "--inverse_confidence_mode",
        type=str,
        choices=["conditioning", "heuristic"],
        default=None,
        help="Confidence model for inverse-target propagation",
    )
    p.add_argument("--inverse_confidence_target_gain", type=float, default=None)
    p.add_argument("--inverse_confidence_floor", type=float, default=None)
    p.add_argument("--inverse_branch_beam_width", type=int, default=None)
    inverse_micro_g = p.add_mutually_exclusive_group()
    inverse_micro_g.add_argument(
        "--inverse_micro_search",
        dest="inverse_micro_search_enable",
        action="store_true",
        help="Enable local subtree micro-search for inverse steering",
    )
    inverse_micro_g.add_argument(
        "--no_inverse_micro_search",
        dest="inverse_micro_search_enable",
        action="store_false",
        help="Disable local subtree micro-search for inverse steering",
    )
    p.set_defaults(inverse_micro_search_enable=None)
    p.add_argument("--inverse_micro_search_max_depth", type=int, default=None)
    p.add_argument("--inverse_micro_search_beam_width", type=int, default=None)
    p.add_argument("--inverse_micro_search_topk", type=int, default=None)
    p.add_argument("--inverse_micro_search_seed_terms", type=int, default=None)
    p.add_argument(
        "--inverse_local_score_mode",
        type=str,
        choices=["strict", "affine", "fitbest"],
        default=None,
    )
    inverse_spec_g = p.add_mutually_exclusive_group()
    inverse_spec_g.add_argument(
        "--inverse_spec",
        dest="inverse_spec_enable",
        action="store_true",
        help="Enable direct-spec inverse proposal generation inside the search action",
    )
    inverse_spec_g.add_argument(
        "--no_inverse_spec",
        dest="inverse_spec_enable",
        action="store_false",
        help="Disable direct-spec inverse proposal generation inside the search action",
    )
    p.set_defaults(inverse_spec_enable=None)
    p.add_argument("--inverse_spec_enum_max_depth", type=int, default=None)
    p.add_argument("--inverse_spec_enum_max_trees", type=int, default=None)
    p.add_argument("--inverse_spec_preview_topk", type=int, default=None)
    p.add_argument(
        "--inverse_spec_local_score_mode",
        type=str,
        choices=["strict", "affine", "fitbest"],
        default=None,
    )
    inverse_spec_seed_g = p.add_mutually_exclusive_group()
    inverse_spec_seed_g.add_argument(
        "--inverse_spec_legacy_seed",
        dest="inverse_spec_include_legacy_seed",
        action="store_true",
        help="Seed the direct-spec solver with legacy inverse local candidates",
    )
    inverse_spec_seed_g.add_argument(
        "--no_inverse_spec_legacy_seed",
        dest="inverse_spec_include_legacy_seed",
        action="store_false",
        help="Do not seed the direct-spec solver with legacy inverse local candidates",
    )
    p.set_defaults(inverse_spec_include_legacy_seed=None)
    p.add_argument("--inverse_spec_complexity_penalty", type=float, default=None)
    p.add_argument("--inverse_spec_repair_quota", type=float, default=None)
    p.add_argument("--hole_search_enable", action="store_true", default=None)
    p.add_argument("--no_hole_search", dest="hole_search_enable", action="store_false")
    p.add_argument("--hole_search_quota", type=float, default=None)
    p.add_argument("--hole_search_exact_budget", type=int, default=None)
    p.add_argument("--hole_search_cooldown_iters", type=int, default=None)
    p.add_argument("--hole_search_mine_cooldown_iters", type=int, default=None)
    p.add_argument("--hole_search_max_frontier", type=int, default=None)
    hole_first_class_g = p.add_mutually_exclusive_group()
    hole_first_class_g.add_argument(
        "--hole_search_first_class_scheduler",
        dest="hole_search_first_class_scheduler_enable",
        action="store_true",
        help="Schedule hole/spec opportunities as first-class agenda items",
    )
    hole_first_class_g.add_argument(
        "--no_hole_search_first_class_scheduler",
        dest="hole_search_first_class_scheduler_enable",
        action="store_false",
        help="Keep hole search subordinate to the expression mutation loop",
    )
    p.set_defaults(hole_search_first_class_scheduler_enable=None)
    hole_route_g = p.add_mutually_exclusive_group()
    hole_route_g.add_argument(
        "--hole_search_route_scheduler",
        dest="hole_search_route_scheduler_enable",
        action="store_true",
        help="Use a route scheduler over expression expansion vs hole opportunities",
    )
    hole_route_g.add_argument(
        "--no_hole_search_route_scheduler",
        dest="hole_search_route_scheduler_enable",
        action="store_false",
        help="Disable the route scheduler and fall back to legacy quota override",
    )
    p.set_defaults(hole_search_route_scheduler_enable=None)
    p.add_argument("--hole_search_route_ucb_c", type=float, default=None)
    p.add_argument("--hole_search_route_eps", type=float, default=None)
    p.add_argument("--hole_search_route_acquisition_weight", type=float, default=None)
    p.add_argument("--hole_search_route_reward_mode", choices=["raw", "per_second", "penalized"], default=None)
    p.add_argument("--hole_search_route_time_penalty", type=float, default=None)
    p.add_argument("--hole_search_route_time_floor", type=float, default=None)
    hole_abs_g = p.add_mutually_exclusive_group()
    hole_abs_g.add_argument(
        "--hole_search_abstraction",
        dest="hole_search_abstraction_enable",
        action="store_true",
        help="Enable explicit hole abstraction from improving/stalled parents",
    )
    hole_abs_g.add_argument(
        "--no_hole_search_abstraction",
        dest="hole_search_abstraction_enable",
        action="store_false",
        help="Disable explicit hole abstraction triggers",
    )
    p.set_defaults(hole_search_abstraction_enable=None)
    hole_abs_improve_g = p.add_mutually_exclusive_group()
    hole_abs_improve_g.add_argument(
        "--hole_search_abstraction_on_improve",
        dest="hole_search_abstraction_on_improve",
        action="store_true",
        help="Emit hole opportunities from accepted/improving children",
    )
    hole_abs_improve_g.add_argument(
        "--no_hole_search_abstraction_on_improve",
        dest="hole_search_abstraction_on_improve",
        action="store_false",
        help="Disable improve-triggered hole abstraction",
    )
    p.set_defaults(hole_search_abstraction_on_improve=None)
    hole_abs_stall_g = p.add_mutually_exclusive_group()
    hole_abs_stall_g.add_argument(
        "--hole_search_abstraction_on_stall",
        dest="hole_search_abstraction_on_stall",
        action="store_true",
        help="Emit hole opportunities from promising archive parents during stalls",
    )
    hole_abs_stall_g.add_argument(
        "--no_hole_search_abstraction_on_stall",
        dest="hole_search_abstraction_on_stall",
        action="store_false",
        help="Disable stall-triggered hole abstraction",
    )
    p.set_defaults(hole_search_abstraction_on_stall=None)
    p.add_argument("--hole_search_abstraction_cooldown_iters", type=int, default=None)
    p.add_argument("--hole_search_abstraction_max_parents", type=int, default=None)
    p.add_argument("--hole_search_abstraction_max_paths_per_parent", type=int, default=None)
    p.add_argument("--hole_search_abstraction_improve_min_delta_log_mse", type=float, default=None)
    hole_abs_stage_g = p.add_mutually_exclusive_group()
    hole_abs_stage_g.add_argument(
        "--hole_search_abstraction_stage",
        dest="hole_search_abstraction_stage_enable",
        action="store_true",
        help="Stage abstraction-origin opportunities before promoting them into the executable hole frontier",
    )
    hole_abs_stage_g.add_argument(
        "--no_hole_search_abstraction_stage",
        dest="hole_search_abstraction_stage_enable",
        action="store_false",
        help="Insert abstraction-origin opportunities directly into the executable hole frontier",
    )
    p.set_defaults(hole_search_abstraction_stage_enable=None)
    p.add_argument("--hole_search_abstraction_stage_max_entries", type=int, default=None)
    p.add_argument("--hole_search_abstraction_promote_topk", type=int, default=None)
    p.add_argument("--hole_search_abstraction_promote_frontier_floor", type=int, default=None)
    p.add_argument("--hole_search_enum_max_depth", type=int, default=None)
    p.add_argument("--hole_search_enum_max_trees", type=int, default=None)
    p.add_argument("--hole_search_preview_topk", type=int, default=None)
    hole_tournament_g = p.add_mutually_exclusive_group()
    hole_tournament_g.add_argument(
        "--hole_search_tournament",
        dest="hole_search_tournament_enable",
        action="store_true",
        help="Enable risk-seeking tournament for hole search (cheap-preview many, full-execute few)",
    )
    hole_tournament_g.add_argument(
        "--no_hole_search_tournament",
        dest="hole_search_tournament_enable",
        action="store_false",
        help="Disable risk-seeking tournament (legacy one-at-a-time selection)",
    )
    p.set_defaults(hole_search_tournament_enable=None)
    p.add_argument("--hole_search_tournament_n", type=int, default=None)
    p.add_argument("--hole_search_tournament_elite_k", type=int, default=None)
    p.add_argument("--hole_search_tournament_preview_trees", type=int, default=None)
    p.add_argument("--stall_window", type=int, default=None)
    p.add_argument("--stall_patience", type=int, default=None)
    p.add_argument("--stall_delta", type=float, default=None)
    inverse_spec_recursive_g = p.add_mutually_exclusive_group()
    inverse_spec_recursive_g.add_argument(
        "--inverse_spec_recursive",
        dest="inverse_spec_recursive_enable",
        action="store_true",
        help="Enable recursive structural decomposition inside the direct-spec solver",
    )
    inverse_spec_recursive_g.add_argument(
        "--no_inverse_spec_recursive",
        dest="inverse_spec_recursive_enable",
        action="store_false",
        help="Disable recursive structural decomposition inside the direct-spec solver",
    )
    p.set_defaults(inverse_spec_recursive_enable=None)
    p.add_argument("--inverse_spec_recursive_max_depth", type=int, default=None)
    p.add_argument("--inverse_spec_recursive_trigger_rel_mse", type=float, default=None)
    p.add_argument("--inverse_spec_recursive_seed_cap", type=int, default=None)
    p.add_argument("--inverse_spec_recursive_branch_topk", type=int, default=None)
    p.add_argument("--inverse_spec_recursive_child_topk", type=int, default=None)
    p.add_argument("--inverse_spec_max_subtree_depth", type=int, default=None)
    p.add_argument("--inverse_spec_fit_cap", type=int, default=96)
    p.add_argument("--inverse_spec_probe_cap", type=int, default=192)
    p.add_argument("--inverse_spec_exact_budget", type=int, default=4)
    p.add_argument(
        "--inverse_target_mode",
        type=str,
        choices=["robust", "full", "identity", "affine", "simple"],
        default=None,
    )
    p.add_argument("--inverse_full_mapping_penalty", type=float, default=None)
    p.add_argument("--inverse_exact_simple_target_bonus", type=float, default=None)
    p.add_argument("--inverse_additive_descend_penalty", type=float, default=None)
    p.add_argument("--inverse_nonadditive_leaf_penalty", type=float, default=None)
    p.add_argument("--inverse_exact_path_eta", type=float, default=None)
    p.add_argument("--inverse_exact_transport_min_lin_rel", type=float, default=None)
    inverse_gate_g = p.add_mutually_exclusive_group()
    inverse_gate_g.add_argument(
        "--inverse_gate",
        dest="inverse_gate_enable",
        action="store_true",
        help="Enable inverse-steering gate",
    )
    inverse_gate_g.add_argument(
        "--no_inverse_gate",
        dest="inverse_gate_enable",
        action="store_false",
        help="Disable inverse-steering gate",
    )
    p.set_defaults(inverse_gate_enable=None)
    p.add_argument("--inverse_gate_warmup", type=int, default=None)
    p.add_argument("--inverse_gate_best_factor", type=float, default=None)
    p.add_argument("--inverse_gate_min_depth", type=int, default=None)
    p.add_argument("--inverse_gate_min_size", type=int, default=None)
    p.add_argument("--inverse_gate_max_paths", type=int, default=None)
    p.add_argument("--inverse_gate_min_structural_score", type=float, default=None)
    p.add_argument("--inverse_gate_min_weighted_rel_gain", type=float, default=None)
    p.add_argument("--inverse_gate_structural_bias", type=float, default=None)
    p.add_argument("--inverse_periodic_min_valid_scale", type=float, default=None)
    p.add_argument("--inverse_periodic_min_confidence_scale", type=float, default=None)
    p.add_argument("--inverse_periodic_path_penalty", type=float, default=None)
    p.add_argument("--inverse_nonperiodic_muldiv_bonus", type=float, default=None)
    p.add_argument("--inverse_nonperiodic_explogsqrt_bonus", type=float, default=None)
    p.add_argument("--inverse_branch_ambiguity_penalty", type=float, default=None)
    p.add_argument("--inverse_transport_min_lin_rel", type=float, default=None)
    p.add_argument("--inverse_transport_min_effective_n", type=float, default=None)
    inverse_exp_g = p.add_mutually_exclusive_group()
    inverse_exp_g.add_argument(
        "--inverse_experiment_log",
        dest="inverse_experiment_log_enable",
        action="store_true",
        help="Record structured diagnostics for every inverse-steering invocation",
    )
    inverse_exp_g.add_argument(
        "--no_inverse_experiment_log",
        dest="inverse_experiment_log_enable",
        action="store_false",
        help="Disable structured inverse-steering experiment logging",
    )
    p.set_defaults(inverse_experiment_log_enable=None)
    repair_ctl_g = p.add_mutually_exclusive_group()
    repair_ctl_g.add_argument(
        "--repair_controller",
        dest="repair_controller_enable",
        action="store_true",
        help="Use the analytic repair controller to trigger inverse repair conditionally",
    )
    repair_ctl_g.add_argument(
        "--no_repair_controller",
        dest="repair_controller_enable",
        action="store_false",
        help="Disable the analytic repair controller",
    )
    p.set_defaults(repair_controller_enable=None)
    p.add_argument("--repair_controller_min_score", type=float, default=None)
    p.add_argument("--repair_controller_steps", type=int, default=None)
    p.add_argument("--repair_controller_ancestor_hops", type=int, default=None)
    p.add_argument("--repair_controller_min_step_rel_improve", type=float, default=None)
    repair_ctl_adapt_g = p.add_mutually_exclusive_group()
    repair_ctl_adapt_g.add_argument(
        "--repair_controller_adaptive",
        dest="repair_controller_adaptive",
        action="store_true",
        help="Adapt the repair threshold from recent controller scores",
    )
    repair_ctl_adapt_g.add_argument(
        "--no_repair_controller_adaptive",
        dest="repair_controller_adaptive",
        action="store_false",
        help="Use only the fixed repair controller threshold",
    )
    p.set_defaults(repair_controller_adaptive=None)
    p.add_argument("--repair_controller_adapt_quantile", type=float, default=None)
    p.add_argument("--repair_controller_adapt_window", type=int, default=None)
    p.add_argument("--repair_controller_adapt_min_samples", type=int, default=None)
    p.add_argument("--repair_controller_min_concentration", type=float, default=None)
    p.add_argument("--repair_controller_potential_weight", type=float, default=None)
    p.add_argument("--repair_controller_concentration_weight", type=float, default=None)
    p.add_argument("--repair_controller_contrast_weight", type=float, default=None)
    p.add_argument("--repair_controller_cost_weight", type=float, default=None)
    p.add_argument("--repair_controller_stagnation_weight", type=float, default=None)
    p.add_argument("--repair_controller_frontier_topk", type=int, default=None)
    p.add_argument("--repair_controller_stagnation_visits", type=int, default=None)
    p.add_argument("--repair_controller_focus_prob", type=float, default=None)
    p.add_argument("--repair_controller_parent_max_repeats", type=int, default=None)
    p.add_argument("--repair_controller_parent_min_eval_gap", type=int, default=None)
    p.add_argument("--repair_controller_parent_reset_rel_improve", type=float, default=None)
    p.add_argument("--repair_controller_critic_enable", action="store_true", default=None)
    p.add_argument("--repair_controller_critic_path", type=str, default=None)
    p.add_argument("--repair_controller_critic_blend", type=float, default=None)
    p.add_argument(
        "--repair_controller_critic_mode",
        type=str,
        choices=["priority", "gate", "decisive"],
        default=None,
    )

    p.add_argument("--brute_depth", type=int, default=None)
    p.add_argument(
        "--early_stop_mse",
        type=float,
        default=None,
        help="Global solved threshold for brute-force and mutation phases",
    )
    p.add_argument("--no_brute_force", action="store_true")
    p.add_argument(
        "--fast_benchmark",
        action="store_true",
        help="Benchmark-only mode: disable brute for faster inner-loop oracle runs",
    )
    p.add_argument(
        "--score_mapping_family_mode",
        type=str,
        choices=["full", "gated", "cheap"],
        default=None,
        help="Outer scoring mapping-family mode for mutation/hole-search scoring",
    )
    p.add_argument(
        "--brute_score_mapping_family_mode",
        type=str,
        choices=["full", "gated", "cheap"],
        default=None,
        help="Outer scoring mapping-family mode used during the brute phase",
    )
    p.add_argument("--score_mapping_expensive_gate_best_factor", type=float, default=None)
    p.add_argument("--score_mapping_expensive_rel_y", type=float, default=None)
    prescreen_g = p.add_mutually_exclusive_group()
    prescreen_g.add_argument(
        "--score_prescreen",
        dest="score_prescreen_enable",
        action="store_true",
        help="Enable route-aware cheap prescreening for generic mutation candidates",
    )
    prescreen_g.add_argument(
        "--no_score_prescreen",
        dest="score_prescreen_enable",
        action="store_false",
        help="Disable route-aware cheap prescreening for generic mutation candidates",
    )
    p.set_defaults(score_prescreen_enable=None)
    p.add_argument(
        "--score_prescreen_family_mode",
        type=str,
        choices=["full", "gated", "cheap"],
        default=None,
        help="Mapping-family mode used for the cheap mutation prescreen",
    )
    p.add_argument(
        "--score_prescreen_residual_family_mode",
        type=str,
        choices=["full", "gated", "cheap"],
        default=None,
        help="Mapping-family mode used for residual-route prescreening",
    )
    residual_hint_g = p.add_mutually_exclusive_group()
    residual_hint_g.add_argument(
        "--score_prescreen_residual_allow_hint",
        dest="score_prescreen_residual_allow_hint",
        action="store_true",
        help="Allow residual prescreen to auto-promote on periodic/exp hints",
    )
    residual_hint_g.add_argument(
        "--no_score_prescreen_residual_allow_hint",
        dest="score_prescreen_residual_allow_hint",
        action="store_false",
        help="Disable hint-only promotion for residual prescreen",
    )
    p.set_defaults(score_prescreen_residual_allow_hint=None)
    residual_global_g = p.add_mutually_exclusive_group()
    residual_global_g.add_argument(
        "--score_prescreen_residual_use_global_best",
        dest="score_prescreen_residual_use_global_best",
        action="store_true",
        help="Allow residual prescreen to use the global-best competitiveness gate",
    )
    residual_global_g.add_argument(
        "--no_score_prescreen_residual_use_global_best",
        dest="score_prescreen_residual_use_global_best",
        action="store_false",
        help="Disable the global-best competitiveness gate for residual prescreen",
    )
    p.set_defaults(score_prescreen_residual_use_global_best=None)
    p.add_argument("--score_prescreen_parent_best_factor", type=float, default=None)
    p.add_argument("--score_prescreen_global_best_factor", type=float, default=None)
    p.add_argument("--score_prescreen_residual_parent_best_factor", type=float, default=None)
    p.add_argument("--score_prescreen_residual_global_best_factor", type=float, default=None)

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

    # Inverse-steering sandbox (kept orthogonal to the full factorized symbolic search run)
    p.add_argument(
        "--inverse_demo",
        action="store_true",
        help="Run the context-sensitive inverse-steering sandbox instead of the full explorer",
    )
    p.add_argument(
        "--inverse_path_sweep",
        action="store_true",
        help="Run the oracle path-sweep diagnostic for inverse steering instead of the full explorer",
    )
    p.add_argument(
        "--inverse_proposal_family_compare",
        action="store_true",
        help="Compare legacy vs direct-spec inverse proposal families at a chosen repair hole",
    )
    p.add_argument(
        "--candidate_expr",
        type=str,
        default=None,
        help="Candidate expression for inverse demo (accepts x0.. aliases or declared variable names)",
    )
    p.add_argument(
        "--corrupt_path",
        type=str,
        default=None,
        help="Path in the oracle truth AST to replace when constructing the inverse-demo candidate",
    )
    p.add_argument(
        "--replacement_expr",
        type=str,
        default=None,
        help="Replacement expression used together with --corrupt_path",
    )
    p.add_argument(
        "--inverse_path",
        type=str,
        default=None,
        help="Target subtree path for inverse steering, e.g. 1/2/1",
    )
    p.add_argument(
        "--inverse_topk",
        type=int,
        default=10,
        help="Number of replacement proposals to report for inverse_demo / inverse_path_sweep",
    )
    p.add_argument(
        "--inverse_sweep_modes",
        type=str,
        default="identity,full,affine",
        help="Comma-separated inverse target modes to compare in inverse_path_sweep",
    )
    p.add_argument(
        "--inverse_sweep_max_paths",
        type=int,
        default=None,
        help="Maximum number of paths to include in inverse_path_sweep",
    )
    p.add_argument(
        "--inverse_sweep_all_paths",
        action="store_true",
        help="Include all candidate-tree paths in inverse_path_sweep instead of only gate/reference paths",
    )

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    spec = load_equation_spec(args.spec)
    hp = _apply_cli_overrides(default_oracle_hyperparams(), args)

    dtype = torch.float64 if str(args.dtype).lower() == "float64" else torch.float32
    inverse_demo = bool(getattr(args, "inverse_demo", False))
    inverse_path_sweep = bool(getattr(args, "inverse_path_sweep", False))
    inverse_family_compare = bool(getattr(args, "inverse_proposal_family_compare", False))
    selected_inverse_modes = int(inverse_demo) + int(inverse_path_sweep) + int(inverse_family_compare)
    if selected_inverse_modes > 1:
        raise SystemExit(
            "Choose only one of --inverse_demo, --inverse_path_sweep, or --inverse_proposal_family_compare"
        )

    if inverse_demo:
        report = run_inverse_steering_lab(
            spec,
            factorized_search_hp=hp,
            seed=args.seed,
            dtype=dtype,
            enforce_dims=not bool(args.ignore_dims),
            candidate_expr=getattr(args, "candidate_expr", None),
            corrupt_path=getattr(args, "corrupt_path", None),
            replacement_expr=getattr(args, "replacement_expr", None),
            inverse_path=getattr(args, "inverse_path", None),
            topk=int(getattr(args, "inverse_topk", 10) or 10),
            verbose=not bool(args.quiet),
        )
        if args.output is not None:
            save_oracle_report(report, args.output)
            print(f"[inverse] report written to {args.output}")
        return 0

    if inverse_path_sweep:
        mode_text = str(getattr(args, "inverse_sweep_modes", "identity,full,affine") or "identity,full,affine")
        compare_modes = [tok.strip() for tok in mode_text.split(",") if tok.strip()]
        report = run_inverse_path_sweep_lab(
            spec,
            factorized_search_hp=hp,
            seed=args.seed,
            dtype=dtype,
            enforce_dims=not bool(args.ignore_dims),
            candidate_expr=getattr(args, "candidate_expr", None),
            corrupt_path=getattr(args, "corrupt_path", None),
            replacement_expr=getattr(args, "replacement_expr", None),
            inverse_path=getattr(args, "inverse_path", None),
            topk=int(getattr(args, "inverse_topk", 10) or 10),
            compare_modes=compare_modes,
            sweep_all_paths=bool(getattr(args, "inverse_sweep_all_paths", False)),
            sweep_max_paths=getattr(args, "inverse_sweep_max_paths", None),
            verbose=not bool(args.quiet),
        )
        if args.output is not None:
            save_oracle_report(report, args.output)
            print(f"[path-sweep] report written to {args.output}")
        return 0

    if inverse_family_compare:
        report = run_inverse_proposal_family_compare_lab(
            spec,
            factorized_search_hp=hp,
            seed=args.seed,
            dtype=dtype,
            enforce_dims=not bool(args.ignore_dims),
            candidate_expr=getattr(args, "candidate_expr", None),
            corrupt_path=getattr(args, "corrupt_path", None),
            replacement_expr=getattr(args, "replacement_expr", None),
            inverse_path=getattr(args, "inverse_path", None),
            verbose=not bool(args.quiet),
        )
        if args.output is not None:
            save_oracle_report(report, args.output)
            print(f"[inverse-compare] report written to {args.output}")
        return 0

    report = run_oracle_equation(
        spec,
        factorized_search_hp=hp,
        seed=args.seed,
        dtype=dtype,
        enforce_dims=not bool(args.ignore_dims),
        verbose=not bool(args.quiet),
        gs_carrier_seed=bool(getattr(args, "gs_carrier_seed", False)),
    )

    best = report.get("best")
    if best is None:
        print(f"[oracle] {spec.id}: no candidate found")
    else:
        print(
            f"[oracle] {spec.id}: best_mse={float(best['mse']):.6g} "
            f"expr={best['expr']} mapping={best.get('mapping_kind', '')}"
        )

    if args.output is not None:
        save_oracle_report(report, args.output)
        print(f"[oracle] report written to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
