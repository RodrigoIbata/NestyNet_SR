# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import torch

from ..basis_state import ProposalContext
from ..basis_scoring import direct_power_variant_nparams, score_bound_closure
from ..expr_ast import (
    dims_eq,
    eval_node,
    is_valid_node,
    node_depth,
    node_dims,
    node_size,
    node_str,
    simplify,
)
from ..expr_enum import enumerate_trees, enumerate_trees_dim
from .common import (
    deadline_exceeded,
    dim0,
    dim_scale,
    dim_sub,
    node_var_count,
    strip_log_scalar_factors,
)
from .closure_eval import (
    finalize_direct_preview_rows,
    make_direct_preview_row,
    scaffold_parent_stats,
)
from .closure_builders import (
    build_affine_latent_candidate,
    build_affine_power_candidate,
    build_linear_wrap_candidate,
    build_multi_term_rational_candidate,
    build_quadratic_sqrt_candidate,
    build_rational_affine_candidate,
)
from .binding_search import (
    collect_shortlisted_hole_candidates,
    dedup_seed_blocks,
    filter_seed_blocks_for_dim,
    pin_dimensionless_ratio_square,
    pin_ratio_square,
    pin_small_trig_carrier,
    pin_single_var_square,
    pin_wrapped_rational_term,
    quadratic_base_priority,
)
from .closure_runners import (
    CustomDirectSearchPlan,
    PairCollectedSearchPlan,
    PreparedCandidatesSearchPlan,
    PreparedClosureCandidate,
    SeedSubsetSearchPlan,
    SingleHoleCollectedSearchPlan,
    execute_direct_search_plan,
)
from .periodic_search import (
    build_exact_bound_periodic_search_plan,
    build_periodic_search_plan,
    direct_periodic_scaffold_kind,
    solve_direct_periodic_add_preview_rows,
)
from .seed_blocks import make_seed_block, seed_anchor_blocks
from .types import OperatorApplication


_DIRECT_EXACT_BOUND_KEEP_MSE = 1.0e-10


_MULTI_TERM_RATIONAL_SUPPORT_CACHE: OrderedDict[tuple[Any, ...], tuple[list[dict[str, Any]], int]] = OrderedDict()


def _mt_env_name(key: str) -> str:
    return "NESTY_" + key.upper()


def _mt_raw_kw(solver_kwargs: Mapping[str, Any] | None, key: str, env_name: str | None = None) -> Any:
    if solver_kwargs and key in solver_kwargs:
        return solver_kwargs[key]
    return os.environ.get(env_name or _mt_env_name(key))


def _mt_kw_bool(solver_kwargs: Mapping[str, Any] | None, key: str, default: bool, env_name: str | None = None) -> bool:
    raw = _mt_raw_kw(solver_kwargs, key, env_name)
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    token = str(raw).strip().lower()
    if token in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _mt_kw_int(solver_kwargs: Mapping[str, Any] | None, key: str, default: int, env_name: str | None = None) -> int:
    raw = _mt_raw_kw(solver_kwargs, key, env_name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _mt_kw_float(
    solver_kwargs: Mapping[str, Any] | None,
    key: str,
    default: float,
    env_name: str | None = None,
) -> float:
    raw = _mt_raw_kw(solver_kwargs, key, env_name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _mt_kw_str(solver_kwargs: Mapping[str, Any] | None, key: str, default: str, env_name: str | None = None) -> str:
    raw = _mt_raw_kw(solver_kwargs, key, env_name)
    if raw is None:
        return str(default)
    return str(raw)


def _mt_node_key(node: Any) -> str:
    try:
        return str(node_str(node))
    except Exception:
        return repr(node)


def _ranked_combo_list(
    indices: Sequence[int],
    arity: int,
    score_by_idx: Mapping[int, float],
    max_count: int = 0,
) -> list[tuple[int, ...]]:
    if arity <= 0:
        return [()]
    idx_list = [int(idx) for idx in indices]
    if arity > len(idx_list):
        return []
    if max_count <= 0:
        return [tuple(int(v) for v in combo) for combo in combinations(idx_list, int(arity))]
    scored: list[tuple[float, tuple[int, ...]]] = []
    for combo in combinations(idx_list, int(arity)):
        tup = tuple(int(v) for v in combo)
        score = sum(float(score_by_idx.get(v, 1.0e300)) for v in tup)
        scored.append((score, tup))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [combo for _, combo in scored[: max(1, int(max_count))]]


def _mt_support_cache_get(key: tuple[Any, ...]) -> tuple[list[dict[str, Any]], int] | None:
    cached = _MULTI_TERM_RATIONAL_SUPPORT_CACHE.get(key)
    if cached is None:
        return None
    _MULTI_TERM_RATIONAL_SUPPORT_CACHE.move_to_end(key)
    best_results, budget_used = cached
    rows: list[dict[str, Any]] = []
    for result in best_results:
        row = dict(result)
        if "coeffs" in row:
            row["coeffs"] = list(row["coeffs"])
        rows.append(row)
    return rows, int(budget_used)


def _mt_support_cache_put(
    key: tuple[Any, ...],
    best_results: Sequence[dict[str, Any]],
    budget_used: int,
    max_entries: int,
) -> None:
    if max_entries <= 0:
        return
    rows: list[dict[str, Any]] = []
    for result in best_results:
        row = dict(result)
        if "coeffs" in row:
            row["coeffs"] = list(row["coeffs"])
        rows.append(row)
    _MULTI_TERM_RATIONAL_SUPPORT_CACHE[key] = (rows, int(budget_used))
    _MULTI_TERM_RATIONAL_SUPPORT_CACHE.move_to_end(key)
    while len(_MULTI_TERM_RATIONAL_SUPPORT_CACHE) > int(max_entries):
        _MULTI_TERM_RATIONAL_SUPPORT_CACHE.popitem(last=False)


def _quadratic_ratio_or_inverse_seed(block: Any) -> bool:
    node = getattr(block, "node", None)
    return isinstance(node, tuple) and len(node) >= 3 and str(node[0]) == "div"


def _quadratic_affine_difference_seed(block: Any) -> bool:
    node = getattr(block, "node", None)
    meta = dict(getattr(block, "metadata", {}) or {})
    return (
        isinstance(node, tuple)
        and len(node) >= 3
        and str(node[0]) == "sub"
        and bool(meta.get("quadratic_affine_seed", False))
    )


def _quadratic_structured_seed(block: Any) -> bool:
    return _quadratic_ratio_or_inverse_seed(block) or _quadratic_affine_difference_seed(block)


def _quadratic_affine_difference_priority(block: Any) -> tuple[int, int, int, str]:
    node = getattr(block, "node", None)
    lhs = node[1] if isinstance(node, tuple) and len(node) >= 2 else None
    rhs = node[2] if isinstance(node, tuple) and len(node) >= 3 else None
    lhs_is_var = isinstance(lhs, tuple) and lhs and str(lhs[0]) == "var"
    rhs_is_var = isinstance(rhs, tuple) and rhs and str(rhs[0]) == "var"
    try:
        size = max(1, int(node_size(node)))
    except Exception:
        size = 99
    try:
        uniq_vars = max(0, int(node_var_count(node)))
    except Exception:
        uniq_vars = 0
    return (0 if lhs_is_var and rhs_is_var else 1, size, -uniq_vars, str(node_str(node)))


def _quadratic_affine_difference_support(block: Any) -> frozenset[int]:
    raw = getattr(block, "active_vars", ())
    out: list[int] = []
    for value in list(raw or ()):
        try:
            out.append(int(value))
        except Exception:
            continue
    return frozenset(out)


def _seed_block_dim_or_none(block: Any, var_dims) -> Any:
    local_dim = getattr(block, "dim", None)
    if local_dim is not None or var_dims is None:
        return local_dim
    node = getattr(block, "node", None)
    if not (isinstance(node, tuple) and node):
        return None
    try:
        return node_dims(node, var_dims)
    except Exception:
        return None


def _build_quadratic_affine_difference_blocks(
    blocks: Sequence[Any],
    *,
    var_dims,
    limit: int,
) -> list[Any]:
    limit_i = max(0, int(limit))
    if limit_i <= 0:
        return []
    candidates: list[Any] = []
    seen_nodes: set[str] = set()
    for block in list(blocks or ()):
        node = getattr(block, "node", None)
        if not (isinstance(node, tuple) and node):
            continue
        if str(node[0]) == "const":
            continue
        if max(0, node_var_count(node)) > 1:
            continue
        dim = _seed_block_dim_or_none(block, var_dims)
        candidates.append((block, dim))

    out: list[Any] = []
    for idx, (left_block, left_dim) in enumerate(candidates):
        for right_block, right_dim in candidates[idx + 1:]:
            if left_dim is not None and right_dim is not None and not dims_eq(left_dim, right_dim):
                continue
            left_node = getattr(left_block, "node", None)
            right_node = getattr(right_block, "node", None)
            if left_node == right_node:
                continue
            diff_node = simplify(("sub", right_node, left_node))
            key = str(node_str(diff_node))
            if key in seen_nodes or not is_valid_node(diff_node):
                continue
            seen_nodes.add(key)
            out.append(
                make_seed_block(
                    diff_node,
                    dim=right_dim if right_dim is not None else left_dim,
                    source="quadratic_affine_diff",
                    builder="affine",
                    metadata={
                        "quadratic_affine_seed": True,
                        "term_nodes": [right_node, left_node],
                        "builder_depth": 1,
                    },
                )
            )
            if len(out) >= limit_i:
                return out
    return out


def _prefer_quadratic_structured_bases(quadratic_kind: str) -> bool:
    kind = str(quadratic_kind or "").strip().lower()
    return kind in {"sqrt", "sqrt_mul"}


def scaffold_form(spec: OperatorApplication) -> str:
    return str(dict(spec.metadata or {}).get("form", "") or "").strip().lower()


def _bound_closure_metadata(spec: Any) -> dict[str, Any]:
    bound = getattr(spec, "bound_closure", None)
    if hasattr(bound, "metadata"):
        return dict(getattr(bound, "metadata", {}) or {})
    return {}


def _bound_head_solver(spec: Any) -> str:
    bound = getattr(spec, "bound_closure", None)
    solver = getattr(getattr(bound, "spec", None), "head_solver", "")
    return str(solver or "").strip().lower()


def _bound_family(spec: Any) -> str:
    bound = getattr(spec, "bound_closure", None)
    family = getattr(getattr(bound, "spec", None), "family", "")
    return str(family or "").strip().lower()


def _spec_metadata(spec: Any) -> dict[str, Any]:
    return dict(getattr(spec, "metadata", {}) or {})


def _operator_id_token(spec: Any) -> str:
    return str(getattr(spec, "operator_id", "") or "").strip().lower()


def _spec_family(spec: Any) -> str:
    family = _family_token(spec)
    return family or _bound_family(spec)


def _best_preview_rows_mse(rows: Sequence[Mapping[str, Any]] | None) -> float:
    best = float("inf")
    for row in list(rows or ()):
        if not isinstance(row, Mapping):
            continue
        for key in ("local_probe_mse", "local_fit_mse"):
            try:
                value = float(row.get(key, float("inf")))
            except Exception:
                value = float("inf")
            if not math.isfinite(value):
                value = float("inf")
            if value < best:
                best = value
    return float(best)


def _robust_target_scale_sq(y_ref: torch.Tensor | None) -> float:
    if not torch.is_tensor(y_ref):
        return 1.0
    try:
        y_1d = y_ref.squeeze(-1).reshape(-1)
    except Exception:
        return 1.0
    if int(y_1d.numel()) <= 0 or not torch.isfinite(y_1d).all():
        return 1.0
    try:
        median = torch.median(y_1d)
        mad = float(torch.median(torch.abs(y_1d - median)).item())
    except Exception:
        mad = 0.0
    scale = float(mad)
    if (not math.isfinite(scale)) or scale <= 0.0:
        try:
            scale = float(torch.sqrt(torch.mean(y_1d * y_1d)).item())
        except Exception:
            scale = 0.0
    if (not math.isfinite(scale)) or scale <= 0.0:
        try:
            scale = float(torch.max(torch.abs(y_1d)).item())
        except Exception:
            scale = 0.0
    if (not math.isfinite(scale)) or scale <= 0.0:
        return 1.0
    return max(1.0e-30, float(scale * scale))


def _normalized_preview_rows_mse(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    y_ref: torch.Tensor | None,
) -> float:
    best_mse = _best_preview_rows_mse(rows)
    if not math.isfinite(best_mse):
        return float("inf")
    return float(best_mse) / float(_robust_target_scale_sq(y_ref))


def _screen_vector_profile(values: torch.Tensor | None) -> torch.Tensor | None:
    if not torch.is_tensor(values):
        return None
    try:
        x = values.squeeze(-1).reshape(-1)
    except Exception:
        return None
    if int(x.numel()) <= 0 or not torch.isfinite(x).all():
        return None
    centered = x - torch.mean(x)
    try:
        norm = float(torch.linalg.vector_norm(centered).item())
        ref = float(torch.linalg.vector_norm(x).item())
    except Exception:
        return None
    if (not math.isfinite(norm)) or (not math.isfinite(ref)):
        return None
    # Constant-like atoms are not useful here: numerator already has an intercept,
    # and denominator is gauge-fixed to start at 1.
    if norm <= 1.0e-8 * max(ref, math.sqrt(max(1, int(x.numel()))), 1.0):
        return None
    return centered / max(norm, 1.0e-12)


def _append_unique_index(target: list[int], idx: int, *, limit: int | None = None) -> bool:
    if idx in target:
        return False
    if limit is not None and len(target) >= int(limit):
        return False
    target.append(int(idx))
    return True


def _select_diverse_screen_indices(
    *,
    scored: Sequence[tuple[float, int]],
    evaluated: Sequence[Mapping[str, Any]],
    max_count: int,
    corr_threshold: float,
    value_key: str = "fit",
    drop_constant_like: bool = True,
) -> list[int]:
    chosen: list[int] = []
    kept_profiles: list[torch.Tensor | None] = []
    kept_constant_like = False
    for _score, idx in list(scored or ()):
        if idx < 0 or idx >= len(list(evaluated or ())):
            continue
        profile = _screen_vector_profile(evaluated[idx].get(str(value_key), None))
        if profile is None:
            if bool(drop_constant_like):
                continue
            if kept_constant_like:
                continue
            kept_constant_like = True
        else:
            duplicate = False
            for kept in list(kept_profiles or ()):
                if kept is None:
                    continue
                try:
                    corr = abs(float(torch.dot(profile, kept).item()))
                except Exception:
                    corr = 0.0
                if corr >= float(corr_threshold):
                    duplicate = True
                    break
            if duplicate:
                continue
        chosen.append(int(idx))
        kept_profiles.append(profile)
        if len(chosen) >= max(1, int(max_count)):
            break
    return chosen


def _mt_dim_matches(actual_dim: Any, desired_dim: Any) -> bool:
    if desired_dim is None or actual_dim is None:
        return True
    try:
        return bool(dims_eq(actual_dim, desired_dim))
    except Exception:
        return False


def _mt_node_dim(node: Any, var_dims) -> Any:
    if var_dims is None or not (isinstance(node, tuple) and is_valid_node(node)):
        return None
    try:
        return node_dims(node, var_dims)
    except Exception:
        return None


def _mt_mul_terms(node: Any) -> list[tuple]:
    if not (isinstance(node, tuple) and is_valid_node(node)):
        return []
    if str(node[0]) != "mul":
        return [node]
    out: list[tuple] = []
    for child in tuple(node[1:]):
        out.extend(_mt_mul_terms(child))
    return out


def _mt_additive_terms(node: Any) -> list[tuple]:
    if not (isinstance(node, tuple) and is_valid_node(node)):
        return []
    op = str(node[0])
    if op == "add":
        out: list[tuple] = []
        for child in tuple(node[1:]):
            out.extend(_mt_additive_terms(child))
        return out
    if op == "sub" and len(node) >= 3:
        out = _mt_additive_terms(node[1])
        for child in _mt_additive_terms(node[2]):
            try:
                neg_child = simplify(("mul", ("const", -1.0), child))
            except Exception:
                continue
            if isinstance(neg_child, tuple) and is_valid_node(neg_child):
                out.append(neg_child)
        return out
    return [node]


def _mt_build_mul(terms: Sequence[tuple]) -> tuple | None:
    valid = [term for term in list(terms or ()) if isinstance(term, tuple) and is_valid_node(term)]
    if not valid:
        return None
    node = valid[0]
    for term in valid[1:]:
        node = ("mul", node, term)
    try:
        node = simplify(node)
    except Exception:
        return None
    return node if isinstance(node, tuple) and is_valid_node(node) else None


def _mt_factor_role_shadows(
    node: Any,
    *,
    var_dims,
    target_dim,
    denom_dim,
    max_carrier_size: int,
) -> list[tuple[tuple, tuple]]:
    """Return `(target_prefactor, dimensionless_carrier)` product shadows."""
    if var_dims is None or target_dim is None or denom_dim is None:
        return []
    if not (isinstance(node, tuple) and is_valid_node(node)):
        return []
    terms = _mt_mul_terms(node)
    if len(terms) < 2:
        return []
    out: list[tuple[tuple, tuple]] = []
    seen: set[tuple[str, str]] = set()
    for carrier_pos, carrier_raw in enumerate(terms):
        try:
            carrier = simplify(carrier_raw)
        except Exception:
            continue
        if not (isinstance(carrier, tuple) and is_valid_node(carrier)):
            continue
        if str(carrier[0]) == "const":
            continue
        if int(node_size(carrier)) > int(max_carrier_size):
            continue
        carrier_dim = _mt_node_dim(carrier, var_dims)
        if not _mt_dim_matches(carrier_dim, denom_dim):
            continue
        prefactor = _mt_build_mul(
            [term for idx, term in enumerate(terms) if idx != carrier_pos]
        )
        if prefactor is None:
            continue
        prefactor_dim = _mt_node_dim(prefactor, var_dims)
        if not _mt_dim_matches(prefactor_dim, target_dim):
            continue
        key = (str(node_str(prefactor)), str(node_str(carrier)))
        if key in seen:
            continue
        seen.add(key)
        out.append((prefactor, carrier))
    return out


def _is_structural_rational_den_seed(node: tuple) -> bool:
    if not (isinstance(node, tuple) and node):
        return False
    op = str(node[0])
    if op in {"mul", "div"}:
        try:
            return max(0, node_var_count(node)) >= 2
        except Exception:
            return True
    if pin_ratio_square(node):
        return True
    if op == "sqrt" and len(node) >= 2:
        inner = node[1]
        if isinstance(inner, tuple) and inner and str(inner[0]) == "div":
            try:
                return max(0, node_var_count(node)) >= 2
            except Exception:
                return True
    return False


def _is_low_cost_rational_den_seed(node: tuple, *, max_size: int) -> bool:
    if not (isinstance(node, tuple) and node):
        return False
    try:
        if int(node_size(node)) > int(max_size):
            return False
    except Exception:
        return False
    op = str(node[0])
    if op in {"var", "sin", "cos", "exp", "log", "sqrt", "sqr"}:
        return True
    return _is_structural_rational_den_seed(node)


def _role_synthetic_carrier_structural_priority(node: tuple) -> tuple[int, int, int, int, str]:
    op = str(node[0]) if isinstance(node, tuple) and node else ""
    try:
        uniq_vars = max(0, int(node_var_count(node)))
    except Exception:
        uniq_vars = 0
    try:
        size = int(node_size(node))
    except Exception:
        size = 999
    try:
        depth = int(node_depth(node))
    except Exception:
        depth = 999
    # Product completion is most useful when a compact transformed carrier was
    # discovered separately from its target-dimensional prefactor. Prefer small
    # unary carriers over generic structural products, then let data ranking
    # decide which synthesized products survive.
    if op in {"sin", "cos", "exp", "log", "sqrt", "sqr"} and uniq_vars >= 2:
        group = 0
    elif op in {"sin", "cos", "exp", "log", "sqrt", "sqr"}:
        group = 1
    elif op == "var":
        group = 2
    else:
        group = 3
    return (group, -uniq_vars, size, depth, str(node_str(node)))


def _rational_den_seed_priority(node: tuple) -> tuple[int, int, int, int, str]:
    op = str(node[0]) if isinstance(node, tuple) and node else ""
    lhs = node[1] if isinstance(node, tuple) and len(node) >= 2 else None
    rhs = node[2] if isinstance(node, tuple) and len(node) >= 3 else None
    lhs_op = str(lhs[0]) if isinstance(lhs, tuple) and lhs else ""
    rhs_op = str(rhs[0]) if isinstance(rhs, tuple) and rhs else ""
    try:
        uniq_vars = max(0, int(node_var_count(node)))
    except Exception:
        uniq_vars = 0
    try:
        size = int(node_size(node))
    except Exception:
        size = 999
    try:
        depth = int(node_depth(node))
    except Exception:
        depth = 999
    if op == "div" and ({lhs_op, rhs_op} & {"mul", "sqr"}):
        group = 0
    elif op == "mul":
        group = 0
    elif op == "div" and ({lhs_op, rhs_op} & {"sqrt"}):
        group = 1
    elif op == "sqr":
        group = 1
    elif op == "div" and ({lhs_op, rhs_op} & {"add", "sub"}):
        group = 3
    elif op == "div":
        group = 2
    elif op == "sqrt":
        group = 2
    else:
        group = 4
    return (group, -uniq_vars, size, depth, str(node_str(node)))


def _power_or_rational_direct_spec(spec: Any) -> bool:
    planner_id = _planner_id_from_bound_closure(spec)
    if planner_id in {"power_wrap", "fractional_head"}:
        return True
    return _spec_family(spec) in {"power", "rational"}


def _periodic_direct_spec(spec: Any) -> bool:
    planner_id = _planner_id_from_bound_closure(spec)
    if planner_id == "harmonic_wrap":
        return True
    return _spec_family(spec) == "periodic"


def _should_force_slot_rebinding(
    spec: Any,
    *,
    exact_rows: Sequence[Mapping[str, Any]] | None,
    solver_kwargs: Mapping[str, Any] | None,
) -> bool:
    if not (_power_or_rational_direct_spec(spec) or _periodic_direct_spec(spec)):
        return False
    keep_threshold = float(dict(solver_kwargs or {}).get("direct_exact_bound_keep_mse", _DIRECT_EXACT_BOUND_KEEP_MSE))
    return _best_preview_rows_mse(exact_rows) > float(keep_threshold)


def _direct_linear_wrap_scaffold_kind(
    spec: OperatorApplication,
    *,
    algebra: "LinearWrapOperatorAlgebraSpec",
) -> str | None:
    family = str(algebra.family or "").strip().lower()
    if _spec_family(spec) not in {"", family} and _bound_family(spec) != family:
        if _bound_head_solver(spec) != "linear":
            return None
    form = scaffold_form(spec)
    if form == f"{family}_add":
        return "add"
    if bool(algebra.supports_mul) and form == f"{family}_mul":
        return "mul"
    if str(getattr(spec, "scaffold_id", "") or "").strip().lower() == f"{family}:base":
        return "base"
    meta = {**_spec_metadata(spec), **_bound_closure_metadata(spec)}
    kind = str(meta.get(f"{family}_kind", "") or "").strip().lower()
    supported = {"base", "add"}
    if bool(algebra.supports_mul):
        supported.add("mul")
    if kind in supported:
        return kind
    operator_id = _operator_id_token(spec)
    if operator_id.endswith(":add") or operator_id.endswith("_add"):
        return "add"
    if bool(algebra.supports_mul) and (operator_id.endswith(":mul") or operator_id.endswith("_mul")):
        return "mul"
    wrap_token = str(meta.get("wrap_op", "") or "").strip().lower()
    if _bound_head_solver(spec) == "linear" and _spec_family(spec) == family and wrap_token in {"", str(algebra.wrap_op)}:
        return "base"
    return None


def _linear_wrap_algebra_by_family(family: str) -> "LinearWrapOperatorAlgebraSpec" | None:
    token = str(family or "").strip().lower()
    for algebra in LINEAR_WRAP_OPERATOR_ALGEBRAS:
        if str(algebra.family or "").strip().lower() == token:
            return algebra
    return None


def direct_exp_scaffold_kind(spec: OperatorApplication) -> str | None:
    algebra = _linear_wrap_algebra_by_family("exp")
    if algebra is None:
        return None
    return _direct_linear_wrap_scaffold_kind(spec, algebra=algebra)


def direct_affine_scaffold_kind(spec: OperatorApplication) -> str | None:
    if _spec_family(spec) not in {"", "affine"} and _bound_family(spec) != "affine":
        return None
    form = scaffold_form(spec)
    if form == "affine_latent":
        return "latent"
    meta = {**_spec_metadata(spec), **_bound_closure_metadata(spec)}
    if str(meta.get("form", "") or "").strip().lower() == "affine_latent":
        return "latent"
    if _bound_head_solver(spec) == "linear" and _spec_family(spec) == "affine":
        return "latent"
    if dict(getattr(getattr(spec, "bound_closure", None), "bindings", {}) or {}).get("terms", None):
        return "latent"
    return None


def direct_log_scaffold_kind(spec: OperatorApplication) -> str | None:
    algebra = _linear_wrap_algebra_by_family("log")
    if algebra is None:
        return None
    return _direct_linear_wrap_scaffold_kind(spec, algebra=algebra)


def direct_rational_scaffold_kind(spec: OperatorApplication) -> str | None:
    if _spec_family(spec) not in {"", "rational"} and _bound_family(spec) != "rational":
        if _bound_head_solver(spec) != "fractional_linear":
            return None
    form = scaffold_form(spec)
    if form == "rational_affine":
        return "affine"
    meta = {**_spec_metadata(spec), **_bound_closure_metadata(spec)}
    if str(meta.get("form", "") or "").strip().lower() == "rational_affine":
        return "affine"
    if _bound_head_solver(spec) == "fractional_linear":
        return "affine"
    return None


def direct_power_scaffold_kind(spec: OperatorApplication) -> str | None:
    if _spec_family(spec) not in {"", "power"} and _bound_family(spec) != "power":
        if _bound_head_solver(spec) != "discrete_power":
            return None
    form = scaffold_form(spec)
    power_forms = {
        "power_sqrt": "sqrt",
        "power_sqrt_mul": "sqrt_mul",
        "power_invsqrt": "invsqrt",
        "power_invsqrt_mul": "invsqrt_mul",
        "power_inv": "inv",
        "power_inv_mul": "inv_mul",
        "power_neg2": "neg2",
        "power_neg2_mul": "neg2_mul",
        "power_sqr": "sqr",
        "power_sqr_mul": "sqr_mul",
    }
    if form in power_forms:
        return power_forms.get(form, None)
    meta = {**_spec_metadata(spec), **_bound_closure_metadata(spec)}
    kind = str(meta.get("power_kind", "") or "").strip().lower()
    if kind in {
        "sqrt",
        "sqrt_mul",
        "invsqrt",
        "invsqrt_mul",
        "inv",
        "inv_mul",
        "neg2",
        "neg2_mul",
        "sqr",
        "sqr_mul",
    }:
        return kind
    return None


def direct_quadratic_scaffold_kind(spec: OperatorApplication) -> str | None:
    if _spec_family(spec) not in {"", "quadratic"} and _bound_family(spec) != "quadratic":
        if _bound_head_solver(spec) != "quadratic_sqrt":
            return None
    form = scaffold_form(spec)
    if form == "quadratic_sqrt":
        return "sqrt"
    if form == "quadratic_sqrt_mul":
        return "sqrt_mul"
    meta = {**_spec_metadata(spec), **_bound_closure_metadata(spec)}
    kind = str(meta.get("quadratic_kind", "") or "").strip().lower()
    if kind in {"sqrt", "sqrt_mul"}:
        return kind
    if _bound_head_solver(spec) == "quadratic_sqrt":
        return "sqrt"
    return None


@dataclass(frozen=True)
class DirectOperatorPlanner:
    planner_id: str
    matcher: Callable[[Any], bool]
    plan_builder: Callable[..., Any]
    operator_kinds: tuple[str, ...] = ()
    composition_modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinearWrapOperatorAlgebraSpec:
    planner_id: str
    family: str
    wrap_op: str
    hole_transform_fn: Callable[[tuple, torch.Tensor], tuple]
    supports_mul: bool = True


def _operator_kind_token(spec: Any) -> str:
    direct = str(getattr(spec, "operator_kind", "") or "").strip().lower()
    if direct:
        return direct
    meta = {**_bound_closure_metadata(spec), **_spec_metadata(spec)}
    return str(meta.get("operator_kind", "") or "").strip().lower()


def _composition_mode_token(spec: Any) -> str:
    meta = {**_bound_closure_metadata(spec), **_spec_metadata(spec)}
    return str(meta.get("composition_mode", "") or "").strip().lower()


def _family_token(spec: Any) -> str:
    return str(getattr(spec, "family", "") or "").strip().lower()


def collect_direct_hole_candidates(
    *,
    nvars: int,
    enum_max_depth: int,
    enum_max_trees: int,
    var_dims,
    target_dim,
    pool_nodes,
    pool_dims,
    deadline_s: float | None = None,
) -> tuple[list[tuple[str, tuple]], dict[str, Any]]:
    started = time.perf_counter()
    rows: list[tuple[str, tuple]] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}
    enum_depth_reached = 0
    enum_tree_count = 0
    enum_s = 0.0
    enum_add_s = 0.0
    pool_add_s = 0.0

    def _add(node: Any, source: str) -> None:
        if not isinstance(node, tuple) or not node:
            return
        try:
            simp = simplify(node)
        except Exception:
            return
        if not is_valid_node(simp):
            return
        if var_dims is not None and target_dim is not None:
            try:
                nd = node_dims(simp, var_dims)
            except Exception:
                nd = None
            if nd is None or not dims_eq(nd, target_dim):
                return
        key = node_str(simp)
        if key in seen:
            return
        seen.add(key)
        rows.append((str(source), simp))
        source_counts[str(source)] = int(source_counts.get(str(source), 0) or 0) + 1

    try:
        if deadline_exceeded(deadline_s):
            return rows, {
                "candidate_source_counts": dict(source_counts),
                "enum_tree_count": 0,
                "enum_depth_reached": int(enum_depth_reached),
                "deadline_exceeded": True,
                "candidate_count_collected": 0,
                "timing_collect_total_s": float(time.perf_counter() - started),
            }
        enum_started = time.perf_counter()
        if var_dims is not None and target_dim is not None:
            enum_nodes, enum_depth_reached = enumerate_trees_dim(
                max(1, int(enum_max_depth)),
                int(nvars),
                var_dims,
                target_dim,
                max_trees=max(1, int(enum_max_trees)),
            )
        else:
            enum_nodes, enum_depth_reached = enumerate_trees(
                max(1, int(enum_max_depth)),
                int(nvars),
                max_trees=max(1, int(enum_max_trees)),
            )
        enum_s = float(time.perf_counter() - enum_started)
    except Exception:
        enum_s = float(time.perf_counter() - enum_started) if "enum_started" in locals() else 0.0
        enum_nodes, enum_depth_reached = [], 0
    enum_tree_count = int(len(list(enum_nodes or ())))
    enum_add_started = time.perf_counter()
    for node in list(enum_nodes or ()):
        if deadline_exceeded(deadline_s):
            break
        _add(node, "enum")
    enum_add_s = float(time.perf_counter() - enum_add_started)

    pool_nodes_list = list(pool_nodes or ())
    pool_dims_list = list(pool_dims or ())
    pool_add_started = time.perf_counter()
    for idx, node in enumerate(pool_nodes_list):
        if deadline_exceeded(deadline_s):
            break
        if var_dims is not None and target_dim is not None and idx < len(pool_dims_list):
            nd = pool_dims_list[idx]
            if nd is None or not dims_eq(nd, target_dim):
                continue
        _add(node, "pool")
    pool_add_s = float(time.perf_counter() - pool_add_started)

    return rows, {
        "candidate_source_counts": dict(source_counts),
        "enum_tree_count": int(enum_tree_count),
        "enum_depth_reached": int(enum_depth_reached),
        "deadline_exceeded": bool(deadline_exceeded(deadline_s)),
        "candidate_count_collected": int(len(rows)),
        "timing_enum_s": float(enum_s),
        "timing_enum_add_s": float(enum_add_s),
        "timing_pool_add_s": float(pool_add_s),
        "timing_collect_total_s": float(time.perf_counter() - started),
    }


def _identity_hole_transform(hole_node: tuple, _x_fit: torch.Tensor) -> tuple:
    return hole_node


def _run_static_direct_status(
    *,
    status: str,
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    return [], str(status), dict(meta or {})


def _static_status_plan(status: str, meta: Mapping[str, Any] | None = None) -> CustomDirectSearchPlan:
    return CustomDirectSearchPlan(
        run_fn=_run_static_direct_status,
        kwargs={"status": str(status), "meta": dict(meta or {})},
    )


LINEAR_WRAP_OPERATOR_ALGEBRAS: tuple[LinearWrapOperatorAlgebraSpec, ...] = (
    LinearWrapOperatorAlgebraSpec(
        planner_id="exp",
        family="exp",
        wrap_op="exp",
        hole_transform_fn=_identity_hole_transform,
        supports_mul=True,
    ),
    LinearWrapOperatorAlgebraSpec(
        planner_id="log",
        family="log",
        wrap_op="log",
        hole_transform_fn=strip_log_scalar_factors,
        supports_mul=False,
    ),
)


def resolve_linear_wrap_operator_algebra(spec: Any) -> LinearWrapOperatorAlgebraSpec | None:
    for algebra in LINEAR_WRAP_OPERATOR_ALGEBRAS:
        try:
            if _direct_linear_wrap_scaffold_kind(spec, algebra=algebra) is not None:
                return algebra
        except Exception:
            continue
    return None


def _linear_wrap_planner_matcher(spec: Any) -> bool:
    operator_kind = _operator_kind_token(spec)
    if operator_kind in {"unary_wrap", "anchored_unary_wrap"}:
        return resolve_linear_wrap_operator_algebra(spec) is not None
    return resolve_linear_wrap_operator_algebra(spec) is not None


def build_direct_unary_linear_search_plan(
    spec: OperatorApplication,
    *,
    family: str,
    kind: str,
    wrap_op: str,
    hole_transform_fn: Callable[[tuple, torch.Tensor], tuple],
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn=collect_direct_hole_candidates,
) -> Any:
    family_token = str(family or "").strip().lower()
    kind_token = str(kind or "").strip().lower()
    wrap_token = str(wrap_op or "").strip().lower()
    if not family_token or not kind_token or not wrap_token:
        return _static_status_plan("direct_unary_unsupported_form")
    allow_slot_rebinding = bool(dict(solver_kwargs or {}).get("allow_slot_rebinding", False))
    if not allow_slot_rebinding:
        exact_plan = _build_exact_bound_unary_linear_plan(
            spec,
            family=family_token,
            kind=kind_token,
            wrap_op=wrap_token,
            x_fit=x_fit,
            x_probe=x_probe,
            var_dims=var_dims,
        )
        if exact_plan is not None:
            return exact_plan

    anchor_node = spec.anchor_node if kind_token in {"add", "mul"} else None
    if kind_token in {"add", "mul"} and (not isinstance(anchor_node, tuple) or not is_valid_node(anchor_node)):
        return _static_status_plan("direct_missing_anchor")

    target_dim = dim0(var_dims) if var_dims is not None else None
    anchor_fit = None
    anchor_probe = None
    if isinstance(anchor_node, tuple):
        try:
            anchor_fit = eval_node(anchor_node, x_fit)
            anchor_probe = eval_node(anchor_node, x_probe)
        except Exception:
            return _static_status_plan("direct_anchor_eval_failed")
        if (not torch.is_tensor(anchor_fit)) or (not torch.is_tensor(anchor_probe)):
            return _static_status_plan("direct_anchor_eval_failed")
        if (not torch.isfinite(anchor_fit).all()) or (not torch.isfinite(anchor_probe).all()):
            return _static_status_plan("direct_anchor_nonfinite")

    parent_stats = scaffold_parent_stats(spec)

    def _prepare_unary(source: str, hole_node: tuple) -> PreparedClosureCandidate | None:
        try:
            hole_node_eff = hole_transform_fn(hole_node, x_fit)
        except Exception:
            return None
        if not isinstance(hole_node_eff, tuple) or not is_valid_node(hole_node_eff):
            return None
        feature_node = (wrap_token, hole_node_eff)
        try:
            feature_fit = eval_node(feature_node, x_fit)
            feature_probe = eval_node(feature_node, x_probe)
        except Exception:
            return None
        if (not torch.is_tensor(feature_fit)) or (not torch.is_tensor(feature_probe)):
            return None
        if (not torch.isfinite(feature_fit).all()) or (not torch.isfinite(feature_probe).all()):
            return None

        design_fit_cols = [feature_fit.squeeze(-1)]
        design_probe_cols = [feature_probe.squeeze(-1)]
        if kind_token == "add":
            if anchor_fit is None or anchor_probe is None:
                return None
            design_fit_cols.append(anchor_fit.squeeze(-1))
            design_probe_cols.append(anchor_probe.squeeze(-1))
        elif kind_token == "mul":
            if anchor_fit is None or anchor_probe is None:
                return None
            design_fit_cols[0] = feature_fit.squeeze(-1) * anchor_fit.squeeze(-1)
            design_probe_cols[0] = feature_probe.squeeze(-1) * anchor_probe.squeeze(-1)
        design_fit_cols.append(torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device))
        design_probe_cols.append(torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device))

        if kind_token == "mul":
            feature_expr = simplify(("mul", feature_node, anchor_node))
            terms = [feature_expr]
            bias_index = 1
        elif kind_token == "add":
            feature_expr = feature_node
            terms = [feature_expr, anchor_node]
            bias_index = 2
        else:
            feature_expr = feature_node
            terms = [feature_expr]
            bias_index = 1

        built = build_linear_wrap_candidate(
            family=family_token,
            scaffold_id=str(spec.scaffold_id),
            kind=kind_token,
            hole_node=hole_node_eff,
            feature_node=feature_expr if kind_token == "mul" else feature_node,
            anchor_node=anchor_node if isinstance(anchor_node, tuple) else None,
            fit_matrix=torch.stack(design_fit_cols, dim=1),
            probe_matrix=torch.stack(design_probe_cols, dim=1),
            terms=terms,
            bias_index=int(bias_index),
            source=str(source),
        )
        return PreparedClosureCandidate(
            built=built,
            candidate_subtree_node=hole_node_eff,
        )

    plan = SingleHoleCollectedSearchPlan(
        nvars=int(nvars),
        enum_max_depth=int(solver_kwargs.get("enum_max_depth", max_depth)),
        enum_max_trees=int(solver_kwargs.get("enum_max_trees", 5000)),
        var_dims=var_dims,
        target_dim=target_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        shortlist_k=max(4, int(solver_kwargs.get("direct_periodic_feature_topk", 8) or 8)),
        pin_predicate=None,
        prepare_candidate_fn=_prepare_unary,
        parent_stats=parent_stats,
        meta=None,
    )
    return plan


# The direct family solvers are copied functionally from closure_search.py
# to make them independently importable by a future steering layer.
def build_direct_rational_affine_search_plan(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn=collect_direct_hole_candidates,
) -> Any:
    if direct_rational_scaffold_kind(spec) is None:
        return _static_status_plan("direct_rational_unsupported_form")

    target_dim = y_dims if y_dims is not None else dim0(var_dims)
    denom_dim = dim0(var_dims) if var_dims is not None else None
    safe_eps = float(solver_kwargs.get("safe_eps", 1.0e-6))
    allow_slot_rebinding = bool(dict(solver_kwargs or {}).get("allow_slot_rebinding", False))
    parent_stats = scaffold_parent_stats(spec)
    bound_u_node, bound_v_node = _rational_bound_nodes(spec)

    if not allow_slot_rebinding and bound_u_node is not None and bound_v_node is not None:
        exact_plan = _build_exact_bound_rational_plan(
            spec,
            x_fit=x_fit,
            x_probe=x_probe,
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            safe_eps=float(safe_eps),
        )
        if exact_plan is not None:
            return exact_plan

    def _pin_rational_left(node: tuple) -> bool:
        if pin_wrapped_rational_term(node):
            return True
        if not (isinstance(node, tuple) and node):
            return False
        if str(node[0]) not in {"mul", "div"}:
            return False
        try:
            return max(0, node_var_count(node)) >= 2
        except Exception:
            return True

    def _pin_rational_right(node: tuple) -> bool:
        return pin_wrapped_rational_term(node)

    def _build_completion_plan(
        *,
        fixed_role: str,
        fixed_node: tuple,
        search_role: str,
        search_target_dim: Any,
    ) -> Any:
        try:
            fixed_fit = eval_node(fixed_node, x_fit)
            fixed_probe = eval_node(fixed_node, x_probe)
        except Exception:
            return _static_status_plan("direct_rational_fixed_eval_failed")
        if (not torch.is_tensor(fixed_fit)) or (not torch.is_tensor(fixed_probe)):
            return _static_status_plan("direct_rational_fixed_eval_failed")
        if (not torch.isfinite(fixed_fit).all()) or (not torch.isfinite(fixed_probe).all()):
            return _static_status_plan("direct_rational_fixed_nonfinite")

        def _prepare_missing(source: str, search_node: tuple) -> PreparedClosureCandidate | None:
            try:
                search_fit = eval_node(search_node, x_fit)
                search_probe = eval_node(search_node, x_probe)
            except Exception:
                return None
            if (not torch.is_tensor(search_fit)) or (not torch.is_tensor(search_probe)):
                return None
            if (not torch.isfinite(search_fit).all()) or (not torch.isfinite(search_probe).all()):
                return None
            if str(fixed_role) == "numerator":
                u_node = fixed_node
                v_node = search_node
                u_fit = fixed_fit
                u_probe = fixed_probe
                v_fit = search_fit
                v_probe = search_probe
                u_source = "bound_numerator"
                v_source = str(source)
            else:
                u_node = search_node
                v_node = fixed_node
                u_fit = search_fit
                u_probe = search_probe
                v_fit = fixed_fit
                v_probe = fixed_probe
                u_source = str(source)
                v_source = "bound_denominator"
            built = build_rational_affine_candidate(
                scaffold_id=str(spec.scaffold_id),
                u_node=u_node,
                v_node=v_node,
                u_fit=u_fit,
                u_probe=u_probe,
                v_fit=v_fit,
                v_probe=v_probe,
                max_depth=int(max_depth),
                var_dims=var_dims,
                y_dims=y_dims,
                safe_eps=float(safe_eps),
                u_source=str(u_source),
                v_source=str(v_source),
            )
            return PreparedClosureCandidate(
                built=built,
                candidate_subtree_node=search_node,
            )

        return SingleHoleCollectedSearchPlan(
            nvars=int(nvars),
            enum_max_depth=int(solver_kwargs.get("enum_max_depth", max_depth)),
            enum_max_trees=int(solver_kwargs.get("enum_max_trees", 5000)),
            var_dims=var_dims,
            target_dim=search_target_dim,
            pool_nodes=pool_nodes,
            pool_dims=pool_dims,
            shortlist_k=max(4, int(solver_kwargs.get("direct_rational_feature_topk", 24) or 24)),
            pin_predicate=_pin_rational_right if str(search_role) == "denominator" else _pin_rational_left,
            prepare_candidate_fn=_prepare_missing,
            parent_stats=parent_stats,
            meta={
                "safe_eps": float(safe_eps),
                "completion_mode": f"fill_{search_role}",
                "preserved_slot": str(fixed_role),
            },
        )

    if not allow_slot_rebinding:
        if bound_u_node is not None and bound_v_node is None:
            return _build_completion_plan(
                fixed_role="numerator",
                fixed_node=bound_u_node,
                search_role="denominator",
                search_target_dim=denom_dim,
            )
        if bound_v_node is not None and bound_u_node is None:
            return _build_completion_plan(
                fixed_role="denominator",
                fixed_node=bound_v_node,
                search_role="numerator",
                search_target_dim=target_dim,
            )

    def _prepare_pair(left, right) -> PreparedClosureCandidate | None:
        u_fit, u_probe, u_source, u_node = left
        v_fit, v_probe, v_source, v_node = right
        built = build_rational_affine_candidate(
            scaffold_id=str(spec.scaffold_id),
            u_node=u_node,
            v_node=v_node,
            u_fit=u_fit,
            u_probe=u_probe,
            v_fit=v_fit,
            v_probe=v_probe,
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            safe_eps=float(safe_eps),
            u_source=str(u_source),
            v_source=str(v_source),
        )
        return PreparedClosureCandidate(
            built=built,
            candidate_subtree_node=u_node,
        )

    plan = PairCollectedSearchPlan(
        nvars=int(nvars),
        enum_max_depth=int(solver_kwargs.get("enum_max_depth", max_depth)),
        enum_max_trees=int(solver_kwargs.get("enum_max_trees", 5000)),
        var_dims=var_dims,
        left_target_dim=target_dim,
        right_target_dim=denom_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        left_shortlist_k=max(4, int(solver_kwargs.get("direct_rational_feature_topk", 24) or 24)),
        right_shortlist_k=max(4, int(solver_kwargs.get("direct_rational_feature_topk", 24) or 24)),
        left_pin_predicate=_pin_rational_left,
        right_pin_predicate=_pin_rational_right,
        prepare_candidate_fn=_prepare_pair,
        parent_stats=parent_stats,
        meta={
            "safe_eps": float(safe_eps),
        },
    )
    return plan


def solve_direct_rational_affine_preview_rows(
    spec: OperatorApplication,
    **kwargs,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows, status, meta_out = solve_direct_operator_preview_rows(spec, **kwargs)
    meta_out = dict(meta_out or {})
    if "left_candidates" in meta_out and "u_candidates" not in meta_out:
        meta_out["u_candidates"] = dict(meta_out.get("left_candidates", {}) or {})
    if "right_candidates" in meta_out and "v_candidates" not in meta_out:
        meta_out["v_candidates"] = dict(meta_out.get("right_candidates", {}) or {})
    if "left_shortlist_count" in meta_out and "u_shortlist_count" not in meta_out:
        meta_out["u_shortlist_count"] = int(meta_out.get("left_shortlist_count", 0) or 0)
    if "right_shortlist_count" in meta_out and "v_shortlist_count" not in meta_out:
        meta_out["v_shortlist_count"] = int(meta_out.get("right_shortlist_count", 0) or 0)
    return rows, status, meta_out


def _run_multi_term_rational_fallback(
    spec: OperatorApplication,
    *,
    single_term_best_mse: float,
    single_term_best_rows: Sequence[Mapping[str, Any]],
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn=collect_direct_hole_candidates,
    **_extra_kwargs,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Multi-term rational fallback: (a0 + sum a_i u_i) / (1 + sum b_j v_j).

    Screen candidate terms data-awarely, then enumerate small support sets.
    Uses collect_direct_hole_candidates_fn for the candidate pool (not raw
    pool_nodes) so compounds like x1*x2/x0^2 are reachable.
    """
    from ..basis_scoring import fit_multi_term_rational_design

    mt_started = time.perf_counter()
    mt_timings: dict[str, float] = {}

    def _mt_meta(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        out = dict(extra or {})
        out.update({str(k): float(v) for k, v in mt_timings.items()})
        out["timing_multi_term_total_s"] = float(time.perf_counter() - mt_started)
        return out

    if deadline_exceeded(deadline_s):
        return [], "multi_term_rational_deadline", _mt_meta()

    target_dim = y_dims if y_dims is not None else dim0(var_dims)
    denom_dim = dim0(var_dims) if var_dims is not None else None
    safe_eps = _mt_kw_float(solver_kwargs, "safe_eps", 1.0e-6, "NESTY_SAFE_EPS")
    screen_u_cap = max(4, _mt_kw_int(solver_kwargs, "multi_term_rational_max_u", 24, "NESTY_MULTI_TERM_RATIONAL_MAX_U"))
    screen_v_cap = max(4, _mt_kw_int(solver_kwargs, "multi_term_rational_max_v", 24, "NESTY_MULTI_TERM_RATIONAL_MAX_V"))
    screen_u_data_topk = max(
        4,
        min(
            screen_u_cap,
            _mt_kw_int(
                solver_kwargs,
                "multi_term_rational_u_data_topk",
                16,
                "NESTY_MULTI_TERM_RATIONAL_U_DATA_TOPK",
            ),
        ),
    )
    screen_v_data_topk = max(
        4,
        min(
            screen_v_cap,
            _mt_kw_int(
                solver_kwargs,
                "multi_term_rational_v_data_topk",
                12,
                "NESTY_MULTI_TERM_RATIONAL_V_DATA_TOPK",
            ),
        ),
    )
    screen_corr_threshold = _mt_kw_float(
        solver_kwargs,
        "multi_term_rational_screen_corr_threshold",
        0.9995,
        "NESTY_MULTI_TERM_RATIONAL_SCREEN_CORR_THRESHOLD",
    )
    role_shadow_enable = _mt_kw_bool(
        solver_kwargs,
        "multi_term_rational_role_shadow_enable",
        True,
        "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_ENABLE",
    )
    role_shadow_max_den = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_den",
            16,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_DEN",
        ),
    )
    role_shadow_max_product_den = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_product_den",
            role_shadow_max_den,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_PRODUCT_DEN",
        ),
    )
    role_shadow_max_low_cost_den = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_low_cost_den",
            role_shadow_max_den,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_LOW_COST_DEN",
        ),
    )
    role_shadow_max_supports = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_supports",
            1024,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_SUPPORTS",
        ),
    )
    role_shadow_max_num_blocks = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_num_blocks",
            64,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_NUM_BLOCKS",
        ),
    )
    role_shadow_max_product_num_blocks = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_product_num_blocks",
            max(8, int(role_shadow_max_num_blocks) // 2),
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_PRODUCT_NUM_BLOCKS",
        ),
    )
    role_shadow_max_carrier_size = max(
        1,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_carrier_size",
            12,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_CARRIER_SIZE",
        ),
    )
    role_shadow_max_node_size = max(
        1,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_node_size",
            12,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_NODE_SIZE",
        ),
    )
    role_shadow_max_add_terms = max(
        1,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_add_terms",
            6,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_ADD_TERMS",
        ),
    )
    role_shadow_max_synthetic_prefactors = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_synthetic_prefactors",
            8,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_SYNTHETIC_PREFACTORS",
        ),
    )
    role_shadow_max_synthetic_carriers = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_synthetic_carriers",
            64,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_SYNTHETIC_CARRIERS",
        ),
    )
    role_shadow_max_synthetic_num_blocks = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_max_synthetic_num_blocks",
            128,
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_MAX_SYNTHETIC_NUM_BLOCKS",
        ),
    )
    parent_stats = scaffold_parent_stats(spec)
    y_fit_1d = y_fit.squeeze(-1)
    y_probe_1d = y_probe.squeeze(-1)

    # Step 1: collect candidate nodes via the standard hole-candidate pipeline
    # (unions enumerated trees with the lane pool, giving us compounds)
    enum_depth = int(dict(solver_kwargs or {}).get("enum_max_depth", max_depth))
    enum_trees = int(dict(solver_kwargs or {}).get("enum_max_trees", 5000))
    collect_num_started = time.perf_counter()
    candidate_rows_num, _meta_num = collect_direct_hole_candidates_fn(
        nvars=int(nvars),
        enum_max_depth=enum_depth,
        enum_max_trees=enum_trees,
        var_dims=var_dims,
        target_dim=target_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        deadline_s=deadline_s,
    )
    mt_timings["timing_multi_collect_num_s"] = float(time.perf_counter() - collect_num_started)
    collect_den_started = time.perf_counter()
    candidate_rows_den, _meta_den = collect_direct_hole_candidates_fn(
        nvars=int(nvars),
        enum_max_depth=enum_depth,
        enum_max_trees=enum_trees,
        var_dims=var_dims,
        target_dim=denom_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        deadline_s=deadline_s,
    )
    mt_timings["timing_multi_collect_den_s"] = float(time.perf_counter() - collect_den_started)

    # Evaluate all candidate nodes on data
    eval_started = time.perf_counter()
    evaluated: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for source_tag, rows_list in [("num", candidate_rows_num), ("den", candidate_rows_den)]:
        for source, node in list(rows_list or ()):
            if not isinstance(node, tuple) or not is_valid_node(node):
                continue
            nkey = node_str(node)
            if nkey in seen_keys:
                continue
            seen_keys.add(nkey)
            if deadline_exceeded(deadline_s):
                break
            try:
                fit_val = eval_node(node, x_fit).squeeze(-1)
                probe_val = eval_node(node, x_probe).squeeze(-1)
            except Exception:
                continue
            if (not torch.is_tensor(fit_val)) or (not torch.is_tensor(probe_val)):
                continue
            if (not torch.isfinite(fit_val).all()) or (not torch.isfinite(probe_val).all()):
                continue
            nd = None
            if var_dims is not None:
                try:
                    nd = node_dims(node, var_dims)
                except Exception:
                    pass
            evaluated.append({
                "node": node,
                "fit": fit_val,
                "probe": probe_val,
                "dim": nd,
                "source": str(source),
            })
    mt_timings["timing_multi_eval_s"] = float(time.perf_counter() - eval_started)

    # Step 2: screen numerator and denominator candidates
    screen_started = time.perf_counter()
    max_u = min(screen_u_cap, len(evaluated))
    max_v = min(screen_v_cap, len(evaluated))

    # Numerator screening: OLS gain of [1, u_i] -> y
    u_scores: list[tuple[float, int]] = []
    for idx, entry in enumerate(evaluated):
        if target_dim is not None and entry["dim"] is not None:
            if not dims_eq(entry["dim"], target_dim):
                continue
        try:
            design = torch.stack([
                torch.ones_like(entry["fit"]),
                entry["fit"],
            ], dim=1)
            sol = torch.linalg.lstsq(design, y_fit_1d).solution
            pred = design @ sol
            mse = float(torch.mean((pred - y_fit_1d) ** 2).item())
            u_scores.append((mse, idx))
        except Exception:
            continue
    u_scores.sort(key=lambda t: t[0])
    u_indices = _select_diverse_screen_indices(
        scored=u_scores,
        evaluated=evaluated,
        max_count=min(max_u, screen_u_data_topk),
        corr_threshold=float(screen_corr_threshold),
        value_key="fit",
        drop_constant_like=True,
    )

    # Denominator screening: OLS gain of [1, -(y*v_j)] -> y
    v_scores: list[tuple[float, int]] = []
    for idx, entry in enumerate(evaluated):
        if denom_dim is not None and entry["dim"] is not None:
            if not dims_eq(entry["dim"], denom_dim):
                continue
        try:
            design = torch.stack([
                torch.ones_like(entry["fit"]),
                -(y_fit_1d * entry["fit"]),
            ], dim=1)
            sol = torch.linalg.lstsq(design, y_fit_1d).solution
            pred = design @ sol
            mse = float(torch.mean((pred - y_fit_1d) ** 2).item())
            v_scores.append((mse, idx))
        except Exception:
            continue
    v_scores.sort(key=lambda t: t[0])
    v_indices = _select_diverse_screen_indices(
        scored=v_scores,
        evaluated=evaluated,
        max_count=min(max_v, screen_v_data_topk),
        corr_threshold=float(screen_corr_threshold),
        value_key="fit",
        drop_constant_like=True,
    )

    # Union with the scout rational support, not the full quotient AST.
    for row in list(single_term_best_rows or ()):
        if not isinstance(row, Mapping):
            continue
        row_meta = dict(row.get("direct_metadata", {}) or {})
        seeded_rows = [
            ("u", _valid_bound_node(row_meta.get("u_node", None))),
            ("v", _valid_bound_node(row_meta.get("v_node", None))),
        ]
        for seed_kind, seed_node in seeded_rows:
            if seed_node is None:
                continue
            seed_key = node_str(seed_node)
            found = False
            for idx, entry in enumerate(evaluated):
                if node_str(entry["node"]) != seed_key:
                    continue
                if seed_kind == "u" and idx not in u_indices:
                    u_indices.append(idx)
                if seed_kind == "v" and idx not in v_indices:
                    v_indices.append(idx)
                found = True
                break
            if found:
                continue
            try:
                fit_val = eval_node(seed_node, x_fit).squeeze(-1)
                probe_val = eval_node(seed_node, x_probe).squeeze(-1)
            except Exception:
                continue
            if (not torch.is_tensor(fit_val)) or (not torch.is_tensor(probe_val)):
                continue
            if (not torch.isfinite(fit_val).all()) or (not torch.isfinite(probe_val).all()):
                continue
            new_idx = len(evaluated)
            evaluated.append(
                {
                    "node": seed_node,
                    "fit": fit_val,
                    "probe": probe_val,
                    "dim": row_meta.get("u_dim" if seed_kind == "u" else "v_dim", None),
                    "source": f"single_term_{seed_kind}",
                }
            )
            if seed_kind == "u":
                u_indices.append(new_idx)
            else:
                v_indices.append(new_idx)

    # Always keep raw target-dimension variables available for cooperative
    # numerator supports such as x1 + x2.
    for idx, entry in enumerate(evaluated):
        node = entry.get("node", None)
        if not (isinstance(node, tuple) and node):
            continue
        if str(node[0]) != "var":
            continue
        if target_dim is not None and entry["dim"] is not None and not dims_eq(entry["dim"], target_dim):
            continue
        _append_unique_index(u_indices, idx, limit=max_u)

    # Preserve a small algebraic denominator pool in addition to the
    # data-ranked columns. The marginal [1, -(y*v)] screen over-favors
    # constant-like dimensionless atoms and can bury useful rational factors.
    structural_v: list[tuple[tuple[int, int, int, int, str], int]] = []
    for idx, entry in enumerate(evaluated):
        node = entry.get("node", None)
        if not isinstance(node, tuple):
            continue
        if denom_dim is not None and entry["dim"] is not None and not dims_eq(entry["dim"], denom_dim):
            continue
        if _screen_vector_profile(entry.get("fit", None)) is None:
            continue
        if not _is_structural_rational_den_seed(node):
            continue
        structural_v.append((_rational_den_seed_priority(node), idx))
    structural_v.sort(key=lambda item: item[0])
    for _priority, idx in list(structural_v or ()):
        _append_unique_index(v_indices, idx, limit=max_v)

    role_shadow_protected_v: list[int] = []
    role_shadow_support_configs: list[tuple[tuple[int, ...], tuple[int, ...], str]] = []
    role_shadow_added_nodes = 0
    role_shadow_low_cost_v_count = 0
    role_shadow_product_pair_count = 0
    role_shadow_product_den_count = 0
    role_shadow_num_block_count = 0
    role_shadow_synthetic_prefactor_count = 0
    role_shadow_synthetic_carrier_count = 0
    role_shadow_synthetic_num_block_count = 0
    role_shadow_budget_explicit = False
    role_shadow_auto_budget_limit = 64
    generic_budget_reserve = 0

    def _role_shadow_meta_fields() -> dict[str, Any]:
        return {
            "role_shadow_enabled": bool(role_shadow_enable),
            "role_shadow_added_nodes": int(role_shadow_added_nodes),
            "role_shadow_low_cost_v_count": int(role_shadow_low_cost_v_count),
            "role_shadow_product_pair_count": int(role_shadow_product_pair_count),
            "role_shadow_product_den_count": int(role_shadow_product_den_count),
            "role_shadow_num_block_count": int(role_shadow_num_block_count),
            "role_shadow_synthetic_prefactor_count": int(role_shadow_synthetic_prefactor_count),
            "role_shadow_synthetic_carrier_count": int(role_shadow_synthetic_carrier_count),
            "role_shadow_synthetic_num_block_count": int(role_shadow_synthetic_num_block_count),
            "role_shadow_support_count": int(len(role_shadow_support_configs)),
            "role_shadow_max_den": int(role_shadow_max_den),
            "role_shadow_max_product_den": int(role_shadow_max_product_den),
            "role_shadow_max_low_cost_den": int(role_shadow_max_low_cost_den),
            "role_shadow_max_supports": int(role_shadow_max_supports),
            "role_shadow_max_num_blocks": int(role_shadow_max_num_blocks),
            "role_shadow_max_product_num_blocks": int(role_shadow_max_product_num_blocks),
            "role_shadow_max_synthetic_prefactors": int(role_shadow_max_synthetic_prefactors),
            "role_shadow_max_synthetic_carriers": int(role_shadow_max_synthetic_carriers),
            "role_shadow_max_synthetic_num_blocks": int(role_shadow_max_synthetic_num_blocks),
            "role_shadow_budget_explicit": bool(role_shadow_budget_explicit),
            "role_shadow_auto_budget_limit": int(role_shadow_auto_budget_limit),
            "role_shadow_generic_budget_reserve": int(generic_budget_reserve),
        }

    def _find_or_add_role_node(node: Any, *, source: str, desired_dim: Any) -> int | None:
        nonlocal role_shadow_added_nodes
        if not (isinstance(node, tuple) and is_valid_node(node)):
            return None
        try:
            simp = simplify(node)
        except Exception:
            return None
        if not (isinstance(simp, tuple) and is_valid_node(simp)):
            return None
        nd = _mt_node_dim(simp, var_dims)
        if not _mt_dim_matches(nd, desired_dim):
            return None
        key = _mt_node_key(simp)
        for existing_idx, existing in enumerate(evaluated):
            if _mt_node_key(existing.get("node", None)) == key:
                return int(existing_idx)
        try:
            fit_val = eval_node(simp, x_fit).squeeze(-1)
            probe_val = eval_node(simp, x_probe).squeeze(-1)
        except Exception:
            return None
        if (not torch.is_tensor(fit_val)) or (not torch.is_tensor(probe_val)):
            return None
        if (not torch.isfinite(fit_val).all()) or (not torch.isfinite(probe_val).all()):
            return None
        new_idx = len(evaluated)
        evaluated.append(
            {
                "node": simp,
                "fit": fit_val,
                "probe": probe_val,
                "dim": nd,
                "source": str(source),
            }
        )
        role_shadow_added_nodes += 1
        return int(new_idx)

    def _dedup_role_indices(values: Sequence[int]) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for raw in list(values or ()):
            try:
                idx = int(raw)
            except Exception:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    def _dedup_role_blocks(values: Sequence[Sequence[int]], *, min_len: int = 2) -> list[tuple[int, ...]]:
        out: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for raw in list(values or ()):
            block = tuple(dict.fromkeys(int(v) for v in tuple(raw or ())).keys())
            if len(block) < int(min_len) or block in seen:
                continue
            seen.add(block)
            out.append(block)
        return out

    if role_shadow_enable:
        low_cost_v: list[tuple[tuple[int, int, int, int, str], int]] = []
        for idx, entry in enumerate(evaluated):
            node = entry.get("node", None)
            if not isinstance(node, tuple):
                continue
            if not _mt_dim_matches(entry.get("dim", None), denom_dim):
                continue
            if _screen_vector_profile(entry.get("fit", None)) is None:
                continue
            if not _is_low_cost_rational_den_seed(node, max_size=int(role_shadow_max_node_size)):
                continue
            low_cost_v.append((_rational_den_seed_priority(node), int(idx)))
        low_cost_v.sort(key=lambda item: item[0])
        low_cost_den_atoms = [idx for _priority, idx in low_cost_v]
        product_den_atoms: list[int] = []
        product_num_blocks: list[tuple[int, ...]] = []
        product_prefactor_atoms: list[int] = []
        synthetic_num_blocks: list[tuple[int, ...]] = []
        singleton_num_blocks: list[tuple[int, ...]] = []

        for idx, entry in enumerate(evaluated):
            node = entry.get("node", None)
            if not (isinstance(node, tuple) and node):
                continue
            if str(node[0]) != "var":
                continue
            if not _mt_dim_matches(entry.get("dim", None), target_dim):
                continue
            product_prefactor_atoms.append(int(idx))

        for entry_idx, entry in list(enumerate(evaluated)):
            node = entry.get("node", None)
            if not isinstance(node, tuple):
                continue
            if not _mt_dim_matches(entry.get("dim", None), target_dim):
                continue
            term_nodes = _mt_additive_terms(node)[: int(role_shadow_max_add_terms)]
            for product_node in term_nodes:
                if not _mt_dim_matches(_mt_node_dim(product_node, var_dims), target_dim):
                    continue
                if _mt_node_key(product_node) == _mt_node_key(node):
                    product_idx = int(entry_idx)
                else:
                    added_idx = _find_or_add_role_node(
                        product_node,
                        source=f"role_shadow_term:{_mt_node_key(node)}",
                        desired_dim=target_dim,
                    )
                    if added_idx is None:
                        continue
                    product_idx = int(added_idx)
                for prefactor_node, carrier_node in _mt_factor_role_shadows(
                    product_node,
                    var_dims=var_dims,
                    target_dim=target_dim,
                    denom_dim=denom_dim,
                    max_carrier_size=int(role_shadow_max_carrier_size),
                ):
                    if len(role_shadow_support_configs) >= int(role_shadow_max_supports):
                        break
                    p_idx = _find_or_add_role_node(
                        prefactor_node,
                        source=f"role_shadow_prefactor:{_mt_node_key(product_node)}",
                        desired_dim=target_dim,
                    )
                    z_idx = _find_or_add_role_node(
                        carrier_node,
                        source=f"role_shadow_carrier:{_mt_node_key(product_node)}",
                        desired_dim=denom_dim,
                    )
                    if p_idx is None or z_idx is None:
                        continue
                    if _screen_vector_profile(evaluated[int(z_idx)].get("fit", None)) is None:
                        continue
                    _append_unique_index(u_indices, int(p_idx), limit=max_u)
                    _append_unique_index(u_indices, int(product_idx), limit=max_u)
                    _append_unique_index(v_indices, int(z_idx), limit=max_v)
                    product_prefactor_atoms.append(int(p_idx))
                    product_den_atoms.append(int(z_idx))
                    role_shadow_product_pair_count += 1
                    u_combo = tuple(dict.fromkeys((int(p_idx), int(product_idx))).keys())
                    if len(u_combo) < 2:
                        continue
                    product_num_blocks.append(u_combo)

        selected_product_den = _dedup_role_indices(product_den_atoms)[: int(role_shadow_max_product_den)]
        selected_low_cost_den = _dedup_role_indices(low_cost_den_atoms)[: int(role_shadow_max_low_cost_den)]
        role_shadow_product_den_count = int(len(selected_product_den))
        role_shadow_low_cost_v_count = int(len(selected_low_cost_den))
        role_shadow_protected_v = _dedup_role_indices(
            [*selected_product_den, *selected_low_cost_den]
        )[: int(role_shadow_max_den)]
        for idx in role_shadow_protected_v:
            _append_unique_index(v_indices, int(idx), limit=max_v)

        selected_synthetic_prefactors = _dedup_role_indices(product_prefactor_atoms)[
            : int(role_shadow_max_synthetic_prefactors)
        ]
        synthetic_carrier_ranked: list[tuple[tuple[int, int, int, int, str], int]] = []
        for idx in _dedup_role_indices([*low_cost_den_atoms, *product_den_atoms]):
            node = evaluated[int(idx)].get("node", None)
            if not isinstance(node, tuple):
                continue
            synthetic_carrier_ranked.append((_role_synthetic_carrier_structural_priority(node), int(idx)))
        synthetic_carrier_ranked.sort(key=lambda item: item[0])
        selected_synthetic_carriers = [
            idx for _priority, idx in synthetic_carrier_ranked[: int(role_shadow_max_synthetic_carriers)]
        ]
        role_shadow_synthetic_prefactor_count = int(len(selected_synthetic_prefactors))
        role_shadow_synthetic_carrier_count = int(len(selected_synthetic_carriers))
        for carrier_idx in selected_synthetic_carriers:
            if _mt_dim_matches(evaluated[int(carrier_idx)].get("dim", None), target_dim):
                singleton_num_blocks.append((int(carrier_idx),))
        seen_synthetic_products: set[tuple[int, int]] = set()
        synthetic_block_candidates: list[tuple[tuple[float, float, int, int, str], tuple[int, ...]]] = []

        def _score_num_block(block: Sequence[int]) -> tuple[float, float, int, int, str]:
            block_tuple = tuple(int(v) for v in tuple(block or ()))
            key = ":".join(_mt_node_key(evaluated[idx].get("node", None)) for idx in block_tuple)
            try:
                cols_fit = [torch.ones_like(y_fit_1d)]
                cols_probe = [torch.ones_like(y_probe_1d)]
                for idx in block_tuple:
                    cols_fit.append(evaluated[idx]["fit"])
                    cols_probe.append(evaluated[idx]["probe"])
                design_fit = torch.stack(cols_fit, dim=1)
                design_probe = torch.stack(cols_probe, dim=1)
                sol = torch.linalg.lstsq(design_fit, y_fit_1d).solution
                pred_fit = design_fit @ sol
                pred_probe = design_probe @ sol
                fit_mse = float(torch.mean((pred_fit - y_fit_1d) ** 2).item())
                probe_mse = float(torch.mean((pred_probe - y_probe_1d) ** 2).item())
                if not math.isfinite(fit_mse):
                    fit_mse = float("inf")
                if not math.isfinite(probe_mse):
                    probe_mse = float("inf")
            except Exception:
                fit_mse = float("inf")
                probe_mse = float("inf")
            try:
                total_size = sum(int(node_size(evaluated[idx]["node"])) for idx in block_tuple)
                total_depth = max(int(node_depth(evaluated[idx]["node"])) for idx in block_tuple)
            except Exception:
                total_size = 999
                total_depth = 999
            return (probe_mse, fit_mse, int(total_size), int(total_depth), key)

        for prefactor_idx in selected_synthetic_prefactors:
            prefactor_node = evaluated[int(prefactor_idx)].get("node", None)
            if not isinstance(prefactor_node, tuple):
                continue
            for carrier_idx in selected_synthetic_carriers:
                carrier_node = evaluated[int(carrier_idx)].get("node", None)
                if not isinstance(carrier_node, tuple):
                    continue
                pair_key = (int(prefactor_idx), int(carrier_idx))
                if pair_key in seen_synthetic_products:
                    continue
                seen_synthetic_products.add(pair_key)
                product_node = _mt_build_mul((prefactor_node, carrier_node))
                if product_node is None:
                    continue
                product_idx = _find_or_add_role_node(
                    product_node,
                    source=(
                        "role_shadow_synthetic_product:"
                        f"{_mt_node_key(prefactor_node)}:{_mt_node_key(carrier_node)}"
                    ),
                    desired_dim=target_dim,
                )
                if product_idx is None or int(product_idx) == int(prefactor_idx):
                    continue
                block = tuple(dict.fromkeys((int(prefactor_idx), int(product_idx))).keys())
                if len(block) < 2:
                    continue
                synthetic_block_candidates.append((_score_num_block(block), block))
        synthetic_block_candidates.sort(key=lambda item: item[0])
        synthetic_num_blocks = [
            block for _score, block in synthetic_block_candidates[: int(role_shadow_max_synthetic_num_blocks)]
        ]
        synthetic_num_blocks = _dedup_role_blocks(
            [*singleton_num_blocks, *synthetic_num_blocks],
            min_len=1,
        )
        role_shadow_synthetic_num_block_count = int(len(synthetic_num_blocks))

        product_blocks = _dedup_role_blocks(product_num_blocks, min_len=2)
        selected_product_blocks = product_blocks[
            : min(int(role_shadow_max_num_blocks), int(role_shadow_max_product_num_blocks))
        ]
        remaining_block_budget = max(0, int(role_shadow_max_num_blocks) - len(selected_product_blocks))
        selected_synthetic_blocks = synthetic_num_blocks[:remaining_block_budget]
        selected_num_blocks = _dedup_role_blocks(
            [*selected_product_blocks, *selected_synthetic_blocks],
            min_len=1,
        )[: int(role_shadow_max_num_blocks)]
        role_shadow_num_block_count = int(len(selected_num_blocks))

        def _role_pair_rank_key(
            u_combo: Sequence[int],
            v_combo: Sequence[int],
        ) -> tuple[float, float, int, int, str]:
            u_tuple = tuple(int(v) for v in tuple(u_combo or ()))
            v_tuple = tuple(int(v) for v in tuple(v_combo or ()))
            key = (
                ":".join(_mt_node_key(evaluated[idx].get("node", None)) for idx in u_tuple)
                + "/"
                + ":".join(_mt_node_key(evaluated[idx].get("node", None)) for idx in v_tuple)
            )
            try:
                cols_fit = [torch.ones_like(y_fit_1d)]
                cols_probe = [torch.ones_like(y_probe_1d)]
                for idx in u_tuple:
                    cols_fit.append(evaluated[idx]["fit"])
                    cols_probe.append(evaluated[idx]["probe"])
                design_fit = torch.stack(cols_fit, dim=1)
                design_probe = torch.stack(cols_probe, dim=1)
                sol = torch.linalg.lstsq(design_fit, y_fit_1d).solution
                n_fit = design_fit @ sol
                n_probe = design_probe @ sol
                num_probe_mse = float(torch.mean((n_probe - y_probe_1d) ** 2).item())
                residual_fit = n_fit - y_fit_1d
                residual_probe = n_probe - y_probe_1d
                q_fit = torch.stack([y_fit_1d * evaluated[idx]["fit"] for idx in v_tuple], dim=1)
                q_probe = torch.stack([y_probe_1d * evaluated[idx]["probe"] for idx in v_tuple], dim=1)
                b_sol = torch.linalg.lstsq(q_fit, residual_fit).solution
                residual_hat = q_probe @ b_sol
                sse = float(torch.sum((residual_probe - residual_hat) ** 2).item())
                centered = residual_probe - torch.mean(residual_probe)
                sst = float(torch.sum(centered * centered).item())
                quotient_r2 = 1.0 - sse / max(1.0e-30, sst)
                if not math.isfinite(num_probe_mse):
                    num_probe_mse = float("inf")
                if not math.isfinite(quotient_r2):
                    quotient_r2 = -float("inf")
            except Exception:
                num_probe_mse = float("inf")
                quotient_r2 = -float("inf")
            try:
                total_size = sum(int(node_size(evaluated[idx]["node"])) for idx in (*u_tuple, *v_tuple))
                total_depth = max(int(node_depth(evaluated[idx]["node"])) for idx in (*u_tuple, *v_tuple))
            except Exception:
                total_size = 999
                total_depth = 999
            return (-float(quotient_r2), float(num_probe_mse), int(total_size), int(total_depth), key)

        pair_candidates: list[tuple[tuple[float, float, int, int, str], tuple[int, ...], tuple[int, ...]]] = []
        seen_role_supports: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for u_combo in selected_num_blocks:
            for den_idx in list(role_shadow_protected_v):
                v_combo = (int(den_idx),)
                support_key = (u_combo, v_combo)
                if support_key in seen_role_supports:
                    continue
                seen_role_supports.add(support_key)
                pair_candidates.append((_role_pair_rank_key(u_combo, v_combo), u_combo, v_combo))
        pair_candidates.sort(key=lambda item: item[0])
        for _rank_key, u_combo, v_combo in pair_candidates[: int(role_shadow_max_supports)]:
            role_shadow_support_configs.append((u_combo, v_combo, "role_shadow"))
    mt_timings["timing_multi_screen_s"] = float(time.perf_counter() - screen_started)

    if not evaluated:
        return [], "multi_term_rational_no_candidates", _mt_meta({
            "evaluated_count": 0,
            **_role_shadow_meta_fields(),
        })

    if not u_indices and not v_indices:
        return [], "multi_term_rational_no_screened", _mt_meta({
            "evaluated_count": int(len(evaluated)),
            **_role_shadow_meta_fields(),
        })

    # Step 3: enumerate support sets
    support_started = time.perf_counter()
    support_configs = [(0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 1), (3, 2)]
    budget_cap = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_budget",
            500_000,
            "NESTY_MULTI_TERM_RATIONAL_BUDGET",
        ),
    )
    ranked_support_enable = _mt_kw_bool(
        solver_kwargs,
        "multi_term_rational_ranked_support_enable",
        False,
        "NESTY_MULTI_TERM_RATIONAL_RANKED_SUPPORT_ENABLE",
    )
    support_combo_cap = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_support_combo_cap",
            0,
            "NESTY_MULTI_TERM_RATIONAL_SUPPORT_COMBO_CAP",
        ),
    )
    support_config_budget = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_support_config_budget",
            0,
            "NESTY_MULTI_TERM_RATIONAL_SUPPORT_CONFIG_BUDGET",
        ),
    )
    support_cache_enable = _mt_kw_bool(
        solver_kwargs,
        "multi_term_rational_cache_support_enable",
        False,
        "NESTY_MULTI_TERM_RATIONAL_CACHE_SUPPORT_ENABLE",
    )
    support_cache_max_entries = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_cache_max_entries",
            16,
            "NESTY_MULTI_TERM_RATIONAL_CACHE_MAX_ENTRIES",
        ),
    )
    role_shadow_budget_raw = _mt_kw_int(
        solver_kwargs,
        "multi_term_rational_role_shadow_budget",
        -1,
        "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_BUDGET",
    )
    role_shadow_budget_explicit = int(role_shadow_budget_raw) >= 0
    # Role-shadow supports are a protected lane, but they run before generic
    # rational supports. Keep the automatic pre-generic tranche small so a
    # crowded role-shadow set cannot crowd out older generic rational stepping
    # stones under wall-clock pressure. Explicit caller budgets are still
    # honored for focused ablations/tests.
    role_shadow_auto_budget_limit = 64
    generic_budget_reserve = 0
    if budget_cap > 1:
        generic_budget_reserve = min(
            4096,
            max(16, int(budget_cap) // 16),
            max(0, int(budget_cap) // 2),
        )
    if int(role_shadow_budget_raw) >= 0:
        role_shadow_budget_cap = max(0, int(role_shadow_budget_raw))
    elif budget_cap <= 0:
        role_shadow_budget_cap = 0
    else:
        role_shadow_budget_cap = min(
            int(role_shadow_max_supports),
            int(role_shadow_auto_budget_limit),
            max(0, int(budget_cap) - int(generic_budget_reserve)),
        )
    role_shadow_preview_quota = max(
        0,
        _mt_kw_int(
            solver_kwargs,
            "multi_term_rational_role_shadow_preview_quota",
            min(8, max(1, int(preview_topk))),
            "NESTY_MULTI_TERM_RATIONAL_ROLE_SHADOW_PREVIEW_QUOTA",
        ),
    )
    budget_used = 0
    best_results: list[dict[str, Any]] = []
    support_cache_hit = False
    support_cache_key: tuple[Any, ...] | None = None

    u_score_by_idx = {int(idx): float(score) for score, idx in u_scores if math.isfinite(float(score))}
    v_score_by_idx = {int(idx): float(score) for score, idx in v_scores if math.isfinite(float(score))}
    for rank, idx in enumerate(u_indices):
        u_score_by_idx.setdefault(int(idx), 1.0e-12 * float(rank + 1))
    for rank, idx in enumerate(v_indices):
        v_score_by_idx.setdefault(int(idx), 1.0e-12 * float(rank + 1))

    role_shadow_scored_count = 0
    role_shadow_fit_improvement_count = 0
    role_shadow_preview_candidate_count = 0
    role_shadow_preview_attempt_count = 0
    role_shadow_preview_scored_count = 0
    role_shadow_preview_score_none_count = 0
    role_shadow_preview_kept_count = 0
    role_shadow_preview_row_none_count = 0
    role_shadow_preserved_count = 0
    role_shadow_budget_used = 0
    scored_support_keys: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def _score_support(u_combo_raw: Sequence[int], v_combo_raw: Sequence[int], *, support_source: str) -> bool:
        nonlocal budget_used, role_shadow_scored_count, role_shadow_fit_improvement_count
        if deadline_exceeded(deadline_s) or budget_used >= budget_cap:
            return False
        u_combo = tuple(int(v) for v in tuple(u_combo_raw or ()))
        v_combo = tuple(int(v) for v in tuple(v_combo_raw or ()))
        support_key = (u_combo, v_combo)
        if support_key in scored_support_keys:
            return False
        scored_support_keys.add(support_key)
        budget_used += 1
        source_token = str(support_source or "generic")
        if source_token == "role_shadow":
            role_shadow_scored_count += 1
        try:
            u_fits_list = [evaluated[i]["fit"] for i in u_combo]
            u_probes_list = [evaluated[i]["probe"] for i in u_combo]
            v_fits_list = [evaluated[i]["fit"] for i in v_combo]
            v_probes_list = [evaluated[i]["probe"] for i in v_combo]
        except Exception:
            return True
        result = fit_multi_term_rational_design(
            u_fits=u_fits_list,
            u_probes=u_probes_list,
            v_fits=v_fits_list,
            v_probes=v_probes_list,
            y_fit=y_fit,
            y_probe=y_probe,
            safe_eps=float(safe_eps),
        )
        if result is None:
            return True
        fit_mse, probe_mse, coeffs = result
        if probe_mse >= single_term_best_mse:
            return True
        u_nodes = [evaluated[i]["node"] for i in u_combo]
        v_nodes = [evaluated[i]["node"] for i in v_combo]
        u_sources = [evaluated[i]["source"] for i in u_combo]
        v_sources = [evaluated[i]["source"] for i in v_combo]
        if source_token == "role_shadow":
            role_shadow_fit_improvement_count += 1
        best_results.append({
            "probe_mse": float(probe_mse),
            "fit_mse": float(fit_mse),
            "coeffs": coeffs,
            "u_nodes": u_nodes,
            "v_nodes": v_nodes,
            "u_sources": u_sources,
            "v_sources": v_sources,
            "u_indices": list(u_combo),
            "v_indices": list(v_combo),
            "support_source": source_token,
            "role_shadow_support": bool(source_token == "role_shadow"),
        })
        return True

    def _support_result_key(result: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (
            tuple(_mt_node_key(node) for node in list(result.get("u_nodes", ()) or ())),
            tuple(_mt_node_key(node) for node in list(result.get("v_nodes", ()) or ())),
        )

    def _select_preview_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal role_shadow_preview_candidate_count, role_shadow_preserved_count
        sorted_rows = sorted(list(rows or ()), key=lambda r: float(r.get("probe_mse", float("inf"))))
        global_rows = sorted_rows[: max(1, int(preview_topk))]
        global_keys = {_support_result_key(row) for row in global_rows}
        role_rows = [
            row for row in sorted_rows
            if bool(row.get("role_shadow_support", False))
        ][: int(role_shadow_preview_quota)]
        merged: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for row in [*global_rows, *role_rows]:
            key = _support_result_key(row)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
        role_shadow_preview_candidate_count = sum(
            1 for row in merged if bool(row.get("role_shadow_support", False))
        )
        role_shadow_preserved_count = sum(
            1
            for row in merged
            if bool(row.get("role_shadow_support", False))
            and _support_result_key(row) not in global_keys
        )
        return merged

    if support_cache_enable:
        support_cache_key = (
            "multi_term_rational_support_v6",
            int(id(x_fit)),
            int(id(x_probe)),
            int(id(y_fit)),
            int(id(y_probe)),
            tuple(int(v) for v in tuple(x_fit.shape)),
            tuple(int(v) for v in tuple(y_fit.shape)),
            tuple(_mt_node_key(entry.get("node", None)) for entry in evaluated),
            tuple(_mt_node_key(evaluated[int(idx)]["node"]) for idx in u_indices),
            tuple(_mt_node_key(evaluated[int(idx)]["node"]) for idx in v_indices),
            tuple(tuple(int(v) for v in config) for config in support_configs),
            tuple(
                (
                    tuple(_mt_node_key(evaluated[int(idx)]["node"]) for idx in u_combo),
                    tuple(_mt_node_key(evaluated[int(idx)]["node"]) for idx in v_combo),
                    str(source),
                )
                for u_combo, v_combo, source in role_shadow_support_configs
            ),
            bool(role_shadow_enable),
            int(role_shadow_max_den),
            int(role_shadow_max_supports),
            int(role_shadow_max_carrier_size),
            int(role_shadow_max_node_size),
            int(role_shadow_max_add_terms),
            int(role_shadow_max_product_den),
            int(role_shadow_max_low_cost_den),
            int(role_shadow_max_num_blocks),
            int(role_shadow_max_product_num_blocks),
            int(role_shadow_max_synthetic_prefactors),
            int(role_shadow_max_synthetic_carriers),
            int(role_shadow_max_synthetic_num_blocks),
            bool(role_shadow_budget_explicit),
            int(role_shadow_auto_budget_limit),
            int(generic_budget_reserve),
            int(role_shadow_budget_cap),
            int(role_shadow_preview_quota),
            int(budget_cap),
            bool(ranked_support_enable),
            int(support_combo_cap),
            int(support_config_budget),
            round(float(single_term_best_mse), 18),
            float(safe_eps),
            int(preview_topk),
        )
        cached = _mt_support_cache_get(support_cache_key)
        if cached is not None:
            best_results, budget_used = cached
            support_cache_hit = True
            role_shadow_fit_improvement_count = sum(
                1 for row in best_results if bool(row.get("role_shadow_support", False))
            )

    if not support_cache_hit:
        for u_combo, v_combo, source in list(role_shadow_support_configs or ()):
            if deadline_exceeded(deadline_s) or budget_used >= budget_cap:
                break
            if role_shadow_budget_used >= int(role_shadow_budget_cap):
                break
            if _score_support(u_combo, v_combo, support_source=source):
                role_shadow_budget_used += 1
        for n_u, n_v in support_configs:
            if deadline_exceeded(deadline_s) or budget_used >= budget_cap:
                break
            if n_u > len(u_indices) or n_v > len(v_indices):
                continue
            if ranked_support_enable:
                u_combos = _ranked_combo_list(u_indices, n_u, u_score_by_idx, support_combo_cap) if n_u > 0 else [()]
                v_combos = _ranked_combo_list(v_indices, n_v, v_score_by_idx, support_combo_cap) if n_v > 0 else [()]
            else:
                u_combos = list(combinations(u_indices, n_u)) if n_u > 0 else [()]
                v_combos = list(combinations(v_indices, n_v)) if n_v > 0 else [()]
            config_used = 0
            for u_combo in u_combos:
                if deadline_exceeded(deadline_s) or budget_used >= budget_cap:
                    break
                if support_config_budget > 0 and config_used >= support_config_budget:
                    break
                for v_combo in v_combos:
                    if budget_used >= budget_cap:
                        break
                    if support_config_budget > 0 and config_used >= support_config_budget:
                        break
                    if deadline_exceeded(deadline_s):
                        break
                    if _score_support(u_combo, v_combo, support_source="generic"):
                        config_used += 1
        if best_results:
            best_results = _select_preview_results(best_results)
        if support_cache_key is not None and not deadline_exceeded(deadline_s):
            _mt_support_cache_put(support_cache_key, best_results, budget_used, support_cache_max_entries)
    else:
        best_results = _select_preview_results(best_results)
    mt_timings["timing_multi_support_s"] = float(time.perf_counter() - support_started)

    if not best_results:
        return [], "multi_term_rational_no_improvement", _mt_meta({
            "budget_used": budget_used,
            "evaluated_count": int(len(evaluated)),
            "u_screen_count": int(len(u_indices)),
            "v_screen_count": int(len(v_indices)),
            "support_cache_enabled": bool(support_cache_enable),
            "support_cache_hit": bool(support_cache_hit),
            "ranked_support_enable": bool(ranked_support_enable),
            "support_config_budget": int(support_config_budget),
            "role_shadow_scored_count": int(role_shadow_scored_count),
            "role_shadow_budget_cap": int(role_shadow_budget_cap),
            "role_shadow_budget_used": int(role_shadow_budget_used),
            "role_shadow_fit_improvement_count": int(role_shadow_fit_improvement_count),
            "role_shadow_preview_quota": int(role_shadow_preview_quota),
            "role_shadow_preview_candidate_count": int(role_shadow_preview_candidate_count),
            "role_shadow_preview_attempt_count": int(role_shadow_preview_attempt_count),
            "role_shadow_preview_scored_count": int(role_shadow_preview_scored_count),
            "role_shadow_preview_score_none_count": int(role_shadow_preview_score_none_count),
            "role_shadow_preview_kept_count": int(role_shadow_preview_kept_count),
            "role_shadow_preview_row_none_count": int(role_shadow_preview_row_none_count),
            "role_shadow_preserved_count": int(role_shadow_preserved_count),
            "role_shadow_hit_count": int(role_shadow_preview_kept_count),
            **_role_shadow_meta_fields(),
        })

    best_results.sort(key=lambda r: float(r.get("probe_mse", float("inf"))))

    # Step 4: Build preview rows with correct signatures
    preview_started = time.perf_counter()
    preview_rows: list[dict[str, Any]] = []
    seen_child_keys: set[str] = set()
    scored_candidate_count = 0
    for result in best_results:
        is_role_shadow_result = bool(result.get("role_shadow_support", False))
        if is_role_shadow_result:
            role_shadow_preview_attempt_count += 1
        u_nodes_r = result["u_nodes"]
        v_nodes_r = result["v_nodes"]
        built = build_multi_term_rational_candidate(
            scaffold_id=str(spec.scaffold_id),
            u_nodes=tuple(u_nodes_r),
            v_nodes=tuple(v_nodes_r),
            u_fits=[evaluated[i]["fit"] for i in result["u_indices"]],
            u_probes=[evaluated[i]["probe"] for i in result["u_indices"]],
            v_fits=[evaluated[i]["fit"] for i in result["v_indices"]],
            v_probes=[evaluated[i]["probe"] for i in result["v_indices"]],
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            safe_eps=float(safe_eps),
            u_sources=result["u_sources"],
            v_sources=result["v_sources"],
        )
        score_result = score_bound_closure(
            built.bound_closure,
            design=built.design,
            y_fit=y_fit,
            y_probe=y_probe,
        )
        if score_result is None:
            if is_role_shadow_result:
                role_shadow_preview_score_none_count += 1
            continue
        scored_candidate_count += 1
        if is_role_shadow_result:
            role_shadow_preview_scored_count += 1
        row = make_direct_preview_row(
            bound_closure=built.bound_closure,
            child_expr=score_result["expr"],
            fit_mse=float(score_result["fit_mse"]),
            probe_mse=float(score_result["probe_mse"]),
            max_depth=int(max_depth) + 2,  # extra depth for multi-term rational
            var_dims=var_dims,
            y_dims=y_dims,
            candidate_subtree_node=(
                u_nodes_r[0]
                if u_nodes_r
                else (v_nodes_r[0] if v_nodes_r else None)
            ),
            parent_sub_size=int(parent_stats.get("parent_sub_size", 1)),
            parent_sub_depth=int(parent_stats.get("parent_sub_depth", 1)),
            parent_size=int(parent_stats.get("parent_size", 1)),
            parent_depth=int(parent_stats.get("parent_depth", 1)),
            generation_source=str(built.generation_source),
            tuple_provenance=str(built.tuple_provenance),
            proposal_family=str(built.proposal_family),
            local_mapping_kind=str(built.local_mapping_kind),
            direct_metadata={
                **dict(built.direct_metadata or {}),
                "support_size": (len(u_nodes_r), len(v_nodes_r)),
                "u_node_exprs": [node_str(n) for n in u_nodes_r],
                "v_node_exprs": [node_str(n) for n in v_nodes_r],
                "support_source": str(result.get("support_source", "generic")),
                "role_shadow_support": bool(result.get("role_shadow_support", False)),
            },
            seen_child_keys=seen_child_keys,
            local_mapping_coeffs=[float(c) for c in result["coeffs"]],
            local_mapping_nparams=len(result["coeffs"]),
        )
        if row is not None:
            preview_rows.append(row)
            if is_role_shadow_result:
                role_shadow_preview_kept_count += 1
        elif is_role_shadow_result:
            role_shadow_preview_row_none_count += 1
    mt_timings["timing_multi_preview_s"] = float(time.perf_counter() - preview_started)

    return finalize_direct_preview_rows(
        preview_rows,
        preview_topk=max(1, int(preview_topk)),
        raw_candidate_count=budget_used,
        scored_candidate_count=scored_candidate_count,
        deadline_s=deadline_s,
        meta=_mt_meta({
            "budget_used": budget_used,
            "candidates_found": len(best_results),
            "evaluated_count": int(len(evaluated)),
            "u_screen_count": len(u_indices),
            "v_screen_count": len(v_indices),
            "support_cache_enabled": bool(support_cache_enable),
            "support_cache_hit": bool(support_cache_hit),
            "ranked_support_enable": bool(ranked_support_enable),
            "support_config_budget": int(support_config_budget),
            "role_shadow_scored_count": int(role_shadow_scored_count),
            "role_shadow_budget_cap": int(role_shadow_budget_cap),
            "role_shadow_budget_used": int(role_shadow_budget_used),
            "role_shadow_fit_improvement_count": int(role_shadow_fit_improvement_count),
            "role_shadow_preview_quota": int(role_shadow_preview_quota),
            "role_shadow_preview_candidate_count": int(role_shadow_preview_candidate_count),
            "role_shadow_preview_attempt_count": int(role_shadow_preview_attempt_count),
            "role_shadow_preview_scored_count": int(role_shadow_preview_scored_count),
            "role_shadow_preview_score_none_count": int(role_shadow_preview_score_none_count),
            "role_shadow_preview_kept_count": int(role_shadow_preview_kept_count),
            "role_shadow_preview_row_none_count": int(role_shadow_preview_row_none_count),
            "role_shadow_preserved_count": int(role_shadow_preserved_count),
            "role_shadow_hit_count": int(role_shadow_preview_kept_count),
            **_role_shadow_meta_fields(),
        }),
    )


def build_direct_power_search_plan(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn=collect_direct_hole_candidates,
) -> Any:
    power_kind = direct_power_scaffold_kind(spec)
    if power_kind is None:
        return _static_status_plan("direct_power_unsupported_form")

    power_map = {
        "sqrt": (0.5, False),
        "sqrt_mul": (0.5, True),
        "invsqrt": (-0.5, False),
        "invsqrt_mul": (-0.5, True),
        "inv": (-1.0, False),
        "inv_mul": (-1.0, True),
        "neg2": (-2.0, False),
        "neg2_mul": (-2.0, True),
        "sqr": (2.0, False),
        "sqr_mul": (2.0, True),
    }
    exponent, needs_anchor = power_map[str(power_kind)]
    allow_slot_rebinding = bool(dict(solver_kwargs or {}).get("allow_slot_rebinding", False))
    if not allow_slot_rebinding:
        exact_plan = _build_exact_bound_power_plan(
            spec,
            power_kind=str(power_kind),
            exponent=float(exponent),
            x_fit=x_fit,
            x_probe=x_probe,
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            safe_eps=float(solver_kwargs.get("safe_eps", 1.0e-8)),
        )
        if exact_plan is not None:
            return exact_plan
    anchor_node = spec.anchor_node if needs_anchor else None
    if needs_anchor and (not isinstance(anchor_node, tuple) or not is_valid_node(anchor_node)):
        return _static_status_plan("direct_missing_anchor")

    target_dim = y_dims
    anchor_fit = None
    anchor_probe = None
    if isinstance(anchor_node, tuple):
        try:
            anchor_fit = eval_node(anchor_node, x_fit)
            anchor_probe = eval_node(anchor_node, x_probe)
        except Exception:
            return _static_status_plan("direct_anchor_eval_failed")
        if (not torch.is_tensor(anchor_fit)) or (not torch.is_tensor(anchor_probe)):
            return _static_status_plan("direct_anchor_eval_failed")
        if (not torch.isfinite(anchor_fit).all()) or (not torch.isfinite(anchor_probe).all()):
            return _static_status_plan("direct_anchor_nonfinite")
        if var_dims is not None and y_dims is not None:
            try:
                anchor_dim = node_dims(anchor_node, var_dims)
            except Exception:
                anchor_dim = None
            if anchor_dim is None:
                return _static_status_plan("direct_anchor_dim_failed")
            target_dim = dim_scale(dim_sub(y_dims, anchor_dim), 1.0 / float(exponent))
    elif y_dims is not None:
        target_dim = dim_scale(y_dims, 1.0 / float(exponent))

    parent_stats = scaffold_parent_stats(spec)

    def _power_variants(exp_value: float) -> tuple[str | None, ...]:
        if float(exp_value) != 2.0:
            return (None,)
        return ("square_only", "bias_square", "linear_square", "full_quadratic")

    def _prepare_power_candidate(
        source: str,
        hole_node: tuple,
        *,
        anchor_node_local: tuple | None,
        anchor_fit_local: torch.Tensor | None,
        anchor_probe_local: torch.Tensor | None,
    ) -> list[PreparedClosureCandidate]:
        try:
            h_fit = eval_node(hole_node, x_fit)
            h_probe = eval_node(hole_node, x_probe)
        except Exception:
            return []
        if (not torch.is_tensor(h_fit)) or (not torch.is_tensor(h_probe)):
            return []
        if (not torch.isfinite(h_fit).all()) or (not torch.isfinite(h_probe).all()):
            return []
        out: list[PreparedClosureCandidate] = []
        for power_variant in _power_variants(float(exponent)):
            built = build_affine_power_candidate(
                scaffold_id=str(spec.scaffold_id),
                power_kind=str(power_kind),
                exponent=float(exponent),
                hole_node=hole_node,
                anchor_node=anchor_node_local,
                h_fit=h_fit,
                h_probe=h_probe,
                anchor_fit=anchor_fit_local,
                anchor_probe=anchor_probe_local,
                max_depth=int(max_depth),
                var_dims=var_dims,
                y_dims=y_dims,
                safe_eps=float(solver_kwargs.get("safe_eps", 1.0e-8)),
                source=str(source),
                power_variant=power_variant,
            )
            out.append(
                PreparedClosureCandidate(
                    built=built,
                    candidate_subtree_node=hole_node,
                    row_patch={
                        "local_mapping_nparams": int(
                            direct_power_variant_nparams(power_variant, exponent=float(exponent))
                        ),
                        "direct_metadata": {
                            "feature_node": built.direct_metadata.get("hole_node"),
                            "power_variant": str(power_variant or ""),
                        },
                    },
                )
            )
        return out

    def _prepare_power(source: str, hole_node: tuple) -> list[PreparedClosureCandidate]:
        return _prepare_power_candidate(
            source,
            hole_node,
            anchor_node_local=anchor_node,
            anchor_fit_local=anchor_fit,
            anchor_probe_local=anchor_probe,
        )

    def _pin_power(node: tuple) -> bool:
        if exponent < 0.0:
            if pin_dimensionless_ratio_square(node, var_dims=var_dims):
                return True
            if pin_ratio_square(node):
                return True
        if exponent == 2.0 and pin_small_trig_carrier(node):
            return True
        return pin_single_var_square(node)

    if needs_anchor and allow_slot_rebinding:
        anchor_topk = max(
            4,
            int(solver_kwargs.get("direct_power_anchor_topk", max(4, int(nvars) * 2)) or max(4, int(nvars) * 2)),
        )
        anchor_blocks = dedup_seed_blocks(
            seed_anchor_blocks(
                nvars=int(nvars),
                pool_nodes=pool_nodes,
                pool_dims=pool_dims,
                var_dims=var_dims,
                max_count=int(anchor_topk),
            )
        )

        prepared_candidates: list[PreparedClosureCandidate] = []
        seen_pairs: set[str] = set()
        hole_shortlists_by_dim: dict[str, list[tuple[str, tuple]]] = {}
        for anchor_block in list(anchor_blocks or ())[: int(anchor_topk)]:
            anchor_node_local = anchor_block.node
            try:
                anchor_fit_local = eval_node(anchor_node_local, x_fit)
                anchor_probe_local = eval_node(anchor_node_local, x_probe)
            except Exception:
                continue
            if (not torch.is_tensor(anchor_fit_local)) or (not torch.is_tensor(anchor_probe_local)):
                continue
            if (not torch.isfinite(anchor_fit_local).all()) or (not torch.isfinite(anchor_probe_local).all()):
                continue
            local_target_dim = target_dim
            if var_dims is not None and y_dims is not None:
                anchor_dim = anchor_block.dim
                if anchor_dim is None:
                    try:
                        anchor_dim = node_dims(anchor_node_local, var_dims)
                    except Exception:
                        anchor_dim = None
                if anchor_dim is None:
                    continue
                local_target_dim = dim_scale(dim_sub(y_dims, anchor_dim), 1.0 / float(exponent))
            dim_key = str(local_target_dim)
            shortlisted = hole_shortlists_by_dim.get(dim_key, None)
            if shortlisted is None:
                shortlisted, _hole_meta = collect_shortlisted_hole_candidates(
                    collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
                    nvars=int(nvars),
                    enum_max_depth=int(solver_kwargs.get("enum_max_depth", max_depth)),
                    enum_max_trees=int(solver_kwargs.get("enum_max_trees", 5000)),
                    var_dims=var_dims,
                    target_dim=local_target_dim,
                    pool_nodes=pool_nodes,
                    pool_dims=pool_dims,
                    shortlist_k=max(4, int(solver_kwargs.get("direct_power_feature_topk", 8) or 8)),
                    deadline_s=deadline_s,
                    pin_predicate=_pin_power,
                )
                hole_shortlists_by_dim[dim_key] = list(shortlisted)
            for source, hole_node in list(shortlisted or ()):
                pair_key = f"{node_str(anchor_node_local)}::{node_str(hole_node)}"
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates = _prepare_power_candidate(
                    source,
                    hole_node,
                    anchor_node_local=anchor_node_local,
                    anchor_fit_local=anchor_fit_local,
                    anchor_probe_local=anchor_probe_local,
                )
                prepared_candidates.extend(list(candidates or ()))

        if prepared_candidates:
            return PreparedCandidatesSearchPlan(
                candidates=tuple(prepared_candidates),
                var_dims=var_dims,
                parent_stats=parent_stats,
                meta={
                    "power_kind": str(power_kind),
                    "power_exponent": float(exponent),
                    "anchor_rebinding": True,
                    "anchor_candidate_count": int(min(len(list(anchor_blocks or ())), int(anchor_topk))),
                    "prepared_candidate_count": int(len(prepared_candidates)),
                },
            )

    shortlisted, shortlist_meta = collect_shortlisted_hole_candidates(
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        nvars=int(nvars),
        enum_max_depth=int(solver_kwargs.get("enum_max_depth", max_depth)),
        enum_max_trees=int(solver_kwargs.get("enum_max_trees", 5000)),
        var_dims=var_dims,
        target_dim=target_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        shortlist_k=max(4, int(solver_kwargs.get("direct_power_feature_topk", 8) or 8)),
        deadline_s=deadline_s,
        pin_predicate=_pin_power,
    )
    prepared_candidates: list[PreparedClosureCandidate] = []
    for source, hole_node in list(shortlisted or ()):
        prepared_candidates.extend(list(_prepare_power(str(source), hole_node) or ()))
    if not prepared_candidates:
        return _static_status_plan(
            "direct_no_hole_candidates"
            if not shortlisted
            else "direct_no_scored_candidates"
        )
    return PreparedCandidatesSearchPlan(
        candidates=tuple(prepared_candidates),
        var_dims=var_dims,
        parent_stats=parent_stats,
        meta={
            "power_kind": str(power_kind),
            "power_exponent": float(exponent),
            **dict(shortlist_meta or {}),
            "prepared_candidate_count": int(len(prepared_candidates)),
        },
    )


def solve_direct_power_preview_rows(
    spec: OperatorApplication,
    **kwargs,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows, status, meta_out = solve_direct_operator_preview_rows(spec, **kwargs)
    for row in rows:
        row_meta = dict(row.get("direct_metadata", {}) or {})
        row_meta["feature_node"] = row.get("expr")
        row["direct_metadata"] = row_meta
    return rows, status, meta_out


def build_direct_quadratic_search_plan(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None = None,
) -> Any:
    quadratic_kind = direct_quadratic_scaffold_kind(spec)
    if quadratic_kind is None:
        return _static_status_plan("direct_quadratic_unsupported_form")
    allow_slot_rebinding = bool(dict(solver_kwargs or {}).get("allow_slot_rebinding", False))
    if not allow_slot_rebinding:
        exact_plan = _build_exact_bound_quadratic_plan(
            spec,
            quadratic_kind=str(quadratic_kind),
            x_fit=x_fit,
            x_probe=x_probe,
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            safe_eps=float(solver_kwargs.get("safe_eps", 1.0e-8)),
        )
        if exact_plan is not None:
            return exact_plan

    anchor_node = spec.anchor_node if isinstance(spec.anchor_node, tuple) and is_valid_node(spec.anchor_node) else None
    anchor_fit = None
    anchor_probe = None
    core_dim = y_dims
    if quadratic_kind == "sqrt_mul":
        if anchor_node is None:
            return _static_status_plan("direct_missing_anchor")
        try:
            anchor_fit = eval_node(anchor_node, x_fit)
            anchor_probe = eval_node(anchor_node, x_probe)
        except Exception:
            return _static_status_plan("direct_anchor_eval_failed")
        if (not torch.is_tensor(anchor_fit)) or (not torch.is_tensor(anchor_probe)):
            return _static_status_plan("direct_anchor_eval_failed")
        if (not torch.isfinite(anchor_fit).all()) or (not torch.isfinite(anchor_probe).all()):
            return _static_status_plan("direct_anchor_nonfinite")
        if var_dims is not None and y_dims is not None:
            try:
                anchor_dim = node_dims(anchor_node, var_dims)
            except Exception:
                anchor_dim = None
            if anchor_dim is None:
                return _static_status_plan("direct_anchor_dim_failed")
            core_dim = dim_sub(y_dims, anchor_dim)

    seed_cap = max(4, int(solver_kwargs.get("direct_quadratic_seed_topk", max(4, int(nvars))) or max(4, int(nvars))))
    base_blocks = seed_anchor_blocks(
        nvars=int(nvars),
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        var_dims=var_dims,
        max_count=seed_cap,
    )
    base_blocks = filter_seed_blocks_for_dim(
        base_blocks,
        target_dim=core_dim,
        var_dims=var_dims,
        drop_const=True,
    )
    prefer_structured = _prefer_quadratic_structured_bases(str(quadratic_kind))
    if prefer_structured:
        diff_limit = max(
            0,
            int(solver_kwargs.get("direct_quadratic_affine_diff_topk", 2) or 2),
        )
        if diff_limit > 0:
            base_blocks = dedup_seed_blocks(
                list(base_blocks)
                + _build_quadratic_affine_difference_blocks(
                    base_blocks,
                    var_dims=var_dims,
                    limit=max(diff_limit * 4, diff_limit),
                )
            )
    if prefer_structured:
        base_blocks = [
            block
            for block in base_blocks
            if max(0, node_var_count(block.node)) <= 1 or _quadratic_structured_seed(block)
        ]
    else:
        base_blocks = [block for block in base_blocks if max(0, node_var_count(block.node)) <= 1]
    if anchor_node is not None:
        anchor_key = str(node_str(anchor_node))
        base_blocks = [block for block in base_blocks if str(node_str(block.node)) != anchor_key]
    if not base_blocks:
        return _static_status_plan("direct_no_quadratic_bases")

    base_blocks = dedup_seed_blocks(base_blocks)
    primary_var_blocks = [
        block
        for block in base_blocks
        if isinstance(block.node, tuple) and block.node and str(block.node[0]) == "var"
    ]
    ratio_blocks = [block for block in base_blocks if _quadratic_ratio_or_inverse_seed(block)]
    affine_diff_blocks = [block for block in base_blocks if _quadratic_affine_difference_seed(block)]
    ratio_keys = {str(node_str(block.node)) for block in ratio_blocks}
    structured_blocks = list(ratio_blocks) + [
        block for block in affine_diff_blocks if str(node_str(block.node)) not in ratio_keys
    ]
    if len(primary_var_blocks) >= 2 and not (prefer_structured and structured_blocks):
        base_blocks = primary_var_blocks
    base_topk = max(2, int(solver_kwargs.get("direct_quadratic_base_topk", 4) or 4))
    if prefer_structured and structured_blocks:
        sorted_ratio_blocks = sorted(ratio_blocks, key=quadratic_base_priority)
        sorted_affine_blocks = sorted(affine_diff_blocks, key=_quadratic_affine_difference_priority)
        reserved_ratio = min(len(sorted_ratio_blocks), min(int(base_topk), 2))
        reserved_affine = min(
            len(sorted_affine_blocks),
            max(0, min(int(base_topk) - reserved_ratio, 2)),
        )
        selected_blocks = list(sorted_ratio_blocks[:reserved_ratio])
        seen_keys = {str(node_str(block.node)) for block in selected_blocks}
        selected_affine_keys: list[str] = []
        used_affine_vars: set[int] = set()
        remaining_affine = list(sorted_affine_blocks)
        while remaining_affine and len(selected_affine_keys) < reserved_affine:
            best_idx = 0
            best_rank = None
            for idx, block in enumerate(remaining_affine):
                key = str(node_str(block.node))
                if key in seen_keys:
                    continue
                support = _quadratic_affine_difference_support(block)
                rank = (
                    -len(list(support.difference(used_affine_vars))),
                    *_quadratic_affine_difference_priority(block),
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = idx
            block = remaining_affine.pop(best_idx)
            key = str(node_str(block.node))
            if key in seen_keys:
                continue
            selected_blocks.append(block)
            selected_affine_keys.append(key)
            seen_keys.add(key)
            used_affine_vars.update(_quadratic_affine_difference_support(block))
        remaining_structured = [
            block
            for block in list(sorted_ratio_blocks[reserved_ratio:]) + list(sorted_affine_blocks)
            if str(node_str(block.node)) not in seen_keys
        ]
        other_blocks = [
            block
            for block in sorted(base_blocks, key=quadratic_base_priority)
            if not _quadratic_structured_seed(block)
        ]
        for block in remaining_structured + other_blocks:
            key = str(node_str(block.node))
            if key in seen_keys:
                continue
            selected_blocks.append(block)
            seen_keys.add(key)
            if len(selected_blocks) >= int(base_topk):
                break
        base_blocks = selected_blocks
    else:
        base_blocks.sort(key=quadratic_base_priority)
        base_blocks = base_blocks[: int(base_topk)]
    subset_max_arity = max(1, int(solver_kwargs.get("direct_quadratic_max_arity", 4) or 4))
    parent_stats = scaffold_parent_stats(spec)

    def _prepare_subset(combo: tuple[dict[str, Any], ...]) -> PreparedClosureCandidate | None:
        quad_fit = torch.stack([row["fit"] ** 2 for row in combo], dim=1)
        quad_probe = torch.stack([row["probe"] ** 2 for row in combo], dim=1)
        base_nodes = [row["block"].node for row in combo]
        base_sources = [str(row["block"].source) for row in combo]
        built = build_quadratic_sqrt_candidate(
            scaffold_id=str(spec.scaffold_id),
            quadratic_kind=str(quadratic_kind),
            base_nodes=tuple(base_nodes),
            anchor_node=anchor_node,
            quad_fit=quad_fit,
            quad_probe=quad_probe,
            anchor_fit=anchor_fit,
            anchor_probe=anchor_probe,
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            safe_eps=float(solver_kwargs.get("safe_eps", 1.0e-8)),
            base_sources=base_sources,
        )
        quad_terms = [simplify(("sqr", node)) for node in base_nodes]
        quad_latent_node = simplify(
            ("add", quad_terms[0], quad_terms[1]) if len(quad_terms) == 2 else quad_terms[0]
        )
        if len(quad_terms) > 2:
            quad_latent_node = quad_terms[0]
            for term in quad_terms[1:]:
                quad_latent_node = simplify(("add", quad_latent_node, term))
        candidate_subtree_node = (
            base_nodes[0]
            if len(base_nodes) == 1
            else simplify(("add", *base_nodes[:2])) if len(base_nodes) >= 2 else None
        )
        return PreparedClosureCandidate(
            built=built,
            candidate_subtree_node=candidate_subtree_node,
            row_patch={
                "direct_metadata": {
                    "quadratic_latent_node": quad_latent_node,
                }
            },
        )

    plan = SeedSubsetSearchPlan(
        seed_blocks=base_blocks,
        prepare_candidate_fn=_prepare_subset,
        subset_max_arity=int(subset_max_arity),
        var_dims=var_dims,
        parent_stats=parent_stats,
        meta={"base_seed_count": int(len(base_blocks))},
    )
    return plan


def solve_direct_quadratic_preview_rows(
    spec: OperatorApplication,
    **kwargs,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    plan = build_direct_quadratic_search_plan(spec, **kwargs)
    rows, status, meta_out = execute_direct_search_plan(
        plan,
        x_fit=kwargs["x_fit"],
        y_fit=kwargs["y_fit"],
        x_probe=kwargs["x_probe"],
        y_probe=kwargs["y_probe"],
        max_depth=int(kwargs["max_depth"]),
        y_dims=kwargs["y_dims"],
        preview_topk=int(kwargs["preview_topk"]),
        deadline_s=kwargs.get("deadline_s", None),
        collect_direct_hole_candidates_fn=None,
    )
    for row in rows:
        row["direct_metadata"] = {
            **dict(row.get("direct_metadata", {}) or {}),
            "feature_node": row["expr"],
        }
    return rows, status, meta_out


def _bound_term_nodes(spec: Any) -> tuple[tuple, ...]:
    bound_closure = getattr(spec, "bound_closure", None)
    if hasattr(bound_closure, "bindings"):
        terms = list(dict(getattr(bound_closure, "bindings", {}) or {}).get("terms", ()) or ())
        out = [node for node in terms if isinstance(node, tuple) and is_valid_node(node)]
        if out:
            return tuple(out)
    bindings = dict(getattr(spec, "bindings", {}) or {})
    out = []
    for raw in list(bindings.get("terms", ()) or ()):
        node = getattr(raw, "node", raw)
        if isinstance(node, tuple) and is_valid_node(node):
            out.append(node)
    return tuple(out)


def _bound_closure_bindings(spec: Any) -> dict[str, Any]:
    bound_closure = getattr(spec, "bound_closure", None)
    if hasattr(bound_closure, "bindings"):
        return dict(getattr(bound_closure, "bindings", {}) or {})
    return {}


def _valid_bound_node(raw: Any) -> tuple | None:
    node = getattr(raw, "node", raw)
    if isinstance(node, tuple) and is_valid_node(node):
        return node
    return None


def _valid_bound_nodes(raw_values: Any) -> tuple[tuple, ...]:
    out: list[tuple] = []
    for raw in list(raw_values or ()):
        node = _valid_bound_node(raw)
        if node is not None:
            out.append(node)
    return tuple(out)


def _required_slot_names(bound_closure: Any) -> tuple[str, ...]:
    slot_specs = tuple(getattr(getattr(bound_closure, "spec", None), "slot_specs", ()) or ())
    return tuple(str(getattr(slot, "name", "") or "").strip() for slot in slot_specs if str(getattr(slot, "name", "") or "").strip())


def _present_slot_names(bound_closure: Any) -> tuple[str, ...]:
    bindings = dict(getattr(bound_closure, "bindings", {}) or {})
    present: list[str] = []
    for name, value in bindings.items():
        token = str(name or "").strip()
        if not token:
            continue
        if _valid_bound_node(value) is not None:
            present.append(token)
            continue
        if _valid_bound_nodes(value):
            present.append(token)
            continue
        if value is not None:
            present.append(token)
    return tuple(present)


def _bound_slots_complete(bound_closure: Any) -> bool:
    required = set(_required_slot_names(bound_closure))
    if not required:
        return False
    present = set(_present_slot_names(bound_closure))
    return required.issubset(present)


def _make_exact_bound_plan(
    *,
    candidate: PreparedClosureCandidate,
    var_dims,
    parent_stats: Mapping[str, int],
    meta: Mapping[str, Any] | None = None,
) -> PreparedCandidatesSearchPlan:
    merged_meta = {"execution_mode": "exact_bound", **dict(meta or {})}
    return PreparedCandidatesSearchPlan(
        candidates=(candidate,),
        var_dims=var_dims,
        parent_stats=parent_stats,
        meta=merged_meta,
    )


def _build_exact_bound_unary_linear_plan(
    spec: OperatorApplication,
    *,
    family: str,
    kind: str,
    wrap_op: str,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    var_dims,
) -> PreparedCandidatesSearchPlan | None:
    bindings = _bound_closure_bindings(spec)
    hole_node = _valid_bound_node(bindings.get("carrier"))
    if hole_node is None:
        return None
    kind_token = str(kind or "").strip().lower()
    wrap_token = str(wrap_op or "").strip().lower()
    family_token = str(family or "").strip().lower()
    anchor_node = _valid_bound_node(bindings.get("anchor"))
    if kind_token in {"add", "mul"} and anchor_node is None:
        return None
    feature_node = _valid_bound_node(bindings.get("feature")) or (wrap_token, hole_node)
    try:
        feature_fit = eval_node(feature_node, x_fit)
        feature_probe = eval_node(feature_node, x_probe)
    except Exception:
        return None
    if (not torch.is_tensor(feature_fit)) or (not torch.is_tensor(feature_probe)):
        return None
    if (not torch.isfinite(feature_fit).all()) or (not torch.isfinite(feature_probe).all()):
        return None
    anchor_fit = None
    anchor_probe = None
    if anchor_node is not None:
        try:
            anchor_fit = eval_node(anchor_node, x_fit)
            anchor_probe = eval_node(anchor_node, x_probe)
        except Exception:
            return None
        if (not torch.is_tensor(anchor_fit)) or (not torch.is_tensor(anchor_probe)):
            return None
        if (not torch.isfinite(anchor_fit).all()) or (not torch.isfinite(anchor_probe).all()):
            return None

    fit_cols = [feature_fit.squeeze(-1)]
    probe_cols = [feature_probe.squeeze(-1)]
    if kind_token == "add":
        fit_cols.append(anchor_fit.squeeze(-1))
        probe_cols.append(anchor_probe.squeeze(-1))
    elif kind_token == "mul":
        fit_cols[0] = feature_fit.squeeze(-1) * anchor_fit.squeeze(-1)
        probe_cols[0] = feature_probe.squeeze(-1) * anchor_probe.squeeze(-1)
    fit_cols.append(torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device))
    probe_cols.append(torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device))

    if kind_token == "mul":
        feature_expr = simplify(("mul", feature_node, anchor_node))
        terms = [feature_expr]
        bias_index = 1
    elif kind_token == "add":
        feature_expr = feature_node
        terms = [feature_expr, anchor_node]
        bias_index = 2
    else:
        feature_expr = feature_node
        terms = [feature_expr]
        bias_index = 1

    built = build_linear_wrap_candidate(
        family=family_token,
        scaffold_id=str(spec.scaffold_id),
        kind=kind_token,
        hole_node=hole_node,
        feature_node=feature_expr if kind_token == "mul" else feature_node,
        anchor_node=anchor_node,
        fit_matrix=torch.stack(fit_cols, dim=1),
        probe_matrix=torch.stack(probe_cols, dim=1),
        terms=terms,
        bias_index=int(bias_index),
        source="bound_carrier",
    )
    return _make_exact_bound_plan(
        candidate=PreparedClosureCandidate(
            built=built,
            candidate_subtree_node=hole_node,
        ),
        var_dims=var_dims,
        parent_stats=scaffold_parent_stats(spec),
        meta={"bound_slot_names": ["carrier", *([] if anchor_node is None else ["anchor"])]},
    )


def _rational_bound_nodes(spec: Any) -> tuple[tuple | None, tuple | None]:
    raw_bindings = dict(getattr(spec, "bindings", {}) or {})
    bound_bindings = _bound_closure_bindings(spec)
    meta = {**_spec_metadata(spec), **_bound_closure_metadata(spec)}
    composition_mode = str(meta.get("composition_mode", "") or "").strip().lower()
    anchor_node = getattr(spec, "anchor_node", None)
    anchor_node = anchor_node if isinstance(anchor_node, tuple) and is_valid_node(anchor_node) else None

    u_node = _valid_bound_node(raw_bindings.get("numerator"))
    v_node = _valid_bound_node(raw_bindings.get("denominator"))

    if u_node is None:
        maybe_u = _valid_bound_node(bound_bindings.get("numerator"))
        if maybe_u is not None and maybe_u != ("const", 0.0):
            u_node = maybe_u
    if v_node is None:
        maybe_v = _valid_bound_node(bound_bindings.get("denominator"))
        if maybe_v is not None and maybe_v != ("const", 0.0):
            v_node = maybe_v

    if composition_mode == "denominator_companion" and v_node is None and anchor_node is not None:
        v_node = anchor_node
    if composition_mode == "numerator_companion" and u_node is None and anchor_node is not None:
        u_node = anchor_node

    return u_node, v_node


def _build_exact_bound_rational_plan(
    spec: OperatorApplication,
    *,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    max_depth: int,
    var_dims,
    y_dims,
    safe_eps: float,
) -> PreparedCandidatesSearchPlan | None:
    u_node, v_node = _rational_bound_nodes(spec)
    if u_node is None or v_node is None:
        return None
    try:
        u_fit = eval_node(u_node, x_fit)
        u_probe = eval_node(u_node, x_probe)
        v_fit = eval_node(v_node, x_fit)
        v_probe = eval_node(v_node, x_probe)
    except Exception:
        return None
    tensors = (u_fit, u_probe, v_fit, v_probe)
    if any((not torch.is_tensor(value)) for value in tensors):
        return None
    if any((not torch.isfinite(value).all()) for value in tensors):
        return None
    built = build_rational_affine_candidate(
        scaffold_id=str(spec.scaffold_id),
        u_node=u_node,
        v_node=v_node,
        u_fit=u_fit,
        u_probe=u_probe,
        v_fit=v_fit,
        v_probe=v_probe,
        max_depth=int(max_depth),
        var_dims=var_dims,
        y_dims=y_dims,
        safe_eps=float(safe_eps),
        u_source="bound_numerator",
        v_source="bound_denominator",
    )
    return _make_exact_bound_plan(
        candidate=PreparedClosureCandidate(
            built=built,
            candidate_subtree_node=u_node,
        ),
        var_dims=var_dims,
        parent_stats=scaffold_parent_stats(spec),
        meta={"bound_slot_names": ["numerator", "denominator"]},
    )


def _build_exact_bound_power_plan(
    spec: OperatorApplication,
    *,
    power_kind: str,
    exponent: float,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    max_depth: int,
    var_dims,
    y_dims,
    safe_eps: float,
) -> PreparedCandidatesSearchPlan | None:
    bindings = _bound_closure_bindings(spec)
    hole_node = _valid_bound_node(bindings.get("carrier"))
    if hole_node is None:
        return None
    anchor_node = _valid_bound_node(bindings.get("anchor"))
    if str(power_kind) in {"sqrt_mul", "invsqrt_mul", "inv_mul", "neg2_mul", "sqr_mul"} and anchor_node is None:
        return None
    try:
        h_fit = eval_node(hole_node, x_fit)
        h_probe = eval_node(hole_node, x_probe)
    except Exception:
        return None
    if (not torch.is_tensor(h_fit)) or (not torch.is_tensor(h_probe)):
        return None
    if (not torch.isfinite(h_fit).all()) or (not torch.isfinite(h_probe).all()):
        return None
    anchor_fit = None
    anchor_probe = None
    if anchor_node is not None:
        try:
            anchor_fit = eval_node(anchor_node, x_fit)
            anchor_probe = eval_node(anchor_node, x_probe)
        except Exception:
            return None
        if (not torch.is_tensor(anchor_fit)) or (not torch.is_tensor(anchor_probe)):
            return None
        if (not torch.isfinite(anchor_fit).all()) or (not torch.isfinite(anchor_probe).all()):
            return None
    variants = (None,)
    if float(exponent) == 2.0:
        variants = ("square_only", "bias_square", "linear_square", "full_quadratic")
    candidates: list[PreparedClosureCandidate] = []
    for power_variant in variants:
        built = build_affine_power_candidate(
            scaffold_id=str(spec.scaffold_id),
            power_kind=str(power_kind),
            exponent=float(exponent),
            hole_node=hole_node,
            anchor_node=anchor_node,
            h_fit=h_fit,
            h_probe=h_probe,
            anchor_fit=anchor_fit,
            anchor_probe=anchor_probe,
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            safe_eps=float(safe_eps),
            source="bound_carrier",
            power_variant=power_variant,
        )
        candidates.append(
            PreparedClosureCandidate(
                built=built,
                candidate_subtree_node=hole_node,
                row_patch={
                    "local_mapping_nparams": int(
                        direct_power_variant_nparams(power_variant, exponent=float(exponent))
                    ),
                    "direct_metadata": {
                        "feature_node": built.direct_metadata.get("hole_node"),
                        "power_variant": str(power_variant or ""),
                    },
                },
            )
        )
    return PreparedCandidatesSearchPlan(
        candidates=tuple(candidates),
        var_dims=var_dims,
        parent_stats=scaffold_parent_stats(spec),
        meta={
            "execution_mode": "exact_bound",
            "bound_slot_names": ["carrier", *([] if anchor_node is None else ["anchor"])],
        },
    )


def _build_exact_bound_quadratic_plan(
    spec: OperatorApplication,
    *,
    quadratic_kind: str,
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    max_depth: int,
    var_dims,
    y_dims,
    safe_eps: float,
) -> PreparedCandidatesSearchPlan | None:
    bindings = _bound_closure_bindings(spec)
    base_nodes = _valid_bound_nodes(bindings.get("bases"))
    if not base_nodes:
        return None
    anchor_node = _valid_bound_node(bindings.get("anchor"))
    if str(quadratic_kind) == "sqrt_mul" and anchor_node is None:
        return None
    base_fit_cols: list[torch.Tensor] = []
    base_probe_cols: list[torch.Tensor] = []
    for node in base_nodes:
        try:
            fit_val = eval_node(node, x_fit)
            probe_val = eval_node(node, x_probe)
        except Exception:
            return None
        if (not torch.is_tensor(fit_val)) or (not torch.is_tensor(probe_val)):
            return None
        if (not torch.isfinite(fit_val).all()) or (not torch.isfinite(probe_val).all()):
            return None
        base_fit_cols.append(fit_val)
        base_probe_cols.append(probe_val)
    quad_fit = torch.stack([col.squeeze(-1) ** 2 for col in base_fit_cols], dim=1)
    quad_probe = torch.stack([col.squeeze(-1) ** 2 for col in base_probe_cols], dim=1)
    anchor_fit = None
    anchor_probe = None
    if anchor_node is not None:
        try:
            anchor_fit = eval_node(anchor_node, x_fit)
            anchor_probe = eval_node(anchor_node, x_probe)
        except Exception:
            return None
        if (not torch.is_tensor(anchor_fit)) or (not torch.is_tensor(anchor_probe)):
            return None
        if (not torch.isfinite(anchor_fit).all()) or (not torch.isfinite(anchor_probe).all()):
            return None
    built = build_quadratic_sqrt_candidate(
        scaffold_id=str(spec.scaffold_id),
        quadratic_kind=str(quadratic_kind),
        base_nodes=tuple(base_nodes),
        anchor_node=anchor_node,
        quad_fit=quad_fit,
        quad_probe=quad_probe,
        anchor_fit=anchor_fit,
        anchor_probe=anchor_probe,
        max_depth=int(max_depth),
        var_dims=var_dims,
        y_dims=y_dims,
        safe_eps=float(safe_eps),
        base_sources=["bound_base" for _ in base_nodes],
    )
    quad_terms = [simplify(("sqr", node)) for node in base_nodes]
    quad_latent_node = quad_terms[0]
    for term in quad_terms[1:]:
        quad_latent_node = simplify(("add", quad_latent_node, term))
    candidate_subtree_node = (
        base_nodes[0]
        if len(base_nodes) == 1
        else simplify(("add", *base_nodes[:2])) if len(base_nodes) >= 2 else None
    )
    return _make_exact_bound_plan(
        candidate=PreparedClosureCandidate(
            built=built,
            candidate_subtree_node=candidate_subtree_node,
            row_patch={"direct_metadata": {"quadratic_latent_node": quad_latent_node}},
        ),
        var_dims=var_dims,
        parent_stats=scaffold_parent_stats(spec),
        meta={"bound_slot_names": ["bases", *([] if anchor_node is None else ["anchor"])]},
    )


def build_exact_bound_plan(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None = None,
) -> Any:
    bound_closure = getattr(spec, "bound_closure", None)
    if bound_closure is None or not _bound_slots_complete(bound_closure):
        return None
    planner_id = _planner_id_from_bound_closure(spec)
    if planner_id == "affine_latent":
        plan = build_direct_affine_search_plan(
            spec,
            nvars=int(nvars),
            max_depth=int(max_depth),
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            var_dims=var_dims,
            y_dims=y_dims,
            pool_nodes=pool_nodes,
            pool_dims=pool_dims,
            preview_topk=int(preview_topk),
            solver_kwargs=solver_kwargs,
            deadline_s=deadline_s,
        )
        if isinstance(plan, PreparedCandidatesSearchPlan):
            return PreparedCandidatesSearchPlan(
                candidates=tuple(plan.candidates),
                var_dims=plan.var_dims,
                parent_stats=plan.parent_stats,
                meta={"execution_mode": "exact_bound", **dict(plan.meta or {})},
            )
        return plan
    if planner_id == "harmonic_wrap":
        return build_exact_bound_periodic_search_plan(
            spec,
            max_depth=int(max_depth),
            x_fit=x_fit,
            x_probe=x_probe,
            var_dims=var_dims,
            y_dims=y_dims,
        )
    if planner_id == "linear_wrap":
        algebra = resolve_linear_wrap_operator_algebra(spec)
        wrap_kind = _direct_linear_wrap_scaffold_kind(spec, algebra=algebra) if algebra is not None else None
        if algebra is None or wrap_kind is None:
            return None
        return _build_exact_bound_unary_linear_plan(
            spec,
            family=str(algebra.family),
            kind=str(wrap_kind),
            wrap_op=str(algebra.wrap_op),
            x_fit=x_fit,
            x_probe=x_probe,
            var_dims=var_dims,
        )
    if planner_id == "power_wrap":
        power_kind = direct_power_scaffold_kind(spec)
        power_map = {
            "sqrt": 0.5,
            "sqrt_mul": 0.5,
            "invsqrt": -0.5,
            "invsqrt_mul": -0.5,
            "inv": -1.0,
            "inv_mul": -1.0,
            "neg2": -2.0,
            "neg2_mul": -2.0,
            "sqr": 2.0,
            "sqr_mul": 2.0,
        }
        if power_kind in power_map:
            return _build_exact_bound_power_plan(
                spec,
                power_kind=str(power_kind),
                exponent=float(power_map[str(power_kind)]),
                x_fit=x_fit,
                x_probe=x_probe,
                max_depth=int(max_depth),
                var_dims=var_dims,
                y_dims=y_dims,
                safe_eps=float(solver_kwargs.get("safe_eps", 1.0e-8)),
            )
        return None
    if planner_id == "quadratic_wrap":
        quadratic_kind = direct_quadratic_scaffold_kind(spec)
        if quadratic_kind in {"sqrt", "sqrt_mul"}:
            return _build_exact_bound_quadratic_plan(
                spec,
                quadratic_kind=str(quadratic_kind),
                x_fit=x_fit,
                x_probe=x_probe,
                max_depth=int(max_depth),
                var_dims=var_dims,
                y_dims=y_dims,
                safe_eps=float(solver_kwargs.get("safe_eps", 1.0e-8)),
            )
        return None
    if planner_id == "fractional_head":
        return _build_exact_bound_rational_plan(
            spec,
            x_fit=x_fit,
            x_probe=x_probe,
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            safe_eps=float(solver_kwargs.get("safe_eps", 1.0e-6)),
        )
    return None


def build_direct_affine_search_plan(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None = None,
) -> Any:
    if direct_affine_scaffold_kind(spec) is None and str(getattr(spec, "operator_kind", "") or "").strip().lower() != "affine_latent":
        return _static_status_plan("direct_affine_unsupported_form")

    term_nodes = _bound_term_nodes(spec)
    if not term_nodes:
        return _static_status_plan("direct_no_affine_terms")

    fit_cols: list[torch.Tensor] = []
    probe_cols: list[torch.Tensor] = []
    for node in term_nodes:
        try:
            fit_val = eval_node(node, x_fit)
            probe_val = eval_node(node, x_probe)
        except Exception:
            return _static_status_plan("direct_affine_eval_failed")
        if (not torch.is_tensor(fit_val)) or (not torch.is_tensor(probe_val)):
            return _static_status_plan("direct_affine_eval_failed")
        if (not torch.isfinite(fit_val).all()) or (not torch.isfinite(probe_val).all()):
            return _static_status_plan("direct_affine_nonfinite")
        fit_cols.append(fit_val.squeeze(-1))
        probe_cols.append(probe_val.squeeze(-1))

    fit_cols.append(torch.ones(int(x_fit.shape[0]), dtype=x_fit.dtype, device=x_fit.device))
    probe_cols.append(torch.ones(int(x_probe.shape[0]), dtype=x_probe.dtype, device=x_probe.device))

    parent_stats = scaffold_parent_stats(spec)
    built = build_affine_latent_candidate(
        scaffold_id=str(spec.scaffold_id),
        term_nodes=term_nodes,
        fit_matrix=torch.stack(fit_cols, dim=1),
        probe_matrix=torch.stack(probe_cols, dim=1),
        source="bound_terms",
    )
    plan = PreparedCandidatesSearchPlan(
        candidates=(
            PreparedClosureCandidate(
                built=built,
                candidate_subtree_node=getattr(spec, "parent_node", None),
            ),
        ),
        var_dims=var_dims,
        parent_stats=parent_stats,
        meta={"term_count": int(len(term_nodes))},
    )
    return plan


def solve_direct_affine_preview_rows(
    spec: OperatorApplication,
    **kwargs,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    plan = build_direct_affine_search_plan(spec, **kwargs)
    return execute_direct_search_plan(
        plan,
        x_fit=kwargs["x_fit"],
        y_fit=kwargs["y_fit"],
        x_probe=kwargs["x_probe"],
        y_probe=kwargs["y_probe"],
        max_depth=int(kwargs["max_depth"]),
        y_dims=kwargs["y_dims"],
        preview_topk=int(kwargs["preview_topk"]),
        deadline_s=kwargs.get("deadline_s", None),
        collect_direct_hole_candidates_fn=None,
    )


def solve_direct_exp_preview_rows(
    spec: OperatorApplication,
    **kwargs,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    algebra = _linear_wrap_algebra_by_family("exp")
    if algebra is None:
        return [], "direct_exp_unsupported_form", {}
    return solve_direct_linear_wrap_preview_rows(spec, algebra=algebra, **kwargs)


def build_direct_linear_wrap_search_plan(
    spec: OperatorApplication,
    *,
    algebra: LinearWrapOperatorAlgebraSpec,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn=collect_direct_hole_candidates,
) -> Any:
    family = str(algebra.family or "").strip().lower()
    wrap_kind = _direct_linear_wrap_scaffold_kind(spec, algebra=algebra)
    if wrap_kind is None:
        return _static_status_plan(f"direct_{family}_unsupported_form")
    return build_direct_unary_linear_search_plan(
        spec,
        family=family,
        kind=str(wrap_kind),
        wrap_op=str(algebra.wrap_op),
        hole_transform_fn=algebra.hole_transform_fn,
        nvars=int(nvars),
        max_depth=int(max_depth),
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=var_dims,
        y_dims=y_dims,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        preview_topk=int(preview_topk),
        solver_kwargs=solver_kwargs,
        deadline_s=deadline_s,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
    )


def solve_direct_linear_wrap_preview_rows(
    spec: OperatorApplication,
    *,
    algebra: LinearWrapOperatorAlgebraSpec,
    **kwargs,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    plan = build_direct_linear_wrap_search_plan(spec, algebra=algebra, **kwargs)
    return execute_direct_search_plan(
        plan,
        x_fit=kwargs["x_fit"],
        y_fit=kwargs["y_fit"],
        x_probe=kwargs["x_probe"],
        y_probe=kwargs["y_probe"],
        max_depth=int(kwargs["max_depth"]),
        y_dims=kwargs["y_dims"],
        preview_topk=int(kwargs["preview_topk"]),
        deadline_s=kwargs.get("deadline_s", None),
        collect_direct_hole_candidates_fn=kwargs.get("collect_direct_hole_candidates_fn", collect_direct_hole_candidates),
    )


def solve_direct_log_preview_rows(
    spec: OperatorApplication,
    **kwargs,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    algebra = _linear_wrap_algebra_by_family("log")
    if algebra is None:
        return [], "direct_log_unsupported_form", {}
    return solve_direct_linear_wrap_preview_rows(spec, algebra=algebra, **kwargs)


def _common_direct_solver_kwargs(
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None,
) -> dict[str, Any]:
    return {
        "nvars": int(nvars),
        "max_depth": int(max_depth),
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_probe": x_probe,
        "y_probe": y_probe,
        "var_dims": var_dims,
        "y_dims": y_dims,
        "pool_nodes": pool_nodes,
        "pool_dims": pool_dims,
        "preview_topk": int(preview_topk),
        "solver_kwargs": solver_kwargs,
        "deadline_s": deadline_s,
    }


def _build_direct_periodic_search_plan(
    spec: OperatorApplication,
    **kwargs,
) -> Any:
    return build_periodic_search_plan(spec, **kwargs)


def _build_direct_generic_linear_wrap_plan(
    spec: OperatorApplication,
    *,
    collect_direct_hole_candidates_fn=collect_direct_hole_candidates,
    **kwargs,
) -> Any:
    algebra = resolve_linear_wrap_operator_algebra(spec)
    if algebra is None:
        return _static_status_plan(
            "direct_not_supported",
            {
                "execution_mode": "slot_search",
                "planner_id": "linear_wrap",
                "status_reason": "linear_wrap_algebra_unresolved",
            },
        )
    return build_direct_linear_wrap_search_plan(
        spec,
        algebra=algebra,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        **kwargs,
    )


def _build_direct_affine_plan(
    spec: OperatorApplication,
    **kwargs,
) -> Any:
    kwargs.pop("collect_direct_hole_candidates_fn", None)
    return build_direct_affine_search_plan(spec, **kwargs)


def _build_direct_power_plan(
    spec: OperatorApplication,
    *,
    collect_direct_hole_candidates_fn=collect_direct_hole_candidates,
    **kwargs,
) -> Any:
    return build_direct_power_search_plan(
        spec,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        **kwargs,
    )


def _build_direct_quadratic_plan(
    spec: OperatorApplication,
    **kwargs,
) -> Any:
    kwargs.pop("collect_direct_hole_candidates_fn", None)
    return build_direct_quadratic_search_plan(spec, **kwargs)


def _build_direct_rational_plan(
    spec: OperatorApplication,
    *,
    collect_direct_hole_candidates_fn=collect_direct_hole_candidates,
    **kwargs,
) -> Any:
    single_term_plan = build_direct_rational_affine_search_plan(
        spec,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        **kwargs,
    )

    def _run_rational_with_multi_term_fallback(
        **run_kwargs,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        # Execute single-term plan first
        single_started = time.perf_counter()
        rows, status, meta = execute_direct_search_plan(
            single_term_plan,
            x_fit=kwargs["x_fit"],
            y_fit=kwargs["y_fit"],
            x_probe=kwargs["x_probe"],
            y_probe=kwargs["y_probe"],
            max_depth=int(kwargs["max_depth"]),
            y_dims=kwargs["y_dims"],
            preview_topk=int(kwargs["preview_topk"]),
            deadline_s=kwargs.get("deadline_s", None),
            collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        )
        single_elapsed_s = float(time.perf_counter() - single_started)
        # Check if multi-term fallback should run
        single_best_mse = _best_preview_rows_mse(rows)
        single_best_norm = _normalized_preview_rows_mse(rows, y_ref=kwargs["y_probe"])
        solver_kwargs = dict(kwargs.get("solver_kwargs", {}) or {})
        threshold = _mt_kw_float(
            solver_kwargs,
            "multi_term_rational_threshold",
            1.0e-8,
            "NESTY_MULTI_TERM_RATIONAL_THRESHOLD",
        )
        meta_out = dict(meta or {})
        meta_out["timing_rational_single_s"] = float(single_elapsed_s)
        meta_out["multi_term_rational_gate_metric"] = "normalized_probe_mse"
        meta_out["multi_term_rational_gate_threshold"] = float(threshold)
        meta_out["multi_term_rational_single_best_mse"] = float(single_best_mse)
        meta_out["multi_term_rational_single_best_norm"] = float(single_best_norm)
        if not _mt_kw_bool(
            solver_kwargs,
            "multi_term_rational_enable",
            True,
            "NESTY_MULTI_TERM_RATIONAL_ENABLE",
        ):
            meta_out["multi_term_rational_status"] = "disabled"
            return rows, status, meta_out
        if single_best_norm <= threshold:
            meta_out["multi_term_rational_status"] = "skipped_below_threshold"
            return rows, status, meta_out
        gate_mode = _mt_kw_str(
            solver_kwargs,
            "multi_term_rational_gate_mode",
            "legacy",
            "NESTY_MULTI_TERM_RATIONAL_GATE_MODE",
        ).strip().lower()
        if gate_mode in {"single_quality", "near_miss"}:
            gate_max_norm = _mt_kw_float(
                solver_kwargs,
                "multi_term_rational_gate_max_norm",
                float("inf"),
                "NESTY_MULTI_TERM_RATIONAL_GATE_MAX_NORM",
            )
            meta_out["multi_term_rational_gate_mode"] = str(gate_mode)
            meta_out["multi_term_rational_gate_max_norm"] = float(gate_max_norm)
            if single_best_norm > gate_max_norm:
                meta_out["multi_term_rational_status"] = "skipped_single_quality"
                return rows, status, meta_out
        # Run multi-term fallback
        try:
            multi_started = time.perf_counter()
            mt_rows, mt_status, mt_meta = _run_multi_term_rational_fallback(
                spec,
                single_term_best_mse=float(single_best_mse),
                single_term_best_rows=rows,
                collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
                **kwargs,
            )
            meta_out["timing_rational_multi_s"] = float(time.perf_counter() - multi_started)
        except Exception as exc:
            meta_out["multi_term_rational_error"] = f"{type(exc).__name__}: {exc}"
            return rows, status, meta_out
        meta_out["multi_term_rational_status"] = str(mt_status)
        meta_out["multi_term_rational_meta"] = dict(mt_meta or {})
        if mt_rows:
            mt_best = _best_preview_rows_mse(mt_rows)
            if mt_best < single_best_mse:
                merged = rows + mt_rows
                merged.sort(
                    key=lambda r: (
                        _best_preview_rows_mse([r]),
                        int(r.get("candidate_child_size", 0) or 0),
                        str(r.get("proposal_key", "") or ""),
                    )
                )
                return merged, status, meta_out
        return rows, status, meta_out

    return CustomDirectSearchPlan(
        run_fn=_run_rational_with_multi_term_fallback,
        kwargs={},
    )


DIRECT_OPERATOR_PLANNERS: tuple[DirectOperatorPlanner, ...] = (
    DirectOperatorPlanner(
        planner_id="harmonic_wrap",
        operator_kinds=("harmonic_wrap",),
        composition_modes=("base", "companion", "prefactor"),
        matcher=lambda spec: _operator_kind_token(spec) == "harmonic_wrap"
        or direct_periodic_scaffold_kind(spec) is not None,
        plan_builder=_build_direct_periodic_search_plan,
    ),
    DirectOperatorPlanner(
        planner_id="linear_wrap",
        operator_kinds=("unary_wrap", "anchored_unary_wrap"),
        composition_modes=("base", "companion", "prefactor"),
        matcher=_linear_wrap_planner_matcher,
        plan_builder=_build_direct_generic_linear_wrap_plan,
    ),
    DirectOperatorPlanner(
        planner_id="affine_latent",
        operator_kinds=("affine_latent",),
        composition_modes=("latent",),
        matcher=lambda spec: _operator_kind_token(spec) == "affine_latent"
        or direct_affine_scaffold_kind(spec) is not None,
        plan_builder=_build_direct_affine_plan,
    ),
    DirectOperatorPlanner(
        planner_id="power_wrap",
        operator_kinds=("power_wrap",),
        composition_modes=("base", "prefactor"),
        matcher=lambda spec: _operator_kind_token(spec) == "power_wrap"
        or direct_power_scaffold_kind(spec) is not None,
        plan_builder=_build_direct_power_plan,
    ),
    DirectOperatorPlanner(
        planner_id="quadratic_wrap",
        operator_kinds=("quadratic_wrap",),
        composition_modes=("base", "prefactor"),
        matcher=lambda spec: _operator_kind_token(spec) == "quadratic_wrap"
        or direct_quadratic_scaffold_kind(spec) is not None,
        plan_builder=_build_direct_quadratic_plan,
    ),
    DirectOperatorPlanner(
        planner_id="fractional_head",
        operator_kinds=("fractional_head",),
        composition_modes=("fractional", "denominator_companion", "numerator_companion"),
        matcher=lambda spec: _operator_kind_token(spec) == "fractional_head"
        or direct_rational_scaffold_kind(spec) is not None,
        plan_builder=_build_direct_rational_plan,
    ),
)


def _planner_id_from_bound_closure(spec: Any) -> str | None:
    operator_kind = _operator_kind_token(spec)
    if operator_kind == "harmonic_wrap":
        return "harmonic_wrap"
    if operator_kind in {"unary_wrap", "anchored_unary_wrap"}:
        return "linear_wrap"
    if operator_kind == "affine_latent":
        return "affine_latent"
    if operator_kind == "power_wrap":
        return "power_wrap"
    if operator_kind == "quadratic_wrap":
        return "quadratic_wrap"
    if operator_kind == "fractional_head":
        return "fractional_head"
    solver = _bound_head_solver(spec)
    family = _bound_family(spec) or _family_token(spec)
    if solver == "harmonic_linear":
        return "harmonic_wrap"
    if solver == "fractional_linear":
        return "fractional_head"
    if solver == "discrete_power":
        return "power_wrap"
    if solver == "quadratic_sqrt":
        return "quadratic_wrap"
    linear_wrap_families = {str(algebra.family) for algebra in LINEAR_WRAP_OPERATOR_ALGEBRAS}
    if solver == "linear" and family in ({"affine"} | linear_wrap_families):
        if family == "affine":
            return "affine_latent"
        if family in linear_wrap_families:
            return "linear_wrap"
    return None


def resolve_direct_operator_planner(spec: Any) -> DirectOperatorPlanner | None:
    hinted_id = _planner_id_from_bound_closure(spec)
    if hinted_id:
        for planner in DIRECT_OPERATOR_PLANNERS:
            if str(getattr(planner, "planner_id", "") or "").strip().lower() == hinted_id:
                return planner
    for planner in DIRECT_OPERATOR_PLANNERS:
        try:
            if bool(planner.matcher(spec)):
                return planner
        except Exception:
            continue
    return None


def solve_direct_operator_preview_rows(
    spec: OperatorApplication,
    *,
    nvars: int,
    max_depth: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_dims,
    preview_topk: int,
    solver_kwargs: Mapping[str, Any],
    proposal_context: ProposalContext | None = None,
    deadline_s: float | None = None,
    collect_direct_hole_candidates_fn=collect_direct_hole_candidates,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    allow_slot_rebinding = bool(dict(solver_kwargs or {}).get("allow_slot_rebinding", False))
    effective_allow_slot_rebinding = bool(allow_slot_rebinding)
    exact_result: tuple[list[dict[str, Any]], str, dict[str, Any]] | None = None
    exact_plan = build_exact_bound_plan(
        spec,
        **_common_direct_solver_kwargs(
            nvars=nvars,
            max_depth=max_depth,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            var_dims=var_dims,
            y_dims=y_dims,
            pool_nodes=pool_nodes,
            pool_dims=pool_dims,
            preview_topk=preview_topk,
            solver_kwargs=solver_kwargs,
            deadline_s=deadline_s,
        ),
    )
    if exact_plan is not None:
        exact_rows, exact_status, exact_meta = execute_direct_search_plan(
            exact_plan,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            max_depth=int(max_depth),
            y_dims=y_dims,
            preview_topk=int(preview_topk),
            deadline_s=deadline_s,
            collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        )
        exact_meta_out = dict(exact_meta or {})
        exact_meta_out.setdefault("execution_mode", "exact_bound")
        exact_result = (exact_rows, exact_status, exact_meta_out)
        if not allow_slot_rebinding:
            if (not exact_rows) and _power_or_rational_direct_spec(spec):
                effective_allow_slot_rebinding = True
            elif exact_rows and _should_force_slot_rebinding(
                spec,
                exact_rows=exact_rows,
                solver_kwargs=solver_kwargs,
            ):
                effective_allow_slot_rebinding = True
        if exact_rows and not effective_allow_slot_rebinding:
            return exact_result

    planner = resolve_direct_operator_planner(spec)
    if planner is None:
        if exact_result is not None:
            return exact_result
        return [], "direct_not_supported", {}
    effective_solver_kwargs = dict(solver_kwargs or {})
    effective_solver_kwargs["allow_slot_rebinding"] = bool(effective_allow_slot_rebinding)
    plan_kwargs = _common_direct_solver_kwargs(
            nvars=nvars,
            max_depth=max_depth,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            var_dims=var_dims,
            y_dims=y_dims,
            pool_nodes=pool_nodes,
            pool_dims=pool_dims,
            preview_topk=preview_topk,
            solver_kwargs=effective_solver_kwargs,
            deadline_s=deadline_s,
        )
    if _periodic_direct_spec(spec):
        plan_kwargs["proposal_context"] = proposal_context
    plan = planner.plan_builder(
        spec,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        **plan_kwargs,
    )
    rows, status, meta = execute_direct_search_plan(
        plan,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        max_depth=int(max_depth),
        y_dims=y_dims,
        preview_topk=int(preview_topk),
        deadline_s=deadline_s,
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
    )
    meta_out = dict(meta or {})
    meta_out.setdefault("execution_mode", "slot_search")
    # A planner that scored no slot candidate hands back its own exact-bound rows
    # labelled "exact_bound". Those are the rows already in exact_result, so merging
    # would duplicate them and misreport the run as a slot search.
    planner_returned_exact = str(meta_out.get("execution_mode", "")) == "exact_bound"
    if rows and exact_result is not None and exact_result[0] and not planner_returned_exact:
        exact_rows, _exact_status, exact_meta = exact_result
        merged_meta = {
            **dict(exact_meta or {}),
            **dict(meta_out),
            "execution_mode": "slot_search+exact_bound",
            "exact_candidate_count_raw": int(dict(exact_meta or {}).get("candidate_count_raw", len(exact_rows))),
            "exact_candidate_count_scored": int(dict(exact_meta or {}).get("candidate_count_scored", len(exact_rows))),
            "slot_candidate_count_raw": int(dict(meta_out or {}).get("candidate_count_raw", len(rows))),
            "slot_candidate_count_scored": int(dict(meta_out or {}).get("candidate_count_scored", len(rows))),
        }
        merged_rows, merged_status, merged_meta_out = finalize_direct_preview_rows(
            list(exact_rows) + list(rows),
            preview_topk=int(preview_topk),
            raw_candidate_count=int(merged_meta["exact_candidate_count_raw"]) + int(merged_meta["slot_candidate_count_raw"]),
            scored_candidate_count=int(merged_meta["exact_candidate_count_scored"]) + int(merged_meta["slot_candidate_count_scored"]),
            deadline_s=deadline_s,
            meta=merged_meta,
        )
        if merged_rows:
            return merged_rows, merged_status, merged_meta_out
    if rows:
        return rows, status, meta_out
    if exact_result is not None:
        return exact_result
    return rows, status, meta_out


__all__ = [
    "DirectOperatorPlanner",
    "LinearWrapOperatorAlgebraSpec",
    "DIRECT_OPERATOR_PLANNERS",
    "LINEAR_WRAP_OPERATOR_ALGEBRAS",
    "direct_affine_scaffold_kind",
    "direct_exp_scaffold_kind",
    "direct_log_scaffold_kind",
    "resolve_direct_operator_planner",
    "resolve_linear_wrap_operator_algebra",
    "solve_direct_linear_wrap_preview_rows",
    "solve_direct_operator_preview_rows",
    "direct_power_scaffold_kind",
    "direct_periodic_scaffold_kind",
    "direct_quadratic_scaffold_kind",
    "direct_rational_scaffold_kind",
    "solve_direct_affine_preview_rows",
    "solve_direct_exp_preview_rows",
    "solve_direct_log_preview_rows",
    "solve_direct_power_preview_rows",
    "solve_direct_periodic_add_preview_rows",
    "solve_direct_quadratic_preview_rows",
    "solve_direct_rational_affine_preview_rows",
]
