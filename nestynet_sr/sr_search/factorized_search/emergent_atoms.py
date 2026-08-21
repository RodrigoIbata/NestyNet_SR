# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Auxiliary atom harvesting for factorized symbolic search.

This module keeps the core FSS proposal machinery intact. It observes good
closure-search candidates, harvests reusable subexpressions, and converts the
accepted atoms into explicit SeedBlocks for the next FSS round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch

from .basis_state import BasisState, FeatureBlock
from .expr_ast import (
    dims_eq,
    eval_node,
    is_valid_node,
    node_depth,
    node_dims,
    node_size,
    node_str,
    simplify,
)
from .proposal_families.common import dim0
from .proposal_families.seed_blocks import SeedBlock, make_seed_block


_DIRECT_METADATA_NODE_KEYS = (
    "feature_node",
    "hole_node",
    "power_inner_node",
    "envelope_node",
    "companion_node",
    "u_node",
    "v_node",
    "quadratic_latent_node",
    "anchor_lift_node",
)
_DIRECT_METADATA_NODE_LIST_KEYS = (
    "term_nodes",
    "harmonic_feature_nodes",
    "companion_nodes",
    "quadratic_base_nodes",
)

_ALLOWED_ROOTS = {"add", "sub", "mul", "sqrt", "sqr", "sin", "cos"}
_DISALLOWED_ROOTS = {"div", "log", "exp", "tan", "asin", "acos", "atan"}
_ROLE_BUCKET_ORDER = (
    "target_term",
    "completed_modulator",
    "pure_dimensional_carrier",
    "envelope_prefactor",
    "numerator_dimensional",
    "carrier_argument",
)


@dataclass(frozen=True)
class EmergentAtom:
    node: tuple
    dim: tuple[float, ...] | None
    kind: str
    score: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
    source_count: int = 0
    roles: tuple[str, ...] = ()
    families: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "expr": str(node_str(self.node)),
            "dim": None if self.dim is None else [float(v) for v in self.dim],
            "kind": str(self.kind),
            "score": float(self.score),
            "source_count": int(self.source_count),
            "roles": [str(v) for v in tuple(self.roles or ())],
            "families": [str(v) for v in tuple(self.families or ())],
            "evidence": _jsonish(self.evidence),
        }


@dataclass
class _AtomEvidence:
    key: str
    node: tuple
    dim: tuple[float, ...] | None
    kind: str
    sources: set[str] = field(default_factory=set)
    parent_exprs: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    families: set[str] = field(default_factory=set)
    sightings: int = 0
    best_parent_probe: float = math.inf
    best_parent_fit: float = math.inf
    best_fit_mse: float = math.inf
    best_probe_mse: float = math.inf
    best_fit_gain_rel: float = 0.0
    best_probe_gain_rel: float = 0.0
    rational_derived: bool = False
    common_denominator_stripped: bool = False
    active_vars: tuple[int, ...] = ()
    active_dimensionless_vars: tuple[int, ...] = ()
    active_dimensional_vars: tuple[int, ...] = ()
    root_op: str = ""


def _jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonish(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    if isinstance(value, torch.Tensor):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _valid_node(node: Any) -> tuple | None:
    if isinstance(node, tuple) and is_valid_node(node):
        return node
    return None


def _as_col_tensor(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    out = value.detach()
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    elif out.ndim >= 2:
        out = out.reshape(out.shape[0], -1)
    else:
        return None
    if out.ndim != 2 or out.shape[0] <= 0:
        return None
    if out.shape[1] > 1:
        out = out[:, :1]
    if not out.is_floating_point():
        out = out.to(dtype=torch.float64)
    return out.to(dtype=torch.float64)


def _safe_eval_node_col(node: tuple, x: torch.Tensor | None) -> torch.Tensor | None:
    if not isinstance(x, torch.Tensor):
        return None
    valid = _valid_node(node)
    if valid is None:
        return None
    try:
        out = eval_node(valid, x)
    except Exception:
        return None
    col = _as_col_tensor(out)
    if col is None or col.shape[0] != x.shape[0]:
        return None
    return col


def _finite_mask(*cols: torch.Tensor | None) -> torch.Tensor | None:
    valid_cols = [col for col in cols if isinstance(col, torch.Tensor)]
    if not valid_cols:
        return None
    mask = torch.ones((valid_cols[0].shape[0],), dtype=torch.bool, device=valid_cols[0].device)
    for col in valid_cols:
        if col.ndim != 2 or col.shape[1] != 1 or col.shape[0] != mask.shape[0]:
            return None
        mask &= torch.isfinite(col.squeeze(-1))
    return mask


def _col_variance(col: torch.Tensor, mask: torch.Tensor | None) -> float:
    if mask is None or int(mask.sum().item()) <= 1:
        return 0.0
    vals = col[mask].to(dtype=torch.float64)
    try:
        return float(torch.var(vals, unbiased=False).item())
    except Exception:
        return 0.0


def _constant_baseline_losses(
    *,
    y_fit: torch.Tensor | None,
    y_probe: torch.Tensor | None,
) -> tuple[float, float]:
    y_fit_col = _as_col_tensor(y_fit)
    y_probe_col = _as_col_tensor(y_probe)
    if y_fit_col is None or y_probe_col is None:
        return math.inf, math.inf
    mask_fit = _finite_mask(y_fit_col)
    mask_probe = _finite_mask(y_probe_col)
    if mask_fit is None or mask_probe is None or int(mask_fit.sum().item()) <= 0 or int(mask_probe.sum().item()) <= 0:
        return math.inf, math.inf
    intercept = torch.mean(y_fit_col[mask_fit].to(dtype=torch.float64))
    fit_resid = y_fit_col.to(dtype=torch.float64) - intercept
    probe_resid = y_probe_col.to(dtype=torch.float64) - intercept
    try:
        return (
            float(torch.mean(fit_resid[mask_fit].square()).item()),
            float(torch.mean(probe_resid[mask_probe].square()).item()),
        )
    except Exception:
        return math.inf, math.inf


def _linear_feature_losses(
    *,
    col_fit: torch.Tensor,
    col_probe: torch.Tensor,
    y_fit: torch.Tensor | None,
    y_probe: torch.Tensor | None,
) -> tuple[float, float]:
    y_fit_col = _as_col_tensor(y_fit)
    y_probe_col = _as_col_tensor(y_probe)
    if y_fit_col is None or y_probe_col is None:
        return math.inf, math.inf
    mask_fit = _finite_mask(y_fit_col, col_fit)
    mask_probe = _finite_mask(y_probe_col, col_probe)
    if mask_fit is None or mask_probe is None or int(mask_fit.sum().item()) <= 2 or int(mask_probe.sum().item()) <= 0:
        return math.inf, math.inf
    z_fit = col_fit[mask_fit].to(dtype=torch.float64)
    target_fit = y_fit_col[mask_fit].to(dtype=torch.float64)
    ones = torch.ones_like(z_fit)
    design = torch.cat([z_fit, ones], dim=1)
    try:
        sol = torch.linalg.lstsq(design, target_fit).solution
    except Exception:
        return math.inf, math.inf
    pred_fit = design @ sol
    z_probe = col_probe[mask_probe].to(dtype=torch.float64)
    target_probe = y_probe_col[mask_probe].to(dtype=torch.float64)
    probe_design = torch.cat([z_probe, torch.ones_like(z_probe)], dim=1)
    pred_probe = probe_design @ sol
    try:
        return (
            float(torch.mean((target_fit - pred_fit).square()).item()),
            float(torch.mean((target_probe - pred_probe).square()).item()),
        )
    except Exception:
        return math.inf, math.inf


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, float, int, str]:
    try:
        probe = float(row.get("local_probe_mse", math.inf) or math.inf)
    except Exception:
        probe = math.inf
    try:
        fit = float(row.get("local_fit_mse", math.inf) or math.inf)
    except Exception:
        fit = math.inf
    expr = _valid_node(row.get("expr", None))
    try:
        size = int(node_size(expr)) if expr is not None else 1_000_000
    except Exception:
        size = 1_000_000
    return (float(probe), float(fit), int(size), str(row.get("proposal_key", "") or row.get("child_key", "") or ""))


def _candidate_family(row: Mapping[str, Any]) -> str:
    for key in ("scaffold_family", "proposal_family", "family"):
        token = str(row.get(key, "") or "").strip()
        if token:
            return token
    return ""


def _candidate_source_key(row: Mapping[str, Any]) -> str:
    parts = [
        _candidate_family(row),
        str(row.get("scaffold_id", "") or ""),
        str(row.get("operator_id", "") or ""),
        str(row.get("spec_key", "") or row.get("proposal_key", "") or row.get("child_key", "") or ""),
        str(row.get("proposal_lane", "") or ""),
    ]
    token = "|".join(part for part in parts if part)
    return token or str(row.get("proposal_key", "") or row.get("child_key", "") or id(row))


def _active_vars(node: tuple) -> tuple[int, ...]:
    seen: set[int] = set()

    def _visit(cur: Any) -> None:
        if not isinstance(cur, tuple) or not cur:
            return
        if str(cur[0]) == "var":
            try:
                seen.add(int(cur[1]))
            except Exception:
                pass
            return
        for child in cur[1:]:
            if isinstance(child, tuple):
                _visit(child)

    _visit(node)
    return tuple(sorted(seen))


def _iter_subtrees(node: tuple | None) -> list[tuple]:
    root = _valid_node(node)
    if root is None:
        return []
    out: list[tuple] = []

    def _walk(cur: Any) -> None:
        valid = _valid_node(cur)
        if valid is None:
            return
        out.append(valid)
        for child in valid[1:]:
            if isinstance(child, tuple):
                _walk(child)

    _walk(root)
    return out


def _append_node(rows: list[tuple[str, tuple]], seen: set[tuple[str, str]], role: str, node: Any) -> None:
    valid = _valid_node(node)
    if valid is None:
        return
    key = (str(role), str(node_str(valid)))
    if key in seen:
        return
    seen.add(key)
    rows.append((str(role), valid))


def _append_feature_block_nodes(
    rows: list[tuple[str, tuple]],
    seen: set[tuple[str, str]],
    block: FeatureBlock | None,
    *,
    prefix: str,
) -> None:
    if not isinstance(block, FeatureBlock):
        return
    for role, node in zip(
        tuple(getattr(block, "head_bundle_roles", ()) or ()),
        tuple(getattr(block, "head_bundle_nodes", ()) or ()),
    ):
        _append_node(rows, seen, f"{prefix}head:{role}", node)
    for role, node in zip(
        tuple(getattr(block, "latent_bundle_roles", ()) or ()),
        tuple(getattr(block, "latent_bundle_nodes", ()) or ()),
    ):
        _append_node(rows, seen, f"{prefix}bundle:{role}", node)
    for node in tuple(getattr(block, "atoms", ()) or ()):
        _append_node(rows, seen, f"{prefix}atom", node)


def _harvest_roots_from_row(row: Mapping[str, Any]) -> list[tuple[str, tuple]]:
    rows: list[tuple[str, tuple]] = []
    seen: set[tuple[str, str]] = set()
    _append_node(rows, seen, "expr", row.get("expr", None))

    for block_key in ("feature_block_obj",):
        _append_feature_block_nodes(rows, seen, row.get(block_key, None), prefix="")

    state = row.get("basis_state_obj", None)
    if isinstance(state, BasisState):
        for block in tuple(getattr(state, "blocks", ()) or ()):
            _append_feature_block_nodes(rows, seen, block, prefix="state_")

    direct_meta = row.get("direct_metadata", None)
    if isinstance(direct_meta, Mapping):
        for key in _DIRECT_METADATA_NODE_KEYS:
            _append_node(rows, seen, f"direct:{key}", direct_meta.get(key, None))
        for key in _DIRECT_METADATA_NODE_LIST_KEYS:
            for node in list(direct_meta.get(key, []) or []):
                _append_node(rows, seen, f"direct:{key}", node)

    candidate_obj = row.get("proposal_candidate_obj", None)
    if candidate_obj is not None:
        _append_node(rows, seen, "proposal_candidate:expr", getattr(candidate_obj, "expr", None))
        _append_feature_block_nodes(
            rows,
            seen,
            getattr(candidate_obj, "feature_block", None),
            prefix="proposal_candidate:",
        )
        candidate_state = getattr(candidate_obj, "basis_state", None)
        if isinstance(candidate_state, BasisState):
            for block in tuple(getattr(candidate_state, "blocks", ()) or ()):
                _append_feature_block_nodes(rows, seen, block, prefix="proposal_candidate_state_")
        bound_closure = getattr(candidate_obj, "bound_closure", None)
        bindings = getattr(bound_closure, "bindings", None)
        if isinstance(bindings, Mapping):
            for key, value in dict(bindings).items():
                if isinstance(value, SeedBlock):
                    _append_node(rows, seen, f"bound:{key}", value.node)
                elif isinstance(value, tuple) and value and isinstance(value[0], str):
                    _append_node(rows, seen, f"bound:{key}", value)
                else:
                    for idx, item in enumerate(tuple(value or ())):
                        if isinstance(item, SeedBlock):
                            _append_node(rows, seen, f"bound:{key}:{idx}", item.node)
                        else:
                            _append_node(rows, seen, f"bound:{key}:{idx}", item)

    return rows


def _dims_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return bool(dims_eq(tuple(float(v) for v in left), tuple(float(v) for v in right)))
    except Exception:
        return False


def _node_dim(node: tuple, var_dims: Sequence[Sequence[float]] | None) -> tuple[float, ...] | None:
    if var_dims is None:
        return None
    try:
        raw = node_dims(node, var_dims)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return tuple(float(v) for v in raw)
    except Exception:
        return None


def _active_var_dim_buckets(
    active_vars: Sequence[int],
    var_dims: Sequence[Sequence[float]] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if var_dims is None:
        return (), ()
    dimless = dim0(var_dims)
    if dimless is None:
        return (), ()
    dimensionless: list[int] = []
    dimensional: list[int] = []
    for idx_raw in tuple(active_vars or ()):
        try:
            idx = int(idx_raw)
            raw_dim = var_dims[idx]
            var_dim = tuple(float(v) for v in raw_dim)
        except Exception:
            continue
        if _dims_match(var_dim, dimless):
            dimensionless.append(idx)
        else:
            dimensional.append(idx)
    return tuple(sorted(set(dimensionless))), tuple(sorted(set(dimensional)))


def _root_op(node: Any) -> str:
    if isinstance(node, tuple) and node:
        return str(node[0])
    return ""


def _const_scalar(node: Any) -> float | None:
    if not (isinstance(node, tuple) and len(node) >= 2 and str(node[0]) == "const"):
        return None
    try:
        value = float(node[1])
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _strip_numeric_multiplier(node: tuple) -> tuple[tuple, bool]:
    """Canonicalize coefficient-scaled atoms to the reusable symbolic atom."""

    cur = node
    stripped = False
    while isinstance(cur, tuple) and len(cur) >= 3 and str(cur[0]) == "mul":
        left = _valid_node(cur[1])
        right = _valid_node(cur[2])
        if left is not None and _const_scalar(left) is not None and right is not None:
            next_node = right
        elif right is not None and _const_scalar(right) is not None and left is not None:
            next_node = left
        else:
            break
        if _root_op(next_node) == "const":
            break
        try:
            cur = simplify(next_node)
        except Exception:
            break
        stripped = True
    return cur, stripped


def _contains_variable_denominator(node: Any) -> bool:
    if not isinstance(node, tuple) or not node:
        return False
    if str(node[0]) == "div" and len(node) >= 3:
        denominator = node[2]
        if not (isinstance(denominator, tuple) and denominator and str(denominator[0]) == "const"):
            return True
    return any(_contains_variable_denominator(child) for child in node[1:] if isinstance(child, tuple))


def _flatten_additive_terms(node: tuple) -> list[tuple[int, tuple]]:
    out: list[tuple[int, tuple]] = []

    def _walk(cur: Any, sign: int) -> None:
        if not isinstance(cur, tuple) or not cur:
            return
        op = str(cur[0])
        if op == "add" and len(cur) >= 3:
            _walk(cur[1], sign)
            _walk(cur[2], sign)
            return
        if op == "sub" and len(cur) >= 3:
            _walk(cur[1], sign)
            _walk(cur[2], -sign)
            return
        if _valid_node(cur) is not None:
            out.append((int(sign), cur))

    _walk(node, 1)
    return out


def _combine_signed_terms(terms: Sequence[tuple[int, tuple]]) -> tuple | None:
    cur: tuple | None = None
    for sign, node in list(terms or ()):
        if cur is None:
            cur = node if int(sign) >= 0 else simplify(("sub", ("const", 0.0), node))
        elif int(sign) >= 0:
            cur = simplify(("add", cur, node))
        else:
            cur = simplify(("sub", cur, node))
    return simplify(cur) if isinstance(cur, tuple) else None


def _common_denominator_relatives(
    node: tuple,
    *,
    var_dims: Sequence[Sequence[float]] | None,
) -> list[tuple[tuple, Mapping[str, Any]]]:
    if _root_op(node) not in {"add", "sub"}:
        return []
    terms = _flatten_additive_terms(node)
    if len(terms) < 2:
        return []
    den_key: str | None = None
    den_node: tuple | None = None
    numerator_terms: list[tuple[int, tuple]] = []
    for sign, term in terms:
        if _root_op(term) != "div" or len(term) < 3:
            return []
        numerator = _valid_node(term[1])
        denominator = _valid_node(term[2])
        if numerator is None or denominator is None:
            return []
        key = str(node_str(simplify(denominator)))
        if den_key is None:
            den_key = key
            den_node = simplify(denominator)
        elif key != den_key:
            return []
        numerator_terms.append((int(sign), simplify(numerator)))
    if den_node is None:
        return []
    dimless = dim0(var_dims)
    den_dim = _node_dim(den_node, var_dims)
    if dimless is not None and (den_dim is None or not _dims_match(den_dim, dimless)):
        return []
    stripped = _combine_signed_terms(numerator_terms)
    if stripped is None or _valid_node(stripped) is None:
        return []
    return [
        (
            stripped,
            {
                "common_denominator_stripped": True,
                "rational_derived": True,
                "stripped_denominator": str(node_str(den_node)),
                "raw_expr": str(node_str(node)),
            },
        )
    ]


def _atom_kind(node: tuple, node_dim: Any, *, y_dims: Sequence[float] | None, var_dims) -> str | None:
    dimless = dim0(var_dims)
    root = _root_op(node)
    if y_dims is not None and node_dim is not None and _dims_match(node_dim, y_dims):
        if root in {"sqrt", "sqr", "mul"}:
            return "carrier"
        return "target_term"
    if dimless is not None and node_dim is not None and _dims_match(node_dim, dimless):
        return "dimensionless_feature"
    return None


def _candidate_score(
    evidence: _AtomEvidence,
    *,
    y_baseline_fit: float,
    y_baseline_probe: float,
) -> float:
    kind_bonus = {
        "target_term": 3.0,
        "carrier": 2.5,
        "dimensionless_feature": 1.8,
    }.get(str(evidence.kind), 1.0)
    source_bonus = (
        math.log1p(max(0, len(evidence.sources)))
        + 0.50 * math.log1p(max(0, len(evidence.families)))
        + 0.25 * math.log1p(max(0, len(evidence.parent_exprs)))
        + 0.08 * math.log1p(max(0, evidence.sightings))
    )
    if evidence.common_denominator_stripped:
        source_bonus += 0.75
    if evidence.rational_derived:
        source_bonus += 0.05
    parent_bonus = 0.0
    if math.isfinite(evidence.best_parent_probe):
        parent_bonus = 1.0 / (1.0 + max(0.0, evidence.best_parent_probe))
    gain_bonus = 0.0
    if math.isfinite(y_baseline_probe) and y_baseline_probe > 0.0 and math.isfinite(evidence.best_probe_mse):
        gain_bonus = max(0.0, 1.0 - evidence.best_probe_mse / max(1.0e-300, y_baseline_probe))
    elif math.isfinite(y_baseline_fit) and y_baseline_fit > 0.0 and math.isfinite(evidence.best_fit_mse):
        gain_bonus = max(0.0, 1.0 - evidence.best_fit_mse / max(1.0e-300, y_baseline_fit))
    try:
        size_penalty = 0.035 * max(1, int(node_size(evidence.node)))
    except Exception:
        size_penalty = 0.35
    role_text = " ".join(str(v).lower() for v in tuple(evidence.roles or ()))
    artifact_penalty = 0.0
    if "denominator" in role_text:
        artifact_penalty += 0.75
        if "numerator" not in role_text and "expr" not in role_text:
            artifact_penalty += 0.55
        if gain_bonus <= 0.0:
            artifact_penalty += 0.35
    if evidence.rational_derived and not evidence.common_denominator_stripped:
        artifact_penalty += 0.15
    return float(kind_bonus + source_bonus + parent_bonus + 2.0 * gain_bonus - size_penalty - artifact_penalty)


def _inc_reject(stats: dict[str, Any], reason: str) -> None:
    counts = stats.get("emergent_aux_atom_reject_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        stats["emergent_aux_atom_reject_counts"] = counts
    counts[str(reason)] = int(counts.get(str(reason), 0) or 0) + 1


def _debug_append(stats: dict[str, Any], row: Mapping[str, Any], *, debug_limit: int) -> None:
    if int(debug_limit) <= 0:
        return
    bucket = stats.get("debug_emergent_aux_atoms", None)
    if not isinstance(bucket, list):
        bucket = []
        stats["debug_emergent_aux_atoms"] = bucket
    if len(bucket) >= int(debug_limit):
        return
    bucket.append(dict(row))


def _parent_losses(row: Mapping[str, Any]) -> tuple[float, float]:
    try:
        fit = float(row.get("local_fit_mse", math.inf) or math.inf)
    except Exception:
        fit = math.inf
    try:
        probe = float(row.get("local_probe_mse", math.inf) or math.inf)
    except Exception:
        probe = math.inf
    return fit, probe


def _consider_atom(
    evidence_by_key: dict[str, _AtomEvidence],
    node: tuple,
    *,
    role: str,
    row: Mapping[str, Any],
    source_key: str,
    parent_key: str,
    family: str,
    var_dims: Sequence[Sequence[float]] | None,
    y_dims: Sequence[float] | None,
    x_fit: torch.Tensor | None,
    y_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    y_probe: torch.Tensor | None,
    y_baseline_fit: float,
    y_baseline_probe: float,
    stats: dict[str, Any],
    debug_limit: int,
    min_column_variance: float,
    max_node_size: int,
    max_node_depth: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    meta = dict(metadata or {})
    try:
        simp = simplify(node)
    except Exception:
        _inc_reject(stats, "simplify_failed")
        return
    if _valid_node(simp) is None:
        _inc_reject(stats, "invalid_node")
        return
    raw_key = str(node_str(simp))
    stripped, stripped_scale = _strip_numeric_multiplier(simp)
    if stripped_scale and _valid_node(stripped) is not None:
        simp = stripped
        meta["numeric_multiplier_stripped"] = True
        meta.setdefault("raw_scaled_expr", raw_key)
    key = str(node_str(simp))
    if parent_key and key == parent_key:
        _inc_reject(stats, "whole_parent")
        return
    root = _root_op(simp)
    if root in _DISALLOWED_ROOTS:
        _inc_reject(stats, f"root_{root}_disallowed")
        return
    if root not in _ALLOWED_ROOTS:
        _inc_reject(stats, f"root_{root}_unsupported")
        return
    if _contains_variable_denominator(simp) and not bool(meta.get("common_denominator_stripped", False)):
        _inc_reject(stats, "raw_variable_denominator")
        return
    try:
        size = int(node_size(simp))
        depth = int(node_depth(simp))
    except Exception:
        _inc_reject(stats, "complexity_failed")
        return
    if size > int(max_node_size) or depth > int(max_node_depth):
        _inc_reject(stats, "too_complex")
        return
    active = _active_vars(simp)
    if not active:
        _inc_reject(stats, "no_active_vars")
        return
    if root in {"add", "sub"} and len(active) < 2:
        _inc_reject(stats, "addsub_too_few_active_vars")
        return
    if len(active) > 4:
        _inc_reject(stats, "too_many_active_vars")
        return
    active_dimless, active_dimensional = _active_var_dim_buckets(active, var_dims)
    node_dim = _node_dim(simp, var_dims)
    kind = _atom_kind(simp, node_dim, y_dims=y_dims, var_dims=var_dims)
    if kind is None:
        _inc_reject(stats, "dim_not_supported")
        return
    col_fit = _safe_eval_node_col(simp, x_fit)
    col_probe = _safe_eval_node_col(simp, x_probe)
    if col_fit is None or col_probe is None:
        _inc_reject(stats, "eval_failed")
        return
    mask_fit = _finite_mask(col_fit)
    mask_probe = _finite_mask(col_probe)
    if mask_fit is None or mask_probe is None or int(mask_fit.sum().item()) <= 2 or int(mask_probe.sum().item()) <= 2:
        _inc_reject(stats, "insufficient_finite")
        return
    if _col_variance(col_fit, mask_fit) <= float(min_column_variance):
        _inc_reject(stats, "low_variance")
        return
    fit_mse = math.inf
    probe_mse = math.inf
    fit_gain = 0.0
    probe_gain = 0.0
    if kind in {"target_term", "carrier"}:
        fit_mse, probe_mse = _linear_feature_losses(
            col_fit=col_fit,
            col_probe=col_probe,
            y_fit=y_fit,
            y_probe=y_probe,
        )
        if math.isfinite(y_baseline_fit) and y_baseline_fit > 0.0 and math.isfinite(fit_mse):
            fit_gain = max(0.0, 1.0 - fit_mse / max(1.0e-300, y_baseline_fit))
        if math.isfinite(y_baseline_probe) and y_baseline_probe > 0.0 and math.isfinite(probe_mse):
            probe_gain = max(0.0, 1.0 - probe_mse / max(1.0e-300, y_baseline_probe))

    parent_fit, parent_probe = _parent_losses(row)
    entry = evidence_by_key.get(key)
    if entry is None:
        entry = _AtomEvidence(
            key=key,
            node=simp,
            dim=node_dim,
            kind=kind,
            active_vars=tuple(active),
            active_dimensionless_vars=tuple(active_dimless),
            active_dimensional_vars=tuple(active_dimensional),
            root_op=str(root),
        )
        evidence_by_key[key] = entry
    entry.active_vars = tuple(sorted(set(entry.active_vars).union(int(v) for v in tuple(active or ()))))
    entry.active_dimensionless_vars = tuple(
        sorted(set(entry.active_dimensionless_vars).union(int(v) for v in tuple(active_dimless or ())))
    )
    entry.active_dimensional_vars = tuple(
        sorted(set(entry.active_dimensional_vars).union(int(v) for v in tuple(active_dimensional or ())))
    )
    if not entry.root_op:
        entry.root_op = str(root)
    entry.sources.add(str(source_key))
    if parent_key:
        entry.parent_exprs.add(str(parent_key))
    if role:
        entry.roles.add(str(role))
    if family:
        entry.families.add(str(family))
    entry.sightings += 1
    entry.best_parent_fit = min(float(entry.best_parent_fit), float(parent_fit))
    entry.best_parent_probe = min(float(entry.best_parent_probe), float(parent_probe))
    entry.best_fit_mse = min(float(entry.best_fit_mse), float(fit_mse))
    entry.best_probe_mse = min(float(entry.best_probe_mse), float(probe_mse))
    entry.best_fit_gain_rel = max(float(entry.best_fit_gain_rel), float(fit_gain))
    entry.best_probe_gain_rel = max(float(entry.best_probe_gain_rel), float(probe_gain))
    entry.rational_derived = bool(entry.rational_derived or meta.get("rational_derived", False) or "rational" in str(family).lower())
    entry.common_denominator_stripped = bool(
        entry.common_denominator_stripped or meta.get("common_denominator_stripped", False)
    )
    _debug_append(
        stats,
        {
            "decision": "accepted_observation",
            "expr": key,
            "kind": kind,
            "role": str(role),
            "family": str(family),
            "source": str(source_key),
            "fit_mse": None if not math.isfinite(fit_mse) else float(fit_mse),
            "probe_mse": None if not math.isfinite(probe_mse) else float(probe_mse),
            "fit_gain_rel": float(fit_gain),
            "probe_gain_rel": float(probe_gain),
            "metadata": _jsonish(meta),
        },
        debug_limit=debug_limit,
    )


def _materialize_atom(evidence: _AtomEvidence, *, y_baseline_fit: float, y_baseline_probe: float) -> EmergentAtom:
    score = _candidate_score(evidence, y_baseline_fit=y_baseline_fit, y_baseline_probe=y_baseline_probe)
    payload = {
        "sources": sorted(str(v) for v in evidence.sources),
        "parent_exprs": sorted(str(v) for v in evidence.parent_exprs)[:8],
        "sightings": int(evidence.sightings),
        "best_parent_fit": None if not math.isfinite(evidence.best_parent_fit) else float(evidence.best_parent_fit),
        "best_parent_probe": None if not math.isfinite(evidence.best_parent_probe) else float(evidence.best_parent_probe),
        "best_fit_mse": None if not math.isfinite(evidence.best_fit_mse) else float(evidence.best_fit_mse),
        "best_probe_mse": None if not math.isfinite(evidence.best_probe_mse) else float(evidence.best_probe_mse),
        "best_fit_gain_rel": float(evidence.best_fit_gain_rel),
        "best_probe_gain_rel": float(evidence.best_probe_gain_rel),
        "rational_derived": bool(evidence.rational_derived),
        "common_denominator_stripped": bool(evidence.common_denominator_stripped),
        "active_vars": [int(v) for v in tuple(evidence.active_vars or ())],
        "active_dimensionless_vars": [int(v) for v in tuple(evidence.active_dimensionless_vars or ())],
        "active_dimensional_vars": [int(v) for v in tuple(evidence.active_dimensional_vars or ())],
        "root_op": str(evidence.root_op or _root_op(evidence.node)),
    }
    return EmergentAtom(
        node=evidence.node,
        dim=evidence.dim,
        kind=str(evidence.kind),
        score=float(score),
        evidence=payload,
        source_count=int(len(evidence.sources)),
        roles=tuple(sorted(str(v) for v in evidence.roles)),
        families=tuple(sorted(str(v) for v in evidence.families)),
    )


def _atom_rank(atom: EmergentAtom) -> tuple[float, int, int, str]:
    try:
        size = int(node_size(atom.node))
    except Exception:
        size = 999
    kind_rank = {"target_term": 0, "carrier": 1, "dimensionless_feature": 2}.get(str(atom.kind), 3)
    return (-float(atom.score), int(kind_rank), int(size), str(node_str(atom.node)))


def _atom_role_text(atom: EmergentAtom) -> str:
    pieces: list[str] = []
    pieces.extend(str(v).lower() for v in tuple(getattr(atom, "roles", ()) or ()))
    pieces.extend(str(v).lower() for v in tuple(getattr(atom, "families", ()) or ()))
    evidence = dict(getattr(atom, "evidence", {}) or {})
    pieces.extend(str(v).lower() for v in tuple(evidence.get("sources", ()) or ()))
    return " ".join(piece for piece in pieces if piece)


def _evidence_int_tuple(evidence: Mapping[str, Any], key: str) -> tuple[int, ...]:
    out: list[int] = []
    for value in tuple(dict(evidence or {}).get(str(key), ()) or ()):
        try:
            out.append(int(value))
        except Exception:
            continue
    return tuple(sorted(set(out)))


def _is_pure_dimensional_carrier(atom: EmergentAtom) -> bool:
    if not isinstance(atom, EmergentAtom) or str(atom.kind) != "carrier":
        return False
    evidence = dict(atom.evidence or {})
    active = _evidence_int_tuple(evidence, "active_vars")
    if not active:
        return False
    active_dimless = _evidence_int_tuple(evidence, "active_dimensionless_vars")
    if active_dimless:
        return False
    active_dimensional = set(_evidence_int_tuple(evidence, "active_dimensional_vars"))
    return set(active).issubset(active_dimensional)


def _atom_role_buckets(atom: EmergentAtom) -> tuple[str, ...]:
    if not isinstance(atom, EmergentAtom):
        return ()
    root = _root_op(atom.node)
    kind = str(atom.kind)
    role_text = _atom_role_text(atom)
    evidence = dict(atom.evidence or {})
    buckets: list[str] = []
    if kind == "target_term":
        buckets.append("target_term")
    if kind == "dimensionless_feature" and root in {"sin", "cos"}:
        buckets.append("completed_modulator")
    if _is_pure_dimensional_carrier(atom):
        buckets.append("pure_dimensional_carrier")
    if kind == "carrier" or (kind != "dimensionless_feature" and root in {"sqrt", "mul", "sqr"}):
        buckets.append("envelope_prefactor")
    if kind != "dimensionless_feature" and (
        "numerator" in role_text
        or bool(evidence.get("rational_derived", False))
        or bool(evidence.get("common_denominator_stripped", False))
    ):
        buckets.append("numerator_dimensional")
    if kind == "dimensionless_feature" and root not in {"sin", "cos"}:
        buckets.append("carrier_argument")
    if kind == "carrier":
        buckets.append("carrier")
    if kind == "dimensionless_feature":
        buckets.append("dimensionless_feature")
    out: list[str] = []
    seen: set[str] = set()
    for bucket in buckets:
        if bucket in seen:
            continue
        seen.add(bucket)
        out.append(bucket)
    return tuple(out)


def _atom_debug_dict(atom: EmergentAtom) -> dict[str, Any]:
    payload = atom.to_dict()
    payload["role_buckets"] = [str(v) for v in _atom_role_buckets(atom)]
    return payload


def _atom_bucket_counts(atoms: Sequence[EmergentAtom]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom in tuple(atoms or ()):
        if not isinstance(atom, EmergentAtom):
            continue
        for bucket in _atom_role_buckets(atom):
            counts[str(bucket)] = int(counts.get(str(bucket), 0) or 0) + 1
    return counts


def _select_diverse_atoms(atoms: Sequence[EmergentAtom], *, max_count: int) -> tuple[EmergentAtom, ...]:
    rows = sorted((atom for atom in tuple(atoms or ()) if isinstance(atom, EmergentAtom)), key=_atom_rank)
    limit = max(0, int(max_count))
    if limit <= 0:
        return ()
    if limit == 1:
        return tuple(rows[:1])

    out: list[EmergentAtom] = []
    seen: set[str] = set()

    def add_atom(atom: EmergentAtom) -> bool:
        if len(out) >= limit:
            return False
        key = str(node_str(atom.node))
        if key in seen:
            return False
        out.append(atom)
        seen.add(key)
        return True

    def add_first_kind(kind: str) -> bool:
        for atom in rows:
            if str(atom.kind) != str(kind):
                continue
            if add_atom(atom):
                return True
        return False

    def add_first_bucket(bucket: str) -> bool:
        for atom in rows:
            if str(bucket) not in _atom_role_buckets(atom):
                continue
            if add_atom(atom):
                return True
        return False

    for bucket in _ROLE_BUCKET_ORDER:
        if len(out) >= limit:
            break
        add_first_bucket(bucket)

    if not any(str(atom.kind) == "target_term" for atom in out):
        add_first_kind("target_term")
    if not any(str(atom.kind) == "carrier" for atom in out):
        add_first_kind("carrier")
    if not any(str(atom.kind) == "dimensionless_feature" for atom in out):
        add_first_kind("dimensionless_feature")

    for atom in rows:
        if len(out) >= limit:
            break
        add_atom(atom)
    return tuple(out[:limit])


def harvest_emergent_atoms(
    *,
    candidate_rows: Sequence[Mapping[str, Any]] | None,
    x_fit: torch.Tensor | None,
    y_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    y_probe: torch.Tensor | None,
    var_dims: Sequence[Sequence[float]] | None,
    y_dims: Sequence[float] | None,
    stats: dict[str, Any] | None = None,
    max_source_rows: int = 48,
    max_harvested_per_row: int = 64,
    max_new: int = 2,
    max_node_size: int = 9,
    max_node_depth: int = 4,
    min_column_variance: float = 1.0e-14,
    debug_limit: int = 0,
    observed_atom_sink: list[EmergentAtom] | None = None,
) -> tuple[EmergentAtom, ...]:
    """Harvest reusable auxiliary atoms from current-round FSS candidates."""

    stats_out = stats if isinstance(stats, dict) else {}
    stats_out["emergent_aux_atom_calls"] = int(stats_out.get("emergent_aux_atom_calls", 0) or 0) + 1
    if int(max_new) <= 0:
        stats_out["emergent_aux_atom_disabled_by_budget"] = int(
            stats_out.get("emergent_aux_atom_disabled_by_budget", 0) or 0
        ) + 1
        return ()
    if var_dims is None or y_dims is None:
        stats_out["emergent_aux_atom_missing_dims"] = int(
            stats_out.get("emergent_aux_atom_missing_dims", 0) or 0
        ) + 1
        return ()
    if not isinstance(x_fit, torch.Tensor) or not isinstance(x_probe, torch.Tensor):
        stats_out["emergent_aux_atom_missing_data"] = int(
            stats_out.get("emergent_aux_atom_missing_data", 0) or 0
        ) + 1
        return ()

    y_baseline_fit, y_baseline_probe = _constant_baseline_losses(y_fit=y_fit, y_probe=y_probe)
    source_rows = [
        row
        for row in sorted(
            [dict(row) for row in tuple(candidate_rows or ()) if isinstance(row, Mapping)],
            key=_candidate_sort_key,
        )[: max(0, int(max_source_rows))]
    ]
    stats_out["emergent_aux_atom_source_rows"] = int(
        stats_out.get("emergent_aux_atom_source_rows", 0) or 0
    ) + len(source_rows)

    evidence_by_key: dict[str, _AtomEvidence] = {}
    total_sightings = 0
    for row in source_rows:
        parent_expr = _valid_node(row.get("expr", None))
        parent_key = str(node_str(parent_expr)) if parent_expr is not None else ""
        source_key = _candidate_source_key(row)
        family = _candidate_family(row)
        harvested_this_row = 0
        for role, root in _harvest_roots_from_row(row):
            if harvested_this_row >= int(max_harvested_per_row):
                break
            for raw_node in _iter_subtrees(root):
                if harvested_this_row >= int(max_harvested_per_row):
                    break
                nodes_to_consider: list[tuple[tuple, Mapping[str, Any]]] = []
                try:
                    base_node = simplify(raw_node)
                except Exception:
                    _inc_reject(stats_out, "simplify_failed")
                    continue
                if _valid_node(base_node) is None:
                    _inc_reject(stats_out, "invalid_node")
                    continue
                nodes_to_consider.append((base_node, {}))
                nodes_to_consider.extend(_common_denominator_relatives(base_node, var_dims=var_dims))
                for node, meta in nodes_to_consider:
                    _consider_atom(
                        evidence_by_key,
                        node,
                        role=role,
                        row=row,
                        source_key=source_key,
                        parent_key=parent_key,
                        family=family,
                        var_dims=var_dims,
                        y_dims=y_dims,
                        x_fit=x_fit,
                        y_fit=y_fit,
                        x_probe=x_probe,
                        y_probe=y_probe,
                        y_baseline_fit=y_baseline_fit,
                        y_baseline_probe=y_baseline_probe,
                        stats=stats_out,
                        debug_limit=int(debug_limit),
                        min_column_variance=float(min_column_variance),
                        max_node_size=int(max_node_size),
                        max_node_depth=int(max_node_depth),
                        metadata=meta,
                    )
                    harvested_this_row += 1
                    total_sightings += 1

    stats_out["emergent_aux_atom_sightings"] = int(
        stats_out.get("emergent_aux_atom_sightings", 0) or 0
    ) + int(total_sightings)
    stats_out["emergent_aux_atom_unique"] = int(
        stats_out.get("emergent_aux_atom_unique", 0) or 0
    ) + int(len(evidence_by_key))
    atoms = [
        _materialize_atom(evidence, y_baseline_fit=y_baseline_fit, y_baseline_probe=y_baseline_probe)
        for evidence in evidence_by_key.values()
    ]
    atoms_sorted = sorted(atoms, key=_atom_rank)
    if isinstance(observed_atom_sink, list):
        observed_atom_sink.extend(atoms_sorted)
    stats_out["emergent_aux_atom_observed_bucket_counts"] = _atom_bucket_counts(atoms_sorted)
    stats_out["emergent_aux_atom_observed_top"] = [
        _atom_debug_dict(atom)
        for atom in atoms_sorted[: max(0, int(debug_limit) if int(debug_limit) > 0 else 12)]
    ]
    out = _select_diverse_atoms(atoms, max_count=int(max_new))
    selected_keys = {str(node_str(atom.node)) for atom in tuple(out or ())}
    stats_out["emergent_aux_atom_seen_not_retained"] = [
        _atom_debug_dict(atom)
        for atom in atoms_sorted
        if str(node_str(atom.node)) not in selected_keys
    ][: max(0, int(debug_limit) if int(debug_limit) > 0 else 12)]
    stats_out["emergent_aux_atom_accepted"] = int(
        stats_out.get("emergent_aux_atom_accepted", 0) or 0
    ) + int(len(out))
    by_kind = stats_out.get("emergent_aux_atom_by_kind", None)
    if not isinstance(by_kind, dict):
        by_kind = {}
        stats_out["emergent_aux_atom_by_kind"] = by_kind
    for atom in out:
        by_kind[str(atom.kind)] = int(by_kind.get(str(atom.kind), 0) or 0) + 1
        _debug_append(
            stats_out,
            {
                "decision": "selected_atom",
                **_atom_debug_dict(atom),
            },
            debug_limit=int(debug_limit),
        )
    return out


def _merge_atom(existing: EmergentAtom, incoming: EmergentAtom) -> EmergentAtom:
    sources = set(str(v) for v in tuple(dict(existing.evidence or {}).get("sources", ()) or ()))
    sources.update(str(v) for v in tuple(dict(incoming.evidence or {}).get("sources", ()) or ()))
    parent_exprs = set(str(v) for v in tuple(dict(existing.evidence or {}).get("parent_exprs", ()) or ()))
    parent_exprs.update(str(v) for v in tuple(dict(incoming.evidence or {}).get("parent_exprs", ()) or ()))
    evidence = dict(existing.evidence or {})
    incoming_evidence = dict(incoming.evidence or {})
    evidence["sources"] = sorted(sources)
    evidence["parent_exprs"] = sorted(parent_exprs)[:8]
    evidence["sightings"] = int(evidence.get("sightings", 0) or 0) + int(incoming_evidence.get("sightings", 0) or 0)
    evidence["rational_derived"] = bool(evidence.get("rational_derived", False) or incoming_evidence.get("rational_derived", False))
    evidence["common_denominator_stripped"] = bool(
        evidence.get("common_denominator_stripped", False)
        or incoming_evidence.get("common_denominator_stripped", False)
    )
    for key in ("active_vars", "active_dimensionless_vars", "active_dimensional_vars"):
        merged = set(_evidence_int_tuple(evidence, key))
        merged.update(_evidence_int_tuple(incoming_evidence, key))
        evidence[key] = [int(v) for v in sorted(merged)]
    if not evidence.get("root_op", "") and incoming_evidence.get("root_op", ""):
        evidence["root_op"] = str(incoming_evidence.get("root_op", ""))
    for key in ("best_fit_mse", "best_probe_mse", "best_parent_fit", "best_parent_probe"):
        left = evidence.get(key, math.inf)
        right = incoming_evidence.get(key, math.inf)
        try:
            left_f = float(left)
        except Exception:
            left_f = math.inf
        try:
            right_f = float(right)
        except Exception:
            right_f = math.inf
        best = min(left_f, right_f)
        evidence[key] = None if not math.isfinite(best) else float(best)
    for key in ("best_fit_gain_rel", "best_probe_gain_rel"):
        try:
            left_f = float(evidence.get(key, 0.0) or 0.0)
        except Exception:
            left_f = 0.0
        try:
            right_f = float(incoming_evidence.get(key, 0.0) or 0.0)
        except Exception:
            right_f = 0.0
        evidence[key] = max(left_f, right_f)
    return replace(
        existing if float(existing.score) >= float(incoming.score) else incoming,
        score=max(float(existing.score), float(incoming.score)),
        evidence=evidence,
        source_count=max(int(existing.source_count), int(incoming.source_count), len(sources)),
        roles=tuple(sorted(set(existing.roles).union(str(v) for v in tuple(incoming.roles or ())))),
        families=tuple(sorted(set(existing.families).union(str(v) for v in tuple(incoming.families or ())))),
    )


def merge_emergent_atom_registry(
    existing: Sequence[EmergentAtom] | None,
    new_atoms: Sequence[EmergentAtom] | None,
    *,
    max_total: int = 8,
    max_target: int = 4,
    max_dimensionless: int = 3,
    max_rational_derived: int = 2,
    stats: dict[str, Any] | None = None,
) -> tuple[EmergentAtom, ...]:
    """Merge new atoms into the persistent registry with small caps."""

    stats_out = stats if isinstance(stats, dict) else {}
    by_key: dict[str, EmergentAtom] = {}
    for atom in tuple(existing or ()):
        if not isinstance(atom, EmergentAtom):
            continue
        by_key[str(node_str(atom.node))] = atom
    new_count = 0
    for atom in tuple(new_atoms or ()):
        if not isinstance(atom, EmergentAtom):
            continue
        key = str(node_str(atom.node))
        if key in by_key:
            by_key[key] = _merge_atom(by_key[key], atom)
        else:
            by_key[key] = atom
            new_count += 1
    rows = sorted(by_key.values(), key=_atom_rank)
    out: list[EmergentAtom] = []
    selected: set[str] = set()
    target_count = 0
    dimless_count = 0
    rational_counts = {"dimensional": 0, "dimensionless": 0}

    def can_add(atom: EmergentAtom) -> bool:
        if len(out) >= max(0, int(max_total)):
            return False
        key = str(node_str(atom.node))
        if key in selected:
            return False
        is_dimless = str(atom.kind) == "dimensionless_feature"
        is_rational = bool(dict(atom.evidence or {}).get("rational_derived", False))
        if is_dimless:
            if dimless_count >= int(max_dimensionless):
                return False
        else:
            if target_count >= int(max_target):
                return False
        rational_bucket = "dimensionless" if is_dimless else "dimensional"
        if is_rational and rational_counts[rational_bucket] >= int(max_rational_derived):
            return False
        return True

    def add_atom(atom: EmergentAtom) -> bool:
        nonlocal target_count, dimless_count
        if not can_add(atom):
            return False
        is_dimless = str(atom.kind) == "dimensionless_feature"
        is_rational = bool(dict(atom.evidence or {}).get("rational_derived", False))
        out.append(atom)
        selected.add(str(node_str(atom.node)))
        if is_dimless:
            dimless_count += 1
        else:
            target_count += 1
        if is_rational:
            rational_bucket = "dimensionless" if is_dimless else "dimensional"
            rational_counts[rational_bucket] += 1
        return True

    for bucket in _ROLE_BUCKET_ORDER:
        for atom in rows:
            if str(bucket) in _atom_role_buckets(atom) and add_atom(atom):
                break
    for kind in ("target_term", "carrier", "dimensionless_feature"):
        for atom in rows:
            if str(atom.kind) == kind and add_atom(atom):
                break
    for atom in rows:
        if len(out) >= max(0, int(max_total)):
            break
        add_atom(atom)
    selected_keys = {str(node_str(atom.node)) for atom in tuple(out or ())}
    stats_out["emergent_aux_atom_registry_size"] = int(len(out))
    stats_out["emergent_aux_atom_registry_new"] = int(stats_out.get("emergent_aux_atom_registry_new", 0) or 0) + int(new_count)
    stats_out["emergent_aux_atom_registry_rational_by_bucket"] = dict(rational_counts)
    stats_out["emergent_aux_atom_registry_by_bucket"] = _atom_bucket_counts(out)
    stats_out["emergent_aux_atom_registry"] = [_atom_debug_dict(atom) for atom in out]
    stats_out["emergent_aux_atom_registry_not_retained"] = [
        _atom_debug_dict(atom)
        for atom in rows
        if str(node_str(atom.node)) not in selected_keys
    ][:12]
    return tuple(out)


def seed_blocks_from_emergent_atoms(
    atoms: Sequence[EmergentAtom] | None,
    *,
    var_dims: Sequence[Sequence[float]] | None = None,
    limit: int = 8,
) -> tuple[SeedBlock, ...]:
    """Convert accepted auxiliary atoms into explicit FSS SeedBlocks."""

    out: list[SeedBlock] = []
    seen: set[str] = set()
    for atom in tuple(atoms or ()):
        if not isinstance(atom, EmergentAtom):
            continue
        node = _valid_node(atom.node)
        if node is None:
            continue
        dim = atom.dim
        if dim is None and var_dims is not None:
            dim = _node_dim(node, var_dims)
        key = str(node_str(node))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            make_seed_block(
                node,
                dim=dim,
                source=f"aux:emergent:{str(atom.kind)}",
                builder="identity",
                metadata={
                    "origin": "aux:emergent",
                    "emergent_atom": atom.to_dict(),
                },
            )
        )
        if len(out) >= max(0, int(limit)):
            break
    return tuple(out)


__all__ = [
    "EmergentAtom",
    "harvest_emergent_atoms",
    "merge_emergent_atom_registry",
    "seed_blocks_from_emergent_atoms",
]
