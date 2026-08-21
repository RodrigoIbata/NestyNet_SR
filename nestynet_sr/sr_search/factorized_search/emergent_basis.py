# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Evidence-driven basis promotion for factorized symbolic search."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .basis_head import fit_basis_state_head, score_basis_state_conditional_gain
from .basis_state import (
    BasisState,
    FeatureBlock,
    basis_state_covers_feature_block,
    basis_state_extend,
    topologically_order_feature_blocks,
)
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


@dataclass
class _Evidence:
    key: str
    node: tuple
    dim: tuple[float, ...]
    sources: set[str] = field(default_factory=set)
    parent_exprs: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    families: set[str] = field(default_factory=set)
    best_parent_probe: float = math.inf
    best_parent_fit: float = math.inf
    sightings: int = 0


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
    return out


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
    return col.to(dtype=torch.float64)


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
        fit_loss = float(torch.mean(fit_resid[mask_fit].square()).item())
        probe_loss = float(torch.mean(probe_resid[mask_probe].square()).item())
    except Exception:
        return math.inf, math.inf
    return fit_loss, probe_loss


def _current_losses(
    current_basis_state: BasisState | None,
    *,
    y_fit: torch.Tensor | None,
    y_probe: torch.Tensor | None,
) -> tuple[float, float]:
    if isinstance(current_basis_state, BasisState):
        try:
            fit_loss = float(getattr(current_basis_state, "fit_loss", math.inf))
            probe_loss = float(getattr(current_basis_state, "probe_loss", math.inf))
        except Exception:
            fit_loss = math.inf
            probe_loss = math.inf
        if math.isfinite(fit_loss) and math.isfinite(probe_loss):
            return fit_loss, probe_loss
    return _constant_baseline_losses(y_fit=y_fit, y_probe=y_probe)


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


def _append_node(rows: list[tuple[str, tuple]], seen: set[tuple[str, str]], role: str, node: Any) -> None:
    valid = _valid_node(node)
    if valid is None:
        return
    key = (str(role), str(node_str(valid)))
    if key in seen:
        return
    seen.add(key)
    rows.append((str(role), valid))


def _harvest_roots_from_row(row: Mapping[str, Any]) -> list[tuple[str, tuple]]:
    rows: list[tuple[str, tuple]] = []
    seen: set[tuple[str, str]] = set()
    expr = row.get("expr", None)
    _append_node(rows, seen, "expr", expr)

    for block_key in ("feature_block_obj",):
        block = row.get(block_key, None)
        if not isinstance(block, FeatureBlock):
            continue
        for role, node in zip(
            tuple(getattr(block, "head_bundle_roles", ()) or ()),
            tuple(getattr(block, "head_bundle_nodes", ()) or ()),
        ):
            _append_node(rows, seen, f"head:{role}", node)
        for role, node in zip(
            tuple(getattr(block, "latent_bundle_roles", ()) or ()),
            tuple(getattr(block, "latent_bundle_nodes", ()) or ()),
        ):
            _append_node(rows, seen, f"bundle:{role}", node)
        for node in tuple(getattr(block, "atoms", ()) or ()):
            _append_node(rows, seen, "atom", node)

    state = row.get("basis_state_obj", None)
    if isinstance(state, BasisState):
        for block in tuple(getattr(state, "blocks", ()) or ()):
            if not isinstance(block, FeatureBlock):
                continue
            for role, node in zip(
                tuple(getattr(block, "head_bundle_roles", ()) or ()),
                tuple(getattr(block, "head_bundle_nodes", ()) or ()),
            ):
                _append_node(rows, seen, f"state_head:{role}", node)
            for role, node in zip(
                tuple(getattr(block, "latent_bundle_roles", ()) or ()),
                tuple(getattr(block, "latent_bundle_nodes", ()) or ()),
            ):
                _append_node(rows, seen, f"state_bundle:{role}", node)
            for node in tuple(getattr(block, "atoms", ()) or ()):
                _append_node(rows, seen, "state_atom", node)

    direct_meta = row.get("direct_metadata", None)
    if isinstance(direct_meta, Mapping):
        for key in _DIRECT_METADATA_NODE_KEYS:
            _append_node(rows, seen, f"direct:{key}", direct_meta.get(key, None))
        for key in _DIRECT_METADATA_NODE_LIST_KEYS:
            for node in list(direct_meta.get(key, []) or []):
                _append_node(rows, seen, f"direct:{key}", node)
    return rows


def _dims_match(node_dim: Sequence[float] | None, y_dims: Sequence[float] | None) -> bool:
    if node_dim is None or y_dims is None:
        return False
    try:
        return bool(dims_eq(tuple(float(v) for v in node_dim), tuple(float(v) for v in y_dims)))
    except Exception:
        return False


def _make_feature_block(
    *,
    node: tuple,
    dim: Sequence[float] | None,
    evidence: Mapping[str, Any],
) -> FeatureBlock:
    key = str(node_str(node))
    metadata = {
        "route": "emergent_basis",
        "source": "emergent_subexpr",
        "block_expr_obj": node,
        "block_expr": key,
        "head_bundle_exprs": [key],
        "head_bundle_roles": ["primary"],
        "promotion_evidence": dict(evidence),
    }
    block = FeatureBlock(
        family="emergent_subexpr",
        atoms=(node,),
        head_type="linear",
        block_id=f"emergent_subexpr::{key}",
        parent_block_ids=(),
        latent_bundle_nodes=(node,),
        latent_bundle_roles=("primary",),
        head_bundle_nodes=(node,),
        head_bundle_roles=("primary",),
        dim_signature=tuple(float(v) for v in dim) if dim is not None else None,
        active_vars=_active_vars(node),
        metadata=metadata,
    )
    ordered = topologically_order_feature_blocks((block,), drop_orphans=True)
    return ordered[0] if ordered else block


def _seed_state_from_block(block: FeatureBlock, *, node: tuple, evidence: Mapping[str, Any]) -> BasisState:
    return BasisState(
        blocks=(block,),
        fit_bundle={"mapping_kind": "linear", "mapping_coeffs": []},
        fit_loss=math.inf,
        probe_loss=math.inf,
        complexity=float(block.complexity()),
        diagnostics={
            "route": "emergent_basis",
            "family": "emergent_subexpr",
            "promotion_evidence": dict(evidence),
        },
        provenance=("emergent_basis:subexpr",),
        compiled_expr=node,
    )


def _inc_reject(stats: dict[str, Any], reason: str) -> None:
    counts = stats.get("emergent_basis_reject_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        stats["emergent_basis_reject_counts"] = counts
    counts[str(reason)] = int(counts.get(str(reason), 0) or 0) + 1


def _debug_append(stats: dict[str, Any], row: Mapping[str, Any], *, debug_limit: int) -> None:
    if int(debug_limit) <= 0:
        return
    bucket = stats.get("debug_emergent_basis", None)
    if not isinstance(bucket, list):
        bucket = []
        stats["debug_emergent_basis"] = bucket
    if len(bucket) >= int(debug_limit):
        return
    bucket.append(dict(row))


def propose_emergent_basis_rows(
    *,
    candidate_rows: Sequence[Mapping[str, Any]] | None,
    current_basis_state: BasisState | None,
    x_fit: torch.Tensor | None,
    y_fit: torch.Tensor | None,
    x_probe: torch.Tensor | None,
    y_probe: torch.Tensor | None,
    var_dims: Sequence[Sequence[float]] | None,
    y_dims: Sequence[float] | None,
    stats: dict[str, Any] | None = None,
    max_source_rows: int = 32,
    max_harvested_per_row: int = 32,
    max_node_size: int = 7,
    max_node_depth: int = 3,
    min_active_vars: int = 2,
    max_active_vars: int = 3,
    min_source_count: int = 1,
    score_topk: int = 8,
    max_promoted_per_round: int = 1,
    max_promoted_total: int = 4,
    min_probe_gain_rel: float = 5.0e-3,
    min_probe_gain_abs: float = 1.0e-12,
    min_conditional_gain_abs: float = 1.0e-12,
    min_probe_to_fit_gain_ratio: float = 0.25,
    min_column_variance: float = 1.0e-14,
    debug_limit: int = 0,
) -> list[dict[str, Any]]:
    """Return candidate rows for small target-dimension subexpressions.

    This pass only proposes additive head features. It does not mutate the seed
    pool directly; returned rows are consumed by the normal basis scoring and
    beam admission machinery.
    """

    stats_out = stats if isinstance(stats, dict) else {}
    stats_out["emergent_basis_calls"] = int(stats_out.get("emergent_basis_calls", 0) or 0) + 1
    if int(max_promoted_per_round) <= 0 or int(score_topk) <= 0:
        stats_out["emergent_basis_disabled_by_budget"] = int(
            stats_out.get("emergent_basis_disabled_by_budget", 0) or 0
        ) + 1
        return []
    if var_dims is None or y_dims is None:
        stats_out["emergent_basis_missing_dims"] = int(stats_out.get("emergent_basis_missing_dims", 0) or 0) + 1
        return []
    if not isinstance(x_fit, torch.Tensor) or not isinstance(x_probe, torch.Tensor):
        stats_out["emergent_basis_missing_data"] = int(stats_out.get("emergent_basis_missing_data", 0) or 0) + 1
        return []

    current_emergent_count = 0
    if isinstance(current_basis_state, BasisState):
        for block in tuple(getattr(current_basis_state, "blocks", ()) or ()):
            if isinstance(block, FeatureBlock) and str(getattr(block, "family", "") or "") == "emergent_subexpr":
                current_emergent_count += 1
    if int(max_promoted_total) >= 0 and current_emergent_count >= int(max_promoted_total):
        stats_out["emergent_basis_total_cap_reached"] = int(
            stats_out.get("emergent_basis_total_cap_reached", 0) or 0
        ) + 1
        return []

    source_rows = [
        row
        for row in sorted(
            [dict(row) for row in tuple(candidate_rows or ()) if isinstance(row, Mapping)],
            key=_candidate_sort_key,
        )[: max(0, int(max_source_rows))]
    ]
    stats_out["emergent_basis_source_rows"] = int(stats_out.get("emergent_basis_source_rows", 0) or 0) + len(source_rows)

    evidence_by_key: dict[str, _Evidence] = {}
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
                try:
                    node = simplify(raw_node)
                except Exception:
                    _inc_reject(stats_out, "simplify_failed")
                    continue
                if _valid_node(node) is None:
                    _inc_reject(stats_out, "invalid_node")
                    continue
                if str(node[0]) not in ("add", "sub"):
                    continue
                key = str(node_str(node))
                if parent_key and key == parent_key:
                    _inc_reject(stats_out, "whole_parent")
                    continue
                try:
                    size = int(node_size(node))
                    depth = int(node_depth(node))
                except Exception:
                    _inc_reject(stats_out, "complexity_failed")
                    continue
                if size > int(max_node_size) or depth > int(max_node_depth):
                    _inc_reject(stats_out, "too_complex")
                    continue
                active = _active_vars(node)
                if len(active) < int(min_active_vars):
                    _inc_reject(stats_out, "too_few_active_vars")
                    continue
                if len(active) > int(max_active_vars):
                    _inc_reject(stats_out, "too_many_active_vars")
                    continue
                try:
                    dim = node_dims(node, var_dims)
                except Exception:
                    _inc_reject(stats_out, "dims_failed")
                    continue
                if not _dims_match(dim, y_dims):
                    _inc_reject(stats_out, "dim_mismatch")
                    continue
                col_fit = _safe_eval_node_col(node, x_fit)
                col_probe = _safe_eval_node_col(node, x_probe)
                if col_fit is None or col_probe is None:
                    _inc_reject(stats_out, "eval_failed")
                    continue
                y_fit_col = _as_col_tensor(y_fit)
                y_probe_col = _as_col_tensor(y_probe)
                mask_fit = _finite_mask(y_fit_col, col_fit)
                mask_probe = _finite_mask(y_probe_col, col_probe)
                if (
                    mask_fit is None
                    or mask_probe is None
                    or int(mask_fit.sum().item()) <= 2
                    or int(mask_probe.sum().item()) <= 2
                ):
                    _inc_reject(stats_out, "nonfinite_or_too_few")
                    continue
                if _col_variance(col_fit, mask_fit) <= float(min_column_variance):
                    _inc_reject(stats_out, "low_variance")
                    continue
                ev = evidence_by_key.get(key, None)
                dim_tuple = tuple(float(v) for v in tuple(dim or ()))
                if ev is None:
                    ev = _Evidence(key=key, node=node, dim=dim_tuple)
                    evidence_by_key[key] = ev
                ev.sources.add(str(source_key))
                if parent_key:
                    ev.parent_exprs.add(str(parent_key))
                ev.roles.add(str(role))
                if family:
                    ev.families.add(str(family))
                try:
                    ev.best_parent_probe = min(ev.best_parent_probe, float(row.get("local_probe_mse", math.inf) or math.inf))
                    ev.best_parent_fit = min(ev.best_parent_fit, float(row.get("local_fit_mse", math.inf) or math.inf))
                except Exception:
                    pass
                ev.sightings += 1
                total_sightings += 1
                harvested_this_row += 1

    stats_out["emergent_basis_sightings"] = int(stats_out.get("emergent_basis_sightings", 0) or 0) + int(total_sightings)
    stats_out["emergent_basis_unique"] = int(stats_out.get("emergent_basis_unique", 0) or 0) + int(len(evidence_by_key))
    if not evidence_by_key:
        return []

    current_fit, current_probe = _current_losses(current_basis_state, y_fit=y_fit, y_probe=y_probe)
    if not math.isfinite(current_probe):
        _inc_reject(stats_out, "nonfinite_current_probe")
        return []
    required_probe_gain = max(
        float(min_probe_gain_abs),
        max(1.0, abs(float(current_probe))) * max(0.0, float(min_probe_gain_rel)),
    )

    ranked_evidence = sorted(
        evidence_by_key.values(),
        key=lambda ev: (
            -len(ev.sources),
            float(ev.best_parent_probe),
            int(node_size(ev.node)),
            int(node_depth(ev.node)),
            str(ev.key),
        ),
    )
    rows: list[dict[str, Any]] = []
    scored = 0
    for ev in ranked_evidence:
        if len(rows) >= int(max_promoted_per_round):
            break
        if scored >= int(score_topk):
            break
        if len(ev.sources) < int(min_source_count):
            _inc_reject(stats_out, "insufficient_sources")
            continue
        evidence = {
            "expr": str(ev.key),
            "sources": sorted(ev.sources),
            "source_count": int(len(ev.sources)),
            "parent_exprs": sorted(ev.parent_exprs)[:8],
            "roles": sorted(ev.roles),
            "families": sorted(ev.families),
            "sightings": int(ev.sightings),
            "best_parent_probe": float(ev.best_parent_probe),
            "best_parent_fit": float(ev.best_parent_fit),
        }
        block = _make_feature_block(node=ev.node, dim=ev.dim, evidence=evidence)
        if basis_state_covers_feature_block(current_basis_state, block):
            _inc_reject(stats_out, "already_covered")
            _debug_append(
                stats_out,
                {**evidence, "decision": "reject", "reason": "already_covered"},
                debug_limit=debug_limit,
            )
            continue
        seed_state = _seed_state_from_block(block, node=ev.node, evidence=evidence)
        conditional = score_basis_state_conditional_gain(
            current_basis_state,
            seed_state,
            x_fit=x_fit,
            y_fit=y_fit,
        )
        cond_gain = 0.0
        if isinstance(conditional, Mapping):
            try:
                cond_gain = float(conditional.get("gain", 0.0) or 0.0)
            except Exception:
                cond_gain = 0.0
        if not math.isfinite(cond_gain) or cond_gain <= float(min_conditional_gain_abs):
            _inc_reject(stats_out, "insufficient_conditional_gain")
            _debug_append(
                stats_out,
                {**evidence, "decision": "reject", "reason": "insufficient_conditional_gain", "conditional_gain": float(cond_gain)},
                debug_limit=debug_limit,
            )
            continue
        preview_state = basis_state_extend(
            current_basis_state,
            seed_state,
            route_name="emergent_basis_prepare",
        )
        if not isinstance(preview_state, BasisState):
            preview_state = seed_state
        scored_state = fit_basis_state_head(
            preview_state,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            route_name="emergent_basis_refit",
        )
        scored += 1
        if not isinstance(scored_state, BasisState):
            _inc_reject(stats_out, "fit_failed")
            continue
        try:
            candidate_fit = float(getattr(scored_state, "fit_loss", math.inf))
            candidate_probe = float(getattr(scored_state, "probe_loss", math.inf))
        except Exception:
            candidate_fit = math.inf
            candidate_probe = math.inf
        if not math.isfinite(candidate_fit) or not math.isfinite(candidate_probe):
            _inc_reject(stats_out, "nonfinite_candidate_loss")
            continue
        probe_gain = float(current_probe - candidate_probe)
        fit_gain = float(current_fit - candidate_fit) if math.isfinite(current_fit) else math.inf
        if probe_gain < required_probe_gain:
            _inc_reject(stats_out, "insufficient_probe_gain")
            _debug_append(
                stats_out,
                {
                    **evidence,
                    "decision": "reject",
                    "reason": "insufficient_probe_gain",
                    "fit_gain": float(fit_gain),
                    "probe_gain": float(probe_gain),
                    "required_probe_gain": float(required_probe_gain),
                    "candidate_probe": float(candidate_probe),
                    "current_probe": float(current_probe),
                    "conditional_gain": float(cond_gain),
                },
                debug_limit=debug_limit,
            )
            continue
        if (
            math.isfinite(fit_gain)
            and fit_gain > 0.0
            and probe_gain < max(0.0, float(min_probe_to_fit_gain_ratio)) * fit_gain
        ):
            _inc_reject(stats_out, "probe_fit_gain_disagree")
            _debug_append(
                stats_out,
                {
                    **evidence,
                    "decision": "reject",
                    "reason": "probe_fit_gain_disagree",
                    "fit_gain": float(fit_gain),
                    "probe_gain": float(probe_gain),
                    "candidate_probe": float(candidate_probe),
                    "current_probe": float(current_probe),
                    "conditional_gain": float(cond_gain),
                },
                debug_limit=debug_limit,
            )
            continue

        proposal_key = f"emergent_basis::{ev.key}"
        expr = scored_state.compiled_expr if _valid_node(scored_state.compiled_expr) is not None else ev.node
        row = {
            "expr": expr,
            "proposal_key": str(proposal_key),
            "child_key": str(proposal_key),
            "scaffold_id": str(proposal_key),
            "proposal_family": "emergent_subexpr",
            "scaffold_family": "emergent_subexpr",
            "operator_id": "emergent_basis",
            "proposal_lane": "core",
            "candidate_child_size": int(node_size(ev.node)),
            "local_fit_mse": float(candidate_fit),
            "local_probe_mse": float(candidate_probe),
            "feature_block_obj": block,
            "feature_block_dict": block.to_dict(),
            "basis_state_obj": scored_state,
            "basis_state_dict": scored_state.to_dict(),
            "basis_state_direct_preserve": False,
            "emergent_basis": True,
            "emergent_basis_expr": str(ev.key),
            "emergent_basis_evidence": {
                **evidence,
                "fit_gain": float(fit_gain),
                "probe_gain": float(probe_gain),
                "required_probe_gain": float(required_probe_gain),
                "candidate_fit": float(candidate_fit),
                "candidate_probe": float(candidate_probe),
                "current_fit": float(current_fit),
                "current_probe": float(current_probe),
                "conditional_gain": float(cond_gain),
                "conditional": dict(conditional or {}),
            },
        }
        rows.append(row)
        _debug_append(
            stats_out,
            {
                **row["emergent_basis_evidence"],
                "decision": "promote_row",
                "proposal_key": str(proposal_key),
                "compiled_expr": str(node_str(expr)) if _valid_node(expr) is not None else "",
            },
            debug_limit=debug_limit,
        )

    stats_out["emergent_basis_scored"] = int(stats_out.get("emergent_basis_scored", 0) or 0) + int(scored)
    stats_out["emergent_basis_rows"] = int(stats_out.get("emergent_basis_rows", 0) or 0) + int(len(rows))
    return rows


__all__ = ["propose_emergent_basis_rows"]
