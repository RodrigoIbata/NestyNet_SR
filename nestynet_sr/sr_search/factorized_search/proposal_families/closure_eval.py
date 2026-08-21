# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
import math
from typing import Any, Mapping, MutableSet

import torch

from ..basis_scoring import (
    direct_power_depth_slack_from_coeffs,
    direct_quadratic_depth_slack_from_coeffs,
    score_bound_closure,
)
from ..closures import BoundClosure, ClosureDesign, bound_closure_identity_key, bound_closure_identity_payload
from ..expr_ast import dims_eq, is_valid_node, node_depth, node_dims, node_size, node_str
from .common import deadline_exceeded
from .types import OperatorApplication


def _finite_sort_value(value: Any, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def scaffold_parent_stats(spec: OperatorApplication) -> dict[str, int]:
    try:
        parent_sub = spec.parent_node
        for idx in tuple(spec.hole_path or ()):
            parent_sub = parent_sub[int(idx)]
    except Exception:
        parent_sub = None
    return {
        "parent_sub_size": int(node_size(parent_sub)) if isinstance(parent_sub, tuple) else 0,
        "parent_sub_depth": int(node_depth(parent_sub)) if isinstance(parent_sub, tuple) else 0,
        "parent_size": int(node_size(spec.parent_node)),
        "parent_depth": int(node_depth(spec.parent_node)),
    }


def _identity_snapshot(value: Any) -> Any:
    if isinstance(value, tuple):
        try:
            return str(node_str(value))
        except Exception:
            return [ _identity_snapshot(v) for v in value ]
    if isinstance(value, Mapping):
        return {str(k): _identity_snapshot(v) for k, v in dict(value).items()}
    if isinstance(value, list):
        return [_identity_snapshot(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def closure_candidate_identity_key(
    *,
    bound_closure: BoundClosure,
    design: ClosureDesign,
    direct_metadata: Mapping[str, Any] | None = None,
) -> str:
    meta = dict(direct_metadata or {})
    explicit = str(meta.get("proposal_key", "") or "").strip()
    if explicit:
        return explicit
    payload = {
        "bound_closure": bound_closure_identity_payload(bound_closure),
        "materializer": str(getattr(design, "materializer", "") or ""),
        "materializer_payload": _identity_snapshot(getattr(design, "materializer_payload", {}) or {}),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def make_direct_preview_row(
    *,
    bound_closure: BoundClosure,
    child_expr: tuple,
    fit_mse: float,
    probe_mse: float,
    max_depth: int,
    var_dims,
    y_dims,
    candidate_subtree_node: tuple | None,
    parent_sub_size: int,
    parent_sub_depth: int,
    parent_size: int,
    parent_depth: int,
    generation_source: str,
    tuple_provenance: str,
    proposal_family: str,
    local_mapping_kind: str,
    direct_metadata: Mapping[str, Any] | None,
    seen_child_keys: MutableSet[str],
    proposal_key: str | None = None,
    local_mapping_coeffs: list[float] | None = None,
    local_mapping_nparams: int | None = None,
) -> dict[str, Any] | None:
    if not isinstance(child_expr, tuple) or not is_valid_node(child_expr):
        return None
    depth_budget = int(max_depth)
    meta = dict(direct_metadata or {})
    if str(meta.get("quadratic_kind", "") or "").strip().lower() == "sqrt_mul":
        base_nodes = [node for node in list(meta.get("quadratic_base_nodes", []) or ()) if isinstance(node, tuple)]
        anchor_node = meta.get("anchor_node", None)
        if isinstance(anchor_node, tuple) and is_valid_node(anchor_node):
            depth_budget += max(0, len(base_nodes) - 2)
    if str(local_mapping_kind or "").strip().lower() in {
        "direct_quadratic_sqrt_head",
        "direct_quadratic_head",
        "quadratic_sqrt",
    }:
        depth_budget += direct_quadratic_depth_slack_from_coeffs(local_mapping_coeffs or ())
    if str(meta.get("power_kind", "") or "").strip().lower() in {
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
        coeffs_preview = [float(v) for v in list(local_mapping_coeffs or ())]
        depth_budget += int(
            direct_power_depth_slack_from_coeffs(
                coeffs_preview,
                exponent=float(meta.get("power_exponent", 0.0) or 0.0),
            )
        )
    if int(node_depth(child_expr)) > int(depth_budget):
        return None
    if var_dims is not None:
        try:
            child_dim = node_dims(child_expr, var_dims)
        except Exception:
            child_dim = None
        if child_dim is None:
            return None
        if y_dims is not None and not dims_eq(child_dim, y_dims):
            return None

    child_key = str(node_str(child_expr))
    closure_key = str(bound_closure_identity_key(bound_closure) or "")
    proposal_key = str(proposal_key or dict(direct_metadata or {}).get("proposal_key", "") or closure_key or child_key)
    if proposal_key in seen_child_keys:
        return None
    seen_child_keys.add(proposal_key)

    coeffs = [float(v) for v in list(local_mapping_coeffs or ())]
    if local_mapping_nparams is None:
        local_mapping_nparams = int(len(coeffs))

    candidate_sub_size = int(node_size(candidate_subtree_node)) if isinstance(candidate_subtree_node, tuple) else 0
    candidate_sub_depth = int(node_depth(candidate_subtree_node)) if isinstance(candidate_subtree_node, tuple) else 0
    child_size = int(node_size(child_expr))
    child_depth = int(node_depth(child_expr))

    return {
        "expr": child_expr,
        "child_expr": str(child_key),
        "child_key": str(child_key),
        "proposal_key": str(proposal_key),
        "closure_identity_key": str(closure_key),
        "rendered_expr_key": str(child_key),
        "local_probe_mse": float(probe_mse),
        "local_fit_mse": float(fit_mse),
        "local_fit_probe_gap": float(max(0.0, float(probe_mse) - float(fit_mse))),
        "local_mapping_kind": str(local_mapping_kind),
        "local_mapping_nparams": int(local_mapping_nparams),
        "local_mapping_coeffs": coeffs,
        "candidate_subtree_size": int(candidate_sub_size),
        "candidate_subtree_depth": int(candidate_sub_depth),
        "candidate_subtree_size_delta": int(candidate_sub_size - int(parent_sub_size)),
        "candidate_subtree_depth_delta": int(candidate_sub_depth - int(parent_sub_depth)),
        "candidate_child_size": int(child_size),
        "candidate_child_depth": int(child_depth),
        "candidate_child_size_delta": int(child_size - int(parent_size)),
        "candidate_child_depth_delta": int(child_depth - int(parent_depth)),
        "candidate_root_op": str(child_expr[0]),
        "generation_source": str(generation_source),
        "tuple_provenance": str(tuple_provenance),
        "proposal_family": str(proposal_family),
        "exact_child_score_observed": False,
        "dedup_kept": False,
        "bound_closure_obj": bound_closure,
        "bound_closure_dict": bound_closure.to_dict(),
        "direct_metadata": dict(direct_metadata or {}),
    }


def score_direct_closure_candidate(
    *,
    bound_closure: BoundClosure,
    design: ClosureDesign,
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    max_depth: int,
    var_dims,
    y_dims,
    candidate_subtree_node: tuple | None,
    parent_sub_size: int,
    parent_sub_depth: int,
    parent_size: int,
    parent_depth: int,
    generation_source: str,
    tuple_provenance: str,
    proposal_family: str,
    local_mapping_kind: str,
    direct_metadata: Mapping[str, Any] | None,
    seen_child_keys: MutableSet[str],
    local_mapping_coeffs: list[float] | None = None,
    local_mapping_nparams: int | None = None,
) -> dict[str, Any] | None:
    scored = score_bound_closure(
        bound_closure,
        design=design,
        y_fit=y_fit,
        y_probe=y_probe,
    )
    if scored is None:
        return None

    coeffs = [float(v) for v in list(scored.get("coeffs", []) or ())]
    if local_mapping_coeffs is None:
        local_mapping_coeffs = coeffs
    proposal_key = closure_candidate_identity_key(
        bound_closure=bound_closure,
        design=design,
        direct_metadata=direct_metadata,
    )
    return make_direct_preview_row(
        bound_closure=bound_closure,
        child_expr=scored["expr"],
        fit_mse=float(scored["fit_mse"]),
        probe_mse=float(scored["probe_mse"]),
        max_depth=int(max_depth),
        var_dims=var_dims,
        y_dims=y_dims,
        candidate_subtree_node=candidate_subtree_node,
        parent_sub_size=int(parent_sub_size),
        parent_sub_depth=int(parent_sub_depth),
        parent_size=int(parent_size),
        parent_depth=int(parent_depth),
        generation_source=str(generation_source),
        tuple_provenance=str(tuple_provenance),
        proposal_family=str(proposal_family),
        local_mapping_kind=str(local_mapping_kind),
        direct_metadata=direct_metadata,
        seen_child_keys=seen_child_keys,
        proposal_key=proposal_key,
        local_mapping_coeffs=list(local_mapping_coeffs or ()),
        local_mapping_nparams=local_mapping_nparams,
    )


def finalize_direct_preview_rows(
    rows: list[dict[str, Any]],
    *,
    preview_topk: int,
    raw_candidate_count: int,
    scored_candidate_count: int,
    deadline_s: float | None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows.sort(
        key=lambda row: (
            _finite_sort_value(row.get("local_probe_mse", float("inf"))),
            _finite_sort_value(row.get("local_fit_mse", float("inf"))),
            int(row.get("candidate_child_size", 0) or 0),
            str(row.get("proposal_key", "") or ""),
            str(row.get("child_key", "") or ""),
        )
    )
    rows = rows[: max(1, int(preview_topk))]
    meta_out = dict(meta or {})
    meta_out["candidate_count_raw"] = int(raw_candidate_count)
    meta_out["candidate_count_scored"] = int(scored_candidate_count)
    meta_out["deadline_exceeded"] = bool(
        deadline_exceeded(deadline_s) or bool(meta_out.get("deadline_exceeded", False))
    )
    if rows:
        status = "direct_ok"
    elif bool(meta_out.get("deadline_exceeded", False)):
        status = "direct_deadline_exceeded"
    else:
        status = "direct_no_scored_candidates"
    return rows, status, meta_out


__all__ = [
    "closure_candidate_identity_key",
    "finalize_direct_preview_rows",
    "make_direct_preview_row",
    "scaffold_parent_stats",
    "score_direct_closure_candidate",
]
