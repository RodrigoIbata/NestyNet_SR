# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .binding_search import (
    collect_shortlisted_hole_candidates,
    evaluate_candidate_node_map,
    evaluate_seed_blocks,
)
from .closure_builders import BuiltClosureCandidate
from .closure_eval import finalize_direct_preview_rows, score_direct_closure_candidate
from .common import deadline_exceeded


@dataclass(frozen=True)
class PreparedClosureCandidate:
    built: BuiltClosureCandidate
    candidate_subtree_node: tuple | None = None
    row_patch: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SingleHoleCollectedSearchPlan:
    nvars: int
    enum_max_depth: int
    enum_max_trees: int
    var_dims: Any
    target_dim: Any
    pool_nodes: Any
    pool_dims: Any
    shortlist_k: int
    pin_predicate: Callable[[tuple], bool] | None
    prepare_candidate_fn: Callable[[str, tuple], PreparedClosureCandidate | None]
    parent_stats: Mapping[str, int]
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PairCollectedSearchPlan:
    nvars: int
    enum_max_depth: int
    enum_max_trees: int
    var_dims: Any
    left_target_dim: Any
    right_target_dim: Any
    pool_nodes: Any
    pool_dims: Any
    left_shortlist_k: int
    right_shortlist_k: int
    prepare_candidate_fn: Callable[
        [tuple[torch.Tensor, torch.Tensor, str, tuple], tuple[torch.Tensor, torch.Tensor, str, tuple]],
        PreparedClosureCandidate | None,
    ]
    parent_stats: Mapping[str, int]
    left_pin_predicate: Callable[[tuple], bool] | None = None
    right_pin_predicate: Callable[[tuple], bool] | None = None
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SeedSubsetSearchPlan:
    seed_blocks: Sequence[Any]
    prepare_candidate_fn: Callable[[tuple[dict[str, Any], ...]], PreparedClosureCandidate | None]
    subset_max_arity: int
    var_dims: Any
    parent_stats: Mapping[str, int]
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PreparedCandidatesSearchPlan:
    candidates: Sequence[PreparedClosureCandidate]
    var_dims: Any
    parent_stats: Mapping[str, int]
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CustomDirectSearchPlan:
    run_fn: Callable[..., tuple[list[dict[str, Any]], str, dict[str, Any]]]
    kwargs: Mapping[str, Any]


def execute_direct_search_plan(
    plan: Any,
    *,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    max_depth: int,
    y_dims,
    preview_topk: int,
    deadline_s: float | None,
    collect_direct_hole_candidates_fn=None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    if isinstance(plan, CustomDirectSearchPlan):
        return plan.run_fn(**dict(plan.kwargs or {}))
    if isinstance(plan, SingleHoleCollectedSearchPlan):
        return run_collected_single_hole_search(
            collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
            nvars=int(plan.nvars),
            enum_max_depth=int(plan.enum_max_depth),
            enum_max_trees=int(plan.enum_max_trees),
            var_dims=plan.var_dims,
            target_dim=plan.target_dim,
            pool_nodes=plan.pool_nodes,
            pool_dims=plan.pool_dims,
            shortlist_k=int(plan.shortlist_k),
            pin_predicate=plan.pin_predicate,
            prepare_candidate_fn=plan.prepare_candidate_fn,
            y_fit=y_fit,
            y_probe=y_probe,
            max_depth=int(max_depth),
            y_dims=y_dims,
            parent_stats=plan.parent_stats,
            preview_topk=int(preview_topk),
            deadline_s=deadline_s,
            meta=plan.meta,
        )
    if isinstance(plan, PairCollectedSearchPlan):
        return run_collected_pair_closure_search(
            collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
            nvars=int(plan.nvars),
            enum_max_depth=int(plan.enum_max_depth),
            enum_max_trees=int(plan.enum_max_trees),
            var_dims=plan.var_dims,
            left_target_dim=plan.left_target_dim,
            right_target_dim=plan.right_target_dim,
            pool_nodes=plan.pool_nodes,
            pool_dims=plan.pool_dims,
            left_shortlist_k=int(plan.left_shortlist_k),
            right_shortlist_k=int(plan.right_shortlist_k),
            left_pin_predicate=plan.left_pin_predicate,
            right_pin_predicate=plan.right_pin_predicate,
            prepare_candidate_fn=plan.prepare_candidate_fn,
            x_fit=x_fit,
            x_probe=x_probe,
            y_fit=y_fit,
            y_probe=y_probe,
            max_depth=int(max_depth),
            y_dims=y_dims,
            parent_stats=plan.parent_stats,
            preview_topk=int(preview_topk),
            deadline_s=deadline_s,
            meta=plan.meta,
        )
    if isinstance(plan, SeedSubsetSearchPlan):
        return run_seed_subset_closure_search(
            seed_blocks=plan.seed_blocks,
            prepare_candidate_fn=plan.prepare_candidate_fn,
            x_fit=x_fit,
            x_probe=x_probe,
            y_fit=y_fit,
            y_probe=y_probe,
            subset_max_arity=int(plan.subset_max_arity),
            max_depth=int(max_depth),
            var_dims=plan.var_dims,
            y_dims=y_dims,
            parent_stats=plan.parent_stats,
            preview_topk=int(preview_topk),
            deadline_s=deadline_s,
            meta=plan.meta,
        )
    if isinstance(plan, PreparedCandidatesSearchPlan):
        return run_prepared_candidate_search(
            candidates=plan.candidates,
            y_fit=y_fit,
            y_probe=y_probe,
            max_depth=int(max_depth),
            var_dims=plan.var_dims,
            y_dims=y_dims,
            parent_stats=plan.parent_stats,
            preview_topk=int(preview_topk),
            deadline_s=deadline_s,
            meta=plan.meta,
        )
    return [], "direct_not_supported", {}


def _apply_row_patch(row: dict[str, Any], row_patch: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row_patch, Mapping):
        return row
    out = dict(row)
    for key, value in dict(row_patch).items():
        if key == "direct_metadata":
            merged = dict(out.get("direct_metadata", {}) or {})
            merged.update(dict(value or {}))
            out["direct_metadata"] = merged
        else:
            out[key] = value
    return out


def run_single_hole_closure_search(
    *,
    candidate_nodes: Sequence[tuple[str, tuple]],
    prepare_candidate_fn: Callable[[str, tuple], PreparedClosureCandidate | None],
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    max_depth: int,
    var_dims,
    y_dims,
    parent_stats: Mapping[str, int],
    preview_topk: int,
    deadline_s: float | None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_child_keys: set[str] = set()
    raw_candidate_count = 0
    scored_candidate_count = 0
    prepared_candidate_count = 0
    prepare_s = 0.0
    score_s = 0.0
    loop_started = time.perf_counter()
    for source, hole_node in list(candidate_nodes or ()):
        if deadline_exceeded(deadline_s):
            break
        raw_candidate_count += 1
        prepare_started = time.perf_counter()
        prepared = prepare_candidate_fn(str(source), hole_node)
        prepare_s += float(time.perf_counter() - prepare_started)
        if not isinstance(prepared, PreparedClosureCandidate):
            continue
        prepared_candidate_count += 1
        score_started = time.perf_counter()
        row = score_direct_closure_candidate(
            bound_closure=prepared.built.bound_closure,
            design=prepared.built.design,
            y_fit=y_fit,
            y_probe=y_probe,
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            candidate_subtree_node=prepared.candidate_subtree_node,
            parent_sub_size=int(parent_stats["parent_sub_size"]),
            parent_sub_depth=int(parent_stats["parent_sub_depth"]),
            parent_size=int(parent_stats["parent_size"]),
            parent_depth=int(parent_stats["parent_depth"]),
            generation_source=prepared.built.generation_source,
            tuple_provenance=prepared.built.tuple_provenance,
            proposal_family=prepared.built.proposal_family,
            local_mapping_kind=prepared.built.local_mapping_kind,
            local_mapping_nparams=prepared.built.local_mapping_nparams,
            direct_metadata=prepared.built.direct_metadata,
            seen_child_keys=seen_child_keys,
        )
        score_s += float(time.perf_counter() - score_started)
        if row is None:
            continue
        rows.append(_apply_row_patch(row, prepared.row_patch))
        scored_candidate_count += 1
    meta_out = dict(meta or {})
    meta_out.update(
        {
            "prepared_candidate_count": int(prepared_candidate_count),
            "timing_single_prepare_s": float(prepare_s),
            "timing_single_score_s": float(score_s),
            "timing_single_loop_s": float(time.perf_counter() - loop_started),
        }
    )
    return finalize_direct_preview_rows(
        rows,
        preview_topk=int(preview_topk),
        raw_candidate_count=int(raw_candidate_count),
        scored_candidate_count=int(scored_candidate_count),
        deadline_s=deadline_s,
        meta=meta_out,
    )


def run_prepared_candidate_search(
    *,
    candidates: Sequence[PreparedClosureCandidate],
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    max_depth: int,
    var_dims,
    y_dims,
    parent_stats: Mapping[str, int],
    preview_topk: int,
    deadline_s: float | None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_child_keys: set[str] = set()
    raw_candidate_count = 0
    scored_candidate_count = 0
    score_s = 0.0
    loop_started = time.perf_counter()
    for prepared in list(candidates or ()):
        if deadline_exceeded(deadline_s):
            break
        if not isinstance(prepared, PreparedClosureCandidate):
            continue
        raw_candidate_count += 1
        score_started = time.perf_counter()
        row = score_direct_closure_candidate(
            bound_closure=prepared.built.bound_closure,
            design=prepared.built.design,
            y_fit=y_fit,
            y_probe=y_probe,
            max_depth=int(max_depth),
            var_dims=var_dims,
            y_dims=y_dims,
            candidate_subtree_node=prepared.candidate_subtree_node,
            parent_sub_size=int(parent_stats["parent_sub_size"]),
            parent_sub_depth=int(parent_stats["parent_sub_depth"]),
            parent_size=int(parent_stats["parent_size"]),
            parent_depth=int(parent_stats["parent_depth"]),
            generation_source=prepared.built.generation_source,
            tuple_provenance=prepared.built.tuple_provenance,
            proposal_family=prepared.built.proposal_family,
            local_mapping_kind=prepared.built.local_mapping_kind,
            local_mapping_nparams=prepared.built.local_mapping_nparams,
            direct_metadata=prepared.built.direct_metadata,
            seen_child_keys=seen_child_keys,
        )
        score_s += float(time.perf_counter() - score_started)
        if row is None:
            continue
        rows.append(_apply_row_patch(row, prepared.row_patch))
        scored_candidate_count += 1
    meta_out = dict(meta or {})
    meta_out.update(
        {
            "prepared_candidate_count": int(raw_candidate_count),
            "timing_prepared_score_s": float(score_s),
            "timing_prepared_loop_s": float(time.perf_counter() - loop_started),
        }
    )
    return finalize_direct_preview_rows(
        rows,
        preview_topk=int(preview_topk),
        raw_candidate_count=int(raw_candidate_count),
        scored_candidate_count=int(scored_candidate_count),
        deadline_s=deadline_s,
        meta=meta_out,
    )


def run_collected_single_hole_search(
    *,
    collect_direct_hole_candidates_fn: Callable[..., tuple[list[tuple[str, tuple]], dict[str, Any]]],
    nvars: int,
    enum_max_depth: int,
    enum_max_trees: int,
    var_dims,
    target_dim,
    pool_nodes,
    pool_dims,
    shortlist_k: int,
    pin_predicate: Callable[[tuple], bool] | None,
    prepare_candidate_fn: Callable[[str, tuple], PreparedClosureCandidate | None],
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    max_depth: int,
    y_dims,
    parent_stats: Mapping[str, int],
    preview_topk: int,
    deadline_s: float | None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    collect_started = time.perf_counter()
    candidate_nodes, collected_meta = collect_shortlisted_hole_candidates(
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        nvars=int(nvars),
        enum_max_depth=int(enum_max_depth),
        enum_max_trees=int(enum_max_trees),
        var_dims=var_dims,
        target_dim=target_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        shortlist_k=int(shortlist_k),
        deadline_s=deadline_s,
        pin_predicate=pin_predicate,
    )
    collect_elapsed_s = float(time.perf_counter() - collect_started)
    merged_meta = {**dict(collected_meta or {}), **dict(meta or {})}
    merged_meta["timing_collect_single_s"] = float(collect_elapsed_s)
    if not candidate_nodes:
        if bool(merged_meta.get("deadline_exceeded", False)):
            return [], "direct_deadline_exceeded", merged_meta
        return [], "direct_no_hole_candidates", merged_meta
    search_started = time.perf_counter()
    rows, status, meta_out = run_single_hole_closure_search(
        candidate_nodes=candidate_nodes,
        prepare_candidate_fn=prepare_candidate_fn,
        y_fit=y_fit,
        y_probe=y_probe,
        max_depth=int(max_depth),
        var_dims=var_dims,
        y_dims=y_dims,
        parent_stats=parent_stats,
        preview_topk=int(preview_topk),
        deadline_s=deadline_s,
        meta=merged_meta,
    )
    meta_final = dict(meta_out or {})
    meta_final["timing_single_search_s"] = float(time.perf_counter() - search_started)
    return rows, status, meta_final


def run_pair_closure_search(
    *,
    left_rows: Mapping[str, tuple[torch.Tensor, torch.Tensor, str, tuple]],
    right_rows: Mapping[str, tuple[torch.Tensor, torch.Tensor, str, tuple]],
    prepare_candidate_fn: Callable[
        [tuple[torch.Tensor, torch.Tensor, str, tuple], tuple[torch.Tensor, torch.Tensor, str, tuple]],
        PreparedClosureCandidate | None,
    ],
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    max_depth: int,
    var_dims,
    y_dims,
    parent_stats: Mapping[str, int],
    preview_topk: int,
    deadline_s: float | None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_child_keys: set[str] = set()
    raw_candidate_count = 0
    scored_candidate_count = 0
    prepared_candidate_count = 0
    prepare_s = 0.0
    score_s = 0.0
    loop_started = time.perf_counter()
    _n_left = len(list(dict(left_rows or {}).values()))
    _n_right = len(list(dict(right_rows or {}).values()))
    _total_pairs = _n_left * _n_right
    _MAX_PAIR_EVALS = max(50000, _total_pairs) if _total_pairs <= 500000 else 50000
    for left in list(dict(left_rows or {}).values()):
        if deadline_exceeded(deadline_s) or raw_candidate_count >= _MAX_PAIR_EVALS:
            break
        for right in list(dict(right_rows or {}).values()):
            if deadline_exceeded(deadline_s) or raw_candidate_count >= _MAX_PAIR_EVALS:
                break
            raw_candidate_count += 1
            prepare_started = time.perf_counter()
            prepared = prepare_candidate_fn(left, right)
            prepare_s += float(time.perf_counter() - prepare_started)
            if not isinstance(prepared, PreparedClosureCandidate):
                continue
            prepared_candidate_count += 1
            score_started = time.perf_counter()
            row = score_direct_closure_candidate(
                bound_closure=prepared.built.bound_closure,
                design=prepared.built.design,
                y_fit=y_fit,
                y_probe=y_probe,
                max_depth=int(max_depth),
                var_dims=var_dims,
                y_dims=y_dims,
                candidate_subtree_node=prepared.candidate_subtree_node,
                parent_sub_size=int(parent_stats["parent_sub_size"]),
                parent_sub_depth=int(parent_stats["parent_sub_depth"]),
                parent_size=int(parent_stats["parent_size"]),
                parent_depth=int(parent_stats["parent_depth"]),
                generation_source=prepared.built.generation_source,
                tuple_provenance=prepared.built.tuple_provenance,
                proposal_family=prepared.built.proposal_family,
                local_mapping_kind=prepared.built.local_mapping_kind,
                local_mapping_nparams=prepared.built.local_mapping_nparams,
                direct_metadata=prepared.built.direct_metadata,
                seen_child_keys=seen_child_keys,
            )
            score_s += float(time.perf_counter() - score_started)
            if row is None:
                continue
            rows.append(_apply_row_patch(row, prepared.row_patch))
            scored_candidate_count += 1
    meta_out = dict(meta or {})
    meta_out.update(
        {
            "left_eval_count": int(_n_left),
            "right_eval_count": int(_n_right),
            "pair_candidate_count_total": int(_total_pairs),
            "pair_eval_limit": int(_MAX_PAIR_EVALS),
            "prepared_candidate_count": int(prepared_candidate_count),
            "timing_pair_prepare_s": float(prepare_s),
            "timing_pair_score_s": float(score_s),
            "timing_pair_loop_s": float(time.perf_counter() - loop_started),
        }
    )
    return finalize_direct_preview_rows(
        rows,
        preview_topk=int(preview_topk),
        raw_candidate_count=int(raw_candidate_count),
        scored_candidate_count=int(scored_candidate_count),
        deadline_s=deadline_s,
        meta=meta_out,
    )


def run_collected_pair_closure_search(
    *,
    collect_direct_hole_candidates_fn: Callable[..., tuple[list[tuple[str, tuple]], dict[str, Any]]],
    nvars: int,
    enum_max_depth: int,
    enum_max_trees: int,
    var_dims,
    left_target_dim,
    right_target_dim,
    pool_nodes,
    pool_dims,
    left_shortlist_k: int,
    right_shortlist_k: int,
    left_pin_predicate: Callable[[tuple], bool] | None,
    right_pin_predicate: Callable[[tuple], bool] | None,
    prepare_candidate_fn: Callable[
        [tuple[torch.Tensor, torch.Tensor, str, tuple], tuple[torch.Tensor, torch.Tensor, str, tuple]],
        PreparedClosureCandidate | None,
    ],
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    max_depth: int,
    y_dims,
    parent_stats: Mapping[str, int],
    preview_topk: int,
    deadline_s: float | None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    left_collect_started = time.perf_counter()
    left_candidates, left_meta = collect_shortlisted_hole_candidates(
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        nvars=int(nvars),
        enum_max_depth=int(enum_max_depth),
        enum_max_trees=int(enum_max_trees),
        var_dims=var_dims,
        target_dim=left_target_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        shortlist_k=int(left_shortlist_k),
        deadline_s=deadline_s,
        pin_predicate=left_pin_predicate,
    )
    left_collect_s = float(time.perf_counter() - left_collect_started)
    right_collect_started = time.perf_counter()
    right_candidates, right_meta = collect_shortlisted_hole_candidates(
        collect_direct_hole_candidates_fn=collect_direct_hole_candidates_fn,
        nvars=int(nvars),
        enum_max_depth=int(enum_max_depth),
        enum_max_trees=int(enum_max_trees),
        var_dims=var_dims,
        target_dim=right_target_dim,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        shortlist_k=int(right_shortlist_k),
        deadline_s=deadline_s,
        pin_predicate=right_pin_predicate,
    )
    right_collect_s = float(time.perf_counter() - right_collect_started)
    merged_meta = {
        "left_candidates": dict(left_meta or {}),
        "right_candidates": dict(right_meta or {}),
        "timing_collect_left_s": float(left_collect_s),
        "timing_collect_right_s": float(right_collect_s),
        **dict(meta or {}),
    }
    if bool(dict(left_meta or {}).get("deadline_exceeded", False)) or bool(
        dict(right_meta or {}).get("deadline_exceeded", False)
    ):
        merged_meta["deadline_exceeded"] = True
        return [], "direct_deadline_exceeded", merged_meta
    if not left_candidates or not right_candidates:
        return [], "direct_no_hole_candidates", merged_meta
    left_eval_started = time.perf_counter()
    left_rows = evaluate_candidate_node_map(
        left_candidates,
        x_fit=x_fit,
        x_probe=x_probe,
        deadline_s=deadline_s,
    )
    left_eval_s = float(time.perf_counter() - left_eval_started)
    right_eval_started = time.perf_counter()
    right_rows = evaluate_candidate_node_map(
        right_candidates,
        x_fit=x_fit,
        x_probe=x_probe,
        deadline_s=deadline_s,
    )
    right_eval_s = float(time.perf_counter() - right_eval_started)
    merged_meta["left_shortlist_count"] = int(len(left_rows))
    merged_meta["right_shortlist_count"] = int(len(right_rows))
    merged_meta["timing_eval_left_s"] = float(left_eval_s)
    merged_meta["timing_eval_right_s"] = float(right_eval_s)
    if not left_rows or not right_rows:
        return [], "direct_no_scored_candidates", merged_meta
    pair_started = time.perf_counter()
    rows, status, meta_out = run_pair_closure_search(
        left_rows=left_rows,
        right_rows=right_rows,
        prepare_candidate_fn=prepare_candidate_fn,
        y_fit=y_fit,
        y_probe=y_probe,
        max_depth=int(max_depth),
        var_dims=var_dims,
        y_dims=y_dims,
        parent_stats=parent_stats,
        preview_topk=int(preview_topk),
        deadline_s=deadline_s,
        meta=merged_meta,
    )
    meta_final = dict(meta_out or {})
    meta_final["timing_pair_search_s"] = float(time.perf_counter() - pair_started)
    return rows, status, meta_final


def run_subset_closure_search(
    *,
    eval_rows: Sequence[dict[str, Any]],
    subset_max_arity: int,
    prepare_candidate_fn: Callable[[tuple[dict[str, Any], ...]], PreparedClosureCandidate | None],
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    max_depth: int,
    var_dims,
    y_dims,
    parent_stats: Mapping[str, int],
    preview_topk: int,
    deadline_s: float | None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_child_keys: set[str] = set()
    raw_candidate_count = 0
    scored_candidate_count = 0
    prepared_candidate_count = 0
    prepare_s = 0.0
    score_s = 0.0
    loop_started = time.perf_counter()
    max_subset = min(int(subset_max_arity), int(len(list(eval_rows or ()))))
    for subset_size in range(1, max_subset + 1):
        for combo in itertools.combinations(list(eval_rows or ()), subset_size):
            if deadline_exceeded(deadline_s):
                break
            raw_candidate_count += 1
            prepare_started = time.perf_counter()
            prepared = prepare_candidate_fn(combo)
            prepare_s += float(time.perf_counter() - prepare_started)
            if not isinstance(prepared, PreparedClosureCandidate):
                continue
            prepared_candidate_count += 1
            score_started = time.perf_counter()
            row = score_direct_closure_candidate(
                bound_closure=prepared.built.bound_closure,
                design=prepared.built.design,
                y_fit=y_fit,
                y_probe=y_probe,
                max_depth=int(max_depth),
                var_dims=var_dims,
                y_dims=y_dims,
                candidate_subtree_node=prepared.candidate_subtree_node,
                parent_sub_size=int(parent_stats["parent_sub_size"]),
                parent_sub_depth=int(parent_stats["parent_sub_depth"]),
                parent_size=int(parent_stats["parent_size"]),
                parent_depth=int(parent_stats["parent_depth"]),
                generation_source=prepared.built.generation_source,
                tuple_provenance=prepared.built.tuple_provenance,
                proposal_family=prepared.built.proposal_family,
                local_mapping_kind=prepared.built.local_mapping_kind,
                local_mapping_nparams=prepared.built.local_mapping_nparams,
                direct_metadata=prepared.built.direct_metadata,
                seen_child_keys=seen_child_keys,
            )
            score_s += float(time.perf_counter() - score_started)
            if row is None:
                continue
            rows.append(_apply_row_patch(row, prepared.row_patch))
            scored_candidate_count += 1
        if deadline_exceeded(deadline_s):
            break
    meta_out = dict(meta or {})
    meta_out.update(
        {
            "prepared_candidate_count": int(prepared_candidate_count),
            "timing_subset_prepare_s": float(prepare_s),
            "timing_subset_score_s": float(score_s),
            "timing_subset_loop_s": float(time.perf_counter() - loop_started),
        }
    )
    return finalize_direct_preview_rows(
        rows,
        preview_topk=int(preview_topk),
        raw_candidate_count=int(raw_candidate_count),
        scored_candidate_count=int(scored_candidate_count),
        deadline_s=deadline_s,
        meta=meta_out,
    )


def run_seed_subset_closure_search(
    *,
    seed_blocks: Sequence[Any],
    prepare_candidate_fn: Callable[[tuple[dict[str, Any], ...]], PreparedClosureCandidate | None],
    x_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    subset_max_arity: int,
    max_depth: int,
    var_dims,
    y_dims,
    parent_stats: Mapping[str, int],
    preview_topk: int,
    deadline_s: float | None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    eval_started = time.perf_counter()
    eval_rows = evaluate_seed_blocks(
        seed_blocks,
        x_fit=x_fit,
        x_probe=x_probe,
        deadline_s=deadline_s,
    )
    merged_meta = {
        **dict(meta or {}),
        "base_count": int(len(eval_rows)),
        "timing_seed_eval_s": float(time.perf_counter() - eval_started),
    }
    if not eval_rows:
        return [], "direct_no_scored_candidates", merged_meta
    return run_subset_closure_search(
        eval_rows=eval_rows,
        subset_max_arity=int(subset_max_arity),
        prepare_candidate_fn=prepare_candidate_fn,
        y_fit=y_fit,
        y_probe=y_probe,
        max_depth=int(max_depth),
        var_dims=var_dims,
        y_dims=y_dims,
        parent_stats=parent_stats,
        preview_topk=int(preview_topk),
        deadline_s=deadline_s,
        meta=merged_meta,
    )


__all__ = [
    "CustomDirectSearchPlan",
    "PairCollectedSearchPlan",
    "PreparedCandidatesSearchPlan",
    "PreparedClosureCandidate",
    "SeedSubsetSearchPlan",
    "SingleHoleCollectedSearchPlan",
    "execute_direct_search_plan",
    "run_collected_pair_closure_search",
    "run_collected_single_hole_search",
    "run_pair_closure_search",
    "run_prepared_candidate_search",
    "run_seed_subset_closure_search",
    "run_single_hole_closure_search",
    "run_subset_closure_search",
]
