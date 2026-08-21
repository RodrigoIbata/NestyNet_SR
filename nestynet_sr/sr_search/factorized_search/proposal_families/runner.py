# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import inspect
import math
import os
import time
from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch

from ..atom_policy import build_aux_policy_plan
from ..basis_state import BasisState, ProposalContext, enrich_closure_candidate_row
from ..expr_ast import build_pool, is_valid_node, node_dims, node_str
from .common import deadline_exceeded, record_status
from .direct import collect_direct_hole_candidates, solve_direct_operator_preview_rows
from .scaffold_enum import enumerate_operator_applications
from .steering import allocate_family_budgets
from .types import OperatorApplication



_OUTER_SCAFFOLD_FASTTRACK_MSE = 1.0e-8


def _proposal_lane_budget_split(
    *,
    max_scaffolds: int,
    families: Sequence[str] | None,
    allow_augmented: bool,
    seed_mode: bool = False,
    force_augmented: bool = False,
) -> tuple[int, int]:
    total = max(0, int(max_scaffolds))
    if total <= 0:
        return 0, 0
    if not allow_augmented:
        return total, 0
    # In the empty-basis seed round, keep the full scaffold budget on the
    # canonical core lane. The augmented lane is more useful once a basis
    # already exists; early siphoning can starve whole families before their
    # mid-order operator forms ever surface.
    if bool(seed_mode):
        return total, 0
    family_count = max(1, len([str(v) for v in list(families or ()) if str(v or "").strip()]))
    aug_reserve = min(max(2, family_count), max(0, total // 4))
    if aug_reserve <= 0 and total >= 4:
        aug_reserve = 1
    if bool(force_augmented):
        if total >= 4:
            # Aux atoms are only useful if the augmented lane can actually
            # score a small beam of cheap affine relation terms.
            aug_reserve = max(int(aug_reserve), min(12, max(3, total - 2)))
        elif total >= 2 and aug_reserve <= 0:
            aug_reserve = 1
    core_budget = max(1, total - aug_reserve)
    aug_budget = max(0, total - core_budget)
    return core_budget, aug_budget


def _proposal_spec_key(spec: OperatorApplication) -> str:
    parent_key = str(node_str(spec.parent_node)) if isinstance(spec.parent_node, tuple) and is_valid_node(spec.parent_node) else ""
    family_key = str(getattr(spec, "family", "") or "")
    operator_key = str(getattr(spec, "operator_id", "") or "")
    if parent_key:
        return f"{family_key}::{operator_key}::{parent_key}"
    return f"{family_key}::{operator_key}::{str(spec.scaffold_id)}"


def _value_uses_aux_seed(value: Any) -> bool:
    def _is_aux_source(text: str) -> bool:
        return text.startswith("aux:emergent") or text.startswith("aux:policy")

    source = str(getattr(value, "source", "") or "")
    metadata = getattr(value, "metadata", None)
    if _is_aux_source(source):
        return True
    if isinstance(metadata, Mapping):
        origin = str(dict(metadata or {}).get("origin", "") or "")
        if _is_aux_source(origin):
            return True
    if isinstance(value, Mapping):
        origin = str(dict(value.get("metadata", {}) or {}).get("origin", "") or "")
        source = str(value.get("source", "") or "")
        if _is_aux_source(origin) or _is_aux_source(source):
            return True
        return any(_value_uses_aux_seed(child) for child in dict(value).values())
    if isinstance(value, (list, tuple)) and not (
        isinstance(value, tuple) and value and isinstance(value[0], str)
    ):
        return any(_value_uses_aux_seed(child) for child in value)
    return False


def _spec_uses_aux_seed(spec: OperatorApplication) -> bool:
    if not isinstance(spec, OperatorApplication):
        return False
    if _value_uses_aux_seed(getattr(spec, "bindings", None)):
        return True
    return _value_uses_aux_seed(getattr(spec, "metadata", None))


def _is_fasttrack_candidate(row: Mapping[str, Any] | None, *, threshold: float = _OUTER_SCAFFOLD_FASTTRACK_MSE) -> bool:
    if not isinstance(row, Mapping):
        return False
    probe_mse = row.get("local_probe_mse", math.inf)
    fit_mse = row.get("local_fit_mse", math.inf)
    try:
        probe_value = float(probe_mse)
    except Exception:
        probe_value = math.inf
    try:
        fit_value = float(fit_mse)
    except Exception:
        fit_value = math.inf
    return (math.isfinite(probe_value) and probe_value <= float(threshold)) or (
        math.isfinite(fit_value) and fit_value <= float(threshold)
    )


def _supports_keyword_arg(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return str(name) in sig.parameters


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def _finite_float(value: Any, default: float = math.inf) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _format_seconds(value: Any) -> str | None:
    seconds = _finite_float(value)
    if not math.isfinite(seconds):
        return None
    if seconds >= 10.0:
        return f"{seconds:.1f}s"
    return f"{seconds:.2f}s"


def _direct_timing_summary(meta: Mapping[str, Any]) -> str:
    if not isinstance(meta, Mapping):
        return ""
    parts: list[str] = []

    def add(label: str, key: str, source: Mapping[str, Any] | None = None) -> None:
        src = meta if source is None else source
        if not isinstance(src, Mapping) or key not in src:
            return
        rendered = _format_seconds(src.get(key))
        if rendered is not None:
            parts.append(f"{label}={rendered}")

    left_meta = meta.get("left_candidates", {})
    right_meta = meta.get("right_candidates", {})
    if not isinstance(left_meta, Mapping):
        left_meta = {}
    if not isinstance(right_meta, Mapping):
        right_meta = {}

    add("collectL", "timing_collect_left_s")
    add("enumL", "timing_enum_s", left_meta)
    add("shortL", "timing_shortlist_s", left_meta)
    add("evalL", "timing_eval_left_s")
    add("collectR", "timing_collect_right_s")
    add("enumR", "timing_enum_s", right_meta)
    add("shortR", "timing_shortlist_s", right_meta)
    add("evalR", "timing_eval_right_s")
    add("pairPrep", "timing_pair_prepare_s")
    add("pairScore", "timing_pair_score_s")
    add("pairLoop", "timing_pair_loop_s")
    add("singleCollect", "timing_collect_single_s")
    add("singlePrep", "timing_single_prepare_s")
    add("singleScore", "timing_single_score_s")
    add("singleLoop", "timing_single_loop_s")
    add("preparedScore", "timing_prepared_score_s")
    add("seedEval", "timing_seed_eval_s")
    add("subsetPrep", "timing_subset_prepare_s")
    add("subsetScore", "timing_subset_score_s")
    add("rationalSingle", "timing_rational_single_s")
    add("rationalMulti", "timing_rational_multi_s")

    mt_meta = meta.get("multi_term_rational_meta", {})
    if isinstance(mt_meta, Mapping):
        add("mtCollectN", "timing_multi_collect_num_s", mt_meta)
        add("mtCollectD", "timing_multi_collect_den_s", mt_meta)
        add("mtEval", "timing_multi_eval_s", mt_meta)
        add("mtScreen", "timing_multi_screen_s", mt_meta)
        add("mtSupport", "timing_multi_support_s", mt_meta)
        add("mtPreview", "timing_multi_preview_s", mt_meta)

    if not parts:
        return ""
    return " timing=" + ",".join(parts)


def _direct_count_summary(meta: Mapping[str, Any]) -> str:
    if not isinstance(meta, Mapping):
        return ""
    fields: list[str] = []
    mt_status = str(meta.get("multi_term_rational_status", "") or "")
    if mt_status:
        fields.append(f"mt_status={mt_status}")
    mt_meta = meta.get("multi_term_rational_meta", {})
    if isinstance(mt_meta, Mapping):
        if "support_cache_hit" in mt_meta:
            fields.append(f"mt_cache_hit={bool(mt_meta.get('support_cache_hit', False))}")
        if "ranked_support_enable" in mt_meta:
            fields.append(f"mt_ranked={bool(mt_meta.get('ranked_support_enable', False))}")
    for label, key in (
        ("left", "left_eval_count"),
        ("right", "right_eval_count"),
        ("pairs", "pair_candidate_count_total"),
        ("prepared", "prepared_candidate_count"),
    ):
        if key not in meta:
            continue
        try:
            value = int(meta.get(key, 0) or 0)
        except Exception:
            continue
        fields.append(f"{label}={value}")
    if not fields:
        return ""
    return " " + " ".join(fields)


def _canonical_core_pool(
    *,
    nvars: int,
    var_dims,
) -> tuple[tuple[tuple, ...], tuple[Any, ...]]:
    try:
        pool_nodes = tuple(build_pool(max(0, int(nvars))) or ())
    except Exception:
        pool_nodes = ()
    pool_dims: list[Any] = []
    if var_dims is None:
        pool_dims = [None] * len(pool_nodes)
    else:
        for node in pool_nodes:
            try:
                pool_dims.append(node_dims(node, var_dims))
            except Exception:
                pool_dims.append(None)
    return pool_nodes, tuple(pool_dims)


def _family_plan_entries(
    *,
    families: Sequence[str] | None,
    max_scaffolds: int,
    anchors_per_family: int,
    proposal_context: ProposalContext | None,
    family_allocator_fn,
) -> list[dict[str, Any]]:
    family_tokens = [str(token or "").strip() for token in list(families or ()) if str(token or "").strip()]
    total = max(0, int(max_scaffolds))
    if not family_tokens or total <= 0:
        return []

    raw_entries: list[dict[str, Any]] = []
    if callable(family_allocator_fn):
        try:
            family_plan = family_allocator_fn(
                families=family_tokens,
                max_scaffolds=int(total),
                anchors_per_family=int(anchors_per_family),
                context=proposal_context,
            )
        except Exception:
            family_plan = None
        if isinstance(family_plan, Mapping):
            for entry in list(family_plan.get("entries", []) or ()):
                if not isinstance(entry, Mapping):
                    continue
                family = str(entry.get("family", "") or "").strip()
                if not family or family not in family_tokens:
                    continue
                try:
                    budget = max(0, int(entry.get("max_scaffolds", 0) or 0))
                except Exception:
                    budget = 0
                try:
                    anchor_cap = max(1, int(entry.get("anchors_per_family", anchors_per_family) or anchors_per_family))
                except Exception:
                    anchor_cap = max(1, int(anchors_per_family))
                if budget <= 0:
                    continue
                raw_entries.append(
                    {
                        "family": family,
                        "max_scaffolds": int(budget),
                        "anchors_per_family": int(anchor_cap),
                    }
                )

    if not raw_entries:
        n_families = max(1, len(family_tokens))
        base_share = int(total // n_families)
        remainder = int(total % n_families)
        for idx, family in enumerate(family_tokens):
            budget = int(base_share + (1 if idx < remainder else 0))
            if budget <= 0:
                continue
            raw_entries.append(
                {
                    "family": family,
                    "max_scaffolds": int(budget),
                    "anchors_per_family": max(1, int(anchors_per_family)),
                }
            )

    used = sum(int(entry.get("max_scaffolds", 0) or 0) for entry in raw_entries)
    remaining = max(0, int(total) - int(used))
    idx = 0
    while remaining > 0 and raw_entries:
        raw_entries[idx % len(raw_entries)]["max_scaffolds"] = int(raw_entries[idx % len(raw_entries)].get("max_scaffolds", 0) or 0) + 1
        remaining -= 1
        idx += 1

    return raw_entries


def _enumerator_kwargs(
    *,
    families: Sequence[str] | None,
    nvars: int,
    y_dims,
    var_dims,
    pool_nodes,
    pool_dims,
    anchors_per_family: int,
    max_scaffolds: int,
    proposal_context: ProposalContext | None,
    enumerator,
    basis_seed_mode: str = "merged",
) -> dict[str, Any]:
    kwargs = {
        "families": families,
        "nvars": int(nvars),
        "y_dims": y_dims,
        "var_dims": var_dims,
        "pool_nodes": pool_nodes,
        "pool_dims": pool_dims,
        "anchors_per_family": int(anchors_per_family),
        "max_scaffolds": int(max_scaffolds),
    }
    if _supports_keyword_arg(enumerator, "basis_seed_mode"):
        kwargs["basis_seed_mode"] = str(basis_seed_mode or "merged")
    if isinstance(proposal_context, ProposalContext) and str(basis_seed_mode or "merged") != "core_only":
        if _supports_keyword_arg(enumerator, "basis_state"):
            kwargs["basis_state"] = proposal_context.basis_state
        if _supports_keyword_arg(enumerator, "basis_state_beam"):
            kwargs["basis_state_beam"] = proposal_context.basis_state_beam
        if _supports_keyword_arg(enumerator, "aux_seed_blocks"):
            kwargs["aux_seed_blocks"] = tuple(getattr(proposal_context, "aux_seed_blocks", ()) or ())
        if _supports_keyword_arg(enumerator, "atom_library"):
            kwargs["atom_library"] = getattr(proposal_context, "atom_library", None)
    return kwargs


def _enumerate_proposals(
    *,
    families: Sequence[str] | None,
    nvars: int,
    y_dims,
    var_dims,
    pool_nodes,
    pool_dims,
    anchors_per_family: int,
    max_scaffolds: int,
    enumerate_operator_applications_fn,
    proposal_context: ProposalContext | None = None,
    basis_seed_mode: str = "merged",
    family_plan_entries: Sequence[Mapping[str, Any]] | None = None,
) -> list[OperatorApplication]:
    if not callable(enumerate_operator_applications_fn):
        return []
    entries = [dict(entry) for entry in list(family_plan_entries or ()) if isinstance(entry, Mapping)]
    if not entries:
        entries = [
            {
                "family": family,
                "max_scaffolds": int(max_scaffolds),
                "anchors_per_family": int(anchors_per_family),
            }
            for family in list(families or ())
            if str(family or "").strip()
        ]
    out: list[OperatorApplication] = []
    for entry in entries:
        remaining = max(0, int(max_scaffolds) - len(out))
        if remaining <= 0:
            break
        family = str(entry.get("family", "") or "").strip()
        if not family:
            continue
        try:
            family_budget = min(int(remaining), max(0, int(entry.get("max_scaffolds", remaining) or remaining)))
        except Exception:
            family_budget = int(remaining)
        if family_budget <= 0:
            continue
        try:
            family_anchor_cap = max(1, int(entry.get("anchors_per_family", anchors_per_family) or anchors_per_family))
        except Exception:
            family_anchor_cap = max(1, int(anchors_per_family))
        apps = enumerate_operator_applications_fn(
            **_enumerator_kwargs(
                families=[family],
                nvars=int(nvars),
                y_dims=y_dims,
                var_dims=var_dims,
                pool_nodes=pool_nodes,
                pool_dims=pool_dims,
                anchors_per_family=int(family_anchor_cap),
                max_scaffolds=int(family_budget),
                proposal_context=proposal_context,
                enumerator=enumerate_operator_applications_fn,
                basis_seed_mode=str(basis_seed_mode or "merged"),
            )
        )
        out.extend(
            spec
            for spec in list(apps or ())
            if isinstance(spec, OperatorApplication)
        )
    return out[: max(0, int(max_scaffolds))]


def run_closure_search_pass_impl(
    *,
    families: Sequence[str] | None,
    nvars: int,
    max_scaffolds: int,
    anchors_per_family: int,
    max_depth: int,
    poly_degree: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    var_dims,
    y_dims,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    safe_eps: float,
    preview_topk: int,
    beam_cfg: Mapping[str, Any],
    solver_kwargs: Mapping[str, Any],
    deadline_s: float | None = None,
    proposal_context: ProposalContext | None = None,
    family_allocator_fn=None,
    enumerate_operator_applications_fn=None,
    solve_direct_operator_preview_rows_fn=None,
    # Legacy parameters kept for backward-compatible call sites but no longer used.
    enumerate_closure_search_specs_fn=None,
    legacy_scaffold_to_operator_fn=None,
    prefer_legacy_scaffold_enumeration: bool = False,
    legacy_direct_dispatch_overridden: bool = False,
    fit_scaffold_mapping_fn=None,
    build_scaffold_beam_state_fn=None,
    solve_inverse_spec_preview_rows_fn=None,
    solve_direct_periodic_add_preview_rows_fn=None,
    solve_direct_exp_preview_rows_fn=None,
    solve_direct_log_preview_rows_fn=None,
    solve_direct_power_preview_rows_fn=None,
    solve_direct_quadratic_preview_rows_fn=None,
    solve_direct_rational_affine_preview_rows_fn=None,
) -> dict[str, Any]:
    # Resolve defaults at call time so monkeypatching works for tests.
    if family_allocator_fn is None:
        family_allocator_fn = allocate_family_budgets
    if enumerate_operator_applications_fn is None:
        enumerate_operator_applications_fn = enumerate_operator_applications
    if solve_direct_operator_preview_rows_fn is None:
        solve_direct_operator_preview_rows_fn = solve_direct_operator_preview_rows
    stats: dict[str, Any] = {
        "families_considered": 0,
        "scaffolds_enumerated": 0,
        "scaffolds_considered": 0,
        "preview_calls": 0,
        "preview_candidates": 0,
        "direct_calls": 0,
        "direct_candidates": 0,
        "direct_anchor_lift_attempts": 0,
        "direct_anchor_lift_applied": 0,
        "deadline_exceeded": False,
        "status_counts": {},
        "failure_examples": [],
        "proposal_object_mode": "operator_native",
    }
    candidate_rows: list[dict[str, Any]] = []
    trace_env = _env_flag("CLOSURE_TRACE_SCAFFOLDS")
    try:
        debug_topk = int(dict(beam_cfg or {}).get("debug_topk", 0) or 0)
    except Exception:
        debug_topk = 0
    trace_scaffolds = bool(debug_topk > 0) if trace_env is None else bool(trace_env)
    if deadline_exceeded(deadline_s):
        stats["deadline_exceeded"] = True
        record_status(stats, "deadline_exceeded")
        return {"candidate_rows": candidate_rows, "stats": stats}
    proposal_specs: list[tuple[str, OperatorApplication]] = []
    stats["proposal_lane_budgets"] = {}
    stats["scaffolds_enumerated_by_lane"] = {}
    stats["fasttrack_candidates"] = 0
    stats["aux_scaffolds_enumerated"] = 0
    stats["protected_aux_scaffolds_enumerated"] = 0
    aux_seed_blocks = (
        tuple(getattr(proposal_context, "aux_seed_blocks", ()) or ())
        if isinstance(proposal_context, ProposalContext)
        else ()
    )
    atom_library = getattr(proposal_context, "atom_library", None) if isinstance(proposal_context, ProposalContext) else None
    stats["atom_policy_library_records"] = int(len(tuple(getattr(atom_library, "records", ()) or ())))
    stats["atom_policy_library_relations"] = int(len(tuple(getattr(atom_library, "relations", ()) or ())))
    stats["aux_seed_blocks_count"] = int(len(aux_seed_blocks))
    stats["aux_seed_block_exprs"] = [
        str(node_str(getattr(block, "node", None)))
        for block in aux_seed_blocks
        if isinstance(getattr(block, "node", None), tuple) and is_valid_node(getattr(block, "node", None))
    ]
    seed_mode = (
        isinstance(proposal_context, ProposalContext)
        and (not isinstance(getattr(proposal_context, "basis_state", None), BasisState))
        and (not bool(tuple(getattr(proposal_context, "basis_state_beam", ()) or ())))
        and (not bool(aux_seed_blocks))
    )
    allow_augmented_lane = bool(pool_nodes) or bool(aux_seed_blocks) or (
        isinstance(proposal_context, ProposalContext)
        and (
            proposal_context.basis_state is not None
            or bool(tuple(getattr(proposal_context, "basis_state_beam", ()) or ()))
        )
    )
    core_pool_nodes, core_pool_dims = _canonical_core_pool(
        nvars=int(nvars),
        var_dims=var_dims,
    )
    if int(max_scaffolds) > 0 and families:
        family_plan = None
        if callable(family_allocator_fn):
            try:
                family_plan = family_allocator_fn(
                    families=families,
                    max_scaffolds=int(max_scaffolds),
                    anchors_per_family=int(anchors_per_family),
                    context=proposal_context,
                )
            except Exception:
                family_plan = None
        if isinstance(proposal_context, ProposalContext):
            stats["proposal_context"] = proposal_context.to_dict()
        if isinstance(family_plan, Mapping):
            stats["family_priority_scores"] = {
                str(key): float(value)
                for key, value in dict(family_plan.get("scores", {}) or {}).items()
                if isinstance(value, (int, float))
            }
            stats["family_budget_plan"] = [
                dict(entry)
                for entry in list(family_plan.get("entries", []) or [])
                if isinstance(entry, Mapping)
            ]
            stats["family_priority_decomposition"] = {
                str(key): dict(value)
                for key, value in dict(family_plan.get("score_decomposition", {}) or {}).items()
                if isinstance(value, Mapping)
            }
            stats["family_steering_applied"] = bool(family_plan.get("steered", False))
        core_budget, aug_reserve = _proposal_lane_budget_split(
            max_scaffolds=int(max_scaffolds),
            families=families,
            allow_augmented=allow_augmented_lane,
            seed_mode=seed_mode,
            force_augmented=bool(aux_seed_blocks),
        )
        stats["proposal_lane_budgets"] = {
            "core": int(core_budget),
            "basis_augmented": int(aug_reserve),
        }
        stats["family_budget_plan_by_lane"] = {}

        core_proposal_context = proposal_context
        if isinstance(proposal_context, ProposalContext):
            core_proposal_context = replace(
                proposal_context,
                aux_seed_blocks=(),
                atom_library=None,
            )
        core_family_plan = _family_plan_entries(
            families=families,
            max_scaffolds=int(core_budget),
            anchors_per_family=int(anchors_per_family),
            proposal_context=core_proposal_context,
            family_allocator_fn=family_allocator_fn,
        )
        stats["family_budget_plan_by_lane"]["core"] = [dict(entry) for entry in core_family_plan]

        core_specs = _enumerate_proposals(
            families=families,
            nvars=int(nvars),
            y_dims=y_dims,
            var_dims=var_dims,
            pool_nodes=core_pool_nodes,
            pool_dims=core_pool_dims,
            anchors_per_family=int(anchors_per_family),
            max_scaffolds=int(core_budget),
            enumerate_operator_applications_fn=enumerate_operator_applications_fn,
            proposal_context=core_proposal_context,
            basis_seed_mode="core_only",
            family_plan_entries=core_family_plan,
        )
        remaining_after_core = max(0, int(max_scaffolds) - int(len(core_specs)))
        aug_budget = 0
        aug_specs: list[OperatorApplication] = []
        if allow_augmented_lane and (not bool(seed_mode)) and remaining_after_core > 0:
            aug_budget = max(int(aug_reserve), int(remaining_after_core))
            aug_budget = min(int(max_scaffolds), int(aug_budget), int(remaining_after_core))
            if aug_budget > 0:
                protected_aug_budget = 0
                protected_aug_specs: list[OperatorApplication] = []
                if aux_seed_blocks:
                    protected_aug_budget = min(
                        int(aug_budget),
                        max(1, min(12, max(3, len(tuple(aux_seed_blocks or ())) * 3))),
                    )
                    protected_aug_plan = build_aux_policy_plan(
                        families=families,
                        library=(
                            getattr(proposal_context, "atom_library", None)
                            if isinstance(proposal_context, ProposalContext)
                            else None
                        ),
                        max_scaffolds=int(protected_aug_budget),
                        anchors_per_family=max(1, int(anchors_per_family)),
                    )
                    if not protected_aug_plan:
                        protected_aug_plan = [
                            {
                                "family": "affine",
                                "max_scaffolds": int(protected_aug_budget),
                                "anchors_per_family": max(1, int(anchors_per_family)),
                                "reason": "aux_fallback_affine",
                            }
                        ]
                    protected_aug_specs = _enumerate_proposals(
                        families=tuple(str(entry.get("family", "")) for entry in protected_aug_plan),
                        nvars=int(nvars),
                        y_dims=y_dims,
                        var_dims=var_dims,
                        pool_nodes=pool_nodes,
                        pool_dims=pool_dims,
                        anchors_per_family=int(anchors_per_family),
                        max_scaffolds=int(protected_aug_budget),
                        enumerate_operator_applications_fn=enumerate_operator_applications_fn,
                        proposal_context=proposal_context,
                        basis_seed_mode="basis_augmented",
                        family_plan_entries=protected_aug_plan,
                    )
                    stats["family_budget_plan_by_lane"]["basis_augmented_protected_aux"] = [
                        dict(entry) for entry in protected_aug_plan
                    ]
                normal_aug_budget = max(0, int(aug_budget) - int(len(protected_aug_specs)))
                aug_family_plan = _family_plan_entries(
                    families=families,
                    max_scaffolds=int(normal_aug_budget),
                    anchors_per_family=int(anchors_per_family),
                    proposal_context=proposal_context,
                    family_allocator_fn=family_allocator_fn,
                )
                stats["family_budget_plan_by_lane"]["basis_augmented"] = [dict(entry) for entry in aug_family_plan]
                normal_aug_specs: list[OperatorApplication] = []
                if normal_aug_budget > 0:
                    normal_aug_specs = _enumerate_proposals(
                        families=families,
                        nvars=int(nvars),
                        y_dims=y_dims,
                        var_dims=var_dims,
                        pool_nodes=pool_nodes,
                        pool_dims=pool_dims,
                        anchors_per_family=int(anchors_per_family),
                        max_scaffolds=int(normal_aug_budget),
                        enumerate_operator_applications_fn=enumerate_operator_applications_fn,
                        proposal_context=proposal_context,
                        basis_seed_mode="basis_augmented",
                        family_plan_entries=aug_family_plan,
                    )
                aug_specs = [*protected_aug_specs, *normal_aug_specs]
                stats["protected_aux_scaffolds_enumerated"] = int(
                    sum(1 for spec in protected_aug_specs if _spec_uses_aux_seed(spec))
                )
        stats["proposal_lane_budgets"]["basis_augmented"] = int(aug_budget)

        tagged_specs: list[tuple[str, OperatorApplication]] = []
        seen_specs: set[str] = set()
        for lane_name, lane_specs in (("core", core_specs), ("basis_augmented", aug_specs)):
            stats["scaffolds_enumerated_by_lane"][lane_name] = int(len(list(lane_specs or ())))
            if str(lane_name) == "basis_augmented":
                stats["aux_scaffolds_enumerated"] = int(
                    sum(1 for spec in list(lane_specs or ()) if _spec_uses_aux_seed(spec))
                )
            for spec in list(lane_specs or ()):
                if not isinstance(spec, OperatorApplication):
                    continue
                spec_key = _proposal_spec_key(spec)
                if spec_key in seen_specs:
                    continue
                seen_specs.add(spec_key)
                tagged_specs.append((lane_name, spec))
        proposal_specs = tagged_specs
    stats["families_considered"] = int(len(list(families or ())))
    stats["scaffolds_enumerated"] = int(len(proposal_specs))
    if not proposal_specs or int(max_scaffolds) <= 0:
        return {"candidate_rows": candidate_rows, "stats": stats}

    for scaffold_idx, (proposal_lane, spec) in enumerate(proposal_specs):
        if deadline_exceeded(deadline_s):
            stats["deadline_exceeded"] = True
            record_status(stats, "deadline_exceeded")
            break
        stats["scaffolds_considered"] = int(stats.get("scaffolds_considered", 0)) + 1
        direct_rows: list[dict[str, Any]] = []
        direct_status = "direct_not_supported"
        direct_meta: Mapping[str, Any] | dict[str, Any] = {}
        direct_started = time.perf_counter()
        if callable(solve_direct_operator_preview_rows_fn):
            lane_pool_nodes = core_pool_nodes if str(proposal_lane) == "core" else pool_nodes
            lane_pool_dims = core_pool_dims if str(proposal_lane) == "core" else pool_dims
            direct_kwargs = {
                "nvars": int(nvars),
                "max_depth": int(max_depth),
                "x_fit": x_fit,
                "y_fit": y_fit,
                "x_probe": x_probe,
                "y_probe": y_probe,
                "var_dims": var_dims,
                "y_dims": y_dims,
                "pool_nodes": lane_pool_nodes,
                "pool_dims": lane_pool_dims,
                "preview_topk": int(preview_topk),
                "solver_kwargs": solver_kwargs,
                "deadline_s": deadline_s,
                "collect_direct_hole_candidates_fn": collect_direct_hole_candidates,
            }
            if _supports_keyword_arg(solve_direct_operator_preview_rows_fn, "proposal_context"):
                direct_kwargs["proposal_context"] = proposal_context
            direct_rows, direct_status, direct_meta = solve_direct_operator_preview_rows_fn(
                spec,
                **direct_kwargs,
            )
        direct_elapsed_s = float(time.perf_counter() - direct_started)
        direct_meta_dict = dict(direct_meta or {})
        best_probe_mse = min(
            (
                _finite_float(row.get("local_probe_mse", math.inf))
                for row in direct_rows
                if isinstance(row, Mapping)
            ),
            default=math.inf,
        )
        stats["direct_wall_seconds"] = float(stats.get("direct_wall_seconds", 0.0) or 0.0) + direct_elapsed_s
        if trace_scaffolds:
            raw_count = direct_meta_dict.get("candidate_count_raw", direct_meta_dict.get("raw_candidate_count", 0))
            scored_count = direct_meta_dict.get("candidate_count_scored", direct_meta_dict.get("scored_candidate_count", 0))
            best_text = f"{best_probe_mse:.3e}" if math.isfinite(best_probe_mse) else "inf"
            count_text = _direct_count_summary(direct_meta_dict)
            timing_text = _direct_timing_summary(direct_meta_dict)
            print(
                f"[closure] scaffold {scaffold_idx + 1}/{len(proposal_specs)} "
                f"lane={proposal_lane} family={getattr(spec, 'family', '')} "
                f"op={getattr(spec, 'operator_id', '') or getattr(spec, 'scaffold_id', '')} "
                f"status={direct_status} raw={raw_count} scored={scored_count} "
                f"rows={len(direct_rows)}{count_text} best_probe={best_text} "
                f"elapsed={direct_elapsed_s:.2f}s{timing_text}",
                flush=True,
            )
        if direct_status == "direct_not_supported":
            record_status(stats, "direct_not_supported")
            continue

        stats["preview_calls"] = int(stats.get("preview_calls", 0)) + 1
        stats["direct_calls"] = int(stats.get("direct_calls", 0)) + 1
        record_status(stats, direct_status)
        if bool(direct_meta_dict.get("deadline_exceeded", False)):
            stats["deadline_exceeded"] = True
        stats["preview_candidates"] = int(stats.get("preview_candidates", 0)) + int(len(direct_rows))
        stats["direct_candidates"] = int(stats.get("direct_candidates", 0)) + int(len(direct_rows))
        stats["direct_anchor_lift_attempts"] = int(stats.get("direct_anchor_lift_attempts", 0)) + int(direct_meta_dict.get("anchor_lift_attempts", 0) or 0)
        stats["direct_anchor_lift_applied"] = int(stats.get("direct_anchor_lift_applied", 0)) + int(direct_meta_dict.get("anchor_lift_applied", 0) or 0)
        for row in direct_rows:
            if not isinstance(row, Mapping):
                continue
            expr = row.get("expr", None)
            if not isinstance(expr, tuple):
                continue
            enriched = dict(row)
            enriched["route"] = "closure_search"
            enriched["action"] = "closure_search"
            enriched["scaffold_family"] = str(spec.family)
            enriched["scaffold_id"] = str(spec.scaffold_id)
            enriched["scaffold_parent_expr"] = str(node_str(spec.parent_node))
            enriched["scaffold_hole_path"] = [int(v) for v in tuple(spec.hole_path)]
            enriched["scaffold_target_mode"] = str(spec.target_mode)
            enriched["scaffold_anchor_expr"] = "" if spec.anchor_node is None else str(node_str(spec.anchor_node))
            enriched["scaffold_anchor_node"] = spec.anchor_node
            enriched["scaffold_metadata"] = dict(spec.metadata or {})
            enriched["operator_id"] = str(getattr(spec, "operator_id", "") or "")
            enriched["operator_spec_key"] = f"{str(spec.family)}::{str(getattr(spec, 'operator_id', '') or spec.scaffold_id)}"
            enriched["proposal_lane"] = str(proposal_lane)
            enriched["proposal_seed_mode"] = "core_only" if str(proposal_lane) == "core" else "basis_augmented"
            enriched["operator_application_obj"] = spec
            enriched["operator_application_dict"] = spec.to_dict() if hasattr(spec, "to_dict") else {}
            if (
                enriched.get("bound_closure_obj", None) is None
                and getattr(spec, "bound_closure", None) is not None
            ):
                enriched["bound_closure_obj"] = spec.bound_closure
                enriched["bound_closure_dict"] = spec.bound_closure.to_dict()
            row_direct_meta = dict(enriched.get("direct_metadata", {}) or {})
            row_anchor_node = row_direct_meta.get("anchor_node", None)
            if not (isinstance(row_anchor_node, tuple) and is_valid_node(row_anchor_node)):
                row_anchor_node = spec.anchor_node
            enriched = enrich_closure_candidate_row(
                enriched,
                family=str(spec.family),
                scaffold_id=str(spec.scaffold_id),
                anchor_node=row_anchor_node,
                scaffold_metadata=dict(spec.metadata or {}),
            )
            enriched["preview_fasttrack"] = bool(_is_fasttrack_candidate(enriched))
            if enriched["preview_fasttrack"]:
                stats["fasttrack_candidates"] = int(stats.get("fasttrack_candidates", 0) or 0) + 1
            candidate_rows.append(enriched)

    return {
        "candidate_rows": candidate_rows,
        "stats": stats,
    }


run_outer_scaffold_pass_impl = run_closure_search_pass_impl


__all__ = ["run_closure_search_pass_impl", "run_outer_scaffold_pass_impl"]
