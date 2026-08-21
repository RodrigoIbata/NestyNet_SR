# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .expr_ast import is_valid_node, node_size, node_str
from .subproblem_spec import (
    SolverProposal,
    serialize_solver_proposal,
)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _row_preview_loss(row: Mapping[str, Any]) -> float | None:
    return _finite_float(
        row.get(
            "witness_energy_total",
            row.get(
                "local_probe_mse",
                row.get("current_best_child_eff_mse", None),
            ),
        )
    )


@dataclass(frozen=True)
class SolverMarketRouteCall:
    route_name: str
    method_name: str
    subroute: str
    runner: Callable[..., Mapping[str, Any]]
    runner_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class _SolverProposalRow:
    proposal: SolverProposal
    row: dict[str, Any]


def _route_row_key(row: Mapping[str, Any], *, fallback_route: str, fallback_rank: int) -> str:
    child_key = str(row.get("child_key", "") or "").strip()
    if child_key:
        return child_key
    expr = row.get("expr", None)
    if expr is not None:
        try:
            return str(node_str(expr))
        except Exception:
            pass
    return f"{str(fallback_route)}:{int(fallback_rank)}"


def _route_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    preview_loss = _row_preview_loss(row)
    fit_mse = _finite_float(row.get("local_fit_mse", None))
    expr = row.get("expr", None)
    try:
        size = int(node_size(expr)) if is_valid_node(expr) else 10**9
    except Exception:
        size = 10**9
    return (
        float("inf") if preview_loss is None else float(preview_loss),
        float("inf") if fit_mse is None else float(fit_mse),
        _safe_int(row.get("solver_market_route_rank", row.get("local_rank", 0)), 0),
        _safe_int(row.get("local_rank", row.get("solver_market_row_rank", 0)), 0),
        int(size),
    )


def _proposal_sort_key(item: _SolverProposalRow) -> tuple[Any, ...]:
    row = item.row
    preview_loss = _finite_float(item.proposal.preview_loss)
    global_probe_mse = _finite_float(item.proposal.global_probe_mse)
    fit_mse = _finite_float(row.get("local_fit_mse", None))
    expr = item.proposal.expr_ast
    try:
        size = int(node_size(expr)) if is_valid_node(expr) else 10**9
    except Exception:
        size = 10**9
    return (
        float("inf") if preview_loss is None else float(preview_loss),
        float("inf") if global_probe_mse is None else float(global_probe_mse),
        float("inf") if fit_mse is None else float(fit_mse),
        _safe_int(row.get("solver_market_route_rank", row.get("local_rank", 0)), 0),
        _safe_int(row.get("local_rank", row.get("solver_market_row_rank", 0)), 0),
        int(size),
    )


def _row_to_solver_proposal(row: Mapping[str, Any]) -> SolverProposal | None:
    if not isinstance(row, Mapping):
        return None
    expr = row.get("expr", None)
    if not is_valid_node(expr):
        return None
    preview_loss = _row_preview_loss(row)
    if preview_loss is None:
        return None
    mapping = row.get("mapping", None)
    mapping_out = dict(mapping) if isinstance(mapping, Mapping) else {}
    local_mapping_kind = str(row.get("local_mapping_kind", "") or "")
    if not mapping_out and local_mapping_kind:
        mapping_out = {"kind": local_mapping_kind}
    family = str(
        row.get("proposal_family", "")
        or row.get("inverse_spec_family", "")
        or row.get("solver_market_subroute", "")
        or row.get("solver_market_route", "")
        or ""
    )
    source = str(
        row.get("generation_source", "")
        or row.get("solver_market_method_name", "")
        or row.get("solver_market_route", "")
        or ""
    )
    metadata = {
        "child_key": str(row.get("child_key", "") or ""),
        "local_fit_mse": _finite_float(row.get("local_fit_mse", None)),
        "solver_market_route": str(row.get("solver_market_route", "") or ""),
        "solver_market_method_name": str(row.get("solver_market_method_name", "") or ""),
        "solver_market_subroute": str(row.get("solver_market_subroute", "") or ""),
        "inverse_spec_generation_kind": str(row.get("inverse_spec_generation_kind", "") or ""),
        "witness_fit_jet_source": str(row.get("witness_fit_jet_source", "") or ""),
        "witness_probe_jet_source": str(row.get("witness_probe_jet_source", "") or ""),
        "witness_numeric_jet_fallback_used": bool(row.get("witness_numeric_jet_fallback_used", False)),
        "witness_exact_jet_used": bool(row.get("witness_exact_jet_used", False)),
    }
    return SolverProposal(
        expr_ast=expr,
        mapping=mapping_out,
        source=source,
        family=family,
        preview_loss=float(preview_loss),
        global_probe_mse=_finite_float(
            row.get(
                "eff_mse",
                row.get(
                    "witness_energy_total",
                    row.get("current_best_child_eff_mse", None),
                ),
            )
        ),
        metadata=metadata,
    )


def _proposal_output_row(item: _SolverProposalRow) -> dict[str, Any]:
    row = dict(item.row)
    row["solver_proposal"] = serialize_solver_proposal(item.proposal)
    return row


def _dedup_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[str, dict[str, Any]] = {}
    for raw_row in list(rows or ()):
        if not isinstance(raw_row, Mapping):
            continue
        row = dict(raw_row)
        route_name = str(row.get("solver_market_route", "") or "")
        route_rank = _safe_int(row.get("solver_market_row_rank", 0), 0)
        key = _route_row_key(row, fallback_route=route_name, fallback_rank=route_rank)
        previous = best_by_key.get(key, None)
        if previous is None or _route_row_sort_key(row) < _route_row_sort_key(previous):
            best_by_key[key] = row
    out = list(best_by_key.values())
    out.sort(key=_route_row_sort_key)
    return out


def _dedup_proposal_rows(rows: Sequence[_SolverProposalRow]) -> list[_SolverProposalRow]:
    best_by_key: dict[str, _SolverProposalRow] = {}
    for item in list(rows or ()):
        if not isinstance(item, _SolverProposalRow):
            continue
        key = _route_row_key(
            item.row,
            fallback_route=str(item.row.get("solver_market_route", "") or ""),
            fallback_rank=_safe_int(item.row.get("solver_market_row_rank", 0), 0),
        )
        previous = best_by_key.get(key, None)
        if previous is None or _proposal_sort_key(item) < _proposal_sort_key(previous):
            best_by_key[key] = item
    out = list(best_by_key.values())
    out.sort(key=_proposal_sort_key)
    return out


def _dedup_child_spec_states(states: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in list(states or ()):
        if not isinstance(row, Mapping):
            continue
        key = (
            str(row.get("spec_kind", "") or ""),
            tuple(int(v) for v in (row.get("path", ()) or ())),
            str(row.get("branch_id", "") or ""),
            tuple(str(v) for v in (row.get("continuation_key", ()) or ())),
            tuple(str(v) for v in (row.get("trace", ()) or ())),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def run_preview_solver_market(
    route_calls: Sequence[SolverMarketRouteCall] | None,
    *,
    preview_topk: int = 4,
    exact_topk: int = 2,
    proposal_objects_enable: bool = False,
) -> dict[str, Any]:
    calls = [call for call in list(route_calls or ()) if isinstance(call, SolverMarketRouteCall)]
    route_summaries: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    raw_proposals: list[_SolverProposalRow] = []
    child_spec_states: list[dict[str, Any]] = []
    solver_meta_by_route: dict[str, dict[str, Any]] = {}

    for route_rank, call in enumerate(calls):
        rows: list[dict[str, Any]] = []
        solver_meta: dict[str, Any] = {}
        error_text = ""
        try:
            result = call.runner(**dict(call.runner_kwargs or {}))
            if isinstance(result, Mapping):
                solver_meta = dict(result.get("solver_meta", {}) or {})
                rows = [
                    dict(row)
                    for row in list(result.get("rows", []) or [])
                    if isinstance(row, Mapping)
                ]
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            solver_meta = {"status": "route_error"}
            rows = []

        annotated_rows: list[dict[str, Any]] = []
        for row_rank, row in enumerate(rows):
            annotated = dict(row)
            annotated["solver_market_route"] = str(call.route_name)
            annotated["solver_market_method_name"] = str(call.method_name)
            annotated["solver_market_subroute"] = str(call.subroute)
            annotated["solver_market_route_rank"] = int(route_rank)
            annotated["solver_market_row_rank"] = int(row_rank)
            annotated_rows.append(annotated)
        raw_rows.extend(annotated_rows)
        if bool(proposal_objects_enable):
            for annotated in annotated_rows:
                proposal = _row_to_solver_proposal(annotated)
                if proposal is None:
                    continue
                raw_proposals.append(_SolverProposalRow(proposal=proposal, row=dict(annotated)))

        route_child_states = [
            dict(row)
            for row in list(solver_meta.get("child_spec_states", []) or [])
            if isinstance(row, Mapping)
        ]
        child_spec_states.extend(route_child_states)
        solver_meta_by_route[str(call.route_name)] = dict(solver_meta)
        best_row = min(annotated_rows, key=_route_row_sort_key) if annotated_rows else None
        route_summaries.append({
            "route_rank": int(route_rank),
            "route_name": str(call.route_name),
            "method_name": str(call.method_name),
            "subroute": str(call.subroute),
            "status": str(solver_meta.get("status", "ok" if annotated_rows else "no_rows") or ""),
            "row_count": int(len(annotated_rows)),
            "child_spec_state_count": int(len(route_child_states)),
            "preview_best_probe_mse": min(
                (
                    float(v)
                    for v in (_finite_float(row.get("local_probe_mse", None)) for row in annotated_rows)
                    if v is not None
                ),
                default=None,
            ),
            "preview_best_fit_jet_source": str(best_row.get("witness_fit_jet_source", "") or "") if best_row else "",
            "preview_best_probe_jet_source": str(best_row.get("witness_probe_jet_source", "") or "") if best_row else "",
            "preview_best_numeric_jet_fallback_used": bool(best_row.get("witness_numeric_jet_fallback_used", False)) if best_row else False,
            "preview_best_exact_jet_used": bool(best_row.get("witness_exact_jet_used", False)) if best_row else False,
            "route_trigger_status": str(solver_meta.get("route_trigger_status", "") or ""),
            "route_trigger_score": _finite_float(solver_meta.get("route_trigger_score", None)),
            "route_trigger_preferred": bool(solver_meta.get("route_trigger_preferred", False)),
            "route_reason_family": str(solver_meta.get("route_reason_family", "") or ""),
            "error": str(error_text),
        })

    unique_rows = _dedup_rows(raw_rows)
    unique_proposals = _dedup_proposal_rows(raw_proposals) if bool(proposal_objects_enable) else []
    preview_limit = max(1, int(preview_topk))
    if bool(proposal_objects_enable):
        final_rows = [_proposal_output_row(item) for item in unique_proposals[:preview_limit]]
    else:
        final_rows = unique_rows[:preview_limit]
    selected_route = str(final_rows[0].get("solver_market_route", "") or "") if final_rows else ""
    primary_meta = dict(solver_meta_by_route.get(selected_route, {}) or {})
    if not primary_meta and route_summaries:
        primary_meta = dict(solver_meta_by_route.get(str(route_summaries[0]["route_name"]), {}) or {})

    dedup_child_states = _dedup_child_spec_states(child_spec_states)
    merged_meta = dict(primary_meta)
    merged_meta["solver_market_proposal_objects_enable"] = bool(proposal_objects_enable)
    merged_meta["solver_market_route_count"] = int(len(route_summaries))
    merged_meta["solver_market_routes"] = route_summaries
    merged_meta["solver_market_candidate_count_raw"] = int(len(raw_rows))
    merged_meta["solver_market_candidate_count_unique"] = int(
        len(unique_proposals) if bool(proposal_objects_enable) else len(unique_rows)
    )
    merged_meta["solver_market_preview_topk"] = int(preview_limit)
    merged_meta["solver_market_exact_topk"] = max(1, int(exact_topk))
    merged_meta["solver_market_selected_route"] = str(selected_route)
    merged_meta["solver_market_selected_method_name"] = (
        str(final_rows[0].get("solver_market_method_name", "") or "") if final_rows else ""
    )
    merged_meta["solver_market_selected_subroute"] = (
        str(final_rows[0].get("solver_market_subroute", "") or "") if final_rows else ""
    )
    merged_meta["solver_market_selected_fit_jet_source"] = (
        str(final_rows[0].get("witness_fit_jet_source", "") or "") if final_rows else ""
    )
    merged_meta["solver_market_selected_probe_jet_source"] = (
        str(final_rows[0].get("witness_probe_jet_source", "") or "") if final_rows else ""
    )
    merged_meta["solver_market_selected_numeric_jet_fallback_used"] = (
        bool(final_rows[0].get("witness_numeric_jet_fallback_used", False)) if final_rows else False
    )
    merged_meta["solver_market_selected_exact_jet_used"] = (
        bool(final_rows[0].get("witness_exact_jet_used", False)) if final_rows else False
    )
    merged_meta["preview_count"] = int(len(final_rows))
    if bool(proposal_objects_enable) and unique_proposals:
        selected_proposal = unique_proposals[0].proposal if final_rows else None
        merged_meta["solver_market_selected_proposal"] = serialize_solver_proposal(selected_proposal)
    merged_meta["child_spec_states"] = dedup_child_states
    merged_meta["child_spec_state_count"] = int(len(dedup_child_states))
    if final_rows:
        merged_meta["status"] = str(merged_meta.get("status", "ok") or "ok")
    elif route_summaries:
        merged_meta["status"] = "no_market_candidates"
    else:
        merged_meta["status"] = "no_market_routes"
    return {
        "rows": final_rows,
        "solver_meta": merged_meta,
    }


__all__ = [
    "SolverMarketRouteCall",
    "run_preview_solver_market",
]
