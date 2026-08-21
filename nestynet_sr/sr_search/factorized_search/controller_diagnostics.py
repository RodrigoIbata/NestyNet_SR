# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from nestynet_sr.sr_search.factorized_search.oracle_lab import generate_oracle_policy_pretrain_dataset
from nestynet_sr.sr_search.factorized_search.repair_critic import (
    load_inverse_experiment_rows,
    load_repair_critic_bundle,
    predict_repair_controller_heads,
)


PathPredictor = Callable[[dict[str, Any], Any], dict[str, Any]]


def _path_tuple(path_like: Any) -> tuple[int, ...] | None:
    try:
        path = tuple(int(v) for v in (path_like or ()))
    except Exception:
        return None
    return path if path else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _path_from_row(row: Mapping[str, Any] | None, key: str) -> tuple[int, ...] | None:
    row = row if isinstance(row, Mapping) else {}
    return _path_tuple(row.get(key, None))


def _policy_attn_top1(guidance: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    guidance = guidance if isinstance(guidance, Mapping) else {}
    rows = list(((guidance.get("path", {}) or {}).get("rows", []) or []))
    best_key = None
    best_path = None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        path = _path_tuple(row.get("path", None))
        if path is None:
            continue
        key = (_safe_float(row.get("policy_weight", float("-inf")), float("-inf")), tuple(path))
        if best_key is None or key > best_key:
            best_key = key
            best_path = path
    return best_path


def _path_head_top1(guidance: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    guidance = guidance if isinstance(guidance, Mapping) else {}
    return _path_tuple(((guidance.get("path", {}) or {}).get("best_path", None)))


def _analytic_candidate_paths(row: Mapping[str, Any] | None) -> list[tuple[int, ...]]:
    row = row if isinstance(row, Mapping) else {}
    out: list[tuple[int, ...]] = []
    for item in list(row.get("path_summaries", []) or []):
        if not isinstance(item, Mapping):
            continue
        path = _path_tuple(item.get("path", None))
        if path is not None:
            out.append(path)
    return out


def _pairwise_agreement(authority_rows: Sequence[dict[str, tuple[int, ...] | None]]) -> dict[str, Any]:
    names = sorted({name for row in authority_rows for name in row.keys()})
    available = {
        name: int(sum(1 for row in authority_rows if row.get(name, None) is not None))
        for name in names
    }
    pairs: dict[str, Any] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            n = 0
            agree = 0
            for row in authority_rows:
                lp = row.get(left, None)
                rp = row.get(right, None)
                if lp is None or rp is None:
                    continue
                n += 1
                if tuple(lp) == tuple(rp):
                    agree += 1
            pairs[f"{left}__{right}"] = {
                "n": int(n),
                "agree": int(agree),
                "rate": None if n <= 0 else float(agree / n),
            }
    return {
        "available": available,
        "pairs": pairs,
    }


def _accuracy_vs_reference(
    authority_rows: Sequence[dict[str, tuple[int, ...] | None]],
    *,
    reference: str,
) -> dict[str, Any]:
    names = sorted({name for row in authority_rows for name in row.keys() if name != reference})
    out: dict[str, Any] = {}
    for name in names:
        n = 0
        correct = 0
        for row in authority_rows:
            ref = row.get(reference, None)
            pred = row.get(name, None)
            if ref is None or pred is None:
                continue
            n += 1
            if tuple(ref) == tuple(pred):
                correct += 1
        out[name] = {
            "n": int(n),
            "correct": int(correct),
            "rate": None if n <= 0 else float(correct / n),
        }
    return out


def _coerce_oracle_rows(payload: dict[str, Any] | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        return [dict(row) for row in list(payload.get("rows", []) or []) if isinstance(row, Mapping)]
    return [dict(row) for row in list(payload or []) if isinstance(row, Mapping)]


def analyze_oracle_path_diagnostics(
    payload: dict[str, Any] | Sequence[dict[str, Any]],
    *,
    bundle: dict[str, Any] | None = None,
    predictor: PathPredictor = predict_repair_controller_heads,
    recall_ks: Sequence[int] = (4, 8),
    max_rows: int | None = None,
) -> dict[str, Any]:
    rows = _coerce_oracle_rows(payload)
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]
    recall_ks = tuple(sorted({max(1, int(k)) for k in recall_ks}))
    by_depth_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, **{f"recall_at_{k}": 0 for k in recall_ks}})
    authority_rows: list[dict[str, tuple[int, ...] | None]] = []
    analyzed_rows = 0
    for row in rows:
        controller_row = row.get("controller_row", None)
        if not isinstance(controller_row, Mapping):
            continue
        target_path = _path_tuple(row.get("target_path", None))
        if target_path is None:
            continue
        truth_depth = _safe_int(row.get("truth_depth", controller_row.get("parent_depth", 0)), 0)
        candidate_paths = _analytic_candidate_paths(controller_row)
        counts = by_depth_counts[int(truth_depth)]
        counts["n"] += 1
        for k in recall_ks:
            if target_path in set(candidate_paths[:k]):
                counts[f"recall_at_{k}"] += 1
        authorities: dict[str, tuple[int, ...] | None] = {
            "oracle_target_path": target_path,
            "analytic_best_path": candidate_paths[0] if candidate_paths else None,
            "selected_path": _path_tuple(row.get("selected_path", None)) or _path_from_row(controller_row, "selected_path"),
        }
        if bundle is not None:
            guidance = predictor(bundle, controller_row)
            authorities["path_head_top1"] = _path_head_top1(guidance)
            authorities["policy_attn_top1"] = _policy_attn_top1(guidance)
        authority_rows.append(authorities)
        analyzed_rows += 1
    by_depth = {
        str(depth): {
            "n": int(stats["n"]),
            **{
                f"proposal_recall_at_{k}": None if stats["n"] <= 0 else float(stats[f"recall_at_{k}"] / stats["n"])
                for k in recall_ks
            },
        }
        for depth, stats in sorted(by_depth_counts.items())
    }
    return {
        "mode": "oracle_path_diagnostics",
        "n_rows": int(analyzed_rows),
        "recall_ks": [int(k) for k in recall_ks],
        "by_depth": by_depth,
        "oracle_target_accuracy": _accuracy_vs_reference(authority_rows, reference="oracle_target_path"),
        "authority_agreement": _pairwise_agreement(authority_rows),
    }


def analyze_search_path_diagnostics(
    report_paths: Sequence[str | Path] | Sequence[dict[str, Any]],
    *,
    bundle: dict[str, Any] | None = None,
    predictor: PathPredictor = predict_repair_controller_heads,
    max_rows: int | None = None,
) -> dict[str, Any]:
    if report_paths and isinstance(report_paths[0], Mapping):  # type: ignore[index]
        rows = [dict(row) for row in report_paths if isinstance(row, Mapping)]  # type: ignore[arg-type]
    else:
        rows = load_inverse_experiment_rows(report_paths)  # type: ignore[arg-type]
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]
    authority_rows: list[dict[str, tuple[int, ...] | None]] = []
    by_action_authorities: dict[str, list[dict[str, tuple[int, ...] | None]]] = defaultdict(list)
    by_depth_authorities: dict[int, list[dict[str, tuple[int, ...] | None]]] = defaultdict(list)
    n_with_path_rows = 0
    for row in rows:
        candidate_paths = _analytic_candidate_paths(row)
        if not candidate_paths:
            continue
        n_with_path_rows += 1
        action_name = str(row.get("controller_policy_action", row.get("macro_action", "")) or "")
        parent_depth = _safe_int(row.get("parent_depth", 0), 0)
        authorities: dict[str, tuple[int, ...] | None] = {
            "analytic_best_path": candidate_paths[0] if candidate_paths else None,
            "selected_path": _path_tuple(row.get("selected_path", None)),
            "executed_path": _path_tuple(row.get("controller_action_path", None)),
            "best_descendant_path": _path_tuple(row.get("best_descendant_path", None)),
        }
        if bundle is not None:
            guidance = predictor(bundle, row)
            authorities["path_head_top1"] = _path_head_top1(guidance)
            authorities["policy_attn_top1"] = _policy_attn_top1(guidance)
        authority_rows.append(authorities)
        by_action_authorities[action_name].append(authorities)
        by_depth_authorities[parent_depth].append(authorities)
    return {
        "mode": "search_path_diagnostics",
        "n_rows": int(len(rows)),
        "n_rows_with_path_summaries": int(n_with_path_rows),
        "authority_agreement": _pairwise_agreement(authority_rows),
        "executed_path_accuracy": _accuracy_vs_reference(authority_rows, reference="executed_path"),
        "by_action": {
            str(action): {
                "n": int(len(items)),
                "executed_path_accuracy": _accuracy_vs_reference(items, reference="executed_path"),
                "authority_agreement": _pairwise_agreement(items),
            }
            for action, items in sorted(by_action_authorities.items())
            if action
        },
        "by_depth": {
            str(depth): {
                "n": int(len(items)),
                "executed_path_accuracy": _accuracy_vs_reference(items, reference="executed_path"),
            }
            for depth, items in sorted(by_depth_authorities.items())
        },
    }


def _safe_mean(values: Sequence[float]) -> float | None:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _route_rows(row: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    row = row if isinstance(row, Mapping) else {}
    explicit = _coerce_mapping_rows(row.get("route_scheduler_route_rows", []))
    if explicit:
        return explicit
    selected_route = str(row.get("route_scheduler_selected_route", "") or "")
    available_routes = [str(route) for route in list(row.get("route_scheduler_available_routes", []) or []) if str(route)]
    if not available_routes:
        return []
    out: list[dict[str, Any]] = []
    for route_name in available_routes:
        out.append({
            "route": str(route_name),
            "route_score": _safe_float(0.0, 0.0),
            "selected": bool(route_name == selected_route),
        })
    return out


def _route_diag_row(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    row = row if isinstance(row, Mapping) else {}
    selected_route = str(row.get("route_scheduler_selected_route", "") or "")
    if selected_route == "":
        return None
    route_rows = _route_rows(row)
    available_routes = [str(route.get("route", "") or "") for route in route_rows if str(route.get("route", "") or "")]
    return {
        "selected_route": selected_route,
        "selection_source": str(row.get("route_scheduler_selection_source", "") or ""),
        "best_available_route": str(row.get("route_scheduler_best_available_route", "") or ""),
        "selected_best_preview_route": bool(row.get("route_scheduler_selected_best_preview_route", False)),
        "available_route_count": int(len(available_routes)),
        "available_routes": available_routes,
        "chosen_route_score": row.get("route_scheduler_chosen_route_score", None),
        "best_available_route_score": row.get("route_scheduler_best_available_route_score", None),
        "preview_gap": row.get("route_scheduler_preview_gap", None),
        "realized_raw_reward": row.get("route_scheduler_realized_raw_reward", None),
        "realized_adjusted_reward": row.get("route_scheduler_realized_adjusted_reward", None),
        "adjusted_regret_proxy": row.get("route_scheduler_adjusted_regret_proxy", None),
        "wall_s": row.get("route_scheduler_wall_s", None),
        "reward_mode": str(row.get("route_scheduler_reward_mode", "") or ""),
        "parent_depth": _safe_int(row.get("parent_depth", 0), 0),
        "status": str(row.get("status", "") or ""),
        "route_rows": route_rows,
    }


def _route_diag_summary(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selection_source_counts: dict[str, int] = defaultdict(int)
    for item in items:
        source = str(item.get("selection_source", "") or "")
        if source:
            selection_source_counts[source] += 1
    return {
        "n": int(len(items)),
        "mean_available_route_count": _safe_mean([float(item.get("available_route_count", 0) or 0) for item in items]),
        "mean_chosen_route_score": _safe_mean([
            float(item["chosen_route_score"])
            for item in items
            if item.get("chosen_route_score", None) is not None and math.isfinite(float(item["chosen_route_score"]))
        ]),
        "mean_best_available_route_score": _safe_mean([
            float(item["best_available_route_score"])
            for item in items
            if item.get("best_available_route_score", None) is not None and math.isfinite(float(item["best_available_route_score"]))
        ]),
        "mean_preview_gap": _safe_mean([
            float(item["preview_gap"])
            for item in items
            if item.get("preview_gap", None) is not None and math.isfinite(float(item["preview_gap"]))
        ]),
        "mean_realized_raw_reward": _safe_mean([
            float(item["realized_raw_reward"])
            for item in items
            if item.get("realized_raw_reward", None) is not None and math.isfinite(float(item["realized_raw_reward"]))
        ]),
        "mean_realized_adjusted_reward": _safe_mean([
            float(item["realized_adjusted_reward"])
            for item in items
            if item.get("realized_adjusted_reward", None) is not None and math.isfinite(float(item["realized_adjusted_reward"]))
        ]),
        "mean_adjusted_regret_proxy": _safe_mean([
            float(item["adjusted_regret_proxy"])
            for item in items
            if item.get("adjusted_regret_proxy", None) is not None and math.isfinite(float(item["adjusted_regret_proxy"]))
        ]),
        "mean_wall_s": _safe_mean([
            float(item["wall_s"])
            for item in items
            if item.get("wall_s", None) is not None and math.isfinite(float(item["wall_s"]))
        ]),
        "selected_best_preview_rate": None if not items else float(
            sum(1 for item in items if bool(item.get("selected_best_preview_route", False))) / len(items)
        ),
        "selection_source_counts": dict(sorted(selection_source_counts.items())),
    }


def analyze_search_route_diagnostics(
    report_paths: Sequence[str | Path] | Sequence[dict[str, Any]],
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    if report_paths and isinstance(report_paths[0], Mapping):  # type: ignore[index]
        rows = [dict(row) for row in report_paths if isinstance(row, Mapping)]  # type: ignore[arg-type]
    else:
        rows = load_inverse_experiment_rows(report_paths)  # type: ignore[arg-type]
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]

    items: list[dict[str, Any]] = []
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item = _route_diag_row(row)
        if item is None:
            continue
        items.append(item)
        by_route[str(item["selected_route"])].append(item)
        by_depth[int(item["parent_depth"])].append(item)

    chosen_route_counts = {
        route: int(len(route_items))
        for route, route_items in sorted(by_route.items())
    }
    best_available_route_counts: dict[str, int] = defaultdict(int)
    for item in items:
        route_name = str(item.get("best_available_route", "") or "")
        if route_name:
            best_available_route_counts[route_name] += 1

    return {
        "mode": "search_route_diagnostics",
        "n_rows": int(len(rows)),
        "n_rows_with_route_diagnostics": int(len(items)),
        "chosen_route_counts": chosen_route_counts,
        "best_available_route_counts": dict(sorted(best_available_route_counts.items())),
        **_route_diag_summary(items),
        "by_route": {
            str(route): _route_diag_summary(route_items)
            for route, route_items in sorted(by_route.items())
        },
        "by_depth": {
            str(depth): _route_diag_summary(depth_items)
            for depth, depth_items in sorted(by_depth.items())
        },
    }


def _coerce_mapping_rows(rows_like: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in list(rows_like or []) if isinstance(row, Mapping)]


def _repair_opportunity_rows(row: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    row = row if isinstance(row, Mapping) else {}
    explicit = _coerce_mapping_rows(row.get("repair_opportunity_slate", []))
    if explicit:
        return explicit
    grouped: dict[tuple[tuple[int, ...] | None, str], dict[str, Any]] = {}
    for slate_row in _coerce_mapping_rows(row.get("inverse_repair_slate", [])):
        if not bool(slate_row.get("dedup_kept", True)):
            continue
        path = _path_tuple(slate_row.get("path", None))
        target_mode = str(slate_row.get("target_mode", "") or "")
        key = (path, target_mode)
        bucket = grouped.setdefault(key, {
            "path": [] if path is None else [int(v) for v in path],
            "target_mode": target_mode,
            "candidate_count_observed": 0,
            "candidate_count_unique": 0,
            "current_best_child_expr": "",
            "best_preview_probe_mse": None,
        })
        bucket["candidate_count_observed"] = int(bucket.get("candidate_count_observed", 0)) + 1
        bucket["candidate_count_unique"] = int(bucket.get("candidate_count_unique", 0)) + 1
        probe_mse = slate_row.get("local_probe_mse", None)
        best_probe = bucket.get("best_preview_probe_mse", None)
        if best_probe is None or (_safe_float(probe_mse, float("inf")) < _safe_float(best_probe, float("inf"))):
            bucket["best_preview_probe_mse"] = probe_mse
            bucket["current_best_child_expr"] = str(slate_row.get("child_expr", "") or slate_row.get("child_key", "") or "")
    return list(grouped.values())


def _build_opportunity_rows(row: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    row = row if isinstance(row, Mapping) else {}
    explicit = _coerce_mapping_rows(row.get("build_opportunity_slate", []))
    if explicit:
        return explicit
    return [
        {
            "action": str(slate_row.get("action", "") or ""),
            "path": list(slate_row.get("path", []) or []),
            "candidate_count_observed": 1 if bool(slate_row.get("child_expr", "") or slate_row.get("candidate_child_size", None) is not None) else 0,
            "candidate_count_unique": 1 if bool(slate_row.get("child_expr", "") or slate_row.get("candidate_child_size", None) is not None) else 0,
            "current_best_child_expr": str(slate_row.get("child_expr", "") or slate_row.get("child_key", "") or ""),
            "current_best_child_eff_mse": slate_row.get("child_eff_mse", None),
            "exact_child_score_observed": bool(slate_row.get("exact_child_score_observed", False)),
            "status": str(slate_row.get("status", "") or ""),
        }
        for slate_row in _coerce_mapping_rows(row.get("controller_build_slate", []))
    ]


def analyze_oracle_opportunity_funnel(
    payload: dict[str, Any] | Sequence[dict[str, Any]],
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    rows = _coerce_oracle_rows(payload)
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]
    by_depth_counts: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "repair_truth_path_in_paths": 0.0,
            "repair_truth_path_in_opportunities": 0.0,
            "repair_truth_candidate_in_slate": 0.0,
            "build_truth_candidate_in_slate": 0.0,
            "repair_opportunity_count_sum": 0.0,
            "build_opportunity_count_sum": 0.0,
            "repair_exact_observed_count_sum": 0.0,
            "build_exact_observed_count_sum": 0.0,
        }
    )
    analyzed_rows = 0
    for row in rows:
        truth_depth = _safe_int(row.get("truth_depth", row.get("parent_depth", 0)), 0)
        truth_path = _path_tuple(row.get("oracle_truth_path", row.get("target_path", None)))
        repair_opp_rows = _repair_opportunity_rows(row)
        build_opp_rows = _build_opportunity_rows(row)
        repair_slate_rows = _coerce_mapping_rows(row.get("inverse_repair_slate", []))
        build_slate_rows = _coerce_mapping_rows(row.get("controller_build_slate", []))
        counts = by_depth_counts[int(truth_depth)]
        counts["n"] += 1.0
        counts["repair_opportunity_count_sum"] += float(len(repair_opp_rows))
        counts["build_opportunity_count_sum"] += float(len(build_opp_rows))
        counts["repair_exact_observed_count_sum"] += float(sum(1 for item in repair_slate_rows if bool(item.get("exact_child_score_observed", False))))
        counts["build_exact_observed_count_sum"] += float(sum(1 for item in build_slate_rows if bool(item.get("exact_child_score_observed", False))))
        if truth_path is not None:
            analytic_paths = set(_analytic_candidate_paths(row))
            if tuple(truth_path) in analytic_paths:
                counts["repair_truth_path_in_paths"] += 1.0
            opp_paths = {
                path
                for path in (_path_tuple(item.get("path", None)) for item in repair_opp_rows)
                if path is not None
            }
            if tuple(truth_path) in opp_paths:
                counts["repair_truth_path_in_opportunities"] += 1.0
        if any(bool(item.get("oracle_is_truth_candidate", False)) for item in repair_slate_rows):
            counts["repair_truth_candidate_in_slate"] += 1.0
        if any(bool(item.get("oracle_is_truth_candidate", False)) for item in build_slate_rows):
            counts["build_truth_candidate_in_slate"] += 1.0
        analyzed_rows += 1
    by_depth = {
        str(depth): {
            "n": int(stats["n"]),
            "repair_truth_path_in_paths_rate": None if stats["n"] <= 0 else float(stats["repair_truth_path_in_paths"] / stats["n"]),
            "repair_truth_path_in_opportunities_rate": None if stats["n"] <= 0 else float(stats["repair_truth_path_in_opportunities"] / stats["n"]),
            "repair_truth_candidate_in_slate_rate": None if stats["n"] <= 0 else float(stats["repair_truth_candidate_in_slate"] / stats["n"]),
            "build_truth_candidate_in_slate_rate": None if stats["n"] <= 0 else float(stats["build_truth_candidate_in_slate"] / stats["n"]),
            "repair_mean_opportunity_count": None if stats["n"] <= 0 else float(stats["repair_opportunity_count_sum"] / stats["n"]),
            "build_mean_opportunity_count": None if stats["n"] <= 0 else float(stats["build_opportunity_count_sum"] / stats["n"]),
            "repair_mean_exact_observed_count": None if stats["n"] <= 0 else float(stats["repair_exact_observed_count_sum"] / stats["n"]),
            "build_mean_exact_observed_count": None if stats["n"] <= 0 else float(stats["build_exact_observed_count_sum"] / stats["n"]),
        }
        for depth, stats in sorted(by_depth_counts.items())
    }
    return {
        "mode": "oracle_opportunity_funnel",
        "n_rows": int(analyzed_rows),
        "by_depth": by_depth,
    }


def analyze_search_opportunity_funnel(
    report_paths: Sequence[str | Path] | Sequence[dict[str, Any]],
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    if report_paths and isinstance(report_paths[0], Mapping):  # type: ignore[index]
        rows = [dict(row) for row in report_paths if isinstance(row, Mapping)]  # type: ignore[arg-type]
    else:
        rows = load_inverse_experiment_rows(report_paths)  # type: ignore[arg-type]
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]
    by_depth_counts: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "repair_opportunity_count_sum": 0.0,
            "build_opportunity_count_sum": 0.0,
            "repair_exact_observed_count_sum": 0.0,
            "build_exact_observed_count_sum": 0.0,
        }
    )
    for row in rows:
        depth = _safe_int(row.get("parent_depth", 0), 0)
        repair_opp_rows = _repair_opportunity_rows(row)
        build_opp_rows = _build_opportunity_rows(row)
        repair_slate_rows = _coerce_mapping_rows(row.get("inverse_repair_slate", []))
        build_slate_rows = _coerce_mapping_rows(row.get("controller_build_slate", []))
        counts = by_depth_counts[int(depth)]
        counts["n"] += 1.0
        counts["repair_opportunity_count_sum"] += float(len(repair_opp_rows))
        counts["build_opportunity_count_sum"] += float(len(build_opp_rows))
        counts["repair_exact_observed_count_sum"] += float(sum(1 for item in repair_slate_rows if bool(item.get("exact_child_score_observed", False))))
        counts["build_exact_observed_count_sum"] += float(sum(1 for item in build_slate_rows if bool(item.get("exact_child_score_observed", False))))
    by_depth = {
        str(depth): {
            "n": int(stats["n"]),
            "repair_mean_opportunity_count": None if stats["n"] <= 0 else float(stats["repair_opportunity_count_sum"] / stats["n"]),
            "build_mean_opportunity_count": None if stats["n"] <= 0 else float(stats["build_opportunity_count_sum"] / stats["n"]),
            "repair_mean_exact_observed_count": None if stats["n"] <= 0 else float(stats["repair_exact_observed_count_sum"] / stats["n"]),
            "build_mean_exact_observed_count": None if stats["n"] <= 0 else float(stats["build_exact_observed_count_sum"] / stats["n"]),
        }
        for depth, stats in sorted(by_depth_counts.items())
    }
    return {
        "mode": "search_opportunity_funnel",
        "n_rows": int(len(rows)),
        "by_depth": by_depth,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_oracle_cli(args: argparse.Namespace) -> dict[str, Any]:
    bundle = None
    if str(args.bundle_path or "").strip():
        bundle = load_repair_critic_bundle(args.bundle_path)
    if str(args.dataset_path or "").strip():
        payload = _load_json(args.dataset_path)
    else:
        spec_paths = sorted(glob.glob(str(args.spec_glob or "")))
        if not spec_paths:
            raise ValueError("oracle diagnostics require either --dataset_path or --spec_glob.")
        payload = generate_oracle_policy_pretrain_dataset(
            spec_paths,
            seeds=[int(s) for s in (args.seeds or [0])],
            depth_min=int(args.depth_min),
            depth_max=int(args.depth_max),
            topk=int(args.topk),
            max_corrupt_paths_per_spec=None if args.max_corrupt_paths_per_spec is None else int(args.max_corrupt_paths_per_spec),
            sweep_max_paths=None if args.sweep_max_paths is None else int(args.sweep_max_paths),
            verbose=not bool(args.quiet),
        )
    analysis = str(getattr(args, "analysis", "path") or "path").strip().lower()
    if analysis == "opportunity":
        return analyze_oracle_opportunity_funnel(
            payload,
            max_rows=None if args.max_rows is None else int(args.max_rows),
        )
    return analyze_oracle_path_diagnostics(
        payload,
        bundle=bundle,
        recall_ks=tuple(int(v) for v in (args.recall_ks or [4, 8])),
        max_rows=None if args.max_rows is None else int(args.max_rows),
    )


def run_search_cli(args: argparse.Namespace) -> dict[str, Any]:
    bundle = None
    if str(args.bundle_path or "").strip():
        bundle = load_repair_critic_bundle(args.bundle_path)
    analysis = str(getattr(args, "analysis", "path") or "path").strip().lower()
    if analysis == "route":
        return analyze_search_route_diagnostics(
            [str(p) for p in (args.report_paths or [])],
            max_rows=None if args.max_rows is None else int(args.max_rows),
        )
    if analysis == "opportunity":
        return analyze_search_opportunity_funnel(
            [str(p) for p in (args.report_paths or [])],
            max_rows=None if args.max_rows is None else int(args.max_rows),
        )
    return analyze_search_path_diagnostics(
        [str(p) for p in (args.report_paths or [])],
        bundle=bundle,
        max_rows=None if args.max_rows is None else int(args.max_rows),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    po = sub.add_parser("oracle")
    po.add_argument("--dataset_path", type=str, default="")
    po.add_argument("--spec_glob", type=str, default="")
    po.add_argument("--bundle_path", type=str, default="")
    po.add_argument("--seeds", nargs="*", type=int, default=[0])
    po.add_argument("--depth_min", type=int, default=3)
    po.add_argument("--depth_max", type=int, default=8)
    po.add_argument("--topk", type=int, default=8)
    po.add_argument("--max_corrupt_paths_per_spec", type=int, default=4)
    po.add_argument("--sweep_max_paths", type=int, default=8)
    po.add_argument("--recall_ks", nargs="*", type=int, default=[4, 8])
    po.add_argument("--analysis", type=str, choices=["path", "opportunity"], default="path")
    po.add_argument("--max_rows", type=int, default=None)
    po.add_argument("--quiet", action="store_true")
    po.add_argument("--output", type=str, default="")

    ps = sub.add_parser("search")
    ps.add_argument("--report_paths", nargs="+", required=True)
    ps.add_argument("--bundle_path", type=str, default="")
    ps.add_argument("--analysis", type=str, choices=["path", "opportunity", "route"], default="path")
    ps.add_argument("--max_rows", type=int, default=None)
    ps.add_argument("--output", type=str, default="")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    report = run_oracle_cli(args) if str(args.cmd) == "oracle" else run_search_cli(args)
    if str(args.output or "").strip():
        out_path = Path(str(args.output))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
