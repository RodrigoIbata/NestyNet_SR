# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Pure controller-guidance helpers used by the factorized symbolic search explorer."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from ..expr_ast import node_str


def _preview_child_eff_mse(candidate_meta: Any) -> float:
    if not isinstance(candidate_meta, Mapping):
        return float("inf")
    try:
        value = float(candidate_meta.get("estimated_child_eff_mse", float("inf")))
    except Exception:
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def _controller_build_slate_id(*parts: Any) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return f"buildslate_{digest.hexdigest()[:12]}"


def _repair_route_compare_decision(
    route_pred: Mapping[str, Any] | None,
    *,
    macro_enabled: bool = False,
    max_repair_prob: float = 0.35,
    min_build_margin: float = 0.05,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "best_route": "",
        "repair_prob": 0.0,
        "build_prob": 0.0,
        "margin_estimate": 0.0,
        "exact_margin": None,
        "veto_repair": False,
        "source": "unavailable",
    }
    if not isinstance(route_pred, Mapping) or not bool(route_pred.get("trained", False)):
        return out
    try:
        repair_prob = min(1.0, max(0.0, float(route_pred.get("repair_prob", 0.0))))
    except Exception:
        repair_prob = 0.0
    try:
        build_prob = min(1.0, max(0.0, float(route_pred.get("build_prob", 1.0 - repair_prob))))
    except Exception:
        build_prob = max(0.0, 1.0 - repair_prob)
    try:
        margin_estimate = float(route_pred.get("margin_estimate", 0.0))
    except Exception:
        margin_estimate = 0.0
    exact_margin = route_pred.get("exact_margin", None)
    try:
        exact_margin = None if exact_margin is None else float(exact_margin)
    except Exception:
        exact_margin = None
    best_route = str(route_pred.get("best_route", "") or "")
    veto = False
    source = "advisory_only"
    if (not bool(macro_enabled)) and best_route == "build":
        veto = bool(repair_prob <= float(max_repair_prob) and margin_estimate <= -float(min_build_margin))
        source = "build_veto" if veto else "build_preferred"
    elif best_route == "repair":
        source = "repair_preferred"
    out.update({
        "trained": True,
        "best_route": best_route,
        "repair_prob": float(repair_prob),
        "build_prob": float(build_prob),
        "margin_estimate": float(margin_estimate),
        "exact_margin": exact_margin,
        "veto_repair": bool(veto),
        "source": str(source),
    })
    return out


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _route_summary_best_eff_mse(summary: Mapping[str, Any] | None) -> float | None:
    if not isinstance(summary, Mapping):
        return None
    rows = [row for row in list(summary.get("rows", []) or []) if isinstance(row, Mapping)]
    best_eff = None
    for row in rows:
        eff = _finite_float_or_none(row.get("child_eff_mse", None))
        if eff is None:
            eff = _finite_float_or_none(row.get("current_best_child_eff_mse", None))
        if eff is None:
            continue
        if best_eff is None or float(eff) < float(best_eff):
            best_eff = float(eff)
    if best_eff is not None:
        return float(best_eff)
    return _finite_float_or_none(summary.get("best_exact_eff_mse", None))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-float(value))
        return float(1.0 / (1.0 + z))
    z = math.exp(float(value))
    return float(z / (1.0 + z))


def _credible_route_logit_scale(
    *,
    max_repair_prob: float,
    min_build_margin: float,
) -> float:
    try:
        repair_prob = float(max_repair_prob)
    except Exception:
        repair_prob = 0.35
    repair_prob = min(0.49, max(0.01, repair_prob))
    try:
        margin = abs(float(min_build_margin))
    except Exception:
        margin = 0.05
    margin = max(1.0e-6, margin)
    try:
        scale = abs(math.log((1.0 - repair_prob) / repair_prob)) / margin
    except Exception:
        scale = 10.0
    return float(max(1.0, scale))


def _summarize_repair_unseen_upside(
    repair_unseen_pred: Mapping[str, Any] | None,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "n_rows": 0,
        "n_unique_beams": 0,
        "residual_unseen_upside": 0.0,
        "best_opportunity_id": "",
        "best_beam_id": "",
        "best_acquisition_estimate": 0.0,
        "best_expected_gain_next_under_executor": 0.0,
        "best_fragility_prob": 0.0,
        "best_cost_estimate": 0.0,
    }
    if not isinstance(repair_unseen_pred, Mapping) or not bool(repair_unseen_pred.get("trained", False)):
        return out
    by_beam: dict[str, dict[str, Any]] = {}
    for raw_row in list(repair_unseen_pred.get("rows", []) or []):
        if not isinstance(raw_row, Mapping):
            continue
        if str(raw_row.get("route_source", "") or "") not in {"", "repair"}:
            continue
        if int(raw_row.get("budget_remaining", 0) or 0) <= 0:
            continue
        acquisition = _finite_float_or_none(raw_row.get("acquisition_estimate", None))
        if acquisition is None:
            acquisition = 0.0
        beam_id = str(raw_row.get("beam_id", "") or raw_row.get("opportunity_id", "") or raw_row.get("row_index", ""))
        current_best = by_beam.get(beam_id, None)
        if current_best is None or float(acquisition) > float(current_best.get("acquisition_estimate", float("-inf"))):
            by_beam[beam_id] = {
                "opportunity_id": str(raw_row.get("opportunity_id", "") or ""),
                "beam_id": str(beam_id),
                "acquisition_estimate": float(acquisition),
                "expected_gain_next_under_executor": float(_finite_float_or_none(raw_row.get("expected_gain_next_under_executor", None)) or 0.0),
                "fragility_prob": float(_finite_float_or_none(raw_row.get("fragility_prob", None)) or 0.0),
                "cost_estimate": float(_finite_float_or_none(raw_row.get("cost_estimate", None)) or 0.0),
            }
    unique_rows = sorted(
        by_beam.values(),
        key=lambda row: (
            float(row.get("acquisition_estimate", float("-inf"))),
            float(row.get("expected_gain_next_under_executor", float("-inf"))),
            str(row.get("beam_id", "")),
        ),
        reverse=True,
    )
    if not unique_rows:
        out["trained"] = True
        return out
    best_row = unique_rows[0]
    out.update({
        "trained": True,
        "n_rows": int(len(list(repair_unseen_pred.get("rows", []) or []))),
        "n_unique_beams": int(len(unique_rows)),
        "residual_unseen_upside": float(max(0.0, float(best_row.get("acquisition_estimate", 0.0) or 0.0))),
        "best_opportunity_id": str(best_row.get("opportunity_id", "") or ""),
        "best_beam_id": str(best_row.get("beam_id", "") or ""),
        "best_acquisition_estimate": float(best_row.get("acquisition_estimate", 0.0) or 0.0),
        "best_expected_gain_next_under_executor": float(best_row.get("expected_gain_next_under_executor", 0.0) or 0.0),
        "best_fragility_prob": float(best_row.get("fragility_prob", 0.0) or 0.0),
        "best_cost_estimate": float(best_row.get("cost_estimate", 0.0) or 0.0),
    })
    return out


def _credible_route_preview_repair_opportunity_rows(
    candidate_meta: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(candidate_meta, Mapping):
        return []
    return [
        dict(row)
        for row in list(candidate_meta.get("repair_opportunity_slate", []) or [])
        if isinstance(row, Mapping)
    ]


def _credible_route_compare_decision(
    route_pred: Mapping[str, Any] | None,
    repair_unseen_pred: Mapping[str, Any] | None = None,
    *,
    credible_route_enable: bool = False,
    macro_enabled: bool = False,
    max_repair_prob: float = 0.35,
    min_build_margin: float = 0.05,
) -> dict[str, Any]:
    legacy = _repair_route_compare_decision(
        route_pred,
        macro_enabled=macro_enabled,
        max_repair_prob=max_repair_prob,
        min_build_margin=min_build_margin,
    )
    out = dict(legacy)
    out.update({
        "credible_route_enable": bool(credible_route_enable),
        "credible_route_used": False,
        "legacy_best_route": str(legacy.get("best_route", "") or ""),
        "legacy_repair_prob": float(legacy.get("repair_prob", 0.0) or 0.0),
        "legacy_build_prob": float(legacy.get("build_prob", 0.0) or 0.0),
        "legacy_margin_estimate": float(legacy.get("margin_estimate", 0.0) or 0.0),
        "repair_observed_score": None,
        "build_observed_score": None,
        "repair_unseen_trained": False,
        "repair_unseen_upside": 0.0,
        "repair_unseen_best_opportunity_id": "",
        "repair_unseen_best_beam_id": "",
        "repair_unseen_best_acquisition_estimate": 0.0,
        "repair_unseen_best_expected_gain_next_under_executor": 0.0,
        "repair_unseen_best_fragility_prob": 0.0,
        "repair_unseen_best_cost_estimate": 0.0,
        "repair_credible_score": None,
        "build_credible_score": None,
    })
    if not bool(credible_route_enable):
        return out
    if not isinstance(route_pred, Mapping) or not bool(route_pred.get("trained", False)):
        return out

    repair_best_eff = _route_summary_best_eff_mse(route_pred.get("repair_summary", None))
    build_best_eff = _route_summary_best_eff_mse(route_pred.get("build_summary", None))
    unseen_summary = _summarize_repair_unseen_upside(repair_unseen_pred)
    out.update({
        "repair_unseen_trained": bool(unseen_summary.get("trained", False)),
        "repair_unseen_upside": float(unseen_summary.get("residual_unseen_upside", 0.0) or 0.0),
        "repair_unseen_best_opportunity_id": str(unseen_summary.get("best_opportunity_id", "") or ""),
        "repair_unseen_best_beam_id": str(unseen_summary.get("best_beam_id", "") or ""),
        "repair_unseen_best_acquisition_estimate": float(unseen_summary.get("best_acquisition_estimate", 0.0) or 0.0),
        "repair_unseen_best_expected_gain_next_under_executor": float(unseen_summary.get("best_expected_gain_next_under_executor", 0.0) or 0.0),
        "repair_unseen_best_fragility_prob": float(unseen_summary.get("best_fragility_prob", 0.0) or 0.0),
        "repair_unseen_best_cost_estimate": float(unseen_summary.get("best_cost_estimate", 0.0) or 0.0),
    })
    if repair_best_eff is None or build_best_eff is None or not bool(unseen_summary.get("trained", False)):
        return out

    repair_observed_score = -float(repair_best_eff)
    build_observed_score = -float(build_best_eff)
    repair_credible_score = float(repair_observed_score + float(unseen_summary.get("residual_unseen_upside", 0.0) or 0.0))
    build_credible_score = float(build_observed_score)
    credible_margin = float(repair_credible_score - build_credible_score)
    logit_scale = _credible_route_logit_scale(
        max_repair_prob=max_repair_prob,
        min_build_margin=min_build_margin,
    )
    repair_prob = _sigmoid(float(logit_scale * credible_margin))
    build_prob = float(1.0 - repair_prob)
    best_route = "repair" if credible_margin >= 0.0 else "build"
    veto = False
    source = "credible_route_repair_unseen"
    if (not bool(macro_enabled)) and best_route == "build":
        veto = bool(repair_prob <= float(max_repair_prob) and credible_margin <= -float(min_build_margin))
        source = "credible_build_veto" if veto else "credible_build_preferred"
    elif best_route == "repair":
        source = "credible_repair_preferred"

    out.update({
        "trained": True,
        "best_route": str(best_route),
        "repair_prob": float(repair_prob),
        "build_prob": float(build_prob),
        "margin_estimate": float(credible_margin),
        "veto_repair": bool(veto),
        "source": str(source),
        "credible_route_used": True,
        "repair_observed_score": float(repair_observed_score),
        "build_observed_score": float(build_observed_score),
        "repair_credible_score": float(repair_credible_score),
        "build_credible_score": float(build_credible_score),
    })
    return out


def _derived_controller_build_rng(
    base_seed: int,
    parent_key: Any,
    parent_expr: Any,
    action_id: int,
    ordinal: int,
) -> random.Random:
    digest = hashlib.sha1()
    digest.update(str(int(base_seed)).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(parent_key).encode("utf-8", errors="ignore"))
    digest.update(b"|")
    digest.update(node_str(parent_expr).encode("utf-8", errors="ignore"))
    digest.update(b"|")
    digest.update(str(int(action_id)).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(int(ordinal)).encode("utf-8"))
    seed_int = int.from_bytes(digest.digest()[:8], "big", signed=False)
    return random.Random(seed_int)


def _choose_repair_execution_preview(
    *,
    analytic_preview_expr: Any,
    analytic_preview_meta: Any,
    analytic_preview_rng: random.Random | None,
    analytic_anchor_path: Sequence[int] | None,
    analytic_preview_paths: Sequence[Sequence[int]] | None,
    analytic_preview_path_target_modes: Mapping[tuple[int, ...], str] | None,
    learned_preview_expr: Any,
    learned_preview_meta: Any,
    learned_preview_rng: random.Random | None,
    learned_anchor_path: Sequence[int] | None,
    learned_preview_paths: Sequence[Sequence[int]] | None,
    learned_preview_path_target_modes: Mapping[tuple[int, ...], str] | None,
    learned_preview_source: str,
    min_rel_gain: float = 0.05,
) -> dict[str, Any]:
    analytic_child_eff = _preview_child_eff_mse(analytic_preview_meta)
    learned_child_eff = _preview_child_eff_mse(learned_preview_meta)
    rel_gain = 0.0
    if math.isfinite(analytic_child_eff) and analytic_child_eff > 1.0e-30 and math.isfinite(learned_child_eff):
        rel_gain = max(0.0, (analytic_child_eff - learned_child_eff) / analytic_child_eff)

    use_learned = False
    if learned_preview_expr is not None and learned_preview_source not in ("", "analytic", "analytic_fallback"):
        if analytic_preview_expr is None:
            use_learned = True
        elif math.isfinite(learned_child_eff):
            if (not math.isfinite(analytic_child_eff)) or rel_gain >= float(max(0.0, min_rel_gain)):
                use_learned = True

    if use_learned:
        return {
            "expr": learned_preview_expr,
            "meta": learned_preview_meta,
            "rng": learned_preview_rng,
            "anchor_path": tuple(int(v) for v in (learned_anchor_path or ())),
            "preview_paths": [
                tuple(int(v) for v in (path or ()))
                for path in list(learned_preview_paths or [])
                if path
            ],
            "path_target_modes": {
                tuple(int(v) for v in path): str(mode)
                for path, mode in dict(learned_preview_path_target_modes or {}).items()
                if path and str(mode)
            } or None,
            "source": str(learned_preview_source),
            "child_eff_mse": float(learned_child_eff),
            "analytic_child_eff_mse": float(analytic_child_eff),
            "relative_gain_vs_analytic": float(rel_gain),
        }
    return {
        "expr": analytic_preview_expr,
        "meta": analytic_preview_meta,
        "rng": analytic_preview_rng,
        "anchor_path": tuple(int(v) for v in (analytic_anchor_path or ())),
        "preview_paths": [
            tuple(int(v) for v in (path or ()))
            for path in list(analytic_preview_paths or [])
            if path
        ],
        "path_target_modes": {
            tuple(int(v) for v in path): str(mode)
            for path, mode in dict(analytic_preview_path_target_modes or {}).items()
            if path and str(mode)
        } or None,
        "source": "analytic",
        "child_eff_mse": float(analytic_child_eff),
        "analytic_child_eff_mse": float(analytic_child_eff),
        "relative_gain_vs_analytic": 0.0,
    }


def _serialize_lineage_key(key):
    if isinstance(key, tuple):
        try:
            return [int(v) for v in key]
        except Exception:
            return [str(v) for v in key]
    if isinstance(key, (int, float, str)):
        return key
    return str(key)


def _logged_action_path_from_row(row):
    if not isinstance(row, dict):
        return []
    for key in ("controller_action_path", "selected_path"):
        path_like = row.get(key, None)
        try:
            path = [int(v) for v in (path_like or [])]
        except Exception:
            path = []
        if path:
            return path
    return []


def _annotate_inverse_experiment_lineage(
    inverse_experiment_log,
    lineage_events,
    *,
    horizon,
    eps,
):
    if (not isinstance(inverse_experiment_log, list)) or (not isinstance(lineage_events, list)):
        return
    try:
        horizon_i = max(0, int(horizon))
    except Exception:
        horizon_i = 0
    if horizon_i <= 0 or not lineage_events:
        return

    children_by_parent = {}
    for ev in lineage_events:
        if not isinstance(ev, dict):
            continue
        parent_key = ev.get("parent_key_raw", None)
        if parent_key is None:
            continue
        children_by_parent.setdefault(parent_key, []).append(ev)

    for ev in lineage_events:
        row_idx = ev.get("row_index", None)
        if not isinstance(row_idx, int) or not (0 <= row_idx < len(inverse_experiment_log)):
            continue
        row = inverse_experiment_log[row_idx]
        if not isinstance(row, dict):
            continue
        parent_eff_mse = ev.get("parent_eff_mse", None)
        child_eff_mse = ev.get("child_eff_mse", None)
        try:
            parent_eff_f = float(parent_eff_mse)
            child_eff_f = float(child_eff_mse)
        except Exception:
            continue
        if (not math.isfinite(parent_eff_f)) or (not math.isfinite(child_eff_f)):
            continue

        best = {
            "eff_mse": float(child_eff_f),
            "raw_mse": float(ev.get("child_raw_mse", child_eff_f)),
            "hops": 1,
            "row": row,
        }
        queue = [(ev.get("child_key_raw", None), 1)]
        seen_hops = {}
        child_key_raw = ev.get("child_key_raw", None)
        if child_key_raw is not None:
            seen_hops[child_key_raw] = 1
        while queue:
            residual_basin_key, hops = queue.pop(0)
            if residual_basin_key is None or hops >= horizon_i:
                continue
            for nxt in children_by_parent.get(residual_basin_key, ()):
                if not isinstance(nxt, dict):
                    continue
                nxt_key = nxt.get("child_key_raw", None)
                nxt_hops = hops + 1
                try:
                    nxt_eff = float(nxt.get("child_eff_mse", float("inf")))
                except Exception:
                    nxt_eff = float("inf")
                if math.isfinite(nxt_eff) and (
                    float(nxt_eff) < float(best["eff_mse"])
                    or (
                        math.isclose(float(nxt_eff), float(best["eff_mse"]))
                        and int(nxt_hops) < int(best["hops"])
                    )
                ):
                    nxt_row_idx = nxt.get("row_index", None)
                    nxt_row = inverse_experiment_log[nxt_row_idx] if isinstance(nxt_row_idx, int) and 0 <= nxt_row_idx < len(inverse_experiment_log) else row
                    best = {
                        "eff_mse": float(nxt_eff),
                        "raw_mse": float(nxt.get("child_raw_mse", nxt_eff)),
                        "hops": int(nxt_hops),
                        "row": nxt_row if isinstance(nxt_row, dict) else row,
                    }
                if nxt_key is None or nxt_hops >= horizon_i:
                    continue
                prev_hops = seen_hops.get(nxt_key, None)
                if prev_hops is not None and int(prev_hops) <= int(nxt_hops):
                    continue
                seen_hops[nxt_key] = int(nxt_hops)
                queue.append((nxt_key, int(nxt_hops)))

        best_row = best["row"] if isinstance(best.get("row", None), dict) else row
        best_path = _logged_action_path_from_row(best_row)
        try:
            descendant_log_gain = math.log(parent_eff_f + float(eps)) - math.log(float(best["eff_mse"]) + float(eps))
        except Exception:
            descendant_log_gain = None
        try:
            current_time_penalty = float(row.get("actor_critic_reward_time_penalty", 0.0) or 0.0)
        except Exception:
            current_time_penalty = 0.0
        try:
            current_novelty = float(row.get("actor_critic_reward_novelty_bonus", 0.0) or 0.0)
        except Exception:
            current_novelty = 0.0
        try:
            current_best_bonus = float(row.get("actor_critic_reward_best_bonus", 0.0) or 0.0)
        except Exception:
            current_best_bonus = 0.0
        descendant_reward = None
        if descendant_log_gain is not None and math.isfinite(float(descendant_log_gain)):
            descendant_reward = float(descendant_log_gain + current_novelty + current_best_bonus - current_time_penalty)
        row.update({
            "lineage_parent_key": _serialize_lineage_key(ev.get("parent_key_raw", None)),
            "lineage_child_key": _serialize_lineage_key(ev.get("child_key_raw", None)),
            "best_descendant_horizon": int(horizon_i),
            "best_descendant_hops": int(best["hops"]),
            "best_descendant_eff_mse": float(best["eff_mse"]),
            "best_descendant_raw_mse": float(best["raw_mse"]),
            "best_descendant_path": best_path,
            "best_descendant_selected_path": (
                [int(v) for v in (best_row.get("selected_path", []) or [])]
                if isinstance(best_row, dict) else []
            ),
            "best_descendant_target_mode": (
                str(best_row.get("selected_target_mode", "") or "")
                if isinstance(best_row, dict) else ""
            ),
            "best_descendant_macro_action": (
                str(best_row.get("macro_action", "") or "")
                if isinstance(best_row, dict) else ""
            ),
            "actor_critic_descendant_log_gain": None if descendant_log_gain is None else float(descendant_log_gain),
            "actor_critic_descendant_reward": None if descendant_reward is None else float(descendant_reward),
        })


__all__ = [
    "_annotate_inverse_experiment_lineage",
    "_choose_repair_execution_preview",
    "_credible_route_compare_decision",
    "_credible_route_preview_repair_opportunity_rows",
    "_controller_build_slate_id",
    "_derived_controller_build_rng",
    "_logged_action_path_from_row",
    "_preview_child_eff_mse",
    "_repair_route_compare_decision",
    "_serialize_lineage_key",
]
