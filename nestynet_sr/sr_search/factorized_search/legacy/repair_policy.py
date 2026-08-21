# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Repair-controller scoring and retry policy helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..expr_ast import get_at, node_str
from ..inverse_core import _normalize_inverse_target_mode
from ..policy.features import coerce_repair_feature_row

def _repair_controller_weights(stats: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(stats, dict):
        return {
            "potential": 1.00,
            "concentration": 0.35,
            "contrast": 0.20,
            "cost": 0.10,
            "stagnation": 0.15,
        }

    def _get(name: str, default: float) -> float:
        try:
            return float(stats.get(name, default))
        except Exception:
            return float(default)

    return {
        "potential": _get("potential_weight", 1.00),
        "concentration": _get("concentration_weight", 0.35),
        "contrast": _get("contrast_weight", 0.20),
        "cost": _get("cost_weight", 0.10),
        "stagnation": _get("stagnation_weight", 0.15),
    }


def _repair_controller_stagnation_state(
    parent_rec,
    stats: dict[str, Any] | None,
) -> dict[str, float]:
    try:
        visits = max(0, int(getattr(parent_rec, "visits", 0)))
    except Exception:
        visits = 0
    try:
        visits_since_improve = max(0, int(getattr(parent_rec, "visits_since_improve", visits)))
    except Exception:
        visits_since_improve = visits
    if visits_since_improve > visits:
        visits_since_improve = visits
    try:
        stagnation_scale = max(1, int((stats or {}).get("stagnation_visits", 1)))
    except Exception:
        stagnation_scale = 1
    stagnation_score = min(1.0, float(visits_since_improve) / float(stagnation_scale))
    stagnation_ratio = min(1.0, float(visits_since_improve) / max(1.0, float(visits)))
    return {
        "visits": float(visits),
        "visits_since_improve": float(visits_since_improve),
        "stagnation_scale": float(stagnation_scale),
        "stagnation_score": float(stagnation_score),
        "stagnation_ratio": float(stagnation_ratio),
    }


def _analytic_repair_controller_score(
    row: Any,
    stats: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    row = coerce_repair_feature_row(row)
    if not isinstance(row, dict):
        return 0.0, {
            "potential": 0.0,
            "concentration": 0.0,
            "contrast": 0.0,
            "cost": 1.0,
        }

    try:
        path_entropy = max(0.0, float(row.get("path_entropy", 0.0)))
    except Exception:
        path_entropy = 0.0
    try:
        top_mass = min(1.0, max(0.0, float(row.get("path_top_mass", 0.0))))
    except Exception:
        top_mass = 0.0
    try:
        n_pos = max(0.0, float(row.get("path_positive_count", 0.0)))
    except Exception:
        n_pos = 0.0
    if n_pos > 1.0:
        h_max = max(math.log(n_pos), 1.0e-12)
        entropy_norm = min(1.0, max(0.0, path_entropy / h_max))
    else:
        entropy_norm = 0.0
    concentration = 0.5 * top_mass + 0.5 * (1.0 - entropy_norm)

    potential = row.get("estimated_one_hole_rel_improve_eff", None)
    if potential is None:
        potential = row.get("proxy_one_hole_potential_eff", None)
    try:
        potential_f = min(1.0, max(0.0, float(potential)))
    except Exception:
        potential_f = 0.0

    contrast = row.get("identity_vs_full_log_mse_contrast", None)
    try:
        contrast_f = max(0.0, float(contrast))
    except Exception:
        contrast_f = 0.0
    contrast_score = math.tanh(contrast_f / 3.0)

    try:
        k = max(0.0, float(row.get("local_candidate_count", 0.0)))
    except Exception:
        k = 0.0
    cost = min(1.0, math.log1p(k) / math.log(17.0))

    try:
        stagnation = min(1.0, max(0.0, float(row.get("parent_stagnation_score", 0.0))))
    except Exception:
        stagnation = 0.0

    weights = _repair_controller_weights(stats)
    score = (
        weights["potential"] * potential_f
        + weights["concentration"] * concentration
        + weights["contrast"] * contrast_score
        + weights["stagnation"] * stagnation
        - weights["cost"] * cost
    )
    return float(score), {
        "potential": float(potential_f),
        "concentration": float(concentration),
        "contrast": float(contrast_score),
        "cost": float(cost),
        "stagnation": float(stagnation),
    }


def _normalize_repair_controller_critic_mode(mode: Any, default: str = "priority") -> str:
    token = str(mode or default).strip().lower().replace("-", "_")
    if token in ("", "default", "priority_bonus", "priority_only", "sidecar"):
        return "priority"
    if token in ("gate_bonus", "gate_only"):
        return "gate"
    if token in ("threshold", "threshold_shift", "gate_threshold", "decisive_gate"):
        return "decisive"
    if token in {"priority", "gate", "decisive"}:
        return token
    return str(default or "priority").strip().lower().replace("-", "_") or "priority"


def _hybrid_repair_controller_scores(
    analytic_score: float,
    critic_preds: dict[str, Any] | None,
    critic_blend: float,
    critic_mode: str = "priority",
) -> dict[str, float | str]:
    gate_score = float(analytic_score)
    mode = _normalize_repair_controller_critic_mode(critic_mode, default="priority")
    try:
        blend = min(1.0, max(0.0, float(critic_blend)))
    except Exception:
        blend = 0.0
    out: dict[str, float | str] = {
        "gate_score": float(gate_score),
        "priority_score": float(gate_score),
        "critic_bonus": 0.0,
        "critic_signal": 0.0,
        "critic_signed_signal": 0.0,
        "critic_gate_delta": 0.0,
        "critic_priority_delta": 0.0,
        "critic_utility_score": 0.0,
        "critic_accept_prob": 0.0,
        "critic_positive_reward_prob": 0.0,
        "critic_reward_per_s_score": 0.0,
        "threshold_shift": 0.0,
        "mode": str(mode),
        "source": "analytic",
    }
    if blend <= 0.0 or not isinstance(critic_preds, dict):
        return out

    try:
        utility_score = min(1.0, max(0.0, float(critic_preds.get("utility_score", 0.0))))
    except Exception:
        utility_score = 0.0
    try:
        accept_prob = min(1.0, max(0.0, float(critic_preds.get("accept_prob", 0.0))))
    except Exception:
        accept_prob = 0.0
    try:
        positive_reward_prob = min(1.0, max(0.0, float(critic_preds.get("positive_reward_prob", 0.0))))
    except Exception:
        positive_reward_prob = 0.0
    try:
        reward_per_s_score = min(1.0, max(0.0, float(critic_preds.get("reward_per_s_score", 0.0))))
    except Exception:
        reward_per_s_score = 0.0

    # Conservative Stage-1 hybrid: the critic can only add bounded priority
    # mass on top of the analytic gate, not veto repair outright.
    critic_signal_raw = (
        0.60 * utility_score
        + 0.25 * positive_reward_prob
        + 0.10 * accept_prob
        + 0.05 * reward_per_s_score
    )
    critic_signal = min(1.0, max(0.0, (critic_signal_raw - 0.35) / 0.65))
    critic_signed_signal = min(1.0, max(-1.0, 2.0 * critic_signal - 1.0))
    priority_delta = 0.0
    gate_delta = 0.0
    threshold_shift = 0.0
    if mode == "priority":
        # Legacy Stage-1 sidecar: the critic can only add bounded priority
        # mass on top of the analytic gate, not veto repair outright.
        priority_delta = 0.20 * blend * critic_signal
        source = "analytic_refine_critic_bonus" if abs(priority_delta) > 1.0e-12 else "analytic"
    elif mode == "gate":
        # Let the critic change the decisive repair gate directly.
        gate_delta = 0.20 * blend * critic_signed_signal
        priority_delta = gate_delta
        source = "analytic_refine_critic_gate" if abs(gate_delta) > 1.0e-12 else "analytic"
    else:
        # Stronger controller mode: critic changes both gate score and the
        # effective threshold cached for repair-frontier parent selection.
        gate_delta = 0.20 * blend * critic_signed_signal
        priority_delta = gate_delta
        threshold_shift = -0.10 * blend * critic_signed_signal
        source = "analytic_refine_critic_decisive" if (abs(gate_delta) > 1.0e-12 or abs(threshold_shift) > 1.0e-12) else "analytic"
    out.update({
        "gate_score": float(gate_score + gate_delta),
        "priority_score": float(gate_score + gate_delta + (priority_delta - gate_delta)),
        "critic_bonus": float(priority_delta),
        "critic_signal": float(critic_signal),
        "critic_signed_signal": float(critic_signed_signal),
        "critic_gate_delta": float(gate_delta),
        "critic_priority_delta": float(priority_delta),
        "critic_utility_score": float(utility_score),
        "critic_accept_prob": float(accept_prob),
        "critic_positive_reward_prob": float(positive_reward_prob),
        "critic_reward_per_s_score": float(reward_per_s_score),
        "threshold_shift": float(threshold_shift),
        "mode": str(mode),
        "source": str(source),
    })
    return out


def _repair_controller_relation_score(row: Mapping[str, Any] | None) -> float:
    if not isinstance(row, Mapping):
        return 0.0
    relation_weights = {
        "same": 1.00,
        "ancestor": 0.70,
        "descendant": 0.45,
        "unknown": 0.10,
        "disjoint": -0.25,
    }
    rel_probs = row.get("relation_probs", None)
    if isinstance(rel_probs, Mapping):
        total = 0.0
        for name, weight in relation_weights.items():
            try:
                prob = min(1.0, max(0.0, float(rel_probs.get(name, 0.0))))
            except Exception:
                prob = 0.0
            total += weight * prob
        return float(total)
    rel = str(row.get("best_relation", "unknown") or "unknown").strip().lower()
    return float(relation_weights.get(rel, 0.0))


def _repair_controller_path_policy(
    controller_heads: Mapping[str, Any] | None,
    *,
    fallback_path: Sequence[int] | None = None,
    fallback_paths: Sequence[Sequence[int]] | None = None,
    max_paths: int = 3,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "selected_path": None,
        "best_target_mode": "",
        "best_relation": "",
        "best_improvement_estimate": 0.0,
        "candidate_paths": [],
        "path_target_modes": {},
        "rows": [],
        "source": "analytic",
    }

    def _append_path(dst: list[tuple[int, ...]], seen: set[tuple[int, ...]], path_like: Any) -> None:
        try:
            path = tuple(int(v) for v in (path_like or ()))
        except Exception:
            return
        if not path or path in seen:
            return
        seen.add(path)
        dst.append(path)

    try:
        max_paths_i = max(1, int(max_paths))
    except Exception:
        max_paths_i = 3
    fallback_best = tuple(int(v) for v in (fallback_path or ()))
    fallback_rows = list((controller_heads or {}).get("path", {}).get("rows", []) or [])
    path_action_rows = list((controller_heads or {}).get("path_action", {}).get("rows", []) or [])
    path_info_map: dict[tuple[int, ...], Mapping[str, Any]] = {}
    for row in fallback_rows:
        if not isinstance(row, Mapping):
            continue
        try:
            path = tuple(int(v) for v in (row.get("path", None) or ()))
        except Exception:
            path = ()
        if path:
            path_info_map[path] = row
    if not bool((controller_heads or {}).get("path", {}).get("trained", False)) or not fallback_rows:
        seen: set[tuple[int, ...]] = set()
        candidate_paths: list[tuple[int, ...]] = []
        _append_path(candidate_paths, seen, fallback_best)
        for path_like in list(fallback_paths or []):
            _append_path(candidate_paths, seen, path_like)
        out["selected_path"] = list(candidate_paths[0]) if candidate_paths else None
        out["candidate_paths"] = [list(path) for path in candidate_paths[:max_paths_i]]
        return out

    ranked_rows = []
    if bool((controller_heads or {}).get("path_action", {}).get("trained", False)) and path_action_rows:
        for row in path_action_rows:
            if not isinstance(row, Mapping):
                continue
            try:
                path = tuple(int(v) for v in (row.get("path", None) or ()))
            except Exception:
                path = ()
            if not path:
                continue
            info_row = path_info_map.get(path, {})
            try:
                path_prob = min(1.0, max(0.0, float(row.get("path_prob", 0.0))))
            except Exception:
                path_prob = 0.0
            q_map = dict(row.get("q_estimates", {}) or {})
            repair_qs = []
            for action_name in ("inv_steer", "repair_option"):
                try:
                    q_value = float(q_map.get(action_name, float("-inf")))
                except Exception:
                    q_value = float("-inf")
                if math.isfinite(q_value):
                    repair_qs.append((q_value, action_name))
            if not repair_qs:
                continue
            repair_qs.sort(reverse=True)
            utility_estimate, best_action = repair_qs[0]
            try:
                improve = min(1.0, max(0.0, float(info_row.get("improvement_estimate", 0.0))))
            except Exception:
                improve = 0.0
            relation_score = _repair_controller_relation_score(info_row)
            try:
                weighted_gain = max(0.0, float(info_row.get("weighted_rel_gain", 0.0)))
            except Exception:
                weighted_gain = 0.0
            mode_name = _normalize_inverse_target_mode(
                str(row.get("target_mode", info_row.get("best_mode", info_row.get("target_mode", ""))) or ""),
                default="",
            )
            if mode_name not in ("identity", "affine", "full"):
                mode_name = ""
            policy_score = (
                float(utility_estimate)
                + 0.10 * path_prob
                + 0.10 * improve
                + 0.05 * relation_score
            )
            ranked_rows.append({
                "path": path,
                "mode": mode_name,
                "best_relation": str(info_row.get("best_relation", "") or ""),
                "improvement_estimate": float(improve),
                "policy_score": float(policy_score),
                "prob": float(path_prob),
                "weighted_rel_gain": float(weighted_gain),
                "utility_estimate": float(utility_estimate),
                "best_action": str(best_action),
            })
    if not ranked_rows:
        ranked_rows = []
        for row in fallback_rows:
            if not isinstance(row, Mapping):
                continue
            try:
                path = tuple(int(v) for v in (row.get("path", None) or ()))
            except Exception:
                path = ()
            if not path:
                continue
            try:
                path_prob = min(1.0, max(0.0, float(row.get("prob", 0.0))))
            except Exception:
                path_prob = 0.0
            try:
                improve = min(1.0, max(0.0, float(row.get("improvement_estimate", 0.0))))
            except Exception:
                improve = 0.0
            relation_score = _repair_controller_relation_score(row)
            try:
                weighted_gain = max(0.0, float(row.get("weighted_rel_gain", 0.0)))
            except Exception:
                weighted_gain = 0.0
            mode_name = _normalize_inverse_target_mode(
                str(row.get("best_mode", row.get("target_mode", "")) or ""),
                default="",
            )
            if mode_name not in ("identity", "affine", "full"):
                mode_name = ""
            policy_score = (
                1.00 * path_prob
                + 0.35 * improve
                + 0.20 * relation_score
                + 0.05 * math.log1p(weighted_gain)
            )
            ranked_rows.append({
                "path": path,
                "mode": mode_name,
                "best_relation": str(row.get("best_relation", "") or ""),
                "improvement_estimate": float(improve),
                "policy_score": float(policy_score),
                "prob": float(path_prob),
                "weighted_rel_gain": float(weighted_gain),
            })
    if not ranked_rows:
        return out

    ranked_rows.sort(
        key=lambda row: (
            float(row.get("policy_score", 0.0)),
            float(row.get("prob", 0.0)),
            float(row.get("improvement_estimate", 0.0)),
            float(row.get("weighted_rel_gain", 0.0)),
        ),
        reverse=True,
    )
    seen = set()
    candidate_paths = []
    path_target_modes: dict[tuple[int, ...], str] = {}
    for row in ranked_rows:
        path = tuple(int(v) for v in row.get("path", ()) or ())
        if not path or path in seen:
            continue
        seen.add(path)
        candidate_paths.append(path)
        mode_name = str(row.get("mode", "") or "")
        if mode_name:
            path_target_modes[path] = mode_name
        if len(candidate_paths) >= max_paths_i:
            break
    _append_path(candidate_paths, seen, fallback_best)
    for path_like in list(fallback_paths or []):
        _append_path(candidate_paths, seen, path_like)
    best_row = ranked_rows[0]
    out.update({
        "trained": True,
        "selected_path": [int(v) for v in best_row["path"]],
        "best_target_mode": str(best_row.get("mode", "") or ""),
        "best_relation": str(best_row.get("best_relation", "") or ""),
        "best_improvement_estimate": float(best_row.get("improvement_estimate", 0.0)),
        "candidate_paths": [list(path) for path in candidate_paths],
        "path_target_modes": {
            tuple(int(v) for v in path): str(mode)
            for path, mode in path_target_modes.items()
            if str(mode)
        },
        "rows": ranked_rows,
        "source": "critic_path_action" if "utility_estimate" in best_row else "critic_path_head",
    })
    return out


def _actor_critic_reward_terms(
    parent_eff_mse: Any,
    child_eff_mse: Any,
    *,
    created_new_residual_basin: bool = False,
    became_global_best: bool = False,
    wall_s: Any = None,
    novelty_bonus: float = 0.0,
    best_bonus: float = 0.0,
    time_penalty: float = 0.0,
    eps: float = 1.0e-30,
) -> dict[str, float]:
    try:
        parent_eff = max(0.0, float(parent_eff_mse))
    except Exception:
        parent_eff = float("nan")
    try:
        child_eff = max(0.0, float(child_eff_mse))
    except Exception:
        child_eff = float("nan")
    try:
        eps_f = max(1.0e-30, float(eps))
    except Exception:
        eps_f = 1.0e-30
    if math.isfinite(parent_eff) and math.isfinite(child_eff):
        log_gain = float(math.log(parent_eff + eps_f) - math.log(child_eff + eps_f))
    else:
        log_gain = 0.0
    novelty_term = float(max(0.0, float(novelty_bonus))) if bool(created_new_residual_basin) else 0.0
    best_term = float(max(0.0, float(best_bonus))) if bool(became_global_best) else 0.0
    try:
        wall = max(0.0, float(wall_s))
    except Exception:
        wall = 0.0
    try:
        time_penalty_f = max(0.0, float(time_penalty))
    except Exception:
        time_penalty_f = 0.0
    time_term = float(time_penalty_f * wall)
    reward = float(log_gain + novelty_term + best_term - time_term)
    return {
        "actor_critic_reward": reward,
        "actor_critic_reward_log_gain": float(log_gain),
        "actor_critic_reward_novelty_bonus": float(novelty_term),
        "actor_critic_reward_best_bonus": float(best_term),
        "actor_critic_reward_time_penalty": float(time_term),
        "actor_critic_reward_wall_s": float(wall),
    }


def _repair_controller_threshold(stats: dict[str, Any] | None) -> float:
    if not isinstance(stats, dict):
        return 0.0
    try:
        thr = max(0.0, float(stats.get("min_score", 0.0)))
    except Exception:
        thr = 0.0
    if not bool(stats.get("adaptive_enable", False)):
        return float(thr)
    hist = list(stats.get("score_hist", []) or [])
    try:
        window = int(stats.get("adapt_window", 0))
    except Exception:
        window = 0
    if window > 0 and len(hist) > window:
        hist = hist[-window:]
    try:
        min_n = int(stats.get("adapt_min_samples", 0))
    except Exception:
        min_n = 0
    if min_n > 0 and len(hist) < min_n:
        return float(thr)
    if not hist:
        return float(thr)
    try:
        q = float(stats.get("adapt_quantile", 0.75))
    except Exception:
        q = 0.75
    q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
    xs = sorted(float(v) for v in hist if math.isfinite(float(v)))
    if not xs:
        return float(thr)
    idx = int(q * (len(xs) - 1))
    idx = 0 if idx < 0 else (len(xs) - 1 if idx >= len(xs) else idx)
    return float(max(thr, float(xs[idx])))


def _repair_controller_component_gate(
    row: dict[str, Any] | None,
    components: dict[str, float] | None,
    stats: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    if not isinstance(stats, dict):
        return True, []
    if not isinstance(components, dict):
        components = {}
    reasons: list[str] = []
    try:
        min_concentration = max(0.0, float(stats.get("min_concentration", 0.0)))
    except Exception:
        min_concentration = 0.0
    try:
        concentration = float(components.get("concentration", 0.0))
    except Exception:
        concentration = 0.0
    if concentration < min_concentration:
        reasons.append("concentration")
    return (len(reasons) == 0), reasons


def _repair_parent_retry_gate(
    parent_key,
    parent_rec,
    n_evaluated: int,
    repair_parent_state: dict[Any, dict[str, Any]] | None,
    stats: dict[str, Any] | None,
) -> tuple[bool, str]:
    if not isinstance(repair_parent_state, dict):
        return True, "ok"
    if not isinstance(stats, dict):
        return True, "ok"
    row = repair_parent_state.get(parent_key, None)
    if not isinstance(row, dict):
        return True, "ok"
    expr_s = node_str(parent_rec.best_expr)
    if str(row.get("expr", "")) != expr_s:
        return True, "ok"
    try:
        parent_eff = float(getattr(parent_rec, "best_mse", float("inf")))
    except Exception:
        parent_eff = float("inf")
    try:
        ref_eff = float(row.get("ref_eff", parent_eff))
    except Exception:
        ref_eff = parent_eff
    try:
        reset_rel = max(0.0, float(stats.get("parent_reset_rel_improve", 0.0)))
    except Exception:
        reset_rel = 0.0
    if (
        math.isfinite(parent_eff)
        and math.isfinite(ref_eff)
        and (reset_rel > 0.0)
        and (parent_eff < ref_eff * (1.0 - reset_rel))
    ):
        return True, "ok"
    try:
        cooldown_until = int(row.get("cooldown_until", -1))
    except Exception:
        cooldown_until = -1
    if int(n_evaluated) < cooldown_until:
        return False, "cooldown"
    try:
        max_repeats = max(0, int(stats.get("parent_max_repeats", 0)))
    except Exception:
        max_repeats = 0
    try:
        attempts = max(0, int(row.get("attempts", 0)))
    except Exception:
        attempts = 0
    if max_repeats > 0 and attempts >= max_repeats:
        return False, "repeat_budget"
    return True, "ok"


def _repair_preview_signature(preview_expr: Any, preview_meta: Mapping[str, Any] | None) -> str:
    expr_s = node_str(preview_expr)
    if not expr_s:
        return ""
    try:
        path = tuple(int(v) for v in (preview_meta or {}).get("selected_path", ()) or ())
    except Exception:
        path = ()
    mode = str((preview_meta or {}).get("selected_target_mode", "") or "")
    return f"{expr_s}||{path}||{mode}"


def _repair_parent_preview_retry_gate(
    parent_key,
    parent_rec,
    preview_expr: Any,
    preview_meta: Mapping[str, Any] | None,
    repair_parent_state: dict[Any, dict[str, Any]] | None,
    stats: dict[str, Any] | None,
) -> tuple[bool, str]:
    if not isinstance(repair_parent_state, dict):
        return True, "ok"
    if not isinstance(stats, dict):
        return True, "ok"
    row = repair_parent_state.get(parent_key, None)
    if not isinstance(row, dict):
        return True, "ok"
    expr_s = node_str(parent_rec.best_expr)
    if str(row.get("expr", "")) != expr_s:
        return True, "ok"
    try:
        parent_eff = float(getattr(parent_rec, "best_mse", float("inf")))
    except Exception:
        parent_eff = float("inf")
    try:
        ref_eff = float(row.get("ref_eff", parent_eff))
    except Exception:
        ref_eff = parent_eff
    try:
        reset_rel = max(0.0, float(stats.get("parent_reset_rel_improve", 0.0)))
    except Exception:
        reset_rel = 0.0
    if (
        math.isfinite(parent_eff)
        and math.isfinite(ref_eff)
        and (reset_rel > 0.0)
        and (parent_eff < ref_eff * (1.0 - reset_rel))
    ):
        return True, "ok"
    signature = _repair_preview_signature(preview_expr, preview_meta)
    if not signature:
        return True, "ok"
    try:
        max_preview_repeats = max(0, int(stats.get("parent_preview_max_repeats", 1)))
    except Exception:
        max_preview_repeats = 1
    if max_preview_repeats <= 0:
        return True, "ok"
    preview_counts = row.get("preview_counts", None)
    if not isinstance(preview_counts, dict):
        return True, "ok"
    try:
        attempts = max(0, int(preview_counts.get(signature, 0)))
    except Exception:
        attempts = 0
    if attempts >= max_preview_repeats:
        return False, "repeat_signature"
    return True, "ok"


def _repair_parent_record_attempt(
    parent_key,
    parent_rec,
    n_evaluated: int,
    repair_parent_state: dict[Any, dict[str, Any]] | None,
    stats: dict[str, Any] | None,
    *,
    count_attempt: bool,
    preview_signature: str = "",
) -> None:
    if not isinstance(repair_parent_state, dict):
        return
    if not isinstance(stats, dict):
        return
    expr_s = node_str(parent_rec.best_expr)
    try:
        parent_eff = float(getattr(parent_rec, "best_mse", float("inf")))
    except Exception:
        parent_eff = float("inf")
    try:
        reset_rel = max(0.0, float(stats.get("parent_reset_rel_improve", 0.0)))
    except Exception:
        reset_rel = 0.0
    try:
        cooldown_gap = max(0, int(stats.get("parent_min_eval_gap", 0)))
    except Exception:
        cooldown_gap = 0

    row = repair_parent_state.get(parent_key, None)
    reset = not isinstance(row, dict) or str(row.get("expr", "")) != expr_s
    attempts = 0
    ref_eff = parent_eff
    preview_counts: dict[str, int] = {}
    if isinstance(row, dict) and not reset:
        try:
            ref_eff = float(row.get("ref_eff", parent_eff))
        except Exception:
            ref_eff = parent_eff
        if (
            math.isfinite(parent_eff)
            and math.isfinite(ref_eff)
            and (reset_rel > 0.0)
            and (parent_eff < ref_eff * (1.0 - reset_rel))
        ):
            reset = True
            ref_eff = parent_eff
        else:
            try:
                attempts = max(0, int(row.get("attempts", 0)))
            except Exception:
                attempts = 0
            prev = row.get("preview_counts", None)
            if isinstance(prev, dict):
                preview_counts = {
                    str(k): max(0, int(v))
                    for k, v in prev.items()
                }
    if count_attempt:
        attempts += 1
        signature = str(preview_signature or "")
        if signature:
            preview_counts[signature] = int(preview_counts.get(signature, 0)) + 1
    repair_parent_state[parent_key] = {
        "expr": expr_s,
        "ref_eff": float(parent_eff if reset else ref_eff),
        "last_eff": float(parent_eff),
        "last_eval": int(n_evaluated),
        "cooldown_until": int(n_evaluated + cooldown_gap),
        "attempts": int(attempts),
        "preview_counts": dict(preview_counts),
    }


def _repair_option_candidate_paths(
    node,
    anchor_path: Sequence[int] | None,
    *,
    ancestor_hops: int = 1,
    include_ancestors: bool = True,
    fallback_paths: Sequence[Sequence[int]] | None = None,
) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def _add_path(path_like) -> None:
        try:
            pp = tuple(int(v) for v in (path_like or ()))
        except Exception:
            return
        if not pp or pp in seen:
            return
        try:
            get_at(node, pp)
        except Exception:
            return
        seen.add(pp)
        out.append(pp)

    try:
        pp = tuple(int(v) for v in (anchor_path or ()))
    except Exception:
        pp = ()
    if pp:
        max_drop = max(0, int(ancestor_hops)) if bool(include_ancestors) else 0
        for drop in range(max_drop + 1):
            cand = pp[: max(0, len(pp) - drop)]
            _add_path(cand)
    if not out:
        for path in list(fallback_paths or []):
            _add_path(path)
    return out
