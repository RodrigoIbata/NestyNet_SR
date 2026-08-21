# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from nestynet_sr.sr_search.factorized_search.engine.signals import (
    InverseSteeringPotential,
    PathStateFeatures,
)
from nestynet_sr.sr_search.factorized_search.policy.features import (
    coerce_repair_feature_record,
    RepairControllerFeatureRecord,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_str(value: Any, default: str = "") -> str:
    try:
        return str(value)
    except Exception:
        return str(default)


def _clamp01(value: Any) -> float:
    return min(1.0, max(0.0, _safe_float(value, 0.0)))


def _bucket01(value: Any, n_bins: int = 5) -> int:
    n_bins = max(1, int(n_bins))
    v = _clamp01(value)
    idx = int(math.floor(v * float(n_bins)))
    if idx >= n_bins:
        idx = n_bins - 1
    if idx < 0:
        idx = 0
    return int(idx)


def _bucket_signed(value: Any, edges: Sequence[float] = (-0.25, -0.05, 0.05, 0.15, 0.35, 0.75)) -> int:
    v = _safe_float(value, 0.0)
    for i, thr in enumerate(edges):
        if v < float(thr):
            return int(i)
    return int(len(tuple(edges)))


def _bucket_neg_log10(value: Any) -> int:
    v = max(1.0e-30, _safe_float(value, 1.0e30))
    mag = max(0.0, -math.log10(v))
    return int(min(9, max(0, math.floor(mag / 2.0))))


def _aggregate_topk_scores(
    values: Sequence[float],
    *,
    top_k: int = 3,
) -> float:
    xs = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    if not xs:
        return 0.0
    xs.sort(reverse=True)
    keep = xs[: max(1, int(top_k))]
    m = max(keep)
    if not math.isfinite(m):
        return 0.0
    total = sum(math.exp(v - m) for v in keep)
    if total <= 1.0e-30:
        return float(m)
    return float(m + math.log(total))


def _path_concentration(path_entropy: Any, path_top_mass: Any, path_positive_count: Any) -> float:
    top_mass = _clamp01(path_top_mass)
    entropy = max(0.0, _safe_float(path_entropy, 0.0))
    n_pos = max(0.0, _safe_float(path_positive_count, 0.0))
    if n_pos > 1.0:
        h_max = max(math.log(n_pos), 1.0e-12)
        entropy_norm = min(1.0, max(0.0, entropy / h_max))
    else:
        entropy_norm = 0.0
    return 0.5 * top_mass + 0.5 * (1.0 - entropy_norm)


def _path_summary_stats(
    path_summaries: Sequence[ControllerPathSummary] | None,
    *,
    top_k: int = 3,
) -> dict[str, float]:
    rows = [row for row in (path_summaries or ())[: max(1, int(top_k))] if isinstance(row, ControllerPathSummary)]
    if not rows:
        return {
            "gain_mass": 0.0,
            "gap": 0.0,
            "support": 0.0,
            "mode_diversity": 0.0,
        }
    gains = [max(0.0, _safe_float(row.weighted_rel_gain, 0.0)) for row in rows]
    top = gains[0] if gains else 0.0
    second = gains[1] if len(gains) > 1 else 0.0
    total = sum(gains)
    if top > 1.0e-12:
        gap = min(1.0, max(0.0, (top - second) / top))
    else:
        gap = 0.0
    support_rows = [
        0.5 * _clamp01(row.valid_frac) + 0.5 * _clamp01(row.confidence)
        for row in rows
    ]
    mode_count = len({str(row.target_mode) for row in rows if str(row.target_mode)})
    return {
        "gain_mass": float(1.0 - math.exp(-max(0.0, total))),
        "gap": float(gap),
        "support": float(sum(support_rows) / max(1, len(support_rows))),
        "mode_diversity": float(min(1.0, mode_count / float(max(1, len(rows))))),
    }


_MACRO_ACTION_ROUTE = {
    "replace": "build",
    "wrap_un": "build",
    "add_rand": "build",
    "mul_rand": "build",
    "residual": "build",
    "boost": "build",
    "inv_steer": "repair",
    "repair_option": "repair",
    "hole_search": "repair",
    "prune": "simplify",
    "crossover": "recombine",
}


def _action_route(action_name: Any) -> str:
    return str(_MACRO_ACTION_ROUTE.get(str(action_name or "").strip(), ""))


@dataclass(frozen=True)
class ControllerPathSummary:
    path: tuple[int, ...] = ()
    target_mode: str = ""
    weighted_rel_gain: float = 0.0
    rel_gain: float = 0.0
    valid_frac: float = 0.0
    confidence: float = 0.0
    static_score: float = 0.0
    transport_rel: float = 0.0
    branch_factor: float = 1.0
    cut_factor: float = 1.0

    @classmethod
    def from_path_row(cls, row: PathStateFeatures | Mapping[str, Any] | None) -> ControllerPathSummary:
        if isinstance(row, PathStateFeatures):
            return cls(
                path=tuple(int(v) for v in row.path),
                target_mode=_safe_str(row.target_mode),
                weighted_rel_gain=_safe_float(row.weighted_rel_gain, 0.0),
                rel_gain=_safe_float(row.rel_gain, 0.0),
                valid_frac=_safe_float(row.valid_frac, 0.0),
                confidence=_safe_float(row.confidence, 0.0),
                static_score=_safe_float(row.static_score, 0.0),
                transport_rel=_safe_float(row.transport_rel, 0.0),
                branch_factor=_safe_float(row.branch_factor, 1.0),
                cut_factor=_safe_float(row.cut_factor, 1.0),
            )
        row = row if isinstance(row, Mapping) else {}
        path_like = row.get("path", ())
        try:
            path = tuple(int(v) for v in (path_like or ()))
        except Exception:
            path = ()
        return cls(
            path=path,
            target_mode=_safe_str(row.get("target_mode", "")),
            weighted_rel_gain=_safe_float(row.get("weighted_rel_gain", 0.0), 0.0),
            rel_gain=_safe_float(row.get("rel_gain", 0.0), 0.0),
            valid_frac=_safe_float(row.get("valid_frac", 0.0), 0.0),
            confidence=_safe_float(row.get("confidence", 0.0), 0.0),
            static_score=_safe_float(row.get("static_score", 0.0), 0.0),
            transport_rel=_safe_float(row.get("transport_rel", 0.0), 0.0),
            branch_factor=_safe_float(row.get("branch_factor", 1.0), 1.0),
            cut_factor=_safe_float(row.get("cut_factor", 1.0), 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": [int(v) for v in self.path],
            "target_mode": str(self.target_mode),
            "weighted_rel_gain": float(self.weighted_rel_gain),
            "rel_gain": float(self.rel_gain),
            "valid_frac": float(self.valid_frac),
            "confidence": float(self.confidence),
            "static_score": float(self.static_score),
            "transport_rel": float(self.transport_rel),
            "branch_factor": float(self.branch_factor),
            "cut_factor": float(self.cut_factor),
        }


@dataclass(frozen=True)
class MacroControllerState:
    parent_key: str = ""
    parent_expr: str = ""
    parent_root_op: str = ""
    parent_depth: int = 0
    parent_size: int = 0
    parent_best_eff_mse: float = float("inf")
    parent_best_raw_mse: float = float("inf")
    parent_visits: float = 0.0
    parent_visits_since_improve: float = 0.0
    parent_stagnation_score: float = 0.0
    parent_stagnation_ratio: float = 0.0
    allowed_actions: tuple[str, ...] = ()
    gate_allowed: bool = False
    gate_reason: str = ""
    repair_preview_available: bool = False
    repair_component_ok: bool = False
    repair_ready: bool = False
    repair_priority_score: float = 0.0
    repair_gate_score: float = 0.0
    repair_threshold: float = 0.0
    repair_potential: float = 0.0
    repair_path_concentration: float = 0.0
    repair_contrast: float = 0.0
    repair_candidate_count: int = 0
    path_entropy: float = 0.0
    path_top_mass: float = 0.0
    path_second_mass: float = 0.0
    path_positive_count: float = 0.0
    best_path: tuple[int, ...] = ()
    best_path_gain: float = 0.0
    best_path_valid_frac: float = 0.0
    best_path_confidence: float = 0.0
    best_path_transport_rel: float = 0.0
    best_path_static_score: float = 0.0
    selected_target_mode: str = ""
    refine_slot_count: int = 0
    refine_gate_potential: float = 0.0
    path_summaries: tuple[ControllerPathSummary, ...] = field(default_factory=tuple)
    source: str = ""

    @property
    def repair_margin(self) -> float:
        return float(self.repair_priority_score - self.repair_threshold)

    @property
    def bandit_state_key(self) -> tuple[Any, ...]:
        path_stats = _path_summary_stats(self.path_summaries)
        return (
            "macro_controller_v2",
            str(self.parent_root_op),
            int(max(0, min(8, self.parent_depth))),
            int(max(0, min(12, self.parent_size // 2))),
            _bucket_neg_log10(self.parent_best_eff_mse),
            _bucket01(self.parent_stagnation_score),
            _bucket01(self.repair_potential),
            _bucket01(self.repair_path_concentration),
            _bucket_signed(self.repair_contrast),
            _bucket_signed(self.repair_margin),
            int(bool(self.repair_ready)),
            int(bool(self.gate_allowed)),
            str(self.selected_target_mode),
            _bucket01(path_stats.get("gain_mass", 0.0)),
            _bucket01(path_stats.get("gap", 0.0)),
            _bucket01(path_stats.get("support", 0.0)),
            _bucket01(path_stats.get("mode_diversity", 0.0)),
            min(4, max(0, int(self.refine_slot_count))),
            _bucket01(self.refine_gate_potential),
            tuple(self.allowed_actions),
        )

    def with_allowed_actions(self, allowed_actions: Sequence[str]) -> MacroControllerState:
        names = tuple(str(a) for a in allowed_actions if str(a))
        return replace(self, allowed_actions=names)

    def to_flat_dict(self) -> dict[str, Any]:
        path_stats = _path_summary_stats(self.path_summaries)
        return {
            "parent_key": str(self.parent_key),
            "parent_expr": str(self.parent_expr),
            "parent_root_op": str(self.parent_root_op),
            "parent_depth": int(self.parent_depth),
            "parent_size": int(self.parent_size),
            "parent_best_eff_mse": float(self.parent_best_eff_mse),
            "parent_best_raw_mse": float(self.parent_best_raw_mse),
            "parent_visits": float(self.parent_visits),
            "parent_visits_since_improve": float(self.parent_visits_since_improve),
            "parent_stagnation_score": float(self.parent_stagnation_score),
            "parent_stagnation_ratio": float(self.parent_stagnation_ratio),
            "allowed_actions": [str(a) for a in self.allowed_actions],
            "gate_allowed": bool(self.gate_allowed),
            "gate_reason": str(self.gate_reason),
            "repair_preview_available": bool(self.repair_preview_available),
            "repair_component_ok": bool(self.repair_component_ok),
            "repair_ready": bool(self.repair_ready),
            "repair_priority_score": float(self.repair_priority_score),
            "repair_gate_score": float(self.repair_gate_score),
            "repair_threshold": float(self.repair_threshold),
            "repair_margin": float(self.repair_margin),
            "repair_potential": float(self.repair_potential),
            "repair_path_concentration": float(self.repair_path_concentration),
            "repair_contrast": float(self.repair_contrast),
            "repair_candidate_count": int(self.repair_candidate_count),
            "path_entropy": float(self.path_entropy),
            "path_top_mass": float(self.path_top_mass),
            "path_second_mass": float(self.path_second_mass),
            "path_positive_count": float(self.path_positive_count),
            "best_path": [int(v) for v in self.best_path],
            "best_path_gain": float(self.best_path_gain),
            "best_path_valid_frac": float(self.best_path_valid_frac),
            "best_path_confidence": float(self.best_path_confidence),
            "best_path_transport_rel": float(self.best_path_transport_rel),
            "best_path_static_score": float(self.best_path_static_score),
            "selected_target_mode": str(self.selected_target_mode),
            "refine_slot_count": int(self.refine_slot_count),
            "refine_gate_potential": float(self.refine_gate_potential),
            "path_summary_gain_mass": float(path_stats.get("gain_mass", 0.0)),
            "path_summary_gap": float(path_stats.get("gap", 0.0)),
            "path_summary_support": float(path_stats.get("support", 0.0)),
            "path_summary_mode_diversity": float(path_stats.get("mode_diversity", 0.0)),
            "path_summaries": [ps.to_dict() for ps in self.path_summaries],
            "source": str(self.source),
        }


@dataclass(frozen=True)
class MacroControllerConfig:
    ucb_c: float = 1.0
    eps: float = 0.10
    ema_alpha: float = 0.10
    global_prior_n: int = 5
    repair_bonus: float = 0.50
    repair_margin_scale: float = 0.75
    build_bias: float = 0.05
    inverse_bonus: float = 0.10
    stagnation_bonus: float = 0.15
    learned_policy_weight: float = 0.75
    learned_route_weight: float = 0.60
    learned_q_weight: float = 0.75
    learned_value_scale: float = 0.75
    learned_primary_min_confidence: float = 0.60
    learned_primary_min_margin: float = 0.10


@dataclass(frozen=True)
class MacroActionDecision:
    action_name: str
    state_key: tuple[Any, ...]
    selected_route: str | None = None
    selected_path: tuple[int, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    route_decision_scores: dict[str, float] = field(default_factory=dict)
    bandit_scores: dict[str, float] = field(default_factory=dict)
    bonus_scores: dict[str, float] = field(default_factory=dict)
    learned_scores: dict[str, float] = field(default_factory=dict)
    learned_action_probs: dict[str, float] = field(default_factory=dict)
    learned_route_scores: dict[str, float] = field(default_factory=dict)
    learned_route_probs: dict[str, float] = field(default_factory=dict)
    learned_action_value_estimates: dict[str, float] = field(default_factory=dict)
    learned_action_value_normalized: dict[str, float] = field(default_factory=dict)
    learned_best_route: str | None = None
    learned_best_path: tuple[int, ...] = ()
    learned_value_estimate: float | None = None
    learned_value_normalized: float | None = None
    learned_confidence: float = 0.0
    policy_source: str = "score_argmax"
    repair_margin: float = 0.0


class MacroController:
    def __init__(
        self,
        action_names: Sequence[str],
        *,
        config: MacroControllerConfig | None = None,
        ucb_c: float | None = None,
        eps: float | None = None,
        ema_alpha: float | None = None,
        global_prior_n: int | None = None,
        repair_bonus: float | None = None,
        repair_margin_scale: float | None = None,
        build_bias: float | None = None,
        inverse_bonus: float | None = None,
        stagnation_bonus: float | None = None,
        learned_policy_weight: float | None = None,
        learned_route_weight: float | None = None,
        learned_q_weight: float | None = None,
        learned_value_scale: float | None = None,
        learned_primary_min_confidence: float | None = None,
        learned_primary_min_margin: float | None = None,
    ):
        cfg = config if isinstance(config, MacroControllerConfig) else MacroControllerConfig()
        if ucb_c is not None:
            cfg = replace(cfg, ucb_c=float(ucb_c))
        if eps is not None:
            cfg = replace(cfg, eps=float(eps))
        if ema_alpha is not None:
            cfg = replace(cfg, ema_alpha=float(ema_alpha))
        if global_prior_n is not None:
            cfg = replace(cfg, global_prior_n=max(1, int(global_prior_n)))
        if repair_bonus is not None:
            cfg = replace(cfg, repair_bonus=float(repair_bonus))
        if repair_margin_scale is not None:
            cfg = replace(cfg, repair_margin_scale=float(repair_margin_scale))
        if build_bias is not None:
            cfg = replace(cfg, build_bias=float(build_bias))
        if inverse_bonus is not None:
            cfg = replace(cfg, inverse_bonus=float(inverse_bonus))
        if stagnation_bonus is not None:
            cfg = replace(cfg, stagnation_bonus=float(stagnation_bonus))
        if learned_policy_weight is not None:
            cfg = replace(cfg, learned_policy_weight=float(learned_policy_weight))
        if learned_route_weight is not None:
            cfg = replace(cfg, learned_route_weight=float(learned_route_weight))
        if learned_q_weight is not None:
            cfg = replace(cfg, learned_q_weight=float(learned_q_weight))
        if learned_value_scale is not None:
            cfg = replace(cfg, learned_value_scale=float(learned_value_scale))
        if learned_primary_min_confidence is not None:
            cfg = replace(cfg, learned_primary_min_confidence=float(learned_primary_min_confidence))
        if learned_primary_min_margin is not None:
            cfg = replace(cfg, learned_primary_min_margin=float(learned_primary_min_margin))
        self.cfg = cfg
        ordered: list[str] = []
        seen: set[str] = set()
        for action_name in action_names:
            name = str(action_name or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        self.action_names = tuple(ordered)
        self.action_set = set(self.action_names)
        self.n_s: dict[tuple[Any, ...], int] = {}
        self.n_sa: dict[tuple[tuple[Any, ...], str], int] = {}
        self.q_sa: dict[tuple[tuple[Any, ...], str], float] = {}
        self.n_g: dict[str, int] = {}
        self.q_g: dict[str, float] = {}

    def _learned_policy_scores(
        self,
        acts: Sequence[str],
        policy_guidance: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        empty_scores = {
            "scores": {str(a): 0.0 for a in acts},
            "macro_scores": {str(a): 0.0 for a in acts},
            "action_probs": {str(a): 0.0 for a in acts},
            "within_route_action_probs": {str(a): 0.0 for a in acts},
            "route_scores": {str(a): 0.0 for a in acts},
            "route_probs": {},
            "q_estimates": {},
            "q_normalized": {},
            "q_scores": {str(a): 0.0 for a in acts},
            "value_estimate": None,
            "value_normalized": None,
            "confidence": 0.0,
            "best_route": None,
        }
        if not isinstance(policy_guidance, Mapping):
            return empty_scores
        path_action_row = policy_guidance.get("path_action", None)
        if isinstance(path_action_row, Mapping) and bool(path_action_row.get("trained", False)):
            path_rows = [row for row in list(path_action_row.get("rows", []) or []) if isinstance(row, Mapping)]
            if path_rows:
                total_tuple_scores: dict[str, list[float]] = {str(a): [] for a in acts}
                macro_score_terms: dict[str, list[float]] = {str(a): [] for a in acts}
                route_score_terms: dict[str, list[float]] = {str(a): [] for a in acts}
                q_score_terms: dict[str, list[float]] = {str(a): [] for a in acts}
                best_tuple_scores = {str(a): float("-inf") for a in acts}
                action_scores = {str(a): 0.0 for a in acts}
                macro_scores = {str(a): 0.0 for a in acts}
                action_prob_mass = {str(a): 0.0 for a in acts}
                route_scores = {str(a): 0.0 for a in acts}
                q_scores = {str(a): 0.0 for a in acts}
                q_estimates_out: dict[str, float] = {}
                q_normalized_out: dict[str, float] = {}
                selected_paths: dict[str, tuple[int, ...]] = {}
                route_probs_out: dict[str, float] = {}
                active_routes = tuple(sorted({_action_route(action_name) for action_name in acts if _action_route(action_name)}))
                baseline_route = 1.0 / float(max(1, len(active_routes)))
                baseline_path = 1.0 / float(max(1, len(path_rows)))
                best_tuple = None
                tuple_beam: list[dict[str, Any]] = []
                for row in path_rows:
                    path_like = row.get("path", ())
                    try:
                        path = tuple(int(v) for v in (path_like or ()))
                    except Exception:
                        path = ()
                    path_prob = _clamp01(row.get("path_prob", row.get("prob", 0.0)))
                    route_probs_map = dict(row.get("route_probs", {}) or {})
                    hier_action_probs = dict(row.get("action_probs", {}) or {})
                    within_route_probs = dict(row.get("within_route_action_probs", {}) or {})
                    q_est_map = dict(row.get("q_estimates", {}) or {})
                    q_norm_map = dict(row.get("normalized_estimates", {}) or {})
                    for route_name, route_prob in route_probs_map.items():
                        route_key = str(route_name)
                        route_probs_out.setdefault(route_key, 0.0)
                        route_probs_out[route_key] += float(path_prob) * _clamp01(route_prob)
                    q_center = sum(_safe_float(q_norm_map.get(name, 0.0), 0.0) for name in acts) / float(max(1, len(acts)))
                    for action_name in acts:
                        route_name = _action_route(action_name)
                        if not route_name:
                            continue
                        route_prob = _clamp01(route_probs_map.get(route_name, 0.0))
                        local_prob = _clamp01(within_route_probs.get(action_name, 0.0))
                        hier_prob = _clamp01(hier_action_probs.get(action_name, 0.0))
                        route_actions = [name for name in acts if _action_route(name) == route_name]
                        baseline_action = 1.0 / float(max(1, len(route_actions)))
                        policy_score = 0.0
                        if path_prob > 0.0 and local_prob > 0.0:
                            policy_score = float(self.cfg.learned_policy_weight) * (
                                math.log(max(1.0e-12, path_prob) / baseline_path)
                                + math.log(max(1.0e-12, local_prob) / baseline_action)
                            )
                        route_score = 0.0
                        if route_prob > 0.0:
                            route_score = float(self.cfg.learned_route_weight) * math.log(max(1.0e-12, route_prob) / baseline_route)
                        q_norm = _safe_float(q_norm_map.get(action_name, 0.0), 0.0)
                        q_score = float(self.cfg.learned_q_weight) * math.tanh(q_norm - q_center)
                        score = float(policy_score + route_score + q_score)
                        total_tuple_scores[action_name].append(float(score))
                        macro_score_terms[action_name].append(float(policy_score))
                        route_score_terms[action_name].append(float(route_score))
                        q_score_terms[action_name].append(float(q_score))
                        action_prob_mass[action_name] += float(path_prob) * float(hier_prob)
                        prev_score = float(best_tuple_scores.get(action_name, float("-inf")))
                        if score > prev_score or (math.isclose(score, prev_score) and path):
                            best_tuple_scores[action_name] = float(score)
                            selected_paths[action_name] = path
                            q_estimates_out[action_name] = _safe_float(q_est_map.get(action_name, 0.0), 0.0)
                            q_normalized_out[action_name] = q_norm
                        tuple_row = {
                            "path": [int(v) for v in path],
                            "route": route_name,
                            "action": action_name,
                            "score": float(score),
                            "policy_score": float(policy_score),
                            "route_score": float(route_score),
                            "q_score": float(q_score),
                            "path_prob": float(path_prob),
                            "action_prob": float(hier_prob),
                            "q_estimate": _safe_float(q_est_map.get(action_name, 0.0), 0.0),
                        }
                        tuple_beam.append(tuple_row)
                        tuple_key = (score, hier_prob, _safe_float(q_est_map.get(action_name, 0.0), 0.0), path, action_name)
                        if best_tuple is None or tuple_key > best_tuple[0]:
                            best_tuple = (
                                tuple_key,
                                {
                                    "path": path,
                                    "action": action_name,
                                    "route": route_name,
                                },
                            )
                route_total = sum(float(v) for v in route_probs_out.values())
                if route_total > 1.0e-12:
                    route_probs_out = {
                        str(name): float(value / route_total)
                        for name, value in route_probs_out.items()
                    }
                action_total = sum(float(v) for v in action_prob_mass.values())
                if action_total > 1.0e-12:
                    action_probs = {
                        str(name): float(value / action_total)
                        for name, value in action_prob_mass.items()
                    }
                else:
                    action_probs = {str(a): 0.0 for a in acts}
                within_route_action_probs = {str(a): 0.0 for a in acts}
                route_groups = {
                    str(route_name): [name for name in acts if _action_route(name) == route_name]
                    for route_name in active_routes
                }
                for route_name, route_actions in route_groups.items():
                    route_mass = sum(float(action_probs.get(name, 0.0)) for name in route_actions)
                    if route_mass <= 1.0e-12:
                        continue
                    for name in route_actions:
                        within_route_action_probs[name] = float(action_probs.get(name, 0.0)) / route_mass
                for action_name in acts:
                    action_scores[action_name] = _aggregate_topk_scores(total_tuple_scores.get(action_name, ()))
                    macro_scores[action_name] = _aggregate_topk_scores(macro_score_terms.get(action_name, ()))
                    route_scores[action_name] = _aggregate_topk_scores(route_score_terms.get(action_name, ()))
                    q_scores[action_name] = _aggregate_topk_scores(q_score_terms.get(action_name, ()))
                confidence_terms = []
                if action_probs:
                    confidence_terms.append(max(action_probs.values(), default=0.0))
                if route_probs_out:
                    confidence_terms.append(max(route_probs_out.values(), default=0.0))
                confidence = max(confidence_terms) if confidence_terms else 0.0
                best_tuple_payload = best_tuple[1] if isinstance(best_tuple, tuple) else {}
                tuple_beam.sort(
                    key=lambda row: (
                        float(row.get("score", float("-inf"))),
                        float(row.get("action_prob", 0.0)),
                        float(row.get("q_estimate", float("-inf"))),
                        tuple(int(v) for v in row.get("path", []) or ()),
                        str(row.get("action", "")),
                    ),
                    reverse=True,
                )
                return {
                    "scores": action_scores,
                    "macro_scores": macro_scores,
                    "action_probs": action_probs,
                    "within_route_action_probs": within_route_action_probs,
                    "route_scores": route_scores,
                    "route_probs": route_probs_out,
                    "q_estimates": q_estimates_out,
                    "q_normalized": q_normalized_out,
                    "q_scores": q_scores,
                    "value_estimate": None,
                    "value_normalized": None,
                    "confidence": float(confidence),
                    "best_route": best_tuple_payload.get("route", None),
                    "best_path": best_tuple_payload.get("path", ()),
                    "selected_paths": selected_paths,
                    "tuple_beam": tuple_beam[: max(1, min(8, len(tuple_beam)))],
                    "used_path_action": True,
                }
        action_probs = {str(a): 0.0 for a in acts}
        macro_row = policy_guidance.get("macro_action", None)
        probs_raw = macro_row.get("probs", None) if isinstance(macro_row, Mapping) and bool(macro_row.get("trained", False)) else None
        if isinstance(probs_raw, Mapping):
            action_probs = {
                str(action_name): _clamp01(probs_raw.get(action_name, 0.0))
                for action_name in acts
            }
            total = sum(action_probs.values())
            if total > 1.0e-12:
                action_probs = {name: float(value / total) for name, value in action_probs.items()}
            else:
                action_probs = {str(a): 0.0 for a in acts}

        route_probs_out: dict[str, float] = {}
        route_scores = {str(a): 0.0 for a in acts}
        route_row = policy_guidance.get("route", None)
        route_probs_raw = route_row.get("probs", None) if isinstance(route_row, Mapping) and bool(route_row.get("trained", False)) else None
        active_routes = tuple(sorted({_action_route(action_name) for action_name in acts if _action_route(action_name)}))
        if isinstance(route_probs_raw, Mapping) and active_routes:
            route_probs = {
                str(route_name): _clamp01(route_probs_raw.get(route_name, 0.0))
                for route_name in active_routes
            }
            total = sum(route_probs.values())
            if total > 1.0e-12:
                route_probs_out = {name: float(value / total) for name, value in route_probs.items()}
                baseline_route = 1.0 / float(max(1, len(active_routes)))
                route_weight = max(0.0, float(self.cfg.learned_route_weight))
                for action_name in acts:
                    route_name = _action_route(action_name)
                    if route_name:
                        route_scores[str(action_name)] = float(
                            route_weight * math.log(max(1.0e-12, route_probs_out.get(route_name, 0.0)) / baseline_route)
                        )

        q_estimates_out: dict[str, float] = {}
        q_normalized_out: dict[str, float] = {}
        q_scores = {str(a): 0.0 for a in acts}
        q_row = policy_guidance.get("action_value", None)
        if isinstance(q_row, Mapping) and bool(q_row.get("trained", False)):
            q_est_raw = q_row.get("estimates", None)
            q_norm_raw = q_row.get("normalized_estimates", None)
            if isinstance(q_est_raw, Mapping):
                q_estimates_out = {
                    str(action_name): _safe_float(q_est_raw.get(action_name, 0.0), 0.0)
                    for action_name in acts
                }
            if isinstance(q_norm_raw, Mapping):
                q_normalized_out = {
                    str(action_name): _safe_float(q_norm_raw.get(action_name, 0.0), 0.0)
                    for action_name in acts
                }
            if not q_normalized_out and q_estimates_out:
                q_vals = list(q_estimates_out.values())
                q_mean = sum(q_vals) / float(max(1, len(q_vals)))
                q_var = sum((val - q_mean) ** 2 for val in q_vals) / float(max(1, len(q_vals)))
                q_std = math.sqrt(max(1.0e-6, q_var))
                q_normalized_out = {
                    name: float((val - q_mean) / q_std)
                    for name, val in q_estimates_out.items()
                }
            if q_normalized_out:
                q_weight = max(0.0, float(self.cfg.learned_q_weight))
                q_center = sum(q_normalized_out.values()) / float(max(1, len(q_normalized_out)))
                q_scores = {
                    name: float(q_weight * math.tanh(q_normalized_out[name] - q_center))
                    for name in q_normalized_out
                }

        value_row = policy_guidance.get("value", None)
        value_estimate = None
        value_norm = None
        if isinstance(value_row, Mapping) and bool(value_row.get("trained", False)):
            try:
                value_estimate = float(value_row.get("estimate", None))
            except Exception:
                value_estimate = None
            try:
                value_norm = float(value_row.get("normalized_estimate", None))
            except Exception:
                value_norm = None
        confidence_terms: list[float] = []
        value_mag = value_norm if (value_norm is not None and math.isfinite(value_norm)) else value_estimate
        if value_mag is not None and math.isfinite(float(value_mag)):
            confidence_terms.append(0.5 + 0.5 * abs(math.tanh(float(self.cfg.learned_value_scale) * float(value_mag))))
        if action_probs:
            confidence_terms.append(max(action_probs.values(), default=0.0))
        if route_probs_out:
            confidence_terms.append(max(route_probs_out.values(), default=0.0))
        if q_normalized_out:
            q_span = max(q_normalized_out.values()) - min(q_normalized_out.values())
            confidence_terms.append(0.5 + 0.5 * abs(math.tanh(q_span)))
        confidence = max(confidence_terms) if confidence_terms else 0.0
        weight = max(0.0, float(self.cfg.learned_policy_weight))
        route_groups: dict[str, list[str]] = {}
        for name in acts:
            route_name = _action_route(name)
            if route_name:
                route_groups.setdefault(str(route_name), []).append(str(name))
        within_route_action_probs = {str(a): 0.0 for a in acts}
        macro_scores = {str(a): 0.0 for a in acts}
        for route_name, route_actions in route_groups.items():
            route_mass = sum(float(action_probs.get(name, 0.0)) for name in route_actions)
            if route_mass <= 1.0e-12:
                continue
            baseline = 1.0 / float(max(1, len(route_actions)))
            for name in route_actions:
                local_prob = float(action_probs.get(name, 0.0)) / route_mass
                within_route_action_probs[name] = float(local_prob)
                if local_prob > 0.0:
                    macro_scores[name] = float(
                        weight * confidence * math.log(max(1.0e-12, local_prob) / baseline)
                    )
        learned_scores = {}
        for name in acts:
            learned_scores[name] = float(
                macro_scores.get(name, 0.0)
                + route_scores.get(name, 0.0)
                + q_scores.get(name, 0.0)
            )
        best_route = None
        if route_probs_out:
            best_route = max(route_probs_out, key=lambda name: (route_probs_out.get(name, 0.0), name))
        return {
            "scores": learned_scores,
            "macro_scores": macro_scores,
            "action_probs": action_probs,
            "within_route_action_probs": within_route_action_probs,
            "route_scores": route_scores,
            "route_probs": route_probs_out,
            "q_estimates": q_estimates_out,
            "q_normalized": q_normalized_out,
            "q_scores": q_scores,
            "value_estimate": value_estimate,
            "value_normalized": value_norm,
            "confidence": float(confidence),
            "best_route": best_route,
            "best_path": (),
            "selected_paths": {},
            "used_path_action": False,
        }

    def _get(self, d: dict[Any, Any], k: Any, default: Any) -> Any:
        return d[k] if k in d else default

    def _bandit_score(self, state_key: tuple[Any, ...], action_name: str) -> float:
        n_s = int(self._get(self.n_s, state_key, 0))
        n_sa = int(self._get(self.n_sa, (state_key, action_name), 0))
        q_local = _safe_float(self._get(self.q_sa, (state_key, action_name), 0.0), 0.0)
        q_global = _safe_float(self._get(self.q_g, action_name, 0.0), 0.0)
        prior_n = max(1, int(self.cfg.global_prior_n))
        w = min(n_sa, prior_n) / float(prior_n)
        q = w * q_local + (1.0 - w) * q_global
        bonus = float(self.cfg.ucb_c) * math.sqrt(math.log(n_s + 1.0) / (n_sa + 1.0))
        return float(q + bonus)

    def _score_margin(self, acts: Sequence[str], scores: Mapping[str, Any] | None) -> float:
        vals = sorted(
            (
                float(scores.get(str(name), float("-inf")))
                for name in acts
                if math.isfinite(float(scores.get(str(name), float("-inf"))))
            ),
            reverse=True,
        )
        if not vals:
            return 0.0
        if len(vals) == 1:
            return float("inf")
        return float(vals[0] - vals[1])

    def _has_nonzero_scores(self, acts: Sequence[str], scores: Mapping[str, Any] | None) -> bool:
        if not isinstance(scores, Mapping):
            return False
        for name in acts:
            try:
                if abs(float(scores.get(str(name), 0.0) or 0.0)) > 1.0e-12:
                    return True
            except Exception:
                continue
        return False

    def _route_decision_scores(
        self,
        route_groups: Mapping[str, Sequence[str]],
        action_scores: Mapping[str, float],
    ) -> dict[str, float]:
        if len(route_groups) <= 1:
            return {}
        out: dict[str, float] = {}
        for route_name, route_actions in route_groups.items():
            out[str(route_name)] = float(
                max(float(action_scores.get(name, float("-inf"))) for name in route_actions)
            )
        return out

    def _action_bonus(self, state: MacroControllerState, action_name: str) -> float:
        name = str(action_name or "")
        repair_margin = max(0.0, float(state.repair_margin))
        stagnation = _clamp01(state.parent_stagnation_score)
        path_stats = _path_summary_stats(state.path_summaries)
        if name == "repair_option":
            if not bool(state.repair_ready):
                return -1.0e9
            cand_cost = min(1.0, math.log1p(max(0.0, float(state.repair_candidate_count))) / math.log(17.0))
            contrast = math.tanh(max(0.0, float(state.repair_contrast)) / 3.0)
            return float(self.cfg.repair_bonus) * (
                repair_margin
                + 0.75 * float(state.repair_potential)
                + 0.25 * float(state.repair_path_concentration)
                + 0.10 * contrast
                + 0.10 * float(path_stats.get("gain_mass", 0.0))
                + 0.05 * float(path_stats.get("support", 0.0))
                + 0.05 * float(path_stats.get("gap", 0.0))
                - 0.10 * cand_cost
            )
        if name == "inv_steer":
            signal = max(0.0, float(state.best_path_gain), float(state.repair_potential))
            bonus = float(self.cfg.inverse_bonus) * signal
            bonus += 0.05 * _clamp01(state.best_path_valid_frac)
            bonus += 0.05 * _clamp01(state.best_path_confidence)
            bonus += 0.05 * float(path_stats.get("support", 0.0))
            if not bool(state.gate_allowed):
                bonus -= 0.25
            if bool(state.repair_ready):
                bonus -= 0.25 * float(self.cfg.repair_margin_scale) * repair_margin
            return float(bonus)
        bonus = float(self.cfg.build_bias)
        if name in {"add_rand", "mul_rand", "residual", "boost", "crossover"}:
            bonus += float(self.cfg.stagnation_bonus) * stagnation
        elif name in {"replace", "wrap_un"}:
            bonus += 0.5 * float(self.cfg.build_bias) * (1.0 - stagnation)
        elif name == "prune":
            bonus += 0.05 * (1.0 - stagnation)
        if bool(state.repair_ready):
            bonus -= 0.5 * float(self.cfg.repair_margin_scale) * repair_margin
        return float(bonus)

    def select_action(
        self,
        state: MacroControllerState,
        rng,
        *,
        allowed_actions: Sequence[str] | None = None,
        policy_guidance: Mapping[str, Any] | None = None,
    ) -> MacroActionDecision:
        acts_raw = list(allowed_actions if allowed_actions is not None else state.allowed_actions)
        if not acts_raw:
            acts_raw = list(self.action_names)
        acts: list[str] = []
        seen: set[str] = set()
        for act in acts_raw:
            name = str(act or "").strip()
            if not name or name not in self.action_set or name in seen:
                continue
            seen.add(name)
            acts.append(name)
        if not acts:
            acts = list(self.action_names)
        state_key = state.bandit_state_key
        bandit_scores: dict[str, float] = {}
        bonus_scores: dict[str, float] = {}
        learned = self._learned_policy_scores(
            acts,
            policy_guidance,
        )
        learned_scores = dict(learned.get("scores", {}) or {})
        learned_macro_scores = dict(learned.get("macro_scores", {}) or {})
        learned_probs = dict(learned.get("action_probs", {}) or {})
        learned_route_scores = dict(learned.get("route_scores", {}) or {})
        learned_route_probs = dict(learned.get("route_probs", {}) or {})
        learned_q_scores = dict(learned.get("q_scores", {}) or {})
        learned_action_value_estimates = dict(learned.get("q_estimates", {}) or {})
        learned_action_value_normalized = dict(learned.get("q_normalized", {}) or {})
        learned_value_estimate = learned.get("value_estimate", None)
        learned_value_normalized = learned.get("value_normalized", None)
        learned_confidence = float(learned.get("confidence", 0.0) or 0.0)
        learned_best_route = learned.get("best_route", None)
        learned_best_path = tuple(int(v) for v in (learned.get("best_path", ()) or ()))
        learned_selected_paths = {
            str(k): tuple(int(v) for v in (path or ()))
            for k, path in dict(learned.get("selected_paths", {}) or {}).items()
        }
        used_path_action = bool(learned.get("used_path_action", False))
        fallback_scores: dict[str, float] = {}
        for action_name in acts:
            b_score = self._bandit_score(state_key, action_name)
            x_score = self._action_bonus(state, action_name)
            bandit_scores[action_name] = float(b_score)
            bonus_scores[action_name] = float(x_score)
            fallback_scores[action_name] = float(b_score + x_score)
        selected_route = None
        selected_path: tuple[int, ...] = ()
        route_groups: dict[str, list[str]] = {}
        for action_name in acts:
            route_name = _action_route(action_name)
            route_key = str(route_name) if route_name else f"action::{action_name}"
            route_groups.setdefault(route_key, []).append(action_name)
        has_learned_signal = (
            self._has_nonzero_scores(acts, learned_scores)
            or self._has_nonzero_scores(acts, learned_macro_scores)
            or self._has_nonzero_scores(acts, learned_route_scores)
            or self._has_nonzero_scores(acts, learned_q_scores)
        )
        learned_margin = self._score_margin(acts, learned_scores)
        use_learned_primary = bool(
            has_learned_signal and (
                float(learned_confidence) >= float(self.cfg.learned_primary_min_confidence)
                or float(learned_margin) >= float(self.cfg.learned_primary_min_margin)
            )
        )
        if use_learned_primary:
            scores = {str(name): float(learned_scores.get(name, 0.0)) for name in acts}
        else:
            scores = {str(name): float(fallback_scores.get(name, float("-inf"))) for name in acts}
        route_decision_scores = self._route_decision_scores(route_groups, scores)
        if rng.random() < float(self.cfg.eps):
            chosen = rng.choice(acts)
            source = "eps_random"
            selected_route = _action_route(chosen) or None
        else:
            if route_decision_scores:
                selected_route = max(route_decision_scores, key=lambda name: (route_decision_scores.get(name, float("-inf")), name))
                route_actions = list(route_groups.get(selected_route, acts))
                chosen = max(route_actions, key=lambda a: (scores.get(a, float("-inf")), fallback_scores.get(a, float("-inf")), a))
                if use_learned_primary:
                    selected_path = tuple(learned_selected_paths.get(chosen, ()))
                if use_learned_primary and used_path_action:
                    source = "path_tuple_hierarchical_argmax"
                elif use_learned_primary and any(abs(v) > 1.0e-12 for v in learned_route_scores.values()):
                    source = "hierarchical_learned_route_argmax"
                elif use_learned_primary and any(abs(v) > 1.0e-12 for v in learned_scores.values()):
                    source = "hierarchical_learned_action_argmax"
                elif has_learned_signal:
                    source = "fallback_hierarchical_score_argmax"
                else:
                    source = "hierarchical_score_argmax"
                if selected_route.startswith("action::"):
                    selected_route = _action_route(chosen) or None
            else:
                chosen = max(acts, key=lambda a: (scores.get(a, float("-inf")), fallback_scores.get(a, float("-inf")), a))
                selected_route = _action_route(chosen) or None
                if use_learned_primary:
                    selected_path = tuple(learned_selected_paths.get(chosen, ()))
                if use_learned_primary and used_path_action:
                    source = "path_tuple_argmax"
                elif use_learned_primary:
                    source = "learned_primary_argmax"
                elif has_learned_signal:
                    source = "fallback_score_argmax"
                else:
                    source = "score_argmax"
        return MacroActionDecision(
            action_name=str(chosen),
            selected_route=None if selected_route is None else str(selected_route),
            selected_path=selected_path,
            state_key=state_key,
            scores=scores,
            route_decision_scores=route_decision_scores,
            bandit_scores=bandit_scores,
            bonus_scores=bonus_scores,
            learned_scores=learned_scores,
            learned_action_probs=learned_probs,
            learned_route_scores=learned_route_scores,
            learned_route_probs=learned_route_probs,
            learned_action_value_estimates=learned_action_value_estimates,
            learned_action_value_normalized=learned_action_value_normalized,
            learned_best_route=None if learned_best_route is None else str(learned_best_route),
            learned_best_path=learned_best_path,
            learned_value_estimate=learned_value_estimate,
            learned_value_normalized=learned_value_normalized,
            learned_confidence=float(learned_confidence),
            policy_source=str(source),
            repair_margin=float(state.repair_margin),
        )

    def update(self, state: MacroControllerState, action_name: str, reward: float) -> None:
        name = str(action_name or "").strip()
        if name not in self.action_set:
            return
        state_key = state.bandit_state_key
        self.n_s[state_key] = int(self._get(self.n_s, state_key, 0)) + 1
        k = (state_key, name)
        n = int(self._get(self.n_sa, k, 0))
        q = _safe_float(self._get(self.q_sa, k, 0.0), 0.0)
        reward_f = _safe_float(reward, 0.0)
        if n == 0:
            q = reward_f
        else:
            q = float(self.cfg.ema_alpha) * reward_f + (1.0 - float(self.cfg.ema_alpha)) * q
        self.n_sa[k] = n + 1
        self.q_sa[k] = float(q)
        n_g = int(self._get(self.n_g, name, 0))
        q_g = _safe_float(self._get(self.q_g, name, 0.0), 0.0)
        if n_g == 0:
            q_g = reward_f
        else:
            q_g = float(self.cfg.ema_alpha) * reward_f + (1.0 - float(self.cfg.ema_alpha)) * q_g
        self.n_g[name] = n_g + 1
        self.q_g[name] = float(q_g)

    def summary(self, topk: int = 8) -> list[tuple[float, str, int]]:
        out: list[tuple[float, str, int]] = []
        qg = {a: [] for a in self.action_names}
        for (_, action_name), q in self.q_sa.items():
            qg.setdefault(action_name, []).append(_safe_float(q, 0.0))
        for action_name in self.action_names:
            arr = qg.get(action_name, [])
            mu = (sum(arr) / max(1, len(arr))) if arr else _safe_float(self.q_g.get(action_name, 0.0), 0.0)
            out.append((float(mu), str(action_name), len(arr)))
        out.sort(reverse=True)
        return out[: max(1, int(topk))]


def _coerce_action_names(
    allowed_actions: Sequence[int | str] | None,
    action_name_map: Mapping[int, str] | None,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for action in allowed_actions or ():
        if isinstance(action, str):
            name = str(action).strip()
        else:
            name = str((action_name_map or {}).get(action, "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def build_macro_controller_state(
    *,
    parent_key: Any,
    parent_expr: Any,
    parent_root_op: str,
    parent_depth: int,
    parent_size: int,
    parent_best_eff_mse: float | None = None,
    parent_best_raw_mse: float | None = None,
    parent_visits: float | None = None,
    parent_visits_since_improve: float | None = None,
    parent_stagnation_score: float | None = None,
    parent_stagnation_ratio: float | None = None,
    allowed_actions: Sequence[int | str] | None = None,
    action_name_map: Mapping[int, str] | None,
    gate_diag: InverseSteeringPotential | None = None,
    controller_row: RepairControllerFeatureRecord | Mapping[str, Any] | None = None,
    repair_priority_score: float | None = None,
    repair_gate_score: float | None = None,
    repair_threshold: float | None = None,
    repair_ready: bool = False,
    repair_preview_available: bool = False,
    repair_component_ok: bool = False,
    refine_slot_count: int = 0,
    refine_gate_potential: float = 0.0,
    top_k_paths: int = 4,
    source: str = "",
) -> MacroControllerState:
    record = coerce_repair_feature_record(
        controller_row,
        gate_diag=gate_diag,
        refine_features={
            "refine_slot_count": int(refine_slot_count),
            "refine_gate_potential": float(refine_gate_potential),
        },
        top_k_paths=top_k_paths,
    )
    action_names = _coerce_action_names(allowed_actions, action_name_map)
    if bool(repair_ready) and "repair_option" not in action_names:
        action_names.append("repair_option")
    payload = record.to_macro_state_payload(
        parent_key=parent_key,
        parent_expr=parent_expr,
        parent_root_op=parent_root_op,
        parent_depth=parent_depth,
        parent_size=parent_size,
        allowed_action_names=action_names,
        repair_priority_score=repair_priority_score,
        repair_gate_score=repair_gate_score,
        repair_threshold=repair_threshold,
        repair_ready=repair_ready,
        repair_preview_available=repair_preview_available,
        repair_component_ok=repair_component_ok,
        source=source,
    )
    path_rows = payload.pop("path_rows", ())
    payload["path_summaries"] = tuple(
        ControllerPathSummary.from_path_row(row)
        for row in path_rows[: max(1, int(top_k_paths))]
    )
    return MacroControllerState(**payload)
