# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Parent-selection policies layered on top of the factorized symbolic search archive."""

from __future__ import annotations

import math
from typing import Any

from ..engine.archive import Rec
from ..expr_ast import node_str
from ..repair_policy import (
    _repair_controller_stagnation_state,
    _repair_controller_threshold,
    _repair_controller_weights,
    _repair_parent_retry_gate,
)


def choose_parent(arch, rng, exploit_frac, exploit_topk):
    if not arch.d:
        return None, None
    items = list(arch.items())
    if rng.random() < exploit_frac:
        items = sorted(items, key=lambda kv: kv[1].best_mse)[: min(exploit_topk, len(items))]
        return rng.choice(items)
    w = [1.0 / (kv[1].visits ** 0.5) for kv in items]
    s = sum(w)
    t = rng.random() * s
    acc = 0.0
    for (k, r), ww in zip(items, w):
        acc += ww
        if acc >= t:
            return k, r
    return items[-1]


def choose_parent_repair_aware(
    arch,
    rng,
    exploit_frac,
    exploit_topk,
    n_evaluated: int,
    repair_parent_cache: dict[Any, dict[str, Any]] | None,
    repair_parent_state: dict[Any, dict[str, Any]] | None,
    repair_controller_stats: dict[str, Any] | None,
):
    fallback = choose_parent(arch, rng, exploit_frac, exploit_topk)
    if (not arch.d) or (not isinstance(repair_controller_stats, dict)) or (not bool(repair_controller_stats.get("enabled", False))):
        return fallback

    repair_controller_stats["parent_considered"] = int(repair_controller_stats.get("parent_considered", 0)) + 1
    try:
        focus_prob = float(repair_controller_stats.get("focus_prob", 0.0))
    except Exception:
        focus_prob = 0.0
    focus_prob = 0.0 if focus_prob < 0.0 else (1.0 if focus_prob > 1.0 else focus_prob)
    if rng.random() >= focus_prob:
        repair_controller_stats["parent_fallback"] = int(repair_controller_stats.get("parent_fallback", 0)) + 1
        return fallback

    try:
        frontier_topk = max(1, int(repair_controller_stats.get("frontier_topk", 1)))
    except Exception:
        frontier_topk = 12
    try:
        stagnation_visits = max(0, int(repair_controller_stats.get("stagnation_visits", 0)))
    except Exception:
        stagnation_visits = 0
    threshold = _repair_controller_threshold(repair_controller_stats)

    items = list(arch.items())
    frontier: list[tuple[Any, Rec]] = []
    seen = set()
    for key, rec in sorted(items, key=lambda kv: kv[1].best_mse)[: min(frontier_topk, len(items))]:
        frontier.append((key, rec))
        seen.add(key)
    if stagnation_visits > 0:
        for key, rec in items:
            if key in seen:
                continue
            if int(getattr(rec, "visits_since_improve", getattr(rec, "visits", 0))) >= stagnation_visits:
                frontier.append((key, rec))
                seen.add(key)

    weighted = []
    weights_cfg = _repair_controller_weights(repair_controller_stats)
    for key, rec in frontier:
        cached = repair_parent_cache.get(key, None) if isinstance(repair_parent_cache, dict) else None
        if not isinstance(cached, dict):
            continue
        retry_ok, retry_reason = _repair_parent_retry_gate(
            key,
            rec,
            int(n_evaluated),
            repair_parent_state,
            repair_controller_stats,
        )
        if not retry_ok:
            repair_controller_stats[f"parent_retry_{retry_reason}"] = int(repair_controller_stats.get(f"parent_retry_{retry_reason}", 0)) + 1
            continue
        if str(cached.get("expr", "")) != node_str(rec.best_expr):
            continue
        try:
            cached_score = float(cached.get("score", 0.0))
            score_base = float(cached.get("score_base", cached_score))
            cached_gate_score = float(cached.get("gate_score", cached_score))
            gate_score_base = float(cached.get("gate_score_base", cached_gate_score))
            cached_threshold = float(cached.get("threshold", threshold))
        except Exception:
            continue
        if bool(cached.get("stagnation_adjustable", False)):
            stag_state = _repair_controller_stagnation_state(rec, repair_controller_stats)
            score = score_base + weights_cfg["stagnation"] * float(stag_state.get("stagnation_score", 0.0))
            gate_score = gate_score_base + weights_cfg["stagnation"] * float(stag_state.get("stagnation_score", 0.0))
        else:
            score = cached_score
            gate_score = cached_gate_score
        gate_threshold = cached_threshold if math.isfinite(cached_threshold) else threshold
        if (not math.isfinite(score)) or (not math.isfinite(gate_score)) or gate_score < gate_threshold:
            continue
        weight = max(1.0e-6, score - gate_threshold + 0.05) / max(1.0, float(getattr(rec, "visits", 1)) ** 0.5)
        weighted.append((key, rec, score, gate_score, float(weight)))

    if not weighted:
        repair_controller_stats["parent_fallback"] = int(repair_controller_stats.get("parent_fallback", 0)) + 1
        return fallback

    repair_controller_stats["parent_frontier_hits"] = int(repair_controller_stats.get("parent_frontier_hits", 0)) + 1
    cand_hist = repair_controller_stats.get("parent_frontier_candidate_hist", None)
    if isinstance(cand_hist, list):
        cand_hist.append(int(len(weighted)))
        if len(cand_hist) > 256:
            del cand_hist[:-256]
    total_w = sum(row[4] for row in weighted)
    t = rng.random() * max(total_w, 1.0e-12)
    acc = 0.0
    chosen = weighted[-1]
    for row in weighted:
        acc += row[4]
        if acc >= t:
            chosen = row
            break
    repair_controller_stats["parent_repair_selected"] = int(repair_controller_stats.get("parent_repair_selected", 0)) + 1
    return chosen[0], chosen[1]


__all__ = [
    "choose_parent",
    "choose_parent_repair_aware",
]
