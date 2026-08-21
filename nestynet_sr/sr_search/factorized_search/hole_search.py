# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Opportunity-driven hole search action for factorized symbolic search.

A_HOLESEARCH selects a promising hole opportunity from a cached frontier,
rebuilds the beam state, runs the inverse-spec solver with its own budgets,
cheaply reranks the top children, and returns the best one. The outer
search loop remains the single full exact scorer.

The frontier is populated from existing inverse path preview data produced
by A_INVSTEER and A_REPAIR actions. Each entry is a snapshot-bound spec
state keyed by the parent identity together with the path, target mode,
and recursive continuation metadata. Path-hole states rebuild beam tensors
lazily at execution time; follow-up local-problem states carry the minimal
solver payload needed to continue the recursive search cleanly.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch

from .expr_ast import collect_paths, eval_node, get_at, is_valid_node, node_depth, node_size, node_str
from .lift_route_evidence import build_local_lift_route_context
from .shared_opportunity import normalize_witness_energy_fields, shared_opportunity_row_dict


def _dim_key(dim: Any) -> tuple[str, ...]:
    if dim is None:
        return ()
    if isinstance(dim, (list, tuple)):
        out: list[str] = []
        for value in dim:
            try:
                out.append(f"{float(value):+.8f}")
            except Exception:
                out.append(str(value))
        return tuple(out)
    return (str(dim),)


def _tuple_key(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(f"{str(k)}={str(v)}" for k, v in sorted(dict(value).items(), key=lambda kv: str(kv[0])))
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    token = str(value)
    return () if not token else (token,)


def _stable_id(*parts: Any) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _preview_value_from_mse(mse: Any) -> float:
    vv = _finite_float(mse)
    if vv is None:
        return 0.0
    return 1.0 / (1.0 + max(0.0, float(vv)))


def _log_gain(parent_eff_mse: Any, child_eff_mse: Any, *, eps: float = 1.0e-30) -> float:
    parent_v = _finite_float(parent_eff_mse)
    child_v = _finite_float(child_eff_mse)
    if parent_v is None or child_v is None:
        return 0.0
    if parent_v <= 0.0 or child_v <= 0.0:
        return 0.0
    try:
        return float(math.log(parent_v + eps) - math.log(child_v + eps))
    except Exception:
        return 0.0


def _ema_update(old_value: Any, new_value: Any, *, alpha: float = 0.35, default: float = 0.0) -> float:
    old_v = _finite_float(old_value)
    new_v = _finite_float(new_value)
    if new_v is None:
        return float(default if old_v is None else old_v)
    if old_v is None:
        return float(new_v)
    aa = min(1.0, max(0.0, float(alpha)))
    return float(aa * new_v + (1.0 - aa) * old_v)


def _spec_cost_hint(*, preview_candidate_count: int = 0, preview_recursive_depth: int = 0, preview_periodic_fired: bool = False, attempts: int = 0) -> float:
    cost = 1.0
    cost += 0.02 * max(0, int(preview_candidate_count))
    cost += 0.35 * max(0, int(preview_recursive_depth))
    if bool(preview_periodic_fired):
        cost += 0.5
    cost += 0.15 * max(0, int(attempts))
    return float(max(0.25, cost))


def _spec_value_hint(*, parent_eff_mse_at_emit: Any = None, best_preview_probe_mse: Any = None, path_gain: Any = 0.0, confidence: Any = 0.0, valid_frac: Any = 0.0) -> float:
    gate = max(0.0, float(_finite_float(path_gain) or 0.0)) * max(0.0, float(_finite_float(confidence) or 0.0)) * max(0.0, float(_finite_float(valid_frac) or 0.0))
    preview = _preview_value_from_mse(best_preview_probe_mse)
    gain = _log_gain(parent_eff_mse_at_emit, best_preview_probe_mse)
    return float(max(gate, preview, gain, 0.0))


def _opportunity_witness_fields(row: Mapping[str, Any] | None) -> dict[str, Any]:
    return normalize_witness_energy_fields(row if isinstance(row, Mapping) else {})


def _lift_route_metric(route_context: Mapping[str, Any] | None, route_name: str) -> float:
    data = dict(route_context or {})
    route_key = "constant_lift" if str(route_name) == "constant_lift_route" else (
        "coordinate_lift" if str(route_name) == "coordinate_lift" else ""
    )
    if not route_key:
        return 0.0
    signal = dict(data.get(route_key, {}) or {})
    score = float(_finite_float(signal.get("score", None)) or 0.0)
    if bool(signal.get("preferred", False)):
        score += 1.0
    return float(score)


def _apply_lift_route_order(
    route_order: Sequence[str],
    *,
    route_context: Mapping[str, Any] | None,
) -> list[str]:
    order = [str(name) for name in list(route_order or ())]
    if "constant_lift_route" not in order or "coordinate_lift" not in order:
        return order
    constant_score = _lift_route_metric(route_context, "constant_lift_route")
    coordinate_score = _lift_route_metric(route_context, "coordinate_lift")
    if abs(float(constant_score) - float(coordinate_score)) < 0.15:
        return order
    preferred = "coordinate_lift" if float(coordinate_score) > float(constant_score) else "constant_lift_route"
    other = "constant_lift_route" if preferred == "coordinate_lift" else "coordinate_lift"
    idx_preferred = order.index(preferred)
    idx_other = order.index(other)
    if idx_preferred < idx_other:
        return order
    order.pop(idx_preferred)
    idx_other = order.index(other)
    order.insert(idx_other, preferred)
    return order

from .expr_mapping import eval_mapping, fit_exp_mapping, fit_pade, fit_poly, fit_power, fit_sine, mean_squared_error_same_shape
from nestynet_sr.sr_search.model_selection import mapping_cost
from .inverse_spec_solver import (
    solve_inverse_spec_preview_rows,
    solve_local_problem_spec_preview_rows,
)
from .constant_lift_solver import solve_local_constant_lift_preview_rows
from .coordinate_lift_solver import solve_local_coordinate_lift_preview_rows
from .local_sr_solver import solve_local_recursive_sr_preview_rows
from .soft_edit_search import solve_local_soft_edit_preview_rows
from .tangent_edit import solve_local_tangent_edit_preview_rows
from .solver_market import SolverMarketRouteCall, run_preview_solver_market
from .subproblem_spec import deserialize_subproblem_spec


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HoleOpportunity:
    """Lightweight record for a cached hole/spec opportunity."""

    parent_key: str
    parent_expr_str: str
    path: tuple[int, ...]
    target_mode: str
    beam_rank: int
    parent_elite_id: str = ""
    parent_snapshot_id: str = ""
    source: str = "inverse_slate"

    # First-class spec identity / continuation metadata
    spec_kind: str = "path_hole"
    direction: str = ""
    branch_id: str = ""
    continuation_key: tuple[str, ...] = field(default_factory=tuple)
    trace: tuple[str, ...] = field(default_factory=tuple)
    target_dim_key: tuple[str, ...] = field(default_factory=tuple)
    wrappers_left: int = 0
    recursion_level: int = 0

    # Preview features
    path_gain: float = 0.0
    confidence: float = 0.0
    valid_frac: float = 0.0
    transport_rel: float = 0.0
    effective_n: float = 0.0
    target_mapping_kind: str = ""
    candidate_count: int = 0
    best_preview_probe_mse: float | None = None
    parent_eff_mse_at_emit: float | None = None
    best_gate_score: float | None = None
    witness_value_loss: float | None = None
    witness_grad_loss: float | None = None
    witness_d2_loss: float | None = None
    witness_diag_loss: float | None = None
    witness_physics_loss: float | None = None
    witness_energy_total: float | None = None
    witness_energy_delta_estimate: float | None = None
    witness_fit_jet_source: str = ""
    witness_probe_jet_source: str = ""
    witness_fit_jet_requested_source: str = ""
    witness_probe_jet_requested_source: str = ""
    witness_fit_jet_fallback_used: bool = False
    witness_probe_jet_fallback_used: bool = False
    witness_numeric_jet_fallback_used: bool = False
    witness_exact_jet_used: bool = False

    # Solver preview / exact feedback
    preview_solvability: float | None = None
    preview_periodic_fired: bool = False
    preview_candidate_count: int = 0
    preview_recursive_depth: int = 0
    best_shortlist_eff_mse: float | None = None
    best_exact_eff_mse: float | None = None
    last_exact_eff_mse: float | None = None

    # Scheduler value/cost estimates
    predicted_value: float = 0.0
    predicted_cost: float = 1.0
    preview_budget: int = 0
    exact_budget: int = 0
    last_reward: float | None = None
    last_wall_seconds: float | None = None

    # Lifecycle
    created_at_iter: int = 0
    attempts: int = 0
    cooldown_until: int = 0
    best_child_eff_mse: float | None = None
    last_attempt_iter: int = 0
    spec_payload: Mapping[str, Any] | None = None

    @property
    def frontier_key(self) -> tuple[Any, ...]:
        parent_token = str(self.parent_snapshot_id or self.parent_key)
        return (
            parent_token,
            str(self.parent_elite_id),
            tuple(int(v) for v in self.path),
            str(self.target_mode),
            str(self.direction or ""),
            str(self.branch_id or ""),
            tuple(str(v) for v in (self.continuation_key or ())),
            tuple(str(v) for v in (self.trace or ())),
            int(self.wrappers_left),
            int(self.recursion_level),
            tuple(str(v) for v in (self.target_dim_key or ())),
        )

    @property
    def legacy_key(self) -> tuple[str, str, tuple[int, ...], str]:
        return (
            str(self.parent_key),
            str(self.parent_elite_id),
            tuple(int(v) for v in self.path),
            str(self.target_mode),
        )

    @property
    def has_strong_spec_identity(self) -> bool:
        return bool(
            str(self.parent_snapshot_id or "")
            or str(self.branch_id or "")
            or tuple(self.continuation_key or ())
            or tuple(self.trace or ())
            or tuple(self.target_dim_key or ())
            or str(self.direction or "")
            or int(self.wrappers_left or 0) != 0
            or int(self.recursion_level or 0) != 0
        )


SpecState = HoleOpportunity


@dataclass
class HoleFrontier:
    """Cache of hole opportunities populated from inverse path previews."""

    cooldown_iters: int = 32
    max_entries: int = 128
    staleness_window: int = 512

    _entries: dict[tuple[Any, ...], HoleOpportunity] = field(
        default_factory=dict, repr=False,
    )

    def _matching_entries(
        self,
        *,
        parent_key: str,
        parent_elite_id: str = "",
        path: Sequence[int] | tuple[int, ...] = (),
        target_mode: str = "",
    ) -> list[HoleOpportunity]:
        path_key = tuple(int(v) for v in (path or ()))
        residual_basin_key = str(parent_key or "")
        elite_key = str(parent_elite_id or "")
        mode_key = str(target_mode or "")
        return [
            opp
            for opp in self._entries.values()
            if str(opp.parent_key) == residual_basin_key
            and str(getattr(opp, "parent_elite_id", "") or "") == elite_key
            and tuple(int(v) for v in (opp.path or ())) == path_key
            and str(getattr(opp, "target_mode", "") or "") == mode_key
        ]

    def _resolve_entry(
        self,
        *,
        opportunity: HoleOpportunity | None = None,
        frontier_key: Sequence[Any] | tuple[Any, ...] | None = None,
        parent_key: str | None = None,
        parent_elite_id: str = "",
        path: Sequence[int] | tuple[int, ...] = (),
        target_mode: str = "",
        allow_legacy_fallback: bool = True,
    ) -> HoleOpportunity | None:
        key: tuple[Any, ...] | None = None
        if opportunity is not None:
            key = tuple(opportunity.frontier_key)
        elif frontier_key is not None:
            key = tuple(frontier_key)
        if key is not None:
            opp = self._entries.get(key)
            if opp is not None:
                return opp
            if opportunity is not None and getattr(opportunity, "has_strong_spec_identity", False):
                return None
            if not allow_legacy_fallback:
                return None
        if parent_key is None:
            if opportunity is None:
                return None
            parent_key = str(getattr(opportunity, "parent_key", "") or "")
            parent_elite_id = str(getattr(opportunity, "parent_elite_id", "") or "")
            path = tuple(int(v) for v in (getattr(opportunity, "path", ()) or ()))
            target_mode = str(getattr(opportunity, "target_mode", "") or "")
        matches = self._matching_entries(
            parent_key=str(parent_key or ""),
            parent_elite_id=str(parent_elite_id or ""),
            path=path,
            target_mode=str(target_mode or ""),
        )
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        strong_matches = [opp for opp in matches if not getattr(opp, "has_strong_spec_identity", False)]
        if len(strong_matches) == 1:
            return strong_matches[0]
        return max(matches, key=self._score)

    def _set_priority_hints(self, opp: HoleOpportunity) -> HoleOpportunity:
        gate_score = _finite_float(getattr(opp, "best_gate_score", None))
        if gate_score is None:
            gate_score = max(0.0, float(getattr(opp, "path_gain", 0.0) or 0.0)) * max(0.0, float(getattr(opp, "confidence", 0.0) or 0.0)) * max(0.0, float(getattr(opp, "valid_frac", 0.0) or 0.0))
        value_hint = max(
            float(_finite_float(getattr(opp, "predicted_value", None)) or 0.0),
            float(gate_score),
            float(_spec_value_hint(
                parent_eff_mse_at_emit=getattr(opp, "parent_eff_mse_at_emit", None),
                best_preview_probe_mse=getattr(opp, "preview_solvability", getattr(opp, "best_preview_probe_mse", None)),
                path_gain=getattr(opp, "path_gain", 0.0),
                confidence=getattr(opp, "confidence", 0.0),
                valid_frac=getattr(opp, "valid_frac", 0.0),
            )),
        )
        cost_hint = _finite_float(getattr(opp, "predicted_cost", None))
        if cost_hint is None or cost_hint <= 0.0:
            cost_hint = _spec_cost_hint(
                preview_candidate_count=int(getattr(opp, "preview_candidate_count", 0) or 0),
                preview_recursive_depth=int(getattr(opp, "preview_recursive_depth", 0) or 0),
                preview_periodic_fired=bool(getattr(opp, "preview_periodic_fired", False)),
                attempts=int(getattr(opp, "attempts", 0) or 0),
            )
        return replace(
            opp,
            best_gate_score=float(gate_score),
            predicted_value=float(value_hint),
            predicted_cost=float(max(0.25, cost_hint)),
        )

    def _refresh_priority_hints_inplace(self, opp: HoleOpportunity | None) -> HoleOpportunity | None:
        if opp is None:
            return None
        hinted = self._set_priority_hints(opp)
        opp.best_gate_score = hinted.best_gate_score
        opp.predicted_value = hinted.predicted_value
        opp.predicted_cost = hinted.predicted_cost
        return opp

    # ---- ingestion ----

    def ingest_opportunity_slate(
        self,
        parent_key: str,
        parent_expr_str: str,
        slate: Sequence[Mapping[str, Any]],
        current_iter: int,
        parent_elite_id: str = "",
        parent_snapshot_id: str = "",
        source: str = "inverse_slate",
    ) -> int:
        """Upsert opportunities from an inverse-action opportunity slate.

        Returns the number of entries added or updated.
        """
        count = 0
        for row in (slate or []):
            if not isinstance(row, Mapping):
                continue
            path_raw = row.get("path", None)
            if path_raw is None:
                continue
            try:
                path = tuple(int(v) for v in path_raw)
            except Exception:
                continue
            target_mode = str(row.get("target_mode", "") or "")
            direction = str(
                row.get(
                    "direction",
                    _spec_direction_from_payload(row.get("spec_payload", None)),
                )
                or ""
            )
            branch_id = str(row.get("branch_id", row.get("inverse_spec_branch_id", "")) or "")
            continuation_key = _tuple_key(row.get("continuation_key", row.get("inverse_spec_continuation_key", ())))
            trace = _tuple_key(row.get("trace", row.get("inverse_spec_trace", ())))
            target_dim_key = _dim_key(row.get("target_dim_key", row.get("target_dim", None)))
            wrappers_left = int(row.get("wrappers_left", row.get("inverse_spec_wrappers_left", 0)) or 0)
            recursion_level = int(row.get("recursion_level", row.get("inverse_spec_recursion_depth", 0)) or 0)
            snapshot_key = str(parent_snapshot_id or parent_key)
            key = (
                snapshot_key,
                str(parent_elite_id),
                path,
                target_mode,
                direction,
                branch_id,
                continuation_key,
                trace,
                wrappers_left,
                recursion_level,
                target_dim_key,
            )

            existing = self._entries.get(key)
            if existing is not None and existing.parent_expr_str != parent_expr_str:
                old_key = existing.frontier_key
                if old_key in self._entries:
                    del self._entries[old_key]
                existing = None

            path_gain = float(row.get("path_gain", 0.0) or 0.0)
            confidence = float(row.get("confidence", 0.0) or 0.0)
            valid_frac = float(row.get("valid_frac", 0.0) or 0.0)
            bpm = row.get("best_preview_probe_mse", None)
            try:
                preview_solvability = None if bpm is None else float(bpm)
            except Exception:
                preview_solvability = None
            try:
                preview_candidate_count = int(
                    row.get(
                        "preview_candidate_count_total",
                        row.get("candidate_count_observed", row.get("candidate_count_unique", 0)),
                    ) or 0
                )
            except Exception:
                preview_candidate_count = 0
            try:
                preview_recursive_depth = int(
                    row.get("inverse_spec_recursion_depth", row.get("preview_recursive_depth", 0)) or 0
                )
            except Exception:
                preview_recursive_depth = 0
            preview_periodic_fired = bool(row.get("preview_periodic_fired", False))
            if not preview_periodic_fired:
                try:
                    gen_kind = str(row.get("inverse_spec_generation_kind", "") or "")
                except Exception:
                    gen_kind = ""
                preview_periodic_fired = "periodic" in gen_kind.lower()

            # Admission gate: only cache entries with real signal
            if confidence < 0.05 or valid_frac < 0.10 or path_gain < 1e-6:
                continue

            gate_score = path_gain * confidence * valid_frac
            candidate_count = int(row.get("candidate_count_observed", row.get("candidate_count_unique", 0)) or 0)
            parent_eff_mse_at_emit = _finite_float(row.get("parent_eff_mse", row.get("estimated_parent_eff_mse", None)))
            witness_fields = _opportunity_witness_fields(row)
            new_opp = HoleOpportunity(
                parent_key=str(parent_key),
                parent_expr_str=str(parent_expr_str),
                path=path,
                target_mode=target_mode,
                beam_rank=int(row.get("beam_rank", getattr(existing, "beam_rank", 0)) or getattr(existing, "beam_rank", 0)),
                parent_elite_id=str(parent_elite_id),
                parent_snapshot_id=str(parent_snapshot_id or getattr(existing, "parent_snapshot_id", "") or ""),
                source=str(source or "inverse_slate"),
                spec_kind=str(row.get("spec_kind", getattr(existing, "spec_kind", "path_hole")) or getattr(existing, "spec_kind", "path_hole")),
                direction=str(direction or getattr(existing, "direction", "") or ""),
                branch_id=str(branch_id or getattr(existing, "branch_id", "") or ""),
                continuation_key=continuation_key or tuple(getattr(existing, "continuation_key", ()) or ()),
                trace=trace or tuple(getattr(existing, "trace", ()) or ()),
                target_dim_key=target_dim_key or tuple(getattr(existing, "target_dim_key", ()) or ()),
                wrappers_left=int(wrappers_left if wrappers_left or existing is None else getattr(existing, "wrappers_left", 0)),
                recursion_level=int(recursion_level if recursion_level or existing is None else getattr(existing, "recursion_level", 0)),
                path_gain=path_gain,
                confidence=confidence,
                valid_frac=valid_frac,
                transport_rel=float(row.get("transport_rel", getattr(existing, "transport_rel", 0.0)) or getattr(existing, "transport_rel", 0.0)),
                effective_n=float(row.get("effective_n", getattr(existing, "effective_n", 0.0)) or getattr(existing, "effective_n", 0.0)),
                target_mapping_kind=str(row.get("target_mapping_kind", getattr(existing, "target_mapping_kind", "")) or getattr(existing, "target_mapping_kind", "")),
                candidate_count=int(candidate_count),
                best_preview_probe_mse=None if bpm is None else float(bpm),
                parent_eff_mse_at_emit=parent_eff_mse_at_emit if parent_eff_mse_at_emit is not None else getattr(existing, "parent_eff_mse_at_emit", None),
                best_gate_score=float(gate_score),
                witness_value_loss=(
                    witness_fields.get("witness_value_loss", None)
                    if witness_fields.get("witness_value_loss", None) is not None
                    else getattr(existing, "witness_value_loss", None)
                ),
                witness_grad_loss=(
                    witness_fields.get("witness_grad_loss", None)
                    if witness_fields.get("witness_grad_loss", None) is not None
                    else getattr(existing, "witness_grad_loss", None)
                ),
                witness_d2_loss=(
                    witness_fields.get("witness_d2_loss", None)
                    if witness_fields.get("witness_d2_loss", None) is not None
                    else getattr(existing, "witness_d2_loss", None)
                ),
                witness_diag_loss=(
                    witness_fields.get("witness_diag_loss", None)
                    if witness_fields.get("witness_diag_loss", None) is not None
                    else getattr(existing, "witness_diag_loss", None)
                ),
                witness_physics_loss=(
                    witness_fields.get("witness_physics_loss", None)
                    if witness_fields.get("witness_physics_loss", None) is not None
                    else getattr(existing, "witness_physics_loss", None)
                ),
                witness_energy_total=(
                    witness_fields.get("witness_energy_total", None)
                    if witness_fields.get("witness_energy_total", None) is not None
                    else getattr(existing, "witness_energy_total", None)
                ),
                witness_energy_delta_estimate=(
                    witness_fields.get("witness_energy_delta_estimate", None)
                    if witness_fields.get("witness_energy_delta_estimate", None) is not None
                    else getattr(existing, "witness_energy_delta_estimate", None)
                ),
                witness_fit_jet_source=str(row.get("witness_fit_jet_source", getattr(existing, "witness_fit_jet_source", "")) or getattr(existing, "witness_fit_jet_source", "")),
                witness_probe_jet_source=str(row.get("witness_probe_jet_source", getattr(existing, "witness_probe_jet_source", "")) or getattr(existing, "witness_probe_jet_source", "")),
                witness_fit_jet_requested_source=str(row.get("witness_fit_jet_requested_source", getattr(existing, "witness_fit_jet_requested_source", "")) or getattr(existing, "witness_fit_jet_requested_source", "")),
                witness_probe_jet_requested_source=str(row.get("witness_probe_jet_requested_source", getattr(existing, "witness_probe_jet_requested_source", "")) or getattr(existing, "witness_probe_jet_requested_source", "")),
                witness_fit_jet_fallback_used=bool(row.get("witness_fit_jet_fallback_used", getattr(existing, "witness_fit_jet_fallback_used", False))),
                witness_probe_jet_fallback_used=bool(row.get("witness_probe_jet_fallback_used", getattr(existing, "witness_probe_jet_fallback_used", False))),
                witness_numeric_jet_fallback_used=bool(row.get("witness_numeric_jet_fallback_used", getattr(existing, "witness_numeric_jet_fallback_used", False))),
                witness_exact_jet_used=bool(row.get("witness_exact_jet_used", getattr(existing, "witness_exact_jet_used", False))),
                preview_solvability=preview_solvability,
                preview_candidate_count=int(preview_candidate_count),
                preview_recursive_depth=int(preview_recursive_depth),
                preview_periodic_fired=bool(preview_periodic_fired),
                predicted_value=float(getattr(existing, "predicted_value", 0.0) or 0.0),
                predicted_cost=float(getattr(existing, "predicted_cost", 1.0) or 1.0),
                preview_budget=int(max(0, getattr(existing, "preview_budget", 0) or 0)),
                exact_budget=int(getattr(existing, "exact_budget", 0) or 0),
                created_at_iter=int(current_iter),
                attempts=int(getattr(existing, "attempts", 0) or 0),
                cooldown_until=int(getattr(existing, "cooldown_until", 0) or 0),
                best_child_eff_mse=getattr(existing, "best_child_eff_mse", None),
                last_attempt_iter=int(getattr(existing, "last_attempt_iter", 0) or 0),
                best_shortlist_eff_mse=getattr(existing, "best_shortlist_eff_mse", None),
                best_exact_eff_mse=getattr(existing, "best_exact_eff_mse", None),
                last_exact_eff_mse=getattr(existing, "last_exact_eff_mse", None),
                last_reward=getattr(existing, "last_reward", None),
                last_wall_seconds=getattr(existing, "last_wall_seconds", None),
                spec_payload=row.get("spec_payload", getattr(existing, "spec_payload", None)),
            )
            new_opp = self._set_priority_hints(new_opp)
            self._entries[new_opp.frontier_key] = new_opp
            count += 1

        # Prune if over capacity
        if len(self._entries) > int(self.max_entries):
            self.prune(current_iter)
        return count

    # ---- selection ----

    def _score(self, opp: HoleOpportunity) -> float:
        """Score a first-class spec item by expected exact gain per unit cost."""
        gate_score = _finite_float(getattr(opp, "best_gate_score", None))
        if gate_score is None:
            gate_score = max(0.0, float(getattr(opp, "path_gain", 0.0) or 0.0)) * max(0.0, float(getattr(opp, "confidence", 0.0) or 0.0)) * max(0.0, float(getattr(opp, "valid_frac", 0.0) or 0.0))
        preview_value = _preview_value_from_mse(
            getattr(opp, "preview_solvability", getattr(opp, "best_preview_probe_mse", None))
        )
        exact_value = max(
            _log_gain(getattr(opp, "parent_eff_mse_at_emit", None), getattr(opp, "best_shortlist_eff_mse", None)),
            _log_gain(getattr(opp, "parent_eff_mse_at_emit", None), getattr(opp, "best_exact_eff_mse", None)),
        )
        value = max(
            float(gate_score),
            float(preview_value),
            float(exact_value),
            float(_finite_float(getattr(opp, "predicted_value", None)) or 0.0),
        )
        cost = _finite_float(getattr(opp, "predicted_cost", None))
        if cost is None or cost <= 0.0:
            cost = _spec_cost_hint(
                preview_candidate_count=int(getattr(opp, "preview_candidate_count", 0) or 0),
                preview_recursive_depth=int(getattr(opp, "preview_recursive_depth", 0) or 0),
                preview_periodic_fired=bool(getattr(opp, "preview_periodic_fired", False)),
                attempts=int(getattr(opp, "attempts", 0) or 0),
            )
        uncertainty = 1.0 / (1.0 + float(max(0, int(getattr(opp, "attempts", 0) or 0))))
        novelty = 1.0 / (1.0 + 0.5 * float(max(0, int(getattr(opp, "preview_recursive_depth", 0) or 0))) + 0.25 * float(max(0, int(getattr(opp, "wrappers_left", 0) or 0))))
        return float(value) / float(max(0.05, cost)) + 0.10 * float(uncertainty) + 0.05 * float(novelty)

    def select(self, current_iter: int, rng) -> HoleOpportunity | None:
        """Select the best eligible opportunity."""
        eligible = [
            opp for opp in self._entries.values()
            if int(opp.cooldown_until) <= int(current_iter)
        ]
        if not eligible:
            return None
        eligible.sort(key=self._score, reverse=True)
        # Small amount of randomness: pick from top-3
        top = eligible[:min(3, len(eligible))]
        return top[rng.randint(0, len(top) - 1)]

    def select_executable(self, current_iter: int, rng, is_executable_fn) -> HoleOpportunity | None:
        """Select the best eligible opportunity whose parent is executable."""
        eligible = []
        for opp in self._entries.values():
            if int(opp.cooldown_until) > int(current_iter):
                continue
            try:
                if not bool(is_executable_fn(opp)):
                    continue
            except Exception:
                continue
            eligible.append(opp)
        if not eligible:
            return None
        eligible.sort(key=self._score, reverse=True)
        top = eligible[:min(3, len(eligible))]
        return top[rng.randint(0, len(top) - 1)]

    def select_n_executable(
        self,
        current_iter: int,
        rng,
        is_executable_fn,
        n: int = 8,
    ) -> list[HoleOpportunity]:
        """Select the top-*n* eligible opportunities whose parents are executable."""
        eligible = []
        for opp in self._entries.values():
            if int(opp.cooldown_until) > int(current_iter):
                continue
            try:
                if not bool(is_executable_fn(opp)):
                    continue
            except Exception:
                continue
            eligible.append(opp)
        if not eligible:
            return []
        eligible.sort(key=self._score, reverse=True)
        return eligible[:max(1, int(n))]

    def enqueue_spec_state(
        self,
        opp: HoleOpportunity,
        *,
        current_iter: int,
        preserve_existing_lifecycle: bool = True,
    ) -> bool:
        return self.upsert(
            opp,
            current_iter=int(current_iter),
            preserve_existing_lifecycle=bool(preserve_existing_lifecycle),
        )

    def select_spec_state(
        self,
        current_iter: int,
        rng,
        is_executable_fn=None,
    ) -> HoleOpportunity | None:
        if callable(is_executable_fn):
            return self.select_executable(int(current_iter), rng, is_executable_fn)
        return self.select(int(current_iter), rng)

    def select_n_spec_states(
        self,
        current_iter: int,
        rng,
        is_executable_fn=None,
        n: int = 8,
    ) -> list[HoleOpportunity]:
        if callable(is_executable_fn):
            return self.select_n_executable(
                int(current_iter),
                rng,
                is_executable_fn,
                n=int(n),
            )
        eligible = self.eligible(int(current_iter))
        eligible.sort(key=self._score, reverse=True)
        return eligible[: max(1, int(n))]

    def drop_spec_state(
        self,
        *,
        opportunity: HoleOpportunity | None = None,
        frontier_key: Sequence[Any] | tuple[Any, ...] | None = None,
    ) -> HoleOpportunity | None:
        if opportunity is not None:
            frontier_key = opportunity.frontier_key
        if frontier_key is None:
            return None
        return self.pop(frontier_key)

    def record_preview_result(
        self,
        opportunity: HoleOpportunity | None,
        *,
        preview_probe_mse: float | None = None,
        preview_candidate_count: int | None = None,
        preview_recursive_depth: int | None = None,
        preview_periodic_fired: bool | None = None,
    ) -> None:
        opp = self._resolve_entry(
            opportunity=opportunity,
            allow_legacy_fallback=(not bool(getattr(opportunity, "has_strong_spec_identity", False))) if opportunity is not None else True,
        )
        if opp is None:
            return
        preview_v = _finite_float(preview_probe_mse)
        if preview_v is not None:
            if opp.preview_solvability is None or float(preview_v) < float(opp.preview_solvability):
                opp.preview_solvability = float(preview_v)
            if opp.best_preview_probe_mse is None or float(preview_v) < float(opp.best_preview_probe_mse):
                opp.best_preview_probe_mse = float(preview_v)
        if preview_candidate_count is not None:
            opp.preview_candidate_count = max(
                int(getattr(opp, "preview_candidate_count", 0) or 0),
                int(preview_candidate_count),
            )
        if preview_recursive_depth is not None:
            opp.preview_recursive_depth = max(
                int(getattr(opp, "preview_recursive_depth", 0) or 0),
                int(preview_recursive_depth),
            )
        if preview_periodic_fired is not None:
            opp.preview_periodic_fired = bool(getattr(opp, "preview_periodic_fired", False) or bool(preview_periodic_fired))
        self._refresh_priority_hints_inplace(opp)

    def record_spec_attempt(
        self,
        *,
        opportunity: HoleOpportunity | None = None,
        frontier_key: Sequence[Any] | tuple[Any, ...] | None = None,
        current_iter: int,
        child_eff_mse: float | None = None,
        parent_key: str | None = None,
        path: tuple[int, ...] = (),
        target_mode: str = "",
        parent_elite_id: str = "",
    ) -> None:
        self.record_attempt(
            parent_key=parent_key,
            path=path,
            target_mode=target_mode,
            current_iter=int(current_iter),
            child_eff_mse=child_eff_mse,
            parent_elite_id=parent_elite_id,
            opportunity=opportunity,
            frontier_key=frontier_key,
        )

    def record_spec_outcome(
        self,
        opportunity: HoleOpportunity | None,
        *,
        current_iter: int,
        exact_eff_mse: float | None = None,
        shortlist_eff_mse: float | None = None,
        reward: float | None = None,
        wall_s: float | None = None,
        parent_eff_mse: float | None = None,
        accepted: bool | None = None,
        status: str = "ok",
    ) -> None:
        self.record_exact_outcome(
            opportunity,
            current_iter=int(current_iter),
            exact_eff_mse=exact_eff_mse,
            shortlist_eff_mse=shortlist_eff_mse,
            reward=reward,
            wall_s=wall_s,
            parent_eff_mse=parent_eff_mse,
            accepted=accepted,
            status=str(status or "ok"),
        )

    def record_attempt(
        self,
        parent_key: str | None = None,
        path: tuple[int, ...] = (),
        target_mode: str = "",
        current_iter: int = 0,
        child_eff_mse: float | None = None,
        parent_elite_id: str = "",
        opportunity: HoleOpportunity | None = None,
        frontier_key: Sequence[Any] | tuple[Any, ...] | None = None,
    ) -> None:
        """Record a spec execution attempt and apply cooldown."""
        opp = self._resolve_entry(
            opportunity=opportunity,
            frontier_key=frontier_key,
            parent_key=parent_key,
            parent_elite_id=str(parent_elite_id or ""),
            path=path,
            target_mode=str(target_mode or ""),
        )
        if opp is None:
            return
        opp.attempts = int(getattr(opp, "attempts", 0) or 0) + 1
        opp.last_attempt_iter = int(current_iter)
        backoff = int(self.cooldown_iters) * (1 + int(opp.attempts))
        opp.cooldown_until = int(current_iter) + backoff
        shortlist_mse = _finite_float(child_eff_mse)
        if shortlist_mse is not None:
            if opp.best_child_eff_mse is None or float(shortlist_mse) < float(opp.best_child_eff_mse):
                opp.best_child_eff_mse = float(shortlist_mse)
            if opp.best_shortlist_eff_mse is None or float(shortlist_mse) < float(opp.best_shortlist_eff_mse):
                opp.best_shortlist_eff_mse = float(shortlist_mse)
        opp.predicted_cost = _ema_update(
            getattr(opp, "predicted_cost", None),
            _spec_cost_hint(
                preview_candidate_count=int(getattr(opp, "preview_candidate_count", 0) or 0),
                preview_recursive_depth=int(getattr(opp, "preview_recursive_depth", 0) or 0),
                preview_periodic_fired=bool(getattr(opp, "preview_periodic_fired", False)),
                attempts=int(getattr(opp, "attempts", 0) or 0),
            ),
            alpha=0.20,
            default=1.0,
        )
        self._refresh_priority_hints_inplace(opp)

    def record_exact_outcome(
        self,
        opportunity: HoleOpportunity | None,
        *,
        current_iter: int,
        exact_eff_mse: float | None = None,
        shortlist_eff_mse: float | None = None,
        reward: float | None = None,
        wall_s: float | None = None,
        parent_eff_mse: float | None = None,
        accepted: bool | None = None,
        status: str = "ok",
    ) -> None:
        if opportunity is None:
            return
        opp = self._resolve_entry(
            opportunity=opportunity,
            allow_legacy_fallback=(not bool(getattr(opportunity, "has_strong_spec_identity", False))),
        )
        if opp is None:
            return
        opp.last_attempt_iter = int(current_iter)
        shortlist_v = _finite_float(shortlist_eff_mse)
        if shortlist_v is not None and (opp.best_shortlist_eff_mse is None or shortlist_v < float(opp.best_shortlist_eff_mse)):
            opp.best_shortlist_eff_mse = float(shortlist_v)
        exact_v = _finite_float(exact_eff_mse)
        if exact_v is not None:
            opp.last_exact_eff_mse = float(exact_v)
            if opp.best_exact_eff_mse is None or float(exact_v) < float(opp.best_exact_eff_mse):
                opp.best_exact_eff_mse = float(exact_v)
            if opp.best_child_eff_mse is None or float(exact_v) < float(opp.best_child_eff_mse):
                opp.best_child_eff_mse = float(exact_v)
        if _finite_float(parent_eff_mse) is not None:
            opp.parent_eff_mse_at_emit = float(parent_eff_mse)
        raw_reward = _finite_float(reward)
        if raw_reward is None:
            raw_reward = _log_gain(parent_eff_mse if parent_eff_mse is not None else getattr(opp, "parent_eff_mse_at_emit", None), exact_eff_mse if exact_eff_mse is not None else shortlist_eff_mse)
        wall_v = _finite_float(wall_s)
        if wall_v is not None:
            opp.last_wall_seconds = float(wall_v)
            opp.predicted_cost = _ema_update(getattr(opp, "predicted_cost", None), max(0.05, float(wall_v)), alpha=0.30, default=1.0)
        if raw_reward is not None:
            opp.last_reward = float(raw_reward)
            opp.predicted_value = _ema_update(getattr(opp, "predicted_value", None), float(raw_reward), alpha=0.30, default=0.0)
        if accepted is True and (raw_reward or 0.0) > 0.0:
            opp.cooldown_until = int(current_iter) + max(1, int(self.cooldown_iters) // 2)
        elif str(status or "") not in ("ok", "scored", "accepted") or accepted is False:
            opp.cooldown_until = max(int(opp.cooldown_until), int(current_iter) + int(self.cooldown_iters))
        self._refresh_priority_hints_inplace(opp)

    def eligible(self, current_iter: int) -> list[HoleOpportunity]:
        return [
            opp for opp in self._entries.values()
            if int(opp.cooldown_until) <= int(current_iter)
        ]

    def pop(self, frontier_key: Sequence[Any] | tuple[Any, ...]) -> HoleOpportunity | None:
        return self._entries.pop(frontier_key, None)

    def upsert(self, opp: HoleOpportunity, *, current_iter: int, preserve_existing_lifecycle: bool = True) -> bool:
        key = opp.frontier_key
        existing = self._entries.get(key)
        if existing is not None and existing.parent_expr_str != opp.parent_expr_str:
            del self._entries[key]
            existing = None
        if existing is not None and preserve_existing_lifecycle:
            merged = replace(
                opp,
                attempts=int(existing.attempts),
                cooldown_until=int(existing.cooldown_until),
                best_child_eff_mse=existing.best_child_eff_mse,
                last_attempt_iter=int(existing.last_attempt_iter),
                best_shortlist_eff_mse=getattr(existing, "best_shortlist_eff_mse", None),
                best_exact_eff_mse=getattr(existing, "best_exact_eff_mse", None),
                last_exact_eff_mse=getattr(existing, "last_exact_eff_mse", None),
                predicted_value=float(getattr(existing, "predicted_value", getattr(opp, "predicted_value", 0.0)) or getattr(opp, "predicted_value", 0.0)),
                predicted_cost=float(getattr(existing, "predicted_cost", getattr(opp, "predicted_cost", 1.0)) or getattr(opp, "predicted_cost", 1.0)),
                last_reward=getattr(existing, "last_reward", None),
                last_wall_seconds=getattr(existing, "last_wall_seconds", None),
                witness_value_loss=(
                    getattr(opp, "witness_value_loss", None)
                    if getattr(opp, "witness_value_loss", None) is not None
                    else getattr(existing, "witness_value_loss", None)
                ),
                witness_grad_loss=(
                    getattr(opp, "witness_grad_loss", None)
                    if getattr(opp, "witness_grad_loss", None) is not None
                    else getattr(existing, "witness_grad_loss", None)
                ),
                witness_d2_loss=(
                    getattr(opp, "witness_d2_loss", None)
                    if getattr(opp, "witness_d2_loss", None) is not None
                    else getattr(existing, "witness_d2_loss", None)
                ),
                witness_diag_loss=(
                    getattr(opp, "witness_diag_loss", None)
                    if getattr(opp, "witness_diag_loss", None) is not None
                    else getattr(existing, "witness_diag_loss", None)
                ),
                witness_physics_loss=(
                    getattr(opp, "witness_physics_loss", None)
                    if getattr(opp, "witness_physics_loss", None) is not None
                    else getattr(existing, "witness_physics_loss", None)
                ),
                witness_energy_total=(
                    getattr(opp, "witness_energy_total", None)
                    if getattr(opp, "witness_energy_total", None) is not None
                    else getattr(existing, "witness_energy_total", None)
                ),
                witness_energy_delta_estimate=(
                    getattr(opp, "witness_energy_delta_estimate", None)
                    if getattr(opp, "witness_energy_delta_estimate", None) is not None
                    else getattr(existing, "witness_energy_delta_estimate", None)
                ),
                spec_payload=getattr(existing, "spec_payload", getattr(opp, "spec_payload", None)),
                created_at_iter=int(current_iter),
            )
            self._entries[key] = self._set_priority_hints(merged)
        else:
            self._entries[key] = self._set_priority_hints(replace(opp, created_at_iter=int(current_iter)))
        if len(self._entries) > int(self.max_entries):
            self.prune(current_iter)
        return True

    # ---- maintenance ----

    def invalidate_parent(self, parent_key: str, *, parent_elite_id: str | None = None) -> int:
        """Remove all entries for a residual_basin, or for a specific elite within it."""
        residual_basin_key = str(parent_key)
        elite_key = None if parent_elite_id is None else str(parent_elite_id)
        to_remove = []
        for k, opp in self._entries.items():
            if opp.parent_key != residual_basin_key:
                continue
            if elite_key is not None and str(opp.parent_elite_id) != elite_key:
                continue
            to_remove.append(k)
        for k in to_remove:
            del self._entries[k]
        return len(to_remove)

    def prune(self, current_iter: int) -> int:
        """Remove stale entries and trim to max_entries."""
        to_remove = []
        for k, opp in self._entries.items():
            age = int(current_iter) - int(opp.created_at_iter)
            if age > int(self.staleness_window):
                to_remove.append(k)
        for k in to_remove:
            del self._entries[k]

        # If still over capacity, drop lowest-scored entries
        if len(self._entries) > int(self.max_entries):
            scored = sorted(self._entries.items(), key=lambda item: self._score(item[1]), reverse=True)
            keep = {k for k, _ in scored[:int(self.max_entries)]}
            to_drop = [k for k in self._entries if k not in keep]
            for k in to_drop:
                del self._entries[k]
            to_remove.extend(to_drop)
        return len(to_remove)

    def nonempty(self, current_iter: int) -> bool:
        """True if at least one eligible (non-cooldown) entry exists."""
        return any(
            int(opp.cooldown_until) <= int(current_iter)
            for opp in self._entries.values()
        )

    def __len__(self) -> int:
        return len(self._entries)

    def active_snapshot_ids(self) -> set[str]:
        return {
            str(opp.parent_snapshot_id)
            for opp in self._entries.values()
            if str(getattr(opp, "parent_snapshot_id", "") or "")
        }


SpecAgenda = HoleFrontier


def export_hole_opportunity_rows(
    frontier_or_rows: HoleFrontier | Sequence[HoleOpportunity] | None,
    *,
    current_iter: int = 0,
    eligible_only: bool = True,
    max_rows: int | None = None,
    decision_id: str = "",
    decision_context_id: str = "",
) -> list[dict[str, Any]]:
    if isinstance(frontier_or_rows, HoleFrontier):
        if bool(eligible_only):
            opportunities = list(frontier_or_rows.eligible(int(current_iter)))
        else:
            opportunities = list(frontier_or_rows._entries.values())
    else:
        opportunities = [opp for opp in list(frontier_or_rows or []) if isinstance(opp, HoleOpportunity)]
    opportunities = sorted(
        opportunities,
        key=lambda opp: (
            -float(getattr(opp, "predicted_value", 0.0) or 0.0),
            float(getattr(opp, "predicted_cost", 1.0) or 1.0),
            str(getattr(opp, "parent_key", "") or ""),
            tuple(int(v) for v in (getattr(opp, "path", ()) or ())),
        ),
    )
    if max_rows is not None:
        opportunities = opportunities[: max(0, int(max_rows))]
    decision_id_out = str(decision_id or f"hole_frontier:{int(current_iter)}")
    decision_context_out = str(decision_context_id or decision_id_out)
    rows: list[dict[str, Any]] = []
    for ordinal, opp in enumerate(opportunities):
        opportunity_id = f"holeopp_{_stable_id(*tuple(opp.frontier_key))}"
        best_exact_eff = _finite_float(getattr(opp, "best_exact_eff_mse", None))
        best_shortlist_eff = _finite_float(getattr(opp, "best_shortlist_eff_mse", None))
        current_best_eff = best_exact_eff if best_exact_eff is not None else best_shortlist_eff
        evidence_level = "preview_only"
        if best_exact_eff is not None:
            evidence_level = "exact_known"
        elif int(getattr(opp, "preview_candidate_count", 0) or 0) > 1:
            evidence_level = "preview_support"
        elif best_shortlist_eff is not None:
            evidence_level = "preview_support"
        rows.append(shared_opportunity_row_dict({
            "route_source": "hole",
            "opportunity_type": "hole_opportunity",
            "opportunity_id": str(opportunity_id),
            "decision_id": str(decision_id_out),
            "decision_context_id": str(decision_context_out),
            "beam_id": f"{decision_id_out}:{int(ordinal)}",
            "parent_key": str(getattr(opp, "parent_key", "") or ""),
            "parent_expr": str(getattr(opp, "parent_expr_str", "") or ""),
            "action": "hole_search",
            "path": [int(v) for v in tuple(getattr(opp, "path", ()) or ())],
            "path_source": "hole_frontier",
            "target_mode": str(getattr(opp, "target_mode", "") or ""),
            "method_name": str(getattr(opp, "source", "hole_frontier") or "hole_frontier"),
            "subroute": str(getattr(opp, "spec_kind", "path_hole") or "path_hole"),
            "evidence_level": str(evidence_level),
            "budget_exact_spent": int(max(0, getattr(opp, "exact_budget", 0) or 0)),
            "budget_remaining": int(max(0, getattr(opp, "preview_candidate_count", 0) or 0)),
            "budget_widen_spent": 0,
            "budget_micro_spent": 0,
            "candidate_count_observed": 1 if best_exact_eff is not None else 0,
            "candidate_count_unique": 1 if best_exact_eff is not None else 0,
            "current_best_child_expr": "",
            "current_best_child_eff_mse": current_best_eff,
            "hole_best_exact_eff_mse": best_exact_eff,
            "hole_best_shortlist_eff_mse": best_shortlist_eff,
            "hole_last_exact_eff_mse": _finite_float(getattr(opp, "last_exact_eff_mse", None)),
            "hole_last_reward": _finite_float(getattr(opp, "last_reward", None)),
            "hole_attempts": int(max(0, getattr(opp, "attempts", 0) or 0)),
            "parent_eff_mse": _finite_float(getattr(opp, "parent_eff_mse_at_emit", None)),
            "best_preview_probe_mse": _finite_float(getattr(opp, "preview_solvability", getattr(opp, "best_preview_probe_mse", None))),
            "preview_solvability": _finite_float(getattr(opp, "preview_solvability", None)),
            "preview_candidate_count_total": int(max(0, getattr(opp, "preview_candidate_count", 0) or 0)),
            "preview_candidate_count_unique_total": int(max(0, getattr(opp, "preview_candidate_count", 0) or 0)),
            "preview_recursive_depth": int(max(0, getattr(opp, "preview_recursive_depth", 0) or 0)),
            "preview_periodic_fired": bool(getattr(opp, "preview_periodic_fired", False)),
            "path_gain": float(getattr(opp, "path_gain", 0.0) or 0.0),
            "confidence": float(getattr(opp, "confidence", 0.0) or 0.0),
            "valid_frac": float(getattr(opp, "valid_frac", 0.0) or 0.0),
            "transport_rel": float(getattr(opp, "transport_rel", 0.0) or 0.0),
            "effective_n": float(getattr(opp, "effective_n", 0.0) or 0.0),
            "target_mapping_kind": str(getattr(opp, "target_mapping_kind", "") or ""),
            "predicted_value": float(getattr(opp, "predicted_value", 0.0) or 0.0),
            "predicted_cost": float(getattr(opp, "predicted_cost", 1.0) or 1.0),
            "cost_estimate": float(getattr(opp, "predicted_cost", 1.0) or 1.0),
            "spec_kind": str(getattr(opp, "spec_kind", "path_hole") or "path_hole"),
            "direction": str(_opportunity_direction(opp) or ""),
            "branch_id": str(getattr(opp, "branch_id", "") or ""),
            "continuation_key": list(getattr(opp, "continuation_key", ()) or ()),
            "trace": list(getattr(opp, "trace", ()) or ()),
            "wrappers_left": int(max(0, getattr(opp, "wrappers_left", 0) or 0)),
            "recursion_level": int(max(0, getattr(opp, "recursion_level", 0) or 0)),
            "observed_wall_seconds": _finite_float(getattr(opp, "last_wall_seconds", None)),
            "observed_exact_evals": 1 if best_exact_eff is not None else 0,
            "observed_preview_evals": int(max(0, getattr(opp, "preview_candidate_count", 0) or 0)),
            "observed_micro_tokens": 0,
            "observed_widen_tokens": 0,
            "witness_value_loss": _finite_float(getattr(opp, "witness_value_loss", None)),
            "witness_grad_loss": _finite_float(getattr(opp, "witness_grad_loss", None)),
            "witness_d2_loss": _finite_float(getattr(opp, "witness_d2_loss", None)),
            "witness_diag_loss": _finite_float(getattr(opp, "witness_diag_loss", None)),
            "witness_physics_loss": _finite_float(getattr(opp, "witness_physics_loss", None)),
            "witness_energy_total": _finite_float(getattr(opp, "witness_energy_total", None)),
            "witness_energy_delta_estimate": _finite_float(getattr(opp, "witness_energy_delta_estimate", None)),
            "witness_fit_jet_source": str(getattr(opp, "witness_fit_jet_source", "") or ""),
            "witness_probe_jet_source": str(getattr(opp, "witness_probe_jet_source", "") or ""),
            "witness_fit_jet_requested_source": str(getattr(opp, "witness_fit_jet_requested_source", "") or ""),
            "witness_probe_jet_requested_source": str(getattr(opp, "witness_probe_jet_requested_source", "") or ""),
            "witness_fit_jet_fallback_used": bool(getattr(opp, "witness_fit_jet_fallback_used", False)),
            "witness_probe_jet_fallback_used": bool(getattr(opp, "witness_probe_jet_fallback_used", False)),
            "witness_numeric_jet_fallback_used": bool(getattr(opp, "witness_numeric_jet_fallback_used", False)),
            "witness_exact_jet_used": bool(getattr(opp, "witness_exact_jet_used", False)),
        }, route_source="hole"))
    return rows


def _build_path_hole_beam_state(
    opportunity: HoleOpportunity,
    *,
    parent_node,
    parent_mapping,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    safe_eps: float,
    beam_cfg: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str]:
    from .inverse_action import (
        _estimate_inverse_action_transport,
        _inverse_action_path_mode_beam_states,
    )

    path = tuple(int(v) for v in tuple(getattr(opportunity, "path", ()) or ()))
    try:
        transport_ctx = _estimate_inverse_action_transport(
            parent_node,
            parent_mapping,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            [path],
            safe_eps=float(safe_eps),
        )
    except Exception:
        return None, "transport_failed"
    try:
        beam_states = _inverse_action_path_mode_beam_states(
            parent_node=parent_node,
            parent_mapping=parent_mapping,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            pool_nodes=pool_nodes,
            pool_phi_fit=pool_phi_fit,
            pool_phi_probe=pool_phi_probe,
            pool_dims=pool_dims,
            all_paths=[path],
            path_target_modes={path: str(opportunity.target_mode)} if opportunity.target_mode else None,
            transport_ctx=transport_ctx,
            cfg=dict(beam_cfg or {}),
            beam_width=1,
        )
    except Exception:
        return None, "beam_state_failed"
    if not beam_states:
        return None, "no_beam_state"
    return dict(beam_states[0]), "ok"


def _build_spec_preview_route_calls(
    opportunity: HoleOpportunity,
    *,
    parent_node,
    parent_mapping,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    max_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims,
    enum_max_depth: int,
    enum_max_trees: int,
    preview_topk: int,
    max_subtree_depth: int | None,
    complexity_penalty: float,
    family_battery_enable: bool,
    recursive_enable: bool,
    recursive_max_depth: int,
    recursive_trigger_rel_mse: float,
    recursive_seed_cap: int,
    recursive_branch_topk: int,
    recursive_child_topk: int,
    recursive_sr_enable: bool,
    recursive_sr_preview_topk: int,
    recursive_sr_exact_budget: int,
    constant_lift_route_enable: bool = False,
    constant_lift_route_topk: int = 2,
    coordinate_lift_enable: bool,
    coordinate_lift_topk: int,
    coordinate_lift_mode: str,
    tangent_edit_enable: bool,
    tangent_edit_topk: int,
    soft_edit_enable: bool,
    soft_edit_steps: int,
    soft_edit_l1: float,
    witness_jets_enable: bool,
    witness_d2_enable: bool,
    witness_max_rows: int,
    witness_loss_enable: bool = False,
    witness_grad_weight: float = 1.0,
    witness_d2_weight: float = 0.0,
    witness_diag_weight: float = 0.0,
    witness_physics_weight: float = 0.0,
    active_var_screen_enable: bool,
    active_var_grad_tol: float,
    active_var_max_count: int,
    directional_market_enable: bool,
    safe_eps: float,
    confidence_mode: str,
    confidence_target_gain: float,
    confidence_floor: float,
    branch_beam_width: int,
    beam_cfg: Mapping[str, Any],
    slate_prefix: str,
    family_battery_mode: str = "outer",
    regime_metadata: Mapping[str, Any] | None = None,
) -> tuple[list[SolverMarketRouteCall], str]:
    spec_kind = str(getattr(opportunity, "spec_kind", "path_hole") or "path_hole")
    direction = str(_opportunity_direction(opportunity) or "")
    path = tuple(int(v) for v in tuple(getattr(opportunity, "path", ()) or ()))
    preview_limit = max(1, int(preview_topk))
    if spec_kind == "path_hole":
        beam_state, status = _build_path_hole_beam_state(
            opportunity,
            parent_node=parent_node,
            parent_mapping=parent_mapping,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            pool_nodes=pool_nodes,
            pool_phi_fit=pool_phi_fit,
            pool_phi_probe=pool_phi_probe,
            pool_dims=pool_dims,
            safe_eps=float(safe_eps),
            beam_cfg=beam_cfg,
        )
        if beam_state is None:
            return [], str(status)
        return [
            SolverMarketRouteCall(
                route_name="inverse_spec_path",
                method_name="inverse_spec_solver",
                subroute="path_hole",
                runner=solve_inverse_spec_preview_rows,
                runner_kwargs={
                    "parent_node": parent_node,
                    "beam_state": beam_state,
                    "regime_metadata": dict(regime_metadata or {}),
                    "beam_rank": int(opportunity.beam_rank),
                    "slate_id": f"{str(slate_prefix)}:{opportunity.parent_key}:{'/'.join(str(v) for v in path)}",
                    "max_depth": int(max_depth),
                    "nvars": int(nvars),
                    "poly_degree": int(poly_degree),
                    "var_dims": var_dims,
                    "pool_nodes": pool_nodes,
                    "pool_dims": pool_dims,
                    "enum_max_depth": int(enum_max_depth),
                    "enum_max_trees": int(enum_max_trees),
                    "max_subtree_depth": max_subtree_depth,
                    "preview_topk": int(preview_limit),
                    "complexity_penalty": float(complexity_penalty),
                    "family_battery_enable": bool(family_battery_enable),
                    "family_battery_mode": str(family_battery_mode or "outer"),
                    "recursive_enable": bool(recursive_enable),
                    "recursive_max_depth": int(recursive_max_depth),
                    "recursive_trigger_rel_mse": float(recursive_trigger_rel_mse),
                    "recursive_seed_cap": int(recursive_seed_cap),
                    "recursive_branch_topk": int(recursive_branch_topk),
                    "recursive_child_topk": int(recursive_child_topk),
                    "witness_jets_enable": bool(witness_jets_enable),
                    "witness_d2_enable": bool(witness_d2_enable),
                    "witness_max_rows": int(max(4, int(witness_max_rows))),
                    "witness_loss_enable": bool(witness_loss_enable),
                    "witness_grad_weight": float(witness_grad_weight),
                    "witness_d2_weight": float(witness_d2_weight),
                    "witness_diag_weight": float(witness_diag_weight),
                    "witness_physics_weight": float(witness_physics_weight),
                    "active_var_screen_enable": bool(active_var_screen_enable),
                    "active_var_grad_tol": float(active_var_grad_tol),
                    "active_var_max_count": max(1, int(active_var_max_count)),
                    "safe_eps": float(safe_eps),
                    "confidence_mode": str(confidence_mode),
                    "confidence_target_gain": float(confidence_target_gain),
                    "confidence_floor": float(confidence_floor),
                    "branch_beam_width": int(branch_beam_width),
                },
            ),
        ], "ok"
    if spec_kind == "local_problem":
        lift_route_context = build_local_lift_route_context(getattr(opportunity, "spec_payload", None))
        route_calls_by_name: dict[str, SolverMarketRouteCall] = {}
        route_calls_by_name["inverse_spec_followup"] = SolverMarketRouteCall(
            route_name="inverse_spec_followup",
            method_name="inverse_spec_solver",
            subroute="local_problem",
            runner=solve_local_problem_spec_preview_rows,
            runner_kwargs={
                "parent_node": parent_node,
                "spec_payload": dict(getattr(opportunity, "spec_payload", {}) or {}),
                "path": path,
                "target_mode": str(opportunity.target_mode),
                "target_mapping_kind": str(getattr(opportunity, "target_mapping_kind", "") or ""),
                "beam_rank": int(opportunity.beam_rank),
                "slate_id": (
                    f"{str(slate_prefix)}:{opportunity.parent_key}:{'/'.join(str(v) for v in path)}:"
                    f"{str(getattr(opportunity, 'branch_id', '') or '')}"
                ),
                "path_gain": float(getattr(opportunity, "path_gain", 0.0) or 0.0),
                "max_depth": int(max_depth),
                "nvars": int(nvars),
                "poly_degree": int(poly_degree),
                "var_dims": var_dims,
                "pool_nodes": pool_nodes,
                "pool_dims": pool_dims,
                "local_score_mode": "affine",
                "enum_max_depth": int(enum_max_depth),
                "enum_max_trees": int(enum_max_trees),
                "max_subtree_depth": max_subtree_depth,
                "preview_topk": int(preview_limit),
                "complexity_penalty": float(complexity_penalty),
                "family_battery_enable": bool(family_battery_enable),
                "family_battery_mode": str(family_battery_mode or "outer"),
                "recursive_enable": bool(recursive_enable),
                "recursive_max_depth": int(recursive_max_depth),
                "recursive_trigger_rel_mse": float(recursive_trigger_rel_mse),
                "recursive_seed_cap": int(recursive_seed_cap),
                "recursive_branch_topk": int(recursive_branch_topk),
                "recursive_child_topk": int(recursive_child_topk),
                "witness_jets_enable": bool(witness_jets_enable),
                "witness_d2_enable": bool(witness_d2_enable),
                "witness_max_rows": int(max(4, int(witness_max_rows))),
                "witness_loss_enable": bool(witness_loss_enable),
                "witness_grad_weight": float(witness_grad_weight),
                "witness_d2_weight": float(witness_d2_weight),
                "witness_diag_weight": float(witness_diag_weight),
                "witness_physics_weight": float(witness_physics_weight),
                "active_var_screen_enable": bool(active_var_screen_enable),
                "active_var_grad_tol": float(active_var_grad_tol),
                "active_var_max_count": max(1, int(active_var_max_count)),
                "safe_eps": float(safe_eps),
                "confidence_mode": str(confidence_mode),
                "confidence_target_gain": float(confidence_target_gain),
                "confidence_floor": float(confidence_floor),
                "branch_beam_width": int(branch_beam_width),
            },
        )
        if bool(recursive_sr_enable):
            route_calls_by_name["recursive_local_sr"] = SolverMarketRouteCall(
                route_name="recursive_local_sr",
                method_name="local_recursive_sr",
                subroute="local_problem_sr",
                runner=solve_local_recursive_sr_preview_rows,
                runner_kwargs={
                    "parent_node": parent_node,
                    "spec_payload": dict(getattr(opportunity, "spec_payload", {}) or {}),
                    "path": path,
                    "target_mode": str(opportunity.target_mode),
                    "target_mapping_kind": str(getattr(opportunity, "target_mapping_kind", "") or ""),
                    "beam_rank": int(opportunity.beam_rank),
                    "slate_id": (
                        f"{str(slate_prefix)}:{opportunity.parent_key}:{'/'.join(str(v) for v in path)}:"
                        f"{str(getattr(opportunity, 'branch_id', '') or '')}:recursive_sr"
                    ),
                    "path_gain": float(getattr(opportunity, "path_gain", 0.0) or 0.0),
                    "max_depth": int(max_depth),
                    "nvars": int(nvars),
                    "poly_degree": int(poly_degree),
                    "var_dims": var_dims,
                    "local_score_mode": "affine",
                    "preview_topk": int(max(1, int(recursive_sr_preview_topk))),
                    "exact_budget": int(max(1, int(recursive_sr_exact_budget))),
                    "max_subtree_depth": max_subtree_depth,
                    "witness_loss_enable": bool(witness_loss_enable),
                    "witness_grad_weight": float(witness_grad_weight),
                    "witness_d2_weight": float(witness_d2_weight),
                    "witness_diag_weight": float(witness_diag_weight),
                    "witness_physics_weight": float(witness_physics_weight),
                },
            )
        if bool(constant_lift_route_enable):
            route_calls_by_name["constant_lift_route"] = SolverMarketRouteCall(
                route_name="constant_lift_route",
                method_name="constant_lift_route",
                subroute="local_problem_constant_lift",
                runner=solve_local_constant_lift_preview_rows,
                runner_kwargs={
                    "parent_node": parent_node,
                    "spec_payload": dict(getattr(opportunity, "spec_payload", {}) or {}),
                    "path": path,
                    "target_mode": str(opportunity.target_mode),
                    "target_mapping_kind": str(getattr(opportunity, "target_mapping_kind", "") or ""),
                    "beam_rank": int(opportunity.beam_rank),
                    "slate_id": (
                        f"{str(slate_prefix)}:{opportunity.parent_key}:{'/'.join(str(v) for v in path)}:"
                        f"{str(getattr(opportunity, 'branch_id', '') or '')}:constant_lift"
                    ),
                    "path_gain": float(getattr(opportunity, "path_gain", 0.0) or 0.0),
                    "max_depth": int(max_depth),
                    "nvars": int(nvars),
                    "poly_degree": int(poly_degree),
                    "var_dims": var_dims,
                    "local_score_mode": "affine",
                    "preview_topk": int(max(1, int(constant_lift_route_topk))),
                    "max_subtree_depth": max_subtree_depth,
                    "constant_lift_topk": int(max(1, int(constant_lift_route_topk))),
                    "lift_route_context": dict(lift_route_context),
                    "witness_loss_enable": bool(witness_loss_enable),
                    "witness_grad_weight": float(witness_grad_weight),
                    "witness_d2_weight": float(witness_d2_weight),
                    "witness_diag_weight": float(witness_diag_weight),
                    "witness_physics_weight": float(witness_physics_weight),
                },
            )
        if bool(coordinate_lift_enable):
            route_calls_by_name["coordinate_lift"] = SolverMarketRouteCall(
                route_name="coordinate_lift",
                method_name="coordinate_lift",
                subroute="local_problem_coordinate",
                runner=solve_local_coordinate_lift_preview_rows,
                runner_kwargs={
                    "parent_node": parent_node,
                    "spec_payload": dict(getattr(opportunity, "spec_payload", {}) or {}),
                    "path": path,
                    "target_mode": str(opportunity.target_mode),
                    "target_mapping_kind": str(getattr(opportunity, "target_mapping_kind", "") or ""),
                    "beam_rank": int(opportunity.beam_rank),
                    "slate_id": (
                        f"{str(slate_prefix)}:{opportunity.parent_key}:{'/'.join(str(v) for v in path)}:"
                        f"{str(getattr(opportunity, 'branch_id', '') or '')}:coordinate_lift"
                    ),
                    "path_gain": float(getattr(opportunity, "path_gain", 0.0) or 0.0),
                    "max_depth": int(max_depth),
                    "nvars": int(nvars),
                    "poly_degree": int(poly_degree),
                    "var_dims": var_dims,
                    "local_score_mode": "affine",
                    "preview_topk": int(preview_limit),
                    "max_subtree_depth": max_subtree_depth,
                    "coordinate_topk": int(max(1, int(coordinate_lift_topk))),
                    "coordinate_mode": str(coordinate_lift_mode or "both"),
                    "lift_route_context": dict(lift_route_context),
                    "witness_loss_enable": bool(witness_loss_enable),
                    "witness_grad_weight": float(witness_grad_weight),
                    "witness_d2_weight": float(witness_d2_weight),
                    "witness_diag_weight": float(witness_diag_weight),
                    "witness_physics_weight": float(witness_physics_weight),
                },
            )
        if bool(tangent_edit_enable):
            route_calls_by_name["tangent_edit"] = SolverMarketRouteCall(
                route_name="tangent_edit",
                method_name="tangent_edit",
                subroute="local_problem_tangent",
                runner=solve_local_tangent_edit_preview_rows,
                runner_kwargs={
                    "parent_node": parent_node,
                    "spec_payload": dict(getattr(opportunity, "spec_payload", {}) or {}),
                    "path": path,
                    "target_mode": str(opportunity.target_mode),
                    "target_mapping_kind": str(getattr(opportunity, "target_mapping_kind", "") or ""),
                    "beam_rank": int(opportunity.beam_rank),
                    "slate_id": (
                        f"{str(slate_prefix)}:{opportunity.parent_key}:{'/'.join(str(v) for v in path)}:"
                        f"{str(getattr(opportunity, 'branch_id', '') or '')}:tangent_edit"
                    ),
                    "path_gain": float(getattr(opportunity, "path_gain", 0.0) or 0.0),
                    "max_depth": int(max_depth),
                    "nvars": int(nvars),
                    "poly_degree": int(poly_degree),
                    "var_dims": var_dims,
                    "pool_nodes": pool_nodes,
                    "pool_dims": pool_dims,
                    "local_score_mode": "affine",
                    "preview_topk": int(max(1, int(tangent_edit_topk))),
                    "max_subtree_depth": max_subtree_depth,
                    "witness_loss_enable": bool(witness_loss_enable),
                    "witness_grad_weight": float(witness_grad_weight),
                    "witness_d2_weight": float(witness_d2_weight),
                    "witness_diag_weight": float(witness_diag_weight),
                    "witness_physics_weight": float(witness_physics_weight),
                },
            )
        if bool(soft_edit_enable):
            route_calls_by_name["soft_edit_search"] = SolverMarketRouteCall(
                route_name="soft_edit_search",
                method_name="soft_edit_search",
                subroute="local_problem_soft",
                runner=solve_local_soft_edit_preview_rows,
                runner_kwargs={
                    "parent_node": parent_node,
                    "spec_payload": dict(getattr(opportunity, "spec_payload", {}) or {}),
                    "path": path,
                    "target_mode": str(opportunity.target_mode),
                    "target_mapping_kind": str(getattr(opportunity, "target_mapping_kind", "") or ""),
                    "beam_rank": int(opportunity.beam_rank),
                    "slate_id": (
                        f"{str(slate_prefix)}:{opportunity.parent_key}:{'/'.join(str(v) for v in path)}:"
                        f"{str(getattr(opportunity, 'branch_id', '') or '')}:soft_edit"
                    ),
                    "path_gain": float(getattr(opportunity, "path_gain", 0.0) or 0.0),
                    "max_depth": int(max_depth),
                    "nvars": int(nvars),
                    "poly_degree": int(poly_degree),
                    "var_dims": var_dims,
                    "pool_nodes": pool_nodes,
                    "pool_dims": pool_dims,
                    "local_score_mode": "affine",
                    "preview_topk": int(preview_limit),
                    "max_subtree_depth": max_subtree_depth,
                    "soft_edit_steps": int(max(1, int(soft_edit_steps))),
                    "soft_edit_l1": float(soft_edit_l1),
                    "witness_loss_enable": bool(witness_loss_enable),
                    "witness_grad_weight": float(witness_grad_weight),
                    "witness_d2_weight": float(witness_d2_weight),
                    "witness_diag_weight": float(witness_diag_weight),
                    "witness_physics_weight": float(witness_physics_weight),
                },
            )
        if bool(directional_market_enable):
            if str(direction or "") == "inside_out":
                route_order = [
                    "constant_lift_route",
                    "coordinate_lift",
                    "recursive_local_sr",
                    "tangent_edit",
                    "soft_edit_search",
                    "inverse_spec_followup",
                ]
            elif str(direction or "") == "outside_in":
                route_order = [
                    "inverse_spec_followup",
                    "constant_lift_route",
                    "coordinate_lift",
                    "recursive_local_sr",
                    "tangent_edit",
                    "soft_edit_search",
                ]
            else:
                route_order = [
                    "inverse_spec_followup",
                    "constant_lift_route",
                    "coordinate_lift",
                    "recursive_local_sr",
                    "tangent_edit",
                    "soft_edit_search",
                ]
        else:
            route_order = [
                "inverse_spec_followup",
                "constant_lift_route",
                "coordinate_lift",
                "recursive_local_sr",
                "tangent_edit",
                "soft_edit_search",
            ]
        route_order = _apply_lift_route_order(route_order, route_context=lift_route_context)
        route_calls = [route_calls_by_name[name] for name in route_order if name in route_calls_by_name]
        return route_calls, "ok"
    return [], "unsupported_spec_kind"


def _rank_abstraction_candidate_paths(parent_expr) -> list[tuple[int, ...]]:
    scored: list[tuple[float, int, int, int, tuple[int, ...]]] = []
    for p in collect_paths(parent_expr):
        if len(p) <= 0:
            continue
        try:
            sub = get_at(parent_expr, p)
            depth = int(node_depth(sub))
            size = int(node_size(sub))
            if depth < 2:
                continue
            op = str(sub[0]) if isinstance(sub, tuple) and sub else ""
        except Exception:
            continue
        op_bonus = 0.0
        if op in {"mul", "div", "sin", "cos", "exp", "log", "sqrt", "sqr"}:
            op_bonus = 0.5
        elif op in {"add", "sub"}:
            op_bonus = 0.25
        score = float(size) + 0.5 * float(depth) + op_bonus - 0.05 * float(len(p))
        scored.append((score, size, depth, -len(p), tuple(int(v) for v in p)))
    scored.sort(reverse=True)
    return [path for *_rest, path in scored]


@torch.no_grad()
def abstract_frontier_from_parent(
    frontier: HoleFrontier,
    *,
    parent_key: str,
    parent_elite_id: str,
    parent_expr,
    parent_mapping,
    parent_eff_mse: float,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    max_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims=None,
    current_iter: int = 0,
    max_paths_per_parent: int = 3,
    preview_enum_max_depth: int = 2,
    preview_enum_max_trees: int = 32,
    preview_topk: int = 4,
    safe_eps: float = 1e-12,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    source: str = "abstraction",
    recursive_enable: bool = False,
    recursive_max_depth: int = 0,
    recursive_trigger_rel_mse: float = 0.25,
    regime_metadata: Mapping[str, Any] | None = None,
    candidate_path_cap: int | None = None,
    snapshot_parent_fn=None,
    debug_stats: dict[str, float] | None = None,
) -> int:
    """Abstract one parent expression into typed hole opportunities."""
    from .inverse_action import (
        _estimate_inverse_action_transport,
        _inverse_action_path_mode_beam_states,
    )

    dm = var_dims is not None
    cfg_base = {
        "max_paths": int(max_paths_per_parent),
        "dm": bool(dm),
        "var_dims": var_dims,
        "max_depth": int(max_depth),
        "poly_degree": int(poly_degree),
        "topk_terms": 4,
        "shortlist_mult": 2,
        "local_mode": "affine",
        "min_valid_frac": 0.10,
        "min_confidence": 0.05,
        "safe_eps": float(safe_eps),
        "confidence_mode": str(confidence_mode),
        "confidence_target_gain": float(confidence_target_gain),
        "confidence_floor": float(confidence_floor),
        "branch_beam_width": 1,
        "micro_search_enable": False,
        "micro_search_max_depth": 3,
        "micro_search_beam_width": 16,
        "micro_search_topk": 8,
        "micro_search_seed_terms": 4,
        "target_mode": "robust",
        "full_mapping_penalty": 0.75,
        "exact_simple_target_bonus": 0.10,
        "additive_descend_penalty": 0.15,
        "nonadditive_leaf_penalty": 0.20,
        "exact_path_eta": 0.98,
        "exact_transport_min_lin_rel": 0.0,
        "periodic_min_valid_scale": 1.25,
        "periodic_min_confidence_scale": 1.35,
        "periodic_path_penalty": 0.65,
        "nonperiodic_muldiv_bonus": 0.10,
        "nonperiodic_explogsqrt_bonus": 0.05,
        "branch_ambiguity_penalty": 0.50,
        "transport_min_lin_rel": 0.02,
        "transport_min_effective_n": 8.0,
    }
    total_ingested = 0

    def _debug_inc(key: str, value: float = 1.0) -> None:
        if debug_stats is None:
            return
        try:
            debug_stats[key] = float(debug_stats.get(key, 0.0) or 0.0) + float(value)
        except Exception:
            pass
    parent_expr_str = node_str(parent_expr)
    parent_snapshot_id = ""
    if callable(snapshot_parent_fn):
        try:
            parent_snapshot_id = str(snapshot_parent_fn(
                parent_key=parent_key,
                parent_elite_id=parent_elite_id,
                parent_expr=parent_expr,
                parent_mapping=parent_mapping,
                parent_expr_str=parent_expr_str,
                parent_eff_mse=float(parent_eff_mse),
                current_iter=int(current_iter),
            ) or "")
        except Exception:
            parent_snapshot_id = ""

    all_paths = _rank_abstraction_candidate_paths(parent_expr)
    _debug_inc("candidate_paths_total", len(all_paths))
    if candidate_path_cap is None:
        candidate_path_cap = max(int(max_paths_per_parent), 4 * int(max_paths_per_parent))
    interesting_paths = list(all_paths[:max(1, int(candidate_path_cap))])
    if not interesting_paths:
        _debug_inc("blocked_no_paths")
        return 0
    _debug_inc("candidate_paths_selected", len(interesting_paths))

    try:
        transport_ctx = _estimate_inverse_action_transport(
            parent_expr, parent_mapping,
            x_fit, y_fit, x_probe, y_probe,
            interesting_paths,
            safe_eps=float(safe_eps),
        )
    except Exception:
        _debug_inc("blocked_transport")
        return 0

    try:
        beam_states = _inverse_action_path_mode_beam_states(
            parent_node=parent_expr,
            parent_mapping=parent_mapping,
            x_fit=x_fit, y_fit=y_fit,
            x_probe=x_probe, y_probe=y_probe,
            pool_nodes=pool_nodes,
            pool_phi_fit=pool_phi_fit,
            pool_phi_probe=pool_phi_probe,
            pool_dims=pool_dims,
            all_paths=interesting_paths,
            path_target_modes=None,
            transport_ctx=transport_ctx,
            cfg=cfg_base,
            beam_width=int(max_paths_per_parent),
        )
    except Exception:
        _debug_inc("blocked_beam_state")
        return 0
    if not beam_states:
        _debug_inc("blocked_beam_state")
        return 0
    _debug_inc("beam_states_total", len(beam_states))

    for beam_rank, beam_state in enumerate(beam_states or []):
        path = tuple(int(v) for v in (beam_state.get("path", ()) or ()))
        target_mode = str(beam_state.get("target_mode", "") or "")
        confidence = float(beam_state.get("confidence", 0.0) or 0.0)
        valid_frac = float(beam_state.get("valid_frac", 0.0) or 0.0)
        path_gain = float(beam_state.get("path_gain", 0.0) or 0.0)

        if confidence < 0.05 or valid_frac < 0.10:
            _debug_inc("filtered_low_confidence_or_validity")
            continue

        try:
            _debug_inc("preview_calls")
            preview_result = solve_inverse_spec_preview_rows(
                parent_node=parent_expr,
                beam_state=beam_state,
                regime_metadata=regime_metadata,
                beam_rank=int(beam_rank),
                slate_id=f"abstract:{source}:{parent_key}:{'/'.join(str(v) for v in path)}",
                max_depth=int(max_depth),
                nvars=int(nvars),
                poly_degree=int(poly_degree),
                var_dims=var_dims,
                pool_nodes=pool_nodes,
                pool_dims=pool_dims,
                enum_max_depth=int(preview_enum_max_depth),
                enum_max_trees=int(preview_enum_max_trees),
                preview_topk=int(preview_topk),
                recursive_enable=bool(recursive_enable),
                recursive_max_depth=max(0, int(recursive_max_depth)),
                recursive_trigger_rel_mse=float(recursive_trigger_rel_mse),
                recursive_seed_cap=4,
                recursive_branch_topk=2,
                recursive_child_topk=1,
                safe_eps=float(safe_eps),
                confidence_mode=str(confidence_mode),
                confidence_target_gain=float(confidence_target_gain),
                confidence_floor=float(confidence_floor),
            )
        except Exception:
            _debug_inc("preview_exception")
            preview_result = {"rows": [], "solver_meta": {}}

        preview_rows = preview_result.get("rows", []) or []
        solver_meta = preview_result.get("solver_meta", {}) or {}
        if not preview_rows:
            _debug_inc("preview_zero")
        best_preview_mse = float("inf")
        best_preview_row = None
        if preview_rows:
            best_preview_row = preview_rows[0]
            best_preview_mse = float(best_preview_row.get("local_probe_mse", float("inf")))
        best_preview_meta_row = best_preview_row if isinstance(best_preview_row, Mapping) else {}
        witness_fields = _opportunity_witness_fields(best_preview_meta_row)

        opp = HoleOpportunity(
            parent_key=str(parent_key),
            parent_expr_str=str(parent_expr_str),
            path=path,
            target_mode=target_mode,
            beam_rank=int(beam_rank),
            parent_elite_id=str(parent_elite_id),
            parent_snapshot_id=str(parent_snapshot_id),
            source=str(source or "abstraction"),
            spec_kind="path_hole",
            branch_id=str(beam_state.get("inverse_spec_branch_id", "") or ""),
            continuation_key=_tuple_key(beam_state.get("inverse_spec_continuation_key", ())),
            trace=_tuple_key(solver_meta.get("trace", beam_state.get("trace", ()) or ())),
            target_dim_key=_dim_key(beam_state.get("target_dim", None)),
            wrappers_left=int(max(0, recursive_max_depth)),
            recursion_level=0,
            path_gain=path_gain,
            confidence=confidence,
            valid_frac=valid_frac,
            transport_rel=float(beam_state.get("transport_rel", 0.0) or 0.0),
            effective_n=float(beam_state.get("effective_n", 0.0) or 0.0),
            target_mapping_kind=str(beam_state.get("target_mapping_kind", "") or ""),
            candidate_count=int(len(preview_rows)),
            best_preview_probe_mse=best_preview_mse if math.isfinite(best_preview_mse) else None,
            parent_eff_mse_at_emit=_finite_float(parent_eff_mse),
            best_gate_score=float(path_gain * confidence * valid_frac),
            witness_value_loss=witness_fields.get("witness_value_loss", None),
            witness_grad_loss=witness_fields.get("witness_grad_loss", None),
            witness_d2_loss=witness_fields.get("witness_d2_loss", None),
            witness_diag_loss=witness_fields.get("witness_diag_loss", None),
            witness_physics_loss=witness_fields.get("witness_physics_loss", None),
            witness_energy_total=witness_fields.get("witness_energy_total", None),
            witness_energy_delta_estimate=witness_fields.get("witness_energy_delta_estimate", None),
            witness_fit_jet_source=str(best_preview_meta_row.get("witness_fit_jet_source", "") or ""),
            witness_probe_jet_source=str(best_preview_meta_row.get("witness_probe_jet_source", "") or ""),
            witness_fit_jet_requested_source=str(best_preview_meta_row.get("witness_fit_jet_requested_source", "") or ""),
            witness_probe_jet_requested_source=str(best_preview_meta_row.get("witness_probe_jet_requested_source", "") or ""),
            witness_fit_jet_fallback_used=bool(best_preview_meta_row.get("witness_fit_jet_fallback_used", False)),
            witness_probe_jet_fallback_used=bool(best_preview_meta_row.get("witness_probe_jet_fallback_used", False)),
            witness_numeric_jet_fallback_used=bool(best_preview_meta_row.get("witness_numeric_jet_fallback_used", False)),
            witness_exact_jet_used=bool(best_preview_meta_row.get("witness_exact_jet_used", False)),
            preview_solvability=best_preview_mse if math.isfinite(best_preview_mse) else None,
            preview_periodic_fired=bool(solver_meta.get("periodic_forward_used", False)),
            preview_candidate_count=int(solver_meta.get("candidate_count_scored", 0) or 0),
            preview_recursive_depth=int(solver_meta.get("recursive_depth_reached", 0) or 0),
            predicted_value=_spec_value_hint(
                parent_eff_mse_at_emit=parent_eff_mse,
                best_preview_probe_mse=best_preview_mse if math.isfinite(best_preview_mse) else None,
                path_gain=path_gain,
                confidence=confidence,
                valid_frac=valid_frac,
            ),
            predicted_cost=_spec_cost_hint(
                preview_candidate_count=int(solver_meta.get("candidate_count_scored", 0) or 0),
                preview_recursive_depth=int(solver_meta.get("recursive_depth_reached", 0) or 0),
                preview_periodic_fired=bool(solver_meta.get("periodic_forward_used", False)),
            ),
            created_at_iter=int(current_iter),
        )
        opp = frontier._set_priority_hints(opp)

        existing = frontier._entries.get(opp.frontier_key)
        if existing is None or (
            opp.preview_solvability is not None
            and (existing.preview_solvability is None or opp.preview_solvability < existing.preview_solvability)
        ):
            frontier._entries[opp.frontier_key] = opp
            total_ingested += 1
            _debug_inc("added")
        else:
            _debug_inc("dedup_rejected")

    if len(frontier) > int(frontier.max_entries):
        frontier.prune(current_iter)
    return total_ingested


@torch.no_grad()
def mine_frontier_from_archive(
    frontier: HoleFrontier,
    archive_records: Sequence[Any],
    *,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    max_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims=None,
    current_iter: int = 0,
    max_parents: int = 5,
    max_paths_per_parent: int = 3,
    preview_enum_max_depth: int = 2,
    preview_enum_max_trees: int = 32,
    preview_topk: int = 4,
    safe_eps: float = 1e-12,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    snapshot_parent_fn=None,
) -> int:
    """Mine hole opportunities from top archive elites."""
    total_ingested = 0
    recs = []
    for rec in (archive_records or []):
        try:
            expr = rec.best_expr
            mapping = rec.mapping
            mse = float(rec.best_mse)
            key = str(getattr(rec, "residual_basin_key", getattr(rec, "key", "")) or node_str(expr))
            elite_id = str(getattr(rec, "best_elite_id", "") or "")
            if expr is not None and mapping is not None and math.isfinite(mse):
                recs.append((mse, key, elite_id, expr, mapping))
        except Exception:
            continue
    recs.sort(key=lambda r: r[0])
    recs = recs[:max(1, int(max_parents))]

    for _mse, parent_key, parent_elite_id, parent_expr, parent_mapping in recs:
        total_ingested += int(abstract_frontier_from_parent(
            frontier,
            parent_key=str(parent_key),
            parent_elite_id=str(parent_elite_id),
            parent_expr=parent_expr,
            parent_mapping=parent_mapping,
            parent_eff_mse=float(_mse),
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            pool_nodes=pool_nodes,
            pool_phi_fit=pool_phi_fit,
            pool_phi_probe=pool_phi_probe,
            pool_dims=pool_dims,
            max_depth=int(max_depth),
            nvars=int(nvars),
            poly_degree=int(poly_degree),
            var_dims=var_dims,
            current_iter=int(current_iter),
            max_paths_per_parent=int(max_paths_per_parent),
            preview_enum_max_depth=int(preview_enum_max_depth),
            preview_enum_max_trees=int(preview_enum_max_trees),
            preview_topk=int(preview_topk),
            safe_eps=float(safe_eps),
            confidence_mode=str(confidence_mode),
            confidence_target_gain=float(confidence_target_gain),
            confidence_floor=float(confidence_floor),
            source="archive_mine",
            snapshot_parent_fn=snapshot_parent_fn,
        ))
    return total_ingested


# ---------------------------------------------------------------------------
# Risk-seeking tournament
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_hole_tournament(
    candidates: list[HoleOpportunity],
    *,
    parent_resolver,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    max_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims=None,
    preview_budget: int = 64,
    preview_topk: int = 4,
    elite_k: int = 2,
    solver_market_enable: bool = False,
    solver_market_preview_topk: int = 4,
    solver_market_exact_topk: int = 2,
    solver_market_proposal_objects_enable: bool = False,
    inverse_spec_recursive_sr_enable: bool = False,
    inverse_spec_recursive_sr_preview_topk: int = 4,
    inverse_spec_recursive_sr_exact_budget: int = 2,
    inverse_spec_constant_lift_route_enable: bool = False,
    inverse_spec_constant_lift_route_topk: int = 2,
    inverse_spec_coordinate_lift_enable: bool = False,
    inverse_spec_coordinate_lift_topk: int = 4,
    inverse_spec_coordinate_lift_mode: str = "both",
    inverse_spec_tangent_edit_enable: bool = False,
    inverse_spec_tangent_edit_topk: int = 8,
    inverse_spec_soft_edit_enable: bool = False,
    inverse_spec_soft_edit_steps: int = 64,
    inverse_spec_soft_edit_l1: float = 1.0e-3,
    inverse_spec_witness_jets_enable: bool = False,
    inverse_spec_witness_d2_enable: bool = False,
    inverse_spec_witness_max_rows: int = 64,
    inverse_spec_witness_loss_enable: bool = False,
    inverse_spec_witness_grad_weight: float = 1.0,
    inverse_spec_witness_d2_weight: float = 0.0,
    inverse_spec_witness_diag_weight: float = 0.0,
    inverse_spec_witness_physics_weight: float = 0.0,
    inverse_spec_active_var_screen_enable: bool = False,
    inverse_spec_active_var_grad_tol: float = 1.0e-3,
    inverse_spec_active_var_max_count: int = 4,
    inverse_spec_directional_market_enable: bool = False,
    family_battery_enable: bool = False,
    family_battery_mode: str = "outer",
    safe_eps: float = 1e-12,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
) -> list[tuple[HoleOpportunity, float]]:
    """Cheap-preview *N* holes, return the top *elite_k* by preview MSE.

    This implements the risk-seeking principle: evaluate many candidates
    cheaply, then concentrate the expensive full-solver budget on only the
    top fraction.  Losers get their ``preview_solvability`` updated in-place
    so the frontier scoring benefits from the new information.

    Parameters
    ----------
    candidates : list[HoleOpportunity]
        Eligible holes selected from the frontier.
    parent_resolver : callable(opp) -> dict | None
        Returns ``{"parent_node": ..., "parent_mapping": ...}`` or *None*
        if the parent cannot be resolved.

    Returns
    -------
    list of (HoleOpportunity, preview_mse) for the top *elite_k*.
    """
    dm = var_dims is not None
    cfg_base = {
        "max_paths": 1,
        "dm": bool(dm),
        "var_dims": var_dims,
        "max_depth": int(max_depth),
        "poly_degree": int(poly_degree),
        "topk_terms": 4,
        "shortlist_mult": 2,
        "local_mode": "affine",
        "min_valid_frac": 0.10,
        "min_confidence": 0.05,
        "safe_eps": float(safe_eps),
        "confidence_mode": str(confidence_mode),
        "confidence_target_gain": float(confidence_target_gain),
        "confidence_floor": float(confidence_floor),
        "branch_beam_width": 1,
        "micro_search_enable": False,
        "micro_search_max_depth": 3,
        "micro_search_beam_width": 16,
        "micro_search_topk": 8,
        "micro_search_seed_terms": 4,
        "target_mode": "robust",
        "full_mapping_penalty": 0.75,
        "exact_simple_target_bonus": 0.10,
        "additive_descend_penalty": 0.15,
        "nonadditive_leaf_penalty": 0.20,
        "exact_path_eta": 0.98,
        "exact_transport_min_lin_rel": 0.0,
        "periodic_min_valid_scale": 1.25,
        "periodic_min_confidence_scale": 1.35,
        "periodic_path_penalty": 0.65,
        "nonperiodic_muldiv_bonus": 0.10,
        "nonperiodic_explogsqrt_bonus": 0.05,
        "branch_ambiguity_penalty": 0.50,
        "transport_min_lin_rel": 0.02,
        "transport_min_effective_n": 8.0,
    }

    scored: list[tuple[float, HoleOpportunity]] = []

    for opp in candidates:
        # Resolve parent expression for this opportunity
        resolved = None
        try:
            resolved = parent_resolver(opp)
        except Exception:
            pass
        if resolved is None:
            continue
        parent_node = resolved["parent_node"]
        parent_mapping = resolved["parent_mapping"]

        route_preview_topk = int(preview_topk)
        if bool(solver_market_enable):
            route_preview_topk = max(route_preview_topk, max(1, int(solver_market_preview_topk)))
        route_calls, status = _build_spec_preview_route_calls(
            opp,
            parent_node=parent_node,
            parent_mapping=parent_mapping,
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            pool_nodes=pool_nodes,
            pool_phi_fit=pool_phi_fit,
            pool_phi_probe=pool_phi_probe,
            pool_dims=pool_dims,
            max_depth=int(max_depth),
            nvars=int(nvars),
            poly_degree=int(poly_degree),
            var_dims=var_dims,
            enum_max_depth=min(int(max_depth), 3),
            enum_max_trees=int(preview_budget),
            preview_topk=int(route_preview_topk),
            max_subtree_depth=None,
            complexity_penalty=0.0,
            family_battery_enable=bool(family_battery_enable),
            family_battery_mode=str(family_battery_mode or "outer"),
            recursive_enable=False,
            recursive_max_depth=0,
            recursive_trigger_rel_mse=1.0,
            recursive_seed_cap=4,
            recursive_branch_topk=2,
            recursive_child_topk=1,
            recursive_sr_enable=bool(inverse_spec_recursive_sr_enable),
            recursive_sr_preview_topk=int(inverse_spec_recursive_sr_preview_topk),
            recursive_sr_exact_budget=int(inverse_spec_recursive_sr_exact_budget),
            constant_lift_route_enable=bool(inverse_spec_constant_lift_route_enable),
            constant_lift_route_topk=int(inverse_spec_constant_lift_route_topk),
            coordinate_lift_enable=bool(inverse_spec_coordinate_lift_enable),
            coordinate_lift_topk=int(inverse_spec_coordinate_lift_topk),
            coordinate_lift_mode=str(inverse_spec_coordinate_lift_mode or "both"),
            tangent_edit_enable=bool(inverse_spec_tangent_edit_enable),
            tangent_edit_topk=int(inverse_spec_tangent_edit_topk),
            soft_edit_enable=bool(inverse_spec_soft_edit_enable),
            soft_edit_steps=int(inverse_spec_soft_edit_steps),
            soft_edit_l1=float(inverse_spec_soft_edit_l1),
            witness_jets_enable=bool(inverse_spec_witness_jets_enable),
            witness_d2_enable=bool(inverse_spec_witness_d2_enable),
            witness_max_rows=int(max(4, int(inverse_spec_witness_max_rows))),
            witness_loss_enable=bool(inverse_spec_witness_loss_enable),
            witness_grad_weight=float(inverse_spec_witness_grad_weight),
            witness_d2_weight=float(inverse_spec_witness_d2_weight),
            witness_diag_weight=float(inverse_spec_witness_diag_weight),
            witness_physics_weight=float(inverse_spec_witness_physics_weight),
            active_var_screen_enable=bool(inverse_spec_active_var_screen_enable),
            active_var_grad_tol=float(inverse_spec_active_var_grad_tol),
            active_var_max_count=max(1, int(inverse_spec_active_var_max_count)),
            directional_market_enable=bool(inverse_spec_directional_market_enable),
            safe_eps=float(safe_eps),
            confidence_mode=str(confidence_mode),
            confidence_target_gain=float(confidence_target_gain),
            confidence_floor=float(confidence_floor),
            branch_beam_width=1,
            beam_cfg=cfg_base,
            slate_prefix="tournament",
        )
        if not route_calls:
            if str(status) != "ok":
                continue
            continue
        try:
            if bool(solver_market_enable):
                market_kwargs = {
                    "preview_topk": max(1, int(solver_market_preview_topk)),
                    "exact_topk": max(1, int(solver_market_exact_topk)),
                }
                if bool(solver_market_proposal_objects_enable):
                    market_kwargs["proposal_objects_enable"] = True
                preview_result = run_preview_solver_market(route_calls, **market_kwargs)
            else:
                preview_result = route_calls[0].runner(**dict(route_calls[0].runner_kwargs or {}))
        except Exception:
            continue

        preview_rows = preview_result.get("rows", []) or []
        if not preview_rows:
            continue
        best_mse = float(preview_rows[0].get("local_probe_mse", float("inf")))
        if not math.isfinite(best_mse):
            continue

        # Update the opportunity's preview score in-place so losers also
        # benefit from the cheap preview data.
        if opp.preview_solvability is None or best_mse < opp.preview_solvability:
            opp.preview_solvability = best_mse
            witness_fields = _opportunity_witness_fields(preview_rows[0])
            opp.witness_value_loss = witness_fields.get("witness_value_loss", None)
            opp.witness_grad_loss = witness_fields.get("witness_grad_loss", None)
            opp.witness_d2_loss = witness_fields.get("witness_d2_loss", None)
            opp.witness_diag_loss = witness_fields.get("witness_diag_loss", None)
            opp.witness_physics_loss = witness_fields.get("witness_physics_loss", None)
            opp.witness_energy_total = witness_fields.get("witness_energy_total", None)
            opp.witness_energy_delta_estimate = witness_fields.get("witness_energy_delta_estimate", None)
            opp.witness_fit_jet_source = str(preview_rows[0].get("witness_fit_jet_source", "") or "")
            opp.witness_probe_jet_source = str(preview_rows[0].get("witness_probe_jet_source", "") or "")
            opp.witness_fit_jet_requested_source = str(preview_rows[0].get("witness_fit_jet_requested_source", "") or "")
            opp.witness_probe_jet_requested_source = str(preview_rows[0].get("witness_probe_jet_requested_source", "") or "")
            opp.witness_fit_jet_fallback_used = bool(preview_rows[0].get("witness_fit_jet_fallback_used", False))
            opp.witness_probe_jet_fallback_used = bool(preview_rows[0].get("witness_probe_jet_fallback_used", False))
            opp.witness_numeric_jet_fallback_used = bool(preview_rows[0].get("witness_numeric_jet_fallback_used", False))
            opp.witness_exact_jet_used = bool(preview_rows[0].get("witness_exact_jet_used", False))
            if opp.best_preview_probe_mse is None or best_mse < opp.best_preview_probe_mse:
                opp.best_preview_probe_mse = best_mse

        scored.append((best_mse, opp))

    scored.sort(key=lambda t: t[0])
    elite_k = max(1, int(elite_k))
    return [(opp, mse) for mse, opp in scored[:elite_k]]


# ---------------------------------------------------------------------------
# Hole search action
# ---------------------------------------------------------------------------

def _normalize_followup_spec_rows(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    opportunity: HoleOpportunity,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    inherited_path = [int(v) for v in tuple(getattr(opportunity, "path", ()) or ())]
    inherited_path_gain = max(1.0e-6, float(getattr(opportunity, "path_gain", 0.0) or 0.0))
    inherited_target_mode = str(getattr(opportunity, "target_mode", "") or "")
    inherited_target_mapping_kind = str(getattr(opportunity, "target_mapping_kind", "") or "")
    inherited_preview_budget = max(0, int(getattr(opportunity, "preview_budget", 0) or 0))
    inherited_exact_budget = max(0, int(getattr(opportunity, "exact_budget", 0) or 0))
    for row in list(rows or []):
        if not isinstance(row, Mapping):
            continue
        followup = dict(row)
        subproblem_spec = deserialize_subproblem_spec(followup.get("spec_payload", None))
        if subproblem_spec is not None:
            if not followup.get("spec_kind", None):
                followup["spec_kind"] = str(subproblem_spec.problem_kind or "local_problem")
            if not followup.get("direction", None):
                followup["direction"] = str(subproblem_spec.direction or "")
            if not followup.get("path", None):
                followup["path"] = [int(v) for v in tuple(subproblem_spec.path or ())]
            if not followup.get("target_mode", None):
                followup["target_mode"] = str(subproblem_spec.target_mode or "")
            if not followup.get("target_mapping_kind", None):
                followup["target_mapping_kind"] = str(subproblem_spec.target_mapping_kind or "")
            if not followup.get("trace", None):
                trace = dict(subproblem_spec.metadata or {}).get("trace", ())
                followup["trace"] = [str(v) for v in tuple(trace or ())]
            if not followup.get("continuation_key", None):
                continuation = [
                    str(dict(frame).get("op", "") or "")
                    for frame in list(subproblem_spec.continuation_frames or [])
                    if isinstance(frame, Mapping)
                ]
                followup["continuation_key"] = [token for token in continuation if token]
            if not followup.get("target_dim_key", None):
                followup["target_dim_key"] = list(_dim_key(subproblem_spec.target_dim))
            if not followup.get("wrappers_left", None):
                followup["wrappers_left"] = int(subproblem_spec.wrappers_left)
            if not followup.get("recursion_level", None):
                followup["recursion_level"] = int(subproblem_spec.recursion_level)
        followup.setdefault("path", inherited_path)
        followup.setdefault("target_mode", inherited_target_mode)
        followup.setdefault("direction", str(_opportunity_direction(opportunity) or ""))
        followup.setdefault("beam_rank", int(getattr(opportunity, "beam_rank", 0) or 0))
        followup.setdefault("path_gain", inherited_path_gain)
        followup.setdefault("transport_rel", float(getattr(opportunity, "transport_rel", 0.0) or 0.0))
        followup.setdefault("effective_n", float(getattr(opportunity, "effective_n", 0.0) or 0.0))
        followup.setdefault("target_mapping_kind", inherited_target_mapping_kind)
        followup.setdefault("parent_eff_mse", getattr(opportunity, "parent_eff_mse_at_emit", None))
        followup.setdefault("preview_budget", inherited_preview_budget)
        followup.setdefault("exact_budget", inherited_exact_budget)
        try:
            path_gain = float(followup.get("path_gain", 0.0) or 0.0)
        except Exception:
            path_gain = 0.0
        if path_gain <= 0.0:
            followup["path_gain"] = inherited_path_gain
        out.append(followup)
    return out

@torch.no_grad()
def run_hole_search_action(
    opportunity: HoleOpportunity,
    *,
    parent_node,
    parent_mapping,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    pool_dims,
    rng,
    max_depth: int,
    nvars: int,
    poly_degree: int,
    var_dims=None,
    # Hole-search-specific budgets
    enum_max_depth: int = 4,
    enum_max_trees: int = 3000,
    preview_topk: int = 8,
    exact_budget: int = 2,
    solver_market_enable: bool = False,
    solver_market_preview_topk: int = 4,
    solver_market_exact_topk: int = 2,
    solver_market_proposal_objects_enable: bool = False,
    inverse_spec_recursive_sr_enable: bool = False,
    inverse_spec_recursive_sr_preview_topk: int = 4,
    inverse_spec_recursive_sr_exact_budget: int = 2,
    inverse_spec_constant_lift_route_enable: bool = False,
    inverse_spec_constant_lift_route_topk: int = 2,
    inverse_spec_coordinate_lift_enable: bool = False,
    inverse_spec_coordinate_lift_topk: int = 4,
    inverse_spec_coordinate_lift_mode: str = "both",
    inverse_spec_tangent_edit_enable: bool = False,
    inverse_spec_tangent_edit_topk: int = 8,
    inverse_spec_soft_edit_enable: bool = False,
    inverse_spec_soft_edit_steps: int = 64,
    inverse_spec_soft_edit_l1: float = 1.0e-3,
    inverse_spec_witness_jets_enable: bool = False,
    inverse_spec_witness_d2_enable: bool = False,
    inverse_spec_witness_max_rows: int = 64,
    inverse_spec_witness_loss_enable: bool = False,
    inverse_spec_witness_grad_weight: float = 1.0,
    inverse_spec_witness_d2_weight: float = 0.0,
    inverse_spec_witness_diag_weight: float = 0.0,
    inverse_spec_witness_physics_weight: float = 0.0,
    inverse_spec_active_var_screen_enable: bool = False,
    inverse_spec_active_var_grad_tol: float = 1.0e-3,
    inverse_spec_active_var_max_count: int = 4,
    inverse_spec_directional_market_enable: bool = False,
    max_subtree_depth: int | None = None,
    complexity_penalty: float = 0.0,
    family_battery_enable: bool = False,
    family_battery_mode: str = "outer",
    # Recursive/periodic solver settings
    recursive_enable: bool = True,
    recursive_max_depth: int = 2,
    recursive_trigger_rel_mse: float = 0.0,
    recursive_seed_cap: int = 6,
    recursive_branch_topk: int = 4,
    recursive_child_topk: int = 2,
    # Inverse confidence settings
    safe_eps: float = 1e-12,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    branch_beam_width: int = 1,
    # Exact scoring
    proj=None,
    fp_mode: str = "bits",
    q_scale: float = 2.0,
    q_clip: float = 8.0,
    score_expr_cfg: dict[str, Any] | None = None,
    score_expr_fn=None,
    return_meta: bool = False,
    inverse_spec_regime_metadata: Mapping[str, Any] | None = None,
):
    """Execute one hole-search step on a selected opportunity.

    Rebuilds the beam state for the opportunity's path, runs the
    inverse-spec solver, cheaply reranks the top candidates, and returns
    the best child expression. The outer search loop performs the single
    full exact score for the selected child.
    """
    started = time.perf_counter()
    dm = var_dims is not None

    meta: dict[str, Any] = {
        "status": "started",
        "hole_search_parent_key": str(opportunity.parent_key),
        "hole_search_parent_elite_id": str(opportunity.parent_elite_id),
        "hole_search_spec_kind": str(getattr(opportunity, "spec_kind", "path_hole") or "path_hole"),
        "hole_search_direction": str(_opportunity_direction(opportunity) or ""),
        "hole_search_path": [int(v) for v in opportunity.path],
        "hole_search_target_mode": str(opportunity.target_mode),
        "hole_search_beam_rank": int(opportunity.beam_rank),
        "hole_search_confidence": float(opportunity.confidence),
        "hole_search_valid_frac": float(opportunity.valid_frac),
        "hole_search_path_gain": float(opportunity.path_gain),
        "hole_search_attempt": int(opportunity.attempts),
        "hole_search_preview_count": 0,
        "hole_search_exact_scored": 0,
        "hole_search_periodic_used": False,
        "hole_search_followup_spec_state_count": 0,
        "hole_search_solver_market_enable": bool(solver_market_enable),
        "hole_search_solver_market_proposal_objects_enable": bool(solver_market_proposal_objects_enable),
        "hole_search_solver_market_route_count": 0,
        "hole_search_solver_market_candidate_count_raw": 0,
        "hole_search_solver_market_candidate_count_unique": 0,
        "hole_search_solver_market_selected_route": "",
        "hole_search_solver_market_selected_method_name": "",
        "hole_search_solver_market_selected_subroute": "",
        "hole_search_solver_market_directional_order": [],
        "hole_search_solver_market_routes": [],
        "hole_search_wall_seconds": 0.0,
        "hole_search_best_eff_mse": None,
        "observed_wall_seconds": 0.0,
        "observed_exact_evals": 0,
        "observed_preview_evals": 0,
        "observed_micro_tokens": 0,
        "observed_widen_tokens": 0,
    }

    def _return(expr, status="ok"):
        meta["status"] = str(status)
        meta["hole_search_wall_seconds"] = float(time.perf_counter() - started)
        meta["observed_wall_seconds"] = float(meta["hole_search_wall_seconds"])
        meta["observed_exact_evals"] = int(meta.get("hole_search_exact_scored", 0) or 0)
        meta["observed_preview_evals"] = int(meta.get("hole_search_preview_count", 0) or 0)
        if return_meta:
            return expr, meta
        return expr
    spec_kind = str(getattr(opportunity, "spec_kind", "path_hole") or "path_hole")
    cfg = {
        "max_paths": 1,
        "dm": bool(dm),
        "var_dims": var_dims,
        "max_depth": int(max_depth),
        "poly_degree": int(poly_degree),
        "topk_terms": 6,
        "shortlist_mult": 4,
        "local_mode": "affine",
        "min_valid_frac": 0.10,
        "min_confidence": 0.05,
        "safe_eps": float(safe_eps),
        "confidence_mode": str(confidence_mode),
        "confidence_target_gain": float(confidence_target_gain),
        "confidence_floor": float(confidence_floor),
        "branch_beam_width": int(branch_beam_width),
        "micro_search_enable": False,
        "micro_search_max_depth": 3,
        "micro_search_beam_width": 24,
        "micro_search_topk": 16,
        "micro_search_seed_terms": 8,
        "target_mode": str(opportunity.target_mode) or "robust",
        "full_mapping_penalty": 0.75,
        "exact_simple_target_bonus": 0.10,
        "additive_descend_penalty": 0.15,
        "nonadditive_leaf_penalty": 0.20,
        "exact_path_eta": 0.98,
        "exact_transport_min_lin_rel": 0.0,
        "periodic_min_valid_scale": 1.25,
        "periodic_min_confidence_scale": 1.35,
        "periodic_path_penalty": 0.65,
        "nonperiodic_muldiv_bonus": 0.10,
        "nonperiodic_explogsqrt_bonus": 0.05,
        "branch_ambiguity_penalty": 0.50,
        "transport_min_lin_rel": 0.02,
        "transport_min_effective_n": 8.0,
    }
    route_preview_topk = int(preview_topk)
    if bool(solver_market_enable):
        route_preview_topk = max(route_preview_topk, max(1, int(solver_market_preview_topk)))
    route_calls, status = _build_spec_preview_route_calls(
        opportunity,
        parent_node=parent_node,
        parent_mapping=parent_mapping,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        pool_nodes=pool_nodes,
        pool_phi_fit=pool_phi_fit,
        pool_phi_probe=pool_phi_probe,
        pool_dims=pool_dims,
        max_depth=int(max_depth),
        nvars=int(nvars),
        poly_degree=int(poly_degree),
        var_dims=var_dims,
        enum_max_depth=int(enum_max_depth),
        enum_max_trees=int(enum_max_trees),
        preview_topk=int(route_preview_topk),
        max_subtree_depth=max_subtree_depth,
        complexity_penalty=float(complexity_penalty),
        family_battery_enable=bool(family_battery_enable),
        family_battery_mode=str(family_battery_mode or "outer"),
        recursive_enable=bool(recursive_enable),
        recursive_max_depth=int(recursive_max_depth),
        recursive_trigger_rel_mse=float(recursive_trigger_rel_mse),
        recursive_seed_cap=int(recursive_seed_cap),
        recursive_branch_topk=int(recursive_branch_topk),
        recursive_child_topk=int(recursive_child_topk),
        recursive_sr_enable=bool(inverse_spec_recursive_sr_enable),
        recursive_sr_preview_topk=int(inverse_spec_recursive_sr_preview_topk),
        recursive_sr_exact_budget=int(inverse_spec_recursive_sr_exact_budget),
        constant_lift_route_enable=bool(inverse_spec_constant_lift_route_enable),
        constant_lift_route_topk=int(inverse_spec_constant_lift_route_topk),
        coordinate_lift_enable=bool(inverse_spec_coordinate_lift_enable),
        coordinate_lift_topk=int(inverse_spec_coordinate_lift_topk),
        coordinate_lift_mode=str(inverse_spec_coordinate_lift_mode or "both"),
        tangent_edit_enable=bool(inverse_spec_tangent_edit_enable),
        tangent_edit_topk=int(inverse_spec_tangent_edit_topk),
        soft_edit_enable=bool(inverse_spec_soft_edit_enable),
        soft_edit_steps=int(inverse_spec_soft_edit_steps),
        soft_edit_l1=float(inverse_spec_soft_edit_l1),
        witness_jets_enable=bool(inverse_spec_witness_jets_enable),
        witness_d2_enable=bool(inverse_spec_witness_d2_enable),
        witness_max_rows=int(max(4, int(inverse_spec_witness_max_rows))),
        witness_loss_enable=bool(inverse_spec_witness_loss_enable),
        witness_grad_weight=float(inverse_spec_witness_grad_weight),
        witness_d2_weight=float(inverse_spec_witness_d2_weight),
        witness_diag_weight=float(inverse_spec_witness_diag_weight),
        witness_physics_weight=float(inverse_spec_witness_physics_weight),
        active_var_screen_enable=bool(inverse_spec_active_var_screen_enable),
        active_var_grad_tol=float(inverse_spec_active_var_grad_tol),
        active_var_max_count=max(1, int(inverse_spec_active_var_max_count)),
        directional_market_enable=bool(inverse_spec_directional_market_enable),
        safe_eps=float(safe_eps),
        confidence_mode=str(confidence_mode),
        confidence_target_gain=float(confidence_target_gain),
        confidence_floor=float(confidence_floor),
        branch_beam_width=int(branch_beam_width),
        beam_cfg=cfg,
        slate_prefix="hole_search" if spec_kind == "path_hole" else "hole_search_followup",
        regime_metadata=inverse_spec_regime_metadata,
    )
    if not route_calls:
        return _return(None, status=status)
    if bool(solver_market_enable):
        market_kwargs = {
            "preview_topk": max(1, int(solver_market_preview_topk)),
            "exact_topk": max(1, int(solver_market_exact_topk)),
        }
        if bool(solver_market_proposal_objects_enable):
            market_kwargs["proposal_objects_enable"] = True
        spec_result = run_preview_solver_market(route_calls, **market_kwargs)
    else:
        spec_result = route_calls[0].runner(**dict(route_calls[0].runner_kwargs or {}))
    preview_rows = [row for row in (spec_result.get("rows", []) or []) if isinstance(row, dict)]
    solver_meta = spec_result.get("solver_meta", {}) or {}
    meta["hole_search_preview_count"] = int(len(preview_rows))
    meta["hole_search_periodic_used"] = bool(solver_meta.get("periodic_forward_used", False))
    meta["hole_search_solver_meta"] = dict(solver_meta)
    meta["hole_search_followup_spec_states"] = _normalize_followup_spec_rows(
        [
            dict(row)
            for row in list(solver_meta.get("child_spec_states", []) or [])
            if isinstance(row, Mapping)
        ],
        opportunity=opportunity,
    )
    meta["hole_search_followup_spec_state_count"] = int(len(list(meta.get("hole_search_followup_spec_states", []) or [])))
    meta["hole_search_solver_market_route_count"] = int(solver_meta.get("solver_market_route_count", 0) or 0)
    meta["hole_search_solver_market_candidate_count_raw"] = int(
        solver_meta.get("solver_market_candidate_count_raw", 0) or 0
    )
    meta["hole_search_solver_market_candidate_count_unique"] = int(
        solver_meta.get("solver_market_candidate_count_unique", 0) or 0
    )
    meta["hole_search_solver_market_selected_route"] = str(
        solver_meta.get("solver_market_selected_route", "") or ""
    )
    meta["hole_search_solver_market_selected_method_name"] = str(
        solver_meta.get("solver_market_selected_method_name", "") or ""
    )
    meta["hole_search_solver_market_selected_subroute"] = str(
        solver_meta.get("solver_market_selected_subroute", "") or ""
    )
    meta["hole_search_solver_market_selected_proposal"] = dict(
        solver_meta.get("solver_market_selected_proposal", {}) or {}
    )
    meta["hole_search_solver_market_directional_order"] = [
        str(row.get("route_name", "") or "")
        for row in list(solver_meta.get("solver_market_routes", []) or [])
        if isinstance(row, Mapping)
    ]
    meta["hole_search_solver_market_routes"] = [
        dict(row)
        for row in list(solver_meta.get("solver_market_routes", []) or [])
        if isinstance(row, Mapping)
    ]

    if not preview_rows:
        return _return(None, status="no_preview_candidates")

    # --- Step 3: Cheap shortlist selection ---
    # The outer loop exact-scores the chosen child. Inside hole search we only
    # need a low-cost numeric shortlist check to avoid obviously bad proposals.
    exact_budget = max(1, int(exact_budget))
    if bool(solver_market_enable):
        exact_budget = min(exact_budget, max(1, int(solver_market_exact_topk)))
    shortlist_rows = [
        row
        for row in preview_rows[:exact_budget]
        if is_valid_node(row.get("expr", None))
    ]
    meta["hole_search_shortlist_count"] = int(len(shortlist_rows))
    meta["hole_search_shortlist_scored"] = 0
    best_expr = None
    best_eff_mse = float("inf")
    best_mapping = None
    best_sort_key = None
    selected_row = None
    fallback_row = shortlist_rows[0] if shortlist_rows else None

    for row in shortlist_rows:
        cand = row.get("expr", None)
        if not is_valid_node(cand):
            continue
        scored = _score_hole_search_expr(
            cand,
            x_fit=x_fit, y_fit=y_fit,
            x_probe=x_probe, y_probe=y_probe,
            poly_degree=int(poly_degree),
            mapping_kind_hint=_hole_search_mapping_kind_hint(opportunity, row),
            estimated_mapping_nparams=int(row.get("local_mapping_nparams", 0) or 0),
            complexity_penalty=float(complexity_penalty),
            proj=proj, fp_mode=fp_mode,
            q_scale=q_scale, q_clip=q_clip,
            score_expr_cfg=score_expr_cfg,
            score_expr_fn=score_expr_fn,
        )
        if scored is None:
            continue
        meta["hole_search_shortlist_scored"] = int(meta.get("hole_search_shortlist_scored", 0)) + 1
        eff_mse = float(scored["eff_mse"])
        raw_mse = float(scored["raw_mse"])
        sort_key = (
            eff_mse,
            raw_mse,
            int(node_size(cand)),
            int(row.get("local_rank", 0) or 0),
        )
        if best_sort_key is None or sort_key < best_sort_key:
            best_sort_key = sort_key
            best_eff_mse = eff_mse
            best_expr = cand
            best_mapping = scored["mapping"]
            selected_row = row

    meta["hole_search_exact_scored"] = 0
    meta["hole_search_best_eff_mse"] = (
        None if not math.isfinite(best_eff_mse) else float(best_eff_mse)
    )
    meta["hole_search_best_mapping"] = best_mapping

    if best_expr is None:
        if fallback_row is None:
            return _return(None, status="no_shortlist_candidate")
        best_expr = fallback_row.get("expr", None)
        if not is_valid_node(best_expr):
            return _return(None, status="malformed_preview_candidate")
        selected_row = fallback_row
        meta["hole_search_selected_preview_only"] = True
        best_eff_mse = _hole_search_preview_eff_mse(
            fallback_row,
            complexity_penalty=float(complexity_penalty),
            default_nparams=int(fallback_row.get("local_mapping_nparams", 0) or 0),
        )
        meta["hole_search_best_eff_mse"] = (
            None if best_eff_mse is None or not math.isfinite(float(best_eff_mse)) else float(best_eff_mse)
        )
    else:
        meta["hole_search_selected_preview_only"] = False

    if not is_valid_node(best_expr):
        return _return(None, status="malformed_shortlist_candidate")

    if isinstance(selected_row, Mapping):
        meta["hole_search_selected_local_mapping_kind"] = str(
            selected_row.get("local_mapping_kind", "") or ""
        )
        probe_mse = selected_row.get("local_probe_mse", None)
        try:
            meta["hole_search_selected_preview_probe_mse"] = None if probe_mse is None else float(probe_mse)
        except Exception:
            meta["hole_search_selected_preview_probe_mse"] = None

    return _return(best_expr, status="ok")


def _hole_search_mapping_kind_hint(opportunity: HoleOpportunity, row: Mapping[str, Any] | None) -> str:
    row = row if isinstance(row, Mapping) else {}
    for value in (
        row.get("local_mapping_kind", ""),
        row.get("target_mapping_kind", ""),
        getattr(opportunity, "target_mapping_kind", ""),
    ):
        kind = str(value or "").strip().lower()
        if kind:
            return kind
    return "affine"


def _hole_search_preview_eff_mse(
    row: Mapping[str, Any] | None,
    *,
    complexity_penalty: float,
    default_nparams: int = 0,
) -> float | None:
    row = row if isinstance(row, Mapping) else {}
    probe_mse = row.get("local_probe_mse", None)
    try:
        raw_mse = None if probe_mse is None else float(probe_mse)
    except Exception:
        raw_mse = None
    cand = row.get("expr", None)
    if raw_mse is None or not math.isfinite(raw_mse) or cand is None:
        return None
    try:
        size = int(node_size(cand))
    except Exception:
        size = 0
    try:
        nparams = int(row.get("local_mapping_nparams", default_nparams) or default_nparams)
    except Exception:
        nparams = int(default_nparams)
    return float(raw_mse + float(complexity_penalty) * float(size + max(0, nparams)))


def _fit_mapping_with_hint(pred, y, *, poly_degree: int, mapping_kind_hint: str):
    kind = str(mapping_kind_hint or "").strip().lower()

    def _fit_affine():
        res = fit_poly(pred, y, 1)
        if res is None:
            return None
        coeffs, mu, std = res
        return {"kind": "poly", "coeffs": coeffs, "mu": mu, "std": std}

    if kind in ("", "identity", "affine", "lin", "linear", "mono", "monomial", "strict", "direct"):
        return _fit_affine()
    if kind == "poly":
        res = fit_poly(pred, y, max(1, int(poly_degree)))
        if res is not None:
            coeffs, mu, std = res
            return {"kind": "poly", "coeffs": coeffs, "mu": mu, "std": std}
        return _fit_affine()
    if kind == "power":
        return fit_power(pred, y) or _fit_affine()
    if kind == "pade":
        return fit_pade(pred, y) or _fit_affine()
    if kind == "sine":
        return fit_sine(pred, y) or _fit_affine()
    if kind == "exp":
        return fit_exp_mapping(pred, y) or _fit_affine()
    return _fit_affine()


def _score_hole_search_expr(
    cand,
    *,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    poly_degree: int,
    mapping_kind_hint: str = "",
    estimated_mapping_nparams: int = 0,
    complexity_penalty: float = 0.0,
    proj=None,
    fp_mode: str = "bits",
    q_scale: float = 2.0,
    q_clip: float = 8.0,
    score_expr_cfg: dict[str, Any] | None = None,
    score_expr_fn=None,
) -> dict[str, Any] | None:
    """Cheap shortlist score for a candidate expression against full data."""
    try:
        pf = eval_node(cand, x_fit)
        pp = eval_node(cand, x_probe)
    except Exception:
        return None
    if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
        return None

    mapping = _fit_mapping_with_hint(
        pf,
        y_fit,
        poly_degree=int(poly_degree),
        mapping_kind_hint=str(mapping_kind_hint or ""),
    )
    if mapping is None:
        return None
    try:
        yh = eval_mapping(pp, mapping)
    except Exception:
        return None
    if not torch.isfinite(yh).all():
        return None
    mse = mean_squared_error_same_shape(y_probe, yh)

    if mapping is None or not math.isfinite(float(mse)):
        return None

    map_cost = float(mapping_cost(mapping))
    if not math.isfinite(map_cost):
        map_cost = float(max(0, int(estimated_mapping_nparams)))
    mse_eff = float(mse + float(complexity_penalty) * float(node_size(cand) + map_cost))
    return {
        "expr": cand,
        "raw_mse": float(mse),
        "eff_mse": float(mse_eff),
        "mapping": mapping,
    }


__all__ = [
    "HoleOpportunity",
    "HoleFrontier",
    "abstract_frontier_from_parent",
    "export_hole_opportunity_rows",
    "mine_frontier_from_archive",
    "run_hole_search_action",
]
def _spec_direction_from_payload(spec_payload: Mapping[str, Any] | None) -> str:
    spec = deserialize_subproblem_spec(spec_payload)
    if spec is None:
        return ""
    return str(spec.direction or "")


def _opportunity_direction(opportunity: HoleOpportunity | None) -> str:
    if opportunity is None:
        return ""
    direction = str(getattr(opportunity, "direction", "") or "")
    if direction:
        return direction
    return _spec_direction_from_payload(getattr(opportunity, "spec_payload", None))
