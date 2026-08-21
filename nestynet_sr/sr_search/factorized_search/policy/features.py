# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Policy-facing factorized symbolic search controller feature records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..engine.signals import (
    CandidateStateFeatures,
    InverseSteeringPotential,
    PathStateFeatures,
    _clamp01,
    _safe_bool,
    _safe_float,
    _safe_int,
    _safe_log1p,
    _safe_neg_log10,
    _safe_str,
    path_concentration,
    path_distribution_metrics,
    path_summary_stats,
    summarize_path_rows,
)


@dataclass(frozen=True)
class ParentStateFeatures:
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

    @classmethod
    def from_flat_row(cls, row: Mapping[str, Any] | None) -> ParentStateFeatures:
        row = row if isinstance(row, Mapping) else {}
        return cls(
            parent_key=_safe_str(row.get("parent_key", "")),
            parent_expr=_safe_str(row.get("parent_expr", "")),
            parent_root_op=_safe_str(row.get("parent_root_op", "")),
            parent_depth=_safe_int(row.get("parent_depth", 0), 0),
            parent_size=_safe_int(row.get("parent_size", 0), 0),
            parent_best_eff_mse=_safe_float(row.get("parent_best_eff_mse", float("inf")), float("inf")),
            parent_best_raw_mse=_safe_float(row.get("parent_best_raw_mse", float("inf")), float("inf")),
            parent_visits=_safe_float(row.get("parent_visits", 0.0), 0.0),
            parent_visits_since_improve=_safe_float(row.get("parent_visits_since_improve", 0.0), 0.0),
            parent_stagnation_score=_safe_float(row.get("parent_stagnation_score", 0.0), 0.0),
            parent_stagnation_ratio=_safe_float(row.get("parent_stagnation_ratio", 0.0), 0.0),
        )

    def to_flat_dict(self) -> dict[str, Any]:
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
        }


@dataclass(frozen=True)
class RepairControllerFeatureRecord:
    parent: ParentStateFeatures = field(default_factory=ParentStateFeatures)
    path: PathStateFeatures | None = None
    candidate: CandidateStateFeatures = field(default_factory=CandidateStateFeatures)
    gate_allowed: bool = False
    gate_reason: str = ""
    path_entropy: float | None = None
    path_top_mass: float | None = None
    path_second_mass: float | None = None
    path_positive_count: float | None = None
    identity_vs_full_log_mse_contrast: float | None = None
    affine_vs_full_log_mse_contrast: float | None = None
    identity_best_alt_probe_mse: float | None = None
    affine_best_alt_probe_mse: float | None = None
    full_best_alt_probe_mse: float | None = None
    path_summary_count: int = 0
    path_summary_gain_mass: float | None = None
    path_summary_gap: float | None = None
    path_summary_support: float | None = None
    path_summary_mode_diversity: float | None = None
    path_summaries: tuple[PathStateFeatures, ...] = ()
    refine_slot_count: int = 0
    refine_gate_potential: float = 0.0
    refine_variant_count: int = 0

    @classmethod
    def from_parts(
        cls,
        *,
        parent: ParentStateFeatures | None = None,
        gate_diag: InverseSteeringPotential | None = None,
        candidate_meta: Mapping[str, Any] | None = None,
        proxy_potential: float | None = None,
        refine_features: Mapping[str, Any] | None = None,
        top_k_paths: int = 4,
    ) -> RepairControllerFeatureRecord:
        diag = gate_diag if isinstance(gate_diag, InverseSteeringPotential) else InverseSteeringPotential()
        candidate_row = dict(candidate_meta or {})
        if proxy_potential is not None:
            candidate_row["proxy_one_hole_potential_eff"] = float(proxy_potential)
        best_row = diag.best_path_row
        path_metrics = path_distribution_metrics(diag.path_rows)
        mode_contrast = best_row.mode_contrast_dict() if isinstance(best_row, PathStateFeatures) else {
            "identity_vs_full_log_mse_contrast": None,
            "affine_vs_full_log_mse_contrast": None,
            "identity_best_alt_probe_mse": None,
            "affine_best_alt_probe_mse": None,
            "full_best_alt_probe_mse": None,
        }
        refine_row = refine_features if isinstance(refine_features, Mapping) else {}
        path_summaries = summarize_path_rows(diag.path_rows, top_k=top_k_paths)
        if (not path_summaries) and isinstance(best_row, PathStateFeatures):
            path_summaries = (best_row,)
        summary_stats = path_summary_stats(diag.path_rows, top_k=top_k_paths)
        return cls(
            parent=parent if isinstance(parent, ParentStateFeatures) else ParentStateFeatures(),
            path=best_row,
            candidate=CandidateStateFeatures.from_mapping(candidate_row),
            gate_allowed=bool(diag.allowed),
            gate_reason=str(diag.reason),
            path_entropy=path_metrics.get("path_entropy", None),
            path_top_mass=path_metrics.get("path_top_mass", None),
            path_second_mass=path_metrics.get("path_second_mass", None),
            path_positive_count=path_metrics.get("path_positive_count", None),
            identity_vs_full_log_mse_contrast=mode_contrast.get("identity_vs_full_log_mse_contrast", None),
            affine_vs_full_log_mse_contrast=mode_contrast.get("affine_vs_full_log_mse_contrast", None),
            identity_best_alt_probe_mse=mode_contrast.get("identity_best_alt_probe_mse", None),
            affine_best_alt_probe_mse=mode_contrast.get("affine_best_alt_probe_mse", None),
            full_best_alt_probe_mse=mode_contrast.get("full_best_alt_probe_mse", None),
            path_summary_count=int(len(tuple(diag.path_rows or ()))),
            path_summary_gain_mass=_safe_float(summary_stats.get("path_summary_gain_mass", 0.0), 0.0),
            path_summary_gap=_safe_float(summary_stats.get("path_summary_gap", 0.0), 0.0),
            path_summary_support=_safe_float(summary_stats.get("path_summary_support", 0.0), 0.0),
            path_summary_mode_diversity=_safe_float(summary_stats.get("path_summary_mode_diversity", 0.0), 0.0),
            path_summaries=tuple(path_summaries),
            refine_slot_count=max(0, _safe_int(refine_row.get("refine_slot_count", 0), 0)),
            refine_gate_potential=max(0.0, _safe_float(refine_row.get("refine_gate_potential", 0.0), 0.0)),
            refine_variant_count=max(0, _safe_int(refine_row.get("refine_variant_count", 0), 0)),
        )

    @classmethod
    def from_flat_row(cls, row: Mapping[str, Any] | None) -> RepairControllerFeatureRecord:
        row = row if isinstance(row, Mapping) else {}
        path = PathStateFeatures.from_gate_features(row)
        path_summaries_raw = row.get("path_summaries", ()) or ()
        path_summaries = tuple(
            PathStateFeatures.from_row(item)
            for item in path_summaries_raw
            if isinstance(item, Mapping)
        )
        if (not path_summaries) and isinstance(path, PathStateFeatures):
            path_summaries = (path,)
        return cls(
            parent=ParentStateFeatures.from_flat_row(row),
            path=path,
            candidate=CandidateStateFeatures.from_mapping(row),
            gate_allowed=_safe_bool(row.get("gate_allowed", False), False),
            gate_reason=_safe_str(row.get("gate_reason", "")),
            path_entropy=None if row.get("path_entropy", None) is None else _safe_float(row.get("path_entropy", 0.0), 0.0),
            path_top_mass=None if row.get("path_top_mass", None) is None else _safe_float(row.get("path_top_mass", 0.0), 0.0),
            path_second_mass=None if row.get("path_second_mass", None) is None else _safe_float(row.get("path_second_mass", 0.0), 0.0),
            path_positive_count=None if row.get("path_positive_count", None) is None else _safe_float(row.get("path_positive_count", 0.0), 0.0),
            identity_vs_full_log_mse_contrast=None
            if row.get("identity_vs_full_log_mse_contrast", None) is None
            else _safe_float(row.get("identity_vs_full_log_mse_contrast", 0.0), 0.0),
            affine_vs_full_log_mse_contrast=None
            if row.get("affine_vs_full_log_mse_contrast", None) is None
            else _safe_float(row.get("affine_vs_full_log_mse_contrast", 0.0), 0.0),
            identity_best_alt_probe_mse=None
            if row.get("identity_best_alt_probe_mse", None) is None
            else _safe_float(row.get("identity_best_alt_probe_mse", float("inf")), float("inf")),
            affine_best_alt_probe_mse=None
            if row.get("affine_best_alt_probe_mse", None) is None
            else _safe_float(row.get("affine_best_alt_probe_mse", float("inf")), float("inf")),
            full_best_alt_probe_mse=None
            if row.get("full_best_alt_probe_mse", None) is None
            else _safe_float(row.get("full_best_alt_probe_mse", float("inf")), float("inf")),
            path_summary_count=max(0, _safe_int(row.get("path_summary_count", len(path_summaries)), len(path_summaries))),
            path_summary_gain_mass=None
            if row.get("path_summary_gain_mass", None) is None
            else _safe_float(row.get("path_summary_gain_mass", 0.0), 0.0),
            path_summary_gap=None
            if row.get("path_summary_gap", None) is None
            else _safe_float(row.get("path_summary_gap", 0.0), 0.0),
            path_summary_support=None
            if row.get("path_summary_support", None) is None
            else _safe_float(row.get("path_summary_support", 0.0), 0.0),
            path_summary_mode_diversity=None
            if row.get("path_summary_mode_diversity", None) is None
            else _safe_float(row.get("path_summary_mode_diversity", 0.0), 0.0),
            path_summaries=path_summaries,
            refine_slot_count=max(0, _safe_int(row.get("refine_slot_count", 0), 0)),
            refine_gate_potential=max(0.0, _safe_float(row.get("refine_gate_potential", 0.0), 0.0)),
            refine_variant_count=max(0, _safe_int(row.get("refine_variant_count", 0), 0)),
        )

    @property
    def path_rows(self) -> tuple[PathStateFeatures, ...]:
        if self.path_summaries:
            return tuple(self.path_summaries)
        if isinstance(self.path, PathStateFeatures):
            return (self.path,)
        return ()

    @property
    def selected_target_mode(self) -> str:
        if self.candidate.selected_target_mode is not None:
            return _safe_str(self.candidate.selected_target_mode)
        if isinstance(self.path, PathStateFeatures):
            return _safe_str(self.path.target_mode)
        return ""

    def path_summary_stats(self, *, top_k: int = 4) -> dict[str, Any]:
        out = path_summary_stats(self.path_rows, top_k=top_k)
        out["path_summary_count"] = int(max(self.path_summary_count, len(self.path_rows)))
        if self.path_summary_gain_mass is not None:
            out["path_summary_gain_mass"] = float(self.path_summary_gain_mass)
        if self.path_summary_gap is not None:
            out["path_summary_gap"] = float(self.path_summary_gap)
        if self.path_summary_support is not None:
            out["path_summary_support"] = float(self.path_summary_support)
        if self.path_summary_mode_diversity is not None:
            out["path_summary_mode_diversity"] = float(self.path_summary_mode_diversity)
        return out

    def repair_potential(self) -> float:
        potential = self.candidate.estimated_one_hole_rel_improve_eff
        if potential is None:
            potential = self.candidate.proxy_one_hole_potential_eff
        return _clamp01(potential)

    def repair_contrast(self) -> float:
        contrast = self.identity_vs_full_log_mse_contrast
        if contrast is None:
            contrast = self.affine_vs_full_log_mse_contrast
        return max(0.0, _safe_float(contrast, 0.0))

    def repair_path_concentration(self) -> float:
        return _clamp01(path_concentration(self.path_entropy, self.path_top_mass, self.path_positive_count))

    def analytic_components(self) -> dict[str, float]:
        candidate_k = max(0.0, _safe_float(self.candidate.local_candidate_count, 0.0))
        cost = min(1.0, math.log1p(candidate_k) / math.log(17.0))
        return {
            "analytic_potential": float(self.repair_potential()),
            "analytic_concentration": float(self.repair_path_concentration()),
            "analytic_contrast": float(math.tanh(self.repair_contrast() / 3.0)),
            "analytic_cost": float(cost),
            "analytic_stagnation": float(_clamp01(self.parent.parent_stagnation_score)),
        }

    def to_repair_critic_features(self) -> dict[str, float]:
        summary_stats = self.path_summary_stats()
        analytic = self.analytic_components()
        selected_mode = self.selected_target_mode.strip().lower()
        path = self.path if isinstance(self.path, PathStateFeatures) else PathStateFeatures()
        n_pos = max(0.0, _safe_float(self.path_positive_count, 0.0))
        if n_pos > 1.0:
            h_max = max(math.log(n_pos), 1.0e-12)
            path_entropy_norm = min(1.0, max(0.0, _safe_float(self.path_entropy, 0.0) / h_max))
        else:
            path_entropy_norm = 0.0
        out = {
            "one_hole_potential": float(_clamp01(self.candidate.estimated_one_hole_rel_improve_eff)),
            "proxy_one_hole_potential": float(_clamp01(self.candidate.proxy_one_hole_potential_eff)),
            "path_top_mass": float(_clamp01(self.path_top_mass)),
            "path_second_mass": float(_clamp01(self.path_second_mass)),
            "path_entropy_norm": float(path_entropy_norm),
            "path_positive_log": float(_safe_log1p(self.path_positive_count)),
            "path_summary_gain_mass": float(_clamp01(summary_stats.get("path_summary_gain_mass", 0.0))),
            "path_summary_gap": float(_clamp01(summary_stats.get("path_summary_gap", 0.0))),
            "path_summary_support": float(_clamp01(summary_stats.get("path_summary_support", 0.0))),
            "path_summary_mode_diversity": float(_clamp01(summary_stats.get("path_summary_mode_diversity", 0.0))),
            "gate_best_weighted_rel_gain": float(_safe_log1p(path.weighted_rel_gain)),
            "gate_best_rel_gain": float(_safe_log1p(path.rel_gain)),
            "gate_best_valid_frac": float(_clamp01(path.valid_frac)),
            "gate_best_confidence": float(_clamp01(path.confidence)),
            "gate_best_transport_rel": float(max(0.0, _safe_float(path.transport_rel, 0.0))),
            "gate_best_static_score": float(max(0.0, _safe_float(path.static_score, 0.0))),
            "gate_best_branch_factor": float(max(0.0, _safe_float(path.branch_factor, 0.0))),
            "gate_best_cut_factor": float(max(0.0, _safe_float(path.cut_factor, 0.0))),
            "selected_path_gain": float(_safe_log1p(self.candidate.selected_path_gain)),
            "selected_path_gain_pre_cut": float(_safe_log1p(self.candidate.selected_path_gain_pre_cut)),
            "selected_rel_gain": float(max(0.0, _safe_float(self.candidate.selected_rel_gain, 0.0))),
            "selected_transport_rel": float(max(0.0, _safe_float(self.candidate.selected_transport_rel, 0.0))),
            "selected_lin_rel": float(max(0.0, _safe_float(self.candidate.selected_lin_rel, 0.0))),
            "selected_branch_factor": float(max(0.0, _safe_float(self.candidate.selected_branch_factor, 0.0))),
            "selected_cut_factor": float(max(0.0, _safe_float(self.candidate.selected_cut_factor, 0.0))),
            "selected_effective_n_log": float(_safe_log1p(self.candidate.selected_effective_n)),
            "selected_mode_identity": 1.0 if selected_mode == "identity" else 0.0,
            "selected_mode_affine": 1.0 if selected_mode == "affine" else 0.0,
            "selected_mode_full": 1.0 if selected_mode == "full" else 0.0,
            "best_exact_monotone": 1.0 if bool(path.profile_exact_monotone) else 0.0,
            "best_has_periodic": 1.0 if bool(path.profile_has_periodic) else 0.0,
            "best_has_muldiv": 1.0 if bool(path.profile_has_muldiv) else 0.0,
            "best_has_explogsqrt": 1.0 if bool(path.profile_has_explogsqrt) else 0.0,
            "identity_vs_full_contrast": float(math.tanh(max(0.0, _safe_float(self.identity_vs_full_log_mse_contrast, 0.0)) / 3.0)),
            "affine_vs_full_contrast": float(math.tanh(max(0.0, _safe_float(self.affine_vs_full_log_mse_contrast, 0.0)) / 3.0)),
            "local_candidate_log": float(_safe_log1p(self.candidate.local_candidate_count)),
            "parent_best_eff_log": float(_safe_neg_log10(self.parent.parent_best_eff_mse)),
            "parent_best_raw_log": float(_safe_neg_log10(self.parent.parent_best_raw_mse)),
            "parent_stagnation_score": float(_clamp01(self.parent.parent_stagnation_score)),
            "parent_stagnation_ratio": float(_clamp01(self.parent.parent_stagnation_ratio)),
            "parent_visits_log": float(_safe_log1p(self.parent.parent_visits)),
            "parent_visits_since_improve_log": float(_safe_log1p(self.parent.parent_visits_since_improve)),
            "gate_allowed": 1.0 if bool(self.gate_allowed) else 0.0,
            "refine_slot_count": float(min(4.0, max(0.0, float(self.refine_slot_count)))),
            "refine_gate_potential": float(_clamp01(self.refine_gate_potential)),
            "refine_variant_log": float(_safe_log1p(self.refine_variant_count)),
        }
        out.update(analytic)
        return out

    def to_macro_state_payload(
        self,
        *,
        parent_key: Any = None,
        parent_expr: Any = None,
        parent_root_op: Any = None,
        parent_depth: int | None = None,
        parent_size: int | None = None,
        allowed_action_names: Sequence[str] | None = None,
        repair_priority_score: float | None = None,
        repair_gate_score: float | None = None,
        repair_threshold: float | None = None,
        repair_ready: bool = False,
        repair_preview_available: bool = False,
        repair_component_ok: bool = False,
        source: str = "",
    ) -> dict[str, Any]:
        best_row = self.path if isinstance(self.path, PathStateFeatures) else (self.path_rows[0] if self.path_rows else None)
        if best_row is not None:
            best_path = tuple(int(v) for v in best_row.path)
            best_path_gain = _safe_float(best_row.weighted_rel_gain, 0.0)
            best_valid = _safe_float(best_row.valid_frac, 0.0)
            best_conf = _safe_float(best_row.confidence, 0.0)
            best_transport = _safe_float(best_row.transport_rel, 0.0)
            best_static = _safe_float(best_row.static_score, 0.0)
        else:
            best_path = ()
            best_path_gain = 0.0
            best_valid = 0.0
            best_conf = 0.0
            best_transport = 0.0
            best_static = 0.0
        return {
            "parent_key": _safe_str(parent_key if parent_key is not None else self.parent.parent_key),
            "parent_expr": _safe_str(parent_expr if parent_expr is not None else self.parent.parent_expr),
            "parent_root_op": _safe_str(parent_root_op if parent_root_op is not None else self.parent.parent_root_op),
            "parent_depth": max(0, _safe_int(self.parent.parent_depth if parent_depth is None else parent_depth, 0)),
            "parent_size": max(0, _safe_int(self.parent.parent_size if parent_size is None else parent_size, 0)),
            "parent_best_eff_mse": _safe_float(self.parent.parent_best_eff_mse, float("inf")),
            "parent_best_raw_mse": _safe_float(self.parent.parent_best_raw_mse, float("inf")),
            "parent_visits": _safe_float(self.parent.parent_visits, 0.0),
            "parent_visits_since_improve": _safe_float(self.parent.parent_visits_since_improve, 0.0),
            "parent_stagnation_score": _safe_float(self.parent.parent_stagnation_score, 0.0),
            "parent_stagnation_ratio": _safe_float(self.parent.parent_stagnation_ratio, 0.0),
            "allowed_actions": tuple(str(name) for name in (allowed_action_names or ()) if str(name)),
            "gate_allowed": bool(self.gate_allowed),
            "gate_reason": str(self.gate_reason),
            "repair_preview_available": bool(repair_preview_available),
            "repair_component_ok": bool(repair_component_ok),
            "repair_ready": bool(repair_ready),
            "repair_priority_score": _safe_float(repair_priority_score, 0.0),
            "repair_gate_score": _safe_float(repair_gate_score, 0.0),
            "repair_threshold": _safe_float(repair_threshold, 0.0),
            "repair_potential": float(self.repair_potential()),
            "repair_path_concentration": float(self.repair_path_concentration()),
            "repair_contrast": float(self.repair_contrast()),
            "repair_candidate_count": max(0, _safe_int(self.candidate.local_candidate_count, 0)),
            "path_entropy": max(0.0, _safe_float(self.path_entropy, 0.0)),
            "path_top_mass": _clamp01(self.path_top_mass),
            "path_second_mass": _clamp01(self.path_second_mass),
            "path_positive_count": max(0.0, _safe_float(self.path_positive_count, 0.0)),
            "best_path": best_path,
            "best_path_gain": max(0.0, best_path_gain),
            "best_path_valid_frac": _clamp01(best_valid),
            "best_path_confidence": _clamp01(best_conf),
            "best_path_transport_rel": max(0.0, best_transport),
            "best_path_static_score": max(0.0, best_static),
            "selected_target_mode": self.selected_target_mode,
            "refine_slot_count": max(0, int(self.refine_slot_count)),
            "refine_gate_potential": max(0.0, _safe_float(self.refine_gate_potential, 0.0)),
            "path_rows": tuple(self.path_rows),
            "source": _safe_str(source),
        }

    def to_inverse_potential(self) -> InverseSteeringPotential:
        path_rows = self.path_rows
        best_row = self.path if isinstance(self.path, PathStateFeatures) else (path_rows[0] if path_rows else None)
        best_path = None if best_row is None else tuple(int(v) for v in best_row.path)
        return InverseSteeringPotential(
            allowed=bool(self.gate_allowed),
            reason=str(self.gate_reason),
            best_path=best_path,
            best_rel_gain=0.0 if best_row is None else float(best_row.rel_gain),
            best_weighted_rel_gain=0.0 if best_row is None else float(best_row.weighted_rel_gain),
            candidate_paths=tuple(tuple(int(v) for v in row.path) for row in path_rows),
            path_rows=tuple(path_rows),
        )

    def to_flat_dict(self) -> dict[str, Any]:
        out = self.parent.to_flat_dict()
        out.update({
            "gate_allowed": bool(self.gate_allowed),
            "gate_reason": str(self.gate_reason),
            "path_entropy": self.path_entropy,
            "path_top_mass": self.path_top_mass,
            "path_second_mass": self.path_second_mass,
            "path_positive_count": self.path_positive_count,
            "identity_vs_full_log_mse_contrast": self.identity_vs_full_log_mse_contrast,
            "affine_vs_full_log_mse_contrast": self.affine_vs_full_log_mse_contrast,
            "identity_best_alt_probe_mse": self.identity_best_alt_probe_mse,
            "affine_best_alt_probe_mse": self.affine_best_alt_probe_mse,
            "full_best_alt_probe_mse": self.full_best_alt_probe_mse,
        })
        if self.path is None:
            out["gate_best_path"] = []
        else:
            out.update(self.path.to_gate_feature_dict())
        out.update(self.candidate.to_flat_dict())
        out.update(self.path_summary_stats())
        out.update({
            "refine_slot_count": int(self.refine_slot_count),
            "refine_gate_potential": float(self.refine_gate_potential),
            "refine_variant_count": int(self.refine_variant_count),
        })
        return out


def build_controller_state_record(
    *,
    parent_key: Any = None,
    parent_expr: Any,
    parent_root_op: str = "",
    parent_depth: int = 0,
    parent_size: int = 0,
    parent_best_eff_mse: float | None = None,
    parent_best_raw_mse: float | None = None,
    parent_visits: float | None = None,
    parent_visits_since_improve: float | None = None,
    parent_stagnation_score: float | None = None,
    parent_stagnation_ratio: float | None = None,
    gate_diag: InverseSteeringPotential | None = None,
    candidate_meta: Mapping[str, Any] | None = None,
    proxy_potential: float | None = None,
    refine_features: Mapping[str, Any] | None = None,
    top_k_paths: int = 4,
) -> RepairControllerFeatureRecord:
    return RepairControllerFeatureRecord.from_parts(
        parent=ParentStateFeatures(
            parent_key=_safe_str(parent_key),
            parent_expr=_safe_str(parent_expr),
            parent_root_op=_safe_str(parent_root_op),
            parent_depth=max(0, _safe_int(parent_depth, 0)),
            parent_size=max(0, _safe_int(parent_size, 0)),
            parent_best_eff_mse=_safe_float(parent_best_eff_mse, float("inf")),
            parent_best_raw_mse=_safe_float(parent_best_raw_mse, float("inf")),
            parent_visits=_safe_float(parent_visits, 0.0),
            parent_visits_since_improve=_safe_float(parent_visits_since_improve, 0.0),
            parent_stagnation_score=_safe_float(parent_stagnation_score, 0.0),
            parent_stagnation_ratio=_safe_float(parent_stagnation_ratio, 0.0),
        ),
        gate_diag=gate_diag,
        candidate_meta=candidate_meta,
        proxy_potential=proxy_potential,
        refine_features=refine_features,
        top_k_paths=top_k_paths,
    )


def coerce_repair_feature_record(
    row: Any,
    *,
    gate_diag: InverseSteeringPotential | None = None,
    refine_features: Mapping[str, Any] | None = None,
    top_k_paths: int = 4,
) -> RepairControllerFeatureRecord:
    if isinstance(row, RepairControllerFeatureRecord):
        record = row
    elif isinstance(row, Mapping):
        record = RepairControllerFeatureRecord.from_flat_row(row)
    else:
        record = RepairControllerFeatureRecord()
    if gate_diag is None and refine_features is None:
        return record
    merged = record.to_flat_dict()
    if gate_diag is not None:
        overlay = RepairControllerFeatureRecord.from_parts(
            parent=record.parent,
            gate_diag=gate_diag,
            candidate_meta=record.candidate.to_flat_dict(),
            proxy_potential=record.candidate.proxy_one_hole_potential_eff,
            refine_features={
                "refine_slot_count": int(record.refine_slot_count),
                "refine_gate_potential": float(record.refine_gate_potential),
                "refine_variant_count": int(record.refine_variant_count),
            },
            top_k_paths=top_k_paths,
        ).to_flat_dict()
        merged["gate_allowed"] = overlay.get("gate_allowed", merged.get("gate_allowed", False))
        merged["gate_reason"] = overlay.get("gate_reason", merged.get("gate_reason", ""))
        if overlay.get("gate_best_path", None):
            for key, value in overlay.items():
                if key.startswith("gate_best_"):
                    merged[key] = value
        if overlay.get("path_summaries", None):
            merged["path_summaries"] = overlay.get("path_summaries", [])
            if "path_summary_count" not in merged or merged.get("path_summary_count", 0) in (0, None):
                merged["path_summary_count"] = overlay.get("path_summary_count", 0)
        for key in (
            "path_entropy",
            "path_top_mass",
            "path_second_mass",
            "path_positive_count",
            "identity_vs_full_log_mse_contrast",
            "affine_vs_full_log_mse_contrast",
            "identity_best_alt_probe_mse",
            "affine_best_alt_probe_mse",
            "full_best_alt_probe_mse",
            "path_summary_gain_mass",
            "path_summary_gap",
            "path_summary_support",
            "path_summary_mode_diversity",
        ):
            if merged.get(key, None) is None:
                merged[key] = overlay.get(key, None)
    if isinstance(refine_features, Mapping):
        for key in ("refine_slot_count", "refine_gate_potential", "refine_variant_count"):
            if key in refine_features:
                merged[key] = refine_features[key]
    return RepairControllerFeatureRecord.from_flat_row(merged)


def coerce_repair_feature_row(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        out = dict(row)
        out.update(coerce_repair_feature_record(row).to_flat_dict())
        return out
    return coerce_repair_feature_record(row).to_flat_dict()


__all__ = [
    "ParentStateFeatures",
    "RepairControllerFeatureRecord",
    "build_controller_state_record",
    "coerce_repair_feature_record",
    "coerce_repair_feature_row",
]
