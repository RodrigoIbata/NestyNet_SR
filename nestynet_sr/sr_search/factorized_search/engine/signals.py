# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Engine-facing factorized symbolic search signals and path summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


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


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _safe_str(value: Any, default: str = "") -> str:
    try:
        return str(value)
    except Exception:
        return str(default)


def _clamp01(value: Any) -> float:
    return min(1.0, max(0.0, _safe_float(value, 0.0)))


def _safe_log1p(value: Any) -> float:
    return math.log1p(max(0.0, _safe_float(value, 0.0)))


def _safe_neg_log10(value: Any, floor: float = 1.0e-30) -> float:
    return -math.log10(max(floor, max(0.0, _safe_float(value, floor))))


def _path_tuple(path_like: Any) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in (path_like or ()))
    except Exception:
        return ()


def _path_list(path_like: Sequence[int] | None) -> list[int]:
    return [int(v) for v in (path_like or ())]


def _selected_or_action_path(row: Mapping[str, Any] | None) -> tuple[int, ...]:
    row = row if isinstance(row, Mapping) else {}
    selected = _path_tuple(row.get("selected_path", ()))
    if selected:
        return selected
    return _path_tuple(row.get("controller_action_path", ()))


def _row_first(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row.get(name, None) is not None:
            return row.get(name)
    return default


@dataclass(frozen=True)
class ModeStateFeatures:
    target_mode: str = ""
    target_mapping_kind: str = ""
    weighted_rel_gain: float = 0.0
    weighted_rel_gain_raw: float = 0.0
    rel_gain: float = 0.0
    best_alt_probe_mse: float = float("inf")
    cur_probe_mse: float = float("inf")
    confidence: float = 0.0
    valid_frac: float = 0.0

    @classmethod
    def from_row(cls, row: Mapping[str, Any] | None) -> ModeStateFeatures:
        row = row if isinstance(row, Mapping) else {}
        return cls(
            target_mode=_safe_str(row.get("target_mode", "")),
            target_mapping_kind=_safe_str(row.get("target_mapping_kind", "")),
            weighted_rel_gain=_safe_float(row.get("weighted_rel_gain", 0.0), 0.0),
            weighted_rel_gain_raw=_safe_float(row.get("weighted_rel_gain_raw", 0.0), 0.0),
            rel_gain=_safe_float(row.get("rel_gain", 0.0), 0.0),
            best_alt_probe_mse=_safe_float(row.get("best_alt_probe_mse", float("inf")), float("inf")),
            cur_probe_mse=_safe_float(row.get("cur_probe_mse", float("inf")), float("inf")),
            confidence=_safe_float(row.get("confidence", 0.0), 0.0),
            valid_frac=_safe_float(row.get("valid_frac", 0.0), 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_mode": str(self.target_mode),
            "target_mapping_kind": str(self.target_mapping_kind),
            "weighted_rel_gain": float(self.weighted_rel_gain),
            "weighted_rel_gain_raw": float(self.weighted_rel_gain_raw),
            "rel_gain": float(self.rel_gain),
            "best_alt_probe_mse": float(self.best_alt_probe_mse),
            "cur_probe_mse": float(self.cur_probe_mse),
            "confidence": float(self.confidence),
            "valid_frac": float(self.valid_frac),
        }


@dataclass(frozen=True)
class PathStateFeatures:
    path: tuple[int, ...] = ()
    branch_id: str = ""
    static_score: float = 0.0
    transport_rel: float = 0.0
    transport_factor: float = 1.0
    nonadditive: int = 0
    valid_frac: float = 0.0
    confidence: float = 0.0
    cur_probe_mse: float = float("inf")
    best_alt_probe_mse: float = float("inf")
    rel_gain: float = 0.0
    weighted_rel_gain_raw: float = 0.0
    target_mode: str = ""
    target_mode_factor: float = 1.0
    target_mapping_kind: str = ""
    weighted_rel_gain: float = 0.0
    branch_factor: float = 1.0
    branch_support: float = 1.0
    branch_positive_count: int = 1
    family_scale: float = 1.0
    min_valid_frac_eff: float = 0.0
    min_confidence_eff: float = 0.0
    profile_has_periodic: bool = False
    profile_has_muldiv: bool = False
    profile_has_explogsqrt: bool = False
    profile_exact_monotone: bool = False
    cut_factor: float = 1.0
    weighted_rel_gain_pre_cut: float = 0.0
    mode_rows: tuple[ModeStateFeatures, ...] = ()
    oracle_relation_to_reference: str = ""
    oracle_is_reference_path: bool = False
    oracle_best_mode: str = ""
    oracle_truth_present_under_path: bool = False
    oracle_best_truth_rank: int = -1
    oracle_second_truth_rank: int = -1
    oracle_truth_rank_margin: float = 0.0

    @classmethod
    def from_row(cls, row: Mapping[str, Any] | None) -> PathStateFeatures:
        row = row if isinstance(row, Mapping) else {}
        mode_rows_raw = row.get("mode_rows", ()) or ()
        return cls(
            path=_path_tuple(row.get("path", ())),
            branch_id=_safe_str(row.get("branch_id", "")),
            static_score=_safe_float(row.get("static_score", 0.0), 0.0),
            transport_rel=_safe_float(row.get("transport_rel", 0.0), 0.0),
            transport_factor=_safe_float(row.get("transport_factor", 1.0), 1.0),
            nonadditive=_safe_int(row.get("nonadditive", 0), 0),
            valid_frac=_safe_float(_row_first(row, "valid_frac", default=0.0), 0.0),
            confidence=_safe_float(_row_first(row, "confidence", default=0.0), 0.0),
            cur_probe_mse=_safe_float(row.get("cur_probe_mse", float("inf")), float("inf")),
            best_alt_probe_mse=_safe_float(
                _row_first(row, "best_alt_probe_mse", "best_alt_mse", default=float("inf")),
                float("inf"),
            ),
            rel_gain=_safe_float(row.get("rel_gain", 0.0), 0.0),
            weighted_rel_gain_raw=_safe_float(
                _row_first(row, "weighted_rel_gain_raw", "path_gain_raw", "gain_raw", default=0.0),
                0.0,
            ),
            target_mode=_safe_str(row.get("target_mode", "")),
            target_mode_factor=_safe_float(row.get("target_mode_factor", 1.0), 1.0),
            target_mapping_kind=_safe_str(row.get("target_mapping_kind", "")),
            weighted_rel_gain=_safe_float(_row_first(row, "weighted_rel_gain", "path_gain", default=0.0), 0.0),
            branch_factor=_safe_float(row.get("branch_factor", 1.0), 1.0),
            branch_support=_safe_float(row.get("branch_support", 1.0), 1.0),
            branch_positive_count=_safe_int(row.get("branch_positive_count", 1), 1),
            family_scale=_safe_float(row.get("family_scale", 1.0), 1.0),
            min_valid_frac_eff=_safe_float(row.get("min_valid_frac_eff", 0.0), 0.0),
            min_confidence_eff=_safe_float(row.get("min_confidence_eff", 0.0), 0.0),
            profile_has_periodic=_safe_bool(row.get("profile_has_periodic", False), False),
            profile_has_muldiv=_safe_bool(row.get("profile_has_muldiv", False), False),
            profile_has_explogsqrt=_safe_bool(row.get("profile_has_explogsqrt", False), False),
            profile_exact_monotone=_safe_bool(row.get("profile_exact_monotone", False), False),
            cut_factor=_safe_float(_row_first(row, "cut_factor", "path_cut_factor", default=1.0), 1.0),
            weighted_rel_gain_pre_cut=_safe_float(
                _row_first(row, "weighted_rel_gain_pre_cut", "path_gain_pre_cut", default=0.0),
                0.0,
            ),
            mode_rows=tuple(ModeStateFeatures.from_row(mr) for mr in mode_rows_raw),
            oracle_relation_to_reference=_safe_str(row.get("oracle_relation_to_reference", "")),
            oracle_is_reference_path=_safe_bool(row.get("oracle_is_reference_path", False), False),
            oracle_best_mode=_safe_str(row.get("oracle_best_mode", "")),
            oracle_truth_present_under_path=_safe_bool(row.get("oracle_truth_present_under_path", False), False),
            oracle_best_truth_rank=_safe_int(row.get("oracle_best_truth_rank", -1), -1),
            oracle_second_truth_rank=_safe_int(row.get("oracle_second_truth_rank", -1), -1),
            oracle_truth_rank_margin=_safe_float(row.get("oracle_truth_rank_margin", 0.0), 0.0),
        )

    @classmethod
    def from_gate_features(cls, row: Mapping[str, Any] | None) -> PathStateFeatures | None:
        row = row if isinstance(row, Mapping) else {}
        path = _path_tuple(row.get("gate_best_path", ()))
        if not path and ("gate_best_weighted_rel_gain" not in row):
            return None
        return cls(
            path=path,
            static_score=_safe_float(row.get("gate_best_static_score", 0.0), 0.0),
            transport_rel=_safe_float(row.get("gate_best_transport_rel", 0.0), 0.0),
            valid_frac=_safe_float(row.get("gate_best_valid_frac", 0.0), 0.0),
            confidence=_safe_float(row.get("gate_best_confidence", 0.0), 0.0),
            rel_gain=_safe_float(row.get("gate_best_rel_gain", 0.0), 0.0),
            weighted_rel_gain=_safe_float(row.get("gate_best_weighted_rel_gain", 0.0), 0.0),
            branch_factor=_safe_float(row.get("gate_best_branch_factor", 1.0), 1.0),
            cut_factor=_safe_float(row.get("gate_best_cut_factor", 1.0), 1.0),
            profile_has_periodic=_safe_bool(row.get("gate_best_profile_has_periodic", False), False),
            profile_has_muldiv=_safe_bool(row.get("gate_best_profile_has_muldiv", False), False),
            profile_has_explogsqrt=_safe_bool(row.get("gate_best_profile_has_explogsqrt", False), False),
            profile_exact_monotone=_safe_bool(row.get("gate_best_profile_exact_monotone", False), False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": _path_list(self.path),
            "branch_id": str(self.branch_id),
            "static_score": float(self.static_score),
            "transport_rel": float(self.transport_rel),
            "transport_factor": float(self.transport_factor),
            "nonadditive": int(self.nonadditive),
            "valid_frac": float(self.valid_frac),
            "confidence": float(self.confidence),
            "cur_probe_mse": float(self.cur_probe_mse),
            "best_alt_probe_mse": float(self.best_alt_probe_mse),
            "rel_gain": float(self.rel_gain),
            "weighted_rel_gain_raw": float(self.weighted_rel_gain_raw),
            "target_mode": str(self.target_mode),
            "target_mode_factor": float(self.target_mode_factor),
            "target_mapping_kind": str(self.target_mapping_kind),
            "weighted_rel_gain": float(self.weighted_rel_gain),
            "branch_factor": float(self.branch_factor),
            "branch_support": float(self.branch_support),
            "branch_positive_count": int(self.branch_positive_count),
            "family_scale": float(self.family_scale),
            "min_valid_frac_eff": float(self.min_valid_frac_eff),
            "min_confidence_eff": float(self.min_confidence_eff),
            "profile_has_periodic": bool(self.profile_has_periodic),
            "profile_has_muldiv": bool(self.profile_has_muldiv),
            "profile_has_explogsqrt": bool(self.profile_has_explogsqrt),
            "profile_exact_monotone": bool(self.profile_exact_monotone),
            "cut_factor": float(self.cut_factor),
            "weighted_rel_gain_pre_cut": float(self.weighted_rel_gain_pre_cut),
            "mode_rows": [mr.to_dict() for mr in self.mode_rows],
            "oracle_relation_to_reference": str(self.oracle_relation_to_reference),
            "oracle_is_reference_path": bool(self.oracle_is_reference_path),
            "oracle_best_mode": str(self.oracle_best_mode),
            "oracle_truth_present_under_path": bool(self.oracle_truth_present_under_path),
            "oracle_best_truth_rank": int(self.oracle_best_truth_rank),
            "oracle_second_truth_rank": int(self.oracle_second_truth_rank),
            "oracle_truth_rank_margin": float(self.oracle_truth_rank_margin),
        }

    def to_gate_feature_dict(self) -> dict[str, Any]:
        return {
            "gate_best_path": _path_list(self.path),
            "gate_best_weighted_rel_gain": float(self.weighted_rel_gain),
            "gate_best_rel_gain": float(self.rel_gain),
            "gate_best_valid_frac": float(self.valid_frac),
            "gate_best_confidence": float(self.confidence),
            "gate_best_transport_rel": float(self.transport_rel),
            "gate_best_static_score": float(self.static_score),
            "gate_best_branch_factor": float(self.branch_factor),
            "gate_best_cut_factor": float(self.cut_factor),
            "gate_best_profile_exact_monotone": bool(self.profile_exact_monotone),
            "gate_best_profile_has_periodic": bool(self.profile_has_periodic),
            "gate_best_profile_has_muldiv": bool(self.profile_has_muldiv),
            "gate_best_profile_has_explogsqrt": bool(self.profile_has_explogsqrt),
        }

    def mode_row(self, mode_name: str) -> ModeStateFeatures | None:
        mode_key = str(mode_name or "").strip().lower()
        for row in self.mode_rows:
            if str(row.target_mode).strip().lower() == mode_key:
                return row
        return None

    def mode_contrast_dict(self) -> dict[str, float | None]:
        m_id = self.mode_row("identity")
        m_aff = self.mode_row("affine")
        m_full = self.mode_row("full")
        out = {
            "identity_vs_full_log_mse_contrast": None,
            "affine_vs_full_log_mse_contrast": None,
            "identity_best_alt_probe_mse": None,
            "affine_best_alt_probe_mse": None,
            "full_best_alt_probe_mse": None,
        }
        if m_id is not None:
            out["identity_best_alt_probe_mse"] = float(m_id.best_alt_probe_mse)
        if m_aff is not None:
            out["affine_best_alt_probe_mse"] = float(m_aff.best_alt_probe_mse)
        if m_full is not None:
            out["full_best_alt_probe_mse"] = float(m_full.best_alt_probe_mse)
        eps = 1.0e-30
        if m_id is not None and m_full is not None:
            out["identity_vs_full_log_mse_contrast"] = float(
                math.log(float(m_full.best_alt_probe_mse) + eps)
                - math.log(float(m_id.best_alt_probe_mse) + eps)
            )
        if m_aff is not None and m_full is not None:
            out["affine_vs_full_log_mse_contrast"] = float(
                math.log(float(m_full.best_alt_probe_mse) + eps)
                - math.log(float(m_aff.best_alt_probe_mse) + eps)
            )
        return out


def path_distribution_metrics(
    path_rows: Sequence[PathStateFeatures | Mapping[str, Any]] | None,
    *,
    eps: float = 1.0e-30,
) -> dict[str, float]:
    rows = list(path_rows or ())
    vals: list[float] = []
    for row in rows:
        if isinstance(row, PathStateFeatures):
            v = float(row.weighted_rel_gain)
        elif isinstance(row, Mapping):
            v = _safe_float(row.get("weighted_rel_gain", 0.0), 0.0)
        else:
            v = 0.0
        if math.isfinite(v) and v > 0.0:
            vals.append(float(v))
    if not vals:
        return {
            "path_entropy": 0.0,
            "path_top_mass": 0.0,
            "path_second_mass": 0.0,
            "path_positive_count": 0.0,
        }
    total = float(sum(vals))
    if (not math.isfinite(total)) or total <= eps:
        return {
            "path_entropy": 0.0,
            "path_top_mass": 0.0,
            "path_second_mass": 0.0,
            "path_positive_count": float(len(vals)),
        }
    probs = [float(v) / total for v in vals]
    entropy = 0.0
    for p in probs:
        if p > 0.0:
            entropy -= float(p) * math.log(float(p))
    probs_s = sorted(probs, reverse=True)
    return {
        "path_entropy": float(entropy),
        "path_top_mass": float(probs_s[0]) if probs_s else 0.0,
        "path_second_mass": float(probs_s[1]) if len(probs_s) >= 2 else 0.0,
        "path_positive_count": float(len(probs)),
    }


def path_concentration(
    path_entropy: Any,
    path_top_mass: Any,
    path_positive_count: Any,
) -> float:
    top_mass = _clamp01(path_top_mass)
    entropy = max(0.0, _safe_float(path_entropy, 0.0))
    n_pos = max(0.0, _safe_float(path_positive_count, 0.0))
    if n_pos > 1.0:
        h_max = max(math.log(n_pos), 1.0e-12)
        entropy_norm = min(1.0, max(0.0, entropy / h_max))
    else:
        entropy_norm = 0.0
    return 0.5 * top_mass + 0.5 * (1.0 - entropy_norm)


def summarize_path_rows(
    path_rows: Sequence[PathStateFeatures | Mapping[str, Any]] | None,
    *,
    top_k: int = 4,
) -> tuple[PathStateFeatures, ...]:
    rows = [
        row if isinstance(row, PathStateFeatures) else PathStateFeatures.from_row(row)
        for row in (path_rows or ())
        if isinstance(row, (PathStateFeatures, Mapping))
    ]
    rows.sort(key=lambda row: float(row.weighted_rel_gain), reverse=True)
    return tuple(rows[: max(1, int(top_k))])


def path_summary_stats(
    path_rows: Sequence[PathStateFeatures | Mapping[str, Any]] | None,
    *,
    top_k: int = 4,
) -> dict[str, Any]:
    top_rows = summarize_path_rows(path_rows, top_k=top_k)
    out: dict[str, Any] = {
        "path_summary_count": int(len(tuple(path_rows or ()))),
        "path_summary_gain_mass": 0.0,
        "path_summary_gap": 0.0,
        "path_summary_support": 0.0,
        "path_summary_mode_diversity": 0.0,
        "path_summaries": [],
    }
    if not top_rows:
        return out
    gains = [max(0.0, float(row.weighted_rel_gain)) for row in top_rows]
    top = gains[0] if gains else 0.0
    second = gains[1] if len(gains) > 1 else 0.0
    total = sum(gains)
    gap = 0.0 if top <= 1.0e-12 else min(1.0, max(0.0, (top - second) / top))
    supports = [
        0.5 * _clamp01(row.valid_frac) + 0.5 * _clamp01(row.confidence)
        for row in top_rows
    ]
    mode_count = len({str(row.target_mode) for row in top_rows if str(row.target_mode)})
    out.update({
        "path_summary_count": int(len(tuple(path_rows or ()))),
        "path_summary_gain_mass": float(1.0 - math.exp(-max(0.0, total))),
        "path_summary_gap": float(gap),
        "path_summary_support": float(sum(supports) / max(1, len(supports))),
        "path_summary_mode_diversity": float(min(1.0, mode_count / float(max(1, len(top_rows))))),
        "path_summaries": [row.to_dict() for row in top_rows],
    })
    for idx, row in enumerate(top_rows):
        prefix = f"path_summary_{int(idx)}_"
        out[f"{prefix}weighted_rel_gain"] = float(row.weighted_rel_gain)
        out[f"{prefix}rel_gain"] = float(row.rel_gain)
        out[f"{prefix}valid_frac"] = float(row.valid_frac)
        out[f"{prefix}confidence"] = float(row.confidence)
        out[f"{prefix}static_score"] = float(row.static_score)
        out[f"{prefix}transport_rel"] = float(row.transport_rel)
        out[f"{prefix}branch_factor"] = float(row.branch_factor)
        out[f"{prefix}cut_factor"] = float(row.cut_factor)
        out[f"{prefix}target_mode"] = str(row.target_mode)
    return out


@dataclass(frozen=True)
class CandidateStateFeatures:
    selected_path: tuple[int, ...] = ()
    selected_target_mode: str | None = None
    selected_path_gain: float | None = None
    selected_path_gain_pre_cut: float | None = None
    selected_rel_gain: float | None = None
    selected_transport_rel: float | None = None
    selected_lin_rel: float | None = None
    selected_branch_factor: float | None = None
    selected_cut_factor: float | None = None
    selected_effective_n: float | None = None
    local_candidate_count: int = 0
    estimated_child_raw_mse: float | None = None
    estimated_child_eff_mse: float | None = None
    estimated_parent_raw_mse: float | None = None
    estimated_parent_eff_mse: float | None = None
    estimated_one_hole_rel_improve_raw: float | None = None
    estimated_one_hole_rel_improve_eff: float | None = None
    proxy_one_hole_potential_eff: float | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any] | None) -> CandidateStateFeatures:
        row = row if isinstance(row, Mapping) else {}
        return cls(
            selected_path=_selected_or_action_path(row),
            selected_target_mode=None
            if row.get("selected_target_mode", None) is None
            else _safe_str(row.get("selected_target_mode", "")),
            selected_path_gain=None
            if row.get("selected_path_gain", None) is None
            else _safe_float(row.get("selected_path_gain", 0.0), 0.0),
            selected_path_gain_pre_cut=None
            if row.get("selected_path_gain_pre_cut", None) is None
            else _safe_float(row.get("selected_path_gain_pre_cut", 0.0), 0.0),
            selected_rel_gain=None
            if row.get("selected_rel_gain", None) is None
            else _safe_float(row.get("selected_rel_gain", 0.0), 0.0),
            selected_transport_rel=None
            if row.get("selected_transport_rel", None) is None
            else _safe_float(row.get("selected_transport_rel", 0.0), 0.0),
            selected_lin_rel=None
            if row.get("selected_lin_rel", None) is None
            else _safe_float(row.get("selected_lin_rel", 0.0), 0.0),
            selected_branch_factor=None
            if row.get("selected_branch_factor", None) is None
            else _safe_float(row.get("selected_branch_factor", 0.0), 0.0),
            selected_cut_factor=None
            if row.get("selected_cut_factor", None) is None
            else _safe_float(row.get("selected_cut_factor", 0.0), 0.0),
            selected_effective_n=None
            if row.get("selected_effective_n", None) is None
            else _safe_float(row.get("selected_effective_n", 0.0), 0.0),
            local_candidate_count=_safe_int(row.get("local_candidate_count", 0), 0),
            estimated_child_raw_mse=None
            if row.get("estimated_child_raw_mse", None) is None
            else _safe_float(row.get("estimated_child_raw_mse", 0.0), 0.0),
            estimated_child_eff_mse=None
            if row.get("estimated_child_eff_mse", None) is None
            else _safe_float(row.get("estimated_child_eff_mse", 0.0), 0.0),
            estimated_parent_raw_mse=None
            if row.get("estimated_parent_raw_mse", None) is None
            else _safe_float(row.get("estimated_parent_raw_mse", 0.0), 0.0),
            estimated_parent_eff_mse=None
            if row.get("estimated_parent_eff_mse", None) is None
            else _safe_float(row.get("estimated_parent_eff_mse", 0.0), 0.0),
            estimated_one_hole_rel_improve_raw=None
            if row.get("estimated_one_hole_rel_improve_raw", None) is None
            else _safe_float(row.get("estimated_one_hole_rel_improve_raw", 0.0), 0.0),
            estimated_one_hole_rel_improve_eff=None
            if row.get("estimated_one_hole_rel_improve_eff", None) is None
            else _safe_float(row.get("estimated_one_hole_rel_improve_eff", 0.0), 0.0),
            proxy_one_hole_potential_eff=None
            if row.get("proxy_one_hole_potential_eff", None) is None
            else _safe_float(row.get("proxy_one_hole_potential_eff", 0.0), 0.0),
        )

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "selected_path": _path_list(self.selected_path),
            "selected_target_mode": self.selected_target_mode,
            "selected_path_gain": self.selected_path_gain,
            "selected_path_gain_pre_cut": self.selected_path_gain_pre_cut,
            "selected_rel_gain": self.selected_rel_gain,
            "selected_transport_rel": self.selected_transport_rel,
            "selected_lin_rel": self.selected_lin_rel,
            "selected_branch_factor": self.selected_branch_factor,
            "selected_cut_factor": self.selected_cut_factor,
            "selected_effective_n": self.selected_effective_n,
            "local_candidate_count": int(self.local_candidate_count),
            "estimated_child_raw_mse": self.estimated_child_raw_mse,
            "estimated_child_eff_mse": self.estimated_child_eff_mse,
            "estimated_parent_raw_mse": self.estimated_parent_raw_mse,
            "estimated_parent_eff_mse": self.estimated_parent_eff_mse,
            "estimated_one_hole_rel_improve_raw": self.estimated_one_hole_rel_improve_raw,
            "estimated_one_hole_rel_improve_eff": self.estimated_one_hole_rel_improve_eff,
            "proxy_one_hole_potential_eff": self.proxy_one_hole_potential_eff,
        }


@dataclass(frozen=True)
class InverseSteeringPotential:
    allowed: bool = False
    reason: str = ""
    best_path: tuple[int, ...] | None = None
    best_rel_gain: float = 0.0
    best_weighted_rel_gain: float = 0.0
    candidate_paths: tuple[tuple[int, ...], ...] = ()
    path_rows: tuple[PathStateFeatures, ...] = ()

    @property
    def best_path_row(self) -> PathStateFeatures | None:
        if self.best_path is None:
            return None
        best = tuple(int(v) for v in self.best_path)
        for row in self.path_rows:
            if tuple(row.path) == best:
                return row
        return None

    def path_row_map(self) -> dict[tuple[int, ...], PathStateFeatures]:
        return {tuple(row.path): row for row in self.path_rows}

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": bool(self.allowed),
            "reason": str(self.reason),
            "best_path": None if self.best_path is None else _path_list(self.best_path),
            "best_rel_gain": float(self.best_rel_gain),
            "best_weighted_rel_gain": float(self.best_weighted_rel_gain),
            "candidate_paths": [_path_list(p) for p in self.candidate_paths],
            "path_rows": [row.to_dict() for row in self.path_rows],
        }


__all__ = [
    "CandidateStateFeatures",
    "InverseSteeringPotential",
    "ModeStateFeatures",
    "PathStateFeatures",
    "path_concentration",
    "path_distribution_metrics",
    "path_summary_stats",
    "summarize_path_rows",
]
