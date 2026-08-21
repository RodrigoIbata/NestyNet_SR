# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Feature schemas and training-row construction for the repair critic."""

import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from nestynet_sr.sr_search.factorized_search.engine.signals import PathStateFeatures
from nestynet_sr.sr_search.factorized_search.policy.features import (
    coerce_repair_feature_record,
    coerce_repair_feature_row,
)
from nestynet_sr.sr_search.factorized_search.shared_candidate import (
    SHARED_CANDIDATE_MASK_FIELD_NAMES,
    shared_candidate_row_dict,
)

if TYPE_CHECKING:
    from ._repair_critic_prediction import (
        predict_build_tuple_slate,
        predict_repair_tuple_slate,
    )

# Bound by the legacy facade after the prediction module has loaded.  Keeping
# these as late-bound globals preserves the original lookup without a cycle.

REPAIR_CRITIC_FEATURE_NAMES: tuple[str, ...] = (
    "one_hole_potential",
    "proxy_one_hole_potential",
    "analytic_potential",
    "analytic_concentration",
    "analytic_contrast",
    "analytic_cost",
    "analytic_stagnation",
    "path_top_mass",
    "path_second_mass",
    "path_entropy_norm",
    "path_positive_log",
    "path_summary_gain_mass",
    "path_summary_gap",
    "path_summary_support",
    "path_summary_mode_diversity",
    "gate_best_weighted_rel_gain",
    "gate_best_rel_gain",
    "gate_best_valid_frac",
    "gate_best_confidence",
    "gate_best_transport_rel",
    "gate_best_static_score",
    "gate_best_branch_factor",
    "gate_best_cut_factor",
    "selected_path_gain",
    "selected_path_gain_pre_cut",
    "selected_rel_gain",
    "selected_transport_rel",
    "selected_lin_rel",
    "selected_branch_factor",
    "selected_cut_factor",
    "selected_effective_n_log",
    "selected_mode_identity",
    "selected_mode_affine",
    "selected_mode_full",
    "best_exact_monotone",
    "best_has_periodic",
    "best_has_muldiv",
    "best_has_explogsqrt",
    "identity_vs_full_contrast",
    "affine_vs_full_contrast",
    "local_candidate_log",
    "parent_best_eff_log",
    "parent_best_raw_log",
    "parent_stagnation_score",
    "parent_stagnation_ratio",
    "parent_visits_log",
    "parent_visits_since_improve_log",
    "gate_allowed",
    "refine_slot_count",
    "refine_gate_potential",
    "refine_variant_log",
)


REPAIR_CRITIC_HEAD_NAMES: tuple[str, ...] = (
    "accept_prob",
    "positive_reward_prob",
    "new_residual_basin_prob",
    "new_best_prob",
    "reward_per_s_score",
    "utility_score",
)


REPAIR_CRITIC_MACRO_ACTION_NAMES: tuple[str, ...] = (
    "replace",
    "wrap_un",
    "add_rand",
    "mul_rand",
    "residual",
    "inv_steer",
    "repair_option",
    "boost",
    "prune",
    "crossover",
)


REPAIR_CRITIC_ROUTE_NAMES: tuple[str, ...] = (
    "build",
    "repair",
    "simplify",
    "recombine",
)


REPAIR_CRITIC_ACTION_ROUTE_MAP: dict[str, str] = {
    "replace": "build",
    "wrap_un": "build",
    "add_rand": "build",
    "mul_rand": "build",
    "residual": "build",
    "boost": "build",
    "inv_steer": "repair",
    "repair_option": "repair",
    "prune": "simplify",
    "crossover": "recombine",
}


REPAIR_CRITIC_PATH_FEATURE_NAMES: tuple[str, ...] = (
    "weighted_rel_gain_log",
    "weighted_rel_gain_raw_log",
    "weighted_rel_gain_pre_cut_log",
    "rel_gain",
    "valid_frac",
    "confidence",
    "transport_rel",
    "static_score_log",
    "branch_factor",
    "cut_factor",
    "branch_support",
    "branch_positive_log",
    "family_scale",
    "min_valid_frac_eff",
    "min_confidence_eff",
    "target_mode_identity",
    "target_mode_affine",
    "target_mode_full",
    "target_mode_other",
    "profile_exact_monotone",
    "profile_has_periodic",
    "profile_has_muldiv",
    "profile_has_explogsqrt",
)


REPAIR_CRITIC_PREVIEW_FEATURE_NAMES: tuple[str, ...] = (
    "local_probe_mse_log",
    "local_fit_mse_log",
    "local_fit_probe_gap_log",
    "local_mapping_identity",
    "local_mapping_affine",
    "local_mapping_full",
    "local_mapping_other",
    "target_mapping_identity",
    "target_mapping_affine",
    "target_mapping_full",
    "target_mapping_other",
    "candidate_subtree_size_log",
    "candidate_subtree_depth",
    "candidate_child_size_log",
    "candidate_child_depth",
    "candidate_subtree_size_delta",
    "candidate_subtree_depth_delta",
    "candidate_child_size_delta",
    "candidate_child_depth_delta",
    "beam_rank_inv",
    "local_rank_inv",
    "local_candidate_log",
    "provenance_count_log",
    "distinct_path_count_log",
    "distinct_mode_count_log",
    "distinct_local_mapping_count_log",
    "best_local_probe_mse_log",
    "mean_local_probe_mse_log",
    "worst_local_probe_mse_log",
    "best_local_fit_mse_log",
    "mean_local_fit_mse_log",
    "worst_local_fit_mse_log",
    "best_second_probe_gap_log",
    "mean_fit_probe_gap_log",
    "local_mapping_identity_frac",
    "local_mapping_affine_frac",
    "local_mapping_full_frac",
    "local_mapping_other_frac",
    "target_mapping_identity_frac",
    "target_mapping_affine_frac",
    "target_mapping_full_frac",
    "target_mapping_other_frac",
    "root_leaf",
    "root_addsub",
    "root_muldiv",
    "root_unary",
    "root_pow",
    "root_other",
    "action_inv_steer",
    "action_repair_option",
)


BUILD_TUPLE_PREVIEW_FEATURE_NAMES: tuple[str, ...] = (
    "candidate_child_size_log",
    "candidate_child_depth",
    "candidate_child_size_delta",
    "candidate_child_depth_delta",
    "path_length_log",
    "provenance_count_log",
    "distinct_action_count_log",
    "distinct_path_count_log",
    "mean_child_size_log",
    "mean_child_depth",
    "mean_child_size_delta",
    "mean_child_depth_delta",
    "mean_path_length_log",
    "root_leaf",
    "root_addsub",
    "root_muldiv",
    "root_unary",
    "root_pow",
    "root_other",
    "action_replace",
    "action_wrap_un",
    "action_residual",
    "action_other",
    "action_replace_frac",
    "action_wrap_un_frac",
    "action_residual_frac",
    "action_other_frac",
    "path_source_critic",
    "path_source_inverse",
    "path_source_random",
    "path_source_other",
    "path_source_critic_frac",
    "path_source_inverse_frac",
    "path_source_random_frac",
    "path_source_other_frac",
)


UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES: tuple[str, ...] = (
    *REPAIR_CRITIC_PREVIEW_FEATURE_NAMES,
    "path_length_log",
    "distinct_action_count_log",
    "mean_child_size_log",
    "mean_child_depth",
    "mean_child_size_delta",
    "mean_child_depth_delta",
    "mean_path_length_log",
    "action_replace",
    "action_wrap_un",
    "action_residual",
    "action_other",
    "action_replace_frac",
    "action_wrap_un_frac",
    "action_residual_frac",
    "action_other_frac",
    "path_source_critic",
    "path_source_inverse",
    "path_source_random",
    "path_source_other",
    "path_source_critic_frac",
    "path_source_inverse_frac",
    "path_source_random_frac",
    "path_source_other_frac",
    "route_source_repair",
    "route_source_build",
    *SHARED_CANDIDATE_MASK_FIELD_NAMES,
)


REPAIR_ROUTE_COMPARE_EXTRA_FEATURE_NAMES: tuple[str, ...] = (
    "repair_total_count_log",
    "repair_path_diversity_log",
    "repair_mode_diversity_log",
    "repair_best_probe_mse_log",
    "repair_second_probe_mse_log",
    "repair_mean_probe_mse_log",
    "repair_best_fit_mse_log",
    "repair_best_fit_probe_gap_log",
    "repair_mean_fit_probe_gap_log",
    "repair_best_support_log",
    "repair_best_action_inv_steer",
    "repair_best_action_repair_option",
    "repair_local_mapping_identity_frac",
    "repair_local_mapping_affine_frac",
    "repair_local_mapping_full_frac",
    "repair_local_mapping_other_frac",
    "repair_target_mapping_identity_frac",
    "repair_target_mapping_affine_frac",
    "repair_target_mapping_full_frac",
    "repair_target_mapping_other_frac",
    "repair_best_child_size_log",
    "repair_mean_child_size_log",
    "repair_best_child_depth",
    "repair_mean_child_depth",
    "build_total_count_log",
    "build_action_diversity_log",
    "build_path_diversity_log",
    "build_best_action_replace",
    "build_best_action_wrap_un",
    "build_best_action_residual",
    "build_best_action_other",
    "build_best_child_size_log",
    "build_mean_child_size_log",
    "build_best_child_depth",
    "build_mean_child_depth",
    "build_best_child_size_delta",
    "build_mean_child_size_delta",
    "build_best_path_len_log",
    "build_mean_path_len_log",
    "repair_learned_best_score",
    "repair_learned_second_score",
    "repair_learned_mean_score",
    "repair_learned_margin",
    "repair_learned_state_value",
    "build_learned_best_score",
    "build_learned_second_score",
    "build_learned_mean_score",
    "build_learned_margin",
    "build_learned_state_value",
    "delta_learned_best_score",
)


REPAIR_CRITIC_PATH_RELATION_NAMES: tuple[str, ...] = (
    "same",
    "ancestor",
    "descendant",
    "disjoint",
    "unknown",
)


REPAIR_CRITIC_MODE_NAMES: tuple[str, ...] = (
    "identity",
    "affine",
    "full",
)


REPAIR_CRITIC_SHARED_MODEL_KIND = "shared_encoder_bundle_v1"


REPAIR_CRITIC_LEGACY_MODEL_KIND = "legacy_aux_only"


REPAIR_CRITIC_ROUTE_COMPARE_MODEL_KIND = "route_compare_bundle_v1"


REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND = "build_tuple_ranker_bundle_v1"


REPAIR_CRITIC_UNIFIED_CANDIDATE_MODEL_KIND = "unified_candidate_ranker_bundle_v1"


REPAIR_CRITIC_SHARED_CANDIDATE_MODEL_KIND = "shared_candidate_dual_bundle_v1"


REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS: dict[str, float] = {
    "accept_prob": 0.35,
    "positive_reward_prob": 1.00,
    "new_residual_basin_prob": 0.25,
    "new_best_prob": 0.15,
    "reward_per_s_score": 0.75,
}


_REPAIR_CRITIC_NEGATIVE_TERMINAL_STATUSES = {
    "proposal_none",
    "dim_invalid",
    "dim_mismatch",
    "score_none",
    "no_paths",
    "no_ranked_paths",
    "no_best_path",
    "no_repair_candidates",
    "no_ranked_repairs",
    "no_global_child",
    "repair_option_none",
    "controller_no_candidate",
}


_REPAIR_CRITIC_DEFAULT_HEAD_SET = frozenset(REPAIR_CRITIC_HEAD_NAMES)


_REPAIR_CRITIC_DEFAULT_MACRO_ACTION_SET = frozenset(REPAIR_CRITIC_MACRO_ACTION_NAMES)


_REPAIR_CRITIC_DEFAULT_ROUTE_SET = frozenset(REPAIR_CRITIC_ROUTE_NAMES)


_REPAIR_CRITIC_DEFAULT_RELATION_SET = frozenset(REPAIR_CRITIC_PATH_RELATION_NAMES)


_REPAIR_CRITIC_DEFAULT_MODE_SET = frozenset(REPAIR_CRITIC_MODE_NAMES)


_REPAIR_CRITIC_DEFAULT_REPAIR_ACTION_SET = frozenset({"inv_steer", "repair_option"})


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _clamp01(value: Any) -> float:
    return min(1.0, max(0.0, _to_float(value, 0.0)))


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _row_first(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row.get(name, None) is not None:
            return row.get(name)
    return default


def _safe_log1p(value: Any) -> float:
    return math.log1p(max(0.0, _to_float(value, 0.0)))


def _safe_neg_log10(value: Any, floor: float = 1.0e-30) -> float:
    return -math.log10(max(floor, max(0.0, _to_float(value, floor))))


def _normalize_action_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _route_name_for_action(value: Any) -> str:
    action_name = _normalize_action_name(value)
    return str(REPAIR_CRITIC_ACTION_ROUTE_MAP.get(action_name, ""))


def _normalize_mode_name(value: Any) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    if mode in ("id",):
        return "identity"
    if mode in ("fitbest", "fitted", "legacy"):
        return "full"
    return mode


def _normalize_mapping_kind(value: Any) -> str:
    kind = str(value or "").strip().lower().replace("-", "_")
    if kind in ("identity", "id"):
        return "identity"
    if kind in ("affine", "lin", "linear"):
        return "affine"
    if kind:
        return kind
    return ""


def _mapping_bucket(value: Any) -> str:
    kind = _normalize_mapping_kind(value)
    if kind == "identity":
        return "identity"
    if kind == "affine":
        return "affine"
    if kind:
        return "full"
    return "other"


def _normalize_relation_name(value: Any) -> str:
    relation = str(value or "").strip().lower().replace("-", "_")
    if relation in _REPAIR_CRITIC_DEFAULT_RELATION_SET:
        return relation
    return "unknown"


def extract_repair_critic_features(row: Any) -> dict[str, float]:
    out = coerce_repair_feature_record(row).to_repair_critic_features()
    for name in REPAIR_CRITIC_FEATURE_NAMES:
        out.setdefault(name, 0.0)
    return out


def extract_repair_path_features(path_row: PathStateFeatures | dict[str, Any]) -> dict[str, float]:
    row = path_row if isinstance(path_row, PathStateFeatures) else PathStateFeatures.from_row(path_row)
    mode = _normalize_mode_name(row.target_mode)
    is_known_mode = mode in {"identity", "affine", "full", ""}
    return {
        "weighted_rel_gain_log": float(_safe_log1p(row.weighted_rel_gain)),
        "weighted_rel_gain_raw_log": float(_safe_log1p(row.weighted_rel_gain_raw)),
        "weighted_rel_gain_pre_cut_log": float(_safe_log1p(row.weighted_rel_gain_pre_cut)),
        "rel_gain": float(max(0.0, _to_float(row.rel_gain, 0.0))),
        "valid_frac": float(_clamp01(row.valid_frac)),
        "confidence": float(_clamp01(row.confidence)),
        "transport_rel": float(max(0.0, _to_float(row.transport_rel, 0.0))),
        "static_score_log": float(_safe_log1p(row.static_score)),
        "branch_factor": float(max(0.0, _to_float(row.branch_factor, 0.0))),
        "cut_factor": float(max(0.0, _to_float(row.cut_factor, 0.0))),
        "branch_support": float(max(0.0, _to_float(row.branch_support, 0.0))),
        "branch_positive_log": float(_safe_log1p(row.branch_positive_count)),
        "family_scale": float(max(0.0, _to_float(row.family_scale, 0.0))),
        "min_valid_frac_eff": float(_clamp01(row.min_valid_frac_eff)),
        "min_confidence_eff": float(_clamp01(row.min_confidence_eff)),
        "target_mode_identity": 1.0 if mode == "identity" else 0.0,
        "target_mode_affine": 1.0 if mode == "affine" else 0.0,
        "target_mode_full": 1.0 if mode == "full" else 0.0,
        "target_mode_other": 1.0 if (mode and not is_known_mode) else 0.0,
        "profile_exact_monotone": 1.0 if bool(row.profile_exact_monotone) else 0.0,
        "profile_has_periodic": 1.0 if bool(row.profile_has_periodic) else 0.0,
        "profile_has_muldiv": 1.0 if bool(row.profile_has_muldiv) else 0.0,
        "profile_has_explogsqrt": 1.0 if bool(row.profile_has_explogsqrt) else 0.0,
    }


def repair_critic_feature_vector(
    row: Any,
    *,
    feature_names: Sequence[str] = REPAIR_CRITIC_FEATURE_NAMES,
) -> list[float]:
    features = extract_repair_critic_features(row)
    return [float(features.get(name, 0.0)) for name in feature_names]


def repair_path_feature_vector(
    path_row: PathStateFeatures | dict[str, Any],
    *,
    feature_names: Sequence[str] = REPAIR_CRITIC_PATH_FEATURE_NAMES,
) -> list[float]:
    features = extract_repair_path_features(path_row)
    return [float(features.get(name, 0.0)) for name in feature_names]


def _preview_root_category(root_name: Any) -> str:
    name = str(root_name or "").strip().lower()
    if not name:
        return "other"
    if name in {"var", "const"}:
        return "leaf"
    if name in {"add", "sub"}:
        return "addsub"
    if name in {"mul", "div"}:
        return "muldiv"
    if name in {"neg", "sin", "cos", "exp", "log", "sqrt", "sqr"}:
        return "unary"
    if name == "pow":
        return "pow"
    return "other"


def _path_source_bucket(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token.startswith("critic"):
        return "critic"
    if token.startswith("inverse"):
        return "inverse"
    if token == "random":
        return "random"
    return "other"


def repair_preview_feature_vector(
    row: Mapping[str, Any] | None,
    *,
    feature_names: Sequence[str] = REPAIR_CRITIC_PREVIEW_FEATURE_NAMES,
) -> list[float]:
    row = row if isinstance(row, Mapping) else {}
    root_kind = _preview_root_category(row.get("candidate_root_op", ""))
    action_name = _normalize_action_name(row.get("action", ""))
    local_probe = _to_float(_row_first(row, "local_probe_mse", default=float("inf")), float("inf"))
    local_fit = _to_float(_row_first(row, "local_fit_mse", default=float("inf")), float("inf"))
    local_fit_probe_gap = max(0.0, _to_float(_row_first(row, "local_fit_probe_gap", default=(local_probe - local_fit)), 0.0))
    local_mapping_bucket = _mapping_bucket(_row_first(row, "local_mapping_kind", default=""))
    target_mapping_bucket = _mapping_bucket(_row_first(row, "target_mapping_kind", default=""))
    provenance_count = max(1.0, _to_float(_row_first(row, "provenance_count", default=1.0), 1.0))
    distinct_path_count = max(1.0, _to_float(_row_first(row, "distinct_path_count", default=1.0), 1.0))
    distinct_mode_count = max(1.0, _to_float(_row_first(row, "distinct_mode_count", default=1.0), 1.0))
    distinct_local_mapping_count = max(1.0, _to_float(_row_first(row, "distinct_local_mapping_count", default=1.0), 1.0))
    best_local_probe = _to_float(_row_first(row, "best_local_probe_mse", default=local_probe), local_probe)
    mean_local_probe = _to_float(_row_first(row, "mean_local_probe_mse", default=local_probe), local_probe)
    worst_local_probe = _to_float(_row_first(row, "worst_local_probe_mse", default=local_probe), local_probe)
    best_local_fit = _to_float(_row_first(row, "best_local_fit_mse", default=local_fit), local_fit)
    mean_local_fit = _to_float(_row_first(row, "mean_local_fit_mse", default=local_fit), local_fit)
    worst_local_fit = _to_float(_row_first(row, "worst_local_fit_mse", default=local_fit), local_fit)
    best_second_probe_gap = max(0.0, _to_float(_row_first(row, "best_second_probe_gap", default=0.0), 0.0))
    mean_fit_probe_gap = max(0.0, _to_float(_row_first(row, "mean_fit_probe_gap", default=local_fit_probe_gap), local_fit_probe_gap))
    local_mapping_identity_frac = _clamp01(_row_first(row, "local_mapping_identity_frac", default=(1.0 if local_mapping_bucket == "identity" else 0.0)))
    local_mapping_affine_frac = _clamp01(_row_first(row, "local_mapping_affine_frac", default=(1.0 if local_mapping_bucket == "affine" else 0.0)))
    local_mapping_full_frac = _clamp01(_row_first(row, "local_mapping_full_frac", default=(1.0 if local_mapping_bucket == "full" else 0.0)))
    local_mapping_other_frac = _clamp01(_row_first(row, "local_mapping_other_frac", default=(1.0 if local_mapping_bucket == "other" else 0.0)))
    target_mapping_identity_frac = _clamp01(_row_first(row, "target_mapping_identity_frac", default=(1.0 if target_mapping_bucket == "identity" else 0.0)))
    target_mapping_affine_frac = _clamp01(_row_first(row, "target_mapping_affine_frac", default=(1.0 if target_mapping_bucket == "affine" else 0.0)))
    target_mapping_full_frac = _clamp01(_row_first(row, "target_mapping_full_frac", default=(1.0 if target_mapping_bucket == "full" else 0.0)))
    target_mapping_other_frac = _clamp01(_row_first(row, "target_mapping_other_frac", default=(1.0 if target_mapping_bucket == "other" else 0.0)))
    features = {
        "local_probe_mse_log": float(_safe_neg_log10(local_probe)),
        "local_fit_mse_log": float(_safe_neg_log10(local_fit)),
        "local_fit_probe_gap_log": float(_safe_log1p(local_fit_probe_gap)),
        "local_mapping_identity": 1.0 if local_mapping_bucket == "identity" else 0.0,
        "local_mapping_affine": 1.0 if local_mapping_bucket == "affine" else 0.0,
        "local_mapping_full": 1.0 if local_mapping_bucket == "full" else 0.0,
        "local_mapping_other": 1.0 if local_mapping_bucket == "other" else 0.0,
        "target_mapping_identity": 1.0 if target_mapping_bucket == "identity" else 0.0,
        "target_mapping_affine": 1.0 if target_mapping_bucket == "affine" else 0.0,
        "target_mapping_full": 1.0 if target_mapping_bucket == "full" else 0.0,
        "target_mapping_other": 1.0 if target_mapping_bucket == "other" else 0.0,
        "candidate_subtree_size_log": float(_safe_log1p(_row_first(row, "candidate_subtree_size", default=0.0))),
        "candidate_subtree_depth": float(_to_float(_row_first(row, "candidate_subtree_depth", default=0.0), 0.0)),
        "candidate_child_size_log": float(_safe_log1p(_row_first(row, "candidate_child_size", default=0.0))),
        "candidate_child_depth": float(_to_float(_row_first(row, "candidate_child_depth", default=0.0), 0.0)),
        "candidate_subtree_size_delta": float(_to_float(_row_first(row, "candidate_subtree_size_delta", default=0.0), 0.0)),
        "candidate_subtree_depth_delta": float(_to_float(_row_first(row, "candidate_subtree_depth_delta", default=0.0), 0.0)),
        "candidate_child_size_delta": float(_to_float(_row_first(row, "candidate_child_size_delta", default=0.0), 0.0)),
        "candidate_child_depth_delta": float(_to_float(_row_first(row, "candidate_child_depth_delta", default=0.0), 0.0)),
        "beam_rank_inv": float(1.0 / (1.0 + max(0.0, _to_float(_row_first(row, "beam_rank", default=0.0), 0.0)))),
        "local_rank_inv": float(1.0 / (1.0 + max(0.0, _to_float(_row_first(row, "local_rank", default=0.0), 0.0)))),
        "local_candidate_log": float(_safe_log1p(_row_first(row, "local_candidate_count", default=0.0))),
        "provenance_count_log": float(_safe_log1p(provenance_count)),
        "distinct_path_count_log": float(_safe_log1p(distinct_path_count)),
        "distinct_mode_count_log": float(_safe_log1p(distinct_mode_count)),
        "distinct_local_mapping_count_log": float(_safe_log1p(distinct_local_mapping_count)),
        "best_local_probe_mse_log": float(_safe_neg_log10(best_local_probe)),
        "mean_local_probe_mse_log": float(_safe_neg_log10(mean_local_probe)),
        "worst_local_probe_mse_log": float(_safe_neg_log10(worst_local_probe)),
        "best_local_fit_mse_log": float(_safe_neg_log10(best_local_fit)),
        "mean_local_fit_mse_log": float(_safe_neg_log10(mean_local_fit)),
        "worst_local_fit_mse_log": float(_safe_neg_log10(worst_local_fit)),
        "best_second_probe_gap_log": float(_safe_log1p(best_second_probe_gap)),
        "mean_fit_probe_gap_log": float(_safe_log1p(mean_fit_probe_gap)),
        "local_mapping_identity_frac": float(local_mapping_identity_frac),
        "local_mapping_affine_frac": float(local_mapping_affine_frac),
        "local_mapping_full_frac": float(local_mapping_full_frac),
        "local_mapping_other_frac": float(local_mapping_other_frac),
        "target_mapping_identity_frac": float(target_mapping_identity_frac),
        "target_mapping_affine_frac": float(target_mapping_affine_frac),
        "target_mapping_full_frac": float(target_mapping_full_frac),
        "target_mapping_other_frac": float(target_mapping_other_frac),
        "root_leaf": 1.0 if root_kind == "leaf" else 0.0,
        "root_addsub": 1.0 if root_kind == "addsub" else 0.0,
        "root_muldiv": 1.0 if root_kind == "muldiv" else 0.0,
        "root_unary": 1.0 if root_kind == "unary" else 0.0,
        "root_pow": 1.0 if root_kind == "pow" else 0.0,
        "root_other": 1.0 if root_kind == "other" else 0.0,
        "action_inv_steer": 1.0 if action_name == "inv_steer" else 0.0,
        "action_repair_option": 1.0 if action_name == "repair_option" else 0.0,
    }
    return [float(features.get(name, 0.0)) for name in feature_names]


def build_preview_feature_vector(
    row: Mapping[str, Any] | None,
    *,
    feature_names: Sequence[str] = BUILD_TUPLE_PREVIEW_FEATURE_NAMES,
) -> list[float]:
    row = row if isinstance(row, Mapping) else {}
    root_kind = _preview_root_category(row.get("candidate_root_op", ""))
    action_name = _normalize_action_name(row.get("action", ""))
    path_source = _path_source_bucket(row.get("path_source", ""))
    provenance_count = max(1.0, _to_float(_row_first(row, "provenance_count", default=1.0), 1.0))
    distinct_action_count = max(1.0, _to_float(_row_first(row, "distinct_action_count", default=1.0), 1.0))
    distinct_path_count = max(1.0, _to_float(_row_first(row, "distinct_path_count", default=1.0), 1.0))
    mean_child_size = _to_float(
        _row_first(row, "mean_child_size", "candidate_child_size", default=0.0),
        0.0,
    )
    mean_child_depth = _to_float(
        _row_first(row, "mean_child_depth", "candidate_child_depth", default=0.0),
        0.0,
    )
    mean_child_size_delta = _to_float(
        _row_first(row, "mean_child_size_delta", "candidate_child_size_delta", default=0.0),
        0.0,
    )
    mean_child_depth_delta = _to_float(
        _row_first(row, "mean_child_depth_delta", "candidate_child_depth_delta", default=0.0),
        0.0,
    )
    mean_path_length = _to_float(
        _row_first(row, "mean_path_length", "path_length", default=0.0),
        0.0,
    )
    action_replace_frac = _to_float(
        _row_first(
            row,
            "action_replace_frac",
            default=1.0 if action_name == "replace" else 0.0,
        ),
        0.0,
    )
    action_wrap_un_frac = _to_float(
        _row_first(
            row,
            "action_wrap_un_frac",
            default=1.0 if action_name == "wrap_un" else 0.0,
        ),
        0.0,
    )
    action_residual_frac = _to_float(
        _row_first(
            row,
            "action_residual_frac",
            default=1.0 if action_name == "residual" else 0.0,
        ),
        0.0,
    )
    action_other_frac = _to_float(
        _row_first(
            row,
            "action_other_frac",
            default=1.0 if action_name not in {"replace", "wrap_un", "residual"} else 0.0,
        ),
        0.0,
    )
    path_source_critic_frac = _to_float(
        _row_first(
            row,
            "path_source_critic_frac",
            default=1.0 if path_source == "critic" else 0.0,
        ),
        0.0,
    )
    path_source_inverse_frac = _to_float(
        _row_first(
            row,
            "path_source_inverse_frac",
            default=1.0 if path_source == "inverse" else 0.0,
        ),
        0.0,
    )
    path_source_random_frac = _to_float(
        _row_first(
            row,
            "path_source_random_frac",
            default=1.0 if path_source == "random" else 0.0,
        ),
        0.0,
    )
    path_source_other_frac = _to_float(
        _row_first(
            row,
            "path_source_other_frac",
            default=1.0 if path_source == "other" else 0.0,
        ),
        0.0,
    )
    features = {
        "candidate_child_size_log": float(_safe_log1p(_row_first(row, "candidate_child_size", default=0.0))),
        "candidate_child_depth": float(_to_float(_row_first(row, "candidate_child_depth", default=0.0), 0.0)),
        "candidate_child_size_delta": float(_to_float(_row_first(row, "candidate_child_size_delta", default=0.0), 0.0)),
        "candidate_child_depth_delta": float(_to_float(_row_first(row, "candidate_child_depth_delta", default=0.0), 0.0)),
        "path_length_log": float(_safe_log1p(_row_first(row, "path_length", default=0.0))),
        "provenance_count_log": float(_safe_log1p(provenance_count)),
        "distinct_action_count_log": float(_safe_log1p(distinct_action_count)),
        "distinct_path_count_log": float(_safe_log1p(distinct_path_count)),
        "mean_child_size_log": float(_safe_log1p(mean_child_size)),
        "mean_child_depth": float(mean_child_depth),
        "mean_child_size_delta": float(mean_child_size_delta),
        "mean_child_depth_delta": float(mean_child_depth_delta),
        "mean_path_length_log": float(_safe_log1p(mean_path_length)),
        "root_leaf": 1.0 if root_kind == "leaf" else 0.0,
        "root_addsub": 1.0 if root_kind == "addsub" else 0.0,
        "root_muldiv": 1.0 if root_kind == "muldiv" else 0.0,
        "root_unary": 1.0 if root_kind == "unary" else 0.0,
        "root_pow": 1.0 if root_kind == "pow" else 0.0,
        "root_other": 1.0 if root_kind == "other" else 0.0,
        "action_replace": 1.0 if action_name == "replace" else 0.0,
        "action_wrap_un": 1.0 if action_name == "wrap_un" else 0.0,
        "action_residual": 1.0 if action_name == "residual" else 0.0,
        "action_other": 1.0 if action_name not in {"replace", "wrap_un", "residual"} else 0.0,
        "action_replace_frac": float(action_replace_frac),
        "action_wrap_un_frac": float(action_wrap_un_frac),
        "action_residual_frac": float(action_residual_frac),
        "action_other_frac": float(action_other_frac),
        "path_source_critic": 1.0 if path_source == "critic" else 0.0,
        "path_source_inverse": 1.0 if path_source == "inverse" else 0.0,
        "path_source_random": 1.0 if path_source == "random" else 0.0,
        "path_source_other": 1.0 if path_source == "other" else 0.0,
        "path_source_critic_frac": float(path_source_critic_frac),
        "path_source_inverse_frac": float(path_source_inverse_frac),
        "path_source_random_frac": float(path_source_random_frac),
        "path_source_other_frac": float(path_source_other_frac),
    }
    return [float(features.get(name, 0.0)) for name in feature_names]


def unified_candidate_preview_feature_vector(
    row: Mapping[str, Any] | None,
    *,
    feature_names: Sequence[str] = UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES,
) -> list[float]:
    row = shared_candidate_row_dict(row)
    repair_features = {
        name: value
        for name, value in zip(
            REPAIR_CRITIC_PREVIEW_FEATURE_NAMES,
            repair_preview_feature_vector(row, feature_names=REPAIR_CRITIC_PREVIEW_FEATURE_NAMES),
        )
    }
    build_features = {
        name: value
        for name, value in zip(
            BUILD_TUPLE_PREVIEW_FEATURE_NAMES,
            build_preview_feature_vector(row, feature_names=BUILD_TUPLE_PREVIEW_FEATURE_NAMES),
        )
    }
    route_source = str(row.get("route_source", "") or "").strip().lower()
    if route_source not in {"repair", "build"}:
        action_name = _normalize_action_name(row.get("action", ""))
        route_source = "repair" if action_name in {"inv_steer", "repair_option"} else "build"
    features = dict(repair_features)
    features.update(build_features)
    features.update({
        "route_source_repair": 1.0 if route_source == "repair" else 0.0,
        "route_source_build": 1.0 if route_source == "build" else 0.0,
    })
    for name in SHARED_CANDIDATE_MASK_FIELD_NAMES:
        features[name] = float(_clamp01(row.get(name, 0.0)))
    return [float(features.get(name, 0.0)) for name in feature_names]


def _row_has_supervised_outcome(row: dict[str, Any]) -> bool:
    row = coerce_repair_feature_row(row)
    status = str(row.get("status", "") or "")
    if status.startswith("controller_blocked_") or status == "controller_prefers_build":
        return False
    if row.get("accepted", None) is not None:
        return True
    if row.get("reward", None) is not None:
        return True
    if row.get("reward_per_s", None) is not None:
        return True
    if "created_new_residual_basin" in row or "became_global_best" in row:
        return True
    return status in _REPAIR_CRITIC_NEGATIVE_TERMINAL_STATUSES


def estimate_reward_per_s_scale(rows: Sequence[Any]) -> float:
    vals: list[float] = []
    for row in rows:
        row = coerce_repair_feature_row(row)
        rps = _to_float(row.get("reward_per_s", 0.0), 0.0)
        if rps > 0.0 and math.isfinite(rps):
            vals.append(float(rps))
    if not vals:
        return 1.0
    vals.sort()
    idx = int(0.75 * (len(vals) - 1))
    idx = max(0, min(len(vals) - 1, idx))
    return max(1.0e-6, float(vals[idx]))


def make_repair_critic_target(
    row: Any,
    *,
    reward_per_s_scale: float,
    utility_weights: dict[str, float] | None = None,
) -> dict[str, float] | None:
    row = coerce_repair_feature_row(row)
    if not row or not _row_has_supervised_outcome(row):
        return None
    accepted = 1.0 if bool(row.get("accepted", False)) else 0.0
    reward = _to_float(row.get("reward", 0.0), 0.0)
    positive_reward = 1.0 if reward > 0.0 else 0.0
    new_residual_basin = 1.0 if bool(row.get("created_new_residual_basin", False)) else 0.0
    new_best = 1.0 if bool(row.get("became_global_best", False)) else 0.0
    reward_per_s = max(0.0, _to_float(row.get("reward_per_s", 0.0), 0.0))
    scale = max(1.0e-6, float(reward_per_s_scale))
    reward_score = math.tanh(reward_per_s / scale)
    weights = dict(REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS)
    if isinstance(utility_weights, dict):
        for key, value in utility_weights.items():
            if key in weights:
                weights[key] = max(0.0, _to_float(value, weights[key]))
    denom = sum(weights.values())
    if denom <= 0.0:
        denom = 1.0
    utility = (
        weights["accept_prob"] * accepted
        + weights["positive_reward_prob"] * positive_reward
        + weights["new_residual_basin_prob"] * new_residual_basin
        + weights["new_best_prob"] * new_best
        + weights["reward_per_s_score"] * reward_score
    ) / denom
    return {
        "accept_prob": float(accepted),
        "positive_reward_prob": float(positive_reward),
        "new_residual_basin_prob": float(new_residual_basin),
        "new_best_prob": float(new_best),
        "reward_per_s_score": float(reward_score),
        "utility_score": float(utility),
    }


def collect_repair_critic_examples(
    rows: Sequence[Any],
    *,
    feature_names: Sequence[str] = REPAIR_CRITIC_FEATURE_NAMES,
    utility_weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, float]], float]:
    rows_list = [coerce_repair_feature_row(row) for row in rows]
    rows_list = [row for row in rows_list if row]
    reward_scale = estimate_reward_per_s_scale(rows_list)
    examples: list[dict[str, float]] = []
    for row in rows_list:
        target = make_repair_critic_target(
            row,
            reward_per_s_scale=reward_scale,
            utility_weights=utility_weights,
        )
        if target is None:
            continue
        features = extract_repair_critic_features(row)
        examples.append({
            **{name: float(features.get(name, 0.0)) for name in feature_names},
            **target,
        })
    return examples, float(reward_scale)


def _extract_macro_action_label(row: Any) -> str | None:
    row = coerce_repair_feature_row(row)
    for key in ("controller_policy_action", "macro_action"):
        action_name = _normalize_action_name(row.get(key, ""))
        if action_name in _REPAIR_CRITIC_DEFAULT_MACRO_ACTION_SET:
            return action_name
    return None


def _extract_route_label(row: Any) -> str | None:
    route_name = _route_name_for_action(_extract_macro_action_label(row))
    if route_name in _REPAIR_CRITIC_DEFAULT_ROUTE_SET:
        return route_name
    return None


def _match_selected_path_index(record: Any) -> int | None:
    selected_path = tuple(int(v) for v in (record.candidate.selected_path or ()))
    if not selected_path:
        return None
    selected_mode = _normalize_mode_name(record.selected_target_mode)
    fallback_idx = None
    for idx, row in enumerate(record.path_rows):
        if tuple(int(v) for v in row.path) != selected_path:
            continue
        row_mode = _normalize_mode_name(row.target_mode)
        if selected_mode and row_mode == selected_mode:
            return idx
        if fallback_idx is None:
            fallback_idx = idx
    return fallback_idx


def _build_training_rows(
    rows: Sequence[Any],
    *,
    utility_weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    rows_list = [coerce_repair_feature_row(row) for row in rows]
    rows_list = [row for row in rows_list if row]
    reward_scale = estimate_reward_per_s_scale(rows_list)
    out: list[dict[str, Any]] = []
    for row in rows_list:
        record = coerce_repair_feature_record(row)
        aux_target = make_repair_critic_target(
            row,
            reward_per_s_scale=reward_scale,
            utility_weights=utility_weights,
        )
        macro_action = _extract_macro_action_label(row)
        route_name = _extract_route_label(row)
        path_rows = tuple(record.path_rows)
        path_target_index = _match_selected_path_index(record)
        if aux_target is None and macro_action is None and route_name is None and path_target_index is None:
            continue
        out.append({
            "features": extract_repair_critic_features(record),
            "aux_target": aux_target,
            "macro_action": macro_action,
            "route_name": route_name,
            "path_rows": path_rows,
            "path_target_index": path_target_index,
        })
    return out, float(reward_scale)


def _build_oracle_pretrain_rows(tasks: Sequence[Any]) -> list[dict[str, Any]]:
    mode_index = {name: idx for idx, name in enumerate(REPAIR_CRITIC_MODE_NAMES)}
    relation_index = {name: idx for idx, name in enumerate(REPAIR_CRITIC_PATH_RELATION_NAMES)}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        controller_row = task.get("controller_row", None)
        if not isinstance(controller_row, dict):
            continue
        record = coerce_repair_feature_record(controller_row)
        path_rows = tuple(record.path_rows)
        if not path_rows:
            continue
        path_label_map = {
            tuple(int(v) for v in (row.get("path", []) or ())): row
            for row in list(task.get("path_labels", []) or [])
            if isinstance(row, dict)
        }
        path_target_index = task.get("target_path_index", None)
        if not isinstance(path_target_index, int) or not (0 <= path_target_index < len(path_rows)):
            path_target_index = None
            target_path = tuple(int(v) for v in (task.get("target_path", []) or ()))
            if target_path:
                for idx, path_row in enumerate(path_rows):
                    if tuple(int(v) for v in path_row.path) == target_path:
                        path_target_index = idx
                        break
        relation_targets = [-100] * len(path_rows)
        mode_targets = [-100] * len(path_rows)
        improve_targets = [0.0] * len(path_rows)
        improve_mask = [False] * len(path_rows)
        for idx, path_row in enumerate(path_rows):
            label_row = path_label_map.get(tuple(int(v) for v in path_row.path), None)
            if not isinstance(label_row, dict):
                continue
            relation_name = _normalize_relation_name(label_row.get("relation", None))
            relation_targets[idx] = int(relation_index.get(relation_name, relation_index["unknown"]))
            mode_name = _normalize_mode_name(label_row.get("best_mode", None))
            if mode_name in mode_index:
                mode_targets[idx] = int(mode_index[mode_name])
            improve_targets[idx] = float(_clamp01(label_row.get("improvement_estimate", 0.0)))
            improve_mask[idx] = True
        if path_target_index is None and not any(v >= 0 for v in relation_targets):
            continue
        rows.append({
            "features": extract_repair_critic_features(record),
            "path_rows": path_rows,
            "path_target_index": path_target_index,
            "relation_targets": relation_targets,
            "mode_targets": mode_targets,
            "improve_targets": improve_targets,
            "improve_mask": improve_mask,
        })
    return rows


def _resolve_actor_critic_reward(
    row: Mapping[str, Any],
    *,
    reward_target: str = "descendant_preferred",
) -> tuple[float | None, str]:
    if not isinstance(row, Mapping):
        return None, ""
    target = str(reward_target or "descendant_preferred").strip().lower()
    if target not in {"descendant_preferred", "descendant_only", "immediate"}:
        target = "descendant_preferred"

    def _read(name: str) -> float | None:
        try:
            value = float(row.get(name, None))
        except Exception:
            return None
        if not math.isfinite(value):
            return None
        return float(value)

    immediate = _read("actor_critic_reward")
    descendant = _read("actor_critic_descendant_reward")
    if target == "immediate":
        return immediate, "immediate" if immediate is not None else ""
    if target == "descendant_only":
        return descendant, "descendant" if descendant is not None else ""
    if descendant is not None:
        return descendant, "descendant"
    if immediate is not None:
        return immediate, "immediate"
    return None, ""


def _common_candidate_q_target(
    utility: float,
    continuation_value: float | None,
    regret_value: float,
    *,
    continuation_weight: float = 0.25,
    regret_weight: float = 1.0,
) -> float:
    cont = 0.0 if continuation_value is None else float(continuation_value)
    return float(utility) + float(max(0.0, continuation_weight)) * cont - float(max(0.0, regret_weight)) * float(regret_value)


def _match_path_mode_index(
    path_rows: Sequence[PathStateFeatures],
    *,
    target_path: Sequence[int] | None,
    target_mode: Any = None,
) -> int | None:
    path = tuple(int(v) for v in (target_path or ()))
    if not path:
        return None
    mode = _normalize_mode_name(target_mode)
    fallback_idx = None
    for idx, path_row in enumerate(path_rows):
        if tuple(int(v) for v in path_row.path) != path:
            continue
        row_mode = _normalize_mode_name(path_row.target_mode)
        if mode and row_mode == mode:
            return idx
        if fallback_idx is None:
            fallback_idx = idx
    return fallback_idx


def _match_candidate_path_index(
    path_rows: Sequence[PathStateFeatures],
    preview_row: Mapping[str, Any],
) -> int | None:
    route_source = str(preview_row.get("route_source", "") or "").strip().lower()
    target_mode = preview_row.get("target_mode", None) if route_source == "repair" else None
    path_idx = _match_path_mode_index(
        path_rows,
        target_path=preview_row.get("path", ()),
        target_mode=target_mode,
    )
    if path_idx is not None:
        return int(path_idx)
    if not path_rows:
        return None
    return 0


def _make_repair_slate_utility(
    *,
    child_eff_mse: Any,
    parent_eff_mse: Any,
    safe_eps: float = 1.0e-12,
) -> float | None:
    child_eff = _finite_float_or_none(child_eff_mse)
    if child_eff is None:
        return None
    child_eff = max(float(safe_eps), float(child_eff))
    parent_eff = _finite_float_or_none(parent_eff_mse)
    if parent_eff is None:
        return float(-math.log(child_eff + float(safe_eps)))
    parent_eff = max(float(safe_eps), float(parent_eff))
    return float(math.log(parent_eff + float(safe_eps)) - math.log(child_eff + float(safe_eps)))


def _group_repair_preview_rows(
    preview_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for idx, preview_row in enumerate(preview_rows):
        if not isinstance(preview_row, Mapping):
            continue
        row_dict = dict(preview_row)
        child_key = str(row_dict.get("child_key", "") or row_dict.get("child_expr", "") or f"__row_{idx}")
        groups.setdefault(child_key, []).append(row_dict)
    out: list[dict[str, Any]] = []
    for child_key, group_rows in groups.items():
        if not group_rows:
            continue
        group_rows.sort(
            key=lambda row: (
                float(_to_float(row.get("local_probe_mse", float("inf")), float("inf"))),
                float(_to_float(row.get("local_fit_mse", float("inf")), float("inf"))),
                float(_to_float(row.get("beam_rank", 0.0), 0.0)),
                float(_to_float(row.get("local_rank", 0.0), 0.0)),
                str(row.get("tuple_provenance", "")),
            )
        )
        rep = dict(group_rows[0])
        probe_vals = [
            float(v)
            for v in (_finite_float_or_none(row.get("local_probe_mse", None)) for row in group_rows)
            if v is not None
        ]
        fit_vals = [
            float(v)
            for v in (_finite_float_or_none(row.get("local_fit_mse", None)) for row in group_rows)
            if v is not None
        ]
        gap_vals = [
            max(0.0, float(probe) - float(fit))
            for probe, fit in zip(probe_vals, fit_vals)
        ]
        sorted_probe = sorted(probe_vals)
        best_probe = float(sorted_probe[0]) if sorted_probe else float("inf")
        second_probe = float(sorted_probe[1]) if len(sorted_probe) > 1 else best_probe
        best_fit = float(min(fit_vals)) if fit_vals else float("inf")
        local_mapping_buckets = [_mapping_bucket(row.get("local_mapping_kind", "")) for row in group_rows]
        target_mapping_buckets = [_mapping_bucket(row.get("target_mapping_kind", "")) for row in group_rows]
        path_keys = {
            tuple(int(v) for v in (row.get("path", []) or ()))
            for row in group_rows
        }
        mode_keys = {
            _normalize_mode_name(row.get("target_mode", ""))
            for row in group_rows
            if _normalize_mode_name(row.get("target_mode", ""))
        }
        local_mapping_keys = {
            _normalize_mapping_kind(row.get("local_mapping_kind", ""))
            for row in group_rows
            if _normalize_mapping_kind(row.get("local_mapping_kind", ""))
        }

        def _bucket_frac(values: Sequence[str], bucket: str) -> float:
            if not values:
                return 0.0
            return float(sum(1 for value in values if value == bucket)) / float(len(values))

        rep.update({
            "child_key": str(child_key),
            "provenance_grouped": True,
            "provenance_count": int(len(group_rows)),
            "distinct_path_count": int(len(path_keys) or 1),
            "distinct_mode_count": int(len(mode_keys) or 1),
            "distinct_local_mapping_count": int(len(local_mapping_keys) or 1),
            "best_local_probe_mse": float(best_probe),
            "mean_local_probe_mse": float(sum(probe_vals) / len(probe_vals)) if probe_vals else float("inf"),
            "worst_local_probe_mse": float(max(probe_vals)) if probe_vals else float("inf"),
            "best_local_fit_mse": float(best_fit),
            "mean_local_fit_mse": float(sum(fit_vals) / len(fit_vals)) if fit_vals else float("inf"),
            "worst_local_fit_mse": float(max(fit_vals)) if fit_vals else float("inf"),
            "best_second_probe_gap": float(max(0.0, second_probe - best_probe)) if probe_vals else 0.0,
            "mean_fit_probe_gap": float(sum(gap_vals) / len(gap_vals)) if gap_vals else 0.0,
            "local_mapping_identity_frac": float(_bucket_frac(local_mapping_buckets, "identity")),
            "local_mapping_affine_frac": float(_bucket_frac(local_mapping_buckets, "affine")),
            "local_mapping_full_frac": float(_bucket_frac(local_mapping_buckets, "full")),
            "local_mapping_other_frac": float(_bucket_frac(local_mapping_buckets, "other")),
            "target_mapping_identity_frac": float(_bucket_frac(target_mapping_buckets, "identity")),
            "target_mapping_affine_frac": float(_bucket_frac(target_mapping_buckets, "affine")),
            "target_mapping_full_frac": float(_bucket_frac(target_mapping_buckets, "full")),
            "target_mapping_other_frac": float(_bucket_frac(target_mapping_buckets, "other")),
            "provenance_rows": [shared_candidate_row_dict(row, route_source="repair") for row in group_rows],
        })
        out.append(shared_candidate_row_dict(rep, route_source="repair"))
    out.sort(
        key=lambda row: (
            float(_to_float(row.get("best_local_probe_mse", row.get("local_probe_mse", float("inf"))), float("inf"))),
            float(_to_float(row.get("best_local_fit_mse", row.get("local_fit_mse", float("inf"))), float("inf"))),
            -float(_to_float(row.get("provenance_count", 1.0), 1.0)),
            str(row.get("child_key", "")),
        )
    )
    return out


def _group_build_preview_rows(
    preview_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for idx, preview_row in enumerate(preview_rows):
        if not isinstance(preview_row, Mapping):
            continue
        row_dict = dict(preview_row)
        child_key = str(row_dict.get("child_key", "") or row_dict.get("child_expr", "") or f"__build_{idx}")
        groups.setdefault(child_key, []).append(row_dict)
    out: list[dict[str, Any]] = []
    for child_key, group_rows in groups.items():
        if not group_rows:
            continue
        group_rows_sorted = sorted(
            group_rows,
            key=lambda row: (
                0 if bool(row.get("exact_child_score_observed", False)) else 1,
                float(_to_float(row.get("child_eff_mse", float("inf")), float("inf"))),
                float(_to_float(row.get("child_raw_mse", float("inf")), float("inf"))),
                str(row.get("action", "")),
            ),
        )
        exact_rows = [row for row in group_rows_sorted if bool(row.get("exact_child_score_observed", False))]
        rep = dict((exact_rows[0] if exact_rows else group_rows_sorted[0]))
        finite_child_sizes = [_finite_float_or_none(row.get("candidate_child_size", None)) for row in group_rows_sorted]
        finite_child_depths = [_finite_float_or_none(row.get("candidate_child_depth", None)) for row in group_rows_sorted]
        finite_child_size_deltas = [_finite_float_or_none(row.get("candidate_child_size_delta", None)) for row in group_rows_sorted]
        finite_child_depth_deltas = [_finite_float_or_none(row.get("candidate_child_depth_delta", None)) for row in group_rows_sorted]
        finite_path_lengths = [
            _finite_float_or_none(_row_first(row, "path_length", default=len(tuple(int(v) for v in (row.get("path", []) or ())))))
            for row in group_rows_sorted
        ]
        child_sizes = [float(v) for v in finite_child_sizes if v is not None]
        child_depths = [float(v) for v in finite_child_depths if v is not None]
        child_size_deltas = [float(v) for v in finite_child_size_deltas if v is not None]
        child_depth_deltas = [float(v) for v in finite_child_depth_deltas if v is not None]
        path_lengths = [float(v) for v in finite_path_lengths if v is not None]
        action_names = {
            _normalize_action_name(row.get("action", ""))
            for row in group_rows_sorted
            if _normalize_action_name(row.get("action", ""))
        }
        path_keys = {
            tuple(int(v) for v in (row.get("path", []) or ()))
            for row in group_rows_sorted
        }
        nonempty_path_keys = {path for path in path_keys if path}
        action_buckets = [
            (
                "replace" if _normalize_action_name(row.get("action", "")) == "replace"
                else "wrap_un" if _normalize_action_name(row.get("action", "")) == "wrap_un"
                else "residual" if _normalize_action_name(row.get("action", "")) == "residual"
                else "other"
            )
            for row in group_rows_sorted
        ]
        path_source_buckets = [_path_source_bucket(row.get("path_source", "")) for row in group_rows_sorted]

        def _bucket_frac(values: Sequence[str], bucket: str) -> float:
            if not values:
                return 0.0
            return float(sum(1 for value in values if value == bucket)) / float(len(values))

        rep.update({
            "child_key": str(child_key),
            "provenance_grouped": True,
            "provenance_count": int(len(group_rows_sorted)),
            "distinct_action_count": int(len(action_names) or 1),
            "distinct_path_count": int(len(nonempty_path_keys) or len(path_keys) or 1),
            "mean_child_size": float(sum(child_sizes) / len(child_sizes)) if child_sizes else float(_to_float(rep.get("candidate_child_size", 0.0), 0.0)),
            "mean_child_depth": float(sum(child_depths) / len(child_depths)) if child_depths else float(_to_float(rep.get("candidate_child_depth", 0.0), 0.0)),
            "mean_child_size_delta": float(sum(child_size_deltas) / len(child_size_deltas)) if child_size_deltas else float(_to_float(rep.get("candidate_child_size_delta", 0.0), 0.0)),
            "mean_child_depth_delta": float(sum(child_depth_deltas) / len(child_depth_deltas)) if child_depth_deltas else float(_to_float(rep.get("candidate_child_depth_delta", 0.0), 0.0)),
            "mean_path_length": float(sum(path_lengths) / len(path_lengths)) if path_lengths else float(_to_float(_row_first(rep, "path_length", default=0.0), 0.0)),
            "action_replace_frac": float(_bucket_frac(action_buckets, "replace")),
            "action_wrap_un_frac": float(_bucket_frac(action_buckets, "wrap_un")),
            "action_residual_frac": float(_bucket_frac(action_buckets, "residual")),
            "action_other_frac": float(_bucket_frac(action_buckets, "other")),
            "path_source_critic_frac": float(_bucket_frac(path_source_buckets, "critic")),
            "path_source_inverse_frac": float(_bucket_frac(path_source_buckets, "inverse")),
            "path_source_random_frac": float(_bucket_frac(path_source_buckets, "random")),
            "path_source_other_frac": float(_bucket_frac(path_source_buckets, "other")),
            "provenance_rows": [shared_candidate_row_dict(row, route_source="build") for row in group_rows_sorted],
        })
        out.append(shared_candidate_row_dict(rep, route_source="build"))
    out.sort(
        key=lambda row: (
            float(_to_float(row.get("child_eff_mse", float("inf")), float("inf"))),
            -float(_to_float(row.get("provenance_count", 1.0), 1.0)),
            str(row.get("child_key", "")),
        )
    )
    return out


def _summarize_route_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_eff: Any,
    safe_eps: float = 1.0e-12,
    build_mode: bool = False,
) -> dict[str, Any] | None:
    preview_rows: list[dict[str, Any]] = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not preview_rows:
        return None
    observed: list[dict[str, Any]] = []
    for row in preview_rows:
        if not bool(row.get("exact_child_score_observed", False)):
            continue
        utility = _make_repair_slate_utility(
            child_eff_mse=row.get("child_eff_mse", None),
            parent_eff_mse=parent_eff,
            safe_eps=float(safe_eps),
        )
        if utility is None:
            continue
        observed.append({
            "row": dict(row),
            "utility": float(utility),
            "child_eff_mse": float(_finite_float_or_none(row.get("child_eff_mse", None)) or 0.0),
        })
    if not observed and not bool(build_mode):
        return None
    preview_sorted = list(preview_rows)
    if build_mode:
        preview_sorted.sort(
            key=lambda row: (
                float(_to_float(row.get("candidate_child_size", float("inf")), float("inf"))),
                float(_to_float(row.get("candidate_child_depth", float("inf")), float("inf"))),
                str(row.get("action", "")),
            )
        )
    else:
        preview_sorted.sort(
            key=lambda row: (
                float(_to_float(row.get("best_local_probe_mse", row.get("local_probe_mse", float("inf"))), float("inf"))),
                float(_to_float(row.get("best_local_fit_mse", row.get("local_fit_mse", float("inf"))), float("inf"))),
                -float(_to_float(row.get("provenance_count", 1.0), 1.0)),
                str(row.get("action", "")),
            )
        )
    preview_best_row = dict(preview_sorted[0])
    if observed:
        observed.sort(
            key=lambda item: (
                float(item["utility"]),
                -float(item["child_eff_mse"]),
                str(item["row"].get("child_key", "")),
            ),
            reverse=True,
        )
        utilities = [float(item["utility"]) for item in observed]
        best_utility = float(utilities[0])
        second_utility = float(utilities[1]) if len(utilities) > 1 else float(utilities[0])
    else:
        utilities = [0.0]
        best_utility = 0.0
        second_utility = 0.0
    action_names = {
        _normalize_action_name(item.get("action", ""))
        for item in preview_rows
        if _normalize_action_name(item.get("action", ""))
    }
    path_keys = {
        tuple(int(v) for v in (item.get("path", []) or ()))
        for item in preview_rows
    }
    out = {
        "rows": [dict(item["row"]) for item in observed],
        "exact_count": int(len(observed)),
        "total_count": int(len(preview_rows)),
        "best_utility": float(best_utility),
        "second_utility": float(second_utility),
        "mean_utility": float(sum(utilities) / len(utilities)),
        "worst_utility": float(min(utilities)),
        "best_gap": float(max(0.0, best_utility - second_utility)),
        "best_action": str(_normalize_action_name(preview_best_row.get("action", ""))),
        "action_diversity": int(len(action_names) or 1),
        "path_diversity": int(len(path_keys) or 1),
    }
    if build_mode:
        child_sizes = [
            float(_to_float(row.get("candidate_child_size", 0.0), 0.0))
            for row in preview_rows
        ]
        child_depths = [
            float(_to_float(row.get("candidate_child_depth", 0.0), 0.0))
            for row in preview_rows
        ]
        child_size_deltas = [
            float(_to_float(row.get("candidate_child_size_delta", 0.0), 0.0))
            for row in preview_rows
        ]
        path_lens = [
            float(len(tuple(int(v) for v in (row.get("path", []) or ()))))
            for row in preview_rows
        ]
        out.update({
            "best_child_size": float(_to_float(preview_best_row.get("candidate_child_size", 0.0), 0.0)),
            "mean_child_size": float(sum(child_sizes) / len(child_sizes)) if child_sizes else 0.0,
            "best_child_depth": float(_to_float(preview_best_row.get("candidate_child_depth", 0.0), 0.0)),
            "mean_child_depth": float(sum(child_depths) / len(child_depths)) if child_depths else 0.0,
            "best_child_size_delta": float(_to_float(preview_best_row.get("candidate_child_size_delta", 0.0), 0.0)),
            "mean_child_size_delta": float(sum(child_size_deltas) / len(child_size_deltas)) if child_size_deltas else 0.0,
            "best_path_len": float(len(tuple(int(v) for v in (preview_best_row.get("path", []) or ())))),
            "mean_path_len": float(sum(path_lens) / len(path_lens)) if path_lens else 0.0,
        })
        return out
    mode_keys = {
        _normalize_mode_name(item.get("target_mode", ""))
        for item in preview_rows
        if _normalize_mode_name(item.get("target_mode", ""))
    }
    local_mapping_rows = [_mapping_bucket(item.get("local_mapping_kind", "")) for item in preview_rows]
    target_mapping_rows = [_mapping_bucket(item.get("target_mapping_kind", "")) for item in preview_rows]
    child_sizes = [float(_to_float(row.get("candidate_child_size", 0.0), 0.0)) for row in preview_rows]
    child_depths = [float(_to_float(row.get("candidate_child_depth", 0.0), 0.0)) for row in preview_rows]

    def _bucket_frac(values: Sequence[str], bucket: str) -> float:
        if not values:
            return 0.0
        return float(sum(1 for value in values if value == bucket)) / float(len(values))

    sorted_probe = sorted(
        float(_to_float(row.get("best_local_probe_mse", row.get("local_probe_mse", float("inf"))), float("inf")))
        for row in preview_rows
    )
    sorted_fit = sorted(
        float(_to_float(row.get("best_local_fit_mse", row.get("local_fit_mse", float("inf"))), float("inf")))
        for row in preview_rows
    )
    gap_vals = [
        float(max(0.0, _to_float(row.get("mean_fit_probe_gap", row.get("local_fit_probe_gap", 0.0)), 0.0)))
        for row in preview_rows
    ]
    out.update({
        "mode_diversity": int(len(mode_keys) or 1),
        "best_probe_mse": float(sorted_probe[0]) if sorted_probe else float("inf"),
        "second_probe_mse": float(sorted_probe[1]) if len(sorted_probe) > 1 else (float(sorted_probe[0]) if sorted_probe else float("inf")),
        "mean_probe_mse": float(sum(sorted_probe) / len(sorted_probe)) if sorted_probe else float("inf"),
        "best_fit_mse": float(sorted_fit[0]) if sorted_fit else float("inf"),
        "best_fit_probe_gap": float(max(0.0, _to_float(preview_best_row.get("mean_fit_probe_gap", preview_best_row.get("local_fit_probe_gap", 0.0)), 0.0))),
        "mean_fit_probe_gap": float(sum(gap_vals) / len(gap_vals)) if gap_vals else 0.0,
        "best_support": float(max(1.0, _to_float(preview_best_row.get("provenance_count", 1.0), 1.0))),
        "local_mapping_identity_frac": float(_bucket_frac(local_mapping_rows, "identity")),
        "local_mapping_affine_frac": float(_bucket_frac(local_mapping_rows, "affine")),
        "local_mapping_full_frac": float(_bucket_frac(local_mapping_rows, "full")),
        "local_mapping_other_frac": float(_bucket_frac(local_mapping_rows, "other")),
        "target_mapping_identity_frac": float(_bucket_frac(target_mapping_rows, "identity")),
        "target_mapping_affine_frac": float(_bucket_frac(target_mapping_rows, "affine")),
        "target_mapping_full_frac": float(_bucket_frac(target_mapping_rows, "full")),
        "target_mapping_other_frac": float(_bucket_frac(target_mapping_rows, "other")),
        "best_child_size": float(_to_float(preview_best_row.get("candidate_child_size", 0.0), 0.0)),
        "mean_child_size": float(sum(child_sizes) / len(child_sizes)) if child_sizes else 0.0,
        "best_child_depth": float(_to_float(preview_best_row.get("candidate_child_depth", 0.0), 0.0)),
        "mean_child_depth": float(sum(child_depths) / len(child_depths)) if child_depths else 0.0,
    })
    return out


def _summarize_learned_route_prediction(
    pred: Mapping[str, Any] | None,
    *,
    score_keys: Sequence[str],
) -> dict[str, float]:
    if not isinstance(pred, Mapping) or not bool(pred.get("trained", False)):
        return {
            "best_score": 0.0,
            "second_score": 0.0,
            "mean_score": 0.0,
            "margin": 0.0,
            "state_value": 0.0,
        }
    rows = [dict(row) for row in list(pred.get("rows", []) or []) if isinstance(row, Mapping)]
    scores: list[float] = []
    for row in rows:
        score = None
        for key in score_keys:
            if key in row and _finite_float_or_none(row.get(key, None)) is not None:
                score = float(_to_float(row.get(key, 0.0), 0.0))
                break
        if score is None:
            score = 0.0
        scores.append(float(score))
    if not scores:
        return {
            "best_score": 0.0,
            "second_score": 0.0,
            "mean_score": 0.0,
            "margin": 0.0,
            "state_value": float(_to_float(pred.get("state_value_estimate", 0.0), 0.0)),
        }
    scores_sorted = sorted(scores, reverse=True)
    best_score = float(scores_sorted[0])
    second_score = float(scores_sorted[1]) if len(scores_sorted) > 1 else float(scores_sorted[0])
    return {
        "best_score": best_score,
        "second_score": second_score,
        "mean_score": float(sum(scores_sorted) / len(scores_sorted)),
        "margin": float(best_score - second_score),
        "state_value": float(_to_float(pred.get("state_value_estimate", 0.0), 0.0)),
    }


def _build_repair_build_route_feature_dict(
    record: Any,
    *,
    repair_summary: Mapping[str, Any],
    build_summary: Mapping[str, Any],
    repair_learned_summary: Mapping[str, Any] | None = None,
    build_learned_summary: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    features = extract_repair_critic_features(record)
    for name in REPAIR_CRITIC_FEATURE_NAMES:
        features.setdefault(name, 0.0)
    route_features = dict(features)
    repair_best_action = str(repair_summary.get("best_action", "") or "")
    build_best_action = str(build_summary.get("best_action", "") or "")
    route_features.update({
        "repair_total_count_log": float(_safe_log1p(repair_summary.get("total_count", 0.0))),
        "repair_path_diversity_log": float(_safe_log1p(repair_summary.get("path_diversity", 0.0))),
        "repair_mode_diversity_log": float(_safe_log1p(repair_summary.get("mode_diversity", 0.0))),
        "repair_best_probe_mse_log": float(_safe_neg_log10(repair_summary.get("best_probe_mse", float("inf")))),
        "repair_second_probe_mse_log": float(_safe_neg_log10(repair_summary.get("second_probe_mse", float("inf")))),
        "repair_mean_probe_mse_log": float(_safe_neg_log10(repair_summary.get("mean_probe_mse", float("inf")))),
        "repair_best_fit_mse_log": float(_safe_neg_log10(repair_summary.get("best_fit_mse", float("inf")))),
        "repair_best_fit_probe_gap_log": float(_safe_log1p(repair_summary.get("best_fit_probe_gap", 0.0))),
        "repair_mean_fit_probe_gap_log": float(_safe_log1p(repair_summary.get("mean_fit_probe_gap", 0.0))),
        "repair_best_support_log": float(_safe_log1p(repair_summary.get("best_support", 0.0))),
        "repair_best_action_inv_steer": 1.0 if repair_best_action == "inv_steer" else 0.0,
        "repair_best_action_repair_option": 1.0 if repair_best_action == "repair_option" else 0.0,
        "repair_local_mapping_identity_frac": float(_clamp01(repair_summary.get("local_mapping_identity_frac", 0.0))),
        "repair_local_mapping_affine_frac": float(_clamp01(repair_summary.get("local_mapping_affine_frac", 0.0))),
        "repair_local_mapping_full_frac": float(_clamp01(repair_summary.get("local_mapping_full_frac", 0.0))),
        "repair_local_mapping_other_frac": float(_clamp01(repair_summary.get("local_mapping_other_frac", 0.0))),
        "repair_target_mapping_identity_frac": float(_clamp01(repair_summary.get("target_mapping_identity_frac", 0.0))),
        "repair_target_mapping_affine_frac": float(_clamp01(repair_summary.get("target_mapping_affine_frac", 0.0))),
        "repair_target_mapping_full_frac": float(_clamp01(repair_summary.get("target_mapping_full_frac", 0.0))),
        "repair_target_mapping_other_frac": float(_clamp01(repair_summary.get("target_mapping_other_frac", 0.0))),
        "repair_best_child_size_log": float(_safe_log1p(repair_summary.get("best_child_size", 0.0))),
        "repair_mean_child_size_log": float(_safe_log1p(repair_summary.get("mean_child_size", 0.0))),
        "repair_best_child_depth": float(_to_float(repair_summary.get("best_child_depth", 0.0), 0.0)),
        "repair_mean_child_depth": float(_to_float(repair_summary.get("mean_child_depth", 0.0), 0.0)),
        "build_total_count_log": float(_safe_log1p(build_summary.get("total_count", 0.0))),
        "build_action_diversity_log": float(_safe_log1p(build_summary.get("action_diversity", 0.0))),
        "build_path_diversity_log": float(_safe_log1p(build_summary.get("path_diversity", 0.0))),
        "build_best_action_replace": 1.0 if build_best_action == "replace" else 0.0,
        "build_best_action_wrap_un": 1.0 if build_best_action == "wrap_un" else 0.0,
        "build_best_action_residual": 1.0 if build_best_action == "residual" else 0.0,
        "build_best_action_other": 1.0 if build_best_action not in {"replace", "wrap_un", "residual"} else 0.0,
        "build_best_child_size_log": float(_safe_log1p(build_summary.get("best_child_size", 0.0))),
        "build_mean_child_size_log": float(_safe_log1p(build_summary.get("mean_child_size", 0.0))),
        "build_best_child_depth": float(_to_float(build_summary.get("best_child_depth", 0.0), 0.0)),
        "build_mean_child_depth": float(_to_float(build_summary.get("mean_child_depth", 0.0), 0.0)),
        "build_best_child_size_delta": float(_to_float(build_summary.get("best_child_size_delta", 0.0), 0.0)),
        "build_mean_child_size_delta": float(_to_float(build_summary.get("mean_child_size_delta", 0.0), 0.0)),
        "build_best_path_len_log": float(_safe_log1p(build_summary.get("best_path_len", 0.0))),
        "build_mean_path_len_log": float(_safe_log1p(build_summary.get("mean_path_len", 0.0))),
        "repair_learned_best_score": float(_to_float((repair_learned_summary or {}).get("best_score", 0.0), 0.0)),
        "repair_learned_second_score": float(_to_float((repair_learned_summary or {}).get("second_score", 0.0), 0.0)),
        "repair_learned_mean_score": float(_to_float((repair_learned_summary or {}).get("mean_score", 0.0), 0.0)),
        "repair_learned_margin": float(_to_float((repair_learned_summary or {}).get("margin", 0.0), 0.0)),
        "repair_learned_state_value": float(_to_float((repair_learned_summary or {}).get("state_value", 0.0), 0.0)),
        "build_learned_best_score": float(_to_float((build_learned_summary or {}).get("best_score", 0.0), 0.0)),
        "build_learned_second_score": float(_to_float((build_learned_summary or {}).get("second_score", 0.0), 0.0)),
        "build_learned_mean_score": float(_to_float((build_learned_summary or {}).get("mean_score", 0.0), 0.0)),
        "build_learned_margin": float(_to_float((build_learned_summary or {}).get("margin", 0.0), 0.0)),
        "build_learned_state_value": float(_to_float((build_learned_summary or {}).get("state_value", 0.0), 0.0)),
        "delta_learned_best_score": float(
            _to_float((repair_learned_summary or {}).get("best_score", 0.0), 0.0)
            - _to_float((build_learned_summary or {}).get("best_score", 0.0), 0.0)
        ),
    })
    return {name: float(route_features.get(name, 0.0)) for name in tuple(REPAIR_CRITIC_FEATURE_NAMES) + tuple(REPAIR_ROUTE_COMPARE_EXTRA_FEATURE_NAMES)}


def _build_repair_build_route_rows(
    rows: Sequence[Any],
    *,
    safe_eps: float = 1.0e-12,
    margin_floor: float = 1.0e-3,
    repair_tuple_bundle: dict[str, Any] | None = None,
    build_tuple_bundle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows_list = [coerce_repair_feature_row(row) for row in rows]
    rows_list = [row for row in rows_list if row]
    out: list[dict[str, Any]] = []
    for row in rows_list:
        build_rows = list(row.get("controller_build_slate", []) or [])
        repair_rows = list(row.get("inverse_repair_slate", []) or [])
        if not build_rows or not repair_rows:
            continue
        record = coerce_repair_feature_record(row)
        parent_eff = row.get(
            "estimated_parent_eff_mse",
            getattr(getattr(record, "parent", None), "parent_best_eff_mse", float("inf")),
        )
        repair_grouped = _group_repair_preview_rows([
            dict(slate_row)
            for slate_row in repair_rows
            if isinstance(slate_row, Mapping) and bool(slate_row.get("dedup_kept", True))
        ])
        build_grouped = _group_build_preview_rows([
            dict(slate_row)
            for slate_row in build_rows
            if isinstance(slate_row, Mapping)
        ])
        repair_summary = _summarize_route_candidate_rows(
            repair_grouped,
            parent_eff=parent_eff,
            safe_eps=float(safe_eps),
            build_mode=False,
        )
        build_summary = _summarize_route_candidate_rows(
            build_grouped,
            parent_eff=parent_eff,
            safe_eps=float(safe_eps),
            build_mode=True,
        )
        if repair_summary is None or build_summary is None:
            continue
        repair_learned_summary = None
        if isinstance(repair_tuple_bundle, dict):
            try:
                repair_pred = predict_repair_tuple_slate(
                    repair_tuple_bundle,
                    row,
                    preview_rows=repair_rows,
                )
                repair_learned_summary = _summarize_learned_route_prediction(
                    repair_pred,
                    score_keys=("allocation_estimate", "combined_estimate", "utility_estimate"),
                )
            except Exception:
                repair_learned_summary = None
        build_learned_summary = None
        if isinstance(build_tuple_bundle, dict):
            try:
                build_pred = predict_build_tuple_slate(build_tuple_bundle, row)
                build_learned_summary = _summarize_learned_route_prediction(
                    build_pred,
                    score_keys=("utility_estimate",),
                )
            except Exception:
                build_learned_summary = None
        margin = float(_to_float(repair_summary.get("best_utility", 0.0), 0.0) - _to_float(build_summary.get("best_utility", 0.0), 0.0))
        if abs(margin) <= float(margin_floor):
            route_label = "tie"
        else:
            route_label = "repair" if margin > 0.0 else "build"
        out.append({
            "features": _build_repair_build_route_feature_dict(
                record,
                repair_summary=repair_summary,
                build_summary=build_summary,
                repair_learned_summary=repair_learned_summary,
                build_learned_summary=build_learned_summary,
            ),
            "route_label": str(route_label),
            "route_target": 1 if margin > 0.0 else 0,
            "route_margin": float(margin),
            "sample_weight": float(max(0.25, abs(margin))),
            "repair_summary": dict(repair_summary),
            "build_summary": dict(build_summary),
            "repair_learned_summary": dict(repair_learned_summary or {}),
            "build_learned_summary": dict(build_learned_summary or {}),
        })
    return out


def _build_build_tuple_rows(
    rows: Sequence[Any],
    *,
    score_gap_floor: float = 1.0e-3,
    safe_eps: float = 1.0e-12,
) -> list[dict[str, Any]]:
    rows_list = [coerce_repair_feature_row(row) for row in rows]
    rows_list = [row for row in rows_list if row]
    out: list[dict[str, Any]] = []
    for row in rows_list:
        build_rows = list(row.get("controller_build_slate", []) or [])
        if len(build_rows) < 2:
            continue
        record = coerce_repair_feature_record(row)
        parent_eff = row.get(
            "estimated_parent_eff_mse",
            getattr(getattr(record, "parent", None), "parent_best_eff_mse", float("inf")),
        )
        grouped_rows = _group_build_preview_rows([
            dict(slate_row)
            for slate_row in build_rows
            if isinstance(slate_row, Mapping)
        ])
        if len(grouped_rows) < 2:
            continue
        observed_entries: list[dict[str, Any]] = []
        for preview_row in grouped_rows:
            if not bool(preview_row.get("exact_child_score_observed", False)):
                continue
            utility = _make_repair_slate_utility(
                child_eff_mse=preview_row.get("child_eff_mse", None),
                parent_eff_mse=parent_eff,
                safe_eps=float(safe_eps),
            )
            if utility is None:
                continue
            observed_entries.append({
                "utility": float(utility),
                "child_eff_mse": float(_finite_float_or_none(preview_row.get("child_eff_mse", None)) or 0.0),
                "preview_row": dict(preview_row),
            })
        if len(observed_entries) < 2:
            continue
        pairwise_pairs: list[tuple[int, int, float]] = []
        for ii, entry_a in enumerate(observed_entries):
            for jj in range(ii + 1, len(observed_entries)):
                entry_b = observed_entries[jj]
                gap = float(entry_a["utility"]) - float(entry_b["utility"])
                if abs(gap) <= float(score_gap_floor):
                    continue
                if gap > 0.0:
                    pairwise_pairs.append((int(ii), int(jj), float(gap)))
                else:
                    pairwise_pairs.append((int(jj), int(ii), float(-gap)))
        total_candidate_count = int(len(grouped_rows))
        full_slate = bool(total_candidate_count == len(observed_entries) and len(observed_entries) >= 2)
        if not full_slate and not pairwise_pairs:
            continue
        best_idx = max(
            range(len(observed_entries)),
            key=lambda idx: (
                float(observed_entries[idx]["utility"]),
                -float(observed_entries[idx]["child_eff_mse"]),
                -int(idx),
            ),
        )
        best_utility = float(observed_entries[best_idx]["utility"])
        out.append({
            "features": extract_repair_critic_features(record),
            "preview_rows": [dict(entry["preview_row"]) for entry in observed_entries],
            "preview_utility_targets": [float(entry["utility"]) for entry in observed_entries],
            "pairwise_pairs": pairwise_pairs,
            "full_slate": bool(full_slate),
            "state_value_target": float(best_utility),
            "n_total_tuples": int(total_candidate_count),
            "n_observed_tuples": int(len(observed_entries)),
        })
    return out


def _build_repair_slate_rows(
    rows: Sequence[Any],
    *,
    reward_target: str = "descendant_preferred",
    repair_action_names: Sequence[str] = ("inv_steer", "repair_option"),
    score_gap_floor: float = 1.0e-3,
    safe_eps: float = 1.0e-12,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows_list = [coerce_repair_feature_row(row) for row in rows]
    rows_list = [row for row in rows_list if row]
    allowed_actions = {_normalize_action_name(name) for name in repair_action_names if _normalize_action_name(name)}
    macro_index = {name: idx for idx, name in enumerate(REPAIR_CRITIC_MACRO_ACTION_NAMES)}
    source_specs = (
        ("inverse_repair_slate", "inverse_path_mode_beam", "estimated_parent_eff_mse", "inverse"),
        (
            "repair_option_final_inverse_repair_slate",
            "repair_option_final_inverse_path_mode_beam",
            "repair_option_final_estimated_parent_eff_mse",
            "repair_option_final",
        ),
    )
    reward_source_counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []

    for row in rows_list:
        record = coerce_repair_feature_record(row)
        aux_reward, aux_reward_source = _resolve_actor_critic_reward(row, reward_target=reward_target)
        for slate_key, beam_key, parent_eff_key, slate_source in source_specs:
            slate_rows = list(row.get(slate_key, []) or [])
            if not slate_rows:
                continue
            beam_rows_raw = list(row.get(beam_key, []) or [])
            if beam_rows_raw:
                path_rows = tuple(PathStateFeatures.from_row(path_row) for path_row in beam_rows_raw)
            else:
                path_rows = tuple(record.path_rows)
            if len(path_rows) < 2:
                continue
            parent_eff = row.get(parent_eff_key, None)
            if _finite_float_or_none(parent_eff) is None:
                parent_eff = row.get(
                    "estimated_parent_eff_mse",
                    getattr(getattr(record, "parent", None), "parent_best_eff_mse", float("inf")),
                )
            best_by_index: dict[int, dict[str, Any]] = {}
            observed_pairs = 0
            for slate_row in slate_rows:
                if not isinstance(slate_row, Mapping):
                    continue
                if not bool(slate_row.get("dedup_kept", True)):
                    continue
                if not bool(slate_row.get("exact_child_score_observed", False)):
                    continue
                action_name = _normalize_action_name(slate_row.get("action", ""))
                if action_name not in allowed_actions:
                    continue
                path_idx = _match_path_mode_index(
                    path_rows,
                    target_path=slate_row.get("path", ()),
                    target_mode=slate_row.get("target_mode", None),
                )
                if path_idx is None:
                    continue
                utility = _make_repair_slate_utility(
                    child_eff_mse=slate_row.get("child_eff_mse", None),
                    parent_eff_mse=parent_eff,
                    safe_eps=float(safe_eps),
                )
                if utility is None:
                    continue
                observed_pairs += 1
                prev = best_by_index.get(int(path_idx), None)
                if prev is None or float(utility) > float(prev["utility"]):
                    best_by_index[int(path_idx)] = {
                        "utility": float(utility),
                        "action_name": str(action_name),
                        "action_index": int(macro_index.get(str(action_name), macro_index["inv_steer"])),
                        "child_eff_mse": float(_finite_float_or_none(slate_row.get("child_eff_mse", None)) or 0.0),
                    }
            if len(best_by_index) < 2:
                continue
            ordered_indices = sorted(best_by_index)
            pairwise_pairs: list[tuple[int, int, float]] = []
            for ii, better_idx in enumerate(ordered_indices):
                for worse_idx in ordered_indices[ii + 1:]:
                    util_a = float(best_by_index[better_idx]["utility"])
                    util_b = float(best_by_index[worse_idx]["utility"])
                    gap = util_a - util_b
                    if abs(gap) <= float(score_gap_floor):
                        continue
                    if gap > 0.0:
                        pairwise_pairs.append((int(better_idx), int(worse_idx), float(gap)))
                    else:
                        pairwise_pairs.append((int(worse_idx), int(better_idx), float(-gap)))
            full_slate = bool(len(best_by_index) == len(path_rows))
            if not full_slate and not pairwise_pairs:
                continue
            utility_targets = [0.0] * len(path_rows)
            utility_mask = [False] * len(path_rows)
            action_indices = [int(macro_index["inv_steer"])] * len(path_rows)
            for idx, info in best_by_index.items():
                utility_targets[int(idx)] = float(info["utility"])
                utility_mask[int(idx)] = True
                action_indices[int(idx)] = int(info["action_index"])
            best_idx = max(
                best_by_index,
                key=lambda idx: (
                    float(best_by_index[idx]["utility"]),
                    -float(best_by_index[idx]["child_eff_mse"]),
                    int(idx),
                ),
            )
            if aux_reward_source:
                reward_source_counts[str(aux_reward_source)] = int(reward_source_counts.get(str(aux_reward_source), 0)) + 1
            out.append({
                "features": extract_repair_critic_features(record),
                "path_rows": path_rows,
                "path_target_index": int(best_idx),
                "path_utility_targets": utility_targets,
                "path_utility_mask": utility_mask,
                "path_action_indices": action_indices,
                "pairwise_pairs": pairwise_pairs,
                "full_slate": bool(full_slate),
                "aux_reward": None if aux_reward is None else float(aux_reward),
                "aux_reward_source": str(aux_reward_source or ""),
                "slate_source": str(slate_source),
                "slate_key": str(slate_key),
                "beam_key": str(beam_key),
                "n_observed_paths": int(len(best_by_index)),
                "n_exact_children": int(observed_pairs),
            })
    return out, reward_source_counts


def _build_repair_tuple_rows(
    rows: Sequence[Any],
    *,
    reward_target: str = "descendant_preferred",
    repair_action_names: Sequence[str] = ("inv_steer", "repair_option"),
    score_gap_floor: float = 1.0e-3,
    safe_eps: float = 1.0e-12,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows_list = [coerce_repair_feature_row(row) for row in rows]
    rows_list = [row for row in rows_list if row]
    child_continuation_by_expr: dict[str, float] = {}
    for row in rows_list:
        parent_expr = str(row.get("parent_expr", "") or "").strip()
        if not parent_expr:
            continue
        cont_gain = _finite_float_or_none(row.get("actor_critic_descendant_log_gain", None))
        if cont_gain is None:
            parent_eff = _finite_float_or_none(row.get("parent_eff_mse", None))
            best_desc_eff = _finite_float_or_none(row.get("best_descendant_eff_mse", None))
            if parent_eff is not None and best_desc_eff is not None:
                cont_gain = float(math.log(parent_eff + float(safe_eps)) - math.log(best_desc_eff + float(safe_eps)))
        if cont_gain is None:
            continue
        prev = child_continuation_by_expr.get(parent_expr, None)
        if prev is None or float(cont_gain) > float(prev):
            child_continuation_by_expr[parent_expr] = float(cont_gain)
    allowed_actions = {_normalize_action_name(name) for name in repair_action_names if _normalize_action_name(name)}
    source_specs = (
        ("inverse_repair_slate", "inverse_path_mode_beam", "estimated_parent_eff_mse", "inverse"),
        (
            "repair_option_final_inverse_repair_slate",
            "repair_option_final_inverse_path_mode_beam",
            "repair_option_final_estimated_parent_eff_mse",
            "repair_option_final",
        ),
    )
    reward_source_counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []

    for row in rows_list:
        record = coerce_repair_feature_record(row)
        aux_reward, aux_reward_source = _resolve_actor_critic_reward(row, reward_target=reward_target)
        for slate_key, beam_key, parent_eff_key, slate_source in source_specs:
            slate_rows = list(row.get(slate_key, []) or [])
            if not slate_rows:
                continue
            beam_rows_raw = list(row.get(beam_key, []) or [])
            if beam_rows_raw:
                path_rows = tuple(PathStateFeatures.from_row(path_row) for path_row in beam_rows_raw)
            else:
                path_rows = tuple(record.path_rows)
            if not path_rows:
                continue
            parent_eff = row.get(parent_eff_key, None)
            if _finite_float_or_none(parent_eff) is None:
                parent_eff = row.get(
                    "estimated_parent_eff_mse",
                    getattr(getattr(record, "parent", None), "parent_best_eff_mse", float("inf")),
                )

            grouped_slate_rows = _group_repair_preview_rows([
                dict(slate_row)
                for slate_row in slate_rows
                if isinstance(slate_row, Mapping)
                and _normalize_action_name(slate_row.get("action", "")) in allowed_actions
            ])
            if not grouped_slate_rows:
                continue

            total_candidate_count = int(len(grouped_slate_rows))
            observed_entries: list[dict[str, Any]] = []
            for slate_row in grouped_slate_rows:
                action_name = _normalize_action_name(slate_row.get("action", ""))
                path_idx = _match_path_mode_index(
                    path_rows,
                    target_path=slate_row.get("path", ()),
                    target_mode=slate_row.get("target_mode", None),
                )
                if path_idx is None:
                    continue
                if not bool(slate_row.get("exact_child_score_observed", False)):
                    continue
                utility = _make_repair_slate_utility(
                    child_eff_mse=slate_row.get("child_eff_mse", None),
                    parent_eff_mse=parent_eff,
                    safe_eps=float(safe_eps),
                )
                if utility is None:
                    continue
                child_expr = str(slate_row.get("child_expr", "") or slate_row.get("child_key", "") or "").strip()
                continuation_value = child_continuation_by_expr.get(child_expr, None)
                observed_entries.append({
                    "path_index": int(path_idx),
                    "utility": float(utility),
                    "continuation_value": None if continuation_value is None else float(continuation_value),
                    "action_name": str(action_name),
                    "child_eff_mse": float(_finite_float_or_none(slate_row.get("child_eff_mse", None)) or 0.0),
                    "preview_row": dict(slate_row),
                })
            if len(observed_entries) < 2:
                continue

            pairwise_pairs: list[tuple[int, int, float]] = []
            for ii, entry_a in enumerate(observed_entries):
                for jj in range(ii + 1, len(observed_entries)):
                    entry_b = observed_entries[jj]
                    gap = float(entry_a["utility"]) - float(entry_b["utility"])
                    if abs(gap) <= float(score_gap_floor):
                        continue
                    if gap > 0.0:
                        pairwise_pairs.append((int(ii), int(jj), float(gap)))
                    else:
                        pairwise_pairs.append((int(jj), int(ii), float(-gap)))

            full_slate = bool(total_candidate_count == len(observed_entries) and len(observed_entries) >= 2)
            if not full_slate and not pairwise_pairs:
                continue
            best_idx = max(
                range(len(observed_entries)),
                key=lambda idx: (
                    float(observed_entries[idx]["utility"]),
                    -float(observed_entries[idx]["child_eff_mse"]),
                    -int(observed_entries[idx]["path_index"]),
                    -int(idx),
                ),
            )
            best_utility = float(observed_entries[best_idx]["utility"])
            if aux_reward_source:
                reward_source_counts[str(aux_reward_source)] = int(reward_source_counts.get(str(aux_reward_source), 0)) + 1
            out.append({
                "features": extract_repair_critic_features(record),
                "path_rows": path_rows,
                "path_target_index": int(observed_entries[best_idx]["path_index"]),
                "preview_rows": [dict(entry["preview_row"]) for entry in observed_entries],
                "preview_path_indices": [int(entry["path_index"]) for entry in observed_entries],
                "preview_utility_targets": [float(entry["utility"]) for entry in observed_entries],
                "preview_value_targets": [
                    0.0 if entry["continuation_value"] is None else float(entry["continuation_value"])
                    for entry in observed_entries
                ],
                "preview_value_mask": [bool(entry["continuation_value"] is not None) for entry in observed_entries],
                "preview_regret_targets": [
                    max(0.0, best_utility - float(entry["utility"]))
                    for entry in observed_entries
                ],
                "pairwise_pairs": pairwise_pairs,
                "full_slate": bool(full_slate),
                "state_value_target": float(best_utility),
                "aux_reward": None if aux_reward is None else float(aux_reward),
                "aux_reward_source": str(aux_reward_source or ""),
                "slate_source": str(slate_source),
                "slate_key": str(slate_key),
                "beam_key": str(beam_key),
                "n_total_tuples": int(total_candidate_count),
                "n_observed_tuples": int(len(observed_entries)),
            "n_grouped_support_rows": int(sum(
                int(_to_float(entry["preview_row"].get("provenance_count", 1.0), 1.0))
                for entry in observed_entries
            )),
        })
    return out, reward_source_counts


def _build_unified_candidate_rows(
    rows: Sequence[Any],
    *,
    score_gap_floor: float = 1.0e-3,
    safe_eps: float = 1.0e-12,
    common_value_weight: float = 0.25,
    common_regret_weight: float = 1.0,
    repair_action_names: Sequence[str] = ("inv_steer", "repair_option"),
    build_action_names: Sequence[str] = ("replace", "wrap_un", "residual"),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows_list = [coerce_repair_feature_row(row) for row in rows]
    rows_list = [row for row in rows_list if row]
    out: list[dict[str, Any]] = []
    reward_source_counts: dict[str, int] = {}
    relation_index = {name: idx for idx, name in enumerate(REPAIR_CRITIC_PATH_RELATION_NAMES)}
    mode_index = {name: idx for idx, name in enumerate(REPAIR_CRITIC_MODE_NAMES)}
    child_continuation_by_expr: dict[str, float] = {}
    for row in rows_list:
        parent_expr = str(row.get("parent_expr", "") or "").strip()
        if not parent_expr:
            continue
        cont_gain = _finite_float_or_none(row.get("actor_critic_descendant_log_gain", None))
        if cont_gain is None:
            parent_eff = _finite_float_or_none(row.get("parent_eff_mse", None))
            best_desc_eff = _finite_float_or_none(row.get("best_descendant_eff_mse", None))
            if parent_eff is not None and best_desc_eff is not None:
                cont_gain = float(math.log(parent_eff + float(safe_eps)) - math.log(best_desc_eff + float(safe_eps)))
        if cont_gain is None:
            continue
        prev = child_continuation_by_expr.get(parent_expr, None)
        if prev is None or float(cont_gain) > float(prev):
            child_continuation_by_expr[parent_expr] = float(cont_gain)
    allowed_repair_actions = {_normalize_action_name(v) for v in repair_action_names}
    allowed_build_actions = {_normalize_action_name(v) for v in build_action_names}
    for row in rows_list:
        record = coerce_repair_feature_record(row)
        path_rows = tuple(record.path_rows)
        if not path_rows:
            continue
        oracle_relation_targets = []
        oracle_mode_targets = []
        for path_row in path_rows:
            relation_raw = getattr(path_row, "oracle_relation_to_reference", None)
            relation_name = str(relation_raw or "").strip()
            oracle_relation_targets.append(
                relation_index.get(_normalize_relation_name(relation_name), -100) if relation_name else -100
            )
            mode_raw = getattr(path_row, "oracle_best_mode", None)
            mode_name = str(mode_raw or "").strip()
            oracle_mode_targets.append(
                mode_index.get(_normalize_mode_name(mode_name), -100) if mode_name else -100
            )
        oracle_path_target_index = row.get("oracle_truth_path_index", None)
        if not isinstance(oracle_path_target_index, int) or not (0 <= oracle_path_target_index < len(path_rows)):
            oracle_path_target_index = None
            for idx, path_row in enumerate(path_rows):
                if bool(getattr(path_row, "oracle_is_reference_path", False)):
                    oracle_path_target_index = int(idx)
                    break
        parent_eff = row.get(
            "estimated_parent_eff_mse",
            getattr(getattr(record, "parent", None), "parent_best_eff_mse", float("inf")),
        )
        repair_rows = _group_repair_preview_rows([
            shared_candidate_row_dict(
                {
                    **dict(slate_row),
                    "route_source": "repair",
                },
                route_source="repair",
            )
            for slate_row in list(row.get("inverse_repair_slate", []) or [])
            if isinstance(slate_row, Mapping)
            and bool(slate_row.get("dedup_kept", True))
            and _normalize_action_name(slate_row.get("action", "")) in allowed_repair_actions
        ])
        build_rows = _group_build_preview_rows([
            shared_candidate_row_dict(
                {
                    **dict(slate_row),
                    "route_source": "build",
                },
                route_source="build",
            )
            for slate_row in list(row.get("controller_build_slate", []) or [])
            if isinstance(slate_row, Mapping)
            and _normalize_action_name(slate_row.get("action", "")) in allowed_build_actions
        ])
        total_candidate_count = int(len(repair_rows) + len(build_rows))
        if total_candidate_count < 2:
            continue
        observed_entries: list[dict[str, Any]] = []
        aux_reward, aux_reward_source = _resolve_actor_critic_reward(row, reward_target="descendant_preferred")
        for preview_row in list(repair_rows) + list(build_rows):
            if not bool(preview_row.get("exact_child_score_observed", False)):
                continue
            utility = _make_repair_slate_utility(
                child_eff_mse=preview_row.get("child_eff_mse", None),
                parent_eff_mse=parent_eff,
                safe_eps=float(safe_eps),
            )
            if utility is None:
                continue
            path_idx = _match_candidate_path_index(path_rows, preview_row)
            if path_idx is None:
                continue
            child_expr = str(preview_row.get("child_expr", "") or preview_row.get("child_key", "") or "").strip()
            continuation_value = child_continuation_by_expr.get(child_expr, None)
            observed_entries.append({
                "utility": float(utility),
                "continuation_value": None if continuation_value is None else float(continuation_value),
                "route_name": str(preview_row.get("route_source", "") or ""),
                "path_index": int(path_idx),
                "action_name": _normalize_action_name(preview_row.get("action", "")),
                "child_eff_mse": float(_finite_float_or_none(preview_row.get("child_eff_mse", None)) or 0.0),
                "oracle_is_truth_candidate": bool(preview_row.get("oracle_is_truth_candidate", False)) if "oracle_is_truth_candidate" in preview_row else None,
                "oracle_mode_is_best": bool(preview_row.get("oracle_mode_is_best", False)) if "oracle_mode_is_best" in preview_row else None,
                "oracle_truth_rank_score": (
                    _finite_float_or_none(preview_row.get("oracle_truth_rank_score", None))
                    if "oracle_truth_rank_score" in preview_row
                    else (
                        None
                        if preview_row.get("oracle_truth_rank", None) is None
                        else float(1.0 / (1.0 + max(0.0, _to_float(preview_row.get("oracle_truth_rank", None), 0.0))))
                    )
                ),
                "oracle_mapping_stable": bool(preview_row.get("oracle_mapping_stable", False)) if "oracle_mapping_stable" in preview_row else None,
                "preview_row": dict(preview_row),
            })
        if len(observed_entries) < 2:
            continue
        pairwise_pairs: list[tuple[int, int, float]] = []
        for ii, entry_a in enumerate(observed_entries):
            for jj in range(ii + 1, len(observed_entries)):
                entry_b = observed_entries[jj]
                gap = float(entry_a["utility"]) - float(entry_b["utility"])
                if abs(gap) <= float(score_gap_floor):
                    continue
                if gap > 0.0:
                    pairwise_pairs.append((int(ii), int(jj), float(gap)))
                else:
                    pairwise_pairs.append((int(jj), int(ii), float(-gap)))
        full_slate = bool(total_candidate_count == len(observed_entries) and len(observed_entries) >= 2)
        if not full_slate and not pairwise_pairs:
            continue
        best_idx = max(
            range(len(observed_entries)),
            key=lambda idx: (
                float(observed_entries[idx]["utility"]),
                -float(observed_entries[idx]["child_eff_mse"]),
                str(observed_entries[idx]["route_name"]),
                -int(observed_entries[idx]["path_index"]),
                -int(idx),
            ),
        )
        best_utility = float(observed_entries[best_idx]["utility"])
        repair_entries = [entry for entry in observed_entries if str(entry["route_name"]) == "repair"]
        build_entries = [entry for entry in observed_entries if str(entry["route_name"]) == "build"]
        repair_best = (
            max(
                repair_entries,
                key=lambda entry: (
                    float(entry["utility"]),
                    -float(entry["child_eff_mse"]),
                    -int(entry["path_index"]),
                ),
            )
            if repair_entries
            else None
        )
        build_best = (
            max(
                build_entries,
                key=lambda entry: (
                    float(entry["utility"]),
                    -float(entry["child_eff_mse"]),
                    -int(entry["path_index"]),
                ),
            )
            if build_entries
            else None
        )
        oracle_truth_in_slate = bool(row.get("oracle_truth_in_slate", any(entry.get("oracle_is_truth_candidate", None) is True for entry in observed_entries)))
        q_targets = [
            _common_candidate_q_target(
                float(entry["utility"]),
                entry["continuation_value"],
                max(0.0, best_utility - float(entry["utility"])),
                continuation_weight=float(common_value_weight),
                regret_weight=float(common_regret_weight),
            )
            for entry in observed_entries
        ]
        best_common_q = max(q_targets) if q_targets else float(best_utility)
        if aux_reward_source:
            reward_source_counts[str(aux_reward_source)] = int(reward_source_counts.get(str(aux_reward_source), 0)) + 1
        out.append({
            "features": extract_repair_critic_features(record),
            "path_rows": path_rows,
            "path_target_index": int(observed_entries[best_idx]["path_index"]),
            "repair_path_target_index": None if repair_best is None else int(repair_best["path_index"]),
            "preview_rows": [dict(entry["preview_row"]) for entry in observed_entries],
            "preview_path_indices": [int(entry["path_index"]) for entry in observed_entries],
            "preview_utility_targets": [float(entry["utility"]) for entry in observed_entries],
            "preview_value_targets": [
                0.0 if entry["continuation_value"] is None else float(entry["continuation_value"])
                for entry in observed_entries
            ],
            "preview_value_mask": [bool(entry["continuation_value"] is not None) for entry in observed_entries],
            "preview_regret_targets": [max(0.0, best_utility - float(entry["utility"])) for entry in observed_entries],
            "preview_q_targets": [float(v) for v in q_targets],
            "preview_route_targets": [str(entry["route_name"]) for entry in observed_entries],
            "oracle_path_target_index": None if oracle_path_target_index is None else int(oracle_path_target_index),
            "oracle_relation_targets": [int(v) for v in oracle_relation_targets],
            "oracle_mode_targets": [int(v) for v in oracle_mode_targets],
            "oracle_truth_in_slate": bool(oracle_truth_in_slate),
            "oracle_truth_in_slate_mask": bool(
                "oracle_truth_in_slate" in row
                or any(entry.get("oracle_is_truth_candidate", None) is not None for entry in observed_entries)
            ),
            "preview_oracle_truth_targets": [
                1.0 if entry.get("oracle_is_truth_candidate", None) is True else 0.0
                for entry in observed_entries
            ],
            "preview_oracle_truth_mask": [
                bool(entry.get("oracle_is_truth_candidate", None) is not None)
                for entry in observed_entries
            ],
            "preview_oracle_mode_best_targets": [
                1.0 if entry.get("oracle_mode_is_best", None) is True else 0.0
                for entry in observed_entries
            ],
            "preview_oracle_mode_best_mask": [
                bool(entry.get("oracle_mode_is_best", None) is not None)
                for entry in observed_entries
            ],
            "preview_oracle_rank_targets": [
                0.0 if entry.get("oracle_truth_rank_score", None) is None else float(entry["oracle_truth_rank_score"])
                for entry in observed_entries
            ],
            "preview_oracle_rank_mask": [
                bool(entry.get("oracle_truth_rank_score", None) is not None)
                for entry in observed_entries
            ],
            "preview_oracle_stability_targets": [
                1.0 if entry.get("oracle_mapping_stable", None) is True else 0.0
                for entry in observed_entries
            ],
            "preview_oracle_stability_mask": [
                bool(entry.get("oracle_mapping_stable", None) is not None)
                for entry in observed_entries
            ],
            "pairwise_pairs": pairwise_pairs,
            "full_slate": bool(full_slate),
            "state_value_target": float(best_utility),
            "common_state_value_target": float(best_common_q),
            "repair_state_value_target": 0.0 if repair_best is None else float(repair_best["utility"]),
            "repair_state_value_mask": bool(repair_best is not None),
            "build_state_value_target": 0.0 if build_best is None else float(build_best["utility"]),
            "build_state_value_mask": bool(build_best is not None),
            "route_target": str(observed_entries[best_idx]["route_name"]),
            "aux_reward": None if aux_reward is None else float(aux_reward),
            "aux_reward_source": str(aux_reward_source or ""),
            "n_total_tuples": int(total_candidate_count),
            "n_observed_tuples": int(len(observed_entries)),
            "n_repair_candidates": int(len(repair_rows)),
            "n_build_candidates": int(len(build_rows)),
        })
    return out, reward_source_counts


def _build_actor_critic_rows(
    rows: Sequence[Any],
    *,
    reward_target: str = "descendant_preferred",
) -> list[dict[str, Any]]:
    rows_list = [coerce_repair_feature_row(row) for row in rows]
    rows_list = [row for row in rows_list if row]
    out: list[dict[str, Any]] = []
    for row in rows_list:
        macro_action = _extract_macro_action_label(row)
        if not isinstance(macro_action, str) or macro_action not in _REPAIR_CRITIC_DEFAULT_MACRO_ACTION_SET:
            continue
        reward_f, reward_source = _resolve_actor_critic_reward(
            row,
            reward_target=reward_target,
        )
        if reward_f is None:
            continue
        record = coerce_repair_feature_record(row)
        path_target_index = _match_selected_path_index(record)
        out.append({
            "features": extract_repair_critic_features(record),
            "path_rows": tuple(record.path_rows),
            "path_target_index": path_target_index,
            "selected_path": tuple(int(v) for v in (record.candidate.selected_path or ())),
            "selected_target_mode": str(record.selected_target_mode or ""),
            "macro_action": macro_action,
            "route_name": _route_name_for_action(macro_action),
            "actor_critic_reward": float(reward_f),
            "actor_critic_reward_source": str(reward_source or ""),
        })
    return out
