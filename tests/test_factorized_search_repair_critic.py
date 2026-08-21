# SPDX-License-Identifier: MPL-2.0

import json

from nestynet_sr.sr_search.factorized_search.actor_critic_train import run_actor_critic_training
from nestynet_sr.sr_search.factorized_search.build_tuple_train import run_build_tuple_training
from nestynet_sr.sr_search.factorized_search.repair_route_train import run_repair_route_training
from nestynet_sr.sr_search.factorized_search.repair_slate_train import run_repair_slate_training
from nestynet_sr.sr_search.factorized_search.repair_tuple_train import run_repair_tuple_training
from nestynet_sr.sr_search.factorized_search.shared_candidate import (
    SHARED_CANDIDATE_MASK_FIELD_NAMES,
    coerce_shared_candidate_record,
    shared_candidate_row_dict,
)
from nestynet_sr.sr_search.factorized_search.shared_candidate_train import run_shared_candidate_training
from nestynet_sr.sr_search.factorized_search.unified_candidate_train import run_unified_candidate_training
from nestynet_sr.sr_search.factorized_search import explorer
from nestynet_sr.sr_search.factorized_search.policy.features import RepairControllerFeatureRecord
from nestynet_sr.sr_search.factorized_search.repair_critic import (
    _build_actor_critic_rows,
    _build_build_tuple_rows,
    _build_repair_build_route_rows,
    _build_repair_slate_rows,
    _build_repair_tuple_rows,
    _build_unified_candidate_rows,
    build_preview_feature_vector,
    collect_repair_critic_examples,
    extract_repair_critic_features,
    load_inverse_experiment_rows,
    load_repair_critic_bundle,
    pretrain_repair_controller_from_oracle_tasks,
    predict_build_tuple_slate,
    predict_repair_controller_heads,
    predict_repair_build_route,
    predict_shared_candidate_dual_slate,
    predict_repair_tuple_slate,
    predict_unified_candidate_slate,
    predict_repair_critic,
    save_repair_critic_bundle,
    train_repair_build_route_comparator,
    train_build_tuple_ranker,
    train_shared_candidate_dual_ranker,
    train_unified_candidate_ranker,
    train_repair_controller_slate_ranker,
    train_repair_controller_tuple_ranker,
    train_repair_controller_actor_critic,
    train_repair_critic,
)


def _row(idx: int, *, good: bool) -> dict:
    if good:
        return {
            "parent_expr": f"good_{idx}",
            "status": "scored",
            "path_entropy": 0.08,
            "path_top_mass": 0.92,
            "path_second_mass": 0.05,
            "path_positive_count": 3,
            "gate_best_weighted_rel_gain": 0.85,
            "gate_best_rel_gain": 0.75,
            "gate_best_valid_frac": 0.90,
            "gate_best_confidence": 0.88,
            "gate_best_transport_rel": 0.72,
            "gate_best_static_score": 0.94,
            "gate_best_branch_factor": 1.00,
            "gate_best_cut_factor": 0.95,
            "gate_best_profile_exact_monotone": True,
            "gate_best_profile_has_periodic": False,
            "gate_best_profile_has_muldiv": True,
            "gate_best_profile_has_explogsqrt": False,
            "selected_target_mode": "identity",
            "selected_path_gain": 0.92,
            "selected_path_gain_pre_cut": 0.96,
            "selected_rel_gain": 0.81,
            "selected_transport_rel": 0.74,
            "selected_lin_rel": 0.63,
            "selected_branch_factor": 1.00,
            "selected_cut_factor": 0.96,
            "selected_effective_n": 18.0,
            "local_candidate_count": 5,
            "estimated_one_hole_rel_improve_eff": 0.78,
            "proxy_one_hole_potential_eff": 0.75,
            "identity_vs_full_log_mse_contrast": 3.8,
            "affine_vs_full_log_mse_contrast": 2.7,
            "parent_best_eff_mse": 1.0e-3,
            "parent_best_raw_mse": 2.0e-3,
            "parent_stagnation_score": 0.55,
            "parent_stagnation_ratio": 0.48,
            "parent_visits": 11,
            "parent_visits_since_improve": 6,
            "gate_allowed": True,
            "accepted": True,
            "created_new_residual_basin": True,
            "became_global_best": idx % 2 == 0,
            "reward": 0.55 + 0.03 * idx,
            "reward_per_s": 0.12 + 0.01 * idx,
        }
    return {
        "parent_expr": f"bad_{idx}",
        "status": "proposal_none" if idx % 2 == 0 else "scored",
        "path_entropy": 1.05,
        "path_top_mass": 0.18,
        "path_second_mass": 0.16,
        "path_positive_count": 5,
        "gate_best_weighted_rel_gain": 0.04,
        "gate_best_rel_gain": 0.05,
        "gate_best_valid_frac": 0.31,
        "gate_best_confidence": 0.26,
        "gate_best_transport_rel": 0.08,
        "gate_best_static_score": 0.62,
        "gate_best_branch_factor": 0.70,
        "gate_best_cut_factor": 0.55,
        "gate_best_profile_exact_monotone": False,
        "gate_best_profile_has_periodic": True,
        "gate_best_profile_has_muldiv": False,
        "gate_best_profile_has_explogsqrt": True,
        "selected_target_mode": "full",
        "selected_path_gain": 0.05,
        "selected_path_gain_pre_cut": 0.09,
        "selected_rel_gain": 0.07,
        "selected_transport_rel": 0.05,
        "selected_lin_rel": 0.0,
        "selected_branch_factor": 0.72,
        "selected_cut_factor": 0.60,
        "selected_effective_n": 6.0,
        "local_candidate_count": 13,
        "estimated_one_hole_rel_improve_eff": 0.04,
        "proxy_one_hole_potential_eff": 0.03,
        "identity_vs_full_log_mse_contrast": 0.2,
        "affine_vs_full_log_mse_contrast": 0.1,
        "parent_best_eff_mse": 0.4,
        "parent_best_raw_mse": 0.6,
        "parent_stagnation_score": 0.10,
        "parent_stagnation_ratio": 0.08,
        "parent_visits": 4,
        "parent_visits_since_improve": 1,
        "gate_allowed": True,
        "accepted": False if idx % 2 else None,
        "created_new_residual_basin": False,
        "became_global_best": False,
        "reward": -0.04 if idx % 2 else None,
        "reward_per_s": -0.02 if idx % 2 else None,
    }


def _policy_row(idx: int, *, repair: bool) -> dict:
    row = _row(idx, good=repair)
    path_repair = {
        "path": [1],
        "target_mode": "identity",
        "weighted_rel_gain": 1.10 if repair else 0.08,
        "weighted_rel_gain_raw": 1.05 if repair else 0.06,
        "weighted_rel_gain_pre_cut": 1.18 if repair else 0.10,
        "rel_gain": 0.92 if repair else 0.07,
        "valid_frac": 0.96 if repair else 0.42,
        "confidence": 0.93 if repair else 0.38,
        "static_score": 1.25 if repair else 0.55,
        "transport_rel": 0.78 if repair else 0.12,
        "branch_factor": 1.0,
        "cut_factor": 1.0,
        "branch_support": 1.0,
        "branch_positive_count": 2,
        "family_scale": 1.0,
        "min_valid_frac_eff": 0.80,
        "min_confidence_eff": 0.75,
        "profile_exact_monotone": True,
        "profile_has_muldiv": True,
    }
    path_build = {
        "path": [2],
        "target_mode": "full",
        "weighted_rel_gain": 0.10 if repair else 1.05,
        "weighted_rel_gain_raw": 0.08 if repair else 1.00,
        "weighted_rel_gain_pre_cut": 0.12 if repair else 1.14,
        "rel_gain": 0.09 if repair else 0.88,
        "valid_frac": 0.40 if repair else 0.95,
        "confidence": 0.35 if repair else 0.91,
        "static_score": 0.50 if repair else 1.18,
        "transport_rel": 0.10 if repair else 0.74,
        "branch_factor": 1.0,
        "cut_factor": 1.0,
        "branch_support": 1.0,
        "branch_positive_count": 2,
        "family_scale": 1.0,
        "min_valid_frac_eff": 0.78,
        "min_confidence_eff": 0.72,
        "profile_has_periodic": True,
        "profile_has_explogsqrt": True,
    }
    row.update({
        "controller_policy_action": "repair_option" if repair else "replace",
        "macro_action": "repair_option" if repair else "replace",
        "selected_path": [1] if repair else [2],
        "selected_target_mode": "identity" if repair else "full",
        "path_summaries": [path_repair, path_build],
    })
    return row


def _oracle_task(idx: int, *, repair: bool) -> dict:
    controller_row = _policy_row(idx, repair=repair)
    return {
        "controller_row": controller_row,
        "target_path": [1] if repair else [2],
        "target_path_index": 0 if repair else 1,
        "path_labels": [
            {
                "path": [1],
                "relation": "same" if repair else "disjoint",
                "best_mode": "identity",
                "improvement_estimate": 0.92 if repair else 0.10,
            },
            {
                "path": [2],
                "relation": "descendant" if repair else "same",
                "best_mode": "full",
                "improvement_estimate": 0.18 if repair else 0.88,
            },
        ],
    }


def _actor_critic_row(idx: int, *, repair: bool) -> dict:
    row = _policy_row(idx, repair=repair)
    reward = 1.0 + 0.05 * idx if repair else -0.8 - 0.03 * idx
    row.update({
        "status": "scored",
        "accepted": bool(repair),
        "created_new_residual_basin": bool(repair),
        "became_global_best": bool(repair and (idx % 3 == 0)),
        "actor_critic_reward": float(reward),
        "actor_critic_reward_log_gain": float(reward),
        "actor_critic_reward_novelty_bonus": 0.0,
        "actor_critic_reward_best_bonus": 0.0,
        "actor_critic_reward_time_penalty": 0.0,
        "actor_critic_reward_wall_s": 0.1,
        "macro_action": "repair_option" if repair else "replace",
        "controller_policy_action": "repair_option" if repair else "replace",
    })
    return row


def _repair_slate_row(
    idx: int,
    *,
    best_path: int = 1,
    partial: bool = False,
) -> dict:
    row = _policy_row(idx, repair=True)
    path_rows = list(row["path_summaries"])
    path_rows.append({
        "path": [3],
        "target_mode": "affine",
        "weighted_rel_gain": 0.42 if best_path != 3 else 1.08,
        "weighted_rel_gain_raw": 0.38 if best_path != 3 else 1.02,
        "weighted_rel_gain_pre_cut": 0.46 if best_path != 3 else 1.16,
        "rel_gain": 0.33 if best_path != 3 else 0.87,
        "valid_frac": 0.76 if best_path != 3 else 0.93,
        "confidence": 0.72 if best_path != 3 else 0.90,
        "static_score": 0.88 if best_path != 3 else 1.21,
        "transport_rel": 0.44 if best_path != 3 else 0.71,
        "branch_factor": 1.0,
        "cut_factor": 1.0,
        "branch_support": 1.0,
        "branch_positive_count": 2,
        "family_scale": 1.0,
        "min_valid_frac_eff": 0.75,
        "min_confidence_eff": 0.71,
    })
    row["path_summaries"] = path_rows
    mode_by_path = {1: "identity", 2: "full", 3: "affine"}
    child_eff = {
        1: 0.30,
        2: 0.24,
        3: 0.28,
    }
    child_eff[int(best_path)] = 0.05
    row.update({
        "status": "scored",
        "estimated_parent_eff_mse": 1.0,
        "selected_path": [int(best_path)],
        "selected_target_mode": mode_by_path[int(best_path)],
        "actor_critic_reward": 0.9 + 0.05 * idx,
        "actor_critic_descendant_reward": 1.1 + 0.05 * idx,
        "controller_policy_action": "inv_steer",
        "macro_action": "inv_steer",
    })
    observed_paths = [1, 2] if partial else [1, 2, 3]
    row["inverse_repair_slate_id"] = f"slate_{idx}"
    row["inverse_repair_slate"] = [
        {
            "slate_id": f"slate_{idx}",
            "tuple_provenance": "beam_local_repair",
            "path": [path_idx],
            "target_mode": mode_by_path[path_idx],
            "action": "inv_steer",
            "beam_rank": int(path_idx - 1),
            "local_rank": 0,
            "local_probe_mse": child_eff[path_idx] * 0.8,
            "local_fit_mse": child_eff[path_idx] * 0.7,
            "candidate_subtree_size": 2 + path_idx,
            "candidate_subtree_depth": 1 + path_idx,
            "candidate_subtree_size_delta": path_idx - 2,
            "candidate_subtree_depth_delta": path_idx - 2,
            "candidate_child_size": 5 + path_idx,
            "candidate_child_depth": 2 + path_idx,
            "candidate_child_size_delta": path_idx - 2,
            "candidate_child_depth_delta": path_idx - 2,
            "candidate_root_op": "add" if path_idx == 1 else ("mul" if path_idx == 2 else "sin"),
            "local_candidate_count": len(observed_paths),
            "child_key": f"{idx}_{path_idx}",
            "child_eff_mse": child_eff[path_idx],
            "dedup_kept": True,
            "exact_child_score_observed": True,
        }
        for path_idx in observed_paths
    ]
    return row


def _repair_tuple_row(
    idx: int,
    *,
    best_tuple: tuple[int, int] = (1, 1),
    partial: bool = False,
) -> dict:
    row = _repair_slate_row(idx, best_path=best_tuple[0], partial=False)
    row["parent_expr"] = f"tuple_state_{idx}"
    row["parent_eff_mse"] = float(row.get("estimated_parent_eff_mse", 1.0) or 1.0)
    row["actor_critic_descendant_log_gain"] = 0.15 + 0.01 * float(idx)
    row["inverse_repair_slate_id"] = f"tuple_slate_{idx}"
    tuple_defs = [
        ((1,), "identity", 0, "inv_steer", 0.28),
        ((1,), "identity", 1, "inv_steer", 0.11),
        ((2,), "full", 0, "repair_option", 0.18),
        ((3,), "affine", 0, "inv_steer", 0.24),
    ]
    best_path, best_local_rank = best_tuple
    tuned_defs = []
    for path, mode, local_rank, action, eff in tuple_defs:
        path_idx = int(path[0])
        if path_idx == int(best_path) and int(local_rank) == int(best_local_rank):
            eff = 0.05
        tuned_defs.append((path, mode, local_rank, action, eff))
    row["selected_path"] = [int(best_path)]
    row["selected_target_mode"] = "identity" if int(best_path) == 1 else ("full" if int(best_path) == 2 else "affine")
    row["inverse_repair_slate"] = []
    for beam_rank, (path, mode, local_rank, action, eff) in enumerate(tuned_defs):
        root_op = "add" if path == (1,) else ("mul" if path == (2,) else "sin")
        exact_observed = not (partial and beam_rank == len(tuned_defs) - 1)
        row["inverse_repair_slate"].append({
            "slate_id": f"tuple_slate_{idx}",
            "tuple_provenance": "beam_local_repair",
            "path": list(path),
            "target_mode": mode,
            "route": "repair",
            "action": action,
            "beam_rank": beam_rank,
            "local_rank": int(local_rank),
            "path_gain": 1.2 - 0.1 * beam_rank,
            "local_probe_mse": eff * 0.9,
            "local_fit_mse": eff * 0.8,
            "candidate_subtree_size": 2 + beam_rank + int(local_rank),
            "candidate_subtree_depth": 1 + int(local_rank),
            "candidate_subtree_size_delta": beam_rank - 1,
            "candidate_subtree_depth_delta": int(local_rank),
            "candidate_child_size": 6 + beam_rank + int(local_rank),
            "candidate_child_depth": 3 + int(local_rank),
            "candidate_child_size_delta": beam_rank - 1,
            "candidate_child_depth_delta": int(local_rank),
            "candidate_root_op": root_op,
            "local_candidate_count": len(tuned_defs),
            "child_key": f"tuple_{idx}_{path[0]}_{local_rank}",
            "child_expr": f"tuple_state_{idx + path[0] + int(local_rank) + 1}",
            "child_eff_mse": eff,
            "dedup_kept": True,
            "exact_child_score_observed": bool(exact_observed),
        })
    return row


def _repair_build_route_row(
    idx: int,
    *,
    repair_better: bool = True,
    small_margin: bool = False,
) -> dict:
    row = _repair_tuple_row(idx, best_tuple=(1, 1), partial=False)
    repair_best_eff = 0.06 if repair_better else 0.22
    build_best_eff = 0.24 if repair_better else 0.05
    if small_margin:
        repair_best_eff = 0.12
        build_best_eff = 0.1205
    for slate_row in row["inverse_repair_slate"]:
        if slate_row["path"] == [1] and int(slate_row["local_rank"]) == 1:
            slate_row["child_eff_mse"] = repair_best_eff
            slate_row["local_probe_mse"] = repair_best_eff * 0.9
            slate_row["local_fit_mse"] = repair_best_eff * 0.8
        else:
            slate_row["child_eff_mse"] = max(float(slate_row["child_eff_mse"]), repair_best_eff + 0.08)
            slate_row["local_probe_mse"] = float(slate_row["child_eff_mse"]) * 0.9
            slate_row["local_fit_mse"] = float(slate_row["child_eff_mse"]) * 0.8
    row["controller_build_slate_id"] = f"build_slate_{idx}"
    row["controller_build_slate"] = [
        {
            "slate_id": f"build_slate_{idx}",
            "slate_rank": 0,
            "tuple_provenance": "build_slate",
            "action": "replace",
            "path": [2],
            "path_source": "critic_path_head",
            "path_length": 1,
            "child_key": f"build_{idx}_replace",
            "child_expr": f"build_expr_{idx}_replace",
            "candidate_root_op": "mul" if repair_better else "var",
            "candidate_child_size": 8.0 if repair_better else 3.0,
            "candidate_child_depth": 4.0 if repair_better else 1.0,
            "candidate_child_size_delta": 2.0 if repair_better else -1.0,
            "candidate_child_depth_delta": 1.0 if repair_better else -1.0,
            "child_raw_mse": build_best_eff * 1.1,
            "child_eff_mse": build_best_eff,
            "exact_child_score_observed": True,
            "accepted": True,
        },
        {
            "slate_id": f"build_slate_{idx}",
            "slate_rank": 1,
            "tuple_provenance": "build_slate",
            "action": "wrap_un",
            "path": [1],
            "path_source": "inverse_gate_best_path",
            "path_length": 1,
            "child_key": f"build_{idx}_wrap",
            "child_expr": f"build_expr_{idx}_wrap",
            "candidate_root_op": "sin",
            "candidate_child_size": 9.0 if repair_better else 4.0,
            "candidate_child_depth": 5.0 if repair_better else 2.0,
            "candidate_child_size_delta": 3.0 if repair_better else 0.0,
            "candidate_child_depth_delta": 2.0 if repair_better else 0.0,
            "child_raw_mse": (build_best_eff + 0.06) * 1.1,
            "child_eff_mse": build_best_eff + 0.06,
            "exact_child_score_observed": True,
            "accepted": True,
        },
        {
            "slate_id": f"build_slate_{idx}",
            "slate_rank": 2,
            "tuple_provenance": "build_slate",
            "action": "residual",
            "path": [1],
            "path_source": "inverse_gate_best_path",
            "path_length": 1,
            "child_key": f"build_{idx}_residual",
            "child_expr": f"build_expr_{idx}_residual",
            "candidate_root_op": "add",
            "candidate_child_size": 10.0 if repair_better else 5.0,
            "candidate_child_depth": 4.0 if repair_better else 2.0,
            "candidate_child_size_delta": 4.0 if repair_better else 1.0,
            "candidate_child_depth_delta": 1.0 if repair_better else 0.0,
            "child_raw_mse": (build_best_eff + 0.10) * 1.1,
            "child_eff_mse": build_best_eff + 0.10,
            "exact_child_score_observed": True,
            "accepted": True,
        },
    ]
    row["controller_build_slate_count"] = 3
    row["controller_build_slate_exact_observed_count"] = 3
    return row


def test_collect_repair_critic_examples_skips_controller_blocked_rows():
    rows = [
        _row(0, good=True),
        _row(1, good=False),
        {
            "parent_expr": "blocked",
            "status": "controller_blocked_low_score",
            "path_entropy": 0.5,
            "path_top_mass": 0.5,
        },
    ]
    examples, reward_scale = collect_repair_critic_examples(rows)
    assert len(examples) == 2
    assert reward_scale > 0.0


def test_collect_repair_critic_examples_skips_unlabeled_typed_records():
    rows = [
        RepairControllerFeatureRecord.from_flat_row(_row(0, good=True)),
        RepairControllerFeatureRecord.from_flat_row(_row(1, good=False)),
    ]
    examples, reward_scale = collect_repair_critic_examples(rows)
    assert examples == []
    assert reward_scale == 1.0


def test_build_actor_critic_rows_preserves_path_rows():
    rows = [_actor_critic_row(0, repair=True), _actor_critic_row(1, repair=False)]
    built = _build_actor_critic_rows(rows)
    assert len(built) == 2
    assert len(built[0]["path_rows"]) == 2
    assert tuple(tuple(int(v) for v in row.path) for row in built[0]["path_rows"]) == ((1,), (2,))
    assert set(tuple(int(v) for v in row.path) for row in built[1]["path_rows"]) == {(1,), (2,)}
    assert tuple(int(v) for v in built[0]["path_rows"][built[0]["path_target_index"]].path) == (1,)
    assert tuple(int(v) for v in built[1]["path_rows"][built[1]["path_target_index"]].path) == (2,)


def test_build_actor_critic_rows_uses_controller_action_path_when_selected_path_missing():
    row = _actor_critic_row(0, repair=False)
    row.pop("selected_path", None)
    row["controller_action_path"] = [2]

    built = _build_actor_critic_rows([row])

    assert len(built) == 1
    assert built[0]["selected_path"] == (2,)
    assert tuple(int(v) for v in built[0]["path_rows"][built[0]["path_target_index"]].path) == (2,)


def test_build_actor_critic_rows_prefers_descendant_reward_when_available():
    row = _actor_critic_row(0, repair=True)
    row["actor_critic_descendant_reward"] = 4.25

    built_default = _build_actor_critic_rows([row])
    built_immediate = _build_actor_critic_rows([row], reward_target="immediate")

    assert len(built_default) == 1
    assert built_default[0]["actor_critic_reward"] == 4.25
    assert built_default[0]["actor_critic_reward_source"] == "descendant"
    assert built_immediate[0]["actor_critic_reward"] == row["actor_critic_reward"]
    assert built_immediate[0]["actor_critic_reward_source"] == "immediate"


def test_build_repair_slate_rows_supports_full_and_partial_slates():
    rows = [
        _repair_slate_row(0, best_path=1, partial=False),
        _repair_slate_row(1, best_path=2, partial=True),
    ]

    built, reward_source_counts = _build_repair_slate_rows(rows, score_gap_floor=0.05)

    assert len(built) == 2
    assert reward_source_counts == {"descendant": 2}
    full_row = next(row for row in built if row["full_slate"])
    partial_row = next(row for row in built if not row["full_slate"])
    assert tuple(int(v) for v in full_row["path_rows"][full_row["path_target_index"]].path) == (1,)
    assert full_row["n_observed_paths"] == 3
    assert tuple(int(v) for v in partial_row["path_rows"][partial_row["path_target_index"]].path) == (2,)
    assert partial_row["n_observed_paths"] == 2
    assert partial_row["pairwise_pairs"]
    assert sum(1 for flag in partial_row["path_utility_mask"] if flag) == 2


def test_build_repair_slate_rows_prefers_logged_beam_candidates():
    row = _repair_slate_row(3, best_path=1, partial=False)
    row["path_summaries"] = [
        {
            "path": [1, 1],
            "target_mode": "identity",
            "weighted_rel_gain": 0.8,
            "weighted_rel_gain_raw": 0.75,
            "weighted_rel_gain_pre_cut": 0.85,
            "rel_gain": 0.7,
            "valid_frac": 0.9,
            "confidence": 0.85,
            "static_score": 1.1,
            "transport_rel": 0.6,
        }
    ]
    row["repair_option_final_inverse_path_mode_beam"] = [
        {
            "path": [1],
            "target_mode": "identity",
            "path_gain": 3.0,
            "path_gain_pre_cut": 3.5,
            "rel_gain": 0.8,
            "transport_rel": 0.4,
            "branch_factor": 1.0,
            "path_cut_factor": 1.0,
            "effective_n": 16.0,
            "best_alt_mse": 0.2,
            "min_valid_frac_eff": 0.8,
            "min_confidence_eff": 0.7,
        },
        {
            "path": [2],
            "target_mode": "full",
            "path_gain": 2.0,
            "path_gain_pre_cut": 2.2,
            "rel_gain": 0.6,
            "transport_rel": 0.3,
            "branch_factor": 1.0,
            "path_cut_factor": 1.0,
            "effective_n": 16.0,
            "best_alt_mse": 0.3,
            "min_valid_frac_eff": 0.8,
            "min_confidence_eff": 0.7,
        },
    ]
    row["repair_option_final_inverse_repair_slate"] = [
        {
            "path": [1],
            "target_mode": "identity",
            "action": "inv_steer",
            "dedup_kept": True,
            "exact_child_score_observed": True,
            "child_eff_mse": 0.1,
        },
        {
            "path": [2],
            "target_mode": "full",
            "action": "inv_steer",
            "dedup_kept": True,
            "exact_child_score_observed": True,
            "child_eff_mse": 0.2,
        },
    ]

    built, _ = _build_repair_slate_rows([row], score_gap_floor=0.01)

    assert len(built) == 1
    assert built[0]["beam_key"] == "repair_option_final_inverse_path_mode_beam"
    assert tuple(int(v) for v in built[0]["path_rows"][built[0]["path_target_index"]].path) == (1,)


def test_build_repair_tuple_rows_preserves_multiple_candidates_on_same_path():
    rows = [
        _repair_tuple_row(0, best_tuple=(1, 1), partial=False),
        _repair_tuple_row(1, best_tuple=(2, 0), partial=True),
    ]

    built, reward_source_counts = _build_repair_tuple_rows(rows, score_gap_floor=0.02)

    assert len(built) == 2
    assert reward_source_counts == {"descendant": 2}
    first = built[0]
    assert len(first["preview_rows"]) == 4
    assert first["n_total_tuples"] == 4
    assert first["n_observed_tuples"] == 4
    assert len(first["preview_value_mask"]) == 4
    assert len(first["preview_regret_targets"]) == 4
    assert min(first["preview_regret_targets"]) == 0.0
    assert tuple(int(v) for v in first["path_rows"][first["path_target_index"]].path) == (1,)
    same_path_count = sum(1 for idx in first["preview_path_indices"] if int(idx) == 0)
    assert same_path_count == 2
    partial = next(row for row in built if not row["full_slate"])
    assert partial["pairwise_pairs"]
    assert partial["n_total_tuples"] == 4
    assert partial["n_observed_tuples"] == 3


def test_build_repair_tuple_rows_groups_duplicate_child_provenances():
    row = _repair_tuple_row(7, best_tuple=(1, 1), partial=False)
    dup = dict(row["inverse_repair_slate"][0])
    dup["path"] = [2]
    dup["target_mode"] = "full"
    dup["target_mapping_kind"] = "poly"
    dup["local_mapping_kind"] = "affine"
    dup["child_key"] = row["inverse_repair_slate"][1]["child_key"]
    dup["child_expr"] = row["inverse_repair_slate"][1]["child_expr"]
    dup["child_eff_mse"] = row["inverse_repair_slate"][1]["child_eff_mse"]
    row["inverse_repair_slate"].append(dup)

    built, _ = _build_repair_tuple_rows([row], score_gap_floor=0.02)

    assert len(built) == 1
    first = built[0]
    assert first["n_total_tuples"] == 4
    grouped = next(pr for pr in first["preview_rows"] if pr["child_key"] == dup["child_key"])
    assert grouped["provenance_count"] == 2
    assert grouped["distinct_path_count"] == 2
    assert grouped["distinct_mode_count"] == 2
    assert grouped["distinct_local_mapping_count"] >= 1


def test_build_repair_build_route_rows_extracts_same_parent_route_examples():
    rows = [
        _repair_build_route_row(0, repair_better=True),
        _repair_build_route_row(1, repair_better=False),
        _repair_build_route_row(2, repair_better=True, small_margin=True),
    ]

    built = _build_repair_build_route_rows(rows, margin_floor=0.01)

    assert len(built) == 3
    assert built[0]["route_label"] == "repair"
    assert built[1]["route_label"] == "build"
    assert built[2]["route_label"] == "tie"
    assert built[0]["repair_summary"]["best_utility"] > built[0]["build_summary"]["best_utility"]
    assert built[1]["repair_summary"]["best_utility"] < built[1]["build_summary"]["best_utility"]
    assert "repair_best_probe_mse_log" in built[0]["features"]
    assert "build_best_child_size_log" in built[0]["features"]
    assert "delta_best_utility" not in built[0]["features"]


def test_build_build_tuple_rows_extract_same_parent_build_slates():
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 5 == 0))
        for i in range(12)
    ]
    built = _build_build_tuple_rows(rows, score_gap_floor=0.01)

    assert len(built) == 12
    first = built[0]
    assert len(first["preview_rows"]) == 3
    assert first["n_total_tuples"] == 3
    assert first["n_observed_tuples"] == 3
    assert len(first["preview_utility_targets"]) == 3
    assert first["state_value_target"] == max(first["preview_utility_targets"])
    assert first["full_slate"] is True


def test_build_build_tuple_rows_groups_provenance_for_same_child():
    row = _repair_build_route_row(0, repair_better=False)
    row["controller_build_slate"].append({
        "slate_id": "build_slate_dup",
        "slate_rank": 3,
        "tuple_provenance": "build_slate",
        "action": "wrap_un",
        "path": [3],
        "path_source": "random",
        "path_length": 2,
        "child_key": row["controller_build_slate"][0]["child_key"],
        "child_expr": row["controller_build_slate"][0]["child_expr"],
        "candidate_root_op": "var",
        "candidate_child_size": 4.0,
        "candidate_child_depth": 2.0,
        "candidate_child_size_delta": 0.0,
        "candidate_child_depth_delta": 0.0,
        "child_raw_mse": row["controller_build_slate"][0]["child_raw_mse"],
        "child_eff_mse": row["controller_build_slate"][0]["child_eff_mse"],
        "exact_child_score_observed": False,
        "accepted": False,
    })
    row["controller_build_slate_count"] = 4

    built = _build_build_tuple_rows([row], score_gap_floor=0.01)

    assert len(built) == 1
    preview_rows = built[0]["preview_rows"]
    assert len(preview_rows) == 3
    grouped = next(item for item in preview_rows if item["child_key"] == row["controller_build_slate"][0]["child_key"])
    assert grouped["provenance_count"] == 2
    assert grouped["distinct_action_count"] == 2
    assert grouped["distinct_path_count"] == 2
    assert grouped["action_replace_frac"] == 0.5
    assert grouped["action_wrap_un_frac"] == 0.5
    assert grouped["path_source_critic_frac"] == 0.5
    assert grouped["path_source_random_frac"] == 0.5
    feats = build_preview_feature_vector(grouped)
    assert feats  # smoke: grouped rows map into the expanded build feature vector


def test_build_unified_candidate_rows_extracts_mixed_same_parent_slates():
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 5 == 0))
        for i in range(12)
    ]

    built, reward_source_counts = _build_unified_candidate_rows(rows, score_gap_floor=0.01)

    assert len(built) == 12
    assert reward_source_counts
    first = built[0]
    assert first["n_total_tuples"] >= 4
    assert first["n_repair_candidates"] >= 1
    assert first["n_build_candidates"] >= 1
    assert "repair" in first["preview_route_targets"]
    assert "build" in first["preview_route_targets"]
    assert first["route_target"] in {"repair", "build"}


def test_train_unified_candidate_ranker(tmp_path):
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 6 == 0))
        for i in range(24)
    ]

    bundle = train_unified_candidate_ranker(
        rows,
        hidden_dim=16,
        epochs=80,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=83,
        score_gap_floor=0.01,
        route_aux_weight=0.0,
    )

    assert bundle["unified_candidate_ranker_trained"] is True
    assert bundle["unified_candidate_ranker_target"] == "same_parent_best_exact_candidate"
    assert int(bundle["metrics"]["n_examples"]) == 24
    assert int(bundle["metrics"]["n_tuple_examples"]) > 24
    assert "candidate_top1_acc" in bundle["metrics"]["val"]
    assert "candidate_pairwise_acc" in bundle["metrics"]["val"]
    assert "candidate_lse_route_acc" in bundle["metrics"]["val"]

    out_path = tmp_path / "unified_candidate_ranker.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)

    repair_pred = predict_unified_candidate_slate(loaded, _repair_build_route_row(100, repair_better=True))
    build_pred = predict_unified_candidate_slate(loaded, _repair_build_route_row(101, repair_better=False))
    assert repair_pred["trained"] is True
    assert build_pred["trained"] is True
    assert repair_pred["best_route"] == "repair"
    assert build_pred["best_route"] == "build"
    assert repair_pred["rows"]
    assert "allocation_estimate" in repair_pred["rows"][0]
    assert repair_pred["route_scores"]


def test_run_unified_candidate_training_cli_helper(tmp_path):
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 6 == 0))
        for i in range(24)
    ]
    report_path = tmp_path / "unified_candidate_report.json"
    report_path.write_text(json.dumps({"inverse_experiment_log": rows}, indent=2), encoding="utf-8")
    out_path = tmp_path / "unified_candidate_bundle.pt"
    summary = run_unified_candidate_training(
        report_paths=[str(report_path)],
        output_path=str(out_path),
        hidden_dim=16,
        epochs=60,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=89,
        score_gap_floor=0.01,
    )
    assert out_path.exists()
    assert (tmp_path / "unified_candidate_bundle.pt.json").exists()
    assert int(summary["metrics"]["n_examples"]) == 24
    assert summary["unified_candidate_ranker_target"] == "same_parent_best_exact_candidate"


def test_train_shared_candidate_dual_ranker(tmp_path):
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 6 == 0))
        for i in range(24)
    ]

    bundle = train_shared_candidate_dual_ranker(
        rows,
        hidden_dim=16,
        epochs=80,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=97,
        score_gap_floor=0.01,
    )

    assert bundle["shared_candidate_dual_trained"] is True
    assert bundle["shared_candidate_dual_target"] == "within_route_best_exact_candidate"
    assert bundle["shared_candidate_common_value_target"].startswith("immediate_refine_")
    assert int(bundle["metrics"]["n_examples"]) == 24
    assert "repair_candidate_top1_acc" in bundle["metrics"]["val"]
    assert "build_candidate_top1_acc" in bundle["metrics"]["val"]
    assert "common_candidate_top1_acc" in bundle["metrics"]["val"]
    assert "common_preview_q_mae" in bundle["metrics"]["val"]

    out_path = tmp_path / "shared_candidate_dual.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)

    repair_pred = predict_shared_candidate_dual_slate(loaded, _repair_build_route_row(100, repair_better=True))
    build_pred = predict_shared_candidate_dual_slate(loaded, _repair_build_route_row(101, repair_better=False))
    assert repair_pred["trained"] is True
    assert build_pred["trained"] is True
    assert repair_pred["repair"]["best_child_key"] is not None
    assert build_pred["build"]["best_child_key"] is not None
    assert repair_pred["rows"]
    assert "route_score_estimate" in repair_pred["rows"][0]
    assert "common_q_estimate" in repair_pred["rows"][0]
    assert repair_pred["common"]["best_q"] is not None


def test_train_shared_candidate_dual_ranker_with_oracle_aux():
    rows = []
    for i in range(24):
        row = _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 6 == 0))
        row["oracle_truth_in_slate"] = True
        row["oracle_truth_path_index"] = 0
        for path_idx, path_row in enumerate(row["path_summaries"]):
            path_row["oracle_relation_to_reference"] = "same" if path_idx == 0 else "disjoint"
            path_row["oracle_is_reference_path"] = bool(path_idx == 0)
            path_row["oracle_best_mode"] = "identity" if path_idx == 0 else "full"
            path_row["oracle_truth_present_under_path"] = bool(path_idx == 0)
            path_row["oracle_best_truth_rank"] = 0 if path_idx == 0 else -1
            path_row["oracle_second_truth_rank"] = 1 if path_idx == 0 else -1
            path_row["oracle_truth_rank_margin"] = 1.0 if path_idx == 0 else 0.0
        for cand in row["inverse_repair_slate"]:
            is_truth = (cand["path"] == [1] and cand["target_mode"] == "identity" and int(cand["local_rank"]) == 1)
            cand["oracle_is_truth_candidate"] = bool(is_truth)
            cand["oracle_mode_is_best"] = bool(cand["target_mode"] == "identity")
            cand["oracle_truth_rank"] = 0 if cand["path"] == [1] else None
            cand["oracle_truth_rank_score"] = 1.0 if cand["path"] == [1] else None
            cand["oracle_mapping_stable"] = bool(cand["target_mode"] == "identity")
        rows.append(row)

    bundle = train_shared_candidate_dual_ranker(
        rows,
        hidden_dim=16,
        epochs=60,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=103,
        score_gap_floor=0.01,
    )

    metrics = bundle["metrics"]
    assert int(metrics["n_oracle_path_examples"]) > 0
    assert int(metrics["n_oracle_relation_examples"]) > 0
    assert int(metrics["n_oracle_mode_examples"]) > 0
    assert int(metrics["n_oracle_truth_examples"]) > 0
    assert int(metrics["n_oracle_rank_examples"]) > 0
    assert int(metrics["n_oracle_coverage_examples"]) > 0
    assert "oracle_preview_truth_acc" in metrics["val"]
    assert "oracle_preview_rank_mae" in metrics["val"]
    assert "oracle_state_coverage_acc" in metrics["val"]


def test_shared_candidate_record_masks_and_feature_flow():
    repair_row = shared_candidate_row_dict(
        {
            "route_source": "repair",
            "action": "inv_steer",
            "child_key": "repair_child",
            "child_expr": "(add x y)",
            "path": [0, 1],
            "path_source": "critic_path_head",
            "target_mode": "affine",
            "provenance_grouped": True,
            "provenance_count": 3,
            "provenance_rows": [{"child_key": "repair_child"}],
            "exact_child_score_observed": False,
        }
    )
    build_row = shared_candidate_row_dict(
        {
            "route_source": "build",
            "action": "replace",
            "child_key": "build_child",
            "child_expr": "(mul x y)",
            "path": [],
            "exact_child_score_observed": True,
        }
    )

    repair_record = coerce_shared_candidate_record(repair_row)
    build_record = coerce_shared_candidate_record(build_row)
    assert repair_record.route_source == "repair"
    assert build_record.route_source == "build"
    assert repair_row["candidate_has_path"] == 1.0
    assert repair_row["candidate_has_target_mode"] == 1.0
    assert repair_row["candidate_evidence_preview_support"] == 1.0
    assert build_row["candidate_has_path"] == 0.0
    assert build_row["candidate_evidence_exact_known"] == 1.0

    unified = _build_unified_candidate_rows([_repair_build_route_row(0, repair_better=True)], score_gap_floor=0.0)[0]
    assert unified
    preview_row = unified[0]["preview_rows"][0]
    for name in SHARED_CANDIDATE_MASK_FIELD_NAMES:
        assert name in preview_row


def test_run_shared_candidate_training_cli_helper(tmp_path):
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 6 == 0))
        for i in range(24)
    ]
    report_path = tmp_path / "shared_candidate_report.json"
    report_path.write_text(json.dumps({"inverse_experiment_log": rows}, indent=2), encoding="utf-8")
    out_path = tmp_path / "shared_candidate_bundle.pt"
    summary = run_shared_candidate_training(
        report_paths=[str(report_path)],
        output_path=str(out_path),
        hidden_dim=16,
        epochs=60,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=101,
        score_gap_floor=0.01,
    )
    assert out_path.exists()
    assert (tmp_path / "shared_candidate_bundle.pt.json").exists()
    assert int(summary["metrics"]["n_examples"]) == 24
    assert summary["shared_candidate_dual_target"] == "within_route_best_exact_candidate"
    assert summary["sample_prediction"]["common"]["best_q"] is not None


def test_train_repair_build_route_comparator(tmp_path):
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 7 == 0))
        for i in range(24)
    ]

    bundle = train_repair_build_route_comparator(
        rows,
        hidden_dim=16,
        epochs=80,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=47,
        margin_floor=0.01,
    )

    assert bundle["repair_build_route_compare_trained"] is True
    assert bundle["repair_build_route_compare_target"] == "same_parent_best_exact_margin"
    assert int(bundle["metrics"]["n_examples"]) == 24
    assert "route_acc" in bundle["metrics"]["val"]
    assert "route_margin_mae" in bundle["metrics"]["val"]

    out_path = tmp_path / "repair_route_compare.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)

    repair_pred = predict_repair_build_route(loaded, _repair_build_route_row(100, repair_better=True))
    build_pred = predict_repair_build_route(loaded, _repair_build_route_row(101, repair_better=False))
    assert repair_pred["trained"] is True
    assert repair_pred["best_route"] == "repair"
    assert build_pred["best_route"] == "build"
    assert repair_pred["repair_prob"] > build_pred["repair_prob"]


def test_train_repair_build_route_comparator_with_learned_tuple_features(tmp_path):
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 7 == 0))
        for i in range(24)
    ]

    repair_bundle = train_repair_controller_tuple_ranker(
        rows,
        hidden_dim=12,
        epochs=40,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=71,
        score_gap_floor=0.01,
    )
    build_bundle = train_build_tuple_ranker(
        rows,
        hidden_dim=12,
        epochs=40,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=73,
        score_gap_floor=0.01,
    )
    bundle = train_repair_build_route_comparator(
        rows,
        hidden_dim=16,
        epochs=60,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=79,
        margin_floor=0.01,
        repair_tuple_bundle=repair_bundle,
        build_tuple_bundle=build_bundle,
    )

    assert bundle["repair_build_route_compare_uses_repair_tuple_features"] is True
    assert bundle["repair_build_route_compare_uses_build_tuple_features"] is True
    out_path = tmp_path / "repair_route_compare_learned.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)
    pred = predict_repair_build_route(
        loaded,
        _repair_build_route_row(100, repair_better=True),
        repair_tuple_bundle=repair_bundle,
        build_tuple_bundle=build_bundle,
    )
    assert pred["trained"] is True
    assert pred["best_route"] == "repair"
    built = _build_repair_build_route_rows(
        [_repair_build_route_row(101, repair_better=True)],
        margin_floor=0.0,
        repair_tuple_bundle=repair_bundle,
        build_tuple_bundle=build_bundle,
    )
    assert built
    feats = built[0]["features"]
    assert "repair_learned_best_score" in feats
    assert "build_learned_best_score" in feats


def test_train_build_tuple_ranker(tmp_path):
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 6 == 0))
        for i in range(24)
    ]

    bundle = train_build_tuple_ranker(
        rows,
        hidden_dim=16,
        epochs=80,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=59,
        score_gap_floor=0.01,
    )

    assert bundle["build_tuple_ranker_trained"] is True
    assert bundle["build_tuple_ranker_target"] == "same_parent_best_exact_utility"
    assert int(bundle["metrics"]["n_examples"]) == 24
    assert int(bundle["metrics"]["n_full_slate_examples"]) == 24
    assert int(bundle["metrics"]["n_tuple_examples"]) == 72
    assert "build_tuple_top1_acc" in bundle["metrics"]["val"]
    assert "build_tuple_pairwise_acc" in bundle["metrics"]["val"]
    assert "state_value_mae" in bundle["metrics"]["val"]

    out_path = tmp_path / "build_tuple_ranker.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)

    pred = predict_build_tuple_slate(loaded, _repair_build_route_row(100, repair_better=False))
    assert pred["trained"] is True
    assert pred["best_action"] == "replace"
    assert pred["best_child_key"]
    assert len(pred["rows"]) == 3
    assert "utility_estimate" in pred["rows"][0]


def test_run_repair_route_training_cli_helper(tmp_path):
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 6 == 0))
        for i in range(24)
    ]
    report_path = tmp_path / "repair_route_report.json"
    report_path.write_text(json.dumps({"inverse_experiment_log": rows}, indent=2), encoding="utf-8")
    out_path = tmp_path / "repair_route_bundle.pt"
    summary = run_repair_route_training(
        report_paths=[str(report_path)],
        output_path=str(out_path),
        hidden_dim=16,
        epochs=60,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=53,
        margin_floor=0.01,
    )
    assert out_path.exists()
    assert (tmp_path / "repair_route_bundle.pt.json").exists()
    assert int(summary["metrics"]["n_examples"]) == 24
    assert summary["repair_build_route_compare_target"] == "same_parent_best_exact_margin"


def test_run_build_tuple_training_cli_helper(tmp_path):
    rows = [
        _repair_build_route_row(i, repair_better=(i < 12), small_margin=(i % 6 == 0))
        for i in range(24)
    ]
    report_path = tmp_path / "build_tuple_report.json"
    report_path.write_text(json.dumps({"inverse_experiment_log": rows}, indent=2), encoding="utf-8")
    out_path = tmp_path / "build_tuple_bundle.pt"
    summary = run_build_tuple_training(
        report_paths=[str(report_path)],
        output_path=str(out_path),
        hidden_dim=16,
        epochs=60,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=61,
        score_gap_floor=0.01,
    )
    assert out_path.exists()
    assert (tmp_path / "build_tuple_bundle.pt.json").exists()
    assert int(summary["metrics"]["n_examples"]) == 24
    assert summary["build_tuple_ranker_target"] == "same_parent_best_exact_utility"


def test_train_repair_controller_tuple_ranker(tmp_path):
    rows = [
        _repair_tuple_row(i, best_tuple=((1, 1) if i < 12 else (2, 0)), partial=(i % 2 == 0))
        for i in range(24)
    ]
    init_bundle = pretrain_repair_controller_from_oracle_tasks(
        [_oracle_task(i, repair=True) for i in range(12)],
        hidden_dim=16,
        epochs=60,
        lr=2.0e-2,
        val_fraction=0.25,
        seed=31,
    )
    bundle = train_repair_controller_tuple_ranker(
        rows,
        hidden_dim=16,
        epochs=80,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=37,
        init_bundle=init_bundle,
        score_gap_floor=0.02,
    )

    assert bundle["repair_tuple_ranker_trained"] is True
    assert bundle["repair_tuple_ranker_target"] == "same_state_exact_child_log_gain"
    assert bundle["repair_tuple_preview_value_target"] == "child_descendant_log_gain"
    assert bundle["repair_tuple_regret_target"] == "same_state_best_exact_regret"
    assert float(bundle["repair_tuple_child_value_lambda"]) > 0.0
    assert float(bundle["repair_tuple_regret_weight"]) > 0.0
    assert len(bundle["provenance_feature_names"]) > 0
    assert bundle["path_head_trained"] is True
    assert bundle["value_head_trained"] is True
    assert bundle["regret_head_trained"] is True
    assert bundle["path_action_head_trained"] is False
    assert int(bundle["metrics"]["n_examples"]) == 24
    assert int(bundle["metrics"]["n_preview_examples"]) == 24
    assert int(bundle["metrics"]["n_tuple_examples"]) > 24
    assert int(bundle["metrics"]["n_provenance_tuples"]) >= int(bundle["metrics"]["n_tuple_examples"])
    assert int(bundle["metrics"]["n_preview_value_tuples"]) > 0
    assert int(bundle["metrics"]["n_preview_regret_tuples"]) > 0
    assert "tuple_top1_acc" in bundle["metrics"]["val"]
    assert "tuple_pairwise_acc" in bundle["metrics"]["val"]
    assert "preview_value_mae" in bundle["metrics"]["val"]
    assert "preview_regret_mae" in bundle["metrics"]["val"]

    out_path = tmp_path / "repair_tuple_ranker.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)
    assert loaded["repair_tuple_ranker_trained"] is True
    assert len(loaded["preview_feature_names"]) > 0
    assert len(loaded["provenance_feature_names"]) > 0

    tuple_pred = predict_repair_tuple_slate(
        loaded,
        RepairControllerFeatureRecord.from_flat_row(_repair_tuple_row(100, best_tuple=(1, 1), partial=False)),
        preview_rows=_repair_tuple_row(100, best_tuple=(1, 1), partial=False)["inverse_repair_slate"],
    )
    assert tuple_pred["trained"] is True
    assert tuple_pred["best_path"] == [1]
    assert tuple_pred["best_action"] == "inv_steer"
    assert float(tuple_pred["child_value_lambda"]) > 0.0
    assert float(tuple_pred["regret_weight"]) > 0.0
    assert tuple_pred["best_child_key"]
    assert len(tuple_pred["rows"]) >= 3
    assert "combined_estimate" in tuple_pred["rows"][0]
    assert "regret_estimate" in tuple_pred["rows"][0]
    assert "allocation_estimate" in tuple_pred["rows"][0]
    assert "provenance_weights" in tuple_pred["rows"][0]


def test_run_repair_tuple_training_cli_helper(tmp_path):
    rows = [
        _repair_tuple_row(i, best_tuple=((1, 1) if i < 12 else (2, 0)), partial=(i % 2 == 1))
        for i in range(24)
    ]
    report_path = tmp_path / "repair_tuple_report.json"
    report_path.write_text(json.dumps({"inverse_experiment_log": rows}, indent=2), encoding="utf-8")
    out_path = tmp_path / "repair_tuple_bundle.pt"
    summary = run_repair_tuple_training(
        report_paths=[str(report_path)],
        output_path=str(out_path),
        hidden_dim=16,
        epochs=60,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=41,
        score_gap_floor=0.02,
    )
    assert out_path.exists()
    assert (tmp_path / "repair_tuple_bundle.pt.json").exists()
    assert int(summary["metrics"]["n_examples"]) == 24
    assert summary["repair_tuple_ranker_target"] == "same_state_exact_child_log_gain"
    assert summary["repair_tuple_preview_value_target"] == "child_descendant_log_gain"
    assert summary["repair_tuple_regret_target"] == "same_state_best_exact_regret"


def test_train_save_load_and_predict_repair_critic(tmp_path):
    rows = [_row(i, good=(i < 8)) for i in range(16)]
    bundle = train_repair_critic(
        rows,
        hidden_dim=16,
        epochs=80,
        lr=2.0e-2,
        val_fraction=0.25,
        seed=3,
    )
    assert int(bundle["metrics"]["n_examples"]) == 16
    assert bundle["model_kind"] == "shared_encoder_bundle_v1"

    out_path = tmp_path / "repair_critic.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)

    good_pred = predict_repair_critic(loaded, RepairControllerFeatureRecord.from_flat_row(_row(100, good=True)))
    bad_pred = predict_repair_critic(loaded, RepairControllerFeatureRecord.from_flat_row(_row(101, good=False)))

    assert good_pred["utility_score"] > bad_pred["utility_score"]
    assert good_pred["positive_reward_prob"] > bad_pred["positive_reward_prob"]
    assert good_pred["accept_prob"] > bad_pred["accept_prob"]


def test_train_save_load_and_predict_shared_macro_and_path_heads(tmp_path):
    rows = [_policy_row(i, repair=(i < 12)) for i in range(24)]
    bundle = train_repair_critic(
        rows,
        hidden_dim=24,
        epochs=120,
        lr=2.0e-2,
        val_fraction=0.25,
        seed=5,
    )
    assert int(bundle["metrics"]["n_macro_examples"]) == 24
    assert int(bundle["metrics"]["n_path_examples"]) == 24
    assert bundle["macro_head_trained"] is True
    assert bundle["path_head_trained"] is True

    out_path = tmp_path / "repair_controller_bundle.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)

    repair_heads = predict_repair_controller_heads(loaded, RepairControllerFeatureRecord.from_flat_row(_policy_row(100, repair=True)))
    build_heads = predict_repair_controller_heads(loaded, RepairControllerFeatureRecord.from_flat_row(_policy_row(101, repair=False)))

    assert repair_heads["macro_action"]["trained"] is True
    assert repair_heads["macro_action"]["best_action"] == "repair_option"
    assert build_heads["macro_action"]["best_action"] == "replace"
    assert repair_heads["route"]["trained"] is True
    assert repair_heads["route"]["best_route"] == "repair"
    assert build_heads["route"]["best_route"] == "build"
    assert repair_heads["path"]["trained"] is True
    assert repair_heads["path_action"]["trained"] is True
    assert repair_heads["path_action"]["best_path"] == [1]
    assert repair_heads["path_action"]["best_action"] == "repair_option"
    assert repair_heads["path"]["best_path"] == [1]
    assert repair_heads["path"]["best_target_mode"] == "identity"
    assert build_heads["path_action"]["best_path"] == [2]
    assert build_heads["path_action"]["best_action"] == "replace"
    assert build_heads["path"]["best_path"] == [2]
    assert build_heads["path"]["best_target_mode"] == "full"


def test_train_repair_controller_actor_critic(tmp_path):
    rows = [_actor_critic_row(i, repair=(i < 12)) for i in range(24)]
    for idx, row in enumerate(rows):
        immediate = float(row["actor_critic_reward"])
        row["actor_critic_descendant_reward"] = immediate + (0.35 if idx < 12 else -0.25)
    init_bundle = train_repair_critic(
        rows,
        hidden_dim=24,
        epochs=80,
        lr=2.0e-2,
        val_fraction=0.25,
        seed=11,
    )
    bundle = train_repair_controller_actor_critic(
        rows,
        hidden_dim=24,
        epochs=120,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=13,
        entropy_weight=0.0,
        policy_ce_weight=5.0,
        init_bundle=init_bundle,
    )
    assert bundle["macro_head_trained"] is True
    assert bundle["route_head_trained"] is True
    assert bundle["q_head_trained"] is True
    assert bundle["value_head_trained"] is True
    assert bundle["path_head_trained"] is True
    assert bundle["path_action_head_trained"] is True
    assert bundle["actor_critic_reward_target"] == "descendant_preferred"
    assert "macro_action_hier_acc" in bundle["metrics"]["val"]
    assert int(bundle["metrics"]["n_path_examples"]) == 24
    assert "path_head_acc" in bundle["metrics"]["val"]
    assert "tuple_route_acc" in bundle["metrics"]["val"]
    assert "tuple_q_value_mae" in bundle["metrics"]["val"]
    assert bundle["metrics"]["reward_source_counts"] == {"descendant": 24}

    out_path = tmp_path / "repair_controller_actor_critic.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)

    repair_heads = predict_repair_controller_heads(loaded, RepairControllerFeatureRecord.from_flat_row(_actor_critic_row(100, repair=True)))
    build_heads = predict_repair_controller_heads(loaded, RepairControllerFeatureRecord.from_flat_row(_actor_critic_row(101, repair=False)))

    assert repair_heads["macro_action"]["best_action"] == "repair_option"
    assert repair_heads["route"]["best_route"] == "repair"
    assert build_heads["route"]["best_route"] == "build"
    assert repair_heads["action_value"]["trained"] is True
    assert repair_heads["path_action"]["trained"] is True
    assert repair_heads["path_action"]["best_path"] == [1]
    assert repair_heads["path_action"]["best_action"] == "repair_option"
    assert build_heads["path_action"]["best_path"] == [2]
    assert "repair_option" in repair_heads["action_value"]["estimates"]
    assert "replace" in build_heads["action_value"]["estimates"]
    assert repair_heads["action_value"]["estimates"]["repair_option"] > build_heads["action_value"]["estimates"]["replace"]
    assert repair_heads["value"]["trained"] is True
    assert repair_heads["value"]["estimate"] > build_heads["value"]["estimate"]


def test_train_repair_controller_slate_ranker(tmp_path):
    rows = [
        _repair_slate_row(i, best_path=(1 if i < 12 else 2), partial=(i % 2 == 0))
        for i in range(24)
    ]
    init_bundle = pretrain_repair_controller_from_oracle_tasks(
        [_oracle_task(i, repair=True) for i in range(12)],
        hidden_dim=16,
        epochs=60,
        lr=2.0e-2,
        val_fraction=0.25,
        seed=19,
    )
    bundle = train_repair_controller_slate_ranker(
        rows,
        hidden_dim=16,
        epochs=90,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=23,
        init_bundle=init_bundle,
        score_gap_floor=0.05,
    )

    assert bundle["repair_slate_ranker_trained"] is True
    assert bundle["repair_slate_ranker_target"] == "same_state_exact_child_log_gain"
    assert bundle["path_head_trained"] is True
    assert bundle["path_action_head_trained"] is True
    assert bundle["route_head_trained"] is True
    assert bundle["value_head_trained"] is True
    assert int(bundle["metrics"]["n_examples"]) == 24
    assert int(bundle["metrics"]["n_full_slate_examples"]) == 12
    assert int(bundle["metrics"]["n_pairwise_examples"]) == 12
    assert bundle["metrics"]["reward_source_counts"] == {"descendant": 24}
    assert "slate_top1_acc" in bundle["metrics"]["val"]
    assert "slate_pairwise_acc" in bundle["metrics"]["val"]

    out_path = tmp_path / "repair_slate_ranker.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)

    pred = predict_repair_controller_heads(
        loaded,
        RepairControllerFeatureRecord.from_flat_row(_repair_slate_row(100, best_path=1, partial=False)),
    )
    assert pred["path_action"]["trained"] is True
    assert pred["path_action"]["best_path"] == [1]
    assert pred["path_action"]["best_action"] == "inv_steer"
    assert pred["path_action"]["best_route"] == "repair"
    assert pred["path"]["best_path"] == [1]
    assert pred["action_value"]["trained"] is True
    assert "inv_steer" in pred["action_value"]["estimates"]


def test_run_repair_slate_training_cli_helper(tmp_path):
    rows = [
        _repair_slate_row(i, best_path=(1 if i < 12 else 2), partial=(i % 2 == 1))
        for i in range(24)
    ]
    report_path = tmp_path / "repair_slate_report.json"
    report_path.write_text(json.dumps({"inverse_experiment_log": rows}, indent=2), encoding="utf-8")
    out_path = tmp_path / "repair_slate_bundle.pt"
    summary = run_repair_slate_training(
        report_paths=[str(report_path)],
        output_path=str(out_path),
        hidden_dim=16,
        epochs=60,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=29,
        score_gap_floor=0.05,
    )
    assert out_path.exists()
    assert (tmp_path / "repair_slate_bundle.pt.json").exists()
    assert int(summary["metrics"]["n_examples"]) == 24
    assert summary["repair_slate_ranker_target"] == "same_state_exact_child_log_gain"
    assert summary["metrics"]["reward_source_counts"] == {"descendant": 24}


def test_run_actor_critic_training_cli_helper(tmp_path):
    rows = [_actor_critic_row(i, repair=(i < 12)) for i in range(24)]
    for idx, row in enumerate(rows):
        immediate = float(row["actor_critic_reward"])
        row["actor_critic_descendant_reward"] = immediate + (0.2 if idx < 12 else -0.15)
    report_path = tmp_path / "actor_critic_report.json"
    report_path.write_text(json.dumps({"inverse_experiment_log": rows}, indent=2), encoding="utf-8")
    out_path = tmp_path / "actor_critic_bundle.pt"
    summary = run_actor_critic_training(
        report_paths=[str(report_path)],
        output_path=str(out_path),
        hidden_dim=16,
        epochs=60,
        lr=1.0e-2,
        val_fraction=0.25,
        seed=17,
    )
    assert out_path.exists()
    assert (tmp_path / "actor_critic_bundle.pt.json").exists()
    assert int(summary["metrics"]["n_examples"]) == 24
    assert summary["actor_critic_reward_target"] == "descendant_preferred"
    assert summary["metrics"]["reward_source_counts"] == {"descendant": 24}


def test_pretrain_repair_controller_from_oracle_tasks(tmp_path):
    tasks = [_oracle_task(i, repair=(i < 10)) for i in range(20)]
    bundle = pretrain_repair_controller_from_oracle_tasks(
        tasks,
        hidden_dim=24,
        epochs=120,
        lr=2.0e-2,
        val_fraction=0.25,
        seed=7,
    )
    assert bundle["path_head_trained"] is True
    assert bundle["path_relation_head_trained"] is True
    assert bundle["path_mode_head_trained"] is True
    assert bundle["path_improve_head_trained"] is True

    out_path = tmp_path / "repair_controller_oracle_pretrain.pt"
    save_repair_critic_bundle(bundle, out_path)
    loaded = load_repair_critic_bundle(out_path)

    repair_heads = predict_repair_controller_heads(loaded, RepairControllerFeatureRecord.from_flat_row(_oracle_task(100, repair=True)["controller_row"]))
    build_heads = predict_repair_controller_heads(loaded, RepairControllerFeatureRecord.from_flat_row(_oracle_task(101, repair=False)["controller_row"]))

    assert repair_heads["path"]["best_path"] == [1]
    assert repair_heads["path"]["rows"][0]["best_mode"] == "identity"
    assert repair_heads["path"]["rows"][0]["best_relation"] == "same"
    assert repair_heads["path"]["rows"][0]["improvement_estimate"] > repair_heads["path"]["rows"][1]["improvement_estimate"]
    assert build_heads["path"]["best_path"] == [2]
    assert build_heads["path"]["rows"][1]["best_mode"] == "full"
    assert build_heads["path"]["rows"][1]["best_relation"] == "same"


def test_hybrid_repair_controller_scores_adds_only_bounded_bonus():
    analytic_score = 0.24
    hybrid = explorer._hybrid_repair_controller_scores(
        analytic_score,
        {
            "utility_score": 0.90,
            "accept_prob": 0.85,
            "positive_reward_prob": 0.80,
            "reward_per_s_score": 0.75,
        },
        1.0,
    )
    assert hybrid["gate_score"] == analytic_score
    assert hybrid["priority_score"] > analytic_score
    assert 0.0 < hybrid["critic_bonus"] <= 0.20
    assert hybrid["source"] == "analytic_refine_critic_bonus"


def test_hybrid_repair_controller_scores_does_not_penalize_low_critic_rows():
    analytic_score = 0.24
    hybrid = explorer._hybrid_repair_controller_scores(
        analytic_score,
        {
            "utility_score": 0.08,
            "accept_prob": 0.10,
            "positive_reward_prob": 0.05,
            "reward_per_s_score": 0.02,
        },
        1.0,
    )
    assert hybrid["gate_score"] == analytic_score
    assert hybrid["priority_score"] == analytic_score
    assert hybrid["critic_bonus"] == 0.0
    assert hybrid["source"] == "analytic"


def test_load_inverse_experiment_rows_from_report(tmp_path):
    rows = [_row(0, good=True), _row(1, good=False)]
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"inverse_experiment_log": rows}, indent=2), encoding="utf-8")
    loaded = load_inverse_experiment_rows([report_path])
    assert len(loaded) == 2
    assert loaded[0]["parent_expr"] == "good_0"


def test_load_inverse_log_alias(tmp_path):
    rows = [_row(2, good=True), _row(3, good=False)]
    report_path = tmp_path / "legacy_report.json"
    report_path.write_text(json.dumps([{"inverse_log": rows}], indent=2), encoding="utf-8")
    loaded = load_inverse_experiment_rows([report_path])
    assert len(loaded) == 2
    assert loaded[1]["parent_expr"] == "bad_3"


def test_extract_repair_critic_features_uses_path_summary_and_refine_fields():
    row = _row(7, good=True)
    row.update({
        "path_summary_gain_mass": 0.82,
        "path_summary_gap": 0.64,
        "path_summary_support": 0.91,
        "path_summary_mode_diversity": 0.50,
        "refine_slot_count": 2,
        "refine_gate_potential": 0.75,
        "refine_variant_count": 3,
    })
    feats = extract_repair_critic_features(row)
    assert feats["path_summary_gain_mass"] == 0.82
    assert feats["path_summary_gap"] == 0.64
    assert feats["path_summary_support"] == 0.91
    assert feats["path_summary_mode_diversity"] == 0.50
    assert feats["refine_slot_count"] == 2.0
    assert feats["refine_gate_potential"] == 0.75
    assert feats["refine_variant_log"] > 0.0


def test_extract_repair_critic_features_matches_typed_canonical_record():
    row = _row(9, good=True)
    row.update({
        "path_summaries": [
            {
                "path": [1],
                "target_mode": "identity",
                "weighted_rel_gain": 0.85,
                "rel_gain": 0.80,
                "valid_frac": 0.95,
                "confidence": 0.90,
                "static_score": 1.10,
                "transport_rel": 0.60,
                "branch_factor": 1.0,
                "cut_factor": 1.0,
            },
            {
                "path": [2],
                "target_mode": "full",
                "weighted_rel_gain": 0.30,
                "rel_gain": 0.25,
                "valid_frac": 0.70,
                "confidence": 0.55,
                "static_score": 0.80,
                "transport_rel": 0.20,
                "branch_factor": 1.0,
                "cut_factor": 1.0,
            },
        ],
        "path_summary_gain_mass": 0.82,
        "path_summary_gap": 0.64,
        "path_summary_support": 0.91,
        "path_summary_mode_diversity": 0.50,
        "refine_slot_count": 2,
        "refine_gate_potential": 0.75,
        "refine_variant_count": 3,
    })
    row_feats = extract_repair_critic_features(row)
    record_feats = extract_repair_critic_features(RepairControllerFeatureRecord.from_flat_row(row))
    assert row_feats == record_feats
