# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Focused regression checks for the macro controller interface.

Run:
  python nestynet_sr/sr_search/factorized_search/smoke_controller_interface.py
"""
import pathlib
import sys
import random

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from nestynet_sr.sr_search.factorized_search.controller import (
    MacroController,
    build_macro_controller_state,
)
from nestynet_sr.sr_search.factorized_search.engine.signals import (
    InverseSteeringPotential,
    PathStateFeatures,
)
from nestynet_sr.sr_search.factorized_search.policy.features import (
    RepairControllerFeatureRecord,
)


n_pass = 0
n_fail = 0


def check(name, ok, detail=""):
    global n_pass, n_fail
    if ok:
        n_pass += 1
        print(f"  PASS  {name}  {detail}")
    else:
        n_fail += 1
        print(f"  FAIL  {name}  {detail}")


ACTION_NAME = {
    0: "replace",
    8: "inv_steer",
}


row_strong = {
    "parent_expr": "(x0+x1)",
    "parent_best_eff_mse": 1.0e-2,
    "parent_best_raw_mse": 1.0e-2,
    "parent_visits": 12,
    "parent_visits_since_improve": 7,
    "parent_stagnation_score": 0.6,
    "parent_stagnation_ratio": 0.5,
    "gate_allowed": True,
    "gate_reason": "ok",
    "gate_best_path": [1],
    "gate_best_weighted_rel_gain": 0.85,
    "gate_best_rel_gain": 0.80,
    "gate_best_valid_frac": 0.95,
    "gate_best_confidence": 0.90,
    "gate_best_transport_rel": 0.60,
    "gate_best_static_score": 1.10,
    "path_entropy": 0.05,
    "path_top_mass": 0.95,
    "path_second_mass": 0.03,
    "path_positive_count": 2,
    "identity_vs_full_log_mse_contrast": 2.5,
    "selected_target_mode": "identity",
    "selected_path": [1],
    "selected_path_gain": 0.85,
    "selected_path_gain_pre_cut": 0.90,
    "selected_rel_gain": 0.80,
    "selected_transport_rel": 0.60,
    "selected_branch_factor": 1.0,
    "selected_cut_factor": 1.0,
    "selected_effective_n": 64,
    "local_candidate_count": 4,
    "estimated_one_hole_rel_improve_eff": 0.70,
}

row_weak = dict(row_strong)
row_weak.update({
    "path_entropy": 0.68,
    "path_top_mass": 0.28,
    "path_second_mass": 0.23,
    "path_positive_count": 5,
    "identity_vs_full_log_mse_contrast": 0.10,
    "estimated_one_hole_rel_improve_eff": 0.05,
    "selected_path_gain": 0.08,
    "gate_best_weighted_rel_gain": 0.08,
})


diag = InverseSteeringPotential(
    allowed=True,
    reason="ok",
    best_path=(1,),
    best_rel_gain=0.8,
    best_weighted_rel_gain=0.85,
    candidate_paths=((1,), (2,)),
    path_rows=(
        PathStateFeatures(path=(1,), weighted_rel_gain=0.85, rel_gain=0.8, valid_frac=0.95, confidence=0.90, static_score=1.10, transport_rel=0.60, target_mode="identity"),
        PathStateFeatures(path=(2,), weighted_rel_gain=0.30, rel_gain=0.25, valid_frac=0.70, confidence=0.55, static_score=0.80, transport_rel=0.20, target_mode="full"),
    ),
)

diag_single = InverseSteeringPotential(
    allowed=True,
    reason="ok",
    best_path=(1,),
    best_rel_gain=0.8,
    best_weighted_rel_gain=0.85,
    candidate_paths=((1,),),
    path_rows=(
        PathStateFeatures(path=(1,), weighted_rel_gain=0.85, rel_gain=0.8, valid_frac=0.95, confidence=0.90, static_score=1.10, transport_rel=0.60, target_mode="identity"),
    ),
)


print("\n=== Test: build_macro_controller_state appends repair_option only when ready ===")
state_ready = build_macro_controller_state(
    parent_key="same-parent",
    parent_expr="(x0+x1)",
    parent_root_op="add",
    parent_depth=3,
    parent_size=5,
    allowed_actions=[0, 8],
    action_name_map=ACTION_NAME,
    gate_diag=diag,
    controller_row=row_strong,
    repair_priority_score=0.9,
    repair_gate_score=0.8,
    repair_threshold=0.2,
    repair_ready=True,
    repair_preview_available=True,
    repair_component_ok=True,
)
check("repair appended", state_ready.allowed_actions == ("replace", "inv_steer", "repair_option"), f"allowed={state_ready.allowed_actions}")
check("path summaries kept", len(state_ready.path_summaries) == 2, f"n={len(state_ready.path_summaries)}")

state_not_ready = build_macro_controller_state(
    parent_key="same-parent",
    parent_expr="(x0+x1)",
    parent_root_op="add",
    parent_depth=3,
    parent_size=5,
    allowed_actions=[0, 8],
    action_name_map=ACTION_NAME,
    gate_diag=diag,
    controller_row=row_strong,
    repair_priority_score=0.9,
    repair_gate_score=0.8,
    repair_threshold=0.2,
    repair_ready=False,
    repair_preview_available=True,
    repair_component_ok=True,
)
check("repair omitted when not ready", state_not_ready.allowed_actions == ("replace", "inv_steer"), f"allowed={state_not_ready.allowed_actions}")

state_ready_record = build_macro_controller_state(
    parent_key="same-parent",
    parent_expr="(x0+x1)",
    parent_root_op="add",
    parent_depth=3,
    parent_size=5,
    allowed_actions=[0, 8],
    action_name_map=ACTION_NAME,
    gate_diag=diag,
    controller_row=RepairControllerFeatureRecord.from_flat_row(row_strong),
    repair_priority_score=0.9,
    repair_gate_score=0.8,
    repair_threshold=0.2,
    repair_ready=True,
    repair_preview_available=True,
    repair_component_ok=True,
)
check(
    "typed canonical record matches dict state",
    state_ready_record.bandit_state_key == state_ready.bandit_state_key,
    f"record={state_ready_record.bandit_state_key} dict={state_ready.bandit_state_key}",
)


print("\n=== Test: state key uses top-k path summaries and plus signals ===")
state_single_path = build_macro_controller_state(
    parent_key="same-parent",
    parent_expr="(x0+x1)",
    parent_root_op="add",
    parent_depth=3,
    parent_size=5,
    allowed_actions=[0, 8],
    action_name_map=ACTION_NAME,
    gate_diag=diag_single,
    controller_row=row_strong,
    repair_priority_score=0.9,
    repair_gate_score=0.8,
    repair_threshold=0.2,
    repair_ready=True,
    repair_preview_available=True,
    repair_component_ok=True,
)
check(
    "path summary state differs",
    state_ready.bandit_state_key != state_single_path.bandit_state_key,
    f"ready={state_ready.bandit_state_key} single={state_single_path.bandit_state_key}",
)

state_plus = build_macro_controller_state(
    parent_key="same-parent",
    parent_expr="(x0+x1)",
    parent_root_op="add",
    parent_depth=3,
    parent_size=5,
    allowed_actions=[0, 8],
    action_name_map=ACTION_NAME,
    gate_diag=diag,
    controller_row=row_strong,
    repair_priority_score=0.9,
    repair_gate_score=0.8,
    repair_threshold=0.2,
    repair_ready=True,
    repair_preview_available=True,
    repair_component_ok=True,
    refine_slot_count=2,
    refine_gate_potential=0.75,
)
check(
    "plus signal state differs",
    state_ready.bandit_state_key != state_plus.bandit_state_key,
    f"ready={state_ready.bandit_state_key} plus={state_plus.bandit_state_key}",
)


print("\n=== Test: state key uses dynamic repair features, not just parent key ===")
state_weak = build_macro_controller_state(
    parent_key="same-parent",
    parent_expr="(x0+x1)",
    parent_root_op="add",
    parent_depth=3,
    parent_size=5,
    allowed_actions=[0, 8],
    action_name_map=ACTION_NAME,
    gate_diag=diag,
    controller_row=row_weak,
    repair_priority_score=0.1,
    repair_gate_score=0.05,
    repair_threshold=0.2,
    repair_ready=False,
    repair_preview_available=False,
    repair_component_ok=False,
)
check(
    "dynamic state key differs",
    state_ready.bandit_state_key != state_weak.bandit_state_key,
    f"ready={state_ready.bandit_state_key} weak={state_weak.bandit_state_key}",
)


print("\n=== Test: MacroController prefers repair when repair is strong ===")
ctl = MacroController(
    ["replace", "inv_steer", "repair_option"],
    ucb_c=0.0,
    eps=0.0,
    build_bias=0.0,
    inverse_bonus=0.0,
    repair_bonus=1.0,
    repair_margin_scale=0.0,
)
decision = ctl.select_action(state_ready, random.Random(0))
check("repair selected", decision.action_name == "repair_option", f"action={decision.action_name} scores={decision.scores}")


print("\n=== Test: MacroController respects allowed actions when repair is unavailable ===")
ctl2 = MacroController(["replace", "repair_option"], ucb_c=0.0, eps=0.0, build_bias=0.0, inverse_bonus=0.0, repair_bonus=1.0)
state_only_replace = state_weak.with_allowed_actions(["replace"])
decision2 = ctl2.select_action(state_only_replace, random.Random(0))
check("replace selected", decision2.action_name == "replace", f"action={decision2.action_name} scores={decision2.scores}")


print("\n=== Test: MacroController can blend learned macro policy/value guidance ===")
ctl3 = MacroController(
    ["replace", "repair_option"],
    ucb_c=0.0,
    eps=0.0,
    build_bias=0.0,
    inverse_bonus=0.0,
    repair_bonus=0.0,
    learned_policy_weight=2.0,
    learned_value_scale=1.0,
)
state_learned = state_ready.with_allowed_actions(["replace", "repair_option"])
decision3 = ctl3.select_action(
    state_learned,
    random.Random(0),
    policy_guidance={
        "macro_action": {
            "trained": True,
            "best_action": "repair_option",
            "probs": {
                "replace": 0.05,
                "repair_option": 0.95,
            },
        },
        "route": {
            "trained": True,
            "best_route": "repair",
            "probs": {
                "build": 0.10,
                "repair": 0.90,
            },
        },
        "action_value": {
            "trained": True,
            "best_action": "repair_option",
            "estimates": {
                "replace": -0.6,
                "repair_option": 1.4,
            },
            "normalized_estimates": {
                "replace": -1.0,
                "repair_option": 1.5,
            },
        },
        "value": {
            "trained": True,
            "estimate": 1.25,
            "normalized_estimate": 2.0,
        },
    },
)
check("learned policy selected", decision3.action_name == "repair_option", f"action={decision3.action_name} scores={decision3.scores}")
check("learned source tagged", "argmax" in decision3.policy_source, f"source={decision3.policy_source}")
check("learned value carried", decision3.learned_value_estimate and decision3.learned_value_estimate > 0.0, f"value={decision3.learned_value_estimate}")
check("learned route carried", decision3.learned_best_route == "repair", f"route={decision3.learned_best_route}")
check("learned q carried", decision3.learned_action_value_estimates.get("repair_option", 0.0) > decision3.learned_action_value_estimates.get("replace", 0.0), f"q={decision3.learned_action_value_estimates}")
check("selected route carried", decision3.selected_route == "repair", f"selected_route={decision3.selected_route}")


print("\n=== Test: MacroController uses route-first selection before within-route action ===")
ctl4 = MacroController(
    ["replace", "wrap_un", "repair_option"],
    ucb_c=0.0,
    eps=0.0,
    build_bias=0.0,
    inverse_bonus=0.0,
    repair_bonus=0.0,
    learned_policy_weight=2.0,
    learned_value_scale=1.0,
)
state_hier = state_ready.with_allowed_actions(["replace", "wrap_un", "repair_option"])
decision4 = ctl4.select_action(
    state_hier,
    random.Random(0),
    policy_guidance={
        "macro_action": {
            "trained": True,
            "best_action": "repair_option",
            "probs": {
                "replace": 0.30,
                "wrap_un": 0.25,
                "repair_option": 0.45,
            },
        },
        "route": {
            "trained": True,
            "best_route": "build",
            "probs": {
                "build": 0.85,
                "repair": 0.15,
            },
        },
        "action_value": {
            "trained": True,
            "best_action": "replace",
            "estimates": {
                "replace": 0.8,
                "wrap_un": 0.3,
                "repair_option": 0.5,
            },
            "normalized_estimates": {
                "replace": 1.1,
                "wrap_un": 0.2,
                "repair_option": 0.6,
            },
        },
        "value": {
            "trained": True,
            "estimate": 0.7,
            "normalized_estimate": 1.0,
        },
    },
)
check("hierarchical route selected build", decision4.selected_route == "build", f"route={decision4.selected_route} scores={decision4.route_decision_scores}")
check("hierarchical action selected within build", decision4.action_name == "replace", f"action={decision4.action_name} scores={decision4.scores}")
check("hierarchical source tagged", decision4.policy_source == "hierarchical_learned_route_argmax", f"source={decision4.policy_source}")


print("\n=== Test: MacroController can choose an explicit path-route-action tuple ===")
ctl5 = MacroController(
    ["replace", "wrap_un", "repair_option"],
    ucb_c=0.0,
    eps=0.0,
    build_bias=0.0,
    inverse_bonus=0.0,
    repair_bonus=0.0,
    learned_policy_weight=1.5,
    learned_route_weight=1.0,
    learned_q_weight=0.5,
)
decision5 = ctl5.select_action(
    state_hier,
    random.Random(0),
    policy_guidance={
        "path_action": {
            "trained": True,
            "best_path": [2],
            "best_route": "build",
            "best_action": "replace",
            "rows": [
                {
                    "path": [1],
                    "path_prob": 0.25,
                    "route_probs": {"build": 0.35, "repair": 0.65},
                    "action_probs": {"replace": 0.05, "wrap_un": 0.05, "repair_option": 0.55},
                    "within_route_action_probs": {"replace": 0.50, "wrap_un": 0.50, "repair_option": 1.0},
                    "q_estimates": {"replace": 0.0, "wrap_un": 0.0, "repair_option": 1.0},
                    "normalized_estimates": {"replace": -0.5, "wrap_un": -0.5, "repair_option": 1.2},
                },
                {
                    "path": [2],
                    "path_prob": 0.75,
                    "route_probs": {"build": 0.90, "repair": 0.10},
                    "action_probs": {"replace": 0.72, "wrap_un": 0.18, "repair_option": 0.10},
                    "within_route_action_probs": {"replace": 0.80, "wrap_un": 0.20, "repair_option": 1.0},
                    "q_estimates": {"replace": 1.3, "wrap_un": 0.4, "repair_option": 0.2},
                    "normalized_estimates": {"replace": 1.1, "wrap_un": 0.1, "repair_option": -0.4},
                },
            ],
        },
    },
)
check("tuple action selected", decision5.action_name == "replace", f"action={decision5.action_name} scores={decision5.scores}")
check("tuple route selected", decision5.selected_route == "build", f"route={decision5.selected_route}")
check("tuple path carried", decision5.selected_path == (2,), f"path={decision5.selected_path}")
check("tuple source tagged", decision5.policy_source == "path_tuple_hierarchical_argmax", f"source={decision5.policy_source}")


print("\n=== Test: path-action aggregation uses top-k support, not single-path max ===")
ctl6 = MacroController(
    ["replace", "wrap_un"],
    ucb_c=0.0,
    eps=0.0,
    build_bias=0.0,
    inverse_bonus=0.0,
    repair_bonus=0.0,
    learned_policy_weight=1.0,
    learned_route_weight=0.0,
    learned_q_weight=0.0,
)
state_build = state_ready.with_allowed_actions(["replace", "wrap_un"])
decision6 = ctl6.select_action(
    state_build,
    random.Random(0),
    policy_guidance={
        "path_action": {
            "trained": True,
            "rows": [
                {
                    "path": [1],
                    "path_prob": 0.30,
                    "route_probs": {"build": 1.0},
                    "action_probs": {"replace": 0.80, "wrap_un": 0.20},
                    "within_route_action_probs": {"replace": 0.80, "wrap_un": 0.20},
                    "q_estimates": {"replace": 0.0, "wrap_un": 0.0},
                    "normalized_estimates": {"replace": 0.0, "wrap_un": 0.0},
                },
                {
                    "path": [2],
                    "path_prob": 0.30,
                    "route_probs": {"build": 1.0},
                    "action_probs": {"replace": 0.80, "wrap_un": 0.20},
                    "within_route_action_probs": {"replace": 0.80, "wrap_un": 0.20},
                    "q_estimates": {"replace": 0.0, "wrap_un": 0.0},
                    "normalized_estimates": {"replace": 0.0, "wrap_un": 0.0},
                },
                {
                    "path": [3],
                    "path_prob": 0.30,
                    "route_probs": {"build": 1.0},
                    "action_probs": {"replace": 0.80, "wrap_un": 0.20},
                    "within_route_action_probs": {"replace": 0.80, "wrap_un": 0.20},
                    "q_estimates": {"replace": 0.0, "wrap_un": 0.0},
                    "normalized_estimates": {"replace": 0.0, "wrap_un": 0.0},
                },
                {
                    "path": [4],
                    "path_prob": 0.60,
                    "route_probs": {"build": 1.0},
                    "action_probs": {"replace": 0.20, "wrap_un": 0.80},
                    "within_route_action_probs": {"replace": 0.20, "wrap_un": 0.80},
                    "q_estimates": {"replace": 0.0, "wrap_un": 0.0},
                    "normalized_estimates": {"replace": 0.0, "wrap_un": 0.0},
                },
            ],
        },
    },
)
check("top-k path aggregation selected replace", decision6.action_name == "replace", f"action={decision6.action_name} scores={decision6.scores}")
check("top-k path aggregation kept tuple source", decision6.policy_source == "path_tuple_argmax", f"source={decision6.policy_source}")
check("top-k path aggregation carried best replace path", decision6.selected_path in ((1,), (2,), (3,)), f"path={decision6.selected_path}")


print("\n=== Test: confident learned policy overrides stale bandit preference ===")
ctl7 = MacroController(
    ["replace", "wrap_un"],
    ucb_c=0.0,
    eps=0.0,
    build_bias=0.0,
    inverse_bonus=0.0,
    repair_bonus=0.0,
    learned_policy_weight=1.0,
)
for _ in range(8):
    ctl7.update(state_build, "wrap_un", 1.0)
decision7 = ctl7.select_action(
    state_build,
    random.Random(0),
    policy_guidance={
        "macro_action": {
            "trained": True,
            "probs": {
                "replace": 0.95,
                "wrap_un": 0.05,
            },
        },
    },
)
check(
    "learned primary beats bandit fallback",
    decision7.action_name == "replace",
    f"action={decision7.action_name} scores={decision7.scores} bandit={decision7.bandit_scores}",
)
check(
    "learned-primary source tagged",
    decision7.policy_source == "learned_primary_argmax",
    f"source={decision7.policy_source}",
)


print("\n=== Test: weak learned signal falls back to bandit preference ===")
ctl8 = MacroController(
    ["replace", "wrap_un"],
    ucb_c=0.0,
    eps=0.0,
    build_bias=0.0,
    inverse_bonus=0.0,
    repair_bonus=0.0,
    learned_policy_weight=1.0,
)
for _ in range(8):
    ctl8.update(state_build, "wrap_un", 1.0)
decision8 = ctl8.select_action(
    state_build,
    random.Random(0),
    policy_guidance={
        "macro_action": {
            "trained": True,
            "probs": {
                "replace": 0.51,
                "wrap_un": 0.49,
            },
        },
    },
)
check(
    "bandit fallback wins on weak learned signal",
    decision8.action_name == "wrap_un",
    f"action={decision8.action_name} scores={decision8.scores} bandit={decision8.bandit_scores}",
)
check(
    "fallback source tagged",
    decision8.policy_source == "fallback_score_argmax",
    f"source={decision8.policy_source}",
)


print(f"\n{'='*50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    sys.exit(1)
print("All tests passed!")
