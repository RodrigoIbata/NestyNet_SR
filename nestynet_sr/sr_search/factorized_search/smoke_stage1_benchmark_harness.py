# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Regression checks for the Stage-1 benchmark harness.

Run:
  python nestynet_sr/sr_search/factorized_search/smoke_stage1_benchmark_harness.py
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from nestynet_sr.sr_search.factorized_search import explorer
from nestynet_sr.sr_search.factorized_search.explorer import _hybrid_repair_controller_scores
from nestynet_sr.sr_search.factorized_search.engine import search as engine_search
from nestynet_sr.sr_search.factorized_search.stage1_benchmark_harness import (
    _arm_configs,
    run_stage1_experiment,
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


print("\n=== Test: hybrid controller modes change the decisive quantities ===")
strong_preds = {
    "utility_score": 0.95,
    "accept_prob": 0.90,
    "positive_reward_prob": 0.90,
    "reward_per_s_score": 0.85,
}
weak_preds = {
    "utility_score": 0.05,
    "accept_prob": 0.05,
    "positive_reward_prob": 0.05,
    "reward_per_s_score": 0.05,
}
priority_scores = _hybrid_repair_controller_scores(0.20, strong_preds, 1.0, "priority")
gate_scores = _hybrid_repair_controller_scores(0.20, strong_preds, 1.0, "gate")
decisive_scores = _hybrid_repair_controller_scores(0.20, weak_preds, 1.0, "decisive")
check("priority keeps analytic gate", abs(float(priority_scores["gate_score"]) - 0.20) < 1.0e-12, f"gate={priority_scores['gate_score']}")
check(
    "priority boosts only priority score",
    float(priority_scores["priority_score"]) > float(priority_scores["gate_score"]),
    f"priority={priority_scores['priority_score']} gate={priority_scores['gate_score']}",
)
check("gate mode raises gate directly", float(gate_scores["gate_score"]) > 0.20, f"gate={gate_scores['gate_score']}")
check("decisive mode can suppress gate", float(decisive_scores["gate_score"]) < 0.20, f"gate={decisive_scores['gate_score']}")
check(
    "decisive mode shifts threshold upward on weak critic",
    float(decisive_scores["threshold_shift"]) > 0.0,
    f"threshold_shift={decisive_scores['threshold_shift']}",
)


print("\n=== Test: arm configs expose priority, gate, and macro variants ===")
arms = _arm_configs(
    critic_path="/tmp/stub_repair_critic.pt",
    blends=[0.50],
    refine_enable=True,
    arm_modes=("priority", "gate", "macro"),
    macro_profile="repair_probe",
    macro_controller_learned_policy_weight=0.0,
    macro_controller_learned_route_weight=0.25,
    macro_controller_learned_q_weight=0.0,
)
arm_map = {name: cfg for name, cfg in arms}
check("baseline arm present", "stage0_selective_plus" in arm_map, f"arms={list(arm_map)}")
check("priority arm present", "stage1_hybrid_b050_plus" in arm_map, f"arms={list(arm_map)}")
check("gate arm present", "stage1_gate_b050_plus" in arm_map, f"arms={list(arm_map)}")
check(
    "macro probe arm present",
    "stage1_macro_repair_probe_b050_plus" in arm_map,
    f"arms={list(arm_map)}",
)
check(
    "gate arm uses decisive critic mode",
    str(arm_map["stage1_gate_b050_plus"].get("repair_controller_critic_mode", "")) == "decisive",
    f"mode={arm_map['stage1_gate_b050_plus'].get('repair_controller_critic_mode')}",
)
check(
    "macro arm enables macro controller",
    bool(arm_map["stage1_macro_repair_probe_b050_plus"].get("macro_controller_enable", False)),
    f"cfg={arm_map['stage1_macro_repair_probe_b050_plus']}",
)
check(
    "macro arm forwards learned policy ablation weight",
    float(arm_map["stage1_macro_repair_probe_b050_plus"].get("macro_controller_learned_policy_weight", -1.0)) == 0.0,
    f"cfg={arm_map['stage1_macro_repair_probe_b050_plus']}",
)
check(
    "macro arm forwards learned q ablation weight",
    float(arm_map["stage1_macro_repair_probe_b050_plus"].get("macro_controller_learned_q_weight", -1.0)) == 0.0,
    f"cfg={arm_map['stage1_macro_repair_probe_b050_plus']}",
)


print("\n=== Test: arm configs expose scheduler advisory and control variants ===")
scheduler_arms = dict(
    _arm_configs(
        critic_path="",
        scheduler_bundle_path="/tmp/stub_scheduler_bundle.pt",
        scheduler_budget_ladder=(1, 2, 4),
        blends=[0.50],
        refine_enable=False,
        arm_modes=("scheduler_advisory", "scheduler_control"),
        macro_profile="repair_probe",
    )
)
check(
    "scheduler advisory arm present",
    "stage1_scheduler_advisory_repair_probe" in scheduler_arms,
    f"arms={list(scheduler_arms)}",
)
check(
    "scheduler control arm present",
    "stage1_scheduler_control_repair_probe" in scheduler_arms,
    f"arms={list(scheduler_arms)}",
)
check(
    "scheduler advisory sets advisory mode",
    bool(scheduler_arms["stage1_scheduler_advisory_repair_probe"].get("scheduler_advisory_only", False)),
    f"cfg={scheduler_arms['stage1_scheduler_advisory_repair_probe']}",
)
check(
    "scheduler control disables advisory mode",
    not bool(scheduler_arms["stage1_scheduler_control_repair_probe"].get("scheduler_advisory_only", True)),
    f"cfg={scheduler_arms['stage1_scheduler_control_repair_probe']}",
)
check(
    "scheduler control enables scheduler",
    bool(scheduler_arms["stage1_scheduler_control_repair_probe"].get("scheduler_enable", False)),
    f"cfg={scheduler_arms['stage1_scheduler_control_repair_probe']}",
)


print("\n=== Test: stage1 smoke runs exercise gate and macro paths with a stub critic ===")
torch.set_num_threads(1)
orig_load = explorer.load_repair_critic_bundle
orig_predict_heads = explorer.predict_repair_controller_heads
orig_load_scheduler_bundle = engine_search.load_scheduler_bundle
explorer.load_repair_critic_bundle = lambda _path: {"stub": True}
explorer.predict_repair_controller_heads = lambda _bundle, _row: {
    "auxiliary": dict(strong_preds),
    "macro_action": {
        "trained": True,
        "best_action": "inv_steer",
        "probs": {
            "replace": 0.01,
            "inv_steer": 0.98,
            "repair_option": 0.01,
        },
    },
    "path": {"trained": False, "best_path": None, "best_target_mode": None, "rows": []},
    "value": {"trained": True, "estimate": 1.5, "normalized_estimate": 2.0},
}
engine_search.load_scheduler_bundle = lambda _path: {"scheduler_critic_trained": True}
try:
    smoke_arms = dict(
        _arm_configs(
            critic_path="/tmp/stub_repair_critic.pt",
            blends=[0.50],
            refine_enable=False,
            arm_modes=("gate", "macro"),
            macro_profile="repair_probe",
        )
    )
    gate_cfg = dict(smoke_arms["stage1_gate_b050"])
    gate_cfg.update({
        "inverse_gate_min_depth": 0,
        "inverse_gate_min_size": 1,
        "inverse_gate_min_structural_score": 0.0,
        "inverse_gate_min_weighted_rel_gain": 0.0,
        "repair_controller_min_score": 0.0,
        "repair_controller_min_concentration": 0.0,
        "repair_controller_adaptive": False,
        "repair_controller_focus_prob": 1.0,
    })
    macro_cfg = dict(smoke_arms["stage1_macro_repair_probe_b050"])
    advisory_cfg = dict(
        _arm_configs(
            critic_path="",
            scheduler_bundle_path="/tmp/stub_scheduler_bundle.pt",
            scheduler_budget_ladder=(1, 2, 4),
            blends=[0.50],
            refine_enable=False,
            arm_modes=("scheduler_advisory", "scheduler_control"),
            macro_profile="repair_probe",
        )
    )["stage1_scheduler_advisory_repair_probe"]
    control_cfg = dict(
        _arm_configs(
            critic_path="",
            scheduler_bundle_path="/tmp/stub_scheduler_bundle.pt",
            scheduler_budget_ladder=(1, 2, 4),
            blends=[0.50],
            refine_enable=False,
            arm_modes=("scheduler_advisory", "scheduler_control"),
            macro_profile="repair_probe",
        )
    )["stage1_scheduler_control_repair_probe"]

    gate_summary = run_stage1_experiment(
        "addsum",
        seed=0,
        arm="stage1_gate_b050",
        arm_cfg=gate_cfg,
        n_iter=10,
        max_depth=4,
        n_fit=64,
        n_probe=128,
        refine_enable=False,
        capture_search_output=True,
        threads=1,
    )
    macro_summary = run_stage1_experiment(
        "addsum",
        seed=0,
        arm="stage1_macro_repair_probe_b050",
        arm_cfg=macro_cfg,
        n_iter=10,
        max_depth=4,
        n_fit=64,
        n_probe=128,
        refine_enable=False,
        capture_search_output=True,
        threads=1,
    )
    advisory_summary = run_stage1_experiment(
        "addsum",
        seed=0,
        arm="stage1_scheduler_advisory_repair_probe",
        arm_cfg=advisory_cfg,
        n_iter=10,
        max_depth=4,
        n_fit=64,
        n_probe=128,
        refine_enable=False,
        capture_search_output=True,
        threads=1,
    )
    control_summary = run_stage1_experiment(
        "addsum",
        seed=0,
        arm="stage1_scheduler_control_repair_probe",
        arm_cfg=control_cfg,
        n_iter=10,
        max_depth=4,
        n_fit=64,
        n_probe=128,
        refine_enable=False,
        capture_search_output=True,
        threads=1,
    )
finally:
    explorer.load_repair_critic_bundle = orig_load
    explorer.predict_repair_controller_heads = orig_predict_heads
    engine_search.load_scheduler_bundle = orig_load_scheduler_bundle

check("gate smoke loaded critic", gate_summary.critic_loaded, f"loaded={gate_summary.critic_loaded}")
check("gate smoke recorded decisive mode", gate_summary.critic_mode == "decisive", f"mode={gate_summary.critic_mode}")
check("gate smoke considered repair", gate_summary.repair_considered > 0, f"considered={gate_summary.repair_considered}")
check(
    "gate smoke logged critic source",
    any("critic" in key for key in gate_summary.controller_score_source_counts),
    f"sources={gate_summary.controller_score_source_counts}",
)
check("macro smoke enabled macro controller", macro_summary.macro_enabled, f"enabled={macro_summary.macro_enabled}")
check("macro smoke selected macro actions", macro_summary.macro_selected > 0, f"selected={macro_summary.macro_selected}")
check(
    "macro smoke used learned blend decisions",
    any(("learned" in str(key)) for key in macro_summary.macro_decision_source_counts),
    f"sources={macro_summary.macro_decision_source_counts}",
)
check(
    "macro smoke retained stdout tail",
    len(macro_summary.search_stdout_tail) > 0,
    f"tail={macro_summary.search_stdout_tail}",
)
check(
    "scheduler advisory smoke enables scheduler",
    advisory_summary.scheduler_enabled and advisory_summary.scheduler_advisory_only,
    f"enabled={advisory_summary.scheduler_enabled} advisory={advisory_summary.scheduler_advisory_only}",
)
check(
    "scheduler advisory smoke loads bundle",
    advisory_summary.scheduler_bundle_loaded,
    f"loaded={advisory_summary.scheduler_bundle_loaded}",
)
check(
    "scheduler control smoke enables control mode",
    control_summary.scheduler_enabled and (not control_summary.scheduler_advisory_only),
    f"enabled={control_summary.scheduler_enabled} advisory={control_summary.scheduler_advisory_only}",
)
check(
    "scheduler control smoke reports route usage",
    sum(int(v) for v in control_summary.route_usage.values()) >= 0,
    f"route_usage={control_summary.route_usage}",
)


print(f"\n{'='*50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    sys.exit(1)
print("All tests passed!")
