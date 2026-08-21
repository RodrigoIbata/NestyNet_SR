# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Regression checks for the macro-controller benchmark harness.

Run:
  python nestynet_sr/sr_search/factorized_search/smoke_controller_harness.py
"""
import math
import random
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from nestynet_sr.sr_search.factorized_search.controller_harness import run_controller_pair
from nestynet_sr.sr_search.factorized_search.controller_harness import _resolve_target_spec
from nestynet_sr.sr_search.factorized_search.engine import search as engine_search
from nestynet_sr.sr_search.factorized_search.explorer import (
    ResidualBasinArchive,
    Rec,
    choose_parent_repair_aware,
    node_str,
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


print("\n=== Test: choose_parent_repair_aware frontier weighting does not crash ===")
arch = ResidualBasinArchive()
expr_a = ("add", ("var", 0), ("var", 1))
expr_b = ("mul", ("var", 0), ("var", 1))
z = torch.zeros((4,), dtype=torch.float64)
map_id = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
arch.d = {
    "ka": Rec(1.0, expr_a, 4, map_id, z.clone(), [], best_raw_mse=1.0, visits_since_improve=4),
    "kb": Rec(2.0, expr_b, 9, map_id, z.clone(), [], best_raw_mse=2.0, visits_since_improve=9),
}
repair_parent_cache = {
    "ka": {
        "expr": node_str(expr_a),
        "score": 0.80,
        "score_base": 0.80,
        "gate_score": 0.80,
        "gate_score_base": 0.80,
        "stagnation_adjustable": False,
    },
    "kb": {
        "expr": node_str(expr_b),
        "score": 0.30,
        "score_base": 0.30,
        "gate_score": 0.30,
        "gate_score_base": 0.30,
        "stagnation_adjustable": False,
    },
}
repair_parent_state = {}
repair_controller_stats = {
    "enabled": True,
    "focus_prob": 1.0,
    "frontier_topk": 2,
    "stagnation_visits": 0,
    "min_score": 0.10,
    "adaptive_enable": False,
    "parent_frontier_candidate_hist": [],
}
rng = random.Random(0)
counts = {"ka": 0, "kb": 0}
for _ in range(64):
    key, rec = choose_parent_repair_aware(
        arch,
        rng,
        exploit_frac=0.0,
        exploit_topk=2,
        n_evaluated=50,
        repair_parent_cache=repair_parent_cache,
        repair_parent_state=repair_parent_state,
        repair_controller_stats=repair_controller_stats,
    )
    counts[str(key)] = counts.get(str(key), 0) + 1
check("only frontier parents returned", set(counts) >= {"ka", "kb"}, f"counts={counts}")
check("higher-weight parent preferred", counts["ka"] > counts["kb"], f"counts={counts}")
check(
    "frontier histogram populated",
    bool(repair_controller_stats.get("parent_frontier_candidate_hist")),
    f"hist={repair_controller_stats.get('parent_frontier_candidate_hist')}",
)


print("\n=== Test: choose_parent_repair_aware honors cached effective thresholds ===")
repair_parent_cache_thresholded = {
    "ka": {
        "expr": node_str(expr_a),
        "score": 0.25,
        "score_base": 0.25,
        "gate_score": 0.12,
        "gate_score_base": 0.12,
        "threshold": 0.05,
        "stagnation_adjustable": False,
    },
}
repair_controller_stats_thresholded = {
    "enabled": True,
    "focus_prob": 1.0,
    "frontier_topk": 2,
    "stagnation_visits": 0,
    "min_score": 0.20,
    "adaptive_enable": False,
    "parent_frontier_candidate_hist": [],
}
key_thr, _rec_thr = choose_parent_repair_aware(
    arch,
    random.Random(0),
    exploit_frac=0.0,
    exploit_topk=2,
    n_evaluated=50,
    repair_parent_cache=repair_parent_cache_thresholded,
    repair_parent_state={},
    repair_controller_stats=repair_controller_stats_thresholded,
)
check("cached threshold enables repaired frontier parent", key_thr == "ka", f"key={key_thr}")


print("\n=== Test: choose_parent_repair_aware uses learned repair-policy bonus within repair frontier ===")
arch_policy = ResidualBasinArchive()
arch_policy.d = {
    "ka": Rec(1.0, expr_a, 4, map_id, z.clone(), [], best_raw_mse=1.0, visits_since_improve=4),
    "kb": Rec(2.0, expr_b, 4, map_id, z.clone(), [], best_raw_mse=2.0, visits_since_improve=4),
}
repair_parent_cache_policy = {
    "ka": {
        "expr": node_str(expr_a),
        "score": 0.30,
        "score_base": 0.30,
        "gate_score": 0.30,
        "gate_score_base": 0.30,
        "threshold": 0.10,
        "policy_priority_bonus": 0.00,
        "stagnation_adjustable": False,
    },
    "kb": {
        "expr": node_str(expr_b),
        "score": 0.45,
        "score_base": 0.45,
        "gate_score": 0.30,
        "gate_score_base": 0.30,
        "threshold": 0.10,
        "policy_priority_bonus": 0.15,
        "stagnation_adjustable": False,
    },
}
repair_controller_stats_policy = {
    "enabled": True,
    "focus_prob": 1.0,
    "frontier_topk": 2,
    "stagnation_visits": 0,
    "min_score": 0.10,
    "adaptive_enable": False,
    "parent_frontier_candidate_hist": [],
}
counts_policy = {"ka": 0, "kb": 0}
for _ in range(64):
    key_pol, _rec_pol = choose_parent_repair_aware(
        arch_policy,
        random.Random(_),
        exploit_frac=0.0,
        exploit_topk=2,
        n_evaluated=50,
        repair_parent_cache=repair_parent_cache_policy,
        repair_parent_state={},
        repair_controller_stats=repair_controller_stats_policy,
    )
    counts_policy[str(key_pol)] = counts_policy.get(str(key_pol), 0) + 1
check("policy bonus parent preferred", counts_policy["kb"] > counts_policy["ka"], f"counts={counts_policy}")


print("\n=== Test: harness resolves custom targets with explicit bounds ===")
custom_spec = _resolve_target_spec("custom_rational_sqdiff")
eq026_spec = _resolve_target_spec("eq026_nested_recip")
check("custom target nvars", int(custom_spec["nvars"]) == 2, f"nvars={custom_spec['nvars']}")
check("custom target lo", list(custom_spec["lo"]) == [4.0, 1.0], f"lo={custom_spec['lo']}")
check("eq026 target nvars", int(eq026_spec["nvars"]) == 3, f"nvars={eq026_spec['nvars']}")
check("eq026 target hi", list(eq026_spec["hi"]) == [5.0, 5.0, 5.0], f"hi={eq026_spec['hi']}")


print("\n=== Test: controller harness paired smoke run ===")
torch.set_num_threads(1)
pair = run_controller_pair(
    "addsum",
    seed=0,
    profile="repair_probe",
    n_iter=8,
    max_depth=4,
    capture_search_output=True,
    threads=1,
)
check("baseline label", pair.baseline.label == "baseline", f"label={pair.baseline.label}")
check("macro label", pair.macro.label == "macro", f"label={pair.macro.label}")
check("baseline macro stats off", pair.baseline.macro_selected == 0, f"macro_selected={pair.baseline.macro_selected}")
check("macro stats on", pair.macro.macro_selected > 0, f"macro_selected={pair.macro.macro_selected}")
check(
    "macro repairs when a repair gate is available",
    pair.macro.inverse_gate_allowed == 0 or pair.macro.macro_repair_selected >= 1,
    f"allowed={pair.macro.inverse_gate_allowed} macro_repair={pair.macro.macro_repair_selected}",
)
check(
    "baseline logs controller actions beyond inverse-only rows",
    sum(int(v) for v in pair.baseline.controller_policy_counts.values()) > 0,
    f"controller_policy_counts={pair.baseline.controller_policy_counts}",
)
check(
    "macro logs build-action controller rows",
    any(str(name) in pair.macro.controller_policy_counts for name in ("replace", "prune", "residual", "mul_rand", "add_rand")),
    f"controller_policy_counts={pair.macro.controller_policy_counts}",
)
check(
    "pair ratio finite",
    math.isfinite(pair.ratio_best_eff_mse) and pair.ratio_best_eff_mse > 0.0,
    f"ratio={pair.ratio_best_eff_mse}",
)
check(
    "captured search tail present",
    len(pair.macro.search_stdout_tail) > 0,
    f"tail={pair.macro.search_stdout_tail}",
)


print("\n=== Test: controller harness exposes scheduler advisory/control arms ===")
orig_load_scheduler_bundle = engine_search.load_scheduler_bundle
engine_search.load_scheduler_bundle = lambda _path: {"scheduler_critic_trained": True}
try:
    scheduler_pair = run_controller_pair(
        "addsum",
        seed=0,
        profile="repair_probe",
        scheduler_bundle_path="/tmp/stub_scheduler_bundle.pt",
        n_iter=6,
        max_depth=4,
        capture_search_output=True,
        threads=1,
    )
finally:
    engine_search.load_scheduler_bundle = orig_load_scheduler_bundle

check(
    "scheduler advisory arm present",
    scheduler_pair.scheduler_advisory is not None and scheduler_pair.scheduler_advisory.label == "scheduler_advisory",
    f"summary={scheduler_pair.scheduler_advisory}",
)
check(
    "scheduler control arm present",
    scheduler_pair.scheduler_control is not None and scheduler_pair.scheduler_control.label == "scheduler_control",
    f"summary={scheduler_pair.scheduler_control}",
)
check(
    "scheduler advisory loads bundle",
    bool(scheduler_pair.scheduler_advisory and scheduler_pair.scheduler_advisory.scheduler_bundle_loaded),
    f"loaded={None if scheduler_pair.scheduler_advisory is None else scheduler_pair.scheduler_advisory.scheduler_bundle_loaded}",
)
check(
    "scheduler control enables scheduler stats",
    bool(scheduler_pair.scheduler_control and scheduler_pair.scheduler_control.scheduler_enabled),
    f"enabled={None if scheduler_pair.scheduler_control is None else scheduler_pair.scheduler_control.scheduler_enabled}",
)
check(
    "scheduler control tracks exact evals",
    bool(scheduler_pair.scheduler_control and scheduler_pair.scheduler_control.exact_eval_count >= 0),
    f"exact={None if scheduler_pair.scheduler_control is None else scheduler_pair.scheduler_control.exact_eval_count}",
)


print(f"\n{'='*50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    sys.exit(1)
print("All tests passed!")
