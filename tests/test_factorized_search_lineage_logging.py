# SPDX-License-Identifier: MPL-2.0

import math

import pytest

from nestynet_sr.sr_search.factorized_search import explorer as explorer_mod
from nestynet_sr.sr_search.factorized_search.policy.guidance import _annotate_inverse_experiment_lineage


def test_explorer_reexports_lineage_guidance_helper():
    assert explorer_mod._annotate_inverse_experiment_lineage is _annotate_inverse_experiment_lineage


def test_annotate_inverse_experiment_lineage_propagates_best_descendant_path_and_reward():
    rows = [
        {
            "macro_action": "replace",
            "controller_action_path": [1],
            "actor_critic_reward_novelty_bonus": 0.2,
            "actor_critic_reward_best_bonus": 0.1,
            "actor_critic_reward_time_penalty": 0.05,
        },
        {
            "macro_action": "add_rand",
            "selected_path": [2],
            "selected_target_mode": "affine",
        },
        {
            "macro_action": "mul_rand",
            "controller_action_path": [3, 1],
            "selected_target_mode": "identity",
        },
    ]
    events = [
        {
            "row_index": 0,
            "parent_key_raw": ("root",),
            "child_key_raw": ("b1",),
            "parent_eff_mse": 20.0,
            "child_eff_mse": 10.0,
            "child_raw_mse": 11.0,
        },
        {
            "row_index": 1,
            "parent_key_raw": ("b1",),
            "child_key_raw": ("b2",),
            "parent_eff_mse": 10.0,
            "child_eff_mse": 5.0,
            "child_raw_mse": 6.0,
        },
        {
            "row_index": 2,
            "parent_key_raw": ("b2",),
            "child_key_raw": ("b3",),
            "parent_eff_mse": 5.0,
            "child_eff_mse": 2.0,
            "child_raw_mse": 3.0,
        },
    ]

    _annotate_inverse_experiment_lineage(rows, events, horizon=3, eps=1.0e-30)

    assert rows[0]["lineage_parent_key"] == ["root"]
    assert rows[0]["lineage_child_key"] == ["b1"]
    assert rows[0]["best_descendant_hops"] == 3
    assert rows[0]["best_descendant_eff_mse"] == pytest.approx(2.0)
    assert rows[0]["best_descendant_raw_mse"] == pytest.approx(3.0)
    assert rows[0]["best_descendant_path"] == [3, 1]
    assert rows[0]["best_descendant_selected_path"] == []
    assert rows[0]["best_descendant_target_mode"] == "identity"
    assert rows[0]["best_descendant_macro_action"] == "mul_rand"
    assert rows[0]["actor_critic_descendant_log_gain"] == pytest.approx(math.log(20.0) - math.log(2.0))
    assert rows[0]["actor_critic_descendant_reward"] == pytest.approx(math.log(20.0) - math.log(2.0) + 0.2 + 0.1 - 0.05)

    assert rows[1]["best_descendant_hops"] == 2
    assert rows[1]["best_descendant_eff_mse"] == pytest.approx(2.0)
    assert rows[2]["best_descendant_hops"] == 1
    assert rows[2]["best_descendant_path"] == [3, 1]


def test_annotate_inverse_experiment_lineage_horizon_limits_descendant_credit():
    rows = [
        {"controller_action_path": [1]},
        {"controller_action_path": [2]},
        {"controller_action_path": [3]},
    ]
    events = [
        {
            "row_index": 0,
            "parent_key_raw": ("root",),
            "child_key_raw": ("b1",),
            "parent_eff_mse": 20.0,
            "child_eff_mse": 10.0,
            "child_raw_mse": 10.5,
        },
        {
            "row_index": 1,
            "parent_key_raw": ("b1",),
            "child_key_raw": ("b2",),
            "parent_eff_mse": 10.0,
            "child_eff_mse": 5.0,
            "child_raw_mse": 5.5,
        },
        {
            "row_index": 2,
            "parent_key_raw": ("b2",),
            "child_key_raw": ("b3",),
            "parent_eff_mse": 5.0,
            "child_eff_mse": 2.0,
            "child_raw_mse": 2.5,
        },
    ]

    _annotate_inverse_experiment_lineage(rows, events, horizon=2, eps=1.0e-30)

    assert rows[0]["best_descendant_hops"] == 2
    assert rows[0]["best_descendant_eff_mse"] == pytest.approx(5.0)
    assert rows[0]["best_descendant_path"] == [2]
