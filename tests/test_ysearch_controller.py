# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_search.ysearch_controller import (
    StageAStateKey,
    YSearchControllerConfig,
    YSearchResult,
    YSearchState,
    YSearchTrial,
    run_depth1_ysearch,
    run_depth1_ysearch_beam,
    run_ysearch_beam,
    run_ysearch_beam_with_split_recursion,
)


def test_depth1_accepts_strong_trigger_without_loss_improvement():
    cfg = YSearchControllerConfig(max_depth=1, expand_k=2, confirm_improve_ratio=0.3)
    state = YSearchState(y_stack=tuple())

    payloads = {
        "a": {"val_loss_base": 10.0, "split_success": False},
        "b": {"val_loss_base": 10.0, "split_success": True},
    }

    res = run_depth1_ysearch(
        parent_state=state,
        candidate_names=["a", "b"],
        evaluate_candidate=lambda n: payloads[n],
        parent_val_loss_base=1.0,
        cfg=cfg,
        strong_structure_trigger_fn=lambda p: bool(p.get("split_success", False)),
    )

    assert res.best_trial is not None
    assert res.best_trial.name == "b"
    assert res.best_trial.accept_branch is True


def test_depth1_selects_best_accepted_loss():
    cfg = YSearchControllerConfig(max_depth=1, expand_k=3, confirm_improve_ratio=0.8)
    state = YSearchState(y_stack=tuple())

    payloads = {
        "a": {"val_loss_base": 0.9, "split_success": False},
        "b": {"val_loss_base": 0.7, "split_success": False},
        "c": {"val_loss_base": 0.6, "split_success": False},
    }

    res = run_depth1_ysearch(
        parent_state=state,
        candidate_names=["a", "b", "c"],
        evaluate_candidate=lambda n: payloads[n],
        parent_val_loss_base=1.0,
        cfg=cfg,
    )

    assert res.best_trial is not None
    assert res.best_trial.name == "c"
    assert len(res.accepted_trials) == 2  # b and c (<= 0.8 * parent)


def test_depth1_beam_pruning_keeps_top_k():
    cfg = YSearchControllerConfig(max_depth=1, beam=2, expand_k=5, confirm_improve_ratio=2.0)
    state = YSearchState(y_stack=tuple())
    payloads = {
        "a": {"val_loss_base": 0.9, "split_success": False},
        "b": {"val_loss_base": 0.7, "split_success": False},
        "c": {"val_loss_base": 0.6, "split_success": True},
        "d": {"val_loss_base": 0.8, "split_success": False},
    }
    res = run_depth1_ysearch_beam(
        parent_state=state,
        candidate_names=["a", "b", "c", "d"],
        evaluate_candidate=lambda n: payloads[n],
        parent_val_loss_base=1.0,
        cfg=cfg,
        strong_structure_trigger_fn=lambda p: bool(p.get("split_success", False)),
    )
    assert len(res.frontier_trials) == 2
    assert [t.name for t in res.frontier_trials] == ["c", "b"]


def test_depth1_cache_reuses_stagea_payload():
    cfg = YSearchControllerConfig(max_depth=1, beam=3, expand_k=2, confirm_improve_ratio=2.0)
    state = YSearchState(y_stack=tuple())
    calls = {"n": 0}

    def _eval(name):
        calls["n"] += 1
        return {"val_loss_base": 0.5 if name == "a" else 0.6, "split_success": False}

    cache = {}

    def _key(name):
        return StageAStateKey(
            y_stack_sig=(name,),
            data_sig=("d",),
            model_sig=("m",),
            train_cfg_sig=("t",),
            seed=0,
            fast=False,
        )

    run_depth1_ysearch_beam(
        parent_state=state,
        candidate_names=["a", "b"],
        evaluate_candidate=_eval,
        parent_val_loss_base=1.0,
        cfg=cfg,
        stagea_cache=cache,
        make_key_fn=_key,
    )
    run_depth1_ysearch_beam(
        parent_state=state,
        candidate_names=["a", "b"],
        evaluate_candidate=_eval,
        parent_val_loss_base=1.0,
        cfg=cfg,
        stagea_cache=cache,
        make_key_fn=_key,
    )

    assert calls["n"] == 2


def test_multidepth_beam_explores_stack():
    cfg = YSearchControllerConfig(max_depth=2, beam=2, expand_k=2, confirm_improve_ratio=1.0)
    state = YSearchState(y_stack=tuple())

    def _eval(stack):
        # Better loss for deeper stack ("a","b")
        if stack == ("a",):
            return {"val_loss_base": 0.95, "split_success": False}
        if stack == ("b",):
            return {"val_loss_base": 0.92, "split_success": False}
        if stack == ("a", "a"):
            return {"val_loss_base": 0.80, "split_success": False}
        if stack == ("a", "b"):
            return {"val_loss_base": 0.30, "split_success": True}
        if stack == ("b", "a"):
            return {"val_loss_base": 0.50, "split_success": False}
        if stack == ("b", "b"):
            return {"val_loss_base": 0.70, "split_success": False}
        return None

    res = run_ysearch_beam(
        parent_state=state,
        candidate_names=["a", "b"],
        evaluate_state=_eval,
        parent_val_loss_base=1.0,
        cfg=cfg,
        strong_structure_trigger_fn=lambda p: bool(p.get("split_success", False)),
    )

    assert res.best_trial is not None
    assert res.best_trial.state.y_stack == ("a", "b")


def test_multidepth_beam_respects_state_eval_budget():
    cfg = YSearchControllerConfig(
        max_depth=3,
        beam=3,
        expand_k=3,
        confirm_improve_ratio=1.0,
        max_state_evals=2,
    )
    state = YSearchState(y_stack=tuple())
    calls = {"n": 0}

    def _eval(stack):
        calls["n"] += 1
        return {"val_loss_base": 1.0 - 0.01 * len(stack), "split_success": False}

    res = run_ysearch_beam(
        parent_state=state,
        candidate_names=["a", "b", "c"],
        evaluate_state=_eval,
        parent_val_loss_base=1.0,
        cfg=cfg,
    )

    assert calls["n"] == 2
    assert res.state_evals == 2
    assert res.budget_exhausted is True


def test_split_recursion_can_improve_best_under_budget():
    cfg = YSearchControllerConfig(
        max_depth=1,
        beam=2,
        expand_k=2,
        confirm_improve_ratio=1.0,
        max_recursive_branches=1,
        max_split_plans_per_state=1,
    )
    state = YSearchState(y_stack=tuple())

    payloads = {
        ("a",): {
            "val_loss_base": 0.8,
            "split_success": True,
            "split_plans": [{"kind": "add"}],
        },
        ("b",): {
            "val_loss_base": 0.7,
            "split_success": False,
            "split_plans": [],
        },
    }

    def _eval(stack):
        return payloads.get(tuple(stack), None)

    def _split_plans(payload):
        return list(payload.get("split_plans", []))

    def _recurse(parent_state, plan):
        if parent_state.y_stack != ("a",):
            return None
        if plan.get("kind") != "add":
            return None
        trial = YSearchTrial(
            name="rec",
            state=YSearchState(y_stack=("a", "rec")),
            val_loss_base=0.2,
            split_success=True,
            strong_structure_trigger=True,
            accept_branch=True,
            payload={"val_loss_base": 0.2, "split_success": True},
        )
        return YSearchResult(
            best_trial=trial,
            accepted_trials=[trial],
            all_trials=[trial],
            frontier_trials=[trial],
            state_evals=1,
        )

    res = run_ysearch_beam_with_split_recursion(
        parent_state=state,
        candidate_names=["a", "b"],
        evaluate_state=_eval,
        parent_val_loss_base=1.0,
        cfg=cfg,
        split_plans_fn=_split_plans,
        recurse_split_fn=_recurse,
    )

    assert res.best_trial is not None
    assert res.best_trial.state.y_stack == ("a", "rec")
    assert res.recursive_calls == 1


def test_split_recursion_respects_global_state_eval_budget():
    cfg = YSearchControllerConfig(
        max_depth=1,
        beam=2,
        expand_k=2,
        confirm_improve_ratio=1.0,
        max_state_evals=2,
        max_recursive_branches=2,
        max_split_plans_per_state=1,
    )
    state = YSearchState(y_stack=tuple())
    recurse_calls = {"n": 0}

    def _eval(stack):
        return {
            "val_loss_base": 0.8 if stack == ("a",) else 0.7,
            "split_success": True,
            "split_plans": [{"kind": "add"}],
        }

    def _split_plans(payload):
        return list(payload.get("split_plans", []))

    def _recurse(parent_state, plan):
        recurse_calls["n"] += 1
        trial = YSearchTrial(
            name="rec",
            state=YSearchState(y_stack=tuple(parent_state.y_stack) + ("rec",)),
            val_loss_base=0.2,
            split_success=True,
            strong_structure_trigger=True,
            accept_branch=True,
            payload={"val_loss_base": 0.2, "split_success": True},
        )
        return YSearchResult(
            best_trial=trial,
            accepted_trials=[trial],
            all_trials=[trial],
            frontier_trials=[trial],
            state_evals=1,
        )

    res = run_ysearch_beam_with_split_recursion(
        parent_state=state,
        candidate_names=["a", "b"],
        evaluate_state=_eval,
        parent_val_loss_base=1.0,
        cfg=cfg,
        split_plans_fn=_split_plans,
        recurse_split_fn=_recurse,
    )

    # Primary search consumes the full budget (2 evals), so recursion is skipped.
    assert res.state_evals == 2
    assert res.budget_exhausted is True
    assert recurse_calls["n"] == 0
