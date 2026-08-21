# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import random

from nestynet_sr.sr_search.factorized_search.controller import MacroController, MacroControllerState


def _build_state() -> MacroControllerState:
    return MacroControllerState(
        allowed_actions=("replace", "wrap_un"),
        repair_ready=False,
        gate_allowed=False,
    )


def test_confident_learned_policy_overrides_bandit_preference():
    state = _build_state()
    ctl = MacroController(
        ["replace", "wrap_un"],
        ucb_c=0.0,
        eps=0.0,
        build_bias=0.0,
        inverse_bonus=0.0,
        repair_bonus=0.0,
        learned_policy_weight=1.0,
    )
    for _ in range(8):
        ctl.update(state, "wrap_un", 1.0)

    decision = ctl.select_action(
        state,
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

    assert decision.action_name == "replace"
    assert decision.policy_source == "learned_primary_argmax"
    assert decision.bandit_scores["wrap_un"] > decision.bandit_scores["replace"]


def test_weak_learned_policy_falls_back_to_bandit():
    state = _build_state()
    ctl = MacroController(
        ["replace", "wrap_un"],
        ucb_c=0.0,
        eps=0.0,
        build_bias=0.0,
        inverse_bonus=0.0,
        repair_bonus=0.0,
        learned_policy_weight=1.0,
    )
    for _ in range(8):
        ctl.update(state, "wrap_un", 1.0)

    decision = ctl.select_action(
        state,
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

    assert decision.action_name == "wrap_un"
    assert decision.policy_source == "fallback_score_argmax"
