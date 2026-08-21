# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from nestynet_sr.sr_search.factorized_search.basis_state import ProposalContext
from nestynet_sr.sr_search.factorized_search.proposal_families.steering import (
    allocate_family_budgets,
    heuristic_family_priority_scores,
)


def test_heuristic_family_priority_scores_use_context_hints_and_residual_tokens():
    context = ProposalContext(
        residual_witness={"signal": "strong periodic oscillation with quotient tail"},
        diagnostics={"note": "exp growth remains"},
        family_hints={"log": 1.5},
    )

    scores = heuristic_family_priority_scores(
        families=["periodic", "exp", "log", "rational"],
        context=context,
    )

    assert scores["periodic"] >= 2.0
    assert scores["exp"] >= 2.0
    assert scores["log"] >= 1.5
    assert scores["rational"] >= 2.0


def test_allocate_family_budgets_preserves_legacy_order_without_signal():
    plan = allocate_family_budgets(
        families=["periodic", "exp", "log"],
        max_scaffolds=6,
        anchors_per_family=3,
        context=ProposalContext(),
    )

    assert plan["steered"] is False
    assert [entry["family"] for entry in plan["entries"]] == ["periodic", "exp", "log"]
    assert [entry["max_scaffolds"] for entry in plan["entries"]] == [2, 2, 2]


def test_allocate_family_budgets_reorders_and_splits_budget_with_signal():
    context = ProposalContext(
        residual_witness="logarithmic ratio structure",
        family_hints={"rational": 3.0},
    )
    plan = allocate_family_budgets(
        families=["periodic", "exp", "log", "rational"],
        max_scaffolds=5,
        anchors_per_family=2,
        context=context,
    )

    assert plan["steered"] is True
    assert plan["entries"][0]["family"] == "rational"
    assert sum(int(entry["max_scaffolds"]) for entry in plan["entries"]) == 5
    assert all(int(entry["anchors_per_family"]) == 2 for entry in plan["entries"])


def test_heuristic_family_priority_scores_cover_power_and_quadratic_tokens():
    context = ProposalContext(
        residual_witness={
            "tags": ["radial", "norm", "sqrt", "inverse_power"],
            "family_probe_scores": {"power": 0.8, "quadratic": 0.7},
        },
        diagnostics={"note": "quadratic radial residual with sqrt tail"},
        family_hints={"power": 1.0},
    )

    scores = heuristic_family_priority_scores(
        families=["periodic", "power", "quadratic"],
        context=context,
    )

    assert scores["power"] >= 3.0
    assert scores["quadratic"] >= 2.0
    assert scores["power"] > scores["periodic"]
