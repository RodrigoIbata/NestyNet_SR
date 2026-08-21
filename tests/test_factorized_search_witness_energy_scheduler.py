# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import pytest

from nestynet_sr.sr_search.factorized_search.scheduler import (
    build_plan_candidates,
    choose_plan,
)


def test_choose_plan_exposes_witness_energy_components(monkeypatch):
    candidates = build_plan_candidates(
        parent_key="parent_0",
        build_opportunity_rows=[
            {
                "opportunity_id": "build_0",
                "action": "replace",
                "witness_value_loss": 0.6,
                "witness_grad_loss": 0.2,
                "witness_energy_total": 0.8,
                "witness_energy_delta_estimate": 0.3,
            }
        ],
        exact_budget_ladder=(1,),
    )

    def fake_predict_scheduler_plan_slate(_bundle, rows, *, acquisition_threshold=0.25):
        assert len(rows) == 1
        row = dict(rows[0])
        row.update({
            "break_prob_0p25_at_budget_1": 0.9,
            "acquisition_estimate_at_budget_1": 0.4,
            "acquisition_sigma_at_budget_1": 0.05,
        })
        return {
            "trained": True,
            "rows": [row],
        }

    monkeypatch.setattr(
        "nestynet_sr.sr_search.factorized_search.scheduler.predict_scheduler_plan_slate",
        fake_predict_scheduler_plan_slate,
    )

    decision = choose_plan(
        {"scheduler_critic_trained": True},
        candidates,
        advisory_only=False,
        acquisition_threshold=0.25,
        uncertainty_bonus=0.05,
    )

    row = decision.rows[0]
    witness = row["plan_prediction_components"]["witness_energy"]
    assert row["plan_witness_energy_total"] == pytest.approx(0.8)
    assert row["plan_witness_energy_delta_estimate"] == pytest.approx(0.3)
    assert witness["value_loss"] == pytest.approx(0.6)
    assert witness["grad_loss"] == pytest.approx(0.2)
    assert witness["total"] == pytest.approx(0.8)
    assert witness["delta_estimate"] == pytest.approx(0.3)
