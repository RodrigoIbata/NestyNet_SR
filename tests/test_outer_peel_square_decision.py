# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import numpy as np
import pytest
import torch

from nestynet_sr.sr_search.features import _build_poly_design_matrix, _cross_hess_rel
from nestynet_sr.sr_search.outer_peel import (
    OuterPeelProposal,
    decide_square_preference,
    square_family_evidence,
)


def test_square_decision_keeps_legacy_gate_when_strong():
    proposal = OuterPeelProposal(
        name="square",
        score=0.01,
        improvement=64.0,
        details={"axis_stats": []},
    )
    y = np.abs(np.random.default_rng(0).normal(size=256))
    decision = decide_square_preference(proposal, y, gain_min=30.0, score_max=0.05)
    assert decision.prefer is True
    assert decision.diagnostics.get("reason") == "prefer_square_legacy"


def test_square_decision_accepts_multi_axis_nontrig_case():
    proposal = OuterPeelProposal(
        name="square",
        score=0.049,
        improvement=17.7,
        details={
            "axis_stats": [
                {"axis": 0, "spread_identity": 0.86, "spread_square": 0.041, "improvement": 21.0},
                {"axis": 1, "spread_identity": 0.82, "spread_square": 0.046, "improvement": 18.3},
                {"axis": 2, "spread_identity": 1.10, "spread_square": 0.72, "improvement": 1.4},
                {"axis": 3, "spread_identity": 1.25, "spread_square": 0.84, "improvement": 1.2},
            ]
        },
    )
    y = np.abs(np.random.default_rng(1).normal(size=512))
    decision = decide_square_preference(
        proposal,
        y,
        gain_min=30.0,
        score_max=0.05,
        min_good_axes=2,
    )
    assert decision.prefer is True
    assert decision.diagnostics.get("reason") == "prefer_square_multi_axis"
    assert decision.diagnostics.get("num_good_axes") == 2
    assert set(decision.diagnostics.get("ignored_axes", [])) >= {2, 3}


def test_square_decision_rejects_when_not_enough_good_axes():
    proposal = OuterPeelProposal(
        name="square",
        score=0.08,
        improvement=12.0,
        details={
            "axis_stats": [
                {"axis": 0, "spread_identity": 0.9, "spread_square": 0.03, "improvement": 14.0},
                {"axis": 1, "spread_identity": 1.2, "spread_square": 0.31, "improvement": 1.1},
            ]
        },
    )
    y = np.abs(np.random.default_rng(2).normal(size=256))
    decision = decide_square_preference(
        proposal,
        y,
        gain_min=30.0,
        score_max=0.05,
        min_good_axes=2,
        auto_trig_axis_reject_factor=0.0,
    )
    assert decision.prefer is False
    assert decision.diagnostics.get("reason") == "insufficient_good_axes"


def test_square_family_evidence_exports_family_record():
    proposal = OuterPeelProposal(
        name="square",
        score=0.049,
        improvement=17.7,
        details={
            "axis_stats": [
                {"axis": 0, "spread_identity": 0.86, "spread_square": 0.041, "improvement": 21.0},
                {"axis": 1, "spread_identity": 0.82, "spread_square": 0.046, "improvement": 18.3},
                {"axis": 2, "spread_identity": 1.10, "spread_square": 0.72, "improvement": 1.4},
                {"axis": 3, "spread_identity": 1.25, "spread_square": 0.84, "improvement": 1.2},
            ]
        },
    )
    y = np.abs(np.random.default_rng(3).normal(size=512))

    evidence = square_family_evidence(
        proposal,
        y,
        gain_min=30.0,
        score_max=0.05,
        min_good_axes=2,
    )

    assert evidence.family_scores["square"] == proposal.improvement
    assert evidence.hard_constraints["prefer"] is True
    assert evidence.hard_constraints["num_good_axes"] == 2
    assert evidence.metadata["status"] == "triggered"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_outer_peel_feature_helpers_keep_cuda_device():
    x = torch.randn(32, 3, dtype=torch.float64, device="cuda")
    phi = _build_poly_design_matrix(x, degree=2)
    assert phi.device.type == "cuda"

    h = torch.randn(32, 3, 3, dtype=torch.float64, device="cuda")
    rel = _cross_hess_rel(h)
    assert isinstance(rel, float)
