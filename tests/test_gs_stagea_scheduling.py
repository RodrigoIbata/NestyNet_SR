# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Protected Stage-A scheduling for GS carrier proposals."""

from nestynet_sr.sr_core.bridges import AddNode, MulNode, Var
from nestynet_sr.sr_search.search import (
    _compound_proposal_sort_key,
    _shortlist_compound_proposals_with_pair_backup,
    _stageA_schedule_gs_compound_lanes,
)


def _ordinary(pattern, ast, confidence, **metadata):
    meta = {"kind": "monomial", "source": "ordinary_detector"}
    meta.update(metadata)
    return (pattern, ast, confidence, None, meta)


def _gs(pattern, ast, confidence, **metadata):
    meta = {
        "kind": "gs_promoted_reduction",
        "source": "generalized_symmetry",
        "candidate_role": "inner_coordinate",
        "carrier_certified": True,
    }
    meta.update(metadata)
    return (pattern, ast, confidence, None, meta)


def _ordinary_baseline(proposals, cap):
    ranked = list(proposals)
    ranked.sort(key=_compound_proposal_sort_key, reverse=True)
    return _shortlist_compound_proposals_with_pair_backup(ranked, cap)


def test_gs_never_changes_the_ordinary_shortlist_or_budget():
    ordinary = [
        _ordinary((1, 1, 1), MulNode(MulNode(Var(0), Var(1)), Var(2)), 0.91),
        _ordinary((1, 1, 0), MulNode(Var(0), Var(1)), 0.88),
        _ordinary((0, 1, 1), MulNode(Var(1), Var(2)), 0.82),
    ]
    gs = [
        _gs((1, 1, 1), AddNode(MulNode(Var(0), Var(1)), Var(2)), 0.999),
        _gs((1, 1, 0), AddNode(Var(0), Var(1)), 0.97),
    ]

    decisive, scheduled_ordinary, fallback = _stageA_schedule_gs_compound_lanes(
        ordinary + gs,
        max_ordinary_proposals=1,
        max_gs_proposals=2,
        decisive_min_confidence=0.995,
        decisive_max_trials=1,
    )

    assert scheduled_ordinary == _ordinary_baseline(ordinary, 1)
    assert len(decisive) == 1
    assert decisive[0][4]["gs_stagea_lane"] == "decisive"
    assert len(fallback) == 1
    assert fallback[0][4]["gs_stagea_lane"] == "fallback"


def test_decisive_lane_requires_a_certified_plain_full_support_carrier():
    candidates = [
        _gs((1, 1, 1), AddNode(AddNode(Var(0), Var(1)), Var(2)), 0.999),
        _gs((1, 1, 0), AddNode(Var(0), Var(1)), 1.0),
        _gs(
            (1, 1, 1),
            MulNode(MulNode(Var(0), Var(1)), Var(2)),
            1.0,
            carrier_certified=False,
        ),
        _gs(
            (1, 1, 1),
            MulNode(AddNode(Var(0), Var(1)), Var(2)),
            1.0,
            extra_input_asts=(Var(0),),
        ),
        _gs(
            (1, 1, 1),
            AddNode(MulNode(Var(0), Var(2)), Var(1)),
            1.0,
            gs_output_beta=1.0,
        ),
    ]

    decisive, ordinary, fallback = _stageA_schedule_gs_compound_lanes(
        candidates,
        max_ordinary_proposals=3,
        max_gs_proposals=5,
        decisive_min_confidence=0.995,
        decisive_max_trials=1,
    )

    assert ordinary == []
    assert len(decisive) == 1
    assert decisive[0][1] == candidates[0][1]
    assert len(fallback) == 4
    assert all(proposal[4]["gs_stagea_lane"] == "fallback" for proposal in fallback)


def test_gs_budget_is_a_hard_total_cap_including_the_decisive_trial():
    candidates = [
        _gs(
            (1, 1, 1),
            AddNode(MulNode(Var(0), Var(1)), MulNode(Var(2), Var(i))),
            1.0 - 0.001 * i,
        )
        for i in range(3, 8)
    ]

    decisive, ordinary, fallback = _stageA_schedule_gs_compound_lanes(
        candidates,
        max_ordinary_proposals=3,
        max_gs_proposals=3,
        decisive_min_confidence=0.995,
        decisive_max_trials=1,
    )

    assert ordinary == []
    assert len(decisive) == 1
    assert len(fallback) == 2
    assert len(decisive) + len(fallback) == 3


def test_no_gs_is_exactly_the_legacy_ordinary_schedule():
    ordinary = [
        _ordinary((1, 1, 1), MulNode(MulNode(Var(0), Var(1)), Var(2)), 0.90),
        _ordinary((0, 1, 1), MulNode(Var(1), Var(2)), 0.80),
    ]
    decisive, scheduled_ordinary, fallback = _stageA_schedule_gs_compound_lanes(
        ordinary,
        max_ordinary_proposals=1,
        max_gs_proposals=4,
        decisive_min_confidence=0.995,
        decisive_max_trials=1,
    )

    assert decisive == []
    assert fallback == []
    assert scheduled_ordinary == _ordinary_baseline(ordinary, 1)
