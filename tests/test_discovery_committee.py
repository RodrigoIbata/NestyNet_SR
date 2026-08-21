# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from nestynet_sr.discovery.committee import build_committee_state, canonicalize_candidate_law
from nestynet_sr.sr_core.bridges import ConstNode, MulNode, Var


def test_canonicalize_candidate_law_deduplicates_scale_equivalent_tuple_asts():
    expr_a = ("mul", ("const", 2.0), ("var", 0))
    expr_b = ("mul", ("const", 5.0), ("var", 0))
    info_a = canonicalize_candidate_law(expr_a)
    info_b = canonicalize_candidate_law(expr_b)

    assert info_a["canonical_key"] == info_b["canonical_key"]


def test_build_committee_state_keeps_best_validation_member_per_structure():
    state = build_committee_state(
        [
            {
                "member_id": "m0",
                "expr": ("mul", ("const", 2.0), ("var", 0)),
                "validation_error": 0.20,
            },
            {
                "member_id": "m1",
                "expr": ("mul", ("const", 5.0), ("var", 0)),
                "validation_error": 0.05,
            },
            {
                "member_id": "m2",
                "expr": ("add", ("var", 0), ("const", 1.0)),
                "validation_error": 0.10,
            },
        ]
    )

    assert len(state.members) == 2
    assert [member.member_id for member in state.members] == ["m1", "m2"]
    assert abs(sum(member.committee_weight for member in state.members) - 1.0) < 1.0e-9
    assert "m0" in state.discarded_member_ids


def test_canonicalize_candidate_law_supports_bridge_nodes():
    expr_a = MulNode(ConstNode(2.0), Var(0))
    expr_b = MulNode(ConstNode(7.0), Var(0))

    info_a = canonicalize_candidate_law(expr_a)
    info_b = canonicalize_candidate_law(expr_b)

    assert info_a["canonical_key"] == info_b["canonical_key"]
