# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

from nestynet_sr.sr_search.factorized_search.subproblem_spec import (
    FamilyEvidence,
    FAMILY_EVIDENCE_HARD_CONSTRAINTS_SCHEMA_NAME,
    SolverProposal,
    SubproblemSpec,
    WitnessBundle,
    deserialize_family_evidence,
    deserialize_solver_proposal,
    deserialize_subproblem_spec,
    serialize_family_evidence,
    serialize_solver_proposal,
    wrap_subproblem_spec_payload,
)


def test_subproblem_spec_round_trip_preserves_witness_and_metadata():
    witness = WitnessBundle(
        x_fit=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        t_fit=torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        x_probe=torch.tensor([[5.0], [6.0]], dtype=torch.float64),
        t_probe=torch.tensor([[7.0], [8.0]], dtype=torch.float64),
        masks={"w_fit": torch.tensor([[1.0], [0.5]], dtype=torch.float64)},
        diagnostics={"confidence": 0.8, "trace": ("root", "child")},
    )
    spec = SubproblemSpec(
        problem_id="toy-problem",
        problem_kind="local_problem",
        parent_expr=("add", ("var", 0), ("const", 1.0)),
        path=(1,),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=("L",),
        continuation_frames=({"wrap_kind": "unary", "op": "exp", "slot": 0, "anchor_node": None},),
        wrappers_left=2,
        recursion_level=1,
        active_vars=(0,),
        witness=witness,
        metadata={
            "hole_sub": ("var", 0),
            "trace": ("root", "child"),
            "teacher_spec": {"source": "numeric_local_quadratic", "requested_source": "oracle"},
        },
    )

    payload = wrap_subproblem_spec_payload(spec, extra_payload={"legacy_flag": True})
    decoded = deserialize_subproblem_spec(payload)

    assert payload["schema_name"] == "factorized_search.subproblem_spec"
    assert payload["schema_version"] == 1
    assert decoded is not None
    assert decoded.problem_id == "toy-problem"
    assert decoded.problem_kind == "local_problem"
    assert decoded.path == (1,)
    assert decoded.continuation_frames[0]["op"] == "exp"
    assert decoded.metadata["hole_sub"] == ("var", 0)
    assert decoded.metadata["teacher_spec"]["source"] == "numeric_local_quadratic"
    assert decoded.witness is not None
    assert torch.equal(decoded.witness.x_fit, witness.x_fit)
    assert torch.equal(decoded.witness.masks["w_fit"], witness.masks["w_fit"])


def test_family_evidence_and_solver_proposal_round_trip():
    evidence = FamilyEvidence(
        family_scores={"exp": 0.9, "power": 0.2},
        hard_constraints={"target_dim": ("L", "T")},
        seed_nodes=(("var", 0), ("exp", ("var", 0))),
        metadata={"gate": "demo"},
    )
    proposal = SolverProposal(
        expr_ast=("exp", ("var", 0)),
        mapping={"kind": "affine", "coeffs": [0.0, 1.0]},
        source="outer_family:exp",
        family="exp",
        preview_loss=0.125,
        global_probe_mse=0.25,
        metadata={"rank": 1},
    )

    evidence_payload = serialize_family_evidence(evidence)
    proposal_payload = serialize_solver_proposal(proposal)
    evidence_decoded = deserialize_family_evidence(evidence_payload)
    proposal_decoded = deserialize_solver_proposal(proposal_payload)

    assert evidence_decoded is not None
    assert evidence_decoded.family_scores["exp"] == 0.9
    assert evidence_decoded.hard_constraints["schema_name"] == FAMILY_EVIDENCE_HARD_CONSTRAINTS_SCHEMA_NAME
    assert evidence_decoded.seed_nodes[1] == ("exp", ("var", 0))
    assert evidence_decoded.hard_constraints["context"]["target_dim"] == ("L", "T")
    assert proposal_decoded is not None
    assert proposal_decoded.family == "exp"
    assert proposal_decoded.preview_loss == 0.125
    assert proposal_decoded.mapping["kind"] == "affine"
