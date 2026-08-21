# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import json

import numpy as np
import pytest

from nestynet_sr.stat_selection import (
    AuditDesign,
    CandidateArchive,
    CandidateSpec,
    ComplexityVector,
    LossAudit,
    UnitLossRecord,
    candidate_id_for,
)


def _design(unit_kind: str = "experiment") -> AuditDesign:
    return AuditDesign(
        loss_name="mean_squared_error",
        unit_kind=unit_kind,
        fit_protocol="external_holdout",
        evaluation_domain={"split": "audit"},
        sampling_assumptions=("Units are exchangeable within the audit split.",),
    )


def _archive() -> CandidateArchive:
    archive = CandidateArchive(archive_label="unit-test", metadata={"grammar": "tiny"})
    archive.add_structure(
        "x0",
        ComplexityVector.from_mapping({"ast_length": 1, "free_parameters": 0}),
        grammar_version="g1",
        provenance=[{"engine": "stageB", "seed": 1}],
    )
    archive.add_structure(
        "a*x0",
        ComplexityVector.from_mapping({"ast_length": 3, "free_parameters": 1}),
        grammar_version="g1",
        refit_recipe={"kind": "linear_least_squares", "parameters": ["a"]},
        provenance=[{"engine": "polisher", "seed": 1}],
    )
    return archive


def test_complexity_vector_uses_named_pareto_partial_order():
    simple = ComplexityVector.from_mapping({"ast_length": 3, "free_parameters": 1})
    complex_model = ComplexityVector.from_mapping({"free_parameters": 2, "ast_length": 5})
    tradeoff = ComplexityVector.from_mapping({"ast_length": 2, "free_parameters": 3})

    assert simple.names == ("ast_length", "free_parameters")
    assert simple.no_worse_than(complex_model)
    assert simple.strictly_better_than(complex_model)
    assert not simple.no_worse_than(tradeoff)
    assert not tradeoff.no_worse_than(simple)


def test_candidate_archive_fingerprint_is_order_independent_and_freezes(tmp_path):
    first = _archive().freeze()

    second = CandidateArchive(archive_label="unit-test", metadata={"grammar": "tiny"})
    for candidate in reversed(first.candidates):
        second.add(CandidateSpec.from_dict(candidate.to_dict()))
    second.freeze()

    assert first.fingerprint == second.fingerprint
    with pytest.raises(RuntimeError, match="frozen"):
        first.add_structure("x1", ComplexityVector.scalar(1.0))

    path = first.write_json(tmp_path / "archive.json")
    restored = CandidateArchive.read_json(path)
    assert restored.fingerprint == first.fingerprint
    assert any(candidate.refit_recipe for candidate in restored.candidates)
    assert json.loads(path.read_text())["frozen"] is True


def test_duplicate_candidate_merges_provenance_but_rejects_core_collision():
    archive = CandidateArchive()
    complexity = ComplexityVector.scalar(2.0)
    candidate = CandidateSpec.from_structure(
        "x0+x1",
        complexity,
        provenance=[{"engine": "A"}],
    )
    archive.add(candidate)
    archive.add(
        CandidateSpec.from_structure(
            "x0+x1",
            complexity,
            provenance=[{"engine": "B"}],
        )
    )
    assert len(archive) == 1
    assert len(archive[candidate.candidate_id].provenance) == 2

    with pytest.raises(ValueError, match="collision"):
        archive.add(
            CandidateSpec(
                candidate_id=candidate.candidate_id,
                canonical_structure="x0-x1",
                complexity=complexity,
            )
        )


def test_candidate_id_depends_on_structure_and_grammar():
    a = candidate_id_for("x0+x1", grammar_version="g1")
    b = candidate_id_for("x0+x1", grammar_version="g2")
    c = candidate_id_for("x0-x1", grammar_version="g1")
    assert len(a) == 24
    assert len({a, b, c}) == 3


def test_loss_audit_requires_common_domain_or_declared_failure_penalty():
    archive = _archive().freeze()
    ids = archive.candidate_ids
    losses = np.array([[0.1, 0.2], [0.2, np.nan], [0.3, 0.4]])

    with pytest.raises(ValueError, match="failures/non-finite"):
        LossAudit.from_matrix(
            candidate_ids=ids,
            unit_ids=("u0", "u1", "u2"),
            design=_design(),
            losses=losses,
            archive=archive,
        )

    audit = LossAudit.from_matrix(
        candidate_ids=ids,
        unit_ids=("u0", "u1", "u2"),
        design=_design(),
        losses=losses,
        archive=archive,
        nonfinite="penalize",
        failure_loss=10.0,
    )
    failed_col = int(np.argwhere(audit.failure_mask)[0, 1])
    assert audit.losses[1, failed_col] == 10.0
    assert audit.failure_counts()[ids[failed_col]] == 1
    assert audit.archive_fingerprint == archive.fingerprint


def test_loss_audit_from_records_rejects_candidate_specific_omission():
    archive = _archive().freeze()
    records = [
        UnitLossRecord("u0", {archive.candidate_ids[0]: 0.1}),
        UnitLossRecord("u1", {archive.candidate_ids[0]: 0.2}),
    ]
    with pytest.raises(ValueError, match="frozen candidate set"):
        LossAudit.from_records(records, design=_design(), archive=archive)


def test_paired_standard_error_uses_loss_differences_not_marginal_errors():
    archive = CandidateArchive()
    archive.add_structure("a", ComplexityVector.scalar(1.0), candidate_id="a")
    archive.add_structure("b", ComplexityVector.scalar(1.0), candidate_id="b")
    archive.freeze()

    shared = np.array([10.0, -7.0, 4.0, 12.0, -3.0, 1.0])
    # The candidates share a large common fluctuation, but their difference is small.
    losses = np.column_stack((shared + 0.1, shared + np.array([0.2, 0.1, 0.2, 0.1, 0.2, 0.1])))
    audit = LossAudit.from_matrix(
        candidate_ids=("a", "b"),
        unit_ids=tuple(f"u{i}" for i in range(len(shared))),
        design=_design(),
        losses=losses,
        archive=archive,
    )

    _, paired_se = audit.paired_difference("b", "a")
    assert paired_se < 0.03
    assert np.min(audit.marginal_standard_errors) > 2.0


def test_loss_audit_arrays_are_read_only_and_fingerprint_changes_with_losses():
    first = LossAudit.from_matrix(
        candidate_ids=("a",),
        unit_ids=("u0", "u1"),
        design=_design(),
        losses=[[1.0], [2.0]],
    )
    second = LossAudit.from_matrix(
        candidate_ids=("a",),
        unit_ids=("u0", "u1"),
        design=_design(),
        losses=[[1.0], [3.0]],
    )
    with pytest.raises(ValueError):
        first.losses[0, 0] = 99.0
    assert first.fingerprint != second.fingerprint


def test_audit_design_is_explicit_and_changes_the_audit_fingerprint():
    with pytest.raises(ValueError, match="evaluation_domain"):
        AuditDesign(
            loss_name="mse",
            unit_kind="row",
            fit_protocol="external_holdout",
            evaluation_domain={},
        )

    first = LossAudit.from_matrix(
        candidate_ids=("a",),
        unit_ids=("u0", "u1"),
        design=_design("experiment"),
        losses=[[1.0], [2.0]],
    )
    second = LossAudit.from_matrix(
        candidate_ids=("a",),
        unit_ids=("u0", "u1"),
        design=_design("trajectory"),
        losses=[[1.0], [2.0]],
    )
    assert first.fingerprint != second.fingerprint
