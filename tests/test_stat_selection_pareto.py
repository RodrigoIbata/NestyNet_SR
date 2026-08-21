# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import json

import numpy as np
import pytest

from nestynet_sr.stat_selection import (
    AuditDesign,
    CandidateArchive,
    ComplexityVector,
    LossAudit,
    bootstrap_front_inclusion_frequencies,
    build_certificate,
    confidence_pareto,
    point_pareto_front,
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
    archive = CandidateArchive(archive_label="pareto-test")
    archive.add_structure(
        "simple",
        ComplexityVector.from_mapping({"ast_length": 2, "free_parameters": 0}),
        candidate_id="simple",
    )
    archive.add_structure(
        "middle",
        ComplexityVector.from_mapping({"ast_length": 4, "free_parameters": 1}),
        candidate_id="middle",
    )
    archive.add_structure(
        "complex",
        ComplexityVector.from_mapping({"ast_length": 8, "free_parameters": 3}),
        candidate_id="complex",
    )
    return archive.freeze()


def test_point_front_retains_risk_complexity_tradeoffs_and_removes_dominated():
    archive = _archive()
    front = point_pareto_front(
        archive.candidate_ids,
        [1.0, 0.8, 0.9],
        archive.complexity_by_id(),
    )
    assert front == ("middle", "simple")


def test_confidence_front_uses_paired_simultaneous_dominance():
    archive = _archive()
    rng = np.random.default_rng(4)
    n = 80
    shared = rng.normal(0.0, 2.0, size=n)
    losses_by_id = {
        "simple": 1.0 + shared,
        "middle": 0.75 + shared + rng.normal(0.0, 0.03, size=n),
        "complex": 0.95 + shared + rng.normal(0.0, 0.03, size=n),
    }
    losses = np.column_stack([losses_by_id[candidate_id] for candidate_id in archive.candidate_ids])
    audit = LossAudit.from_matrix(
        candidate_ids=archive.candidate_ids,
        unit_ids=tuple(f"u{i}" for i in range(n)),
        design=_design(),
        losses=losses,
        archive=archive,
    )
    result = confidence_pareto(
        audit,
        archive,
        n_resamples=1200,
        seed=7,
    )

    assert result.point_front == ("middle", "simple")
    assert result.confidence_front == ("middle", "simple")
    assert ("middle", "complex") in result.strict_dominance_edges
    assert result.dominators_of("complex") == ("middle",)
    assert result.critical_value > 0.0


def test_practical_front_prefers_simpler_model_only_with_declared_margin():
    archive = _archive()
    rng = np.random.default_rng(9)
    n = 120
    common = rng.normal(size=n)
    # Middle is around 0.01 better than simple, but not by enough to matter under delta=0.03.
    losses_by_id = {
        "simple": 1.00 + common + rng.normal(0.0, 0.01, size=n),
        "middle": 0.99 + common + rng.normal(0.0, 0.01, size=n),
        "complex": 1.20 + common + rng.normal(0.0, 0.01, size=n),
    }
    losses = np.column_stack([losses_by_id[candidate_id] for candidate_id in archive.candidate_ids])
    audit = LossAudit.from_matrix(
        candidate_ids=archive.candidate_ids,
        unit_ids=tuple(f"u{i}" for i in range(n)),
        design=_design(),
        losses=losses,
        archive=archive,
    )
    strict = confidence_pareto(
        audit,
        archive,
        delta=0.0,
        n_resamples=1000,
        seed=10,
    )
    practical = confidence_pareto(
        audit,
        archive,
        delta=0.03,
        n_resamples=1000,
        seed=10,
    )

    assert "middle" in strict.confidence_front
    assert practical.practical_front == ("simple",)
    assert ("simple", "middle") in practical.practical_dominance_edges


def test_degenerate_pair_is_conservatively_non_estimable():
    archive = CandidateArchive()
    archive.add_structure("a", ComplexityVector.scalar(1.0), candidate_id="a")
    archive.add_structure("b", ComplexityVector.scalar(2.0), candidate_id="b")
    archive.freeze()
    audit = LossAudit.from_matrix(
        candidate_ids=("a", "b"),
        unit_ids=tuple(f"u{i}" for i in range(12)),
        design=_design(),
        losses=np.ones((12, 2)),
        archive=archive,
    )
    result = confidence_pareto(audit, archive, delta=0.1, n_resamples=300, seed=2)

    comparison = next(
        item
        for item in result.comparisons
        if item.challenger_id == "a" and item.incumbent_id == "b"
    )
    assert comparison.estimable is False
    assert np.isinf(comparison.upper_confidence_bound)
    assert set(result.practical_front) == {"a", "b"}


def test_max_t_seed_is_deterministic_and_multiplicity_is_simultaneous():
    archive = _archive()
    rng = np.random.default_rng(20)
    losses = rng.normal(size=(50, 3))
    audit = LossAudit.from_matrix(
        candidate_ids=archive.candidate_ids,
        unit_ids=tuple(f"u{i}" for i in range(50)),
        design=_design(),
        losses=losses,
        archive=archive,
    )
    first = confidence_pareto(audit, archive, n_resamples=700, seed=55)
    second = confidence_pareto(audit, archive, n_resamples=700, seed=55)

    assert first.critical_value == second.critical_value
    assert first.to_dict() == second.to_dict()
    # A one-sided simultaneous critical value over several comparisons should
    # exceed the ordinary one-comparison 95% normal value in this deterministic run.
    assert first.critical_value > 1.64


def test_bootstrap_front_frequency_is_a_reproducible_stability_diagnostic():
    archive = _archive()
    losses_by_id = {
        "simple": np.array([1.0, 1.1, 0.9, 1.0]),
        "middle": np.array([0.9, 0.8, 0.9, 0.8]),
        "complex": np.array([1.2, 1.3, 1.1, 1.2]),
    }
    losses = np.column_stack([losses_by_id[candidate_id] for candidate_id in archive.candidate_ids])
    audit = LossAudit.from_matrix(
        candidate_ids=archive.candidate_ids,
        unit_ids=("u0", "u1", "u2", "u3"),
        design=_design(),
        losses=losses,
        archive=archive,
    )
    first = bootstrap_front_inclusion_frequencies(
        audit,
        archive,
        n_resamples=300,
        seed=8,
    )
    second = bootstrap_front_inclusion_frequencies(
        audit,
        archive,
        n_resamples=300,
        seed=8,
    )
    assert first == second
    assert first["complex"] == 0.0
    assert 0.0 <= first["middle"] <= 1.0


def test_predeclared_ineligible_candidates_are_recorded_but_not_inferred():
    archive = _archive()
    losses = np.column_stack(
        [
            {
                "simple": np.linspace(1.0, 1.2, 30),
                "middle": np.linspace(0.8, 1.0, 30),
                "complex": np.zeros(30),
            }[candidate_id]
            for candidate_id in archive.candidate_ids
        ]
    )
    audit = LossAudit.from_matrix(
        candidate_ids=archive.candidate_ids,
        unit_ids=tuple(f"u{i}" for i in range(30)),
        design=_design(),
        losses=losses,
        archive=archive,
    )
    result = confidence_pareto(
        audit,
        archive,
        eligible_candidate_ids=("simple", "middle"),
        n_resamples=300,
        seed=13,
    )
    frequencies = bootstrap_front_inclusion_frequencies(
        audit,
        archive,
        eligible_candidate_ids=("simple", "middle"),
        n_resamples=100,
        seed=14,
    )

    assert result.eligible_candidate_ids == ("middle", "simple")
    assert result.ineligible_candidate_ids == ("complex",)
    assert "complex" not in result.point_front
    assert "complex" not in result.confidence_front
    assert all(
        comparison.challenger_id != "complex" and comparison.incumbent_id != "complex"
        for comparison in result.comparisons
    )
    assert frequencies["complex"] == 0.0


def test_certificate_roundtrip_contains_frozen_archive_and_loss_hash(tmp_path):
    archive = _archive()
    audit = LossAudit.from_matrix(
        candidate_ids=archive.candidate_ids,
        unit_ids=tuple(f"u{i}" for i in range(20)),
        design=_design(),
        losses=np.column_stack(
            [
                {
                    "simple": np.linspace(1.0, 2.0, 20),
                    "middle": np.linspace(0.8, 1.8, 20),
                    "complex": np.linspace(1.2, 2.2, 20),
                }[candidate_id]
                for candidate_id in archive.candidate_ids
            ]
        ),
        archive=archive,
    )
    result = confidence_pareto(audit, archive, n_resamples=300, seed=3)
    certificate = build_certificate(archive, audit, result)
    path = certificate.write_json(tmp_path / "certificate.json")
    payload = json.loads(path.read_text())

    assert payload["archive"]["fingerprint"] == archive.fingerprint
    assert payload["audit"]["audit_fingerprint"] == audit.fingerprint
    assert "Conditional on the frozen candidate archive" in payload["claim"]
    assert payload["pareto"]["confidence_front"] == list(result.confidence_front)


def test_archive_backed_inference_rejects_an_unbound_loss_table():
    archive = _archive()
    audit = LossAudit.from_matrix(
        candidate_ids=archive.candidate_ids,
        unit_ids=("u0", "u1", "u2"),
        design=_design(),
        losses=np.ones((3, 3)),
    )
    with pytest.raises(ValueError, match="not bound"):
        confidence_pareto(audit, archive, n_resamples=100, seed=1)


def test_bonferroni_method_uses_closed_form_t_critical_value():
    from scipy.stats import t as student_t

    archive = _archive()
    rng = np.random.default_rng(11)
    n = 80
    shared = rng.normal(0.0, 2.0, size=n)
    losses = np.column_stack(
        [
            1.0 + shared,
            0.75 + shared + rng.normal(0.0, 0.03, size=n),
            0.95 + shared + rng.normal(0.0, 0.03, size=n),
        ]
    )
    audit = LossAudit.from_matrix(
        candidate_ids=archive.candidate_ids,
        unit_ids=tuple(f"u{i}" for i in range(n)),
        design=_design(),
        losses=losses,
        archive=archive,
    )
    bonferroni = confidence_pareto(
        audit,
        archive,
        alpha=0.05,
        n_resamples=200,
        seed=3,
        method="bonferroni_t",
        bonferroni_comparisons=10,
    )
    assert bonferroni.critical_value_method == "bonferroni_t"
    assert bonferroni.critical_value == pytest.approx(
        float(student_t.ppf(1.0 - 0.05 / 10, n - 1))
    )
    assert bonferroni.to_dict()["critical_value_method"] == "bonferroni_t"

    bootstrap = confidence_pareto(
        audit, archive, alpha=0.05, n_resamples=2000, seed=3
    )
    assert bootstrap.critical_value_method == "multiplier_max_t"
    # Over a declared family of 10 the closed form is the conservative one.
    assert bonferroni.critical_value > bootstrap.critical_value

    with pytest.raises(ValueError, match="method"):
        confidence_pareto(audit, archive, n_resamples=50, seed=1, method="nope")
    with pytest.raises(ValueError, match="non-negative"):
        confidence_pareto(
            audit,
            archive,
            n_resamples=50,
            seed=1,
            bonferroni_comparisons=-1,
        )
