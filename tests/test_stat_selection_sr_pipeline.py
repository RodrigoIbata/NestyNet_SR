# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nestynet_sr.stat_selection import (
    AuditDesign,
    CandidateArchive,
    ComplexityVector,
    LossAudit,
    NoPortableAnalyticCandidatesError,
    SRArchiveBuild,
    build_sr_candidate_archive,
    certify_sr_archive,
    evaluate_sr_archive,
    prepare_sr_audit_plan,
    update_report_with_sr_statistical_selection,
)
from nestynet_sr.stat_selection.sr_pipeline import _identification_selection


def _write_csv(path: Path, x, y):
    pd.DataFrame({"x0": np.asarray(x), "y": np.asarray(y)}).to_csv(path, index=False)
    return path


def _fixed_archive(expressions):
    archive = CandidateArchive(archive_label="fixed-test")
    for i, expression in enumerate(expressions):
        archive.add_structure(
            f"structure-{i}-{expression}",
            ComplexityVector.from_mapping(
                {"ast_nodes": i + 1, "free_parameters": 0, "tree_depth": i + 1}
            ),
            candidate_id=f"c{i}",
            metadata={
                "expression": expression,
                "symbol_values": {},
                "coefficient_metadata": None,
                "unit_admissibility": None,
            },
            provenance=[{"source": "test"}],
        )
    return archive.freeze()


def _identification_probe(losses, *, delta=0.0, n_resamples=500):
    candidate_ids = tuple(f"c{i}" for i in range(np.asarray(losses).shape[1]))
    audit = LossAudit.from_matrix(
        candidate_ids=candidate_ids,
        unit_ids=tuple(f"u{i}" for i in range(np.asarray(losses).shape[0])),
        design=AuditDesign("test_loss", "iid_row", "frozen", {"kind": "test"}),
        losses=np.asarray(losses, dtype=np.float64),
    )
    complexities = {candidate_id: ComplexityVector.from_mapping(
        {"free_parameters": 0, "constant_code": 0,
         "ast_nodes": i + 1, "tree_depth": i + 1}
    ) for i, candidate_id in enumerate(candidate_ids)}
    return _identification_selection(
        audit, complexities, eligible_candidate_ids=candidate_ids, alpha=0.05,
        delta=delta, n_resamples=n_resamples, seed=41, multiplier="normal",
    )


def test_identification_occam_edge_cases():
    base = np.linspace(0.1, 0.2, 300)
    result = _identification_probe(np.column_stack([base + 1.0e-16, base]))
    assert result.selected_candidate_id == "c0"
    challenge = result.to_dict()["challenges"][0]
    assert challenge["evidential_state"] == "numerical_tie"
    assert challenge["max_abs_unit_loss_difference"] <= challenge["numerical_tie_tolerance"]

    phase = np.linspace(0.0, 4.0 * np.pi, 300)
    complex_losses = 0.1 + 0.01 * np.cos(phase)
    simple_losses = complex_losses + 0.02 + 0.002 * np.sin(phase)
    result = _identification_probe(
        np.column_stack([simple_losses, complex_losses]), delta=0.05
    )
    assert result.selected_candidate_id == "c0"
    assert result.to_dict()["challenges"][0]["evidential_state"] == (
        "improvement_certified_but_below_margin"
    )

    result = _identification_probe(np.column_stack([base + 1.0, base]))
    challenge = result.to_dict()["challenges"][0]
    assert challenge["numerical_tie"] is False
    assert challenge["evidential_state"] == "no_certified_improvement_nonestimable"


def test_identification_complexity_must_earn_a_material_improvement():
    phase = np.linspace(0.0, 4.0 * np.pi, 300)
    complex_losses = 0.1 + 0.01 * np.cos(phase)
    middle_losses = complex_losses + 0.2 + 0.01 * np.sin(phase)
    simple_losses = middle_losses + 0.2 + 0.01 * np.cos(phase)
    result = _identification_probe(np.column_stack(
        [simple_losses, middle_losses, complex_losses]), delta=0.05)

    assert result.selected_candidate_id == "c2"
    assert result.comparison_family_size_pre_audit == 3
    challenges = result.to_dict()["challenges"]
    assert [row["incumbent_before"] for row in challenges] == ["c0", "c1"]
    assert all(
        row["evidential_state"] == "complexity_earned_by_certified_improvement"
        and row["lower_confidence_bound"] > 0.05
        for row in challenges
    )


def _named_coefficient_metadata(name, value, *, kind):
    fixed = kind == "fixed_const"
    return {
        "schema": "coefficient_metadata_v1",
        "valid": True,
        "code": "coefficient_metadata_ok",
        "reason": "coefficient metadata is valid",
        "source": "test",
        "dimension_basis": [],
        "dataset_id": None,
        "dataset_index": None,
        "record_count": 1,
        "symbol_count": 1,
        "records": [
            {
                "identity": f"{kind}:{name}",
                "kind": kind,
                "name": name,
                "symbol": name,
                "display": "symbol",
                "value": float(value),
                "dimension": None,
                "dimension_status": "unavailable",
                "scope": "fixed" if fixed else "experiment",
                "trainable": not fixed,
                "value_source": "fixed_buffer" if fixed else "fitted_parameter",
                "dataset_id": None,
                "dataset_index": None,
                "occurrences": [],
            }
        ],
    }


def test_prepare_sr_audit_plan_physically_withholds_contiguous_tail(tmp_path):
    source = _write_csv(tmp_path / "problem.csv", np.arange(20), np.arange(20) ** 2)
    plan = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        audit_rows=5,
        minimum_search_rows=10,
    )

    search = pd.read_csv(plan.search_path)
    audit = pd.read_csv(plan.audit_path)
    assert plan.audit_kind == "contiguous_tail_untouched"
    assert plan.search_rows == 15
    assert plan.audit_rows == 5
    assert search["x0"].tolist() == list(range(15))
    assert audit["x0"].tolist() == list(range(15, 20))
    assert Path(plan.search_path).name != source.name
    assert plan.search_sha256 != plan.audit_sha256


def test_prepare_sr_audit_plan_external_file_keeps_search_source(tmp_path):
    source = _write_csv(tmp_path / "problem.csv", np.arange(10), np.arange(10))
    audit = _write_csv(tmp_path / "audit.csv", np.arange(10, 15), np.arange(10, 15))
    plan = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        external_audit_path=audit,
        minimum_search_rows=10,
    )
    assert plan.audit_kind == "external_untouched"
    assert Path(plan.search_path) == source.resolve()
    assert Path(plan.audit_path) == audit.resolve()


def test_prepare_sr_audit_plan_rejects_same_external_audit_file(tmp_path):
    source = _write_csv(tmp_path / "problem.csv", np.arange(10), np.arange(10))
    with pytest.raises(ValueError, match="distinct"):
        prepare_sr_audit_plan(
            source,
            results_dir=tmp_path / "results",
            external_audit_path=source,
        )


def test_prepare_sr_audit_plan_rejects_byte_identical_external_copy(tmp_path):
    source = _write_csv(tmp_path / "problem.csv", np.arange(10), np.arange(10))
    audit = tmp_path / "audit-copy.csv"
    audit.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="byte-identical"):
        prepare_sr_audit_plan(
            source,
            results_dir=tmp_path / "results",
            external_audit_path=audit,
        )


def test_prepare_sr_audit_plan_rounds_tail_to_complete_units(tmp_path):
    source = _write_csv(tmp_path / "problem.csv", np.arange(30), np.arange(30))
    plan = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        audit_rows=5,
        unit_size=3,
        minimum_search_rows=20,
    )
    assert plan.audit_rows == 6
    assert plan.search_rows == 24
    assert plan.unit_size == 3
    assert plan.checkpoint_contract()["unit_size"] == 3


def test_prepare_sr_audit_plan_rejects_partial_external_units(tmp_path):
    source = _write_csv(tmp_path / "problem.csv", np.arange(10), np.arange(10))
    audit = _write_csv(tmp_path / "audit.csv", np.arange(10, 15), np.arange(10, 15))
    with pytest.raises(ValueError, match="not divisible"):
        prepare_sr_audit_plan(
            source,
            results_dir=tmp_path / "results",
            external_audit_path=audit,
            unit_size=2,
        )


def test_prepare_sr_audit_plan_rejects_negative_row_request(tmp_path):
    source = _write_csv(tmp_path / "problem.csv", np.arange(10), np.arange(10))
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_sr_audit_plan(
            source,
            results_dir=tmp_path / "results",
            audit_rows=-1,
        )


def test_prepare_sr_audit_plan_infers_single_y_prefixed_target(tmp_path):
    source = tmp_path / "problem.csv"
    pd.DataFrame({"x0": np.arange(10), "y0": 2 * np.arange(10)}).to_csv(
        source, index=False
    )
    plan = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        audit_rows=4,
    )
    assert plan.target_column == "y0"
    assert list(pd.read_csv(plan.search_path).columns) == ["x0", "y0"]


def test_prepare_sr_audit_plan_rejects_multioutput_y_schema(tmp_path):
    source = tmp_path / "problem.csv"
    pd.DataFrame(
        {
            "x0": np.arange(10),
            "y": np.arange(10),
            "y_aux": np.arange(10),
        }
    ).to_csv(source, index=False)
    with pytest.raises(ValueError, match="exactly one y-prefixed"):
        prepare_sr_audit_plan(
            source,
            results_dir=tmp_path / "results",
            audit_rows=4,
        )


def test_prepare_sr_audit_plan_rejects_nonnumeric_schema(tmp_path):
    source = tmp_path / "problem.csv"
    pd.DataFrame({"x0": ["a", "b", "c", "d"], "y": [1, 2, 3, 4]}).to_csv(
        source, index=False
    )
    with pytest.raises(ValueError, match="numeric columns"):
        prepare_sr_audit_plan(
            source,
            results_dir=tmp_path / "results",
            audit_rows=2,
        )


def test_global_archive_includes_all_polish_candidates_and_merges_provenance():
    final_polish = {
        "all_candidates": [
            {"expr": "x0", "label": "a", "n_free_params": 0},
            {"expr": "x0 + 0", "label": "same", "n_free_params": 0},
            {"expr": "2*x0", "label": "b", "n_free_params": 0},
        ],
        "recommended": {"expr": "2*x0", "label": "recommended", "n_free_params": 0},
        "seed_expr": "x0",
    }
    build = build_sr_candidate_archive(
        stageB_data=None,
        final_polish_summary=final_polish,
        max_candidates=20,
    )

    assert build.discovered_count >= 3
    assert build.canonical_count == 2
    assert len(build.archive) == 2
    expressions = {candidate.metadata["expression"] for candidate in build.archive.candidates}
    assert expressions == {"x0", "2*x0"}
    x0 = next(candidate for candidate in build.archive.candidates if candidate.metadata["expression"] == "x0")
    assert len(x0.provenance) >= 2
    two_x0 = next(
        candidate
        for candidate in build.archive.candidates
        if candidate.metadata["expression"] == "2*x0"
    )
    assert {record["source"] for record in two_x0.provenance} >= {
        "final_polish:recommended",
        "final_polish:all_candidates",
    }
    assert x0.complexity.names == (
        "ast_nodes",
        "constant_code",
        "free_parameters",
        "tree_depth",
    )


def test_archive_preserves_fixed_physical_symbol_in_candidate_identity():
    build = build_sr_candidate_archive(
        stageB_data=None,
        final_polish_summary={
            "all_candidates": [
                {
                    "expr": "c*x0",
                    "label": "physical",
                    "coefficient_metadata": _named_coefficient_metadata(
                        "c", 2.0, kind="fixed_const"
                    ),
                },
                {"expr": "2*x0", "label": "literal"},
            ]
        },
    )
    assert len(build.archive) == 2
    structures = {
        candidate.metadata["expression"]: candidate.canonical_structure
        for candidate in build.archive.candidates
    }
    assert structures["c*x0"] != structures["2*x0"]
    assert "Symbol('c'" in structures["c*x0"]


def test_archive_does_not_merge_conflicting_fixed_constant_contracts():
    build = build_sr_candidate_archive(
        stageB_data=None,
        final_polish_summary={
            "all_candidates": [
                {
                    "expr": "c*x0",
                    "label": "c-two",
                    "coefficient_metadata": _named_coefficient_metadata(
                        "c", 2.0, kind="fixed_const"
                    ),
                },
                {
                    "expr": "c*x0",
                    "label": "c-three",
                    "coefficient_metadata": _named_coefficient_metadata(
                        "c", 3.0, kind="fixed_const"
                    ),
                },
            ]
        },
    )
    assert len(build.archive) == 2
    assert len({candidate.canonical_structure for candidate in build.archive.candidates}) == 2


def test_archive_counts_named_fitted_parameter_even_when_artifact_omits_count():
    build = build_sr_candidate_archive(
        stageB_data=None,
        final_polish_summary={
            "all_candidates": [
                {
                    "expr": "a*x0",
                    "label": "fitted",
                    "n_free_params": 0,
                    "coefficient_metadata": _named_coefficient_metadata(
                        "a", 2.0, kind="free_const"
                    ),
                }
            ]
        },
    )
    candidate = build.archive.candidates[0]
    assert candidate.complexity.as_dict()["free_parameters"] == 1.0
    assert "Float" in candidate.canonical_structure


def test_common_domain_evaluation_penalizes_undefined_candidate(tmp_path):
    audit_path = _write_csv(
        tmp_path / "audit.csv",
        np.linspace(1.0, 3.0, 12),
        2.0 * np.linspace(1.0, 3.0, 12),
    )
    archive = _fixed_archive(("2*x0", "1/(x0-x0)"))
    evaluation = evaluate_sr_archive(
        archive,
        audit_path=audit_path,
        loss_scale=2.0,
        unit_size=3,
        failure_loss=1000.0,
    )

    assert evaluation.audit.n_units == 4
    assert np.allclose(evaluation.audit.losses[:, 0], 0.0)
    assert np.all(evaluation.audit.losses[:, 1] == 1000.0)
    assert np.all(evaluation.audit.failure_mask[:, 1])
    assert evaluation.candidate_failures["c1"]["failed_units"] == 4
    assert evaluation.eligible_candidate_ids == ("c0",)


def test_evaluation_enforces_sealed_unit_schema(tmp_path):
    source = _write_csv(tmp_path / "problem.csv", np.arange(30), np.arange(30))
    plan = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        audit_rows=6,
        unit_size=3,
        minimum_search_rows=20,
    )
    archive = _fixed_archive(("x0",))
    with pytest.raises(ValueError, match="does not match"):
        evaluate_sr_archive(
            archive,
            audit_path=plan.audit_path,
            split_plan=plan,
            loss_scale=1.0,
            unit_size=2,
        )


def test_evaluation_rejects_unsealed_partial_unit(tmp_path):
    audit_path = _write_csv(tmp_path / "audit.csv", np.arange(5), np.arange(5))
    archive = _fixed_archive(("x0",))
    with pytest.raises(ValueError, match="not divisible"):
        evaluate_sr_archive(
            archive,
            audit_path=audit_path,
            loss_scale=1.0,
            unit_size=2,
        )


def test_failed_candidate_is_retained_but_excluded_from_certified_fronts(tmp_path):
    audit_path = _write_csv(
        tmp_path / "audit.csv",
        np.linspace(1.0, 3.0, 12),
        2.0 * np.linspace(1.0, 3.0, 12),
    )
    archive = _fixed_archive(("2*x0", "1/(x0-x0)"))
    evaluation = evaluate_sr_archive(
        archive,
        audit_path=audit_path,
        loss_scale=2.0,
        unit_size=1,
        failure_loss=1000.0,
    )
    build = SRArchiveBuild(
        archive=archive,
        discovered_count=2,
        canonical_count=2,
        excluded=(),
        cap_applied=False,
        cap_policy="test",
    )
    outcome = certify_sr_archive(
        archive_build=build,
        evaluation=evaluation,
        output_dir=tmp_path / "certificate",
        n_resamples=200,
        seed=12,
    )
    payload = json.loads(Path(outcome.certificate_path).read_text())
    assert payload["schema_version"] == 2
    # The front now compares functional classes, not spellings, so its ids are
    # class ids.  The certificate must still let a reader recover which
    # candidate each class stands for.
    eligible = payload["pareto"]["eligible_candidate_ids"]
    assert len(eligible) == 1
    entries = {c["candidate_id"]: c for c in payload["archive"]["candidates"]}
    represented = {
        entries[cid]["metadata"]["representative_candidate_id"] for cid in eligible
    }
    assert represented == {"c0"}
    ineligible = payload["pareto"]["ineligible_candidate_ids"]
    assert len(ineligible) == 1
    assert {
        entries[cid]["metadata"]["representative_candidate_id"] for cid in ineligible
    } == {"c1"}
    assert "c1" not in payload["pareto"]["point_front"]
    assert "c1" not in payload["pareto"]["confidence_front"]
    assert "c1" not in payload["pareto"]["practical_front"]
    ineligible_class = next(iter(ineligible))
    assert payload["ordinary_sr_deployment"]["comparisons"][ineligible_class]["eligible"] is False


def test_standardized_audit_is_invariant_to_global_y_rescaling(tmp_path):
    x = np.linspace(-2.0, 2.0, 30)
    first_path = _write_csv(tmp_path / "first.csv", x, 2.0 * x + 0.1)
    second_path = _write_csv(tmp_path / "second.csv", x, 20.0 * x + 1.0)
    first_archive = _fixed_archive(("2*x0", "1.8*x0"))
    second_archive = _fixed_archive(("20*x0", "18*x0"))

    first = evaluate_sr_archive(first_archive, audit_path=first_path, loss_scale=2.0)
    second = evaluate_sr_archive(second_archive, audit_path=second_path, loss_scale=20.0)
    assert np.allclose(first.audit.losses, second.audit.losses, rtol=1e-12, atol=1e-12)


def test_end_to_end_certificate_updates_final_report_authoritatively(tmp_path):
    x = np.linspace(0.5, 4.0, 80)
    source = _write_csv(tmp_path / "problem.csv", x, 2.0 * x)
    plan = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        audit_rows=30,
        minimum_search_rows=40,
    )
    final_polish = {
        "all_candidates": [
            {"expr": "x0", "label": "simple", "n_free_params": 0},
            {"expr": "2*x0", "label": "correct", "n_free_params": 0},
            {"expr": "3*x0", "label": "wrong", "n_free_params": 0},
        ],
        "recommended": {"expr": "x0", "label": "legacy", "n_free_params": 0},
        "seed_expr": "x0",
    }
    build = build_sr_candidate_archive(
        stageB_data=None,
        final_polish_summary=final_polish,
        split_plan=plan,
    )
    evaluation = evaluate_sr_archive(
        build.archive,
        audit_path=plan.audit_path,
        split_plan=plan,
        loss_scale=4.0,
        unit_size=1,
        failure_loss=1000.0,
    )
    outcome = certify_sr_archive(
        archive_build=build,
        evaluation=evaluation,
        output_dir=tmp_path / "cert",
        n_resamples=500,
        seed=9,
    )
    selected = build.archive[outcome.selected_candidate_id]
    assert selected.metadata["expression"] == "2*x0"
    assert Path(outcome.archive_path).is_file()
    assert Path(outcome.certificate_path).is_file()

    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"final_selection": {"source": "final_polish", "expr": "x0"}}),
        encoding="utf-8",
    )
    summary = update_report_with_sr_statistical_selection(report_path, outcome, split_plan=plan)
    report = json.loads(report_path.read_text())
    assert report["legacy_search_selection"]["expr"] == "x0"
    assert report["final_selection"]["source"] == "statistical_selection"
    assert report["final_selection"]["mode"] == (
        "confidence_pareto_with_occam_identification_selection"
    )
    assert report["final_selection"]["expr"] == "2*x0"
    assert summary["selection_basis"] == (
        "simplicity_default_complexity_requires_simultaneous_evidence"
    )
    assert summary["identification_selection"]["selected_candidate_id"] == (
        summary["selected_functional_class"]
    )
    assert summary["practical_front"]


def test_split_plan_detects_audit_mutation(tmp_path):
    source = _write_csv(tmp_path / "problem.csv", np.arange(20), np.arange(20))
    plan = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        audit_rows=5,
        minimum_search_rows=10,
    )
    archive = _fixed_archive(("x0",))
    with open(plan.audit_path, "a", encoding="utf-8") as handle:
        handle.write("99,99\n")
    with pytest.raises(RuntimeError, match="changed"):
        evaluate_sr_archive(
            archive,
            audit_path=plan.audit_path,
            split_plan=plan,
            loss_scale=1.0,
        )


def test_split_checkpoint_contract_rejects_old_or_different_firewall(tmp_path):
    source = _write_csv(
        tmp_path / "problem.csv",
        np.arange(20.0),
        2.0 * np.arange(20.0),
    )
    first = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        audit_rows=5,
        minimum_search_rows=10,
    )
    first.assert_checkpoint_compatible(first.checkpoint_contract())
    with pytest.raises(ValueError, match="predates"):
        first.assert_checkpoint_compatible(None)
    second = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results2",
        audit_rows=6,
        minimum_search_rows=10,
    )
    with pytest.raises(ValueError, match="does not match"):
        second.assert_checkpoint_compatible(first.checkpoint_contract())


def test_split_checkpoint_contract_is_path_independent(tmp_path):
    source = _write_csv(
        tmp_path / "problem.csv",
        np.arange(20.0),
        2.0 * np.arange(20.0),
    )
    first = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results_a",
        audit_rows=5,
        minimum_search_rows=10,
    )
    second = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results_b",
        audit_rows=5,
        minimum_search_rows=10,
    )
    second.assert_checkpoint_compatible(first.checkpoint_contract())
    assert second.contract_fingerprint == first.contract_fingerprint
    assert second.fingerprint != first.fingerprint


def test_search_view_mutation_is_detected_before_archive_freeze(tmp_path):
    source = _write_csv(
        tmp_path / "problem.csv",
        np.arange(20.0),
        2.0 * np.arange(20.0),
    )
    plan = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        audit_rows=5,
        minimum_search_rows=10,
    )
    with Path(plan.search_path).open("a", encoding="utf-8") as handle:
        handle.write("100,200\n")
    with pytest.raises(RuntimeError, match="search CSV changed"):
        build_sr_candidate_archive(
            stageB_data=None,
            final_polish_summary={
                "all_candidates": [{"expr": "2*x0", "label": "candidate"}]
            },
            split_plan=plan,
        )


def test_stat_selection_cli_flags_parse(monkeypatch):
    from nestynet_sr.run_sr_args import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_SR.py",
            "--filepath",
            "dummy.csv",
            "--stat-selection",
            "--stat-audit-rows",
            "50",
            "--stat-unit-size",
            "5",
            "--stat-alpha",
            "0.1",
            "--stat-delta",
            "0.02",
            "--stat-resamples",
            "777",
            "--stat-multiplier",
            "rademacher",
        ],
    )
    args = parse_args()
    assert args.stat_selection is True
    assert args.stat_audit_rows == 50
    assert args.stat_unit_size == 5
    assert args.stat_alpha == pytest.approx(0.1)
    assert args.stat_delta == pytest.approx(0.02)
    assert args.stat_resamples == 777
    assert args.stat_multiplier == "rademacher"


def test_schur_profile_audit_accounts_for_x_error(tmp_path):
    x=np.linspace(-2,2,20); y=2*x
    audit=_write_csv(tmp_path/"audit_xy.csv",x,y)
    archive=_fixed_archive(["2*x0", "2.2*x0"])
    exact=evaluate_sr_archive(archive,audit_path=audit,loss_scale=.1,unit_size=1)
    prof=evaluate_sr_archive(archive,audit_path=audit,loss_scale=.1,unit_size=1,
                             x_sigma="0.5",x_error_loss="profile_chi2")
    assert prof.audit.metadata["joint_xy_errors"] is True
    assert prof.audit.metadata["schur_profiled_latent_inputs"] is True
    assert prof.audit.risks[1] < exact.audit.risks[1]
    assert prof.candidate_failures["c1"]["median_x_variance_inflation"] > 1.0


def test_schur_profile_full_covariance_npz(tmp_path):
    x=np.linspace(-1,1,12); y=x
    audit=_write_csv(tmp_path/"audit_cov.csv",x,y)
    cov=tmp_path/"cov.npz"; np.savez(cov,x_cov=np.asarray([[.04]]))
    archive=_fixed_archive(["x0"])
    out=evaluate_sr_archive(archive,audit_path=audit,loss_scale=.2,unit_size=1,
                            x_cov_npz=cov,x_error_loss="marginal_gaussian_nll")
    model=out.audit.design.evaluation_domain["x_error_model"]
    assert model["source"]=="npz"
    assert len(model["sha256"])==64
    assert np.isfinite(out.audit.losses).all()


def test_empty_archive_error_reports_upstream_portability_drops():
    """The pb101/pb119 failure mode: every candidate carries an unresolved NN
    leaf, the collector silently drops all of them, and the old error said
    ``exclusions=[]`` - hiding the one fact that identifies the mechanism."""
    final_polish = {
        "all_candidates": [
            {"expr": "x0 * nn0(x1, x2)", "label": "a", "n_free_params": 0},
            {"expr": "leaf4(x1/x0, x6) + x3", "label": "b", "n_free_params": 0},
        ],
        "recommended": {"expr": "x2 * rpoly0(x1)", "label": "rec", "n_free_params": 0},
    }
    with pytest.raises(NoPortableAnalyticCandidatesError) as excinfo:
        build_sr_candidate_archive(
            stageB_data=None,
            final_polish_summary=final_polish,
            max_candidates=20,
        )
    message = str(excinfo.value)
    assert "unresolved_nn_leaf" in message
    assert "local_fitted_wrapper" in message
    assert "no portable analytic SR candidates" in message


def test_empty_archive_error_reports_uncertified_stagec_leaf():
    with pytest.raises(NoPortableAnalyticCandidatesError) as excinfo:
        build_sr_candidate_archive(
            stageB_data={
                "y_expr_str": "x0 * leaf4(x1/x0, x6)",
                "sympy_meta": {"kind": "unresolved_symbolic_leaves", "accepted": False},
            },
            final_polish_summary=None,
            max_candidates=20,
        )

    message = str(excinfo.value)
    assert "upstream_drops={'unresolved_nn_leaf': 1}" in message
    assert "no portable analytic SR candidates" in message


def test_successful_archive_records_upstream_drop_telemetry():
    final_polish = {
        "all_candidates": [
            {"expr": "x0 + x1", "label": "good", "n_free_params": 0},
            {"expr": "x0 * nn0(x1)", "label": "bad", "n_free_params": 0},
        ],
    }
    build = build_sr_candidate_archive(
        stageB_data=None,
        final_polish_summary=final_polish,
        max_candidates=20,
    )
    assert build.upstream_drop_reasons() == {"unresolved_nn_leaf": 1}
    payload = build.to_dict()["upstream_drops"]
    assert payload["count"] == 1
    assert payload["reasons"] == {"unresolved_nn_leaf": 1}
    assert payload["samples"][0]["expr_preview"] == "x0 * nn0(x1)"
    assert payload["samples"][0]["source"] == "final_polish:all_candidates"


def _certified_probe(
    tmp_path,
    *,
    expressions,
    audit_rows=30,
    noise_sigma=0.05,
    delta_function=None,
    seed=9,
    target=None,
):
    """End-to-end probe: data from 2*x0 (plus noise), fixed expression archive."""
    rng = np.random.default_rng(77)
    x = np.linspace(0.5, 4.0, 80)
    truth = target(x) if target is not None else 2.0 * x
    y = truth + rng.normal(0.0, noise_sigma, size=x.size)
    source = _write_csv(tmp_path / "problem.csv", x, y)
    plan = prepare_sr_audit_plan(
        source,
        results_dir=tmp_path / "results",
        audit_rows=audit_rows,
        minimum_search_rows=40,
    )
    final_polish = {
        "all_candidates": [
            {"expr": expr, "label": f"cand{i}", "n_free_params": 0}
            for i, expr in enumerate(expressions)
        ],
        "recommended": {"expr": expressions[0], "label": "legacy", "n_free_params": 0},
        "seed_expr": expressions[0],
    }
    build = build_sr_candidate_archive(
        stageB_data=None,
        final_polish_summary=final_polish,
        split_plan=plan,
    )
    evaluation = evaluate_sr_archive(
        build.archive,
        audit_path=plan.audit_path,
        split_plan=plan,
        loss_scale=4.0,
        unit_size=1,
        failure_loss=1000.0,
    )
    kwargs = {} if delta_function is None else {"delta_function": delta_function}
    outcome = certify_sr_archive(
        archive_build=build,
        evaluation=evaluation,
        output_dir=tmp_path / "cert",
        n_resamples=500,
        seed=seed,
        **kwargs,
    )
    certificate = json.loads(Path(outcome.certificate_path).read_text())
    return build, outcome, certificate


def test_compression_certificate_names_class_level_winner(tmp_path):
    build, outcome, certificate = _certified_probe(
        tmp_path, expressions=["x0", "2*x0", "3*x0"]
    )
    compression = certificate["compression"]
    assert "error" not in compression
    # Statistical selection and the compression claim must agree on the law.
    assert compression["model_expression"] == "2*x0"
    assert build.archive[outcome.selected_candidate_id].metadata["expression"] == "2*x0"
    # The winning class's audit loss is charged, not a defaulted zero.
    assert compression["data_code_bits"] > 0.0
    assert np.isfinite(compression["model_code_bits"])
    assert compression["model_code_bits"] > 0.0


def test_exact_spellings_merge_before_audit(tmp_path):
    # Factored and expanded spellings survive archive canonicalisation as
    # distinct candidates (SymPy does not auto-distribute a symbol over a
    # sum), yet denote one function; the exact quotient must merge them
    # before the audit.
    build, outcome, certificate = _certified_probe(
        tmp_path,
        expressions=["x0*(x0 + 1)", "x0**2 + x0", "x0"],
        target=lambda x: x * (x + 1.0),
    )
    classes = certificate["functional_classes"]
    assert classes["equivalence_kind"] == "exact_algebraic"
    assert classes["n_classes"] == 2
    regime = certificate["inference_regime"]
    assert regime["quotiented_by"] == "exact_algebraic_identity"
    assert regime["n_candidates_before_quotient"] == 3
    assert regime["n_candidates_frozen"] == 2
    selected = build.archive[outcome.selected_candidate_id]
    assert selected.metadata["expression"] in {"x0*(x0 + 1)", "x0**2 + x0"}


def test_exact_spellings_collapsing_to_single_class_certify_without_comparisons(
    tmp_path,
):
    build, outcome, certificate = _certified_probe(
        tmp_path,
        expressions=["x0*(x0 + 1)", "x0**2 + x0"],
        target=lambda x: x * (x + 1.0),
    )

    classes = certificate["functional_classes"]
    assert classes["n_candidates"] == 2
    assert classes["n_classes"] == 1
    regime = certificate["inference_regime"]
    assert regime["comparison_family_size_pre_audit"] == 0
    assert regime["comparison_family_size_estimable"] == 0
    assert outcome.pareto.comparisons == ()
    assert outcome.pareto.critical_value == 0.0
    assert len(outcome.pareto.confidence_front) == 1
    selected = build.archive[outcome.selected_candidate_id]
    assert selected.metadata["expression"] in {"x0*(x0 + 1)", "x0**2 + x0"}


def test_exact_decimal_half_merges_with_rational_half_and_uses_simpler_spelling(
    tmp_path,
):
    decimal = "exp(-0.5*x0**2)"
    rational = "exp(-x0**2/2)"
    build, outcome, certificate = _certified_probe(
        tmp_path,
        expressions=[decimal, rational],
        noise_sigma=0.0,
        target=lambda x: np.exp(-(x**2) / 2.0),
    )

    classes = certificate["functional_classes"]
    assert classes["n_candidates"] == 2
    assert classes["n_classes"] == 1
    assert certificate["inference_regime"]["comparison_family_size_pre_audit"] == 0
    selected = build.archive[outcome.selected_candidate_id]
    assert selected.metadata["expression"] == rational


def test_decimal_approximation_does_not_merge_with_exact_rational(tmp_path):
    _, _, certificate = _certified_probe(
        tmp_path,
        expressions=["x0/3", "0.3333333333333333*x0"],
        noise_sigma=0.0,
        target=lambda x: x / 3.0,
    )

    classes = certificate["functional_classes"]
    assert classes["n_candidates"] == 2
    assert classes["n_classes"] == 2


def test_out_of_envelope_regime_executes_bonferroni_t(tmp_path):
    from scipy.stats import t as student_t

    build, outcome, certificate = _certified_probe(
        tmp_path, expressions=["x0", "2*x0", "3*x0"], audit_rows=30
    )
    regime = certificate["inference_regime"]
    # G=30 sits below every validated calibration cell, so the lookup must
    # refuse the bootstrap...
    assert regime["method"] == "bonferroni_t"
    assert regime["decision"] in {"fallback", "beyond_grid"}
    # ...and the front must have EXECUTED the fallback, not merely recorded it.
    assert regime["method_executed"] == "bonferroni_t"
    assert outcome.pareto.critical_value_method == "bonferroni_t"
    k_pre = int(regime["comparison_family_size_pre_audit"])
    n_units = int(regime["independent_units"])
    expected = float(student_t.ppf(1.0 - 0.05 / k_pre, n_units - 1))
    assert outcome.pareto.critical_value == pytest.approx(expected)
    assert regime["critical_value"] == pytest.approx(expected)


def test_identification_and_deployment_roles_are_separate(tmp_path):
    _, outcome, certificate = _certified_probe(tmp_path, expressions=["x0", "2*x0"])
    deployment = certificate["ordinary_sr_deployment"]
    identification = certificate["ordinary_sr_identification"]
    assert deployment["role"] == "secondary_deployment_firewall" and "does not choose" in deployment["note"]
    assert outcome.summary()["deployment_noninferiority"]["role"] == deployment["role"]
    assert (identification["role"], identification["selection_basis"]) == (
        "authoritative_identification_decision", "simplicity_default_complexity_requires_simultaneous_evidence"
    )
    assert identification["complexity_order_rule"].startswith("lexicographic(")
    assert (len(identification["complexity_order_records"]), identification["n_resamples"], identification["seed"], identification["multiplier"]) == (2, 500, 12, "normal")


def test_delta_function_argument_is_honored_descriptively(tmp_path):
    _, _, certificate = _certified_probe(
        tmp_path, expressions=["x0", "2*x0", "3*x0"], delta_function=1.0e6
    )
    near = certificate["near_equivalence_descriptive"]
    assert near["delta_function"] == 1.0e6
    assert near["delta_function_rule"] == "caller-declared"
    assert near["role"] == "descriptive_only"
    # A huge declared tolerance merges everything descriptively...
    assert near["n_classes"] == 1
    # ...but must not shrink the inferential family, which stays exact.
    assert certificate["functional_classes"]["n_classes"] == 3
    assert certificate["inference_regime"]["n_candidates_frozen"] == 3


def test_drop_addend_candidate_enters_archive_and_wins_identification():
    """pb114 replay: the snapped drop+refit candidate must enter the archive
    from the polish frontier and sort BEFORE the integer-coefficient seed in
    the predeclared identification complexity order (free_parameters first,
    then constant_code, ast_nodes, tree_depth)."""

    from nestynet_sr.stat_selection.sr_pipeline import (
        _identification_complexity_key,
    )

    seed_expr = (
        "23328772*x0*x2/(23328772*x0**2 + 23596222*x0*x1 "
        "- 179905*sqrt(2)*x1**2)"
    )
    truth_expr = "x2/(x0 + x1)"
    final_polish = {
        "all_candidates": [
            {"expr": seed_expr, "label": "seed", "n_free_params": 0},
            {
                "expr": truth_expr,
                "label": "drop_addend_refit_snap:s0:d1",
                "n_free_params": 0,
                "source_hints": ["drop_addend_refit"],
            },
        ],
        "seed_expr": seed_expr,
    }
    build = build_sr_candidate_archive(
        stageB_data=None,
        final_polish_summary=final_polish,
        max_candidates=20,
    )
    expressions = {
        candidate.metadata["expression"] for candidate in build.archive.candidates
    }
    assert truth_expr in expressions
    assert seed_expr in expressions
    complexity_by_id = {
        candidate.candidate_id: candidate.complexity
        for candidate in build.archive.candidates
    }
    by_expr = {
        candidate.metadata["expression"]: candidate.candidate_id
        for candidate in build.archive.candidates
    }
    truth_key = _identification_complexity_key(
        by_expr[truth_expr], complexity_by_id
    )
    seed_key = _identification_complexity_key(
        by_expr[seed_expr], complexity_by_id
    )
    assert truth_key < seed_key


def test_polish_row_selection_n_free_params_overrides_display_count():
    """A polish row declaring frozen literals enters the archive with 0 free
    parameters and sorts before an equally-frozen seed spelling with more
    ops/constant code (the pb071 identification fix)."""

    from nestynet_sr.stat_selection.sr_pipeline import (
        _identification_complexity_key,
    )

    seed_expr = "0.0449*sqrt(495.98*x0**2 + 2.0*x0 - 4903.75)"
    drop_expr = "0.044865*sqrt(496.8*x0**2 - 4903.75)"
    final_polish = {
        "all_candidates": [
            {"expr": seed_expr, "label": "seed", "n_free_params": 3},
            {
                "expr": drop_expr,
                "label": "drop_addend_refit:s0:d1",
                "n_free_params": 3,
                "selection_n_free_params": 0,
            },
        ],
        "seed_expr": seed_expr,
    }
    build = build_sr_candidate_archive(
        stageB_data=None,
        final_polish_summary=final_polish,
        max_candidates=20,
    )
    by_expr = {
        candidate.metadata["expression"]: candidate
        for candidate in build.archive.candidates
    }
    drop = by_expr[drop_expr]
    seedc = by_expr[seed_expr]
    complexity_by_id = {
        candidate.candidate_id: candidate.complexity
        for candidate in build.archive.candidates
    }
    drop_fp = dict(zip(drop.complexity.names, drop.complexity.entries))[
        "free_parameters"
    ] if hasattr(drop.complexity, "entries") else None
    key_drop = _identification_complexity_key(drop.candidate_id, complexity_by_id)
    key_seed = _identification_complexity_key(seedc.candidate_id, complexity_by_id)
    assert key_drop < key_seed
