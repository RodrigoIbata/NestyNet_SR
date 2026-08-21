# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import json

from nestynet_sr.run_sr_coe import _apply_coe_final_adjudication
from nestynet_sr.sr_search.coe_committee import (
    CommitteeEvalCache,
    ProposalReservoir,
    SliceSpec,
    StageAProposalReservoir,
    _clean_expr_verbose,
    _nontrivial_float_literal_count,
    build_stageA_proposal_reservoir,
    build_slice_specs,
    collect_final_candidates,
    evaluate_candidate_on_slice_cached,
    merge_stageA_proposal_reservoir_payloads,
    merge_proposal_reservoir_payloads,
    run_final_committee_audit,
    stageA_terminal_proposals_as_expression_reservoir,
    stageA_y_branch_names_from_proposal_reservoir,
)


def _named_c_metadata(value):
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
                "identity": "free_const:c",
                "kind": "free_const",
                "name": "c",
                "symbol": "c",
                "display": "symbol",
                "value": float(value),
                "dimension": None,
                "dimension_status": "unavailable",
                "scope": "experiment",
                "trainable": True,
                "value_source": "fitted_parameter",
                "dataset_id": None,
                "dataset_index": None,
                "occurrences": [],
            }
        ],
    }


def test_build_slice_specs_excludes_reference_slice_but_keeps_requested_count():
    specs = build_slice_specs(
        n_slices=3,
        ndata_train=10,
        ndata_val=5,
        start_slice=0,
        skip_slice_ids={0},
    )

    assert [s.slice_id for s in specs] == [1, 2, 3]
    assert [(s.train_start, s.val_stop) for s in specs] == [(15, 30), (30, 45), (45, 60)]


def test_build_slice_specs_does_not_run_past_available_rows():
    specs = build_slice_specs(
        n_slices=25,
        ndata_train=2000,
        ndata_val=2000,
        start_slice=0,
        skip_slice_ids={0},
        max_rows=100_000,
    )

    assert [s.slice_id for s in specs[:2]] == [1, 2]
    assert specs[-1].slice_id == 24
    assert len(specs) == 24


def test_collect_final_candidates_keeps_reservoir_mode_fenced():
    reservoir = ProposalReservoir(max_candidates=4)
    assert reservoir.add_expr("x0 + x1", source="scout:slice1")
    stageb = {
        "y_expr_str": "x0",
        "coe_proposal_reservoir": reservoir.to_dict(),
    }

    fenced = collect_final_candidates(
        stageB_data=stageb,
        final_polish_summary=None,
        max_candidates=8,
        include_reservoir=False,
    )
    open_pool = collect_final_candidates(
        stageB_data=stageb,
        final_polish_summary=None,
        max_candidates=8,
        include_reservoir=True,
    )

    assert [c.expr for c in fenced] == ["x0"]
    assert [c.expr for c in open_pool] == ["x0", "x0 + x1"]


def test_collect_final_candidates_rejects_nonportable_local_wrappers():
    reservoir = ProposalReservoir(max_candidates=4)
    assert not reservoir.add_expr("poly0((x0 * x1))", source="stageB")
    assert not reservoir.add_expr("rpoly((x2)^-1)", source="stageB")
    assert not reservoir.add_expr("\x1b[35mexp_poly0(x0)\x1b[0m", source="stageB")
    assert reservoir.add_expr("x0*x1", source="stageB")

    candidates = collect_final_candidates(
        stageB_data={"coe_proposal_reservoir": reservoir.to_dict()},
        final_polish_summary=None,
        max_candidates=8,
        include_reservoir=True,
    )

    assert [c.expr for c in candidates] == ["x0*x1"]


def test_pb042_prefixed_local_wrappers_share_portability_boundary():
    expressions = (
        "x0 * exp_ratpoly0(z=x1)",
        "x0 * tanh_linear0(z=x1)",
        "x0 * logshifted0(z=x1)",
        "x0 * logshifted0(z=x1**-1)",
        "x0 * polylog0(z=x1)",
        "x0 * tanh_linear0(z=x1**-1)",
    )

    for expr in expressions:
        assert _clean_expr_verbose(expr) == (None, "local_fitted_wrapper")


def test_unknown_calls_are_not_portable_but_supported_analytic_calls_are():
    supported = (
        "sqrt(Abs(x0)) + sin(x0) + cos(x0) + tan(x0)",
        "tanh(x0) + sinh(x0) + cosh(x0) + exp(x0) + log(x0)",
        "asin(x0) + acos(x0) + atan(x0) + asinh(x0) + acosh(x0) + atanh(x0)",
        "arcsin(x0) + arccos(x0) + arctan(x0) + arcsinh(x0) + arccosh(x0) + arctanh(x0)",
    )

    assert _clean_expr_verbose("mystery_transform(x0)") == (None, "local_fitted_wrapper")
    for expr in supported:
        assert _clean_expr_verbose(expr) == (expr, None)


def test_collect_final_candidates_rejects_uncertified_stagec_expression():
    candidates = collect_final_candidates(
        stageB_data={
            "y_expr_str": "x0 + 1",
            "sympy_meta": {"kind": "bad_pretty_print", "accepted": False},
        },
        final_polish_summary=None,
        max_candidates=8,
        include_reservoir=False,
    )

    assert candidates == []


def test_uncertified_stagec_expression_records_specific_drop_reason():
    cases = (
        ("x0 * leaf4(x1/x0, x6)", {"accepted": False}, "unresolved_nn_leaf"),
        (
            "x0 + x1",
            {
                "accepted": False,
                "unit_admissibility": {"checked": True, "valid": False},
            },
            "unit_invalid",
        ),
        ("x0 + x1", {"kind": "uncertified", "accepted": False}, "stagec_uncertified"),
    )

    for expr, sympy_meta, expected_reason in cases:
        drops = []
        candidates = collect_final_candidates(
            stageB_data={"y_expr_str": expr, "sympy_meta": sympy_meta},
            final_polish_summary=None,
            max_candidates=8,
            include_reservoir=False,
            dropped_log=drops,
        )
        assert candidates == []
        assert [row["reason"] for row in drops] == [expected_reason]


def test_collect_final_candidates_carries_stagec_unit_certificate():
    certificate = {
        "checked": True,
        "valid": True,
        "checker": "sympy_units_v1",
        "expression_space": "phi_and_y",
    }
    candidates = collect_final_candidates(
        stageB_data={
            "y_expr_str": "x0",
            "sympy_meta": {
                "accepted": True,
                "units_checked": True,
                "units_ok": True,
                "unit_admissibility": certificate,
            },
        },
        final_polish_summary=None,
        max_candidates=8,
        include_reservoir=False,
    )

    assert len(candidates) == 1
    assert candidates[0].metadata["unit_admissibility"] == certificate


def test_collect_final_candidates_carries_repaired_final_polish_certificates():
    valid_certificate = {
        "checked": True,
        "valid": True,
        "checker": "sympy_units_v1",
        "expression_space": "y",
    }
    invalid_seed_certificate = {
        "checked": True,
        "valid": False,
        "checker": "sympy_units_v1",
        "expression_space": "y",
    }
    candidates = collect_final_candidates(
        stageB_data={
            "y_expr_str": "x0 + 1",
            "sympy_meta": {
                "accepted": False,
                "numeric_fidelity_ok": True,
                "unit_admissibility": invalid_seed_certificate,
            },
        },
        final_polish_summary={
            "status": "success",
            "seed_expr": "x0 + 1",
            "seed_unit_admissibility": invalid_seed_certificate,
            "recommended": {
                "expr": "x0",
                "unit_admissibility": valid_certificate,
            },
            "strict_pareto": [
                {
                    "expr": "2*x0",
                    "unit_admissibility": valid_certificate,
                }
            ],
        },
        max_candidates=8,
        include_reservoir=False,
    )

    by_source = {candidate.source: candidate for candidate in candidates}
    assert by_source["final_polish:recommended"].metadata[
        "unit_admissibility"
    ] == valid_certificate
    assert by_source["final_polish:seed"].metadata["unit_admissibility"] == (
        invalid_seed_certificate
    )
    assert by_source["final_polish:strict_pareto"].metadata[
        "unit_admissibility"
    ] == valid_certificate


def test_named_coefficient_candidates_keep_distinct_values_and_cache_entries(tmp_path):
    import pandas as pd

    metadata_2 = _named_c_metadata(2.0)
    metadata_3 = _named_c_metadata(3.0)
    candidates = collect_final_candidates(
        stageB_data={
            "y_branch_artifacts": [
                {
                    "expr": "c*x0",
                    "label": "two",
                    "metadata": {"coefficient_metadata": metadata_2},
                },
                {
                    "expr": "c*x0",
                    "label": "three",
                    "metadata": {"coefficient_metadata": metadata_3},
                },
            ]
        },
        final_polish_summary=None,
        max_candidates=4,
        include_reservoir=False,
    )
    assert len(candidates) == 2
    assert [
        candidate.metadata["coefficient_metadata"]["records"][0]["value"]
        for candidate in candidates
    ] == [2.0, 3.0]

    data_path = tmp_path / "named_coefficients.csv"
    pd.DataFrame(
        {"x0": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}
    ).to_csv(data_path, index=False)
    spec = SliceSpec(0, 0, 2, 2, 4)
    cache = CommitteeEvalCache()
    two = evaluate_candidate_on_slice_cached(
        candidates[0], filepath=data_path, spec=spec, cache=cache
    )
    three = evaluate_candidate_on_slice_cached(
        candidates[1], filepath=data_path, spec=spec, cache=cache
    )
    assert two.val_mse == 0.0
    assert three.val_mse > 0.0
    assert cache.stats()["misses"] == 2
    assert cache.stats()["entries"] == 2


def test_parallel_committee_scores_named_coefficient_metadata(tmp_path):
    import pandas as pd

    data_path = tmp_path / "named_coefficients.csv"
    pd.DataFrame(
        {"x0": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}
    ).to_csv(data_path, index=False)
    decision = run_final_committee_audit(
        filepath=data_path,
        stageB_data={
            "y_branch_artifacts": [
                {
                    "expr": "c*x0",
                    "label": "two",
                    "complexity": 1.0,
                    "metadata": {
                        "coefficient_metadata": _named_c_metadata(2.0)
                    },
                },
                {
                    "expr": "c*x0",
                    "label": "three",
                    "complexity": 1.0,
                    "metadata": {
                        "coefficient_metadata": _named_c_metadata(3.0)
                    },
                },
            ]
        },
        final_polish_summary=None,
        mode="committee_gated",
        n_slices=1,
        start_slice=0,
        ndata_train=2,
        ndata_val=2,
        max_candidates=4,
        witness_parallelism=2,
    )
    assert decision.status == "success"
    assert decision.recommended_expr == "c*x0"
    recommended = next(
        row["candidate"]
        for row in decision.candidate_summary
        if row["candidate"]["candidate_id"] == decision.recommended_id
    )
    assert recommended["metadata"]["coefficient_metadata"]["records"][0][
        "value"
    ] == 2.0


def test_collect_final_candidates_rejects_unit_invalid_stagec_even_if_numeric_accepted():
    candidates = collect_final_candidates(
        stageB_data={
            "y_expr_str": "x0 + x0/x1",
            "sympy_meta": {
                "accepted": True,
                "units_checked": True,
                "units_ok": False,
                "unit_admissibility": {
                    "checked": True,
                    "valid": False,
                },
            },
        },
        final_polish_summary=None,
        max_candidates=8,
        include_reservoir=False,
    )

    assert candidates == []


def test_collect_final_candidates_rejects_incomplete_checked_unit_certificate():
    candidates = collect_final_candidates(
        stageB_data={
            "y_expr_str": "x0",
            "sympy_meta": {
                "accepted": True,
                "units_checked": True,
                "units_ok": None,
                "unit_admissibility": {"checked": True, "valid": None},
            },
        },
        final_polish_summary=None,
        max_candidates=8,
        include_reservoir=False,
    )

    assert candidates == []


def test_final_committee_prefers_exact_seed_over_decimal_tie(tmp_path):
    data_path = tmp_path / "toy.csv"
    import pandas as pd

    pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0], "y": [0.5, 1.0, 1.5, 2.0]}).to_csv(
        data_path,
        index=False,
    )

    decision = run_final_committee_audit(
        filepath=data_path,
        stageB_data=None,
        final_polish_summary={
            "seed_expr": "x0/2",
            "recommended": {"expr": "0.5*x0", "complexity": 1.0, "n_free_params": 1},
        },
        mode="committee_gated",
        n_slices=1,
        start_slice=0,
        ndata_train=2,
        ndata_val=2,
        max_candidates=4,
        noise_floor_raw=1.0e-6,
        witness_parallelism=2,
    )

    assert decision.recommended_expr == "x0/2"
    assert decision.config["witness_parallelism"] == 2
    assert decision.config["witness_executor"]["parallelism"] == 2


def _no_safe_coe_report():
    return {
        "metadata": {"dataset": "/tmp/pb061_data.csv"},
        "final_polish": {
            "status": "no_safe_unit_valid_replacement",
            "recommended": None,
        },
        "final_selection": {
            "source": "stageB",
            "applied": False,
            "eligible_for_success": False,
            "expr": "x0 + x0/x1",
        },
        "truth_eval": {
            "success": False,
            "reason": "final_selection_ineligible",
        },
    }


def _successful_incumbent_coe_summary(*, certified: bool):
    metadata = {}
    if certified:
        metadata["unit_admissibility"] = {"checked": True, "valid": True}
    return {
        "mode": "committee_gated",
        "status": "success",
        "recommended_id": "c000",
        "incumbent_id": "c000",
        "n_slices": 3,
        "config": {"selection_basis": "noise_tied_complexity"},
        "candidate_summary": [
            {
                "candidate": {
                    "candidate_id": "c000",
                    "expr": "x0 + x0/x1",
                    "source": "final_polish:seed",
                    "metadata": metadata,
                },
                "n_success": 3,
                "median_val_mse": 1.0e-6,
                "mean_val_mse": 1.0e-6,
                "noise_tied_with_best": True,
            }
        ],
    }


def test_coe_cannot_repromote_no_safe_incumbent_without_unit_certificate(tmp_path):
    report_path = tmp_path / "pb061.report.json"
    report_path.write_text(json.dumps(_no_safe_coe_report()))
    summary = _successful_incumbent_coe_summary(certified=False)

    selection = _apply_coe_final_adjudication(str(report_path), summary)

    assert selection is None
    report = json.loads(report_path.read_text())
    assert report["final_selection"]["eligible_for_success"] is False
    assert report["truth_eval"]["success"] is False
    assert report["coe_final_adjudication"]["status"] == "blocked"
    assert report["coe_final_adjudication"]["reason"] == (
        "coe_selection_lacks_unit_admissibility_certificate"
    )


def test_coe_cannot_bypass_stagec_unit_failure_without_candidate_certificate(tmp_path):
    report = _no_safe_coe_report()
    report.pop("final_polish")
    report["stageC"] = {
        "certified": False,
        "symbolic_status": "unit_invalid",
        "units_checked": True,
        "units_ok": False,
        "unit_admissibility": {"checked": True, "valid": False},
    }
    report_path = tmp_path / "pb061.report.json"
    report_path.write_text(json.dumps(report))
    summary = _successful_incumbent_coe_summary(certified=False)

    selection = _apply_coe_final_adjudication(str(report_path), summary)

    assert selection is None
    updated = json.loads(report_path.read_text())
    assert updated["coe_final_adjudication"]["status"] == "blocked"
    assert updated["final_selection"]["eligible_for_success"] is False


def test_coe_cannot_bypass_unavailable_required_raw_unit_check(tmp_path):
    report = _no_safe_coe_report()
    report.pop("final_polish")
    report["stageC"] = {
        "certified": False,
        "symbolic_status": "uncertified_expression",
        "units_checked": False,
        "units_ok": None,
        "unit_admissibility": {
            "checked": False,
            "valid": None,
            "code": "expression_unavailable",
            "coordinate_space": "raw",
        },
        "sympy_meta": {"kind": "unit_check_expression_unavailable"},
    }
    report_path = tmp_path / "pb061.report.json"
    report_path.write_text(json.dumps(report))
    summary = _successful_incumbent_coe_summary(certified=False)

    selection = _apply_coe_final_adjudication(str(report_path), summary)

    assert selection is None
    updated = json.loads(report_path.read_text())
    assert updated["coe_final_adjudication"]["status"] == "blocked"
    assert updated["coe_final_adjudication"]["reason"] == (
        "coe_selection_lacks_unit_admissibility_certificate"
    )


def test_coe_never_applies_candidate_with_explicit_invalid_unit_certificate(tmp_path):
    report_path = tmp_path / "ordinary.report.json"
    report_path.write_text(json.dumps({"metadata": {"dataset": "/tmp/toy.csv"}}))
    summary = _successful_incumbent_coe_summary(certified=False)
    summary["candidate_summary"][0]["candidate"]["metadata"][
        "unit_admissibility"
    ] = {"checked": True, "valid": False, "reason": "unit mismatch"}

    selection = _apply_coe_final_adjudication(str(report_path), summary)

    assert selection is None
    updated = json.loads(report_path.read_text())
    assert updated["coe_final_adjudication"]["reason"] == (
        "coe_selection_has_invalid_unit_admissibility_certificate"
    )


def test_certified_coe_incumbent_supersedes_no_safe_truth_marker(
    monkeypatch,
    tmp_path,
):
    import nestynet_sr.sr_search.truth_eval as truth_eval_mod

    monkeypatch.setattr(
        truth_eval_mod,
        "evaluate_canary",
        lambda **_kwargs: {"success": True, "rmse_rel": 0.0},
    )
    report_path = tmp_path / "pb061.report.json"
    report_path.write_text(json.dumps(_no_safe_coe_report()))
    summary = _successful_incumbent_coe_summary(certified=True)

    selection = _apply_coe_final_adjudication(str(report_path), summary)

    assert selection is not None
    assert selection["eligible_for_success"] is True
    report = json.loads(report_path.read_text())
    assert report["final_selection"]["eligible_for_success"] is True
    assert report["final_selection"]["unit_admissibility"] == {
        "checked": True,
        "valid": True,
    }
    assert report["truth_eval"]["success"] is True
    assert report["truth_eval"]["source"] == "coe_final_adjudication"
    assert report["truth_eval_pre_coe"]["reason"] == "final_selection_ineligible"


def test_coe_adjudication_uses_selected_candidate_coefficient_metadata(
    monkeypatch,
    tmp_path,
):
    import nestynet_sr.sr_search.truth_eval as truth_eval_mod

    truth_calls = []

    def fake_evaluate_canary(**kwargs):
        truth_calls.append(kwargs)
        return {"success": True, "rmse_rel": 0.0}

    monkeypatch.setattr(truth_eval_mod, "evaluate_canary", fake_evaluate_canary)
    report_path = tmp_path / "named.report.json"
    report_path.write_text(
        json.dumps(
            {
                "metadata": {"dataset": "/tmp/pb000_named.csv"},
                "stageB": {"coefficient_metadata": _named_c_metadata(2.0)},
            }
        )
    )
    selected_metadata = _named_c_metadata(3.0)
    summary = {
        "mode": "committee_gated",
        "status": "success",
        "recommended_id": "c001",
        "incumbent_id": "c000",
        "n_slices": 1,
        "config": {"selection_basis": "median_loss"},
        "candidate_summary": [
            {
                "candidate": {
                    "candidate_id": "c001",
                    "expr": "c*x0",
                    "source": "stageB_y_branch",
                    "metadata": {
                        "coefficient_metadata": selected_metadata,
                    },
                },
                "n_success": 1,
                "median_val_mse": 0.0,
                "mean_val_mse": 0.0,
            }
        ],
    }

    selection = _apply_coe_final_adjudication(str(report_path), summary)

    assert selection is not None
    assert selection["coefficient_metadata"] == selected_metadata
    assert truth_calls[0]["symbol_values"] == {"c": 3.0}
    updated = json.loads(report_path.read_text())
    assert updated["final_selection"]["coefficient_metadata"] == selected_metadata


def test_y_branch_artifact_persists_candidate_specific_metadata(monkeypatch):
    from types import SimpleNamespace

    import nestynet_sr.run_sr_stageb_utils as stageb_utils

    monkeypatch.setattr(
        stageb_utils,
        "_stageB_candidate_metrics",
        lambda *_args, **_kwargs: {
            "num_nn": 0,
            "original_y_val_loss": 0.0,
            "val_loss": 0.0,
            "complexity_score": 1.0,
            "raw_family": None,
            "raw_protected_family": False,
            "raw_generic_approximant": False,
            "inverse_y_transform_wrapped": False,
            "accepted_labels": [],
        },
    )
    metadata = _named_c_metadata(3.0)
    state = SimpleNamespace(
        y_expr_raw_str="c*x0",
        y_expr_str="c*x0",
        coefficient_metadata=metadata,
        coefficient_metadata_by_dataset=None,
    )

    artifact = stageb_utils._stageB_y_branch_artifact(
        state,
        y_name="identity",
        rank=0,
    )

    assert artifact["metadata"]["coefficient_metadata"] == metadata


def test_float_literal_count_catches_scientific_notation_without_variables():
    assert _nontrivial_float_literal_count("1e-5*x0 + x1") == 1
    assert _nontrivial_float_literal_count("x1 + x2") == 0


def test_reservoir_support_counts_unique_sources_not_duplicates():
    reservoir = ProposalReservoir(max_candidates=4)
    assert reservoir.add_expr("x0 + x1", source="slice1")
    assert reservoir.add_expr("x0 + x1", source="slice1")
    assert reservoir.add_expr("x0 + x1", source="slice2")

    row = reservoir.to_dict()["candidates"][0]
    assert row["support_count"] == 2
    assert row["sources"] == ["slice1", "slice2"]

    merged = merge_proposal_reservoir_payloads(
        [
            {
                "source": "report_a",
                "candidates": [
                    {
                        "expr": "x0 + x1",
                        "source": "slice1",
                        "sources": ["slice1", "slice1", "slice3"],
                        "support_count": 3,
                    }
                ],
            }
        ]
    )
    merged_row = merged["candidates"][0]
    assert merged_row["support_count"] == 2
    assert merged_row["sources"] == ["slice1", "slice3"]


def test_stagea_proposal_reservoir_dedupes_portable_split_records():
    reservoir = StageAProposalReservoir(max_candidates=4)
    assert reservoir.add_proposal(
        kind="split_partition",
        payload={"kind": "mul", "group1": [0, 1], "group2": [1, 2]},
        source="slice1",
        score=0.9,
    )
    assert reservoir.add_proposal(
        kind="split_partition",
        payload={"group2": [1, 2], "group1": [0, 1], "kind": "mul"},
        source="slice2",
        score=0.8,
    )

    row = reservoir.to_dict()["candidates"][0]
    assert row["support_count"] == 2
    assert row["sources"] == ["slice1", "slice2"]


def test_stagea_compound_replay_key_ignores_source_confidence():
    reservoir = StageAProposalReservoir(max_candidates=4)
    replay_key = {
        "schema": "stageA_replay_v1",
        "proposal_class": "strict_visible_arity_reducing_compound",
        "problem_id": "pbReplay_full_stem",
        "Nxvars": 3,
        "parent_key": {
            "parent_effective_input_fps": ["var:0", "var:1", "var:2"],
            "parent_hole_context_fp": "hole",
        },
        "candidate_key": {
            "pattern": [1, 1, 0],
            "z_expr_payload": {"node": "mul", "args": [{"node": "var", "idx": 0}, {"node": "var", "idx": 1}]},
            "extra_input_selectors": [{"kind": "parent_input", "index": 2}],
            "old_arity": 3,
            "new_arity": 2,
            "proposal_kind": "monomial",
        },
    }
    assert reservoir.add_proposal(
        kind="compound_coordinate_replay",
        payload={"replay_key": replay_key, "source_evidence": {"confidence": 0.72}},
        source="slice1",
        score=0.72,
    )
    assert reservoir.add_proposal(
        kind="compound_coordinate_replay",
        payload={"replay_key": replay_key, "source_evidence": {"confidence": 0.98}},
        source="slice2",
        score=0.98,
    )

    row = reservoir.to_dict()["candidates"][0]
    assert row["support_count"] == 2
    assert row["sources"] == ["slice1", "slice2"]
    assert row["score"] == 0.98


def test_build_stagea_proposal_reservoir_from_move_records():
    payload = build_stageA_proposal_reservoir(
        stageA_data={
            "stageA_status": "split",
            "y_op_name": "identity",
            "stageA_move_records": [
                {
                    "seq": 1,
                    "move_kind": "separability_split",
                    "risk_tags": ["split_accept", "overlap_split"],
                    "candidate_loss": 1.0e-4,
                    "details": {
                        "op": "mul",
                        "group1": [0, 1],
                        "group2": [1, 2],
                        "has_overlap": True,
                        "split_score": 0.95,
                    },
                }
            ],
        },
        source="reference",
    )

    assert payload["kind"] == "stageA_proposal_reservoir"
    assert payload["total_unique"] == 1
    row = payload["candidates"][0]
    assert row["kind"] == "split_partition"
    assert row["payload"]["kind"] == "mul"
    assert row["payload"]["has_overlap"] is True


def test_merge_stagea_proposal_reservoir_support_counts_sources():
    a = StageAProposalReservoir(max_candidates=4)
    b = StageAProposalReservoir(max_candidates=4)
    assert a.add_proposal(kind="y_branch", payload={"y_transform": "sqrt"}, source="slice1")
    assert b.add_proposal(kind="y_branch", payload={"y_transform": "sqrt"}, source="slice2")

    merged = merge_stageA_proposal_reservoir_payloads([a.to_dict(), b.to_dict()])

    row = merged["candidates"][0]
    assert row["support_count"] == 2
    assert row["sources"] == ["slice1", "slice2"]


def test_stagea_terminal_proposals_feed_final_expression_reservoir():
    stagea = StageAProposalReservoir(max_candidates=4)
    assert stagea.add_proposal(
        kind="terminal_expression",
        payload={"expr": "x0 + x1"},
        source="slice1",
        loss=1.0e-6,
    )
    assert stagea.add_proposal(
        kind="split_partition",
        payload={"kind": "add", "group1": [0], "group2": [1]},
        source="slice2",
    )

    expr_payload = stageA_terminal_proposals_as_expression_reservoir(stagea.to_dict())

    assert [row["expr"] for row in expr_payload["candidates"]] == ["x0 + x1"]
    assert expr_payload["candidates"][0]["sources"] == ["stageA_terminal:slice1"]


def test_stagea_y_branch_names_extracts_supported_nonidentity_branches():
    stagea = StageAProposalReservoir(max_candidates=8)
    assert stagea.add_proposal(kind="y_branch", payload={"y_transform": "sqrt"}, source="slice1")
    assert stagea.add_proposal(kind="y_branch", payload={"y_transform": "sqrt"}, source="slice2")
    assert stagea.add_proposal(kind="y_branch", payload={"y_transform": "log"}, source="slice1")
    assert stagea.add_proposal(kind="y_branch", payload={"y_transform": "identity"}, source="slice3")
    assert stagea.add_proposal(kind="split_partition", payload={"kind": "mul"}, source="slice4")

    assert stageA_y_branch_names_from_proposal_reservoir(stagea.to_dict()) == ["sqrt", "log"]
    assert stageA_y_branch_names_from_proposal_reservoir(
        stagea.to_dict(),
        min_support=2,
    ) == ["sqrt"]
