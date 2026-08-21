# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import json
from types import SimpleNamespace

import torch

from scripts.summarize_aifeyn_results import summarize_results
from nestynet_sr.run_SR import (
    _final_selection_from_report,
    _format_final_selection_report,
    _format_final_polish_report,
    _format_truth_eval_summary,
    _select_final_polish_seed,
    _update_report_with_final_polish,
    write_json_report,
)
from nestynet_sr.run_sr_final_polish import (
    _certify_final_polish_recommendation_units,
    _run_final_pareto_polish,
)
from nestynet_sr.run_sr_reports import _refresh_final_selection_truth_eval
from nestynet_sr.sr_core.bridges import FixedConst, MulNode, Var, build_composite_from_ast
from nestynet_sr.sr_core.coefficient_metadata import (
    collect_coefficient_metadata,
    empty_coefficient_metadata,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec


def _length_units_spec():
    unit_system = UnitSystem(("L", "T"))
    return UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dim({"L": 1}),),
        y_dim=unit_system.dim({"L": 1}),
    )


def _guarded_polish_args():
    return SimpleNamespace(
        final_polish=True,
        final_polish_subprocess=True,
        final_polish_worker_max_seconds=60.0,
        final_polish_worker_mem_fraction=0.5,
        final_polish_max_rows=128,
        final_polish_val_fraction=0.2,
        final_polish_max_candidates=16,
        final_polish_max_seconds=10.0,
        final_polish_full_dataset_snap=False,
        enforce_units=True,
        units_policy="free_const_only",
        nn_units_semantics="unknown",
    )


def _write_linear_csv(path, *, scale):
    rows = ["x0,y"]
    for index in range(1, 65):
        x0 = float(index) / 16.0
        rows.append(f"{x0:.16g},{scale * x0:.16g}")
    path.write_text("\n".join(rows) + "\n")


def _guarded_polish_stageb_data(expr, coefficient_metadata):
    return {
        "y_selected": "identity",
        "phi_expr_str": expr,
        "y_expr_str": expr,
        "sympy_meta": {
            "accepted": True,
            "parse_success": True,
            "kind": "parsed",
            "max_err": 0.0,
            "tol": 1.0e-12,
        },
        "coefficient_metadata": coefficient_metadata,
        "num_nn_atoms": 0,
    }


def _run_guarded_polish_case(
    tmp_path,
    *,
    name,
    scale,
    expr,
    units_payload,
    coefficient_metadata,
):
    filepath = tmp_path / f"{name}_data.csv"
    _write_linear_csv(filepath, scale=scale)
    return _run_final_pareto_polish(
        args=_guarded_polish_args(),
        filepath=str(filepath),
        filepaths=[str(filepath)],
        report_path=str(tmp_path / f"{name}.report.json"),
        results_dir=str(tmp_path),
        base_filename=name,
        stageB_data=_guarded_polish_stageb_data(expr, coefficient_metadata),
        seed=1234,
        units_payload=units_payload,
        noise_sigma_y=0.0,
    )


def test_guarded_final_polish_preserves_five_axis_unitless_basis(tmp_path):
    unit_system = UnitSystem(("L", "T", "M", "I", "Θ"))
    zero = unit_system.dimless()
    metadata = empty_coefficient_metadata(dimension_basis=unit_system.base)

    summary = _run_guarded_polish_case(
        tmp_path,
        name="unitless_guarded",
        scale=1.0,
        expr="x0",
        units_payload={
            "unit_system": unit_system,
            "x_dims": (zero,),
            "y_dim": zero,
            "free_const_dims": {},
            "free_const_scope": {},
            "fixed_const_dims": {},
            "fixed_const_values": {},
            "fixed_const_mode": "strict",
        },
        coefficient_metadata=metadata,
    )

    assert summary["guarded_subprocess"] is True
    assert summary["status"] == "success"
    assert summary["coefficient_metadata"]["dimension_basis"] == list(
        unit_system.base
    )
    assert summary["seed_unit_admissibility"]["valid"] is True
    assert summary["unit_admissibility"]["valid"] is True


def test_guarded_final_polish_preserves_unitful_fixed_constant(tmp_path):
    unit_system = UnitSystem(("L", "T"))
    zero = unit_system.dimless()
    length = unit_system.dim({"L": 1})
    root = MulNode(FixedConst("c", value=2.0), Var(0))
    model = build_composite_from_ast(root, dtype=torch.float64)
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(zero,),
        y_dim=length,
        fixed_const_dims={"c": length},
        fixed_const_values={"c": 2.0},
    )
    metadata = collect_coefficient_metadata(root, model, spec)

    summary = _run_guarded_polish_case(
        tmp_path,
        name="unitful_constant_guarded",
        scale=2.0,
        expr="c*x0",
        units_payload={
            "unit_system": unit_system,
            "x_dims": (zero,),
            "y_dim": length,
            "free_const_dims": {},
            "free_const_scope": {},
            "fixed_const_dims": {"c": length},
            "fixed_const_values": {"c": 2.0},
            "fixed_const_mode": "strict",
        },
        coefficient_metadata=metadata,
    )

    assert summary["guarded_subprocess"] is True
    assert summary["status"] == "success"
    assert summary["coefficient_metadata"]["records"][0]["symbol"] == "c"
    assert summary["seed_unit_admissibility"]["valid"] is True
    assert summary["unit_admissibility"]["valid"] is True


def test_final_polish_refuses_bad_pretty_print_phi_seed():
    seed, space, reason = _select_final_polish_seed(
        {
            "y_selected": "identity",
            "phi_expr_str": "0.414227371571348*(bad pretty expression)",
            "sympy_meta": {
                "accepted": False,
                "parse_success": True,
                "kind": "bad_pretty_print",
                "max_err": 1.5e-2,
                "tol": 3.0e-8,
            },
        }
    )

    assert seed is None
    assert space is None
    assert "pretty-print failed" in reason


def test_final_polish_accepts_verified_identity_phi_seed():
    seed, space, reason = _select_final_polish_seed(
        {
            "y_selected": "identity",
            "phi_expr_str": "x0*x1",
            "sympy_meta": {
                "accepted": True,
                "parse_success": True,
                "kind": "parsed",
                "max_err": 1.0e-14,
                "tol": 1.0e-8,
            },
        }
    )

    assert seed == "x0*x1"
    assert space == "phi_identity"
    assert reason is None


def test_final_polish_rechecks_the_post_snap_recommendation_units():
    rec = SimpleNamespace(expr="x0", is_recommended=True)
    result = SimpleNamespace(
        recommended=rec,
        selection_status="selected",
        selection_reason=None,
        warnings=[],
    )

    certificate = _certify_final_polish_recommendation_units(
        result,
        variable_names=("x0",),
        units_spec=_length_units_spec(),
        require_valid=True,
    )

    assert certificate["checked"] is True
    assert certificate["valid"] is True
    assert result.recommended is rec
    assert result.selection_status == "selected"


def test_final_polish_fails_closed_if_post_snap_recommendation_is_unit_invalid():
    rec = SimpleNamespace(expr="x0 + 1", is_recommended=True)
    result = SimpleNamespace(
        recommended=rec,
        selection_status="selected",
        selection_reason=None,
        warnings=[],
    )

    certificate = _certify_final_polish_recommendation_units(
        result,
        variable_names=("x0",),
        units_spec=_length_units_spec(),
        require_valid=True,
    )

    assert certificate["valid"] is False
    assert result.recommended is None
    assert rec.is_recommended is False
    assert result.selection_status == "no_safe_unit_valid_replacement"
    assert result.selection_reason == certificate["reason"]


def test_final_polish_refuses_unverified_y_seed_too():
    seed, space, reason = _select_final_polish_seed(
        {
            "y_selected": "sqrt",
            "y_expr_str": "sqrt(bad_phi)",
            "sympy_meta": {
                "accepted": False,
                "parse_success": False,
                "error": "parse failed",
            },
        }
    )

    assert seed is None
    assert space is None
    assert "failed to parse" in reason


def test_final_polish_refuses_accepted_leaf_placeholder_seed():
    seed, space, reason = _select_final_polish_seed(
        {
            "y_selected": "identity",
            "phi_expr_str": "x0*leaf1(x1, x2)",
            "sympy_meta": {
                "accepted": True,
                "parse_success": True,
                "kind": "parsed",
            },
        }
    )

    assert seed is None
    assert space is None
    assert "unresolved NN atoms or leaf functions" in reason


def test_json_report_marks_leaf_placeholder_unresolved_and_skips_truth_eval(tmp_path):
    report_path = tmp_path / "toy.report.json"

    write_json_report(
        filepath=str(tmp_path / "toy_data.csv"),
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=123,
        walltime=0.0,
        stageB_data={
            "phi_expr_str": "x0*leaf1(x1, x2)",
            "sympy_meta": {
                "accepted": True,
                "parse_success": True,
                "kind": "parsed",
            },
        },
        enable_truth_eval=True,
    )

    report = json.loads(report_path.read_text())
    assert report["stageC"]["symbolic_status"] == "unresolved_nn"
    assert report["stageC"]["sympy_meta"]["accepted"] is False
    assert report["stageC"]["sympy_meta"]["kind"] == "unresolved_symbolic_leaves"
    assert report["truth_eval"]["success"] is False
    assert report["truth_eval"]["skipped"] is True
    assert report["truth_eval"]["reason"] == "unresolved_symbolic_leaves"


def test_json_report_skips_truth_eval_for_bad_pretty_print(tmp_path):
    report_path = tmp_path / "toy.report.json"

    write_json_report(
        filepath=str(tmp_path / "toy_data.csv"),
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=123,
        walltime=0.0,
        stageB_data={
            "y_expr_str": "0.123*(bad_pretty_but_evaluable)",
            "sympy_meta": {
                "accepted": False,
                "parse_success": True,
                "kind": "bad_pretty_print",
                "max_err": 23.6,
                "tol": 3.9e-8,
            },
            "num_nn_atoms": 0,
        },
        enable_truth_eval=True,
    )

    report = json.loads(report_path.read_text())
    assert report["stageC"]["symbolic_status"] == "uncertified_expression"
    assert report["stageC"]["certified"] is False
    assert "pretty-print failed" in report["stageC"]["certification_reason"]
    assert report["truth_eval"]["success"] is False
    assert report["truth_eval"]["skipped"] is True
    assert report["truth_eval"]["reason"] == "stagec_expression_uncertified"


def test_json_report_marks_unit_invalid_stagec_and_skips_truth_eval(tmp_path):
    report_path = tmp_path / "toy.report.json"
    certificate = {
        "checked": True,
        "valid": False,
        "checker": "sympy_units_v1",
        "reason": "addends have incompatible dimensions",
        "expression_space": "phi_and_y",
    }

    write_json_report(
        filepath=str(tmp_path / "toy_data.csv"),
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=123,
        walltime=0.0,
        stageB_data={
            "y_expr_str": "x0 + x0/x1",
            "sympy_meta": {
                "accepted": False,
                "numeric_fidelity_ok": True,
                "units_checked": True,
                "units_ok": False,
                "units_reason": certificate["reason"],
                "unit_admissibility": certificate,
                "kind": "no_unit_valid_candidate",
            },
            "num_nn_atoms": 0,
        },
        enable_truth_eval=True,
    )

    report = json.loads(report_path.read_text())
    assert report["stageC"]["symbolic_status"] == "unit_invalid"
    assert report["stageC"]["certified"] is False
    assert report["stageC"]["unit_admissibility"] == certificate
    assert report["truth_eval"]["reason"] == "stagec_expression_uncertified"


def test_truth_eval_summary_renders_noiseless_error():
    rendered = _format_truth_eval_summary(
        {
            "success": True,
            "rmse_abs": 1.2e-14,
            "rmse_rel": 3.4e-15,
            "max_abs_err": 5.6e-14,
            "max_rel_err": 7.8e-15,
            "frac_valid": 1.0,
            "n_valid": 10000,
            "n_total": 10000,
        }
    )

    assert "Noiseless Ground Truth Check" in rendered
    assert "rmse_abs=1.2000e-14" in rendered
    assert "rmse_rel=3.4000e-15" in rendered
    assert "valid_points=100.0% (10000/10000)" in rendered


def test_final_polish_report_includes_truth_eval_summary():
    rendered = _format_final_polish_report(
        {
            "status": "success",
            "recommended": {
                "expr": "x0*x1",
                "val_mse": 1.0e-6,
                "complexity": 2.0,
            },
            "truth_eval": {
                "success": True,
                "rmse_abs": 0.0,
                "rmse_rel": 0.0,
                "max_abs_err": 0.0,
                "max_rel_err": 0.0,
                "frac_valid": 1.0,
                "n_valid": 10,
                "n_total": 10,
            },
        }
    )

    assert "Final Polish Ground Truth Check" in rendered
    assert "rmse_abs=0.0000e+00" in rendered


def test_final_selection_report_renders_post_coe_truth_at_final_end():
    rendered = _format_final_selection_report(
        {
            "source": "coe_committee",
            "mode": "committee_gated",
            "candidate_id": "c003",
            "candidate_source": "final_polish:strict_pareto",
            "selection_basis": "noise_tied_complexity",
            "expr": "sqrt(2)/(2*sqrt(pi)*x0)",
            "truth_eval": {
                "success": True,
                "rmse_abs": 3.0e-17,
                "rmse_rel": 9.0e-16,
                "max_abs_err": 4.0e-16,
                "max_rel_err": 1.0e-15,
                "frac_valid": 1.0,
                "n_valid": 10000,
                "n_total": 10000,
            },
        }
    )

    assert rendered.startswith("=== Final Selected Result ===")
    assert "source: coe_committee" in rendered
    assert "candidate_id: c003" in rendered
    assert "expr: sqrt(2)/(2*sqrt(pi)*x0)" in rendered
    assert "Final Selected Ground Truth Check" in rendered
    assert "rmse_rel=9.0000e-16" in rendered


def test_final_selection_from_report_uses_final_selection_before_legacy_truth():
    selection = _final_selection_from_report(
        {
            "truth_eval": {
                "success": True,
                "rmse_rel": 2.0e-6,
            },
            "truth_eval_pre_coe": {
                "success": True,
                "rmse_rel": 2.0e-6,
            },
            "final_selection": {
                "source": "coe_committee",
                "expr": "sqrt(2)/(2*sqrt(pi)*x0)",
                "truth_eval": {
                    "success": True,
                    "rmse_rel": 9.0e-16,
                },
            },
        }
    )

    assert selection["source"] == "coe_committee"
    assert selection["expr"] == "sqrt(2)/(2*sqrt(pi)*x0)"
    assert selection["truth_eval"]["rmse_rel"] == 9.0e-16


def test_final_selection_from_report_falls_back_to_stageb_expression():
    selection = _final_selection_from_report(
        {
            "stageC": {
                "y_expr_str": "x0*x1",
            },
            "truth_eval": {
                "success": True,
                "rmse_rel": 0.0,
            },
        }
    )

    assert selection["source"] == "stageB"
    assert selection["expr"] == "x0*x1"
    assert selection["truth_eval"]["rmse_rel"] == 0.0


def test_final_polish_update_promotes_truth_eval_to_final_selection(tmp_path):
    report_path = tmp_path / "toy.report.json"
    report_path.write_text(
        json.dumps(
            {
                "metadata": {"dataset": "toy.csv"},
                "truth_eval": {
                    "success": True,
                    "rmse_abs": 1.0e-3,
                    "rmse_rel": 1.0e-2,
                },
            }
        )
    )

    _update_report_with_final_polish(
        str(report_path),
        {
            "status": "success",
            "recommended": {
                "expr": "pi*x0",
                "unit_admissibility": {"checked": True, "valid": True},
            },
            "truth_eval": {
                "success": True,
                "rmse_abs": 0.0,
                "rmse_rel": 0.0,
            },
        },
    )

    report = json.loads(report_path.read_text())
    assert report["truth_eval_pre_final_polish"]["rmse_rel"] == 1.0e-2
    assert report["truth_eval"]["source"] == "final_polish"
    assert report["truth_eval"]["expr"] == "pi*x0"
    assert report["truth_eval"]["rmse_rel"] == 0.0
    assert report["final_selection"]["source"] == "final_polish"
    assert report["final_selection"]["unit_admissibility"] == {
        "checked": True,
        "valid": True,
    }


def test_final_polish_no_safe_replacement_preserves_diagnostic_incumbent(tmp_path):
    report_path = tmp_path / "toy.report.json"
    original_truth = {
        "success": True,
        "rmse_abs": 5.0e-3,
        "rmse_rel": 6.0e-3,
    }
    report_path.write_text(
        json.dumps(
            {
                "metadata": {"dataset": "toy.csv"},
                "stageC": {
                    "y_expr_str": "x0 + 0.5*x0/x1",
                    "unit_admissibility": {
                        "checked": True,
                        "valid": False,
                        "reason": "additive unit mismatch",
                    },
                },
                "truth_eval": original_truth,
            }
        )
    )

    _update_report_with_final_polish(
        str(report_path),
        {
            "status": "no_safe_unit_valid_replacement",
            "reason": "admissible recommendation worsens raw seed validation loss",
            "recommended": None,
            "needs_escalation": True,
            "escalation_reason": "final_polish_no_safe_unit_valid_replacement",
        },
    )

    report = json.loads(report_path.read_text())
    assert report["truth_eval"]["success"] is False
    assert report["truth_eval"]["reason"] == "final_selection_ineligible"
    assert report["truth_eval_diagnostic_incumbent"] == original_truth
    assert "truth_eval_pre_final_polish" not in report
    selection = report["final_selection"]
    assert selection["source"] == "stageB"
    assert selection["expr"] == "x0 + 0.5*x0/x1"
    assert selection["applied"] is False
    assert selection["eligible_for_success"] is False
    assert selection["status"] == "no_safe_unit_valid_replacement"
    assert selection["truth_eval"] == original_truth
    assert selection["unit_admissibility"]["valid"] is False
    rendered = _format_final_selection_report(selection)
    assert rendered.startswith("=== Diagnostic Incumbent")
    assert "applied: False" in rendered


def test_final_polish_non_success_status_cannot_promote_report_truth(tmp_path):
    report_path = tmp_path / "toy.report.json"
    report_path.write_text(
        json.dumps(
            {
                "stageC": {"y_expr_str": "x0"},
                "truth_eval": {"success": True, "rmse_rel": 1.0e-3},
            }
        )
    )

    _update_report_with_final_polish(
        str(report_path),
        {
            "status": "no_safe_unit_valid_replacement",
            "recommended": {"expr": "exp(x0)"},
            "truth_eval": {"success": True, "rmse_rel": 1.0},
        },
    )

    report = json.loads(report_path.read_text())
    assert report["truth_eval"]["success"] is False
    assert report["truth_eval"]["reason"] == "final_selection_ineligible"
    assert report["truth_eval_diagnostic_incumbent"]["rmse_rel"] == 1.0e-3
    assert report["final_selection"]["expr"] == "x0"
    assert report["final_selection"]["eligible_for_success"] is False


def test_aifeyn_summary_does_not_count_diagnostic_incumbent_as_success(tmp_path):
    report_path = tmp_path / "pb061_II_11_17_data.report.json"
    report_path.write_text(
        json.dumps(
            {
                "metadata": {"walltime_hours": 1.0},
                "stageA": {"y_transform": "identity"},
                "stageB": {},
                "stageC": {"y_expr_str": "x0 + x0/x1"},
                "truth_eval": {
                    "success": True,
                    "rmse_abs": 0.0,
                    "rmse_rel": 0.0,
                },
                "final_polish": {
                    "status": "no_safe_unit_valid_replacement",
                },
                "final_selection": {
                    "source": "stageB",
                    "applied": False,
                    "eligible_for_success": False,
                    "reason": "no safe unit-valid replacement",
                    "expr": "x0 + x0/x1",
                },
            }
        )
    )

    rows = summarize_results(tmp_path)

    assert len(rows) == 1
    assert rows[0]["truth_success"] is False
    assert rows[0]["final_selection_eligible"] is False
    assert rows[0]["final_selection_ineligible_reason"] == (
        "no safe unit-valid replacement"
    )


def test_aifeyn_summary_recomputes_truth_for_effective_final_selection(
    monkeypatch,
    tmp_path,
):
    import nestynet_sr.sr_search.truth_eval as truth_eval_mod

    evaluated = []

    def fake_evaluate_canary(*, dataset_stem, discovered_expr_str, verbose=False):
        del dataset_stem, verbose
        evaluated.append(discovered_expr_str)
        return {
            "success": discovered_expr_str == "pi*x0",
            "rmse_abs": 0.0,
            "rmse_rel": 0.0,
        }

    monkeypatch.setattr(truth_eval_mod, "evaluate_canary", fake_evaluate_canary)
    report_path = tmp_path / "pb061_II_11_17_data.report.json"
    report_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "stageA": {"y_transform": "sqrt"},
                "stageB": {},
                "stageC": {"phi_expr_str": "x0"},
                "final_polish": {
                    "status": "no_safe_unit_valid_replacement",
                },
                "final_selection": {
                    "source": "coe_committee",
                    "applied": True,
                    "eligible_for_success": True,
                    "expr": "pi*x0",
                    "unit_admissibility": {"checked": True, "valid": True},
                },
            }
        )
    )

    rows = summarize_results(tmp_path, recompute_truth=True)

    assert evaluated == ["pi*x0"]
    assert rows[0]["expression"] == "pi*x0"
    assert rows[0]["truth_success"] is True
    assert rows[0]["truth_exact"] is True


def test_aifeyn_summary_repairs_stale_selected_expression_metric_by_default(
    monkeypatch,
    tmp_path,
):
    import nestynet_sr.sr_search.truth_eval as truth_eval_mod

    evaluated = []

    def fake_evaluate_canary(*, dataset_stem, discovered_expr_str, verbose=False):
        del dataset_stem, verbose
        evaluated.append(discovered_expr_str)
        return {
            "success": True,
            "rmse_abs": 2.0e-17,
            "rmse_rel": 3.0e-16,
        }

    monkeypatch.setattr(truth_eval_mod, "evaluate_canary", fake_evaluate_canary)
    (tmp_path / "pb001_I_6_2_data.report.json").write_text(
        json.dumps(
            {
                "metadata": {"dataset": "/data/noise_0.000/pb001.csv"},
                "stageA": {"y_transform": "identity"},
                "stageB": {},
                "stageC": {"y_expr_str": "legacy_approximant(x0)"},
                "truth_eval": {
                    "success": True,
                    "rmse_abs": 1.4e-6,
                    "rmse_rel": 5.4e-5,
                },
                "final_selection": {
                    "source": "statistical_selection",
                    "applied": True,
                    "eligible_for_success": True,
                    "expr": "exp(-x0**2)",
                },
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_results(tmp_path)

    assert evaluated == ["exp(-x0**2)"]
    assert rows[0]["truth_rmse_abs"] == 2.0e-17
    assert rows[0]["truth_exact"] is True


def test_aifeyn_summary_distinguishes_evaluation_success_from_exact_recovery(
    tmp_path,
):
    expr = "x0 + 1e-5"
    (tmp_path / "pb001_I_6_2_data.report.json").write_text(
        json.dumps(
            {
                "metadata": {"dataset": "/data/noise_0.000/pb001.csv"},
                "stageA": {"y_transform": "identity"},
                "stageB": {},
                "stageC": {"y_expr_str": expr},
                "truth_eval": {
                    "source": "statistical_selection",
                    "expr": expr,
                    "success": True,
                    "rmse_abs": 1.0e-6,
                    "rmse_rel": 2.0e-6,
                },
                "final_selection": {
                    "source": "statistical_selection",
                    "applied": True,
                    "eligible_for_success": True,
                    "expr": expr,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_results(tmp_path)

    assert rows[0]["truth_success"] is True
    assert rows[0]["truth_exact"] is False


def test_aifeyn_summary_does_not_fail_noise_limited_recovery(tmp_path):
    expr = "x0 + 1e-3"
    (tmp_path / "pb001_I_6_2_data.report.json").write_text(
        json.dumps(
            {
                "metadata": {"dataset": "/data/noise_0.010/pb001.csv"},
                "stageA": {"y_transform": "identity"},
                "stageB": {},
                "stageC": {"y_expr_str": expr},
                "truth_eval": {
                    "source": "statistical_selection",
                    "expr": expr,
                    "success": True,
                    "rmse_abs": 1.0e-3,
                    "rmse_rel": 2.0e-2,
                },
                "final_selection": {
                    "source": "statistical_selection",
                    "applied": True,
                    "eligible_for_success": True,
                    "expr": expr,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_results(tmp_path)

    assert rows[0]["noise_level"] == 0.01
    assert rows[0]["truth_success"] is True
    assert rows[0]["truth_exact"] is None


def test_refresh_final_selection_truth_eval_replaces_stale_metric(
    monkeypatch,
    tmp_path,
):
    import nestynet_sr.sr_search.truth_eval as truth_eval_mod

    evaluated = []

    def fake_evaluate_canary(*, dataset_stem, discovered_expr_str, verbose=False):
        del verbose
        evaluated.append((dataset_stem, discovered_expr_str))
        return {"success": True, "rmse_abs": 0.0, "rmse_rel": 0.0}

    monkeypatch.setattr(truth_eval_mod, "evaluate_canary", fake_evaluate_canary)
    report_path = tmp_path / "pb001.report.json"
    stale = {"success": True, "rmse_abs": 1.4e-6, "rmse_rel": 5.4e-5}
    report_path.write_text(
        json.dumps(
            {
                "metadata": {"dataset": "/data/pb001_I_6_2_data.csv"},
                "truth_eval": stale,
                "final_selection": {
                    "source": "statistical_selection",
                    "applied": True,
                    "eligible_for_success": True,
                    "expr": "exp(-x0**2)",
                },
            }
        ),
        encoding="utf-8",
    )

    refreshed = _refresh_final_selection_truth_eval(
        report_path,
        source="statistical_selection",
        preserve_as="truth_eval_pre_statistical_selection",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert evaluated == [("pb001_I_6_2_data", "exp(-x0**2)")]
    assert report["truth_eval_pre_statistical_selection"] == stale
    assert refreshed["source"] == "statistical_selection"
    assert refreshed["expr"] == "exp(-x0**2)"
    assert report["truth_eval"] == refreshed
    assert report["final_selection"]["truth_eval"] == refreshed


def test_refresh_final_selection_truth_eval_fails_soft_on_bad_metadata(
    monkeypatch,
    tmp_path,
):
    import nestynet_sr.sr_search.truth_eval as truth_eval_mod

    evaluated = []
    monkeypatch.setattr(
        truth_eval_mod,
        "evaluate_canary",
        lambda **kwargs: evaluated.append(kwargs),
    )
    report_path = tmp_path / "pb001.report.json"
    report_path.write_text(
        json.dumps(
            {
                "metadata": {"dataset": "/data/pb001_I_6_2_data.csv"},
                "final_selection": {
                    "source": "statistical_selection",
                    "applied": True,
                    "eligible_for_success": True,
                    "expr": "exp(-x0**2)",
                    "coefficient_metadata": {"schema": "broken"},
                },
            }
        ),
        encoding="utf-8",
    )

    refreshed = _refresh_final_selection_truth_eval(
        report_path,
        source="statistical_selection",
        preserve_as="truth_eval_pre_statistical_selection",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert evaluated == []
    assert refreshed["success"] is False
    assert refreshed["skipped"] is True
    assert refreshed["reason"] == "statistical_selection_truth_eval_error"
    assert refreshed["source"] == "statistical_selection"
    assert refreshed["expr"] == "exp(-x0**2)"
    assert report["final_selection"]["eligible_for_success"] is True
    assert report["final_selection"]["truth_eval"] == refreshed


def test_proposal_only_final_polish_does_not_replace_final_selection(tmp_path):
    report_path = tmp_path / "report.json"
    original = {"source": "stageC", "expr": "x0"}
    report_path.write_text(json.dumps({"final_selection": original}), encoding="utf-8")
    _update_report_with_final_polish(
        str(report_path),
        {
            "enabled": True,
            "status": "success",
            "proposal_only": True,
            "recommended": {"expr": "2*x0"},
            "all_candidates": [{"expr": "x0"}, {"expr": "2*x0"}],
        },
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["final_selection"] == original
    assert report["final_polish"]["proposal_only"] is True
    assert len(report["final_polish"]["all_candidates"]) == 2
