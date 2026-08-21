# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import numpy as np
import sympy as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

import nestynet_sr.sr_search.representation as representation
from nestynet_sr.sr_core.sympy_units import check_sympy_units
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.run_sr_reports import (
    _final_selection_from_report,
    _report_final_selection_eligibility,
    _select_final_polish_seed,
    _stagec_expression_is_verified,
)
from nestynet_sr.sr_search.stageB.main import _with_stagec_unit_certificates
from nestynet_sr.sr_search.representation import _unit_consistent_additive_prunings


def _length_target_spec():
    unit_system = UnitSystem(("L", "T"))
    return UnitsSpec(
        unit_system=unit_system,
        x_dims=(
            unit_system.dim({"L": 1}),
            unit_system.dim({"L": 1}),
            unit_system.dimless(),
        ),
        y_dim=unit_system.dim({"L": 1}),
    )


def _stagec_arrays():
    x0 = np.linspace(1.0, 2.0, 32)
    x1 = np.ones_like(x0)
    x2 = np.full_like(x0, np.pi / 2.0)
    xs = np.column_stack((x0, x1, x2))
    return xs, x0


def _pb061_spec():
    unit_system = UnitSystem(("L", "T", "M", "I", "Theta"))
    return UnitsSpec(
        unit_system=unit_system,
        x_dims=tuple(
            unit_system.dim(dim)
            for dim in (
                {},
                {"L": 2, "T": -2, "M": 1, "I": -1},
                {"I": 1},
                {},
                {"L": 3, "T": -2, "M": 1, "Theta": -1},
                {"L": -1, "Theta": 1},
            )
        ),
        y_dim=unit_system.dimless(),
    )


def test_stagec_selects_unit_valid_runner_up_before_ranking(monkeypatch):
    xs, ys = _stagec_arrays()

    def invalid_but_exact(_expr, _nxvars, **_kwargs):
        x0 = sp.Symbol("x0", positive=True)
        x1 = sp.Symbol("x1", positive=True)
        return x0 / x1

    monkeypatch.setattr(representation, "aggressive_simplify", invalid_but_exact)

    phi_str, y_str, meta = representation._sympy_simplify_expression(
        "x0*sin(x2)",
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=3,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=_length_target_spec(),
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=ys,
    )

    assert sp.simplify(sp.sympify(phi_str) - sp.sympify("x0*sin(x2)")) == 0
    assert y_str is None
    assert meta["accepted"] is True
    assert meta["units_ok"] is True
    assert meta["unit_reject_count"] >= 1
    assert meta["proposal_budget"]["requested_count"] == 1
    assert meta["proposal_budget"]["emitted"] == 1


def test_stagec_without_units_keeps_the_numeric_complexity_ranking(monkeypatch):
    xs, ys = _stagec_arrays()

    def simpler_exact_candidate(_expr, _nxvars, **_kwargs):
        x0 = sp.Symbol("x0", positive=True)
        x1 = sp.Symbol("x1", positive=True)
        return x0 / x1

    monkeypatch.setattr(representation, "aggressive_simplify", simpler_exact_candidate)

    phi_str, _y_str, meta = representation._sympy_simplify_expression(
        "x0*sin(x2)",
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=3,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=None,
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=ys,
    )

    assert sp.simplify(sp.sympify(phi_str) - sp.sympify("x0/x1")) == 0
    assert meta["accepted"] is True
    assert meta["units_checked"] is False
    assert meta["units_ok"] is None


def test_unit_guided_generator_counts_valid_emissions_not_raw_attempts():
    unit_system = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(
            unit_system.dim({"L": 1}),
            unit_system.dim({"L": 1}),
            unit_system.dimless(),
            unit_system.dimless(),
        ),
        y_dim=unit_system.dim({"L": 1}),
    )
    symbols = sp.symbols("x0:4")

    proposals, stats = _unit_consistent_additive_prunings(
        sum(symbols),
        tuple(str(symbol) for symbol in symbols),
        spec,
        requested_count=3,
        max_attempts=128,
        max_depth=3,
    )

    assert len(proposals) == 3
    assert stats["emitted"] == 3
    assert stats["raw_attempted"] > stats["emitted"]
    assert stats["exhausted"] is False
    for _trace, expr, certificate in proposals:
        assert certificate["valid"] is True
        assert check_sympy_units(expr, ("x0", "x1", "x2", "x3"), spec).ok


def test_unit_guided_generator_reports_attempt_budget_for_unvisited_sibling():
    symbols = sp.symbols("x0:3")

    _proposals, stats = _unit_consistent_additive_prunings(
        sum(symbols),
        tuple(str(symbol) for symbol in symbols),
        _length_target_spec(),
        requested_count=3,
        max_attempts=1,
        max_depth=1,
    )

    assert stats["raw_attempted"] == 1
    assert stats["exhausted"] is True
    assert stats["truncated_by_attempt_budget"] is True
    assert stats["exhaustion_reason"] == "attempt_budget_exhausted"


def test_unit_guided_generator_reports_true_finite_space_exhaustion():
    x0, x1 = sp.symbols("x0 x1")

    proposals, stats = _unit_consistent_additive_prunings(
        x0 + x1,
        ("x0", "x1"),
        _length_target_spec(),
        requested_count=3,
        max_attempts=2,
        max_depth=1,
    )

    assert len(proposals) == 2
    assert stats["raw_attempted"] == 2
    assert stats["truncated_by_attempt_budget"] is False
    assert stats["exhaustion_reason"] == "candidate_space_exhausted"


def test_unit_guided_generator_does_not_count_excluded_duplicates_as_emissions():
    x0, x1 = sp.symbols("x0 x1")

    proposals, stats = _unit_consistent_additive_prunings(
        x0 + x1,
        ("x0", "x1"),
        _length_target_spec(),
        requested_count=1,
        max_attempts=8,
        max_depth=1,
        excluded_keys={sp.srepr(x0), sp.srepr(x1)},
    )

    assert proposals == []
    assert stats["emitted"] == 0
    assert stats["deduplicated"] == 2
    assert stats["exhaustion_reason"] == "candidate_space_exhausted"


def test_guarded_stagec_worker_receives_units_spec():
    unit_system = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dim({"L": 1}),),
        y_dim=unit_system.dim({"L": 1}),
    )
    x = torch.linspace(1.0, 2.0, 32, dtype=torch.float64).reshape(-1, 1)
    loader = DataLoader(TensorDataset(x, x[:, 0]), batch_size=16)

    phi_str, y_str, meta = representation.guarded_sympy_simplify_expression(
        "x0",
        model=torch.nn.Identity(),
        val_loader=loader,
        device=torch.device("cpu"),
        Nxvars=1,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=spec,
        verbose=False,
        max_seconds=30.0,
        mem_fraction=0.0,
    )

    assert phi_str == "x0"
    assert y_str is None
    assert meta["accepted"] is True
    assert meta["guarded_subprocess"] is True
    assert meta["unit_admissibility"]["checked"] is True
    assert meta["unit_admissibility"]["valid"] is True


def test_stagec_reports_bounded_exhaustion_when_all_numeric_candidates_are_illegal(
    monkeypatch,
):
    xs, _ = _stagec_arrays()
    ys = xs[:, 0] + xs[:, 0] / xs[:, 1]
    monkeypatch.setattr(
        representation,
        "aggressive_simplify",
        lambda expr, _nxvars, **_kwargs: expr,
    )

    phi_str, y_str, meta = representation._sympy_simplify_expression(
        "x0 + x0/x1",
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=3,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=_length_target_spec(),
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=ys,
    )

    assert phi_str is None
    assert y_str is None
    assert meta["accepted"] is False
    assert meta["numeric_fidelity_ok"] is True
    assert meta["kind"] == "no_unit_valid_candidate"
    assert meta["unit_admissibility"]["valid"] is False
    budget = meta["proposal_budget"]
    assert budget["requested_count"] == 1
    assert budget["raw_attempted"] >= meta["candidate_count"]
    assert budget["unit_rejected"] >= meta["unit_reject_count"]
    assert budget["emitted"] == 0
    assert budget["exhausted"] is True
    assert budget["exhaustion_reason"] == (
        "candidate_space_exhausted_no_unit_valid_numeric_candidate"
    )


def test_stagec_propagates_unit_repair_attempt_budget_exhaustion(monkeypatch):
    xs, _ = _stagec_arrays()
    ys = xs[:, 0] + xs[:, 0] / xs[:, 1]
    monkeypatch.setattr(
        representation,
        "aggressive_simplify",
        lambda expr, _nxvars, **_kwargs: expr,
    )

    def exhausted_repair(*_args, **kwargs):
        return [], {
            "requested_count": kwargs["requested_count"],
            "raw_attempted": kwargs["max_attempts"],
            "unit_rejected": kwargs["max_attempts"],
            "deduplicated": 0,
            "emitted": 0,
            "exhausted": True,
            "exhaustion_reason": "attempt_budget_exhausted",
            "truncated_by_attempt_budget": True,
            "max_attempts": kwargs["max_attempts"],
            "max_depth": kwargs["max_depth"],
        }

    monkeypatch.setattr(
        representation,
        "_unit_consistent_additive_prunings",
        exhausted_repair,
    )

    _phi_str, _y_str, meta = representation._sympy_simplify_expression(
        "x0 + x0/x1",
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=3,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=_length_target_spec(),
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=ys,
    )

    assert meta["accepted"] is False
    assert meta["unit_guided_generation"]["exhaustion_reason"] == (
        "attempt_budget_exhausted"
    )
    assert meta["proposal_budget"]["exhaustion_reason"] == (
        "attempt_budget_exhausted_no_unit_valid_numeric_candidate"
    )


def test_stageb_parent_certificate_checks_phi_and_raw_y_spaces():
    unit_system = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dim({"L": 1}),),
        y_dim=unit_system.dim({"L": 2}),
        y_transform_name="sqrt",
    )

    valid = _with_stagec_unit_certificates(
        {"accepted": True, "numeric_fidelity_ok": True},
        phi_expr_str="x0",
        y_expr_str="x0**2",
        Nxvars=1,
        units_spec=spec,
        enforce_units=True,
    )
    invalid = _with_stagec_unit_certificates(
        {"accepted": True, "numeric_fidelity_ok": True},
        phi_expr_str="x0",
        y_expr_str="x0",
        Nxvars=1,
        units_spec=spec,
        enforce_units=True,
    )

    assert valid["accepted"] is True
    assert valid["unit_admissibility"]["valid"] is True
    assert valid["unit_admissibility"]["phi"]["valid"] is True
    assert valid["unit_admissibility"]["y"]["valid"] is True
    assert invalid["accepted"] is False
    assert invalid["raw_accepted_before_unit_check"] is True
    assert invalid["kind"] == "raw_y_unit_check_failed"
    assert invalid["proposal_budget"]["emitted"] == 0


def test_unit_postcheck_preserves_an_earlier_stagec_failure_reason():
    meta = _with_stagec_unit_certificates(
        {
            "accepted": False,
            "parse_success": False,
            "kind": "bad_pretty_print",
            "reason": "numeric parse did not reproduce the model",
        },
        phi_expr_str=None,
        y_expr_str=None,
        Nxvars=1,
        units_spec=_length_target_spec(),
        enforce_units=True,
    )

    assert meta["accepted"] is False
    assert meta["kind"] == "bad_pretty_print"
    assert meta["reason"] == "numeric parse did not reproduce the model"
    assert meta["unit_admissibility"]["code"] == "expression_unavailable"


def test_raw_coordinate_rewrite_is_recertified_and_cannot_hide_unitful_offset():
    unit_system = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dim({"L": 1}),),
        y_dim=unit_system.dim({"L": 1}),
    )
    internal = _with_stagec_unit_certificates(
        {"accepted": True, "numeric_fidelity_ok": True},
        phi_expr_str="x0",
        y_expr_str="x0",
        Nxvars=1,
        units_spec=spec,
        enforce_units=True,
        coordinate_space="internal",
    )
    raw = _with_stagec_unit_certificates(
        internal,
        phi_expr_str="x0 - 1.0",
        y_expr_str="x0 - 1.0",
        Nxvars=1,
        units_spec=spec,
        enforce_units=True,
        coordinate_space="raw",
    )

    assert internal["unit_admissibility"]["valid"] is True
    assert raw["accepted"] is False
    assert raw["kind"] == "phi_unit_check_failed"
    assert raw["unit_admissibility"]["coordinate_space"] == "raw"
    assert raw["unit_admissibility"]["internal_coordinates"]["valid"] is True


def test_missing_required_raw_coordinate_expression_invalidates_internal_acceptance():
    internal = _with_stagec_unit_certificates(
        {"accepted": True, "numeric_fidelity_ok": True},
        phi_expr_str="x0",
        y_expr_str="x0",
        Nxvars=1,
        units_spec=_length_target_spec(),
        enforce_units=True,
        coordinate_space="internal",
    )

    raw = _with_stagec_unit_certificates(
        internal,
        phi_expr_str=None,
        y_expr_str=None,
        Nxvars=1,
        units_spec=_length_target_spec(),
        enforce_units=True,
        coordinate_space="raw",
    )

    assert raw["accepted"] is False
    assert raw["kind"] == "unit_check_expression_unavailable"
    assert raw["unit_admissibility"]["coordinate_space"] == "raw"
    assert raw["unit_admissibility"]["internal_coordinates"]["valid"] is True
    verified, reason = _stagec_expression_is_verified(
        {
            "y_selected": "identity",
            "phi_expr_str": "x0",
            "y_expr_str": "x0",
            "sympy_meta": raw,
        }
    )
    assert verified is False
    assert "unavailable" in reason


def test_direct_stagec_certifies_transformed_phi_and_raw_y_outputs():
    unit_system = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dim({"L": 1}),),
        y_dim=unit_system.dim({"L": 2}),
        y_transform_name="sqrt",
    )
    xs = np.linspace(1.0, 2.0, 32).reshape(-1, 1)

    phi_str, y_str, meta = representation._sympy_simplify_expression(
        "x0",
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=1,
        y_op_inv=torch.square,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=spec,
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=xs[:, 0],
    )

    assert phi_str == "x0"
    assert sp.simplify(sp.sympify(y_str) - sp.Symbol("x0") ** 2) == 0
    assert meta["accepted"] is True
    assert meta["unit_admissibility"]["expression_space"] == "phi_and_y"
    assert meta["unit_admissibility"]["phi"]["valid"] is True
    assert meta["unit_admissibility"]["y"]["valid"] is True


def test_unit_invalid_stagec_metadata_is_ineligible_downstream():
    certificate = {
        "checked": True,
        "valid": False,
        "reason": "addends have incompatible dimensions",
    }
    stageb_data = {
        "y_selected": "identity",
        "phi_expr_str": "x0 + x0/x1",
        "y_expr_str": "x0 + x0/x1",
        "sympy_meta": {
            "accepted": True,
            "numeric_fidelity_ok": True,
            "units_checked": True,
            "units_ok": False,
            "unit_admissibility": certificate,
        },
        "num_nn_atoms": 0,
    }

    verified, reason = _stagec_expression_is_verified(stageb_data)
    seed, space, seed_reason = _select_final_polish_seed(stageb_data)
    report = {
        "stageC": {
            "certified": False,
            "certification_reason": reason,
            "y_expr_str": stageb_data["y_expr_str"],
        }
    }

    assert verified is False
    assert "incompatible" in reason
    assert seed == stageb_data["y_expr_str"]
    assert space == "y_diagnostic_unit_invalid"
    assert seed_reason is None
    assert _report_final_selection_eligibility(report) == (False, reason)
    fallback = _final_selection_from_report(report)
    assert fallback["applied"] is False
    assert fallback["eligible_for_success"] is False


def test_accepted_stagec_requires_complete_checked_unit_certificate():
    verified, reason = _stagec_expression_is_verified(
        {
            "y_expr_str": "x0",
            "sympy_meta": {
                "accepted": True,
                "units_checked": True,
                "units_ok": None,
                "unit_admissibility": {"checked": True, "valid": None},
            },
        }
    )

    assert verified is False
    assert reason == "Stage C unit certificate is incomplete"


def test_explicit_final_selection_cannot_override_its_own_invalid_certificate():
    eligible, reason = _report_final_selection_eligibility(
        {
            "final_selection": {
                "source": "coe_committee",
                "applied": True,
                "eligible_for_success": True,
                "expr": "x0 + x0/x1",
                "unit_admissibility": {
                    "checked": True,
                    "valid": False,
                    "reason": "additive unit mismatch",
                },
            }
        }
    )

    assert eligible is False
    assert reason == "additive unit mismatch"


def test_unit_risk_override_requires_an_explicit_valid_certificate():
    eligible, reason = _report_final_selection_eligibility(
        {
            "stageC": {
                "certified": False,
                "symbolic_status": "unit_invalid",
                "units_checked": True,
                "units_ok": False,
            },
            "final_selection": {
                "source": "coe_committee",
                "applied": True,
                "eligible_for_success": True,
                "expr": "x0",
            },
        }
    )

    assert eligible is False
    assert reason == "unit-risk final selection lacks a valid unit certificate"


def test_unavailable_required_raw_unit_check_is_also_a_unit_risk():
    eligible, reason = _report_final_selection_eligibility(
        {
            "stageC": {
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
            },
            "final_selection": {
                "source": "coe_committee",
                "applied": True,
                "eligible_for_success": True,
                "expr": "x0",
            },
        }
    )

    assert eligible is False
    assert reason == "unit-risk final selection lacks a valid unit certificate"


def test_pb061_stagec_expression_is_rejected_at_the_stagec_boundary():
    expr = sp.sympify(
        "(10*pi**3*x0*x1*x2 - sqrt(2)*x0*x2 "
        "+ 10*pi**3*x0*x4*x5*cos(x3))/(10*pi**3*x1*x2)"
    )

    result = check_sympy_units(
        expr,
        tuple(f"x{i}" for i in range(6)),
        _pb061_spec(),
        expression_space="phi",
    )

    assert result.ok is False
    assert result.code == "add_dimension_mismatch"
    assert result.failure_path.startswith("$expr")


def test_pb061_unit_guided_addend_prune_surfaces_the_valid_expression(monkeypatch):
    rng = np.random.default_rng(61)
    xs = rng.uniform(1.0, 3.0, size=(128, 6))
    x0, x1, x2, x3, x4, x5 = (xs[:, i] for i in range(6))
    ys_valid = x0 * (1.0 + x4 * x5 * np.cos(x3) / (x1 * x2))
    monkeypatch.setattr(
        representation,
        "aggressive_simplify",
        lambda expr, _nxvars, **_kwargs: expr,
    )
    pb061_expr = (
        "(10*pi**3*x0*x1*x2 - sqrt(2)*x0*x2 "
        "+ 10*pi**3*x0*x4*x5*cos(x3))/(10*pi**3*x1*x2)"
    )

    phi_str, _y_str, meta = representation._sympy_simplify_expression(
        pb061_expr,
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=6,
        noise_floor_raw=1.0e-8,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=_pb061_spec(),
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=ys_valid,
    )

    assert phi_str is not None, meta
    recovered = sp.sympify(phi_str)
    expected = sp.sympify("x0*(1 + x4*x5*cos(x3)/(x1*x2))")
    assert sp.simplify(recovered - expected) == 0, (phi_str, meta)
    assert meta["accepted"] is True
    assert meta["kind"].startswith("unit_addend_prune:")
    assert meta["unit_guided_generation"]["emitted"] >= 1
    assert meta["proposal_budget"]["emitted"] == 1
