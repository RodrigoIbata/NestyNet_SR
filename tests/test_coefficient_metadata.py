# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
import os
import time
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.equation_polisher import (
    ArtifactHints,
    PolishConfig,
    apply_full_dataset_snap_adjudication,
    load_artifact_hints,
    polish_expression,
)
from nestynet_sr.run_sr_final_polish import _stageB_data_for_final_polish_worker
from nestynet_sr.run_sr_de import _de_stageB_report_payload
from nestynet_sr.run_sr_reports import (
    _report_final_selection_eligibility,
    _stagec_expression_is_verified,
    write_json_report,
)
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    FixedConst,
    FreeConst,
    MulNode,
    Scale,
    Var,
    ast_equals,
    ast_to_human_readable,
    build_composite_from_ast,
    ensure_atom_tag,
)
from nestynet_sr.sr_core.coefficient_metadata import (
    CoefficientMetadataError,
    coefficient_symbol_for_name,
    coefficient_symbol_values,
    coefficient_symbol_values_for_expression,
    collect_coefficient_metadata,
    empty_coefficient_metadata,
    normalize_coefficient_metadata,
    normalize_coefficient_metadata_by_dataset,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search.representation import (
    _sympy_simplify_expression,
    guarded_sympy_simplify_expression,
    pretty_print_state,
)
from nestynet_sr.sr_search.coe_committee import (
    CandidateArtifact,
    SliceSpec,
    evaluate_candidate_on_slice,
)
from nestynet_sr.sr_search.truth_eval import evaluate_against_truth


def _core(leaf):
    return getattr(leaf, "core", getattr(leaf, "model", leaf))


def _model(root):
    return build_composite_from_ast(root, dtype=torch.float64)


def _length_constant_case(*, fixed: bool = False):
    unit_system = UnitSystem(("L", "T"))
    length = unit_system.dim({"L": 1})
    atom = FixedConst("g", value=9.81) if fixed else FreeConst("c", init=2.5)
    root = MulNode(atom, Var(0))
    model = _model(root)
    if fixed:
        spec = UnitsSpec(
            unit_system=unit_system,
            x_dims=(unit_system.dimless(),),
            y_dim=length,
            fixed_const_dims={"g": length},
            fixed_const_values={"g": 9.81},
        )
    else:
        spec = UnitsSpec(
            unit_system=unit_system,
            x_dims=(unit_system.dimless(),),
            y_dim=length,
            free_const_dims={"c": length},
        )
    metadata = collect_coefficient_metadata(root, model, spec)
    return root, model, spec, metadata


def test_unitless_and_unitful_named_coefficients_share_one_record_path():
    unit_system = UnitSystem(("L", "T"))
    root = FreeConst("c", init=2.5, scope="experiment")
    model = _model(root)
    unitless_spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dimless(),),
        y_dim=unit_system.dimless(),
        free_const_dims={"c": unit_system.dimless()},
    )
    length_spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dimless(),),
        y_dim=unit_system.dim({"L": 1}),
        free_const_dims={"c": unit_system.dim({"L": 1})},
    )

    unitless = collect_coefficient_metadata(root, model, unitless_spec)
    unitful = collect_coefficient_metadata(root, model, length_spec)
    assert unitless["valid"] is True
    assert unitful["valid"] is True
    rec0 = dict(unitless["records"][0])
    rec1 = dict(unitful["records"][0])
    assert rec0.pop("dimension") == ["0", "0"]
    assert rec1.pop("dimension") == ["1", "0"]
    assert rec0 == rec1


def test_metadata_is_exact_json_roundtrippable_and_exposes_fitted_values():
    unit_system = UnitSystem(("L", "T"))
    half_length = unit_system.dim({"L": Fraction(1, 2)})
    root = FreeConst("c", init=2.5, scope="class")
    model = _model(root)
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dimless(),),
        y_dim=half_length,
        free_const_dims={"c": half_length},
        free_const_scope={"c": "class"},
    )

    metadata = collect_coefficient_metadata(root, model, spec)
    roundtripped = json.loads(json.dumps(metadata))
    assert normalize_coefficient_metadata(roundtripped, require_values=True) == metadata
    assert metadata["records"][0]["dimension"] == ["1/2", "0"]
    assert coefficient_symbol_values(metadata) == {"c": 2.5}


def test_pretty_printer_retains_named_constants_but_not_anonymous_scale():
    free_root = MulNode(FreeConst("c", init=2.5), Var(0))
    fixed_root = MulNode(FixedConst("g", value=9.81), Var(0))
    scale_root = MulNode(Scale("s", init=2.5), Var(0))

    free_expr = pretty_print_state(
        SimpleNamespace(root=free_root, model=_model(free_root)), sig=16
    )
    fixed_expr = pretty_print_state(
        SimpleNamespace(root=fixed_root, model=_model(fixed_root)), sig=16
    )
    scale_expr = pretty_print_state(
        SimpleNamespace(root=scale_root, model=_model(scale_root)), sig=16
    )

    assert sp.simplify(sp.sympify(free_expr) - sp.Symbol("c") * sp.Symbol("x0")) == 0
    assert sp.simplify(sp.sympify(fixed_expr) - sp.Symbol("g") * sp.Symbol("x0")) == 0
    assert "c" in free_expr and "2.5" not in free_expr
    assert "g" in fixed_expr and "9.81" not in fixed_expr
    assert "s" not in scale_expr and "2.5" in scale_expr
    assert ast_to_human_readable(FixedConst("g", value=9.81)) == "g"

    mul_scale_root = AtomNode(
        kind="mul_scale",
        var_idxs=(),
        kwargs={"name": "gauge", "init": 1.25},
        tag="gauge",
    )
    mul_scale_metadata = collect_coefficient_metadata(
        mul_scale_root,
        _model(mul_scale_root),
    )
    assert mul_scale_metadata["records"][0]["kind"] == "scale"
    assert mul_scale_metadata["records"][0]["display"] == "numeric"


def test_stagec_roundtrip_keeps_trainable_and_fixed_constants_symbolic():
    for fixed, symbol, value in ((False, "c", 2.5), (True, "g", 9.81)):
        root, model, spec, metadata = _length_constant_case(fixed=fixed)
        expr = pretty_print_state(SimpleNamespace(root=root, model=model), sig=16)
        xs = np.linspace(1.0, 2.0, 32).reshape(-1, 1)
        ys = value * xs[:, 0]

        phi_str, y_str, meta = _sympy_simplify_expression(
            expr,
            model=None,
            val_loader=None,
            device=torch.device("cpu"),
            Nxvars=1,
            prefer_stable_trig=False,
            prune_trig_poly_args=False,
            linearize_leaves=False,
            units_spec=spec,
            coefficient_metadata=metadata,
            verbose=False,
            precomputed_xs_np=xs,
            precomputed_ys_model=ys,
        )

        assert sp.simplify(
            sp.sympify(phi_str) - sp.Symbol(symbol) * sp.Symbol("x0")
        ) == 0
        assert y_str is None
        assert meta["accepted"] is True
        assert meta["units_ok"] is True
        assert meta["coefficient_metadata"] == metadata
        assert meta["coefficient_symbols_available"] == [symbol]
        assert meta["coefficient_symbols_used"] == [symbol]


def test_guarded_stagec_worker_receives_coefficient_values():
    root, model, spec, metadata = _length_constant_case()
    expr = pretty_print_state(SimpleNamespace(root=root, model=model), sig=16)
    x = torch.linspace(1.0, 2.0, 32, dtype=torch.float64).reshape(-1, 1)
    loader = DataLoader(TensorDataset(x, 2.5 * x[:, 0]), batch_size=16)

    phi_str, y_str, meta = guarded_sympy_simplify_expression(
        expr,
        model=model,
        val_loader=loader,
        device=torch.device("cpu"),
        Nxvars=1,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=spec,
        coefficient_metadata=metadata,
        verbose=False,
        max_seconds=30.0,
        mem_fraction=0.0,
    )

    assert sp.simplify(sp.sympify(phi_str) - sp.Symbol("c") * sp.Symbol("x0")) == 0
    assert y_str is None
    assert meta["accepted"] is True
    assert meta["guarded_subprocess"] is True
    assert meta["coefficient_metadata"] == metadata


def test_stagec_fails_closed_when_symbol_value_metadata_is_missing():
    xs = np.linspace(1.0, 2.0, 16).reshape(-1, 1)
    phi_str, y_str, meta = _sympy_simplify_expression(
        "c*x0",
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=1,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=2.5 * xs[:, 0],
    )

    assert phi_str is None
    assert y_str is None
    assert meta["accepted"] is False
    assert meta["kind"] == "coefficient_value_missing"
    assert meta["missing_symbols"] == ["c"]


def test_independent_duplicate_symbol_identities_fail_closed():
    left = FreeConst("c", tag="c_left", init=2.0)
    right = FreeConst("c", tag="c_right", init=3.0)
    root = AddNode(left, right)
    model = _model(root)
    assert float(_core(model.leaf[0]).value.detach()) == 2.0
    assert float(_core(model.leaf[1]).value.detach()) == 3.0
    unit_system = UnitSystem(("L",))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dimless(),),
        y_dim=unit_system.dimless(),
        free_const_dims={"c": unit_system.dimless()},
    )

    metadata = collect_coefficient_metadata(root, model, spec)
    assert metadata["valid"] is False
    assert metadata["code"] == "coefficient_identity_topology_conflict"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda payload: payload.update(records=7), "coefficient_records_invalid"),
        (
            lambda payload: payload.update(record_count=99),
            "coefficient_count_mismatch",
        ),
        (
            lambda payload: payload["records"][0].update(occurrences="bad"),
            "coefficient_occurrences_invalid",
        ),
        (
            lambda payload: payload["records"][0].update(value=True),
            "coefficient_value_invalid",
        ),
        (
            lambda payload: payload["records"][0].update(kind="mystery"),
            "coefficient_kind_invalid",
        ),
        (
            lambda payload: payload["records"][0].update(scope="fixed"),
            "coefficient_free_record_invalid",
        ),
        (
            lambda payload: payload["records"][0].update(trainable="false"),
            "coefficient_trainable_invalid",
        ),
        (
            lambda payload: payload["records"][0].update(
                identity="fixed_const:g"
            ),
            "coefficient_identity_invalid",
        ),
        (
            lambda payload: payload["records"][0].update(
                dimension_status="banana"
            ),
            "coefficient_dimension_status_invalid",
        ),
        (
            lambda payload: payload["records"][0].update(name="other"),
            "coefficient_name_invalid",
        ),
        (
            lambda payload: payload["records"][0].update(
                value_source="fixed_buffer"
            ),
            "coefficient_value_source_invalid",
        ),
    ],
)
def test_malformed_explicit_metadata_fails_closed(mutation, code):
    _, _, _, metadata = _length_constant_case()
    malformed = json.loads(json.dumps(metadata))
    mutation(malformed)
    with pytest.raises(CoefficientMetadataError) as exc_info:
        normalize_coefficient_metadata(malformed, require_values=True)
    assert exc_info.value.code == code


def test_fixed_and_scale_metadata_cross_field_invariants_fail_closed():
    _, _, _, fixed = _length_constant_case(fixed=True)
    malformed_fixed = json.loads(json.dumps(fixed))
    malformed_fixed["records"][0].update(scope="experiment", trainable=True)
    with pytest.raises(CoefficientMetadataError) as fixed_exc:
        normalize_coefficient_metadata(malformed_fixed, require_values=True)
    assert fixed_exc.value.code == "coefficient_fixed_record_invalid"

    unit_system = UnitSystem(("L", "T"))
    scale_root = Scale("s", init=1.25)
    scale_spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dimless(),),
        y_dim=unit_system.dimless(),
    )
    scale = collect_coefficient_metadata(
        scale_root,
        _model(scale_root),
        scale_spec,
    )
    symbolic_scale = json.loads(json.dumps(scale))
    symbolic_scale["records"][0].update(symbol="s", display="symbol")
    with pytest.raises(CoefficientMetadataError) as symbol_exc:
        normalize_coefficient_metadata(symbolic_scale, require_values=True)
    assert symbol_exc.value.code == "coefficient_scale_record_invalid"

    unitful_scale = json.loads(json.dumps(scale))
    unitful_scale["records"][0].update(dimension=["1", "0"])
    with pytest.raises(CoefficientMetadataError) as dimension_exc:
        normalize_coefficient_metadata(unitful_scale, require_values=True)
    assert dimension_exc.value.code == "coefficient_scale_dimension_invalid"


def test_polisher_scores_named_constant_numerically_without_erasing_symbol():
    _, _, spec, metadata = _length_constant_case()
    x = np.linspace(1.0, 3.0, 80)
    X = x.reshape(-1, 1)
    y = 2.5 * x
    result = polish_expression(
        "c*x0",
        X[:60],
        y[:60],
        X[60:],
        y[60:],
        variable_names=["x0"],
        units_spec=spec,
        artifact_hints=ArtifactHints(coefficient_metadata=metadata),
        config=PolishConfig(
            max_candidates=16,
            use_artifact_hints=False,
        ),
    )

    assert result.seed_baseline is not None
    assert result.seed_baseline.val_mse == 0.0
    assert result.seed_baseline.n_free_params == 1
    assert result.recommended is not None
    assert "c" in result.recommended.expr
    assert result.seed_units_ok is True


def test_polisher_does_not_count_declared_fixed_unitful_constant_as_learned():
    _, _, spec, metadata = _length_constant_case(fixed=True)
    x = np.linspace(1.0, 3.0, 80)
    X = x.reshape(-1, 1)
    y = 9.81 * x
    result = polish_expression(
        "g*x0",
        X[:60],
        y[:60],
        X[60:],
        y[60:],
        variable_names=["x0"],
        units_spec=spec,
        artifact_hints=ArtifactHints(coefficient_metadata=metadata),
        config=PolishConfig(max_candidates=16, use_artifact_hints=False),
    )

    assert result.seed_baseline is not None
    assert result.seed_baseline.n_free_params == 0
    assert result.seed_units_ok is True


def test_full_dataset_adjudication_evaluates_named_unitful_constant():
    _, _, spec, metadata = _length_constant_case()
    x = np.linspace(1.0, 3.0, 80)
    X = x.reshape(-1, 1)
    y = 2.5 * x
    config = PolishConfig(max_candidates=1, use_artifact_hints=False)
    result = polish_expression(
        "c*x0",
        X[:60],
        y[:60],
        X[60:],
        y[60:],
        variable_names=["x0"],
        units_spec=spec,
        artifact_hints=ArtifactHints(coefficient_metadata=metadata),
        config=config,
    )

    result, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0"],
        units_spec=spec,
        # Deliberately pass the original config: adjudication must recover the
        # named value from result metadata just as polish_expression does.
        config=config,
    )

    assert summary["seed_full_error"] is None
    assert summary["n_evaluated"] >= 1
    assert result.recommended is not None
    assert result.recommended.full_dataset_mse == 0.0
    assert result.recommended.n_free_params == 1


def test_truth_evaluation_substitutes_named_constants_and_fails_without_values():
    domain = {"x0": (1.0, 3.0)}
    missing = evaluate_against_truth(
        "c*x0",
        "2.5*x0",
        domain,
        n_samples=64,
    )
    assert missing["success"] is False
    assert missing["error_message"] == "Missing coefficient values for symbols: c"

    evaluated = evaluate_against_truth(
        "c*x0",
        "2.5*x0",
        domain,
        n_samples=64,
        symbol_values={"c": 2.5},
    )
    assert evaluated["success"] is True
    assert evaluated["rmse_abs"] == 0.0
    assert evaluated["coefficient_symbols_used"] == ["c"]


def test_ast_tagging_scope_and_report_worker_preserve_metadata(tmp_path, monkeypatch):
    compound = AtomNode(
        kind="poly",
        var_idxs=(0, 1),
        kwargs={"degree": 2},
        tag=None,
        inputs=(MulNode(Var(0), Var(1)),),
        scope="class",
    )
    tagged = ensure_atom_tag(compound, context="coefficient")
    assert tagged.scope == "class"
    assert ast_equals(tagged.inputs[0], compound.inputs[0])
    assert tagged.inputs[0] is not compound.inputs[0]
    other_scope = AtomNode(
        kind=tagged.kind,
        var_idxs=tagged.var_idxs,
        kwargs=dict(tagged.kwargs),
        tag=tagged.tag,
        inputs=tagged.inputs,
        scope="experiment",
    )
    assert not ast_equals(tagged, other_scope)

    root, _, _, metadata = _length_constant_case()
    report_path = tmp_path / "report.json"
    stageB_data = {
        "ast": root,
        "phi_expr_str": "c*x0",
        "y_expr_str": "c*x0",
        "sympy_meta": {"accepted": True, "parse_success": True},
        "coefficient_metadata": metadata,
    }
    truth_calls = []

    def fake_evaluate_canary(
        *, dataset_stem, discovered_expr_str, verbose=False, symbol_values=None
    ):
        truth_calls.append(
            (dataset_stem, discovered_expr_str, verbose, symbol_values)
        )
        return {"success": True, "rmse_abs": 0.0, "rmse_rel": 0.0}

    import nestynet_sr.sr_search.truth_eval as truth_eval_module

    monkeypatch.setattr(
        truth_eval_module,
        "evaluate_canary",
        fake_evaluate_canary,
    )
    write_json_report(
        filepath="pb000_coefficients.csv",
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=1,
        walltime=0.0,
        stageB_data=stageB_data,
        enable_truth_eval=True,
    )
    report = json.loads(report_path.read_text())
    assert report["stageB"]["coefficient_metadata"] == metadata
    assert report["stageC"]["coefficient_metadata"] == metadata
    assert truth_calls == [
        ("pb000_coefficients", "c*x0", False, {"c": 2.5})
    ]
    hints = load_artifact_hints(report_json=report_path)
    assert hints.coefficient_metadata == metadata
    worker_payload = _stageB_data_for_final_polish_worker(stageB_data)
    assert worker_payload["coefficient_metadata"] == metadata
    de_payload = _de_stageB_report_payload(
        SimpleNamespace(
            val_loss=0.0,
            phi_expr_str="c*x0",
            coefficient_metadata=metadata,
            coefficient_metadata_by_dataset=None,
        ),
        phi_expr_strs=["c*x0"],
    )
    assert de_payload["phi_expr_str"] == "c*x0"
    assert de_payload["coefficient_metadata"] == metadata

    eligible, reason = _report_final_selection_eligibility(
        {
            "final_selection": {
                "expr": "c*x0",
                "applied": True,
                "coefficient_metadata": metadata,
            }
        }
    )
    assert eligible is True
    assert reason is None

    invalid_metadata = dict(metadata)
    invalid_metadata["valid"] = False
    invalid_metadata["reason"] = "tampered coefficient metadata"
    eligible, reason = _report_final_selection_eligibility(
        {
            "final_selection": {
                "expr": "c*x0",
                "applied": True,
                "coefficient_metadata": invalid_metadata,
            }
        }
    )
    assert eligible is False
    assert "tampered coefficient metadata" in reason


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda payload: payload.update(dimension_basis=False),
            "coefficient_dimension_basis_invalid",
        ),
        (
            lambda payload: payload.update(records={}),
            "coefficient_records_invalid",
        ),
        (
            lambda payload: payload["records"][0].update(occurrences=False),
            "coefficient_occurrences_invalid",
        ),
        (
            lambda payload: payload.update(record_count=False),
            "coefficient_count_invalid",
        ),
        (
            lambda payload: payload.update(record_count=0.1),
            "coefficient_count_invalid",
        ),
        (
            lambda payload: payload.update(dataset_index=1.7),
            "coefficient_dataset_index_invalid",
        ),
    ],
)
def test_falsey_and_coercive_schema_fields_fail_closed(mutation, code):
    _, _, _, metadata = _length_constant_case()
    malformed = json.loads(json.dumps(metadata))
    mutation(malformed)
    with pytest.raises(CoefficientMetadataError) as exc_info:
        normalize_coefficient_metadata(malformed, require_values=True)
    assert exc_info.value.code == code


def test_active_units_spec_rejects_tampered_dimension_in_stagec():
    root, model, spec, metadata = _length_constant_case()
    malformed = json.loads(json.dumps(metadata))
    malformed["records"][0]["dimension"] = ["0", "1"]
    xs = np.linspace(1.0, 2.0, 16).reshape(-1, 1)

    phi_str, y_str, meta = _sympy_simplify_expression(
        pretty_print_state(SimpleNamespace(root=root, model=model), sig=16),
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=1,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=spec,
        coefficient_metadata=malformed,
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=2.5 * xs[:, 0],
    )

    assert phi_str is None
    assert y_str is None
    assert meta["kind"] == "invalid_coefficient_metadata"
    assert meta["coefficient_metadata_error"]["code"] == (
        "coefficient_dimension_conflict"
    )


def test_reserved_pi_constant_uses_safe_symbol_and_declared_value():
    unit_system = UnitSystem(("L",))
    root = MulNode(FixedConst("pi", value=3.0), Var(0))
    model = _model(root)
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dimless(),),
        y_dim=unit_system.dimless(),
        fixed_const_dims={"pi": unit_system.dimless()},
        fixed_const_values={"pi": 3.0},
    )
    metadata = collect_coefficient_metadata(root, model, spec)
    expr = pretty_print_state(SimpleNamespace(root=root, model=model), sig=16)

    assert coefficient_symbol_for_name("pi") == "coef_pi"
    assert metadata["valid"] is True
    assert metadata["records"][0]["name"] == "pi"
    assert metadata["records"][0]["symbol"] == "coef_pi"
    assert sp.simplify(
        sp.sympify(expr) - sp.Symbol("coef_pi") * sp.Symbol("x0")
    ) == 0
    assert ast_to_human_readable(root) == "(coef_pi * x0)"
    assert coefficient_symbol_values_for_expression(metadata, expr) == {
        "coef_pi": 3.0
    }

    xs = np.linspace(1.0, 2.0, 16).reshape(-1, 1)
    phi_str, _, meta = _sympy_simplify_expression(
        expr,
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=1,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        units_spec=spec,
        coefficient_metadata=metadata,
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=3.0 * xs[:, 0],
    )
    assert sp.simplify(
        sp.sympify(phi_str) - sp.Symbol("coef_pi") * sp.Symbol("x0")
    ) == 0
    assert meta["accepted"] is True


def test_coefficient_symbol_escaping_is_parser_safe_and_injective():
    assert coefficient_symbol_for_name("c") == "c"
    assert coefficient_symbol_for_name("pi") == "coef_pi"
    assert coefficient_symbol_for_name("gamma") == "coef_gamma"
    assert coefficient_symbol_for_name("N") == "coef_N"
    assert coefficient_symbol_for_name("lambda") == "coef_lambda"
    assert coefficient_symbol_for_name("None") == "coef_None"
    assert coefficient_symbol_for_name("x0") == "coef_x0"
    assert coefficient_symbol_for_name("coef_pi") == "coef_coef_pi"


def test_input_namespace_coefficient_name_stays_distinct_from_variable():
    root = MulNode(FreeConst("x0", init=2.0), Var(0))
    model = _model(root)
    metadata = collect_coefficient_metadata(root, model)
    expr = pretty_print_state(SimpleNamespace(root=root, model=model), sig=16)

    assert metadata["valid"] is True
    assert metadata["records"][0]["name"] == "x0"
    assert metadata["records"][0]["symbol"] == "coef_x0"
    assert expr == "coef_x0 * x0"
    assert coefficient_symbol_values_for_expression(
        metadata,
        expr,
        variable_names=("x0",),
    ) == {"coef_x0": 2.0}


@pytest.mark.parametrize(
    "expression",
    (
        "__import__('os').getcwd()",
        "__import__('builtins').len([1, 2])",
        "open('/tmp/should_not_exist', 'w')",
    ),
)
def test_coefficient_expression_parser_rejects_executable_syntax_before_eval(
    monkeypatch,
    expression,
):
    executed = []
    monkeypatch.setattr(os, "getcwd", lambda: executed.append(True) or "/executed")

    with pytest.raises(CoefficientMetadataError) as exc_info:
        coefficient_symbol_values_for_expression(None, expression)

    assert exc_info.value.code == "coefficient_expression_unsafe"
    assert executed == []


def test_coefficient_expression_parser_retains_supported_math_grammar():
    expression = (
        "sqrt(x0**2) + sin(x0) + sign(x0) + "
        "(1/2)*Abs(x0)"
    )

    assert coefficient_symbol_values_for_expression(None, expression) == {}


@pytest.mark.parametrize(
    "expression",
    (
        "Pow(10, 10000000)",
        "10**10000000",
        "10**(1000*1000)",
        "Float(1, 10000000)",
    ),
)
def test_coefficient_expression_parser_rejects_eager_or_unbounded_forms_fast(
    expression,
):
    started = time.perf_counter()
    with pytest.raises(CoefficientMetadataError) as exc_info:
        coefficient_symbol_values_for_expression(None, expression)
    elapsed = time.perf_counter() - started

    assert exc_info.value.code == "coefficient_expression_unsafe"
    assert elapsed < 1.0


def test_sympy_global_coefficient_names_survive_stagec_polisher_and_coe(tmp_path):
    unit_system = UnitSystem(("L",))
    xs = np.linspace(1.0, 3.0, 32)
    X = xs.reshape(-1, 1)
    ys = 2.0 * xs
    data_path = tmp_path / "reserved_coefficient_names.csv"
    np.savetxt(
        data_path,
        np.column_stack([xs, ys]),
        delimiter=",",
        header="x0,y",
        comments="",
    )

    for logical_name, symbol in (("gamma", "coef_gamma"), ("N", "coef_N")):
        root = MulNode(FreeConst(logical_name, init=2.0), Var(0))
        model = _model(root)
        spec = UnitsSpec(
            unit_system=unit_system,
            x_dims=(unit_system.dimless(),),
            y_dim=unit_system.dimless(),
            free_const_dims={logical_name: unit_system.dimless()},
        )
        metadata = collect_coefficient_metadata(root, model, spec)
        expr = pretty_print_state(SimpleNamespace(root=root, model=model), sig=16)

        assert metadata["valid"] is True
        assert metadata["records"][0]["symbol"] == symbol
        assert expr == f"{symbol} * x0"

        phi_str, _, stagec_meta = _sympy_simplify_expression(
            expr,
            model=None,
            val_loader=None,
            device=torch.device("cpu"),
            Nxvars=1,
            prefer_stable_trig=False,
            prune_trig_poly_args=False,
            linearize_leaves=False,
            units_spec=spec,
            coefficient_metadata=metadata,
            verbose=False,
            precomputed_xs_np=X,
            precomputed_ys_model=ys,
        )
        assert stagec_meta["accepted"] is True
        assert symbol in str(phi_str)

        polished = polish_expression(
            expr,
            X[:24],
            ys[:24],
            X[24:],
            ys[24:],
            variable_names=["x0"],
            units_spec=spec,
            artifact_hints=ArtifactHints(coefficient_metadata=metadata),
            config=PolishConfig(max_candidates=8, use_artifact_hints=False),
        )
        assert polished.seed_baseline is not None
        assert polished.seed_baseline.val_mse == 0.0
        assert polished.recommended is not None
        assert symbol in polished.recommended.expr

        result = evaluate_candidate_on_slice(
            CandidateArtifact(
                candidate_id=logical_name,
                expr=expr,
                source="test",
                metadata={"coefficient_metadata": metadata},
            ),
            filepath=data_path,
            spec=SliceSpec(0, 0, 24, 24, 32),
        )
        assert result.status == "success"
        assert result.val_mse == 0.0


def test_coefficient_identity_tracks_shared_parameter_topology():
    unit_system = UnitSystem(("L",))
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dimless(),),
        y_dim=unit_system.dimless(),
        free_const_dims={
            "c": unit_system.dimless(),
            "d": unit_system.dimless(),
        },
    )
    independent = AddNode(
        FreeConst("c", tag="left", init=2.0),
        FreeConst("c", tag="right", init=2.0),
    )
    independent_metadata = collect_coefficient_metadata(
        independent,
        _model(independent),
        spec,
    )
    assert independent_metadata["code"] == (
        "coefficient_identity_topology_conflict"
    )

    aliased = AddNode(
        FreeConst("c", tag="shared", init=2.0),
        FreeConst("d", tag="shared", init=2.0),
    )
    aliased_metadata = collect_coefficient_metadata(
        aliased,
        _model(aliased),
        spec,
    )
    assert aliased_metadata["code"] == "coefficient_parameter_alias_conflict"


def test_expression_metadata_coverage_is_required_for_final_eligibility():
    _, _, _, metadata = _length_constant_case()
    empty = empty_coefficient_metadata()
    for expression, payload in (
        ("c*x0", None),
        ("c*x0", empty),
        ("d*x0", metadata),
    ):
        eligible, reason = _report_final_selection_eligibility(
            {
                "final_selection": {
                    "expr": expression,
                    "applied": True,
                    "coefficient_metadata": payload,
                }
            }
        )
        assert eligible is False
        assert "no coefficient value metadata" in str(reason)


def test_per_dataset_metadata_allows_experiment_values_but_not_shared_conflicts():
    unit_system = UnitSystem(("L",))
    root = FreeConst("c", init=2.0, scope="experiment")
    spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dimless(),),
        y_dim=unit_system.dimless(),
        free_const_dims={"c": unit_system.dimless()},
    )
    first_model = _model(root)
    second_model = _model(root)
    _core(second_model.leaf[0]).value.data.fill_(3.0)
    experiment_bundles = [
        collect_coefficient_metadata(
            root, first_model, spec, dataset_id="a", dataset_index=0
        ),
        collect_coefficient_metadata(
            root, second_model, spec, dataset_id="b", dataset_index=1
        ),
    ]
    normalized = normalize_coefficient_metadata_by_dataset(
        experiment_bundles,
        units_spec=spec,
        expected_count=2,
    )
    assert [bundle["records"][0]["value"] for bundle in normalized] == [2.0, 3.0]

    class_root = FreeConst("c", init=2.0, scope="class")
    class_spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(unit_system.dimless(),),
        y_dim=unit_system.dimless(),
        free_const_dims={"c": unit_system.dimless()},
        free_const_scope={"c": "class"},
    )
    class_first = _model(class_root)
    class_second = _model(class_root)
    _core(class_second.leaf[0]).value.data.fill_(3.0)
    class_bundles = [
        collect_coefficient_metadata(
            class_root, class_first, class_spec, dataset_id="a", dataset_index=0
        ),
        collect_coefficient_metadata(
            class_root, class_second, class_spec, dataset_id="b", dataset_index=1
        ),
    ]
    with pytest.raises(CoefficientMetadataError) as exc_info:
        normalize_coefficient_metadata_by_dataset(
            class_bundles,
            units_spec=class_spec,
            expected_count=2,
        )
    assert exc_info.value.code == "coefficient_shared_value_conflict"


def test_stagec_verification_rejects_invalid_secondary_dataset_bundle():
    root, model, spec, _ = _length_constant_case()
    first = collect_coefficient_metadata(
        root, model, spec, dataset_id="a", dataset_index=0
    )
    second = collect_coefficient_metadata(
        root, _model(root), spec, dataset_id="b", dataset_index=1
    )
    second["valid"] = False
    second["reason"] = "tampered secondary dataset"
    verified, reason = _stagec_expression_is_verified(
        {
            "dataset_ids": ["a", "b"],
            "y_expr_str": "c*x0",
            "sympy_meta": {"accepted": True, "parse_success": True},
            "coefficient_metadata": first,
            "coefficient_metadata_by_dataset": [first, second],
        }
    )
    assert verified is False
    assert "tampered secondary dataset" in str(reason)


def test_stagec_verification_rejects_swapped_or_divergent_dataset_metadata():
    root, model, spec, _ = _length_constant_case()
    first = collect_coefficient_metadata(
        root, model, spec, dataset_id="a", dataset_index=0
    )
    second_model = _model(root)
    _core(second_model.leaf[0]).value.data.fill_(3.0)
    second = collect_coefficient_metadata(
        root, second_model, spec, dataset_id="b", dataset_index=1
    )
    base = {
        "dataset_ids": ["a", "b"],
        "y_expr_str": "c*x0",
        "sympy_meta": {"accepted": True, "parse_success": True},
        "coefficient_metadata": first,
        "coefficient_metadata_by_dataset": [first, second],
    }

    swapped_first = json.loads(json.dumps(first))
    swapped_second = json.loads(json.dumps(second))
    swapped_first["dataset_id"] = "b"
    swapped_second["dataset_id"] = "a"
    for record in swapped_first["records"]:
        record["dataset_id"] = "b"
    for record in swapped_second["records"]:
        record["dataset_id"] = "a"
    swapped = dict(base)
    swapped["coefficient_metadata"] = swapped_first
    swapped["coefficient_metadata_by_dataset"] = [
        swapped_first,
        swapped_second,
    ]
    verified, reason = _stagec_expression_is_verified(swapped)
    assert verified is False
    assert "expected 'a'" in str(reason)

    divergent = dict(base)
    divergent_primary = json.loads(json.dumps(first))
    divergent_primary["records"][0]["value"] = 4.0
    divergent["coefficient_metadata"] = divergent_primary
    verified, reason = _stagec_expression_is_verified(divergent)
    assert verified is False
    assert "does not match dataset bundle 0" in str(reason)
