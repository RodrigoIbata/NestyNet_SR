# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from pathlib import Path
import warnings

import numpy as np
import pytest
import sympy as sp
from sympy.utilities.exceptions import SymPyDeprecationWarning

import nestynet_sr.equation_polisher as equation_polisher
from nestynet_sr.sr_search.representation import _sympy_simplify_expression
from nestynet_sr.equation_polisher import (
    ArtifactHints,
    CandidateRecord,
    CandidateSpec,
    PolishConfig,
    PolishResult,
    _artifact_expression_candidates,
    _denominator_coefficient_ratio_snap_specs,
    _radical_coefficient_ratio_snap_specs,
    _recommend,
    _guarded_sympy_candidates,
    _resolve_seed_expr,
    apply_full_dataset_snap_adjudication,
    expression_cost_components,
    expression_complexity,
    load_artifact_hints,
    load_csv_data,
    polish_expression,
)
from nestynet_sr.sr_search.polish_utils import (
    canonicalize_trig_phases,
    final_polish_snap_targets,
    snap_numeric_constants,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec


def _pb005_data(n=2400, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(1.0, 5.0, n)
    x1 = rng.uniform(1.0, 2.0, n)
    x2 = rng.uniform(3.0, 10.0, n)
    X = np.column_stack([x0, x1, x2])
    y_clean = x0 / np.sqrt(1.0 - (x1 / x2) ** 2)
    y = y_clean + rng.normal(0.0, noise, n)
    return X[:1800], y[:1800], X[1800:], y[1800:]


def _pb018_data(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(1.0, 5.0, n)
    x1 = rng.uniform(1.0, 2.0, n)
    x2 = rng.uniform(3.0, 10.0, n)
    X = np.column_stack([x0, x1, x2])
    y = x0 * x1 / np.sqrt(1.0 - (x1 / x2) ** 2)
    return X[:900], y[:900], X[900:], y[900:]


def _pb000_data(n=2400, noise=1.0e-4, seed=1):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(1.0, 3.0, n)
    X = x0.reshape(-1, 1)
    y_clean = np.exp(-(x0**2) / 2.0) / np.sqrt(2.0 * np.pi)
    y = y_clean + rng.normal(0.0, noise, n)
    return X[:1800], y[:1800], X[1800:], y[1800:]


def _pb003_data(n=800, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, size=(n, 4))
    y = np.sqrt((X[:, 1] - X[:, 0]) ** 2 + (X[:, 3] - X[:, 2]) ** 2)
    return X[:600], y[:600], X[600:], y[600:]


def _pb019_data(n=1400, noise=1.0e-3, seed=6):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(1.0, 5.0, n)
    x1 = rng.uniform(1.0, 5.0, n)
    x2 = rng.uniform(1.0, 5.0, n)
    X = np.column_stack([x0, x1, x2])
    y_clean = x0**2 * (x1 + x2) / (x0**2 + x1 * x2)
    y = y_clean + rng.normal(0.0, noise, n)
    return X[:1000], y[:1000], X[1000:], y[1000:]


def _pb113_data(n=1200, seed=113):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(1.0, 5.0, n)
    x1 = rng.uniform(0.0, 6.0, n)
    x2 = rng.uniform(1.0, 5.0, n)
    x3 = rng.uniform(1.0, 5.0, n)
    x4 = rng.uniform(1.0, 5.0, n)
    X = np.column_stack([x0, x1, x2, x3, x4])
    y = x0 * (
        -x2 + x3**3 * (x4 - 1.0) / ((x4 + 2.0) * x2**2)
    ) * np.cos(x1)
    return X[:900], y[:900], X[900:], y[900:]


def _pb112_data(n=1200, seed=112):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(1.0, 5.0, n)
    x1 = rng.uniform(1.0, 5.0, n)
    x2 = rng.uniform(1.0, 5.0, n)
    x3 = rng.uniform(0.3, 2.8, n)
    x4 = rng.uniform(1.0, 5.0, n)
    X = np.column_stack([x0, x1, x2, x3, x4])
    law_cosine = x1**2 - 2.0 * x1 * x2 * np.cos(x3) + x2**2
    y = x0 / (4.0 * np.pi * x4 * np.sqrt(law_cosine))
    return X[:900], y[:900], X[900:], y[900:]


def _record(
    expr: str,
    *,
    val_mse: float,
    val_mse_se: float,
    complexity: float,
    n_free_params: int = 0,
    n_snapped_consts: int = 0,
    label: str | None = None,
):
    return CandidateRecord(
        expr=expr,
        display_expr=expr,
        label=label or expr,
        train_mse=val_mse,
        val_mse=val_mse,
        val_mse_se=val_mse_se,
        complexity=complexity,
        structural_complexity=complexity,
        coefficient_complexity=0.0,
        n_free_params=n_free_params,
        n_snapped_consts=n_snapped_consts,
        frac_valid=1.0,
        seed_nrmse=0.0,
        assumptions=[],
        source_hints=[],
        rewrite_trace=[],
        distance_from_seed=0.0,
    )


def _equivalent(expr_a: str, expr_b: str) -> bool:
    x0, x1, x2 = sp.symbols("x0 x1 x2", positive=True)
    loc = {
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "pi": sp.pi,
        "x0": x0,
        "x1": x1,
        "x2": x2,
    }
    a = sp.sympify(expr_a, locals=loc)
    b = sp.sympify(expr_b, locals=loc)
    return sp.simplify(a - b) == 0


def _equivalent4(expr_a: str, expr_b: str) -> bool:
    x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3", positive=True)
    loc = {
        "sqrt": sp.sqrt,
        "x0": x0,
        "x1": x1,
        "x2": x2,
        "x3": x3,
    }
    a = sp.sympify(expr_a, locals=loc)
    b = sp.sympify(expr_b, locals=loc)
    return sp.simplify(a - b) == 0


def _equivalent5(expr_a: str, expr_b: str) -> bool:
    x0, x1, x2, x3, x4 = sp.symbols("x0 x1 x2 x3 x4", positive=True)
    loc = {
        "log": sp.log,
        "x0": x0,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
    }
    a = sp.sympify(expr_a, locals=loc)
    b = sp.sympify(expr_b, locals=loc)
    return sp.simplify(a - b) == 0


def test_complexity_penalizes_long_floats():
    simple = expression_complexity("x0/sqrt(1 - (x1/x2)**2)")
    ugly = expression_complexity(
        "x0/sqrt(1.000015098813941 + 9.198727922739731e-5*(x1/x2) - "
        "1.000312759455579*(x1/x2)**2)"
    )
    assert ugly > simple


def test_trig_phase_canonicalization_uses_pi_quadrants():
    x = sp.symbols("x")
    assert sp.simplify(canonicalize_trig_phases(sp.sin(x + sp.Float("3.141592653589794"))) + sp.sin(x)) == 0
    assert sp.simplify(canonicalize_trig_phases(sp.cos(x + sp.Float("1.5707963267948966"))) + sp.sin(x)) == 0
    assert sp.simplify(canonicalize_trig_phases(sp.sin(x + sp.Float("1.5707963267948966"))) - sp.cos(x)) == 0


def test_trig_phase_canonicalization_does_not_route_phase_through_pi_powers():
    x = sp.symbols("x")
    raw = sp.sin(x + sp.Float("23.562685"))
    canonical = canonicalize_trig_phases(raw, snap_rel_tol=2.0e-2)

    assert not canonical.has(sp.pi**3)
    assert sp.simplify(canonical - sp.sin(x + 3 * sp.pi / 2)) == 0


def test_stagec_keeps_untouched_raw_trig_candidate_when_snap_fails_gate():
    rng = np.random.default_rng(70)
    X = rng.uniform(0.8, 2.0, size=(256, 3))
    expr = "3.9995730556*x0*sin(0.4999626462*x1*x2 + 1.754e-5)**2"
    y = (
        3.9995730556
        * X[:, 0]
        * np.sin(0.4999626462 * X[:, 1] * X[:, 2] + 1.754e-5) ** 2
    )

    phi_str, _y_str, meta = _sympy_simplify_expression(
        expr,
        model=None,
        val_loader=None,
        device=None,
        Nxvars=3,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        verbose=False,
        precomputed_xs_np=X,
        precomputed_ys_model=y,
    )

    assert meta["accepted"] is True
    assert meta["kind"] == "pretty_print_raw"
    parsed = sp.sympify(
        phi_str,
        locals={f"x{i}": sp.Symbol(f"x{i}") for i in range(3)},
    )
    assert any(abs(float(v) - 1.754e-5) < 1.0e-12 for v in parsed.atoms(sp.Float))


def test_final_polish_scores_the_literal_seed_before_phase_canonicalization():
    rng = np.random.default_rng(97)
    X = rng.uniform(0.5, 1.5, size=(120, 3))
    expr = "x0*(-1.000605*x1*sin(x2 + 23.562685) + 1.000734)"
    y = X[:, 0] * (
        -1.000605 * X[:, 1] * np.sin(X[:, 2] + 23.562685) + 1.000734
    )

    result = polish_expression(
        expr,
        X[:80],
        y[:80],
        X[80:],
        y[80:],
        variable_names=("x0", "x1", "x2"),
        config=PolishConfig(max_candidates=16, max_seconds=2.0, snap_rel_tol=2.0e-2),
    )

    assert result.seed_baseline is not None
    assert result.seed_baseline.val_mse < 1.0e-24
    assert "23.562685" in result.seed_baseline.expr


def test_final_polish_counts_anonymous_seed_float_as_learned_parameter():
    rng = np.random.default_rng(57)
    X = rng.uniform(1.0, 5.0, size=(400, 3))
    exact_scale = 3.0 / (20.0 * np.pi)
    y = exact_scale * X[:, 0] ** 2 / (X[:, 1] * X[:, 2])
    seed = "0.04773626689447054*x0**2/(x1*x2)"

    config = PolishConfig(
        max_candidates=64,
        max_seconds=5.0,
        noise_floor_raw=1.0e-8,
    )
    result = polish_expression(
        seed,
        X[:300],
        y[:300],
        X[300:],
        y[300:],
        variable_names=("x0", "x1", "x2"),
        config=config,
    )

    assert result.seed_baseline is not None
    assert result.seed_baseline.n_free_params == 1
    result, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=("x0", "x1", "x2"),
        config=config,
    )
    exact = [
        rec
        for rec in result.all_candidates
        if _equivalent(rec.expr, "3*x0**2/(20*pi*x1*x2)")
    ]
    assert exact
    assert min(rec.n_free_params for rec in exact) == 0
    assert summary["best_label"] == "full_dataset_snap_symbolic_constants"
    assert result.recommended is not None
    assert _equivalent(result.recommended.expr, "3*x0**2/(20*pi*x1*x2)")


def test_denominator_coefficient_ratio_snap_recovers_pb113_exactly():
    seed = sp.sympify(
        "762688269*x0*(-x2**3*x4 - 2*x2**3 + x3**3*x4 - x3**3)*cos(x1)"
        "/(x2**2*(762688269*x4 + 1525376540))"
    )
    target = sp.sympify(
        "x0*(-x2 + x3**3*(x4 - 1)/((x4 + 2)*x2**2))*cos(x1)"
    )
    config = PolishConfig()

    specs = _denominator_coefficient_ratio_snap_specs(
        seed,
        ("x0", "x1", "x2", "x3", "x4"),
        config,
    )

    exact = [
        spec
        for spec in specs
        if sp.cancel(sp.together(spec.expr - target)) == 0
    ]
    assert len(specs) <= 12
    assert exact
    assert exact[0].n_snapped_consts == 1
    assert exact[0].label.startswith("denominator_coefficient_ratio_snap")
    assert expression_complexity(exact[0].expr, config) < expression_complexity(
        seed,
        config,
    )


def test_final_polish_selects_pb113_denominator_ratio_snap(monkeypatch):
    monkeypatch.setattr(equation_polisher, "aggressive_simplify", None)
    X_train, y_train, X_val, y_val = _pb113_data()
    seed = (
        "762688269*x0*(-x2**3*x4 - 2*x2**3 + x3**3*x4 - x3**3)*cos(x1)"
        "/(x2**2*(762688269*x4 + 1525376540))"
    )
    target = sp.sympify(
        "x0*(-x2 + x3**3*(x4 - 1)/((x4 + 2)*x2**2))*cos(x1)"
    )
    unit_system = UnitSystem(("u0", "u1", "u2", "u3", "u4"))
    units = UnitsSpec(
        unit_system=unit_system,
        x_dims=tuple(
            unit_system.dim(dim)
            for dim in (
                (-1, 0, 0, 0, 1),
                (0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0),
                (1, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
            )
        ),
        y_dim=unit_system.dim((0, 0, 0, 0, 1)),
    )

    result = polish_expression(
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        variable_names=("x0", "x1", "x2", "x3", "x4"),
        units_spec=units,
        config=PolishConfig(max_candidates=64, max_seconds=5.0),
    )

    assert result.recommended is not None
    recommended = sp.sympify(result.recommended.expr)
    assert sp.cancel(sp.together(recommended - target)) == 0
    assert result.recommended.label.startswith(
        "denominator_coefficient_ratio_snap"
    )
    assert result.recommended.val_mse < result.seed_baseline.val_mse


def test_radical_coefficient_ratio_snap_recovers_pb112_nested_reciprocal():
    seed = sp.sympify(
        "0.21399096431481586*x0*sqrt(1/("
        "7.2312037680144936*x1**2"
        " - 14.462407536028962*x1*x2*cos(x3)"
        " + 7.2312037680144723*x2**2"
        "))/x4"
    )
    target = sp.sympify(
        "x0/(4*pi*x4*sqrt("
        "x1**2 - 2*x1*x2*cos(x3) + x2**2"
        "))"
    )

    views = equation_polisher._sqrt_factorizations(seed)
    specs = _radical_coefficient_ratio_snap_specs(
        seed,
        ("x0", "x1", "x2", "x3", "x4"),
        PolishConfig(),
    )

    assert views
    assert float(views[0][2].exp) == pytest.approx(-0.5)
    exact = [
        spec
        for spec in specs
        if sp.simplify(spec.expr - target) == 0
    ]
    assert len(specs) <= 12
    assert exact
    assert exact[0].label.startswith("radical_coefficient_ratio_snap")
    assert expression_complexity(exact[0].expr) < expression_complexity(seed)


def test_final_polish_selects_pb112_radical_ratio_snap(monkeypatch):
    monkeypatch.setattr(equation_polisher, "aggressive_simplify", None)
    X_train, y_train, X_val, y_val = _pb112_data()
    seed = (
        "0.21399096431481586*x0*sqrt(1/("
        "7.2312037680144936*x1**2"
        " - 14.462407536028962*x1*x2*cos(x3)"
        " + 7.2312037680144723*x2**2"
        "))/x4"
    )
    target = (
        "x0/(4*pi*x4*sqrt("
        "x1**2 - 2*x1*x2*cos(x3) + x2**2"
        "))"
    )
    unit_system = UnitSystem(("L", "T", "M"))
    units = UnitsSpec(
        unit_system=unit_system,
        x_dims=(
            unit_system.dim((0, 0, 1)),
            unit_system.dim((1, 0, 0)),
            unit_system.dim((1, 0, 0)),
            unit_system.dim((0, 0, 0)),
            unit_system.dim((0, 1, 0)),
        ),
        y_dim=unit_system.dim((-1, -1, 1)),
    )

    result = polish_expression(
        seed,
        X_train,
        y_train,
        X_val,
        y_val,
        variable_names=("x0", "x1", "x2", "x3", "x4"),
        units_spec=units,
        config=PolishConfig(max_candidates=64, max_seconds=5.0),
    )

    assert result.recommended is not None
    assert _equivalent5(result.recommended.expr, target)
    assert result.recommended.label.startswith(
        "radical_coefficient_ratio_snap"
    )
    assert result.recommended.n_free_params == 0


def test_radical_coefficient_ratio_snap_leaves_named_constants_untouched():
    x0, x1, x2, C0 = sp.symbols("x0 x1 x2 C0")
    expr = x0 / sp.sqrt(C0 * x1**2 - 2 * C0 * x1 * x2 + C0 * x2**2)

    specs = _radical_coefficient_ratio_snap_specs(
        expr,
        ("x0", "x1", "x2"),
        PolishConfig(),
    )

    assert specs == []


def test_radical_ratio_snap_fast_skips_expressions_without_half_powers(
    monkeypatch,
):
    x0, x1 = sp.symbols("x0 x1")

    def fail_if_entered(*_args, **_kwargs):
        raise AssertionError("ordinary expression reached radical setup")

    monkeypatch.setattr(
        equation_polisher,
        "_sqrt_factorizations",
        fail_if_entered,
    )
    config = PolishConfig()
    monkeypatch.setattr(config, "snap_targets", fail_if_entered)

    specs = _radical_coefficient_ratio_snap_specs(
        x0 + 2 * x1,
        ("x0", "x1"),
        config,
    )

    assert specs == []


def test_denominator_ratio_snap_cannot_override_better_exact_seed(monkeypatch):
    monkeypatch.setattr(equation_polisher, "aggressive_simplify", None)
    x0, x1 = sp.symbols("x0 x1")
    seed_expr = x0 / (1_000_000 * x1 + 2_001_000)
    specs = _denominator_coefficient_ratio_snap_specs(
        seed_expr,
        ("x0", "x1"),
        PolishConfig(),
    )
    assert specs

    x0_data = np.linspace(1.0, 3.0, 600)
    x1_data = np.linspace(0.5, 2.5, 600)
    X = np.column_stack([x0_data, x1_data])
    y = x0_data / (1_000_000 * x1_data + 2_001_000)
    result = polish_expression(
        str(seed_expr),
        X[:450],
        y[:450],
        X[450:],
        y[450:],
        variable_names=("x0", "x1"),
        config=PolishConfig(max_candidates=32, max_seconds=5.0),
    )

    assert result.recommended is not None
    recommended = sp.sympify(result.recommended.expr)
    assert sp.cancel(sp.together(recommended - seed_expr)) == 0


def test_denominator_ratio_snap_leaves_named_constants_untouched(monkeypatch):
    x0, x1, C0 = sp.symbols("x0 x1 C0")
    expr = x0 / (10_000 * x1 + 20_001 * C0)
    original_together = equation_polisher.sp.together
    calls = 0

    def tracked_together(candidate):
        nonlocal calls
        calls += 1
        return original_together(candidate)

    monkeypatch.setattr(equation_polisher.sp, "together", tracked_together)

    specs = _denominator_coefficient_ratio_snap_specs(
        expr,
        ("x0", "x1"),
        PolishConfig(),
    )

    assert calls == 1
    assert specs == []


def test_denominator_ratio_snap_fast_skips_ordinary_coefficients(monkeypatch):
    x0, x1 = sp.symbols("x0 x1")

    def fail_if_normalized(_expr):
        raise AssertionError("ordinary expression reached together()")

    monkeypatch.setattr(equation_polisher.sp, "together", fail_if_normalized)

    specs = _denominator_coefficient_ratio_snap_specs(
        x0 / (x1 + 2),
        ("x0", "x1"),
        PolishConfig(),
    )

    assert specs == []


def test_denominator_ratio_snap_fast_skips_large_numerator(monkeypatch):
    x0, x1 = sp.symbols("x0 x1")

    def fail_if_normalized(_expr):
        raise AssertionError("numerator-only large integer reached together()")

    monkeypatch.setattr(equation_polisher.sp, "together", fail_if_normalized)

    specs = _denominator_coefficient_ratio_snap_specs(
        10_001 * x0 / (x1 + 2),
        ("x0", "x1"),
        PolishConfig(),
    )

    assert specs == []


def test_full_dataset_partial_snap_counts_surviving_learned_literal():
    rng = np.random.default_rng(58)
    X = rng.uniform(0.5, 2.0, size=(400, 2))
    y = 1.23456789 * X[:, 0] + 0.5 * X[:, 1]
    seed = "1.23456789*x0 + 0.500001*x1"
    config = PolishConfig(
        # Keep the split-polish frontier to the seed so the partial snap is
        # created specifically by full-dataset adjudication.
        max_candidates=1,
        max_seconds=3.0,
        noise_floor_raw=1.0e-8,
    )
    result = polish_expression(
        seed,
        X[:300],
        y[:300],
        X[300:],
        y[300:],
        variable_names=("x0", "x1"),
        config=config,
    )

    result, _summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=("x0", "x1"),
        config=config,
    )
    partial_snaps = [
        rec
        for rec in result.all_candidates
        if _equivalent(rec.expr, "1.23456789*x0 + x1/2")
    ]
    assert partial_snaps
    assert min(rec.n_free_params for rec in partial_snaps) == 1


def test_symbolic_constant_snap_finds_inverse_four_pi():
    x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3")
    expr = sp.Float("0.0795775") * x1 * sp.cos(x2) / (x0 * x3**2)
    snapped = snap_numeric_constants(
        expr,
        snap_targets=final_polish_snap_targets(),
        snap_rel_tol=5.0e-3,
    )
    assert sp.simplify(snapped - x1 * sp.cos(x2) / (4 * sp.pi * x0 * x3**2)) == 0


def test_symbolic_constant_snap_finds_inverse_two_pi_cubed():
    x0 = sp.symbols("x0")
    expr = sp.Float("0.01612576721659974") * x0
    snapped = snap_numeric_constants(
        expr,
        snap_targets=final_polish_snap_targets(),
        snap_rel_tol=1.0e-4,
    )
    assert sp.simplify(snapped - x0 / (2 * sp.pi**3)) == 0


def test_artifact_path_skips_internal_poly_placeholders_without_sympy_warning():
    hints = ArtifactHints(
        y_transform="identity",
        simplification_path=[
            {"stage": "B", "step": 1, "expression": "sqrt(poly(x0))"},
            {"stage": "C", "step": 2, "expression": "sqrt(x0)"},
            {"stage": "C", "step": 3, "expression": "1/cosh(x0)"},
        ],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        specs = _artifact_expression_candidates(hints, ["x0"], {"x0": ">0"})

    assert not any(issubclass(w.category, SymPyDeprecationWarning) for w in caught)
    labels = {spec.label for spec in specs}
    assert "artifact:path_step_1" not in labels
    assert "artifact:path_step_2" in labels
    assert "artifact:path_step_3" in labels


def test_coefficient_cost_prefers_symbolic_pi_over_ugly_rational():
    symbolic, symbolic_struct, symbolic_coeff = expression_cost_components(
        "x1*cos(x2)/(4*pi*x0*x3**2)"
    )
    ugly_rat, ugly_struct, ugly_coeff = expression_cost_components(
        "113*x1*cos(x2)/(1420*x0*x3**2)"
    )
    short_float, _float_struct, float_coeff = expression_cost_components(
        "0.0795775*x1*cos(x2)/(x0*x3**2)"
    )
    assert symbolic < short_float < ugly_rat
    assert symbolic_coeff < float_coeff < ugly_coeff
    assert symbolic_struct >= ugly_struct


def test_final_polish_recommender_treats_numerical_zero_losses_as_equivalent():
    exact = _record(
        "sqrt(2)*exp(-x0**2/2)/(2*sqrt(pi))",
        val_mse=3.44e-33,
        val_mse_se=7.75e-35,
        complexity=13.18,
    )
    float_refit = _record(
        "0.3989422804014326*exp(-x0**2/2)",
        val_mse=2.22e-33,
        val_mse_se=5.90e-35,
        complexity=13.93,
        n_free_params=1,
    )

    assert (
        _recommend([float_refit, exact], k=1.0, loss_equiv_abs_floor=1.0e-24)
        is exact
    )


def test_full_dataset_snap_adjudication_can_select_pi_candidate():
    current = _record(
        "3.14159*x0",
        val_mse=1.0e-8,
        val_mse_se=1.0e-9,
        complexity=7.0,
        n_free_params=1,
        label="seed",
    )
    current.is_recommended = True
    result = PolishResult(
        seed_expr=current.expr,
        all_candidates=[current],
        strict_pareto=[current],
        epsilon_pareto=[current],
        recommended=current,
        rewrite_trace=[],
        warnings=[],
    )
    x0 = np.linspace(1.0, 3.0, 1000)
    X = x0.reshape(-1, 1)
    y = np.pi * x0

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0"],
        config=PolishConfig(noise_floor_raw=1.0e-8),
    )

    assert summary["status"] == "selected"
    assert summary["n_generated"] >= 1
    assert _equivalent(updated.recommended.expr, "pi*x0")
    assert updated.recommended.full_dataset_snap_selected
    assert updated.recommended.full_dataset_mse == pytest.approx(0.0, abs=1.0e-28)
    assert not current.is_recommended


def test_full_dataset_snap_adjudication_leaves_better_current_recommendation():
    current = _record(
        "2*x0",
        val_mse=0.0,
        val_mse_se=0.0,
        complexity=3.0,
        label="seed",
    )
    snapped = _record(
        "pi*x0",
        val_mse=1.0,
        val_mse_se=0.0,
        complexity=4.0,
        n_snapped_consts=1,
        label="snap_symbolic_constants",
    )
    current.is_recommended = True
    result = PolishResult(
        seed_expr=current.expr,
        all_candidates=[current, snapped],
        strict_pareto=[current],
        epsilon_pareto=[current],
        recommended=current,
        rewrite_trace=[],
        warnings=[],
    )
    x0 = np.linspace(1.0, 2.0, 200)
    X = x0.reshape(-1, 1)
    y = 2.0 * x0

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0"],
        config=PolishConfig(noise_floor_raw=1.0e-8),
    )

    assert summary["status"] == "unchanged"
    assert updated.recommended is current
    assert current.full_dataset_snap_selected


def test_full_dataset_adjudication_retains_unit_valid_seed_when_snap_regresses():
    seed = _record(
        "0.999684*x0*x2*x3/x1**2",
        val_mse=1.0e-8,
        val_mse_se=1.0e-9,
        complexity=10.0,
        n_free_params=1,
        label="seed",
    )
    snapped = _record(
        "x0*x2*x3/x1**2",
        val_mse=9.0e-9,
        val_mse_se=1.0e-9,
        complexity=8.0,
        n_snapped_consts=1,
        label="snap_symbolic_constants",
    )
    snapped.is_recommended = True
    result = PolishResult(
        seed_expr=seed.expr,
        all_candidates=[seed, snapped],
        strict_pareto=[seed, snapped],
        epsilon_pareto=[seed, snapped],
        recommended=snapped,
        rewrite_trace=[],
        warnings=[],
        seed_baseline=seed,
        seed_units_ok=True,
    )
    X = np.ones((400, 4), dtype=np.float64)
    y = np.full(400, 0.999684, dtype=np.float64)
    us = UnitSystem(("L",))
    length = us.dim([1])
    units = UnitsSpec(
        unit_system=us,
        y_dim=length,
        x_dims=(length, length, length, length),
    )

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0", "x1", "x2", "x3"],
        config=PolishConfig(noise_floor_raw=0.0),
        units_spec=units,
    )

    assert summary["status"] == "selected"
    assert summary["selected_label"] == "seed"
    assert summary["n_seed_safe"] == 1
    assert summary["unit_reject_count"] == 0
    assert updated.recommended is seed
    assert seed.full_dataset_snap_selected
    assert seed.is_recommended
    assert seed.full_dataset_mse == pytest.approx(0.0, abs=1.0e-28)
    assert not snapped.is_recommended


def test_full_dataset_adjudication_refuses_regression_from_units_invalid_seed():
    invalid_seed = _record(
        "2*x0 + 0.5*x0/x1",
        val_mse=0.0,
        val_mse_se=0.0,
        complexity=7.0,
        label="seed",
    )
    current = _record(
        "2*x0",
        val_mse=1.0e-6,
        val_mse_se=0.0,
        complexity=3.0,
        label="unit_valid_replacement",
    )
    current.is_recommended = True
    result = PolishResult(
        seed_expr="2*x0 + 0.5*x0/x1",
        all_candidates=[current],
        strict_pareto=[current],
        epsilon_pareto=[current],
        recommended=current,
        rewrite_trace=[],
        warnings=[],
        seed_baseline=invalid_seed,
        seed_units_ok=False,
    )
    x0 = np.linspace(1.0, 2.0, 400)
    x1 = np.linspace(1.5, 3.0, 400)
    X = np.column_stack([x0, x1])
    y = 2.0 * x0 + 0.5 * x0 / x1
    us = UnitSystem(("L",))
    units = UnitsSpec(
        unit_system=us,
        y_dim=us.dim([1]),
        x_dims=(us.dim([1]), us.dim([1])),
    )

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0", "x1"],
        config=PolishConfig(noise_floor_raw=1.0e-8),
        units_spec=units,
    )

    assert summary["status"] == "no_safe_unit_valid_replacement"
    assert summary["seed_full_mse"] == pytest.approx(0.0, abs=1.0e-28)
    assert summary["n_seed_safe"] == 0
    assert summary["unit_reject_count"] == 1
    assert updated.recommended is None
    assert updated.selection_status == "no_safe_unit_valid_replacement"
    assert not current.is_recommended
    assert not current.full_dataset_snap_selected


def test_full_dataset_snap_adjudication_can_select_pi_with_larger_denominator():
    current = _record(
        "179*x0/(3749)",
        val_mse=1.0e-8,
        val_mse_se=1.0e-9,
        complexity=15.0,
        n_free_params=1,
        label="seed",
    )
    current.is_recommended = True
    result = PolishResult(
        seed_expr=current.expr,
        all_candidates=[current],
        strict_pareto=[current],
        epsilon_pareto=[current],
        recommended=current,
        rewrite_trace=[],
        warnings=[],
    )
    x0 = np.linspace(1.0, 5.0, 1000)
    X = x0.reshape(-1, 1)
    y = (3.0 / (20.0 * np.pi)) * x0

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0"],
        config=PolishConfig(noise_floor_raw=1.0e-8),
    )

    assert summary["status"] == "selected"
    assert _equivalent(updated.recommended.expr, "3*x0/(20*pi)")
    assert updated.recommended.full_dataset_mse == pytest.approx(0.0, abs=1.0e-28)


def test_full_dataset_snap_adjudication_refits_and_snaps_global_rational_scale():
    current = _record(
        "-9065*x0/1417",
        val_mse=1.0,
        val_mse_se=0.1,
        complexity=16.0,
        n_free_params=0,
        label="factor_terms",
    )
    current.is_recommended = True
    result = PolishResult(
        seed_expr=current.expr,
        all_candidates=[current],
        strict_pareto=[current],
        epsilon_pareto=[current],
        recommended=current,
        rewrite_trace=[],
        warnings=[],
    )
    x0 = np.linspace(0.5, 5.0, 10000)
    X = x0.reshape(-1, 1)
    y = (-32.0 / 5.0) * x0

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0"],
        config=PolishConfig(noise_floor_raw=1.0e-6),
    )

    assert summary["status"] == "selected"
    assert summary["n_generated"] >= 1
    assert _equivalent(updated.recommended.expr, "-32*x0/5")
    assert updated.recommended.label == "full_dataset_coeff_rational_snap"
    assert updated.recommended.full_dataset_mse == pytest.approx(0.0, abs=1.0e-28)


def test_full_dataset_snap_adjudication_cascades_independent_snaps_pb016():
    current = _record(
        "x2*(1.000063056736845*sqrt(pi)*x0 - 2*sqrt(2)*pi*x1*x3/5)"
        "/(sqrt(pi)*sqrt(-x1**2 + x2**2))",
        val_mse=1.0e-4,
        val_mse_se=1.0e-6,
        complexity=29.0,
        n_free_params=0,
        label="factor_terms|snap_symbolic_constant:-1.262837017498095->-2*pi/5",
    )
    current.is_recommended = True
    result = PolishResult(
        seed_expr=current.expr,
        all_candidates=[current],
        strict_pareto=[current],
        epsilon_pareto=[current],
        recommended=current,
        rewrite_trace=[],
        warnings=[],
    )
    rng = np.random.default_rng(16)
    x0 = rng.uniform(0.5, 2.0, 1200)
    x1 = rng.uniform(0.2, 0.8, 1200)
    x2 = rng.uniform(1.4, 3.0, 1200)
    x3 = rng.uniform(0.5, 2.0, 1200)
    X = np.column_stack([x0, x1, x2, x3])
    y = (x0 - x1 * x3) / np.sqrt(1.0 - (x1 / x2) ** 2)

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0", "x1", "x2", "x3"],
        config=PolishConfig(noise_floor_raw=1.0e-8),
    )

    assert summary["status"] == "selected"
    assert summary["n_generated"] >= 2
    assert _equivalent4(updated.recommended.expr, "x2*(x0 - x1*x3)/sqrt(-x1**2 + x2**2)")
    assert updated.recommended.full_dataset_mse == pytest.approx(0.0, abs=1.0e-28)


def test_full_dataset_snap_adjudication_snaps_generic_exp_quadratic_coefficients():
    current = _record(
        "1.49*exp(0.3335*x0 - 0.6665*x0**2)",
        val_mse=1.0e-5,
        val_mse_se=1.0e-6,
        complexity=18.0,
        n_free_params=3,
        label="aggressive_simplify",
    )
    current.is_recommended = True
    result = PolishResult(
        seed_expr=current.expr,
        all_candidates=[current],
        strict_pareto=[current],
        epsilon_pareto=[current],
        recommended=current,
        rewrite_trace=[],
        warnings=[],
    )
    x0 = np.linspace(0.5, 3.0, 12000)
    X = x0.reshape(-1, 1)
    y = 1.5 * np.exp(x0 / 3.0 - 2.0 * x0**2 / 3.0)

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0"],
        config=PolishConfig(noise_floor_raw=1.0e-6),
    )

    assert summary["status"] == "selected"
    assert updated.recommended.label == "full_dataset_exp_quadratic_coeff_snap"
    assert _equivalent(updated.recommended.expr, "3*exp(x0/3 - 2*x0**2/3)/2")
    assert updated.recommended.full_dataset_mse == pytest.approx(0.0, abs=1.0e-28)


def test_full_dataset_snap_adjudication_snaps_coupled_radical_gauge():
    current = _record(
        "3.14398005126181*sqrt("
        "0.101142*x0**2*x2**2/x1**2 + "
        "0.000407849372546869*x0*x2/x1 - 1"
        ")/x2",
        val_mse=1.0e-5,
        val_mse_se=1.0e-6,
        complexity=28.0,
        n_free_params=3,
        label="final_polish:recommended",
    )
    current.is_recommended = True
    result = PolishResult(
        seed_expr=current.expr,
        all_candidates=[current],
        strict_pareto=[current],
        epsilon_pareto=[current],
        recommended=current,
        rewrite_trace=[],
        warnings=[],
    )
    rng = np.random.default_rng(71)
    x0 = rng.uniform(3.5, 6.0, 12000)
    x1 = rng.uniform(1.0, 1.4, 12000)
    x2 = rng.uniform(1.5, 3.0, 12000)
    X = np.column_stack([x0, x1, x2])
    y = np.sqrt(x0**2 / x1**2 - np.pi**2 / x2**2)

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0", "x1", "x2"],
        config=PolishConfig(noise_floor_raw=1.0e-8),
    )

    assert summary["status"] == "selected"
    assert summary["n_generated"] >= 1
    assert updated.recommended.label == "full_dataset_radical_gauge_snap"
    assert _equivalent(updated.recommended.expr, "sqrt(x0**2/x1**2 - pi**2/x2**2)")
    assert updated.recommended.full_dataset_mse == pytest.approx(0.0, abs=1.0e-28)


def test_full_dataset_radical_gauge_snap_does_not_drop_real_linear_term():
    current = _record(
        "3.14398005126181*sqrt("
        "0.101142*x0**2*x2**2/x1**2 + "
        "0.000407849372546869*x0*x2/x1 - 1"
        ")/x2",
        val_mse=0.0,
        val_mse_se=0.0,
        complexity=28.0,
        n_free_params=3,
        label="final_polish:recommended",
    )
    current.is_recommended = True
    result = PolishResult(
        seed_expr=current.expr,
        all_candidates=[current],
        strict_pareto=[current],
        epsilon_pareto=[current],
        recommended=current,
        rewrite_trace=[],
        warnings=[],
    )
    rng = np.random.default_rng(72)
    x0 = rng.uniform(3.5, 6.0, 6000)
    x1 = rng.uniform(1.0, 1.4, 6000)
    x2 = rng.uniform(1.5, 3.0, 6000)
    X = np.column_stack([x0, x1, x2])
    t = x0 * x2 / x1
    y = 3.14398005126181 * np.sqrt(0.101142 * t**2 + 0.000407849372546869 * t - 1) / x2

    updated, summary = apply_full_dataset_snap_adjudication(
        result,
        X,
        y,
        variable_names=["x0", "x1", "x2"],
        config=PolishConfig(noise_floor_raw=0.0),
    )

    assert summary["status"] == "unchanged"
    assert summary["n_generated"] >= 1
    assert updated.recommended is current
    assert current.full_dataset_snap_selected


def test_artifact_parser_extracts_pb005_hints(tmp_path):
    report = tmp_path / "pb005.report.json"
    report.write_text(
        """
{
  "metadata": {"dataset": "data/pb005.csv"},
  "stageA": {"ast_human": "(x0 * NN[(x1 * (x2)**-1)])", "val_loss": 1.1e-5},
  "stageB": {
    "ast_human": "(x0 * 1/sqrt(poly((x1 * (x2)**-1))))",
    "val_loss": 1.0e-5,
    "params": 3,
    "enabled_patterns": ["sqrt_poly"]
  },
  "stageC": {
    "phi_expr_str": "x0*x2**1.5/sqrt(-1.0003*x1**2*x2 + 1.0*x2**3)",
    "y_expr_str": "x0*x2**1.5/sqrt(-1.0003*x1**2*x2 + 1.0*x2**3)"
  },
  "simplification_path": []
}
""",
        encoding="utf-8",
    )
    log = tmp_path / "pb005.log"
    log.write_text(
        """
[Compound] Selected best variant (z) (monomial) z=(x1 * (x2)**-1) (pattern=(0, 1, -1)), val-loss 1.1e-05
[Stage B]   x1*x2**-1 (kind=ratio, indices=(1, 2))
[Stage C DEBUG] leaf[1] coeffs: [ 1.00001510e+00  9.19872792e-05 -1.00031276e+00]
[Stage C] SymPy variable assumptions from data: x0>0, x1>0, x2>0
[PruneParam]   Param coeffs[1] in 123 (poly): val=9.199e-05, sig=2.570e-05
[PruneParam]   Zeroed coeffs[1] in 123, refit 300 epochs
[PruneParam]   MSE: 1.116e-05 -> 1.116e-05, AIC: -1.0 -> -2.0 -- ACCEPTED
""",
        encoding="utf-8",
    )

    hints = load_artifact_hints(report_json=report, allstages_log=log)

    assert "sqrt_poly" in hints.accepted_patterns
    assert "x1/x2" in hints.compound_exprs
    assert hints.variable_assumptions == {"x0": ">0", "x1": ">0", "x2": ">0"}
    assert hints.initial_sqrt_poly_coeffs == pytest.approx(
        [1.00001510, 9.19872792e-05, -1.00031276]
    )
    assert any(h.index == 1 and h.accepted for h in hints.coefficient_prune_hints)


def test_artifact_parser_ignores_textual_none_y_expr(tmp_path):
    report = tmp_path / "pb020.report.json"
    report.write_text(
        """
{
  "metadata": {"dataset": "data/pb020.csv"},
  "stageA": {"y_transform": "identity"},
  "stageB": {},
  "stageC": {
    "phi_expr_str": "(x0*x2 + x1*x3)/(x0 + x1)",
    "y_expr_str": null
  },
  "simplification_path": []
}
""",
        encoding="utf-8",
    )
    final_human = tmp_path / "pb020.final.human"
    final_human.write_text(
        """
Expression (y-space): None
Expression (φ-space): (x0*x2 + x1*x3)/(x0 + x1)
""",
        encoding="utf-8",
    )

    hints = load_artifact_hints(report_json=report, final_human=final_human)

    assert hints.y_expr is None
    assert hints.phi_expr == "(x0*x2 + x1*x3)/(x0 + x1)"
    assert _resolve_seed_expr(None, hints) == "(x0*x2 + x1*x3)/(x0 + x1)"

    hints_from_final_only = load_artifact_hints(final_human=final_human)
    assert hints_from_final_only.y_expr is None
    assert hints_from_final_only.phi_expr == "(x0*x2 + x1*x3)/(x0 + x1)"
    assert _resolve_seed_expr(None, hints_from_final_only) == "(x0*x2 + x1*x3)/(x0 + x1)"


def test_seed_resolution_does_not_use_phi_for_nonidentity_transform():
    report_hints = load_artifact_hints()
    report_hints.y_transform = "sin"
    report_hints.phi_expr = "x0*x1"
    report_hints.y_expr = None
    report_hints.seed_expr = "x0*x1"

    assert _resolve_seed_expr(None, report_hints) is None
    assert _resolve_seed_expr("asin(x0*x1)", report_hints) == "asin(x0*x1)"


def test_homogeneous_radical_projection_finds_clean_pb005_form():
    Xtr, ytr, Xva, yva = _pb005_data()
    seed = (
        "x0*x2**1.5/sqrt("
        "-1.000312759455579*x1**2*x2**1.0 "
        "+ 9.198727922739731e-5*x1*x2**2.0 "
        "+ 1.000015098813941*x2**3.0)"
    )

    result = polish_expression(
        seed,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=["x0", "x1", "x2"],
        config=PolishConfig(),
    )

    clean = "x0*x2/sqrt(-x1**2 + x2**2)"
    assert result.recommended is not None
    assert _equivalent(result.recommended.expr, clean)
    assert any(_equivalent(rec.expr, clean) for rec in result.all_candidates)


def test_guarded_sympy_second_pass_snaps_after_cancel_pb018():
    x0, x1, x2 = sp.symbols("x0 x1 x2", positive=True)
    seed = (
        sp.Float("0.00396992")
        * x0
        * x1
        / sp.sqrt(sp.Float("1.57603e-05") - sp.Float("1.57603e-05") * x1**2 * x2**-2)
    )

    specs = _guarded_sympy_candidates(seed, ["x0", "x1", "x2"], PolishConfig())

    clean = "x0*x1*x2/sqrt(-x1**2 + x2**2)"
    assert any("cancel|snap_symbolic_constants" in spec.label for spec in specs)
    assert any(_equivalent(sp.sstr(spec.expr), clean) for spec in specs)


def test_final_polish_second_pass_snap_recovers_clean_pb018_form():
    Xtr, ytr, Xva, yva = _pb018_data()
    seed = (
        "0.00396992*x0*x1"
        "/sqrt(1.57603e-05 - 1.57603e-05*x1**2*x2**-2)"
    )

    result = polish_expression(
        seed,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=["x0", "x1", "x2"],
        config=PolishConfig(max_candidates=64),
    )

    clean = "x0*x1*x2/sqrt(-x1**2 + x2**2)"
    assert result.recommended is not None
    assert _equivalent(result.recommended.expr, clean)
    # The clean form must be on the frontier from a non-seed producer.  Which
    # battery reaches it first has changed over time (guarded-sympy second
    # pass, radical ratio snap, drop-addend d0 recalibration) and the scoring
    # loop deduplicates later spellings, so the assertion is producer-agnostic.
    assert any(
        rec.label != "seed" and _equivalent(rec.expr, clean)
        for rec in result.all_candidates
    )


def test_exp_poly_projector_finds_clean_pb000_gaussian():
    Xtr, ytr, Xva, yva = _pb000_data()
    seed = (
        "acos((0.002776714166610668*x0**4 - 0.02900423023370854*x0**3 "
        "+ 0.113818880700647*x0**2 - 0.7408109636137355)"
        "/(0.1991422802458677*x0 - 0.8720962089549625))"
    )

    result = polish_expression(
        seed,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=["x0"],
        config=PolishConfig(),
    )

    clean = "exp(-x0**2/2)/sqrt(2*pi)"
    assert result.recommended is not None
    assert _equivalent(result.recommended.expr, clean)
    assert any(_equivalent(rec.expr, clean) for rec in result.all_candidates)


def test_noisy_sparse_rational_refit_finds_clean_pb019_form():
    Xtr, ytr, Xva, yva = _pb019_data()
    seed = (
        "(1.01*x0**2*x1 + 0.99*x0**2*x2 + 0.001*x1**2)"
        "/(0.98*x0**2 + 1.02*x1*x2 + 0.001*x2**2)"
    )

    result = polish_expression(
        seed,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=["x0", "x1", "x2"],
        config=PolishConfig(max_candidates=64, noise_floor_raw=1.0e-6),
    )

    clean = "x0**2*(x1 + x2)/(x0**2 + x1*x2)"
    assert any(_equivalent(rec.expr, clean) for rec in result.all_candidates)


def test_units_filter_prefers_dimensionally_valid_log_ratio_pb047():
    rng = np.random.default_rng(3)
    X = rng.uniform(1.0, 5.0, size=(320, 5))
    y = X[:, 0] * X[:, 1] * X[:, 2] * np.log(X[:, 4] / X[:, 3])
    us = UnitSystem(("L", "T", "M", "I", "Theta"))
    units = UnitsSpec(
        unit_system=us,
        y_dim=us.dim([2, -2, 1, 0, 0]),
        x_dims=tuple(
            us.dim(v)
            for v in [
                [0, 0, 0, 0, 0],
                [2, -2, 1, -1, 0],
                [0, 0, 0, 1, 0],
                [3, 0, 0, 0, 0],
                [3, 0, 0, 0, 0],
            ]
        ),
    )
    seed = "-x0*x1*x2*log(x3) + x0*x1*x2*log(x4)"

    result = polish_expression(
        seed,
        X[:240],
        y[:240],
        X[240:],
        y[240:],
        variable_names=["x0", "x1", "x2", "x3", "x4"],
        units_spec=units,
        config=PolishConfig(max_candidates=64),
    )

    clean = "x0*x1*x2*log(x4/x3)"
    assert result.recommended is not None
    assert _equivalent5(result.recommended.expr, clean)
    assert any(_equivalent5(rec.expr, clean) for rec in result.all_candidates)
    assert not any("**(x0*x1*x2)" in rec.expr for rec in result.all_candidates)
    assert any("units filter rejected" in w for w in result.warnings)


def test_units_invalid_seed_remains_loss_incumbent_but_is_not_promoted(
    monkeypatch,
):
    x0 = np.linspace(1.0, 2.0, 240)
    x1 = np.linspace(1.5, 3.0, 240)
    X = np.column_stack([x0, x1])
    y = x0 + 0.5 * x0 / x1
    us = UnitSystem(("L",))
    units = UnitsSpec(
        unit_system=us,
        y_dim=us.dim([1]),
        x_dims=(us.dim([1]), us.dim([1])),
    )

    monkeypatch.setattr(
        equation_polisher,
        "_guarded_sympy_candidates",
        lambda *_args, **_kwargs: [
            CandidateSpec(sp.Symbol("x0"), label="unit_valid_but_worse")
        ],
    )
    monkeypatch.setattr(
        equation_polisher,
        "_sparse_rational_seed_support_candidates",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        equation_polisher,
        "_homogeneous_radical_candidates",
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        equation_polisher,
        "_fit_exp_poly_templates",
        lambda *_args, **_kwargs: [],
    )

    result = polish_expression(
        "x0 + 0.5*x0/x1",
        X[:180],
        y[:180],
        X[180:],
        y[180:],
        variable_names=["x0", "x1"],
        units_spec=units,
        config=PolishConfig(max_candidates=8, use_artifact_hints=False),
    )

    assert result.seed_baseline is not None
    assert result.seed_baseline.val_mse == pytest.approx(0.0, abs=1.0e-28)
    assert result.seed_units_ok is False
    assert "add-dim-mismatch" in result.seed_units_reason
    assert any(rec.label == "unit_valid_but_worse" for rec in result.all_candidates)
    assert result.recommended is None
    assert result.selection_status == "no_safe_unit_valid_replacement"
    assert "worsens raw seed validation loss" in result.selection_reason


def test_seed_guard_chooses_safe_candidate_when_simplest_tied_one_is_unsafe(
    monkeypatch,
):
    us = UnitSystem(("L",))
    units = UnitsSpec(
        unit_system=us,
        y_dim=us.dim([1]),
        x_dims=(us.dim([1]), us.dim([1])),
    )
    X = np.ones((20, 2), dtype=np.float64)
    y = np.ones(20, dtype=np.float64)

    monkeypatch.setattr(
        equation_polisher,
        "_guarded_sympy_candidates",
        lambda *_args, **_kwargs: [
            CandidateSpec(sp.Symbol("x0"), label="safe_loss"),
            CandidateSpec(2 * sp.Symbol("x0"), label="simple_but_unsafe_loss"),
        ],
    )
    monkeypatch.setattr(
        equation_polisher,
        "_sparse_rational_seed_support_candidates",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        equation_polisher,
        "_homogeneous_radical_candidates",
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        equation_polisher,
        "_fit_exp_poly_templates",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        equation_polisher,
        "_drop_addend_refit_specs",
        lambda *_args, **_kwargs: [],
    )

    def fake_score(spec, *_args, **_kwargs):
        if spec.label == "seed":
            return _record(
                str(spec.expr),
                val_mse=1.0,
                val_mse_se=0.0,
                complexity=10.0,
                label=spec.label,
            )
        if spec.label == "safe_loss":
            return _record(
                str(spec.expr),
                val_mse=0.5,
                val_mse_se=10.0,
                complexity=5.0,
                label=spec.label,
            )
        return _record(
            str(spec.expr),
            val_mse=5.0,
            val_mse_se=0.0,
            complexity=1.0,
            label=spec.label,
        )

    monkeypatch.setattr(equation_polisher, "_score_candidate", fake_score)

    result = polish_expression(
        "x0 + x0/x1",
        X[:10],
        y[:10],
        X[10:],
        y[10:],
        variable_names=["x0", "x1"],
        units_spec=units,
        config=PolishConfig(max_candidates=8, use_artifact_hints=False),
    )

    assert result.recommended is not None
    assert result.recommended.label == "safe_loss"
    assert result.selection_status == "selected"


def test_artifact_stageb_expression_can_win_over_expanded_stagec_pb003():
    Xtr, ytr, Xva, yva = _pb003_data()
    expanded = "sqrt(x0**2 - 2*x0*x1 + x1**2 + x2**2 - 2*x2*x3 + x3**2)"
    compact = "sqrt((((x0 + (-1 * x1)))**2 + ((x2 + (-1 * x3)))**2))"
    hints = ArtifactHints(
        y_transform="identity",
        stageB_expr=compact,
        simplification_path=[
            {"step": 3, "stage": "B", "expression": compact},
            {"step": 4, "stage": "C", "expression": expanded},
        ],
    )

    result = polish_expression(
        expanded,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=["x0", "x1", "x2", "x3"],
        artifact_hints=hints,
        config=PolishConfig(max_candidates=32),
    )

    clean = "sqrt((x0 - x1)**2 + (x2 - x3)**2)"
    assert result.recommended is not None
    assert _equivalent4(result.recommended.expr, clean)
    assert result.recommended.complexity < expression_complexity(expanded)
    assert any(rec.label == "artifact:stageB_expr" for rec in result.all_candidates)


def test_phi_space_stageb_artifact_is_not_scored_for_nonidentity_y_transform():
    rng = np.random.default_rng(4)
    X = rng.uniform(0.0, 0.8, size=(320, 2))
    y = np.arcsin(X[:, 0] * X[:, 1])
    hints = ArtifactHints(
        y_transform="sin",
        stageB_expr="x0*x1",
        y_expr="asin(x0*x1)",
    )

    result = polish_expression(
        "asin(x0*x1)",
        X[:240],
        y[:240],
        X[240:],
        y[240:],
        variable_names=["x0", "x1"],
        artifact_hints=hints,
        config=PolishConfig(max_candidates=32),
    )

    assert result.recommended is not None
    assert not any(rec.label == "artifact:stageB_expr" for rec in result.all_candidates)


def test_srbench_pb005_artifact_integration_if_available():
    base = Path("../SRBench_0.001")
    data = base / "data/pb005_I_10_7_data.csv"
    report = base / "results/pb005_I_10_7_data.report.json"
    decisions = base / "results/pb005_I_10_7_data.decisions.json"
    log = base / "results/pb005_I_10_7_data_allstages.log"
    if not (data.exists() and report.exists() and decisions.exists() and log.exists()):
        pytest.skip("SRBench pb005 artifacts are not available")

    hints = load_artifact_hints(report_json=report, decisions_json=decisions, allstages_log=log)
    Xtr, ytr, Xva, yva, names = load_csv_data(data, max_rows=5000, seed=3)
    result = polish_expression(
        hints.y_expr,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=names,
        artifact_hints=hints,
        config=PolishConfig(),
    )

    clean = "x0*x2/sqrt(-x1**2 + x2**2)"
    assert hints.y_expr is not None
    assert "x1/x2" in hints.compound_exprs
    assert result.recommended is not None
    assert _equivalent(result.recommended.expr, clean)


def test_srbench_pb000_artifact_integration_if_available():
    base = Path("../SRBench_0.001")
    data = base / "data/pb000_I_6_2a_data.csv"
    report = base / "results/pb000_I_6_2a_data.report.json"
    decisions = base / "results/pb000_I_6_2a_data.decisions.json"
    log = base / "results/pb000_I_6_2a_data_allstages.log"
    if not (data.exists() and report.exists() and decisions.exists() and log.exists()):
        pytest.skip("SRBench pb000 artifacts are not available")

    hints = load_artifact_hints(report_json=report, decisions_json=decisions, allstages_log=log)
    Xtr, ytr, Xva, yva, names = load_csv_data(data, max_rows=5000, seed=3)
    result = polish_expression(
        hints.y_expr,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=names,
        artifact_hints=hints,
        config=PolishConfig(),
    )

    clean = "exp(-x0**2/2)/sqrt(2*pi)"
    assert hints.y_expr is not None
    assert hints.accepted_patterns or hints.candidate_family_hints
    assert result.recommended is not None
    assert _equivalent(result.recommended.expr, clean)


# ---------------------------------------------------------------------------
# Drop-small-addend refit battery
# ---------------------------------------------------------------------------


def _pb114_analog_data(n=3000, noise_frac=0.001, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(2.0, 4.0, n)
    x1 = rng.uniform(0.1, 1.0, n)
    x2 = rng.uniform(0.5, 2.0, n)
    x3 = rng.uniform(0.0, 3.0, n)
    X = np.column_stack([x0, x1, x2, x3])
    y_clean = (
        x2 * np.sqrt(x0**2 - x1**2) * x0**1.5 / (x0**2.5 + x0**1.5 * x1 * np.cos(x3))
    )
    sigma = noise_frac * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    ntr = int(0.75 * n)
    return X[:ntr], y[:ntr], X[ntr:], y[ntr:], sigma


_PB114_ANALOG_SEED = (
    "23328772*x0**(3/2)*x2*sqrt(x0**2 - x1**2)/(23328772*x0**(5/2) "
    "+ 23596222*x0**(3/2)*x1*cos(x3) "
    "- 179905*sqrt(2)*x1*sqrt(x0 + x1)*sqrt(x0**2 - x1**2)*cos(x3))"
)


def _drop_refit_config(sigma, n_val):
    return PolishConfig(
        noise_floor_raw=float(sigma**2),
        loss_equiv_abs_floor=float(sigma**2 * np.sqrt(2.0 / n_val)),
    )


def test_drop_addend_refit_recovers_pb114_denominator_soft_direction():
    """The coupled drop+refit move must recover the exact truth spelling.

    The spurious denominator term cancels a miscalibrated companion
    coefficient (ratio 1.0115 where the truth has exactly 1), so plain
    zeroing without a refit is much worse than the seed, while drop+refit+
    re-snap collapses the rationalized eight-digit integers entirely.
    """

    Xtr, ytr, Xva, yva, sigma = _pb114_analog_data()
    result = polish_expression(
        _PB114_ANALOG_SEED,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=["x0", "x1", "x2", "x3"],
        config=_drop_refit_config(sigma, len(yva)),
    )
    drop_records = [
        r for r in result.all_candidates if r.label.startswith("drop_addend_refit")
    ]
    assert drop_records, "drop-addend refit battery emitted no candidates"
    truth = sp.sympify("x2*sqrt(x0**2 - x1**2)/(x0 + x1*cos(x3))")
    winners = [
        r
        for r in drop_records
        if sp.simplify(sp.sympify(r.expr) - truth) == 0
    ]
    assert winners, "no drop candidate is algebraically identical to the truth"
    assert any(r.n_free_params == 0 for r in winners)
    assert result.recommended is not None
    assert sp.simplify(sp.sympify(result.recommended.expr) - truth) == 0
    seed_cx = result.seed_baseline.complexity
    assert result.recommended.complexity < seed_cx


def test_drop_addend_sites_exclude_exponent_position():
    x0, x1, x2 = sp.symbols("x0 x1 x2", positive=True)
    expr = x2 ** (x0 + x1) + x0 + x1
    nodes = equation_polisher._value_position_add_nodes(expr)
    reprs = {sp.srepr(node) for node in nodes}
    assert sp.srepr(x0 + x1) not in reprs or len(nodes) == 1
    # The top-level Add is a value-position site; the exponent Add is not a
    # separate site beyond it.
    assert len(nodes) == 1


def test_addend_numeric_coefficient_handles_all_representations():
    x0 = sp.Symbol("x0", positive=True)
    value, coeff, rest = equation_polisher._addend_numeric_coefficient(
        sp.Integer(179905) * sp.sqrt(2) * x0
    )
    assert abs(value - 179905.0 * np.sqrt(2.0)) < 1.0e-9
    assert rest == x0
    value2, _c2, rest2 = equation_polisher._addend_numeric_coefficient(
        sp.Rational(5, 7) * sp.sqrt(2) * x0
    )
    assert abs(value2 - float(5 * np.sqrt(2.0) / 7)) < 1.0e-12
    assert rest2 == x0
    value3, _c3, rest3 = equation_polisher._addend_numeric_coefficient(x0**2)
    assert value3 == 1.0
    assert rest3 == x0**2


def test_drop_addend_refit_removes_subnoise_radicand_term():
    """pb071-style: a sub-noise linear term inside a sqrt radicand is dropped."""

    rng = np.random.default_rng(1)
    n = 3000
    x0 = rng.uniform(8.0, 12.0, n)
    x1 = rng.uniform(1.0, 2.0, n)
    x2 = rng.uniform(2.0, 4.0, n)
    X = np.column_stack([x0, x1, x2])
    y_clean = np.sqrt(x0**2 / x1**2 - np.pi**2 / x2**2)
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    seed = (
        "0.0448967643799847*sqrt(495.9767748*x0**2*x2**2/x1**2 "
        "+ 2*x0*x2/x1 - 4903.75)/x2"
    )
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1", "x2"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    drop_records = [
        r for r in result.all_candidates if r.label.startswith("drop_addend")
    ]
    assert drop_records
    # At least one drop candidate removed the spurious linear cross term.
    linear = sp.sympify("x0*x2/x1")
    removed = []
    for r in drop_records:
        expr = sp.sympify(r.expr)
        radicands = [
            node.base
            for node in sp.preorder_traversal(expr)
            if isinstance(node, sp.Pow) and node.exp == sp.Rational(1, 2)
        ]
        if radicands and all(
            sp.expand(rad).coeff(linear) == 0 for rad in radicands
        ):
            removed.append(r)
    assert removed, "no candidate removed the sub-noise radicand term"
    best = min(removed, key=lambda r: r.val_mse)
    assert best.val_mse <= result.seed_baseline.val_mse * (1.0 + 5.0e-2)
    assert best.complexity < result.seed_baseline.complexity


def test_drop_addend_refit_iterates_over_two_spurious_terms():
    rng = np.random.default_rng(2)
    n = 3000
    x0 = rng.uniform(0.5, 1.5, n)
    x1 = rng.uniform(0.5, 1.5, n)
    X = np.column_stack([x0, x1])
    y_clean = x0 * x1 / (1.0 + x0)
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    seed = "x0*x1/(1 + x0 + 0.0011*x0**2 + 0.0009*x0*x1)"
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    truth = sp.sympify("x0*x1/(1 + x0)")
    matches = [
        r
        for r in result.all_candidates
        if r.label.startswith("drop_addend")
        and sp.simplify(sp.sympify(r.expr) - truth) == 0
    ]
    assert matches, "no drop candidate reached the double-drop truth"
    assert result.recommended is not None
    assert sp.simplify(sp.sympify(result.recommended.expr) - truth) == 0


def test_drop_addend_refit_does_not_drop_real_small_term():
    """A real small term whose contribution exceeds the noise band survives."""

    rng = np.random.default_rng(3)
    n = 2400
    x0 = rng.uniform(1.0, 2.0, n)
    x1 = rng.uniform(1.0, 2.0, n)
    X = np.column_stack([x0, x1])
    y_clean = x0 + 0.05 * x1
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    seed = "x0 + 0.05*x1"
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    assert result.recommended is not None
    truth = sp.sympify(seed)
    assert sp.simplify(sp.sympify(result.recommended.expr) - truth) == 0


def test_drop_addend_refit_units_compliant_and_preserves_named_symbols():
    us = UnitSystem(("L", "T"))
    units = UnitsSpec(
        unit_system=us,
        y_dim=us.dim([1, 0]),
        x_dims=(us.dim([1, 0]), us.dim([1, 0])),
    )
    rng = np.random.default_rng(4)
    n = 2400
    x0 = rng.uniform(1.0, 2.0, n)
    x1 = rng.uniform(1.0, 2.0, n)
    X = np.column_stack([x0, x1])
    y_clean = x0 + 2.0 * x1
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n) + 0.001 * x0
    seed = "x0 + 2.0*x1 + 0.001*x0"
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1"],
        units_spec=units,
        config=_drop_refit_config(sigma, n - ntr),
    )
    # Any emitted drop candidate passed the units filter by construction
    # (candidates are only scored after _sympy_expr_units_check).
    for r in result.all_candidates:
        if r.label.startswith("drop_addend"):
            ok, _reason = equation_polisher._sympy_expr_units_check(
                sp.sympify(r.expr), ["x0", "x1"], units
            )
            assert ok, f"units-invalid drop candidate emitted: {r.expr}"


def test_drop_addend_refit_never_judges_named_symbol_terms():
    """Terms carrying non-variable symbols are never proposed for deletion."""

    x0, x1 = sp.symbols("x0 x1", positive=True)
    c0 = sp.Symbol("c0", real=True)
    expr = x0 + 0.001 * x1 + c0 * x1
    rng = np.random.default_rng(5)
    X = rng.uniform(1.0, 2.0, size=(600, 2))
    y = X[:, 0] + 0.001 * X[:, 1] + 2.0 * X[:, 1]
    cfg = PolishConfig(symbol_values={"c0": 2.0})
    specs = equation_polisher._drop_addend_refit_specs(
        expr, X, y, ["x0", "x1"], None, cfg
    )
    for spec in specs:
        assert sp.Symbol("c0", real=True) in spec.expr.free_symbols or "c0" in str(
            spec.expr
        ), f"named-symbol term was dropped: {spec.expr}"


def test_drop_addend_refit_spec_list_is_deterministic():
    Xtr, ytr, _Xva, _yva, sigma = _pb114_analog_data(n=1600)
    expr = equation_polisher.parse_sympy_expr(
        _PB114_ANALOG_SEED, ["x0", "x1", "x2", "x3"]
    )
    cfg = _drop_refit_config(sigma, 400)
    specs_a = equation_polisher._drop_addend_refit_specs(
        expr, Xtr, ytr, ["x0", "x1", "x2", "x3"], None, cfg
    )
    specs_b = equation_polisher._drop_addend_refit_specs(
        expr, Xtr, ytr, ["x0", "x1", "x2", "x3"], None, cfg
    )
    assert [s.label for s in specs_a] == [s.label for s in specs_b]
    assert [sp.srepr(s.expr) for s in specs_a] == [
        sp.srepr(s.expr) for s in specs_b
    ]


def test_drop_addend_refit_disabled_by_config_flag():
    Xtr, ytr, Xva, yva, sigma = _pb114_analog_data(n=1600)
    cfg = _drop_refit_config(sigma, len(yva))
    cfg = equation_polisher.replace(cfg, enable_drop_addend_refit=False)
    result = polish_expression(
        _PB114_ANALOG_SEED,
        Xtr,
        ytr,
        Xva,
        yva,
        variable_names=["x0", "x1", "x2", "x3"],
        config=cfg,
    )
    assert not any(
        r.label.startswith("drop_addend") for r in result.all_candidates
    )


def test_drop_addend_float_candidate_declares_frozen_selection_params():
    """Float-coefficient drop candidates must declare selection_n_free_params=0.

    The frontier's conservative anonymous-float count stays (display), but the
    statistical-selection archive must charge frozen refitted literals like
    Stage-B/C fitted constants (0 free parameters), or the candidate sorts
    after the seed in the identification walk despite lower risk (pb071).
    """

    rng = np.random.default_rng(1)
    n = 2400
    x0 = rng.uniform(8.0, 12.0, n)
    x1 = rng.uniform(1.0, 2.0, n)
    x2 = rng.uniform(2.0, 4.0, n)
    X = np.column_stack([x0, x1, x2])
    y_clean = np.sqrt(x0**2 / x1**2 - np.pi**2 / x2**2)
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    seed = (
        "0.0448967643799847*sqrt(495.9767748*x0**2*x2**2/x1**2 "
        "+ 2*x0*x2/x1 - 4903.75)/x2"
    )
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1", "x2"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    drop_records = [
        r for r in result.all_candidates if r.label.startswith("drop_addend")
    ]
    assert drop_records
    for r in drop_records:
        assert r.selection_n_free_params == 0
    # The report row serialization must carry the declaration downstream.
    from nestynet_sr.run_sr_reports import _polish_record_for_report

    row = _polish_record_for_report(drop_records[0])
    assert row.get("selection_n_free_params") == 0


def test_drop_addend_radical_gauge_snap_reaches_exact_truth_pb071():
    """The gauge-normalized parameterization must reach byte-exact constants.

    In the raw gauge the fitted literals (0.0449, 496.8, 4903.75) have no snap
    targets; only the gauge combinations do (c**2*a -> 1, ratio -> -pi**2).
    Normalizing the radicand by its positive leading coefficient and absorbing
    a**p into the top scale exposes them to per-parameter snapping.
    """

    rng = np.random.default_rng(1)
    n = 3000
    x0 = rng.uniform(8.0, 12.0, n)
    x1 = rng.uniform(1.0, 2.0, n)
    x2 = rng.uniform(2.0, 4.0, n)
    X = np.column_stack([x0, x1, x2])
    y_clean = np.sqrt(x0**2 / x1**2 - np.pi**2 / x2**2)
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    seed = (
        "0.0448967643799847*sqrt(495.9767748*x0**2*x2**2/x1**2 "
        "+ 2*x0*x2/x1 - 4903.75)/x2"
    )
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1", "x2"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    assert result.recommended is not None
    assert result.recommended.label.startswith("drop_addend_refit_snap")
    assert result.recommended.n_free_params == 0
    px0, px1, px2 = sp.symbols("x0 x1 x2", positive=True)
    rec = sp.sympify(result.recommended.expr, locals={"x0": px0, "x1": px1, "x2": px2})
    truth = sp.sqrt(px0**2 / px1**2 - sp.pi**2 / px2**2)
    assert sp.simplify(rec - truth) == 0


def test_drop_addend_gauge_handles_nested_pow_chain_pb112():
    """sqrt(1/A) parses as (A**-1)**(1/2); the gauge must follow the chain.

    pb112-style: the drop battery finds exact hidden ratios inside a nested
    inverse-sqrt radicand, and the chain-exponent gauge exposes them plus the
    1/(4*pi) prefactor to per-parameter snapping.
    """

    rng = np.random.default_rng(7)
    n = 3000
    x1 = rng.uniform(3.0, 6.0, n)
    x2 = rng.uniform(0.5, 1.5, n)
    x3 = rng.uniform(0.0, 3.0, n)
    x0 = rng.uniform(1.0, 2.0, n)
    x4 = rng.uniform(1.0, 2.0, n)
    X = np.column_stack([x0, x1, x2, x3, x4])
    y_clean = x0 / (4 * np.pi * x4 * np.sqrt(x1**2 - 2 * x1 * x2 * np.cos(x3) + x2**2))
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    # pb112-style seed: correct nested structure + one small spurious radicand
    # term + miscalibrated companions, all in the raw float gauge.
    seed = (
        "0.0795938*x0*sqrt(1/(0.0009*cos(x3) "
        "+ 1.00021 - 2.0004*x2*cos(x3)/x1 "
        "+ 1.00034*x2**2/x1**2))/(x1*x4)"
    )
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1", "x2", "x3", "x4"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    truth = y_clean
    vs = [sp.Symbol(name) for name in ["x0", "x1", "x2", "x3", "x4"]]
    matches = []
    for r in result.all_candidates:
        if not r.label.startswith("drop_addend"):
            continue
        try:
            fn = sp.lambdify(vs, sp.sympify(r.expr), "numpy")
            pred = np.asarray(fn(x0, x1, x2, x3, x4), dtype=float)
        except Exception:
            continue
        m = np.isfinite(pred)
        if m.sum() > 100 and np.all(
            np.abs(pred[m] - truth[m]) <= 1.0e-9 * np.maximum(1e-12, np.abs(truth[m]))
        ):
            matches.append(r)
    assert matches, "gauge chain did not expose the exact pb112 constants"
    assert any(r.n_free_params == 0 for r in matches)
    assert any("pi" in r.expr for r in matches)


def test_drop_addend_d0_recalibration_recovers_pb107_compton():
    """Zero-drop recalibration: a rationalized-integer pair hiding exact 2:1
    ratios must be recovered even when there is no term to drop."""

    rng = np.random.default_rng(3)
    n = 3000
    x0 = rng.uniform(0.5, 2.0, n)
    x1 = rng.uniform(0.5, 2.0, n)
    x2 = rng.uniform(0.5, 2.0, n)
    x3 = rng.uniform(0.0, 3.0, n)
    X = np.column_stack([x0, x1, x2, x3])
    y_clean = x0 / (x0 * (1 - np.cos(x3)) / (x1 * x2**2) + 1)
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    seed = "23217942*x0*x1*x2**2/(46428063*x0*sin(x3/2)**2 + 23213489*x1*x2**2)"
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1", "x2", "x3"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    assert result.recommended is not None
    assert ":d0" in result.recommended.label
    assert result.recommended.n_free_params == 0
    truth = sp.sympify("x0*x1*x2**2/(2*x0*sin(x3/2)**2 + x1*x2**2)")
    assert sp.simplify(sp.sympify(result.recommended.expr) - truth) == 0


def test_drop_addend_d0_not_emitted_for_plain_sites():
    """d0 recalibration is gated to anchor/gauge sites; a plain polynomial
    seed with float coefficients must not spawn recalibration candidates."""

    rng = np.random.default_rng(4)
    n = 1200
    x0 = rng.uniform(1.0, 2.0, n)
    x1 = rng.uniform(1.0, 2.0, n)
    X = np.column_stack([x0, x1])
    y_clean = 1.37 * x0 + 0.62 * x1
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    seed = "1.37*x0 + 0.62*x1"
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    assert not any(":d0" in r.label for r in result.all_candidates)


def test_drop_addend_d0_per_addend_gauge_recovers_pb115():
    """Gauge site inside one addend of a top-level Add (pb115): the a**p
    compensation folds into that addend's coefficient and sibling addend
    coefficients recalibrate alongside, reaching the exact truth."""

    rng = np.random.default_rng(5)
    n = 3000
    cols = [rng.uniform(0.5, 2.0, n) for _ in range(6)]
    x0, x1, x2, x3, x4, x5 = cols
    X = np.column_stack(cols)
    y_clean = x3 * x5 + x1 * np.sqrt(x0**2 * x1**2 + (x2 - x3 * x4) ** 2)
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    seed = (
        "0.99862360354116856*x1*sqrt(1.0027355123640573*x0**2*x1**2 + x2**2 "
        "- 2.0034780671053891*x2*x3*x4 + 2*sqrt(2)*sqrt(pi)*x3**2*x4**2/5) "
        "+ 1.0*x3*x5"
    )
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1", "x2", "x3", "x4", "x5"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    assert result.recommended is not None
    assert ":d0" in result.recommended.label
    assert result.recommended.n_free_params == 0
    truth = sp.sympify("x3*x5 + x1*sqrt(x0**2*x1**2 + (x2 - x3*x4)**2)")
    assert sp.simplify(sp.sympify(result.recommended.expr) - truth) == 0


def test_drop_addend_exp_rate_refit_recovers_pb079_sech():
    """Miscalibrated exp-argument rates (a two-sided exponential written with
    rate imposters 2*sqrt(2)*sqrt(pi)/5 and 18361/9171) must refit and snap to
    the exact integers, collapsing to the sech form."""

    rng = np.random.default_rng(9)
    n = 3000
    x0 = rng.uniform(0.5, 2.0, n)
    x1 = rng.uniform(0.8, 2.0, n)
    x2 = rng.uniform(0.8, 2.0, n)
    x3 = rng.uniform(0.2, 1.5, n)
    x4 = rng.uniform(0.2, 1.5, n)
    X = np.column_stack([x0, x1, x2, x3, x4])
    z = x3 * x4 / (x1 * x2)
    y_clean = x0 / (np.exp(-z) + np.exp(z))
    sigma = 0.001 * float(np.sqrt(np.mean(y_clean**2)))
    y = y_clean + rng.normal(0.0, sigma, n)
    seed = (
        "3388*x0*exp(2*sqrt(2)*sqrt(pi)*x3*x4/(5*x1*x2))"
        "/(3389*exp(18361*x3*x4/(9171*x1*x2)) + 3389)"
    )
    ntr = int(0.75 * n)
    result = polish_expression(
        seed,
        X[:ntr],
        y[:ntr],
        X[ntr:],
        y[ntr:],
        variable_names=["x0", "x1", "x2", "x3", "x4"],
        config=_drop_refit_config(sigma, n - ntr),
    )
    assert result.recommended is not None
    assert result.recommended.label.startswith("drop_addend_refit_snap")
    truth = sp.sympify(
        "x0*exp(x3*x4/(x1*x2))/(exp(2*x3*x4/(x1*x2)) + 1)"
    )
    assert sp.simplify(sp.sympify(result.recommended.expr) - truth) == 0


def test_drop_addend_exp_rate_leaves_structural_rates_frozen():
    """exp(z/2)-style small exact rational rates are structure, not fit
    residue: the rate parameterizer must not touch them, and variable-base
    Pow exponents stay excluded entirely."""

    x0, x1 = sp.symbols("x0 x1", positive=True)
    expr = 2 * sp.exp(x0 / 2) + sp.exp(sp.Rational(3, 4) * x1) + x0 ** sp.Rational(5, 2)
    params: list = []
    inits: list = []
    out, n_rates = equation_polisher._exp_rate_parameterize(
        expr, params, inits, PolishConfig()
    )
    assert n_rates == 0
    assert out == expr
    # A float rate IS refittable.
    expr2 = sp.exp(sp.Float("1.0027") * x0)
    out2, n2 = equation_polisher._exp_rate_parameterize(
        expr2, params, inits, PolishConfig()
    )
    assert n2 == 1
    assert out2 != expr2
