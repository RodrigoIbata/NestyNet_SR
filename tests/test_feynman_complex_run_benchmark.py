# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")

REPO_ROOT = Path(__file__).resolve().parent.parent
FEYNMAN_COMPLEX_DIR = REPO_ROOT / "examples" / "feynman_complex"


def _load_complex_run_benchmark():
    prev_problem_defs = sys.modules.get("problem_defs")
    problem_defs_path = FEYNMAN_COMPLEX_DIR / "problem_defs.py"
    run_benchmark_path = FEYNMAN_COMPLEX_DIR / "run_benchmark.py"

    pd_spec = importlib.util.spec_from_file_location("problem_defs", problem_defs_path)
    assert pd_spec is not None and pd_spec.loader is not None
    pd_mod = importlib.util.module_from_spec(pd_spec)

    try:
        sys.modules["problem_defs"] = pd_mod
        pd_spec.loader.exec_module(pd_mod)

        rb_spec = importlib.util.spec_from_file_location(
            "feynman_complex_run_benchmark_testmod",
            run_benchmark_path,
        )
        assert rb_spec is not None and rb_spec.loader is not None
        rb_mod = importlib.util.module_from_spec(rb_spec)
        rb_spec.loader.exec_module(rb_mod)
        return rb_mod
    finally:
        if prev_problem_defs is not None:
            sys.modules["problem_defs"] = prev_problem_defs
        else:
            sys.modules.pop("problem_defs", None)


rb = _load_complex_run_benchmark()


def test_cli_dispatches_sparse_engine_and_writes_sparse_summary(monkeypatch, tmp_path):
    problem = rb.ComplexProblemDef(
        id="C999",
        type="complex_ode",
        order=1,
        axes="t",
        x_axis=0,
        fields="A",
        equation="test",
        description="synthetic dispatch case",
        ref="-",
        complex_ops="-",
        params=[],
        param_ranges=[],
    )
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(rb, "load_complex_problems", lambda _path: {"C999": problem})
    monkeypatch.setattr(
        rb,
        "run_problem_sparse",
        lambda _problem, **kwargs: calls.append(("sparse", kwargs)) or {
            "id": "C999",
            "description": problem.description,
            "engine": "sparse",
            "status": "PASS",
            "message": "ok",
            "discovered_equations": "du/dt = 0",
        },
    )
    monkeypatch.setattr(
        rb,
        "run_problem_factorized_search",
        lambda _problem, **kwargs: calls.append(("factorized_search", kwargs)) or {
            "id": "C999",
            "description": problem.description,
            "engine": "factorized_search",
            "status": "PASS",
            "message": "ok",
        },
    )

    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    rc = rb.main([
        "--only",
        "C999",
        "--engine",
        "stlsq",
        "--results_dir",
        str(results_dir),
        "--data_dir",
        str(data_dir),
    ])

    assert rc == 0
    assert calls and calls[0][0] == "sparse"
    assert calls[0][1]["skip_generate"] is False
    assert calls[0][1]["sparse_library"] == "class"
    payload = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["engine"] == "sparse"
    assert payload["sparse_library"] == "class"
    assert not (results_dir / "summary_factorized_search.json").exists()


def test_cli_dispatches_factorized_engine_and_writes_factorized_summary(monkeypatch, tmp_path):
    problem = rb.ComplexProblemDef(
        id="C999",
        type="complex_ode",
        order=1,
        axes="t",
        x_axis=0,
        fields="A",
        equation="test",
        description="synthetic dispatch case",
        ref="-",
        complex_ops="-",
        params=[],
        param_ranges=[],
    )
    calls: list[str] = []

    monkeypatch.setattr(rb, "load_complex_problems", lambda _path: {"C999": problem})
    monkeypatch.setattr(
        rb,
        "run_problem_sparse",
        lambda _problem, **_kwargs: calls.append("sparse") or {
            "id": "C999",
            "description": problem.description,
            "engine": "sparse",
            "status": "PASS",
            "message": "ok",
        },
    )
    monkeypatch.setattr(
        rb,
        "run_problem_factorized_search",
        lambda _problem, **_kwargs: calls.append("factorized_search") or {
            "id": "C999",
            "description": problem.description,
            "engine": "factorized_search",
            "status": "PASS",
            "message": "ok",
            "discovered": [],
        },
    )

    results_dir = tmp_path / "results"
    rc = rb.main([
        "--only",
        "C999",
        "--engine",
        "fss",
        "--results_dir",
        str(results_dir),
        "--data_dir",
        str(tmp_path / "data"),
    ])

    assert rc == 0
    assert calls == ["factorized_search"]
    payload = json.loads((results_dir / "summary_factorized_search.json").read_text(encoding="utf-8"))
    assert payload["engine"] == "factorized_search"
    assert not (results_dir / "summary.json").exists()


def test_class_sparse_library_is_metadata_driven_for_temporal_pde():
    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C000"]
    terms = [repr(term) for term in rb.build_class_sparse_library(problem, coord_mins=np.array([0.0, 0.0]))]

    assert "u" in terms
    assert "u1" in terms
    assert "u_x1" in terms
    assert "u1_x1" in terms
    assert "u_x1x1" in terms
    assert "u1_x1x1" in terms
    assert "(((u ** 2) + (u1 ** 2)) * u)" in terms
    assert "(((u ** 2) + (u1 ** 2)) * u1)" in terms
    assert all("_x0x0" not in term for term in terms)
    # the class library enumerates fields, derivatives, and modulus terms
    assert len(terms) >= 8


def test_sparse_class_variant_recovers_analytic_c000():
    class _AnalyticSurrogate(torch.nn.Module):
        def forward(self, x):
            theta = 2.0 * x[:, 1] - 2.0 * x[:, 0]
            return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

        def grad(self, x):
            out = self.forward(x)
            u = out[:, 0]
            v = out[:, 1]
            g = torch.zeros((x.shape[0], 2, 2), dtype=x.dtype, device=x.device)
            g[:, 0, 0] = 2.0 * v
            g[:, 0, 1] = -2.0 * v
            g[:, 1, 0] = -2.0 * u
            g[:, 1, 1] = 2.0 * u
            return g

        def grad_grad(self, x):
            out = self.forward(x)
            u = out[:, 0]
            v = out[:, 1]
            h = torch.zeros((x.shape[0], 2, 2, 2), dtype=x.dtype, device=x.device)
            h[:, 0, 0, 0] = -4.0 * u
            h[:, 0, 0, 1] = 4.0 * u
            h[:, 0, 1, 0] = 4.0 * u
            h[:, 0, 1, 1] = -4.0 * u
            h[:, 1, 0, 0] = -4.0 * v
            h[:, 1, 0, 1] = 4.0 * v
            h[:, 1, 1, 0] = 4.0 * v
            h[:, 1, 1, 1] = -4.0 * v
            return h

    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C000"]
    t = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    x = torch.linspace(0.0, 2.0 * np.pi, 32, dtype=torch.float64)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    X = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)
    result, variant, n_terms, _lam = rb.discover_sparse_class_system(
        _AnalyticSurrogate(),
        X,
        problem,
    )

    discovered = rb._build_discovered_map_system(result)
    assert variant == "spatial_second"
    assert n_terms == 2
    assert discovered[(0, "u1_x1x1")] == pytest.approx(0.5, abs=1e-8)
    assert discovered[(1, "u_x1x1")] == pytest.approx(-0.5, abs=1e-8)


def test_sparse_class_selector_keeps_simpler_adequate_variant(monkeypatch):
    class _FakeCombined(torch.nn.Module):
        def grad(self, x):
            return torch.ones((x.shape[0], 2, 2), dtype=x.dtype, device=x.device)

    def _fake_discover(_combined, _X, _problem, *, stlsq_lambda, library_terms):
        names = {repr(term) for term in library_terms}
        if names == {"u_x1x1", "u1_x1x1"}:
            rms = [4.0e-3, 4.0e-3]
        elif {"u", "u1"}.issubset(names):
            rms = [5.0e-4, 5.0e-4]
        else:
            rms = [5.0e-2, 5.0e-2]
        return SimpleNamespace(
            rms_train=rms,
            coeffs=torch.ones((2, len(library_terms)), dtype=torch.float64),
            lambda_used=float(stlsq_lambda),
        )

    monkeypatch.setattr(rb, "discover_sparse_coupled_system", _fake_discover)

    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C000"]
    X = torch.zeros((8, 2), dtype=torch.float64)
    result, variant, n_terms, _lam = rb.discover_sparse_class_system(_FakeCombined(), X, problem)

    assert variant == "spatial_second"
    assert n_terms == 2
    assert result.rms_train == [4.0e-3, 4.0e-3]


def test_sparse_class_selector_prefers_supported_nonlinear_family(monkeypatch):
    class _FakeCombined(torch.nn.Module):
        def grad(self, x):
            return torch.ones((x.shape[0], 4, 2), dtype=x.dtype, device=x.device)

    def _fake_discover(_combined, _X, _problem, *, stlsq_lambda, library_terms):
        names = {repr(term) for term in library_terms}
        has_modulus = any("** 2" in name and "*" in name for name in names)
        rms = [2.0e-3] * 4 if has_modulus else [1.0e-4] * 4
        return SimpleNamespace(
            rms_train=rms,
            coeffs=torch.ones((4, len(library_terms)), dtype=torch.float64),
            lambda_used=float(stlsq_lambda),
        )

    monkeypatch.setattr(rb, "discover_sparse_coupled_system", _fake_discover)

    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C102"]
    X = torch.zeros((8, 2), dtype=torch.float64)
    result, variant, _n_terms, _lam = rb.discover_sparse_class_system(_FakeCombined(), X, problem)

    assert variant == "multifield_modulus"
    assert result.rms_train == [2.0e-3] * 4


def test_class_sparse_algebraic_library_avoids_identity_terms():
    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C204"]
    terms = [repr(term) for term in rb.build_class_sparse_library(problem, coord_mins=np.array([1.0]))]
    variants = rb.build_class_sparse_library_variants(problem, coord_mins=np.array([1.0]))
    variant_terms = [repr(term) for _name, nodes in variants for term in nodes]

    assert "u" not in terms
    assert "u1" not in terms
    assert "u" not in variant_terms
    assert "u1" not in variant_terms
    assert "(x0 * u)" in terms
    assert "((x0 ** -1) * u1)" in terms


def test_class_sparse_nonlinear_pde_variants_use_dimension_metadata():
    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C003"]
    variants = rb.build_class_sparse_library_variants(problem, coord_mins=np.array([0.0, 0.0]))

    assert rb._problem_supports_modulus_nonlinearity(problem)
    assert [name for name, _terms in variants[:2]] == ["nonlinear_modulus", "reaction_diffusion_modulus"]


def test_ode_class_variant_dimension_filter_rejects_c104_decoys():
    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C104"]
    variants = dict(rb.build_class_sparse_library_variants(problem, coord_mins=np.array([0.0])))

    assert rb._class_variant_dimensionally_supported(problem, variants["polynomial_coupling"])
    assert not rb._class_variant_dimensionally_supported(problem, variants["linear_system"])
    assert not rb._class_variant_dimensionally_supported(problem, variants["field_sine"])


def test_ode_class_variant_dimension_filter_allows_common_ode_classes():
    problems = rb.load_complex_problems(rb.BENCHMARK_FILE)

    c200_variants = dict(rb.build_class_sparse_library_variants(problems["C200"], coord_mins=np.array([0.0])))
    assert rb._class_variant_dimensionally_supported(problems["C200"], c200_variants["damped_second_order"])

    c105_variants = dict(rb.build_class_sparse_library_variants(problems["C105"], coord_mins=np.array([0.0])))
    assert rb._class_variant_dimensionally_supported(problems["C105"], c105_variants["field_trig_forcing"])


def test_class_variant_dimension_filter_prefers_radial_singular_terms():
    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C007"]
    variants = dict(rb.build_class_sparse_library_variants(problem, coord_mins=np.array([1.0])))

    assert not rb._class_variant_dimensionally_supported(problem, variants["second_order_linear"])
    assert not rb._class_variant_dimensionally_supported(problem, variants["damped_second_order"])
    assert rb._class_variant_dimensionally_supported(problem, variants["radial_singular"])


def test_class_sparse_first_order_pde_uses_first_spatial_derivatives():
    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C008"]
    variants = rb.build_class_sparse_library_variants(problem, coord_mins=np.array([0.0, 0.0]))
    names = [name for name, _terms in variants]
    terms = {name: [repr(term) for term in nodes] for name, nodes in variants}

    assert names[0] == "spatial_first_system"
    assert "multifield_nonlinear" not in names
    assert "spatial_second" not in names
    assert "u_x1" in terms["spatial_first_system"]
    assert all("x1x1" not in term for nodes in terms.values() for term in nodes)


def test_sparse_ode_surrogate_uses_multiple_trajectories_without_rhs_derivatives(monkeypatch):
    problem = rb.load_complex_problems(rb.BENCHMARK_FILE)["C200"]

    def _fail_exact_derivatives(*_args, **_kwargs):
        raise AssertionError("sparse ODE class discovery must use surrogate derivatives")

    class _FakeC200Combined:
        def __init__(self, phase: float):
            self.phase = float(phase)

        def _state(self, X):
            t = X[:, 0]
            z0 = t + self.phase
            z1 = 1.3 * t - 0.5 * self.phase
            u0 = torch.sin(z0) + 0.2 * torch.cos(2.0 * z0)
            u1 = torch.cos(z1) + 0.3 * torch.sin(2.0 * z1)
            du0 = torch.cos(z0) - 0.4 * torch.sin(2.0 * z0)
            du1 = -1.3 * torch.sin(z1) + 0.78 * torch.cos(2.0 * z1)
            h0 = -4.0 * u0 - 0.6 * du0
            h1 = -4.0 * u1 - 0.6 * du1
            return u0, u1, du0, du1, h0, h1

        def forward(self, X):
            u0, u1, *_rest = self._state(X)
            return torch.stack([u0, u1], dim=1)

        def grad(self, X):
            _u0, _u1, du0, du1, _h0, _h1 = self._state(X)
            return torch.stack([du0, du1], dim=1).unsqueeze(-1)

        def grad_grad(self, X):
            _u0, _u1, _du0, _du1, h0, h1 = self._state(X)
            return torch.stack([h0, h1], dim=1).reshape(X.shape[0], 2, 1, 1)

    monkeypatch.setattr(rb, "_compute_ode_derivatives", _fail_exact_derivatives)

    xs = [
        torch.linspace(0.05, 1.6, 80, dtype=torch.float64).reshape(-1, 1),
        torch.linspace(0.10, 1.8, 85, dtype=torch.float64).reshape(-1, 1),
        torch.linspace(0.15, 2.0, 90, dtype=torch.float64).reshape(-1, 1),
    ]
    surrogates = [_FakeC200Combined(0.0), _FakeC200Combined(0.7), _FakeC200Combined(1.4)]

    result, variant, n_terms, _lam = rb.discover_sparse_ode_surrogate_system(
        problem,
        surrogates,
        xs,
        stlsq_lambda=1.0e-8,
    )

    assert variant == "damped_second_order"
    assert n_terms == 4
    assert result.order == 2
    discovered = rb._build_discovered_map_system(result)
    assert discovered[(0, "u")] == pytest.approx(4.0, abs=1e-8)
    assert discovered[(0, "u_x0")] == pytest.approx(0.6, abs=1e-8)
    assert discovered[(1, "u1")] == pytest.approx(4.0, abs=1e-8)
    assert discovered[(1, "u1_x0")] == pytest.approx(0.6, abs=1e-8)


def test_build_ode_system_rhs_uses_shared_feature_predictors(monkeypatch):
    compiled: list[str] = []

    def _fake_compile(candidate):
        compiled.append(str(candidate["name"]))
        value = float(candidate["value"])
        return lambda feats: value

    monkeypatch.setattr(rb, "factorized_search_candidate_to_feature_predictor", _fake_compile)

    rhs = rb._build_ode_system_rhs(
        [
            {"name": "eq0", "value": 1.5, "expr": ("const", 0.0), "mapping": {}},
            {"name": "eq1", "value": -2.5, "expr": ("const", 0.0), "mapping": {}},
        ],
        pid="C200",
        params=rb.DEFAULT_PARAMS["C200"],
        ncomp=2,
        order=1,
        anchor_order=1,
    )

    out = rhs(0.0, [10.0, 20.0])
    assert compiled == ["eq0", "eq1"]
    assert out == pytest.approx([1.5, -2.5])


def test_validate_pde_residual_uses_shared_batch_candidate_eval(monkeypatch):
    calls: list[int] = []

    def _fake_eval(candidate, features, *, dtype):
        calls.append(int(len(features)))
        full = np.asarray(candidate["yhat"], dtype=np.float64)
        row_ids = np.asarray(features[:, 0], dtype=np.int64)
        return full[row_ids]

    monkeypatch.setattr(rb, "evaluate_factorized_search_candidate", _fake_eval)

    discovered = [
        {
            "expr": ("var", 0),
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
            "yhat": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        },
        {
            "expr": ("var", 0),
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
            "yhat": np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64),
        },
    ]
    features = np.zeros((4, 3), dtype=np.float64)
    features[:, 0] = np.arange(4, dtype=np.float64)
    targets = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64),
    ]

    status, message, scores = rb.validate_pde_residual(
        discovered,
        features,
        targets,
        ncomp=2,
        pass_nrmse=1.0e-9,
        partial_nrmse=1.0e-6,
        n_test=4,
        seed=0,
    )

    assert status == "PASS"
    assert "Residual NRMSE" in message
    assert [row["eq"] for row in scores] == [0, 1]
    assert calls == [4, 4]


def test_build_ode_feature_table_gates_illegal_trig_and_returns_dims():
    t = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    state = np.vstack(
        [
            np.linspace(1.0, 1.4, 5, dtype=np.float64),
            np.linspace(-0.2, 0.2, 5, dtype=np.float64),
        ]
    )
    trajs = [{"t": t, "state": state, "y0": [float(state[0, 0]), float(state[1, 0])]}]

    features, targets, names, feature_dims, target_dims = rb.build_ode_feature_table(
        "C105",
        rb.DEFAULT_PARAMS["C105"],
        trajs,
        ncomp=2,
        order=1,
        anchor_order=1,
    )

    dims = rb.get_complex_problem_dims("C105")
    assert feature_dims is not None
    assert target_dims is not None
    assert features.shape[1] == len(names) == len(feature_dims)
    assert len(targets) == len(target_dims) == 2
    assert "sin(t)" not in names
    assert "cos(t)" not in names
    assert "sin(Delta*x)" in names
    assert "cos(Delta*x)" in names
    assert all(not name.startswith("sin(u") for name in names)
    assert all(not name.startswith("cos(u") for name in names)
    assert target_dims == [
        rb._dim_sub(tuple(dims.component_dims[i]), tuple(dims.axis_dims[0]))
        for i in range(2)
    ]


def test_build_ode_feature_table_skips_complex_invariants_for_c302():
    t = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    state = np.vstack(
        [
            np.linspace(0.1, 0.3, 5, dtype=np.float64),
            np.linspace(-0.2, 0.0, 5, dtype=np.float64),
        ]
    )
    trajs = [{"t": t, "state": state, "y0": [float(state[0, 0]), float(state[1, 0])]}]

    features, _targets, names, feature_dims, _target_dims = rb.build_ode_feature_table(
        "C302",
        rb.DEFAULT_PARAMS["C302"],
        trajs,
        ncomp=2,
        order=1,
        anchor_order=1,
    )

    assert feature_dims is not None
    assert features.shape[1] == len(names) == len(feature_dims)
    assert all(not name.startswith("abs") for name in names)


def test_build_pde_feature_table_returns_target_dims_for_c204():
    pde_data = {
        "grid": np.array([[0.5], [1.0], [1.5]], dtype=np.float64),
        "components": [
            np.array([0.2, 0.15, 0.1], dtype=np.float64),
            np.array([0.0, -0.05, -0.1], dtype=np.float64),
        ],
        "shape": (3,),
    }

    features, targets, names, feature_dims, target_dims = rb.build_pde_feature_table(
        "C204",
        rb.DEFAULT_PARAMS["C204"],
        pde_data,
        ncomp=2,
        nxvars=1,
        anchor_order=0,
    )

    assert feature_dims is not None
    assert target_dims == [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
    assert features.shape[1] == len(names) == len(feature_dims)
    assert np.allclose(targets[0], 0.0)
    assert np.allclose(targets[1], 0.0)
    assert feature_dims[names.index("omega")] == (-1.0, 0.0, 0.0)
    assert feature_dims[names.index("V")] == (0.0, 0.0, 1.0)


def test_try_linear_prefit_rejects_nonmatching_feature_dims():
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    res = rb._try_linear_prefit(
        x,
        y,
        x,
        y,
        var_dims=[(0.0,), (0.0,)],
        y_dims=(1.0,),
    )

    assert res is None


def test_run_discovery_single_eq_forwards_units_to_explorer(monkeypatch):
    calls: list[dict[str, object]] = []

    class _Rec:
        best_mse = 1.0
        best_expr = ("var", 0)
        mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}

    class _Arch:
        def best(self, _k):
            return [_Rec()]

    def _fake_run_explorer_core(**kwargs):
        calls.append(kwargs)
        return _Arch()

    monkeypatch.setattr(rb, "_try_linear_prefit", lambda *args, **kwargs: None)
    monkeypatch.setattr(rb, "run_explorer_core", _fake_run_explorer_core)

    x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    res = rb.run_discovery_single_eq(
        x,
        y,
        x,
        y,
        n_iter=1,
        max_depth=1,
        n_seeds=1,
        fast=True,
        var_dims=[(0.0,), (1.0,)],
        y_dims=(1.0,),
    )

    assert res["expr_str"] == "x0"
    assert calls
    assert calls[0]["var_dims"] == [(0.0,), (1.0,)]
    assert calls[0]["y_dims"] == (1.0,)
