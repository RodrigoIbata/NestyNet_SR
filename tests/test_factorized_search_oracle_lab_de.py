# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("sympy")
pytest.importorskip("scipy")

import nestynet_sr.sr_search.factorized_search.oracle_lab_de as oracle_lab_de_mod
from nestynet_sr.sr_search.factorized_search.oracle_lab_de import (
    DEFeatureTensors,
    DELabSpec,
    SurrogateDerivProvider,
    derivative_provider,
    equation_de_spec_from_dict,
    run_oracle_de_from_features,
    run_oracle_de_equation,
)
from nestynet_sr.sr_search.config import FactorizedSearchConfig


def _write_csv(path: Path, x: np.ndarray, u: np.ndarray, du: np.ndarray, d2u: np.ndarray) -> None:
    arr = np.column_stack([u, x, du, d2u])
    np.savetxt(str(path), arr, delimiter=",", header="y,x0,du,d2u", comments="")


def _payload(csv_path: Path) -> dict:
    return {
        "id": "decay_precomputed",
        "csv_paths": [str(csv_path)],
        "order_candidates": [1, 2],
        "x_axis": 0,
        "include_x": True,
        "include_u": True,
        "include_du": True,
        "x_col": "x0",
        "u_col": "y",
        "deriv": {
            "method": "precomputed",
            "du_col": "du",
            "d2u_col": "d2u",
        },
        "dims": {
            "basis": ["U", "T"],
            "x": [0, 1],
            "u": [1, 0],
        },
        "validate_integrate_topk": 1,
    }


def test_default_oracle_de_hyperparams_enable_structural_pade():
    hp = oracle_lab_de_mod.default_oracle_de_hyperparams()

    assert hp.score_head_enable is False
    assert hp.de_score_head_policy == "proposal_only"
    assert hp.de_accept_hidden_score_head is False
    assert hp.de_score_head_untyped_enable is False
    assert hp.score_pade_structural_enable is True
    assert hp.score_pade_structural_max_degree == 2
    assert hp.score_pade_structural_max_total_degree == 3


def test_derivative_provider_spline_and_precomputed_consistent():
    x = np.linspace(0.0, 1.0, 200)
    u = np.sin(x)

    x_s, u_s, du_s, d2u_s = derivative_provider(x, u, method="spline", spline_s=0.0, spline_k=3)
    assert np.allclose(x_s, x)
    assert np.allclose(u_s, u)
    assert np.isfinite(du_s).all()
    assert np.isfinite(d2u_s).all()

    du = np.cos(x)
    d2u = -np.sin(x)
    x_p, u_p, du_p, d2u_p = derivative_provider(x, u, method="precomputed", du_pre=du, d2u_pre=d2u)
    assert np.allclose(x_p, x)
    assert np.allclose(u_p, u)
    assert np.allclose(du_p, du)
    assert np.allclose(d2u_p, d2u)


def test_spec_accepts_filepaths_alias_and_phase2_fields(tmp_path: Path):
    csv_path = tmp_path / "traj.csv"
    x = np.linspace(0.0, 1.0, 16)
    u = np.sin(x)
    du = np.cos(x)
    d2u = -np.sin(x)
    _write_csv(csv_path, x, u, du, d2u)

    payload = {
        "id": "alias_case",
        "filepaths": [str(csv_path)],
        "order_candidates": [1],
        "x_axis": 0,
        "include_x": True,
        "include_u": True,
        "include_du": True,
        "x_col": "x0",
        "u_col": "y",
        "out_idx": 0,
        "y_transform": "identity",
        "deriv": {
            "method": "precomputed",
            "du_col": "du",
            "d2u_col": "d2u",
        },
    }
    spec = equation_de_spec_from_dict(payload, source="alias-test")
    assert spec.csv_paths == (str(csv_path),)
    assert spec.filepaths == (str(csv_path),)
    assert int(spec.out_idx) == 0
    assert str(spec.y_transform) == "identity"


def test_spec_accepts_multi_trajectory_manifest(tmp_path: Path):
    x = np.linspace(0.0, 1.0, 20)
    u = np.sin(x)
    du = np.cos(x)
    d2u = -np.sin(x)
    p0 = tmp_path / "ic0.csv"
    p1 = tmp_path / "ic1.csv"
    _write_csv(p0, x, u, du, d2u)
    _write_csv(p1, x, u + 0.2, du, d2u)

    payload = {
        "id": "multi_case",
        "trajectories": [
            {"id": "ic0", "csv": str(p0)},
            {"id": "ic1", "csv": str(p1)},
        ],
        "order_candidates": [1, 2],
        "split_mode": "traj_holdout",
        "traj_metric": "max",
        "deriv": {
            "method": "precomputed",
            "du_col": "du",
            "d2u_col": "d2u",
        },
    }
    spec = equation_de_spec_from_dict(payload, source="multi-manifest")
    assert spec.csv_paths == (str(p0), str(p1))
    assert [t.id for t in spec.trajectories] == ["ic0", "ic1"]
    assert str(spec.split_mode) == "traj_holdout"
    assert str(spec.traj_metric) == "max"


class _SquareSurrogate(torch.nn.Module):
    """Produces t(x)=(x+2)^2; with y_transform='square', inverse is u=x+2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, :1]
        return (x0 + 2.0) ** 2

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        b, nx = int(x.shape[0]), int(x.shape[1])
        g = torch.zeros((b, nx), dtype=x.dtype, device=x.device)
        g[:, 0] = 2.0 * (x[:, 0] + 2.0)
        return g

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        b, nx = int(x.shape[0]), int(x.shape[1])
        h = torch.zeros((b, nx, nx), dtype=x.dtype, device=x.device)
        h[:, 0, 0] = 2.0
        return h


def test_surrogate_deriv_provider_inverse_chain_rule():
    x = torch.linspace(-1.5, 1.5, 80, dtype=torch.float64).reshape(-1, 1)
    y0 = torch.zeros_like(x)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y0),
        batch_size=16,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y0),
        batch_size=20,
        shuffle=False,
    )

    spec = DELabSpec(
        id="chain_rule",
        csv_paths=("dummy.csv",),
        order_candidates=(1, 2),
        x_axis=0,
        out_idx=0,
        y_transform="square",
    )
    provider = SurrogateDerivProvider(
        _SquareSurrogate(),
        y_transform="square",
        out_idx=0,
        eval_batch=32,
    )
    feats = provider.build_features_from_loaders(
        train_loader,
        val_loader,
        spec=spec,
        seed=3,
        dtype=torch.float64,
        device=torch.device("cpu"),
        n_fit=50,
        n_probe=40,
        max_batches=8,
        max_points_factor=2,
    )

    assert feats.x_fit.shape == (50, 1)
    assert feats.x_probe.shape == (40, 1)
    assert torch.allclose(feats.u_fit, feats.x_fit[:, :1] + 2.0, atol=1e-10)
    assert torch.allclose(feats.du_fit, torch.ones_like(feats.du_fit), atol=1e-10)
    assert torch.allclose(feats.d2u_fit, torch.zeros_like(feats.d2u_fit), atol=1e-10)


def test_run_oracle_de_equation_per_order_and_fairness(tmp_path: Path):
    k = 0.7
    x = np.linspace(0.01, 2.0, 160, dtype=np.float64)
    u = np.exp(-k * x)
    du = -k * u
    d2u = (k**2) * u

    csv_path = tmp_path / "traj.csv"
    _write_csv(csv_path, x, u, du, d2u)

    spec = equation_de_spec_from_dict(_payload(csv_path), source="unit-test")

    hp = FactorizedSearchConfig()
    hp.n_iter = 120
    hp.max_depth = 3
    hp.poly_degree = 3
    hp.return_topk = 3
    hp.n_fit = 96
    hp.n_probe = 120
    hp.n_seeds = 1
    hp.split_iter_across_seeds = True
    hp.brute_depth = 1
    hp.brute_max_expressions = 250
    hp.refine_enable = False

    report = run_oracle_de_equation(spec, factorized_search_hp=hp, seed=7, dtype=torch.float64, verbose=False)

    assert report["best"] is not None
    assert report["resolved_config"]["includes_disabled_lanes"] is True
    assert report["resolved_config"]["resolved"]["refine_enable"] is False
    assert len(report["per_order"]) == 2

    by_order = {int(row["order"]): row for row in report["per_order"]}
    assert set(by_order.keys()) == {1, 2}

    o1 = by_order[1]
    o2 = by_order[2]

    assert int(o1["nvars"]) == 2  # x, u  (du intentionally excluded for fairness)
    assert int(o2["nvars"]) == 3  # x, u, du
    assert "du" not in [str(n).lower() for n in o1["feature_names"]]

    assert o1["best"] is not None
    assert o2["best"] is not None
    assert math.isfinite(float(o1["best"]["mse"]))
    assert math.isfinite(float(o2["best"]["mse"]))

    # Remapped ASTs should be produced for at least the best row.
    assert o1["best"].get("rhs_mapped_ast", None) is not None
    assert o1["best"].get("residual_ast", None) is not None

    # Integrate-back validation keys should be present for top-k=1.
    assert "integrate_mse" in o1["results"][0]
    assert "integrate_ok" in o1["results"][0]


def test_build_sparse_combo_rows_recovers_two_term_linear_target():
    hp = oracle_lab_de_mod.default_oracle_de_hyperparams()
    hp.poly_degree = 1
    hp.max_depth = 4
    hp.return_topk = 8
    hp.score_mapping_family_mode = "poly_only"
    hp.de_sparse_combo_enable = True
    hp.de_sparse_combo_pool_topk = 4
    hp.de_sparse_combo_max_terms = 2

    spec = DELabSpec(
        id="combo_unit",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=False,
    )

    x_fit = torch.tensor(
        [[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [3.0, 2.0]],
        dtype=torch.float64,
    )
    y_fit = (2.0 * x_fit[:, :1]) - (0.5 * x_fit[:, 1:2])
    x_probe = torch.tensor(
        [[0.5, 0.0], [1.5, 1.0], [2.5, -0.5]],
        dtype=torch.float64,
    )
    y_probe = (2.0 * x_probe[:, :1]) - (0.5 * x_probe[:, 1:2])

    rows = oracle_lab_de_mod._build_sparse_combo_rows(
        spec=spec,
        order=1,
        base_rows=[
            {"expr": "x0", "score": 1.0, "_expr_obj": ("var", 0)},
            {"expr": "u", "score": 1.1, "_expr_obj": ("var", 1)},
        ],
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        probe_meta=None,
        traj_metric="mean",
    )

    assert rows
    best = min(rows, key=lambda r: float(r["score"]))
    assert best["construction"] == "basis_state_combo"
    assert set(best["combo_source_exprs"]) == {"x0", "u"}
    assert float(best["mse"]) < 1.0e-10
    assert best["combo_mapping_mode"] == "affine_only"
    assert len(best["mapping"]["coeffs"]) <= 2


def test_build_sparse_combo_rows_reports_periodic_combo_diagnostics():
    hp = oracle_lab_de_mod.default_oracle_de_hyperparams()
    hp.poly_degree = 1
    hp.max_depth = 5
    hp.return_topk = 8
    hp.score_mapping_family_mode = "poly_only"
    hp.de_sparse_combo_enable = True
    hp.de_sparse_combo_pool_topk = 4
    hp.de_sparse_combo_max_terms = 2

    spec = DELabSpec(
        id="periodic_combo_unit",
        csv_paths=(),
        order_candidates=(2,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=False,
    )

    x = torch.linspace(0.0, 6.0, 48, dtype=torch.float64)
    u = torch.linspace(-1.0, 1.0, 48, dtype=torch.float64)
    z = torch.stack([x, u], dim=1)
    x_fit = z[::2]
    x_probe = z[1::2]
    omega = 1.75
    y_fit = (2.0 * x_fit[:, 1:2]) + (1.25 * torch.cos(omega * x_fit[:, :1]))
    y_probe = (2.0 * x_probe[:, 1:2]) + (1.25 * torch.cos(omega * x_probe[:, :1]))
    diag: dict[str, object] = {}

    rows = oracle_lab_de_mod._build_sparse_combo_rows(
        spec=spec,
        order=2,
        base_rows=[
            {"expr": "u", "score": 1.0, "_expr_obj": ("var", 1)},
            {
                "expr": "cos((1.75*x0))",
                "score": 1.1,
                "_expr_obj": ("cos", ("mul", ("const", omega), ("var", 0))),
            },
        ],
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        probe_meta=None,
        traj_metric="mean",
        diagnostics_out=diag,
    )

    assert rows
    best = min(rows, key=lambda r: float(r["score"]))
    assert best["combo_contains_periodic"] is True
    assert any(abs(float(m["omega"]) - omega) < 1.0e-12 for m in best["combo_periodic_matches"])
    assert int(diag["periodic_base_rows_count"]) == 1
    assert int(diag["periodic_carrier_rows_count"]) == 1
    assert int(diag["periodic_terms_count"]) == 1
    assert int(diag["periodic_combo_rows_count"]) >= 1
    assert any(
        abs(float(match["omega"]) - omega) < 1.0e-12
        for row in diag["periodic_combo_rows"]
        for match in row["periodic_matches"]
    )


def test_build_contextual_atom_rows_promotes_periodic_additive_context():
    hp = oracle_lab_de_mod.default_oracle_de_hyperparams()
    hp.poly_degree = 1
    hp.max_depth = 5
    hp.return_topk = 8
    hp.score_mapping_family_mode = "poly_only"
    hp.de_sparse_combo_enable = True
    hp.de_sparse_combo_pool_topk = 6
    hp.de_sparse_combo_max_terms = 3

    spec = DELabSpec(
        id="contextual_periodic_unit",
        csv_paths=(),
        order_candidates=(2,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=False,
    )

    x = torch.linspace(0.0, 10.0, 160, dtype=torch.float64)
    u = torch.sin(3.0 * x) + 0.2 * torch.cos(0.7 * x)
    z = torch.stack([x, u], dim=1)
    x_fit = z[::2]
    x_probe = z[1::2]
    omega = 1.75
    y_fit = (-9.0 * x_fit[:, 1:2]) + (1.25 * torch.cos(omega * x_fit[:, :1]))
    y_probe = (-9.0 * x_probe[:, 1:2]) + (1.25 * torch.cos(omega * x_probe[:, :1]))
    diag: dict[str, object] = {}

    rows = oracle_lab_de_mod._build_contextual_atom_rows(
        spec=spec,
        order=2,
        base_rows=[
            {"expr": "u", "score": 1.0, "_expr_obj": ("var", 1)},
            {
                "expr": "cos((1.75*x0))",
                "score": 1.1,
                "_expr_obj": ("cos", ("mul", ("const", omega), ("var", 0))),
            },
        ],
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        fit_meta=None,
        probe_meta=None,
        traj_metric="mean",
        diagnostics_out=diag,
    )

    assert rows
    best = min(rows, key=lambda r: float(r["score"]))
    assert best["construction"] == "contextual_atom_promotion"
    assert best["acceptance_basis"] == "contextual_atom_delta"
    assert "x1" in set(best["contextual_base"])
    assert any("cos" in expr for expr in best["contextual_source_exprs"])
    assert float(best["mse"]) < 1.0e-10
    assert float(best["contextual_delta_vs_base"]) > 0.1
    assert int(diag["promoted_rows"]) >= 1
    assert any(row.get("promoted") for row in diag["trace"])


def test_periodic_seed_rows_preserve_pooled_hints_with_window_metadata(monkeypatch):
    from nestynet_sr.sr_search.factorized_search.engine import search as search_mod

    hp = oracle_lab_de_mod.default_oracle_de_hyperparams()
    hp.periodic_seed_enable = True
    hp.periodic_seed_max_hints = 2

    x = torch.linspace(0.0, 10.0, 128, dtype=torch.float64).reshape(-1, 1)
    y = torch.cos(1.75 * x)
    x_w = x[:64].clone()
    y_w = torch.cos(2.955 * x_w)

    def fake_periodogram(x_tbl, y_tbl, *, max_hints, min_prominence, **_kwargs):
        assert max_hints == 2
        assert min_prominence == hp.periodic_seed_min_prominence
        if x_tbl is x:
            return [(0, 1.75), (0, 3.0)]
        return [(0, 2.955), (0, 12.574)]

    monkeypatch.setattr(search_mod, "_periodogram_frequency_hints", fake_periodogram)

    rows = oracle_lab_de_mod._periodic_seed_atom_rows(
        x,
        y,
        hp,
        fit_meta=[("w0", x_w, y_w)],
    )
    exprs = {str(row["expr"]) for row in rows}

    assert "cos((1.75*x0))" in exprs
    assert "sin((1.75*x0))" in exprs
    assert "cos((3*x0))" in exprs
    assert any(row.get("periodic_hint_source") == "pooled" for row in rows)

    ctx_rows = oracle_lab_de_mod._contextual_periodic_seed_rows(
        base_terms=(),
        x_fit=x,
        y_fit=y,
        hp=hp,
        fit_meta=[("w0", x_w, y_w)],
    )
    ctx_exprs = {str(row["expr"]) for row in ctx_rows}
    assert "cos((1.75*x0))" in ctx_exprs
    assert "sin((1.75*x0))" in ctx_exprs
    assert any(row.get("periodic_hint_source") == "pooled" for row in ctx_rows)


def test_build_sparse_combo_rows_rejects_depth_truncated_compile():
    """Regression: _compile_linear_combo may drop terms to satisfy max_depth.

    Emitting such a row would pair a truncated expr_ast with the full-combo
    mapping/score (de123 pendulum benchmark failure). The composer must reject
    states whose compiled AST is not faithful to the scored linear span, and
    keep faithful ones when the depth budget allows the full combination.
    """
    from nestynet_sr.sr_search.factorized_search.expr_mapping import eval_mapping
    from nestynet_sr.sr_search.factorized_search.explorer import eval_node

    u = torch.linspace(-2.4, 2.4, 600, dtype=torch.float64)
    x_fit = torch.stack([u[::2], torch.cos(2.0 * u[::2])], dim=1)
    x_probe = torch.stack([u[1::2], torch.cos(2.0 * u[1::2])], dim=1)

    def target(x: torch.Tensor) -> torch.Tensor:
        return (-7.92 * torch.sin(x[:, :1])) + (1.5 * torch.cos(0.5434 * x[:, :1]) * x[:, :1])

    y_fit = target(x_fit)
    y_probe = target(x_probe)

    spec = DELabSpec(
        id="combo_truncation_unit",
        csv_paths=(),
        order_candidates=(2,),
        x_axis=0,
        include_x=False,
        include_u=True,
        include_du=True,
    )
    base_rows = [
        {"expr": "sin(x0)", "score": 0.05, "_expr_obj": ("sin", ("var", 0))},
        {
            "expr": "(cos((0.5434*x0))*x0)",
            "score": 0.06,
            "_expr_obj": ("mul", ("cos", ("mul", ("const", 0.5434), ("var", 0))), ("var", 0)),
        },
    ]

    def run(max_depth: int) -> tuple[list[dict], dict]:
        hp = oracle_lab_de_mod.default_oracle_de_hyperparams()
        hp.de_sparse_combo_enable = True
        hp.de_sparse_combo_pool_topk = 4
        hp.de_sparse_combo_max_terms = 2
        hp.max_depth = max_depth
        diag: dict = {}
        rows = oracle_lab_de_mod._build_sparse_combo_rows(
            spec=spec,
            order=2,
            base_rows=[dict(row) for row in base_rows],
            x_fit=x_fit,
            y_fit=y_fit,
            x_probe=x_probe,
            y_probe=y_probe,
            hp=hp,
            probe_meta=None,
            traj_metric="mean",
            diagnostics_out=diag,
        )
        return rows, diag

    # With a budget that fits one scaled term but not the add tree (the
    # composer grants max_depth+3 headroom), the compile truncates and the
    # faithfulness gate must reject the state.
    rows_shallow, diag_shallow = run(max_depth=2)
    assert rows_shallow == []
    assert int(diag_shallow.get("compile_unfaithful_rejected", 0)) >= 1

    # With enough depth budget the combo compiles faithfully: the serialized
    # (expr_ast, mapping) pair must reproduce the stored probe mse.
    rows_deep, diag_deep = run(max_depth=3)
    assert rows_deep
    assert int(diag_deep.get("compile_unfaithful_rejected", 0)) == 0
    best = min(rows_deep, key=lambda r: float(r["score"]))

    def to_tuple(node):
        if isinstance(node, list):
            return tuple(to_tuple(v) for v in node)
        return node

    pred = eval_node(to_tuple(best["expr_ast"]), x_probe)
    yhat = eval_mapping(pred, best["mapping"]).reshape(-1)
    reeval_mse = float(torch.mean((yhat - y_probe.reshape(-1)) ** 2))
    stored_mse = float(best["mse"])
    assert reeval_mse <= max(10.0 * stored_mse, 1.0e-10)


def test_build_sparse_combo_rows_forces_affine_mapping_even_when_global_poly_degree_is_high():
    hp = oracle_lab_de_mod.default_oracle_de_hyperparams()
    hp.poly_degree = 4
    hp.max_depth = 4
    hp.return_topk = 8
    hp.score_mapping_family_mode = "full"
    hp.de_sparse_combo_enable = True
    hp.de_sparse_combo_pool_topk = 4
    hp.de_sparse_combo_max_terms = 2

    spec = DELabSpec(
        id="combo_affine_guard",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=False,
    )
    x_fit = torch.tensor(
        [[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [3.0, 2.0]],
        dtype=torch.float64,
    )
    y_fit = (2.0 * x_fit[:, :1]) - (0.5 * x_fit[:, 1:2])
    x_probe = torch.tensor(
        [[0.5, 0.0], [1.5, 1.0], [2.5, -0.5]],
        dtype=torch.float64,
    )
    y_probe = (2.0 * x_probe[:, :1]) - (0.5 * x_probe[:, 1:2])
    diag: dict[str, object] = {}

    rows = oracle_lab_de_mod._build_sparse_combo_rows(
        spec=spec,
        order=1,
        base_rows=[
            {"expr": "x0", "score": 1.0, "_expr_obj": ("var", 0)},
            {"expr": "u", "score": 1.1, "_expr_obj": ("var", 1)},
        ],
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        probe_meta=None,
        traj_metric="mean",
        diagnostics_out=diag,
    )

    assert rows
    assert diag["effective_mapping_mode"] == "affine_only"
    assert diag["effective_poly_degree"] == 1
    assert all(len(row["mapping"]["coeffs"]) <= 2 for row in rows)


def test_build_sparse_combo_rows_functionally_dedupes_scaled_atoms():
    hp = oracle_lab_de_mod.default_oracle_de_hyperparams()
    hp.poly_degree = 1
    hp.max_depth = 5
    hp.return_topk = 8
    hp.de_sparse_combo_enable = True
    hp.de_sparse_combo_pool_topk = 4
    hp.de_sparse_combo_max_terms = 3
    hp.de_sparse_combo_corr_eps = 1.0e-8

    spec = DELabSpec(
        id="combo_dedup",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=False,
    )
    x_fit = torch.tensor(
        [[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [3.0, 2.0]],
        dtype=torch.float64,
    )
    y_fit = (2.0 * x_fit[:, :1]) - (0.5 * x_fit[:, 1:2])
    x_probe = torch.tensor(
        [[0.5, 0.0], [1.5, 1.0], [2.5, -0.5]],
        dtype=torch.float64,
    )
    y_probe = (2.0 * x_probe[:, :1]) - (0.5 * x_probe[:, 1:2])
    diag: dict[str, object] = {}

    rows = oracle_lab_de_mod._build_sparse_combo_rows(
        spec=spec,
        order=1,
        base_rows=[
            {"expr": "x0", "score": 1.0, "_expr_obj": ("var", 0)},
            {"expr": "u", "score": 1.1, "_expr_obj": ("var", 1)},
            {"expr": "2*u", "score": 1.2, "_expr_obj": ("mul", ("const", 2.0), ("var", 1))},
        ],
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        probe_meta=None,
        traj_metric="mean",
        diagnostics_out=diag,
    )

    assert rows
    assert diag["functional_duplicates"] >= 1
    assert all("2*u" not in set(row["combo_source_exprs"]) for row in rows)


class _FakeOrderArch:
    def __init__(self, records):
        self._records = list(records)
        self.refine_diagnostics = {}
        self.refine_slate_stats = {}

    def best(self, k, strategy="mse"):
        return list(self._records[: int(k)])


def _fake_order_rec(expr, *, score=1.0):
    return SimpleNamespace(
        best_mse=float(score),
        best_raw_mse=float(score),
        best_expr=expr,
        mapping={"kind": "identity"},
    )


def test_run_order_search_pre_mutation_combo_early_returns_without_full_mutation(monkeypatch):
    calls: list[int] = []

    def _fake_run_explorer_core(**kwargs):
        calls.append(int(kwargs["n_iter"]))
        if int(kwargs["n_iter"]) != 0:
            raise AssertionError("full mutation search should not run after clean pre-mutation combo")
        return _FakeOrderArch(
            [
                _fake_order_rec(("var", 0), score=1.0),
                _fake_order_rec(("var", 1), score=1.1),
            ]
        )

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", _fake_run_explorer_core)

    hp = oracle_lab_de_mod.default_oracle_de_hyperparams()
    hp.n_iter = 77
    hp.return_topk = 4
    hp.max_depth = 4
    hp.poly_degree = 1
    hp.score_mapping_family_mode = "poly_only"
    hp.de_sparse_combo_enable = True
    hp.de_sparse_combo_pre_mutation_enable = True
    hp.de_sparse_combo_pool_topk = 4
    hp.de_sparse_combo_max_terms = 2
    hp.early_stop_mse = 1.0e-8

    spec = DELabSpec(
        id="pre_combo_unit",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=False,
    )
    x_fit = torch.tensor(
        [[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [3.0, 2.0]],
        dtype=torch.float64,
    )
    y_fit = (2.0 * x_fit[:, :1]) - (0.5 * x_fit[:, 1:2])
    x_probe = torch.tensor(
        [[0.5, 0.0], [1.5, 1.0], [2.5, -0.5]],
        dtype=torch.float64,
    )
    y_probe = (2.0 * x_probe[:, :1]) - (0.5 * x_probe[:, 1:2])
    diagnostics: dict[str, object] = {}

    rows, _n_seeds, _n_seeds_ran, _n_iter_each = oracle_lab_de_mod._run_order_search(
        spec=spec,
        order=1,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        dtype=torch.float64,
        run_seed=0,
        var_dims=None,
        y_dims=None,
        verbose=False,
        diagnostics_out=diagnostics,
    )

    assert calls == [0]
    assert rows
    assert rows[0]["construction"] == "basis_state_combo"
    assert rows[0]["combo_phase"] == "pre_mutation"
    assert float(rows[0]["mse"]) < 1.0e-10
    additive_diag = diagnostics["additive_fss"]
    assert additive_diag["pre_mutation_attempted"] is True
    assert additive_diag["pre_mutation_early_return"] is True
    assert additive_diag["pre_mutation_combo_rows"] >= 1


def test_run_order_search_falls_through_when_pre_mutation_combo_absent(monkeypatch):
    calls: list[int] = []

    def _fake_run_explorer_core(**kwargs):
        n_iter = int(kwargs["n_iter"])
        calls.append(n_iter)
        if n_iter == 0:
            return _FakeOrderArch([_fake_order_rec(("var", 0), score=1.0)])
        return _FakeOrderArch(
            [
                _fake_order_rec(("var", 0), score=1.0),
                _fake_order_rec(("var", 1), score=1.1),
            ]
        )

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", _fake_run_explorer_core)

    hp = oracle_lab_de_mod.default_oracle_de_hyperparams()
    hp.n_iter = 77
    hp.return_topk = 4
    hp.max_depth = 4
    hp.poly_degree = 1
    hp.score_mapping_family_mode = "poly_only"
    hp.de_sparse_combo_enable = True
    hp.de_sparse_combo_pre_mutation_enable = True
    hp.de_sparse_combo_pool_topk = 4
    hp.de_sparse_combo_max_terms = 2
    hp.early_stop_mse = 1.0e-8

    spec = DELabSpec(
        id="pre_combo_fallback_unit",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=False,
    )
    x_fit = torch.tensor(
        [[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [3.0, 2.0]],
        dtype=torch.float64,
    )
    y_fit = (x_fit[:, :1] * x_fit[:, 1:2]) + (0.25 * x_fit[:, :1])
    x_probe = torch.tensor(
        [[0.5, 0.0], [1.5, 1.0], [2.5, -0.5]],
        dtype=torch.float64,
    )
    y_probe = (x_probe[:, :1] * x_probe[:, 1:2]) + (0.25 * x_probe[:, :1])
    diagnostics: dict[str, object] = {}

    rows, _n_seeds, _n_seeds_ran, _n_iter_each = oracle_lab_de_mod._run_order_search(
        spec=spec,
        order=1,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        dtype=torch.float64,
        run_seed=0,
        var_dims=None,
        y_dims=None,
        verbose=False,
        diagnostics_out=diagnostics,
    )

    assert calls == [0, 77]
    assert rows
    additive_diag = diagnostics["additive_fss"]
    assert additive_diag["pre_mutation_attempted"] is True
    assert additive_diag["pre_mutation_early_return"] is False
    assert additive_diag["post_mutation_combo_rows"] >= 1


def test_seed_sweep_early_stop(monkeypatch, tmp_path: Path):
    x = np.linspace(0.01, 1.0, 64, dtype=np.float64)
    u = np.exp(-0.5 * x)
    du = -0.5 * u
    d2u = 0.25 * u
    csv_path = tmp_path / "traj.csv"
    _write_csv(csv_path, x, u, du, d2u)

    payload = _payload(csv_path)
    payload["order_candidates"] = [1]
    payload["validate_integrate_topk"] = 0
    spec = equation_de_spec_from_dict(payload, source="seed-early-stop")

    calls = {"n": 0}

    class _FakeArch:
        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-12,
                best_expr=("var", 1),  # u
                mapping={"kind": "poly", "coeffs": [0.0, -0.5], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    def _fake_run_explorer_core(**kwargs):
        calls["n"] += 1
        return _FakeArch()

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", _fake_run_explorer_core)

    hp = FactorizedSearchConfig()
    hp.n_iter = 200
    hp.max_depth = 3
    hp.poly_degree = 3
    hp.return_topk = 1
    hp.n_fit = 32
    hp.n_probe = 48
    hp.n_seeds = 5
    hp.split_iter_across_seeds = True
    hp.brute_depth = 1
    hp.brute_max_expressions = 200
    hp.early_stop_mse = 1.0e-8
    hp.refine_enable = False

    report = run_oracle_de_equation(spec, factorized_search_hp=hp, seed=7, dtype=torch.float64, verbose=False)
    assert calls["n"] == 1
    assert int(report["per_order"][0]["n_seeds"]) == 5
    assert int(report["per_order"][0]["n_seeds_ran"]) == 1


def test_seed_sweep_does_not_early_stop_on_pooled_metric_only(monkeypatch, tmp_path: Path):
    x = np.linspace(0.01, 1.0, 64, dtype=np.float64)
    u = np.exp(-0.5 * x)
    du = -0.5 * u
    d2u = 0.25 * u
    csv_path = tmp_path / "traj.csv"
    _write_csv(csv_path, x, u, du, d2u)

    payload = _payload(csv_path)
    payload["order_candidates"] = [1]
    payload["validate_integrate_topk"] = 0
    payload["traj_metric"] = "max"
    spec = equation_de_spec_from_dict(payload, source="seed-no-pooled-early-stop")

    calls = {"n": 0}

    class _FakeArch:
        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-12,  # intentionally tiny pooled metric
                best_expr=("var", 0),  # x -> poor du prediction
                mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    def _fake_run_explorer_core(**kwargs):
        calls["n"] += 1
        return _FakeArch()

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", _fake_run_explorer_core)

    hp = FactorizedSearchConfig()
    hp.n_iter = 200
    hp.max_depth = 3
    hp.poly_degree = 3
    hp.return_topk = 1
    hp.n_fit = 32
    hp.n_probe = 48
    hp.n_seeds = 4
    hp.split_iter_across_seeds = True
    hp.brute_depth = 1
    hp.brute_max_expressions = 200
    hp.early_stop_mse = 1.0e-8
    hp.refine_enable = False

    report = run_oracle_de_equation(spec, factorized_search_hp=hp, seed=7, dtype=torch.float64, verbose=False)
    assert calls["n"] == 4
    assert int(report["per_order"][0]["n_seeds"]) == 4
    assert int(report["per_order"][0]["n_seeds_ran"]) == 4


def test_run_order_search_threads_mapping_and_prescreen_knobs(monkeypatch):
    captured: dict[str, object] = {}

    class _DummyArch:
        def best(self, k):
            return []

    def _fake_run_explorer_core(**kwargs):
        captured.update(kwargs)
        return _DummyArch()

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", _fake_run_explorer_core)

    hp = FactorizedSearchConfig()
    hp.n_iter = 8
    hp.n_seeds = 1
    hp.split_iter_across_seeds = True
    hp.score_mapping_family_mode = "poly_only"
    hp.brute_score_mapping_family_mode = "poly_only"
    hp.score_pade_structural_enable = True
    hp.score_pade_structural_max_degree = 1
    hp.score_pade_structural_max_total_degree = 2
    hp.score_pade_structural_max_depth = 7
    hp.score_pade_structural_max_size = 31
    hp.score_pade_structural_coeff_tol = 1.0e-9
    hp.score_pade_structural_mse_rel_tol = 1.0e-5
    hp.score_mapping_expensive_gate_best_factor = 7.0
    hp.score_mapping_expensive_rel_y = 0.03
    hp.score_prescreen_enable = True
    hp.score_prescreen_family_mode = "cheap"
    hp.score_prescreen_residual_family_mode = "poly_only"
    hp.score_prescreen_residual_allow_hint = True
    hp.score_prescreen_residual_use_global_best = True
    hp.score_prescreen_parent_best_factor = 1.25
    hp.score_prescreen_global_best_factor = 2.5
    hp.score_prescreen_residual_parent_best_factor = 1.05
    hp.score_prescreen_residual_global_best_factor = 1.2
    hp.score_domain_projection_enable = True
    hp.score_domain_projection_abs_tol = 2.0e-8
    hp.score_domain_projection_rel_tol = 3.0e-5
    hp.score_domain_projection_max_frac = 0.75
    hp.score_domain_projection_positive_floor = 4.0e-12
    hp.plateau_stop_enable = True
    hp.plateau_stop_max_soft_restarts = 3
    hp.plateau_stop_min_evals = 1234

    spec = DELabSpec(
        id="threading",
        csv_paths=(),
        order_candidates=(1,),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
        u_col="u",
    )
    x_fit = torch.linspace(0.0, 1.0, 16, dtype=torch.float64).reshape(-1, 1)
    y_fit = -x_fit.clone()
    x_probe = torch.linspace(1.0, 2.0, 16, dtype=torch.float64).reshape(-1, 1)
    y_probe = -x_probe.clone()

    rows, n_seeds, n_seeds_ran, n_iter_each = oracle_lab_de_mod._run_order_search(
        spec=spec,
        order=1,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        hp=hp,
        dtype=torch.float64,
        run_seed=11,
        var_dims=None,
        y_dims=None,
        verbose=False,
        fit_meta=None,
        probe_meta=None,
        traj_metric="mean",
        stop_event=None,
    )

    assert rows == []
    assert int(n_seeds) == 1
    assert int(n_seeds_ran) == 1
    assert int(n_iter_each) == 8
    assert captured["score_mapping_family_mode"] == "poly_only"
    assert captured["brute_score_mapping_family_mode"] == "poly_only"
    assert bool(captured["score_pade_structural_enable"]) is True
    assert int(captured["score_pade_structural_max_degree"]) == 1
    assert int(captured["score_pade_structural_max_total_degree"]) == 2
    assert int(captured["score_pade_structural_max_depth"]) == 7
    assert int(captured["score_pade_structural_max_size"]) == 31
    assert float(captured["score_pade_structural_coeff_tol"]) == pytest.approx(1.0e-9)
    assert float(captured["score_pade_structural_mse_rel_tol"]) == pytest.approx(1.0e-5)
    assert float(captured["score_mapping_expensive_gate_best_factor"]) == pytest.approx(7.0)
    assert float(captured["score_mapping_expensive_rel_y"]) == pytest.approx(0.03)
    assert bool(captured["score_prescreen_enable"]) is True
    assert captured["score_prescreen_family_mode"] == "cheap"
    assert captured["score_prescreen_residual_family_mode"] == "poly_only"
    assert bool(captured["score_prescreen_residual_allow_hint"]) is True
    assert bool(captured["score_prescreen_residual_use_global_best"]) is True
    assert float(captured["score_prescreen_parent_best_factor"]) == pytest.approx(1.25)
    assert float(captured["score_prescreen_global_best_factor"]) == pytest.approx(2.5)
    assert float(captured["score_prescreen_residual_parent_best_factor"]) == pytest.approx(1.05)
    assert float(captured["score_prescreen_residual_global_best_factor"]) == pytest.approx(1.2)
    assert bool(captured["score_domain_projection_enable"]) is True
    assert float(captured["score_domain_projection_abs_tol"]) == pytest.approx(2.0e-8)
    assert float(captured["score_domain_projection_rel_tol"]) == pytest.approx(3.0e-5)
    assert float(captured["score_domain_projection_max_frac"]) == pytest.approx(0.75)
    assert float(captured["score_domain_projection_positive_floor"]) == pytest.approx(4.0e-12)
    assert bool(captured["plateau_stop_enable"]) is True
    assert int(captured["plateau_stop_max_soft_restarts"]) == 3
    assert int(captured["plateau_stop_min_evals"]) == 1234


def test_per_traj_point_build_tables_are_disjoint():
    x = np.linspace(0.01, 1.0, 20, dtype=np.float64)
    u = np.exp(-0.4 * x)
    du = -0.4 * u
    d2u = 0.16 * u
    tr = oracle_lab_de_mod._Trajectory(
        traj_id="ic0",
        path="/tmp/ic0.csv",
        x=x,
        u=u,
        du=du,
        d2u=d2u,
    )

    spec = DELabSpec(
        id="disjoint_split",
        csv_paths=("/tmp/ic0.csv",),
        order_candidates=(1,),
        split_mode="per_traj_point",
        include_x=True,
        include_u=True,
        include_du=True,
    )

    hp = FactorizedSearchConfig()
    hp.n_fit = 20
    hp.n_probe = 20

    tables = oracle_lab_de_mod._build_multi_tables_for_order(
        spec,
        [tr],
        order=1,
        hp=hp,
        seed=13,
        dtype=torch.float64,
    )

    fit_x = {round(float(v), 12) for v in tables.x_fit[:, 0].tolist()}
    probe_x = {round(float(v), 12) for v in tables.x_probe[:, 0].tolist()}
    assert fit_x.isdisjoint(probe_x)
    assert int(tables.x_fit.shape[0]) + int(tables.x_probe.shape[0]) <= int(x.shape[0])


def test_mapping_complexity_penalty_affects_score(monkeypatch):
    class _FakeArch:
        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0,
                best_expr=("var", 0),
                mapping={"kind": "poly", "coeffs": [0.0, 1.0, 2.0], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    hp = FactorizedSearchConfig()
    hp.n_iter = 32
    hp.max_depth = 3
    hp.poly_degree = 3
    hp.return_topk = 1
    hp.n_seeds = 1
    hp.refine_enable = False
    hp.mapping_complexity_penalty = 0.5

    spec = DELabSpec(
        id="penalty_check",
        csv_paths=("dummy.csv",),
        order_candidates=(1,),
        include_x=True,
        include_u=False,
    )

    x = torch.linspace(0.1, 1.0, 12, dtype=torch.float64).reshape(-1, 1)
    y = torch.zeros_like(x)
    rows, _, _, _ = oracle_lab_de_mod._run_order_search(
        spec=spec,
        order=1,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        hp=hp,
        dtype=torch.float64,
        run_seed=11,
        var_dims=None,
        y_dims=None,
        verbose=False,
        probe_meta=None,
        traj_metric="mean",
    )

    row = rows[0]
    assert int(row["mapping_complexity"]) == 3
    assert float(row["score_raw"]) == pytest.approx(1.0)
    assert float(row["score"]) == pytest.approx(2.5)


def test_mapping_complexity_charges_hidden_score_head():
    mapping = {
        "kind": "poly",
        "coeffs": [0.0, 1.0],
        "mu": 0.0,
        "std": 1.0,
        "_lin_head": {
            "terms": [("var", 0)],
            "coeffs": [0.1, 2.0],
        },
    }

    assert oracle_lab_de_mod._mapping_complexity(mapping) == 5


def test_run_order_search_rejects_hidden_score_head_by_default(monkeypatch):
    class _FakeArch:
        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-12,
                best_expr=("var", 0),
                mapping={
                    "kind": "poly",
                    "coeffs": [0.0, 1.0],
                    "mu": 0.0,
                    "std": 1.0,
                    "_lin_head": {
                        "terms": [("var", 0)],
                        "coeffs": [0.0, 1.0],
                    },
                    "_score_decomp": {
                        "mse_core": 1.0,
                        "mse_with_head": 1.0e-12,
                        "head_rel_gain": 1.0,
                    },
                },
            )
            return [rec]

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    hp = FactorizedSearchConfig()
    hp.n_iter = 8
    hp.max_depth = 3
    hp.poly_degree = 1
    hp.return_topk = 1
    hp.n_seeds = 1
    hp.refine_enable = False
    hp.de_accept_hidden_score_head = False

    spec = DELabSpec(
        id="hidden_head_reject",
        csv_paths=("dummy.csv",),
        order_candidates=(1,),
        include_x=True,
        include_u=False,
    )
    x = torch.linspace(0.1, 1.0, 12, dtype=torch.float64).reshape(-1, 1)
    y = torch.zeros_like(x)
    diagnostics: dict[str, object] = {}

    rows, _, _, _ = oracle_lab_de_mod._run_order_search(
        spec=spec,
        order=1,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        hp=hp,
        dtype=torch.float64,
        run_seed=11,
        var_dims=None,
        y_dims=None,
        verbose=False,
        probe_meta=None,
        traj_metric="mean",
        diagnostics_out=diagnostics,
    )

    assert rows == []
    assert int(diagnostics["hidden_score_head_skipped"]) == 1


def test_run_order_search_can_explicitly_accept_hidden_score_head(monkeypatch):
    class _FakeArch:
        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-12,
                best_expr=("var", 0),
                mapping={
                    "kind": "poly",
                    "coeffs": [0.0, 1.0],
                    "mu": 0.0,
                    "std": 1.0,
                    "_lin_head": {
                        "terms": [("var", 0)],
                        "coeffs": [0.0, 1.0],
                    },
                },
            )
            return [rec]

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    hp = FactorizedSearchConfig()
    hp.n_iter = 8
    hp.max_depth = 3
    hp.poly_degree = 1
    hp.return_topk = 1
    hp.n_seeds = 1
    hp.refine_enable = False
    hp.de_accept_hidden_score_head = True

    spec = DELabSpec(
        id="hidden_head_accept",
        csv_paths=("dummy.csv",),
        order_candidates=(1,),
        include_x=True,
        include_u=False,
    )
    x = torch.linspace(0.1, 1.0, 12, dtype=torch.float64).reshape(-1, 1)
    y = torch.zeros_like(x)
    diagnostics: dict[str, object] = {}

    rows, _, _, _ = oracle_lab_de_mod._run_order_search(
        spec=spec,
        order=1,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        hp=hp,
        dtype=torch.float64,
        run_seed=11,
        var_dims=None,
        y_dims=None,
        verbose=False,
        probe_meta=None,
        traj_metric="mean",
        diagnostics_out=diagnostics,
    )

    assert len(rows) == 1
    assert rows[0]["hidden_score_head"] is True
    assert int(diagnostics["hidden_score_head_skipped"]) == 0


def test_run_oracle_de_from_features_seam_and_fairness(monkeypatch):
    calls = {"n": 0}

    class _FakeArch:
        refine_diagnostics = {
            "score_calls": 3,
            "refinement_attempts": 1,
            "accepted_refinements": 1,
            "attempt_cache_hits": 1,
            "attempt_cache_misses": 1,
            "grid_evals": 5,
            "lbfgs_closures": 4,
            "linear_solves": 2,
            "attempt_cache_size": 1,
        }
        refine_slate_stats = {"total_passes": 1, "total_trials_used": 1}

        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-9,
                best_expr=("var", 0),
                mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    def _fake_run_explorer_core(**kwargs):
        calls["n"] += 1
        return _FakeArch()

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", _fake_run_explorer_core)

    x_fit = torch.linspace(0.1, 1.0, 24, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 24, dtype=torch.float64).reshape(-1, 1)
    feats = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=torch.sin(x_fit),
        du_fit=torch.cos(x_fit),
        d2u_fit=-torch.sin(x_fit),
        x_probe=x_probe,
        u_probe=torch.sin(x_probe),
        du_probe=torch.cos(x_probe),
        d2u_probe=-torch.sin(x_probe),
    )

    spec = DELabSpec(
        id="from_features",
        csv_paths=("dummy.csv",),
        order_candidates=(1, 2),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
    )

    hp = FactorizedSearchConfig()
    hp.n_iter = 100
    hp.max_depth = 3
    hp.poly_degree = 3
    hp.return_topk = 1
    hp.n_fit = 24
    hp.n_probe = 24
    hp.n_seeds = 1
    hp.split_iter_across_seeds = True
    hp.brute_depth = 1
    hp.refine_enable = False

    report = run_oracle_de_from_features(
        spec,
        feats,
        factorized_search_hp=hp,
        seed=17,
        dtype=torch.float64,
        verbose=False,
    )
    assert calls["n"] == 2  # one explorer run per candidate order
    assert report["best"] is not None
    assert report["resolved_config"]["resolved"]["n_fit"] == 24
    assert report["refine_diagnostics"]["refinement_attempts"] == 2
    assert report["refine_cost_summary"]["accepted_per_attempt"] == pytest.approx(1.0)
    assert report["refine_cost_summary"]["attempt_cache_hit_rate"] == pytest.approx(0.5)

    by_order = {int(row["order"]): row for row in report["per_order"]}
    assert int(by_order[1]["nvars"]) == 2  # x, u (no du in order-1 features)
    assert int(by_order[2]["nvars"]) == 3  # x, u, du
    assert by_order[1]["refine_diagnostics"]["refinement_attempts"] == 1
    assert by_order[1]["refine_slate_stats_by_seed"][0]["total_trials_used"] == 1


def test_run_oracle_de_from_features_preserves_full_search_diagnostics(monkeypatch):
    def _fake_run_order_search(*, order, diagnostics_out=None, **_kwargs):
        if diagnostics_out is not None:
            diagnostics_out.update(
                {
                    "refine_diagnostics": {"score_calls": int(order)},
                    "refine_cost_summary": {"accepted_per_attempt": 1.0},
                    "refine_diagnostics_by_seed": [{"seed_search": int(order)}],
                    "refine_slate_stats_by_seed": [{"total_trials_used": int(order)}],
                    "additive_fss": {
                        "enabled": True,
                        "pre_mutation_context_rows": int(order),
                        "post_mutation_context_rows": int(order) + 1,
                        "pre_mutation_combo_rows": int(order) + 2,
                        "post_mutation_combo_rows": int(order) + 3,
                        "pre_mutation_early_return": bool(int(order) == 1),
                        "pre_mutation_contextual_atom_diagnostics": {
                            "enabled": True,
                            "atoms_considered": 4,
                            "residual_periodic_seed_rows": 2,
                            "contexts_tested": 7,
                            "promoted_rows": int(order),
                        },
                    },
                }
            )
        row = {
            "order": int(order),
            "expr": "x0",
            "expr_ast": ("var", 0),
            "mse": float(order),
            "score": float(order),
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            "mapping_kind": "poly",
            "size": 1,
        }
        return [row], 1, 1, 0

    monkeypatch.setattr(oracle_lab_de_mod, "_run_order_search", _fake_run_order_search)

    x_fit = torch.linspace(0.1, 1.0, 8, dtype=torch.float64).reshape(-1, 1)
    x_probe = torch.linspace(1.1, 2.0, 8, dtype=torch.float64).reshape(-1, 1)
    feats = DEFeatureTensors(
        x_fit=x_fit,
        u_fit=torch.sin(x_fit),
        du_fit=torch.cos(x_fit),
        d2u_fit=-torch.sin(x_fit),
        x_probe=x_probe,
        u_probe=torch.sin(x_probe),
        du_probe=torch.cos(x_probe),
        d2u_probe=-torch.sin(x_probe),
    )
    spec = DELabSpec(
        id="diag_propagation",
        csv_paths=("dummy.csv",),
        order_candidates=(1, 2),
        x_axis=0,
        include_x=True,
        include_u=True,
        include_du=True,
    )

    report = run_oracle_de_from_features(
        spec,
        feats,
        factorized_search_hp=FactorizedSearchConfig(),
        seed=3,
        dtype=torch.float64,
        verbose=False,
        parallel_orders=False,
    )

    by_order = {int(row["order"]): row for row in report["per_order"]}
    assert by_order[1]["search_diagnostics"]["additive_fss"]["pre_mutation_context_rows"] == 1
    assert by_order[2]["additive_fss"]["post_mutation_combo_rows"] == 5
    summary = report["search_diagnostics_summary"]
    assert summary["n_orders"] == 2
    assert summary["additive_fss_orders"] == 2
    assert summary["pre_mutation_context_rows"] == 3
    assert summary["post_mutation_context_rows"] == 5
    assert summary["pre_mutation_combo_rows"] == 7
    assert summary["post_mutation_combo_rows"] == 9
    assert summary["pre_mutation_early_return_orders"] == 1
    ctx_summary = summary["orders"][0]["pre_mutation_contextual_atom_diagnostics"]
    assert ctx_summary["atoms_considered"] == 4
    assert ctx_summary["residual_periodic_seed_rows"] == 2
    assert ctx_summary["contexts_tested"] == 7
    assert summary["orders"][0]["pre_mutation_contextual_atom_diagnostics"]["promoted_rows"] == 1


def test_multi_trajectory_scoring_and_holdout(monkeypatch, tmp_path: Path):
    x = np.linspace(0.01, 1.0, 72, dtype=np.float64)
    u0 = np.exp(-0.4 * x)
    du0 = -0.4 * u0
    d2u0 = 0.16 * u0
    u1 = np.exp(-0.9 * x)
    du1 = -0.9 * u1
    d2u1 = 0.81 * u1
    p0 = tmp_path / "ic0.csv"
    p1 = tmp_path / "ic1.csv"
    _write_csv(p0, x, u0, du0, d2u0)
    _write_csv(p1, x, u1, du1, d2u1)

    payload = {
        "id": "multi_score",
        "trajectories": [
            {"id": "ic0", "csv": str(p0)},
            {"id": "ic1", "csv": str(p1)},
        ],
        "order_candidates": [1],
        "split_mode": "traj_holdout",
        "traj_metric": "max",
        "include_x": True,
        "include_u": True,
        "include_du": True,
        "x_col": "x0",
        "u_col": "y",
        "deriv": {
            "method": "precomputed",
            "du_col": "du",
            "d2u_col": "d2u",
        },
    }
    spec = equation_de_spec_from_dict(payload, source="multi-score")

    class _FakeArch:
        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-6,
                best_expr=("var", 0),
                mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    hp = FactorizedSearchConfig()
    hp.n_iter = 80
    hp.max_depth = 3
    hp.poly_degree = 3
    hp.return_topk = 1
    hp.n_fit = 48
    hp.n_probe = 48
    hp.n_seeds = 1
    hp.split_iter_across_seeds = True
    hp.brute_depth = 1
    hp.refine_enable = False

    report = run_oracle_de_equation(spec, factorized_search_hp=hp, seed=11, dtype=torch.float64, verbose=False)
    assert report["best"] is not None
    assert report["split_mode"] == "traj_holdout"
    assert report["traj_metric"] == "max"

    order_row = report["per_order"][0]
    assert int(order_row["n_traj_total"]) == 2
    assert int(order_row["n_traj_fit"]) == 1
    assert int(order_row["n_traj_probe"]) == 1
    assert len(order_row["results"]) == 1
    best = order_row["results"][0]
    assert "mse_traj" in best
    assert len(best["mse_traj"]) == 1
    assert math.isfinite(float(best["score"]))
