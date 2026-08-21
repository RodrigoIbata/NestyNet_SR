# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import torch

from nestynet_sr.sr_search.factorized_search.engine.scoring import score_expr
from nestynet_sr.sr_search.factorized_search.explorer import eval_pade, eval_poly, fit_pade, fit_poly
from nestynet_sr.sr_search.features import _fit_rational_for_degrees
from nestynet_sr.sr_core.atoms import _eval_monomials
from nestynet_sr.sr_search.fitting_utils import _fit_poly_coeffs_1d, _fit_rational_coeffs_1d
from nestynet_sr.sr_search.rational_sparsify import (
    RationalSparsifyConfig,
    stlsq_sparsify_poly_coeffs,
    stlsq_sparsify_rational_coeffs,
)


def _poly_design_1d(x: torch.Tensor, degree: int) -> torch.Tensor:
    cols = []
    xp = torch.ones_like(x)
    for _ in range(degree + 1):
        cols.append(xp)
        xp = xp * x
    return torch.stack(cols, dim=1)


def test_stlsq_sparsify_rational_coeffs_prunes_tiny_terms():
    x = torch.linspace(-1.0, 1.0, 2000, dtype=torch.float64)
    Phi_num = _poly_design_1d(x, degree=4)
    Phi_den = _poly_design_1d(x, degree=4)

    a_true = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    b_true = torch.tensor([1.0, -0.5, 0.0, 0.0, 0.0], dtype=torch.float64)
    y = (Phi_num @ a_true) / (Phi_den @ b_true)

    a_dense = a_true.clone()
    b_dense = b_true.clone()
    a_dense[2:] = torch.tensor([8e-4, -6e-4, 5e-4], dtype=torch.float64)
    b_dense[2:] = torch.tensor([-7e-4, 9e-4, -4e-4], dtype=torch.float64)

    cfg = RationalSparsifyConfig(
        ridge=1e-10,
        max_iter=15,
        lam_abs=0.0,
        lam_rel=2e-2,
        pivot_index=0,
        prefer_den_constant=True,
        unbiased_refit=True,
        resid_increase_tolerance=10.0,
    )
    a_sparse, b_sparse, meta = stlsq_sparsify_rational_coeffs(
        Phi_num=Phi_num,
        Phi_den=Phi_den,
        y=y,
        coeffs_num=a_dense,
        coeffs_den=b_dense,
        cfg=cfg,
    )

    assert int(meta.get("nnz_sparse", 999)) <= int(meta.get("nnz_seed", 0))
    assert abs(float(a_sparse[2])) < 1e-8
    assert abs(float(a_sparse[3])) < 1e-8
    assert abs(float(a_sparse[4])) < 1e-8
    assert abs(float(b_sparse[2])) < 1e-8
    assert abs(float(b_sparse[3])) < 1e-8
    assert abs(float(b_sparse[4])) < 1e-8

    y_hat = (Phi_num @ a_sparse) / (Phi_den @ b_sparse)
    mse = float(((y_hat - y) ** 2).mean().item())
    assert mse < 1e-10


def test_stlsq_sparsify_poly_coeffs_prunes_tiny_terms():
    x = torch.linspace(-1.0, 1.0, 2000, dtype=torch.float64)
    Phi = _poly_design_1d(x, degree=6)
    y = 1.0 + 2.0 * x + 0.5 * x * x

    c_dense = torch.tensor(
        [1.0, 2.0, 0.5, 8e-4, -7e-4, 6e-4, -5e-4], dtype=torch.float64
    )
    cfg = RationalSparsifyConfig(lam_rel=1e-2, pivot_index=0, prefer_den_constant=True)
    c_sparse, meta = stlsq_sparsify_poly_coeffs(Phi=Phi, y=y, coeffs=c_dense, cfg=cfg)

    assert int(meta.get("nnz_sparse", 999)) <= int(meta.get("nnz_seed", 0))
    assert abs(float(c_sparse[3])) < 1e-8
    assert abs(float(c_sparse[4])) < 1e-8
    assert abs(float(c_sparse[5])) < 1e-8
    assert abs(float(c_sparse[6])) < 1e-8

    y_hat = Phi @ c_sparse
    mse = float(((y_hat - y) ** 2).mean().item())
    assert mse < 1e-10


def test_fit_rational_coeffs_1d_returns_compact_support():
    x = torch.linspace(-1.0, 1.0, 2000, dtype=torch.float64)
    y = (1.0 + 2.0 * x) / (1.0 - 0.5 * x)

    fit = _fit_rational_coeffs_1d(
        x=x,
        f=y,
        deg_num=4,
        deg_den=4,
        min_points=200,
        dtype=torch.float64,
    )
    assert fit is not None
    a, b = fit

    assert abs(float(b[0]) - 1.0) < 1e-6
    nnz = int((torch.cat([a, b]).abs() > 1e-5).sum().item())
    assert nnz <= 5

    Phi_num = _poly_design_1d(x, degree=4)
    Phi_den = _poly_design_1d(x, degree=4)
    y_hat = (Phi_num @ a) / (Phi_den @ b)
    mse = float(((y_hat - y) ** 2).mean().item())
    assert mse < 1e-8


def test_fit_rational_coeffs_1d_can_return_sparse_support():
    x = torch.linspace(-1.0, 1.0, 2000, dtype=torch.float64)
    y = (1.0 + 2.0 * x) / (1.0 - 0.5 * x)

    fit = _fit_rational_coeffs_1d(
        x=x,
        f=y,
        deg_num=4,
        deg_den=4,
        min_points=200,
        dtype=torch.float64,
        return_support=True,
    )
    assert fit is not None
    a_sparse, b_sparse, exps_num_sparse, exps_den_sparse = fit

    assert exps_num_sparse.tolist() == [[0], [1]]
    assert exps_den_sparse.tolist() == [[0], [1]]
    assert int(a_sparse.numel()) == 2
    assert int(b_sparse.numel()) == 2

    Phi_num = _eval_monomials(x.view(-1, 1), exps_num_sparse)
    Phi_den = _eval_monomials(x.view(-1, 1), exps_den_sparse)
    y_hat = (Phi_num @ a_sparse) / (Phi_den @ b_sparse)
    mse = float(((y_hat - y) ** 2).mean().item())
    assert mse < 1e-8


def test_fit_poly_coeffs_1d_returns_compact_support():
    x = torch.linspace(-1.0, 1.0, 2000, dtype=torch.float64)
    y = 1.0 + 2.0 * x + 0.5 * x * x
    coeffs = _fit_poly_coeffs_1d(
        x=x,
        f=y,
        degree=6,
        min_points=200,
        dtype=torch.float64,
    )
    assert coeffs is not None
    nnz = int((coeffs.abs() > 1e-5).sum().item())
    assert nnz <= 4
    Phi = _poly_design_1d(x, degree=6)
    mse = float(((Phi @ coeffs - y) ** 2).mean().item())
    assert mse < 1e-8


def test_fit_pade_prunes_high_order_noise_terms():
    x = torch.linspace(-1.0, 1.0, 1600, dtype=torch.float64).unsqueeze(-1)
    y = ((1.0 + 2.0 * x) / (1.0 - 0.5 * x)).to(dtype=torch.float64)

    mapping = fit_pade(x, y, numer_deg=4, denom_deg=4, n_iters=12)
    assert mapping is not None
    assert mapping["kind"] == "pade"

    numer = mapping["numer"]
    denom = mapping["denom"]
    nnz = int((torch.cat([numer, denom]).abs() > 1e-5).sum().item())
    assert nnz <= 5

    y_hat = eval_pade(x, mapping)
    mse = float(((y_hat - y) ** 2).mean().item())
    assert mse < 1e-8


def test_factorized_search_fit_poly_prunes_high_order_terms():
    x = torch.linspace(-1.0, 1.0, 1600, dtype=torch.float64).unsqueeze(-1)
    y = (1.0 + 2.0 * x + 0.5 * x * x).to(dtype=torch.float64)

    fit = fit_poly(x, y, degree=6)
    assert fit is not None
    coeffs, mu, std = fit
    nnz = int((coeffs.abs() > 1e-5).sum().item())
    assert nnz <= 4
    y_hat = eval_poly(x, coeffs, mu, std)
    mse = float(((y_hat - y) ** 2).mean().item())
    assert mse < 1e-8


def test_factorized_search_fit_poly_affine_fast_matches_degree_one_fit():
    x = torch.linspace(-2.0, 2.0, 2400, dtype=torch.float64).unsqueeze(-1)
    z = (x - x.mean()) / x.std()
    y = 1.25 - 3.5 * z + 0.01 * torch.sin(x)

    baseline = fit_poly(x, y, degree=1)
    diagnostics: dict[str, float] = {}
    fast = fit_poly(x, y, degree=1, affine_fast=True, diagnostics=diagnostics)

    assert baseline is not None
    assert fast is not None
    coeffs_base, mu_base, std_base = baseline
    coeffs_fast, mu_fast, std_fast = fast
    assert torch.allclose(coeffs_fast, coeffs_base, atol=1e-10, rtol=1e-10)
    assert abs(float(mu_fast) - float(mu_base)) < 1e-12
    assert abs(float(std_fast) - float(std_base)) < 1e-12

    y_base = eval_poly(x, coeffs_base, mu_base, std_base)
    y_fast = eval_poly(x, coeffs_fast, mu_fast, std_fast)
    assert torch.allclose(y_fast, y_base, atol=1e-10, rtol=1e-10)
    assert diagnostics["fit_poly_affine_fast_calls"] == 1
    assert diagnostics.get("fit_poly_lstsq_calls", 0) == 0
    assert diagnostics.get("fit_poly_stlsq_calls", 0) == 0
    assert diagnostics["fit_poly_wall_seconds"] >= diagnostics["fit_poly_affine_fast_wall_seconds"] >= 0.0


def test_score_expr_poly_only_degree_one_reports_affine_fast_diagnostics():
    x_fit = torch.linspace(-1.0, 1.0, 96, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(-0.9, 1.1, 128, dtype=torch.float64).unsqueeze(-1)
    y_fit = 0.75 + 2.0 * x_fit
    y_probe = 0.75 + 2.0 * x_probe
    proj = torch.randn(x_probe.shape[0], 4, dtype=torch.float64)
    diagnostics: dict[str, float] = {}

    scored = score_expr(
        ("var", 0),
        x_fit,
        y_fit,
        x_probe,
        y_probe,
        proj,
        "bits",
        2.0,
        6,
        1,
        refine_enable=False,
        refine_cfg={
            "score_mapping_family_mode": "poly_only",
            "score_prescreen_enable": True,
            "score_prescreen_family_mode": "cheap",
            "diagnostics": diagnostics,
        },
        return_expr=True,
    )

    assert scored is not None
    assert diagnostics["fit_poly_affine_fast_calls"] >= 1
    assert diagnostics.get("fit_poly_lstsq_calls", 0) == 0
    assert diagnostics.get("fit_poly_stlsq_calls", 0) == 0
    assert diagnostics["fit_poly_wall_seconds"] >= diagnostics["fit_poly_affine_fast_wall_seconds"] >= 0.0


def test_features_rational_fit_reports_compact_support():
    x = torch.linspace(-1.0, 1.0, 2000, dtype=torch.float64).unsqueeze(1)
    y = (1.0 + 2.0 * x[:, 0]) / (1.0 - 0.5 * x[:, 0])

    spec = _fit_rational_for_degrees(
        X=x,
        F=y,
        Nxvars=1,
        deg_num=4,
        deg_den=4,
        min_points=200,
    )
    assert spec is not None
    assert spec.rms_rel < 1e-6
    # Baseline dense basis would report 5+5 terms; de-Padeifier should compact.
    assert int(spec.n_terms_num + spec.n_terms_den) <= 6
