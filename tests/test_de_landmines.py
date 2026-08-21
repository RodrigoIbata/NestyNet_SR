# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nestynet_sr.run_de import write_de_json_report
from nestynet_sr.sr_de.de_search import (
    DESearchConfig,
    DESearchResult,
    _maybe_scale_normalized_refit_matrix,
    _validation_prune_multi_support,
    discover_de_from_surrogate,
    ridge_lstsq,
)
from nestynet_sr.sr_de.varpro_de import _group_stlsq_multi


class _IdentitySurrogate(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


class _ExponentialDecaySurrogate(torch.nn.Module):
    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.rate * x[:, :1])

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return -self.rate * torch.exp(-self.rate * x[:, :1])


def test_group_stlsq_multi_uses_group_stlsq(monkeypatch):
    calls = {}

    def _fake_group_stlsq(phi_list, y_list, *, ridge, lam, max_iter):
        calls["n"] = int(calls.get("n", 0)) + 1
        calls["shapes"] = [tuple(phi.shape) for phi in phi_list]
        return (
            torch.tensor([[3.0], [4.0]], dtype=torch.float64),
            torch.tensor([True]),
        )

    import nestynet_sr.sr_de.de_search as de_search_mod

    monkeypatch.setattr(de_search_mod, "group_stlsq", _fake_group_stlsq)

    x_list = [
        torch.ones((5, 1), dtype=torch.float64),
        torch.ones((7, 1), dtype=torch.float64),
    ]
    y_list = [
        torch.zeros(5, dtype=torch.float64),
        torch.zeros(7, dtype=torch.float64),
    ]

    coeffs_list, keep_mask = _group_stlsq_multi(
        X_list=x_list,
        y_list=y_list,
        term_asts=[None],
        surrogates=[torch.nn.Identity(), torch.nn.Identity()],
        order=1,
        x_axis=0,
        ridge=1.0e-6,
        lam=1.0e-3,
        max_iter=3,
        device=torch.device("cpu"),
    )

    assert int(calls["n"]) == 1
    assert calls["shapes"] == [(5, 1), (7, 1)]
    assert keep_mask.tolist() == [True]
    assert [float(v.item()) for v in coeffs_list] == [3.0, 4.0]


def test_discover_de_from_surrogate_persists_condition_number():
    x = torch.linspace(1.0, 2.0, 32, dtype=torch.float64).reshape(-1, 1)
    y = torch.zeros_like(x)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=16,
        shuffle=False,
    )

    cfg = DESearchConfig(
        x_axis=0,
        order_candidates=(1,),
        include_const=False,
        include_x=False,
        include_u=True,
        include_du=False,
        include_d2u=False,
        include_xu=False,
        include_xdu=False,
        stlsq_lambda=0.0,
        ridge=1.0e-8,
    )
    result = discover_de_from_surrogate(
        _IdentitySurrogate(),
        loader,
        loader,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert result.condition_number is not None
    assert float(result.condition_number) >= 1.0


def test_discover_de_from_surrogate_validation_residual_uses_anchor_sign():
    x_train = torch.linspace(0.0, 1.0, 64, dtype=torch.float64).reshape(-1, 1)
    x_val = torch.linspace(1.2, 2.0, 64, dtype=torch.float64).reshape(-1, 1)
    y_train = torch.zeros_like(x_train)
    y_val = torch.zeros_like(x_val)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=32,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val),
        batch_size=32,
        shuffle=False,
    )

    result = discover_de_from_surrogate(
        _ExponentialDecaySurrogate(rate=0.8),
        train_loader,
        val_loader,
        cfg=DESearchConfig(
            x_axis=0,
            order_candidates=(1,),
            include_const=True,
            include_x=False,
            include_u=True,
            max_u_power=3,
            include_du=False,
            include_d2u=False,
            include_xu=False,
            include_xdu=False,
            stlsq_lambda=0.0,
            ridge=1.0e-12,
        ),
        device=torch.device("cpu"),
    )

    assert len(result.term_asts) == 1
    assert result.coeffs.tolist() == pytest.approx([0.8], abs=1.0e-10)
    assert result.rms_train < 1.0e-10
    assert result.rms_val is not None
    assert result.rms_val < 1.0e-10


def test_scale_normalized_refit_ignores_singular_boundary_leverage():
    x = torch.linspace(1.0e-3, 10.0, 5000, dtype=torch.float64)
    phi = (x.square().reciprocal()).reshape(-1, 1)
    y = phi[:, 0].clone()
    y[0] = 0.0

    ordinary = ridge_lstsq(phi, y, ridge=0.0)
    refit, used = _maybe_scale_normalized_refit_matrix(phi, y, ordinary)

    assert used
    assert float(ordinary[0]) < 0.75
    assert float(refit[0]) == pytest.approx(1.0, abs=1.0e-3)


def test_guarded_scale_normalized_refit_handles_radial_boundary_row():
    x = torch.linspace(1.0e-3, 35.9, 5000, dtype=torch.float64)
    h = x[1] - x[0]
    k = 1.75
    r0 = x[0]
    u0 = torch.tensor(0.9922092925553776, dtype=torch.float64)
    du0 = torch.tensor(-0.02172236744541437, dtype=torch.float64)
    basis = torch.tensor(
        [
            [torch.sin(k * r0) / r0, torch.cos(k * r0) / r0],
            [
                k * torch.cos(k * r0) / r0 - torch.sin(k * r0) / r0.square(),
                -k * torch.sin(k * r0) / r0 - torch.cos(k * r0) / r0.square(),
            ],
        ],
        dtype=torch.float64,
    )
    a, b = torch.linalg.solve(basis, torch.stack([u0, du0]))
    u = (a * torch.sin(k * x) + b * torch.cos(k * x)) / x

    def _grad(z: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(z)
        out[1:-1] = (z[2:] - z[:-2]) / (2.0 * h)
        out[0] = (-3.0 * z[0] + 4.0 * z[1] - z[2]) / (2.0 * h)
        out[-1] = (3.0 * z[-1] - 4.0 * z[-2] + z[-3]) / (2.0 * h)
        return out

    du_fd = _grad(u)
    d2u_fd = _grad(du_fd)
    phi = torch.stack([u, du_fd / x], dim=1)
    truth = torch.tensor([k * k, 2.0], dtype=torch.float64)
    y = -d2u_fd

    ordinary = ridge_lstsq(phi, y, ridge=0.0)
    refit, used = _maybe_scale_normalized_refit_matrix(phi, y, ordinary)

    assert used
    ordinary_err = torch.linalg.vector_norm(ordinary - truth).item()
    refit_err = torch.linalg.vector_norm(refit - truth).item()
    assert ordinary_err > 0.5
    assert refit_err < 0.02


def test_validation_prune_keeps_real_low_amplitude_nonlinear_term():
    u = torch.linspace(0.07, 0.7, 256, dtype=torch.float64)
    Phi = torch.stack([u, u.square()], dim=1)
    y = -(u + u.square())
    keep = torch.tensor([True, True])

    keep_out, coeffs, rms_train, rms_val = _validation_prune_multi_support(
        [Phi],
        [y],
        keep,
        Phi_vals=[Phi],
        y_vals=[y],
        sparsity_penalty=0.1,
    )

    assert keep_out.tolist() == [True, True]
    assert coeffs.reshape(-1).tolist() == pytest.approx([-1.0, -1.0], abs=1.0e-10)
    assert rms_train[0] < 1.0e-10
    assert rms_val is not None
    assert rms_val[0] < 1.0e-10


def test_write_de_json_report_includes_condition_number(tmp_path: Path):
    result = DESearchResult(
        order=1,
        x_axis=0,
        term_asts=[None],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=1.0e-3,
        rms_val=2.0e-3,
        condition_number=123.0,
    )
    report_path = tmp_path / "report.json"

    args = SimpleNamespace(
        device=None,
        num_segments=8,
        epochs=100,
        loss_target=1.0e-8,
        order_candidates="1",
        max_x_power=1,
        max_u_power=1,
        max_xu_total_degree=0,
        include_xdu=False,
        include_inv_xdu=False,
        include_inv_xu=False,
        include_inv_x2u=False,
        include_du=False,
        include_d2u=False,
        include_udu=False,
        stlsq_lambda=1.0e-3,
        sparsity_penalty=1.0e-3,
        enforce_units=False,
        units_policy=None,
        nn_units_semantics=None,
        stageb_refine_residual=False,
        stageb_epochs=0,
    )

    write_de_json_report(
        ["dummy.csv"],
        str(report_path),
        [1.0e-4],
        result,
        args,
        walltime=0.0,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert float(payload["de_discovery"]["condition_number"]) == 123.0
    assert payload["de_discovery"]["selected_engine"] == "stlsq"
    assert len(payload["de_discovery"]["proposal_slate"]) == 1
    assert payload["de_discovery"]["proposal_slate"][0]["engine"] == "stlsq"
