# SPDX-License-Identifier: MPL-2.0

from types import SimpleNamespace

import pytest
import torch

from nestynet_sr.sr_core.bridges import D2U, Mul, Pow, U, Var
from nestynet_sr.sr_gs.prolongation import (
    AffinePointGenerator,
    build_known_lie_point_generators,
    score_affine_point_generators_from_jets,
    score_known_lie_point_generators_from_jets,
)


def _row(meta, name):
    for item in meta["generators"]:
        if item.get("name") == name:
            return item
    raise AssertionError(f"missing generator {name}")


def test_known_lie_bank_includes_graph_rotation_and_lorentz_without_general_affine():
    cfg = SimpleNamespace(
        gs_known_generators=True,
        gs_general_affine=False,
        gs_translations=True,
        gs_diagonal_translations=True,
        gs_scalings=True,
        gs_rotations=True,
        gs_lorentz_boosts=True,
    )
    x = torch.linspace(0.0, 4.0, 64, dtype=torch.float64).unsqueeze(1)
    u = torch.sin(x) + 2.0

    gens = build_known_lie_point_generators(cfg=cfg, x=x, u=u)
    by_name = {g.name: g for g in gens}
    families = {g.family for g in gens}

    assert "graph_rotation_xu" in by_name
    assert any(name.startswith("graph_lorentz_boost_") for name in by_name)
    assert "rotation" in families
    assert "lorentz" in families
    assert "sparse_affine" not in families
    assert all("gamma" not in g.description.lower() for g in gens)


def test_graph_lorentz_velocity_and_acceleration_action_are_prolonged():
    cfg = SimpleNamespace(gs_known_generators=True, gs_general_affine=False, gs_lorentz_boosts=True)
    gen = next(g for g in build_known_lie_point_generators(cfg=cfg) if g.name == "graph_lorentz_boost_unit")
    x = torch.tensor([[2.0]], dtype=torch.float64)
    u = torch.tensor([[3.0]], dtype=torch.float64)
    u1 = torch.tensor([[0.5]], dtype=torch.float64)
    u2 = torch.tensor([[0.7]], dtype=torch.float64)

    xi, eta, eta1, eta2 = gen.fields(x, u, u1, u2)

    assert torch.allclose(xi, u)
    assert torch.allclose(eta, x)
    assert torch.allclose(eta1, 1.0 - u1.pow(2))
    assert torch.allclose(eta2, -3.0 * u1 * u2)


def test_graph_rotation_velocity_and_acceleration_action_are_prolonged():
    cfg = SimpleNamespace(gs_known_generators=True, gs_general_affine=False, gs_rotations=True)
    gen = next(g for g in build_known_lie_point_generators(cfg=cfg) if g.name == "graph_rotation_xu")
    x = torch.tensor([[2.0]], dtype=torch.float64)
    u = torch.tensor([[3.0]], dtype=torch.float64)
    u1 = torch.tensor([[0.5]], dtype=torch.float64)
    u2 = torch.tensor([[0.7]], dtype=torch.float64)

    xi, eta, eta1, eta2 = gen.fields(x, u, u1, u2)

    assert torch.allclose(xi, -u)
    assert torch.allclose(eta, x)
    assert torch.allclose(eta1, 1.0 + u1.pow(2))
    assert torch.allclose(eta2, 3.0 * u1 * u2)


def test_score_reports_known_lie_bank_and_lorentz_family_when_enabled():
    cfg = SimpleNamespace(
        gs_known_generators=True,
        gs_general_affine=False,
        gs_rotations=True,
        gs_lorentz_boosts=True,
        gs_de_lie_prolongation_tol=0.05,
    )
    x = torch.linspace(0.1, 3.0, 80, dtype=torch.float64).unsqueeze(1)
    u = torch.sin(x)
    u1 = torch.cos(x)
    u2 = -torch.sin(x)

    meta = score_known_lie_point_generators_from_jets(
        order=2,
        x=x,
        u=u,
        u1=u1,
        u2=u2,
        term_asts=[U()],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        cfg=cfg,
    )

    assert meta["generator_bank"] == "known_lie_point_affine_xu"
    assert "lorentz" in meta["tested_generator_families"]
    assert "rotation" in meta["tested_generator_families"]
    assert meta["tested_generators"] >= 10
    assert all("gamma" not in str(row).lower() for row in meta["generators"])


def test_harmonic_oscillator_accepts_output_scaling():
    x = torch.linspace(0.1, 3.0, 160, dtype=torch.float64).unsqueeze(1)
    u = torch.sin(x)
    u1 = torch.cos(x)
    u2 = -torch.sin(x)

    meta = score_affine_point_generators_from_jets(
        order=2,
        x=x,
        u=u,
        u1=u1,
        u2=u2,
        term_asts=[U()],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        include_known=True,
        include_general_affine=False,
        tol=1.0e-8,
    )

    item = _row(meta, "u_scaling")
    assert meta["status"] == "scored"
    assert item["accepted"] is True
    assert item["on_shell_metric"] < 1.0e-8


def test_nonhomogeneous_oscillator_rejects_output_scaling():
    x = torch.linspace(0.1, 3.0, 160, dtype=torch.float64).unsqueeze(1)
    u = torch.sin(x)
    u1 = torch.cos(x)
    u2 = -torch.sin(x)

    meta = score_affine_point_generators_from_jets(
        order=2,
        x=x,
        u=u,
        u1=u1,
        u2=u2,
        term_asts=[U(), None],
        coeffs=torch.tensor([1.0, 1.0], dtype=torch.float64),
        include_known=True,
        include_general_affine=False,
        tol=0.05,
    )

    item = _row(meta, "u_scaling")
    assert item["accepted"] is False
    assert item["on_shell_metric"] > 0.05


def test_radial_inflow_accepts_scaling_symmetry():
    x = torch.linspace(0.2, 4.0, 160, dtype=torch.float64).unsqueeze(1)
    u = 1.0 / x
    u1 = -1.0 / x.pow(2)
    u2 = 2.0 / x.pow(3)
    radial_term = Mul(Pow(Var(0), -1.0), U())

    meta = score_affine_point_generators_from_jets(
        order=1,
        x=x,
        u=u,
        u1=u1,
        u2=u2,
        term_asts=[radial_term],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        include_known=True,
        include_general_affine=False,
        tol=1.0e-8,
    )

    assert "x_scaling" in set(meta["accepted_generator_names"])
    assert _row(meta, "x_scaling")["on_shell_metric"] < 1.0e-8


def test_de_search_prolongation_metric_handles_nonfinite_values():
    from nestynet_sr.sr_de.de_search import _prolongation_metric

    assert _prolongation_metric(None) is None
    assert _prolongation_metric({"best_metric": "not-a-number"}) is None
    assert _prolongation_metric({"best_metric": float("inf")}) is None
    assert _prolongation_metric({"best_metric": 0.125}) == 0.125


def test_prolongation_score_is_invariant_to_generator_rescaling():
    x = torch.linspace(0.2, 2.0, 192, dtype=torch.float64).unsqueeze(1)
    u = torch.sin(x)
    u1 = torch.cos(x)
    u2 = -torch.sin(x)
    term = Mul(Var(0), U())
    metrics = []
    accepted = []
    for scale in (1.0, 0.1, 1.0e-6):
        meta = score_affine_point_generators_from_jets(
            order=2,
            x=x,
            u=u,
            u1=u1,
            u2=u2,
            term_asts=[term],
            coeffs=torch.tensor([1.0], dtype=torch.float64),
            generators=[AffinePointGenerator("scaled_x_translation", "translation", a0=scale)],
            tol=0.05,
        )
        row = meta["generators"][0]
        metrics.append(row["on_shell_metric"])
        accepted.append(row["accepted"])
    assert max(metrics) - min(metrics) < 1.0e-12
    assert accepted == [False, False, False]


def test_prolongation_rejects_terms_containing_anchor_derivative():
    x = torch.linspace(0.2, 2.0, 64, dtype=torch.float64).unsqueeze(1)
    u = torch.sin(x)
    u1 = torch.cos(x)
    u2 = -torch.sin(x)
    with pytest.raises(ValueError, match="anchor derivative"):
        score_affine_point_generators_from_jets(
            order=2,
            x=x,
            u=u,
            u1=u1,
            u2=u2,
            term_asts=[D2U(0, 0)],
            coeffs=torch.tensor([0.25], dtype=torch.float64),
            include_known=True,
        )


def test_prolongation_rejects_term_order_above_anchor_order():
    x = torch.linspace(0.2, 2.0, 64, dtype=torch.float64).unsqueeze(1)
    u = torch.sin(x)
    u1 = torch.cos(x)
    u2 = -torch.sin(x)
    with pytest.raises(ValueError, match="exceeds residual anchor order"):
        score_affine_point_generators_from_jets(
            order=1,
            x=x,
            u=u,
            u1=u1,
            u2=u2,
            term_asts=[D2U(0, 0)],
            coeffs=torch.tensor([0.25], dtype=torch.float64),
            include_known=True,
        )
