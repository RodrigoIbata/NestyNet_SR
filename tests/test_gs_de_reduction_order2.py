# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import numpy as np
import pytest
import torch


def _lane_emden_ensemble(n: int = 400):
    x = np.linspace(0.05, 2.8, n)
    trajs, u1s, u2s = [], [], []
    for a in (0.6, 1.0, 1.5):
        u = a * np.sin(x) / x
        u1 = a * (np.cos(x) * x - np.sin(x)) / x**2
        u2 = -u - 2.0 * u1 / x
        trajs.append((x, u))
        u1s.append(u1)
        u2s.append(u2)
    return trajs, u1s, u2s


@pytest.mark.parametrize(
    "coeffs",
    [
        (0, 0, 0, 0, 0, 1),      # u d_u
        (0, 0, 0, 1, 0, 0.5),    # shifted u flow
        (1, 1, 0, 0, 0, 0),      # (1+x) d_x
        (0, 1, 0, -1, 0, 0),     # x d_x - d_u
        (1, 0, 0, 0, 0, 1),      # d_x + u d_u
        (0, 1, 0, 0, 0, 2),      # joint scaling
    ],
)
def test_chart_partials_match_finite_differences(coeffs):
    from nestynet_sr.sr_gs.de_reduction import chart_partials, compile_canonical_chart

    chart = compile_canonical_chart(coeffs)
    parts_fn = chart_partials(chart)
    rng = np.random.default_rng(2)
    x = rng.uniform(0.3, 2.0, 64)
    u = rng.uniform(0.4, 2.0, 64)
    h = 1.0e-5
    parts = parts_fn(x, u)
    for name, fn in (("r", chart.r_fn), ("s", chart.s_fn)):
        fx = (fn(x + h, u) - fn(x - h, u)) / (2 * h)
        fu = (fn(x, u + h) - fn(x, u - h)) / (2 * h)
        fxx = (fn(x + h, u) - 2 * fn(x, u) + fn(x - h, u)) / h**2
        fuu = (fn(x, u + h) - 2 * fn(x, u) + fn(x, u - h)) / h**2
        fxu = (fn(x + h, u + h) - fn(x + h, u - h) - fn(x - h, u + h) + fn(x - h, u - h)) / (4 * h * h)
        np.testing.assert_allclose(parts[f"{name}_x"], fx, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(parts[f"{name}_u"], fu, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(parts[f"{name}_xx"], fxx, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(parts[f"{name}_uu"], fuu, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(parts[f"{name}_xu"], fxu, rtol=1e-3, atol=1e-3)


def test_lane_emden_reduces_to_exact_riccati():
    from nestynet_sr.sr_gs.de_reduction import (
        compile_canonical_chart,
        fit_reduced_first_order,
        reduce_trajectories_order2,
    )

    trajs, u1s, u2s = _lane_emden_ensemble()
    chart = compile_canonical_chart((0, 0, 0, 0, 0, 1))
    reduced = reduce_trajectories_order2(trajs, chart, u1_list=u1s, u2_list=u2s)
    assert reduced["status"] == "reduced"
    fit = fit_reduced_first_order(reduced["r"], reduced["v"], reduced["dvdr"])
    assert fit is not None and fit["val_rmse_rel"] < 1e-10
    got = {item["name"]: item["coeff"] for item in fit["selected"]}
    assert set(got) == {"1", "v^2", "v/r"}
    assert got["1"] == pytest.approx(-1.0, rel=1e-6)
    assert got["v^2"] == pytest.approx(-1.0, rel=1e-6)
    assert got["v/r"] == pytest.approx(-2.0, rel=1e-6)


def test_order2_pullback_matches_true_rhs_numerically():
    from nestynet_sr.sr_gs.de_reduction import (
        compile_canonical_chart,
        fit_reduced_first_order,
        pullback_order2,
        reduce_trajectories_order2,
    )
    from nestynet_sr.sr_gs.prolongation import _eval_term_on_jets

    trajs, u1s, u2s = _lane_emden_ensemble()
    chart = compile_canonical_chart((0, 0, 0, 0, 0, 1))
    reduced = reduce_trajectories_order2(trajs, chart, u1_list=u1s, u2_list=u2s)
    fit = fit_reduced_first_order(reduced["r"], reduced["v"], reduced["dvdr"])
    pulled = pullback_order2(chart, fit)
    x, u = trajs[0]
    u1, u2 = u1s[0], u2s[0]
    xt = torch.as_tensor(x).reshape(-1, 1)
    ut = torch.as_tensor(u).reshape(-1, 1)
    u1t = torch.as_tensor(u1).reshape(-1, 1)
    rhs = _eval_term_on_jets(
        pulled["rhs_ast"], x=xt, u=ut, u1=u1t, u2=torch.zeros_like(xt), x_axis=0
    ).detach().cpu().numpy().reshape(-1)
    np.testing.assert_allclose(rhs, u2, rtol=1e-8, atol=1e-10)


def test_driver_order2_emits_rows_and_rejects_bad_order():
    from nestynet_sr.sr_gs.de_reduction import symmetry_reduction_proposals

    trajs, u1s, u2s = _lane_emden_ensemble()
    result = symmetry_reduction_proposals(trajs, u1_list=u1s, u2_list=u2s, order=2)
    assert result["status"] == "ok" and result["order"] == 2
    assert result["proposals"], "expected an order-2 proposal"
    fams = {fam for _t, _s, fam in result["library_rows"]}
    assert any(f.startswith("reduction2_") for f in fams)
    with pytest.raises(ValueError):
        symmetry_reduction_proposals(trajs, order=3)


def test_order2_finite_difference_path_without_derivatives():
    from nestynet_sr.sr_gs.de_reduction import (
        compile_canonical_chart,
        fit_reduced_first_order,
        reduce_trajectories_order2,
    )

    trajs, _u1s, _u2s = _lane_emden_ensemble(n=800)
    chart = compile_canonical_chart((0, 0, 0, 0, 0, 1))
    reduced = reduce_trajectories_order2(trajs, chart)  # FD fallback
    assert reduced["status"] == "reduced"
    fit = fit_reduced_first_order(reduced["r"], reduced["v"], reduced["dvdr"])
    assert fit is not None and fit["val_rmse_rel"] < 1e-2
    got = {item["name"] for item in fit["selected"]}
    assert {"1", "v^2", "v/r"} <= got
