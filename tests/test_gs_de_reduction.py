# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import numpy as np
import pytest


def _trajs_900(n: int = 400):
    xg = np.linspace(0.0, 3.0, n)
    trajs = [(xg, c / (1.0 + xg)) for c in (0.7, 1.3, 2.1)]
    u1s = [-c / (1.0 + xg) ** 2 for c in (0.7, 1.3, 2.1)]
    return trajs, u1s


def test_chart_dispatch_kinds_and_shear_rejection():
    from nestynet_sr.sr_gs.de_reduction import compile_canonical_chart

    assert compile_canonical_chart((0, 0, 0, 0, 0, 1)).kind == "u_flow"
    assert compile_canonical_chart((1, 1, 0, 0, 0, 0)).kind == "x_flow"
    assert compile_canonical_chart((0, 1, 0, -1, 0, 0)).kind == "mixed_xscale_utrans"
    assert compile_canonical_chart((1, 0, 0, 0, 0, 1)).kind == "xtrans_uflow"
    assert compile_canonical_chart((0, 1, 0, 0, 0, 2)).kind == "joint_scaling"
    with pytest.raises(NotImplementedError):
        compile_canonical_chart((0, 0, 1, 0, 0, 0))  # u d_x shear


def test_reduction_recovers_shifted_inverse_carrier():
    """900: u' = -u/(1+x) -> u d_u chart gives G(x) = -(1+x)^-1 exactly."""

    from nestynet_sr.sr_gs.de_reduction import (
        compile_canonical_chart,
        fit_univariate_families,
        reduce_trajectories,
        select_univariate_fit,
    )

    trajs, u1s = _trajs_900()
    chart = compile_canonical_chart((0, 0, 0, 0, 0, 1))
    reduced = reduce_trajectories(trajs, chart, u1_list=u1s)
    assert reduced["status"] == "reduced"
    fit = select_univariate_fit(fit_univariate_families(reduced["z"], reduced["g"]))
    assert fit is not None
    assert fit.family == "shifted_power"
    assert fit.params["p"] == pytest.approx(-1.0, abs=1e-6)
    assert fit.params["c"] == pytest.approx(1.0, abs=1e-6)
    assert fit.params["a"] == pytest.approx(-1.0, rel=1e-6)


def test_half_order_reduction_recovers_continuous_exponent():
    from nestynet_sr.sr_gs.de_reduction import (
        compile_canonical_chart,
        fit_univariate_families,
        reduce_trajectories,
        select_univariate_fit,
    )

    k = 0.6
    trajs, u1s = [], []
    for u0 in (1.5, 2.5, 4.0):
        x = np.linspace(0.0, 0.9 * 2 * np.sqrt(u0) / k, 300)
        u = (np.sqrt(u0) - 0.5 * k * x) ** 2
        trajs.append((x, u))
        u1s.append(-k * np.sqrt(u))
    chart = compile_canonical_chart((1, 0, 0, 0, 0, 0))  # d_x
    reduced = reduce_trajectories(trajs, chart, u1_list=u1s)
    fit = select_univariate_fit(fit_univariate_families(reduced["z"], reduced["g"]))
    assert fit is not None
    assert fit.family == "shifted_power"
    assert fit.params["p"] == pytest.approx(0.5, abs=1e-6)
    assert fit.params["a"] == pytest.approx(-k, rel=1e-6)


def test_ensemble_generator_discovery_no_equation_needed():
    from nestynet_sr.sr_gs.de_reduction import discover_ensemble_generators

    trajs, _ = _trajs_900()
    found = discover_ensemble_generators(trajs)
    rays = set()
    for gen in found:
        arr = np.asarray(gen["coefficients"])
        arr = arr / np.max(np.abs(arr))
        rays.add(tuple(np.round(arr, 3)))
    # u d_u supported for the linear equation
    assert (0.0, 0.0, 0.0, 0.0, 0.0, 1.0) in rays
    # shifted scaling (1+x) d_x = d_x + x d_x
    assert (1.0, 1.0, 0.0, 0.0, 0.0, 0.0) in rays


def test_full_pipeline_emits_pulled_back_rows_and_dedupes():
    from nestynet_sr.sr_gs.de_reduction import symmetry_reduction_proposals

    trajs, u1s = _trajs_900()
    result = symmetry_reduction_proposals(trajs, u1_list=u1s)
    assert result["status"] == "ok"
    assert result["proposals"], "expected at least one pulled-back proposal"
    reprs = [repr(t) for t, _s, _f in result["library_rows"]]
    assert any("(1 + x0) ** -1" in r and "u" in r for r in reprs)
    # near-duplicate rows from equivalent charts are deduplicated
    assert len(result["library_rows"]) == 1


def test_reduction_rows_reach_de_library_builder():
    from nestynet_sr.sr_de.de_search import DESearchConfig, build_de_library_terms_with_sources
    from nestynet_sr.sr_gs.de_reduction import symmetry_reduction_proposals

    trajs, u1s = _trajs_900()
    result = symmetry_reduction_proposals(trajs, u1_list=u1s)
    cfg = DESearchConfig(x_axis=0)
    cfg.gs_enable = True
    cfg.gs_de_reduction_rows = result["library_rows"]
    terms, sources = build_de_library_terms_with_sources(cfg, order=1)
    assert "gs_de_reduction" in set(sources)
