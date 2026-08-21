# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import j0, j1


def _linear_ensemble(p, q, amplitudes=(0.6, 1.0, 1.5, 2.1), x_end=1.6, n=500):
    def rhs(u, v):
        return -p * v - q * u

    trajs, u1s, u2s = [], [], []
    x = np.linspace(0.02, x_end, n)
    for a in amplitudes:
        sol = solve_ivp(lambda t, s: [s[1], rhs(s[0], s[1])], (0.02, x_end), [a, 0.0],
                        t_eval=x, rtol=1e-11, atol=1e-13)
        u, u1 = sol.y[0], sol.y[1]
        trajs.append((x, u))
        u1s.append(u1)
        u2s.append(rhs(u, u1))
    return trajs, u1s, u2s


def _bessel_ensemble(x_end=2.4, n=500):
    trajs, u1s, u2s = [], [], []
    x = np.linspace(0.1, x_end, n)
    for a in (0.6, 1.0, 1.5, 2.1):
        u = a * j0(x)
        u1 = -a * j1(x)
        trajs.append((x, u))
        u1s.append(u1)
        u2s.append(-u1 / x - u)
    return trajs, u1s, u2s


def test_recognizer_three_regimes():
    from nestynet_sr.sr_gs.de_reduction import recognize_constant_coeff_linear_solution

    def sel(**terms):
        return [{"name": k, "coeff": v} for k, v in terms.items()]

    # SHO: dv/dr = -1.69 - v^2  (p=0, q=1.69) -> complex roots, omega=1.3
    sho = recognize_constant_coeff_linear_solution(sel(**{"1": -1.69, "v^2": -1.0}))
    assert sho["regime"] == "complex_roots"
    assert sho["roots"][0][1] == pytest.approx(1.3, rel=1e-4)

    # exponential: dv/dr = 1.44 - v^2 (p=0, q=-1.44) -> two real roots +-1.2
    exp = recognize_constant_coeff_linear_solution(sel(**{"1": 1.44, "v^2": -1.0}))
    assert exp["regime"] == "two_real_roots"
    assert sorted(exp["roots"]) == pytest.approx([-1.2, 1.2], rel=1e-4)

    # critically damped: dv/dr = -1 - 2v - v^2 (p=2, q=1) -> double root -1
    dbl = recognize_constant_coeff_linear_solution(sel(**{"1": -1.0, "v": -2.0, "v^2": -1.0}))
    assert dbl["regime"] == "double_root"
    assert dbl["roots"][0] == pytest.approx(-1.0, abs=1e-6)

    # r-dependent -> not a constant-coefficient reduction
    assert recognize_constant_coeff_linear_solution(sel(**{"1": -1.0, "v/r": -2.0, "v^2": -1.0})) is None
    # wrong leading term -> not the u_x/u reduction
    assert recognize_constant_coeff_linear_solution(sel(**{"1": -1.0, "v^2": -0.5})) is None


def test_reduced_equation_symmetry_autonomy_and_scale_invariance():
    from nestynet_sr.sr_gs.de_reduction import reduced_equation_symmetry

    # autonomous -> translation d/dr (original-level d/dx)
    auto = reduced_equation_symmetry([{"name": "1", "coeff": -1.7}, {"name": "v^2", "coeff": -1.0}])
    assert auto["kind"] == "translation_r"
    assert tuple(auto["generator"]) == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert tuple(auto["algebra_generator"]) == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # scale-invariant (v^2, v/r, 1/r^2) -> reduced r d/dr - v d/dv, original x d/dx
    scale = reduced_equation_symmetry(
        [{"name": "v^2", "coeff": -1.0}, {"name": "v/r", "coeff": -2.0}, {"name": "1/r^2", "coeff": 6.0}]
    )
    assert scale["kind"] == "scaling"
    assert tuple(scale["generator"]) == (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    assert tuple(scale["algebra_generator"]) == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    # mixed degree (a constant 1 breaks scale invariance and it is not autonomous)
    assert reduced_equation_symmetry(
        [{"name": "1", "coeff": -1.0}, {"name": "v^2", "coeff": -1.0}, {"name": "v/r", "coeff": -2.0}]
    ) is None


def test_equidimensional_recognizer_three_regimes():
    from nestynet_sr.sr_gs.de_reduction import recognize_equidimensional_solution

    def sel(**terms):
        return [{"name": k, "coeff": v} for k, v in terms.items()]

    # a=2, b=-6: indicial m^2 + (a-1)m + b = m^2 + m - 6 = 0 -> roots 2, -3
    real = recognize_equidimensional_solution(sel(**{"v^2": -1.0, "v/r": -2.0, "1/r^2": 6.0}))
    assert real["regime"] == "two_real_exponents"
    assert sorted(real["exponents"]) == pytest.approx([-3.0, 2.0], abs=1e-6)

    # a=1, b=1: m^2 + 1 = 0 -> +-i
    cplx = recognize_equidimensional_solution(sel(**{"v^2": -1.0, "v/r": -1.0, "1/r^2": -1.0}))
    assert cplx["regime"] == "complex_exponents"
    assert cplx["exponents"][0][1] == pytest.approx(1.0, rel=1e-4)

    # a=-3, b=4: m^2 - 4m + 4 = 0 -> double root 2
    dbl = recognize_equidimensional_solution(sel(**{"v^2": -1.0, "v/r": 3.0, "1/r^2": -4.0}))
    assert dbl["regime"] == "double_exponent"
    assert dbl["exponents"][0] == pytest.approx(2.0, abs=1e-6)

    # autonomous / constant-coefficient input -> not equidimensional
    assert recognize_equidimensional_solution(sel(**{"1": -1.7, "v^2": -1.0})) is None


def test_algebra_bracket_and_solvability():
    from nestynet_sr.sr_gs.de_reduction import _affine_bracket, _bracket_in_span

    u_du = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    d_x = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bracket = _affine_bracket(u_du, d_x)
    assert bracket == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # abelian
    assert _bracket_in_span(bracket, u_du, d_x) is True


def test_per_trajectory_order2_reduction():
    from nestynet_sr.sr_gs.de_reduction import compile_canonical_chart, reduce_trajectories_order2

    trajs, u1s, u2s = _linear_ensemble(0.0, 1.69)
    chart = compile_canonical_chart((0, 0, 0, 0, 0, 1))  # u d/du
    reduced = reduce_trajectories_order2(
        trajs, chart, u1_list=u1s, u2_list=u2s, return_per_trajectory=True
    )
    assert reduced["status"] == "reduced"
    per = reduced["per_trajectory"]
    assert len(per) == len(trajs)
    for r, v, g in per:
        assert r.shape == v.shape == g.shape and r.size >= 8


def test_cascade_fires_on_sho_with_closed_form():
    from nestynet_sr.sr_gs.de_reduction import solvable_cascade_reduction

    trajs, u1s, u2s = _linear_ensemble(0.0, 1.69)  # SHO omega=1.3
    rep = solvable_cascade_reduction(trajs, u1_list=u1s, u2_list=u2s)
    assert rep["cascade_fired"] is True
    best = rep["best"]
    assert tuple(best["V1"]) == (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # u d/du
    assert tuple(best["V2"]) == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # d/dr
    assert best["algebra_is_solvable"] is True
    cf = best["closed_form"]
    assert cf["regime"] == "complex_roots"
    assert cf["roots"][0][1] == pytest.approx(1.3, rel=1e-3)


def _cauchy_euler_real_ensemble(x0=0.3, x_end=3.0, n=600, amps=(0.6, 1.0, 1.5, 2.1)):
    # x^2 u'' + 2 x u' - 6 u = 0 -> u = A x^2 + B x^-3
    x = np.linspace(x0, x_end, n)
    trajs = [(x, c * (x**2 + 0.3 * x**-3)) for c in amps]
    u1s = [c * (2 * x - 0.9 * x**-4) for c in amps]
    u2s = [c * (2 + 3.6 * x**-5) for c in amps]
    return trajs, u1s, u2s


def test_cascade_fires_on_cauchy_euler_with_power_law_solution():
    from nestynet_sr.sr_gs.de_reduction import solvable_cascade_reduction

    trajs, u1s, u2s = _cauchy_euler_real_ensemble()
    rep = solvable_cascade_reduction(trajs, u1_list=u1s, u2_list=u2s)
    assert rep["cascade_fired"] is True
    best = rep["best"]
    assert tuple(best["V1"]) == (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # u d/du
    assert tuple(best["V2"]) == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)  # x d/dx (equidimensional)
    assert tuple(best["V2_reduced_level"]) == (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)  # r d/dr - v d/dv
    assert best["V2_kind"] == "scaling"
    assert best["algebra_is_solvable"] is True
    cf = best["closed_form"]
    assert cf["regime"] == "two_real_exponents"
    assert sorted(cf["exponents"]) == pytest.approx([-3.0, 2.0], abs=1e-3)


def test_cascade_declines_on_bessel():
    from nestynet_sr.sr_gs.de_reduction import solvable_cascade_reduction

    trajs, u1s, u2s = _bessel_ensemble()
    rep = solvable_cascade_reduction(trajs, u1_list=u1s, u2_list=u2s)
    assert rep["cascade_fired"] is False
    # the order-2 reduction still ran; it just retains explicit r-dependence
    assert rep["attempts"], "expected a level-1 reduction attempt"
    assert "r-dependence" in rep["attempts"][0]["reason"]
