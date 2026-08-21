# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import math

import numpy as np
import pytest


def test_anchor_elimination_rejects_nonlinear_or_nonmonic_highest_jet():
    from nestynet_sr.sr_gs.de_certificates import build_certificate_samples

    x = np.linspace(0.1, 1.0, 32)
    u = 0.4 + x

    def nonlinear_anchor(env):
        return env["u_x"] + env["u_x"] ** 3 - env["u"]

    def nonmonic_anchor(env):
        return 2.0 * env["u_x"] - env["u"]

    for residual in (nonlinear_anchor, nonmonic_anchor):
        with pytest.raises(ValueError, match="affine and monic"):
            build_certificate_samples(residual=residual, x=x, u=u, order=1)


def test_named_term_certificate_recovers_linear_ode_symmetries():
    from nestynet_sr.sr_gs.de_certificates import certify_scalar_ode_candidate

    rng = np.random.default_rng(11)
    x = rng.uniform(0.1, 2.0, 256)
    u = rng.uniform(0.4, 2.0, 256)
    result = certify_scalar_ode_candidate(x=x, u=u, coeffs=[-1.0], term_names=["u"])
    names = {gen.name for gen in result.generators}
    assert result.status == "recovered"
    assert "u_d_u" in names  # linear in u
    assert "d_x" in names  # autonomous


def test_ast_term_certificate_matches_named_path():
    from nestynet_sr.sr_core.bridges import U
    from nestynet_sr.sr_gs.de_certificates import certify_scalar_ode_candidate

    rng = np.random.default_rng(12)
    x = rng.uniform(0.1, 2.0, 200)
    u = rng.uniform(0.4, 2.0, 200)
    named = certify_scalar_ode_candidate(x=x, u=u, coeffs=[-1.0], term_names=["u"])
    ast = certify_scalar_ode_candidate(x=x, u=u, coeffs=[-1.0], term_asts=[U()])
    assert {g.name for g in named.generators} == {g.name for g in ast.generators}


def test_nullspace_combination_discriminates_exp_alias():
    """u_x + exp(u) admits the mixed generator -x d_x + d_u; a cubic alias does not."""

    from nestynet_sr.sr_gs.de_certificates import certify_scalar_ode_candidate

    rng = np.random.default_rng(13)
    x = rng.uniform(0.1, 2.0, 300)
    u = rng.uniform(-1.5, 0.5, 300)

    truth = certify_scalar_ode_candidate(x=x, u=u, coeffs=[1.0], term_names=["exp(u)"])
    truth_names = {g.name for g in truth.generators}
    assert "d_x" in truth_names
    assert any(g.family == "nullspace_combination" for g in truth.generators)
    assert truth.determining_nullity >= 2

    alias = certify_scalar_ode_candidate(
        x=x, u=u,
        coeffs=[0.93, 1.1, 0.61, 0.13],
        term_names=["1", "u", "u^2", "u^3"],
    )
    alias_names = {g.name for g in alias.generators}
    assert alias_names == {"d_x"}
    assert alias.determining_nullity == 1


def test_shifted_scaling_combination_recovered_for_shifted_inverse():
    """u_x + u/(1+x) admits (1+x) d_x, i.e. the combination d_x + x_d_x."""

    from nestynet_sr.sr_gs.de_certificates import certify_scalar_ode_candidate

    rng = np.random.default_rng(14)
    x = rng.uniform(0.1, 2.0, 300)
    u = rng.uniform(0.4, 2.0, 300)
    result = certify_scalar_ode_candidate(x=x, u=u, coeffs=[1.0], term_names=["u/(1+x0)"])
    names = {g.name for g in result.generators}
    assert "u_d_u" in names
    combos = [g for g in result.generators if g.family == "nullspace_combination"]
    assert combos, "expected a nullspace combination generator"
    coeffs = np.asarray(combos[0].coefficients)
    # (1+x) d_x direction: equal a0 and a1 components, no u components
    assert coeffs[0] == pytest.approx(coeffs[1], rel=1.0e-6)
    assert float(np.max(np.abs(coeffs[2:]))) <= 1.0e-6


def test_flow_test_supports_true_generator_and_refutes_false_one():
    from nestynet_sr.sr_gs.de_certificates import generator_ensemble_support

    xg = np.linspace(0.0, 3.0, 400)
    trajectories = [(xg, c / (1.0 + xg)) for c in (0.7, 1.3, 2.1)]

    supported = generator_ensemble_support(trajectories, (0, 0, 0, 0, 0, 1))  # u d_u
    refuted = generator_ensemble_support(trajectories, (0, 0, 0, 0, 1, 0))  # x d_u
    assert supported["status"] == "tested" and supported["supported"] is True
    assert refuted["status"] == "tested" and refuted["supported"] is False


def test_fitted_offset_needs_matching_tolerance():
    """A realistic fitted offset breaks exact symmetry; tolerance must absorb it."""

    from nestynet_sr.sr_gs.de_certificates import certify_scalar_ode_candidate

    rng = np.random.default_rng(15)
    x = rng.uniform(0.1, 2.0, 256)
    u = rng.uniform(0.4, 2.0, 256)
    names = ["1", "u/(1+x0)"]
    coeffs = [6.58e-4, 0.9987]
    strict = certify_scalar_ode_candidate(x=x, u=u, coeffs=coeffs, term_names=names)
    loose = certify_scalar_ode_candidate(
        x=x, u=u, coeffs=coeffs, term_names=names,
        on_shell_tol=3.0e-3, off_shell_tol=3.0e-3,
    )
    # Strictly, the offset breaks the clean scaling symmetry (the certificate
    # may still find exact affine combinations of the *fitted* equation).
    assert "u_d_u" not in {g.name for g in strict.generators}
    loose_names = {g.name for g in loose.generators}
    assert "u_d_u" in loose_names


def test_de_bridge_passes_discovered_generators_through():
    from types import SimpleNamespace

    from nestynet_sr.sr_gs.de_bridge import generalized_symmetry_de_term_rows
    from nestynet_sr.sr_gs.de_determining import RecoveredDEGenerator

    cfg = SimpleNamespace(
        gs_enable=True,
        gs_de_invariant_library=True,
        x_axis=0,
    )
    discovered = [
        RecoveredDEGenerator(
            name="u_d_u",
            family="scaling",
            coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            multiplier=1.0,
            on_shell_residual_rel=0.0,
            off_shell_relative_residual_rel=0.0,
            accepted=True,
        )
    ]
    rows = generalized_symmetry_de_term_rows(cfg, order=1, generators=discovered)
    sources = {source for _term, source, _family in rows}
    assert "gs_de_differential_invariant" in sources
    reprs = " | ".join(repr(term) for term, _s, _f in rows)
    assert "u" in reprs  # the u_x/u log-derivative row family is compiled


def test_run_de_certificate_report_is_json_serializable():
    import json

    from nestynet_sr.sr_gs.de_certificates import certify_scalar_ode_candidate

    rng = np.random.default_rng(16)
    x = rng.uniform(0.1, 2.0, 128)
    u = rng.uniform(0.4, 2.0, 128)
    report = certify_scalar_ode_candidate(x=x, u=u, coeffs=[-1.0], term_names=["u"]).to_report()
    text = json.dumps(report)
    assert "generators" in report and isinstance(text, str)
    assert all(math.isfinite(g["multiplier"]) for g in report["generators"])
