# SPDX-License-Identifier: MPL-2.0

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import Add, DU, Mul, Pow, U, Var
from nestynet_sr.sr_de.de_search import DESearchConfig, build_de_library_terms_with_sources
from nestynet_sr.sr_gs.de_upgrades import (
    ast_discrete_signature,
    contact_jet_de_rows,
    discover_determining_equation_generators_from_jets,
    discrete_symmetry_de_rows,
    noether_variational_de_rows,
    radial_reduction_de_rows,
    score_discrete_symmetry_terms,
    score_noether_candidate,
    symmetry_upgrade_de_term_rows,
    weighted_scaling_de_rows,
)


def _cfg(**kwargs):
    base = dict(
        gs_enable=True,
        gs_de_upgrade_max_terms=128,
        x_axis=0,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_upgrade_1_determining_equation_search_recovers_output_scaling():
    x = torch.linspace(0.1, 3.0, 96, dtype=torch.float64).unsqueeze(1)
    u = torch.sin(x)
    u1 = torch.cos(x)
    u2 = -torch.sin(x)

    meta = discover_determining_equation_generators_from_jets(
        order=2,
        x=x,
        u=u,
        u1=u1,
        u2=u2,
        term_asts=[U()],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        max_degree=1,
        tol=1.0e-8,
    )

    assert meta["status"] == "scored"
    assert "det_eta_u" in set(meta["accepted_generator_names"])


def test_upgrade_2_contact_rows_include_even_velocity_state_terms():
    rows = contact_jet_de_rows(_cfg(gs_de_contact_templates=True), order=2)
    terms = {repr(term) for term, _source, _family in rows}

    assert repr(Mul(U(), Pow(DU(0), 2))) in terms
    assert any(source == "de_prior_velocity_monomial" for _term, source, _family in rows)


def test_upgrade_3_noether_rows_and_score_prefer_autonomous_even_velocity():
    rows = noether_variational_de_rows(_cfg(gs_de_noether_templates=True), order=2)
    terms = [term for term, _source, _family in rows]
    score = score_noether_candidate([U(), Mul(U(), Pow(DU(0), 2)), Mul(Var(0), U())])

    assert repr(Pow(U(), 3)) in {repr(term) for term in terms}
    assert score["accepted_terms"] == 2


def test_upgrade_4_discrete_symmetry_signature_and_rows():
    term = Mul(U(), Pow(DU(0), 2))
    sig = ast_discrete_signature(term)
    score = score_discrete_symmetry_terms([term, Mul(Pow(U(), 2), DU(0))])
    rows = discrete_symmetry_de_rows(_cfg(gs_de_discrete_symmetry_templates=True), order=2)

    assert sig == {"x": 0, "u": 1, "du": 2}
    assert score["accepted_terms"] == 1
    assert repr(term) in {repr(row[0]) for row in rows}


def test_discrete_symmetry_signature_rejects_mixed_and_fractional_parity():
    mixed = Add(U(), Pow(U(), 2))
    fractional = Pow(U(), 0.5)
    score = score_discrete_symmetry_terms([mixed, fractional])

    assert ast_discrete_signature(mixed)["status"] == "mixed"
    assert ast_discrete_signature(fractional)["status"] == "unknown_fractional_power"
    assert score["accepted_terms"] == 0
    assert {row["signature_status"] for row in score["terms"]} == {
        "mixed",
        "unknown_fractional_power",
    }


def test_upgrade_5_weighted_scaling_rows_cover_radial_and_velocity_realizations():
    rows = weighted_scaling_de_rows(
        _cfg(
            gs_de_weighted_scaling_templates=True,
            gs_de_weighted_max_abs_x_power=2,
            gs_de_weighted_max_u_power=3,
            gs_de_weighted_max_du_power=2,
        ),
        order=2,
    )
    terms = {repr(term) for term, _source, _family in rows}

    assert repr(Mul(Pow(Var(0), -2), U())) in terms
    assert repr(Mul(U(), Pow(DU(0), 2))) in terms


def test_upgrade_6_radial_reduction_rows_generalize_baseline_inverse_x_terms():
    rows = radial_reduction_de_rows(_cfg(gs_de_radial_reduction_templates=True), order=2)
    terms = {repr(term) for term, _source, _family in rows}

    assert repr(Mul(Pow(Var(0), -1), DU(0))) in terms
    assert repr(Mul(Pow(Var(0), -2), U())) in terms
    assert any(source == "de_prior_radial_shape" for _term, source, _family in rows)


def test_upgrade_7_unit_torus_remains_separate_and_upgrade_bridge_is_source_aware():
    cfg = _cfg(gs_de_all_upgrades=True)
    rows = symmetry_upgrade_de_term_rows(cfg, order=2)
    sources = {source for _term, source, _family in rows}

    assert {
        "de_prior_velocity_monomial",
        "de_prior_autonomous_even_velocity",
        "de_prior_parity",
        "de_prior_assumed_weight_profile",
        "de_prior_radial_shape",
    } <= sources


def test_de_search_smoke_emits_upgrade_sources_when_enabled():
    cfg = DESearchConfig(
        gs_enable=True,
        gs_de_all_upgrades=True,
        gs_de_templates=False,
        gs_de_upgrade_max_terms=128,
        order_candidates=(2,),
        max_u_power=1,
        max_x_power=1,
    )
    _terms, sources = build_de_library_terms_with_sources(cfg, order=2)

    assert "de_prior_velocity_monomial" in sources
    assert "de_prior_assumed_weight_profile" in sources
    assert "de_prior_radial_shape" in sources
