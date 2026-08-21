from types import SimpleNamespace

import pytest

from nestynet_sr.sr_core.bridges import DU, Mul, Pow, U, Var
from nestynet_sr.sr_de.de_search import DESearchConfig, build_de_library_terms_with_sources
from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.de_determining import RecoveredDEGenerator
from nestynet_sr.sr_gs.de_invariants import (
    compile_de_invariant_library,
    de_invariant_library_report,
    de_invariant_library_rows,
)
from nestynet_sr.sr_gs.jet_bundle import JetSpaceSpec


def _gen(name, family="scaling", multiplier=0.0):
    return RecoveredDEGenerator(
        name=name,
        family=family,
        coefficients=(),
        multiplier=multiplier,
        on_shell_residual_rel=1.0e-12,
        off_shell_relative_residual_rel=1.0e-12,
        accepted=True,
        source="test",
    )


def test_compile_de_invariant_library_emits_rows_with_generator_provenance():
    rows = compile_de_invariant_library(
        jet_space=JetSpaceSpec(independent=("x",), dependent=("u",), max_order=1),
        generators=[_gen("u_d_u", multiplier=1.0)],
        order=1,
    )

    terms = {repr(row.term) for row in rows}
    assert repr(Mul(DU(0), Pow(U(), -1))) in terms
    assert repr(Mul(Var(0), Mul(DU(0), Pow(U(), -1)))) in terms
    assert all(row.source == "gs_de_differential_invariant" for row in rows)
    assert all(row.generator_name == "u_d_u" for row in rows)
    assert rows[0].provenance["multiplier"] == 1.0


def test_autonomous_generator_produces_coordinate_and_derivative_invariants():
    rows = compile_de_invariant_library(
        jet_space=JetSpaceSpec(independent=("x",), dependent=("u",), max_order=2),
        generators=[_gen("d_x", family="translation")],
        order=2,
    )

    terms = {repr(row.term) for row in rows}
    assert repr(U()) in terms
    assert repr(DU(0)) in terms
    assert repr(Pow(DU(0), 2)) in terms
    assert any(row.family == "autonomous_derivative" for row in rows)


def test_de_search_uses_differential_invariant_rows_only_when_enabled():
    disabled = DESearchConfig(gs_enable=True, order_candidates=(1,), include_x=False, include_u=False)
    _terms_disabled, sources_disabled = build_de_library_terms_with_sources(disabled, order=1)
    assert "gs_de_differential_invariant" not in sources_disabled

    enabled = DESearchConfig(
        gs_enable=True,
        gs_de_invariant_library=True,
        order_candidates=(1,),
        include_x=False,
        include_u=False,
        gs_de_invariant_seed_generators=("u_d_u",),
    )
    terms_enabled, sources_enabled = build_de_library_terms_with_sources(enabled, order=1)

    assert "gs_de_differential_invariant" in sources_enabled
    assert repr(Mul(DU(0), Pow(U(), -1))) in {repr(term) for term in terms_enabled}


def test_gs_all_upgrades_enables_invariant_library_rows():
    cfg = GeneralizedSymmetryConfig(enabled=True, de_all_upgrades=True)

    rows = de_invariant_library_rows(cfg, order=1)

    assert rows
    assert any(source == "gs_de_differential_invariant" for _term, source, _family in rows)


def test_gs_config_seed_and_limit_controls_invariant_rows():
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        de_invariant_library=True,
        de_invariant_seed_generators=("u_d_u",),
        de_invariant_max_terms=1,
    )

    rows = de_invariant_library_rows(cfg, order=1)

    assert len(rows) == 1
    assert repr(rows[0][0]) == repr(Mul(DU(0), Pow(U(), -1)))


def test_invariant_library_report_identifies_symmetry_sources():
    cfg = SimpleNamespace(gs_enable=True, gs_de_invariant_library=True, x_axis=0)
    report = de_invariant_library_report(
        cfg,
        order=2,
        generators=[_gen("x_d_x", family="scaling")],
        jet_space=JetSpaceSpec(independent=("x",), dependent=("u",), max_order=2),
    )

    assert report["enabled"]
    assert report["scalar_ode_only"]
    assert any(row["generator_name"] == "x_d_x" for row in report["rows"])
    assert any(row["family"] == "domain_scaling_derivative" for row in report["rows"])


def test_invariant_library_does_not_claim_pde_support():
    cfg = SimpleNamespace(gs_enable=True, gs_de_invariant_library=True, x_axis=0)
    with pytest.raises(NotImplementedError, match="vector/PDE prolongation"):
        de_invariant_library_rows(
            cfg,
            order=1,
            jet_space=JetSpaceSpec(independent=("t", "x"), dependent=("u",), max_order=1),
        )
