# SPDX-License-Identifier: MPL-2.0

from fractions import Fraction
from types import SimpleNamespace

from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_gs.de_bridge import generalized_symmetry_de_term_rows, hard_tail_de_term_rows
from nestynet_sr.sr_gs.unit_torus import (
    dimensions_from_units_spec,
    enumerate_nullspace_exponents,
    enumerate_prefactor_exponents,
    unit_torus_generators_from_units_spec,
)


def test_unit_torus_generators_follow_dimension_columns():
    us = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([1, 0]), us.dim([0, 1])),
        y_dim=us.dim([1, -1]),
    )

    gens = unit_torus_generators_from_units_spec(spec)

    assert [g.basis_name for g in gens] == ["L", "T"]
    assert gens[0].x_weights == (Fraction(1), Fraction(0))
    assert gens[0].y_weight == Fraction(1)
    assert gens[1].x_weights == (Fraction(0), Fraction(1))
    assert gens[1].y_weight == Fraction(-1)


def test_buckingham_pi_enumeration_finds_dimensionless_monomial():
    # x0=L, x1=T, x2=L/T, so x0/(x1*x2) is dimensionless.
    exps = enumerate_nullspace_exponents(
        ((1, 0), (0, 1), (1, -1)),
        max_exponent=2,
        max_l1=4,
        max_proposals=10,
        max_basis=3,
    )

    assert (Fraction(1), Fraction(-1), Fraction(-1)) in exps


def test_prefactor_enumeration_finds_de_anchor_dimension():
    # variables are t=T, u=L, du=L/T; target is u_tt=L/T^2.
    exps = enumerate_prefactor_exponents(
        ((0, 1), (1, 0), (1, -1)),
        (1, -2),
        max_exponent=3,
        max_l1=4,
        max_proposals=10,
        max_basis=3,
    )

    assert (Fraction(-2), Fraction(1), Fraction(0)) in exps
    assert (Fraction(-1), Fraction(0), Fraction(1)) in exps


def test_de_bridge_imports_reporting_and_emits_source_tagged_template_rows():
    cfg = SimpleNamespace(
        gs_enable=True,
        gs_mode="auto",
        gs_policy="replace-shadowed",
        de_hard_tail_templates=True,
        de_hard_tail_radial_templates=True,
        de_hard_tail_velocity_templates=True,
        gs_unit_torus=False,
        x_axis=0,
    )

    rows = hard_tail_de_term_rows(cfg, order=2)

    assert rows
    assert any(source == "de_prior_hard_tail" for _term, source, _family in rows)
    assert generalized_symmetry_de_term_rows(cfg, order=2) == []


def test_dimensions_from_units_spec_accepts_y_phi_without_eager_y_fallback():
    spec = SimpleNamespace(
        x_dims=((1, 0), (0, 1)),
        y_phi_dim=(1, -1),
        unit_system=SimpleNamespace(base=("L", "T")),
    )

    x_dims, y_dim, base = dimensions_from_units_spec(spec)

    assert x_dims == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    assert y_dim == (Fraction(1), Fraction(-1))
    assert base == ("L", "T")
