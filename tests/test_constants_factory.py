# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import pytest

from nestynet_sr.sr_core.constants import (
    make_unit_aware_scalar_atom,
    scalar_constant_variants,
    unit_aware_scalar_choice,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec


def test_unit_aware_scalar_choice_dimless_prefers_scale():
    us = UnitSystem(base=("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
    )
    choice = unit_aware_scalar_choice(us.dimless(), spec)
    assert choice == {"kind": "scale"}


def test_unit_aware_scalar_choice_unitful_picks_declared_free_const():
    us = UnitSystem(base=("L", "T"))
    req = us.dim([1, -2])
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        free_const_dims={"g": req},
        free_const_scope={"g": "global"},
    )
    choice = unit_aware_scalar_choice(req, spec, prefer_scope="experiment")
    assert choice is not None
    assert choice["kind"] == "free_const"
    assert choice["name"] == "g"
    assert choice["scope"] == "class"


def test_make_unit_aware_scalar_atom_returns_declared_free_const():
    us = UnitSystem(base=("L", "T"))
    req = us.dim([1, -2])
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        free_const_dims={"g": req},
        free_const_scope={"g": "experiment"},
    )
    atom = make_unit_aware_scalar_atom(req, spec, base_tag="k_tmp", init=9.81, strict=True)
    assert str(atom.kind).lower() == "free_const"
    assert atom.tag == "g"
    assert str(getattr(atom, "scope", "")) == "experiment"
    assert abs(float(atom.kwargs.get("init")) - 9.81) < 1.0e-12


def test_make_unit_aware_scalar_atom_strict_raises_when_missing():
    us = UnitSystem(base=("L", "T"))
    req = us.dim([1, -2])
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
    )
    with pytest.raises(ValueError):
        make_unit_aware_scalar_atom(req, spec, base_tag="k_tmp", strict=True)


def test_scalar_constant_variants_respects_fixed_const_mode():
    us = UnitSystem(base=("L", "T"))
    fixed_dims = {"pi": us.dimless()}
    fixed_vals = {"pi": 3.14159}

    spec_strict = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        fixed_const_dims=fixed_dims,
        fixed_const_values=fixed_vals,
        fixed_const_mode="strict",
    )
    spec_min = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        fixed_const_dims=fixed_dims,
        fixed_const_values=fixed_vals,
        fixed_const_mode="minimal",
    )
    spec_off = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        fixed_const_dims=fixed_dims,
        fixed_const_values=fixed_vals,
        fixed_const_mode="off",
    )

    v_strict = scalar_constant_variants(spec_strict, base_tag="k", scale_init=1.0)
    v_min = scalar_constant_variants(spec_min, base_tag="k", scale_init=1.0)
    v_off = scalar_constant_variants(spec_off, base_tag="k", scale_init=1.0)

    assert len(v_strict) == 2
    assert len(v_min) == 1
    assert len(v_off) == 1
