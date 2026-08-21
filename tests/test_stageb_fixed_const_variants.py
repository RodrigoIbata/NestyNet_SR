# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_core.constants import (
    build_scalar_atom_from_variant,
    matching_fixed_const_specs,
    scalar_constant_variants,
)


def test_matching_fixed_const_specs_filters_by_dim_and_sorts():
    us = UnitSystem(base=("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        fixed_const_dims={
            "z": us.dimless(),
            "a": us.dimless(),
            "g": us.dim([1, -2]),
            "bad": us.dimless(),
        },
        fixed_const_values={
            "z": 2.0,
            "a": 1.0,
            "g": 9.81,
            "bad": float("nan"),
        },
    )
    dimless = matching_fixed_const_specs(spec)
    assert dimless == [("a", 1.0), ("z", 2.0)]

    accel = matching_fixed_const_specs(spec, required_dim=us.dim([1, -2]))
    assert accel == [("g", 9.81)]


def test_scalar_constant_variants_include_scale_then_fixed():
    us = UnitSystem(base=("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        fixed_const_dims={"pi": us.dimless(), "tau": us.dimless()},
        fixed_const_values={"tau": 6.28318, "pi": 3.14159},
    )
    variants = scalar_constant_variants(spec, base_tag="k", scale_init=0.25, max_fixed=4)
    assert len(variants) == 3

    assert variants[0]["mode"] == "scale"
    assert variants[0]["tag"] == "k"
    assert abs(float(variants[0]["value"]) - 0.25) < 1.0e-12

    fixed_names = [v["name"] for v in variants[1:]]
    assert fixed_names == ["pi", "tau"]
    for v in variants[1:]:
        assert v["mode"] == "fixed"
        assert str(v["tag"]).startswith("k__fx_")
        assert str(v["label_suffix"]).startswith("[fixed:")


def test_build_scalar_atom_from_variant_builds_scale_and_fixed_const():
    scale_atom = build_scalar_atom_from_variant(
        {"mode": "scale", "name": "k", "tag": "k", "value": 0.5}
    )
    assert str(scale_atom.kind).lower() == "scale"
    assert scale_atom.tag == "k"
    assert abs(float(scale_atom.kwargs["init"]) - 0.5) < 1.0e-12

    fixed_atom = build_scalar_atom_from_variant(
        {"mode": "fixed", "name": "g", "tag": "c__fx_g", "value": 9.81}
    )
    assert str(fixed_atom.kind).lower() == "fixed_const"
    assert fixed_atom.tag == "c__fx_g"
    assert abs(float(fixed_atom.kwargs["value"]) - 9.81) < 1.0e-12
