# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Tests for free-constant scope normalization."""

import pytest

from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, normalize_free_const_scope


def _dummy_spec(scope_map):
    us = UnitSystem(("L", "T"))
    return UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}),),
        y_dim=us.dim({"L": 1}),
        free_const_scope=scope_map,
    )


def test_normalize_free_const_scope_aliases():
    assert normalize_free_const_scope("local") == "experiment"
    assert normalize_free_const_scope("experiment") == "experiment"
    assert normalize_free_const_scope("global") == "class"
    assert normalize_free_const_scope("class") == "class"
    assert normalize_free_const_scope(None) == "experiment"


def test_units_spec_normalizes_scope_map_on_init():
    spec = _dummy_spec(
        {
            "k_local": "local",
            "k_global": "global",
            "k_explicit_e": "experiment",
            "k_explicit_c": "class",
        }
    )
    assert spec.free_const_scope["k_local"] == "experiment"
    assert spec.free_const_scope["k_global"] == "class"
    assert spec.free_const_scope["k_explicit_e"] == "experiment"
    assert spec.free_const_scope["k_explicit_c"] == "class"


def test_invalid_scope_raises():
    with pytest.raises(ValueError):
        normalize_free_const_scope("not_a_scope")
    with pytest.raises(ValueError):
        _dummy_spec({"k_bad": "not_a_scope"})
