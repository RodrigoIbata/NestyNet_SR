# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from nestynet_sr.sr_search.factorized_search.explorer import (
    _dedup_new,
    _mapping_equiv_root,
    node_depth,
    node_size,
    node_str,
)


def test_mapping_equiv_root_collapses_repeated_linear_same_core():
    t = ("add", ("add", ("var", 0), ("var", 0)), ("var", 0))
    c = _mapping_equiv_root(t)
    assert c == ("var", 0)


def test_mapping_equiv_root_collapses_sign_flipped_linear_same_core():
    t = ("sub", ("var", 0), ("add", ("var", 0), ("var", 0)))
    c = _mapping_equiv_root(t)
    assert c == ("var", 0)


def test_dedup_new_uses_mapping_equivalent_root_key():
    a = ("add", ("add", ("var", 0), ("var", 0)), ("var", 0))
    b = ("sub", ("var", 0), ("add", ("var", 0), ("var", 0)))
    seen = set()
    out = _dedup_new([a, b], seen)
    assert len(out) == 1
    assert node_str(_mapping_equiv_root(out[0])) == "x0"


def test_node_size_and_depth_treat_hparam_as_leaf():
    hp = ("hparam", 0)
    assert node_size(hp) == 1
    assert node_depth(hp) == 1
    tree = ("sin", ("mul", hp, ("var", 0)))
    assert node_size(tree) >= 3
    assert node_depth(tree) >= 2
