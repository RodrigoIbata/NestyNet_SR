# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import torch

from nestynet_sr.sr_core.bridges import AtomNode, FreeConst, Var, collect_all_atoms
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search.factorized_search.bridge import (
    embed_mapping_in_ast,
    mapping_embedding_roundtrip,
)
from nestynet_sr.sr_search.factorized_search.adapters.nestynet.api import fraction_to_dims
from nestynet_sr.sr_search.factorized_search.adapters.nestynet.stageb_prep import (
    _append_declared_constant_columns,
    _build_input_exprs_with_declared_constants,
    _declared_constant_specs_for_explorer,
    prepare_stageb_explorer_inputs,
)
from nestynet_sr.sr_search.factorized_search.adapters.nestynet.stageb_runner import _node_has_free_const
from nestynet_sr.sr_search.factorized_search.adapters.nestynet.wrapper_utils import (
    _normalize_outer_wrapper_name,
    _outer_wrapper_forward,
    _outer_wrapper_inverse_ast,
    _outer_wrapper_transformed_y_dims,
)


def test_declared_constant_specs_include_free_and_fixed():
    us = UnitSystem(base=("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        free_const_dims={"c1": us.dim([1, -1])},
        free_const_scope={"c1": "global"},
        fixed_const_dims={
            "g": us.dim([1, -2]),
            "missing": us.dim([0, 0]),
        },
        fixed_const_values={"g": 9.81},
    )

    declared = _declared_constant_specs_for_explorer(spec)
    assert [d["name"] for d in declared] == ["c1", "g"]
    assert [d["kind"] for d in declared] == ["free", "fixed"]
    assert declared[0]["scope"] == "class"
    assert abs(float(declared[1]["value"]) - 9.81) < 1.0e-12


def test_declared_constant_specs_skip_fixed_when_name_collides_with_free():
    us = UnitSystem(base=("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        free_const_dims={"c1": us.dim([1, -1])},
        fixed_const_dims={"c1": us.dim([1, -1])},
        fixed_const_values={"c1": 2.0},
    )
    declared = _declared_constant_specs_for_explorer(spec)
    assert len(declared) == 1
    assert declared[0]["name"] == "c1"
    assert declared[0]["kind"] == "free"


def test_append_declared_constant_columns_uses_values():
    x = torch.zeros(4, 2, dtype=torch.float64)
    declared = [
        {"name": "c1", "kind": "free", "value": 1.0, "scope": "experiment"},
        {"name": "g", "kind": "fixed", "value": 9.81, "scope": "fixed"},
    ]
    out = _append_declared_constant_columns(x, declared)
    assert tuple(out.shape) == (4, 4)
    assert torch.allclose(out[:, 2], torch.ones(4, dtype=out.dtype))
    assert torch.allclose(out[:, 3], torch.full((4,), 9.81, dtype=out.dtype))


def test_build_input_exprs_with_declared_constants_builds_free_and_fixed_leaves():
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0")
    declared = [
        {"name": "c1", "kind": "free", "value": 1.0, "scope": "class"},
        {"name": "g", "kind": "fixed", "value": 9.81, "scope": "fixed"},
    ]
    exprs = _build_input_exprs_with_declared_constants(target, declared)

    # 2 original vars + 2 declared constants.
    assert len(exprs) == 4
    free_leaf = exprs[2]
    fixed_leaf = exprs[3]

    assert isinstance(free_leaf, AtomNode)
    assert str(free_leaf.kind).lower() == "free_const"
    assert str(getattr(free_leaf, "scope", "")) == "class"
    assert free_leaf.tag == "c1"

    assert isinstance(fixed_leaf, AtomNode)
    assert str(fixed_leaf.kind).lower() == "fixed_const"
    assert fixed_leaf.tag == "g"
    assert abs(float(fixed_leaf.kwargs.get("value")) - 9.81) < 1.0e-12


def test_prepare_stageb_explorer_inputs_builds_dims_consts_and_inputs():
    us = UnitSystem(base=("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([0, 1]),),
        y_dim=us.dim([1, 0]),
        free_const_dims={"c1": us.dim([1, -1])},
        fixed_const_dims={"g": us.dim([1, -2])},
        fixed_const_values={"g": 9.81},
    )
    target = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")

    prep = prepare_stageb_explorer_inputs(root=target, target=target, units_spec=spec)

    assert [d["name"] for d in prep.declared_consts] == ["c1", "g"]
    assert prep.var_dims == [
        fraction_to_dims(spec.x_dims[0]),
        fraction_to_dims(spec.free_const_dims["c1"]),
        fraction_to_dims(spec.fixed_const_dims["g"]),
    ]
    assert prep.y_dims == fraction_to_dims(spec.y_dim)
    assert len(prep.input_exprs) == 3


def _has_z_alpha_scale(node) -> bool:
    for atom in collect_all_atoms(node):
        if str(getattr(atom, "kind", "")).lower() != "scale":
            continue
        if str(getattr(atom, "tag", "")).endswith("__z_alpha"):
            return True
    return False


def test_alpha_guard_keeps_z_alpha_trainable_when_unused_free_const_not_in_replacement():
    mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
    input_exprs = [Var(0), FreeConst("c_unused", tag="c_unused", init=1.0, scope="experiment")]
    replacement = embed_mapping_in_ast(
        Var(0), mapping, input_exprs,
        trainable_dimless=True,
        tag_prefix="phase6_keep",
        z_affine=True,
        z_alpha_init=1.0,
        z_beta_init=None,
        z_train_alpha=True,
        sin_arg_mode="wu",
    )
    assert replacement is not None
    assert not _node_has_free_const(replacement)
    assert _has_z_alpha_scale(replacement)


def test_alpha_guard_freezes_z_alpha_when_replacement_contains_free_const():
    mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}
    input_exprs = [Var(0), FreeConst("c_used", tag="c_used", init=1.0, scope="experiment")]
    replacement = embed_mapping_in_ast(
        Var(1), mapping, input_exprs,
        trainable_dimless=True,
        tag_prefix="phase6_freeze",
        z_affine=True,
        z_alpha_init=1.0,
        z_beta_init=None,
        z_train_alpha=True,
        sin_arg_mode="wu",
    )
    assert replacement is not None
    assert _node_has_free_const(replacement)
    replacement_frozen = embed_mapping_in_ast(
        Var(1), mapping, input_exprs,
        trainable_dimless=True,
        tag_prefix="phase6_freeze",
        z_affine=True,
        z_alpha_init=1.0,
        z_beta_init=None,
        z_train_alpha=False,
        sin_arg_mode="wu",
    )
    assert replacement_frozen is not None
    assert not _has_z_alpha_scale(replacement_frozen)


def test_embed_mapping_preserves_linear_head_for_mapping_families():
    x = torch.tensor(
        [
            [0.7, -0.4],
            [1.1, 0.2],
            [1.8, 0.9],
            [2.4, -1.2],
        ],
        dtype=torch.float64,
    )
    head = {
        "terms": [("var", 1)],
        "coeffs": [0.25, -1.5],
    }
    mappings = [
        {"kind": "poly", "coeffs": [0.7, 1.2, -0.4], "mu": 0.1, "std": 1.3},
        {"kind": "pade", "numer": [1.0, 0.5], "denom": [1.0, -0.1], "mu": 0.2, "std": 1.1},
        {"kind": "sine", "A": 1.2, "B": -0.7, "c": 0.3, "omega": 2.0, "mu": 0.1, "std": 0.9},
        {"kind": "exp", "a": 0.8, "b": -0.3, "c": 0.2, "mu": 0.0, "std": 1.0},
        {"kind": "power", "log_a": 0.13, "b": 1.4, "sgn_f": 1.0, "sgn_y": 1.0, "std": 1.0},
    ]

    for mapping in mappings:
        for units_mode in ("raw", "scaled"):
            for with_head in (False, True):
                mapping_case = dict(mapping)
                if with_head:
                    mapping_case["_lin_head"] = dict(head)
                diag = mapping_embedding_roundtrip(
                    ("var", 0),
                    mapping_case,
                    [Var(0), Var(1)],
                    x,
                    units_mode=units_mode,
                    scale_name="unit_scale" if units_mode == "scaled" else None,
                )
                assert diag["ok"], (mapping["kind"], units_mode, with_head, diag)


def test_embed_mapping_preserves_linear_head_for_scaled_sine_omega_z():
    x = torch.tensor(
        [
            [0.7, -0.4],
            [1.1, 0.2],
            [1.8, 0.9],
            [2.4, -1.2],
        ],
        dtype=torch.float64,
    )
    mapping = {
        "kind": "sine",
        "A": 1.2,
        "B": -0.7,
        "c": 0.3,
        "omega": 2.0,
        "mu": 0.1,
        "std": 0.9,
        "_lin_head": {
            "terms": [("var", 1)],
            "coeffs": [0.25, -1.5],
        },
    }
    for sin_arg_mode in ("omega_z", "wu", "wu_phi"):
        diag = mapping_embedding_roundtrip(
            ("var", 0),
            mapping,
            [Var(0), Var(1)],
            x,
            units_mode="scaled",
            scale_name="unit_scale",
            sin_arg_mode=sin_arg_mode,
        )
        assert diag["ok"], (sin_arg_mode, diag)


def test_embed_mapping_roundtrips_basis_state_native_with_linear_head():
    x = torch.tensor(
        [
            [0.7, -0.4],
            [1.1, 0.2],
            [1.8, 0.9],
            [2.4, -1.2],
        ],
        dtype=torch.float64,
    )
    mapping = {
        "kind": "basis_state_native",
        "_lin_head": {
            "terms": [("var", 1)],
            "coeffs": [0.25, -1.5],
        },
    }
    diag = mapping_embedding_roundtrip(
        ("mul", ("var", 0), ("var", 1)),
        mapping,
        [Var(0), Var(1)],
        x,
        units_mode="raw",
    )
    assert diag["ok"], diag


def test_outer_wrapper_name_normalization():
    assert _normalize_outer_wrapper_name("recip") == "reciprocal"
    assert _normalize_outer_wrapper_name("reciprocal") == "reciprocal"
    assert _normalize_outer_wrapper_name(" log ") == "log"
    assert _normalize_outer_wrapper_name("unsupported") is None


def test_outer_wrapper_forward_square_requires_sign_consistency():
    y = torch.tensor([1.0, 2.0, 3.0, -0.1], dtype=torch.float64)
    m, t, sign_hint, reason = _outer_wrapper_forward(y, "square", square_sign_consistency=0.74)
    assert bool(m.any())
    assert abs(float(sign_hint) - 1.0) < 1e-12
    assert reason == "ok"
    assert torch.allclose(t[m], y[m] * y[m])

    m2, t2, sign_hint2, reason2 = _outer_wrapper_forward(y, "square", square_sign_consistency=0.99)
    assert not bool(m2.any())
    assert reason2 == "square_sign_ambiguous"
    assert abs(float(sign_hint2) - 1.0) < 1e-12


def test_outer_wrapper_inverse_ast_builds_expected_nodes():
    inner = Var(0)
    assert str(type(_outer_wrapper_inverse_ast(inner, "log")).__name__) == "ExpNode"
    assert str(type(_outer_wrapper_inverse_ast(inner, "exp")).__name__) == "LogNode"
    assert str(type(_outer_wrapper_inverse_ast(inner, "reciprocal")).__name__) == "PowNode"
    sq = _outer_wrapper_inverse_ast(inner, "square", sign_hint=-1.0)
    assert sq is not None
    assert str(type(sq).__name__) == "MulNode"


def test_outer_wrapper_transformed_dims_rules():
    y_dim = (1.0, -2.0)
    ok, d, reason = _outer_wrapper_transformed_y_dims(y_dim, "reciprocal")
    assert ok and reason == "ok"
    assert d == (-1.0, 2.0)

    ok, d, reason = _outer_wrapper_transformed_y_dims(y_dim, "square")
    assert ok and reason == "ok"
    assert d == (2.0, -4.0)

    ok, d, reason = _outer_wrapper_transformed_y_dims(y_dim, "sqrt")
    assert ok and reason == "ok"
    assert d == (0.5, -1.0)

    ok, d, reason = _outer_wrapper_transformed_y_dims(y_dim, "log")
    assert (not ok) and (d is None)
    assert reason == "requires_dimensionless_target"

    ok, d, reason = _outer_wrapper_transformed_y_dims((0.0, 0.0), "exp")
    assert ok and reason == "ok"
    assert d == (0.0, 0.0)
