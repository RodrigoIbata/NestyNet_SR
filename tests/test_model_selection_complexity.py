# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from nestynet_sr.sr_core.bridges import (
    Abs,
    Add,
    Arg,
    AtomNode,
    Conj,
    ConstNode,
    Log,
    Sin,
    Var,
)
from nestynet_sr.sr_search.model_selection import (
    ast_operator_cost,
    complexity_key,
    mapping_cost,
    pareto_front_indices_2d,
)


def test_ast_operator_cost_penalizes_heavy_unary_wrappers():
    simple = Add(Var(0), Var(1))
    heavy = Abs(Arg(Conj(Add(Var(0), Var(1)))))
    assert ast_operator_cost(heavy) > ast_operator_cost(simple)


def test_ast_operator_cost_handles_mul_scale_and_exp_ratpoly_kinds():
    mul_scale = AtomNode(kind="mul_scale", var_idxs=(0,), kwargs={})
    affine = AtomNode(kind="lin", var_idxs=(0,), kwargs={})
    exp_ratpoly = AtomNode(
        kind="exp_ratpoly", var_idxs=(0,), kwargs={"deg_num": 2, "deg_den": 2}
    )
    ratpoly = AtomNode(kind="ratpoly", var_idxs=(0,), kwargs={"deg_num": 2, "deg_den": 2})
    assert ast_operator_cost(mul_scale) < ast_operator_cost(affine)
    assert ast_operator_cost(exp_ratpoly) > ast_operator_cost(ratpoly)


def test_ast_operator_cost_prefers_sparse_ratpoly_overrides():
    dense = AtomNode(kind="ratpoly", var_idxs=(0, 1), kwargs={"deg_num": 3, "deg_den": 3})
    sparse = AtomNode(
        kind="ratpoly",
        var_idxs=(0, 1),
        kwargs={
            "deg_num": 3,
            "deg_den": 3,
            "support_num_override": [0, 5],
            "support_den_override": [0, 4, 8],
        },
    )
    assert ast_operator_cost(sparse) < ast_operator_cost(dense)


def test_ast_operator_cost_prefers_sparse_rratpoly_overrides():
    dense = AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={"deg_num": 4, "deg_den": 4})
    sparse = AtomNode(
        kind="rratpoly",
        var_idxs=(0,),
        kwargs={
            "deg_num": 4,
            "deg_den": 4,
            "exps_num_override": [[1]],
            "exps_den_override": [[0], [2], [4]],
        },
    )
    assert ast_operator_cost(sparse) < ast_operator_cost(dense)


def test_complexity_key_penalizes_symbolic_ops_when_params_equal():
    simple = Add(Var(0), Var(1))
    complex_expr = Add(Sin(Var(0)), Log(Add(Var(1), ConstNode(1.0))))
    assert complexity_key(simple, 5) < complexity_key(complex_expr, 5)


def test_complexity_key_keeps_nn_structural_signal():
    # Same parameter count; multivariate-NN structure should remain more complex.
    simple_nn = AtomNode(kind="nn", var_idxs=(0,), kwargs={})
    hard_nn = Add(
        AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}),
        AtomNode(kind="nn", var_idxs=(1, 2), kwargs={}),
    )
    assert complexity_key(simple_nn, 10) < complexity_key(hard_nn, 10)


def test_unresolved_nn_cost_dominates_visible_analytic_formula():
    nn_leaf = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={})
    analytic = Add(Sin(Var(0)), Log(Add(Var(1), ConstNode(1.0))))
    for i in range(24):
        analytic = Add(analytic, Sin(Add(Var(i % 3), ConstNode(float(i + 2)))))

    assert ast_operator_cost(analytic) < ast_operator_cost(nn_leaf)
    assert complexity_key(analytic, 1000) < complexity_key(nn_leaf, 1)


def test_mapping_cost_penalizes_approximative_maps():
    mono = {"kind": "monomial"}
    affine_kind = {"kind": "affine"}
    poly_affine = {"kind": "poly", "coeffs": [0.0, 1.0]}
    poly_deg4 = {"kind": "poly", "coeffs": [0.0, 1.0, 0.0, 0.0, 0.0]}
    pade_22 = {"kind": "pade", "numer": [0.0, 1.0, 0.0], "denom": [1.0, 0.0, 0.0]}
    assert mapping_cost(mono) < mapping_cost(poly_affine)
    assert mapping_cost(affine_kind) <= mapping_cost(poly_affine)
    assert mapping_cost(poly_affine) < mapping_cost(poly_deg4)
    assert mapping_cost(poly_deg4) < mapping_cost(pade_22)
    assert mapping_cost(poly_deg4) >= 12.0
    assert mapping_cost(pade_22) >= 30.0


def test_pareto_front_indices_2d_filters_dominated_points():
    points = [
        (1.0e-6, 10.0),  # keep
        (5.0e-7, 12.0),  # keep (better loss, worse complexity)
        (2.0e-6, 15.0),  # dominated by point 0
        (5.0e-7, 20.0),  # dominated by point 1
    ]
    keep = pareto_front_indices_2d(points)
    assert keep == [1, 0]
