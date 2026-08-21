# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import pytest

pytest.importorskip("sympy")

from nestynet_sr.sr_search.factorized_search.expr_ast import node_str
from nestynet_sr.sr_search.factorized_search.micro_search import (
    MicroSearchGrammar,
    enumerate_micro_search_expressions,
    run_single_hole_micro_search,
)
from nestynet_sr.sr_search.factorized_search.oracle_lab import equation_spec_from_dict
from nestynet_sr.sr_search.config import FactorizedSearchConfig


def _payload_exp_mul():
    return {
        "id": "toy_exp_mul",
        "basis": ["L"],
        "variables": [
            {"name": "x0", "bounds": [0.1, 1.0], "dim": [0.0]},
            {"name": "x1", "bounds": [0.1, 1.0], "dim": [0.0]},
            {"name": "x2", "bounds": [0.1, 1.0], "dim": [0.0]},
        ],
        "constants": [],
        "target": {"expr": "exp(x0 + x1*x2)", "dim": [0.0]},
    }


def test_enumerate_micro_search_expressions_finds_truth_subtree():
    grammar = MicroSearchGrammar(max_depth=2, unary_ops=(), binary_ops=("mul",))
    nodes = enumerate_micro_search_expressions(3, grammar=grammar)
    exprs = {node_str(node) for node in nodes}

    assert "x0" in exprs
    assert "(x1*x2)" in exprs
    assert len(exprs) == len(nodes)


def test_enumerate_micro_search_expressions_respects_target_dims():
    grammar = MicroSearchGrammar(max_depth=2, unary_ops=(), binary_ops=("add", "mul"))
    nodes = enumerate_micro_search_expressions(
        2,
        grammar=grammar,
        var_dims=((1.0,), (-1.0,)),
        target_dim=(0.0,),
    )
    exprs = {node_str(node) for node in nodes}

    assert "(x0*x1)" in exprs
    assert "x0" not in exprs
    assert "(x0+x0)" not in exprs


def test_run_single_hole_micro_search_recovers_truth_under_inverse_target():
    spec = equation_spec_from_dict(_payload_exp_mul(), source="micro-search")

    hp = FactorizedSearchConfig()
    hp.n_fit = 64
    hp.n_probe = 96
    hp.poly_degree = 3

    report = run_single_hole_micro_search(
        spec,
        factorized_search_hp=hp,
        seed=0,
        corrupt_path=(1, 2),
        grammar=MicroSearchGrammar(max_depth=2, unary_ops=(), binary_ops=("mul",)),
        report_topk=8,
    )

    assert report["mode"] == "micro_search_single_hole"
    assert report["hole_path"] == [1, 2]
    assert report["grammar"]["truth_in_grammar"] is True
    assert report["hole_truth_expr"] == "(x1*x2)"
    assert report["metrics"]["inverse"]["truth_rank"] == 1
    assert report["metrics"]["inverse"]["solve_at_budget"]["1"] is True
    truth_row = report["rankings"]["inverse"][0]
    assert truth_row["is_truth"] is True
    assert truth_row["expr"] == "(x1*x2)"
    assert truth_row["full_probe_mse"] < 1.0e-20
    residual_rank = report["metrics"]["residual"]["truth_rank"]
    assert residual_rank is None or report["metrics"]["inverse"]["truth_rank"] <= residual_rank
