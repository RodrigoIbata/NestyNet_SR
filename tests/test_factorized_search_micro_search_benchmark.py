# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import pytest

pytest.importorskip("sympy")

from nestynet_sr.sr_search.factorized_search.micro_search import (
    MicroSearchGrammar,
    generate_micro_search_dataset,
)
from nestynet_sr.sr_search.factorized_search.micro_search_benchmark import evaluate_micro_search_dataset
from nestynet_sr.sr_search.factorized_search.oracle_lab import equation_spec_from_dict
from nestynet_sr.sr_search.config import FactorizedSearchConfig


def _payload_exp_mul():
    return {
        "id": "toy_exp_mul_benchmark",
        "basis": ["L"],
        "variables": [
            {"name": "x0", "bounds": [0.1, 1.0], "dim": [0.0]},
            {"name": "x1", "bounds": [0.1, 1.0], "dim": [0.0]},
            {"name": "x2", "bounds": [0.1, 1.0], "dim": [0.0]},
        ],
        "constants": [],
        "target": {"expr": "exp(x0 + x1*x2)", "dim": [0.0]},
    }


def _small_hp():
    hp = FactorizedSearchConfig()
    hp.n_fit = 48
    hp.n_probe = 64
    hp.poly_degree = 3
    return hp


def test_generate_micro_search_dataset_emits_samples_and_splits():
    spec = equation_spec_from_dict(_payload_exp_mul(), source="micro-search-dataset")
    payload = generate_micro_search_dataset(
        [spec],
        factorized_search_hp=_small_hp(),
        seeds=[0],
        depth_min=1,
        depth_max=8,
        max_corrupt_paths_per_spec=1,
        grammar=MicroSearchGrammar(max_depth=3, unary_ops=(), binary_ops=("add", "mul")),
        split_unit="state",
        split_fractions=(1.0, 0.0, 0.0),
        include_samples=True,
        include_completion_tables=True,
        verbose=False,
    )

    assert payload["mode"] == "micro_search_dataset"
    assert payload["n_rows"] == 1
    row = payload["rows"][0]
    assert row["state_id"]
    assert row["split"] == "train"
    assert row["truth_depth"] >= 1
    assert row["samples"]["inverse_target_fit"]
    assert row["samples"]["inverse_valid_mask_fit"]
    assert row["units"]["var_dims"]
    assert row["rankings"]["inverse"]
    assert row["rankings"]["residual"]


def test_generate_micro_search_dataset_split_assignment_is_stable():
    spec = equation_spec_from_dict(_payload_exp_mul(), source="micro-search-stable")
    kwargs = dict(
        specs=[spec],
        factorized_search_hp=_small_hp(),
        seeds=[0, 1],
        depth_min=1,
        depth_max=8,
        max_corrupt_paths_per_spec=3,
        grammar=MicroSearchGrammar(max_depth=3, unary_ops=(), binary_ops=("add", "mul")),
        split_unit="state",
        include_samples=False,
        include_completion_tables=False,
        verbose=False,
    )

    payload_a = generate_micro_search_dataset(**kwargs)
    payload_b = generate_micro_search_dataset(**kwargs)

    splits_a = {row["state_id"]: row["split"] for row in payload_a["rows"]}
    splits_b = {row["state_id"]: row["split"] for row in payload_b["rows"]}
    assert splits_a == splits_b


def test_evaluate_micro_search_dataset_reports_policy_metrics():
    spec = equation_spec_from_dict(_payload_exp_mul(), source="micro-search-benchmark")
    payload = generate_micro_search_dataset(
        [spec],
        factorized_search_hp=_small_hp(),
        seeds=[0],
        depth_min=1,
        depth_max=8,
        max_corrupt_paths_per_spec=1,
        grammar=MicroSearchGrammar(max_depth=3, unary_ops=(), binary_ops=("add", "mul")),
        split_unit="state",
        split_fractions=(1.0, 0.0, 0.0),
        include_samples=False,
        include_completion_tables=True,
        verbose=False,
    )

    benchmark = evaluate_micro_search_dataset(payload, policies=("inverse", "oracle", "random"))

    assert benchmark["mode"] == "micro_search_benchmark"
    train = benchmark["splits"]["train"]
    assert train["n_rows"] == 1
    assert train["policies"]["inverse"]["n_states"] == 1
    assert train["policies"]["oracle"]["n_states"] == 1
    assert train["policies"]["oracle"]["solve_at_budget"]["1"] == 1.0
    assert train["policies"]["oracle"]["solve_at_budget"]["1"] >= train["policies"]["inverse"]["solve_at_budget"]["1"]
