# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import pytest

pytest.importorskip("sympy")

from nestynet_sr.sr_search.factorized_search.micro_search import (
    MicroSearchGrammar,
    generate_micro_search_dataset,
)
from nestynet_sr.sr_search.factorized_search.micro_search_benchmark import evaluate_micro_search_dataset
from nestynet_sr.sr_search.factorized_search.micro_search_proposer import (
    MicroSearchProposerConfig,
    load_micro_search_proposer_bundle,
    predict_micro_search_proposer,
    save_micro_search_proposer_bundle,
    train_micro_search_closed_vocab_proposer,
)
from nestynet_sr.sr_search.factorized_search.oracle_lab import equation_spec_from_dict
from nestynet_sr.sr_search.config import FactorizedSearchConfig


def _payload_exp_mul():
    return {
        "id": "toy_exp_mul_proposer",
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


def _make_dataset():
    spec = equation_spec_from_dict(_payload_exp_mul(), source="micro-search-proposer")
    return generate_micro_search_dataset(
        [spec],
        factorized_search_hp=_small_hp(),
        seeds=[0],
        depth_min=1,
        depth_max=8,
        max_corrupt_paths_per_spec=4,
        grammar=MicroSearchGrammar(max_depth=3, unary_ops=("exp",), binary_ops=("add", "mul")),
        split_unit="state",
        split_fractions=(1.0, 0.0, 0.0),
        include_samples=True,
        include_completion_tables=True,
        verbose=False,
    )


def test_train_micro_search_closed_vocab_proposer_and_predict(tmp_path):
    payload = _make_dataset()
    bundle = train_micro_search_closed_vocab_proposer(
        payload,
        config=MicroSearchProposerConfig(
            n_probe_points=12,
            max_input_vars=3,
            max_basis_dims=1,
            hidden_dim=64,
        ),
        epochs=150,
        lr=5.0e-3,
        seed=0,
    )

    assert bundle["trained"] is True
    assert bundle["metrics"]["n_train_rows"] >= 1
    assert bundle["metrics"]["train"]["top1_accuracy"] is not None
    assert bundle["metrics"]["train"]["top1_accuracy"] >= 0.5

    bundle_path = tmp_path / "micro_search_proposer.pt"
    save_micro_search_proposer_bundle(bundle, bundle_path)
    loaded = load_micro_search_proposer_bundle(bundle_path)

    row = payload["rows"][0]
    ranked = predict_micro_search_proposer(loaded, row, topk=5)
    assert ranked
    assert ranked[0]["expr"]
    valid_exprs = {cand["expr"] for cand in row["rankings"]["inverse"]} | {cand["expr"] for cand in row["rankings"]["residual"]}
    assert ranked[0]["expr"] in valid_exprs


def test_micro_search_benchmark_supports_learned_policy():
    payload = _make_dataset()
    bundle = train_micro_search_closed_vocab_proposer(
        payload,
        config=MicroSearchProposerConfig(
            n_probe_points=12,
            max_input_vars=3,
            max_basis_dims=1,
            hidden_dim=64,
        ),
        epochs=150,
        lr=5.0e-3,
        seed=0,
    )

    benchmark = evaluate_micro_search_dataset(
        payload,
        policies=("learned", "random"),
        learned_bundle=bundle,
    )
    train = benchmark["splits"]["train"]
    assert train["policies"]["learned"]["n_states"] >= 1
    learned_solve = train["policies"]["learned"]["solve_at_budget"]["1"]
    random_solve = train["policies"]["random"]["solve_at_budget"]["1"]
    assert learned_solve is not None
    assert random_solve is not None
    assert learned_solve >= random_solve
