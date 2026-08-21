# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Budget and termination contracts for pre-mutation seed routes."""

from __future__ import annotations

import threading

import torch

import nestynet_sr.sr_search.factorized_search.engine.search as search_mod
import nestynet_sr.sr_search.factorized_search.explorer as explorer_mod


def _fake_structural_scorer(mse: float):
    calls = []

    def score(node, _x_fit, _y_fit, x_probe, _y_probe, *_args, **_kwargs):
        calls.append(node)
        z = torch.zeros((x_probe.shape[0], 1), dtype=x_probe.dtype)
        mapping = {
            "kind": "poly",
            "coeffs": [0.0, 1.0],
            "mu": 0.0,
            "std": 1.0,
        }
        return float(mse), ("seed", len(calls)), z, mapping, node

    return calls, score


def _run_search(score, **overrides):
    options = {
        "n_iter": 20,
        "max_depth": 1,
        "poly_degree": 4,
        "lo": 0.2,
        "hi": 0.8,
        "seed": 0,
        "dtype": torch.float64,
        "no_residual": True,
        "brute_depth": 0,
        "print_every": 0,
        "verbose": False,
        "refine_enable": False,
        "_score_expr_fn": score,
    }
    options.update(overrides)
    return explorer_mod.run_explorer_core(lambda x: x[:, :1], 1, **options)


def test_periodic_seeds_respect_base_evaluation_budget(monkeypatch):
    monkeypatch.setattr(
        search_mod,
        "_periodogram_frequency_hints",
        lambda *_args, **_kwargs: [(0, 2.0), (0, 3.0)],
    )
    calls, score = _fake_structural_scorer(1.0)

    arch = _run_search(
        score,
        n_iter=1,
        early_stop_mse=0.0,
        periodic_seed_enable=True,
        carrier_seed_exprs=(("var", 0),),
    )

    assert len(calls) == 1
    assert calls[0][0] == "sin"
    assert arch.n_eval == 1
    assert arch.search_stop_reason == "n_iter"


def test_solving_periodic_seed_stops_later_search_phases(monkeypatch):
    monkeypatch.setattr(
        search_mod,
        "_periodogram_frequency_hints",
        lambda *_args, **_kwargs: [(0, 2.0)],
    )
    calls, score = _fake_structural_scorer(1.0e-12)

    arch = _run_search(
        score,
        early_stop_mse=1.0e-8,
        periodic_seed_enable=True,
        carrier_seed_exprs=(("var", 0),),
    )

    assert len(calls) == 1
    assert calls[0][0] == "sin"
    assert arch.n_eval == 1
    assert arch.search_stop_reason == "early_stop_mse"


def test_carrier_seeds_share_base_evaluation_budget():
    carriers = (
        ("var", 0),
        ("sin", ("var", 0)),
        ("cos", ("var", 0)),
    )
    calls, score = _fake_structural_scorer(1.0)

    arch = _run_search(
        score,
        n_iter=2,
        early_stop_mse=0.0,
        periodic_seed_enable=False,
        carrier_seed_exprs=carriers,
    )

    assert calls == list(carriers[:2])
    assert arch.n_eval == 2
    assert arch.search_stop_reason == "n_iter"


def test_pre_set_stop_event_skips_seed_scoring(monkeypatch):
    def unexpected_periodogram(*_args, **_kwargs):
        raise AssertionError("periodogram must not run after an external stop")

    def unexpected_score(*_args, **_kwargs):
        raise AssertionError("seed scoring must not run after an external stop")

    monkeypatch.setattr(search_mod, "_periodogram_frequency_hints", unexpected_periodogram)
    stop_event = threading.Event()
    stop_event.set()

    arch = _run_search(
        unexpected_score,
        periodic_seed_enable=True,
        carrier_seed_exprs=(("var", 0),),
        stop_event=stop_event,
    )

    assert arch.n_eval == 0
    assert arch.search_stop_reason == "stop_event"
