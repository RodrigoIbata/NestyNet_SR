# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Tests for ResidualBasinArchive.best ranking policy."""

from typing import Optional

import torch

from nestynet_sr.sr_search.factorized_search import explorer as explorer_mod
from nestynet_sr.sr_search.factorized_search.engine.archive import ResidualBasinArchive, Elite, Rec


def test_explorer_reexports_engine_archive_types():
    assert explorer_mod.ResidualBasinArchive is ResidualBasinArchive
    assert explorer_mod.Elite is Elite
    assert explorer_mod.Rec is Rec


def _mk_elite(mse: float, size: float, tag: str, *, raw_mse: Optional[float] = None) -> Elite:
    return Elite(
        mse=float(mse),
        expr=("var", 0) if tag == "x0" else ("mul", ("var", 0), ("var", 1)),
        mapping={"kind": "poly", "coeffs": [0.0, 1.0]},
        z=None,  # not used by best()
        size=float(size),
        raw_mse=float(mse if raw_mse is None else raw_mse),
    )


def test_best_k1_remains_pure_best_mse():
    arch = ResidualBasinArchive()
    # Build archive manually (best() only reads d/elites/visits).
    arch.d = {
        1: Rec(1.0e-6, ("var", 0), 3, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(1.0e-6, 100.0, "x0"),
        ]),
        2: Rec(2.0e-6, ("var", 0), 2, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(2.0e-6, 5.0, "x0"),
        ]),
    }
    out = arch.best(1)
    assert len(out) == 1
    assert out[0].best_mse == 1.0e-6


def test_best_default_strategy_is_legacy_mse():
    arch = ResidualBasinArchive()
    arch.d = {
        1: Rec(1.0e-8, ("var", 0), 3, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(1.0e-8, 120.0, "x0"),
            _mk_elite(8.0e-8, 6.0, "x0"),
            _mk_elite(5.0e-8, 10.0, "x0"),
        ]),
        2: Rec(2.0e-7, ("var", 0), 2, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(2.0e-7, 4.0, "x0"),
        ]),
    }
    out = arch.best(3)
    mses = [float(r.best_mse) for r in out]
    assert mses == [1.0e-8, 5.0e-8, 8.0e-8]


def test_best_k_roundrobins_decades_and_prefers_small_size_within_decade():
    arch = ResidualBasinArchive()
    # Same best decade has both huge and small-size elites; next decade exists.
    # Policy should include:
    # 1) global best-MSE first,
    # 2) then simple candidate from same decade,
    # 3) then candidate from next decade (round-robin).
    arch.d = {
        1: Rec(1.0e-8, ("var", 0), 3, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(1.0e-8, 120.0, "x0"),   # global best mse, very complex
            _mk_elite(8.0e-8, 6.0, "x0"),     # same decade, simple
            _mk_elite(5.0e-8, 10.0, "x0"),    # same decade, medium
        ]),
        2: Rec(2.0e-7, ("var", 0), 2, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(2.0e-7, 4.0, "x0"),     # next decade, simple
        ]),
    }

    out = arch.best(3, strategy="mse_decade_size")
    mses = [float(r.best_mse) for r in out]

    # First item is still global best-MSE.
    assert mses[0] == 1.0e-8
    # Round-robin + size ordering should include same-decade simple candidate.
    assert 8.0e-8 in mses
    # And include the next decade candidate in top-3.
    assert 2.0e-7 in mses


def test_best_decade_strategy_bins_by_raw_mse_not_objective_mse():
    arch = ResidualBasinArchive()
    arch.d = {
        1: Rec(1.0e-8, ("var", 0), 3, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(1.0e-8, 50.0, "x0", raw_mse=1.0e-8),   # global best objective
            _mk_elite(1.5e-8, 1.0, "x0", raw_mse=1.0e-4),    # would win if objective-decade only
            _mk_elite(2.0e-8, 2.0, "x0", raw_mse=1.0e-7),    # preferred by raw-mse decade
        ]),
    }
    out = arch.best(3, strategy="mse_decade_size")
    mses = [float(r.best_mse) for r in out]
    assert mses == [1.0e-8, 2.0e-8, 1.5e-8]


def test_best_unknown_strategy_falls_back_to_legacy_mse():
    arch = ResidualBasinArchive()
    arch.d = {
        1: Rec(1.0e-8, ("var", 0), 3, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(1.0e-8, 120.0, "x0", raw_mse=1.0e-3),
            _mk_elite(5.0e-8, 6.0, "x0", raw_mse=1.0e-6),
        ]),
        2: Rec(2.0e-8, ("var", 0), 2, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(2.0e-8, 4.0, "x0", raw_mse=2.0e-7),
        ]),
    }
    out = arch.best(3, strategy="unknown_strategy_name")
    mses = [float(r.best_mse) for r in out]
    assert mses == [1.0e-8, 2.0e-8, 5.0e-8]


def test_best_decade_strategy_ignores_far_worse_decades_by_default():
    arch = ResidualBasinArchive()
    arch.d = {
        1: Rec(1.0e-8, ("var", 0), 3, {"kind": "poly", "coeffs": [0.0, 1.0]}, None, [
            _mk_elite(1.0e-8, 50.0, "x0", raw_mse=1.0e-8),
            _mk_elite(2.0e-8, 2.0, "x0", raw_mse=2.0e-8),
            _mk_elite(3.0e-8, 3.0, "x0", raw_mse=3.0e-8),
            _mk_elite(4.0e-8, 1.0, "x0", raw_mse=1.0e2),  # far decade (should be excluded)
        ]),
    }
    out = arch.best(3, strategy="mse_decade_size")
    mses = [float(r.best_mse) for r in out]
    assert 4.0e-8 not in mses


def test_basin_best_raw_mse_stays_coherent_with_best_elite():
    arch = ResidualBasinArchive(elite_k=4, elite_merge_cos=0.99)

    complex_expr = ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 1))
    simple_expr = ("var", 0)
    mapping = {"kind": "poly", "coeffs": [0.0, 1.0]}

    arch.update(
        "same_basin",
        1.0,
        complex_expr,
        torch.tensor([1.0, 0.0]),
        mapping,
        raw_mse=1.0e-6,
    )
    arch.update(
        "same_basin",
        1.04,
        simple_expr,
        torch.tensor([0.0, 1.0]),
        mapping,
        raw_mse=1.0e-2,
    )

    rec = arch.d["same_basin"]
    assert rec.best_expr == simple_expr
    assert float(rec.best_mse) == 1.04
    assert float(rec.best_raw_mse) == 1.0e-2
    assert float(rec.min_raw_mse) == 1.0e-6
    assert arch.audit_coherence()["ok"]


def test_archive_rejects_nonfinite_objective_scores():
    arch = ResidualBasinArchive()
    mapping = {"kind": "poly", "coeffs": [0.0, 1.0]}

    assert not arch.update(
        "bad",
        float("nan"),
        ("var", 0),
        torch.tensor([1.0, 0.0]),
        mapping,
        raw_mse=1.0,
    )
    assert "bad" not in arch.d

    assert arch.update(
        "good",
        1.0,
        ("var", 0),
        torch.tensor([1.0, 0.0]),
        mapping,
        raw_mse=float("nan"),
    )
    assert float(arch.d["good"].best_raw_mse) == 1.0
