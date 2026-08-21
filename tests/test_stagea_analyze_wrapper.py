# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import nestynet_sr.sr_search.search as search_mod


class _DummyModel:
    def __init__(self, loss, signals=None):
        self._best_val_loss_base = float(loss)
        if signals is not None:
            self._stageA_signals = dict(signals)


def test_stagea_analyze_packages_output(monkeypatch):
    def _fake_run(**kwargs):
        return (
            True,
            _DummyModel(1.23e-4),
            [0],
            [1],
            [True, False, True],
            SimpleNamespace(name="ast"),
            False,
            True,
        )

    monkeypatch.setattr(search_mod, "run_separability_for_transform", _fake_run)

    out = search_mod.stageA_analyze(
        i_op=0,
        y_op=None,
        y_op_inv=None,
        candidate_sep_ops=[True, False, False],
        y_transform_names=["identity", "square", "log"],
        initial_ast=SimpleNamespace(name="init"),
        filepath="dummy.csv",
        Nxvars=2,
        y_med=0.0,
        y_mad=1.0,
        np_dtype=None,
        dtype=None,
        device=None,
        data_hp=None,
        model_hp=None,
        lm_hp=None,
        search_hp=None,
        leaf_builder=None,
        model_output="dummy.mod",
        model_sep_output="dummy_sep.mod",
    )

    assert out.success is True
    assert out.full_compound_solved is True
    assert out.y_transform_name == "identity"
    assert out.val_loss_base == 1.23e-4
    assert out.candidate_sep_ops == [True, False, True]
    assert isinstance(out.signals, dict)
    assert out.signals.get("best_split_score") == 1.0
    assert out.signals.get("full_compound_compressed") == 1.0
    assert out.signals.get("full_compound_solved") == 0.0
    assert len(out.split_plans) == 2
    assert {p.kind for p in out.split_plans} == {"add", "mul"}


def test_stagea_analyze_prefers_model_stagea_signals(monkeypatch):
    model_signals = {
        "trig_affine_conf": 0.42,
        "sep_score": 0.67,
        "best_split_score": 0.73,
        "split_success": 0.0,
        "sep_candidates_seen": 5.0,
    }

    def _fake_run(**kwargs):
        return (
            False,
            _DummyModel(3.21e-3, signals=model_signals),
            None,
            None,
            [False, True, False],
            SimpleNamespace(name="ast"),
            False,
            False,
        )

    monkeypatch.setattr(search_mod, "run_separability_for_transform", _fake_run)

    out = search_mod.stageA_analyze(
        i_op=1,
        y_op=None,
        y_op_inv=None,
        candidate_sep_ops=[False, True, False],
        y_transform_names=["identity", "square", "log"],
        initial_ast=SimpleNamespace(name="init"),
        filepath="dummy.csv",
        Nxvars=2,
        y_med=0.0,
        y_mad=1.0,
        np_dtype=None,
        dtype=None,
        device=None,
        data_hp=None,
        model_hp=None,
        lm_hp=None,
        search_hp=None,
        leaf_builder=None,
        model_output="dummy.mod",
        model_sep_output="dummy_sep.mod",
    )

    assert out.signals["trig_affine_conf"] == 0.42
    assert out.signals["sep_score"] == 0.67
    assert out.signals["best_split_score"] == 0.73
    assert out.split_plans == []
