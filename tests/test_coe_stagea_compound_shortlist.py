# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import pandas as pd
import torch

from nestynet_sr.sr_search.search import _stageA_compound_shortlist_committee_rank


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


class ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


class X0Model(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


def _lm_hp(data_path):
    return SimpleNamespace(
        coe_mode="committee_gated",
        coe_filepath=str(data_path),
        coe_num_slices=1,
        coe_stageB_gate_slices=1,
        coe_ndata_train=1,
        coe_ndata_val=1,
        coe_start_slice=0,
        coe_reference_slice=0,
        coe_noise_floor_raw=0.0,
        coe_noise_mult=3.0,
        coe_rel_tol=1.0e-6,
    )


def _candidate(name, model, ref_loss):
    return {
        "z_name": name,
        "kind": "monomial",
        "model": model.double(),
        "val_loss": float(ref_loss),
        "old_arity": 3,
        "new_arity": 1,
        "z_readable": name,
    }


def test_compound_shortlist_selects_witness_supported_candidate(tmp_path):
    data_path = tmp_path / "toy.csv"
    pd.DataFrame({"x0": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]}).to_csv(
        data_path,
        index=False,
    )

    good = _candidate("good", IdentityModel(), 1.0e-6)
    bad = _candidate("bad", ZeroModel(), 1.0e-7)
    selected, reason, summary = _stageA_compound_shortlist_committee_rank(
        base_model=ZeroModel().double(),
        candidates=[bad, good],
        lm_hp=_lm_hp(data_path),
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert selected is good
    assert "accepted-coe-stageA-compound-shortlist" in reason
    assert summary["excluded_slice_ids"] == [0]
    assert summary["decision"] == "select_candidate"
    assert summary["selected"] == "good"


def test_compound_shortlist_vetoes_all_witness_regressions(tmp_path):
    data_path = tmp_path / "toy.csv"
    pd.DataFrame({"x0": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]}).to_csv(
        data_path,
        index=False,
    )

    bad = _candidate("bad", ZeroModel(), 1.0e-7)
    selected, reason, summary = _stageA_compound_shortlist_committee_rank(
        base_model=IdentityModel().double(),
        candidates=[bad],
        lm_hp=_lm_hp(data_path),
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert selected is None
    assert "reject-coe-stageA-compound-shortlist" in reason
    assert summary["decision"] == "veto_all"
    assert summary["results"][0]["losses"] == 1


def test_compound_shortlist_prefers_protected_full_reduction_when_noise_tied(tmp_path):
    data_path = tmp_path / "toy.csv"
    rows = []
    # With ntrain=1,nval=1 and reference slice 0 excluded, witness val rows are
    # 3,5,7,9,11.  The protected candidate loses two witness votes but has a
    # zero median delta, matching the pb090-style full-radial case.
    val_x = {3: 0.0, 5: 0.0, 7: 0.0, 9: 0.01, 11: 0.01}
    for i in range(12):
        rows.append({"x0": val_x.get(i, 0.0), "y": 0.0})
    pd.DataFrame(rows).to_csv(data_path, index=False)

    hp = _lm_hp(data_path)
    hp.coe_num_slices = 5
    hp.coe_stageB_gate_slices = 5
    hp.coe_noise_floor_raw = 1.0e-5
    full = {
        "z_name": "full_radial",
        "kind": "radial",
        "model": X0Model().double(),
        "val_loss": 1.0e-6,
        "old_arity": 3,
        "new_arity": 1,
        "pattern": (1, 1, 1),
        "z_readable": "x0^2+x1^2+x2^2",
    }
    partial = {
        "z_name": "partial_radial",
        "kind": "radial",
        "model": ZeroModel().double(),
        "val_loss": 1.0e-6,
        "old_arity": 3,
        "new_arity": 2,
        "pattern": (1, 1, 0),
        "z_readable": "x0^2+x1^2",
    }

    selected, reason, summary = _stageA_compound_shortlist_committee_rank(
        base_model=ZeroModel().double(),
        candidates=[partial, full],
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert selected is full
    assert "accepted-coe-stageA-compound-shortlist" in reason
    full_summary = next(row for row in summary["results"] if row["z_name"] == "full_radial")
    assert full_summary["losses"] == 2
    assert full_summary["protected_tie_allowed"] is True


def test_compound_shortlist_allows_median_better_radial_completion_with_one_loss(tmp_path):
    data_path = tmp_path / "toy.csv"
    rows = []
    # With ntrain=1,nval=1 and reference slice 0 excluded, witness val rows are
    # 3,5,7,9,11.  The radial completion wins four witnesses and loses one
    # non-catastrophically, matching the pb055 completion shape.
    val_rows = {
        3: (1.0, 1.0),
        5: (1.0, 1.0),
        7: (1.0, 1.0),
        9: (1.0, 1.0),
        11: (0.01, 0.0),
    }
    for i in range(12):
        x0, y = val_rows.get(i, (0.0, 0.0))
        rows.append({"x0": x0, "y": y})
    pd.DataFrame(rows).to_csv(data_path, index=False)

    hp = _lm_hp(data_path)
    hp.coe_num_slices = 5
    hp.coe_stageB_gate_slices = 5
    hp.coe_noise_floor_raw = 1.0e-5
    cand = {
        "z_name": "radial_completion",
        "kind": "power_pair_sumdiff",
        "model": X0Model().double(),
        "val_loss": 1.0e-6,
        "old_arity": 3,
        "new_arity": 2,
        "pattern": (1, 1, 0),
        "z_readable": "x1^2+x2^2",
    }

    selected, reason, summary = _stageA_compound_shortlist_committee_rank(
        base_model=ZeroModel().double(),
        candidates=[cand],
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert selected is cand
    assert "accepted-coe-stageA-compound-shortlist" in reason
    row = summary["results"][0]
    assert row["wins"] == 4
    assert row["losses"] == 1
    assert row["median_delta"] < 0.0
    assert row["structurally_protected"] is True
    assert row["protected_tie_allowed"] is True
