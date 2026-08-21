# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import pandas as pd
import torch

from nestynet_sr.sr_search.search import _stageA_overlap_split_committee_gate


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


class ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


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
        coe_stageA_split_near_floor_mult=25.0,
    )


def _write_toy(path):
    pd.DataFrame({"x0": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]}).to_csv(
        path,
        index=False,
    )


def test_overlap_multiplicative_split_gate_vetoes_witness_regression(tmp_path):
    data_path = tmp_path / "toy.csv"
    _write_toy(data_path)

    ok, reason, summary = _stageA_overlap_split_committee_gate(
        base_model=IdentityModel().double(),
        cand_model=ZeroModel().double(),
        split_kind="mul",
        has_overlap=True,
        base_val_loss=1.0e-6,
        cand_val_loss=1.0e-7,
        noise_floor=0.0,
        under_protest=False,
        lm_hp=_lm_hp(data_path),
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert ok is False
    assert "reject-coe-stageA-overlap-split-gate" in reason
    assert summary["excluded_slice_ids"] == [0]
    assert summary["risk_tags"] == ["overlap_multiplicative_split"]
    assert summary["losses"] == 1


def test_overlap_split_gate_allows_non_high_risk_split_without_witnesses(tmp_path):
    data_path = tmp_path / "toy.csv"
    _write_toy(data_path)

    ok, reason, summary = _stageA_overlap_split_committee_gate(
        base_model=IdentityModel().double(),
        cand_model=ZeroModel().double(),
        split_kind="add",
        has_overlap=False,
        base_val_loss=1.0e-6,
        cand_val_loss=1.0e-7,
        noise_floor=0.0,
        under_protest=False,
        lm_hp=_lm_hp(data_path),
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert ok is True
    assert "not-high-risk" in reason
    assert summary["gate_status"] == "skipped"
    assert summary["results"] == []


def test_under_protest_split_gate_vetoes_witness_regression(tmp_path):
    data_path = tmp_path / "toy.csv"
    _write_toy(data_path)

    ok, reason, summary = _stageA_overlap_split_committee_gate(
        base_model=IdentityModel().double(),
        cand_model=ZeroModel().double(),
        split_kind="add",
        has_overlap=False,
        base_val_loss=1.0e-6,
        cand_val_loss=1.0e-7,
        noise_floor=0.0,
        under_protest=True,
        lm_hp=_lm_hp(data_path),
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert ok is False
    assert summary["risk_tags"] == ["under_protest_split"]
    assert summary["losses"] == 1
