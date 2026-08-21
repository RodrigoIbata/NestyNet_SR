# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import pandas as pd
import torch

from nestynet_sr.sr_core.bridges import ConstNode
from nestynet_sr.sr_search.search import _stageA_terminal_closure_committee_gate


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
    )


def test_stagea_terminal_gate_vetoes_witness_regression(tmp_path):
    data_path = tmp_path / "toy.csv"
    pd.DataFrame({"x0": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]}).to_csv(
        data_path,
        index=False,
    )

    ok, reason, summary = _stageA_terminal_closure_committee_gate(
        base_ast=ConstNode(0.0),
        cand_ast=ConstNode(0.0),
        base_model=IdentityModel().double(),
        cand_model=ZeroModel().double(),
        label="bad_terminal",
        gate_kind="test",
        lm_hp=_lm_hp(data_path),
        loss_floor=0.0,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert ok is False
    assert "reject-coe-stageA-terminal-gate" in reason
    assert summary["excluded_slice_ids"] == [0]
    assert summary["losses"] == 1


def test_stagea_terminal_gate_allows_witness_tie(tmp_path):
    data_path = tmp_path / "toy.csv"
    pd.DataFrame({"x0": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]}).to_csv(
        data_path,
        index=False,
    )

    ok, reason, summary = _stageA_terminal_closure_committee_gate(
        base_ast=ConstNode(0.0),
        cand_ast=ConstNode(0.0),
        base_model=IdentityModel().double(),
        cand_model=IdentityModel().double(),
        label="good_terminal",
        gate_kind="test",
        lm_hp=_lm_hp(data_path),
        loss_floor=0.0,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert ok is True
    assert "accepted" in reason
    assert summary["ties"] == 1
    assert summary["losses"] == 0
