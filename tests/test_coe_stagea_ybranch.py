# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from nestynet_sr.run_SR import _coe_stageA_ybranch_committee_rank


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


class SquareModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1] ** 2


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


def test_stagea_ybranch_committee_vetoes_raw_y_regression(tmp_path):
    data_path = tmp_path / "toy.csv"
    pd.DataFrame({"x0": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]}).to_csv(
        data_path,
        index=False,
    )

    legacy = {
        "branch_id": "bad",
        "name": "bad",
        "model": ZeroModel().double(),
        "y_op": None,
        "y_op_inv": None,
        "rank_key": (0, "bad"),
    }
    selected, reason, summary = _coe_stageA_ybranch_committee_rank(
        lm_hp=_lm_hp(data_path),
        filepath=data_path,
        identity_branch={
            "branch_id": "identity",
            "model": IdentityModel().double(),
            "y_op": None,
            "y_op_inv": None,
        },
        candidate_branches=[legacy],
        legacy_selected_branch=legacy,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert selected is None
    assert "reject-coe-stageA-ybranch" in reason
    assert summary["excluded_slice_ids"] == [0]
    assert summary["decision"] == "identity"
    assert summary["branches"][0]["losses"] == 1


def test_stagea_ybranch_committee_votes_in_raw_y_space(tmp_path):
    data_path = tmp_path / "toy.csv"
    xs = [1.0, 2.0, 3.0, 4.0]
    pd.DataFrame({"x0": xs, "y": [x * x for x in xs]}).to_csv(data_path, index=False)

    sqrt_branch = {
        "branch_id": "sqrt",
        "name": "sqrt",
        "model": IdentityModel().double(),
        "y_op": np.sqrt,
        "y_op_inv": torch.square,
        "rank_key": (0, "sqrt"),
    }
    selected, reason, summary = _coe_stageA_ybranch_committee_rank(
        lm_hp=_lm_hp(data_path),
        filepath=data_path,
        identity_branch={
            "branch_id": "identity",
            "model": SquareModel().double(),
            "y_op": None,
            "y_op_inv": None,
        },
        candidate_branches=[sqrt_branch],
        legacy_selected_branch=sqrt_branch,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert selected is sqrt_branch
    assert "accepted-coe-stageA-ybranch" in reason
    assert summary["decision"] == "select_branch"
    assert summary["branches"][0]["ties"] == 1
    assert summary["branches"][0]["losses"] == 0
