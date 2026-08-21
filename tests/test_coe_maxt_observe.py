# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Observe-only max-T wiring at the CoE decision sites (phase 3).

These tests drive the real gate code paths and assert the calibrated
verdict is recorded next to the legacy vote without changing any decision.
"""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import pandas as pd
import torch

from nestynet_sr.run_SR import _coe_stageA_ybranch_committee_rank
from nestynet_sr.sr_core.bridges import AtomNode, ConstNode
from nestynet_sr.sr_search.stageB.engine import Candidate, StageBContext, StageBState


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


class ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


def _write_linear_csv(path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    pd.DataFrame({"x0": x, "y": x}).to_csv(path, index=False)


def test_stageb_fixed_expression_gate_records_maxt_observation(tmp_path):
    data_path = tmp_path / "toy_gate.csv"
    _write_linear_csv(data_path)

    lm_hp = SimpleNamespace(
        loss_target=1.0e-9,
        loss_acceptable=1.0,
        loss_in_MAD_units=False,
        fit_y_link=None,
    )
    state = StageBState(
        root=AtomNode(kind="var", var_idxs=(0,)),
        model=torch.nn.Linear(1, 1),
        reuse={},
        val_loss=0.0,
    )
    cand_state = StageBState(
        root=ConstNode(0.0), model=torch.nn.Linear(1, 1), reuse={}, val_loss=1.0
    )
    ctx = StageBContext(
        state=state,
        train_loader=object(),
        val_loader=object(),
        lm_hp=lm_hp,
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=5,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        stageA_x_transforms={},
        xcoords=None,
        xcoords_applied=False,
        coe_mode="committee_gated",
        coe_filepath=str(data_path),
        coe_num_slices=1,
        coe_stageB_gate_slices=1,
        coe_stageB_initial_gate_slices=1,
        coe_ndata_train=2,
        coe_ndata_val=50,
        coe_reference_slice=0,
        coe_noise_floor_raw=0.0,
        coe_noise_mult=3.0,
        coe_rel_tol=1.0e-3,
        coe_min_valid_fraction=0.8,
        coe_inference="maxt_observe",
        coe_maxt_seed=17,
        acceptance_noise_n_eff=1,
    )
    ctx.record_coe_stageB_dry_run = MethodType(
        lambda self, **kw: {"risk_tags": ["generic_approximant"]},
        ctx,
    )

    ok, reason = ctx.coe_stageB_committee_gate(
        rule="dummy",
        label="bad_const",
        reason="accepted",
        target="root",
        target_uid="root",
        cand=Candidate(label="candidate", root=ConstNode(0.0)),
        cand_state=cand_state,
        n_params_base=0,
        n_params_cand=0,
    )

    # Legacy decision unchanged: the constant is vetoed.
    assert ok is False
    assert "reject-coe-stageB-gate" in str(reason)

    record = ctx.coe_stageB_gate_log[-1]
    observed = record["summary"]["maxt_observe"]
    assert observed["method"] == "committee_paired_maxt"
    assert observed["maxt_gate_equivalent"] == "veto"
    assert observed["legacy_gate"] == "veto"
    assert observed["agrees_with_legacy"] is True
    (verdict,) = observed["member_verdicts"]
    assert verdict["member_id"] == "candidate"
    assert verdict["verdict"] == "worse"
    assert observed["n_units"] == 50
    assert observed["inference_regime"]


def test_stageb_gate_without_observe_flag_records_no_maxt(tmp_path):
    data_path = tmp_path / "toy_gate_legacy.csv"
    _write_linear_csv(data_path)
    lm_hp = SimpleNamespace(
        loss_target=1.0e-9,
        loss_acceptable=1.0,
        loss_in_MAD_units=False,
        fit_y_link=None,
    )
    ctx = StageBContext(
        state=StageBState(
            root=AtomNode(kind="var", var_idxs=(0,)),
            model=torch.nn.Linear(1, 1),
            reuse={},
            val_loss=0.0,
        ),
        train_loader=object(),
        val_loader=object(),
        lm_hp=lm_hp,
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=5,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        stageA_x_transforms={},
        xcoords=None,
        xcoords_applied=False,
        coe_mode="committee_gated",
        coe_filepath=str(data_path),
        coe_num_slices=1,
        coe_stageB_gate_slices=1,
        coe_stageB_initial_gate_slices=1,
        coe_ndata_train=2,
        coe_ndata_val=50,
        coe_reference_slice=0,
        coe_noise_floor_raw=0.0,
        coe_noise_mult=3.0,
        coe_rel_tol=1.0e-3,
        coe_min_valid_fraction=0.8,
        acceptance_noise_n_eff=1,
    )
    ctx.record_coe_stageB_dry_run = MethodType(
        lambda self, **kw: {"risk_tags": ["generic_approximant"]},
        ctx,
    )
    ok, _reason = ctx.coe_stageB_committee_gate(
        rule="dummy",
        label="bad_const",
        reason="accepted",
        target="root",
        target_uid="root",
        cand=Candidate(label="candidate", root=ConstNode(0.0)),
        cand_state=StageBState(
            root=ConstNode(0.0), model=torch.nn.Linear(1, 1), reuse={}, val_loss=1.0
        ),
        n_params_base=0,
        n_params_cand=0,
    )
    assert ok is False
    assert "maxt_observe" not in ctx.coe_stageB_gate_log[-1]["summary"]


def test_stagea_ybranch_records_maxt_observation_per_branch(tmp_path):
    data_path = tmp_path / "toy_ybranch.csv"
    _write_linear_csv(data_path, seed=3)

    lm_hp = SimpleNamespace(
        coe_mode="committee_gated",
        coe_filepath=str(data_path),
        coe_num_slices=1,
        coe_stageB_gate_slices=1,
        coe_ndata_train=2,
        coe_ndata_val=50,
        coe_start_slice=0,
        coe_reference_slice=0,
        coe_noise_floor_raw=0.0,
        coe_noise_mult=3.0,
        coe_rel_tol=1.0e-6,
        coe_inference="maxt_observe",
        coe_maxt_seed=5,
    )
    bad = {
        "branch_id": "bad",
        "name": "bad",
        "model": ZeroModel().double(),
        "y_op": None,
        "y_op_inv": None,
        "rank_key": (0, "bad"),
    }
    selected, reason, summary = _coe_stageA_ybranch_committee_rank(
        lm_hp=lm_hp,
        filepath=data_path,
        identity_branch={
            "branch_id": "identity",
            "model": IdentityModel().double(),
            "y_op": None,
            "y_op_inv": None,
        },
        candidate_branches=[bad],
        legacy_selected_branch=bad,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    # Legacy decision unchanged: the zero branch loses to identity.
    assert selected is None
    assert "reject-coe-stageA-ybranch" in reason

    observed = summary["maxt_observe"]
    assert observed["method"] == "committee_paired_maxt"
    per_branch = observed["per_branch"]["bad"]
    assert per_branch["verdict"] == "worse"
    assert per_branch["legacy_allowed"] is False
    assert per_branch["maxt_allowed"] is False
    assert per_branch["agrees_with_legacy"] is True


def test_refit_gate_records_cluster_keyed_maxt_observation(tmp_path):
    """The refit ladder observer: rows are units, slices are CLUSTERS."""
    from types import MethodType

    from nestynet_sr.sr_search.stageB.engine import Candidate as _Candidate

    data_path = tmp_path / "toy_refit.csv"
    _write_linear_csv(data_path, n=400, seed=7)

    lm_hp = SimpleNamespace(
        loss_target=1.0e-9,
        loss_acceptable=1.0,
        loss_in_MAD_units=False,
        coe_mode="committee_gated",
        coe_stageB_dry_run=False,
        coe_filepath=str(data_path),
        coe_num_slices=3,
        coe_stageB_gate_slices=3,
        coe_stageB_initial_gate_slices=3,
        coe_ndata_train=2,
        coe_ndata_val=40,
        coe_reference_slice=0,
        coe_stageB_refit_epochs=2,
        coe_stageB_refit_escalate_epochs=0,
        coe_noise_floor_raw=0.0,
        coe_noise_mult=3.0,
        coe_rel_tol=1.0e-6,
        coe_min_valid_fraction=0.8,
        coe_stageB_refit_gate=True,
        coe_witness_parallelism=1,
        coe_inference="maxt_observe",
        coe_maxt_seed=3,
        fit_y_link=None,
    )
    ctx = StageBContext(
        state=StageBState(
            root=AtomNode(kind="var", var_idxs=(0,)),
            model=IdentityModel().double(),
            reuse={},
            val_loss=0.0,
        ),
        train_loader=object(),
        val_loader=object(),
        lm_hp=lm_hp,
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=2,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        dataset_ids=["reference"],
        loss_scales=[1.0],
        agg_mode="mean",
        agg_weights=None,
        _cache={},
        acceptance_noise_n_eff=1,
    )

    def fake_fit_candidate(self, cand, epochs_override=None):
        # Incumbent refits to the exact law; candidate refits to zero.
        if "candidate" in str(cand.label):
            return StageBState(
                root=cand.root, model=ZeroModel().double(), reuse={}, val_loss=1.0
            )
        return StageBState(
            root=cand.root, model=IdentityModel().double(), reuse={}, val_loss=1e-12
        )

    ctx.fit_candidate = MethodType(fake_fit_candidate, ctx)
    records = []

    ok, reason = ctx._coe_stageB_refit_committee_gate(
        rule="dummy",
        label="bad_zero",
        reason="accepted",
        target="root",
        target_uid="root",
        cand=_Candidate(label="candidate", root=ConstNode(0.0)),
        cand_state=StageBState(
            root=ConstNode(0.0), model=ZeroModel().double(), reuse={}, val_loss=1.0
        ),
        n_params_base=0,
        n_params_cand=0,
        risk_tags=["generic_approximant"],
        gate_record_fn=lambda **kw: records.append(kw) or kw,
        incumbent_snapshot="x0",
        candidate_snapshot="0",
    )

    assert ok is False
    assert "reject-coe-stageB-refit-gate" in str(reason)

    observed = records[-1]["summary"]["maxt_observe"]
    assert observed["method"] == "committee_paired_maxt"
    assert observed["cluster_by_slice"] is True
    assert observed["n_clusters"] == 3
    assert observed["n_units"] == 120  # 3 slices x 40 val rows
    assert observed["maxt_gate_equivalent"] == "veto"
    assert observed["legacy_gate"] == "veto"
    assert observed["agrees_with_legacy"] is True
    (verdict,) = observed["member_verdicts"]
    assert verdict["verdict"] == "worse"


def test_cluster_bridge_refuses_single_slice():
    from nestynet_sr.stat_selection.committee_inference import (
        maxt_decision_from_slice_rows,
    )

    with pytest.raises(ValueError, match="at least 2 shared witness slices"):
        maxt_decision_from_slice_rows(
            baseline_rows={0: (0, np.ones(50))},
            member_rows={"m": {0: (0, np.zeros(50))}},
            seed=1,
            cluster_by_slice=True,
        )


def test_process_refit_worker_exports_row_losses(tmp_path):
    """The portable process worker honors return_row_losses end to end."""
    import copy

    from nestynet_sr.sr_core.bridges import FreeConst, MulNode, Var
    from nestynet_sr.sr_search.coe_witness import (
        _stageB_refit_pair_worker,
        coe_stageB_refit_ast_to_payload,
    )
    from nestynet_sr.sr_search.config import LMHyperparams

    data_path = tmp_path / "toy_worker.csv"
    _write_linear_csv(data_path, n=120, seed=11)

    lm_hp = LMHyperparams()
    payload = {
        "schema": "coe_stageB_refit_witness_v1",
        "filepath": str(data_path),
        "incumbent_root": coe_stageB_refit_ast_to_payload(
            MulNode(FreeConst("c", init=1.0), Var(0))
        ),
        "candidate_root": coe_stageB_refit_ast_to_payload(
            MulNode(FreeConst("d", init=0.1), Var(0))
        ),
        "incumbent_reuse": {},
        "candidate_reuse": {},
        "lm_hp": copy.deepcopy(lm_hp),
        "dtype": "float64",
        "device": "cpu",
        "force_cpu": True,
        "epochs": 1,
        "loss_scale": 1.0,
        "batch_size": 0,
        "y_transform_name": "identity",
        "xcoords_active": False,
        "xcoords": None,
        "trig_by_axis": None,
        "refit_tier": "tier0",
        "atom_factory": None,
        "return_row_losses": True,
        "spec": {
            "slice_id": 1,
            "train_start": 0,
            "train_stop": 40,
            "val_start": 40,
            "val_stop": 100,
        },
        "seed": 1,
    }
    row = _stageB_refit_pair_worker(payload)
    assert row["status"] == "success", row.get("error")
    assert len(row["incumbent_row_losses"]) == 60
    assert len(row["candidate_row_losses"]) == 60
    # Means of the exported rows reconstruct the scalar comparison losses.
    inc_mean = float(np.nanmean(row["incumbent_row_losses"]))
    assert inc_mean == pytest.approx(row["incumbent_compare_loss"], rel=1e-6)
