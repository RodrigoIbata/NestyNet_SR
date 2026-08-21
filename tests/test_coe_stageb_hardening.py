# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import random
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import torch

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    ConstNode,
    FreeConst,
    MulNode,
    Var,
    build_composite_from_ast,
)
from nestynet_sr.sr_search.config import LMHyperparams
from nestynet_sr.sr_search.stageB.engine import Candidate, StageBContext, StageBState
from nestynet_sr.sr_search.stageB.engine import _restore_rng_state, _snapshot_rng_state
from nestynet_sr.sr_search.xcoord import XCoordSystem


def test_coe_rng_snapshot_restore_covers_python_numpy_and_torch():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    snap = _snapshot_rng_state()
    expected_py = random.random()
    expected_np = float(np.random.random())
    expected_torch = torch.rand(3)

    random.random()
    np.random.random()
    torch.rand(5)

    _restore_rng_state(snap)

    assert random.random() == expected_py
    assert float(np.random.random()) == expected_np
    assert torch.allclose(torch.rand(3), expected_torch)


def test_coe_refit_gate_restores_context_and_rng(tmp_path):
    data_path = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "x0": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    ).to_csv(data_path, index=False)

    lm_hp = SimpleNamespace(
        loss_target=1.0e-9,
        loss_acceptable=1.0,
        loss_in_MAD_units=False,
        coe_mode="committee_gated",
        coe_stageB_dry_run=True,
        coe_filepath=str(data_path),
        coe_num_slices=1,
        coe_stageB_gate_slices=1,
        coe_stageB_initial_gate_slices=1,
        coe_ndata_train=1,
        coe_ndata_val=1,
        coe_reference_slice=0,
        coe_stageB_refit_epochs=5,
        coe_stageB_refit_escalate_epochs=0,
        coe_noise_floor_raw=0.0,
        coe_noise_mult=3.0,
        coe_rel_tol=1.0e-3,
        coe_min_valid_fraction=0.8,
        coe_stageB_refit_gate=True,
        coe_witness_parallelism=2,
        fit_y_link=None,
    )
    base_model = torch.nn.Linear(1, 1)
    cand_model = torch.nn.Linear(1, 1)
    state = StageBState(root=ConstNode(1.0), model=base_model, reuse={}, val_loss=1.0)
    cand_state = StageBState(root=ConstNode(2.0), model=cand_model, reuse={}, val_loss=0.5)
    original_train = object()
    original_val = object()
    original_cache = {"sentinel": object()}
    ctx = StageBContext(
        state=state,
        train_loader=original_train,
        val_loader=original_val,
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
        dataset_ids=["reference"],
        loss_scales=[1.0],
        agg_mode="weighted",
        agg_weights=[1.0],
        _cache=original_cache,
        acceptance_noise_n_eff=1,
    )

    def fake_fit_candidate(self, cand, epochs_override=None):
        random.random()
        np.random.random()
        torch.rand(2)
        val = 0.5 if "candidate" in str(cand.label) else 1.0
        return StageBState(root=cand.root, model=torch.nn.Linear(1, 1), reuse={}, val_loss=val)

    ctx.fit_candidate = MethodType(fake_fit_candidate, ctx)
    records = []

    random.seed(321)
    np.random.seed(321)
    torch.manual_seed(321)
    snap = _snapshot_rng_state()
    expected_py = random.random()
    expected_np = float(np.random.random())
    expected_torch = torch.rand(3)
    _restore_rng_state(snap)

    ok, reason = ctx._coe_stageB_refit_committee_gate(
        rule="dummy",
        label="dummy_candidate",
        reason="accepted",
        target="root",
        target_uid="root",
        cand=Candidate(label="candidate", root=ConstNode(2.0)),
        cand_state=cand_state,
        n_params_base=1,
        n_params_cand=1,
        risk_tags=["generic_approximant"],
        gate_record_fn=lambda **kw: records.append(kw) or kw,
        incumbent_snapshot="1",
        candidate_snapshot="2",
    )

    assert ok is True
    assert "accepted" in str(reason)
    assert ctx.train_loader is original_train
    assert ctx.val_loader is original_val
    assert ctx.dataset_ids == ["reference"]
    assert ctx.loss_scales == [1.0]
    assert ctx.agg_mode == "weighted"
    assert ctx.agg_weights == [1.0]
    assert ctx._cache is original_cache
    assert records and records[-1]["gate_status"] == "refit_accepted"
    assert records[-1]["summary"]["excluded_slice_ids"] == [0]
    assert records[-1]["summary"]["witness_executor"] == {
        "backend": "serial",
        "parallelism": 2,
        "effective_backend": "serial",
        "parallel_disabled_reason": "stageB_refit_custom_fit_candidate_live_context",
    }

    assert random.random() == expected_py
    assert float(np.random.random()) == expected_np
    assert torch.allclose(torch.rand(3), expected_torch)


def test_coe_refit_gate_replays_x_coordinate_transform(tmp_path):
    data_path = tmp_path / "toy_xcoords.csv"
    pd.DataFrame(
        {
            "x0": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 4.0, 9.0],
        }
    ).to_csv(data_path, index=False)

    lm_hp = SimpleNamespace(
        loss_target=1.0e-9,
        loss_acceptable=1.0,
        loss_in_MAD_units=False,
        coe_mode="committee_gated",
        coe_stageB_dry_run=True,
        coe_filepath=str(data_path),
        coe_num_slices=1,
        coe_stageB_gate_slices=1,
        coe_stageB_initial_gate_slices=1,
        coe_ndata_train=1,
        coe_ndata_val=1,
        coe_reference_slice=0,
        coe_stageB_refit_epochs=5,
        coe_stageB_refit_escalate_epochs=0,
        coe_noise_floor_raw=0.0,
        coe_noise_mult=3.0,
        coe_rel_tol=1.0e-3,
        coe_min_valid_fraction=0.8,
        coe_stageB_refit_gate=True,
        coe_witness_parallelism=2,
        fit_y_link=None,
    )
    xcoords = XCoordSystem.from_map({0: {"pipeline": [{"kind": "square"}]}}, Nx_raw=1)
    state = StageBState(root=ConstNode(1.0), model=torch.nn.Linear(1, 1), reuse={}, val_loss=1.0)
    cand_state = StageBState(root=ConstNode(2.0), model=torch.nn.Linear(1, 1), reuse={}, val_loss=0.5)
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
        dataset_ids=["reference"],
        stageA_x_transforms={0: {"pipeline": [{"kind": "square"}]}},
        xcoords=xcoords,
        xcoords_applied=True,
        acceptance_noise_n_eff=1,
    )

    seen_train_x = []

    def fake_fit_candidate(self, cand, epochs_override=None):
        batch_x, _batch_y = next(iter(self.train_loader))
        seen_train_x.append(float(batch_x[0, 0]))
        val = 0.5 if "candidate" in str(cand.label) else 1.0
        return StageBState(root=cand.root, model=torch.nn.Linear(1, 1), reuse={}, val_loss=val)

    ctx.fit_candidate = MethodType(fake_fit_candidate, ctx)
    records = []

    ok, reason = ctx._coe_stageB_refit_committee_gate(
        rule="dummy",
        label="dummy_candidate",
        reason="accepted",
        target="root",
        target_uid="root",
        cand=Candidate(label="candidate", root=ConstNode(2.0)),
        cand_state=cand_state,
        n_params_base=1,
        n_params_cand=1,
        risk_tags=["generic_approximant"],
        gate_record_fn=lambda **kw: records.append(kw) or kw,
        incumbent_snapshot="1",
        candidate_snapshot="2",
    )

    assert ok is True
    assert "accepted" in str(reason)
    assert seen_train_x == [4.0, 4.0]
    assert records[-1]["summary"]["x_transform_active"] is True
    assert records[-1]["summary"]["x_coordinate_space"] == "internal_x"
    assert (
        records[-1]["summary"]["witness_executor"]["parallel_disabled_reason"]
        == "stageB_refit_custom_fit_candidate_live_context"
    )


def test_coe_refit_gate_uses_process_payload_for_replayable_same_history(tmp_path):
    data_path = tmp_path / "toy_process.csv"
    xs = list(range(8))
    pd.DataFrame({"x0": [float(x) for x in xs], "y": [float(x) for x in xs]}).to_csv(
        data_path,
        index=False,
    )

    lm_hp = LMHyperparams(
        epochs=1,
        epochs_min=0,
        nval_patience=1,
        log_to_console=False,
        LM_verbose=False,
        loss_target=1.0e-12,
        loss_acceptable=1.0,
        loss_in_MAD_units=False,
        fit_y_link=None,
    )
    for key, value in {
        "coe_mode": "committee_gated",
        "coe_stageB_dry_run": True,
        "coe_filepath": str(data_path),
        "coe_num_slices": 2,
        "coe_stageB_gate_slices": 1,
        "coe_stageB_initial_gate_slices": 1,
        "coe_ndata_train": 2,
        "coe_ndata_val": 2,
        "coe_reference_slice": 0,
        "coe_stageB_refit_epochs": 1,
        "coe_stageB_refit_escalate_epochs": 0,
        "coe_noise_floor_raw": 0.0,
        "coe_noise_mult": 3.0,
        "coe_rel_tol": 1.0e-6,
        "coe_min_valid_fraction": 0.8,
        "coe_stageB_refit_gate": True,
        "coe_witness_parallelism": 2,
    }.items():
        setattr(lm_hp, key, value)
    state = StageBState(root=Var(0), model=torch.nn.Identity(), reuse={}, val_loss=0.0)
    cand_root = MulNode(ConstNode(2.0), Var(0))
    cand_state = StageBState(root=cand_root, model=torch.nn.Identity(), reuse={}, val_loss=1.0)
    ctx = StageBContext(
        state=state,
        train_loader=object(),
        val_loader=object(),
        lm_hp=lm_hp,
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-12,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        dataset_ids=["reference"],
        acceptance_noise_n_eff=2,
    )
    records = []

    ok, reason = ctx._coe_stageB_refit_committee_gate(
        rule="dummy",
        label="bad_scale",
        reason="accepted",
        target="root",
        target_uid="root",
        cand=Candidate(label="candidate", root=cand_root),
        cand_state=cand_state,
        n_params_base=0,
        n_params_cand=0,
        risk_tags=[],
        gate_record_fn=lambda **kw: records.append(kw) or kw,
        incumbent_snapshot="x0",
        candidate_snapshot="2*x0",
    )

    assert ok is False
    assert "reject-coe-stageB-refit-gate" in str(reason)
    assert records[-1]["gate_status"] == "refit_veto"
    executor_meta = records[-1]["summary"]["witness_executor"]
    assert executor_meta["parallelism"] == 2
    assert executor_meta["process_payload_attempted"] is True
    assert executor_meta["effective_backend"] in {"process", "serial"}
    if executor_meta["effective_backend"] == "serial":
        assert records[-1]["results"][0].get("executor_fallback_reason")


def test_coe_fixed_expression_gate_rewrites_internal_x_to_raw(tmp_path):
    data_path = tmp_path / "toy_fixed_xcoords.csv"
    pd.DataFrame(
        {
            "x0": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 4.0, 9.0],
        }
    ).to_csv(data_path, index=False)

    lm_hp = SimpleNamespace(
        loss_target=1.0e-9,
        loss_acceptable=1.0,
        loss_in_MAD_units=False,
        fit_y_link=None,
    )
    xcoords = XCoordSystem.from_map({0: {"pipeline": [{"kind": "square"}]}}, Nx_raw=1)
    state = StageBState(
        root=AtomNode(kind="var", var_idxs=(0,)),
        model=torch.nn.Linear(1, 1),
        reuse={},
        val_loss=0.0,
    )
    cand_state = StageBState(root=ConstNode(0.0), model=torch.nn.Linear(1, 1), reuse={}, val_loss=1.0)
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
        stageA_x_transforms={0: {"pipeline": [{"kind": "square"}]}},
        xcoords=xcoords,
        xcoords_applied=True,
        coe_mode="committee_gated",
        coe_filepath=str(data_path),
        coe_num_slices=1,
        coe_stageB_gate_slices=1,
        coe_stageB_initial_gate_slices=1,
        coe_ndata_train=1,
        coe_ndata_val=1,
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

    assert ok is False
    assert "reject-coe-stageB-gate" in str(reason)
    assert ctx.coe_stageB_gate_log[-1]["incumbent_expr"] == "x0**2"
    assert ctx.coe_stageB_gate_log[-1]["summary"]["x_transform_active"] is True


def test_coe_fixed_expression_gate_scores_named_candidate_coefficients(tmp_path):
    data_path = tmp_path / "toy_named_coefficients.csv"
    pd.DataFrame(
        {
            "x0": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 2.0, 4.0, 6.0],
        }
    ).to_csv(data_path, index=False)

    lm_hp = SimpleNamespace(
        loss_target=1.0e-9,
        loss_acceptable=1.0,
        loss_in_MAD_units=False,
        fit_y_link=None,
    )
    incumbent_root = MulNode(FreeConst("c", init=2.0), Var(0))
    candidate_root = MulNode(FreeConst("d", init=3.0), Var(0))
    incumbent_state = StageBState(
        root=incumbent_root,
        model=build_composite_from_ast(incumbent_root, dtype=torch.float64),
        reuse={},
        val_loss=0.0,
    )
    candidate_state = StageBState(
        root=candidate_root,
        model=build_composite_from_ast(candidate_root, dtype=torch.float64),
        reuse={},
        val_loss=1.0,
    )
    ctx = StageBContext(
        state=incumbent_state,
        train_loader=object(),
        val_loader=object(),
        lm_hp=lm_hp,
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        coe_mode="committee_gated",
        coe_filepath=str(data_path),
        coe_num_slices=1,
        coe_stageB_gate_slices=1,
        coe_stageB_initial_gate_slices=1,
        coe_ndata_train=1,
        coe_ndata_val=1,
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

    ok, reason = ctx.coe_stageB_committee_gate(
        rule="dummy",
        label="wrong_named_coefficient",
        reason="accepted",
        target="root",
        target_uid="root",
        cand=Candidate(label="candidate", root=candidate_root),
        cand_state=candidate_state,
        n_params_base=1,
        n_params_cand=1,
    )

    gate = ctx.coe_stageB_gate_log[-1]
    assert ok is False
    assert "reject-coe-stageB-gate" in str(reason)
    assert gate["gate_status"] == "veto"
    assert gate["summary"]["n_paired_success"] == 1
    assert gate["incumbent_expr"] == "(c * x0)"
    assert gate["candidate_expr"] == "(d * x0)"
