# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from nestynet_sr.run_SR import (
    _apply_stageA_provisional_guard,
    _enforce_stageA_provisional_guard_on_report,
    _stageA_ledgers_from_model_or_checkpoint,
    _stageA_provisional_confirmation_summary,
)
from nestynet_sr.sr_search.search import (
    _stageA_budgeted_witness_admission,
    _stageA_compound_shortlist_committee_rank,
    _stageA_overlap_split_committee_gate,
    _stageA_provisional_move_reason,
)


class ConstantModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value, dtype=x.dtype, device=x.device)


class FailingModel(torch.nn.Module):
    def forward(self, _x):
        raise RuntimeError("deliberate witness failure")


class SelectiveFailModel(torch.nn.Module):
    def forward(self, x):
        if bool(torch.any(x[:, 0] >= 5.0)):
            raise RuntimeError("deliberate second-slice failure")
        return torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)


def _hp(path):
    return SimpleNamespace(
        coe_mode="committee_gated",
        coe_filepath=str(path),
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


def _data(path):
    pd.DataFrame({"x0": [0.0] * 4, "y": [0.0] * 4}).to_csv(path, index=False)


def _candidate(budget):
    return {
        "z_name": "ratio",
        "kind": "monomial",
        "model": ConstantModel(2.0).double(),
        "val_loss": 4.0,
        "old_arity": 3,
        "new_arity": 2,
        "z_readable": "x0/x1",
        "structural_budget_multiplier": float(budget),
    }


def test_compound_structural_budget_is_permanent_coe_policy(tmp_path):
    path = tmp_path / "toy.csv"
    _data(path)
    base = ConstantModel(1.0).double()

    hp = _hp(path)
    hp.coe_stageA_structural_admission = "strict"  # Stale callers cannot disable policy.
    selected, reason, summary = _stageA_compound_shortlist_committee_rank(
        base_model=base,
        candidates=[_candidate(5.0)],
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert selected is not None
    assert reason == "provisional-budget-coe-stageA-compound-shortlist"
    assert summary["decision"] == "select_provisional_candidate"
    assert summary["results"][0]["budgeted_witness_admission"]["max_excess_ratio"] == 4.0


def test_budget_is_a_hard_cross_slice_guardrail(tmp_path):
    path = tmp_path / "toy.csv"
    _data(path)
    hp = _hp(path)
    selected, _, summary = _stageA_compound_shortlist_committee_rank(
        base_model=ConstantModel(1.0).double(),
        candidates=[_candidate(3.0)],
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    ok, _, split = _stageA_overlap_split_committee_gate(
        base_model=ConstantModel(1.0).double(),
        cand_model=ConstantModel(2.0).double(),
        split_kind="mul",
        has_overlap=True,
        base_val_loss=1.0,
        cand_val_loss=4.0,
        noise_floor=0.0,
        under_protest=False,
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        structural_simplification=True,
        structural_budget_multiplier=3.0,
    )

    assert selected is None and summary["decision"] == "veto_all"
    assert ok is False and split["gate_status"] == "veto"


@pytest.mark.parametrize("unavailable", ("no_file", "no_base", "x_transform", "no_slices"))
def test_compound_simplification_fails_closed_without_required_witnesses(
    tmp_path,
    unavailable,
):
    path = tmp_path / "toy.csv"
    _data(path)
    hp = _hp(path)
    base = ConstantModel(1.0).double()
    candidate = _candidate(5.0)
    if unavailable == "no_file":
        hp.coe_filepath = None
    elif unavailable == "no_base":
        base = None
    elif unavailable == "x_transform":
        candidate["model"]._x_transform = {0: "opaque"}
    else:
        hp.coe_num_slices = 0

    selected, reason, summary = _stageA_compound_shortlist_committee_rank(
        base_model=base,
        candidates=[candidate],
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert selected is None
    assert reason == "reject-coe-stageA-compound-budget-witness-unavailable"
    assert summary["gate_status"] == "veto"
    assert summary["decision"] == "veto_all"
    assert summary["budgeted_witnesses_required"] is True


def test_overlap_simplification_gets_provisional_admission_and_reporting(tmp_path):
    path = tmp_path / "toy.csv"
    _data(path)
    ok, reason, summary = _stageA_overlap_split_committee_gate(
        base_model=ConstantModel(1.0).double(),
        cand_model=ConstantModel(2.0).double(),
        split_kind="mul",
        has_overlap=True,
        base_val_loss=1.0,
        cand_val_loss=4.0,
        noise_floor=0.0,
        under_protest=False,
        lm_hp=_hp(path),
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        structural_simplification=True,
        structural_budget_multiplier=5.0,
    )
    provisional_reason = _stageA_provisional_move_reason(
        "separability_split",
        {"split_accept"},
        {"coe_provisional_budget_admission": True},
    )
    confirmation = _stageA_provisional_confirmation_summary(
        {
            "ast": None,
            "stageA_provisional_commits": [
                {"seq": 1, "active": False},
                {"seq": 2, "active": True, "candidate_loss": 4.0},
            ],
        },
        None,
    )

    assert ok is True and reason.startswith("provisional-budget-")
    assert summary["gate_status"] == "accepted_provisional"
    assert provisional_reason == "coe_budgeted_structural_admission_requires_stageB_confirmation"
    assert confirmation["total"] == 1 and confirmation["commits"][0]["seq"] == 2


def test_nonoverlap_simplification_still_requires_budgeted_witnesses(tmp_path):
    path = tmp_path / "toy.csv"
    _data(path)
    hp = _hp(path)
    hp.coe_stageA_structural_admission = "strict"  # Stale callers cannot disable policy.
    ok, reason, summary = _stageA_overlap_split_committee_gate(
        base_model=ConstantModel(1.0).double(),
        cand_model=ConstantModel(2.0).double(),
        split_kind="mul",
        has_overlap=False,
        base_val_loss=1.0,
        cand_val_loss=4.0,
        noise_floor=0.0,
        under_protest=False,
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        structural_simplification=True,
        structural_budget_multiplier=5.0,
    )

    assert ok is True and reason.startswith("provisional-budget-")
    assert summary["risk_tags"] == []
    assert summary["budgeted_witnesses_required"] is True
    assert summary["gate_status"] == "accepted_provisional"
    assert summary["provisional_budget_admission"] is True
    assert len(summary["results"]) == 1


def test_permanent_budgeted_policy_does_not_change_non_coe_search(tmp_path):
    path = tmp_path / "toy.csv"
    _data(path)
    hp = _hp(path)
    hp.coe_mode = "off"

    ok, reason, summary = _stageA_overlap_split_committee_gate(
        base_model=ConstantModel(1.0).double(),
        cand_model=ConstantModel(2.0).double(),
        split_kind="mul",
        has_overlap=False,
        base_val_loss=1.0,
        cand_val_loss=4.0,
        noise_floor=0.0,
        under_protest=False,
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        structural_simplification=True,
        structural_budget_multiplier=5.0,
    )

    assert ok is True
    assert reason == "legacy-coe-stageA-overlap-split-disabled"
    assert summary["budgeted_witnesses_required"] is False
    assert summary["results"] == []


@pytest.mark.parametrize(
    "unavailable",
    (
        "no_file",
        "missing_file",
        "no_model",
        "x_transform",
        "no_slices",
        "no_valid_slices",
    ),
)
def test_nonoverlap_simplification_fails_closed_without_required_witnesses(
    tmp_path,
    unavailable,
):
    path = tmp_path / "toy.csv"
    _data(path)
    hp = _hp(path)
    base = ConstantModel(1.0).double()
    candidate = ConstantModel(2.0).double()
    if unavailable == "no_file":
        hp.coe_filepath = None
    elif unavailable == "missing_file":
        hp.coe_filepath = str(tmp_path / "missing.csv")
    elif unavailable == "no_model":
        candidate = None
    elif unavailable == "x_transform":
        base._x_transform = {0: "opaque"}
    elif unavailable == "no_valid_slices":
        hp.coe_start_slice = 99
    else:
        hp.coe_num_slices = 0

    ok, reason, summary = _stageA_overlap_split_committee_gate(
        base_model=base,
        cand_model=candidate,
        split_kind="mul",
        has_overlap=False,
        base_val_loss=1.0,
        cand_val_loss=4.0,
        noise_floor=0.0,
        under_protest=False,
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        structural_simplification=True,
        structural_budget_multiplier=5.0,
    )

    assert ok is False
    assert reason == "reject-coe-stageA-overlap-split-budget-witness-unavailable"
    assert summary["gate_status"] == "veto"
    assert summary["decision"] == "veto"
    assert summary["budgeted_witnesses_required"] is True


def test_partial_invalid_witness_cannot_pass_as_strict_nonregression(tmp_path):
    path = tmp_path / "toy.csv"
    pd.DataFrame({"x0": list(range(12)), "y": [1.0] * 12}).to_csv(path, index=False)
    hp = _hp(path)
    hp.coe_num_slices = 2
    hp.coe_stageB_gate_slices = 2

    ok, reason, summary = _stageA_overlap_split_committee_gate(
        base_model=ConstantModel(1.0).double(),
        cand_model=SelectiveFailModel().double(),
        split_kind="mul",
        has_overlap=False,
        base_val_loss=0.0,
        cand_val_loss=0.0,
        noise_floor=0.0,
        under_protest=False,
        lm_hp=hp,
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        structural_simplification=True,
        structural_budget_multiplier=5.0,
    )

    assert ok is False and reason.startswith("reject-coe-stageA-overlap-split-gate")
    assert summary["n_paired_success"] == 1
    assert summary["invalid"] == 1
    assert summary["budgeted_witness_admission"]["invalid_slices"] == 1


def test_budgeted_overlap_rejects_when_all_witnesses_fail(tmp_path):
    path = tmp_path / "toy.csv"
    _data(path)
    ok, reason, summary = _stageA_overlap_split_committee_gate(
        base_model=ConstantModel(1.0).double(),
        cand_model=FailingModel().double(),
        split_kind="mul",
        has_overlap=True,
        base_val_loss=1.0,
        cand_val_loss=4.0,
        noise_floor=0.0,
        under_protest=False,
        lm_hp=_hp(path),
        y_op=None,
        y_op_inv=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        structural_simplification=True,
        structural_budget_multiplier=5.0,
    )

    assert ok is False
    assert reason == "reject-coe-stageA-overlap-split-budget-witness-unavailable"
    assert summary["gate_status"] == "veto"


def test_pb119_observed_vetoes_are_inside_a_25x_structural_budget():
    # Values are from the three completed pb119 compound-gate summaries.  The
    # old gate rejected every candidate because each lost all witnesses.
    observed = (
        (9.545124e-6, 1.658950e-4),
        (9.545124e-6, 6.014676e-5),
        (7.27608e-6, 6.344627e-5),
    )
    for base, candidate in observed:
        decision = _stageA_budgeted_witness_admission(
            [
                {
                    "status": "success",
                    "slice_id": 9,
                    "base": base,
                    "candidate": candidate,
                    "tolerance": 0.0,
                }
            ],
            base_loss_key="base",
            candidate_loss_key="candidate",
            budget_multiplier=25.0,
        )
        assert decision["all_slices_within_budget"] is True
        assert decision["max_excess_ratio"] < 25.0


def test_allstages_suite_has_no_structural_admission_switch(tmp_path, monkeypatch):
    from nestynet_sr import run_allstages_suite as suite

    captured = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            captured["cmd"] = cmd

        def wait(self):
            return 0

    monkeypatch.setattr(suite.subprocess, "Popen", FakeProcess)
    data_path = tmp_path / "pb000_data.csv"
    data_path.write_text("x0,y\n0,0\n", encoding="utf-8")
    result = suite.run_allstages_on_problem(
        "pb000",
        str(data_path),
        str(tmp_path),
        coe_mode="reservoir_discovery",
    )

    assert result["success"] is True
    assert "--coe_stageA_structural_admission" not in captured["cmd"]


def test_json_report_omits_stale_structural_admission_control(tmp_path):
    from nestynet_sr.run_sr_reports import write_json_report

    data_path = tmp_path / "pb000_data.csv"
    report_path = tmp_path / "report.json"
    data_path.write_text("x0,y\n0,0\n", encoding="utf-8")
    write_json_report(
        filepath=str(data_path),
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        walltime=0.0,
        stageA_data={"coe_stageA_structural_admission": "strict"},
        enable_truth_eval=False,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "coe_stageA_structural_admission" not in report["stageA"]


def test_checkpoint_resume_preserves_active_debt_and_final_guard():
    fitted = SimpleNamespace(
        _stageA_move_records=[{"seq": 7, "outcome": "provisional"}],
        _stageA_provisional_commits=[{"seq": 7, "active": True}],
        _stageA_rejection_records=[],
    )
    checkpoint = pickle.loads(
        pickle.dumps(_stageA_ledgers_from_model_or_checkpoint(model=fitted))
    )
    restored = _stageA_ledgers_from_model_or_checkpoint(
        model=SimpleNamespace(_stageA_provisional_commits=[]),
        checkpoint=checkpoint,
    )
    stage_a = {
        "ast": None,
        "val_loss": 1.0,
        **restored,
    }
    stage_a["stageA_provisional_confirmation"] = _stageA_provisional_confirmation_summary(
        stage_a, None
    )
    stage_b = {"sympy_meta": {"accepted": True}}
    guard = _apply_stageA_provisional_guard(
        args=SimpleNamespace(coe_mode="reservoir_discovery"),
        stageA_data=stage_a,
        stageB_data=stage_b,
    )

    assert restored["stageA_provisional_commits"] == [{"seq": 7, "active": True}]
    assert guard["decision"] == "mark_uncertified"
    assert stage_b["sympy_meta"]["accepted"] is False


def test_malformed_checkpoint_debt_fails_closed():
    with pytest.raises(ValueError, match="stageA_provisional_commits"):
        _stageA_ledgers_from_model_or_checkpoint(
            checkpoint={"stageA_provisional_commits": {"active": True}}
        )


def test_later_stat_selection_cannot_overwrite_unconfirmed_guard(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "statistical_selection": {"status": "certified"},
                "final_selection": {
                    "source": "statistical_selection",
                    "applied": True,
                    "eligible_for_success": True,
                    "status": "certified",
                    "expr": "x0",
                },
            }
        ),
        encoding="utf-8",
    )
    guard = {
        "decision": "mark_uncertified",
        "status": "unconfirmed",
        "reason": "active Stage-A debt",
    }
    final = _enforce_stageA_provisional_guard_on_report(str(report_path), guard)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))

    assert final["eligible_for_success"] is False
    assert persisted["final_selection"]["status"] == "stageA_provisional_unconfirmed"
    assert persisted["blocked_final_selection"]["status"] == "certified"
    assert persisted["statistical_selection"]["eligible_for_success"] is False
