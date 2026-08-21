# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.run_SR import (
    _apply_stageA_provisional_guard,
    _build_coe_stageA_dry_run_records,
    _run_coe_stageA_exit_audit,
    _stageA_provisional_confirmation_summary,
)
import pandas as pd
import torch

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    FreeConst,
    MulNode,
    Var,
    build_composite_from_ast,
)
from types import SimpleNamespace


def _stageA_with_provisional():
    return {
        "stageA_status": "compound_unresolved",
        "y_op_name": "identity",
        "fit_y_link": None,
        "ast": AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf", inputs=(Var(0), Var(1))),
        "val_loss": 1.0e-6,
        "stageA_provisional_commits": [
            {
                "seq": 1,
                "move_kind": "compound_exhaustion",
                "provisional": True,
                "provisional_reason": "full_compound_compression_requires_stageB_confirmation",
                "candidate_loss": 1.0e-6,
            }
        ],
    }


def test_stageA_provisional_confirmation_pending_without_stageB():
    summary = _stageA_provisional_confirmation_summary(_stageA_with_provisional(), None)

    assert summary["enabled"] is True
    assert summary["confirmed"] == 0
    assert summary["unconfirmed"] == 1
    assert summary["status"] == "pending_no_stageB"
    assert summary["commits"][0]["confirmation_status"] == "pending_no_stageB"


def test_stageA_provisional_confirmation_terminal_stageB_confirms():
    summary = _stageA_provisional_confirmation_summary(
        _stageA_with_provisional(),
        {
            "val_loss": 1.1e-6,
            "num_nn_atoms": 0,
            "num_multivar_nn_atoms": 0,
            "max_nn_arity": 0,
        },
    )

    assert summary["confirmed"] == 1
    assert summary["unconfirmed"] == 0
    assert summary["status"] == "confirmed_terminal_stageB"
    assert summary["commits"][0]["confirmed_by_downstream"] is True


def test_stageA_provisional_confirmation_burden_reduction_confirms():
    summary = _stageA_provisional_confirmation_summary(
        _stageA_with_provisional(),
        {
            "val_loss": 1.1e-6,
            "num_nn_atoms": 1,
            "num_multivar_nn_atoms": 0,
            "max_nn_arity": 1,
        },
    )

    assert summary["confirmed"] == 1
    assert summary["status"] == "confirmed_stageB_burden_reduction"


def test_stageA_provisional_confirmation_same_burden_stays_pending():
    summary = _stageA_provisional_confirmation_summary(
        _stageA_with_provisional(),
        {
            "val_loss": 1.1e-6,
            "num_nn_atoms": 1,
            "num_multivar_nn_atoms": 1,
            "max_nn_arity": 2,
            "decision_log_summary": {"accept": 0},
        },
    )

    assert summary["confirmed"] == 0
    assert summary["unconfirmed"] == 1
    assert summary["status"] == "pending_unconfirmed"


def test_stageA_dry_run_records_include_provisional_confirmation_summary():
    stageA_data = _stageA_with_provisional()
    stageA_data["stageA_provisional_confirmation"] = {
        "enabled": True,
        "total": 1,
        "confirmed": 0,
        "unconfirmed": 1,
        "status": "pending_unconfirmed",
        "reason": "needs downstream confirmation",
    }

    records = _build_coe_stageA_dry_run_records(stageA_data)
    provisional_rows = [
        row for row in records if row.get("mode") == "stageA_provisional_confirmation"
    ]

    assert len(provisional_rows) == 1
    assert provisional_rows[0]["outcome"] == "pending"
    assert provisional_rows[0]["committee_status"] == "pending_unconfirmed"


def test_stageA_dry_run_recognizes_budgeted_split_as_supported_real_gate():
    records = _build_coe_stageA_dry_run_records(
        {
            "stageA_status": "split_confirmed",
            "stageA_move_records": [
                {
                    "seq": 1,
                    "move_kind": "separability_split",
                    "provisional": True,
                    "details": {
                        "coe_stageA_overlap_split_gate": {
                            "gate_status": "accepted_provisional",
                            "decision": "allow_provisional",
                        }
                    },
                }
            ],
        }
    )
    move = next(row for row in records if row.get("mode") == "stageA_move_record")

    assert move["committee_status"] == "accepted_provisional"
    assert move["outcome"] == "allow_provisional"
    assert move["real_gate_supported"] is True


def test_stageA_dry_run_recognizes_budgeted_compound_as_supported_real_gate():
    records = _build_coe_stageA_dry_run_records(
        {
            "stageA_status": "compound_unresolved",
            "coe_stageA_compound_shortlist": {
                "gate_status": "accepted_provisional",
                "decision": "select_provisional_candidate",
                "legacy_selected": "legacy",
                "selected": "ratio",
            },
        }
    )
    compound = next(
        row for row in records if row.get("mode") == "stageA_compound_shortlist_rank"
    )

    assert compound["committee_status"] == "accepted_provisional"
    assert compound["outcome"] == "select_provisional_candidate"
    assert compound["real_gate_supported"] is True
    assert compound["would_change_decision"] is True


def test_stageA_provisional_guard_is_noop_outside_coe_gate_modes():
    stageA_data = {
        "stageA_provisional_confirmation": {
            "enabled": True,
            "total": 1,
            "confirmed": 0,
            "unconfirmed": 1,
            "status": "pending_unconfirmed",
        }
    }
    stageB_data = {"sympy_meta": {"accepted": True, "parse_success": True}}

    guard = _apply_stageA_provisional_guard(
        args=SimpleNamespace(coe_mode="off"),
        stageA_data=stageA_data,
        stageB_data=stageB_data,
    )

    assert guard["enabled"] is False
    assert stageB_data["sympy_meta"]["accepted"] is True
    assert "stageA_provisional_guard" not in stageB_data


def test_stageA_provisional_guard_marks_uncertified_in_coe_gate_mode():
    stageA_data = {
        "stageA_provisional_confirmation": {
            "enabled": True,
            "total": 1,
            "confirmed": 0,
            "unconfirmed": 1,
            "status": "pending_unconfirmed",
        }
    }
    stageB_data = {"sympy_meta": {"accepted": True, "parse_success": True}}

    guard = _apply_stageA_provisional_guard(
        args=SimpleNamespace(coe_mode="committee_gated"),
        stageA_data=stageA_data,
        stageB_data=stageB_data,
    )

    assert guard["decision"] == "mark_uncertified"
    assert stageB_data["sympy_meta"]["accepted"] is False
    assert stageB_data["sympy_meta"]["raw_accepted_before_stageA_provisional_guard"] is True
    assert stageB_data["sympy_meta"]["reason"] == "stageA_provisional_unconfirmed"


def test_stageA_provisional_guard_allows_confirmed_commits_in_coe_gate_mode():
    stageA_data = {
        "stageA_provisional_confirmation": {
            "enabled": True,
            "total": 1,
            "confirmed": 1,
            "unconfirmed": 0,
            "status": "confirmed_terminal_stageB",
        }
    }
    stageB_data = {"sympy_meta": {"accepted": True, "parse_success": True}}

    guard = _apply_stageA_provisional_guard(
        args=SimpleNamespace(coe_mode="reservoir_discovery"),
        stageA_data=stageA_data,
        stageB_data=stageB_data,
    )

    assert guard["decision"] == "allow"
    assert guard["status"] == "confirmed"
    assert stageB_data["sympy_meta"]["accepted"] is True


def test_stageA_exit_audit_marks_local_analytic_leaf_as_nonportable(tmp_path):
    summary = _run_coe_stageA_exit_audit(
        args=SimpleNamespace(
            coe_mode="committee_gated",
            coe_stageA_dry_run=False,
            data_slice=0,
            coe_num_slices=3,
            coe_stageB_gate_slices=3,
            ndata_train=10,
            ndata_val=10,
        ),
        filepath=tmp_path / "unused.csv",
        results_dir=str(tmp_path),
        base_filename="case",
        stageA_data={
            "stageA_status": "unresolved",
            "y_op_name": "identity",
            "fit_y_link": None,
            "initial_ast": AtomNode(
                kind="nn", var_idxs=(0,), tag="n0", inputs=(Var(0),)
            ),
            "ast": AtomNode(kind="poly", var_idxs=(0,), tag="p0", inputs=(Var(0),)),
        },
        noise_sigma_y=0.0,
        y_op_inv=None,
    )

    assert summary["status"] == "unsupported"
    assert summary["committee_status"] == "unsupported-nonportable-fixed-expression"
    assert summary["candidate_expr"] == "poly(x0)"


def test_stageA_exit_audit_scores_named_candidate_coefficients(tmp_path):
    data_path = tmp_path / "named_coefficients.csv"
    pd.DataFrame(
        {
            "x0": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 2.0, 4.0, 6.0],
        }
    ).to_csv(data_path, index=False)
    initial_ast = MulNode(FreeConst("c", init=2.0), Var(0))
    final_ast = MulNode(FreeConst("d", init=3.0), Var(0))
    initial_model = build_composite_from_ast(initial_ast, dtype=torch.float64)
    final_model = build_composite_from_ast(final_ast, dtype=torch.float64)

    summary = _run_coe_stageA_exit_audit(
        args=SimpleNamespace(
            coe_mode="committee_gated",
            coe_stageA_dry_run=False,
            data_slice=0,
            coe_num_slices=1,
            coe_stageB_gate_slices=1,
            coe_start_slice=0,
            ndata_train=1,
            ndata_val=1,
            coe_witness_parallelism=1,
            coe_min_valid_fraction=0.8,
            coe_noise_mult=3.0,
            coe_rel_tol=1.0e-3,
        ),
        filepath=data_path,
        results_dir=str(tmp_path),
        base_filename="case",
        stageA_data={
            "stageA_status": "analytic",
            "y_op_name": "identity",
            "fit_y_link": None,
            "initial_ast": initial_ast,
            "ast": final_ast,
        },
        noise_sigma_y=0.0,
        y_op_inv=None,
        initial_model=initial_model,
        final_model=final_model,
    )

    assert summary["status"] == "evaluated"
    assert summary["committee_status"] == "evaluated-paired"
    assert summary["candidate_n_success"] == 1
    assert summary["incumbent_n_success"] == 1
    assert summary["losses"] == 1
