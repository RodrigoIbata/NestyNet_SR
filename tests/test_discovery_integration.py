# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode, Var
from nestynet_sr.discovery.integration import (
    deserialize_committee_members,
    discovery_summary_from_payload,
    run_closed_loop_from_discovery_payload,
    run_sr_discovery_integration,
)
from nestynet_sr.run_SR import write_json_report


class _WitnessModel:
    def __call__(self, x):
        return (x[:, :1] * x[:, :1]) + 2.0 * x[:, 1:2]

    def grad(self, x):
        grad = torch.stack((2.0 * x[:, 0], 2.0 * torch.ones_like(x[:, 1])), dim=1)
        return grad.unsqueeze(1)

    def grad_grad(self, x):
        out = torch.zeros((int(x.shape[0]), 1, int(x.shape[1]), int(x.shape[1])), dtype=x.dtype)
        out[:, 0, 0, 0] = 2.0
        return out


class _LinearXModel:
    def __call__(self, x):
        return x[:, :1]


class _ReverseXModel:
    def __call__(self, x):
        return 2.0 - x[:, :1]


def test_run_sr_discovery_integration_builds_committee_summary(monkeypatch, tmp_path):
    def _fake_predict(candidate, x):
        if candidate.member.member_id == "stageA_final":
            return torch.zeros(x.shape[0], dtype=x.dtype)
        if candidate.member.member_id == "stageB_final":
            return torch.ones(x.shape[0], dtype=x.dtype)
        return None

    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        _fake_predict,
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": "x0",
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": "(x0 + x1)",
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [
                {
                    "step": 1,
                    "expression": "(x0 + x1)",
                    "mse_raw": 0.12,
                    "mse_eff": 0.13,
                    "n_params": 5,
                    "ast_cost": 3,
                    "action": "rewrite add",
                },
                {
                    "step": 2,
                    "expression": "(x0 - x1)",
                    "mse_raw": 0.11,
                    "mse_eff": 0.12,
                    "n_params": 5,
                    "ast_cost": 3,
                    "action": "rewrite sub",
                },
            ],
        },
        final_model=object(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=object(), x_transform_map=None),
        committee_topk=4,
        nvars=2,
        dtype=torch.float64,
    )

    assert payload["mode"] == "sr_discovery_integration"
    assert payload["committee_summary"]["member_count"] >= 2
    assert payload["physics_summary"]
    assert payload["runtime_summary"]["candidate_count"] >= 2


def test_run_sr_discovery_integration_selects_high_disagreement_experiment(monkeypatch, tmp_path):
    def _fake_predict(candidate, x):
        if candidate.member.member_id == "stageA_final":
            return x[:, 0]
        if candidate.member.member_id == "stageB_final":
            return -x[:, 0]
        return None

    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        _fake_predict,
    )

    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "flat", "type": "points", "points": [[0.0, 0.0]]},
                    {"experiment_id": "spread", "type": "points", "points": [[1.0, 0.0], [2.0, 0.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": "x0",
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": "(x0 + x1)",
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [],
        },
        final_model=object(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=object(), x_transform_map=None),
        committee_topk=4,
        experiment_manifest_path=str(manifest_path),
        beta=0.0,
        gamma=0.0,
        nvars=2,
        dtype=torch.float64,
    )

    assert payload["experiment_selection"]["selected"]["experiment_id"] == "spread"
    assert payload["experiment_candidates"]
    summary = discovery_summary_from_payload(
        payload,
        results_path=tmp_path / "toy.discovery.json",
    )
    assert summary["enabled"] is True
    assert summary["selected_experiment"]["experiment_id"] == "spread"


def test_run_sr_discovery_integration_witness_mode_sees_shape_disagreement(monkeypatch, tmp_path):
    def _fake_predict(candidate, x):
        if candidate.member.member_id == "stageA_final":
            return x[:, 0]
        if candidate.member.member_id == "stageB_final":
            return 2.0 - x[:, 0]
        return None

    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        _fake_predict,
    )

    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "shape_agree", "type": "points", "points": [[1.0, 0.0], [1.0, 0.0]]},
                    {"experiment_id": "shape_split", "type": "points", "points": [[0.0, 0.0], [2.0, 0.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": ("var", 0),
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": ("sub", ("const", 2.0), ("var", 0)),
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [],
        },
        final_model=object(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=object(), x_transform_map=None),
        committee_topk=4,
        experiment_manifest_path=str(manifest_path),
        beta=0.0,
        gamma=0.0,
        disagreement_mode="witness",
        nvars=2,
        dtype=torch.float64,
    )

    assert payload["config"]["disagreement_mode"] == "witness"
    assert payload["experiment_selection"]["disagreement_mode"] == "witness"
    assert payload["experiment_selection"]["selected"]["experiment_id"] == "shape_split"


def test_run_sr_discovery_integration_defaults_to_witness_mode(monkeypatch, tmp_path):
    def _fake_predict(candidate, x):
        if candidate.member.member_id == "stageA_final":
            return x[:, 0]
        if candidate.member.member_id == "stageB_final":
            return 2.0 - x[:, 0]
        return None

    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        _fake_predict,
    )

    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "shape_agree", "type": "points", "points": [[1.0, 0.0], [1.0, 0.0]]},
                    {"experiment_id": "shape_split", "type": "points", "points": [[0.0, 0.0], [2.0, 0.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": ("var", 0),
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": ("sub", ("const", 2.0), ("var", 0)),
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [],
        },
        final_model=object(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=object(), x_transform_map=None),
        committee_topk=4,
        experiment_manifest_path=str(manifest_path),
        beta=0.0,
        gamma=0.0,
        nvars=2,
        dtype=torch.float64,
    )

    assert payload["config"]["disagreement_mode"] == "witness"
    assert payload["experiment_selection"]["disagreement_mode"] == "witness"
    assert payload["experiment_selection"]["selected"]["experiment_id"] == "shape_split"


def test_run_sr_discovery_integration_explicit_legacy_profile_uses_witness_default(monkeypatch, tmp_path):
    def _fake_predict(candidate, x):
        if candidate.member.member_id == "stageA_final":
            return x[:, 0]
        if candidate.member.member_id == "stageB_final":
            return 2.0 - x[:, 0]
        return None

    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        _fake_predict,
    )

    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "shape_agree", "type": "points", "points": [[1.0, 0.0], [1.0, 0.0]]},
                    {"experiment_id": "shape_split", "type": "points", "points": [[0.0, 0.0], [2.0, 0.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": ("var", 0),
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": ("sub", ("const", 2.0), ("var", 0)),
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [],
        },
        final_model=object(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=object(), x_transform_map=None),
        committee_topk=4,
        experiment_manifest_path=str(manifest_path),
        research_profile="legacy",
        beta=0.0,
        gamma=0.0,
        nvars=2,
        dtype=torch.float64,
    )

    assert payload["config"]["research_profile"] == "legacy"
    assert payload["config"]["disagreement_mode"] == "witness"
    assert payload["experiment_selection"]["disagreement_mode"] == "witness"
    assert payload["experiment_selection"]["selected"]["experiment_id"] == "shape_split"


def test_run_sr_discovery_integration_retired_scalar_request_uses_witness(monkeypatch, tmp_path):
    def _fake_predict(candidate, x):
        if candidate.member.member_id == "stageA_final":
            return x[:, 0]
        if candidate.member.member_id == "stageB_final":
            return 2.0 - x[:, 0]
        return None

    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        _fake_predict,
    )

    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "shape_agree", "type": "points", "points": [[1.0, 0.0], [1.0, 0.0]]},
                    {"experiment_id": "shape_split", "type": "points", "points": [[0.0, 0.0], [2.0, 0.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": ("var", 0),
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": ("sub", ("const", 2.0), ("var", 0)),
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [],
        },
        final_model=object(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=object(), x_transform_map=None),
        committee_topk=4,
        experiment_manifest_path=str(manifest_path),
        disagreement_mode="scalar",
        beta=0.0,
        gamma=0.0,
        nvars=2,
        dtype=torch.float64,
    )

    assert payload["config"]["research_profile"] == "default"
    assert payload["config"]["disagreement_mode"] == "witness"
    assert payload["experiment_selection"]["disagreement_mode"] == "witness"
    assert payload["experiment_selection"]["selected"]["experiment_id"] == "shape_split"


def test_write_json_report_includes_discovery_summary(tmp_path):
    report_path = tmp_path / "toy.report.json"
    write_json_report(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        walltime=0.0,
        stageA_data=None,
        stageB_data=None,
        de_data=None,
        discovery_summary={
            "enabled": True,
            "results_path": str(tmp_path / "toy.discovery.json"),
            "committee_member_count": 2,
        },
        enable_truth_eval=False,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["discovery"]["enabled"] is True
    assert payload["discovery"]["committee_member_count"] == 2


def test_run_sr_discovery_integration_populates_shared_and_local_constants(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        lambda candidate, x: torch.zeros(x.shape[0], dtype=x.dtype),
    )

    class_sr_result = SimpleNamespace(
        class_tags=["shared_leaf"],
        experiment_tags=["local_leaf"],
        class_params={"shared_leaf": torch.tensor([2.0], dtype=torch.float64)},
        experiment_params=[
            {"local_leaf": torch.tensor([3.0], dtype=torch.float64)},
            {"local_leaf": torch.tensor([5.0], dtype=torch.float64)},
        ],
        val_losses=[0.1, 0.2],
        val_loss_agg=0.15,
        val_loss_agg_mode="mean",
    )
    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["d0.csv", "d1.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data=None,
        stageB_data={
            "ast": "shared_leaf*x0 + local_leaf",
            "val_loss": 0.1,
            "params": 2,
            "dataset_ids": ["d0", "d1"],
            "simplification_path": [],
        },
        final_model=None,
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=None, x_transform_map=None, dataset_ids=["d0", "d1"]),
        class_sr_result=class_sr_result,
        committee_topk=2,
        nvars=1,
        dtype=torch.float64,
    )

    stageb_member = next(row for row in payload["committee_members"] if row["member_id"] == "stageB_final")
    assert stageb_member["shared_constants"]["shared_leaf"] == 2.0
    assert stageb_member["local_constants_by_experiment"]["d0"]["local_leaf"] == 3.0
    assert stageb_member["local_constants_by_experiment"]["d1"]["local_leaf"] == 5.0
    assert payload["class_sr_summary"]["class_tags"] == ["shared_leaf"]


def test_run_sr_discovery_integration_emits_constant_lift_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        lambda candidate, x: torch.zeros(x.shape[0], dtype=x.dtype),
    )

    def fake_solve_constant_lift_task(**kwargs):
        return {
            "solver": "factorized_search",
            "expr": "x0",
            "expr_ast": ["var", 0],
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            "fit_mse": 1.0e-4,
            "probe_mse": 1.0e-4,
            "baseline_mse": 1.0,
            "improvement_ratio": 10.0,
            "regime_ids": ["d0", "d1", "d2"],
            "regime_points": [[1.0], [2.0], [3.0]],
            "regime_values": [1.0, 2.0, 4.0],
            "feature_names": ["temperature"],
            "feature_source": "dataset_metadata",
        }

    monkeypatch.setattr(
        "nestynet_sr.discovery.constant_lift.solve_constant_lift_task",
        fake_solve_constant_lift_task,
    )

    class_sr_result = SimpleNamespace(
        class_tags=["shared_leaf"],
        experiment_tags=["local_leaf"],
        class_params={"shared_leaf": torch.tensor([2.0], dtype=torch.float64)},
        experiment_params=[
            {"local_leaf": torch.tensor([1.0], dtype=torch.float64)},
            {"local_leaf": torch.tensor([2.0], dtype=torch.float64)},
            {"local_leaf": torch.tensor([4.0], dtype=torch.float64)},
        ],
        val_losses=[0.1, 0.2, 0.3],
        val_loss_agg=0.2,
        val_loss_agg_mode="mean",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["d0.csv", "d1.csv", "d2.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data=None,
        stageB_data={
            "ast": "shared_leaf*x0 + local_leaf",
            "val_loss": 0.1,
            "params": 2,
            "dataset_ids": ["d0", "d1", "d2"],
            "dataset_metadata": [
                {"temperature": 1.0},
                {"temperature": 2.0},
                {"temperature": 3.0},
            ],
            "simplification_path": [],
        },
        final_model=None,
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=None, x_transform_map=None, dataset_ids=["d0", "d1", "d2"]),
        class_sr_result=class_sr_result,
        committee_topk=2,
        nvars=1,
        dtype=torch.float64,
        discovery_constant_lift_enable=True,
        discovery_constant_lift_min_regimes=3,
        discovery_constant_lift_trigger_mean_cv=0.2,
    )

    assert payload["config"]["discovery_constant_lift_enable"] is True
    assert payload["constant_lift_summary"]["proposal_count"] == 1
    proposal = payload["constant_lift_summary"]["members"][0]["proposals"][0]
    assert proposal["constant_name"] == "local_leaf"
    assert proposal["feature_source"] == "dataset_metadata"
    physics = payload["physics_summary"]["stageB_final"]["checks"]["parameter_stability"]
    assert physics["passed"] is False
    assert physics["details"]["parameter_cvs"]["local_leaf"] > 0.2


def test_run_sr_discovery_integration_applies_constant_lift_proposals(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        lambda candidate, x: torch.zeros(x.shape[0], dtype=x.dtype),
    )

    def fake_solve_constant_lift_task(**kwargs):
        return {
            "solver": "factorized_search",
            "expr": "x0",
            "expr_ast": ["var", 0],
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            "fit_mse": 1.0e-4,
            "probe_mse": 1.0e-4,
            "baseline_mse": 1.0,
            "improvement_ratio": 10.0,
            "regime_ids": ["d0", "d1", "d2"],
            "regime_points": [[1.0], [2.0], [3.0]],
            "regime_values": [1.0, 2.0, 4.0],
            "feature_names": ["temperature"],
            "feature_source": "dataset_metadata",
        }

    monkeypatch.setattr(
        "nestynet_sr.discovery.constant_lift.solve_constant_lift_task",
        fake_solve_constant_lift_task,
    )

    class_sr_result = SimpleNamespace(
        class_tags=["shared_leaf"],
        experiment_tags=["local_leaf"],
        class_params={"shared_leaf": torch.tensor([2.0], dtype=torch.float64)},
        experiment_params=[
            {"local_leaf": torch.tensor([1.0], dtype=torch.float64)},
            {"local_leaf": torch.tensor([2.0], dtype=torch.float64)},
            {"local_leaf": torch.tensor([4.0], dtype=torch.float64)},
        ],
        val_losses=[0.1, 0.2, 0.3],
        val_loss_agg=0.2,
        val_loss_agg_mode="mean",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["d0.csv", "d1.csv", "d2.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data=None,
        stageB_data={
            "ast": "shared_leaf*x0 + local_leaf",
            "val_loss": 0.1,
            "params": 2,
            "dataset_ids": ["d0", "d1", "d2"],
            "dataset_metadata": [
                {"temperature": 1.0},
                {"temperature": 2.0},
                {"temperature": 3.0},
            ],
            "simplification_path": [],
        },
        final_model=None,
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=None, x_transform_map=None, dataset_ids=["d0", "d1", "d2"]),
        class_sr_result=class_sr_result,
        committee_topk=2,
        nvars=1,
        dtype=torch.float64,
        discovery_constant_lift_enable=True,
        discovery_constant_lift_min_regimes=3,
        discovery_constant_lift_trigger_mean_cv=0.2,
        discovery_constant_lift_apply_enable=True,
        discovery_constant_lift_apply_topk=1,
        discovery_constant_lift_min_rel_gain=2.0,
    )

    assert payload["config"]["discovery_constant_lift_apply_enable"] is True
    assert payload["constant_lift_summary"]["applied_member_count"] == 1
    assert payload["constant_lift_summary"]["surviving_applied_member_count"] == 1
    lifted_member = next(
        row
        for row in payload["committee_members"]
        if row["metadata"].get("constant_lift_applied", False)
    )
    assert lifted_member["metadata"]["constant_lift_parent_member_id"] == "stageB_final"
    assert lifted_member["display_expr"] == "shared_leaf*x0 + (temperature(x0))"
    assert lifted_member["local_constants_by_experiment"] == {}


def test_run_sr_discovery_integration_serializes_ast_applied_constant_lift(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        lambda candidate, x: torch.zeros(x.shape[0], dtype=x.dtype),
    )

    def fake_solve_constant_lift_task(**kwargs):
        return {
            "solver": "factorized_search",
            "expr": "x0",
            "expr_ast": ["var", 0],
            "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            "fit_mse": 1.0e-4,
            "probe_mse": 1.0e-4,
            "baseline_mse": 1.0,
            "improvement_ratio": 10.0,
            "regime_ids": ["d0", "d1", "d2"],
            "regime_points": [[1.0], [2.0], [3.0]],
            "regime_values": [1.0, 2.0, 4.0],
            "feature_names": ["temperature"],
            "feature_source": "dataset_metadata",
        }

    monkeypatch.setattr(
        "nestynet_sr.discovery.constant_lift.solve_constant_lift_task",
        fake_solve_constant_lift_task,
    )

    class_sr_result = SimpleNamespace(
        class_tags=["shared_leaf"],
        experiment_tags=["local_leaf"],
        class_params={"shared_leaf": torch.tensor([2.0], dtype=torch.float64)},
        experiment_params=[
            {"local_leaf": torch.tensor([1.0], dtype=torch.float64)},
            {"local_leaf": torch.tensor([2.0], dtype=torch.float64)},
            {"local_leaf": torch.tensor([4.0], dtype=torch.float64)},
        ],
        val_losses=[0.1, 0.2, 0.3],
        val_loss_agg=0.2,
        val_loss_agg_mode="mean",
    )
    stageb_ast = AddNode(
        MulNode(
            AtomNode(kind="free_const", var_idxs=(), kwargs={"name": "shared_leaf"}, tag="shared_leaf"),
            Var(0),
        ),
        AtomNode(kind="free_const", var_idxs=(), kwargs={"name": "local_leaf"}, tag="local_leaf"),
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["d0.csv", "d1.csv", "d2.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data=None,
        stageB_data={
            "ast": stageb_ast,
            "val_loss": 0.1,
            "params": 2,
            "dataset_ids": ["d0", "d1", "d2"],
            "dataset_metadata": [
                {"temperature": 1.0},
                {"temperature": 2.0},
                {"temperature": 3.0},
            ],
            "simplification_path": [],
        },
        final_model=None,
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=None, x_transform_map=None, dataset_ids=["d0", "d1", "d2"]),
        class_sr_result=class_sr_result,
        committee_topk=2,
        nvars=1,
        dtype=torch.float64,
        discovery_constant_lift_enable=True,
        discovery_constant_lift_min_regimes=3,
        discovery_constant_lift_trigger_mean_cv=0.2,
        discovery_constant_lift_apply_enable=True,
        discovery_constant_lift_apply_topk=1,
        discovery_constant_lift_min_rel_gain=2.0,
    )

    lifted_row = next(
        row
        for row in payload["committee_members"]
        if row["metadata"].get("constant_lift_applied", False)
    )
    assert lifted_row["metadata"]["constant_lift_symbolic_structure_mode"] == "ast"
    assert isinstance(lifted_row["symbolic_structure"], dict)
    assert lifted_row["symbolic_structure"]["__bridge_node__"] == "AddNode"

    restored = deserialize_committee_members([lifted_row])[0]
    assert not isinstance(restored.symbolic_structure, str)
    assert "shared_leaf" in restored.display_expr
    assert "temperature(x0)" in restored.display_expr


def test_run_sr_discovery_integration_populates_witness_predictions(tmp_path):
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "witness_box",
                        "type": "points",
                        "points": [[1.0, 2.0], [3.0, 4.0]],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": "x0",
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data=None,
        final_model=_WitnessModel(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=None, x_transform_map=None),
        committee_topk=2,
        experiment_manifest_path=str(manifest_path),
        witness_capture_enable=True,
        witness_hessian_diag_enable=True,
        diagnostic_set="extended",
        nvars=2,
        dtype=torch.float64,
    )

    candidate = payload["experiment_candidates_full"][0]
    assert candidate["derivative_predictions"]["stageA_final"] == [[2.0, 2.0], [6.0, 2.0]]
    diag = candidate["diagnostic_predictions"]["stageA_final"]
    assert diag["hdiag_abs_mean"] == 1.0
    assert "value_q50" in diag
    assert payload["config"]["witness_capture_enable"] is True
    assert payload["config"]["witness_hessian_diag_enable"] is True


def test_run_sr_discovery_integration_applies_research_profile_overrides(tmp_path):
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "profile_box",
                        "type": "points",
                        "points": [[0.8, 0.0], [1.2, 0.0]],
                        "bounds": {"0": [0.0, 2.0], "1": [0.0, 0.0]},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": ("var", 0),
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": ("sub", ("const", 2.0), ("var", 0)),
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [],
        },
        final_model=_LinearXModel(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=_ReverseXModel(), x_transform_map=None),
        committee_topk=4,
        experiment_manifest_path=str(manifest_path),
        research_profile="teacher_witness_full",
        nvars=2,
        dtype=torch.float64,
    )

    config = payload["config"]
    assert config["research_profile"] == "teacher_witness_full"
    assert config["witness_capture_enable"] is True
    assert config["witness_hessian_diag_enable"] is True
    assert config["diagnostic_set"] == "physics"
    assert config["beta"] == 1.0
    assert config["gamma"] == 0.5
    assert config["disagreement_mode"] == "witness"
    assert config["experiment_optimize_enable"] is True
    assert config["theory_benchmark_enable"] is True
    assert config["discovery_constant_lift_enable"] is True
    assert config["discovery_constant_lift_apply_enable"] is True
    assert config["discovery_constant_lift_apply_topk"] == 2
    activation = payload["research_activation"]
    assert activation["research_profile"] == "teacher_witness_full"
    assert activation["witness_mode_selected"] is True
    assert activation["witness_capture_active"] is True
    assert activation["experiment_optimization_used"] is True
    summary = discovery_summary_from_payload(
        payload,
        results_path=tmp_path / "toy.discovery.json",
    )
    assert summary["research_profile"] == "teacher_witness_full"
    assert summary["research_activation"]["witness_mode_selected"] is True
    assert summary["research_activation"]["experiment_optimization_used"] is True


def test_run_sr_discovery_integration_can_optimize_experiment_candidates(tmp_path):
    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "opt_box",
                        "type": "points",
                        "points": [[0.8, 0.0], [1.2, 0.0]],
                        "bounds": {"0": [0.0, 2.0], "1": [0.0, 0.0]},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": ("var", 0),
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": ("sub", ("const", 2.0), ("var", 0)),
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [],
        },
        final_model=_LinearXModel(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=_ReverseXModel(), x_transform_map=None),
        committee_topk=4,
        experiment_manifest_path=str(manifest_path),
        disagreement_mode="witness",
        nvars=2,
        dtype=torch.float64,
        experiment_optimize_enable=True,
        experiment_opt_steps=24,
        experiment_opt_lr=0.1,
    )

    assert payload["config"]["experiment_optimize_enable"] is True
    assert payload["experiment_selection"]["optimization"]["optimized_candidate_count"] == 1
    candidate = payload["experiment_candidates_full"][0]
    assert candidate["conditions"]["optimized"] is True
    assert candidate["metadata"]["continuous_optimization"]["score_after"] >= candidate["metadata"]["continuous_optimization"]["score_before"]


def test_run_closed_loop_from_discovery_payload_replays_saved_candidates(monkeypatch, tmp_path):
    def _fake_predict(candidate, x):
        if candidate.member.member_id == "stageA_final":
            return x[:, 0]
        if candidate.member.member_id == "stageB_final":
            return -x[:, 0]
        return None

    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        _fake_predict,
    )

    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "flat", "type": "points", "points": [[0.0, 0.0]]},
                    {"experiment_id": "spread", "type": "points", "points": [[1.0, 0.0], [2.0, 0.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": ("var", 0),
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": ("neg", ("var", 0)),
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [],
        },
        final_model=object(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=object(), x_transform_map=None),
        committee_topk=4,
        experiment_manifest_path=str(manifest_path),
        beta=0.0,
        gamma=0.0,
        nvars=2,
        dtype=torch.float64,
    )

    replay = run_closed_loop_from_discovery_payload(payload, beta=0.0, gamma=0.0)
    assert replay["selected_experiment"]["experiment_id"] == "spread"
    assert replay["committee_summary"]["member_count"] >= 2


def test_run_closed_loop_from_discovery_payload_replays_witness_mode(monkeypatch, tmp_path):
    def _fake_predict(candidate, x):
        if candidate.member.member_id == "stageA_final":
            return x[:, 0]
        if candidate.member.member_id == "stageB_final":
            return 2.0 - x[:, 0]
        return None

    monkeypatch.setattr(
        "nestynet_sr.discovery.integration._predict_runtime_candidate",
        _fake_predict,
    )

    manifest_path = tmp_path / "experiments.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "shape_agree", "type": "points", "points": [[1.0, 0.0], [1.0, 0.0]]},
                    {"experiment_id": "shape_split", "type": "points", "points": [[0.0, 0.0], [2.0, 0.0]]},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    payload = run_sr_discovery_integration(
        filepath="toy.csv",
        filepaths=["toy.csv"],
        report_path=str(tmp_path / "toy.report.json"),
        stageA_data={
            "ast": ("var", 0),
            "val_loss": 0.2,
            "nn_n_params": 8,
            "y_op_name": "identity",
        },
        stageB_data={
            "ast": ("sub", ("const", 2.0), ("var", 0)),
            "val_loss": 0.1,
            "params": 5,
            "simplification_path": [],
        },
        final_model=object(),
        final_y_op_name="identity",
        stageB_state=SimpleNamespace(model=object(), x_transform_map=None),
        committee_topk=4,
        experiment_manifest_path=str(manifest_path),
        beta=0.0,
        gamma=0.0,
        disagreement_mode="witness",
        nvars=2,
        dtype=torch.float64,
    )

    replay = run_closed_loop_from_discovery_payload(payload, beta=0.0, gamma=0.0)
    assert replay["selected_experiment"]["experiment_id"] == "shape_split"
    assert replay["config"]["disagreement_mode"] == "witness"
