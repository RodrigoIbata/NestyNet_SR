# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import json
import os
import subprocess
import sys
from types import SimpleNamespace

from nestynet_sr.sr_search.coe_witness import (
    CoEWitnessExecutor,
    CoEWitnessJob,
    coe_pair_vote,
    coe_stageB_refit_ast_from_payload,
    coe_stageB_refit_ast_to_payload,
    coe_witness_execution_metadata,
    coe_witness_jobs_from_specs,
    run_fixed_expression_candidate_witnesses,
    run_fixed_expression_pair_witnesses,
    run_stageB_refit_pair_witnesses,
    run_threaded_witnesses,
)
from nestynet_sr.sr_search.coe_committee import CandidateArtifact, SliceSpec
from nestynet_sr.sr_core.bridges import AtomNode, ConstNode, MulNode, Var, ast_to_human_readable
from nestynet_sr.sr_search.config import LMHyperparams


def test_worker_thread_cap_overrides_inherited_master_budget():
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "12"
    env["COE_WORKER_THREADS"] = "1"
    script = """
import json
import os
import torch
from threadpoolctl import threadpool_info
from nestynet_sr.sr_search.coe_witness import _thread_cap_initializer
before = torch.get_num_threads()
_thread_cap_initializer()
print(json.dumps({
    'before': before,
    'torch': torch.get_num_threads(),
    'env': {key: os.environ[key] for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
    'native': [row['num_threads'] for row in threadpool_info()],
}))
"""
    payload = json.loads(
        subprocess.check_output([sys.executable, "-c", script], env=env, text=True)
    )

    assert payload["before"] > 1
    assert payload["torch"] == 1
    assert payload["native"] and all(value == 1 for value in payload["native"])
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert payload["env"][key] == "1"


def test_in_process_witness_worker_preserves_master_thread_budget(tmp_path):
    data_path = tmp_path / "serial-thread-budget.csv"
    data_path.write_text("x0,y\n0,0\n1,1\n", encoding="utf-8")
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "6"
    env["COE_WORKER_THREADS"] = "1"
    env["WITNESS_TEST_DATA"] = str(data_path)
    script = """
import json
import os
import torch
from nestynet_sr.sr_search.coe_witness import _fixed_expression_candidate_worker
payload = {
    'spec': {'slice_id': 0, 'train_start': 0, 'train_stop': 1,
             'val_start': 0, 'val_stop': 2},
    'candidate': {'candidate_id': 'x', 'expr': 'x0', 'source': 'test',
                  'label': 'x'},
    'filepath': os.environ['WITNESS_TEST_DATA'],
    'min_valid_fraction': 0.8,
}
row = _fixed_expression_candidate_worker(payload)
print(json.dumps({
    'status': row['status'],
    'torch': torch.get_num_threads(),
    'env': {key: os.environ[key] for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
}))
"""
    payload = json.loads(
        subprocess.check_output([sys.executable, "-c", script], env=env, text=True)
    )

    assert payload["status"] == "success"
    assert payload["torch"] == 6
    assert all(value == "6" for value in payload["env"].values())


def test_serial_witness_executor_preserves_order_and_stop():
    specs = [
        SimpleNamespace(slice_id=7, train_start=0, train_stop=10, val_start=10, val_stop=20),
        SimpleNamespace(slice_id=8, train_start=20, train_stop=30, val_start=30, val_stop=40),
        SimpleNamespace(slice_id=9, train_start=40, train_stop=50, val_start=50, val_stop=60),
    ]
    jobs = coe_witness_jobs_from_specs(specs, prefix="unit")
    seen: list[int] = []

    def worker(job):
        seen.append(job.slice_id)
        return {"status": "success", "value": job.slice_id * 2}

    executor = CoEWitnessExecutor(parallelism=4)
    rows = executor.run(jobs, worker, stop_after=lambda rows_i: len(rows_i) >= 2)

    assert seen == [7, 8]
    assert [row["slice_id"] for row in rows] == [7, 8]
    assert [row["job_id"] for row in rows] == ["unit:7", "unit:8"]
    assert all(row["executor_backend"] == "serial" for row in rows)
    assert executor.metadata() == {"backend": "serial", "parallelism": 4}


def test_witness_helpers_build_metadata_and_votes():
    spec = SimpleNamespace(slice_id=2, train_start=3, train_stop=5, val_start=8, val_stop=13)
    (job,) = coe_witness_jobs_from_specs([spec], prefix="meta")

    assert job.payload is spec
    assert job.metadata["train_rows"] == [3, 5]
    assert job.metadata["val_rows"] == [8, 13]
    assert coe_pair_vote(-0.2, 0.1) == "win"
    assert coe_pair_vote(0.2, 0.1) == "loss"
    assert coe_pair_vote(0.05, 0.1) == "tie"


def test_witness_metadata_reports_serial_parallel_disabled_reason():
    executor = CoEWitnessExecutor(parallelism=3)
    rows = [{"status": "success", "slice_id": 1, "executor_backend": "serial"}]

    meta = coe_witness_execution_metadata(
        executor,
        rows,
        parallel_disabled_reason="refit_gate_mutates_live_context",
    )

    assert meta == {
        "backend": "serial",
        "parallelism": 3,
        "effective_backend": "serial",
        "parallel_disabled_reason": "refit_gate_mutates_live_context",
    }


def test_stageB_refit_ast_payload_round_trips_compound_inputs():
    root = AtomNode(
        "nn",
        (0, 1),
        kwargs={"input_ast": MulNode(Var(0), Var(1)), "num_segments": 4},
        tag="leaf0",
        inputs=(MulNode(Var(0), Var(1)),),
    )

    payload = coe_stageB_refit_ast_to_payload(root)
    rebuilt = coe_stageB_refit_ast_from_payload(payload)

    assert ast_to_human_readable(rebuilt) == ast_to_human_readable(root)
    assert rebuilt.tag == "leaf0"
    assert rebuilt.kwargs["num_segments"] == 4
    assert ast_to_human_readable(rebuilt.kwargs["input_ast"]) == "(x0 * x1)"


def test_stageB_refit_pair_worker_runs_serial_and_process(tmp_path):
    data_path = tmp_path / "toy_refit.csv"
    data_path.write_text("x0,y\n0,0\n1,1\n2,2\n3,3\n", encoding="utf-8")
    spec = SliceSpec(slice_id=0, train_start=0, train_stop=2, val_start=2, val_stop=4)
    lm_hp = LMHyperparams(
        epochs=1,
        epochs_min=0,
        nval_patience=1,
        loss_target=1.0e-12,
        loss_acceptable=1.0,
        loss_in_MAD_units=False,
        log_to_console=False,
        LM_verbose=False,
    )
    payload = {
        "schema": "coe_stageB_refit_witness_v1",
        "filepath": str(data_path),
        "spec": spec.to_dict(),
        "incumbent_root": coe_stageB_refit_ast_to_payload(Var(0)),
        "candidate_root": coe_stageB_refit_ast_to_payload(MulNode(ConstNode(2.0), Var(0))),
        "incumbent_reuse": {},
        "candidate_reuse": {},
        "lm_hp": lm_hp,
        "dtype": "float64",
        "device": "cpu",
        "force_cpu": True,
        "epochs": 1,
        "loss_scale": 1.0,
        "batch_size": 2,
        "y_transform_name": "identity",
        "refit_tier": "unit",
    }

    serial_rows = run_stageB_refit_pair_witnesses(
        payloads=[payload],
        executor=CoEWitnessExecutor(parallelism=1),
        prefix="refit",
    )
    parallel_rows = run_stageB_refit_pair_witnesses(
        payloads=[payload],
        executor=CoEWitnessExecutor(parallelism=2),
        prefix="refit",
    )

    for rows in (serial_rows, parallel_rows):
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == "success", row.get("error")
        assert row["incumbent_compare_loss"] == 0.0
        assert row["candidate_compare_loss"] > 0.0
        assert row["comparison_space"] == "fit_space"
    assert serial_rows[0]["executor_backend"] == "serial"
    assert parallel_rows[0]["executor_backend"] in {"process", "serial"}


def test_threaded_witness_helper_preserves_order_and_reports_backend():
    jobs = [CoEWitnessJob(job_id=f"j{i}", slice_id=i) for i in range(4)]

    def worker(job):
        return {"status": "success", "value": job.slice_id}

    executor = CoEWitnessExecutor(parallelism=2)
    rows = run_threaded_witnesses(jobs, worker, executor=executor)

    assert [row["slice_id"] for row in rows] == [0, 1, 2, 3]
    assert [row["value"] for row in rows] == [0, 1, 2, 3]
    assert all(row["executor_backend"] == "thread" for row in rows)
    assert coe_witness_execution_metadata(executor, rows) == {
        "backend": "serial",
        "parallelism": 2,
        "effective_backend": "thread",
    }


def test_fixed_expression_pair_witnesses_can_run_in_process_pool(tmp_path):
    data_path = tmp_path / "toy.csv"
    data_path.write_text("x0,y\n0,0\n1,1\n2,2\n3,3\n", encoding="utf-8")
    specs = [
        SliceSpec(slice_id=0, train_start=0, train_stop=1, val_start=0, val_stop=2),
        SliceSpec(slice_id=1, train_start=2, train_stop=3, val_start=2, val_stop=4),
    ]
    incumbent = CandidateArtifact(
        candidate_id="incumbent",
        expr="x0",
        source="test",
        label="incumbent",
    )
    candidate = CandidateArtifact(
        candidate_id="candidate",
        expr="0",
        source="test",
        label="candidate",
    )

    rows = run_fixed_expression_pair_witnesses(
        specs=specs,
        incumbent=incumbent,
        candidate=candidate,
        filepath=str(data_path),
        min_valid_fraction=0.80,
        executor=CoEWitnessExecutor(parallelism=2),
        prefix="fixed",
    )

    assert [row["slice_id"] for row in rows] == [0, 1]
    assert all(row["status"] == "success" for row in rows)
    assert all(row["executor_backend"] in {"process", "serial"} for row in rows)
    if rows[0]["executor_backend"] == "serial":
        assert rows[0].get("executor_fallback_reason")
    assert all(row["incumbent_result"]["val_mse"] == 0.0 for row in rows)
    assert all(row["candidate_result"]["val_mse"] > 0.0 for row in rows)


def test_fixed_expression_candidate_witnesses_preserve_final_committee_order(tmp_path):
    data_path = tmp_path / "toy.csv"
    data_path.write_text("x0,y\n0,0\n1,1\n2,2\n3,3\n", encoding="utf-8")
    specs = [
        SliceSpec(slice_id=0, train_start=0, train_stop=1, val_start=0, val_stop=2),
        SliceSpec(slice_id=1, train_start=2, train_stop=3, val_start=2, val_stop=4),
    ]
    candidates = [
        CandidateArtifact(candidate_id="x", expr="x0", source="test", label="x"),
        CandidateArtifact(candidate_id="zero", expr="0", source="test", label="zero"),
    ]

    rows = run_fixed_expression_candidate_witnesses(
        specs=specs,
        candidates=candidates,
        filepath=str(data_path),
        min_valid_fraction=0.80,
        executor=CoEWitnessExecutor(parallelism=2),
        prefix="final",
    )

    assert [(row["slice_id"], row["candidate_id"]) for row in rows] == [
        (0, "x"),
        (0, "zero"),
        (1, "x"),
        (1, "zero"),
    ]
    assert all(row["executor_backend"] in {"process", "serial"} for row in rows)
    assert rows[0]["val_mse"] == 0.0
    assert rows[2]["val_mse"] == 0.0
