# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_search.stagea_fit_tournament import (
    StageAFitLane,
    _apply_lane_train_loss_cap,
    _full_parameter_snapshot,
    _load_full_parameter_snapshot,
    _loader_tensor_input_dim,
    _make_model_recipe,
    _quiet_fit_worker,
    _rebuild_model,
    _row_losses,
    _tensor_loader,
    choose_stageA_fit_lane,
    fit_initial_model_with_tournament,
    fit_stageA_candidate_with_tournament,
    validate_stageA_fit_slice_firewall,
)


def _lane(slice_id, rows, *, accepted=True, status="success"):
    rows = np.asarray(rows, dtype=float)
    return StageAFitLane(
        slice_id,
        status,
        accepted=accepted,
        canonical_init_applied=True,
        canonical_fingerprint=f"canonical-{slice_id}",
        comparison_mse=float(np.mean(rows)),
        row_losses=rows.tolist(),
    )


def _choose(lanes, **overrides):
    kwargs = dict(
        master_slice=0,
        alpha=0.05,
        min_rel_improvement=0.01,
        noise_floor=0.0,
        target_scale=1.0,
        unit_keys=np.arange(len(lanes[0].row_losses)),
        seed=17,
    )
    kwargs.update(overrides)
    return choose_stageA_fit_lane(lanes, **kwargs)


def test_equal_lanes_keep_master_and_invalid_workers_fail_soft():
    rows = np.linspace(0.8, 1.2, 100) ** 2
    selected, summary = _choose([
        _lane(0, rows),
        _lane(1, rows),
        _lane(2, rows * 0.1, accepted=False),
        StageAFitLane(3, "error", error="boom"),
    ])
    assert selected.slice_id == 0
    assert summary["decision"] == "keep_master"


def test_lowest_valid_common_comparison_mse_wins():
    rows = np.linspace(0.8, 1.2, 100) ** 2
    selected, _ = _choose([_lane(0, rows), _lane(1, 0.5 * rows)])
    assert selected.slice_id == 1

    selected, summary = _choose([_lane(0, rows), _lane(1, 0.99 * rows)])
    assert selected.slice_id == 1
    assert summary["selection_rule"] == "best_valid_common_comparison_mse"


def test_zero_standard_error_does_not_block_a_better_fit():
    selected, _ = _choose([_lane(0, np.ones(100)), _lane(1, np.full(100, 0.5))])
    assert selected.slice_id == 1


def test_lane_without_applied_canonical_initialization_cannot_win():
    master = _lane(0, np.linspace(0.8, 1.2, 100) ** 2)
    challenger = _lane(1, np.linspace(0.1, 0.2, 100) ** 2)
    challenger.canonical_init_applied = False
    challenger.canonical_fingerprint = None
    selected, summary = _choose([master, challenger])
    assert selected.slice_id == 0
    assert summary["reason"] == "no valid challengers"


def test_training_sanity_cap_disqualifies_only_bad_lane_before_selection():
    rows = np.linspace(0.8, 1.2, 100) ** 2
    master = _lane(0, rows)
    master.train_loss = 0.2
    challenger = _lane(1, 0.5 * rows)
    challenger.train_loss = 20.0
    _apply_lane_train_loss_cap([master, challenger], 1.0)
    assert master.accepted
    assert not challenger.accepted
    selected, summary = _choose([master, challenger])
    assert selected.slice_id == 0
    assert summary["reason"] == "no valid challengers"


def test_fit_slice_firewall_rejects_witness_overlap_including_start_zero():
    assert validate_stageA_fit_slice_firewall(
        "1 8", witness_start=9, witness_count=11
    ) == [1, 8]
    with pytest.raises(ValueError, match="cannot overlap"):
        validate_stageA_fit_slice_firewall("1 9", witness_start=9, witness_count=11)
    with pytest.raises(ValueError, match="cannot overlap"):
        validate_stageA_fit_slice_firewall("1", witness_start=0, witness_count=11)
    assert validate_stageA_fit_slice_firewall(
        "1 9", witness_start=0, witness_count=0
    ) == [1, 9]


def test_extra_lanes_can_supply_an_even_better_valid_fit():
    rng = np.random.default_rng(0)
    rows = (1.0 + rng.normal(0.0, 0.2, 80)) ** 2
    borderline = rows - 0.014 + rng.normal(0.0, 0.1, 80)
    distractors = [rows + rng.normal(0.0, 0.25, 80) for _ in range(7)]
    selected_one, _ = _choose(
        [_lane(0, rows), _lane(1, borderline)], min_rel_improvement=0.0
    )
    lanes = [_lane(0, rows), _lane(1, borderline)] + [
        _lane(i + 2, losses) for i, losses in enumerate(distractors)
    ]
    selected_many, summary = _choose(
        lanes,
        min_rel_improvement=0.0,
    )
    assert selected_one.slice_id == 1
    assert selected_many.slice_id == min(
        lanes, key=lambda lane: (lane.comparison_mse, lane.slice_id)
    ).slice_id
    assert summary["selection_rule"] == "best_valid_common_comparison_mse"


def test_common_row_losses_are_computed_in_original_y_space():
    model = torch.nn.Identity()
    x = torch.tensor([[np.log(2.0)], [np.log(5.0)]], dtype=torch.float64)
    target = torch.tensor([[np.log(3.0)], [np.log(7.0)]], dtype=torch.float64)
    loader = DataLoader(TensorDataset(x, target), batch_size=2)
    losses, finite = _row_losses(model, loader, torch.device("cpu"), torch.exp)
    np.testing.assert_allclose(losses, [1.0, 4.0])
    assert finite == 1.0


def test_portable_loader_can_preserve_source_input_declaration_contract():
    x = torch.zeros(6, 2, dtype=torch.float64)
    y = torch.zeros(6, 1, dtype=torch.float64)
    tensor_loader = _tensor_loader(x, y, 6)
    opaque_loader = _tensor_loader(x, y, 6, expose_tensors=False)
    assert _loader_tensor_input_dim(tensor_loader) == 2
    assert _loader_tensor_input_dim(opaque_loader) is None


@pytest.mark.parametrize(
    ("source_input_dim", "initial_fit", "expected_model_input_dim"),
    [(None, True, None), (1, True, 1), (1, False, None)],
)
def test_fit_worker_does_not_invent_tensor_loader_input_declaration(
    monkeypatch, source_input_dim, initial_fit, expected_model_input_dim
):
    import nestynet_sr.sr_search.stagea_fit_tournament as tournament_module

    source = _real_ast_model()
    source._global_input_dim = None
    recipe = _make_model_recipe(source)
    seen = {}

    class FakeOptimizer:
        _sr_canonical_init_applied = True
        _sr_canonical_state_fingerprint = "canonical-worker"

    def record_fit(model, train_dl):
        seen["model_input_dim"] = model._global_input_dim
        seen["loader_exposes_tensors"] = hasattr(train_dl.dataset, "tensors")

    def fake_initial_train(model, train_dl, _val_dl, **_kwargs):
        record_fit(model, train_dl)
        return 1.0, 1.0, None, FakeOptimizer()

    def fake_candidate_train(model, train_dl, _val_dl, **_kwargs):
        record_fit(model, train_dl)
        return True, 1.0, 1.0, None, FakeOptimizer()

    monkeypatch.setattr(
        tournament_module, "train_initial_model", fake_initial_train
    )
    monkeypatch.setattr(
        tournament_module, "train_candidate_model", fake_candidate_train
    )
    x = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64).reshape(-1, 1)
    y = torch.exp(-0.5 * x.square())
    lane = _quiet_fit_worker(
        {
            "slice_id": 1,
            "train_start": 8,
            "train_stop": 12,
            "model_recipe": recipe,
            "train_x": x[:4],
            "train_y": y[:4],
            "fit_val_x": x[4:6],
            "fit_val_y": y[4:6],
            "comparison_x": x[6:],
            "train_tensor_input_dim": source_input_dim,
            "batch_size": 4,
            "lm_hp": SimpleNamespace(log_file=None, log_to_console=False),
            "epochs": 1,
            "LM_strategy": "direct_solve",
            "nval_patience": 1,
            "loss_target": None,
            "epochs_min": 0,
            "chisq_tol": 1.0e-12,
            "epochs_awful_check": None,
            "awful_threshold": None,
            "log_level": "WARNING",
            "initial_fit": initial_fit,
            "accept_threshold": float("inf"),
        }
    )
    assert lane.status == "success", lane.error
    assert seen == {
        "model_input_dim": expected_model_input_dim,
        "loader_exposes_tensors": False,
    }


def test_feature_off_delegates_directly_without_splitting(monkeypatch):
    import nestynet_sr.sr_search.stagea_fit_tournament as tournament_module

    sentinel = (True, 1.0, 2.0, object(), object())
    seen = {}

    def fake_train(model, train_dl, val_dl, **kwargs):
        seen.update(model=model, train_dl=train_dl, val_dl=val_dl, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(tournament_module, "train_candidate_model", fake_train)
    model, train_dl, val_dl = object(), object(), object()
    hp = SimpleNamespace(coe_stageA_fit_tournament=False, coe_mode="reservoir_discovery")
    result = fit_stageA_candidate_with_tournament(
        model,
        train_dl,
        val_dl,
        lm_hp=hp,
        device=torch.device("cpu"),
        accept_threshold=3.0,
        epochs=4,
    )
    assert result is sentinel
    assert seen["model"] is model
    assert seen["train_dl"] is train_dl and seen["val_dl"] is val_dl
    assert seen["kwargs"]["accept_threshold"] == 3.0
    assert seen["kwargs"]["epochs"] == 4


def test_initial_feature_off_delegates_to_original_initial_fit(monkeypatch):
    import nestynet_sr.sr_search.stagea_fit_tournament as tournament_module

    sentinel = (1.0, 2.0, object(), object())
    seen = {}

    def fake_train(model, train_dl, val_dl, **kwargs):
        seen.update(model=model, train_dl=train_dl, val_dl=val_dl, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(tournament_module, "train_initial_model", fake_train)
    model, train_dl, val_dl = object(), object(), object()
    hp = SimpleNamespace(
        coe_stageA_fit_tournament=False, coe_mode="reservoir_discovery"
    )
    result = fit_initial_model_with_tournament(
        model,
        train_dl,
        val_dl,
        lm_hp=hp,
        device=torch.device("cpu"),
        epochs=4,
    )
    assert result is not sentinel
    assert result == sentinel
    assert seen["model"] is model
    assert seen["train_dl"] is train_dl and seen["val_dl"] is val_dl
    assert seen["kwargs"]["epochs"] == 4


def test_internal_noncanonical_initial_restart_bypasses_tournament(monkeypatch):
    import nestynet_sr.sr_search.stagea_fit_tournament as tournament_module

    calls = []
    sentinel = (1.0, 2.0, object(), object())

    def fake_train(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(tournament_module, "train_initial_model", fake_train)
    hp = SimpleNamespace(
        coe_stageA_fit_tournament=True,
        coe_mode="reservoir_discovery",
        canonical_init=False,
        coe_filepath="unused.csv",
    )
    result = fit_initial_model_with_tournament(
        object(), object(), object(), lm_hp=hp, device=torch.device("cpu"), epochs=4
    )
    assert result == sentinel
    assert len(calls) == 1
    assert not hasattr(hp, "coe_stageA_fit_tournament_records")


def test_unsupported_canonical_provider_keeps_single_master_fit(monkeypatch):
    import nestynet_sr.sr_search.stagea_fit_tournament as tournament_module

    sentinel = (True, 1.0, 2.0, object(), object())
    calls = []

    def fake_train(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(tournament_module, "train_candidate_model", fake_train)
    hp = SimpleNamespace(
        coe_stageA_fit_tournament=True,
        coe_mode="reservoir_discovery",
        canonical_init=True,
        coe_filepath="unused.csv",
    )
    result = fit_stageA_candidate_with_tournament(
        torch.nn.Identity(),
        object(),
        object(),
        lm_hp=hp,
        device=torch.device("cpu"),
        accept_threshold=3.0,
        epochs=4,
    )
    assert result is sentinel
    assert len(calls) == 1
    assert not hasattr(hp, "coe_stageA_fit_tournament_records")


def test_allstages_suite_forwards_fit_tournament_controls(tmp_path, monkeypatch):
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
        canonical_init=True,
        coe_mode="reservoir_discovery",
        coe_stageA_fit_tournament=True,
        coe_stageA_fit_slices="1 3 5",
        coe_stageA_fit_alpha=0.04,
        coe_stageA_fit_comparison_fraction=0.4,
        coe_stageA_fit_min_rel_improvement=0.02,
        coe_scout_parallelism=8,
    )
    assert result["success"]
    cmd = captured["cmd"]
    assert "--coe_stageA_fit_tournament" in cmd
    assert cmd[cmd.index("--coe_stageA_fit_slices") + 1] == "1 3 5"
    assert cmd[cmd.index("--coe_stageA_fit_alpha") + 1] == "0.04"
    assert cmd[cmd.index("--coe_stageA_fit_comparison_fraction") + 1] == "0.4"
    assert cmd[cmd.index("--coe_stageA_fit_min_rel_improvement") + 1] == "0.02"
    assert cmd[cmd.index("--coe_scout_parallelism") + 1] == "8"


def test_json_report_persists_fit_tournament_records(tmp_path):
    import json

    from nestynet_sr.run_sr_reports import write_json_report

    record = {
        "fit_kind": "initial_teacher",
        "decision": "replace_master",
        "selected_slice": 3,
    }
    report_path = tmp_path / "run.report.json"
    write_json_report(
        filepath="pb000.csv",
        report_path=str(report_path),
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        walltime=0.0,
        stageA_data={
            "stageA_status": "resolved",
            "coe_stageA_fit_tournament_records": [record],
        },
        enable_truth_eval=False,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["stageA"]["coe_stageA_fit_tournament_records"] == [record]


def _real_ast_model():
    from nestynet.default_setup import dtype

    from nestynet_sr.sr_core.bridges import build_initial_ast
    from nestynet_sr.sr_search.model_builders import LeafBuilder, build_composite_ast

    hp = SimpleNamespace(
        model_base_name="G_Model",
        Gmodel_scale=0.1,
        Nout_size=1,
        block_size_target=None,
    )
    ast = build_initial_ast(1, num_segments=2, dual_layer=True)
    model, _nparam, _ast = build_composite_ast(
        ast,
        None,
        None,
        LeafBuilder(hp, torch.device("cpu"), dtype),
        torch.device("cpu"),
        dtype,
    )
    model.declare_global_input_dim(1)
    return model


def test_production_dual_ast_uses_a_pickle_safe_reconstruction_recipe():
    model = _real_ast_model()
    with pytest.raises(AttributeError, match="Can't pickle local object"):
        pickle.dumps(model)
    recipe = _make_model_recipe(model)
    pickle.dumps(recipe)
    rebuilt = _rebuild_model(recipe)
    x = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(x), model(x))
    assert set(rebuilt.state_dict()) == set(model.state_dict())
    source_parameters, source_layout = _full_parameter_snapshot(model)
    rebuilt_parameters, rebuilt_layout = _full_parameter_snapshot(rebuilt)
    assert recipe.parameter_layout_fingerprint == source_layout == rebuilt_layout
    for expected, actual in zip(source_parameters, rebuilt_parameters):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_recipe_full_tsop_state_is_authoritative_for_canonical_initialization():
    from nestynet_sr.sr_search.config import LMHyperparams
    from nestynet_sr.sr_search.training import train_initial_model

    model = _real_ast_model()
    stale_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    original_vectors, _layout = _full_parameter_snapshot(model)
    altered_vectors = [vector.clone() for vector in original_vectors]
    for stage_index, vector in enumerate(altered_vectors):
        vector.add_(0.01 * float(stage_index + 1))
    _load_full_parameter_snapshot(model, altered_vectors)

    # Emulate a segmented model whose registered state_dict is incomplete or
    # stale relative to its authoritative full TSOP storage. The worker recipe
    # must restore the latter before running data-dependent canonical init.
    recipe = replace(_make_model_recipe(model), state_dict=stale_state)
    rebuilt = _rebuild_model(recipe)
    rebuilt_vectors, rebuilt_layout = _full_parameter_snapshot(rebuilt)
    assert rebuilt_layout == recipe.parameter_layout_fingerprint
    for expected, actual in zip(altered_vectors, rebuilt_vectors):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    x = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64).reshape(-1, 1)
    y = torch.exp(-0.5 * x.square())
    train_dl = DataLoader(TensorDataset(x[:12], y[:12]), batch_size=12)
    val_dl = DataLoader(TensorDataset(x[12:], y[12:]), batch_size=4)
    hp = LMHyperparams(canonical_init=True, evidence_enable=False)

    outcomes = []
    for fitted in (model, rebuilt):
        torch.manual_seed(71)
        val_loss, train_loss, _params, opt = train_initial_model(
            fitted,
            train_dl,
            val_dl,
            epochs=0,
            LM_strategy=hp.strategy,
            nval_patience=hp.nval_patience,
            loss_target=None,
            epochs_min=0,
            chisq_tol=hp.chisq_tol,
            device=torch.device("cpu"),
            log_to_console=False,
            lm_hp=hp,
        )
        outcomes.append(
            (
                opt._sr_canonical_state_fingerprint,
                float(val_loss),
                float(train_loss),
            )
        )
    assert outcomes[0][0] == outcomes[1][0]
    np.testing.assert_allclose(outcomes[0][1:], outcomes[1][1:], rtol=0.0, atol=0.0)


def test_full_tsop_layout_ignores_transient_fit_registration_differences():
    target = _real_ast_model()
    source = _real_ast_model()
    source.load_state_dict(target.state_dict())
    source_base = source.leaf[0].stage0.base_model
    source_base.set_fitting_parameters(
        source_base.a_fit_indices,
        source_base.b_fit_indices,
        None,
        source_base.K_fit_indices,
        preserve_current=True,
    )
    assert set(source.state_dict()) != set(target.state_dict())
    source_vectors, source_layout = _full_parameter_snapshot(source)
    _target_vectors, target_layout = _full_parameter_snapshot(target)
    assert source_layout == target_layout
    _load_full_parameter_snapshot(target, source_vectors)
    x = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        torch.testing.assert_close(target(x), source(x))


def test_recipe_rebuild_ignores_transient_fit_registration_shapes():
    source = _real_ast_model()
    stage0 = source.leaf[0].stage0.base_model
    stage0.set_fitting_parameters(
        stage0.a_fit_indices[::2],
        stage0.b_fit_indices,
        None,
        stage0.K_fit_indices,
        preserve_current=True,
    )
    recipe = _make_model_recipe(source)
    rebuilt = _rebuild_model(recipe)
    source_vectors, source_layout = _full_parameter_snapshot(source)
    rebuilt_vectors, rebuilt_layout = _full_parameter_snapshot(rebuilt)
    assert source_layout == rebuilt_layout == recipe.parameter_layout_fingerprint
    for expected, actual in zip(source_vectors, rebuilt_vectors):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    x = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(x), source(x), rtol=0.0, atol=0.0)


def test_recipe_rebuild_rejects_persistent_state_mismatch():
    model = _real_ast_model()
    recipe = _make_model_recipe(model)
    persistent = next(
        name for name in recipe.state_dict if name.endswith("._ever_active")
    )
    broken_state = dict(recipe.state_dict)
    broken_state.pop(persistent)
    with pytest.raises(ValueError, match="incompatible persistent state"):
        _rebuild_model(replace(recipe, state_dict=broken_state))


@pytest.mark.filterwarnings("ignore:.*overflow.*")
@pytest.mark.parametrize("fit_mode", ["candidate", "initial", "initial_master_error"])
def test_spawned_real_ast_lanes_use_slice_data_and_preserve_master_threads(
    tmp_path, monkeypatch, fit_mode
):
    import nestynet_sr.sr_search.stagea_fit_tournament as tournament_module

    initial_fit = fit_mode != "candidate"
    x = np.linspace(-1.0, 1.0, 24)
    # slice 0, then slice 1: four train + four val rows per slice.
    data = np.column_stack([x[:16], 2.0 + 0.25 * x[:16]])
    path = tmp_path / "slices.csv"
    np.savetxt(path, data, delimiter=",", header="x0,y", comments="")

    train_x = torch.as_tensor(data[:4, :1], dtype=torch.float64)
    target_values = np.sqrt(data[:, 1:]) if initial_fit else data[:, 1:]
    train_y = torch.as_tensor(target_values[:4], dtype=torch.float64)
    val_x = torch.as_tensor(data[4:8, :1], dtype=torch.float64)
    val_y = torch.as_tensor(target_values[4:8], dtype=torch.float64)
    train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=4)
    val_dl = DataLoader(TensorDataset(val_x, val_y), batch_size=4)
    hp = SimpleNamespace(
        coe_stageA_fit_tournament=True,
        coe_mode="reservoir_discovery",
        canonical_init=True,
        coe_filepath=str(path),
        coe_reference_slice=0,
        coe_scout_parallelism=2,
        coe_stageA_fit_slices=[1],
        coe_stageA_fit_comparison_fraction=0.5,
        coe_ndata_train=4,
        coe_ndata_val=4,
        coe_min_valid_fraction=1.0,
        coe_stageA_fit_alpha=0.05,
        coe_stageA_fit_min_rel_improvement=0.01,
        coe_noise_floor_raw=0.0,
        coe_maxt_seed=3,
        coe_scout_timeout_seconds=60.0,
        evidence_enable=False,
        fit_y_link=None,
        log_file=None,
        log_to_console=False,
    )
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "3")
    monkeypatch.setenv("COE_WORKER_THREADS", "1")
    model = _real_ast_model()
    selected_predictions = []

    def select_worker_if_available(lanes, **_kwargs):
        workers = [lane for lane in lanes if lane.slice_id == 1 and lane.status == "success"]
        if not workers:
            return lanes[0], {"decision": "keep_master", "selected_slice": 0}
        worker = workers[0]
        selected_predictions.append(torch.as_tensor(worker.comparison_predictions))
        return worker, {"decision": "replace_master", "selected_slice": 1}

    monkeypatch.setattr(tournament_module, "choose_stageA_fit_lane", select_worker_if_available)
    if fit_mode == "initial_master_error":
        def fail_local_master(*_args, **_kwargs):
            raise RuntimeError("synthetic local master failure")

        monkeypatch.setattr(
            tournament_module, "train_initial_model", fail_local_master
        )
    common = dict(
        lm_hp=hp,
        device=torch.device("cpu"),
        epochs=2,
        LM_strategy="direct_solve",
        nval_patience=2,
        loss_target=None,
        epochs_min=1,
        chisq_tol=1.0e-12,
        epochs_awful_check=None,
        awful_threshold=None,
        log_file=None,
        log_to_console=False,
        log_level="WARNING",
        lm_verbose=False,
        y_op=np.sqrt if initial_fit else None,
        y_op_inv=torch.square if initial_fit else None,
    )
    if initial_fit:
        try:
            result = fit_initial_model_with_tournament(
                model, train_dl, val_dl, **common
            )
        except RuntimeError:
            records = getattr(hp, "coe_stageA_fit_tournament_records", [])
            if records and any(
                "Operation not permitted" in str(row.get("error"))
                for row in records[-1].get("lanes", [])
            ):
                pytest.skip("sandbox forbids multiprocessing semaphores")
            raise
        params, optimizer = result[2], result[3]
        assert len(result) == 4
    else:
        result = fit_stageA_candidate_with_tournament(
            model, train_dl, val_dl, accept_threshold=float("inf"), **common
        )
        params, optimizer = result[3], result[4]
        assert len(result) == 5
    record = hp.coe_stageA_fit_tournament_records[-1]
    assert record["fit_kind"] == (
        "initial_teacher" if initial_fit else "stageA_candidate"
    )
    assert record["total_fit_parallelism"] == 2
    assert {row["slice_id"] for row in record["lanes"]} == {0, 1}
    if any("Operation not permitted" in str(row.get("error")) for row in record["lanes"]):
        pytest.skip("sandbox forbids multiprocessing semaphores")
    lane_by_slice = {row["slice_id"]: row for row in record["lanes"]}
    assert lane_by_slice[1]["status"] == "success", record
    if fit_mode == "initial_master_error":
        assert lane_by_slice[0]["status"] == "error"
        assert "synthetic local master failure" in lane_by_slice[0]["error"]
    else:
        assert lane_by_slice[0]["status"] == "success", record
    assert record["fit_validation_rows"] == [4, 6]
    assert record["comparison_rows"] == [6, 8]
    assert {
        (row["train_start"], row["train_stop"]) for row in record["lanes"]
    } == {(0, 4), (8, 12)}
    successful = [row for row in record["lanes"] if row["status"] == "success"]
    fingerprints = {row["canonical_fingerprint"] for row in successful}
    assert all(row["canonical_init_applied"] for row in successful)
    assert None not in fingerprints
    assert len(fingerprints) == (1 if fit_mode == "initial_master_error" else 2)
    assert record["selected_slice"] == 1
    with torch.no_grad():
        torch.testing.assert_close(
            model(val_x[2:]).reshape(-1),
            selected_predictions[0].to(dtype=torch.float64),
        )
    assert lane_by_slice[1]["parameter_count"] == model.num_parameters()
    # Caller compatibility: applying the returned best vector cannot undo the
    # fully adopted worker model.
    before = model(val_x).detach().clone()
    optimizer._update_param_groups(params)
    torch.testing.assert_close(model(val_x), before)
    assert os.environ["OMP_NUM_THREADS"] == "3"
