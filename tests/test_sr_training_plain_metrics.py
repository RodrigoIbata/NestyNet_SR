# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import pytest
import torch

from nestynet.optimizer.evidence import EvidenceController
from nestynet_sr.sr_search.callbacks import SRCallback, SRState
from nestynet_sr.sr_search.config import LMHyperparams
from nestynet_sr.sr_search.training import (
    _attach_sr_evidence_runtime_flags,
    _sr_maybe_pull_forward_prior_decay_on_halt,
    _sr_maybe_trigger_prior_decay_from_stall,
    _run_lm_loop,
    _validate_sr_evidence_provider,
    _sr_latest_joint_loss_metrics,
    _sr_latest_single_target_loss_metrics,
    _sr_canonical_init_data_for_atom,
    _sr_canonical_init_provider,
    _sr_canonical_init_provider_and_atom,
    _sr_lm_halt_is_lam_max_rejects,
    build_sr_evidence_config,
)
from nestynet_sr.sr_core.bridges import AtomNode, Mul, Var


class _DummyOptimizer:
    def __init__(self, metrics_by_idx, *, n_base: int, n_val: int | None):
        self._metrics_by_idx = dict(metrics_by_idx)
        self.base_residual_modules = [object() for _ in range(int(n_base))]
        self.residual_modules = list(self.base_residual_modules)
        self.residual_modules_val = (
            [object() for _ in range(int(n_val))]
            if n_val is not None
            else None
        )

    def latest_loss_metrics(self, *, target_module_index=None):
        return dict(self._metrics_by_idx[int(target_module_index)])


def test_single_target_helper_allows_extra_train_only_modules_under_evidence():
    opt = _DummyOptimizer(
        {
            0: {
                "evidence_active": True,
                "train_data_mean_loss": 1.25,
                "val_data_mean_loss": 2.5,
            }
        },
        n_base=2,
        n_val=1,
    )

    metrics = _sr_latest_single_target_loss_metrics(opt, label="[test] ")

    assert metrics["evidence_active"] is True
    assert metrics["train_data_mean_loss"] == pytest.approx(1.25)
    assert metrics["val_data_mean_loss"] == pytest.approx(2.5)


def test_single_target_helper_rejects_joint_validation_targets_under_evidence():
    opt = _DummyOptimizer(
        {
            0: {
                "evidence_active": True,
                "train_data_mean_loss": 1.0,
                "val_data_mean_loss": 2.0,
            }
        },
        n_base=2,
        n_val=2,
    )

    with pytest.raises(NotImplementedError, match="plain-data aggregation policy"):
        _sr_latest_single_target_loss_metrics(opt, label="[test] ")


def test_joint_helper_sums_per_target_plain_mean_losses():
    opt = _DummyOptimizer(
        {
            0: {
                "evidence_active": True,
                "objective_is_augmented": True,
                "train_data_mean_loss": 1.5,
                "val_data_mean_loss": 2.25,
            },
            1: {
                "evidence_active": True,
                "objective_is_augmented": False,
                "train_data_mean_loss": 0.75,
                "val_data_mean_loss": 1.0,
            },
        },
        n_base=2,
        n_val=2,
    )

    metrics = _sr_latest_joint_loss_metrics(opt, target_count=2, label="[test] ")

    assert metrics["evidence_active"] is True
    assert metrics["objective_is_augmented"] is True
    assert metrics["train_data_mean_loss"] == pytest.approx(2.25)
    assert metrics["val_data_mean_loss"] == pytest.approx(3.25)
    assert len(metrics["per_target"]) == 2


def test_build_sr_evidence_config_collapses_fully_disabled_request_to_noop():
    lm_hp = LMHyperparams(
        evidence_enable=True,
        evidence_disable_residual_whitening=True,
        evidence_disable_segment_priors=True,
    )

    cfg = build_sr_evidence_config(lm_hp)

    assert cfg is None


def test_build_sr_evidence_config_rejects_residual_whitening_terms():
    lm_hp = LMHyperparams(
        evidence_enable=True,
        evidence_disable_segment_priors=True,
        evidence_lambda_patch=1.0e-2,
    )

    with pytest.raises(ValueError, match="no longer admits residual-whitening"):
        build_sr_evidence_config(lm_hp)


def test_build_sr_evidence_config_uses_fixed_decay_defaults():
    lm_hp = LMHyperparams(evidence_enable=True)

    cfg = build_sr_evidence_config(lm_hp, epochs=2000)

    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.lambda_patch == pytest.approx(0.0)
    assert cfg.patch_include_mean is False
    assert cfg.patch_include_slope is False
    assert cfg.patch_include_quad is False
    assert cfg.segment_alpha_init == pytest.approx(1.0)
    assert cfg.prior_decay_start_iter == 800
    assert cfg.prior_decay_end_iter == 900
    assert lm_hp.evidence_prior_cutoff_tol == pytest.approx(1.0e-8)
    assert cfg.prior_decay_shape == "cosine"
    assert cfg.prior_decay_final_scale == pytest.approx(0.0)
    assert cfg.log_every_accepted == 0
    assert cfg.update_alpha_every_accepted == 0
    assert cfg.allow_linear_refinement is True
    assert cfg.allow_geodesic_acceleration is False


def test_build_sr_evidence_config_moves_auto_start_earlier_to_preserve_interval():
    lm_hp = LMHyperparams(evidence_enable=True)

    cfg = build_sr_evidence_config(lm_hp, epochs=500)

    assert cfg is not None
    assert cfg.prior_decay_start_iter == 400
    assert cfg.prior_decay_end_iter == 500


def test_attach_sr_evidence_runtime_flags_floors_cutoff_tol_to_chisq_tol():
    lm_hp = LMHyperparams(
        evidence_enable=True,
        chisq_tol=1.0e-10,
        evidence_prior_cutoff_tol=1.0e-12,
    )
    cfg = build_sr_evidence_config(lm_hp, epochs=2000)

    cfg = _attach_sr_evidence_runtime_flags(cfg, lm_hp)

    assert cfg is not None
    assert cfg.sr_prior_cutoff_tol == pytest.approx(1.0e-10)
    assert cfg.sr_prior_decay_duration == 100
    assert cfg.sr_suppress_pre_decay_convergence_halt is True


def test_build_sr_evidence_config_uses_start_refine_interval():
    lm_hp = LMHyperparams(
        evidence_enable=True,
        evidence_prior_decay_auto=False,
        evidence_prior_decay_start=123,
        evidence_prior_decay_interval=45,
    )

    cfg = build_sr_evidence_config(lm_hp, epochs=1000)

    assert cfg is not None
    assert cfg.prior_decay_start_iter == 123
    assert cfg.prior_decay_end_iter == 168


def test_sr_canonical_init_provider_accepts_pure_nn_composite():
    leaf = SimpleNamespace(
        base_model=SimpleNamespace(canonical_init_greedy=lambda *a, **k: None)
    )
    model = SimpleNamespace(ast_root=SimpleNamespace(kind="nn"), leaf=[leaf])

    assert _sr_canonical_init_provider(model) is leaf
    provider, atom = _sr_canonical_init_provider_and_atom(model)
    assert provider is leaf
    assert atom is model.ast_root


def test_sr_canonical_init_provider_rejects_structured_composite():
    leaf = SimpleNamespace(
        base_model=SimpleNamespace(canonical_init_greedy=lambda *a, **k: None)
    )
    model = SimpleNamespace(ast_root=SimpleNamespace(kind="mul"), leaf=[leaf])

    assert _sr_canonical_init_provider(model) is None


def test_sr_canonical_init_data_uses_compound_effective_input():
    atom = AtomNode(kind="nn", var_idxs=(0, 1), inputs=(Mul(Var(0), Var(1)),))
    provider = SimpleNamespace(base_model=SimpleNamespace(Nx_size=1))
    x = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)
    y = torch.tensor([[7.0], [11.0]], dtype=torch.float64)

    x_eff, y_eff = _sr_canonical_init_data_for_atom(
        atom,
        provider,
        [(x, y)],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert torch.equal(x_eff, torch.tensor([[6.0], [20.0]], dtype=torch.float64))
    assert torch.equal(y_eff, y)


def test_prior_cutoff_trigger_uses_selection_loss_and_reschedules_decay():
    opt = SimpleNamespace(
        state={"iter": 601},
        iter_check=100,
        evidence_cfg=SimpleNamespace(
            enabled=True,
            sr_prior_cutoff_tol=1.0e-10,
            sr_prior_decay_duration=200,
        ),
        evidence_controller=SimpleNamespace(
            cfg=SimpleNamespace(
                prior_decay_start_iter=800,
                prior_decay_end_iter=1000,
            ),
            prior_decay_enabled=lambda: True,
            prior_decay_complete=lambda: False,
        ),
    )

    last_selection, triggered = _sr_maybe_trigger_prior_decay_from_stall(
        opt,
        loss_metrics={"train_selection_loss": 4.4789678482349323e-10},
        prev_report_train_selection_loss=5.243506485165248e-10,
        epochs=2500,
        label="[test] ",
    )

    assert triggered is True
    assert last_selection == pytest.approx(4.4789678482349323e-10)
    assert opt.evidence_controller.cfg.prior_decay_start_iter == 601
    assert opt.evidence_controller.cfg.prior_decay_end_iter == 801
    assert opt.state["sr_prior_cutoff_metric"] == "train_selection_loss"


def test_prior_cutoff_trigger_does_not_fire_on_selection_loss_regression():
    opt = SimpleNamespace(
        state={"iter": 201},
        iter_check=100,
        evidence_cfg=SimpleNamespace(
            enabled=True,
            sr_prior_cutoff_tol=1.0e-9,
            sr_prior_decay_duration=200,
        ),
        evidence_controller=SimpleNamespace(
            cfg=SimpleNamespace(
                prior_decay_start_iter=800,
                prior_decay_end_iter=1000,
            ),
            prior_decay_enabled=lambda: True,
            prior_decay_complete=lambda: False,
        ),
    )

    last_selection, triggered = _sr_maybe_trigger_prior_decay_from_stall(
        opt,
        loss_metrics={"train_selection_loss": 194.019},
        prev_report_train_selection_loss=61.2588,
        epochs=2500,
        label="[test] ",
    )

    assert triggered is False
    assert last_selection == pytest.approx(194.019)
    assert opt.evidence_controller.cfg.prior_decay_start_iter == 800
    assert opt.evidence_controller.cfg.prior_decay_end_iter == 1000


def test_pre_decay_halt_pulls_forward_decay_start_and_preserves_interval():
    opt = SimpleNamespace(
        state={"iter": 144, "halt": True},
        evidence_cfg=SimpleNamespace(
            enabled=True,
            sr_prior_decay_duration=200,
        ),
        evidence_controller=SimpleNamespace(
            cfg=SimpleNamespace(
                prior_decay_start_iter=800,
                prior_decay_end_iter=1000,
            ),
            prior_decay_enabled=lambda: True,
            prior_decay_complete=lambda: False,
        ),
    )

    changed = _sr_maybe_pull_forward_prior_decay_on_halt(
        opt,
        epochs=2500,
        label="[test] ",
    )

    assert changed is True
    assert opt.evidence_controller.cfg.prior_decay_start_iter == 144
    assert opt.evidence_controller.cfg.prior_decay_end_iter == 344
    assert opt.state["sr_prior_decay_trigger"]["kind"] == "lm_pre_decay_halt"


def test_lam_max_halt_does_not_pull_forward_prior_decay():
    opt = SimpleNamespace(
        state={
            "iter": 44,
            "halt": True,
            "lam": 1.0e12,
            "lam_at_cap_rejects": 20,
        },
        lam_max=1.0e12,
        max_lam_at_cap_rejects=20,
        evidence_cfg=SimpleNamespace(
            enabled=True,
            sr_prior_decay_duration=200,
            sr_prior_cutoff_tol=1.0e-8,
        ),
        evidence_controller=SimpleNamespace(
            cfg=SimpleNamespace(
                prior_decay_start_iter=800,
                prior_decay_end_iter=1000,
            ),
            prior_decay_enabled=lambda: True,
            prior_decay_complete=lambda: False,
        ),
    )

    changed = _sr_maybe_pull_forward_prior_decay_on_halt(
        opt,
        epochs=2500,
        label="[test] ",
    )
    current_selection, triggered = _sr_maybe_trigger_prior_decay_from_stall(
        opt,
        loss_metrics={"train_selection_loss": 1.0},
        prev_report_train_selection_loss=1.0,
        epochs=2500,
        label="[test] ",
    )

    assert _sr_lm_halt_is_lam_max_rejects(opt) is True
    assert changed is False
    assert current_selection == pytest.approx(1.0)
    assert triggered is False
    assert opt.evidence_controller.cfg.prior_decay_start_iter == 800
    assert opt.evidence_controller.cfg.prior_decay_end_iter == 1000
    assert "sr_prior_decay_trigger" not in opt.state


def test_evidence_after_step_skips_evaluate_when_log_and_alpha_updates_disabled():
    eval_calls = []
    sync_calls = []
    ctrl = SimpleNamespace(
        cfg=SimpleNamespace(
            log_every_accepted=0,
            update_alpha_every_accepted=0,
            update_only_in_normal_subphase=True,
            update_during_anneal=False,
        ),
        optimizer=SimpleNamespace(state={"trainer_subphase": "Normal", "is_anneal": False}),
        state=SimpleNamespace(accepted_steps=0),
        _maybe_recapture_prior_anchors_on_enable=lambda: False,
        _sync_state_to_optimizer=lambda: sync_calls.append("sync"),
        evaluate=lambda: eval_calls.append("evaluate") or {},
        maybe_update_segment_alphas=lambda metrics: False,
    )

    changed = EvidenceController.after_step(ctrl, accepted=True)

    assert changed is False
    assert ctrl.state.accepted_steps == 1
    assert eval_calls == []
    assert sync_calls == ["sync"]


class _DummySegmentedBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_segments = 4


class _DummySegmentedProvider(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = _DummySegmentedBase()


class _DummyCompositeProvider(torch.nn.Module):
    pass


def test_validate_sr_evidence_provider_allows_segmented_provider():
    cfg = object()

    assert _validate_sr_evidence_provider(_DummySegmentedProvider(), cfg) is cfg


def test_validate_sr_evidence_provider_disables_evidence_for_unsupported_composite():
    cfg = object()

    assert _validate_sr_evidence_provider(_DummyCompositeProvider(), cfg) is None


class _DummyEvidenceController:
    def __init__(self, *, complete: bool):
        self._complete = bool(complete)

    def prior_decay_enabled(self):
        return True

    def prior_decay_complete(self):
        return self._complete


class _LoopCallback(SRCallback):
    def __init__(self):
        self.epoch_val_losses = []
        self.final_val_loss = None

    def on_lm_epoch(self, state: SRState, epoch: int, loss: float, val_loss: float, **kwargs):
        self.epoch_val_losses.append(float(val_loss))

    def on_lm_complete(
        self, state: SRState, reason: str, final_loss: float, final_val_loss: float, **kwargs
    ):
        self.final_val_loss = float(final_val_loss)


class _DummyLoopOptimizer(_DummyOptimizer):
    def __init__(self, *, gate_complete: bool):
        super().__init__(
            {
                0: {
                    "evidence_active": True,
                    "train_data_mean_loss": 1.25,
                    "val_data_mean_loss": 2.5,
                }
            },
            n_base=1,
            n_val=1,
        )
        self.params = [torch.nn.Parameter(torch.tensor([0.0]))]
        self.state = {"halt": False, "iter": 0}
        self.iter_check = 1
        self.datagen_val = object()
        self.evidence_cfg = SimpleNamespace(
            enabled=True,
            sr_gate_metrics_until_prior_decay=True,
        )
        self.evidence_controller = _DummyEvidenceController(complete=gate_complete)

    def step(self):
        self.state["iter"] = 1
        self.state["halt"] = True
        return 1.25, 2.5


def test_run_lm_loop_tracks_pre_decay_data_validation_best_but_hides_visible_metrics():
    opt = _DummyLoopOptimizer(gate_complete=False)
    state = SRState()
    callback = _LoopCallback()

    accepted, best_val_loss, best_train_loss, best_param_vec = _run_lm_loop(
        opt,
        epochs=0,
        epochs_min=0,
        nval_patience=1,
        loss_target=10.0,
        accept_threshold=3.0,
        callback=callback,
        state=state,
    )

    assert accepted is True
    assert best_val_loss == pytest.approx(2.5)
    assert best_train_loss == pytest.approx(1.25)
    assert best_param_vec is None
    assert state.current_val_loss == float("inf")
    assert state.best_val_loss == pytest.approx(2.5)
    assert callback.epoch_val_losses == [float("inf")]
    assert callback.final_val_loss == pytest.approx(2.5)


class _DeferredGateLoopOptimizer(_DummyOptimizer):
    def __init__(self):
        super().__init__(
            {
                0: {
                    "evidence_active": True,
                    "train_data_mean_loss": 1.25,
                    "val_data_mean_loss": 2.5,
                }
            },
            n_base=1,
            n_val=1,
        )
        self.params = [torch.nn.Parameter(torch.tensor([0.0]))]
        self.state = {"halt": False, "iter": 0}
        self.iter_check = 1
        self.datagen_val = object()
        self.evidence_cfg = SimpleNamespace(
            enabled=True,
            sr_gate_metrics_until_prior_decay=True,
        )
        self.evidence_controller = SimpleNamespace(
            prior_decay_enabled=lambda: True,
            prior_decay_complete=lambda: int(self.state.get("iter", 0)) >= 2,
        )

    def step(self):
        self.state["iter"] = int(self.state.get("iter", 0)) + 1
        self.state["halt"] = True
        return 1.25, 2.5


def test_run_lm_loop_defers_lm_halt_until_metrics_gate_opens():
    opt = _DeferredGateLoopOptimizer()

    accepted, best_val_loss, best_train_loss, best_param_vec = _run_lm_loop(
        opt,
        epochs=3,
        epochs_min=0,
        nval_patience=1,
        loss_target=10.0,
        accept_threshold=3.0,
        track_params=True,
    )

    assert accepted is True
    assert best_val_loss == pytest.approx(2.5)
    assert best_train_loss == pytest.approx(1.25)
    assert torch.equal(best_param_vec, torch.tensor([0.0]))
    assert opt.state["_sr_deferred_halt_until_metrics_ready"] is True


class _LamMaxHaltLoopOptimizer(_DummyOptimizer):
    def __init__(self):
        super().__init__(
            {
                0: {
                    "evidence_active": True,
                    "train_data_mean_loss": 1.25,
                    "val_data_mean_loss": 2.5,
                }
            },
            n_base=1,
            n_val=1,
        )
        self.params = [torch.nn.Parameter(torch.tensor([0.0]))]
        self.state = {"halt": False, "iter": 0, "lam": 1.0e12}
        self.iter_check = 1
        self.datagen_val = object()
        self.lam_max = 1.0e12
        self.max_lam_at_cap_rejects = 20
        self.evidence_cfg = SimpleNamespace(
            enabled=True,
            sr_gate_metrics_until_prior_decay=True,
            sr_prior_decay_duration=200,
        )
        self.evidence_controller = SimpleNamespace(
            cfg=SimpleNamespace(
                prior_decay_start_iter=800,
                prior_decay_end_iter=1000,
            ),
            prior_decay_enabled=lambda: True,
            prior_decay_complete=lambda: False,
        )
        self.step_calls = 0

    def step(self):
        self.step_calls += 1
        self.state["iter"] = 44
        self.state["halt"] = True
        self.state["lam_at_cap_rejects"] = 20
        return 1.25, 2.5


def test_run_lm_loop_honors_lam_max_halt_before_metrics_gate_opens():
    opt = _LamMaxHaltLoopOptimizer()

    accepted, best_val_loss, best_train_loss, best_param_vec = _run_lm_loop(
        opt,
        epochs=3,
        epochs_min=0,
        nval_patience=1,
        loss_target=10.0,
        accept_threshold=1.0,
        track_params=True,
    )

    assert opt.step_calls == 1
    assert opt.state["halt"] is True
    assert opt.state["sr_pre_decay_terminal_halt"]["kind"] == "lam_max_rejects"
    assert "_sr_deferred_halt_until_metrics_ready" not in opt.state
    assert opt.evidence_controller.cfg.prior_decay_start_iter == 800
    assert opt.evidence_controller.cfg.prior_decay_end_iter == 1000
    assert accepted is False
    assert best_val_loss == pytest.approx(2.5)
    assert best_train_loss == pytest.approx(1.25)
    assert torch.equal(best_param_vec, torch.tensor([0.0]))


class _FirstStepFailureOptimizer(_DummyOptimizer):
    """LM step raises on the very first call (the pb107 crash mode)."""

    def __init__(self, exc: Exception):
        super().__init__({0: {}}, n_base=1, n_val=1)
        self.iter_check = 1
        self._exc = exc
        self.step_calls = 0

    def step(self):
        self.step_calls += 1
        raise self._exc


class _LateStepFailureOptimizer(_DummyOptimizer):
    """LM step succeeds once (recording a best), then raises."""

    def __init__(self):
        super().__init__(
            {
                0: {
                    "train_data_mean_loss": 1.25,
                    "val_data_mean_loss": 2.5,
                }
            },
            n_base=1,
            n_val=1,
        )
        self.iter_check = 1
        self.params = [torch.nn.Parameter(torch.tensor([0.0]))]
        self.state = {"halt": False, "iter": 0}
        self.step_calls = 0

    def step(self):
        self.step_calls += 1
        if self.step_calls == 1:
            self.state["iter"] = 1
            return 1.25, 2.5
        raise RuntimeError(
            "Dense direct solve failed: Cholesky, dense solve, and residual-gated "
            "ridged Cholesky fallback all failed."
        )


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError(
            "Dense direct solve failed: Cholesky, dense solve, and residual-gated "
            "ridged Cholesky fallback all failed."
        ),
        FloatingPointError("overflow encountered in exp"),
    ],
)
def test_run_lm_loop_rejects_candidate_on_first_step_solver_failure(exc):
    # A pathological candidate whose first LM step blows up must be rejected
    # gracefully (inf loss, no params), not crash the run (pb107 regression).
    opt = _FirstStepFailureOptimizer(exc)

    accepted, best_val_loss, best_train_loss, best_param_vec = _run_lm_loop(
        opt,
        epochs=5,
        epochs_min=0,
        nval_patience=1,
        accept_threshold=1.0e-6,
    )

    assert opt.step_calls == 1  # halted on the first failing step
    assert accepted is False
    assert best_val_loss == float("inf")
    assert best_train_loss == float("inf")
    assert best_param_vec is None


def test_run_lm_loop_keeps_best_so_far_on_later_step_solver_failure():
    # A step failure after a successful epoch keeps the best-so-far rather than
    # discarding progress or crashing.
    opt = _LateStepFailureOptimizer()

    accepted, best_val_loss, best_train_loss, best_param_vec = _run_lm_loop(
        opt,
        epochs=5,
        epochs_min=0,
        nval_patience=3,
        accept_threshold=3.0,
        track_params=True,
    )

    assert opt.step_calls == 2  # one good step, then the failing one halts
    assert best_val_loss == pytest.approx(2.5)  # best-so-far preserved, not discarded
    assert best_train_loss == pytest.approx(1.25)
    assert accepted is True  # best-so-far cleared the accept threshold
    del best_param_vec  # param-vec capture needs real optimizer internals
