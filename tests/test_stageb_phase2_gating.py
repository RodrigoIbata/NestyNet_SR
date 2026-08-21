# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import AtomNode
from nestynet_sr.sr_search.stageB.engine import (
    StageBContext,
    StageBEngine,
    StageBRule,
    StageBState,
    _phase2_trigger_flags,
)


class _RecordingRule(StageBRule):
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def iter_targets(self, _ctx):
        self.calls.append(self.name)
        return []

    def propose(self, _ctx, _target):
        return []


def _single_nn_context():
    state = StageBState(
        root=AtomNode("nn", (0,), tag="leaf0"),
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.0,
        num_nn_atoms=1,
        num_multivar_nn_atoms=0,
        max_nn_arity=1,
    )
    return StageBContext(
        state=state,
        train_loader=[],
        val_loader=[],
        lm_hp=SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-7,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )


def test_trig_power_runs_before_generic_univariate_rewrite():
    calls = []
    engine = StageBEngine(
        [
            _RecordingRule("last_hard_trig_power", calls),
            _RecordingRule("univariate_nn", calls),
        ]
    )

    engine.run(_single_nn_context(), max_outer_iters=1)

    assert calls == ["last_hard_trig_power", "univariate_nn"]


def test_phase2_runs_when_no_phase1_improvement():
    run_phase2, only_nonstruct, only_nonstruct_map = _phase2_trigger_flags(
        improved=False,
        phase1_accept_count=0,
        phase1_structural_accept_count=0,
        phase1_mapping_accept_count=0,
        phase1_mapping_structural_accept_count=0,
    )
    assert run_phase2 is True
    assert only_nonstruct is False
    assert only_nonstruct_map is False


def test_phase2_runs_when_phase1_only_nonstructural_accepts():
    run_phase2, only_nonstruct, only_nonstruct_map = _phase2_trigger_flags(
        improved=True,
        phase1_accept_count=2,
        phase1_structural_accept_count=0,
        phase1_mapping_accept_count=0,
        phase1_mapping_structural_accept_count=0,
    )
    assert run_phase2 is True
    assert only_nonstruct is True
    assert only_nonstruct_map is False


def test_phase2_runs_when_mapping_accepts_are_only_nonstructural():
    # Structural-label accepts can still hide a non-structural mapping win.
    run_phase2, only_nonstruct, only_nonstruct_map = _phase2_trigger_flags(
        improved=True,
        phase1_accept_count=1,
        phase1_structural_accept_count=1,
        phase1_mapping_accept_count=1,
        phase1_mapping_structural_accept_count=0,
    )
    assert run_phase2 is True
    assert only_nonstruct is False
    assert only_nonstruct_map is True


def test_phase2_skips_when_phase1_has_structural_progress():
    run_phase2, only_nonstruct, only_nonstruct_map = _phase2_trigger_flags(
        improved=True,
        phase1_accept_count=2,
        phase1_structural_accept_count=2,
        phase1_mapping_accept_count=1,
        phase1_mapping_structural_accept_count=1,
    )
    assert run_phase2 is False
    assert only_nonstruct is False
    assert only_nonstruct_map is False
