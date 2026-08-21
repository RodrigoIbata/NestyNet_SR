# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from nestynet_sr.sr_search.stageB.engine import (
    Candidate,
    StageBContext,
    StageBRule,
    StageBState,
    _are_we_done_yet,
    _are_we_done_yet_reason,
    _best_seen_restore_decision,
    _below_floor_regression_cap,
    _below_floor_regression_rejected,
    _candidate_can_beat_floor_locked_state,
    _candidate_min_free_params,
    _effective_loss_floor,
    _is_separability_candidate,
)
from nestynet_sr.sr_core.bridges import Add, AtomNode, Var
from types import SimpleNamespace

import torch


class _DummyModel(torch.nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([float(value)], dtype=torch.float64))
        self.leaf = torch.nn.ModuleList([])

    def forward(self, x):
        return x[:, :1] * 0.0 + self.weight

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def _make_ctx(*, root=None, val_loss=1.0e-9) -> StageBContext:
    if root is None:
        root = Add(Var(0), Var(1))
    state = StageBState(root=root, model=_DummyModel(1.0), reuse={}, val_loss=val_loss)
    lm_hp = SimpleNamespace(
        fit_y_link=None,
        fit_y_link_scale=1.0,
        loss_acceptable=1.0,
        acceptance_noise_floor=None,
        acceptance_noise_floor_raw=None,
        stageB_overcap_fallback=False,
        select_stageB_max_decades_over_floor=1.0,
        select_count_weight=1.0,
        select_floor_guard_decades=2.0,
        select_below_floor_max_regress_decades=1.0,
    )
    batch = torch.ones(8, 2, dtype=torch.float64)
    return StageBContext(
        state=state,
        train_loader=[batch],
        val_loader=[batch],
        lm_hp=lm_hp,
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=5,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-3,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )


def test_below_floor_regression_rejects_non_separability_large_regressions():
    cap = _below_floor_regression_cap(base_loss=1.0e-12, max_regress_decades=1.0)
    assert _below_floor_regression_rejected(
        cand_loss=1.0e-10,
        below_floor_regress_cap=cap,
        is_separability_rewrite=False,
    )


def test_below_floor_regression_allows_separability_even_when_large():
    cap = _below_floor_regression_cap(base_loss=1.0e-12, max_regress_decades=0.0)
    assert not _below_floor_regression_rejected(
        cand_loss=1.0e-8,
        below_floor_regress_cap=cap,
        is_separability_rewrite=True,
    )


def test_below_floor_regression_accepts_within_cap():
    cap = _below_floor_regression_cap(base_loss=1.0e-12, max_regress_decades=2.0)
    assert not _below_floor_regression_rejected(
        cand_loss=1.0e-11,
        below_floor_regress_cap=cap,
        is_separability_rewrite=False,
    )


def test_metadata_marked_candidate_counts_as_separability_like():
    cand = Candidate(
        label="outer_identity_mul",
        root=Add(Var(0), Var(1)),
        meta={"structural": True, "separability_like": True},
    )

    assert _is_separability_candidate(cand) is True


def test_below_floor_accept_uses_metadata_separability_bonus():
    root = Add(Var(0), Var(1))
    ctx = _make_ctx(root=root, val_loss=7.85e-10)
    cand = Candidate(
        label="outer_identity_mul",
        root=root,
        meta={"structural": True, "separability_like": True},
    )
    cand_state = StageBState(root=root, model=_DummyModel(1.0), reuse={}, val_loss=3.37e-9)

    accepted, reason = ctx.should_accept(cand, cand_state)

    assert accepted is True
    assert reason == "loss-below-floor-separability-pass"


def test_effective_loss_floor_uses_nominal_floor_when_base_is_zero():
    floor_eff = _effective_loss_floor(loss_floor=1.0e-7, base_loss=0.0, guard_decades=2.0)
    assert floor_eff == 1.0e-7


def test_best_seen_restore_keeps_below_floor_analytic_state_over_lower_loss_nn_state():
    cur_root = Add(Var(0), Var(1))
    best_root = Add(Var(0), AtomNode(kind="nn", var_idxs=(1, 3, 4), kwargs={}))

    restore, reason = _best_seen_restore_decision(
        cur_loss=2.3090e-11,
        best_loss=1.4660e-13,
        cur_root=cur_root,
        best_root=best_root,
        n_params_cur=18,
        n_params_best=961,
        loss_floor=3.1918e-2,
        loss_floor_eff=1.4660e-11,
        count_weight=1.0,
    )

    assert not restore
    assert "keeping current complexity" in reason


def test_best_seen_restore_still_prefers_better_loss_when_nn_structure_is_same():
    cur_root = Add(Var(0), Var(1))
    best_root = Add(Var(0), Var(1))

    restore, reason = _best_seen_restore_decision(
        cur_loss=2.3090e-11,
        best_loss=1.4660e-13,
        cur_root=cur_root,
        best_root=best_root,
        n_params_cur=18,
        n_params_best=18,
        loss_floor=3.1918e-2,
        loss_floor_eff=1.4660e-11,
        count_weight=1.0,
    )

    assert restore
    assert "loss=" in reason


def test_best_seen_restore_keeps_noise_equivalent_terminal_state_over_nn_state():
    cur_root = Add(Var(0), Var(1))
    best_root = Add(Var(0), AtomNode(kind="nn", var_idxs=(1,), kwargs={}))

    restore, reason = _best_seen_restore_decision(
        cur_loss=9.389e-12,
        best_loss=8.750e-12,
        cur_root=cur_root,
        best_root=best_root,
        n_params_cur=3,
        n_params_best=384,
        loss_floor=6.736e-12,
        loss_floor_eff=6.736e-12,
        count_weight=1.0,
        losses_noise_equivalent=True,
    )

    assert not restore
    assert "keeping current complexity" in reason


def test_best_seen_restore_still_uses_loss_when_not_noise_equivalent():
    cur_root = Add(Var(0), Var(1))
    best_root = Add(Var(0), AtomNode(kind="nn", var_idxs=(1,), kwargs={}))

    restore, reason = _best_seen_restore_decision(
        cur_loss=9.389e-12,
        best_loss=8.750e-12,
        cur_root=cur_root,
        best_root=best_root,
        n_params_cur=3,
        n_params_best=384,
        loss_floor=6.736e-12,
        loss_floor_eff=6.736e-12,
        count_weight=1.0,
        losses_noise_equivalent=False,
    )

    assert restore
    assert "loss=" in reason


def test_should_accept_preserves_hard_cap_by_default():
    base_root = Add(AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}), Var(0))
    cand_root = Add(Var(0), Var(1))

    ctx = _make_ctx(root=base_root, val_loss=2.0e-2)
    cand = Candidate(
        label="drop_nn",
        root=cand_root,
        meta={"structural": True},
    )
    cand_state = StageBState(root=cand_root, model=_DummyModel(1.0), reuse={}, val_loss=2.1e-2)

    accepted, reason = ctx.should_accept(cand, cand_state)

    assert accepted is False
    assert reason.startswith("simpler-over-budget")


def test_should_accept_ignores_hard_cap_when_fallback_is_enabled():
    base_root = Add(AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}), Var(0))
    cand_root = Add(Var(0), Var(1))

    ctx = _make_ctx(root=base_root, val_loss=2.0e-2)
    ctx.lm_hp.stageB_overcap_fallback = True
    cand = Candidate(
        label="drop_nn",
        root=cand_root,
        meta={"structural": True},
    )
    cand_state = StageBState(root=cand_root, model=_DummyModel(1.0), reuse={}, val_loss=2.1e-2)

    accepted, reason = ctx.should_accept(cand, cand_state)

    assert accepted is True
    assert reason.startswith("simpler-within-budget")


def test_should_accept_allows_noise_equivalent_simplification_with_explicit_noise_floor():
    base_root = Add(AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}), Var(0))
    cand_root = Add(Var(0), Var(1))

    ctx = _make_ctx(root=base_root, val_loss=2.0e-2)
    ctx.lm_hp.stageB_overcap_fallback = True
    ctx.lm_hp.acceptance_noise_floor_raw = 5.0e-3
    cand = Candidate(
        label="drop_nn",
        root=cand_root,
        meta={"structural": True},
    )
    cand_state = StageBState(root=cand_root, model=_DummyModel(1.0), reuse={}, val_loss=2.1e-2)

    accepted, reason = ctx.should_accept(cand, cand_state)

    assert accepted is True
    assert reason.startswith("noise-equivalent-simpler")


def test_should_accept_rejects_noisy_gauge_sideways_simplification():
    base_root = Add(AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}), Var(0))
    cand_root = Add(Var(0), Var(1))

    ctx = _make_ctx(root=base_root, val_loss=2.0e-2)
    ctx.lm_hp.stageB_overcap_fallback = True
    ctx.lm_hp.acceptance_noise_floor_raw = 5.0e-3
    cand = Candidate(
        label="common_prefactor",
        root=cand_root,
        meta={
            "structural": True,
            "noisy_gauge_requires_strict_improvement": True,
        },
    )
    cand_state = StageBState(root=cand_root, model=_DummyModel(1.0), reuse={}, val_loss=2.1e-2)

    accepted, reason = ctx.should_accept(cand, cand_state)

    assert accepted is False
    assert reason.startswith("reject-noisy-gauge-sideways")


def test_should_accept_allows_noisy_gauge_strict_improvement():
    base_root = Add(AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}), Var(0))
    cand_root = Add(Var(0), Var(1))

    ctx = _make_ctx(root=base_root, val_loss=2.0e-2)
    ctx.lm_hp.stageB_overcap_fallback = True
    ctx.lm_hp.acceptance_noise_floor_raw = 5.0e-3
    cand = Candidate(
        label="common_prefactor",
        root=cand_root,
        meta={
            "structural": True,
            "noisy_gauge_requires_strict_improvement": True,
        },
    )
    cand_state = StageBState(root=cand_root, model=_DummyModel(1.0), reuse={}, val_loss=1.0e-3)

    accepted, reason = ctx.should_accept(cand, cand_state)

    assert accepted is True
    assert reason == "better-loss"


def test_should_accept_uses_n_eff_scaled_noise_equivalence_when_available():
    base_root = Add(AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}), Var(0))
    cand_root = Add(Var(0), Var(1))

    ctx = _make_ctx(root=base_root, val_loss=2.0e-2)
    ctx.lm_hp.stageB_overcap_fallback = True
    ctx.lm_hp.acceptance_noise_floor_raw = 5.0e-3
    ctx.acceptance_noise_n_eff = 2000.0
    cand = Candidate(
        label="drop_nn",
        root=cand_root,
        meta={"structural": True},
    )
    cand_state = StageBState(root=cand_root, model=_DummyModel(1.0), reuse={}, val_loss=2.1e-2)

    accepted, reason = ctx.should_accept(cand, cand_state)

    assert accepted is False
    assert reason.startswith("simpler-over-budget")


def test_should_accept_rejects_non_equivalent_overcap_with_explicit_noise_floor():
    base_root = Add(AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}), Var(0))
    cand_root = Add(Var(0), Var(1))

    ctx = _make_ctx(root=base_root, val_loss=2.0e-2)
    ctx.lm_hp.stageB_overcap_fallback = True
    ctx.lm_hp.acceptance_noise_floor_raw = 5.0e-3
    cand = Candidate(
        label="drop_nn",
        root=cand_root,
        meta={"structural": True},
    )
    cand_state = StageBState(root=cand_root, model=_DummyModel(1.0), reuse={}, val_loss=3.0e-2)

    accepted, reason = ctx.should_accept(cand, cand_state)

    assert accepted is False
    assert reason.startswith("simpler-over-budget")


def test_are_we_done_yet_allows_fitted_analytic_constants_below_floor():
    ctx = _make_ctx(root=Add(Var(0), Var(1)), val_loss=1.0e-9)
    ctx.state.num_nn_atoms = 0

    assert ctx.state.model.num_parameters() > 0
    assert _are_we_done_yet(ctx) is True


def test_are_we_done_yet_uses_noise_equivalent_completion_boundary():
    noise_floor = 1.0e-4
    floor = 1.0e-3
    ctx = _make_ctx(root=Add(Var(0), Var(1)), val_loss=noise_floor + floor + 5.0e-7)
    ctx.loss_floor = floor
    ctx.lm_hp.acceptance_noise_floor_raw = noise_floor
    ctx.state.acceptance_noise_floor_raw = noise_floor
    ctx.state.acceptance_noise_n_eff = 10000
    ctx.state.num_nn_atoms = 0

    assert _are_we_done_yet(ctx) is True

    ctx.state.val_loss = noise_floor + floor + 5.0e-6
    assert _are_we_done_yet(ctx) is False


def test_are_we_done_yet_reason_reports_noise_n_eff_without_crashing():
    ctx = _make_ctx(root=Add(Var(0), Var(1)), val_loss=1.0e-9)
    ctx.state.num_nn_atoms = 0
    ctx.state.acceptance_noise_n_eff = 2000

    reason = _are_we_done_yet_reason(ctx)

    assert "noise_n_eff=2000.0" in reason


def test_are_we_done_yet_waits_for_following_lower_min_param_candidates():
    ctx = _make_ctx(root=Add(Var(0), Var(1)), val_loss=1.0e-9)
    ctx.state.num_nn_atoms = 0
    assert ctx.state.model.num_parameters() == 1

    lower = Candidate(label="zero_param_candidate", root=Add(Var(0), Var(1)), meta={"min_free_params": 0})
    equal = Candidate(label="one_param_candidate", root=Add(Var(0), Var(1)), meta={"min_free_params": 1})
    higher = Candidate(label="two_param_candidate", root=Add(Var(0), Var(1)), meta={"min_free_params": 2})

    assert _are_we_done_yet(ctx, following_candidates=[lower]) is False
    assert _are_we_done_yet(ctx, following_candidates=[equal]) is True
    assert _are_we_done_yet(ctx, following_candidates=[higher]) is True


def test_are_we_done_yet_rejects_unresolved_or_above_floor_states():
    nn_root = Add(AtomNode(kind="nn", var_idxs=(0,), kwargs={}), Var(1))
    ctx = _make_ctx(root=nn_root, val_loss=1.0e-9)
    ctx.state.num_nn_atoms = 1

    assert _are_we_done_yet(ctx) is False

    analytic_ctx = _make_ctx(root=Add(Var(0), Var(1)), val_loss=1.0)
    analytic_ctx.state.num_nn_atoms = 0

    assert _are_we_done_yet(analytic_ctx) is False


def test_candidate_min_free_params_centralizes_explicit_lower_bound():
    root = Add(Var(0), Var(1))

    assert _candidate_min_free_params(Candidate(label="plain", root=root)) == 0
    assert _candidate_min_free_params(
        Candidate(label="legacy", root=root, meta={"n_free_params": 3})
    ) == 3
    assert _candidate_min_free_params(
        Candidate(
            label="explicit",
            root=root,
            meta={"n_free_params": 3, "min_free_params": 1},
        )
    ) == 1


def test_stageb_rule_can_publish_candidate_min_free_params():
    class _Rule(StageBRule):
        def iter_targets(self, ctx):
            return []

        def propose(self, ctx, target):
            return []

        def candidate_min_free_params(self, cand):
            return 0

    ctx = _make_ctx(root=Add(Var(0), Var(1)), val_loss=1.0e-9)
    ctx.state.num_nn_atoms = 0
    cand = Candidate(label="metadata_says_larger", root=Add(Var(0), Var(1)), meta={"min_free_params": 5})
    rule = _Rule()

    assert rule.candidate_min_free_params(cand) == 0
    assert _are_we_done_yet(
        ctx,
        following_candidates=[cand],
        candidate_min_free_params_fn=rule.candidate_min_free_params,
    ) is False


def test_floor_locked_bypass_requires_simpler_minimum_complexity():
    root = Add(Var(0), Var(1))
    ctx = _make_ctx(root=root, val_loss=1.0e-9)
    ctx.state.num_nn_atoms = 0

    worse = Candidate(label="generic_ratpoly", root=root, meta={"min_free_params": 2})
    better = Candidate(label="snapped_formula", root=root, meta={"min_free_params": 0})

    assert _candidate_can_beat_floor_locked_state(ctx, worse) is False
    assert _candidate_can_beat_floor_locked_state(ctx, better) is True


def test_are_we_done_yet_waits_on_unpromoted_generic_when_menu_remains():
    ctx = _make_ctx(root=Add(Var(0), Var(1)), val_loss=1.0e-9)
    ctx.state.num_nn_atoms = 0
    ctx.state.generic_approximant_unpromoted = True

    equal = Candidate(label="one_param_candidate", root=Add(Var(0), Var(1)), meta={"min_free_params": 1})

    assert _are_we_done_yet(ctx, following_candidates=[equal]) is False
    assert _are_we_done_yet(ctx, following_candidates=[]) is True


def test_floor_locked_bypass_allows_removing_remaining_nn_even_with_params():
    nn_root = Add(AtomNode(kind="nn", var_idxs=(0,), kwargs={}), Var(1))
    ctx = _make_ctx(root=nn_root, val_loss=1.0e-9)
    ctx.state.num_nn_atoms = 1

    analytic = Candidate(label="analytic_closure", root=Add(Var(0), Var(1)), meta={"min_free_params": 5})

    assert _candidate_can_beat_floor_locked_state(ctx, analytic) is True
