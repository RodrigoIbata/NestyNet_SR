# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Tests for Stage-B Pareto-style decision logging."""

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import Add, Log, Var
from nestynet_sr.sr_search.stageB.engine import Candidate, StageBContext, StageBState


class _DummyModel(torch.nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([float(value)], dtype=torch.float64))
        self.leaf = torch.nn.ModuleList([])

    def forward(self, x):
        return x[:, :1] * 0.0 + self.weight

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def _make_ctx(*, root=None) -> StageBContext:
    if root is None:
        root = Add(Var(0), Var(1))
    state = StageBState(root=root, model=_DummyModel(1.0), reuse={}, val_loss=1.0e-4)
    lm_hp = SimpleNamespace(
        fit_y_link=None,
        fit_y_link_scale=1.0,
        loss_acceptable=1.0,
        select_stageB_max_decades_over_floor=1.0,
        select_count_weight=1.0,
    )
    return StageBContext(
        state=state,
        train_loader=object(),
        val_loader=object(),
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


def test_record_decision_logs_mapping_and_complexity_metrics():
    ctx = _make_ctx()
    cand_root = Log(Var(0))
    cand = Candidate(
        label="factorized_search_candidate",
        root=cand_root,
        meta={
            "factorized_mapping": {"kind": "pade", "numer": [0.0, 1.0, 0.0], "denom": [1.0, 0.0, 0.0]},
            "mapping_cost": 12.0,
            "coordinate_variant": "z_inv",
            "coordinate_variant_display": "1/z",
            "pattern_family": "factorized_search",
        },
    )

    rec = ctx._record_decision(
        outcome="reject",
        rule="factorized_search",
        label=cand.label,
        reason="test",
        target="x0",
        base_loss=1.0e-4,
        cand_loss=1.0e-5,
        n_params_base=3,
        n_params_cand=2,
        cand=cand,
        base_root=ctx.state.root,
        cand_root=cand_root,
        base_mapping_cost=0.0,
        cand_mapping_cost=12.0,
    )

    assert rec["cand_mapping_kind"] == "pade"
    assert rec["cand_mapping_class"] == "approximative"
    assert rec["cand_mapping_is_structural"] is False
    assert rec["coordinate_variant"] == "z_inv"
    assert rec["coordinate_variant_display"] == "1/z"
    assert rec["pattern_family"] == "factorized_search"
    assert rec["cand_ast_cost"] is not None
    assert rec["cand_complexity_score"] is not None
    assert rec["cand_complexity_total"] is not None
    assert rec["cand_complexity_total"] > rec["cand_complexity_score"]
    assert rec["pareto_trackable"] is True
    assert rec["ast_snapshot"]
    assert "log" in rec["ast_snapshot"].lower()


def test_pareto_front_records_keeps_only_non_dominated():
    ctx = _make_ctx()

    cand_a = Candidate(label="a_simple", root=Var(0))
    cand_b = Candidate(label="b_better_loss", root=Log(Log(Var(0))))
    cand_c = Candidate(label="c_dominated", root=Log(Var(0)))

    ctx._record_decision(
        outcome="reject",
        rule="r",
        label=cand_a.label,
        reason="test",
        target="x0",
        base_loss=1.0e-4,
        cand_loss=1.0e-6,
        n_params_base=3,
        n_params_cand=2,
        cand=cand_a,
        base_root=ctx.state.root,
        cand_root=cand_a.root,
        cand_mapping_cost=0.0,
    )
    ctx._record_decision(
        outcome="reject",
        rule="r",
        label=cand_b.label,
        reason="test",
        target="x0",
        base_loss=1.0e-4,
        cand_loss=5.0e-7,
        n_params_base=3,
        n_params_cand=2,
        cand=cand_b,
        base_root=ctx.state.root,
        cand_root=cand_b.root,
        cand_mapping_cost=0.0,
    )
    ctx._record_decision(
        outcome="reject",
        rule="r",
        label=cand_c.label,
        reason="test",
        target="x0",
        base_loss=1.0e-4,
        cand_loss=2.0e-6,
        n_params_base=3,
        n_params_cand=2,
        cand=cand_c,
        base_root=ctx.state.root,
        cand_root=cand_c.root,
        cand_mapping_cost=0.0,
    )

    front = ctx.pareto_front_records(outcomes={"reject"})
    labels = [rec["label"] for rec in front]

    assert "a_simple" in labels
    assert "b_better_loss" in labels
    assert "c_dominated" not in labels
