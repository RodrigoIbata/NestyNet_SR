# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import AtomNode, MulNode, PowNode, Var
from nestynet_sr.sr_search.stageB.engine import Candidate, StageBContext, StageBEngine, StageBState
from nestynet_sr.sr_search.stageB.homogeneous_gauge_scope import (
    HomogeneousGaugeScopeIndex,
    homogeneous_gauge_global_score,
    parse_ratio_monomial,
)
from nestynet_sr.sr_search.stageB.rules import RuleMultiplicativeHomogeneityTransfer


def _ratio_nn(tag="leaf"):
    z = MulNode(Var(2), PowNode(Var(1), -1.0))
    return AtomNode(kind="nn", var_idxs=(1, 2), tag=tag, inputs=(z,))


def _root(atom=None):
    atom = atom or _ratio_nn()
    return MulNode(PowNode(Var(1), 2.0), atom)


def _ctx(root, reuse=None, train_loader=None):
    state = StageBState(
        root=root,
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse=reuse or {},
        val_loss=1.0e-8,
    )
    return StageBContext(
        state=state,
        train_loader=train_loader,
        val_loader=train_loader,
        lm_hp=SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-8,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )


def test_parse_ratio_monomial_handles_reciprocal_product():
    z = MulNode(Var(2), PowNode(Var(1), -1.0))
    parsed = parse_ratio_monomial(z)

    assert parsed is not None
    assert parsed.numerator_var == 2
    assert parsed.denominator_var == 1


def test_homogeneous_gauge_scope_detects_power_times_ratio_nn():
    atom = _ratio_nn()
    root = _root(atom)
    idx = HomogeneousGaugeScopeIndex(root)

    assert len(idx.unresolved_scopes) == 1
    scope = idx.unresolved_scopes[0]
    assert scope.ratio_atom is atom
    assert scope.power_factor.var_idx == 1
    assert scope.power_factor.degree == 2.0
    assert scope.ratio.numerator_var == 2
    assert scope.ratio.denominator_var == 1
    assert idx.scope_for_target(atom) is scope


def test_homogeneous_gauge_scope_reads_rpoly_monomial_degree():
    atom = _ratio_nn()
    rpoly_power = AtomNode(kind="rpoly", var_idxs=(1,), kwargs={"degree": 2, "min_total": 2}, tag="p")
    root = MulNode(rpoly_power, atom)
    idx = HomogeneousGaugeScopeIndex(root)

    assert len(idx.unresolved_scopes) == 1
    scope = idx.unresolved_scopes[0]
    assert scope.power_factor.node is rpoly_power
    assert scope.power_factor.var_idx == 1
    assert scope.power_factor.degree == 2.0


def test_homogeneous_gauge_scope_detects_powered_ratio_nn_factor():
    atom = _ratio_nn()
    root = MulNode(PowNode(Var(1), 2.0), PowNode(atom, -1.0))
    idx = HomogeneousGaugeScopeIndex(root)

    assert len(idx.unresolved_scopes) == 1
    scope = idx.unresolved_scopes[0]
    assert scope.ratio_factor is root.right
    assert scope.ratio_atom is atom
    assert scope.ratio_power == -1.0
    assert idx.scope_for_target(atom) is scope


def test_homogeneous_gauge_score_improves_when_ratio_nn_is_closed():
    base = _root(_ratio_nn())
    closed = MulNode(PowNode(Var(2), 2.0), PowNode(AtomNode(kind="poly", var_idxs=(1, 2), inputs=(MulNode(Var(1), PowNode(Var(2), -1.0)),)), -0.5))

    assert homogeneous_gauge_global_score(closed) < homogeneous_gauge_global_score(base)


def test_homogeneous_gauge_acceptance_gate_rejects_marked_no_improvement():
    root = _root(_ratio_nn())
    ctx = _ctx(root)
    cand = Candidate(
        "ratpoly_1d",
        root,
        meta={
            "homogeneous_gauge_requires_scope_improvement": True,
            "homogeneous_gauge_score_before": ctx.homogeneous_gauge_global_score(),
        },
    )
    cand_state = StageBState(
        root=root,
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse={},
        val_loss=1.0e-12,
    )

    ok, reason = ctx.gauge_acceptance_gate(cand, cand_state, "better-loss")

    assert not ok
    assert reason == "reject-unresolved-homogeneous-gauge-local-compression"


def test_engine_marks_homogeneous_gauge_sensitive_univariate_candidates():
    atom = _ratio_nn()
    root = _root(atom)
    ctx = _ctx(root)
    cand = Candidate("ratpoly_1d", atom)

    marked = StageBEngine([])._mark_gauge_tainted_candidates(
        ctx,
        "univariate_nn",
        atom,
        [cand],
    )

    assert marked[0].meta["homogeneous_gauge_sensitive"] is False
    assert marked[0].meta["homogeneous_gauge_requires_scope_improvement"] is True
    assert marked[0].meta["homogeneous_gauge_scope_uid"] == "hom:0"
    assert marked[0].meta["homogeneous_gauge_score_before"] == ctx.homogeneous_gauge_global_score()


def test_engine_marks_homogeneous_gauge_sensitive_local_compression_rules():
    atom = _ratio_nn()
    root = _root(atom)
    ctx = _ctx(root)
    cand = Candidate("homogeneity_peel", atom)

    marked = StageBEngine([])._mark_gauge_tainted_candidates(
        ctx,
        "homogeneity_peel",
        atom,
        [cand],
    )

    assert marked[0].meta["homogeneous_gauge_sensitive"] is True
    assert marked[0].meta["homogeneous_gauge_requires_scope_improvement"] is True


class _BadRepresentativeTeacher(torch.nn.Module):
    def forward(self, x):
        z = x[:, :1]
        return (z * z) / torch.sqrt(torch.clamp(1.0 - z.reciprocal() ** 2, min=1.0e-12))


class _PoweredBadRepresentativeTeacher(torch.nn.Module):
    def forward(self, x):
        z = x[:, :1]
        return torch.sqrt(torch.clamp(1.0 - z.reciprocal() ** 2, min=1.0e-12)) / (z * z)


def test_multiplicative_homogeneity_transfer_proposes_visible_sqrt_poly_closure():
    atom = _ratio_nn("bad")
    root = _root(atom)
    x1 = torch.linspace(1.0, 2.0, 48, dtype=torch.float64)
    x2 = torch.linspace(3.0, 5.0, 48, dtype=torch.float64)
    grid1, grid2 = torch.meshgrid(x1, x2, indexing="ij")
    x = torch.stack([torch.ones_like(grid1).reshape(-1), grid1.reshape(-1), grid2.reshape(-1)], dim=1)
    loader = DataLoader(TensorDataset(x, torch.zeros(x.shape[0], 1, dtype=torch.float64)), batch_size=512)
    ctx = _ctx(root, reuse={"bad": _BadRepresentativeTeacher()}, train_loader=loader)

    cands = RuleMultiplicativeHomogeneityTransfer().propose(ctx, root)
    labels = [str(c.label) for c in cands]

    assert any("sqrt_poly" in label for label in labels)
    assert all(c.meta["pattern"] == "multiplicative_homogeneity_transfer" for c in cands)
    assert all(c.meta["homogeneous_gauge_confirmed"] is True for c in cands)
    assert all(homogeneous_gauge_global_score(c.root) < homogeneous_gauge_global_score(root) for c in cands)


def test_multiplicative_homogeneity_transfer_handles_powered_ratio_nn_factor():
    atom = _ratio_nn("powered")
    root = MulNode(PowNode(Var(1), 2.0), PowNode(atom, -1.0))
    x1 = torch.linspace(1.0, 2.0, 48, dtype=torch.float64)
    x2 = torch.linspace(3.0, 5.0, 48, dtype=torch.float64)
    grid1, grid2 = torch.meshgrid(x1, x2, indexing="ij")
    x = torch.stack([torch.ones_like(grid1).reshape(-1), grid1.reshape(-1), grid2.reshape(-1)], dim=1)
    loader = DataLoader(TensorDataset(x, torch.zeros(x.shape[0], 1, dtype=torch.float64)), batch_size=512)
    ctx = _ctx(root, reuse={"powered": _PoweredBadRepresentativeTeacher()}, train_loader=loader)

    cands = RuleMultiplicativeHomogeneityTransfer().propose(ctx, root)

    assert cands
    assert any(c.meta["homogeneous_gauge_ratio_power"] == -1.0 for c in cands)
    assert any("sqrt_poly" in str(c.label) for c in cands)
    assert all(homogeneous_gauge_global_score(c.root) < homogeneous_gauge_global_score(root) for c in cands)
