# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import AtomNode, MulNode, PowNode, Var
from nestynet_sr.sr_search.monomial_screen import (
    candidate_monomial_exponent,
    fit_univariate_monomial_screen,
)
from nestynet_sr.sr_search.search import _stageA_monomial_has_shared_multiplicative_nn_support
from nestynet_sr.sr_search.search import _stageA_monomial_should_use_reduced_form
from nestynet_sr.sr_search.search import _stageA_cap_terminal_analytic_threshold
from nestynet_sr.sr_search.stageB.engine import (
    Candidate,
    StageBContext,
    StageBEngine,
    StageBRule,
    StageBState,
)
from nestynet_sr.sr_search.stageB.rules import RuleMonomialPeelPriority, RuleUniNN


class _SquareTeacher(torch.nn.Module):
    def forward(self, x):
        return x[:, :1] ** 2


class _ExpTeacher(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x[:, :1])


class _SqrtTeacher(torch.nn.Module):
    def __init__(self, reciprocal=False, abs_input=False):
        super().__init__()
        self.reciprocal = bool(reciprocal)
        self.abs_input = bool(abs_input)

    def forward(self, x):
        z = x[:, :1].to(dtype=torch.float64)
        if self.abs_input:
            z = torch.abs(z)
        z = torch.clamp(z, min=1.0e-12)
        out = torch.sqrt(z)
        if self.reciprocal:
            out = torch.reciprocal(out)
        return out


class _ParamCountModel(torch.nn.Module):
    def __init__(self, n_params=0):
        super().__init__()
        self._n_params = int(n_params)
        if self._n_params > 0:
            self.p = torch.nn.Parameter(torch.zeros(self._n_params, dtype=torch.float64))

    def num_parameters(self):
        return self._n_params


class _FakeRule(StageBRule):
    name = "univariate_nn"
    exhaustive = True

    def iter_targets(self, ctx):
        return []

    def propose(self, ctx, target):
        return []


def _ctx(root, reuse, loader):
    state = StageBState(
        root=root,
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse=reuse,
        val_loss=1.0e-10,
    )
    return StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
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


def _exact_stop_ctx(root):
    state = StageBState(
        root=root,
        model=_ParamCountModel(4),
        reuse={},
        val_loss=1.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=[],
        val_loader=[],
        lm_hp=SimpleNamespace(stageB_candidate_policy="sequential"),
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
    ctx.loss_floor = 1.0e-8
    ctx._record_decision = lambda **kwargs: {}
    ctx.maybe_shadow_polish_subtrees_after_accept = lambda **kwargs: False
    ctx.maybe_polish_after_accept = lambda: False
    ctx.precheck_candidate = (
        lambda rule_name, cand, record_attempt=False: SimpleNamespace(
            ok=True,
            reason=None,
            signature=None,
        )
    )
    ctx.should_accept = lambda cand, cand_state: (True, "loss-below-floor-simpler")
    ctx.gauge_acceptance_gate = lambda cand, cand_state, reason: (True, reason)
    return ctx


def _state(root, val_loss):
    return StageBState(
        root=root,
        model=_ParamCountModel(1),
        reuse={},
        val_loss=float(val_loss),
    )


def _cand(label, root):
    return Candidate(label=label, root=root, meta={"n_free_params": 1})


def _run_exact_stop_case(base_root, candidates, states):
    ctx = _exact_stop_ctx(base_root)
    fit_order = []
    accepted = []

    def _fit_candidate(cand, epochs_override=None):
        fit_order.append(cand.label)
        return states[cand.label]

    def _accept(cand, cand_state, reason):
        accepted.append((cand.label, reason))
        ctx.state = cand_state

    ctx.fit_candidate = _fit_candidate
    ctx.accept = _accept

    ok = StageBEngine([])._try_candidates_for_target(
        ctx,
        _FakeRule(),
        "univariate_nn",
        base_root,
        candidates,
        exhaustive=True,
    )
    return ok, fit_order, accepted


def test_loglog_screen_recovers_clean_square():
    x = torch.linspace(1.0, 5.0, 256, dtype=torch.float64)
    y = 3.0 * x**2

    screen = fit_univariate_monomial_screen(x, y)

    assert screen.ok
    assert abs(screen.k_hat - 2.0) < 1.0e-10
    assert screen.rel_rms < 1.0e-10


def test_candidate_monomial_exponent_reads_reciprocal_aliases():
    assert candidate_monomial_exponent("monomial_deg2") == 2.0
    assert candidate_monomial_exponent("monomial_deg2[z_inv]") == -2.0
    assert candidate_monomial_exponent("monomial_pow1_2") == 0.5
    assert candidate_monomial_exponent("monomial_pow3_2[z_inv]") == -1.5
    assert candidate_monomial_exponent("ratpoly_1d") is None


def test_univariate_monomial_rule_snaps_clean_positive_half_power():
    atom = AtomNode(kind="nn", var_idxs=(0,), tag="leaf")
    x = torch.linspace(1.0, 5.0, 128, dtype=torch.float64).reshape(-1, 1)
    loader = DataLoader(TensorDataset(x, torch.zeros_like(x)), batch_size=128)
    ctx = _ctx(atom, {"leaf": _SqrtTeacher()}, loader)

    cands = RuleUniNN(monomial_only=True).propose(ctx, atom)
    labels = [str(c.label) for c in cands]

    assert "monomial_pow1_2" in labels


def test_univariate_monomial_rule_uses_z_inv_for_inverse_half_power():
    atom = AtomNode(kind="nn", var_idxs=(0,), tag="leaf")
    x = torch.linspace(1.0, 5.0, 128, dtype=torch.float64).reshape(-1, 1)
    loader = DataLoader(TensorDataset(x, torch.zeros_like(x)), batch_size=128)
    ctx = _ctx(atom, {"leaf": _SqrtTeacher(reciprocal=True)}, loader)

    cands = RuleUniNN(monomial_only=True).propose(ctx, atom)
    labels = [str(c.label) for c in cands]

    assert "monomial_pow1_2" not in labels
    assert "monomial_pow1_2[z_inv]" in labels


def test_univariate_monomial_rule_snaps_clean_compound_half_power():
    z_expr = MulNode(PowNode(Var(0), 3.0), PowNode(Var(1), -1.0))
    atom = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf", inputs=(z_expr,))
    x0 = torch.linspace(1.0, 4.0, 128, dtype=torch.float64)
    x1 = torch.linspace(2.0, 5.0, 128, dtype=torch.float64)
    x = torch.stack([x0, x1], dim=1)
    loader = DataLoader(TensorDataset(x, torch.zeros(x.shape[0], 1, dtype=torch.float64)), batch_size=128)
    ctx = _ctx(atom, {"leaf": _SqrtTeacher()}, loader)

    cands = RuleUniNN(monomial_only=True).propose(ctx, atom)
    labels = [str(c.label) for c in cands]

    assert "monomial_pow1_2" in labels


def test_univariate_monomial_rule_rejects_half_power_on_negative_domain():
    atom = AtomNode(kind="nn", var_idxs=(0,), tag="leaf")
    x = torch.linspace(-5.0, -1.0, 128, dtype=torch.float64).reshape(-1, 1)
    loader = DataLoader(TensorDataset(x, torch.zeros_like(x)), batch_size=128)
    ctx = _ctx(atom, {"leaf": _SqrtTeacher(abs_input=True)}, loader)

    cands = RuleUniNN(monomial_only=True).propose(ctx, atom)
    labels = [str(c.label) for c in cands]

    assert "monomial_pow1_2" not in labels
    assert "monomial_pow1_2[z_inv]" not in labels


def test_stageB_global_monomial_priority_prefers_clean_raw_variable_peel():
    ratio = MulNode(Var(2), PowNode(Var(1), -1.0))
    ratio_atom = AtomNode(kind="nn", var_idxs=(1, 2), tag="ratio", inputs=(ratio,))
    raw_atom = AtomNode(kind="nn", var_idxs=(1,), tag="raw")
    root = MulNode(ratio_atom, raw_atom)

    x1 = torch.linspace(1.0, 5.0, 96, dtype=torch.float64)
    x2 = torch.linspace(2.0, 7.0, 96, dtype=torch.float64)
    grid1, grid2 = torch.meshgrid(x1, x2, indexing="ij")
    x = torch.stack(
        [
            torch.ones_like(grid1).reshape(-1),
            grid1.reshape(-1),
            grid2.reshape(-1),
        ],
        dim=1,
    )
    loader = DataLoader(TensorDataset(x, torch.zeros(x.shape[0], 1, dtype=torch.float64)), batch_size=1024)
    ctx = _ctx(root, {"ratio": _ExpTeacher(), "raw": _SquareTeacher()}, loader)

    rule = RuleMonomialPeelPriority()
    targets = sorted(rule.iter_targets(ctx), key=lambda a: tuple(a.var_idxs))
    entries = rule.propose_global_candidates(ctx, targets)

    assert entries
    first_target, first_cand = entries[0]
    assert first_target is raw_atom
    assert first_cand.label == "monomial_deg2"
    assert first_cand.meta["pattern"] == "monomial_peel_priority"
    assert first_cand.meta["monomial_screen_ok"] is True


def test_stageB_exact_final_leaf_monomial_stops_exhaustive_search():
    base = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0")
    mono = AtomNode(kind="poly", var_idxs=(0,), tag="mono")
    later = AtomNode(kind="sqrt_poly", var_idxs=(0,), tag="sqrt")

    ok, fit_order, accepted = _run_exact_stop_case(
        base,
        [_cand("monomial_deg2", mono), _cand("sqrt_poly", later)],
        {
            "monomial_deg2": _state(mono, 1.0e-12),
            "sqrt_poly": _state(later, 1.0e-12),
        },
    )

    assert ok is True
    assert fit_order == ["monomial_deg2"]
    assert accepted == [("monomial_deg2", "loss-below-floor-simpler")]


def test_stageB_exact_nonmonomial_final_leaf_does_not_stop_exhaustive_search():
    base = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0")
    sqrt_root = AtomNode(kind="sqrt_poly", var_idxs=(0,), tag="sqrt")
    trig_root = AtomNode(kind="sin_linear", var_idxs=(0,), tag="trig")

    ok, fit_order, accepted = _run_exact_stop_case(
        base,
        [_cand("sqrt_poly", sqrt_root), _cand("trig", trig_root)],
        {
            "sqrt_poly": _state(sqrt_root, 1.0e-12),
            "trig": _state(trig_root, 2.0e-12),
        },
    )

    assert ok is True
    assert fit_order == ["sqrt_poly", "trig"]
    assert accepted == [("sqrt_poly", "loss-below-floor-simpler")]


def test_stageB_monomial_below_floor_with_remaining_nn_does_not_stop_exhaustive_search():
    base = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0")
    remaining = AtomNode(kind="nn", var_idxs=(1,), tag="leaf1")
    mono_with_nn = MulNode(AtomNode(kind="poly", var_idxs=(0,), tag="mono"), remaining)
    sqrt_root = AtomNode(kind="sqrt_poly", var_idxs=(0,), tag="sqrt")

    ok, fit_order, accepted = _run_exact_stop_case(
        base,
        [_cand("monomial_deg2", mono_with_nn), _cand("sqrt_poly", sqrt_root)],
        {
            "monomial_deg2": _state(mono_with_nn, 1.0e-12),
            "sqrt_poly": _state(sqrt_root, 2.0e-12),
        },
    )

    assert ok is True
    assert fit_order == ["monomial_deg2", "sqrt_poly"]
    assert accepted == [("monomial_deg2", "loss-below-floor-simpler")]


def test_stageA_monomial_guard_allows_disjoint_product_but_blocks_shared_nn_gauge():
    mono = AtomNode(kind="nn", var_idxs=(0,), tag="mono")
    disjoint_rest = AtomNode(kind="nn", var_idxs=(1, 2), tag="rest")
    shared_rest = AtomNode(kind="nn", var_idxs=(0, 1), tag="shared")

    assert not _stageA_monomial_has_shared_multiplicative_nn_support(
        MulNode(mono, disjoint_rest),
        mono,
    )
    assert _stageA_monomial_has_shared_multiplicative_nn_support(
        MulNode(mono, shared_rest),
        mono,
    )


def test_stageA_monomial_uses_reduced_form_only_when_nn_sibling_absorbs_scale():
    mono = AtomNode(kind="nn", var_idxs=(0,), tag="mono")
    nn_rest = AtomNode(kind="nn", var_idxs=(1, 2), tag="rest")

    assert _stageA_monomial_should_use_reduced_form(MulNode(mono, nn_rest), mono)
    assert not _stageA_monomial_should_use_reduced_form(MulNode(mono, Var(1)), mono)
    assert not _stageA_monomial_should_use_reduced_form(mono, mono)


def test_stageA_terminal_analytic_closure_uses_absolute_cap():
    base = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0")
    terminal = AtomNode(kind="poly", var_idxs=(0,), tag="leaf0")
    nonterminal = MulNode(AtomNode(kind="poly", var_idxs=(0,), tag="leaf0"), AtomNode(kind="nn", var_idxs=(1,), tag="leaf1"))

    threshold, capped = _stageA_cap_terminal_analytic_threshold(
        base_ast=base,
        cand_ast=terminal,
        accept_threshold=1.0,
        absolute_cap=1.0e-4,
    )
    assert capped is True
    assert threshold == 1.0e-4

    threshold2, capped2 = _stageA_cap_terminal_analytic_threshold(
        base_ast=base,
        cand_ast=nonterminal,
        accept_threshold=1.0,
        absolute_cap=1.0e-4,
    )
    assert capped2 is False
    assert threshold2 == 1.0
