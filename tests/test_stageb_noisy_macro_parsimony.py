# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, Var
from nestynet_sr.sr_search.stageB.engine import Candidate, StageBEngine, StageBState


class _TinyModel(torch.nn.Module):
    def num_parameters(self):
        return 0


class _DummyRule:
    def describe_target(self, _target):
        return "dummy-target"

    def candidate_min_free_params(self, cand):
        return int((cand.meta or {}).get("min_free_params", 0))


class _Ctx(SimpleNamespace):
    def __init__(self, *, noise_floor, coe_veto_labels=()):
        root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
        super().__init__(
            state=StageBState(root=root, model=_TinyModel(), reuse={}, val_loss=1.0, num_nn_atoms=1),
            lm_hp=SimpleNamespace(
                stageB_candidate_policy="sequential",
                stageB_screen_enable=False,
                acceptance_noise_floor_raw=float(noise_floor),
                acceptance_noise_n_eff=100.0,
                macro_noisy_parsimony_noise_mult=2.0,
                macro_noisy_parsimony_rel_tol=1.0e-3,
                stageB_noisy_terminal_noise_mult=1.0,
                stageB_noisy_terminal_rel_tol=1.0e-3,
            ),
            loss_scale=1.0,
            loss_floor=1.0e-12,
            loss_good_enough_raw=1.0e-12,
            score_tol=0.0,
            attempted_transformations={},
            _rejected_keys=set(),
            _red_set=set(),
            _last_amber_key=None,
            logs=[],
            decisions=[],
            accepted=[],
            coe_veto_labels=set(str(x) for x in coe_veto_labels),
        )

    def log(self, msg):
        self.logs.append(str(msg))

    def _record_decision(self, **kwargs):
        self.decisions.append(dict(kwargs))

    def precheck_candidate(self, _rule_name, cand, record_attempt=True):
        return SimpleNamespace(ok=True, reason=None, signature=getattr(cand, "signature", None))

    def fit_candidate(self, cand, epochs_override=None):
        return StageBState(
            root=cand.root,
            model=_TinyModel(),
            reuse={},
            val_loss=float((cand.meta or {}).get("fake_loss", 1.0)),
            num_nn_atoms=0,
        )

    def should_accept(self, _cand, _cand_state):
        return True, "test-accept"

    def _select_ratpoly_candidate(self, cand, cand_state):
        return True, cand, cand_state, "test-accept"

    def gauge_acceptance_gate(self, _cand, _cand_state, reason):
        return True, reason

    def coe_stageB_committee_gate(self, **_kwargs):
        if str(_kwargs.get("label", "")) in self.coe_veto_labels:
            return False, "reject-test-coe"
        return True, "coe-stageB-refit-accepted"

    def accept(self, cand, cand_state, reason):
        self.accepted.append((cand.label, reason, float(cand_state.val_loss)))
        self.state = cand_state

    def maybe_shadow_polish_subtrees_after_accept(self, **_kwargs):
        return None

    def maybe_polish_after_accept(self):
        return None


def _clean_singleton(loss=1.01e-3):
    return Candidate(
        label="cf_inv_sqrt1m",
        root=Var(0),
        meta={
            "fake_loss": float(loss),
            "macro_clean_singleton": True,
            "macro_combo": False,
            "n_terms": 1,
            "min_free_params": 2,
        },
        signature=("clean",),
    )


def _rich_combo(loss=1.00e-3):
    return Candidate(
        label="cf_combo",
        root=AddNode(Var(0), Var(0)),
        meta={
            "fake_loss": float(loss),
            "macro_clean_singleton": False,
            "macro_combo": True,
            "n_terms": 2,
            "min_free_params": 4,
        },
        signature=("combo",),
    )


def _sparse_ratpoly(loss=1.01e-3):
    return Candidate(
        label="ratpoly_1d",
        root=Var(0),
        meta={
            "fake_loss": float(loss),
            "pattern_family": "ratpoly_1d",
            "terminal_family": "ratpoly_1d",
            "terminal_protected": True,
            "terminal_priority_family": "sparse_mobius_1d",
            "deg_num": 1,
            "deg_den": 1,
            "n_terms_num": 2,
            "n_terms_den": 2,
            "terminal_n_terms": 4,
            "min_free_params": 4,
        },
        signature=("ratpoly",),
    )


def _leaftr(loss=1.00e-3):
    return Candidate(
        label="leaftr_sqrt_poly3",
        root=AddNode(Var(0), Var(0)),
        meta={
            "fake_loss": float(loss),
            "terminal_family": "leaf_transform_poly",
            "terminal_flexible_approximant": True,
            "leaftr_transform": "sqrt",
            "leaftr_degree": 3,
            "terminal_n_terms": 4,
            "min_free_params": 4,
        },
        signature=("leaftr",),
    )


def test_noisy_macro_parsimony_prefers_tied_clean_singleton():
    ctx = _Ctx(noise_floor=1.0e-3)
    engine = StageBEngine([])

    ok = engine._try_candidates_for_target(
        ctx,
        _DummyRule(),
        "compound_fn_macros",
        ctx.state.root,
        [_clean_singleton(), _rich_combo()],
        exhaustive=False,
    )

    assert ok is True
    assert ctx.accepted
    assert ctx.accepted[0][0] == "cf_inv_sqrt1m"
    assert any("Noisy macro parsimony tournament" in msg for msg in ctx.logs)


def test_noiseless_macro_path_keeps_proposal_order():
    ctx = _Ctx(noise_floor=0.0)
    engine = StageBEngine([])

    ok = engine._try_candidates_for_target(
        ctx,
        _DummyRule(),
        "compound_fn_macros",
        ctx.state.root,
        [_rich_combo(), _clean_singleton()],
        exhaustive=False,
    )

    assert ok is True
    assert ctx.accepted
    assert ctx.accepted[0][0] == "cf_combo"
    assert not any("Noisy macro parsimony tournament" in msg for msg in ctx.logs)


def test_noisy_terminal_tournament_prefers_tied_sparse_ratpoly():
    ctx = _Ctx(noise_floor=1.0e-3)
    engine = StageBEngine([])

    ok = engine._try_candidates_for_target(
        ctx,
        _DummyRule(),
        "univariate_nn",
        ctx.state.root,
        [_leaftr(loss=1.00e-3), _sparse_ratpoly(loss=1.01e-3)],
        exhaustive=True,
    )

    assert ok is True
    assert ctx.accepted
    assert ctx.accepted[0][0] == "ratpoly_1d"
    assert any("Noisy terminal tournament tied set" in msg for msg in ctx.logs)


def test_noiseless_terminal_tournament_keeps_loss_order():
    ctx = _Ctx(noise_floor=0.0)
    engine = StageBEngine([])

    ok = engine._try_candidates_for_target(
        ctx,
        _DummyRule(),
        "univariate_nn",
        ctx.state.root,
        [_leaftr(loss=1.00e-3), _sparse_ratpoly(loss=1.01e-3)],
        exhaustive=True,
    )

    assert ok is True
    assert ctx.accepted
    assert ctx.accepted[0][0] == "leaftr_sqrt_poly3"
    assert not any("Noisy terminal tournament tied set" in msg for msg in ctx.logs)


def test_noisy_terminal_tournament_keeps_materially_better_leaftr():
    ctx = _Ctx(noise_floor=1.0e-3)
    engine = StageBEngine([])

    ok = engine._try_candidates_for_target(
        ctx,
        _DummyRule(),
        "univariate_nn",
        ctx.state.root,
        [_leaftr(loss=1.00e-3), _sparse_ratpoly(loss=1.30e-3)],
        exhaustive=True,
    )

    assert ok is True
    assert ctx.accepted
    assert ctx.accepted[0][0] == "leaftr_sqrt_poly3"
    assert not any("Noisy terminal tournament tied set" in msg for msg in ctx.logs)


def test_noisy_terminal_tournament_falls_back_after_coe_veto():
    ctx = _Ctx(noise_floor=1.0e-3, coe_veto_labels={"ratpoly_1d"})
    engine = StageBEngine([])

    ok = engine._try_candidates_for_target(
        ctx,
        _DummyRule(),
        "univariate_nn",
        ctx.state.root,
        [_leaftr(loss=1.00e-3), _sparse_ratpoly(loss=1.01e-3)],
        exhaustive=True,
    )

    assert ok is True
    assert ctx.accepted
    assert ctx.accepted[0][0] == "leaftr_sqrt_poly3"
    assert any(d.get("label") == "ratpoly_1d" and d.get("outcome") == "reject" for d in ctx.decisions)
