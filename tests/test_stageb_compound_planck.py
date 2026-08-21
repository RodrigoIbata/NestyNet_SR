# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import AtomNode, MulNode, Var, ast_equals
from nestynet_sr.sr_search import candidate_builders
from nestynet_sr.sr_search.stageB import helpers
from nestynet_sr.sr_search.stageB.engine import Candidate
from nestynet_sr.sr_search.stageB.rules import RuleCompoundPlanck


def _prod(i: int, j: int):
    return MulNode(Var(i), Var(j))


def _ctx(root):
    return SimpleNamespace(
        state=SimpleNamespace(root=root, reuse={}),
        train_loader_probe=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def test_compound_planck_tries_both_two_compound_orientations(monkeypatch):
    p = _prod(0, 1)
    q = _prod(2, 3)
    target = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3), tag="leaf0", inputs=(p, q))
    calls = []

    def fake_builder(**kwargs):
        calls.append((kwargs["label"], kwargs["z_expr"], kwargs["extra_prod"]))
        return Candidate(
            label=kwargs["label"],
            root=kwargs["root"],
            meta={"pattern_family": "compound_planck"},
            signature=kwargs["signature_extra"],
        )

    monkeypatch.setattr(helpers, "_build_planck_derived_feature_candidate", fake_builder)

    rule = RuleCompoundPlanck()
    assert rule.iter_targets(_ctx(target)) == [target]

    cands = rule.propose(_ctx(target), target)

    assert [c.label for c in cands] == ["compound_planck[0]", "compound_planck[1]"]
    assert any(ast_equals(z, p) and ast_equals(u, q) for _label, z, u in calls)
    assert any(ast_equals(z, q) and ast_equals(u, p) for _label, z, u in calls)
    assert len({c.signature for c in cands}) == 2


def test_compound_planck_preserves_legacy_compound_refine_raw_extras(monkeypatch):
    q = _prod(2, 3)
    target = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="leaf0",
        inputs=(q, Var(0), Var(1)),
    )
    calls = []

    def fake_builder(**kwargs):
        calls.append((kwargs["label"], kwargs["z_expr"], kwargs["extra_prod"]))
        return Candidate(
            label=kwargs["label"],
            root=kwargs["root"],
            meta={"pattern_family": "compound_planck"},
            signature=kwargs["signature_extra"],
        )

    monkeypatch.setattr(helpers, "_build_planck_derived_feature_candidate", fake_builder)

    cands = RuleCompoundPlanck().propose(_ctx(target), target)

    assert len(cands) == 3
    assert any(ast_equals(z, q) and ast_equals(u, _prod(0, 1)) for _label, z, u in calls)


def test_reduced_planck_emits_small_structural_power_dictionary(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0")
    root = target

    def fake_gather(*_args, **_kwargs):
        x = torch.linspace(1.0, 10.0, 500, dtype=torch.float64)
        f = torch.exp(-0.2 * x)
        return x, f

    def fake_fixed_fit(_x, _f, *, p_fixed, **_kwargs):
        return float(p_fixed), 0.2 + 0.01 * float(p_fixed), -1.0, abs(float(p_fixed) - 1.0)

    monkeypatch.setattr(candidate_builders, "_gather_teacher_data_1d", fake_gather)
    monkeypatch.setattr(candidate_builders, "_fit_planck_tail_fixed_power", fake_fixed_fit)

    cands = candidate_builders._build_planck_1d_candidates(
        root=root,
        target=target,
        reuse={"leaf0": torch.nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert [label for label, _root, _init, _meta in cands] == ["planck_p0", "planck_p1", "planck_p2"]
    assert [meta["planck_power"] for _label, _root, _init, meta in cands] == [0.0, 1.0, 2.0]
    assert all(meta["pattern_family"] == "planck" for _label, _root, _init, meta in cands)
    assert all(meta["min_free_params"] == 2 for _label, _root, _init, meta in cands)
