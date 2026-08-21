# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import torch

from nestynet_sr.sr_core import ast_to_human_readable
from nestynet_sr.sr_core.bridges import AddNode, AtomNode, MulNode, PowNode, Var
from nestynet_sr.sr_search.stageB import rules as stageb_rules
from nestynet_sr.sr_search.stageB.rules import RuleJointProductMonomialClosure


class _PowerLeaf(torch.nn.Module):
    def __init__(self, power: float):
        super().__init__()
        self.power = float(power)

    def forward(self, x):
        z = torch.clamp(x[:, :1].to(dtype=torch.float64), min=1.0e-12)
        return z.pow(self.power)


def _pb116_like_root():
    u = PowNode(MulNode(PowNode(Var(0), -2.0), Var(2)), 2.0)
    leaf_u = AtomNode(kind="nn", var_idxs=(0, 2), tag="leaf_u", inputs=(u,))
    leaf_x2 = AtomNode(kind="nn", var_idxs=(2,), tag="leaf_x2", inputs=(Var(2),))
    return MulNode(leaf_u, leaf_x2), leaf_u, leaf_x2


def _ctx(root, leaf_u, leaf_x2):
    grid0 = torch.linspace(1.2, 4.0, 24, dtype=torch.float64)
    grid2 = torch.linspace(0.8, 3.2, 24, dtype=torch.float64)
    x0, x2 = torch.meshgrid(grid0, grid2, indexing="ij")
    x1 = torch.ones_like(x0)
    X = torch.stack((x0.reshape(-1), x1.reshape(-1), x2.reshape(-1)), dim=1)
    y = torch.zeros((X.shape[0], 1), dtype=torch.float64)
    logs = []
    return SimpleNamespace(
        state=SimpleNamespace(root=root, model=object()),
        train_loader_probe=[(X, y)],
        device=torch.device("cpu"),
        dtype=torch.float64,
        units_spec=None,
        enforce_units=False,
        lm_hp=SimpleNamespace(),
        log=logs.append,
        _logs=logs,
        _leaf_map={
            id(leaf_u): _PowerLeaf(0.25),
            id(leaf_x2): _PowerLeaf(1.5),
        },
    )


def _nn_atoms(node):
    out = []

    def _walk(n):
        if isinstance(n, AtomNode):
            if str(n.kind).lower() == "nn":
                out.append(n)
        elif isinstance(n, (AddNode, MulNode)):
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, PowNode):
            _walk(n.base)

    _walk(node)
    return out


def test_joint_product_monomial_closure_pb116_retained_axis_gauge(monkeypatch):
    root, leaf_u, leaf_x2 = _pb116_like_root()
    ctx = _ctx(root, leaf_u, leaf_x2)
    monkeypatch.setattr(stageb_rules, "build_atom_to_leaf_map", lambda _root, _model: ctx._leaf_map)

    cands = RuleJointProductMonomialClosure().propose(ctx, root)

    assert len(cands) == 1
    cand = cands[0]
    assert cand.label == "joint_product_monomial_closure"
    assert cand.meta["joint_product_support"] == (0, 2)
    assert cand.meta["joint_product_exponents"] == (-1, 2)
    assert cand.meta["min_free_params"] == 1
    assert not _nn_atoms(cand.root)
    text = ast_to_human_readable(cand.root).replace(" ", "")
    assert "(x0)**-1" in text
    assert "(x2)**2" in text


def test_joint_product_monomial_closure_rejects_nonmonomial_product(monkeypatch):
    root, leaf_u, leaf_x2 = _pb116_like_root()
    ctx = _ctx(root, leaf_u, leaf_x2)

    class _NonMonomialLeaf(torch.nn.Module):
        def forward(self, x):
            z = x[:, :1].to(dtype=torch.float64)
            return 1.0 + z

    ctx._leaf_map[id(leaf_u)] = _NonMonomialLeaf()
    monkeypatch.setattr(stageb_rules, "build_atom_to_leaf_map", lambda _root, _model: ctx._leaf_map)

    cands = RuleJointProductMonomialClosure().propose(ctx, root)

    assert cands == []
    assert any("reason=" in msg for msg in ctx._logs)


def test_joint_product_integer_monomial_screen_allows_validation_probe_window():
    x0_vals = torch.linspace(1.0, 1.5, 24, dtype=torch.float64)
    x2_vals = torch.linspace(0.8, 3.2, 24, dtype=torch.float64)
    x0, x2 = torch.meshgrid(x0_vals, x2_vals, indexing="ij")
    x1 = torch.ones_like(x0)
    X = torch.stack((x0.reshape(-1), x1.reshape(-1), x2.reshape(-1)), dim=1)
    y = 0.5 * x0.reshape(-1).pow(-0.92) * x2.reshape(-1).pow(2.0)

    diag = {}
    fit = stageb_rules._fit_joint_product_integer_monomial(X, y, (0, 2), diag=diag)

    assert fit is not None, diag
    assert fit["exponents"] == (-1, 2)
    assert 0.05 < max(abs(a - b) for a, b in zip(fit["raw_exponents"], fit["exponents"])) < 0.10
    assert 1.0e-3 < fit["rel_rms"] < 1.0e-2
