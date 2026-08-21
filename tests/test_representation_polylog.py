# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import torch

from nestynet_sr.sr_core.atoms import PolyLogLeaf
from nestynet_sr.sr_core.bridges import AtomNode, MulNode, PowNode, Var
from nestynet_sr.sr_search.representation import _leaf_to_repr, _leaf_to_str, _polylog_str


def _assert_uses_ratio_coordinate(expr: str):
    compact = expr.replace(" ", "")
    assert compact.startswith("-log(")
    assert "x3" in compact
    assert "x4" in compact
    assert compact != "-log(x3)"


def test_polylog_str_uses_compound_input_expr():
    exps = torch.tensor([[0], [1]], dtype=torch.int64)
    coeffs = torch.tensor([0.0, -1.0], dtype=torch.float64)
    z_expr = MulNode(Var(3), PowNode(Var(4), -1.0))

    expr = _polylog_str(exps, coeffs, (3, 4), input_expr=z_expr)

    _assert_uses_ratio_coordinate(expr)


def test_polylog_leaf_printers_use_compound_input_expr():
    z_expr = MulNode(Var(3), PowNode(Var(4), -1.0))
    atom = AtomNode(kind="polylog", var_idxs=(3, 4), tag="leaf3", inputs=(z_expr,))
    leaf = PolyLogLeaf(n_in=1, degree=1, dtype=torch.float64)
    with torch.no_grad():
        leaf.coeffs.zero_()
        linear_idx = int((leaf.exps[:, 0] == 1).nonzero(as_tuple=False)[0, 0])
        leaf.coeffs[linear_idx] = -1.0

    legacy_expr = _leaf_to_str(atom, leaf)
    scale, structured_expr = _leaf_to_repr(atom, leaf)

    assert scale == 1.0
    _assert_uses_ratio_coordinate(legacy_expr)
    _assert_uses_ratio_coordinate(structured_expr)
