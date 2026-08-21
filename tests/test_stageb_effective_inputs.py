# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    LogNode,
    MulNode,
    PowNode,
    Var,
    ast_to_human_readable,
    get_input_exprs,
)
from nestynet_sr.sr_search.candidate_builders import (
    _build_affine_decomp_candidate,
    _build_nonlinear_sub_candidate,
)
from nestynet_sr.sr_search.feature_grammar import build_arg_pool


def _prod_var(i: int, j: int) -> MulNode:
    return MulNode(left=Var(i), right=Var(j))


def _two_compound_target() -> AtomNode:
    return AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        kwargs={},
        tag="leaf0",
        inputs=(_prod_var(0, 1), _prod_var(2, 3)),
    )


def _find_atom(node, kinds):
    if isinstance(node, AtomNode) and str(node.kind).lower() in kinds:
        return node
    for child in (
        getattr(node, "left", None),
        getattr(node, "right", None),
        getattr(node, "base", None),
        getattr(node, "arg", None),
    ):
        if child is None:
            continue
        found = _find_atom(child, kinds)
        if found is not None:
            return found
    return None


def test_feature_pool_uses_all_effective_compound_inputs():
    target = _two_compound_target()

    pool = build_arg_pool(
        target,
        max_vars=4,
        max_args=256,
        include_compound_expr=True,
        trig=False,
    )

    by_kind_desc = {(entry.kind, entry.desc) for entry in pool}
    assert ("compound", "arg0") in by_kind_desc
    assert ("compound_input", "arg1") in by_kind_desc
    assert ("effective_prod", "(arg0*arg1)") in by_kind_desc
    assert ("effective_ratio", "(arg0/arg1)") in by_kind_desc
    assert ("effective_ratio", "(arg1/arg0)") in by_kind_desc


def test_feature_pool_includes_reciprocal_effective_coordinate():
    target = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        kwargs={},
        tag="leaf0",
        inputs=(MulNode(left=Var(0), right=PowNode(base=Var(1), exponent=-1.0)),),
    )

    pool = build_arg_pool(
        target,
        max_vars=2,
        max_args=256,
        include_compound_expr=True,
        trig=False,
    )

    by_kind_desc = {(entry.kind, entry.desc) for entry in pool}
    assert ("compound", "arg0") in by_kind_desc
    assert ("compound", "1/arg0") in by_kind_desc
    assert ("compound_sq", "arg0^2") in by_kind_desc
    assert ("compound_sq", "1/arg0^2") in by_kind_desc


def test_nonlinear_substitution_wraps_selected_compound_input():
    torch.manual_seed(12)
    n = 900
    x0 = torch.rand(n, dtype=torch.float64) + 1.0
    x1 = torch.rand(n, dtype=torch.float64) + 1.0
    x2 = torch.rand(n, dtype=torch.float64) + 1.0
    x3 = torch.rand(n, dtype=torch.float64) + 1.0
    x_full = torch.stack((x0, x1, x2, x3), dim=1)

    target = _two_compound_target()

    class Teacher(torch.nn.Module):
        def forward(self, x_in):
            p = x_in[:, 0]
            q = x_in[:, 1]
            return ((p + torch.log(q)) / (1.0 + p)).unsqueeze(1)

    teacher = Teacher().double()
    y = teacher(torch.stack((x0 * x1, x2 * x3), dim=1))
    loader = DataLoader(TensorDataset(x_full, y), batch_size=n, shuffle=False)

    result = _build_nonlinear_sub_candidate(
        root=target,
        target=target,
        reuse={"leaf0": teacher},
        train_loader=loader,
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit={
            "col_idx": 1,
            "transform": "log",
            "deg_num": 1,
            "deg_den": 1,
            "outer_transform": "identity",
        },
    )

    assert result is not None
    cand_root, _init_fn, _meta = result
    rat_atom = _find_atom(cand_root, {"ratpoly", "rratpoly"})
    assert rat_atom is not None
    inputs = get_input_exprs(rat_atom)
    assert len(inputs) == 2
    assert ast_to_human_readable(inputs[0]) == ast_to_human_readable(get_input_exprs(target)[0])
    assert isinstance(inputs[1], LogNode)
    assert ast_to_human_readable(inputs[1].arg) == ast_to_human_readable(get_input_exprs(target)[1])


def test_affine_decomp_uses_effective_compound_w_and_z_inputs():
    target = _two_compound_target()
    hit = {
        "g_name": "identity",
        "h_name": "identity",
        "col_w": 1,
        "a_values": [0.0, 1.0, 2.0],
        "b_values": [1.0, 1.0, 1.0],
        "z_centers": [0.5, 1.0, 1.5],
        "median_r2": 0.999,
    }

    result = _build_affine_decomp_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit=hit,
    )

    assert result is not None
    cand_root, _init_fn, _meta = result
    assert isinstance(cand_root, AddNode)
    left_atom = cand_root.left
    assert isinstance(left_atom, AtomNode)
    assert get_input_exprs(left_atom)
    assert ast_to_human_readable(get_input_exprs(left_atom)[0]) == ast_to_human_readable(get_input_exprs(target)[0])
    assert isinstance(cand_root.right, MulNode)
    assert ast_to_human_readable(cand_root.right.right) == ast_to_human_readable(get_input_exprs(target)[1])


def test_affine_decomp_allows_reversed_effective_orientation():
    target = _two_compound_target()
    hit = {
        "g_name": "identity",
        "h_name": "identity",
        "col_w": 0,
        "a_values": [0.0, 1.0, 2.0],
        "b_values": [1.0, 1.0, 1.0],
        "z_centers": [0.5, 1.0, 1.5],
        "median_r2": 0.999,
    }

    result = _build_affine_decomp_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit=hit,
    )

    assert result is not None
    cand_root, _init_fn, _meta = result
    left_atom = cand_root.left
    assert isinstance(left_atom, AtomNode)
    assert ast_to_human_readable(get_input_exprs(left_atom)[0]) == ast_to_human_readable(get_input_exprs(target)[1])
    assert ast_to_human_readable(cand_root.right.right) == ast_to_human_readable(get_input_exprs(target)[0])


def test_affine_decomp_builds_visible_one_minus_cos_coordinate():
    target = _two_compound_target()
    hit = {
        "g_name": "identity",
        "h_name": "one_minus_cos",
        "omega": 2.0,
        "col_w": 1,
        "a_values": [0.0, 1.0, 2.0],
        "b_values": [1.0, 1.0, 1.0],
        "z_centers": [0.5, 1.0, 1.5],
        "median_r2": 0.999,
    }

    result = _build_affine_decomp_candidate(
        root=target,
        target=target,
        reuse={"leaf0": object()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit=hit,
    )

    assert result is not None
    cand_root, _init_fn, _meta = result
    rendered = ast_to_human_readable(cand_root)
    assert "cos" in rendered
    assert "-1" in rendered or " - " in rendered or "+ (-" in rendered
