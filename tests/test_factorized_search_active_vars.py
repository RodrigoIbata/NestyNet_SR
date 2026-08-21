# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_search.factorized_search.subproblem_active_vars import (
    infer_subproblem_active_vars,
    remap_local_node_vars,
)


def test_infer_subproblem_active_vars_uses_gradient_when_only_anchor_structure_exists():
    grad_fit = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    active_vars, diagnostics = infer_subproblem_active_vars(
        hole_sub=("const", 1.0),
        continuation_frames=[{"wrap_kind": "binary", "op": "add", "slot": 1, "anchor_node": ("var", 2)}],
        grad_fit=grad_fit,
        grad_probe=None,
        nvars=3,
        screen_enable=True,
        grad_tol=1.0e-3,
        max_count=4,
    )

    assert active_vars == (1,)
    assert diagnostics["active_var_hole"] == []
    assert diagnostics["active_var_anchor"] == [2]
    assert diagnostics["active_var_source"] == "gradient"


def test_infer_subproblem_active_vars_refines_hole_vars_with_gradient_screen():
    grad_fit = torch.tensor(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    active_vars, diagnostics = infer_subproblem_active_vars(
        hole_sub=("add", ("var", 0), ("var", 2)),
        continuation_frames=[{"wrap_kind": "binary", "op": "mul", "slot": 2, "anchor_node": ("var", 1)}],
        grad_fit=grad_fit,
        grad_probe=None,
        nvars=3,
        screen_enable=True,
        grad_tol=1.0e-3,
        max_count=4,
    )

    assert active_vars == (2,)
    assert diagnostics["active_var_hole"] == [0, 2]
    assert diagnostics["active_var_anchor"] == [1]
    assert diagnostics["active_var_source"] == "hole+gradient"


def test_remap_local_node_vars_maps_subset_back_to_global_indices():
    node = ("add", ("var", 0), ("mul", ("var", 1), ("const", 2.0)))
    remapped = remap_local_node_vars(node, active_vars=(1, 3))
    assert remapped == ("add", ("var", 1), ("mul", ("var", 3), ("const", 2.0)))
