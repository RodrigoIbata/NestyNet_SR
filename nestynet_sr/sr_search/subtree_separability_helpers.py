# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch

from nestynet_sr.sr_core import check_separability

OpType = Callable


def run_subtree_separability(
    model_u: torch.nn.Module,
    datagen,
    var_indices: List[int],
    device: torch.device,
    dtype: torch.dtype,
    allow_partial: bool = False,
    very_verbose: bool = False,
) -> Optional[Tuple[OpType, List[int], List[int]]]:
    """
    Stage-A-style separability analysis on a subtree u(x_{var_indices}).

    Parameters
    ----------
    model_u : torch.nn.Module
        Model representing the subtree u(x). It is expected to behave like a
        Stage-A "teacher": it implements forward(x) -> (B, 1) and provides
        analytic input derivatives via grad(...) and grad_grad(...) (or
        value_grad_grad(...)).
        In practice we pass it the full x from datagen, but it should only
        depend on the columns listed in var_indices.
    datagen : iterable
        Dataloader yielding (x, y) batches; only x is used here.
    var_indices : list[int]
        Global variable indices that the subtree actually depends on.
    device, dtype : torch.device, torch.dtype
        Used for any temporary tensors or buffers.
    allow_partial : bool
        If True, allow overlapping variable groups (partial separability).
        This is useful for expressions like sqrt(f(x0,x1) + g(x1,x2,x3)) where
        x1 appears in both terms. Default False for backward compatibility.

    Returns
    -------
    (op, group1_local, group2_local) or None
        - op is torch.add or torch.multiply
        - group1_local, group2_local are lists of *local* indices in
          {0, ..., len(var_indices)-1}. If allow_partial=False, these are
          non-overlapping. If allow_partial=True, they may share variables.
        Return None if no reliable separability is detected.
    """
    print(f"[SubtreeSeparability] called on {type(model_u).__name__}, vars={var_indices}")
    if len(var_indices) < 2:
        return None

    has_grad = callable(getattr(model_u, "grad", None))
    has_second = callable(getattr(model_u, "grad_grad", None)) or callable(
        getattr(model_u, "value_grad_grad", None)
    )
    if not (has_grad and has_second):
        return None

    precision = 1e-3
    symb = [int(j) for j in var_indices]

    try:
        cand_list, _rest_add, _rest_mult, _, _ = check_separability(
            symb,
            0,
            model_u,
            datagen,
            precision_sum=precision,
            precision_mult=precision,
            device=device,
            very_verbose=very_verbose,
        )
    except Exception:
        return None

    if not cand_list:
        return None

    s_vars = set(var_indices)
    idx_map = {v: k for k, v in enumerate(var_indices)}

    for cand in cand_list:
        if not cand:
            continue
        op = cand[0]
        if op not in (torch.add, torch.multiply):
            continue

        g1_global = cand[1] if len(cand) > 1 else []
        g2_global = cand[2] if len(cand) > 2 else []
        if not g1_global or not g2_global:
            continue

        if not set(g1_global).issubset(s_vars):
            continue
        if not set(g2_global).issubset(s_vars):
            continue
        # Skip overlapping groups unless allow_partial is True AND this is an additive split
        # (multiplicative splits with overlap are not meaningful rewrites)
        has_overlap = bool(set(g1_global) & set(g2_global))
        if has_overlap and not (allow_partial and op == torch.add):
            continue

        g1_local = [idx_map[j] for j in g1_global]
        g2_local = [idx_map[j] for j in g2_global]
        if g1_local and g2_local:
            return op, g1_local, g2_local

    return None
