# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Focused regression checks for continuous skeleton refinement fixes.

Run:
  python nestynet_sr/sr_search/factorized_search/smoke_refine_fixes.py
"""
import sys

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    LogNode,
    MulNode,
    SinNode,
    Var,
)
from nestynet_sr.sr_search.factorized_search.bridge import promote_argument_const_scales
from nestynet_sr.sr_search.factorized_search import explorer


n_pass = 0
n_fail = 0


def check(name, ok, detail=""):
    global n_pass, n_fail
    if ok:
        n_pass += 1
        print(f"  PASS  {name}  {detail}")
    else:
        n_fail += 1
        print(f"  FAIL  {name}  {detail}")


def _collect_scale_nodes(node):
    out = []

    def _walk(n):
        if isinstance(n, AtomNode):
            if str(getattr(n, "kind", "")) == "scale":
                out.append(n)
            return
        if isinstance(n, AddNode):
            _walk(n.left)
            _walk(n.right)
            return
        if isinstance(n, MulNode):
            _walk(n.left)
            _walk(n.right)
            return
        if hasattr(n, "arg"):
            _walk(n.arg)
            return
        if hasattr(n, "base"):
            _walk(n.base)

    _walk(node)
    return out


print("\n=== Test: _build_init_logs stays in range ===")
logs = explorer._build_init_logs(
    n_params=3,
    restarts=2,
    log_min=-1.5,
    log_max=1.5,
    dtype=torch.float64,
    device=torch.device("cpu"),
)
check("shape", tuple(logs.shape) == (2, 3), f"shape={tuple(logs.shape)}")
check("row0 is zero", bool(torch.allclose(logs[0], torch.zeros_like(logs[0]))))
check(
    "within range",
    bool((logs >= -1.5 - 1e-12).all() and (logs <= 1.5 + 1e-12).all()),
    f"min={float(logs.min()):.3g} max={float(logs.max()):.3g}",
)
logs_pos = explorer._build_init_logs(
    n_params=2,
    restarts=3,
    log_min=0.2,
    log_max=1.0,
    dtype=torch.float64,
    device=torch.device("cpu"),
)
check(
    "neutral seed clamped to range",
    abs(float(logs_pos[0, 0]) - 0.2) < 1e-12 and abs(float(logs_pos[0, 1]) - 0.2) < 1e-12,
    f"row0={logs_pos[0].tolist()}",
)


print("\n=== Test: node_str handles hparam ===")
s0 = explorer.node_str(("sin", ("mul", ("hparam", 0), ("var", 0))))
s1 = explorer.node_str(("sin", ("mul", ("hparam", 1), ("var", 0))))
check("hparam token is explicit", "hp0" in s0, f"s0={s0}")
check("different hparams stringify differently", s0 != s1, f"s0={s0}, s1={s1}")


print("\n=== Test: promote_argument_const_scales ===")
root = AddNode(
    SinNode(MulNode(ConstNode(2.5), Var(0))),
    LogNode(MulNode(Var(0), ConstNode(0.7))),
)
promoted = promote_argument_const_scales(root, tag_prefix="t")
scales = _collect_scale_nodes(promoted)
check("two argument scales promoted", len(scales) == 2, f"n_scales={len(scales)}")
if len(scales) == 2:
    vals = sorted(float(s.kwargs.get("init", 0.0)) for s in scales)
    check("scale init values preserved", vals == [0.7, 2.5], f"vals={vals}")
    names = [str(s.kwargs.get("name", "")) for s in scales]
    check("scale names are unique", len(set(names)) == len(names), f"names={names}")

outer_mul = MulNode(ConstNode(3.0), SinNode(Var(0)))
outer_promoted = promote_argument_const_scales(outer_mul, tag_prefix="t2")
outer_scales = _collect_scale_nodes(outer_promoted)
check("does not promote non-argument multipliers", len(outer_scales) == 0)


print("\n=== Test: complexity penalty uses structural expr size ===")
orig_score_expr = explorer.score_expr
mapping = {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0}


def _fake_score_expr(expr, *args, **kwargs):
    scored_expr = (
        "add",
        ("add", ("var", 0), ("var", 0)),
        ("add", ("var", 0), ("var", 0)),
    )
    z = torch.zeros((8,), dtype=torch.float64)
    return 1.0, 123, z, mapping, scored_expr


try:
    explorer.score_expr = _fake_score_expr
    arch = explorer.run_explorer_core(
        target_fn=lambda x: x[:, 0:1],
        nvars=1,
        n_iter=1,
        max_depth=1,
        n_fit=8,
        n_probe=8,
        brute_depth=0,
        complexity_penalty=1.0,
        print_every=0,
        dtype=torch.float64,
        seed=0,
    )
finally:
    explorer.score_expr = orig_score_expr

best = arch.best(1)[0].best_mse
expected = 1.0 + explorer.node_size(("var", 0)) + explorer.mapping_cost(mapping)
check(
    "penalty uses expr size plus mapping cost (not scored_expr size)",
    abs(best - expected) < 1e-12,
    f"best={best:.6g} expected={expected:.6g}",
)


print(f"\n{'='*50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    sys.exit(1)
print("All tests passed!")
