# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Focused regression checks for tangent/soft edit maximal extensions.

Run:
  PYTHONPATH=. python nestynet_sr/sr_search/factorized_search/smoke_tangent_edit_extensions.py
"""
import sys

import torch

from nestynet_sr.sr_search.factorized_search.subproblem_spec import SubproblemSpec, WitnessBundle, serialize_subproblem_spec
from nestynet_sr.sr_search.factorized_search.tangent_edit import (
    _capture_node_value_grad,
    _enumerate_tangent_edit_nodes,
    solve_local_tangent_edit_preview_rows,
)


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


def _solve_rows(base_node, x_fit, t_fit, g_fit, *, max_depth=8, preview_topk=6):
    witness = WitnessBundle(
        x_fit=x_fit,
        t_fit=t_fit,
        x_probe=x_fit,
        t_probe=t_fit,
        grad_fit=g_fit,
        grad_probe=g_fit,
    )
    spec = SubproblemSpec(
        problem_id="test",
        problem_kind="local_problem",
        parent_expr=base_node,
        path=(),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="identity",
        target_dim=None,
        active_vars=(0,),
        witness=witness,
        metadata={"hole_sub": base_node},
    )
    return solve_local_tangent_edit_preview_rows(
        parent_node=base_node,
        spec_payload=serialize_subproblem_spec(spec),
        path=(),
        target_mode="identity",
        target_mapping_kind="identity",
        beam_rank=0,
        slate_id="unit",
        path_gain=0.0,
        max_depth=max_depth,
        nvars=1,
        poly_degree=3,
        var_dims=None,
        pool_nodes=None,
        pool_dims=None,
        preview_topk=preview_topk,
    )


print("\n=== Test: scale/shift promotion and sparse combo families ===")
x = torch.linspace(-1.0, 1.0, 33, dtype=torch.float64).unsqueeze(-1)
t = 2.0 + (3.0 * x)
g = torch.full_like(x, 3.0)
base = ("var", 0)
base_value, base_grad = _capture_node_value_grad(base, x, capture_gradients=True)
cands = _enumerate_tangent_edit_nodes(
    base,
    target_dim=None,
    nvars=1,
    active_vars=(0,),
    pool_nodes=None,
    var_dims=None,
    x_rank=x,
    t_rank=t,
    target_grad_rank=g,
    base_value_fit=base_value,
    base_grad_fit=base_grad,
)
edit_kinds = {str(row.get("edit_kind", "")) for row in cands}
check(
    "promote route emitted",
    any(kind.startswith("promote:") for kind in edit_kinds),
    f"kinds={[k for k in sorted(edit_kinds) if k.startswith('promote:')][:4]}",
)
check(
    "sparse replace route emitted",
    any(kind.startswith("replace:sparse_combo") for kind in edit_kinds),
    f"kinds={[k for k in sorted(edit_kinds) if k.startswith('replace:sparse_combo')][:4]}",
)
rows = _solve_rows(base, x, t, g)
best = rows["rows"][0] if rows["rows"] else {}
check(
    "scale/shift exact local recovery",
    bool(rows["rows"]) and float(best.get("local_probe_mse", 1.0)) < 1.0e-10,
    f"best_kind={best.get('tangent_edit_kind', None)} mse={best.get('local_probe_mse', None)}",
)


print("\n=== Test: explicit integer-power steps beyond sqr ===")
x = torch.linspace(0.2, 1.2, 41, dtype=torch.float64).unsqueeze(-1)
t = x ** 3
g = 3.0 * (x ** 2)
base = ("var", 0)
base_value, base_grad = _capture_node_value_grad(base, x, capture_gradients=True)
cands = _enumerate_tangent_edit_nodes(
    base,
    target_dim=None,
    nvars=1,
    active_vars=(0,),
    pool_nodes=None,
    var_dims=None,
    x_rank=x,
    t_rank=t,
    target_grad_rank=g,
    base_value_fit=base_value,
    base_grad_fit=base_grad,
)
power_kinds = {str(row.get("edit_kind", "")) for row in cands if str(row.get("edit_kind", "")).startswith("power:")}
check("power:+3 emitted", "power:+3" in power_kinds, f"power_kinds={sorted(power_kinds)}")
rows = _solve_rows(base, x, t, g)
best_power = rows["rows"][0] if rows["rows"] else {}
check(
    "cubic exact local recovery",
    bool(rows["rows"]) and str(best_power.get("tangent_edit_kind", "")) == "power:+3" and float(best_power.get("local_probe_mse", 1.0)) < 1.0e-12,
    f"best_kind={best_power.get('tangent_edit_kind', None)} mse={best_power.get('local_probe_mse', None)}",
)


print("\n=== Test: explicit constant untie/promote operators ===")
x = torch.linspace(0.2, 1.2, 41, dtype=torch.float64).unsqueeze(-1)
t = torch.sin(3.0 * x)
g = 3.0 * torch.cos(3.0 * x)
base = ("sin", ("mul", ("const", 2.0), ("var", 0)))
base_value, base_grad = _capture_node_value_grad(base, x, capture_gradients=True)
cands = _enumerate_tangent_edit_nodes(
    base,
    target_dim=None,
    nvars=1,
    active_vars=(0,),
    pool_nodes=None,
    var_dims=None,
    x_rank=x,
    t_rank=t,
    target_grad_rank=g,
    base_value_fit=base_value,
    base_grad_fit=base_grad,
)
untie_kinds = {str(row.get("edit_kind", "")) for row in cands if str(row.get("edit_kind", "")).startswith("untie_const")}
check("untie leaf emitted", "untie_const:leaf" in untie_kinds, f"untie_kinds={sorted(untie_kinds)}")
rows = _solve_rows(base, x, t, g)
best_untie = None
for row in rows["rows"]:
    if str(row.get("tangent_edit_kind", "")).startswith("untie_const"):
        best_untie = row
        break
check(
    "untie route survives preview truncation",
    best_untie is not None,
    f"kinds={[row.get('tangent_edit_kind', None) for row in rows.get('rows', [])]}",
)
check(
    "untie route materially improves local fit",
    best_untie is not None and float(best_untie.get("local_probe_mse", 1.0)) < 1.0e-2,
    f"mse={None if best_untie is None else best_untie.get('local_probe_mse', None)}",
)


print(f"\n{'='*50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    sys.exit(1)
print("All tests passed!")
