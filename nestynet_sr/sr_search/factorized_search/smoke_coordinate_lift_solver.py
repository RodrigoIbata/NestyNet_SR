# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Focused regression checks for coordinate-lift candidate generation.

Run:
  python nestynet_sr/sr_search/factorized_search/smoke_coordinate_lift_solver.py
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from nestynet_sr.sr_search.factorized_search.coordinate_lift_solver import (
    _build_coordinate_candidates,
    _select_diverse_coordinate_candidates,
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


def _make_problem(*, nvars: int):
    g_fit = torch.Generator().manual_seed(0)
    g_probe = torch.Generator().manual_seed(1)
    xf = 0.5 + torch.rand((64, int(nvars)), generator=g_fit, dtype=torch.float64)
    xp = 0.5 + torch.rand((96, int(nvars)), generator=g_probe, dtype=torch.float64)
    return SimpleNamespace(
        xf=xf,
        tf=(xf[:, :1] * xf[:, 1:2]) / xf[:, 2:3] if int(nvars) >= 3 else xf[:, :1],
        wf=None,
        xp=xp,
        tp=(xp[:, :1] * xp[:, 1:2]) / xp[:, 2:3] if int(nvars) >= 3 else xp[:, :1],
        wp=None,
        target_dim=(0.0, 0.0),
        confidence=1.0,
        valid_frac=1.0,
        wrappers_left=1,
        recursion_level=0,
        trace=(),
        grad_fit=None,
        grad_probe=None,
        d2_fit=None,
        d2_probe=None,
        diagnostics={},
    )


print("\n=== Test: dimensionless-group candidate generation ===")
problem = _make_problem(nvars=3)
proposals, diagnostics = _build_coordinate_candidates(
    problem=problem,
    active_vars=(0, 1, 2),
    coordinate_mode="both",
    subproblem_spec=None,
    lift_route_context={},
    var_dims=[(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
)
expected_dimless = ("div", ("mul", ("var", 0), ("var", 1)), ("var", 2))
expected_dimless_alt = ("div", ("var", 2), ("mul", ("var", 0), ("var", 1)))
kind_counts = dict(diagnostics.get("candidate_kind_counts", {}) or {})
check(
    "dimensionless group family emitted",
    int(kind_counts.get("dimensionless_group", 0) or 0) > 0,
    f"kinds={kind_counts}",
)
check(
    "buckingham-pi seed includes x0*x1/x2",
    any(row.get("node") == expected_dimless or row.get("node") == expected_dimless_alt for row in proposals),
)


print("\n=== Test: same-dimension radial invariant generation ===")
problem2 = _make_problem(nvars=3)
proposals2, diagnostics2 = _build_coordinate_candidates(
    problem=problem2,
    active_vars=(0, 1, 2),
    coordinate_mode="both",
    subproblem_spec=None,
    lift_route_context={},
    var_dims=[(1.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
)
radial_sq = ("add", ("sqr", ("var", 0)), ("sqr", ("var", 1)))
radial_norm = ("sqrt", radial_sq)
check(
    "radial square seed emitted",
    any(row.get("node") == radial_sq for row in proposals2),
)
check(
    "radial norm seed emitted",
    any(row.get("node") == radial_norm for row in proposals2),
)


print("\n=== Test: top-k selection keeps invariant diversity ===")
selected = _select_diverse_coordinate_candidates(proposals2, topk=6)
families = {str(row.get("candidate_family", "")) for row in selected}
check(
    "diversified selection keeps radial family",
    "radial" in families,
    f"families={sorted(families)}",
)
check(
    "diversified selection keeps raw fallback family",
    "raw_var" in families,
    f"families={sorted(families)}",
)


print(f"\nSummary: pass={n_pass} fail={n_fail}")
if n_fail:
    raise SystemExit(1)
