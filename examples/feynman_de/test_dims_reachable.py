#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Regression test: verify every DIMS_REGISTRY problem's target dimension
is reachable by ``compute_reachable``, and that ground-truth term
dimensions land in the reachable set.

Run as:
    python examples/feynman_de/test_dims_reachable.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work standalone.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from nestynet_sr.sr_core.problem_dims import canonical_to_factorized_search_dims
from nestynet_sr.sr_search.factorized_search.explorer import compute_reachable, dim_round
from examples.feynman_de.problem_defs import (
    DIMS_REGISTRY,
    GROUND_TRUTH,
    get_canonical_problem_dims,
)

# ---------------------------------------------------------------------------
# Helpers: scalar target/feature dims should flow through the shared adapter.
# ---------------------------------------------------------------------------

def _dim_sub(d1, d2):
    return tuple(float(a) - float(b) for a, b in zip(d1, d2))


def _dim_add(d1, d2):
    return tuple(float(a) + float(b) for a, b in zip(d1, d2))


def _dim_scale(d, s):
    return tuple(float(x) * s for x in d)


def build_var_dims_and_y_dims(pid: str, order: int):
    """Build var_dims and y_dims via the shared canonical dim adapter."""
    canonical_dims = get_canonical_problem_dims(str(pid))
    if canonical_dims is None:
        raise ValueError(f"{pid}: missing canonical dims")
    return canonical_to_factorized_search_dims(
        canonical_dims,
        order=int(order),
        x_axis=0,
        component_idx=0,
        include_x=True,
        include_u=True,
        include_du=True,
        constant_names=tuple(canonical_dims.constant_dims.keys()),
    )


# ---------------------------------------------------------------------------
# Term-key dimension parser
# ---------------------------------------------------------------------------

def _term_dim(key: str, x_dim, u_dim, du_dim):
    """Parse a GROUND_TRUTH term key and return its dimension vector.

    Handles: "1", "u", "u_x0", "x0", "(u ** N)", "(x0 * u)",
    "((x0 ** N) * u)", "((x0 ** N) * u_x0)".
    """
    ndim = len(x_dim)
    dim0 = (0.0,) * ndim

    key = key.strip()

    if key == "1":
        return dim0
    if key == "u":
        return u_dim
    if key == "u_x0":
        return du_dim
    if key == "x0":
        return x_dim

    # (u ** N)
    if key.startswith("(u ** ") and key.endswith(")"):
        exp = float(key[len("(u ** "):-1])
        return _dim_scale(u_dim, exp)

    # (x0 * u)
    if key == "(x0 * u)":
        return _dim_add(x_dim, u_dim)

    # ((x0 ** N) * u)
    if key.startswith("((x0 ** ") and key.endswith(") * u)"):
        inner = key[len("((x0 ** "):]
        exp = float(inner.split(")")[0])
        return _dim_add(_dim_scale(x_dim, exp), u_dim)

    # ((x0 ** N) * u_x0)
    if key.startswith("((x0 ** ") and key.endswith(") * u_x0)"):
        inner = key[len("((x0 ** "):]
        exp = float(inner.split(")")[0])
        return _dim_add(_dim_scale(x_dim, exp), du_dim)

    raise ValueError(f"Cannot parse term key: {key!r}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

MAX_DEPTH = 5


def test_dims_reachable():
    common_ids = sorted(set(DIMS_REGISTRY) & set(GROUND_TRUTH))
    if not common_ids:
        print("WARNING: no problems in both DIMS_REGISTRY and GROUND_TRUTH")
        return

    failures = []

    for pid in common_ids:
        gt = GROUND_TRUTH[pid]
        order = gt.order

        var_dims, y_dims = build_var_dims_and_y_dims(pid, order)
        y_rounded = dim_round(y_dims)

        reach = compute_reachable(var_dims, max_depth=MAX_DEPTH, target_dim=y_dims)

        # Check 1: y_dims is reachable at some depth <= MAX_DEPTH
        y_reachable = any(y_rounded in reach[d] for d in range(1, MAX_DEPTH + 1))
        if not y_reachable:
            failures.append(f"  {pid}: target dim {y_rounded} NOT reachable at depth <= {MAX_DEPTH}")

        # Check 2: each ground-truth term dimension is reachable
        dims = DIMS_REGISTRY[pid]
        x_dim = tuple(float(v) for v in dims.x_dim)
        u_dim = tuple(float(v) for v in dims.u_dim)
        du_dim = _dim_sub(u_dim, x_dim)

        for term_key in gt.terms:
            try:
                td = _term_dim(term_key, x_dim, u_dim, du_dim)
            except ValueError:
                # Skip unparseable exotic terms
                continue
            td_rounded = dim_round(td)
            td_reachable = any(td_rounded in reach[d] for d in range(1, MAX_DEPTH + 1))
            if not td_reachable:
                failures.append(
                    f"  {pid}: term {term_key!r} dim {td_rounded} NOT reachable"
                )

    # Report
    n_tested = len(common_ids)
    print(f"Tested {n_tested} problems: {', '.join(common_ids)}")
    if failures:
        print(f"\nFAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f)
        sys.exit(1)
    else:
        print("ALL PASSED")


if __name__ == "__main__":
    test_dims_reachable()
