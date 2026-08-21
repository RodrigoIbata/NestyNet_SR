# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Regression checks for typed quadratic seed generation.

Run:
  python nestynet_sr/sr_search/factorized_search/smoke_quadratic_seed_filters.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestynet_sr.sr_search.factorized_search.expr_ast import node_str, simplify
from nestynet_sr.sr_search.factorized_search.proposal_families.seed_blocks import (
    build_recursive_seed_pool,
    generate_quadratic_seed_blocks,
    make_seed_block,
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


def _dims(*vals):
    return tuple(float(v) for v in vals)


print("\n=== Test: target-aware quadratic seeds stay in the right dim bucket ===")
qdim = _dims(1.0, 0.0)
edim = _dims(0.0, 1.0)
var_dims = [qdim, edim, edim, edim]
seed_blocks = [
    make_seed_block(("const", 1.0), dim=_dims(0.0, 0.0), source="const"),
    make_seed_block(("var", 0), dim=qdim, source="var"),
    make_seed_block(("var", 1), dim=edim, source="var"),
    make_seed_block(("var", 2), dim=edim, source="var"),
    make_seed_block(("var", 3), dim=edim, source="var"),
    make_seed_block(("mul", ("var", 0), ("var", 1)), dim=_dims(1.0, 1.0), source="pool", builder="product"),
    make_seed_block(("mul", ("var", 0), ("var", 2)), dim=_dims(1.0, 1.0), source="pool", builder="product"),
    make_seed_block(("mul", ("var", 0), ("var", 3)), dim=_dims(1.0, 1.0), source="pool", builder="product"),
]
expected = simplify((
    "add",
    simplify(("add", ("sqr", ("var", 1)), ("sqr", ("var", 2)))),
    ("sqr", ("var", 3)),
))
rows = generate_quadratic_seed_blocks(
    seed_blocks,
    limit=1,
    max_arity=3,
    max_builder_depth=3,
    max_nonlinear_depth=2,
    var_dims=var_dims,
    required_expr_dims=[_dims(0.0, 2.0)],
)
check("one quadratic emitted", len(rows) == 1, f"n={len(rows)}")
if rows:
    row = rows[0]
    base_nodes = tuple(dict(row.metadata or {}).get("base_nodes", ()) or ())
    check("descending arity reaches 3-term norm first", len(base_nodes) == 3, f"bases={[node_str(v) for v in base_nodes]}")
    check("typed filter finds x1,x2,x3 norm", row.node == expected, f"expr={node_str(row.node)}")
    check("wrong-dim anchor variable excluded", all(node != ("var", 0) for node in base_nodes), f"bases={[node_str(v) for v in base_nodes]}")
    check("quadratic dim recorded eagerly", tuple(row.dim or ()) == _dims(0.0, 2.0), f"dim={row.dim}")
    check(
        "base dims preserved in metadata",
        tuple(dict(row.metadata or {}).get("base_dims", ())) == (edim, edim, edim),
        f"base_dims={dict(row.metadata or {}).get('base_dims', None)}",
    )


print("\n=== Test: build_recursive_seed_pool threads target-aware dims into quadratic generation ===")
recursive = build_recursive_seed_pool(
    seed_blocks,
    rounds=1,
    include_product=False,
    include_monomial=False,
    include_quadratic=True,
    include_affine=False,
    quadratic_max_arity=3,
    quadratic_limit=1,
    quadratic_required_dims=[_dims(0.0, 2.0)],
    max_builder_depth=3,
    max_nonlinear_depth=2,
    var_dims=var_dims,
)
quad_rows = [block for block in recursive if str(getattr(block, "builder", "")) == "quadratic"]
check("recursive pool emits one typed quadratic", len(quad_rows) == 1, f"n={len(quad_rows)}")
if quad_rows:
    check("recursive quadratic matches target norm", quad_rows[0].node == expected, f"expr={node_str(quad_rows[0].node)}")

print(f"\n{'='*50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    sys.exit(1)
print("All tests passed!")
