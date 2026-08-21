# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Tests for sinusoidal and exponential mapping families in factorized symbolic search.

Run:  python nestynet_sr/sr_search/factorized_search/smoke_bridge.py
"""
import sys

import torch

from nestynet_sr.sr_core.bridges import (
    Var, eval_input_expr,
    AddNode, MulNode, ConstNode, SinNode, AtomNode, Scale,
)
from nestynet_sr.sr_search.factorized_search.bridge import (
    factorized_search_to_nestynet,
    embed_mapping_in_ast,
    promote_const_to_scale,
)
from nestynet_sr.sr_search.factorized_search.explorer import (
    eval_exp_mapping,
    eval_mapping,
    eval_sine,
    fit_best,
    fit_exp_mapping,
    fit_sine,
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


# ── Test: fit_sine ──────────────────────────────────────────────────

print("\n=== Test: fit_sine ===")
torch.manual_seed(42)
f = torch.linspace(-3, 3, 500, dtype=torch.float64).unsqueeze(-1)
# y = 3·sin(2·f + 0.5) + 1  =  3·cos(0.5)·sin(2f) + 3·sin(0.5)·cos(2f) + 1
y_true = 3.0 * torch.sin(2.0 * f + 0.5) + 1.0
m = fit_sine(f, y_true)
check("fit_sine returns dict", m is not None)
if m is not None:
    y_hat = eval_sine(f, m)
    mse = float(((y_true - y_hat) ** 2).mean())
    check("fit_sine MSE < 1e-10", mse < 1e-10, f"mse={mse:.3g}")
    check("fit_sine kind", m["kind"] == "sine")

# ── Test: eval_sine dispatch via eval_mapping ──────────────────────

print("\n=== Test: eval_sine via eval_mapping ===")
if m is not None:
    y_hat2 = eval_mapping(f, m)
    diff = float(((y_hat - y_hat2) ** 2).mean())
    check("eval_mapping('sine') matches eval_sine", diff < 1e-30, f"diff={diff:.3g}")

# ── Test: fit_exp_mapping ──────────────────────────────────────────

print("\n=== Test: fit_exp_mapping ===")
f_exp = torch.linspace(0, 3, 500, dtype=torch.float64).unsqueeze(-1)
y_exp_true = 2.0 * torch.exp(-f_exp) + 0.5
m_exp = fit_exp_mapping(f_exp, y_exp_true)
check("fit_exp_mapping returns dict", m_exp is not None)
if m_exp is not None:
    y_hat_exp = eval_exp_mapping(f_exp, m_exp)
    mse_exp = float(((y_exp_true - y_hat_exp) ** 2).mean())
    check("fit_exp_mapping MSE < 1e-10", mse_exp < 1e-10, f"mse={mse_exp:.3g}")
    check("fit_exp_mapping kind", m_exp["kind"] == "exp")

# ── Test: eval_exp_mapping dispatch via eval_mapping ───────────────

print("\n=== Test: eval_exp via eval_mapping ===")
if m_exp is not None:
    y_hat_exp2 = eval_mapping(f_exp, m_exp)
    diff_exp = float(((y_hat_exp - y_hat_exp2) ** 2).mean())
    check("eval_mapping('exp') matches eval_exp_mapping", diff_exp < 1e-30, f"diff={diff_exp:.3g}")

# ── Test: fit_best picks sine on sine data ─────────────────────────

print("\n=== Test: fit_best picks sine on sine data ===")
fb_sine = fit_best(f, y_true, poly_degree=4)
check("fit_best returns result on sine data", fb_sine is not None)
if fb_sine is not None:
    mse_fb, map_fb = fb_sine
    check("fit_best picks 'sine' kind", map_fb["kind"] == "sine", f"got kind={map_fb['kind']}")
    check("fit_best sine MSE < 1e-10", mse_fb < 1e-10, f"mse={mse_fb:.3g}")

# ── Test: fit_best picks exp on exp data ───────────────────────────

print("\n=== Test: fit_best picks exp on exp data ===")
fb_exp = fit_best(f_exp, y_exp_true, poly_degree=4)
check("fit_best returns result on exp data", fb_exp is not None)
if fb_exp is not None:
    mse_fb_exp, map_fb_exp = fb_exp
    check("fit_best picks 'exp' kind", map_fb_exp["kind"] == "exp", f"got kind={map_fb_exp['kind']}")
    check("fit_best exp MSE < 1e-10", mse_fb_exp < 1e-10, f"mse={mse_fb_exp:.3g}")

# ── Test: embed_mapping_in_ast round-trip (sine) ───────────────────

print("\n=== Test: embed_mapping_in_ast (sine) ===")
if m is not None:
    # Skeleton: just x0
    toy_ast = ("var", 0)
    nn_skeleton = factorized_search_to_nestynet(toy_ast)
    input_exprs = [Var(0)]
    full_ast = embed_mapping_in_ast(nn_skeleton, m, input_exprs)
    check("embed sine returns Node", full_ast is not None)
    if full_ast is not None:
        x_test = f.clone()
        y_ast = eval_input_expr(full_ast, x_test)
        mse_rt = float(((y_true - y_ast) ** 2).mean())
        check("embed sine round-trip MSE < 1e-10", mse_rt < 1e-10, f"mse={mse_rt:.3g}")

# ── Test: embed_mapping_in_ast round-trip (exp) ────────────────────

print("\n=== Test: embed_mapping_in_ast (exp) ===")
if m_exp is not None:
    toy_ast = ("var", 0)
    nn_skeleton = factorized_search_to_nestynet(toy_ast)
    input_exprs = [Var(0)]
    full_ast = embed_mapping_in_ast(nn_skeleton, m_exp, input_exprs)
    check("embed exp returns Node", full_ast is not None)
    if full_ast is not None:
        x_test = f_exp.clone()
        y_ast = eval_input_expr(full_ast, x_test)
        mse_rt = float(((y_exp_true - y_ast) ** 2).mean())
        check("embed exp round-trip MSE < 1e-10", mse_rt < 1e-10, f"mse={mse_rt:.3g}")

# ── Test: promote_const_to_scale on existing NestyNet AST ─────────

def _is_scale(node):
    return isinstance(node, AtomNode) and node.kind == "scale"

print("\n=== Test: promote_const_to_scale ===")

# Build a NestyNet AST with ConstNodes (as if from sympy_to_nestynet)
ast_with_consts = AddNode(
    MulNode(ConstNode(0.5), Var(0)),
    MulNode(ConstNode(3.0), SinNode(Var(0))),
)
promoted = promote_const_to_scale(ast_with_consts, tag_prefix="test")

check("promote: root is AddNode", isinstance(promoted, AddNode))
if isinstance(promoted, AddNode):
    lm = promoted.left
    rm = promoted.right
    if isinstance(lm, MulNode):
        check("promote: left coeff is Scale", _is_scale(lm.left),
              f"got {type(lm.left).__name__}")
        if _is_scale(lm.left):
            check("promote: left Scale init=0.5",
                  abs(lm.left.kwargs["init"] - 0.5) < 1e-12)
    if isinstance(rm, MulNode):
        check("promote: right coeff is Scale", _is_scale(rm.left),
              f"got {type(rm.left).__name__}")
        if _is_scale(rm.left):
            check("promote: right Scale init=3.0",
                  abs(rm.left.kwargs["init"] - 3.0) < 1e-12)

# Existing Scale atoms should pass through unchanged
ast_with_scale = MulNode(Scale("s", tag="s", init=2.0), Var(0))
promoted2 = promote_const_to_scale(ast_with_scale)
check("promote: existing Scale passes through",
      _is_scale(promoted2.left) and promoted2.left.kwargs["init"] == 2.0)


# ── Summary ────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Results: {n_pass} passed, {n_fail} failed")
if n_fail > 0:
    sys.exit(1)
print("All tests passed!")
