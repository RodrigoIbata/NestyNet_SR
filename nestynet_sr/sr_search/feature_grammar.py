# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Small feature grammar helpers for Stage A/B proposals.

This module centralises a few pieces of logic that were starting to drift across:
  - compound-function macro proposals (tier-1/2/3),
  - outer-algebra factor enumeration,
  - and ad-hoc constant/scale candidate lists.

The intent is *not* to define a full symbolic grammar (the AST already is), but
to provide a shared, cost-biased, deduplicated pool of low-order expressions
that multiple proposal mechanisms can reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AbsNode,
    Add,
    AddNode,
    AsinNode,
    ArgNode,
    AtanNode,
    AtomNode,
    ConjNode,
    ConstNode,
    CosNode,
    Div,
    ExpNode,
    ImagNode,
    LogNode,
    Mul,
    MulNode,
    Node,
    Pow,
    PowNode,
    RealNode,
    SinNode,
    Var,
    clone_ast,
    get_input_exprs,
    has_nontrivial_input,
    is_trivial_input,
)

# -----------------------------------------------------------------------------
# Shared constants / snapping
# -----------------------------------------------------------------------------


# A small list of constants that show up disproportionately often in the
# AI-Feynman benchmark and common physics expressions.
PHYSICS_SCALES: Tuple[float, ...] = (
    0.5,
    1.0 / 3.0,
    1.0 / 4.0,
    2.0,
    3.0,
    4.0,
    math.pi,
    2.0 * math.pi,
    4.0 * math.pi,
    8.0 * math.pi,
    1.0 / math.pi,
    1.0 / (2.0 * math.pi),
    1.0 / (4.0 * math.pi),
    1.0 / (8.0 * math.pi),
)


OMEGA_SNAP_CANDS: Tuple[float, ...] = (
    0.5,
    1.0,
    2.0,
    math.pi / 2.0,
    math.pi,
    2.0 * math.pi,
)


def snap_to_scales(x: float, scales: Sequence[float], *, rel_tol: float = 0.25, abs_tol: float = 0.25) -> float:
    """Snap a scalar x to the closest scale if it is clearly near."""
    try:
        v = float(x)
    except Exception:
        return x
    if not scales:
        return v
    best = min(scales, key=lambda c: abs(v - float(c)))
    if abs(v - best) <= max(abs_tol, rel_tol * abs(best)):
        return float(best)
    return float(v)


# -----------------------------------------------------------------------------
# Canonical structural key
# -----------------------------------------------------------------------------


def ast_key(node: Node) -> Tuple:
    """Hashable structural key for deduplication.

    Unlike sr_core.bridges.atom_content_hash, this key is intended for small
    *analytic* feature pools.

    It canonicalises commutative binary ops (Add/Mul) so that a*b and b*a share
    the same key, which reduces pool blow-up when combining features.
    """

    if isinstance(node, AtomNode):
        k = str(getattr(node, "kind", "")).lower()
        if k in ("var", "x", "input"):
            return ("Var", int(node.var_idxs[0]))
        if k in ("free_const", "freeconst", "free_constant"):
            nm = None
            try:
                nm = node.kwargs.get("name", None)
            except Exception:
                nm = None
            return ("FreeConst", str(nm), str(getattr(node, "tag", None)))
        return ("Atom", k, tuple(int(i) for i in getattr(node, "var_idxs", ()) or ()))

    if isinstance(node, ConstNode):
        v = node.value
        if isinstance(v, complex):
            return ("ConstC", float(v.real), float(v.imag))
        return ("Const", float(v))

    if isinstance(node, AddNode):
        a = ast_key(node.left)
        b = ast_key(node.right)
        if b < a:
            a, b = b, a
        return ("Add", a, b)

    if isinstance(node, MulNode):
        a = ast_key(node.left)
        b = ast_key(node.right)
        if b < a:
            a, b = b, a
        return ("Mul", a, b)

    if isinstance(node, PowNode):
        return ("Pow", ast_key(node.base), float(node.exponent))
    if isinstance(node, LogNode):
        return ("Log", ast_key(node.arg))
    if isinstance(node, ExpNode):
        return ("Exp", ast_key(node.arg))
    if isinstance(node, SinNode):
        return ("Sin", ast_key(node.arg))
    if isinstance(node, CosNode):
        return ("Cos", ast_key(node.arg))
    if isinstance(node, AsinNode):
        return ("Asin", ast_key(node.arg))
    if isinstance(node, AcosNode):
        return ("Acos", ast_key(node.arg))
    if isinstance(node, AtanNode):
        return ("Atan", ast_key(node.arg))

    # Complex unary ops
    if isinstance(node, ConjNode):
        return ("Conj", ast_key(node.arg))
    if isinstance(node, RealNode):
        return ("Real", ast_key(node.arg))
    if isinstance(node, ImagNode):
        return ("Imag", ast_key(node.arg))
    if isinstance(node, AbsNode):
        return ("Abs", ast_key(node.arg))
    if isinstance(node, ArgNode):
        return ("Arg", ast_key(node.arg))

    return ("Unknown", repr(node))


# -----------------------------------------------------------------------------
# Feature pool dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureExpr:
    expr: Node
    kind: str
    cost: int
    desc: str


def _sub(a: Node, b: Node) -> Node:
    return Add(a, Mul(ConstNode(-1.0), b))


def _sq(x: Node) -> Node:
    return Pow(x, 2.0)


def build_arg_pool(
    target: AtomNode,
    *,
    max_vars: int,
    max_args: int,
    include_compound_expr: bool = True,
    scales: Optional[Sequence[float]] = None,
    trig: bool = True,
    trig_squares: bool = True,
    trig_max_bases: int = 16,
    extra_exprs: Optional[List["FeatureExpr"]] = None,
) -> List[FeatureExpr]:
    """Build a small, high-yield argument pool for macro/template instantiation."""

    axes = [int(i) for i in getattr(target, "var_idxs", ()) or ()]
    if len(axes) > max_vars:
        axes = axes[:max_vars]

    # Scales: rich by default, but keep the pool *diverse* when max_args is
    # tight (macro_max_arg_exprs is often ~64). Otherwise the cost-sort can
    # fill the budget with dozens of (c*x_i) variants and crowd out higher-ROI
    # cost-2 structure (products/ratios) that matter for many AIF equations.
    if scales is None:
        scales = tuple(PHYSICS_SCALES)
        try:
            ma = int(max_args)
        except Exception:
            ma = None
        if ma is not None and ma <= 64 and len(scales) > 10:
            # Keep a small, diverse subset for compact pools.
            scales = (
                0.5,
                1.0 / 3.0,
                1.0 / 4.0,
                2.0,
                3.0,
                4.0,
                math.pi,
                2.0 * math.pi,
                1.0 / math.pi,
                1.0 / (2.0 * math.pi),
            )
    else:
        scales = tuple(scales)

    pool: List[FeatureExpr] = []
    seen: Set[Tuple] = set()

    def add(expr: Node, kind: str, cost: int, desc: str):
        k = ast_key(expr)
        if k in seen:
            return
        seen.add(k)
        pool.append(FeatureExpr(expr=expr, kind=str(kind), cost=int(cost), desc=str(desc)))

    # 0a) Inject externally-supplied expressions (e.g. from LogDeriv analysis).
    if extra_exprs:
        for fe in extra_exprs:
            add(fe.expr, fe.kind, fe.cost, fe.desc)

    # 0c) Small additive constants.
    #
    # The AI-Feynman benchmark (and many physics formulae) contain frequent
    # *additive* integer offsets: (x±1), (x±2), (1±x^2), (x4+2), etc.
    #
    # These are hard to reach via the macro/template arg pools unless we
    # explicitly seed a few constants. We keep this deliberately tiny to
    # avoid blowing up the pool.
    for c in (1.0, -1.0, 2.0, -2.0):
        add(ConstNode(float(c)), "const", 0, f"{c:g}")

    # 0) Raw vars
    for i in axes:
        add(Var(i), "var", 0, f"x{i}")

    # 0b) Common powers of single variables
    for i in axes:
        add(Pow(Var(i), 2.0), "pow2", 1, f"(x{i}**2)")
        add(Pow(Var(i), 4.0), "pow4", 2, f"(x{i}**4)")

    def clone(expr: Node) -> Node:
        try:
            return clone_ast(expr)
        except Exception:
            return expr

    # 1) Nontrivial effective input expressions z_i(x) as arguments (if present).
    #
    # Older callers only ever built atoms of the form NN[z(x), raw extras].
    # Stage A can now surface atoms like NN[p(x), q(x)], so every effective
    # input must be treated symmetrically by the shared feature grammar.
    if include_compound_expr and has_nontrivial_input(target):
        for local_idx, input_expr in enumerate(get_input_exprs(target)):
            if is_trivial_input(input_expr):
                continue
            kind = "compound" if local_idx == 0 else "compound_input"
            add(clone(input_expr), kind, 0, f"arg{local_idx}")
            # Treat z and 1/z symmetrically for exposed effective coordinates.
            # Stage A may choose either orientation for a valid compound; Stage B
            # macros should not depend on that arbitrary choice.
            add(Pow(clone(input_expr), -1.0), kind, 0, f"1/arg{local_idx}")
            add(Pow(clone(input_expr), 2.0), "compound_sq", 1, f"arg{local_idx}^2")
            add(Pow(clone(input_expr), -2.0), "compound_sq", 1, f"1/arg{local_idx}^2")

    # 1b) If nontrivial inputs exist, generate products and ratios in effective
    # input coordinates. This covers both legacy NN[z, x0, x1] and newer
    # multi-compound atoms such as NN[x0*x1, x2*x3].
    if include_compound_expr and has_nontrivial_input(target):
        input_exprs = tuple(get_input_exprs(target))

        for a in range(len(input_exprs)):
            for b in range(a + 1, len(input_exprs)):
                prod_expr = Mul(clone(input_exprs[a]), clone(input_exprs[b]))
                add(prod_expr, "effective_prod", 1, f"(arg{a}*arg{b})")
                add(
                    Div(clone(input_exprs[a]), clone(input_exprs[b])),
                    "effective_ratio",
                    1,
                    f"(arg{a}/arg{b})",
                )
                add(
                    Div(clone(input_exprs[b]), clone(input_exprs[a])),
                    "effective_ratio",
                    1,
                    f"(arg{b}/arg{a})",
                )

        # Products of two effective inputs compared to a third one are the
        # Planck-style feature family: e.g. NN[z=x2*x3, x0, x1] needs
        # (x0*x1)/z, while NN[p=x0*x1, q=x2*x3] needs p/q.
        if len(input_exprs) >= 3:
            for a in range(len(input_exprs)):
                for b in range(a + 1, len(input_exprs)):
                    prod_expr = Mul(clone(input_exprs[a]), clone(input_exprs[b]))
                    for c in range(len(input_exprs)):
                        if c in (a, b):
                            continue
                        add(
                            Div(clone(prod_expr), clone(input_exprs[c])),
                            "effective_prod_over_input",
                            1,
                            f"(arg{a}*arg{b})/arg{c}",
                        )
                        add(
                            Div(clone(input_exprs[c]), clone(prod_expr)),
                            "effective_input_over_prod",
                            2,
                            f"arg{c}/(arg{a}*arg{b})",
                        )

    # 2) Scaled vars
    for i in axes:
        for s in scales:
            add(Mul(ConstNode(float(s)), Var(i)), "scaled_var", 1, f"({s:g}*x{i})")

    # 3) Diffs and ratios
    for a in range(len(axes)):
        for b in range(a + 1, len(axes)):
            i, j = axes[a], axes[b]
            add(_sub(Var(i), Var(j)), "diff", 1, f"(x{i}-x{j})")
            add(_sub(Var(j), Var(i)), "diff", 1, f"(x{j}-x{i})")
            add(Div(Var(i), Var(j)), "ratio", 2, f"(x{i}/x{j})")
            add(Div(Var(j), Var(i)), "ratio", 2, f"(x{j}/x{i})")

    # 3b) Trig wrappers for a few cheap phase-like expressions.
    if trig:
        trig_bases = [p for p in pool if p.kind in ("var", "diff", "scaled_var", "compound", "compound_input")]
        try:
            nmax = int(trig_max_bases)
        except Exception:
            nmax = 16
        trig_bases = trig_bases[: min(max(0, nmax), len(trig_bases))]
        for tb in trig_bases:
            try:
                base = clone_ast(tb.expr)
            except Exception:
                base = tb.expr
            for nm, expr in build_trig_wrappers(
                base, omega=1.0, include_sin=True, include_cos=True, include_sin2=bool(trig_squares), include_cos2=False
            ):
                try:
                    if nm == "sin":
                        add(expr, "sin", tb.cost + 1, f"sin({tb.desc})")
                    elif nm == "cos":
                        add(expr, "cos", tb.cost + 1, f"cos({tb.desc})")
                    elif nm == "sin2":
                        add(expr, "sin2", tb.cost + 2, f"(sin({tb.desc})**2)")
                    elif nm == "cos2":
                        add(expr, "cos2", tb.cost + 2, f"(cos({tb.desc})**2)")
                except Exception:
                    pass

    # 3c) Affine shifts by small integers for a few cheap bases.
    #
    # This unlocks motifs like (1 - x^2), (1 + u), (x - 1), (x + 2) that
    # commonly appear as *prefactors* or inside simple denominators.
    #
    # We only apply this to a small prefix of cheap bases to keep the pool
    # bounded (macro_max_arg_exprs is often small by design).
    shift_consts = (1.0, 2.0)
    shift_kinds = (
        "var",
        "pow2",
        "ratio",
        "diff",
        "compound",
        "compound_input",
        "effective_ratio",
        "sin",
        "cos",
    )
    shift_bases = [p for p in pool if p.kind in shift_kinds]
    shift_bases = shift_bases[: min(12, len(shift_bases))]
    for b in shift_bases:
        for c in shift_consts:
            try:
                add(Add(clone_ast(b.expr), ConstNode(float(c))), "affine", b.cost + 1, f"({b.desc}+{c:g})")
                add(Add(clone_ast(b.expr), ConstNode(float(-c))), "affine", b.cost + 1, f"({b.desc}-{c:g})")
                add(_sub(ConstNode(float(c)), clone_ast(b.expr)), "affine", b.cost + 1, f"({c:g}-{b.desc})")
            except Exception:
                pass

    # 3d) Products of a few cheap ratio/trig terms (e.g., (x1/x0)*cos(x3)).
    cheapA = [
        p for p in pool
        if p.kind in (
            "ratio",
            "diff",
            "scaled_var",
            "sin",
            "cos",
            "compound",
            "compound_input",
            "compound_sq",
            "effective_ratio",
        )
    ]
    cheapB = [p for p in pool if p.kind in ("ratio", "var", "sin", "cos")]
    cheapA = cheapA[: min(12, len(cheapA))]
    cheapB = cheapB[: min(12, len(cheapB))]
    for a in cheapA:
        ka = ast_key(a.expr)
        for b in cheapB:
            kb = ast_key(b.expr)
            if kb < ka:
                continue
            try:
                add(Mul(clone_ast(a.expr), clone_ast(b.expr)), "prod", a.cost + b.cost + 1, f"({a.desc}*{b.desc})")
            except Exception:
                pass

    # 4) Simple products
    for a in range(len(axes)):
        for b in range(a + 1, len(axes)):
            i, j = axes[a], axes[b]
            add(Mul(Var(i), Var(j)), "prod", 1, f"(x{i}*x{j})")

    # var * (var^2) and var * (var^4)
    vars_basic = [p for p in pool if p.kind in ("var", "scaled_var")]
    pow_terms = [p for p in pool if p.kind in ("pow2", "pow4")]
    for v in vars_basic[: min(12, len(vars_basic))]:
        for p2 in pow_terms[: min(12, len(pow_terms))]:
            try:
                add(Mul(v.expr, p2.expr), "prod", v.cost + p2.cost + 1, f"({v.desc}*{p2.desc})")
            except Exception:
                pass

    # scaled_var * diff
    scaled_vars = [p for p in pool if p.kind == "scaled_var"]
    diffs = [p for p in pool if p.kind == "diff"]
    for sv in scaled_vars[: min(12, len(scaled_vars))]:
        for d in diffs[: min(12, len(diffs))]:
            try:
                add(Mul(sv.expr, d.expr), "prod", sv.cost + d.cost + 1, f"({sv.desc}*{d.desc})")
            except Exception:
                pass

    # 5) Affine mixes: var ± prod, and pow2 ± pow2
    var_terms = [p for p in pool if p.kind == "var"]
    prod_terms = [p for p in pool if p.kind == "prod"]
    pow2_terms = [p for p in pool if p.kind == "pow2"]

    for v in var_terms[: min(10, len(var_terms))]:
        for pr in prod_terms[: min(12, len(prod_terms))]:
            try:
                add(_sub(clone_ast(v.expr), clone_ast(pr.expr)), "affine", v.cost + pr.cost + 1, f"({v.desc}-{pr.desc})")
                add(Add(clone_ast(v.expr), clone_ast(pr.expr)), "affine", v.cost + pr.cost + 1, f"({v.desc}+{pr.desc})")
            except Exception:
                pass

    pow2_terms = pow2_terms[: min(10, len(pow2_terms))]
    for i in range(len(pow2_terms)):
        for j in range(i + 1, len(pow2_terms)):
            a, b = pow2_terms[i], pow2_terms[j]
            try:
                add(_sub(clone_ast(a.expr), clone_ast(b.expr)), "diff_sq", a.cost + b.cost + 1, f"({a.desc}-{b.desc})")
                add(Add(clone_ast(a.expr), clone_ast(b.expr)), "sum_sq", a.cost + b.cost + 1, f"({a.desc}+{b.desc})")
            except Exception:
                pass

    pool.sort(key=lambda p: (p.cost, len(p.desc)))
    if len(pool) > max_args:
        pool = pool[: int(max_args)]
    return pool


def build_factor_pool(
    pool: List[FeatureExpr],
    *,
    max_base_factors: int,
    max_factor_exprs: int,
    base_kinds: Optional[Set[str]] = None,
) -> List[FeatureExpr]:
    """Build a small pool of *outer algebra* factors for macro/template proposals."""

    if base_kinds is None:
        base_kinds = {
            "var",
            "pow2",
            "pow4",
            "scaled_var",
            "prod",
            "ratio",
            "diff",
            "compound",
            "compound_input",
            "compound_sq",
            "compound_ratio",
            "compound_sq_ratio",
            "extra_prod",
            "prod_over_compound",
            "compound_over_prod",
            "effective_prod",
            "effective_ratio",
            "effective_prod_over_input",
            "effective_input_over_prod",
            "sin",
            "cos",
            "sin2",
            "affine",
            "diff_sq",
            "sum_sq",
        }

    base = [a for a in pool if a.kind in base_kinds]
    if not base:
        return []
    base = base[: max(1, int(max_base_factors))]

    factors: List[FeatureExpr] = []
    seen: Set[Tuple] = set()

    def add(expr: Node, kind: str, cost: int, desc: str):
        k = ast_key(expr)
        if k in seen:
            return
        seen.add(k)
        factors.append(FeatureExpr(expr=expr, kind=str(kind), cost=int(cost), desc=str(desc)))

    # 1) Base factors as-is
    for b in base:
        try:
            add(clone_ast(b.expr), b.kind, b.cost, b.desc)
        except Exception:
            add(b.expr, b.kind, b.cost, b.desc)

    # 2) Inverses of base factors (skip scaled_var inverses; constants can be absorbed by alpha)
    for b in base:
        if b.kind == "scaled_var":
            continue
        try:
            add(Pow(clone_ast(b.expr), -1.0), "inv", b.cost + 1, f"1/({b.desc})")
        except Exception:
            pass

    # 3) Pairwise products of (cheap) factors
    atoms = sorted(factors, key=lambda a: (a.cost, len(a.desc)))
    atoms = atoms[: min(len(atoms), max(8, int(max_base_factors)))]
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            a, b = atoms[i], atoms[j]
            if a.kind == "scaled_var" and b.kind == "scaled_var":
                continue
            try:
                add(Mul(clone_ast(a.expr), clone_ast(b.expr)), "prod", a.cost + b.cost + 1, f"({a.desc}*{b.desc})")
            except Exception:
                pass

    factors.sort(key=lambda a: (a.cost, len(a.desc)))
    if len(factors) > max_factor_exprs:
        factors = factors[: int(max_factor_exprs)]
    return factors


# -----------------------------------------------------------------------------
# Wrapper generation
# -----------------------------------------------------------------------------


def build_rational_wrappers(z_ast: Node, *, flavor: str = "default") -> List[Tuple[str, Node]]:
    """Generate a small set of algebraic/rational wrappers around a base expression.

    This is used primarily for Stage-A compound-variable variants, but is
    generic enough for Stage-B feature construction too.

    Parameters
    ----------
    flavor : str
        "default" : generic wrappers for ratio-like / squared-difference forms.
        "radial"  : bias towards (1+z) denominators (e.g. 1/(1+r^2)).
    """

    z2 = PowNode(z_ast, 2.0)
    z4 = PowNode(z_ast, 4.0)
    den_zm1 = AddNode(z_ast, ConstNode(-1.0))  # (z - 1)
    den_zp1 = AddNode(z_ast, ConstNode(1.0))  # (z + 1)
    den_z2m1 = AddNode(z2, ConstNode(-1.0))  # (z^2 - 1)
    den_z2p1 = AddNode(z2, ConstNode(1.0))  # (z^2 + 1)

    if str(flavor).lower() == "radial":
        return [
            ("rat_inv_zp1", PowNode(den_zp1, -1.0)),
            ("rat_z_over_zp1", MulNode(z_ast, PowNode(den_zp1, -1.0))),
            ("rat_inv", PowNode(z_ast, -1.0)),
            ("rat_inv_z2p1", PowNode(den_z2p1, -1.0)),
        ]

    return [
        ("rat_inv", PowNode(z_ast, -1.0)),
        ("rat_z2", z2),
        ("rat_inv_z2", PowNode(z_ast, -2.0)),
        ("rat_z4", z4),
        ("rat_inv_zp1", PowNode(den_zp1, -1.0)),
        ("rat_z_over_zp1", MulNode(z_ast, PowNode(den_zp1, -1.0))),
        ("rat_inv_z2m1", PowNode(den_z2m1, -1.0)),
        ("rat_inv_z2p1", PowNode(den_z2p1, -1.0)),
        ("rat_z2_over_z2m1", MulNode(z2, PowNode(den_z2m1, -1.0))),
        ("rat_z2_over_z2p1", MulNode(z2, PowNode(den_z2p1, -1.0))),
        ("rat_z4_over_z2m1_sq", MulNode(z4, PowNode(den_z2m1, -2.0))),
        ("rat_z4_over_z2p1_sq", MulNode(z4, PowNode(den_z2p1, -2.0))),
        ("rat_inv_zm1", PowNode(den_zm1, -1.0)),
    ]


def build_shape_wrappers(
    expr: Node,
    *,
    include_square: bool = False,
    include_abs: bool = False,
    include_sqrt: bool = False,
) -> List[Tuple[str, Node]]:
    """Generate simple power/shape wrappers (sq/abs/sqrt) for an expression.

    Names are chosen to match Stage-A selection heuristics.
    """

    out: List[Tuple[str, Node]] = []
    if include_sqrt:
        out.append(("sqrt", PowNode(expr, 0.5)))
    if include_square:
        out.append(("sq", PowNode(expr, 2.0)))
    if include_abs:
        out.append(("abs", PowNode(PowNode(expr, 2.0), 0.5)))
    return out


def build_trig_wrappers(
    expr: Node,
    *,
    omega: float = 1.0,
    include_sin: bool = True,
    include_cos: bool = True,
    include_sin2: bool = False,
    include_cos2: bool = False,
) -> List[Tuple[str, Node]]:
    """Generate trig wrappers around an expression.

    If omega != 1, uses sin(omega*expr), cos(omega*expr).

    Returns pairs (name, expr) where name is one of: sin, cos, sin2, cos2.
    """

    arg = expr
    try:
        w = float(omega)
    except Exception:
        w = 1.0
    if abs(w - 1.0) > 1e-12:
        arg = MulNode(ConstNode(w), expr)

    out: List[Tuple[str, Node]] = []
    if include_cos:
        out.append(("cos", CosNode(arg)))
    if include_sin:
        out.append(("sin", SinNode(arg)))
    if include_cos2:
        out.append(("cos2", PowNode(CosNode(arg), 2.0)))
    if include_sin2:
        out.append(("sin2", PowNode(SinNode(arg), 2.0)))
    return out


def build_wrapper_variants(
    expr: Node,
    *,
    rational: bool = False,
    rational_flavor: str = "default",
    trig: bool = False,
    omega: float = 1.0,
    square: bool = False,
    abs_: bool = False,
    sqrt: bool = False,
    include_sin: bool = True,
    include_cos: bool = True,
) -> List[Tuple[str, Node]]:
    """Convenience helper: combine multiple wrapper families for a base expr.

    This returns *only* wrapped variants (not the raw expr itself).
    """

    out: List[Tuple[str, Node]] = []
    if square or abs_ or sqrt:
        out.extend(build_shape_wrappers(expr, include_square=square, include_abs=abs_, include_sqrt=sqrt))
    if rational:
        out.extend(build_rational_wrappers(expr, flavor=rational_flavor))
    if trig:
        out.extend(build_trig_wrappers(expr, omega=omega, include_sin=include_sin, include_cos=include_cos))
    return out
