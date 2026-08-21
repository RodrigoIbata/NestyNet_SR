# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compound-function macro library + proposal pipeline.

This is the Stage-B-facing counterpart to *compound variables*.

Instead of only proposing new coordinates z(x), this module proposes
high-payoff *functional motifs* ("macros") built from the existing
primitive AST grammar (Add/Mul/Pow/Sin/Cos/Exp/Log + Var/Const).

Design goals
------------
- **Small surface area**: adding a new macro should be ~10 lines.
- **Non-invasive**: macros expand to ordinary AST nodes (no new runtime op).
- **Fast screening**: rank candidates via a cheap affine fit on a cached batch.
- **Compound-aware**: if a target NN leaf already has a compound coordinate
  z(x) (via atom.inputs), we treat that z-expression as a first-class argument candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

# -----------------------------------------------------------------------------
# torch compatibility: nanstd
# -----------------------------------------------------------------------------
# Some Stage-B macro screening logic historically used `torch.nanstd`, but
# PyTorch does not ship this helper on many versions (including 2.5). When
# missing, `compound_fn_macros` can crash and silently disable a key path to
# solving rational/trig benchmark structures.
#
# Provide a small, robust fallback so any accidental `torch.nanstd` call inside
# the macro pipeline (or future edits) stays portable.
if not hasattr(torch, "nanstd"):
    def _nanstd(x: torch.Tensor, dim=None, unbiased: bool = False, keepdim: bool = False) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        m = torch.isfinite(x)
        if dim is None:
            x_ok = x[m]
            if int(x_ok.numel()) <= 1:
                return torch.zeros((), dtype=x.dtype, device=x.device)
            return x_ok.std(unbiased=unbiased)
        x0 = torch.where(m, x, torch.zeros_like(x))
        cnt = m.sum(dim=dim, keepdim=True).clamp(min=1)
        mean = x0.sum(dim=dim, keepdim=True) / cnt
        d = torch.where(m, x - mean, torch.zeros_like(x))
        var = (d * d).sum(dim=dim, keepdim=True) / cnt
        out = var.sqrt()
        if not keepdim:
            out = out.squeeze(dim)
        return out

    torch.nanstd = _nanstd  # type: ignore[attr-defined]

from nestynet_sr.sr_core.bridges import (
    Add,
    AtomNode,
    Cos,
    ConstNode,
    Div,
    Scale,
    Mul,
    Node,
    Pow,
    Sin,
    Var,
    clone_ast,
    eval_input_expr,
    get_input_exprs,
    has_nontrivial_input,
    replace_atom_in_ast,
)
from nestynet_sr.sr_core.units import eval_analytic_expr_dim

from .batch_utils import first_batch_xy
from .candidate_builders import _build_atom_input_tensor
from .feature_grammar import FeatureExpr as ArgExpr
from .feature_grammar import ast_key, build_arg_pool, build_factor_pool
from .wrapper_policy import macro_arg_wrapper_policy

# -----------------------------------------------------------------------------
# Macro registry
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroSpec:
    name: str
    arity: int
    build: Callable[..., Node]
    arg_kinds: Optional[Tuple[Optional[Sequence[str]], ...]] = None
    commutative: bool = False
    affine: bool = True
    prefer_factor: float = 0.1


_CLEAN_SINGLETON_MACROS = {
    "inv_sqrt1m",
    "sqrt1m",
    "inv1m",
    "inv1p",
    "sqrt1p",
}


def _sub(a: Node, b: Node) -> Node:
    return Add(a, Mul(ConstNode(-1.0), b))


def _sqrt(x: Node) -> Node:
    return Pow(x, 0.5)


def _sq(x: Node) -> Node:
    return Pow(x, 2.0)


def _default_macros() -> List[MacroSpec]:
    """Default macro library (high ROI, physics-heavy motifs)."""

    def half(z: Node) -> Node:
        return Mul(ConstNode(0.5), z)

    def sinc(z: Node) -> Node:
        return Div(Sin(z), z)

    def sinc_sq(z: Node) -> Node:
        return _sq(sinc(z))

    def sin_half_sq(z: Node) -> Node:
        return _sq(Sin(half(z)))

    def cos_half_sq(z: Node) -> Node:
        return _sq(Cos(half(z)))

    def one_minus_cos(z: Node) -> Node:
        return _sub(ConstNode(1.0), Cos(z))

    def inv_sin_half_sq(z: Node) -> Node:
        return Pow(Sin(half(z)), -2.0)

    def inv_sin_half4(z: Node) -> Node:
        return Pow(Sin(half(z)), -4.0)

    def sin_ratio(a: Node, z: Node) -> Node:
        return Div(Sin(Mul(a, z)), Sin(z))

    def sin_ratio_sq(a: Node, z: Node) -> Node:
        return _sq(sin_ratio(a, z))

    def sqrt1m(r: Node) -> Node:
        return _sqrt(_sub(ConstNode(1.0), _sq(r)))

    def inv_sqrt1m(r: Node) -> Node:
        return Pow(_sub(ConstNode(1.0), _sq(r)), -0.5)

    def sqrt1p(u: Node) -> Node:
        return _sqrt(Add(ConstNode(1.0), u))

    def inv1m(u: Node) -> Node:
        return Pow(_sub(ConstNode(1.0), u), -1.0)

    def inv1p(u: Node) -> Node:
        return Pow(Add(ConstNode(1.0), u), -1.0)

    def hypot(u: Node, v: Node) -> Node:
        return _sqrt(Add(_sq(u), _sq(v)))

    return [
        MacroSpec(
            "sinc",
            1,
            sinc,
            arg_kinds=(("var", "scaled_var", "diff", "prod", "extra_prod", "ratio", "prod_over_compound", "compound_over_prod", "compound"),),
            prefer_factor=0.2,
        ),
        MacroSpec(
            "sinc_sq",
            1,
            sinc_sq,
            arg_kinds=(("var", "scaled_var", "diff", "prod", "extra_prod", "ratio", "prod_over_compound", "compound_over_prod", "compound"),),
            prefer_factor=0.2,
        ),
        MacroSpec(
            "sin_half_sq",
            1,
            sin_half_sq,
            arg_kinds=(("var", "scaled_var", "diff", "prod", "extra_prod", "ratio", "prod_over_compound", "compound_over_prod", "compound"),),
            prefer_factor=0.2,
        ),
        MacroSpec(
            "cos_half_sq",
            1,
            cos_half_sq,
            arg_kinds=(("var", "scaled_var", "diff", "prod", "extra_prod", "ratio", "prod_over_compound", "compound_over_prod", "compound"),),
            prefer_factor=0.2,
        ),
        MacroSpec(
            "one_minus_cos",
            1,
            one_minus_cos,
            arg_kinds=(("var", "scaled_var", "diff", "prod", "extra_prod", "ratio", "prod_over_compound", "compound_over_prod", "compound"),),
            prefer_factor=0.2,
        ),
        MacroSpec(
            "inv_sin_half_sq",
            1,
            inv_sin_half_sq,
            arg_kinds=(("var", "scaled_var", "diff", "prod", "extra_prod", "ratio", "prod_over_compound", "compound_over_prod", "compound"),),
            prefer_factor=0.2,
        ),
        MacroSpec(
            "inv_sin_half4",
            1,
            inv_sin_half4,
            arg_kinds=(("var", "scaled_var", "diff", "prod", "extra_prod", "ratio", "prod_over_compound", "compound_over_prod", "compound"),),
            prefer_factor=0.2,
        ),
        MacroSpec(
            "sin_ratio",
            2,
            sin_ratio,
            arg_kinds=(
                ("var", "scaled_var"),
                ("var", "scaled_var", "diff", "compound"),
            ),
            prefer_factor=0.15,
        ),
        MacroSpec(
            "sin_ratio_sq",
            2,
            sin_ratio_sq,
            arg_kinds=(
                ("var", "scaled_var"),
                ("var", "scaled_var", "diff", "compound"),
            ),
            prefer_factor=0.15,
        ),
        MacroSpec(
            "sqrt1m",
            1,
            sqrt1m,
            arg_kinds=(("ratio", "prod_over_compound", "compound_over_prod", "var", "scaled_var", "compound", "compound_sq", "compound_ratio", "compound_sq_ratio"),),
            prefer_factor=0.25,
        ),
        MacroSpec(
            "inv_sqrt1m",
            1,
            inv_sqrt1m,
            arg_kinds=(("ratio", "prod_over_compound", "compound_over_prod", "var", "scaled_var", "compound", "compound_sq", "compound_ratio", "compound_sq_ratio"),),
            prefer_factor=0.25,
        ),
        MacroSpec(
            "sqrt1p",
            1,
            sqrt1p,
            arg_kinds=(("ratio", "prod_over_compound", "compound_over_prod", "var", "scaled_var", "prod", "extra_prod", "compound", "compound_sq", "compound_ratio", "compound_sq_ratio"),),
            prefer_factor=0.25,
        ),
        MacroSpec(
            "inv1m",
            1,
            inv1m,
            arg_kinds=(("ratio", "prod_over_compound", "compound_over_prod", "var", "scaled_var", "prod", "extra_prod", "compound", "compound_sq", "compound_ratio", "compound_sq_ratio"),),
            prefer_factor=0.15,
        ),
        MacroSpec(
            "inv1p",
            1,
            inv1p,
            arg_kinds=(("ratio", "prod_over_compound", "compound_over_prod", "var", "scaled_var", "prod", "extra_prod", "compound", "compound_sq", "compound_ratio", "compound_sq_ratio"),),
            prefer_factor=0.15,
        ),
        MacroSpec(
            "hypot",
            2,
            hypot,
            arg_kinds=(
                ("var", "scaled_var", "diff", "prod", "extra_prod", "compound"),
                ("var", "scaled_var", "diff", "prod", "extra_prod", "compound"),
            ),
            commutative=True,
            prefer_factor=0.2,
        ),
    ]


# -----------------------------------------------------------------------------
# Argument/factor pool builders live in sr_search.feature_grammar.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Screening + candidate building
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroHit:
    spec: MacroSpec
    args: Tuple[ArgExpr, ...]
    score: float
    alpha: float
    beta: float
    ok_frac: float
    mse: float
    tier: int = 1
    outer_op: Optional[str] = None  # 'mul' or 'div'
    outer_factor: Optional[ArgExpr] = None


def _macro_hit_arg_key(h: MacroHit) -> Tuple[str, ...]:
    try:
        return tuple(str(ast_key(a.expr)) for a in h.args)
    except Exception:
        return tuple(str(getattr(a, "desc", "?")) for a in h.args)


def _macro_hit_identity(h: MacroHit) -> Tuple[Any, ...]:
    try:
        outer_key = (
            ast_key(h.outer_factor.expr)
            if getattr(h, "outer_factor", None) is not None
            else None
        )
    except Exception:
        outer_key = str(getattr(getattr(h, "outer_factor", None), "desc", None))
    return (
        str(getattr(h.spec, "name", "")),
        int(getattr(h, "tier", 1)),
        _macro_hit_arg_key(h),
        str(getattr(h, "outer_op", "") or ""),
        outer_key,
    )


def _is_clean_singleton_macro_hit(h: MacroHit) -> bool:
    try:
        return bool(
            str(getattr(h.spec, "name", "")) in _CLEAN_SINGLETON_MACROS
            and int(getattr(h, "tier", 1)) == 1
            and int(getattr(h.spec, "arity", 0)) == 1
            and getattr(h, "outer_factor", None) is None
            and getattr(h, "outer_op", None) is None
        )
    except Exception:
        return False


def _macro_noise_aware(ctx: Any) -> bool:
    try:
        from .model_selection import resolve_acceptance_noise_floor_raw

        nf = float(
            resolve_acceptance_noise_floor_raw(
                getattr(ctx, "lm_hp", None),
                float(getattr(ctx, "loss_scale", 1.0) or 1.0),
            )
        )
        return bool(math.isfinite(nf) and nf > 0.0)
    except Exception:
        return False


def _macro_hit_meta(
    h: MacroHit,
    *,
    log: str,
    feature_key: Any,
    no_intercept: bool = False,
) -> Dict[str, Any]:
    clean = _is_clean_singleton_macro_hit(h)
    meta: Dict[str, Any] = {
        "log": log,
        "screen_r2": float(getattr(h, "score", -float("inf"))),
        "ok_frac": float(getattr(h, "ok_frac", 0.0)),
        "n_terms": 1,
        "macro_tier": int(getattr(h, "tier", 1)),
        "macro_name": str(getattr(h.spec, "name", "")),
        "macro_arg_key": str(_macro_hit_arg_key(h)),
        "macro_feature_key": str(feature_key),
        "macro_clean_singleton": bool(clean),
        "macro_combo": False,
    }
    if no_intercept:
        meta["macro_no_intercept"] = True
    return meta


def _rich_macro_meta(
    *,
    log: str,
    screen_r2: float,
    ok_frac: float,
    n_terms: int,
    macro_name: str,
    feature_keys: Sequence[Any] = (),
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "log": log,
        "screen_r2": float(screen_r2),
        "ok_frac": float(ok_frac),
        "n_terms": int(n_terms),
        "macro_tier": 3,
        "macro_name": str(macro_name),
        "macro_feature_keys": [str(k) for k in feature_keys],
        "macro_clean_singleton": False,
        "macro_combo": True,
    }
    if extra:
        meta.update(extra)
    return meta


def _affine_fit(m: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    m = m.view(-1)
    y = y.view(-1)
    m_mean = float(m.mean().item())
    y_mean = float(y.mean().item())
    dm = m - m_mean
    dy = y - y_mean
    var_m = float((dm * dm).mean().item())
    if not math.isfinite(var_m) or var_m < 1e-18:
        return 0.0, y_mean
    cov = float((dm * dy).mean().item())
    alpha = cov / var_m
    beta = y_mean - alpha * m_mean
    return float(alpha), float(beta)


def _score_affine(m: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> Tuple[float, float]:
    y = y.view(-1)
    pred = (alpha * m.view(-1) + beta)
    resid = pred - y
    mse = float((resid * resid).mean().item())
    var_y = float(((y - float(y.mean().item())) ** 2).mean().item())
    if not math.isfinite(var_y) or var_y < 1e-18:
        return -float("inf"), mse
    r2 = 1.0 - (mse / var_y)
    return float(r2), mse


def _macro_expr_from(spec: MacroSpec, args: Sequence[ArgExpr]) -> Node:
    if spec.arity != len(args):
        raise ValueError("macro arity mismatch")
    exprs = [clone_ast(a.expr) for a in args]
    return spec.build(*exprs)


def _macro_units_ok(ctx: Any, expr: Node) -> bool:
    """Reject macro internals with unit-invalid transcendental arguments.

    This is a proposal/ranking prefilter only.  The full candidate still goes
    through Stage B's normal units precheck and validation after it is built.
    """

    if not bool(getattr(ctx, "enforce_units", False)):
        return True
    spec = getattr(ctx, "units_spec", None)
    if spec is None:
        return True
    try:
        x_dims = tuple(getattr(spec, "x_dims", ()) or ())
    except Exception:
        x_dims = ()
    if not x_dims:
        return True
    try:
        return eval_analytic_expr_dim(expr, x_dims) is not None
    except Exception:
        return False


def _allowed_args_for_pos(spec: MacroSpec, pool: List[ArgExpr], pos: int) -> List[ArgExpr]:
    if not spec.arg_kinds or pos >= len(spec.arg_kinds) or spec.arg_kinds[pos] is None:
        return pool
    allowed = set(str(k) for k in (spec.arg_kinds[pos] or ()))
    return [a for a in pool if a.kind in allowed]


def _select_args_for_pos(
    spec: MacroSpec,
    pool: List[ArgExpr],
    pos: int,
    max_args: int,
    *,
    n_trig_prods: int = 4,
    n_plain_prods: int = 2,
    n_affines: int = 2,
) -> List[ArgExpr]:
    """Select a small, diverse set of arguments for a macro position.

    Why: the global pool is cost-sorted and dominated by cheap scaled vars.
    For many benchmark formulas we need at least a few *structured* terms
    (e.g. var*cos(z), ratio*cos(theta), 1-x^2) to survive the shortlist even
    when they are not among the first `max_args` entries.

    This function stays bounded: it still returns <= max_args.
    """
    allowed = _allowed_args_for_pos(spec, pool, pos)
    if max_args <= 0 or len(allowed) <= max_args:
        return allowed

    used = set()
    out: List[ArgExpr] = []

    def _add(a: ArgExpr) -> None:
        k = ast_key(a.expr)
        if k in used:
            return
        used.add(k)
        out.append(a)

    # 1) Force in a few trig-products (these are high-value for many AIF leaves).
    trig_prod_kinds = {"prod", "extra_prod", "prod_over_compound", "compound_over_prod"}
    trig_prods = [a for a in allowed if (a.kind in trig_prod_kinds) and ("sin" in a.desc or "cos" in a.desc)]
    plain_prods = [a for a in allowed if (a.kind in trig_prod_kinds) and a not in trig_prods]
    affines = [a for a in allowed if a.kind == "affine"]

    for a in trig_prods[: max(0, int(n_trig_prods))]:
        _add(a)
        if len(out) >= max_args:
            return out

    for a in plain_prods[: max(0, int(n_plain_prods))]:
        _add(a)
        if len(out) >= max_args:
            return out

    for a in affines[: max(0, int(n_affines))]:
        _add(a)
        if len(out) >= max_args:
            return out

    # 2) Fill the remainder with the globally cheapest allowed args.
    for a in allowed:
        _add(a)
        if len(out) >= max_args:
            break
    return out


def _seed_square_prefactors(target: AtomNode) -> List[ArgExpr]:
    """A tiny set of high-ROI prefactors that are easy to miss via cost-sorting.

    This is intentionally small and generic; it helps lots of AIF-style forms.
    """
    axes = [int(i) for i in (getattr(target, "var_idxs", ()) or ())]
    seen = set()
    out: List[ArgExpr] = []

    def _add(expr: Node, desc: str) -> None:
        k = ast_key(expr)
        if k in seen:
            return
        seen.add(k)
        out.append(ArgExpr(expr=expr, kind="affine", cost=2, desc=desc))

    for i in axes[:6]:
        xi = Var(i)
        x2 = Pow(clone_ast(xi), 2.0)
        _add(Add(ConstNode(1.0), Mul(ConstNode(-1.0), clone_ast(x2))), f"(1-x{i}**2)")
        _add(Add(ConstNode(1.0), clone_ast(x2)), f"(1+x{i}**2)")
    return out


def propose_compound_function_macros(
    ctx: Any,
    target: AtomNode,
    *,
    macros: Optional[List[MacroSpec]] = None,
) -> List[Any]:
    """Return Stage-B `Candidate`s that replace `target` with macro-based motifs.

    This function is intentionally "StageB-light": it only relies on ctx.state,
    the dataloaders, and core AST utilities.
    """

    # Lazy import to avoid hard dependency / circular import at module import time.
    from nestynet_sr.sr_search.stageB import Candidate, build_atom_to_leaf_map


    st = ctx.state

    # Hyperparameters (kept under lm_hp to avoid proliferating config objects)
    hp = getattr(ctx, "lm_hp", None)
    enable = bool(getattr(hp, "macro_enable", True))
    if not enable:
        return []
    max_vars = int(getattr(hp, "macro_max_vars", 6))
    max_args = int(getattr(hp, "macro_max_arg_exprs", 64))
    max_hits = int(getattr(hp, "macro_max_candidates", 10))
    max_per_macro = int(getattr(hp, "macro_max_per_macro", 3))
    max_pos_args = int(getattr(hp, "macro_max_pos_args", 20))
    ok_frac_min = float(getattr(hp, "macro_domain_ok_frac", 0.98))
    noise_aware_macro_selection = _macro_noise_aware(ctx)
    clean_singleton_cap = max(0, int(getattr(hp, "macro_clean_singleton_cap", 3) or 0))

    # Tier-2: macro(args) multiplied/divided by a simple outer factor(x).
    tier2_enable = bool(getattr(hp, "macro_tier2_enable", True))
    tier2_try_mul = bool(getattr(hp, "macro_tier2_try_mul", True))
    tier2_try_div = bool(getattr(hp, "macro_tier2_try_div", True))
    tier2_max_base_factors = int(getattr(hp, "macro_tier2_max_base_factors", 24))
    tier2_max_factor_exprs = int(getattr(hp, "macro_tier2_max_factor_exprs", 96))
    tier2_max_per_macro = int(getattr(hp, "macro_tier2_max_per_macro", 3))
    tier2_div_eps = float(getattr(hp, "macro_tier2_div_eps", 1.0e-12))

    # Tier-3: small linear combinations of a few analytic features (additive).
    tier3_enable = bool(getattr(hp, "macro_tier3_enable", True))
    tier3_try_pairs = bool(getattr(hp, "macro_tier3_try_pairs", True))
    tier3_try_triples = bool(getattr(hp, "macro_tier3_try_triples", False))
    tier3_max_macro_features = int(getattr(hp, "macro_tier3_max_macro_features", 12))
    tier3_max_aux_features = int(getattr(hp, "macro_tier3_max_aux_features", 24))
    tier3_max_candidates = int(getattr(hp, "macro_tier3_max_candidates", 6))
    tier3_ridge = float(getattr(hp, "macro_tier3_ridge", 1.0e-8))
    tier3_allow_macro_macro = bool(getattr(hp, "macro_tier3_allow_macro_macro", True))
    tier3_allow_aux_aux = bool(getattr(hp, "macro_tier3_allow_aux_aux", False))
    # Prune numerically-insignificant terms in tier-3 combos.
    # This helps avoid structurally-ugly expressions caused by ~0 coefficients.
    tier3_prune_rel_contrib = float(getattr(hp, "macro_tier3_prune_rel_contrib", 1.0e-10))
    tier3_subset_r2_tol = float(getattr(hp, "macro_tier3_subset_r2_tol", 1.0e-12))
    # Tier-3b: shared-prefactor mode (multiplicative structure)
    #   y ≈ p(x) * (a*macro(args) + b)
    # i.e. fit (y/p) as an affine function of a macro core.
    sharedpref_enable = bool(getattr(hp, "macro_tier3_shared_prefactor_enable", True))
    sharedpref_max_prefactors = int(getattr(hp, "macro_tier3_shared_prefactor_max_prefactors", 24))
    sharedpref_max_pos_args = int(getattr(hp, "macro_tier3_shared_prefactor_max_pos_args", 12))
    sharedpref_max_macro_exprs = int(getattr(hp, "macro_tier3_shared_prefactor_max_macro_exprs", 256))
    sharedpref_max_per_prefactor = int(getattr(hp, "macro_tier3_shared_prefactor_max_per_prefactor", 8))
    sharedpref_max_pos_args_arity2 = int(
        getattr(
            hp,
            "macro_tier3_shared_prefactor_max_pos_args_arity2",
            max(4, min(int(sharedpref_max_pos_args), 8)),
        )
    )

    macro_factors_enable = bool(getattr(hp, "macro_tier2_macro_factors_enable", True))
    macro_factors_max = int(getattr(hp, "macro_tier2_macro_factors_max", 8))
    macro_factors_per_spec = int(getattr(hp, "macro_tier2_macro_factors_per_spec", 1))
    macro_factors_sharedpref = bool(
        getattr(hp, "macro_tier3_shared_prefactor_allow_macro_prefactors", macro_factors_enable)
    )
    pool_min_args_for_compound = int(getattr(hp, "macro_pool_min_arg_exprs_for_compound", 96))
    sharedpref_max_candidates = int(getattr(hp, "macro_tier3_shared_prefactor_max_candidates", 6))
    sharedpref_div_eps = float(getattr(hp, "macro_tier3_shared_prefactor_div_eps", 1.0e-12))

    # Tier-3c: shared-prefactor + residual constant term
    #   y ≈ p(x) * (a*macro(args) + b) + c
    # Screened as a small linear regression in original y-space:
    #   y ≈ a*(p*m) + b*p + c.
    sharedpref_resid_enable = bool(getattr(hp, "macro_tier3_shared_prefactor_residual_enable", True))
    sharedpref_resid_max_per_prefactor = int(getattr(hp, "macro_tier3_shared_prefactor_residual_max_per_prefactor", sharedpref_max_per_prefactor))
    sharedpref_resid_max_candidates = int(getattr(hp, "macro_tier3_shared_prefactor_residual_max_candidates", sharedpref_max_candidates))
    sharedpref_resid_ridge = float(getattr(hp, "macro_tier3_shared_prefactor_residual_ridge", tier3_ridge))


    if macros is None:
        macros = _default_macros()

    # Optional macro filtering to avoid redundancies / overlaps with other rule families.
    try:
        allow = list(getattr(hp, "macro_allow_names", []) or [])
        deny = list(getattr(hp, "macro_deny_names", []) or [])
        disable_trig_dups = bool(getattr(hp, "macro_disable_trig_duplicates", True))
    except Exception:
        allow, deny, disable_trig_dups = [], [], False

    deny_set = {str(s).strip() for s in deny if str(s).strip()}
    if disable_trig_dups:
        # Only suppress trig-motif macros when Stage-B already detected trig structure
        # on at least one axis feeding this target leaf (the hint-driven template
        # families will cover these cases).
        try:
            trig_by_axis = getattr(ctx, 'trig_by_axis', {}) or {}
            has_trig_hint = False
            for j in getattr(target, 'var_idxs', ()) or ():
                if trig_by_axis.get(int(j)) is not None:
                    has_trig_hint = True
                    break
            if has_trig_hint:
                deny_set.update({'sinc', 'sinc_sq', 'sin_ratio', 'sin_ratio_sq'})
        except Exception:
            pass
    allow_set = {str(s).strip() for s in allow if str(s).strip()}
    if allow_set:
        macros = [m for m in macros if m.name in allow_set]
    if deny_set:
        macros = [m for m in macros if m.name not in deny_set]
    if not macros:
        return []

    # A single cached batch
    try:
        xb, _ = first_batch_xy(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
    except Exception:
        return []

    # Teacher output for the leaf (value-only; do not require leaf.grad / Hessians).
    try:
        atom_to_leaf = build_atom_to_leaf_map(st.root, st.model)
        leaf = atom_to_leaf.get(id(target), None)
        if leaf is None:
            return []
        is_compound = has_nontrivial_input(target)
        with torch.no_grad():
            x_in = _build_atom_input_tensor(target, xb)
            y = leaf(x_in)
            if y.dim() == 2:
                y = y[:, 0]
    except Exception:
        return []

    # LogDeriv probe: detect power-product compounds in the NN leaf.
    # Runs on all multivariate NN atoms uniformly (no compound gating).
    # Detected compounds are injected into the arg pool as extra expressions.
    logderiv_extras: list[ArgExpr] = []
    axes = [int(i) for i in getattr(target, "var_idxs", ()) or ()]
    if len(axes) >= 2:
        try:
            from nestynet_sr.sr_core.separability_math import check_monomial_compound_logderiv
            from nestynet_sr.sr_search.stageB.splits import _gather_nn_atom_value_grad_hess

            gdata = _gather_nn_atom_value_grad_hess(
                root=st.root, model=st.model, atom=target,
                train_loader=ctx.train_loader, device=ctx.device, dtype=ctx.dtype,
                max_points=2048,
            )
            if gdata is not None:
                X_loc, _X_raw, u_vals, du_vals, _H = gdata
                m = X_loc.shape[1]
                results, sigma_ratio, _b_perp = check_monomial_compound_logderiv(
                    var_idxs=tuple(range(m)),
                    x_vals=X_loc, y_vals=u_vals.ravel(), dydx_vals=du_vals,
                    max_exponent=3, precision=0.05,
                )
                if sigma_ratio is not None and sigma_ratio < 0.05 and results:
                    input_exprs = get_input_exprs(target)
                    for a_int, conf in results[:2]:
                        if conf < 0.55 or len(a_int) != len(input_exprs):
                            continue
                        # Build z = prod(input_expr[k]^a_int[k])
                        factors = []
                        for k, exp in enumerate(a_int):
                            if exp == 0:
                                continue
                            factors.append(Pow(clone_ast(input_exprs[k]), float(exp)))
                        if not factors:
                            continue
                        z_ast = factors[0]
                        for f in factors[1:]:
                            z_ast = Mul(z_ast, f)
                        desc_parts = [f"e{k}^{e}" for k, e in enumerate(a_int) if e != 0]
                        logderiv_extras.append(
                            ArgExpr(expr=z_ast, kind="compound",
                                    cost=0, desc=f"z_ld({','.join(desc_parts)})")
                        )
                        # Also z^2
                        logderiv_extras.append(
                            ArgExpr(expr=Pow(clone_ast(z_ast), 2.0),
                                    kind="compound_sq", cost=1,
                                    desc=f"z_ld({','.join(desc_parts)})^2")
                        )
                        # Also 1/z (reciprocal — needed because SVD direction sign is arbitrary)
                        logderiv_extras.append(
                            ArgExpr(expr=Pow(clone_ast(z_ast), -1.0),
                                    kind="compound", cost=0,
                                    desc=f"1/z_ld({','.join(desc_parts)})")
                        )
        except Exception:
            pass

    # If Stage-A exposed a compound variable z(x), keep a slightly richer pool so
    # that cost-2 cross terms (var*cos(z), ratio*cos(theta), 1-x^2, …) don't get
    # squeezed out by cheap scaled vars.
    max_args_eff = int(max_args)
    if is_compound and int(pool_min_args_for_compound) > max_args_eff:
        max_args_eff = int(pool_min_args_for_compound)

    use_pool_policy = bool(getattr(hp, 'macro_arg_use_wrapper_policy', True))
    if use_pool_policy:
        pol = macro_arg_wrapper_policy(ctx, hp, target)
        pool = build_arg_pool(
            target,
            max_vars=max_vars,
            max_args=max_args_eff,
            include_compound_expr=True,
            trig=bool(pol.trig),
            trig_squares=bool(pol.trig_squares),
            trig_max_bases=int(pol.trig_max_bases),
            extra_exprs=logderiv_extras or None,
        )
    else:
        pool = build_arg_pool(target, max_vars=max_vars, max_args=max_args_eff, include_compound_expr=True, extra_exprs=logderiv_extras or None)
    if not pool:
        return []

    hits: List[MacroHit] = []
    tier1_by_spec: Dict[str, List[MacroHit]] = {}

    # Screen macros with a cheap affine fit: y ≈ alpha*macro(args) + beta.
    for spec in macros:
        if spec.arity < 1 or spec.arity > 2:
            continue

        if spec.arity == 1:
            args0 = _select_args_for_pos(spec, pool, 0, max_pos_args)
            local_hits: List[MacroHit] = []
            for a0 in args0:
                try:
                    expr = _macro_expr_from(spec, (a0,))
                    if not _macro_units_ok(ctx, expr):
                        continue
                    with torch.no_grad():
                        m = eval_input_expr(expr, xb).view(-1)
                    ok = torch.isfinite(m)
                    ok_frac = float(ok.float().mean().item())
                    if ok_frac < ok_frac_min:
                        continue
                    m_ok = m[ok]
                    y_ok = y.view(-1)[ok]
                    alpha, beta = _affine_fit(m_ok, y_ok)
                    r2, mse = _score_affine(m_ok, y_ok, alpha, beta)
                    if not math.isfinite(r2):
                        continue
                    local_hits.append(
                        MacroHit(spec=spec, args=(a0,), score=float(r2), alpha=alpha, beta=beta, ok_frac=ok_frac, mse=mse)
                    )
                except Exception:
                    continue
            local_hits.sort(key=lambda h: h.score, reverse=True)
            keep = local_hits[:max_per_macro]
            hits.extend(keep)
            tier1_by_spec.setdefault(spec.name, []).extend(keep)

        if spec.arity == 2:
            args0 = _select_args_for_pos(spec, pool, 0, max_pos_args)
            args1 = _select_args_for_pos(spec, pool, 1, max_pos_args)
            local_hits = []
            for a0 in args0:
                for a1 in args1:
                    if spec.commutative and (ast_key(a1.expr) < ast_key(a0.expr)):
                        # Enforce an ordering for commutative macros.
                        continue
                    try:
                        expr = _macro_expr_from(spec, (a0, a1))
                        if not _macro_units_ok(ctx, expr):
                            continue
                        with torch.no_grad():
                            m = eval_input_expr(expr, xb).view(-1)
                        ok = torch.isfinite(m)
                        ok_frac = float(ok.float().mean().item())
                        if ok_frac < ok_frac_min:
                            continue
                        m_ok = m[ok]
                        y_ok = y.view(-1)[ok]
                        alpha, beta = _affine_fit(m_ok, y_ok)
                        r2, mse = _score_affine(m_ok, y_ok, alpha, beta)
                        if not math.isfinite(r2):
                            continue
                        local_hits.append(
                            MacroHit(spec=spec, args=(a0, a1), score=float(r2), alpha=alpha, beta=beta, ok_frac=ok_frac, mse=mse)
                        )
                    except Exception:
                        continue
            local_hits.sort(key=lambda h: h.score, reverse=True)
            keep = local_hits[:max_per_macro]
            hits.extend(keep)
            tier1_by_spec.setdefault(spec.name, []).extend(keep)

    if not hits:
        return []

    # ------------------------------------------------------------------
    # Optional: macro-as-factor pool (macro-macro multiplicative chaining)
    #
    # Many physics expressions are products of two motifs. Tier-2 already
    # multiplies a macro-core by a simple factor; here we allow a *few* macro-cores
    # to appear as factors too, but we keep it tightly bounded.
    macro_factor_pool: List[ArgExpr] = []
    if macro_factors_enable and int(macro_factors_max) > 0:
        seen_mf = set()
        for spec in macros:
            base_hits = tier1_by_spec.get(spec.name, [])
            for h0 in base_hits[: max(0, int(macro_factors_per_spec))]:
                expr = _macro_expr_from(spec, h0.args)
                k = ast_key(expr)
                if k in seen_mf:
                    continue
                seen_mf.add(k)
                macro_factor_pool.append(
                    ArgExpr(
                        expr=expr,
                        kind="macro",
                        cost=sum(int(a.cost) for a in h0.args) + 1,
                        desc=f"{spec.name}({', '.join(a.desc for a in h0.args)})",
                    )
                )
                if len(macro_factor_pool) >= int(macro_factors_max):
                    break
            if len(macro_factor_pool) >= int(macro_factors_max):
                break

    # Tier-2 expansion: multiply/divide the best tier-1 macro cores by a simple factor(x).
    if tier2_enable and (tier2_try_mul or tier2_try_div):
        try:
            fpool = build_factor_pool(
                pool,
                max_base_factors=tier2_max_base_factors,
                max_factor_exprs=tier2_max_factor_exprs,
            )
        except Exception:
            fpool = []

        if fpool:
            # Precompute factor values once on the cached batch.
            fvals: List[torch.Tensor] = []
            with torch.no_grad():
                for f in fpool:
                    try:
                        fv = eval_input_expr(f.expr, xb).view(-1)
                    except Exception:
                        fv = torch.full_like(y.view(-1), float("nan"))
                    fvals.append(fv)

                # Extend factor pool with a few screened macro-cores (macro-macro products).
                if macro_factors_enable and macro_factor_pool:
                    base_keys = {ast_key(ff.expr) for ff in fpool}
                    for f in macro_factor_pool:
                        k = ast_key(f.expr)
                        if k in base_keys:
                            continue
                        try:
                            fv = eval_input_expr(f.expr, xb).view(-1)
                        except Exception:
                            continue
                        ok_f = torch.isfinite(fv)
                        ok_frac = float(ok_f.float().mean().item())
                        if ok_frac < ok_frac_min:
                            continue
                        # torch has nanmean/nansum but (as of 2.x) no nanstd.
                        # Compute a finite-only std here to keep the rule robust.
                        fv_ok = fv[ok_f]
                        if int(fv_ok.numel()) < 2:
                            continue
                        try:
                            fv_std = float(fv_ok.std(unbiased=False).item())
                        except Exception:
                            fv_std = float("nan")
                        if (not math.isfinite(fv_std)) or (fv_std < 1e-12):
                            continue
                        fpool.append(f)
                        fvals.append(fv)
                        base_keys.add(k)

            tier2_hits: List[MacroHit] = []

            for spec in macros:
                base_hits = tier1_by_spec.get(spec.name, [])
                if not base_hits:
                    continue

                local2: List[MacroHit] = []
                # Seed from the best tier-1 hits for this macro.
                for h0 in base_hits[: max_per_macro]:
                    try:
                        macro_core = _macro_expr_from(h0.spec, h0.args)
                        with torch.no_grad():
                            m = eval_input_expr(macro_core, xb).view(-1)
                        ok_m = torch.isfinite(m)
                    except Exception:
                        continue

                    for f, fv in zip(fpool, fvals):
                        ok_f = torch.isfinite(fv)

                        if tier2_try_mul:
                            ok = ok_m & ok_f
                            ok_frac = float(ok.float().mean().item())
                            if ok_frac >= ok_frac_min:
                                try:
                                    t = (m[ok] * fv[ok])
                                    alpha, beta = _affine_fit(t, y.view(-1)[ok])
                                    r2, mse = _score_affine(t, y.view(-1)[ok], alpha, beta)
                                    if math.isfinite(r2):
                                        local2.append(
                                            MacroHit(
                                                spec=h0.spec,
                                                args=h0.args,
                                                score=float(r2),
                                                alpha=float(alpha),
                                                beta=float(beta),
                                                ok_frac=ok_frac,
                                                mse=float(mse),
                                                tier=2,
                                                outer_op="mul",
                                                outer_factor=f,
                                            )
                                        )
                                except Exception:
                                    pass

                        # Be conservative: allow macro-factors for multiplication only.
                        if tier2_try_div and f.kind != "macro":
                            # Avoid division by zero blowups during screening.
                            ok = ok_m & ok_f & (torch.abs(fv) > float(tier2_div_eps))
                            ok_frac = float(ok.float().mean().item())
                            if ok_frac >= ok_frac_min:
                                try:
                                    t = (m[ok] / fv[ok])
                                    alpha, beta = _affine_fit(t, y.view(-1)[ok])
                                    r2, mse = _score_affine(t, y.view(-1)[ok], alpha, beta)
                                    if math.isfinite(r2):
                                        local2.append(
                                            MacroHit(
                                                spec=h0.spec,
                                                args=h0.args,
                                                score=float(r2),
                                                alpha=float(alpha),
                                                beta=float(beta),
                                                ok_frac=ok_frac,
                                                mse=float(mse),
                                                tier=2,
                                                outer_op="div",
                                                outer_factor=f,
                                            )
                                        )
                                except Exception:
                                    pass

                local2.sort(key=lambda h: h.score, reverse=True)
                tier2_hits.extend(local2[:tier2_max_per_macro])

            if tier2_hits:
                hits.extend(tier2_hits)

    # Build tier-1/2 single-feature candidates (keep full hit list for tier-3).
    hits_all = sorted(hits, key=lambda h: h.score, reverse=True)
    hits_for_cands = list(hits_all[:max_hits])
    if noise_aware_macro_selection and clean_singleton_cap > 0:
        seen_hit_keys = {_macro_hit_identity(h) for h in hits_for_cands}
        protected_hits = [h for h in hits_all if _is_clean_singleton_macro_hit(h)]
        for h in protected_hits[:clean_singleton_cap]:
            key = _macro_hit_identity(h)
            if key in seen_hit_keys:
                continue
            seen_hit_keys.add(key)
            hits_for_cands.append(h)

    cands: List[Candidate] = []

    # ------------------------------------------------------------------
    # Tier-3b: shared-prefactor mode
    #   y ≈ p(x) * (a*macro(args) + b)
    # i.e. divide out a candidate prefactor p(x), fit an affine macro to (y/p),
    # then rebuild a multiplicative candidate in the original space.
    # This is extremely helpful for Feynman forms like
    #   y = prefactor(x) * (sinc(phase)^2)   or   y = prefactor(x) * (sqrt1p(u)*cos(Δ) + 1).
    # ------------------------------------------------------------------
    if sharedpref_enable or sharedpref_resid_enable:
        try:
            yv = y.view(-1)
        except Exception:
            yv = y

        yfin = torch.isfinite(yv)

        shared_hits = []
        shared_resid_hits = []
        try:
            fpool_p = build_factor_pool(
                pool,
                max_base_factors=max(4, int(tier2_max_base_factors)),
                max_factor_exprs=max(int(tier2_max_factor_exprs), int(sharedpref_max_prefactors) * 8),
            )
        except Exception:
            fpool_p = []

        if fpool_p:
            # Evaluate a cheap prefix of the prefactor pool for numerical stability.
            pref_eval = []
            pref_cands = fpool_p[: max(int(sharedpref_max_prefactors) * 8, int(sharedpref_max_prefactors))]
            # Optionally allow a few previously-screened macro-cores as prefactors
            # (enables macro-macro chained patterns with a shared prefactor).
            base_keys = {ast_key(ff.expr) for ff in pref_cands}
            if macro_factors_sharedpref and macro_factor_pool:
                for f in macro_factor_pool:
                    k = ast_key(f.expr)
                    if k in base_keys:
                        continue
                    pref_cands.append(f)
                    base_keys.add(k)

            # Seed a few square-complement prefactors (e.g. 1-x^2) which are
            # common in AIF forms but can be crowded out by cheaper candidates.
            seed_prefactors = _seed_square_prefactors(target)
            seed_keys = {ast_key(s.expr) for s in seed_prefactors}
            for f in seed_prefactors:
                k = ast_key(f.expr)
                if k in base_keys:
                    continue
                pref_cands.append(f)
                base_keys.add(k)

            with torch.no_grad():
                for f in pref_cands:
                    try:
                        fv = eval_input_expr(f.expr, xb).view(-1)
                    except Exception:
                        continue
                    ok_f = torch.isfinite(fv) & (torch.abs(fv) > float(sharedpref_div_eps))
                    ok_frac_f = float(ok_f.float().mean().item())
                    if ok_frac_f < ok_frac_min:
                        continue
                    try:
                        if float(fv[ok_f].std().item()) < 1.0e-12:
                            continue
                    except Exception:
                        pass
                    pref_eval.append((f, fv, ok_f, ok_frac_f))

            # Prefer simple, stable prefactors.
            pref_eval.sort(key=lambda t: (int(getattr(t[0], 'cost', 10**9)), -float(t[3]), len(str(getattr(t[0], 'desc', '')))))
            pref_main = pref_eval[: int(sharedpref_max_prefactors)]
            # Force-include the seeded square-prefactors if they survived numeric screening.
            if seed_keys:
                have = {ast_key(t[0].expr) for t in pref_main}
                for t in pref_eval:
                    k = ast_key(t[0].expr)
                    if (k in seed_keys) and (k not in have):
                        pref_main.append(t)
                        have.add(k)
            pref_eval = pref_main

            # Build a bounded macro-expression list once (independent of prefactor).
            macro_exprs = []
            seen_m = set()

            def _add_macro_expr(spec, args):
                try:
                    expr = _macro_expr_from(spec, args)
                    if not _macro_units_ok(ctx, expr):
                        return
                    key = ast_key(expr)
                    if key in seen_m:
                        return
                    with torch.no_grad():
                        mv = eval_input_expr(expr, xb).view(-1)
                    ok_m = torch.isfinite(mv)
                    ok_frac_m = float(ok_m.float().mean().item())
                    if ok_frac_m < ok_frac_min:
                        return
                    desc = f"{spec.name}({','.join(a.desc for a in args)})"
                    macro_exprs.append((expr, mv, ok_m, key, desc))
                    seen_m.add(key)
                except Exception:
                    return

            # Build the macro-expression shortlist in two passes to avoid letting
            # arity-2 macros consume the entire budget (which can starve inv1p/sqrt1m etc).

            # Pass 1: all arity-1 macros first.
            for spec in macros:
                if spec.arity != 1:
                    continue
                args0 = _select_args_for_pos(spec, pool, 0, int(sharedpref_max_pos_args))
                for a0 in args0:
                    _add_macro_expr(spec, (a0,))
                    if len(macro_exprs) >= int(sharedpref_max_macro_exprs):
                        break
                if len(macro_exprs) >= int(sharedpref_max_macro_exprs):
                    break

            # Pass 2: arity-2 macros with a smaller arg budget.
            if len(macro_exprs) < int(sharedpref_max_macro_exprs):
                max2 = int(min(int(sharedpref_max_pos_args), int(sharedpref_max_pos_args_arity2)))
                for spec in macros:
                    if spec.arity != 2:
                        continue
                    args0 = _select_args_for_pos(spec, pool, 0, max2)
                    args1 = _select_args_for_pos(spec, pool, 1, max2)
                    for a0 in args0:
                        for a1 in args1:
                            if spec.commutative and (ast_key(a1.expr) < ast_key(a0.expr)):
                                continue
                            _add_macro_expr(spec, (a0, a1))
                            if len(macro_exprs) >= int(sharedpref_max_macro_exprs):
                                break
                        if len(macro_exprs) >= int(sharedpref_max_macro_exprs):
                            break
                    if len(macro_exprs) >= int(sharedpref_max_macro_exprs):
                        break

            # Screen shared-prefactor candidates.
            for f, fv, ok_f, ok_frac_f in pref_eval:
                local = []
                local_resid = []
                ok_f_finite = torch.isfinite(fv) & yfin

                for expr, mv, ok_m, key_m, desc_m in macro_exprs:
                    # Shared-prefactor (no residual): y ≈ p*(a*m + b)
                    ok1 = ok_f & ok_m & yfin
                    ok_frac1 = float(ok1.float().mean().item())
                    if sharedpref_enable and ok_frac1 >= ok_frac_min:
                        try:
                            y_ok = yv[ok1]
                            f_ok = fv[ok1]
                            m_ok = mv[ok1]
                            y_div = y_ok / f_ok
                            alpha, beta = _affine_fit(m_ok, y_div)

                            # Score in original y space for comparability with other tiers.
                            pred_y = f_ok * (alpha * m_ok + beta)
                            resid = pred_y - y_ok
                            mse = float((resid * resid).mean().item())
                            var_y = float(((y_ok - float(y_ok.mean().item())) ** 2).mean().item())
                            if (math.isfinite(var_y)) and var_y >= 1.0e-18:
                                r2 = 1.0 - (mse / var_y)
                                if math.isfinite(float(r2)):
                                    local.append((float(r2), float(mse), float(alpha), float(beta), float(ok_frac1), f, expr, key_m, desc_m))
                        except Exception:
                            pass

                    # Shared-prefactor + residual constant: y ≈ p*(a*m + b) + c
                    if sharedpref_resid_enable:
                        ok2 = ok_f_finite & ok_m
                        ok_frac2 = float(ok2.float().mean().item())
                        if ok_frac2 < ok_frac_min:
                            continue
                        try:
                            y2 = yv[ok2]
                            p2 = fv[ok2]
                            m2 = mv[ok2]
                            phi1 = p2 * m2
                            phi2 = p2
                            A = torch.stack([phi1, phi2, torch.ones_like(phi2)], dim=1)  # [N,3]

                            AtA = A.T @ A
                            if float(sharedpref_resid_ridge) > 0.0:
                                AtA = AtA + float(sharedpref_resid_ridge) * torch.eye(AtA.shape[0], device=A.device, dtype=A.dtype)
                            AtY = A.T @ y2

                            cvec = torch.linalg.solve(AtA, AtY)  # [a,b,c]
                            a = float(cvec[0].item())
                            b = float(cvec[1].item())
                            c0 = float(cvec[2].item())

                            pred = A @ cvec
                            resid = pred - y2
                            mse = float((resid * resid).mean().item())
                            var_y = float(((y2 - float(y2.mean().item())) ** 2).mean().item())
                            if (math.isfinite(var_y)) and var_y >= 1.0e-18:
                                r2 = 1.0 - (mse / var_y)
                                if math.isfinite(float(r2)):
                                    local_resid.append((float(r2), float(mse), float(a), float(b), float(c0), float(ok_frac2), f, expr, key_m, desc_m))
                        except Exception:
                            continue

                if local:
                    local.sort(key=lambda t: t[0], reverse=True)
                    shared_hits.extend(local[: int(sharedpref_max_per_prefactor)])
                if local_resid:
                    local_resid.sort(key=lambda t: t[0], reverse=True)
                    shared_resid_hits.extend(local_resid[: int(sharedpref_resid_max_per_prefactor)])

        # Build up to a few candidates across all prefactors.
        if shared_hits:
            shared_hits.sort(key=lambda t: t[0], reverse=True)
            shared_hits = shared_hits[: int(sharedpref_max_candidates)]
            ttag = str(target.tag) if getattr(target, 'tag', None) is not None else 't'

            for kk, (r2, mse, alpha, beta, ok_frac, f, mexpr, key_m, desc_m) in enumerate(shared_hits):
                try:
                    a_tag = f"cfp_a_{ttag}_{kk}"
                    b_tag = f"cfp_b_{ttag}_{kk}"
                    inner = Add(
                        Scale(name=b_tag, tag=b_tag, init=float(beta)),
                        Mul(Scale(name=a_tag, tag=a_tag, init=float(alpha)), clone_ast(mexpr)),
                    )
                    subtree = Mul(clone_ast(f.expr), inner)
                    cand_root = replace_atom_in_ast(st.root, target, subtree)
                    if cand_root is None:
                        continue
                    sig = ("cfp", ast_key(f.expr), key_m)
                    log = (
                        f"[Stage B]  Trying shared-prefactor macro on NN leaf vars {list(getattr(target, 'var_idxs', ()))}, "
                        f"pref={getattr(f, 'desc', '?')}, macro={desc_m}, screen_r2={float(r2):.3f}, ok_frac={float(ok_frac):.3f}"
                    )
                    cands.append(
                        Candidate(
                            label="cf_sharedpref",
                            root=cand_root,
                            init_fn=None,
                            meta=_rich_macro_meta(
                                log=log,
                                screen_r2=float(r2),
                                ok_frac=float(ok_frac),
                                n_terms=1,
                                macro_name="sharedpref",
                                feature_keys=(ast_key(f.expr), key_m),
                            ),
                            signature=sig,
                        )
                    )
                except Exception:
                    continue

        if shared_resid_hits:
            shared_resid_hits.sort(key=lambda t: t[0], reverse=True)
            shared_resid_hits = shared_resid_hits[: int(sharedpref_resid_max_candidates)]
            ttag = str(target.tag) if getattr(target, 'tag', None) is not None else 't'

            for kk, (r2, mse, a0, b0, c0, ok_frac, f, mexpr, key_m, desc_m) in enumerate(shared_resid_hits):
                try:
                    a_tag = f"cfpr_a_{ttag}_{kk}"
                    b_tag = f"cfpr_b_{ttag}_{kk}"
                    c_tag = f"cfpr_c_{ttag}_{kk}"
                    inner = Add(
                        Scale(name=b_tag, tag=b_tag, init=float(b0)),
                        Mul(Scale(name=a_tag, tag=a_tag, init=float(a0)), clone_ast(mexpr)),
                    )
                    subtree = Add(
                        Scale(name=c_tag, tag=c_tag, init=float(c0)),
                        Mul(clone_ast(f.expr), inner),
                    )
                    cand_root = replace_atom_in_ast(st.root, target, subtree)
                    if cand_root is None:
                        continue
                    sig = ("cfpr", ast_key(f.expr), key_m)
                    log = (
                        f"[Stage B]  Trying shared-prefactor+residual macro on NN leaf vars {list(getattr(target, 'var_idxs', ()))}, "
                        f"pref={getattr(f, 'desc', '?')}, macro={desc_m}, screen_r2={float(r2):.3f}, ok_frac={float(ok_frac):.3f}"
                    )
                    cands.append(
                        Candidate(
                            label="cf_sharedpref_resid",
                            root=cand_root,
                            init_fn=None,
                            meta=_rich_macro_meta(
                                log=log,
                                screen_r2=float(r2),
                                ok_frac=float(ok_frac),
                                n_terms=1,
                                macro_name="sharedpref_resid",
                                feature_keys=(ast_key(f.expr), key_m),
                            ),
                            signature=sig,
                        )
                    )
                except Exception:
                    continue

    def _hit_feature_expr(h: MacroHit):
        expr = _macro_expr_from(h.spec, h.args)
        outer_desc = ""
        if getattr(h, "tier", 1) == 2 and getattr(h, "outer_op", None) is not None and getattr(h, "outer_factor", None) is not None:
            try:
                fexpr = clone_ast(h.outer_factor.expr)
            except Exception:
                fexpr = h.outer_factor.expr
            if str(h.outer_op) == "mul":
                expr = Mul(fexpr, expr)
                outer_desc = f" *({h.outer_factor.desc})"
            elif str(h.outer_op) == "div":
                expr = Mul(expr, Pow(fexpr, -1.0))
                outer_desc = f" /({h.outer_factor.desc})"
        desc = f"{h.spec.name}{outer_desc}"
        key = ast_key(expr)
        return expr, desc, key

    # --- Tier-1/2 candidates: y ≈ a*phi(x)+b --------------------------------
    for k, h in enumerate(hits_for_cands):
        try:
            phi, _desc, _key = _hit_feature_expr(h)

            a_tag = f"cf_{h.spec.name}_a_{target.tag}_{k}"
            b_tag = f"cf_{h.spec.name}_b_{target.tag}_{k}"
            a = Scale(name=a_tag, tag=a_tag, init=float(h.alpha))
            if h.spec.affine:
                b = Scale(name=b_tag, tag=b_tag, init=float(h.beta))
                new_subtree = Add(b, Mul(a, phi))
            else:
                new_subtree = Mul(a, phi)

            cand_root = replace_atom_in_ast(st.root, target, new_subtree)
            if cand_root is None:
                continue

            sig = (
                "cf",  # tier tag
                hash(h.spec.name),
                hash(tuple(ast_key(a.expr) for a in h.args)),
                hash(str(getattr(h, "outer_op", ""))),
                hash(ast_key(getattr(getattr(h, "outer_factor", None), "expr", ConstNode(0.0)))),
            )

            tier = int(getattr(h, "tier", 1))
            if tier == 2 and getattr(h, "outer_op", None) is not None and getattr(h, "outer_factor", None) is not None:
                outer_desc = f" {h.outer_op}({h.outer_factor.desc})"
            else:
                outer_desc = ""

            log = (
                f"[Stage B]  Trying macro {h.spec.name}{outer_desc} on NN leaf vars {list(getattr(target, 'var_idxs', ()))}, "
                f"args={[a.desc for a in h.args]}, screen_r2={h.score:.3f}, ok_frac={h.ok_frac:.3f}"
            )
            cands.append(
                Candidate(
                    label=f"cf_{h.spec.name}",
                    root=cand_root,
                    init_fn=None,
                    meta=_macro_hit_meta(h, log=log, feature_key=_key),
                    signature=sig,
                )
            )

            # Non-affine fallback: emit a*phi without the offset b.
            # When the atom output is dimensionful, Add(Scale_b, Mul(Scale_a, phi))
            # fails the units precheck because Scale is forced dimensionless.
            # The non-affine version Mul(a, phi) avoids that constraint.
            if h.spec.affine:
                a2_tag = f"cf_{h.spec.name}_a_{target.tag}_{k}_noa"
                a2 = Scale(name=a2_tag, tag=a2_tag, init=float(h.alpha))
                noa_subtree = Mul(a2, clone_ast(phi))
                noa_root = replace_atom_in_ast(st.root, target, noa_subtree)
                if noa_root is not None:
                    noa_sig = ("cf_noa",) + sig[1:]
                    cands.append(
                        Candidate(
                            label=f"cf_{h.spec.name}",
                            root=noa_root,
                            init_fn=None,
                            meta=_macro_hit_meta(h, log=log, feature_key=_key, no_intercept=True),
                            signature=noa_sig,
                        )
                    )
        except Exception:
            continue

    # --- Tier-3: small additive linear combos (2 or 3 features + intercept) ---
    if tier3_enable and (tier3_try_pairs or tier3_try_triples) and hits_all:
        # Feature pools
        macro_feats = []
        seen_feat = set()

        yv = y.view(-1)

        def _eval_feature(expr: Node):
            with torch.no_grad():
                v = eval_input_expr(expr, xb).view(-1)
            ok = torch.isfinite(v)
            ok_frac = float(ok.float().mean().item())
            return v, ok, ok_frac

        def _feature_r2(v: torch.Tensor, ok: torch.Tensor):
            try:
                v_ok = v[ok]
                y_ok = yv[ok]
                alpha0, beta0 = _affine_fit(v_ok, y_ok)
                r2, _mse = _score_affine(v_ok, y_ok, alpha0, beta0)
                return float(r2)
            except Exception:
                return -float("inf")

        # 1) Macro-derived features (from tier-1/2 hits)
        for h in hits_all:
            try:
                expr, desc, key = _hit_feature_expr(h)
                if key in seen_feat:
                    continue
                v, ok, ok_frac = _eval_feature(expr)
                if ok_frac < ok_frac_min:
                    continue
                r2 = _feature_r2(v, ok)
                if not math.isfinite(r2):
                    continue
                macro_feats.append({"expr": expr, "desc": desc, "key": key, "v": v, "ok": ok, "r2": float(r2)})
                seen_feat.add(key)
                if len(macro_feats) >= int(tier3_max_macro_features):
                    break
            except Exception:
                continue

        # 2) Auxiliary features (monomial/ratio/trig prefactors)
        aux_feats = []
        if macro_feats or tier3_allow_aux_aux:
            try:
                # Reuse the tier-2 factor pool builder; it already dedupes and cost-sorts.
                fpool3 = build_factor_pool(
                    pool,
                    max_base_factors=tier2_max_base_factors,
                    max_factor_exprs=max(int(tier2_max_factor_exprs), int(tier3_max_aux_features) * 2),
                )
            except Exception:
                fpool3 = []

            # Evaluate a slightly larger cheap prefix, then keep the best by single-feature R^2.
            fpool3 = fpool3[: max(0, int(tier3_max_aux_features) * 3)]
            for f in fpool3:
                try:
                    key = ast_key(f.expr)
                    if key in seen_feat:
                        continue
                    v, ok, ok_frac = _eval_feature(f.expr)
                    if ok_frac < ok_frac_min:
                        continue
                    r2 = _feature_r2(v, ok)
                    if not math.isfinite(r2):
                        continue
                    aux_feats.append({"expr": f.expr, "desc": f.desc, "key": key, "v": v, "ok": ok, "r2": float(r2)})
                    seen_feat.add(key)
                except Exception:
                    continue
            aux_feats.sort(key=lambda z: z["r2"], reverse=True)
            aux_feats = aux_feats[: int(tier3_max_aux_features)]

        def _fit_lin_combo(feats, ok_mask):
            # feats: list of feature dicts, ok_mask already includes finiteness of all
            n_ok = int(ok_mask.sum().item())
            if n_ok < max(8, 2 * len(feats)):
                return None
            yy = yv[ok_mask]
            cols = [f["v"][ok_mask] for f in feats]
            cols.append(torch.ones_like(yy))  # intercept
            A = torch.stack(cols, dim=1)  # [N, k+1]
            AtA = A.T @ A
            if float(tier3_ridge) > 0.0:
                AtA = AtA + float(tier3_ridge) * torch.eye(AtA.shape[0], device=A.device, dtype=A.dtype)
            AtY = A.T @ yy
            try:
                c = torch.linalg.solve(AtA, AtY)
            except Exception:
                return None
            pred = A @ c
            resid = pred - yy
            mse = float((resid * resid).mean().item())
            var_y = float(((yy - float(yy.mean().item())) ** 2).mean().item())
            if not math.isfinite(var_y) or var_y < 1e-18:
                return None
            r2 = 1.0 - (mse / var_y)
            if not math.isfinite(float(r2)):
                return None
            coeffs = [float(ci.item()) for ci in c[:-1]]
            beta = float(c[-1].item())
            return coeffs, beta, float(r2), mse

        def _combo_ok_mask(feats):
            ok = None
            for f in feats:
                if ok is None:
                    ok = f["ok"].clone()
                else:
                    ok &= f["ok"]
            return ok

        def _reduce_combo_by_subset(feats, ok_ref, r2_full):
            """Prefer the smallest strict subset whose screening r2 matches the full combo."""
            try:
                import itertools
            except Exception:
                return feats, None

            if len(feats) <= 1:
                return feats, None
            tol = float(tier3_subset_r2_tol)
            if tol <= 0.0:
                return feats, None

            k = len(feats)
            for size in range(1, k):
                best = None
                best_fit = None
                for idxs in itertools.combinations(range(k), size):
                    sub = [feats[i] for i in idxs]
                    fit = _fit_lin_combo(sub, ok_ref)
                    if fit is None:
                        continue
                    coeffs_s, beta_s, r2_s, mse_s = fit
                    if not math.isfinite(float(r2_s)):
                        continue
                    if float(r2_s) >= float(r2_full) - tol:
                        if best is None or float(r2_s) > float(best_fit[2]):
                            best = sub
                            best_fit = (coeffs_s, beta_s, r2_s, mse_s)
                if best is not None:
                    return best, best_fit
            return feats, None

        def _prune_lin_combo_terms(feats, coeffs, beta, ok_ref):
            """Drop terms whose contribution is negligible vs the strongest term."""
            if not feats or not coeffs:
                return feats, coeffs, beta, False
            if ok_ref is None or int(ok_ref.sum().item()) < max(8, 2 * len(feats)):
                return feats, coeffs, beta, False

            # Contribution proxy: |c| * RMS(feature)
            contrib = []
            for f, c in zip(feats, coeffs):
                try:
                    v_ok = f["v"][ok_ref]
                    if v_ok.numel() == 0:
                        rms = 0.0
                    else:
                        rms = float(torch.sqrt((v_ok * v_ok).mean()).item())
                except Exception:
                    rms = 0.0
                contrib.append(abs(float(c)) * float(rms))

            m = max(contrib) if contrib else 0.0
            if not math.isfinite(float(m)) or float(m) <= 0.0:
                return feats, coeffs, beta, False

            rel = float(tier3_prune_rel_contrib)
            if rel <= 0.0:
                return feats, coeffs, beta, False
            tol = rel * float(m)

            keep = [i for i, cc in enumerate(contrib) if float(cc) > tol]
            if not keep:
                return feats, coeffs, beta, False

            pruned = len(keep) != len(feats)
            if pruned:
                feats = [feats[i] for i in keep]
                coeffs = [coeffs[i] for i in keep]

            beta2 = 0.0 if abs(float(beta)) <= tol else float(beta)
            if beta2 != float(beta):
                pruned = True

            return feats, coeffs, beta2, pruned

        combo_hits = []
        seen_combo = set()

        # Pair combos
        if tier3_try_pairs:
            # macro + aux
            for mf in macro_feats:
                for af in aux_feats:
                    if mf["key"] == af["key"]:
                        continue
                    combo_key = ("pair", tuple(sorted([mf["key"], af["key"]])))
                    if combo_key in seen_combo:
                        continue
                    ok = mf["ok"] & af["ok"]
                    ok_frac = float(ok.float().mean().item())
                    if ok_frac < ok_frac_min:
                        continue
                    fit = _fit_lin_combo([mf, af], ok)
                    if fit is None:
                        continue
                    coeffs, beta, r2, mse = fit
                    combo_hits.append({"feats": [mf, af], "coeffs": coeffs, "beta": beta, "r2": r2, "mse": mse, "ok_frac": ok_frac})
                    seen_combo.add(combo_key)

            # macro + macro
            if tier3_allow_macro_macro and len(macro_feats) >= 2:
                for i in range(len(macro_feats)):
                    for j in range(i + 1, len(macro_feats)):
                        a, b = macro_feats[i], macro_feats[j]
                        if a["key"] == b["key"]:
                            continue
                        combo_key = ("pair", tuple(sorted([a["key"], b["key"]])))
                        if combo_key in seen_combo:
                            continue
                        ok = a["ok"] & b["ok"]
                        ok_frac = float(ok.float().mean().item())
                        if ok_frac < ok_frac_min:
                            continue
                        fit = _fit_lin_combo([a, b], ok)
                        if fit is None:
                            continue
                        coeffs, beta, r2, mse = fit
                        combo_hits.append({"feats": [a, b], "coeffs": coeffs, "beta": beta, "r2": r2, "mse": mse, "ok_frac": ok_frac})
                        seen_combo.add(combo_key)

            # aux + aux (optional)
            if tier3_allow_aux_aux and len(aux_feats) >= 2:
                for i in range(len(aux_feats)):
                    for j in range(i + 1, len(aux_feats)):
                        a, b = aux_feats[i], aux_feats[j]
                        combo_key = ("pair", tuple(sorted([a["key"], b["key"]])))
                        if combo_key in seen_combo:
                            continue
                        ok = a["ok"] & b["ok"]
                        ok_frac = float(ok.float().mean().item())
                        if ok_frac < ok_frac_min:
                            continue
                        fit = _fit_lin_combo([a, b], ok)
                        if fit is None:
                            continue
                        coeffs, beta, r2, mse = fit
                        combo_hits.append({"feats": [a, b], "coeffs": coeffs, "beta": beta, "r2": r2, "mse": mse, "ok_frac": ok_frac})
                        seen_combo.add(combo_key)

        # Triple combos (macro + aux + aux), optional
        if tier3_try_triples and macro_feats and len(aux_feats) >= 2:
            aux_small = aux_feats[: min(len(aux_feats), 12)]
            for mf in macro_feats:
                for i in range(len(aux_small)):
                    for j in range(i + 1, len(aux_small)):
                        a, b = aux_small[i], aux_small[j]
                        combo_key = ("triple", tuple(sorted([mf["key"], a["key"], b["key"]])))
                        if combo_key in seen_combo:
                            continue
                        ok = mf["ok"] & a["ok"] & b["ok"]
                        ok_frac = float(ok.float().mean().item())
                        if ok_frac < ok_frac_min:
                            continue
                        fit = _fit_lin_combo([mf, a, b], ok)
                        if fit is None:
                            continue
                        coeffs, beta, r2, mse = fit
                        combo_hits.append({"feats": [mf, a, b], "coeffs": coeffs, "beta": beta, "r2": r2, "mse": mse, "ok_frac": ok_frac})
                        seen_combo.add(combo_key)

        combo_hits.sort(key=lambda h: (h["r2"], -len(h.get("feats", ()))), reverse=True)
        combo_hits = combo_hits[: int(tier3_max_candidates)]

        for kk, ch in enumerate(combo_hits):
            try:
                feats0 = list(ch["feats"])
                coeffs0 = list(ch["coeffs"])
                beta0 = float(ch["beta"])
                r2_0 = float(ch["r2"])
                ok_ref0 = _combo_ok_mask(feats0)
                if ok_ref0 is None:
                    continue

                feats = feats0
                coeffs = coeffs0
                beta = beta0
                screen_r2 = r2_0

                notes = []

                # Prefer a strict sub-combo if it matches the full screening fit.
                reduced_feats, best_fit = _reduce_combo_by_subset(feats, ok_ref0, screen_r2)
                if best_fit is not None and reduced_feats:
                    feats = reduced_feats
                    coeffs, beta, screen_r2, _mse = best_fit
                    if len(feats) != len(feats0):
                        notes.append(f"reduced={len(feats0)}->{len(feats)}")

                ok_ref = _combo_ok_mask(feats)
                if ok_ref is None:
                    continue

                # Prune negligible-contribution terms (and an almost-zero intercept), then refit.
                feats_p, coeffs_p, beta_p, did_prune = _prune_lin_combo_terms(feats, coeffs, beta, ok_ref)
                if did_prune:
                    if len(feats_p) != len(feats):
                        notes.append(f"pruned={len(feats)}->{len(feats_p)}")
                    if beta_p == 0.0 and beta != 0.0:
                        notes.append("pruned_intercept")
                    feats, coeffs, beta = feats_p, coeffs_p, beta_p
                    ok_ref = _combo_ok_mask(feats)
                    fit = _fit_lin_combo(feats, ok_ref)
                    if fit is not None:
                        coeffs, beta, screen_r2, _mse = fit
                        feats2, coeffs2, beta2, did2 = _prune_lin_combo_terms(feats, coeffs, beta, ok_ref)
                        if did2:
                            if len(feats2) != len(feats):
                                notes.append(f"pruned={len(feats)}->{len(feats2)}")
                            if beta2 == 0.0 and beta != 0.0:
                                notes.append("pruned_intercept")
                            feats, coeffs, beta = feats2, coeffs2, beta2
                            ok_ref = _combo_ok_mask(feats)
                            fit2 = _fit_lin_combo(feats, ok_ref)
                            if fit2 is not None:
                                coeffs, beta, screen_r2, _mse = fit2

                if not feats:
                    continue
                ok_frac = float(ok_ref.float().mean().item())

                # Build subtree
                subtree = None
                if beta != 0.0:
                    b_tag = f"cf3_b_{target.tag}_{kk}"
                    subtree = Scale(name=b_tag, tag=b_tag, init=float(beta))

                keys = []
                descs = []
                for ii, (f, c0) in enumerate(zip(feats, coeffs)):
                    a_tag = f"cf3_a{ii}_{target.tag}_{kk}"
                    a = Scale(name=a_tag, tag=a_tag, init=float(c0))
                    term = Mul(a, clone_ast(f["expr"]))
                    subtree = term if subtree is None else Add(subtree, term)
                    keys.append(f["key"])
                    descs.append(f["desc"])

                if subtree is None:
                    continue

                cand_root = replace_atom_in_ast(st.root, target, subtree)
                if cand_root is None:
                    continue

                sorted_keys = tuple(sorted(keys))
                sig = ("cf3", len(feats), sorted_keys)
                extra = f" ({', '.join(notes)})" if notes else ""
                log = (
                    f"[Stage B]  Trying tier-3 linear combo on NN leaf vars {list(getattr(target, 'var_idxs', ()))}, "
                    f"terms={descs}, screen_r2={float(screen_r2):.3f}, ok_frac={float(ok_frac):.3f}{extra}"
                )
                cands.append(
                    Candidate(
                        label="cf_combo",
                        root=cand_root,
                        init_fn=None,
                        meta=_rich_macro_meta(
                            log=log,
                            screen_r2=float(screen_r2),
                            ok_frac=float(ok_frac),
                            n_terms=int(len(feats)),
                            macro_name="combo",
                            feature_keys=keys,
                        ),
                        signature=sig,
                    )
                )

                # No-intercept fallback: omit the Scale offset b.
                # When features are dimensionful, Add(Scale_dimless, dimensionful)
                # fails the units precheck.  The no-intercept version avoids that.
                if beta != 0.0:
                    noa_subtree = None
                    for ii, (f, c0) in enumerate(zip(feats, coeffs)):
                        a_tag2 = f"cf3_a{ii}_{target.tag}_{kk}_noa"
                        a2 = Scale(name=a_tag2, tag=a_tag2, init=float(c0))
                        term2 = Mul(a2, clone_ast(f["expr"]))
                        noa_subtree = term2 if noa_subtree is None else Add(noa_subtree, term2)
                    if noa_subtree is not None:
                        noa_root = replace_atom_in_ast(st.root, target, noa_subtree)
                        if noa_root is not None:
                            noa_sig = ("cf3_noa", len(feats), sorted_keys)
                            cands.append(
                                Candidate(
                                    label="cf_combo",
                                    root=noa_root,
                                    init_fn=None,
                                    meta=_rich_macro_meta(
                                        log=log,
                                        screen_r2=float(screen_r2),
                                        ok_frac=float(ok_frac),
                                        n_terms=int(len(feats)),
                                        macro_name="combo",
                                        feature_keys=keys,
                                        extra={"macro_no_intercept": True},
                                    ),
                                    signature=noa_sig,
                                )
                            )
            except Exception:
                continue

    # Final selection across tiers.  In noisy mode, protect a small clean
    # singleton lane so overcomplete tier-3 dictionaries cannot crowd out the
    # physically simpler one-function motifs before Stage B/CoE can compare them.
    def _screen_key(c: Candidate) -> Tuple[float, int]:
        meta = c.meta if isinstance(getattr(c, "meta", None), dict) else {}
        return (
            float(meta.get("screen_r2", -1.0e30)),
            -int(meta.get("n_terms", 10**9)),
        )

    if noise_aware_macro_selection and clean_singleton_cap > 0:
        protected = [
            c
            for c in cands
            if bool((c.meta or {}).get("macro_clean_singleton", False))
        ]
        ordinary = [
            c
            for c in cands
            if not bool((c.meta or {}).get("macro_clean_singleton", False))
        ]

        def _clean_key(c: Candidate) -> Tuple[int, float, int]:
            meta = c.meta if isinstance(getattr(c, "meta", None), dict) else {}
            # Prefer the affine singleton before its no-intercept fallback.
            return (
                1 if bool(meta.get("macro_no_intercept", False)) else 0,
                -float(meta.get("screen_r2", -1.0e30)),
                int(meta.get("n_terms", 99)),
            )

        protected.sort(key=_clean_key)
        ordinary.sort(key=_screen_key, reverse=True)
        return protected[:clean_singleton_cap] + ordinary[:max_hits]

    cands.sort(key=_screen_key, reverse=True)
    return cands[:max_hits]
