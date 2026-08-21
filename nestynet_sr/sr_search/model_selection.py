# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Shared model-selection utilities.

The SR pipeline contains multiple decision points where we compare a current
model against candidate rewrites/simplifications. Historically, these decision
points evolved independently, leading to inconsistent heuristics and
parameters.

This module centralises the *core policy primitives*:

  - a cheap, NN-focused structural complexity proxy
  - a smooth "loss-budget" rule that converts simplification into an allowed
    regression in validation loss (in log10 decades)

The intent is **not** to impose an AIC/BIC-style strong parameter penalty.
Instead we (a) define a meaningful loss floor, below which loss differences are
treated as noise, and (b) allow limited loss regressions when they buy
simplification.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AbsNode,
    AddNode,
    AsinNode,
    ArgNode,
    AtanNode,
    AtomNode,
    ConjNode,
    ConstNode,
    CosNode,
    ExpNode,
    ImagNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    RealNode,
    SinNode,
    collect_nn_atoms,
    effective_arity,
    get_input_exprs,
    has_nontrivial_input,
)

NNComplexity = Tuple[int, int, int]  # (n_multivar, sum_squared_arities, max_arity)

# An unresolved neural atom is not just another analytic operator: it is a
# placeholder for an unsolved subproblem.  Below the meaningful loss floor, any
# visible analytic closure should therefore outrank a state that still contains
# NN leaves, even when the printed analytic expression is somewhat longer.
NN_ANALYTIC_DOMINATION_COST = 1.0e6


def _clamp_nonnegative_finite(value, default: float = 0.0) -> float:
    """Parse float; return default if non-finite or negative."""
    try:
        out = float(value)
    except Exception:
        return float(default)
    if (not math.isfinite(out)) or out < 0.0:
        return float(default)
    return float(out)


def resolve_acceptance_noise_floor_raw(lm_hp, loss_scale: float) -> float:
    """Resolve externally supplied acceptance noise floor in raw loss units."""
    try:
        raw = float(getattr(lm_hp, "acceptance_noise_floor_raw", None))
    except Exception:
        raw = None
    if raw is not None and math.isfinite(raw) and raw >= 0.0:
        return float(raw)

    try:
        base = float(getattr(lm_hp, "acceptance_noise_floor", None))
    except Exception:
        return 0.0
    if (not math.isfinite(base)) or base < 0.0:
        return 0.0

    if bool(getattr(lm_hp, "loss_in_MAD_units", False)):
        scale = _clamp_nonnegative_finite(loss_scale, default=1.0)
        return float(base * scale)
    return float(base)


def clamp_threshold_to_noise_floor(
    value,
    noise_floor,
    *,
    min_factor: float = 1.0,
) -> float:
    """Clamp a raw threshold so it is not set below a multiple of the loss floor."""
    thr = _clamp_nonnegative_finite(value, default=0.0)
    nf = _clamp_nonnegative_finite(noise_floor, default=0.0)
    fac = _clamp_nonnegative_finite(min_factor, default=1.0)
    return float(max(thr, fac * nf))


def noise_equivalence_tolerance(
    loss_a,
    loss_b=None,
    *,
    noise_floor: Optional[float] = None,
    n_eff: Optional[float] = None,
    noise_mult: float = 1.0,
    rel_tol: float = 1.0e-3,
    abs_floor: float = 0.0,
) -> float:
    """Return a raw-loss tolerance for deciding whether two losses are tied.

    ``noise_floor`` is an irreducible MSE-like floor in the same raw units as
    the losses.  When ``n_eff`` is not supplied, this preserves the historical
    scalar-floor policy and treats differences below ``noise_mult*noise_floor``
    as ties.

    When ``n_eff`` is supplied, the noise-floor tolerance is scaled by the
    validation uncertainty of a mean chi-square statistic, ``sqrt(2/n_eff)``.
    This avoids accepting statistically significant regressions merely because
    they are smaller than one full noise floor.
    """

    nf = _clamp_nonnegative_finite(noise_floor, default=0.0)
    mult = _clamp_nonnegative_finite(noise_mult, default=1.0)
    rel = _clamp_nonnegative_finite(rel_tol, default=0.0)
    abs_tol = _clamp_nonnegative_finite(abs_floor, default=0.0)
    n_eff_f: Optional[float] = None
    try:
        n_eff_f = float(n_eff) if n_eff is not None else None
    except Exception:
        n_eff_f = None
    if n_eff_f is not None and (not math.isfinite(n_eff_f) or n_eff_f <= 0.0):
        n_eff_f = None

    refs = []
    for value in (loss_a, loss_b):
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            refs.append(abs(v))
    ref = max(refs) if refs else 0.0
    noise_tol = mult * nf
    if nf > 0.0 and n_eff_f is not None:
        noise_tol *= math.sqrt(2.0 / n_eff_f)
    return float(max(abs_tol, noise_tol, rel * max(ref, nf, 1.0e-300)))


def noise_equivalent(
    loss_a,
    loss_b,
    *,
    noise_floor: Optional[float] = None,
    n_eff: Optional[float] = None,
    noise_mult: float = 1.0,
    rel_tol: float = 1.0e-3,
    abs_floor: float = 0.0,
) -> bool:
    """Return True when two raw losses are equivalent for noisy selection."""

    try:
        a = float(loss_a)
        b = float(loss_b)
    except Exception:
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    tol = noise_equivalence_tolerance(
        a,
        b,
        noise_floor=noise_floor,
        n_eff=n_eff,
        noise_mult=noise_mult,
        rel_tol=rel_tol,
        abs_floor=abs_floor,
    )
    return bool(abs(a - b) <= tol)


def loss_within_floor_or_noise_equivalent(
    loss,
    floor,
    *,
    noise_floor: Optional[float] = None,
    n_eff: Optional[float] = None,
    noise_mult: float = 1.0,
    rel_tol: float = 1.0e-3,
    abs_floor: float = 0.0,
) -> bool:
    """Return True when ``loss`` is below a floor up to noisy sampling error.

    The historical noiseless predicate was ``loss <= floor``.  With an
    irreducible MSE floor we compare in *excess above the noise floor* first,
    then allow a narrow chi-square sampling band around the boundary
    ``noise_floor + floor``.  With no noise floor this intentionally reduces
    to the old comparison.
    """

    try:
        loss_f = float(loss)
        floor_f = float(floor)
    except Exception:
        return False
    if (not math.isfinite(loss_f)) or (not math.isfinite(floor_f)) or floor_f < 0.0:
        return False

    nf = _clamp_nonnegative_finite(noise_floor, default=0.0)
    excess = loss_excess_above_floor(loss_f, nf)
    if excess is None:
        return False
    if float(excess) <= floor_f:
        return True
    if nf <= 0.0:
        return False
    boundary = nf + floor_f
    return noise_equivalent(
        loss_f,
        boundary,
        noise_floor=nf,
        n_eff=n_eff,
        noise_mult=noise_mult,
        rel_tol=rel_tol,
        abs_floor=abs_floor,
    )


def noisy_rel_rms_threshold(
    base_rel,
    *,
    noise_floor: Optional[float] = None,
    y_rms: Optional[float] = None,
    noise_mult: float = 2.0,
    cap: Optional[float] = None,
) -> float:
    """Loosen a relative-RMS pre-screen just enough for known noisy data.

    Rule-local screens compare ``rms_residual / rms_signal`` before the normal
    LM/global validation path.  For noisy benchmark data, a correct structural
    candidate can have irreducible relative RMS around ``sigma / rms_signal``.
    This helper leaves noiseless behavior unchanged and only raises the screen
    threshold when an explicit raw MSE noise floor is available.
    """

    base = _clamp_nonnegative_finite(base_rel, default=0.0)
    nf = _clamp_nonnegative_finite(noise_floor, default=0.0)
    scale = _clamp_nonnegative_finite(y_rms, default=0.0)
    mult = _clamp_nonnegative_finite(noise_mult, default=2.0)
    out = float(base)
    if nf > 0.0 and scale > 0.0:
        out = max(out, mult * math.sqrt(nf) / max(scale, 1.0e-300))
    if cap is not None:
        cap_f = _clamp_nonnegative_finite(cap, default=out)
        if cap_f > 0.0:
            out = min(out, cap_f)
    return float(out)


def apply_noise_floor_to_acceptance_thresholds(
    *,
    loss_target_raw,
    loss_acceptable_raw,
    accept_threshold_raw,
    noise_floor_raw,
    target_factor: float = 0.5,
    acceptable_factor: float = 3.0,
    candidate_factor: float = 2.0,
) -> Tuple[float, float, float]:
    """Clamp the main Stage-A/Stage-B thresholds against a raw loss floor."""
    return (
        clamp_threshold_to_noise_floor(
            loss_target_raw,
            noise_floor_raw,
            min_factor=target_factor,
        ),
        clamp_threshold_to_noise_floor(
            loss_acceptable_raw,
            noise_floor_raw,
            min_factor=acceptable_factor,
        ),
        clamp_threshold_to_noise_floor(
            accept_threshold_raw,
            noise_floor_raw,
            min_factor=candidate_factor,
        ),
    )


def estimate_transform_noise_floor_raw(
    y_raw,
    y_transform,
    sigma_y,
    *,
    fit_link=None,
    fit_link_scale: float = 1.0,
    n_mc: int = 8,
    seed: int = 12345,
    min_valid_frac: float = 0.5,
) -> Optional[float]:
    """Estimate the irreducible raw loss floor in the current fit space.

    ``y_transform`` is the active outer y-transform ``phi(y)`` and ``fit_link``
    is any LM-only residual-space transform ``t(phi(y))``.
    """
    sigma = _clamp_nonnegative_finite(sigma_y, default=0.0)
    if sigma <= 0.0:
        return None

    try:
        import numpy as np
        import torch
        from nestynet_sr.sr_core.fit_links import fit_link_torch
    except Exception:
        return None

    try:
        y = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if y.size <= 0:
        return None

    finite_y = np.isfinite(y)
    if not np.any(finite_y):
        return None
    y = y[finite_y]

    def _apply_y_transform(arr):
        if y_transform is None:
            return np.asarray(arr, dtype=np.float64).reshape(-1)
        out = y_transform(arr)
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def _apply_fit_link(arr):
        t = torch.as_tensor(arr, dtype=torch.float64)
        out = fit_link_torch(t, fit_link, scale=float(fit_link_scale))
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float64).reshape(-1)

    try:
        base = _apply_y_transform(y)
    except Exception:
        return None
    if base.shape != y.shape:
        return None

    try:
        base_fit = _apply_fit_link(base)
    except Exception:
        return None
    if base_fit.shape != y.shape:
        return None

    good_base = np.isfinite(base) & np.isfinite(base_fit)
    n_base = int(np.count_nonzero(good_base))
    if n_base <= 0:
        return None

    frac_req = min(max(float(min_valid_frac), 0.0), 1.0)
    need = max(1, int(math.ceil(frac_req * float(n_base))))
    rng = np.random.default_rng(int(seed))
    vals = []

    for _ in range(max(1, int(n_mc))):
        y_pert = y + rng.normal(0.0, sigma, size=y.shape)
        try:
            pert = _apply_y_transform(y_pert)
            pert_fit = _apply_fit_link(pert)
        except Exception:
            continue
        if pert.shape != y.shape or pert_fit.shape != y.shape:
            continue
        good = good_base & np.isfinite(pert) & np.isfinite(pert_fit)
        if int(np.count_nonzero(good)) < need:
            continue
        delta = np.asarray(pert_fit[good] - base_fit[good], dtype=np.float64)
        vals.append(float(np.mean(delta * delta)))

    if not vals:
        return None
    out = float(np.median(vals))
    if (not math.isfinite(out)) or out < 0.0:
        return None
    return float(out)


def nn_multivar_complexity(root: Node) -> NNComplexity:
    """Return a compact structural complexity proxy for NN leaves.

    Only NN atoms are considered. Univariate atoms (effective arity == 1) are
    ignored.
    """
    n_multivar = 0
    sum_squared_arities = 0
    max_arity = 0
    for atom in collect_nn_atoms(root):
        try:
            ar = int(effective_arity(atom))
        except Exception:
            ar = int(len(getattr(atom, "var_idxs", ()) or ()))
        if ar <= 1:
            continue
        n_multivar += 1
        sum_squared_arities += ar * ar
        if ar > max_arity:
            max_arity = ar
    return int(n_multivar), int(sum_squared_arities), int(max_arity)


def nn_structural_score(root: Node, *, count_weight: float = 1.0) -> float:
    """Scalar structural score used for smooth loss-budget scaling."""
    n_multivar, ar_sq, _ = nn_multivar_complexity(root)
    return float(ar_sq) + float(count_weight) * float(n_multivar)


def ast_cost_physics_prior(root: Node) -> float:
    """Weighted AST complexity with an explicit physics/textbook prior."""

    def _near(a: float, b: float, tol: float = 1e-12) -> bool:
        return abs(a - b) <= tol

    def _atom_kind_cost(kind: object, kwargs: object) -> float:
        k = str(kind).strip().lower()
        kw = kwargs if isinstance(kwargs, dict) else {}

        def _override_terms_maxdeg(
            exp_key: str,
            support_key: str,
        ) -> Optional[Tuple[int, int]]:
            rows = kw.get(exp_key, None)
            if not isinstance(rows, (list, tuple)):
                try:
                    support = sorted({int(i) for i in kw.get(support_key, [])})
                except Exception:
                    support = []
                if len(support) > 0:
                    # No degree information in index-only support; caller falls
                    # back to nominal degree for that part.
                    return int(len(support)), -1
                return None
            else:
                n_terms = 0
                max_deg = 0
                for row in rows:
                    if not isinstance(row, (list, tuple)):
                        continue
                    try:
                        exp = [int(v) for v in row]
                    except Exception:
                        continue
                    if len(exp) == 0:
                        continue
                    n_terms += 1
                    d = int(sum(exp))
                    if d > max_deg:
                        max_deg = d
                if n_terms <= 0:
                    return None
                return int(n_terms), int(max_deg)

        if k in ("var", "x", "input"):
            return 0.0
        if k in (
            "free_const",
            "freeconst",
            "free_constant",
            "fixed_const",
            "fixedconst",
            "fixed_constant",
        ):
            return 0.25
        if k in ("scale", "mul_scale"):
            return 0.25
        if k == "nn":
            return NN_ANALYTIC_DOMINATION_COST
        if k in ("lin", "linear", "affine"):
            return 1.0
        if k in ("power", "pow"):
            return 2.0
        if k in ("sin_linear", "sinlinear", "cos_linear", "coslinear"):
            return 4.0
        if k in ("tanh_linear", "tanhlinear"):
            return 5.0

        if k in ("poly", "rpoly"):
            deg = kw.get("deg", kw.get("degree", 1))
            try:
                deg = int(deg)
            except Exception:
                deg = 1
            return 1.5 + 1.2 * max(0, deg - 1)

        if k in ("ratio_poly", "ratiopoly", "ratio"):
            dn = kw.get("deg_num", kw.get("p", 1))
            dd = kw.get("deg_den", kw.get("q", 1))
            try:
                dn = int(dn)
            except Exception:
                dn = 1
            try:
                dd = int(dd)
            except Exception:
                dd = 1
            return 5.5 + 1.0 * max(0, dn) + 1.0 * max(0, dd)

        if k in (
            "ratpoly",
            "rationalpoly",
            "rational_poly",
            "rratpoly",
            "rrational_poly",
            "rrationalpolynomial",
        ):
            dn = kw.get("deg_num", kw.get("p", 1))
            dd = kw.get("deg_den", kw.get("q", 1))
            try:
                dn = int(dn)
            except Exception:
                dn = 1
            try:
                dd = int(dd)
            except Exception:
                dd = 1

            # Sparse override-aware scoring: when explicit monomial supports are
            # provided, reflect both effective degree and active term count.
            num_stats = _override_terms_maxdeg("exps_num_override", "support_num_override")
            den_stats = _override_terms_maxdeg("exps_den_override", "support_den_override")
            if num_stats is not None or den_stats is not None:
                n_num = int(num_stats[0]) if num_stats is not None else max(1, int(dn) + 1)
                n_den = int(den_stats[0]) if den_stats is not None else max(1, int(dd) + 1)
                dn_eff = int(num_stats[1]) if (num_stats is not None and int(num_stats[1]) >= 0) else int(dn)
                dd_eff = int(den_stats[1]) if (den_stats is not None and int(den_stats[1]) >= 0) else int(dd)
                n_terms = int(n_num + n_den)
                return 5.5 + 0.35 * max(1, n_terms) + 1.0 * max(0, dn_eff) + 1.0 * max(0, dd_eff)

            return 7.0 + 1.5 * max(0, dn) + 1.5 * max(0, dd)

        if k in ("exp_poly", "exppoly", "rexp_poly", "rexppoly"):
            deg = kw.get("deg", kw.get("degree", 1))
            try:
                deg = int(deg)
            except Exception:
                deg = 1
            return 9.0 + 1.0 * max(0, deg)

        if k in (
            "exp_ratpoly",
            "exp_rat",
            "exprat",
            "exp_rational",
            "exprationalpoly",
        ):
            dn = kw.get("deg_num", kw.get("p", 1))
            dd = kw.get("deg_den", kw.get("q", 1))
            try:
                dn = int(dn)
            except Exception:
                dn = 1
            try:
                dd = int(dd)
            except Exception:
                dd = 1
            return 14.0 + 2.0 * max(0, dn) + 2.0 * max(0, dd)

        if k in ("polylog", "rpolylog"):
            deg = kw.get("deg", kw.get("degree", 1))
            try:
                deg = int(deg)
            except Exception:
                deg = 1
            return 10.0 + 1.0 * max(0, deg)

        return 6.0

    def _pow_node_cost(exponent: object) -> float:
        if isinstance(exponent, (int, float)):
            ef = float(exponent)
            if _near(ef, 1.0):
                return 0.2
            if _near(ef, 2.0):
                return 1.5
            if _near(ef, -1.0):
                return 2.5
            if _near(ef, 0.5):
                return 2.5
            if _near(ef, -0.5):
                return 3.0
            if _near(ef, 0.0):
                return 0.2
            if _near(ef, round(ef)):
                return 2.0 + 0.25 * abs(int(round(ef)))
            return 6.0
        return 12.0

    def _rec(node: Node) -> float:
        if isinstance(node, ConstNode):
            return 0.2
        if isinstance(node, AtomNode):
            base = _atom_kind_cost(getattr(node, "kind", ""), getattr(node, "kwargs", {}) or {})
            try:
                if has_nontrivial_input(node):
                    for inp in get_input_exprs(node):
                        base += _rec(inp)
            except Exception:
                pass
            return float(base)
        if isinstance(node, AddNode):
            return 1.0 + _rec(node.left) + _rec(node.right)
        if isinstance(node, MulNode):
            return 1.0 + _rec(node.left) + _rec(node.right)
        if isinstance(node, PowNode):
            return _rec(node.base) + _pow_node_cost(getattr(node, "exponent", None))
        if isinstance(node, LogNode):
            return 4.5 + _rec(node.arg)
        if isinstance(node, ExpNode):
            return 4.5 + _rec(node.arg)
        if isinstance(node, SinNode):
            return 4.0 + _rec(node.arg)
        if isinstance(node, CosNode):
            return 4.0 + _rec(node.arg)
        if isinstance(node, (AsinNode, AcosNode, AtanNode)):
            return 5.0 + _rec(node.arg)
        if isinstance(node, AbsNode):
            return 20.0 + _rec(node.arg)
        if isinstance(node, ArgNode):
            return 20.0 + _rec(node.arg)
        if isinstance(node, (ConjNode, RealNode, ImagNode)):
            return 12.0 + _rec(node.arg)

        # Defensive fallback for future node classes.
        total = 5.0
        for attr in ("left", "right", "base", "arg"):
            if not hasattr(node, attr):
                continue
            try:
                child = getattr(node, attr)
            except Exception:
                child = None
            if child is not None:
                total += _rec(child)
        return float(total)

    try:
        return float(_rec(root))
    except Exception:
        return float("inf")


def ast_operator_cost(root: Node, *, include_atom_inputs: bool = True) -> float:
    """Backward-compatible alias for the physics-prior AST complexity."""
    # `include_atom_inputs` is retained for API compatibility; physics prior
    # always includes non-trivial input expressions.
    _ = include_atom_inputs
    return ast_cost_physics_prior(root)


def mapping_cost(mapping: Optional[dict]) -> float:
    """Explicit complexity for factorized symbolic search output mappings."""
    m = mapping or {}
    kind = str(m.get("kind", "")).strip().lower()

    if kind == "affine":
        cost = 1.0
    elif kind == "poly":
        deg = max(0, len(m.get("coeffs", [])) - 1)
        if deg <= 1:
            cost = 1.0 + 0.5 * float(deg)
        else:
            cost = 3.0 + 3.0 * float(deg - 1)
    elif kind == "power":
        cost = 2.5
    elif kind in ("monomial", "mono"):
        cost = 0.5
    elif kind in ("sine", "exp"):
        cost = 3.5
    elif kind == "pade":
        p = max(0, len(m.get("numer", [])) - 1)
        q = max(0, len(m.get("denom", [])) - 1)
        cost = 18.0 + 3.0 * float(p + q)
    elif kind in ("", "identity"):
        cost = 0.0
    else:
        cost = 6.0

    # Optional additive linear head used during scoring (extra coefficients / terms).
    head = m.get("_lin_head", None)
    if isinstance(head, dict):
        coeffs = head.get("coeffs", None)
        if isinstance(coeffs, (list, tuple)):
            eps = 1e-12
            n_active = 0
            for c in coeffs:
                try:
                    if abs(float(c)) > eps:
                        n_active += 1
                except Exception:
                    n_active += 1
            # Keep this modest: it should encourage fewer head terms without
            # overwhelming the tree-size prior.
            cost += 0.5 * float(n_active)

    return float(cost)


def param_ratio_decades(base_params: Optional[int], cand_params: Optional[int]) -> float:
    """Return log10(base_params / cand_params) if cand is smaller, else 0."""
    try:
        b = float(base_params) if base_params is not None else float("nan")
        c = float(cand_params) if cand_params is not None else float("nan")
    except Exception:
        return 0.0
    if (not math.isfinite(b)) or (not math.isfinite(c)):
        return 0.0
    b = max(1.0, b)
    c = max(1.0, c)
    if c >= b:
        return 0.0
    return float(math.log10(b / c))


def simplification_budget_decades(
    *,
    base_ast: Node,
    cand_ast: Node,
    base_params: Optional[int],
    cand_params: Optional[int],
    count_weight: float = 1.0,
    struct_gamma: float = 0.05,
    param_gamma: float = 0.30,
    base_bonus_decades: float = 0.0,
    sep_bonus_decades: float = 0.05,
    partial_sep_bonus_decades: float = 0.02,
    is_separability: bool = False,
    is_partial_separability: bool = False,
    extra_bonus_decades: float = 0.0,
) -> float:
    """Convert simplification into an allowed loss-regression budget (log10 decades)."""
    try:
        s_base = float(nn_structural_score(base_ast, count_weight=float(count_weight)))
        s_cand = float(nn_structural_score(cand_ast, count_weight=float(count_weight)))
    except Exception:
        s_base, s_cand = 0.0, 0.0
    gain_struct = max(0.0, s_base - s_cand)
    gain_param = max(0.0, float(param_ratio_decades(base_params, cand_params)))

    bonus = 0.0
    if bool(is_partial_separability):
        bonus += float(partial_sep_bonus_decades)
    elif bool(is_separability):
        bonus += float(sep_bonus_decades)

    budget = (
        float(base_bonus_decades)
        + float(extra_bonus_decades)
        + float(struct_gamma) * float(gain_struct)
        + float(param_gamma) * float(gain_param)
        + float(bonus)
    )
    if not math.isfinite(budget):
        return 0.0
    return float(max(0.0, budget))


def _compute_accept_threshold_impl(
    *,
    base_loss: Optional[float],
    best_loss: Optional[float],
    base_ast: Node,
    cand_ast: Node,
    base_params: Optional[int],
    cand_params: Optional[int],
    loss_floor: float,
    loss_cap: float,
    count_weight: float = 1.0,
    struct_gamma: float = 0.05,
    param_gamma: float = 0.30,
    base_bonus_decades: float = 0.0,
    sep_bonus_decades: float = 0.05,
    partial_sep_bonus_decades: float = 0.02,
    is_separability: bool = False,
    is_partial_separability: bool = False,
    extra_bonus_decades: float = 0.0,
    max_worsening_factor: Optional[float] = None,
    worsening_floor: Optional[float] = None,
    hard_ceiling: Optional[float] = None,
) -> float:
    """Compute an LM accept-threshold for a candidate rewrite."""
    try:
        cap = float(loss_cap)
    except Exception:
        cap = float("inf")

    # If we don't have a meaningful base loss, fall back to the global cap.
    if base_loss is None:
        return cap
    try:
        base = float(base_loss)
    except Exception:
        return cap
    if not math.isfinite(base):
        return cap

    ref = base
    if best_loss is not None:
        try:
            b = float(best_loss)
            if math.isfinite(b):
                ref = min(ref, b)
        except Exception:
            pass

    floor = float(loss_floor)
    if not math.isfinite(floor) or floor <= 0:
        floor = 0.0
    ref = max(ref, floor)

    budget_dec = simplification_budget_decades(
        base_ast=base_ast,
        cand_ast=cand_ast,
        base_params=base_params,
        cand_params=cand_params,
        count_weight=float(count_weight),
        struct_gamma=float(struct_gamma),
        param_gamma=float(param_gamma),
        base_bonus_decades=float(base_bonus_decades),
        sep_bonus_decades=float(sep_bonus_decades),
        partial_sep_bonus_decades=float(partial_sep_bonus_decades),
        is_separability=bool(is_separability),
        is_partial_separability=bool(is_partial_separability),
        extra_bonus_decades=float(extra_bonus_decades),
    )

    try:
        threshold = ref * (10.0 ** float(budget_dec))
    except Exception:
        threshold = ref

    if math.isfinite(cap):
        threshold = min(float(threshold), float(cap))

    # Optional legacy safeguard using max_worsening_factor:
    # - For separability: acts as FLOOR (minimum guarantee of headroom)
    # - For other rewrites: acts as CAP (maximum allowed regression)
    if max_worsening_factor is not None:
        try:
            mf = float(max_worsening_factor)
            if math.isfinite(mf) and mf > 0:
                wcap = ref * mf
                if worsening_floor is not None:
                    wf = float(worsening_floor)
                    if math.isfinite(wf):
                        wcap = max(wcap, wf)
                if is_separability or is_partial_separability:
                    # Separability: guarantee at least 100x headroom from baseline
                    threshold = max(float(threshold), float(wcap))
                else:
                    # Other rewrites: cap to prevent excessive regression
                    threshold = min(float(threshold), float(wcap))
        except Exception:
            pass

    # Global best-loss guard: hard ceiling that even the worsening floor cannot exceed.
    if hard_ceiling is not None:
        try:
            hc = float(hard_ceiling)
            if math.isfinite(hc) and hc > 0:
                threshold = min(float(threshold), hc)
        except Exception:
            pass

    return float(threshold)


def _loss_excess_above_floor(value: Optional[float], noise_floor: float) -> Optional[float]:
    """Return max(value - noise_floor, 0) preserving None / non-finite as None."""
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(max(0.0, out - float(noise_floor)))


def loss_excess_above_floor(value: Optional[float], noise_floor: float) -> Optional[float]:
    """Public helper for converting a raw loss ceiling/value to excess-loss space.

    This is mainly useful at call sites that need to translate an already-raw
    threshold (for example a global hard ceiling) before passing it into
    ``compute_accept_threshold(..., noise_floor=...)``.
    """
    return _loss_excess_above_floor(value, noise_floor)


def compute_accept_threshold(
    *,
    base_loss: Optional[float],
    best_loss: Optional[float],
    base_ast: Node,
    cand_ast: Node,
    base_params: Optional[int],
    cand_params: Optional[int],
    loss_floor: float,
    loss_cap: float,
    count_weight: float = 1.0,
    struct_gamma: float = 0.05,
    param_gamma: float = 0.30,
    base_bonus_decades: float = 0.0,
    sep_bonus_decades: float = 0.05,
    partial_sep_bonus_decades: float = 0.02,
    is_separability: bool = False,
    is_partial_separability: bool = False,
    extra_bonus_decades: float = 0.0,
    # Optional additional hard ceiling (legacy worsening safeguard)
    max_worsening_factor: Optional[float] = None,
    worsening_floor: Optional[float] = None,
    # Global best-loss guard: hard ceiling that even the worsening floor cannot exceed.
    hard_ceiling: Optional[float] = None,
    # Optional externally supplied irreducible floor used for acceptance only.
    # When provided, base_loss/best_loss are compared in excess-loss space above
    # this floor. Other threshold-like arguments (loss_floor/loss_cap/
    # worsening_floor/hard_ceiling) are interpreted in that excess-loss space.
    noise_floor: Optional[float] = None,
) -> float:
    """Compute an LM accept-threshold for a candidate rewrite.

    The returned value is a *loss ceiling*: a candidate is acceptable if its
    validation loss falls below this ceiling.
    """
    nf = _clamp_nonnegative_finite(noise_floor, default=0.0)
    if nf <= 0.0:
        return _compute_accept_threshold_impl(
            base_loss=base_loss,
            best_loss=best_loss,
            base_ast=base_ast,
            cand_ast=cand_ast,
            base_params=base_params,
            cand_params=cand_params,
            loss_floor=loss_floor,
            loss_cap=loss_cap,
            count_weight=count_weight,
            struct_gamma=struct_gamma,
            param_gamma=param_gamma,
            base_bonus_decades=base_bonus_decades,
            sep_bonus_decades=sep_bonus_decades,
            partial_sep_bonus_decades=partial_sep_bonus_decades,
            is_separability=is_separability,
            is_partial_separability=is_partial_separability,
            extra_bonus_decades=extra_bonus_decades,
            max_worsening_factor=max_worsening_factor,
            worsening_floor=worsening_floor,
            hard_ceiling=hard_ceiling,
        )

    thr_excess = _compute_accept_threshold_impl(
        base_loss=_loss_excess_above_floor(base_loss, nf),
        best_loss=_loss_excess_above_floor(best_loss, nf),
        base_ast=base_ast,
        cand_ast=cand_ast,
        base_params=base_params,
        cand_params=cand_params,
        loss_floor=loss_floor,
        loss_cap=loss_cap,
        count_weight=count_weight,
        struct_gamma=struct_gamma,
        param_gamma=param_gamma,
        base_bonus_decades=base_bonus_decades,
        sep_bonus_decades=sep_bonus_decades,
        partial_sep_bonus_decades=partial_sep_bonus_decades,
        is_separability=is_separability,
        is_partial_separability=is_partial_separability,
        extra_bonus_decades=extra_bonus_decades,
        max_worsening_factor=max_worsening_factor,
        worsening_floor=worsening_floor,
        hard_ceiling=hard_ceiling,
    )
    return float(nf + float(thr_excess))


def complexity_key(
    root: Node,
    n_params: Optional[int],
    *,
    count_weight: float = 1.0,
) -> Tuple[float, int]:
    """Key used for tie-breaking when losses are below the meaningful floor."""
    try:
        s_nn = float(nn_structural_score(root, count_weight=float(count_weight)))
        s_ast = float(ast_cost_physics_prior(root))
        s = float(s_ast) + 0.25 * float(s_nn)
    except Exception:
        s = float("inf")
    try:
        p = int(n_params) if n_params is not None else int(1e18)
    except Exception:
        p = int(1e18)
    return (float(s), int(p))


def pareto_front_indices_2d(
    points: Sequence[Tuple[float, float]],
) -> list[int]:
    """Return non-dominated indices for 2D minimisation objectives.

    Each point is ``(loss, complexity)``. A point ``i`` is dominated if there
    exists another point ``j`` with both objectives no worse and at least one
    strictly better.

    Non-finite points are ignored.
    """
    finite: list[tuple[int, float, float]] = []
    for i, p in enumerate(points):
        try:
            a = float(p[0])
            b = float(p[1])
        except Exception:
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        finite.append((int(i), a, b))

    if not finite:
        return []

    # O(n log n) frontier for 2D minimisation:
    # 1) sort by (loss, complexity, idx)
    # 2) process equal-loss groups together
    # 3) keep only complexity minima that beat all previous-loss groups
    finite.sort(key=lambda t: (t[1], t[2], t[0]))

    keep: list[int] = []
    best_prev_b = float("inf")
    n = len(finite)
    i = 0
    while i < n:
        loss_i = finite[i][1]
        j = i + 1
        while j < n and finite[j][1] == loss_i:
            j += 1

        group = finite[i:j]
        group_min_b = group[0][2]  # sorted by complexity within the group

        # If prior losses already achieved <= group_min_b complexity, the whole
        # group is dominated by a lower-loss point.
        if group_min_b < best_prev_b:
            # Within equal-loss group, only min-complexity points are non-dominated
            # (duplicates at the same (loss, complexity) are retained).
            for idx, _, b in group:
                if b == group_min_b:
                    keep.append(int(idx))
                else:
                    break
            best_prev_b = group_min_b

        i = j

    keep.sort(key=lambda idx: (float(points[idx][0]), float(points[idx][1]), int(idx)))
    return keep
