# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Univariate NN rewrite rules and their Stage-B helper builders."""

from __future__ import annotations

import copy
import math
import sys
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    MulNode,
    Node,
    PowNode,
    Scale,
    SinNode,
    Var,
    _collect_var_idxs_from_node,
    ast_to_human_readable,
    clone_ast,
    clone_inputs,
    compound_input_expr,
    effective_arity,
    eval_input_expr,
    get_input_exprs,
    has_nontrivial_input,
    replace_atom_in_ast,
)
from nestynet_sr.sr_core.constants import (
    build_scalar_atom_from_variant as _build_scalar_atom_from_variant,
    make_unit_aware_scalar_atom as _make_unit_aware_scalar_atom,
    scalar_constant_variants as _scalar_constant_variants,
)
from nestynet_sr.sr_search.candidate_builders import (
    _build_atom_input_tensor,
    _gather_atom_teacher_data,
)
from nestynet_sr.sr_search.monomial_screen import (
    MonomialScreenResult,
    candidate_priority_from_screen,
    fit_univariate_monomial_screen,
    half_power_domain_ok,
    monomial_power_label,
    snap_to_half_integer_monomial_power,
    snap_to_integer_monomial_power,
)
from nestynet_sr.sr_search.template_library import (
    propose_sinc_family,
)
from nestynet_sr.sr_search.wrapper_policy import macro_arg_wrapper_policy, snap_omega
from nestynet_sr.sr_search.phase_scan import stable_int_hash

from .engine import Candidate, StageBContext, StageBRule, atom_content_hash
from .helpers import (
    _best_scale_spec_for_axis,
    _build_expm1_1d_candidate,
    _build_inv_poly_candidates,
    _build_log_poly_candidate,
    _build_log_ratpoly_candidate,
    _build_planck_1d_candidates,
    _build_planck_full_1d_candidate,
    _build_power_exp_1d_candidate,
    _build_ratpoly_1d_candidates,
    _build_sqrt_poly_candidate,
    _build_sqrt_ratpoly_1d_candidates,
    _build_symexp_denom_1d_candidate,
    _collect_all_atoms,
    _collect_univariate_nn_atoms,
    _estimate_trig_params_on_compound,
    _estimate_univariate_trig_amplitude,
    _is_strong_scaling_spec,
    _leaf_coeff_param,
    _make_affine_trig_rewrite,
    _make_exp_poly_rewrite,
    _make_exp_ratpoly_rewrite,
    _make_logshifted_1d_rewrite,
    _make_poly_1d_rewrite,
    _make_polylog_1d_rewrite,
    _make_scaling_based_rewrite,
    _make_tanh_based_rewrite,
    _make_trig_based_rewrite,
    _set_constant_leaf_value,
    build_atom_to_leaf_map,
)
from .rules_common import (
    _HALF_POWER_SCREEN_REL_RMS_MAX,
    _INTEGER_POWER_SCREEN_MAX_POWER,
    _INTEGER_POWER_SCREEN_REL_RMS_MAX,
    _MONOMIAL_DEGREES,
    _effective_input_dims_for_atom,
    _mark_reciprocal_coordinate_candidate,
    _merge_reciprocal_aliases_pairwise,
    _reciprocal_alias_repeat_reason,
    _stageB_noise_floor_raw,
    _stageB_noisy_rel_rms_threshold,
    _wrap_reuse_for_reciprocal_coordinate,
)
from .models import _SubtreeModel

try:
    from nestynet_sr.sr_core.units import (
        _dim_in_rational_span as _dim_in_rational_span,
        is_dimless as _is_dimless,
        scale_dim as _scale_dim,
    )
except Exception:  # pragma: no cover
    _dim_in_rational_span = None  # type: ignore
    _is_dimless = None  # type: ignore
    _scale_dim = None  # type: ignore


def _ctx_pattern_disabled(ctx: StageBContext, name: str) -> bool:
    checker = getattr(ctx, "is_pattern_disabled", None)
    if checker is None:
        return False
    return bool(checker(name))


def _sync_stageb_rules_compat_overrides() -> None:
    """Honor legacy tests/tools that monkeypatch helpers through ``stageB.rules``."""
    rules_mod = sys.modules.get("nestynet_sr.sr_search.stageB.rules")
    if rules_mod is None:
        return
    for name in (
        "_stageB_target_raw_teacher_data",
        "_build_ratpoly_1d_candidates",
    ):
        if hasattr(rules_mod, name):
            globals()[name] = getattr(rules_mod, name)


def _prepare_univariate_units_probe(
    ctx: StageBContext,
    target: AtomNode,
    units_spec: Any,
) -> Tuple[bool, Optional[Tuple[Any, ...]], Optional[List[Tuple[Any, ...]]]]:
    """Prepare 1D probe metadata from the atom's effective inputs."""
    if not getattr(ctx, "enforce_units", False) or units_spec is None:
        return False, None, None

    dimless = tuple(0 for _ in units_spec.unit_system.base)
    eff_x_dims = _effective_input_dims_for_atom(target, units_spec)
    inv_homo = any(d != dimless for d in eff_x_dims)

    try:
        target_dim = tuple(ctx.infer_target_dim(target) or ())
        if not target_dim:
            return inv_homo, None, None
    except Exception:
        return inv_homo, None, None

    return inv_homo, target_dim, (eff_x_dims or None)


def _stageB_target_raw_teacher_data(
    ctx: StageBContext,
    target: AtomNode,
    *,
    max_points: int = 5000,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Gather raw ``X`` and the target NN teacher values for a Stage-B atom."""
    try:
        atom_to_leaf = build_atom_to_leaf_map(ctx.state.root, ctx.state.model)
        subtree = _SubtreeModel(root=target, atom_to_leaf=atom_to_leaf)
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        total = 0
        for batch in ctx.train_loader_probe:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device=ctx.device, dtype=ctx.dtype)
            with torch.no_grad():
                yb = subtree.forward(xb)
            yb = yb.reshape(xb.shape[0], -1)[:, 0]
            xs.append(xb.detach())
            ys.append(yb.detach())
            total += int(xb.shape[0])
            if total >= int(max_points):
                break
        if not xs:
            return None
        X = torch.cat(xs, dim=0)[: int(max_points)]
        y = torch.cat(ys, dim=0)[: int(max_points)].reshape(-1)
        if int(X.shape[0]) < 200:
            return None
        return X, y
    except Exception:
        return None


def _stageB_expr_dim(ctx: StageBContext, expr: Node) -> Optional[Tuple[Any, ...]]:
    if not bool(getattr(ctx, "enforce_units", False)):
        return None
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return None
    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim

        dim = eval_analytic_expr_dim(
            expr,
            units_spec.x_dims,
            free_const_dims=getattr(units_spec, "free_const_dims", {}) or {},
            fixed_const_dims=getattr(units_spec, "fixed_const_dims", {}) or {},
        )
        return None if dim is None else tuple(dim)
    except Exception:
        return None


def _stageB_dimless(ctx: StageBContext) -> Optional[Tuple[Any, ...]]:
    units_spec = getattr(ctx, "units_spec", None)
    if units_spec is None:
        return None
    try:
        return tuple(units_spec.unit_system.dimless())
    except Exception:
        return None


def _stageB_required_scalar_dim(
    target_dim: Optional[Tuple[Any, ...]],
    feature_dim: Optional[Tuple[Any, ...]],
) -> Optional[Tuple[Any, ...]]:
    if target_dim is None or feature_dim is None:
        return None
    try:
        from nestynet_sr.sr_core.units import sub_dim

        return tuple(sub_dim(tuple(target_dim), tuple(feature_dim)))
    except Exception:
        return None


def _stageB_trig_arg_ast(expr: Node, omega: float) -> Node:
    try:
        w = float(snap_omega(float(omega)))
    except Exception:
        w = float(omega)
    if not math.isfinite(w):
        w = 1.0
    if abs(w - 1.0) <= 1.0e-12:
        return clone_ast(expr)
    return MulNode(ConstNode(float(w)), clone_ast(expr))


def _stageB_ast_product(a: Node, b: Node) -> Node:
    return MulNode(clone_ast(a), clone_ast(b))


def _stageB_ast_inverse(a: Node) -> Node:
    return PowNode(clone_ast(a), -1.0)


def _stageB_ast_sum(nodes: Sequence[Node]) -> Optional[Node]:
    out: Optional[Node] = None
    for node in nodes:
        out = clone_ast(node) if out is None else AddNode(out, clone_ast(node))
    return out


def _stageB_effective_input_expr(target: AtomNode) -> Optional[Node]:
    try:
        inputs = tuple(get_input_exprs(target))
        if inputs:
            return clone_ast(inputs[0])
    except Exception:
        pass
    try:
        z = compound_input_expr(target)
        if z is not None:
            return clone_ast(z)
    except Exception:
        pass
    try:
        if len(tuple(getattr(target, "var_idxs", ()) or ())) == 1:
            return Var(int(target.var_idxs[0]))
    except Exception:
        pass
    return None


def _build_fixed_trig_factor_candidates(
    ctx: StageBContext,
    target: AtomNode,
    trig_spec: Any,
    *,
    max_candidates: int = 3,
) -> List[Candidate]:
    """One-parameter visible trig factors from strong trig hints.

    This is intentionally narrower than ``sin_linear``/``affine_trig``: it
    proposes only ``scale*sin(ωz)``, ``scale*cos(ωz)``, or
    ``scale*(1-cos(ωz))``.  These exact leaves should get a chance before
    generic rational approximants when the trig hint is already strong.
    """
    _sync_stageb_rules_compat_overrides()
    expr = _stageB_effective_input_expr(target)
    if expr is None:
        return []
    if bool(getattr(ctx, "enforce_units", False)):
        arg_dim = _stageB_expr_dim(ctx, expr)
        dimless = _stageB_dimless(ctx)
        if arg_dim is None or dimless is None or tuple(arg_dim) != tuple(dimless):
            return []

    try:
        omega = float(snap_omega(float(getattr(trig_spec, "omega", 1.0))))
    except Exception:
        omega = 1.0
    if not math.isfinite(omega) or omega <= 0:
        omega = 1.0

    basis = str(getattr(trig_spec, "basis_fn", "") or getattr(trig_spec, "trig_fn", "")).lower()
    trig_fn = str(getattr(trig_spec, "trig_fn", "")).lower()
    kinds: List[str] = []
    if basis == "one_minus_cos":
        kinds.extend(["one_minus_cos", "cos"])
    elif trig_fn == "sin":
        kinds.append("sin")
    else:
        kinds.extend(["cos", "one_minus_cos"])

    # De-duplicate while preserving priority.
    seen_kinds = set()
    kinds = [k for k in kinds if not (k in seen_kinds or seen_kinds.add(k))]

    teacher = None
    try:
        teacher = ctx.state.reuse.get(getattr(target, "tag", None), None)
    except Exception:
        teacher = None
    data = None
    if teacher is not None:
        try:
            data = _gather_atom_teacher_data(
                train_loader=ctx.train_loader_probe,
                atom=target,
                teacher=teacher,
                device=ctx.device,
                dtype=ctx.dtype,
                max_points=int(getattr(ctx.lm_hp, "stageB_fixed_trig_factor_max_points", 5000) or 5000),
            )
        except Exception:
            data = None

    scale_inits: Dict[str, Tuple[float, float]] = {}
    if data is not None:
        try:
            X, F = data
            if X.ndim == 2 and int(X.shape[1]) >= 1:
                z = X[:, 0].detach().to(dtype=torch.float64).reshape(-1)
                y = F.detach().to(dtype=torch.float64).reshape(-1)
                m = torch.isfinite(z) & torch.isfinite(y)
                z = z[m]
                y = y[m]
                if int(z.numel()) >= 128:
                    centered = y - torch.mean(y)
                    denom_y = torch.sqrt(torch.mean(centered * centered)).clamp_min(1.0e-30)
                    arg_vals = float(omega) * z
                    basis_vals = {
                        "sin": torch.sin(arg_vals),
                        "cos": torch.cos(arg_vals),
                        "one_minus_cos": 1.0 - torch.cos(arg_vals),
                    }
                    for k in kinds:
                        phi = basis_vals[k]
                        den = torch.sum(phi * phi)
                        if (not torch.isfinite(den)) or float(den.item()) <= 1.0e-30:
                            continue
                        scale = torch.sum(phi * y) / den
                        pred = scale * phi
                        rel = torch.sqrt(torch.mean((y - pred) * (y - pred))) / denom_y
                        scale_f = float(scale.item())
                        rel_f = float(rel.item())
                        if math.isfinite(scale_f) and math.isfinite(rel_f):
                            scale_inits[k] = (scale_f, rel_f)
        except Exception:
            scale_inits = {}

    try:
        max_rel = float(getattr(ctx.lm_hp, "stageB_fixed_trig_factor_screen_rel_rms", 5.0e-2))
    except Exception:
        max_rel = 5.0e-2
    if data is not None:
        try:
            max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=y)
        except Exception:
            pass
    if data is not None:
        kinds = [k for k in kinds if k in scale_inits and float(scale_inits[k][1]) <= max_rel]
    if not kinds:
        return []

    try:
        target_dim = tuple(ctx.infer_target_dim(target) or ())
        if not target_dim:
            target_dim = None
    except Exception:
        target_dim = None

    base_tag = str(getattr(target, "tag", None) or "leaf")
    out: List[Candidate] = []
    for kind in kinds[: max(0, int(max_candidates))]:
        arg_ast = _stageB_trig_arg_ast(expr, omega)
        if kind == "sin":
            core = SinNode(arg_ast)
        elif kind == "one_minus_cos":
            core = AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), CosNode(arg_ast)))
        else:
            core = CosNode(arg_ast)
        scale_init, rel = scale_inits.get(kind, (1.0, float("nan")))
        scale_tag = f"{base_tag}_fixed_trig_{kind}_scale"
        try:
            scale_node = _make_unit_aware_scalar_atom(
                target_dim,
                getattr(ctx, "units_spec", None),
                base_tag=scale_tag,
                init=float(scale_init),
                strict=bool(getattr(ctx, "enforce_units", False)),
            )
        except Exception:
            continue
        scale_node_tag = str(getattr(scale_node, "tag", scale_tag))
        new_subtree = MulNode(scale_node, core)
        root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
        if root_new is None:
            continue

        def _init(root_new_inner: Node, model_new: nn.Module, *, _tag=scale_node_tag, _scale=float(scale_init)):
            atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
            for atom in _collect_all_atoms(root_new_inner):
                if not isinstance(atom, AtomNode) or str(getattr(atom, "tag", "")) != str(_tag):
                    continue
                leaf = atom_to_leaf.get(id(atom), None)
                if leaf is None:
                    continue
                try:
                    _set_constant_leaf_value(leaf, float(_scale))
                except Exception:
                    pass

        _init._after_analytic_init = True
        label = f"fixed_trig_factor_{kind}"
        meta = {
            "structural": True,
            "exact_non_generic": True,
            "pattern": "fixed_trig_factor",
            "pattern_family": "fixed_trig_factor",
            "trig_kind": kind,
            "omega": float(omega),
            "screen_rel_rms": float(rel),
            "min_free_params": 1,
            "signature": (
                int(atom_content_hash(target)),
                stable_int_hash("fixed_trig_factor"),
                stable_int_hash(kind),
                int(round(float(omega) * 1.0e6)),
            ),
            "log": (
                f"[Stage B]  Trying fixed trig factor {kind}(ωz) on NN vars {target.var_idxs}, "
                f"ω≈{omega:.6g}, scale≈{float(scale_init):.4g}, rel={float(rel):.2e}"
            ),
        }
        out.append(Candidate(label=label, root=root_new, init_fn=_init, meta=meta))
    return out


def _stageB_trig_feature_contexts(
    ctx: StageBContext,
    target: AtomNode,
    trig_axis: int,
    X: torch.Tensor,
    *,
    max_contexts: int = 18,
) -> List[Dict[str, Any]]:
    """Small context pool for trig-feature linear closure."""
    try:
        inputs = tuple(get_input_exprs(target))
    except Exception:
        inputs = ()
    if not inputs:
        return []

    raw: List[Tuple[str, Node, torch.Tensor, int]] = []
    ones = torch.ones((int(X.shape[0]),), dtype=torch.float64, device=X.device)
    raw.append(("1", ConstNode(1.0), ones, 0))

    for i, expr in enumerate(inputs):
        try:
            vars_i = set(int(v) for v in _collect_var_idxs_from_node(expr))
        except Exception:
            vars_i = set()
        if vars_i == {int(trig_axis)}:
            continue
        try:
            vals = eval_input_expr(expr, X).reshape(-1).detach().to(dtype=torch.float64)
        except Exception:
            continue
        try:
            label = ast_to_human_readable(expr)
        except Exception:
            label = f"u{i}"
        raw.append((label, clone_ast(expr), vals, 1))
        finite = torch.isfinite(vals) & (torch.abs(vals) > 1.0e-12)
        if int(finite.sum().item()) >= max(200, int(0.98 * vals.numel())):
            raw.append((f"1/({label})", _stageB_ast_inverse(expr), torch.reciprocal(vals), 2))

    # Add a very small number of pair products/reciprocals.  This catches
    # common physics prefactors while keeping the matrix tiny under the strong
    # trig gate.
    base_nontrig = [r for r in raw if r[0] != "1" and r[3] <= 1]
    for a_idx in range(len(base_nontrig)):
        for b_idx in range(a_idx + 1, len(base_nontrig)):
            la, ea, va, ca = base_nontrig[a_idx]
            lb, eb, vb, cb = base_nontrig[b_idx]
            vals = va * vb
            expr = _stageB_ast_product(ea, eb)
            raw.append((f"({la}*{lb})", expr, vals, ca + cb + 1))
            finite = torch.isfinite(vals) & (torch.abs(vals) > 1.0e-12)
            if int(finite.sum().item()) >= max(200, int(0.98 * vals.numel())):
                raw.append((f"1/({la}*{lb})", _stageB_ast_inverse(expr), torch.reciprocal(vals), ca + cb + 2))

    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for label, expr, vals, cost in sorted(raw, key=lambda t: (int(t[3]), len(str(t[0])))):
        key = str(label)
        if key in seen:
            continue
        seen.add(key)
        if not torch.isfinite(vals).any():
            continue
        out.append({"label": label, "expr": expr, "values": vals, "cost": int(cost)})
        if len(out) >= int(max_contexts):
            break
    return out


def _stageB_sparse_linear_screen(
    Phi: torch.Tensor,
    y: torch.Tensor,
    *,
    max_terms: int = 6,
    fit_frac: float = 0.75,
) -> Optional[Dict[str, Any]]:
    """Tiny OMP-style screen for sparse linear trig-feature candidates."""
    try:
        Phi = Phi.detach().to(dtype=torch.float64)
        y = y.detach().reshape(-1).to(dtype=torch.float64)
        finite = torch.isfinite(y) & torch.isfinite(Phi).all(dim=1)
        Phi = Phi[finite]
        y = y[finite]
        n, p = int(Phi.shape[0]), int(Phi.shape[1])
        if n < 200 or p <= 0:
            return None
        order = torch.arange(n, device=Phi.device)
        val_mask = (order % 4) == 0
        if int(val_mask.sum().item()) < 50 or int((~val_mask).sum().item()) < 100:
            n_fit = max(100, int(float(fit_frac) * n))
            val_mask = torch.zeros(n, dtype=torch.bool, device=Phi.device)
            val_mask[n_fit:] = True
        fit_mask = ~val_mask
        Xf, yf = Phi[fit_mask], y[fit_mask]
        Xv, yv = Phi[val_mask], y[val_mask]
        yv_center = yv - torch.mean(yv)
        denom = torch.sqrt(torch.mean(yv_center * yv_center)).clamp_min(1.0e-30)
        col_norm = torch.sqrt(torch.sum(Xf * Xf, dim=0)).clamp_min(1.0e-30)
        active: List[int] = []
        best: Optional[Dict[str, Any]] = None
        resid = yf.clone()
        for _ in range(max(1, int(max_terms))):
            score = torch.abs(torch.mv(Xf.T, resid)) / col_norm
            if active:
                score[torch.tensor(active, dtype=torch.long, device=score.device)] = -1.0
            j = int(torch.argmax(score).item())
            if not math.isfinite(float(score[j].item())) or float(score[j].item()) <= 1.0e-20:
                break
            active.append(j)
            Xa = Xf[:, active]
            sol = torch.linalg.lstsq(Xa, yf).solution.reshape(-1)
            if sol.numel() != len(active) or not torch.isfinite(sol).all():
                break
            resid = yf - Xa @ sol
            pred_v = Xv[:, active] @ sol
            rel = torch.sqrt(torch.mean((yv - pred_v) ** 2)) / denom
            if torch.isfinite(rel):
                rec = {
                    "support": tuple(int(i) for i in active),
                    "coeffs": tuple(float(v) for v in sol.detach().cpu().tolist()),
                    "rel_rms": float(rel.item()),
                }
                if best is None or rec["rel_rms"] < float(best["rel_rms"]):
                    best = rec
            if best is not None and float(best["rel_rms"]) <= 1.0e-8:
                break
        return best
    except Exception:
        return None


def _build_trig_feature_linear_candidates(
    ctx: StageBContext,
    target: AtomNode,
    trig_spec: Any,
    *,
    max_candidates: int = 2,
) -> List[Candidate]:
    """Build sparse visible linear candidates over context x trig features.

    This is a cheap, strong-trig-gated alternative to generic ratpoly.  It is
    designed for structures such as ``1/y = 1/x0 + (1-cos(theta))/A``.
    """
    _sync_stageb_rules_compat_overrides()
    if trig_spec is None or not isinstance(target, AtomNode):
        return []
    try:
        rel_std = float(getattr(trig_spec, "rel_std", 1.0))
    except Exception:
        rel_std = 1.0
    if not math.isfinite(rel_std) or rel_std > 0.1:
        return []
    try:
        axis = int(getattr(trig_spec, "axis"))
    except Exception:
        return []

    data = _stageB_target_raw_teacher_data(ctx, target, max_points=5000)
    if data is None:
        return []
    X, y = data
    X = X.detach().to(dtype=torch.float64)
    y = y.detach().to(dtype=torch.float64).reshape(-1)

    input_exprs = tuple(get_input_exprs(target))
    trig_expr = None
    for expr in input_exprs:
        try:
            if set(int(v) for v in _collect_var_idxs_from_node(expr)) == {axis}:
                trig_expr = expr
                break
        except Exception:
            continue
    if trig_expr is None:
        return []

    try:
        theta = eval_input_expr(trig_expr, X).reshape(-1).detach().to(dtype=torch.float64)
    except Exception:
        return []
    omega = float(getattr(trig_spec, "omega", 1.0) or 1.0)
    try:
        omega = float(snap_omega(omega))
    except Exception:
        pass
    if not math.isfinite(omega):
        omega = 1.0
    arg_vals = omega * theta
    cos_vals = torch.cos(arg_vals)
    sin_vals = torch.sin(arg_vals)
    one_minus_cos_vals = 1.0 - cos_vals
    sin_sq_vals = sin_vals * sin_vals

    arg_ast = _stageB_trig_arg_ast(trig_expr, omega)
    cos_ast = CosNode(clone_ast(arg_ast))
    sin_ast = SinNode(clone_ast(arg_ast))
    one_minus_cos_ast = AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), clone_ast(cos_ast)))
    sin_sq_ast = PowNode(clone_ast(sin_ast), 2.0)

    contexts = _stageB_trig_feature_contexts(ctx, target, axis, X, max_contexts=18)
    if not contexts:
        return []

    try:
        target_dim = tuple(ctx.infer_target_dim(target) or ())
        if not target_dim:
            target_dim = None
    except Exception:
        target_dim = None
    dimless = _stageB_dimless(ctx)

    feature_rows: List[Dict[str, Any]] = []
    variants = [
        ("base", ConstNode(1.0), torch.ones_like(y), 0),
        ("cos", cos_ast, cos_vals, 1),
        ("one_minus_cos", one_minus_cos_ast, one_minus_cos_vals, 1),
        ("sin", sin_ast, sin_vals, 2),
        ("sin_sq", sin_sq_ast, sin_sq_vals, 2),
    ]

    for ctx_fe in contexts:
        ctx_label = str(ctx_fe["label"])
        ctx_expr = ctx_fe["expr"]
        ctx_vals = ctx_fe["values"].to(device=y.device, dtype=torch.float64).reshape(-1)
        for var_label, var_ast, var_vals, var_cost in variants:
            if var_label == "base":
                feat_expr = clone_ast(ctx_expr)
                feat_vals = ctx_vals
                feat_label = ctx_label
            else:
                feat_expr = MulNode(clone_ast(ctx_expr), clone_ast(var_ast))
                feat_vals = ctx_vals * var_vals
                feat_label = f"{ctx_label}*{var_label}"
            feat_dim = _stageB_expr_dim(ctx, feat_expr)
            if bool(getattr(ctx, "enforce_units", False)):
                if feat_dim is None or target_dim is None:
                    continue
                req_dim = _stageB_required_scalar_dim(target_dim, feat_dim)
                if req_dim is None:
                    continue
                if dimless is not None and tuple(req_dim) != tuple(dimless):
                    # The feature-linear rule fits arbitrary coefficients.  Under
                    # strict units, keep v1 to dimensionless coefficients rather
                    # than introducing undeclared unit-bearing constants.
                    continue
                try:
                    _make_unit_aware_scalar_atom(
                        req_dim,
                        getattr(ctx, "units_spec", None),
                        base_tag="__probe_trig_feature_scale",
                        init=1.0,
                        strict=True,
                    )
                except Exception:
                    continue
            else:
                req_dim = None
            if dimless is not None and feat_dim == dimless and target_dim is not None and target_dim != dimless:
                # Avoid introducing unitful constants under strict AIF policy.
                continue
            finite = torch.isfinite(feat_vals)
            if int(finite.sum().item()) < max(200, int(0.95 * feat_vals.numel())):
                continue
            feature_rows.append({
                "label": feat_label,
                "expr": feat_expr,
                "values": feat_vals,
                "req_dim": req_dim,
                "cost": int(ctx_fe.get("cost", 0)) + int(var_cost),
            })

    # Deduplicate labels and keep the low-cost, physics-motivated features first.
    seen: set[str] = set()
    features: List[Dict[str, Any]] = []
    for row in sorted(feature_rows, key=lambda r: (int(r["cost"]), len(str(r["label"])))):
        label = str(row["label"])
        if label in seen:
            continue
        seen.add(label)
        features.append(row)
        if len(features) >= 80:
            break
    if not features:
        return []

    Phi = torch.stack([r["values"].to(device=y.device, dtype=torch.float64).reshape(-1) for r in features], dim=1)
    screen = _stageB_sparse_linear_screen(Phi, y, max_terms=6)
    if screen is None:
        return []
    rel = float(screen.get("rel_rms", float("inf")))
    try:
        max_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_screen_rel_rms", 2.0e-2))
    except Exception:
        max_rel = 2.0e-2
    max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=y)
    max_rel = min(max_rel, 2.0e-3)
    if not math.isfinite(rel) or rel > max_rel:
        if getattr(ctx, "verbose", False):
            ctx.log(
                f"[Stage B TrigFeature] screen rejected vars={target.var_idxs}: "
                f"rel={rel:.2e} > {max_rel:.2e}"
            )
        return []

    support = tuple(int(i) for i in screen["support"])
    coeffs = tuple(float(c) for c in screen["coeffs"])
    if not support or len(support) != len(coeffs):
        return []

    terms: List[Node] = []
    scale_tags: List[str] = []
    labels: List[str] = []
    coeffs_selected: List[float] = []
    base_tag = str(getattr(target, "tag", None) or "leaf")
    for k, (idx, coeff) in enumerate(zip(support, coeffs)):
        if abs(float(coeff)) <= 1.0e-12:
            continue
        feat = features[int(idx)]
        tag = f"{base_tag}_trigfeat_{k}"
        try:
            scale_node = _make_unit_aware_scalar_atom(
                feat.get("req_dim", None),
                getattr(ctx, "units_spec", None),
                base_tag=tag,
                init=float(coeff),
                strict=bool(getattr(ctx, "enforce_units", False)),
            )
        except Exception:
            return []
        scale_tags.append(str(getattr(scale_node, "tag", tag)))
        labels.append(str(feat["label"]))
        coeffs_selected.append(float(coeff))
        terms.append(MulNode(scale_node, clone_ast(feat["expr"])))
    new_subtree = _stageB_ast_sum(terms)
    if new_subtree is None:
        return []
    root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
    if root_new is None:
        return []

    coeffs_eff = tuple(float(c) for c in coeffs_selected)

    def _init(root_new_inner: Node, model_new: nn.Module, *, _tags=tuple(scale_tags), _coeffs=coeffs_eff):
        atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
        tag_to_coeff = {str(t): float(c) for t, c in zip(_tags, _coeffs)}
        for atom in _collect_all_atoms(root_new_inner):
            if not isinstance(atom, AtomNode):
                continue
            tag = str(getattr(atom, "tag", "") or "")
            if tag not in tag_to_coeff:
                continue
            leaf = atom_to_leaf.get(id(atom), None)
            if leaf is None:
                continue
            try:
                _set_constant_leaf_value(leaf, tag_to_coeff[tag])
            except Exception:
                pass

    _init._after_analytic_init = True
    label = "trig_feature_linear"
    meta = {
        "structural": True,
        "pattern": "trig_feature_linear",
        "pattern_family": "trig_feature_linear",
        "trig_feature_linear": True,
        "screen_rel_rms": rel,
        "omega": float(omega),
        "trig_axis": int(axis),
        "feature_labels": tuple(labels),
        "signature": (
            int(atom_content_hash(target)),
            stable_int_hash("trig_feature_linear"),
            int(axis),
            int(round(float(omega) * 1.0e6)),
            tuple(stable_int_hash(lab) for lab in labels),
        ),
        "log": (
            f"[Stage B TrigFeature] Trying sparse trig-feature linear closure "
            f"axis=x{axis}, omega≈{omega:.6g}, rel={rel:.2e}, terms={list(labels)}"
        ),
    }
    return [Candidate(label=label, root=root_new, init_fn=_init, meta=meta)][: int(max_candidates)]


def _one_minus_cos_over_z2_omega_grid(trig_spec: Any) -> List[float]:
    """Small omega grid for ``(1-cos(omega*z))/z**2`` candidates."""

    vals: List[float] = []

    def _add(w: Any) -> None:
        try:
            ww = float(w)
        except Exception:
            return
        if not math.isfinite(ww) or abs(ww) <= 1.0e-12:
            return
        ww = abs(ww)
        try:
            ww = float(snap_omega(ww))
        except Exception:
            pass
        for old in vals:
            if abs(old - ww) <= max(1.0e-10, 1.0e-8 * max(abs(old), abs(ww))):
                return
        vals.append(float(ww))

    try:
        omega = float(getattr(trig_spec, "omega", 1.0) or 1.0)
    except Exception:
        omega = 1.0
    for w in (omega, 0.5 * omega, 2.0 * omega):
        _add(w)
    for w in (1.0, 0.5, 2.0, math.pi / 2.0, math.pi, 2.0 * math.pi):
        _add(w)
    return vals


def _relative_rms(y: torch.Tensor, resid: torch.Tensor) -> float:
    """RMS residual normalised to the target's centred RMS, with a safe fallback."""

    try:
        yy = y.detach().to(dtype=torch.float64).reshape(-1)
        rr = resid.detach().to(dtype=torch.float64).reshape(-1)
        denom = torch.sqrt(torch.mean((yy - torch.mean(yy)) ** 2))
        if (not torch.isfinite(denom)) or float(denom.item()) <= 1.0e-30:
            denom = torch.sqrt(torch.mean(yy * yy)).clamp_min(1.0e-30)
        return float((torch.sqrt(torch.mean(rr * rr)) / denom).item())
    except Exception:
        return float("inf")


def _build_one_minus_cos_over_z2_candidates(
    ctx: StageBContext,
    target: AtomNode,
    trig_spec: Any,
    *,
    max_candidates: int = 3,
) -> List[Candidate]:
    """Visible 1D closure for ``scale * (1-cos(omega*z)) / z**2``.

    This is the low-parameter exact counterpart to the more flexible ``sinc_p2``
    and generic rational-polynomial templates.  It is only proposed when a
    strong trig hint exists and a cheap direct screen already sees the shape.
    """
    _sync_stageb_rules_compat_overrides()

    def _diag(reason: str) -> None:
        if bool(getattr(ctx, "verbose", False)):
            ctx.log(f"[Stage B]  one_minus_cos_over_z2 skip NN vars {getattr(target, 'var_idxs', '?')}: {reason}")

    if trig_spec is None or not isinstance(target, AtomNode) or effective_arity(target) != 1:
        _diag("missing trig spec or target is not effective-1D")
        return []
    try:
        inputs = tuple(get_input_exprs(target))
    except Exception:
        inputs = ()
    if len(inputs) != 1:
        _diag(f"expected one effective input, got {len(inputs)}")
        return []
    z_expr = inputs[0]
    try:
        z_label = ast_to_human_readable(z_expr)
    except Exception:
        z_label = "?"

    if bool(getattr(ctx, "enforce_units", False)):
        z_dim = _stageB_expr_dim(ctx, z_expr)
        if z_dim is None:
            _diag(f"cannot infer phase units for z={z_label}")
            return []
        if _is_dimless is not None and not bool(_is_dimless(z_dim)):
            _diag(f"phase is not dimensionless for z={z_label}, dim={z_dim}")
            return []

    data = _stageB_target_raw_teacher_data(ctx, target, max_points=5000)
    if data is None:
        _diag("teacher-data screen unavailable")
        return []
    X, y = data
    X = X.detach().to(device=ctx.device, dtype=torch.float64)
    y = y.detach().to(device=ctx.device, dtype=torch.float64).reshape(-1)
    try:
        z = eval_input_expr(z_expr, X).detach().to(device=ctx.device, dtype=torch.float64).reshape(-1)
    except Exception as exc:
        _diag(f"failed to evaluate z={z_label}: {type(exc).__name__}: {exc}")
        return []

    finite_base = torch.isfinite(z) & torch.isfinite(y) & (torch.abs(z) > 1.0e-10)
    n_base = int(finite_base.sum().item())
    n_need = max(200, int(0.95 * z.numel()))
    if n_base < n_need:
        _diag(f"insufficient finite safe points for z={z_label}: {n_base}/{z.numel()} need {n_need}")
        return []
    z_ok = z[finite_base]
    y_ok = y[finite_base]
    inv_z2 = torch.reciprocal(z_ok * z_ok)

    try:
        max_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_screen_rel_rms", 2.0e-2))
    except Exception:
        max_rel = 2.0e-2
    max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=y_ok)
    try:
        max_offset_rel = float(getattr(ctx.lm_hp, "stageB_last_hard_trig_power_max_offset_rel", 0.15))
    except Exception:
        max_offset_rel = 0.15

    exact_hits: List[Tuple[float, float, float]] = []  # (rel, omega, scale)
    affine_hits: List[Tuple[float, float, float, float, float]] = []  # (rel, omega, a, b, offset_rel)
    best_exact: Tuple[float, Optional[float]] = (float("inf"), None)
    best_affine: Tuple[float, Optional[float], Optional[float]] = (float("inf"), None, None)

    omega_grid = _one_minus_cos_over_z2_omega_grid(trig_spec)
    if not omega_grid:
        _diag("omega grid is empty")
        return []

    for omega in omega_grid:
        arg = float(omega) * z_ok
        cos_v = torch.cos(arg)
        one_minus = 1.0 - cos_v
        feat = one_minus * inv_z2
        finite = torch.isfinite(feat)
        if int(finite.sum().item()) < max(200, int(0.95 * feat.numel())):
            continue
        f = feat[finite]
        yy = y_ok[finite]
        denom = torch.dot(f, f)
        if (not torch.isfinite(denom)) or float(denom.item()) <= 1.0e-30:
            continue
        scale = float((torch.dot(f, yy) / denom).item())
        rel = _relative_rms(yy, yy - float(scale) * f)
        if math.isfinite(rel) and rel < best_exact[0]:
            best_exact = (float(rel), float(omega))
        if math.isfinite(rel) and rel <= float(max_rel):
            exact_hits.append((float(rel), float(omega), float(scale)))

        # Fallback with one extra coefficient.  Keep it only when the fitted
        # numerator is genuinely one-minus-cos-like (a + b*cos, a≈-b), so this
        # does not become a generic trig rational substitute.
        phi0 = inv_z2
        phi1 = cos_v * inv_z2
        A = torch.stack([phi0[finite], phi1[finite]], dim=1)
        try:
            sol = torch.linalg.lstsq(A, yy).solution.reshape(-1)
        except Exception:
            continue
        if sol.numel() != 2 or not torch.isfinite(sol).all():
            continue
        pred = A @ sol
        rel_aff = _relative_rms(yy, yy - pred)
        a0 = float(sol[0].item())
        b0 = float(sol[1].item())
        denom_ab = max(abs(a0), abs(b0), 1.0e-30)
        offset_rel = abs(a0 + b0) / denom_ab
        if math.isfinite(rel_aff) and rel_aff < best_affine[0]:
            best_affine = (float(rel_aff), float(omega), float(offset_rel))
        if (
            math.isfinite(rel_aff)
            and rel_aff <= float(max_rel)
            and math.isfinite(offset_rel)
            and offset_rel <= float(max_offset_rel)
        ):
            affine_hits.append((float(rel_aff), float(omega), a0, b0, float(offset_rel)))

    exact_hits.sort(key=lambda t: t[0])
    affine_hits.sort(key=lambda t: t[0])
    if not exact_hits and not affine_hits:
        exact_msg = (
            f"exact best rel={best_exact[0]:.2e} at omega≈{best_exact[1]:.6g}"
            if best_exact[1] is not None
            else "exact no finite screen"
        )
        affine_msg = (
            f"affine best rel={best_affine[0]:.2e}, offset_rel={best_affine[2]:.2e} "
            f"at omega≈{best_affine[1]:.6g}"
            if best_affine[1] is not None and best_affine[2] is not None
            else "affine no finite screen"
        )
        _diag(
            f"screen failed for z={z_label}; {exact_msg}; {affine_msg}; "
            f"thresholds rel<={float(max_rel):.2e}, offset_rel<={float(max_offset_rel):.2e}"
        )
        return []

    try:
        target_dim = tuple(ctx.infer_target_dim(target) or ())
        if not target_dim:
            target_dim = None
    except Exception:
        target_dim = None

    def _scale_for_feature(feature_expr: Node, tag: str, init: float) -> Optional[Node]:
        feature_dim = _stageB_expr_dim(ctx, feature_expr)
        if bool(getattr(ctx, "enforce_units", False)):
            if target_dim is None or feature_dim is None:
                return None
            req_dim = _stageB_required_scalar_dim(target_dim, feature_dim)
        else:
            req_dim = None
        try:
            return _make_unit_aware_scalar_atom(
                req_dim,
                getattr(ctx, "units_spec", None),
                base_tag=tag,
                init=float(init),
                strict=bool(getattr(ctx, "enforce_units", False)),
            )
        except Exception:
            return None

    out: List[Candidate] = []
    base_tag = str(getattr(target, "tag", None) or "leaf")

    for idx, (rel, omega, scale) in enumerate(exact_hits[: max(1, int(max_candidates))]):
        z_for_arg = clone_ast(z_expr)
        arg_ast = _stageB_trig_arg_ast(z_for_arg, float(omega))
        one_minus_ast = AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), CosNode(arg_ast)))
        feature_ast = MulNode(one_minus_ast, PowNode(clone_ast(z_expr), -2.0))
        scale_tag = f"{base_tag}_omc_z2_scale_{idx}"
        scale_node = _scale_for_feature(feature_ast, scale_tag, float(scale))
        if scale_node is None:
            _diag(f"screen hit but unit-aware scale build failed for exact candidate z={z_label}")
            continue
        new_subtree = MulNode(scale_node, clone_ast(feature_ast))
        root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
        if root_new is None:
            _diag(f"screen hit but AST replacement failed for exact candidate z={z_label}")
            continue

        def _init(root_new_inner: Node, model_new: nn.Module, *, _tag=str(getattr(scale_node, "tag", scale_tag)), _scale=float(scale)):
            atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
            for atom in _collect_all_atoms(root_new_inner):
                if not isinstance(atom, AtomNode):
                    continue
                if str(getattr(atom, "tag", "") or "") != _tag:
                    continue
                leaf = atom_to_leaf.get(id(atom), None)
                if leaf is not None:
                    _set_constant_leaf_value(leaf, float(_scale))

        _init._after_analytic_init = True
        meta = {
            "structural": True,
            "pattern": "one_minus_cos_over_z2",
            "pattern_family": "one_minus_cos_over_z2",
            "precheck_rel_rms": float(rel),
            "omega": float(omega),
            "log": (
                f"[Stage B]  Trying one_minus_cos_over_z2 on NN vars {target.var_idxs}, "
                f"omega≈{float(omega):.6g}, rel={float(rel):.2e}"
            ),
        }
        sig = (
            int(atom_content_hash(target)),
            stable_int_hash("one_minus_cos_over_z2"),
            int(round(float(omega) * 1.0e6)),
        )
        out.append(Candidate(label="one_minus_cos_over_z2", root=root_new, init_fn=_init, meta=meta, signature=sig))
        if len(out) >= int(max_candidates):
            return out

    for idx, (rel, omega, a0, b0, offset_rel) in enumerate(affine_hits):
        inv_expr_a = PowNode(clone_ast(z_expr), -2.0)
        inv_expr_b = PowNode(clone_ast(z_expr), -2.0)
        arg_ast = _stageB_trig_arg_ast(clone_ast(z_expr), float(omega))
        cos_over_z2 = MulNode(CosNode(arg_ast), inv_expr_b)

        tag_a = f"{base_tag}_omc_z2_a_{idx}"
        tag_b = f"{base_tag}_omc_z2_b_{idx}"
        node_a = _scale_for_feature(inv_expr_a, tag_a, float(a0))
        node_b = _scale_for_feature(cos_over_z2, tag_b, float(b0))
        if node_a is None or node_b is None:
            _diag(f"screen hit but unit-aware scale build failed for affine candidate z={z_label}")
            continue
        new_subtree = AddNode(MulNode(node_a, clone_ast(inv_expr_a)), MulNode(node_b, clone_ast(cos_over_z2)))
        root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
        if root_new is None:
            _diag(f"screen hit but AST replacement failed for affine candidate z={z_label}")
            continue
        tag_a_eff = str(getattr(node_a, "tag", tag_a))
        tag_b_eff = str(getattr(node_b, "tag", tag_b))

        def _init_affine(
            root_new_inner: Node,
            model_new: nn.Module,
            *,
            _tag_a=tag_a_eff,
            _tag_b=tag_b_eff,
            _a=float(a0),
            _b=float(b0),
        ):
            atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
            for atom in _collect_all_atoms(root_new_inner):
                if not isinstance(atom, AtomNode):
                    continue
                tag = str(getattr(atom, "tag", "") or "")
                val = _a if tag == _tag_a else (_b if tag == _tag_b else None)
                if val is None:
                    continue
                leaf = atom_to_leaf.get(id(atom), None)
                if leaf is not None:
                    _set_constant_leaf_value(leaf, float(val))

        _init_affine._after_analytic_init = True
        meta = {
            "structural": True,
            "pattern": "one_minus_cos_over_z2",
            "pattern_family": "one_minus_cos_over_z2",
            "precheck_rel_rms": float(rel),
            "omega": float(omega),
            "one_minus_cos_offset_rel": float(offset_rel),
            "log": (
                f"[Stage B]  Trying affine one_minus_cos_over_z2 on NN vars {target.var_idxs}, "
                f"omega≈{float(omega):.6g}, rel={float(rel):.2e}, "
                f"offset_rel={float(offset_rel):.2e}"
            ),
        }
        sig = (
            int(atom_content_hash(target)),
            stable_int_hash("one_minus_cos_affine_over_z2"),
            int(round(float(omega) * 1.0e6)),
        )
        out.append(Candidate(label="one_minus_cos_affine_over_z2", root=root_new, init_fn=_init_affine, meta=meta, signature=sig))
        if len(out) >= int(max_candidates):
            break

    if not out:
        _diag(f"screen hit but no buildable candidate survived for z={z_label}")
    return out


_SPARSE_FACTOR_1D_BASES = ("1", "z", "1-z", "1+z", "z+2", "1-z^2", "sqrt(1-z^2)")


def _sparse_factor_1d_base_ast(z_expr: Node, key: str) -> Node:
    z = clone_ast(z_expr)
    if key == "1":
        return ConstNode(1.0)
    if key == "z":
        return z
    if key == "1-z":
        return AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), z))
    if key == "1+z":
        return AddNode(ConstNode(1.0), z)
    if key == "z+2":
        return AddNode(z, ConstNode(2.0))
    if key == "1-z^2":
        return AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), PowNode(z, 2.0)))
    if key == "sqrt(1-z^2)":
        inner = AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), PowNode(z, 2.0)))
        return PowNode(inner, 0.5)
    raise ValueError(f"unknown sparse factor base: {key}")


def _sparse_factor_1d_power_ast(z_expr: Node, key: str, exponent: int) -> Optional[Node]:
    e = int(exponent)
    if e == 0 or key == "1":
        return None
    if key == "sqrt(1-z^2)":
        z = clone_ast(z_expr)
        inner = AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), PowNode(z, 2.0)))
        power = 0.5 * float(e)
        if abs(power - 1.0) <= 1.0e-12:
            return inner
        return PowNode(inner, power)
    base = _sparse_factor_1d_base_ast(z_expr, key)
    if e == 1:
        return base
    return PowNode(base, float(e))


def _sparse_factor_1d_feature_ast(z_expr: Node, exponents: Dict[str, int]) -> Optional[Node]:
    out: Optional[Node] = None
    for key in _SPARSE_FACTOR_1D_BASES:
        node = _sparse_factor_1d_power_ast(z_expr, key, int(exponents.get(key, 0)))
        if node is None:
            continue
        out = node if out is None else MulNode(out, node)
    return out


def _sparse_factor_1d_basis_label(exponents: Dict[str, int]) -> str:
    parts: List[str] = []
    for key in _SPARSE_FACTOR_1D_BASES:
        if key == "1":
            continue
        e = int(exponents.get(key, 0))
        if e == 0:
            continue
        if key == "sqrt(1-z^2)" and e % 2 == 0:
            half = e // 2
            parts.append("1-z^2" if half == 1 else f"(1-z^2)^{half}")
        else:
            parts.append(key if e == 1 else f"{key}^{e}")
    return "*".join(parts) if parts else "1"


def _sparse_factor_1d_candidate_exponents() -> List[Dict[str, int]]:
    """Sparse support<=2 exponent vectors matching the requested u^k v^p scan."""
    powers = [p for p in range(-4, 5) if p != 0]
    seen: set[Tuple[int, ...]] = set()
    out: List[Dict[str, int]] = []
    keys = list(_SPARSE_FACTOR_1D_BASES)
    for i in range(len(keys)):
        for p in powers:
            vec = [0] * len(keys)
            vec[i] = int(p)
            sig = tuple(vec)
            if sig in seen:
                continue
            seen.add(sig)
            out.append({k: int(v) for k, v in zip(keys, vec) if int(v) != 0})
        for j in range(i + 1, len(keys)):
            for p in powers:
                for q in powers:
                    vec = [0] * len(keys)
                    vec[i] = int(p)
                    vec[j] = int(q)
                    sig = tuple(vec)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    out.append({k: int(v) for k, v in zip(keys, vec) if int(v) != 0})
    return out


def _build_sparse_factor_1d_candidates(
    ctx: StageBContext,
    target: AtomNode,
    *,
    max_candidates: int = 4,
) -> List[Candidate]:
    """One-scale sparse factors such as ``c*z^4/(1-z^2)^2``.

    This is a deliberately small pre-ratpoly family.  It fits only one scalar
    coefficient after a direct data screen and only uses unit-valid
    dimensionless coordinates in strict-units mode.
    """
    _sync_stageb_rules_compat_overrides()

    def _diag(reason: str) -> None:
        if bool(getattr(ctx, "verbose", False)):
            ctx.log(f"[Stage B]  sparse_factor_1d skip NN vars {getattr(target, 'var_idxs', '?')}: {reason}")

    if not isinstance(target, AtomNode) or effective_arity(target) != 1:
        _diag("target is not effective-1D")
        return []

    z_expr = _stageB_effective_input_expr(target)
    if z_expr is None:
        _diag("no effective coordinate expression")
        return []
    try:
        z_label = ast_to_human_readable(z_expr)
    except Exception:
        z_label = "z"

    if bool(getattr(ctx, "enforce_units", False)):
        z_dim = _stageB_expr_dim(ctx, z_expr)
        dimless = _stageB_dimless(ctx)
        if z_dim is None or dimless is None or tuple(z_dim) != tuple(dimless):
            _diag(f"coordinate is not dimensionless for z={z_label}, dim={z_dim}")
            return []

    try:
        max_points = int(getattr(ctx.lm_hp, "stageB_sparse_factor_1d_max_points", 5000) or 5000)
    except Exception:
        max_points = 5000
    data = _stageB_target_raw_teacher_data(ctx, target, max_points=max_points)
    if data is None:
        _diag("teacher-data screen unavailable")
        return []

    X, y = data
    X = X.detach().to(device=ctx.device, dtype=torch.float64)
    y = y.detach().to(device=ctx.device, dtype=torch.float64).reshape(-1)
    try:
        z = eval_input_expr(z_expr, X).detach().to(device=ctx.device, dtype=torch.float64).reshape(-1)
    except Exception as exc:
        _diag(f"failed to evaluate z={z_label}: {type(exc).__name__}: {exc}")
        return []

    mask0 = torch.isfinite(z) & torch.isfinite(y)
    if int(mask0.sum().item()) < max(200, int(0.95 * z.numel())):
        _diag(f"insufficient finite z/y points: {int(mask0.sum().item())}/{int(z.numel())}")
        return []
    z = z[mask0]
    y = y[mask0]
    n = int(z.numel())
    if n < 200:
        _diag(f"too few screened points: {n}")
        return []

    try:
        max_rel = float(getattr(ctx.lm_hp, "stageB_sparse_factor_1d_screen_rel_rms", 2.0e-2))
    except Exception:
        max_rel = 2.0e-2
    max_rel = _stageB_noisy_rel_rms_threshold(ctx, max_rel, y_values=y)
    try:
        min_finite_frac = float(getattr(ctx.lm_hp, "stageB_sparse_factor_1d_min_finite_frac", 0.95))
    except Exception:
        min_finite_frac = 0.95
    n_need = max(200, int(float(min_finite_frac) * n))

    one = torch.ones_like(z)
    base_vals: Dict[str, Optional[torch.Tensor]] = {
        "1": one,
        "z": z,
        "1-z": one - z,
        "1+z": one + z,
        "z+2": z + 2.0,
        "1-z^2": one - z * z,
    }
    inner = one - z * z
    sqrt_domain = torch.isfinite(inner) & (inner >= -1.0e-12)
    if int(sqrt_domain.sum().item()) >= n_need:
        base_vals["sqrt(1-z^2)"] = torch.sqrt(torch.clamp(inner, min=0.0))
    else:
        base_vals["sqrt(1-z^2)"] = None

    best_pure_monomial_rel = float("inf")
    for power in range(-4, 5):
        if power == 0:
            continue
        finite = torch.isfinite(z)
        if power < 0:
            finite = finite & (torch.abs(z) > 1.0e-10)
        if int(finite.sum().item()) < n_need:
            continue
        zz = z[finite]
        yy = y[finite]
        try:
            feat = zz.pow(int(power)) if power > 0 else torch.reciprocal(zz.pow(int(-power)))
        except Exception:
            continue
        if int(torch.isfinite(feat).sum().item()) < n_need:
            continue
        finite_feat = torch.isfinite(feat)
        feat = feat[finite_feat]
        yy = yy[finite_feat]
        den = torch.dot(feat, feat)
        if (not torch.isfinite(den)) or float(den.item()) <= 1.0e-30:
            continue
        scale = torch.dot(feat, yy) / den
        if not torch.isfinite(scale):
            continue
        rel = _relative_rms(yy, yy - scale * feat)
        if math.isfinite(rel):
            best_pure_monomial_rel = min(best_pure_monomial_rel, float(rel))
    try:
        pure_monomial_rel_floor = float(
            getattr(ctx.lm_hp, "stageB_sparse_factor_1d_pure_monomial_rel_rms", 1.0e-8)
        )
    except Exception:
        pure_monomial_rel_floor = 1.0e-8
    pure_monomial_rel_floor = _stageB_noisy_rel_rms_threshold(
        ctx,
        pure_monomial_rel_floor,
        y_values=y,
    )
    if best_pure_monomial_rel <= float(pure_monomial_rel_floor):
        _diag(
            f"pure monomial screen already exact enough for z={z_label}; "
            f"best rel={best_pure_monomial_rel:.2e}"
        )
        return []

    def _feature_values(exponents: Dict[str, int]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        feat = torch.ones_like(z)
        finite = torch.isfinite(feat)
        for key, exp_raw in exponents.items():
            exp = int(exp_raw)
            if exp == 0 or key == "1":
                continue
            vals = base_vals.get(key)
            if vals is None:
                return None
            vals = vals.to(dtype=torch.float64, device=z.device)
            finite = finite & torch.isfinite(vals)
            if exp < 0:
                finite = finite & (torch.abs(vals) > 1.0e-10)
            try:
                if exp > 0:
                    factor = vals.pow(int(exp))
                else:
                    factor = torch.reciprocal(vals.pow(int(-exp)))
            except Exception:
                return None
            feat = feat * factor
            finite = finite & torch.isfinite(feat)
        if int(finite.sum().item()) < n_need:
            return None
        return feat[finite], y[finite]

    hits: List[Tuple[float, int, float, Dict[str, int], str]] = []
    best_rel = float("inf")
    for exps in _sparse_factor_1d_candidate_exponents():
        nontrivial_keys = {str(k) for k, v in exps.items() if str(k) != "1" and int(v) != 0}
        if not nontrivial_keys:
            continue
        # Leave all pure monomial lanes alone.  Those are handled by the
        # monomial and reciprocal-coordinate monomial rules before this family.
        if nontrivial_keys == {"z"}:
            continue
        values = _feature_values(exps)
        if values is None:
            continue
        feat, yy = values
        den = torch.dot(feat, feat)
        if (not torch.isfinite(den)) or float(den.item()) <= 1.0e-30:
            continue
        scale = torch.dot(feat, yy) / den
        if not torch.isfinite(scale):
            continue
        rel = _relative_rms(yy, yy - scale * feat)
        if math.isfinite(rel):
            best_rel = min(best_rel, float(rel))
        if not math.isfinite(rel) or float(rel) > float(max_rel):
            continue
        label = _sparse_factor_1d_basis_label(exps)
        complexity = int(sum(abs(int(v)) for v in exps.values()) + 2 * max(0, len(exps) - 1))
        hits.append((float(rel), int(complexity), float(scale.item()), dict(exps), label))

    if not hits:
        _diag(f"screen failed for z={z_label}; best rel={best_rel:.2e}, threshold={float(max_rel):.2e}")
        return []

    hits.sort(key=lambda t: (t[0], t[1], t[4]))
    dedup_hits: List[Tuple[float, int, float, Dict[str, int], str]] = []
    seen_labels: set[str] = set()
    for hit in hits:
        if hit[4] in seen_labels:
            continue
        seen_labels.add(hit[4])
        dedup_hits.append(hit)
    hits = dedup_hits

    try:
        target_dim = tuple(ctx.infer_target_dim(target) or ())
        if not target_dim:
            target_dim = None
    except Exception:
        target_dim = None

    if bool(getattr(ctx, "enforce_units", False)) and target_dim is None:
        _diag("cannot infer target units for unit-aware scale")
        return []

    req_scale_dim = target_dim if bool(getattr(ctx, "enforce_units", False)) else None
    base_tag = str(getattr(target, "tag", None) or "leaf")
    out: List[Candidate] = []

    for idx, (rel, complexity, scale_init, exps, basis_label) in enumerate(hits[: max(1, int(max_candidates))]):
        feature_ast = _sparse_factor_1d_feature_ast(z_expr, exps)
        if feature_ast is None:
            continue
        scale_tag = f"{base_tag}_sparse_factor_1d_scale_{idx}"
        try:
            scale_node = _make_unit_aware_scalar_atom(
                req_scale_dim,
                getattr(ctx, "units_spec", None),
                base_tag=scale_tag,
                init=float(scale_init),
                strict=bool(getattr(ctx, "enforce_units", False)),
            )
        except Exception as exc:
            _diag(f"unit-aware scale build failed for {basis_label}: {type(exc).__name__}: {exc}")
            continue
        scale_tag_eff = str(getattr(scale_node, "tag", scale_tag))
        new_subtree = MulNode(scale_node, clone_ast(feature_ast))
        root_new = replace_atom_in_ast(ctx.state.root, target, new_subtree)
        if root_new is None:
            _diag(f"AST replacement failed for {basis_label}")
            continue

        def _init(root_new_inner: Node, model_new: nn.Module, *, _tag=scale_tag_eff, _scale=float(scale_init)):
            atom_to_leaf = build_atom_to_leaf_map(root_new_inner, model_new)
            for atom in _collect_all_atoms(root_new_inner):
                if not isinstance(atom, AtomNode):
                    continue
                if str(getattr(atom, "tag", "") or "") != str(_tag):
                    continue
                leaf = atom_to_leaf.get(id(atom), None)
                if leaf is not None:
                    _set_constant_leaf_value(leaf, float(_scale))

        _init._after_analytic_init = True
        label = "sparse_factor_1d" if idx == 0 else f"sparse_factor_1d[{idx}]"
        sig = (
            int(atom_content_hash(target)),
            stable_int_hash("sparse_factor_1d"),
            stable_int_hash(basis_label),
        )
        meta = {
            "structural": True,
            "exact_non_generic": True,
            "pattern": "sparse_factor_1d",
            "pattern_family": "sparse_factor_1d",
            "sparse_factor_1d": True,
            "sparse_factor_1d_basis": basis_label,
            "sparse_factor_1d_exponents": dict(exps),
            "screen_rel_rms": float(rel),
            "precheck_rel_rms": float(rel),
            "scale_init": float(scale_init),
            "min_free_params": 1,
            "signature": sig,
            "log": (
                f"[Stage B]  Trying sparse_factor_1d {basis_label} on NN vars {target.var_idxs}, "
                f"scale≈{float(scale_init):.4g}, rel={float(rel):.2e}"
            ),
        }
        out.append(Candidate(label=label, root=root_new, init_fn=_init, meta=meta, signature=sig))

    if not out:
        _diag(f"screen hit but no buildable candidate survived for z={z_label}")
    return out

def _has_mul_parent(root: Node, target: Node) -> bool:
    """Return True if *target*'s immediate parent in *root* is a MulNode."""
    if root is target:
        return False
    if isinstance(root, MulNode):
        if root.left is target or root.right is target:
            return True
    for child in (
        [root.left, root.right] if isinstance(root, (AddNode, MulNode)) else
        [root.base] if isinstance(root, PowNode) else
        []
    ):
        if _has_mul_parent(child, target):
            return True
    return False


def _screen_univariate_monomial_target(
    ctx: StageBContext,
    target: AtomNode,
) -> tuple[Optional[MonomialScreenResult], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return the shared monomial screen plus sampled coordinate/teacher values."""

    try:
        tag = getattr(target, "tag", None)
        teacher = ctx.state.reuse.get(tag, None) if isinstance(ctx.state.reuse, dict) else None
        if teacher is None:
            atom_to_leaf = build_atom_to_leaf_map(ctx.state.root, ctx.state.model)
            teacher = atom_to_leaf.get(id(target), None)
        if teacher is None:
            return None, None, None
        data = _gather_atom_teacher_data(
            train_loader=ctx.train_loader_probe,
            atom=target,
            teacher=teacher,
            device=ctx.device,
            dtype=ctx.dtype,
            max_points=int(getattr(ctx.lm_hp, "stageB_monomial_screen_max_points", 5000) or 5000),
        )
        if data is None:
            return None, None, None
        X, F = data
        if X.ndim != 2 or X.shape[1] < 1:
            return None, None, None
        return fit_univariate_monomial_screen(X[:, 0], F), X[:, 0], F
    except Exception as exc:
        if bool(getattr(ctx, "verbose", False)):
            ctx.log(
                f"[Stage B] monomial screen failed for NN vars={getattr(target, 'var_idxs', ())}: {exc}"
            )
        return None, None, None


def _single_effective_input_dim(ctx: StageBContext, target: AtomNode):
    """Infer the units of the single effective coordinate of a 1D target."""

    spec = getattr(ctx, "units_spec", None)
    if spec is None:
        return None
    try:
        inputs = get_input_exprs(target)
        if len(inputs) == 1:
            from nestynet_sr.sr_core.units import eval_analytic_expr_dim

            return eval_analytic_expr_dim(
                inputs[0],
                spec.x_dims,
                free_const_dims=getattr(spec, "free_const_dims", {}) or {},
                fixed_const_dims=getattr(spec, "fixed_const_dims", {}) or {},
            )
        if len(getattr(target, "var_idxs", ())) == 1:
            ax = int(target.var_idxs[0])
            if 0 <= ax < len(spec.x_dims):
                return spec.x_dims[ax]
    except Exception:
        return None
    return None


def _fit_fixed_power_amplitude(x: torch.Tensor, y: torch.Tensor, exponent: float) -> float:
    """Least-squares scalar initialiser for ``y ~= amp * x**exponent``."""

    try:
        xv = x.detach().reshape(-1).to(dtype=torch.float64, device=torch.device("cpu"))
        yv = y.detach().reshape(-1).to(dtype=torch.float64, device=torch.device("cpu"))
        n = min(int(xv.numel()), int(yv.numel()))
        if n <= 0:
            return 1.0
        xv = xv[:n]
        yv = yv[:n]
        mask = torch.isfinite(xv) & torch.isfinite(yv) & (xv > 0)
        if int(mask.sum().item()) <= 0:
            return 1.0
        basis = torch.pow(xv[mask], float(exponent))
        den = torch.sum(basis * basis)
        if not torch.isfinite(den) or float(den.item()) <= 1.0e-30:
            return 1.0
        amp = torch.sum(basis * yv[mask]) / den
        amp_f = float(amp.item())
        return amp_f if math.isfinite(amp_f) else 1.0
    except Exception:
        return 1.0


def _make_fixed_power_monomial_rewrite(
    root: Node,
    target: AtomNode,
    *,
    exponent: Fraction,
    with_scale: bool,
    amp_init: float = 1.0,
) -> Optional[Node]:
    """Replace a 1D NN leaf by a fixed visible ``z**exponent`` monomial."""

    if not isinstance(target, AtomNode):
        return None
    if str(getattr(target, "kind", "")).lower() != "nn" or effective_arity(target) != 1:
        return None
    inputs = get_input_exprs(target)
    if len(inputs) != 1:
        return None
    power_node = PowNode(clone_ast(inputs[0]), float(exponent))
    if with_scale:
        tag_base = str(getattr(target, "tag", None) or "half_power")
        label = monomial_power_label(exponent).replace("[", "_").replace("]", "")
        scale = Scale(
            name=f"{tag_base}_{label}_scale",
            tag=f"{tag_base}_{label}_scale",
            init=float(amp_init),
        )
        new_subtree = MulNode(scale, power_node)
    else:
        new_subtree = power_node
    return replace_atom_in_ast(root, target, new_subtree)


def _half_power_monomial_candidates(
    ctx: StageBContext,
    target: AtomNode,
    *,
    screen: Optional[MonomialScreenResult],
    x_sample: Optional[torch.Tensor],
    y_sample: Optional[torch.Tensor],
    d_req,
    input_dim,
    with_scale: bool,
) -> List[Candidate]:
    """Build fixed half-power candidates only for very clean log-slope evidence."""

    if screen is None or not screen.ok:
        return []
    max_rel = _stageB_noisy_rel_rms_threshold(
        ctx,
        _HALF_POWER_SCREEN_REL_RMS_MAX,
        y_values=y_sample,
    )
    if not math.isfinite(float(screen.rel_rms)) or float(screen.rel_rms) > max_rel:
        return []
    snapped = snap_to_half_integer_monomial_power(screen.k_hat)
    if snapped is None:
        return []
    if x_sample is None:
        return []
    ok_domain, domain_reason = half_power_domain_ok(x_sample, y_sample)
    if not ok_domain:
        if bool(getattr(ctx, "verbose", False)):
            ctx.log(
                f"[Stage B] half-power monomial screen rejected for NN vars={target.var_idxs}: "
                f"{domain_reason}"
            )
        return []

    if d_req is not None and input_dim is not None and _scale_dim is not None:
        try:
            if _scale_dim(input_dim, snapped) != d_req:
                return []
        except Exception:
            return []

    # Reciprocal aliases turn negative powers into positive powers on 1/z.
    if float(screen.k_hat) < 0:
        return []

    amp_init = _fit_fixed_power_amplitude(x_sample, y_sample, float(snapped)) if with_scale and y_sample is not None else 1.0
    root_hp = _make_fixed_power_monomial_rewrite(
        ctx.state.root,
        target,
        exponent=snapped,
        with_scale=bool(with_scale),
        amp_init=amp_init,
    )
    if root_hp is None:
        return []
    label = monomial_power_label(snapped)
    return [
        Candidate(
            label,
            root_hp,
            meta={
                "log": (
                    f"[Stage B]  Trying fixed half-power monomial "
                    f"(z^{float(snapped):.3g}, k≈{float(screen.k_hat):.3g}, "
                    f"rel={float(screen.rel_rms):.3g}) on NN leaf vars {target.var_idxs}"
                ),
                "monomial_fixed_power": float(snapped),
                "monomial_screen_k": float(screen.k_hat),
                "monomial_screen_rel_rms": float(screen.rel_rms),
                "structural": True,
            },
        )
    ]


def _screen_integer_power_monomial_candidates(
    ctx: StageBContext,
    target: AtomNode,
    *,
    screen: Optional[MonomialScreenResult],
    x_sample: Optional[torch.Tensor],
    y_sample: Optional[torch.Tensor],
    d_req,
    input_dim,
    with_scale: bool,
) -> List[Candidate]:
    """Build exact small-integer monomials beyond the fixed cheap menu."""

    if screen is None or not screen.ok:
        return []
    max_rel = _stageB_noisy_rel_rms_threshold(
        ctx,
        _INTEGER_POWER_SCREEN_REL_RMS_MAX,
        y_values=y_sample,
    )
    if not math.isfinite(float(screen.rel_rms)) or float(screen.rel_rms) > max_rel:
        return []
    snapped = snap_to_integer_monomial_power(
        screen.k_hat,
        max_power=_INTEGER_POWER_SCREEN_MAX_POWER,
    )
    if snapped is None:
        return []
    if int(snapped) in set(int(d) for d in _MONOMIAL_DEGREES):
        return []

    # Reciprocal aliases turn negative powers into positive powers on 1/z.
    if float(screen.k_hat) < 0:
        return []

    if d_req is not None and input_dim is not None and _scale_dim is not None:
        try:
            if _scale_dim(input_dim, snapped) != d_req:
                return []
        except Exception:
            return []

    amp_init = _fit_fixed_power_amplitude(x_sample, y_sample, float(snapped)) if with_scale and x_sample is not None and y_sample is not None else 1.0
    root_power = _make_fixed_power_monomial_rewrite(
        ctx.state.root,
        target,
        exponent=snapped,
        with_scale=bool(with_scale),
        amp_init=amp_init,
    )
    if root_power is None:
        return []
    label = monomial_power_label(snapped)
    return [
        Candidate(
            label,
            root_power,
            meta={
                "log": (
                    f"[Stage B]  Trying fixed integer-power monomial "
                    f"(z^{int(snapped)}, k≈{float(screen.k_hat):.3g}, "
                    f"rel={float(screen.rel_rms):.3g}) on NN leaf vars {target.var_idxs}"
                ),
                "monomial_fixed_power": float(snapped),
                "monomial_screen_k": float(screen.k_hat),
                "monomial_screen_rel_rms": float(screen.rel_rms),
                "structural": True,
            },
        )
    ]


class RuleUniNN(StageBRule):
    """
    Rule for univariate NN atom rewrites.

    Identifies 1D NN leaves and proposes various analytic rewrites:
    - Scaling (x^k) based on detected homogeneity
    - Planck distribution (x^k / (exp(x) - 1))
    - Power-exponential (x^k * exp(poly))
    - Polynomial, log-polynomial
    - Exponential-polynomial, exponential-rational
    - Trigonometric (sin/cos with detected frequency)

    Pattern labels: scale, planck, 1d_powexp, poly, polylog, log_poly, log_ratpoly, exp, exp_rat, trig
    """

    name = "univariate_nn"
    exhaustive = True
    multi_probe_native = True

    def __init__(self, factorized_search_rule=None, monomial_only=False):
        self.factorized_search_rule = factorized_search_rule
        self.monomial_only = monomial_only
        if monomial_only:
            self.name = "univariate_mono"

    def iter_targets(self, ctx: StageBContext):
        """Return all univariate (1D) NN atoms in the current AST."""
        return _collect_univariate_nn_atoms(ctx.state.root)

    def _can_reach_target_dim_with_inputs(self, ctx: StageBContext, target: AtomNode, d_req) -> bool:
        """Return True when the leaf inputs/constants can span the required output units."""
        if d_req is None or _dim_in_rational_span is None:
            return True
        spec = getattr(ctx, "units_spec", None)
        if spec is None:
            return True
        try:
            basis_dims = []
            from nestynet_sr.sr_core.units import eval_analytic_expr_dim

            for inp in get_input_exprs(target):
                dim = eval_analytic_expr_dim(
                    inp,
                    spec.x_dims,
                    free_const_dims=getattr(spec, "free_const_dims", {}) or {},
                    fixed_const_dims=getattr(spec, "fixed_const_dims", {}) or {},
                )
                if dim is not None:
                    basis_dims.append(dim)
            basis_dims.extend(list((getattr(spec, "free_const_dims", {}) or {}).values()))
            basis_dims.extend(list((getattr(spec, "fixed_const_dims", {}) or {}).values()))
            return bool(_dim_in_rational_span(d_req, basis_dims))
        except Exception:
            # Never let a pruning heuristic block a candidate on inference failure.
            return True

    def _propose_reciprocal_coordinate_alias(self, ctx: StageBContext, target: AtomNode) -> List[Candidate]:
        """Reuse univariate rewrite families on the reciprocal effective coordinate."""
        if bool(getattr(ctx, "_stageB_coord_alias_active", False)):
            return []
        if not isinstance(target, AtomNode) or effective_arity(target) != 1:
            return []
        tag = getattr(target, "tag", None)
        if tag is None:
            return []

        z_expr = compound_input_expr(target)
        if z_expr is None:
            return []

        reuse_override = _wrap_reuse_for_reciprocal_coordinate(ctx.state.reuse, tag)
        if reuse_override is None:
            return []

        reuses_override = None
        state_reuses = getattr(ctx.state, "reuses", None)
        if isinstance(state_reuses, (list, tuple)):
            wrapped_reuses = []
            ok_all = True
            for reuse_i in state_reuses:
                wrapped = _wrap_reuse_for_reciprocal_coordinate(reuse_i, tag)
                if wrapped is None:
                    ok_all = False
                    break
                wrapped_reuses.append(wrapped)
            if ok_all:
                reuses_override = wrapped_reuses

        alias_target = AtomNode(
            kind="nn",
            var_idxs=tuple(int(j) for j in target.var_idxs),
            kwargs=copy.deepcopy(getattr(target, "kwargs", {}) or {}),
            tag=tag,
            inputs=(PowNode(z_expr, -1.0),),
        )
        alias_root = replace_atom_in_ast(ctx.state.root, target, alias_target)
        if alias_root is None:
            return []

        alias_state = copy.copy(ctx.state)
        alias_state.root = alias_root
        alias_state.reuse = reuse_override
        if reuses_override is not None:
            alias_state.reuses = reuses_override

        alias_ctx = copy.copy(ctx)
        alias_ctx.state = alias_state
        alias_ctx._stageB_coord_alias_active = True
        for cache_name in ("_cache", "_dim_cache"):
            if hasattr(alias_ctx, cache_name):
                try:
                    setattr(alias_ctx, cache_name, {})
                except Exception:
                    pass
        if hasattr(alias_ctx, "_dim_cache_root_id"):
            try:
                alias_ctx._dim_cache_root_id = None
            except Exception:
                pass

        alias_rule = RuleUniNN(factorized_search_rule=None, monomial_only=self.monomial_only)
        try:
            alias_cands = alias_rule.propose(alias_ctx, alias_target) or []
        except Exception as exc:
            if bool(getattr(ctx, "verbose", False)):
                ctx.log(f"[Stage B]  reciprocal-coordinate alias proposal failed: {exc}")
            return []

        out: List[Candidate] = []
        for cand in alias_cands:
            if cand is None:
                continue
            repeat_reason = _reciprocal_alias_repeat_reason(cand)
            if repeat_reason is not None:
                if bool(getattr(ctx, "verbose", False)):
                    ctx.log(
                        f"[Stage B]  Skipping reciprocal-coordinate alias {cand.label}: "
                        f"{repeat_reason}"
                    )
                continue
            out.append(
                _mark_reciprocal_coordinate_candidate(
                    cand,
                    reuse_override=reuse_override,
                    reuses_override=reuses_override,
                )
            )
        return out

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        """
        Generate rewrite candidates for a univariate NN target.

        Args:
            ctx: Stage B context
            target: Univariate NN atom

        Returns:
            List of Candidate rewrites (prioritized by strength of hints)
        """
        _sync_stageb_rules_compat_overrides()
        if (
            not isinstance(target, AtomNode)
            or str(target.kind).lower() != "nn"
            or effective_arity(target) != 1
        ):
            return []

        st = ctx.state
        reuse = st.reuse if isinstance(st.reuse, dict) else {}

        # Shared wrapper-policy gate for Stage-B trig candidates.
        # Use the same policy layer as compound-function macros so a global
        # "disable trig" setting applies consistently across Stage B.
        trig_policy = None
        trig_enabled = True
        try:
            trig_policy = macro_arg_wrapper_policy(ctx, ctx.lm_hp, target)
            trig_enabled = bool(getattr(trig_policy, "trig", True))
        except Exception:
            trig_policy = None
            trig_enabled = True

        is_nontrivial = has_nontrivial_input(target)

        # Nontrivial input expressions don't have a meaningful single axis for hint lookup
        if is_nontrivial:
            axis = None
            spec_scale = None
            # For compound atoms, look for trig hints on underlying variables.
            # The omega detected on individual axes applies to the compound z too.
            spec_trig = None
            if getattr(ctx, "trig_by_axis", None):
                for vi in target.var_idxs:
                    vi_spec = ctx.trig_by_axis.get(int(vi), None)
                    if vi_spec is not None and math.isfinite(float(vi_spec.omega)) and float(vi_spec.omega) > 0:
                        spec_trig = vi_spec
                        break
        else:
            axis = int(target.var_idxs[0])
            spec_scale = _best_scale_spec_for_axis(ctx.scaling_by_axis, axis)
            spec_trig = ctx.trig_by_axis.get(axis, None)
        strong_scale_hint = _is_strong_scaling_spec(spec_scale) if spec_scale is not None else False

        cands: List[Candidate] = []
        trig_added = False

        # Dimensional pruning: infer required output dimension once.
        d_req = ctx.infer_target_dim(target)
        _skip_dimless = (d_req is not None) and (_is_dimless is not None) and (not _is_dimless(d_req))

        # For monomial degree pruning on simple (non-compound) atoms
        _x_input_dim = None
        if d_req is not None and (not is_nontrivial):
            try:
                _ax = int(target.var_idxs[0])
                _spec = getattr(ctx, "units_spec", None)
                if _spec is not None and 0 <= _ax < len(_spec.x_dims):
                    _x_input_dim = _spec.x_dims[_ax]
            except Exception:
                pass
        _half_power_input_dim = _single_effective_input_dim(ctx, target) if d_req is not None else None
        _monomial_screen = None
        _monomial_x = None
        _monomial_y = None

        # 1a. Prior: Stage-A trig hint — if Stage A substituted this axis with a trig
        # function (e.g. x1 -> cos(x1)), treat that as a strong prior and try a trig leaf early.
        stageA_tr = None
        if (axis is not None) and (not is_nontrivial):
            try:
                # If Stage B is already operating in internal x-coordinates, avoid double trig.
                if not bool(getattr(ctx, "xcoords_applied", False)):
                    stageA_tr = (getattr(ctx, "stageA_x_transforms", None) or {}).get(axis, None)
            except Exception:
                stageA_tr = None

        stageA_fn = ""
        omega0 = 1.0
        shift0 = 0.0
        if isinstance(stageA_tr, dict):
            if "pipeline" in stageA_tr and isinstance(stageA_tr.get("pipeline"), list):
                # New-style xcoord pipeline spec (Stage A may store canonical form).
                pipe = stageA_tr.get("pipeline")
                # Find the last trig step.
                trig_step = None
                trig_idx = None
                for i in range(len(pipe) - 1, -1, -1):
                    k = str(pipe[i].get("kind", "")).lower().strip()
                    if k in ("sin", "cos", "tan"):
                        trig_step = pipe[i]
                        trig_idx = i
                        break
                if trig_step is not None:
                    stageA_fn = str(trig_step.get("kind", "")).lower().strip()
                    # Extract a representative (shift, omega) from preceding shift/scale steps.
                    for j in range(0, int(trig_idx)):
                        kj = str(pipe[j].get("kind", "")).lower().strip()
                        if kj == "shift":
                            try:
                                shift0 = float(pipe[j].get("shift", 0.0))
                            except Exception:
                                pass
                        if kj == "scale":
                            try:
                                omega0 = float(pipe[j].get("scale", 1.0))
                            except Exception:
                                pass

        if trig_enabled and stageA_fn in ("sin", "cos", "tan") and not _skip_dimless:
            phase0 = (math.pi / 2) if stageA_fn == "cos" else 0.0
            bias0 = phase0 - omega0 * shift0

            root_trig_hint = replace_atom_in_ast(
                st.root,
                target,
                AtomNode(kind="sin_linear", var_idxs=target.var_idxs, kwargs={}, tag=target.tag),
            )
            if root_trig_hint is not None:

                def _init_trig_from_stageA(root_new, model_new, *, _tag=target.tag, _omega=omega0, _bias=bias0):
                    atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
                    for _atom in _collect_all_atoms(root_new):
                        if isinstance(_atom, AtomNode) and (_atom.tag == _tag) and (str(_atom.kind).lower() == "sin_linear"):
                            leaf = atom_to_leaf.get(id(_atom), None)
                            if leaf is None:
                                continue
                            try:
                                with torch.no_grad():
                                    if hasattr(leaf, "weight") and leaf.weight.numel() >= 1:
                                        leaf.weight.zero_()
                                        leaf.weight.view(-1)[0] = float(_omega)
                                    if hasattr(leaf, "bias") and leaf.bias.numel() >= 1:
                                        leaf.bias.fill_(float(_bias))
                                    if hasattr(leaf, "amp") and leaf.amp.numel() >= 1:
                                        leaf.amp.fill_(1.0)
                            except Exception:
                                pass

                _init_trig_from_stageA._after_analytic_init = True
                cands.append(
                    Candidate(
                        "trig",
                        root_trig_hint,
                        init_fn=_init_trig_from_stageA,
                        meta={
                            "log": f"[Stage B]  Trying trig rewrite (Stage-A x-transform hint: {stageA_fn}, ω≈{omega0:.3g}) on NN leaf vars {target.var_idxs}"
                        },
                    )
                )
                trig_added = True

        # 1b. Prior: strong scaling hint (prioritize)
        if strong_scale_hint and spec_scale is not None:
            root_scale = _make_scaling_based_rewrite(st.root, target, spec_scale)
            if root_scale is not None:
                cands.append(
                    Candidate(
                        "scale",
                        root_scale,
                        meta={
                            "log": f"[Stage B]  Trying scaling rewrite (strong hint k≈{float(spec_scale.k_hat):.3f}) on NN leaf vars {target.var_idxs}"
                        },
                    )
                )

        # 1c. Fixed half-integer monomials.  These are deliberately not a
        # free-power fit: they are only proposed when the existing log-slope
        # evidence is essentially exact and the sampled coordinate is safely
        # in the real half-power domain.  Negative powers are supplied by the
        # reciprocal-coordinate alias pass.
        _monomial_screen, _monomial_x, _monomial_y = _screen_univariate_monomial_target(ctx, target)
        cands.extend(
            _half_power_monomial_candidates(
                ctx,
                target,
                screen=_monomial_screen,
                x_sample=_monomial_x,
                y_sample=_monomial_y,
                d_req=d_req,
                input_dim=_half_power_input_dim,
                with_scale=not (self.monomial_only and _has_mul_parent(st.root, target)),
            )
        )
        cands.extend(
            _screen_integer_power_monomial_candidates(
                ctx,
                target,
                screen=_monomial_screen,
                x_sample=_monomial_x,
                y_sample=_monomial_y,
                d_req=d_req,
                input_dim=_half_power_input_dim,
                with_scale=not (self.monomial_only and _has_mul_parent(st.root, target)),
            )
        )

        # 2a. Pure monomials of increasing degree (Occam's razor: simplest first).
        #
        # When monomial_only=True, positive-degree monomials with a MulNode
        # parent use rpoly (zero free params, literally x^k).  The sibling
        # NN atoms absorb the gauge during LM refit.  Isolated additive
        # atoms keep poly (trainable c·x^k).
        # When monomial_only=False, this always uses poly — the downstream
        # gauge_fix_multiplicative pass handles rpoly+Scale.
        for mono_deg in _MONOMIAL_DEGREES:
            # Dimensional pruning: skip degrees whose units can't match d_req.
            if d_req is not None and _x_input_dim is not None and _scale_dim is not None:
                from fractions import Fraction
                if _scale_dim(_x_input_dim, Fraction(mono_deg)) != d_req:
                    continue
            root_mono = _make_poly_1d_rewrite(
                st.root, target, degree=mono_deg, min_total=mono_deg,
                rpoly=self.monomial_only and _has_mul_parent(st.root, target),
            )
            init_mono = None
            cand_name = f"monomial_deg{mono_deg}"
            log_msg = f"[Stage B]  Trying 1D monomial (c·x^{mono_deg}) rewrite on NN leaf vars {target.var_idxs}"
            if root_mono is not None:
                cands.append(
                    Candidate(
                        cand_name,
                        root_mono,
                        init_mono,
                        meta={"log": log_msg},
                    )
                )

        # Early exit: monomial-only mode returns just the cheap monomial candidates.
        if self.monomial_only:
            cands = _merge_reciprocal_aliases_pairwise(
                cands, self._propose_reciprocal_coordinate_alias(ctx, target)
            )
            return cands

        # 2b. Linear polynomial with constant (if pure monomials don't suffice)
        if not _skip_dimless:
            root_poly = _make_poly_1d_rewrite(st.root, target, degree=1, min_total=0)
        else:
            root_poly = None
        if root_poly is not None:
            cands.append(
                Candidate(
                    "poly",
                    root_poly,
                    meta={
                        "log": f"[Stage B]  Trying 1D poly rewrite on NN leaf vars {target.var_idxs}, degree=1"
                    },
                )
            )

        # 2c. factorized symbolic search explorer: lazy group (see RuleMultiDNN for details).
        can_try_factorized_search = True
        if (
            self.factorized_search_rule is not None
            and d_req is not None
            and getattr(ctx, "enforce_units", False)
        ):
            can_try_factorized_search = self._can_reach_target_dim_with_inputs(ctx, target, d_req)
            if not can_try_factorized_search:
                ctx.log(
                    f"[Stage B]  Skipping factorized symbolic search fallback for NN vars={target.var_idxs}: "
                    f"target units {d_req} are unreachable from leaf inputs/constants"
                )

        if self.factorized_search_rule is not None and can_try_factorized_search:
            _bsr_cache_u: Dict[str, Any] = {}

            def _bsr_get_u(idx: int, _cache=_bsr_cache_u):
                if "cands" not in _cache:
                    _cache["cands"] = self.factorized_search_rule.propose(ctx, target) or []
                cl = _cache["cands"]
                if idx >= len(cl):
                    return None
                c = cl[idx]
                meta = dict(c.meta) if c.meta else {}
                meta["_label"] = c.label
                return (c.root, c.init_fn, meta)

            _max_bsr_u = getattr(self.factorized_search_rule, "return_topk", 5) * 2
            for _bi in range(_max_bsr_u):
                cands.append(Candidate(
                    label=f"factorized_search[{_bi}]",
                    builder=lambda _idx=_bi: _bsr_get_u(_idx),
                ))

        # 3a. Planck distribution
        if not _skip_dimless:
            planck_candidates = _build_planck_1d_candidates(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
            )
            for planck_label, planck_root, planck_init, planck_meta in planck_candidates:
                meta = dict(planck_meta or {})
                p_sig = int(round(float(meta.get("planck_power", 0.0)) * 1000.0))
                cands.append(
                    Candidate(
                        planck_label,
                        planck_root,
                        planck_init,
                        meta=meta,
                        signature=(int(atom_content_hash(target)), 43007, p_sig),
                    )
                )

            planck_full_root, planck_full_init = _build_planck_full_1d_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
            )
            if planck_full_root is not None:
                cands.append(
                    Candidate(
                        "planck_full",
                        planck_full_root,
                        planck_full_init,
                        meta={
                            "min_free_params": 4,
                            "log": f"[Stage B]  Trying full 1D Planck rewrite on NN leaf vars {target.var_idxs}"
                        },
                    )
                )

        # 3b. Expm1: amp * (exp(a*z + b) - 1)
        if not _skip_dimless:
            expm1_root, expm1_init = _build_expm1_1d_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
            )
            if expm1_root is not None:
                cands.append(
                    Candidate(
                        "expm1",
                        expm1_root,
                        expm1_init,
                        meta={
                            "log": f"[Stage B]  Trying 1D Expm1 rewrite on NN leaf vars {target.var_idxs}"
                        },
                    )
                )

        # 3c. Symmetric exponential denominator: scale / (exp(r(z)) + exp(-r(z)))
        # Good for sech-like forms: 1/(exp(z)+exp(-z)), 1/(exp(1/z)+exp(-1/z))
        # Returns multiple candidates: standard r(z) and reciprocal r(1/z) for compound atoms
        if not _skip_dimless:
            symexp_candidates = _build_symexp_denom_1d_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                units_spec=ctx.units_spec,
            )
            for symexp_root, symexp_init, symexp_label in symexp_candidates:
                cands.append(
                    Candidate(
                        symexp_label,
                        symexp_root,
                        symexp_init,
                        meta={
                            "log": f"[Stage B]  Trying 1D {symexp_label} rewrite (scale/(exp(r)+exp(-r))) on NN leaf vars {target.var_idxs}"
                        },
                    )
                )

        # 3d. Power * exp-poly
        if not _skip_dimless:
            powexp_root, powexp_init = _build_power_exp_1d_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
            )
            if powexp_root is not None:
                cands.append(
                    Candidate(
                        "1d_powexp",
                        powexp_root,
                        powexp_init,
                        meta={
                            "log": f"[Stage B]  Trying 1D power*exp-poly rewrite on NN leaf vars {target.var_idxs}"
                        },
                    )
                )

        # 4a. Poly-log (polynomial in log(x))
        if not _skip_dimless:
            root_polylog = _make_polylog_1d_rewrite(st.root, target, degree=1)
            if root_polylog is not None:
                cands.append(
                    Candidate(
                        "polylog",
                        root_polylog,
                        meta={
                            "log": f"[Stage B]  Trying 1D polylog rewrite on NN leaf vars {target.var_idxs}, degree=1"
                        },
                    )
                )

        # 4b. Shifted-log (for log(x-b) patterns like log(x-1))
        if not _skip_dimless:
            root_logshifted = _make_logshifted_1d_rewrite(st.root, target)
            if root_logshifted is not None:
                cands.append(
                    Candidate(
                        "logshifted",
                        root_logshifted,
                        meta={
                            "log": f"[Stage B]  Trying 1D shifted-log rewrite on NN leaf vars {target.var_idxs}"
                        },
                    )
            )

        # 4c. 1D inverse polynomial: 1/P(x)  (handles e.g. 1/(x+1), 1/(1+x²))
        # Extract dimensional info once for all 1D poly/ratpoly builders.
        _us_u = getattr(ctx, "units_spec", None)
        _eu_u = getattr(ctx, "enforce_units", False)
        _inv_homo_u = False
        _ratpoly_1d_td = None
        _ratpoly_1d_xd = None
        if _eu_u and _us_u is not None:
            _inv_homo_u, _ratpoly_1d_td, _ratpoly_1d_xd = _prepare_univariate_units_probe(
                ctx, target, _us_u
            )

        if not _skip_dimless and (not _ctx_pattern_disabled(ctx, "inv_poly")):
            inv_poly_results = _build_inv_poly_candidates(
                root=st.root,
                target=target,
                train_loader=ctx.train_loader_probe,
                reuse=reuse,
                device=ctx.device,
                dtype=ctx.dtype,
                max_degree=2,
                rel_rms_threshold=1.0e-3,
                homogeneous=_inv_homo_u,
                target_dim=_ratpoly_1d_td,
                x_dims=_ratpoly_1d_xd,
            )
            for ip_i, (ip_root, ip_init, ip_meta) in enumerate(inv_poly_results):
                cands.append(
                    Candidate(
                        "inv_poly" if ip_i == 0 else f"inv_poly[{ip_i}]",
                        ip_root,
                        ip_init,
                        meta=ip_meta,
                    )
                )

        # 5a. 1D sqrt / invsqrt of a low-degree polynomial.
        #
        # Unlike exp/log/trig/Planck-style families, sqrt(poly) can be a
        # perfectly unitful closure: e.g. dim(z)=L^2, dim(NN)=L.  The builder
        # receives the effective-input dimensions and restricts the radicand
        # basis in strict unit mode, so do not hide this behind the broad
        # dimensionless-output gate.
        if not _ctx_pattern_disabled(ctx, "sqrt_poly"):
            sqrt_poly_root, sqrt_poly_init = _build_sqrt_poly_candidate(
                root=st.root,
                target=target,
                train_loader=ctx.train_loader_probe,
                reuse=reuse,
                device=ctx.device,
                dtype=ctx.dtype,
                degree=2,
                rel_rms_threshold=1.0e-3,
                noise_floor_raw=_stageB_noise_floor_raw(ctx),
                homogeneous=_inv_homo_u,
                target_dim=_ratpoly_1d_td,
                x_dims=_ratpoly_1d_xd,
            )
            if sqrt_poly_root is not None:
                cands.append(
                    Candidate(
                        "sqrt_poly",
                        sqrt_poly_root,
                        sqrt_poly_init,
                        meta={
                            "log": f"[Stage B]  Trying sqrt/invsqrt(poly) rewrite on NN leaf vars {target.var_idxs}, degree=2"
                        },
                    )
                )

        # 5a'. Compound-function macros for effective-1D leaves.
        #
        # If Stage A has already compressed a multivariate pattern to
        # NN[z(x)], the multivariate macro rule may no longer see the original
        # variable pair.  Reuse the same macro library here before generic
        # rational-polynomial fallbacks so forms such as
        # sqrt(1 - 1/z**2)/(1 - 1/z) remain first-class exact closures.
        if (not _ctx_pattern_disabled(ctx, "compound_fn_macros")) and has_nontrivial_input(target):
            try:
                from nestynet_sr.sr_search.compound_functions import propose_compound_function_macros

                macro_cands = list(propose_compound_function_macros(ctx, target) or [])
                for mc in macro_cands:
                    if mc is None:
                        continue
                    meta = dict(mc.meta) if isinstance(getattr(mc, "meta", None), dict) else {}
                    label = str(getattr(mc, "label", ""))
                    if "log" not in meta:
                        meta["log"] = (
                            f"[Stage B]  Trying 1D compound macro {label} "
                            f"on NN leaf vars {target.var_idxs}"
                        )
                    else:
                        meta["log"] = str(meta["log"]).replace(
                            "[Stage B]  Trying ",
                            "[Stage B]  Trying 1D compound macro ",
                            1,
                        )
                    meta["pattern_family"] = label or "compound_fn_macros"
                    meta["compound_1d_macro"] = True
                    cands.append(
                        Candidate(
                            label=label,
                            root=mc.root,
                            init_fn=mc.init_fn,
                            builder=mc.builder,
                            meta=meta,
                            signature=getattr(mc, "signature", None),
                        )
                    )
            except Exception as exc:
                if bool(getattr(ctx, "verbose", False)):
                    ctx.log(f"[Stage B]  1D compound macro proposal failed: {type(exc).__name__}: {exc}")

        # 5a''. Exact one-minus-cos-over-z^2 closure.
        #
        # This is the low-parameter sinc-square counterpart needed for leaves
        # such as sin(z/2)^2/z^2.  Try it before generic 1D rational families
        # whenever a strong trig hint exists; the direct screen and normal
        # Stage-B validation still decide acceptance.
        if trig_enabled and spec_trig is not None:
            cands.extend(_build_one_minus_cos_over_z2_candidates(ctx, target, spec_trig))
            cands.extend(_build_fixed_trig_factor_candidates(ctx, target, spec_trig))
        elif trig_enabled and bool(getattr(ctx, "verbose", False)):
            ctx.log(f"[Stage B]  one_minus_cos_over_z2 skip NN vars {target.var_idxs}: no trig hint")

        # 5a'''. Sparse one-scale factor products.
        #
        # This is the physics-style exact lane for forms such as
        # c*z^4/(1-z^2)^2.  It remains much narrower than ratpoly_1d: a direct
        # screen chooses a sparse support<=2 product over unit-valid
        # dimensionless factors, and the candidate carries only one fitted
        # scale before normal Stage-B validation.
        if not _skip_dimless and (not _ctx_pattern_disabled(ctx, "sparse_factor_1d")):
            cands.extend(_build_sparse_factor_1d_candidates(ctx, target))

        # 5b. 1D sqrt / invsqrt of a rational polynomial.
        #
        # This is the effective-1D counterpart of the multivariate
        # sqrt_ratpoly rule.  Greedy Stage A can expose a scalar compound
        # coordinate z(x), in which case sqrt(P(z)/Q(z)) should be tried before
        # the generic P(z)/Q(z) fallback but after exact compound macros.
        if (
            (not _ctx_pattern_disabled(ctx, "sqrt_ratpoly"))
            and (not _ctx_pattern_disabled(ctx, "sqrt_ratpoly_1d"))
        ):
            sqrt_ratpoly_1d_results = _build_sqrt_ratpoly_1d_candidates(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                enforce_units=_eu_u,
                target_dim=_ratpoly_1d_td,
                x_dims=_ratpoly_1d_xd,
            )
            for sr_i, (sr_root, sr_init, sr_meta) in enumerate(sqrt_ratpoly_1d_results):
                cands.append(
                    Candidate(
                        "sqrt_ratpoly_1d" if sr_i == 0 else f"sqrt_ratpoly_1d[{sr_i}]",
                        sr_root,
                        sr_init,
                        meta=sr_meta,
                    )
                )

        # 5c. 1D Rational polynomial (P(x)/Q(x))
        if not _ctx_pattern_disabled(ctx, "ratpoly_1d"):
            ratpoly_1d_results = _build_ratpoly_1d_candidates(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                enforce_units=_eu_u,
                target_dim=_ratpoly_1d_td,
                x_dims=_ratpoly_1d_xd,
            )
            for rp_i, (rp_root, rp_init, rp_meta) in enumerate(ratpoly_1d_results):
                cands.append(
                    Candidate(
                        "ratpoly_1d" if rp_i == 0 else f"ratpoly_1d[{rp_i}]",
                        rp_root,
                        rp_init,
                        meta=rp_meta,
                    )
                )

        # 5c. Exponential-polynomial
        if not _skip_dimless:
            root_exp = _make_exp_poly_rewrite(st.root, target, degree=2)
            if root_exp is not None:
                cands.append(
                    Candidate(
                        "exp",
                        root_exp,
                        meta={
                            "log": f"[Stage B]  Trying exp-poly rewrite on NN leaf vars {target.var_idxs}, degree=2"
                        },
                    )
                )

        # 5d. Exponential-rational
        if not _skip_dimless:
            root_exp_rat = _make_exp_ratpoly_rewrite(st.root, target, deg_num=2, deg_den=2)
            if root_exp_rat is not None:
                cands.append(
                    Candidate(
                        "exp_rat",
                        root_exp_rat,
                        meta={
                            "log": f"[Stage B]  Trying exp-ratpoly rewrite on NN leaf vars {target.var_idxs}, degs=2/2"
                        },
                    )
                )

        # 5e. Log of polynomial: log(poly(x))
        if not _skip_dimless:
            log_poly_root, log_poly_init = _build_log_poly_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                degree=2,
                homogeneous=_inv_homo_u,
            )
            if log_poly_root is not None:
                cands.append(
                    Candidate(
                        "log_poly",
                        log_poly_root,
                        log_poly_init,
                        meta={
                            "log": f"[Stage B]  Trying log(poly) rewrite on NN leaf vars {target.var_idxs}, degree=2"
                        },
                    )
                )

        # 5f. Log of rational polynomial: log(P(x)/Q(x))
        if not _skip_dimless:
            log_ratpoly_root, log_ratpoly_init = _build_log_ratpoly_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                enforce_units=_eu_u,
                target_dim=_ratpoly_1d_td,
                x_dims=_ratpoly_1d_xd,
            )
            if log_ratpoly_root is not None:
                cands.append(
                    Candidate(
                        "log_ratpoly",
                        log_ratpoly_root,
                        log_ratpoly_init,
                        meta={
                            "log": f"[Stage B]  Trying log(ratpoly) rewrite on NN leaf vars {target.var_idxs}"
                        },
                    )
                )

        # 6. Prior: weak scaling hint
        if (spec_scale is not None) and (not strong_scale_hint):
            root_scale = _make_scaling_based_rewrite(st.root, target, spec_scale)
            if root_scale is not None:
                cands.append(
                    Candidate(
                        "scale",
                        root_scale,
                        meta={
                            "log": f"[Stage B]  Trying scaling rewrite on NN leaf vars {target.var_idxs}, k≈{float(spec_scale.k_hat):.3f}"
                        },
                    )
                )

        # 7. Hyperbolic tangent (bounded nonlinearity)
        if not _skip_dimless:
            root_tanh = _make_tanh_based_rewrite(st.root, target)
            if root_tanh is not None:
                cands.append(
                    Candidate(
                        "tanh",
                        root_tanh,
                        meta={
                            "log": f"[Stage B]  Trying tanh rewrite on NN leaf vars {target.var_idxs}"
                        },
                    )
                )

        # 8a. Trigonometric rewrite (unified for plain and compound atoms)
        # Plain variables are treated as compound variables with identity input_expr,
        # so _estimate_trig_params_on_compound works for both.  spec_trig (from FFT
        # detection on individual axes) is used as an optional extra omega source.
        class _SyntheticTrigSpec:
            """Minimal stand-in for TrigAxisSpec when no FFT hint is available."""
            def __init__(self, omega, axis, strength=1000.0):
                self.omega = omega
                self.axis = axis
                self.strength = strength

        input_expr_trig = compound_input_expr(target)
        trig_params_compound = None
        if trig_enabled and (not trig_added) and (spec_trig is not None) and not _skip_dimless:
            # Estimate (omega, amp, phase, offset) on z-space for all atoms
            trig_params_compound = _estimate_trig_params_on_compound(
                ctx.train_loader_probe, reuse, target, input_expr_trig,
                device=ctx.device, dtype=ctx.dtype,
            )

            # Build omega candidates
            omega_candidates = []
            amp_init, phase_init, offset_init = 1.0, 0.0, 0.0

            if trig_params_compound is not None:
                omega_est, amp_init, phase_init, offset_init = trig_params_compound
                if math.isfinite(omega_est) and omega_est > 0.1 and amp_init > 1e-4:
                    omega_candidates.append(omega_est)

            # Incorporate spec_trig hint if available (optional boost, not required)
            if spec_trig is not None:
                spec_omega = float(spec_trig.omega)
                if spec_omega > 0 and not any(abs(w - spec_omega) < 0.1 for w in omega_candidates):
                    omega_candidates.append(spec_omega)

            # Exploratory candidates around the best omega
            if omega_candidates:
                best = omega_candidates[0]
                if abs(best - 1.0) > 0.1:
                    omega_candidates.append(1.0)
                omega_candidates.extend([best * 0.5, best * 2.0])

            if omega_candidates:
                # Build the trig AST rewrite via _make_trig_based_rewrite
                if spec_trig is not None:
                    _spec_for_rewrite = spec_trig
                else:
                    _spec_for_rewrite = _SyntheticTrigSpec(
                        omega_candidates[0], target.var_idxs[0],
                    )
                root_trig = _make_trig_based_rewrite(st.root, target, _spec_for_rewrite)

                if root_trig is not None:
                    for omega_init in omega_candidates:
                        # Create init_fn to set omega, amp, phase in SinLinearLeaf
                        def _init_trig_params(root_new, model_new, *, _tag=target.tag,
                                              _omega=omega_init, _amp=amp_init, _phase=phase_init):
                            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
                            for _atom in _collect_all_atoms(root_new):
                                if isinstance(_atom, AtomNode) and (_atom.tag == _tag) and (str(_atom.kind).lower() == "sin_linear"):
                                    leaf = atom_to_leaf.get(id(_atom), None)
                                    if leaf is None:
                                        continue
                                    inner = getattr(leaf, "base_model", getattr(leaf, "model", leaf))
                                    try:
                                        with torch.no_grad():
                                            if hasattr(inner, "weight") and inner.weight.numel() >= 1:
                                                inner.weight.zero_()
                                                inner.weight.view(-1)[0] = float(_omega)
                                            if hasattr(inner, "bias"):
                                                inner.bias.fill_(float(_phase))
                                            if hasattr(inner, "amp"):
                                                inner.amp.fill_(float(_amp))
                                    except Exception:
                                        pass
                        _init_trig_params._after_analytic_init = True

                        cands.append(
                            Candidate(
                                "trig",
                                root_trig,
                                init_fn=_init_trig_params,
                                meta={
                                    "log": f"[Stage B]  Trying trig on NN vars {target.var_idxs}, omega≈{omega_init:.3f}, amp≈{amp_init:.3f}"
                                },
                            )
                        )

                    # sinc candidates (available for both plain and compound atoms)
                    try:
                        sinc_cands = propose_sinc_family(ctx, target, _spec_for_rewrite, degree_arg=2, p=2)
                        cands.extend(sinc_cands)
                    except Exception as e:
                        if getattr(ctx, "verbose", False):
                            ctx.log(f"[Stage B]  sinc_p2 proposal failed: {e}")

                    trig_added = True

        # 8b. Affine trig: c + A*sin(ωz+φ) for shifted cosine forms.
        #
        # Offer this whenever trig evidence exists, not only when the pure
        # trig rewrite failed to materialize.  Leaves such as ``1-cos(z)`` are
        # exactly affine-trig but are badly represented by a pure sinusoid.
        if trig_enabled and (spec_trig is not None) and not _skip_dimless:
            root_affine_trig = _make_affine_trig_rewrite(st.root, target, spec_trig)
            if root_affine_trig is not None:
                # Determine trig parameter initialization (same logic as trig)
                _ie_aff = compound_input_expr(target)
                if has_nontrivial_input(target):
                    trig_params_aff = _estimate_trig_params_on_compound(
                        ctx.train_loader_probe, reuse, target, _ie_aff,
                        device=ctx.device, dtype=ctx.dtype,
                    )
                    if trig_params_aff is not None:
                        omega_est_aff, amp_init_aff, phase_init_aff, offset_init_aff = trig_params_aff
                    else:
                        omega_est_aff, amp_init_aff, phase_init_aff, offset_init_aff = 1.0, 1.0, 0.0, 0.0
                    # Compound atoms: single omega candidate
                    omega_candidates_aff = [omega_est_aff]
                else:
                    base_omega_aff = float(spec_trig.omega)
                    # Estimate amplitude and offset from teacher NN output
                    amp_init_aff, offset_init_aff = _estimate_univariate_trig_amplitude(
                        ctx.train_loader_probe, reuse, target, axis,
                        device=ctx.device, dtype=ctx.dtype,
                    )
                    phase_init_aff = 0.0  # Let LM optimize phase

                    # Try multiple omega values for better coverage
                    omega_candidates_aff = [base_omega_aff]
                    if abs(base_omega_aff - 1.0) > 0.1:
                        omega_candidates_aff.append(1.0)
                    omega_candidates_aff.extend([base_omega_aff * 0.5, base_omega_aff * 2.0])

                # Create an affine trig candidate for each omega value
                for omega_init_aff in omega_candidates_aff:
                    # Create init_fn for affine trig - sets sin_linear params and offset scalar.
                    def _init_affine_trig_params(root_new, model_new, *, _tag=target.tag,
                                                  _omega=omega_init_aff, _amp=amp_init_aff,
                                                  _phase=phase_init_aff, _offset=offset_init_aff):
                        atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
                        for _atom in _collect_all_atoms(root_new):
                            # Initialize sin_linear leaf
                            if isinstance(_atom, AtomNode) and str(_atom.kind).lower() == "sin_linear" and _atom.tag == _tag:
                                leaf = atom_to_leaf.get(id(_atom), None)
                                if leaf is not None:
                                    # leaf is an adaptor wrapper - get inner model
                                    inner = getattr(leaf, "base_model", getattr(leaf, "model", leaf))
                                    try:
                                        with torch.no_grad():
                                            if hasattr(inner, "weight") and inner.weight.numel() >= 1:
                                                inner.weight.zero_()
                                                inner.weight.view(-1)[0] = float(_omega)
                                            if hasattr(inner, "bias"):
                                                inner.bias.fill_(float(_phase))
                                            if hasattr(inner, "amp"):
                                                inner.amp.fill_(float(_amp))
                                    except Exception:
                                        pass
                            # Initialize offset scalar leaf tagged as "<tag>_c".
                            elif isinstance(_atom, AtomNode) and (_tag is not None) and _atom.tag == f"{_tag}_c":
                                leaf = atom_to_leaf.get(id(_atom), None)
                                if leaf is not None:
                                    try:
                                        _set_constant_leaf_value(leaf, float(_offset))
                                    except Exception:
                                        pass
                    _init_affine_trig_params._after_analytic_init = True

                    cands.append(
                        Candidate(
                            "affine_trig",
                            root_affine_trig,
                            init_fn=_init_affine_trig_params,
                            meta={
                                "structural": True,  # Accept even with excellent baseline
                                "log": f"[Stage B]  Trying affine trig (c + A*sin) on NN leaf vars {target.var_idxs}, omega≈{omega_init_aff:.3f}, amp≈{amp_init_aff:.3f}, offset≈{offset_init_aff:.3f}"
                            },
                        )
                    )

        # 8c. Leaf-transform wrappers for univariate leaves
        # Try fitting v(z)=T(u(z)) with a small inner model family (poly / affine trig),
        # then wrap back with T^{-1}. This generalizes patterns like:
        #   u(z)=1/(1+cos(az+b))  (recip + trig)
        #   u(z)=exp(sin(az+b))   (log + trig)
        # without requiring bespoke composite templates.
        if bool(getattr(ctx.lm_hp, "stageB_leaf_transforms_enable", True)) and not _skip_dimless:
            try:
                leaf_tag = getattr(target, "tag", None)
                teacher_leaf = reuse.get(leaf_tag, None) if leaf_tag is not None else None
                if teacher_leaf is None:
                    # Fallback: direct map lookup (works even if tags aren't in reuse)
                    try:
                        atom_to_leaf_local = build_atom_to_leaf_map(st.root, st.model)
                        teacher_leaf = atom_to_leaf_local.get(id(target), None)
                    except Exception:
                        teacher_leaf = None

                if teacher_leaf is not None:
                    # Support multi-dataset contexts: ctx.train_loader may be a list of loaders.
                    loaders = ctx.train_loader
                    if not isinstance(loaders, (list, tuple)):
                        loaders = [loaders]

                    max_pts = int(getattr(ctx.lm_hp, "stageB_leaf_transforms_max_points", 5000))
                    min_pts = int(getattr(ctx.lm_hp, "stageB_leaf_transforms_min_points", 300))
                    min_dom = float(getattr(ctx.lm_hp, "stageB_leaf_transforms_min_domain_ok_frac", 0.98))
                    rms_thr = float(getattr(ctx.lm_hp, "stageB_leaf_transforms_screen_rms_rel_max", 0.25))
                    max_hits = int(getattr(ctx.lm_hp, "stageB_leaf_transforms_max_candidates", 3))
                    poly_degs = list(getattr(ctx.lm_hp, "stageB_leaf_transforms_poly_degrees", (1, 2, 3)))
                    poly_degs = [int(d) for d in poly_degs if int(d) >= 0]
                    if not poly_degs:
                        poly_degs = [2]

                    # Gather (z, u) samples for this leaf
                    Xs = []
                    Us = []
                    n = 0
                    teacher_leaf.eval()
                    with torch.no_grad():
                        for dl in loaders:
                            if dl is None:
                                continue
                            for batch in dl:
                                if isinstance(batch, (list, tuple)):
                                    x_full = batch[0]
                                else:
                                    x_full = batch
                                x_full = x_full.to(device=ctx.device, dtype=ctx.dtype)
                                x_sub = _build_atom_input_tensor(target, x_full)
                                u = teacher_leaf(x_sub)
                                if u.dim() == 2:
                                    u = u[:, 0]
                                else:
                                    u = u.view(-1)
                                finite = torch.isfinite(u)
                                if finite.any():
                                    Xs.append(x_sub[finite].detach().cpu())
                                    Us.append(u[finite].detach().cpu())
                                    n += int(finite.sum().item())
                                if n >= max_pts:
                                    break
                            if n >= max_pts:
                                break
                    if Xs:
                        X = torch.cat(Xs, dim=0)[:max_pts]
                        U = torch.cat(Us, dim=0)[:max_pts]

                        # Work in float64 for stability
                        x = X[:, 0].to(dtype=torch.float64)
                        u = U.to(dtype=torch.float64)

                        # Omega candidates (trig core): use hint if available, else fall back to a range-based grid.
                        omega_cands = []
                        if trig_enabled:
                            base_omega = None
                            if spec_trig is not None:
                                try:
                                    base_omega = float(getattr(spec_trig, "omega", 1.0))
                                except Exception:
                                    base_omega = None
                            elif trig_params_compound is not None:
                                try:
                                    base_omega = float(trig_params_compound[0])
                                except Exception:
                                    base_omega = None

                            if base_omega is not None and math.isfinite(base_omega) and abs(base_omega) > 1e-12:
                                omega_cands = [base_omega]
                                if abs(base_omega - 1.0) > 0.1:
                                    omega_cands.append(1.0)
                                omega_cands.extend([0.5 * base_omega, 2.0 * base_omega])
                            else:
                                # Heuristic: pick a few harmonics across the observed z-domain.
                                try:
                                    xr = float((x.max() - x.min()).item())
                                except Exception:
                                    xr = 0.0
                                if math.isfinite(xr) and xr > 1e-9:
                                    base = 2.0 * math.pi / xr
                                    omega_cands = [base * k for k in (1.0, 2.0, 3.0, 4.0)]
                                    omega_cands.append(1.0)

                            # Deduplicate while preserving order
                            _seen = set()
                            omega_cands = [w for w in omega_cands if (math.isfinite(w) and (w not in _seen) and (not _seen.add(w)))]

                        # Small helper: affine trig fit v ~ c + A_sin sin(ωx) + A_cos cos(ωx)
                        def _fit_affine_trig_ls(xv: torch.Tensor, vv: torch.Tensor, omega: float, ridge: float = 1e-12):
                            t = float(omega) * xv
                            Phi = torch.stack([torch.sin(t), torch.cos(t), torch.ones_like(t)], dim=1)  # [N,3]
                            A = Phi.T @ Phi
                            b = Phi.T @ vv
                            I = torch.eye(3, dtype=Phi.dtype, device=Phi.device)
                            try:
                                beta = torch.linalg.solve(A + float(ridge) * I, b)
                            except Exception:
                                beta = torch.linalg.lstsq(Phi, vv).solution
                            vhat = Phi @ beta
                            resid = vv - vhat
                            denom = torch.sqrt(torch.mean(vv * vv)) + 1e-12
                            rms_rel = torch.sqrt(torch.mean(resid * resid)) / denom
                            A_sin = beta[0]
                            A_cos = beta[1]
                            c0 = beta[2]
                            amp = torch.sqrt(A_sin * A_sin + A_cos * A_cos)
                            phase = torch.atan2(A_cos, A_sin)
                            return float(rms_rel.item()), float(c0.item()), float(amp.item()), float(phase.item()), vhat

                        # Small helper: polynomial LS fit v ~ Σ c_k x^k
                        def _fit_poly_ls(xv: torch.Tensor, vv: torch.Tensor, deg: int, ridge: float = 1e-12):
                            d = int(deg)
                            Phi = torch.stack([xv ** k for k in range(d + 1)], dim=1)  # [N, d+1]
                            A = Phi.T @ Phi
                            b = Phi.T @ vv
                            I = torch.eye(d + 1, dtype=Phi.dtype, device=Phi.device)
                            try:
                                beta = torch.linalg.solve(A + float(ridge) * I, b)
                            except Exception:
                                beta = torch.linalg.lstsq(Phi, vv).solution
                            vhat = Phi @ beta
                            resid = vv - vhat
                            denom = torch.sqrt(torch.mean(vv * vv)) + 1e-12
                            rms_rel = torch.sqrt(torch.mean(resid * resid)) / denom
                            return float(rms_rel.item()), beta, vhat

                        leaf_transforms = list(getattr(ctx.lm_hp, "stageB_leaf_transforms", ["identity", "log", "recip", "square", "sqrt"]))
                        eps_u = 1.0e-12

                        # hits store dicts with keys: rms_u, t_name, inner, and per-inner params
                        hits = []

                        for t_name in leaf_transforms:
                            t_name = str(t_name)
                            if t_name == "identity":
                                continue

                            # Domain mask + forward transform
                            mask = torch.isfinite(u) & torch.isfinite(x)
                            sign = 1.0
                            inv_kind = None
                            if t_name == "recip":
                                mask = mask & (u.abs() > eps_u)
                                v = torch.reciprocal(u)
                                inv_kind = "recip"
                            elif t_name == "sqrt":
                                mask = mask & (u >= 0.0)
                                v = torch.sqrt(u.clamp_min(0.0))
                                inv_kind = "square"  # u = v^2
                            elif t_name == "square":
                                u2 = u * u
                                mask = mask & torch.isfinite(u2)
                                v = u2
                                inv_kind = "sqrt"  # u = ±sqrt(v)
                                # Only keep sign-consistent leaves (otherwise sqrt is ambiguous)
                                up = u[mask]
                                if up.numel() == 0:
                                    continue
                                frac_pos = float((up > 0.0).double().mean().item())
                                frac_neg = float((up < 0.0).double().mean().item())
                                frac_best = max(frac_pos, frac_neg)
                                if frac_best < 0.98:
                                    continue
                                sign = 1.0 if frac_pos >= frac_neg else -1.0
                            elif t_name == "log":
                                mask = mask & (u > eps_u)
                                v = torch.log(u)
                                inv_kind = "exp"  # u = exp(v)
                            else:
                                continue

                            dom_frac = float(mask.double().mean().item()) if mask.numel() else 0.0
                            if dom_frac < min_dom:
                                continue

                            xv = x[mask]
                            uv = u[mask]
                            vv = v[mask]
                            if int(vv.numel()) < min_pts:
                                continue

                            # Helper: wrap vhat back to uhat for scoring.
                            def _wrap_back(vhat: torch.Tensor):
                                if inv_kind == "recip":
                                    vhat_safe = torch.where(vhat.abs() > 1e-9, vhat, vhat.sign() * 1e-9 + (vhat == 0.0) * 1e-9)
                                    return torch.reciprocal(vhat_safe)
                                if inv_kind == "square":
                                    return vhat * vhat
                                if inv_kind == "sqrt":
                                    return float(sign) * torch.sqrt(vhat.clamp_min(0.0))
                                if inv_kind == "exp":
                                    return torch.exp(vhat)
                                return vhat

                            # 1) Poly core on v(z)
                            best_poly = None
                            for deg in poly_degs:
                                try:
                                    rms_v, beta, vhat = _fit_poly_ls(xv, vv, deg)
                                except Exception:
                                    continue
                                uhat = _wrap_back(vhat)
                                if not torch.isfinite(uhat).all():
                                    continue
                                denom_u = torch.sqrt(torch.mean(uv * uv)) + 1e-12
                                rms_u = torch.sqrt(torch.mean((uhat - uv) * (uhat - uv))) / denom_u
                                rms_u = float(rms_u.item())
                                if best_poly is None or rms_u < best_poly[0]:
                                    best_poly = (rms_u, int(deg), beta.detach().cpu())

                            if best_poly is not None and best_poly[0] <= rms_thr:
                                rms_u, deg, beta_cpu = best_poly
                                hits.append({
                                    "rms_u": float(rms_u),
                                    "t_name": t_name,
                                    "inner": "poly",
                                    "deg": int(deg),
                                    "beta": beta_cpu,
                                    "sign": float(sign),
                                })

                            # 2) Affine trig core on v(z) (optional)
                            if trig_enabled and omega_cands:
                                best_trig = None
                                for omega in omega_cands:
                                    rms_v, offset0, amp0, phase0, vhat = _fit_affine_trig_ls(xv, vv, omega)
                                    uhat = _wrap_back(vhat)
                                    if not torch.isfinite(uhat).all():
                                        continue
                                    denom_u = torch.sqrt(torch.mean(uv * uv)) + 1e-12
                                    rms_u = torch.sqrt(torch.mean((uhat - uv) * (uhat - uv))) / denom_u
                                    rms_u = float(rms_u.item())
                                    if best_trig is None or rms_u < best_trig[0]:
                                        best_trig = (rms_u, float(omega), float(amp0), float(phase0), float(offset0), float(sign))

                                if best_trig is not None and best_trig[0] <= rms_thr:
                                    rms_u, omega0, amp0, phase0, offset0, sign0 = best_trig
                                    hits.append({
                                        "rms_u": float(rms_u),
                                        "t_name": t_name,
                                        "inner": "trig",
                                        "omega": float(omega0),
                                        "amp": float(amp0),
                                        "phase": float(phase0),
                                        "offset": float(offset0),
                                        "sign": float(sign0),
                                    })

                        # Keep best-first and cap
                        hits.sort(key=lambda h: float(h.get("rms_u", 1e9)))
                        hits = hits[: max_hits]

                        for h in hits:
                            t_name = str(h.get("t_name"))
                            inner = str(h.get("inner"))
                            rms_u = float(h.get("rms_u", 1e9))

                            # --------------------------------------------------
                            # Poly-core candidate: u = T^{-1}(poly(z))
                            # --------------------------------------------------
                            if inner == "poly":
                                deg = int(h.get("deg", 2))
                                beta_cpu = h.get("beta", None)
                                if beta_cpu is None:
                                    continue

                                poly_tag = (
                                    f"{target.tag}_leaftr_{t_name}_poly{deg}" if getattr(target, "tag", None) else f"leaftr_poly_{id(target)}_{t_name}_{deg}"
                                )
                                poly_atom = AtomNode(
                                    kind="poly",
                                    var_idxs=tuple(int(j) for j in target.var_idxs),
                                    kwargs={"degree": int(deg), "min_total": 0},
                                    tag=poly_tag,
                                    inputs=clone_inputs(target),
                                )

                                # Inverse wrapper: u = T^{-1}(poly_atom)
                                if t_name == "recip":
                                    new_sub = PowNode(poly_atom, -1.0)
                                elif t_name == "sqrt":
                                    new_sub = PowNode(poly_atom, 2.0)
                                elif t_name == "square":
                                    base = PowNode(poly_atom, 0.5)
                                    if float(h.get("sign", 1.0)) < 0.0:
                                        new_sub = MulNode(ConstNode(-1.0), base)
                                    else:
                                        new_sub = base
                                elif t_name == "log":
                                    new_sub = ExpNode(poly_atom)
                                else:
                                    continue

                                root_leaftr = replace_atom_in_ast(st.root, target, new_sub)
                                if root_leaftr is None:
                                    continue

                                def _init_leaftr_poly_params(root_new, model_new, *, _poly_tag=poly_tag, _beta=beta_cpu):
                                    atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
                                    for _atom in _collect_all_atoms(root_new):
                                        if isinstance(_atom, AtomNode) and (_atom.tag == _poly_tag):
                                            leaf = atom_to_leaf.get(id(_atom), None)
                                            if leaf is None:
                                                break
                                            try:
                                                p = _leaf_coeff_param(leaf)
                                            except Exception:
                                                p = None
                                            if p is not None:
                                                try:
                                                    with torch.no_grad():
                                                        c = torch.as_tensor(_beta, dtype=p.dtype, device=p.device).view(-1)
                                                        p_flat = p.view(-1)
                                                        p_flat.zero_()
                                                        ncopy = min(int(p_flat.numel()), int(c.numel()))
                                                        if ncopy > 0:
                                                            p_flat[:ncopy].copy_(c[:ncopy])
                                                except Exception:
                                                    pass
                                            break

                                _init_leaftr_poly_params._after_analytic_init = True

                                cands.append(
                                    Candidate(
                                        f"leaftr_{t_name}_poly{deg}",
                                        root_leaftr,
                                        init_fn=_init_leaftr_poly_params,
                                        meta={
                                            "structural": True,
                                            "terminal_family": "leaf_transform_poly",
                                            "terminal_flexible_approximant": True,
                                            "leaftr_transform": str(t_name),
                                            "leaftr_degree": int(deg),
                                            "terminal_n_terms": int(deg + 1),
                                            "log": (
                                                f"[Stage B]  Leaf-transform '{t_name}' poly(deg={deg}): rms≈{rms_u:.3f}"
                                            ),
                                        },
                                    )
                                )
                                continue

                            # --------------------------------------------------
                            # Trig-core candidate: u = T^{-1}(c + A*sin(ωz+φ))
                            # --------------------------------------------------
                            if inner == "trig":
                                omega0 = float(h.get("omega", 1.0))
                                amp0 = float(h.get("amp", 1.0))
                                phase0 = float(h.get("phase", 0.0))
                                offset0 = float(h.get("offset", 0.0))
                                sign0 = float(h.get("sign", 1.0))

                                sin_atom = AtomNode(
                                    kind="sin_linear",
                                    var_idxs=tuple(int(j) for j in target.var_idxs),
                                    kwargs={},
                                    tag=getattr(target, "tag", None),
                                    inputs=clone_inputs(target),
                                )
                                const_tag = (
                                    f"{target.tag}_c" if getattr(target, "tag", None) else f"leaftr_c_{id(target)}"
                                )
                                const_variants = _scalar_constant_variants(
                                    getattr(ctx, "units_spec", None),
                                    base_tag=const_tag,
                                    scale_init=float(offset0),
                                )

                                for cvar in const_variants:
                                    const_atom = _build_scalar_atom_from_variant(cvar)
                                    core_expr = AddNode(const_atom, sin_atom)

                                    # Inverse wrapper: u = T^{-1}(core_expr)
                                    if t_name == "recip":
                                        new_sub = PowNode(core_expr, -1.0)
                                    elif t_name == "sqrt":
                                        new_sub = PowNode(core_expr, 2.0)
                                    elif t_name == "square":
                                        base = PowNode(core_expr, 0.5)
                                        if float(sign0) < 0.0:
                                            new_sub = MulNode(ConstNode(-1.0), base)
                                        else:
                                            new_sub = base
                                    elif t_name == "log":
                                        new_sub = ExpNode(core_expr)
                                    else:
                                        continue

                                    root_leaftr = replace_atom_in_ast(st.root, target, new_sub)
                                    if root_leaftr is None:
                                        continue

                                    # Init fn: set ω, amp, phase on sin_linear and set the chosen offset scalar.
                                    def _init_leaftr_trig_params(
                                        root_new,
                                        model_new,
                                        *,
                                        _tag=target.tag,
                                        _const_tag=str(cvar["tag"]),
                                        _omega=omega0,
                                        _amp=amp0,
                                        _phase=phase0,
                                        _offset=float(cvar["value"]),
                                    ):
                                        atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
                                        for _atom in _collect_all_atoms(root_new):
                                            if (
                                                isinstance(_atom, AtomNode)
                                                and str(_atom.kind).lower() == "sin_linear"
                                                and (_atom.tag == _tag)
                                            ):
                                                leaf = atom_to_leaf.get(id(_atom), None)
                                                if leaf is not None:
                                                    inner = getattr(leaf, "base_model", getattr(leaf, "model", leaf))
                                                    try:
                                                        with torch.no_grad():
                                                            if hasattr(inner, "weight") and inner.weight.numel() >= 1:
                                                                inner.weight.zero_()
                                                                inner.weight.view(-1)[0] = float(_omega)
                                                            if hasattr(inner, "bias"):
                                                                inner.bias.fill_(float(_phase))
                                                            if hasattr(inner, "amp"):
                                                                inner.amp.fill_(float(_amp))
                                                    except Exception:
                                                        pass
                                            elif (
                                                isinstance(_atom, AtomNode)
                                                and (_atom.tag == _const_tag)
                                            ):
                                                leaf = atom_to_leaf.get(id(_atom), None)
                                                if leaf is not None:
                                                    try:
                                                        _set_constant_leaf_value(leaf, float(_offset))
                                                    except Exception:
                                                        pass

                                    _init_leaftr_trig_params._after_analytic_init = True

                                    label_suffix = str(cvar.get("label_suffix", ""))
                                    cands.append(
                                        Candidate(
                                            f"leaftr_{t_name}_affine_trig{label_suffix}",
                                            root_leaftr,
                                            init_fn=_init_leaftr_trig_params,
                                            meta={
                                                "structural": True,
                                                "terminal_family": "leaf_transform_trig",
                                                "terminal_flexible_approximant": True,
                                                "leaftr_transform": str(t_name),
                                                "terminal_n_terms": 4,
                                                "log": (
                                                    f"[Stage B]  Leaf-transform '{t_name}' trig: rms≈{rms_u:.3f}, "
                                                    f"omega≈{omega0:.3f}, amp≈{amp0:.3f}, "
                                                    f"offset≈{float(cvar['value']):.3f}{label_suffix}"
                                                ),
                                            },
                                        )
                                    )
            except Exception:
                # Leaf-transform logic is strictly optional; never fail the rule.
                pass

        # HACK: try factorized symbolic search candidates first (temporary for benchmarking).
        # To restore normal order, set _FACTORIZED_SEARCH_FIRST = False.
        _FACTORIZED_SEARCH_FIRST = False  # disabled — factorized_search runs in normal priority
        if _FACTORIZED_SEARCH_FIRST:
            bsr = [c for c in cands if c.label.startswith("factorized_search[")]
            rest = [c for c in cands if not c.label.startswith("factorized_search[")]
            cands = bsr + rest

        cands = _merge_reciprocal_aliases_pairwise(
            cands, self._propose_reciprocal_coordinate_alias(ctx, target)
        )

        # Debug: log candidate order
        if cands:
            cand_labels = [c.label for c in cands]
            ctx.log(
                f"[Stage B] RuleUniNN proposing {len(cands)} candidates for NN vars={target.var_idxs}: {cand_labels}"
            )

        return cands


class RuleMonomialPeelPriority(StageBRule):
    """Global priority pass for cheap univariate monomial peels.

    This rule deliberately reuses ``RuleUniNN(monomial_only=True)`` for the
    actual candidates. Its only job is to rank monomial candidates across all
    1D NN leaves before Stage B falls back to target-order traversal.
    """

    name = "monomial_peel_priority"
    exhaustive = False
    global_candidate_priority = True
    multi_probe_native = True

    def __init__(self):
        self._base_rule = RuleUniNN(monomial_only=True)

    def iter_targets(self, ctx: StageBContext):
        return _collect_univariate_nn_atoms(ctx.state.root)

    def describe_target(self, target: Node) -> str:
        return self._base_rule.describe_target(target)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        return self._base_rule.propose(ctx, target)

    def _screen_target(self, ctx: StageBContext, target: AtomNode) -> Optional[MonomialScreenResult]:
        screen, _, _ = _screen_univariate_monomial_target(ctx, target)
        return screen

    def _scale_hint_for_target(self, ctx: StageBContext, target: AtomNode):
        if has_nontrivial_input(target):
            return None
        try:
            axis = int(target.var_idxs[0])
        except Exception:
            return None
        spec = _best_scale_spec_for_axis(ctx.scaling_by_axis, axis)
        return spec if _is_strong_scaling_spec(spec) else None

    def propose_global_candidates(
        self,
        ctx: StageBContext,
        targets: List[Node],
    ) -> List[Tuple[Node, Candidate]]:
        rows: List[Tuple[tuple, Node, Candidate, Optional[MonomialScreenResult]]] = []
        for target in targets:
            if not isinstance(target, AtomNode):
                continue
            screen = self._screen_target(ctx, target)
            scale_hint = self._scale_hint_for_target(ctx, target)
            is_raw = not has_nontrivial_input(target)
            cands = self._base_rule.propose(ctx, target) or []
            for cand in cands:
                if cand is None:
                    continue
                meta = dict(cand.meta) if isinstance(getattr(cand, "meta", None), dict) else {}
                meta["pattern"] = "monomial_peel_priority"
                meta["structural"] = True
                meta["separability_like"] = True
                meta["monomial_priority_is_raw_variable"] = bool(is_raw)
                if screen is not None:
                    meta["monomial_screen_ok"] = bool(screen.ok)
                    meta["monomial_screen_k"] = float(screen.k_hat)
                    meta["monomial_screen_rel_rms"] = float(screen.rel_rms)
                    meta["monomial_screen_support_frac"] = float(screen.support_frac)
                    meta["monomial_screen_points"] = int(screen.n_points)
                    if screen.reason:
                        meta["monomial_screen_reason"] = screen.reason
                if scale_hint is not None:
                    try:
                        meta["monomial_scale_hint_k"] = float(getattr(scale_hint, "k_hat"))
                        meta["monomial_scale_hint_rel_std"] = float(
                            getattr(scale_hint, "rel_std", float("inf"))
                        )
                    except Exception:
                        pass
                old_log = meta.get("log")
                if isinstance(old_log, str) and old_log:
                    meta["log"] = old_log.replace("[Stage B]  Trying", "[Stage B]  Priority trying", 1)
                else:
                    meta["log"] = (
                        f"[Stage B]  Priority trying {cand.label} on "
                        f"{self.describe_target(target)}"
                    )
                cand.meta = meta
                key = candidate_priority_from_screen(
                    label=cand.label,
                    screen=screen,
                    is_raw_variable=is_raw,
                    scale_hint=scale_hint,
                )
                rows.append((key, target, cand, screen))

        rows.sort(key=lambda r: r[0])
        if rows:
            preview = []
            for _, target, cand, screen in rows[:8]:
                if screen is not None and screen.ok:
                    score_s = f"k={screen.k_hat:.3g}, rel={screen.rel_rms:.3g}"
                elif screen is not None:
                    score_s = f"screen={screen.reason or 'failed'}"
                else:
                    score_s = "screen=missing"
                preview.append(f"{cand.label}@{self.describe_target(target)} ({score_s})")
            ctx.log("[Stage B] monomial_peel_priority shortlist: " + "; ".join(preview))
        return [(target, cand) for _, target, cand, _ in rows]
