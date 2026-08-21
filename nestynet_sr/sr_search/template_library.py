# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Phase 4: Generic template families.

This module provides *hint-driven* template builders that emit Stage-B
`Candidate`s.  These templates are intended to replace a growing list of
bespoke, problem-specific rewrite recipes.

Design notes
------------
Templates here:
  - build a parameterised AST subtree;
  - provide a robust custom initialiser (optional);
  - rely on LM to refine.

The functions are intentionally lightweight and avoid importing Stage-B
internals at module import time to prevent circular dependencies.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from nestynet_sr.sr_core.atoms import (
    RationalPolyLeaf,
    _enumerate_exponents,
    _eval_monomials,
)
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    CosNode,
    ExpNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    Var,
    clone_ast,
    clone_inputs,
    collect_all_atoms,
    compound_input_expr,
    count_atom_params,
    eval_input_expr,
    extra_input_var_idxs,
    has_nontrivial_input,
    replace_atom_in_ast,
    trivial_input_position,
)

from .batch_utils import first_batch_xy
import logging as _logging

from .rational_sparsify import (
    DEFAULT_RAT_STLSQ_CFG,
    RationalSparsifyConfig,
    _log_sparsify_result,
    stlsq_sparsify_rational_coeffs,
)
from .rational_supports import dense_rational_support_kwargs
from .model_selection import resolve_acceptance_noise_floor_raw
from .stageB import Candidate
from .wrapper_policy import macro_arg_wrapper_policy, snap_omega

_log = _logging.getLogger(__name__)


def _ctx_rat_sparsify_cfg(ctx: Any) -> RationalSparsifyConfig:
    try:
        noise_floor = float(
            getattr(getattr(ctx, "state", None), "acceptance_noise_floor_raw", 0.0)
            or 0.0
        )
    except Exception:
        noise_floor = 0.0
    if not (math.isfinite(noise_floor) and noise_floor > 0.0):
        try:
            noise_floor = float(
                resolve_acceptance_noise_floor_raw(
                    getattr(ctx, "lm_hp", None),
                    float(getattr(ctx, "loss_scale", 1.0)),
                )
            )
        except Exception:
            noise_floor = 0.0
    if math.isfinite(noise_floor) and noise_floor > 0.0:
        return replace(DEFAULT_RAT_STLSQ_CFG, proposal_noise_floor=float(noise_floor))
    return DEFAULT_RAT_STLSQ_CFG


def _first_batch_x(train_loader, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    xb, _ = first_batch_xy(train_loader, device=device, dtype=dtype)
    return xb


def _fit_poly_coeffs_ls(
    X: torch.Tensor, y: torch.Tensor, degree: int
) -> Tuple[torch.Tensor, List[Tuple[int, ...]]]:
    """Least-squares fit y ≈ Poly(X) over monomials up to `degree`.

    Returns
    -------
    coeffs : Tensor [n_terms]
    exps_list : list of exponent tuples in the same order.
    """
    X64 = X.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64).view(-1)
    exps_list = _enumerate_exponents(int(X64.shape[1]), int(degree))
    exps = torch.tensor(exps_list, dtype=torch.int64, device=X64.device)
    Phi = _eval_monomials(X64, exps)
    # robust LS via pinv (small systems)
    coeffs = (torch.linalg.pinv(Phi) @ y64.unsqueeze(1)).squeeze(1)
    return coeffs, exps_list


def _find_leaf_by_tag(root: Node, model: nn.Module, tag: str):
    from .stageB import _collect_all_atoms, build_atom_to_leaf_map

    atom_to_leaf = build_atom_to_leaf_map(root, model)
    for a in _collect_all_atoms(root):
        if isinstance(a, AtomNode) and a.tag == tag:
            return atom_to_leaf.get(id(a), None)
    return None


def _set_poly_like_coeffs_by_vector(leaf: nn.Module, coeffs: torch.Tensor):
    """Copy a coefficient vector into a poly-like core (.coeffs)."""
    core = getattr(leaf, "core", getattr(leaf, "model", leaf))
    p = getattr(core, "coeffs", None)
    if p is None:
        raise RuntimeError(f"Leaf core {type(core)} has no .coeffs")
    with torch.no_grad():
        p_flat = p.view(-1)
        c_flat = coeffs.to(device=p_flat.device, dtype=p_flat.dtype).view(-1)
        if p_flat.numel() != c_flat.numel():
            raise RuntimeError(
                f"Coeff size mismatch: core has {p_flat.numel()} terms, fit gave {c_flat.numel()}"
            )
        p_flat.copy_(c_flat)


def _set_ratpoly_deg0_1_from_linear_fit(leaf: nn.Module, denom_coeffs: torch.Tensor):
    """Initialise RationalPolyLeaf with deg_num=0, deg_den=1.

    Sets numerator constant to 1 and denominator coefficients from `denom_coeffs`
    (ordered to match core.exps_den).
    """
    core = getattr(leaf, "core", getattr(leaf, "model", leaf))
    if not isinstance(core, RationalPolyLeaf):
        raise RuntimeError(f"Expected RationalPolyLeaf, got {type(core)}")
    with torch.no_grad():
        core.coeffs_num.zero_()
        core.coeffs_den.zero_()
        # numerator constant
        core.coeffs_num.view(-1)[0] = 1.0
        # denominator
        d = denom_coeffs.to(device=core.coeffs_den.device, dtype=core.coeffs_den.dtype).view(-1)
        if d.numel() != core.coeffs_den.numel():
            raise RuntimeError(
                f"Denom coeff size mismatch: core has {core.coeffs_den.numel()}, fit gave {d.numel()}"
            )
        core.coeffs_den.copy_(d)


def propose_trig_rational(
    ctx: Any,
    target: AtomNode,
    trapped_res: Any,
    label: str = "trig_rational",
) -> Optional[Candidate]:
    """Template family: A(xL) * sin(N(x))^p * sin(D(xL))^-p.

    Current implementation focuses on the high-payoff 2-variable case where
    the trapped-variable probe indicates a *product* structure.
    """
    st = ctx.state

    # Shared wrapper-policy gate: allow disabling trig templates globally,
    # and optionally avoid squared variants when trig-squares are disabled.
    trig_squares_ok = True
    try:
        hp = getattr(ctx, "lm_hp", None)
        if hp is not None:
            pol = macro_arg_wrapper_policy(ctx, hp, target)
            if not bool(pol.trig):
                return None
            trig_squares_ok = bool(pol.trig_squares)
    except Exception:
        trig_squares_ok = True

    if getattr(trapped_res, "candidate_P", "product") != "product":
        return None
    leaky = int(trapped_res.leaky_idx)
    trapped = int(trapped_res.trapped_idx)
    v = [int(j) for j in target.var_idxs]
    if leaky not in v or trapped not in v or leaky == trapped:
        return None

    tagS = f"tr_{label}_scale_{leaky}_{trapped}"
    tagN = f"tr_{label}_num_{leaky}_{trapped}"
    tagD = f"tr_{label}_den_{leaky}_{trapped}"

    # A(xL): for now constant scale (poly degree 0)
    scale_atom = AtomNode(kind="poly", var_idxs=(leaky,), kwargs={"degree": 0}, tag=tagS)
    # N(xL,xT): degree-2 poly seeded to k*xL*xT (full basis for flexibility)
    poly_num = AtomNode(
        kind="poly", var_idxs=(leaky, trapped), kwargs={"degree": 2, "min_total": 0}, tag=tagN
    )
    # D(xL): degree-1 poly seeded to k*xL (full basis for flexibility)
    poly_den = AtomNode(
        kind="poly", var_idxs=(leaky,), kwargs={"degree": 1, "min_total": 0}, tag=tagD
    )

    # Pick p ∈ {1,2} based on sign behaviour (p=1 if clearly sign-changing).
    p0 = 2.0
    try:
        from .stageB import _SubtreeModel, build_atom_to_leaf_map

        xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
        atom_to_leaf_old = build_atom_to_leaf_map(st.root, st.model)
        subtree_old = _SubtreeModel(root=target, atom_to_leaf=atom_to_leaf_old)
        with torch.no_grad():
            f_old = subtree_old.forward(xb).view(-1)
        pos = (f_old > 0).float().mean().item()
        neg = (f_old < 0).float().mean().item()
        if pos > 0.05 and neg > 0.05:
            p0 = 1.0
    except Exception:
        pass

    # If squared trig wrappers are disabled by policy, force p=1.
    if (not trig_squares_ok) and float(p0) != 1.0:
        p0 = 1.0
    numer = PowNode(SinNode(poly_num), exponent=p0)
    denom = PowNode(SinNode(poly_den), exponent=-p0)
    new_subtree = MulNode(scale_atom, MulNode(numer, denom))
    # Note: replace_atom_in_ast is already imported at module level from nestynet_sr.sr_core.bridges
    cand_root = replace_atom_in_ast(st.root, target, new_subtree)

    def _init(
        root_new: Node,
        model_new: nn.Module,
        _old_state=st,
        _target=target,
        _leaky=leaky,
        _trapped=trapped,
        _p=float(p0),
    ):
        from .stageB import (
            _collect_all_atoms,
            _poly_zero_and_set,
            _SubtreeModel,
            build_atom_to_leaf_map,
        )

        atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
        subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)

        xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
        with torch.no_grad():
            f_old = subtree_old.forward(xb).view(-1)

        xL = xb[:, _leaky].view(-1)
        xT = xb[:, _trapped].view(-1)

        denom_eps = 1e-3
        tiny = 1e-18
        k_best, s_best, err_best = 0.5, 1.0, float("inf")

        xL_max = float(xL.abs().max().clamp_min(1e-6).item())
        k_max_safe = 0.98 * (math.pi / xL_max)
        # Minimum-frequency identifiability guard: when k is too small,
        # sin(kx)≈kx and the trigonometric template degenerates into a
        # polynomial with poorly identified oscillating coefficients.
        k_min = 0.1  # Exclude near-linear sine approximations.
        k_max = float(max(k_min + 1e-3, min(2.5, k_max_safe)))
        n_grid = 120
        ks = torch.linspace(k_min, k_max, steps=n_grid, device=xb.device, dtype=xb.dtype)

        def _err_at(k: float):
            # Best-scale residual of A * (sin(k xL xT)/sin(k xL))^p against f_old.
            den = torch.sin(k * xL)
            den = torch.sign(den) * den.abs().clamp_min(denom_eps)
            base = (torch.sin(k * xL * xT) / den) ** _p
            m = torch.isfinite(base) & torch.isfinite(f_old)
            if int(m.sum().item()) < 200:
                return float("inf"), 1.0
            bb, ff = base[m], f_old[m]
            s = (bb * ff).sum() / (bb * bb).sum().clamp_min(tiny)
            err = torch.mean((s * bb - ff) ** 2).item()
            return err, float(s.item())

        with torch.no_grad():
            for k in ks:
                err, s = _err_at(float(k.item()))
                if err < err_best:
                    err_best, k_best, s_best = err, float(k.item()), s

            # The sin-ratio error can be a razor-sharp unimodal well in k:
            # physically coherent ratios may need k accurate to ~1e-5 while
            # the coarse grid spacing is ~1e-2. Refine within one grid step of
            # the coarse best; local unimodality makes this a bounded sharpening.
            if math.isfinite(err_best):
                step = (k_max - k_min) / max(1, n_grid - 1)
                a = max(k_min, k_best - step)
                b = min(k_max, k_best + step)
                if b - a > 1e-9:
                    gr = 0.5 * (math.sqrt(5.0) - 1.0)
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    ec, _ = _err_at(c)
                    ed, _ = _err_at(d)
                    for _ in range(60):
                        if b - a <= 1e-8:
                            break
                        if ec < ed:
                            b, d, ed = d, c, ec
                            c = b - gr * (b - a)
                            ec, _ = _err_at(c)
                        else:
                            a, c, ec = c, d, ed
                            d = a + gr * (b - a)
                            ed, _ = _err_at(d)
                    k_ref = 0.5 * (a + b)
                    err_ref, s_ref = _err_at(k_ref)
                    if err_ref < err_best:
                        err_best, k_best, s_best = err_ref, k_ref, s_ref

        # Locate new leaves via tags
        atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
        aS = aN = aD = None
        for a in _collect_all_atoms(root_new):
            if not isinstance(a, AtomNode):
                continue
            if a.tag == tagS:
                aS = a
            elif a.tag == tagN:
                aN = a
            elif a.tag == tagD:
                aD = a

        leafS = atom_to_leaf_new.get(id(aS), None) if aS is not None else None
        leafN = atom_to_leaf_new.get(id(aN), None) if aN is not None else None
        leafD = atom_to_leaf_new.get(id(aD), None) if aD is not None else None
        if leafS is None or leafN is None or leafD is None:
            return

        # Judge the grid search relatively. An absolute err_best cannot
        # distinguish "found the frequency" from "nothing on the grid matches
        # this teacher"; a gauge-shifted or polluted slice can otherwise seed
        # a divergent start. If no searched k explains a reasonable fraction
        # of the teacher's power, use the finite fallback and let it lose
        # honestly (its swapped-assignment sibling may still win).
        f_ms = float(torch.mean(f_old * f_old).clamp_min(tiny).item())
        rel_err = err_best / f_ms if math.isfinite(err_best) else float("inf")
        if rel_err > 0.25:
            print(
                f"[Stage B trig_rational init] SKIP: rel_err={rel_err:.3g} > 0.25 "
                f"(no searched k matches teacher; k_best={k_best:.3g}, "
                f"scale0={s_best:.3g}; safe fallback)"
            )
            _poly_zero_and_set(leafS, {(0,): 1.0})
            _poly_zero_and_set(leafD, {(0,): 0.0, (1,): 1.0})
            _poly_zero_and_set(leafN, {(0, 0): 0.0, (1, 1): 1.0})
            return

        # Recheck frequency identifiability after refinement. Below this
        # threshold, sin(kx)≈kx and the trigonometric template becomes a
        # polynomial in disguise with poorly identified coefficients.
        k_min_threshold = 0.05
        if k_best < k_min_threshold:
            print(
                f"[Stage B trig_rational init] SKIP: k_best={k_best:.6g} < {k_min_threshold} "
                "(polynomial in disguise, template not appropriate)"
            )
            # Set coefficients to produce a simple constant (fallback)
            _poly_zero_and_set(leafS, {(0,): 1.0})
            _poly_zero_and_set(leafD, {(0,): 0.0, (1,): 1.0})  # k=1 to avoid singularities
            _poly_zero_and_set(leafN, {(0, 0): 0.0, (1, 1): 1.0})
            return

        _poly_zero_and_set(leafS, {(0,): s_best})
        _poly_zero_and_set(leafD, {(0,): 0.0, (1,): k_best})
        # seed numerator to k*xL*xT (exp=(1,1))
        _poly_zero_and_set(leafN, {(0, 0): 0.0, (1, 1): k_best})
        print(
            f"[Stage B trig_rational init] p={_p:g} k0={k_best:.6g} scale0={s_best:.6g} err={err_best:.3e}"
        )

    _init._after_analytic_init = True
    return Candidate(label, cand_root, _init)


def propose_exp_of_quadratic(
    ctx: Any,
    target: AtomNode,
    logquad_hint: Any,
    label_base: str = "exp_quad",
) -> List[Candidate]:
    """Template family: scale * exp(q(x)) [ / (linear(x)) ]

    Driven by QuadraticHint(type="log"). We propose two variants:
      1) scale * exp(q(x))
      2) scale * exp(q(x)) * (1 / linear(x))   (implemented as ratpoly deg0/deg1)
    """
    st = ctx.state
    if logquad_hint is None or not getattr(logquad_hint, "ok", False):
        return []

    if target.kind.lower() != "nn":
        return []

    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) < 1:
        return []

    tag_scale = f"{label_base}_scale_{'_'.join(map(str, var_idxs))}"
    tag_exp = f"{label_base}_exp_{'_'.join(map(str, var_idxs))}"
    tag_rat = f"{label_base}_rat_{'_'.join(map(str, var_idxs))}"

    scale_atom = AtomNode(kind="poly", var_idxs=var_idxs, kwargs={"degree": 0}, tag=tag_scale)
    exp_atom = AtomNode(kind="exp_poly", var_idxs=var_idxs, kwargs={"degree": 2}, tag=tag_exp)
    rat_atom = AtomNode(
        kind="ratpoly",
        var_idxs=var_idxs,
        kwargs=dense_rational_support_kwargs(
            n_inputs=len(var_idxs),
            degree_num=0,
            degree_den=1,
        ),
        tag=tag_rat,
    )

    # Note: replace_atom_in_ast is already imported at module level from nestynet_sr.sr_core.bridges

    cand_roots: List[Tuple[str, Node]] = []
    cand_roots.append(
        (label_base, replace_atom_in_ast(st.root, target, MulNode(scale_atom, exp_atom)))
    )
    cand_roots.append(
        (
            label_base + "_divlin",
            replace_atom_in_ast(st.root, target, MulNode(MulNode(scale_atom, exp_atom), rat_atom)),
        )
    )

    def _init_common(
        root_new: Node, model_new: nn.Module, include_divlin: bool, _old_state=st, _target=target
    ):
        from .stageB import _SubtreeModel, build_atom_to_leaf_map

        atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
        subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)

        xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
        xsub = xb[:, list(var_idxs)].to(dtype=torch.float64)
        with torch.no_grad():
            f_old = subtree_old.forward(xb).view(-1).to(dtype=torch.float64)

        # Require (approximately) sign-definite to take log(abs(f)).
        eps = 1e-12
        pos = (f_old > eps).float().mean().item()
        neg = (f_old < -eps).float().mean().item()
        if pos > 0.05 and neg > 0.05:
            return
        sign = 1.0 if pos >= neg else -1.0

        m = torch.isfinite(f_old) & (f_old.abs() > 1e-12)
        if int(m.sum().item()) < 50:
            return

        y = torch.log((sign * f_old[m]).abs().clamp_min(1e-12))
        X = xsub[m]
        try:
            coeffs_q, _ = _fit_poly_coeffs_ls(X, y, degree=2)
        except Exception:
            return

        leaf_scale = _find_leaf_by_tag(root_new, model_new, tag_scale)
        leaf_exp = _find_leaf_by_tag(root_new, model_new, tag_exp)
        leaf_rat = _find_leaf_by_tag(root_new, model_new, tag_rat) if include_divlin else None
        if leaf_scale is None or leaf_exp is None:
            return

        # Set scale ≈ sign
        from .stageB import _poly_zero_and_set

        _poly_zero_and_set(leaf_scale, {tuple([0] * len(var_idxs)): float(sign)})

        # Set quadratic exponent coefficients
        _set_poly_like_coeffs_by_vector(leaf_exp, coeffs_q)

        if include_divlin and leaf_rat is not None:
            # Fit linear denominator to lin_target = scale*exp(q)/f
            with torch.no_grad():
                q = torch.exp(
                    (
                        _eval_monomials(
                            X,
                            torch.tensor(
                                _enumerate_exponents(int(X.shape[1]), 2),
                                dtype=torch.int64,
                                device=X.device,
                            ),
                        )
                        @ coeffs_q
                    ).view(-1)
                )
                lin_target = (float(sign) * q) / (f_old[m].to(X.device))
                coeffs_lin, _ = _fit_poly_coeffs_ls(X, lin_target, degree=1)
            _set_ratpoly_deg0_1_from_linear_fit(leaf_rat, coeffs_lin)

    cands: List[Candidate] = []
    for lab, r in cand_roots:
        include_divlin = lab.endswith("_divlin")

        def _init(root_new: Node, model_new: nn.Module, _inc=include_divlin):
            return _init_common(root_new, model_new, include_divlin=_inc)

        _init._after_analytic_init = True
        cands.append(Candidate(lab, r, _init))
    return cands


def propose_exp_poly_from_log_hint(
    ctx: Any,
    target: AtomNode,
    transform_hint: Any,
    *,
    degrees: Tuple[int, ...] = (1, 2),
    label_base: str = "exp_poly_log",
) -> List[Candidate]:
    """Template family: scale * exp(poly(x)) driven by TransformHint(best_name="log").

    If log(u) looks simpler (typically polynomial-ish), propose:
        u(x) ≈ s * exp_poly(x; degree=d)

    where `s` is a constant poly(deg=0) capturing sign and mean-log amplitude.

    Initialisation strategy
    -----------------------
    On a sign-consistent region, fit:
        y = log(sign * u)

    We subtract the mean μ to keep exponent magnitudes small (avoid ExpPolyLeaf clamp),
    fit a polynomial to (y - μ), and set:
        s ≈ sign * exp(μ).
    """
    st = ctx.state
    if transform_hint is None or not getattr(transform_hint, "ok", False):
        return []
    if getattr(transform_hint, "best_name", "") != "log":
        return []
    if target.kind.lower() != "nn":
        return []

    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) < 1:
        return []

    # Note: replace_atom_in_ast is already imported at module level from nestynet_sr.sr_core.bridges

    cands: List[Candidate] = []
    for deg in degrees:
        try:
            deg_i = int(deg)
        except Exception:
            continue
        if deg_i < 1:
            continue

        label = f"{label_base}_d{deg_i}"
        tag_scale = f"{label}_scale_{'_'.join(map(str, var_idxs))}"
        tag_exp = f"{label}_exp_{'_'.join(map(str, var_idxs))}"

        scale_atom = AtomNode(kind="poly", var_idxs=var_idxs, kwargs={"degree": 0}, tag=tag_scale)
        # Carry the target's compound input expressions (None for simple
        # targets): the exp_poly then consumes e.g. (x0/x1, x6) instead of the
        # raw variables, matching both the units semantics and the fit path.
        exp_atom = AtomNode(
            kind="exp_poly",
            var_idxs=var_idxs,
            kwargs={"degree": deg_i},
            tag=tag_exp,
            inputs=clone_inputs(target),
        )
        new_sub = MulNode(scale_atom, exp_atom)
        cand_root = replace_atom_in_ast(st.root, target, new_sub)

        def _init(
            root_new: Node,
            model_new: nn.Module,
            _old_state=st,
            _target=target,
            _var_idxs=var_idxs,
            _deg=deg_i,
            _tag_scale=tag_scale,
            _tag_exp=tag_exp,
        ):
            from .stageB import _SubtreeModel, build_atom_to_leaf_map

            atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
            subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)

            xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
            if has_nontrivial_input(_target):
                from ._candidate_builders_common import _build_atom_input_tensor

                xsub = _build_atom_input_tensor(_target, xb).to(dtype=torch.float64)
            else:
                xsub = xb[:, list(_var_idxs)].to(dtype=torch.float64)
            with torch.no_grad():
                f_old = subtree_old.forward(xb).view(-1).to(dtype=torch.float64)

            eps = 1e-12
            pos = float((f_old > eps).float().mean().item())
            neg = float((f_old < -eps).float().mean().item())
            if pos > 0.05 and neg > 0.05:
                # sign-changing: avoid log initialisation
                return
            sign = 1.0 if pos >= neg else -1.0

            m = torch.isfinite(f_old) & ((sign * f_old) > eps)
            if int(m.sum().item()) < 50:
                return

            y = torch.log((sign * f_old[m]).clamp_min(eps))
            X = xsub[m]
            mu = float(y.mean().item())
            y_c = y - mu

            try:
                coeffs_p, _ = _fit_poly_coeffs_ls(X, y_c, degree=int(_deg))
            except Exception:
                return

            leaf_scale = _find_leaf_by_tag(root_new, model_new, _tag_scale)
            leaf_exp = _find_leaf_by_tag(root_new, model_new, _tag_exp)
            if leaf_scale is None or leaf_exp is None:
                return

            s0 = sign * math.exp(mu)
            if not math.isfinite(s0):
                return

            _set_poly_like_coeffs_by_vector(
                leaf_scale, torch.tensor([float(s0)], dtype=torch.float64)
            )
            _set_poly_like_coeffs_by_vector(leaf_exp, coeffs_p)

        _init._after_analytic_init = True
        cands.append(Candidate(label, cand_root, _init))

    return cands


def propose_rational_linear(
    ctx: Any,
    target: AtomNode,
    transform_hint: Any,
    label: str = "rat_linear",
) -> Optional[Candidate]:
    """Template family: 1 / linear(x)

    Driven by TransformHint(best_name="recip") meaning that 1/u looks additive/poly.
    We propose a rational polynomial with deg_num=0, deg_den=1.
    """
    st = ctx.state
    if transform_hint is None or not getattr(transform_hint, "ok", False):
        return None
    if getattr(transform_hint, "best_name", "") != "recip":
        return None
    if target.kind.lower() != "nn":
        return None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) < 1:
        return None

    tag_rat = f"{label}_{'_'.join(map(str, var_idxs))}"
    rat_atom = AtomNode(
        kind="ratpoly",
        var_idxs=var_idxs,
        kwargs=dense_rational_support_kwargs(
            n_inputs=len(var_idxs),
            degree_num=0,
            degree_den=1,
        ),
        tag=tag_rat,
    )
    # Note: replace_atom_in_ast is already imported at module level from nestynet_sr.sr_core.bridges
    cand_root = replace_atom_in_ast(st.root, target, rat_atom)

    def _init(root_new: Node, model_new: nn.Module, _old_state=st, _target=target):
        from .stageB import _SubtreeModel, build_atom_to_leaf_map

        atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
        subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)

        xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
        xsub = xb[:, list(var_idxs)].to(dtype=torch.float64)
        with torch.no_grad():
            f_old = subtree_old.forward(xb).view(-1).to(dtype=torch.float64)

        m = torch.isfinite(f_old) & (f_old.abs() > 1e-12)
        if int(m.sum().item()) < 50:
            return
        y = (1.0 / f_old[m]).to(dtype=torch.float64)
        X = xsub[m]
        try:
            coeffs_lin, _ = _fit_poly_coeffs_ls(X, y, degree=1)
        except Exception:
            return

        leaf_rat = _find_leaf_by_tag(root_new, model_new, tag_rat)
        if leaf_rat is None:
            return
        _set_ratpoly_deg0_1_from_linear_fit(leaf_rat, coeffs_lin)

    _init._after_analytic_init = True
    return Candidate(label, cand_root, _init)


def propose_sin_cos_from_inverse_hint(
    ctx: Any,
    target: AtomNode,
    transform_hint: Any,
    degree: int = 2,
) -> List[Candidate]:
    """Inverse-trig driven template families.

    Supported hints (TransformHint.best_name):
      - "asin"         ->  sin(poly(x))
      - "acos"         ->  cos(poly(x))
      - "asin_affine"  ->  beta + alpha * sin(poly(x))
      - "acos_affine"  ->  beta + alpha * cos(poly(x))

    For affine hints, (alpha, beta) are expected in transform_hint.best.params
    (as produced by probe_output_transforms).
    """
    st = ctx.state
    if transform_hint is None or not getattr(transform_hint, "ok", False):
        return []
    best = getattr(transform_hint, "best_name", "")
    if best not in ("asin", "acos", "asin_affine", "acos_affine"):
        return []
    if target.kind.lower() != "nn":
        return []

    # Shared wrapper-policy gate: allow disabling trig templates globally.
    try:
        hp = getattr(ctx, "lm_hp", None)
        if hp is not None:
            pol = macro_arg_wrapper_policy(ctx, hp, target)
            if not bool(pol.trig):
                return []
    except Exception:
        pass

    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) < 1:
        return []

    # Parse hint
    use_affine = best.endswith("_affine")
    is_asin = best.startswith("asin")

    # Read affine parameters (if any)
    alpha0: Optional[float] = None
    beta0: Optional[float] = None
    if use_affine:
        spec = getattr(transform_hint, "best", None)
        params = getattr(spec, "params", None) or {}
        try:
            alpha0 = float(params.get("alpha", float("nan")))
            beta0 = float(params.get("beta", float("nan")))
        except Exception:
            alpha0, beta0 = None, None
        if (
            (alpha0 is None)
            or (beta0 is None)
            or (not math.isfinite(alpha0))
            or (abs(alpha0) < 1e-12)
            or (not math.isfinite(beta0))
        ):
            # Fall back to the non-affine template if the hint is malformed.
            use_affine = False
            alpha0, beta0 = None, None

    op = SinNode if is_asin else CosNode
    inv = torch.asin if is_asin else torch.acos
    if use_affine:
        label = "sin_affine_from_asin" if is_asin else "cos_affine_from_acos"
    else:
        label = "sin_from_asin" if is_asin else "cos_from_acos"
    tag_poly = f"{label}_{'_'.join(map(str, var_idxs))}"
    poly_atom = AtomNode(
        kind="poly", var_idxs=var_idxs, kwargs={"degree": int(degree), "min_total": 0}, tag=tag_poly
    )
    tag_alpha = None
    tag_beta = None
    if use_affine:
        tag_alpha = f"{label}_alpha_{'_'.join(map(str, var_idxs))}"
        tag_beta = f"{label}_beta_{'_'.join(map(str, var_idxs))}"
        alpha_atom = AtomNode(kind="poly", var_idxs=var_idxs, kwargs={"degree": 0}, tag=tag_alpha)
        beta_atom = AtomNode(kind="poly", var_idxs=var_idxs, kwargs={"degree": 0}, tag=tag_beta)
        new_sub = AddNode(beta_atom, MulNode(alpha_atom, op(poly_atom)))
    else:
        new_sub = op(poly_atom)
    # Note: replace_atom_in_ast is already imported at module level from nestynet_sr.sr_core.bridges
    cand_root = replace_atom_in_ast(st.root, target, new_sub)

    def _init(root_new: Node, model_new: nn.Module, _old_state=st, _target=target):
        from .stageB import _SubtreeModel, build_atom_to_leaf_map

        atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
        subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)

        xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
        xsub = xb[:, list(var_idxs)].to(dtype=torch.float64)
        with torch.no_grad():
            f_old = subtree_old.forward(xb).view(-1).to(dtype=torch.float64)

        # Domain clamp for inverse trig
        if use_affine:
            a = float(alpha0)
            b = float(beta0)
            v = (f_old - b) / a
            m = torch.isfinite(v) & (v.abs() <= (1.0 - 1e-6))
            if int(m.sum().item()) < 50:
                return
            y = inv(v[m]).to(dtype=torch.float64)
        else:
            m = torch.isfinite(f_old) & (f_old.abs() <= (1.0 - 1e-6))
            if int(m.sum().item()) < 50:
                return
            y = inv(f_old[m]).to(dtype=torch.float64)
        X = xsub[m]
        try:
            coeffs_p, _ = _fit_poly_coeffs_ls(X, y, degree=int(degree))
        except Exception:
            return

        leaf_poly = _find_leaf_by_tag(root_new, model_new, tag_poly)
        if leaf_poly is None:
            return
        if use_affine and tag_alpha is not None and tag_beta is not None:
            leaf_alpha = _find_leaf_by_tag(root_new, model_new, tag_alpha)
            leaf_beta = _find_leaf_by_tag(root_new, model_new, tag_beta)
            if leaf_alpha is not None and leaf_beta is not None:
                _set_poly_like_coeffs_by_vector(
                    leaf_alpha, torch.tensor([float(alpha0)], dtype=torch.float64)
                )
                _set_poly_like_coeffs_by_vector(
                    leaf_beta, torch.tensor([float(beta0)], dtype=torch.float64)
                )

    _init._after_analytic_init = True
    return [Candidate(label, cand_root, _init)]


# ==============================
# Additional Phase-4 templates
#   - tanh family
#   - sinc family
# ==============================


def _fit_rational_coeffs_nd_eig(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    deg_num: int,
    deg_den: int,
    cfg: Optional[RationalSparsifyConfig] = None,
    max_points: int = 1500,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return (a,b,exps_num,exps_den) for y≈P/Q via smallest-eigenvector fit."""
    try:
        X = X.to(dtype=dtype)
        y = y.view(-1).to(dtype=dtype)
        if X.ndim != 2 or y.ndim != 1:
            return None
        N, dim = X.shape
        Np = int(min(N, max_points))
        if Np < 50:
            return None

        Xp = X[:Np]
        yp = y[:Np]

        exps_num = _enumerate_exponents(dim, int(deg_num))
        exps_den = _enumerate_exponents(dim, int(deg_den))
        exps_num_t = torch.tensor(exps_num, dtype=torch.int64, device=Xp.device)
        exps_den_t = torch.tensor(exps_den, dtype=torch.int64, device=Xp.device)

        Phi_num = _eval_monomials(Xp, exps_num_t)
        Phi_den = _eval_monomials(Xp, exps_den_t)
        M_num = int(Phi_num.shape[1])
        M_den = int(Phi_den.shape[1])
        if Np < (M_num + M_den + 5):
            return None

        A = torch.cat([Phi_num, -(yp.unsqueeze(1) * Phi_den)], dim=1)
        Gram = (A.T @ A) / float(Np)
        _, vecs = torch.linalg.eigh(Gram)
        c = vecs[:, 0]
        a = c[:M_num].clone()
        b = c[M_num:].clone()

        if b.numel() > 0:
            pivot = b[0]
            if float(pivot.abs()) < eps:
                pivot = b[torch.argmax(b.abs())]
            if float(pivot.abs()) >= eps:
                a = a / pivot
                b = b / pivot

        try:
            a_sparse, b_sparse, meta = stlsq_sparsify_rational_coeffs(
                Phi_num=Phi_num,
                Phi_den=Phi_den,
                y=yp,
                coeffs_num=a,
                coeffs_den=b,
                cfg=cfg or DEFAULT_RAT_STLSQ_CFG,
            )
            _log_sparsify_result(
                "_fit_rational_coeffs_nd_eig", a, b, a_sparse, b_sparse, meta,
            )
            a, b = a_sparse, b_sparse
        except Exception as exc:
            _log.debug("[_fit_rational_coeffs_nd_eig] rational sparsify failed: %s", exc)

        return a, b, exps_num_t, exps_den_t
    except Exception:
        return None


def _set_ratpoly_coeffs_by_vectors(leaf: nn.Module, a: torch.Tensor, b: torch.Tensor) -> bool:
    core = getattr(leaf, "core", getattr(leaf, "model", leaf))
    if not (hasattr(core, "coeffs_num") and hasattr(core, "coeffs_den")):
        return False
    with torch.no_grad():
        cn = getattr(core, "coeffs_num")
        cd = getattr(core, "coeffs_den")
        if cn.dim() == 1:
            if a.numel() != cn.numel():
                return False
            cn.copy_(a.to(device=cn.device, dtype=cn.dtype))
        else:
            if a.numel() != cn.shape[0]:
                return False
            cn.copy_(a.to(device=cn.device, dtype=cn.dtype).view(-1, 1).expand_as(cn))
        if cd.dim() == 1:
            if b.numel() != cd.numel():
                return False
            cd.copy_(b.to(device=cd.device, dtype=cd.dtype))
        else:
            if b.numel() != cd.shape[0]:
                return False
            cd.copy_(b.to(device=cd.device, dtype=cd.dtype).view(-1, 1).expand_as(cd))
    return True


def propose_tanh_family(
    ctx: Any,
    target: AtomNode,
    transform_hint: Optional[Any] = None,
    sat_specs: Optional[Sequence[Any]] = None,
) -> List[Candidate]:
    """
    tanh family:
      - tanh_rat:      s * tanh(r(x))        (s is constant poly)
      - tanh_rat_amp:  A(x) * tanh(r(x))     (A is degree-2 poly)

    Implement tanh via: (exp(2r)-1)/(exp(2r)+1) using existing AST nodes.
    """
    st = ctx.state
    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) < 1:
        return []

    _key = "_".join(str(i) for i in var_idxs)

    want_simple = False
    want_amp = False
    if transform_hint is not None and getattr(transform_hint, "ok", False):
        if getattr(transform_hint, "best_name", "") == "atanh":
            want_simple = True
    if sat_specs is not None and len(sat_specs) > 0:
        want_amp = True
        want_simple = True
    if not (want_simple or want_amp):
        return []

    # --- shared tanh core (exp(2r)-1)/(exp(2r)+1) ---
    # NOTE: Avoid DAGs in the AST: do not reuse the same node object twice
    # (ASTCompositeAdaptor caches by node id and expects a strict tree).
    # We therefore build independent (but identically initialised) +/- branches.
    tag_r_p = f"tr_tanh_rp_2_2_{_key}"
    tag_r_m = f"tr_tanh_rm_2_2_{_key}"
    tag_c2_p = f"tr_tanh_c2p_{_key}"
    tag_c2_m = f"tr_tanh_c2m_{_key}"
    tag_p1 = f"tr_tanh_p1_{_key}"
    tag_m1 = f"tr_tanh_m1_{_key}"

    r_atom_p = AtomNode(
        kind="ratpoly",
        var_idxs=var_idxs,
        kwargs=dense_rational_support_kwargs(
            n_inputs=len(var_idxs),
            degree_num=2,
            degree_den=2,
        ),
        tag=tag_r_p,
    )
    r_atom_m = AtomNode(
        kind="ratpoly",
        var_idxs=var_idxs,
        kwargs=dense_rational_support_kwargs(
            n_inputs=len(var_idxs),
            degree_num=2,
            degree_den=2,
        ),
        tag=tag_r_m,
    )
    c2_p = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_c2_p)
    c2_m = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_c2_m)
    p1 = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_p1)
    m1 = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_m1)
    exp2r_p = ExpNode(MulNode(c2_p, r_atom_p))
    exp2r_m = ExpNode(MulNode(c2_m, r_atom_m))
    tanh_expr = MulNode(AddNode(exp2r_p, m1), PowNode(AddNode(exp2r_m, p1), exponent=-1.0))

    cands: List[Candidate] = []

    if want_simple:
        tag_s = f"tr_tanh_scale_{_key}"
        s_atom = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_s)
        new_sub = MulNode(s_atom, tanh_expr)
        cand_root = replace_atom_in_ast(st.root, target, new_sub)
        label = "tanh_rat"

        def _init(root_new: Node, model_new: nn.Module, _old_state=st, _target=target):
            from .stageB import _poly_zero_and_set, _SubtreeModel, build_atom_to_leaf_map

            atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
            subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)

            xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
            xsub = xb[:, list(var_idxs)].to(dtype=torch.float64)
            with torch.no_grad():
                f_old = subtree_old.forward(xb).view(-1).to(dtype=torch.float64)

            leaf_s = _find_leaf_by_tag(root_new, model_new, tag_s)
            leaf_r_p = _find_leaf_by_tag(root_new, model_new, tag_r_p)
            leaf_r_m = _find_leaf_by_tag(root_new, model_new, tag_r_m)
            leaf_c2_p = _find_leaf_by_tag(root_new, model_new, tag_c2_p)
            leaf_c2_m = _find_leaf_by_tag(root_new, model_new, tag_c2_m)
            leaf_p1 = _find_leaf_by_tag(root_new, model_new, tag_p1)
            leaf_m1 = _find_leaf_by_tag(root_new, model_new, tag_m1)
            if leaf_s is None or leaf_r_p is None or leaf_r_m is None:
                return

            if leaf_c2_p is not None:
                _poly_zero_and_set(leaf_c2_p, {(0,): 2.0})
            if leaf_c2_m is not None:
                _poly_zero_and_set(leaf_c2_m, {(0,): 2.0})
            if leaf_p1 is not None:
                _poly_zero_and_set(leaf_p1, {(0,): 1.0})
            if leaf_m1 is not None:
                _poly_zero_and_set(leaf_m1, {(0,): -1.0})

            fin = torch.isfinite(f_old)
            if int(fin.sum().item()) < 150:
                return
            q = f_old[fin].abs()
            s0 = float(torch.quantile(q, 0.95).item()) if q.numel() > 0 else 1.0
            s0 = max(s0, 1e-6) * 1.05

            u = (f_old / s0).clamp(-0.999999, 0.999999)
            m = torch.isfinite(u) & (u.abs() < 0.999999)
            if int(m.sum().item()) < 200:
                return

            y = torch.atanh(u[m]).to(dtype=torch.float64)
            X = xsub[m]
            fit = _fit_rational_coeffs_nd_eig(
                X,
                y,
                deg_num=2,
                deg_den=2,
                cfg=_ctx_rat_sparsify_cfg(ctx),
            )
            if fit is not None:
                a, b, _, _ = fit
                _set_ratpoly_coeffs_by_vectors(leaf_r_p, a, b)
                _set_ratpoly_coeffs_by_vectors(leaf_r_m, a, b)
            _poly_zero_and_set(leaf_s, {(0,): float(s0)})

        _init._after_analytic_init = True
        cands.append(Candidate(label, cand_root, _init))

    if want_amp:
        tag_a = f"tr_tanh_amp_deg2_{_key}"
        a_atom = AtomNode(
            kind="poly", var_idxs=var_idxs, kwargs={"degree": 2, "min_total": 0}, tag=tag_a
        )
        new_sub = MulNode(a_atom, tanh_expr)
        cand_root = replace_atom_in_ast(st.root, target, new_sub)
        label = "tanh_rat_amp"

        def _init(root_new: Node, model_new: nn.Module, _old_state=st, _target=target):
            from .stageB import _poly_zero_and_set, _SubtreeModel, build_atom_to_leaf_map

            atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
            subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)

            xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
            xsub = xb[:, list(var_idxs)].to(dtype=torch.float64)
            with torch.no_grad():
                f_old = subtree_old.forward(xb).view(-1).to(dtype=torch.float64)

            leaf_a = _find_leaf_by_tag(root_new, model_new, tag_a)
            leaf_r_p = _find_leaf_by_tag(root_new, model_new, tag_r_p)
            leaf_r_m = _find_leaf_by_tag(root_new, model_new, tag_r_m)
            leaf_c2_p = _find_leaf_by_tag(root_new, model_new, tag_c2_p)
            leaf_c2_m = _find_leaf_by_tag(root_new, model_new, tag_c2_m)
            leaf_p1 = _find_leaf_by_tag(root_new, model_new, tag_p1)
            leaf_m1 = _find_leaf_by_tag(root_new, model_new, tag_m1)
            if leaf_a is None or leaf_r_p is None or leaf_r_m is None:
                return

            if leaf_c2_p is not None:
                _poly_zero_and_set(leaf_c2_p, {(0,): 2.0})
            if leaf_c2_m is not None:
                _poly_zero_and_set(leaf_c2_m, {(0,): 2.0})
            if leaf_p1 is not None:
                _poly_zero_and_set(leaf_p1, {(0,): 1.0})
            if leaf_m1 is not None:
                _poly_zero_and_set(leaf_m1, {(0,): -1.0})

            fin = torch.isfinite(f_old)
            if int(fin.sum().item()) < 250:
                return
            q = f_old[fin].abs()
            A0 = float(torch.quantile(q, 0.95).item()) if q.numel() > 0 else 1.0
            A0 = max(A0, 1e-6) * 1.05

            # 1) fit r from global A0
            u0 = (f_old / A0).clamp(-0.999999, 0.999999)
            m0 = torch.isfinite(u0) & (u0.abs() < 0.999999)
            if int(m0.sum().item()) < 250:
                return
            y0 = torch.atanh(u0[m0]).to(dtype=torch.float64)
            X0 = xsub[m0]
            fit0 = _fit_rational_coeffs_nd_eig(
                X0,
                y0,
                deg_num=2,
                deg_den=2,
                cfg=_ctx_rat_sparsify_cfg(ctx),
            )
            if fit0 is not None:
                a0, b0, _, _ = fit0
                _set_ratpoly_coeffs_by_vectors(leaf_r_p, a0, b0)
                _set_ratpoly_coeffs_by_vectors(leaf_r_m, a0, b0)

            # 2) fit A(x) from f / tanh(r)
            with torch.no_grad():
                r_pred = leaf_r_p(xsub).view(-1).to(dtype=torch.float64)
                tanh_r = torch.tanh(r_pred)
            mA = torch.isfinite(tanh_r) & (tanh_r.abs() > 0.15) & torch.isfinite(f_old)
            if int(mA.sum().item()) >= 250:
                A_tgt = (f_old[mA] / tanh_r[mA]).to(dtype=torch.float64)
                X_A = xsub[mA]
                try:
                    coeffA, _ = _fit_poly_coeffs_ls(X_A, A_tgt, degree=2)
                    _set_poly_like_coeffs_by_vector(leaf_a, coeffA)
                except Exception:
                    pass

            # 3) refit r from atanh(f/A(x))
            with torch.no_grad():
                A_pred = leaf_a(xsub).view(-1).to(dtype=torch.float64)
            m1 = torch.isfinite(A_pred) & (A_pred.abs() > 1e-6) & torch.isfinite(f_old)
            if int(m1.sum().item()) < 250:
                return
            u1 = (f_old[m1] / A_pred[m1]).clamp(-0.999999, 0.999999)
            m1b = torch.isfinite(u1) & (u1.abs() < 0.999999)
            if int(m1b.sum().item()) < 250:
                return
            y1 = torch.atanh(u1[m1b]).to(dtype=torch.float64)
            X1 = xsub[m1][m1b]
            fit1 = _fit_rational_coeffs_nd_eig(
                X1,
                y1,
                deg_num=2,
                deg_den=2,
                cfg=_ctx_rat_sparsify_cfg(ctx),
            )
            if fit1 is not None:
                a1, b1, _, _ = fit1
                _set_ratpoly_coeffs_by_vectors(leaf_r_p, a1, b1)
                _set_ratpoly_coeffs_by_vectors(leaf_r_m, a1, b1)

        _init._after_analytic_init = True
        cands.append(Candidate(label, cand_root, _init))

    return cands


def propose_symexp_denom_family(
    ctx: Any,
    target: AtomNode,
    transform_hint: Optional[Any] = None,
    sat_specs: Optional[Sequence[Any]] = None,
) -> List[Candidate]:
    st = ctx.state
    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) < 1:
        return []
    _key = "_".join(str(i) for i in var_idxs)
    want_simple = False
    want_amp = False
    if transform_hint is not None and getattr(transform_hint, "ok", False):
        if getattr(transform_hint, "best_name", "") == "recip":
            want_simple = True
    if sat_specs is not None and len(sat_specs) > 0:
        want_simple = True
        want_amp = True
    if not (want_simple or want_amp):
        return []
    # NOTE: Avoid DAGs in the AST: build independent +/- branches.
    tag_r_p = f"tr_symexp_rp_2_2_{_key}"
    tag_r_m = f"tr_symexp_rm_2_2_{_key}"
    tag_c_p = f"tr_symexp_cp_{_key}"
    tag_c_m = f"tr_symexp_cm_{_key}"
    tag_m1 = f"tr_symexp_m1_{_key}"
    r_atom_p = AtomNode(
        kind="ratpoly",
        var_idxs=var_idxs,
        kwargs=dense_rational_support_kwargs(
            n_inputs=len(var_idxs),
            degree_num=2,
            degree_den=2,
        ),
        tag=tag_r_p,
    )
    r_atom_m = AtomNode(
        kind="ratpoly",
        var_idxs=var_idxs,
        kwargs=dense_rational_support_kwargs(
            n_inputs=len(var_idxs),
            degree_num=2,
            degree_den=2,
        ),
        tag=tag_r_m,
    )
    c_atom_p = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_c_p)
    c_atom_m = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_c_m)
    m1_atom = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_m1)
    cr_p = MulNode(c_atom_p, r_atom_p)
    cr_m = MulNode(c_atom_m, r_atom_m)
    exp_p = ExpNode(cr_p)
    exp_m = ExpNode(MulNode(m1_atom, cr_m))
    den = AddNode(exp_p, exp_m)
    inv_den = PowNode(den, exponent=-1.0)
    cands: List[Candidate] = []
    if want_simple:
        tag_s = f"tr_symexp_scale_{_key}"
        s_atom = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_s)
        new_sub = MulNode(s_atom, inv_den)
        cand_root = replace_atom_in_ast(st.root, target, new_sub)
        label = "symexp_denom_rat"
        def _init(root_new: Node, model_new: nn.Module, _old_state=st, _target=target):
            from .stageB import _poly_zero_and_set, _SubtreeModel, build_atom_to_leaf_map
            atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
            subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)
            xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
            xsub = xb[:, list(var_idxs)].to(dtype=torch.float64)
            with torch.no_grad():
                f_old = subtree_old.forward(xb).view(-1).to(dtype=torch.float64)
            leaf_s = _find_leaf_by_tag(root_new, model_new, tag_s)
            leaf_r_p = _find_leaf_by_tag(root_new, model_new, tag_r_p)
            leaf_r_m = _find_leaf_by_tag(root_new, model_new, tag_r_m)
            leaf_c_p = _find_leaf_by_tag(root_new, model_new, tag_c_p)
            leaf_c_m = _find_leaf_by_tag(root_new, model_new, tag_c_m)
            leaf_m1 = _find_leaf_by_tag(root_new, model_new, tag_m1)
            if leaf_s is None or leaf_r_p is None or leaf_r_m is None:
                return
            if leaf_c_p is not None:
                _poly_zero_and_set(leaf_c_p, {(0,): 1.0})
            if leaf_c_m is not None:
                _poly_zero_and_set(leaf_c_m, {(0,): 1.0})
            if leaf_m1 is not None:
                _poly_zero_and_set(leaf_m1, {(0,): -1.0})
            fin = torch.isfinite(f_old)
            if int(fin.sum().item()) < 250:
                return
            fF = f_old[fin]
            xF = xsub[fin]
            pos = float((fF > 0).to(dtype=torch.float64).mean().item())
            neg = float((fF < 0).to(dtype=torch.float64).mean().item())
            if max(pos, neg) < 0.85:
                return
            sgn = 1.0 if pos >= neg else -1.0
            fP = (sgn * fF)
            mP = torch.isfinite(fP) & (fP > 1e-12)
            if int(mP.sum().item()) < 250:
                return
            fP2 = fP[mP]
            s0 = float(torch.quantile(fP2, 0.99).item()) if fP2.numel() > 0 else 1.0
            s0 = max(s0, 1e-8) * 2.0 * 1.05
            for _ in range(6):
                u = s0 / (2.0 * fP2)
                frac = float((u > 1.0 + 1e-6).to(dtype=torch.float64).mean().item())
                if frac >= 0.6:
                    break
                s0 *= 2.0
            u = s0 / (2.0 * fP2)
            mU = torch.isfinite(u) & (u > 1.0 + 1e-6)
            if int(mU.sum().item()) < 250:
                return
            uu = u[mU].to(dtype=torch.float64)
            yy = torch.log(uu + torch.sqrt(torch.clamp(uu * uu - 1.0, min=1e-12)))
            X = xF[mP][mU]
            fit = _fit_rational_coeffs_nd_eig(
                X,
                yy,
                deg_num=2,
                deg_den=2,
                cfg=_ctx_rat_sparsify_cfg(ctx),
            )
            if fit is not None:
                a, b, _, _ = fit
                _set_ratpoly_coeffs_by_vectors(leaf_r_p, a, b)
                _set_ratpoly_coeffs_by_vectors(leaf_r_m, a, b)
            _poly_zero_and_set(leaf_s, {(0,): float(sgn * s0)})
        _init._after_analytic_init = True
        cands.append(Candidate(label, cand_root, _init))
    if want_amp:
        tag_a = f"tr_symexp_amp_deg2_{_key}"
        a_atom = AtomNode(kind="poly", var_idxs=var_idxs, kwargs={"degree": 2, "min_total": 0}, tag=tag_a)
        new_sub = MulNode(a_atom, inv_den)
        cand_root = replace_atom_in_ast(st.root, target, new_sub)
        label = "symexp_denom_rat_amp"
        def _init(root_new: Node, model_new: nn.Module, _old_state=st, _target=target):
            from .stageB import _poly_zero_and_set, _SubtreeModel, build_atom_to_leaf_map
            atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
            subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)
            xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
            xsub = xb[:, list(var_idxs)].to(dtype=torch.float64)
            with torch.no_grad():
                f_old = subtree_old.forward(xb).view(-1).to(dtype=torch.float64)
            leaf_a = _find_leaf_by_tag(root_new, model_new, tag_a)
            leaf_r_p = _find_leaf_by_tag(root_new, model_new, tag_r_p)
            leaf_r_m = _find_leaf_by_tag(root_new, model_new, tag_r_m)
            leaf_c_p = _find_leaf_by_tag(root_new, model_new, tag_c_p)
            leaf_c_m = _find_leaf_by_tag(root_new, model_new, tag_c_m)
            leaf_m1 = _find_leaf_by_tag(root_new, model_new, tag_m1)
            if leaf_a is None or leaf_r_p is None or leaf_r_m is None:
                return
            if leaf_c_p is not None:
                _poly_zero_and_set(leaf_c_p, {(0,): 1.0})
            if leaf_c_m is not None:
                _poly_zero_and_set(leaf_c_m, {(0,): 1.0})
            if leaf_m1 is not None:
                _poly_zero_and_set(leaf_m1, {(0,): -1.0})
            fin = torch.isfinite(f_old)
            if int(fin.sum().item()) < 350:
                return
            fF = f_old[fin]
            xF = xsub[fin]
            pos = float((fF > 0).to(dtype=torch.float64).mean().item())
            neg = float((fF < 0).to(dtype=torch.float64).mean().item())
            if max(pos, neg) < 0.85:
                return
            sgn = 1.0 if pos >= neg else -1.0
            fP = (sgn * fF)
            mP = torch.isfinite(fP) & (fP > 1e-12)
            if int(mP.sum().item()) < 350:
                return
            fP2 = fP[mP]
            A0 = float(torch.quantile(fP2, 0.99).item()) if fP2.numel() > 0 else 1.0
            A0 = max(A0, 1e-8) * 2.0 * 1.05
            for _ in range(6):
                u = A0 / (2.0 * fP2)
                frac = float((u > 1.0 + 1e-6).to(dtype=torch.float64).mean().item())
                if frac >= 0.6:
                    break
                A0 *= 2.0
            u0 = A0 / (2.0 * fP2)
            mU0 = torch.isfinite(u0) & (u0 > 1.0 + 1e-6)
            if int(mU0.sum().item()) < 350:
                return
            uu0 = u0[mU0].to(dtype=torch.float64)
            y0 = torch.log(uu0 + torch.sqrt(torch.clamp(uu0 * uu0 - 1.0, min=1e-12)))
            X0 = xF[mP][mU0]
            fit0 = _fit_rational_coeffs_nd_eig(
                X0,
                y0,
                deg_num=2,
                deg_den=2,
                cfg=_ctx_rat_sparsify_cfg(ctx),
            )
            if fit0 is not None:
                a0, b0, _, _ = fit0
                _set_ratpoly_coeffs_by_vectors(leaf_r_p, a0, b0)
                _set_ratpoly_coeffs_by_vectors(leaf_r_m, a0, b0)
            with torch.no_grad():
                r_pred = leaf_r_p(xsub).view(-1).to(dtype=torch.float64)
                r_pred = r_pred.clamp(-30.0, 30.0)
                den = torch.exp(r_pred) + torch.exp(-r_pred)
            mA = torch.isfinite(den) & torch.isfinite(f_old) & (den > 1e-12)
            if int(mA.sum().item()) >= 350:
                A_tgt = (f_old[mA] * den[mA]).to(dtype=torch.float64)
                Aabs = A_tgt.abs()
                if Aabs.numel() > 0:
                    thr = float(torch.quantile(Aabs, 0.98).item())
                    mA2 = mA.clone()
                    mA2[mA] = Aabs <= max(thr, 1e-12)
                else:
                    mA2 = mA
                if int(mA2.sum().item()) >= 350:
                    X_A = xsub[mA2]
                    A_fit = (f_old[mA2] * den[mA2]).to(dtype=torch.float64)
                    try:
                        coeffA, _ = _fit_poly_coeffs_ls(X_A, A_fit, degree=2)
                        _set_poly_like_coeffs_by_vector(leaf_a, coeffA)
                    except Exception:
                        pass
            with torch.no_grad():
                A_pred = leaf_a(xsub).view(-1).to(dtype=torch.float64)
            m1b = torch.isfinite(A_pred) & torch.isfinite(f_old) & (A_pred.abs() > 1e-8) & (f_old.abs() > 1e-12) & (A_pred * f_old > 0)
            if int(m1b.sum().item()) < 350:
                return
            u1 = (A_pred[m1b] / (2.0 * f_old[m1b])).to(dtype=torch.float64)
            mU1 = torch.isfinite(u1) & (u1 > 1.0 + 1e-6)
            if int(mU1.sum().item()) < 350:
                return
            uu1 = u1[mU1].to(dtype=torch.float64)
            y1 = torch.log(uu1 + torch.sqrt(torch.clamp(uu1 * uu1 - 1.0, min=1e-12)))
            X1 = xsub[m1b][mU1]
            fit1 = _fit_rational_coeffs_nd_eig(
                X1,
                y1,
                deg_num=2,
                deg_den=2,
                cfg=_ctx_rat_sparsify_cfg(ctx),
            )
            if fit1 is not None:
                a1, b1, _, _ = fit1
                _set_ratpoly_coeffs_by_vectors(leaf_r_p, a1, b1)
                _set_ratpoly_coeffs_by_vectors(leaf_r_m, a1, b1)
        _init._after_analytic_init = True
        cands.append(Candidate(label, cand_root, _init))
    return cands


def _tie_poly_like(src_leaf: nn.Module, dst_leaf: nn.Module) -> None:
    """Tie polynomial parameters so dst shares src's .coeffs (stabilises LM)."""
    try:
        src = getattr(src_leaf, "model", getattr(src_leaf, "core", src_leaf))
        dst = getattr(dst_leaf, "model", getattr(dst_leaf, "core", dst_leaf))
        if hasattr(src, "coeffs") and hasattr(dst, "coeffs"):
            if getattr(src, "coeffs").shape == getattr(dst, "coeffs").shape:
                with torch.no_grad():
                    dst.coeffs.copy_(src.coeffs)
                dst.coeffs = src.coeffs
    except Exception:
        return


def propose_sinc_family(
    ctx: Any,
    target: AtomNode,
    trig_spec: Optional[Any] = None,
    *,
    degree_arg: int = 2,
    p: int = 2,
) -> List[Candidate]:
    """sinc family templates.

    Base form:
        s * (sin(P(x))^p) * (P(x)^(-p))
      i.e. s * (sin(P)/P)^p

    For AIF-style problems we often need a *prefactor* as well, e.g.
        x_k * sinc((x_i-x_j)*x_k)^2
    where the same variable appears both in the argument and as a multiplicative
    factor. The original template could never fit the odd-in-x_k cases.

    This routine now proposes (when possible) an additional candidate:
        s * A(x_m) * (sin(P(x))^p) * (P(x)^(-p))
    where A is a *univariate* degree-1 polynomial on a "mod" axis.

    It also uses trig-structure hints (difference/product) to seed *bilinear*
    argument terms like (x_i-x_j)*x_k, and ties the numerator/denominator
    argument polynomials to share parameters to stabilise LM.
    """
    st = ctx.state
    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) < 1:
        return []

    if trig_spec is None:
        return []

    # Detect compound univariate atoms: for z = f(x_i, x_j, ...), we want P(z) = az + b
    # instead of P(x_i, x_j, ...) which has too many parameters.
    is_compound_univariate = has_nontrivial_input(target) and target.n_in == 1
    # Compound multivariate atoms (e.g. nn(x0/x1, x6)) operate in INPUT
    # position space: polys consume the compound input expressions, and raw
    # trig axes must map to a trivial input slot.
    is_compound_multi = has_nontrivial_input(target) and target.n_in >= 2

    axis = int(getattr(trig_spec, "axis", -1))
    if axis < 0 or (axis not in var_idxs):
        return []
    axis_input_pos: Optional[int] = None
    if is_compound_multi:
        axis_input_pos = trivial_input_position(target, axis)
        if axis_input_pos is None:
            # The trig axis lives inside a nontrivial compound input; this
            # family cannot isolate it.  Skip gracefully — never fall back to
            # raw var_idxs, which would drop the compound semantics.
            return []

    # Shared wrapper-policy gate: allow disabling trig templates globally and
    # optionally avoid squared variants when trig-squares are disabled.
    try:
        hp = getattr(ctx, "lm_hp", None)
        if hp is not None:
            pol = macro_arg_wrapper_policy(ctx, hp, target)
            if not bool(pol.trig):
                return []
            # Honour explicit global disable of trig squares, but don't suppress p>1
            # merely because global trig-hint discovery missed (common for compound variables).
            trig_sq_enable = bool(getattr(hp, "macro_arg_trig_squares_enable", True))
            if (int(p) != 1) and (not trig_sq_enable):
                p = 1
    except Exception:
        pass

    # Snap omega to a simple nearby value when it is clearly close.
    try:
        omega_raw = float(getattr(trig_spec, "omega", 0.0))
    except Exception:
        omega_raw = 0.0
    omega = float(snap_omega(omega_raw))

    # For sin^2, dominant frequency tends to be ~2*k. Seed k≈omega/2.
    k0 = omega / float(p) if p != 0 else omega
    if not math.isfinite(k0) or abs(k0) < 1e-6:
        k0 = 1.0

    _key = "_".join(str(i) for i in var_idxs)
    label = f"sinc_p{int(p)}"

    # Optional trig-structure hint (difference/product) can improve initialisation
    # for arguments like (x_i-x_j)*x_k or x_i*x_j.
    hint = None
    try:
        hint = getattr(ctx, "trig_structure_by_axis", {}) or {}
        hint = hint.get(int(axis), None)
    except Exception:
        hint = None

    # Variants: (mode, partner_axis, mod_axis, use_amp)
    # use_amp: if True, multiply by a univariate linear prefactor A(x_mod)
    variants: List[Tuple[str, Optional[int], Optional[int], bool]] = [("axis_linear", None, None, False)]
    try:
        kind = getattr(hint, "kind", None)
        partner = getattr(hint, "partner", None)
        partner_i = int(partner) if partner is not None else None
        if (
            kind in ("difference", "product")
            and partner_i is not None
            and partner_i in var_idxs
            and (
                not is_compound_multi
                or trivial_input_position(target, partner_i) is not None
            )
        ):
            if kind == "difference":
                if len(var_idxs) == 2 or int(degree_arg) < 2:
                    variants.insert(0, ("diff_linear", partner_i, None, False))
                else:
                    others = [int(i) for i in var_idxs if int(i) not in (axis, partner_i)]
                    if is_compound_multi:
                        others = [
                            m
                            for m in others
                            if trivial_input_position(target, m) is not None
                        ]
                        if not others:
                            # All modulation axes are buried inside compound
                            # inputs, but axis and partner are both trivial:
                            # the plain difference variant is still valid.
                            variants.insert(0, ("diff_linear", partner_i, None, False))
                    for m in others:
                        # Add both with and without amplitude prefactor
                        variants.insert(0, ("diff_bilin", partner_i, int(m), False))
                        variants.insert(0, ("diff_bilin", partner_i, int(m), True))
            elif kind == "product" and int(degree_arg) >= 2:
                variants.insert(0, ("prod_bilin", partner_i, None, False))
    except Exception:
        pass

    def _make_init(
        mode: str,
        partner_axis: Optional[int],
        mod_axis: Optional[int],
        use_amp: bool,
        tag_s_inner: str,
        tag_p_sin_inner: str,
        tag_p_den_inner: str,
        tag_amp_inner: Optional[str],
        z_expr_inner: Optional[Node] = None,
        is_compound_univariate_inner: bool = False,
        is_compound_multi_inner: bool = False,
        input_positions_inner: Optional[
            Tuple[Optional[int], Optional[int], Optional[int]]
        ] = None,
    ):
        def _init(root_new: Node, model_new: nn.Module, _old_state=st, _target=target):
            from .stageB import _poly_zero_and_set, _SubtreeModel, build_atom_to_leaf_map

            atom_to_leaf_old = build_atom_to_leaf_map(_old_state.root, _old_state.model)
            subtree_old = _SubtreeModel(root=_target, atom_to_leaf=atom_to_leaf_old)

            xb = _first_batch_x(ctx.train_loader, device=ctx.device, dtype=ctx.dtype)
            xsub = xb[:, list(var_idxs)].to(dtype=torch.float64)
            x_med = xsub.median(dim=0).values
            with torch.no_grad():
                f_old = subtree_old.forward(xb).view(-1).to(dtype=torch.float64)

            leaf_s = _find_leaf_by_tag(root_new, model_new, tag_s_inner)
            leaf_p1 = _find_leaf_by_tag(root_new, model_new, tag_p_sin_inner)
            leaf_p2 = _find_leaf_by_tag(root_new, model_new, tag_p_den_inner)
            if leaf_s is None or leaf_p1 is None or leaf_p2 is None:
                return

            xin = None
            if is_compound_multi_inner:
                # Compound multivariate: seeds and evaluation live in INPUT
                # position space (columns = the atom's input expressions).
                from ._candidate_builders_common import _build_atom_input_tensor

                with torch.no_grad():
                    try:
                        xin = _build_atom_input_tensor(_target, xb).to(
                            dtype=torch.float64
                        )
                    except Exception:
                        return
                n = int(xin.shape[1])
                x_med_eff = xin.median(dim=0).values
                pos_axis = int(input_positions_inner[0])
            else:
                n = len(var_idxs)
                x_med_eff = x_med
                pos_axis = int(var_idxs.index(axis))

            def _partner_pos() -> int:
                if is_compound_multi_inner:
                    return int(input_positions_inner[1])
                return int(var_idxs.index(int(partner_axis)))

            def _mod_pos() -> int:
                if is_compound_multi_inner:
                    return int(input_positions_inner[2])
                return int(var_idxs.index(int(mod_axis)))

            # Precompute compound input values (z) for compound-univariate targets.
            z_vals = None
            if is_compound_univariate_inner and z_expr_inner is not None:
                with torch.no_grad():
                    try:
                        z_vals = eval_input_expr(z_expr_inner, xb).to(dtype=torch.float64)
                        if z_vals.dim() == 1:
                            z_vals = z_vals.unsqueeze(1)
                    except Exception:
                        # If we can't evaluate the compound expression on the batch,
                        # skip custom initialisation.
                        z_vals = None

            # Optional prefactor A(x_mod) ≈ x_mod
            amp_vals = None
            if use_amp and mod_axis is not None and tag_amp_inner is not None:
                leaf_amp = _find_leaf_by_tag(root_new, model_new, tag_amp_inner)
                if leaf_amp is not None:
                    _poly_zero_and_set(leaf_amp, {(0,): 0.0, (1,): 1.0})
                    with torch.no_grad():
                        xamp = xb[:, [int(mod_axis)]].to(dtype=torch.float64)
                        amp_vals = leaf_amp(xamp).view(-1).to(dtype=torch.float64)

            seeds: Dict[Tuple[int, ...], float] = {}

            # For compound univariate: polynomial is P(z) = az + b (degree 1 in z)
            # Seed as P(z) = k0*z (linear in z, no constant term initially)
            if is_compound_univariate_inner:
                # NOTE: omega estimates from a sine fit can be a poor proxy for
                # sinc-like envelopes. We therefore do a small coarse search over
                # plausible k values to find a good initialization for P(z)=k*z.
                k_seed = float(k0)
                k_best = abs(k_seed) if math.isfinite(k_seed) and abs(k_seed) > 1e-12 else 1.0
                best_mse = float("inf")
                if z_vals is not None:
                    z = z_vals[:, 0].view(-1)
                    # Candidate grid: powers of two around k_seed + a few common constants.
                    k0_base = abs(k_seed) if math.isfinite(k_seed) and abs(k_seed) > 1e-12 else 1.0
                    k_cands = [k0_base * (2.0 ** m) for m in (-3, -2, -1, 0, 1, 2, 3)]
                    k_cands += [0.5, 1.0, math.pi / 2.0, math.pi, 2.0 * math.pi, 1.0 / math.pi, 1.0 / (2.0 * math.pi)]
                    # Deduplicate while preserving order.
                    seen = set()
                    k_grid = []
                    for kk in k_cands:
                        if not math.isfinite(kk) or kk <= 0:
                            continue
                        if kk in seen:
                            continue
                        seen.add(kk)
                        k_grid.append(float(kk))

                    with torch.no_grad():
                        y = f_old.view(-1).to(dtype=torch.float64)
                        a = amp_vals.view(-1).to(dtype=torch.float64) if amp_vals is not None else None
                        for kk in k_grid:
                            arg = (kk * z).to(dtype=torch.float64)
                            base = torch.sinc(arg / math.pi)
                            prod = base.pow(float(p))
                            if a is not None:
                                prod = prod * a
                            denom = (prod @ prod).clamp_min(1e-12)
                            s = (y @ prod) / denom
                            mse = ((y - s * prod) ** 2).mean().item()
                            if math.isfinite(mse) and mse < best_mse:
                                best_mse = mse
                                k_best = float(kk)

                # sinc(arg) is even in arg, so we keep k positive.
                seeds = {(0,): 0.0, (1,): float(k_best)}
            else:
                def _lin(pos: int):
                    return tuple(1 if i == pos else 0 for i in range(n))

                def _cross(pos_a: int, pos_b: int):
                    return tuple(1 if (i == pos_a or i == pos_b) else 0 for i in range(n))

                # Default: seed linear term along detected trig axis.
                if mode == "axis_linear" or partner_axis is None:
                    seeds[_lin(pos_axis)] = float(k0)

                elif mode == "diff_linear":
                    pos_p = _partner_pos()
                    seeds[_lin(pos_axis)] = float(k0)
                    seeds[_lin(pos_p)] = -float(k0)

                elif mode == "prod_bilin":
                    pos_p = _partner_pos()
                    xref = float(x_med_eff[pos_p].item())
                    if not math.isfinite(xref) or abs(xref) < 1e-10:
                        seeds[_lin(pos_axis)] = float(k0)
                    else:
                        seeds[_cross(pos_axis, pos_p)] = float(k0) / float(xref)

                elif mode == "diff_bilin":
                    # Use a bilinear seed to represent (x_axis - x_partner)*x_mod.
                    # This is crucial for sinc((x_i-x_j)*x_k) cases.
                    pos_p = _partner_pos()
                    if mod_axis is None or int(mod_axis) not in var_idxs:
                        seeds[_lin(pos_axis)] = float(k0)
                    else:
                        pos_m = _mod_pos()
                        xref = float(x_med_eff[pos_m].item())
                        if not math.isfinite(xref) or abs(xref) < 1e-10:
                            seeds[_lin(pos_axis)] = float(k0)
                            seeds[_lin(pos_p)] = -float(k0)
                        else:
                            c = float(k0) / float(xref)
                            seeds[_cross(pos_axis, pos_m)] = c
                            seeds[_cross(pos_p, pos_m)] = -c

            _poly_zero_and_set(leaf_p1, seeds)
            _poly_zero_and_set(leaf_p2, seeds)
            _tie_poly_like(leaf_p1, leaf_p2)

            # Fit scale by LS against the seeded base, using full P(x) (not just a single axis).
            # For compound univariate atoms: P(z) where z = z_expr_inner(x)
            # The leaf expects a single-column input (the compound variable value), not raw xsub.
            with torch.no_grad():
                if is_compound_univariate_inner and z_expr_inner is not None:
                    # (z_vals already computed above)
                    if z_vals is None:
                        return
                    arg = leaf_p1(z_vals.to(device=ctx.device, dtype=torch.float64))
                elif is_compound_multi_inner:
                    if xin is None:
                        return
                    arg = leaf_p1(xin.to(device=ctx.device, dtype=torch.float64))
                else:
                    arg = leaf_p1(xsub.to(device=ctx.device, dtype=torch.float64))
                if arg.dim() == 2:
                    arg = arg[:, 0]
                # Use torch.sinc for numerical stability and correct behaviour near 0.
                base = torch.sinc(arg / math.pi)
                base = base ** float(p) if p != 1 else base
                if amp_vals is not None:
                    base = base * amp_vals
                m = torch.isfinite(base) & torch.isfinite(f_old)
                if int(m.sum().item()) < 80:
                    return
                b = base[m]
                y = f_old[m]
                s = float((b @ y) / (b @ b + 1e-12))
                _poly_zero_and_set(leaf_s, {(0,): float(s)})

        _init._after_analytic_init = True
        return _init

    cands: List[Candidate] = []
    for mode, partner_axis, mod_axis, use_amp in variants:
        # Create unique tags for this variant
        amp_suffix = f"_{mode}_{int(use_amp)}"
        tag_s_v = f"tr_sinc_s_p{int(p)}_{_key}{amp_suffix}"
        tag_p_sin_v = f"tr_sinc_p_sin_p{int(p)}_deg{int(degree_arg)}_{_key}{amp_suffix}"
        tag_p_den_v = f"tr_sinc_p_den_p{int(p)}_deg{int(degree_arg)}_{_key}{amp_suffix}"
        tag_amp_v: Optional[str] = None

        # Build AST for this variant
        scale_atom_v = AtomNode(kind="poly", var_idxs=(var_idxs[0],), kwargs={"degree": 0}, tag=tag_s_v)

        # For compound univariate atoms, use degree=1 polynomial on z (2 params: az+b)
        # instead of full polynomial on all var_idxs (many more params).
        #
        # IMPORTANT: each AtomNode must own its own input_expr tree. Reusing the same
        # compound AST in multiple places makes the candidate a DAG, which Stage B
        # rejects (ast-not-tree).
        if is_compound_univariate:
            _z_expr = compound_input_expr(target)
            extra = list(extra_input_var_idxs(target))
            poly_kwargs_sin = {"degree": 1, "min_total": 0}
            poly_kwargs_den = {"degree": 1, "min_total": 0}
            sin_inputs = tuple([clone_ast(_z_expr)] + [Var(int(v)) for v in extra])
            den_inputs = tuple([clone_ast(_z_expr)] + [Var(int(v)) for v in extra])
        elif is_compound_multi:
            # Polys consume the compound input expressions (two INDEPENDENT
            # clones: reusing one tree would make the candidate a DAG, which
            # Stage B rejects as ast-not-tree).
            poly_kwargs_sin = {"degree": int(degree_arg), "min_total": 0}
            poly_kwargs_den = {"degree": int(degree_arg), "min_total": 0}
            sin_inputs = clone_inputs(target)
            den_inputs = clone_inputs(target)
        else:
            poly_kwargs_sin = {"degree": int(degree_arg), "min_total": 0}
            poly_kwargs_den = {"degree": int(degree_arg), "min_total": 0}
            sin_inputs = None
            den_inputs = None

        p_sin_v = AtomNode(
            kind="poly",
            var_idxs=var_idxs,
            kwargs={**poly_kwargs_sin},
            tag=tag_p_sin_v,
            inputs=sin_inputs,
        )
        p_den_v = AtomNode(
            kind="poly",
            var_idxs=var_idxs,
            kwargs={**poly_kwargs_den},
            tag=tag_p_den_v,
            inputs=den_inputs,
        )

        core = MulNode(PowNode(SinNode(p_sin_v), exponent=float(p)), PowNode(p_den_v, exponent=-float(p)))
        if use_amp and mod_axis is not None:
            tag_amp_v = f"tr_sinc_amp_{_key}{amp_suffix}"
            amp_atom = AtomNode(
                kind="poly",
                var_idxs=(int(mod_axis),),
                kwargs={"degree": 1, "min_total": 0},
                tag=tag_amp_v,
            )
            new_sub_v = MulNode(scale_atom_v, MulNode(amp_atom, core))
        else:
            new_sub_v = MulNode(scale_atom_v, core)

        cand_root_v = replace_atom_in_ast(st.root, target, new_sub_v)

        # Build log message
        if is_compound_multi:
            msg = (
                f"[Stage B]  Trying {label} on compound nn vars={var_idxs} "
                f"(arity={target.n_in}, input-position seeds, axis pos={axis_input_pos})"
            )
        elif is_compound_univariate:
            msg = f"[Stage B]  Trying {label} on compound nn vars={var_idxs} (poly: 1D degree=1, seed: k0*z)"
        elif mode == "axis_linear":
            msg = f"[Stage B]  Trying {label} on nn vars={var_idxs} (seed: axis x{axis})"
        elif mode == "diff_linear":
            msg = f"[Stage B]  Trying {label} on nn vars={var_idxs} (seed: x{axis}-x{partner_axis})"
        elif mode == "prod_bilin":
            msg = f"[Stage B]  Trying {label} on nn vars={var_idxs} (seed: x{axis}*x{partner_axis})"
        elif mode == "diff_bilin" and use_amp:
            msg = f"[Stage B]  Trying {label} on nn vars={var_idxs} (seed: (x{axis}-x{partner_axis})*x{mod_axis}, prefactor: x{mod_axis})"
        else:
            msg = f"[Stage B]  Trying {label} on nn vars={var_idxs} (seed: (x{axis}-x{partner_axis})*x{mod_axis})"

        sig_partner = -1 if partner_axis is None else int(partner_axis)
        sig_mod = -1 if mod_axis is None else int(mod_axis)
        sig = (int(p), int(degree_arg), int(axis), sig_partner, sig_mod, 1 if use_amp else 0)
        _pos_partner_in = (
            trivial_input_position(target, int(partner_axis))
            if (is_compound_multi and partner_axis is not None)
            else None
        )
        _pos_mod_in = (
            trivial_input_position(target, int(mod_axis))
            if (is_compound_multi and mod_axis is not None)
            else None
        )
        init_fn = _make_init(
            mode, partner_axis, mod_axis, use_amp, tag_s_v, tag_p_sin_v, tag_p_den_v, tag_amp_v,
            z_expr_inner=compound_input_expr(target), is_compound_univariate_inner=is_compound_univariate,
            is_compound_multi_inner=is_compound_multi,
            input_positions_inner=(axis_input_pos, _pos_partner_in, _pos_mod_in),
        )
        # The denominator polynomial is tied to the sin-argument polynomial
        # via _tie_poly_like, so its params are not free.  Declare the true
        # free-parameter count so the exhaustive sort in engine.py orders
        # sinc candidates correctly relative to untied candidates.
        _n_tied = count_atom_params(p_den_v)
        _n_full = sum(count_atom_params(a) for a in collect_all_atoms(cand_root_v))
        _meta = {"log": msg, "n_free_params": _n_full - _n_tied}
        cands.append(Candidate(label, cand_root_v, init_fn, meta=_meta, signature=sig))

    return cands


# ==============================
