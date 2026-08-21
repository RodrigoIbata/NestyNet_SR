# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Transform-based utilities for Stage B.

This module provides functions for sampling model values, checking transform
domain compatibility, and building transform-based rewrite candidates.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ExpNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    _select_inputs_for_var_group,
    replace_atom_in_ast,
    separability_proposal_to_ast,
)

from .atom_mapping import _collect_all_atoms, build_atom_to_leaf_map
from .models import _OuterTransformedSubtreeModel, _SubtreeModel


def _sample_u_values(
    model_u: nn.Module,
    train_loader,
    *,
    device,
    dtype,
    max_points: int = 2048,
    max_batches: int = 16,
):
    """Sample output values from a model over a dataloader."""
    ys = []
    n = 0
    model_u.eval()
    with torch.no_grad():
        for bi, batch in enumerate(train_loader):
            if bi >= max_batches or n >= max_points:
                break
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device=device, dtype=dtype)
            yb = model_u(xb)
            if yb.dim() == 1:
                yb = yb.view(-1, 1)
            ys.append(yb.detach().view(-1, 1))
            n += int(yb.numel())
    if not ys:
        return None
    u = torch.cat(ys, dim=0)
    return u[:max_points] if u.size(0) > max_points else u


def _domain_ok_frac(u: torch.Tensor, transform: str, eps: float = 1e-12) -> float:
    """Check what fraction of values are in the domain of the given transform."""
    if u is None or u.numel() == 0:
        return 0.0
    m = torch.isfinite(u)
    t = str(transform)
    if t == "identity":
        pass
    elif t == "log":
        m = m & (u > eps)
    elif t == "sqrt":
        m = m & (u >= -eps)
    elif t == "square":
        # Must check that u*u is finite (not just u), since u can overflow when squared
        m = m & torch.isfinite(u * u)
    elif t == "recip":
        m = m & (u.abs() > eps)
    elif t == "arcsin":
        m = m & (u.abs() < 1.0 - eps)
    else:
        return 0.0
    return float(m.float().mean().item())


def _sign_consistent_frac(u: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Check what fraction of values have consistent sign (mostly positive or mostly negative).

    Returns the fraction of samples that agree with the majority sign.
    For square-peel to work correctly (u = sqrt(v)), we need u to be mostly nonnegative
    (or mostly nonpositive, in which case we'd use -sqrt).
    """
    if u is None or u.numel() == 0:
        return 0.0
    u = u.detach()
    m = torch.isfinite(u)
    u_finite = u[m]
    if u_finite.numel() == 0:
        return 0.0
    n_pos = (u_finite >= -eps).sum().item()
    n_neg = (u_finite <= eps).sum().item()
    n_total = u_finite.numel()
    # Return the fraction that agrees with the majority sign
    return max(n_pos, n_neg) / n_total


def _domain_ok_frac_for_transform(u: torch.Tensor, transform: str, *, eps: float = 1e-12) -> float:
    """Fraction of samples where T(u) is real+finite."""
    u = u.detach()
    m = torch.isfinite(u)
    t = str(transform)
    if t == "identity":
        return float(m.float().mean().item()) if u.numel() else 0.0
    if t == "log":
        m = m & (u > eps)
        return float(m.float().mean().item()) if u.numel() else 0.0
    if t == "sqrt":
        m = m & (u >= -eps)
        return float(m.float().mean().item()) if u.numel() else 0.0
    if t == "square":
        # Must check that u*u is finite (not just u), since u can overflow when squared
        m = m & torch.isfinite(u * u)
        return float(m.float().mean().item()) if u.numel() else 0.0
    if t == "recip":
        m = m & (u.abs() > eps)
        return float(m.float().mean().item()) if u.numel() else 0.0
    if t == "arcsin":
        m = m & (u.abs() < 1.0 - eps)
        return float(m.float().mean().item()) if u.numel() else 0.0
    return 0.0


def _sample_subtree_values(
    model_u: nn.Module,
    datagen,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int = 2048,
    max_batches: int = 16,
):
    """Sample values from a subtree model over a data generator."""
    xs = []
    n = 0
    model_u.eval()
    with torch.no_grad():
        for bi, batch in enumerate(datagen):
            if bi >= max_batches or n >= max_points:
                break
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device=device, dtype=dtype)
            yb = model_u(xb)
            if yb.dim() == 1:
                yb = yb.view(-1, 1)
            xs.append(yb.detach().view(-1, 1))
            n += int(yb.numel())
    if not xs:
        return None
    u = torch.cat(xs, dim=0)
    if u.size(0) > max_points:
        u = u[:max_points]
    return u


def _groups_to_global(var_indices: List[int], g1, g2):
    """Convert local group indices to global variable indices."""
    var_set = set(var_indices)
    if all(int(i) in var_set for i in g1) and all(int(i) in var_set for i in g2):
        return [int(i) for i in g1], [int(i) for i in g2]
    return [var_indices[int(i)] for i in g1], [var_indices[int(i)] for i in g2]


def _compute_per_group_grad_mads(
    model_v: nn.Module,
    train_loader,
    var_indices: List[int],
    g1_global: List[int],
    g2_global: List[int],
    device,
    dtype,
    max_points: int = 2048,
    max_batches: int = 16,
) -> Tuple[float, float]:
    """Compute the sum of per-axis gradient MADs for each variable group.

    Returns (g1_mag, g2_mag) — aggregate gradient magnitudes for groups 1 and 2.
    Used to detect when one group contributes negligibly (variable pruning).
    """
    grads = []
    n = 0
    model_v.eval()
    with torch.no_grad():
        for bi, batch in enumerate(train_loader):
            if bi >= max_batches or n >= max_points:
                break
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device=device, dtype=dtype)
            gb = model_v.grad(xb)  # [B, 1, Nx]
            if gb.dim() == 3:
                gb = gb[:, 0, :]   # [B, Nx]
            grads.append(gb.detach().cpu())
            n += int(gb.shape[0])
    if not grads:
        return 1.0, 1.0
    G = torch.cat(grads, dim=0)[:max_points]  # [N, Nx]

    def _group_mag(g_global):
        total = 0.0
        for vi in g_global:
            col = G[:, vi]
            med = torch.median(col)
            mad = torch.median(torch.abs(col - med)).item()
            total += mad
        return total

    return _group_mag(g1_global), _group_mag(g2_global)


def _marginal_profile(
    X: torch.Tensor,
    v: torch.Tensor,
    target_cols: List[int],
    marginal_cols: List[int],
    n_bins: int = 20,
) -> torch.Tensor:
    """Estimate the marginal profile f(target) by binning target_cols and medianing v.

    For v ≈ f(target) * g(marginal), the profile f(target) is estimated by
    binning samples along the target_cols axes and computing median(v) in each
    bin.  Since g(marginal) varies within each target-bin but f(target) is
    roughly constant, the median cancels out the marginal variation, leaving
    a per-sample estimate that only varies along target_cols.

    Parameters
    ----------
    X : [N, D]  full input data (global columns)
    v : [N]     function values
    target_cols : global column indices for the target factor
    marginal_cols : global column indices to marginalize over (unused but kept for API symmetry)
    n_bins : number of bins per target axis

    Returns
    -------
    profile : [N]  per-sample marginal profile values
    """
    N = X.shape[0]
    if not target_cols:
        return v.clone()

    # Assign each sample to a target bin (product of per-axis bins)
    bin_ids = torch.zeros(N, dtype=torch.long)
    stride = 1
    for col in target_cols:
        xc = X[:, col].clone()
        lo, hi = xc.min().item(), xc.max().item()
        if hi - lo < 1e-30:
            continue
        # Quantile-based binning for robustness to outliers
        edges = torch.quantile(xc, torch.linspace(0, 1, n_bins + 1, device=xc.device, dtype=xc.dtype))
        bi = torch.bucketize(xc, edges[1:-1])  # [N], values in [0, n_bins-1]
        bin_ids = bin_ids + bi * stride
        stride *= n_bins

    # For each unique bin of target_cols, compute median(v).
    # Within each target-bin, f(target) ≈ const, so median(v) ≈ f(target) * median(g).
    # The global median(g) factor is common to all bins and cancels when we normalize.
    profile = v.clone()
    unique_bins = bin_ids.unique()
    for b in unique_bins:
        mask = (bin_ids == b)
        if mask.sum() > 0:
            profile[mask] = torch.median(v[mask])

    return profile


def _finite_median(x: torch.Tensor, default: float = 0.0) -> float:
    m = torch.isfinite(x)
    if not bool(m.any()):
        return float(default)
    val = float(torch.median(x[m]).item())
    return val if math.isfinite(val) else float(default)


def _finite_quantile(x: torch.Tensor, q: float, default: float = 0.0) -> float:
    m = torch.isfinite(x)
    if not bool(m.any()):
        return float(default)
    try:
        val = float(torch.quantile(x[m], float(q)).item())
    except Exception:
        val = float(default)
    return val if math.isfinite(val) else float(default)


def _adaptive_profile_bins(n_points: int, n_cols: int) -> int:
    if n_cols <= 0:
        return 1
    # Keep the marginal bins coarse for high-dimensional groups; otherwise
    # almost every sample gets its own bin and the initializer just memorizes.
    target_cells = max(4.0, float(max(1, n_points)) / 8.0)
    per_axis = int(round(target_cells ** (1.0 / float(n_cols))))
    return max(2, min(20, per_axis))


def _shared_gauge_transfer(
    X: torch.Tensor,
    F: torch.Tensor,
    G: torch.Tensor,
    g1_global: List[int],
    g2_global: List[int],
    *,
    shared_deg: int = 4,
    rel_tol: float = 1e-4,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Functional gauge transfer for an overlapping-square additive split.

    ``sqrt(F(g1) + G(g2))`` with ``s = g1 ∩ g2`` non-empty has a gauge freedom
    ``F -> F + h(s)``, ``G -> G - h(s)`` that preserves the represented function
    for ANY function ``h`` of the shared coordinates only. The marginal
    projection distributes that shared content arbitrarily, so a child that is
    really ``[power law] + h(s)`` is left non-monomial and its downstream
    power-law probe can fail despite an exact bulk power-law relation.

    This recovers a shared-only gauge that turns a child into a clean power law
    and transfers it. It fires only when the certificate residual is essentially
    zero (a real gauge, not a fit artifact), and only when the resulting
    components stay non-negative (required for the sqrt). Because ``F + G`` is
    invariant, it can never change the represented sqrt.
    """
    diag: Dict[str, float] = {"applied": 0.0}
    shared = sorted(set(int(a) for a in g1_global) & set(int(a) for a in g2_global))
    if not shared:
        return F, G, diag

    def _col(idx: int) -> torch.Tensor:
        return X[:, idx].to(dtype=F.dtype)

    def _shared_basis() -> Tuple[torch.Tensor, int]:
        # {prod_{j in shared} x_j^q : total degree q <= shared_deg}, incl. constant
        cols = [torch.ones_like(F)]
        for j in shared:
            xj = _col(j)
            acc = torch.ones_like(F)
            for _ in range(shared_deg):
                acc = acc * xj
                cols.append(acc)
        M = torch.stack(cols, dim=1)
        return M, M.shape[1]

    def _try_clean(child: torch.Tensor, child_vars: List[int]):
        ns = [v for v in child_vars if v not in shared]
        if not ns:
            return None
        pos = child > eps
        for v in child_vars:
            pos = pos & (_col(v) > eps)
        if int(pos.sum().item()) < max(200, child.numel() // 5):
            return None
        # Center guess for each non-shared var's power from a log-log OLS. The
        # additive gauge biases this (F = x1^4*(x0^2 + h) reads as x0^~1), so we
        # SCAN a small window of integer powers around the guess and keep the
        # combination whose [ns_mono * poly(s)] + poly(s) fit is cleanest.
        logF = torch.log(child[pos])
        design = torch.stack([torch.log(_col(v)[pos]) for v in child_vars]
                             + [torch.ones_like(logF)], dim=1)
        sol = torch.linalg.lstsq(design, logF.unsqueeze(1)).solution.squeeze(1)
        est = {v: float(sol[i].item()) for i, v in enumerate(child_vars) if v in ns}

        cand_powers = {}
        for v, e in est.items():
            base = int(round(e))
            cand = sorted({p for p in (base - 1, base, base + 1, base + 2)
                           if 1 <= p <= shared_deg + 2})
            cand_powers[v] = cand or [1]
        combos = list(itertools.product(*[cand_powers[v] for v in ns]))
        if len(combos) > 64:
            combos = combos[:64]

        Sb, ns_cols = _shared_basis()
        best = None
        for combo in combos:
            ns_mono = torch.ones_like(child)
            for v, p in zip(ns, combo):
                ns_mono = ns_mono * _col(v) ** p
            A = torch.cat([Sb * ns_mono.unsqueeze(1), Sb], dim=1)
            coef = torch.linalg.lstsq(A, child.unsqueeze(1)).solution.squeeze(1)
            rel = float((child - A @ coef).norm() / child.norm().clamp_min(eps))
            if best is None or rel < best[0]:
                gauge = Sb @ coef[ns_cols:]     # the pure-shared-coordinate part
                best = (rel, gauge, dict(zip(ns, combo)))
        if best is None or best[0] > rel_tol:
            return None
        return best[1], best[0], best[2]

    # Try to clean F (move its shared gauge into G), else clean G.
    for child, cvars, sign in ((F, g1_global, +1.0), (G, g2_global, -1.0)):
        res = _try_clean(child, [int(v) for v in cvars])
        if res is None:
            continue
        gauge, rel, ns_pow = res
        if sign > 0:
            F2, G2 = F - gauge, G + gauge
        else:
            F2, G2 = F + gauge, G - gauge
        # non-negativity guard (square-space components feed a sqrt)
        neg = float((torch.relu(-F2).mean() + torch.relu(-G2).mean()).item())
        base_neg = float((torch.relu(-F).mean() + torch.relu(-G).mean()).item())
        if neg > base_neg + 1e-9:
            continue
        diag = {"applied": 1.0, "cleaned": (1.0 if sign > 0 else 2.0),
                "cert_rel": float(rel), "n_shared": float(len(shared))}
        return F2, G2, diag
    return F, G, diag


def _square_additive_leaf_targets(
    X: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    g1_global: List[int],
    g2_global: List[int],
    *,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Build NN-leaf targets for ``u = sqrt(F(g1) + G(g2))``.

    The separability detector works in ``v = u^2`` space.  The visible
    candidate, however, is ``sqrt(NN(g1)^2 + NN(g2)^2)``.  Therefore the child
    leaves must be initialized to square-roots of non-negative square-space
    components, not to the square-space components themselves.
    """
    v_pos = torch.where(torch.isfinite(v), v, torch.zeros_like(v)).clamp_min(0.0)
    u_abs = torch.where(torch.isfinite(u), u.abs(), torch.zeros_like(u)).clamp_min(0.0)
    n_points = int(v_pos.numel())
    v_med = _finite_median(v_pos, 1.0)
    if v_med <= eps:
        v_med = max(float(torch.mean(v_pos).item()) if v_pos.numel() else 1.0, 1.0)

    bins1 = _adaptive_profile_bins(n_points, len(g1_global))
    bins2 = _adaptive_profile_bins(n_points, len(g2_global))
    mode = "projected"

    try:
        F = _marginal_profile(X, v_pos, g1_global, g2_global, n_bins=bins1)
        G = v_pos - F
        for _ in range(3):
            G = _marginal_profile(X, v_pos - F, g2_global, g1_global, n_bins=bins2)
            F = _marginal_profile(X, v_pos - G, g1_global, g2_global, n_bins=bins1)
        if not torch.isfinite(F).all() or not torch.isfinite(G).all():
            raise ValueError("non-finite projected component")
    except Exception:
        F = 0.5 * v_pos
        G = 0.5 * v_pos
        mode = "balanced"

    # Only a constant part of the residual is distributed here; per-sample
    # residual injection would leak all-variable structure into both targets.
    resid_med = _finite_median(v_pos - (F + G), 0.0)
    F = F + 0.5 * resid_med
    G = G + 0.5 * resid_med

    # The additive decomposition has gauge freedom.  Choose a constant gauge
    # that reduces negative square-space components before clamping.
    f_lo = _finite_quantile(F, 0.05, 0.0)
    g_lo = _finite_quantile(G, 0.05, 0.0)
    gauge_candidates = [0.0, -f_lo, g_lo, 0.5 * (-f_lo + g_lo)]
    best_c = 0.0
    best_score = float("inf")
    for c in gauge_candidates:
        if not math.isfinite(float(c)):
            continue
        Fc = F + float(c)
        Gc = G - float(c)
        score = float(torch.relu(-Fc).mean().item() + torch.relu(-Gc).mean().item())
        if score < best_score:
            best_score = score
            best_c = float(c)
    F = F + best_c
    G = G - best_c

    # If the children share coordinates, recover an s-only gauge that makes one
    # child a clean power law and transfer it. This preserves F+G and fires only
    # on a near-zero certificate residual and non-negative result.
    gauge_diag = {"applied": 0.0}
    try:
        F, G, gauge_diag = _shared_gauge_transfer(X, F, G, g1_global, g2_global, eps=eps)
    except Exception:
        gauge_diag = {"applied": 0.0}

    F = torch.where(torch.isfinite(F), F, torch.zeros_like(F)).clamp_min(0.0)
    G = torch.where(torch.isfinite(G), G, torch.zeros_like(G)).clamp_min(0.0)

    comp_med = _finite_median(F + G, v_med)
    if comp_med > eps:
        scale = v_med / comp_med
        if math.isfinite(scale) and scale > 0:
            F = F * scale
            G = G * scale

    L_target = torch.sqrt(F.clamp_min(0.0))
    R_target = torch.sqrt(G.clamp_min(0.0))
    pred_scale = torch.sqrt((L_target * L_target + R_target * R_target).clamp_min(0.0))

    # If the projected decomposition is scale-pathological, keep the candidate
    # in the right local optimum with a balanced sqrt split and let the real LM fit win
    # or reject it.
    u_med = _finite_median(u_abs, math.sqrt(max(v_med, eps)))
    pred_med = _finite_median(pred_scale, u_med)
    ratio = pred_med / max(u_med, eps)
    if (not math.isfinite(ratio)) or ratio < 0.1 or ratio > 10.0:
        half = 0.5 * v_pos
        L_target = torch.sqrt(half.clamp_min(0.0))
        R_target = torch.sqrt(half.clamp_min(0.0))
        pred_scale = torch.sqrt((L_target * L_target + R_target * R_target).clamp_min(0.0))
        pred_med = _finite_median(pred_scale, u_med)
        ratio = pred_med / max(u_med, eps)
        mode = "balanced-scale"

    diag = {
        "mode": mode,
        "bins1": float(bins1),
        "bins2": float(bins2),
        "u_median": float(u_med),
        "init_pred_median": float(pred_med),
        "median_ratio": float(ratio),
        "gauge": float(best_c),
        "shared_gauge_applied": float(gauge_diag.get("applied", 0.0)),
        "shared_gauge_cert_rel": float(gauge_diag.get("cert_rel", float("nan"))),
    }
    return L_target, R_target, diag


def _make_transform_sep_init_fn(
    *,
    model_u: nn.Module,
    t: str,
    op,
    var_indices,
    g1_global,
    g2_global,
    parent_tag: str,
    train_loader,
    device,
    dtype,
):
    """Build a custom_init_fn that quick-fits new NN leaves after a log/sqrt/square separability split.

    Decomposes T(u) into per-group marginal targets, then fits each child leaf
    via Adam so the new leaves start in a reasonable local optimum.
    """
    # Pre-sample X and u on construction (outside the init_fn closure)
    xs, us = [], []
    model_u.eval()
    n = 0
    with torch.no_grad():
        for bi, batch in enumerate(train_loader):
            if bi >= 16 or n >= 2048:
                break
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device=device, dtype=dtype)
            yb = model_u(xb)
            if yb.dim() == 1:
                yb = yb.view(-1, 1)
            xs.append(xb.detach().cpu())
            us.append(yb.detach().cpu().view(-1))
            n += int(yb.numel())

    if not xs:
        return None
    X_all = torch.cat(xs, dim=0)[:2048]
    u_all = torch.cat(us, dim=0)[:2048]

    # Apply transform to get v = T(u)
    eps = 1e-12
    if t == "log":
        v_all = torch.log(u_all.clamp(min=eps))
    elif t == "sqrt":
        v_all = torch.sqrt(u_all.clamp(min=0.0))
    elif t == "square":
        v_all = u_all * u_all
    elif t == "arcsin":
        v_all = torch.arcsin(u_all.clamp(-1.0 + eps, 1.0 - eps))
    else:
        v_all = u_all.clone()

    # Decompose v into per-group targets
    X_cpu = X_all

    is_add = (op == torch.add)
    v_med = float(v_all.median().item())

    square_add_diag = None
    if is_add and t == "square":
        v_L_target, v_R_target, square_add_diag = _square_additive_leaf_targets(
            X_cpu, u_all, v_all, g1_global, g2_global, eps=eps
        )
    elif is_add:
        # v ≈ f(g1) + g(g2), initialize f ≈ v_med, g ≈ v - v_med ≈ 0
        v_L_target = torch.full_like(v_all, v_med)
        v_R_target = v_all - v_med
    else:
        # v ≈ f(g1) * g(g2) — use marginal profiles for better initialization.
        # f(g1) ∝ median_over_g2(v), g(g2) ∝ median_over_g1(v)
        v_L_target = _marginal_profile(X_cpu, v_all, g1_global, g2_global)
        v_R_target = _marginal_profile(X_cpu, v_all, g2_global, g1_global)
        # Guard against degenerate marginals
        L_med = v_L_target.median().item()
        R_med = v_R_target.median().item()
        if abs(L_med) < eps or abs(R_med) < eps:
            # Fall back to simple constant / rescaled init
            if abs(v_med) < eps:
                v_med = 1.0
            v_L_target = torch.full_like(v_all, v_med)
            v_R_target = v_all / v_med
        else:
            # Normalize so product ≈ v (geometric mean correction)
            scale = (abs(v_med) / (abs(L_med * R_med) + 1e-30)) ** 0.5
            v_L_target = v_L_target * scale
            v_R_target = v_R_target * scale

    tag_L = f"{parent_tag}_L"
    tag_R = f"{parent_tag}_R"

    def _init_fn(root_new, model_new):
        try:
            atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
        except Exception as e:
            print(f"[transform_init] build_atom_to_leaf_map failed: {e}")
            return

        # Build tag -> leaf lookup
        tag_to_leaf = {}
        for a in _collect_all_atoms(root_new):
            if isinstance(a, AtomNode) and getattr(a, "tag", None) is not None:
                leaf_mod = atom_to_leaf_new.get(id(a))
                if leaf_mod is not None:
                    tag_to_leaf[str(a.tag)] = leaf_mod

        def _adam_fit(leaf_mod, x_data, y_data, steps=80, label="?"):
            """Quick-fit leaf to target via Adam."""
            try:
                dev_dt = None
                for p in leaf_mod.parameters():
                    if isinstance(p, torch.Tensor):
                        dev_dt = (p.device, p.dtype)
                        break
                if dev_dt is None:
                    dev_dt = (device, dtype)
                dev, dt = dev_dt
                x = x_data.to(dev, dt)
                y = y_data.to(dev, dt).view(-1)
                n_use = min(x.shape[0], 512)
                x, y = x[:n_use], y[:n_use]

                leaf_mod.train()
                opt = torch.optim.Adam(leaf_mod.parameters(), lr=1e-2)
                initial_loss = None
                final_loss = None
                last_step = 0
                for step in range(steps):
                    opt.zero_grad(set_to_none=True)
                    pred = leaf_mod(x)
                    if pred.dim() == 2:
                        pred = pred[:, 0]
                    pred = pred.view(-1)
                    loss = (pred - y).pow(2).mean()
                    if not torch.isfinite(loss):
                        break
                    if initial_loss is None:
                        initial_loss = float(loss.item())
                    final_loss = float(loss.item())
                    last_step = step
                    loss.backward()
                    opt.step()
                    if float(loss.item()) < 1e-10:
                        break
                print(
                    f"[transform_init] _adam_fit({label}): "
                    f"initial_loss={initial_loss:.3e}, final_loss={final_loss:.3e} "
                    f"after {last_step + 1} steps"
                )
                for p in leaf_mod.parameters():
                    p.grad = None
                leaf_mod.eval()
            except Exception as e:
                print(f"[transform_init] _adam_fit({label}) exception: {e}")
                try:
                    leaf_mod.eval()
                except Exception:
                    pass

        leaf_L = tag_to_leaf.get(tag_L)
        leaf_R = tag_to_leaf.get(tag_R)

        # Fallback: if tag lookup failed, try matching by var_idxs
        if leaf_L is None:
            for a in _collect_all_atoms(root_new):
                if isinstance(a, AtomNode) and a.kind == "nn":
                    if set(int(v) for v in a.var_idxs) == set(int(v) for v in g1_global):
                        leaf_L = atom_to_leaf_new.get(id(a))
                        if leaf_L is not None:
                            print(f"[transform_init] Fallback match for L by var_idxs={g1_global}")
                            break
        if leaf_R is None:
            for a in _collect_all_atoms(root_new):
                if isinstance(a, AtomNode) and a.kind == "nn":
                    if set(int(v) for v in a.var_idxs) == set(int(v) for v in g2_global):
                        leaf_R = atom_to_leaf_new.get(id(a))
                        if leaf_R is not None:
                            print(f"[transform_init] Fallback match for R by var_idxs={g2_global}")
                            break

        if leaf_L is None:
            print(f"[transform_init] WARNING: leaf_L not found (tag={tag_L}, available={list(tag_to_leaf.keys())})")
        if leaf_R is None:
            print(f"[transform_init] WARNING: leaf_R not found (tag={tag_R}, available={list(tag_to_leaf.keys())})")

        if square_add_diag is not None:
            print(
                "[transform_init] square-additive sqrt targets: "
                f"mode={square_add_diag['mode']}, "
                f"bins=({int(square_add_diag['bins1'])},{int(square_add_diag['bins2'])}), "
                f"median_ratio={square_add_diag['median_ratio']:.3e}, "
                f"u_med={square_add_diag['u_median']:.3e}, "
                f"init_med={square_add_diag['init_pred_median']:.3e}"
            )

        if leaf_L is not None:
            _adam_fit(leaf_L, X_cpu[:, g1_global], v_L_target, label=f"L({g1_global})")
        if leaf_R is not None:
            _adam_fit(leaf_R, X_cpu[:, g2_global], v_R_target, label=f"R({g2_global})")

    _init_fn._after_analytic_init = True
    _init_fn._candidate_seed_key = (
        f"transform-separability:{t}:{getattr(op, '__name__', str(op))}:"
        f"{tuple(int(v) for v in var_indices)}:"
        f"{tuple(int(v) for v in g1_global)}:{tuple(int(v) for v in g2_global)}"
    )
    _init_fn._candidate_max_starts = 3
    _init_fn._candidate_retry_nonfinite = True
    return _init_fn


def _build_subtree_separability_outer_transform_candidates(
    *,
    root: Node,
    u_node: AtomNode,
    model: nn.Module,
    reuse: Dict[str, nn.Module],
    train_loader,
    device,
    dtype,
    transforms: Tuple[str, ...] = ("log", "sqrt", "square", "arcsin"),
    domain_ok_frac_min: float = 0.90,
    eps: float = 1e-12,
    very_verbose: bool = False,
):
    """
    Try separability on v=T(u_node) using teacher derivatives; if separable, rewrite u_node via T^{-1}.
    Returns: list of (label, cand_root, init_fn, meta)
    """
    # Import locally to avoid circular dependency
    try:
        from ..subtree_separability_helpers import run_subtree_separability
    except Exception:
        run_subtree_separability = None

    if run_subtree_separability is None:
        return []
    if not isinstance(u_node, AtomNode) or str(u_node.kind).lower() != "nn":
        return []
    if len(u_node.var_idxs) < 2:
        return []

    # Build teacher model_u for this leaf (global-x derivatives).
    atom_to_leaf = build_atom_to_leaf_map(root, model)
    model_u = _SubtreeModel(root=u_node, atom_to_leaf=atom_to_leaf)

    # Sample u(x) once for domain screening.
    u_samp = _sample_u_values(model_u, train_loader, device=device, dtype=dtype, max_points=2048)

    var_indices = [int(j) for j in u_node.var_idxs]

    # Import _infer_nn_hyperparams_from_root from subtree_utils
    from .subtree_utils import _infer_nn_hyperparams_from_root

    num_segments, dual_layer = _infer_nn_hyperparams_from_root(root)
    parent_tag = u_node.tag

    out = []
    for t in transforms:
        t = str(t)
        if _domain_ok_frac(u_samp, t, eps=eps) < domain_ok_frac_min:
            continue

        # For "square" transform, also check sign consistency (u = sqrt(v) requires u >= 0)
        if t == "square" and _sign_consistent_frac(u_samp, eps=eps) < domain_ok_frac_min:
            continue

        model_v = _OuterTransformedSubtreeModel(model_u, t, eps=eps)
        # For "square" transform, allow partial separability (overlapping groups)
        # This helps with expressions like sqrt(f(x0,x1) + g(x1,x2)) where x1 is shared
        allow_partial = (t == "square")
        res = run_subtree_separability(
            model_u=model_v,
            datagen=train_loader,
            var_indices=var_indices,
            device=device,
            dtype=dtype,
            allow_partial=allow_partial,
            very_verbose=very_verbose,
        )
        if res is None:
            continue
        op, g1, g2 = res
        if not g1 or not g2:
            continue

        g1_global, g2_global = _groups_to_global(var_indices, g1, g2)

        # --- Variable pruning for square+additive case ---
        # If u² = f(g1) + g(g2) but one group has negligible gradients,
        # the NN barely depends on those variables.  Drop them and propose
        # a reduced-arity atom instead of sqrt(a² + b²).
        if t == "square" and op == torch.add:
            g1_mag, g2_mag = _compute_per_group_grad_mads(
                model_v, train_loader, var_indices,
                g1_global, g2_global, device, dtype,
            )
            ratio = min(g1_mag, g2_mag) / max(g1_mag, g2_mag, 1e-30)
            print(
                f"[Stage B]  sq+add grad-MAD ratio={ratio:.4f}  "
                f"g1({g1_global})={g1_mag:.3e}  g2({g2_global})={g2_mag:.3e}"
            )
            if ratio < 0.01:
                # One group contributes <1% — drop it
                keep = g1_global if g1_mag > g2_mag else g2_global
                drop = g2_global if g1_mag > g2_mag else g1_global
                keep_inputs = _select_inputs_for_var_group(u_node, keep)
                new_atom = AtomNode("nn", tuple(keep),
                                    kwargs=u_node.kwargs.copy(), tag=None,
                                    inputs=keep_inputs)
                cand_root = replace_atom_in_ast(root, u_node, new_atom)
                logmsg = (
                    f"[Stage B]  Variable pruning: dropped vars {drop} "
                    f"(ratio={ratio:.4f}), keeping nn({keep})"
                )
                out.append(("nn_variable_prune", cand_root, None,
                            {"log": logmsg, "outer_transform": t, "pruned_vars": drop}))
                # Still also emit the normal sqrt(a²+b²) candidate below
                # so the engine can compare both

        # v_split is an AST in terms of new NN leaves over (g1_global, g2_global)
        v_split = separability_proposal_to_ast(
            op,
            g1_global,
            g2_global,
            num_segments=num_segments,
            dual_layer=dual_layer,
            parent_tag=parent_tag,
            parent_atom=u_node,
        )

        # Invert the outer transform to get a rewrite for u
        if t == "log":
            # log(u)=v => u=exp(v). This inversion is only useful for *additive*
            # splits in log-space, since exp(a+b)=exp(a)*exp(b). A multiplicative
            # split v=f(g1)*g(g2) would imply u=exp(f*g), which is typically
            # ill-conditioned and not a meaningful separability for SR.
            if op != torch.add:
                continue
            new_sub = ExpNode(v_split)
            label = "nn_leaf_separability_log"
            logmsg = f"[Stage B]  NN-leaf separability split under log() vars={tuple(var_indices)}"
        elif t == "sqrt":
            # sqrt(u)=v => u = v^2
            new_sub = PowNode(v_split, 2.0)
            label = "nn_leaf_separability_sqrt"
            logmsg = f"[Stage B]  NN-leaf separability split under sqrt() vars={tuple(var_indices)}"
        elif t == "square":
            # square(u)=v => u = sqrt(v). For additive splits, use sqrt(NN_a² + NN_b²)
            # to ensure non-negativity inside the sqrt.
            if op == torch.add and isinstance(v_split, AddNode):
                # sqrt(NN_a² + NN_b²) is always non-negative
                left_sq = PowNode(v_split.left, 2.0)
                right_sq = PowNode(v_split.right, 2.0)
                v_squared_sum = AddNode(left_sq, right_sq)
                new_sub = PowNode(v_squared_sum, 0.5)
            else:
                # Multiplicative or other cases: use sqrt(v) directly
                new_sub = PowNode(v_split, 0.5)
            label = "nn_leaf_separability_sq"
            logmsg = (
                f"[Stage B]  NN-leaf separability split under square() vars={tuple(var_indices)}"
            )
        elif t == "recip":
            # 1/u = v => u = 1/v
            new_sub = PowNode(v_split, -1.0)
            label = "nn_leaf_separability_recip"
            logmsg = f"[Stage B]  NN-leaf separability split under reciprocal() vars={tuple(var_indices)}"
        elif t == "arcsin":
            # arcsin(u) = v => u = sin(v)
            new_sub = SinNode(v_split)
            label = "nn_leaf_separability_arcsin"
            logmsg = f"[Stage B]  NN-leaf separability split under arcsin() vars={tuple(var_indices)}"
        else:
            continue

        cand_root = replace_atom_in_ast(root, u_node, new_sub)
        init_fn = _make_transform_sep_init_fn(
            model_u=model_u, t=t, op=op,
            var_indices=var_indices, g1_global=g1_global, g2_global=g2_global,
            parent_tag=parent_tag,
            train_loader=train_loader, device=device, dtype=dtype,
        )
        out.append((label, cand_root, init_fn, {"log": logmsg, "outer_transform": t}))

    return out


def _build_nn_leaf_local_outer_transform_candidates(
    *,
    root: Node,
    nn_atom: AtomNode,
    model: nn.Module,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    transforms: Tuple[str, ...] = ("identity", "recip", "log", "sqrt", "square", "arcsin"),
    domain_ok_frac_min: float = 0.90,
    eps: float = 1e-12,
    very_verbose: bool = False,
):
    """Try Stage-A separability on v=T(u_leaf); invert T to rewrite u_leaf."""
    # Import locally to avoid circular dependency
    try:
        from ..subtree_separability_helpers import run_subtree_separability
    except Exception:
        run_subtree_separability = None

    if run_subtree_separability is None:
        return []
    if not isinstance(nn_atom, AtomNode) or str(nn_atom.kind).lower() != "nn":
        return []
    if len(nn_atom.var_idxs) < 2:
        return []

    var_indices = [int(j) for j in nn_atom.var_idxs]

    try:
        atom_to_leaf = build_atom_to_leaf_map(root, model)
    except Exception:
        return []

    model_u = _SubtreeModel(root=nn_atom, atom_to_leaf=atom_to_leaf)
    u_samp = _sample_subtree_values(
        model_u, train_loader, device=device, dtype=dtype, max_points=2048
    )

    # Import _infer_nn_hyperparams_from_root from subtree_utils
    from .subtree_utils import _infer_nn_hyperparams_from_root

    num_segments, dual_layer = _infer_nn_hyperparams_from_root(root)
    parent_tag = nn_atom.tag

    out = []
    for t in transforms:
        t = str(t)
        if t != "identity":
            if u_samp is None:
                continue
            ok_frac = _domain_ok_frac_for_transform(u_samp, t, eps=eps)
            if ok_frac < domain_ok_frac_min:
                continue
            # For "square" transform, also check sign consistency (u = sqrt(v) requires u >= 0)
            if t == "square" and _sign_consistent_frac(u_samp, eps=eps) < domain_ok_frac_min:
                continue

        model_v = model_u if t == "identity" else _OuterTransformedSubtreeModel(model_u, t, eps=eps)
        # For "square" transform, allow partial separability (overlapping groups)
        allow_partial = (t == "square")
        res = run_subtree_separability(
            model_u=model_v,
            datagen=train_loader,
            var_indices=var_indices,
            device=device,
            dtype=dtype,
            allow_partial=allow_partial,
            very_verbose=very_verbose,
        )
        if res is None:
            continue
        op, group1_local, group2_local = res
        if not group1_local or not group2_local:
            continue
        group1_global = [var_indices[i] for i in group1_local]
        group2_global = [var_indices[i] for i in group2_local]

        # --- Variable pruning for square+additive case ---
        if t == "square" and op == torch.add:
            g1_mag, g2_mag = _compute_per_group_grad_mads(
                model_v, train_loader, var_indices,
                group1_global, group2_global, device, dtype,
            )
            ratio = min(g1_mag, g2_mag) / max(g1_mag, g2_mag, 1e-30)
            print(
                f"[Stage B]  sq+add grad-MAD ratio={ratio:.4f}  "
                f"g1({group1_global})={g1_mag:.3e}  g2({group2_global})={g2_mag:.3e}"
            )
            if ratio < 0.01:
                keep = group1_global if g1_mag > g2_mag else group2_global
                drop = group2_global if g1_mag > g2_mag else group1_global
                keep_inputs = _select_inputs_for_var_group(nn_atom, keep)
                new_atom = AtomNode("nn", tuple(keep),
                                    kwargs=nn_atom.kwargs.copy(), tag=parent_tag,
                                    inputs=keep_inputs)
                cand_root = replace_atom_in_ast(root, nn_atom, new_atom)
                logmsg = (
                    f"[Stage B]  Variable pruning: dropped vars {drop} "
                    f"(ratio={ratio:.4f}), keeping nn({keep})"
                )
                out.append(("nn_variable_prune", cand_root, None,
                            {"log": logmsg, "outer_transform": t, "pruned_vars": drop}))

        v_split = separability_proposal_to_ast(
            op,
            group1_global,
            group2_global,
            num_segments=num_segments,
            dual_layer=dual_layer,
            parent_tag=parent_tag,
            parent_atom=nn_atom,
        )

        if t == "identity":
            new_sub = v_split
            label = "nn_leaf_separability"
            logmsg = f"[Stage B]  NN-leaf separability split (identity) vars={tuple(var_indices)}"

        elif t == "log":
            # log(u)=v => u=exp(v). This is only useful for additive splits, since
            # exp(a+b)=exp(a)*exp(b). A multiplicative split in log-space implies
            # u=exp(f*g), which is usually ill-conditioned and not a true
            # separability in the original u-space.
            if op != torch.add:
                continue
            if op == torch.add and isinstance(v_split, AddNode):
                new_sub = MulNode(ExpNode(v_split.left), ExpNode(v_split.right))
            else:
                new_sub = ExpNode(v_split)
            label = "nn_leaf_separability_log"
            logmsg = f"[Stage B]  NN-leaf separability split under log() vars={tuple(var_indices)}"

        elif t == "sqrt":
            # sqrt(u)=v => u=v^2
            new_sub = PowNode(v_split, 2.0)
            label = "nn_leaf_separability_sqrt"
            logmsg = f"[Stage B]  NN-leaf separability split under sqrt() vars={tuple(var_indices)}"

        elif t == "square":
            # square(u)=v => u=sqrt(v). For additive splits, use sqrt(NN_a² + NN_b²)
            # to ensure non-negativity inside the sqrt.
            if op == torch.add and isinstance(v_split, AddNode):
                # sqrt(NN_a² + NN_b²) is always non-negative
                left_sq = PowNode(v_split.left, 2.0)
                right_sq = PowNode(v_split.right, 2.0)
                v_squared_sum = AddNode(left_sq, right_sq)
                new_sub = PowNode(v_squared_sum, 0.5)
            else:
                # Multiplicative or other cases: use sqrt(v) directly
                new_sub = PowNode(v_split, 0.5)
            label = "nn_leaf_separability_sq"
            logmsg = f"[Stage B]  NN-leaf separability split under square() vars={tuple(var_indices)}"

        elif t == "arcsin":
            # arcsin(u)=v => u=sin(v)
            new_sub = SinNode(v_split)
            label = "nn_leaf_separability_arcsin"
            logmsg = f"[Stage B]  NN-leaf separability split under arcsin() vars={tuple(var_indices)}"

        else:
            continue

        cand_root = replace_atom_in_ast(root, nn_atom, new_sub)
        meta = {"log": logmsg, "outer_transform": t}
        if t != "identity":
            init_fn = _make_transform_sep_init_fn(
                model_u=model_u, t=t, op=op,
                var_indices=var_indices, g1_global=group1_global, g2_global=group2_global,
                parent_tag=parent_tag,
                train_loader=train_loader, device=device, dtype=dtype,
            )
        else:
            init_fn = None
        out.append((label, cand_root, init_fn, meta))

    return out
