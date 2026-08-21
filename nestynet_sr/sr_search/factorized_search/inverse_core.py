# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Inverse steering target construction and local inverse scoring helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .expr_ast import (
    UNARY_OPS,
    collect_paths,
    eval_node,
    node_str,
)
from .expr_mapping import eval_mapping, fit_best


@dataclass(frozen=True)
class InverseStep:
    parent_path: tuple[int, ...]
    op: str
    child_slot: int
    valid_fraction: float
    confidence: float
    note: str = ""


@dataclass(frozen=True)
class InverseTarget:
    path: tuple[int, ...]
    target: torch.Tensor
    valid_mask: torch.Tensor
    point_weight: torch.Tensor
    confidence: float
    mapping_inverted: bool
    mapping_kind: str
    steps: tuple[InverseStep, ...]
    branch_id: str = ""


def _ensure_col(v: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(v):
        raise TypeError(f"expected torch.Tensor, got {type(v).__name__}")
    if v.ndim == 1:
        return v.unsqueeze(-1)
    if v.ndim == 2 and v.shape[1] == 1:
        return v
    raise ValueError(f"expected [N] or [N,1] tensor, got shape={tuple(v.shape)}")


def _bool_col(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(-1)
    if mask.ndim != 2 or mask.shape[1] != 1:
        raise ValueError(f"expected mask shape [N,1], got {tuple(mask.shape)}")
    return mask.to(dtype=torch.bool)


def _mask_fraction(mask: torch.Tensor) -> float:
    m = _bool_col(mask)
    if m.numel() <= 0:
        return 0.0
    return float(m.float().mean().item())


def _finite_mask(*xs: torch.Tensor) -> torch.Tensor:
    if not xs:
        raise ValueError("_finite_mask requires at least one tensor")
    mask = torch.ones_like(_ensure_col(xs[0]), dtype=torch.bool)
    for x in xs:
        mask = mask & torch.isfinite(_ensure_col(x))
    return mask


def _conditioning_confidence_from_gain(
    gain: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    target_gain: float = 4.0,
    floor: float = 0.05,
) -> float:
    g = _ensure_col(gain).abs()
    valid = _finite_mask(g)
    if mask is not None:
        valid = valid & _bool_col(mask)
    n_valid = int(valid.sum().item())
    if n_valid <= 0:
        return 0.0
    try:
        tg = max(1.0e-12, float(target_gain))
    except Exception:
        tg = 4.0
    g = torch.clamp(g, min=0.0, max=1.0e12)
    excess = torch.clamp(g - 1.0, min=0.0)
    score = 1.0 / (1.0 + excess / tg)
    conf = float(score[valid].mean().item())
    if not math.isfinite(conf):
        return 0.0
    try:
        fl = float(floor)
    except Exception:
        fl = 0.05
    fl = min(1.0, max(0.0, fl))
    return float(min(1.0, max(fl, conf)))


def _combine_inverse_confidence(
    base_conf: float,
    gain: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    confidence_mode: str,
    confidence_target_gain: float,
    confidence_floor: float,
) -> float:
    bc = float(base_conf)
    if not math.isfinite(bc):
        return 0.0
    bc = min(1.0, max(0.0, bc))
    mode = str(confidence_mode).strip().lower()
    if mode in ("heuristic", "fixed", "legacy"):
        return bc
    cc = _conditioning_confidence_from_gain(
        gain,
        mask=mask,
        target_gain=float(confidence_target_gain),
        floor=float(confidence_floor),
    )
    return float(min(1.0, max(0.0, bc * cc)))


def _conditioning_point_weight_from_gain(
    gain: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    confidence_mode: str,
    target_gain: float,
    floor: float,
) -> torch.Tensor:
    g = _ensure_col(gain).abs()
    valid = _finite_mask(g)
    if mask is not None:
        valid = valid & _bool_col(mask)
    mode = str(confidence_mode).strip().lower()
    if mode in ("heuristic", "fixed", "legacy"):
        base = torch.ones_like(g)
    else:
        try:
            tg = max(1.0e-12, float(target_gain))
        except Exception:
            tg = 4.0
        excess = torch.clamp(torch.clamp(g, min=0.0, max=1.0e12) - 1.0, min=0.0)
        base = 1.0 / (1.0 + excess / tg)
        try:
            fl = float(floor)
        except Exception:
            fl = 0.05
        fl = min(1.0, max(0.0, fl))
        base = torch.clamp(base, min=fl, max=1.0)
    return torch.where(valid, base, torch.zeros_like(base))


def _prepare_nonnegative_weights(
    w: torch.Tensor | None,
    ref: torch.Tensor,
) -> torch.Tensor:
    rr = _ensure_col(ref)
    if w is None:
        return torch.ones_like(rr)
    ww = _ensure_col(w).to(dtype=rr.dtype, device=rr.device)
    if int(ww.shape[0]) != int(rr.shape[0]):
        return torch.ones_like(rr)
    ww = torch.where(torch.isfinite(ww), ww, torch.zeros_like(ww))
    return torch.clamp(ww, min=0.0)


def _weighted_mse_cols(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    w: torch.Tensor | None = None,
) -> float | None:
    yt = _ensure_col(y_true)
    yp = _ensure_col(y_pred)
    ww = _prepare_nonnegative_weights(w, yt)
    m = _finite_mask(yt, yp, ww) & (ww > 0.0)
    if int(m.sum().item()) < 1:
        return None
    err2 = (yt[m] - yp[m]) ** 2
    ws = ww[m]
    den = float(ws.sum().item())
    if (not math.isfinite(den)) or den <= 1.0e-18:
        return None
    num = float((ws * err2).sum().item())
    if not math.isfinite(num):
        return None
    return float(num / den)


def _weighted_centered_mse(y: torch.Tensor, w: torch.Tensor | None = None) -> float:
    yy = _ensure_col(y)
    ww = _prepare_nonnegative_weights(w, yy)
    m = _finite_mask(yy, ww) & (ww > 0.0)
    if int(m.sum().item()) < 1:
        return float(((yy - yy.mean()) ** 2).mean().item())
    yv = yy[m]
    wv = ww[m]
    den = torch.clamp(wv.sum(), min=1.0e-18)
    mu = (wv * yv).sum() / den
    mse = ((wv * (yv - mu) ** 2).sum() / den).item()
    return float(mse) if math.isfinite(float(mse)) else float(((yy - yy.mean()) ** 2).mean().item())


def _masked_point_weight(
    point_weight_full: torch.Tensor | None,
    mask_full: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    m = _bool_col(mask_full).squeeze(-1)
    n = int(m.sum().item())
    if n <= 0:
        return torch.zeros((0, 1), dtype=dtype, device=device)
    if point_weight_full is None:
        return torch.ones((n, 1), dtype=dtype, device=device)
    pw = _ensure_col(point_weight_full)
    if int(pw.shape[0]) != int(mask_full.shape[0]):
        return torch.ones((n, 1), dtype=dtype, device=device)
    pw = pw.to(dtype=dtype, device=device)
    pw = torch.where(torch.isfinite(pw), pw, torch.zeros_like(pw))
    pw = torch.clamp(pw, min=0.0, max=1.0)
    return pw[m]


def _slice_by_mask(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = _bool_col(mask).squeeze(-1)
    return x[m], _ensure_col(y)[m]


def _collect_nodes_preorder(node: tuple, path: tuple[int, ...] = (), out: dict[tuple[int, ...], tuple] | None = None) -> dict[tuple[int, ...], tuple]:
    if out is None:
        out = {}
    out[path] = node
    op = str(node[0])
    if op in ("var", "const", "hparam"):
        return out
    if op in UNARY_OPS:
        _collect_nodes_preorder(node[1], path + (1,), out)
        return out
    _collect_nodes_preorder(node[1], path + (1,), out)
    _collect_nodes_preorder(node[2], path + (2,), out)
    return out


def _mapping_output_derivative(
    pred: torch.Tensor,
    mapping: dict[str, Any] | None,
    *,
    safe_eps: float = 1.0e-12,
) -> torch.Tensor:
    p = _ensure_col(pred)
    m = mapping if isinstance(mapping, dict) else {}
    kind = str(m.get("kind", "identity")).strip().lower()
    eps = float(max(1.0e-12, safe_eps))

    try:
        if kind in ("", "identity"):
            return torch.ones_like(p)
        if kind in ("affine", "mono", "monomial"):
            a = float(m.get("a", 1.0))
            return torch.full_like(p, float(a))
        if kind == "poly":
            coeffs = m.get("coeffs", None)
            if isinstance(coeffs, torch.Tensor):
                coeffs = coeffs.detach().cpu().tolist()
            if not isinstance(coeffs, (list, tuple)) or len(coeffs) <= 1:
                return torch.ones_like(p)
            mu = float(m.get("mu", 0.0))
            std = float(m.get("std", 1.0))
            if abs(std) <= eps:
                return torch.zeros_like(p)
            z = (p - mu) / std
            dz = torch.zeros_like(z)
            for k, ck in enumerate(coeffs[1:], start=1):
                try:
                    c = float(ck)
                except Exception:
                    c = 0.0
                if k == 1:
                    term = torch.ones_like(z)
                else:
                    term = z ** (k - 1)
                dz = dz + float(k) * c * term
            d = dz / std
            return torch.where(torch.isfinite(d), d, torch.zeros_like(d))
        if kind == "power":
            b = float(m.get("b", 0.0))
            sf = float(m.get("sgn_f", m.get("sf", 1.0)))
            y = eval_mapping(p, m)
            den = torch.clamp(p.abs(), min=eps) * torch.sign(p)
            base_mask = (sf * p > eps) & torch.isfinite(y)
            d = (float(b) * y) / den
            d = torch.where(base_mask, d, torch.zeros_like(d))
            return torch.where(torch.isfinite(d), d, torch.zeros_like(d))
        if kind == "exp":
            a = float(m.get("a", 1.0))
            b = float(m.get("b", 1.0))
            mu = float(m.get("mu", 0.0))
            std = float(m.get("std", 1.0))
            if abs(std) <= eps:
                return torch.zeros_like(p)
            z = (p - mu) / std
            ez = torch.exp(torch.clamp(float(b) * z, min=-80.0, max=80.0))
            d = float(a) * float(b) * ez / std
            return torch.where(torch.isfinite(d), d, torch.zeros_like(d))
        if kind == "sine":
            A = float(m.get("A", 0.0))
            B = float(m.get("B", 0.0))
            omega = float(m.get("omega", 1.0))
            mu = float(m.get("mu", 0.0))
            std = float(m.get("std", 1.0))
            if abs(std) <= eps:
                return torch.zeros_like(p)
            z = (p - mu) / std
            wz = float(omega) * z
            d = (float(omega) / std) * (float(A) * torch.cos(wz) - float(B) * torch.sin(wz))
            return torch.where(torch.isfinite(d), d, torch.zeros_like(d))
        if kind == "pade":
            numer = m.get("numer", None)
            denom = m.get("denom", None)
            if isinstance(numer, torch.Tensor):
                numer = numer.detach().cpu().tolist()
            if isinstance(denom, torch.Tensor):
                denom = denom.detach().cpu().tolist()
            if not (isinstance(numer, (list, tuple)) and isinstance(denom, (list, tuple)) and len(numer) <= 2 and len(denom) <= 2):
                return torch.ones_like(p)
            n0 = float(numer[0]) if len(numer) >= 1 else 0.0
            n1 = float(numer[1]) if len(numer) >= 2 else 0.0
            d0 = float(denom[0]) if len(denom) >= 1 else 1.0
            d1 = float(denom[1]) if len(denom) >= 2 else 0.0
            mu = float(m.get("mu", 0.0))
            std = float(m.get("std", 1.0))
            if abs(std) <= eps:
                return torch.zeros_like(p)
            z = (p - mu) / std
            denz = d0 + d1 * z
            numc = (n1 * d0 - n0 * d1)
            d = (numc / torch.clamp(denz * denz, min=eps)) / std
            return torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    except Exception:
        pass

    # Numeric fallback for unknown/unstable mappings.
    try:
        h = 1.0e-4 * max(1.0, float(p.detach().abs().mean().item()))
    except Exception:
        h = 1.0e-4
    h = float(max(eps, min(1.0, h)))
    try:
        yp = eval_mapping(p + h, m)
        ym = eval_mapping(p - h, m)
        d = (yp - ym) / (2.0 * h)
        return torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    except Exception:
        return torch.ones_like(p)


@torch.no_grad()
def _compute_path_influences(
    candidate_ast: tuple,
    x: torch.Tensor,
    mapping: dict[str, Any] | None,
    *,
    safe_eps: float = 1.0e-12,
) -> tuple[dict[tuple[int, ...], torch.Tensor], dict[tuple[int, ...], torch.Tensor], torch.Tensor, torch.Tensor]:
    nodes = _collect_nodes_preorder(candidate_ast)
    outputs: dict[tuple[int, ...], torch.Tensor] = {}
    root_pred: torch.Tensor | None = None
    for p in sorted(nodes.keys(), key=len):
        try:
            out = _ensure_col(eval_node(nodes[p], x))
        except Exception:
            out = None
        if out is None:
            outputs[p] = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        else:
            outputs[p] = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        if p == ():
            root_pred = outputs[p]
    if root_pred is None:
        root_pred = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
    yhat = eval_mapping_total(root_pred, mapping, x)
    yhat = torch.where(torch.isfinite(yhat), yhat, torch.zeros_like(yhat))
    root_adj = _mapping_output_derivative(root_pred, mapping, safe_eps=safe_eps)
    root_adj = torch.where(torch.isfinite(root_adj), root_adj, torch.zeros_like(root_adj))
    adj = {p: torch.zeros_like(root_pred) for p in nodes.keys()}
    adj[()] = root_adj

    eps = float(max(1.0e-12, safe_eps))
    for p in sorted(nodes.keys(), key=len):
        node = nodes[p]
        g = adj.get(p, None)
        if g is None:
            continue
        g = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
        op = str(node[0])
        if op in ("var", "const", "hparam"):
            continue
        if op in UNARY_OPS:
            cp = p + (1,)
            u = outputs.get(cp, None)
            if u is None:
                continue
            if op == "neg":
                dg = -g
            elif op == "sin":
                dg = g * torch.cos(u)
            elif op == "cos":
                dg = g * (-torch.sin(u))
            elif op == "exp":
                dg = g * torch.exp(torch.clamp(u, min=-80.0, max=80.0))
            elif op == "log":
                ok = u.abs() > eps
                dg = g / torch.where(ok, u, torch.ones_like(u))
                dg = torch.where(ok, dg, torch.zeros_like(dg))
            elif op == "sqrt":
                ok = u > eps
                den = 2.0 * torch.sqrt(torch.clamp(u, min=eps))
                dg = g / den
                dg = torch.where(ok, dg, torch.zeros_like(dg))
            elif op == "sqr":
                dg = g * (2.0 * u)
            else:
                dg = torch.zeros_like(g)
            dg = torch.where(torch.isfinite(dg), dg, torch.zeros_like(dg))
            adj[cp] = adj.get(cp, torch.zeros_like(root_pred)) + dg
            continue

        a = outputs.get(p + (1,), None)
        b = outputs.get(p + (2,), None)
        if a is None or b is None:
            continue
        if op == "add":
            da = g
            db = g
        elif op == "sub":
            da = g
            db = -g
        elif op == "mul":
            da = g * b
            db = g * a
        elif op == "div":
            ok = b.abs() > eps
            da = g / torch.where(ok, b, torch.ones_like(b))
            db = -g * a / torch.where(ok, b * b, torch.ones_like(b))
            da = torch.where(ok, da, torch.zeros_like(da))
            db = torch.where(ok, db, torch.zeros_like(db))
        else:
            da = torch.zeros_like(g)
            db = torch.zeros_like(g)
        da = torch.where(torch.isfinite(da), da, torch.zeros_like(da))
        db = torch.where(torch.isfinite(db), db, torch.zeros_like(db))
        adj[p + (1,)] = adj.get(p + (1,), torch.zeros_like(root_pred)) + da
        adj[p + (2,)] = adj.get(p + (2,), torch.zeros_like(root_pred)) + db

    return adj, outputs, root_pred, yhat


def _weighted_inner_cols(
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor | None = None,
) -> float | None:
    aa = _ensure_col(a)
    bb = _ensure_col(b)
    ww = _prepare_nonnegative_weights(w, aa)
    m = _finite_mask(aa, bb, ww) & (ww > 0.0)
    if int(m.sum().item()) < 1:
        return None
    val = float((ww[m] * aa[m] * bb[m]).sum().item())
    if not math.isfinite(val):
        return None
    return val


def _effective_sample_size(w: torch.Tensor | None, ref: torch.Tensor) -> float:
    rr = _ensure_col(ref)
    ww = _prepare_nonnegative_weights(w, rr)
    m = _finite_mask(rr, ww) & (ww > 0.0)
    if int(m.sum().item()) <= 0:
        return 0.0
    wv = ww[m].squeeze(-1)
    s1 = float(wv.sum().item())
    s2 = float((wv * wv).sum().item())
    if (not math.isfinite(s1)) or (not math.isfinite(s2)) or s2 <= 1.0e-18:
        return 0.0
    return float((s1 * s1) / s2)


def _path_transport_scalar(
    residual: torch.Tensor,
    influence: torch.Tensor,
    node_output: torch.Tensor,
    *,
    w: torch.Tensor | None = None,
    safe_eps: float = 1.0e-12,
) -> float:
    r = _ensure_col(residual)
    g = _ensure_col(influence)
    u = _ensure_col(node_output)
    ww = _prepare_nonnegative_weights(w, r)
    m = _finite_mask(r, g, u, ww) & (ww > 0.0)
    if int(m.sum().item()) < 4:
        return 0.0
    rr = r[m]
    gg = g[m]
    uu = u[m]
    ww_m = ww[m]
    den_w = float(ww_m.sum().item())
    if (not math.isfinite(den_w)) or den_w <= 1.0e-18:
        return 0.0
    mu = float((ww_m * uu).sum().item() / den_w)
    du = uu - mu
    h = gg * du
    num = _weighted_inner_cols(rr, h, ww_m)
    den = _weighted_inner_cols(h, h, ww_m)
    if num is None or den is None or den <= float(max(1.0e-12, safe_eps)):
        return 0.0
    score = (num * num) / max(den, float(max(1.0e-12, safe_eps)))
    if not math.isfinite(score):
        return 0.0
    return float(max(0.0, score))


def _linearized_residual_gain(
    residual: torch.Tensor,
    influence: torch.Tensor,
    delta_u: torch.Tensor,
    *,
    w: torch.Tensor | None = None,
    safe_eps: float = 1.0e-12,
) -> float:
    r = _ensure_col(residual)
    g = _ensure_col(influence)
    du = _ensure_col(delta_u)
    h = g * du
    num = _weighted_inner_cols(r, h, w)
    den = _weighted_inner_cols(h, h, w)
    if num is None or den is None:
        return -float("inf")
    gain = 2.0 * num - den
    if not math.isfinite(gain):
        return -float("inf")
    return float(gain)


def _blend_inverse_backprop_target(
    t_inv: torch.Tensor,
    u_cur: torch.Tensor,
    residual: torch.Tensor,
    influence: torch.Tensor,
    w_inv: torch.Tensor | None,
    *,
    eta: float = 0.75,
    safe_eps: float = 1.0e-12,
    step_clip_q: float = 0.90,
    step_clip_mult: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend inverse pseudo-target with a local backprop target u + r/g.

    The backprop term is clipped robustly to avoid exploding updates when local
    influence is tiny.  We also downweight low-influence points in the returned
    pointwise weights.
    """
    t0 = _ensure_col(t_inv)
    u0 = _ensure_col(u_cur)
    r0 = _ensure_col(residual)
    g0 = _ensure_col(influence)

    base_w = _prepare_nonnegative_weights(w_inv, t0)
    base_w = torch.where(torch.isfinite(base_w), base_w, torch.zeros_like(base_w))
    base_w = torch.clamp(base_w, min=0.0, max=1.0)

    n = int(t0.shape[0])
    if int(u0.shape[0]) != n or int(r0.shape[0]) != n or int(g0.shape[0]) != n:
        return t0, base_w

    eps = float(max(1.0e-12, safe_eps))
    stable = _finite_mask(t0, u0, r0, g0, base_w) & (g0.abs() > eps)
    if int(stable.sum().item()) < 2:
        return t0, base_w

    signed_den = torch.where(g0 >= 0.0, torch.clamp(g0, min=eps), torch.clamp(g0, max=-eps))
    step = torch.where(stable, r0 / signed_den, torch.zeros_like(r0))
    step = torch.where(torch.isfinite(step), step, torch.zeros_like(step))

    try:
        q = float(step_clip_q)
    except Exception:
        q = 0.90
    q = min(0.99, max(0.50, q))
    try:
        mult = float(step_clip_mult)
    except Exception:
        mult = 3.0
    mult = max(1.0, mult)

    abs_step = step[stable].abs().squeeze(-1)
    if int(abs_step.numel()) >= 4:
        try:
            clip = float(torch.quantile(abs_step, q).item())
        except Exception:
            clip = float(abs_step.median().item())
    else:
        clip = float(abs_step.max().item()) if int(abs_step.numel()) > 0 else 0.0
    if (not math.isfinite(clip)) or clip <= 0.0:
        clip = 1.0
    clip = max(eps, clip * mult)
    step = torch.clamp(step, min=-clip, max=clip)

    t_bp = u0 + step
    t_bp = torch.where(torch.isfinite(t_bp), t_bp, t0)

    g_abs = torch.where(stable, g0.abs(), torch.zeros_like(g0))
    g_vals = g_abs[stable].squeeze(-1)
    if int(g_vals.numel()) >= 4:
        try:
            kappa = float(torch.quantile(g_vals, 0.50).item())
        except Exception:
            kappa = float(g_vals.median().item())
    else:
        kappa = float(g_vals.mean().item()) if int(g_vals.numel()) > 0 else 0.0
    if (not math.isfinite(kappa)) or kappa <= 0.0:
        kappa = eps
    w_bp = g_abs / (g_abs + float(kappa))
    w_bp = torch.where(stable, w_bp, torch.zeros_like(w_bp))
    w_bp = torch.clamp(w_bp, min=0.0, max=1.0)

    try:
        eta_f = float(eta)
    except Exception:
        eta_f = 0.75
    eta_f = min(1.0, max(0.0, eta_f))
    t_mix = eta_f * t0 + (1.0 - eta_f) * t_bp
    t_mix = torch.where(torch.isfinite(t_mix), t_mix, t0)

    w_mix = torch.sqrt(torch.clamp(base_w, min=0.0, max=1.0) * torch.clamp(w_bp, min=0.0, max=1.0))
    w_mix = torch.where(torch.isfinite(w_mix), w_mix, base_w)
    w_mix = torch.clamp(w_mix, min=0.0, max=1.0)
    return t_mix, w_mix


@torch.no_grad()
def _estimate_path_transport_scores(
    candidate_ast: tuple,
    mapping: dict[str, Any] | None,
    x: torch.Tensor,
    y_target: torch.Tensor,
    paths: Sequence[tuple[int, ...]],
    *,
    safe_eps: float = 1.0e-12,
) -> tuple[dict[tuple[int, ...], float], dict[tuple[int, ...], torch.Tensor], dict[tuple[int, ...], torch.Tensor], torch.Tensor]:
    adj, outputs, _pred, yhat = _compute_path_influences(
        candidate_ast,
        x,
        mapping,
        safe_eps=float(safe_eps),
    )
    r = _ensure_col(y_target) - _ensure_col(yhat)
    scores: dict[tuple[int, ...], float] = {}
    for p in paths:
        g = adj.get(tuple(p), None)
        u = outputs.get(tuple(p), None)
        if g is None or u is None:
            scores[tuple(p)] = 0.0
            continue
        scores[tuple(p)] = _path_transport_scalar(r, g, u, w=None, safe_eps=float(safe_eps))
    return scores, adj, outputs, r


def _eval_linear_head(head: dict[str, Any] | None, x: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(head, dict):
        return None
    terms = head.get("terms", None)
    coeffs = head.get("coeffs", None)
    if not isinstance(terms, (list, tuple)) or not isinstance(coeffs, (list, tuple)):
        return None
    if len(coeffs) != (len(terms) + 1):
        return None
    try:
        out = torch.full((x.shape[0], 1), float(coeffs[0]), dtype=x.dtype, device=x.device)
    except Exception:
        return None
    for c, term in zip(coeffs[1:], terms):
        try:
            v = eval_node(term, x)
        except Exception:
            return None
        if not torch.isfinite(v).all():
            return None
        out = out + float(c) * v
    return out


def eval_mapping_total(pred, mapping, x=None):
    y_hat = eval_mapping(pred, mapping)
    if isinstance(mapping, dict) and x is not None:
        head_pred = _eval_linear_head(mapping.get("_lin_head", None), x)
        if head_pred is not None and torch.isfinite(head_pred).all():
            y_hat = y_hat + head_pred
    return y_hat


def _invert_shifted_sinusoid(
    target: torch.Tensor,
    *,
    A: float,
    B: float,
    c: float,
    omega: float,
    ref: torch.Tensor | None,
    output_scale: float = 1.0,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, str]:
    y = _ensure_col(target)
    if not math.isfinite(float(omega)) or abs(float(omega)) < 1.0e-12:
        mask = _finite_mask(y)
        return torch.zeros_like(y), mask & False, 0.0, torch.zeros_like(y), "omega too small for sinusoid inversion"

    R = math.hypot(float(A), float(B))
    if not math.isfinite(R) or R < 1.0e-12:
        mask = _finite_mask(y)
        return torch.zeros_like(y), mask & False, 0.0, torch.zeros_like(y), "sinusoid amplitude too small for inversion"

    phi = math.atan2(float(B), float(A))
    u = (y - float(c)) / float(R)
    mask = _finite_mask(u) & (u.abs() <= 1.0 + 1.0e-7)
    u_clip = torch.clamp(u, -1.0, 1.0)
    theta = torch.asin(u_clip)

    period = 2.0 * math.pi / float(omega)
    z1 = (theta - phi) / float(omega)
    z2 = ((math.pi - theta) - phi) / float(omega)

    if ref is not None:
        z_ref = _ensure_col(ref)
        ref_mask = _finite_mask(z_ref)
        k1 = torch.round((z_ref - z1) / period)
        k2 = torch.round((z_ref - z2) / period)
        z1 = z1 + k1 * period
        z2 = z2 + k2 * period
        d1 = (z1 - z_ref).abs()
        d2 = (z2 - z_ref).abs()
        choose_1 = d1 <= d2
        choose_1 = torch.where(ref_mask, choose_1, torch.ones_like(choose_1, dtype=torch.bool))
        z = torch.where(choose_1, z1, z2)
        base_conf = 0.70
        note = "sinusoid inverse with nearest-branch selection around current prediction"
    else:
        choose_1 = z1.abs() <= z2.abs()
        z = torch.where(choose_1, z1, z2)
        base_conf = 0.40
        note = "sinusoid inverse without reference branch; chose smallest-magnitude branch"

    denom = torch.sqrt(torch.clamp(1.0 - u_clip * u_clip, min=1.0e-12))
    gain = (1.0 / max(abs(float(omega)), 1.0e-12)) / denom
    gain = gain * abs(float(output_scale))
    confidence = _combine_inverse_confidence(
        base_conf,
        gain,
        mask=mask,
        confidence_mode=confidence_mode,
        confidence_target_gain=confidence_target_gain,
        confidence_floor=confidence_floor,
    )
    point_weight = _conditioning_point_weight_from_gain(
        gain,
        mask=mask,
        confidence_mode=confidence_mode,
        target_gain=confidence_target_gain,
        floor=confidence_floor,
    )
    z = torch.where(mask, z, torch.zeros_like(z))
    return z, mask, confidence, point_weight, note


def _invert_shifted_sinusoid_branches(
    target: torch.Tensor,
    *,
    A: float,
    B: float,
    c: float,
    omega: float,
    ref: torch.Tensor | None,
    output_scale: float = 1.0,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    branch_beam_width: int = 2,
) -> list[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, str, str]]:
    def _sheet_offsets(kmax: int) -> list[int]:
        out = [0]
        k = 1
        while len(out) < max(1, int(kmax)):
            out.append(k)
            if len(out) >= max(1, int(kmax)):
                break
            out.append(-k)
            k += 1
        return out

    def _masked_mean_abs(v: torch.Tensor, mask_: torch.Tensor) -> float:
        m = _bool_col(mask_)
        if int(m.sum().item()) <= 0:
            return float("inf")
        vv = _ensure_col(v)
        s = vv[m].abs()
        if int(s.numel()) <= 0:
            return float("inf")
        return float(s.mean().item())

    y = _ensure_col(target)
    out: list[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, str, str]] = []
    try:
        bw = max(1, int(branch_beam_width))
    except Exception:
        bw = 2
    if bw <= 1:
        z, mask, conf, pw, note = _invert_shifted_sinusoid(
            y,
            A=A,
            B=B,
            c=c,
            omega=omega,
            ref=ref,
            output_scale=output_scale,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        return [(z, mask, conf, pw, note, "main")]

    if not math.isfinite(float(omega)) or abs(float(omega)) < 1.0e-12:
        mask = _finite_mask(y)
        z = torch.zeros_like(y)
        return [(z, mask & False, 0.0, torch.zeros_like(y), "omega too small for sinusoid inversion", "main")]

    R = math.hypot(float(A), float(B))
    if not math.isfinite(R) or R < 1.0e-12:
        mask = _finite_mask(y)
        z = torch.zeros_like(y)
        return [(z, mask & False, 0.0, torch.zeros_like(y), "sinusoid amplitude too small for inversion", "main")]

    phi = math.atan2(float(B), float(A))
    u = (y - float(c)) / float(R)
    mask = _finite_mask(u) & (u.abs() <= 1.0 + 1.0e-7)
    u_clip = torch.clamp(u, -1.0, 1.0)
    theta = torch.asin(u_clip)
    period = 2.0 * math.pi / float(omega)

    z1 = (theta - phi) / float(omega)
    z2 = ((math.pi - theta) - phi) / float(omega)

    if ref is not None:
        z_ref = _ensure_col(ref)
        ref_mask = _finite_mask(z_ref)
        k1 = torch.round((z_ref - z1) / period)
        k2 = torch.round((z_ref - z2) / period)
        z1 = z1 + k1 * period
        z2 = z2 + k2 * period
        d1 = (z1 - z_ref).abs()
        d2 = (z2 - z_ref).abs()
        choose_1 = d1 <= d2
        choose_1 = torch.where(ref_mask, choose_1, torch.ones_like(choose_1, dtype=torch.bool))
        z_main = torch.where(choose_1, z1, z2)
        z_alt = torch.where(choose_1, z2, z1)
        note_main = "sinusoid inverse with nearest-branch selection around current prediction"
        note_alt = "sinusoid inverse alternate branch around current prediction"
        base_main = 0.70
        base_alt = 0.50
    else:
        choose_1 = z1.abs() <= z2.abs()
        z_main = torch.where(choose_1, z1, z2)
        z_alt = torch.where(choose_1, z2, z1)
        note_main = "sinusoid inverse without reference branch; chose smallest-magnitude branch"
        note_alt = "sinusoid inverse alternate branch without reference"
        base_main = 0.40
        base_alt = 0.30

    denom = torch.sqrt(torch.clamp(1.0 - u_clip * u_clip, min=1.0e-12))
    gain = (1.0 / max(abs(float(omega)), 1.0e-12)) / denom
    gain = gain * abs(float(output_scale))

    conf_main = _combine_inverse_confidence(
        base_main,
        gain,
        mask=mask,
        confidence_mode=confidence_mode,
        confidence_target_gain=confidence_target_gain,
        confidence_floor=confidence_floor,
    )
    pw_main = _conditioning_point_weight_from_gain(
        gain,
        mask=mask,
        confidence_mode=confidence_mode,
        target_gain=confidence_target_gain,
        floor=confidence_floor,
    )
    conf_alt0 = _combine_inverse_confidence(
        base_alt,
        gain,
        mask=mask,
        confidence_mode=confidence_mode,
        confidence_target_gain=confidence_target_gain,
        confidence_floor=confidence_floor,
    )
    pw_alt0 = _conditioning_point_weight_from_gain(
        gain,
        mask=mask,
        confidence_mode=confidence_mode,
        target_gain=confidence_target_gain,
        floor=confidence_floor,
    )

    # Multi-sheet periodic hypotheses around the local nearest sheets.
    # This keeps branch beam semantics backward compatible for bw<=2, and
    # adds additional +/-k*period hypotheses for wider beams.
    ref_mask = _finite_mask(_ensure_col(ref)) if ref is not None else None
    branch_rows: list[tuple[float, int, int, float, torch.Tensor, torch.Tensor, float, torch.Tensor, str, str]] = []
    base_defs = [
        ("main", z_main, float(base_main), note_main, 0, float(conf_main), pw_main),
        ("alt", z_alt, float(base_alt), note_alt, 1, float(conf_alt0), pw_alt0),
    ]
    offsets = _sheet_offsets(max(1, int(bw)))
    for token_base, z_base, base_conf, base_note, base_rank, base_conf0, base_pw0 in base_defs:
        for off in offsets:
            zc = z_base + float(off) * period
            zc_masked = torch.where(mask, zc, torch.zeros_like(zc))
            if int(off) == 0:
                conf_i = float(base_conf0)
                pw_i = base_pw0
            else:
                # Penalize farther sheets but keep them admissible in the beam.
                off_scale = 1.0 / (1.0 + 0.35 * abs(int(off)))
                conf_i = _combine_inverse_confidence(
                    float(base_conf) * float(off_scale),
                    gain,
                    mask=mask,
                    confidence_mode=confidence_mode,
                    confidence_target_gain=confidence_target_gain,
                    confidence_floor=confidence_floor,
                )
                pw_i = _conditioning_point_weight_from_gain(
                    gain * (1.0 + 0.25 * abs(int(off))),
                    mask=mask,
                    confidence_mode=confidence_mode,
                    target_gain=confidence_target_gain,
                    floor=confidence_floor,
                )
            if ref is not None:
                z_ref = _ensure_col(ref)
                score = _masked_mean_abs(zc - z_ref, mask & ref_mask)
            else:
                score = _masked_mean_abs(zc, mask)
            token = str(token_base) if int(off) == 0 else f"{token_base}:k{int(off):+d}"
            note = str(base_note) if int(off) == 0 else f"{base_note}; sheet offset {int(off):+d}"
            branch_rows.append((
                float(score),
                int(base_rank),
                abs(int(off)),
                -float(conf_i),
                zc_masked,
                mask,
                float(conf_i),
                pw_i,
                note,
                token,
            ))

    branch_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    out = [(row[4], row[5], row[6], row[7], row[8], row[9]) for row in branch_rows[: max(1, int(bw))]]
    return out


def invert_mapping_target(
    y_target: torch.Tensor,
    mapping: dict[str, Any] | None,
    *,
    pred_ref: torch.Tensor | None = None,
    x_ref: torch.Tensor | None = None,
    safe_eps: float = 1.0e-12,
    allow_identity_fallback: bool = True,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, float, bool, str]:
    y = _ensure_col(y_target)
    mask = _finite_mask(y)
    m = mapping or {"kind": "identity"}
    kind_raw = m.get("kind", "identity")
    kind = str(kind_raw).strip().lower() if kind_raw is not None else "identity"

    head_note = ""
    head_conf = 1.0
    if x_ref is not None and isinstance(m, dict) and (m.get("_lin_head", None) is not None):
        head_pred = _eval_linear_head(m.get("_lin_head", None), x_ref)
        if head_pred is not None and torch.isfinite(head_pred).all():
            y = y - head_pred
            mask = mask & _finite_mask(head_pred)
            head_note = "subtracted linear head; "
            head_conf = 0.95
        else:
            head_note = "could not subtract linear head; "
            head_conf = 0.75

    if kind in ("", "identity"):
        return y.clone(), mask, float(head_conf), True, head_note + "identity mapping"

    if kind in ("affine", "mono", "monomial"):
        a = float(m.get("a", 1.0))
        b = float(m.get("b", 0.0))
        if abs(a) <= safe_eps:
            if allow_identity_fallback:
                return y.clone(), mask, 0.25 * float(head_conf), False, head_note + "degenerate affine mapping; using raw target"
            return torch.zeros_like(y), mask & False, 0.0, False, head_note + "degenerate affine mapping"
        z = (y - b) / a
        out_mask = mask & _finite_mask(z)
        gain = torch.full_like(z, abs(1.0 / a))
        conf = _combine_inverse_confidence(
            1.0 * float(head_conf),
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        return z, out_mask, conf, True, head_note + "inverted affine mapping"

    if kind == "poly":
        coeffs = m.get("coeffs", None)
        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.detach().cpu().tolist()
        if isinstance(coeffs, (list, tuple)) and len(coeffs) == 2:
            c0 = float(coeffs[0])
            c1 = float(coeffs[1])
            mu = float(m.get("mu", 0.0))
            std = float(m.get("std", 1.0))
            if abs(c1) <= safe_eps or abs(std) <= safe_eps:
                if allow_identity_fallback:
                    return y.clone(), mask, 0.25 * float(head_conf), False, head_note + "degenerate affine polynomial mapping; using raw target"
                return torch.zeros_like(y), mask & False, 0.0, False, head_note + "degenerate affine polynomial mapping"
            fn = (y - c0) / c1
            f = mu + std * fn
            out_mask = mask & _finite_mask(f)
            gain = torch.full_like(f, abs(std / c1))
            conf = _combine_inverse_confidence(
                1.0 * float(head_conf),
                gain,
                mask=out_mask,
                confidence_mode=confidence_mode,
                confidence_target_gain=confidence_target_gain,
                confidence_floor=confidence_floor,
            )
            return f, out_mask, conf, True, head_note + "inverted affine polynomial mapping"
        if allow_identity_fallback:
            deg = 0 if coeffs is None else max(0, len(coeffs) - 1)
            return y.clone(), mask, 0.20 * float(head_conf), False, head_note + f"polynomial degree {deg} not inverted"
        return torch.zeros_like(y), mask & False, 0.0, False, head_note + "non-affine polynomial mapping unsupported"

    if kind == "power":
        b = float(m.get("b", 0.0))
        log_a = float(m.get("log_a", 0.0))
        sgn_f = float(m.get("sgn_f", 1.0))
        sgn_y = float(m.get("sgn_y", 1.0))
        if abs(b) <= safe_eps:
            if allow_identity_fallback:
                return y.clone(), mask, 0.20 * float(head_conf), False, head_note + "power exponent too small; using raw target"
            return torch.zeros_like(y), mask & False, 0.0, False, head_note + "power exponent too small"
        y_pos = sgn_y * y
        mask = mask & (y_pos > 0.0)
        f_pos = torch.exp((torch.log(torch.clamp(y_pos, min=safe_eps)) - log_a) / b)
        f = sgn_f * f_pos
        f = torch.where(mask, f, torch.zeros_like(f))
        out_mask = mask & _finite_mask(f)
        gain = f_pos / (max(abs(b), safe_eps) * torch.clamp(y_pos.abs(), min=safe_eps))
        conf = _combine_inverse_confidence(
            0.85 * float(head_conf),
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        return f, out_mask, conf, True, head_note + "inverted power mapping"

    if kind == "exp":
        a = float(m.get("a", 1.0))
        b = float(m.get("b", 0.0))
        c = float(m.get("c", 0.0))
        mu = float(m.get("mu", 0.0))
        std = float(m.get("std", 1.0))
        if abs(a) <= safe_eps or abs(b) <= safe_eps or abs(std) <= safe_eps:
            if allow_identity_fallback:
                return y.clone(), mask, 0.20 * float(head_conf), False, head_note + "degenerate exponential mapping; using raw target"
            return torch.zeros_like(y), mask & False, 0.0, False, head_note + "degenerate exponential mapping"
        ratio = (y - c) / a
        mask = mask & (ratio > 0.0)
        z = torch.log(torch.clamp(ratio, min=safe_eps)) / b
        f = mu + std * z
        f = torch.where(mask, f, torch.zeros_like(f))
        out_mask = mask & _finite_mask(f)
        gain = abs(std / b) / torch.clamp((y - c).abs(), min=safe_eps)
        conf = _combine_inverse_confidence(
            0.85 * float(head_conf),
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        return f, out_mask, conf, True, head_note + "inverted exponential mapping"

    if kind == "sine":
        mu = float(m.get("mu", 0.0))
        std = float(m.get("std", 1.0))
        if abs(std) <= safe_eps:
            if allow_identity_fallback:
                return y.clone(), mask, 0.20 * float(head_conf), False, head_note + "degenerate sine mapping; using raw target"
            return torch.zeros_like(y), mask & False, 0.0, False, head_note + "degenerate sine mapping"
        z_ref = None
        if pred_ref is not None:
            z_ref = (_ensure_col(pred_ref) - mu) / std
        z, zmask, conf, _pw, note = _invert_shifted_sinusoid(
            y,
            A=float(m.get("A", 0.0)),
            B=float(m.get("B", 0.0)),
            c=float(m.get("c", 0.0)),
            omega=float(m.get("omega", 1.0)),
            ref=z_ref,
            output_scale=float(std),
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        f = mu + std * z
        return f, zmask & _finite_mask(f), float(conf) * float(head_conf), True, head_note + f"outer {note}"

    if kind == "pade":
        numer = m.get("numer", None)
        denom = m.get("denom", None)
        if isinstance(numer, torch.Tensor):
            numer = numer.detach().cpu().tolist()
        if isinstance(denom, torch.Tensor):
            denom = denom.detach().cpu().tolist()
        if isinstance(numer, (list, tuple)) and isinstance(denom, (list, tuple)) and len(numer) <= 2 and len(denom) <= 2:
            n0 = float(numer[0]) if len(numer) >= 1 else 0.0
            n1 = float(numer[1]) if len(numer) >= 2 else 0.0
            d0 = float(denom[0]) if len(denom) >= 1 else 1.0
            d1 = float(denom[1]) if len(denom) >= 2 else 0.0
            mu = float(m.get("mu", 0.0))
            std = float(m.get("std", 1.0))
            if abs(std) <= safe_eps:
                if allow_identity_fallback:
                    return y.clone(), mask, 0.20 * float(head_conf), False, head_note + "degenerate Padé mapping; using raw target"
                return torch.zeros_like(y), mask & False, 0.0, False, head_note + "degenerate Padé mapping"
            den = y * d1 - n1
            mask = mask & (den.abs() > safe_eps)
            z = (n0 - y * d0) / torch.where(mask, den, torch.ones_like(den))
            f = mu + std * z
            f = torch.where(mask, f, torch.zeros_like(f))
            out_mask = mask & _finite_mask(f)
            jac_num = abs(d0 * n1 - d1 * n0)
            gain = abs(std) * jac_num / torch.clamp(den.abs() * den.abs(), min=safe_eps)
            conf = _combine_inverse_confidence(
                0.70 * float(head_conf),
                gain,
                mask=out_mask,
                confidence_mode=confidence_mode,
                confidence_target_gain=confidence_target_gain,
                confidence_floor=confidence_floor,
            )
            return f, out_mask, conf, True, head_note + "inverted linear-fractional Padé mapping"
        if allow_identity_fallback:
            return y.clone(), mask, 0.20 * float(head_conf), False, head_note + "higher-order Padé mapping not inverted"
        return torch.zeros_like(y), mask & False, 0.0, False, head_note + "higher-order Padé mapping unsupported"

    if allow_identity_fallback:
        return y.clone(), mask, 0.10 * float(head_conf), False, head_note + f"mapping kind '{kind}' not inverted"
    return torch.zeros_like(y), mask & False, 0.0, False, head_note + f"mapping kind '{kind}' unsupported"


def _mapping_inverse_point_weight(
    mapping: dict[str, Any] | None,
    y_target: torch.Tensor,
    inv_target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    x_ref: torch.Tensor | None,
    safe_eps: float,
    confidence_mode: str,
    confidence_target_gain: float,
    confidence_floor: float,
) -> torch.Tensor:
    y = _ensure_col(y_target).clone()
    t = _ensure_col(inv_target)
    mask = _bool_col(valid_mask) & _finite_mask(y, t)
    m = mapping if isinstance(mapping, dict) else {}

    # Match invert_mapping_target: if a linear head exists and can be evaluated,
    # invert the outer map against y - head(x).
    if x_ref is not None and (m.get("_lin_head", None) is not None):
        try:
            head_pred = _eval_linear_head(m.get("_lin_head", None), x_ref)
        except Exception:
            head_pred = None
        if head_pred is not None and torch.isfinite(head_pred).all() and int(head_pred.shape[0]) == int(y.shape[0]):
            y = y - head_pred
            mask = mask & _finite_mask(y)

    kind = str(m.get("kind", "identity")).strip().lower()
    eps = float(max(1.0e-12, safe_eps))
    gain = torch.ones_like(t)

    try:
        if kind in ("", "identity"):
            gain = torch.ones_like(t)
        elif kind == "affine":
            a = float(m.get("a", 1.0))
            gain = torch.full_like(t, abs(1.0 / max(abs(a), eps)))
        elif kind == "poly":
            coeffs = m.get("coeffs", None)
            if isinstance(coeffs, torch.Tensor):
                coeffs = coeffs.detach().cpu().tolist()
            if isinstance(coeffs, (list, tuple)) and len(coeffs) == 2:
                c1 = float(coeffs[1])
                std = float(m.get("std", 1.0))
                gain = torch.full_like(t, abs(std / max(abs(c1), eps)))
            else:
                gain = torch.ones_like(t)
        elif kind == "power":
            b = float(m.get("b", 1.0))
            sy = float(m.get("sy", 1.0))
            y_pos = sy * y
            mask = mask & (y_pos > eps)
            gain = t.abs() / (max(abs(b), eps) * torch.clamp(y_pos.abs(), min=eps))
        elif kind == "exp":
            b = float(m.get("b", 1.0))
            c = float(m.get("c", 0.0))
            std = float(m.get("std", 1.0))
            gain = abs(std / max(abs(b), eps)) / torch.clamp((y - c).abs(), min=eps)
        elif kind == "sine":
            A = float(m.get("A", 0.0))
            B = float(m.get("B", 0.0))
            c = float(m.get("c", 0.0))
            omega = float(m.get("omega", 1.0))
            std = float(m.get("std", 1.0))
            R = math.hypot(A, B)
            if R <= eps:
                gain = torch.full_like(t, 1.0e6)
            else:
                u = (y - c) / R
                mask = mask & (u.abs() <= 1.0 + 1.0e-7)
                u_clip = torch.clamp(u, min=-1.0, max=1.0)
                den = torch.sqrt(torch.clamp(1.0 - u_clip * u_clip, min=eps))
                gain = (abs(std) / max(abs(omega) * R, eps)) / den
        elif kind == "pade":
            numer = m.get("numer", None)
            denom = m.get("denom", None)
            if isinstance(numer, torch.Tensor):
                numer = numer.detach().cpu().tolist()
            if isinstance(denom, torch.Tensor):
                denom = denom.detach().cpu().tolist()
            if isinstance(numer, (list, tuple)) and isinstance(denom, (list, tuple)) and len(numer) <= 2 and len(denom) <= 2:
                n0 = float(numer[0]) if len(numer) >= 1 else 0.0
                n1 = float(numer[1]) if len(numer) >= 2 else 0.0
                d0 = float(denom[0]) if len(denom) >= 1 else 1.0
                d1 = float(denom[1]) if len(denom) >= 2 else 0.0
                std = float(m.get("std", 1.0))
                den = y * d1 - n1
                mask = mask & (den.abs() > eps)
                jac_num = abs(d0 * n1 - d1 * n0)
                gain = abs(std) * jac_num / torch.clamp(den.abs() * den.abs(), min=eps)
            else:
                gain = torch.ones_like(t)
        else:
            gain = torch.ones_like(t)
    except Exception:
        gain = torch.ones_like(t)

    pw = _conditioning_point_weight_from_gain(
        gain,
        mask=mask,
        confidence_mode=confidence_mode,
        target_gain=confidence_target_gain,
        floor=confidence_floor,
    )
    return torch.where(mask, pw, torch.zeros_like(pw))


def _invert_unary_context(
    op: str,
    parent_target: torch.Tensor,
    *,
    child_pred_ref: torch.Tensor | None,
    safe_eps: float,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, str]:
    t = _ensure_col(parent_target)
    mask = _finite_mask(t)

    if op == "neg":
        child_t = -t
        out_mask = mask & _finite_mask(child_t)
        gain = torch.ones_like(child_t)
        conf = _combine_inverse_confidence(
            1.0,
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw = _conditioning_point_weight_from_gain(
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        return child_t, out_mask, conf, pw, "neg inverse"

    if op == "exp":
        step_mask = mask & (t > 0.0)
        child_t = torch.log(torch.clamp(t, min=safe_eps))
        child_t = torch.where(step_mask, child_t, torch.zeros_like(child_t))
        out_mask = step_mask & _finite_mask(child_t)
        gain = 1.0 / torch.clamp(t.abs(), min=safe_eps)
        conf = _combine_inverse_confidence(
            0.95,
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw = _conditioning_point_weight_from_gain(
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        return child_t, out_mask, conf, pw, "exp inverse via log"

    if op == "log":
        child_t = torch.exp(torch.clamp(t, min=-50.0, max=50.0))
        out_mask = mask & _finite_mask(child_t)
        gain = child_t.abs()
        conf = _combine_inverse_confidence(
            0.95,
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw = _conditioning_point_weight_from_gain(
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        return child_t, out_mask, conf, pw, "log inverse via exp"

    if op == "sqrt":
        step_mask = mask & (t >= 0.0)
        child_t = t * t
        child_t = torch.where(step_mask, child_t, torch.zeros_like(child_t))
        out_mask = step_mask & _finite_mask(child_t)
        gain = 2.0 * t.abs()
        conf = _combine_inverse_confidence(
            0.90,
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw = _conditioning_point_weight_from_gain(
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        return child_t, out_mask, conf, pw, "sqrt inverse via square"

    if op == "sqr":
        step_mask = mask & (t >= 0.0)
        mag = torch.sqrt(torch.clamp(t, min=0.0))
        if child_pred_ref is not None:
            ref = _ensure_col(child_pred_ref)
            sign = torch.where(ref >= 0.0, torch.ones_like(ref), -torch.ones_like(ref))
            child_t = sign * mag
            note = "square inverse via signed sqrt using current subtree sign"
            base_conf = 0.65
        else:
            child_t = mag
            note = "square inverse via principal sqrt"
            base_conf = 0.35
        child_t = torch.where(step_mask, child_t, torch.zeros_like(child_t))
        out_mask = step_mask & _finite_mask(child_t)
        gain = 1.0 / torch.clamp(2.0 * mag, min=safe_eps)
        conf = _combine_inverse_confidence(
            base_conf,
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw = _conditioning_point_weight_from_gain(
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        return child_t, out_mask, conf, pw, note

    if op == "sin":
        return _invert_shifted_sinusoid(
            t,
            A=1.0,
            B=0.0,
            c=0.0,
            omega=1.0,
            ref=child_pred_ref,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )

    if op == "cos":
        return _invert_shifted_sinusoid(
            t,
            A=0.0,
            B=1.0,
            c=0.0,
            omega=1.0,
            ref=child_pred_ref,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )

    return torch.zeros_like(t), mask & False, 0.0, torch.zeros_like(t), f"unary op '{op}' not supported"


def _invert_unary_context_branches(
    op: str,
    parent_target: torch.Tensor,
    *,
    child_pred_ref: torch.Tensor | None,
    safe_eps: float,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    branch_beam_width: int = 2,
) -> list[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, str, str]]:
    try:
        bw = max(1, int(branch_beam_width))
    except Exception:
        bw = 2
    if bw <= 1 or op not in ("sin", "cos", "sqr"):
        t, m, c, pw, note = _invert_unary_context(
            op,
            parent_target,
            child_pred_ref=child_pred_ref,
            safe_eps=safe_eps,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        return [(t, m, c, pw, note, "main")]

    t = _ensure_col(parent_target)
    mask = _finite_mask(t)

    if op == "sqr":
        step_mask = mask & (t >= 0.0)
        mag = torch.sqrt(torch.clamp(t, min=0.0))
        if child_pred_ref is not None:
            ref = _ensure_col(child_pred_ref)
            sign = torch.where(ref >= 0.0, torch.ones_like(ref), -torch.ones_like(ref))
            child_main = sign * mag
            child_alt = -sign * mag
            note_main = "square inverse via signed sqrt using current subtree sign"
            note_alt = "square inverse alternate signed branch"
            base_main = 0.65
            base_alt = 0.45
        else:
            child_main = mag
            child_alt = -mag
            note_main = "square inverse via principal sqrt"
            note_alt = "square inverse via negative principal sqrt"
            base_main = 0.35
            base_alt = 0.25

        gain = 1.0 / torch.clamp(2.0 * mag, min=safe_eps)
        conf_main = _combine_inverse_confidence(
            base_main,
            gain,
            mask=step_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        conf_alt = _combine_inverse_confidence(
            base_alt,
            gain,
            mask=step_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw_main = _conditioning_point_weight_from_gain(
            gain,
            mask=step_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        pw_alt = _conditioning_point_weight_from_gain(
            gain,
            mask=step_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        out = [
            (
                torch.where(step_mask, child_main, torch.zeros_like(child_main)),
                step_mask & _finite_mask(child_main),
                conf_main,
                pw_main,
                note_main,
                "main",
            ),
            (
                torch.where(step_mask, child_alt, torch.zeros_like(child_alt)),
                step_mask & _finite_mask(child_alt),
                conf_alt,
                pw_alt,
                note_alt,
                "alt",
            ),
        ]
        return out[:bw]

    if op == "sin":
        return _invert_shifted_sinusoid_branches(
            t,
            A=1.0,
            B=0.0,
            c=0.0,
            omega=1.0,
            ref=child_pred_ref,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
            branch_beam_width=bw,
        )

    if op == "cos":
        return _invert_shifted_sinusoid_branches(
            t,
            A=0.0,
            B=1.0,
            c=0.0,
            omega=1.0,
            ref=child_pred_ref,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
            branch_beam_width=bw,
        )

    return []


def _invert_binary_context(
    op: str,
    parent_target: torch.Tensor,
    *,
    child_slot: int,
    other_pred: torch.Tensor,
    safe_eps: float,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, str]:
    t = _ensure_col(parent_target)
    o = _ensure_col(other_pred)
    mask = _finite_mask(t, o)

    if op == "add":
        child_t = t - o
        out_mask = mask & _finite_mask(child_t)
        gain = torch.ones_like(child_t)
        conf = _combine_inverse_confidence(
            1.0,
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw = _conditioning_point_weight_from_gain(
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        return child_t, out_mask, conf, pw, "add inverse"

    if op == "sub":
        child_t = t + o if int(child_slot) == 1 else o - t
        note = "sub inverse (left child)" if int(child_slot) == 1 else "sub inverse (right child)"
        out_mask = mask & _finite_mask(child_t)
        gain = torch.ones_like(child_t)
        conf = _combine_inverse_confidence(
            1.0,
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw = _conditioning_point_weight_from_gain(
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        return child_t, out_mask, conf, pw, note

    if op == "mul":
        step_mask = mask & (o.abs() > safe_eps)
        child_t = t / torch.where(step_mask, o, torch.ones_like(o))
        child_t = torch.where(step_mask, child_t, torch.zeros_like(child_t))
        out_mask = step_mask & _finite_mask(child_t)
        gain = 1.0 / torch.clamp(o.abs(), min=safe_eps)
        conf = _combine_inverse_confidence(
            0.90,
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw = _conditioning_point_weight_from_gain(
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        return child_t, out_mask, conf, pw, "mul inverse via division by sibling"

    if op == "div":
        if int(child_slot) == 1:
            child_t = t * o
            out_mask = mask & _finite_mask(child_t)
            gain = o.abs()
            conf = _combine_inverse_confidence(
                0.95,
                gain,
                mask=out_mask,
                confidence_mode=confidence_mode,
                confidence_target_gain=confidence_target_gain,
                confidence_floor=confidence_floor,
            )
            pw = _conditioning_point_weight_from_gain(
                gain,
                mask=out_mask,
                confidence_mode=confidence_mode,
                target_gain=confidence_target_gain,
                floor=confidence_floor,
            )
            return child_t, out_mask, conf, pw, "div inverse for numerator"
        step_mask = mask & (t.abs() > safe_eps)
        child_t = o / torch.where(step_mask, t, torch.ones_like(t))
        child_t = torch.where(step_mask, child_t, torch.zeros_like(child_t))
        out_mask = step_mask & _finite_mask(child_t)
        gain = o.abs() / torch.clamp(t.abs() * t.abs(), min=safe_eps)
        conf = _combine_inverse_confidence(
            0.80,
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            confidence_target_gain=confidence_target_gain,
            confidence_floor=confidence_floor,
        )
        pw = _conditioning_point_weight_from_gain(
            gain,
            mask=out_mask,
            confidence_mode=confidence_mode,
            target_gain=confidence_target_gain,
            floor=confidence_floor,
        )
        return child_t, out_mask, conf, pw, "div inverse for denominator"

    return torch.zeros_like(t), mask & False, 0.0, torch.zeros_like(t), f"binary op '{op}' not supported"


@torch.no_grad()
def invert_context_target_beam(
    candidate_ast: tuple,
    path: Sequence[int] | None,
    x: torch.Tensor,
    y_target: torch.Tensor,
    *,
    mapping: dict[str, Any] | None = None,
    safe_eps: float = 1.0e-12,
    allow_identity_fallback: bool = True,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    branch_beam_width: int = 1,
) -> list[InverseTarget]:
    pp = tuple(int(v) for v in (path or ()))
    y = _ensure_col(y_target)
    all_paths = set(collect_paths(candidate_ast))
    if pp not in all_paths:
        raise ValueError(f"path={pp} is not present in candidate AST {node_str(candidate_ast)}")

    root_pred = eval_node(candidate_ast, x)
    root_t, mask, conf, mapping_ok, mapping_note = invert_mapping_target(
        y,
        mapping,
        pred_ref=root_pred,
        x_ref=x,
        safe_eps=safe_eps,
        allow_identity_fallback=allow_identity_fallback,
        confidence_mode=confidence_mode,
        confidence_target_gain=confidence_target_gain,
        confidence_floor=confidence_floor,
    )
    root_pw = _mapping_inverse_point_weight(
        mapping,
        y,
        root_t,
        mask,
        x_ref=x,
        safe_eps=safe_eps,
        confidence_mode=confidence_mode,
        confidence_target_gain=confidence_target_gain,
        confidence_floor=confidence_floor,
    )
    conf0 = min(1.0, max(0.0, float(conf)))
    conf0_log = math.log(max(conf0, 1.0e-12))
    root_pw_log = torch.log(torch.clamp(root_pw, min=1.0e-12))
    state0 = {
        "cur": candidate_ast,
        "cur_path": tuple(),
        "target": root_t,
        "mask": mask,
        "point_weight": root_pw,
        "pw_log_sum": root_pw_log,
        "pw_steps": 1,
        "total_conf": conf0,
        "conf_log_sum": conf0_log,
        "conf_steps": 1,
        "steps": [
            InverseStep(
                parent_path=(),
                op=f"mapping:{str((mapping or {}).get('kind', 'identity'))}",
                child_slot=0,
                valid_fraction=_mask_fraction(mask),
                confidence=float(conf),
                note=mapping_note,
            )
        ],
        "branch_tokens": [],
    }
    try:
        beam_w = max(1, int(branch_beam_width))
    except Exception:
        beam_w = 1
    states = [state0]

    for slot in pp:
        next_states = []
        for st in states:
            cur = st["cur"]
            cur_path = tuple(st["cur_path"])
            cur_target = st["target"]
            cur_mask = st["mask"]
            cur_pw = st["point_weight"]
            cur_pw_log_sum = st.get("pw_log_sum", torch.log(torch.clamp(cur_pw, min=1.0e-12)))
            cur_pw_steps = int(st.get("pw_steps", 1))
            total_conf = float(st["total_conf"])
            conf_log_sum = float(st.get("conf_log_sum", math.log(max(total_conf, 1.0e-12))))
            conf_steps = int(st.get("conf_steps", 1))
            steps = list(st["steps"])
            branch_tokens = list(st["branch_tokens"])

            op = str(cur[0])
            if op in ("var", "const", "hparam"):
                raise ValueError(f"path {pp} descends beyond a leaf at prefix {cur_path}")

            if op in ("sin", "cos", "exp", "log", "sqrt", "sqr", "neg"):
                if int(slot) != 1:
                    raise ValueError(f"bad unary slot {slot} under {op} at path {cur_path}")
                child = cur[1]
                child_pred = eval_node(child, x)
                step_branches = _invert_unary_context_branches(
                    op,
                    cur_target,
                    child_pred_ref=child_pred,
                    safe_eps=safe_eps,
                    confidence_mode=confidence_mode,
                    confidence_target_gain=confidence_target_gain,
                    confidence_floor=confidence_floor,
                    branch_beam_width=beam_w,
                )
                if not step_branches:
                    continue
                for step_t, step_mask, step_conf, step_pw, note, token in step_branches:
                    new_cur_path = cur_path + (int(slot),)
                    new_mask = cur_mask & step_mask
                    step_pw_log = torch.log(torch.clamp(step_pw, min=1.0e-12))
                    new_pw_log_sum = cur_pw_log_sum + step_pw_log
                    new_pw_steps = cur_pw_steps + 1
                    new_pw = torch.exp(new_pw_log_sum / max(1, new_pw_steps))
                    new_pw = torch.where(new_mask, new_pw, torch.zeros_like(new_pw))
                    step_c = min(1.0, max(0.0, float(step_conf)))
                    new_conf_log = conf_log_sum + math.log(max(step_c, 1.0e-12))
                    new_conf_steps = conf_steps + 1
                    new_conf = math.exp(new_conf_log / max(1, new_conf_steps))
                    new_target = torch.where(new_mask, step_t, torch.zeros_like(step_t))
                    new_steps = steps + [
                        InverseStep(
                            parent_path=new_cur_path[:-1],
                            op=op,
                            child_slot=int(slot),
                            valid_fraction=_mask_fraction(new_mask),
                            confidence=float(step_conf),
                            note=note,
                        )
                    ]
                    new_tokens = branch_tokens + ([f"{new_cur_path}:{op}:{token}"] if token != "main" else [])
                    next_states.append(
                        {
                            "cur": child,
                            "cur_path": new_cur_path,
                            "target": new_target,
                            "mask": new_mask,
                            "point_weight": new_pw,
                            "pw_log_sum": new_pw_log_sum,
                            "pw_steps": new_pw_steps,
                            "total_conf": new_conf,
                            "conf_log_sum": new_conf_log,
                            "conf_steps": new_conf_steps,
                            "steps": new_steps,
                            "branch_tokens": new_tokens,
                        }
                    )
            else:
                if int(slot) not in (1, 2):
                    raise ValueError(f"bad binary slot {slot} under {op} at path {cur_path}")
                child = cur[int(slot)]
                other = cur[2 if int(slot) == 1 else 1]
                other_pred = eval_node(other, x)
                step_t, step_mask, step_conf, step_pw, note = _invert_binary_context(
                    op,
                    cur_target,
                    child_slot=int(slot),
                    other_pred=other_pred,
                    safe_eps=safe_eps,
                    confidence_mode=confidence_mode,
                    confidence_target_gain=confidence_target_gain,
                    confidence_floor=confidence_floor,
                )
                new_cur_path = cur_path + (int(slot),)
                new_mask = cur_mask & step_mask
                step_pw_log = torch.log(torch.clamp(step_pw, min=1.0e-12))
                new_pw_log_sum = cur_pw_log_sum + step_pw_log
                new_pw_steps = cur_pw_steps + 1
                new_pw = torch.exp(new_pw_log_sum / max(1, new_pw_steps))
                new_pw = torch.where(new_mask, new_pw, torch.zeros_like(new_pw))
                step_c = min(1.0, max(0.0, float(step_conf)))
                new_conf_log = conf_log_sum + math.log(max(step_c, 1.0e-12))
                new_conf_steps = conf_steps + 1
                new_conf = math.exp(new_conf_log / max(1, new_conf_steps))
                new_target = torch.where(new_mask, step_t, torch.zeros_like(step_t))
                new_steps = steps + [
                    InverseStep(
                        parent_path=new_cur_path[:-1],
                        op=op,
                        child_slot=int(slot),
                        valid_fraction=_mask_fraction(new_mask),
                        confidence=float(step_conf),
                        note=note,
                    )
                ]
                next_states.append(
                    {
                        "cur": child,
                        "cur_path": new_cur_path,
                        "target": new_target,
                        "mask": new_mask,
                        "point_weight": new_pw,
                        "pw_log_sum": new_pw_log_sum,
                        "pw_steps": new_pw_steps,
                        "total_conf": new_conf,
                        "conf_log_sum": new_conf_log,
                        "conf_steps": new_conf_steps,
                        "steps": new_steps,
                        "branch_tokens": branch_tokens,
                    }
                )

        if not next_states:
            states = []
            break

        # Keep a small branch beam by (confidence * valid_fraction).
        next_states.sort(
            key=lambda st: (
                float(st["total_conf"]) * _mask_fraction(st["mask"]),
                float(st["total_conf"]),
                _mask_fraction(st["mask"]),
            ),
            reverse=True,
        )
        pruned = []
        seen = set()
        for st in next_states:
            bid = "|".join(st["branch_tokens"]) if st["branch_tokens"] else "main"
            if bid in seen:
                continue
            seen.add(bid)
            pruned.append(st)
            if len(pruned) >= beam_w:
                break
        states = pruned

    out = []
    mapping_kind = str((mapping or {}).get("kind", "identity"))
    for st in states:
        bid = "|".join(st["branch_tokens"]) if st["branch_tokens"] else "main"
        out.append(
            InverseTarget(
                path=pp,
                target=st["target"],
                valid_mask=st["mask"],
                point_weight=st["point_weight"],
                confidence=float(st["total_conf"]),
                mapping_inverted=bool(mapping_ok),
                mapping_kind=mapping_kind,
                steps=tuple(st["steps"]),
                branch_id=str(bid),
            )
        )
    if not out:
        out = [
            InverseTarget(
                path=pp,
                target=torch.zeros_like(root_t),
                valid_mask=mask & False,
                point_weight=torch.zeros_like(root_t),
                confidence=0.0,
                mapping_inverted=bool(mapping_ok),
                mapping_kind=mapping_kind,
                steps=tuple(state0["steps"]),
                branch_id="main",
            )
        ]
    return out


@torch.no_grad()
def invert_context_target(
    candidate_ast: tuple,
    path: Sequence[int] | None,
    x: torch.Tensor,
    y_target: torch.Tensor,
    *,
    mapping: dict[str, Any] | None = None,
    safe_eps: float = 1.0e-12,
    allow_identity_fallback: bool = True,
    confidence_mode: str = "conditioning",
    confidence_target_gain: float = 4.0,
    confidence_floor: float = 0.05,
    branch_beam_width: int = 1,
) -> InverseTarget:
    targets = invert_context_target_beam(
        candidate_ast,
        path,
        x,
        y_target,
        mapping=mapping,
        safe_eps=safe_eps,
        allow_identity_fallback=allow_identity_fallback,
        confidence_mode=confidence_mode,
        confidence_target_gain=confidence_target_gain,
        confidence_floor=confidence_floor,
        branch_beam_width=branch_beam_width,
    )
    return targets[0]


def _cheap_affine_probe_stats_from_preds(
    pred_fit: torch.Tensor,
    pred_probe: torch.Tensor,
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    *,
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
) -> tuple[float, float] | None:
    pf = _ensure_col(pred_fit)
    pp = _ensure_col(pred_probe)
    tf = _ensure_col(t_fit)
    tp = _ensure_col(t_probe)
    wf = _prepare_nonnegative_weights(w_fit, tf)
    wp = _prepare_nonnegative_weights(w_probe, tp)

    mfit = (_finite_mask(pf, tf, wf) & (wf > 0.0)).squeeze(-1)
    mprobe = (_finite_mask(pp, tp, wp) & (wp > 0.0)).squeeze(-1)
    if int(mfit.sum().item()) < 4 or int(mprobe.sum().item()) < 4:
        return None

    f = pf[mfit, 0]
    y = tf[mfit, 0]
    w = wf[mfit, 0]
    A = torch.stack([torch.ones_like(f), f], dim=1)
    try:
        sw = torch.sqrt(torch.clamp(w, min=0.0))
        Aw = A * sw.unsqueeze(1)
        yw = y * sw
        gram = Aw.T @ Aw
        rhs = Aw.T @ yw
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        sol = torch.linalg.solve(gram + 1.0e-12 * eye, rhs)
    except Exception:
        return None
    if not torch.isfinite(sol).all():
        return None

    fp = pp[mprobe, 0]
    yhat = (sol[0] + sol[1] * fp).unsqueeze(-1)
    if not torch.isfinite(yhat).all():
        return None
    yt = tp[mprobe]
    fit_hat = (sol[0] + sol[1] * f).unsqueeze(-1)
    mse = _weighted_mse_cols(yt, yhat, wp[mprobe])
    fit_mse = _weighted_mse_cols(tf[mfit], fit_hat, wf[mfit])
    if mse is None or fit_mse is None:
        return None
    return fit_mse, mse


def _score_predictions_on_target(
    pred_fit: torch.Tensor,
    pred_probe: torch.Tensor,
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    poly_degree: int,
) -> tuple[float, float, dict[str, Any]] | None:
    pf = _ensure_col(pred_fit)
    pp = _ensure_col(pred_probe)
    tf = _ensure_col(t_fit)
    tp = _ensure_col(t_probe)
    if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
        return None
    fb = fit_best(pf, tf, poly_degree)
    if fb is None:
        return None
    fit_mse, mapping = fb
    try:
        tp_hat = eval_mapping(pp, mapping)
    except Exception:
        return None
    if not torch.isfinite(tp_hat).all():
        return None
    probe_mse = float(((tp - tp_hat) ** 2).mean().item())
    if not math.isfinite(probe_mse):
        return None
    return float(fit_mse), float(probe_mse), mapping


def _normalize_inverse_local_score_mode(mode: str | None, *, default: str = "affine") -> str:
    def _norm_single(v: str | None) -> str:
        mm = str("" if v is None else v).strip().lower()
        if mm in ("lin", "linear"):
            return "affine"
        if mm in ("direct",):
            return "strict"
        if mm in ("full", "mapping"):
            return "fitbest"
        if mm in ("strict", "affine", "fitbest"):
            return mm
        return ""

    mm = _norm_single(mode)
    if mm != "":
        return mm
    fallback = _norm_single(default)
    if fallback != "":
        return fallback
    return "affine"


def _normalize_inverse_target_mode(mode: str | None, *, default: str = "robust") -> str:
    def _norm_single(v: str | None) -> str:
        mm = str("" if v is None else v).strip().lower()
        if mm in ("", "auto"):
            return ""
        if mm in ("legacy",):
            return "full"
        if mm in ("identity_affine", "simple", "robust", "full", "identity", "affine"):
            return mm
        return ""

    mm = _norm_single(mode)
    if mm != "":
        return mm
    fallback = _norm_single(default)
    if fallback != "":
        return fallback
    return "robust"


def _score_inverse_local_predictions(
    pred_fit: torch.Tensor,
    pred_probe: torch.Tensor,
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    *,
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
    poly_degree: int,
    mode: str = "affine",
) -> tuple[float, float] | None:
    mm = _normalize_inverse_local_score_mode(mode, default="affine")
    if mm in ("strict", "direct"):
        pf = _ensure_col(pred_fit)
        pp = _ensure_col(pred_probe)
        tf = _ensure_col(t_fit)
        tp = _ensure_col(t_probe)
        wf = _prepare_nonnegative_weights(w_fit, tf)
        wp = _prepare_nonnegative_weights(w_probe, tp)
        mfit = (_finite_mask(pf, tf, wf) & (wf > 0.0)).squeeze(-1)
        mprobe = (_finite_mask(pp, tp, wp) & (wp > 0.0)).squeeze(-1)
        if int(mfit.sum().item()) < 4 or int(mprobe.sum().item()) < 4:
            return _cheap_affine_probe_stats_from_preds(
                pred_fit,
                pred_probe,
                t_fit,
                t_probe,
                w_fit=w_fit,
                w_probe=w_probe,
            )
        fit_mse = _weighted_mse_cols(tf[mfit], pf[mfit], wf[mfit])
        probe_mse = _weighted_mse_cols(tp[mprobe], pp[mprobe], wp[mprobe])
        if fit_mse is None or probe_mse is None:
            return _cheap_affine_probe_stats_from_preds(
                pred_fit,
                pred_probe,
                t_fit,
                t_probe,
                w_fit=w_fit,
                w_probe=w_probe,
            )
        if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
            return _cheap_affine_probe_stats_from_preds(
                pred_fit,
                pred_probe,
                t_fit,
                t_probe,
                w_fit=w_fit,
                w_probe=w_probe,
            )
        return fit_mse, probe_mse
    if mm in ("affine", "lin", "linear"):
        return _cheap_affine_probe_stats_from_preds(
            pred_fit,
            pred_probe,
            t_fit,
            t_probe,
            w_fit=w_fit,
            w_probe=w_probe,
        )
    pf = _ensure_col(pred_fit)
    pp = _ensure_col(pred_probe)
    tf = _ensure_col(t_fit)
    tp = _ensure_col(t_probe)
    if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
        return None
    fb = fit_best(pf, tf, poly_degree)
    if fb is None:
        return None
    _, mapping = fb
    try:
        pf_hat = eval_mapping(pf, mapping)
        pp_hat = eval_mapping(pp, mapping)
    except Exception:
        return None
    fit_mse = _weighted_mse_cols(tf, pf_hat, w_fit)
    probe_mse = _weighted_mse_cols(tp, pp_hat, w_probe)
    if fit_mse is None or probe_mse is None:
        return None
    return float(fit_mse), float(probe_mse)


def _mapping_kind_lower(mapping: dict[str, Any] | None) -> str:
    if not isinstance(mapping, dict):
        return "identity"
    kind = mapping.get("kind", "identity")
    if kind is None:
        return "identity"
    return str(kind).strip().lower()


def _mapping_cache_signature(mapping: dict[str, Any] | None) -> tuple[Any, ...]:
    if not isinstance(mapping, dict):
        return ("identity",)
    kind = _mapping_kind_lower(mapping)

    def _norm(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            try:
                return tuple(float(x) for x in v.detach().cpu().reshape(-1).tolist())
            except Exception:
                return ("tensor", tuple(int(x) for x in v.shape))
        if isinstance(v, (list, tuple)):
            out = []
            for x in list(v)[:16]:
                try:
                    out.append(float(x))
                except Exception:
                    out.append(str(x))
            return tuple(out)
        try:
            return float(v)
        except Exception:
            return str(v)

    keys = ("a", "b", "c", "mu", "std", "log_a", "A", "B", "omega", "sf", "sy", "coeffs", "numer", "denom")
    items = [kind]
    for k in keys:
        if k in mapping:
            items.append((k, _norm(mapping.get(k))))
    head = mapping.get("_lin_head", None)
    if isinstance(head, dict):
        try:
            coeffs = _norm(head.get("coeffs", ()))
            items.append(("_lin_head", coeffs))
        except Exception:
            items.append(("_lin_head", "present"))
    return tuple(items)


def _fit_affine_mapping_from_pair(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    safe_eps: float = 1.0e-12,
) -> dict[str, Any] | None:
    p = _ensure_col(pred)
    t = _ensure_col(y)
    mask = _finite_mask(p, t).squeeze(-1)
    if int(mask.sum().item()) < 2:
        return None
    f = p[mask, 0]
    yy = t[mask, 0]
    A = torch.stack([torch.ones_like(f), f], dim=1)
    try:
        sol = torch.linalg.lstsq(A, yy).solution
    except Exception:
        try:
            gram = A.T @ A
            rhs = A.T @ yy
            eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            sol = torch.linalg.solve(gram + 1.0e-12 * eye, rhs)
        except Exception:
            return None
    if int(sol.numel()) < 2 or (not torch.isfinite(sol).all()):
        return None
    b = float(sol[0].item())
    a = float(sol[1].item())
    eps = float(max(1.0e-12, safe_eps))
    if (not math.isfinite(a)) or abs(a) <= eps or (not math.isfinite(b)):
        return None
    return {"kind": "affine", "a": float(a), "b": float(b)}


def _inverse_target_mode_rows(
    parent_node,
    parent_mapping,
    path: Sequence[int] | None,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    *,
    profile: dict[str, Any],
    safe_eps: float,
    confidence_mode: str,
    confidence_target_gain: float,
    confidence_floor: float,
    branch_beam_width: int,
    target_mode: str = "robust",
    full_mapping_penalty: float = 0.75,
    exact_simple_target_bonus: float = 0.10,
) -> list[dict[str, Any]]:
    tm = str(target_mode or "robust").strip().lower()
    if tm in ("", "auto"):
        tm = "robust"
    full_mapping = parent_mapping if isinstance(parent_mapping, dict) else {"kind": "identity"}
    mapping_complex = bool(profile.get("mapping_is_complex", False))
    exact_monotone = bool(profile.get("exact_monotone", False))

    affine_mapping = None
    if tm in ("robust", "simple", "identity_affine", "affine", "identity"):
        try:
            root_pred_fit = eval_node(parent_node, x_fit)
            affine_mapping = _fit_affine_mapping_from_pair(root_pred_fit, y_fit, safe_eps=float(safe_eps))
        except Exception:
            affine_mapping = None

    if tm in ("full", "legacy"):
        mode_specs = [("full", full_mapping)]
    elif tm in ("identity",):
        mode_specs = [("identity", {"kind": "identity"})]
    elif tm in ("affine",):
        mode_specs = [("affine", affine_mapping)]
    elif tm in ("simple", "identity_affine"):
        mode_specs = [("affine", affine_mapping), ("identity", {"kind": "identity"})]
    else:
        if exact_monotone and mapping_complex:
            mode_specs = [
                ("identity", {"kind": "identity"}),
                ("affine", affine_mapping),
                ("full", full_mapping),
            ]
        else:
            mode_specs = [
                ("full", full_mapping),
                ("affine", affine_mapping),
                ("identity", {"kind": "identity"}),
            ]

    out: list[dict[str, Any]] = []
    seen_sigs: set[tuple[Any, ...]] = set()
    for mode_name, mode_mapping in mode_specs:
        if mode_mapping is None:
            continue
        sig = _mapping_cache_signature(mode_mapping)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        mode_factor = 1.0
        if exact_monotone and mapping_complex:
            if mode_name == "full":
                try:
                    mode_factor *= max(0.0, float(full_mapping_penalty))
                except Exception:
                    mode_factor *= 0.75
            elif mode_name in ("identity", "affine"):
                try:
                    mode_factor *= (1.0 + max(0.0, float(exact_simple_target_bonus)))
                except Exception:
                    mode_factor *= 1.10
        try:
            inv_fit_list = invert_context_target_beam(
                parent_node,
                path,
                x_fit,
                y_fit,
                mapping=mode_mapping,
                safe_eps=float(safe_eps),
                allow_identity_fallback=True,
                confidence_mode=str(confidence_mode),
                confidence_target_gain=float(confidence_target_gain),
                confidence_floor=float(confidence_floor),
                branch_beam_width=int(branch_beam_width),
            )
            inv_probe_list = invert_context_target_beam(
                parent_node,
                path,
                x_probe,
                y_probe,
                mapping=mode_mapping,
                safe_eps=float(safe_eps),
                allow_identity_fallback=True,
                confidence_mode=str(confidence_mode),
                confidence_target_gain=float(confidence_target_gain),
                confidence_floor=float(confidence_floor),
                branch_beam_width=int(branch_beam_width),
            )
        except Exception:
            continue
        out.append(
            {
                "mode": str(mode_name),
                "mapping": mode_mapping,
                "mode_factor": float(mode_factor),
                "fit_list": list(inv_fit_list),
                "probe_list": list(inv_probe_list),
            }
        )
    return out
