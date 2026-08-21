# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Inverse steering candidate search and path diagnostics helpers."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .expr_ast import (
    BINARY_OPS,
    UNARY_OPS,
    collect_paths,
    dims_eq,
    eval_node,
    get_at,
    node_depth,
    node_dims,
    node_size,
    node_str,
    replace_at,
    simplify,
)
from .inverse_core import (
    _bool_col,
    _ensure_col,
    _estimate_path_transport_scores,
    _inverse_target_mode_rows,
    _mask_fraction,
    _masked_point_weight,
    _normalize_inverse_local_score_mode,
    _prepare_nonnegative_weights,
    _score_inverse_local_predictions,
    _slice_by_mask,
    _weighted_centered_mse,
    _weighted_mse_cols,
)
from .engine.signals import InverseSteeringPotential, PathStateFeatures

_INVERSE_STATIC_OP_WEIGHTS = {
    "add": 0.0,
    "sub": 0.0,
    "neg": 0.0,
    "mul": 1.0,
    "div": 1.1,
    "sqrt": 1.0,
    "sqr": 0.8,
    "exp": 1.2,
    "log": 1.2,
    "sin": 1.4,
    "cos": 1.4,
}


def _inverse_pool_shortlist(
    pool_phi_fit: torch.Tensor,
    target_fit: torch.Tensor,
    valid_mask_fit: torch.Tensor,
    *,
    pool_dims=None,
    target_dim=None,
    shortlist_k=16,
):
    if pool_phi_fit is None or pool_phi_fit.numel() <= 0:
        return []
    m = _bool_col(valid_mask_fit).squeeze(-1)
    if int(m.sum().item()) < 4:
        return []
    Phi = pool_phi_fit[m]
    t = _ensure_col(target_fit)[m, 0]
    t = t - t.mean()
    Phi = Phi - Phi.mean(dim=0, keepdim=True)
    norms = (Phi * Phi).sum(dim=0)
    scores = (t @ Phi) ** 2 / (norms + 1.0e-12)
    valid = torch.isfinite(scores) & torch.isfinite(norms) & (norms > 1.0e-12)
    if (pool_dims is not None) and (target_dim is not None):
        dim_valid = torch.tensor([
            (pool_dims[i] is not None) and dims_eq(pool_dims[i], target_dim)
            for i in range(len(pool_dims))
        ], dtype=torch.bool, device=scores.device)
        if int(dim_valid.numel()) == int(valid.numel()):
            valid = valid & dim_valid
    n_valid = int(valid.sum().item())
    if n_valid <= 0:
        return []
    scores = scores.masked_fill(~valid, float('-inf'))
    k = min(max(1, int(shortlist_k)), n_valid)
    return [int(i) for i in scores.topk(k).indices.tolist()]


def _weighted_linear_fit(
    X: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor | None:
    if X.ndim != 2:
        return None
    yy = y.reshape(-1)
    ww = w.reshape(-1)
    if int(X.shape[0]) != int(yy.shape[0]) or int(yy.shape[0]) != int(ww.shape[0]):
        return None
    m = torch.isfinite(X).all(dim=1) & torch.isfinite(yy) & torch.isfinite(ww) & (ww > 0.0)
    if int(m.sum().item()) < max(4, int(X.shape[1]) + 1):
        return None
    Xm = X[m]
    ym = yy[m]
    wm = ww[m]
    sw = torch.sqrt(torch.clamp(wm, min=0.0))
    Xw = Xm * sw.unsqueeze(1)
    yw = ym * sw
    try:
        gram = Xw.T @ Xw
        rhs = Xw.T @ yw
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        sol = torch.linalg.solve(gram + 1.0e-10 * eye, rhs)
    except Exception:
        return None
    if not torch.isfinite(sol).all():
        return None
    return sol


def _quantize_monomial_exponent(alpha: float) -> float:
    if not math.isfinite(alpha):
        return 0.0
    if abs(alpha) < 0.35:
        return 0.0
    allowed = (-2.0, -1.0, 1.0, 2.0)
    return float(min(allowed, key=lambda a: abs(float(alpha) - a)))


def _node_pow_small_int(base_node: tuple, exp_q: float) -> tuple | None:
    e = float(exp_q)
    if e == 1.0:
        return base_node
    if e == 2.0:
        return ("sqr", base_node)
    if e == -1.0:
        return ("div", ("const", 1.0), base_node)
    if e == -2.0:
        return ("div", ("const", 1.0), ("sqr", base_node))
    return None


def _eval_quantized_monomial_from_pool(
    phi: torch.Tensor,
    factors: list[tuple[int, float]],
    *,
    scale: float,
    safe_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if phi.ndim != 2:
        raise ValueError("phi must be 2D [N,K]")
    n = int(phi.shape[0])
    out = torch.ones((n, 1), dtype=phi.dtype, device=phi.device)
    valid = torch.ones((n, 1), dtype=torch.bool, device=phi.device)
    for idx, exp_q in factors:
        u = phi[:, int(idx):int(idx) + 1]
        if float(exp_q) == 1.0:
            fac = u
            fac_ok = torch.isfinite(fac)
        elif float(exp_q) == 2.0:
            fac = u * u
            fac_ok = torch.isfinite(fac)
        elif float(exp_q) == -1.0:
            den_ok = torch.isfinite(u) & (u.abs() > float(safe_eps))
            fac = 1.0 / torch.where(den_ok, u, torch.ones_like(u))
            fac_ok = torch.isfinite(fac) & den_ok
        elif float(exp_q) == -2.0:
            den_ok = torch.isfinite(u) & (u.abs() > float(safe_eps))
            fac = 1.0 / torch.where(den_ok, u * u, torch.ones_like(u))
            fac_ok = torch.isfinite(fac) & den_ok
        else:
            fac = torch.ones_like(u)
            fac_ok = torch.zeros_like(valid)
        out = out * fac
        valid = valid & fac_ok
    sc = float(scale)
    if not math.isfinite(sc):
        sc = 1.0
    out = out * sc
    valid = valid & torch.isfinite(out)
    out = torch.where(valid, out, torch.zeros_like(out))
    return out, valid


@torch.no_grad()
def _inverse_muldiv_monomial_candidates(
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    candidate_indices: Sequence[int],
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    *,
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
    poly_degree: int = 2,
    local_score_mode: str = "affine",
    topk_terms: int = 8,
    max_pair_terms: int = 4,
    max_out: int = 8,
    safe_eps: float = 1.0e-8,
) -> list[tuple]:
    tf = _ensure_col(t_fit)
    tp = _ensure_col(t_probe)
    wf = _prepare_nonnegative_weights(w_fit, tf)
    wp = _prepare_nonnegative_weights(w_probe, tp)
    if pool_phi_fit is None or pool_phi_probe is None:
        return []
    if pool_phi_fit.ndim != 2 or pool_phi_probe.ndim != 2:
        return []
    if int(pool_phi_fit.shape[0]) != int(tf.shape[0]) or int(pool_phi_probe.shape[0]) != int(tp.shape[0]):
        return []
    if int(pool_phi_fit.shape[1]) <= 0 or int(pool_phi_probe.shape[1]) <= 0:
        return []

    idxs = []
    seen_idx = set()
    kmax = int(pool_phi_fit.shape[1])
    for idx in candidate_indices:
        try:
            ii = int(idx)
        except Exception:
            continue
        if ii < 0 or ii >= kmax or ii in seen_idx:
            continue
        seen_idx.add(ii)
        idxs.append(ii)
    if not idxs:
        return []

    eps = float(max(1.0e-12, safe_eps))
    yfit = tf[:, 0]
    ymask = torch.isfinite(yfit) & torch.isfinite(wf[:, 0]) & (wf[:, 0] > 0.0) & (yfit.abs() > eps)
    if int(ymask.sum().item()) < 8:
        return []
    ylog = torch.log(torch.clamp(yfit.abs(), min=eps))

    single_rows = []
    for idx in idxs:
        ufit = pool_phi_fit[:, idx]
        umask = ymask & torch.isfinite(ufit) & (ufit.abs() > eps)
        if int(umask.sum().item()) < 8:
            continue
        x = torch.log(torch.clamp(ufit.abs(), min=eps))
        X = torch.stack([torch.ones_like(x), x], dim=1)
        sol = _weighted_linear_fit(X[umask], ylog[umask], wf[umask, 0])
        if sol is None:
            continue
        c = float(sol[0].item())
        alpha = float(sol[1].item())
        aq = _quantize_monomial_exponent(alpha)
        if aq == 0.0:
            continue
        yhat = X[umask] @ sol
        sse = _weighted_mse_cols(ylog[umask].unsqueeze(-1), yhat.unsqueeze(-1), wf[umask])
        if sse is None:
            continue
        single_rows.append((float(sse), int(idx), float(c), float(aq)))
    if not single_rows:
        return []
    single_rows.sort(key=lambda row: row[0])
    seed_rows = single_rows[: max(1, int(topk_terms))]

    proposals: list[tuple[float, float, tuple]] = []
    seen_nodes: set[str] = set()

    def _maybe_add_candidate(factors: list[tuple[int, float]], intercept_c: float):
        ff = [(int(i), float(e)) for (i, e) in factors if float(e) != 0.0]
        if not ff:
            return
        ast_factors = []
        for idx, eq in ff:
            base_node = pool_nodes[int(idx)]
            fac_node = _node_pow_small_int(base_node, float(eq))
            if fac_node is None:
                return
            ast_factors.append(fac_node)
        node = ast_factors[0]
        for fac in ast_factors[1:]:
            node = ("mul", node, fac)
        c = float(intercept_c)
        if not math.isfinite(c):
            c = 0.0
        scale = math.exp(max(-4.0, min(4.0, c)))
        if abs(scale - 1.0) > 1.0e-6:
            node = ("mul", ("const", float(scale)), node)
        try:
            node = simplify(node)
        except Exception:
            pass
        key = node_str(node)
        if key in seen_nodes:
            return

        pf, vfit = _eval_quantized_monomial_from_pool(
            pool_phi_fit,
            ff,
            scale=scale,
            safe_eps=eps,
        )
        pp, vprobe = _eval_quantized_monomial_from_pool(
            pool_phi_probe,
            ff,
            scale=scale,
            safe_eps=eps,
        )
        wf_eff = wf * vfit.to(dtype=wf.dtype)
        wp_eff = wp * vprobe.to(dtype=wp.dtype)
        sc = _score_inverse_local_predictions(
            pf,
            pp,
            tf,
            tp,
            w_fit=wf_eff,
            w_probe=wp_eff,
            poly_degree=int(poly_degree),
            mode=str(local_score_mode),
        )
        if sc is None:
            return
        fit_mse, probe_mse = sc
        if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
            return
        seen_nodes.add(key)
        proposals.append((float(probe_mse), float(fit_mse), node))

    for _sse, idx, c, aq in seed_rows:
        _maybe_add_candidate([(idx, aq)], c)

    pair_pool = seed_rows[: max(2, min(len(seed_rows), int(max_pair_terms)))]
    for i in range(len(pair_pool)):
        for j in range(i + 1, len(pair_pool)):
            idx1 = int(pair_pool[i][1])
            idx2 = int(pair_pool[j][1])
            u1 = pool_phi_fit[:, idx1]
            u2 = pool_phi_fit[:, idx2]
            mm = ymask & torch.isfinite(u1) & torch.isfinite(u2) & (u1.abs() > eps) & (u2.abs() > eps)
            if int(mm.sum().item()) < 10:
                continue
            x1 = torch.log(torch.clamp(u1.abs(), min=eps))
            x2 = torch.log(torch.clamp(u2.abs(), min=eps))
            X = torch.stack([torch.ones_like(x1), x1, x2], dim=1)
            sol = _weighted_linear_fit(X[mm], ylog[mm], wf[mm, 0])
            if sol is None:
                continue
            c = float(sol[0].item())
            a1 = _quantize_monomial_exponent(float(sol[1].item()))
            a2 = _quantize_monomial_exponent(float(sol[2].item()))
            if a1 == 0.0 and a2 == 0.0:
                continue
            _maybe_add_candidate([(idx1, a1), (idx2, a2)], c)

    if not proposals:
        return []
    proposals.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
    return [row[2] for row in proposals[: max(1, int(max_out))]]


@torch.no_grad()
def _inverse_additive_combo_candidates(
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    candidate_indices: Sequence[int],
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    *,
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
    poly_degree: int = 2,
    local_score_mode: str = "affine",
    topk_terms: int = 8,
    max_pair_terms: int = 5,
    max_out: int = 8,
) -> list[tuple]:
    tf = _ensure_col(t_fit)
    tp = _ensure_col(t_probe)
    wf = _prepare_nonnegative_weights(w_fit, tf)
    wp = _prepare_nonnegative_weights(w_probe, tp)
    if pool_phi_fit is None or pool_phi_probe is None:
        return []
    if pool_phi_fit.ndim != 2 or pool_phi_probe.ndim != 2:
        return []
    if int(pool_phi_fit.shape[0]) != int(tf.shape[0]) or int(pool_phi_probe.shape[0]) != int(tp.shape[0]):
        return []

    idxs = []
    seen_idx = set()
    kmax = int(pool_phi_fit.shape[1])
    for idx in candidate_indices:
        try:
            ii = int(idx)
        except Exception:
            continue
        if ii < 0 or ii >= kmax or ii in seen_idx:
            continue
        seen_idx.add(ii)
        idxs.append(ii)
    if not idxs:
        return []

    proposals: list[tuple[float, float, tuple, torch.Tensor, torch.Tensor]] = []
    seen_nodes = set()

    def _add(node: tuple, pf: torch.Tensor, pp: torch.Tensor):
        sc = _score_inverse_local_predictions(
            pf,
            pp,
            tf,
            tp,
            w_fit=wf,
            w_probe=wp,
            poly_degree=int(poly_degree),
            mode=str(local_score_mode),
        )
        if sc is None:
            return
        fit_mse, probe_mse = sc
        if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
            return
        key = node_str(node)
        if key in seen_nodes:
            return
        seen_nodes.add(key)
        proposals.append((float(probe_mse), float(fit_mse), node, pf, pp))

    for idx in idxs:
        base = pool_nodes[int(idx)]
        pf = pool_phi_fit[:, int(idx):int(idx) + 1]
        pp = pool_phi_probe[:, int(idx):int(idx) + 1]
        if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
            continue
        _add(base, pf, pp)
        _add(("neg", base), -pf, -pp)

    if not proposals:
        return []

    proposals.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
    seeds = proposals[: max(2, min(len(proposals), int(topk_terms)))]
    pair_pool = seeds[: max(2, min(len(seeds), int(max_pair_terms)))]
    for i in range(len(pair_pool)):
        _, _, ni, pfi, ppi = pair_pool[i]
        for j in range(i + 1, len(pair_pool)):
            _, _, nj, pfj, ppj = pair_pool[j]
            _add(("add", ni, nj), pfi + pfj, ppi + ppj)
            _add(("sub", ni, nj), pfi - pfj, ppi - ppj)
            _add(("sub", nj, ni), pfj - pfi, ppj - ppi)

    proposals.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
    return [row[2] for row in proposals[: max(1, int(max_out))]]


@torch.no_grad()
def _inverse_sqrt_quadratic_candidates(
    pool_nodes,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    candidate_indices: Sequence[int],
    t_fit: torch.Tensor,
    t_probe: torch.Tensor,
    *,
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
    poly_degree: int = 2,
    local_score_mode: str = "affine",
    topk_terms: int = 8,
    max_pair_terms: int = 5,
    max_out: int = 8,
) -> list[tuple]:
    tf = _ensure_col(t_fit)
    tp = _ensure_col(t_probe)
    wf = _prepare_nonnegative_weights(w_fit, tf)
    wp = _prepare_nonnegative_weights(w_probe, tp)
    if pool_phi_fit is None or pool_phi_probe is None:
        return []
    if pool_phi_fit.ndim != 2 or pool_phi_probe.ndim != 2:
        return []
    if int(pool_phi_fit.shape[0]) != int(tf.shape[0]) or int(pool_phi_probe.shape[0]) != int(tp.shape[0]):
        return []

    idxs = []
    seen_idx = set()
    kmax = int(pool_phi_fit.shape[1])
    for idx in candidate_indices:
        try:
            ii = int(idx)
        except Exception:
            continue
        if ii < 0 or ii >= kmax or ii in seen_idx:
            continue
        seen_idx.add(ii)
        idxs.append(ii)
    if not idxs:
        return []

    proposals: list[tuple[float, float, tuple, torch.Tensor, torch.Tensor]] = []
    seen_nodes = set()

    def _add(node: tuple, pf: torch.Tensor, pp: torch.Tensor):
        sc = _score_inverse_local_predictions(
            pf,
            pp,
            tf,
            tp,
            w_fit=wf,
            w_probe=wp,
            poly_degree=int(poly_degree),
            mode=str(local_score_mode),
        )
        if sc is None:
            return
        fit_mse, probe_mse = sc
        if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
            return
        key = node_str(node)
        if key in seen_nodes:
            return
        seen_nodes.add(key)
        proposals.append((float(probe_mse), float(fit_mse), node, pf, pp))

    for idx in idxs:
        base = pool_nodes[int(idx)]
        pf = pool_phi_fit[:, int(idx):int(idx) + 1]
        pp = pool_phi_probe[:, int(idx):int(idx) + 1]
        if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
            continue
        _add(("sqr", base), pf * pf, pp * pp)
    base_cross = idxs[: max(2, min(len(idxs), int(max_pair_terms)))]
    for i in range(len(base_cross)):
        ii = int(base_cross[i])
        ni = pool_nodes[ii]
        pfi = pool_phi_fit[:, ii:ii + 1]
        ppi = pool_phi_probe[:, ii:ii + 1]
        for j in range(i + 1, len(base_cross)):
            jj = int(base_cross[j])
            nj = pool_nodes[jj]
            pfj = pool_phi_fit[:, jj:jj + 1]
            ppj = pool_phi_probe[:, jj:jj + 1]
            if (not torch.isfinite(pfi).all()) or (not torch.isfinite(ppi).all()):
                continue
            if (not torch.isfinite(pfj).all()) or (not torch.isfinite(ppj).all()):
                continue
            _add(("mul", ni, nj), pfi * pfj, ppi * ppj)

    atom_seed = sorted(proposals, key=lambda row: (row[0], row[1]))[: max(2, min(len(proposals), int(topk_terms)))]
    pair_pool = atom_seed[: max(2, min(len(atom_seed), int(max_pair_terms)))]
    for i in range(len(pair_pool)):
        _, _, ni, pfi, ppi = pair_pool[i]
        for j in range(i + 1, len(pair_pool)):
            _, _, nj, pfj, ppj = pair_pool[j]
            _add(("add", ni, nj), pfi + pfj, ppi + ppj)
            _add(("sub", ni, nj), pfi - pfj, ppi - ppj)
            _add(("sub", nj, ni), pfj - pfi, ppj - ppi)

    proposals.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
    return [row[2] for row in proposals[: max(1, int(max_out))]]


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


def _pool_cache_signature(pool_nodes: Sequence[tuple] | None) -> tuple[Any, ...]:
    if not pool_nodes:
        return (0,)
    n = int(len(pool_nodes))
    head = tuple(node_str(pool_nodes[i]) for i in range(min(4, n)))
    tail = tuple(node_str(pool_nodes[n - i - 1]) for i in range(min(4, n)))
    return (n, head, tail)


def _inverse_mapping_static_weight(mapping: dict[str, Any] | None) -> float:
    kind = _mapping_kind_lower(mapping)
    if kind in ("", "identity", "affine", "mono", "monomial"):
        return 0.0
    if kind == "poly":
        coeffs = None if not isinstance(mapping, dict) else mapping.get("coeffs", None)
        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.detach().cpu().tolist()
        deg = max(0, len(coeffs) - 1) if isinstance(coeffs, (list, tuple)) else 0
        return 0.5 if deg >= 2 else 0.0
    if kind == "power":
        return 0.8
    if kind == "exp":
        return 1.0
    if kind == "sine":
        return 1.1
    if kind == "pade":
        return 0.8
    return 0.35


def _inverse_static_path_score(
    parent_node,
    path: Sequence[int] | None,
    mapping: dict[str, Any] | None = None,
) -> tuple[float, int]:
    pp = tuple(int(v) for v in (path or ()))
    score = float(_inverse_mapping_static_weight(mapping))
    nonadditive = 1 if score >= 0.75 else 0
    cur = parent_node
    for slot in pp:
        op = str(cur[0])
        w = float(_INVERSE_STATIC_OP_WEIGHTS.get(op, 0.25 if op in UNARY_OPS or op in BINARY_OPS else 0.0))
        score += w
        if w >= 0.75:
            nonadditive += 1
        try:
            cur = cur[int(slot)]
        except Exception:
            break
    try:
        sub = get_at(parent_node, pp)
        score += 0.10 * max(0, len(pp) - 1)
        score += 0.05 * max(0, min(node_size(sub), 8) - 1)
        if sub[0] in ("var", "const", "hparam"):
            score -= 0.10
    except Exception:
        pass
    return float(score), int(nonadditive)


def _inverse_path_profile(
    parent_node,
    path: Sequence[int] | None,
    mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pp = tuple(int(v) for v in (path or ()))
    ops = []
    cur = parent_node
    for slot in pp:
        try:
            op = str(cur[0])
            ops.append(op)
            cur = cur[int(slot)]
        except Exception:
            break
    mk = _mapping_kind_lower(mapping)
    has_periodic = any(op in ("sin", "cos") for op in ops) or (mk == "sine")
    has_muldiv = any(op in ("mul", "div") for op in ops) or (mk in ("power", "pade"))
    has_explogsqrt = any(op in ("exp", "log", "sqrt", "sqr") for op in ops) or (mk == "exp")
    has_ambiguous_inverse = any(op in ("sin", "cos", "sqr") for op in ops) or (mk == "sine")
    exact_monotone_ops = {"add", "sub", "mul", "div", "neg", "sqrt", "exp", "log"}
    exact_monotone = (not has_periodic) and all(op in exact_monotone_ops for op in ops)
    last_additive_prefix_len = -1
    for i, op in enumerate(ops):
        if op in ("add", "sub"):
            last_additive_prefix_len = i
    cut_parent_op = str(ops[-1]) if ops else ""
    mapping_is_complex = mk not in ("", "identity", "affine", "mono", "monomial")
    return {
        "ops": tuple(ops),
        "mapping_kind": mk,
        "mapping_is_complex": bool(mapping_is_complex),
        "has_periodic": bool(has_periodic),
        "has_muldiv": bool(has_muldiv),
        "has_explogsqrt": bool(has_explogsqrt),
        "has_ambiguous_inverse": bool(has_ambiguous_inverse),
        "exact_monotone": bool(exact_monotone),
        "last_additive_prefix_len": int(last_additive_prefix_len),
        "cut_parent_op": cut_parent_op,
    }


def _inverse_effective_thresholds(
    min_valid_frac: float,
    min_confidence: float,
    *,
    profile: dict[str, Any],
    periodic_min_valid_scale: float = 1.25,
    periodic_min_confidence_scale: float = 1.35,
) -> tuple[float, float]:
    try:
        mv = float(min_valid_frac)
    except Exception:
        mv = 0.25
    try:
        mc = float(min_confidence)
    except Exception:
        mc = 0.10
    mv = min(1.0, max(0.0, mv))
    mc = min(1.0, max(0.0, mc))
    if not bool(profile.get("has_periodic", False)):
        return float(mv), float(mc)
    try:
        vsc = float(periodic_min_valid_scale)
    except Exception:
        vsc = 1.25
    try:
        csc = float(periodic_min_confidence_scale)
    except Exception:
        csc = 1.35
    vsc = max(0.0, vsc)
    csc = max(0.0, csc)
    return float(min(1.0, mv * vsc)), float(min(1.0, mc * csc))


def _inverse_family_gain_scale(
    profile: dict[str, Any],
    *,
    periodic_path_penalty: float = 0.65,
    nonperiodic_muldiv_bonus: float = 0.10,
    nonperiodic_explogsqrt_bonus: float = 0.05,
) -> float:
    if bool(profile.get("has_periodic", False)):
        try:
            pp = float(periodic_path_penalty)
        except Exception:
            pp = 0.65
        return float(max(0.0, pp))
    scale = 1.0
    if bool(profile.get("has_muldiv", False)):
        try:
            scale *= (1.0 + max(0.0, float(nonperiodic_muldiv_bonus)))
        except Exception:
            scale *= 1.10
    if bool(profile.get("has_explogsqrt", False)):
        try:
            scale *= (1.0 + max(0.0, float(nonperiodic_explogsqrt_bonus)))
        except Exception:
            scale *= 1.05
    return float(max(0.0, scale))


def _inverse_branch_beam_factor(
    branch_rows: Sequence[dict[str, Any]],
    *,
    ambiguity_penalty: float = 0.50,
) -> tuple[float, float, int]:
    vals = []
    for row in branch_rows:
        try:
            v = float(row.get("weighted_rel_gain_raw", row.get("weighted_rel_gain", 0.0)))
        except Exception:
            v = 0.0
        if math.isfinite(v) and v > 0.0:
            vals.append(v)
    if len(vals) <= 1:
        return 1.0, 1.0, len(vals)
    vals.sort(reverse=True)
    best = float(vals[0])
    total = float(sum(vals))
    if (not math.isfinite(total)) or total <= 1.0e-12:
        return 1.0, 1.0, len(vals)
    dominance = best / max(total, 1.0e-12)
    second = float(vals[1]) if len(vals) >= 2 else 0.0
    margin = max(0.0, (best - second) / max(best, 1.0e-12))
    support = max(0.0, min(1.0, 0.5 * dominance + 0.5 * margin))
    try:
        ap = float(ambiguity_penalty)
    except Exception:
        ap = 0.50
    ap = max(0.0, min(1.0, ap))
    factor = (1.0 - ap) + ap * support
    return float(max(0.0, factor)), float(support), len(vals)


def _inverse_effective_branch_beam_width(
    profile: dict[str, Any],
    base_branch_beam_width: int,
) -> int:
    try:
        bw = max(1, int(base_branch_beam_width))
    except Exception:
        bw = 1
    if bool(profile.get("has_ambiguous_inverse", False)):
        return max(2, bw)
    # Monotone / branch-safe contexts do not need wide inverse beams.
    return 1


def _inverse_path_cut_factor(
    parent_node,
    path: Sequence[int] | None,
    profile: dict[str, Any],
    *,
    additive_descend_penalty: float = 0.15,
    nonadditive_leaf_penalty: float = 0.20,
) -> float:
    pp = tuple(int(v) for v in (path or ()))
    factor = 1.0
    try:
        add_pen = float(additive_descend_penalty)
    except Exception:
        add_pen = 0.15
    add_pen = min(0.95, max(0.0, add_pen))
    try:
        leaf_pen = float(nonadditive_leaf_penalty)
    except Exception:
        leaf_pen = 0.20
    leaf_pen = min(0.95, max(0.0, leaf_pen))

    last_additive_prefix_len = int(profile.get("last_additive_prefix_len", -1))
    if last_additive_prefix_len >= 0:
        descend_below_add = max(0, len(pp) - last_additive_prefix_len)
        if descend_below_add > 0:
            factor *= (1.0 - add_pen) ** int(descend_below_add)

    try:
        sub = get_at(parent_node, pp)
        cut_is_leaf = str(sub[0]) in ("var", "const", "hparam")
    except Exception:
        cut_is_leaf = False
    cut_parent_op = str(profile.get("cut_parent_op", ""))
    if cut_is_leaf:
        if cut_parent_op in ("sqr", "sqrt", "exp", "log", "sin", "cos"):
            factor *= (1.0 - leaf_pen)
        elif cut_parent_op in ("mul", "div"):
            factor *= (1.0 - 0.5 * leaf_pen)
    return float(max(0.0, factor))


def _deterministic_row_subset(cap: int | None, *xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if not xs:
        return tuple()
    if cap is None:
        return tuple(xs)
    try:
        cap_i = int(cap)
    except Exception:
        cap_i = 0
    if cap_i <= 0:
        return tuple(xs)
    n = int(xs[0].shape[0])
    if n <= cap_i:
        return tuple(xs)
    idx = torch.linspace(0, n - 1, steps=cap_i, device=xs[0].device)
    idx = torch.round(idx).to(dtype=torch.long)
    return tuple(x.index_select(0, idx) for x in xs)


@torch.no_grad()
def estimate_inverse_steering_potential(
    parent_node,
    parent_mapping,
    x_fit,
    y_fit,
    x_probe,
    y_probe,
    pool_phi_fit,
    pool_phi_probe,
    pool_dims,
    pool_nodes=None,
    *,
    var_dims=None,
    max_paths=6,
    topk_terms=4,
    shortlist_mult=2,
    min_valid_frac=0.25,
    min_confidence=0.10,
    min_structural_score=0.75,
    min_weighted_rel_gain=0.05,
    structural_bias=0.20,
    safe_eps=1.0e-12,
    confidence_mode="conditioning",
    confidence_target_gain=4.0,
    confidence_floor=0.05,
    branch_beam_width=1,
    local_score_mode="affine",
    target_mode="robust",
    full_mapping_penalty=0.75,
    exact_simple_target_bonus=0.10,
    additive_descend_penalty=0.15,
    nonadditive_leaf_penalty=0.20,
    periodic_min_valid_scale=1.25,
    periodic_min_confidence_scale=1.35,
    periodic_path_penalty=0.65,
    nonperiodic_muldiv_bonus=0.10,
    nonperiodic_explogsqrt_bonus=0.05,
    branch_ambiguity_penalty=0.50,
    fit_cap=16,
    probe_cap=32,
):
    dm = var_dims is not None
    x_fit, y_fit, pool_phi_fit = _deterministic_row_subset(fit_cap, x_fit, y_fit, pool_phi_fit)
    x_probe, y_probe, pool_phi_probe = _deterministic_row_subset(probe_cap, x_probe, y_probe, pool_phi_probe)

    all_paths = [p for p in collect_paths(parent_node) if p]
    if not all_paths:
        return InverseSteeringPotential(
            allowed=False,
            reason="no_paths",
            best_path=None,
            best_rel_gain=0.0,
            best_weighted_rel_gain=0.0,
            candidate_paths=(),
            path_rows=(),
        )

    # Path-conditioned residual transport signal (symbolic backprop influence).
    transport_scores_probe: dict[tuple[int, ...], float] = {}
    try:
        transport_scores_probe, _transport_adj_probe, _transport_out_probe, _transport_res_probe = _estimate_path_transport_scores(
            parent_node,
            parent_mapping,
            x_probe,
            y_probe,
            all_paths,
            safe_eps=float(safe_eps),
        )
    except Exception:
        transport_scores_probe = {}
    if transport_scores_probe:
        vals = [float(v) for v in transport_scores_probe.values() if math.isfinite(float(v))]
        transport_max = max(vals) if vals else 0.0
    else:
        transport_max = 0.0
    transport_den = max(1.0e-18, float(transport_max))

    scored_paths = []
    for path in all_paths:
        try:
            sub = get_at(parent_node, path)
        except Exception:
            continue
        static_score, nonadditive = _inverse_static_path_score(parent_node, path, parent_mapping)
        if static_score < float(min_structural_score):
            continue
        transport_rel = max(0.0, float(transport_scores_probe.get(tuple(path), 0.0)) / transport_den)
        scored_paths.append((
            float(static_score),
            float(transport_rel),
            int(nonadditive),
            len(path),
            -node_size(sub),
            tuple(path),
        ))

    if not scored_paths:
        return InverseSteeringPotential(
            allowed=False,
            reason="no_structural_paths",
            best_path=None,
            best_rel_gain=0.0,
            best_weighted_rel_gain=0.0,
            candidate_paths=(),
            path_rows=(),
        )

    scored_paths.sort(reverse=True)
    try:
        max_paths_i = max(1, int(max_paths))
    except Exception:
        max_paths_i = 6
    try:
        topk_terms_i = max(1, int(topk_terms))
    except Exception:
        topk_terms_i = 4
    try:
        shortlist_mult_i = max(1, int(shortlist_mult))
    except Exception:
        shortlist_mult_i = 2
    local_mode = _normalize_inverse_local_score_mode(local_score_mode, default="affine")

    candidate_paths = [row[-1] for row in scored_paths[:max_paths_i]]
    best_path = None
    best_rel_gain = 0.0
    best_weighted_rel_gain = 0.0
    path_rows = []

    for static_score, transport_rel, nonadditive, _depth, _neg_size, path in scored_paths[:max_paths_i]:
        try:
            sub = get_at(parent_node, path)
        except Exception:
            continue
        profile = _inverse_path_profile(parent_node, path, parent_mapping)
        min_valid_eff, min_conf_eff = _inverse_effective_thresholds(
            float(min_valid_frac),
            float(min_confidence),
            profile=profile,
            periodic_min_valid_scale=float(periodic_min_valid_scale),
            periodic_min_confidence_scale=float(periodic_min_confidence_scale),
        )
        family_scale = _inverse_family_gain_scale(
            profile,
            periodic_path_penalty=float(periodic_path_penalty),
            nonperiodic_muldiv_bonus=float(nonperiodic_muldiv_bonus),
            nonperiodic_explogsqrt_bonus=float(nonperiodic_explogsqrt_bonus),
        )
        transport_factor = 1.0 + 0.35 * max(0.0, float(transport_rel))
        path_beam_width = _inverse_effective_branch_beam_width(profile, int(branch_beam_width))
        cut_factor = _inverse_path_cut_factor(
            parent_node,
            path,
            profile,
            additive_descend_penalty=float(additive_descend_penalty),
            nonadditive_leaf_penalty=float(nonadditive_leaf_penalty),
        )
        target_mode_rows = _inverse_target_mode_rows(
            parent_node,
            parent_mapping,
            path,
            x_fit,
            y_fit,
            x_probe,
            y_probe,
            profile=profile,
            safe_eps=float(safe_eps),
            confidence_mode=str(confidence_mode),
            confidence_target_gain=float(confidence_target_gain),
            confidence_floor=float(confidence_floor),
            branch_beam_width=int(path_beam_width),
            target_mode=str(target_mode),
            full_mapping_penalty=float(full_mapping_penalty),
            exact_simple_target_bonus=float(exact_simple_target_bonus),
        )
        if not target_mode_rows:
            continue

        target_dim = node_dims(sub, var_dims) if dm else None
        mode_best_rows = []

        for mode_row in target_mode_rows:
            mode_name = str(mode_row.get("mode", "full"))
            mode_factor = float(mode_row.get("mode_factor", 1.0))
            inv_fit_list = list(mode_row.get("fit_list", []) or [])
            inv_probe_list = list(mode_row.get("probe_list", []) or [])
            probe_by_branch = {str(t.branch_id): t for t in inv_probe_list}
            branch_rows = []

            for inv_fit in inv_fit_list:
                inv_probe = probe_by_branch.get(str(inv_fit.branch_id), None)
                if inv_probe is None and inv_probe_list:
                    inv_probe = inv_probe_list[0]
                if inv_probe is None:
                    continue

                valid_frac = min(_mask_fraction(inv_fit.valid_mask), _mask_fraction(inv_probe.valid_mask))
                conf = min(float(inv_fit.confidence), float(inv_probe.confidence))
                if valid_frac < float(min_valid_eff) or conf < float(min_conf_eff):
                    continue

                mfit = _bool_col(inv_fit.valid_mask).squeeze(-1)
                mprobe = _bool_col(inv_probe.valid_mask).squeeze(-1)
                if int(mfit.sum().item()) < 4 or int(mprobe.sum().item()) < 4:
                    continue

                xf, tf = _slice_by_mask(x_fit, inv_fit.target, inv_fit.valid_mask)
                xp, tp = _slice_by_mask(x_probe, inv_probe.target, inv_probe.valid_mask)
                if int(xf.shape[0]) < 4 or int(xp.shape[0]) < 4:
                    continue
                wf = _masked_point_weight(
                    inv_fit.point_weight,
                    inv_fit.valid_mask,
                    dtype=tf.dtype,
                    device=tf.device,
                )
                wp = _masked_point_weight(
                    inv_probe.point_weight,
                    inv_probe.valid_mask,
                    dtype=tp.dtype,
                    device=tp.device,
                )

                try:
                    cur_pf = eval_node(sub, xf)
                    cur_pp = eval_node(sub, xp)
                except Exception:
                    continue
                cur_stats = _score_inverse_local_predictions(
                    cur_pf,
                    cur_pp,
                    tf,
                    tp,
                    w_fit=wf,
                    w_probe=wp,
                    poly_degree=2,
                    mode=local_mode,
                )
                if cur_stats is None:
                    cur_probe_mse = _weighted_centered_mse(tp, wp)
                else:
                    cur_probe_mse = float(cur_stats[1])
                if (not math.isfinite(cur_probe_mse)) or cur_probe_mse <= 0.0:
                    continue

                idxs = _inverse_pool_shortlist(
                    pool_phi_fit,
                    inv_fit.target,
                    inv_fit.valid_mask,
                    pool_dims=pool_dims if dm else None,
                    target_dim=target_dim,
                    shortlist_k=max(topk_terms_i, topk_terms_i * shortlist_mult_i),
                )
                if not idxs:
                    continue

                best_here = float('inf')
                for idx in idxs[: max(1, min(len(idxs), topk_terms_i * 2))]:
                    try:
                        cand_pf = pool_phi_fit[mfit, idx:idx + 1]
                        cand_pp = pool_phi_probe[mprobe, idx:idx + 1]
                    except Exception:
                        continue
                    cand_stats = _score_inverse_local_predictions(
                        cand_pf,
                        cand_pp,
                        tf,
                        tp,
                        w_fit=wf,
                        w_probe=wp,
                        poly_degree=2,
                        mode=local_mode,
                    )
                    if cand_stats is None:
                        continue
                    best_here = min(best_here, float(cand_stats[1]))

                if pool_nodes is not None and len(pool_nodes) > 0:
                    family_cands = _inverse_collect_local_repair_candidates(
                        parent_node=parent_node,
                        path=path,
                        sub=sub,
                        target_dim=target_dim,
                        xf=xf,
                        tf=tf,
                        xp=xp,
                        tp=tp,
                        wf=wf,
                        wp=wp,
                        mfit=mfit,
                        mprobe=mprobe,
                        pool_nodes=pool_nodes,
                        pool_dims=pool_dims,
                        pool_phi_fit=pool_phi_fit,
                        pool_phi_probe=pool_phi_probe,
                        idxs=idxs,
                        poly_degree=2,
                        local_mode=local_mode,
                        topk_terms=max(2, int(topk_terms_i)),
                        shortlist_mult=max(1, int(shortlist_mult_i)),
                        safe_eps=float(safe_eps),
                        var_dims=var_dims if dm else None,
                        max_depth=None,
                        micro_search_enable=False,
                    )
                    if family_cands:
                        fam_rows = _inverse_rank_local_repair_candidates(
                            family_cands,
                            xf=xf,
                            tf=tf,
                            xp=xp,
                            tp=tp,
                            wf=wf,
                            wp=wp,
                            poly_degree=2,
                            local_mode=local_mode,
                        )
                        if fam_rows:
                            best_here = min(best_here, float(fam_rows[0][0]))

                if not math.isfinite(best_here):
                    continue

                rel_gain = max(0.0, cur_probe_mse - best_here) / max(cur_probe_mse, 1.0e-12)
                weighted_rel_gain = rel_gain * max(0.0, conf) * max(0.0, valid_frac)
                weighted_rel_gain *= (1.0 + float(structural_bias) * max(0.0, float(static_score)))
                weighted_rel_gain *= float(transport_factor)
                weighted_rel_gain *= float(mode_factor)
                row = {
                    "path": tuple(path),
                    "branch_id": str(inv_fit.branch_id),
                    "static_score": float(static_score),
                    "transport_rel": float(transport_rel),
                    "transport_factor": float(transport_factor),
                    "nonadditive": int(nonadditive),
                    "valid_frac": float(valid_frac),
                    "confidence": float(conf),
                    "cur_probe_mse": float(cur_probe_mse),
                    "best_alt_probe_mse": float(best_here),
                    "rel_gain": float(rel_gain),
                    "weighted_rel_gain_raw": float(weighted_rel_gain),
                    "target_mode": mode_name,
                    "target_mode_factor": float(mode_factor),
                    "target_mapping_kind": str((mode_row.get("mapping") or {}).get("kind", "identity")),
                }
                branch_rows.append(row)

            if not branch_rows:
                continue
            branch_rows.sort(
                key=lambda row: (float(row.get("weighted_rel_gain_raw", 0.0)), float(row.get("rel_gain", 0.0))),
                reverse=True,
            )
            mode_best_row = dict(branch_rows[0])
            if bool(profile.get("has_ambiguous_inverse", False)):
                branch_factor, branch_support, branch_positive = _inverse_branch_beam_factor(
                    branch_rows,
                    ambiguity_penalty=float(branch_ambiguity_penalty),
                )
            else:
                branch_factor, branch_support, branch_positive = 1.0, 1.0, 1
            wrg_raw = float(mode_best_row.get("weighted_rel_gain_raw", 0.0))
            wrg = wrg_raw * float(family_scale) * float(branch_factor)
            mode_best_row["weighted_rel_gain"] = float(wrg)
            mode_best_row["branch_factor"] = float(branch_factor)
            mode_best_row["branch_support"] = float(branch_support)
            mode_best_row["branch_positive_count"] = int(branch_positive)
            mode_best_row["family_scale"] = float(family_scale)
            mode_best_row["min_valid_frac_eff"] = float(min_valid_eff)
            mode_best_row["min_confidence_eff"] = float(min_conf_eff)
            mode_best_row["profile_has_periodic"] = bool(profile.get("has_periodic", False))
            mode_best_row["profile_has_muldiv"] = bool(profile.get("has_muldiv", False))
            mode_best_row["profile_has_explogsqrt"] = bool(profile.get("has_explogsqrt", False))
            mode_best_row["profile_exact_monotone"] = bool(profile.get("exact_monotone", False))
            mode_best_row["transport_rel"] = float(transport_rel)
            mode_best_row["transport_factor"] = float(transport_factor)
            mode_best_rows.append(mode_best_row)

        if not mode_best_rows:
            continue
        mode_best_rows.sort(
            key=lambda row: (float(row.get("weighted_rel_gain", 0.0)), float(row.get("rel_gain", 0.0))),
            reverse=True,
        )
        path_best_row = dict(mode_best_rows[0])
        wrg_pre_cut = float(path_best_row.get("weighted_rel_gain", 0.0))
        wrg = wrg_pre_cut * float(cut_factor)
        path_best_row["weighted_rel_gain_pre_cut"] = float(wrg_pre_cut)
        path_best_row["weighted_rel_gain"] = float(wrg)
        path_best_row["cut_factor"] = float(cut_factor)
        path_best_row["mode_rows"] = [
            {
                "target_mode": str(mr.get("target_mode", "")),
                "target_mapping_kind": str(mr.get("target_mapping_kind", "")),
                "weighted_rel_gain": float(mr.get("weighted_rel_gain", 0.0)),
                "weighted_rel_gain_raw": float(mr.get("weighted_rel_gain_raw", 0.0)),
                "rel_gain": float(mr.get("rel_gain", 0.0)),
                "best_alt_probe_mse": float(mr.get("best_alt_probe_mse", float("inf"))),
                "cur_probe_mse": float(mr.get("cur_probe_mse", float("inf"))),
                "confidence": float(mr.get("confidence", 0.0)),
                "valid_frac": float(mr.get("valid_frac", 0.0)),
            }
            for mr in sorted(
                mode_best_rows,
                key=lambda row: str(row.get("target_mode", "")),
            )
        ]

        path_rows.append(path_best_row)
        rg = float(path_best_row["rel_gain"])
        if (wrg > best_weighted_rel_gain) or (abs(wrg - best_weighted_rel_gain) <= 1.0e-12 and rg > best_rel_gain):
            best_path = tuple(path)
            best_rel_gain = rg
            best_weighted_rel_gain = wrg

    promising_paths = [
        row["path"]
        for row in sorted(path_rows, key=lambda row: (row["weighted_rel_gain"], row["rel_gain"], row["static_score"]), reverse=True)
        if float(row.get("weighted_rel_gain", 0.0)) > 0.0
    ]
    if promising_paths:
        candidate_paths = promising_paths[:max_paths_i]

    reason = "ok" if (best_path is not None and best_weighted_rel_gain >= float(min_weighted_rel_gain)) else (
        "low_gain" if path_rows else "no_viable_paths"
    )
    return InverseSteeringPotential(
        allowed=bool(best_path is not None and best_weighted_rel_gain >= float(min_weighted_rel_gain)),
        reason=str(reason),
        best_path=None if best_path is None else tuple(int(v) for v in best_path),
        best_rel_gain=float(best_rel_gain),
        best_weighted_rel_gain=float(best_weighted_rel_gain),
        candidate_paths=tuple(tuple(int(v) for v in p) for p in candidate_paths),
        path_rows=tuple(PathStateFeatures.from_row(row) for row in path_rows),
    )


@torch.no_grad()
def _inverse_subtree_micro_search(
    seed_nodes,
    x_fit,
    t_fit,
    x_probe,
    t_probe,
    poly_degree,
    *,
    var_dims=None,
    target_dim=None,
    max_depth=3,
    beam_width=24,
    topk=16,
    seed_term_cap=8,
    local_score_mode="affine",
    w_fit: torch.Tensor | None = None,
    w_probe: torch.Tensor | None = None,
):
    """Small local beam search for subtree repair against a pseudo-target."""
    dm = var_dims is not None
    try:
        max_d = max(1, int(max_depth))
    except Exception:
        max_d = 3
    try:
        beam_k = max(1, int(beam_width))
    except Exception:
        beam_k = 24
    try:
        top_k = max(1, int(topk))
    except Exception:
        top_k = 16
    try:
        seed_cap = max(1, int(seed_term_cap))
    except Exception:
        seed_cap = 8

    seen = set()
    scored: list[tuple[float, float, tuple]] = []

    def _dim_ok(node):
        if not dm:
            return True
        d = node_dims(node, var_dims)
        if d is None:
            return False
        if target_dim is None:
            return True
        return dims_eq(d, target_dim)

    def _try_node(node):
        if node is None:
            return
        try:
            nn = simplify(node)
        except Exception:
            nn = node
        if node_depth(nn) > max_d:
            return
        if not _dim_ok(nn):
            return
        key = node_str(nn)
        if key in seen:
            return
        seen.add(key)
        try:
            pf = eval_node(nn, x_fit)
            pp = eval_node(nn, x_probe)
        except Exception:
            return
        if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
            return
        sc = _score_inverse_local_predictions(
            pf,
            pp,
            t_fit,
            t_probe,
            w_fit=w_fit,
            w_probe=w_probe,
            poly_degree=poly_degree,
            mode=local_score_mode,
        )
        if sc is None:
            return
        fit_mse, probe_mse = sc
        scored.append((float(probe_mse), float(fit_mse), nn))

    for node in seed_nodes:
        _try_node(node)
    if not scored:
        return []

    scored.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
    seed_terms = [row[2] for row in scored[:seed_cap]]
    beam_nodes = [row[2] for row in scored[:beam_k]]

    for _ in range(max(0, max_d - 1)):
        expansions = []
        for base in beam_nodes:
            expansions.append(("neg", base))
            for term in seed_terms:
                expansions.append(("add", base, term))
                expansions.append(("sub", base, term))
                expansions.append(("sub", term, base))
                expansions.append(("mul", base, term))
                expansions.append(("div", base, term))
                expansions.append(("mul", term, base))
                expansions.append(("div", term, base))
            # Dimensionless-only unary wrappers.
            if not dm or (target_dim is not None and dims_eq(target_dim, (0.0,) * len(target_dim))):
                for op in ("sin", "cos", "exp", "log"):
                    expansions.append((op, base))
            # sqrt/sqr may remain dimensionful; rely on _dim_ok to filter.
            for op in ("sqrt", "sqr"):
                expansions.append((op, base))

        n_before = len(scored)
        for cand in expansions:
            _try_node(cand)
        if len(scored) <= n_before:
            break
        scored.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
        beam_nodes = [row[2] for row in scored[:beam_k]]

    scored.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
    return [row[2] for row in scored[:top_k]]


@torch.no_grad()
def _inverse_collect_local_repair_candidates(
    *,
    parent_node,
    path,
    sub,
    target_dim,
    xf: torch.Tensor,
    tf: torch.Tensor,
    xp: torch.Tensor,
    tp: torch.Tensor,
    wf: torch.Tensor | None,
    wp: torch.Tensor | None,
    mfit: torch.Tensor,
    mprobe: torch.Tensor,
    pool_nodes,
    pool_dims,
    pool_phi_fit: torch.Tensor,
    pool_phi_probe: torch.Tensor,
    idxs: Sequence[int],
    poly_degree: int,
    local_mode: str,
    topk_terms: int,
    shortlist_mult: int,
    safe_eps: float,
    var_dims=None,
    max_depth: int | None = None,
    micro_search_enable: bool = False,
    micro_search_max_depth: int = 3,
    micro_search_beam_width: int = 24,
    micro_search_topk: int = 16,
    micro_search_seed_terms: int = 8,
) -> list[tuple]:
    dm = var_dims is not None
    dim0 = (0.0,) * len(var_dims[0]) if dm else None
    cand_subtrees: list[tuple] = []
    seen: set[str] = set()
    pp = tuple(int(v) for v in path)

    def _add_subtree(t):
        if t is None:
            return
        try:
            tt = simplify(t)
        except Exception:
            tt = t
        if dm:
            d = node_dims(tt, var_dims)
            if d is None or target_dim is None or (not dims_eq(d, target_dim)):
                return
        if max_depth is not None:
            try:
                repaired = replace_at(parent_node, pp, tt)
            except Exception:
                return
            if node_depth(repaired) > int(max_depth):
                return
            if dm and node_dims(repaired, var_dims) is None:
                return
        key = node_str(tt)
        if key in seen:
            return
        seen.add(key)
        cand_subtrees.append(tt)

    # Unary wrappers around current subtree.
    _add_subtree(("neg", sub))
    if not dm or (target_dim is not None and dim0 is not None and dims_eq(target_dim, dim0)):
        for op in ("sin", "cos", "exp", "log"):
            _add_subtree((op, sub))
    for op in ("sqrt", "sqr"):
        _add_subtree((op, sub))

    shortlist = [int(i) for i in list(idxs)[: max(1, min(len(idxs), int(topk_terms) * int(shortlist_mult)))]]
    for idx in shortlist:
        if idx < 0 or idx >= len(pool_nodes):
            continue
        term = pool_nodes[int(idx)]
        term_dim = pool_dims[int(idx)] if (dm and pool_dims is not None and int(idx) < len(pool_dims)) else None
        _add_subtree(term)

        if (not dm) or (target_dim is not None and term_dim is not None and dims_eq(term_dim, target_dim)):
            _add_subtree(("add", sub, term))
            _add_subtree(("sub", sub, term))
            _add_subtree(("sub", term, sub))

        if (not dm) or (dim0 is not None and term_dim is not None and dims_eq(term_dim, dim0)):
            _add_subtree(("mul", sub, term))
            _add_subtree(("div", sub, term))

    parent_op = None
    if pp:
        try:
            parent_op = str(get_at(parent_node, pp[:-1])[0])
        except Exception:
            parent_op = None
    try:
        phi_fit_m = pool_phi_fit[mfit]
        phi_probe_m = pool_phi_probe[mprobe]
    except Exception:
        phi_fit_m = None
        phi_probe_m = None
    if (phi_fit_m is not None) and (phi_probe_m is not None):
        if parent_op in ("mul", "div"):
            fam = _inverse_muldiv_monomial_candidates(
                pool_nodes,
                phi_fit_m,
                phi_probe_m,
                shortlist,
                tf,
                tp,
                w_fit=wf,
                w_probe=wp,
                poly_degree=poly_degree,
                local_score_mode=local_mode,
                topk_terms=max(4, int(topk_terms)),
                max_pair_terms=max(3, int(topk_terms // 2 + 1)),
                max_out=max(4, int(topk_terms)),
                safe_eps=float(safe_eps),
            )
            for node in fam:
                _add_subtree(node)
                _add_subtree(("neg", node))
        elif parent_op == "sqrt":
            fam = _inverse_sqrt_quadratic_candidates(
                pool_nodes,
                phi_fit_m,
                phi_probe_m,
                shortlist,
                tf,
                tp,
                w_fit=wf,
                w_probe=wp,
                poly_degree=poly_degree,
                local_score_mode=local_mode,
                topk_terms=max(4, int(topk_terms)),
                max_pair_terms=max(3, int(topk_terms // 2 + 1)),
                max_out=max(4, int(topk_terms)),
            )
            for node in fam:
                _add_subtree(node)
        elif parent_op in ("exp", "log"):
            fam = _inverse_additive_combo_candidates(
                pool_nodes,
                phi_fit_m,
                phi_probe_m,
                shortlist,
                tf,
                tp,
                w_fit=wf,
                w_probe=wp,
                poly_degree=poly_degree,
                local_score_mode=local_mode,
                topk_terms=max(4, int(topk_terms)),
                max_pair_terms=max(3, int(topk_terms // 2 + 1)),
                max_out=max(4, int(topk_terms)),
            )
            for node in fam:
                _add_subtree(node)

    if bool(micro_search_enable):
        seed_nodes = [sub] + list(cand_subtrees)
        micro_nodes = _inverse_subtree_micro_search(
            seed_nodes,
            xf,
            tf,
            xp,
            tp,
            poly_degree,
            var_dims=var_dims if dm else None,
            target_dim=target_dim if dm else None,
            max_depth=int(micro_search_max_depth),
            beam_width=int(micro_search_beam_width),
            topk=int(micro_search_topk),
            seed_term_cap=int(micro_search_seed_terms),
            local_score_mode=local_mode,
            w_fit=wf,
            w_probe=wp,
        )
        for node in micro_nodes:
            _add_subtree(node)

    return cand_subtrees


@torch.no_grad()
def _inverse_rank_local_repair_candidates(
    candidates: Sequence[tuple],
    *,
    xf: torch.Tensor,
    tf: torch.Tensor,
    xp: torch.Tensor,
    tp: torch.Tensor,
    wf: torch.Tensor | None,
    wp: torch.Tensor | None,
    poly_degree: int,
    local_mode: str,
) -> list[tuple[float, float, tuple]]:
    rows: list[tuple[float, float, tuple]] = []
    for cand_sub in candidates:
        try:
            pf = eval_node(cand_sub, xf)
            pp = eval_node(cand_sub, xp)
        except Exception:
            continue
        if (not torch.isfinite(pf).all()) or (not torch.isfinite(pp).all()):
            continue
        sc_local = _score_inverse_local_predictions(
            pf,
            pp,
            tf,
            tp,
            w_fit=wf,
            w_probe=wp,
            poly_degree=poly_degree,
            mode=local_mode,
        )
        if sc_local is None:
            continue
        fit_mse_loc, probe_mse_loc = sc_local
        rows.append((float(probe_mse_loc), float(fit_mse_loc), cand_sub))
    rows.sort(key=lambda row: (row[0], row[1], node_size(row[2]), node_str(row[2])))
    return rows
