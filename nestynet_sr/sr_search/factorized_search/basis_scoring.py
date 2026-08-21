# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

import torch

from .closures import BoundClosure, ClosureDesign
from .expr_ast import is_valid_node, node_depth, node_dims, node_str, simplify, dims_eq


HeadFitFn = Any
ClosureMaterializerFn = Any


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(str(name))
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def snap_direct_coeff(value: float, *, tol: float = 5.0e-2) -> float:
    try:
        vv = float(value)
    except Exception:
        return float(value)
    if not math.isfinite(vv):
        return vv
    snaps = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
    for target in snaps:
        if abs(vv - float(target)) <= float(tol):
            return float(target)
    return vv


def direct_power_depth_slack_from_coeffs(
    coeffs: Sequence[float],
    *,
    exponent: float,
) -> int:
    coeff_list = [float(v) for v in list(coeffs or ())]
    if not coeff_list:
        return 0
    active_terms = 0
    a0 = float(snap_direct_coeff(coeff_list[0])) if len(coeff_list) >= 1 else 0.0
    a1 = float(snap_direct_coeff(coeff_list[1])) if len(coeff_list) >= 2 else 0.0
    if abs(a0) > 1.0e-12:
        active_terms += 1
    if abs(a1) > 1.0e-12:
        active_terms += 1
    if float(exponent) == 2.0 and len(coeff_list) >= 3:
        c2 = float(snap_direct_coeff(coeff_list[2]))
        if abs(c2) > 1.0e-12:
            active_terms += 1
        # The square head can require one additional AST level beyond the
        # additive term count when the deepest active term carries an explicit
        # fitted coefficient, e.g. `0.8 * sqr(h)` instead of `sqr(h)`.
        # Without this, sparse square variants like `square_only` are fit
        # correctly but dropped by preview/materialization depth checks.
        deepest_coeff = None
        if abs(c2) > 1.0e-12:
            deepest_coeff = float(c2)
        elif abs(a1) > 1.0e-12:
            deepest_coeff = float(a1)
        term_scale_extra = 0
        if deepest_coeff is not None and abs(float(deepest_coeff) - 1.0) > 1.0e-12:
            term_scale_extra = 1
        return max(0, int(active_terms) - 1) + int(term_scale_extra)
    return max(0, int(active_terms) - 1)


def direct_power_variant_nparams(
    variant: str | None,
    *,
    exponent: float,
) -> int:
    token = str(variant or "").strip().lower()
    if float(exponent) != 2.0:
        return 2
    layouts = {
        "square_only": 1,
        "bias_square": 2,
        "linear_square": 2,
        "full_quadratic": 3,
    }
    return int(layouts.get(token, 3))


def direct_quadratic_depth_slack_from_coeffs(coeffs: Sequence[float]) -> int:
    coeff_list = [float(v) for v in list(coeffs or ())]
    for coeff in coeff_list:
        if not math.isfinite(float(coeff)) or abs(float(coeff)) <= 1.0e-12:
            continue
        if abs(abs(float(coeff)) - 1.0) > 1.0e-8:
            return 1
    return 0


def direct_multi_term_rational_coeff_depth_slack(
    coeffs: Sequence[float],
    *,
    n_u: int,
    n_v: int,
) -> int:
    coeff_list = [float(v) for v in list(coeffs or ())]
    if len(coeff_list) < int(n_u) + 1 + int(n_v):
        return 0
    # Non-unit fitted scales materialize as explicit multiplication nodes.
    # This is independent of the additive-term slack below and is needed for
    # snapped forms such as `x0 - x0*cos(t)` and `1 - cos(x)`.
    active_scales = [
        coeff_list[idx + 1]
        for idx in range(max(0, int(n_u)))
        if idx + 1 < len(coeff_list)
    ]
    den_start = int(n_u) + 1
    active_scales.extend(
        coeff_list[den_start + idx]
        for idx in range(max(0, int(n_v)))
        if den_start + idx < len(coeff_list)
    )
    for coeff in active_scales:
        try:
            snapped = snap_direct_coeff(float(coeff))
        except Exception:
            continue
        if not math.isfinite(float(snapped)) or abs(float(snapped)) <= 1.0e-12:
            continue
        if abs(float(snapped) - 1.0) > 1.0e-12:
            return 1
    return 0


def scaled_node(coeff: float, node: tuple) -> tuple | None:
    cc = snap_direct_coeff(float(coeff))
    if not math.isfinite(cc) or abs(cc) < 1.0e-12:
        return None
    if abs(cc - 1.0) < 1.0e-12:
        return node
    return simplify(("mul", ("const", float(cc)), node))


def add_terms(*terms: tuple | None) -> tuple:
    out: tuple | None = None
    for term in terms:
        if not isinstance(term, tuple) or not term:
            continue
        out = term if out is None else simplify(("add", out, term))
    if isinstance(out, tuple) and out:
        return out
    return ("const", 0.0)


def fit_direct_linear_design(
    *,
    design_fit: torch.Tensor,
    y_fit: torch.Tensor,
    design_probe: torch.Tensor,
    y_probe: torch.Tensor,
) -> tuple[float, float, list[float]] | None:
    if (not torch.is_tensor(design_fit)) or (not torch.is_tensor(design_probe)):
        return None
    if int(design_fit.shape[0]) != int(y_fit.shape[0]) or int(design_probe.shape[0]) != int(y_probe.shape[0]):
        return None
    if int(design_fit.shape[1]) <= 0 or int(design_probe.shape[1]) != int(design_fit.shape[1]):
        return None
    n_rows = int(design_fit.shape[0])
    n_cols = int(design_fit.shape[1])
    min_rows_per_param = max(0.0, _env_float("NESTY_LINEAR_HEAD_MIN_ROWS_PER_PARAM", 1.0))
    if min_rows_per_param > 0.0 and float(n_rows) < float(n_cols) * min_rows_per_param:
        return None
    if (not torch.isfinite(design_fit).all()) or (not torch.isfinite(design_probe).all()):
        return None
    if (not torch.isfinite(y_fit).all()) or (not torch.isfinite(y_probe).all()):
        return None
    try:
        max_cond = max(0.0, _env_float("NESTY_LINEAR_HEAD_MAX_COND", 1.0e12))
        if max_cond > 0.0 and n_cols > 1:
            singular_values = torch.linalg.svdvals(design_fit)
            if not torch.isfinite(singular_values).all():
                return None
            smax = float(torch.max(singular_values).item())
            smin = float(torch.min(singular_values).item())
            eps = max(1.0e-300, 1.0e-12 * max(1.0, smax))
            if smin > eps and (smax / smin) > max_cond:
                return None
        sol = torch.linalg.lstsq(design_fit, y_fit.squeeze(-1)).solution
        fit_pred = design_fit @ sol
        probe_pred = design_probe @ sol
    except Exception:
        return None
    if (not torch.isfinite(sol).all()) or (not torch.isfinite(fit_pred).all()) or (not torch.isfinite(probe_pred).all()):
        return None
    max_abs_coeff = max(0.0, _env_float("NESTY_LINEAR_HEAD_MAX_ABS_COEFF", 1.0e8))
    if max_abs_coeff > 0.0 and float(torch.max(torch.abs(sol)).item()) > max_abs_coeff:
        return None
    fit_mse = float(torch.mean((fit_pred - y_fit.squeeze(-1)) ** 2).item())
    probe_mse = float(torch.mean((probe_pred - y_probe.squeeze(-1)) ** 2).item())
    if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
        return None
    coeffs = [float(v) for v in sol.detach().cpu().reshape(-1).tolist()]
    return fit_mse, probe_mse, coeffs


def fit_identity_design(
    *,
    design_fit: torch.Tensor,
    y_fit: torch.Tensor,
    design_probe: torch.Tensor,
    y_probe: torch.Tensor,
) -> tuple[float, float, list[float]] | None:
    if (not torch.is_tensor(design_fit)) or (not torch.is_tensor(design_probe)):
        return None
    if int(design_fit.shape[0]) != int(y_fit.shape[0]) or int(design_probe.shape[0]) != int(y_probe.shape[0]):
        return None
    if int(design_fit.shape[1]) <= 0 or int(design_probe.shape[1]) <= 0:
        return None
    if (not torch.isfinite(design_fit[:, 0]).all()) or (not torch.isfinite(design_probe[:, 0]).all()):
        return None
    if (not torch.isfinite(y_fit).all()) or (not torch.isfinite(y_probe).all()):
        return None
    fit_pred = design_fit[:, 0]
    probe_pred = design_probe[:, 0]
    fit_mse = float(torch.mean((fit_pred - y_fit.squeeze(-1)) ** 2).item())
    probe_mse = float(torch.mean((probe_pred - y_probe.squeeze(-1)) ** 2).item())
    if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
        return None
    return fit_mse, probe_mse, [1.0]


def materialize_direct_linear_combo(
    terms: Sequence[tuple[float, tuple]],
    *,
    bias: float = 0.0,
    coeff_zero_tol: float = 1.0e-8,
    coeff_snap_tol: float = 1.0e-6,
    embed_coefficients: bool = False,
):
    pieces: list[tuple] = []

    def _snap_coeff(value: float) -> float:
        if not math.isfinite(value) or abs(value) <= coeff_zero_tol:
            return 0.0
        if abs(value - 1.0) <= coeff_snap_tol:
            return 1.0
        if abs(value + 1.0) <= coeff_snap_tol:
            return -1.0
        return float(value)

    for raw_coeff, term in list(terms or ()):
        if not isinstance(term, tuple) or not is_valid_node(term):
            continue
        coeff = _snap_coeff(float(raw_coeff))
        if coeff == 0.0:
            continue
        if bool(embed_coefficients):
            scaled = scaled_node(float(coeff), term)
            if scaled is not None:
                pieces.append(scaled)
        elif coeff < 0.0:
            pieces.append(("neg", term))
        else:
            pieces.append(term)

    bias_coeff = _snap_coeff(float(bias))
    if bias_coeff != 0.0:
        pieces.append(("const", float(bias_coeff) if bool(embed_coefficients) else 1.0))

    if not pieces:
        return ("const", 0.0)
    cur = pieces[0]
    for term in pieces[1:]:
        cur = ("add", cur, term)
    return simplify(cur)


def fit_direct_rational_design(
    *,
    u_fit: torch.Tensor,
    u_probe: torch.Tensor,
    v_fit: torch.Tensor,
    v_probe: torch.Tensor,
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    safe_eps: float,
) -> tuple[float, float, list[float]] | None:
    if (
        (not torch.is_tensor(u_fit))
        or (not torch.is_tensor(u_probe))
        or (not torch.is_tensor(v_fit))
        or (not torch.is_tensor(v_probe))
    ):
        return None
    try:
        u_fit_1d = u_fit.squeeze(-1)
        u_probe_1d = u_probe.squeeze(-1)
        v_fit_1d = v_fit.squeeze(-1)
        v_probe_1d = v_probe.squeeze(-1)
        y_fit_1d = y_fit.squeeze(-1)
        y_probe_1d = y_probe.squeeze(-1)
    except Exception:
        return None
    cols_fit = torch.stack(
        [
            torch.ones(int(y_fit.shape[0]), dtype=y_fit.dtype, device=y_fit.device),
            u_fit_1d,
            -(y_fit_1d * v_fit_1d),
        ],
        dim=1,
    )
    cols_probe = torch.stack(
        [
            torch.ones(int(y_probe.shape[0]), dtype=y_probe.dtype, device=y_probe.device),
            u_probe_1d,
            -(y_probe_1d * v_probe_1d),
        ],
        dim=1,
    )
    fit_ret = fit_direct_linear_design(
        design_fit=cols_fit,
        y_fit=y_fit,
        design_probe=cols_probe,
        y_probe=y_probe,
    )
    if fit_ret is None:
        return None
    _linear_fit_mse, _linear_probe_mse, coeffs = fit_ret
    try:
        a0, a1, b1 = [float(v) for v in coeffs[:3]]
    except Exception:
        return None
    denom_fit = 1.0 + float(b1) * v_fit_1d
    denom_probe = 1.0 + float(b1) * v_probe_1d
    eps = max(1.0e-8, float(safe_eps))
    if (not torch.isfinite(denom_fit).all()) or (not torch.isfinite(denom_probe).all()):
        return None
    if float(torch.min(torch.abs(denom_fit)).item()) <= eps:
        return None
    if float(torch.min(torch.abs(denom_probe)).item()) <= eps:
        return None
    num_fit = float(a0) + float(a1) * u_fit_1d
    num_probe = float(a0) + float(a1) * u_probe_1d
    fit_pred = num_fit / denom_fit
    probe_pred = num_probe / denom_probe
    if (not torch.isfinite(fit_pred).all()) or (not torch.isfinite(probe_pred).all()):
        return None
    fit_mse = float(torch.mean((fit_pred - y_fit_1d) ** 2).item())
    probe_mse = float(torch.mean((probe_pred - y_probe_1d) ** 2).item())
    if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
        return None
    return fit_mse, probe_mse, [float(a0), float(a1), float(b1)]


def fit_multi_term_rational_design(
    *,
    u_fits: Sequence[torch.Tensor],
    u_probes: Sequence[torch.Tensor],
    v_fits: Sequence[torch.Tensor],
    v_probes: Sequence[torch.Tensor],
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    safe_eps: float = 1.0e-6,
) -> tuple[float, float, list[float]] | None:
    """Fit (a0 + sum_i a_i u_i) / (1 + sum_j b_j v_j) via cross-multiply linearisation."""
    u_fits_list = list(u_fits or [])
    u_probes_list = list(u_probes or [])
    v_fits_list = list(v_fits or [])
    v_probes_list = list(v_probes or [])
    n_u = len(u_fits_list)
    n_v = len(v_fits_list)
    if n_u != len(u_probes_list) or n_v != len(v_probes_list):
        return None
    if n_u == 0 and n_v == 0:
        return None
    # Validate all inputs are finite tensors
    for tensor_list in (u_fits_list, u_probes_list, v_fits_list, v_probes_list):
        for t in tensor_list:
            if not torch.is_tensor(t):
                return None
            if not torch.isfinite(t).all():
                return None
    if (not torch.is_tensor(y_fit)) or (not torch.is_tensor(y_probe)):
        return None
    if (not torch.isfinite(y_fit).all()) or (not torch.isfinite(y_probe).all()):
        return None
    try:
        y_fit_1d = y_fit.squeeze(-1)
        y_probe_1d = y_probe.squeeze(-1)
    except Exception:
        return None
    n_fit = int(y_fit_1d.shape[0])
    n_probe = int(y_probe_1d.shape[0])
    # Build design columns: [ones, u1, u2, ..., -(y*v1), -(y*v2), ...]
    dtype = y_fit.dtype
    device = y_fit.device
    fit_cols: list[torch.Tensor] = [torch.ones(n_fit, dtype=dtype, device=device)]
    probe_cols: list[torch.Tensor] = [torch.ones(n_probe, dtype=dtype, device=device)]
    for idx in range(n_u):
        u_f = u_fits_list[idx].squeeze(-1)
        u_p = u_probes_list[idx].squeeze(-1)
        if int(u_f.shape[0]) != n_fit or int(u_p.shape[0]) != n_probe:
            return None
        fit_cols.append(u_f)
        probe_cols.append(u_p)
    for idx in range(n_v):
        v_f = v_fits_list[idx].squeeze(-1)
        v_p = v_probes_list[idx].squeeze(-1)
        if int(v_f.shape[0]) != n_fit or int(v_p.shape[0]) != n_probe:
            return None
        fit_cols.append(-(y_fit_1d * v_f))
        probe_cols.append(-(y_probe_1d * v_p))
    design_fit = torch.stack(fit_cols, dim=1)
    design_probe = torch.stack(probe_cols, dim=1)
    fit_ret = fit_direct_linear_design(
        design_fit=design_fit,
        y_fit=y_fit,
        design_probe=design_probe,
        y_probe=y_probe,
    )
    if fit_ret is None:
        return None
    _linear_fit_mse, _linear_probe_mse, coeffs = fit_ret
    # Extract a-coefficients (first n_u+1) and b-coefficients (remaining n_v)
    a_coeffs = [float(v) for v in coeffs[: n_u + 1]]
    b_coeffs = [float(v) for v in coeffs[n_u + 1: n_u + 1 + n_v]]
    # Reconstruct denominator and check safety
    denom_fit = torch.ones(n_fit, dtype=dtype, device=device)
    denom_probe = torch.ones(n_probe, dtype=dtype, device=device)
    for idx in range(n_v):
        denom_fit = denom_fit + float(b_coeffs[idx]) * v_fits_list[idx].squeeze(-1)
        denom_probe = denom_probe + float(b_coeffs[idx]) * v_probes_list[idx].squeeze(-1)
    if (not torch.isfinite(denom_fit).all()) or (not torch.isfinite(denom_probe).all()):
        return None
    # Quantile-based denominator safety
    eps = max(1.0e-8, float(safe_eps))
    for den in (denom_fit, denom_probe):
        q05 = float(torch.quantile(torch.abs(den), 0.05).item())
        median_abs = float(torch.median(torch.abs(den)).item())
        if q05 <= max(eps, 0.01 * median_abs):
            return None
    # Reconstruct numerator
    num_fit = torch.full((n_fit,), float(a_coeffs[0]), dtype=dtype, device=device)
    num_probe = torch.full((n_probe,), float(a_coeffs[0]), dtype=dtype, device=device)
    for idx in range(n_u):
        num_fit = num_fit + float(a_coeffs[idx + 1]) * u_fits_list[idx].squeeze(-1)
        num_probe = num_probe + float(a_coeffs[idx + 1]) * u_probes_list[idx].squeeze(-1)
    # True rational prediction
    fit_pred = num_fit / denom_fit
    probe_pred = num_probe / denom_probe
    if (not torch.isfinite(fit_pred).all()) or (not torch.isfinite(probe_pred).all()):
        return None
    fit_mse = float(torch.mean((fit_pred - y_fit_1d) ** 2).item())
    probe_mse = float(torch.mean((probe_pred - y_probe_1d) ** 2).item())
    if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
        return None
    return fit_mse, probe_mse, [float(v) for v in a_coeffs + b_coeffs]


def materialize_multi_term_rational_expr(
    *,
    u_nodes: Sequence[tuple],
    v_nodes: Sequence[tuple],
    coeffs: Sequence[float],
    max_depth: int,
    var_dims,
    y_dims,
) -> tuple | None:
    """Materialize (a0 + sum a_i u_i) / (1 + sum b_j v_j) into a tuple-AST node."""
    u_node_list = [node for node in list(u_nodes or []) if isinstance(node, tuple) and is_valid_node(node)]
    v_node_list = [node for node in list(v_nodes or []) if isinstance(node, tuple) and is_valid_node(node)]
    n_u = len(u_node_list)
    n_v = len(v_node_list)
    coeff_list = [float(v) for v in list(coeffs or [])]
    if len(coeff_list) < n_u + 1 + n_v:
        return None
    a_coeffs = coeff_list[: n_u + 1]
    b_coeffs = coeff_list[n_u + 1: n_u + 1 + n_v]
    # Build numerator: a0 + sum(a_i * u_i) using structural terms
    num_terms: list[tuple | None] = []
    a0_snap = snap_direct_coeff(float(a_coeffs[0]))
    if abs(a0_snap) >= 1.0e-12:
        num_terms.append(("const", float(a0_snap)))
    for idx in range(n_u):
        term = scaled_node(float(a_coeffs[idx + 1]), u_node_list[idx])
        if term is not None:
            num_terms.append(term)
    num_node = add_terms(*num_terms) if num_terms else ("const", 0.0)
    # Build denominator: 1 + sum(b_j * v_j) using structural terms
    den_terms: list[tuple | None] = [("const", 1.0)]
    for idx in range(n_v):
        term = scaled_node(float(b_coeffs[idx]), v_node_list[idx])
        if term is not None:
            den_terms.append(term)
    den_node = add_terms(*den_terms)
    try:
        expr = simplify(("div", num_node, den_node))
    except Exception:
        return None
    if not is_valid_node(expr):
        return None
    # Allow extra depth for multi-term rational structures
    depth_budget = (
        int(max_depth)
        + max(0, n_u + n_v - 2)
        + direct_multi_term_rational_coeff_depth_slack(coeff_list, n_u=n_u, n_v=n_v)
    )
    if int(node_depth(expr)) > int(depth_budget):
        return None
    if var_dims is not None:
        try:
            expr_dim = node_dims(expr, var_dims)
        except Exception:
            expr_dim = None
        if expr_dim is None:
            return None
        if y_dims is not None and not dims_eq(expr_dim, y_dims):
            return None
    return expr


def fit_direct_quadratic_sqrt_design(
    *,
    quad_fit: torch.Tensor,
    quad_probe: torch.Tensor,
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    anchor_fit: torch.Tensor | None = None,
    anchor_probe: torch.Tensor | None = None,
    safe_eps: float = 1.0e-8,
) -> tuple[float, float, list[float]] | None:
    if (not torch.is_tensor(quad_fit)) or (not torch.is_tensor(quad_probe)):
        return None
    if int(quad_fit.shape[0]) != int(y_fit.shape[0]) or int(quad_probe.shape[0]) != int(y_probe.shape[0]):
        return None
    if int(quad_fit.shape[1]) <= 0 or int(quad_probe.shape[1]) != int(quad_fit.shape[1]):
        return None
    if (not torch.isfinite(quad_fit).all()) or (not torch.isfinite(quad_probe).all()):
        return None
    y_fit_1d = y_fit.squeeze(-1)
    y_probe_1d = y_probe.squeeze(-1)
    target_fit = y_fit_1d
    target_probe = y_probe_1d
    if anchor_fit is not None or anchor_probe is not None:
        if (not torch.is_tensor(anchor_fit)) or (not torch.is_tensor(anchor_probe)):
            return None
        anchor_fit_1d = anchor_fit.squeeze(-1)
        anchor_probe_1d = anchor_probe.squeeze(-1)
        if (not torch.isfinite(anchor_fit_1d).all()) or (not torch.isfinite(anchor_probe_1d).all()):
            return None
        eps = max(1.0e-12, float(safe_eps))
        if float(torch.min(torch.abs(anchor_fit_1d)).item()) <= eps:
            return None
        if float(torch.min(torch.abs(anchor_probe_1d)).item()) <= eps:
            return None
        target_fit = y_fit_1d / anchor_fit_1d
        target_probe = y_probe_1d / anchor_probe_1d
    if (not torch.isfinite(target_fit).all()) or (not torch.isfinite(target_probe).all()):
        return None
    fit_ret = fit_direct_linear_design(
        design_fit=quad_fit,
        y_fit=(target_fit**2).unsqueeze(-1),
        design_probe=quad_probe,
        y_probe=(target_probe**2).unsqueeze(-1),
    )
    if fit_ret is None:
        return None
    _sq_fit_mse, _sq_probe_mse, coeffs_raw = fit_ret
    coeff_vec = torch.tensor(coeffs_raw, dtype=quad_fit.dtype, device=quad_fit.device)
    eps = max(1.0e-12, float(safe_eps))
    quad_pred_fit = quad_fit @ coeff_vec
    quad_pred_probe = quad_probe @ coeff_vec
    if (not torch.isfinite(quad_pred_fit).all()) or (not torch.isfinite(quad_pred_probe).all()):
        return None
    if float(torch.min(quad_pred_fit).item()) <= eps or float(torch.min(quad_pred_probe).item()) <= eps:
        return None
    fit_pred = torch.sqrt(quad_pred_fit)
    probe_pred = torch.sqrt(quad_pred_probe)
    if anchor_fit is not None and anchor_probe is not None:
        fit_pred = fit_pred * anchor_fit.squeeze(-1)
        probe_pred = probe_pred * anchor_probe.squeeze(-1)
    if (not torch.isfinite(fit_pred).all()) or (not torch.isfinite(probe_pred).all()):
        return None
    fit_mse = float(torch.mean((fit_pred - y_fit_1d) ** 2).item())
    probe_mse = float(torch.mean((probe_pred - y_probe_1d) ** 2).item())
    if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
        return None
    coeffs = [float(v) for v in coeff_vec.detach().cpu().tolist()]
    return fit_mse, probe_mse, coeffs


def fit_direct_power_design(
    *,
    h_fit: torch.Tensor,
    h_probe: torch.Tensor,
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
    exponent: float,
    variant: str | None = None,
    anchor_fit: torch.Tensor | None = None,
    anchor_probe: torch.Tensor | None = None,
    safe_eps: float = 1.0e-8,
) -> tuple[float, float, list[float]] | None:
    if (not torch.is_tensor(h_fit)) or (not torch.is_tensor(h_probe)):
        return None
    if int(h_fit.shape[0]) != int(y_fit.shape[0]) or int(h_probe.shape[0]) != int(y_probe.shape[0]):
        return None
    if (not torch.isfinite(h_fit).all()) or (not torch.isfinite(h_probe).all()):
        return None
    y_fit_1d = y_fit.squeeze(-1)
    y_probe_1d = y_probe.squeeze(-1)
    core_fit = y_fit_1d
    core_probe = y_probe_1d
    eps = max(1.0e-12, float(safe_eps))
    if anchor_fit is not None or anchor_probe is not None:
        if (not torch.is_tensor(anchor_fit)) or (not torch.is_tensor(anchor_probe)):
            return None
        anchor_fit_1d = anchor_fit.squeeze(-1)
        anchor_probe_1d = anchor_probe.squeeze(-1)
        if (not torch.isfinite(anchor_fit_1d).all()) or (not torch.isfinite(anchor_probe_1d).all()):
            return None
        if float(torch.min(torch.abs(anchor_fit_1d)).item()) <= eps:
            return None
        if float(torch.min(torch.abs(anchor_probe_1d)).item()) <= eps:
            return None
        core_fit = y_fit_1d / anchor_fit_1d
        core_probe = y_probe_1d / anchor_probe_1d
    if (not torch.isfinite(core_fit).all()) or (not torch.isfinite(core_probe).all()):
        return None

    exp_value = float(exponent)
    if exp_value == 0.5:
        if float(torch.min(core_fit).item()) <= eps or float(torch.min(core_probe).item()) <= eps:
            return None
        target_fit = core_fit**2
        target_probe = core_probe**2
    elif exp_value == -0.5:
        if float(torch.min(core_fit).item()) <= eps or float(torch.min(core_probe).item()) <= eps:
            return None
        target_fit = torch.square(torch.reciprocal(core_fit))
        target_probe = torch.square(torch.reciprocal(core_probe))
    elif exp_value == -1.0:
        if float(torch.min(core_fit).item()) <= eps or float(torch.min(core_probe).item()) <= eps:
            return None
        target_fit = torch.reciprocal(core_fit)
        target_probe = torch.reciprocal(core_probe)
    elif exp_value == -2.0:
        if float(torch.min(core_fit).item()) <= eps or float(torch.min(core_probe).item()) <= eps:
            return None
        target_fit = torch.sqrt(torch.reciprocal(core_fit))
        target_probe = torch.sqrt(torch.reciprocal(core_probe))
    elif exp_value == 2.0:
        variant_token = str(variant or "full_quadratic").strip().lower()
        layouts: dict[str, tuple[int, ...]] = {
            "square_only": (2,),
            "bias_square": (0, 2),
            "linear_square": (1, 2),
            "full_quadratic": (0, 1, 2),
        }
        cols = [
            torch.ones(int(h_fit.shape[0]), dtype=h_fit.dtype, device=h_fit.device),
            h_fit.squeeze(-1),
            torch.square(h_fit.squeeze(-1)),
        ]
        cols_probe = [
            torch.ones(int(h_probe.shape[0]), dtype=h_probe.dtype, device=h_probe.device),
            h_probe.squeeze(-1),
            torch.square(h_probe.squeeze(-1)),
        ]
        active_idx = layouts.get(variant_token, layouts["full_quadratic"])
        fit_ret = fit_direct_linear_design(
            design_fit=torch.stack([cols[idx] for idx in active_idx], dim=1),
            y_fit=core_fit.unsqueeze(-1),
            design_probe=torch.stack([cols_probe[idx] for idx in active_idx], dim=1),
            y_probe=core_probe.unsqueeze(-1),
        )
        if fit_ret is None:
            return None
        _quad_fit_mse, _quad_probe_mse, coeffs_active = fit_ret
        coeffs = [0.0, 0.0, 0.0]
        try:
            for idx, coeff_value in zip(active_idx, list(coeffs_active or ())):
                coeffs[int(idx)] = float(coeff_value)
        except Exception:
            return None
        try:
            c0, c1, c2 = [float(v) for v in coeffs[:3]]
        except Exception:
            return None
        pred_fit = float(c0) + float(c1) * h_fit.squeeze(-1) + float(c2) * torch.square(h_fit.squeeze(-1))
        pred_probe = float(c0) + float(c1) * h_probe.squeeze(-1) + float(c2) * torch.square(h_probe.squeeze(-1))
        if anchor_fit is not None and anchor_probe is not None:
            pred_fit = pred_fit * anchor_fit.squeeze(-1)
            pred_probe = pred_probe * anchor_probe.squeeze(-1)
        if (not torch.isfinite(pred_fit).all()) or (not torch.isfinite(pred_probe).all()):
            return None
        fit_mse = float(torch.mean((pred_fit - y_fit_1d) ** 2).item())
        probe_mse = float(torch.mean((pred_probe - y_probe_1d) ** 2).item())
        if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
            return None
        return fit_mse, probe_mse, [float(c0), float(c1), float(c2)]
    else:
        return None

    fit_ret = fit_direct_linear_design(
        design_fit=torch.stack(
            [
                torch.ones(int(h_fit.shape[0]), dtype=h_fit.dtype, device=h_fit.device),
                h_fit.squeeze(-1),
            ],
            dim=1,
        ),
        y_fit=target_fit.unsqueeze(-1),
        design_probe=torch.stack(
            [
                torch.ones(int(h_probe.shape[0]), dtype=h_probe.dtype, device=h_probe.device),
                h_probe.squeeze(-1),
            ],
            dim=1,
        ),
        y_probe=target_probe.unsqueeze(-1),
    )
    if fit_ret is None:
        return None
    _aff_fit_mse, _aff_probe_mse, coeffs = fit_ret
    try:
        a0, a1 = [float(v) for v in coeffs[:2]]
    except Exception:
        return None
    inner_fit = float(a0) + float(a1) * h_fit.squeeze(-1)
    inner_probe = float(a0) + float(a1) * h_probe.squeeze(-1)
    if (not torch.isfinite(inner_fit).all()) or (not torch.isfinite(inner_probe).all()):
        return None
    if exp_value in {2.0, -2.0}:
        if float(torch.min(torch.abs(inner_fit)).item()) <= eps or float(torch.min(torch.abs(inner_probe)).item()) <= eps:
            return None
    elif float(torch.min(inner_fit).item()) <= eps or float(torch.min(inner_probe).item()) <= eps:
        return None
    pred_fit = torch.pow(inner_fit, exp_value)
    pred_probe = torch.pow(inner_probe, exp_value)
    if anchor_fit is not None and anchor_probe is not None:
        pred_fit = pred_fit * anchor_fit.squeeze(-1)
        pred_probe = pred_probe * anchor_probe.squeeze(-1)
    if (not torch.isfinite(pred_fit).all()) or (not torch.isfinite(pred_probe).all()):
        return None
    fit_mse = float(torch.mean((pred_fit - y_fit_1d) ** 2).item())
    probe_mse = float(torch.mean((pred_probe - y_probe_1d) ** 2).item())
    if (not math.isfinite(fit_mse)) or (not math.isfinite(probe_mse)):
        return None
    return fit_mse, probe_mse, [float(a0), float(a1)]


HEAD_SOLVER_REGISTRY: dict[str, HeadFitFn] = {
    "identity": lambda *, design, y_fit, y_probe: fit_identity_design(
        design_fit=design.fit_matrix,
        y_fit=y_fit,
        design_probe=design.probe_matrix,
        y_probe=y_probe,
    ),
    "linear": lambda *, design, y_fit, y_probe: fit_direct_linear_design(
        design_fit=design.fit_matrix,
        y_fit=y_fit,
        design_probe=design.probe_matrix,
        y_probe=y_probe,
    ),
    "harmonic_linear": lambda *, design, y_fit, y_probe: fit_direct_linear_design(
        design_fit=design.fit_matrix,
        y_fit=y_fit,
        design_probe=design.probe_matrix,
        y_probe=y_probe,
    ),
    "fractional_linear": lambda *, design, y_fit, y_probe: fit_direct_rational_design(
        u_fit=dict(design.payload or {})["u_fit"],
        u_probe=dict(design.payload or {})["u_probe"],
        v_fit=dict(design.payload or {})["v_fit"],
        v_probe=dict(design.payload or {})["v_probe"],
        y_fit=y_fit,
        y_probe=y_probe,
        safe_eps=float(dict(design.payload or {}).get("safe_eps", 1.0e-6)),
    ),
    "quadratic_sqrt": lambda *, design, y_fit, y_probe: fit_direct_quadratic_sqrt_design(
        quad_fit=dict(design.payload or {})["quad_fit"],
        quad_probe=dict(design.payload or {})["quad_probe"],
        y_fit=y_fit,
        y_probe=y_probe,
        anchor_fit=dict(design.payload or {}).get("anchor_fit", None),
        anchor_probe=dict(design.payload or {}).get("anchor_probe", None),
        safe_eps=float(dict(design.payload or {}).get("safe_eps", 1.0e-8)),
    ),
    "discrete_power": lambda *, design, y_fit, y_probe: fit_direct_power_design(
        h_fit=dict(design.payload or {})["h_fit"],
        h_probe=dict(design.payload or {})["h_probe"],
        y_fit=y_fit,
        y_probe=y_probe,
        exponent=float(dict(design.payload or {})["exponent"]),
        variant=dict(design.payload or {}).get("power_variant", None),
        anchor_fit=dict(design.payload or {}).get("anchor_fit", None),
        anchor_probe=dict(design.payload or {}).get("anchor_probe", None),
        safe_eps=float(dict(design.payload or {}).get("safe_eps", 1.0e-8)),
    ),
    "multi_term_fractional": lambda *, design, y_fit, y_probe: fit_multi_term_rational_design(
        u_fits=list(dict(design.payload or {}).get("u_fits", []) or []),
        u_probes=list(dict(design.payload or {}).get("u_probes", []) or []),
        v_fits=list(dict(design.payload or {}).get("v_fits", []) or []),
        v_probes=list(dict(design.payload or {}).get("v_probes", []) or []),
        y_fit=y_fit,
        y_probe=y_probe,
        safe_eps=float(dict(design.payload or {}).get("safe_eps", 1.0e-6)),
    ),
}


def materialize_direct_rational_expr(
    *,
    u_node: tuple,
    v_node: tuple,
    coeffs: Sequence[float],
    max_depth: int,
    var_dims,
    y_dims,
) -> tuple | None:
    try:
        a0, a1, b1 = [float(v) for v in list(coeffs or [0.0, 1.0, 0.0])[:3]]
    except Exception:
        return None
    num_node = add_terms(
        None if abs(snap_direct_coeff(a0)) < 1.0e-12 else ("const", float(snap_direct_coeff(a0))),
        scaled_node(float(a1), u_node),
    )
    den_node = add_terms(
        ("const", 1.0),
        scaled_node(float(b1), v_node),
    )
    try:
        expr = simplify(("div", num_node, den_node))
    except Exception:
        return None
    if not is_valid_node(expr):
        return None
    if int(node_depth(expr)) > int(max_depth):
        return None
    if var_dims is not None:
        try:
            expr_dim = node_dims(expr, var_dims)
        except Exception:
            expr_dim = None
        if expr_dim is None:
            return None
        if y_dims is not None and not dims_eq(expr_dim, y_dims):
            return None
    return expr


def materialize_direct_quadratic_expr(
    *,
    base_nodes: Sequence[tuple],
    coeffs: Sequence[float],
    anchor_node: tuple | None,
    max_depth: int,
    var_dims,
    y_dims,
) -> tuple | None:
    terms: list[tuple[float, tuple]] = []
    for coeff, base_node in zip(list(coeffs or ()), list(base_nodes or ())):
        if not (isinstance(base_node, tuple) and is_valid_node(base_node)):
            continue
        terms.append((float(coeff), simplify(("sqr", base_node))))
    quad_expr = materialize_direct_linear_combo(
        terms,
        bias=0.0,
        embed_coefficients=True,
        coeff_snap_tol=1.0e-8,
    )
    if not (isinstance(quad_expr, tuple) and is_valid_node(quad_expr)):
        return None
    expr = simplify(("sqrt", quad_expr))
    if isinstance(anchor_node, tuple) and is_valid_node(anchor_node):
        expr = simplify(("mul", anchor_node, expr))
    if not is_valid_node(expr):
        return None
    depth_budget = int(max_depth)
    if isinstance(anchor_node, tuple) and is_valid_node(anchor_node):
        # Typed quadratic prefactors expand into a binary AST with one extra
        # level for the outer mul and additional nesting for 3-term sums.
        depth_budget += max(0, len(list(base_nodes or ())) - 2)
    depth_budget += direct_quadratic_depth_slack_from_coeffs(coeffs)
    if int(node_depth(expr)) > int(depth_budget):
        return None
    if var_dims is not None:
        try:
            expr_dim = node_dims(expr, var_dims)
        except Exception:
            expr_dim = None
        if expr_dim is None:
            return None
        if y_dims is not None and not dims_eq(expr_dim, y_dims):
            return None
    return expr


def materialize_direct_power_expr(
    *,
    hole_node: tuple,
    coeffs: Sequence[float],
    exponent: float,
    anchor_node: tuple | None,
    max_depth: int,
    var_dims,
    y_dims,
) -> tuple | None:
    coeff_list = [float(v) for v in list(coeffs or ())]
    if not coeff_list:
        return None
    a0 = float(coeff_list[0]) if len(coeff_list) >= 1 else 0.0
    a1 = float(coeff_list[1]) if len(coeff_list) >= 2 else 1.0
    a0_snap = float(snap_direct_coeff(a0))
    a1_snap = float(snap_direct_coeff(a1))
    if abs(a0_snap) < 1.0e-12 and abs(a1_snap - 1.0) < 1.0e-12:
        inner_node = hole_node
    elif abs(a0_snap - 1.0) < 1.0e-12 and abs(a1_snap + 1.0) < 1.0e-12:
        inner_node = simplify(("sub", ("const", 1.0), hole_node))
    elif abs(a0_snap + 1.0) < 1.0e-12 and abs(a1_snap - 1.0) < 1.0e-12:
        inner_node = simplify(("sub", hole_node, ("const", 1.0)))
    else:
        inner_node = add_terms(
            None if abs(a0_snap) < 1.0e-12 else ("const", float(a0_snap)),
            scaled_node(float(a1_snap), hole_node),
        )
    if not (isinstance(inner_node, tuple) and is_valid_node(inner_node)):
        return None
    exp_value = float(exponent)
    anchored = isinstance(anchor_node, tuple) and is_valid_node(anchor_node)
    if exp_value == 0.5:
        core_node = simplify(("sqrt", inner_node))
        expr = simplify(("mul", anchor_node, core_node)) if anchored else core_node
    elif exp_value == -0.5:
        denom = simplify(("sqrt", inner_node))
        expr = simplify(("div", anchor_node, denom)) if anchored else simplify(("div", ("const", 1.0), denom))
    elif exp_value == -1.0:
        expr = simplify(("div", anchor_node, inner_node)) if anchored else simplify(("div", ("const", 1.0), inner_node))
    elif exp_value == -2.0:
        denom = simplify(("sqr", inner_node))
        expr = simplify(("div", anchor_node, denom)) if anchored else simplify(("div", ("const", 1.0), denom))
    elif exp_value == 2.0:
        if len(coeff_list) >= 3:
            c2_snap = float(snap_direct_coeff(float(coeff_list[2])))
            core_node = add_terms(
                None if abs(a0_snap) < 1.0e-12 else ("const", float(a0_snap)),
                scaled_node(float(a1_snap), hole_node),
                scaled_node(float(c2_snap), simplify(("sqr", hole_node))),
            )
        else:
            core_node = simplify(("sqr", inner_node))
        expr = simplify(("mul", anchor_node, core_node)) if anchored else core_node
    else:
        return None
    if not is_valid_node(expr):
        return None
    depth_budget = int(max_depth) + int(
        direct_power_depth_slack_from_coeffs(coeff_list, exponent=float(exponent))
    )
    if int(node_depth(expr)) > int(depth_budget):
        return None
    if var_dims is not None:
        try:
            expr_dim = node_dims(expr, var_dims)
        except Exception:
            expr_dim = None
        if expr_dim is None:
            return None
        if y_dims is not None and not dims_eq(expr_dim, y_dims):
            return None
    return expr


def make_additive_basis_transition(
    *,
    core_expr: tuple,
    term_nodes: Sequence[tuple],
    coeffs: Sequence[float],
    compiled_expr: tuple,
    ridge: float,
    prune_rel: float,
) -> dict[str, Any]:
    coeff_list_raw = [float(v) for v in list(coeffs or ())]
    core_coeff = float(coeff_list_raw[0]) if coeff_list_raw else 1.0
    bias_coeff = float(coeff_list_raw[len(list(term_nodes or ())) + 1]) if len(coeff_list_raw) > len(list(term_nodes or ())) + 1 else 0.0
    rel_tol = max(float(prune_rel), 1.0e-8)
    scale = max(1.0e-12, abs(float(core_coeff)), abs(float(bias_coeff)))
    kept_terms: list[tuple] = []
    kept_coeffs: list[float] = []
    for idx, node in enumerate(list(term_nodes or ())):
        if not (isinstance(node, tuple) and is_valid_node(node)):
            continue
        coeff_idx = int(idx) + 1
        coeff = float(coeff_list_raw[coeff_idx]) if coeff_idx < len(coeff_list_raw) else 0.0
        scale = max(scale, abs(float(coeff)))
        kept_terms.append(node)
        kept_coeffs.append(float(coeff))
    filtered_terms: list[tuple] = []
    filtered_coeffs: list[float] = []
    for node, coeff in zip(kept_terms, kept_coeffs):
        if abs(float(coeff)) <= float(scale) * float(rel_tol):
            continue
        filtered_terms.append(node)
        filtered_coeffs.append(float(coeff))
    compact_coeffs = [float(core_coeff), *filtered_coeffs, float(bias_coeff)]
    compact_expr = materialize_direct_linear_combo(
        [(float(core_coeff), core_expr), *[(float(c), node) for c, node in zip(filtered_coeffs, filtered_terms)]],
        bias=float(bias_coeff),
    )
    return {
        "kind": "additive_basis_admission",
        "core_expr": core_expr,
        "term_nodes": filtered_terms,
        "coeffs": compact_coeffs,
        "ridge": float(ridge),
        "prune_rel": float(prune_rel),
        "compiled_expr": (
            compact_expr
            if isinstance(compact_expr, tuple) and is_valid_node(compact_expr)
            else (compiled_expr if isinstance(compiled_expr, tuple) and is_valid_node(compiled_expr) else None)
        ),
    }


def materialize_bound_closure_expr(
    bound_closure: BoundClosure,
    *,
    design: ClosureDesign,
    coeffs: Sequence[float],
) -> tuple | None:
    materializer = str(design.materializer or "literal")
    payload = dict(design.materializer_payload or {})
    materializer_fn = MATERIALIZER_REGISTRY.get(materializer)
    if callable(materializer_fn):
        return materializer_fn(payload=payload, coeffs=coeffs)
    raise ValueError(
        f"unsupported closure materializer {materializer!r} for "
        f"{bound_closure.spec.closure_id!r}"
    )


MATERIALIZER_REGISTRY: dict[str, ClosureMaterializerFn] = {
    "literal": lambda *, payload, coeffs: (
        payload.get("expr", None)
        if isinstance(payload.get("expr", None), tuple) and is_valid_node(payload.get("expr", None))
        else None
    ),
    "linear_combo": lambda *, payload, coeffs: (
        materialize_direct_linear_combo(
            [
                (float(list(coeffs)[idx]), term)
                for idx, term in enumerate(list(payload.get("terms", []) or []))
                if idx < len(list(coeffs or ())) and isinstance(term, tuple) and is_valid_node(term)
            ],
            bias=(
                float(list(coeffs)[int(payload.get("bias_index"))])
                if isinstance(payload.get("bias_index", None), int)
                and 0 <= int(payload.get("bias_index")) < len(list(coeffs or ()))
                else 0.0
            ),
        )
    ),
    "linear_combo_scaled": lambda *, payload, coeffs: (
        materialize_direct_linear_combo(
            [
                (float(list(coeffs)[idx]), term)
                for idx, term in enumerate(list(payload.get("terms", []) or []))
                if idx < len(list(coeffs or ())) and isinstance(term, tuple) and is_valid_node(term)
            ],
            bias=(
                float(list(coeffs)[int(payload.get("bias_index"))])
                if isinstance(payload.get("bias_index", None), int)
                and 0 <= int(payload.get("bias_index")) < len(list(coeffs or ()))
                else 0.0
            ),
            embed_coefficients=True,
        )
    ),
    "rational_affine": lambda *, payload, coeffs: materialize_direct_rational_expr(
        u_node=payload["u_node"],
        v_node=payload["v_node"],
        coeffs=coeffs,
        max_depth=int(payload["max_depth"]),
        var_dims=payload.get("var_dims", None),
        y_dims=payload.get("y_dims", None),
    ),
    "quadratic_sqrt": lambda *, payload, coeffs: materialize_direct_quadratic_expr(
        base_nodes=payload["base_nodes"],
        coeffs=coeffs,
        anchor_node=payload.get("anchor_node", None),
        max_depth=int(payload["max_depth"]),
        var_dims=payload.get("var_dims", None),
        y_dims=payload.get("y_dims", None),
    ),
    "affine_power": lambda *, payload, coeffs: materialize_direct_power_expr(
        hole_node=payload["hole_node"],
        coeffs=coeffs,
        exponent=float(payload["exponent"]),
        anchor_node=payload.get("anchor_node", None),
        max_depth=int(payload["max_depth"]),
        var_dims=payload.get("var_dims", None),
        y_dims=payload.get("y_dims", None),
    ),
    "multi_term_rational": lambda *, payload, coeffs: materialize_multi_term_rational_expr(
        u_nodes=list(payload.get("u_nodes", []) or []),
        v_nodes=list(payload.get("v_nodes", []) or []),
        coeffs=coeffs,
        max_depth=int(payload.get("max_depth", 8)),
        var_dims=payload.get("var_dims", None),
        y_dims=payload.get("y_dims", None),
    ),
}


def score_bound_closure(
    bound_closure: BoundClosure,
    *,
    design: ClosureDesign,
    y_fit: torch.Tensor,
    y_probe: torch.Tensor,
) -> dict[str, Any] | None:
    head_solver = str(bound_closure.spec.head_solver or "").strip().lower()
    fit_fn = HEAD_SOLVER_REGISTRY.get(head_solver)
    if not callable(fit_fn):
        raise ValueError(f"unsupported closure head solver {head_solver!r}")
    fit_ret = fit_fn(design=design, y_fit=y_fit, y_probe=y_probe)
    if fit_ret is None:
        return None
    fit_mse, probe_mse, coeffs = fit_ret

    expr = materialize_bound_closure_expr(
        bound_closure,
        design=design,
        coeffs=coeffs,
    )
    if expr is None:
        return None

    return {
        "expr": expr,
        "fit_mse": float(fit_mse),
        "probe_mse": float(probe_mse),
        "coeffs": [float(v) for v in coeffs],
    }


def scaffold_candidate_anchor_head_context(
    *,
    route_name: str,
    candidate_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scaffold_meta = dict(candidate_meta or {}) if isinstance(candidate_meta, Mapping) else {}
    scaffold_form = str(
        dict(scaffold_meta.get("scaffold_metadata", {}) or {}).get("form", "") or ""
    ).strip().lower()
    scaffold_anchor_node = scaffold_meta.get("scaffold_anchor_node", None)
    use_anchor_head = (
        str(route_name) == "closure_search"
        and scaffold_form.endswith("_add")
        and isinstance(scaffold_anchor_node, tuple)
        and is_valid_node(scaffold_anchor_node)
    )
    return {
        "candidate_meta": scaffold_meta,
        "scaffold_form": scaffold_form,
        "scaffold_anchor_node": scaffold_anchor_node,
        "use_anchor_head": bool(use_anchor_head),
    }


def build_scaffold_candidate_score_cfg(
    score_cfg_base: Mapping[str, Any],
    *,
    route_name: str,
    candidate_meta: Mapping[str, Any] | None,
    stats: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = scaffold_candidate_anchor_head_context(
        route_name=route_name,
        candidate_meta=candidate_meta,
    )
    score_cfg = dict(score_cfg_base)
    if not bool(context.get("use_anchor_head", False)):
        return score_cfg, context
    scaffold_anchor_node = context.get("scaffold_anchor_node", None)
    anchor_terms = list(score_cfg.get("score_head_var_terms", []) or [])
    anchor_key = node_str(scaffold_anchor_node)
    seen_terms = {node_str(t) for t in anchor_terms if isinstance(t, tuple)}
    if anchor_key not in seen_terms:
        anchor_terms.append(scaffold_anchor_node)
    score_cfg["score_head_enable"] = True
    score_cfg["score_head_vars_enable"] = True
    score_cfg["score_head_var_terms"] = anchor_terms
    score_cfg["score_head_only"] = True
    if isinstance(stats, dict):
        stats["anchor_head_attempts"] = int(stats.get("anchor_head_attempts", 0)) + 1
    return score_cfg, context


def record_anchor_head_compare(
    stats: dict[str, Any] | None,
    *,
    context: Mapping[str, Any],
    expr: tuple,
    base_mse: float,
    head_mse: float,
) -> None:
    if not isinstance(stats, dict):
        return
    if (not math.isfinite(base_mse)) or (not math.isfinite(head_mse)):
        return
    delta = float(base_mse - head_mse)
    stats["anchor_head_compare_delta_sum"] = float(
        stats.get("anchor_head_compare_delta_sum", 0.0) or 0.0
    ) + float(delta)
    if delta > 1.0e-12:
        stats["anchor_head_compare_improved"] = int(
            stats.get("anchor_head_compare_improved", 0)
        ) + 1
    elif delta < -1.0e-12:
        stats["anchor_head_compare_worsened"] = int(
            stats.get("anchor_head_compare_worsened", 0)
        ) + 1
    else:
        stats["anchor_head_compare_neutral"] = int(
            stats.get("anchor_head_compare_neutral", 0)
        ) + 1
    examples = stats.get("anchor_head_compare_examples", None)
    if isinstance(examples, list) and len(examples) < 8:
        candidate_meta = dict(context.get("candidate_meta", {}) or {})
        scaffold_anchor_node = context.get("scaffold_anchor_node", None)
        examples.append(
            {
                "scaffold_id": str(candidate_meta.get("scaffold_id", "") or ""),
                "expr": str(node_str(expr)),
                "anchor_expr": str(
                    candidate_meta.get("scaffold_anchor_expr", "")
                    or (node_str(scaffold_anchor_node) if isinstance(scaffold_anchor_node, tuple) else "")
                ),
                "base_raw_mse": float(base_mse),
                "anchor_head_raw_mse": float(head_mse),
                "delta_raw_mse": float(delta),
            }
        )


__all__ = [
    "add_terms",
    "build_scaffold_candidate_score_cfg",
    "direct_quadratic_depth_slack_from_coeffs",
    "direct_multi_term_rational_coeff_depth_slack",
    "fit_identity_design",
    "fit_direct_linear_design",
    "fit_direct_power_design",
    "fit_direct_rational_design",
    "fit_multi_term_rational_design",
    "HEAD_SOLVER_REGISTRY",
    "make_additive_basis_transition",
    "MATERIALIZER_REGISTRY",
    "materialize_bound_closure_expr",
    "materialize_direct_linear_combo",
    "materialize_direct_power_expr",
    "materialize_direct_rational_expr",
    "materialize_multi_term_rational_expr",
    "record_anchor_head_compare",
    "scaled_node",
    "scaffold_candidate_anchor_head_context",
    "score_bound_closure",
    "snap_direct_coeff",
]
