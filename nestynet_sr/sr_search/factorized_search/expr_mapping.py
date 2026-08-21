# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Mapping fit/eval helpers shared across factorized symbolic search explorer subsystems."""

from __future__ import annotations

import logging
import math
import time

import torch

from nestynet_sr.sr_search.rational_sparsify import (
    DEFAULT_POLY_STLSQ_CFG,
    DEFAULT_RAT_STLSQ_CFG,
    _log_sparsify_result,
    stlsq_sparsify_poly_coeffs,
    stlsq_sparsify_rational_coeffs,
)


_log = logging.getLogger(__name__)


def _diag_inc(diagnostics, key, amount=1):
    if isinstance(diagnostics, dict):
        diagnostics[str(key)] = int(diagnostics.get(str(key), 0)) + int(amount)


def _diag_add_time(diagnostics, key, elapsed):
    if isinstance(diagnostics, dict):
        diagnostics[str(key)] = float(diagnostics.get(str(key), 0.0)) + float(max(0.0, elapsed))


def mean_squared_error_same_shape(y_true, y_pred) -> float:
    """Compute MSE without allowing accidental broadcast expansion."""
    if not torch.is_tensor(y_pred):
        y_pred = torch.as_tensor(y_pred)
    if torch.is_tensor(y_true):
        y_cmp = y_true.to(device=y_pred.device, dtype=y_pred.dtype)
    else:
        y_cmp = torch.as_tensor(y_true, device=y_pred.device, dtype=y_pred.dtype)
    if tuple(y_cmp.shape) != tuple(y_pred.shape):
        if int(y_cmp.numel()) != int(y_pred.numel()):
            return float("nan")
        y_cmp = y_cmp.reshape_as(y_pred)
    mse = torch.mean((y_cmp - y_pred) ** 2)
    return float(mse)


def _fit_affine_fast(pred, y):
    """Closed-form y ~= c0 + c1 * ((pred - mu) / std) for scorer hot loops."""
    f = pred.squeeze(-1)
    yy = y.squeeze(-1)
    mu_t = f.mean()
    std_t = f.std()
    mu = float(mu_t)
    std = float(std_t)
    if std < 1e-12:
        c = pred.new_zeros(2)
        c[0] = yy.mean()
        if not torch.isfinite(c).all():
            return None
        return c, mu, 1.0

    z = (f - mu_t) / std_t
    z_mean = z.mean()
    y_mean = yy.mean()
    zc = z - z_mean
    yc = yy - y_mean
    den = (zc * zc).sum().clamp_min(1e-30)
    c1 = (zc * yc).sum() / den
    c0 = y_mean - c1 * z_mean
    coeffs = torch.stack([c0, c1])
    if not torch.isfinite(coeffs).all():
        return None
    return coeffs, mu, std


def fit_poly(pred, y, degree, *, affine_fast: bool = False, diagnostics: dict | None = None):
    """Fit y ≈ c0 + c1 f + ... + cd f^d with normalized f."""
    degree_i = int(degree)
    started = time.perf_counter()
    _diag_inc(diagnostics, "fit_poly_calls")
    _diag_inc(diagnostics, f"fit_poly_degree{degree_i}_calls")
    try:
        if not torch.isfinite(pred).all():
            return None
        if bool(affine_fast) and degree_i == 1:
            fast_started = time.perf_counter()
            try:
                return _fit_affine_fast(pred, y)
            finally:
                _diag_inc(diagnostics, "fit_poly_affine_fast_calls")
                _diag_add_time(
                    diagnostics,
                    "fit_poly_affine_fast_wall_seconds",
                    time.perf_counter() - fast_started,
                )

        f = pred.squeeze(-1)
        mu = float(f.mean())
        std = float(f.std())
        if std < 1e-12:
            _diag_inc(diagnostics, "fit_poly_constant_calls")
            c = pred.new_zeros(degree_i + 1)
            c[0] = y.squeeze(-1).mean()
            if not torch.isfinite(c).all():
                return None
            return c, mu, 1.0
        fn = (f - mu) / std
        cols = [torch.ones_like(fn)]
        for _ in range(1, degree_i + 1):
            cols.append(cols[-1] * fn)
        A = torch.stack(cols, dim=1)
        lstsq_started = time.perf_counter()
        try:
            sol = torch.linalg.lstsq(A, y.squeeze(-1)).solution
        finally:
            _diag_inc(diagnostics, "fit_poly_lstsq_calls")
            _diag_add_time(
                diagnostics,
                "fit_poly_lstsq_wall_seconds",
                time.perf_counter() - lstsq_started,
            )
        if not torch.isfinite(sol).all():
            return None
        try:
            stlsq_started = time.perf_counter()
            try:
                sol_sparse, meta = stlsq_sparsify_poly_coeffs(
                    Phi=A,
                    y=y.squeeze(-1),
                    coeffs=sol,
                    cfg=DEFAULT_POLY_STLSQ_CFG,
                )
            finally:
                _diag_inc(diagnostics, "fit_poly_stlsq_calls")
                _diag_add_time(
                    diagnostics,
                    "fit_poly_stlsq_wall_seconds",
                    time.perf_counter() - stlsq_started,
                )
            _log_sparsify_result("fit_poly", sol, None, sol_sparse, None, meta)
            sol = sol_sparse
        except Exception as exc:
            _log.debug("[fit_poly] poly sparsify failed: %s", exc)
        return sol, mu, std
    finally:
        _diag_add_time(diagnostics, "fit_poly_wall_seconds", time.perf_counter() - started)


def eval_poly(pred, coeffs, mu, std):
    """Evaluate polynomial mapping via Horner's method."""
    f = pred.squeeze(-1)
    fn = (f - mu) / std
    out = torch.zeros_like(fn)
    for k in range(len(coeffs) - 1, -1, -1):
        out = coeffs[k] + fn * out
    return out.unsqueeze(-1)


def fit_power(pred, y):
    """Fit y ≈ a * f^b under a consistent-sign assumption."""
    if not torch.isfinite(pred).all():
        return None
    f = pred.squeeze(-1)
    ys = y.squeeze(-1)

    if (ys > 0).all():
        sgn_y = 1.0
    elif (ys < 0).all():
        sgn_y = -1.0
    else:
        return None

    if (f > 0).all():
        sgn_f = 1.0
    elif (f < 0).all():
        sgn_f = -1.0
    else:
        return None

    f_pos = sgn_f * f
    y_pos = sgn_y * ys
    if (f_pos <= 1e-30).any() or (y_pos <= 1e-30).any():
        return None

    log_f = torch.log(f_pos)
    log_y = torch.log(y_pos)
    A = torch.stack([torch.ones_like(log_f), log_f], dim=1)
    sol = torch.linalg.lstsq(A, log_y).solution
    if not torch.isfinite(sol).all():
        return None
    mu = float(f.mean())
    std = float(f.std())
    if not math.isfinite(std) or std < 1e-12:
        std = 1.0
    return {
        "kind": "power",
        "log_a": float(sol[0]),
        "b": float(sol[1]),
        "mu": mu,
        "std": std,
        "sgn_f": float(sgn_f),
        "sgn_y": float(sgn_y),
    }


def eval_power(pred, mapping):
    """Evaluate power-law mapping y = sgn_y * exp(log_a + b*log(sgn_f*f))."""
    f = pred.squeeze(-1)
    sgn_f = float(mapping.get("sgn_f", 1.0))
    sgn_y = float(mapping.get("sgn_y", 1.0))
    f_pos = sgn_f * f
    y_hat = sgn_y * torch.exp(mapping["log_a"] + mapping["b"] * torch.log(f_pos))
    return y_hat.unsqueeze(-1)


def fit_pade(pred, y, numer_deg=2, denom_deg=2, n_iters=10):
    """Fit Padé [numer_deg/denom_deg] via Sanathanan-Koerner iteration."""
    if not torch.isfinite(pred).all():
        return None
    f = pred.squeeze(-1)
    ys = y.squeeze(-1)
    mu = float(f.mean())
    std = float(f.std())
    if std < 1e-12:
        return None
    fn = (f - mu) / std
    w = torch.ones_like(fn)
    p_coeffs = None
    q_coeffs = None
    for _ in range(n_iters):
        cols = []
        fpow = torch.ones_like(fn)
        for _ in range(numer_deg + 1):
            cols.append(fpow * w)
            fpow = fpow * fn
        fpow_d = fn.clone()
        for _ in range(1, denom_deg + 1):
            cols.append(-ys * fpow_d * w)
            fpow_d = fpow_d * fn
        A = torch.stack(cols, dim=1)
        rhs = ys * w
        sol = torch.linalg.lstsq(A, rhs).solution
        if not torch.isfinite(sol).all():
            return None
        p_coeffs = sol[: numer_deg + 1]
        q_coeffs = sol[numer_deg + 1 :]
        den = torch.ones_like(fn)
        fpow_d = fn.clone()
        for k in range(denom_deg):
            den = den + q_coeffs[k] * fpow_d
            fpow_d = fpow_d * fn
        if (den.abs() < 1e-10).any():
            return None
        w = 1.0 / den.abs()

    den_final = torch.ones_like(fn)
    fpow_d = fn.clone()
    for k in range(denom_deg):
        den_final = den_final + q_coeffs[k] * fpow_d
        fpow_d = fpow_d * fn
    if (den_final.abs() < 1e-6).any():
        return None

    numer = p_coeffs
    denom = torch.cat([torch.ones(1, dtype=pred.dtype, device=pred.device), q_coeffs])

    try:
        cols_num = []
        fpow = torch.ones_like(fn)
        for _ in range(numer_deg + 1):
            cols_num.append(fpow)
            fpow = fpow * fn
        Phi_num = torch.stack(cols_num, dim=1)

        cols_den = []
        fpow = torch.ones_like(fn)
        for _ in range(denom_deg + 1):
            cols_den.append(fpow)
            fpow = fpow * fn
        Phi_den = torch.stack(cols_den, dim=1)

        num_sparse, den_sparse, meta = stlsq_sparsify_rational_coeffs(
            Phi_num=Phi_num,
            Phi_den=Phi_den,
            y=ys,
            coeffs_num=numer,
            coeffs_den=denom,
            cfg=DEFAULT_RAT_STLSQ_CFG,
        )
        _log_sparsify_result("fit_pade", numer, denom, num_sparse, den_sparse, meta)
        numer = num_sparse
        denom = den_sparse
    except Exception as exc:
        _log.debug("[fit_pade] rational sparsify failed: %s", exc)

    den_final = torch.zeros_like(fn)
    fpow_d = torch.ones_like(fn)
    for k in range(len(denom)):
        den_final = den_final + denom[k] * fpow_d
        fpow_d = fpow_d * fn
    if (den_final.abs() < 1e-6).any():
        return None

    return {"kind": "pade", "numer": numer, "denom": denom, "mu": mu, "std": std}


def eval_pade(pred, mapping):
    """Evaluate Padé mapping."""
    f = pred.squeeze(-1)
    fn = (f - mapping["mu"]) / mapping["std"]
    numer = mapping["numer"]
    denom = mapping["denom"]
    num = torch.zeros_like(fn)
    fpow = torch.ones_like(fn)
    for k in range(len(numer)):
        num = num + numer[k] * fpow
        fpow = fpow * fn
    den = torch.zeros_like(fn)
    fpow = torch.ones_like(fn)
    for k in range(len(denom)):
        den = den + denom[k] * fpow
        fpow = fpow * fn
    y_hat = num / den
    return y_hat.unsqueeze(-1)


def _sine_sweep_batch(z, ys, omegas):
    """Solve for (A, B, c) at all omegas in one batched lstsq."""
    M = len(omegas)
    wz = omegas[:, None] * z[None, :]
    A = torch.stack([torch.sin(wz), torch.cos(wz), torch.ones_like(wz)], dim=2)
    rhs = ys[None, :, None].expand(M, -1, 1)
    sol = torch.linalg.lstsq(A, rhs).solution
    residual = rhs - A @ sol
    mses = (residual * residual).mean(dim=1).squeeze(-1)
    valid = torch.isfinite(mses) & torch.isfinite(sol.squeeze(-1)).all(dim=1)
    if not valid.any():
        return float("inf"), None, None
    mses[~valid] = float("inf")
    idx = int(mses.argmin())
    return float(mses[idx]), float(omegas[idx]), sol[idx, :, 0]


def fit_sine(pred, y, n_omega=40, omega_lo=0.5, omega_hi=30.0, n_refine=8):
    """Fit y ≈ A*sin(w*z) + B*cos(w*z) + c where z = (f - mu)/std."""
    if not torch.isfinite(pred).all():
        return None
    f = pred.squeeze(-1)
    mu = float(f.mean())
    std = float(f.std())
    if std < 1e-12:
        return None
    z = (f - mu) / std
    ys = y.squeeze(-1)

    best_mse, best_omega, best_sol = _sine_sweep_batch(
        z,
        ys,
        torch.logspace(
            math.log10(omega_lo),
            math.log10(omega_hi),
            n_omega,
            dtype=pred.dtype,
            device=pred.device,
        ),
    )
    if best_omega is None:
        return None

    span = best_omega * 0.3
    for _ in range(n_refine):
        lo = max(omega_lo, best_omega - span)
        hi = min(omega_hi, best_omega + span)
        mse, omega, sol = _sine_sweep_batch(
            z,
            ys,
            torch.linspace(lo, hi, n_omega, dtype=pred.dtype, device=pred.device),
        )
        if omega is not None and mse < best_mse:
            best_mse, best_omega, best_sol = mse, omega, sol
        span *= 0.3

    if best_omega > 30.0:
        return None
    return {
        "kind": "sine",
        "A": float(best_sol[0]),
        "B": float(best_sol[1]),
        "c": float(best_sol[2]),
        "omega": best_omega,
        "mu": mu,
        "std": std,
    }


def eval_sine(pred, mapping):
    """Evaluate sinusoidal mapping y = A*sin(w*z) + B*cos(w*z) + c."""
    f = pred.squeeze(-1)
    z = (f - mapping["mu"]) / mapping["std"]
    wz = mapping["omega"] * z
    y_hat = mapping["A"] * torch.sin(wz) + mapping["B"] * torch.cos(wz) + mapping["c"]
    return y_hat.unsqueeze(-1)


def _exp_sweep_batch(z, ys, bs):
    """Solve for (a, c) at all b values in one batched lstsq."""
    M = len(bs)
    bz = (bs[:, None] * z[None, :]).clamp(-20, 20)
    ebz = torch.exp(bz)
    A = torch.stack([ebz, torch.ones_like(ebz)], dim=2)
    rhs = ys[None, :, None].expand(M, -1, 1)
    sol = torch.linalg.lstsq(A, rhs).solution
    residual = rhs - A @ sol
    mses = (residual * residual).mean(dim=1).squeeze(-1)
    valid = torch.isfinite(mses) & torch.isfinite(sol.squeeze(-1)).all(dim=1)
    if not valid.any():
        return float("inf"), None, None
    mses[~valid] = float("inf")
    idx = int(mses.argmin())
    return float(mses[idx]), float(bs[idx]), sol[idx, :, 0]


def fit_exp_mapping(pred, y, n_b=40, b_max=5.0, n_refine=5):
    """Fit y ≈ a*exp(b*z) + c where z = (f - mu)/std."""
    if not torch.isfinite(pred).all():
        return None
    f = pred.squeeze(-1)
    mu = float(f.mean())
    std = float(f.std())
    if std < 1e-12:
        return None
    z = (f - mu) / std
    ys = y.squeeze(-1)

    b_pos = torch.logspace(
        math.log10(0.1),
        math.log10(b_max),
        n_b,
        dtype=pred.dtype,
        device=pred.device,
    )
    b_cands = torch.cat([-b_pos.flip(0), b_pos])
    best_mse, best_b, best_sol = _exp_sweep_batch(z, ys, b_cands)
    if best_b is None:
        return None

    span = max(abs(best_b) * 0.3, 0.2)
    for _ in range(n_refine):
        lo = best_b - span
        hi = best_b + span
        mse, b, sol = _exp_sweep_batch(
            z,
            ys,
            torch.linspace(lo, hi, n_b, dtype=pred.dtype, device=pred.device),
        )
        if b is not None and mse < best_mse:
            best_mse, best_b, best_sol = mse, b, sol
        span *= 0.3

    return {
        "kind": "exp",
        "a": float(best_sol[0]),
        "b": best_b,
        "c": float(best_sol[1]),
        "mu": mu,
        "std": std,
    }


def eval_exp_mapping(pred, mapping):
    """Evaluate exponential mapping y = a*exp(b*z) + c."""
    f = pred.squeeze(-1)
    z = (f - mapping["mu"]) / mapping["std"]
    bz = mapping["b"] * z
    y_hat = mapping["a"] * torch.exp(bz) + mapping["c"]
    return y_hat.unsqueeze(-1)


def _mapping_nparams(m):
    """Return the effective number of free parameters for a mapping."""
    kind = m.get("kind", "")
    if kind == "poly":
        n = len(m.get("coeffs", []))
    elif kind == "power":
        n = 2
    elif kind == "pade":
        n = len(m.get("numer", [])) + len(m.get("denom", [])) - 1
    elif kind == "sine":
        n = 4
    elif kind == "exp":
        n = 3
    else:
        n = 99

    head = m.get("_lin_head", None)
    if isinstance(head, dict):
        coeffs = head.get("coeffs", None)
        if isinstance(coeffs, (list, tuple)):
            n += len(coeffs)

    return n


def _normalize_mapping_family_mode(mode: str | None) -> str:
    mode_name = str(mode or "full").strip().lower()
    if mode_name in ("", "full", "all"):
        return "full"
    if mode_name in ("poly_only", "poly", "linear_only", "affine_only"):
        return "poly_only"
    if mode_name in ("cheap", "light"):
        return "cheap"
    if mode_name in ("gated", "gate", "cheap_gated", "prefit"):
        return "gated"
    return "full"


def _mapping_hint_matches(family_hint: str | None, family_name: str) -> bool:
    hint = str(family_hint or "").strip().lower()
    if not hint:
        return False
    if family_name == "sine":
        return any(tok in hint for tok in ("periodic", "trig", "sin", "cos"))
    if family_name == "exp":
        return any(tok in hint for tok in ("exp", "log"))
    return False


def _should_try_expensive_family(
    *,
    family_mode: str,
    family_name: str,
    family_hint: str | None,
    cheap_best_mse: float,
    gate_best_mse: float | None,
    y_var: float,
    gate_best_factor: float,
    gate_rel_y: float,
) -> bool:
    mode_name = _normalize_mapping_family_mode(family_mode)
    if mode_name == "full":
        return True
    if mode_name == "cheap":
        return False
    if _mapping_hint_matches(family_hint, family_name):
        return True
    if math.isfinite(float(cheap_best_mse)) and math.isfinite(float(y_var)):
        if float(cheap_best_mse) <= max(1.0e-30, float(y_var)) * max(0.0, float(gate_rel_y)):
            return True
    if gate_best_mse is not None and math.isfinite(float(gate_best_mse)) and math.isfinite(float(cheap_best_mse)):
        return float(cheap_best_mse) <= max(1.0e-30, float(gate_best_mse)) * max(1.0, float(gate_best_factor))
    return False


def mapping_is_structural(mapping):
    """Return True if a mapping is considered structural for solved-gating."""
    m = mapping or {}
    kind_raw = m.get("kind", "")
    kind = str(kind_raw).strip().lower() if kind_raw is not None else ""
    if kind in ("", "identity"):
        return True
    if kind in ("monomial", "mono", "affine"):
        return True
    if kind == "poly":
        deg = max(0, len(m.get("coeffs", [])) - 1)
        return deg <= 1
    return kind in ("power", "sine", "exp")


def fit_best(
    pred,
    y,
    poly_degree,
    *,
    pred_probe=None,
    family_mode: str = "full",
    expensive_gate_best_mse: float | None = None,
    expensive_gate_best_factor: float = 5.0,
    expensive_gate_rel_y: float = 0.10,
    family_hint: str | None = None,
    affine_fast: bool = False,
    diagnostics: dict | None = None,
):
    """Return the simplest mapping with comparable error, or the best one."""
    family_mode = _normalize_mapping_family_mode(family_mode)
    candidates = []
    y_var = max(float((y ** 2).mean()), 1e-30)

    def _probe_valid(mapping: dict) -> bool:
        if pred_probe is None:
            return True
        if not torch.is_tensor(pred_probe) or not torch.isfinite(pred_probe).all():
            return False
        kind = str(mapping.get("kind", "") or "")
        try:
            if kind == "poly":
                y_hat_probe = eval_poly(pred_probe, mapping["coeffs"], mapping["mu"], mapping["std"])
            elif kind == "power":
                y_hat_probe = eval_power(pred_probe, mapping)
            elif kind == "pade":
                y_hat_probe = eval_pade(pred_probe, mapping)
            elif kind == "sine":
                y_hat_probe = eval_sine(pred_probe, mapping)
            elif kind == "exp":
                y_hat_probe = eval_exp_mapping(pred_probe, mapping)
            else:
                return False
        except Exception:
            return False
        return bool(torch.isfinite(y_hat_probe).all())

    pf = fit_poly(pred, y, poly_degree, affine_fast=affine_fast, diagnostics=diagnostics)
    if pf is not None:
        coeffs, mu, std = pf
        y_hat = eval_poly(pred, coeffs, mu, std)
        mse = mean_squared_error_same_shape(y, y_hat)
        if math.isfinite(mse):
            best_poly_deg = len(coeffs) - 1
            best_poly = (mse, {"kind": "poly", "coeffs": coeffs, "mu": mu, "std": std})
            for deg in range(1, best_poly_deg):
                pf_lo = fit_poly(pred, y, deg, affine_fast=affine_fast, diagnostics=diagnostics)
                if pf_lo is None:
                    continue
                c_lo, mu_lo, std_lo = pf_lo
                y_lo = eval_poly(pred, c_lo, mu_lo, std_lo)
                mse_lo = mean_squared_error_same_shape(y, y_lo)
                if math.isfinite(mse_lo) and mse_lo <= max(mse * 2.0, float((y ** 2).mean()) * 1e-8):
                    best_poly = (mse_lo, {"kind": "poly", "coeffs": c_lo, "mu": mu_lo, "std": std_lo})
                    break
            if _probe_valid(best_poly[1]):
                candidates.append(best_poly)

    if family_mode == "poly_only":
        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[0], _mapping_nparams(t[1])))
        return candidates[0]

    pw = fit_power(pred, y)
    if pw is not None:
        y_hat = eval_power(pred, pw)
        if torch.isfinite(y_hat).all():
            mse = mean_squared_error_same_shape(y, y_hat)
            if math.isfinite(mse):
                if _probe_valid(pw):
                    candidates.append((mse, pw))

    pa = fit_pade(pred, y)
    if pa is not None:
        y_hat = eval_pade(pred, pa)
        if torch.isfinite(y_hat).all():
            mse = mean_squared_error_same_shape(y, y_hat)
            if math.isfinite(mse):
                if _probe_valid(pa):
                    candidates.append((mse, pa))

    cheap_best_mse = min((float(mse) for mse, _ in candidates), default=float("inf"))
    gate_best_mse = None if expensive_gate_best_mse is None else float(expensive_gate_best_mse)

    if _should_try_expensive_family(
        family_mode=family_mode,
        family_name="sine",
        family_hint=family_hint,
        cheap_best_mse=cheap_best_mse,
        gate_best_mse=gate_best_mse,
        y_var=y_var,
        gate_best_factor=float(expensive_gate_best_factor),
        gate_rel_y=float(expensive_gate_rel_y),
    ):
        sn = fit_sine(pred, y)
        if sn is not None:
            y_hat = eval_sine(pred, sn)
            if torch.isfinite(y_hat).all():
                mse = mean_squared_error_same_shape(y, y_hat)
                if math.isfinite(mse):
                    if _probe_valid(sn):
                        candidates.append((mse, sn))

    if _should_try_expensive_family(
        family_mode=family_mode,
        family_name="exp",
        family_hint=family_hint,
        cheap_best_mse=cheap_best_mse,
        gate_best_mse=gate_best_mse,
        y_var=y_var,
        gate_best_factor=float(expensive_gate_best_factor),
        gate_rel_y=float(expensive_gate_rel_y),
    ):
        ex = fit_exp_mapping(pred, y)
        if ex is not None:
            y_hat = eval_exp_mapping(pred, ex)
            if torch.isfinite(y_hat).all():
                mse = mean_squared_error_same_shape(y, y_hat)
                if math.isfinite(mse):
                    if _probe_valid(ex):
                        candidates.append((mse, ex))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    best_mse = candidates[0][0]

    mse_tol = max(best_mse * 3.0, y_var * 1e-8)
    comparable = [(mse, m) for mse, m in candidates if mse <= mse_tol]
    if comparable:
        comparable.sort(key=lambda t: (_mapping_nparams(t[1]), t[0]))
        return comparable[0]
    return candidates[0]


def eval_mapping(pred, mapping):
    """Dispatch to the appropriate eval function based on mapping['kind']."""
    kind = mapping["kind"]
    if kind == "poly":
        return eval_poly(pred, mapping["coeffs"], mapping["mu"], mapping["std"])
    if kind == "power":
        return eval_power(pred, mapping)
    if kind == "pade":
        return eval_pade(pred, mapping)
    if kind == "sine":
        return eval_sine(pred, mapping)
    if kind == "exp":
        return eval_exp_mapping(pred, mapping)
    if kind == "basis_state_native":
        return pred
    raise ValueError(f"Unknown mapping kind: {kind}")
