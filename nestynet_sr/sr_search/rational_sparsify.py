# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""STLSQ-based sparsification helpers for rational polynomial coefficients.

This module is a lightweight "de-Padeifier" for fits of the form:

    y(x) ~= P(x) / Q(x)

where P and Q are linear in coefficient vectors. We reuse the STLSQ numerics
from :mod:`nestynet_sr.sr_core.numerics` to drop low-value terms while keeping
fit quality close to the dense baseline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from nestynet_sr.sr_core.bridges import ConstNode
from nestynet_sr.sr_core.numerics import ridge_lstsq, stlsq
from nestynet_sr.sr_search.model_selection import compute_accept_threshold

log = logging.getLogger(__name__)


@dataclass
class RationalSparsifyConfig:
    """Configuration for STLSQ rational coefficient pruning."""

    ridge: float = 1e-10
    max_iter: int = 10
    lam_abs: float = 0.0
    lam_rel: float = 1e-3
    pivot_eps: float = 1e-12
    prefer_den_constant: bool = True
    pivot_index: Optional[int] = None
    unbiased_refit: bool = False
    resid_increase_tolerance: float = 5.0
    greedy_prune: bool = True
    # Legacy fallback when proposal_mode=False.
    greedy_tolerance: float = 1.02
    greedy_max_drops: int = 32
    # Proposal mode: compare sparse vs dense via shared complexity-vs-accuracy
    # acceptance threshold policy (same primitive used by Stage B).
    proposal_mode: bool = True
    proposal_param_gamma: float = 0.30
    proposal_base_bonus_decades: float = 0.0
    # Base loss floor in MAD(y)^2 units; mirrors SearchConfig.loss_target default.
    proposal_loss_floor: float = 1.0e-7
    # Scale proposal_loss_floor by robust target scale (MAD(y)^2).
    proposal_scale_floor_with_y_mad: bool = True
    # Optional raw MSE floor for noisy-data sparsification comparisons.  This
    # keeps STLSQ pruning from treating irreducible noise as meaningful loss.
    proposal_noise_floor: float = 0.0
    proposal_loss_cap: float = float("inf")
    proposal_max_worsening_factor: Optional[float] = None
    proposal_hard_ceiling: Optional[float] = None
    # (B) Denominator evaluation mode for _prediction_mse.
    # When True, use q.clamp(min=eps) instead of |q|>eps masking so that
    # the MSE metric matches the behavior of clamp_min-based rational leaves
    # (RationalPolyLeaf, ExpRationalPolyLeaf).  Default True because most
    # callers are fitting leaves that use clamp_min.
    den_clamp_mode: bool = True
    # (C) Use effect-size (|c_i| * RMS(A[:,i])) instead of raw |c_i| for
    # the pre-freeze threshold.  More scale-aware for ND monomial bases
    # where column magnitudes vary by orders of magnitude.
    effect_size_threshold: bool = True
    # (D) When True (default), freeze seed terms below lam_eff before
    # running STLSQ — prevents support drift in ill-conditioned problems.
    # When False, run STLSQ on the full design matrix.
    freeze_below_lambda: bool = True


# Shared default configs — import these instead of re-creating per call site.
DEFAULT_RAT_STLSQ_CFG = RationalSparsifyConfig(
    ridge=1e-10,
    max_iter=10,
    lam_abs=0.0,
    lam_rel=1e-3,
    pivot_eps=1e-12,
    prefer_den_constant=True,
    unbiased_refit=False,
    resid_increase_tolerance=5.0,
)

DEFAULT_POLY_STLSQ_CFG = RationalSparsifyConfig(
    ridge=1e-10,
    max_iter=10,
    lam_abs=0.0,
    lam_rel=1e-3,
    pivot_eps=1e-12,
    prefer_den_constant=True,
    pivot_index=0,
    unbiased_refit=False,
    resid_increase_tolerance=5.0,
)


# ---------------------------------------------------------------------------
# Coefficient summary helpers (for logging before/after sparsification)
# ---------------------------------------------------------------------------

def _coeffs_summary(coeffs: torch.Tensor, label: str = "coeffs") -> str:
    """One-line summary of a coefficient vector: nnz, max|c|, values."""
    n = int(coeffs.numel())
    if n == 0:
        return f"{label}: (empty)"
    nnz = int((coeffs.abs() > 1e-14).sum().item())
    maxabs = float(coeffs.abs().max().item())
    vals = ", ".join(f"{float(c):.4g}" for c in coeffs)
    return f"{label}[{nnz}/{n} active, max|c|={maxabs:.3g}]: [{vals}]"


def _active_indices_summary(coeffs: torch.Tensor, eps: float = 1e-14) -> str:
    """Compact summary of active coefficient indices."""
    if coeffs.numel() == 0:
        return "[]"
    idx = torch.nonzero(coeffs.abs() > float(eps), as_tuple=False).view(-1).tolist()
    if len(idx) <= 24:
        return str([int(i) for i in idx])
    head = ", ".join(str(int(i)) for i in idx[:12])
    tail = ", ".join(str(int(i)) for i in idx[-12:])
    return f"[{head}, ..., {tail}] (n={len(idx)})"


def _emit_sparsify_line(msg: str, *args: object) -> None:
    """Emit a sparsify log line when INFO logging is enabled."""
    if log.isEnabledFor(logging.INFO):
        log.info(msg, *args)
        return
    # The stdout fallback is intentionally disabled to keep library calls quiet.


def _log_sparsify_result(
    caller: str,
    coeffs_num_before: torch.Tensor,
    coeffs_den_before: Optional[torch.Tensor],
    coeffs_num_after: torch.Tensor,
    coeffs_den_after: Optional[torch.Tensor],
    meta: Dict[str, float],
) -> None:
    """Log before/after coefficient vectors and key metrics at INFO level."""
    accepted = meta.get("accepted", 0.0)
    mse_seed = meta.get("mse_seed", float("nan"))
    mse_sparse = meta.get("mse_sparse", float("nan"))
    nnz_seed = meta.get("nnz_seed", float("nan"))
    nnz_sparse = meta.get("nnz_sparse", float("nan"))
    status = "ACCEPTED" if accepted > 0.5 else "REJECTED (kept dense)"
    _emit_sparsify_line(
        "[%s] Sparsify %s: nnz %g -> %g, MSE %.3e -> %.3e",
        caller, status, nnz_seed, nnz_sparse, mse_seed, mse_sparse,
    )
    _emit_sparsify_line("  BEFORE num: %s", _coeffs_summary(coeffs_num_before, "a"))
    _emit_sparsify_line("  AFTER  num: %s", _coeffs_summary(coeffs_num_after, "a"))
    _emit_sparsify_line(
        "  ACTIVE idx num: %s -> %s",
        _active_indices_summary(coeffs_num_before),
        _active_indices_summary(coeffs_num_after),
    )
    if coeffs_den_before is not None:
        _emit_sparsify_line("  BEFORE den: %s", _coeffs_summary(coeffs_den_before, "b"))
    if coeffs_den_after is not None:
        _emit_sparsify_line("  AFTER  den: %s", _coeffs_summary(coeffs_den_after, "b"))
        _emit_sparsify_line(
            "  ACTIVE idx den: %s -> %s",
            _active_indices_summary(coeffs_den_before)
            if coeffs_den_before is not None
            else "[]",
            _active_indices_summary(coeffs_den_after),
        )


_DUMMY_AST = ConstNode(1.0)


def _proposal_threshold(
    *,
    base_mse: float,
    base_params: int,
    cand_params: int,
    loss_floor: float,
    loss_cap: float,
    cfg: RationalSparsifyConfig,
) -> float:
    """Reuse shared complexity-vs-accuracy threshold logic for coefficient proposals."""
    return float(
        compute_accept_threshold(
            base_loss=float(base_mse),
            best_loss=float(base_mse),
            base_ast=_DUMMY_AST,
            cand_ast=_DUMMY_AST,
            base_params=int(max(1, base_params)),
            cand_params=int(max(1, cand_params)),
            loss_floor=float(loss_floor),
            loss_cap=float(loss_cap),
            count_weight=1.0,
            struct_gamma=0.0,  # same AST, only coefficient-count simplification matters here
            param_gamma=float(cfg.proposal_param_gamma),
            base_bonus_decades=float(cfg.proposal_base_bonus_decades),
            sep_bonus_decades=0.0,
            partial_sep_bonus_decades=0.0,
            is_separability=False,
            is_partial_separability=False,
            extra_bonus_decades=0.0,
            max_worsening_factor=cfg.proposal_max_worsening_factor,
            worsening_floor=None,
            hard_ceiling=cfg.proposal_hard_ceiling,
            noise_floor=float(cfg.proposal_noise_floor),
        )
    )


def _choose_denominator_pivot(
    coeffs_den: torch.Tensor,
    cfg: RationalSparsifyConfig,
) -> Optional[int]:
    """Pick denominator pivot index for gauge fixing."""
    n_den = int(coeffs_den.numel())
    if n_den == 0:
        return None

    eps = float(cfg.pivot_eps)
    if cfg.pivot_index is not None:
        j = int(cfg.pivot_index)
        if 0 <= j < n_den and abs(float(coeffs_den[j])) >= eps:
            return j

    if bool(cfg.prefer_den_constant) and abs(float(coeffs_den[0])) >= eps:
        return 0

    abs_b = coeffs_den.abs()
    j = int(abs_b.argmax().item())
    if float(abs_b[j]) < eps:
        return None
    return j


def stlsq_sparsify_rational_coeffs(
    Phi_num: torch.Tensor,
    Phi_den: torch.Tensor,
    y: torch.Tensor,
    coeffs_num: torch.Tensor,
    coeffs_den: torch.Tensor,
    *,
    cfg: Optional[RationalSparsifyConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Sparsify rational coefficients with STLSQ on a gauge-fixed linear system.

    Parameters
    ----------
    Phi_num : torch.Tensor
        Numerator design matrix, shape (N, M_num).
    Phi_den : torch.Tensor
        Denominator design matrix, shape (N, M_den).
    y : torch.Tensor
        Target vector, shape (N,) or (N,1).
    coeffs_num : torch.Tensor
        Dense numerator coefficients, shape (M_num,).
    coeffs_den : torch.Tensor
        Dense denominator coefficients, shape (M_den,).
    cfg : RationalSparsifyConfig, optional
        STLSQ and acceptance parameters.

    Returns
    -------
    (a_sparse, b_sparse, meta)
        `a_sparse` and `b_sparse` have the same shapes as inputs.
        `meta` includes simple diagnostics (pivot, lambdas, residuals, counts).
    """
    cfg = cfg or RationalSparsifyConfig()

    if Phi_num.ndim != 2 or Phi_den.ndim != 2:
        raise ValueError("Phi_num and Phi_den must be rank-2 matrices")
    if Phi_num.shape[0] != Phi_den.shape[0]:
        raise ValueError("Phi_num and Phi_den must have the same row count")

    N = int(Phi_num.shape[0])
    if N == 0:
        return coeffs_num.clone(), coeffs_den.clone(), {"accepted": 0.0}

    yv = y.view(-1)
    if int(yv.numel()) != N:
        raise ValueError(f"y has {int(yv.numel())} entries but expected {N}")

    if int(coeffs_num.numel()) != int(Phi_num.shape[1]):
        raise ValueError("coeffs_num shape mismatch")
    if int(coeffs_den.numel()) != int(Phi_den.shape[1]):
        raise ValueError("coeffs_den shape mismatch")

    if int(coeffs_den.numel()) == 0:
        return coeffs_num.clone(), coeffs_den.clone(), {"accepted": 0.0}

    pivot = _choose_denominator_pivot(coeffs_den, cfg)
    if pivot is None:
        return coeffs_num.clone(), coeffs_den.clone(), {"accepted": 0.0}

    pivot_val = coeffs_den[pivot]
    if abs(float(pivot_val)) < float(cfg.pivot_eps):
        return coeffs_num.clone(), coeffs_den.clone(), {"accepted": 0.0}

    # Gauge-fix by setting denominator pivot coefficient to +1.
    a0 = coeffs_num / pivot_val
    b0 = coeffs_den / pivot_val

    M_num = int(Phi_num.shape[1])
    M_den = int(Phi_den.shape[1])
    other = [j for j in range(M_den) if j != pivot]

    rhs = yv * Phi_den[:, pivot]
    if other:
        Phi_den_other = Phi_den[:, other]
        A = torch.cat([Phi_num, -(yv.unsqueeze(1) * Phi_den_other)], dim=1)
        theta_seed = torch.cat([a0, b0[other]], dim=0)
    else:
        A = Phi_num
        theta_seed = a0

    if int(A.shape[1]) == 0:
        return coeffs_num.clone(), coeffs_den.clone(), {"accepted": 0.0}

    with torch.no_grad():
        y_center = yv - yv.median()
        y_mad = float(y_center.abs().median().item())
        if not math.isfinite(y_mad) or y_mad <= 0.0:
            y_mad = float(yv.abs().median().item())
        if not math.isfinite(y_mad) or y_mad <= 0.0:
            y_mad = 1.0
    if bool(cfg.proposal_scale_floor_with_y_mad):
        loss_floor_eff = float(cfg.proposal_loss_floor) * float(y_mad * y_mad)
    else:
        loss_floor_eff = float(cfg.proposal_loss_floor)
    if (not math.isfinite(loss_floor_eff)) or loss_floor_eff < 0.0:
        loss_floor_eff = 0.0
    loss_cap_eff = float(cfg.proposal_loss_cap)
    if not math.isfinite(loss_cap_eff):
        loss_cap_eff = float("inf")

    eps_pred = max(float(cfg.pivot_eps), 1.0e-8)

    def _theta_to_coeffs(theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aa = theta[:M_num].clone()
        bb = coeffs_den.new_zeros(M_den)
        bb[pivot] = 1.0
        if other:
            bb[other] = theta[M_num:]
        return aa, bb

    use_clamp = bool(cfg.den_clamp_mode)

    def _prediction_mse(aa: torch.Tensor, bb: torch.Tensor) -> float:
        q = Phi_den @ bb
        if use_clamp:
            # Match clamp_min-based leaf behavior: Q is clamped to eps,
            # so negative Q regions evaluate as eps (not masked out).
            q = q.clamp(min=eps_pred)
            pred = (Phi_num @ aa) / q
            err = pred - yv
        else:
            mask = q.abs() > eps_pred
            if int(mask.sum().item()) < max(10, int(0.1 * N)):
                return float("inf")
            pred = (Phi_num @ aa) / q
            err = pred[mask] - yv[mask]
        mse = float((err.square().mean()).item())
        if not math.isfinite(mse):
            return float("inf")
        return mse

    # Compute per-column effect size: |c_i| * RMS(A[:,i]).
    # More scale-aware than raw |c_i| for ND monomial bases where column
    # magnitudes vary by orders of magnitude.
    with torch.no_grad():
        col_rms = A.square().mean(dim=0).sqrt().clamp_min(1e-30)  # (K,)

    if bool(cfg.effect_size_threshold):
        effect = theta_seed.abs() * col_rms
    else:
        effect = theta_seed.abs()

    lam_eff = float(cfg.lam_abs)
    if float(cfg.lam_rel) > 0.0:
        maxeff = float(effect.max().item()) if effect.numel() > 0 else 0.0
        lam_eff = max(lam_eff, float(cfg.lam_rel) * maxeff)

    if bool(cfg.freeze_below_lambda):
        # Freeze obviously tiny seed terms so STLSQ cannot drift into an
        # equally-good but denser support due to null-space ambiguity.
        seed_keep = effect >= lam_eff
        if int(seed_keep.sum().item()) == 0:
            kmax = int(effect.argmax().item())
            seed_keep = torch.zeros_like(seed_keep, dtype=torch.bool)
            seed_keep[kmax] = True

        keep_idx = torch.nonzero(seed_keep, as_tuple=False).view(-1)
        A_work = A[:, keep_idx]
    else:
        # Run STLSQ on the full design matrix.
        keep_idx = torch.arange(int(A.shape[1]), device=A.device)
        A_work = A
        seed_keep = torch.ones(int(A.shape[1]), dtype=torch.bool, device=A.device)

    # STLSQ thresholds in coefficient space, so convert effect-size lambda
    # back to an equivalent coefficient-scale lambda for STLSQ's own
    # thresholding (which already does its own column-scaling internally).
    if bool(cfg.effect_size_threshold):
        # Use a representative coefficient-scale lambda: the relative
        # threshold times the max |coefficient|.
        maxabs = float(theta_seed.abs().max().item()) if theta_seed.numel() > 0 else 0.0
        lam_stlsq = max(float(cfg.lam_abs), float(cfg.lam_rel) * maxabs)
    else:
        lam_stlsq = lam_eff

    theta_work, keep_work = stlsq(
        A_work,
        rhs,
        ridge=float(cfg.ridge),
        lam=lam_stlsq,
        max_iter=int(cfg.max_iter),
    )
    theta_sparse = theta_seed.new_zeros(int(A.shape[1]))
    theta_sparse[keep_idx] = theta_work

    keep_full = torch.zeros_like(theta_sparse, dtype=torch.bool)
    keep_full[keep_idx] = keep_work

    if bool(cfg.unbiased_refit) and int(keep_full.sum().item()) > 0:
        theta_refit = theta_sparse.clone()
        theta_refit.zero_()
        theta_sel = ridge_lstsq(A[:, keep_full], rhs, ridge=float(cfg.ridge))
        theta_refit[keep_full] = theta_sel
        theta_sparse = theta_refit

    # Helper: build an "active" mask that respects effect-size mode.
    # When effect_size_threshold is True, the significance of coefficient i
    # is |c_i| * col_rms[i], and lam_eff lives in that same space.
    def _active_mask(theta: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
        lam_half = max(lam_eff * frac, 1e-14)
        if bool(cfg.effect_size_threshold):
            return (theta.abs() * col_rms) >= lam_half
        return theta.abs() >= lam_half

    if bool(cfg.greedy_prune):
        theta_cur = theta_sparse.clone()
        active = _active_mask(theta_cur)
        if int(active.sum().item()) == 0:
            kmax = int(theta_cur.abs().argmax().item())
            active[kmax] = True

        with torch.no_grad():
            a_cur, b_cur = _theta_to_coeffs(theta_cur)
            base_mse = _prediction_mse(a_cur, b_cur)

        n_drops = 0
        while n_drops < int(cfg.greedy_max_drops):
            active_idx = torch.nonzero(active, as_tuple=False).view(-1)
            if int(active_idx.numel()) <= 1:
                break

            best_mse = None
            best_theta = None
            best_active = None

            n_num_active = int(active[:M_num].sum().item())
            for j in active_idx.tolist():
                # Keep at least one numerator feature active.
                if j < M_num and n_num_active <= 1:
                    continue

                cand_active = active.clone()
                cand_active[j] = False
                if int(cand_active.sum().item()) == 0:
                    continue

                theta_c = theta_cur.new_zeros(theta_cur.numel())
                coeff_c = ridge_lstsq(A[:, cand_active], rhs, ridge=float(cfg.ridge))
                theta_c[cand_active] = coeff_c
                a_c, b_c = _theta_to_coeffs(theta_c)
                mse_c = _prediction_mse(a_c, b_c)

                if best_mse is None or mse_c < best_mse:
                    best_mse = mse_c
                    best_theta = theta_c
                    best_active = cand_active

            if best_mse is None or best_theta is None or best_active is None:
                break

            if bool(cfg.proposal_mode):
                n_base = int(active.sum().item())
                n_cand = int(best_active.sum().item())
                thr = _proposal_threshold(
                    base_mse=float(base_mse),
                    base_params=n_base,
                    cand_params=n_cand,
                    loss_floor=float(loss_floor_eff),
                    loss_cap=float(loss_cap_eff),
                    cfg=cfg,
                )
                accept_drop = bool(math.isfinite(best_mse)) and (best_mse <= thr)
            else:
                tol = base_mse * max(float(cfg.greedy_tolerance), 1.0) + 1e-20
                accept_drop = bool(math.isfinite(best_mse)) and (best_mse <= tol)

            if accept_drop:
                theta_cur = best_theta
                active = best_active
                base_mse = best_mse
                n_drops += 1
                continue
            break

        theta_sparse = theta_cur

    with torch.no_grad():
        a_seed, b_seed = _theta_to_coeffs(theta_seed)
        a_sparse, b_sparse = _theta_to_coeffs(theta_sparse)
        mse_seed = _prediction_mse(a_seed, b_seed)
        mse_sparse = _prediction_mse(a_sparse, b_sparse)
        n_seed = int(_active_mask(theta_seed).sum().item())
        n_sparse = int(_active_mask(theta_sparse).sum().item())
        if bool(cfg.proposal_mode):
            thr = _proposal_threshold(
                base_mse=float(mse_seed),
                base_params=n_seed,
                cand_params=n_sparse,
                loss_floor=float(loss_floor_eff),
                loss_cap=float(loss_cap_eff),
                cfg=cfg,
            )
            accept = bool(math.isfinite(mse_sparse)) and (mse_sparse <= thr)
        else:
            tol = mse_seed * max(float(cfg.resid_increase_tolerance), 1.0) + 1e-20
            accept = bool(math.isfinite(mse_sparse)) and (mse_sparse <= tol)

    theta_use = theta_sparse if accept else theta_seed

    a_out = theta_use[:M_num].clone()
    b_out = coeffs_den.new_zeros(M_den)
    b_out[pivot] = 1.0
    if other:
        b_out[other] = theta_use[M_num:]

    # Canonicalise to denominator constant 1 when available.
    if bool(cfg.prefer_den_constant) and M_den > 0:
        maxb = float(b_out.abs().max().item()) if b_out.numel() > 0 else 0.0
        min_ref = max(float(cfg.pivot_eps), 1e-3 * maxb)
        if abs(float(b_out[0])) >= min_ref:
            scl = b_out[0]
            a_out = a_out / scl
            b_out = b_out / scl

    # Keep denominator orientation stable for clamp_min-based leaves.
    with torch.no_grad():
        q_vals = Phi_den @ b_out
        frac_positive = float((q_vals > 0).sum().item()) / max(1, int(q_vals.numel()))
        if frac_positive < 0.5:
            a_out = -a_out
            b_out = -b_out
            q_vals = -q_vals
            frac_positive = 1.0 - frac_positive

    meta: Dict[str, float] = {
        "accepted": 1.0 if accept else 0.0,
        "pivot_index": float(pivot),
        "lambda_effective": float(lam_eff),
        "mse_seed": float(mse_seed),
        "mse_sparse": float(mse_sparse),
        "nnz_seed": float(int(_active_mask(theta_seed).sum().item())),
        "nnz_sparse": float(int(_active_mask(theta_sparse).sum().item())),
        "frac_positive_den": float(frac_positive),
    }
    return a_out, b_out, meta


def stlsq_sparsify_poly_coeffs(
    Phi: torch.Tensor,
    y: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    cfg: Optional[RationalSparsifyConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sparsify polynomial coefficients via the rational sparsifier with Q=1.

    This is the special-case wrapper for simple polynomial models
    ``y ~= P(x)`` (equivalent to ``P(x)/1``).
    """
    if Phi.ndim != 2:
        raise ValueError("Phi must be a rank-2 matrix")
    if int(Phi.shape[1]) != int(coeffs.numel()):
        raise ValueError("Phi/coeffs shape mismatch")

    cfg_use = cfg or RationalSparsifyConfig(
        ridge=1e-10,
        max_iter=10,
        lam_abs=0.0,
        lam_rel=1e-3,
        pivot_eps=1e-12,
        prefer_den_constant=True,
        pivot_index=0,
        unbiased_refit=False,
        resid_increase_tolerance=5.0,
        greedy_prune=True,
        greedy_tolerance=1.02,
        greedy_max_drops=32,
    )
    ones = Phi.new_ones(int(Phi.shape[0]), 1)
    den0 = coeffs.new_ones(1)
    c_sparse, _, meta = stlsq_sparsify_rational_coeffs(
        Phi_num=Phi,
        Phi_den=ones,
        y=y,
        coeffs_num=coeffs,
        coeffs_den=den0,
        cfg=cfg_use,
    )
    return c_sparse, meta
