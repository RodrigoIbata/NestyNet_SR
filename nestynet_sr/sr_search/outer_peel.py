# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Outer-function peeling proposals.

This module provides *proposal* heuristics for choosing an outer y-transform
even when classical add/mul separability does not appear.

The initial use-case is to robustly suggest the `square` transform for targets
that look like `sqrt(u(x))`, by detecting a dramatic simplification in the
input-space curvature when we form t(x)=f(x)^2.

Design goals
------------
1) Proposals, not commitments.
2) Use the already-fitted identity model as the "teacher" for ranking.
3) Keep heuristics conservative to avoid disrupting existing working cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .features import _cross_hess_rel, _hess_const_rel, _poly_fit_rms_rel, _scaling_rel_std
from .fitting_utils import _rational_probe_nd
from .y_transforms import get_separability_y_ops


@dataclass
class OuterPeelProposal:
    """A ranked suggestion for a y-transform."""

    name: str
    score: float
    improvement: float
    details: Dict


@dataclass
class OuterPeelDecision:
    """A conservative yes/no decision plus diagnostics."""

    prefer: bool
    proposal: OuterPeelProposal
    diagnostics: Dict


@dataclass
class TransformSimplicity:
    """Scored simplicity stats for a candidate y-transform φ(y).

    This is used for *ranking proposals* without committing to a transform.
    Scores are heuristic and intentionally conservative.
    """

    name: str
    domain_ok_frac: float
    n_points: int
    poly2_rms_rel: float
    rat_rms_rel: float
    hess_const_rel: float
    hess_diag_const_rel_min: float
    hess_diag_const_best_axis: Optional[int]
    scaling_rel_std: float
    cross_hess_rel: float
    axis_exp_int_score: float
    axis_exp_n_axes: int
    structure_screen_score: float
    score: float
    score_improvement: float
    details: Dict


def _mad_1d(x: torch.Tensor) -> torch.Tensor:
    """Median absolute deviation for a 1D tensor."""
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    med = x.median()
    return (x - med).abs().median()


def _collect_points_from_loader(
    datagen, device: torch.device, max_points: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    n = 0
    for batch in datagen:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Expected dataloader batches to be (x,y) tuples")
        xs.append(x)
        ys.append(y)
        n += x.shape[0]
        if n >= max_points:
            break
    if not xs:
        raise ValueError("Empty dataloader: no batches")
    x = torch.cat(xs, dim=0)[:max_points].to(device)
    y = torch.cat(ys, dim=0)[:max_points].to(device)
    return x, y


def score_square_from_identity(
    model: torch.nn.Module,
    datagen,
    device: torch.device,
    max_points: int = 2048,
    eps: float = 1e-12,
    abs_floor: float = 1e-8,
) -> OuterPeelProposal:
    """Score whether `square` is a good outer-peel candidate.

    Uses only the already-fitted *identity* model f(x) and its analytic
    derivatives. For t(x)=f(x)^2:
        ∇t = 2 f ∇f
        H_t = 2 f H_f + 2 (∇f)(∇f)^T
    We score how much more *constant* the diagonal of H_t becomes compared to
    the diagonal of H_f.
    """
    x, _ = _collect_points_from_loader(datagen, device=device, max_points=max_points)

    # Model value + input derivatives (analytic)
    with torch.no_grad():
        f = model(x)
    g = model.grad(x)
    h = model.grad_grad(x)

    # Slice scalar output if needed
    if f.dim() == 2:
        f = f[:, 0]
    else:
        f = f.view(-1)
    if g.dim() == 3:
        g = g[:, 0, :]
    if h.dim() == 4:
        h = h[:, 0, :, :]

    if h.dim() != 3:
        raise ValueError(f"Expected Hessian shape [N,Nx,Nx], got {tuple(h.shape)}")

    diag_f = h.diagonal(dim1=-2, dim2=-1)  # [N, Nx]
    diag_sq = 2.0 * f.unsqueeze(-1) * diag_f + 2.0 * (g * g)  # [N, Nx]

    Nx = diag_f.shape[1]
    axes = []
    spread_f = []
    spread_sq = []
    medabs_sq = []
    improvement = []
    axis_stats: List[Dict] = []

    for i in range(Nx):
        df = diag_f[:, i]
        ds = diag_sq[:, i]

        medabs_df = df.abs().median()
        medabs_ds = ds.abs().median()
        if not torch.isfinite(medabs_ds) or float(medabs_ds) <= abs_floor:
            continue

        mad_df = _mad_1d(df)
        mad_ds = _mad_1d(ds)

        # Normalised spreads: 0 => perfectly constant, larger => more variable
        sf = float(mad_df / (medabs_df + eps))
        ss = float(mad_ds / (medabs_ds + eps))

        # If sf is tiny (already constant), improvement is ~0 (won't trigger)
        imp = sf / (ss + eps)

        axes.append(i)
        spread_f.append(sf)
        spread_sq.append(ss)
        medabs_sq.append(float(medabs_ds))
        improvement.append(float(imp))
        axis_stats.append(
            {
                "axis": int(i),
                "spread_identity": float(sf),
                "spread_square": float(ss),
                "median_abs_square": float(medabs_ds),
                "improvement": float(imp),
            }
        )

    if not axes:
        return OuterPeelProposal(
            name="square",
            score=float("inf"),
            improvement=0.0,
            details={"reason": "no_valid_axes", "abs_floor": abs_floor},
        )

    k = int(np.nanargmax(np.asarray(improvement)))
    best_axis = axes[k]

    details = {
        "best_axis": best_axis,
        "spread_identity": spread_f[k],
        "spread_square": spread_sq[k],
        "median_abs_square": medabs_sq[k],
        "improvement": improvement[k],
        "num_valid_axes": len(axes),
        "axes": axes,
        "axis_stats": axis_stats,
    }
    return OuterPeelProposal(
        name="square",
        score=float(spread_sq[k]),
        improvement=float(improvement[k]),
        details=details,
    )


def decide_square_preference(
    proposal: OuterPeelProposal,
    y_data_np: np.ndarray,
    *,
    frac_negative_max: float = 0.01,
    gain_min: float = 30.0,
    score_max: float = 0.05,
    min_good_axes: int = 2,
    trig_like_axes: Optional[Sequence[int]] = None,
    auto_trig_axis_reject_factor: float = 3.0,
    multi_axis_gain_floor: float = 8.0,
) -> OuterPeelDecision:
    """Convert a raw proposal into a conservative yes/no decision."""
    y = np.asarray(y_data_np).reshape(-1)
    if y.size == 0:
        frac_neg = 0.0
    else:
        frac_neg = float(np.mean(y < 0.0))

    diag = {
        "frac_negative_y": frac_neg,
        "frac_negative_max": frac_negative_max,
        "gain_min": gain_min,
        "score_max": score_max,
        "min_good_axes": int(max(1, int(min_good_axes))),
    }

    if frac_neg > frac_negative_max:
        diag["reason"] = "too_many_negative_y"
        return OuterPeelDecision(False, proposal, diag)

    axis_stats_raw = proposal.details.get("axis_stats", [])
    axis_stats: List[Dict] = []
    for row in axis_stats_raw if isinstance(axis_stats_raw, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            axis = int(row.get("axis"))
            sf = float(row.get("spread_identity", float("inf")))
            ss = float(row.get("spread_square", float("inf")))
            imp = float(row.get("improvement", 0.0))
        except Exception:
            continue
        axis_stats.append(
            {
                "axis": axis,
                "spread_identity": sf,
                "spread_square": ss,
                "improvement": imp,
            }
        )

    trig_like_set = set()
    for ax in trig_like_axes or ():
        try:
            trig_like_set.add(int(ax))
        except Exception:
            continue

    auto_ignored = set()
    if axis_stats and auto_trig_axis_reject_factor > 0.0:
        trig_spread_cut = float(auto_trig_axis_reject_factor) * float(score_max)
        trig_gain_cut = max(1.0, 0.5 * float(gain_min))
        for row in axis_stats:
            ss = float(row["spread_square"])
            imp = float(row["improvement"])
            if np.isfinite(ss) and (ss > trig_spread_cut):
                if (not np.isfinite(imp)) or (imp < trig_gain_cut):
                    auto_ignored.add(int(row["axis"]))

    ignored_axes = sorted(trig_like_set | auto_ignored)
    usable_axes = [r for r in axis_stats if int(r["axis"]) not in set(ignored_axes)]
    good_axes = [
        r
        for r in usable_axes
        if np.isfinite(float(r["spread_square"]))
        and (float(r["spread_square"]) <= float(score_max))
    ]
    good_imps = [float(r["improvement"]) for r in good_axes if np.isfinite(float(r["improvement"]))]

    if usable_axes:
        req_good_axes = max(1, min(int(min_good_axes), len(usable_axes)))
    else:
        req_good_axes = 0
    multi_axis_gain = float(np.median(np.asarray(good_imps))) if good_imps else 0.0
    multi_axis_gain_min = max(
        float(multi_axis_gain_floor),
        float(gain_min) / max(1, req_good_axes),
    )

    legacy_gain_ok = bool(np.isfinite(proposal.improvement) and (proposal.improvement >= gain_min))
    legacy_score_ok = bool(np.isfinite(proposal.score) and (proposal.score <= score_max))
    multi_axis_ok = bool(
        req_good_axes > 0
        and (len(good_axes) >= req_good_axes)
        and np.isfinite(multi_axis_gain)
        and (multi_axis_gain >= multi_axis_gain_min)
    )

    diag.update(
        {
            "legacy_gain": float(proposal.improvement),
            "legacy_score": float(proposal.score),
            "legacy_gain_ok": bool(legacy_gain_ok),
            "legacy_score_ok": bool(legacy_score_ok),
            "num_axes": int(len(axis_stats)),
            "ignored_axes": list(int(a) for a in ignored_axes),
            "trig_like_axes": list(int(a) for a in sorted(trig_like_set)),
            "auto_ignored_axes": list(int(a) for a in sorted(auto_ignored)),
            "num_usable_axes": int(len(usable_axes)),
            "num_good_axes": int(len(good_axes)),
            "required_good_axes": int(req_good_axes),
            "good_axes": list(int(r["axis"]) for r in good_axes),
            "multi_axis_gain": float(multi_axis_gain),
            "multi_axis_gain_min": float(multi_axis_gain_min),
            "auto_trig_axis_reject_factor": float(auto_trig_axis_reject_factor),
            "multi_axis_gain_floor": float(multi_axis_gain_floor),
        }
    )

    if legacy_gain_ok and legacy_score_ok:
        diag["reason"] = "prefer_square_legacy"
        return OuterPeelDecision(True, proposal, diag)

    if multi_axis_ok:
        diag["reason"] = "prefer_square_multi_axis"
        return OuterPeelDecision(True, proposal, diag)

    if req_good_axes > 0 and len(good_axes) < req_good_axes:
        diag["reason"] = "insufficient_good_axes"
    elif not np.isfinite(multi_axis_gain) or (multi_axis_gain < multi_axis_gain_min):
        diag["reason"] = "insufficient_multi_axis_gain"
    elif not legacy_gain_ok:
        diag["reason"] = "insufficient_gain"
    else:
        diag["reason"] = "square_not_constant_enough"
    return OuterPeelDecision(False, proposal, diag)


def square_family_evidence(
    proposal: OuterPeelProposal,
    y_data_np: np.ndarray,
    **decision_kwargs,
):
    """Express the square proposal as a shared FamilyEvidence record."""
    from .factorized_search.subproblem_tests import build_square_family_evidence

    decision = decide_square_preference(proposal, y_data_np, **decision_kwargs)
    return build_square_family_evidence(
        proposal_name=str(proposal.name or "square"),
        proposal_score=float(proposal.score),
        proposal_improvement=float(proposal.improvement),
        proposal_details=dict(proposal.details or {}),
        prefer=bool(decision.prefer),
        diagnostics=dict(decision.diagnostics or {}),
    )


def propose_outer_y_transform(
    *,
    identity_model: torch.nn.Module,
    identity_datagen,
    y_data_np: np.ndarray,
    device: torch.device,
    max_points: int = 2048,
    frac_negative_max: float = 0.01,
    gain_min: float = 30.0,
    score_max: float = 0.05,
    min_good_axes: int = 2,
    trig_like_axes: Optional[Sequence[int]] = None,
    auto_trig_axis_reject_factor: float = 3.0,
    multi_axis_gain_floor: float = 8.0,
) -> OuterPeelDecision:
    """High-level helper: score + decide for `square` vs identity."""
    prop = score_square_from_identity(
        model=identity_model,
        datagen=identity_datagen,
        device=device,
        max_points=max_points,
    )
    return decide_square_preference(
        prop,
        y_data_np=y_data_np,
        frac_negative_max=frac_negative_max,
        gain_min=gain_min,
        score_max=score_max,
        min_good_axes=min_good_axes,
        trig_like_axes=trig_like_axes,
        auto_trig_axis_reject_factor=auto_trig_axis_reject_factor,
        multi_axis_gain_floor=multi_axis_gain_floor,
    )


def _diag_const_rel_min(
    H: torch.Tensor,
    *,
    eps: float = 1e-12,
    abs_floor: float = 1e-8,
) -> Tuple[float, Optional[int]]:
    """Return (min_rel, best_axis) over diagonals of H.

    rel_i = MAD(H_ii) / (median(|H_ii|) + eps)

    We ignore axes whose median(|H_ii|) is below abs_floor to avoid
    treating linear directions as "perfect".
    """
    if H.numel() == 0 or H.dim() != 3:
        return float("inf"), None
    diag = H.diagonal(dim1=-2, dim2=-1)  # [N, d]
    d = int(diag.shape[1])
    best = float("inf")
    best_axis: Optional[int] = None
    for i in range(d):
        v = diag[:, i]
        if not torch.isfinite(v).all():
            v = v[torch.isfinite(v)]
        if v.numel() < 50:
            continue
        medabs = float(v.abs().median().item())
        if not math.isfinite(medabs) or medabs <= abs_floor:
            continue
        rel = float((_mad_1d(v) / (medabs + eps)).item())
        if rel < best:
            best = rel
            best_axis = i
    return float(best), best_axis


def _bonus(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    v = max(v, 1e-12)
    b = -math.log10(v)
    return max(0.0, b)


def _best_nonlinear_substitution_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    max_points: int = 1200,
    min_points: int = 128,
    exp_abs_cap: float = 10.0,
) -> Tuple[float, Dict]:
    """Cheap score: best single-axis nonlinear substitution before rational fit.

    This mirrors the Stage-B nonlinear-substitution spirit, but keeps runtime
    bounded by trying a tiny transform set and a single rational degree.
    """
    X = X.to(dtype=torch.float64)
    y = y.view(-1).to(dtype=torch.float64)
    if X.numel() == 0 or y.numel() == 0 or X.shape[0] != y.shape[0]:
        return float("inf"), {"reason": "shape_mismatch"}
    if X.shape[0] < int(min_points) or X.shape[1] < 1:
        return float("inf"), {"reason": "too_few_points_or_dims"}

    n_total = int(X.shape[0])
    n = int(min(int(max_points), n_total))
    if n < n_total:
        # Order-robust subsample: dataloaders may be non-shuffled.
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        idx = torch.randperm(n_total, generator=g)[:n].to(device=X.device)
        Xs = X[idx]
        ys = y[idx]
    else:
        Xs = X
        ys = y

    best_err = float("inf")
    best_meta: Dict = {"reason": "no_valid_substitution"}

    transforms = (
        ("cos", torch.cos),
        ("sin", torch.sin),
        ("exp", torch.exp),
        ("log", torch.log),
    )

    for col_idx in range(int(Xs.shape[1])):
        v = Xs[:, col_idx]
        if not torch.isfinite(v).any():
            continue
        for tname, tfun in transforms:
            m = torch.isfinite(v) & torch.isfinite(ys)
            if tname == "log":
                m = m & (v > 1e-12)
            elif tname == "exp":
                m = m & (v.abs() <= float(exp_abs_cap))
            if int(m.sum().item()) < int(min_points):
                continue

            vv = v[m]
            yv = ys[m]
            Xv = Xs[m].clone()
            with torch.no_grad():
                try:
                    Xv[:, col_idx] = tfun(vv)
                except Exception:
                    continue
            if not torch.isfinite(Xv[:, col_idx]).all():
                continue
            if float(Xv[:, col_idx].std(unbiased=False).item()) < 1e-12:
                continue

            err = float(
                _rational_probe_nd(
                    Xv,
                    yv,
                    deg_num=2,
                    deg_den=2,
                    min_points=min_points,
                    max_points=max_points,
                    dtype=torch.float64,
                    filter_outliers=True,
                    error_metric="median_rel",
                )
            )
            if err < best_err:
                best_err = err
                best_meta = {
                    "transform": tname,
                    "col_idx": int(col_idx),
                    "n_points": int(m.sum().item()),
                    "error": float(err),
                }

    return float(best_err), best_meta


def _score_simplicity(
    *,
    domain_ok_frac: float,
    poly2_rms_rel: float,
    rat_rms_rel: float,
    hess_const_rel: float,
    hess_diag_const_rel_min: float,
    scaling_rel_std: float,
    cross_hess_rel: float,
    axis_exp_int_score: float,
    nls_subst_err: float,
) -> float:
    # Similar spirit to features._probe_score, but also includes
    # a diagonal-Hessian constancy term to catch "partially quadratic"
    # structures like AIF #028 under squaring.
    score = 0.0
    score += 1.00 * _bonus(poly2_rms_rel)
    score += 0.60 * _bonus(rat_rms_rel)
    score += 0.50 * _bonus(hess_const_rel)
    score += 0.90 * _bonus(hess_diag_const_rel_min)
    score += 0.35 * _bonus(scaling_rel_std)
    score += 0.35 * _bonus(cross_hess_rel)
    # Extra nudge for cases that become (approximately) monomial/rational after
    # an outer peel: reward per-axis homogeneity exponents that are
    # (i) stable across samples and (ii) close to small integers.
    score += 0.40 * float(axis_exp_int_score)
    # Reward transforms that become simple after one cheap per-axis
    # substitution, such as cos(x_i).
    score += 0.25 * _bonus(nls_subst_err)
    return float(domain_ok_frac) * float(score)


def _structure_screen_score(
    *,
    rat_rms_rel: float,
    nls_subst_err: float,
) -> Tuple[float, Dict]:
    """Cheap structure-screen score from rational and substitution probes.

    The score is used as a tie-break/priority signal in outer-peel autorun:
    whichever cheap probe is better (raw rational vs. 1-axis substitution+rational)
    should boost selection priority.
    """
    rat_finite = math.isfinite(rat_rms_rel)
    nls_finite = math.isfinite(nls_subst_err)
    rat_bonus = _bonus(rat_rms_rel) if rat_finite else 0.0
    nls_bonus = _bonus(nls_subst_err) if nls_finite else 0.0

    best_err = float("inf")
    best_kind = "none"
    if rat_finite and (rat_rms_rel < best_err):
        best_err = float(rat_rms_rel)
        best_kind = "ratpoly"
    if nls_finite and (nls_subst_err < best_err):
        best_err = float(nls_subst_err)
        best_kind = "nls_subst"

    # Main signal: winner between rational and substitution screens.
    score = max(rat_bonus, nls_bonus)
    # Small extra credit when both screens are good.
    if rat_bonus > 0.0 and nls_bonus > 0.0:
        score += 0.20 * min(rat_bonus, nls_bonus)

    return float(score), {
        "best_err": float(best_err),
        "best_kind": str(best_kind),
        "rat_bonus": float(rat_bonus),
        "nls_bonus": float(nls_bonus),
    }


def _axis_exponent_int_score(
    X: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float = 1e-12,
    min_points_per_axis: int = 128,
    int_eps: float = 1e-3,
    x_abs_floor: float = 1e-10,
    max_abs_int: int = 8,
) -> Tuple[float, Dict]:
    """Score whether per-axis homogeneity exponents look like small integers.

    For each axis i we form:

        α_i(x) = x_i * ∂y/∂x_i / y

    If y is (approximately) a monomial/rational in x, α_i is (approximately)
    constant and often an integer (e.g. +1, -1).

    Returns
    -------
    score, details
        score is in "bonus" units (comparable to _bonus terms), not yet
        multiplied by domain_ok_frac.
    """
    X = X.to(dtype=torch.float64)
    y = y.view(-1).to(dtype=torch.float64)
    g = g.to(dtype=torch.float64)

    if X.numel() == 0 or y.numel() == 0 or g.numel() == 0:
        return 0.0, {"reason": "empty"}
    if X.shape[0] != y.shape[0] or X.shape[0] != g.shape[0]:
        return 0.0, {"reason": "shape_mismatch"}

    y_scale = float(torch.median(y.abs()).item())
    y_eps = max(eps, 1e-9 * max(y_scale, 1.0))
    base = torch.isfinite(y) & (y.abs() > y_eps)
    n_base = int(base.sum().item())
    if n_base < max(50, min_points_per_axis):
        return 0.0, {"reason": "too_few_nonzero_y", "n": n_base, "y_eps": y_eps}

    # α = (x ⊙ ∂y/∂x) / y  (per-axis)
    with torch.no_grad():
        alpha = (X * g) / y.view(-1, 1)

    d = int(alpha.shape[1])
    per_axis: List[Dict] = []
    total = 0.0

    for i in range(d):
        ai = alpha[:, i]
        xi = X[:, i]
        m = base & torch.isfinite(ai) & torch.isfinite(xi) & (xi.abs() > x_abs_floor)
        n = int(m.sum().item())
        if n < min_points_per_axis:
            continue
        v = ai[m]

        med = float(v.median().item())
        mad = float(_mad_1d(v).item())
        rel_mad = float(mad / (abs(med) + 1.0))

        k = int(np.round(med))
        int_delta = float(abs(med - float(k)))

        # Reward both constancy (low rel_mad) and integer-ness (low int_delta).
        const_bonus = _bonus(max(rel_mad, 1e-12))
        int_bonus = _bonus(max(int_delta, float(int_eps)))

        # Downweight very large integers: they are rarer and can be spurious.
        if abs(k) > int(max_abs_int):
            int_bonus *= 0.25

        axis_score = 0.55 * float(int_bonus) + 0.45 * float(const_bonus)
        total += axis_score

        per_axis.append(
            {
                "axis": int(i),
                "n": int(n),
                "median": float(med),
                "mad": float(mad),
                "rel_mad": float(rel_mad),
                "nearest_int": int(k),
                "int_delta": float(int_delta),
                "score": float(axis_score),
            }
        )

    if not per_axis:
        return 0.0, {"reason": "no_valid_axes"}

    score = float(total / max(1, len(per_axis)))
    details = {
        "n_axes": int(len(per_axis)),
        "per_axis": per_axis,
    }
    return score, details


def _domain_mask_for_transform(
    tname: str,
    y: torch.Tensor,
    *,
    f_eps: float,
    eps_trig: float = 1e-6,
) -> torch.Tensor:
    """Conservative domain mask for φ(y) proposals.

    For non-invertible trigs (sin/cos/tan), we restrict y to the principal
    branch of the corresponding inverse so the peel is meaningful.
    """
    m = torch.isfinite(y)
    if tname == "log":
        m = m & (y > f_eps)
    elif tname == "reciprocal":
        m = m & (y.abs() > f_eps)
    elif tname == "sqrt":
        m = m & (y >= 0.0)
    elif tname in ("arcsin", "arccos"):
        m = m & (y.abs() <= (1.0 - float(eps_trig)))
    elif tname == "sin":
        # Inverse is arcsin: principal range [-pi/2, +pi/2]
        m = m & (y.abs() <= (math.pi / 2.0 - float(eps_trig)))
    elif tname == "cos":
        # Inverse is arccos: principal range [0, pi]
        m = m & (y >= (0.0 - float(eps_trig))) & (y <= (math.pi + float(eps_trig)))
    elif tname == "tan":
        # Inverse is arctan: principal range (-pi/2, +pi/2)
        m = m & (y.abs() <= (math.pi / 2.0 - float(eps_trig)))
    # arctan, exp, square, identity handled by finiteness checks only.
    return m


@dataclass
class CompoundAffinePeel:
    """1D diagnostic: can φ(y) be explained as an affine function of a compound z?"""

    name: str
    domain_ok_frac: float
    n_points: int
    rms_rel: float
    a: float
    b: float
    details: Dict


def probe_affine_outer_peels_on_z(
    *,
    y: torch.Tensor,
    z: torch.Tensor,
    transform_names: Sequence[str],
    min_points: int = 256,
    min_domain_frac: float = 0.20,
    eps_domain: float = 1e-12,
) -> List[CompoundAffinePeel]:
    """Rank φ candidates by how well φ(y) ≈ a*z + b on the data.

    Intended for the special case where Stage A already found a full-variable
    compound coordinate z(x). If y = Φ(z) for some outer Φ, then applying the
    inverse peel φ = Φ^{-1} often makes φ(y) ~ z up to affine scaling.
    """
    y = y.view(-1).to(dtype=torch.float64)
    z = z.view(-1).to(dtype=torch.float64)
    if y.numel() == 0 or z.numel() == 0 or y.shape[0] != z.shape[0]:
        return []

    y_scale = float(torch.median(y.abs()).item())
    f_eps = max(eps_domain, 1e-9 * max(y_scale, 1.0))

    specs, y_ops, _, _ = get_separability_y_ops(list(transform_names))
    name_to_op = {str(getattr(sp, "name", "")): op for sp, op in zip(specs, y_ops)}

    out: List[CompoundAffinePeel] = []

    for tname in transform_names:
        tname = str(tname)
        if tname not in name_to_op:
            continue
        op = name_to_op[tname]

        with torch.no_grad():
            try:
                u = op(y)
            except Exception:
                u = torch.full_like(y, float("nan"))

        m = _domain_mask_for_transform(tname, y, f_eps=f_eps) & torch.isfinite(u) & torch.isfinite(z)

        dom_frac = float(m.float().mean().item())
        n_ok = int(m.sum().item())
        if dom_frac < float(min_domain_frac) or n_ok < int(min_points):
            out.append(
                CompoundAffinePeel(
                    name=tname,
                    domain_ok_frac=dom_frac,
                    n_points=n_ok,
                    rms_rel=float("inf"),
                    a=float("nan"),
                    b=float("nan"),
                    details={"reason": "domain_too_small"},
                )
            )
            continue

        zm = z[m].view(-1, 1)
        um = u[m].view(-1, 1)

        # Least-squares fits:
        #   affine:    u ≈ a*z + b
        #   quadratic: u ≈ q2*z^2 + a*z + b
        # Keep the lower-rms model to improve robustness on mildly curved links.
        def _fit_lstsq(A: torch.Tensor, yv: torch.Tensor) -> torch.Tensor:
            try:
                return torch.linalg.lstsq(A, yv).solution
            except Exception:
                ATA = A.T @ A
                ATy = A.T @ yv
                ridge = 1e-12 * float(torch.trace(ATA).item())
                ATA = ATA + ridge * torch.eye(
                    ATA.shape[0], dtype=ATA.dtype, device=ATA.device
                )
                return torch.linalg.solve(ATA, ATy)

        A_aff = torch.cat([zm, torch.ones_like(zm)], dim=1)  # [n, 2]
        sol_aff = _fit_lstsq(A_aff, um)
        a_aff = float(sol_aff[0, 0].item())
        b_aff = float(sol_aff[1, 0].item())
        pred_aff = a_aff * zm + b_aff
        resid_aff = (pred_aff - um).view(-1)
        rms_aff = float(torch.sqrt(torch.mean(resid_aff * resid_aff)).item())

        A_quad = torch.cat([zm * zm, zm, torch.ones_like(zm)], dim=1)  # [n, 3]
        sol_quad = _fit_lstsq(A_quad, um)
        q2 = float(sol_quad[0, 0].item())
        a_quad = float(sol_quad[1, 0].item())
        b_quad = float(sol_quad[2, 0].item())
        pred_quad = q2 * (zm * zm) + a_quad * zm + b_quad
        resid_quad = (pred_quad - um).view(-1)
        rms_quad = float(torch.sqrt(torch.mean(resid_quad * resid_quad)).item())

        use_quad = bool(math.isfinite(rms_quad) and (rms_quad < rms_aff))
        if use_quad:
            a = a_quad
            b = b_quad
            rms = rms_quad
            fit_kind = "quadratic"
        else:
            a = a_aff
            b = b_aff
            rms = rms_aff
            fit_kind = "affine"

        denom = float(torch.median(um.abs()).item())
        denom = max(denom, 1e-12)
        rms_rel = float(rms / denom)

        out.append(
            CompoundAffinePeel(
                name=tname,
                domain_ok_frac=dom_frac,
                n_points=n_ok,
                rms_rel=rms_rel,
                a=a,
                b=b,
                details={"fit_kind": fit_kind, "q2": q2},
            )
        )

    out_sorted = sorted(out, key=lambda r: float(r.rms_rel))
    return out_sorted


def rank_outer_y_transforms(
    *,
    identity_model: torch.nn.Module,
    identity_datagen,
    Nxvars: int,
    transform_names: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    max_points: int = 2048,
    min_points: int = 256,
    min_domain_frac: float = 0.20,
    rat_deg_num: int = 1,
    rat_deg_den: int = 1,
    eps_domain: float = 1e-12,
) -> Tuple[TransformSimplicity, List[TransformSimplicity]]:
    """Rank candidate y-transforms φ(y) by a heuristic simplicity score.

    This is a *proposal* mechanism: it does not alter the current workflow.

    Parameters
    ----------
    identity_model:
        The already-trained identity-space model approximating y(x).
    identity_datagen:
        Dataloader/iterator yielding (x,y) batches in identity-space.
    Nxvars:
        Number of input variables.
    transform_names:
        Names from nestynet_sr.sr_search.y_transforms registry. If None, uses a conservative
        set focusing on monotone/invertible candidates.

    Returns
    -------
    baseline, ranked
        baseline is the identity transform stats. ranked is a list of stats
        sorted by descending score (including identity as the first element).
    """
    if transform_names is None:
        transform_names = (
            "identity",
            "square",
            "log",
            "exp",
            "reciprocal",
            "sqrt",
            "sin",
            "cos",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
        )

    dev = device
    if dev is None:
        try:
            dev = next(identity_model.parameters()).device
        except Exception:
            dev = torch.device("cpu")

    X, _ = _collect_points_from_loader(identity_datagen, device=dev, max_points=max_points)
    X = X.view(X.shape[0], -1)
    Xv = X[:, :Nxvars]

    # Get f, grad, Hessian from analytic methods.
    with torch.no_grad():
        F = identity_model.forward(X)
        G = identity_model.grad(X)
        H = identity_model.grad_grad(X)

    # Extract scalar output
    if F.dim() == 2:
        f = F[:, 0]
    else:
        f = F.view(-1)
    if G.dim() == 3:
        g = G[:, 0, :Nxvars]
    else:
        g = G[:, :Nxvars]
    if H.dim() == 4:
        h = H[:, 0, :Nxvars, :Nxvars]
    else:
        h = H[:, :Nxvars, :Nxvars]

    # Promote to float64 for stability in probes
    Xv = Xv.to(dtype=torch.float64)
    f = f.to(dtype=torch.float64)
    g = g.to(dtype=torch.float64)
    h = h.to(dtype=torch.float64)

    f_scale = float(torch.median(f.abs()).item())
    f_eps = max(eps_domain, 1e-9 * max(f_scale, 1.0))

    # Pull transform ops/derivs from registry
    specs, y_ops, dy_ops, d2y_ops = get_separability_y_ops(list(transform_names))
    name_to_ops = {
        str(getattr(sp, "name", "")): (op, d1, d2)
        for sp, op, d1, d2 in zip(specs, y_ops, dy_ops, d2y_ops)
    }

    out: List[TransformSimplicity] = []

    for tname in transform_names:
        tname = str(tname)
        if tname not in name_to_ops:
            continue
        op, d1, d2 = name_to_ops[tname]

        with torch.no_grad():
            try:
                z = op(f)
                dz = d1(f)
                d2z = d2(f)
            except Exception:
                z = torch.full_like(f, float("nan"))
                dz = torch.full_like(f, float("nan"))
                d2z = torch.full_like(f, float("nan"))

        m = _domain_mask_for_transform(tname, f, f_eps=f_eps)
        m = m & torch.isfinite(z) & torch.isfinite(dz) & torch.isfinite(d2z)

        dom_frac = float(m.float().mean().item())
        n_ok = int(m.sum().item())
        if dom_frac < 0.01 or n_ok < min_points:
            out.append(
                TransformSimplicity(
                    name=tname,
                    domain_ok_frac=dom_frac,
                    n_points=n_ok,
                    poly2_rms_rel=float("inf"),
                    rat_rms_rel=float("inf"),
                    hess_const_rel=float("inf"),
                    hess_diag_const_rel_min=float("inf"),
                    hess_diag_const_best_axis=None,
                    scaling_rel_std=float("inf"),
                    cross_hess_rel=float("inf"),
                    axis_exp_int_score=float("inf"),
                    axis_exp_n_axes=0,
                    structure_screen_score=0.0,
                    score=0.0,
                    score_improvement=0.0,
                    details={"reason": "domain_too_small"},
                )
            )
            continue

        Xm = Xv[m]
        _ym = f[m]
        gm = g[m]
        Hm = h[m]
        zm = z[m]
        dzm = dz[m]
        d2zm = d2z[m]

        # Chain rule for derivatives of z(x) = op(f(x))
        with torch.no_grad():
            gz = dzm.view(-1, 1) * gm
            outer = gm.unsqueeze(2) * gm.unsqueeze(1)
            Hz = d2zm.view(-1, 1, 1) * outer + dzm.view(-1, 1, 1) * Hm

        poly2 = float(_poly_fit_rms_rel(Xm, zm, degree=2))
        rat = float(
            _rational_probe_nd(
                Xm, zm, deg_num=rat_deg_num, deg_den=rat_deg_den, dtype=torch.float64
            )
        )
        hrel, _ = _hess_const_rel(Hz)
        hdiag_rel, hdiag_axis = _diag_const_rel_min(Hz)
        srel = float(_scaling_rel_std(Xm, zm, gz))
        crel = float(_cross_hess_rel(Hz))
        ax_score, ax_details = _axis_exponent_int_score(Xm, zm, gz)
        nls_err = float("inf")
        nls_meta: Dict = {"reason": "skipped"}
        if int(Xm.shape[1]) >= 2:
            nls_err, nls_meta = _best_nonlinear_substitution_probe(
                Xm,
                zm,
                max_points=min(1200, int(Xm.shape[0])),
                min_points=max(128, int(min_points) // 2),
            )
        struct_score, struct_meta = _structure_screen_score(
            rat_rms_rel=float(rat),
            nls_subst_err=float(nls_err),
        )

        score = _score_simplicity(
            domain_ok_frac=dom_frac,
            poly2_rms_rel=poly2,
            rat_rms_rel=rat,
            hess_const_rel=float(hrel),
            hess_diag_const_rel_min=float(hdiag_rel),
            scaling_rel_std=srel,
            cross_hess_rel=crel,
            axis_exp_int_score=float(ax_score),
            nls_subst_err=float(nls_err),
        )

        out.append(
            TransformSimplicity(
                name=tname,
                domain_ok_frac=dom_frac,
                n_points=n_ok,
                poly2_rms_rel=poly2,
                rat_rms_rel=rat,
                hess_const_rel=float(hrel),
                hess_diag_const_rel_min=float(hdiag_rel),
                hess_diag_const_best_axis=hdiag_axis,
                scaling_rel_std=srel,
                cross_hess_rel=crel,
                axis_exp_int_score=float(ax_score),
                axis_exp_n_axes=int(ax_details.get("n_axes", 0) if isinstance(ax_details, dict) else 0),
                structure_screen_score=float(struct_score),
                score=float(score),
                score_improvement=0.0,  # filled after baseline is known
                details={
                    "axis_exponent": ax_details,
                    "structure_screen_score": float(struct_score),
                    "structure_probe_err": float(struct_meta.get("best_err", float("inf"))),
                    "structure_probe_kind": struct_meta.get("best_kind", None),
                    "structure_rat_bonus": float(struct_meta.get("rat_bonus", 0.0)),
                    "structure_nls_bonus": float(struct_meta.get("nls_bonus", 0.0)),
                    "nls_subst_err": float(nls_err),
                    "nls_subst_transform": nls_meta.get("transform", None),
                    "nls_subst_col": nls_meta.get("col_idx", None),
                    "nls_subst_n_points": nls_meta.get("n_points", None),
                },
            )
        )

    # Identify baseline identity
    baseline = None
    for s in out:
        if s.name == "identity":
            baseline = s
            break
    if baseline is None:
        # Fallback: treat the highest-domain candidate as baseline
        baseline = (
            max(out, key=lambda q: q.domain_ok_frac)
            if out
            else TransformSimplicity(
                name="identity",
                domain_ok_frac=0.0,
                n_points=0,
                poly2_rms_rel=float("inf"),
                rat_rms_rel=float("inf"),
                hess_const_rel=float("inf"),
                hess_diag_const_rel_min=float("inf"),
                hess_diag_const_best_axis=None,
                scaling_rel_std=float("inf"),
                cross_hess_rel=float("inf"),
                axis_exp_int_score=float("inf"),
                axis_exp_n_axes=0,
                structure_screen_score=0.0,
                score=0.0,
                score_improvement=0.0,
                details={"reason": "no_baseline"},
            )
        )

    base_score = float(baseline.score)
    for i, s in enumerate(out):
        out[i].score_improvement = float(s.score - base_score)

    # Rank primarily by overall score, with structure-screen as tie-break/priority.
    ranked = sorted(
        out,
        key=lambda s: (
            float(s.score),
            float(s.structure_screen_score),
            float(s.domain_ok_frac),
            1.0 if s.name == "identity" else 0.0,
        ),
        reverse=True,
    )

    # Optionally drop transforms with very small domain
    ranked = [s for s in ranked if (s.domain_ok_frac >= min_domain_frac) or (s.name == "identity")]

    return baseline, ranked
