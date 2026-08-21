# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Cheap monomial screens shared by Stage A and Stage B.

The helpers in this module are scheduling evidence only.  They do not accept
or reject symbolic rewrites; normal Stage A/B candidate training and validation
must still confirm any move that used these scores.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class MonomialScreenResult:
    """Summary of a univariate log-log monomial fit."""

    ok: bool
    k_hat: float = 0.0
    rel_rms: float = float("inf")
    support_frac: float = 0.0
    n_points: int = 0
    source: str = "loglog"
    reason: str = ""

    @property
    def confidence(self) -> float:
        if not self.ok or not math.isfinite(float(self.rel_rms)):
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(self.rel_rms)))


_MONOMIAL_LABEL_RE = re.compile(r"^monomial_deg(?P<degree>[0-9]+)(?P<zinv>\[z_inv\])?$")
_MONOMIAL_POWER_LABEL_RE = re.compile(
    r"^monomial_pow(?P<num>[0-9]+)_(?P<den>[0-9]+)(?P<zinv>\[z_inv\])?$"
)

HALF_INTEGER_MONOMIAL_POWERS: tuple[Fraction, ...] = (
    Fraction(1, 2),
    Fraction(3, 2),
    Fraction(5, 2),
)


def candidate_monomial_exponent(label: Any) -> Optional[float]:
    """Return the effective power represented by a monomial candidate label."""

    s = str(label or "")
    m = _MONOMIAL_LABEL_RE.match(s)
    if m:
        degree = float(int(m.group("degree")))
        return -degree if m.group("zinv") else degree
    m = _MONOMIAL_POWER_LABEL_RE.match(s)
    if m:
        den = int(m.group("den"))
        if den <= 0:
            return None
        power = float(Fraction(int(m.group("num")), den))
        return -power if m.group("zinv") else power
    return None


def monomial_power_label(power: Fraction | float | int) -> str:
    """Return the candidate label for a fixed positive monomial power."""

    frac = Fraction(power).limit_denominator(16)
    if frac.denominator == 1:
        return f"monomial_deg{frac.numerator}"
    return f"monomial_pow{frac.numerator}_{frac.denominator}"


def snap_to_half_integer_monomial_power(
    k_hat: float,
    *,
    tol: float = 3.0e-2,
) -> Optional[Fraction]:
    """Snap ``|k_hat|`` to the small fixed half-integer monomial dictionary."""

    try:
        kval = abs(float(k_hat))
    except Exception:
        return None
    if not math.isfinite(kval):
        return None
    best: Optional[Fraction] = None
    best_err = float("inf")
    for power in HALF_INTEGER_MONOMIAL_POWERS:
        err = abs(kval - float(power))
        if err < best_err:
            best = power
            best_err = err
    if best is None or best_err > float(tol):
        return None
    return best


def snap_to_integer_monomial_power(
    k_hat: float,
    *,
    tol: float = 3.0e-2,
    max_power: int = 6,
) -> Optional[Fraction]:
    """Snap ``|k_hat|`` to a small positive integer monomial power."""

    try:
        kval = abs(float(k_hat))
    except Exception:
        return None
    if not math.isfinite(kval):
        return None
    nearest = int(round(kval))
    if nearest < 1 or nearest > int(max_power):
        return None
    if abs(kval - float(nearest)) > float(tol):
        return None
    return Fraction(nearest, 1)


def half_power_domain_ok(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    min_positive_frac: float = 0.995,
    min_sign_frac: float = 0.995,
) -> tuple[bool, str]:
    """Check that sampled data are compatible with a real fixed half-power.

    ``PowerLeaf`` clamps negative inputs, but these fixed monomial candidates
    are intended to mean the real expression ``z**(n/2)``.  Only propose them
    when the sampled coordinate is genuinely positive.  If a teacher value is
    supplied, require an essentially constant sign as well, since a scalar
    multiple of a positive half-power cannot change sign.
    """

    try:
        xv = x.detach().reshape(-1).to(dtype=torch.float64, device=torch.device("cpu"))
    except Exception as exc:
        return False, f"x-conversion:{exc}"
    if xv.numel() <= 0:
        return False, "empty-coordinate"

    finite_x = torch.isfinite(xv)
    n_finite = int(finite_x.sum().item())
    if n_finite <= 0:
        return False, "no-finite-coordinate"
    x_f = xv[finite_x]
    scale_x = max(1.0e-30, 1.0e-12 * float(torch.nanmedian(torch.abs(x_f)).item()))
    positive = x_f > scale_x
    positive_frac = float(positive.sum().item() / max(n_finite, 1))
    if positive_frac < float(min_positive_frac):
        return False, f"coordinate-not-positive(frac={positive_frac:.3g})"

    if y is None:
        return True, ""

    try:
        yv = y.detach().reshape(-1).to(dtype=torch.float64, device=torch.device("cpu"))
    except Exception as exc:
        return False, f"y-conversion:{exc}"
    n = min(int(xv.numel()), int(yv.numel()))
    if n <= 0:
        return False, "empty-teacher"
    xv_n = xv[:n]
    yv_n = yv[:n]
    finite = torch.isfinite(xv_n) & torch.isfinite(yv_n)
    if int(finite.sum().item()) <= 0:
        return False, "no-finite-teacher"
    yf = yv_n[finite]
    scale_y = max(1.0e-30, 1.0e-12 * float(torch.nanmedian(torch.abs(yf)).item()))
    nz = torch.abs(yf) > scale_y
    if int(nz.sum().item()) <= 0:
        return False, "zero-teacher"
    yf = yf[nz]
    pos_frac = float((yf > 0).sum().item() / max(int(yf.numel()), 1))
    neg_frac = float((yf < 0).sum().item() / max(int(yf.numel()), 1))
    if max(pos_frac, neg_frac) < float(min_sign_frac):
        return False, f"teacher-sign-changing(frac={max(pos_frac, neg_frac):.3g})"
    return True, ""


def fit_univariate_monomial_screen(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    min_points: int = 64,
) -> MonomialScreenResult:
    """Fit ``log|y| ~= k log|x| + c`` and return a relative residual score."""

    try:
        xv = x.detach().reshape(-1).to(dtype=torch.float64, device=torch.device("cpu"))
        yv = y.detach().reshape(-1).to(dtype=torch.float64, device=torch.device("cpu"))
    except Exception as exc:
        return MonomialScreenResult(False, reason=f"tensor-conversion:{exc}")

    n_total = int(min(xv.numel(), yv.numel()))
    if n_total <= 0:
        return MonomialScreenResult(False, reason="empty")
    if xv.numel() != yv.numel():
        n = min(xv.numel(), yv.numel())
        xv = xv[:n]
        yv = yv[:n]

    eps_x = max(1.0e-30, 1.0e-12 * float(torch.nanmedian(torch.abs(xv)).item() if xv.numel() else 1.0))
    eps_y = max(1.0e-30, 1.0e-12 * float(torch.nanmedian(torch.abs(yv)).item() if yv.numel() else 1.0))
    mask = torch.isfinite(xv) & torch.isfinite(yv) & (torch.abs(xv) > eps_x) & (torch.abs(yv) > eps_y)
    n_valid = int(mask.sum().item())
    if n_valid < int(min_points):
        return MonomialScreenResult(
            False,
            support_frac=float(n_valid / max(1, n_total)),
            n_points=n_valid,
            reason="insufficient-finite-support",
        )

    lx = torch.log(torch.abs(xv[mask]))
    ly = torch.log(torch.abs(yv[mask]))
    lx_c = lx - torch.mean(lx)
    ly_c = ly - torch.mean(ly)
    den = torch.sum(lx_c * lx_c)
    if not torch.isfinite(den) or float(den.item()) <= 1.0e-30:
        return MonomialScreenResult(
            False,
            support_frac=float(n_valid / max(1, n_total)),
            n_points=n_valid,
            reason="degenerate-coordinate",
        )

    k_hat_t = torch.sum(lx_c * ly_c) / den
    intercept = torch.mean(ly) - k_hat_t * torch.mean(lx)
    resid = ly - (k_hat_t * lx + intercept)
    scale = torch.sqrt(torch.mean(ly_c * ly_c)).clamp_min(1.0e-30)
    rel_rms = torch.sqrt(torch.mean(resid * resid)) / scale

    k_hat = float(k_hat_t.item())
    rel = float(rel_rms.item())
    if not math.isfinite(k_hat) or not math.isfinite(rel):
        return MonomialScreenResult(
            False,
            support_frac=float(n_valid / max(1, n_total)),
            n_points=n_valid,
            reason="nonfinite-fit",
        )

    return MonomialScreenResult(
        True,
        k_hat=k_hat,
        rel_rms=rel,
        support_frac=float(n_valid / max(1, n_total)),
        n_points=n_valid,
    )


def candidate_priority_from_screen(
    *,
    label: Any,
    screen: Optional[MonomialScreenResult],
    is_raw_variable: bool,
    scale_hint: Any = None,
) -> tuple:
    """Sort key for monomial candidates; lower is better."""

    exp = candidate_monomial_exponent(label)
    if exp is None and str(label or "") == "scale" and scale_hint is not None:
        try:
            exp = float(getattr(scale_hint, "k_hat"))
        except Exception:
            exp = None

    if screen is not None and screen.ok:
        degree_err = abs(float(exp) - float(screen.k_hat)) if exp is not None else 0.0
        return (
            0,
            float(screen.rel_rms),
            float(degree_err),
            0 if bool(is_raw_variable) else 1,
            abs(float(exp)) if exp is not None else 99.0,
            str(label),
        )

    # Strong Stage-A scale hints should still outrank unknown screens.
    if scale_hint is not None:
        try:
            rel_std = abs(float(getattr(scale_hint, "rel_std", 1.0)))
        except Exception:
            rel_std = 1.0
        return (
            1,
            rel_std,
            0.0,
            0 if bool(is_raw_variable) else 1,
            abs(float(exp)) if exp is not None else 99.0,
            str(label),
        )

    return (
        2,
        float("inf"),
        0.0,
        0 if bool(is_raw_variable) else 1,
        abs(float(exp)) if exp is not None else 99.0,
        str(label),
    )
