# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Charts for the affine determining operator.

A chart re-expresses the samples and gradients fed to
:func:`~nestynet_sr.sr_gs.affine_algebra.discover_affine_algebra` so that
non-affine symmetry families become affine in the transformed coordinates.
Slice 1 ships two charts:

* ``identity`` — the raw coordinates (today's behavior, always eligible);
* ``log`` — ``u_i = log(x_i)`` with chain-ruled gradients
  ``df/du_i = x_i * df/dx_i``.  Scaling symmetries appear as translations in
  ``u`` and monomial invariants ``prod_i x_i**a_i`` appear as linear invariant
  covectors ``sum_i a_i u_i``, so the log chart subsumes the legacy Stage-A
  monomial compound detector through the same determining operator.

Charts never modify the determining operator, the algebra certificates, or the
promotion gates; they only transform the arrays handed to the solver and tag
the resulting :class:`SymmetryAlgebraSpec` (``chart`` field) so the quotient
compiler renders chart-correct coordinates.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from .affine_algebra import SymmetryAlgebraSpec
from .unit_torus import _as_fraction, projective_exponent_key

_MAX_ABS_SNAPPED_EXPONENT = 8
# Absolute ceiling on the snapped-ray determining residual, applied even in
# calibrated mode: a "symmetry" violated at >10% of the gradient magnitude is
# not one, whatever the baseline noise level says.
_SNAP_SANITY_CEILING = 0.1


@dataclass(frozen=True)
class ChartSpec:
    """A coordinate chart for the affine determining operator."""

    name: str

    def eligibility(self, x: Any) -> tuple[bool, str]:
        """Whether this chart can be applied to the sampled inputs."""

        if self.name == "identity":
            return True, "always_eligible"
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return False, "empty_or_non_2d_samples"
        if arr.shape[1] < 2:
            return False, "univariate_leaf"
        if not np.all(np.isfinite(arr)):
            return False, "non_finite_samples"
        if self.name == "log":
            if not np.all(arr > 0.0):
                return False, "non_positive_samples"
            return True, "eligible"
        if self.name == "reciprocal":
            # 1/x is smooth only away from zero and only within a fixed sign
            # branch, so each column must be nonzero and not cross zero.
            if np.any(np.abs(arr) < 1.0e-12):
                return False, "near_zero_samples"
            all_pos = np.all(arr > 0.0, axis=0)
            all_neg = np.all(arr < 0.0, axis=0)
            if not np.all(all_pos | all_neg):
                return False, "sign_crossing_samples"
            return True, "eligible"
        return False, f"unknown_chart:{self.name}"

    def transform(self, x: Any, grad: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return chart coordinates and chain-ruled gradients."""

        x_arr = np.asarray(x, dtype=float)
        grad_arr = np.asarray(grad, dtype=float)
        if self.name == "identity":
            return x_arr, grad_arr
        if self.name == "log":
            # u = log x ; df/du = x df/dx
            return np.log(x_arr), x_arr * grad_arr
        # reciprocal: u = 1/x ; df/du = -x**2 df/dx
        return 1.0 / x_arr, -(x_arr**2) * grad_arr

    def provenance(self) -> dict[str, Any]:
        return {"chart": str(self.name)}


IDENTITY_CHART = ChartSpec(name="identity")
LOG_CHART = ChartSpec(name="log")
RECIPROCAL_CHART = ChartSpec(name="reciprocal")
_CHARTS_BY_NAME = {
    "identity": IDENTITY_CHART,
    "log": LOG_CHART,
    "reciprocal": RECIPROCAL_CHART,
}


def resolve_charts(cfg: Any) -> tuple[ChartSpec, ...]:
    """Charts requested by the config; identity-only when unspecified."""

    try:
        names = tuple(cfg.general_affine_chart_names())
    except Exception:
        names = ("identity",)
    return tuple(_CHARTS_BY_NAME[name] for name in names if name in _CHARTS_BY_NAME)


def snap_log_chart_algebra(
    algebra: SymmetryAlgebraSpec,
    *,
    grad_u: Any,
    max_denominator: int = 4,
    residual_tol: float = 1.0e-8,
    calibration_factor: float | None = None,
    chart_name: str = "log",
) -> tuple[SymmetryAlgebraSpec | None, dict[str, Any]]:
    """Snap a chart-transformed invariant covector to a primitive integer ray.

    The snapping and its determining-residual revalidation are chart-agnostic
    (they operate purely on the covector and the chart-transformed gradient),
    so the same routine serves the log chart (covector = monomial exponents)
    and the reciprocal chart (covector = coefficients of ``sum_i c_i/x_i``);
    ``chart_name`` only tags the returned algebra for the quotient compiler.

    The raw SVD covector is a float direction; on the positive domain the
    monomial invariant is defined only up to a nonzero power, so the canonical
    representative is the primitive integer ray (matching the legacy Stage-A
    exponent snapping, extended from integers to small rationals).  The snap is
    *revalidated as a determining check*: for snapped exponents ``e``, the
    claimed scaling symmetries are translations (in ``u = log x``) along
    ``null(e)``, whose determining rows are exactly ``grad_u . v``; the snap is
    accepted only when the relative residual of those rows stays within
    ``residual_tol``.  With ``calibration_factor`` set (noise-calibrated
    mode), the tolerance is instead
    ``max(residual_tol, calibration_factor * baseline)`` where ``baseline``
    is the residual of the *unsnapped* covector — i.e. snapping may not
    degrade the determining residual by more than that factor, whatever the
    surrogate's noise level.

    Returns ``(snapped_algebra, report)`` on success and ``(None, report)``
    when the snap is rejected.  When the algebra has zero or multiple invariant
    covectors the covectors are left unsnapped and the quotient compiler emits
    the corresponding audit plan.
    """

    covectors = np.asarray(algebra.linear_invariant_covectors, dtype=float)
    n_covectors = int(covectors.shape[0]) if covectors.ndim == 2 else 0
    report: dict[str, Any] = {
        "status": "skipped",
        "chart": str(chart_name),
        "covector_count": n_covectors,
        "max_denominator": int(max_denominator),
        "residual_tol": float(residual_tol),
    }
    if n_covectors != 1:
        report["reason"] = f"covector_count_{n_covectors}_not_snappable"
        tagged = dataclasses.replace(
            algebra,
            chart=str(chart_name),
            evidence={**dict(algebra.evidence), "chart_snap": dict(report)},
        )
        return tagged, report

    row = covectors[0].astype(float)
    max_abs = float(np.max(np.abs(row))) if row.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        report.update(status="rejected", reason="zero_covector")
        return None, report
    scaled = row / max_abs
    fractions = [_as_fraction(float(v), max_den=int(max_denominator)) for v in scaled]
    ints = projective_exponent_key(fractions)
    report["raw_covector"] = [float(v) for v in row]
    report["snapped_fractions"] = [str(f) for f in fractions]
    report["exponents"] = [int(v) for v in ints]
    if not ints or not any(int(v) != 0 for v in ints):
        report.update(status="rejected", reason="all_zero_after_snap")
        return None, report
    if sum(1 for v in ints if int(v) != 0) < 2:
        report.update(status="rejected", reason="degenerate_single_axis")
        return None, report
    if max(abs(int(v)) for v in ints) > _MAX_ABS_SNAPPED_EXPONENT:
        report.update(status="rejected", reason="exponent_magnitude_exceeds_limit")
        return None, report

    e = np.asarray([float(v) for v in ints], dtype=float)
    annihilator = _nullspace_of_row(e)
    grad_arr = np.asarray(grad_u, dtype=float)
    denom = max(float(np.linalg.norm(grad_arr)), 1.0e-300)
    residual = float(np.linalg.norm(grad_arr @ annihilator) / denom) if annihilator.size else 1.0
    report["residual_rel"] = residual
    effective_tol = float(residual_tol)
    if calibration_factor is not None:
        baseline_annihilator = _nullspace_of_row(row)
        baseline = (
            float(np.linalg.norm(grad_arr @ baseline_annihilator) / denom)
            if baseline_annihilator.size
            else 1.0
        )
        effective_tol = max(effective_tol, float(calibration_factor) * baseline)
        effective_tol = min(effective_tol, _SNAP_SANITY_CEILING)
        report["baseline_residual_rel"] = baseline
        report["calibration_factor"] = float(calibration_factor)
    report["effective_residual_tol"] = effective_tol
    if residual > effective_tol:
        report.update(status="rejected", reason="snapped_residual_exceeds_tol")
        return None, report

    report["status"] = "snapped"
    snapped = dataclasses.replace(
        algebra,
        chart=str(chart_name),
        linear_invariant_covectors=e.reshape(1, -1),
        evidence={**dict(algebra.evidence), "chart_snap": dict(report)},
    )
    return snapped, report


def _nullspace_of_row(row: np.ndarray) -> np.ndarray:
    """Orthonormal basis (columns) of the nullspace of a single covector row."""

    arr = np.asarray(row, dtype=float).reshape(1, -1)
    _u, s, vt = np.linalg.svd(arr, full_matrices=True)
    tol = max(1.0e-12, 1.0e-9 * float(s[0]) if s.size else 0.0)
    rank = int(np.sum(s > tol))
    return vt[rank:].T
