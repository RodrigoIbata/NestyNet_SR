# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Bridge: GS graph-symmetry discovery compiled into executable input charts.

Division of labor: ``sr_gs`` DISCOVERS symmetries (determining operators,
nullspaces, snapping, certificates); ``nestynet.charts`` EXECUTES charts
(differentiable embeddings, comb-split fitting, key-sharpness gating). This
module is the seam between them, in the dependency-correct direction
(NestyNet_SR imports nestynet, never the reverse).

Pipeline:
  1. fit an identity-chart NestyNet surrogate to a densely sampled record
     (t, y) -- interpolation only, comb split;
  2. feed its ANALYTIC values and gradients to the affine graph-symmetry
     determining operator (:func:`~nestynet_sr.sr_gs.affine_algebra.
     discover_affine_algebra`), which solves
     grad(f)(x).(Ax+b) - alpha - beta f(x) = 0 jointly for input and output
     affine actions;
  3. compile each discovered generator into an executable chart:
     a scaling-type input action (A != 0) rectifies to the shifted-log warp
     u = log(t - t0) with t0 = -b/A (``nestynet.charts.WarpChart``); a pure
     translation means t is already rectified; output actions
     (alpha, beta) are reported as cofactor suggestions (future work);
  4. gate every compiled chart with the key-sharpness certificate: the
     profiled validation well of the chart parameter must open at some
     delta << 1, or the proposal is not a key.

The surrogate's gradients carry fit-level error, so the sensible residual
acceptance scale here is ~1e-2..1e-3, not the exact-gradient default of the
Stage-A scan; ``spectral_gap`` nullity detection is the noise-calibrated
choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from nestynet.charts import (
    FitConfig,
    IdentityChart,
    SharpnessConfig,
    SharpnessReport,
    WarpChart,
    fit_chart,
)

from .affine_algebra import SymmetryAlgebraSpec, discover_affine_algebra


@dataclass
class ChartProposal:
    """One discovered generator, classified and (when possible) compiled."""

    kind: str  # "scaling_warp" | "translation" | "output_only"
    A: float
    b: float
    alpha: float
    beta: float
    chart: WarpChart | None
    note: str
    sharpness: SharpnessReport | None = None
    exponent: float | None = None  # profiled output exponent at refined t0
    law_certified: bool = False    # defect passes over the tested range

    @property
    def certified(self) -> bool:
        # The LAW gate certifies the chart (defect small over the tested
        # e-foldings). The sharpness well is the IDENTIFIABILITY report for
        # the fixed point: is_key False with law_certified True means "the
        # scaling law holds; its fixed point is only loosely pinned".
        return self.chart is not None and self.law_certified


@dataclass
class ChartScanResult:
    spec: SymmetryAlgebraSpec
    proposals: list[ChartProposal] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    surrogate_val_rel_rmse: float = float("nan")

    def certified_charts(self) -> list[WarpChart]:
        return [p.chart for p in self.proposals if p.certified]


def _scaling_residual(t, f, fp, t0):
    """Relative defect of the scaling law f'.(t-t0) = alpha + beta f, with
    (alpha, beta) profiled out by weighted least squares.

    Rows are weighted by 1/(t-t0): the Haar measure of the scaling group
    (uniform in the warped coordinate d log(t-t0)). This simultaneously
    equalizes the heteroscedastic surrogate-derivative noise, whose row
    error scales as (t-t0) * delta_fprime. Returns (rho, alpha, beta).
    """
    w = 1.0 / (t - t0)
    sw = np.sqrt(w / np.sum(w))
    lhs = fp * (t - t0)
    B = np.stack([np.ones_like(f), f], axis=1)
    coef, *_ = np.linalg.lstsq(B * sw[:, None], lhs * sw, rcond=None)
    resid = (lhs - B @ coef) * sw
    rho = float(np.sqrt(np.sum(resid**2)) / max(np.sqrt(np.sum((lhs * sw) ** 2)), 1e-300))
    return rho, float(coef[0]), float(coef[1])


def _refine_t0(t, f, fp, t0_init):
    """Golden-section refinement of the scaling fixed point on the
    profiled determining residual (no refits needed)."""
    gap = float(t.min()) - t0_init
    lo = t0_init - 3.0 * abs(gap)
    hi = float(t.min()) - 1e-9 * (float(t.max()) - float(t.min()))
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(80):
        if _scaling_residual(t, f, fp, c)[0] < _scaling_residual(t, f, fp, d)[0]:
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
    return 0.5 * (a + b)


def _gs_sharpness(t, f, fp, t0, cfg: SharpnessConfig, gs_floor_tol: float):
    """Key-sharpness of a diffeomorphic warp, measured in the SYMMETRY-DEFECT
    metric.

    A diffeomorphic chart has NO value-fit well: any smooth monotone
    reparameterization leaves a fittable 1D problem, so the flexible lock
    absorbs the perturbation entirely (measured: C == 1). The key lives in
    the symmetry law itself, so the profiled well is taken over the
    determining residual rho(t0*(1+delta)), with the output action profiled
    out. A key must have BOTH a low floor (the symmetry actually holds) and
    a well (the fixed point is identified).
    """
    floor, _, _ = _scaling_residual(t, f, fp, t0)
    rep = SharpnessReport(param="t0[gs-residual]", floor=floor)
    for d in cfg.deltas:
        vals = []
        for s in (+1.0, -1.0):
            t0p = t0 * (1.0 + s * d)
            if t0p >= float(t.min()):
                continue
            vals.append(_scaling_residual(t, f, fp, t0p)[0])
        if not vals:
            continue
        C = float(np.mean(vals) / max(floor, 1e-300))
        rep.contrasts[d] = C
        if rep.delta_star is None and C > cfg.key_contrast:
            rep.delta_star = d
        rep.c_max = max(rep.c_max, C)
    rep.is_key = rep.c_max > cfg.key_contrast and floor < gs_floor_tol
    return rep



def _self_consistent_rho(t, y, t0, X, cfg):
    """Defect of the scaling law at t0, with derivatives from a fit ON the
    candidate warp (self-consistent; far more accurate than the identity
    surrogate where the scaling information is richest)."""
    wres = fit_chart(t, y, WarpChart(t0=t0), cfg)
    wcm = wres.charted_model
    with torch.no_grad():
        f_w = wcm(X).squeeze(1).cpu().numpy()
        fp_w = wcm.grad(X).squeeze(1).squeeze(1).cpu().numpy()
    rho, _, beta = _scaling_residual(t, f_w, fp_w, t0)
    return rho, beta, f_w, fp_w


def _gap_scan_refine(t, y, X, cfg, pass_tol, log):
    """Select the scaling fixed point as the SMALLEST gap that passes.

    The warp family has a non-compact degeneracy: as t0 -> -inf the log
    compresses the data onto a short arc where any smooth curve satisfies
    the scaling law (the translation limit), so the defect minimum slides
    down an artificial valley. The hard-to-vary key is the one tested over
    the LARGEST range, i.e. the smallest gap g = t_min - t0 whose
    self-consistent defect passes; refinement then stays inside that basin.
    """
    span = float(t.max() - t.min())
    best = None
    for g in np.geomspace(0.02 * span, 3.0 * span, 16):
        t0 = float(t.min()) - g
        rho, beta, f_w, fp_w = _self_consistent_rho(t, y, t0, X, cfg)
        if rho < pass_tol:
            log.append(
                f"    gap scan: first pass at gap {g:.4g} (rho {rho:.2e}, "
                f"exponent {beta:.4f}); refining inside the basin"
            )
            lo, hi = g / 1.6, g * 1.6
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a, b_ = lo, hi
            c, d = b_ - gr * (b_ - a), a + gr * (b_ - a)
            for _ in range(10):
                rc = _self_consistent_rho(t, y, float(t.min()) - c, X, cfg)[0]
                rd = _self_consistent_rho(t, y, float(t.min()) - d, X, cfg)[0]
                if rc < rd:
                    b_, d = d, c
                    c = b_ - gr * (b_ - a)
                else:
                    a, c = c, d
                    d = a + gr * (b_ - a)
            g_ref = 0.5 * (a + b_)
            t0 = float(t.min()) - g_ref
            rho, beta, f_w, fp_w = _self_consistent_rho(t, y, t0, X, cfg)
            best = (t0, rho, beta, f_w, fp_w)
            break
    return best


def scan_and_compile_charts(
    t: np.ndarray,
    y: np.ndarray,
    *,
    fit_cfg: FitConfig | None = None,
    sharp_fit_cfg: FitConfig | None = None,
    sharpness_cfg: SharpnessConfig | None = None,
    gate_sharpness: bool = True,
    nullity_strategy: str = "spectral_gap",
    acceptance_residual_tol: float = 1e-2,
    rank_rtol: float | None = None,
    min_spectral_gap: float | None = None,
    heldout_fraction: float = 0.2,
    coeff_rel_tol: float = 0.05,
) -> ChartScanResult:
    """Discover affine graph symmetries of y(t) and compile them to charts."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if t.shape != y.shape:
        raise ValueError("t and y must be 1D arrays of equal length")
    fit_cfg = fit_cfg or FitConfig(segments=24, epochs=400, restarts=2)
    sharp_fit_cfg = sharp_fit_cfg or FitConfig(segments=12, epochs=150, restarts=2)
    log: list[str] = []

    # 1. identity surrogate with analytic derivatives
    ident = IdentityChart.from_data(torch.as_tensor(t, dtype=torch.float64).unsqueeze(1))
    res = fit_chart(t, y, ident, fit_cfg)
    cm = res.charted_model
    X = torch.as_tensor(t, dtype=torch.float64).unsqueeze(1)
    with torch.no_grad():
        f = cm(X).squeeze(1).cpu().numpy()
        fprime = cm.grad(X).squeeze(1).squeeze(1).cpu().numpy()
    log.append(f"identity surrogate: val relRMSE {res.val_rel_rmse:.2e}")

    # 2. the GS determining operator on the surrogate's analytic graph data
    scan_kwargs: dict = {}
    if rank_rtol is not None:
        scan_kwargs["rank_rtol"] = float(rank_rtol)
    if min_spectral_gap is not None:
        scan_kwargs["min_spectral_gap"] = float(min_spectral_gap)
    spec = discover_affine_algebra(
        t[:, None],
        f,
        fprime[:, None],
        heldout_fraction=heldout_fraction,
        nullity_strategy=nullity_strategy,
        acceptance_residual_tol=acceptance_residual_tol,
        **scan_kwargs,
    )
    log.append(
        f"affine graph scan: nullity {spec.nullity}, train residual "
        f"{spec.train_residual_rel:.2e}, heldout {spec.heldout_residual_rel:.2e}, "
        f"promotable {spec.promotable}"
    )

    result = ChartScanResult(
        spec=spec, log=log, surrogate_val_rel_rmse=res.val_rel_rmse
    )
    if spec.nullity == 0:
        log.append("no symmetry generators discovered; no charts proposed")
        return result

    # 3. classify and compile each generator view
    for i, view in enumerate(spec.basis_generators):
        A = float(view.A_physical[0][0])
        b = float(view.b_physical[0])
        alpha = float(view.alpha_physical)
        beta = float(view.beta_physical)
        An = abs(float(view.A_normalized[0][0]))
        bn = abs(float(view.b_normalized[0]))
        alphan = abs(float(view.alpha_normalized))
        betan = abs(float(view.beta_normalized))
        biggest = max(An, bn, alphan, betan, 1e-300)

        chart: WarpChart | None = None
        notes: list[str] = []
        exponent: float | None = None
        f_use, fp_use = f, fprime
        if An > coeff_rel_tol * biggest:
            kind = "scaling_warp"
            t0 = -b / A
            if t0 < float(t.min()):
                # Smallest-passing-gap selection with self-consistent
                # derivatives (see _gap_scan_refine): avoids the t0 -> -inf
                # degeneracy valley and keeps the hardest-to-vary key.
                pass_tol = min(
                    acceptance_residual_tol,
                    max(3.0 * res.val_rel_rmse, 1e-3),
                )
                found = _gap_scan_refine(t, y, X, sharp_fit_cfg, pass_tol, notes)
                if found is not None:
                    t0, rho0, exponent, f_use, fp_use = found
                    chart = WarpChart(t0=t0)
                    notes.append(
                        f"scaling about refined t0={t0:.6e}; warp u=log(t-t0); "
                        f"profiled exponent {exponent:.6f} (defect {rho0:.2e})"
                    )
                else:
                    notes.append(
                        f"scaling direction found (raw t0={t0:.6e}) but no gap "
                        f"passes the self-consistent defect tol {pass_tol:.1e}"
                    )
            elif t0 > float(t.max()):
                notes.append(
                    f"scaling about t0={t0:.6e} beyond the data (mirrored "
                    "warp log(t0-t) not implemented)"
                )
            else:
                notes.append(
                    f"scaling fixed point t0={t0:.6e} inside the data span; "
                    "no single-branch warp"
                )
        elif bn > coeff_rel_tol * biggest:
            kind = "translation"
            notes.append("pure input translation; t already rectified")
        else:
            kind = "output_only"
        if max(alphan, betan) > coeff_rel_tol * biggest:
            if kind == "translation" and abs(b) > 0:
                notes.append(
                    f"output cofactor suggestion: rate beta/b = {beta / b:.6e} "
                    "(exp detrend, future work)"
                )
            elif kind == "scaling_warp" and abs(A) > 0:
                notes.append(
                    f"output cofactor suggestion: exponent beta/A = {beta / A:.6e} "
                    "(power detrend, future work)"
                )

        prop = ChartProposal(
            kind=kind, A=A, b=b, alpha=alpha, beta=beta, chart=chart,
            note="; ".join(notes), exponent=exponent,
        )

        # 4. key-sharpness gate, in the SYMMETRY-DEFECT metric. A
        # diffeomorphic warp has no value-fit well (the network absorbs any
        # smooth reparameterization), so the certificate profiles the GS
        # determining residual instead -- no refits required.
        if chart is not None and gate_sharpness:
            # warp wells are floor-calibrated like all key wells; with a
            # noisy floor they open at delta of order one, so scan wide.
            warp_cfg = sharpness_cfg or SharpnessConfig(
                deltas=(1e-3, 1e-2, 1e-1, 3e-1, 1.0)
            )
            prop.sharpness = _gs_sharpness(
                t, f_use, fp_use, chart.get_param("t0"), warp_cfg,
                gs_floor_tol=2.0 * acceptance_residual_tol,
            )
            prop.law_certified = bool(
                prop.sharpness.floor < 2.0 * acceptance_residual_tol
            )
            log.append(
                f"generator {i}: law "
                f"{'CERTIFIED' if prop.law_certified else 'not certified'} "
                f"(defect {prop.sharpness.floor:.2e}); identifiability "
                f"{prop.sharpness.summary()}"
            )
        log.append(f"generator {i}: kind {kind}; {prop.note}")
        result.proposals.append(prop)

    return result
