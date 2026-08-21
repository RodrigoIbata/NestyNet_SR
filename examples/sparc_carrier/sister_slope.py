# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Sister-model posterior of the local logarithmic slope s(z) = dlogF/dlogz.

Fits a 1-input NestyNet segmented model to (u, y) = (log10 z, log10 g_obs)
on the gold sample with per-row sigmas, builds the sister (paper II core,
nestynet.stat), certifies it (stationarity, tangent adequacy, rcond audit),
then draws coherent (f, f') pairs: the slope posterior rides
prediction_gradient_jacobian so every function draw carries its own slope.

Uncertainty flavor (2026-08-20): rows within a galaxy are not independent
(each galaxy is a thin locus), so the reported bands use the cluster-robust
sandwich with galaxies as clusters, Cov = H^+ (sum_g m_g m_g^T) H^+ with
m_g = J_g^T r_g, realized as a galaxy-level multiplier bootstrap
dtheta = H^+ sum_g eps_g m_g (Liang-Zeger; G/(G-1) small-cluster factor).
The row-independent Laplace posterior is computed alongside and printed for
comparison; it is 3-5x narrower and is NOT what the paper reports.

Outputs results/sister_slope.npz with the grid, bands, and draws summary,
plus the candidate outer-law slope curves for the fig4 overlay.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import nestynet
from nestynet.stat import SisterModel
from nestynet.stat.sister import weak_scale_prior

from nuisance_release import LN10, SIG_FLOOR, load_gold
from outer_map import UPS, make_candidates

HERE = Path(__file__).resolve().parent
DTYPE = torch.float64
NSEG = 4
SEED = 0


def fit_1d(U, Y, S, n_segments=NSEG, max_iter=1500):
    torch.manual_seed(SEED)
    model = nestynet.nets.NestyNet_Model("G_Model", 1, 1, n_segments, 0.15,
                                         DTYPE, "cpu")
    dl = DataLoader(TensorDataset(U, Y, S), batch_size=len(U), shuffle=False)

    def residual_factory(_opt):
        adaptor = nestynet.adaptors.SegmentedAdaptor(
            model, segments=torch.arange(n_segments))
        return nestynet.optimizer.ResidualsModule([adaptor], dataloader=dl,
                                                  device="cpu")

    cfg = nestynet.optimizer.LMConfig(LM_strategy="direct_solve",
                                      spla_enable=False, max_iter=max_iter,
                                      chisq_tol=1e-6, verbose=False,
                                      colinearity_warning_enable=False)
    opt = nestynet.optimizer.Predictive_LM_Optimizer(
        list(model.parameters()), [residual_factory], cfg=cfg)
    for _ in range(cfg.max_iter):
        opt.step()
        if opt.state.get("halt"):
            break
    return model


def cluster_draws(curv, resid_std, labels, n, generator, rcond_note=""):
    """Galaxy-level multiplier-bootstrap parameter draws, shape (n, P).

    dtheta = sqrt(G/(G-1)) H^+ sum_g eps_g (J_g^T r_g), eps_g ~ N(0,1) i.i.d.
    over clusters, so Cov(dtheta) is the cluster-robust sandwich.  ``curv.J``
    is the sigma-standardized Jacobian and ``resid_std`` the matching
    standardized residual, both in the rows the model was fitted to.
    """
    J = curv.J
    r = torch.as_tensor(np.asarray(resid_std), dtype=J.dtype).reshape(-1)
    _names, inv = np.unique(np.asarray(labels), return_inverse=True)
    G = int(inv.max()) + 1
    M = torch.zeros(G, J.shape[1], dtype=J.dtype)
    M.index_add_(0, torch.as_tensor(inv, dtype=torch.long), J * r.unsqueeze(1))
    eps = torch.randn(n, G, generator=generator, dtype=J.dtype)
    return math.sqrt(G / (G - 1)) * curv.solve(M.T @ eps.T).T


def main():
    gal, g_gas, g_disk, g_obs, e_frac, _meta = load_gold()
    z = g_gas + UPS * g_disk
    ok = z > 0
    gal_ok = np.asarray(gal)[ok]
    u = np.log10(z[ok])
    y = np.log10(g_obs[ok])
    sig = np.sqrt((e_frac[ok] / LN10) ** 2 + SIG_FLOOR ** 2)

    um, us = float(u.mean()), float(u.std())
    U = torch.tensor(((u - um) / us).reshape(-1, 1), dtype=DTYPE)
    ym = float(y.mean())
    Y = torch.tensor((y - ym).reshape(-1, 1), dtype=DTYPE)
    S = torch.tensor(sig.reshape(-1, 1), dtype=DTYPE)

    model = fit_1d(U, Y, S)
    with torch.no_grad():
        resid = ((Y - model(U)) / S).reshape(-1)
    chi = float((resid ** 2).mean()) ** 0.5
    print(f"1D law fit: {len(u)} rows, chi rms {chi:.3f}, dex rms "
          f"{float(((Y - model(U))**2).mean())**0.5:.4f}")

    # error renormalization: absorb intrinsic scatter so chi^2/dof = 1
    # (uniform sigma rescaling leaves the weighted LS solution unchanged)
    S = S * chi
    resid = resid / chi
    print(f"sigma renormalized by {chi:.3f} before building the sister")

    prior = weak_scale_prior(model)  # regularizes near-null directions only
    sister = SisterModel.from_fit(model, U, Y, S, flavor="posterior",
                                  rcond=1e-10, prior_precision=prior)
    rep = sister.curvature.stationarity(resid)
    print(f"stationarity: {rep.stationary} (score rms {rep.score_rms:.3e})")

    # posterior s(z) band via coherent (f, f') draws
    qlo, qhi = np.quantile(U.numpy().ravel(), [0.005, 0.995])
    grid = torch.linspace(qlo, qhi, 160, dtype=DTYPE).unsqueeze(1)
    adequacy = sister.tangent_adequacy(grid, chi_min=1.0, effect_atol=1e-6)
    flag_raw = np.asarray(adequacy["flagged"])
    if flag_raw.dtype == bool and len(flag_raw) == len(grid):
        flag_mask = flag_raw
    else:  # array of flagged indices
        flag_mask = np.zeros(len(grid), dtype=bool)
        flag_mask[flag_raw.astype(int)] = True
    print(f"tangent adequacy: {int(flag_mask.sum())} flagged grid points "
          f"of {len(grid)}")

    gen = torch.Generator().manual_seed(SEED + 1)
    n_gal = len(np.unique(gal_ok))
    draws = cluster_draws(sister.curvature, resid.numpy(), gal_ok, 2000, gen)
    gen_row = torch.Generator().manual_seed(SEED + 1)
    draws_row = sister.param_draws(2000, generator=gen_row)  # row-independent, for comparison
    print(f"draw flavor: galaxy-cluster sandwich over {n_gal} galaxies "
          f"(row-independent posterior kept only for comparison)")
    Jf = sister.prediction_jacobian(grid)
    Jg = sister.prediction_gradient_jacobian(grid)
    with torch.no_grad():
        fhat = model(grid).reshape(-1)
        ghat = nestynet.adaptors.SegmentedAdaptor(
            model, segments=torch.arange(NSEG)).grad(grid)[:, 0, :].reshape(-1)
    F = (fhat.unsqueeze(0) + draws @ Jf.T).numpy()          # (n, B) in y' units
    Sl = ((ghat.unsqueeze(0) + draws @ Jg.T) / us).numpy()  # slope dlogF/dlogz
    F_row = (fhat.unsqueeze(0) + draws_row @ Jf.T).numpy()
    Sl_row = ((ghat.unsqueeze(0) + draws_row @ Jg.T) / us).numpy()

    # rcond audit: widths must be stable under one-decade rcond change;
    # points failing it (or tangent adequacy) are excluded from the certified
    # domain rather than silently reported
    sister8 = SisterModel.from_fit(model, U, Y, S, flavor="posterior",
                                   rcond=1e-8, prior_precision=prior)
    Jg8 = sister8.prediction_gradient_jacobian(grid)
    gen8 = torch.Generator().manual_seed(SEED + 1)
    draws8 = cluster_draws(sister8.curvature, resid.numpy(), gal_ok, 2000, gen8)
    Sl8 = ((ghat.unsqueeze(0) + draws8 @ Jg8.T) / us).numpy()
    w10 = np.subtract(*np.percentile(Sl, [84, 16], axis=0))
    w8 = np.subtract(*np.percentile(Sl8, [84, 16], axis=0))
    abs_drift = np.abs(w10 - w8)
    drift_k = abs_drift / np.maximum(w10, 1e-12)
    # relative 10% OR absolute 0.01 slope units: a vanishing band whose width
    # moves by <0.01 is stable for every scientific purpose
    certified = ((drift_k < 0.10) | (abs_drift < 0.01)) & ~flag_mask
    print(f"rcond audit (1e-10 vs 1e-8): max band-width drift "
          f"{float(drift_k.max()):.1%}; certified grid points "
          f"{int(certified.sum())}/{len(grid)}")
    if certified.sum():
        lgz_grid = grid.numpy().ravel() * us + um
        print(f"certified z-range: 10^{lgz_grid[certified].min():.2f} .. "
              f"10^{lgz_grid[certified].max():.2f} m/s^2")

    lgz = grid.numpy().ravel() * us + um
    s_med, s_lo, s_hi = (np.percentile(Sl, 50, axis=0),
                         np.percentile(Sl, 16, axis=0),
                         np.percentile(Sl, 84, axis=0))
    f_med = np.percentile(F, 50, axis=0) + ym
    f_lo = np.percentile(F, 16, axis=0) + ym
    f_hi = np.percentile(F, 84, axis=0) + ym
    f_lo99 = np.percentile(F, 0.5, axis=0) + ym
    f_hi99 = np.percentile(F, 99.5, axis=0) + ym

    s_lo_row, s_hi_row = np.percentile(Sl_row, 16, axis=0), np.percentile(Sl_row, 84, axis=0)
    f_lo_row, f_hi_row = (np.percentile(F_row, 16, axis=0) + ym,
                          np.percentile(F_row, 84, axis=0) + ym)
    for zq in (1e-11, 3e-11, 1e-10):
        k = int(np.argmin(np.abs(lgz - math.log10(zq))))
        print(f"s(z={zq:.0e}) = {s_med[k]:.3f}  68% [{s_lo[k]:.3f}, {s_hi[k]:.3f}]"
              f"  (row-independent: [{s_lo_row[k]:.3f}, {s_hi_row[k]:.3f}])"
              f"  certified={bool(certified[k])}")
    hw_c = 0.5 * (f_hi - f_lo); hw_r = 0.5 * (f_hi_row - f_lo_row)
    print(f"law band half-width [dex]: cluster median {np.median(hw_c):.4f} "
          f"(range {hw_c.min():.4f}-{hw_c.max():.4f}); row-independent median "
          f"{np.median(hw_r):.4f}")

    # candidate-family slope curves on the same grid (numeric d logF / d logz)
    fam = np.load(HERE / "results" / "outer_map_family.npy",
                  allow_pickle=True).item()
    cands = make_candidates()
    fam_slopes = {}
    zgrid = 10.0 ** lgz
    for name, (fn, _p0) in cands.items():
        p = fam[name]["p"]
        h = 1e-3
        fam_slopes[name] = (fn(10 ** (lgz + h), p) - fn(10 ** (lgz - h), p)) / (2 * h)

    out = HERE / "results" / "sister_slope.npz"
    np.savez(out, lgz=lgz, s_med=s_med, s_lo=s_lo, s_hi=s_hi,
             f_med=f_med, f_lo=f_lo, f_hi=f_hi, f_lo99=f_lo99, f_hi99=f_hi99,
             u_mean=um, u_std=us, y_mean=ym,
             certified=certified,
             s_lo_row=s_lo_row, s_hi_row=s_hi_row, f_lo_row=f_lo_row, f_hi_row=f_hi_row,
             flavor="galaxy_cluster_sandwich",
             **{f"slope_{k}": v for k, v in fam_slopes.items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
