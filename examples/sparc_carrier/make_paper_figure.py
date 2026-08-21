# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Single 2x3 paper figure for the SPARC vignette (quanta-condensed).

Panels: (a) carrier-direction profile, (b) collapse with held-out galaxies
and a residual-histogram inset, (c) determining-operator spectra for the
planted control, (d) recovered Upsilon_d by lane, (e) outer-law family in
the sister band, (f) certified slope posterior. Replaces the four separate
figures produced by make_figures.py / make_fig4.py.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_pilot import (HERE, fit_surrogate, knn1d_fit_predict, load_rows,
                       orientation_carrier, split_galaxies, surrogate_grad_sample,
                       warp, warp_grad_factor)
from classical_check import collapse_scatter, profile_ups
from synth_check import UPS_TRUE, rar
from make_figures import determining_tail, heldout_closure_curve
from outer_map import make_candidates

BLUE, VERM, TEAL = "#0072B2", "#D55E00", "#009E73"
GRAY, INK, MUTED = "#888888", "#1a1a1a", "#666666"
FIGDIR = HERE / "figures"

plt.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.5, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    # explicit: the make_figures import sets these False at module level
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})

FAM_STYLE = {
    "rar_exp": (TEAL, (0, (5, 2))),
    "powerlaw": (INK, (0, (1.5, 1.5))),
    "superpos": (MUTED, (0, (4, 1.5, 1, 1.5))),
}


def main():
    gal, g_gas, g_disk, g_obs = load_rows(HERE / "data" / "sparc_carrier_gold.csv")
    scale = float(np.median(g_disk))
    X = np.stack([g_gas / scale, g_disk / scale], axis=1)
    y = np.log10(g_obs)
    disc_names, held_names = split_galaxies(gal, 0)
    disc = np.isin(gal, disc_names)
    held = ~disc
    names = np.unique(gal)
    by = defaultdict(list)
    for i, g in enumerate(gal):
        by[g].append(i)

    grid_fit = np.linspace(0.05, 2.5, 99)
    ups_disc, _ = profile_ups(g_gas[disc], g_disk[disc], g_obs[disc], grid_fit)
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(200):
        idx = np.concatenate([by[n] for n in rng.choice(names, len(names), True)])
        boots.append(profile_ups(g_gas[idx], g_disk[idx], g_obs[idx], grid_fit)[0])
    b16, b50, b84 = np.percentile(boots, [16, 50, 84])

    # collapse law (quantile-binned median) + held-out residuals
    z_phys = (X @ [1.0, ups_disc]) * scale
    lg_z = np.log10(np.clip(z_phys, 1e-14, None))
    lz_tr, y_tr = lg_z[disc], y[disc]
    edges = np.quantile(lz_tr, np.linspace(0, 1, 26))
    idx_b = np.clip(np.searchsorted(edges, lz_tr, side="right") - 1, 0, 24)
    med = np.array([np.median(y_tr[idx_b == b]) if np.any(idx_b == b) else np.nan
                    for b in range(25)])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    good = ~np.isnan(med)
    law_fn = lambda q: np.interp(q, ctr[good], med[good])
    res1 = y[held] - law_fn(lg_z[held])
    rms1 = float(np.sqrt(np.mean(res1 ** 2)))

    warp_a = np.array([np.median(np.abs(X[:, 0])), np.median(np.abs(X[:, 1]))])
    U = warp(X, warp_a)
    _m, leaf, _b = fit_surrogate(U[disc], y[disc], epochs=800, seed=0)
    yh_ho, _g = surrogate_grad_sample(leaf, U[held])
    res2 = y[held] - yh_ho
    rms2 = float(np.sqrt(np.mean(res2 ** 2)))

    # planted-control spectra + recovered-by-lane
    rng2 = np.random.default_rng(42)
    z_true = np.clip(g_gas + UPS_TRUE * g_disk, 1e-14, None)
    y_clean = np.log10(rar(z_true))
    gal_off = {n: rng2.normal(0.0, 0.10) for n in names}
    y_iid = y_clean + rng2.normal(0.0, 0.12, len(y_clean))
    y_gal = (y_clean + np.array([gal_off[n] for n in gal])
             + rng2.normal(0.0, 0.06, len(y_clean)))
    cases = {}
    for label, yy, ep in (("clean", y_clean, 600), ("iid", y_iid, 600),
                          ("gal", y_gal, 600), ("real", y, 800)):
        _mm, lf, _bv = fit_surrogate(U[disc], yy[disc], epochs=ep, seed=0)
        yh, Gu = surrogate_grad_sample(lf, U[disc])
        G = Gu * warp_grad_factor(X[disc], warp_a)
        r1, _c = orientation_carrier(G, gal[disc])
        cases[label] = {"tail": determining_tail(X[disc], yh, G),
                        "ups": r1["upsilon_d"]}

    ss = np.load(HERE / "results" / "sister_slope.npz", allow_pickle=True)
    fam = np.load(HERE / "results" / "outer_map_family.npy",
                  allow_pickle=True).item()
    cands = make_candidates()
    lgz_s = ss["lgz"]
    cert = ss["certified"].astype(bool)

    # ------------------------------------------------------------- layout
    fig, axes = plt.subplots(3, 2, figsize=(7.05, 8.6))
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axes

    # (a) carrier-direction profile
    ups_grid = np.linspace(0.0, 2.5, 76)
    scat = np.array([collapse_scatter(u, g_gas, g_disk, g_obs) for u in ups_grid])
    closure = heldout_closure_curve(ups_grid, X, y, disc, held)
    ax_a.axvspan(b16, b84, color=BLUE, alpha=0.13, lw=0, zorder=0)
    ax_a.plot(ups_grid, scat, color=BLUE, lw=1.3, zorder=3)
    ax_a.plot(ups_grid, closure, color=VERM, lw=1.3, zorder=3)
    ax_a.axvline(0.5, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax_a.text(1.45, 0.102, "collapse scatter", color=BLUE, fontsize=7)
    ax_a.text(1.45, 0.185, "held-out error", color=VERM, fontsize=7)
    ax_a.text(2.42, 0.385, rf"$ϒ_d={b50:.2f}^{{+{b84-b50:.2f}}}_{{-{b50-b16:.2f}}}$",
              color=BLUE, fontsize=8, ha="right")
    ax_a.set_xlabel(r"carrier coefficient $ϒ_d$")
    ax_a.set_ylabel("rms about 1D law [dex]")
    ax_a.set_xlim(0.0, 2.5); ax_a.set_ylim(0.09, 0.42)
    ax_a.set_title("(a)  carrier-direction profile", fontsize=8, loc="left")

    # (b) collapse with residual-histogram inset
    zz = np.geomspace(3e-13, 3e-9, 300)
    ax_b.plot(np.log10(zz), np.log10(rar(zz)), color=TEAL, lw=1.0,
              ls=(0, (5, 2)), zorder=2)
    ax_b.scatter(lg_z[disc], y[disc], s=3, color=GRAY, alpha=0.4, lw=0, zorder=3)
    ax_b.scatter(lg_z[held], y[held], s=6, color=VERM, alpha=0.8, lw=0, zorder=4)
    lg_grid = np.linspace(lz_tr.min(), lz_tr.max(), 220)
    ax_b.plot(lg_grid, law_fn(lg_grid), color=BLUE, lw=1.4, zorder=5)
    ax_b.text(-12.05, -9.55, "1D law (fit on discovery set only)", color=BLUE,
              fontsize=7)
    ax_b.text(-10.4, -11.05, "RAR interpolation\n(not fitted)", color=TEAL,
              fontsize=7)
    ax_b.set_xlabel(r"$\log_{10} z\ \,[{\rm m\,s^{-2}}]$")
    ax_b.set_ylabel(r"$\log_{10} g_{\rm obs}\ \,[{\rm m\,s^{-2}}]$")
    ax_b.set_xlim(-12.15, -9.3); ax_b.set_ylim(-11.6, -9.3)
    ax_b.set_title("(b)  held-out galaxies on the discovered law",
                   fontsize=8, loc="left")
    # (inset of held-out residual histograms removed 2026-08-20: the 2D rms is
    # tail-dominated and is now described in the paper text instead)

    # (c) determining-operator spectra
    kk = np.arange(1, 9)
    ax_c.plot(kk, cases["clean"]["tail"], "o-", color=BLUE, ms=3, lw=1.1)
    ax_c.plot(kk, cases["iid"]["tail"], "s-", color=VERM, ms=2.8, lw=1.1)
    ax_c.plot(kk, cases["real"]["tail"], "^-", color=INK, ms=3, lw=1.1)
    ax_c.set_yscale("log")
    ax_c.text(2.5, 2.0e-4, "planted, no noise", color=BLUE, fontsize=7)
    ax_c.text(4.2, 1.2e-1, "planted, 0.12 dex", color=VERM, fontsize=7)
    ax_c.text(5.0, 3.0e-3, "real SPARC", color=INK, fontsize=7)
    ax_c.text(7.9, 1.1e-4, "$\\times$50 drop\n(threshold 10)", color=BLUE,
              fontsize=6.5, ha="right")
    ax_c.set_xlabel("singular-value index")
    ax_c.set_ylabel(r"$s_k/s_1$")
    ax_c.set_title("(c)  symmetry gap exists only in the clean limit",
                   fontsize=8, loc="left")

    # (d) recovered Upsilon by lane
    rows = [
        ("planted, clean", cases["clean"]["ups"], None, BLUE, "recovers"),
        ("planted, 0.12 dex", cases["iid"]["ups"], None, VERM, "abstains"),
        ("planted, galaxy-corr.", cases["gal"]["ups"], None, VERM, "abstains"),
        ("real SPARC", cases["real"]["ups"], None, VERM, "abstains"),
        ("collapse, real SPARC", b50, (b50 - b16, b84 - b50), BLUE, "recovers"),
    ]
    ypos = np.arange(len(rows))[::-1]
    ax_d.axvline(UPS_TRUE, color=MUTED, lw=0.8, ls=":")
    ax_d.set_ylim(-0.55, len(rows) - 0.25)
    for (label, u, err, color, verdict), yy in zip(rows, ypos):
        if err is not None:
            ax_d.errorbar([u], [yy], xerr=[[err[0]], [err[1]]], fmt="o",
                          color=color, ms=4.5, capsize=2.5, lw=1.1)
        else:
            ax_d.plot([u], [yy], "o", color=color, ms=4.5)
        ax_d.text(1.35, yy, verdict, color=color, fontsize=7, va="center")
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax_d.text(0.53, len(rows) - 0.52, r"planted $ϒ_d$", color=MUTED,
              fontsize=6.5)
    ax_d.set_xlabel(r"recovered $ϒ_d$ (gradient lane vs collapse)")
    ax_d.set_xlim(-0.3, 1.85)
    ax_d.set_title(r"(d)  abstention is calibrated", fontsize=8, loc="left")
    ax_d.spines["left"].set_visible(False)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.tick_params(top=False, right=False)
    ax_d.tick_params(axis="y", length=0)

    # (e) outer-law family in the sister band
    ax_e.scatter(lg_z[disc], y[disc], s=2.5, color=GRAY, alpha=0.3, lw=0,
                 zorder=1)
    ax_e.fill_between(lgz_s, ss["f_lo99"], ss["f_hi99"], color=BLUE, alpha=0.14,
                      lw=0, zorder=2)
    ax_e.fill_between(lgz_s, ss["f_lo"], ss["f_hi"], color=BLUE, alpha=0.30,
                      lw=0, zorder=3)
    for name, (color, ls) in FAM_STYLE.items():
        fn, _ = cands[name]
        ax_e.plot(lgz_s, fn(10.0 ** lgz_s, fam[name]["p"]), color=color,
                  ls=ls, lw=1.0, zorder=4)
    # shared legend handles for (e) and (f): same line scheme in both panels
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    FAM_LABEL = {"rar_exp": "RAR form", "powerlaw": "power law",
                 "superpos": "Newton + deep sum"}
    law_handles = [Line2D([], [], color=c, ls=ls, lw=1.0)
                   for name, (c, ls) in FAM_STYLE.items()]
    law_labels = [FAM_LABEL[name] for name in FAM_STYLE]
    band_patch = Patch(facecolor=BLUE, alpha=0.30, lw=0)
    band99_patch = Patch(facecolor=BLUE, alpha=0.14, lw=0)
    ax_e.legend(law_handles + [band_patch, band99_patch],
                law_labels + ["sister 68% band", "sister 99% band"],
                loc="upper left", fontsize=7, frameon=False,
                handlelength=2.6, borderaxespad=0.4)

    ax_e.set_xlabel(r"$\log_{10} z\ \,[{\rm m\,s^{-2}}]$")
    ax_e.set_ylabel(r"$\log_{10} g_{\rm obs}\ \,[{\rm m\,s^{-2}}]$")
    ax_e.set_xlim(lgz_s.min(), lgz_s.max())
    ax_e.set_title("(e)  outer laws indistinguishable within the band",
                   fontsize=8, loc="left")

    # (f) certified slope posterior
    seg = np.ma.masked_where(~cert, ss["s_med"])
    lo = np.ma.masked_where(~cert, ss["s_lo"])
    hi = np.ma.masked_where(~cert, ss["s_hi"])
    ax_f.fill_between(lgz_s, lo, hi, color=BLUE, alpha=0.30, lw=0, zorder=3)
    ax_f.plot(lgz_s, seg, color=BLUE, lw=1.3, zorder=4)
    if (~cert).any():
        for k in np.flatnonzero(np.diff(np.r_[False, ~cert, False]) != 0
                                ).reshape(-1, 2):
            ax_f.axvspan(lgz_s[k[0]], lgz_s[min(k[1], len(lgz_s) - 1)],
                         color=GRAY, alpha=0.12, lw=0, zorder=1)
    for name, (color, ls) in FAM_STYLE.items():
        ax_f.plot(lgz_s, ss[f"slope_{name}"], color=color, ls=ls, lw=1.0,
                  zorder=2)
    ax_f.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax_f.axhline(0.5, color=MUTED, lw=0.7, ls=":")
    ax_f.text(-9.92, 1.02, r"Newtonian $s=1$", color=MUTED, fontsize=7)
    ax_f.text(-10.25, 0.42, r"deep regime $s=\frac{1}{2}$", color=MUTED,
              fontsize=7)
    post_handle = (Patch(facecolor=BLUE, alpha=0.30, lw=0),
                   Line2D([], [], color=BLUE, lw=1.3))
    ax_f.legend(law_handles + [post_handle],
                law_labels + ["sister posterior"],
                loc="upper left", fontsize=7, frameon=True, framealpha=0.85,
                edgecolor="none", handlelength=2.6, borderaxespad=0.4,
                handler_map={tuple: HandlerTuple(ndivide=None)})
    ax_f.text(-11.68, 0.24, "grey: audit fails,\nno claim", color=MUTED,
              fontsize=6.5)
    ax_f.set_xlabel(r"$\log_{10} z\ \,[{\rm m\,s^{-2}}]$")
    ax_f.set_ylabel(r"$s(z)=d\log F/d\log z$")
    ax_f.set_xlim(lgz_s.min(), lgz_s.max()); ax_f.set_ylim(0.15, 1.15)
    ax_f.set_title("(f)  what the data identify about the law",
                   fontsize=8, loc="left")

    fig.savefig(FIGDIR / "fig_sparc_vignette.pdf")
    fig.savefig(FIGDIR / "fig_sparc_vignette.png", dpi=220)
    print(f"rms1={rms1:.3f} rms2={rms2:.3f} boot=[{b16:.3f},{b50:.3f},{b84:.3f}]")
    print("wrote figures/fig_sparc_vignette.{pdf,png}")


if __name__ == "__main__":
    main()
