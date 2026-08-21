# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Paper-candidate figures for the SPARC baryonic-carrier vignette (gold sample).

fig1  support geometry (thin galaxy loci + iso-z lines) and the carrier-
      direction profile (in-sample collapse scatter + held-out closure, with
      the galaxy-bootstrap 68% interval)
fig2  the money plot: g_obs against the discovered coordinate z, held-out
      galaxies highlighted, law fitted on discovery only; residual strip and
      1D-vs-2D held-out residual histograms
fig3  the certificate story: determining-operator spectra for the planted
      carrier (clean vs noisy) and real data; recovered Upsilon by lane

All discovery/held-out handling matches run_pilot.py (seed-0 galaxy split;
law and Upsilon fitted on discovery galaxies only).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_pilot import (HERE, fit_surrogate, knn1d_fit_predict, load_rows,
                       orientation_carrier, split_galaxies, surrogate_grad_sample,
                       warp, warp_grad_factor)
from classical_check import collapse_scatter, profile_ups
from synth_check import GDAG, UPS_TRUE, rar
from nestynet_sr.sr_gs.affine_algebra import (_fit_normalization,
                                              build_affine_determining_matrix)

BLUE, VERM, TEAL = "#0072B2", "#D55E00", "#009E73"
GRAY, INK, MUTED = "#888888", "#1a1a1a", "#666666"
FIGDIR = HERE / "figures"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def heldout_closure_curve(ups_grid, X, y, disc, held):
    out = []
    for u in ups_grid:
        z_tr, z_te = X[disc] @ [1.0, u], X[held] @ [1.0, u]
        ok_tr, ok_te = z_tr > 0, z_te > 0
        if ok_tr.sum() < 50 or ok_te.sum() < 50:
            out.append(np.nan)
            continue
        pred = knn1d_fit_predict(z_tr[ok_tr], y[disc][ok_tr], z_te[ok_te])
        out.append(float(np.sqrt(np.mean((y[held][ok_te] - pred) ** 2))))
    return np.array(out)


def determining_tail(X, y_hat, G):
    norm = _fit_normalization(X, y_hat)
    xs = np.asarray(norm.x_scale, dtype=float)
    D = build_affine_determining_matrix(
        norm.normalize_x(X), norm.normalize_y(y_hat),
        G * (xs / float(norm.y_scale)))
    s = np.linalg.svd(D / np.sqrt(len(D)), compute_uv=False)
    return s / s[0]


def main():
    FIGDIR.mkdir(exist_ok=True)
    gal, g_gas, g_disk, g_obs = load_rows(HERE / "data" / "sparc_carrier_gold.csv")
    scale = float(np.median(g_disk))
    X = np.stack([g_gas / scale, g_disk / scale], axis=1)
    y = np.log10(g_obs)
    disc_names, held_names = split_galaxies(gal, 0)
    disc = np.isin(gal, disc_names)
    held = ~disc

    # discovery-only carrier estimate and law
    grid_fit = np.linspace(0.05, 2.5, 99)
    ups_disc, _ = profile_ups(g_gas[disc], g_disk[disc], g_obs[disc], grid_fit)

    # galaxy bootstrap of the profile minimum (full gold sample)
    rng = np.random.default_rng(1)
    names = np.unique(gal)
    by = defaultdict(list)
    for i, g in enumerate(gal):
        by[g].append(i)
    boots = []
    for _ in range(200):
        idx = np.concatenate([by[n] for n in rng.choice(names, len(names), True)])
        boots.append(profile_ups(g_gas[idx], g_disk[idx], g_obs[idx], grid_fit)[0])
    b16, b50, b84 = np.percentile(boots, [16, 50, 84])

    # ---------------------------------------------------------------- fig 1
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for n in names:
        idx = np.array(by[n])
        m = g_gas[idx] > 0
        if m.sum() < 3:
            continue
        ax_a.plot(np.log10(g_gas[idx][m]), np.log10(g_disk[idx][m]),
                  color=GRAY, lw=0.6, alpha=0.55, zorder=1)
    for z0 in (10 ** -11.5, 10 ** -10.75, 10 ** -10.0, 10 ** -9.25):
        gg = np.geomspace(z0 * 3e-3, z0 * 0.999, 200)
        gd = (z0 - gg) / b50
        m = gd > 0
        ax_a.plot(np.log10(gg[m]), np.log10(gd[m]), color=BLUE, lw=0.9,
                  ls=(0, (4, 2)), zorder=2)
    ax_a.text(-12.62, -8.95, r"iso-$z$:  $z=g_{\rm gas}+\Upsilon_d\,g_{\rm disk}$",
              color=BLUE, fontsize=8)
    ax_a.text(-12.62, -12.72, "one line per galaxy", color=MUTED, fontsize=7.5)
    ax_a.set_xlabel(r"$\log_{10}\, g_{\rm gas}\ \,[{\rm m\,s^{-2}}]$")
    ax_a.set_ylabel(r"$\log_{10}\, g_{\rm disk}^{(\Upsilon=1)}\ \,[{\rm m\,s^{-2}}]$")
    ax_a.set_xlim(-12.7, -10.4)
    ax_a.set_ylim(-12.9, -8.7)
    ax_a.set_title("(a)  component space: thin galaxy loci", fontsize=8.5, loc="left")

    ups_grid = np.linspace(-0.4, 2.5, 88)
    scat = np.array([collapse_scatter(u, g_gas, g_disk, g_obs) for u in ups_grid])
    closure = heldout_closure_curve(ups_grid, X, y, disc, held)
    ax_b.axvspan(b16, b84, color=BLUE, alpha=0.13, lw=0, zorder=0)
    ax_b.plot(ups_grid, scat, color=BLUE, lw=1.4, zorder=3)
    ax_b.plot(ups_grid, closure, color=VERM, lw=1.4, zorder=3)
    ax_b.axvline(0.5, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax_b.text(0.55, 0.245, "population synthesis\n" r"$\Upsilon_d\simeq0.5$",
              color=MUTED, fontsize=7)
    ax_b.text(1.52, 0.100, "collapse scatter (all gold galaxies)", color=BLUE,
              fontsize=7.5)
    ax_b.text(1.52, 0.180, "held-out-galaxy\nprediction error", color=VERM,
              fontsize=7.5)
    ax_b.text(2.45, 0.395, rf"galaxy bootstrap 68%:  $\Upsilon_d={b50:.2f}"
              rf"^{{+{b84-b50:.2f}}}_{{-{b50-b16:.2f}}}$",
              color=BLUE, fontsize=7.5, ha="right")
    ax_b.set_xlabel(r"carrier coefficient $\Upsilon_d$")
    ax_b.set_ylabel("rms about 1D law  [dex]")
    ax_b.set_xlim(-0.4, 2.5)
    ax_b.set_ylim(0.09, 0.42)
    ax_b.set_title("(b)  carrier-direction profile", fontsize=8.5, loc="left")
    fig.savefig(FIGDIR / "fig1_carrier_geometry.pdf")
    fig.savefig(FIGDIR / "fig1_carrier_geometry.png", dpi=250)
    plt.close(fig)

    # ---------------------------------------------------------------- fig 2
    z_phys = (X @ [1.0, ups_disc]) * scale
    lg_z = np.log10(np.clip(z_phys, 1e-14, None))

    # quantile-binned median law in log z (same estimator as the profile)
    lz_tr, y_tr = lg_z[disc], y[disc]
    nb = 25
    edges = np.quantile(lz_tr, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(edges, lz_tr, side="right") - 1, 0, nb - 1)
    med = np.array([np.median(y_tr[idx == b]) if np.any(idx == b) else np.nan
                    for b in range(nb)])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    good = ~np.isnan(med)

    def law_fn(q):
        return np.interp(q, ctr[good], med[good])

    lg_zgrid = np.linspace(lz_tr.min(), lz_tr.max(), 220)
    law = law_fn(lg_zgrid)
    res1 = y[held] - law_fn(lg_z[held])
    rms1 = float(np.sqrt(np.mean(res1 ** 2)))

    warp_a = np.array([np.median(np.abs(X[:, 0])), np.median(np.abs(X[:, 1]))])
    U = warp(X, warp_a)
    _m, leaf, _bv = fit_surrogate(U[disc], y[disc], epochs=800, seed=0)
    yh_ho, _g = surrogate_grad_sample(leaf, U[held])
    res2 = y[held] - yh_ho
    rms2 = float(np.sqrt(np.mean(res2 ** 2)))

    fig = plt.figure(figsize=(4.6, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.15], width_ratios=[2.6, 1.0])
    ax = fig.add_subplot(gs[0, :])
    axr = fig.add_subplot(gs[1, 0])
    axh = fig.add_subplot(gs[1, 1])

    zz = np.geomspace(3e-13, 3e-9, 300)
    ax.plot(np.log10(zz), np.log10(rar(zz)), color=TEAL, lw=1.1, ls=(0, (5, 2)),
            zorder=2)
    ax.plot(np.log10(zz), np.log10(zz), color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.scatter(lg_z[disc], y[disc], s=4, color=GRAY, alpha=0.45, lw=0, zorder=3)
    ax.scatter(lg_z[held], y[held], s=9, color=VERM, alpha=0.85, lw=0, zorder=4)
    ax.plot(lg_zgrid, law, color=BLUE, lw=1.6, zorder=5)
    ax.text(-12.05, -10.55, "discovered 1D law\n(fit on discovery only)",
            color=BLUE, fontsize=7.5)
    ax.text(-10.15, -10.72, "RAR interpolation\n(reference, not fitted)",
            color=TEAL, fontsize=7.5)
    ax.text(-9.55, -9.72, r"$g_{\rm obs}=z$", color=MUTED, fontsize=7.5)
    ax.scatter([], [], s=9, color=VERM, label=f"held-out galaxies ({len(held_names)})")
    ax.scatter([], [], s=5, color=GRAY, label=f"discovery galaxies ({len(disc_names)})")
    ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.2)
    ax.set_xlabel(rf"$\log_{{10}}\, z = \log_{{10}}(g_{{\rm gas}}"
                  rf"+{ups_disc:.2f}\,g_{{\rm disk}})\ \,[{{\rm m\,s^{{-2}}}}]$")
    ax.set_ylabel(r"$\log_{10}\, g_{\rm obs}\ \,[{\rm m\,s^{-2}}]$")
    ax.set_xlim(-12.15, -9.3)
    ax.set_ylim(-11.6, -9.3)

    ax.set_title("the discovered coordinate organizes galaxies it never saw",
                 fontsize=8.5, loc="left")

    axr.axhline(0.0, color=MUTED, lw=0.7)
    axr.axhspan(-rms1, rms1, color=VERM, alpha=0.10, lw=0)
    axr.scatter(lg_z[held], res1, s=7, color=VERM, alpha=0.85, lw=0)
    axr.set_xlabel(r"$\log_{10}\, z\ \,[{\rm m\,s^{-2}}]$")
    axr.set_ylabel("residual [dex]")
    axr.set_xlim(-12.15, -9.3)
    axr.set_ylim(-0.75, 0.75)
    axr.text(-12.1, 0.52, f"held-out rms {rms1:.2f} dex", color=VERM, fontsize=7.5)

    bins = np.linspace(-1.6, 1.6, 33)
    axh.hist(res2, bins=bins, orientation="horizontal", color=GRAY, alpha=0.55,
             lw=0, label="2D")
    axh.hist(res1, bins=bins, orientation="horizontal", histtype="step",
             color=VERM, lw=1.3, label="1D")
    axh.axhline(0.0, color=MUTED, lw=0.7)
    axh.set_ylim(-1.6, 1.6)
    axh.set_xticks([])
    axh.set_ylabel("")
    axh.text(0.95, 0.94, f"1D law: {rms1:.2f} dex", color=VERM, fontsize=7,
             ha="right", transform=axh.transAxes)
    axh.text(0.95, 0.84, f"2D fit: {rms2:.2f} dex", color=MUTED, fontsize=7,
             ha="right", transform=axh.transAxes)
    axh.set_title("held-out residuals", fontsize=7.5, loc="left")
    fig.savefig(FIGDIR / "fig2_collapse.pdf")
    fig.savefig(FIGDIR / "fig2_collapse.png", dpi=250)
    plt.close(fig)

    # ---------------------------------------------------------------- fig 3
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
        _mm, lf, _b = fit_surrogate(U[disc], yy[disc], epochs=ep, seed=0)
        yh, Gu = surrogate_grad_sample(lf, U[disc])
        G = Gu * warp_grad_factor(X[disc], warp_a)
        r1, _c = orientation_carrier(G, gal[disc])
        cases[label] = {"tail": determining_tail(X[disc], yh, G),
                        "ups": r1["upsilon_d"]}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                     gridspec_kw={"width_ratios": [1.0, 1.25]})
    kk = np.arange(1, 9)
    ax_a.plot(kk, cases["clean"]["tail"], "o-", color=BLUE, ms=3.5, lw=1.2)
    ax_a.plot(kk, cases["iid"]["tail"], "s-", color=VERM, ms=3.2, lw=1.2)
    ax_a.plot(kk, cases["real"]["tail"], "^-", color=INK, ms=3.4, lw=1.2)
    ax_a.set_yscale("log")
    ax_a.annotate("", xy=(8, cases["clean"]["tail"][-1] * 1.6),
                  xytext=(8, cases["clean"]["tail"][-2] * 0.65),
                  arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.9))
    ax_a.text(6.9, np.sqrt(cases["clean"]["tail"][-1] * cases["clean"]["tail"][-2]),
              "gap 50\n(gate: 10)", color=BLUE, fontsize=7, ha="right", va="center")
    ax_a.text(3.1, 2.2e-4, "planted carrier, no noise", color=BLUE, fontsize=7.5)
    ax_a.text(4.3, 1.1e-1, "planted, 0.12 dex noise", color=VERM, fontsize=7.5)
    ax_a.text(4.4, 6.0e-3, "real SPARC", color=INK, fontsize=7.5)
    ax_a.set_xlabel("singular-value index of the determining operator")
    ax_a.set_ylabel(r"$s_k / s_1$")
    ax_a.set_title("(a)  a symmetry gap exists only in the clean limit",
                   fontsize=8.5, loc="left")

    rows = [
        ("gradient lane\nplanted, clean", cases["clean"]["ups"], None, BLUE, True),
        ("gradient lane\nplanted, 0.12 dex", cases["iid"]["ups"], None, VERM, False),
        ("gradient lane\nplanted, galaxy-corr.", cases["gal"]["ups"], None, VERM, False),
        ("gradient lane\nreal SPARC", cases["real"]["ups"], None, VERM, False),
        ("collapse lane\nreal SPARC", b50, (b50 - b16, b84 - b50), BLUE, True),
    ]
    ypos = np.arange(len(rows))[::-1]
    ax_b.axvline(UPS_TRUE, color=MUTED, lw=0.8, ls=":")
    ax_b.set_ylim(-0.55, len(rows) - 0.25)
    ax_b.text(0.56, len(rows) - 0.52, r"planted / population $\Upsilon_d$",
              color=MUTED, fontsize=7, ha="left")
    for (label, u, err, color, ok), yy in zip(rows, ypos):
        if err is not None:
            ax_b.errorbar([u], [yy], xerr=[[err[0]], [err[1]]], fmt="o",
                          color=color, ms=5, capsize=2.5, lw=1.2)
        else:
            ax_b.plot([u], [yy], "o", color=color, ms=5)
        ax_b.text(1.62, yy, "recovers" if ok else "fails (abstains)",
                  color=color, fontsize=7.5, va="center")
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax_b.set_xlabel(r"recovered carrier coefficient $\Upsilon_d$")
    ax_b.set_xlim(-0.35, 2.1)
    ax_b.set_title(r"(b)  recovered $\Upsilon_d$ by lane and case", fontsize=8.5, loc="left")
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    fig.savefig(FIGDIR / "fig3_certificate.pdf")
    fig.savefig(FIGDIR / "fig3_certificate.png", dpi=250)
    plt.close(fig)

    print(f"ups_disc={ups_disc:.3f}  boot [{b16:.3f},{b50:.3f},{b84:.3f}]  "
          f"rms1={rms1:.3f} rms2={rms2:.3f}")
    print(f"orientation ups by case: "
          + ", ".join(f"{k}:{v['ups']:+.3f}" for k, v in cases.items()))
    print(f"wrote {FIGDIR}/fig{{1,2,3}}*.pdf/png")


if __name__ == "__main__":
    main()
