# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Figure 4: the outer-map family and the sister slope posterior s(z)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nuisance_release import load_gold
from outer_map import UPS, make_candidates
from run_pilot import split_galaxies

HERE = Path(__file__).resolve().parent
BLUE, VERM, TEAL = "#0072B2", "#D55E00", "#009E73"
GRAY, INK, MUTED = "#888888", "#1a1a1a", "#666666"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})

FAM_STYLE = {
    "rar_exp": (TEAL, (0, (5, 2)), "RAR interpolation"),
    "powerlaw": (INK, (0, (1.5, 1.5)), "power law"),
    "superpos": (MUTED, (0, (4, 1.5, 1, 1.5)), "Newton + deep-regime sum"),
}


def main():
    ss = np.load(HERE / "results" / "sister_slope.npz", allow_pickle=True)
    fam = np.load(HERE / "results" / "outer_map_family.npy",
                  allow_pickle=True).item()
    cands = make_candidates()

    gal, g_gas, g_disk, g_obs, _e, _m = load_gold()
    z = g_gas + UPS * g_disk
    ok = z > 0
    gal, lgz_d, y_d = gal[ok], np.log10(z[ok]), np.log10(g_obs[ok])
    disc = np.isin(gal, split_galaxies(gal, 0)[0])

    lgz = ss["lgz"]
    cert = ss["certified"].astype(bool)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.1))

    # (a) F(z): data + sister band + family curves
    ax_a.scatter(lgz_d[disc], y_d[disc], s=3.5, color=GRAY, alpha=0.35, lw=0,
                 zorder=1)
    ax_a.scatter(lgz_d[~disc], y_d[~disc], s=5.5, color=VERM, alpha=0.6, lw=0,
                 zorder=2)
    ax_a.fill_between(lgz, ss["f_lo"], ss["f_hi"], color=BLUE, alpha=0.30,
                      lw=0, zorder=3)
    for name, (color, ls, label) in FAM_STYLE.items():
        fn, _ = cands[name]
        ax_a.plot(lgz, fn(10.0 ** lgz, fam[name]["p"]), color=color, ls=ls,
                  lw=1.1, zorder=4, label=label)
    ax_a.plot([], [], color=BLUE, lw=5, alpha=0.35, label="sister 68% band")
    ax_a.legend(loc="upper left", fontsize=7)
    ax_a.set_xlabel(r"$\log_{10}\, z\ \,[{\rm m\,s^{-2}}]$")
    ax_a.set_ylabel(r"$\log_{10}\, g_{\rm obs}\ \,[{\rm m\,s^{-2}}]$")
    ax_a.set_xlim(lgz.min(), lgz.max())
    ax_a.set_title("(a)  every candidate law lives inside the band",
                   fontsize=8.5, loc="left")

    # (b) slope posterior with certified domain
    seg = np.ma.masked_where(~cert, ss["s_med"])
    lo = np.ma.masked_where(~cert, ss["s_lo"])
    hi = np.ma.masked_where(~cert, ss["s_hi"])
    ax_b.fill_between(lgz, lo, hi, color=BLUE, alpha=0.30, lw=0, zorder=3)
    ax_b.plot(lgz, seg, color=BLUE, lw=1.4, zorder=4)
    if (~cert).any():
        for k in np.flatnonzero(np.diff(np.r_[False, ~cert, False]) != 0
                                ).reshape(-1, 2):
            ax_b.axvspan(lgz[k[0]], lgz[min(k[1], len(lgz) - 1)],
                         color=GRAY, alpha=0.12, lw=0, zorder=1)
    for name, (color, ls, _label) in FAM_STYLE.items():
        ax_b.plot(lgz, ss[f"slope_{name}"], color=color, ls=ls, lw=1.1,
                  zorder=2)
    ax_b.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax_b.axhline(0.5, color=MUTED, lw=0.7, ls=":")
    ax_b.text(-11.7, 1.02, "Newtonian  $s=1$", color=MUTED, fontsize=7)
    ax_b.text(-9.85, 0.42, "deep regime  $s=\\frac{1}{2}$", color=MUTED,
              fontsize=7)
    ax_b.text(-11.05, 0.80, "sister posterior\n(certified domain)",
              color=BLUE, fontsize=7.5)
    ax_b.text(-11.62, 0.30, "grey: rcond/adequacy\naudit fails, no claim",
              color=MUTED, fontsize=6.5)
    ax_b.set_xlabel(r"$\log_{10}\, z\ \,[{\rm m\,s^{-2}}]$")
    ax_b.set_ylabel(r"$s(z)=d\log F/d\log z$")
    ax_b.set_xlim(lgz.min(), lgz.max())
    ax_b.set_ylim(0.15, 1.15)
    ax_b.set_title("(b)  what the data constrain about the law",
                   fontsize=8.5, loc="left")

    fig.savefig(HERE / "figures" / "fig4_outer_map.pdf")
    fig.savefig(HERE / "figures" / "fig4_outer_map.png", dpi=250)
    print("wrote figures/fig4_outer_map.{pdf,png}")


if __name__ == "__main__":
    main()
