#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Generate two talk-ready conceptual slides for factorized symbolic search/continuous skeleton refinement.

Outputs (default):
- docs/source/_static/factorized_search_talk_slide1_mapping_fingerprint.{svg,png}
- docs/source/_static/factorized_search_talk_slide2_directions.{svg,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


def _panel(ax, x, y, w, h, *, face, edge, lw=1.5):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.16",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=1,
    )
    ax.add_patch(box)


def _box(ax, x, y, w, h, *, face, edge, text, color, fs=9.0, weight="normal"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        fontweight=weight,
        linespacing=1.15,
        zorder=4,
    )


def _arrow(ax, start, end, *, color, lw=1.8, rad=0.0, ms=14):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=5,
    )
    ax.add_patch(arr)


def _setup_figure():
    fig = plt.figure(figsize=(16, 9), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11.25)
    ax.axis("off")
    return fig, ax


def build_slide1() -> plt.Figure:
    fig, ax = _setup_figure()
    palette = {
        "bg": "#f8f7f2",
        "ink": "#0f172a",
        "muted": "#475569",
        "left": "#e7edf8",
        "left_edge": "#355272",
        "right": "#e8f4ef",
        "right_edge": "#256457",
        "chip": "#c8d8ea",
        "chip2": "#cae8dd",
        "gold": "#f2d48d",
        "accent": "#0f766e",
        "dot1": "#2563eb",
        "dot2": "#0ea5a4",
        "dot3": "#22c55e",
        "dot4": "#0891b2",
    }
    fig.patch.set_facecolor(palette["bg"])

    ax.text(
        0.7,
        10.78,
        "Factorized symbolic search: Find a Good Collective Coordinate, Then Track Residual-Basin Families",
        ha="left",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=palette["ink"],
    )
    ax.text(
        0.7,
        10.27,
        r"$\mathbf{x} \rightarrow z=f(\mathbf{x}) \rightarrow \hat y=m(z)$  and  residual_basin key = fingerprint$(y-\hat y)$",
        ha="left",
        va="center",
        fontsize=12,
        color=palette["muted"],
    )

    _panel(ax, 0.7, 1.45, 9.0, 8.35, face=palette["left"], edge=palette["left_edge"])
    _panel(ax, 10.3, 1.45, 9.0, 8.35, face=palette["right"], edge=palette["right_edge"])

    ax.text(1.05, 9.26, "1) Geometric Mapping", fontsize=16, fontweight="bold", color=palette["ink"])
    ax.text(10.65, 9.26, "2) Residual Fingerprinting", fontsize=16, fontweight="bold", color=palette["ink"])

    # Left panel: x -> z -> m(z)
    ax.add_patch(Rectangle((1.2, 3.5), 2.6, 3.2, linewidth=1.1, edgecolor=palette["left_edge"], facecolor="white", zorder=2))
    ax.text(1.28, 6.85, r"raw data in $\mathbf{x}$", fontsize=9, color=palette["muted"])
    rng = np.random.default_rng(2)
    px = 1.32 + 2.35 * rng.random(50)
    py = 3.75 + 2.65 * rng.random(50)
    ax.scatter(px, py, s=15, color=palette["dot1"], alpha=0.88, edgecolors="none", zorder=3)

    _box(
        ax,
        4.42,
        5.2,
        2.18,
        1.35,
        face=palette["chip"],
        edge=palette["left_edge"],
        text="symbolic skeleton\n$z=f(\\mathbf{x})$",
        color=palette["ink"],
        fs=10,
        weight="bold",
    )

    ax.add_patch(Rectangle((7.1, 4.1), 2.15, 2.6, linewidth=1.1, edgecolor=palette["left_edge"], facecolor="white", zorder=2))
    ax.text(7.2, 6.85, r"1D coordinate $z$", fontsize=9, color=palette["muted"])
    ax.plot([7.28, 9.0], [5.1, 5.1], color=palette["ink"], linewidth=1.0, zorder=3)
    zvals = np.linspace(7.35, 8.9, 13)
    ax.scatter(zvals, 5.1 + 0.09 * np.sin(np.linspace(0, 2 * np.pi, len(zvals))), s=17, color=palette["dot4"], zorder=3)

    ax.add_patch(Rectangle((4.1, 2.15), 5.35, 1.65, linewidth=1.1, edgecolor=palette["left_edge"], facecolor="white", zorder=2))
    ax.text(4.2, 3.86, r"fit simple mapping $m(z)$", fontsize=9, color=palette["muted"])
    ax.plot([4.45, 4.45], [2.42, 3.48], color=palette["ink"], linewidth=1.0)
    ax.plot([4.45, 9.2], [2.42, 2.42], color=palette["ink"], linewidth=1.0)
    zx = np.linspace(4.58, 9.0, 160)
    t = (zx - 4.58) / (9.0 - 4.58)
    zy = 2.72 + 0.24 * np.sin(2.6 * np.pi * t) + 0.5 * t
    ax.plot(zx, zy, color=palette["accent"], linewidth=2.1, zorder=4)

    _arrow(ax, (3.82, 5.9), (4.42, 5.9), color=palette["left_edge"])
    _arrow(ax, (6.6, 5.9), (7.1, 5.9), color=palette["left_edge"])
    _arrow(ax, (8.2, 4.05), (7.28, 3.63), color=palette["left_edge"])

    # Right panel: residual -> projection -> quantize -> key -> archive
    ax.add_patch(Rectangle((10.8, 5.7), 2.5, 2.05, linewidth=1.1, edgecolor=palette["right_edge"], facecolor="white", zorder=2))
    ax.text(10.92, 7.86, r"residual vector $r=y-\hat y$", fontsize=9, color=palette["muted"])
    bars = np.array([0.27, 0.7, 0.35, 0.78, 0.43, 0.58, 0.36, 0.83])
    x0 = 10.98
    for i, b in enumerate(bars):
        ax.add_patch(Rectangle((x0 + i * 0.24, 5.9), 0.14, 1.6 * b, linewidth=0, facecolor="#5b5bd6", alpha=0.9, zorder=3))

    _box(ax, 13.6, 5.98, 2.0, 1.45, face=palette["chip2"], edge=palette["right_edge"], text="project\n$u=Rr$", color=palette["ink"], fs=9.8)
    _box(ax, 15.95, 5.98, 1.7, 1.45, face=palette["chip2"], edge=palette["right_edge"], text="quantize\n$q=Q(u)$", color=palette["ink"], fs=9.4)
    _box(ax, 18.0, 5.98, 1.3, 1.45, face=palette["gold"], edge=palette["right_edge"], text="residual_basin key\n10110110", color=palette["ink"], fs=9.0, weight="bold")
    _arrow(ax, (13.3, 6.68), (13.6, 6.68), color=palette["right_edge"])
    _arrow(ax, (15.6, 6.68), (15.95, 6.68), color=palette["right_edge"])
    _arrow(ax, (17.65, 6.68), (18.0, 6.68), color=palette["right_edge"])

    ax.add_patch(Rectangle((10.8, 2.35), 8.45, 2.55, linewidth=1.1, edgecolor=palette["right_edge"], facecolor="white", zorder=2))
    ax.text(10.93, 5.02, "archive in fingerprint space", fontsize=9, color=palette["muted"])
    ax.plot([11.25, 11.25], [2.72, 4.55], color=palette["muted"], linewidth=0.9)
    ax.plot([11.25, 18.75], [2.72, 2.72], color=palette["muted"], linewidth=0.9)

    rng2 = np.random.default_rng(21)
    cx = 11.5 + 6.95 * rng2.random(38)
    cy = 3.0 + 1.4 * rng2.random(38)
    colors = np.where(cx > 15.7, palette["dot3"], palette["dot2"])
    ax.scatter(cx, cy, s=25, color=colors, alpha=0.85, edgecolors="white", linewidths=0.25, zorder=3)

    center = (16.4, 3.82)
    ax.add_patch(Circle(center, 0.45, edgecolor=palette["accent"], facecolor="none", linewidth=2.2, zorder=4))
    ax.text(16.95, 3.9, "current residual_basin", fontsize=8.8, color=palette["accent"], va="center")
    _arrow(ax, (18.65, 5.92), (16.75, 4.25), color=palette["right_edge"], lw=1.6)

    # Bottom takeaways
    _panel(ax, 0.7, 0.28, 18.6, 0.92, face="#eff3f9", edge="#94a3b8", lw=1.2)
    ax.text(1.0, 0.8, "Why this helps: constants and outer transforms are handled in m(z), while residual_basin keys prevent re-polishing the same residual family.", fontsize=10.3, color=palette["muted"], va="center")

    return fig


def build_slide2() -> plt.Figure:
    fig, ax = _setup_figure()
    palette = {
        "bg": "#fbf8f2",
        "ink": "#111827",
        "muted": "#475569",
        "p1": "#e8eef8",
        "p1e": "#355272",
        "p2": "#e8f6ef",
        "p2e": "#236555",
        "chip1": "#d4e0ef",
        "chip2": "#cde9de",
        "gold": "#f5d79a",
        "accent": "#0f766e",
        "accent2": "#1d4ed8",
    }
    fig.patch.set_facecolor(palette["bg"])

    ax.text(
        0.7,
        10.78,
        "Where Does the Improvement Direction Come From?",
        ha="left",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=palette["ink"],
    )
    ax.text(
        0.7,
        10.27,
        "factorized symbolic search uses a residual-aligned symbolic direction; continuous skeleton refinement adds local gradient directions on internal scales.",
        ha="left",
        va="center",
        fontsize=11.4,
        color=palette["muted"],
    )

    _panel(ax, 0.7, 4.72, 18.6, 5.0, face=palette["p1"], edge=palette["p1e"])
    ax.text(1.0, 9.2, "A) Discrete Direction in Symbolic Space (factorized symbolic search)", fontsize=15, fontweight="bold", color=palette["ink"])

    _box(ax, 1.0, 7.25, 2.35, 1.35, face=palette["chip1"], edge=palette["p1e"], text="Current best\nexpr + mapping", color=palette["ink"], fs=9.4, weight="bold")
    _box(ax, 3.9, 7.25, 2.65, 1.35, face=palette["chip1"], edge=palette["p1e"], text="Residual\n$r=y-\\hat y$", color=palette["ink"], fs=10.1, weight="bold")

    ax.add_patch(Rectangle((6.95, 6.65), 4.35, 2.0, linewidth=1.1, edgecolor=palette["p1e"], facecolor="white", zorder=2))
    ax.text(7.12, 8.43, r"Direction score for pool term $\phi_j$", fontsize=9.6, color=palette["muted"])
    ax.text(7.12, 7.88, r"$s_j = (r\cdot\phi_j)^2/(\|\phi_j\|^2+\epsilon)$", fontsize=14, color=palette["accent2"], fontweight="bold")
    ax.text(7.12, 7.28, r"pick top-k terms (residual-guided proposals)", fontsize=9.6, color=palette["muted"])

    # Tiny vector sketch for the projection idea.
    ax.plot([9.72, 10.98], [6.94, 6.94], color=palette["muted"], linewidth=0.9, zorder=3)
    ax.plot([9.72, 9.72], [6.94, 8.15], color=palette["muted"], linewidth=0.9, zorder=3)
    _arrow(ax, (9.72, 6.94), (10.58, 7.84), color=palette["accent2"], lw=1.5, ms=10)
    _arrow(ax, (9.72, 6.94), (10.24, 7.17), color=palette["accent"], lw=1.4, ms=10)
    ax.text(10.6, 7.86, r"$r$", fontsize=8.5, color=palette["accent2"], va="center")
    ax.text(10.26, 7.2, r"$\phi_j$", fontsize=8.5, color=palette["accent"], va="center")
    ax.plot([10.24, 10.1], [7.17, 7.66], color=palette["accent"], linewidth=1.0, zorder=4)
    ax.text(10.04, 7.7, r"$r\!\cdot\!\phi_j$", fontsize=7.8, color=palette["muted"], va="center")

    _box(ax, 11.85, 7.25, 2.75, 1.35, face=palette["chip1"], edge=palette["p1e"], text="Generate candidates\n(root-add / subtree-add /\nleaf-replace)", color=palette["ink"], fs=8.9)
    _box(ax, 15.05, 7.25, 1.9, 1.35, face=palette["chip1"], edge=palette["p1e"], text="Refit m(z)\n+ score MSE", color=palette["ink"], fs=9.2)
    _box(ax, 17.25, 7.25, 1.8, 1.35, face=palette["gold"], edge=palette["p1e"], text="Keep best\nif improves", color=palette["ink"], fs=9.2, weight="bold")

    _arrow(ax, (3.35, 7.92), (3.9, 7.92), color=palette["p1e"])
    _arrow(ax, (6.55, 7.92), (6.95, 7.92), color=palette["p1e"])
    _arrow(ax, (11.3, 7.92), (11.85, 7.92), color=palette["p1e"])
    _arrow(ax, (14.6, 7.92), (15.05, 7.92), color=palette["p1e"])
    _arrow(ax, (16.95, 7.92), (17.25, 7.92), color=palette["p1e"])

    # UCB / archive feedback
    ax.add_patch(Rectangle((1.0, 5.2), 8.1, 1.7, linewidth=1.1, edgecolor=palette["p1e"], facecolor="white", zorder=2))
    ax.text(1.2, 6.58, "How action direction is chosen next", fontsize=10.2, color=palette["muted"], fontweight="bold")
    ax.text(1.2, 6.1, r"reward = log(parent_mse) - log(child_mse) + novelty_bonus", fontsize=10.1, color=palette["ink"])
    ax.text(1.2, 5.65, "UCB policy balances high-reward actions and underexplored actions", fontsize=10.1, color=palette["ink"])

    ax.add_patch(Rectangle((9.55, 5.2), 9.5, 1.7, linewidth=1.1, edgecolor=palette["p1e"], facecolor="white", zorder=2))
    ax.text(9.78, 6.58, "Archive feedback", fontsize=10.2, color=palette["muted"], fontweight="bold")
    ax.text(9.78, 6.1, "Fingerprint key decides residual_basin identity; one best representative per residual_basin", fontsize=10.1, color=palette["ink"])
    ax.text(9.78, 5.65, "Parent selection favors good local optimums but also underexplored ones", fontsize=10.1, color=palette["ink"])

    _arrow(ax, (18.15, 7.22), (15.4, 6.92), color=palette["p1e"], rad=-0.08, lw=1.4, ms=12)
    _arrow(ax, (13.0, 5.15), (4.9, 6.95), color=palette["p1e"], rad=0.18, lw=1.4, ms=12)

    # Lower panel: continuous skeleton refinement continuous direction
    _panel(ax, 0.7, 0.4, 18.6, 3.95, face=palette["p2"], edge=palette["p2e"])
    ax.text(1.0, 3.9, "B) Continuous Direction in Parameter Space (continuous skeleton refinement)", fontsize=15, fontweight="bold", color=palette["ink"])

    _box(
        ax,
        1.0,
        1.7,
        4.0,
        1.65,
        face=palette["chip2"],
        edge=palette["p2e"],
        text="Insert 1-2 scale params\n$\\sin(u)\\to\\sin(\\theta u)$\n$\\exp(u)\\to\\exp(\\theta u)$\n$\\log(u)\\to\\log(\\theta u)$",
        color=palette["ink"],
        fs=8.9,
    )

    ax.add_patch(Rectangle((5.45, 1.65), 6.05, 1.75, linewidth=1.1, edgecolor=palette["p2e"], facecolor="white", zorder=2))
    ax.text(5.7, 3.06, r"Optimize direction by gradient in $\theta$ (LBFGS)", fontsize=10.0, color=palette["muted"])
    ax.text(5.7, 2.52, r"$\mathcal{L}(\theta)=\mathrm{MSE}+\lambda\|\theta\|^2+\mathrm{safe\ penalties}$", fontsize=12.5, color=palette["accent"], fontweight="bold")
    ax.text(5.7, 2.02, "Linear amplitudes solved by least squares (variable projection)", fontsize=10.0, color=palette["ink"])

    _box(ax, 11.95, 1.7, 3.0, 1.65, face=palette["chip2"], edge=palette["p2e"], text="Materialize refined\nexpression + rescore", color=palette["ink"], fs=9.4)
    _box(ax, 15.4, 1.7, 3.7, 1.65, face="#d7f1e5", edge=palette["p2e"], text="Accept if MSE improves\nthen re-fingerprint and re-archive", color=palette["ink"], fs=9.6, weight="bold")

    _arrow(ax, (5.0, 2.52), (5.45, 2.52), color=palette["p2e"])
    _arrow(ax, (11.5, 2.52), (11.95, 2.52), color=palette["p2e"])
    _arrow(ax, (14.95, 2.52), (15.4, 2.52), color=palette["p2e"])

    return fig


def save_figure(fig: plt.Figure, svg_path: Path, png_path: Path, dpi: int):
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate two talk slides for factorized symbolic search/continuous skeleton refinement")
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI")
    parser.add_argument(
        "--slide1_svg",
        type=Path,
        default=Path("docs/source/_static/factorized_search_talk_slide1_mapping_fingerprint.svg"),
    )
    parser.add_argument(
        "--slide1_png",
        type=Path,
        default=Path("docs/source/_static/factorized_search_talk_slide1_mapping_fingerprint.png"),
    )
    parser.add_argument(
        "--slide2_svg",
        type=Path,
        default=Path("docs/source/_static/factorized_search_talk_slide2_directions.svg"),
    )
    parser.add_argument(
        "--slide2_png",
        type=Path,
        default=Path("docs/source/_static/factorized_search_talk_slide2_directions.png"),
    )
    args = parser.parse_args()

    fig1 = build_slide1()
    save_figure(fig1, args.slide1_svg, args.slide1_png, args.dpi)
    print(f"Wrote: {args.slide1_svg}")
    print(f"Wrote: {args.slide1_png}")

    fig2 = build_slide2()
    save_figure(fig2, args.slide2_svg, args.slide2_png, args.dpi)
    print(f"Wrote: {args.slide2_svg}")
    print(f"Wrote: {args.slide2_png}")


if __name__ == "__main__":
    main()
