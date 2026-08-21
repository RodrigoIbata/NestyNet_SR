#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Generate a conceptual slide focused on factorized symbolic search mapping + fingerprinting.

Usage:
    python scripts/make_skeleton_refinement_concept_slide.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


def _panel(ax, x, y, w, h, *, face, edge, title, title_color, body=None, body_color=None):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.16",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.35,
        y + h - 0.42,
        title,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=title_color,
        zorder=4,
    )
    if body:
        ax.text(
            x + 0.35,
            y + h - 0.82,
            body,
            ha="left",
            va="center",
            fontsize=9.5,
            color=body_color if body_color else title_color,
            zorder=4,
        )


def _box(ax, x, y, w, h, *, face, edge, text, text_color, size=9, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=text_color,
        fontweight=weight,
        zorder=4,
        linespacing=1.15,
    )


def _arrow(ax, start, end, *, color, lw=1.8, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=5,
    )
    ax.add_patch(patch)


def _draw_mapping_panel(ax, palette):
    _panel(
        ax,
        0.7,
        1.65,
        9.0,
        8.2,
        face=palette["left_panel"],
        edge=palette["left_edge"],
        title="1) Geometric Mapping View",
        title_color=palette["ink"],
        body=r"$\mathbf{x}\;\rightarrow\; z=f(\mathbf{x})\;\rightarrow\;\hat y=m(z)$",
        body_color=palette["muted"],
    )

    # Raw variable space
    raw = Rectangle((1.15, 3.7), 2.45, 3.15, linewidth=1.1, edgecolor=palette["left_edge"], facecolor="white", zorder=2)
    ax.add_patch(raw)
    ax.text(1.22, 6.98, r"Raw space: $\mathbf{x}$", fontsize=8.5, color=palette["muted"], va="bottom")

    rng = np.random.default_rng(9)
    px = 1.2 + 2.3 * rng.random(46)
    py = 3.9 + 2.7 * rng.random(46)
    ax.scatter(px, py, s=16, color=palette["dot1"], alpha=0.88, edgecolors="none", zorder=3)

    # Candidate collective coordinate
    _box(
        ax,
        4.25,
        5.08,
        2.25,
        1.48,
        face=palette["chip"],
        edge=palette["left_edge"],
        text="Candidate skeleton\n$z = f(\\mathbf{x})$\n(constant-light)",
        text_color=palette["ink"],
        size=8.8,
    )

    # Collapsed 1D coordinate axis
    collapse = Rectangle((7.05, 4.4), 2.15, 2.45, linewidth=1.1, edgecolor=palette["left_edge"], facecolor="white", zorder=2)
    ax.add_patch(collapse)
    ax.text(7.14, 6.98, r"Collapsed coordinate: $z$", fontsize=8.5, color=palette["muted"], va="bottom")
    ax.plot([7.25, 9.0], [5.0, 5.0], color=palette["ink"], linewidth=1.0, zorder=3)
    zvals = np.linspace(7.34, 8.86, 13)
    ax.scatter(zvals, 5.0 + 0.11 * np.sin(np.linspace(0, 2 * np.pi, len(zvals))), s=18, color=palette["dot2"], zorder=3)

    # Mapping family fit
    map_box = Rectangle((4.1, 2.3), 5.35, 1.9, linewidth=1.1, edgecolor=palette["left_edge"], facecolor="white", zorder=2)
    ax.add_patch(map_box)
    ax.text(
        4.22,
        4.06,
        "Fit simple 1D mapping family $m(z)$: polynomial / power / Pade / sinusoid / exponential",
        fontsize=8.3,
        color=palette["muted"],
        va="bottom",
    )
    ax.plot([4.45, 4.45], [2.6, 3.85], color=palette["ink"], linewidth=1.0)
    ax.plot([4.45, 9.2], [2.6, 2.6], color=palette["ink"], linewidth=1.0)
    zx = np.linspace(4.55, 9.0, 120)
    t = (zx - 4.55) / (9.0 - 4.55)
    zy = 2.88 + 0.26 * np.sin(2.8 * np.pi * t) + 0.58 * t
    ax.plot(zx, zy, color=palette["accent"], linewidth=2.0, zorder=4)
    ax.scatter(
        4.55 + 4.35 * rng.random(18),
        2.82 + 1.0 * rng.random(18),
        s=14,
        color=palette["dot2"],
        alpha=0.55,
        edgecolors="none",
        zorder=3,
    )

    _arrow(ax, (3.62, 5.28), (4.25, 5.28), color=palette["left_edge"])
    _arrow(ax, (6.52, 5.28), (7.05, 5.28), color=palette["left_edge"])
    _arrow(ax, (8.1, 4.35), (7.25, 4.08), color=palette["left_edge"])


def _draw_fingerprint_panel(ax, palette):
    _panel(
        ax,
        10.3,
        1.65,
        9.0,
        8.2,
        face=palette["right_panel"],
        edge=palette["right_edge"],
        title="2) Fingerprint Residual-Basin View",
        title_color=palette["ink"],
        body=r"Residual geometry defines residual_basin identity: $r=y-m(f(\mathbf{x}))$",
        body_color=palette["muted"],
    )

    # Residual vector
    rbox = Rectangle((10.75, 5.7), 2.4, 2.35, linewidth=1.1, edgecolor=palette["right_edge"], facecolor="white", zorder=2)
    ax.add_patch(rbox)
    ax.text(10.9, 8.12, "Residual profile $r$", fontsize=8.5, color=palette["muted"], va="bottom")
    bars = np.array([0.25, 0.67, 0.35, 0.74, 0.43, 0.54, 0.3, 0.82])
    x0 = 10.95
    for i, b in enumerate(bars):
        ax.add_patch(
            Rectangle(
                (x0 + i * 0.24, 5.92),
                0.14,
                1.78 * b,
                linewidth=0,
                facecolor=palette["bar"],
                alpha=0.88,
                zorder=3,
            )
        )

    _box(
        ax,
        13.45,
        6.05,
        1.95,
        1.5,
        face=palette["chip2"],
        edge=palette["right_edge"],
        text="Random\nprojection\n$u = Rr$",
        text_color=palette["ink"],
        size=8.4,
    )
    _box(
        ax,
        15.75,
        6.05,
        1.55,
        1.5,
        face=palette["chip2"],
        edge=palette["right_edge"],
        text="Quantize\n$q = Q(u)$",
        text_color=palette["ink"],
        size=8.4,
    )
    _box(
        ax,
        17.58,
        6.05,
        1.25,
        1.5,
        face=palette["chip3"],
        edge=palette["right_edge"],
        text="key\n10110110",
        text_color=palette["ink"],
        size=8.2,
        weight="bold",
    )

    _arrow(ax, (13.14, 6.8), (13.45, 6.8), color=palette["right_edge"])
    _arrow(ax, (15.4, 6.8), (15.75, 6.8), color=palette["right_edge"])
    _arrow(ax, (17.3, 6.8), (17.58, 6.8), color=palette["right_edge"])

    # ResidualBasin archive geometry
    residual_basin = Rectangle((10.75, 2.45), 8.1, 2.65, linewidth=1.1, edgecolor=palette["right_edge"], facecolor="white", zorder=2)
    ax.add_patch(residual_basin)
    ax.text(10.9, 5.14, "Archive in fingerprint space (one best expression per residual_basin)", fontsize=8.5, color=palette["muted"], va="bottom")
    ax.plot([11.2, 11.2], [2.85, 4.8], color=palette["muted"], linewidth=0.9)
    ax.plot([11.2, 18.35], [2.85, 2.85], color=palette["muted"], linewidth=0.9)
    ax.text(18.32, 2.64, "fp-1", fontsize=7.5, color=palette["muted"], ha="right")
    ax.text(11.03, 4.74, "fp-2", fontsize=7.5, color=palette["muted"], rotation=90, va="top")

    rng = np.random.default_rng(21)
    cx = 11.45 + 6.65 * rng.random(38)
    cy = 3.05 + 1.45 * rng.random(38)
    colors = np.where((cx + 0.4 * cy) > 14.8, palette["dot3"], palette["dot4"])
    ax.scatter(cx, cy, s=26, color=colors, alpha=0.82, edgecolors="white", linewidths=0.25, zorder=3)

    center = (16.4, 3.9)
    ax.add_patch(Circle(center, 0.47, edgecolor=palette["accent2"], facecolor="none", linewidth=2.1, zorder=4))
    ax.text(center[0] + 0.58, center[1] + 0.08, "current residual_basin", fontsize=8.2, color=palette["accent2"], va="center")
    _arrow(ax, (18.2, 6.0), (16.7, 4.35), color=palette["right_edge"], rad=0.04, lw=1.5)


def _draw_refine_band(ax, palette):
    _panel(
        ax,
        0.7,
        0.35,
        18.6,
        1.1,
        face=palette["band"],
        edge=palette["band_edge"],
        title="continuous skeleton refinement on promising candidates only",
        title_color=palette["ink"],
        body=(
            "Insert internal scales (e.g., sin(θu), exp(θu), log(θu)); optimize θ with LBFGS; "
            "solve additive linear amplitudes by least squares; then re-fingerprint and re-archive."
        ),
        body_color=palette["muted"],
    )


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(16, 9), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11.25)
    ax.axis("off")

    palette = {
        "bg": "#f8f7f3",
        "ink": "#101828",
        "muted": "#475467",
        "accent": "#0f766e",
        "accent2": "#115e59",
        "left_panel": "#e8eef8",
        "left_edge": "#2f4b68",
        "right_panel": "#e8f6ef",
        "right_edge": "#255f53",
        "chip": "#c9dbee",
        "chip2": "#c7e8dc",
        "chip3": "#f5d999",
        "band": "#f1f5f9",
        "band_edge": "#94a3b8",
        "bar": "#4f46e5",
        "dot1": "#2563eb",
        "dot2": "#0369a1",
        "dot3": "#059669",
        "dot4": "#0ea5a4",
    }
    fig.patch.set_facecolor(palette["bg"])

    ax.text(
        0.7,
        10.78,
        "factorized symbolic search as Geometric Mapping + Residual Fingerprinting",
        ha="left",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=palette["ink"],
    )
    ax.text(
        0.7,
        10.28,
        "Physics-native story: find a compact collective coordinate, fit a simple mapping, then explore distinct residual-basin families.",
        ha="left",
        va="center",
        fontsize=11.2,
        color=palette["muted"],
    )

    _draw_mapping_panel(ax, palette)
    _draw_fingerprint_panel(ax, palette)
    _draw_refine_band(ax, palette)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate continuous skeleton refinement conceptual slide figure")
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("docs/source/_static/skeleton_refinement_concept_slide.svg"),
        help="output SVG path",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("docs/source/_static/skeleton_refinement_concept_slide.png"),
        help="output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI")
    args = parser.parse_args()

    fig = build_figure()
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.svg, format="svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(args.png, format="png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote: {args.svg}")
    print(f"Wrote: {args.png}")


if __name__ == "__main__":
    main()
