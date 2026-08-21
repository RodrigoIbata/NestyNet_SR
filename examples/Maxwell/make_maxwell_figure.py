#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Maxwell broad-library summary figure, driven entirely by run data.

Three rows: field snapshot -> library audit (Gram / mu) -> sparse selection.
All quantitative content is generated from the actual pipeline -- field
snapshots from the data generators, Gram matrices and mu from the conditioning
audit, and recovered coefficients from the perfect-information discovery runs.
Nothing is curated.  This is the cheap, no-training (exact-derivative) path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from nestynet_sr.sr_de.system_de_search import (  # noqa: E402
    VectorSystemDESearchConfig,
    discover_vector_system_de_from_surrogate,
)

from conditioning_audit import apply_alias_drops, audit_vector_library  # noqa: E402
from problem_defs import PROBLEM_REGISTRY, _vec_key, build_problem_data, build_vector_terms  # noqa: E402
from spectral_derivatives import build_spectral_spatial_hessian_diag  # noqa: E402
from tabulated_surrogate import TabulatedVectorSurrogate  # noqa: E402

# Fixed display order and labels for the 7-operator union library.
DISPLAY_TERMS = ["curl(E)", "curl(B)", "laplacian(E)", "laplacian(B)", "E", "B", "J"]
DISPLAY_LABELS = [r"$\nabla\!\times\!E$", r"$\nabla\!\times\!B$", r"$\nabla^2 E$",
                  r"$\nabla^2 B$", r"$E$", r"$B$", r"$J$"]
O = len(DISPLAY_TERMS)

CASES = ["mw000", "mw003", "mw002", "mw001"]
CASE_TITLES = ["vacuum single mode", "vacuum multimode", "conductive medium", "wire source"]


# ---------------------------------------------------------------------------
# Run the perfect-information pipeline for one case -> audit + selection
# ---------------------------------------------------------------------------


def run_case(cid: str) -> dict:
    problem = PROBLEM_REGISTRY[cid]
    X, Y, G, _meta = build_problem_data(problem, fast=False)
    H = build_spectral_spatial_hessian_diag(X, Y, spatial_cols=tuple(problem.spatial_axes), time_col=0)
    surrogate = TabulatedVectorSurrogate(X, Y, G, H)

    vector_terms, name_by_key, named = build_vector_terms(problem, include_laplacian=True)
    orig_lib_names = set(name_by_key.values())
    audit = audit_vector_library(surrogate, X, vector_terms, name_by_key, problem.equations, return_gram=True)

    # Gram for the dE/dt (Ampere) equation -- the richer one (carries J/sigma E).
    eq0 = audit["equations"][0]
    gram_engine = np.asarray(eq0["gram"], dtype=float)
    term_names = audit["term_names"]
    gram_disp = _to_display_matrix(gram_engine, term_names)

    # Discovery on the (de-aliased) library.
    vt, nbk, nv = vector_terms, name_by_key, named
    drop_names = set(audit["drop_terms"]) if audit["rank_status"] == "NONIDENTIFIABLE_ALIAS" else set()
    if drop_names:
        vt, nbk, nv = apply_alias_drops(vector_terms, name_by_key, named, drop_names)
    cfg = VectorSystemDESearchConfig(
        x_axis=0, order_candidates=(1,), include_const=False, stlsq_lambda=5e-4,
        stlsq_max_iter=20, sparsity_penalty=1e-6, share_support_across_equations=False,
        max_points=25000,
    )
    loader = DataLoader(TensorDataset(X), batch_size=int(X.shape[0]), shuffle=False)
    res = discover_vector_system_de_from_surrogate(
        surrogate, loader, cfg=cfg, equations=problem.equations, vector_terms=vt, device=torch.device("cpu"),
    )
    support = _physical_support(res, nbk, orig_lib_names, drop_names)

    return {
        "gram": gram_disp,
        "mu": float(audit["max_offdiag_corr"]),
        "rank_status": audit["rank_status"],
        "alias_pairs": audit["alias_pairs"],
        "support": support,
        "orig_lib_names": orig_lib_names,
        "drop_names": drop_names,
    }


def _to_display_matrix(gram_engine: np.ndarray, term_names: list[str]) -> np.ndarray:
    idx = {nm: i for i, nm in enumerate(term_names)}
    out = np.full((O, O), np.nan)
    for a, na in enumerate(DISPLAY_TERMS):
        for b, nb in enumerate(DISPLAY_TERMS):
            if na in idx and nb in idx:
                out[a, b] = gram_engine[idx[na], idx[nb]]
    return out


def _physical_support(res, name_by_key: dict, orig_lib_names: set, drop_names: set) -> np.ndarray:
    """Physical RHS coefficients (= -residual coeffs): column 0 = dE/dt, 1 = dB/dt.

    NaN where the term is absent from the case library or dropped as an alias.
    """
    coeff: dict[str, tuple[float, float]] = {}
    for j, t in enumerate(res.term_vecs):
        if t is None:
            continue
        nm = name_by_key.get(_vec_key(t))
        if nm is None:
            continue
        coeff[nm] = (-float(res.coeffs[0, j]), -float(res.coeffs[1, j]))

    M = np.full((O, 2), np.nan)
    for r, nm in enumerate(DISPLAY_TERMS):
        if nm not in orig_lib_names or nm in drop_names:
            continue  # masked
        c = coeff.get(nm, (0.0, 0.0))
        M[r, 0], M[r, 1] = c[0], c[1]
    return M


# ---------------------------------------------------------------------------
# Field snapshots, sampled finely from the real generators for display
# ---------------------------------------------------------------------------


def field_snapshot(cid: str) -> dict:
    problem = PROBLEM_REGISTRY[cid]
    if cid == "mw000":
        X, Y, _G, _m = build_problem_data(problem, nz=400, nt=1, nx=1, ny=1)
        z = X[:, 3].cpu().numpy()
        return {"type": "line", "z": z, "Ex": Y[:, 0].cpu().numpy(), "By": Y[:, 4].cpu().numpy()}
    if cid == "mw003":
        X, Y, _G, _m = build_problem_data(problem, nt=1, ny=1, nx=220, nz=220)
        return _slice_2d(X, Y, problem, ax_a=1, ax_b=3, n_a=220, n_b=220, comp_pref=1)
    if cid == "mw002":
        X, Y, _G, _m = build_problem_data(problem, nt=1, ny=1, nx=220, nz=220)
        snap = _slice_2d(X, Y, problem, ax_a=1, ax_b=3, n_a=220, n_b=220)
        snap["damping_sigma"] = float(problem.param_defaults.get("sigma", 0.6))
        return snap
    if cid == "mw001":
        X, Y, _G, _m = build_problem_data(problem, nt=1, nz=1, nx=220, ny=220)
        return _wire_slice(X, Y, n_a=220, n_b=220)
    raise ValueError(cid)


def _grid(Y_col: torch.Tensor, nt: int, nx: int, ny: int, nz: int) -> np.ndarray:
    return Y_col.reshape(nt, nx, ny, nz).cpu().numpy()


def _slice_2d(X, Y, problem, *, ax_a, ax_b, n_a, n_b, comp_pref=None):
    # ny=1 slice: grid (1, n_a, 1, n_b); pick the field component with most structure.
    nt, nx, ny, nz = 1, n_a, 1, n_b
    best_c, best_var = 0, -1.0
    n_field = min(6, Y.shape[1])
    for c in range(n_field):
        v = float(Y[:, c].var())
        if v > best_var:
            best_var, best_c = v, c
    comp = comp_pref if (comp_pref is not None and float(Y[:, comp_pref].var()) > 1e-6) else best_c
    Zg = _grid(Y[:, comp], nt, nx, ny, nz)[0, :, 0, :]  # (a, b)
    a = X[:, ax_a].reshape(nt, nx, ny, nz)[0, :, 0, 0].cpu().numpy()
    b = X[:, ax_b].reshape(nt, nx, ny, nz)[0, 0, 0, :].cpu().numpy()
    names = problem.field_names
    return {"type": "image", "Z": Zg, "extent": (a.min(), a.max(), b.min(), b.max()),
            "comp_name": names[comp] if comp < len(names) else f"c{comp}"}


def _wire_slice(X, Y, *, n_a, n_b):
    nt, nx, ny, nz = 1, n_a, n_b, 1
    Jz = _grid(Y[:, 8], nt, nx, ny, nz)[0, :, :, 0]
    Bx = _grid(Y[:, 3], nt, nx, ny, nz)[0, :, :, 0]
    By = _grid(Y[:, 4], nt, nx, ny, nz)[0, :, :, 0]
    a = X[:, 1].reshape(nt, nx, ny, nz)[0, :, 0, 0].cpu().numpy()
    b = X[:, 2].reshape(nt, nx, ny, nz)[0, 0, :, 0].cpu().numpy()
    return {"type": "wire", "Jz": Jz, "Bx": Bx, "By": By,
            "a": a, "b": b, "extent": (a.min(), a.max(), b.min(), b.max())}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


FONT_SCALE = 1.9  # the 15.5 in canvas prints at ~7 in (x0.45); 7 pt x 1.9 x 0.45 ~ 6 pt on the page


def render(results: list[dict], snaps: list[dict], out_base: Path) -> None:
    fig = plt.figure(figsize=(15.5, 11.8))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1.05, 1.18, 1.18], hspace=0.78, wspace=0.38, figure=fig)
    cmap_field = plt.get_cmap("viridis")
    cmap_corr = plt.get_cmap("coolwarm").copy(); cmap_corr.set_bad("0.88")
    cmap_coef = plt.get_cmap("coolwarm").copy(); cmap_coef.set_bad("0.82")

    # ---- Row 0: field snapshots ----
    for col, (snap, title) in enumerate(zip(snaps, CASE_TITLES)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, fontsize=9 * FONT_SCALE, pad=8, fontweight="bold")
        if snap["type"] == "line":
            ax.plot(snap["z"], snap["Ex"], lw=2.0, label=r"$E_x$")
            ax.plot(snap["z"], snap["By"] + 2.4, lw=2.0, label=r"$B_y$ (offset)")
            ax.axhline(0, lw=0.5, color="0.6"); ax.axhline(2.4, lw=0.5, color="0.6")
            ax.set_xlim(snap["z"].min(), snap["z"].max()); ax.set_yticks([])
            # headroom above the offset B_y peak for the one-row legend, and below the
            # E_x trough for the alias annotation, so neither overlaps the curves
            lo, hi = ax.get_ylim(); span = hi - lo
            ax.set_ylim(lo - 0.42 * span, hi + 0.30 * span)
            ax.set_xlabel(r"$z$")
            ax.legend(frameon=False, fontsize=8 * FONT_SCALE, loc="upper center", ncol=2,
                      columnspacing=1.2, handlelength=1.5, borderaxespad=0.2)
            ax.text(0.04, 0.06, r"$\nabla^2E=-E,\ \nabla^2B=-B$", transform=ax.transAxes, fontsize=9 * FONT_SCALE,
                    bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9, boxstyle="round,pad=0.25"))
        elif snap["type"] == "image":
            ax.imshow(snap["Z"].T, extent=snap["extent"], origin="lower", cmap=cmap_field, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.04, 0.06, snap["comp_name"], transform=ax.transAxes, fontsize=9 * FONT_SCALE, color="white",
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.45, boxstyle="round,pad=0.25"))
            if "damping_sigma" in snap:
                iax = ax.inset_axes([0.60, 0.60, 0.36, 0.33])
                tt = np.linspace(0.0, 6.0, 120)
                iax.plot(tt, np.exp(-0.5 * snap["damping_sigma"] * tt), lw=1.6, color="white")
                iax.set_facecolor((0, 0, 0, 0.35))
                iax.set_xticks([]); iax.set_yticks([])
                for sp in iax.spines.values():
                    sp.set_color("white"); sp.set_linewidth(0.6)
                iax.set_title(r"$\|E\|\propto e^{-\sigma t/2}$", fontsize=5.5 * FONT_SCALE, pad=1.0, color="white")
        else:  # wire
            ax.imshow(snap["Jz"].T, extent=snap["extent"], origin="lower", cmap=cmap_field, aspect="auto")
            n = snap["Jz"].shape[0]; st = max(1, n // 16); sl = slice(st // 2, n, st)
            AA, BB = np.meshgrid(snap["a"][sl], snap["b"][sl], indexing="ij")
            ax.quiver(AA, BB, snap["Bx"][sl][:, sl], snap["By"][sl][:, sl],
                      color="white", scale=None, width=0.006, alpha=0.85)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.04, 0.06, r"$J_z$ + circulating $B$", transform=ax.transAxes, fontsize=9 * FONT_SCALE, color="white",
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.45, boxstyle="round,pad=0.25"))

    # ---- Row 1: Gram matrices ----
    corr_im = None
    for col, r in enumerate(results):
        ax = fig.add_subplot(gs[1, col])
        corr_im = ax.imshow(np.ma.masked_invalid(r["gram"]), vmin=-1, vmax=1, cmap=cmap_corr)
        ax.set_xticks(range(O)); ax.set_yticks(range(O))
        ax.set_xticklabels(DISPLAY_LABELS, rotation=55, ha="right", fontsize=7 * FONT_SCALE)
        ax.set_yticklabels(DISPLAY_LABELS if col == 0 else [""] * O, fontsize=8 * FONT_SCALE)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(r["rank_status"].replace("_", " ") + "\n" + rf"$\mu={r['mu']:.3f}$",
                     fontsize=8.5 * FONT_SCALE, pad=6)
        ax.set_xticks(np.arange(-.5, O, 1), minor=True); ax.set_yticks(np.arange(-.5, O, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        # box exact aliases (|corr|>0.999) and the dominant off-diagonal coherence cell
        G = r["gram"]
        off = np.array(G, dtype=float)
        np.fill_diagonal(off, 0.0)
        off = np.nan_to_num(off)
        if np.nanmax(np.abs(off)) > 0.999:
            for (i, j) in zip(*np.where(np.abs(off) > 0.999)):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="k", lw=1.7))
            ax.text(0.5, -0.46, "exact aliases detected", transform=ax.transAxes, fontsize=8 * FONT_SCALE, ha="center", va="top")
        elif np.nanmax(np.abs(off)) > 0.90:
            i, j = np.unravel_index(np.argmax(np.abs(off)), off.shape)
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="k", lw=1.7))
            ax.add_patch(Rectangle((i - .5, j - .5), 1, 1, fill=False, edgecolor="k", lw=1.7))
            ax.text(0.5, -0.46, "identifiable but coherent", transform=ax.transAxes, fontsize=8 * FONT_SCALE, ha="center", va="top")

    cax = fig.add_axes([0.93, 0.40, 0.011, 0.20])
    cb = fig.colorbar(corr_im, cax=cax); cb.set_label("feature correlation", fontsize=9 * FONT_SCALE); cb.ax.tick_params(labelsize=8 * FONT_SCALE)

    # ---- Row 2: recovered support ----
    coef_im = None
    for col, r in enumerate(results):
        ax = fig.add_subplot(gs[2, col])
        M = r["support"]
        coef_im = ax.imshow(np.ma.masked_invalid(M), vmin=-1, vmax=1, cmap=cmap_coef, aspect="auto")
        ax.set_xticks([0, 1]); ax.set_xticklabels([r"$\partial_t E$", r"$\partial_t B$"], fontsize=10 * FONT_SCALE)
        ax.set_yticks(range(O)); ax.set_yticklabels(DISPLAY_LABELS if col == 0 else [""] * O, fontsize=8 * FONT_SCALE)
        ax.tick_params(length=0)
        ax.set_title("recovered support", fontsize=10 * FONT_SCALE, pad=6)
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True); ax.set_yticks(np.arange(-.5, O, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(O):
            for j in range(2):
                if np.isnan(M[i, j]):
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="0.84", edgecolor="0.55",
                                           hatch="///", lw=0.0, alpha=0.9))
                elif abs(M[i, j]) > 0.05:
                    ax.text(j, i, f"{M[i, j]:+.2f}".rstrip("0").rstrip("."), ha="center", va="center",
                            fontsize=8.5 * FONT_SCALE, fontweight="bold",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, boxstyle="round,pad=0.12"))
                else:
                    # in-library candidate that was rejected (coefficient driven to ~0)
                    ax.text(j, i, "0", ha="center", va="center", fontsize=7 * FONT_SCALE, color="0.55")
        if r["drop_names"]:
            note = "grey = dropped alias / absent"
        elif "J" in r["orig_lib_names"]:
            note = r"$J$ selected; $\nabla^2$ rejected"
        else:
            note = r"$\nabla^2$ decoys rejected"
        ax.text(0.5, -0.16, note, transform=ax.transAxes, fontsize=8 * FONT_SCALE, ha="center", va="top")

    cax2 = fig.add_axes([0.93, 0.09, 0.011, 0.20])
    cb2 = fig.colorbar(coef_im, cax=cax2); cb2.set_label("coefficient sign", fontsize=9 * FONT_SCALE); cb2.ax.tick_params(labelsize=8 * FONT_SCALE)

    fig.text(0.012, 0.83, "field snapshot", rotation=90, va="center", ha="center", fontsize=12 * FONT_SCALE, fontweight="bold")
    fig.text(0.012, 0.50, "library audit", rotation=90, va="center", ha="center", fontsize=12 * FONT_SCALE, fontweight="bold")
    fig.text(0.012, 0.19, "sparse selection", rotation=90, va="center", ha="center", fontsize=12 * FONT_SCALE, fontweight="bold")
    fig.suptitle("Maxwell discovery with decoy operators: aliases, conditioning, and selected support",
                 fontsize=10 * FONT_SCALE, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.09, right=0.915, top=0.92, bottom=0.08)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Generate the Paper IV Maxwell decoy-operator figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=repo_root / "NestyNet_papers" / "figures_paper4" / "maxwell_decoy_operators",
        help="Output path without extension; both PDF and PNG are written",
    )
    args = parser.parse_args(argv)
    out_base = args.output_base.expanduser().resolve()
    results = [run_case(c) for c in CASES]
    snaps = [field_snapshot(c) for c in CASES]
    for c, r in zip(CASES, results):
        print(f"{c}: status={r['rank_status']}, mu={r['mu']:.3f}")
    render(results, snaps, out_base)


if __name__ == "__main__":
    main()
