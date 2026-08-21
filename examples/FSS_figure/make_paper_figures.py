#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile
import textwrap
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "xdg-cache"),
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sr_report_path() -> Path:
    return (
        _repo_root()
        / "results"
        / "phase10_oracle_regression_quick12_method_attribution_discovery"
        / "individual_reports"
        / "trig_affine_demo.factorized_search_only.refine_off.n1000.r0.json"
    )


def _default_de_report_path() -> Path:
    return _repo_root() / "results" / "de902_factorized_xlane_diag_v6" / "de902_ic_multi4_de.json"


def _default_progress_log_path() -> Path:
    return (
        _repo_root()
        / "results"
        / "feynman_de_compositional_900_903_factorized_de_v2_direct_exp"
        / "de902"
        / "de902_factorized_de_first.log"
    )


def _default_output_dir() -> Path:
    return _repo_root() / "results" / "FSS_figure"


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing JSON report: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _shorten(s: object, width: int = 58) -> str:
    return textwrap.shorten(str(s), width=width, placeholder=" ...")


def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _neg_log10(value: object) -> float:
    x = _safe_float(value, 1.0)
    return -math.log10(max(x, 1.0e-300))


def _selected_de_rows(de_report: dict[str, Any]) -> tuple[list[dict[str, Any]], int | None]:
    de = dict(de_report.get("de_discovery") or {})
    rescue = dict(de.get("factorized_rescue") or {})
    diagnostics = dict(rescue.get("diagnostics") or {})
    rows = list(diagnostics.get("shortlist_rows") or de.get("shortlist") or [])
    rank = diagnostics.get("selected_shortlist_rank")
    try:
        rank_i = int(rank)
    except (TypeError, ValueError):
        rank_i = None
    return rows, rank_i


def _resolve_dataset_path(repo_root: Path, path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return repo_root / p


def _load_x0_extent(de_report: dict[str, Any], repo_root: Path) -> tuple[float, float]:
    datasets = list(dict(de_report.get("metadata") or {}).get("datasets") or [])
    values: list[float] = []
    for dataset in datasets:
        p = _resolve_dataset_path(repo_root, str(dataset))
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if "x0" not in (reader.fieldnames or []):
                continue
            for row in reader:
                values.append(float(row["x0"]))
    if not values:
        return 0.0, 1.0
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return 0.0, 1.0
    pad = 0.02 * (hi - lo)
    return lo - pad, hi + pad


def _eval_numpy_expr(expr: str, *, x0: np.ndarray) -> np.ndarray:
    tree = ast.parse(expr, mode="eval")
    env = {
        "x0": x0,
        "pi": np.pi,
        "e": np.e,
    }
    funcs = {
        "abs": np.abs,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "sin": np.sin,
        "sqrt": np.sqrt,
        "sqr": np.square,
        "tan": np.tan,
    }

    def visit(node: ast.AST) -> np.ndarray | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"unsupported symbol in coefficient expression: {node.id}")
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return np.power(left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in funcs:
                raise ValueError(f"unsupported function in coefficient expression: {node.func.id}")
            args = [visit(arg) for arg in node.args]
            return funcs[node.func.id](*args)
        raise ValueError(f"unsupported coefficient expression node: {type(node).__name__}")

    y = visit(tree)
    if np.isscalar(y):
        return np.full_like(x0, float(y), dtype=np.float64)
    return np.asarray(y, dtype=np.float64)


_MUTATE_RE = re.compile(
    r"\[mutate\]\s+(?P<iter>\d+)/(?P<total>\d+)\s+evals,\s+"
    r"residual_basins=(?P<basins>\d+),\s+best_mse=(?P<mse>[0-9.eE+-]+)"
)
_MUTATE_DONE_RE = re.compile(
    r"\[mutate\]\s+done:\s+(?P<iter>\d+)\s+evals,\s+"
    r"residual_basins=(?P<basins>\d+),\s+best_mse=(?P<mse>[0-9.eE+-]+)"
)
_BRUTE_DONE_RE = re.compile(
    r"\[brute\]\s+done:.*residual_basins=(?P<basins>\d+),\s+best_mse=(?P<mse>[0-9.eE+-]+)"
)


def _parse_progress_log(path: str | Path | None) -> list[dict[str, float]]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []

    rows: list[dict[str, float]] = []
    offset = 0.0
    last_iter: float | None = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _MUTATE_RE.search(line) or _MUTATE_DONE_RE.search(line)
        if match:
            local_iter = float(match.group("iter"))
            if last_iter is not None and local_iter < last_iter:
                offset += last_iter
            rows.append(
                {
                    "eval_global": offset + local_iter,
                    "eval_local": local_iter,
                    "residual_basins": float(match.group("basins")),
                    "best_mse": float(match.group("mse")),
                }
            )
            last_iter = local_iter
            continue
        match = _BRUTE_DONE_RE.search(line)
        if match:
            rows.append(
                {
                    "eval_global": offset,
                    "eval_local": 0.0,
                    "residual_basins": float(match.group("basins")),
                    "best_mse": float(match.group("mse")),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, output_stem: Path, *, show: bool) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_sr_ranking(ax: plt.Axes, sr_report: dict[str, Any]) -> None:
    results = sorted(list(sr_report.get("results") or []), key=lambda row: _safe_float(row.get("mse"), math.inf))[:5]
    labels = [_shorten(row.get("expr", ""), 50) for row in results]
    scores = [_neg_log10(row.get("mse")) for row in results]
    kinds = [str((row.get("mapping") or {}).get("kind") or row.get("mapping_kind") or "?") for row in results]
    palette = {
        "poly": "#3b6fb6",
        "sine": "#1b9e77",
        "pade": "#9a6fb0",
        "power": "#c75d2c",
        "exp": "#d49f28",
    }
    colors = [palette.get(kind, "#777777") for kind in kinds]
    y = np.arange(len(results), dtype=np.float64)

    ax.barh(y, scores, color=colors, edgecolor="#222222", linewidth=0.55)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(\mathrm{probe\ MSE})$")
    ax.set_title("A. SR archive ranks real candidates")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    for i, row in enumerate(results):
        mse = _safe_float(row.get("mse"))
        ax.text(scores[i] + 0.35, y[i], f"{kinds[i]}, MSE={mse:.1e}", va="center", fontsize=8.2)

    best = dict(sr_report.get("best") or {})
    stop = ", ".join(map(str, sr_report.get("search_stop_reasons") or [])) or "not recorded"
    ax.text(
        0.02,
        0.04,
        "best: "
        + _shorten(best.get("expr", ""), 60)
        + f"\niterations: {dict(sr_report.get('hp') or {}).get('n_iter', '?')}; stop: {stop}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cccccc", "alpha": 0.92},
    )

    action = dict(sr_report.get("action_distribution") or {})
    fractions = dict(action.get("fractions") or {})
    if fractions:
        inset = ax.inset_axes([0.58, 0.18, 0.38, 0.34])
        top = sorted(fractions.items(), key=lambda kv: float(kv[1]), reverse=True)[:5]
        inset.bar(
            np.arange(len(top)),
            [100.0 * float(v) for _, v in top],
            color="#d49f28",
            edgecolor="#333333",
            linewidth=0.45,
        )
        inset.set_xticks(np.arange(len(top)), [str(k) for k, _ in top], rotation=35, ha="right", fontsize=6.7)
        inset.set_ylabel("% moves", fontsize=7.2)
        inset.set_title("actual move mix", fontsize=7.4)
        inset.tick_params(axis="y", labelsize=6.7)
        inset.grid(axis="y", alpha=0.18, lw=0.5)


def _plot_progress(ax: plt.Axes, progress_rows: list[dict[str, float]]) -> None:
    ax.set_title("B. Search steering trace from the DE log")
    if not progress_rows:
        ax.text(0.5, 0.5, "No progress log found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.asarray([row["eval_global"] for row in progress_rows], dtype=np.float64)
    mse = np.asarray([row["best_mse"] for row in progress_rows], dtype=np.float64)
    basins = np.asarray([row["residual_basins"] for row in progress_rows], dtype=np.float64)

    ax.plot(x, mse, color="#355c7d", lw=2.0, marker="o", ms=3.2, label="best MSE")
    ax.set_yscale("log")
    ax.set_xlabel("cumulative evaluated skeletons")
    ax.set_ylabel("best MSE", color="#355c7d")
    ax.tick_params(axis="y", labelcolor="#355c7d")
    ax.grid(alpha=0.25, lw=0.6)

    ax2 = ax.twinx()
    ax2.plot(x, basins, color="#c65d2e", lw=1.8, marker="s", ms=3.0, alpha=0.82, label="residual basins")
    ax2.set_ylabel("residual basins", color="#c65d2e")
    ax2.tick_params(axis="y", labelcolor="#c65d2e")

    handles = [
        Line2D([0], [0], color="#355c7d", lw=2.0, marker="o", ms=4, label="best MSE"),
        Line2D([0], [0], color="#c65d2e", lw=1.8, marker="s", ms=4, label="residual basins"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)


def _plot_de_shortlist(ax: plt.Axes, de_rows: list[dict[str, Any]], selected_rank: int | None) -> None:
    ax.set_title("C. DE shortlist: evidence steers selection")
    if not de_rows:
        ax.text(0.5, 0.5, "No DE shortlist rows found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    lane_palette = {
        "x_coeff_on_u": "#3b6fb6",
        "state_lane": "#1b9e77",
        "rhs": "#c75d2c",
    }
    marker_by_family = {
        "log": "o",
        "explorer": "s",
        "poly2": "^",
        "poly": "D",
        "exp": "P",
    }
    for i, row in enumerate(de_rows):
        probe_rms = max(_safe_float(row.get("probe_rms"), math.nan), 1.0e-10)
        shape_score = max(_safe_float(row.get("shape_score"), 1.0), 1.0e-12)
        lane = str(row.get("lane", "?"))
        family = str(row.get("family", "?"))
        color = lane_palette.get(lane, "#777777")
        marker = marker_by_family.get(family, "o")
        is_selected = selected_rank is not None and i == selected_rank
        ax.scatter(
            probe_rms,
            shape_score,
            s=150 if is_selected else 76,
            marker=marker,
            facecolor=color if is_selected else "white",
            edgecolor="#222222" if is_selected else color,
            linewidth=1.4 if is_selected else 1.0,
            zorder=4 if is_selected else 3,
        )
        label = "selected" if is_selected else f"{family}"
        ax.annotate(
            label,
            (probe_rms, shape_score),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color="#222222" if is_selected else "#555555",
        )

    probe_vals = np.asarray([_safe_float(row.get("probe_rms"), math.nan) for row in de_rows], dtype=np.float64)
    if np.any(np.isfinite(probe_vals)):
        best_probe_idx = int(np.nanargmin(probe_vals))
        if selected_rank is None or best_probe_idx != selected_rank:
            row = de_rows[best_probe_idx]
            ax.scatter(
                max(_safe_float(row.get("probe_rms"), 1.0), 1.0e-10),
                max(_safe_float(row.get("shape_score"), 1.0), 1.0e-12),
                s=210,
                marker="o",
                facecolor="none",
                edgecolor="#d49f28",
                linewidth=1.8,
                zorder=5,
            )
            ax.annotate(
                "lowest RMS",
                (
                    max(_safe_float(row.get("probe_rms"), 1.0), 1.0e-10),
                    max(_safe_float(row.get("shape_score"), 1.0), 1.0e-12),
                ),
                xytext=(8, -14),
                textcoords="offset points",
                fontsize=8,
                color="#805d00",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.margins(x=0.16, y=0.16)
    ax.set_xlabel("probe RMS")
    ax.set_ylabel("shape score (lower is better)")
    ax.grid(alpha=0.25, lw=0.6)


def _plot_de_coefficients(
    ax: plt.Axes,
    de_rows: list[dict[str, Any]],
    selected_rank: int | None,
    x0_extent: tuple[float, float],
) -> None:
    ax.set_title("D. Operator-slot coefficients on data support")
    if not de_rows:
        ax.text(0.5, 0.5, "No DE coefficient rows found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x0 = np.linspace(float(x0_extent[0]), float(x0_extent[1]), 420, dtype=np.float64)
    curves: list[dict[str, Any]] = []
    reference_y: np.ndarray | None = None
    if x0_extent[0] > -0.999:
        reference_y = np.log1p(x0)

    for i, row in enumerate(de_rows):
        expr = str(row.get("coeff_ast") or row.get("coeff_expr") or "")
        if not expr:
            continue
        try:
            y = _eval_numpy_expr(expr, x0=x0)
        except ValueError:
            continue
        if not np.all(np.isfinite(y)):
            continue
        is_selected = selected_rank is not None and i == selected_rank
        curves.append({"expr": expr, "y": y, "is_selected": is_selected})

    base_arrays = [curve["y"] for curve in curves if curve["is_selected"]]
    if reference_y is not None:
        base_arrays.append(reference_y)
    if not base_arrays and curves:
        base_arrays.append(curves[0]["y"])
    if not base_arrays:
        ax.text(0.5, 0.5, "No x0-only coefficient expressions", ha="center", va="center", transform=ax.transAxes)
        return

    base_values = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in base_arrays])
    base_scale = max(1.0, float(np.nanpercentile(np.abs(base_values), 95.0)))
    offscale = 0
    plotted_arrays: list[np.ndarray] = []
    for curve in curves:
        y = np.asarray(curve["y"], dtype=np.float64)
        is_selected = bool(curve["is_selected"])
        if not is_selected and float(np.nanmax(np.abs(y))) > 25.0 * base_scale:
            offscale += 1
            continue
        color = "#c23b22" if is_selected else "#7f7f7f"
        lw = 2.8 if is_selected else 1.3
        alpha = 0.95 if is_selected else 0.45
        label = "selected: " + _shorten(curve["expr"], 44) if is_selected else _shorten(curve["expr"], 40)
        ax.plot(x0, y, color=color, lw=lw, alpha=alpha, label=label, zorder=3 if is_selected else 2)
        plotted_arrays.append(y)

    if reference_y is not None:
        ax.plot(
            x0,
            reference_y,
            color="#111111",
            lw=2.0,
            ls="--",
            label=r"benchmark target $\log(1+x_0)$",
            zorder=1,
        )
        plotted_arrays.append(reference_y)
    if not plotted_arrays:
        ax.text(0.5, 0.5, "No x0-only coefficient expressions", ha="center", va="center", transform=ax.transAxes)
        return
    y_values = np.concatenate([arr.ravel() for arr in plotted_arrays])
    y_lo = float(np.nanpercentile(y_values, 1.0))
    y_hi = float(np.nanpercentile(y_values, 99.0))
    if math.isfinite(y_lo) and math.isfinite(y_hi) and y_hi > y_lo:
        margin = 0.10 * (y_hi - y_lo)
        ax.set_ylim(y_lo - margin, y_hi + margin)
    if offscale:
        ax.text(
            0.98,
            0.04,
            f"{offscale} off-scale rejected candidate omitted",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#dddddd", "alpha": 0.92},
        )
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel("coefficient multiplying the operator slot")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="best", frameon=True, fontsize=7.4)


def _make_figure(
    sr_report: dict[str, Any],
    de_report: dict[str, Any],
    progress_rows: list[dict[str, float]],
    output_dir: Path,
    *,
    show: bool,
) -> None:
    de_rows, selected_rank = _selected_de_rows(de_report)
    x0_extent = _load_x0_extent(de_report, _repo_root())

    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 10.6,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.2,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.2), constrained_layout=True)
    _plot_sr_ranking(axes[0, 0], sr_report)
    _plot_progress(axes[0, 1], progress_rows)
    _plot_de_shortlist(axes[1, 0], de_rows, selected_rank)
    _plot_de_coefficients(axes[1, 1], de_rows, selected_rank, x0_extent)
    fig.suptitle(
        "Factorized symbolic search steering, from archived SR and DE runs",
        fontsize=13.0,
        y=1.02,
    )
    _save_figure(fig, output_dir / "factorized_search_steering", show=show)


def _write_sidecars(
    sr_report: dict[str, Any],
    de_report: dict[str, Any],
    progress_rows: list[dict[str, float]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    de_rows, selected_rank = _selected_de_rows(de_report)
    shortlist_out = []
    for i, row in enumerate(de_rows):
        shortlist_out.append(
            {
                "rank": i,
                "selected": bool(selected_rank is not None and i == selected_rank),
                "lane": row.get("lane"),
                "family": row.get("family"),
                "base_mode": row.get("base_mode"),
                "coord_ast": row.get("coord_ast"),
                "carrier_ast": row.get("carrier_ast"),
                "coeff_ast": row.get("coeff_ast"),
                "probe_rms": row.get("probe_rms"),
                "shape_score": row.get("shape_score"),
                "consistency_score": row.get("consistency_score"),
                "evidence_tier": row.get("evidence_tier"),
            }
        )
    _write_csv(
        output_dir / "de_shortlist_rows.csv",
        shortlist_out,
        [
            "rank",
            "selected",
            "lane",
            "family",
            "base_mode",
            "coord_ast",
            "carrier_ast",
            "coeff_ast",
            "probe_rms",
            "shape_score",
            "consistency_score",
            "evidence_tier",
        ],
    )
    _write_csv(
        output_dir / "progress_rows.csv",
        progress_rows,
        ["eval_global", "eval_local", "residual_basins", "best_mse"],
    )

    de = dict(de_report.get("de_discovery") or {})
    summary = {
        "inputs": {
            "sr_report": str(Path(args.sr_report).resolve()),
            "de_report": str(Path(args.de_report).resolve()),
            "progress_log": str(Path(args.progress_log).resolve()) if args.progress_log else None,
        },
        "outputs": {
            "figure_png": str((output_dir / "factorized_search_steering.png").resolve()),
            "figure_pdf": str((output_dir / "factorized_search_steering.pdf").resolve()),
            "progress_rows_csv": str((output_dir / "progress_rows.csv").resolve()),
            "de_shortlist_rows_csv": str((output_dir / "de_shortlist_rows.csv").resolve()),
        },
        "sr": {
            "spec_id": sr_report.get("spec_id"),
            "target_expr": sr_report.get("target_expr"),
            "best_expr": dict(sr_report.get("best") or {}).get("expr"),
            "best_mse": dict(sr_report.get("best") or {}).get("mse"),
            "action_distribution": sr_report.get("action_distribution"),
            "stop_reasons": sr_report.get("search_stop_reasons"),
        },
        "de": {
            "selected_engine": de.get("selected_engine"),
            "canonical_equation": de.get("canonical_equation"),
            "probe_rms": de.get("probe_rms"),
            "selected_shortlist_rank": selected_rank,
            "shortlist_rows": len(de_rows),
        },
        "progress": {
            "rows": len(progress_rows),
            "final_eval_global": progress_rows[-1]["eval_global"] if progress_rows else None,
            "final_best_mse": progress_rows[-1]["best_mse"] if progress_rows else None,
            "final_residual_basins": progress_rows[-1]["residual_basins"] if progress_rows else None,
        },
    }
    (output_dir / "factorized_search_steering_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a real-data FSS steering figure from archived SR and DE reports.",
    )
    parser.add_argument("--sr-report", type=Path, default=_default_sr_report_path())
    parser.add_argument("--de-report", type=Path, default=_default_de_report_path())
    parser.add_argument("--progress-log", type=Path, default=_default_progress_log_path())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr_report = _load_json(args.sr_report)
    de_report = _load_json(args.de_report)
    progress_rows = _parse_progress_log(args.progress_log)

    _make_figure(sr_report, de_report, progress_rows, output_dir, show=bool(args.show))
    _write_sidecars(sr_report, de_report, progress_rows, args, output_dir)

    print(f"Wrote {output_dir / 'factorized_search_steering.png'}")
    print(f"Wrote {output_dir / 'factorized_search_steering.pdf'}")
    print(f"Wrote {output_dir / 'factorized_search_steering_summary.json'}")
    print(f"Wrote {output_dir / 'progress_rows.csv'}")
    print(f"Wrote {output_dir / 'de_shortlist_rows.csv'}")


if __name__ == "__main__":
    main()
