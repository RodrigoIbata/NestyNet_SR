# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Plot utilities for oracle factorized symbolic search/continuous skeleton refinement suite outputs."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
from collections import defaultdict
from typing import Any, Sequence


def _as_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    return f


def load_summary_csv(path: str | pathlib.Path) -> list[dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                {
                    "mode": str(row.get("mode", "")),
                    "budget": int(float(row.get("budget", "nan"))),
                    "n_runs": int(float(row.get("n_runs", "nan"))),
                    "solve_rate": _as_float(row.get("solve_rate", "nan")),
                    "best_mse_median": _as_float(row.get("best_mse_median", "nan")),
                    "best_mse_mean": _as_float(row.get("best_mse_mean", "nan")),
                    "wall_seconds_mean": _as_float(row.get("wall_seconds_mean", "nan")),
                }
            )

    if not out:
        raise ValueError(f"No rows found in summary CSV: {p}")
    return out


def plot_suite_summary(
    summary_rows: Sequence[dict[str, Any]],
    *,
    output_dir: str | pathlib.Path,
    title_prefix: str = "Oracle factorized symbolic search",
) -> dict[str, str]:
    """Render solve-rate / mse / time plots from suite summary rows."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("plot_suite_summary requires matplotlib") from exc

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in summary_rows:
        grouped[str(r["mode"])].append(dict(r))
    for mode in grouped:
        grouped[mode] = sorted(grouped[mode], key=lambda x: int(x["budget"]))

    # 1) Solve rate vs budget
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for mode, rows in grouped.items():
        x = [int(r["budget"]) for r in rows]
        y = [float(r["solve_rate"]) for r in rows]
        ax1.plot(x, y, marker="o", linewidth=2, label=mode)
    ax1.set_xscale("log")
    ax1.set_xlabel("Iteration Budget (n_iter)")
    ax1.set_ylabel("Solve Rate")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(f"{title_prefix}: Solve Rate vs Budget")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    p1 = out_dir / "solve_rate_vs_budget.png"
    fig1.tight_layout()
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)

    # 2) Median MSE vs budget
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for mode, rows in grouped.items():
        x = [int(r["budget"]) for r in rows]
        y = [max(float(r["best_mse_median"]), 1e-30) for r in rows]
        ax2.plot(x, y, marker="o", linewidth=2, label=mode)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Iteration Budget (n_iter)")
    ax2.set_ylabel("Median Best MSE")
    ax2.set_title(f"{title_prefix}: Median MSE vs Budget")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    p2 = out_dir / "median_mse_vs_budget.png"
    fig2.tight_layout()
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)

    # 3) Median MSE vs mean wall time (Pareto-like curve)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for mode, rows in grouped.items():
        x = [max(float(r["wall_seconds_mean"]), 1e-9) for r in rows]
        y = [max(float(r["best_mse_median"]), 1e-30) for r in rows]
        lbls = [int(r["budget"]) for r in rows]
        ax3.plot(x, y, marker="o", linewidth=2, label=mode)
        for xi, yi, bi in zip(x, y, lbls):
            if math.isfinite(xi) and math.isfinite(yi):
                ax3.annotate(str(bi), (xi, yi), fontsize=8, alpha=0.8)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Mean Wall Time (s)")
    ax3.set_ylabel("Median Best MSE")
    ax3.set_title(f"{title_prefix}: MSE vs Time")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    p3 = out_dir / "mse_vs_time.png"
    fig3.tight_layout()
    fig3.savefig(p3, dpi=160)
    plt.close(fig3)

    return {
        "solve_rate_vs_budget": str(p1),
        "median_mse_vs_budget": str(p2),
        "mse_vs_time": str(p3),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot oracle suite summary outputs")
    p.add_argument("--summary_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--title_prefix", type=str, default="Oracle factorized symbolic search")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = load_summary_csv(args.summary_csv)
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = str(pathlib.Path(args.summary_csv).resolve().parent)
    out = plot_suite_summary(summary, output_dir=out_dir, title_prefix=args.title_prefix)
    for k, v in out.items():
        print(f"[plot] {k}: {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
