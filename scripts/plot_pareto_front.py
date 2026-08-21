#!/usr/bin/env python3
"""Plot the certified statistical Pareto fronts for one SRBench problem."""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path


def _problem_id(raw: str) -> str:
    match = re.fullmatch(r"(?:pb)?(\d{1,3})", raw.strip(), flags=re.IGNORECASE)
    if match is None or not 0 <= int(match.group(1)) <= 119:
        raise argparse.ArgumentTypeError("problem must be an ID from 000 to 119")
    return f"{int(match.group(1)):03d}"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read JSON {path}: {exc}") from exc


def _newest(paths: list[Path], description: str) -> Path:
    if not paths:
        raise SystemExit(f"no {description} found")
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _find_certificate(results_dir: Path, problem_id: str) -> Path:
    paths = list(
        results_dir.glob(
            f"pb{problem_id}_*_stat_selection_*/*.sr_pareto_certificate.json"
        )
    )
    return _newest(paths, f"Pareto certificate for pb{problem_id} in {results_dir}")


def _find_report(results_dir: Path, problem_id: str) -> Path | None:
    paths = list(results_dir.glob(f"pb{problem_id}_*.report.json"))
    return _newest(paths, f"report for pb{problem_id}") if paths else None


def _certificate_from_report(report: dict | None) -> Path | None:
    if not report:
        return None
    selection = report.get("final_selection") or {}
    raw_path = selection.get("certificate_path")
    if not raw_path:
        raw_path = (report.get("statistical_selection") or {}).get("certificate_path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser().resolve()
    return path if path.is_file() else None


def _candidate_rows(certificate: dict) -> list[dict]:
    pareto = certificate.get("pareto") or {}
    risks = pareto.get("risks") or {}
    archive = certificate.get("archive") or {}
    candidates = archive.get("candidates") or []
    rows = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id not in risks:
            continue
        metadata = candidate.get("metadata") or {}
        rows.append(
            {
                "id": candidate_id,
                "expression": str(metadata.get("expression") or ""),
                "risk": float(risks[candidate_id]),
                "complexity": {
                    str(key): float(value)
                    for key, value in (candidate.get("complexity") or {}).items()
                },
            }
        )
    if not rows:
        raise SystemExit("certificate contains no plottable candidate risks")
    return rows


def _selected_class(report: dict | None) -> str | None:
    if not report:
        return None
    selection = report.get("final_selection") or {}
    selected = selection.get("functional_class")
    return str(selected) if selected else None


def _selected_expression(report: dict | None) -> str | None:
    if not report:
        return None
    expression = (report.get("final_selection") or {}).get("expr")
    return str(expression) if expression else None


def _equation_contenders(
    rows: list[dict], pareto: dict, selected: str | None, count: int
) -> list[dict]:
    """Return winner first, then the lowest-risk certified front alternatives."""
    if count <= 0:
        return []
    by_id = {row["id"]: row for row in rows}
    front_ids = set(pareto.get("confidence_front") or [])
    front_ids.update(pareto.get("practical_front") or [])
    front_ids.update(pareto.get("point_front") or [])
    alternatives = [
        row for row in rows if row["id"] != selected and row["id"] in front_ids
    ]
    alternatives.sort(key=lambda row: (row["risk"], row["id"]))
    ordered = ([by_id[selected]] if selected in by_id else []) + alternatives
    if len(ordered) < count:
        used = {row["id"] for row in ordered}
        remainder = sorted(
            (row for row in rows if row["id"] not in used),
            key=lambda row: (row["risk"], row["id"]),
        )
        ordered.extend(remainder)
    return ordered[:count]


def _pretty_complexity(name: str) -> str:
    return name.replace("_", " ").title()


def _format_number(value: float) -> str:
    return f"{value:.6g}" if math.isfinite(value) else "n/a"


def _complexity_summary(row: dict, complexity_names: list[str]) -> str:
    short_names = {
        "ast_nodes": "nodes",
        "constant_code": "constant-code",
        "free_parameters": "free-params",
        "tree_depth": "depth",
    }
    complexity = row["complexity"]
    return ", ".join(
        f"{short_names.get(name, name)}={_format_number(complexity.get(name, math.nan))}"
        for name in complexity_names
    )


def _configure_risk_axis(axis, values: list[float], metric: str) -> None:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if positive and max(positive) / min(positive) >= 100.0:
        axis.set_yscale("symlog", linthresh=min(positive) / 10.0)
    elif values and all(value == 0.0 for value in values):
        # Matplotlib gives constant-zero data symmetric negative ticks by
        # default, which is misleading for a nonnegative error metric.
        axis.set_yticks([0.0])
    axis.set_ylabel(
        "Standardized audit RMSE"
        if metric == "rmse"
        else "Standardized audit risk (MSE)"
    )


def _front_summary(
    rows: list[dict],
    pareto: dict,
    selected: str | None,
    metric: str,
    complexity_names: list[str],
) -> str:
    by_id = {row["id"]: row for row in rows}
    ordered = []
    for group in ("confidence_front", "practical_front", "point_front"):
        for candidate_id in pareto.get(group) or []:
            if candidate_id not in ordered:
                ordered.append(candidate_id)
    lines = []
    for candidate_id in ordered:
        row = by_id.get(candidate_id)
        if row is None:
            continue
        value = max(0.0, row["risk"])
        if metric == "rmse":
            value = math.sqrt(value)
        marker = "*" if candidate_id == selected else " "
        complexity = _complexity_summary(row, complexity_names)
        lines.append(
            f"{marker} {candidate_id}: {metric}={value:.6g}  "
            f"complexity=[{complexity}]  {row['expression']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all certified candidates for one problem across every "
            "complexity component."
        )
    )
    parser.add_argument("problem", type=_problem_id, help="problem ID, e.g. 000 or pb010")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="results directory (default: BENCH_ROOT/results_CoE)",
    )
    parser.add_argument("--certificate", type=Path, help="explicit certificate JSON")
    parser.add_argument("--report", type=Path, help="explicit report JSON")
    parser.add_argument("--output", type=Path, help="output PNG/PDF path")
    parser.add_argument(
        "--metric",
        choices=("rmse", "risk"),
        default="rmse",
        help="vertical metric (default: rmse)",
    )
    parser.add_argument(
        "--labels",
        choices=("front", "all", "none"),
        default="front",
        help="which class IDs to annotate (default: front)",
    )
    parser.add_argument(
        "--include-ineligible",
        action="store_true",
        help="include audit-failed candidates (normally omitted to preserve scale)",
    )
    parser.add_argument(
        "--equations",
        type=int,
        default=3,
        metavar="N",
        help="show the winner plus N-1 best front contenders (default: 3; 0 hides)",
    )
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()
    if args.equations < 0:
        parser.error("--equations must be nonnegative")

    # Preserve the path used to invoke the script instead of resolving a
    # benchmark symlink to the canonical NestyNet_SR copy.  This keeps the
    # default results directory anchored to that benchmark checkout.
    bench_root = Path(__file__).absolute().parents[1]
    results_dir = (args.results_dir or bench_root / "results_CoE").resolve()
    report_path = (
        args.report.resolve()
        if args.report
        else _find_report(results_dir, args.problem)
    )
    report = _load_json(report_path) if report_path else None
    certificate_path = (
        args.certificate.resolve()
        if args.certificate
        else _certificate_from_report(report)
        or _find_certificate(results_dir, args.problem)
    )
    certificate = _load_json(certificate_path)
    pareto = certificate.get("pareto") or {}
    rows = _candidate_rows(certificate)
    selected = _selected_class(report)
    selected_expression = _selected_expression(report)

    complexity_names = [str(name) for name in pareto.get("complexity_names") or []]
    if not complexity_names:
        complexity_names = sorted(
            {name for row in rows for name in row["complexity"]}
        )
    if not complexity_names:
        raise SystemExit("certificate contains no complexity components")

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    point_front = set(pareto.get("point_front") or [])
    confidence_front = set(pareto.get("confidence_front") or [])
    practical_front = set(pareto.get("practical_front") or [])
    eligible = set(pareto.get("eligible_candidate_ids") or [])
    n_ineligible = sum(row["id"] not in eligible for row in rows)
    if not args.include_ineligible:
        rows = [row for row in rows if row["id"] in eligible]
        if not rows:
            raise SystemExit("certificate contains no eligible candidates to plot")
    front_union = point_front | confidence_front | practical_front
    y_values = [
        math.sqrt(max(0.0, row["risk"])) if args.metric == "rmse" else row["risk"]
        for row in rows
    ]
    ncols = min(2, len(complexity_names))
    nrows = math.ceil(len(complexity_names) / ncols)
    equation_rows = _equation_contenders(rows, pareto, selected, args.equations)
    equation_height = 1.2 + 0.52 * len(equation_rows) if equation_rows else 0.0
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 4.5 * nrows + equation_height),
        sharey=True,
        squeeze=False,
        constrained_layout=False,
    )
    axes_flat = list(axes.flat)
    for axis, complexity_name in zip(axes_flat, complexity_names):
        label_slots: dict[str, int] = {}
        if axis is axes_flat[0] and args.labels != "none":
            grouped_labels: dict[float, list[tuple[str, float]]] = {}
            for row, y_value in zip(rows, y_values):
                candidate_id = row["id"]
                annotate = args.labels == "all" or (
                    args.labels == "front"
                    and (candidate_id in front_union or candidate_id == selected)
                )
                x_value = row["complexity"].get(complexity_name, math.nan)
                if annotate and math.isfinite(x_value) and math.isfinite(y_value):
                    grouped_labels.setdefault(x_value, []).append((candidate_id, y_value))
            for group in grouped_labels.values():
                for slot, (candidate_id, _) in enumerate(
                    sorted(group, key=lambda item: (item[1], item[0]))
                ):
                    label_slots[candidate_id] = slot
        for row, y_value in zip(rows, y_values):
            candidate_id = row["id"]
            x_value = row["complexity"].get(complexity_name, math.nan)
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                continue
            axis.scatter(
                x_value,
                y_value,
                s=34,
                marker="o",
                facecolor="0.72" if candidate_id in eligible else "none",
                edgecolor="0.35" if candidate_id in eligible else "0.65",
                linewidth=0.8,
                zorder=2,
            )
            if candidate_id in point_front:
                axis.scatter(
                    x_value, y_value, s=70, marker="s", facecolor="none",
                    edgecolor="#7a7a7a", linewidth=1.4, zorder=3,
                )
            if candidate_id in confidence_front:
                axis.scatter(
                    x_value, y_value, s=105, marker="o", facecolor="none",
                    edgecolor="#276fbf", linewidth=2.0, zorder=4,
                )
            if candidate_id in practical_front:
                axis.scatter(
                    x_value, y_value, s=100, marker="D", facecolor="none",
                    edgecolor="#e07a1f", linewidth=1.8, zorder=5,
                )
            if candidate_id == selected:
                axis.scatter(
                    x_value, y_value, s=180, marker="*", facecolor="#c33c54",
                    edgecolor="white", linewidth=0.8, zorder=6,
                )
            annotate = axis is axes_flat[0] and (args.labels == "all" or (
                args.labels == "front"
                and (candidate_id in front_union or candidate_id == selected)
            ))
            if annotate:
                label_slot = label_slots.get(candidate_id, 0)
                x_values = [
                    item["complexity"].get(complexity_name, math.nan)
                    for item in rows
                ]
                finite_x = [value for value in x_values if math.isfinite(value)]
                right_side = finite_x and x_value > (min(finite_x) + max(finite_x)) / 2
                axis.annotate(
                    candidate_id,
                    (x_value, y_value),
                    xytext=((-5 if right_side else 5), 4 + 9 * label_slot),
                    textcoords="offset points",
                    fontsize=8,
                    ha="right" if right_side else "left",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
                    arrowprops={"arrowstyle": "-", "color": "0.55", "linewidth": 0.45},
                    clip_on=False,
                )
        axis.set_xlabel(_pretty_complexity(complexity_name))
        axis.grid(True, alpha=0.22, linewidth=0.7)
        _configure_risk_axis(axis, y_values, args.metric)
    for axis in axes_flat[len(complexity_names):]:
        axis.set_visible(False)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color="0.5", label="Eligible"),
        Line2D([], [], marker="s", linestyle="none", markerfacecolor="none", color="#7a7a7a", label="Point front"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", color="#276fbf", markeredgewidth=2, label="Confidence front"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="none", color="#e07a1f", label="Practical front"),
    ]
    if selected:
        handles.append(
            Line2D([], [], marker="*", linestyle="none", color="#c33c54", markersize=12, label="Selected")
        )
    plot_bottom = equation_height / (4.5 * nrows + equation_height) + 0.04
    fig.subplots_adjust(
        top=0.84,
        bottom=plot_bottom if equation_rows else 0.08,
        hspace=0.26,
        wspace=0.12,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncols=len(handles),
        frameon=False,
    )
    shown = len(rows)
    omitted = f"; {n_ineligible} ineligible omitted" if n_ineligible and not args.include_ineligible else ""
    class_word = "class" if shown == 1 else "classes"
    fig.suptitle(
        f"pb{args.problem} statistical Pareto certificate — {shown} {class_word} shown{omitted}",
        fontsize=13,
        y=0.985,
    )
    if equation_rows:
        equation_axis = fig.add_axes([0.06, 0.01, 0.88, plot_bottom - 0.09])
        equation_axis.axis("off")
        equation_axis.text(
            0.0,
            0.97,
            "Selected equation and best certified contenders",
            transform=equation_axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        line_y = 0.76
        line_step = 0.68 / max(1, len(equation_rows))
        for rank, row in enumerate(equation_rows):
            is_winner = row["id"] == selected
            role = "WINNER" if is_winner else f"CONTENDER {rank}"
            expression = (
                selected_expression
                if is_winner and selected_expression
                else row["expression"]
            )
            rmse = math.sqrt(max(0.0, row["risk"]))
            complexity = _complexity_summary(row, complexity_names)
            metadata_line = (
                f"{role:<12} {row['id']}   RMSE={rmse:.6g}   "
                f"COMPLEXITY [{complexity}]"
            )
            expression_prefix = " " * 15 + "EQUATION  "
            wrapped = textwrap.wrap(
                expression,
                width=max(30, 145 - len(expression_prefix)),
                subsequent_indent=expression_prefix,
                break_long_words=False,
                break_on_hyphens=False,
            ) or ["(expression unavailable)"]
            equation_axis.text(
                0.0,
                line_y,
                metadata_line + "\n" + expression_prefix + ("\n".join(wrapped)),
                transform=equation_axis.transAxes,
                fontsize=8.5,
                family="monospace",
                color="#a52843" if is_winner else "0.2",
                va="top",
            )
            line_y -= line_step

    output = (
        args.output.resolve()
        if args.output
        else results_dir / f"pb{args.problem}_pareto_front.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Certificate: {certificate_path}")
    if report_path:
        print(f"Report:      {report_path}")
    print(f"Plot:        {output}")
    summary = _front_summary(
        rows, pareto, selected, args.metric, complexity_names
    )
    if summary:
        print("\nFront candidates (* = authoritative selection):")
        print(summary)
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
