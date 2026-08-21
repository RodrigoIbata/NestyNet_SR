#!/usr/bin/env python3
"""Compact Feynman-DE baseline-vs-GS report helper.

This helper is intentionally tolerant of evolving run_de.py JSON schemas.  It
collects *_de.json files from two result directories, extracts the most useful
status/equation/term/source fields it can find, and writes upload-friendly JSON,
CSV and Markdown summaries.  It is a lightweight engineering reporter; the
benchmark driver remains examples/feynman_de/run_benchmark.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _case_id_from_path(path: Path) -> str:
    m = re.search(r"de(\d+)", path.name)
    if m:
        return m.group(1).zfill(3)
    m = re.search(r"(\d+)", path.stem)
    return m.group(1).zfill(3) if m else path.stem


def _iter_case_jsons(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*_de.json")):
        cid = _case_id_from_path(path)
        # Prefer shallow/latest-looking matches but remain deterministic.
        out.setdefault(cid, path)
    return out


def _dig(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _first_nonempty(*vals: Any) -> Any:
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None


def _extract(payload: dict[str, Any]) -> dict[str, Any]:
    de = payload.get("de_discovery", {}) if isinstance(payload.get("de_discovery"), dict) else {}
    selected = _first_nonempty(
        de.get("selected_candidate"),
        de.get("selected"),
        payload.get("selected_candidate"),
        payload.get("result"),
    )
    if not isinstance(selected, dict):
        selected = de if de else payload
    sim = _first_nonempty(payload.get("simulation_validation"), de.get("simulation_validation"), payload.get("validation"))
    if not isinstance(sim, dict):
        sim = {}
    status = _first_nonempty(payload.get("status"), sim.get("status"), de.get("status"), selected.get("status"), "UNKNOWN")
    mean_nrmse = _first_nonempty(sim.get("mean_nrmse"), payload.get("mean_nrmse"), selected.get("mean_nrmse"))
    max_nrmse = _first_nonempty(sim.get("max_nrmse"), payload.get("max_nrmse"), selected.get("max_nrmse"))
    terms = _first_nonempty(selected.get("terms"), selected.get("term_strings"), selected.get("feature_names"), [])
    sources = _first_nonempty(selected.get("term_sources"), [])
    equation = _first_nonempty(selected.get("canonical_equation"), de.get("canonical_equation"), payload.get("canonical_equation"))
    return {
        "status": str(status),
        "mean_nrmse": mean_nrmse,
        "max_nrmse": max_nrmse,
        "canonical_equation": equation,
        "terms": terms if isinstance(terms, list) else [terms],
        "term_sources": sources if isinstance(sources, list) else [sources],
    }


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--gs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    baseline_dir = Path(args.baseline_dir)
    gs_dir = Path(args.gs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b_files = _iter_case_jsons(baseline_dir)
    g_files = _iter_case_jsons(gs_dir)
    ids = sorted(set(b_files) | set(g_files))
    rows: list[dict[str, Any]] = []
    for cid in ids:
        b = _extract(_load_json(b_files[cid])) if cid in b_files else {"status": "MISSING"}
        g = _extract(_load_json(g_files[cid])) if cid in g_files else {"status": "MISSING"}
        bmax = _as_float(b.get("max_nrmse"))
        gmax = _as_float(g.get("max_nrmse"))
        if b.get("status") != g.get("status"):
            transition = "status_changed"
        elif bmax is not None and gmax is not None and abs(gmax - bmax) > 1e-12:
            transition = "lower_nrmse" if gmax < bmax else "higher_nrmse"
        else:
            transition = "unchanged"
        rows.append({
            "id": cid,
            "baseline_status": b.get("status"),
            "gs_status": g.get("status"),
            "transition": transition,
            "baseline_max_nrmse": b.get("max_nrmse"),
            "gs_max_nrmse": g.get("max_nrmse"),
            "delta_max_nrmse_gs_minus_baseline": (gmax - bmax) if bmax is not None and gmax is not None else None,
            "baseline_canonical_equation": b.get("canonical_equation"),
            "gs_canonical_equation": g.get("canonical_equation"),
            "baseline_terms": b.get("terms"),
            "gs_terms": g.get("terms"),
            "gs_term_sources": g.get("term_sources"),
        })

    payload = {
        "baseline_dir": str(baseline_dir),
        "gs_dir": str(gs_dir),
        "n_rows": len(rows),
        "baseline_counts": dict(Counter(r["baseline_status"] for r in rows)),
        "gs_counts": dict(Counter(r["gs_status"] for r in rows)),
        "transition_counts": dict(Counter(r["transition"] for r in rows)),
        "rows": rows,
    }
    (out_dir / "baseline_vs_gs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "baseline_vs_gs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    md = ["# Feynman-DE GS comparison", "", f"Baseline: `{baseline_dir}`", f"GS: `{gs_dir}`", "", "## Counts", ""]
    md.append("| Variant | " + " | ".join(sorted(set(payload["baseline_counts"]) | set(payload["gs_counts"]))) + " |")
    labels = sorted(set(payload["baseline_counts"]) | set(payload["gs_counts"]))
    md.append("|---" + "|---:" * len(labels) + "|")
    md.append("| Baseline | " + " | ".join(str(payload["baseline_counts"].get(k, 0)) for k in labels) + " |")
    md.append("| GS | " + " | ".join(str(payload["gs_counts"].get(k, 0)) for k in labels) + " |")
    md.extend(["", "## Transitions", "", "```json", json.dumps(payload["transition_counts"], indent=2), "```", ""])
    (out_dir / "Feynman_DE_GS_Ablation_Report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
