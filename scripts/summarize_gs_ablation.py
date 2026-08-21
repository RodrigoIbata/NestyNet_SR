#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Summarize baseline-vs-GS ablation manifests and GS reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Ablation manifest files or directories containing them")
    args = ap.parse_args()
    manifests: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            manifests.extend(sorted(p.rglob("ablation_manifest.json")))
        elif p.name == "ablation_manifest.json":
            manifests.append(p)
    rows = []
    for m in manifests:
        data = _load(m)
        gs_reports = [Path(p) for p in data.get("gs_reports", [])]
        if not gs_reports:
            gs_reports = sorted(Path(data.get("gs_dir", m.parent)).rglob("*.gs_report.json"))
        gs_summary = {}
        if gs_reports:
            gs_summary = (_load(gs_reports[0]).get("summary", {}) or {})
        recs = {r.get("label"): r for r in data.get("records", []) if isinstance(r, dict)}
        rows.append({
            "suite": data.get("suite", ""),
            "manifest": str(m),
            "baseline_rc": recs.get("baseline", {}).get("returncode"),
            "gs_rc": recs.get("gs_auto", {}).get("returncode"),
            "satisfied": gs_summary.get("satisfied_generators"),
            "switched_off": gs_summary.get("switched_off_generators"),
            "proposals": gs_summary.get("quotient_coordinate_proposals"),
        })
    print("suite,baseline_rc,gs_rc,satisfied,switched_off,proposals,manifest")
    for r in rows:
        print(
            f"{r['suite']},{r['baseline_rc']},{r['gs_rc']},{r['satisfied']},{r['switched_off']},{r['proposals']},{r['manifest']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
