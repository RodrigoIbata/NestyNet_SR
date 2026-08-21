# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Collect a compact reproducibility capsule from an SRBench workspace.

A capsule is the audit evidence for one benchmark table cell (noise level x
protocol arm): every per-problem ``pb*.report.json``, the ``summary.csv``,
and a campaign manifest recording the per-run git revisions, seeds, and
walltimes found in the reports.  Logs, models, and checkpoints are excluded
by design; a full 120-problem cell collapses from gigabytes to ~10 MB.

Run it on any machine that holds the workspace (laptop or HPC):

    python3 collect_srbench_capsule.py /path/to/SRBench_0.000 \
        --cell noise0.000_ndata2k --output capsules/

Committee-of-experts campaigns write to ``results_CoE`` rather than
``results``; point ``--results-dir`` at it (those workspaces' own
``scripts/summarize.sh`` already defaults to that directory and honours a
``RESULTS_DIR`` override, so run it as usual before collecting):

    python3 collect_srbench_capsule.py /path/to/SRBench_0.010_CoE \
        --results-dir results_CoE \
        --cell noise0.010_coe50k --output capsules/ --tar

The capsule directory can then be shipped (tar/rsync) instead of the
workspace.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _git_state(repo: Path) -> dict:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        ).stdout.strip()
        return {"git_revision": rev, "working_tree_dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "working_tree_dirty": None}


def _arm_signature(reports: list[Path]) -> dict:
    """Summarize the campaign's actual configuration from its own reports.

    Arms differ in ways a directory name cannot be trusted to convey (data
    sizes, whether the committee or the statistical layer holds selection
    authority, the sealed audit partition), so every field is counted across
    reports and any variation stays visible.
    """
    fields: dict[str, Counter] = {
        "ndata_train": Counter(), "ndata_val": Counter(),
        "stat_enabled": Counter(), "stat_alpha": Counter(),
        "stat_authority": Counter(), "audit_rows": Counter(),
        "audit_unit_size": Counter(), "coe_enabled": Counter(),
        "coe_authoritative": Counter(), "coe_mode": Counter(),
        "source_git_dirty": Counter(),
    }
    for path in reports:
        try:
            d = json.loads(path.read_text())
        except ValueError:
            continue
        ss = d.get("statistical_selection") or {}
        au = ss.get("audit") or {}
        coe = d.get("coe_committee") or {}
        cfg = coe.get("config") or {}
        src = (((d.get("metadata") or {}).get("provenance") or {}).get("source") or {}).get("nestynet_sr") or {}
        for key, value in (
            ("ndata_train", cfg.get("ndata_train")),
            ("ndata_val", cfg.get("ndata_val")),
            ("stat_enabled", ss.get("enabled")),
            ("stat_alpha", ss.get("alpha")),
            ("stat_authority", ss.get("authority")),
            ("audit_rows", au.get("n_rows")),
            ("audit_unit_size", au.get("unit_size")),
            ("coe_enabled", coe.get("enabled") if coe else None),
            ("coe_authoritative", coe.get("authoritative") if coe else None),
            ("coe_mode", cfg.get("mode")),
            ("source_git_dirty", src.get("git_dirty")),
        ):
            fields[key][str(value)] += 1
    return {k: dict(v) for k, v in fields.items()}


def _build_manifest(cell: str, workspace: Path, results: Path,
                    reports: list[Path], summary: Path) -> dict:
    git_hashes: Counter = Counter()
    seeds: Counter = Counter()
    walltimes: list[float] = []
    incomplete: list[str] = []
    for path in reports:
        try:
            md = json.loads(path.read_text()).get("metadata") or {}
        except ValueError:
            incomplete.append(path.name)
            continue
        git_hashes[str(md.get("git_hash"))] += 1
        seeds[str(md.get("seed"))] += 1
        wt = md.get("walltime_hours")
        if isinstance(wt, (int, float)):
            walltimes.append(float(wt))
    return {
        "schema_version": 2,
        "cell": cell,
        "workspace": workspace.name,
        "results_dir": results.name,
        "n_reports": len(reports),
        "unreadable_reports": incomplete,
        "run_git_revisions": dict(git_hashes),
        "run_seeds": dict(seeds),
        "walltime_hours_total": round(sum(walltimes), 3),
        "arm_signature": _arm_signature(reports),
        "summary_csv_included": summary.is_file(),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "collector_source": _git_state(Path(__file__).resolve().parents[1]),
        "excluded_by_design": ["*.log", "models/", "checkpoints/", "*.state.pkl", "*.expressions.pkl"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--cell", required=True, help="Cell label, e.g. noise0.010_coe50k")
    parser.add_argument("--output", type=Path, default=Path("capsules"))
    parser.add_argument("--tar", action="store_true", help="Also write <cell>.tgz beside the capsule dir")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory to collect: a name relative to the workspace "
        "or an absolute path (committee campaigns use results_CoE)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Treat the positional argument as an existing capsule and rebuild "
        "its manifest from the reports it already contains",
    )
    args = parser.parse_args()

    if args.refresh:
        capsule = args.workspace.expanduser().resolve()
        reports = sorted((capsule / "reports").glob("pb*.report.json"))
        if not reports:
            raise SystemExit(f"no reports under {capsule}/reports")
        previous = {}
        mpath = capsule / "campaign_manifest.json"
        if mpath.is_file():
            previous = json.loads(mpath.read_text())
        manifest = _build_manifest(
            previous.get("cell", capsule.name),
            Path(previous.get("workspace", capsule.name)),
            Path(previous.get("results_dir", "results")),
            reports,
            capsule / "summary.csv",
        )
        for key in ("collected_at_utc", "collector_source"):
            if key in previous:
                manifest[f"original_{key}"] = previous[key]
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"refreshed {mpath}")
        print("  arm signature:", json.dumps(manifest["arm_signature"], indent=1)[:400])
        return 0

    workspace = args.workspace.expanduser().resolve()
    results_arg = Path(args.results_dir).expanduser()
    results = results_arg if results_arg.is_absolute() else (workspace / results_arg)
    results = results.resolve()
    if not results.is_dir():
        raise SystemExit(f"results directory not found: {results}")
    reports = sorted(results.glob("pb*.report.json"))
    if not reports:
        raise SystemExit(f"no pb*.report.json under {results}; refusing to collect an empty capsule")

    capsule = args.output.expanduser().resolve() / args.cell
    if capsule.exists():
        raise SystemExit(f"capsule already exists: {capsule}")
    (capsule / "reports").mkdir(parents=True)

    for path in reports:
        shutil.copy2(path, capsule / "reports" / path.name)
    summary = results / "summary.csv"
    if summary.is_file():
        shutil.copy2(summary, capsule / "summary.csv")
    manifest = _build_manifest(args.cell, workspace, results, reports, summary)
    git_hashes = manifest["run_git_revisions"]
    (capsule / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.tar:
        tar_path = capsule.parent / f"{args.cell}.tgz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(capsule, arcname=args.cell)
        print(f"tarball: {tar_path}")

    print(f"capsule: {capsule}")
    print(f"  reports: {len(reports)}, revisions: {dict(git_hashes)}, "
          f"total walltime: {manifest['walltime_hours_total']} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
