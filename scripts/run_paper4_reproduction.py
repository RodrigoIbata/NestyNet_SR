#!/usr/bin/env python3
"""Run the frozen NestyNet Paper IV experiment protocol.

The command recipes live in ``reproducibility/paper4/protocol.json`` so the
published protocol is inspectable without executing this utility.  This
wrapper resolves paths, records provenance, and never invokes a shell.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "reproducibility" / "paper4" / "protocol.json"
DEFAULT_KEPLER_MANIFEST = (
    "kepler/raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.localized.json"
)
DEFAULT_KEPLER_SELECTION = (
    "kepler/selection_jpl_ssodnet_mass_gt_1e17_arc15000_summary.json"
)
DEFAULT_KEPLER_CACHE = "kepler/surrogate_accels_1d"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("steps"), list):
        raise RuntimeError(f"unsupported Paper IV protocol: {path}")
    ids = [str(row.get("id", "")) for row in payload["steps"]]
    if not all(ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"protocol step IDs must be non-empty and unique: {path}")
    return payload


def _git_state(repo: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout
        return {"git_revision": revision, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "working_tree_dirty": None}


def _nestynet_core_state() -> dict[str, Any]:
    """Git state of the NestyNet core the run imports (not just its version string).

    The core is resolved from the imported package so the recorded revision is
    the code actually used, wherever the checkout lives.
    """
    try:
        import nestynet

        core_repo = Path(nestynet.__file__).resolve().parent.parent
    except Exception:
        return {"path": None, "git_revision": None, "working_tree_dirty": None}
    return {"path": str(core_repo), **_git_state(core_repo)}


def _installed_versions() -> dict[str, str | None]:
    names = ("nestynet", "nestynet-sr", "torch", "numpy", "scipy", "pandas", "matplotlib", "sympy", "astropy")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _remove_option(argv: list[str], option: str) -> list[str]:
    """Remove a simple CLI option and its following value, when present."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == option:
            i += 1
            if i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(token)
        i += 1
    return out


def _apply_smoke_overrides(argv: list[str], overrides: list[str]) -> list[str]:
    out = list(argv)
    if "--only" in overrides:
        out = [token for token in out if token != "--all"]
    i = 0
    while i < len(overrides):
        option = overrides[i]
        if not option.startswith("--"):
            raise RuntimeError(f"invalid smoke override token: {option!r}")
        has_value = i + 1 < len(overrides) and not overrides[i + 1].startswith("--")
        out = _remove_option(out, option)
        out.append(option)
        if has_value:
            out.append(overrides[i + 1])
            i += 1
        i += 1
    return out


def _resolve_argv(
    row: dict[str, Any],
    *,
    python: str,
    data_root: Path,
    results_root: Path,
    kepler_manifest: Path,
    kepler_selection: Path,
    kepler_cache: Path,
    jobs: int,
    smoke: bool,
) -> list[str]:
    replacements = {
        "{python}": python,
        "{repo}": str(REPO_ROOT),
        "{data}": str(data_root),
        "{results}": str(results_root),
        "{jobs}": str(jobs),
        "{kepler_manifest}": str(kepler_manifest),
        "{kepler_selection}": str(kepler_selection),
        "{kepler_cache}": str(kepler_cache),
    }
    argv = []
    for raw in row["argv"]:
        value = str(raw)
        for marker, replacement in replacements.items():
            value = value.replace(marker, replacement)
        argv.append(value)
    if smoke and row.get("smoke_argv"):
        argv = _apply_smoke_overrides(argv, [str(v) for v in row["smoke_argv"]])
    return argv


def _select_steps(
    protocol: dict[str, Any], requested: list[str], groups: list[str], run_all: bool
) -> list[dict[str, Any]]:
    rows = list(protocol["steps"])
    by_id = {str(row["id"]): row for row in rows}
    known_groups = {str(row["group"]) for row in rows}
    missing_ids = sorted(set(requested) - set(by_id))
    missing_groups = sorted(set(groups) - known_groups)
    if missing_ids:
        raise RuntimeError(f"unknown step(s): {', '.join(missing_ids)}")
    if missing_groups:
        raise RuntimeError(f"unknown group(s): {', '.join(missing_groups)}")
    if run_all:
        return rows
    wanted = set(requested)
    wanted_groups = set(groups)
    return [
        row
        for row in rows
        if str(row["id"]) in wanted or str(row["group"]) in wanted_groups
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _print_catalog(protocol: dict[str, Any]) -> None:
    width = max(len(str(row["id"])) for row in protocol["steps"])
    for row in protocol["steps"]:
        print(f"{str(row['id']):<{width}}  {row['group']:<13}  {row['description']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or inspect the frozen NestyNet Paper IV protocol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("steps", nargs="*", help="Step IDs to run")
    parser.add_argument("--group", action="append", default=[], help="Run every step in this group")
    parser.add_argument("--all", action="store_true", help="Select every protocol step")
    parser.add_argument("--list", action="store_true", help="List protocol steps and exit")
    parser.add_argument("--run", action="store_true", help="Execute; without this flag commands are only printed")
    parser.add_argument("--smoke", action="store_true", help="Apply each step's reduced smoke override")
    parser.add_argument("--workspace", type=Path, default=Path("paper4_reproduction"), help="Run workspace")
    parser.add_argument("--data-root", type=Path, default=None, help="Exact archived input root")
    parser.add_argument("--results-root", type=Path, default=None, help="Output root")
    parser.add_argument("--kepler-manifest", type=Path, default=None, help="Exact 308-body daily HORIZONS manifest override")
    parser.add_argument("--kepler-selection", type=Path, default=None, help="Kepler/SsODNet selection-summary override")
    parser.add_argument("--kepler-cache", type=Path, default=None, help="Validated per-body surrogate-acceleration cache override")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--jobs", type=int, default=1, help="Explicit worker count for parallel-capable steps")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed step")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH, help="Protocol JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = args.protocol.expanduser().resolve()
    protocol = _load_protocol(protocol_path)
    if args.list:
        _print_catalog(protocol)
        return 0

    selected = _select_steps(protocol, args.steps, args.group, bool(args.all))
    if not selected:
        _print_catalog(protocol)
        print("\nSelect step IDs, --group GROUP, or --all.", file=sys.stderr)
        return 2
    if int(args.jobs) < 1:
        raise RuntimeError("--jobs must be at least 1")

    if args.smoke:
        unavailable = [row for row in selected if "smoke_argv" not in row]
        selected = [row for row in selected if "smoke_argv" in row]
        for row in unavailable:
            print(
                f"[skip smoke] {row['id']}: no bounded smoke recipe is declared",
                file=sys.stderr,
            )
        if not selected:
            print("No selected step has a bounded smoke recipe.", file=sys.stderr)
            return 2

    workspace = args.workspace.expanduser().resolve()
    data_root = (args.data_root or (workspace / "data")).expanduser().resolve()
    results_root = (args.results_root or (workspace / "results")).expanduser().resolve()
    kepler_manifest = (
        args.kepler_manifest or (data_root / DEFAULT_KEPLER_MANIFEST)
    ).expanduser().resolve()
    kepler_selection = (
        args.kepler_selection or (data_root / DEFAULT_KEPLER_SELECTION)
    ).expanduser().resolve()
    kepler_cache = (
        args.kepler_cache or (data_root / DEFAULT_KEPLER_CACHE)
    ).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    log_root = workspace / "logs"
    manifest_path = workspace / "paper4_run_manifest.json"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "paper": protocol["paper"],
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "protocol_path": str(protocol_path),
        "protocol": protocol,
        "source": {"path": str(REPO_ROOT), **_git_state(REPO_ROOT)},
        "nestynet_core": _nestynet_core_state(),
        "environment": {
            "python_executable": args.python,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": _installed_versions(),
        },
        "workspace": str(workspace),
        "data_root": str(data_root),
        "results_root": str(results_root),
        "kepler_manifest": str(kepler_manifest),
        "kepler_selection": str(kepler_selection),
        "kepler_cache": str(kepler_cache),
        "jobs": int(args.jobs),
        "smoke": bool(args.smoke),
        "executed": bool(args.run),
        "runs": [],
    }

    failed = False
    for row in selected:
        step_id = str(row["id"])
        command = _resolve_argv(
            row,
            python=str(args.python),
            data_root=data_root,
            results_root=results_root,
            kepler_manifest=kepler_manifest,
            kepler_selection=kepler_selection,
            kepler_cache=kepler_cache,
            jobs=int(args.jobs),
            smoke=bool(args.smoke),
        )
        rendered = shlex.join(command)
        print(f"\n[{step_id}] {row['description']}\n{rendered}")
        run_record: dict[str, Any] = {
            "id": step_id,
            "group": row["group"],
            "command": command,
            "started_at_utc": None,
            "completed_at_utc": None,
            "returncode": None,
            "log": None,
        }
        manifest["runs"].append(run_record)
        if not args.run:
            continue

        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{step_id}.log"
        run_record["started_at_utc"] = _utc_now()
        run_record["log"] = str(log_path)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {rendered}\n\n")
            log.flush()
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        run_record["returncode"] = int(process.returncode)
        run_record["completed_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        if process.returncode:
            failed = True
            print(f"[{step_id}] FAILED ({process.returncode}); see {log_path}", file=sys.stderr)
            if not args.keep_going:
                break
        else:
            print(f"[{step_id}] complete; log: {log_path}")

    manifest["completed_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)
    print(f"\nRun manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
