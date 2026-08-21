#!/usr/bin/env python3
"""Stage the exact NestyNet Paper IV data and compact reference artifacts.

The default (strict) mode validates the headline benchmark totals before it
creates an archive.  ``--allow-incomplete`` exists only for testing the
packaging machinery before the final reruns have completed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "reproducibility" / "paper4" / "protocol.json"
KEPLER_MANIFEST_NAME = "raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.json"
KEPLER_LOCALIZED_NAME = "raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.localized.json"
KEPLER_RAW_DIR_NAME = "raw_jpl_ssodnet_mass_gt_1e17_arc15000_1d"
KEPLER_CACHE_DIR_NAME = "surrogate_accels_1d"
KEPLER_METADATA_NAMES = (
    "horizons_sources_jpl_ssodnet_mass_gt_1e17_arc15000.json",
    "selection_jpl_ssodnet_mass_gt_1e17_arc15000_summary.json",
)
PAPER_FIGURES = (
    "maxwell_decoy_operators.pdf",
    "kepler_showcase_sixpanel.pdf",
)
GENERATED_FIGURE_SOURCES = {
    "maxwell_decoy_operators.pdf": "maxwell_exact/figures/maxwell_decoy_operators.pdf",
    "kepler_showcase_sixpanel.pdf": "kepler_direct/figures/kepler_showcase_sixpanel.pdf",
}
DATA_DIRS = (
    "feynman_de",
    "feynman_complex",
    "feynman_de_compositional",
    "maxwell_benchmark",
)
RESULT_DIRS = (
    "scalar_stlsq",
    "scalar_hybrid",
    "complex_stlsq",
    "compositional_fss",
    "compositional_dictionary",
    "symmetry_alias",
    "symmetry_reduction",
    "symmetry_order2",
    "symmetry_solvable",
    "maxwell_exact",
    "maxwell_noise",
    "kepler_direct",
    "kepler_noether",
    "sealed_test",
)
REQUIRED_RESULT_FILES = {
    "scalar_stlsq": "summary.json",
    "scalar_hybrid": "summary.json",
    "complex_stlsq": "summary.json",
    "compositional_fss": "compositional_summary.json",
    "compositional_dictionary": "stlsq_dictionary_baselines_summary.json",
    "symmetry_alias": "gs_alias_arbitration_summary.json",
    "symmetry_reduction": "gs_reduction_experiment_summary.json",
    "symmetry_order2": "gs_order2_reduction_summary.json",
    "symmetry_solvable": "gs_solvable_cascade_summary.json",
    "maxwell_exact": "summary.json",
    "maxwell_noise": "noise_sweep.json",
    "kepler_direct": "kepler_ephemeris_weathered_summary.json",
    "kepler_noether": "noether_kepler_summary.json",
    "sealed_test": "scalar_hybrid/summary.json",
}
REQUIRED_ADDITIONAL_RESULT_FILES = {
    "sealed_test": ("scalar_stlsq/summary.json", "complex_stlsq/summary.json"),
    "maxwell_exact": ("figures/maxwell_decoy_operators.pdf",),
    "kepler_direct": ("figures/kepler_showcase_sixpanel.pdf",),
}
COMPACT_RESULT_SUFFIXES = {
    ".json", ".csv", ".human", ".md", ".txt", ".pdf",
}
TEXT_SUFFIXES = {".json", ".csv", ".human", ".md", ".txt"}


def _utc_now() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        return datetime.fromtimestamp(int(epoch), timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(repo: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
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


def _versions() -> dict[str, str | None]:
    packages = ("nestynet", "nestynet-sr", "torch", "numpy", "scipy", "pandas", "matplotlib", "sympy", "astropy")
    out: dict[str, str | None] = {}
    for package in packages:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _normalizers(*roots: tuple[Path, str]) -> list[tuple[str, str]]:
    pairs = []
    for root, marker in roots:
        pairs.append((str(root.resolve()), marker))
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def _copy_file(source: Path, destination: Path, replacements: list[tuple[str, str]]) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"artifact source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8")
        for old, new in replacements:
            text = text.replace(old, new)
        destination.write_text(text, encoding="utf-8")
    else:
        shutil.copyfile(source, destination)
    os.chmod(destination, 0o644)


def _copy_tree(
    source: Path,
    destination: Path,
    replacements: list[tuple[str, str]],
    *,
    suffixes: set[str] | None = None,
) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlinks are not permitted in the artifact: {path}")
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        _copy_file(path, destination / path.relative_to(source), replacements)
        count += 1
    return count


def _benchmark_ids(spec_path: Path) -> list[str]:
    ids = []
    for line in spec_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.append(stripped.split(maxsplit=1)[0])
    return ids


def _copy_selected_data(
    name: str,
    source: Path,
    destination: Path,
    replacements: list[tuple[str, str]],
    *,
    strict: bool,
) -> int:
    if name == "feynman_de":
        prefixes = [f"de{pid}" for pid in _benchmark_ids(REPO_ROOT / "data" / "feynman_de_benchmark.txt")]
        required = {
            f"{prefix}_ic{trajectory}.csv"
            for prefix in prefixes
            for trajectory in range(6)
        }
    elif name == "feynman_complex":
        prefixes = _benchmark_ids(REPO_ROOT / "data" / "feynman_complex_benchmark.txt")
        required = {f"{prefix}_{component}.csv" for prefix in prefixes for component in ("u", "v")}
    elif name == "feynman_de_compositional":
        prefixes = ["de900", "de901", "de902", "de903"]
        required = {
            f"{prefix}_ic{trajectory}.csv"
            for prefix in prefixes
            for trajectory in range(6)
        }
    elif name == "maxwell_benchmark":
        prefixes = ["mw000", "mw001", "mw002", "mw003"]
        required = {
            filename
            for prefix in prefixes
            for filename in (f"{prefix}.npz", f"{prefix}.meta.json")
        }
    else:
        raise RuntimeError(f"no Paper IV data-selection rule for {name}")

    def selected(filename: str) -> bool:
        return any(filename == prefix or filename.startswith(prefix + "_") or filename.startswith(prefix + ".") for prefix in prefixes)

    available = {
        path.name
        for path in source.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    missing = sorted(required - available)
    if strict and missing:
        preview = "\n  ".join(missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise RuntimeError(
            f"incomplete {name} input inventory; missing {len(missing)} required files:\n  "
            f"{preview}{suffix}"
        )

    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlinks are not permitted in the artifact: {path}")
        if not selected(path.name):
            continue
        _copy_file(path, destination / path.relative_to(source), replacements)
        count += 1
    if count == 0:
        raise RuntimeError(f"no selected Paper IV data files found in {source}")
    return count


def _summary_rows(summary_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = _load_json(summary_path)
    problems = payload.get("problems")
    if not isinstance(problems, list):
        raise RuntimeError(f"summary has no problems list: {summary_path}")
    if not all(isinstance(row, dict) for row in problems):
        raise RuntimeError(f"summary problems must be JSON objects: {summary_path}")
    counts = Counter(str(row.get("status")) for row in problems)
    return problems, dict(counts)


def _validate_headline_results(results_root: Path, protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = protocol["expected_results"]
    for name in ("scalar_stlsq", "scalar_hybrid", "complex_stlsq"):
        path = results_root / name / "summary.json"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        try:
            problems, counts = _summary_rows(path)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
            continue
        wanted = expected[name]
        n_problems = len(problems)
        if n_problems != int(wanted["problems"]):
            errors.append(f"{name}: expected {wanted['problems']} problems, found {n_problems}")
        if name == "complex_stlsq":
            wanted_ids = _benchmark_ids(REPO_ROOT / "data" / "feynman_complex_benchmark.txt")
            found_ids = [str(row.get("id", "")) for row in problems]
        else:
            wanted_ids = _benchmark_ids(REPO_ROOT / "data" / "feynman_de_benchmark.txt")
            found_ids = [str(row.get("id", "")).removeprefix("de") for row in problems]
        duplicate_ids = sorted(pid for pid, count in Counter(found_ids).items() if count > 1)
        if duplicate_ids:
            errors.append(f"{name}: duplicate problem IDs: {duplicate_ids}")
        if set(found_ids) != set(wanted_ids):
            missing = sorted(set(wanted_ids) - set(found_ids))
            extra = sorted(set(found_ids) - set(wanted_ids))
            errors.append(f"{name}: problem-ID mismatch; missing={missing}, extra={extra}")
        expected_status_by_id = {
            str(pid): str(status)
            for pid, status in wanted.get("status_by_id", {}).items()
        }
        found_status_by_id = {
            pid: str(row.get("status"))
            for pid, row in zip(found_ids, problems, strict=True)
        }
        status_mismatches = {
            pid: {"expected": status, "found": found_status_by_id.get(pid)}
            for pid, status in expected_status_by_id.items()
            if found_status_by_id.get(pid) != status
        }
        if status_mismatches:
            errors.append(f"{name}: per-problem status mismatch: {status_mismatches}")
        for status, wanted_count in wanted["statuses"].items():
            if counts.get(status, 0) != int(wanted_count):
                errors.append(
                    f"{name}: expected {wanted_count} {status}, found {counts.get(status, 0)}"
                )

    comp_path = results_root / "compositional_fss" / "compositional_summary.json"
    if not comp_path.is_file():
        errors.append(f"missing {comp_path}")
    else:
        payload = _load_json(comp_path)
        ids = [str(value) for value in payload.get("ids", [])]
        wanted_ids = [str(value) for value in expected["compositional_fss"]["problem_ids"]]
        counts = payload.get("counts", {})
        if len(ids) != len(set(ids)) or set(ids) != set(wanted_ids):
            errors.append(f"compositional_fss: expected IDs {wanted_ids}, found {ids}")
        if int(counts.get("PASS", 0)) != 4:
            errors.append(f"compositional_fss: expected 4 PASS, found {counts.get('PASS', 0)}")
        cases = payload.get("cases", [])
        case_by_id = {
            str(row.get("problem_id")): row
            for row in cases
            if isinstance(row, dict)
        }
        if set(case_by_id) != set(wanted_ids) or len(cases) != len(wanted_ids):
            errors.append("compositional_fss: per-case records do not cover 900--903 uniquely")
        else:
            non_pass = sorted(pid for pid, row in case_by_id.items() if row.get("status") != "PASS")
            if non_pass:
                errors.append(f"compositional_fss: non-PASS per-case records: {non_pass}")
    return errors


VERDICT_EXIT_STEPS = {"scalar_stlsq", "scalar_hybrid", "complex_stlsq"}
PER_CASE_MERGED_STEPS = {"scalar_stlsq", "scalar_hybrid"}


def _normalize_kepler_cache_argument(command: list[str]) -> list[str]:
    """Treat source and archived copies of the content-addressed cache alike."""
    normalized = list(command)
    if "--accel_cache_dir" in normalized:
        index = normalized.index("--accel_cache_dir")
        if index + 1 < len(normalized):
            normalized[index + 1] = "$KEPLER_CACHE"
    return normalized


def _step_command_errors(
    protocol_row: dict[str, Any],
    run_row: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    replacements = {
        "{python}": str(payload.get("environment", {}).get("python_executable", "")),
        "{repo}": str(payload.get("source", {}).get("path", "")),
        "{data}": str(payload.get("data_root", "")),
        "{results}": str(payload.get("results_root", "")),
        "{jobs}": str(payload.get("jobs", "")),
        "{kepler_manifest}": str(payload.get("kepler_manifest", "")),
        "{kepler_selection}": str(payload.get("kepler_selection", "")),
        "{kepler_cache}": str(payload.get("kepler_cache", "")),
    }
    expected_command = []
    for raw in protocol_row["argv"]:
        value = str(raw)
        for marker, replacement in replacements.items():
            value = value.replace(marker, replacement)
        expected_command.append(value)
    actual_command = [str(value) for value in run_row.get("command", [])]
    expected_command = _normalize_kepler_cache_argument(expected_command)
    actual_command = _normalize_kepler_cache_argument(actual_command)
    if actual_command != expected_command:
        return [
            f"step manifest command mismatch for {protocol_row['id']}: "
            f"expected={expected_command}, found={actual_command}"
        ]
    return []


def _validate_assembled_provenance(
    workspace: Path,
    results_root: Path,
    protocol: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Validate consolidated per-step provenance instead of one monolithic run.

    The scientific gate stays _validate_headline_results (per-case statuses vs
    the frozen expectations).  This mode establishes that every protocol step
    has attributable provenance: a single-step runner manifest under
    par/<step>/ whose command matches the frozen protocol, or, for the
    per-case-parallel benchmark arms, a merged summary with full by-id case
    coverage.  Verdict-bearing benchmarks exit 1 by design when cases FAIL,
    so their recorded returncode may be 0 or 1; per-step git state (including
    dirty flags) is recorded into the archive rather than asserted, with the
    narrative in reproducibility/paper4/PROVENANCE_NOTES.md, whose presence
    is required.
    """
    errors: list[str] = []
    record: dict[str, Any] = {"provenance_mode": "assembled", "steps": []}
    notes = REPO_ROOT / "reproducibility" / "paper4" / "PROVENANCE_NOTES.md"
    if not notes.is_file():
        errors.append(f"assembled provenance requires the narrative notes: {notes}")

    for protocol_row in protocol["steps"]:
        sid = str(protocol_row["id"])
        manifest_path = workspace / "par" / sid / "paper4_run_manifest.json"
        if manifest_path.is_file():
            try:
                payload = _load_json(manifest_path)
            except (OSError, ValueError) as exc:
                errors.append(f"invalid step manifest {manifest_path}: {exc}")
                continue
            if payload.get("executed") is not True or payload.get("smoke") is not False:
                errors.append(f"{sid}: step manifest must describe an executed non-smoke run")
            run_row = next(
                (
                    row
                    for row in payload.get("runs", [])
                    if isinstance(row, dict) and str(row.get("id")) == sid
                ),
                None,
            )
            if run_row is None:
                errors.append(f"{sid}: step manifest does not contain a run record for the step")
                continue
            if not run_row.get("completed_at_utc"):
                errors.append(f"{sid}: step run record is incomplete")
            rc = run_row.get("returncode")
            allowed = (0, 1) if sid in VERDICT_EXIT_STEPS else (0,)
            if rc not in allowed:
                errors.append(f"{sid}: step returncode {rc} not in {allowed}")
            errors.extend(_step_command_errors(protocol_row, run_row, payload))
            record["steps"].append(
                {
                    "id": sid,
                    "mode": "runner_step_manifest",
                    "returncode": rc,
                    "completed_at_utc": run_row.get("completed_at_utc"),
                    "source": payload.get("source", {}),
                    "nestynet_core": payload.get("nestynet_core", None),
                }
            )
            continue
        if sid in PER_CASE_MERGED_STEPS:
            summary_path = results_root / sid / "summary.json"
            if not summary_path.is_file():
                errors.append(f"{sid}: no step manifest and no merged summary {summary_path}")
                continue
            payload = _load_json(summary_path)
            if payload.get("merged_from_per_case_runs") is not True:
                errors.append(f"{sid}: summary is not a per-case merge")
            wanted_ids = sorted(
                str(pid)
                for pid in protocol["expected_results"][sid]["status_by_id"]
            )
            missing_cases = [
                pid
                for pid in wanted_ids
                if not (results_root / sid / "by_id" / pid / "summary.json").is_file()
            ]
            if missing_cases:
                errors.append(f"{sid}: missing per-case records for {missing_cases}")
            record["steps"].append(
                {
                    "id": sid,
                    "mode": "per_case_merged",
                    "n_cases": len(wanted_ids) - len(missing_cases),
                    "case_root": "by_id",
                }
            )
            continue
        errors.append(f"{sid}: no provenance found (no par/{sid} manifest)")
    return errors, record


def _validate_run_manifest(
    workspace: Path,
    protocol: dict[str, Any],
    source_state: dict[str, Any],
) -> list[str]:
    path = workspace / "paper4_run_manifest.json"
    if not path.is_file():
        return [f"missing full-protocol run manifest: {path}"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        return [f"invalid run manifest {path}: {exc}"]
    errors: list[str] = []
    if payload.get("executed") is not True or payload.get("smoke") is not False:
        errors.append("run manifest must describe an executed non-smoke protocol")
    if payload.get("protocol") != protocol:
        errors.append("run manifest embedded protocol does not match the frozen Paper IV protocol")
    wanted_ids = [str(row["id"]) for row in protocol["steps"]]
    runs = payload.get("runs", [])
    found_ids = [str(row.get("id")) for row in runs if isinstance(row, dict)]
    if found_ids != wanted_ids:
        errors.append(f"run manifest step mismatch; expected={wanted_ids}, found={found_ids}")
    failed = [
        str(row.get("id"))
        for row in runs
        if not isinstance(row, dict) or row.get("returncode") != 0
    ]
    if failed:
        errors.append(f"run manifest has failed/incomplete steps: {failed}")
    run_source = payload.get("source", {})
    if run_source.get("git_revision") != source_state.get("git_revision"):
        errors.append(
            "run/source Git revision mismatch: "
            f"run={run_source.get('git_revision')}, build={source_state.get('git_revision')}"
        )
    if run_source.get("working_tree_dirty") is not False:
        errors.append("the full protocol was executed from a dirty or unknown working tree")

    replacements = {
        "{python}": str(payload.get("environment", {}).get("python_executable", "")),
        "{repo}": str(run_source.get("path", "")),
        "{data}": str(payload.get("data_root", "")),
        "{results}": str(payload.get("results_root", "")),
        "{jobs}": str(payload.get("jobs", "")),
        "{kepler_manifest}": str(payload.get("kepler_manifest", "")),
        "{kepler_selection}": str(payload.get("kepler_selection", "")),
        "{kepler_cache}": str(payload.get("kepler_cache", "")),
    }
    for protocol_row, run_row in zip(protocol["steps"], runs, strict=False):
        if not isinstance(run_row, dict):
            continue
        expected_command = []
        for raw in protocol_row["argv"]:
            value = str(raw)
            for marker, replacement in replacements.items():
                value = value.replace(marker, replacement)
            expected_command.append(value)
        actual_command = [str(value) for value in run_row.get("command", [])]
        expected_command = _normalize_kepler_cache_argument(expected_command)
        actual_command = _normalize_kepler_cache_argument(actual_command)
        if actual_command != expected_command:
            errors.append(
                f"run manifest command mismatch for {protocol_row['id']}: "
                f"expected={expected_command}, found={actual_command}"
            )
    return errors


def _resolve_csv_path(raw: str, manifest: Path, kepler_data_root: Path) -> Path:
    value = Path(raw)
    candidates = []
    if value.is_absolute():
        candidates.append(value)
    else:
        candidates.extend((REPO_ROOT / value, manifest.parent / value, kepler_data_root / value.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n".join(f"  {path}" for path in candidates)
    raise RuntimeError(f"cannot resolve Kepler CSV {raw!r}; checked:\n{checked}")


def _copy_kepler_csv(source: Path, destination: Path) -> int:
    """Copy one normalized state CSV and return its number of data rows."""
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"Kepler source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    newline_count = 0
    last_byte = b""
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        for block in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
            output_stream.write(block)
    os.chmod(destination, 0o644)
    line_count = newline_count + (1 if last_byte and last_byte != b"\n" else 0)
    return max(0, line_count - 1)


def _stage_kepler(
    manifest_path: Path,
    kepler_data_root: Path,
    destination: Path,
    replacements: list[tuple[str, str]],
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        if allow_incomplete:
            return {"present": False, "reason": f"missing {manifest_path}"}
        raise RuntimeError(f"missing Kepler manifest: {manifest_path}")
    rows = _load_json(manifest_path)
    if not isinstance(rows, list):
        raise RuntimeError(f"Kepler manifest must contain a list: {manifest_path}")
    if not allow_incomplete and len(rows) != 308:
        raise RuntimeError(f"expected 308 Kepler bodies, found {len(rows)} in {manifest_path}")

    raw_destination = destination / KEPLER_RAW_DIR_NAME
    localized_rows = []
    names: set[str] = set()
    orbit_ids: set[str] = set()
    horizons_commands: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("every Kepler manifest row must be a JSON object")
        orbit_id = str(row.get("orbit_id", "")).strip()
        horizons_command = str(row.get("horizons_command", "")).strip()
        if not orbit_id or orbit_id in orbit_ids:
            raise RuntimeError(f"missing or duplicate Kepler orbit_id: {orbit_id!r}")
        if not horizons_command or horizons_command in horizons_commands:
            raise RuntimeError(
                f"missing or duplicate Kepler horizons_command: {horizons_command!r}"
            )
        orbit_ids.add(orbit_id)
        horizons_commands.add(horizons_command)
        if not allow_incomplete:
            expected_fields = {
                "split": "candidate",
                "start_date": "1980-01-01",
                "stop_date": "2009-12-31",
                "cadence_days": 1.0,
                "n_rows": 10958,
            }
            for field, expected_value in expected_fields.items():
                if row.get(field) != expected_value:
                    raise RuntimeError(
                        f"Kepler {orbit_id} has {field}={row.get(field)!r}; "
                        f"expected {expected_value!r}"
                    )
            if not str(row.get("center_name", "")).strip().startswith("Sun (10)"):
                raise RuntimeError(
                    f"Kepler {orbit_id} is not Sun-centred: {row.get('center_name')!r}"
                )
        source = _resolve_csv_path(str(row["csv_path"]), manifest_path, kepler_data_root)
        name = source.name
        if name in names:
            raise RuntimeError(f"duplicate Kepler CSV basename: {name}")
        names.add(name)
        copied_rows = _copy_kepler_csv(source, raw_destination / name)
        if not allow_incomplete and copied_rows != int(row["n_rows"]):
            raise RuntimeError(
                f"Kepler {orbit_id} CSV has {copied_rows} data rows; "
                f"manifest declares {row['n_rows']}"
            )
        localized = dict(row)
        localized["csv_path"] = f"data/kepler/{KEPLER_RAW_DIR_NAME}/{name}"
        localized_rows.append(localized)

    _write_json(destination / KEPLER_MANIFEST_NAME, localized_rows)
    _write_json(destination / KEPLER_LOCALIZED_NAME, localized_rows)
    for name in KEPLER_METADATA_NAMES:
        path = kepler_data_root / name
        if path.is_file():
            _copy_file(path, destination / name, replacements)
        elif not allow_incomplete:
            raise RuntimeError(f"missing Kepler provenance file: {path}")
    return {
        "present": True,
        "bodies": len(localized_rows),
        "raw_directory": KEPLER_RAW_DIR_NAME,
        "manifest": KEPLER_LOCALIZED_NAME,
    }


def _stage_kepler_cache(
    cache_directory: Path,
    orbit_ids: list[str],
    destination: Path,
    replacements: list[tuple[str, str]],
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    """Copy and validate the per-body analytic-surrogate acceleration cache."""
    if not cache_directory.is_dir():
        if allow_incomplete:
            return {"present": False, "reason": f"missing {cache_directory}"}
        raise RuntimeError(f"missing Kepler surrogate cache: {cache_directory}")

    expected = {f"{orbit_id}.npz" for orbit_id in orbit_ids}
    available = {
        path.name
        for path in cache_directory.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".npz"
    }
    missing = sorted(expected - available)
    extra = sorted(available - expected)
    if not allow_incomplete and (missing or extra):
        raise RuntimeError(
            "Kepler surrogate-cache inventory mismatch; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    certificate_norms: list[float] = []
    invalid: list[str] = []
    copied = 0
    required_arrays = {"ax", "ay", "input_sha", "provenance_json"}
    for name in sorted(expected & available):
        source = cache_directory / name
        try:
            with np.load(source, allow_pickle=False) as payload:
                absent = required_arrays - set(payload.files)
                if absent:
                    raise ValueError(f"missing arrays {sorted(absent)}")
                if payload["ax"].shape != (10958,) or payload["ay"].shape != (10958,):
                    raise ValueError(
                        f"unexpected acceleration shapes ax={payload['ax'].shape}, "
                        f"ay={payload['ay'].shape}"
                    )
                input_sha = str(payload["input_sha"].item())
                if re.fullmatch(r"[0-9a-f]{64}", input_sha) is None:
                    raise ValueError("input_sha is not a lowercase SHA-256 digest")
                provenance = json.loads(str(payload["provenance_json"].item()))
                measured = provenance["certificate"]["measured_derivative_rel_rmse"]
                x_error = float(measured["x"])
                y_error = float(measured["y"])
                if not np.isfinite(x_error) or not np.isfinite(y_error):
                    raise ValueError("certificate contains a non-finite error")
                certificate_norms.append(float(np.hypot(x_error, y_error) / np.sqrt(2.0)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append(f"{name}: {exc}")
            if not allow_incomplete:
                raise RuntimeError(f"invalid Kepler surrogate cache entry {name}: {exc}") from exc
        _copy_file(source, destination / name, replacements)
        copied += 1

    return {
        "present": True,
        "directory": KEPLER_CACHE_DIR_NAME,
        "files": copied,
        "entries_with_measured_certificates": len(certificate_norms),
        "median_per_body_derivative_certificate_rel_rmse": (
            float(np.median(certificate_norms)) if certificate_norms else None
        ),
        "missing": missing,
        "extra": extra,
        "invalid": invalid,
    }


def _scan_private_paths(root: Path) -> None:
    leaks = []
    private_markers = ("/" + "Users/", "/" + "DataDir/")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in private_markers):
            leaks.append(str(path.relative_to(root)))
    if leaks:
        raise RuntimeError("local absolute paths leaked into artifact:\n  " + "\n  ".join(leaks))


def _write_checksums(root: Path) -> int:
    checksum_path = root / "SHA256SUMS"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    if info.isfile():
        info.mode = 0o644
    elif info.isdir():
        info.mode = 0o755
    return info


def _make_archive(source: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(prefix=f".{archive.name}.", dir=archive.parent, delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            with gzip.GzipFile(fileobj=temporary, mode="wb", mtime=0, filename="") as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    tar.add(source, arcname=source.name, recursive=True, filter=_tar_filter)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, archive)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the NestyNet Paper IV Zenodo data/reference-artifact archive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", type=Path, required=True, help="Completed Paper IV run workspace")
    parser.add_argument("--data-root", type=Path, default=None, help="Input data root")
    parser.add_argument("--results-root", type=Path, default=None, help="Reference result root")
    parser.add_argument("--kepler-data-root", type=Path, default=None, help="Directory holding exact HORIZONS files and provenance")
    parser.add_argument("--kepler-manifest", type=Path, default=None, help="Exact 308-body daily manifest")
    parser.add_argument("--kepler-cache-dir", type=Path, default=None, help="Per-body analytic-surrogate acceleration cache")
    parser.add_argument("--paper-figures", type=Path, default=REPO_ROOT.parent / "NestyNet_papers" / "figures_paper4", help="Fallback figure directory used only with --allow-incomplete")
    parser.add_argument("--output-dir", type=Path, required=True, help="Staged artifact directory")
    parser.add_argument("--archive", type=Path, default=None, help="Output .tgz path")
    parser.add_argument("--allow-incomplete", action="store_true", help="Testing only: stage available material without final-result validation")
    parser.add_argument("--allow-dirty", action="store_true", help="Permit an uncommitted source tree (recorded in provenance; never recommended for a final archive)")
    parser.add_argument(
        "--provenance",
        choices=("single-run", "assembled"),
        default="single-run",
        help="single-run: require one full-protocol workspace manifest; "
        "assembled: validate per-step manifests plus per-case-merged benchmark "
        "arms, recording per-step git state into the archive",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory/archive")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    data_root = (args.data_root or (workspace / "data")).expanduser().resolve()
    results_root = (args.results_root or (workspace / "results")).expanduser().resolve()
    kepler_data_root = (
        args.kepler_data_root
        or (data_root / "kepler")
    ).expanduser().resolve()
    kepler_manifest = (
        args.kepler_manifest
        or (kepler_data_root / KEPLER_MANIFEST_NAME)
    ).expanduser().resolve()
    kepler_cache_dir = (
        args.kepler_cache_dir
        or (kepler_data_root / KEPLER_CACHE_DIR_NAME)
    ).expanduser().resolve()
    paper_figures = args.paper_figures.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    archive = (args.archive or output_dir.with_suffix(".tgz")).expanduser().resolve()

    if output_dir.exists() and not args.force:
        raise RuntimeError(f"output directory already exists (use --force): {output_dir}")
    if archive.exists() and not args.force:
        raise RuntimeError(f"archive already exists (use --force): {archive}")

    protocol = _load_json(PROTOCOL_PATH)
    source_state = _git_state(REPO_ROOT)
    validation_errors = _validate_headline_results(results_root, protocol)
    assembled_record: dict[str, Any] | None = None
    if args.provenance == "assembled":
        assembled_errors, assembled_record = _validate_assembled_provenance(
            workspace, results_root, protocol
        )
        validation_errors.extend(assembled_errors)
    else:
        validation_errors.extend(_validate_run_manifest(workspace, protocol, source_state))
    if validation_errors and not args.allow_incomplete:
        raise RuntimeError(
            "refusing to build final Paper IV artifact:\n  " + "\n  ".join(validation_errors)
        )
    if not args.allow_incomplete and not args.allow_dirty:
        if source_state["git_revision"] is None:
            raise RuntimeError("strict Paper IV artifact builds require a Git checkout")
        if source_state["working_tree_dirty"] is not False:
            raise RuntimeError(
                "strict Paper IV artifact builds require a clean NestyNet-SR working tree; "
                "commit the final rerun code first or use --allow-dirty only for a non-final audit"
            )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    if archive.exists():
        archive.unlink()
    output_dir.mkdir(parents=True)

    replacements = _normalizers(
        (workspace, "$PAPER4_WORKSPACE"),
        (data_root, "$PAPER4_DATA"),
        (results_root, "$PAPER4_RESULTS"),
        (REPO_ROOT, "$NESTYNET_SR"),
        # Last resort for machine-specific stragglers (interpreter paths in
        # recorded commands and resource reports); _normalizers sorts by
        # length so the specific roots above always win first.
        (Path.home(), "$HOME"),
    )
    if assembled_record is not None:
        core_paths = sorted(
            {
                str(core.get("path"))
                for step in assembled_record["steps"]
                for core in (step.get("nestynet_core") or {},)
                if core.get("path")
            }
        )
        for core_path in core_paths:
            replacements.append((core_path, "$NESTYNET_CORE"))
        replacements.sort(key=lambda pair: len(pair[0]), reverse=True)
    inventory: dict[str, Any] = {"data": {}, "results": {}, "figures": []}

    for name in DATA_DIRS:
        source = data_root / name
        if not source.is_dir():
            inventory["data"][name] = {"present": False}
            if not args.allow_incomplete:
                raise RuntimeError(f"missing final input directory: {source}")
            continue
        count = _copy_selected_data(
            name,
            source,
            output_dir / "data" / name,
            replacements,
            strict=not bool(args.allow_incomplete),
        )
        inventory["data"][name] = {"present": True, "files": count}

    inventory["data"]["kepler"] = _stage_kepler(
        kepler_manifest,
        kepler_data_root,
        output_dir / "data" / "kepler",
        replacements,
        allow_incomplete=bool(args.allow_incomplete),
    )
    kepler_rows = _load_json(kepler_manifest) if kepler_manifest.is_file() else []
    orbit_ids = [
        str(row["orbit_id"])
        for row in kepler_rows
        if isinstance(row, dict) and row.get("orbit_id")
    ]
    inventory["data"]["kepler"]["surrogate_cache"] = _stage_kepler_cache(
        kepler_cache_dir,
        orbit_ids,
        output_dir / "data" / "kepler" / KEPLER_CACHE_DIR_NAME,
        replacements,
        allow_incomplete=bool(args.allow_incomplete),
    )

    spec_destination = output_dir / "specs"
    for name in ("feynman_de_benchmark.txt", "feynman_complex_benchmark.txt", "feynman_de_compositional.txt"):
        _copy_file(REPO_ROOT / "data" / name, spec_destination / name, replacements)

    for name in RESULT_DIRS:
        source = results_root / name
        if not source.is_dir():
            inventory["results"][name] = {"present": False}
            if not args.allow_incomplete:
                raise RuntimeError(f"missing final result directory: {source}")
            continue
        required_result = source / REQUIRED_RESULT_FILES[name]
        if not required_result.is_file() and not args.allow_incomplete:
            raise RuntimeError(f"missing required compact result: {required_result}")
        for relative in REQUIRED_ADDITIONAL_RESULT_FILES.get(name, ()):
            required_additional = source / relative
            if not required_additional.is_file() and not args.allow_incomplete:
                raise RuntimeError(f"missing required generated paper figure: {required_additional}")
        count = _copy_tree(
            source, output_dir / "results_ref" / name, replacements,
            suffixes=COMPACT_RESULT_SUFFIXES,
        )
        inventory["results"][name] = {"present": True, "files": count}

    for name in PAPER_FIGURES:
        source = results_root / GENERATED_FIGURE_SOURCES[name]
        if not source.is_file() and args.allow_incomplete:
            source = paper_figures / name
        if source.is_file():
            _copy_file(source, output_dir / "figures" / name, replacements)
            inventory["figures"].append(name)
        elif not args.allow_incomplete:
            raise RuntimeError(f"missing Paper IV figure: {source}")

    reproducibility = output_dir / "reproducibility"
    _copy_file(
        REPO_ROOT / "reproducibility" / "paper4" / "README.md",
        output_dir / "README.md",
        replacements,
    )
    _copy_file(
        REPO_ROOT / "reproducibility" / "paper4" / "README.md",
        reproducibility / "README.md",
        replacements,
    )
    _copy_file(PROTOCOL_PATH, reproducibility / "protocol.json", replacements)
    _copy_file(
        REPO_ROOT / "reproducibility" / "paper4" / "requirements-paper4.txt",
        reproducibility / "requirements-paper4.txt",
        replacements,
    )
    _copy_file(
        REPO_ROOT / "PAPER4_REPRODUCIBILITY.md",
        reproducibility / "PAPER4_REPRODUCIBILITY.md",
        replacements,
    )
    notes_src = REPO_ROOT / "reproducibility" / "paper4" / "PROVENANCE_NOTES.md"
    _copy_file(notes_src, reproducibility / "PROVENANCE_NOTES.md", replacements)
    inventory["reproducibility"] = [
        "README.md",
        "protocol.json",
        "requirements-paper4.txt",
        "PAPER4_REPRODUCIBILITY.md",
        "PROVENANCE_NOTES.md",
    ]
    run_manifest_path = workspace / "paper4_run_manifest.json"
    if assembled_record is None and run_manifest_path.is_file():
        _copy_file(
            run_manifest_path,
            reproducibility / "paper4_run_manifest.json",
            replacements,
        )
        inventory["reproducibility"].append("paper4_run_manifest.json")
    if assembled_record is not None:
        record_text = json.dumps(assembled_record, indent=2, sort_keys=True) + "\n"
        for marker, replacement in replacements:
            record_text = record_text.replace(marker, replacement)
        target = reproducibility / "assembled_run_provenance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(record_text, encoding="utf-8")
        for step_row in assembled_record["steps"]:
            if step_row.get("mode") != "runner_step_manifest":
                continue
            step_manifest = workspace / "par" / str(step_row["id"]) / "paper4_run_manifest.json"
            _copy_file(
                step_manifest,
                reproducibility / "step_manifests" / f"{step_row['id']}.json",
                replacements,
            )
        inventory["reproducibility"].extend(
            ["assembled_run_provenance.json", "step_manifests/"]
        )

    manifest = {
        "schema_version": 1,
        "title": "NestyNet Paper IV data and reference artifacts",
        "created_at_utc": _utc_now(),
        "software": {
            "repository": "https://github.com/RodrigoIbata/NestyNet_SR",
            **source_state,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _versions(),
        },
        "strict_validation": not bool(args.allow_incomplete),
        "validation_errors": validation_errors,
        "inventory": inventory,
    }
    _write_json(output_dir / "ARTIFACT.json", manifest)
    _scan_private_paths(output_dir)
    n_hashed = _write_checksums(output_dir)
    _make_archive(output_dir, archive)

    record = {
        "schema_version": 1,
        "doi": None,
        "record_url": None,
        "download_url": None,
        "filename": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
        "files_hashed": n_hashed,
    }
    record_path = archive.with_name(f"{archive.name}.record.json")
    _write_json(record_path, record)
    print(f"Staged artifact: {output_dir}")
    print(f"Archive: {archive}")
    print(f"Bytes: {record['bytes']}")
    print(f"SHA-256: {record['sha256']}")
    print(f"Record template: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
