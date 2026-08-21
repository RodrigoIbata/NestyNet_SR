# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Assemble the Paper III benchmark-evidence artifact.

The artifact carries the evidence behind the two benchmark tables, not the
campaigns that produced it: twelve capsules (per-problem reports, summary
CSV, campaign manifest, structural audit) for the noisy full-pipeline
table, and the oracle run's summary plus structural audit for the
factorized-search table.  Models, checkpoints and logs stay where they
were produced; the benchmark input data is a separate, already-published
Zenodo record.

The build refuses to proceed unless

* every frozen cell in ``reproducibility/paper3/expected_results.json``
  validates (completeness and agreement with the recorded totals),
* no shipped text retains a machine-specific absolute path,
* no shipped text retains an assistant or model attribution.

    python3 scripts/build_paper3_artifact.py \\
        --capsules /path/to/NestyNet_paper3_capsules
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".human"}

# Attribution markers that must never reach a released artifact.
ATTRIBUTION = re.compile(
    r"claude|chatgpt|gpt-?[45]|co-authored-by|openai|anthropic|copilot|"
    r"generated with|ai[- ]assisted",
    re.IGNORECASE,
)
PRIVATE_PATH = re.compile(r"/(?:Users|DataDir)/|/home2020/|/home/[a-z]")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state(repo: Path) -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               check=True).stdout.strip()
        return {"git_revision": rev, "working_tree_dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "working_tree_dirty": None}


def _normalizers(*roots: tuple[Path, str]) -> list[tuple[str, str]]:
    pairs = [(str(root.resolve()), marker) for root, marker in roots]
    # Longest first, so a nested root never loses to its parent.
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def _copy_file(src: Path, dst: Path, replacements: list[tuple[str, str]]) -> None:
    if src.is_symlink() or not src.is_file():
        raise RuntimeError(f"artifact source must be a regular file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_SUFFIXES:
        text = src.read_text(encoding="utf-8", errors="strict")
        for marker, replacement in replacements:
            text = text.replace(marker, replacement)
        # HPC paths carry an institution and user segment before the
        # workspace; collapse the whole prefix rather than one segment.
        text = re.sub(r"/home2020/home/[^/\s\"',}\]]+/[^/\s\"',}\]]+/"
                      r"(?P<ws>[A-Za-z0-9_.-]+)", r"$HPC_WORKSPACE/\g<ws>", text)
        text = re.sub(r"/home2020/home/[^/\s\"',}\]]+(?:/[^/\s\"',}\]]+)?",
                      "$HPC_HOME", text)
        # Whatever the specific roots miss: the benchmark inputs live under a
        # separate data root, and interpreter paths are recorded verbatim.
        text = text.replace("$HOME/" + "DataDir/Machine_Learning/AIFeynman_data",
                            "$AIFEYNMAN_DATA")
        text = re.sub(r"\$HOME/[A-Za-z0-9_.-]+/bin/python[0-9.]*", "$PYTHON", text)
        dst.write_text(text, encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def _scan(root: Path) -> None:
    private, attributed = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(root))
        if PRIVATE_PATH.search(text):
            private.append(rel)
        if ATTRIBUTION.search(text):
            attributed.append(rel)
    if private:
        raise RuntimeError("machine-specific paths leaked into the artifact:\n  "
                           + "\n  ".join(private[:20]))
    if attributed:
        raise RuntimeError("assistant/model attribution leaked into the artifact:\n  "
                           + "\n  ".join(attributed[:20]))


def _validate(capsules: Path) -> None:
    rc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_paper3_results.py"),
         "--capsules", str(capsules)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(rc.stdout)
    if rc.returncode != 0:
        raise RuntimeError("frozen cells do not validate; refusing to build")


def _validate_oracle_evidence(audit_path: Path, summary_path: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_audit_fields = {
        "criterion", "tol", "n_cases", "structural_recoveries",
        "classes", "details", "counts",
    }
    if set(audit) != expected_audit_fields:
        raise RuntimeError(
            "oracle structural audit does not match the final release schema"
        )
    if audit.get("n_cases") != 115 or audit.get("structural_recoveries") != 88:
        raise RuntimeError("oracle structural audit must report 88/115")
    classes = audit.get("classes", {})
    if len(classes) != 115 or sum(value == "structural" for value in classes.values()) != 88:
        raise RuntimeError("oracle structural-audit case labels do not reproduce 88/115")
    final = summary.get("summary", {})
    if final.get("attempted") != 115 or final.get("structural_recoveries") != 88:
        raise RuntimeError("oracle final summary must report 88/115")
    per_case = summary.get("per_case", {})
    if len(per_case) != 115:
        raise RuntimeError("oracle final summary must contain all 115 cases")
    recovered = sum(
        row.get("structural_recovery") is True
        for row in per_case.values()
        if isinstance(row, dict)
    )
    if recovered != 88:
        raise RuntimeError("oracle final-summary case labels do not reproduce 88/115")


def _write_checksums(root: Path) -> int:
    lines, n = [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root)}")
        n += 1
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capsules", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("NestyNet_paper3_artifacts"))
    ap.add_argument("--archive", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    capsules = args.capsules.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    archive = (args.archive or out.with_suffix(".tgz")).expanduser().resolve()

    if out.exists() and not args.force:
        raise RuntimeError(f"output directory exists (use --force): {out}")
    if archive.exists() and not args.force:
        raise RuntimeError(f"archive exists (use --force): {archive}")

    _validate(capsules)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    replacements = _normalizers(
        (capsules, "$PAPER3_CAPSULES"),
        (REPO_ROOT, "$NESTYNET_SR"),
        # The campaigns record dataset paths under the local workspace root;
        # collapse it before the generic home, which would leave the rest of
        # the private path behind.
        (capsules.parent, "$WORKSPACE"),
        (Path.home(), "$HOME"),
    )

    inventory: dict = {"cells": {}, "oracle": {}, "reproducibility": []}

    for cell_dir in sorted(d for d in capsules.iterdir() if d.is_dir()):
        n = 0
        for src in sorted(cell_dir.rglob("*")):
            if src.is_file():
                _copy_file(src, out / "capsules" / cell_dir.name / src.relative_to(cell_dir),
                           replacements)
                n += 1
        inventory["cells"][cell_dir.name] = n
        print(f"  capsule {cell_dir.name}: {n} files")

    oracle_source = REPO_ROOT / "reproducibility" / "paper3"
    audit = oracle_source / "fss_structural_audit.json"
    summary = oracle_source / "fss_oracle_final_summary.json"
    _validate_oracle_evidence(audit, summary)
    for src, name in (
        (audit, "structural_audit.json"),
        (summary, "oracle_summary.json"),
    ):
        _copy_file(src, out / "oracle" / name, replacements)
        inventory["oracle"][name] = True

    for name in ("expected_results.json", "README.md", "PROVENANCE_NOTES.md"):
        src = REPO_ROOT / "reproducibility" / "paper3" / name
        if src.is_file():
            _copy_file(src, out / "reproducibility" / name, replacements)
            inventory["reproducibility"].append(name)

    readme = REPO_ROOT / "reproducibility" / "paper3" / "README.md"
    _copy_file(readme, out / "README.md", replacements)
    full_guide = REPO_ROOT / "PAPER3_REPRODUCIBILITY.md"
    _copy_file(
        full_guide,
        out / "reproducibility" / "PAPER3_REPRODUCIBILITY.md",
        replacements,
    )
    inventory["reproducibility"].append("PAPER3_REPRODUCIBILITY.md")

    manifest = {
        "schema_version": 1,
        "title": "NestyNet Paper III benchmark evidence",
        "built_at_utc": _utc_now(),
        "source": {"path": "$NESTYNET_SR", **_git_state(REPO_ROOT)},
        "contents": inventory,
        "notes": [
            "Capsules hold per-problem reports, the summary CSV, a campaign "
            "manifest recording the run configuration read from the runs "
            "themselves, and the structural audit for each benchmark cell.",
            "Benchmark input data is published separately and is not "
            "duplicated here.",
            "Models, checkpoints and logs are excluded by design.",
            "The authoritative oracle recovery count is 88/115 in "
            "oracle/structural_audit.json and oracle/oracle_summary.json.",
            "Jacobi and SPARC vignette materials remain in the versioned code "
            "repository; this archive is intentionally benchmark-scoped.",
        ],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    _scan(out)
    n_files = _write_checksums(out)

    with tarfile.open(archive, "w:gz") as tf:
        tf.add(out, arcname=out.name)
    size = archive.stat().st_size
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = {"doi": None, "url": None, "bytes": size, "sha256": digest,
              "archive": archive.name, "built_at_utc": manifest["built_at_utc"]}
    Path(str(archive) + ".record.json").write_text(json.dumps(record, indent=2) + "\n")

    print(f"\nStaged artifact: {out}")
    print(f"Archive: {archive}")
    print(f"Files: {n_files}   Bytes: {size}   SHA-256: {digest}")
    print(f"Record template: {archive}.record.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
