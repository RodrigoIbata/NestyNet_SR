# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Validate Paper III benchmark capsules against the frozen expectations.

For every FROZEN cell in ``reproducibility/paper3/expected_results.json``
the gate checks what is checkable without any recomputation:

- the capsule exists and is complete (manifest, summary.csv, all reports),
  allowing for problems the cell declares under ``not_completed`` (runs that
  never returned within the campaign's wall-clock budget; they are absent
  from the capsule by construction and count as non-solves, but are recorded
  separately from problems that ran and failed),
- the summary's exact-recovery count equals the frozen ``exact``,
- when a structural audit is present, its per-problem verdicts are
  internally consistent and its structural-wins total equals the frozen
  ``structural_wins``.

Pending cells are reported but never fail the gate, so it is useful while
the campaign is still filling.  The gate makes no cross-revision or
cross-platform reproduction claim: the frozen numbers are the measured
aggregates of the recorded runs (see expected_results.json comment).

    python3 scripts/validate_paper3_results.py --capsules /path/to/capsules
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = REPO_ROOT / "reproducibility" / "paper3" / "expected_results.json"


def _exact_count(summary_csv: Path) -> tuple[int, int]:
    n = exact = 0
    for row in csv.DictReader(summary_csv.open()):
        if not re.match(r"pb\d+", str(row.get("problem", ""))):
            continue
        n += 1
        if str(row.get("truth_exact", "")).strip().lower() in ("yes", "true", "1"):
            exact += 1
    return exact, n


def validate_cell(name: str, wanted: dict, capsule: Path, n_problems: int) -> list[str]:
    errors: list[str] = []
    if not capsule.is_dir():
        return [f"{name}: capsule missing at {capsule}"]

    not_completed = [str(pid) for pid in wanted.get("not_completed", [])]
    n_expected = n_problems - len(not_completed)

    manifest_path = capsule / "campaign_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"{name}: no campaign_manifest.json")
    else:
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("n_reports", -1)) != n_expected:
            errors.append(
                f"{name}: manifest reports {manifest.get('n_reports')} runs, expected {n_expected}"
                + (f" ({n_problems} minus {len(not_completed)} not completed)" if not_completed else "")
            )
        if manifest.get("unreadable_reports"):
            errors.append(f"{name}: unreadable reports {manifest['unreadable_reports']}")

    report_files = sorted((capsule / "reports").glob("pb*.report.json"))
    if len(report_files) != n_expected:
        errors.append(f"{name}: {len(report_files)} report files, expected {n_expected}")
    present_ids = {re.match(r"pb(\d+)", f.name).group(1) for f in report_files}
    resurfaced = sorted(pid for pid in not_completed if pid.removeprefix("pb").lstrip("0").rjust(3, "0") in present_ids
                        or pid in present_ids)
    if resurfaced:
        errors.append(
            f"{name}: declared not_completed but present in the capsule: {resurfaced}"
        )

    summary_csv = capsule / "summary.csv"
    if not summary_csv.is_file():
        errors.append(f"{name}: no summary.csv")
        return errors
    exact, n_rows = _exact_count(summary_csv)
    if n_rows != n_expected:
        errors.append(f"{name}: summary covers {n_rows} problems, expected {n_expected}")
    if wanted.get("exact") is not None and exact != int(wanted["exact"]):
        errors.append(f"{name}: exact count {exact} != frozen {wanted['exact']}")

    audit_path = capsule / "structural_audit.json"
    if wanted.get("structural_wins") is not None:
        if not audit_path.is_file():
            errors.append(f"{name}: structural_wins frozen but no structural_audit.json")
        else:
            audit = json.loads(audit_path.read_text())
            counts = audit.get("counts", {})
            total = sum(int(v) for v in counts.values())
            if total != n_expected:
                errors.append(f"{name}: audit covers {total} problems, expected {n_expected}")
            if int(audit.get("exact", -1)) != exact:
                errors.append(
                    f"{name}: audit exact {audit.get('exact')} != summary exact {exact}"
                )
            wins = int(audit.get("structural_wins_total", -1))
            if wins != int(wanted["structural_wins"]):
                errors.append(
                    f"{name}: structural wins {wins} != frozen {wanted['structural_wins']}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsules", type=Path, required=True)
    parser.add_argument("--expected", type=Path, default=EXPECTED_PATH)
    args = parser.parse_args()

    expected = json.loads(args.expected.expanduser().resolve().read_text())
    n_problems = int(expected["n_problems"])
    capsules_root = args.capsules.expanduser().resolve()

    errors: list[str] = []
    for name, wanted in expected["cells"].items():
        status = wanted.get("status")
        if status not in ("frozen", "reference"):
            present = (capsules_root / name).is_dir()
            print(f"[pending] {name}: capsule {'present' if present else 'absent'}")
            continue
        capsule = capsules_root / name
        if status == "reference" and not capsule.is_dir():
            print(f"[ref]     {name}: capsule absent (superseded arm; not required)")
            continue
        cell_errors = validate_cell(name, wanted, capsule, n_problems)
        tag = "[ok]     " if status == "frozen" else "[ref ok] "
        if cell_errors:
            errors.extend(cell_errors)
            print(f"[FAIL]    {name}")
        else:
            note = ""
            if wanted.get("not_completed"):
                note = (f"  [{len(wanted['not_completed'])} not completed: "
                        f"{', '.join(str(x) for x in wanted['not_completed'])}]")
            print(f"{tag} {name}: exact {wanted['exact']}, "
                  f"structural wins {wanted['structural_wins']} of {n_problems}{note}")

    if errors:
        print("\nvalidation errors:\n  " + "\n  ".join(errors))
        return 1
    print("\nall frozen cells validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(sys.exit(main()))
