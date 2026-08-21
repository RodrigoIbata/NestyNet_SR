#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Run the 16 failing problems in parallel batches with timeout protection."""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FAILING_IDS = [
    "113", "114", "115", "117", "118", "126", "127",
]
TIMEOUT_SECONDS = 10800  # 3 hours
MAX_WORKERS = 5


def run_one(pid: str, fast: bool = True) -> dict:
    """Run a single problem and return status info."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "examples" / "feynman_de" / "run_benchmark.py"),
        "--only", pid,
        "--engine", "factorized_search_oracle",
        "--skip_generate",
        "--results_dir", str(REPO_ROOT / "results" / "feynman_de_patch3"),
    ]
    if fast:
        cmd.append("--fast")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(REPO_ROOT / "examples" / "feynman_de"),
        )
        elapsed = time.time() - t0
        # Extract the status line from output
        status_line = ""
        for line in proc.stdout.splitlines():
            if "[OK]" in line or "[XX]" in line or "[~~]" in line or "[!!]" in line or "[??]" in line:
                status_line = line.strip()
                break
        # Also grab discovered equation
        discovered = ""
        for line in proc.stdout.splitlines():
            if "Discovered:" in line:
                discovered = line.strip()
                break
        return {
            "id": pid,
            "returncode": proc.returncode,
            "status_line": status_line,
            "discovered": discovered,
            "elapsed": elapsed,
            "timeout": False,
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {
            "id": pid,
            "returncode": -1,
            "status_line": "TIMEOUT",
            "discovered": "",
            "elapsed": elapsed,
            "timeout": True,
            "stderr_tail": "",
        }


def main():
    fast = "--full" not in sys.argv
    mode = "FAST" if fast else "FULL"
    print(f"Running {len(FAILING_IDS)} failing problems in {mode} mode")
    print(f"  Max workers: {MAX_WORKERS}, Timeout: {TIMEOUT_SECONDS}s per problem")
    print("  Results dir: results/feynman_de_patch3/")
    print()

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, pid, fast): pid for pid in FAILING_IDS}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                res = future.result()
            except Exception as exc:
                res = {
                    "id": pid,
                    "returncode": -2,
                    "status_line": f"EXCEPTION: {exc}",
                    "discovered": "",
                    "elapsed": 0,
                    "timeout": False,
                    "stderr_tail": "",
                }
            results.append(res)
            tag = "TIMEOUT" if res["timeout"] else ("OK" if res["returncode"] == 0 else "FAIL")
            print(f"  de{res['id']}: {tag} ({res['elapsed']:.0f}s) {res['status_line']}", flush=True)

    # Print summary sorted by ID
    print("\n" + "=" * 80)
    print(f"PATCH TEST SUMMARY ({mode} mode)")
    print("=" * 80)
    results.sort(key=lambda r: r["id"])
    n_pass = n_fail = n_timeout = 0
    for r in results:
        if r["timeout"]:
            status = "TIMEOUT"
            n_timeout += 1
        elif "[OK]" in r["status_line"]:
            status = "PASS"
            n_pass += 1
        else:
            status = "FAIL"
            n_fail += 1
        print(f"  de{r['id']}  {status:<10s}  {r['elapsed']:6.0f}s  {r['status_line'][:70]}")
        if r["discovered"]:
            print(f"         {r['discovered'][:80]}")

    print("-" * 80)
    print(f"PASS: {n_pass}  FAIL: {n_fail}  TIMEOUT: {n_timeout}  (of {len(results)})")
    print("=" * 80)


if __name__ == "__main__":
    main()
