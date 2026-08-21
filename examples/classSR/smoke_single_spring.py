#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Verify that nestynet-sr can discover the damped-spring formula from a single dataset.

Runs: nestynet-sr --filepath examples/classSR/data/spring_1.csv --factorized-search
Checks that Stage B discovers a cos * exp pattern.
"""

import os
import pathlib
import subprocess
import sys

SPRING_CSV = pathlib.Path(__file__).parent / "data" / "spring_1.csv"


def main():
    if not SPRING_CSV.exists():
        print("Data not found — generating...")
        subprocess.check_call(
            [sys.executable, str(pathlib.Path(__file__).parent / "generate_damped_springs.py")]
        )

    cmd = [
        sys.executable, "-u", "-m", "nestynet_sr.run_SR",
        "--filepath", str(SPRING_CSV),
        "--factorized-search",
        "--log_level", "INFO",
    ]

    # Log file alongside results
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f"{SPRING_CSV.stem}.log"

    print(f"Running: {' '.join(cmd)}")
    print(f"Log: {log_path}")

    # Stream output to both console and log file
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="")
        process.wait()

    if process.returncode != 0:
        print(f"\nCommand exited with code {process.returncode}")
        print(f"  Full log: {log_path}")
        sys.exit(1)
    human_files = sorted(results_dir.glob("spring_1*stageB*human")) if results_dir.exists() else []
    if human_files:
        print(f"\n--- Stage B result: {human_files[-1]} ---")
        print(human_files[-1].read_text()[:2000])
    else:
        print("\nNo Stage B human-readable result found in results/")

    print("\nSingle-spring verification complete.")


if __name__ == "__main__":
    main()
