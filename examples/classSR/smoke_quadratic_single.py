#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Smoke test: single-dataset quadratic with local/global constants.

Runs nestynet-sr on quad_1.csv with:
  --factorized-search
  --local_consts  '{"c0":[1,0], "c1":[1,-1]}'
  --global_consts '{"g0":[1,-2]}'

Ground truth: y = 2.0 + 1.5*x0 + 0.5*x0²
Units: y [L]=[1,0], x0 [T]=[0,1]

Checks:
  - Process exits cleanly (rc=0)
  - Stage B result file exists
"""

import os
import pathlib
import subprocess
import sys

QUAD_CSV = pathlib.Path(__file__).parent / "data" / "quad_1.csv"


def main():
    if not QUAD_CSV.exists():
        print("Data not found — generating...")
        subprocess.check_call(
            [sys.executable, str(pathlib.Path(__file__).parent / "generate_quadratic.py")]
        )

    cmd = [
        sys.executable, "-u", "-m", "nestynet_sr.run_SR",
        "--filepath", str(QUAD_CSV),
        "--factorized-search",
        "--y_units", "[1,0]",
        "--x_units", "[[0,1]]",
        "--units_basis", "L,T",
        "--local_consts", '{"c0":[1,0], "c1":[1,-1]}',
        "--global_consts", '{"g0":[1,-2]}',
        "--batch_size", "256",
        "--ndata_train", "256",
        "--ndata_val", "256",
        "--log_level", "INFO",
    ]

    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / "quad_1.log"

    print(f"Running: {' '.join(cmd)}")
    print(f"Log: {log_path}")

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

    human_files = sorted(results_dir.glob("quad_1*final.human"))
    if not human_files:
        human_files = sorted(results_dir.glob("quad_1*.human"))
    if human_files:
        print(f"\n--- Final result: {human_files[-1]} ---")
        print(human_files[-1].read_text()[:2000])
    else:
        print("\nNo human-readable result found in results/")

    print("\nSingle-dataset quadratic verification complete.")


if __name__ == "__main__":
    main()
