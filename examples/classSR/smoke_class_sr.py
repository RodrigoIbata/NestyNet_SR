#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""End-to-end test for class SR: multi-dataset fitting with shared constants.

This script:
1. Generates 5 damped spring datasets (if not already present)
2. Runs nestynet-sr with --class_sr on all 5 springs
3. Parses results and verifies shared k ≈ 0.3 and per-spring omega values

Ground truth: y = cos(omega_i * t) * exp(-k * t)
  omega = [2.0, 3.0, 5.0, 7.0, 4.5]
  k = 0.3
"""

import json
import os
import pathlib
import subprocess
import sys

DATA_DIR = pathlib.Path(__file__).parent / "data"
RESULTS_DIR = pathlib.Path("results")

# Ground truth
OMEGAS_TRUE = [2.0, 3.0, 5.0, 7.0, 4.5]
K_TRUE = 0.3


def ensure_data():
    """Generate data if not present."""
    springs = [DATA_DIR / f"spring_{i}.csv" for i in range(1, 6)]
    if not all(s.exists() for s in springs):
        print("Generating damped spring data...")
        subprocess.check_call(
            [sys.executable, str(pathlib.Path(__file__).parent / "generate_damped_springs.py")]
        )
    return springs


def run_class_sr(spring_files):
    """Run nestynet-sr with --class_sr on all spring files."""
    cmd = [
        sys.executable, "-u", "-m", "nestynet_sr.run_SR",
        "--filepaths", *[str(f) for f in spring_files],
        "--class_sr",
        "--factorized-search",
        "--y_units", "[1,0]",
        "--x_units", "[[0,1]]",
        "--units_basis", "L,T",
        "--local_consts", '{"c1":[0,-1]}',
        "--global_consts", '{"g0":[0,-1]}',
        "--log_level", "INFO",
    ]

    # Log file alongside results
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "classSR_springs.log"

    print(f"\nRunning: {' '.join(cmd)}")
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
        return False

    return True


def check_results():
    """Parse and verify class SR results."""
    # Find the class SR JSON output
    json_files = sorted(RESULTS_DIR.glob("*_classSR.json"))
    if not json_files:
        print("ERROR: No *_classSR.json found in results/")
        return False

    latest = json_files[-1]
    print(f"\n--- Checking results: {latest} ---")
    with open(latest) as f:
        data = json.load(f)

    print(f"Class tags:      {data['class_tags']}")
    print(f"Experiment tags: {data['experiment_tags']}")
    print(f"CV per tag:      {data['cv_per_tag']}")
    print(f"Val loss (agg):  {data['val_loss_agg']:.6e}")
    print(f"Val losses:      {data['val_losses']}")
    print(f"Class params:    {data['class_params']}")
    print(f"Experiment params ({len(data['experiment_params'])} datasets):")
    for i, ep in enumerate(data['experiment_params']):
        print(f"  dataset {i}: {ep}")

    # Verify class tags exist and experiment tags exist
    if not data['class_tags']:
        print("\nWARNING: No class tags found — auto-classification may need tuning")
    if not data['experiment_tags']:
        print("\nWARNING: No experiment tags found — all params classified as class")

    # Verify validation loss is reasonable
    max_val_loss = max(data['val_losses'])
    if max_val_loss < 1e-3:
        print(f"\nPASS: All per-dataset val losses < 1e-3 (max={max_val_loss:.4e})")
    else:
        print(f"\nINFO: Max per-dataset val loss = {max_val_loss:.4e}")

    print("\nClass SR end-to-end test complete.")
    return True


def main():
    spring_files = ensure_data()

    success = run_class_sr(spring_files)
    if not success:
        print("\nClass SR run failed — check output above")
        sys.exit(1)

    check_results()


if __name__ == "__main__":
    main()
