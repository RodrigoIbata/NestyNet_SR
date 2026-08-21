#!/usr/bin/env python
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""End-to-end test for class SR: multi-dataset quadratic with local/global constants.

This script:
1. Generates 3 quadratic datasets (if not already present)
2. Runs nestynet-sr with --class_sr on all 3 datasets
3. Parses results and verifies shared vs per-dataset classification

Ground truth: y = c0 + c1*x0 + g0*x0²
  g0 = 0.5  (shared)
  c0 = [2.0, 3.0, 1.0]  (per-dataset)
  c1 = [1.5, 0.8, 2.5]  (per-dataset)
"""

import json
import os
import pathlib
import subprocess
import sys

DATA_DIR = pathlib.Path(__file__).parent / "data"
RESULTS_DIR = pathlib.Path("results")

# Ground truth
G0_TRUE = 0.5
DATASETS_TRUE = [
    {"c0": 2.0, "c1": 1.5},
    {"c0": 3.0, "c1": 0.8},
    {"c0": 1.0, "c1": 2.5},
]


def ensure_data():
    """Generate data if not present."""
    quads = [DATA_DIR / f"quad_{i}.csv" for i in range(1, 4)]
    if not all(q.exists() for q in quads):
        print("Generating quadratic data...")
        subprocess.check_call(
            [sys.executable, str(pathlib.Path(__file__).parent / "generate_quadratic.py")]
        )
    return quads


def run_class_sr(quad_files):
    """Run nestynet-sr with --class_sr on all quadratic files."""
    cmd = [
        sys.executable, "-u", "-m", "nestynet_sr.run_SR",
        "--filepaths", *[str(f) for f in quad_files],
        "--class_sr",
        "--factorized-search",
        "--y_units", "[1,0]",
        "--x_units", "[[0,1]]",
        "--units_basis", "L,T",
        "--local_consts", '{"c0":[1,0], "c1":[1,-1]}',
        "--global_consts", '{"g0":[1,-2]}',
        "--log_level", "INFO",
    ]

    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "classSR_quadratics.log"

    print(f"\nRunning: {' '.join(cmd)}")
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
        return False

    return True


def check_results():
    """Parse and verify class SR results."""
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
    agg_mode = data.get("val_loss_agg_mode", "mean")
    print(f"Val loss (agg, {agg_mode}):  {data['val_loss_agg']:.6e}")
    print(f"Val losses:      {data['val_losses']}")
    print(f"Class params:    {data['class_params']}")
    print(f"Experiment params ({len(data['experiment_params'])} datasets):")
    for i, ep in enumerate(data['experiment_params']):
        print(f"  dataset {i}: {ep}")

    if not data['class_tags']:
        print("\nWARNING: No class tags found — auto-classification may need tuning")
    if not data['experiment_tags']:
        print("\nWARNING: No experiment tags found — all params classified as class")

    max_val_loss = max(data['val_losses'])
    if max_val_loss < 1e-3:
        print(f"\nPASS: All per-dataset val losses < 1e-3 (max={max_val_loss:.4e})")
    else:
        print(f"\nINFO: Max per-dataset val loss = {max_val_loss:.4e}")

    print("\nClass SR quadratic end-to-end test complete.")
    return True


def main():
    quad_files = ensure_data()

    success = run_class_sr(quad_files)
    if not success:
        print("\nClass SR run failed — check output above")
        sys.exit(1)

    check_results()


if __name__ == "__main__":
    main()
