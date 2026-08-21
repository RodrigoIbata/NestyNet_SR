# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Emit the surrogate-accel cache key for every manifest body (no fitting).

The key (schema xyz_v2) hashes the pre-projection 3D state series plus the
configuration flags.  Running this on two machines and diffing the output
proves cache portability BEFORE any compute is spent: identical lists mean
every entry fitted on one machine will be a verified cache hit on the other.

    python3 emit_accel_cache_keys.py --raw_manifest <manifest.json> > keys.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kepler_demo_utils as kdu  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_manifest", type=str, required=True)
    parser.add_argument("--no_certificate", action="store_true")
    parser.add_argument("--accel_harmonic", action="store_true")
    args = parser.parse_args()

    specs = kdu.build_default_orbit_specs(
        provider="raw_csv", raw_manifest=args.raw_manifest
    )
    for spec in specs:
        t_days, positions_xyz, velocities_xyz = kdu._load_normalized_state_csv(spec.csv_path)
        sha = kdu._surrogate_accel_input_sha(
            t_days,
            positions_xyz,
            velocities_xyz,
            certificate=not args.no_certificate,
            harmonic=bool(args.accel_harmonic),
        )
        print(f"{sha}  {spec.orbit_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
