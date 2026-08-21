#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
from pathlib import Path

from kepler_demo_utils import (
    DEFAULT_CADENCE_DAYS,
    DEFAULT_PROFILE,
    DEFAULT_PROVIDER,
    DEFAULT_RAW_MANIFEST_PATH,
    DEFAULT_SOLAR_MU_AU_DAY,
    DEFAULT_START_DATE,
    DEFAULT_YEARS,
    build_generation_provenance,
    build_default_kepler_datasets,
    write_generated_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reduced-Kepler datasets from the HORIZONS-backed real-data scaffold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mu", type=float, default=float(DEFAULT_SOLAR_MU_AU_DAY), help="Shared gravitational parameter in AU^3/day^2")
    parser.add_argument("--provider", choices=("astropy_builtin", "raw_csv"), default=DEFAULT_PROVIDER, help="Ephemeris source")
    parser.add_argument("--profile", choices=("clean", "weathered"), default=DEFAULT_PROFILE, help="Use exact two-body propagation from real initial states or the raw ephemeris trajectories")
    parser.add_argument("--start_date", type=str, default=DEFAULT_START_DATE, help="Ephemeris start date")
    parser.add_argument("--years", type=float, default=float(DEFAULT_YEARS), help="Span of the generated trajectory window")
    parser.add_argument("--cadence_days", type=float, default=float(DEFAULT_CADENCE_DAYS), help="Time step in days")
    parser.add_argument("--raw_manifest", type=str, default=str(DEFAULT_RAW_MANIFEST_PATH), help="JSON manifest for normalized external heliocentric state CSVs")
    parser.add_argument("--seed", type=int, default=123, help="Legacy no-op kept for interface compatibility")
    parser.add_argument("--train_samples", type=int, default=1024, help="Legacy no-op kept for interface compatibility")
    parser.add_argument("--validation_samples", type=int, default=1024, help="Legacy no-op kept for interface compatibility")
    parser.add_argument("--holdout_samples", type=int, default=2048, help="Legacy no-op kept for interface compatibility")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Example root where the data/ folder will be written",
    )
    args = parser.parse_args()

    datasets = build_default_kepler_datasets(
        mu=float(args.mu),
        seed=int(args.seed),
        train_samples=int(args.train_samples),
        validation_samples=int(args.validation_samples),
        holdout_samples=int(args.holdout_samples),
        provider=str(args.provider),
        profile=str(args.profile),
        start_date=str(args.start_date),
        years=float(args.years),
        cadence_days=float(args.cadence_days),
        raw_manifest=args.raw_manifest,
    )
    provenance = build_generation_provenance(
        provider=str(args.provider),
        profile=str(args.profile),
        mu=float(args.mu),
        start_date=str(args.start_date),
        years=float(args.years),
        cadence_days=float(args.cadence_days),
        raw_manifest=args.raw_manifest,
    )
    result = write_generated_artifacts(
        Path(args.output_root),
        datasets,
        generation_provenance=provenance,
    )
    print(f"Wrote {result['n_orbits']} ephemeris orbit datasets")
    print(f"Profile: {args.profile}")
    print(f"Provider: {args.provider}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Metadata: {result['metadata_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
