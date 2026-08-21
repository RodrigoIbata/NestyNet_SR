#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Parallel, resumable precompute of the surrogate accelerations per body.

Fits the cylinder-chart surrogates (velocity channels + certificate position
channels) for every body in a raw-states manifest and stores the analytic
accelerations in the content-addressed cache consumed by
``smoke_kepler_discovery.py --accel_source surrogate --accel_cache_dir ...``.
Bodies whose cache entry matches the current input data are skipped, so the
run can be interrupted and relaunched freely.

Production example (the paper-4 308-body ensemble):

    python3 precompute_kepler_surrogate_accels.py \
        --raw_manifest data/raw_states_manifest_jpl_ssodnet_mass_gt_1e17_arc15000_1d.json \
        --cache_dir data/surrogate_accels_1d --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Spawn-mode workers re-execute this module; make kepler_demo_utils importable
# in both the parent and every child.
_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))


def _worker(payload: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", str(payload["threads_per_worker"]))
    import kepler_demo_utils as kdu

    spec = kdu.EphemerisBodySpec(**payload["spec"])
    started = time.time()
    dataset = kdu.generate_kepler_dataset(
        spec,
        mu=payload["mu"],
        profile="weathered",
        accel_source="surrogate",
        accel_certificate=bool(payload["certificate"]),
        accel_harmonic=bool(payload["harmonic"]),
        accel_cache_dir=payload["cache_dir"],
    )
    provenance = dataset.accel_provenance or {}
    cache = provenance.get("accel_cache", {})
    channels = provenance.get("channels", {})
    certificate = provenance.get("certificate", {})
    return {
        "orbit_id": spec.orbit_id,
        "seconds": time.time() - started,
        "cache_hit": bool(cache.get("hit", False)),
        "val_rel_rmse": {k: v.get("val_rel_rmse") for k, v in channels.items()},
        "fd_rel_diff": provenance.get("fd_rel_diff", {}),
        "certificate_rel_rmse": certificate.get("measured_derivative_rel_rmse", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw_manifest", type=str, required=True, help="Raw-states manifest JSON")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory to fill")
    parser.add_argument("--workers", type=int, default=4, help="Parallel body workers")
    parser.add_argument("--threads_per_worker", type=int, default=2, help="BLAS threads per worker")
    parser.add_argument("--mu", type=float, default=None, help="Gravitational parameter override")
    parser.add_argument("--accel_harmonic", action="store_true", help="Add the 2-omega circle")
    parser.add_argument(
        "--no_certificate",
        action="store_true",
        help="Skip the position-channel certificate fits (halves the cost)",
    )
    parser.add_argument("--only", type=str, default=None, help="Comma-separated orbit_id subset")
    args = parser.parse_args()

    import kepler_demo_utils as kdu

    specs = kdu.build_default_orbit_specs(provider="raw_csv", raw_manifest=args.raw_manifest)
    if args.only:
        wanted = {token.strip() for token in args.only.split(",") if token.strip()}
        specs = [spec for spec in specs if spec.orbit_id in wanted]
    if not specs:
        raise SystemExit("no bodies selected")

    payloads = [
        {
            "spec": {
                "orbit_id": spec.orbit_id,
                "body_name": spec.body_name,
                "split": spec.split,
                "csv_path": spec.csv_path,
            },
            "mu": args.mu,
            "certificate": not args.no_certificate,
            "harmonic": bool(args.accel_harmonic),
            "cache_dir": str(Path(args.cache_dir)),
            "threads_per_worker": int(args.threads_per_worker),
        }
        for spec in specs
    ]

    print(f"{len(payloads)} bodies -> {args.cache_dir} with {args.workers} workers", flush=True)
    started = time.time()
    reports = []
    failures = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(_worker, payload): payload for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            orbit_id = futures[future]["spec"]["orbit_id"]
            try:
                report = future.result()
            except Exception as exc:
                failures.append({"orbit_id": orbit_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(payloads)}] {orbit_id}: FAILED {exc}", flush=True)
                continue
            reports.append(report)
            values = report["val_rel_rmse"]
            certificate = report["certificate_rel_rmse"]
            state = "cache" if report["cache_hit"] else f"{report['seconds']:.0f}s"
            print(
                f"[{index}/{len(payloads)}] {report['orbit_id']}: {state} "
                f"val vx:{values.get('vx', float('nan')):.2e} vy:{values.get('vy', float('nan')):.2e}"
                + (
                    f" cert x:{certificate.get('x', float('nan')):.2e} y:{certificate.get('y', float('nan')):.2e}"
                    if certificate
                    else ""
                ),
                flush=True,
            )

    summary_path = Path(args.cache_dir) / "precompute_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "raw_manifest": str(Path(args.raw_manifest).resolve()),
                "n_bodies": len(payloads),
                "n_completed": len(reports),
                "failures": failures,
                "wall_seconds": time.time() - started,
                "harmonic": bool(args.accel_harmonic),
                "certificate": not args.no_certificate,
                "reports": sorted(reports, key=lambda row: row["orbit_id"]),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wall {time.time() - started:.0f}s; summary -> {summary_path}", flush=True)
    if failures:
        print(f"{len(failures)} FAILURES; rerun to retry (completed bodies are cached)", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
