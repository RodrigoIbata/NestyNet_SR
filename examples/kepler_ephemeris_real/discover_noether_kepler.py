#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Noether reduction of the Kepler problem from real asteroid ephemerides.

The §VII.D capstone recovers the reduced Kepler hierarchy but assembles the
Hamiltonian from separately-fitted ingredients, noting that the centrifugal
coefficient relation ``k_d = ell_d^2`` is an empirical cross-orbit fit rather
than a derived identity.  This closes that hedge with a Noether reduction.

From the Cartesian phase-space trajectories ``(x,y,z, vx,vy,vz)`` alone:

1. Discover the symmetry -- scan candidate phase-space generators (rotations,
   translations, dilation), form each one's Noether momentum map, and keep
   those whose charge is conserved along the data.  The three rotations survive
   (SO(3): angular momentum ``L`` conserved); translations and dilation do not.
2. Reduce by the discovered rotational symmetry.  ``L = r x p`` fixes the orbit
   plane and ``ell = |L|`` is the areal constant ``r^2 theta_dot``; eliminating
   the cyclic angle gives ``r_ddot = ell^2/r^3 - mu/r^2``, so the centrifugal
   coefficient ``k = ell^2`` is *derived* -- one discovered symmetry supplies
   both the areal law and the centrifugal term.

Nothing about "angle" is presupposed: the rotation is discovered from Cartesian
data as the symmetry whose Noether charge the ephemerides conserve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from discover_third_body_residuals import (
    DEFAULT_BULK_RAW_MANIFEST,
    load_state_series_from_manifest,
)
from kepler_demo_utils import DEFAULT_SOLAR_MU_AU_DAY, _jsonable, cached_surrogate_rddot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nestynet_sr.sr_gs.noether_reduction import (  # noqa: E402
    discover_noether_symmetries,
    noether_kepler_reduction,
)


def _phase_space_trajectories(series, *, stride: int, edge_trim: int):
    trajs, times = [], []
    for item in series:
        q = np.asarray(item.position_au, dtype=np.float64)
        v = np.asarray(item.velocity_au_per_d, dtype=np.float64)
        t = np.asarray(item.t_day, dtype=np.float64)
        sl = slice(edge_trim, q.shape[0] - edge_trim, stride)
        Z = np.column_stack([q[sl], v[sl]])
        trajs.append(Z)
        times.append(t[sl])
    return trajs, times


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Noether reduction of Kepler from real asteroid phase-space data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_manifest", type=str, default=str(DEFAULT_BULK_RAW_MANIFEST))
    parser.add_argument("--max_bodies", type=int, default=0, help="0 = all")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--edge_trim", type=int, default=4)
    parser.add_argument("--conservation_tol", type=float, default=1.0e-2,
                        help="admit a charge as conserved below this relative drift; the real "
                        "ephemerides are planet-perturbed so angular momentum drifts at ~1e-3 "
                        "(vs ~50 for non-symmetries), 4 orders of magnitude of separation")
    parser.add_argument("--results_dir", type=str, default=str(Path("results") / "noether_kepler"))
    parser.add_argument(
        "--accel_cache_dir",
        type=str,
        default=None,
        help="Precomputed surrogate-acceleration cache (precompute_kepler_surrogate_accels.py). "
        "When set, the reduction's radial acceleration is exact algebra on the cached analytic "
        "(ax, ay) instead of finite-differencing rdot; every requested body must be cached.",
    )
    parser.add_argument("--accel_harmonic", action="store_true",
                        help="Cache-key flag: the cache was built with the 2-omega harmonic")
    parser.add_argument("--accel_no_certificate", action="store_true",
                        help="Cache-key flag: the cache was built without certificate fits")
    args = parser.parse_args(argv)

    series = load_state_series_from_manifest(
        args.raw_manifest, max_bodies=None if int(args.max_bodies) <= 0 else int(args.max_bodies)
    )
    trajs, times = _phase_space_trajectories(series, stride=int(args.stride), edge_trim=int(args.edge_trim))
    print(f"Loaded {len(trajs)} bodies ({trajs[0].shape[0]} phase-space samples each after stride)")

    rddot_series = None
    if args.accel_cache_dir is not None:
        rddot_series = []
        for item in series:
            rddot_full = cached_surrogate_rddot(
                args.accel_cache_dir,
                item.orbit_id,
                t_days=np.asarray(item.t_day, dtype=np.float64),
                positions_xyz=np.asarray(item.position_au, dtype=np.float64),
                velocities_xyz=np.asarray(item.velocity_au_per_d, dtype=np.float64),
                certificate=not bool(args.accel_no_certificate),
                harmonic=bool(args.accel_harmonic),
            )
            if rddot_full is None:
                raise SystemExit(
                    f"no valid surrogate-accel cache entry for {item.orbit_id!r} in "
                    f"{args.accel_cache_dir}; run precompute_kepler_surrogate_accels.py first"
                )
            sl = slice(int(args.edge_trim), rddot_full.shape[0] - int(args.edge_trim), int(args.stride))
            rddot_series.append(rddot_full[sl])
        print(f"Radial accelerations: exact algebra on cached analytic (ax, ay) for {len(rddot_series)} bodies")

    disc = discover_noether_symmetries(trajs, n=3, conservation_tol=float(args.conservation_tol))
    print("\n== Symmetry discovery (Noether charge conservation along the data) ==")
    for r in disc["all"]:
        tag = "ADMITTED" if r.conserved else "rejected"
        print(f"  {r.name:>16s} [{r.family:>11s}]  rel-drift {r.conservation_rel_drift:.2e}"
              f"  scale-drift {r.conservation_scale_drift:.2e}  -> {tag}")
    rotations = [r for r in disc["admitted"] if r.family == "rotation"]
    print(f"\n  => SO(3) rotational symmetry {'FULLY' if len(rotations) >= 3 else 'partially' if rotations else 'NOT'} "
          f"admitted ({len(rotations)}/3 rotations); angular momentum conserved.")

    result = noether_kepler_reduction(
        trajs,
        times=times,
        conservation_tol=float(args.conservation_tol),
        rddot_series=rddot_series,
    )
    red = result.get("reduction")
    summary: dict[str, Any] = {
        "n_bodies": len(trajs),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "solar_mu_reference": float(DEFAULT_SOLAR_MU_AU_DAY),
        "discovery": {
            "admitted": [(r.name, r.family, r.conservation_rel_drift) for r in disc["admitted"]],
            "rejected": [(r.name, r.family, r.conservation_rel_drift) for r in disc["rejected"]],
            "scale_drift": {r.name: r.conservation_scale_drift for r in disc["all"]},
        },
        "so3_fully_admitted": bool(len(rotations) >= 3),
    }
    if red:
        med = red["k_over_ell_squared_median"]
        maxdev = red["k_over_ell_squared_max_abs_dev"]
        print("\n== Reduction by the rotational Noether charge ==")
        print(f"  shared mu recovered      : {red['mu']:.6e}  (reference {DEFAULT_SOLAR_MU_AU_DAY:.6e})")
        print("  centrifugal coefficient  : k = ell^2 by construction (ell = |r x p|)")
        print(f"  k / ell^2 across {red['n_orbits']:>3d} bodies : median {med:.6f}, max |dev| {maxdev:.2e}")
        print(f"  reduced Hamiltonian      : {red['reduced_hamiltonian']}")
        print(f"\n  {red['derivation']}")
        summary["reduction"] = {
            "rddot_source": red.get("rddot_source", "finite_difference"),
            "mu": red["mu"],
            "k_over_ell_squared_median": med,
            "k_over_ell_squared_max_abs_dev": maxdev,
            "reduced_hamiltonian": red["reduced_hamiltonian"],
            "derivation": red["derivation"],
            "per_orbit": red["per_orbit"],
        }

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "noether_kepler_summary.json"
    out.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(f"\nSummary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
