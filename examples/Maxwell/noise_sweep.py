#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Noise / support-stability sweep for the broadened Maxwell library.

Headline (a self-contained identifiability result): how stable is the SELECTED
support as field noise grows?  For each (case, derivative-provider, noise level)
we run many seeds and report the per-term selection frequency P(term selected),
true-positive rate, false-discovery rate, and median coefficient error /
residual.  The wire is the stress case: its broadened library is highly coherent
(support VIF ~ 23), so it is where a Laplacian decoy will flicker first.

Derivative providers (all expose forward/grad/grad_grad on the grid):

* ``perfect``   - exact analytic derivatives (noise-free reference / upper bound).
* ``surrogate`` - a NestyNet segmented surrogate trained on the noisy values,
                  using its OWN analytic derivatives (the on-thesis method).
* ``fd``        - central finite differences of the noisy values.
* ``fft``       - spectral (FFT) spatial derivatives of the noisy values, with a
                  finite-difference time derivative (time is not periodic).

The FD and (especially) spectral providers are deliberately strong, standard
estimators, not strawmen: for band-limited periodic fields FFT differentiation
is near-optimal, so we never claim to beat it where it is exact.  The point is
the regime map -- noisy, localized, non-periodic -- where a trained surrogate's
implicit regularization helps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_de.system_de_search import (
    VectorSystemDESearchConfig,
    discover_vector_system_de_from_surrogate,
)

from conditioning_audit import apply_alias_drops, audit_vector_library
from problem_defs import (
    GROUND_TRUTH,
    PROBLEM_REGISTRY,
    _vec_key,
    build_problem_data,
    build_vector_terms,
    maxwell_units_spec,
)
from spectral_derivatives import (
    _tensor_grid_from_rows,
    build_derivative_targets,
    build_spectral_lowpass,
    build_spectral_spatial_derivative_targets,
    build_spectral_spatial_hessian_diag,
)
from tabulated_surrogate import TabulatedVectorSurrogate

SPATIAL = (1, 2, 3)
TIME = 0

# Defaults for the trained-surrogate provider.  objective="value" is the honest
# noise test: the surrogate fits only the (noisy) field values and its analytic
# derivatives are used downstream -- NO oracle/Sobolev derivative targets.
# NB: train/val/batch sizes are chosen ADAPTIVELY from the grid in
# build_provider (see _adaptive_data_sizes) -- a fixed (e.g. val=2000) default
# would collapse val->1, batch->1 on a small grid, and a batch of 1 makes the LM
# assemble its global system from thousands of minibatch pieces per epoch
# (huge within-epoch memory).  Keep batch healthy.
DEFAULT_SURROGATE_KW: dict[str, Any] = {
    "num_segments": 16,
    "epochs": 2500,
    "loss_target": 1e-8,
    "surrogate_objective": "value",
    "component_workers": 1,
}
DEFAULT_SURROGATE_DATA_DIR = Path(__file__).resolve().parent / "_noise_sweep_work"


def _adaptive_data_sizes(n_rows: int) -> dict[str, int]:
    """Pick train/val/batch sizes that keep the LM batch healthy on any grid.

    Batch is kept LARGE (few minibatches/epoch, like the validated mw002 run):
    a tiny batch both wastes memory (the LM assembles its global system from
    n_train/batch pieces per epoch) and hurts convergence.
    """
    # Caps (4000 train / 2000 val / 2000 batch) match the validated mw002
    # Sobolev configuration, which converges to val-loss ~1e-6.
    n = int(n_rows)
    nval = max(16, min(2000, n // 5))
    ntrain = max(16, min(4000, n - nval))
    batch = max(16, min(2000, ntrain, nval))
    return {"ndata_train": ntrain, "ndata_val": nval, "batch_size": batch}


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def add_relative_noise(Y_clean: torch.Tensor, noise_rel: float, seed: int) -> torch.Tensor:
    """Add Gaussian noise scaled per-component by that component's std.

    Structural-zero components (std == 0) are left untouched -- injecting noise
    into an identically-zero field component would be unphysical.
    """
    if noise_rel <= 0.0:
        return Y_clean.clone()
    rng = np.random.default_rng(int(seed))
    Y = Y_clean.clone()
    arr = Y.cpu().numpy()
    for c in range(arr.shape[1]):
        sd = float(arr[:, c].std())
        if sd > 0.0:
            arr[:, c] = arr[:, c] + rng.normal(0.0, noise_rel * sd, size=arr.shape[0])
    return torch.from_numpy(arr).to(Y_clean.dtype)


# ---------------------------------------------------------------------------
# Finite-difference derivatives on the tensor grid
# ---------------------------------------------------------------------------


def _fd_tables(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference first derivatives (all axes) and second derivatives
    (spatial diagonal) on the (t,x,y,z) tensor grid.

    Returns ``(G_fd (N,Ny,4), H_fd (N,Ny,4,4))``.
    """
    Y_grid, linear, vals, shape_t = _tensor_grid_from_rows(
        X.detach(), Y.detach(), grid_cols=(TIME, *SPATIAL)
    )
    dims = [int(v) for v in shape_t.tolist()]
    Yg = Y_grid.cpu().numpy()  # (nt,nx,ny,nz,Ny)
    coords = [v.cpu().numpy() for v in vals]
    Ny = Yg.shape[-1]

    G = np.zeros((*dims, Ny, 4), dtype=np.float64)
    H = np.zeros((*dims, Ny, 4, 4), dtype=np.float64)
    for a in range(4):
        if dims[a] < 2:
            continue
        ga = np.gradient(Yg, coords[a], axis=a)  # (*dims, Ny)
        G[..., a] = ga
        if a in SPATIAL and dims[a] >= 3:
            H[..., a, a] = np.gradient(ga, coords[a], axis=a)

    lin = linear.cpu().numpy()
    G_flat = G.reshape(-1, Ny, 4)[lin]
    H_flat = H.reshape(-1, Ny, 4, 4)[lin]
    return (
        torch.from_numpy(G_flat).to(Y.dtype),
        torch.from_numpy(H_flat).to(Y.dtype),
    )


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def build_provider(
    kind: str,
    problem,
    X: torch.Tensor,
    Y_clean: torch.Tensor,
    Y_noisy: torch.Tensor,
    G_exact: torch.Tensor,
    *,
    surrogate_kwargs: dict | None = None,
) -> torch.nn.Module:
    """Return a forward/grad/grad_grad provider for the requested derivative source."""
    kind = kind.lower()
    if kind == "perfect":
        H = build_spectral_spatial_hessian_diag(X, Y_clean, spatial_cols=SPATIAL, time_col=TIME)
        return TabulatedVectorSurrogate(X, Y_clean, G_exact, H)

    if kind == "fd":
        G_fd, H_fd = _fd_tables(X, Y_noisy)
        return TabulatedVectorSurrogate(X, Y_noisy, G_fd, H_fd)

    if kind == "fft":
        # Spectral spatial derivatives; finite-difference time (time is not periodic).
        G = build_spectral_spatial_derivative_targets(
            X, Y_noisy, G_time=None, spatial_cols=SPATIAL, time_col=TIME
        ).clone()
        G_fd, _ = _fd_tables(X, Y_noisy)
        G[:, :, TIME] = G_fd[:, :, TIME]
        H = build_spectral_spatial_hessian_diag(X, Y_noisy, spatial_cols=SPATIAL, time_col=TIME)
        return TabulatedVectorSurrogate(X, Y_noisy, G, H)

    if kind == "denoised":
        # Friend's diagnostic #1: spatially low-pass the noisy field (declared
        # bandlimit prior), then build the WHOLE feature table (values, curl,
        # Laplacian) from that single consistent denoised field.  Tests whether
        # the noisy samples + a smoothness prior support decoy rejection (an
        # upper bound separating data-limited from surrogate/pipeline-limited).
        cutoff = float((surrogate_kwargs or {}).get("lowpass_cutoff", 0.5))
        Y_lp = build_spectral_lowpass(
            X, Y_noisy, spatial_cols=SPATIAL, time_col=TIME, cutoff_frac=cutoff
        )
        G = build_spectral_spatial_derivative_targets(
            X, Y_lp, G_time=None, spatial_cols=SPATIAL, time_col=TIME
        ).clone()
        # Oracle time derivative (anchor), matching the Sobolev surrogate's
        # spectral_spatial_exact_time setup -- so this isolates the spatial
        # derivative-from-noisy-data question (FD time would inject its own
        # noise-independent bias that leaks the decoys even at zero noise).
        G[:, :, TIME] = G_exact[:, :, TIME]
        H = build_spectral_spatial_hessian_diag(X, Y_lp, spatial_cols=SPATIAL, time_col=TIME)
        return TabulatedVectorSurrogate(X, Y_lp, G, H)

    if kind in ("surrogate", "surrogate_h2"):
        # Lazy import: only pull in the (heavy) training stack when needed.
        import run_benchmark as rb

        kw = dict(DEFAULT_SURROGATE_KW)
        kw.update(_adaptive_data_sizes(int(X.shape[0])))  # healthy train/val/batch for this grid
        kw.update(surrogate_kwargs or {})  # explicit caller overrides win
        kw.pop("lowpass_cutoff", None)  # provider-config for 'denoised', not a trainer arg
        h2_weight = float(kw.pop("h2_weight", 0.0) or 0.0)  # provider-config, not a trainer arg
        data_dir = Path(kw.pop("data_dir", DEFAULT_SURROGATE_DATA_DIR))
        data_dir.mkdir(parents=True, exist_ok=True)
        # The H^2 (curvature-supervised) variant needs the Sobolev path and a
        # positive curvature weight; the plain 'surrogate' kind never gets H^2, so
        # it remains the established (value+gradient) ceiling baseline.
        use_h2 = (kind == "surrogate_h2") and h2_weight > 0.0
        if use_h2:
            kw["surrogate_objective"] = "sobolev"
        objective = str(kw.get("surrogate_objective", "value")).lower()
        G_target = G_exact  # unused by the value path
        H_target = None
        if objective == "sobolev":
            # Derivative-aware targets built from the (noisy) field values:
            # spatial derivatives via FFT; the time derivative from the exact
            # generator (oracle dt) -- the validated mw002 configuration.
            sob_target = str(kw.get("sobolev_target", "spectral_spatial_exact_time"))
            G_target = build_derivative_targets(sob_target, X, Y_noisy, G_exact)
            if use_h2:
                # Curvature targets: the spectral spatial Hessian diagonal of the
                # (noisy) field -- the same construction the 'denoised' diagnostic
                # showed rejects the Laplacian decoys to ~1e-4.  Supervising the
                # spatial diagonals (the Laplacian's constituents) is the lever the
                # friend identified for the surrogate's H^2 ceiling: make the
                # learned 2nd derivatives accurate enough to reject the decoys.
                H_target = build_spectral_spatial_hessian_diag(
                    X, Y_noisy, spatial_cols=SPATIAL, time_col=TIME
                )
                kw["hess_weight"] = h2_weight
                kw.setdefault("hess_pairs", [(int(a), int(a)) for a in SPATIAL])
        surro, _val, _info = rb.train_vector_nestynet_surrogate(
            problem,
            X,
            Y_noisy,
            data_dir=data_dir,
            device=torch.device("cpu"),
            dtype=torch.float64,
            G_target=G_target,
            G_exact=G_exact,
            H_target=H_target,
            **kw,
        )
        return surro

    raise ValueError(f"unknown derivative provider {kind!r}")


# ---------------------------------------------------------------------------
# One discovery run
# ---------------------------------------------------------------------------


def run_one(
    provider: torch.nn.Module,
    X: torch.Tensor,
    problem,
    *,
    stlsq_lambda: float = 5e-4,
    active_thresh: float = 1e-3,
    max_points: int = 25000,
    units_spec: Any = None,
    enforce_units: bool = False,
) -> dict[str, Any]:
    """Run broad-library discovery on one provider; return the selected support."""
    vector_terms, name_by_key, named_vecs = build_vector_terms(problem, include_laplacian=True)
    audit = audit_vector_library(provider, X, vector_terms, name_by_key, problem.equations)
    alias = audit["rank_status"] == "NONIDENTIFIABLE_ALIAS"
    if alias and audit["drop_terms"]:
        vector_terms, name_by_key, named_vecs = apply_alias_drops(
            vector_terms, name_by_key, named_vecs, audit["drop_terms"]
        )

    cfg = VectorSystemDESearchConfig(
        x_axis=TIME,
        order_candidates=(1,),
        include_const=False,
        stlsq_lambda=stlsq_lambda,
        stlsq_max_iter=20,
        sparsity_penalty=1e-6,
        share_support_across_equations=False,
        max_points=max_points,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
    )
    loader = DataLoader(TensorDataset(X), batch_size=int(X.shape[0]), shuffle=False)
    res = discover_vector_system_de_from_surrogate(
        provider, loader, cfg=cfg, equations=problem.equations,
        vector_terms=vector_terms, device=torch.device("cpu"),
    )

    coeff_by_name: dict[str, float] = {}
    for j, t in enumerate(res.term_vecs):
        if t is None:
            continue
        nm = name_by_key.get(_vec_key(t))
        if nm is None:
            continue
        coeff_by_name[nm] = float(res.coeffs[:, j].abs().max())
    active = {nm for nm, c in coeff_by_name.items() if c > active_thresh}
    max_rms = max((max(e) for e in res.rms_train), default=float("nan"))
    return {
        "coeff_by_name": coeff_by_name,
        "active": active,
        "max_rms": float(max_rms),
        "alias": alias,
        "rank_deficient": bool(getattr(res, "rank_deficient", False)),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _true_support(problem) -> tuple[set[str], dict[str, float], list[str]]:
    """Return (true-term names, expected magnitude per true term, full term universe)."""
    gt = GROUND_TRUTH[problem.id]
    expected_mag = {
        nm: max(abs(v) for v in eqmap.values()) for nm, eqmap in gt.expected_coeffs.items()
    }
    true_terms = set(expected_mag)
    universe = sorted(true_terms | set(gt.decoy_terms))
    return true_terms, expected_mag, universe


def aggregate(runs: Sequence[dict], problem) -> dict[str, Any]:
    """Aggregate per-seed runs into selection frequencies + stability metrics."""
    true_terms, expected_mag, universe = _true_support(problem)
    n = max(len(runs), 1)

    p_select = {t: sum(1 for r in runs if t in r["active"]) / n for t in universe}
    # Median |coefficient| per term across seeds (0 if a term never appears),
    # so decoy magnitudes are visible vs the discovery's tolerance, not just P(selected).
    coeff_med = {
        t: float(np.median([r["coeff_by_name"].get(t, 0.0) for r in runs])) if runs else 0.0
        for t in universe
    }
    tprs, fdrs, cerrs, rmss = [], [], [], []
    for r in runs:
        found = set(r["active"])
        tprs.append(len(found & true_terms) / max(len(true_terms), 1))
        fdrs.append(len(found - true_terms) / max(len(found), 1))
        cerrs.append(
            max((abs(r["coeff_by_name"].get(t, 0.0) - expected_mag[t]) for t in true_terms), default=0.0)
        )
        rmss.append(r["max_rms"])

    return {
        "n_seeds": len(runs),
        "p_select": p_select,
        "coeff_median": coeff_med,
        "tpr_mean": float(np.mean(tprs)) if tprs else float("nan"),
        "fdr_mean": float(np.mean(fdrs)) if fdrs else float("nan"),
        "coeff_err_median": float(np.median(cerrs)) if cerrs else float("nan"),
        "residual_median": float(np.median(rmss)) if rmss else float("nan"),
        "true_terms": sorted(true_terms),
        "universe": universe,
        "n_alias": sum(1 for r in runs if r["alias"]),
        "n_rank_deficient": sum(1 for r in runs if r["rank_deficient"]),
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def sweep(
    case_ids: Sequence[str],
    providers: Sequence[str],
    noise_levels: Sequence[float],
    seeds: Sequence[int],
    *,
    fast: bool = False,
    surrogate_kwargs: dict | None = None,
    verbose: bool = True,
    units_on: bool = False,
) -> dict[str, Any]:
    """Run the full sweep and return a nested results dict.

    When ``units_on`` is set, each surrogate is trained once and discovery is run
    *twice* on the same provider -- units-off (the adversarial broad-library stress
    test) and units-on (the principled pipeline that prunes dimensionally
    inadmissible terms, e.g. the second-order Laplacian decoys).  The units-on
    aggregates are stored under provider key ``"<prov>_unitson"``."""
    results: dict[str, Any] = {
        "providers": list(providers),
        "noise_levels": list(noise_levels),
        "seeds": list(seeds),
        "cases": {},
    }
    for cid in case_ids:
        problem = PROBLEM_REGISTRY[cid]
        X, Y_clean, G_exact, _meta = build_problem_data(problem, fast=fast)
        case_out: dict[str, Any] = {}
        uspec = maxwell_units_spec(problem) if units_on else None
        for prov in providers:
            prov_out: dict[str, Any] = {}
            prov_out_on: dict[str, Any] = {}
            for nz in noise_levels:
                seed_iter = [0] if (prov == "perfect" or nz == 0.0) else list(seeds)
                runs = []
                runs_on = []
                for sd in seed_iter:
                    Y_noisy = add_relative_noise(Y_clean, float(nz), int(sd))
                    provider = build_provider(  # trained once; reused for both discoveries
                        prov, problem, X, Y_clean, Y_noisy, G_exact,
                        surrogate_kwargs=surrogate_kwargs,
                    )
                    runs.append(run_one(provider, X, problem))
                    if uspec is not None:
                        runs_on.append(run_one(
                            provider, X, problem,
                            units_spec=uspec, enforce_units=True,
                        ))
                agg = aggregate(runs, problem)
                prov_out[f"{nz:g}"] = agg
                if verbose:
                    _print_row(cid, prov, nz, agg)
                if uspec is not None:
                    agg_on = aggregate(runs_on, problem)
                    prov_out_on[f"{nz:g}"] = agg_on
                    if verbose:
                        _print_row(cid, prov + "[units]", nz, agg_on)
            case_out[prov] = prov_out
            if uspec is not None:
                case_out[prov + "_unitson"] = prov_out_on
        results["cases"][cid] = case_out
    return results


def _print_row(cid: str, prov: str, nz: float, agg: dict) -> None:
    ps = agg["p_select"]
    cols = "  ".join(f"{t}={ps[t]:.2f}" for t in agg["universe"])
    print(
        f"{cid:7s} {prov:9s} noise={nz:<7g} TPR={agg['tpr_mean']:.2f} FDR={agg['fdr_mean']:.2f} "
        f"cerr~{agg['coeff_err_median']:.2e} res~{agg['residual_median']:.2e} | {cols}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Maxwell broad-library noise / support-stability sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cases", type=str, default="mw001,mw003",
                    help="Comma-separated case ids (mw000,mw001,mw002,mw003)")
    ap.add_argument("--providers", type=str, default="perfect,fd,fft",
                    help="Comma-separated providers (perfect,fd,fft,denoised,surrogate,surrogate_h2)")
    ap.add_argument("--noise", type=str, default="0,1e-3,1e-2",
                    help="Comma-separated relative-noise levels")
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds per noisy config")
    ap.add_argument("--fast", action="store_true", help="Reduced grids")
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    # Trained-surrogate provider cost controls (ignored by perfect/fd/fft):
    ap.add_argument("--surrogate_epochs", type=int, default=None, help="Override surrogate epochs")
    ap.add_argument("--surrogate_segments", type=int, default=None, help="Override surrogate segments")
    ap.add_argument("--lowpass_cutoff", type=float, default=0.5,
                    help="Spatial low-pass cutoff (fraction of max |k|) for the 'denoised' provider")
    ap.add_argument("--component_workers", type=int, default=1, help="Parallel component fits")
    ap.add_argument("--objective", type=str, default="value", choices=["value", "sobolev"],
                    help="Trained-surrogate objective (sobolev = derivative-aware, oracle dt)")
    ap.add_argument("--canonical_init", action="store_true",
                    help="Force canonical (deterministic) init for the surrogate, incl. Sobolev")
    ap.add_argument("--grad_weight_ramp", type=str, default=None,
                    help="Comma-separated Sobolev grad-weight ramp, e.g. 0,1e-3,1e-2,1e-1,1.0")
    ap.add_argument("--hess_weight", type=float, default=1.0,
                    help="Curvature (H^2) supervision weight for the 'surrogate_h2' provider "
                         "(spectral spatial-Hessian targets; forces the Sobolev objective). "
                         "Only the 'surrogate_h2' provider consumes it; 'surrogate' stays H^1.")
    ap.add_argument("--units_on", action="store_true",
                    help="Also run units-on discovery (declared Maxwell coeff basis prunes "
                         "dimensionally inadmissible terms incl. the Laplacian decoys); "
                         "stored under provider key '<prov>_unitson' (trained once, discovered twice)")
    ap.add_argument("--max_restarts", type=int, default=0,
                    help="Random-restart failed components (val >> sibling median) up to N times")
    ap.add_argument("--restart_factor", type=float, default=50.0,
                    help="A component fails if its val >= restart_factor x median(siblings)")
    ap.add_argument("--restart_epochs", type=int, default=1500,
                    help="Per-restart epochs (single-stage gw=1, from-scratch random init)")
    args = ap.parse_args()

    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    noise_levels = [float(x) for x in args.noise.split(",") if x.strip()]
    seeds = list(range(int(args.seeds)))

    surrogate_kwargs: dict[str, Any] = {
        "component_workers": int(args.component_workers),
        "surrogate_objective": str(args.objective),
    }
    if args.surrogate_epochs is not None:
        surrogate_kwargs["epochs"] = int(args.surrogate_epochs)
    surrogate_kwargs["lowpass_cutoff"] = float(args.lowpass_cutoff)
    surrogate_kwargs["h2_weight"] = float(args.hess_weight)
    if args.surrogate_segments is not None:
        surrogate_kwargs["num_segments"] = int(args.surrogate_segments)
    if args.canonical_init:
        surrogate_kwargs["canonical_init"] = True
    if args.grad_weight_ramp:
        surrogate_kwargs["grad_weight_ramp"] = [float(x) for x in args.grad_weight_ramp.split(",") if x.strip()]
    if args.max_restarts > 0:
        surrogate_kwargs["restart_max"] = int(args.max_restarts)
        surrogate_kwargs["restart_factor"] = float(args.restart_factor)
        surrogate_kwargs["restart_epochs"] = int(args.restart_epochs)

    print("=" * 100)
    print(f"Maxwell noise sweep: cases={case_ids} providers={providers} "
          f"noise={noise_levels} seeds={len(seeds)} fast={args.fast}")
    print("-" * 100)
    results = sweep(
        case_ids, providers, noise_levels, seeds,
        fast=args.fast, surrogate_kwargs=surrogate_kwargs,
        units_on=bool(args.units_on),
    )
    print("=" * 100)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
