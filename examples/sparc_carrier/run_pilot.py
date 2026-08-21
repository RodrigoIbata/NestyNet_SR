# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""SPARC baryonic-acceleration-carrier pilot (bulgeless rung).

Wires the real pipeline: Stage-A segmented surrogate (analytic gradients) ->
generalized-symmetry determining solve (noise-calibrated spectral-gap) ->
quotient compilation -> promotion decision, plus the go/no-go battery:

  1. carrier certificate on the discovery galaxies (covector = (1, Upsilon_d))
  2. galaxy bootstrap of the GS solve (fixed surrogate) and a smaller
     retrain bootstrap (surrogate refitted per resample)
  3. held-out-galaxy closure: 1D law through discovered z vs the unrestricted
     2D surrogate, plus a wrong-carrier closure curve
  4. null controls: row-shuffled y and component-shuffled g_gas (each with a
     freshly trained surrogate)

Inputs are the two component accelerations in a common scale (median g_disk),
target is log10 g_obs: a monotone map of the target preserves the rank-one
gradient structure and the carrier covector, while conditioning the fit.

Usage:  python3 run_pilot.py [--quick]  (quick: fewer epochs/bootstraps)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from nestynet.dataloader import PhysDataset
from torch.utils.data import DataLoader

from nestynet_sr.sr_core import build_initial_ast
from nestynet_sr.sr_search.config import LMHyperparams, ModelHyperparams
from nestynet_sr.sr_search.model_builders import LeafBuilder, build_composite_ast
from nestynet_sr.sr_search.training import train_initial_model
from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.affine_algebra import _fit_normalization, discover_affine_algebra
from nestynet_sr.sr_gs.promotion import evaluate_reduction_promotion
from nestynet_sr.sr_gs.quotient import compile_reduction_plan

HERE = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")
DTYPE = torch.float64

GS_KW = dict(
    heldout_fraction=0.25,
    nullity_strategy="spectral_gap",
    min_spectral_gap=10.0,
    closure_tol=3.0e-2,
    bootstrap_angle_tol=0.10,
    heldout_consistency_factor=3.0,
)


# ----------------------------------------------------------------- data


def load_rows(path: Path):
    rows = list(csv.DictReader(open(path)))
    gal = np.array([r["galaxy"] for r in rows])
    g_gas = np.array([float(r["g_gas"]) for r in rows])
    g_disk = np.array([float(r["g_disk"]) for r in rows])
    g_obs = np.array([float(r["g_obs"]) for r in rows])
    return gal, g_gas, g_disk, g_obs


def split_galaxies(gal: np.ndarray, seed: int, frac_disc: float = 0.7):
    names = np.unique(gal)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_disc = int(round(frac_disc * len(names)))
    return names[:n_disc], names[n_disc:]


# ------------------------------------------------------- surrogate fit


def fit_surrogate(X: np.ndarray, y: np.ndarray, *, epochs: int, seed: int,
                  num_segments: int = 32):
    """Train a 2-input segmented Stage-A surrogate; return (model, leaf, best_val)."""
    torch.manual_seed(seed)
    n = len(X)
    n_train = int(0.8 * n)
    n_val = n - n_train
    common = dict(X_data=X, y_data=y.reshape(-1, 1),
                  split_policy="random", split_seed=seed,
                  ndata_select=n_train, ndata_select_val=n_val)
    ds_train = PhysDataset(mode="train", **common)
    ds_val = PhysDataset(mode="validation", **common)
    dl_train = DataLoader(ds_train, batch_size=n_train, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=n_val, shuffle=False)

    model_hp = ModelHyperparams()
    root = build_initial_ast(2, num_segments=num_segments, dual_layer=False)
    model, _nparam, root = build_composite_ast(
        root, num_segments, False, LeafBuilder(model_hp, DEVICE, DTYPE), DEVICE, DTYPE)

    lm_hp = LMHyperparams(strategy="direct_solve", epochs=epochs)
    best_val, _best_tr, best_p, lm_opt = train_initial_model(
        model, dl_train, dl_val, epochs, lm_hp.strategy,
        nval_patience=150, loss_target=1.0e-9, epochs_min=100,
        chisq_tol=lm_hp.chisq_tol, device=DEVICE, lm_hp=lm_hp,
        log_to_console=False)
    lm_opt._update_param_groups(best_p)
    return model, model.leaf[0], float(best_val)


def warp(X: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Per-column asinh input warp: spreads 4-5 decades of acceleration so the
    segmented surrogate resolves the dwarf corner; gradients are chain-ruled
    back to physical coordinates exactly (the pipeline's warp-chart idea)."""
    return np.arcsinh(X / a)


def warp_grad_factor(X: np.ndarray, a: np.ndarray) -> np.ndarray:
    """du/dx for u = asinh(x/a), elementwise: converts G_u to G_x."""
    return 1.0 / np.sqrt(a ** 2 + X ** 2)


def surrogate_grad_sample(leaf, X: np.ndarray):
    """Analytic surrogate values and input gradients at the data rows."""
    xt = torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        f = leaf(xt).reshape(-1)
        g = leaf.grad({"x": xt})
        if g.dim() == 2:
            g = g.unsqueeze(1)
        g = g[:, 0, :]
    return f.cpu().numpy(), g.cpu().numpy()


# ------------------------------------------------------------ GS layer


def gs_certificate(X, y_hat, G, *, bootstrap: int = 8):
    """Determining solve + quotient compilation + promotion. Returns dict + covector."""
    alg = discover_affine_algebra(X, y_hat, G, bootstrap=bootstrap, **GS_KW)
    plan = compile_reduction_plan(alg)
    cfg = GeneralizedSymmetryConfig(
        enabled=True, mode="propose", general_affine=True,
        general_affine_promotion_noise_calibrated=True)
    decision = evaluate_reduction_promotion(plan, cfg)

    cov = None
    if plan.status == "compiled" and plan.invariant_coordinates:
        prov = plan.invariant_coordinates[0].provenance
        if "covector" in prov:
            cov = np.asarray(prov["covector"], dtype=float)
            if cov[0] < 0:
                cov = -cov

    ev = alg.evidence or {}
    cert = {
        "nullity": alg.nullity,
        "discovered_nullity": alg.discovered_nullity,
        "distribution_rank": getattr(alg, "distribution_rank", None),
        "spectral_gap": ev.get("spectral_gap"),
        "covector_shrink_steps": ev.get("covector_shrink_steps"),
        "train_residual_rel": alg.train_residual_rel,
        "heldout_residual_rel": alg.heldout_residual_rel,
        "bracket_closure_residual": alg.bracket_closure_residual,
        "bootstrap_max_principal_angle": (max(alg.bootstrap_principal_angles)
                                          if alg.bootstrap_principal_angles else None),
        "quotient_ready": bool(alg.certificate.quotient_ready) if alg.certificate else None,
        "quotient_policy": alg.certificate.quotient_policy if alg.certificate else None,
        "plan_status": plan.status,
        "plan_reason": plan.reason,
        "output_action": plan.output_action.to_report() if plan.output_action else None,
        "promotion": decision.to_report(),
        "covector": None if cov is None else [float(v) for v in cov],
        "upsilon_d": None if cov is None or abs(cov[0]) < 1e-12 else float(cov[1] / cov[0]),
    }
    return cert, cov


def covector_angle_deg(cov: np.ndarray) -> float:
    return math.degrees(math.atan2(cov[1], cov[0]))


def orientation_carrier(G, gals, mag_floor_pct=20.0):
    """Rank-one carrier readout from gradient DIRECTIONS, per-galaxy weighted.

    For y = F(c.x) every gradient row points along c, so the orientation
    tensor M = <u u^T> of unit gradient directions has top eigenvector c and
    eigenvalue contrast lam0/lam1 -> inf for an exact carrier. Direction (not
    energy) statistics avoid domination by the huge log-space gradients of
    the gas-rich dwarf rows; per-galaxy averaging stops row-rich galaxies from
    out-voting the rest; the magnitude floor drops rows whose gradient is too
    small to define a direction (a near-constant control surrogate then shows
    contrast ~ 1 instead of a spurious rank-one signature).
    """
    mag = np.linalg.norm(G, axis=1)
    keep = mag > np.percentile(mag, mag_floor_pct)
    U = G[keep] / mag[keep][:, None]
    gk = np.asarray(gals)[keep]
    tensors = []
    for name in np.unique(gk):
        Un = U[gk == name]
        tensors.append((Un[:, :, None] * Un[:, None, :]).mean(0))
    M = np.mean(tensors, axis=0)
    w, V = np.linalg.eigh(M)  # ascending eigenvalues
    c = V[:, -1].copy()
    if c[0] < 0:
        c = -c
    ups = float(c[1] / c[0]) if abs(c[0]) > 1e-12 else math.inf
    return {
        "eigenvalues": [float(x) for x in w[::-1]],
        "contrast": float(w[-1] / max(w[0], 1e-300)),
        "n_rows_kept": int(keep.sum()),
        "n_galaxies": len(tensors),
        "covector": [float(x) for x in c],
        "upsilon_d": ups,
    }, c


def translation_sector(X, y_hat, G):
    """Determining solve restricted to constant generators (b, alpha, beta).

    The full-affine spectral-gap gate is calibrated for ~1e-3 gradient noise
    and abstains at real SPARC scatter (~0.12 dex). The 4-unknown translation
    sector (rows: g.b - alpha - beta*y = 0) is nullity-1 for a rank-one
    carrier and much better conditioned, so it gives the soft-carrier readout
    the strict gate declines to certify. Output purity ~ 0 marks a genuine
    input-space annihilator; the y-shuffle control shows beta-dominated tails.
    """
    norm = _fit_normalization(X, y_hat)
    xs = np.asarray(norm.x_scale, dtype=float)
    yn = norm.normalize_y(y_hat)
    gn = G * (xs / float(norm.y_scale))
    n = len(yn)
    D = np.column_stack([gn, -np.ones(n), -yn]) / math.sqrt(n)
    _, s, Vt = np.linalg.svd(D, full_matrices=False)
    v = Vt[-1]
    b, alpha, beta = v[:2], v[2], v[3]
    c_norm = np.array([-b[1], b[0]])
    c_phys = c_norm / xs
    if c_phys[0] < 0:
        c_phys = -c_phys
    nrm = np.linalg.norm(c_phys)
    cov = c_phys / nrm if nrm > 0 else c_phys
    ups = float(cov[1] / cov[0]) if abs(cov[0]) > 1e-12 else math.inf
    return {
        "singular_values": [float(x) for x in s],
        "gap": float(s[-2] / s[-1]) if s[-1] > 0 else math.inf,
        "residual_rel": float(s[-1] / s[0]),
        "output_purity": float(math.hypot(alpha, beta)),
        "covector": [float(x) for x in cov],
        "upsilon_d": ups,
    }, cov


# ------------------------------------------------------------- closure


def knn1d_fit_predict(z_tr, y_tr, z_te, k=25):
    ztr = np.log10(np.clip(z_tr, 1e-6, None))[:, None]
    zte = np.log10(np.clip(z_te, 1e-6, None))[:, None]
    mu, sd = ztr.mean(), ztr.std() + 1e-12
    A, B = (ztr - mu) / sd, (zte - mu) / sd
    d2 = (B - A.T) ** 2
    nn = np.argsort(d2, axis=1)[:, :k]
    dn = np.take_along_axis(d2, nn, axis=1)
    w = 1.0 / (dn + 1e-6)
    return (w * y_tr[nn]).sum(1) / w.sum(1)


def closure_rms(cov, X_tr, y_tr, X_te, y_te) -> float:
    z_tr, z_te = X_tr @ cov, X_te @ cov
    ok_tr, ok_te = z_tr > 0, z_te > 0
    pred = knn1d_fit_predict(z_tr[ok_tr], y_tr[ok_tr], z_te[ok_te])
    return float(np.sqrt(np.mean((y_te[ok_te] - pred) ** 2)))


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("bulgeless", "gold"), default="bulgeless")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--segments", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot_gs", type=int, default=200)
    ap.add_argument("--boot_retrain", type=int, default=12)
    ap.add_argument("--quick", action="store_true",
                    help="smoke settings: fewer epochs and bootstraps")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.boot_gs, args.boot_retrain = 200, 20, 2

    t0 = time.time()
    gal, g_gas, g_disk, g_obs = load_rows(
        HERE / "data" / f"sparc_carrier_{args.dataset}.csv")
    scale = float(np.median(g_disk))
    X = np.stack([g_gas / scale, g_disk / scale], axis=1)
    y = np.log10(g_obs)

    disc_names, held_names = split_galaxies(gal, args.seed)
    disc = np.isin(gal, disc_names)
    held = ~disc
    print(f"rows: {disc.sum()} discovery ({len(disc_names)} galaxies), "
          f"{held.sum()} held-out ({len(held_names)} galaxies); "
          f"input scale {scale:.3e} m/s^2")

    report: dict = {"n_disc_rows": int(disc.sum()), "n_held_rows": int(held.sum()),
                    "n_disc_gal": len(disc_names), "n_held_gal": len(held_names),
                    "input_scale": scale, "args": vars(args)}

    # --- 1. surrogate + carrier certificate on discovery galaxies
    warp_a = np.array([np.median(np.abs(X[:, 0])), np.median(np.abs(X[:, 1]))])
    U = warp(X, warp_a)
    report["warp_a"] = [float(v) for v in warp_a]

    t = time.time()
    model, leaf, best_val = fit_surrogate(U[disc], y[disc],
                                          epochs=args.epochs, seed=args.seed,
                                          num_segments=args.segments)
    fit_secs = time.time() - t
    y_hat, G_u = surrogate_grad_sample(leaf, U[disc])
    G = G_u * warp_grad_factor(X[disc], warp_a)
    fit_rms = float(np.sqrt(np.mean((y_hat - y[disc]) ** 2)))
    print(f"surrogate: best val loss {best_val:.3e}, "
          f"train-row rms {fit_rms:.4f} dex, {fit_secs:.1f}s")

    cert, cov = gs_certificate(X[disc], y_hat, G)
    sector, _cov_sector = translation_sector(X[disc], y_hat, G)
    rank1, cov_rank1 = orientation_carrier(G, gal[disc])
    report["surrogate"] = {"best_val_loss": best_val, "fit_rms_dex": fit_rms,
                           "fit_seconds": fit_secs}
    report["certificate"] = cert
    report["translation_sector"] = sector
    report["gradient_rank1"] = rank1
    if cov is None:
        print(f"strict full-affine certificate: ABSTAINS "
              f"({cert['quotient_policy']}, nullity {cert['nullity']})")
    else:
        print(f"strict carrier: covector {cov.tolist()}, "
              f"Upsilon_d = {cert['upsilon_d']:.3f}, "
              f"promotion {cert['promotion'].get('state')}")
    print(f"orientation carrier: Upsilon_d = {rank1['upsilon_d']:.3f}, "
          f"angle {covector_angle_deg(cov_rank1):.2f} deg, "
          f"contrast lam0/lam1 = {rank1['contrast']:.2f} "
          f"({rank1['n_rows_kept']} rows kept)")
    print(f"translation sector: Upsilon_d = {sector['upsilon_d']:.3f}, "
          f"gap {sector['gap']:.2f}, output purity {sector['output_purity']:.3f}")
    if cov is None:
        cov = cov_rank1  # soft-carrier readout drives the rest of the battery

    # support certification: for a true carrier, Upsilon must be flat across
    # gas-rich vs star-dominated rows (disk-fraction terciles)
    f_d = (0.5 * X[disc][:, 1]) / (np.abs(X[disc][:, 0]) + 0.5 * X[disc][:, 1] + 1e-12)
    edges = np.quantile(f_d, [0.0, 1 / 3, 2 / 3, 1.0])
    support = {}
    for b in range(3):
        m = (f_d >= edges[b]) & (f_d <= edges[b + 1])
        try:
            rb, _cb = orientation_carrier(G[m], gal[disc][m])
            support[f"fd_tercile_{b}"] = {
                "fd_range": [float(edges[b]), float(edges[b + 1])],
                "upsilon_d": rb["upsilon_d"], "contrast": rb["contrast"],
                "n_rows": int(m.sum())}
        except Exception:
            support[f"fd_tercile_{b}"] = None
    report["support_terciles"] = support
    ok_terc = [v for v in support.values() if v]
    print("Upsilon by disk-fraction tercile: "
          + ", ".join(f"[{v['fd_range'][0]:.2f}-{v['fd_range'][1]:.2f}]:"
                      f"{v['upsilon_d']:.3f}" for v in ok_terc))

    # --- 2a. galaxy bootstrap of the GS solve (fixed surrogate)
    rng = np.random.default_rng(args.seed + 1)
    by_gal = defaultdict(list)
    disc_idx = np.flatnonzero(disc)
    for i in disc_idx:
        by_gal[gal[i]].append(i)
    local = {n: np.searchsorted(disc_idx, np.array(v)) for n, v in by_gal.items()}
    ups_gs, ang_gs, n_fail = [], [], 0
    for _ in range(args.boot_gs):
        pick = rng.choice(disc_names, size=len(disc_names), replace=True)
        rows = np.concatenate([local[n] for n in pick])
        try:
            r1_b, cv = orientation_carrier(G[rows], gal[disc][rows])
        except Exception:
            cv, r1_b = None, None
        if cv is None or not math.isfinite(r1_b["upsilon_d"]):
            n_fail += 1
            continue
        ups_gs.append(cv[1] / cv[0])
        ang_gs.append(covector_angle_deg(cv))
    if ups_gs:
        lo_q, med, hi_q = np.percentile(ups_gs, [16, 50, 84])
        print(f"GS galaxy bootstrap ({len(ups_gs)}/{args.boot_gs} compiled): "
              f"Upsilon_d {med:.3f} [{lo_q:.3f}, {hi_q:.3f}], "
              f"angle sd {np.std(ang_gs):.2f} deg")
    report["bootstrap_gs"] = {"n_ok": len(ups_gs), "n_fail": n_fail,
                              "upsilon_16_50_84": (list(np.percentile(ups_gs, [16, 50, 84]))
                                                   if ups_gs else None),
                              "angle_sd_deg": float(np.std(ang_gs)) if ang_gs else None}

    # --- 2b. retrain bootstrap (surrogate refit per galaxy resample)
    ups_rt, ang_rt, n_fail_rt = [], [], 0
    for b in range(args.boot_retrain):
        pick = rng.choice(disc_names, size=len(disc_names), replace=True)
        rows = np.concatenate([local[n] for n in pick])
        try:
            _m, lf, _bv = fit_surrogate(U[disc][rows], y[disc][rows],
                                        epochs=max(300, args.epochs // 2),
                                        seed=args.seed + 100 + b,
                                        num_segments=args.segments)
            yh, Gu_b = surrogate_grad_sample(lf, U[disc][rows])
            Gb = Gu_b * warp_grad_factor(X[disc][rows], warp_a)
            r1_b, cv = orientation_carrier(Gb, gal[disc][rows])
        except Exception:
            cv, r1_b = None, None
        if cv is None or not math.isfinite(r1_b["upsilon_d"]):
            n_fail_rt += 1
            continue
        ups_rt.append(cv[1] / cv[0])
        ang_rt.append(covector_angle_deg(cv))
    if ups_rt:
        lo_q, med, hi_q = np.percentile(ups_rt, [16, 50, 84])
        print(f"retrain bootstrap ({len(ups_rt)}/{args.boot_retrain} compiled): "
              f"Upsilon_d {med:.3f} [{lo_q:.3f}, {hi_q:.3f}], "
              f"angle sd {np.std(ang_rt):.2f} deg")
    report["bootstrap_retrain"] = {"n_ok": len(ups_rt), "n_fail": n_fail_rt,
                                   "upsilon_16_50_84": (list(np.percentile(ups_rt, [16, 50, 84]))
                                                        if ups_rt else None),
                                   "angle_sd_deg": float(np.std(ang_rt)) if ang_rt else None}

    # --- 3. held-out-galaxy closure
    if cov is not None:
        rms_1d = closure_rms(cov, X[disc], y[disc], X[held], y[held])
        yh_held, _ = surrogate_grad_sample(leaf, U[held])
        rms_2d = float(np.sqrt(np.mean((y[held] - yh_held) ** 2)))
        print(f"held-out closure: 1D through discovered z {rms_1d:.4f} dex, "
              f"2D surrogate {rms_2d:.4f} dex")
        curve = {}
        for u_w in (-0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            curve[u_w] = closure_rms(np.array([1.0, u_w]), X[disc], y[disc],
                                     X[held], y[held])
        ups_used = float(cov[1] / cov[0])
        curve[round(ups_used, 3)] = rms_1d
        print("wrong-carrier closure curve (Upsilon -> heldout dex): "
              + ", ".join(f"{u:g}:{v:.3f}" for u, v in sorted(curve.items())))
        report["closure"] = {"rms_1d_dex": rms_1d, "rms_2d_dex": rms_2d,
                             "wrong_carrier_curve": {str(k): v for k, v in curve.items()}}

    # --- 4. null controls (fresh surrogate each)
    controls = {}
    rngc = np.random.default_rng(args.seed + 2)
    for name, mutate in (
        ("y_shuffle", lambda Xd, yd: (Xd, yd[rngc.permutation(len(yd))])),
        ("gas_shuffle", lambda Xd, yd: (
            np.stack([Xd[rngc.permutation(len(yd)), 0], Xd[:, 1]], axis=1), yd)),
    ):
        Xc, yc = mutate(X[disc].copy(), y[disc].copy())
        Uc = warp(Xc, warp_a)
        try:
            _m, lf, bv = fit_surrogate(Uc, yc, epochs=max(300, args.epochs // 2),
                                       seed=args.seed + 500,
                                       num_segments=args.segments)
            yh, Gc_u = surrogate_grad_sample(lf, Uc)
            Gc = Gc_u * warp_grad_factor(Xc, warp_a)
            c_ctrl, cv_ctrl = gs_certificate(Xc, yh, Gc, bootstrap=8)
            r_ctrl, _cv_r = orientation_carrier(Gc, gal[disc])
            s_ctrl, _cv_s = translation_sector(Xc, yh, Gc)
        except Exception as exc:
            c_ctrl, cv_ctrl, r_ctrl, s_ctrl, bv = {"error": str(exc)}, None, None, None, None
        promoted = (c_ctrl.get("promotion") or {}).get("accepted")
        print(f"control {name}: val {bv if bv is None else format(bv, '.3e')}, "
              f"strict compiled {cv_ctrl is not None}, promoted {promoted}, "
              f"rank1 Upsilon {None if r_ctrl is None else round(r_ctrl['upsilon_d'], 3)}, "
              f"rank1 contrast {None if r_ctrl is None else round(r_ctrl['contrast'], 2)}, "
              f"sector purity {None if s_ctrl is None else round(s_ctrl['output_purity'], 3)}")
        controls[name] = {"best_val_loss": bv, "certificate": c_ctrl,
                          "gradient_rank1": r_ctrl, "translation_sector": s_ctrl}
    report["controls"] = controls

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = "quick" if args.quick else "full"
    path = out / f"pilot_report_{args.dataset}_{stamp}_seed{args.seed}.json"
    json.dump(report, open(path, "w"), indent=2, default=str)
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
