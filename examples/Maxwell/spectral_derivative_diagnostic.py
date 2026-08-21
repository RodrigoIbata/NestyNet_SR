#!/usr/bin/env python3
"""Spectral-derivative diagnostic: separate value interpolation from jet recovery.

Four-rung ladder for mw002 (conductive Maxwell), all scored by the *same* STLSQ
discovery + coefficient/RMS gates as the benchmark:

  1. exact generator derivatives                         -> discovery
  2. FFT spatial curls of ANALYTIC values (+exact dt)     -> discovery
  3. analytic grad of a value-only surrogate              -> discovery  (expected FAIL)
  4. FFT spatial curls of the SAME surrogate's LEARNED     -> discovery  (the "gem")
     values (+exact dt)

If 1,2,4 close and 3 fails, the failure is isolated to *differentiating the
value-only surrogate*, not to value interpolation or the sparse-regression layer.

Time is non-periodic (damped evolution), so FFT differentiation is spatial only;
the dt anchor uses the exact ("trusted") temporal derivative in rungs 2 and 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_benchmark as rb  # noqa: E402
from problem_defs import GROUND_TRUTH, PROBLEM_REGISTRY, build_problem_data  # noqa: E402

PID = sys.argv[1] if len(sys.argv) > 1 else "mw002"
problem = PROBLEM_REGISTRY[PID]
gt = GROUND_TRUTH[PID]

X, Y, G_exact, meta = build_problem_data(problem, fast=False)
X = X.to(torch.float64)
Y = Y.to(torch.float64)
G_exact = G_exact.to(torch.float64)
nt, nx, ny, nz = (int(meta["nt"]), int(meta["nx"]), int(meta["ny"]), int(meta["nz"]))
ncomp = int(Y.shape[1])
L = 2.0 * np.pi  # periodic box length on every spatial axis


def fft_spatial_grad(Y_np: np.ndarray) -> np.ndarray:
    """Spatial first-derivatives via FFT on the periodic box.

    Returns G (N, ncomp, 4); spatial columns (1,2,3) filled, time column 0 = 0.
    """
    Yg = Y_np.reshape(nt, nx, ny, nz, ncomp)
    G = np.zeros((nt, nx, ny, nz, ncomp, 4), dtype=np.float64)
    for gridaxis, n in ((1, nx), (2, ny), (3, nz)):
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)  # angular wavenumbers
        shape = [1, 1, 1, 1, 1]
        shape[gridaxis] = n
        ik = (1j * k).reshape(shape)
        Yhat = np.fft.fft(Yg, axis=gridaxis)
        G[..., gridaxis] = np.real(np.fft.ifft(Yhat * ik, axis=gridaxis))
    return G.reshape(-1, ncomp, 4)


def discover_with(label: str, Yvals: torch.Tensor, Gderiv: torch.Tensor) -> str:
    surrogate = rb.TabulatedVectorSurrogate(X, Yvals.contiguous(), Gderiv.contiguous())
    loader = DataLoader(TensorDataset(X), batch_size=int(X.shape[0]), shuffle=False)
    vt, name_by_key, named_vecs = rb.build_vector_terms(problem)
    cfg = rb.VectorSystemDESearchConfig(
        x_axis=0,
        order_candidates=(1,),
        include_const=False,
        stlsq_lambda=5e-4,
        stlsq_max_iter=20,
        sparsity_penalty=1e-6,
        share_support_across_equations=False,
        max_points=25000,
    )
    disc = rb.discover_vector_system_de_from_surrogate(
        surrogate, loader, cfg=cfg, equations=problem.equations,
        vector_terms=vt, device=torch.device("cpu"),
    )
    cs, _cm, cd = rb.validate_coefficients(disc, gt, name_by_key, named_vecs, verbose=False)
    rs, _rm, rd = rb.validate_rms(disc, gt, verbose=False)
    status = "FAIL" if "FAIL" in (cs, rs) else ("PARTIAL" if "PARTIAL" in (cs, rs) else "PASS")
    print(f"  {label:<48} {status:<8} coeff_err={cd.get('max_coeff_error', float('nan')):.4f}  rms={rd.get('max_rms', float('nan')):.3e}")
    return status


print(f"# {PID}: grid nt/nx/ny/nz = {nt}/{nx}/{ny}/{nz}, ncomp={ncomp}")
print(f"  {'rung':<48} {'status':<8} details")

discover_with("1. exact generator derivatives", Y, G_exact)

G2 = fft_spatial_grad(Y.numpy())
G2[:, :, 0] = G_exact.numpy()[:, :, 0]  # trusted temporal derivative
discover_with("2. FFT curls of ANALYTIC values (+exact dt)", Y, torch.from_numpy(G2))

surr, val_loss, _info = rb.train_vector_nestynet_surrogate(
    problem, X, Y,
    data_dir=HERE.parent.parent / "data" / "maxwell",
    num_segments=16, epochs=2500, loss_target=1e-8, batch_size=2000,
    ndata_train=4000, ndata_val=2000,
    device=torch.device("cpu"), dtype=torch.float64, verbose=False,
)
print(f"# value-only surrogate trained: val_loss={val_loss:.3e}")
with torch.no_grad():
    Y_learned = surr.forward(X).detach().to(torch.float64)
    G_analytic = surr.grad(X).detach().to(torch.float64)

discover_with("3. analytic grad of value-only surrogate", Y_learned, G_analytic)

G4 = fft_spatial_grad(Y_learned.numpy())
G4[:, :, 0] = G_exact.numpy()[:, :, 0]  # trusted temporal derivative
discover_with("4. FFT curls of LEARNED values (gem) (+exact dt)", Y_learned, torch.from_numpy(G4))
