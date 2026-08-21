#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Maxwell vector-PDE benchmark runner.

Runs 3 Maxwell problems (vacuum, wire, conductive) through a unified pipeline:
data generation -> tabulated surrogate -> discovery -> validation.

Supports two discovery engines:
- ``stlsq``: vector STLSQ (existing)
- ``factorized_search``: per-component scalar factorized symbolic search

Usage::

    python examples/Maxwell/run_benchmark.py --all --verbose
    python examples/Maxwell/run_benchmark.py --only mw000,mw001
    python examples/Maxwell/run_benchmark.py --all --fast
    python examples/Maxwell/run_benchmark.py --all --engine factorized_search --verbose
    python examples/Maxwell/run_benchmark.py --all --engine both --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Ensure examples/Maxwell is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from nestynet_sr.sr_de.system_de_search import (
    VectorSystemDESearchConfig,
    discover_vector_system_de_from_surrogate,
)
from nestynet_sr.adaptors.sobolev_residual import SobolevGradientAdaptor
from nestynet_sr.sr_core import build_initial_ast
from nestynet_sr.sr_search.config import DataHyperparams, LMHyperparams, ModelHyperparams
from nestynet_sr.sr_search.data_utils import _interleaved_row_order, build_datasets
from nestynet_sr.sr_search.model_builders import LeafBuilder, build_composite_ast
from nestynet_sr.sr_search.training import train_initial_model

from problem_defs import (
    COMPONENT_GROUND_TRUTH,
    GROUND_TRUTH,
    PROBLEM_REGISTRY,
    VectorGroundTruth,
    VectorProblemDef,
    _vec_key,
    build_problem_data,
    build_vector_terms,
)
from tabulated_surrogate import TabulatedVectorSurrogate
from spectral_derivatives import build_derivative_targets, build_spectral_spatial_hessian_diag
from conditioning_audit import apply_alias_drops, audit_vector_library, support_conditioning

# ---------------------------------------------------------------------------
# Status markers (mirror feynman_de)
# ---------------------------------------------------------------------------

STATUS_MARKERS = {
    "PASS": "OK",
    "PARTIAL": "~~",
    "FAIL": "XX",
    "ERROR": "!!",
    # Broadened-library identifiability/conditioning taxonomy:
    "FULL_RANK_STABLE": "OK",            # identifiable + well-conditioned support
    "FULL_RANK_HIGH_COHERENCE": "HC",    # identifiable but coherent decoys / ill-conditioned
    "NONIDENTIFIABLE_ALIAS": "AL",       # exact alias columns -> support correct modulo aliases
    "RANK_DEFICIENT": "RD",              # rank loss recovered by the engine min-norm fallback
}

# Soft-conditioning thresholds (textbook rules of thumb; reported regardless).
COHERENCE_MU_WARN = 0.95       # max off-diagonal feature correlation
COHERENCE_KAPPA_WARN = 30.0    # condition number of the normalized library
COHERENCE_VIF_WARN = 10.0      # variance-inflation factor of a selected coefficient


# ---------------------------------------------------------------------------
# Learned surrogate helpers
# ---------------------------------------------------------------------------


def _write_scalar_surrogate_csv(
    problem: VectorProblemDef,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    field_name: str,
    data_dir: Path,
) -> Path:
    """Write a scalar NestyNet training CSV for one Maxwell field component."""
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in str(field_name))
    path = data_dir / f"{problem.id}_{safe_name}_surrogate.csv"
    x_names = [f"x{i}" for i in range(int(X.shape[1]))]
    table = np.column_stack(
        [
            y.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False),
            X.detach().cpu().numpy().astype(np.float64, copy=False),
        ]
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header=",".join(["y", *x_names]),
        comments="",
    )
    return path


class CombinedVectorSurrogate(torch.nn.Module):
    """Wrap scalar NestyNet component surrogates as one vector-valued system."""

    def __init__(self, *surrogates: torch.nn.Module):
        super().__init__()
        self.surrogates = torch.nn.ModuleList(surrogates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parts = []
            for surr in self.surrogates:
                y = surr(x)
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                parts.append(y)
        return torch.cat(parts, dim=1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parts = []
            for surr in self.surrogates:
                g = surr.grad(x)
                if g.ndim == 2:
                    g = g.unsqueeze(1)
                parts.append(g)
        return torch.cat(parts, dim=1)

    def grad_grad(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parts = []
            for surr in self.surrogates:
                h = surr.grad_grad(x)
                if h.ndim == 3:
                    h = h.unsqueeze(1)
                parts.append(h)
        return torch.cat(parts, dim=1)


def _split_counts_for_surrogate(
    n_rows: int,
    *,
    requested_train: int,
    requested_val: int,
    requested_batch: int,
) -> tuple[int, int, int]:
    """Choose train/validation sizes compatible with NestyNet dataloaders."""
    n = int(n_rows)
    if n < 4:
        raise ValueError(f"Need at least 4 rows to train a surrogate, got {n}")

    train = min(max(1, int(requested_train)), n - 1)
    val = min(max(1, int(requested_val)), n - train)
    if val <= 0:
        val = 1
        train = n - 1
    if train + val > n:
        val = max(1, min(val, n // 3))
        train = max(1, n - val)
    batch = max(1, min(int(requested_batch), train, val))
    if batch < 2 and train >= 8:
        # A batch of 1 makes the LM assemble its global system from ~n_train
        # minibatch pieces per epoch (within-epoch memory ~ n_train/batch) and
        # can exhaust RAM.  This happens when the validation split collapses
        # (requested_val too large for the grid).  Fail loudly instead.
        raise ValueError(
            f"surrogate batch collapsed to 1 (n_rows={n}, train={train}, val={val}): "
            "reduce requested_val / requested_train so the validation split leaves a "
            "healthy batch (>=2); a batch of 1 can exhaust memory during LM training."
        )
    return int(train), int(val), int(batch)


_SOB_AXIS_NAME_TO_INDEX = {
    "t": 0,
    "time": 0,
    "x": 1,
    "y": 2,
    "z": 3,
}


def parse_sobolev_axes(raw: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(raw, (tuple, list)):
        axes = tuple(int(v) for v in raw)
    else:
        axes_list = []
        for tok in str(raw).replace(",", " ").split():
            key = tok.strip().lower()
            if not key:
                continue
            if key in _SOB_AXIS_NAME_TO_INDEX:
                axes_list.append(_SOB_AXIS_NAME_TO_INDEX[key])
            else:
                axes_list.append(int(key))
        axes = tuple(axes_list)
    if not axes:
        raise ValueError("sobolev axes must not be empty")
    if len(set(axes)) != len(axes):
        raise ValueError(f"sobolev axes must be unique, got {axes}")
    if any(a < 0 for a in axes):
        raise ValueError(f"sobolev axes must be non-negative, got {axes}")
    return axes


def _tensor_interleaved_loaders(
    X: torch.Tensor,
    Y_aug: torch.Tensor,
    *,
    n_train: int,
    n_val: int,
    batch_size: int,
) -> tuple[TensorDataset, TensorDataset, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    order = _interleaved_row_order(int(X.shape[0]), int(n_train), int(n_val))
    if len(order) < int(n_train) + int(n_val):
        raise ValueError("interleaved row order did not produce enough rows")
    train_idx = torch.as_tensor(order[: int(n_train)], device=X.device, dtype=torch.long)
    val_idx = torch.as_tensor(
        order[int(n_train) : int(n_train) + int(n_val)],
        device=X.device,
        dtype=torch.long,
    )
    ds_tr = TensorDataset(X.index_select(0, train_idx), Y_aug.index_select(0, train_idx))
    ds_va = TensorDataset(X.index_select(0, val_idx), Y_aug.index_select(0, val_idx))
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=False, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=int(batch_size), shuffle=False, drop_last=True)
    return ds_tr, ds_va, dl_tr, dl_va, train_idx, val_idx


def _rms_float(t: torch.Tensor, eps: float = 1e-12) -> float:
    if int(t.numel()) == 0:
        return float(eps)
    return float(torch.sqrt(torch.mean(t.detach().to(torch.float64) ** 2)).item())


def make_sobolev_channel_scales(
    y_component: torch.Tensor,
    g_component: torch.Tensor,
    axes: tuple[int, ...],
    normalize: str = "rms",
    zero_tol: float = 1e-8,
    eps: float = 1e-12,
) -> tuple[float, torch.Tensor]:
    """Return value and per-axis derivative scales for Sobolev residuals."""
    mode = str(normalize).strip().lower()
    if mode == "none":
        return 1.0, torch.ones(len(axes), dtype=y_component.dtype, device=y_component.device)
    if mode != "rms":
        raise ValueError(f"unsupported Sobolev normalization {normalize!r}")

    value_rms = max(_rms_float(y_component, eps), float(eps))
    selected = g_component[:, list(axes)].detach().to(torch.float64)
    axis_rms = torch.sqrt(torch.mean(selected * selected, dim=0)).to(
        device=y_component.device,
        dtype=y_component.dtype,
    )
    deriv_ref = float(torch.sqrt(torch.mean(selected * selected)).item()) if int(selected.numel()) else 1.0
    deriv_ref = max(deriv_ref, float(eps))

    denoms = []
    for val in axis_rms:
        v = float(val.item())
        if v <= float(zero_tol) * deriv_ref:
            v = deriv_ref
        denoms.append(max(v, float(eps)))
    grad_scales = torch.as_tensor(
        [1.0 / d for d in denoms],
        dtype=y_component.dtype,
        device=y_component.device,
    )
    return 1.0 / value_rms, grad_scales


def make_hess_channel_scales(
    h_component: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    normalize: str = "rms",
    zero_tol: float = 1e-8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-pair scales for the H^2 residual (mirrors the gradient channel scales).

    ``h_component`` is the ``(N, D, D)`` Hessian target for one field; ``pairs`` are
    the second-derivative axis pairs being supervised (e.g. spatial diagonals for
    the Laplacian).  ``rms`` mode normalizes each channel by its own RMS so the
    second-derivative residual is comparable in magnitude to the value/gradient
    channels (the same noise-aware-by-scale philosophy as ``grad_scales``)."""
    mode = str(normalize).strip().lower()
    n = len(pairs)
    if mode == "none" or n == 0:
        return torch.ones(n, dtype=h_component.dtype, device=h_component.device)
    if mode != "rms":
        raise ValueError(f"unsupported Hessian normalization {normalize!r}")
    sel = torch.stack(
        [h_component[:, int(a), int(b)].to(torch.float64) for (a, b) in pairs], dim=1
    )
    chan_rms = torch.sqrt(torch.mean(sel * sel, dim=0))
    ref = float(torch.sqrt(torch.mean(sel * sel)).item()) if int(sel.numel()) else 1.0
    ref = max(ref, float(eps))
    denoms = []
    for v in chan_rms.tolist():
        v = float(v)
        if v <= float(zero_tol) * ref:
            v = ref
        denoms.append(max(v, float(eps)))
    return torch.as_tensor(
        [1.0 / d for d in denoms], dtype=h_component.dtype, device=h_component.device
    )


def _component_sobolev_diagnostics(
    surrogate_i: torch.nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    G_target: torch.Tensor,
    G_exact: torch.Tensor | None,
    *,
    out_idx: int,
    axes: tuple[int, ...],
    val_idx: torch.Tensor,
) -> dict[str, Any]:
    x_val = X.index_select(0, val_idx)
    y_val = Y.index_select(0, val_idx)[:, out_idx]
    gt_val = G_target.index_select(0, val_idx)[:, out_idx, :]
    with torch.no_grad():
        pred_val = surrogate_i(x_val).reshape(-1)
        pred_grad = surrogate_i.grad(x_val)
        if pred_grad.ndim == 3:
            pred_grad = pred_grad[:, 0, :]
    value_val_mse = float(torch.mean((pred_val - y_val) ** 2).item())
    gdiff_target = pred_grad[:, list(axes)] - gt_val[:, list(axes)]
    grad_vs_target = torch.sqrt(torch.mean(gdiff_target * gdiff_target, dim=0))

    cells = []
    axis_names = ["t", "x", "y", "z"]
    for k, axis in enumerate(axes):
        cells.append(
            {
                "axis": axis_names[axis] if axis < len(axis_names) else str(axis),
                "abs_rms": float(grad_vs_target[k].item()),
            }
        )
    cells.sort(key=lambda d: d["abs_rms"], reverse=True)
    diag: dict[str, Any] = {
        "value_val_mse": value_val_mse,
        "grad_vs_target_abs_rms": float(torch.sqrt(torch.mean(gdiff_target * gdiff_target)).item()),
        "worst_grad_channel": cells[0] if cells else None,
    }
    if G_exact is not None:
        ge_val = G_exact.index_select(0, val_idx)[:, out_idx, :]
        gdiff_exact = pred_grad[:, list(axes)] - ge_val[:, list(axes)]
        diag["grad_vs_exact_abs_rms"] = float(
            torch.sqrt(torch.mean(gdiff_exact * gdiff_exact)).item()
        )
    return diag


def _train_single_component(
    out_idx: int,
    *,
    problem: VectorProblemDef,
    X: torch.Tensor,
    Y: torch.Tensor,
    G_target: torch.Tensor | None,
    G_exact: torch.Tensor | None,
    data_dir: Path,
    num_segments: int,
    epochs: int,
    loss_target: float,
    batch_size: int,
    ndata_train: int,
    ndata_val: int,
    device: torch.device,
    dtype: torch.dtype,
    objective: str,
    axes: tuple[int, ...],
    sobolev_target: str,
    sobolev_value_weight: float,
    sobolev_grad_weight: float,
    sobolev_normalize: str,
    H_target: torch.Tensor | None = None,
    hess_pairs: list[tuple[int, int]] | None = None,
    hess_weight: float = 0.0,
    hess_normalize: str = "rms",
    hess_trace_dims: list[int] | None = None,
    canonical_init: bool | None = None,
    grad_weight_ramp: list[float] | None = None,
    init_seed: int | None = None,
    verbose: bool = False,
) -> tuple[torch.nn.Module, float, dict[str, Any]]:
    """Train one scalar field-component surrogate (value or Sobolev objective).

    Self-contained (rebuilds its own dataloaders/model/leaf-builder) so it can run
    in-process or inside a worker subprocess.
    """
    n_train, n_val, batch = _split_counts_for_surrogate(
        int(X.shape[0]),
        requested_train=int(ndata_train),
        requested_val=int(ndata_val),
        requested_batch=int(batch_size),
    )
    data_hp = DataHyperparams(
        batch_size=batch,
        ndata_select=n_train,
        ndata_select_val=n_val,
        data_split_strategy="interleaved",
    )
    model_hp = ModelHyperparams(
        double_precision=(dtype == torch.float64),
        repeatable_runs=True,
        model_base_name="G_Model",
        num_segments_min=int(num_segments),
        num_segments_max=int(num_segments),
        Nout_size=1,
    )
    X_fit_all = X.to(device=device, dtype=dtype)
    Y_fit_all = Y.to(device=device, dtype=dtype)
    G_target_fit = G_target.to(device=device, dtype=dtype) if G_target is not None else None
    G_exact_fit = G_exact.to(device=device, dtype=dtype) if G_exact is not None else None
    H_target_fit = H_target.to(device=device, dtype=dtype) if H_target is not None else None
    # H^2 (curvature) supervision is active only when a target, channels, and a
    # positive weight are all present; otherwise the surrogate stays H^1 (the
    # established value+gradient Sobolev objective) and behaviour is unchanged.
    # Two mutually-exclusive modes: per-pair diagonals, or a single Laplacian TRACE
    # (Σ_a ∂²/∂x_a²) -- the trace avoids the noise-amplified structurally-zero
    # channels and supervises exactly the ∇² decoy quantity.
    pairs = [(int(a), int(b)) for (a, b) in (hess_pairs or ())]
    trace_dims = [int(a) for a in (hess_trace_dims or ())]
    if pairs and trace_dims:
        raise ValueError("hess_pairs and hess_trace_dims are mutually exclusive")
    use_hess = (
        objective == "sobolev"
        and H_target_fit is not None
        and (len(pairs) > 0 or len(trace_dims) > 0)
        and float(hess_weight) > 0.0
    )

    field_name = (
        problem.field_names[out_idx]
        if out_idx < len(problem.field_names)
        else f"y{out_idx}"
    )
    # Deterministic restart control: the segmented init draws from the global
    # torch RNG, so seeding here makes each restart's random init reproducible
    # and distinct (used by the canonical-then-random multi-start policy).
    if init_seed is not None:
        torch.manual_seed(int(init_seed))
    leaf_builder = LeafBuilder(model_hp, device, dtype)
    use_dual_layer = True
    ast0 = build_initial_ast(
        Nxvars=int(X.shape[1]),
        num_segments=int(num_segments),
        dual_layer=use_dual_layer,
        tag=f"A{out_idx}",
    )
    surrogate_i, nparam, _ = build_composite_ast(
        ast0,
        int(num_segments),
        dual_layer=use_dual_layer,
        leaf_builder=leaf_builder,
        device=device,
        dtype=dtype,
    )

    fit_provider: torch.nn.Module
    component_entry: dict[str, Any] = {
        "field": str(field_name),
        "objective": objective,
        "dual_layer": bool(use_dual_layer),
        "n_parameters": int(nparam),
    }
    val_idx_for_diag: torch.Tensor | None = None
    if objective == "value":
        csv_path = _write_scalar_surrogate_csv(
            problem, X, Y[:, out_idx], field_name=field_name, data_dir=data_dir,
        )
        ds_tr, ds_va, dl_tr, dl_va = build_datasets(
            str(csv_path), int(X.shape[1]), np.float64, data_hp, None,
        )
        if dl_tr is None or dl_va is None or ds_tr is None or ds_va is None:
            raise RuntimeError(f"Failed to build Maxwell surrogate datasets for {field_name}")
        fit_provider = surrogate_i
        component_entry["csv_path"] = str(csv_path)
    else:
        assert G_target_fit is not None
        value_scale, grad_scales = make_sobolev_channel_scales(
            Y_fit_all[:, out_idx], G_target_fit[:, out_idx, :], axes, normalize=sobolev_normalize,
        )
        hess_scales = None
        aug_cols = [Y_fit_all[:, out_idx : out_idx + 1], G_target_fit[:, out_idx, list(axes)]]
        if use_hess:
            assert H_target_fit is not None
            if trace_dims:
                # Single Laplacian-trace channel: target = Σ_a H_target[a,a],
                # scaled by 1/RMS(trace) (no per-axis noise amplification).
                trace_tgt = sum(H_target_fit[:, out_idx, a, a] for a in trace_dims).reshape(-1, 1)
                tr_rms = max(float(torch.sqrt(torch.mean(trace_tgt.double() ** 2)).item()), 1e-12)
                hess_scales = torch.as_tensor([1.0 / tr_rms], dtype=dtype, device=device)
                aug_cols.append(trace_tgt)
            else:
                hess_scales = make_hess_channel_scales(
                    H_target_fit[:, out_idx, :, :], pairs, normalize=hess_normalize,
                )
                aug_cols.append(
                    torch.stack([H_target_fit[:, out_idx, a, b] for (a, b) in pairs], dim=1)
                )
        y_aug = torch.cat(aug_cols, dim=1)
        ds_tr, ds_va, dl_tr, dl_va, _train_idx, val_idx_for_diag = _tensor_interleaved_loaders(
            X_fit_all, y_aug, n_train=n_train, n_val=n_val, batch_size=batch,
        )
        fit_provider = SobolevGradientAdaptor(
            surrogate_i,
            axes=axes,
            value_weight=float(sobolev_value_weight),
            grad_weight=float(sobolev_grad_weight),
            value_scale=float(value_scale),
            grad_scales=grad_scales,
            hess_pairs=(pairs if (use_hess and not trace_dims) else None),
            hess_trace_dims=(trace_dims if (use_hess and trace_dims) else None),
            hess_weight=float(hess_weight) if use_hess else 0.0,
            hess_scales=hess_scales,
        )
        component_entry.update(
            {
                "sobolev_target": str(sobolev_target),
                "sobolev_axes": list(axes),
                "sobolev_value_weight": float(sobolev_value_weight),
                "sobolev_grad_weight": float(sobolev_grad_weight),
                "sobolev_normalize": str(sobolev_normalize),
                "value_scale": float(value_scale),
                "grad_scales": [float(v) for v in grad_scales.detach().cpu().tolist()],
            }
        )
        if use_hess:
            component_entry.update(
                {
                    "hess_pairs": [list(p) for p in pairs],
                    "hess_weight": float(hess_weight),
                    "hess_normalize": str(hess_normalize),
                    "hess_scales": [float(v) for v in hess_scales.detach().cpu().tolist()],
                }
            )

    if verbose:
        print(
            f"  Training NestyNet surrogate for {field_name} ({objective}): "
            f"rows={int(X.shape[0])}, train={n_train}, val={n_val}, batch={batch}, params={nparam}"
        )

    use_canonical = (objective == "value") if canonical_init is None else bool(canonical_init)
    # grad_weight schedule.  A ramp (Sobolev only) keeps the deterministic value
    # basin from canonical init and eases the derivative jet in over stages, so
    # full-weight gradient matching does not kick the LM into a divergent ravine.
    if objective == "sobolev" and grad_weight_ramp:
        gw_stages: list[float | None] = [float(w) for w in grad_weight_ramp]
    elif objective == "sobolev":
        gw_stages = [float(sobolev_grad_weight)]
    else:
        gw_stages = [None]  # value path: fit_provider is the bare surrogate

    # All-diagonal H^2 (the Laplacian's spatial ∂²/∂x_a² entries -- our case) has a
    # fast dense parameter-Jacobian via the leaf's analytic selected-diagonal route,
    # so direct_solve stays fast and we keep the established, precise strategy.  Only
    # OFF-diagonal H^2 (no dense fast path) would fall to the per-parameter jvp loop,
    # for which the Jacobian-free matfree (PCG) strategy is used instead.
    hess_offdiag = use_hess and not all(int(a) == int(b) for (a, b) in pairs)
    lm_strategy = "matfree" if hess_offdiag else "direct_solve"

    best_val = float("nan")
    best_p = None
    lm_opt = None
    gw_final = float(gw_stages[-1]) if (objective == "sobolev" and gw_stages[-1]) else 0.0
    for stage, gw in enumerate(gw_stages):
        if objective == "sobolev":
            # Continuation for the curvature term: ease H^2 in proportionally with
            # the gradient weight (off while gw=0 pure-value, full once gw reaches
            # its final value) so the noisier 2nd-derivative target enters last.
            hw_stage = (
                float(hess_weight) * (float(gw) / gw_final)
                if (use_hess and gw_final > 0.0)
                else 0.0
            )
            fit_provider = SobolevGradientAdaptor(
                surrogate_i,
                axes=axes,
                value_weight=float(sobolev_value_weight),
                grad_weight=float(gw),
                value_scale=value_scale,
                grad_scales=grad_scales,
                hess_pairs=(pairs if (use_hess and not trace_dims) else None),
                hess_trace_dims=(trace_dims if (use_hess and trace_dims) else None),
                hess_weight=hw_stage,
                hess_scales=hess_scales,
            )
        canon_stage = use_canonical and (stage == 0)  # canonical init only seeds stage 0
        if verbose and len(gw_stages) > 1:
            print(
                f"  [ramp] {field_name} stage {stage + 1}/{len(gw_stages)}: "
                f"grad_weight={gw:g}, hess_weight={hw_stage if objective == 'sobolev' else 0.0:g}, "
                f"canonical_init={canon_stage}"
            )
        best_val, _best_train, best_p, lm_opt = train_initial_model(
            fit_provider,
            dl_tr,
            dl_va,
            epochs=int(epochs),
            LM_strategy=lm_strategy,
            nval_patience=250,
            loss_target=float(loss_target),
            epochs_min=max(1, min(300, int(epochs))),
            chisq_tol=1e-10,
            device=device,
            lm_hp=LMHyperparams(
                strategy=lm_strategy,
                epochs=int(epochs),
                nval_patience=250,
                loss_target=float(loss_target),
                chisq_tol=1e-10,
                canonical_init=canon_stage,
            ),
        )
        lm_opt._update_param_groups(best_p)

    surrogate_i.eval()
    component_entry["val_loss"] = float(best_val)
    if objective == "sobolev" and val_idx_for_diag is not None and G_target_fit is not None:
        component_entry["sobolev_val_loss"] = float(best_val)
        component_entry.update(
            _component_sobolev_diagnostics(
                surrogate_i, X_fit_all, Y_fit_all, G_target_fit, G_exact_fit,
                out_idx=out_idx, axes=axes, val_idx=val_idx_for_diag,
            )
        )
    return surrogate_i, float(best_val), component_entry


def _rebuild_component(
    out_idx: int, nxvars: int, num_segments: int,
    device: torch.device, dtype: torch.dtype, state_dict: dict,
) -> torch.nn.Module:
    """Reconstruct a trained component surrogate from a worker's state_dict."""
    model_hp = ModelHyperparams(
        double_precision=(dtype == torch.float64),
        repeatable_runs=True,
        model_base_name="G_Model",
        num_segments_min=int(num_segments),
        num_segments_max=int(num_segments),
        Nout_size=1,
    )
    leaf_builder = LeafBuilder(model_hp, device, dtype)
    ast0 = build_initial_ast(
        Nxvars=int(nxvars), num_segments=int(num_segments), dual_layer=True, tag=f"A{out_idx}",
    )
    surrogate_i, _nparam, _ = build_composite_ast(
        ast0, int(num_segments), dual_layer=True, leaf_builder=leaf_builder,
        device=device, dtype=dtype,
    )
    surrogate_i.load_state_dict(state_dict)
    surrogate_i.eval()
    return surrogate_i


_JOB_OVERRIDE_KEYS = ("canonical_init", "init_seed", "grad_weight_ramp", "epochs")


def _run_jobs(
    problem: VectorProblemDef,
    X: torch.Tensor,
    Y: torch.Tensor,
    G_target: torch.Tensor | None,
    G_exact: torch.Tensor | None,
    cfg: dict[str, Any],
    jobs: list[dict[str, Any]],
    workers: int,
    device: torch.device,
    dtype: torch.dtype,
    H_target: torch.Tensor | None = None,
) -> list[tuple[int, torch.nn.Module, float, dict[str, Any]]]:
    """Train a list of jobs and return ``(component, surrogate, best_val, entry)``.

    Each job is ``{"component": i, <override>...}`` where overrides
    (``canonical_init``, ``init_seed``, ``grad_weight_ramp``, ``epochs``) replace
    the corresponding ``cfg`` defaults for that job only.  This serves both the
    first wave (one job per component, no overrides) and the restart rounds
    (failed components re-fit from random seeds).  Parallel across jobs when
    ``workers > 1`` (one process per job); otherwise in-process and sequential."""
    if int(workers) > 1 and len(jobs) > 1:
        return _run_jobs_subprocess(
            problem, X, Y, G_target, G_exact, cfg, jobs, workers, device, dtype,
            H_target=H_target,
        )
    base = {k: v for k, v in cfg.items() if k != "dtype_str"}
    base["data_dir"] = Path(cfg["data_dir"])
    base["axes"] = tuple(cfg["axes"])
    out: list[tuple[int, torch.nn.Module, float, dict[str, Any]]] = []
    for job in jobs:
        kw = dict(base)
        for key in _JOB_OVERRIDE_KEYS:
            if key in job:
                kw[key] = job[key]
        comp = int(job["component"])
        s, bv, e = _train_single_component(
            comp, problem=problem, X=X, Y=Y, G_target=G_target, G_exact=G_exact,
            H_target=H_target, device=device, dtype=dtype, **kw,
        )
        out.append((comp, s, float(bv), e))
    return out


def _run_jobs_subprocess(
    problem: VectorProblemDef,
    X: torch.Tensor,
    Y: torch.Tensor,
    G_target: torch.Tensor | None,
    G_exact: torch.Tensor | None,
    cfg: dict[str, Any],
    jobs: list[dict[str, Any]],
    workers: int,
    device: torch.device,
    dtype: torch.dtype,
    H_target: torch.Tensor | None = None,
) -> list[tuple[int, torch.nn.Module, float, dict[str, Any]]]:
    """Run ``jobs`` in N worker processes (true multi-core; sidesteps the GIL).
    Each worker trains one job (a component, with optional init overrides) and
    writes back a state_dict; the parent rebuilds and assembles them."""
    worker_script = Path(__file__).resolve().parent / "component_worker.py"
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    results: list[tuple[int, torch.nn.Module, float, dict[str, Any]]] = []
    with tempfile.TemporaryDirectory(prefix="maxwell_comp_") as td:
        tdp = Path(td)
        bundle = tdp / "bundle.pt"
        torch.save(
            {
                "problem_id": problem.id,
                "X": X.detach().cpu(),
                "Y": Y.detach().cpu(),
                "G_target": None if G_target is None else G_target.detach().cpu(),
                "G_exact": None if G_exact is None else G_exact.detach().cpu(),
                "H_target": None if H_target is None else H_target.detach().cpu(),
                "cfg": cfg,
            },
            bundle,
        )

        def _run_one(j: int) -> tuple[int, int, Path]:
            job = jobs[j]
            i = int(job["component"])
            out = tdp / f"job_{j}.pt"
            cmd = [sys.executable, str(worker_script),
                   "--bundle", str(bundle), "--out_idx", str(i), "--result", str(out)]
            if "canonical_init" in job:
                ci = job["canonical_init"]
                cmd += ["--canonical_init", "none" if ci is None else str(int(bool(ci)))]
            if job.get("init_seed") is not None:
                cmd += ["--init_seed", str(int(job["init_seed"]))]
            if "grad_weight_ramp" in job:
                gw = job["grad_weight_ramp"]
                cmd += ["--grad_weight_ramp",
                        "none" if gw is None else ",".join(repr(float(x)) for x in gw)]
            if job.get("epochs") is not None:
                cmd += ["--epochs", str(int(job["epochs"]))]
            subprocess.run(cmd, check=True, env=env, cwd=str(worker_script.parent))
            return j, i, out

        with ThreadPoolExecutor(max_workers=min(int(workers), len(jobs))) as ex:
            launched = sorted(ex.map(_run_one, range(len(jobs))))

        nxvars = int(X.shape[1])
        num_segments = int(cfg["num_segments"])
        for _j, i, out in launched:
            blob = torch.load(out, weights_only=False)
            surr = _rebuild_component(i, nxvars, num_segments, device, dtype, blob["state_dict"])
            results.append((i, surr, float(blob["best_val"]), blob["entry"]))
    return results


def _restart_failed_components(
    problem: VectorProblemDef,
    X: torch.Tensor,
    Y: torch.Tensor,
    G_target: torch.Tensor | None,
    G_exact: torch.Tensor | None,
    cfg: dict[str, Any],
    results: list[tuple[torch.nn.Module, float, dict[str, Any]] | None],
    *,
    restart_max: int,
    restart_factor: float,
    restart_epochs: int,
    workers: int,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool,
    H_target: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    """Random-restart the components that failed relative to their siblings.

    A component "fails" if its first-wave validation loss is >= ``restart_factor``
    times the median of all components -- an unambiguous outlier at any noise
    level (the median tracks the achievable floor, which rises with noise).  Only
    failed components are re-fit, from random seeds (``canonical_init=False``,
    single-stage ``gw=1`` -- the validated config).  All candidate seeds for all
    failed components are launched CONCURRENTLY (single-thread worker processes,
    pooled at ``workers``) rather than one component/round at a time -- this uses
    the otherwise-idle cores.  Since every seed runs to completion in the wave,
    selection is the BEST (lowest-validation-loss) attempt -- there is no early
    stop to preserve, so keeping the best beats keeping the first passer.  This is
    legitimate global optimization: selection is by the surrogate's own held-out
    Sobolev validation loss, blind to the downstream discovery.  The component is
    only overwritten if the chosen restart improves on the first wave.  ``results``
    is mutated in-place.  Returns a diagnostics dict (or ``None`` when restarts are
    disabled / not applicable)."""
    n = len(results)
    if int(restart_max) <= 0 or n <= 1:
        return None
    vals = [float(results[i][1]) for i in range(n)]  # type: ignore[index]
    finite = sorted(v for v in vals if math.isfinite(v) and v > 0.0)
    if not finite:
        return None
    med = statistics.median(finite)
    thresh = float(restart_factor) * med
    failed = [i for i in range(n) if (not math.isfinite(vals[i])) or vals[i] > thresh]
    if not failed:
        print(f"  [restart] median={med:.3e} thresh={thresh:.3e} (x{restart_factor:g}); "
              f"no sibling outliers -> no restart", flush=True)
        return {"median_val": med, "threshold": thresh, "failed_components": [], "attempts": []}
    flagged = ", ".join(f"{i}:{vals[i]:.2e}" for i in failed)
    print(f"  [restart] median={med:.3e} thresh={thresh:.3e} (x{restart_factor:g})  "
          f"failed=[{flagged}]; launching {len(failed)}x{int(restart_max)} seed attempts in parallel",
          flush=True)
    # Build all (failed component x candidate seed) jobs; run concurrently.
    seeds = list(range(int(restart_max)))
    jobs = [{"component": int(i), "init_seed": int(s), "canonical_init": False,
             "grad_weight_ramp": None, "epochs": int(restart_epochs)}
            for i in failed for s in seeds]
    ran = _run_jobs(problem, X, Y, G_target, G_exact, cfg, jobs, workers, device, dtype,
                    H_target=H_target)
    by_comp: dict[int, list[tuple[int, Any, float, dict[str, Any]]]] = {int(i): [] for i in failed}
    attempts: list[dict[str, Any]] = []
    for job, (comp, surr, bv, entry) in zip(jobs, ran):
        s = int(job["init_seed"])
        by_comp[int(comp)].append((s, surr, float(bv), entry))
        attempts.append({"component": int(comp), "seed": s, "best_val": float(bv),
                         "passed": bool(math.isfinite(bv) and bv <= thresh)})
    resolved: list[dict[str, Any]] = []
    for i in failed:
        # Best-of-N by held-out validation loss (all seeds already ran in parallel).
        chosen = min(by_comp[int(i)], key=lambda t: t[2] if math.isfinite(t[2]) else math.inf)
        if chosen[2] < results[i][1]:  # type: ignore[index]  # keep only if it improves on the first wave
            results[i] = (chosen[1], chosen[2], chosen[3])
        if math.isfinite(results[i][1]) and results[i][1] <= thresh:  # type: ignore[index]
            resolved.append({"component": int(i), "seed": int(chosen[0]),
                             "best_val": float(results[i][1])})  # type: ignore[index]
    if verbose:
        for a in attempts:
            print(f"    [restart] comp={a['component']} seed={a['seed']} "
                  f"val={a['best_val']:.3e} {'PASS' if a['passed'] else 'fail'}", flush=True)
    unresolved = [int(i) for i in failed
                  if not (math.isfinite(results[i][1]) and results[i][1] <= thresh)]  # type: ignore[index]
    res_str = ", ".join(f"{r['component']}:seed{r['seed']}={r['best_val']:.2e}" for r in resolved) or "none"
    print(f"  [restart] resolved=[{res_str}]  unresolved={unresolved}", flush=True)
    return {"median_val": med, "threshold": thresh,
            "failed_components": [int(i) for i in failed],
            "resolved": resolved, "unresolved_components": unresolved,
            "attempts": attempts}


def train_vector_nestynet_surrogate(
    problem: VectorProblemDef,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    data_dir: Path,
    num_segments: int,
    epochs: int,
    loss_target: float,
    batch_size: int,
    ndata_train: int,
    ndata_val: int,
    device: torch.device,
    dtype: torch.dtype,
    surrogate_objective: str = "value",
    sobolev_target: str = "spectral_spatial_exact_time",
    sobolev_axes: tuple[int, ...] = (0, 1, 2, 3),
    sobolev_value_weight: float = 1.0,
    sobolev_grad_weight: float = 1.0,
    sobolev_normalize: str = "rms",
    G_target: torch.Tensor | None = None,
    G_exact: torch.Tensor | None = None,
    H_target: torch.Tensor | None = None,
    hess_pairs: list[tuple[int, int]] | None = None,
    hess_weight: float = 0.0,
    hess_normalize: str = "rms",
    component_workers: int = 1,
    canonical_init: bool | None = None,
    grad_weight_ramp: list[float] | None = None,
    restart_max: int = 0,
    restart_factor: float = 50.0,
    restart_epochs: int = 1500,
    verbose: bool = False,
) -> tuple[torch.nn.Module, float, dict[str, Any]]:
    """Train scalar NestyNet surrogates and wrap them as a vector system.

    The six field components are independent fits; ``component_workers>1`` trains
    them concurrently in threads (each builds its own ``leaf_builder``/model, so
    no mutable state is shared).  Results are order-stable and, for the
    deterministic (canonical-init) value objective, bit-identical to the
    sequential path.
    """
    objective = str(surrogate_objective).strip().lower()
    if objective not in {"value", "sobolev"}:
        raise ValueError(f"unsupported surrogate_objective={surrogate_objective!r}")
    axes = parse_sobolev_axes(sobolev_axes)
    if objective == "sobolev":
        if G_target is None:
            raise ValueError("surrogate_objective='sobolev' requires G_target")
        if G_target.shape[:2] != Y.shape[:2] or int(G_target.shape[2]) < int(X.shape[1]):
            raise ValueError(
                f"G_target shape {tuple(G_target.shape)} incompatible with "
                f"X={tuple(X.shape)} Y={tuple(Y.shape)}"
            )
        if max(axes) >= int(X.shape[1]):
            raise ValueError(f"sobolev axis {max(axes)} outside X dimension {int(X.shape[1])}")

    cfg: dict[str, Any] = {
        "data_dir": str(data_dir),
        "num_segments": int(num_segments),
        "epochs": int(epochs),
        "loss_target": float(loss_target),
        "batch_size": int(batch_size),
        "ndata_train": int(ndata_train),
        "ndata_val": int(ndata_val),
        "objective": objective,
        "axes": list(axes),
        "sobolev_target": str(sobolev_target),
        "sobolev_value_weight": float(sobolev_value_weight),
        "sobolev_grad_weight": float(sobolev_grad_weight),
        "sobolev_normalize": str(sobolev_normalize),
        "hess_pairs": [list(p) for p in hess_pairs] if hess_pairs else None,
        "hess_weight": float(hess_weight),
        "hess_normalize": str(hess_normalize),
        "canonical_init": canonical_init,
        "grad_weight_ramp": list(grad_weight_ramp) if grad_weight_ramp else None,
        "verbose": bool(verbose),
        "dtype_str": "float64" if dtype == torch.float64 else "float32",
    }
    n_components = int(Y.shape[1])
    workers = max(1, int(component_workers))

    # First wave: one job per component, using the configured init (canonical +
    # ramp).  Parallel across components when workers > 1.
    first = _run_jobs(
        problem, X, Y, G_target, G_exact, cfg,
        [{"component": i} for i in range(n_components)], workers, device, dtype,
        H_target=H_target,
    )
    results: list[tuple[torch.nn.Module, float, dict[str, Any]] | None] = [None] * n_components
    for comp, surr, bv, entry in first:
        results[comp] = (surr, bv, entry)

    # Restart rounds: re-fit only the components that failed *relative to their
    # siblings* (val >= restart_factor x median) from random seeds (single-stage
    # gw=1, the validated config), in parallel across the failed set; keep the
    # best surrogate per component and stop once it is back in the sibling range.
    restart_info = _restart_failed_components(
        problem, X, Y, G_target, G_exact, cfg, results,
        restart_max=int(restart_max), restart_factor=float(restart_factor),
        restart_epochs=int(restart_epochs), workers=workers,
        device=device, dtype=dtype, verbose=bool(verbose), H_target=H_target,
    )

    results = [r for r in results]  # all slots now filled
    surrogates = [r[0] for r in results]
    component_losses = [r[1] for r in results]
    component_info = [r[2] for r in results]

    n_train, n_val, batch = _split_counts_for_surrogate(
        int(X.shape[0]),
        requested_train=int(ndata_train),
        requested_val=int(ndata_val),
        requested_batch=int(batch_size),
    )
    surrogate = CombinedVectorSurrogate(*surrogates)
    info = {
        "component_surrogates": component_info,
        "surrogate_objective": objective,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "batch_size": int(batch),
        "num_segments": int(num_segments),
        "epochs": int(epochs),
        "n_parameters": int(sum(int(c.get("n_parameters", 0)) for c in component_info)),
        "component_workers": int(workers),
        "restart": restart_info,
    }
    return surrogate, float(max(component_losses, default=float("nan"))), info


# ---------------------------------------------------------------------------
# factorized symbolic search helpers
# ---------------------------------------------------------------------------


# Feature column definitions for the per-component feature table.
# Each entry: (name, source, idx1, idx2) where source is "G" (gradient) or
# "Y" (field value).  For "G" the value is G[:, idx1, idx2]; for "Y" it is
# Y[:, idx1] (idx2 is ignored).
_BASE_FEATURE_DEFS: list[tuple[str, str, int, int]] = [
    # B spatial cross-derivatives
    ("dBx_dy", "G", 3, 2),
    ("dBx_dz", "G", 3, 3),
    ("dBy_dx", "G", 4, 1),
    ("dBy_dz", "G", 4, 3),
    ("dBz_dx", "G", 5, 1),
    ("dBz_dy", "G", 5, 2),
    # E spatial cross-derivatives
    ("dEx_dy", "G", 0, 2),
    ("dEx_dz", "G", 0, 3),
    ("dEy_dx", "G", 1, 1),
    ("dEy_dz", "G", 1, 3),
    ("dEz_dx", "G", 2, 1),
    ("dEz_dy", "G", 2, 2),
    # Field values
    ("Ex", "Y", 0, 0),
    ("Ey", "Y", 1, 0),
    ("Ez", "Y", 2, 0),
    ("Bx", "Y", 3, 0),
    ("By", "Y", 4, 0),
    ("Bz", "Y", 5, 0),
]

_J_FEATURE_DEFS: list[tuple[str, str, int, int]] = [
    ("Jx", "Y", 6, 0),
    ("Jy", "Y", 7, 0),
    ("Jz", "Y", 8, 0),
]

# Target definitions: (component_key, field_idx_for_time_deriv)
_TARGET_DEFS: list[tuple[str, int]] = [
    ("ampere_x", 0),   # dEx/dt
    ("ampere_y", 1),   # dEy/dt
    ("ampere_z", 2),   # dEz/dt
    ("faraday_x", 3),  # dBx/dt
    ("faraday_y", 4),  # dBy/dt
    ("faraday_z", 5),  # dBz/dt
]


def build_component_features(
    G: torch.Tensor,
    Y: torch.Tensor,
    *,
    has_J: bool = False,
    seed: int = 42,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
    """Build per-component (x_fit, y_fit, x_probe, y_probe, feature_names).

    The full feature matrix and targets are built from G and Y, then split
    70/30 into fit and probe sets.

    Returns dict mapping component_key -> (x_fit, y_fit, x_probe, y_probe, names).
    """
    fdefs = list(_BASE_FEATURE_DEFS)
    if has_J:
        fdefs.extend(_J_FEATURE_DEFS)

    names = [d[0] for d in fdefs]
    cols = []
    for _name, src, i1, i2 in fdefs:
        if src == "G":
            cols.append(G[:, i1, i2])
        else:
            cols.append(Y[:, i1])
    features = torch.stack(cols, dim=1)  # (N, n_features)

    # 70/30 random split
    N = features.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=gen)
    n_fit = max(1, int(0.7 * N))
    fit_idx = perm[:n_fit]
    probe_idx = perm[n_fit:]

    x_fit = features[fit_idx]
    x_probe = features[probe_idx]

    result = {}
    for key, field_idx in _TARGET_DEFS:
        target = G[:, field_idx, 0]  # time derivative
        y_fit = target[fit_idx].unsqueeze(-1)
        y_probe = target[probe_idx].unsqueeze(-1)
        result[key] = (x_fit, y_fit, x_probe, y_probe, names)

    return result


def _node_str_named(node, names: list[str]) -> str:
    """Like explorer.node_str but substitutes feature names for variables."""
    from nestynet_sr.sr_search.factorized_search.explorer import node_str

    s = node_str(node)
    # Replace in reverse index order to avoid x1 matching inside x10
    for i in range(len(names) - 1, -1, -1):
        s = s.replace(f"x{i}", names[i])
    return s


def run_problem_factorized_search(
    problem: VectorProblemDef,
    *,
    fast: bool = False,
    skip_generate: bool = False,
    data_dir: Path,
    results_dir: Path,
    max_points: int = 25000,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run factorized symbolic search discovery for one Maxwell problem.

    Decomposes the vector PDE into per-component scalar regressions and runs
    factorized symbolic search on each component independently.
    """
    from nestynet_sr.sr_search.factorized_search.explorer import (
        eval_mapping,
        eval_node,
        node_str,
        run_explorer_core,
    )

    result_entry: dict[str, Any] = {
        "id": problem.id,
        "description": problem.description,
        "engine": "factorized_search",
        "status": "ERROR",
        "message": "",
    }

    comp_gt_list = COMPONENT_GROUND_TRUTH.get(problem.id)
    if comp_gt_list is None:
        result_entry["message"] = f"No component ground truth for {problem.id}"
        return result_entry

    # 1. Generate or load data (reuse same logic as STLSQ path)
    npz_path = data_dir / f"{problem.id}.npz"
    meta_path = data_dir / f"{problem.id}.meta.json"

    if skip_generate and npz_path.exists():
        if verbose:
            print(f"  [factorized_search] Loading existing data: {npz_path}")
        blob = np.load(npz_path)
        X = torch.from_numpy(np.asarray(blob["X"], dtype=np.float64))
        Y = torch.from_numpy(np.asarray(blob["Y"], dtype=np.float64))
        G = torch.from_numpy(np.asarray(blob["G"], dtype=np.float64))
    else:
        if verbose:
            print(f"  [factorized_search] Generating data (fast={fast})...")
        X, Y, G, meta = build_problem_data(problem, fast=fast)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, X=X.numpy(), Y=Y.numpy(), G=G.numpy())
        if meta_path.exists() is False:
            meta_path.write_text(json.dumps(meta, indent=2))
        if verbose:
            print(f"  [factorized_search] Saved {npz_path} ({X.shape[0]} points)")

    # Cap points
    N = int(X.shape[0])
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        X, Y, G = X[idx], Y[idx], G[idx]
        N = max_points

    result_entry["n_points"] = N

    # 2. Build per-component feature tables
    has_J = problem.n_fields > 6
    components = build_component_features(G, Y, has_J=has_J)

    # 3. factorized symbolic search hyperparameters
    if fast:
        bsr_kwargs = dict(
            n_iter=15000, max_depth=5, poly_degree=1,
            brute_max_expressions=20000,
            refine_enable=True, refine_linear_combo_enable=True,
            print_every=0, verbose=False,
        )
    else:
        bsr_kwargs = dict(
            n_iter=60000, max_depth=6, poly_degree=1,
            brute_max_expressions=50000,
            refine_enable=True, refine_linear_combo_enable=True,
            print_every=0, verbose=False,
        )

    # 4. Run factorized symbolic search for each component
    comp_results: list[dict[str, Any]] = []
    errors: list[str] = []
    t0 = time.perf_counter()

    for cgt in comp_gt_list:
        key = f"{cgt.equation_name.lower()}_{cgt.component}"
        if key not in components:
            errors.append(f"{key}: missing component data")
            continue

        x_fit, y_fit, x_probe, y_probe, feat_names = components[key]
        nvars = x_fit.shape[1]

        if verbose:
            print(f"  [factorized_search] {cgt.target_name} ({key}): {nvars} features, "
                  f"{x_fit.shape[0]} fit / {x_probe.shape[0]} probe points")

        try:
            arch = run_explorer_core(
                None, nvars,
                x_fit_data=x_fit.float(),
                y_fit_data=y_fit.float(),
                x_probe_data=x_probe.float(),
                y_probe_data=y_probe.float(),
                seed=hash(key) % (2**31),
                **bsr_kwargs,
            )
        except Exception as exc:
            errors.append(f"{key}: explorer error: {exc}")
            comp_results.append({"key": key, "status": "ERROR", "message": str(exc)})
            continue

        top = arch.best(1)
        if not top:
            errors.append(f"{key}: no candidates found")
            comp_results.append({"key": key, "status": "FAIL", "message": "empty archive"})
            continue

        best = top[0]
        best_mse = float(best.best_mse)
        expr_str = node_str(best.best_expr)
        named_str = _node_str_named(best.best_expr, feat_names)

        # Evaluate on probe set for independent MSE
        pred_raw = eval_node(best.best_expr, x_probe.float())
        pred = eval_mapping(pred_raw, best.mapping)
        probe_mse = float(((pred.squeeze() - y_probe.float().squeeze()) ** 2).mean())

        comp_entry = {
            "key": key,
            "target": cgt.target_name,
            "best_mse": best_mse,
            "probe_mse": probe_mse,
            "expr": expr_str,
            "named_expr": named_str,
        }

        if probe_mse < cgt.mse_tol:
            comp_entry["status"] = "PASS"
        else:
            comp_entry["status"] = "FAIL"
            errors.append(f"{key}: probe_mse={probe_mse:.3e} > tol={cgt.mse_tol:.1e}")

        if verbose:
            status_mark = STATUS_MARKERS.get(comp_entry["status"], "??")
            print(f"    [{status_mark}] {cgt.target_name} = {named_str}  "
                  f"(mse={best_mse:.3e}, probe={probe_mse:.3e})")

        comp_results.append(comp_entry)

    elapsed = time.perf_counter() - t0
    result_entry["discovery_time_s"] = round(elapsed, 3)
    result_entry["components"] = comp_results

    # 5. Overall status
    if errors:
        result_entry["status"] = "FAIL"
        result_entry["message"] = "; ".join(errors[:3])
    else:
        result_entry["status"] = "PASS"
        max_mse = max((c.get("probe_mse", 0) for c in comp_results), default=0)
        result_entry["message"] = f"all 6 components PASS, max_probe_mse={max_mse:.3e}"

    return result_entry


# ---------------------------------------------------------------------------
# Gauss-law check
# ---------------------------------------------------------------------------


def _gauss_check(
    G_np: np.ndarray, n_fields: int
) -> tuple[float, float]:
    """Compute div(E) and div(B) RMS from gradient table.

    Assumes coordinates are [t, x, y, z] and fields start with
    [Ex, Ey, Ez, Bx, By, Bz, ...].
    """
    div_e = G_np[:, 0, 1] + G_np[:, 1, 2] + G_np[:, 2, 3]
    div_b = G_np[:, 3, 1] + G_np[:, 4, 2] + G_np[:, 5, 3]
    div_e_rms = float(np.sqrt(np.mean(div_e * div_e)))
    div_b_rms = float(np.sqrt(np.mean(div_b * div_b)))
    return div_e_rms, div_b_rms


# ---------------------------------------------------------------------------
# Coefficient validation
# ---------------------------------------------------------------------------


def validate_coefficients(
    result,
    gt: VectorGroundTruth,
    name_by_key: dict[str, str],
    named_vecs: dict[str, tuple],
    *,
    verbose: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    """Compare discovered coefficients against ground truth.

    Returns (status, message, details_dict).
    """
    details: dict[str, Any] = {}

    # Build selected map: vec_key -> column index
    selected: dict[str, int] = {}
    for j, tvec in enumerate(result.term_vecs):
        if tvec is None:
            selected["const"] = j
        else:
            selected[_vec_key(tvec)] = j

    if verbose:
        print("  Selected term columns:")
        for j, tvec in enumerate(result.term_vecs):
            if tvec is None:
                label = "const"
            else:
                key = _vec_key(tvec)
                label = name_by_key.get(key, f"term_{j}")
            print(f"    [{j}] {label}")

    # Check order
    if result.order != gt.order:
        return "FAIL", f"Wrong order: expected {gt.order}, got {result.order}", details

    coeffs = result.coeffs  # shape (Q, K_sel)
    errors: list[str] = []
    warnings: list[str] = []
    max_err = 0.0

    # Check expected coefficients
    for term_name, eq_map in gt.expected_coeffs.items():
        vec = named_vecs.get(term_name)
        if vec is None:
            errors.append(f"Term {term_name} not in named_vecs")
            continue
        key = _vec_key(vec)
        if key not in selected:
            errors.append(f"{term_name} not selected by STLSQ")
            continue
        j = selected[key]
        for eq_idx, expected in eq_map.items():
            actual = float(coeffs[eq_idx, j].item())
            err = abs(actual - expected)
            max_err = max(max_err, err)
            details[f"{term_name}_eq{eq_idx}"] = {
                "expected": expected,
                "actual": actual,
                "error": err,
            }
            if verbose:
                print(f"    {term_name} eq{eq_idx}: {actual:+.6f} (expected {expected:+.3f}, err={err:.3e})")
            if err > gt.coeff_tol:
                errors.append(
                    f"{term_name} eq{eq_idx}: expected {expected}, got {actual:.6f} (err={err:.3e})"
                )

    # Check that expected terms have ~0 coefficients in other equations
    for term_name, eq_map in gt.expected_coeffs.items():
        vec = named_vecs.get(term_name)
        if vec is None:
            continue
        key = _vec_key(vec)
        if key not in selected:
            continue
        j = selected[key]
        n_eq = coeffs.shape[0]
        for eq_idx in range(n_eq):
            if eq_idx in eq_map:
                continue
            actual = float(coeffs[eq_idx, j].item())
            err = abs(actual)
            if verbose:
                print(f"    {term_name} eq{eq_idx}: {actual:+.6f} (expected ~0)")
            if err > gt.coeff_tol:
                warnings.append(
                    f"{term_name} eq{eq_idx}: expected ~0, got {actual:.6f}"
                )

    # Check decoy terms
    for decoy_name in gt.decoy_terms:
        vec = named_vecs.get(decoy_name)
        if vec is None:
            continue
        key = _vec_key(vec)
        if key not in selected:
            if verbose:
                print(f"    Decoy {decoy_name}: not selected (good)")
            continue
        j = selected[key]
        for eq_idx in range(coeffs.shape[0]):
            actual = float(coeffs[eq_idx, j].item())
            err = abs(actual)
            if verbose:
                print(f"    Decoy {decoy_name} eq{eq_idx}: {actual:+.3e}")
            if err > gt.decoy_tol:
                warnings.append(
                    f"Decoy {decoy_name} eq{eq_idx}: expected ~0, got {actual:.6f}"
                )

    details["max_coeff_error"] = max_err

    if errors:
        return "FAIL", "; ".join(errors), details
    if warnings:
        return "PARTIAL", "; ".join(warnings), details
    return "PASS", f"max coeff error: {max_err:.4f}", details


# ---------------------------------------------------------------------------
# RMS validation
# ---------------------------------------------------------------------------


def validate_rms(
    result, gt: VectorGroundTruth, *, verbose: bool = False
) -> tuple[str, str, dict[str, Any]]:
    """Check per-equation per-component RMS residuals."""
    details: dict[str, Any] = {}
    errors: list[str] = []
    max_rms = 0.0

    for q, eq_rms in enumerate(result.rms_train):
        for ci, rms in enumerate(eq_rms):
            rms_val = float(rms)
            max_rms = max(max_rms, rms_val)
            details[f"eq{q}_comp{ci}"] = rms_val
            if verbose:
                print(f"    eq{q} comp{ci}: RMS = {rms_val:.3e}")
            if rms_val > gt.rms_tol:
                errors.append(f"eq{q} comp{ci}: RMS={rms_val:.3e} > {gt.rms_tol:.3e}")

    details["max_rms"] = max_rms

    if errors:
        return "FAIL", "; ".join(errors), details
    return "PASS", f"max RMS: {max_rms:.3e}", details


# ---------------------------------------------------------------------------
# Per-problem runner
# ---------------------------------------------------------------------------


def run_problem(
    problem: VectorProblemDef,
    *,
    fast: bool = False,
    skip_generate: bool = False,
    data_dir: Path,
    results_dir: Path,
    surrogate_mode: str = "tabulated",
    surrogate_num_segments: int = 16,
    surrogate_epochs: int = 2500,
    surrogate_loss_target: float = 1e-8,
    surrogate_batch_size: int = 2000,
    surrogate_train_points: int = 4000,
    surrogate_val_points: int = 2000,
    surrogate_rms_tol: float | None = 1.5e-1,
    surrogate_objective: str = "value",
    sobolev_target: str = "spectral_spatial_exact_time",
    sobolev_axes: tuple[int, ...] = (0, 1, 2, 3),
    sobolev_value_weight: float = 1.0,
    sobolev_grad_weight: float = 1.0,
    sobolev_normalize: str = "rms",
    component_workers: int = 1,
    stlsq_lambda: float = 5e-4,
    max_points: int = 25000,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run discovery and validation for one Maxwell problem.

    Returns a result dict compatible with the summary writer.
    """
    result_entry: dict[str, Any] = {
        "id": problem.id,
        "description": problem.description,
        "status": "ERROR",
        "message": "",
        "surrogate": str(surrogate_mode),
        "surrogate_objective": str(surrogate_objective),
    }

    mode = str(surrogate_mode).strip().lower()
    objective = str(surrogate_objective).strip().lower()
    axes = parse_sobolev_axes(sobolev_axes)
    if objective not in {"value", "sobolev"}:
        raise ValueError(f"Unsupported surrogate_objective: {surrogate_objective!r}")
    if objective == "sobolev":
        result_entry["sobolev_target"] = str(sobolev_target)
        result_entry["sobolev_axes"] = list(axes)
        result_entry["sobolev_value_weight"] = float(sobolev_value_weight)
        result_entry["sobolev_grad_weight"] = float(sobolev_grad_weight)
        result_entry["sobolev_normalize"] = str(sobolev_normalize)

    gt = GROUND_TRUTH.get(problem.id)
    if gt is None:
        result_entry["status"] = "ERROR"
        result_entry["message"] = f"No ground truth for {problem.id}"
        return result_entry
    if mode == "nestynet" and surrogate_rms_tol is not None:
        gt = replace(gt, rms_tol=max(float(gt.rms_tol), float(surrogate_rms_tol)))
        result_entry["effective_rms_tol"] = float(gt.rms_tol)

    # ------------------------------------------------------------------
    # 1. Generate or load data
    # ------------------------------------------------------------------
    npz_path = data_dir / f"{problem.id}.npz"
    meta_path = data_dir / f"{problem.id}.meta.json"

    generator_overrides: dict[str, Any] = {}
    if mode == "nestynet":
        if problem.id == "mw000":
            generator_overrides.update(
                {
                    "nx": 3 if fast else 5,
                    "ny": 3 if fast else 5,
                    "x_max": 1.0,
                    "y_max": 1.0,
                }
            )
        elif problem.id == "mw001":
            generator_overrides.update(
                {
                    "nz": 3,
                    "z_max": 1.0,
                }
            )

    if skip_generate and npz_path.exists():
        if verbose:
            print(f"  Loading existing data: {npz_path}")
        blob = np.load(npz_path)
        X = torch.from_numpy(np.asarray(blob["X"], dtype=np.float64))
        Y = torch.from_numpy(np.asarray(blob["Y"], dtype=np.float64))
        G = torch.from_numpy(np.asarray(blob["G"], dtype=np.float64))
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
    else:
        if verbose:
            print(f"  Generating data (fast={fast})...")
        X, Y, G, meta = build_problem_data(problem, fast=fast, **generator_overrides)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            X=X.numpy(),
            Y=Y.numpy(),
            G=G.numpy(),
        )
        meta_path.write_text(json.dumps(meta, indent=2))
        if verbose:
            print(f"  Saved {npz_path} ({X.shape[0]} points)")

    result_entry["n_points"] = int(X.shape[0])

    G_target = None
    if mode == "nestynet" and objective == "sobolev":
        G_target = build_derivative_targets(str(sobolev_target), X, Y, G)
        target_err = G_target - G
        spatial_err = target_err[:, :, list(problem.spatial_axes)]
        spatial_abs_rms = float(torch.sqrt(torch.mean(spatial_err * spatial_err)).item())
        time_abs_rms = float(torch.sqrt(torch.mean(target_err[:, :, 0].square())).item())
        result_entry["sobolev_target_spatial_abs_rms"] = spatial_abs_rms
        result_entry["sobolev_target_time_abs_rms"] = time_abs_rms
        if verbose:
            print(
                "  Sobolev derivative target: "
                f"mode={sobolev_target}, spatial_abs_rms="
                f"{spatial_abs_rms:.3e}, time_abs_rms={time_abs_rms:.3e}"
            )

    # ------------------------------------------------------------------
    # 2. Gauss-law check (if 3D spatial fields)
    # ------------------------------------------------------------------
    G_np = G.numpy()
    if problem.n_fields >= 6:
        div_e_rms, div_b_rms = _gauss_check(G_np, problem.n_fields)
        result_entry["div_e_rms"] = div_e_rms
        result_entry["div_b_rms"] = div_b_rms
        if verbose:
            print(f"  Gauss: div(E)={div_e_rms:.3e}, div(B)={div_b_rms:.3e}")
        gauss_tol = gt.gauss_tol
        if div_e_rms > gauss_tol or div_b_rms > gauss_tol:
            # Not a hard failure — just a warning in verbose mode
            if verbose:
                print(f"  Warning: Gauss-law RMS exceeds {gauss_tol:.1e}")

    # ------------------------------------------------------------------
    # 3. Build surrogate and terms
    # ------------------------------------------------------------------
    if mode == "tabulated":
        # Perfect-information surrogate: exact tabulated values/first-derivatives,
        # plus an exact spatial Hessian (FFT spectral second derivative) so the
        # broadened library can offer the ∇²E / ∇²B second-order decoys.  This is
        # machine-precision for the periodic plane-wave cases and near-exact for
        # the well-localized wire field.
        H = build_spectral_spatial_hessian_diag(
            X, Y, spatial_cols=tuple(problem.spatial_axes), time_col=0
        )
        surrogate = TabulatedVectorSurrogate(X, Y, G, H)
        result_entry["hessian_target_abs_rms"] = float(torch.sqrt(torch.mean(H * H)))
    elif mode == "nestynet":
        surrogate, val_loss, surrogate_info = train_vector_nestynet_surrogate(
            problem,
            X,
            Y,
            data_dir=data_dir,
            num_segments=int(surrogate_num_segments),
            epochs=int(surrogate_epochs),
            loss_target=float(surrogate_loss_target),
            batch_size=int(surrogate_batch_size),
            ndata_train=int(surrogate_train_points),
            ndata_val=int(surrogate_val_points),
            device=torch.device("cpu"),
            dtype=torch.float64,
            surrogate_objective=objective,
            sobolev_target=str(sobolev_target),
            sobolev_axes=axes,
            sobolev_value_weight=float(sobolev_value_weight),
            sobolev_grad_weight=float(sobolev_grad_weight),
            sobolev_normalize=str(sobolev_normalize),
            G_target=G_target,
            G_exact=G,
            component_workers=int(component_workers),
            verbose=verbose,
        )
        result_entry["surrogate_val_loss"] = float(val_loss)
        result_entry["surrogate_info"] = surrogate_info
        if verbose:
            print(f"  NestyNet surrogate val_loss={val_loss:.6e}")
    else:
        raise ValueError(f"Unsupported surrogate mode: {surrogate_mode!r}")

    # ------------------------------------------------------------------
    # 3b. Derivative-accuracy diagnostic.  Compare the surrogate's analytic
    #     first derivatives against the exact tabulated G, per (component,
    #     axis).  The discovery residual can be no better than these, so this
    #     isolates derivative accuracy from the discovery/selection step.
    # ------------------------------------------------------------------
    try:
        with torch.no_grad():
            g_pred = surrogate.grad(X).detach()
        ncomp = min(int(g_pred.shape[1]), int(G.shape[1]))
        naxis = min(int(g_pred.shape[2]), int(G.shape[2]))
        gerr = g_pred[:, :ncomp, :naxis] - G[:, :ncomp, :naxis]
        abs_rms = torch.sqrt(torch.mean(gerr * gerr, dim=0))
        true_rms = torch.sqrt(torch.mean(G[:, :ncomp, :naxis] ** 2, dim=0))
        axis_names = ["t", "x", "y", "z"][:naxis]
        cells = []
        for c in range(ncomp):
            cname = problem.field_names[c] if c < len(problem.field_names) else f"c{c}"
            for j in range(naxis):
                cells.append({
                    "component": cname,
                    "axis": axis_names[j],
                    "abs_rms": float(abs_rms[c, j]),
                    "true_rms": float(true_rms[c, j]),
                    "rel_rms": float(abs_rms[c, j] / max(float(true_rms[c, j]), 1e-12)),
                })
        cells.sort(key=lambda d: d["abs_rms"], reverse=True)
        result_entry["deriv_diag"] = {
            "overall_grad_abs_rms": float(torch.sqrt(torch.mean(gerr * gerr))),
            "worst": cells[:8],
        }
        if verbose:
            print(
                "  Deriv error vs exact G: overall abs-RMS="
                f"{result_entry['deriv_diag']['overall_grad_abs_rms']:.3e}; worst:"
            )
            for cell in cells[:6]:
                print(
                    f"    d{cell['component']}/d{cell['axis']}: "
                    f"abs={cell['abs_rms']:.3e} rel={cell['rel_rms']:.1%}"
                )
    except Exception as exc:  # a diagnostic must never break the run
        result_entry["deriv_diag_error"] = str(exc)

    loader = DataLoader(
        TensorDataset(X), batch_size=int(X.shape[0]), shuffle=False
    )
    # Broaden the candidate menu with ∇² decoys only on the perfect-information
    # (tabulated) path, where the exact Hessian is available.
    vector_terms, name_by_key, named_vecs = build_vector_terms(
        problem, include_laplacian=(mode == "tabulated")
    )

    # ------------------------------------------------------------------
    # 3c. Conditioning / alias audit (before STLSQ).  Detect exact feature
    #     aliases that make the design rank-deficient (e.g. lap(E) = -k^2 E on a
    #     single-mode plane wave).  Report rather than crash: aliased columns are
    #     dropped so the solve stays well-posed, and the overall status is flagged
    #     NONIDENTIFIABLE_ALIAS (support correct modulo aliases).
    audit = audit_vector_library(
        surrogate, X, vector_terms, name_by_key, problem.equations
    )
    result_entry["conditioning_audit"] = audit
    alias_detected = audit["rank_status"] == "NONIDENTIFIABLE_ALIAS"
    if verbose:
        print(
            f"  Conditioning audit: rank_status={audit['rank_status']}, "
            f"max off-diag corr={audit['max_offdiag_corr']:.3e}"
        )
        if alias_detected:
            for pr in audit["alias_pairs"]:
                print(f"    ALIAS: {pr['a']} ~ {pr['b']} (corr={pr['corr']:+.3f})")
            print(f"    -> dropping aliased columns: {audit['drop_terms']}")
        else:
            print("    no exact Laplacian aliases; decoy columns are identifiable")
    if alias_detected and audit["drop_terms"]:
        vector_terms, name_by_key, named_vecs = apply_alias_drops(
            vector_terms, name_by_key, named_vecs, audit["drop_terms"]
        )

    # ------------------------------------------------------------------
    # 4. Run discovery
    # ------------------------------------------------------------------
    cfg = VectorSystemDESearchConfig(
        x_axis=0,
        order_candidates=(1,),
        include_const=False,
        stlsq_lambda=stlsq_lambda,
        stlsq_max_iter=20,
        sparsity_penalty=1e-6,
        share_support_across_equations=False,
        max_points=max_points,
    )

    t0 = time.perf_counter()
    disc_result = discover_vector_system_de_from_surrogate(
        surrogate,
        loader,
        cfg=cfg,
        equations=problem.equations,
        vector_terms=vector_terms,
        device=torch.device("cpu"),
    )
    elapsed = time.perf_counter() - t0
    result_entry["discovery_time_s"] = round(elapsed, 3)
    engine_rank_deficient = bool(getattr(disc_result, "rank_deficient", False))
    result_entry["engine_rank_deficient"] = engine_rank_deficient

    # Conditioning of the discovered support (is the *answer* stable?), distinct
    # from the full-library coherence (are coherent decoys present?).
    support_cond = support_conditioning(
        surrogate, X, disc_result.term_vecs, problem.equations, name_by_key=name_by_key
    )
    result_entry["support_conditioning"] = support_cond
    # Soft-conditioning regime: identifiable (full rank) but coherent.  Any of
    # high pairwise correlation, ill-conditioned library, or an inflated
    # selected-coefficient variance flags it (friend's taxonomy).
    high_coherence = (
        float(audit.get("max_offdiag_corr", 0.0)) > COHERENCE_MU_WARN
        or float(audit.get("max_cond_number", 1.0)) > COHERENCE_KAPPA_WARN
        or float(support_cond.get("max_vif", 1.0)) > COHERENCE_VIF_WARN
    )

    if verbose:
        print(f"  Discovery completed in {elapsed:.2f}s, order={disc_result.order}")
        print(
            "  Conditioning: library kappa="
            f"{audit.get('max_cond_number', float('nan')):.2f}, "
            f"support kappa={support_cond.get('max_cond_number', float('nan')):.2f}, "
            f"max VIF={support_cond.get('max_vif', float('nan')):.2f}"
            + ("  [HIGH COHERENCE]" if high_coherence else "")
        )
        print(f"  Recovered system:\n{disc_result.format_system()}")

    # ------------------------------------------------------------------
    # 5. Validate: coefficients
    # ------------------------------------------------------------------
    if verbose:
        print("  Coefficient validation:")
    coeff_status, coeff_msg, coeff_details = validate_coefficients(
        disc_result, gt, name_by_key, named_vecs, verbose=verbose
    )
    result_entry["coeff_status"] = coeff_status
    result_entry["coeff_message"] = coeff_msg
    result_entry["coeff_details"] = coeff_details

    # ------------------------------------------------------------------
    # 6. Validate: RMS residuals
    # ------------------------------------------------------------------
    if verbose:
        print("  RMS validation:")
    rms_status, rms_msg, rms_details = validate_rms(
        disc_result, gt, verbose=verbose
    )
    result_entry["rms_status"] = rms_status
    result_entry["rms_message"] = rms_msg
    result_entry["rms_details"] = rms_details

    # ------------------------------------------------------------------
    # 7. Overall status
    # ------------------------------------------------------------------
    if coeff_status == "FAIL" or rms_status == "FAIL":
        result_entry["status"] = "FAIL"
        msgs = []
        if coeff_status == "FAIL":
            msgs.append(f"coeff: {coeff_msg}")
        if rms_status == "FAIL":
            msgs.append(f"rms: {rms_msg}")
        result_entry["message"] = "; ".join(msgs)
    elif coeff_status == "PARTIAL" or rms_status == "PARTIAL":
        result_entry["status"] = "PARTIAL"
        result_entry["message"] = coeff_msg if coeff_status == "PARTIAL" else rms_msg
    elif alias_detected:
        # Support recovered on the de-aliased library, but the broadened library
        # is genuinely rank-deficient on this excitation: report it as such
        # rather than as a clean broad-library selection pass.
        pairs = ", ".join(f"{p['a']}~{p['b']}" for p in audit["alias_pairs"])
        result_entry["status"] = "NONIDENTIFIABLE_ALIAS"
        result_entry["message"] = (
            f"support correct modulo aliases ({pairs}); "
            f"coeff_err={coeff_details.get('max_coeff_error', 0):.4f}"
        )
    elif engine_rank_deficient:
        # The audit did not flag an exact alias, but the selected support was
        # still rank-deficient and recovered by the engine's min-norm fallback.
        # Report it honestly rather than as a clean unique-support pass.
        result_entry["status"] = "RANK_DEFICIENT"
        result_entry["message"] = (
            "rank-deficient support solved by min-norm fallback; "
            f"coeff_err={coeff_details.get('max_coeff_error', 0):.4f}, "
            f"rms={rms_details.get('max_rms', 0):.3e}"
        )
    elif (mode == "tabulated") and high_coherence:
        # Identifiable (full rank, support correct) but the broadened library
        # contains highly coherent directions: the selection is correct with
        # perfect derivatives, so the noise sweep tests *support stability*
        # rather than mere noiseless recoverability.
        result_entry["status"] = "FULL_RANK_HIGH_COHERENCE"
        result_entry["message"] = (
            f"identifiable but coherent: mu={audit.get('max_offdiag_corr', 0):.3f}, "
            f"lib_kappa={audit.get('max_cond_number', 0):.1f}, "
            f"support_VIF={support_cond.get('max_vif', 0):.1f}; "
            f"coeff_err={coeff_details.get('max_coeff_error', 0):.4f}"
        )
    elif mode == "tabulated":
        result_entry["status"] = "FULL_RANK_STABLE"
        result_entry["message"] = (
            f"identifiable, well-conditioned (mu={audit.get('max_offdiag_corr', 0):.3f}); "
            f"coeff_err={coeff_details.get('max_coeff_error', 0):.4f}, "
            f"rms={rms_details.get('max_rms', 0):.3e}"
        )
    else:
        result_entry["status"] = "PASS"
        result_entry["message"] = f"coeff_err={coeff_details.get('max_coeff_error', 0):.4f}, rms={rms_details.get('max_rms', 0):.3e}"

    return result_entry


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    """Print ASCII summary table and return status counts."""
    print()
    print("=" * 86)
    print("MAXWELL VECTOR-PDE BENCHMARK SUMMARY")
    print("=" * 86)
    print(f"{'ID':<8} {'Engine':<9} {'Description':<26} {'Status':<10} {'Details'}")
    print("-" * 86)

    counts: dict[str, int] = {}
    for r in results:
        status = r["status"]
        counts[status] = counts.get(status, 0) + 1
        marker = STATUS_MARKERS.get(status, "??")
        desc = r["description"][:24]
        eng = r.get("engine", "stlsq")[:7]
        msg = r["message"][:46] if r.get("message") else ""
        print(f"[{marker}] {r['id']:<5} {eng:<9} {desc:<26} {status:<10} {msg}")

    print("-" * 86)
    total = sum(counts.values())
    parts = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"Total: {total} | {parts}")
    print("=" * 86)
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Maxwell vector-PDE benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated problem IDs (e.g. mw000,mw001)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all problems",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["stlsq", "factorized_search", "both"],
        default="stlsq",
        help="Discovery engine: stlsq, factorized_search, or both",
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        choices=["tabulated", "nestynet"],
        default="tabulated",
        help="Surrogate source for the vector STLSQ engine",
    )
    parser.add_argument("--fast", action="store_true", help="Reduced grid resolution")
    parser.add_argument(
        "--skip_generate",
        action="store_true",
        help="Reuse existing .npz files if available",
    )
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    parser.add_argument(
        "--stlsq_lambda", type=float, default=5e-4, help="STLSQ threshold"
    )
    parser.add_argument(
        "--max_points", type=int, default=25000, help="Discovery max_points"
    )
    parser.add_argument(
        "--surrogate_num_segments",
        type=int,
        default=16,
        help="NestyNet segments for --surrogate nestynet",
    )
    parser.add_argument(
        "--surrogate_epochs",
        type=int,
        default=2500,
        help="Maximum NestyNet training epochs for --surrogate nestynet",
    )
    parser.add_argument(
        "--surrogate_loss_target",
        type=float,
        default=1e-8,
        help="NestyNet validation loss target for --surrogate nestynet",
    )
    parser.add_argument(
        "--surrogate_batch_size",
        type=int,
        default=2000,
        help="NestyNet surrogate training batch size",
    )
    parser.add_argument(
        "--surrogate_train_points",
        type=int,
        default=4000,
        help="Training rows sampled for the NestyNet vector surrogate",
    )
    parser.add_argument(
        "--surrogate_val_points",
        type=int,
        default=2000,
        help="Validation rows sampled for the NestyNet vector surrogate",
    )
    parser.add_argument(
        "--surrogate_rms_tol",
        type=float,
        default=1.5e-1,
        help="RMS validation tolerance used for --surrogate nestynet",
    )
    parser.add_argument(
        "--surrogate_objective",
        type=str,
        choices=["value", "sobolev"],
        default="value",
        help="Objective for --surrogate nestynet",
    )
    parser.add_argument(
        "--component_workers",
        type=int,
        default=1,
        help="Train the field-component surrogates in N worker processes (>1 = true multi-core)",
    )
    parser.add_argument(
        "--sobolev_target",
        type=str,
        choices=["exact", "spectral_spatial_exact_time"],
        default="spectral_spatial_exact_time",
        help="Derivative target construction mode for Sobolev surrogate training",
    )
    parser.add_argument(
        "--sobolev_axes",
        type=str,
        default="t,x,y,z",
        help="Comma/space-separated derivative axes for Sobolev training",
    )
    parser.add_argument(
        "--sobolev_value_weight",
        type=float,
        default=1.0,
        help="Value residual weight for Sobolev surrogate training",
    )
    parser.add_argument(
        "--sobolev_grad_weight",
        type=float,
        default=1.0,
        help="Gradient residual weight for Sobolev surrogate training",
    )
    parser.add_argument(
        "--sobolev_normalize",
        type=str,
        choices=["rms", "none"],
        default="rms",
        help="Channel normalization for Sobolev surrogate training",
    )
    parser.add_argument(
        "--coeff_tol",
        type=float,
        default=None,
        help="Override coefficient tolerance (default: per-problem)",
    )
    parser.add_argument(
        "--rms_tol",
        type=float,
        default=None,
        help="Override RMS tolerance (default: per-problem)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Directory for generated data",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Directory for results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = _SCRIPT_DIR.parent.parent
    data_dir = args.data_dir or (repo_root / "data" / "maxwell_benchmark")
    results_dir = args.results_dir or (repo_root / "results" / "maxwell")

    # Resolve problem IDs
    if args.all:
        problem_ids = list(PROBLEM_REGISTRY.keys())
    else:
        raw = [s.strip() for s in args.only.split(",")]
        problem_ids = []
        for r in raw:
            if r in PROBLEM_REGISTRY:
                problem_ids.append(r)
            else:
                print(f"Warning: unknown problem ID '{r}', skipping")

    if not problem_ids:
        print("No problems to run.")
        sys.exit(1)

    # Override tolerances if provided
    if args.coeff_tol is not None or args.rms_tol is not None:
        for pid in problem_ids:
            gt = GROUND_TRUTH.get(pid)
            if gt is None:
                continue
            if args.coeff_tol is not None:
                gt.coeff_tol = args.coeff_tol
                gt.decoy_tol = args.coeff_tol
            if args.rms_tol is not None:
                gt.rms_tol = args.rms_tol

    engines_to_run: list[str] = (
        ["stlsq", "factorized_search"] if args.engine == "both" else [args.engine]
    )

    results: list[dict[str, Any]] = []

    for pid in problem_ids:
        problem = PROBLEM_REGISTRY[pid]
        print()
        print("=" * 78)
        print(f"{pid}: {problem.description}")
        print(f"  Fields: {problem.field_names}")
        print(f"  Equations: {[eq.name for eq in problem.equations]}")
        print(f"  Engine(s): {', '.join(engines_to_run)}")
        print("=" * 78)

        for engine in engines_to_run:
            try:
                if engine == "stlsq":
                    entry = run_problem(
                        problem,
                        fast=args.fast,
                        skip_generate=args.skip_generate,
                        data_dir=data_dir,
                        results_dir=results_dir,
                        surrogate_mode=args.surrogate,
                        surrogate_num_segments=args.surrogate_num_segments,
                        surrogate_epochs=args.surrogate_epochs,
                        surrogate_loss_target=args.surrogate_loss_target,
                        surrogate_batch_size=args.surrogate_batch_size,
                        surrogate_train_points=args.surrogate_train_points,
                        surrogate_val_points=args.surrogate_val_points,
                        surrogate_rms_tol=(
                            args.surrogate_rms_tol if args.rms_tol is None else None
                        ),
                        surrogate_objective=args.surrogate_objective,
                        sobolev_target=args.sobolev_target,
                        sobolev_axes=parse_sobolev_axes(args.sobolev_axes),
                        sobolev_value_weight=args.sobolev_value_weight,
                        sobolev_grad_weight=args.sobolev_grad_weight,
                        sobolev_normalize=args.sobolev_normalize,
                        component_workers=args.component_workers,
                        stlsq_lambda=args.stlsq_lambda,
                        max_points=args.max_points,
                        verbose=args.verbose,
                    )
                else:
                    entry = run_problem_factorized_search(
                        problem,
                        fast=args.fast,
                        skip_generate=args.skip_generate,
                        data_dir=data_dir,
                        results_dir=results_dir,
                        max_points=args.max_points,
                        verbose=args.verbose,
                    )
            except Exception:
                entry = {
                    "id": pid,
                    "description": problem.description,
                    "engine": engine,
                    "status": "ERROR",
                    "message": traceback.format_exc().splitlines()[-1],
                }
                traceback.print_exc()

            entry.setdefault("engine", engine)
            status = entry["status"]
            marker = STATUS_MARKERS.get(status, "??")
            print(f"\n  [{marker}] {engine}: {status}: {entry.get('message', '')}")
            results.append(entry)

    # Summary
    counts = print_summary(results)

    # Write JSON
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.json"
    summary = {
        "engine": args.engine,
        "surrogate": args.surrogate,
        "fast": args.fast,
        "stlsq_lambda": args.stlsq_lambda,
        "surrogate_num_segments": args.surrogate_num_segments,
        "surrogate_epochs": args.surrogate_epochs,
        "surrogate_loss_target": args.surrogate_loss_target,
        "surrogate_train_points": args.surrogate_train_points,
        "surrogate_val_points": args.surrogate_val_points,
        "surrogate_rms_tol": args.surrogate_rms_tol,
        "surrogate_objective": args.surrogate_objective,
        "sobolev_target": args.sobolev_target,
        "sobolev_axes": parse_sobolev_axes(args.sobolev_axes),
        "sobolev_value_weight": args.sobolev_value_weight,
        "sobolev_grad_weight": args.sobolev_grad_weight,
        "sobolev_normalize": args.sobolev_normalize,
        "problems": results,
        "counts": counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to {summary_path}")

    # Exit code
    if counts.get("FAIL", 0) > 0 or counts.get("ERROR", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
