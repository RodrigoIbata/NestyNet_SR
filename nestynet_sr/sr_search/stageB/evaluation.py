# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Model evaluation utilities for Stage B.

This module provides functions for computing metrics (MSE, MAD, etc.) and
diagnostic statistics over validation data during Stage B refinement.
"""

from __future__ import annotations

import math

import torch

from nestynet_sr.sr_core.fit_links import fit_link_torch


def _compute_y_med_mad_from_loader(
    dl,
    device: torch.device,
    *,
    fit_link=None,
    fit_link_scale: float = 1.0,
):
    """
    Compute median and MAD of scalar targets y from a dataloader.
    Used in Stage B to normalise LM thresholds when loss_in_MAD_units is True.
    """
    ys = []
    for batch in dl:
        if isinstance(batch, (list, tuple)):
            _, y = batch
        else:
            y = batch
        y = y.to(device)
        if y.dim() > 1:
            y = y[:, 0]
        ys.append(y.detach().cpu())
    if not ys:
        return None, None
    y_all = torch.cat(ys, dim=0)
    y_all = fit_link_torch(y_all, fit_link, scale=float(fit_link_scale))
    med = torch.median(y_all)
    mad = torch.median(torch.abs(y_all - med))
    return float(med), float(mad)


def _eval_val_mse(model: torch.nn.Module, val_loader, device: torch.device) -> float:
    """
    Evaluate validation MSE.

    Used by Stage B post-check flow and debugging.
    """
    model.eval()
    se_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                raise ValueError("Validation dataloader must yield (x, y) batches.")
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            if y_pred.dim() == 2:
                y_pred = y_pred[:, 0]
            else:
                y_pred = y_pred.view(-1)
            if y.dim() == 2:
                y_true = y[:, 0]
            else:
                y_true = y.view(-1)
            diff = y_pred - y_true
            se_sum += float((diff * diff).sum().cpu())
            n_total += diff.numel()
    if n_total == 0:
        return float("+inf")
    return se_sum / float(n_total)


def _eval_original_y_mse_with_inverse(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    y_op_inv,
) -> float:
    """Evaluate validation MSE after mapping φ-space predictions back to y-space."""
    if y_op_inv is None:
        return float("nan")
    model.eval()
    se_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for batch in val_loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Validation dataloader must yield (x, y) batches.")
            x, phi_true = batch[0].to(device), batch[1].to(device)
            phi_pred = model(x)
            if phi_pred.dim() == 2 and phi_pred.shape[1] == 1:
                phi_pred = phi_pred[:, 0]
            else:
                phi_pred = phi_pred.view(-1)
            if phi_true.dim() == 2 and phi_true.shape[1] == 1:
                phi_true = phi_true[:, 0]
            else:
                phi_true = phi_true.view(-1)
            y_pred = y_op_inv(phi_pred)
            y_true = y_op_inv(phi_true)
            y_pred = (
                y_pred[:, 0]
                if getattr(y_pred, "dim", lambda: 0)() == 2
                else y_pred.view(-1)
            )
            y_true = (
                y_true[:, 0]
                if getattr(y_true, "dim", lambda: 0)() == 2
                else y_true.view(-1)
            )
            finite = torch.isfinite(y_pred) & torch.isfinite(y_true)
            if not torch.all(finite):
                return float("inf")
            diff = y_pred - y_true
            se_sum += float((diff * diff).sum().cpu())
            n_total += diff.numel()
    if n_total == 0:
        return float("inf")
    return se_sum / float(n_total)


def _compute_original_y_mad2_with_inverse(
    val_loader,
    device: torch.device,
    y_op_inv,
) -> float:
    """Compute MAD(y)^2 when the validation loader stores φ(y)."""
    if y_op_inv is None:
        return float("nan")
    chunks = []
    with torch.no_grad():
        for batch in val_loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            phi = batch[1].to(device)
            phi = phi[:, 0] if phi.dim() == 2 and phi.shape[1] == 1 else phi.view(-1)
            y = y_op_inv(phi)
            y = (
                y[:, 0]
                if getattr(y, "dim", lambda: 0)() == 2
                else y.view(-1)
            )
            finite = torch.isfinite(y)
            if torch.any(finite):
                chunks.append(y[finite].detach().cpu())
    if not chunks:
        return float("nan")
    y_all = torch.cat(chunks)
    if y_all.numel() == 0:
        return float("nan")
    med = torch.median(y_all)
    mad = torch.median(torch.abs(y_all - med))
    return float(mad * mad)


def _shuffle_axis_sensitivity(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    axes=(2, 3),
    *,
    verbose: bool = True,
):
    """
    Measure sensitivity of model outputs to shuffling selected input axes across a batch.

    Uses a single validation batch and reports max |Δ| and RMSE between original
    predictions and predictions after independently permuting the specified axes.
    """
    if val_loader is None:
        if verbose:
            print("[Stage B] Shuffle diagnostic skipped: val_loader is None.")
        return None
    try:
        batch = next(iter(val_loader))
    except Exception as e:
        if verbose:
            print(f"[Stage B] Shuffle diagnostic skipped: failed to read val batch ({e}).")
        return None
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    if not torch.is_tensor(x):
        try:
            x = torch.as_tensor(x)
        except Exception as e:
            if verbose:
                print(
                    f"[Stage B] Shuffle diagnostic skipped: could not convert batch to tensor ({e})."
                )
            return None
    if x.ndim != 2:
        if verbose:
            print(
                f"[Stage B] Shuffle diagnostic skipped: expected 2D inputs, got shape {tuple(x.shape)}."
            )
        return None
    if any((ax < 0 or ax >= x.shape[1]) for ax in axes):
        if verbose:
            print(
                "[Stage B] Shuffle diagnostic skipped: input has only "
                f"{x.shape[1]} columns, axes={axes} not available."
            )
        return None

    model.eval()
    x = x.to(device)
    with torch.no_grad():
        yp = model(x)
        yp = yp[:, 0] if yp.dim() == 2 else yp.view(-1)

        xs = x.clone()
        gen = torch.Generator(device=xs.device)
        gen.manual_seed(0)
        perm = torch.randperm(xs.shape[0], device=xs.device, generator=gen)
        for ax in axes:
            xs[:, ax] = xs[perm, ax]

        yps = model(xs)
        yps = yps[:, 0] if yps.dim() == 2 else yps.view(-1)

        diff = yp - yps
        max_abs = diff.abs().max().item()
        rmse = torch.sqrt((diff**2).mean()).item()

    if verbose:
        ax_str = ",".join(str(a) for a in axes)
        print(
            f"[Stage B] Shuffle diagnostic (axes {ax_str}): max|Δ|={max_abs:.3g}, rmse={rmse:.3g}"
        )

    return {"axes": tuple(int(a) for a in axes), "max_abs": max_abs, "rmse": rmse}


def _phi_pred_error_from_loader(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    *,
    y_is_phi: bool = True,
    verbose: bool = True,
):
    """
    Compare model outputs to target φ(y) over the validation loader.

    If y_is_phi is False, the loader is assumed to provide raw y and we
    set φ(y) = y^2 to match a common transform used in Stage B experiments.
    """
    if val_loader is None:
        if verbose:
            print("[Stage B] Phi diagnostic skipped: val_loader is None.")
        return None

    model.eval()
    max_err = 0.0
    se_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                xb, yb = batch
            else:
                raise ValueError("Validation dataloader must yield (x, y) batches.")
            xb = xb.to(device)
            yb = yb.to(device)
            if yb.dim() == 2:
                yb = yb[:, 0]
            else:
                yb = yb.view(-1)
            phi_true = yb if y_is_phi else (yb * yb)

            phi_pred = model(xb)
            if phi_pred.dim() == 2:
                phi_pred = phi_pred[:, 0]
            else:
                phi_pred = phi_pred.view(-1)

            diff = phi_pred - phi_true
            max_err = max(max_err, float(diff.abs().max().item()))
            se_sum += float((diff * diff).sum().item())
            n_total += diff.numel()

    if n_total == 0:
        rmse = float("nan")
    else:
        rmse = math.sqrt(se_sum / n_total)

    if verbose:
        label = "phi_true=y" if y_is_phi else "phi_true=y^2"
        print(f"[Stage B] Phi diagnostic ({label}): max|Δ|={max_err:.3g}, rmse={rmse:.3g}")

    return {"y_is_phi": bool(y_is_phi), "max_abs": max_err, "rmse": rmse}


def _print_val_batch_stats(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    *,
    verbose: bool = True,
):
    """
    Print min/max/median stats for a single validation batch (y_true and y_pred).
    """
    if not verbose:
        return None
    if val_loader is None:
        print("[Stage B] Batch stats skipped: val_loader is None.")
        return None
    try:
        batch = next(iter(val_loader))
    except Exception as e:
        print(f"[Stage B] Batch stats skipped: failed to read val batch ({e}).")
        return None
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        print("[Stage B] Batch stats skipped: expected (x, y) from val_loader.")
        return None
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        yp = model(x)

    def _stats(t):
        t = t.detach().flatten()
        if t.numel() == 0:
            return float("nan"), float("nan"), float("nan")
        return float(t.min()), float(t.max()), float(t.median())

    y_min, y_max, y_med = _stats(y)
    yp_min, yp_max, yp_med = _stats(yp)
    print(f"[Stage B] y_true stats: min={y_min:.3g}, max={y_max:.3g}, median={y_med:.3g}")
    print(f"[Stage B] y_pred stats: min={yp_min:.3g}, max={yp_max:.3g}, median={yp_med:.3g}")
    return {
        "y_true": {"min": y_min, "max": y_max, "median": y_med},
        "y_pred": {"min": yp_min, "max": yp_max, "median": yp_med},
    }


def _eval_mse_and_rms(model: torch.nn.Module, loader, device: torch.device):
    """
    Recompute validation MSE and RMS directly from the provided loader.
    """
    model.eval()
    se_chunks = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            yp = model(x)
            if yp.dim() == 2 and yp.shape[1] == 1:
                yp = yp[:, 0]
            else:
                yp = yp.view(-1)
            if y.dim() == 2 and y.shape[1] == 1:
                y = y[:, 0]
            else:
                y = y.view(-1)
            se_chunks.append(((yp - y) ** 2).detach().cpu())
    if not se_chunks:
        return float("nan"), float("nan")
    se = torch.cat(se_chunks)
    return float(se.mean()), float(se.sqrt().mean())
