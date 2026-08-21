#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Derivative-target construction for Maxwell grid surrogates."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _as_axis_tuple(cols: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(c) for c in cols)
    if len(out) == 0:
        raise ValueError("at least one axis column is required")
    if len(set(out)) != len(out):
        raise ValueError(f"axis columns must be unique, got {out}")
    if any(c < 0 for c in out):
        raise ValueError(f"axis columns must be non-negative, got {out}")
    return out


def _tensor_grid_from_rows(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    grid_cols: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return ``Y`` on a sorted tensor grid and row-to-grid linear indices."""
    cols = _as_axis_tuple(grid_cols)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N,D), got {tuple(X.shape)}")
    if Y.ndim != 2:
        raise ValueError(f"Y must have shape (N,n_fields), got {tuple(Y.shape)}")
    if int(X.shape[0]) != int(Y.shape[0]):
        raise ValueError(f"X/Y row mismatch: {int(X.shape[0])} != {int(Y.shape[0])}")
    if max(cols) >= int(X.shape[1]):
        raise ValueError(f"grid column {max(cols)} outside X dimension {int(X.shape[1])}")

    invs: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    for c in cols:
        unique, inv = torch.unique(X[:, c].detach().cpu(), sorted=True, return_inverse=True)
        vals.append(unique.to(device=X.device, dtype=X.dtype))
        invs.append(inv.to(device=X.device, dtype=torch.long))

    shape = [int(v.numel()) for v in vals]
    n_grid = 1
    for n in shape:
        n_grid *= int(n)
    n_rows = int(X.shape[0])
    if n_grid != n_rows:
        raise ValueError(
            f"X is not a full tensor-product grid over columns {cols}: "
            f"product={n_grid}, rows={n_rows}"
        )

    linear = invs[0].clone()
    for inv, n in zip(invs[1:], shape[1:]):
        linear = linear * int(n) + inv
    uniq, counts = torch.unique(linear.detach().cpu(), sorted=True, return_counts=True)
    if int(uniq.numel()) != n_rows or bool(torch.any(counts != 1)):
        raise ValueError("X grid rows are not a unique tensor-product mapping")

    Y_flat = torch.empty(
        n_rows,
        int(Y.shape[1]),
        device=Y.device,
        dtype=Y.dtype,
    )
    Y_flat[linear.to(device=Y.device)] = Y
    return Y_flat.reshape(*shape, int(Y.shape[1])), linear, vals, torch.tensor(shape)


def _periodic_spacing(vals: torch.Tensor) -> float:
    n = int(vals.numel())
    if n <= 1:
        return 1.0
    diffs = vals[1:] - vals[:-1]
    step = float(torch.median(diffs).item())
    if step <= 0.0:
        raise ValueError("grid axis values must be strictly increasing")
    err = torch.max(torch.abs(diffs - step)).item()
    if err > 1e-8 * max(1.0, abs(step)):
        raise ValueError(f"grid spacing is not uniform enough for FFT differentiation (max err={err:.3e})")
    return step


def _time_target_from_G(G_time: torch.Tensor, *, time_col: int) -> torch.Tensor:
    if G_time.ndim == 3:
        return G_time[:, :, int(time_col)]
    if G_time.ndim == 2:
        return G_time
    raise ValueError(
        "G_time must have shape (N,n_fields) or (N,n_fields,n_axes), "
        f"got {tuple(G_time.shape)}"
    )


def build_spectral_lowpass(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    spatial_cols: Sequence[int] = (1, 2, 3),
    time_col: int = 0,
    cutoff_frac: float = 0.5,
) -> torch.Tensor:
    """Spatially low-pass ``Y`` on the tensor-product grid (a declared bandlimit prior).

    Keeps Fourier modes with ``|k| <= cutoff_frac * max|k|`` along each spatial
    axis (zeroes the rest) and inverse-transforms; the time axis is untouched.
    Returns the smoothed field in the original row order, so all downstream
    feature columns (values, gradients, Hessians) are computed from one
    consistent denoised latent field.
    """
    grid_cols = (int(time_col), *[int(c) for c in spatial_cols])
    Y_grid, linear, vals, _shape = _tensor_grid_from_rows(X, Y, grid_cols=grid_cols)
    Yf = Y_grid
    for grid_axis in range(1, len(grid_cols)):  # spatial axes only
        n = int(vals[grid_axis].numel())
        if n <= 1:
            continue
        step = _periodic_spacing(vals[grid_axis])
        k = 2.0 * torch.pi * torch.fft.fftfreq(n, d=step, device=Yf.device, dtype=Yf.dtype)
        kmax = float(torch.abs(k).max().item())
        keep = torch.abs(k) <= float(cutoff_frac) * kmax
        m_shape = [1] * Yf.ndim
        m_shape[grid_axis] = n
        mask = keep.reshape(m_shape)
        y_hat = torch.fft.fft(Yf, dim=grid_axis)
        Yf = torch.fft.ifft(y_hat * mask, dim=grid_axis).real
    Y_lp_flat = Yf.reshape(int(X.shape[0]), int(Y.shape[1]))
    return Y_lp_flat[linear.to(device=Y.device)]


def build_spectral_spatial_derivative_targets(
    X: torch.Tensor,
    Y: torch.Tensor,
    G_time: torch.Tensor | None = None,
    *,
    spatial_cols: Sequence[int] = (1, 2, 3),
    time_col: int = 0,
    periodic: bool = True,
) -> torch.Tensor:
    """Build ``(N,n_fields,D)`` targets from FFT spatial derivatives.

    ``X`` is expected to contain a full tensor-product grid over
    ``(time_col, *spatial_cols)``.  Spatial axes are differentiated by FFT;
    the time column is copied from ``G_time`` when provided and otherwise left
    as zero.
    """
    if not periodic:
        raise NotImplementedError("non-periodic spatial derivative targets are not implemented")

    spatial = _as_axis_tuple(spatial_cols)
    tcol = int(time_col)
    if tcol < 0:
        raise ValueError(f"time_col must be non-negative, got {time_col}")
    if tcol in spatial:
        raise ValueError(f"time_col={tcol} must not also be a spatial column {spatial}")

    Xd = X.detach()
    Yd = Y.detach()
    grid_cols = (tcol, *spatial)
    Y_grid, linear, vals, shape_t = _tensor_grid_from_rows(Xd, Yd, grid_cols=grid_cols)

    n_rows = int(X.shape[0])
    n_fields = int(Y.shape[1])
    n_axes = int(X.shape[1])
    G_grid = torch.zeros(
        *[int(v) for v in shape_t.tolist()],
        n_fields,
        n_axes,
        device=Y.device,
        dtype=Y.dtype,
    )

    for grid_axis, coord_col in enumerate(grid_cols[1:], start=1):
        n = int(vals[grid_axis].numel())
        if n <= 1:
            continue
        step = _periodic_spacing(vals[grid_axis])
        k = 2.0 * torch.pi * torch.fft.fftfreq(
            n,
            d=step,
            device=Y_grid.device,
            dtype=Y_grid.dtype,
        )
        k_shape = [1] * Y_grid.ndim
        k_shape[grid_axis] = n
        ik = (1j * k).reshape(k_shape)
        y_hat = torch.fft.fft(Y_grid, dim=grid_axis)
        deriv = torch.fft.ifft(y_hat * ik, dim=grid_axis).real
        G_grid[..., int(coord_col)] = deriv

    G_flat = G_grid.reshape(n_rows, n_fields, n_axes)
    out = G_flat[linear.to(device=Y.device)]
    if G_time is not None:
        gt = _time_target_from_G(G_time.detach().to(device=Y.device, dtype=Y.dtype), time_col=tcol)
        if tuple(gt.shape) != (n_rows, n_fields):
            raise ValueError(
                f"G_time shape {tuple(gt.shape)} incompatible with expected {(n_rows, n_fields)}"
            )
        out[:, :, tcol] = gt
    return out.contiguous()


def build_spectral_spatial_hessian_diag(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    spatial_cols: Sequence[int] = (1, 2, 3),
    time_col: int = 0,
    periodic: bool = True,
) -> torch.Tensor:
    """Build ``(N,n_fields,D,D)`` spatial Hessian diagonals via FFT.

    Only the spatial second derivatives ``H[:, f, a, a]`` for ``a`` in
    ``spatial_cols`` are populated (by multiplying the FFT by ``(i k)^2 = -k^2``);
    all time and off-diagonal entries are left zero.  This is exactly what the
    vector Laplacian needs (``lap f = sum_a d2 f / d x_a^2``), and it mirrors the
    machine-precision first-derivative path in
    :func:`build_spectral_spatial_derivative_targets`.

    ``X`` must contain a full tensor-product grid over ``(time_col, *spatial_cols)``.
    Targets are machine-precision for periodic band-limited fields and
    near-exact for well-localized fields that decay to ~0 at the box edges.
    """
    if not periodic:
        raise NotImplementedError("non-periodic spatial Hessian targets are not implemented")

    spatial = _as_axis_tuple(spatial_cols)
    tcol = int(time_col)
    if tcol < 0:
        raise ValueError(f"time_col must be non-negative, got {time_col}")
    if tcol in spatial:
        raise ValueError(f"time_col={tcol} must not also be a spatial column {spatial}")

    Xd = X.detach()
    Yd = Y.detach()
    grid_cols = (tcol, *spatial)
    Y_grid, linear, vals, shape_t = _tensor_grid_from_rows(Xd, Yd, grid_cols=grid_cols)

    n_rows = int(X.shape[0])
    n_fields = int(Y.shape[1])
    n_axes = int(X.shape[1])
    H_grid = torch.zeros(
        *[int(v) for v in shape_t.tolist()],
        n_fields,
        n_axes,
        n_axes,
        device=Y.device,
        dtype=Y.dtype,
    )

    for grid_axis, coord_col in enumerate(grid_cols[1:], start=1):
        n = int(vals[grid_axis].numel())
        if n <= 1:
            continue
        step = _periodic_spacing(vals[grid_axis])
        k = 2.0 * torch.pi * torch.fft.fftfreq(
            n,
            d=step,
            device=Y_grid.device,
            dtype=Y_grid.dtype,
        )
        k_shape = [1] * Y_grid.ndim
        k_shape[grid_axis] = n
        minus_k2 = (-(k * k)).reshape(k_shape)  # (i k)^2 = -k^2
        y_hat = torch.fft.fft(Y_grid, dim=grid_axis)
        d2 = torch.fft.ifft(y_hat * minus_k2, dim=grid_axis).real
        H_grid[..., int(coord_col), int(coord_col)] = d2

    H_flat = H_grid.reshape(n_rows, n_fields, n_axes, n_axes)
    out = H_flat[linear.to(device=Y.device)]
    return out.contiguous()


def build_derivative_targets(
    mode: str,
    X: torch.Tensor,
    Y: torch.Tensor,
    G: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build derivative targets for Maxwell Sobolev surrogate training."""
    key = str(mode).strip().lower()
    if key == "exact":
        if G is None:
            raise ValueError("build_derivative_targets('exact') requires G")
        return G.detach().clone().contiguous()
    if key == "spectral_spatial_exact_time":
        if G is None:
            raise ValueError("spectral_spatial_exact_time requires exact G for the time column")
        return build_spectral_spatial_derivative_targets(
            X,
            Y,
            G_time=G[:, :, 0],
            spatial_cols=(1, 2, 3),
            time_col=0,
            periodic=True,
        )
    raise ValueError(
        f"unknown derivative target mode {mode!r}; expected 'exact' or "
        "'spectral_spatial_exact_time'"
    )
