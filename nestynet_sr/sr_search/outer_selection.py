# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Prescan: Automatic outer function selection for symbolic regression.

This module provides a fast pre-screening of candidate y-transforms before
running the full Stage A separability search. It uses a lightweight
DualSegmentedAdaptor model to quickly evaluate which outer transformation
puts the data in the easiest space for neural network fitting.

The approach is adapted from NestyNet/examples/fitting/NestyNet_fit_with_outer_function.py.

Key features:
- 22 different outer transformations (identity, sqrt, log, trig, etc.)
- Units-aware filtering (rejects nonsensical transforms like "1 + 5 meters")
- Fair comparison (all evaluated in original y-space)
- Automatic selection based on validation loss

Usage:
    from nestynet_sr.sr_search.outer_selection import run_quickscan_selection

    winner_name, winner_transform, results = run_quickscan_selection(
        filepath=filepath,
        Nxvars=Nxvars,
        device=device,
        dtype=dtype,
        np_dtype=np_dtype,
        units_payload=units_payload,  # Optional
        trial_epochs=500,
        num_segments=4,
    )
"""

import copy
import math
import os
import tempfile
from typing import Callable, Dict, Optional, Tuple

import nestynet
import numpy as np
import pandas as pd
import torch

from nestynet_sr.sr_core.units import is_dimless

# ──────────────────────────────────────────────────────────────
# Transformation Class
# ──────────────────────────────────────────────────────────────


class Transformation:
    """A y-transform pair with validity and units checking.

    Each transformation defines:
    - inverse_fn: applied to y-values to create training targets t = g^(-1)(y)
    - forward_fn: applied to NN output to get predictions y = g(t)
    - check_fn: validates if transformation is valid for given data (domain check)
    - requires_dimless: True if transform requires dimensionless y (units check)
    """

    def __init__(
        self,
        name: str,
        inverse_fn: Callable[[torch.Tensor], torch.Tensor],
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        check_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        requires_dimless: bool = False,
    ):
        self.name = name
        self.inverse_fn = inverse_fn
        self.forward_fn = forward_fn
        self.check_fn = (
            check_fn
            if check_fn is not None
            else lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
        )
        self.requires_dimless = requires_dimless

    def apply_inverse(self, y: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Apply inverse transformation to y-values (creates training targets)."""
        mask = self.check_fn(y)
        if not mask.all():
            return None, mask
        return self.inverse_fn(y), mask

    def apply_forward(self, t: torch.Tensor) -> torch.Tensor:
        """Apply forward transformation to NN output (creates predictions)."""
        return self.forward_fn(t)

    def is_valid(self, y: torch.Tensor) -> bool:
        """Check if transformation is valid for given y-values (domain check)."""
        mask = self.check_fn(y)
        return mask.all().item()

    def is_valid_for_units(self, units_payload: Optional[Dict]) -> bool:
        """Check if transformation is physically valid given units.

        Returns True if:
        - No units provided (units checking disabled)
        - Transform doesn't require dimensionless y
        - Transform requires dimensionless y AND y is dimensionless
        """
        if units_payload is None:
            return True

        if not self.requires_dimless:
            return True

        y_dim = units_payload.get("y_dim")
        if y_dim is None:
            return True

        return is_dimless(y_dim)


# ──────────────────────────────────────────────────────────────
# Transformation Factories (22 total)
# ──────────────────────────────────────────────────────────────


def make_identity():
    return Transformation(
        name="identity",
        inverse_fn=lambda y: y,
        forward_fn=lambda t: t,
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=False,
    )


def make_sqrt():
    """sqrt transform: t = sqrt(|y|) (inverse), y = t^2 (forward)
    Named 'sqrt' because we take sqrt of y to get training space."""
    return Transformation(
        name="sqrt",
        inverse_fn=lambda y: torch.sqrt(torch.abs(y)),
        forward_fn=lambda t: t**2,
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=False,
    )


def make_square():
    """square transform: t = y^2 (inverse), y = sqrt(t) (forward)
    Named 'square' because we square y to get training space."""
    return Transformation(
        name="square",
        inverse_fn=lambda y: y**2,
        forward_fn=lambda t: torch.sqrt(torch.clamp(t, min=0.0)),
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=False,
    )


def make_log():
    """log transform: t = log(y) (inverse), y = exp(t) (forward)
    Named 'log' because we take log of y to get training space."""
    return Transformation(
        name="log",
        inverse_fn=lambda y: torch.log(y),
        forward_fn=lambda t: torch.exp(t),
        check_fn=lambda y: y > 0,  # Need positive values for log
        requires_dimless=True,  # log requires dimensionless argument
    )


def make_exp():
    """exp transform: t = exp(y) (inverse), y = log(t) (forward)
    Named 'exp' because we exponentiate y to get training space."""
    return Transformation(
        name="exp",
        inverse_fn=lambda y: torch.exp(y),
        forward_fn=lambda t: torch.log(torch.clamp(t, min=1e-10)),
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=True,  # exp of dimensionless gives dimensionless
    )


def make_expneg():
    """expneg transform: t = exp(-y) (inverse), y = -log(t) (forward)
    Named 'expneg' because we exponentiate -y (exp of negative y).
    Useful when y = -log(f(x)), i.e., f(x) = exp(-y)."""
    return Transformation(
        name="expneg",
        inverse_fn=lambda y: torch.exp(-y),
        forward_fn=lambda t: -torch.log(torch.clamp(t, min=1e-10)),
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=True,  # exp of dimensionless gives dimensionless
    )


def make_logneg():
    """logneg transform: t = log(-y) (inverse), y = -exp(t) (forward)
    Named 'logneg' because we take log of -y (log of negative y).
    Useful when y < 0 and y = -f(x) where f(x) is a product of powers,
    since log(-y) = log(f(x)) becomes a sum of logs."""
    return Transformation(
        name="logneg",
        inverse_fn=lambda y: torch.log(-y),
        forward_fn=lambda t: -torch.exp(t),
        check_fn=lambda y: y < 0,  # Need negative values for log(-y)
        requires_dimless=True,  # log requires dimensionless argument
    )


def make_cube():
    """cube transform: t = y^3 (inverse), y = cbrt(t) (forward)
    Named 'cube' because we cube y to get training space."""
    return Transformation(
        name="cube",
        inverse_fn=lambda y: y**3,
        forward_fn=lambda t: torch.sign(t) * torch.abs(t) ** (1 / 3),
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=False,
    )


def make_cbrt():
    """cbrt transform: t = cbrt(y) (inverse), y = t^3 (forward)
    Named 'cbrt' because we take cube root of y to get training space."""
    return Transformation(
        name="cbrt",
        inverse_fn=lambda y: torch.sign(y) * torch.abs(y) ** (1 / 3),
        forward_fn=lambda t: t**3,
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=False,
    )


def make_inv():
    """inv transform: t = 1/y (inverse), y = 1/t (forward)"""
    return Transformation(
        name="inv",
        inverse_fn=lambda y: 1.0 / y,
        forward_fn=lambda t: 1.0 / t,
        check_fn=lambda y: torch.abs(y) > 1e-10,  # Need non-zero values
        requires_dimless=False,  # Inverts dimensions: m → 1/m
    )


def make_softplus():
    """softplus transform: t = softrefine_inv(y) (inverse), y = softplus(t) (forward)"""

    def softrefine_inv(y):
        # inverse of softplus: log(exp(y) - 1)
        return torch.log(torch.exp(y) - 1.0)

    return Transformation(
        name="softplus",
        inverse_fn=softrefine_inv,
        forward_fn=lambda t: torch.nn.functional.softplus(t),
        check_fn=lambda y: y > 1e-6,  # softrefine_inv needs y > 0
        requires_dimless=True,  # Has "+1" operation
    )


def make_tanh():
    """tanh transform: t = arctanh(y) (inverse), y = tanh(t) (forward)"""
    return Transformation(
        name="tanh",
        inverse_fn=lambda y: torch.atanh(y),
        forward_fn=lambda t: torch.tanh(t),
        check_fn=lambda y: torch.abs(y) < 0.9999,  # arctanh needs |y| < 1
        requires_dimless=True,  # tanh/arctanh require dimensionless
    )


def make_invsqrt():
    """invsqrt transform: t = 1/(y^2) (inverse), y = 1/sqrt(t) (forward)"""
    eps = 1e-12
    return Transformation(
        name="invsqrt",
        inverse_fn=lambda y: 1.0 / (y**2 + eps),
        forward_fn=lambda t: 1.0 / torch.sqrt(torch.clamp(t, min=eps)),
        check_fn=lambda y: y > 0,  # 1/sqrt(·) is nonnegative
        requires_dimless=False,  # Power-law transform
    )


def make_sqrt1p():
    """sqrt1p transform: t = y^2 - 1 (inverse), y = sqrt(1 + t) (forward)."""
    eps = 1e-12
    return Transformation(
        name="sqrt1p",
        inverse_fn=lambda y: y**2 - 1.0,
        forward_fn=lambda t: torch.sqrt(torch.clamp(1.0 + t, min=eps)),
        check_fn=lambda y: y >= 0,  # sqrt(·) outputs nonnegative
        requires_dimless=True,  # Has "1 +" operation
    )


def make_invsqrt1p():
    """invsqrt1p transform: t = 1/y^2 - 1 (inverse), y = 1/sqrt(1 + t) (forward)."""
    eps = 1e-12
    return Transformation(
        name="invsqrt1p",
        inverse_fn=lambda y: 1.0 / (y**2 + eps) - 1.0,
        forward_fn=lambda t: 1.0 / torch.sqrt(torch.clamp(1.0 + t, min=eps)),
        check_fn=lambda y: y > eps,
        requires_dimless=True,  # Has "1 +" operation
    )


def make_arcsin():
    """arcsin transform: t = arcsin(y) (inverse), y = sin(t) (forward).
    Named 'arcsin' because we take arcsin of y to get training space."""
    eps = 1e-6
    return Transformation(
        name="arcsin",
        inverse_fn=lambda y: torch.asin(torch.clamp(y, -1.0 + eps, 1.0 - eps)),
        forward_fn=lambda t: torch.sin(t),
        check_fn=lambda y: torch.abs(y) <= (1.0 + 1e-6),
        requires_dimless=True,  # Trig functions require dimensionless
    )


def make_arccos():
    """arccos transform: t = arccos(y) (inverse), y = cos(t) (forward).
    Named 'arccos' because we take arccos of y to get training space."""
    eps = 1e-6
    return Transformation(
        name="arccos",
        inverse_fn=lambda y: torch.acos(torch.clamp(y, -1.0 + eps, 1.0 - eps)),
        forward_fn=lambda t: torch.cos(t),
        check_fn=lambda y: torch.abs(y) <= (1.0 + 1e-6),
        requires_dimless=True,  # Trig functions require dimensionless
    )


def make_arctan():
    """arctan transform: t = arctan(y) (inverse), y = tan(t) (forward).
    Named 'arctan' because we take arctan of y to get training space."""
    lim = (math.pi / 2) - 1e-6
    return Transformation(
        name="arctan",
        inverse_fn=lambda y: torch.atan(y),
        forward_fn=lambda t: torch.tan(torch.clamp(t, -lim, lim)),
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=True,  # Trig functions require dimensionless
    )


def make_sin():
    """sin transform: t = sin(y) (inverse), y = arcsin(t) (forward).
    Named 'sin' because we take sin of y to get training space."""
    eps = 1e-6
    lim = (math.pi / 2) - 1e-6
    return Transformation(
        name="sin",
        inverse_fn=lambda y: torch.sin(y),
        forward_fn=lambda t: torch.asin(torch.clamp(t, -1.0 + eps, 1.0 - eps)),
        check_fn=lambda y: (y >= -lim) & (y <= lim),  # asin outputs in [-pi/2, pi/2]
        requires_dimless=True,  # Trig functions require dimensionless
    )


def make_cos():
    """cos transform: t = cos(y) (inverse), y = arccos(t) (forward).
    Named 'cos' because we take cos of y to get training space."""
    eps = 1e-6
    return Transformation(
        name="cos",
        inverse_fn=lambda y: torch.cos(y),
        forward_fn=lambda t: torch.acos(torch.clamp(t, -1.0 + eps, 1.0 - eps)),
        check_fn=lambda y: (y >= -1e-6) & (y <= (math.pi + 1e-6)),  # acos outputs in [0, pi]
        requires_dimless=True,  # Trig functions require dimensionless
    )


def make_tan():
    """tan transform: t = tan(y) (inverse), y = arctan(t) (forward).
    Named 'tan' because we take tan of y to get training space."""
    lim = (math.pi / 2) - 1e-6
    return Transformation(
        name="tan",
        inverse_fn=lambda y: torch.tan(y),
        forward_fn=lambda t: torch.atan(t),
        check_fn=lambda y: (y > -lim) & (y < lim),  # keep tan(y) finite
        requires_dimless=True,  # Trig functions require dimensionless
    )


def make_sinh():
    """sinh transform: t = asinh(y) (inverse), y = sinh(t) (forward)."""
    return Transformation(
        name="sinh",
        inverse_fn=lambda y: torch.asinh(y),
        forward_fn=lambda t: torch.sinh(t),
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=True,  # Hyperbolic functions require dimensionless
    )


def make_expm1():
    """expm1 transform: t = expm1(y) (inverse), y = log1p(t) (forward).
    Named 'expm1' because we compute exp(y)-1 to get training space."""
    return Transformation(
        name="expm1",
        inverse_fn=lambda y: torch.expm1(torch.clamp(y, -50.0, 50.0)),
        forward_fn=lambda t: torch.log1p(torch.clamp(t, min=-1.0 + 1e-6)),
        check_fn=lambda y: torch.ones(y.shape[0], dtype=torch.bool, device=y.device),
        requires_dimless=True,  # Has "- 1" operation
    )


def make_log1p():
    """log1p transform: t = log1p(y) (inverse), y = expm1(t) (forward).
    Named 'log1p' because we compute log(1+y) to get training space."""
    return Transformation(
        name="log1p",
        inverse_fn=lambda y: torch.log1p(y),
        forward_fn=lambda t: torch.expm1(torch.clamp(t, -50.0, 50.0)),
        check_fn=lambda y: y > (-1.0 + 1e-6),  # log1p needs y > -1
        requires_dimless=True,  # Has "1 +" operation
    )


def make_invexpm1():
    """invexpm1 transform: y = 1/(exp(t)-1),  t = log1p(1/y)."""
    eps = 1e-12
    return Transformation(
        name="invexpm1",
        inverse_fn=lambda y: torch.log1p(1.0 / torch.clamp(y, min=eps)),
        forward_fn=lambda t: 1.0 / torch.clamp(torch.expm1(torch.clamp(t, -50.0, 50.0)), min=eps),
        check_fn=lambda y: y > eps,
        requires_dimless=True,  # Has "1 +" and "- 1" operations
    )


# Registry of transformations used by the main SR pipeline.
#
# We keep the quickscan transform set aligned with Stage-A y-transforms (see
# sr_search.y_transforms) so that quickscan winners are always actionable.
#
# NOTE: The older, larger quickscan-only factory set (invsqrt, softplus, etc.)
# is still available in this module, but we no longer make it the default
# to avoid drift and name mismatches.
try:
    from nestynet_sr.sr_search.y_transforms import get_y_transform_registry

    def _factory_from_ytransform(yt):
        def _all_true(y):
            return torch.ones_like(y, dtype=torch.bool)

        check = getattr(yt, 'check_fn', None) or _all_true
        inv_fn = yt.torch_op if getattr(yt, 'torch_op', None) is not None else (lambda y: y)
        fwd_fn = yt.torch_inv if getattr(yt, 'torch_inv', None) is not None else (lambda t: t)
        requires_dimless = bool(getattr(yt, 'requires_dimless', False))

        def _make():
            return Transformation(
                name=str(getattr(yt, 'name', 'unknown')),
                inverse_fn=inv_fn,
                forward_fn=fwd_fn,
                check_fn=check,
                requires_dimless=requires_dimless,
            )

        return _make

    _YTS = get_y_transform_registry()
    TRANSFORMATIONS = {yt.name: _factory_from_ytransform(yt) for yt in _YTS}
except Exception:
    # Fallback to the legacy local factory list if y_transforms cannot be imported.
    TRANSFORMATIONS = {
        'identity': make_identity,
        'sqrt': make_sqrt,
        'invsqrt': make_invsqrt,
        'sqrt1p': make_sqrt1p,
        'invsqrt1p': make_invsqrt1p,
        'square': make_square,
        'log': make_log,
        'logneg': make_logneg,
        'exp': make_exp,
        'expneg': make_expneg,
        'expm1': make_expm1,
        'log1p': make_log1p,
        'invexpm1': make_invexpm1,
        'cube': make_cube,
        'cbrt': make_cbrt,
        'inv': make_inv,
        'softplus': make_softplus,
        'tanh': make_tanh,
        'sinh': make_sinh,
        'arcsin': make_arcsin,
        'arccos': make_arccos,
        'arctan': make_arctan,
        'sin': make_sin,
        'cos': make_cos,
        'tan': make_tan,
    }


# ──────────────────────────────────────────────────────────────
# Model Creation and Training
# ──────────────────────────────────────────────────────────────

# Cache to ensure all transformations see identical train/val splits
_CSV_CACHE = {}


def _cached_csv_loader(fp):
    """Cache CSV loads to avoid repeated I/O and ensure consistent data."""
    v = _CSV_CACHE.get(fp, None)
    if v is None:
        v = nestynet.dataloader.get_csv_data_as_pandas(fp)
        _CSV_CACHE[fp] = v
    return v


def create_model_with_transformation(
    filepath: str,
    transformation: Transformation,
    nseg: int,
    device: torch.device,
    dtype: torch.dtype,
    np_dtype: np.dtype,
    batch_size: int,
    ndata_select: int,
    ndata_select_val: int,
    Gmodel_scale: float = 0.1,
    lm_verbose: bool = False,
):
    """Create DualSegmentedAdaptor with specified transformation.

    Returns:
        lm_optimizer, adaptor, train_loader, val_loader, validity_info
        Returns None, None, None, None, validity_info if transformation is invalid.
    """
    # Load data (using cache for consistency)
    x_data, y_data, Nxvars = _cached_csv_loader(filepath)
    Nyvars = y_data.shape[1]

    # Convert to tensors for transformation
    y_tensor = torch.tensor(y_data.values, dtype=dtype, device=device)

    # Check if transformation is valid for this data (domain check)
    if not transformation.is_valid(y_tensor):
        mask = transformation.check_fn(y_tensor)
        valid_frac = mask.float().mean().item()
        return None, None, None, None, {"valid": False, "valid_frac": valid_frac}

    # Apply inverse transformation to create training targets
    t_tensor, mask = transformation.apply_inverse(y_tensor)

    if t_tensor is None:
        valid_frac = mask.float().mean().item() if mask is not None else 0.0
        return None, None, None, None, {"valid": False, "valid_frac": valid_frac}

    # Convert back to numpy for dataset creation
    t_data = t_tensor.cpu().numpy()

    # Save transformed data temporarily
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Create dataframe with x and transformed y
        df = pd.DataFrame(x_data)
        for i in range(Nyvars):
            df[f"y{i}"] = t_data[:, i]
        df.to_csv(f.name, index=False)
        transformed_filepath = f.name

    # Create models
    Nout_mid = Nxvars + 2
    model1 = (
        nestynet.nets.NestyNet_Model(
            "G_Model", Nout_mid, Nxvars, nseg, Gmodel_scale, dtype, device, seg_width=1
        )
        .to(device)
        .to(dtype)
    )

    model2 = (
        nestynet.nets.NestyNet_Model("G_Model", Nyvars, Nout_mid, nseg, Gmodel_scale, dtype, device)
        .to(device)
        .to(dtype)
    )

    # Create datasets with transformed targets (using cached loader for consistency)
    dataset_train = nestynet.dataloader.PhysDataset(
        transformed_filepath,
        mode="train",
        # Source-order split, matching the rest of the SR pipeline; the
        # untransformed validation split below must select the same rows.
        split_policy="contiguous",
        data_loader=_cached_csv_loader,
        ndata_select=ndata_select,
        ndata_select_val=ndata_select_val,
        Nxvars=Nxvars,
        np_dtype=np_dtype,
    )
    datagen_train_noshuffle = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=False, drop_last=True
    )

    dataset_val = nestynet.dataloader.PhysDataset(
        transformed_filepath,
        mode="validation",
        split_policy="contiguous",  # must match dataset_train; see note above
        data_loader=_cached_csv_loader,
        ndata_select=ndata_select,
        ndata_select_val=ndata_select_val,
        Nxvars=Nxvars,
        np_dtype=np_dtype,
    )
    datagen_val_noshuffle = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # Create adaptors
    seg_adapt1 = nestynet.adaptors.SegmentedAdaptor(
        model1, segments=torch.arange(nseg), block_size_target=None, shuffle_parameters=False
    )
    seg_adapt2 = nestynet.adaptors.SegmentedAdaptor(
        model2, segments=torch.arange(nseg), block_size_target=None, shuffle_parameters=False
    )

    # Create DualSegmentedAdaptor (trains on transformed targets)
    adapt = nestynet.adaptors.DualSegmentedAdaptor(seg_adapt1, seg_adapt2)

    params = list(adapt.parameters())

    def factory(dataloader):
        def _factory(_optim):
            return nestynet.optimizer.ResidualsModule(
                providers=[adapt],
                dataloader=dataloader,
                device=device,
            )

        return _factory

    residual_module_factories = [factory(datagen_train_noshuffle)]
    residual_module_factories_val = [factory(datagen_val_noshuffle)]

    # Prescan: disable early convergence to ensure all transforms get full trial_epochs
    from nestynet_sr.sr_search.training import SR_LM_OVERRIDES

    cfg = nestynet.optimizer.LMConfig(LM_strategy="direct_solve", iter_info=50, chisq_tol=0.0, verbose=lm_verbose, **SR_LM_OVERRIDES)

    lm_optimizer = nestynet.optimizer.Predictive_LM_Optimizer(
        params,
        residual_module_factories,
        residual_module_factories_val=residual_module_factories_val,
        cfg=cfg,
    )

    # Clean up temporary file
    os.unlink(transformed_filepath)

    return (
        lm_optimizer,
        adapt,
        datagen_train_noshuffle,
        datagen_val_noshuffle,
        {"valid": True, "valid_frac": 1.0},
    )


def compute_loss_in_y_space(
    adaptor,
    transformation: Transformation,
    filepath: str,
    device: torch.device,
    dtype: torch.dtype,
    np_dtype: np.dtype,
    ndata_select: int,
    ndata_select_val: int,
):
    """Compute validation loss in original y-space.

    This evaluates: ||y - g(NN(x))||^2 where g is the forward transformation.
    This ensures fair comparison between different transformations.
    """
    # Load original (untransformed) validation data (using cache for consistency)
    x_data, y_data, Nxvars = _cached_csv_loader(filepath)

    # Get validation split using the SAME dataset creation logic
    dataset_val = nestynet.dataloader.PhysDataset(
        filepath,
        mode="validation",
        split_policy="contiguous",  # same rows as the transformed-target split
        data_loader=_cached_csv_loader,
        ndata_select=ndata_select,
        ndata_select_val=ndata_select_val,
        Nxvars=Nxvars,
        np_dtype=np_dtype,
    )

    # Collect all validation data by iterating through the dataset
    x_val_list = []
    y_val_list = []
    for i in range(len(dataset_val)):
        x_i, y_i = dataset_val[i]
        x_val_list.append(torch.tensor(x_i, dtype=dtype))
        y_val_list.append(torch.tensor(y_i, dtype=dtype))

    # Stack into tensors and move to device
    x_val_tensor = torch.stack(x_val_list).to(device)
    y_val_tensor = torch.stack(y_val_list).to(device)

    # Evaluate on full validation set at once
    adaptor.eval()

    with torch.no_grad():
        # Get NN predictions in t-space
        t_pred = adaptor(x_val_tensor)

        # Apply forward transformation to get y-space predictions
        y_pred = transformation.apply_forward(t_pred)

        # Compute MSE in y-space
        mse = torch.mean((y_pred - y_val_tensor) ** 2)

    adaptor.train()
    return mse.item()


def _quickscan_separability_check(
    model,
    train_loader,
    Nxvars: int,
    device: torch.device,
    precision: float = 0.01,
) -> Tuple[bool, Optional[float]]:
    """Quick separability check for quickscan models.

    Returns:
        has_separability: True if additive or multiplicative separability is detected.
        sep_metric: The minimum median absolute mixed second derivative across all
                   variable pairs (normalized by y_mad). Lower = more separable.
                   None if computation failed.

    Uses relaxed precision (0.01) for speed, same as _quick_separability_check.
    """
    from nestynet_sr.sr_core.separability_math import check_separability

    try:
        symb = list(range(Nxvars))

        # Run separability check - now returns the sep_metric as 5th element
        proposed, _, _, _, sep_metric = check_separability(
            symb=symb,
            index=0,
            model=model,
            datagen=train_loader,
            precision_sum=precision,
            precision_mult=precision,
            device=device,
        )
        has_separability = len(proposed) > 0

        return has_separability, sep_metric
    except Exception:
        return False, None


def trial_transformation(
    filepath: str,
    transformation_name: str,
    transformation: Transformation,
    max_epochs: int,
    nseg: int,
    device: torch.device,
    dtype: torch.dtype,
    np_dtype: np.dtype,
    batch_size: int,
    ndata_select: int,
    ndata_select_val: int,
    verbose: bool = True,
    lm_verbose: bool = False,
):
    """Trial run with specified transformation.

    Returns:
        best_train_loss_t, best_val_loss_y, valid, adaptor, train_loader

    Note: Returns loss at BEST validation checkpoint (t-space), evaluated in y-space.
    Also returns adaptor and train_loader for optional separability check.
    """
    if verbose:
        print(f"  Trying: {transformation_name:<15}", end=" ", flush=True)

    from nestynet_sr.sr_search.training import _sr_latest_single_target_loss_metrics

    result = create_model_with_transformation(
        filepath,
        transformation,
        nseg,
        device,
        dtype,
        np_dtype,
        batch_size,
        ndata_select,
        ndata_select_val,
        lm_verbose=lm_verbose,
    )

    if result[0] is None:
        if verbose:
            validity_info = result[4]
            valid_pct = 100 * validity_info["valid_frac"]
            # Show more precision if close to 100% to avoid confusing "100.0% valid" skip messages
            if valid_pct >= 99.9:
                print(f"SKIP ({valid_pct:.4f}% of data valid, need 100%)")
            else:
                print(f"SKIP ({valid_pct:.1f}% of data valid, need 100%)")
        return None, None, False, None, None

    lm_opt, adaptor, train_loader, val_loader, validity_info = result

    # Track best model state
    best_val_loss_t = float("inf")
    best_epoch = 0
    best_state_dict = None
    training_losses_t = []

    for epoch in range(max_epochs + 1):
        loss_obj_t, loss_val_obj_t = lm_opt.step()
        loss_metrics = _sr_latest_single_target_loss_metrics(
            lm_opt, label="[quickscan] "
        )
        loss_t = float(loss_metrics.get("train_data_mean_loss", loss_obj_t))
        raw_val_t = loss_metrics.get("val_data_mean_loss", loss_val_obj_t)
        loss_val_t = None if raw_val_t is None else float(raw_val_t)
        training_losses_t.append(loss_t)

        # Save best checkpoint based on t-space validation loss
        if loss_val_t is not None and loss_val_t < best_val_loss_t:
            best_val_loss_t = loss_val_t
            best_epoch = epoch
            best_state_dict = copy.deepcopy(adaptor.state_dict())

        if lm_opt.state.get("halt"):
            break

    # Restore best checkpoint
    if best_state_dict is not None:
        adaptor.load_state_dict(best_state_dict)
        best_train_loss_t = training_losses_t[best_epoch]
    else:
        best_train_loss_t = training_losses_t[-1]

    # CRITICAL: Evaluate in original y-space using BEST checkpoint
    best_val_loss_y = compute_loss_in_y_space(
        adaptor, transformation, filepath, device, dtype, np_dtype, ndata_select, ndata_select_val
    )

    if verbose:
        print(f"y-loss: {best_val_loss_y:.6e} (epoch {best_epoch})")

    return best_train_loss_t, best_val_loss_y, True, adaptor, train_loader


def run_quickscan_selection(
    filepath: str,
    Nxvars: int,
    device: torch.device,
    dtype: torch.dtype,
    np_dtype: np.dtype,
    units_payload: Optional[Dict] = None,
    trial_epochs: int = 500,
    num_segments: int = 4,
    batch_size: int = 500,
    ndata_select: Optional[int] = None,
    ndata_select_val: Optional[int] = None,
    verbose: bool = True,
    margin_ratio: float = 10.0,
    quickscan_max_valloss: float = 1.0,
    lm_verbose: bool = False,
) -> Tuple[Optional[str], Optional[Transformation], Optional[Dict]]:
    """Run quickscan: automatic outer function selection.

    Tests all 22 transformations (filtered by units if applicable) and selects
    the best one based on validation loss in original y-space.

    The winner is only accepted if it beats the second-best by a factor of
    margin_ratio (default 10.0). Otherwise returns None to indicate no clear
    winner was found.

    Args:
        filepath: Path to CSV data file
        Nxvars: Number of input variables
        device: PyTorch device
        dtype: PyTorch dtype
        np_dtype: NumPy dtype
        units_payload: Optional units information (from run_SR.py)
        trial_epochs: Number of epochs for each trial
        num_segments: Number of segments for DualSegmentedAdaptor
        batch_size: Batch size for training
        ndata_select: Training data size (default: batch_size * 1)
        ndata_select_val: Validation data size (default: batch_size * 1)
        verbose: Print progress information
        margin_ratio: Required ratio of 2nd_best_loss / winner_loss for
            winner to be accepted (default 10.0)
        quickscan_max_valloss: Maximum val_loss (y-space) for a transform to be
            considered in SepMetric ranking (default 1.0). Transforms with
            higher val_loss are excluded from SepMetric-based selection.

    Returns:
        best_transform_name: Name of winning transformation, or None if no clear winner
        best_transformation: Winning Transformation object, or None if no clear winner
        results: Dict with all trial results
    """
    if ndata_select is None:
        ndata_select = batch_size * 1
    if ndata_select_val is None:
        ndata_select_val = batch_size * 1

    if verbose:
        print("\n" + "=" * 80)
        print("QUICKSCAN: Outer Function Auto-Selection")
        print("=" * 80)
        print(f"Testing transformations with {trial_epochs} epochs each")
        print(f"Using DualSegmentedAdaptor with {num_segments} segments")
        print("Selection metric: Validation loss in ORIGINAL y-space")
        print("Winner will be tried first in Stage A (with fallback)")

        # Report units filtering if applicable
        if units_payload is not None:
            y_dim = units_payload.get("y_dim")
            if y_dim is not None:
                us = units_payload.get("unit_system")
                if is_dimless(y_dim):
                    print("Units: y is dimensionless (all transforms allowed)")
                else:
                    y_dim_str = us.format_dim(y_dim) if us else str(y_dim)
                    print(f"Units: y has dimension {y_dim_str}")
                    print("       Filtering out transforms requiring dimensionless y")
        print()

    # Filter transformations based on units (if provided)
    active_transforms = {}
    skipped_by_units = []

    for name, factory in TRANSFORMATIONS.items():
        transform = factory()
        if transform.is_valid_for_units(units_payload):
            active_transforms[name] = factory
        else:
            skipped_by_units.append(name)

    if verbose and skipped_by_units:
        print(f"Skipped by units: {', '.join(skipped_by_units)}")
        print()

    # Run trials
    results = {}

    for transform_name, transform_factory in active_transforms.items():
        transformation = transform_factory()
        train_loss_t, val_loss_y, valid, adaptor, train_loader = trial_transformation(
            filepath,
            transform_name,
            transformation,
            trial_epochs,
            num_segments,
            device,
            dtype,
            np_dtype,
            batch_size,
            ndata_select,
            ndata_select_val,
            verbose=verbose,
            lm_verbose=lm_verbose,
        )

        if valid:
            # Quick separability check on the trained model
            has_separability, sep_metric = _quickscan_separability_check(
                adaptor, train_loader, Nxvars, device
            )
            results[transform_name] = {
                "train_loss_t": train_loss_t,
                "val_loss_y": val_loss_y,
                "transformation": transformation,
                "has_separability": has_separability,
                "sep_metric": sep_metric,
            }

    if not results:
        print("ERROR: No valid transformations found!")
        return None, None, None

    # Select best based on validation loss IN Y-SPACE
    sorted_results = sorted(results.items(), key=lambda x: x[1]["val_loss_y"])
    best_transform_name = sorted_results[0][0]
    best_transformation = results[best_transform_name]["transformation"]
    best_loss = sorted_results[0][1]["val_loss_y"]

    # Check if winner beats 2nd place by required margin
    actual_ratio = None
    if len(sorted_results) >= 2:
        second_best_loss = sorted_results[1][1]["val_loss_y"]
        if best_loss > 0:
            actual_ratio = second_best_loss / best_loss
        else:
            # Perfect fit (loss = 0), consider it a clear winner
            actual_ratio = float("inf")
    else:
        # Only one valid transform, consider it a clear winner by default
        pass

    if verbose:
        print("\n" + "=" * 80)
        print("QUICKSCAN RESULTS")
        print("=" * 80)
        print(f"{'Transformation':<15} {'Train (t-space)':>17} {'Val (y-space)':>17} {'Factor':>8} {'Sep':>5} {'SepMetric':>9}")
        print("-" * 81)

        # Show all results
        shown_names = set()
        for i, (transform_name, info) in enumerate(sorted_results):
            marker = " <-- BEST" if transform_name == best_transform_name else ""
            factor = info['val_loss_y'] / best_loss if best_loss > 0 else float('inf')
            has_sep = info.get('has_separability', False)
            sep_yn = "  YES" if has_sep else "   NO"
            sep_metric = info.get('sep_metric')
            sep_str = f"{sep_metric:>9.1e}" if sep_metric is not None else "        -"
            print(
                f"{transform_name:<15} {info['train_loss_t']:>17.6e} {info['val_loss_y']:>17.6e} {factor:>8.2f} {sep_yn} {sep_str}{marker}"
            )
            shown_names.add(transform_name)

        print("-" * 81)
        print(f"Best transform: {best_transform_name}")
        print(f"Best y-space validation loss: {best_loss:.6e}")

        if len(sorted_results) >= 2:
            second_name = sorted_results[1][0]
            second_loss = sorted_results[1][1]["val_loss_y"]
            print(f"2nd best: {second_name} (loss: {second_loss:.6e})")
            print(f"Margin ratio: {actual_ratio:.2f}x (required: {margin_ratio:.1f}x)")

        print("\n(Final selection determined by SepMetric ranking below)")
        print("=" * 80)

    # --- SepMetric Ranking (separable transforms only) ---
    separable_results = [(name, info) for name, info in results.items()
                         if info.get("has_separability") and info.get("sep_metric") is not None]

    if separable_results:
        # Filter out transforms with val_loss above the threshold (likely bad fits).
        # A low SepMetric is meaningless if the model can't fit the data.
        max_acceptable_loss = quickscan_max_valloss

        filtered_sep = [(name, info) for name, info in separable_results
                        if info["val_loss_y"] <= max_acceptable_loss]

        if filtered_sep:
            # Sort by SepMetric (primary), then val_loss_y (tiebreaker)
            sep_sorted = sorted(filtered_sep,
                               key=lambda x: (x[1]["sep_metric"], x[1]["val_loss_y"]))
        else:
            # All separable transforms have high val_loss - fail quickscan
            if verbose:
                n_filtered = len(separable_results)
                print("\nSEPMETRIC RANKING (separable transforms only):")
                print(f"  ({n_filtered} transform(s) excluded due to val_loss > {max_acceptable_loss:.1e})")
                print("  No acceptable separable transforms found - quickscan failed.")
            return None, None, results

        if verbose:
            n_filtered = len(separable_results) - len(filtered_sep)
            print("\nSEPMETRIC RANKING (separable transforms only):")
            if n_filtered > 0:
                print(f"  ({n_filtered} transform(s) excluded due to val_loss > {max_acceptable_loss:.1e})")
            print(f"{'Rank':<6} {'Transform':<15} {'SepMetric':>12} {'Val (y-space)':>17}")
            print("-" * 52)
            for rank, (name, info) in enumerate(sep_sorted, 1):
                marker = " <-- SELECTED" if rank == 1 else ""
                print(f"{rank:<6} {name:<15} {info['sep_metric']:>12.1e} {info['val_loss_y']:>17.6e}{marker}")
            print(f"\nSELECTED: '{sep_sorted[0][0]}' (lowest SepMetric + val_loss tiebreaker)")

        # Override selection to use SepMetric winner
        best_transform_name = sep_sorted[0][0]
        best_transformation = results[best_transform_name]["transformation"]
        return best_transform_name, best_transformation, results
    else:
        if verbose:
            print("\nNo separable transforms found - using fit accuracy winner.")
            print(f"SELECTED: '{best_transform_name}' (best fit accuracy)")
        return best_transform_name, best_transformation, results
