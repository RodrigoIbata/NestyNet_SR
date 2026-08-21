# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import math

import numpy as np
import sympy as sp
import torch

from nestynet_sr.sr_core.atoms import PlanckFullLeaf, PlanckLeaf
from nestynet_sr.sr_core.bridges import Var
from nestynet_sr.sr_search.polish_utils import (
    final_polish_snap_targets,
    snap_numeric_constants,
)
from nestynet_sr.sr_search.representation import (
    _planck_leaf_repr,
    _sympy_simplify_expression,
)


def _set_log_parameter(parameter, value):
    with torch.no_grad():
        parameter.copy_(torch.tensor(math.log(value), dtype=parameter.dtype))


def test_reduced_planck_matches_its_analytic_formula_away_from_singularity():
    leaf = PlanckLeaf(1, dtype=torch.float64, p=0.0)
    amp = 1.000005534878195
    rate = 0.1591557529029906
    _set_log_parameter(leaf.log_amp, amp)
    _set_log_parameter(leaf.log_a, rate)
    z = torch.linspace(0.044, 20.0, 512, dtype=torch.float64).reshape(-1, 1)

    actual = leaf(z)[:, 0]
    expected = amp / torch.expm1(rate * z[:, 0])

    assert torch.allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)


def test_full_planck_uses_a_local_singularity_guard_without_global_bias():
    leaf = PlanckFullLeaf(1, dtype=torch.float64)
    amp = 1.7
    rate = 0.35
    offset = -0.2
    power = 1.25
    _set_log_parameter(leaf.log_amp, amp)
    _set_log_parameter(leaf.log_a, rate)
    with torch.no_grad():
        leaf.b.copy_(torch.tensor(offset, dtype=torch.float64))
        leaf.p.copy_(torch.tensor(power, dtype=torch.float64))
    z = torch.linspace(1.0, 8.0, 256, dtype=torch.float64).reshape(-1, 1)

    actual = leaf(z)[:, 0]
    expected = amp * z[:, 0].pow(power) / torch.expm1(rate * z[:, 0] + offset)

    assert torch.allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)

    singular_z = torch.tensor([[-offset / rate]], dtype=torch.float64)
    assert torch.isfinite(leaf(singular_z)).all()


def test_pb085_planck_serialization_passes_stagec_fidelity_before_snapping():
    leaf = PlanckLeaf(1, dtype=torch.float64, p=0.0)
    amp = 1.000005534878195
    rate = 0.1591557529029906
    _set_log_parameter(leaf.log_amp, amp)
    _set_log_parameter(leaf.log_a, rate)
    amp_print, core_print = _planck_leaf_repr(
        leaf,
        (0,),
        input_expr=Var(0),
        sig=17,
    )
    expr = f"{amp_print:.17g}*({core_print})"
    xs = np.linspace(0.044, 20.0, 512, dtype=float).reshape(-1, 1)
    with torch.no_grad():
        ys = leaf(torch.as_tensor(xs, dtype=torch.float64))[:, 0].numpy()

    phi_str, _y_str, meta = _sympy_simplify_expression(
        expr,
        model=None,
        val_loader=None,
        device=torch.device("cpu"),
        Nxvars=1,
        prefer_stable_trig=False,
        prune_trig_poly_args=False,
        linearize_leaves=False,
        verbose=False,
        precomputed_xs_np=xs,
        precomputed_ys_model=ys,
    )

    assert meta["accepted"] is True
    assert meta["kind"] != "bad_pretty_print"
    assert phi_str is not None

    snapped = snap_numeric_constants(
        sp.sympify(expr),
        snap_targets=final_polish_snap_targets(),
        snap_rel_tol=1.0e-4,
    )
    x0 = sp.Symbol("x0")
    exact = 1 / (sp.exp(x0 / (2 * sp.pi)) - 1)
    assert sp.simplify(snapped - exact) == 0
