# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.

"""Cross-package regressions for VarPro LM operator construction."""

import math

import pytest
import torch

from nestynet_sr.sr_de.de_templates import get_template
from nestynet_sr.sr_de.varpro_de import (
    _lm_optimize_template_params,
    _lm_optimize_template_params_multi,
)


class _ZeroSurrogate(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)

    def grad(self, x):
        return torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)

    def grad_grad(self, x):
        return torch.zeros(
            x.shape[0], x.shape[1], x.shape[1], dtype=x.dtype, device=x.device
        )


@pytest.fixture
def exponential_problem():
    x = torch.linspace(-0.5, 0.5, 12, dtype=torch.float64).unsqueeze(1)
    y = 1.7 * torch.exp(0.8 * x[:, 0])
    template = get_template("exp").build_instances(
        x_vars=[0], include_u=False, include_du=False, x_axis=0
    )[0]
    return x, y, template


@pytest.mark.parametrize("strategy", ["direct_solve", "matfree"])
def test_single_dataset_varpro_lm_builds_jtj_operators(exponential_problem, strategy):
    x, y, template = exponential_problem

    psi, loss = _lm_optimize_template_params(
        baseline_term_asts=[],
        template=template,
        surrogate=_ZeroSurrogate(),
        X_train=x,
        y_train=y,
        order=1,
        x_axis=0,
        device=torch.device("cpu"),
        ridge=1e-10,
        lm_epochs=2,
        lm_epochs_min=0,
        lm_nval_patience=2,
        lm_strategy=strategy,
        verbose=False,
    )

    assert math.isfinite(loss)
    assert math.isfinite(psi["k_x0"])
    assert psi["k_x0"] != pytest.approx(template.params["k_x0"])


def test_multi_dataset_varpro_lm_builds_shared_jtj_operators(exponential_problem):
    x, y, template = exponential_problem

    psi, loss = _lm_optimize_template_params_multi(
        baseline_term_asts=[],
        template=template,
        surrogates=[_ZeroSurrogate(), _ZeroSurrogate()],
        X_train_list=[x, x],
        y_train_list=[y, 0.5 * y],
        order=1,
        x_axis=0,
        device=torch.device("cpu"),
        ridge=1e-10,
        lm_epochs=2,
        lm_epochs_min=0,
        lm_nval_patience=2,
        lm_strategy="direct_solve",
        verbose=False,
    )

    assert math.isfinite(loss)
    assert math.isfinite(psi["k_x0"])
    assert psi["k_x0"] != pytest.approx(template.params["k_x0"])
