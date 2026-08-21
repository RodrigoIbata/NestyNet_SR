# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core.bridges import U
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_de.de_search import required_coeff_dim_for_term
from nestynet_sr.sr_de.system_de_search import (
    VectorDESearchConfig,
    VectorEquationSpec,
    VectorSystemDESearchConfig,
    _vector_term_units_for_equation,
    discover_vector_system_de_from_surrogate,
    share_coeff_by_term,
)

torch.set_default_dtype(torch.float64)


def _spec():
    us = UnitSystem(("A", "B", "C", "T"))
    return us, UnitsSpec(
        unit_system=us,
        x_dims=(us.dim((0, 0, 0, 1)),),
        y_dim=us.dim((1, 0, 0, 0)),
        output_dims=(
            us.dim((1, 0, 0, 0)),
            us.dim((1, 0, 0, 0)),
            us.dim((0, 1, 0, 0)),
            us.dim((0, 0, 1, 0)),
        ),
        free_const_dims={
            "c_ab": us.dim((1, 0, -1, -1)),
            "c_ac": us.dim((1, -1, 0, -1)),
            "c_t": us.dim((0, 0, 0, -1)),
            "c_ba_t": us.dim((-1, 1, 0, -1)),
        },
    )


class _ZeroVectorSurrogate(torch.nn.Module):
    def __init__(self, ny: int):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self._ny = int(ny)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self._ny, dtype=x.dtype, device=x.device)


def test_required_coeff_dim_for_term_uses_output_dims():
    us, spec = _spec()
    local_spec = replace(spec, y_dim=spec.output_dims[0])

    req = required_coeff_dim_for_term(
        U(out_idx=2),
        order=1,
        x_axis=0,
        units_spec=local_spec,
    )

    assert req == us.dim((1, -1, 0, -1))


def test_vector_term_units_rejects_component_tie_dim_mismatch():
    _us, spec = _spec()

    ok, req_dim, why = _vector_term_units_for_equation(
        (U(out_idx=2), U(out_idx=3)),
        equation=VectorEquationSpec(out_idxs=(0, 1), name="bad_vec"),
        order=1,
        x_axis=0,
        units_spec=spec,
        output_dims=spec.output_dims,
        enforce_units=True,
    )

    assert ok is False
    assert req_dim is None
    assert "inconsistent dimensions across components" in why
    assert "out_idx=0" in why
    assert "out_idx=1" in why


def test_vector_system_coeff_share_group_rejects_cross_equation_dim_mismatch():
    _us, spec = _spec()
    X = torch.linspace(0.0, 1.0, 16, dtype=torch.float64).unsqueeze(1)
    loader = DataLoader(TensorDataset(X), batch_size=len(X), shuffle=False)
    surrogate = _ZeroVectorSurrogate(ny=4)

    cfg = VectorSystemDESearchConfig(
        x_axis=0,
        order_candidates=(1,),
        include_const=False,
        units_spec=spec,
        enforce_units=True,
        coeff_share_groups=(share_coeff_by_term("g_shared", 0, eq_idxs=(0, 1)),),
    )

    equations = (
        VectorEquationSpec(out_idxs=(0,), name="eq0"),
        VectorEquationSpec(out_idxs=(2,), name="eq1"),
    )

    with pytest.raises(ValueError, match="g_shared.*incompatible coefficient dimensions"):
        discover_vector_system_de_from_surrogate(
            surrogate,
            loader,
            cfg=cfg,
            equations=equations,
            vector_terms=[(U(out_idx=1),)],
            device=torch.device("cpu"),
        )


def test_vector_de_config_exposes_units_fields():
    _us, spec = _spec()
    cfg = VectorDESearchConfig(units_spec=spec, enforce_units=True)
    assert cfg.units_spec is spec
    assert cfg.enforce_units is True
