# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from nestynet_sr.sr_core.bridges import AtomNode, MulNode
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search.stageB.transfer_basis import build_transfer_basis


def _spec(us, x_dims, y_dim, *, free_const_dims=None):
    return UnitsSpec(
        unit_system=us,
        x_dims=tuple(x_dims),
        y_dim=y_dim,
        free_const_dims=free_const_dims or {},
    )


def _descs(features):
    return [f.desc for f in features]


def test_transfer_basis_generates_log_and_sqrt_only_for_dimensionless_ratio():
    us = UnitSystem(("L", "T"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L], dimless)

    features = build_transfer_basis(
        shared_vars=(0, 1),
        units_spec=spec,
        required_dim=dimless,
        max_features=64,
    )
    descs = _descs(features)

    assert any("log(x0/x1)" in d or "log(x1/x0)" in d for d in descs)
    assert any("sqrt(1-(x0/x1)^2)" in d or "sqrt(1-(x1/x0)^2)" in d for d in descs)
    assert not any("log(x0)" in d or "log(x1)" in d for d in descs)
    assert not any("sqrt(1-(x0)^2)" in d or "sqrt(1-(x1)^2)" in d for d in descs)


def test_transfer_basis_keeps_dimensionless_ratio_in_small_budget():
    us = UnitSystem(("L", "T"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L], dimless)

    features = build_transfer_basis(
        shared_vars=(0, 1),
        units_spec=spec,
        required_dim=dimless,
        max_features=16,
    )
    descs = _descs(features)

    assert any("x0/x1" in d or "x1/x0" in d for d in descs)
    assert any("sqrt(1-(x0/x1)^2)" in d or "sqrt(1-(x1/x0)^2)" in d for d in descs)


def test_transfer_basis_requires_declared_unitful_constant_for_unitful_constant_transfer():
    us = UnitSystem(("L", "T"))
    L = us.dim({"L": 1})

    no_const = _spec(us, [L], L)
    features_no_const = build_transfer_basis(
        shared_vars=(0,),
        units_spec=no_const,
        required_dim=L,
        max_features=16,
    )
    assert "C" not in _descs(features_no_const)

    with_const = _spec(us, [L], L, free_const_dims={"length_scale": L})
    features_with_const = build_transfer_basis(
        shared_vars=(0,),
        units_spec=with_const,
        required_dim=L,
        max_features=16,
    )
    const_feature = next(f for f in features_with_const if f.desc == "C")

    assert isinstance(const_feature.expr, AtomNode)
    assert str(const_feature.expr.kind).lower() == "free_const"
    assert const_feature.expr.kwargs["name"] == "length_scale"


def test_transfer_basis_allows_dimensionful_z_when_coefficient_units_are_dimless():
    us = UnitSystem(("L", "T"))
    L = us.dim({"L": 1})
    spec = _spec(us, [L], L)

    features = build_transfer_basis(
        shared_vars=(0,),
        units_spec=spec,
        required_dim=L,
        max_features=16,
    )
    descs = _descs(features)

    assert "C*x0" in descs
    feat = next(f for f in features if f.desc == "C*x0")
    assert isinstance(feat.expr, MulNode)
    assert not any("log(x0)" in d for d in descs)


def test_transfer_basis_uses_shared_input_expressions():
    us = UnitSystem(("L", "T"))
    L = us.dim({"L": 1})
    dimless = us.dimless()
    spec = _spec(us, [L, L], dimless)
    product = MulNode(AtomNode(kind="var", var_idxs=(0,)), AtomNode(kind="var", var_idxs=(1,)))

    features = build_transfer_basis(
        shared_vars=(0, 1),
        shared_inputs=(product,),
        units_spec=spec,
        required_dim=us.dim({"L": 2}),
        max_features=16,
        strict_units=False,
    )

    assert any(f.desc == "C*arg0" for f in features)


def test_transfer_basis_without_units_allows_nonlinear_terms():
    features = build_transfer_basis(
        shared_vars=(0,),
        units_spec=None,
        required_dim=None,
        max_features=16,
    )
    descs = _descs(features)

    assert "C*log(x0)" in descs
    assert "C*sqrt(1-(x0)^2)" in descs
