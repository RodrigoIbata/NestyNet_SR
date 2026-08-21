# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import importlib.util
import pytest
import sys
from pathlib import Path

from nestynet_sr.sr_core.problem_dims import (
    CanonicalProblemDims,
    canonical_constant_payload,
    canonical_scalar_dims_payload,
    canonical_target_dim_for_order,
    canonical_to_factorized_search_dims,
    canonical_to_units_spec,
    derivative_dim,
    require_dimensionless,
    units_spec_from_dim_vectors,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    prev = sys.modules.get(module_name)
    try:
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if prev is not None:
            sys.modules[module_name] = prev
        else:
            sys.modules.pop(module_name, None)


scalar_pd = _load_module(
    "feynman_de_problem_defs_testmod",
    REPO_ROOT / "examples" / "feynman_de" / "problem_defs.py",
)
complex_pd = _load_module(
    "feynman_complex_problem_defs_testmod",
    REPO_ROOT / "examples" / "feynman_complex" / "problem_defs.py",
)


def test_scalar_problem_dims_convert_to_canonical():
    dims = scalar_pd.DIMS_REGISTRY["008"]
    canonical = scalar_pd.to_canonical_problem_dims(dims)

    assert canonical.basis == ("T", "M")
    assert canonical.axis_dims == ((1.0, 0.0),)
    assert canonical.component_dims == ((0.0, 0.0),)
    assert canonical.constant_dims["g"] == (-1.0, 0.0)
    assert canonical.constant_dims["k"] == (-1.0, 1.0)
    assert canonical.constant_dims["m"] == (0.0, 1.0)
    assert canonical.complex_pairs == ()
    assert canonical.target_dims is None


def test_complex_problem_dims_convert_to_canonical():
    dims = complex_pd.get_complex_problem_dims("C004")
    assert dims is not None

    canonical = complex_pd.to_canonical_problem_dims(dims)

    assert canonical.basis == ("T",)
    assert canonical.n_axes == 1
    assert canonical.n_components == 4
    assert canonical.complex_pairs == ((0, 1), (2, 3))
    assert canonical.constant_dims["V12"] == (-1.0,)
    assert canonical.component_dims[0] == canonical.component_dims[1]


def test_lookup_helpers_return_canonical_dims():
    scalar = scalar_pd.get_canonical_problem_dims("105")
    complex_dims = complex_pd.get_canonical_problem_dims("C105")

    assert scalar is not None
    assert complex_dims is not None
    assert scalar.rank == 2
    assert complex_dims.rank == 2


def test_canonical_problem_dims_rejects_mismatched_complex_pair_units():
    with pytest.raises(ValueError, match="must share units"):
        CanonicalProblemDims(
            basis=("T",),
            axis_dims=((1.0,),),
            component_dims=((0.0,), (-1.0,)),
            complex_pairs=((0, 1),),
        )


def test_shared_dim_helpers_cover_derivative_and_dimensionless_checks():
    assert derivative_dim((0.0, 1.0), (1.0, 0.0), order=2) == (-2.0, 1.0)
    require_dimensionless((0.0, 0.0), where="trig_arg")
    with pytest.raises(ValueError, match="dimensionless"):
        require_dimensionless((1.0, 0.0), where="trig_arg")


def test_canonical_target_and_factorized_search_dims_share_order_logic():
    dims = CanonicalProblemDims.scalar(
        basis=("U", "T"),
        x_dim=(0.0, 1.0),
        u_dim=(1.0, 0.0),
        constant_dims={"omega": (0.0, -1.0)},
    )

    var_dims_1, y_dims_1 = canonical_to_factorized_search_dims(
        dims,
        order=1,
        include_x=True,
        include_u=True,
        include_du=True,
        constant_names=("omega",),
    )
    var_dims_2, y_dims_2 = canonical_to_factorized_search_dims(
        dims,
        order=2,
        include_x=True,
        include_u=True,
        include_du=True,
        constant_names=("omega",),
    )

    assert var_dims_1 == [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0)]
    assert y_dims_1 == (1.0, -1.0)
    assert var_dims_2 == [(0.0, 1.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0)]
    assert y_dims_2 == (1.0, -2.0)
    assert canonical_target_dim_for_order(dims, order=2) == (1.0, -2.0)


def test_canonical_target_override_wins_over_anchor_order():
    dims = CanonicalProblemDims(
        basis=("T",),
        axis_dims=((1.0,),),
        component_dims=((0.0,),),
        target_dims=((-3.0,),),
    )

    assert canonical_target_dim_for_order(dims, order=1) == (-3.0,)
    assert canonical_target_dim_for_order(dims, order=2) == (-3.0,)


def test_units_spec_adapters_preserve_output_dims_and_constant_modes():
    dims = CanonicalProblemDims(
        basis=("A", "T"),
        axis_dims=((0.0, 1.0),),
        component_dims=((1.0, 0.0), (2.0, 0.0)),
        constant_dims={"m": (1.0, 0.0), "omega": (0.0, -1.0)},
    )

    spec_from_canonical = canonical_to_units_spec(
        dims,
        y_component=0,
        free_constant_names=("m",),
        free_const_scope={"m": "class"},
        fixed_constant_values={"omega": 2.0},
        fixed_const_mode="minimal",
    )
    spec_from_vectors = units_spec_from_dim_vectors(
        basis=("A", "T"),
        x_dims=((0.0, 1.0),),
        y_dim=(1.0, 0.0),
        output_dims=((1.0, 0.0), (2.0, 0.0)),
        free_const_dims={"m": (1.0, 0.0)},
        free_const_scope={"m": "class"},
        fixed_const_dims={"omega": (0.0, -1.0)},
        fixed_const_values={"omega": 2.0},
        fixed_const_mode="minimal",
    )

    assert spec_from_canonical.output_dims == ((1.0, 0.0), (2.0, 0.0))
    assert spec_from_canonical.free_const_dims["m"] == (1.0, 0.0)
    assert spec_from_canonical.free_const_scope["m"] == "class"
    assert spec_from_canonical.fixed_const_dims["omega"] == (0.0, -1.0)
    assert float(spec_from_canonical.fixed_const_values["omega"]) == pytest.approx(2.0)
    assert spec_from_canonical.fixed_const_mode == "minimal"

    assert spec_from_vectors.output_dims == spec_from_canonical.output_dims
    assert spec_from_vectors.free_const_dims == spec_from_canonical.free_const_dims
    assert spec_from_vectors.fixed_const_dims == spec_from_canonical.fixed_const_dims


def test_scalar_payload_helpers_preserve_scalar_benchmark_shapes():
    dims = CanonicalProblemDims.scalar(
        basis=("D",),
        x_dim=(1.0,),
        u_dim=(0.0,),
        constant_dims={"lambda": (-1.0,), "K": (0.0,)},
    )

    dims_payload = canonical_scalar_dims_payload(dims)
    const_payload = canonical_constant_payload(
        dims,
        {"lambda": 0.5, "K": 2.0},
        names=("lambda", "K"),
    )

    assert dims_payload == {"basis": ["D"], "x": [1], "u": [0]}
    assert const_payload == [
        {"name": "lambda", "value": 0.5, "dim": [-1]},
        {"name": "K", "value": 2.0, "dim": [0]},
    ]


def test_constant_payload_omits_dim_for_undeclared_constants():
    payload = canonical_constant_payload(
        None,
        {"alpha": 3.0},
        names=("alpha",),
    )

    assert payload == [{"name": "alpha", "value": 3.0}]
