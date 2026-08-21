# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import numpy as np
import pytest

from nestynet_sr.sr_search.representation import (
    _HAVE_SYMPY,
    _abs_node_count,
    _build_sympy_input_symbols_from_data,
    _canonicalize_inverse_ratio_powers,
    _infer_sympy_symbol_assumptions_from_samples,
)


@pytest.mark.skipif(not _HAVE_SYMPY, reason="sympy not available")
def test_infer_symbol_assumptions_positive():
    col = np.array([0.1, 1.0, 2.5], dtype=float)
    assumptions = _infer_sympy_symbol_assumptions_from_samples(col)
    assert assumptions.get("positive", False) is True


@pytest.mark.skipif(not _HAVE_SYMPY, reason="sympy not available")
def test_infer_symbol_assumptions_mixed_real_only():
    col = np.array([-2.0, 0.0, 3.0], dtype=float)
    assumptions = _infer_sympy_symbol_assumptions_from_samples(col)
    assert assumptions.get("positive", False) is False
    assert assumptions.get("nonnegative", False) is False
    assert assumptions.get("negative", False) is False
    assert assumptions.get("nonpositive", False) is False


@pytest.mark.skipif(not _HAVE_SYMPY, reason="sympy not available")
def test_positive_symbol_avoids_abs_for_sqrt_square():
    import sympy as sp

    xs_np = np.array([[1.0], [2.0], [3.0]], dtype=float)
    _, local_dict, labels = _build_sympy_input_symbols_from_data(xs_np, Nxvars=1)
    expr = sp.sympify("sqrt(x0**2)", locals=local_dict)
    assert labels["x0"] == ">0"
    assert _abs_node_count(expr) == 0


@pytest.mark.skipif(not _HAVE_SYMPY, reason="sympy not available")
def test_abs_node_count_counts_multiplicity():
    import sympy as sp

    x0 = sp.Symbol("x0", real=True)
    expr = sp.Add(sp.Abs(x0), sp.Abs(x0), evaluate=False)
    assert _abs_node_count(expr) == 2


@pytest.mark.skipif(not _HAVE_SYMPY, reason="sympy not available")
def test_canonicalize_inverse_ratio_float_exponents():
    import sympy as sp

    x0, x1 = sp.symbols("x0 x1", real=True)
    expr = sp.sympify("((x0*(x1)**-1.0)**-1.0)", locals={"x0": x0, "x1": x1})
    out = _canonicalize_inverse_ratio_powers(expr)
    assert sp.sstr(out) == "x1/x0"


@pytest.mark.skipif(not _HAVE_SYMPY, reason="sympy not available")
def test_canonicalize_double_inverse_float_exponents():
    import sympy as sp

    x0 = sp.symbols("x0", real=True)
    expr = sp.sympify("((x0**-1.0)**-1.0)", locals={"x0": x0})
    out = _canonicalize_inverse_ratio_powers(expr)
    assert sp.sstr(out) == "x0"
