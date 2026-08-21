# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_feynman_de_stlsq_dictionary_baselines.py"

spec = importlib.util.spec_from_file_location("run_feynman_de_stlsq_dictionary_baselines", SCRIPT_PATH)
assert spec is not None
baselines = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baselines
assert spec.loader is not None
spec.loader.exec_module(baselines)


def _term_names(preset: str, pid: str) -> set[str]:
    return {term.name for term in baselines.build_library_terms(preset, pid)}


def test_standard_dictionary_is_non_oracle_for_compositional_atoms():
    assert "u/(1+x0)" not in _term_names("standard", "900")
    assert "u/(1+x0)^2" not in _term_names("standard", "901")
    assert "u*log(1+x0)" not in _term_names("standard", "902")
    assert "exp(u)" not in _term_names("standard", "903")


def test_expanded_closure_dictionary_contains_carrier_products():
    names = _term_names("expanded_closure", "900")
    assert "u/(1+x0)" in names
    assert "u/(1+x0)^2" in names
    assert "u*log(1+x0)" in names
    assert "exp(u)" in names


def test_oracle_dictionary_is_case_specific():
    assert "u/(1+x0)" in _term_names("oracle", "900")
    assert "u/(1+x0)^2" in _term_names("oracle", "901")
    assert "u*log(1+x0)" in _term_names("oracle", "902")
    assert "exp(u)" in _term_names("oracle", "903")
    assert "u/(1+x0)^2" not in _term_names("oracle", "900")
    assert "exp(u)" not in _term_names("oracle", "902")


def test_standard_singular_terms_do_not_emit_runtime_warnings():
    x = np.asarray([0.0, 1.0], dtype=np.float64)
    u = np.asarray([1.0, 1.0], dtype=np.float64)
    singular_terms = [
        term for term in baselines.build_library_terms("standard", "900") if term.name in {"u/x0", "u/x0^2"}
    ]
    assert len(singular_terms) == 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = [term.fn(x, u) for term in singular_terms]

    assert caught == []
    assert all(np.isinf(value[0]) and np.isfinite(value[1]) for value in values)
