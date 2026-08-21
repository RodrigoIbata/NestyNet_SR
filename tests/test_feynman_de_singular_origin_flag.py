# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""The inverse-coordinate atom gate is keyed on declared class metadata.

The benchmark harness enables the inverse-coordinate atoms (u'/x, u/x, u/x^2)
for problems whose benchmark line declares the ``singular_origin`` flag.  An
earlier implementation string-matched the ground-truth RHS instead, which
contradicted the paper's answer-blind library claim.  These tests pin two
facts: the declared flag set is exactly the historical nine cases, and it
coincides with what the retired string-match would select on the current data
file, so the re-key changed no benchmark result.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FEYNMAN_DE_DIR = REPO_ROOT / "examples" / "feynman_de"
if str(FEYNMAN_DE_DIR) not in sys.path:
    sys.path.insert(0, str(FEYNMAN_DE_DIR))

import problem_defs as pd  # noqa: E402


EXPECTED_SINGULAR_ORIGIN_IDS = {
    "010",  # Radial inflow
    "113",  # Lane-Emden (general n)
    "114",  # Lane-Emden n=1
    "115",  # Lane-Emden n=3
    "116",  # Isothermal sphere
    "117",  # Bessel equation
    "118",  # Bessel J_0
    "128",  # Spherical acoustic wave
    "206",  # Exact, separable (-y/x)
}


def _problems():
    return pd.load_problems(REPO_ROOT / "data" / "feynman_de_benchmark.txt")


def _retired_rhs_string_match(problem: pd.ProblemDef) -> bool:
    """The retired answer-reading gate, kept here only to pin equivalence."""
    rhs = ""
    if "=" in str(problem.equation):
        rhs = str(problem.equation).split("=", 1)[1]
    iv = str(problem.indep_var)
    patterns = (
        f"/{iv}",
        f"{iv}**-1",
        f"{iv}**(-1)",
        f"{iv}**-2",
        f"{iv}**(-2)",
    )
    return any(p in rhs for p in patterns)


def test_declared_flag_set_is_the_historical_nine():
    problems = _problems()
    declared = {pid for pid, p in problems.items() if "singular_origin" in p.flags}
    assert declared == EXPECTED_SINGULAR_ORIGIN_IDS


def test_declared_flags_match_retired_string_match():
    problems = _problems()
    declared = {pid for pid, p in problems.items() if "singular_origin" in p.flags}
    matched = {pid for pid, p in problems.items() if _retired_rhs_string_match(p)}
    assert declared == matched


def test_flags_default_empty_and_parse():
    problems = _problems()
    assert problems["000"].flags == []
    assert problems["113"].flags == ["singular_origin"]


def test_harness_gate_uses_declared_flags():
    import run_benchmark as rb

    problems = _problems()
    gated = {pid for pid, p in problems.items() if rb._declares_singular_origin(p)}
    assert gated == EXPECTED_SINGULAR_ORIGIN_IDS
    assert not hasattr(rb, "_rhs_uses_inverse_indep_var")
