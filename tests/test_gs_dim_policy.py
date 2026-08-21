# SPDX-License-Identifier: MPL-2.0

from nestynet_sr.sr_gs.dim_policy import combine_dimensional_decision


def test_audit_policy_keeps_baseline_decision():
    d = combine_dimensional_decision(
        candidate="x0*x1^-1",
        baseline_accept=False,
        gs_accept=True,
        policy="audit",
    )

    assert d.final_accept is False
    assert d.baseline_accept is False
    assert d.gs_accept is True


def test_replace_rref_policy_uses_gs_decision():
    d = combine_dimensional_decision(
        candidate="x0*x1^-1",
        baseline_accept=False,
        gs_accept=True,
        policy="replace-rref",
    )

    assert d.final_accept is True


def test_both_require_both_rejects_disagreement():
    d = combine_dimensional_decision(
        candidate="x0*x1^-1",
        baseline_accept=True,
        gs_accept=False,
        policy="both",
        both_rule="require-both",
    )

    assert d.final_accept is False
