# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Pin the trig-frequency walkers behind the two-block frequency polish.

Regression guards for the de301 (driven harmonic oscillator) fix: the
polish rescales sin/cos(c*x) constants inside candidate ASTs, and the
adversarial review found two walker defects worth pinning forever —
aliased ConstNodes must be scaled once per object (not once per
reference), and Pow-wrapped trig subtrees must be traversed so the
same-frequency ambiguity guard sees every constant.
"""

from __future__ import annotations

from nestynet_sr.sr_core.bridges import Add, ConstNode, Cos, Mul, Pow, Sin, Var
from nestynet_sr.sr_de._factorized_de_rescue import (
    _trig_frequency_consts,
    _trig_frequency_rescaled_copy,
)


def test_finds_const_on_either_side_of_mul():
    assert [c.value for c in _trig_frequency_consts(Cos(Mul(ConstNode(2.5), Var(0))), x_axis=0)] == [2.5]
    assert [c.value for c in _trig_frequency_consts(Cos(Mul(Var(0), ConstNode(2.5))), x_axis=0)] == [2.5]


def test_other_axis_and_bare_var_are_ignored():
    assert _trig_frequency_consts(Cos(Mul(ConstNode(3.0), Var(1))), x_axis=0) == []
    assert _trig_frequency_consts(Sin(Var(0)), x_axis=0) == []


def test_rescale_deep_copies_and_preserves_original():
    ast = Mul(ConstNode(-1.1), Cos(Mul(ConstNode(1.714), Var(0))))
    dup = _trig_frequency_rescaled_copy(ast, x_axis=0, scale=2.0)
    assert abs(_trig_frequency_consts(dup, x_axis=0)[0].value - 3.428) < 1e-12
    assert abs(_trig_frequency_consts(ast, x_axis=0)[0].value - 1.714) < 1e-12


def test_aliased_constant_scaled_once_per_object():
    shared = Mul(ConstNode(3.0), Var(0))
    dup = _trig_frequency_rescaled_copy(Add(Sin(shared), Cos(shared)), x_axis=0, scale=2.0)
    assert sorted({round(c.value, 9) for c in _trig_frequency_consts(dup, x_axis=0)}) == [6.0]


def test_pow_wrapped_trig_is_traversed_and_scaled_coherently():
    ast = Add(
        Mul(ConstNode(2.0), Sin(Mul(ConstNode(3.0), Var(0)))),
        Mul(ConstNode(0.5), Pow(Sin(Mul(ConstNode(3.0), Var(0))), 2.0)),
    )
    consts = _trig_frequency_consts(ast, x_axis=0)
    assert len(consts) == 2
    dup = _trig_frequency_rescaled_copy(ast, x_axis=0, scale=1.1)
    assert all(abs(c.value - 3.3) < 1e-9 for c in _trig_frequency_consts(dup, x_axis=0))


def test_sin_plus_cos_same_frequency_scale_together():
    ast = Add(
        Mul(ConstNode(-0.08), Sin(Mul(ConstNode(1.7435), Var(0)))),
        Mul(ConstNode(1.245), Cos(Mul(ConstNode(1.7435), Var(0)))),
    )
    dup = _trig_frequency_rescaled_copy(ast, x_axis=0, scale=0.5)
    vals = {round(c.value, 9) for c in _trig_frequency_consts(dup, x_axis=0)}
    assert vals == {round(1.7435 * 0.5, 9)}
