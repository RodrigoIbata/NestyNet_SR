# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""GS -> charts bridge: discovered graph symmetries compile to executable,
sharpness-certified input charts.

Contracts: a shifted power law must certify its scaling symmetry and compile
to the shifted-log warp with the right fixed point; a pure exponential must
be recognized as translation + output cofactor (no input warp); an aperiodic
control must yield no certified chart."""

import numpy as np
import pytest

from nestynet.charts import FitConfig
from nestynet_sr.sr_gs.chart_bridge import scan_and_compile_charts

N = 1600
FIT = FitConfig(segments=24, epochs=300, restarts=2)
SHARP_FIT = FitConfig(segments=12, epochs=150, restarts=2)


def _scan(t, y):
    return scan_and_compile_charts(t, y, fit_cfg=FIT, sharp_fit_cfg=SHARP_FIT)


def test_shifted_power_law_compiles_to_warp():
    t = np.linspace(0.5, 60.0, N)
    y = 2.5 * (t + 3.0) ** 1.7
    res = _scan(t, y)
    certified = [p for p in res.proposals if p.certified]
    assert certified, "\n".join(res.log)
    p = certified[0]
    t0 = p.chart.get_param("t0")
    assert abs(t0 - (-3.0)) < 0.15, f"t0 {t0}"
    # the output action should suggest the power-law exponent beta/A ~ 1.7
    assert abs(p.beta / p.A - 1.7) < 0.1


def test_exponential_recognized_as_translation_with_cofactor():
    t = np.linspace(0.0, 60.0, N)
    y = 1.3 * np.exp(0.08 * t)
    res = _scan(t, y)
    trans = [p for p in res.proposals if p.kind == "translation"]
    assert trans, "\n".join(res.log)
    p = trans[0]
    assert p.chart is None
    assert abs(p.beta / p.b - 0.08) < 0.01
    assert "cofactor" in p.note


def test_aperiodic_control_certifies_nothing():
    t = np.linspace(0.0, 60.0, N)
    y = 0.7 * np.exp(-(((t - 30.0) / 12.0) ** 2)) + 0.01 * t
    res = _scan(t, y)
    assert not [p for p in res.proposals if p.certified], "\n".join(res.log)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
