# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Cleanliness gating for the typed/full-direct challenger lanes.

Covers the multi-term regularized implicit certificate and the optional
clean-gate RMS overrides in ``_direct_result_needs_typed_lane``.
"""

from __future__ import annotations

import pytest
import torch

from nestynet_sr.run_de import (
    _clean_regularized_implicit_linear_result,
    _direct_result_needs_typed_lane,
    _factorized_de_preferred,
    _first_line_certified,
)
from nestynet_sr.sr_de.factorized_de import (
    FactorizedSearchDERescueConfig,
    FactorizedSearchDEResult,
)
from nestynet_sr.sr_search.factorized_search.engine.archive import ResidualBasinArchive
from nestynet_sr.sr_search.factorized_search.engine.search import (
    _archive_best_stall_mse,
    _degenerate_abort_should_stop,
    _plateau_stop_should_stop,
    _relative_best_improvement,
)


def _implicit_result(
    *,
    probe_score: float = 1.723e-3,
    fit_score: float | None = None,
    b_exprs=("u", "du"),
    b_coeffs=(9.027, 1.103),
    a_expr: str = "1",
    multiplier=None,
) -> FactorizedSearchDEResult:
    implicit = {
        "a_expr": a_expr,
        "b_exprs": list(b_exprs),
        "b_coeffs": list(b_coeffs),
        "b_coeff_source": "derivative_residual",
        "normalized_probe_score": float(probe_score),
        "normalized_fit_score": float(probe_score if fit_score is None else fit_score),
    }
    if multiplier is not None:
        implicit["multiplier"] = multiplier
    return FactorizedSearchDEResult(
        order=2,
        x_axis=0,
        rhs_ast=None,
        residual_ast=None,
        canonical_equation="",
        probe_mse=float(probe_score) ** 2,
        probe_rms=float(probe_score),
        expr_ast=None,
        mapping={},
        mapping_kind="poly",
        feature_names=["x0", "u", "du"],
        diagnostics={
            "candidate_source": "regularized_implicit_residual",
            "probe_rel_rms": float(probe_score),
            "implicit_residual": implicit,
        },
    )


def _direct_result(*, probe_rms: float, probe_rel_rms: float) -> FactorizedSearchDEResult:
    return FactorizedSearchDEResult(
        order=2,
        x_axis=0,
        rhs_ast=None,
        residual_ast=None,
        canonical_equation="",
        probe_mse=float(probe_rms) ** 2,
        probe_rms=float(probe_rms),
        expr_ast=None,
        mapping={},
        mapping_kind="poly",
        feature_names=["x0", "u", "du"],
        diagnostics={"probe_rel_rms": float(probe_rel_rms)},
    )


def test_multi_term_implicit_certified_clean():
    # de103-like: exact damped oscillator at normalized score 1.7e-3.
    cfg = FactorizedSearchDERescueConfig()
    result = _implicit_result()
    assert _clean_regularized_implicit_linear_result(result, cfg)
    assert not _direct_result_needs_typed_lane(result, cfg)


def test_implicit_above_clean_score_still_needs_typed():
    cfg = FactorizedSearchDERescueConfig()
    result = _implicit_result(probe_score=8.0e-3)
    assert not _clean_regularized_implicit_linear_result(result, cfg)
    assert _direct_result_needs_typed_lane(result, cfg)


def test_implicit_overfit_rejected():
    cfg = FactorizedSearchDERescueConfig()
    result = _implicit_result(probe_score=4.0e-3, fit_score=1.0e-5)
    assert not _clean_regularized_implicit_linear_result(result, cfg)


def test_implicit_degenerate_coeff_rejected():
    cfg = FactorizedSearchDERescueConfig()
    result = _implicit_result(b_coeffs=(9.027, 1.0e9))
    assert not _clean_regularized_implicit_linear_result(result, cfg)


def test_implicit_bad_multiplier_rejected():
    cfg = FactorizedSearchDERescueConfig()
    result = _implicit_result(a_expr="x0", multiplier={"ok": True, "sign_ok": False})
    assert not _clean_regularized_implicit_linear_result(result, cfg)


def test_non_implicit_result_unaffected():
    cfg = FactorizedSearchDERescueConfig()
    result = _direct_result(probe_rms=2.8e-1, probe_rel_rms=2.4e-1)
    assert not _clean_regularized_implicit_linear_result(result, cfg)
    assert _direct_result_needs_typed_lane(result, cfg)


def test_clean_gate_override_relaxes_rms_trigger():
    # de014-like near miss: probe RMS 1.03e-3 against the default 1e-3 trigger.
    result = _direct_result(probe_rms=1.03e-3, probe_rel_rms=6.8e-3)
    cfg_default = FactorizedSearchDERescueConfig()
    assert _direct_result_needs_typed_lane(result, cfg_default)
    cfg_relaxed = FactorizedSearchDERescueConfig(clean_gate_val_rms=5.0e-3)
    assert not _direct_result_needs_typed_lane(result, cfg_relaxed)


def test_clean_gate_invalid_override_falls_back():
    result = _direct_result(probe_rms=1.03e-3, probe_rel_rms=6.8e-3)
    cfg = FactorizedSearchDERescueConfig(clean_gate_val_rms=-1.0)
    assert _direct_result_needs_typed_lane(result, cfg)


def test_first_line_certified_for_implicit_and_not_direct():
    cfg = FactorizedSearchDERescueConfig()
    assert _first_line_certified(_implicit_result(), cfg)
    assert not _first_line_certified(_direct_result(probe_rms=2.8e-1, probe_rel_rms=2.4e-1), cfg)
    assert not _first_line_certified(None, cfg)


def test_certified_incumbent_not_displaced_by_2x_typed_candidate():
    # de118-like: exact Bessel J0 in the implicit lane at 2.0e-3; a typed
    # candidate at 3.5e-3 must not win on the cleaner-rank bonus.
    cfg = FactorizedSearchDERescueConfig()
    incumbent = _implicit_result(probe_score=2.0e-3)
    typed = _direct_result(probe_rms=3.5e-3, probe_rel_rms=3.5e-3)
    factor = float(cfg.replace_rel_factor) if _first_line_certified(incumbent, cfg) else 2.0
    assert factor == cfg.replace_rel_factor
    assert not _factorized_de_preferred(
        typed, "factorized", incumbent, "regularized_implicit_residual",
        same_lane_rel_factor=1.0, cleaner_lane_factor=factor, dirtier_lane_factor=0.5,
    )
    # A strictly better typed candidate still wins.
    typed_better = _direct_result(probe_rms=1.0e-3, probe_rel_rms=1.0e-3)
    assert _factorized_de_preferred(
        typed_better, "factorized", incumbent, "regularized_implicit_residual",
        same_lane_rel_factor=1.0, cleaner_lane_factor=factor, dirtier_lane_factor=0.5,
    )


def test_uncertified_incumbent_keeps_cleaner_lane_bonus():
    cfg = FactorizedSearchDERescueConfig()
    incumbent = _implicit_result(probe_score=8.0e-3)  # above clean score: uncertified
    assert not _first_line_certified(incumbent, cfg)
    typed = _direct_result(probe_rms=1.2e-2, probe_rel_rms=1.2e-2)
    assert _factorized_de_preferred(
        typed, "factorized", incumbent, "regularized_implicit_residual",
        same_lane_rel_factor=1.0, cleaner_lane_factor=2.0, dirtier_lane_factor=0.5,
    )


def _abort_kwargs(**overrides):
    kwargs = dict(
        n_evaluated=2000,
        accepted_total=2,
        start_best=3.336,
        current_best=3.336,
        enable=True,
        min_evals=1000,
        max_accepted=8,
        stall_delta=1e-4,
    )
    kwargs.update(overrides)
    return kwargs


def test_degenerate_abort_fires_on_flat_starved_launch():
    # de118-like degenerate typed launch: accepted~0, best flat since brute.
    assert _degenerate_abort_should_stop(**_abort_kwargs())


def test_degenerate_abort_respects_gates():
    assert not _degenerate_abort_should_stop(**_abort_kwargs(enable=False))
    assert not _degenerate_abort_should_stop(**_abort_kwargs(n_evaluated=500))
    assert not _degenerate_abort_should_stop(**_abort_kwargs(accepted_total=700))
    # Real improvement since the brute phase.
    assert not _degenerate_abort_should_stop(**_abort_kwargs(current_best=3.0))
    # Finding any finite score from an empty archive counts as progress.
    assert not _degenerate_abort_should_stop(
        **_abort_kwargs(start_best=float("inf"), current_best=3.336)
    )
    # Still-empty archive with nothing accepted is degenerate.
    assert _degenerate_abort_should_stop(
        **_abort_kwargs(start_best=float("inf"), current_best=float("inf"))
    )


def test_relative_best_improvement_handles_empty_archive_transition():
    assert _relative_best_improvement(float("inf"), 3.0) == float("inf")
    assert _relative_best_improvement(float("inf"), float("inf")) == pytest.approx(0.0)
    assert _relative_best_improvement(10.0, 9.0) == pytest.approx(0.1)
    assert _relative_best_improvement(10.0, float("inf")) == pytest.approx(0.0)


def test_archive_best_stall_mse_can_ignore_effective_complexity_improvements():
    arch = ResidualBasinArchive()
    z = torch.ones((4, 1), dtype=torch.float64)
    arch.update("best_raw", 0.20, ("var", 0), z, {}, raw_mse=0.10)
    arch.update("best_effective", 0.05, ("add", ("var", 0), ("const", 1.0)), z * 2.0, {}, raw_mse=0.50)

    assert _archive_best_stall_mse(arch, prefer_raw=False) == pytest.approx(0.05)
    assert _archive_best_stall_mse(arch, prefer_raw=True) == pytest.approx(0.10)


def test_plateau_stop_requires_enabled_archive_and_restart_cap():
    kwargs = dict(
        enable=True,
        n_evaluated=3500,
        min_evals=2000,
        consecutive_soft_restarts=2,
        max_soft_restarts=2,
        has_archive=True,
    )
    assert _plateau_stop_should_stop(**kwargs)
    assert not _plateau_stop_should_stop(**{**kwargs, "enable": False})
    assert not _plateau_stop_should_stop(**{**kwargs, "has_archive": False})
    assert not _plateau_stop_should_stop(**{**kwargs, "n_evaluated": 1500})
    assert not _plateau_stop_should_stop(**{**kwargs, "consecutive_soft_restarts": 1})
    assert not _plateau_stop_should_stop(**{**kwargs, "max_soft_restarts": 0})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
