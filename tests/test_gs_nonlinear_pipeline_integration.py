from __future__ import annotations

import numpy as np
import pytest
import torch

from nestynet_sr.sr_core.bridges import Mul, Pow, U, Var
from nestynet_sr.sr_de.de_search import (
    DESearchConfig,
    _certify_de_determining_candidate,
)
from nestynet_sr.sr_gs.config import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.de_bridge import (
    generalized_symmetry_de_term_rows,
    nonlinear_invariant_de_term_rows,
)
from nestynet_sr.sr_gs.de_certificates import certify_scalar_ode_candidate
from nestynet_sr.sr_gs.de_invariant_compiler import (
    InvariantCompilerConfig,
    compile_point_invariants,
)
from nestynet_sr.run_de import (
    _attach_automatic_gs_carriers,
    _bound_automatic_gs_fss,
)
from nestynet_sr.sr_de.factorized_de import (
    FactorizedSearchDERescueConfig,
    default_physics_rescue_hp,
)
from nestynet_sr.sr_gs.de_upgrades import PolynomialPointGenerator
from nestynet_sr.sr_gs.nonlinear_de_symmetry import (
    PolynomialDESymmetryConfig,
    PolynomialDESymmetryResult,
    project_generator_direction,
)
from nestynet_sr.sr_gs.prolongation import _eval_term_on_jets
from nestynet_sr.sr_search.factorized_search.gs_carrier_seed import (
    nonlinear_invariant_carrier_seeds,
)


def test_affine_defaults_stay_off_and_quadratic_multiplier_is_bounded() -> None:
    gs = GeneralizedSymmetryConfig()
    de = DESearchConfig()
    assert not gs.de_determining_equations
    assert not gs.de_nonlinear_invariants
    assert gs.de_auto_nonlinear
    assert gs.de_auto_fss
    assert gs.de_auto_fss_max_attempts == 1
    assert gs.de_determining_max_degree == 2
    assert gs.de_determining_multiplier_degree == 2
    assert not de.gs_de_determining_equations
    assert not de.gs_de_nonlinear_invariants
    assert de.gs_de_auto_nonlinear
    assert de.gs_de_auto_fss
    assert de.gs_de_auto_fss_max_attempts == 1
    assert de.gs_de_auto_fss_n_iter == 1500
    assert de.gs_de_determining_max_degree == 2
    assert de.gs_de_determining_multiplier_degree == 2


class _CertificateCache:
    def __init__(self, u: torch.Tensor, u1: torch.Tensor):
        self._u = u
        self._u1 = u1
        self.u = None
        self.g = None
        self.H = None

    def reset(self) -> None:
        self.u = None
        self.g = None
        self.H = None

    def ensure(self, x, need_grad=False, need_hess=False) -> None:
        n = int(x.shape[0])
        self.u = self._u[:n, None]
        self.g = self._u1[:n, None, None]
        self.H = torch.zeros((n, 1, 1, 1), dtype=x.dtype, device=x.device)


def test_gs_enable_automatically_escalates_and_hands_nontrivial_carriers_to_fss() -> None:
    rng = np.random.default_rng(218)
    n = 360
    x = torch.as_tensor(rng.uniform(0.5, 2.0, n), dtype=torch.float64)
    u = torch.as_tensor(rng.uniform(0.7, 2.2, n), dtype=torch.float64)
    u1 = torch.as_tensor(rng.uniform(-1.5, 1.5, n), dtype=torch.float64)
    cfg = DESearchConfig(
        gs_enable=True,
        gs_de_determining_bootstraps=2,
        gs_de_certificate_tol=1.0e-8,
        gs_de_nonlinear_invariant_tol=1.0e-8,
    )
    report = _certify_de_determining_candidate(
        order=2,
        X=x[:, None],
        cache=_CertificateCache(u, u1),
        term_asts=(Pow(U(), -3),),
        coeffs=torch.tensor([-1.7], dtype=torch.float64),
        cfg=cfg,
    )

    assert report is not None
    escalation = report["automatic_escalation"]
    assert escalation["affine_certified_nullity"] == 2
    assert escalation["quadratic_certified_nullity"] == 3
    assert escalation["selected_degree"] == 2
    assert escalation["reason"] == "quadratic_added_certified_directions"
    assert report["nonlinear_carriers"]["status"] == "recovered"

    primary = type("Primary", (), {"determining_certificate": report})()
    handoff = _attach_automatic_gs_carriers(primary, cfg)
    assert handoff["attached"]
    assert handoff["nontrivial_carrier_count"] >= 1
    assert handoff["trigger_fss"]
    assert cfg.gs_de_compiled_nonlinear_invariants is report.compiled_invariants

    cfg.gs_de_auto_fss = False
    disabled_handoff = _attach_automatic_gs_carriers(primary, cfg)
    assert not disabled_handoff["trigger_fss"]
    cfg.gs_de_auto_fss = True

    rescue_cfg = FactorizedSearchDERescueConfig(hp=default_physics_rescue_hp())
    budgets = _bound_automatic_gs_fss(rescue_cfg, cfg)
    assert budgets == {
        "n_iter": 1500,
        "n_fit": 1024,
        "n_probe": 1024,
        "max_depth": 4,
        "return_topk": 8,
    }
    assert rescue_cfg.budget_scope == "global"


def test_explicit_disable_preserves_affine_only_legacy_routing() -> None:
    x = torch.linspace(0.5, 2.0, 120, dtype=torch.float64)
    u = 1.0 + 0.2 * x
    cfg = DESearchConfig(
        gs_enable=True,
        gs_de_auto_nonlinear=False,
        gs_de_determining_certificate=False,
        gs_de_determining_equations=False,
    )

    report = _certify_de_determining_candidate(
        order=1,
        X=x[:, None],
        cache=_CertificateCache(u, torch.full_like(u, 0.2)),
        term_asts=(U(),),
        coeffs=torch.tensor([-1.0], dtype=torch.float64),
        cfg=cfg,
    )

    assert report is None


def test_automatic_quadratic_escalation_fails_closed_on_collapsed_support() -> None:
    x = torch.linspace(-1.0, 1.0, 160, dtype=torch.float64)
    u = torch.zeros_like(x)
    cfg = DESearchConfig(gs_enable=True)
    report = _certify_de_determining_candidate(
        order=1,
        X=x[:, None],
        cache=_CertificateCache(u, torch.zeros_like(u)),
        term_asts=(U(),),
        coeffs=torch.tensor([-1.0], dtype=torch.float64),
        cfg=cfg,
    )

    assert report is not None
    assert report["status"] == "failed"
    primary = type("Primary", (), {"determining_certificate": report})()
    handoff = _attach_automatic_gs_carriers(primary, cfg)
    assert not handoff["trigger_fss"]
    assert handoff["reason"] == "no_certified_carrier_compilation"


def test_selected_equation_certificate_routes_to_coupled_quadratic_solver() -> None:
    x = np.linspace(-1.0, 1.0, 80)
    u = 0.4 + x + 0.25 * x * x
    u1 = 1.0 + 0.5 * x
    result = certify_scalar_ode_candidate(
        x=x,
        u=u,
        u1=u1,
        coeffs=(),
        term_asts=(),
        order=2,
        generator_max_degree=2,
        multiplier_max_degree=2,
        bootstrap=2,
        max_generators=4,
        on_shell_tol=1.0e-7,
        off_shell_tol=1.0e-7,
    )
    assert isinstance(result, PolynomialDESymmetryResult)
    report = result.to_report()
    assert report["coefficient_convention"] == "pair_major_xi_eta_per_monomial"
    assert report["config"]["generator_degree"] == 2
    assert report["config"]["multiplier_degree"] == 2
    assert report["on_shell_projector"]
    assert "bracket_certificates" in report


def test_coupled_affine_regression_keeps_functional_multiplier_lane() -> None:
    x = np.linspace(-1.0, 1.0, 80)
    u = np.exp(x)
    result = certify_scalar_ode_candidate(
        x=x,
        u=u,
        coeffs=(-1.0,),
        term_asts=(U(),),
        order=1,
        generator_max_degree=1,
        multiplier_max_degree=2,
        use_coupled_polynomial_solver=True,
        max_generators=4,
    )
    assert isinstance(result, PolynomialDESymmetryResult)
    assert result.config.generator_degree == 1
    assert result.config.multiplier_degree == 2


def test_certified_nonlinear_carriers_cross_de_and_fss_bridges() -> None:
    generator = PolynomialPointGenerator(
        "common_scaling",
        "quadratic_test",
        xi_terms=((1.0, 1, 0),),
        eta_terms=((1.0, 0, 1),),
    )
    x = torch.linspace(0.5, 2.0, 100, dtype=torch.float64)
    u = 1.2 + 0.3 * torch.sin(x)
    points = torch.stack((x, u), dim=1)
    ratio = Mul(Var(1), Pow(Var(0), -1))
    compilation = compile_point_invariants(
        (generator,),
        points[:70],
        points[70:],
        candidate_asts=(ratio,),
        config=InvariantCompilerConfig(max_invariants=1),
    )
    assert compilation.status == "recovered"

    rows = nonlinear_invariant_de_term_rows(compilation)
    assert len(rows) == 1
    assert rows[0][1:] == ("gs_nonlinear_invariant", "nonlinear_point_invariant")
    zeros = torch.zeros_like(x)
    evaluated = _eval_term_on_jets(
        rows[0][0], x=x, u=u, u1=zeros, u2=zeros, x_axis=0
    )
    assert torch.allclose(evaluated, u / x)

    cfg = DESearchConfig(
        gs_enable=True,
        gs_de_compiled_nonlinear_invariants=compilation,
    )
    bridged = generalized_symmetry_de_term_rows(cfg, order=1)
    assert any(source == "gs_nonlinear_invariant" for _, source, _ in bridged)

    seeds, diagnostics = nonlinear_invariant_carrier_seeds(compilation)
    assert len(seeds) == 1
    assert diagnostics[0]["gs_source_family"] == "nonlinear_point_invariant"
    assert diagnostics[0]["certificate"]["accepted"]
    assert nonlinear_invariant_carrier_seeds(compilation, max_seeds=0) == ([], [])


def test_off_shell_certificate_rejects_generator_confined_to_one_orbit() -> None:
    # F = u_x - x - u has the particular solution u=-x-1.  The quadratic
    # field eta=(u+x+1)^2 vanishes to first order on that orbit, so trajectory-
    # paired probes falsely place it in the determining nullspace.  Independent
    # support-box probes must reject that decorative direction.
    x = np.linspace(-1.0, 1.0, 120)
    u = -x - 1.0
    result = certify_scalar_ode_candidate(
        x=x,
        u=u,
        coeffs=(-1.0, -1.0),
        term_asts=(Var(0), U()),
        order=1,
        generator_max_degree=2,
        multiplier_max_degree=2,
        bootstrap=2,
        max_generators=12,
        on_shell_tol=1.0e-7,
        off_shell_tol=1.0e-7,
    )
    assert isinstance(result, PolynomialDESymmetryResult)
    # Monomial order: 1, x, u, x^2, xu, u^2; pair-major xi/eta.
    decorative = np.zeros(12)
    decorative[[1, 3, 5, 7, 9, 11]] = (1.0, 2.0, 2.0, 1.0, 2.0, 1.0)
    _, membership_residual = project_generator_direction(result, decorative)
    assert membership_residual > 0.1


def test_nonlinear_certificate_refuses_collapsed_transverse_support() -> None:
    x = np.linspace(-1.0, 1.0, 120)
    u = np.zeros_like(x)
    with pytest.raises(ValueError, match="insufficient state-space support"):
        certify_scalar_ode_candidate(
            x=x,
            u=u,
            coeffs=(-1.0,),
            term_asts=(U(),),
            order=1,
            generator_max_degree=2,
            multiplier_max_degree=2,
        )


def test_cubic_generator_lane_is_explicitly_deferred() -> None:
    with pytest.raises(ValueError, match="cubic and higher are deferred"):
        PolynomialDESymmetryConfig(generator_degree=3).validated()
