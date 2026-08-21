import numpy as np
import pytest


def test_scalar_de_relative_invariance_uses_on_shell_recovery_and_off_shell_certificate():
    from nestynet_sr.sr_gs.de_determining import recover_de_generators
    from nestynet_sr.sr_gs.jet_bundle import JetSpaceSpec

    rng = np.random.default_rng(401)
    x = rng.uniform(-1.0, 1.0, size=128)
    u = rng.uniform(0.4, 2.0, size=128)
    u_x = rng.uniform(-1.5, 1.5, size=128)

    # F = u_x - u. On shell, u_x is eliminated as u.
    jet_space = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=1)
    result = recover_de_generators(
        jet_space=jet_space,
        residual="u_x - u",
        on_shell_samples={"x": x, "u": u, "u_x": u},
        off_shell_samples={"x": x, "u": u, "u_x": u_x},
    )

    assert result.on_shell_residual_rel < 1.0e-8
    assert result.off_shell_relative_residual_rel < 1.0e-8
    assert result.multiplier == pytest.approx(1.0, rel=1.0e-8)
    assert any(gen.name == "u_d_u" for gen in result.generators)
    report = result.to_report()
    assert report["evidence"]["on_shell_operator"] == "prV(F)=0"
    assert report["evidence"]["off_shell_certificate"] == "prV(F)-lambda*F=0"
    assert report["determining_nullity"] >= 1


def test_de_determining_rejects_non_scalar_ode_scope_before_recovery():
    from nestynet_sr.sr_gs.de_determining import recover_de_generators
    from nestynet_sr.sr_gs.jet_bundle import JetSpaceSpec

    jet_space = JetSpaceSpec(independent=("t", "x"), dependent=("u",), max_order=1)

    with pytest.raises(NotImplementedError, match="vector/PDE prolongation"):
        recover_de_generators(
            jet_space=jet_space,
            residual="u_t - u",
            on_shell_samples={"t": [0.0], "x": [0.0], "u": [1.0], "u_t": [1.0]},
            off_shell_samples={"t": [0.0], "x": [0.0], "u": [1.0], "u_t": [0.5]},
        )


def test_jet_space_spec_rejects_unsupported_vector_pde_cases_explicitly():
    from nestynet_sr.sr_gs.jet_bundle import JetSpaceSpec

    jet_space = JetSpaceSpec(independent=("t", "x"), dependent=("u", "v"), max_order=2)

    with pytest.raises(NotImplementedError, match="vector/PDE prolongation"):
        jet_space.require_scalar_ode_phase_one()
