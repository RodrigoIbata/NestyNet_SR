# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import numpy as np
import pytest
from scipy.integrate import solve_ivp

MU = 2.959e-4


def _kepler_orbit(a_au, ecc, incl, node, n=1500):
    def rhs(t, s):
        q = s[:3]
        v = s[3:]
        r = np.linalg.norm(q)
        return [*v, *(-MU * q / r**3)]

    rp = a_au * (1 - ecc)
    vp = np.sqrt(MU * (1 + ecc) / (a_au * (1 - ecc)))
    s0 = np.array([rp, 0.0, 0.0, 0.0, vp, 0.0])
    ci, si = np.cos(incl), np.sin(incl)
    cn, sn = np.cos(node), np.sin(node)
    Rz = np.array([[cn, -sn, 0], [sn, cn, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    R = Rz @ Rx
    s0 = np.concatenate([R @ s0[:3], R @ s0[3:]])
    period = 2 * np.pi * np.sqrt(a_au**3 / MU)
    t = np.linspace(0.0, 0.7 * period, n)
    sol = solve_ivp(rhs, (0.0, t[-1]), s0, t_eval=t, rtol=1e-11, atol=1e-13)
    return sol.y.T, t


def _kepler_ensemble():
    specs = [(2.3, 0.10, 0.2, 0.5), (2.7, 0.15, 0.3, 1.1),
             (3.1, 0.05, 0.1, 2.3), (2.5, 0.20, 0.4, 0.8)]
    trajs, times = [], []
    for a, e, inc, nod in specs:
        Z, t = _kepler_orbit(a, e, inc, nod)
        trajs.append(Z)
        times.append(t)
    return trajs, times


def test_symplectic_matrix_structure():
    from nestynet_sr.sr_gs.noether_reduction import symplectic_matrix

    J = symplectic_matrix(3)
    assert J.shape == (6, 6)
    assert np.allclose(J.T, -J)  # antisymmetric
    assert np.allclose(J @ J, -np.eye(6))  # J^2 = -I


def test_momentum_maps_are_the_physical_charges():
    from nestynet_sr.sr_gs.noether_reduction import canonical_generators, momentum_map

    rng = np.random.default_rng(1)
    z = rng.normal(size=(5, 6))
    x, y, zc, px, py, pz = z.T
    gens = {g.name: g for g in canonical_generators(3)}

    def mm(name):
        return momentum_map(gens[name], z, n=3)

    np.testing.assert_allclose(mm("rotation_xy"), x * py - y * px, rtol=1e-10)   # L_z
    np.testing.assert_allclose(mm("rotation_yz"), y * pz - zc * py, rtol=1e-10)  # L_x
    np.testing.assert_allclose(mm("rotation_xz"), -(zc * px - x * pz), rtol=1e-10)  # -L_y
    np.testing.assert_allclose(mm("translation_x"), px, rtol=1e-10)
    np.testing.assert_allclose(mm("dilation"), x * px + y * py + zc * pz, rtol=1e-10)


def test_discovers_so3_and_rejects_non_symmetries():
    from nestynet_sr.sr_gs.noether_reduction import discover_noether_symmetries

    trajs, _times = _kepler_ensemble()
    disc = discover_noether_symmetries(trajs, n=3, conservation_tol=1e-6)
    admitted = {r.name for r in disc["admitted"]}
    assert {"rotation_xy", "rotation_xz", "rotation_yz"} <= admitted  # SO(3)
    rejected = {r.name for r in disc["rejected"]}
    assert {"translation_x", "translation_y", "translation_z", "dilation"} <= rejected
    # rotations conserved to ~machine precision on ideal Kepler
    for r in disc["admitted"]:
        assert r.conservation_rel_drift < 1e-6


def test_reduction_derives_k_equals_ell_squared():
    from nestynet_sr.sr_gs.noether_reduction import noether_kepler_reduction

    trajs, times = _kepler_ensemble()
    res = noether_kepler_reduction(trajs, times=times, conservation_tol=1e-6)
    assert res["so3_fully_admitted"] is True
    red = res["reduction"]
    # centrifugal coefficient equals ell^2 to high precision on ideal Kepler
    assert abs(red["k_over_ell_squared_median"] - 1.0) < 1e-4
    assert red["k_over_ell_squared_max_abs_dev"] < 1e-3
    assert red["mu"] == pytest.approx(MU, rel=1e-3)


def test_non_central_field_breaks_rotational_symmetry():
    """A field with an anisotropic term does not conserve all of L: SO(3) breaks."""
    from nestynet_sr.sr_gs.noether_reduction import discover_noether_symmetries

    def rhs(t, s):
        q = s[:3]
        v = s[3:]
        r = np.linalg.norm(q)
        a = -MU * q / r**3
        a[2] -= 3.0e-3 * q[2]  # anisotropic z-restoring: breaks rotations about x, y
        return [*v, *a]

    trajs = []
    for a_au, e, inc, nod in [(2.4, 0.1, 0.5, 0.3), (2.8, 0.15, 0.6, 1.4)]:
        rp = a_au * (1 - e)
        vp = np.sqrt(MU * (1 + e) / (a_au * (1 - e)))
        s0 = np.array([rp, 0, 0, 0, vp * np.cos(inc), vp * np.sin(inc)])
        period = 2 * np.pi * np.sqrt(a_au**3 / MU)
        t = np.linspace(0, 0.7 * period, 1500)
        sol = solve_ivp(rhs, (0, t[-1]), s0, t_eval=t, rtol=1e-11, atol=1e-13)
        trajs.append(sol.y.T)
    disc = discover_noether_symmetries(trajs, n=3, conservation_tol=1e-6)
    admitted = {r.name for r in disc["admitted"]}
    # rotation about z (L_z) still conserved; rotations about x, y broken
    assert "rotation_xy" in admitted
    assert "rotation_xz" not in admitted and "rotation_yz" not in admitted


def test_reduction_accepts_provided_analytic_rddot():
    from nestynet_sr.sr_gs.noether_reduction import noether_kepler_reduction

    trajs, times = _kepler_ensemble()
    rddots = []
    for Z in trajs:
        q, v = Z[:, :3], Z[:, 3:]
        r = np.linalg.norm(q, axis=1)
        L = np.cross(q, v)
        ell_squared = np.sum(L * L, axis=1)
        rddots.append(ell_squared / r**3 - MU / r**2)

    res = noether_kepler_reduction(
        trajs, times=times, conservation_tol=1e-6, rddot_series=rddots
    )
    red = res["reduction"]
    assert red["rddot_source"] == "provided_analytic"
    # exact analytic rddot -> the fit sharpens well past the FD-based tolerances
    assert red["mu"] == pytest.approx(MU, rel=1e-6)
    assert abs(red["k_over_ell_squared_median"] - 1.0) < 1e-6

    with pytest.raises(ValueError):
        noether_kepler_reduction(
            trajs, times=times, conservation_tol=1e-6, rddot_series=rddots[:-1]
        )
