# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from nestynet_sr.sr_gs.algebra_casimirs import (
    certify_charge_brackets,
    discover_algebra_casimirs,
    extract_affine_structure_constants,
    extract_phase_structure_constants,
)
from nestynet_sr.sr_gs.noether_reduction import (
    PhaseGenerator,
    canonical_generators,
    momentum_map,
    symplectic_matrix,
)


MU = 2.959e-4


def _kepler_ensemble():
    trajectories = []
    times = []
    for a, eccentricity, inclination, node in (
        (2.3, 0.10, 0.2, 0.4),
        (2.7, 0.15, 0.3, 1.1),
        (3.1, 0.05, 0.1, 2.2),
    ):
        radius = a * (1.0 - eccentricity)
        speed = np.sqrt(MU * (1.0 + eccentricity) / radius)
        state0 = np.array(
            [radius, 0.0, 0.0, 0.0, speed * np.cos(inclination), speed * np.sin(inclination)]
        )
        cosine, sine = np.cos(node), np.sin(node)
        rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        state0 = np.concatenate((rotation @ state0[:3], rotation @ state0[3:]))
        period = 2.0 * np.pi * np.sqrt(a**3 / MU)
        time = np.linspace(0.0, 0.6 * period, 900)

        def rhs(_time, state):
            q = state[:3]
            v = state[3:]
            return np.concatenate((v, -MU * q / np.linalg.norm(q) ** 3))

        solution = solve_ivp(
            rhs,
            (0.0, time[-1]),
            state0,
            t_eval=time,
            rtol=1.0e-11,
            atol=1.0e-13,
        )
        trajectories.append(solution.y.T)
        times.append(time)
    return trajectories, times


def _rotation_charge_gradients(generators, state):
    j = symplectic_matrix(3)
    return np.stack(
        [np.einsum("ni,ji->nj", state, -j @ gen.M) + (-j @ gen.c) for gen in generators],
        axis=1,
    )


def test_so3_structure_constants_yield_quadratic_algebra_casimir():
    rotations = tuple(
        gen for gen in canonical_generators(3) if gen.family == "rotation"
    )
    structure = extract_phase_structure_constants(rotations)
    assert structure.accepted
    assert structure.jacobi_residual < 1.0e-12

    result = discover_algebra_casimirs(
        structure,
        max_degree=2,
        sample_count=320,
        random_seed=501,
    )
    assert result.complete
    assert result.expected_corank == 1
    assert len(result.casimirs.candidates) == 1
    candidate = result.casimirs.candidates[0]
    assert candidate.complexity == 3
    assert candidate.ast is not None


def test_certified_affine_gs_algebra_uses_the_same_casimir_pipeline():
    generators = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[first, second] = -1.0
        matrix[second, first] = 1.0
        generators.append(
            np.concatenate((matrix.reshape(-1), np.zeros(3), np.zeros(2)))
        )
    algebra = SimpleNamespace(
        nullspace_basis=np.stack(generators, axis=1),
        input_dim=3,
        promotable=True,
    )
    structure = extract_affine_structure_constants(algebra)
    casimirs = discover_algebra_casimirs(
        structure, max_degree=2, sample_count=300, random_seed=503
    )

    assert structure.accepted
    assert structure.jacobi_residual < 1.0e-12
    assert casimirs.accepted
    assert casimirs.expected_corank == 1


def test_rotation_charge_brackets_certify_with_explicit_convention():
    rotations = tuple(
        gen for gen in canonical_generators(3) if gen.family == "rotation"
    )
    structure = extract_phase_structure_constants(rotations)
    state = np.random.default_rng(502).normal(size=(200, 6))
    charges = np.stack(
        [momentum_map(gen, state, n=3) for gen in rotations], axis=1
    )
    gradients = _rotation_charge_gradients(rotations, state)
    poisson = np.broadcast_to(symplectic_matrix(3), (state.shape[0], 6, 6))

    certificate = certify_charge_brackets(
        charges,
        gradients,
        poisson,
        structure,
        allow_central_cocycle=True,
        residual_tol=1.0e-11,
    )

    assert certificate.accepted
    assert certificate.relative_residual < 1.0e-12
    assert certificate.convention == "{J_a,J_b}=-c_ab^c J_c+kappa_ab"
    assert np.max(np.abs(certificate.central_cocycle)) < 1.0e-12


def test_nonzero_central_cocycle_is_explicit_and_cannot_be_silently_dropped():
    generators = (
        PhaseGenerator(
            name="q_translation",
            family="translation",
            M=np.zeros((2, 2)),
            c=np.array([1.0, 0.0]),
        ),
        PhaseGenerator(
            name="p_translation",
            family="translation",
            M=np.zeros((2, 2)),
            c=np.array([0.0, 1.0]),
        ),
    )
    structure = extract_phase_structure_constants(generators)
    state = np.random.default_rng(504).normal(size=(160, 2))
    charges = np.stack(
        [momentum_map(generator, state, n=1) for generator in generators], axis=1
    )
    gradients = np.stack(
        [
            np.broadcast_to(-symplectic_matrix(1) @ generator.c, state.shape)
            for generator in generators
        ],
        axis=1,
    )
    poisson = np.broadcast_to(
        symplectic_matrix(1), (state.shape[0], 2, 2)
    )
    admitted = certify_charge_brackets(
        charges,
        gradients,
        poisson,
        structure,
        allow_central_cocycle=True,
    )
    rejected = certify_charge_brackets(
        charges,
        gradients,
        poisson,
        structure,
        allow_central_cocycle=False,
    )

    assert admitted.accepted
    assert abs(admitted.central_cocycle[0, 1]) == 1.0
    assert not rejected.accepted


def test_kepler_reduction_uses_recovered_algebra_casimir():
    from nestynet_sr.sr_gs.noether_reduction import noether_kepler_reduction

    trajectories, times = _kepler_ensemble()
    report = noether_kepler_reduction(
        trajectories, times=times, conservation_tol=1.0e-6
    )

    assert report["so3_fully_admitted"]
    assert report["algebra_casimir"]["accepted"]
    assert report["algebra_casimir"]["complete"]
    assert report["algebra_casimir"]["charge_brackets_accepted"]
    assert report["reduction"]["ell_squared_source"] == "recovered_algebra_casimir"
    assert report["reduction"]["centrifugal_coefficient_is_algebra_casimir"]
