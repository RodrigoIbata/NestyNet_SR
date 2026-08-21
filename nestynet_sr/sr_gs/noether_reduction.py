# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Noether reduction: discover phase-space symmetries and reduce by their charges.

This is the symplectic (Hamiltonian) analogue of the scalar-ODE cascade in
:mod:`de_reduction`.  Instead of point symmetries of a graph ``u(x)`` it works
with one-parameter symmetries of phase space ``z = (q, p)`` and their Noether
charges (momentum maps).

The pattern is the same "read the symmetry off the data" idea:

1. Scan a small basis of candidate linear/affine phase-space generators
   (rotations, translations, dilation).  Each generator ``V`` (an infinitesimal
   action ``delta z = M z + c``) has a momentum map -- the conserved charge of
   Noether's theorem -- given by the symplectic pairing ``G = (1/2) z^T (-J M) z
   + (-J c) . z``.
2. Keep the generators whose charge is conserved along the trajectory data.
   For a central force the three rotations survive (angular momentum ``L``);
   translations (linear momentum) and dilation do not.
3. Reduce by the discovered rotational symmetry: the conserved ``L = r x p``
   fixes the orbit plane and its magnitude ``ell = |L|`` is the areal constant
   ``r^2 theta_dot``.  Eliminating the cyclic angle gives the effective radial
   dynamics ``r_ddot = ell^2 / r^3 - mu / r^2`` -- so the centrifugal
   coefficient equals ``ell^2`` **by construction**, i.e. the empirical
   coefficient relation ``k = ell^2`` is *derived* from the symmetry rather
   than fitted.

Nothing about "angle" is presupposed: the rotation is discovered from Cartesian
phase-space data as the symmetry whose Noether charge the data conserve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch


def symplectic_matrix(n: int) -> np.ndarray:
    """Canonical symplectic form J = [[0, I], [-I, 0]] on R^{2n} (q, p)."""

    J = np.zeros((2 * n, 2 * n), dtype=np.float64)
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    return J


@dataclass(frozen=True)
class PhaseGenerator:
    """A one-parameter phase-space symmetry, action delta z = M z + c."""

    name: str
    family: str
    M: np.ndarray  # (2n, 2n) linear part
    c: np.ndarray  # (2n,) constant part
    description: str = ""


def _rotation_generator(n: int, i: int, j: int) -> np.ndarray:
    """Simultaneous rotation of q_i,q_j and p_i,p_j (an so(n) element in sp(2n))."""

    M = np.zeros((2 * n, 2 * n), dtype=np.float64)
    for base in (0, n):  # positions block, then momenta block
        a, b = base + i, base + j
        M[a, b] = -1.0
        M[b, a] = 1.0
    return M


def canonical_generators(n: int) -> list[PhaseGenerator]:
    """A physically-interpretable candidate basis for n-dof phase space.

    Rotations (so(n)), translations (one per position axis), and the isotropic
    dilation ``q -> q, p -> -p`` on the generator level.  This is the small,
    interpretable slate scanned against the data -- not all of sp(2n).
    """

    gens: list[PhaseGenerator] = []
    axis = "xyz"
    for i in range(n):
        for j in range(i + 1, n):
            lbl = f"{axis[i] if i < 3 else i}{axis[j] if j < 3 else j}"
            gens.append(PhaseGenerator(
                name=f"rotation_{lbl}", family="rotation",
                M=_rotation_generator(n, i, j), c=np.zeros(2 * n),
                description=f"rotation in the q{i}q{j} plane (and conjugate momenta)"))
    for i in range(n):
        c = np.zeros(2 * n)
        c[i] = 1.0  # translation in q_i
        gens.append(PhaseGenerator(
            name=f"translation_{axis[i] if i < 3 else i}", family="translation",
            M=np.zeros((2 * n, 2 * n)), c=c, description=f"translation in q{i}"))
    Md = np.zeros((2 * n, 2 * n), dtype=np.float64)
    for i in range(n):
        Md[i, i] = 1.0        # q -> q
        Md[n + i, n + i] = -1.0  # p -> -p
    gens.append(PhaseGenerator(
        name="dilation", family="dilation", M=Md, c=np.zeros(2 * n),
        description="isotropic dilation q->q, p->-p (virial generator)"))
    return gens


def momentum_map(gen: PhaseGenerator, Z: np.ndarray, *, n: int) -> np.ndarray:
    """Noether charge G(z) of a generator, from the symplectic pairing.

    For delta z = M z + c the conserved function satisfies J grad G = M z + c, so
    ``G(z) = (1/2) z^T (-J M) z + (-J c) . z``.
    """

    J = symplectic_matrix(n)
    S = -J @ gen.M
    S = 0.5 * (S + S.T)  # only the symmetric part enters the quadratic form
    lin = -J @ gen.c
    Zc = np.asarray(Z, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        quad = np.einsum("ni,ij,nj->n", Zc, S, Zc)
        return 0.5 * quad + Zc @ lin  # G = (1/2) z^T (-J M) z + (-J c) . z


def momentum_map_gradient(
    gen: PhaseGenerator, Z: np.ndarray, *, n: int
) -> np.ndarray:
    """Analytic gradient of the canonical affine momentum-map fast path."""

    J = symplectic_matrix(n)
    S = -J @ gen.M
    S = 0.5 * (S + S.T)
    linear = -J @ gen.c
    points = np.asarray(Z, dtype=np.float64)
    return np.einsum("ni,ji->nj", points, S) + linear


@dataclass
class DiscoveredSymmetry:
    name: str
    family: str
    charge_mean_per_trajectory: list[float]
    conservation_rel_drift: float
    conserved: bool
    description: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    # Excursion of the charge along each trajectory divided by a scale common to
    # every quadratic charge, max_t |q||p| (median over trajectories).  Unlike
    # ``conservation_rel_drift`` it is not inflated for charges whose orbital
    # mean vanishes (e.g. the virial q.p), so it is the number to compare
    # across generators.
    conservation_scale_drift: float = float("nan")


def discover_noether_symmetries(
    trajectories: Sequence[np.ndarray],
    *,
    n: int,
    generators: Sequence[PhaseGenerator] | None = None,
    conservation_tol: float = 1.0e-3,
) -> dict[str, Any]:
    """Discover the admitted symmetries: charges conserved along the data.

    Each ``trajectories`` entry is a time-ordered ``(N_i, 2n)`` array of
    phase-space samples ``(q, p)``.  For each candidate generator the Noether
    charge is evaluated along every trajectory and its relative drift measured;
    a generator is *admitted* when the charge is conserved (drift below
    ``conservation_tol``) across the ensemble.
    """

    gens = list(generators) if generators is not None else canonical_generators(n)
    trajs = [np.asarray(z, dtype=np.float64) for z in trajectories]
    results: list[DiscoveredSymmetry] = []
    for gen in gens:
        per_traj_mean: list[float] = []
        per_traj_drift: list[float] = []
        per_traj_scale_drift: list[float] = []
        for z in trajs:
            g = momentum_map(gen, z, n=n)
            finite = np.isfinite(g)
            g = g[finite]
            if g.size < 4:
                continue
            mean = float(np.mean(g))
            # scale by the charge magnitude, with a floor from the phase-space
            # scale so an identically-zero charge is not spuriously "conserved"
            scale = max(abs(mean), 1.0e-12)
            excursion = float(np.max(np.abs(g - mean)))
            drift = float(excursion / scale)
            # common phase-space scale for all quadratic charges: max_t |q||p|
            qp = np.linalg.norm(z[finite, :n], axis=1) * np.linalg.norm(z[finite, n:], axis=1)
            qp_scale = float(np.max(qp)) if qp.size else 0.0
            per_traj_mean.append(mean)
            per_traj_drift.append(drift)
            per_traj_scale_drift.append(excursion / qp_scale if qp_scale > 0.0 else float("inf"))
        if not per_traj_drift:
            continue
        rel_drift = float(np.median(per_traj_drift))
        scale_drift = float(np.median(per_traj_scale_drift))
        # a charge that is ~0 on every trajectory is not an informative symmetry
        informative = float(np.max(np.abs(per_traj_mean))) > 1.0e-9
        conserved = bool(rel_drift <= float(conservation_tol) and informative)
        results.append(DiscoveredSymmetry(
            name=gen.name, family=gen.family,
            charge_mean_per_trajectory=per_traj_mean,
            conservation_rel_drift=rel_drift, conserved=conserved,
            conservation_scale_drift=scale_drift,
            description=gen.description,
            provenance={"informative": informative},
        ))
    admitted = [r for r in results if r.conserved]
    return {
        "status": "ok",
        "n_dof": int(n),
        "admitted": admitted,
        "rejected": [r for r in results if not r.conserved],
        "all": results,
    }


def angular_momentum(Z: np.ndarray) -> np.ndarray:
    """L = q x p per sample, for 3-dof phase space Z=(x,y,z,px,py,pz)."""

    Zc = np.asarray(Z, dtype=np.float64)
    q = Zc[:, :3]
    p = Zc[:, 3:6]
    return np.cross(q, p)


def central_force_reduction(
    trajectories: Sequence[np.ndarray],
    *,
    accelerations: Sequence[np.ndarray] | None = None,
    times: Sequence[np.ndarray] | None = None,
    casimir_values: Sequence[np.ndarray] | None = None,
    casimir_expression: str | None = None,
    rddot_series: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Reduce a discovered central force by its rotational Noether charge.

    Uses the conserved ``L = r x p`` (the rotational momentum map) to derive the
    effective radial dynamics.  The centrifugal coefficient is ``ell^2`` with
    ``ell = |L|`` *by construction*; we additionally fit the radial acceleration
    ``r_ddot`` against ``{1/r^3, 1/r^2}`` to confirm ``k = ell^2`` and recover
    the shared ``mu``.  Returns per-orbit ``ell``, fitted ``k``, ``k/ell^2``,
    and the assembled reduced Hamiltonian.
    """

    del accelerations
    trajs = [np.asarray(z, dtype=np.float64) for z in trajectories]
    if casimir_values is not None and len(casimir_values) != len(trajs):
        raise ValueError("casimir_values must contain one array per trajectory")
    if rddot_series is not None and len(rddot_series) != len(trajs):
        raise ValueError("rddot_series must contain one array per trajectory")
    rows: list[dict[str, Any]] = []
    inv_r3_all: list[np.ndarray] = []
    inv_r2_all: list[np.ndarray] = []
    rddot_all: list[np.ndarray] = []
    block_index: list[int] = []

    for i, z in enumerate(trajs):
        q = z[:, :3]
        v = z[:, 3:6]
        r = np.linalg.norm(q, axis=1)
        L = np.cross(q, v)
        if casimir_values is None:
            ell_squared = float(np.median(np.sum(L * L, axis=1)))
            ell_squared_source = "explicit_angular_momentum_fallback"
        else:
            values = np.asarray(casimir_values[i], dtype=np.float64).reshape(-1)
            if values.shape[0] != z.shape[0] or not np.all(np.isfinite(values)):
                raise ValueError("each Casimir array must be finite and match its trajectory")
            ell_squared = float(np.median(values))
            if ell_squared <= 0.0:
                raise ValueError("recovered quadratic algebra Casimir must be positive")
            ell_squared_source = "recovered_algebra_casimir"
        ell = float(np.sqrt(max(ell_squared, 0.0)))
        # radial coordinate second derivative from the data
        rdot = np.sum(q * v, axis=1) / np.maximum(r, 1e-30)
        if times is not None:
            t = np.asarray(times[i], dtype=np.float64)
        else:
            t = np.arange(r.size, dtype=np.float64)
        if rddot_series is not None:
            rddot = np.asarray(rddot_series[i], dtype=np.float64).reshape(-1)
            if rddot.shape[0] != r.size:
                raise ValueError("each rddot_series array must match its trajectory length")
        else:
            rddot = np.gradient(rdot, t, edge_order=2)
        # trim edges (finite-difference bias; kept for provided rddot so the
        # fit windows stay identical between the two sources)
        sl = slice(3, r.size - 3)
        rr, rd2 = r[sl], rddot[sl]
        good = np.isfinite(rr) & np.isfinite(rd2) & (rr > 0)
        rows.append({
            "trajectory": i,
            "ell": ell,
            "ell_squared": ell_squared,
            "ell_squared_source": ell_squared_source,
            "L_dir_mean": [float(np.mean(L[:, k] / np.maximum(np.linalg.norm(L, axis=1), 1e-30))) for k in range(3)],
        })
        inv_r3_all.append(1.0 / rr[good] ** 3)
        inv_r2_all.append(1.0 / rr[good] ** 2)
        rddot_all.append(rd2[good])
        block_index.append(int(np.sum(good)))

    # shared-mu, per-orbit-k fit: r_ddot = k_i / r^3 - mu / r^2
    n_orbits = len(trajs)
    cols = []
    for i in range(n_orbits):
        c = np.zeros((int(block_index[i]), n_orbits + 1), dtype=np.float64)
        c[:, i] = inv_r3_all[i]
        c[:, n_orbits] = inv_r2_all[i]
        cols.append(c)
    Phi = np.concatenate(cols, axis=0)
    y = np.concatenate(rddot_all, axis=0)
    coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    mu = float(-coeffs[n_orbits])
    for i, row in enumerate(rows):
        k = float(coeffs[i])
        row["k_fit"] = k
        row["k_over_ell_squared"] = float(k / row["ell_squared"]) if row["ell_squared"] > 0 else float("nan")

    k_over = np.asarray([row["k_over_ell_squared"] for row in rows], dtype=np.float64)
    return {
        "status": "ok",
        "rddot_source": "provided_analytic" if rddot_series is not None else "finite_difference",
        "mu": mu,
        "n_orbits": n_orbits,
        "per_orbit": rows,
        "k_over_ell_squared_median": float(np.median(k_over)),
        "k_over_ell_squared_max_abs_dev": float(np.max(np.abs(k_over - 1.0))),
        "reduced_hamiltonian": "H(r, p_r; ell) = (1/2) p_r^2 + ell^2 / (2 r^2) - mu / r",
        "centrifugal_coefficient_is_ell_squared": True,
        "centrifugal_coefficient_is_algebra_casimir": casimir_values is not None,
        "ell_squared_source": (
            "recovered_algebra_casimir"
            if casimir_values is not None
            else "explicit_angular_momentum_fallback"
        ),
        "casimir_expression": casimir_expression,
        "derivation": (
            "the recovered rotation-charge algebra has a positive quadratic "
            "Casimir K; its pullback fixes ell^2=K on each orbit. Eliminating "
            "the cyclic angle gives r_ddot=ell^2/r^3-mu/r^2, so k=K by "
            "construction rather than an independent fit."
            if casimir_values is not None
            else
            "rotational Noether charge L = r x p is conserved => ell = |L| is the "
            "areal constant r^2 theta_dot; eliminating the cyclic angle gives "
            "r_ddot = ell^2/r^3 - mu/r^2, so k = ell^2 by construction (not fitted)."
        ),
    }


def noether_kepler_reduction(
    trajectories: Sequence[np.ndarray],
    *,
    times: Sequence[np.ndarray] | None = None,
    conservation_tol: float = 1.0e-3,
    rddot_series: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    """End-to-end: discover the rotational symmetry, then reduce by its charge.

    ``trajectories`` are time-ordered ``(N_i, 6)`` phase-space arrays
    ``(x, y, z, vx, vy, vz)``.  Returns the discovered symmetries and, when the
    rotations are admitted, the central-force reduction that derives ``k=ell^2``.
    """

    disc = discover_noether_symmetries(trajectories, n=3, conservation_tol=conservation_tol)
    rotations = [r for r in disc["admitted"] if r.family == "rotation"]
    generators_by_name = {generator.name: generator for generator in canonical_generators(3)}
    rotation_generators = tuple(
        generators_by_name[row.name]
        for row in rotations
        if row.name in generators_by_name
    )
    out: dict[str, Any] = {
        "status": "ok",
        "discovered": {
            "admitted": [(r.name, r.family, r.conservation_rel_drift) for r in disc["admitted"]],
            "rejected": [(r.name, r.family, r.conservation_rel_drift) for r in disc["rejected"]],
            "scale_drift": {r.name: r.conservation_scale_drift for r in disc["all"]},
        },
        "rotational_symmetry_admitted": len(rotations) >= 1,
        "so3_fully_admitted": len(rotations) >= 3,
    }
    if len(rotation_generators) >= 3:
        from nestynet_sr.sr_core.bridges import ast_to_human_readable
        from nestynet_sr.sr_de.poisson_invariants import (
            evaluate_scalar_library,
            scalar_combination_ast,
        )
        from nestynet_sr.sr_gs.algebra_casimirs import (
            certify_charge_brackets,
            discover_algebra_casimirs,
            extract_phase_structure_constants,
            normalized_positive_quadratic_casimir,
        )

        structure = extract_phase_structure_constants(rotation_generators)
        algebra = discover_algebra_casimirs(
            structure,
            max_degree=2,
            sample_count=384,
            random_seed=17,
        )
        charge_blocks = [
            np.stack(
                [momentum_map(generator, z, n=3) for generator in rotation_generators],
                axis=1,
            )
            for z in trajectories
        ]
        gradient_blocks = [
            np.stack(
                [
                    momentum_map_gradient(generator, z, n=3)
                    for generator in rotation_generators
                ],
                axis=1,
            )
            for z in trajectories
        ]
        all_charges = np.concatenate(charge_blocks, axis=0)
        all_gradients = np.concatenate(gradient_blocks, axis=0)
        canonical_poisson = np.broadcast_to(
            symplectic_matrix(3),
            (all_charges.shape[0], 6, 6),
        )
        charge_brackets = certify_charge_brackets(
            all_charges,
            all_gradients,
            canonical_poisson,
            structure,
            allow_central_cocycle=True,
            residual_tol=1.0e-9,
        )

        normalized_coefficients = None
        normalized_expression = None
        casimir_blocks = None
        normalization_failure = None
        if algebra.accepted and algebra.casimirs.candidates:
            candidate = algebra.casimirs.candidates[0]
            try:
                normalized_coefficients = normalized_positive_quadratic_casimir(
                    candidate,
                    algebra.casimirs.terms,
                    dimension=len(rotation_generators),
                )
                coefficient_tensor = torch.as_tensor(
                    normalized_coefficients, dtype=torch.float64
                )
                normalized_ast = scalar_combination_ast(
                    algebra.casimirs.terms, coefficient_tensor
                )
                normalized_expression = (
                    None
                    if normalized_ast is None
                    else ast_to_human_readable(normalized_ast)
                )
                casimir_blocks = []
                for charges in charge_blocks:
                    points = torch.as_tensor(charges, dtype=torch.float64)
                    library = evaluate_scalar_library(
                        points, algebra.casimirs.terms
                    )
                    values = library.values @ coefficient_tensor
                    casimir_blocks.append(values.detach().cpu().numpy())
            except ValueError as exc:
                normalization_failure = str(exc)

        algebra_casimir_accepted = bool(
            algebra.accepted
            and charge_brackets.accepted
            and casimir_blocks is not None
        )
        out["algebra_casimir"] = {
            "accepted": algebra_casimir_accepted,
            "complete": algebra.complete,
            "expected_corank": algebra.expected_corank,
            "expressions": [
                candidate.expression for candidate in algebra.casimirs.candidates
            ],
            "normalized_expression": normalized_expression,
            "normalization_failure": normalization_failure,
            "charge_brackets_accepted": charge_brackets.accepted,
            "structure": structure.to_report(),
            "charge_brackets": charge_brackets.to_report(),
            "global_momentum_map_proven": False,
        }
        if algebra_casimir_accepted:
            assert casimir_blocks is not None
            out["reduction"] = central_force_reduction(
                trajectories,
                times=times,
                casimir_values=casimir_blocks,
                casimir_expression=normalized_expression,
                rddot_series=rddot_series,
            )
        else:
            out["reduction_status"] = "algebra_casimir_not_certified"
    elif rotations:
        out["reduction_status"] = "full_rotation_algebra_not_admitted"
    return out


__all__ = [
    "DiscoveredSymmetry",
    "PhaseGenerator",
    "angular_momentum",
    "canonical_generators",
    "central_force_reduction",
    "discover_noether_symmetries",
    "momentum_map",
    "momentum_map_gradient",
    "noether_kepler_reduction",
    "symplectic_matrix",
]
