# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""High-level symmetry certificates for scalar-ODE candidates.

This module closes the gap between the determining-equation solver in
:mod:`de_determining` (which needs a residual plus jet samples) and the two
places candidates actually live: STLSQ dictionaries expressed as named terms
with coefficients, and DE-search results expressed as term ASTs with
coefficients.  It also provides the ensemble *flow test*: an equation symmetry
maps solutions to other solutions, so a generator accepted for a candidate can
be checked against data alone by flowing one trajectory of a multi-trajectory
ensemble onto another with ``exp(eps*V)``.

The certificate is an intrinsic property of the candidate equation: on-shell
samples eliminate the anchor derivative using the candidate's own right-hand
side, so certificate quality does not depend on how well the candidate fits
the data.  The data enter only through the flow test.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .de_determining import _eval_residual_expr, recover_de_generators
from .jet_bundle import JetSpaceSpec

NodeLike = Any

_EPS = 1.0e-12


# ---------------------------------------------------------------------------
# Residual construction
# ---------------------------------------------------------------------------

def translate_term_name(name: str) -> str:
    """Translate dictionary term names (``x0`` convention) to jet symbols."""

    return str(name).replace("x0", "x")


def residual_string_from_named_terms(
    term_names: Sequence[str],
    coeffs: Sequence[float],
    *,
    anchor: str = "u_x",
    coeff_prune_tol: float = 0.0,
) -> str:
    """Build an anchored residual string ``anchor + sum_k c_k * term_k``.

    ``coeff_prune_tol`` drops terms whose |coefficient| falls below the
    threshold; small spurious offsets otherwise break exact symmetries and the
    certificate then reports the (physically real) violation they introduce.
    """

    if len(term_names) != len(coeffs):
        raise ValueError("term_names and coeffs must have equal length")
    pieces = [str(anchor)]
    for name, coeff in zip(term_names, coeffs):
        c = float(coeff)
        if abs(c) <= float(coeff_prune_tol):
            continue
        pieces.append(f"({c!r})*({translate_term_name(name)})")
    return " + ".join(pieces)


def residual_callable_from_ast_terms(
    term_asts: Sequence[NodeLike],
    coeffs: Sequence[float],
    *,
    order: int = 1,
    x_axis: int = 0,
    description: str = "",
) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
    """Wrap DE-search term ASTs into a residual callable on jet environments."""

    from .prolongation import _eval_term_on_jets

    terms = list(term_asts)
    weights = [float(c) for c in coeffs]
    if len(terms) != len(weights):
        raise ValueError("term_asts and coeffs must have equal length")
    order_i = int(order)
    if order_i not in (1, 2):
        raise ValueError(f"scalar-ODE certificate supports order 1 or 2; got {order_i}")

    def _residual(env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = env["x"]
        u = env["u"]
        u1 = env["u_x"]
        u2 = env.get("u_xx")
        if u2 is None:
            u2 = torch.zeros_like(u1)
        anchor = u1 if order_i == 1 else u2
        total = anchor
        for weight, term in zip(weights, terms):
            total = total + weight * _eval_term_on_jets(
                term, x=x, u=u, u1=u1, u2=u2, x_axis=int(x_axis)
            )
        return total

    _residual.description = description or f"anchored_ast_residual(order={order_i}, terms={len(terms)})"
    return _residual


# ---------------------------------------------------------------------------
# Jet-sample builders
# ---------------------------------------------------------------------------

def _as_1d(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr


def _has_resolved_span(values: np.ndarray) -> bool:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array).all():
        return False
    scale = max(1.0, float(np.max(np.abs(array))))
    return bool(float(np.ptp(array)) > 1.0e-10 * scale)


def _candidate_rhs_values(
    residual: Any,
    x: np.ndarray,
    u: np.ndarray,
    u1: np.ndarray | None,
    *,
    order: int,
) -> np.ndarray:
    """Eliminate a highest-jet anchor after verifying the monic-affine contract."""

    xt = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)
    ut = torch.as_tensor(u, dtype=torch.float64).reshape(-1, 1)
    if int(order) == 2:
        if u1 is None:
            raise ValueError("order-2 candidates need measured u_x samples")
        u1t = torch.as_tensor(u1, dtype=torch.float64).reshape(-1, 1)
    else:
        u1t = torch.zeros_like(xt)

    def evaluate(anchor: torch.Tensor) -> torch.Tensor:
        env = {
            "x": xt,
            "u": ut,
            "u_x": anchor if int(order) == 1 else u1t,
            "u_xx": anchor if int(order) == 2 else torch.zeros_like(anchor),
        }
        value = (
            residual(env)
            if callable(residual) and not isinstance(residual, str)
            else _eval_residual_expr(str(residual), env)
        )
        return torch.as_tensor(value, dtype=torch.float64).reshape_as(anchor)

    # Sampling away from the elimination point catches nonlinear anchors such
    # as u_x + u_x**3 whose first derivative happens to be one at zero.
    for probe in (-0.731, 0.0, 0.619):
        anchor = torch.full_like(xt, float(probe), requires_grad=True)
        value = evaluate(anchor)
        derivative = torch.autograd.grad(
            value.sum(), anchor, allow_unused=True
        )[0]
        if derivative is None or not torch.allclose(
            derivative,
            torch.ones_like(anchor),
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise ValueError(
                "candidate residual must be affine and monic in its highest jet"
            )

    F0 = evaluate(torch.zeros_like(xt))
    return -np.asarray(F0.detach().cpu()).reshape(-1)


def build_certificate_samples(
    *,
    residual: Any,
    x: Any,
    u: Any,
    u1: Any | None = None,
    order: int = 1,
    off_shell_spread: float = 0.75,
    max_samples: int = 1024,
    seed: int = 0,
    require_state_space_support: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """On-shell (candidate-eliminated anchor) and off-shell jet samples."""

    x_arr = _as_1d(x)
    u_arr = _as_1d(u)
    if x_arr.size != u_arr.size:
        raise ValueError("x and u must have equal length")
    u1_arr = _as_1d(u1) if u1 is not None else None
    mask = np.isfinite(x_arr) & np.isfinite(u_arr)
    if u1_arr is not None:
        mask &= np.isfinite(u1_arr)
    x_arr, u_arr = x_arr[mask], u_arr[mask]
    if u1_arr is not None:
        u1_arr = u1_arr[mask]
    if x_arr.size < 8:
        raise ValueError("need at least 8 finite (x, u) samples for a certificate")
    if x_arr.size > int(max_samples):
        idx = np.linspace(0, x_arr.size - 1, int(max_samples)).round().astype(int)
        x_arr, u_arr = x_arr[idx], u_arr[idx]
        if u1_arr is not None:
            u1_arr = u1_arr[idx]

    if bool(require_state_space_support):
        required = [("x", x_arr), ("u", u_arr)]
        if int(order) == 2:
            if u1_arr is None:
                raise ValueError("order-2 candidates need measured u_x samples")
            required.append(("u_x", u1_arr))
        collapsed = [name for name, values in required if not _has_resolved_span(values)]
        if collapsed:
            joined = ", ".join(collapsed)
            raise ValueError(
                "insufficient state-space support for nonlinear symmetry certification; "
                f"negligible span in {joined}. Supply multiple trajectories or explicit "
                "off-trajectory probes before promoting a nonlinear generator."
            )

    order_i = int(order)
    anchor_vals = _candidate_rhs_values(residual, x_arr, u_arr, u1_arr, order=order_i)
    finite = np.isfinite(anchor_vals)
    x_arr, u_arr, anchor_vals = x_arr[finite], u_arr[finite], anchor_vals[finite]
    if u1_arr is not None:
        u1_arr = u1_arr[finite]

    rng = np.random.default_rng(int(seed))

    # Off-shell certification must leave the observed trajectory manifold.
    # Reusing paired (x,u) samples lets a flexible generator vanish on one
    # orbit and masquerade as an equation symmetry, no matter how thoroughly
    # that orbit is bootstrapped.  Probe the independently sampled support box
    # instead, then perturb the candidate's own anchor value at those points.
    # This remains inside each observed coordinate range and is deterministic.
    n_probe = int(anchor_vals.size)
    off_x = rng.uniform(float(np.min(x_arr)), float(np.max(x_arr)), size=n_probe)
    off_u = rng.uniform(float(np.min(u_arr)), float(np.max(u_arr)), size=n_probe)
    off_u1_base = None
    if order_i == 2:
        if u1_arr is None:
            raise ValueError("order-2 candidates need measured u_x samples")
        off_u1_base = rng.uniform(float(np.min(u1_arr)), float(np.max(u1_arr)), size=n_probe)
    off_rhs = _candidate_rhs_values(
        residual,
        off_x,
        off_u,
        off_u1_base,
        order=order_i,
    )
    finite_off = np.isfinite(off_x) & np.isfinite(off_u) & np.isfinite(off_rhs)
    if off_u1_base is not None:
        finite_off &= np.isfinite(off_u1_base)
    off_x, off_u, off_rhs = off_x[finite_off], off_u[finite_off], off_rhs[finite_off]
    if off_u1_base is not None:
        off_u1_base = off_u1_base[finite_off]
    if off_rhs.size < 8:
        raise ValueError(
            "fewer than 8 finite independent support-box probes remain for off-shell certification"
        )

    combined_anchor = np.concatenate((anchor_vals, off_rhs))
    lo, hi = float(np.min(combined_anchor)), float(np.max(combined_anchor))
    span = max(hi - lo, 1.0e-6, 0.1 * max(abs(lo), abs(hi)))
    pad = float(off_shell_spread) * span
    if pad > 0.0:
        magnitude = rng.uniform(0.25 * pad, pad, size=off_rhs.size)
        sign = rng.choice(np.asarray([-1.0, 1.0]), size=off_rhs.size)
        off_anchor = off_rhs + sign * magnitude
    else:
        off_anchor = off_rhs.copy()

    if order_i == 1:
        on = {"x": x_arr, "u": u_arr, "u_x": anchor_vals}
        off = {"x": off_x, "u": off_u, "u_x": off_anchor}
    else:
        if u1_arr is None:
            raise ValueError("order-2 candidates need measured u_x samples")
        on = {"x": x_arr, "u": u_arr, "u_x": u1_arr, "u_xx": anchor_vals}
        off = {"x": off_x, "u": off_u, "u_x": off_u1_base, "u_xx": off_anchor}
    return on, off


def certify_scalar_ode_candidate(
    *,
    x: Any,
    u: Any,
    coeffs: Sequence[float],
    term_names: Sequence[str] | None = None,
    term_asts: Sequence[NodeLike] | None = None,
    order: int = 1,
    x_axis: int = 0,
    u1: Any | None = None,
    coeff_prune_tol: float = 0.0,
    on_shell_tol: float = 1.0e-6,
    off_shell_tol: float = 1.0e-6,
    off_shell_spread: float = 0.75,
    max_samples: int = 1024,
    seed: int = 0,
    generator_max_degree: int = 1,
    multiplier_max_degree: int = 2,
    bootstrap: int = 0,
    max_generators: int = 4,
    sparse_rotation: bool = True,
    bracket_certificate: bool = True,
    use_coupled_polynomial_solver: bool = False,
) -> Any:
    """Certify the point symmetries of an anchored scalar-ODE candidate.

    The candidate is given either as dictionary term names + coefficients or
    as DE-search term ASTs + coefficients; the anchor (``u_x`` or ``u_xx``) is
    implicit with unit coefficient, matching both callers' conventions.
    """

    if (term_names is None) == (term_asts is None):
        raise ValueError("provide exactly one of term_names or term_asts")
    if term_names is not None:
        residual: Any = residual_string_from_named_terms(
            term_names, coeffs, anchor="u_x" if int(order) == 1 else "u_xx",
            coeff_prune_tol=float(coeff_prune_tol),
        )
    else:
        kept_terms: list[NodeLike] = []
        kept_coeffs: list[float] = []
        for term, coeff in zip(term_asts or (), coeffs):
            if abs(float(coeff)) <= float(coeff_prune_tol):
                continue
            kept_terms.append(term)
            kept_coeffs.append(float(coeff))
        residual = residual_callable_from_ast_terms(
            kept_terms, kept_coeffs, order=int(order), x_axis=int(x_axis)
        )
    on, off = build_certificate_samples(
        residual=residual,
        x=x,
        u=u,
        u1=u1,
        order=int(order),
        off_shell_spread=float(off_shell_spread),
        max_samples=int(max_samples),
        seed=int(seed),
        require_state_space_support=(
            bool(use_coupled_polynomial_solver) or int(generator_max_degree) > 1
        ),
    )
    jet_space = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=int(order))
    if bool(use_coupled_polynomial_solver) or int(generator_max_degree) > 1:
        from .nonlinear_de_symmetry import (
            PolynomialDESymmetryConfig,
            recover_polynomial_de_symmetries,
        )

        # A zero threshold/zero iterations leaves the certified subspace basis
        # untouched while retaining coordinate-projection representatives.
        sparse_threshold = 0.04 if bool(sparse_rotation) else 0.0
        sparse_iterations = 24 if bool(sparse_rotation) else 0
        bracket_tol = 1.0e-7
        nonlinear_cfg = PolynomialDESymmetryConfig(
            generator_degree=int(generator_max_degree),
            multiplier_degree=max(0, int(multiplier_max_degree)),
            bootstrap=max(0, int(bootstrap)),
            random_seed=int(seed),
            on_shell_tol=float(on_shell_tol),
            off_shell_tol=float(off_shell_tol),
            sparse_threshold=float(sparse_threshold),
            sparse_iterations=int(sparse_iterations),
            max_candidates=max(1, int(max_generators)),
            evaluate_bracket_closure=bool(bracket_certificate),
            bracket_closure_tol=float(bracket_tol),
        )
        return recover_polynomial_de_symmetries(
            jet_space=jet_space,
            residual=residual,
            on_shell_samples=on,
            off_shell_samples=off,
            config=nonlinear_cfg,
        )

    return recover_de_generators(
        jet_space=jet_space,
        residual=residual,
        on_shell_samples=on,
        off_shell_samples=off,
        on_shell_tol=float(on_shell_tol),
        off_shell_tol=float(off_shell_tol),
    )


# ---------------------------------------------------------------------------
# Ensemble flow test: exp(eps*V) maps solutions to solutions
# ---------------------------------------------------------------------------

def affine_flow(points_xu: np.ndarray, gen_coeffs: Sequence[float], eps: float) -> np.ndarray:
    """Flow (x, u) points by ``exp(eps*V)`` for the affine generator V.

    V = (a0 + a1 x + a2 u) d/dx + (b0 + b1 x + b2 u) d/du acts linearly on the
    homogeneous vector (x, u, 1), so the flow is a 3x3 matrix exponential.
    """

    from scipy.linalg import expm

    a0, a1, a2, b0, b1, b2 = (float(v) for v in gen_coeffs)
    gen_matrix = np.asarray(
        [[a1, a2, a0], [b1, b2, b0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    flow = expm(float(eps) * gen_matrix)
    pts = np.asarray(points_xu, dtype=np.float64)
    hom = np.column_stack([pts[:, 0], pts[:, 1], np.ones(pts.shape[0])])
    out = hom @ flow.T
    return out[:, :2]


def _pair_flow_mismatch(
    xa: np.ndarray,
    ua: np.ndarray,
    xb: np.ndarray,
    ub: np.ndarray,
    gen_coeffs: Sequence[float],
    eps: float,
    *,
    min_overlap_fraction: float,
) -> float:
    flowed = affine_flow(np.column_stack([xa, ua]), gen_coeffs, eps)
    xf, uf = flowed[:, 0], flowed[:, 1]
    lo, hi = float(np.min(xb)), float(np.max(xb))
    inside = (xf >= lo) & (xf <= hi) & np.isfinite(xf) & np.isfinite(uf)
    if float(np.mean(inside)) < float(min_overlap_fraction):
        return math.inf
    xf, uf = xf[inside], uf[inside]
    order = np.argsort(xb)
    ub_interp = np.interp(xf, xb[order], ub[order])
    diff = uf - ub_interp
    scale = max(float(np.sqrt(np.mean(np.square(ub_interp)))), 1.0e-12)
    return float(np.sqrt(np.mean(np.square(diff))) / scale)


def _generator_eps_scale(
    trajs: Sequence[tuple[np.ndarray, np.ndarray]], gen_coeffs: Sequence[float]
) -> float:
    """Flow-parameter range so displacements span the data spread, per active axis."""

    pooled_x = np.concatenate([t[0] for t in trajs])
    pooled_u = np.concatenate([t[1] for t in trajs])
    a0, a1, a2, b0, b1, b2 = (float(v) for v in gen_coeffs)
    xi_rms = float(np.sqrt(np.mean(np.square(a0 + a1 * pooled_x + a2 * pooled_u))))
    eta_rms = float(np.sqrt(np.mean(np.square(b0 + b1 * pooled_x + b2 * pooled_u))))
    ratios = []
    if xi_rms > 1.0e-12:
        ratios.append(max(float(np.ptp(pooled_x)), 1.0e-6) / xi_rms)
    if eta_rms > 1.0e-12:
        ratios.append(max(float(np.ptp(pooled_u)), 1.0e-6) / eta_rms)
    if not ratios:
        return 1.0
    return 2.0 * float(min(ratios))


def generator_ensemble_support(
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
    gen_coeffs: Sequence[float],
    *,
    n_eps: int = 121,
    min_overlap_fraction: float = 0.3,
    support_rel_tol: float = 5.0e-3,
    baseline_improvement_factor: float = 3.0,
) -> dict[str, Any]:
    """Test a generator against a solution ensemble by flowing trajectories.

    For every ordered trajectory pair (A, B) the flow parameter ``eps`` is
    scanned; if some ``exp(eps*V)`` maps A onto B (relative RMS below
    tolerance), the pair supports V.  To exclude vacuous matches (B already
    close to A), a supporting pair must also *improve* on its ``eps=0``
    baseline by ``baseline_improvement_factor``; pairs whose baseline is
    itself tiny are uninformative and are skipped.  The generator is
    *data-supported* when the median best mismatch across informative pairs
    is below tolerance.
    """

    trajs = [(_as_1d(x), _as_1d(u)) for x, u in trajectories]
    if len(trajs) < 2:
        return {"status": "skipped", "reason": "need at least two trajectories"}

    eps_max = _generator_eps_scale(trajs, gen_coeffs)
    eps_grid = np.linspace(-eps_max, eps_max, int(n_eps))

    pair_rows: list[dict[str, Any]] = []
    for ia in range(len(trajs)):
        for ib in range(len(trajs)):
            if ia == ib:
                continue
            xa, ua = trajs[ia]
            xb, ub = trajs[ib]
            baseline = _pair_flow_mismatch(
                xa, ua, xb, ub, gen_coeffs, 0.0,
                min_overlap_fraction=float(min_overlap_fraction),
            )
            if not math.isfinite(baseline) or baseline <= 5.0 * float(support_rel_tol):
                # B is (nearly) A, or no overlap: uninformative for this pair
                continue
            vals = np.asarray(
                [
                    _pair_flow_mismatch(
                        xa, ua, xb, ub, gen_coeffs, float(e),
                        min_overlap_fraction=float(min_overlap_fraction),
                    )
                    for e in eps_grid
                ]
            )
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            k = int(np.nanargmin(np.where(finite, vals, np.inf)))
            # local refinement around the best grid epsilon
            from scipy.optimize import minimize_scalar

            lo = eps_grid[max(k - 1, 0)]
            hi = eps_grid[min(k + 1, eps_grid.size - 1)]
            try:
                opt = minimize_scalar(
                    lambda e: _pair_flow_mismatch(
                        xa, ua, xb, ub, gen_coeffs, float(e),
                        min_overlap_fraction=float(min_overlap_fraction),
                    ),
                    bounds=(float(min(lo, hi)), float(max(lo, hi))),
                    method="bounded",
                    options={"xatol": 1.0e-10},
                )
                best_val = float(min(opt.fun, vals[k]))
                best_eps = float(opt.x if opt.fun <= vals[k] else eps_grid[k])
            except Exception:
                best_val, best_eps = float(vals[k]), float(eps_grid[k])
            improved = bool(baseline >= float(baseline_improvement_factor) * max(best_val, 1.0e-12))
            pair_rows.append(
                {
                    "pair": (ia, ib),
                    "best_eps": best_eps,
                    "best_rel_rms": best_val,
                    "baseline_rel_rms": float(baseline),
                    "improved_over_baseline": improved,
                }
            )

    if not pair_rows:
        return {"status": "skipped", "reason": "no informative trajectory pair"}
    best_vals = np.asarray([row["best_rel_rms"] for row in pair_rows])
    improved = np.asarray([bool(row["improved_over_baseline"]) for row in pair_rows])
    effective = np.where(improved, best_vals, np.inf)
    median_val = float(np.median(effective))
    return {
        "status": "tested",
        "n_pairs": len(pair_rows),
        "median_best_rel_rms": median_val,
        "fraction_pairs_supporting": float(
            np.mean((best_vals <= float(support_rel_tol)) & improved)
        ),
        "supported": bool(median_val <= float(support_rel_tol)),
        "support_rel_tol": float(support_rel_tol),
        "pairs": pair_rows,
    }


__all__ = [
    "affine_flow",
    "build_certificate_samples",
    "certify_scalar_ode_candidate",
    "generator_ensemble_support",
    "residual_callable_from_ast_terms",
    "residual_string_from_named_terms",
    "translate_term_name",
]
