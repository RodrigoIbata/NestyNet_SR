# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Second-order output-link separability witnesses.

For a scalar output transform ``g = psi(f)``, cross-block additivity of ``g``
implies

    f_ij + r(f) f_i f_j = 0,  where r = psi'' / psi'.

This module fits the undivided implicit residual directly.  It never forms the
raw ratio ``-f_ij / (f_i f_j)``, so stationary directions do not create
division blow-ups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence

import numpy as np

_EPS = 1.0e-12


@dataclass(frozen=True)
class OutputLinkWitness:
    """Certificate for an output transform that makes blocks additive."""

    accepted: bool
    link_family: str
    basis_names: tuple[str, ...] = ()
    theta: tuple[float, ...] = ()
    blocks: tuple[tuple[int, ...], ...] = ()
    cross_pairs: tuple[tuple[int, int], ...] = ()
    max_cross_pair_residual: float = math.inf
    rel_cross_pair_residual: float = math.inf
    pair_consistency_residual: float = math.inf
    resampling_stability: float = math.inf
    fit_rows: int = 0
    ignored_stationary_rows: int = 0
    uses_implicit_residual: bool = True
    computed_raw_ratio: bool = False
    psi_prime_nonzero: bool = False
    gauge: str = "psi(1)=0, psi_prime(1)=1"
    power_exponent: float | None = None
    r_expression: str = "0"
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "link_family": self.link_family,
            "basis_names": list(self.basis_names),
            "theta": [float(v) for v in self.theta],
            "blocks": [[int(v) for v in block] for block in self.blocks],
            "cross_pairs": [[int(i), int(j)] for i, j in self.cross_pairs],
            "max_cross_pair_residual": float(self.max_cross_pair_residual),
            "rel_cross_pair_residual": float(self.rel_cross_pair_residual),
            "pair_consistency_residual": float(self.pair_consistency_residual),
            "resampling_stability": float(self.resampling_stability),
            "fit_rows": int(self.fit_rows),
            "ignored_stationary_rows": int(self.ignored_stationary_rows),
            "uses_implicit_residual": bool(self.uses_implicit_residual),
            "computed_raw_ratio": bool(self.computed_raw_ratio),
            "psi_prime_nonzero": bool(self.psi_prime_nonzero),
            "gauge": self.gauge,
            "power_exponent": None if self.power_exponent is None else float(self.power_exponent),
            "r_expression": self.r_expression,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class _BasisCandidate:
    name: str
    basis_names: tuple[str, ...]
    evaluator: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class _FitResult:
    candidate: _BasisCandidate
    theta: np.ndarray
    max_abs: float
    max_rel: float
    pair_consistency: float
    stability: float
    fit_rows: int
    ignored_stationary_rows: int
    finite: bool


def discover_output_link_separability(
    x: Any,
    y: Any,
    grad: Any,
    hess: Any,
    *,
    blocks: Sequence[Sequence[int]] | None = None,
    residual_atol: float = 1.0e-8,
    residual_rtol: float = 1.0e-8,
    stability_tol: float = 1.0e-6,
    min_fit_rows: int | None = None,
    stationary_rtol: float = 1.0e-10,
) -> OutputLinkWitness:
    """Fit a low-complexity output-link witness from first and second jets."""

    del x  # Coordinates are not needed until level-set consistency is promoted.
    y_arr = _to_vector(y, "y")
    grad_arr = _to_matrix(grad, "grad")
    hess_arr = _to_hessian(hess, "hess")
    if grad_arr.shape[0] != y_arr.shape[0] or hess_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("y, grad, and hess must have the same sample count")
    if hess_arr.shape[1] != grad_arr.shape[1] or hess_arr.shape[2] != grad_arr.shape[1]:
        raise ValueError("hess shape must be (N, n, n) and match grad")

    block_tuple = _normalize_blocks(blocks, grad_arr.shape[1])
    pairs = _cross_pairs(block_tuple)
    if not pairs:
        return OutputLinkWitness(
            accepted=False,
            link_family="none",
            blocks=block_tuple,
            reason="no_cross_block_pairs",
            evidence={"input_dim": int(grad_arr.shape[1])},
        )

    candidates = _basis_candidates(y_arr)
    min_rows = int(min_fit_rows) if min_fit_rows is not None else max(8, 3 * max(1, len(pairs)))
    fits = [
        _fit_candidate(
            candidate,
            y_arr,
            grad_arr,
            hess_arr,
            pairs,
            min_fit_rows=min_rows,
            stationary_rtol=float(stationary_rtol),
        )
        for candidate in candidates
    ]
    fits = [fit for fit in fits if fit.finite]
    if not fits:
        return OutputLinkWitness(
            accepted=False,
            link_family="none",
            blocks=block_tuple,
            cross_pairs=tuple(pairs),
            reason="no_finite_basis_fit",
        )

    fits.sort(key=lambda fit: (fit.max_abs, fit.max_rel, len(fit.candidate.basis_names), abs(float(np.linalg.norm(fit.theta)))))
    best = fits[0]
    family, exponent, expr = _classify_link(best.candidate, best.theta)
    psi_prime_ok = _psi_prime_nonzero(family, exponent, y_arr)
    accepted = bool(
        psi_prime_ok
        and best.fit_rows >= min_rows
        and (best.max_abs <= float(residual_atol) or best.max_rel <= float(residual_rtol))
        and (not math.isfinite(best.stability) or best.stability <= float(stability_tol))
    )
    reason = "accepted" if accepted else "residual_or_stability_threshold_failed"
    return OutputLinkWitness(
        accepted=accepted,
        link_family=family,
        basis_names=best.candidate.basis_names,
        theta=tuple(float(v) for v in best.theta),
        blocks=block_tuple,
        cross_pairs=tuple(pairs),
        max_cross_pair_residual=float(best.max_abs),
        rel_cross_pair_residual=float(best.max_rel),
        pair_consistency_residual=float(best.pair_consistency),
        resampling_stability=float(best.stability),
        fit_rows=int(best.fit_rows),
        ignored_stationary_rows=int(best.ignored_stationary_rows),
        psi_prime_nonzero=bool(psi_prime_ok),
        power_exponent=exponent,
        r_expression=expr,
        reason=reason,
        evidence={
            "candidate_basis": best.candidate.name,
            "residual_atol": float(residual_atol),
            "residual_rtol": float(residual_rtol),
            "stability_tol": float(stability_tol),
            "stationary_rtol": float(stationary_rtol),
            "available_families": [fit.candidate.name for fit in fits],
        },
    )


def _fit_candidate(
    candidate: _BasisCandidate,
    y: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    *,
    min_fit_rows: int,
    stationary_rtol: float,
) -> _FitResult:
    phi = candidate.evaluator(y)
    if phi.ndim == 1:
        phi = phi.reshape(-1, 1)
    if phi.shape[0] != y.shape[0]:
        raise ValueError(f"basis {candidate.name!r} returned {phi.shape[0]} rows, expected {y.shape[0]}")
    finite_phi = np.isfinite(phi).all(axis=1)

    rows = []
    rhs = []
    residual_blocks = []
    pair_max_abs = []
    ignored_stationary_rows = 0
    for i, j in pairs:
        product = grad[:, int(i)] * grad[:, int(j)]
        hij = hess[:, int(i), int(j)]
        finite = finite_phi & np.isfinite(product) & np.isfinite(hij)
        if phi.shape[1] > 0:
            product_scale = _robust_scale(product[finite])
            active = np.abs(product) > max(_EPS, float(stationary_rtol) * product_scale)
            fit_mask = finite & active
            ignored_stationary_rows += int(np.sum(finite & ~active))
        else:
            fit_mask = finite
        if phi.shape[1] > 0:
            rows.append(product[fit_mask, None] * phi[fit_mask])
            rhs.append(-hij[fit_mask])
        residual_blocks.append((i, j, finite, fit_mask, product, hij))

    if phi.shape[1] == 0:
        theta = np.zeros(0, dtype=float)
        fit_rows = int(sum(np.sum(block[2]) for block in residual_blocks))
    else:
        if not rows:
            return _failed_fit(candidate)
        A = np.concatenate(rows, axis=0)
        b = np.concatenate(rhs, axis=0)
        valid = np.isfinite(A).all(axis=1) & np.isfinite(b)
        A = A[valid]
        b = b[valid]
        fit_rows = int(A.shape[0])
        if fit_rows < max(1, min_fit_rows) or A.shape[1] == 0:
            return _failed_fit(candidate, fit_rows=fit_rows, ignored_stationary_rows=ignored_stationary_rows)
        theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        if not np.isfinite(theta).all():
            return _failed_fit(candidate, fit_rows=fit_rows, ignored_stationary_rows=ignored_stationary_rows)

    all_resid = []
    all_scale = []
    for i, j, finite, fit_mask, product, hij in residual_blocks:
        if phi.shape[1] == 0:
            pred = np.zeros_like(hij)
            score_mask = finite
        else:
            pred = product * (phi @ theta)
            score_mask = fit_mask
        resid = hij + pred
        scored = resid[score_mask]
        if scored.size:
            pair_max_abs.append(float(np.max(np.abs(scored))))
            all_resid.append(scored)
            all_scale.append(np.abs(hij[score_mask]) + np.abs(pred[score_mask]))

    if not all_resid:
        return _failed_fit(candidate, fit_rows=fit_rows, ignored_stationary_rows=ignored_stationary_rows)
    resid_vec = np.concatenate(all_resid)
    scale_vec = np.concatenate(all_scale)
    scale = max(float(np.nanmedian(scale_vec)), 1.0, _EPS)
    max_abs = float(np.max(np.abs(resid_vec)))
    max_rel = float(max_abs / scale)
    pair_consistency = float(np.max(pair_max_abs) - np.min(pair_max_abs)) if len(pair_max_abs) > 1 else 0.0
    stability = _resampling_stability(candidate, y, grad, hess, pairs, theta, stationary_rtol=stationary_rtol)
    return _FitResult(
        candidate=candidate,
        theta=np.asarray(theta, dtype=float),
        max_abs=max_abs,
        max_rel=max_rel,
        pair_consistency=pair_consistency,
        stability=stability,
        fit_rows=fit_rows,
        ignored_stationary_rows=ignored_stationary_rows,
        finite=True,
    )


def _resampling_stability(
    candidate: _BasisCandidate,
    y: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    theta: np.ndarray,
    *,
    stationary_rtol: float,
) -> float:
    if len(theta) == 0:
        return 0.0
    idx_even = np.arange(y.shape[0]) % 2 == 0
    idx_odd = ~idx_even
    if int(np.sum(idx_even)) < len(theta) + 2 or int(np.sum(idx_odd)) < len(theta) + 2:
        return math.inf
    fits = []
    for mask in (idx_even, idx_odd):
        sub_fit = _fit_candidate_no_stability(
            candidate,
            y[mask],
            grad[mask],
            hess[mask],
            pairs,
            stationary_rtol=stationary_rtol,
        )
        if sub_fit is None:
            return math.inf
        fits.append(sub_fit)
    denom = max(1.0, float(np.linalg.norm(theta)))
    return float(max(np.linalg.norm(fits[0] - theta), np.linalg.norm(fits[1] - theta), np.linalg.norm(fits[0] - fits[1])) / denom)


def _fit_candidate_no_stability(
    candidate: _BasisCandidate,
    y: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    *,
    stationary_rtol: float,
) -> np.ndarray | None:
    phi = candidate.evaluator(y)
    if phi.ndim == 1:
        phi = phi.reshape(-1, 1)
    if phi.shape[1] == 0:
        return np.zeros(0, dtype=float)
    rows = []
    rhs = []
    finite_phi = np.isfinite(phi).all(axis=1)
    for i, j in pairs:
        product = grad[:, int(i)] * grad[:, int(j)]
        hij = hess[:, int(i), int(j)]
        finite = finite_phi & np.isfinite(product) & np.isfinite(hij)
        product_scale = _robust_scale(product[finite])
        active = np.abs(product) > max(_EPS, float(stationary_rtol) * product_scale)
        fit_mask = finite & active
        rows.append(product[fit_mask, None] * phi[fit_mask])
        rhs.append(-hij[fit_mask])
    if not rows:
        return None
    A = np.concatenate(rows, axis=0)
    b = np.concatenate(rhs, axis=0)
    valid = np.isfinite(A).all(axis=1) & np.isfinite(b)
    A = A[valid]
    b = b[valid]
    if A.shape[0] < max(1, A.shape[1]):
        return None
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    if not np.isfinite(theta).all():
        return None
    return np.asarray(theta, dtype=float)


def _basis_candidates(y: np.ndarray) -> tuple[_BasisCandidate, ...]:
    candidates = [
        _BasisCandidate("additive", (), lambda yy: np.zeros((yy.shape[0], 0), dtype=float)),
        _BasisCandidate("constant", ("1",), lambda yy: np.ones((yy.shape[0], 1), dtype=float)),
    ]
    nonzero = np.isfinite(y) & (np.abs(y) > _EPS)
    if int(np.sum(nonzero)) >= max(8, y.shape[0] // 2):
        candidates.append(_BasisCandidate("inverse", ("1/y",), lambda yy: (1.0 / yy).reshape(-1, 1)))
        candidates.append(
            _BasisCandidate(
                "constant_plus_inverse",
                ("1", "1/y"),
                lambda yy: np.column_stack([np.ones(yy.shape[0], dtype=float), 1.0 / yy]),
            )
        )
    return tuple(candidates)


def _classify_link(candidate: _BasisCandidate, theta: np.ndarray) -> tuple[str, float | None, str]:
    coeff_tol = 1.0e-6
    if candidate.name == "additive" or theta.size == 0 or float(np.linalg.norm(theta)) <= coeff_tol:
        return "additive", 1.0, "0"
    if candidate.name == "constant":
        c = float(theta[0])
        return "exponential", None, f"{c:.12g}"
    if candidate.name == "inverse":
        k = float(theta[0])
        return _classify_inverse_coefficient(k)
    if candidate.name == "constant_plus_inverse":
        c = float(theta[0])
        k = float(theta[1])
        if abs(c) <= coeff_tol:
            return _classify_inverse_coefficient(k)
        return "exponential_power", None, f"{c:.12g} + {k:.12g}/y"
    return candidate.name, None, " + ".join(f"{float(v):.12g}*{name}" for v, name in zip(theta, candidate.basis_names))


def _classify_inverse_coefficient(k: float) -> tuple[str, float | None, str]:
    if abs(k + 1.0) <= 1.0e-6:
        return "log", 0.0, "-1/y"
    if abs(k + 2.0) <= 1.0e-6:
        return "reciprocal", -1.0, "-2/y"
    exponent = float(k + 1.0)
    if abs(exponent - 1.0) <= 1.0e-6:
        return "additive", 1.0, "0"
    return "power", exponent, f"{k:.12g}/y"


def _psi_prime_nonzero(family: str, exponent: float | None, y: np.ndarray) -> bool:
    if family in {"additive", "exponential"}:
        return True
    if family in {"log", "reciprocal", "power", "exponential_power"}:
        if not np.all(np.isfinite(y)):
            return False
        if float(np.min(np.abs(y))) <= _EPS:
            return False
        if family == "power" and exponent is not None and abs(float(exponent)) <= _EPS:
            return False
        return True
    return False


def _normalize_blocks(blocks: Sequence[Sequence[int]] | None, input_dim: int) -> tuple[tuple[int, ...], ...]:
    if blocks is None:
        return tuple((i,) for i in range(int(input_dim)))
    out = []
    seen: set[int] = set()
    for block in blocks:
        item = tuple(sorted(int(v) for v in block))
        if not item:
            continue
        for idx in item:
            if idx < 0 or idx >= int(input_dim):
                raise ValueError(f"block index {idx} is outside input dimension {input_dim}")
            if idx in seen:
                raise ValueError(f"block index {idx} appears in more than one block")
            seen.add(idx)
        out.append(item)
    return tuple(out)


def _cross_pairs(blocks: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for a in range(len(blocks)):
        for b in range(a + 1, len(blocks)):
            for i in blocks[a]:
                for j in blocks[b]:
                    pairs.append((int(i), int(j)))
    return pairs


def _failed_fit(
    candidate: _BasisCandidate,
    *,
    fit_rows: int = 0,
    ignored_stationary_rows: int = 0,
) -> _FitResult:
    return _FitResult(
        candidate=candidate,
        theta=np.zeros(len(candidate.basis_names), dtype=float),
        max_abs=math.inf,
        max_rel=math.inf,
        pair_consistency=math.inf,
        stability=math.inf,
        fit_rows=int(fit_rows),
        ignored_stationary_rows=int(ignored_stationary_rows),
        finite=False,
    )


def _robust_scale(v: np.ndarray) -> float:
    arr = np.asarray(v, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    return max(float(np.nanmedian(np.abs(arr))), float(np.nanstd(arr)), 1.0)


def _to_vector(a: Any, name: str) -> np.ndarray:
    arr = _to_numpy(a).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).any():
        raise ValueError(f"{name} has no finite values")
    return arr.astype(float)


def _to_matrix(a: Any, name: str) -> np.ndarray:
    arr = _to_numpy(a)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (N, n)")
    return arr.astype(float)


def _to_hessian(a: Any, name: str) -> np.ndarray:
    arr = _to_numpy(a)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (N, n, n)")
    return arr.astype(float)


def _to_numpy(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


__all__ = [
    "OutputLinkWitness",
    "discover_output_link_separability",
]
