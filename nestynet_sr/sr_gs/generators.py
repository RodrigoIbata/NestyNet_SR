# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Low-complexity infinitesimal-generator probes for NestyNet-SR.

V3 keeps the named V2 generators and adds a sparse affine probe for

    X = (A x + b) . grad_x,      X f ≈ alpha + beta f.

The module emits diagnostics/proposals only.  Stage-A validation remains the
place where accepted AST changes are confirmed or rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import math
import numpy as np

from .config import GeneralizedSymmetryConfig


@dataclass(frozen=True)
class GeneratorSpec:
    """A discovered/audited infinitesimal symmetry or equivariance candidate."""

    family: str
    kind: str
    axes: tuple[int, ...]
    xi_coeffs: tuple[float, ...] = ()
    output_alpha: float = 0.0
    output_beta: float = 0.0
    residual_rel: float = math.inf
    confidence: float = 0.0
    invariant: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    accepted: bool = True

    @property
    def is_equivariant(self) -> bool:
        return abs(float(self.output_alpha)) > 0.0 or abs(float(self.output_beta)) > 0.0


def _to_numpy(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _as_2d_grad(g: Any) -> np.ndarray:
    arr = _to_numpy(g)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"gradient array must be (N,k) or (N,1,k); got {arr.shape}")
    return arr


def _safe_scale(v: np.ndarray, floor: float = 1.0e-12) -> float:
    vv = np.asarray(v, dtype=float).reshape(-1)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return float(floor)
    med = float(np.median(np.abs(vv)))
    rms = float(np.sqrt(np.mean(vv * vv)))
    return max(floor, med, 0.25 * rms)



def _variation_scale(v: np.ndarray, floor: float = 1.0e-12) -> float:
    """Offset-invariant robust scale for a sampled scalar field."""

    vv = np.asarray(v, dtype=float).reshape(-1)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return float(floor)
    centered = vv - float(np.median(vv))
    mad = 1.4826 * float(np.median(np.abs(centered)))
    std = float(np.std(vv))
    q25, q75 = np.quantile(vv, [0.25, 0.75])
    iqr = 0.7413 * float(abs(q75 - q25))
    return max(float(floor), mad, std, iqr)

def _fit_output_action(q: np.ndarray, y: np.ndarray, enable: bool) -> tuple[float, float, float, float]:
    """Fit q ≈ alpha + beta*y and return alpha,beta,relative_residual,r2."""

    q = np.asarray(q, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(q) & np.isfinite(y)
    if int(mask.sum()) < 8:
        return 0.0, 0.0, math.inf, -math.inf
    q = q[mask]
    y = y[mask]
    scale = _safe_scale(q)
    if not enable:
        return 0.0, 0.0, float(np.sqrt(np.mean(q * q)) / scale), 0.0

    # Center before fitting. The direct [1,y] least-squares system becomes
    # badly conditioned when y carries a large irrelevant offset.
    q_mean = float(np.mean(q))
    y_mean = float(np.mean(y))
    qc = q - q_mean
    yc = y - y_mean
    yy = float(np.dot(yc, yc))
    beta = float(np.dot(yc, qc) / yy) if yy > np.finfo(float).eps * max(1.0, float(np.dot(y, y))) else 0.0
    alpha = float(q_mean - beta * y_mean)
    pred = q_mean + beta * yc
    res = q - pred
    ss_res = float(np.mean(res * res))
    ss_tot = float(np.mean(qc * qc))
    tiny = (64.0 * np.finfo(float).eps * max(scale, 1.0)) ** 2
    if ss_tot <= tiny:
        r2 = 1.0 if ss_res <= tiny else -math.inf
    else:
        r2 = 1.0 - ss_res / ss_tot
    return alpha, beta, float(math.sqrt(max(0.0, ss_res)) / scale), float(r2)


def _confidence_from_residual(resid: float, cfg: GeneralizedSymmetryConfig) -> float:
    if not math.isfinite(float(resid)):
        return 0.0
    lo = max(1.0e-12, float(cfg.residual_tol))
    hi = max(lo * 1.01, float(cfg.audit_residual_tol))
    if resid <= lo:
        return float(max(0.0, min(1.0, 1.0 - 0.5 * resid / lo)))
    if resid >= hi:
        return 0.0
    return float(max(0.0, min(1.0, (hi - resid) / (hi - lo) * 0.5)))


def _accept(resid: float, confidence: float, cfg: GeneralizedSymmetryConfig) -> bool:
    if confidence >= float(cfg.min_confidence):
        return True
    return bool(cfg.active()) and not cfg.proposing() and resid <= float(cfg.audit_residual_tol)


def _add_named_spec(
    all_specs: list[GeneratorSpec],
    specs: list[GeneratorSpec],
    *,
    cfg: GeneralizedSymmetryConfig,
    cols: Sequence[int],
    y: np.ndarray,
    family: str,
    kind: str,
    axes_local: tuple[int, ...],
    q: np.ndarray,
    invariant: str,
    xi_coeffs=(),
    evidence=None,
) -> None:
    enable_eta = bool(cfg.output_equivariance)
    alpha, beta, resid, r2 = _fit_output_action(q, y, enable_eta)
    strict_resid = float(np.sqrt(np.mean(np.asarray(q, dtype=float) ** 2)) / _variation_scale(y))
    used_output_action = False
    if strict_resid <= resid:
        alpha2, beta2, resid2, r22 = 0.0, 0.0, strict_resid, 0.0
    else:
        alpha2, beta2, resid2, r22 = alpha, beta, resid, r2
        used_output_action = bool(enable_eta and (abs(alpha2) > 1.0e-12 or abs(beta2) > 1.0e-12))
    conf = _confidence_from_residual(resid2, cfg)
    accepted = bool(_accept(resid2, conf, cfg))
    if used_output_action and r22 < float(getattr(cfg, "equivariance_min_r2", 0.985)):
        accepted = False
    axes_global = tuple(int(cols[i]) for i in axes_local)
    spec = GeneratorSpec(
        family=family,
        kind=kind,
        axes=axes_global,
        xi_coeffs=tuple(float(v) for v in xi_coeffs),
        output_alpha=float(alpha2),
        output_beta=float(beta2),
        residual_rel=float(resid2),
        confidence=float(conf),
        invariant=invariant,
        evidence={
            "local_axes": axes_local,
            "r2_output_action": float(r22),
            "used_output_action": bool(used_output_action),
            "equivariance_min_r2": float(getattr(cfg, "equivariance_min_r2", 0.985)),
            **(evidence or {}),
        },
        accepted=accepted,
    )
    all_specs.append(spec)
    if accepted:
        specs.append(spec)


def _pairs(k: int) -> Iterable[tuple[int, int]]:
    for i in range(k):
        for j in range(i + 1, k):
            yield i, j


def _rms(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def _snap_small(v: np.ndarray, tol: float = 0.20) -> tuple[float, ...]:
    out = []
    allowed = np.asarray([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    for x in np.asarray(v, dtype=float).reshape(-1):
        j = int(np.argmin(np.abs(allowed - x)))
        out.append(float(allowed[j]) if abs(float(allowed[j] - x)) <= tol else float(x))
    return tuple(out)


def _classify_affine_pair(raw: np.ndarray, cfg: GeneralizedSymmetryConfig) -> tuple[str, str | None, tuple[float, ...], dict[str, Any]]:
    """Classify pair coefficients [b_i,b_j,A_ii,A_ij,A_ji,A_jj]."""

    c = np.asarray(raw, dtype=float).reshape(-1)
    if c.size != 6:
        return "unclassified_pair", None, tuple(float(x) for x in c), {}
    scale = float(np.max(np.abs(c)))
    if scale <= 1e-12:
        return "unclassified_pair", None, tuple(float(x) for x in c), {}
    n = c / scale
    b_i, b_j, a_ii, a_ij, a_ji, a_jj = n.tolist()
    tol = max(0.08, float(getattr(cfg, "general_affine_snap_tol", getattr(cfg, "snap_tol", 0.20))))
    ev = {"normalized_coeffs": [float(x) for x in n]}

    # Pure translation generator b_i d_i + b_j d_j: invariant b_j*x_i - b_i*x_j.
    if max(abs(a_ii), abs(a_ij), abs(a_ji), abs(a_jj)) < tol and max(abs(b_i), abs(b_j)) > 0.5:
        # Translation directions are projective and need not have small-rational
        # coefficients. Snapping 1/sqrt(2) to 1/2, for example, constructs the
        # wrong quotient coordinate even though the generator residual is exact.
        coeffs = np.asarray([b_i, b_j], dtype=float)
        coeffs /= max(float(np.max(np.abs(coeffs))), 1.0e-30)
        nz = np.flatnonzero(np.abs(coeffs) > 1.0e-12)
        if nz.size and coeffs[int(nz[0])] < 0.0:
            coeffs = -coeffs
        return "affine_translation_pair", "linear_affine_invariant", tuple(float(v) for v in coeffs), ev

    # Diagonal scaling forms.
    if max(abs(b_i), abs(b_j), abs(a_ij), abs(a_ji)) < tol and max(abs(a_ii), abs(a_jj)) > 0.5:
        if abs(a_ii - a_jj) < 2.0 * tol:
            return "affine_common_scaling_pair", "ratio coordinate", (1.0, 1.0), ev
        if abs(a_ii + a_jj) < 2.0 * tol:
            return "affine_opposite_scaling_pair", "product coordinate", (1.0, -1.0), ev
        snapped = _snap_small([a_ii, a_jj], tol=tol)
        return "affine_diagonal_scaling_pair", "monomial invariant", tuple(snapped), ev

    # Off-diagonal compact/non-compact pair generators.
    if max(abs(b_i), abs(b_j), abs(a_ii), abs(a_jj)) < tol and max(abs(a_ij), abs(a_ji)) > 0.5:
        if abs(a_ij + a_ji) < 2.0 * tol:
            return "affine_rotation_pair", "radial coordinate", (-1.0, 1.0), ev
        if abs(a_ij - a_ji) < 2.0 * tol:
            return "affine_lorentz_pair", "Minkowski interval", (1.0, 1.0), ev
        snapped = _snap_small([a_ij, a_ji], tol=tol)
        return "affine_offdiag_pair", "quadratic invariant candidate", tuple(snapped), ev

    return "unclassified_pair", None, tuple(float(x) for x in n), ev


def _scaled_smallest_right_singular_vector(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a physical-coordinate null vector after column equilibration."""

    M = np.asarray(M, dtype=float)
    col_scale = np.sqrt(np.mean(M * M, axis=0))
    ref = max(float(np.max(col_scale)), 1.0)
    col_scale = np.maximum(col_scale, np.finfo(float).eps * ref)
    Z = M / col_scale.reshape(1, -1)
    _u, svals, vh = np.linalg.svd(Z, full_matrices=False)
    if vh.size == 0:
        raise ValueError("empty SVD")
    coeff = vh[-1, :] / col_scale
    cscale = float(np.max(np.abs(coeff)))
    if not math.isfinite(cscale) or cscale <= 1.0e-12:
        raise ValueError("degenerate affine generator")
    return coeff / cscale, svals, col_scale


def _finite_matvec(M: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Matrix-vector product that turns nonfinite affine actions into rejects."""

    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        q = np.asarray(M, dtype=float) @ np.asarray(coeff, dtype=float)
    if not np.all(np.isfinite(q)):
        raise ValueError("non-finite affine action")
    return q


def _project_out_output_span(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Project columns of M orthogonally to span{1,y}."""

    M_arr = np.asarray(M, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    yc = y - float(np.mean(y))
    ys = max(float(np.sqrt(np.mean(yc * yc))), 1.0e-12)
    B = np.stack([np.ones_like(y), yc / ys], axis=1)
    U, svals, _vh = np.linalg.svd(B, full_matrices=False)
    if svals.size == 0:
        return M_arr
    rank = int(np.sum(svals > np.finfo(float).eps * max(B.shape) * float(svals[0])))
    if rank <= 0:
        return M_arr
    Q = U[:, :rank]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        projected = M_arr - Q @ (Q.T @ M_arr)
    if not np.all(np.isfinite(projected)):
        return M_arr
    return projected


def _discover_structured_translation_pair_spec(
    Mm: np.ndarray,
    ym: np.ndarray,
    Gm: np.ndarray,
    *,
    axes_global: tuple[int, int],
    cfg: GeneralizedSymmetryConfig,
    grad_support_fraction: float,
) -> GeneratorSpec | None:
    """Learn an arbitrary constant translation direction on one axis pair.

    This two-column subproblem is deliberately solved before the full six-term
    affine system. A function of one linear coordinate has a multi-dimensional
    nullspace in the full affine dictionary, so the smallest full-SVD vector is
    neither unique nor reliably the constant generator we need.
    """

    searches: list[tuple[str, np.ndarray]] = [("strict_invariant", Mm)]
    if bool(cfg.output_equivariance):
        searches.append(("projected_output_equivariance", _project_out_output_span(Mm, ym)))
    g_norm = _rms(np.sqrt(Gm[:, 0] * Gm[:, 0] + Gm[:, 1] * Gm[:, 1]))
    max_null_ratio = float(getattr(cfg, "general_affine_max_null_ratio", 0.25))
    rows: list[dict[str, Any]] = []
    for objective, search_matrix in searches:
        try:
            coeff, svals, col_scale = _scaled_smallest_right_singular_vector(search_matrix)
            q = _finite_matvec(Mm, coeff)
        except Exception:
            continue
        c_norm = float(np.linalg.norm(coeff))
        strict_resid = _rms(q) / max(1.0e-12, c_norm * g_norm)
        alpha, beta, eta_resid, r2 = _fit_output_action(q, ym, bool(cfg.output_equivariance))
        if strict_resid <= eta_resid:
            alpha, beta, resid, r2 = 0.0, 0.0, strict_resid, 0.0
            used_output_action = False
        else:
            resid = eta_resid
            used_output_action = bool(abs(alpha) > 1.0e-12 or abs(beta) > 1.0e-12)
        sv_ratio = float(svals[-1] / max(svals[0], 1.0e-30)) if len(svals) else math.inf
        confidence = _confidence_from_residual(resid, cfg)
        accepted = bool(_accept(resid, confidence, cfg))
        accepted = accepted and sv_ratio <= max_null_ratio
        accepted = accepted and grad_support_fraction >= float(getattr(cfg, "min_grad_fraction", 0.05))
        if used_output_action and r2 < float(getattr(cfg, "equivariance_min_r2", 0.985)):
            accepted = False
        rows.append(
            {
                "coeff": coeff,
                "resid": float(resid),
                "strict_resid": float(strict_resid),
                "alpha": float(alpha),
                "beta": float(beta),
                "r2": float(r2),
                "used_output_action": bool(used_output_action),
                "confidence": float(confidence),
                "accepted": bool(accepted),
                "sv_ratio": float(sv_ratio),
                "objective": objective,
                "column_scales": [float(v) for v in col_scale],
            }
        )
    if not rows:
        return None
    chosen = min(rows, key=lambda row: (0 if row["accepted"] else 1, row["resid"], row["sv_ratio"]))
    coeff = np.asarray(chosen["coeff"], dtype=float)
    coeff /= max(float(np.max(np.abs(coeff))), 1.0e-30)
    nz = np.flatnonzero(np.abs(coeff) > 1.0e-12)
    if nz.size and coeff[int(nz[0])] < 0.0:
        coeff = -coeff
        chosen["alpha"] = -float(chosen["alpha"])
        chosen["beta"] = -float(chosen["beta"])
    return GeneratorSpec(
        family="general_affine",
        kind="affine_translation_pair",
        axes=axes_global,
        xi_coeffs=tuple(float(v) for v in coeff),
        output_alpha=float(chosen["alpha"]),
        output_beta=float(chosen["beta"]),
        residual_rel=float(chosen["resid"]),
        confidence=float(chosen["confidence"]),
        invariant=f"linear_affine_invariant on x{axes_global[0]},x{axes_global[1]}",
        evidence={
            "structured_family": "translation",
            "sv_min_over_sv_max": float(chosen["sv_ratio"]),
            "max_null_ratio": max_null_ratio,
            "gradient_support_fraction": float(grad_support_fraction),
            "min_grad_fraction": float(getattr(cfg, "min_grad_fraction", 0.05)),
            "gradient_rms": float(g_norm),
            "strict_residual_rel": float(chosen["strict_resid"]),
            "r2_output_action": float(chosen["r2"]),
            "used_output_action": bool(chosen["used_output_action"]),
            "equivariance_min_r2": float(getattr(cfg, "equivariance_min_r2", 0.985)),
            "svd_objective": str(chosen["objective"]),
            "svd_column_scales": list(chosen["column_scales"]),
        },
        accepted=bool(chosen["accepted"]),
    )


def _discover_general_affine_pair_specs(
    X: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    cols: Sequence[int],
    cfg: GeneralizedSymmetryConfig,
) -> tuple[list[GeneratorSpec], list[GeneratorSpec]]:
    """Discover sparse pairwise affine generators by projected null tests."""

    accepted: list[GeneratorSpec] = []
    all_specs: list[GeneratorSpec] = []
    max_pairs = max(1, int(getattr(cfg, "max_pair_generators", 16)))
    for count, (i, j) in enumerate(_pairs(min(X.shape[1], G.shape[1]))):
        if count >= max_pairs:
            break
        M = np.stack(
            [
                G[:, i],
                G[:, j],
                X[:, i] * G[:, i],
                X[:, j] * G[:, i],
                X[:, i] * G[:, j],
                X[:, j] * G[:, j],
            ],
            axis=1,
        )
        mask = np.all(np.isfinite(M), axis=1) & np.isfinite(y)
        if int(mask.sum()) < 16:
            continue
        Mm = M[mask]
        ym = y[mask]
        Xm = X[mask]
        Gm = G[mask]
        searches: list[tuple[str, np.ndarray]] = [("strict_invariant", Mm)]
        if bool(cfg.output_equivariance):
            searches.append(("projected_output_equivariance", _project_out_output_span(Mm, ym)))

        grad_pair = np.sqrt(Gm[:, i] * Gm[:, i] + Gm[:, j] * Gm[:, j])
        grad_scale = _safe_scale(grad_pair)
        grad_support_fraction = float(np.mean(np.abs(grad_pair) > 1.0e-8 * grad_scale))
        min_grad_fraction = float(getattr(cfg, "min_grad_fraction", 0.05))
        max_null_ratio = float(getattr(cfg, "general_affine_max_null_ratio", 0.25))

        structured = _discover_structured_translation_pair_spec(
            Mm[:, :2],
            ym,
            Gm[:, [i, j]],
            axes_global=(int(cols[i]), int(cols[j])),
            cfg=cfg,
            grad_support_fraction=grad_support_fraction,
        )
        if structured is not None:
            all_specs.append(structured)
            if structured.accepted:
                accepted.append(structured)
                continue

        candidates: list[dict[str, Any]] = []
        for objective, search_matrix in searches:
            try:
                coeff, svals, col_scale = _scaled_smallest_right_singular_vector(search_matrix)
                q = _finite_matvec(Mm, coeff)
            except Exception:
                continue
            b_i, b_j, a_ii, a_ij, a_ji, a_jj = coeff.tolist()
            vi = b_i + a_ii * Xm[:, i] + a_ij * Xm[:, j]
            vj = b_j + a_ji * Xm[:, i] + a_jj * Xm[:, j]
            v_norm = _rms(np.sqrt(vi * vi + vj * vj))
            g_norm = _rms(np.sqrt(Gm[:, i] * Gm[:, i] + Gm[:, j] * Gm[:, j]))
            strict_resid = _rms(q) / max(1.0e-12, v_norm * g_norm)
            alpha, beta, eta_resid, r2 = _fit_output_action(q, ym, bool(cfg.output_equivariance))
            if strict_resid <= eta_resid:
                alpha, beta, resid, r2 = 0.0, 0.0, strict_resid, 0.0
                used_output_action = False
            else:
                resid = eta_resid
                used_output_action = bool(abs(alpha) > 1.0e-12 or abs(beta) > 1.0e-12)
            sv_min_over_sv_max = float(svals[-1] / max(svals[0], 1.0e-30)) if len(svals) else math.inf
            sv_min_over_prev = float(svals[-1] / max(svals[-2], 1.0e-30)) if len(svals) >= 2 else math.inf
            conf = _confidence_from_residual(resid, cfg)
            ok = bool(_accept(resid, conf, cfg))
            if grad_support_fraction < min_grad_fraction or sv_min_over_prev > max_null_ratio:
                ok = False
            if used_output_action and r2 < float(getattr(cfg, "equivariance_min_r2", 0.985)):
                ok = False
            candidates.append(
                {
                    "coeff": coeff,
                    "resid": float(resid),
                    "strict_resid": float(strict_resid),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "r2": float(r2),
                    "used_output_action": bool(used_output_action),
                    "confidence": float(conf),
                    "accepted": bool(ok),
                    "sv_min_over_sv_max": sv_min_over_sv_max,
                    "sv_min_over_prev": sv_min_over_prev,
                    "vector_field_rms": float(v_norm),
                    "gradient_rms": float(g_norm),
                    "svd_objective": objective,
                    "column_scales": [float(v) for v in col_scale],
                }
            )
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda row: (0 if bool(row["accepted"]) else 1, float(row["resid"]), 1 if bool(row["used_output_action"]) else 0),
        )
        coeff = np.asarray(chosen["coeff"], dtype=float)
        kind, inv_label, xi_coeffs, ev2 = _classify_affine_pair(coeff, cfg)
        if kind == "unclassified_pair" and not bool(getattr(cfg, "general_affine_report_unclassified", True)):
            continue
        axes_global = (int(cols[i]), int(cols[j]))
        invariant = f"{inv_label} on x{axes_global[0]},x{axes_global[1]}" if inv_label is not None else None
        spec = GeneratorSpec(
            family="general_affine",
            kind=kind,
            axes=axes_global,
            xi_coeffs=tuple(float(v) for v in xi_coeffs),
            output_alpha=float(chosen["alpha"]),
            output_beta=float(chosen["beta"]),
            residual_rel=float(chosen["resid"]),
            confidence=float(chosen["confidence"]),
            invariant=invariant,
            evidence={
                "local_axes": (i, j),
                "sv_min_over_sv_max": float(chosen["sv_min_over_sv_max"]),
                "sv_min_over_prev": float(chosen["sv_min_over_prev"]),
                "max_null_ratio": max_null_ratio,
                "gradient_support_fraction": grad_support_fraction,
                "min_grad_fraction": min_grad_fraction,
                "vector_field_rms": float(chosen["vector_field_rms"]),
                "gradient_rms": float(chosen["gradient_rms"]),
                "strict_residual_rel": float(chosen["strict_resid"]),
                "r2_output_action": float(chosen["r2"]),
                "used_output_action": bool(chosen["used_output_action"]),
                "equivariance_min_r2": float(getattr(cfg, "equivariance_min_r2", 0.985)),
                "svd_objective": str(chosen["svd_objective"]),
                "svd_column_scales": list(chosen["column_scales"]),
                "candidate_objectives": [str(row["svd_objective"]) for row in candidates],
                **ev2,
            },
            accepted=bool(chosen["accepted"]),
        )
        all_specs.append(spec)
        if spec.accepted:
            accepted.append(spec)
    return accepted, all_specs


def discover_generator_specs(
    x_vals: Any,
    y_vals: Any,
    dydx_vals: Any,
    *,
    cols: Sequence[int] | None = None,
    cfg: GeneralizedSymmetryConfig | None = None,
    include_rejected: bool = False,
) -> list[GeneratorSpec]:
    """Discover named and/or learned affine-generator witnesses."""

    cfg = cfg or GeneralizedSymmetryConfig()
    if not cfg.active():
        return []

    X = _to_numpy(x_vals).astype(float)
    G = _as_2d_grad(dydx_vals).astype(float)
    y = _to_numpy(y_vals).astype(float).reshape(-1)
    if X.ndim != 2 or G.ndim != 2:
        return []
    n = min(X.shape[0], G.shape[0], y.shape[0])
    if n < 16:
        return []
    X = X[:n]
    G = G[:n]
    y = y[:n]
    k = min(X.shape[1], G.shape[1])
    X = X[:, :k]
    G = G[:, :k]
    if cols is None:
        cols = tuple(range(k))
    cols = tuple(int(c) for c in cols[:k])

    specs: list[GeneratorSpec] = []
    all_specs: list[GeneratorSpec] = []

    if bool(getattr(cfg, "known_generators", True)) and bool(getattr(cfg, "known_lie", True)):
        if cfg.translations:
            for i in range(k):
                _add_named_spec(all_specs, specs, cfg=cfg, cols=cols, y=y, family="translation", kind="axis", axes_local=(i,), q=G[:, i], invariant=f"drop_x{cols[i]}", xi_coeffs=(1.0,))

        if cfg.diagonal_translations and k >= 2:
            for i, j in list(_pairs(k))[: int(cfg.max_pair_generators)]:
                _add_named_spec(all_specs, specs, cfg=cfg, cols=cols, y=y, family="translation", kind="diagonal_plus", axes_local=(i, j), q=G[:, i] + G[:, j], invariant=f"x{cols[i]}-x{cols[j]}", xi_coeffs=(1.0, 1.0))
                _add_named_spec(all_specs, specs, cfg=cfg, cols=cols, y=y, family="translation", kind="diagonal_minus", axes_local=(i, j), q=G[:, i] - G[:, j], invariant=f"x{cols[i]}+x{cols[j]}", xi_coeffs=(1.0, -1.0))

        if cfg.scalings:
            for i in range(k):
                _add_named_spec(all_specs, specs, cfg=cfg, cols=cols, y=y, family="scaling", kind="axis", axes_local=(i,), q=X[:, i] * G[:, i], invariant=f"scale_x{cols[i]}", xi_coeffs=(1.0,))
            if k >= 2:
                for i, j in list(_pairs(k))[: int(cfg.max_pair_generators)]:
                    _add_named_spec(all_specs, specs, cfg=cfg, cols=cols, y=y, family="scaling", kind="common_pair", axes_local=(i, j), q=X[:, i] * G[:, i] + X[:, j] * G[:, j], invariant=f"x{cols[i]}/x{cols[j]}", xi_coeffs=(1.0, 1.0))
                    _add_named_spec(all_specs, specs, cfg=cfg, cols=cols, y=y, family="scaling", kind="opposite_pair", axes_local=(i, j), q=X[:, i] * G[:, i] - X[:, j] * G[:, j], invariant=f"x{cols[i]}*x{cols[j]}", xi_coeffs=(1.0, -1.0))

        if cfg.rotations and k >= 2:
            for i, j in list(_pairs(k))[: int(cfg.max_pair_generators)]:
                q = -X[:, j] * G[:, i] + X[:, i] * G[:, j]
                _add_named_spec(all_specs, specs, cfg=cfg, cols=cols, y=y, family="rotation", kind="so2_pair", axes_local=(i, j), q=q, invariant=f"x{cols[i]}^2+x{cols[j]}^2", xi_coeffs=(-1.0, 1.0))

        if cfg.lorentz_boosts and k >= 2:
            for i, j in list(_pairs(k))[: int(cfg.max_pair_generators)]:
                q = X[:, j] * G[:, i] + X[:, i] * G[:, j]
                _add_named_spec(all_specs, specs, cfg=cfg, cols=cols, y=y, family="lorentz", kind="boost_pair", axes_local=(i, j), q=q, invariant=f"x{cols[i]}^2-x{cols[j]}^2", xi_coeffs=(1.0, 1.0))

    if bool(getattr(cfg, "general_affine", False)) or bool(getattr(cfg, "affine_dense", False)):
        acc, all_aff = _discover_general_affine_pair_specs(X, G, y, cols, cfg)
        specs.extend(acc)
        all_specs.extend(all_aff)

    def key(s: GeneratorSpec):
        equiv_pen = 0 if (abs(s.output_alpha) < 1e-10 and abs(s.output_beta) < 1e-10) else 1
        fam_pri = {"translation": 0, "scaling": 1, "rotation": 2, "lorentz": 3, "general_affine": 4}.get(s.family, 9)
        unclass_pen = 1 if str(s.kind).startswith("unclassified") else 0
        return (equiv_pen, fam_pri, unclass_pen, len(s.axes), s.residual_rel, -s.confidence)

    specs = sorted(specs, key=key)
    if include_rejected:
        rejected = [s for s in all_specs if not bool(getattr(s, "accepted", False))]
        rejected = sorted(rejected, key=lambda s: (s.residual_rel, -s.confidence))
        k_rej = max(0, int(getattr(cfg, "report_top_k_rejected", 40)))
        return specs[: max(0, int(cfg.max_stagea_proposals))] + rejected[:k_rej]
    return specs[: max(0, int(cfg.max_stagea_proposals))]


def summarize_specs(specs: Sequence[GeneratorSpec]) -> list[dict[str, Any]]:
    """JSON-friendly summary used by examples and diagnostics."""

    out = []
    for s in specs:
        out.append(
            {
                "family": s.family,
                "kind": s.kind,
                "axes": list(s.axes),
                "invariant": s.invariant,
                "xi_coeffs": list(s.xi_coeffs),
                "output_alpha": s.output_alpha,
                "output_beta": s.output_beta,
                "residual_rel": s.residual_rel,
                "confidence": s.confidence,
                "evidence": dict(s.evidence or {}),
                "accepted": bool(getattr(s, "accepted", True)),
            }
        )
    return out
