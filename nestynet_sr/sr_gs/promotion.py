# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Promotion gates for generalized-symmetry reductions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nestynet_sr.sr_core.ast_simplify import node_count
_EPS = 1.0e-12


@dataclass(frozen=True)
class ReductionPromotionDecision:
    """Audit record for deciding whether a reduction may enter Stage A/FSS."""

    state: str
    accepted: bool
    reason: str
    confidence: float = 0.0
    residual_score: float = float("inf")
    residual_tol: float = 0.0
    complexity_score: float = 0.0
    complexity_limit: float = 0.0
    stability_score: float = 0.0
    stability_limit: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "confidence": float(self.confidence),
            "residual_score": float(self.residual_score),
            "residual_tol": float(self.residual_tol),
            "complexity_score": float(self.complexity_score),
            "complexity_limit": float(self.complexity_limit),
            "stability_score": float(self.stability_score),
            "stability_limit": float(self.stability_limit),
            "evidence": dict(self.evidence),
        }


def evaluate_reduction_promotion(plan: Any, cfg: Any) -> ReductionPromotionDecision:
    """Return a conservative promotion decision for a compiled reduction plan."""

    if not bool(getattr(cfg, "active", lambda: False)()):
        return _decision("disabled", False, "gs_disabled")
    if not bool(getattr(cfg, "general_affine_promotion", True)):
        return _decision("disabled", False, "promotion_disabled")
    if not bool(getattr(cfg, "proposing", lambda: False)()):
        return _decision("audit", False, "not_in_proposing_mode", plan=plan, cfg=cfg)

    if str(getattr(plan, "status", "")) != "compiled":
        return _decision("rejected", False, f"reduction_not_compiled:{getattr(plan, 'reason', '')}", plan=plan, cfg=cfg)

    algebra = getattr(plan, "algebra", None)
    cert = getattr(algebra, "certificate", None)
    if cert is not None and not bool(getattr(cert, "quotient_ready", False)):
        return _decision("rejected", False, f"certificate:{getattr(cert, 'quotient_policy', '')}", plan=plan, cfg=cfg)

    output_action = getattr(plan, "output_action", None)
    if bool(getattr(output_action, "is_equivariant", False)) and not bool(getattr(cfg, "general_affine_promote_output_equivariant", False)):
        return _decision("rejected", False, "output_equivariant_normal_form_not_stagea_target", plan=plan, cfg=cfg)

    coord = _first_visible_invariant(plan)
    if coord is None:
        return _decision("rejected", False, "no_visible_invariant_coordinate", plan=plan, cfg=cfg)

    raw_support = tuple(int(v) for v in tuple(getattr(coord, "raw_var_idxs", ()) or getattr(coord, "raw_support", ()) or ()))
    if not raw_support:
        return _decision("rejected", False, "coordinate_has_no_raw_support", plan=plan, cfg=cfg, coord=coord)

    # The reduced target H(z) is well-defined only when the invariant
    # coordinates span the full quotient. A rank-varying distribution can
    # yield fewer global (linear) invariants than the quotient codimension —
    # e.g. an exact rotation-like symmetry in two of four variables has a
    # 3-dim quotient but only 2 linear annihilator coordinates — and
    # promoting a partial chart would propose a coordinate the target does
    # not factor through.
    codimension = int(getattr(plan, "quotient_codimension", 0) or 0)
    n_visible_invariants = sum(
        1
        for c in (getattr(plan, "invariant_coordinates", ()) or ())
        if getattr(c, "ast", None) is not None and not isinstance(getattr(c, "ast", None), str)
    )
    if codimension > 0 and n_visible_invariants < codimension:
        return _decision(
            "rejected",
            False,
            "invariant_coordinates_underspan_quotient",
            plan=plan,
            cfg=cfg,
            coord=coord,
        )

    residual = _residual_score(plan)
    residual_tol = float(getattr(cfg, "general_affine_promotion_residual_tol", getattr(cfg, "residual_tol", 0.03)))
    residual_tol = max(float(residual_tol), _EPS)
    calibrated_tier = False
    calibration: dict[str, Any] = {}
    if not np.isfinite(residual) or residual > residual_tol:
        # Noise-calibrated tier (opt-in): promote on surrogate-noise-relative
        # evidence — spectral-gap nullity contrast, held-out consistency, and
        # (below) bootstrap subspace stability — instead of the absolute
        # residual tolerance that only oracle-exact gradients can meet.
        calibrated_tier, calibration, calibrated_reason = _noise_calibrated_residual_gate(plan, cfg)
        if not calibrated_tier:
            return _decision(
                "rejected",
                False,
                calibrated_reason,
                plan=plan,
                cfg=cfg,
                coord=coord,
                residual=residual,
                residual_tol=residual_tol,
                extra_evidence=calibration or None,
            )

    stability = _stability_score(plan)
    stability_limit = float(getattr(cfg, "general_affine_promotion_max_bootstrap_angle", 1.0e-3))
    if calibrated_tier:
        stability_limit = float(getattr(cfg, "noise_calibrated_bootstrap_angle_tol", 0.10))
        algebra_angles = tuple(getattr(getattr(plan, "algebra", None), "bootstrap_principal_angles", ()) or ())
        if not algebra_angles:
            return _decision("rejected", False, "noise_calibrated_requires_bootstrap_evidence", plan=plan, cfg=cfg, coord=coord, residual=residual, residual_tol=residual_tol, stability=stability, stability_limit=stability_limit, extra_evidence=calibration)
    if np.isfinite(stability) and stability > stability_limit:
        return _decision("rejected", False, "bootstrap_instability_above_threshold", plan=plan, cfg=cfg, coord=coord, residual=residual, residual_tol=residual_tol, stability=stability, stability_limit=stability_limit, extra_evidence=calibration or None)

    complexity = float(node_count(getattr(coord, "ast")))
    complexity_limit = float(getattr(cfg, "general_affine_promotion_max_chart_complexity", 24))
    if complexity > complexity_limit:
        return _decision("rejected", False, "chart_complexity_above_limit", plan=plan, cfg=cfg, coord=coord, residual=residual, residual_tol=residual_tol, stability=stability, stability_limit=stability_limit, complexity=complexity, complexity_limit=complexity_limit, extra_evidence=calibration or None)

    if calibrated_tier:
        gap = float(calibration.get("spectral_gap", 0.0))
        min_gap = max(float(calibration.get("min_spectral_gap", 10.0)), _EPS)
        confidence = max(0.0, min(1.0, 1.0 - min_gap / max(gap, min_gap)))
    else:
        confidence = max(0.0, min(1.0, 1.0 - residual / residual_tol))
    extra = {"promotion_tier": "noise_calibrated" if calibrated_tier else "absolute"}
    if calibration:
        extra["noise_calibration"] = calibration
    return _decision(
        "promoted",
        True,
        "passed_promotion_gate",
        plan=plan,
        cfg=cfg,
        coord=coord,
        residual=residual,
        residual_tol=residual_tol,
        stability=stability,
        stability_limit=stability_limit,
        complexity=complexity,
        complexity_limit=complexity_limit,
        confidence=confidence,
        extra_evidence=extra,
    )


def _noise_calibrated_residual_gate(plan: Any, cfg: Any) -> tuple[bool, dict[str, Any], str]:
    """Surrogate-noise-relative residual evidence for the calibrated tier.

    Requires the algebra to have been solved with the spectral-gap nullity
    strategy (the gap is the calibration statistic) and its held-out residual
    to be consistent with the train residual.  Returns
    ``(passed, calibration_info, rejection_reason)``.
    """

    if not bool(getattr(cfg, "general_affine_promotion_noise_calibrated", False)):
        return False, {}, "heldout_residual_above_threshold"
    algebra = getattr(plan, "algebra", None)
    evidence = dict(getattr(algebra, "evidence", {}) or {})
    gap = float(evidence.get("spectral_gap", 0.0) or 0.0)
    min_gap = float(getattr(cfg, "noise_calibrated_min_spectral_gap", 10.0))
    train = float(getattr(algebra, "train_residual_rel", float("inf")))
    heldout = float(getattr(algebra, "heldout_residual_rel", float("inf")))
    factor = float(getattr(cfg, "noise_calibrated_heldout_factor", 3.0))
    calibration = {
        "spectral_gap": gap,
        "min_spectral_gap": min_gap,
        "nullity_strategy": str(evidence.get("nullity_strategy", "")),
        "train_residual_rel": train,
        "heldout_residual_rel": heldout,
        "heldout_factor": factor,
    }
    if gap < min_gap:
        return False, calibration, "noise_calibrated_spectral_gap_below_threshold"
    if not (np.isfinite(train) and np.isfinite(heldout)):
        return False, calibration, "noise_calibrated_nonfinite_residuals"
    if heldout > max(factor * train, 10.0 * _EPS):
        return False, calibration, "noise_calibrated_heldout_inconsistent"
    return True, calibration, ""


def _decision(
    state: str,
    accepted: bool,
    reason: str,
    *,
    plan: Any | None = None,
    cfg: Any | None = None,
    coord: Any | None = None,
    residual: float | None = None,
    residual_tol: float | None = None,
    stability: float | None = None,
    stability_limit: float | None = None,
    complexity: float | None = None,
    complexity_limit: float | None = None,
    confidence: float = 0.0,
    extra_evidence: dict[str, Any] | None = None,
) -> ReductionPromotionDecision:
    residual_v = _residual_score(plan) if residual is None and plan is not None else float("inf") if residual is None else float(residual)
    residual_tol_v = (
        float(getattr(cfg, "general_affine_promotion_residual_tol", getattr(cfg, "residual_tol", 0.03)))
        if residual_tol is None and cfg is not None
        else 0.0
        if residual_tol is None
        else float(residual_tol)
    )
    stability_v = _stability_score(plan) if stability is None and plan is not None else 0.0 if stability is None else float(stability)
    stability_limit_v = (
        float(getattr(cfg, "general_affine_promotion_max_bootstrap_angle", 1.0e-3))
        if stability_limit is None and cfg is not None
        else 0.0
        if stability_limit is None
        else float(stability_limit)
    )
    if complexity is None and coord is not None:
        complexity_v = float(node_count(getattr(coord, "ast")))
    else:
        complexity_v = 0.0 if complexity is None else float(complexity)
    complexity_limit_v = (
        float(getattr(cfg, "general_affine_promotion_max_chart_complexity", 24))
        if complexity_limit is None and cfg is not None
        else 0.0
        if complexity_limit is None
        else float(complexity_limit)
    )
    evidence = _evidence(plan, coord)
    if extra_evidence:
        evidence.update(extra_evidence)
    return ReductionPromotionDecision(
        state=str(state),
        accepted=bool(accepted),
        reason=str(reason),
        confidence=float(confidence),
        residual_score=float(residual_v),
        residual_tol=float(residual_tol_v),
        complexity_score=float(complexity_v),
        complexity_limit=float(complexity_limit_v),
        stability_score=float(stability_v),
        stability_limit=float(stability_limit_v),
        evidence=evidence,
    )


def _first_visible_invariant(plan: Any) -> Any | None:
    for coord in getattr(plan, "invariant_coordinates", ()) or ():
        ast = getattr(coord, "ast", None)
        if ast is not None and not isinstance(ast, str):
            return coord
    return None


def _residual_score(plan: Any | None) -> float:
    if plan is None:
        return float("inf")
    algebra = getattr(plan, "algebra", None)
    vals = [
        float(getattr(algebra, "train_residual_rel", float("inf"))),
        float(getattr(algebra, "heldout_residual_rel", float("inf"))),
    ]
    finite = [v for v in vals if np.isfinite(v)]
    return float(max(finite)) if finite else float("inf")


def _stability_score(plan: Any | None) -> float:
    if plan is None:
        return 0.0
    algebra = getattr(plan, "algebra", None)
    angles = tuple(float(v) for v in tuple(getattr(algebra, "bootstrap_principal_angles", ()) or ()))
    finite = [v for v in angles if np.isfinite(v)]
    return float(max(finite)) if finite else 0.0


def _evidence(plan: Any | None, coord: Any | None) -> dict[str, Any]:
    algebra = getattr(plan, "algebra", None) if plan is not None else None
    return {
        "reduction_status": str(getattr(plan, "status", "")) if plan is not None else "",
        "reduction_reason": str(getattr(plan, "reason", "")) if plan is not None else "",
        "algebra_promotable": bool(getattr(algebra, "promotable", False)),
        "generic_orbit_rank": int(getattr(plan, "generic_orbit_rank", 0)) if plan is not None else 0,
        "quotient_codimension": int(getattr(plan, "quotient_codimension", 0)) if plan is not None else 0,
        "coordinate_name": str(getattr(coord, "name", "")) if coord is not None else "",
        "coordinate_kind": str(getattr(coord, "kind", "")) if coord is not None else "",
        "raw_support": [int(v) for v in tuple(getattr(coord, "raw_var_idxs", ()) or getattr(coord, "raw_support", ()) or ())] if coord is not None else [],
        "singular_strata": list(getattr(plan, "singular_strata", ()) or ()) if plan is not None else [],
        "normal_form_kind": str(getattr(getattr(plan, "normal_form", None), "kind", "")) if plan is not None else "",
    }


__all__ = [
    "ReductionPromotionDecision",
    "evaluate_reduction_promotion",
]
