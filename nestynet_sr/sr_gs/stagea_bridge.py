# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Bridge generalized-symmetry witnesses into Stage-A compound proposals."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nestynet_sr.sr_core import build_linear_ast, build_radial_r2_ast
from nestynet_sr.sr_core.ast_simplify import SimplifyOptions, node_count, simplify_ast, stable_ast_key
from nestynet_sr.sr_core.bridges import AddNode, ConstNode, MulNode, PowNode, Var, ast_to_human_readable, get_input_exprs
from nestynet_sr.sr_core.carrier_units import (
    CARRIER_INTERNAL_UNITS_INVALID,
    mark_inner_coordinate_metadata,
)

from .affine_algebra import discover_affine_algebra
from .charts import resolve_charts, snap_log_chart_algebra
from .config import GeneralizedSymmetryConfig
from .generators import GeneratorSpec, discover_generator_specs, summarize_specs
from .reporting import record_stagea_event
from .pi_bridge import stageA_unit_torus_pi_proposals
from .promotion import evaluate_reduction_promotion
from .quotient import compile_reduction_plan, compose_reduction_plan_with_inputs
from .unit_torus import projective_exponent_key



def _stagea_simplify_options(cfg: GeneralizedSymmetryConfig) -> SimplifyOptions:
    return SimplifyOptions(
        enabled=bool(getattr(cfg, "ast_simplify", False)),
        level=str(getattr(cfg, "ast_simplify_level", "safe") or "safe"),
        domain_policy=str(getattr(cfg, "ast_simplify_domain_policy", "strict") or "strict"),
        context="stagea_invariant",
        max_passes=int(getattr(cfg, "ast_simplify_max_passes", 12)),
        trace=bool(getattr(cfg, "ast_simplify_trace", False)),
        fail_closed=True,
    )


def _prefer_stagea_proposal(old: tuple, new: tuple) -> bool:
    try:
        if float(new[2]) > float(old[2]) + 1.0e-12:
            return True
        if abs(float(new[2]) - float(old[2])) <= 1.0e-12:
            return node_count(new[1]) < node_count(old[1])
    except Exception:
        pass
    return False


def _mark_promoted_carrier_roles(proposals, *, units_spec=None):
    """Attach the reusable inner-coordinate role to every promoted GS form."""

    out = []
    for proposal in list(proposals or ()):
        if len(proposal) < 5 or not isinstance(proposal[4], dict):
            out.append(proposal)
            continue
        meta = dict(proposal[4])
        if str(meta.get("kind", "")) != "gs_promoted_reduction":
            out.append(proposal)
            continue
        carrier_dim = None
        carrier_certified = units_spec is None
        if units_spec is not None:
            try:
                from nestynet_sr.sr_core.units import eval_analytic_expr_dim

                carrier_dim = eval_analytic_expr_dim(
                    proposal[1],
                    units_spec.x_dims,
                )
                carrier_certified = carrier_dim is not None
            except Exception:
                carrier_certified = False
        meta = mark_inner_coordinate_metadata(
            meta,
            source=str(meta.get("source", "generalized_symmetry") or "generalized_symmetry"),
            certified=carrier_certified,
        )
        if units_spec is not None:
            if carrier_dim is not None:
                meta["carrier_dim"] = [float(value) for value in carrier_dim]
            else:
                meta["carrier_unit_diagnostic"] = CARRIER_INTERNAL_UNITS_INVALID
            meta["target_dim"] = [float(value) for value in units_spec.y_phi_dim]
        out.append((*proposal[:4], meta, *proposal[5:]))
    return out


def _to_numpy(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _eval_leaf_values(leaf, x_vals, device=None):
    if leaf is None:
        return None
    try:
        import torch

        x = torch.as_tensor(x_vals, dtype=torch.float64)
        if device is not None:
            x = x.to(device)
        with torch.no_grad():
            y = leaf(x)
        return y.detach().cpu().reshape(-1).numpy()
    except Exception:
        return None


def _eval_leaf_grad_hess(leaf, x_vals, device=None):
    """Analytic ``(grad, Hess)`` of the leaf via autograd, for warp discovery.

    The fixed charts need only gradients (passed in); the warp chart additionally
    needs the Hessian to test the pair-independent normalized-Hessian
    certificate, so we differentiate the torch-callable leaf twice here.  Returns
    ``None`` on any failure (the caller then skips the warp chart).
    """

    if leaf is None:
        return None
    try:
        import torch

        x = torch.as_tensor(x_vals, dtype=torch.float64)
        if device is not None:
            x = x.to(device)
        x = x.requires_grad_(True)
        y = leaf(x).reshape(-1)
        (g,) = torch.autograd.grad(y.sum(), x, create_graph=True)
        n = x.shape[1]
        rows = []
        for i in range(n):
            (row,) = torch.autograd.grad(g[:, i].sum(), x, create_graph=True, retain_graph=True)
            rows.append(row)
        hess = torch.stack(rows, dim=1)
        return g.detach().cpu().numpy(), hess.detach().cpu().numpy()
    except Exception:
        return None


def _mul_const(c: float, node):
    return MulNode(ConstNode(float(c)), node)


def _diff_squares(i: int, j: int):
    return AddNode(PowNode(Var(i), 2.0), _mul_const(-1.0, PowNode(Var(j), 2.0)))


def _linear_two(i: int, j: int, ci: float, cj: float):
    try:
        if abs(float(ci) - round(float(ci))) < 1.0e-9 and abs(float(cj) - round(float(cj))) < 1.0e-9:
            return build_linear_ast((int(i), int(j)), (int(round(float(ci))), int(round(float(cj)))))
    except Exception:
        pass
    return AddNode(_mul_const(float(ci), Var(int(i))), _mul_const(float(cj), Var(int(j))))


def _sum_two(i: int, j: int):
    return build_linear_ast((int(i), int(j)), (1, 1))


def _diff_two(i: int, j: int):
    return build_linear_ast((int(i), int(j)), (1, -1))


def _product_two(i: int, j: int):
    return MulNode(Var(int(i)), Var(int(j)))


def _ratio_two(i: int, j: int):
    return MulNode(Var(int(i)), PowNode(Var(int(j)), -1.0))


def _ast_for_generator(spec: GeneratorSpec):
    axes = tuple(int(a) for a in spec.axes)
    kind = str(spec.kind)
    # Named V2 families and V3 learned sparse-affine generators share these
    # canonical kind names when they imply a safe quotient coordinate.
    if kind in {"diagonal_plus", "learned_diagonal_translation"} and len(axes) == 2:
        return _diff_two(axes[0], axes[1])
    if kind in {"diagonal_minus", "learned_signed_translation"} and len(axes) == 2:
        return _sum_two(axes[0], axes[1])
    if kind in {"common_pair", "learned_common_scaling"} and len(axes) == 2:
        return _ratio_two(axes[0], axes[1])
    if kind in {"opposite_pair", "learned_opposite_scaling"} and len(axes) == 2:
        return _product_two(axes[0], axes[1])
    if kind in {"so2_pair", "learned_rotation"} and len(axes) == 2:
        return build_radial_r2_ast(axes)
    if kind in {"boost_pair", "learned_lorentz_boost"} and len(axes) == 2:
        return _diff_squares(axes[0], axes[1])
    if spec.family == "general_affine" and len(axes) == 2:
        if kind == "affine_translation_pair":
            coeffs = tuple(float(v) for v in (spec.xi_coeffs or (1.0, 1.0)))
            if len(coeffs) >= 2:
                # Generator b_i d_i + b_j d_j has invariant b_j*x_i - b_i*x_j.
                return _linear_two(axes[0], axes[1], coeffs[1], -coeffs[0])
        if kind == "affine_common_scaling_pair":
            return _ratio_two(axes[0], axes[1])
        if kind == "affine_opposite_scaling_pair":
            return _product_two(axes[0], axes[1])
        if kind == "affine_rotation_pair":
            return build_radial_r2_ast(axes)
        if kind == "affine_lorentz_pair":
            return _diff_squares(axes[0], axes[1])
    return None


def _pattern_for_generator(spec: GeneratorSpec, cols: Sequence[int]) -> tuple[int, ...] | None:
    cols_t = tuple(int(c) for c in cols)
    axes = set(int(a) for a in spec.axes)
    if len(axes) == 0:
        return None
    return tuple(1 if int(c) in axes else 0 for c in cols_t)


def _local_inputs_for_stagea_atom(atom, cols: Sequence[int], x_vals) -> tuple:
    try:
        inputs = tuple(get_input_exprs(atom)) if atom is not None else ()
        if inputs and len(inputs) == int(np.asarray(x_vals).shape[1]):
            return inputs
    except Exception:
        pass
    return tuple(Var(int(c)) for c in cols)


def _stagea_shadow_reduction_diagnostics(
    *,
    atom,
    x_vals,
    dydx_vals,
    y_vals,
    cols: Sequence[int],
    cfg: GeneralizedSymmetryConfig,
) -> list[dict[str, Any]]:
    """Compile global-affine reductions as audit/shadow diagnostics only."""

    _proposals, diagnostics = _stagea_reduction_promotion_bundle(
        atom=atom,
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        y_vals=y_vals,
        cols=cols,
        cfg=cfg,
        units_spec=None,
    )
    return diagnostics


def _shadow_reduction_stub_row(
    *,
    cols: Sequence[int],
    chart: str,
    promotion_state: str,
    promotion_reason: str,
) -> dict[str, Any]:
    return {
        "family": "generalized_symmetry",
        "kind": "shadow_reduction",
        "accepted": False,
        "used_for_proposal": False,
        "used_for_selection": False,
        "shadow_only": True,
        "active_candidate": False,
        "cols": tuple(int(c) for c in cols),
        "promotion_state": str(promotion_state),
        "promotion_reason": str(promotion_reason),
        "chart": str(chart),
    }


def _cross_chart_proposal_key(proposal: tuple, plan, cols: Sequence[int]) -> tuple | None:
    """Dedup key so the same monomial invariant is emitted only once per atom.

    The identity-chart common-scaling plan and the log-chart monomial plan can
    both describe ``x_i/x_j`` (up to a monotone recoding on the positive
    domain), so they share the ``("monomial", support, ray)`` key.  Other plan
    kinds return ``None`` (no cross-chart dedup).
    """

    try:
        pattern = tuple(int(v) for v in proposal[0])
        support = tuple(int(c) for c, v in zip(cols, pattern) if v)
        meta = proposal[4] if len(proposal) >= 5 else {}
        exps_key = meta.get("gs_monomial_exponents_key") if isinstance(meta, dict) else None
        if exps_key:
            return ("monomial", support, tuple(int(v) for v in exps_key))
        if str(getattr(plan, "reason", "")) == "common_diagonal_scaling":
            return ("monomial", support, (1, -1))
    except Exception:
        return None
    return None


def _stagea_reduction_promotion_bundle(
    *,
    atom,
    x_vals,
    dydx_vals,
    y_vals,
    cols: Sequence[int],
    cfg: GeneralizedSymmetryConfig,
    units_spec=None,
    leaf=None,
    device=None,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Compile global-affine reductions and promote only certified candidates.

    Runs one determining solve per configured chart.  With the default
    identity-only chart list this reproduces the single-solve behavior; the
    log chart additionally exposes monomial invariants (scaling symmetries)
    through the same certificate and promotion gates.
    """

    if not bool(getattr(cfg, "general_affine_active", lambda: False)()):
        return [], []
    proposals: list[tuple] = []
    diagnostics: list[dict[str, Any]] = []
    seen_keys: set[tuple] = set()
    max_proposals = max(1, int(getattr(cfg, "max_stagea_proposals", 12) or 12))
    calibrated = bool(getattr(cfg, "noise_calibrated_promotion_active", lambda: False)())
    calibrated_discover_kwargs: dict[str, Any] = {}
    if calibrated:
        calibrated_discover_kwargs = {
            "nullity_strategy": "spectral_gap",
            "min_spectral_gap": float(getattr(cfg, "noise_calibrated_min_spectral_gap", 10.0) or 10.0),
            "closure_tol": float(getattr(cfg, "noise_calibrated_closure_tol", 3.0e-2) or 3.0e-2),
            "bootstrap_angle_tol": float(getattr(cfg, "noise_calibrated_bootstrap_angle_tol", 0.10) or 0.10),
            "heldout_consistency_factor": float(getattr(cfg, "noise_calibrated_heldout_factor", 3.0) or 3.0),
        }
    for chart in resolve_charts(cfg):
        try:
            eligible, elig_reason = chart.eligibility(x_vals)
            if not eligible:
                diagnostics.append(
                    _shadow_reduction_stub_row(
                        cols=cols,
                        chart=chart.name,
                        promotion_state="skipped",
                        promotion_reason=f"chart_ineligible:{elig_reason}",
                    )
                )
                continue
            u_vals, grad_u = chart.transform(x_vals, dydx_vals)
            algebra = discover_affine_algebra(
                u_vals,
                y_vals,
                grad_u,
                heldout_fraction=0.25,
                bootstrap=(max(1, int(getattr(cfg, "noise_calibrated_bootstrap", 8) or 8)) if calibrated else 0),
                acceptance_residual_tol=max(1.0e-10, float(getattr(cfg, "residual_tol", 0.03))),
                **calibrated_discover_kwargs,
            )
            if chart.name in ("log", "reciprocal"):
                algebra, snap_report = snap_log_chart_algebra(
                    algebra,
                    grad_u=grad_u,
                    max_denominator=int(getattr(cfg, "general_affine_chart_snap_denominator", 4) or 4),
                    residual_tol=float(getattr(cfg, "general_affine_promotion_residual_tol", 1.0e-8) or 1.0e-8),
                    calibration_factor=(float(getattr(cfg, "noise_calibrated_snap_factor", 3.0) or 3.0) if calibrated else None),
                    chart_name=chart.name,
                )
                if algebra is None:
                    row = _shadow_reduction_stub_row(
                        cols=cols,
                        chart=chart.name,
                        promotion_state="rejected",
                        promotion_reason=f"chart_snap_rejected:{snap_report.get('reason', 'unknown')}",
                    )
                    row["chart_snap"] = dict(snap_report)
                    diagnostics.append(row)
                    continue
            plan = compile_reduction_plan(algebra)
            local_inputs = _local_inputs_for_stagea_atom(atom, cols, x_vals)
            raw_plan = compose_reduction_plan_with_inputs(plan, local_inputs)
            decision = evaluate_reduction_promotion(raw_plan, cfg)
            proposal = _proposal_from_promoted_reduction(
                raw_plan, decision, cols=cols, cfg=cfg, units_spec=units_spec, chart=chart.name
            )
            cross_chart_duplicate = False
            proposal_capped = False
            if proposal is not None:
                key = _cross_chart_proposal_key(proposal, plan, cols)
                if key is not None and key in seen_keys:
                    cross_chart_duplicate = True
                    proposal = None
                elif len(proposals) >= max_proposals:
                    proposal_capped = True
                    proposal = None
                elif key is not None:
                    seen_keys.add(key)
            promoted = proposal is not None
            row = {
                "family": "generalized_symmetry",
                "kind": "shadow_reduction",
                "accepted": bool(getattr(plan, "status", "") == "compiled"),
                "used_for_proposal": bool(promoted),
                "used_for_selection": bool(promoted),
                "shadow_only": not bool(promoted),
                "active_candidate": bool(promoted),
                "cols": tuple(int(c) for c in cols),
                "promotion_state": decision.state,
                "promotion_reason": decision.reason,
                "promotion": decision.to_report(),
                "reduction": raw_plan.to_report(),
                "algebra": algebra.to_report(),
                "chart": str(chart.name),
                "noise_calibrated": bool(calibrated),
            }
            if cross_chart_duplicate:
                row["cross_chart_duplicate"] = True
            if proposal_capped:
                row["proposal_capped"] = True
            diagnostics.append(row)
            if proposal is not None:
                proposals.append(proposal)
        except Exception as exc:
            row = _shadow_reduction_stub_row(
                cols=cols,
                chart=chart.name,
                promotion_state="rejected",
                promotion_reason=f"{type(exc).__name__}: {exc}",
            )
            row["reason"] = f"{type(exc).__name__}: {exc}"
            diagnostics.append(row)

    # Discovered-warp chart (gated by "warp" in the same --gs-charts list): learns
    # the per-axis warp that linearizes a generalized-additive symmetry, then
    # emits the recovered coordinate through the standard promoted-reduction
    # contract (so cross-chart dedup and shared-bank recursion both apply).
    if "warp" in cfg.general_affine_chart_names():
        try:
            wprop, wrow = _warp_chart_proposal(
                atom=atom, leaf=leaf, x_vals=x_vals, dydx_vals=dydx_vals,
                y_vals=y_vals, cols=cols, cfg=cfg, device=device, units_spec=units_spec,
            )
            if wprop is not None:
                key = _cross_chart_proposal_key(wprop, None, cols)
                if key is not None and key in seen_keys:
                    wrow.update(used_for_proposal=False, shadow_only=True, cross_chart_duplicate=True)
                elif len(proposals) >= max_proposals:
                    wrow.update(used_for_proposal=False, shadow_only=True, proposal_capped=True)
                else:
                    if key is not None:
                        seen_keys.add(key)
                    proposals.append(wprop)
            diagnostics.append(wrow)
        except Exception as exc:
            row = _shadow_reduction_stub_row(
                cols=cols, chart="warp", promotion_state="rejected",
                promotion_reason=f"{type(exc).__name__}: {exc}",
            )
            row["reason"] = f"{type(exc).__name__}: {exc}"
            diagnostics.append(row)
    return proposals, diagnostics


def _proposal_from_promoted_reduction(
    plan,
    decision,
    *,
    cols: Sequence[int],
    cfg: GeneralizedSymmetryConfig,
    units_spec=None,
    chart: str = "identity",
):
    if not bool(getattr(decision, "accepted", False)):
        return None
    coord = None
    for item in getattr(plan, "invariant_coordinates", ()) or ():
        ast = getattr(item, "ast", None)
        if ast is not None and not isinstance(ast, str):
            coord = item
            break
    if coord is None:
        return None
    z_ast = getattr(coord, "ast", None)
    pattern = _pattern_for_coordinate(coord, cols)
    if pattern is None or not any(int(v) != 0 for v in pattern):
        return None
    simp_stats = None
    if bool(getattr(cfg, "ast_simplify", False)):
        z_ast, simp_stats = simplify_ast(z_ast, _stagea_simplify_options(cfg), units_spec=units_spec)
    promotion_report = decision.to_report()
    meta = {
        "kind": "gs_promoted_reduction",
        "source": "generalized_symmetry",
        "gs_source_family": "general_affine",
        "gs_family": "general_affine",
        "gs_kind": str(getattr(plan, "reason", "")),
        "gs_promotion_state": str(getattr(decision, "state", "")),
        "gs_promotion_reason": str(getattr(decision, "reason", "")),
        "gs_promotion": promotion_report,
        "gs_reduction": plan.to_report(),
        "gs_coordinate_name": str(getattr(coord, "name", "")),
        "gs_coordinate_kind": str(getattr(coord, "kind", "")),
        "gs_confidence": float(getattr(decision, "confidence", 0.0)),
        "gs_evidence": dict(getattr(decision, "evidence", {}) or {}),
        "gs_chart": str(chart),
    }
    coord_prov = dict(getattr(coord, "provenance", {}) or {})
    if str(getattr(coord, "kind", "")) == "monomial" and coord_prov.get("exponents"):
        exponents = tuple(int(v) for v in coord_prov["exponents"])
        meta["gs_monomial_exponents"] = exponents
        try:
            meta["gs_monomial_exponents_key"] = tuple(int(v) for v in projective_exponent_key(exponents))
        except Exception:
            meta["gs_monomial_exponents_key"] = exponents
    if coord_prov.get("covector") is not None:
        try:
            meta["gs_linear_covector"] = tuple(float(v) for v in coord_prov["covector"])
        except Exception:
            pass
    if simp_stats is not None:
        meta["ast_simplify"] = simp_stats.to_dict()
    try:
        meta["z_human"] = ast_to_human_readable(z_ast)
    except Exception:
        pass
    return (pattern, z_ast, float(getattr(decision, "confidence", 0.0)), None, meta)


def _pattern_for_coordinate(coord, cols: Sequence[int]) -> tuple[int, ...] | None:
    support = set(int(v) for v in tuple(getattr(coord, "raw_var_idxs", ()) or getattr(coord, "raw_support", ()) or ()))
    if not support:
        return None
    return tuple(1 if int(c) in support else 0 for c in cols)


def _proposal_from_warp(cert, cols: Sequence[int], *, cfg, units_spec=None):
    """Promoted-reduction proposal from a validated warp-discovery certificate."""

    from .warp_discovery import warp_coordinate_ast

    built = warp_coordinate_ast(cert.warps, list(cert.evidence.get("covector") or []), cols)
    if built is None:
        return None
    z_ast, support, covector = built
    pattern = tuple(1 if int(c) in set(support) else 0 for c in cols)
    if not any(pattern):
        return None
    simp_stats = None
    if bool(getattr(cfg, "ast_simplify", False)):
        z_ast, simp_stats = simplify_ast(z_ast, _stagea_simplify_options(cfg), units_spec=units_spec)
    conf = float(np.clip(1.0 - float(cert.warp_validation_residual or 0.0), 0.0, 1.0))
    kinds = tuple(w.kind for w in cert.warps)
    meta = {
        "kind": "gs_promoted_reduction",
        "source": "generalized_symmetry",
        "gs_source_family": "warp_discovery",
        "gs_family": "warp_discovery",
        "gs_kind": "warp_linear_invariant",
        "gs_promotion_state": "promoted",
        "gs_promotion_reason": str(cert.reason),
        "gs_coordinate_kind": "warp_linear",
        "gs_confidence": conf,
        "gs_chart": "warp",
        "gs_warp_kinds": kinds,
        "gs_warp_exponents": tuple(w.exponent for w in cert.warps),
        "gs_pair_consistency": float(cert.pair_consistency),
        "gs_warp_validation_residual": float(cert.warp_validation_residual or 0.0),
        "gs_linear_covector": tuple(float(v) for v in covector),
    }
    # Cross-chart dedup: an all-log warp is a log-chart monomial in disguise
    # (sum c_i log x_i = log(prod x_i^c_i)); share the monomial key.
    participating = [(k, c) for k, c in zip(kinds, covector) if abs(float(c)) > 1.0e-6]
    if participating and all(k == "log" for k, _c in participating):
        rounded = [round(float(c)) for c in covector]
        if all(abs(float(c) - r) < 1.0e-6 for c, r in zip(covector, rounded)) and any(rounded):
            try:
                meta["gs_monomial_exponents_key"] = tuple(int(v) for v in projective_exponent_key(tuple(rounded)))
            except Exception:
                pass
    if simp_stats is not None:
        meta["ast_simplify"] = simp_stats.to_dict()
    try:
        meta["z_human"] = ast_to_human_readable(z_ast)
    except Exception:
        pass
    return (pattern, z_ast, conf, None, meta)


def _warp_chart_proposal(
    *, atom, leaf, x_vals, dydx_vals, y_vals, cols, cfg, device=None, units_spec=None
):
    """Run warp discovery as a chart; return ``(proposal_or_None, diag_row)``.

    Warp discovery *discovers* the per-axis coordinate warp (from the
    pair-independent normalized Hessian) that linearizes a generalized-additive
    symmetry, rather than enumerating a fixed chart.  It needs the leaf Hessian
    and >=3 variables for the certificate to be non-vacuous.
    """

    from .warp_discovery import discover_warp

    base = {
        "family": "generalized_symmetry",
        "kind": "shadow_reduction",
        "chart": "warp",
        "cols": tuple(int(c) for c in cols),
        "accepted": False,
        "shadow_only": True,
        "used_for_proposal": False,
    }
    gh = _eval_leaf_grad_hess(leaf, x_vals, device)
    if gh is None:
        base.update(promotion_state="skipped", promotion_reason="warp_grad_hess_unavailable")
        return None, base
    grad, hess = gh
    if grad.ndim != 2 or grad.shape[1] < 3:
        base.update(promotion_state="skipped", promotion_reason="warp_needs_3plus_vars")
        return None, base
    cert = discover_warp(x_vals, grad, hess)
    base["gs_pair_consistency"] = float(cert.pair_consistency)
    base["interacting_pairs"] = [list(p) for p in cert.interacting_pairs]
    if not cert.is_separable_after_warp:
        base.update(promotion_state="skipped", promotion_reason="warp_not_pair_consistent")
        return None, base
    if cert.reason != "warp_recovered" or cert.warp_is_trivial:
        base.update(promotion_state="rejected", promotion_reason=f"warp_{cert.reason}")
        return None, base
    proposal = _proposal_from_warp(cert, cols, cfg=cfg, units_spec=units_spec)
    if proposal is None:
        base.update(promotion_state="rejected", promotion_reason="warp_ast_build_failed")
        return None, base
    base.update(
        kind="shadow_reduction",
        accepted=True,
        shadow_only=False,
        used_for_proposal=True,
        used_for_selection=True,
        active_candidate=True,
        promotion_state="promoted",
        promotion_reason=str(cert.reason),
        coordinate_human=cert.coordinate_human,
    )
    return proposal, base


def _discover_generalized_symmetry_proposal_tuples(
    *,
    atom,
    leaf,
    x_vals,
    dydx_vals,
    cols: Sequence[int],
    y_vals=None,
    device=None,
    cfg: GeneralizedSymmetryConfig | None = None,
    units_spec=None,
    _include_recursive: bool = False,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Internal GS discovery expressed in the legacy proposal tuple shape.

    This is the primitive discovery implementation used by the consumer-neutral
    carrier bank.  Public consumers should call ``discover_gs_carriers`` or the
    Stage-A adapter below rather than depending on this tuple-level helper.

    ``_include_recursive`` is retained for compatibility while the carrier bank
    owns the normal recursive expansion path.
    """

    cfg = cfg or GeneralizedSymmetryConfig()
    if not cfg.active():
        return [], []
    if y_vals is None:
        y_vals = _eval_leaf_values(leaf, x_vals, device=device)
    else:
        y_vals = _to_numpy(y_vals).reshape(-1)
    if y_vals is None:
        diag = [
            {
                "family": "generalized_symmetry",
                "kind": "stagea_leaf_eval",
                "accepted": False,
                "reason": "leaf_evaluation_failed",
            }
        ]
        return [], diag

    specs_all = discover_generator_specs(
        x_vals,
        y_vals,
        dydx_vals,
        cols=tuple(cols),
        cfg=cfg,
        include_rejected=bool(getattr(cfg, "report_rejected", True) or getattr(cfg, "automatic", lambda: False)()),
    )
    # Jet-level separability witnesses are handled inside the ordinary
    # separability checker, where the required normalized Hessian and gradient
    # tensors are already available.  This bridge is intentionally restricted to
    # quotient-coordinate proposals from point/affine generators.
    specs = [s for s in specs_all if bool(getattr(s, "accepted", True))]
    proposals = []
    proposal_by_key: dict[tuple, tuple] = {}
    simp_opts = _stagea_simplify_options(cfg) if bool(getattr(cfg, "ast_simplify", False)) else None
    for spec in specs:
        if spec.family == "general_affine":
            # Learned affine reductions are promoted through ReductionPlan
            # evidence below.  Do not bypass that gate with the older
            # generator-to-coordinate shortcut.
            continue
        if spec.kind == "axis":
            # Axis translations/scalings are useful diagnostics, but they do not
            # create a new scalar coordinate proposal here. Existing pruners and
            # homogeneity rules handle them downstream.
            continue
        if abs(float(spec.output_alpha)) > 1.0e-10 or abs(float(spec.output_beta)) > 1.0e-10:
            # Xf≈alpha+beta*f is an equivariance/prefactor hint rather than a
            # proof that f descends to the invariant alone.  Keep it in audit
            # diagnostics for later prefactor work, but do not emit a quotient
            # coordinate in this first conservative prototype.
            continue
        z_ast = _ast_for_generator(spec)
        if z_ast is None:
            continue
        simp_stats = None
        proposal_key = None
        if simp_opts is not None:
            z_ast, simp_stats = simplify_ast(z_ast, simp_opts, units_spec=units_spec)
            proposal_key = stable_ast_key(z_ast, ignore_tags=True, context="stagea_invariant")
        pattern = _pattern_for_generator(spec, cols)
        if pattern is None or not any(int(v) != 0 for v in pattern):
            continue
        meta = {
            "kind": f"gs_{spec.family}",
            "gs_family": spec.family,
            "gs_kind": spec.kind,
            "gs_axes": tuple(int(a) for a in spec.axes),
            "gs_generator_coeffs": tuple(float(v) for v in spec.xi_coeffs),
            "gs_output_alpha": float(spec.output_alpha),
            "gs_output_beta": float(spec.output_beta),
            "gs_residual_rel": float(spec.residual_rel),
            "gs_confidence": float(spec.confidence),
            "gs_invariant": spec.invariant,
            "gs_evidence": dict(spec.evidence or {}),
            "source": "generalized_symmetry",
            "gs_source_family": spec.family,
        }
        if simp_stats is not None:
            meta["ast_simplify"] = simp_stats.to_dict()
        try:
            meta["z_human"] = ast_to_human_readable(z_ast)
        except Exception:
            pass
        if cfg.proposing():
            proposal = (pattern, z_ast, float(spec.confidence), None, meta)
            if proposal_key is None:
                proposals.append(proposal)
            else:
                old = proposal_by_key.get(proposal_key)
                if old is None or _prefer_stagea_proposal(old, proposal):
                    proposal_by_key[proposal_key] = proposal
    if proposal_by_key:
        proposals.extend(proposal_by_key.values())
    diag = summarize_specs(specs_all)
    pi_proposals, pi_diag = stageA_unit_torus_pi_proposals(
        cols=tuple(int(c) for c in cols),
        units_spec=units_spec,
        cfg=cfg,
    )
    if pi_proposals:
        proposals.extend(pi_proposals)
    if pi_diag:
        diag.extend(pi_diag)
    reduction_proposals, reduction_diag = _stagea_reduction_promotion_bundle(
        atom=atom,
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        y_vals=y_vals,
        cols=cols,
        cfg=cfg,
        units_spec=units_spec,
        leaf=leaf,
        device=device,
    )
    if reduction_proposals:
        proposals.extend(reduction_proposals)
    if reduction_diag:
        diag.extend(reduction_diag)
    if bool(getattr(cfg, "pairwise_composition_active", lambda: False)()):
        try:
            from .pairwise_composition import compose_pairwise_witness_proposals, ray_key_from_covector

            composed_proposals, composed_diag = compose_pairwise_witness_proposals(
                specs,
                x_vals=x_vals,
                dydx_vals=dydx_vals,
                cols=cols,
                cfg=cfg,
            )

            # Dedup against rays the reduction bundle already promoted.
            def _reduction_ray_key(prop):
                meta_r = prop[4] if len(prop) >= 5 and isinstance(prop[4], dict) else {}
                mono = meta_r.get("gs_monomial_exponents_key")
                if mono:
                    return (tuple(prop[0]), ("monomial",) + tuple(int(v) for v in mono))
                cov = meta_r.get("gs_linear_covector")
                if cov is not None:
                    ray = meta_r.get("gs_linear_ray_key") or ray_key_from_covector(cov)
                    if ray:
                        return (tuple(prop[0]), ("linear",) + tuple(int(v) for v in ray))
                if str(meta_r.get("gs_coordinate_kind", "")) in {"radial", "quadratic_radius"}:
                    return (tuple(prop[0]), ("radial",))
                if str(meta_r.get("gs_coordinate_kind", "")) == "quadratic_form":
                    sig = tuple(int(v) for v in (meta_r.get("gs_quadratic_signature") or ()))
                    return (tuple(prop[0]), ("quadratic",) + sig)
                return None

            reduction_key_to_index: dict = {}
            for prop in reduction_proposals or ():
                key = _reduction_ray_key(prop)
                if key is not None:
                    for idx in range(len(proposals)):
                        if proposals[idx] is prop:
                            reduction_key_to_index[key] = idx
                            break
            seen_rays = set(reduction_key_to_index)
            for prop in composed_proposals:
                key = _reduction_ray_key(prop)
                if key is not None and key in seen_rays:
                    # Same ray from both routes. Monomial reductions are
                    # already integer-snapped — keep the richer reduction
                    # evidence. Linear reductions carry raw float covectors,
                    # so the composed integer coordinate is the cleaner
                    # Stage-A proposal: replace in place.
                    if key[1][0] == "linear" and key in reduction_key_to_index:
                        proposals[reduction_key_to_index[key]] = prop
                    continue
                if key is not None:
                    seen_rays.add(key)
                proposals.append(prop)
            if composed_diag:
                diag.extend(composed_diag)
        except Exception as exc:
            diag.append(
                {
                    "family": "generalized_symmetry",
                    "kind": "pairwise_composition",
                    "accepted": False,
                    "status": "rejected",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if (
        bool(_include_recursive)
        and bool(getattr(cfg, "recursive_composition_active", lambda: False)())
        and y_vals is not None
    ):
        try:
            from .recursive_composition import compose_recursive_coordinate_proposals

            promoted_first_level = [
                p for p in proposals
                if len(p) >= 5 and isinstance(p[4], dict) and p[4].get("kind") == "gs_promoted_reduction"
            ]
            if promoted_first_level:
                rec_proposals, rec_diag = compose_recursive_coordinate_proposals(
                    promoted_first_level,
                    x_vals=x_vals,
                    y_vals=y_vals,
                    dydx_vals=dydx_vals,
                    cols=cols,
                    cfg=cfg,
                )
                existing_ast = {repr(p[1]) for p in proposals if len(p) >= 2}
                for prop in rec_proposals:
                    if repr(prop[1]) in existing_ast:
                        continue
                    existing_ast.add(repr(prop[1]))
                    proposals.append(prop)
                if rec_diag:
                    diag.extend(rec_diag)
        except Exception as exc:
            diag.append(
                {
                    "family": "generalized_symmetry",
                    "kind": "recursive_composition",
                    "accepted": False,
                    "status": "rejected",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    proposals = _mark_promoted_carrier_roles(proposals, units_spec=units_spec)
    return proposals, diag


def stageA_generalized_symmetry_proposals(
    *,
    atom,
    leaf,
    x_vals,
    dydx_vals,
    cols: Sequence[int],
    y_vals=None,
    device=None,
    cfg: GeneralizedSymmetryConfig | None = None,
    units_spec=None,
) -> tuple[list[tuple], list[dict[str, Any]]]:
    """Return GS carriers through the ordinary Stage-A proposal contract.

    GS owns discovery, certification, and bounded recursive composition.  This
    adapter only converts the shared carrier records into the legacy compound
    tuple shape; Stage A retains sole authority to accept a trained rewrite.
    """

    from .carrier_bank import discover_gs_carriers

    cfg = cfg or GeneralizedSymmetryConfig()
    if not cfg.active():
        return [], []
    carriers, diag = discover_gs_carriers(
        atom=atom,
        leaf=leaf,
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        cols=cols,
        y_vals=y_vals,
        device=device,
        cfg=cfg,
        units_spec=units_spec,
    )
    proposals = [carrier.to_stagea_proposal() for carrier in carriers]
    context = {
        "mode": str(getattr(cfg, "mode", "propose")),
        "policy": str(getattr(cfg, "policy", "augment")),
    }
    if any(row.get("reason") == "leaf_evaluation_failed" for row in diag):
        context["status"] = "skipped"
    try:
        record_stagea_event(
            cols=tuple(int(c) for c in cols),
            diagnostics=diag,
            proposals=proposals,
            context=context,
        )
    except Exception:
        pass
    return proposals, diag
