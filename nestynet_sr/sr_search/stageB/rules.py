# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Stage B Rules: Rewrite rules for Stage B refinement.

This module contains the main rule classes:
- RuleMultiDNN: Multi-dimensional NN rewrites (generalized additivity, trapped variables, etc.)
- RulePolySplit: Polynomial splitting into additive components
- RuleSubtreeSeparability: SubtreeSeparability rewrites (under composite nodes)
- RuleUniNN: Univariate NN rewrites (scaling, Planck, exp-poly, trig, etc.)
- RuleOuterTransformSplitNN: Outer-transform fallback splits (log/sqrt separability)
- RuleNNLeafSeparability: Separability analysis for stubborn multivariate NN leaves (makes Stage B fully iterative)

Each rule implements:
- iter_targets(ctx): Return nodes that are candidates for this rule
- propose(ctx, target): Generate rewrite candidates for a target node
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AddNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    Var,
    _collect_var_idxs_from_node,
    clone_ast,
    clone_inputs,
    effective_arity,
    get_input_exprs,
    has_nontrivial_input,
    is_trivial_input,
    replace_atom_in_ast,
    trivial_input_position,
)
from nestynet_sr.sr_core.constants import (
    build_scalar_atom_from_variant as _build_scalar_atom_from_variant,
    make_unit_aware_scalar_atom as _make_unit_aware_scalar_atom,
    scalar_constant_variants as _scalar_constant_variants,
)
from nestynet_sr.sr_core.separability_math import (
    build_monomial_ast,
    check_coupled_leaf_ratio_from_derivs as check_coupled_leaf_ratio_from_derivs,
    check_ratio_invariance,
)
from nestynet_sr.sr_search import candidate_builders
from nestynet_sr.sr_search.candidate_builders import _build_atom_input_tensor
from nestynet_sr.sr_search.features import (
    QuadraticHint,
    TransformHint,
    detect_log_hessian_quadratic,
    detect_square_hessian_quadratic,
    discover_saturating_axes,
    probe_output_transforms,
)
from nestynet_sr.sr_search.monomial_peel_plan import split_clean_integer_powers

# Phase-4 template families (hint-driven)
from nestynet_sr.sr_search.template_library import (
    propose_exp_of_quadratic,
    propose_exp_poly_from_log_hint,
    propose_rational_linear,
    propose_sin_cos_from_inverse_hint,
    propose_sinc_family,
    propose_symexp_denom_family,
    propose_tanh_family,
    propose_trig_rational,
)
from nestynet_sr.sr_search.wrapper_policy import macro_arg_wrapper_policy

from .engine import (
    Candidate,
    StageBContext,
    StageBRule,
    atom_content_hash,
    candidate_pattern_name,
)
from .rules_common import (
    _HomogeneousGaugeTeacher as _HomogeneousGaugeTeacher,
    RuleR1OperatorCertificate as RuleR1OperatorCertificate,
    _effective_input_dims_for_atom,
    _stageB_noise_floor_raw,
    _stageB_noisy_rel_rms_threshold,
    _subtree_content_hash,
)
from .rules_compound import (
    RuleAffineDecomposition as RuleAffineDecomposition,
    RuleBarycentricCompound as RuleBarycentricCompound,
    RuleCompoundFunctionMacros as RuleCompoundFunctionMacros,
    RuleCompoundPlanck as RuleCompoundPlanck,
    RuleLogExpCompound as RuleLogExpCompound,
    RuleMetricDistance as RuleMetricDistance,
    RuleMonomialPrefactorCompound as RuleMonomialPrefactorCompound,
    RuleNonlinearSubstitution as RuleNonlinearSubstitution,
)
from .rules_nn_leaf import (
    RuleNNLeafSeparability as RuleNNLeafSeparability,
    RuleOuterTransformSplitNN as RuleOuterTransformSplitNN,
    _OuterTransformedSubtreeModel as _OuterTransformedSubtreeModel,
    _eval_subtree_with_leaf_map as _eval_subtree_with_leaf_map,
    _flatten_mul as _flatten_mul,
    _iter_add_nodes as _iter_add_nodes,
    _iter_mul_nodes as _iter_mul_nodes,
    _rebuild_mul as _rebuild_mul,
    _replace_node_in_ast as _replace_node_in_ast,
    _vars_in_subtree_simple as _vars_in_subtree_simple,
)
from .rules_phase_trig import (
    RuleInverseTrigOuterClosure as RuleInverseTrigOuterClosure,
    RuleInverseTrigRationalOuterClosure as RuleInverseTrigRationalOuterClosure,
    RuleLastHardTrigPower1D as RuleLastHardTrigPower1D,
    RuleLastHardTrigSquare1D as RuleLastHardTrigSquare1D,
    RulePhaseContextTrigClosure as RulePhaseContextTrigClosure,
    RulePhaseHintReciprocalTrigPower as RulePhaseHintReciprocalTrigPower,
    RulePhaseHintTrigClosure as RulePhaseHintTrigClosure,
    _last_hard_atom_context as _last_hard_atom_context,
    _stageB_phase_hints_for_atom as _stageB_phase_hints_for_atom,
)
from .rules_preconditioner import RulePreconditionerFallbackNN as RulePreconditionerFallbackNN
from .rules_problem import RuleNonsenseUnitsZeroPrune as RuleNonsenseUnitsZeroPrune
from .rules_univariate import (
    RuleMonomialPeelPriority as RuleMonomialPeelPriority,
    RuleUniNN as RuleUniNN,
    _build_fixed_trig_factor_candidates as _build_fixed_trig_factor_candidates,
    _build_one_minus_cos_over_z2_candidates as _build_one_minus_cos_over_z2_candidates,
    _build_sparse_factor_1d_candidates as _build_sparse_factor_1d_candidates,
    _build_trig_feature_linear_candidates as _build_trig_feature_linear_candidates,
    _prepare_univariate_units_probe as _prepare_univariate_units_probe,
    _stageB_target_raw_teacher_data as _stageB_target_raw_teacher_data,
)
from .rules_gauge_homogeneity import (
    RuleAdditiveGaugeTransfer as RuleAdditiveGaugeTransfer,
    RuleCommonPrefactor as RuleCommonPrefactor,
    RuleCounterfactorAddSplitNN as RuleCounterfactorAddSplitNN,
    RuleCountertermMulSplitNN as RuleCountertermMulSplitNN,
    RuleCoupledLeafRatio as RuleCoupledLeafRatio,
    RuleHomogeneityPeel as RuleHomogeneityPeel,
    RuleMultiplicativeHomogeneityTransfer as RuleMultiplicativeHomogeneityTransfer,
    RuleOverlapCountertermPeelNN as RuleOverlapCountertermPeelNN,
    RuleOverlapPrefactorPeelNN as RuleOverlapPrefactorPeelNN,
    RuleProductHomogeneity as RuleProductHomogeneity,
    RuleRatioInvariance as RuleRatioInvariance,
    _check_ratio_invariance_on_leaf_inputs as _check_ratio_invariance_on_leaf_inputs,
    _eval_transfer_basis_expr as _eval_transfer_basis_expr,
    _gauge_transfer_term_view as _gauge_transfer_term_view,
    _has_nn_atom as _has_nn_atom,
    _homogeneity_product_ratio_units_ok as _homogeneity_product_ratio_units_ok,
    _homogeneity_ratio_units_ok as _homogeneity_ratio_units_ok,
    _homogeneous_reuses_override as _homogeneous_reuses_override,
    _homogeneous_scope_units_ok as _homogeneous_scope_units_ok,
    _is_one_like as _is_one_like,
    _make_gauge_private_atom as _make_gauge_private_atom,
    _make_homogeneity_peel_init_fn as _make_homogeneity_peel_init_fn,
    _make_homogeneity_peel_values_init_fn as _make_homogeneity_peel_values_init_fn,
    _multiply_dims as _multiply_dims,
    _multiply_exprs as _multiply_exprs,
    _power_factor_for_var as _power_factor_for_var,
    _raw_vars_for_atom as _raw_vars_for_atom,
    _same_gauge_prefactor as _same_gauge_prefactor,
    _select_inputs_subset as _select_inputs_subset,
    _shared_input_exprs_for_atom as _shared_input_exprs_for_atom,
    _transfer_feature_domain_ok_frac as _transfer_feature_domain_ok_frac,
    _univariate_collapse_score as _univariate_collapse_score,
    meta_prefactor_label as meta_prefactor_label,
)


_UNARY_AST_NODES = (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode)


def _powerlaw_logfit(log_X: np.ndarray, log_F: np.ndarray, eps: float = 1e-12):
    """OLS of log_F on [log_X, 1]. Returns (coeffs, r2, ss_tot) or None."""
    ones = np.ones((log_X.shape[0], 1), dtype=np.float64)
    A = np.concatenate([log_X, ones], axis=1)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, log_F, rcond=None)
    except np.linalg.LinAlgError:
        return None
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        predicted = A @ coeffs
    if not np.all(np.isfinite(predicted)):
        return None
    ss_res = np.sum((log_F - predicted) ** 2)
    ss_tot = np.sum((log_F - np.mean(log_F)) ** 2)
    if ss_tot < eps:
        return None
    return coeffs, 1.0 - ss_res / ss_tot, ss_tot


def _powerlaw_probe(
    X_pos: np.ndarray,
    F_pos: np.ndarray,
    *,
    r2_gate: float = 0.98,
    trim_q: float = 0.30,
    min_trim_points: int = 200,
):
    """Two-stage power-law probe: full-set log fit, then a magnitude-trimmed
    retry when the full fit misses the gate.

    Additive teacher pollution from an imperfect upstream split is amplified
    in log space at small |F| and can sink the full-set R² even when the
    power-law structure is exact on the bulk of the range. Refitting on the
    top (1-trim_q) fraction by |F| restores the measurement without loosening
    the gate itself.

    Returns dict with coeffs/log arrays of the accepted fit, r2_full,
    r2_trim (or None), subset tag, and passed flag; or None if no fit is
    possible at all.
    """
    log_X = np.log(X_pos)
    log_F = np.log(F_pos)
    fit_full = _powerlaw_logfit(log_X, log_F)
    if fit_full is None:
        return None
    coeffs, r2_full, ss_tot = fit_full
    out = {
        "coeffs": coeffs,
        "log_X": log_X,
        "log_F": log_F,
        "ss_tot": ss_tot,
        "r2_full": float(r2_full),
        "r2_trim": None,
        "subset": "full",
        "passed": bool(r2_full >= r2_gate),
    }
    if out["passed"]:
        return out
    thr = np.quantile(F_pos, trim_q)
    m_trim = F_pos >= thr
    if int(m_trim.sum()) < min_trim_points:
        return out
    fit_trim = _powerlaw_logfit(log_X[m_trim], log_F[m_trim])
    if fit_trim is None:
        return out
    coeffs_t, r2_trim, ss_tot_t = fit_trim
    out["r2_trim"] = float(r2_trim)
    if r2_trim >= r2_gate:
        out.update(
            {
                "coeffs": coeffs_t,
                "log_X": log_X[m_trim],
                "log_F": log_F[m_trim],
                "ss_tot": ss_tot_t,
                "subset": f"trim{int(round((1.0 - trim_q) * 100))}",
                "passed": True,
            }
        )
    return out


def _ctx_pattern_disabled(ctx: StageBContext, name: str) -> bool:
    checker = getattr(ctx, "is_pattern_disabled", None)
    if checker is None:
        return False
    return bool(checker(name))



try:
    from nestynet_sr.sr_core.units import (
        _dim_in_rational_span as _dim_in_rational_span,
        is_dimless as _is_dimless,
        scale_dim as _scale_dim,
    )
except Exception:  # pragma: no cover
    _dim_in_rational_span = None  # type: ignore
    _is_dimless = None  # type: ignore
    _scale_dim = None  # type: ignore

# Import helper functions from helpers.py
# These are pure functions used by the rules (no circular import issues)
from .helpers import (
    _build_additive_poly_split_candidate,
    _build_affine_split_candidate,
    _build_counterfactor_add_split_candidate as _build_counterfactor_add_split_candidate,
    _build_counterterm_mul_split_candidate as _build_counterterm_mul_split_candidate,
    _build_coupled_ratio_candidate as _build_coupled_ratio_candidate,
    _build_inv_poly_candidates,
    _build_last_hard_ratio_candidates,
    _build_log_poly_candidate,
    _build_log_ratpoly_candidate,
    _build_overlap_counterterm_peel_candidates as _build_overlap_counterterm_peel_candidates,
    _build_overlap_prefactor_peel_candidates as _build_overlap_prefactor_peel_candidates,
    _build_power_exp_rat_candidate,
    _build_product_homogeneity_candidate as _build_product_homogeneity_candidate,
    _build_pure_exp_rat_candidate,
    _build_quadratic_poly_candidate,
    _build_ratio_invariance_candidate as _build_ratio_invariance_candidate,
    _build_ratpoly_1d_candidates,
    _build_ratpoly_candidates,
    _build_sqrt_poly_candidate,
    _build_sqrt_ratpoly_candidate,
    _build_subtree_separability_candidate,
    _collect_all_atoms,
    _collect_multivariate_nn_atoms,
    _collect_multivariate_poly_atoms,
    _collect_subtree_separability_targets,
    _collect_univariate_nn_atoms,
    _compute_trapped_factorization,
    _fit_poly_1d_trapped,
    _leaf_coeff_param,
    _make_multid_trig_pair_rewrite,
    _make_multid_trig_rewrite,
    _poly_like_core,
    _poly_zero_and_set,
    _set_constant_leaf_value,
    _probe_genadd_for_nn_leaf,
    _probe_trapped_for_nn_leaf,
    _SubtreeModel,
    build_atom_to_leaf_map,
    run_subtree_separability,
)

# ---------------------------------------------------------------------------
# Trig-diff amplitude + phase estimation helper
# ---------------------------------------------------------------------------


def _estimate_trig_amp_phase(
    target: AtomNode,
    st: Any,
    ctx: StageBContext,
    trig_spec: Any,
    axis: int,
    partner: int,
) -> Tuple[float, float]:
    """
    Fit nn_out ≈ A*cos(wΔ) + B*sin(wΔ) + C via least squares, return (R, φ).

    Where:
        Δ = x_axis - x_partner
        R = sqrt(A^2 + B^2)  (amplitude)
        φ = atan2(-B, A)     (phase)

    This gives: R*cos(wΔ + φ) = A*cos(wΔ) + B*sin(wΔ)

    Returns (0.0, 0.0) on failure.
    """
    estimated_amp = 0.0
    estimated_phase = 0.0

    # trig-diff is defined in terms of *global* axes (x_axis - x_partner).
    # For nontrivial input expressions, the leaf input is [z, extras...] and
    # there is no consistent mapping from global axis -> leaf coordinate here.
    if has_nontrivial_input(target):
        return estimated_amp, estimated_phase

    try:
        target_leaf = st.reuse.get(target.tag)
        if target_leaf is None:
            if ctx.verbose:
                print(f"[Stage B] trig-diff: target.tag={target.tag} not found in st.reuse")
            return estimated_amp, estimated_phase

        # Get a batch of data
        for batch in ctx.train_loader_probe:
            xb = batch[0].to(device=ctx.device, dtype=ctx.dtype)
            break

        # Extract the relevant columns for this leaf
        cols = list(target.var_idxs)
        x_sub = xb[:, cols]  # non-compound only (guarded above)

        # Evaluate the NN leaf
        with torch.no_grad():
            nn_out = target_leaf(x_sub)
            if nn_out.dim() == 2:
                nn_out = nn_out[:, 0]
            else:
                nn_out = nn_out.view(-1)

        # Compute cos(arg) and sin(arg) where arg = omega * (x_axis - x_partner)
        w = float(trig_spec.omega)
        axis_local = cols.index(axis)
        partner_local = cols.index(partner)
        arg_vals = w * (x_sub[:, axis_local] - x_sub[:, partner_local])
        cos_vals = torch.cos(arg_vals)
        sin_vals = torch.sin(arg_vals)
        ones = torch.ones_like(cos_vals)

        # Least squares: [cos, sin, 1] @ [A, B, C]^T = nn_out
        # Using normal equations: (X^T X)^{-1} X^T y
        design = torch.stack([cos_vals, sin_vals, ones], dim=1)  # [N, 3]
        # Solve via lstsq for robustness
        coeffs = torch.linalg.lstsq(design, nn_out.unsqueeze(1)).solution  # [3, 1]
        A = float(coeffs[0, 0])
        B = float(coeffs[1, 0])
        C = float(coeffs[2, 0])

        # Convert to amplitude + phase: R*cos(wΔ + φ) = A*cos(wΔ) + B*sin(wΔ)
        # Using: R*cos(wΔ + φ) = R*cos(φ)*cos(wΔ) - R*sin(φ)*sin(wΔ)
        # So: A = R*cos(φ), B = -R*sin(φ)
        R = math.sqrt(A * A + B * B)
        phi = math.atan2(-B, A)  # phase shift

        if ctx.verbose:
            print(
                f"[Stage B] trig-diff amp+phase estimation: "
                f"A={A:.4f}, B={B:.4f}, C={C:.4f}, R={R:.4f}, phi={phi:.4f}, omega={w:.3f}"
            )

        if math.isfinite(R) and abs(R) < 1e6:
            estimated_amp = R
            estimated_phase = phi

    except Exception as e:
        if ctx.verbose:
            print(f"[Stage B] trig-diff amplitude+phase estimation failed: {e}")

    return estimated_amp, estimated_phase


def _poly_leaf_homogeneous_for_dims(
    input_dims: List[Tuple[Any, ...]],
    units_spec: Any,
) -> bool:
    """Return whether a PolyLeaf over these inputs needs a homogeneous basis.

    The AST-level units checker only accepts a unitful PolyLeaf when all of its
    inputs are commensurate and the basis is homogeneous.  This helper is
    intentionally per-leaf: a target NN may also contain dimensionless trig
    axes that are not inputs to a particular offset/amplitude polynomial.
    """
    if units_spec is None or not input_dims:
        return False
    dimless = tuple(0 for _ in units_spec.unit_system.base)
    dims = [tuple(d) for d in input_dims]
    d0 = dims[0]
    if d0 == dimless:
        return False
    return all(d == d0 for d in dims)


def _poly_leaf_homogeneous_for_raw_var_idxs(
    var_idxs: List[int],
    units_spec: Any,
) -> bool:
    """Per-leaf homogeneous-basis decision for raw-variable PolyLeafs."""
    if units_spec is None or not var_idxs:
        return False
    dims: List[Tuple[Any, ...]] = []
    for idx in var_idxs:
        j = int(idx)
        if j < 0 or j >= len(units_spec.x_dims):
            return False
        dims.append(tuple(units_spec.x_dims[j]))
    return _poly_leaf_homogeneous_for_dims(dims, units_spec)




class RuleMultiDNN(StageBRule):
    """
    Rule for multi-dimensional NN atom rewrites.

    This rule identifies multi-dimensional NN leaves and proposes various analytic
    rewrites based on:
    - Generalized additivity probes (sin/cos/exp compositions)
    - Trapped-variable patterns (multiplicative factorization, sin² ratios)
    - Rational polynomials, sqrt compositions, exponential branches
    - Trigonometric compositions (if trig axes detected)

    Pattern labels: trapped_sin_ratio, trapped_mult, genadd_sin, genadd_cos, genadd_exp,
                    sqrt_ratpoly, sqrt_poly, quad_poly, exp_branch, pure_exp_branch,
                    log_ratpoly, trig_comp
    """

    name = "multid_nn"

    def __init__(self, factorized_search_rule=None):
        self.factorized_search_rule = factorized_search_rule

    def iter_targets(self, ctx: StageBContext):
        """Return all multi-dimensional NN atoms in the current AST."""
        return _collect_multivariate_nn_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        """
        Generate rewrite candidates for a multi-dimensional NN target.

        Args:
            ctx: Stage B context with state, data, hyperparameters
            target: Multi-dimensional NN atom node

        Returns:
            List of Candidate rewrites (may be empty)
        """
        if not isinstance(target, AtomNode) or str(target.kind).lower() != "nn":
            return []

        st = ctx.state
        reuse = st.reuse if isinstance(st.reuse, dict) else {}

        # Guard: if this bivariate NN is *strongly* homogeneous (incl. ratio-invariant),
        # prefer dedicated structural decompositions (homogeneity_peel / ratio_invariance)
        # over early MultiDNN 'collapse-to-Padé' rewrites (e.g. sqrt_ratpoly).
        # Accepting a high-order ratpoly too early can block later, cleaner decompositions.
        try:
            if effective_arity(target) == 2 and len(target.var_idxs) == 2:
                atom_to_leaf_h = build_atom_to_leaf_map(st.root, st.model)
                subtree_h = _SubtreeModel(root=target, atom_to_leaf=atom_to_leaf_h)

                # Degree-k (k≠0) homogeneity: xi*df/dxi + xj*df/dxj ≈ k*f
                from nestynet_sr.sr_core.separability_math import check_homogeneous_degree

                hres = check_homogeneous_degree(
                    model=subtree_h,
                    datagen=ctx.train_loader_probe,
                    xi_idx=int(target.var_idxs[0]),
                    xj_idx=int(target.var_idxs[1]),
                    device=ctx.device,
                    dtype=ctx.dtype,
                    threshold=0.01,
                    n_points=2048,
                )

                # Degree-0 (ratio invariance): xi*df/dxi + xj*df/dxj ≈ 0
                rres = None
                if not getattr(hres, "ok", False):
                    try:
                        rres = check_ratio_invariance(
                            model=subtree_h,
                            datagen=ctx.train_loader_probe,
                            xi_idx=int(target.var_idxs[0]),
                            xj_idx=int(target.var_idxs[1]),
                            device=ctx.device,
                            dtype=ctx.dtype,
                            threshold=0.01,
                            n_points=2048,
                        )
                    except Exception:
                        rres = None

                if getattr(hres, "ok", False) or (rres is not None and getattr(rres, "ok", False)):
                    if getattr(hres, "ok", False):
                        ctx.log(
                            f"[Stage B] MultiDNN: strong homogeneity detected for NN vars={target.var_idxs} "
                            f"(k≈{float(getattr(hres, 'degree', 0.0)):.3g}, "
                            f"euler_residual={float(getattr(hres, 'euler_residual', 0.0)):.4f}); "
                            "deferring MultiDNN so homogeneity/ratio rules can decompose first."
                        )
                    else:
                        ctx.log(
                            f"[Stage B] MultiDNN: strong ratio-invariance detected for NN vars={target.var_idxs} "
                            f"(euler_score={float(getattr(rres, 'euler_score', 0.0)):.4f}); "
                            "deferring MultiDNN so ratio_invariance can decompose first."
                        )
                    return []
        except Exception:
            pass

        # Shared wrapper-policy gate for Stage-B trig candidates.
        # Use the same policy layer as compound-function macros so a global
        # "disable trig" setting applies consistently across Stage B.
        trig_policy = None
        trig_enabled = True
        try:
            trig_policy = macro_arg_wrapper_policy(ctx, ctx.lm_hp, target)
            trig_enabled = bool(getattr(trig_policy, "trig", True))
        except Exception:
            trig_policy = None
            trig_enabled = True

        # Step 1: Run probes to detect structure
        # Generalized additivity probe: can we decompose as g(NN1 + NN2)?
        genadd_res = _probe_genadd_for_nn_leaf(
            root=st.root,
            model=st.model,
            target=target,
            train_loader=ctx.train_loader_probe,
            device=ctx.device,
            dtype=ctx.dtype,
            n_points=2048,
            poly_deg=2,
        )
        if genadd_res is not None and getattr(genadd_res, "ok", False):
            bp = genadd_res.best_pair
            ctx.log(
                f"[Stage B] GenAdd probe on NN leaf vars {target.var_idxs}: "
                f"best_pair=(i={bp.i}, j={bp.j}), "
                f"rel_res_x={bp.rel_res_x:.3g}, rel_res_y={bp.rel_res_y:.3g}, ok={genadd_res.ok}"
            )

        # Trapped-variable probe: can we factor as A(x_leaky) * B(x_leaky, x_trapped)?
        trapped_res = _probe_trapped_for_nn_leaf(
            root=st.root,
            model=st.model,
            target=target,
            train_loader=ctx.train_loader_probe,
            device=ctx.device,
            dtype=ctx.dtype,
            n_points=2048,
            kind="multiplicative",
            candidate_P="product",
        )

        trapped_fact = None
        if trapped_res is not None and getattr(trapped_res, "ok", False):
            ctx.log(
                f"[Stage B] Trapped-variable probe on NN leaf vars {target.var_idxs}: "
                f"trapped={trapped_res.trapped_idx}, leaky={trapped_res.leaky_idx}, "
                f"rel_res={trapped_res.rel_res:.3g}"
            )
            try:
                atom_to_leaf_tv = build_atom_to_leaf_map(st.root, st.model)
                subtree_tv = _SubtreeModel(root=target, atom_to_leaf=atom_to_leaf_tv)
                fact = _compute_trapped_factorization(
                    model=subtree_tv,
                    datagen=ctx.train_loader_probe,
                    trapped_idx=int(trapped_res.trapped_idx),
                    leaky_idx=int(trapped_res.leaky_idx),
                    device=ctx.device,
                    dtype=ctx.dtype,
                    candidate_P="product",
                    n_points=4096,
                    min_points=400,
                )
                if fact is not None:
                    P_vals, xL_vals, logB_vals, logA_vals = fact
                    degA, degB = 3, 3
                    coeffA = _fit_poly_1d_trapped(xL_vals, logA_vals, deg=degA)
                    coeffB = _fit_poly_1d_trapped(P_vals, logB_vals, deg=degB)
                    if coeffA is not None and coeffB is not None:
                        trapped_fact = dict(
                            trapped=int(trapped_res.trapped_idx),
                            leaky=int(trapped_res.leaky_idx),
                            degA=degA,
                            degB=degB,
                            coeffA=coeffA.detach().cpu(),
                            coeffB=coeffB.detach().cpu(),
                        )
            except Exception as e:
                ctx.log(f"[Stage B] Trapped-variable factorisation failed: {e}")

        # Find best trig spec for this target (if any)
        # Prefer lowest rel_std (cleanest oracle fit); break ties by highest strength.
        trig_spec_for_target = None
        if ctx.trig_by_axis:
            best_rel_std = float("inf")
            best_strength = 0.0
            for j in target.var_idxs:
                spec = ctx.trig_by_axis.get(int(j))
                if spec is None:
                    continue
                rs = float(getattr(spec, "rel_std", 1.0))
                s_str = float(spec.strength)
                if rs < best_rel_std or (rs == best_rel_std and s_str > best_strength):
                    best_rel_std = rs
                    best_strength = s_str
                    trig_spec_for_target = spec

        # ------------------------------------------------------------------
        # Phase 3 probes: output-transform hint + (log/square) Hessian-const hints
        # These are *hints only* (templates will consume them in Phase 4+), but
        # we already use them here to lightly reorder candidate attempts.
        # ------------------------------------------------------------------
        transform_hint: Optional[TransformHint] = None
        logquad_hint: Optional[QuadraticHint] = None
        squarequad_hint: Optional[QuadraticHint] = None
        sat_specs = None  # Initialize to prevent UnboundLocalError if exception occurs

        try:
            atom_to_leaf_local = build_atom_to_leaf_map(st.root, st.model)
            leaf = atom_to_leaf_local.get(id(target), None)
            if leaf is not None:
                Nx = effective_arity(target)  # Handles compound atoms

                def _datagen_sub():
                    for batch in ctx.train_loader_probe:
                        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                        xb = xb.to(device=ctx.device, dtype=ctx.dtype)
                        yield _build_atom_input_tensor(target, xb)  # Handles compound atoms

                transform_hint = ctx.cached(
                    ("phase3", "transform", id(target)),
                    lambda: probe_output_transforms(
                        leaf,
                        _datagen_sub,
                        Nxvars=Nx,
                        device=ctx.device,
                        max_points=2048,
                    ),
                )
                logquad_hint = ctx.cached(
                    ("phase3", "logquad", id(target)),
                    lambda: detect_log_hessian_quadratic(
                        leaf,
                        _datagen_sub,
                        Nxvars=Nx,
                        device=ctx.device,
                        max_points=2048,
                    ),
                )
                squarequad_hint = ctx.cached(
                    ("phase3", "squarequad", id(target)),
                    lambda: detect_square_hessian_quadratic(
                        leaf,
                        _datagen_sub,
                        Nxvars=Nx,
                        device=ctx.device,
                        max_points=2048,
                    ),
                )
                sat_specs = ctx.cached(
                    ("phase3", "saturating_axes", id(target)),
                    lambda: discover_saturating_axes(
                        leaf,
                        _datagen_sub,
                        Nxvars=Nx,
                        device=ctx.device,
                        max_points=6000,
                        n_line=256,
                    ),
                )

                if ctx.verbose:
                    if transform_hint is not None and transform_hint.ok:
                        extra = ""
                        try:
                            if str(getattr(transform_hint, "best_name", "")).endswith("_affine"):
                                params = (
                                    getattr(getattr(transform_hint, "best", None), "params", None)
                                    or {}
                                )
                                a = params.get("alpha", None)
                                b = params.get("beta", None)
                                if a is not None and b is not None:
                                    extra = f", alpha≈{float(a):.3g}, beta≈{float(b):.3g}"
                        except Exception:
                            extra = ""
                        ctx.log(
                            f"[Stage B] Phase3 transform-hint on NN vars {target.var_idxs}: "
                            f"best={transform_hint.best_name}, Δscore≈{transform_hint.score_improvement:.2f} "
                            f"(domain={transform_hint.best.domain_ok_frac:.2f}){extra}"
                        )
                    if logquad_hint is not None and logquad_hint.ok:
                        ctx.log(
                            f"[Stage B] Phase3 log-quadratic hint on NN vars {target.var_idxs}: "
                            f"rel_const={logquad_hint.hess_const_rel:.3f}, rank≈{logquad_hint.rank} "
                            f"(domain={logquad_hint.domain_ok_frac:.2f})"
                        )
                    if squarequad_hint is not None and squarequad_hint.ok:
                        ctx.log(
                            f"[Stage B] Phase3 square-quadratic hint on NN vars {target.var_idxs}: "
                            f"rel_const={squarequad_hint.hess_const_rel:.3f}, rank≈{squarequad_hint.rank}"
                        )
                    if sat_specs:
                        ctx.log(f"[Stage B] Phase3 saturating axes: {[s.axis for s in sat_specs]}")
        except Exception as e:
            if ctx.verbose:
                ctx.log(f"[Stage B] Phase3 probes failed on NN leaf vars {target.var_idxs}: {e}")

        # Helper: build generalized-additivity composition g(NN1 + NN2)
        def build_genadd_compose(outer_ctor: Callable[[Node], Node]):
            if genadd_res is None or not getattr(genadd_res, "ok", False):
                return None
            var_idxs = [int(j) for j in target.var_idxs]
            if len(var_idxs) != 2:
                return None
            i, j = int(genadd_res.best_pair.i), int(genadd_res.best_pair.j)
            if i == j or i not in var_idxs or j not in var_idxs:
                return None

            num_segments = int((target.kwargs or {}).get("num_segments", 16))
            dual_layer = bool((target.kwargs or {}).get("dual_layer", False))
            nn1 = AtomNode(
                kind="nn",
                var_idxs=(i,),
                kwargs={"num_segments": num_segments, "dual_layer": dual_layer},
                tag=None,
            )
            nn2 = AtomNode(
                kind="nn",
                var_idxs=(j,),
                kwargs={"num_segments": num_segments, "dual_layer": dual_layer},
                tag=None,
            )
            inner = AddNode(nn1, nn2)
            new_sub = outer_ctor(inner)
            return replace_atom_in_ast(st.root, target, new_sub)

        cands: List[Candidate] = []

        # Step 2: Build candidate rewrites

        # 2.1 Trig-rational family (sin-ratio).
        if trapped_res is not None and getattr(trapped_res, "ok", False):
            cand_tr = propose_trig_rational(ctx, target, trapped_res, label="trig_rational")
            if cand_tr is not None:
                leaky = int(getattr(trapped_res, "leaky_idx", -1))
                trapped = int(getattr(trapped_res, "trapped_idx", -1))
                cand_tr.meta.setdefault(
                    "log",
                    f"[Stage B]  Trying trig-rational rewrite on nn vars={target.var_idxs} (leaky={leaky}, trapped={trapped})",
                )
                cands.append(cand_tr)

            # The leaky/trapped assignment comes from an upstream probe and the
            # denominator variable is baked into the candidate AST, so a
            # drift-flipped hint is unrecoverable by initialization. Propose
            # the swapped-assignment sibling too and let the greedy race decide;
            # its quality gate prevents hopeless divergent starts.
            try:
                import types as _types

                _swapped = _types.SimpleNamespace(
                    ok=True,
                    candidate_P=getattr(trapped_res, "candidate_P", "product"),
                    leaky_idx=int(getattr(trapped_res, "trapped_idx")),
                    trapped_idx=int(getattr(trapped_res, "leaky_idx")),
                )
                cand_sw = propose_trig_rational(
                    ctx, target, _swapped, label="trig_rational_swap"
                )
            except (AttributeError, TypeError, ValueError):
                cand_sw = None
            if cand_sw is not None:
                cand_sw.meta.setdefault(
                    "log",
                    f"[Stage B]  Trying trig-rational rewrite (swapped assignment) on nn vars={target.var_idxs} "
                    f"(leaky={int(getattr(trapped_res, 'trapped_idx', -1))}, trapped={int(getattr(trapped_res, 'leaky_idx', -1))})",
                )
                cands.append(cand_sw)

        # 2.2 Trapped multiplicative: exp(poly_A(xL)) * exp(poly_B(xL, xT))
        if trapped_fact is not None:
            leaky = int(trapped_fact["leaky"])
            trapped_idx_global = int(trapped_fact["trapped"])
            coeffA = trapped_fact["coeffA"]
            coeffB = trapped_fact["coeffB"]
            degA = int(trapped_fact["degA"])
            degB = int(trapped_fact["degB"])

            tagA = f"tm_A_{leaky}_{trapped_idx_global}"
            tagB = f"tm_B_{leaky}_{trapped_idx_global}"
            A_atom = AtomNode(kind="exp_poly", var_idxs=(leaky,), kwargs={"degree": degA}, tag=tagA)
            B_atom = AtomNode(
                kind="exp_poly",
                var_idxs=(leaky, trapped_idx_global),
                kwargs={"degree": 2 * degB},
                tag=tagB,
            )
            cand_root = replace_atom_in_ast(st.root, target, MulNode(A_atom, B_atom))

            def _init(root_new: Node, model_new: nn.Module, _coeffA=coeffA, _coeffB=coeffB):
                try:
                    atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
                    aA = aB = None
                    for a in _collect_all_atoms(root_new):
                        if not isinstance(a, AtomNode):
                            continue
                        if a.tag == tagA:
                            aA = a
                        elif a.tag == tagB:
                            aB = a
                    leafA = atom_to_leaf_new.get(id(aA), None) if aA is not None else None
                    leafB = atom_to_leaf_new.get(id(aB), None) if aB is not None else None
                    if leafA is None or leafB is None:
                        return

                    with torch.no_grad():
                        if _coeffA is not None:
                            pA = _leaf_coeff_param(leafA)
                            if pA is not None:
                                cA = _coeffA.to(device=pA.device, dtype=pA.dtype)
                                pA_flat = pA.view(-1)
                                nA = min(pA_flat.numel(), cA.numel())
                                pA_flat.zero_()
                                pA_flat[:nA].copy_(cA[:nA])

                        if _coeffB is not None:
                            pB = _leaf_coeff_param(leafB)
                            if pB is None:
                                return
                            cB = _coeffB.to(device=pB.device, dtype=pB.dtype)
                            coreB = _poly_like_core(leafB)
                            expsB = getattr(coreB, "exps", None)
                            if expsB is None:
                                return
                            expsB = expsB.to(pB.device)
                            pB.zero_()
                            for k in range(cB.numel()):
                                mask = (expsB[:, 0] == k) & (expsB[:, 1] == k)
                                idx = mask.nonzero(as_tuple=False)
                                if idx.numel() == 0:
                                    continue
                                i = int(idx[0, 0])
                                if pB.dim() == 1:
                                    pB[i] = cB[k]
                                else:
                                    pB[i, :] = cB[k]
                except Exception as e:
                    print("[Stage B] trapped_mult init failed:", e)

            cands.append(Candidate("trapped_mult", cand_root, _init))

        # 2.3 Generalized-additivity compositions: sin(NN1+NN2), cos(NN1+NN2), exp(NN1+NN2)
        for label, outer_ctor in [
            ("genadd_sin", SinNode),
            ("genadd_cos", CosNode),
            ("genadd_exp", ExpNode),
        ]:
            # Shared wrapper-policy gate: honour global trig enable.
            if (outer_ctor in (SinNode, CosNode)) and (not trig_enabled):
                continue
            cand_root = build_genadd_compose(outer_ctor)
            if cand_root is not None:
                cands.append(Candidate(label, cand_root))

        # 2.4 Hint-driven generic templates (Phase 4)
        #     These are small, parameterized families trained by LM.
        if logquad_hint is not None and getattr(logquad_hint, "ok", False):
            cands.extend(propose_exp_of_quadratic(ctx, target, logquad_hint))

        if transform_hint is not None and getattr(transform_hint, "ok", False):
            # Reciprocal-linear template: u(x) ≈ 1 / (a + b·x)
            if transform_hint.best_name == "recip":
                cand_rl = propose_rational_linear(ctx, target, transform_hint, label="rat_linear")
                if cand_rl is not None:
                    cands.append(cand_rl)

            # Log-driven exponential polynomial: if log(u) looks simple -> propose scale*exp_poly.
            if transform_hint.best_name == "log":
                cands.extend(
                    propose_exp_poly_from_log_hint(ctx, target, transform_hint, degrees=(1, 2))
                )

            # Inverse-trig driven: if asin(u) is simple -> propose sin(poly); likewise for acos.
            cands.extend(propose_sin_cos_from_inverse_hint(ctx, target, transform_hint, degree=2))

        # tanh family (atanh-transform or saturating-axis hint)
        cands.extend(
            propose_tanh_family(ctx, target, transform_hint=transform_hint, sat_specs=sat_specs)
        )

        # symexp_denom family (symexp-denom transform hint)
        cands.extend(
            propose_symexp_denom_family(ctx, target, transform_hint=transform_hint, sat_specs=sat_specs)
        )

        # sinc family (sin(P)/P) driven by trig-axis evidence
        if trig_enabled and trig_spec_for_target is not None:
            cands.extend(propose_sinc_family(ctx, target, trig_spec_for_target, degree_arg=2, p=2))

        # Sparse trig-feature linear closure.  This is a cheap, strong-oracle
        # route for reciprocal/affine trig structures before generic ratpoly
        # gets a chance to approximate the trig function polynomially.
        if trig_enabled and trig_spec_for_target is not None:
            cands.extend(_build_trig_feature_linear_candidates(ctx, target, trig_spec_for_target))

        # 2.6 Remaining multi-D patterns
        # Builders are deferred: they are registered as lazy Candidate
        # objects and only evaluated by the engine when reached in priority
        # order.  This avoids running expensive builders (e.g. factorized symbolic search)
        # when a higher-priority template candidate gets accepted first.
        def add_builder(
            label: str,
            fn: Callable[[], Tuple[Optional[Node], Optional[Callable[..., Any]]]],
            log: Optional[str] = None,
        ):
            meta: Dict[str, Any] = {}
            if log is not None:
                meta["log"] = log
            cands.append(Candidate(label=label, builder=fn, meta=meta))

        # Direct rational polynomial fit on the NN leaf output.
        # This is particularly helpful for classic Feynman-style laws that are
        # exactly rational but not separable.
        _eu = getattr(ctx, "enforce_units", False)
        _us = getattr(ctx, "units_spec", None)
        _poly_homo = False
        _eff_x_dims: List[Tuple[Any, ...]] = []
        if _eu and _us is not None:
            _dimless = tuple(0 for _ in _us.unit_system.base)
            _eff_x_dims = _effective_input_dims_for_atom(target, _us)
            _poly_homo = bool(_eff_x_dims) and all(d != _dimless for d in _eff_x_dims)
        # Extract dimensional info for the degree probe (new path).
        _ratpoly_target_dim = None
        _ratpoly_x_dims = None
        if _eu and _us is not None:
            try:
                _ratpoly_target_dim = tuple(ctx.infer_target_dim(target) or ())
                if _ratpoly_target_dim:
                    _ratpoly_x_dims = _eff_x_dims or None
                else:
                    _ratpoly_target_dim = None
            except Exception:
                _ratpoly_target_dim = None
        # Eagerly probe all rational polynomial degree pairs and return
        # multiple candidates ordered by complexity so the engine can fall
        # back to higher-degree fits when a simpler one is rejected.
        if not _ctx_pattern_disabled(ctx, "ratpoly"):
            _ratpoly_results = _build_ratpoly_candidates(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                enforce_units=_eu,
                target_dim=_ratpoly_target_dim,
                x_dims=_ratpoly_x_dims,
            )
            for _rp_i, (_rp_root, _rp_init, _rp_meta) in enumerate(_ratpoly_results):
                cands.append(Candidate(
                    label="ratpoly" if _rp_i == 0 else f"ratpoly[{_rp_i}]",
                    root=_rp_root,
                    init_fn=_rp_init,
                    meta=_rp_meta,
                ))

        add_builder(
            "sqrt_ratpoly",
            lambda _enforce=_eu, _td=_ratpoly_target_dim, _xd=_ratpoly_x_dims: _build_sqrt_ratpoly_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                enforce_units=_enforce,
                target_dim=_td,
                x_dims=_xd,
            ),
        )
        add_builder(
            "sqrt_poly",
            lambda _h=_poly_homo, _td=_ratpoly_target_dim, _xd=_ratpoly_x_dims: _build_sqrt_poly_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                degree=2,
                noise_floor_raw=_stageB_noise_floor_raw(ctx),
                homogeneous=_h,
                target_dim=_td,
                x_dims=_xd,
            ),
        )
        if not _ctx_pattern_disabled(ctx, "inv_poly"):
            _inv_poly_results = _build_inv_poly_candidates(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                max_degree=2,
                homogeneous=_poly_homo,
                target_dim=_ratpoly_target_dim,
                x_dims=_ratpoly_x_dims,
            )
            for _ip_i, (_ip_root, _ip_init, _ip_meta) in enumerate(_inv_poly_results):
                cands.append(Candidate(
                    label="inv_poly" if _ip_i == 0 else f"inv_poly[{_ip_i}]",
                    root=_ip_root,
                    init_fn=_ip_init,
                    meta=_ip_meta,
                ))
        add_builder(
            "quad_poly",
            lambda _h=_poly_homo, _td=_ratpoly_target_dim, _xd=_ratpoly_x_dims: _build_quadratic_poly_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                degree=2,
                homogeneous=_h,
                target_dim=_td,
                x_dims=_xd,
            ),
        )
        # High-degree polynomial fallback for bivariate NN leaves. For
        # f(x2,x3)=x2²x3²(x2+x3)=x2³x3²+x2²x3³, the quadratic family is
        # insufficient, so retain a degree-5 fallback for two-input atoms.
        if effective_arity(target) == 2:
            add_builder(
                "highdeg_poly",
                lambda _h=_poly_homo, _td=_ratpoly_target_dim, _xd=_ratpoly_x_dims: _build_quadratic_poly_candidate(
                    root=st.root,
                    target=target,
                    reuse=st.reuse,
                    train_loader=ctx.train_loader_probe,
                    device=ctx.device,
                    dtype=ctx.dtype,
                    degree=5,
                    rel_rms_threshold=5e-3,  # Slightly looser threshold for complex polynomials
                    homogeneous=_h,
                    target_dim=_td,
                    x_dims=_xd,
                ),
            )
        add_builder(
            "exp_branch",
            lambda: _build_power_exp_rat_candidate(
                root=st.root,
                target=target,
                scale_specs=ctx.scale_specs,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
            ),
        )
        add_builder(
            "pure_exp_branch",
            lambda: _build_pure_exp_rat_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
            ),
        )
        add_builder(
            "log_poly",
            lambda _h=_poly_homo: _build_log_poly_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                degree=2,
                homogeneous=_h,
            ),
        )
        add_builder(
            "log_ratpoly",
            lambda _enforce=_eu, _td=_ratpoly_target_dim, _xd=_ratpoly_x_dims: _build_log_ratpoly_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                enforce_units=_enforce,
                target_dim=_td,
                x_dims=_xd,
            ),
        )
        add_builder(
            "log_poly",
            lambda _h=_poly_homo: _build_log_poly_candidate(
                root=st.root,
                target=target,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                homogeneous=_h,
            ),
        )

        # 2.7 Affine-in-variable split: u(z, t) = A(z) + t * B(z)
        # For 2D atoms where one variable is affine (H[t,t] ≈ 0), splits into two 1D problems
        add_builder(
            "affine_split",
            lambda: _build_affine_split_candidate(
                root=st.root,
                target=target,
                model=st.model,
                reuse=reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                units_spec=getattr(ctx, "units_spec", None),
                enforce_units=getattr(ctx, "enforce_units", False),
            ),
        )

        # 2.8 Trig composition (and trig-difference when phase depends on a partner axis)
        hint_for_trig = None
        if trig_enabled and trig_spec_for_target is not None:
            hint_for_trig = ctx.trig_structure_by_axis.get(int(trig_spec_for_target.axis))

        if (
            trig_enabled
            and trig_spec_for_target is not None
            and hint_for_trig is not None
            and getattr(hint_for_trig, "kind", None) == "difference"
            # NOTE: discover_trig_argument_structure returns a PeriodicityStructureHint
            # with the partner axis stored as `partner` (not `partner_axis`).
            and int(getattr(hint_for_trig, "partner", getattr(hint_for_trig, "partner_axis", -1)))
            in target.var_idxs
        ):

            from ..candidate_builders import _build_trig_diff_affine_envelope_candidate

            def _trig_diff_affine_env(_h=_poly_homo):
                int(trig_spec_for_target.axis)
                partner = int(
                    getattr(hint_for_trig, "partner", getattr(hint_for_trig, "partner_axis", -1))
                )
                return _build_trig_diff_affine_envelope_candidate(
                    root=st.root,
                    target=target,
                    trig_spec=trig_spec_for_target,
                    model=st.model,
                    train_loader=ctx.train_loader_probe,
                    device=ctx.device,
                    dtype=ctx.dtype,
                    partner_axis=partner,
                    homogeneous=_h,
                )

            axis = int(trig_spec_for_target.axis)
            partner = int(
                getattr(hint_for_trig, "partner", getattr(hint_for_trig, "partner_axis", -1))
            )
            log = (
                f"[Stage B]  Trying trig-diff-affine-envelope on NN leaf with vars {target.var_idxs}, "
                f"arg≈x{axis}-x{partner}"
            )
            add_builder("trig_diff_affine_env", _trig_diff_affine_env, log=log)

            def _trig_diff():
                axis = int(trig_spec_for_target.axis)
                partner = int(
                    getattr(hint_for_trig, "partner", getattr(hint_for_trig, "partner_axis", -1))
                )

                # Estimate amplitude and phase from NN output via A*cos + B*sin + C fit
                estimated_amp, estimated_phase = _estimate_trig_amp_phase(
                    target, st, ctx, trig_spec_for_target, axis, partner
                )

                w = float(trig_spec_for_target.omega)

                # Compute arg poly coefficients: [phase, -omega, omega] for poly(x_axis, x_partner)
                # This gives: phase + omega*x_axis - omega*x_partner = phase + omega*(x_axis - x_partner)
                init_arg_coeffs = [estimated_phase, -w, w]

                # Compute amp poly coefficients: [amplitude] for constant amplitude
                # (amplitude poly has degree=2 over remaining vars, but we just set constant term)
                init_amp_coeffs = [estimated_amp]

                r = _make_multid_trig_pair_rewrite(
                    root=st.root,
                    target=target,
                    axis=axis,
                    partner_axis=partner,
                    degree_arg=1,
                    degree_amp=2,
                    trig_kind="cos",
                    init_arg_coeffs=init_arg_coeffs,
                    init_amp_coeffs=init_amp_coeffs,
                    homogeneous=_poly_homo,
                )

                # No _init callback needed! Coefficients are set during leaf creation.
                return (r, None)

            axis = int(trig_spec_for_target.axis)
            partner = int(
                getattr(hint_for_trig, "partner", getattr(hint_for_trig, "partner_axis", -1))
            )
            log = f"[Stage B]  Trying trig-diff rewrite on NN leaf with vars {target.var_idxs}, arg≈x{axis}-x{partner}"
            add_builder("trig_diff", _trig_diff, log=log)

        # 2.6 Trig composition (if strong trig axis detected)
        if trig_enabled and trig_spec_for_target is not None:
            _trig_axis = int(trig_spec_for_target.axis)
            if has_nontrivial_input(target):
                # Compound target: the amp poly consumes the remaining INPUT
                # expressions, so homogeneity must be judged on their computed
                # dims, not on raw variable dims.
                _axis_pos_tc = trivial_input_position(target, _trig_axis)
                _amp_dims_tc = (
                    [
                        d
                        for j, d in enumerate(_eff_x_dims)
                        if j != _axis_pos_tc
                    ]
                    if (_eff_x_dims and _axis_pos_tc is not None)
                    else []
                )
                _trig_amp_homo = (
                    _poly_leaf_homogeneous_for_dims(_amp_dims_tc, _us)
                    if (_eu and _us is not None and _amp_dims_tc)
                    else False
                )
            else:
                _trig_amp_vars = [
                    int(v) for v in target.var_idxs if int(v) != _trig_axis
                ]
                _trig_amp_homo = (
                    _poly_leaf_homogeneous_for_raw_var_idxs(_trig_amp_vars, _us)
                    if (_eu and _us is not None)
                    else False
                )

            def _trig_comp(_h=_trig_amp_homo):
                r = _make_multid_trig_rewrite(
                    root=st.root,
                    target=target,
                    spec=trig_spec_for_target,
                    degree_arg=1,
                    degree_amp=1,
                    trig_kind="sin",
                    homogeneous=_h,
                )
                return (r, None)

            log = f"[Stage B]  Trying trig-compose rewrite on NN leaf with vars {target.var_idxs}, axis={int(trig_spec_for_target.axis)}, omega≈{float(trig_spec_for_target.omega):.3g}"
            add_builder("trig_comp", _trig_comp, log=log)

        # 2.7 Trig affine envelope (derivative-based extraction)
        # For y = A(u) + B(u)*cos(ωt+φ), uses derivative identities to extract A and B²
        if trig_enabled and trig_spec_for_target is not None and len(target.var_idxs) > 1:
            from ..candidate_builders import _build_trig_affine_envelope_candidate

            _trig_axis = int(trig_spec_for_target.axis)
            _trig_env_other_vars = [
                int(v) for v in target.var_idxs if int(v) != _trig_axis
            ]
            _trig_env_homo = (
                _poly_leaf_homogeneous_for_raw_var_idxs(_trig_env_other_vars, _us)
                if (_eu and _us is not None)
                else False
            )

            def _trig_affine_env(_h=_trig_env_homo):
                return _build_trig_affine_envelope_candidate(
                    root=st.root,
                    target=target,
                    trig_spec=trig_spec_for_target,
                    model=st.model,
                    train_loader=ctx.train_loader_probe,
                    device=ctx.device,
                    dtype=ctx.dtype,
                    deg_offset=2,
                    deg_amp2=3,
                    homogeneous=_h,
                )

            log_affine = f"[Stage B]  Trying trig-affine-envelope on NN leaf with vars {target.var_idxs}, axis={int(trig_spec_for_target.axis)}, omega≈{float(trig_spec_for_target.omega):.3g}"
            add_builder("trig_affine_env", _trig_affine_env, log=log_affine)

        # factorized symbolic search explorer: lazy group.  Register placeholder candidates
        # that share a compute-once cache.  The expensive explorer only
        # runs when the engine reaches the first placeholder in priority
        # order; if higher-priority templates (ratpoly, sqrt_ratpoly) are
        # accepted first, factorized symbolic search never fires.
        if self.factorized_search_rule is not None:
            _bsr_cache: Dict[str, Any] = {}

            def _bsr_get(idx: int, _cache=_bsr_cache):
                if "cands" not in _cache:
                    _cache["cands"] = self.factorized_search_rule.propose(ctx, target) or []
                cl = _cache["cands"]
                if idx >= len(cl):
                    return None
                c = cl[idx]
                meta = dict(c.meta) if c.meta else {}
                meta["_label"] = c.label
                return (c.root, c.init_fn, meta)

            _max_bsr = getattr(self.factorized_search_rule, "return_topk", 5) * 2
            for _bi in range(_max_bsr):
                cands.append(Candidate(
                    label=f"factorized_search[{_bi}]",
                    builder=lambda _idx=_bi: _bsr_get(_idx),
                ))

        # Light hint-guided ordering (greedy StageB engine accepts first improvement).
        # NOTE: Phase 5 will replace this with a small frontier/beam.
        if cands:
            prio: dict[str, float] = {}

            # Baseline ordering: a well-fit rational polynomial is a very strong
            # simplification candidate and is cheap to validate.
            prio["ratpoly"] = 70.0

            # High-value Phase-4 templates
            if trapped_res is not None and getattr(trapped_res, "ok", False):
                prio["trig_rational"] = 90.0

            if logquad_hint is not None and getattr(logquad_hint, "ok", False):
                prio["exp_quad"] = 82.0
                prio["exp_quad_divlin"] = 80.0

            if transform_hint is not None and getattr(transform_hint, "ok", False):
                if transform_hint.best_name == "recip":
                    prio["rat_linear"] = 60.0
                if transform_hint.best_name in ("asin", "asin_affine"):
                    prio["sin_from_asin"] = 55.0
                    prio["sin_affine_from_asin"] = 55.0
                if transform_hint.best_name in ("acos", "acos_affine"):
                    prio["cos_from_acos"] = 55.0
                    prio["cos_affine_from_acos"] = 55.0
                if transform_hint.best_name == "atanh":
                    prio["tanh_rat"] = 62.0

            if trig_spec_for_target is not None:
                prio["trig_feature_linear"] = 88.0
                prio["trig_diff"] = 29.0
                prio["trig_diff_affine_env"] = 33.0  # Like trig_diff but also models an offset term
                prio["trig_affine_env"] = 32.0  # Higher than trig_comp; affine envelope is more specific
                prio["trig_comp"] = 25.0

                # Soft deferral: in a 2D leaf with a confident trig-difference hint,
                # try trig_diff/trig_comp first, but keep sinc_p2 available.
                have_trig_diff = any(getattr(c, "label", None) == "trig_diff" for c in cands)
                sinc_prio = 28.0
                if have_trig_diff and len(getattr(target, "var_idxs", ()) or ()) == 2:
                    hint = ctx.trig_structure_by_axis.get(int(trig_spec_for_target.axis))
                    if hint is not None and getattr(hint, "kind", None) == "difference":
                        partner = getattr(hint, "partner", getattr(hint, "partner_axis", None))
                        try:
                            partner_i = int(partner) if partner is not None else None
                        except Exception:
                            partner_i = None
                        score = float(getattr(hint, "score", 0.0) or 0.0)
                        if (
                            partner_i is not None
                            and partner_i in target.var_idxs
                            and math.isfinite(score)
                            and score >= 0.25
                        ):
                            # Place sinc behind trig_comp (and behind trig_diff via prio above), but keep it available.
                            sinc_prio = 24.0
                            if getattr(ctx, "verbose", False):
                                ctx.log(
                                    f"[Stage B]  Deferring sinc_p2 on vars={target.var_idxs} due to trig-diff hint "
                                    f"(axis={int(trig_spec_for_target.axis)}, partner={partner_i}, score={score:.2f})"
                                )
                prio["sinc_p2"] = sinc_prio

            if squarequad_hint is not None and getattr(squarequad_hint, "ok", False):
                prio["sqrt_poly"] = 60.0
                prio["sqrt_ratpoly"] = 55.0

            if logquad_hint is not None and getattr(logquad_hint, "ok", False):
                prio["pure_exp_branch"] = 70.0
                prio["exp_branch"] = 65.0
                prio["log_poly"] = 62.0
                prio["log_ratpoly"] = 60.0

            if sat_specs is not None and len(sat_specs) > 0:
                prio["tanh_rat_amp"] = 60.0

            if transform_hint is not None and getattr(transform_hint, "ok", False):
                if transform_hint.best_name == "log":
                    prio["exp_poly_log_d1"] = max(prio.get("exp_poly_log_d1", 0.0), 66.0)
                    prio["exp_poly_log_d2"] = max(prio.get("exp_poly_log_d2", 0.0), 64.0)
                    prio["log_ratpoly"] = max(prio.get("log_ratpoly", 0.0), 55.0)
                    prio["pure_exp_branch"] = max(prio.get("pure_exp_branch", 0.0), 60.0)
                elif transform_hint.best_name == "exp":
                    best = transform_hint.best
                    if getattr(best, "poly2_rms_rel", 1e9) <= 0.15:
                        prio["log_poly"] = max(prio.get("log_poly", 0.0), 60.0)
                    if getattr(best, "rat_rms_rel", 1e9) <= 0.15:
                        prio["log_ratpoly"] = max(prio.get("log_ratpoly", 0.0), 55.0)
                elif transform_hint.best_name == "sqrt":
                    prio["sqrt_poly"] = max(prio.get("sqrt_poly", 0.0), 55.0)
                    prio["sqrt_ratpoly"] = max(prio.get("sqrt_ratpoly", 0.0), 50.0)

            if any(getattr(c, "label", None) == "quad_poly" for c in cands):
                prio["quad_poly"] = max(prio.get("quad_poly", 0.0), 86.0)
            # Prefer the structurally narrower high-degree polynomial family
            # over a generic rational fit when both are available.
            if any(getattr(c, "label", None) == "highdeg_poly" for c in cands):
                prio["highdeg_poly"] = max(prio.get("highdeg_poly", 0.0), 85.0)

            # sqrt patterns are more interpretable than generic ratpoly - give higher priority
            # so they get tried first when both are valid candidates (ratpoly has default 70.0)
            prio["sqrt_ratpoly"] = max(prio.get("sqrt_ratpoly", 0.0), 75.0)
            prio["sqrt_poly"] = max(prio.get("sqrt_poly", 0.0), 75.0)
            prio["inv_poly"] = max(prio.get("inv_poly", 0.0), 75.0)

            # Affine split: u(z, t) = A(z) + t * B(z)
            # Priority between sqrt patterns and exp_branch
            prio["affine_split"] = max(prio.get("affine_split", 0.0), 65.0)

            # Stable sort by priority (descending)
            cands.sort(
                key=lambda c: float(prio.get(c.label, prio.get(candidate_pattern_name(c), 0.0))),
                reverse=True,
            )

        return cands


class RulePolySplit(StageBRule):
    """
    Rule for polynomial splitting into additive components.

    Detects multi-dimensional polynomial atoms and proposes splitting them into
    sums of lower-dimensional polynomials.

    Pattern label: poly_split
    """

    name = "poly_split"

    def iter_targets(self, ctx: StageBContext):
        """Return all multi-dimensional polynomial atoms in the current AST."""
        return _collect_multivariate_poly_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        """
        Generate additive polynomial split candidates.

        Args:
            ctx: Stage B context
            target: Multi-dimensional polynomial atom

        Returns:
            List with one Candidate (poly_split) or empty list
        """
        if not isinstance(target, AtomNode):
            return []

        st = ctx.state
        cand_root, init_fn = _build_additive_poly_split_candidate(
            root=st.root, target=target, model=st.model
        )
        if cand_root is None:
            return []

        return [
            Candidate(
                "poly_split",
                cand_root,
                init_fn,
                meta={
                    "log": f"[Stage B]  Trying additive poly split on vars {target.var_idxs}",
                    "structural": True,
                },
            )
        ]


class RuleSubtreeSeparability(StageBRule):
    """
    Rule for SubtreeSeparability rewrites.

    Applies Stage-A separability search locally under specific AST nodes
    (e.g., under Add/Mul branches) to discover finer-grained separable structure.

    Pattern label: subtree_separability
    """

    name = "subtree_separability"

    def iter_targets(self, ctx: StageBContext):
        """
        Return nodes that are candidates for SubtreeSeparability search.

        Returns empty if run_subtree_separability is not available.
        """
        if run_subtree_separability is None:
            return []
        return _collect_subtree_separability_targets(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        """
        Generate SubtreeSeparability candidate.

        Args:
            ctx: Stage B context
            target: Node under which to run SubtreeSeparability search

        Returns:
            List with one Candidate (subtree_separability) or empty list
        """
        if run_subtree_separability is None:
            return []

        st = ctx.state
        cand_root, init_fn = _build_subtree_separability_candidate(
            root=st.root,
            u_node=target,
            model=st.model,
            reuse=st.reuse,
            train_loader=ctx.train_loader_probe,
            device=ctx.device,
            dtype=ctx.dtype,
            very_verbose=ctx.verbose_separabilities,
        )
        if cand_root is None:
            return []

        return [
            Candidate(
                "subtree_separability",
                cand_root,
                init_fn,
                meta={
                    "log": f"[Stage B]  Trying SubtreeSeparability split under node {type(target).__name__}",
                    "structural": True,
                },
            )
        ]


class RuleLastHardAtomRescue(StageBRule):
    """Capped exhaustive rescue for a single remaining low-arity NN atom."""

    name = "last_hard_atom_rescue"
    exhaustive = True

    def iter_targets(self, ctx: StageBContext):
        atoms = [
            atom
            for atom in _collect_all_atoms(ctx.state.root)
            if isinstance(atom, AtomNode)
            and str(getattr(atom, "kind", "")).lower() == "nn"
        ]
        if len(atoms) != 1:
            return []
        target = atoms[0]
        return [target] if _last_hard_atom_context(ctx, target) is not None else []

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        gate = _last_hard_atom_context(ctx, target)
        if gate is None:
            return []
        if _ctx_pattern_disabled(ctx, self.name):
            return []

        target_dim, x_dims = gate
        st = ctx.state
        cands: List[Candidate] = []

        ratpoly_disabled = _ctx_pattern_disabled(ctx, "ratpoly") or _ctx_pattern_disabled(ctx, "ratpoly_1d")
        if not ratpoly_disabled and int(effective_arity(target)) == 2:
            for label, root_new, init_fn, meta in _build_last_hard_ratio_candidates(
                root=st.root,
                target=target,
                reuse=st.reuse,
                train_loader=ctx.train_loader_probe,
                device=ctx.device,
                dtype=ctx.dtype,
                target_dim=target_dim,
                x_dims=x_dims,
                max_p=4,
                max_q=2,
            ):
                meta = dict(meta or {})
                meta["last_hard_atom_rescue"] = True
                cands.append(Candidate(label=label, root=root_new, init_fn=init_fn, meta=meta))

        if not ratpoly_disabled:
            if int(effective_arity(target)) == 1:
                results_1d = _build_ratpoly_1d_candidates(
                    root=st.root,
                    target=target,
                    reuse=st.reuse,
                    train_loader=ctx.train_loader_probe,
                    device=ctx.device,
                    dtype=ctx.dtype,
                    max_deg_num=4,
                    max_deg_den=4,
                    rel_rms_threshold=2e-2,
                    enforce_units=True,
                    target_dim=target_dim,
                    x_dims=x_dims,
                )
                for i, (root_new, init_fn, meta) in enumerate(results_1d):
                    meta = dict(meta or {})
                    meta["last_hard_atom_rescue"] = True
                    cands.append(
                        Candidate(
                            label="last_ratpoly_1d" if i == 0 else f"last_ratpoly_1d[{i}]",
                            root=root_new,
                            init_fn=init_fn,
                            meta=meta,
                        )
                    )
            elif int(effective_arity(target)) == 2:
                results_nd = _build_ratpoly_candidates(
                    root=st.root,
                    target=target,
                    reuse=st.reuse,
                    train_loader=ctx.train_loader_probe,
                    device=ctx.device,
                    dtype=ctx.dtype,
                    max_deg_num=4,
                    max_deg_den=4,
                    max_terms_total=90,
                    rel_rms_threshold=2e-2,
                    enforce_units=True,
                    target_dim=target_dim,
                    x_dims=x_dims,
                )
                for i, (root_new, init_fn, meta) in enumerate(results_nd):
                    meta = dict(meta or {})
                    meta["last_hard_atom_rescue"] = True
                    cands.append(
                        Candidate(
                            label="last_ratpoly" if i == 0 else f"last_ratpoly[{i}]",
                            root=root_new,
                            init_fn=init_fn,
                            meta=meta,
                        )
                    )

        if cands:
            ctx.log(
                f"[Stage B] RuleLastHardAtomRescue proposing {len(cands)} capped candidate(s) "
                f"for final NN vars={target.var_idxs}: {[c.label for c in cands]}"
            )
        return cands


class RulePowerProduct(StageBRule):
    """Probe multivariate NN atoms for power-product structure.

    Detects NN atoms that approximate  c · x_i^a · x_j^b · ...  via
    log-linear regression:  log|nn(x)| ≈ a₁·log|x₁| + a₂·log|x₂| + ... + c

    For example, x4³/x3 gives a₃ = -1, a₄ = 3 (exact R² ≈ 1).

    Pattern label: power_product
    """

    name = "power_product"

    def iter_targets(self, ctx: StageBContext):
        return _collect_multivariate_nn_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, AtomNode):
            return []
        if str(target.kind).lower() != "nn":
            return []
        eff = effective_arity(target)
        if eff < 2:
            return []

        st = ctx.state
        try:
            atom_to_leaf = build_atom_to_leaf_map(st.root, st.model)
        except Exception:
            ctx.log(f"[Stage B]  power_product skip NN vars={target.var_idxs}: build_atom_to_leaf_map failed")
            return []
        teacher = atom_to_leaf.get(id(target))
        if teacher is None:
            ctx.log(f"[Stage B]  power_product skip NN vars={target.var_idxs}: leaf not found in model")
            return []

        # Gather (X_atom, f_teacher) data

        data = candidate_builders._gather_atom_teacher_data(
            train_loader=ctx.train_loader_probe,
            atom=target,
            teacher=teacher,
            device=ctx.device,
            dtype=ctx.dtype,
            max_points=5000,
        )
        if data is None:
            ctx.log(f"[Stage B]  power_product skip NN vars={target.var_idxs}: teacher data gather failed")
            return []

        X, F = data  # X: [N, m], F: [N]
        X_np = X.detach().cpu().numpy().astype(np.float64)
        F_np = F.detach().cpu().numpy().astype(np.float64).ravel()
        N, m = X_np.shape

        # Filter to rows where all inputs > 0 and |output| > 0.
        # Use |F| so that uniformly-negative NN outputs (e.g. nn ≈ -x4³/x3)
        # are still detected; the sign is absorbed into the FreeConst.
        eps = 1e-12
        F_abs = np.abs(F_np)
        mask = (F_abs > eps)
        for j in range(m):
            mask &= (X_np[:, j] > eps)

        if mask.sum() < max(200, N // 3):
            ctx.log(
                f"[Stage B]  power_product skip NN vars={target.var_idxs}: "
                f"only {mask.sum()}/{N} positive-domain points"
            )
            return []

        X_pos = X_np[mask]
        F_pos = F_abs[mask]

        probe = _powerlaw_probe(X_pos, F_pos, r2_gate=0.98, trim_q=0.30)
        if probe is None:
            return []
        r2_best = (
            probe["r2_trim"]
            if (probe["subset"] != "full" and probe["r2_trim"] is not None)
            else probe["r2_full"]
        )
        try:
            from nestynet_sr.sr_search.gate_telemetry import record_gate

            record_gate(
                "power_product",
                "log_misfit",
                1.0 - float(r2_best),
                0.02,
                accepted=bool(probe["passed"]),
                context={
                    "r2_full": probe["r2_full"],
                    "r2_trim": probe["r2_trim"],
                    "subset": probe["subset"],
                    "vars": str(target.var_idxs),
                },
            )
        except Exception:
            pass
        if not probe["passed"]:
            _trim_note = (
                f", trimmed R²_log={probe['r2_trim']:.4f}"
                if probe["r2_trim"] is not None
                else ""
            )
            ctx.log(
                f"[Stage B]  power_product probe on NN vars={target.var_idxs}: "
                f"R²_log={probe['r2_full']:.4f} < 0.98{_trim_note} → skip"
            )
            return []
        coeffs = probe["coeffs"]
        log_X = probe["log_X"]
        log_F = probe["log_F"]
        ss_tot = probe["ss_tot"]
        r2_log = float(r2_best)
        if probe["subset"] != "full":
            ctx.log(
                f"[Stage B]  power_product probe on NN vars={target.var_idxs}: "
                f"full R²_log={probe['r2_full']:.4f} < 0.98 but {probe['subset']} "
                f"R²_log={probe['r2_trim']:.4f} passes (small-|F| pollution)"
            )

        exponents = coeffs[:m]
        log_c = coeffs[m]

        # Snap exponents to nearest integer or half-integer (within ±0.25)
        snapped = np.zeros(m, dtype=np.float64)
        for j in range(m):
            rounded_half = round(exponents[j] * 2.0) / 2.0
            if abs(exponents[j] - rounded_half) < 0.25:
                snapped[j] = rounded_half
            else:
                snapped[j] = exponents[j]  # keep raw

        # Verify snapped fit is still good (in log-space)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            predicted_snap = log_X @ snapped + log_c
        if not np.all(np.isfinite(predicted_snap)):
            snapped = exponents  # fall back to raw (already validated)
        else:
            ss_res_snap = np.sum((log_F - predicted_snap) ** 2)
            r2_snap = 1.0 - ss_res_snap / ss_tot
            if r2_snap < 0.95:
                # Snapped exponents degrade fit too much; use raw
                snapped = exponents

        ctx.log(
            f"[Stage B]  power_product probe on NN vars={target.var_idxs}: "
            f"R²_log={r2_log:.6f}, exponents={[f'{e:.2f}' for e in snapped]}"
        )

        # Determine sign of original F for FreeConst initialisation.
        # The regression was done on |F|, so exp(log_c) > 0 always.
        # If the NN output was predominantly negative we need -exp(log_c).
        sign_c = 1.0 if np.median(F_np[mask]) >= 0 else -1.0

        # Build replacement AST:  FreeConst * Π u_j^a_j, where u_j are the
        # target's effective inputs.  For a simple atom u_j=x_j; for a compound
        # atom u_0 may be z(x), followed by retained raw extras.  This keeps
        # power-product handling symmetric between variables and compounds.
        input_exprs = tuple(get_input_exprs(target))
        if len(input_exprs) != m:
            ctx.log(
                f"[Stage B]  power_product skip NN vars={target.var_idxs}: "
                f"effective input count {len(input_exprs)} != data dimension {m}"
            )
            return []

        parent_tag = target.tag if target.tag is not None else f"pp_{id(target)}"
        if not any(abs(float(e)) >= 1e-10 for e in snapped):
            return []

        cands: List[Candidate] = []

        # Mixed clean/dirty powers should first be offered as a visible
        # partial peel, not forced into a fractional monomial closure.  The
        # residual NN keeps the ambiguous inputs and normal Stage-B validation
        # decides whether the structural move is worthwhile.
        try:
            split_plan = split_clean_integer_powers(
                tuple(float(e) for e in snapped),
                max_abs_clean_power=8,
                min_clean_support=1,
                min_residual_support=1,
            )
        except Exception:
            split_plan = None
        if split_plan is not None and len(split_plan.residual_indices) < m:
            try:
                pref_factors: List[Node] = []
                for j, exp_int in enumerate(split_plan.clean_powers):
                    if int(exp_int) == 0:
                        continue
                    inp_expr = input_exprs[j]
                    if is_trivial_input(inp_expr):
                        idx = int(inp_expr.var_idxs[0])
                        base_expr: Node = AtomNode(
                            kind="var",
                            var_idxs=(idx,),
                            inputs=(Var(idx),),
                        )
                    else:
                        base_expr = clone_ast(inp_expr)
                    pref_factors.append(
                        base_expr if int(exp_int) == 1 else PowNode(base_expr, int(exp_int))
                    )

                residual_inputs = tuple(
                    clone_ast(input_exprs[int(i)]) for i in split_plan.residual_indices
                )
                residual_vars = sorted(
                    {
                        int(v)
                        for expr in residual_inputs
                        for v in _collect_var_idxs_from_node(expr)
                    }
                )
                if pref_factors and residual_inputs and residual_vars:
                    new_kwargs = dict(getattr(target, "kwargs", {}) or {})
                    new_kwargs.pop("input_expr", None)
                    new_kwargs.pop("extra_var_idxs", None)
                    new_kwargs.pop("compound", None)
                    residual_tag = None
                    if getattr(target, "tag", None) is not None:
                        residual_idx_token = "_".join(
                            str(int(i)) for i in split_plan.residual_indices
                        )
                        clean_power_token = "_".join(
                            str(int(v)) for v in split_plan.clean_powers
                        )
                        residual_tag = (
                            f"{target.tag}_pp_resid_{residual_idx_token}"
                            f"_pref_{clean_power_token}"
                        )
                    residual_atom = AtomNode(
                        kind=getattr(target, "kind", "nn"),
                        var_idxs=tuple(residual_vars),
                        kwargs=new_kwargs,
                        # Arity-changing peels must not reuse the parent NN leaf.
                        tag=residual_tag,
                        inputs=residual_inputs,
                    )
                    pref_ast = pref_factors[0]
                    for f in pref_factors[1:]:
                        pref_ast = MulNode(pref_ast, f)
                    replacement_partial = MulNode(pref_ast, residual_atom)
                    new_root_partial = replace_atom_in_ast(st.root, target, replacement_partial)
                    if new_root_partial is not None:
                        sig_partial = (
                            atom_content_hash(target),
                            hash(("power_product_partial", tuple(split_plan.clean_powers))),
                            hash(tuple(int(i) for i in split_plan.residual_indices)),
                        )
                        cands.append(
                            Candidate(
                                label="power_product_partial_peel",
                                root=new_root_partial,
                                meta={
                                    "structural": True,
                                    "pattern": "power_product_partial_peel",
                                    "partial_forced_monomial_peel": True,
                                    "log": (
                                        f"[Stage B]  Trying power_product partial peel on "
                                        f"NN vars={target.var_idxs}: prefactor="
                                        f"{tuple(int(v) for v in split_plan.clean_powers)}, "
                                        f"residual_indices={tuple(int(i) for i in split_plan.residual_indices)}, "
                                        f"full_exponents={[str(p) for p in split_plan.full_powers]}"
                                    ),
                                },
                                signature=sig_partial,
                            )
                        )
            except Exception as exc:
                ctx.log(
                    f"[Stage B]  power_product partial peel build failed "
                    f"for NN vars={target.var_idxs}: {type(exc).__name__}: {exc}"
                )

        # Build variants for the scalar coefficient:
        # baseline trainable Scale + declared fixed constants (if any).
        const_tag = f"{parent_tag}_c"
        const_variants = _scalar_constant_variants(
            getattr(ctx, "units_spec", None),
            base_tag=const_tag,
            scale_init=float(sign_c * math.exp(log_c)),
        )

        for cvar in const_variants:
            # Rebuild factors per variant so atom ids match this candidate's tree.
            var_factors: List[Node] = []
            new_atom_ids_var: List[int] = []
            for j in range(m):
                if abs(snapped[j]) < 1e-10:
                    continue  # x^0 = 1, skip
                inp_expr = input_exprs[j]
                factor_var_idxs = tuple(int(v) for v in _collect_var_idxs_from_node(inp_expr))
                if not factor_var_idxs:
                    continue
                factor_inputs = None if is_trivial_input(inp_expr) else (clone_ast(inp_expr),)
                tag_part = (
                    f"x{int(inp_expr.var_idxs[0])}"
                    if is_trivial_input(inp_expr)
                    else f"u{j}"
                )
                var_atom = AtomNode(
                    kind="rpoly",
                    var_idxs=factor_var_idxs,
                    kwargs={"degree": 1},
                    tag=f"{parent_tag}_v{tag_part}",
                    inputs=factor_inputs,
                )
                new_atom_ids_var.append(id(var_atom))
                if abs(snapped[j] - 1.0) < 1e-10:
                    var_factors.append(var_atom)
                else:
                    var_factors.append(PowNode(base=var_atom, exponent=float(snapped[j])))

            if not var_factors:
                continue

            scalar_atom = _build_scalar_atom_from_variant(cvar)
            new_atom_ids_var.append(id(scalar_atom))
            product: Node = var_factors[0]
            for f in var_factors[1:]:
                product = MulNode(left=product, right=f)
            replacement = MulNode(left=scalar_atom, right=product)

            # Replace the target atom in the AST
            new_root = replace_atom_in_ast(st.root, target, replacement)
            if new_root is None:
                continue

            # Build init_fn for the new scalar/analytic leaves. Without this,
            # rpoly coeffs start at 0.0 and any negative exponent produces
            # PowNode(0, -k) = inf, killing the candidate.
            _pp_atom_ids = list(new_atom_ids_var)  # snapshot
            _pp_const_val = float(cvar["value"])

            def _init_power_product(root_new, model_new, *, _aids=_pp_atom_ids, _cval=_pp_const_val):
                atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
                with torch.no_grad():
                    for aid in _aids:
                        leaf = atom_to_leaf_new.get(aid)
                        if leaf is None:
                            continue
                        if _set_constant_leaf_value(leaf, float(_cval)):
                            continue
                        core = getattr(leaf, "model", leaf)
                        if not hasattr(core, "coeffs"):
                            continue
                        # Variable rpoly(x): zero learnable params (pure identity).
                        # Only init if the leaf actually has coefficients.
                        if core.coeffs.numel() > 0:
                            core.coeffs.fill_(0.0)
                            core.coeffs[-1] = 1.0

            _init_power_product._after_analytic_init = True

            label_suffix = str(cvar.get("label_suffix", ""))
            sig = (
                atom_content_hash(target),
                hash(tuple(float(s) for s in snapped)),
                hash(str(cvar.get("tag", ""))),
            )
            cands.append(
                Candidate(
                    label=f"power_product{label_suffix}",
                    root=new_root,
                    init_fn=_init_power_product,
                    meta={
                        "structural": True,
                        "log": (
                            f"[Stage B]  Trying power_product on NN vars={target.var_idxs}: "
                            f"exponents={[f'{e:.2f}' for e in snapped]}, R²_log={r2_log:.6f}, "
                            f"c≈{float(cvar['value']):.4g}{label_suffix}"
                        ),
                    },
                    signature=sig,
                )
            )
        return cands


def _joint_product_atom_support(atom: AtomNode) -> Tuple[int, ...]:
    support: set[int] = set()
    try:
        for expr in get_input_exprs(atom):
            support.update(int(v) for v in _collect_var_idxs_from_node(expr))
    except Exception:
        pass
    if not support:
        try:
            support.update(int(v) for v in getattr(atom, "raw_var_idxs", ()) or ())
        except Exception:
            pass
    if not support:
        support.update(int(v) for v in getattr(atom, "var_idxs", ()) or ())
    return tuple(sorted(support))


def _joint_product_has_compound_input(atom: AtomNode) -> bool:
    try:
        return any(not is_trivial_input(inp) for inp in get_input_exprs(atom))
    except Exception:
        return bool(has_nontrivial_input(atom))


def _joint_product_gather_data(
    ctx: StageBContext,
    atoms: Sequence[AtomNode],
    *,
    max_points: int = 5000,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    try:
        atom_to_leaf = build_atom_to_leaf_map(ctx.state.root, ctx.state.model)
    except Exception:
        ctx.log("[Stage B]  joint_product_monomial_closure skip: build_atom_to_leaf_map failed")
        return None

    leaves = []
    for atom in atoms:
        leaf = atom_to_leaf.get(id(atom))
        if leaf is None:
            ctx.log(
                "[Stage B]  joint_product_monomial_closure skip: "
                f"missing leaf for tag={getattr(atom, 'tag', None)}"
            )
            return None
        leaves.append(leaf)

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    n_collected = 0
    for batch in ctx.train_loader_probe:
        x_full = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_full = x_full.to(device=ctx.device, dtype=ctx.dtype)
        prod = torch.ones((x_full.shape[0],), device=ctx.device, dtype=ctx.dtype)
        try:
            for atom, leaf in zip(atoms, leaves):
                x_in = _build_atom_input_tensor(atom, x_full)
                y = leaf(x_in).reshape(-1)
                prod = prod * y
        except Exception as exc:
            ctx.log(
                "[Stage B]  joint_product_monomial_closure skip: "
                f"teacher evaluation failed ({type(exc).__name__}: {exc})"
            )
            return None
        xs.append(x_full.detach().cpu())
        ys.append(prod.detach().cpu())
        n_collected += int(x_full.shape[0])
        if n_collected >= int(max_points):
            break

    if not xs:
        return None
    X = torch.cat(xs, dim=0)[:max_points].to(dtype=torch.float64)
    Y = torch.cat(ys, dim=0)[:max_points].to(dtype=torch.float64).reshape(-1)
    return X, Y


def _fit_joint_product_integer_monomial(
    X_raw: torch.Tensor,
    y: torch.Tensor,
    support: Sequence[int],
    *,
    min_points: int = 256,
    max_abs_exp: int = 8,
    snap_tol: float = 0.10,
    min_log_r2: float = 0.99,
    max_rel_rms: float = 1.0e-2,
    diag: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    def _fail(reason: str, **details) -> None:
        if diag is None:
            return
        diag.clear()
        diag["reason"] = str(reason)
        diag.update(details)

    support = tuple(int(v) for v in support)
    if not support:
        _fail("empty-support")
        return None
    if max(support) >= int(X_raw.shape[1]):
        _fail("support-out-of-range", support=support, n_cols=int(X_raw.shape[1]))
        return None

    Xs = X_raw[:, list(support)].detach().cpu().numpy().astype(np.float64)
    Y = y.detach().cpu().numpy().astype(np.float64).reshape(-1)
    eps = 1.0e-12
    mask = np.isfinite(Y) & (np.abs(Y) > eps)
    mask &= np.all(np.isfinite(Xs), axis=1)
    mask &= np.all(Xs > eps, axis=1)
    if int(mask.sum()) < int(min_points):
        _fail("too-few-positive-domain-points", n_points=int(mask.sum()), min_points=int(min_points))
        return None

    Xm = Xs[mask]
    Ym = Y[mask]
    log_X = np.log(Xm)
    log_Y = np.log(np.abs(Ym))
    A = np.concatenate([log_X, np.ones((log_X.shape[0], 1), dtype=np.float64)], axis=1)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, log_Y, rcond=None)
    except np.linalg.LinAlgError:
        _fail("lstsq-failed")
        return None
    raw_exp = coeffs[:-1]
    log_c = float(coeffs[-1])
    if not np.all(np.isfinite(raw_exp)) or not math.isfinite(log_c):
        _fail("nonfinite-loglinear-fit")
        return None

    snapped = np.round(raw_exp).astype(int)
    snap_err = np.abs(raw_exp - snapped.astype(np.float64))
    if np.any(snap_err > float(snap_tol)):
        _fail(
            "integer-snap-failed",
            raw_exponents=tuple(float(v) for v in raw_exp.tolist()),
            snapped=tuple(int(v) for v in snapped.tolist()),
            max_snap_error=float(np.max(snap_err)),
            snap_tol=float(snap_tol),
            n_points=int(mask.sum()),
        )
        return None
    if np.any(np.abs(snapped) > int(max_abs_exp)):
        _fail(
            "integer-exponent-too-large",
            raw_exponents=tuple(float(v) for v in raw_exp.tolist()),
            snapped=tuple(int(v) for v in snapped.tolist()),
            max_abs_exp=int(max_abs_exp),
        )
        return None
    if not np.any(snapped != 0):
        _fail(
            "all-zero-exponents",
            raw_exponents=tuple(float(v) for v in raw_exp.tolist()),
            snapped=tuple(int(v) for v in snapped.tolist()),
        )
        return None

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pred_log = A @ coeffs
    if not np.all(np.isfinite(pred_log)):
        _fail(
            "nonfinite-log-prediction",
            raw_exponents=tuple(float(v) for v in raw_exp.tolist()),
            snapped=tuple(int(v) for v in snapped.tolist()),
        )
        return None
    ss_res = float(np.sum((log_Y - pred_log) ** 2))
    ss_tot = float(np.sum((log_Y - float(np.mean(log_Y))) ** 2))
    if ss_tot <= eps:
        _fail("zero-log-variance")
        return None
    log_r2 = 1.0 - ss_res / ss_tot
    if not math.isfinite(log_r2) or log_r2 < float(min_log_r2):
        _fail(
            "log-r2-below-threshold",
            raw_exponents=tuple(float(v) for v in raw_exp.tolist()),
            snapped=tuple(int(v) for v in snapped.tolist()),
            log_r2=float(log_r2),
            min_log_r2=float(min_log_r2),
        )
        return None

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mon = np.prod(np.power(Xm, snapped.reshape(1, -1)), axis=1)
    if not np.all(np.isfinite(mon)):
        _fail(
            "nonfinite-snapped-monomial",
            raw_exponents=tuple(float(v) for v in raw_exp.tolist()),
            snapped=tuple(int(v) for v in snapped.tolist()),
        )
        return None
    denom = float(np.dot(mon, mon))
    if denom <= eps:
        _fail("zero-monomial-norm", snapped=tuple(int(v) for v in snapped.tolist()))
        return None
    scale = float(np.dot(mon, Ym) / denom)
    if not math.isfinite(scale) or abs(scale) <= eps:
        _fail("invalid-scale", snapped=tuple(int(v) for v in snapped.tolist()), scale=float(scale))
        return None
    resid = Ym - scale * mon
    rms = float(np.sqrt(np.mean(resid * resid)))
    y_rms = float(np.sqrt(np.mean(Ym * Ym)))
    if y_rms <= eps:
        _fail("zero-product-rms")
        return None
    rel_rms = rms / y_rms
    if not math.isfinite(rel_rms) or rel_rms > float(max_rel_rms):
        _fail(
            "real-space-rms-too-high",
            raw_exponents=tuple(float(v) for v in raw_exp.tolist()),
            snapped=tuple(int(v) for v in snapped.tolist()),
            log_r2=float(log_r2),
            rel_rms=float(rel_rms),
            max_rel_rms=float(max_rel_rms),
            scale=float(scale),
        )
        return None
    sign_pred = np.sign(scale * mon)
    sign_true = np.sign(Ym)
    sign_frac = float(np.mean(sign_pred == sign_true))
    if sign_frac < 0.995:
        _fail(
            "sign-mismatch",
            raw_exponents=tuple(float(v) for v in raw_exp.tolist()),
            snapped=tuple(int(v) for v in snapped.tolist()),
            log_r2=float(log_r2),
            rel_rms=float(rel_rms),
            sign_frac=float(sign_frac),
        )
        return None

    return {
        "support": support,
        "exponents": tuple(int(v) for v in snapped.tolist()),
        "raw_exponents": tuple(float(v) for v in raw_exp.tolist()),
        "scale": scale,
        "log_c": log_c,
        "log_r2": float(log_r2),
        "rel_rms": float(rel_rms),
        "sign_frac": float(sign_frac),
        "n_points": int(mask.sum()),
    }


def _joint_product_diag_text(diag: Dict[str, Any]) -> str:
    if not diag:
        return "reason=unknown"
    parts = [f"reason={diag.get('reason', 'unknown')}"]
    if "n_points" in diag:
        parts.append(f"n={diag['n_points']}")
    if "min_points" in diag:
        parts.append(f"min={diag['min_points']}")
    if "raw_exponents" in diag:
        parts.append("raw=[" + ", ".join(f"{float(v):.3g}" for v in diag["raw_exponents"]) + "]")
    if "snapped" in diag:
        parts.append("snap=[" + ", ".join(str(int(v)) for v in diag["snapped"]) + "]")
    if "max_snap_error" in diag:
        parts.append(f"snap_err={float(diag['max_snap_error']):.3g}")
    if "log_r2" in diag:
        parts.append(f"R2log={float(diag['log_r2']):.6g}")
    if "rel_rms" in diag:
        parts.append(f"rel={float(diag['rel_rms']):.3g}")
    if "sign_frac" in diag:
        parts.append(f"sign={float(diag['sign_frac']):.3g}")
    if "scale" in diag:
        parts.append(f"scale={float(diag['scale']):.3g}")
    return ", ".join(parts)


class RuleJointProductMonomialClosure(StageBRule):
    """Close a connected product of NN leaves when their product is a raw monomial."""

    name = "joint_product_monomial_closure"

    def iter_targets(self, ctx: StageBContext):
        out: List[MulNode] = []
        for mn in _iter_mul_nodes(ctx.state.root):
            factors = _flatten_mul(mn)
            nn_factors = [
                f
                for f in factors
                if isinstance(f, AtomNode) and str(getattr(f, "kind", "")).lower() == "nn"
            ]
            if len(nn_factors) < 2:
                continue
            if len(nn_factors) > 4:
                continue
            supports = [set(_joint_product_atom_support(atom)) for atom in nn_factors]
            has_overlap = any(supports[i] & supports[j] for i in range(len(supports)) for j in range(i + 1, len(supports)))
            has_compound = any(_joint_product_has_compound_input(atom) for atom in nn_factors)
            if not (has_overlap or has_compound):
                continue
            out.append(mn)
        return out

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if not isinstance(target, MulNode):
            return []
        factors = _flatten_mul(target)
        nn_positions = [
            i
            for i, factor in enumerate(factors)
            if isinstance(factor, AtomNode) and str(getattr(factor, "kind", "")).lower() == "nn"
        ]
        if len(nn_positions) < 2 or len(nn_positions) > 4:
            return []
        atoms = tuple(factors[i] for i in nn_positions if isinstance(factors[i], AtomNode))
        support = tuple(sorted({v for atom in atoms for v in _joint_product_atom_support(atom)}))
        if not support or len(support) > 6:
            ctx.log(
                "[Stage B]  joint_product_monomial_closure skip: "
                f"raw support size {len(support)} outside allowed range, support={support}"
            )
            return []

        atom_tags = tuple(str(getattr(atom, "tag", None)) for atom in atoms)
        atom_supports = tuple(_joint_product_atom_support(atom) for atom in atoms)
        compound_flags = tuple(bool(_joint_product_has_compound_input(atom)) for atom in atoms)
        ctx.log(
            "[Stage B]  joint_product_monomial_closure target: "
            f"tags={atom_tags}, supports={atom_supports}, compound={compound_flags}, "
            f"fit_support={support}"
        )

        data = _joint_product_gather_data(ctx, atoms, max_points=5000)
        if data is None:
            return []
        X_raw, y_prod = data
        diag: Dict[str, Any] = {}
        max_rel_rms = _stageB_noisy_rel_rms_threshold(
            ctx,
            1.0e-2,
            y_values=y_prod,
        )
        fit = _fit_joint_product_integer_monomial(
            X_raw,
            y_prod,
            support,
            max_rel_rms=max_rel_rms,
            diag=diag,
        )
        if fit is None:
            ctx.log(
                "[Stage B]  joint_product_monomial_closure: no clean integer monomial "
                f"for {len(atoms)} NN factor(s), support={support}; "
                f"{_joint_product_diag_text(diag)}"
            )
            return []

        try:
            nonzero = [(int(v), int(e)) for v, e in zip(fit["support"], fit["exponents"]) if int(e) != 0]
            monomial = build_monomial_ast(tuple(v for v, _e in nonzero), tuple(e for _v, e in nonzero))
        except Exception:
            ctx.log(
                "[Stage B]  joint_product_monomial_closure skip: "
                f"failed to build monomial AST for support={fit.get('support')} "
                f"exponents={fit.get('exponents')}"
            )
            return []

        parent_tag = "joint_product_monomial"
        try:
            tag_parts = [str(getattr(atom, "tag", "")) for atom in atoms if getattr(atom, "tag", None) is not None]
            if tag_parts:
                parent_tag = "jpm_" + "_".join(tag_parts)
        except Exception:
            pass

        try:
            scale_node = _make_unit_aware_scalar_atom(
                None,
                getattr(ctx, "units_spec", None),
                base_tag=f"{parent_tag}_scale",
                init=float(fit["scale"]),
                strict=bool(getattr(ctx, "enforce_units", False)),
            )
        except Exception:
            ctx.log(
                "[Stage B]  joint_product_monomial_closure skip: "
                "unit-aware scalar construction failed"
            )
            return []

        replacement = MulNode(scale_node, monomial)
        nn_pos_set = set(nn_positions)
        rebuilt_factors: List[Node] = []
        inserted = False
        for idx, factor in enumerate(factors):
            if idx in nn_pos_set:
                if not inserted:
                    rebuilt_factors.append(replacement)
                    inserted = True
                continue
            rebuilt_factors.append(clone_ast(factor))
        new_subtree = _rebuild_mul(rebuilt_factors)
        new_root = _replace_node_in_ast(ctx.state.root, target, new_subtree)
        if new_root is None:
            return []

        sig = (
            _subtree_content_hash(target),
            hash(tuple(int(v) for v in fit["support"])),
            hash(tuple(int(e) for e in fit["exponents"])),
        )
        label_bits = " ".join(f"x{v}^{e}" if e != 1 else f"x{v}" for v, e in nonzero)
        return [
            Candidate(
                label=self.name,
                root=new_root,
                meta={
                    "structural": True,
                    "pattern_family": self.name,
                    "min_free_params": 1,
                    "joint_product_monomial": True,
                    "requires_original_y_sanity": True,
                    "joint_product_support": tuple(int(v) for v in fit["support"]),
                    "joint_product_exponents": tuple(int(e) for e in fit["exponents"]),
                    "joint_product_raw_exponents": tuple(float(e) for e in fit["raw_exponents"]),
                    "joint_product_rel_rms": float(fit["rel_rms"]),
                    "joint_product_log_r2": float(fit["log_r2"]),
                    "joint_product_sign_frac": float(fit["sign_frac"]),
                    "log": (
                        "[Stage B]  Trying joint_product_monomial_closure: "
                        f"{len(atoms)} NN factors -> {label_bits}, "
                        f"scale≈{float(fit['scale']):.4g}, "
                        f"R²_log={float(fit['log_r2']):.6f}, "
                        f"rel={float(fit['rel_rms']):.3g}"
                    ),
                },
                signature=sig,
            )
        ]




def _fraction_dim(dim: Any) -> Tuple[Fraction, ...]:
    return tuple(Fraction(v).limit_denominator(128) for v in dim)


def _dimensionless_log_monomial_powers(
    axis_a: int,
    axis_b: int,
    units_spec: Any,
) -> Optional[Tuple[Fraction, Fraction]]:
    """Return powers ``(p_a, p_b)`` making ``x_a**p_a * x_b**p_b`` dimensionless.

    This is deliberately a units-level helper rather than an expression-level
    exception. It handles any pair of commensurate axes. For example, if
    ``dim(x_a)=dim(x_b)`` it returns ``(1, -1)``; if
    ``dim(x_a)=2*dim(x_b)`` it returns ``(1, -2)``.
    """
    if units_spec is None:
        return Fraction(1), Fraction(-1)
    try:
        dims = getattr(units_spec, "x_dims", ())
        d_a = _fraction_dim(dims[int(axis_a)])
        d_b = _fraction_dim(dims[int(axis_b)])
    except Exception:
        return None

    zero = tuple(Fraction(0) for _ in d_a)
    if d_a == zero and d_b == zero:
        return Fraction(1), Fraction(-1)
    if d_a == zero or d_b == zero:
        return None

    ratio: Optional[Fraction] = None
    for a_i, b_i in zip(d_a, d_b):
        if b_i == 0:
            if a_i != 0:
                return None
            continue
        r_i = a_i / b_i
        if ratio is None:
            ratio = r_i
        elif ratio != r_i:
            return None
    if ratio is None or ratio == 0:
        return None

    # d_a = ratio*d_b, so denominator*d_a - numerator*d_b = 0.
    return Fraction(ratio.denominator), Fraction(-ratio.numerator)


def _pow_var_for_log_monomial(axis: int, power: Fraction) -> Node:
    if power == 1:
        return Var(axis)
    return PowNode(Var(axis), float(power))


def _make_log_monomial_expr(axis_a: int, p_a: Fraction, axis_b: int, p_b: Fraction) -> Node:
    if p_a == 0:
        return _pow_var_for_log_monomial(axis_b, p_b)
    if p_b == 0:
        return _pow_var_for_log_monomial(axis_a, p_a)
    return MulNode(
        _pow_var_for_log_monomial(axis_a, p_a),
        _pow_var_for_log_monomial(axis_b, p_b),
    )


def _format_log_monomial(axis_a: int, p_a: Fraction, axis_b: int, p_b: Fraction) -> str:
    def _part(axis: int, power: Fraction) -> str:
        if power == 1:
            return f"x{axis}"
        return f"x{axis}^{power}"

    return f"{_part(axis_a, p_a)}*{_part(axis_b, p_b)}"


class RuleAdditiveLogRatio(StageBRule):
    """
    Detect when additive pairs of univariate NN atoms form log-ratio patterns.

    Targets patterns like:
        AddNode(NN[xA], NN[xB])  where A ≠ B

    and checks if the learned functions have log signatures:
        NN[xA] ≈ a*log(xA)
        NN[xB] ≈ b*log(xB)

    When detected, first proposes a 1D polylog over a dimensionless compound
    monomial ``z=xA**pA*xB**pB`` when the two axes have commensurate units.
    It also keeps the direct two-input polylog(xA, xB) fallback, which is
    valid for dimensionless axes.

    For example, with y = x0*x1*x2*log(x4/x3), Stage A can detect:
        NN[x0] * NN[x1] * NN[x2] * (NN[x3] + NN[x4])
    and this rule detects that NN[x3] + NN[x4] ≈ log(x4) - log(x3) = log(x4/x3).

    Pattern label: log_ratio
    """

    name = "log_ratio"

    def iter_targets(self, ctx: StageBContext):
        """
        Find AddNode patterns where both children are univariate NN atoms
        with different variables.
        """
        targets = []

        def _find_additive_pairs(node: Node) -> None:
            if isinstance(node, AddNode):
                left, right = node.left, node.right
                # Check if both children are univariate NN atoms
                if (
                    isinstance(left, AtomNode)
                    and isinstance(right, AtomNode)
                    and str(left.kind).lower() == "nn"
                    and str(right.kind).lower() == "nn"
                    and effective_arity(left) == 1
                    and effective_arity(right) == 1
                    # This rule is defined on *global* 1D axes; nontrivial univariate NNs
                    # do not satisfy that assumption (their input is z=input_expr(x)).
                    and (not has_nontrivial_input(left))
                    and (not has_nontrivial_input(right))
                    and left.var_idxs[0] != right.var_idxs[0]
                ):
                    targets.append((node, left, right))
                # Recurse into children
                _find_additive_pairs(left)
                _find_additive_pairs(right)
            elif isinstance(node, MulNode):
                _find_additive_pairs(node.left)
                _find_additive_pairs(node.right)
            elif isinstance(node, PowNode):
                _find_additive_pairs(node.base)
            elif isinstance(node, _UNARY_AST_NODES):
                _find_additive_pairs(node.arg)

        _find_additive_pairs(ctx.state.root)
        return targets

    def propose(self, ctx: StageBContext, target) -> List[Candidate]:
        """
        Test if both NN atoms have log signatures and propose a polylog rewrite.
        """
        if not isinstance(target, tuple) or len(target) != 3:
            return []

        add_node, left_atom, right_atom = target

        if not isinstance(add_node, AddNode):
            return []
        if not isinstance(left_atom, AtomNode) or not isinstance(right_atom, AtomNode):
            return []

        st = ctx.state
        axisA = int(left_atom.var_idxs[0])
        axisB = int(right_atom.var_idxs[0])

        ctx.log(
            f"[Stage B] Testing log-ratio pattern for AddNode: "
            f"NN[x{axisA}] + NN[x{axisB}]"
        )

        # Get data batch
        try:
            X_batch = next(iter(ctx.train_loader_probe))[0].to(ctx.device, ctx.dtype)[:1024]
        except Exception as e:
            ctx.log(f"[Stage B] log_ratio: failed to get data batch: {e}")
            return []

        # Get leaf models for the atoms
        try:
            atom_to_leaf = build_atom_to_leaf_map(st.root, st.model)
            leaf_A = atom_to_leaf.get(id(left_atom), None)
            leaf_B = atom_to_leaf.get(id(right_atom), None)
            if leaf_A is None or leaf_B is None:
                ctx.log("[Stage B] log_ratio: could not find leaf models")
                return []
        except Exception as e:
            ctx.log(f"[Stage B] log_ratio: failed to get leaf models: {e}")
            return []

        # Evaluate atoms
        try:
            xA = X_batch[:, axisA : axisA + 1]  # [B, 1]
            xB = X_batch[:, axisB : axisB + 1]  # [B, 1]

            # Check for positive values (required for log)
            if (xA <= 0).any() or (xB <= 0).any():
                ctx.log("[Stage B] log_ratio: negative values in x - log pattern unlikely")
                return []

            # Get NN outputs
            yA = leaf_A(xA)  # [B, 1]
            yB = leaf_B(xB)  # [B, 1]

            if yA.dim() == 2:
                yA = yA[:, 0]
            if yB.dim() == 2:
                yB = yB[:, 0]
        except Exception as e:
            ctx.log(f"[Stage B] log_ratio: evaluation failed: {e}")
            return []

        # Check Jacobian pattern: for log(x), d/dx[log(x)] = 1/x
        # So d/dx[a*log(x)] * x = a (constant)
        try:
            cache_A = {"x": xA}
            cache_B = {"x": xB}
            grad_A = leaf_A.grad(cache_A)  # [B, 1, 1]
            grad_B = leaf_B.grad(cache_B)  # [B, 1, 1]

            if grad_A.dim() == 3:
                grad_A = grad_A[:, 0, 0]
            else:
                grad_A = grad_A.view(-1)
            if grad_B.dim() == 3:
                grad_B = grad_B[:, 0, 0]
            else:
                grad_B = grad_B.view(-1)

            # Product: dy/dx * x should be constant for log pattern
            productA = grad_A * xA.view(-1)
            productB = grad_B * xB.view(-1)

            # Check if products are nearly constant
            rel_std_A = productA.std() / productA.abs().mean().clamp(min=1e-12)
            rel_std_B = productB.std() / productB.abs().mean().clamp(min=1e-12)

            if rel_std_A > 0.15 or rel_std_B > 0.15:
                ctx.log(
                    f"[Stage B] log_ratio: Jacobian check failed - "
                    f"rel_std_A={rel_std_A:.4f}, rel_std_B={rel_std_B:.4f}"
                )
                return []

            ctx.log(
                f"[Stage B] log_ratio: Jacobian check passed - "
                f"rel_std_A={rel_std_A:.4f}, rel_std_B={rel_std_B:.4f}"
            )
        except Exception as e:
            ctx.log(f"[Stage B] log_ratio: Jacobian check failed with error: {e}")
            return []

        # Fit a*log(xA) + b*log(xB) to the sum yA + yB
        try:
            log_xA = torch.log(xA.view(-1).clamp(min=1e-12))
            log_xB = torch.log(xB.view(-1).clamp(min=1e-12))
            y_sum = yA + yB

            # Linear regression: [log(xA), log(xB), 1] @ [a, b, c] = y_sum
            Phi = torch.stack([log_xA, log_xB, torch.ones_like(log_xA)], dim=1)  # [B, 3]
            coeffs = torch.linalg.lstsq(Phi, y_sum.unsqueeze(-1)).solution.squeeze(-1)  # [3]
            a_coeff = float(coeffs[0])
            b_coeff = float(coeffs[1])
            c_coeff = float(coeffs[2])

            # Check fit quality
            y_pred = Phi @ coeffs
            residuals = y_sum - y_pred
            rms_res = residuals.pow(2).mean().sqrt()
            scale = y_sum.abs().mean().clamp(min=1e-12)
            rel_rms = rms_res / scale
            log_ratio_rel_max = _stageB_noisy_rel_rms_threshold(
                ctx,
                2.0e-2,
                y_rms=float(scale.item()),
            )

            if rel_rms > log_ratio_rel_max:
                ctx.log(
                    f"[Stage B] log_ratio: fit rejected - "
                    f"rel_rms={rel_rms:.4f} > {log_ratio_rel_max:.4f}"
                )
                return []

            ctx.log(
                f"[Stage B] log_ratio detected: {a_coeff:.4g}*log(x{axisA}) + "
                f"{b_coeff:.4g}*log(x{axisB}) + {c_coeff:.4g}, rel_rms={rel_rms:.6f}"
            )
        except Exception as e:
            ctx.log(f"[Stage B] log_ratio: fitting failed: {e}")
            return []

        cands: List[Candidate] = []

        # Dimensionless compound-log form.  If the two axes are commensurate,
        # build z = xA^pA * xB^pB with pA*dim(xA) + pB*dim(xB) = 0 and refit
        # c + k*log(z).  This is the general dimensional form behind log-ratio
        # cases, and covers xA/xB, xA/xB^2, xA^2/xB, etc.
        powers = _dimensionless_log_monomial_powers(
            axisA,
            axisB,
            getattr(ctx, "units_spec", None),
        )
        if powers is not None:
            pA, pB = powers
            try:
                log_z = float(pA) * log_xA + float(pB) * log_xB
                Phi_z = torch.stack([log_z, torch.ones_like(log_z)], dim=1)
                coeffs_z = torch.linalg.lstsq(
                    Phi_z,
                    y_sum.unsqueeze(-1),
                ).solution.squeeze(-1)
                k_z = float(coeffs_z[0])
                c_z = float(coeffs_z[1])
                y_z = Phi_z @ coeffs_z
                z_resid = y_sum - y_z
                z_rel_rms = z_resid.pow(2).mean().sqrt() / scale
            except Exception:
                z_rel_rms = torch.as_tensor(float("inf"), device=ctx.device)
                k_z = 0.0
                c_z = 0.0

            if float(z_rel_rms) <= log_ratio_rel_max and math.isfinite(float(k_z)):
                if k_z < 0.0:
                    pA = -pA
                    pB = -pB
                    k_z = -k_z
                z_tag = f"logmonomial_{axisA}_{pA}_x{axisB}_{pB}".replace("/", "o")
                z_expr = _make_log_monomial_expr(axisA, pA, axisB, pB)
                z_atom = AtomNode(
                    kind="polylog",
                    var_idxs=tuple(sorted([axisA, axisB])),
                    kwargs={"degree": 1},
                    tag=z_tag,
                    inputs=(z_expr,),
                )
                z_root = _replace_node_in_ast(st.root, add_node, z_atom)

                def _init_log_monomial_polylog(
                    root_new: Node,
                    model_new: nn.Module,
                    *,
                    _tag=z_tag,
                    _c=c_z,
                    _k=k_z,
                ):
                    try:
                        atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
                        for atom in _collect_all_atoms(root_new):
                            if (
                                isinstance(atom, AtomNode)
                                and atom.tag == _tag
                                and str(atom.kind).lower() == "polylog"
                            ):
                                leaf = atom_to_leaf_new.get(id(atom), None)
                                if leaf is None:
                                    continue
                                with torch.no_grad():
                                    if hasattr(leaf, "coeffs") and leaf.coeffs.numel() >= 2:
                                        # 1D PolyLogLeaf coeffs order: [const, log(z)]
                                        leaf.coeffs[0] = _c
                                        leaf.coeffs[1] = _k
                    except Exception:
                        pass

                cands.append(
                    Candidate(
                        self.name,
                        z_root,
                        _init_log_monomial_polylog,
                        meta={
                            "pattern": "log_ratio",
                            "log_monomial_powers": {
                                int(axisA): str(pA),
                                int(axisB): str(pB),
                            },
                            "coeffs": {"k": k_z, "c": c_z},
                            "log": (
                                f"[Stage B]  Trying dimensionless log-monomial rewrite as "
                                f"{k_z:.4g}*log({_format_log_monomial(axisA, pA, axisB, pB)}) "
                                f"+ {c_z:.4g} (rel_rms={float(z_rel_rms):.6f})"
                            ),
                        },
                    )
                )

        # General fallback: replace AddNode with PolyLogLeaf(n_in=2, degree=1).
        # This remains useful when both axes are dimensionless.  For unitful
        # axes, the units precheck should reject it unless the dimensional
        # policy explicitly permits such a form.
        var_idxs = tuple(sorted([axisA, axisB]))
        new_tag = f"logratio_{axisA}_{axisB}"

        new_atom = AtomNode(
            kind="polylog",
            var_idxs=var_idxs,
            kwargs={"degree": 1},
            tag=new_tag,
        )

        # Replace the AddNode with the new atom using functional replacement
        new_root = _replace_node_in_ast(st.root, add_node, new_atom)

        # Create init function to set coefficients
        # PolyLogLeaf(n_in=2, degree=1) has 3 coefficients: [1, log(x0), log(x1)]
        # i.e., c + a*log(x_first) + b*log(x_second) where first < second
        if axisA < axisB:
            coeff_first, coeff_second = a_coeff, b_coeff
        else:
            coeff_first, coeff_second = b_coeff, a_coeff

        def _init_polylog(
            root_new: Node,
            model_new: nn.Module,
            *,
            _tag=new_tag,
            _c=c_coeff,
            _a=coeff_first,
            _b=coeff_second,
        ):
            try:
                atom_to_leaf_new = build_atom_to_leaf_map(root_new, model_new)
                for atom in _collect_all_atoms(root_new):
                    if (
                        isinstance(atom, AtomNode)
                        and atom.tag == _tag
                        and str(atom.kind).lower() == "polylog"
                    ):
                        leaf = atom_to_leaf_new.get(id(atom), None)
                        if leaf is None:
                            continue
                        with torch.no_grad():
                            if hasattr(leaf, "coeffs") and leaf.coeffs.numel() >= 3:
                                # PolyLogLeaf coeffs order: [const, log(x0), log(x1)]
                                leaf.coeffs[0] = _c
                                leaf.coeffs[1] = _a
                                leaf.coeffs[2] = _b
            except Exception:
                pass

        cands.append(
            Candidate(
                self.name,
                new_root,
                _init_polylog,
                meta={
                    "pattern": "log_ratio",
                    "coeffs": {"a": a_coeff, "b": b_coeff, "c": c_coeff},
                    "vars": [axisA, axisB],
                },
            )
        )
        return cands




# ---------------------------------------------------------------------------
# RuleUnivariateOracleInvariants — univariate oracle-driven differential templates
# ---------------------------------------------------------------------------


class RuleUnivariateOracleInvariants(StageBRule):
    """Univariate counterfactor/counterterm templates driven by differential identities.

    This rule intentionally avoids FFT-based frequency discovery.

    It treats the trained NN leaf as a high-accuracy *oracle* and evaluates
    y(z), y'(z) and (when available) y''(z) on a clean uniform grid in the
    atom's effective 1D coordinate z.  It then fits small parametric families
    by least squares and proposes analytic rewrites when the differential
    residuals are convincingly small.

    Targeted families (in approximate priority order):
      - ``exp(mu*z) * sin(omega*z + phi)`` via constant-coefficient ODE
      - Logistic curve ``y0 + A/(1 + exp(-(k*z + b)))`` via quadratic y' law
      - Chirp ``A*sin(a*z^2 + b*z + c)`` via ratio invariant
      - Product ``A*sin(quadratic) * cos(omega*z + phi)`` via scanning (omega,phi)
      - Low-harmonic trig sum ``a1*sin(omega*z+phi1) + a3*sin(3*omega*z+phi3)``

    Notes
    -----
    - This rule is meant for the hard univariate leaves where RuleUniNN's
      generic templates and RuleUnivariateMulPeel's log-derivative probes
      can struggle (e.g. exp*trig products, chirps, logistic).
    - The proposals are *initialised* from closed-form fits; LM then refines
      all parameters jointly.
    """

    name = "univariate_oracle_invariants"
    exhaustive = True
    multi_probe_native = True

    def iter_targets(self, ctx: StageBContext):
        return _collect_univariate_nn_atoms(ctx.state.root)

    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if (
            not isinstance(target, AtomNode)
            or str(getattr(target, "kind", "")).lower() != "nn"
            or effective_arity(target) != 1
        ):
            return []

        st = ctx.state
        tag = getattr(target, "tag", None)
        if tag is None:
            return []
        teacher = st.reuse.get(tag, None)
        if teacher is None:
            return []

        oracle = self._eval_teacher_on_uniform_grid_1d(
            teacher,
            ctx.train_loader_probe,
            target,
            device=ctx.device,
            dtype=ctx.dtype,
            n_grid=768,
        )
        if oracle is None:
            return []

        z, y, y1, y2 = oracle
        if z.numel() < 128:
            return []

        cands: List[Candidate] = []

        # 1) exp(mu*z) * sin(omega*z + phi) via constant-coefficient ODE
        try:
            cand = self._cand_exp_trig_from_ccode(ctx, target, z, y, y1, y2)
            if cand is not None:
                cands.append(cand)
        except Exception:
            pass

        # 2) logistic: y0 + A/(1 + exp(-(k*z + b))) via y' quadratic law
        try:
            cand = self._cand_logistic_from_quadratic(ctx, target, z, y, y1)
            if cand is not None:
                cands.append(cand)
        except Exception:
            pass

        # 3) product: A*sin(quadratic) * cos(omega*z + phi) via scanning omega,phi
        try:
            cand = self._cand_chirp_cos_product(ctx, target, z, y, y1)
            if cand is not None:
                cands.append(cand)
        except Exception:
            pass

        # 4) chirp: A*sin(a*z^2 + b*z + c) via ratio invariant
        try:
            cand = self._cand_chirp_sin_quadratic(ctx, target, z, y, y1)
            if cand is not None:
                cands.append(cand)
        except Exception:
            pass

        # 5) low-harmonic trig sum: a1*sin(w*z+phi1) + a3*sin(3*w*z+phi3)
        try:
            cand = self._cand_trig_harmonics_1_3(ctx, target, z, y)
            if cand is not None:
                cands.append(cand)
        except Exception:
            pass

        return cands

    # ------------------------------------------------------------------
    # uniform-grid oracle eval (y, y', y'')
    # ------------------------------------------------------------------
    @staticmethod
    def _eval_teacher_on_uniform_grid_1d(
        teacher: nn.Module,
        train_loader,
        target: AtomNode,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_grid: int = 512,
        max_points: int = 20000,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Evaluate y(z), y'(z), y''(z) for a univariate atom on a clean grid.

        Returns CPU tensors (z_grid, y, y1, y2).  If teacher.grad_grad() is not
        available, y2 is estimated by second finite differences on y.
        """
        z_all: List[torch.Tensor] = []
        n_tot = 0
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device=device, dtype=dtype)
            z = _build_atom_input_tensor(target, x)  # [B, 1]
            z_all.append(z.detach().cpu().view(-1))
            n_tot += int(z.numel())
            if n_tot >= int(max_points):
                break
        if not z_all:
            return None
        z_cat = torch.cat(z_all)
        z_cat = z_cat[torch.isfinite(z_cat)]
        if z_cat.numel() < 16:
            return None

        z_min = float(z_cat.min().item())
        z_max = float(z_cat.max().item())
        if (not math.isfinite(z_min)) or (not math.isfinite(z_max)) or (z_max - z_min) < 1e-12:
            return None

        z_grid = torch.linspace(z_min, z_max, int(n_grid), dtype=dtype)
        if z_grid.numel() < 8:
            return None
        dz = float(z_grid[1].item() - z_grid[0].item())
        if not math.isfinite(dz) or abs(dz) < 1e-18:
            return None

        z_in = z_grid.unsqueeze(-1).to(device=device, dtype=dtype)  # [n_grid, 1]

        teacher.eval()
        with torch.no_grad():
            y = teacher(z_in)
            if y.dim() == 2:
                y = y[:, 0]
            else:
                y = y.view(-1)

            cache = {"x": z_in}
            try:
                g = teacher.grad(cache)
            except Exception:
                return None
            if g.dim() >= 3:
                y1 = g[:, 0, 0]
            else:
                y1 = g.view(-1)

            y2 = None
            try:
                gg = teacher.grad_grad(cache)
                if gg is not None:
                    if gg.dim() >= 4:
                        y2 = gg[:, 0, 0, 0]
                    else:
                        y2 = gg.view(-1)
            except Exception:
                y2 = None

        y = y.detach().cpu()
        y1 = y1.detach().cpu()
        if y2 is not None:
            y2 = y2.detach().cpu()
        else:
            # Second finite-difference on y (uniform grid)
            y2 = torch.zeros_like(y)
            if y.numel() >= 3:
                y2[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (dz * dz)
                y2[0] = y2[1]
                y2[-1] = y2[-2]
            else:
                return None

        m = torch.isfinite(z_grid) & torch.isfinite(y) & torch.isfinite(y1) & torch.isfinite(y2)
        if int(m.sum().item()) < 128:
            return None

        return z_grid[m], y[m], y1[m], y2[m]

    # ------------------------------------------------------------------
    # small linear-algebra helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return (x * x).mean().sqrt()

    @staticmethod
    def _lstsq(Phi: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Least-squares solve Phi @ w ≈ y with a safe fallback."""
        try:
            sol = torch.linalg.lstsq(Phi, y).solution
            return sol
        except Exception:
            # Normal equations with tiny ridge
            PhiT = Phi.transpose(0, 1)
            ATA = PhiT @ Phi
            eye = torch.eye(ATA.shape[0], dtype=ATA.dtype, device=ATA.device)
            ATy = PhiT @ y
            return torch.linalg.solve(ATA + 1e-12 * eye, ATy)

    @staticmethod
    def _try_quantile_abs(y: torch.Tensor, q: float) -> float:
        try:
            return float(torch.quantile(y.abs(), q).item())
        except Exception:
            yn = y.detach().cpu().numpy()
            return float(np.quantile(np.abs(yn), q))

    # ------------------------------------------------------------------
    # Candidate: exp(mu*z) * sin(omega*z + phi)
    # ------------------------------------------------------------------
    def _cand_exp_trig_from_ccode(
        self,
        ctx: StageBContext,
        target: AtomNode,
        z: torch.Tensor,
        y: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
    ) -> Optional[Candidate]:
        # Fit y'' + a*y' + b*y = 0
        if z.numel() < 256:
            return None

        Phi = torch.stack([y1, y], dim=1)  # [N,2]
        rhs = (-y2).unsqueeze(-1)  # [N,1]
        sol = self._lstsq(Phi, rhs).view(-1)
        if sol.numel() < 2:
            return None
        a = float(sol[0].item())
        b = float(sol[1].item())

        res = y2 + a * y1 + b * y
        rel_ode = float(self._rms(res) / (self._rms(y2) + 1e-12))
        if (not math.isfinite(rel_ode)) or rel_ode > 5.0e-2:
            return None

        mu = -0.5 * a
        omega2 = b - (mu * mu)
        if (not math.isfinite(mu)) or (not math.isfinite(omega2)) or omega2 <= 1e-12:
            return None
        omega = float(math.sqrt(omega2))

        span = float((z.max() - z.min()).item())
        if span <= 0:
            return None
        n_cycles = omega * span / (2.0 * math.pi)
        if n_cycles < 1.0:
            return None

        # Demodulate exponential and fit sin/cos
        y_demod = y * torch.exp(torch.tensor(-mu, dtype=y.dtype) * z)
        S = torch.sin(torch.tensor(omega, dtype=y.dtype) * z)
        C = torch.cos(torch.tensor(omega, dtype=y.dtype) * z)
        Phi2 = torch.stack([S, C], dim=1)
        sol2 = self._lstsq(Phi2, y_demod.unsqueeze(-1)).view(-1)
        if sol2.numel() < 2:
            return None
        p = float(sol2[0].item())
        q = float(sol2[1].item())

        y_hat = (Phi2 @ sol2.unsqueeze(-1)).view(-1)
        rel_fit = float(self._rms(y_demod - y_hat) / (self._rms(y_demod) + 1e-12))
        if (not math.isfinite(rel_fit)) or rel_fit > 5.0e-2:
            return None

        amp = float(math.sqrt(p * p + q * q))
        if (not math.isfinite(amp)) or amp < 1e-10:
            return None
        phi = float(math.atan2(q, p))

        st = ctx.state
        base_tag = str(getattr(target, "tag", ""))
        tag_exp = f"{base_tag}_uinv_exp"
        tag_sin = f"{base_tag}_uinv_sin"

        exp_atom = AtomNode(
            kind="exp_poly",
            var_idxs=target.var_idxs,
            inputs=clone_inputs(target),
            kwargs={"degree": 1},
            tag=tag_exp,
        )
        sin_atom = AtomNode(
            kind="sin_linear",
            var_idxs=target.var_idxs,
            inputs=clone_inputs(target),
            kwargs={},
            tag=tag_sin,
        )
        new_sub = MulNode(exp_atom, sin_atom)
        new_root = replace_atom_in_ast(st.root, target, new_sub)
        if new_root is None:
            return None

        def _init(root_new: Node, model_new: nn.Module, *, _mu=mu, _omega=omega, _amp=amp, _phi=phi):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for atom in _collect_all_atoms(root_new):
                if not isinstance(atom, AtomNode):
                    continue
                if atom.tag == tag_exp:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is None:
                        continue
                    try:
                        _poly_zero_and_set(leaf, {(0,): 0.0, (1,): float(_mu)})
                    except Exception:
                        pass
                if atom.tag == tag_sin:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is None:
                        continue
                    leaf = getattr(leaf, 'model', leaf)  # unwrap AutogradAdaptor
                    try:
                        with torch.no_grad():
                            if hasattr(leaf, "weight") and leaf.weight.numel() >= 1:
                                leaf.weight.zero_()
                                leaf.weight.view(-1)[0] = float(_omega)
                            if hasattr(leaf, "bias"):
                                leaf.bias.fill_(float(_phi))
                            if hasattr(leaf, "amp"):
                                leaf.amp.fill_(float(_amp))
                    except Exception:
                        pass

        _init._after_analytic_init = True
        log = (
            f"[Stage B]  Trying exp*trig(ODE) on NN vars={target.var_idxs}: "
            f"mu≈{mu:.4g}, omega≈{omega:.4g}, rel_ode={rel_ode:.2e}, rel_fit={rel_fit:.2e}"
        )
        return Candidate(
            "exp_trig_ode",
            new_root,
            _init,
            meta={"log": log, "mu": mu, "omega": omega, "rel_ode": rel_ode, "rel_fit": rel_fit},
        )

    # ------------------------------------------------------------------
    # Candidate: logistic y0 + A/(1 + exp(-(k*z + b)))
    # ------------------------------------------------------------------
    def _cand_logistic_from_quadratic(
        self,
        ctx: StageBContext,
        target: AtomNode,
        z: torch.Tensor,
        y: torch.Tensor,
        y1: torch.Tensor,
    ) -> Optional[Candidate]:
        if z.numel() < 256:
            return None

        # Monotonicity gate: logistic should have mostly one sign for y'
        sgn = torch.sign(y1)
        frac_pos = float((sgn > 0).float().mean().item())
        frac_neg = float((sgn < 0).float().mean().item())
        if max(frac_pos, frac_neg) < 0.80:
            return None

        # Robust baseline + amplitude estimate
        try:
            y_lo = float(torch.quantile(y, 0.02).item())
            y_hi = float(torch.quantile(y, 0.98).item())
        except Exception:
            yn = y.detach().cpu().numpy()
            y_lo = float(np.quantile(yn, 0.02))
            y_hi = float(np.quantile(yn, 0.98))

        A0 = float(y_hi - y_lo)
        if (not math.isfinite(A0)) or abs(A0) < 1e-8:
            return None

        y0 = float(y_lo)
        y_t = y - y0

        # Fit y' ≈ c0 + alpha*y_t + beta*y_t^2 on interior points
        # The constant term is needed when data covers only part of the
        # logistic (y_lo >> y0_true), which introduces an offset in the ODE.
        eps = 0.05 * abs(A0)
        m = torch.isfinite(y_t) & torch.isfinite(y1) & (y_t > eps) & (y_t < (A0 - eps))
        if int(m.sum().item()) < 128:
            return None

        yt = y_t[m]
        y1t = y1[m]
        Phi = torch.stack([torch.ones_like(yt), yt, yt * yt], dim=1)
        sol = self._lstsq(Phi, y1t.unsqueeze(-1)).view(-1)
        if sol.numel() < 3:
            return None
        c0 = float(sol[0].item())
        alpha = float(sol[1].item())
        beta = float(sol[2].item())

        y1_hat = (Phi @ sol.unsqueeze(-1)).view(-1)
        rel_q = float(self._rms(y1t - y1_hat) / (self._rms(y1t) + 1e-12))
        if (not math.isfinite(rel_q)) or rel_q > 5.0e-2:
            return None

        if abs(beta) < 1e-12:
            return None

        # Recover true y0 offset from the constant term.
        # The logistic ODE y' = k*(y-y0_true)*(1-(y-y0_true)/L) expanded
        # in y_t = y - y_lo gives: y' = c0 + alpha*y_t + beta*y_t^2 where
        #   d = y_lo - y0_true,  c0 = k*d*(1-d/L),  alpha = k*(1-2d/L),
        #   beta = -k/L.  Eliminating k and L yields beta*d^2 - alpha*d + c0 = 0.
        disc = alpha * alpha - 4.0 * beta * c0
        if disc < 0:
            d_offset = 0.0
        else:
            sq = math.sqrt(disc)
            d1 = (alpha + sq) / (2.0 * beta)
            d2 = (alpha - sq) / (2.0 * beta)
            cands = [d for d in (d1, d2) if math.isfinite(d) and d >= -0.05 * abs(A0)]
            d_offset = min(cands, key=lambda x: abs(x)) if cands else 0.0

        y0 = y0 - d_offset          # shift from y_lo to true lower asymptote
        y_t = y - y0                 # re-shift to true baseline

        k_from_q = alpha - 2.0 * beta * d_offset
        A_est = k_from_q / (-beta)
        if (not math.isfinite(A_est)) or abs(A_est) < 1e-6:
            return None

        # Ensure consistency: increasing logistic => k>0, beta<0
        if frac_pos > frac_neg and not (k_from_q > 0 and beta < 0 and A_est > 0):
            return None
        if frac_neg > frac_pos and not (k_from_q < 0 and beta > 0 and A_est < 0):
            return None

        # Solve for (k,b) from log(A/y - 1) ≈ -(k*z + b)
        A_use = float(A_est)
        # For decreasing logistic, A_est could be negative; flip by absorbing sign into y0
        if A_use < 0:
            # y0 + A/(1+exp(-s)) with A<0 is equivalent to (y0+A) + (-A)/(1+exp(s))
            y0 = y0 + A_use
            A_use = -A_use
            y_t = y - y0
        eps2 = 0.08 * abs(A_use)
        m2 = torch.isfinite(y_t) & (y_t > eps2) & (y_t < (A_use - eps2))
        if int(m2.sum().item()) < 64:
            return None

        z2 = z[m2]
        yt2 = y_t[m2]
        s = torch.log((A_use / yt2) - 1.0)
        Phi2 = torch.stack([z2, torch.ones_like(z2)], dim=1)
        sol2 = self._lstsq(Phi2, s.unsqueeze(-1)).view(-1)
        if sol2.numel() < 2:
            return None
        m_slope = float(sol2[0].item())
        c_int = float(sol2[1].item())

        s_hat = (Phi2 @ sol2.unsqueeze(-1)).view(-1)
        rel_lin = float(self._rms(s - s_hat) / (self._rms(s) + 1e-12))
        if (not math.isfinite(rel_lin)) or rel_lin > 1.0e-1:
            return None

        # poly(z) = m*z + c = -(k*z + b)
        k_est = -m_slope
        b_est = -c_int
        if (not math.isfinite(k_est)) or abs(k_est) < 1e-6:
            return None

        st = ctx.state
        base_tag = str(getattr(target, "tag", ""))
        tag_y0 = f"{base_tag}_uinv_y0"
        tag_A = f"{base_tag}_uinv_A"
        tag_p = f"{base_tag}_uinv_logit"

        y0_atom = AtomNode(kind="free_const", var_idxs=(), kwargs={"init": float(y0)}, tag=tag_y0)
        A_atom = AtomNode(kind="free_const", var_idxs=(), kwargs={"init": float(A_use)}, tag=tag_A)
        poly_atom = AtomNode(
            kind="poly",
            var_idxs=target.var_idxs,
            inputs=clone_inputs(target),
            kwargs={"degree": 1},
            tag=tag_p,
        )
        denom = AddNode(ConstNode(1.0), ExpNode(poly_atom))
        inv = PowNode(denom, exponent=-1.0)
        new_sub = AddNode(y0_atom, MulNode(A_atom, inv))
        new_root = replace_atom_in_ast(st.root, target, new_sub)
        if new_root is None:
            return None

        def _init(root_new: Node, model_new: nn.Module, *, _y0=y0, _A=A_use, _m=m_slope, _c=c_int):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for atom in _collect_all_atoms(root_new):
                if not isinstance(atom, AtomNode):
                    continue
                if atom.tag == tag_y0:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is not None:
                        try:
                            _set_constant_leaf_value(leaf, float(_y0))
                        except Exception:
                            pass
                if atom.tag == tag_A:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is not None:
                        try:
                            _set_constant_leaf_value(leaf, float(_A))
                        except Exception:
                            pass
                if atom.tag == tag_p:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is None:
                        continue
                    try:
                        _poly_zero_and_set(leaf, {(0,): float(_c), (1,): float(_m)})
                    except Exception:
                        pass

        _init._after_analytic_init = True
        log = (
            f"[Stage B]  Trying logistic(ODE) on NN vars={target.var_idxs}: "
            f"A≈{A_use:.4g}, k≈{k_est:.4g}, b≈{b_est:.4g}, rel_q={rel_q:.2e}, rel_lin={rel_lin:.2e}"
        )
        return Candidate(
            "logistic_ode",
            new_root,
            _init,
            meta={
                "log": log,
                "A": float(A_use),
                "k": float(k_est),
                "b": float(b_est),
                "y0": float(y0),
                "rel_q": rel_q,
                "rel_lin": rel_lin,
            },
        )

    # ------------------------------------------------------------------
    # Candidate: chirp A*sin(a*z^2 + b*z + c) via ratio invariant
    # ------------------------------------------------------------------
    def _chirp_fit_from_ratio_invariant(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        y1: torch.Tensor,
        *,
        min_points: int = 160,
        max_rel_poly: float = 5.0e-2,
        max_rel_trig: float = 5.0e-2,
        require_quadratic: bool = True,
    ) -> Optional[Dict[str, float]]:
        """Fit chirp parameters from (y,y') using g'^2 ratio invariant.

        Returns dict with keys: amp, a2, a1, a0, rel_poly, rel_trig.
        The phase polynomial is a2*z^2 + a1*z + a0.
        """
        if z.numel() < min_points:
            return None

        A0 = self._try_quantile_abs(y, 0.995)
        if (not math.isfinite(A0)) or A0 < 1e-8:
            return None

        denom = (A0 * A0) - (y * y)
        m = torch.isfinite(denom) & (denom > (0.05 * A0) ** 2) & torch.isfinite(y1)
        if int(m.sum().item()) < min_points:
            return None

        zz = z[m]
        rr = (y1[m] * y1[m]) / denom[m]
        rr = rr[torch.isfinite(rr)]
        zz = zz[: rr.numel()]
        if zz.numel() < min_points:
            return None

        # Fit rr ≈ c2*z^2 + c1*z + c0
        Phi = torch.stack([zz * zz, zz, torch.ones_like(zz)], dim=1)
        sol = self._lstsq(Phi, rr.unsqueeze(-1)).view(-1)
        if sol.numel() < 3:
            return None
        c2 = float(sol[0].item())
        c1 = float(sol[1].item())
        c0 = float(sol[2].item())

        rr_hat = (Phi @ sol.unsqueeze(-1)).view(-1)
        rel_poly = float(self._rms(rr - rr_hat) / (self._rms(rr) + 1e-12))
        if (not math.isfinite(rel_poly)) or rel_poly > max_rel_poly:
            return None

        if require_quadratic and abs(c2) < 1e-6:
            return None
        if c2 <= 0:
            return None

        # Perfect square test: (m*z + n)^2 => discriminant ~ 0
        # Include c2^2 in denom_d so near-pure-monomial cases (c1≈0, c0≈0)
        # are not spuriously rejected (otherwise ratio ≈ 1 for any tiny c0).
        disc = (c1 * c1) - (4.0 * c0 * c2)
        denom_d = max(1e-12, (c1 * c1) + abs(4.0 * c0 * c2) + (c2 * c2))
        if abs(disc) / denom_d > 0.10:
            return None

        mlin = math.sqrt(max(0.0, c2))
        if mlin < 1e-8:
            return None
        nlin = c1 / (2.0 * mlin)

        # Phase without constant: g0 = 0.5*m*z^2 + n*z
        g0 = (0.5 * mlin) * (z * z) + (nlin * z)

        S = torch.sin(g0)
        C = torch.cos(g0)
        Phi2 = torch.stack([S, C], dim=1)
        sol2 = self._lstsq(Phi2, y.unsqueeze(-1)).view(-1)
        if sol2.numel() < 2:
            return None
        p = float(sol2[0].item())
        q = float(sol2[1].item())

        y_hat = (Phi2 @ sol2.unsqueeze(-1)).view(-1)
        rel_trig = float(self._rms(y - y_hat) / (self._rms(y) + 1e-12))
        if (not math.isfinite(rel_trig)) or rel_trig > max_rel_trig:
            return None

        amp = float(math.sqrt(p * p + q * q))
        if (not math.isfinite(amp)) or amp < 1e-10:
            return None
        a0 = float(math.atan2(q, p))
        a1 = float(nlin)
        a2 = float(0.5 * mlin)

        return {
            "amp": amp,
            "a2": a2,
            "a1": a1,
            "a0": a0,
            "rel_poly": float(rel_poly),
            "rel_trig": float(rel_trig),
        }

    def _cand_chirp_sin_quadratic(
        self,
        ctx: StageBContext,
        target: AtomNode,
        z: torch.Tensor,
        y: torch.Tensor,
        y1: torch.Tensor,
    ) -> Optional[Candidate]:
        fit = self._chirp_fit_from_ratio_invariant(
            z,
            y,
            y1,
            min_points=200,
            max_rel_poly=5.0e-2,
            max_rel_trig=5.0e-2,
            require_quadratic=True,
        )
        if fit is None:
            return None

        amp = float(fit["amp"])
        a2 = float(fit["a2"])
        a1 = float(fit["a1"])
        a0 = float(fit["a0"])
        rel_poly = float(fit["rel_poly"])
        rel_trig = float(fit["rel_trig"])

        st = ctx.state
        base_tag = str(getattr(target, "tag", ""))
        tag_A = f"{base_tag}_uinv_amp"
        tag_p = f"{base_tag}_uinv_phase2"

        A_atom = AtomNode(kind="free_const", var_idxs=(), kwargs={"init": float(amp)}, tag=tag_A)
        p_atom = AtomNode(
            kind="poly",
            var_idxs=target.var_idxs,
            inputs=clone_inputs(target),
            kwargs={"degree": 2, "min_total": 0},
            tag=tag_p,
        )
        new_sub = MulNode(A_atom, SinNode(p_atom))
        new_root = replace_atom_in_ast(st.root, target, new_sub)
        if new_root is None:
            return None

        def _init(root_new: Node, model_new: nn.Module, *, _amp=amp, _a2=a2, _a1=a1, _a0=a0):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for atom in _collect_all_atoms(root_new):
                if not isinstance(atom, AtomNode):
                    continue
                if atom.tag == tag_A:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is not None:
                        try:
                            _set_constant_leaf_value(leaf, float(_amp))
                        except Exception:
                            pass
                if atom.tag == tag_p:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is None:
                        continue
                    try:
                        _poly_zero_and_set(
                            leaf,
                            {(0,): float(_a0), (1,): float(_a1), (2,): float(_a2)},
                        )
                    except Exception:
                        pass

        _init._after_analytic_init = True
        log = (
            f"[Stage B]  Trying chirp-sin(z^2) on NN vars={target.var_idxs}: "
            f"a2≈{a2:.4g}, a1≈{a1:.4g}, amp≈{amp:.4g}, rel_poly={rel_poly:.2e}, rel_trig={rel_trig:.2e}"
        )
        return Candidate(
            "chirp_sin_quad",
            new_root,
            _init,
            meta={"log": log, **fit},
        )

    # ------------------------------------------------------------------
    # Candidate: A*sin(quadratic) * cos(omega*z + phi)
    # ------------------------------------------------------------------
    def _cand_chirp_cos_product(
        self,
        ctx: StageBContext,
        target: AtomNode,
        z: torch.Tensor,
        y: torch.Tensor,
        y1: torch.Tensor,
    ) -> Optional[Candidate]:
        """Fit A*sin(a2*z²+a1*z+a0)*cos(omega*z+phi) via multi-restart L-BFGS."""
        span = float((z.max() - z.min()).item())
        if (not math.isfinite(span)) or span <= 0:
            return None

        # Quick frequency scale estimate from derivative energy (no FFT)
        omega_est = float((self._rms(y1) / (self._rms(y) + 1e-12)).item())
        if (not math.isfinite(omega_est)) or omega_est <= 0:
            omega_est = 1.0

        w_min = max(0.1, 0.3 * omega_est)
        w_max = min(25.0, 3.0 * omega_est)
        if w_max <= w_min:
            w_min, w_max = 0.1, 10.0

        # Ensure we cover at least a couple cycles if possible
        if w_max * span / (2.0 * math.pi) < 1.0:
            w_max = min(50.0, max(w_max, (2.0 * math.pi) / max(1e-9, span)))

        # Amplitude estimate
        amp_est = float(self._try_quantile_abs(y, 0.995))
        if (not math.isfinite(amp_est)) or amp_est < 1e-10:
            return None
        rms_y = float(self._rms(y).item())
        if rms_y < 1e-12:
            return None

        z_mid = float(((z.max() + z.min()) / 2.0).item())

        # --- Multi-restart L-BFGS on A*sin(a2*z^2+a1*z+a0)*cos(omega*z+phi) ---
        zt = z.detach().clone()
        yt = y.detach().clone()

        best_loss = float("inf")
        best_vals = None

        start_omegas = np.linspace(w_min, w_max, 12).tolist()
        start_phases = [0.0, 0.5 * math.pi]

        for w0 in start_omegas:
            # Estimate chirp rate: remaining frequency after subtracting cos part
            a2_est = max(0.01, (omega_est - w0) / (2.0 * max(abs(z_mid), 0.1)))
            for ph0 in start_phases:
                p = torch.tensor(
                    [amp_est, a2_est, 0.0, 0.0, w0, ph0],
                    dtype=zt.dtype,
                ).requires_grad_(True)
                opt = torch.optim.LBFGS(
                    [p], lr=1.0, max_iter=20, line_search_fn="strong_wolfe",
                )
                for _ in range(5):
                    def closure():
                        opt.zero_grad()
                        phase_s = p[1] * zt * zt + p[2] * zt + p[3]
                        phase_c = p[4] * zt + p[5]
                        y_hat = p[0] * torch.sin(phase_s) * torch.cos(phase_c)
                        loss = torch.mean((y_hat - yt) ** 2)
                        loss.backward()
                        return loss
                    opt.step(closure)

                with torch.no_grad():
                    phase_s = p[1] * zt * zt + p[2] * zt + p[3]
                    phase_c = p[4] * zt + p[5]
                    y_hat = p[0] * torch.sin(phase_s) * torch.cos(phase_c)
                    loss = float(torch.mean((y_hat - yt) ** 2).item())

                if math.isfinite(loss) and loss < best_loss:
                    best_loss = loss
                    best_vals = p.detach().clone()

        if best_vals is None:
            return None

        rel_rms = math.sqrt(best_loss) / rms_y
        max_rel = _stageB_noisy_rel_rms_threshold(ctx, 5.0e-2, y_rms=rms_y)
        if (not math.isfinite(rel_rms)) or rel_rms > max_rel:
            return None

        amp = float(best_vals[0].item())
        a2 = float(best_vals[1].item())
        a1 = float(best_vals[2].item())
        a0 = float(best_vals[3].item())
        omega = float(best_vals[4].item())
        phi = float(best_vals[5].item())

        # Require non-trivial chirp (otherwise other rules handle trig products)
        if abs(a2) < 1e-4:
            return None

        st = ctx.state
        base_tag = str(getattr(target, "tag", ""))
        tag_A = f"{base_tag}_uinv_amp"
        tag_ps = f"{base_tag}_uinv_phase2"
        tag_pc = f"{base_tag}_uinv_phase1"

        A_atom = AtomNode(kind="free_const", var_idxs=(), kwargs={"init": float(amp)}, tag=tag_A)
        p_sin = AtomNode(
            kind="poly",
            var_idxs=target.var_idxs,
            inputs=clone_inputs(target),
            kwargs={"degree": 2, "min_total": 0},
            tag=tag_ps,
        )
        p_cos = AtomNode(
            kind="poly",
            var_idxs=target.var_idxs,
            inputs=clone_inputs(target),
            kwargs={"degree": 1, "min_total": 0},
            tag=tag_pc,
        )

        new_sub = MulNode(A_atom, MulNode(SinNode(p_sin), CosNode(p_cos)))
        new_root = replace_atom_in_ast(st.root, target, new_sub)
        if new_root is None:
            return None

        def _init(root_new: Node, model_new: nn.Module, *, _amp=amp, _a2=a2, _a1=a1, _a0=a0, _omega=omega, _phi=phi):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for atom in _collect_all_atoms(root_new):
                if not isinstance(atom, AtomNode):
                    continue
                if atom.tag == tag_A:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is not None:
                        try:
                            _set_constant_leaf_value(leaf, float(_amp))
                        except Exception:
                            pass
                if atom.tag == tag_ps:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is not None:
                        try:
                            _poly_zero_and_set(leaf, {(0,): float(_a0), (1,): float(_a1), (2,): float(_a2)})
                        except Exception:
                            pass
                if atom.tag == tag_pc:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is not None:
                        try:
                            _poly_zero_and_set(leaf, {(0,): float(_phi), (1,): float(_omega)})
                        except Exception:
                            pass

        _init._after_analytic_init = True
        log = (
            f"[Stage B]  Trying chirp*cos product on NN vars={target.var_idxs}: "
            f"omega≈{omega:.4g}, a2≈{a2:.4g}, rel_rms={rel_rms:.2e}"
        )
        meta = {"amp": amp, "a2": a2, "a1": a1, "a0": a0, "omega": omega, "phi": phi}
        meta["log"] = log
        meta["rel_rms"] = float(rel_rms)
        return Candidate("chirp_cos_prod", new_root, _init, meta=meta)

    # ------------------------------------------------------------------
    # Candidate: harmonic trig sum a1*sin(wz+φ1)+a3*sin(3wz+φ3)
    # ------------------------------------------------------------------
    def _cand_trig_harmonics_1_3(
        self,
        ctx: StageBContext,
        target: AtomNode,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> Optional[Candidate]:
        if z.numel() < 256:
            return None

        # Frequency scale estimate from zero crossings (no FFT)
        span = float((z.max() - z.min()).item())
        if (not math.isfinite(span)) or span <= 0:
            return None

        # Rough omega guess from energy ratio: for pure sin, ||y'||/||y|| ≈ omega
        # Use finite diff derivative to avoid depending on teacher.grad_grad etc.
        dz = float((z[1] - z[0]).item())
        if not math.isfinite(dz) or abs(dz) < 1e-18:
            return None
        y_fd = torch.zeros_like(y)
        y_fd[1:-1] = (y[2:] - y[:-2]) / (2.0 * dz)
        y_fd[0] = (y[1] - y[0]) / dz
        y_fd[-1] = (y[-1] - y[-2]) / dz
        omega_est = float((self._rms(y_fd) / (self._rms(y) + 1e-12)).item())
        if (not math.isfinite(omega_est)) or omega_est <= 0:
            omega_est = 1.0

        w_min = max(0.1, 0.3 * omega_est)
        w_max = min(25.0, 3.0 * omega_est)
        if w_max <= w_min:
            w_min, w_max = 0.1, 10.0

        # Scan omegas; score by relative RMS residual
        omegas = np.linspace(w_min, w_max, 81)
        best = None
        best_rel = float("inf")

        for w in omegas:
            w = float(w)
            ang1 = w * z
            ang3 = (3.0 * w) * z
            Phi = torch.stack(
                [torch.sin(ang1), torch.cos(ang1), torch.sin(ang3), torch.cos(ang3)],
                dim=1,
            )
            sol = self._lstsq(Phi, y.unsqueeze(-1)).view(-1)
            y_hat = (Phi @ sol.unsqueeze(-1)).view(-1)
            rel = float(self._rms(y - y_hat) / (self._rms(y) + 1e-12))
            if math.isfinite(rel) and rel < best_rel:
                best_rel = rel
                best = {"omega": w, "coef": sol.detach().cpu()}

        if best is None or (not math.isfinite(best_rel)) or best_rel > 3.0e-2:
            return None

        w = float(best["omega"])
        sol = best["coef"].view(-1)
        if sol.numel() < 4:
            return None
        p1, q1, p3, q3 = [float(sol[i].item()) for i in range(4)]

        amp1 = float(math.sqrt(p1 * p1 + q1 * q1))
        amp3 = float(math.sqrt(p3 * p3 + q3 * q3))
        if amp1 < 1e-10 and amp3 < 1e-10:
            return None
        phi1 = float(math.atan2(q1, p1))
        phi3 = float(math.atan2(q3, p3))

        st = ctx.state
        base_tag = str(getattr(target, "tag", ""))
        tag1 = f"{base_tag}_uinv_s1"
        tag3 = f"{base_tag}_uinv_s3"

        s1 = AtomNode(
            kind="sin_linear",
            var_idxs=target.var_idxs,
            inputs=clone_inputs(target),
            kwargs={},
            tag=tag1,
        )
        s3 = AtomNode(
            kind="sin_linear",
            var_idxs=target.var_idxs,
            inputs=clone_inputs(target),
            kwargs={},
            tag=tag3,
        )
        new_sub = AddNode(s1, s3)
        new_root = replace_atom_in_ast(st.root, target, new_sub)
        if new_root is None:
            return None

        def _init(root_new: Node, model_new: nn.Module, *, _w=w, _amp1=amp1, _phi1=phi1, _amp3=amp3, _phi3=phi3):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for atom in _collect_all_atoms(root_new):
                if not isinstance(atom, AtomNode):
                    continue
                if atom.tag == tag1:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is not None:
                        leaf = getattr(leaf, 'model', leaf)  # unwrap AutogradAdaptor
                        try:
                            with torch.no_grad():
                                if hasattr(leaf, "weight") and leaf.weight.numel() >= 1:
                                    leaf.weight.zero_(); leaf.weight.view(-1)[0] = float(_w)
                                if hasattr(leaf, "bias"):
                                    leaf.bias.fill_(float(_phi1))
                                if hasattr(leaf, "amp"):
                                    leaf.amp.fill_(float(_amp1))
                        except Exception:
                            pass
                if atom.tag == tag3:
                    leaf = atom_to_leaf.get(id(atom), None)
                    if leaf is not None:
                        leaf = getattr(leaf, 'model', leaf)  # unwrap AutogradAdaptor
                        try:
                            with torch.no_grad():
                                if hasattr(leaf, "weight") and leaf.weight.numel() >= 1:
                                    leaf.weight.zero_(); leaf.weight.view(-1)[0] = float(3.0 * _w)
                                if hasattr(leaf, "bias"):
                                    leaf.bias.fill_(float(_phi3))
                                if hasattr(leaf, "amp"):
                                    leaf.amp.fill_(float(_amp3))
                        except Exception:
                            pass

        _init._after_analytic_init = True
        log = (
            f"[Stage B]  Trying trig harmonics (1&3) on NN vars={target.var_idxs}: "
            f"omega≈{w:.4g}, rel={best_rel:.2e}"
        )
        return Candidate(
            "trig_harmonics_1_3",
            new_root,
            _init,
            meta={"log": log, "omega": w, "rel": float(best_rel), "amp1": amp1, "amp3": amp3},
        )

# ---------------------------------------------------------------------------
# RuleUnivariateMulPeel — univariate multiplicative decomposition
# ---------------------------------------------------------------------------


class RuleUnivariateMulPeel(StageBRule):
    """Decompose a univariate NN atom that is a product of different families.

    Target pattern: ``f(z) = exp(a*z) * sin(b*z + phi)`` and similar products.

    **Detection** uses log-derivative analysis:
    ``L(z) = f'(z)/f(z)`` decomposes additively when ``f`` is a product.
    A constant ``mu`` in ``L`` signals an exponential factor; a dominant
    oscillation in ``L - mu`` signals a trigonometric factor.

    Pattern labels: ``exp_trig_peel``, ``exp_peel_nn_resid``,
    ``mono_trig_peel``, ``mono_exp_peel``, ``mono_exp_trig_peel``,
    ``mono_peel_nn_resid``
    """

    name = "univariate_mul_peel"

    def iter_targets(self, ctx: StageBContext):
        """Return all univariate (1D) NN atoms in the current AST."""
        return _collect_univariate_nn_atoms(ctx.state.root)

    # ------------------------------------------------------------------
    # propose
    # ------------------------------------------------------------------
    def propose(self, ctx: StageBContext, target: Node) -> List[Candidate]:
        if (
            not isinstance(target, AtomNode)
            or str(target.kind).lower() != "nn"
            or effective_arity(target) != 1
        ):
            return []

        st = ctx.state
        tag = target.tag
        if tag is None or tag not in st.reuse:
            return []

        teacher = st.reuse[tag]

        # ---- Phase 0: evaluate teacher on clean uniform grid (NN oracle) ------
        oracle = self._eval_teacher_on_uniform_grid(
            teacher, ctx.train_loader_probe, target, ctx.device, ctx.dtype,
        )
        if oracle is None:
            return []
        z_grid, f_vals, fp_vals = oracle  # all 1-D torch tensors on CPU

        z_np = z_grid.numpy()
        f_np = f_vals.numpy()
        fp_np = fp_vals.numpy()

        # ---- Phase 1: monomial detection ---------------------------------
        k_result = self._estimate_monomial_power(z_np, f_np, fp_np)
        k, mu_envelope = k_result if k_result[0] is not None else (0.0, None)

        # ---- Phase 2: exponential detection (corrected for monomial) --
        # When envelope fitting was used (oscillatory data), trust its mu
        # estimate — the median-L approach is biased by trig divergences.
        if mu_envelope is not None:
            mu = mu_envelope if abs(mu_envelope) >= 0.05 else 0.0
        else:
            mu = self._estimate_exp_rate(f_np, fp_np, z=z_np, k=k)
            if mu is None or abs(mu) < 0.01:
                mu = 0.0

        # If neither monomial nor exp detected, nothing to do
        if k == 0.0 and mu == 0.0:
            return []

        # ---- Phase 3: trig detection on corrected residual ------------
        from nestynet_sr.sr_search.features import discover_trig_from_data

        # Divide out detected factors
        residual = f_vals.clone()
        if k != 0.0:
            z_safe = z_grid.clamp_min(1e-8)
            residual = residual / z_safe.pow(k)
        if mu != 0.0:
            residual = residual * torch.exp(-mu * z_grid)

        trig_spec = discover_trig_from_data(
            z_grid, residual,
            strength_threshold=5.0,
            max_omega=1000.0,
        )
        omega = None
        if trig_spec is not None:
            # Require at least 2 full cycles to avoid false positives
            z_span = float(z_grid[-1] - z_grid[0])
            n_cycles = trig_spec.omega * z_span / (2.0 * math.pi)
            if n_cycles >= 2.0:
                omega = float(trig_spec.omega)
            else:
                trig_spec = None

        has_mono = k != 0.0
        has_exp = mu != 0.0
        has_trig = omega is not None

        # Log what was detected
        parts = []
        if has_mono:
            parts.append(f"k={k:.2g}")
        if has_exp:
            parts.append(f"mu={mu:.4g}")
        if has_trig:
            parts.append(f"omega={omega:.4g}")
        ctx.log(
            f"[Stage B] RuleUnivariateMulPeel: detected [{'+'.join(parts)}] "
            f"on NN vars={target.var_idxs}"
        )

        # ---- Phase 4: build candidates based on detected factors ------
        cands: List[Candidate] = []

        if has_mono and has_exp and has_trig:
            # mono * exp * trig — fully analytical
            root_c, init_c = self._build_mono_exp_trig_candidate(
                st.root, target, k, mu, omega,
            )
            if root_c is not None:
                cands.append(Candidate(
                    "mono_exp_trig_peel",
                    root_c,
                    init_c,
                    meta={"log": (
                        f"[Stage B]  Trying mono*exp*trig peel (k={k:.2g}, mu={mu:.4g}, "
                        f"omega={omega:.4g}) on NN leaf vars {target.var_idxs}"
                    )},
                ))

        if has_mono and has_trig and not has_exp:
            # mono * trig
            root_mt, init_mt = self._build_mono_trig_candidate(
                st.root, target, k, omega,
            )
            if root_mt is not None:
                cands.append(Candidate(
                    "mono_trig_peel",
                    root_mt,
                    init_mt,
                    meta={"log": (
                        f"[Stage B]  Trying mono*trig peel (k={k:.2g}, omega={omega:.4g}) "
                        f"on NN leaf vars {target.var_idxs}"
                    )},
                ))

        if has_mono and has_exp and not has_trig:
            # mono * exp
            root_me, init_me = self._build_mono_exp_candidate(
                st.root, target, k, mu,
            )
            if root_me is not None:
                cands.append(Candidate(
                    "mono_exp_peel",
                    root_me,
                    init_me,
                    meta={"log": (
                        f"[Stage B]  Trying mono*exp peel (k={k:.2g}, mu={mu:.4g}) "
                        f"on NN leaf vars {target.var_idxs}"
                    )},
                ))

        if not has_mono and has_exp and has_trig:
            # exp * trig (existing pattern)
            root_a, init_a = self._build_exp_trig_candidate(
                st.root, target, mu, omega,
            )
            if root_a is not None:
                cands.append(Candidate(
                    "exp_trig_peel",
                    root_a,
                    init_a,
                    meta={"log": (
                        f"[Stage B]  Trying exp*trig peel (mu={mu:.4g}, omega={omega:.4g}) "
                        f"on NN leaf vars {target.var_idxs}"
                    )},
                ))

        # ---- fallback: *_nn_resid variants (peel detected factor, keep NN) ----
        if has_mono:
            root_mn, init_mn = self._build_mono_nn_residual_candidate(
                st.root, target, k,
            )
            if root_mn is not None:
                cands.append(Candidate(
                    "mono_peel_nn_resid",
                    root_mn,
                    init_mn,
                    meta={"log": (
                        f"[Stage B]  Trying mono-peel + NN residual (k={k:.2g}) "
                        f"on NN leaf vars {target.var_idxs}"
                    )},
                ))

        if has_exp:
            root_b, init_b = self._build_exp_nn_residual_candidate(
                st.root, target, mu,
            )
            if root_b is not None:
                cands.append(Candidate(
                    "exp_peel_nn_resid",
                    root_b,
                    init_b,
                    meta={"log": (
                        f"[Stage B]  Trying exp-peel + NN residual (mu={mu:.4g}) "
                        f"on NN leaf vars {target.var_idxs}"
                    )},
                ))

        return cands

    # ------------------------------------------------------------------
    # helper: evaluate teacher NN on clean uniform grid
    # ------------------------------------------------------------------
    @staticmethod
    def _eval_teacher_on_uniform_grid(
        teacher: nn.Module,
        train_loader,
        target: AtomNode,
        device: torch.device,
        dtype: torch.dtype,
        n_grid: int = 512,
    ) -> "Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]":
        """Evaluate f(z) and f'(z) on a uniform grid spanning the training data.

        Uses the NN as an oracle: evaluates on a clean, evenly-spaced grid
        rather than the (potentially scattered) training points.

        Returns (z_grid, f_vals, fp_vals) on CPU, or None on failure.
        """
        from nestynet_sr.sr_search.candidate_builders import _build_atom_input_tensor

        # 1. Scan training data to find z range
        z_all: List[torch.Tensor] = []
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device=device, dtype=dtype)
            z = _build_atom_input_tensor(target, x)  # [B, 1]
            z_all.append(z.detach().cpu().view(-1))
        if not z_all:
            return None
        z_cat = torch.cat(z_all)
        z_min, z_max = float(z_cat.min()), float(z_cat.max())
        if z_max - z_min < 1e-10:
            return None

        # 2. Build uniform grid and evaluate teacher
        z_grid = torch.linspace(z_min, z_max, n_grid, dtype=dtype)
        z_in = z_grid.unsqueeze(-1).to(device=device)  # [n_grid, 1]

        teacher.eval()
        with torch.no_grad():
            f_vals = teacher(z_in)  # [n_grid, 1]
            if f_vals.dim() == 2:
                f_vals = f_vals[:, 0]
            cache = {"x": z_in}
            try:
                g = teacher.grad(cache)  # [n_grid, 1, 1]
            except Exception:
                return None
            fp_vals = g.view(-1)  # [n_grid]

        return z_grid.cpu(), f_vals.detach().cpu(), fp_vals.detach().cpu()

    # ------------------------------------------------------------------
    # helper: estimate exponential rate from log-derivative
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_exp_rate(
        f: np.ndarray,
        f_prime: np.ndarray,
        z: "Optional[np.ndarray]" = None,
        k: float = 0.0,
    ) -> "Optional[float]":
        """Estimate mu from L(z) = f'(z)/f(z) using median (robust to cot divergence).

        When *k* != 0 and *z* is provided, subtracts the monomial contribution
        ``k/z`` from L before computing the median, giving the corrected
        exponential rate for ``f(z) = z^k * exp(mu*z) * g(z)``.

        Returns mu or None if estimation fails.
        """
        if len(f) < 50:
            return None

        abs_f = np.abs(f)
        threshold = max(1e-8, 0.01 * float(np.median(abs_f)))
        mask = abs_f > threshold

        # When correcting for monomial, also exclude points near z=0
        if k != 0.0 and z is not None:
            z_span = float(np.max(z) - np.min(z))
            mask = mask & (np.abs(z) > 0.1 * z_span)

        if mask.sum() < 20:
            return None

        L = f_prime[mask] / f[mask]

        # Subtract monomial contribution k/z
        if k != 0.0 and z is not None:
            z_masked = z[mask]
            # Extra safety: avoid division by tiny z values
            safe = np.abs(z_masked) > 1e-10
            L = L[safe]
            z_masked = z_masked[safe]
            if len(L) < 20:
                return None
            L = L - k / z_masked

        finite = np.isfinite(L)
        if finite.sum() < 20:
            return None
        L = L[finite]

        mu = float(np.median(L))
        if not np.isfinite(mu):
            return None
        return mu

    # ------------------------------------------------------------------
    # helper: estimate monomial power (hybrid: envelope + L-fitting)
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_monomial_power(
        z: np.ndarray,
        f: np.ndarray,
        f_prime: np.ndarray,
    ) -> "Tuple[Optional[float], Optional[float]]":
        """Estimate monomial exponent k from f(z) = z^k * g(z).

        Uses two strategies:
        1. **Envelope fitting** (for oscillatory g): fit log(|f|) at local
           maxima of |f| to k*log(z) + c1*z + c0.  Works when g has trig.
        2. **L-fitting** (for smooth g): IQR-trimmed fit of L(z)=f'/f ~ k/z + c.
           Works when g is exp-like (no trig divergences).

        Returns (snapped_k, mu_envelope) where mu_envelope is from the
        envelope fit (or None).  Returns (None, None) if detection fails.
        """
        if len(z) < 50:
            return None, None

        # PowerLeaf uses z.clamp_min(eps) — only works for z > 0.
        # Skip monomial detection if >20% of data has z <= 0.
        frac_nonpositive = np.mean(z <= 0.0)
        if frac_nonpositive > 0.20:
            return None, None

        snap_tol = 0.20

        # --- Strategy 1: envelope fitting at local maxima of |f| ---
        env_result = RuleUnivariateMulPeel._estimate_k_from_envelope(z, f, snap_tol)
        k_env = env_result[0] if env_result is not None else None
        mu_env = env_result[1] if env_result is not None else None

        # --- Strategy 2: IQR-trimmed L-fitting ---
        k_Lfit = RuleUnivariateMulPeel._estimate_k_from_Lfit(z, f, f_prime, snap_tol)

        # Pick the best: prefer envelope when available (handles trig better),
        # fall back to L-fitting for non-oscillatory functions.
        k_raw = None
        if k_env is not None and k_Lfit is not None:
            snap_env = round(k_env * 2.0) / 2.0
            snap_Lfit = round(k_Lfit * 2.0) / 2.0
            if abs(k_env - snap_env) <= abs(k_Lfit - snap_Lfit):
                k_raw = k_env
            else:
                k_raw = k_Lfit
        elif k_env is not None:
            k_raw = k_env
        elif k_Lfit is not None:
            k_raw = k_Lfit
        else:
            return None, None

        # Final snap and threshold
        k_snap = round(k_raw * 2.0) / 2.0
        if abs(k_raw - k_snap) > snap_tol:
            return None, None
        if abs(k_snap) < 0.3:
            return None, None
        return k_snap, mu_env

    @staticmethod
    def _estimate_k_from_envelope(
        z: np.ndarray, f: np.ndarray, snap_tol: float,
    ) -> "Optional[Tuple[float, float]]":
        """Estimate (k, mu) from log-envelope of |f| at local maxima.

        For f(z) = z^k * exp(mu*z) * trig(omega*z), the local maxima of |f|
        trace out z^k * exp(mu*z), so log(|f_max|) = k*log(z) + mu*z + const.

        Returns (k_raw, mu_raw) or None on failure.
        """
        from scipy.signal import argrelextrema

        abs_f = np.abs(f)
        max_idx = argrelextrema(abs_f, np.greater, order=3)[0]
        if len(max_idx) < 5:
            return None

        z_max = z[max_idx]
        f_max = abs_f[max_idx]

        # Keep only maxima with z > 0 and f > 0
        keep = (z_max > 1e-10) & (f_max > 1e-10)
        z_max = z_max[keep]
        f_max = f_max[keep]
        if len(z_max) < 5:
            return None

        log_z = np.log(z_max)
        log_f = np.log(f_max)

        # Fit log(|f_max|) = k*log(z) + mu*z + c
        A = np.column_stack([log_z, z_max, np.ones_like(z_max)])
        try:
            result = np.linalg.lstsq(A, log_f, rcond=None)
            k_raw = float(result[0][0])
            mu_raw = float(result[0][1])
        except np.linalg.LinAlgError:
            return None

        if not np.isfinite(k_raw) or not np.isfinite(mu_raw):
            return None
        return (k_raw, mu_raw)

    @staticmethod
    def _estimate_k_from_Lfit(
        z: np.ndarray, f: np.ndarray, f_prime: np.ndarray, snap_tol: float,
    ) -> "Optional[float]":
        """Estimate k from IQR-trimmed fit of L(z) = f'/f ~ k/z + constant."""
        abs_f = np.abs(f)
        threshold = max(1e-8, 0.01 * float(np.median(abs_f)))
        z_span = float(np.max(z) - np.min(z))

        mask = (np.abs(z) > 0.1 * z_span) & (abs_f > threshold)
        if mask.sum() < 20:
            return None

        z_sel = z[mask]
        L_sel = f_prime[mask] / f[mask]
        finite = np.isfinite(L_sel)
        if finite.sum() < 20:
            return None
        z_sel = z_sel[finite]
        L_sel = L_sel[finite]

        # IQR-based outlier trimming
        q25, q75 = np.percentile(L_sel, [25, 75])
        iqr = q75 - q25
        if iqr < 1e-10:
            iqr = max(abs(q25), abs(q75), 1.0)
        lower = q25 - 2.0 * iqr
        upper = q75 + 2.0 * iqr
        trim = (L_sel >= lower) & (L_sel <= upper)
        if trim.sum() < 15:
            return None
        z_trim = z_sel[trim]
        L_trim = L_sel[trim]

        inv_z = 1.0 / z_trim
        A = np.column_stack([inv_z, np.ones_like(inv_z)])
        try:
            result = np.linalg.lstsq(A, L_trim, rcond=None)
            k_raw = float(result[0][0])
        except np.linalg.LinAlgError:
            return None

        if not np.isfinite(k_raw):
            return None
        return k_raw

    # ------------------------------------------------------------------
    # helper: build Candidate A — exp_poly * sin_linear
    # ------------------------------------------------------------------
    @staticmethod
    def _build_exp_trig_candidate(
        root: Node, target: AtomNode, mu: float, omega: float,
    ) -> "Tuple[Optional[Node], Optional[Callable]]":
        """Replace target with MulNode(exp_poly(z;1), sin_linear(z))."""
        exp_tag = f"{target.tag}_exppeel"
        trig_tag = f"{target.tag}_trigpeel"

        exp_atom = AtomNode(
            kind="exp_poly",
            var_idxs=target.var_idxs,
            kwargs={"degree": 1},
            tag=exp_tag,
            inputs=clone_inputs(target),
        )
        trig_atom = AtomNode(
            kind="sin_linear",
            var_idxs=target.var_idxs,
            kwargs={},
            tag=trig_tag,
            inputs=clone_inputs(target),
        )
        new_subtree = MulNode(exp_atom, trig_atom)
        new_root = replace_atom_in_ast(root, target, new_subtree)
        if new_root is None:
            return None, None

        def _init_exp_trig(root_new, model_new, *, _exp_tag=exp_tag, _trig_tag=trig_tag,
                           _mu=mu, _omega=omega):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for _atom in _collect_all_atoms(root_new):
                if not isinstance(_atom, AtomNode):
                    continue
                leaf = atom_to_leaf.get(id(_atom), None)
                if leaf is None:
                    continue
                leaf = getattr(leaf, 'model', leaf)  # unwrap AutogradAdaptor
                try:
                    with torch.no_grad():
                        if _atom.tag == _exp_tag and str(_atom.kind).lower() == "exp_poly":
                            # exp_poly coeffs: exp(c0 + c1*z) → c0=0, c1=mu
                            if hasattr(leaf, "coeffs"):
                                leaf.coeffs.zero_()
                                if leaf.coeffs.numel() >= 2:
                                    leaf.coeffs[1] = float(_mu)
                            elif hasattr(leaf, "weight") and hasattr(leaf, "bias"):
                                leaf.weight.zero_()
                                leaf.bias.zero_()
                                if leaf.weight.numel() >= 1:
                                    leaf.weight.view(-1)[0] = float(_mu)
                        elif _atom.tag == _trig_tag and str(_atom.kind).lower() == "sin_linear":
                            # sin_linear: amp * sin(omega*z + phase)
                            if hasattr(leaf, "weight") and leaf.weight.numel() >= 1:
                                leaf.weight.zero_()
                                leaf.weight.view(-1)[0] = float(_omega)
                            if hasattr(leaf, "bias") and leaf.bias.numel() >= 1:
                                leaf.bias.fill_(0.0)
                            if hasattr(leaf, "amp") and leaf.amp.numel() >= 1:
                                leaf.amp.fill_(1.0)
                except Exception:
                    pass

        _init_exp_trig._after_analytic_init = True
        return new_root, _init_exp_trig

    # ------------------------------------------------------------------
    # helper: build Candidate B — exp_poly * nn (residual)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_exp_nn_residual_candidate(
        root: Node, target: AtomNode, mu: float,
    ) -> "Tuple[Optional[Node], Optional[Callable]]":
        """Replace target with MulNode(exp_poly(z;1), nn(z)) reusing teacher."""
        exp_tag = f"{target.tag}_exppeel"
        nn_tag = f"{target.tag}_nnresid"

        exp_atom = AtomNode(
            kind="exp_poly",
            var_idxs=target.var_idxs,
            kwargs={"degree": 1},
            tag=exp_tag,
            inputs=clone_inputs(target),
        )
        nn_atom = AtomNode(
            kind="nn",
            var_idxs=target.var_idxs,
            kwargs=dict(target.kwargs),
            tag=nn_tag,
            inputs=clone_inputs(target),
        )
        new_subtree = MulNode(exp_atom, nn_atom)
        new_root = replace_atom_in_ast(root, target, new_subtree)
        if new_root is None:
            return None, None

        def _init_exp_nn(root_new, model_new, *, _exp_tag=exp_tag, _mu=mu):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for _atom in _collect_all_atoms(root_new):
                if not isinstance(_atom, AtomNode):
                    continue
                leaf = atom_to_leaf.get(id(_atom), None)
                if leaf is None:
                    continue
                leaf = getattr(leaf, 'model', leaf)  # unwrap AutogradAdaptor
                try:
                    with torch.no_grad():
                        if _atom.tag == _exp_tag and str(_atom.kind).lower() == "exp_poly":
                            if hasattr(leaf, "coeffs"):
                                leaf.coeffs.zero_()
                                if leaf.coeffs.numel() >= 2:
                                    leaf.coeffs[1] = float(_mu)
                            elif hasattr(leaf, "weight") and hasattr(leaf, "bias"):
                                leaf.weight.zero_()
                                leaf.bias.zero_()
                                if leaf.weight.numel() >= 1:
                                    leaf.weight.view(-1)[0] = float(_mu)
                except Exception:
                    pass

        _init_exp_nn._after_analytic_init = True
        return new_root, _init_exp_nn

    # ------------------------------------------------------------------
    # helper: build mono * trig candidate — power(z;k) * sin_linear(z)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mono_trig_candidate(
        root: Node, target: AtomNode, k: float, omega: float,
    ) -> "Tuple[Optional[Node], Optional[Callable]]":
        """Replace target with MulNode(power(z;k), sin_linear(z))."""
        mono_tag = f"{target.tag}_monopeel"
        trig_tag = f"{target.tag}_trigpeel"

        mono_atom = AtomNode(
            kind="power",
            var_idxs=target.var_idxs,
            kwargs={"exponent_init": k},
            tag=mono_tag,
            inputs=clone_inputs(target),
        )
        trig_atom = AtomNode(
            kind="sin_linear",
            var_idxs=target.var_idxs,
            kwargs={},
            tag=trig_tag,
            inputs=clone_inputs(target),
        )
        new_subtree = MulNode(mono_atom, trig_atom)
        new_root = replace_atom_in_ast(root, target, new_subtree)
        if new_root is None:
            return None, None

        def _init_mono_trig(root_new, model_new, *, _mono_tag=mono_tag,
                            _trig_tag=trig_tag, _k=k, _omega=omega):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for _atom in _collect_all_atoms(root_new):
                if not isinstance(_atom, AtomNode):
                    continue
                leaf = atom_to_leaf.get(id(_atom), None)
                if leaf is None:
                    continue
                leaf = getattr(leaf, 'model', leaf)  # unwrap AutogradAdaptor
                try:
                    with torch.no_grad():
                        if _atom.tag == _mono_tag and str(_atom.kind).lower() == "power":
                            if hasattr(leaf, "exponent"):
                                leaf.exponent.fill_(float(_k))
                            if hasattr(leaf, "amp"):
                                leaf.amp.fill_(1.0)
                        elif _atom.tag == _trig_tag and str(_atom.kind).lower() == "sin_linear":
                            if hasattr(leaf, "weight") and leaf.weight.numel() >= 1:
                                leaf.weight.zero_()
                                leaf.weight.view(-1)[0] = float(_omega)
                            if hasattr(leaf, "bias") and leaf.bias.numel() >= 1:
                                leaf.bias.fill_(0.0)
                            if hasattr(leaf, "amp") and leaf.amp.numel() >= 1:
                                leaf.amp.fill_(1.0)
                except Exception:
                    pass

        _init_mono_trig._after_analytic_init = True
        return new_root, _init_mono_trig

    # ------------------------------------------------------------------
    # helper: build mono * exp candidate — power(z;k) * exp_poly(z;1)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mono_exp_candidate(
        root: Node, target: AtomNode, k: float, mu: float,
    ) -> "Tuple[Optional[Node], Optional[Callable]]":
        """Replace target with MulNode(power(z;k), exp_poly(z;1))."""
        mono_tag = f"{target.tag}_monopeel"
        exp_tag = f"{target.tag}_exppeel"

        mono_atom = AtomNode(
            kind="power",
            var_idxs=target.var_idxs,
            kwargs={"exponent_init": k},
            tag=mono_tag,
            inputs=clone_inputs(target),
        )
        exp_atom = AtomNode(
            kind="exp_poly",
            var_idxs=target.var_idxs,
            kwargs={"degree": 1},
            tag=exp_tag,
            inputs=clone_inputs(target),
        )
        new_subtree = MulNode(mono_atom, exp_atom)
        new_root = replace_atom_in_ast(root, target, new_subtree)
        if new_root is None:
            return None, None

        def _init_mono_exp(root_new, model_new, *, _mono_tag=mono_tag,
                           _exp_tag=exp_tag, _k=k, _mu=mu):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for _atom in _collect_all_atoms(root_new):
                if not isinstance(_atom, AtomNode):
                    continue
                leaf = atom_to_leaf.get(id(_atom), None)
                if leaf is None:
                    continue
                leaf = getattr(leaf, 'model', leaf)  # unwrap AutogradAdaptor
                try:
                    with torch.no_grad():
                        if _atom.tag == _mono_tag and str(_atom.kind).lower() == "power":
                            if hasattr(leaf, "exponent"):
                                leaf.exponent.fill_(float(_k))
                            if hasattr(leaf, "amp"):
                                leaf.amp.fill_(1.0)
                        elif _atom.tag == _exp_tag and str(_atom.kind).lower() == "exp_poly":
                            if hasattr(leaf, "coeffs"):
                                leaf.coeffs.zero_()
                                if leaf.coeffs.numel() >= 2:
                                    leaf.coeffs[1] = float(_mu)
                            elif hasattr(leaf, "weight") and hasattr(leaf, "bias"):
                                leaf.weight.zero_()
                                leaf.bias.zero_()
                                if leaf.weight.numel() >= 1:
                                    leaf.weight.view(-1)[0] = float(_mu)
                except Exception:
                    pass

        _init_mono_exp._after_analytic_init = True
        return new_root, _init_mono_exp

    # ------------------------------------------------------------------
    # helper: build mono * exp * trig candidate
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mono_exp_trig_candidate(
        root: Node, target: AtomNode, k: float, mu: float, omega: float,
    ) -> "Tuple[Optional[Node], Optional[Callable]]":
        """Replace target with MulNode(power(z;k), MulNode(exp_poly(z;1), sin_linear(z)))."""
        mono_tag = f"{target.tag}_monopeel"
        exp_tag = f"{target.tag}_exppeel"
        trig_tag = f"{target.tag}_trigpeel"

        mono_atom = AtomNode(
            kind="power",
            var_idxs=target.var_idxs,
            kwargs={"exponent_init": k},
            tag=mono_tag,
            inputs=clone_inputs(target),
        )
        exp_atom = AtomNode(
            kind="exp_poly",
            var_idxs=target.var_idxs,
            kwargs={"degree": 1},
            tag=exp_tag,
            inputs=clone_inputs(target),
        )
        trig_atom = AtomNode(
            kind="sin_linear",
            var_idxs=target.var_idxs,
            kwargs={},
            tag=trig_tag,
            inputs=clone_inputs(target),
        )
        exp_trig = MulNode(exp_atom, trig_atom)
        new_subtree = MulNode(mono_atom, exp_trig)
        new_root = replace_atom_in_ast(root, target, new_subtree)
        if new_root is None:
            return None, None

        def _init_mono_exp_trig(root_new, model_new, *, _mono_tag=mono_tag,
                                _exp_tag=exp_tag, _trig_tag=trig_tag,
                                _k=k, _mu=mu, _omega=omega):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for _atom in _collect_all_atoms(root_new):
                if not isinstance(_atom, AtomNode):
                    continue
                leaf = atom_to_leaf.get(id(_atom), None)
                if leaf is None:
                    continue
                leaf = getattr(leaf, 'model', leaf)  # unwrap AutogradAdaptor
                try:
                    with torch.no_grad():
                        if _atom.tag == _mono_tag and str(_atom.kind).lower() == "power":
                            if hasattr(leaf, "exponent"):
                                leaf.exponent.fill_(float(_k))
                            if hasattr(leaf, "amp"):
                                leaf.amp.fill_(1.0)
                        elif _atom.tag == _exp_tag and str(_atom.kind).lower() == "exp_poly":
                            if hasattr(leaf, "coeffs"):
                                leaf.coeffs.zero_()
                                if leaf.coeffs.numel() >= 2:
                                    leaf.coeffs[1] = float(_mu)
                            elif hasattr(leaf, "weight") and hasattr(leaf, "bias"):
                                leaf.weight.zero_()
                                leaf.bias.zero_()
                                if leaf.weight.numel() >= 1:
                                    leaf.weight.view(-1)[0] = float(_mu)
                        elif _atom.tag == _trig_tag and str(_atom.kind).lower() == "sin_linear":
                            if hasattr(leaf, "weight") and leaf.weight.numel() >= 1:
                                leaf.weight.zero_()
                                leaf.weight.view(-1)[0] = float(_omega)
                            if hasattr(leaf, "bias") and leaf.bias.numel() >= 1:
                                leaf.bias.fill_(0.0)
                            if hasattr(leaf, "amp") and leaf.amp.numel() >= 1:
                                leaf.amp.fill_(1.0)
                except Exception:
                    pass

        _init_mono_exp_trig._after_analytic_init = True
        return new_root, _init_mono_exp_trig

    # ------------------------------------------------------------------
    # helper: build mono * nn (residual) candidate
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mono_nn_residual_candidate(
        root: Node, target: AtomNode, k: float,
    ) -> "Tuple[Optional[Node], Optional[Callable]]":
        """Replace target with MulNode(power(z;k), nn(z)) reusing teacher."""
        mono_tag = f"{target.tag}_monopeel"
        nn_tag = f"{target.tag}_nnresid"

        mono_atom = AtomNode(
            kind="power",
            var_idxs=target.var_idxs,
            kwargs={"exponent_init": k},
            tag=mono_tag,
            inputs=clone_inputs(target),
        )
        nn_atom = AtomNode(
            kind="nn",
            var_idxs=target.var_idxs,
            kwargs=dict(target.kwargs),
            tag=nn_tag,
            inputs=clone_inputs(target),
        )
        new_subtree = MulNode(mono_atom, nn_atom)
        new_root = replace_atom_in_ast(root, target, new_subtree)
        if new_root is None:
            return None, None

        def _init_mono_nn(root_new, model_new, *, _mono_tag=mono_tag, _k=k):
            atom_to_leaf = build_atom_to_leaf_map(root_new, model_new)
            for _atom in _collect_all_atoms(root_new):
                if not isinstance(_atom, AtomNode):
                    continue
                leaf = atom_to_leaf.get(id(_atom), None)
                if leaf is None:
                    continue
                leaf = getattr(leaf, 'model', leaf)  # unwrap AutogradAdaptor
                try:
                    with torch.no_grad():
                        if _atom.tag == _mono_tag and str(_atom.kind).lower() == "power":
                            if hasattr(leaf, "exponent"):
                                leaf.exponent.fill_(float(_k))
                            if hasattr(leaf, "amp"):
                                leaf.amp.fill_(1.0)
                except Exception:
                    pass

        _init_mono_nn._after_analytic_init = True
        return new_root, _init_mono_nn
