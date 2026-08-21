# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Gradient-based compound-coordinate detection and early compound construction."""

from typing import TYPE_CHECKING
import math
from typing import Dict, Tuple
import torch
from nestynet_sr.sr_core import Var, ast_to_human_readable, build_linear_ast, build_mixed_compound_ast, build_monomial_ast, build_radial_r2_ast, check_linear_compound, check_mixed_compound, check_monomial_compound, check_monomial_compound_logderiv, replace_atom_in_ast
from nestynet_sr.sr_core.bridges import AddNode, AtomNode, ConstNode, CosNode, MulNode, PowNode, SinNode, _collect_var_idxs_from_inputs, clone_ast, effective_arity, eval_inputs, get_input_exprs, has_nontrivial_input, is_trivial_input
from .features import TrigAxisSpec, discover_trig_axes, probe_oracle_scaling, probe_trig_scaling
from .monomial_peel_plan import clean_subset_patterns
from .shadow_coordinates import ShadowRegistry
from .compound_proposals import build_barycentric_compound_proposals, build_logexp_compound_proposals, build_metric_distance_compound_proposals, stageA_tuple_from_proposal
from .model_builders import build_composite_ast
from .model_selection import compute_accept_threshold as _compute_accept_threshold, resolve_acceptance_noise_floor_raw as _resolve_acceptance_noise_floor_raw
from .stagea_fit_tournament import fit_stageA_candidate_with_tournament
from .wrapper_policy import snap_omega

from ._search_shadow import (
    GREEN,
    RED,
    RESET,
    _apply_fit_link_to_model,
    _clone_reuse_leaves,
    _oracle_trig_for_axis,
    _oracle_trig_to_axis_specs,
    _stageA_record_logexp_shadows,
    _stageA_record_shadow_coordinate,
    _stageA_shadow_composite_proposals,
    _stageA_shadow_preserved_factor_proposals,
    _stageA_shadow_trig_factor_peel_proposals,
    _stageA_shadow_unit_status,
    _stageA_trig_shadow_from_spec,
    _trig_kind_from_phase,
)
from ._search_training import (
    _test_difference_product_power_structure,
    _test_difference_product_structure,
)

if TYPE_CHECKING:
    from ._search_proposals import (
        _COMPOUND_Z_TOKEN,
        _append_compound_extra_input_asts,
        _atom_compound_cols,
        _build_monomial_ast_from_cols,
        _build_radial_r2_ast_from_cols,
        _clean_monomial_product_proposal_from_scaling,
        _compound_ast_for_token,
        _compound_extra_input_asts_after_prefactor_peel,
        _compound_pattern_entry_is_zero,
        _is_compound_token,
        _stageA_noisy_soft_monomial_product_proposals_from_scaling,
    )
    from ._search_policy import (
        _accept_threshold_with_structural_target,
        _nn_split_signature,
    )

def _scan_gradient_ratio_pairs(
    x_vals, dydx_vals, max_power=4, r2_threshold=0.95, int_threshold=0.25
):
    """Scan all variable pairs for power-difference signatures via gradient ratio.

    For z = u^n - w^n:  log|∂f/∂u / ∂f/∂w| = (n-1)*log(u/w) + const
    The slope (n-1) is snapped to the nearest integer.

    Parameters
    ----------
    x_vals : np.ndarray [N, k]
        Variable values (local indices).
    dydx_vals : np.ndarray [N, k]
        Gradient values (local indices).
    max_power : int
        Maximum power n to accept.
    r2_threshold : float
        Minimum R² for the log-log regression.
    int_threshold : float
        Maximum deviation from integer slope to accept.

    Returns
    -------
    list of (i_local, j_local, n, confidence)
        n is always a positive integer (snapped from regression slope).
    """
    import numpy as np

    k = x_vals.shape[1]
    results = []

    for i in range(k):
        for j in range(i + 1, k):
            gi = dydx_vals[:, i]
            gj = dydx_vals[:, j]

            # Filter points where both gradients are non-negligible
            mag_i = np.abs(gi)
            mag_j = np.abs(gj)
            thresh_i = np.percentile(mag_i, 5) + 1e-30
            thresh_j = np.percentile(mag_j, 5) + 1e-30
            mask = (mag_i > thresh_i) & (mag_j > thresh_j)

            xi = x_vals[mask, i]
            xj = x_vals[mask, j]
            gi_m = gi[mask]
            gj_m = gj[mask]

            # Need positive values for log
            mask2 = (np.abs(xi) > 1e-30) & (np.abs(xj) > 1e-30)
            if np.sum(mask2) < 50:
                continue
            xi = xi[mask2]
            xj = xj[mask2]
            gi_m = gi_m[mask2]
            gj_m = gj_m[mask2]

            ratio = gi_m / gj_m
            log_abs_ratio = np.log(np.abs(ratio) + 1e-30)
            log_xratio = np.log(np.abs(xi / xj) + 1e-30)

            # Filter NaN/Inf
            valid = np.isfinite(log_abs_ratio) & np.isfinite(log_xratio)
            if np.sum(valid) < 50:
                continue
            log_abs_ratio = log_abs_ratio[valid]
            log_xratio = log_xratio[valid]
            ratio_valid = ratio[mask2][valid]

            # Linear regression: log_abs_ratio = a + b * log_xratio
            xm = log_xratio - np.mean(log_xratio)
            ym = log_abs_ratio - np.mean(log_abs_ratio)
            ss_xx = np.dot(xm, xm)
            if ss_xx < 1e-30:
                continue
            b = np.dot(xm, ym) / ss_xx
            a = np.mean(log_abs_ratio) - b * np.mean(log_xratio)

            # R²
            y_pred = a + b * log_xratio
            ss_res = np.sum((log_abs_ratio - y_pred) ** 2)
            ss_tot = np.sum(ym ** 2)

            # Special case: log|ratio| has very low variance → ratio is ~constant
            # This means slope ≈ 0 → n = 1 (linear difference).
            # Confirm by checking that |ratio| is tightly clustered around its mean.
            if ss_tot < 0.1 * len(ym):
                # ratio is approximately constant; check relative spread
                abs_ratio = np.abs(ratio_valid)
                rel_spread = np.std(abs_ratio) / (np.mean(abs_ratio) + 1e-30)
                if rel_spread > 0.15:
                    continue
                # Slope is ~0, so n=1
                b = 0.0
                r2 = 1.0 - rel_spread  # proxy confidence
            else:
                r2 = 1.0 - ss_res / ss_tot
                if r2 < r2_threshold:
                    continue

            # Snap slope to integer
            k_int = round(b)
            if abs(b - k_int) >= int_threshold:
                continue

            n = k_int + 1
            if abs(n) > max_power or n == 0:
                continue

            # Check sign: difference has negative ratio
            neg_frac = np.mean(ratio_valid < 0)

            if neg_frac > 0.8:
                # Difference pattern (xi^n - xj^n)
                int_dev = abs(b - k_int) / int_threshold
                conf = r2 * (1.0 - int_dev) * neg_frac
                results.append((i, j, abs(n), float(conf)))
            # (neg_frac < 0.2 would be a sum pattern — future extension)

    return results


def _test_power_difference_structure(x_vals, dydx_vals, i, j, n, precision=0.1):
    """Test if gradients are consistent with f(z) where z = xi^n - xj^n.

    Key relation: xj^(n-1) * ∂f/∂xi + xi^(n-1) * ∂f/∂xj ≈ 0.
    For n=1 this reduces to ∂f/∂xi + ∂f/∂xj ≈ 0.

    Parameters
    ----------
    x_vals : np.ndarray [N, k]
    dydx_vals : np.ndarray [N, k]
    i, j : int
        Local indices.
    n : int
        Power (positive integer).
    precision : float

    Returns
    -------
    float
        Confidence in [0, 1].
    """
    import numpy as np

    gi = dydx_vals[:, i]
    gj = dydx_vals[:, j]
    xi = x_vals[:, i]
    xj = x_vals[:, j]
    n = int(n)

    if n == 1:
        # Simple difference: gi + gj ≈ 0
        lhs = gi + gj
    else:
        # z = xi^n - xj^n  =>  dz/dxi = n*xi^(n-1), dz/dxj = -n*xj^(n-1)
        # f'(z)*n*xi^(n-1) * xj^(n-1) + f'(z)*(-n)*xj^(n-1) * xi^(n-1)
        # Simpler: gi * xj^(n-1) + gj * xi^(n-1) ≈ 0
        # (because gi = f'(z)*n*xi^(n-1) and gj = -f'(z)*n*xj^(n-1))
        lhs = gi * np.abs(xj) ** (n - 1) + gj * np.abs(xi) ** (n - 1)

    norm = np.linalg.norm(gi * np.abs(xj) ** (n - 1)) + np.linalg.norm(gj * np.abs(xi) ** (n - 1)) + 1e-12
    if n == 1:
        norm = np.linalg.norm(gi) + np.linalg.norm(gj) + 1e-12
    rel_residual = np.linalg.norm(lhs) / norm

    conf = 1.0 - rel_residual
    return max(0.0, float(conf))


def _test_power_diff_product_structure(
    x_vals, dydx_vals, i, j, k, n, precision=0.1, f_vals=None
):
    """Test if gradients are consistent with f(z) where z = (xi^n - xj^n) * xk.

    Test 1: xj^(n-1)*gi + xi^(n-1)*gj ≈ 0  (power-difference structure)
    Test 2: gi * (xi^n - xj^n) ≈ gk * n * xi^(n-1) * xk  (product relation)

    Also tries outer prefactor variants f = xk^p * g(z).

    Parameters
    ----------
    x_vals, dydx_vals : np.ndarray [N, k]
    i, j, k : int
        Local indices.
    n : int
        Power in the difference.
    precision : float
    f_vals : np.ndarray [N] or [N,1], optional
        Function values for outer prefactor detection.

    Returns
    -------
    (confidence, outer_power) : (float, int)
    """
    import numpy as np

    gi = dydx_vals[:, i]
    gj = dydx_vals[:, j]
    gk = dydx_vals[:, k]
    xi = x_vals[:, i]
    xj = x_vals[:, j]
    xk = x_vals[:, k]
    n = int(n)

    # Test 1: power-difference structure
    if n == 1:
        lhs1 = gi + gj
        norm1 = np.linalg.norm(gi) + np.linalg.norm(gj) + 1e-12
    else:
        lhs1 = gi * np.abs(xj) ** (n - 1) + gj * np.abs(xi) ** (n - 1)
        norm1 = (
            np.linalg.norm(gi * np.abs(xj) ** (n - 1))
            + np.linalg.norm(gj * np.abs(xi) ** (n - 1))
            + 1e-12
        )
    rel_residual_1 = np.linalg.norm(lhs1) / norm1

    # Test 2: product relation
    # For z = (xi^n - xj^n)*xk: dz/dxi = n*xi^(n-1)*xk, dz/dxk = xi^n - xj^n
    # So gi * (xi^n - xj^n) = gk * n * xi^(n-1) * xk
    diff_n = np.sign(xi) * np.abs(xi) ** n - np.sign(xj) * np.abs(xj) ** n
    if n % 2 == 0:
        diff_n = xi ** n - xj ** n
    lhs2 = gi * diff_n
    rhs2 = gk * n * np.sign(xi) * np.abs(xi) ** (n - 1) * xk
    if n % 2 == 0:
        rhs2 = gk * n * xi ** (n - 1) * xk
    rel_residual_2 = np.linalg.norm(lhs2 - rhs2) / (
        np.linalg.norm(lhs2) + np.linalg.norm(rhs2) + 1e-12
    )

    best_residual_2 = rel_residual_2
    best_power = 0

    # Alternative Test 2 for f = xk^p * g(z) patterns
    if f_vals is not None and rel_residual_2 > 0.2:
        f = f_vals.squeeze() if f_vals.ndim > 1 else f_vals
        lhs_alt = gi * diff_n
        for p in [1, 2, -1]:
            rhs_alt = gk * xk - p * f
            residual = np.linalg.norm(lhs_alt - rhs_alt) / (
                np.linalg.norm(lhs_alt) + np.linalg.norm(rhs_alt) + 1e-12
            )
            if residual < best_residual_2:
                best_residual_2 = residual
                best_power = p

    conf = 1.0 - max(rel_residual_1, best_residual_2)
    return max(0.0, float(conf)), int(best_power)


def _detect_pure_difference_compounds(
    x_vals,
    dydx_vals,
    var_idxs,
    invariance_feats=None,
    precision=0.1,
    z_ast_existing=None,
    units_spec=None,
    enforce_units: bool = False,
):
    """
    Detect pure pair compounds: z = xi^n - xj^n, plus closely related
    low-cost sum/difference coordinates such as 1/xi - 1/xj.

    Primary mechanism: gradient-ratio scan (works for all integer powers n).
    Fallback for n=1: invariance features (if provided).

    Parameters
    ----------
    x_vals : np.ndarray of shape [N, k]
        Variable values for the atom's variables.
    dydx_vals : np.ndarray of shape [N, k]
        Gradient values for the atom's variables.
    var_idxs : tuple of int
        Global variable indices for this atom (may contain _COMPOUND_Z_TOKEN).
    invariance_feats : list of FeatureSpec, optional
        Invariance features discovered from gradients (used as fallback for n=1).
    precision : float
        Tolerance threshold for acceptance.
    z_ast_existing : AST node, optional
        When extending an existing compound atom, the AST for the current z.

    Returns
    -------
    list of (coeffs, z_ast, confidence, None, meta) proposals.
    """
    import numpy as np

    from nestynet_sr.sr_core.separability_math import (
        build_linear_ast,
        build_power_difference_ast,
    )

    proposals = []
    tested_pairs = set()

    def _pair_key(i_global, j_global):
        return tuple(sorted([i_global, j_global], key=lambda x: (isinstance(x, str), str(x))))

    def _safe_idx(global_idx):
        return global_idx if isinstance(global_idx, str) else int(global_idx)

    def _label(global_idx):
        return str(global_idx) if _is_compound_token(global_idx) else f"x{int(global_idx)}"

    def _difference_dims_compatible(i_global, j_global):
        if (not bool(enforce_units)) or units_spec is None:
            return True
        try:
            from nestynet_sr.sr_core.units import eval_analytic_expr_dim

            def dim_for(global_idx):
                if _is_compound_token(global_idx):
                    z_expr = _compound_ast_for_token(z_ast_existing, global_idx)
                    return eval_analytic_expr_dim(z_expr, units_spec.x_dims)
                idx = int(global_idx)
                if 0 <= idx < len(units_spec.x_dims):
                    return tuple(units_spec.x_dims[idx])
                return None

            di = dim_for(i_global)
            dj = dim_for(j_global)
            # Unknown dimensions are left to the later full-AST units gate.
            return (di is None) or (dj is None) or tuple(di) == tuple(dj)
        except Exception:
            return True

    def _build_linear_difference_ast(i_global, j_global):
        if z_ast_existing is not None and (
            _is_compound_token(i_global) or _is_compound_token(j_global)
        ):
            left = clone_ast(_compound_ast_for_token(z_ast_existing, i_global)) if _is_compound_token(i_global) else Var(int(i_global))
            right = clone_ast(_compound_ast_for_token(z_ast_existing, j_global)) if _is_compound_token(j_global) else Var(int(j_global))
            return AddNode(left, MulNode(ConstNode(-1.0), right))
        return build_linear_ast((int(i_global), int(j_global)), (1, -1))

    def _power_pair_term_ast(global_idx, power: int, inverse: bool):
        if _is_compound_token(global_idx):
            base = clone_ast(_compound_ast_for_token(z_ast_existing, global_idx))
        else:
            base = Var(int(global_idx))
        exp = -int(power) if inverse else int(power)
        if exp == 1:
            return base
        return PowNode(base, float(exp))

    def _build_power_pair_sumdiff_ast(
        i_global,
        j_global,
        *,
        power: int,
        left_inverse: bool,
        right_inverse: bool,
        op: str,
    ):
        left = _power_pair_term_ast(i_global, power, left_inverse)
        right = _power_pair_term_ast(j_global, power, right_inverse)
        if str(op) == "plus":
            return AddNode(left, right)
        return AddNode(left, MulNode(ConstNode(-1.0), right))

    def _power_pair_term_label(global_idx, power: int, inverse: bool):
        label = _label(global_idx)
        exp = -int(power) if inverse else int(power)
        if exp == 1:
            return label
        return f"{label}^{exp}"

    def _power_pair_dims_compatible(
        i_global,
        j_global,
        *,
        power: int,
        left_inverse: bool,
        right_inverse: bool,
    ):
        if (not bool(enforce_units)) or units_spec is None:
            return True
        try:
            from nestynet_sr.sr_core.units import eval_analytic_expr_dim

            left = _power_pair_term_ast(i_global, power, left_inverse)
            right = _power_pair_term_ast(j_global, power, right_inverse)
            dl = eval_analytic_expr_dim(left, units_spec.x_dims)
            dr = eval_analytic_expr_dim(right, units_spec.x_dims)
            # Unknown dimensions are left to the later full-AST units gate.
            return (dl is None) or (dr is None) or tuple(dl) == tuple(dr)
        except Exception:
            return True

    def _power_pair_gradient_confidence(
        i_local,
        j_local,
        *,
        power: int,
        left_inverse: bool,
        right_inverse: bool,
        op: str,
    ):
        xi = np.asarray(x_vals[:, i_local], dtype=float)
        xj = np.asarray(x_vals[:, j_local], dtype=float)
        gi = np.asarray(dydx_vals[:, i_local], dtype=float)
        gj = np.asarray(dydx_vals[:, j_local], dtype=float)

        exp_i = -int(power) if left_inverse else int(power)
        exp_j = -int(power) if right_inverse else int(power)
        sign_j = 1.0 if str(op) == "plus" else -1.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            dzi = float(exp_i) * np.power(xi, exp_i - 1)
            dzj = sign_j * float(exp_j) * np.power(xj, exp_j - 1)

        finite = (
            np.isfinite(xi)
            & np.isfinite(xj)
            & np.isfinite(gi)
            & np.isfinite(gj)
            & np.isfinite(dzi)
            & np.isfinite(dzj)
        )
        if int(np.sum(finite)) < 50:
            return None
        gi = gi[finite]
        gj = gj[finite]
        dzi = dzi[finite]
        dzj = dzj[finite]

        pair_energy = float(np.linalg.norm(gi) + np.linalg.norm(gj))
        z_energy = float(np.linalg.norm(dzi) + np.linalg.norm(dzj))
        if (
            (not np.isfinite(pair_energy))
            or (not np.isfinite(z_energy))
            or pair_energy <= 1e-12
            or z_energy <= 1e-12
        ):
            return None
        try:
            total_energy = float(np.linalg.norm(np.nan_to_num(dydx_vals, nan=0.0, posinf=0.0, neginf=0.0)))
        except Exception:
            total_energy = pair_energy
        if pair_energy <= max(1e-12, 1e-10 * (total_energy + 1e-12)):
            return None

        lhs = gi * dzj
        rhs = gj * dzi
        denom = float(np.linalg.norm(lhs) + np.linalg.norm(rhs) + 1e-12)
        rel_residual = float(np.linalg.norm(lhs - rhs) / denom)
        if not np.isfinite(rel_residual):
            return None
        return 1.0 - rel_residual, rel_residual

    def _power_pair_sumdiff_proposal(
        i_local,
        j_local,
        conf,
        *,
        residual,
        power: int,
        left_inverse: bool,
        right_inverse: bool,
        op: str,
    ):
        i_global = var_idxs[i_local]
        j_global = var_idxs[j_local]
        pair_has_z = _is_compound_token(i_global) or _is_compound_token(j_global)
        if z_ast_existing is not None and not pair_has_z and len(var_idxs) <= 2:
            return None
        if not _power_pair_dims_compatible(
            i_global,
            j_global,
            power=power,
            left_inverse=left_inverse,
            right_inverse=right_inverse,
        ):
            return None

        exp_i = -int(power) if left_inverse else int(power)
        exp_j = -int(power) if right_inverse else int(power)
        signed_exp_j = exp_j if str(op) == "plus" else -exp_j
        coeffs = tuple(
            exp_i if idx == i_global else signed_exp_j if idx == j_global else 0
            for idx in var_idxs
        )
        try:
            z_ast = _build_power_pair_sumdiff_ast(
                i_global,
                j_global,
                power=power,
                left_inverse=left_inverse,
                right_inverse=right_inverse,
                op=op,
            )
        except Exception:
            return None

        if z_ast_existing is not None:
            extra_override = sorted(
                int(idx) for idx, coeff in zip(var_idxs, coeffs)
                if isinstance(idx, int) and int(coeff) == 0
            )
        else:
            extra_override = None

        meta = {
            "kind": "power_pair_sumdiff",
            "family": "power_pair_sumdiff",
            "power": int(power),
            "op": str(op),
            "left_inverse": bool(left_inverse),
            "right_inverse": bool(right_inverse),
            "indices": (_safe_idx(i_global), _safe_idx(j_global)),
            "source": "gradient_colinearity",
            "gradient_colinearity_residual": float(residual),
            "from_invariance": False,
        }
        if extra_override is not None:
            meta["extra_override"] = extra_override
        if z_ast_existing is not None and not pair_has_z:
            try:
                meta["preserve_z_ast"] = clone_ast(_compound_ast_for_token(z_ast_existing, _COMPOUND_Z_TOKEN))
            except Exception:
                pass
        return (coeffs, z_ast, float(conf), extra_override, meta)

    def _linear_difference_proposal(
        i_local,
        j_local,
        conf,
        *,
        source,
        residual=None,
    ):
        i_global = var_idxs[i_local]
        j_global = var_idxs[j_local]
        pair_has_z = _is_compound_token(i_global) or _is_compound_token(j_global)
        if z_ast_existing is not None and not pair_has_z and len(var_idxs) <= 2:
            return None
        if not _difference_dims_compatible(i_global, j_global):
            return None

        linear_coeffs = tuple(
            1 if idx == i_global else -1 if idx == j_global else 0
            for idx in var_idxs
        )
        try:
            z_ast = _build_linear_difference_ast(i_global, j_global)
        except Exception:
            return None

        if z_ast_existing is not None:
            extra_override = sorted(
                int(idx) for idx, coeff in zip(var_idxs, linear_coeffs)
                if isinstance(idx, int) and coeff == 0
            )
        else:
            extra_override = None

        meta = {
            "kind": "power_difference",
            "power": 1,
            "indices": (_safe_idx(i_global), _safe_idx(j_global)),
            "source": str(source),
            "from_invariance": str(source) == "invariance_fallback",
        }
        if residual is not None:
            meta["anti_gradient_residual"] = float(residual)
        if extra_override is not None:
            meta["extra_override"] = extra_override
        if z_ast_existing is not None and not pair_has_z:
            try:
                meta["preserve_z_ast"] = clone_ast(_compound_ast_for_token(z_ast_existing, _COMPOUND_Z_TOKEN))
            except Exception:
                pass
        return (linear_coeffs, z_ast, float(conf), extra_override, meta)

    def _anti_gradient_confidence(i_local, j_local):
        gi = np.asarray(dydx_vals[:, i_local], dtype=float)
        gj = np.asarray(dydx_vals[:, j_local], dtype=float)
        finite = np.isfinite(gi) & np.isfinite(gj)
        if int(np.sum(finite)) < 50:
            return None
        gi = gi[finite]
        gj = gj[finite]
        pair_energy = float(np.linalg.norm(gi) + np.linalg.norm(gj))
        if (not np.isfinite(pair_energy)) or pair_energy <= 1e-12:
            return None
        try:
            total_energy = float(np.linalg.norm(np.nan_to_num(dydx_vals, nan=0.0, posinf=0.0, neginf=0.0)))
        except Exception:
            total_energy = pair_energy
        if pair_energy <= max(1e-12, 1e-10 * (total_energy + 1e-12)):
            return None
        residual = float(np.linalg.norm(gi + gj) / (pair_energy + 1e-12))
        if not np.isfinite(residual):
            return None
        return 1.0 - residual, residual

    def _make_disjoint_difference_bundle(single_props):
        """Build one proposal that applies disjoint linear differences together."""
        linear_props = []
        for prop in single_props:
            if len(prop) < 5:
                continue
            coeffs, z_expr, conf, _extra, meta = prop
            meta = meta or {}
            if str(meta.get("kind", "")) != "power_difference":
                continue
            try:
                if int(meta.get("power", 0)) != 1:
                    continue
            except Exception:
                continue
            idxs = tuple(meta.get("indices", ()))
            if len(idxs) != 2:
                continue
            linear_props.append((prop, tuple(idxs), float(conf)))

        if len(linear_props) < 2:
            return None

        linear_props.sort(key=lambda item: (-item[2], tuple(str(v) for v in item[1])))
        selected = []
        used = set()
        for prop, idxs, _conf in linear_props:
            if any(idx in used for idx in idxs):
                continue
            selected.append(prop)
            used.update(idxs)

        if len(selected) < 2:
            return None

        bundle_coeffs = [0 for _ in var_idxs]
        z_exprs = []
        confs = []
        component_indices = []
        component_sources = []
        for coeffs, z_expr, conf, _extra, meta in selected:
            z_exprs.append(clone_ast(z_expr))
            confs.append(float(conf))
            meta = meta or {}
            component_indices.append(tuple(meta.get("indices", ())))
            component_sources.append(str(meta.get("source", "")))
            for k, c in enumerate(coeffs):
                try:
                    if int(c) != 0:
                        bundle_coeffs[k] = int(c)
                except Exception:
                    pass

        extra_raw = [
            int(idx) for idx, coeff in zip(var_idxs, bundle_coeffs)
            if isinstance(idx, int) and int(coeff) == 0
        ]

        extra_input_asts = [clone_ast(expr) for expr in z_exprs[1:]]
        for idx, coeff in zip(var_idxs, bundle_coeffs):
            if int(coeff) != 0 or not _is_compound_token(idx):
                continue
            try:
                extra_input_asts.append(clone_ast(_compound_ast_for_token(z_ast_existing, idx)))
            except Exception:
                pass

        meta = {
            "kind": "power_difference_bundle",
            "power": 1,
            "indices": tuple(component_indices),
            "source": "bundle",
            "component_sources": tuple(component_sources),
            "bundle_size": int(len(selected)),
            "extra_input_asts": extra_input_asts,
            "from_invariance": any((p[4] or {}).get("from_invariance", False) for p in selected),
        }
        return (
            tuple(bundle_coeffs),
            z_exprs[0],
            float(min(confs)),
            extra_raw,
            meta,
        )

    # --- Primary: gradient-ratio scan (detects all n) ---
    ratio_hits = _scan_gradient_ratio_pairs(x_vals, dydx_vals)

    for i_local, j_local, n, ratio_conf in ratio_hits:
        i_global = var_idxs[i_local]
        j_global = var_idxs[j_local]

        # When extending a compound with only 2 inputs (z + 1 extra),
        # only z-involving pairs make sense (the extra alone can't form a diff).
        pair_has_z = _is_compound_token(i_global) or _is_compound_token(j_global)
        if z_ast_existing is not None and not pair_has_z and len(var_idxs) <= 2:
            continue

        pair = _pair_key(i_global, j_global)
        if pair in tested_pairs:
            continue
        if not _difference_dims_compatible(i_global, j_global):
            continue

        # Verify with gradient structure test
        verify_conf = _test_power_difference_structure(
            x_vals, dydx_vals, i_local, j_local, n, precision
        )
        conf = min(ratio_conf, verify_conf)

        if conf < 1.0 - precision:
            continue
        tested_pairs.add(pair)

        # Build AST
        linear_coeffs = tuple(
            (n if idx == i_global else -n if idx == j_global else 0)
            for idx in var_idxs
        )
        if z_ast_existing is not None and (_is_compound_token(i_global) or _is_compound_token(j_global)):
            # Z-token-aware AST: inline each matching existing compound input.
            def _power_diff_part(global_idx):
                if _is_compound_token(global_idx):
                    base = clone_ast(_compound_ast_for_token(z_ast_existing, global_idx))
                else:
                    base = Var(int(global_idx))
                return base if n == 1 else PowNode(base, n)

            left_part = _power_diff_part(i_global)
            right_part = _power_diff_part(j_global)
            z_ast = AddNode(left_part, MulNode(ConstNode(-1.0), right_part))
        elif n == 1:
            z_ast = build_linear_ast((i_global, j_global), (1, -1))
        else:
            z_ast = build_power_difference_ast(i_global, j_global, n)

        # Extra variables with zero coefficient become extra_override.
        # For extra-vs-extra pairs the existing compound z is an extra too.
        if z_ast_existing is not None:
            extra_override = sorted(
                int(idx) for idx, coeff in zip(var_idxs, linear_coeffs)
                if isinstance(idx, int) and coeff == 0
            )
            # extra-vs-extra: z_ast_existing preserved via meta["preserve_z_ast"] below
        else:
            extra_override = None

        meta = {
            "kind": "power_difference",
            "power": int(n),
            "indices": (_safe_idx(i_global), _safe_idx(j_global)),
            "source": "gradient_ratio",
            "from_invariance": False,
        }
        if extra_override is not None:
            meta["extra_override"] = extra_override
        if z_ast_existing is not None and not pair_has_z:
            try:
                meta["preserve_z_ast"] = clone_ast(_compound_ast_for_token(z_ast_existing, _COMPOUND_Z_TOKEN))
            except Exception:
                pass
        proposals.append((linear_coeffs, z_ast, conf, extra_override, meta))
        print(
            "[Compound PureDiff] Found z = "
            f"{_power_pair_term_label(i_global, int(n), False)} - "
            f"{_power_pair_term_label(j_global, int(n), False)}, conf={conf:.3f}"
        )

    # --- Direct n=1 anti-gradient scan: gi + gj ≈ 0 -----------------------
    # This is the most direct certificate for dependence on xi - xj and does
    # not require an invariance feature to tell us which pair to test.
    for i_local in range(len(var_idxs)):
        for j_local in range(i_local + 1, len(var_idxs)):
            i_global = var_idxs[i_local]
            j_global = var_idxs[j_local]
            pair = _pair_key(i_global, j_global)
            if pair in tested_pairs:
                continue
            ag = _anti_gradient_confidence(i_local, j_local)
            if ag is None:
                continue
            conf, residual = ag
            if conf < 1.0 - precision:
                continue
            prop = _linear_difference_proposal(
                i_local,
                j_local,
                conf,
                source="anti_gradient",
                residual=residual,
            )
            if prop is None:
                continue
            tested_pairs.add(pair)
            proposals.append(prop)
            print(
                f"[Compound PureDiff] Found z = {_label(i_global)} - {_label(j_global)} "
                f"(anti-gradient residual={residual:.3g}), conf={conf:.3f}"
            )

    # --- General signed pair powers: z = z0^p +/- z1^p --------------------
    # This covers common physics coordinates such as 1/x0 - 1/x1.  Each fixed
    # form is cheap, but it is still only proposal evidence: require the local
    # gradient to be colinear with grad(z), then let the normal Stage A
    # validation/acceptance path decide whether the visible coordinate pays off.
    pair_variant_specs = (
        (1, False, False, "plus"),
        (1, True, True, "minus"),
        (1, True, True, "plus"),
        (2, False, False, "minus"),
        (2, False, False, "plus"),
        (2, True, True, "minus"),
        (2, True, True, "plus"),
        (1, True, False, "minus"),
        (1, False, True, "minus"),
        (1, True, False, "plus"),
        (1, False, True, "plus"),
        (2, True, False, "minus"),
        (2, False, True, "minus"),
        (2, True, False, "plus"),
        (2, False, True, "plus"),
    )
    tested_power_pair_variants = set()
    for i_local in range(len(var_idxs)):
        for j_local in range(i_local + 1, len(var_idxs)):
            i_global = var_idxs[i_local]
            j_global = var_idxs[j_local]
            pair_has_z = _is_compound_token(i_global) or _is_compound_token(j_global)
            if z_ast_existing is not None and not pair_has_z and len(var_idxs) <= 2:
                continue
            pair = _pair_key(i_global, j_global)
            for power, left_inv, right_inv, op in pair_variant_specs:
                # Plain xi-xj is already handled above by the ratio and
                # anti-gradient detectors.  Avoid producing duplicate linear
                # difference proposals from this broader family.
                if power == 1 and not left_inv and not right_inv and op == "minus":
                    continue
                if (
                    op == "minus"
                    and not left_inv
                    and not right_inv
                    and pair in tested_pairs
                ):
                    continue
                sig = (pair, int(power), bool(left_inv), bool(right_inv), str(op))
                if sig in tested_power_pair_variants:
                    continue
                tested_power_pair_variants.add(sig)
                if not _power_pair_dims_compatible(
                    i_global,
                    j_global,
                    power=power,
                    left_inverse=left_inv,
                    right_inverse=right_inv,
                ):
                    continue
                pc = _power_pair_gradient_confidence(
                    i_local,
                    j_local,
                    power=power,
                    left_inverse=left_inv,
                    right_inverse=right_inv,
                    op=op,
                )
                if pc is None:
                    continue
                conf, residual = pc
                if conf < 1.0 - precision:
                    continue
                prop = _power_pair_sumdiff_proposal(
                    i_local,
                    j_local,
                    conf,
                    residual=residual,
                    power=power,
                    left_inverse=left_inv,
                    right_inverse=right_inv,
                    op=op,
                )
                if prop is None:
                    continue
                proposals.append(prop)
                op_s = "+" if op == "plus" else "-"
                print(
                    "[Compound PureDiff] Found z = "
                    f"{_power_pair_term_label(i_global, power, left_inv)} {op_s} "
                    f"{_power_pair_term_label(j_global, power, right_inv)} "
                    f"(gradient-colinearity residual={residual:.3g}), conf={conf:.3f}"
                )

    # --- Fallback: invariance features for n=1 (catches cases ratio scan misses) ---
    # Invariance features reference original variable indices, not z-tokens,
    # so skip this fallback when extending an existing compound.
    if invariance_feats and z_ast_existing is None:
        for feat in invariance_feats:
            if feat.kind != "integer_linear":
                continue

            coeffs = feat.coeffs.numpy()
            nonzero_idxs = [i for i, c in enumerate(coeffs) if abs(c) > 0.5]
            if len(nonzero_idxs) != 2:
                continue

            c0, c1 = coeffs[nonzero_idxs[0]], coeffs[nonzero_idxs[1]]
            if c0 * c1 < 0:
                continue

            i_global, j_global = nonzero_idxs[0], nonzero_idxs[1]
            if i_global not in var_idxs or j_global not in var_idxs:
                continue

            pair = _pair_key(i_global, j_global)
            if pair in tested_pairs:
                continue

            i_local = list(var_idxs).index(i_global)
            j_local = list(var_idxs).index(j_global)

            gi = dydx_vals[:, i_local]
            gj = dydx_vals[:, j_local]
            sum_ij = gi + gj
            rel_residual = np.linalg.norm(sum_ij) / (
                np.linalg.norm(gi) + np.linalg.norm(gj) + 1e-12
            )
            conf = 1.0 - rel_residual
            if conf < 1.0 - precision:
                continue
            prop = _linear_difference_proposal(
                i_local,
                j_local,
                conf,
                source="invariance_fallback",
                residual=rel_residual,
            )
            if prop is None:
                continue
            tested_pairs.add(pair)
            proposals.append(prop)
            print(f"[Compound PureDiff] Found z = x{i_global} - x{j_global} (invariance fallback), conf={conf:.3f}")

    bundle = _make_disjoint_difference_bundle(proposals)
    if bundle is not None:
        meta = bundle[4] or {}
        try:
            readable = [
                f"({_label(i)} - {_label(j)})"
                for i, j in meta.get("indices", ())
            ]
        except Exception:
            readable = []
        print(
            "[Compound PureDiff] Bundled disjoint differences: "
            + (", ".join(readable) if readable else f"{int(meta.get('bundle_size', 0))} components")
            + f", conf={float(bundle[2]):.3f}"
        )
        proposals.insert(0, bundle)

    return proposals



def _detect_difference_product_compounds(
    x_vals,
    dydx_vals,
    var_idxs,
    invariance_feats=None,
    precision=0.1,
    f_vals=None,
):
    """
    Detect power-difference-product compounds: z = (xi^n - xj^n) * xk^p.

    Primary mechanism: gradient-ratio scan (works for all integer powers n).
    Fallback for n=1: invariance features (if provided).

    Parameters
    ----------
    x_vals : np.ndarray of shape [N, k]
        Variable values for the atom's variables.
    dydx_vals : np.ndarray of shape [N, k]
        Gradient values for the atom's variables.
    var_idxs : tuple of int
        Global variable indices for this atom.
    invariance_feats : list of FeatureSpec, optional
        Invariance features (used as fallback for n=1).
    precision : float
        Tolerance threshold for acceptance.
    f_vals : np.ndarray of shape [N] or [N, 1], optional
        Function values. Enables detection of outer prefactors.

    Returns
    -------
    list of (exponents, z_ast, confidence, None, meta)
        Each proposal contains exponent tuple, AST for z, and confidence score.
    """
    from nestynet_sr.sr_core.separability_math import (
        build_difference_product_ast,
        build_difference_product_power_ast,
        build_power_difference_product_ast,
    )

    proposals = []

    # Multiplier powers to try for xk inside the compound
    inner_powers = (-1, 2, -2)
    try:
        inner_powers = tuple(sorted({int(p) for p in inner_powers if int(p) not in (0, 1)}))
    except Exception:
        inner_powers = (-1,)

    # --- Primary: gradient-ratio scan (detects all n) ---
    ratio_hits = _scan_gradient_ratio_pairs(x_vals, dydx_vals)
    tested_triples = set()

    for i_local, j_local, n, ratio_conf in ratio_hits:
        i_global = var_idxs[i_local]
        j_global = var_idxs[j_local]

        print(
            f"[Compound DiffProd] Gradient-ratio found x{i_global}^{n} - x{j_global}^{n}, "
            f"ratio_conf={ratio_conf:.3f}, trying multiplier variables..."
        )

        # Try each remaining variable as the multiplier
        for k_local, k_global in enumerate(var_idxs):
            if k_global in (i_global, j_global):
                continue

            triple = (min(i_global, j_global), max(i_global, j_global), k_global, n)
            if triple in tested_triples:
                continue
            tested_triples.add(triple)

            # --- Base test: z = (xi^n - xj^n) * xk ---
            if n == 1:
                conf, outer_power = _test_difference_product_structure(
                    x_vals, dydx_vals, i_local, j_local, k_local, precision, f_vals=f_vals
                )
            else:
                conf, outer_power = _test_power_diff_product_structure(
                    x_vals, dydx_vals, i_local, j_local, k_local, n, precision, f_vals=f_vals
                )

            print(
                f"[Compound DiffProd] Testing (x{i_global}^{n} - x{j_global}^{n}) * x{k_global}: "
                f"conf={conf:.4f}, threshold={1.0 - precision:.2f}, outer_power={outer_power}"
            )

            if conf > 1.0 - precision:
                if n == 1:
                    z_ast = build_difference_product_ast(i_global, j_global, k_global)
                else:
                    z_ast = build_power_difference_product_ast(i_global, j_global, n, k_global)

                exponents = tuple(
                    n if idx == i_global else
                    -n if idx == j_global else
                    1 if idx == k_global else
                    0
                    for idx in var_idxs
                )

                if outer_power != 0:
                    prefactor_exps = tuple(
                        outer_power if idx == k_global else 0
                        for idx in var_idxs
                    )
                else:
                    prefactor_exps = None

                meta = {
                    "kind": "power_diffprod",
                    "power": int(n),
                    "indices": (int(i_global), int(j_global), int(k_global)),
                    "outer_power": int(outer_power),
                    "prefactor_exponents": prefactor_exps,
                }
                proposals.append((exponents, z_ast, float(conf), None, meta))
                print(
                    f"[Compound DiffProd] Found (x{i_global}^{n} - x{j_global}^{n}) * x{k_global}, "
                    f"conf={conf:.3f}, outer_power={outer_power}"
                )

            # --- Power-scaled variants: z = (xi^n - xj^n) * xk^p ---
            for p_inner in inner_powers:
                try:
                    if n == 1:
                        conf_p = _test_difference_product_power_structure(
                            x_vals, dydx_vals, i_local, j_local, k_local, p_inner, precision
                        )
                    else:
                        # For n>1, use the generalized test with xk^p
                        conf_p = _test_power_difference_structure(
                            x_vals, dydx_vals, i_local, j_local, n, precision
                        )
                        # Simple confidence reduction for non-unit multiplier power
                        conf_p *= 0.9
                except Exception:
                    continue

                if conf_p <= 1.0 - precision:
                    continue

                if n == 1:
                    z_ast = build_difference_product_power_ast(i_global, j_global, k_global, p_inner)
                else:
                    z_ast = build_power_difference_product_ast(i_global, j_global, n, k_global, p_inner)

                exponents = tuple(
                    n if idx == i_global else
                    -n if idx == j_global else
                    int(p_inner) if idx == k_global else
                    0
                    for idx in var_idxs
                )

                meta = {
                    "kind": "power_diffprod",
                    "power": int(n),
                    "indices": (int(i_global), int(j_global), int(k_global)),
                    "multiplier_power": int(p_inner),
                }
                proposals.append((exponents, z_ast, float(conf_p), None, meta))
                print(
                    f"[Compound DiffProdPow] Found (x{i_global}^{n} - x{j_global}^{n}) * x{k_global}^{int(p_inner)}, "
                    f"conf={conf_p:.3f}"
                )

    # --- Fallback: invariance features for n=1 ---
    if invariance_feats:
        for feat in invariance_feats:
            if feat.kind != "integer_linear":
                continue
            coeffs = feat.coeffs.numpy()
            nonzero_idxs = [i for i, c in enumerate(coeffs) if abs(c) > 0.5]
            if len(nonzero_idxs) != 2:
                continue
            c0, c1 = coeffs[nonzero_idxs[0]], coeffs[nonzero_idxs[1]]
            if c0 * c1 < 0:
                continue

            i_global, j_global = nonzero_idxs[0], nonzero_idxs[1]
            if i_global not in var_idxs or j_global not in var_idxs:
                continue

            i_local = list(var_idxs).index(i_global)
            j_local = list(var_idxs).index(j_global)

            for k_local, k_global in enumerate(var_idxs):
                if k_global in (i_global, j_global):
                    continue

                triple = (min(i_global, j_global), max(i_global, j_global), k_global, 1)
                if triple in tested_triples:
                    continue
                tested_triples.add(triple)

                conf, outer_power = _test_difference_product_structure(
                    x_vals, dydx_vals, i_local, j_local, k_local, precision, f_vals=f_vals
                )
                if conf <= 1.0 - precision:
                    continue

                z_ast = build_difference_product_ast(i_global, j_global, k_global)
                exponents = tuple(
                    1 if idx == i_global else
                    -1 if idx == j_global else
                    1 if idx == k_global else
                    0
                    for idx in var_idxs
                )

                if outer_power != 0:
                    prefactor_exps = tuple(
                        outer_power if idx == k_global else 0
                        for idx in var_idxs
                    )
                else:
                    prefactor_exps = None

                meta = {
                    "kind": "power_diffprod",
                    "power": 1,
                    "indices": (int(i_global), int(j_global), int(k_global)),
                    "outer_power": int(outer_power),
                    "prefactor_exponents": prefactor_exps,
                }
                proposals.append((exponents, z_ast, float(conf), None, meta))
                print(
                    f"[Compound DiffProd] Found (x{i_global} - x{j_global}) * x{k_global} "
                    f"(invariance fallback), conf={conf:.3f}"
                )

    return proposals



def _detect_compound_variable_for_atom(
    model,
    atom,
    leaf,
    datagen_train,
    device,
    max_exponent: int = 2,
    precision: float = 0.05,
    max_batches: int = 4,
    enable_linear: bool = True,
    max_linear_coeff: int = 2,
    enable_radial: bool = True,
    radial_max_group_size: int = 3,
    radial_cos_threshold: float = 0.95,
    radial_try_sqrt: bool = True,
    enable_shift: bool = True,
    shift_min_r2: float = 0.85,
    shift_min_abs_slope: float = 1.0e-6,
    shift_require_in_range: bool = True,
    shift_max_axes_per_atom: int = 2,
    scaling_features=None,
    invariance_features=None,
    trig_axis_specs=None,
    enable_mixed_compound: bool = True,
    enable_retained_axis_wrappers: bool = True,
    units_spec=None,
    enforce_units: bool = False,
    shadow_registry: ShadowRegistry | None = None,
    gs_cfg=None,
    gs_only: bool = False,
):
    """
    Detect if an NN leaf depends on a compound variable z = ∏ xᵢ^aᵢ.

    Uses SVD-based rank-1 test: if f = g(z) with z = ∏xᵢ^aᵢ, then
    uᵢ = xᵢ · ∂f/∂xᵢ is collinear with the exponent vector a.

    If full compound detection fails but we have 3+ variables, tries subset
    detection by excluding one variable at a time (prioritized by scaling
    feature frequency). This handles cases like f(x1,x2,x3,x4) = x1 * g(x1*x2/(x3*x4))
    where x1 is both an outer factor AND inside the compound.

    As a final fallback for 3+ variable atoms, tries difference-product detection
    using invariance features to find patterns like z = (xi - xj) * xk.

    Parameters
    ----------
    model : ASTCompositeAdaptor
        The current model.
    atom : AtomNode
        The atom to check.
    leaf_idx : int
        Index of the leaf in model.leaf.
    datagen_train : DataLoader
        Training data loader.
    device : torch.device
        Device to use.
    max_exponent : int
        Maximum exponent magnitude (e.g., 2 means -2 to +2).
    precision : float
        Threshold for rank-1 detection.
    max_batches : int
        Maximum batches to collect.
    scaling_features : list of ScaleSpec, optional
        Scaling features for prioritizing which variable to exclude in subset detection.
    invariance_features : list of FeatureSpec, optional
        Invariance features for detecting difference-product compounds.
    trig_axis_specs : list of TrigAxisSpec, optional
        Trig axis specifications from discover_trig_axes, used for mixed compound detection.
    enable_mixed_compound : bool
        Whether to attempt mixed (monomial * trig) compound detection.
    gs_only : bool
        Return immediately after shared GS carrier discovery.  This is used by
        the protected preflight that runs before legacy early-compound passes;
        it deliberately avoids constructing any ordinary proposal family.

    Returns
    -------
    proposals : list[tuple[tuple[int, ...], Node, float]]
        List of (exponents, z_ast, confidence) tuples.
        For subset compounds, exponents will have 0 for the excluded variable.
    local_trig_scale_specs : list[TrigScaleSpec]
        Oracle trig probe results from the leaf's local input space.
    """
    import numpy as np

    # Helper: check if a given axis in the leaf shows linear behavior (high R²).
    # When an axis is linear, trig detection can falsely flag it because feeding
    # sinusoidal input to f(x) = a*x + b produces sinusoidal output.
    def _is_axis_linear(leaf_model, axis_idx, eff_in, n_samples=500, r2_threshold=0.90):
        """Check if axis shows linear behavior (high R² for linear fit).

        Important: We hold OTHER axes constant while varying only the test axis.
        This correctly detects linearity even when the function is additive in
        multiple variables (e.g., f(z, x1) = z + x1 is linear in x1).
        """
        try:
            # Hold other axes at zero, vary only the test axis
            x_base = torch.zeros(eff_in, device=device)
            t_vals = torch.linspace(-2.0, 2.0, n_samples, device=device)
            x_test = x_base.unsqueeze(0).expand(n_samples, -1).clone()
            x_test[:, axis_idx] = t_vals

            with torch.no_grad():
                y_out = leaf_model(x_test)
            if y_out.dim() > 1:
                y_out = y_out[:, 0]

            x_axis = t_vals.cpu().numpy()
            y_vals = y_out.cpu().numpy()

            # Linear regression: y = slope * x + intercept
            x_mean = np.mean(x_axis)
            y_mean = np.mean(y_vals)
            cov_xy = np.mean((x_axis - x_mean) * (y_vals - y_mean))
            var_x = np.mean((x_axis - x_mean) ** 2)
            if var_x < 1e-12:
                return False  # Constant x, can't determine linearity
            slope = cov_xy / var_x
            intercept = y_mean - slope * x_mean

            # Compute R²
            y_pred = slope * x_axis + intercept
            ss_res = np.sum((y_vals - y_pred) ** 2)
            ss_tot = np.sum((y_vals - y_mean) ** 2)
            if ss_tot < 1e-12:
                return True  # Constant output, trivially "linear"
            r2 = 1.0 - ss_res / ss_tot

            return r2 > r2_threshold
        except Exception:
            return False  # If check fails, don't skip the axis

    kind = getattr(atom, "kind", None)
    if kind is None or str(kind).lower() != "nn":
        return [], []

    if leaf is None:
        return [], []

    cols, z_ast_existing = _atom_compound_cols(atom)
    is_compound = z_ast_existing is not None

    eff_ar = int(effective_arity(atom))
    if eff_ar <= 1:
        return [], []  # Single variable can't have compound structure

    x_list = []
    y_list = []
    dydx_list = []

    n_batches = 0
    for batch in datagen_train:
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch
        x = x.to(device)

        with torch.no_grad():
            x_sub, _, _ = eval_inputs(atom, x, need_grad=False, need_hess=False)
            f = leaf(x_sub)
            if f.dim() == 1:
                f = f.view(-1, 1)
            cache = {"x": x_sub}
            g = leaf.grad(cache)
            # Expected shape [B, O, k]; fall back gracefully from [B, k].
            if g.dim() == 2:
                g = g.unsqueeze(1)
            g = g[:, 0, :]  # [B, k]

        x_list.append(x_sub.detach().cpu().numpy())
        y_list.append(f[:, 0].detach().cpu().numpy())
        dydx_list.append(g.detach().cpu().numpy())

        n_batches += 1
        if n_batches >= max_batches:
            break

    if not x_list:
        return [], []

    x_vals = np.concatenate(x_list, axis=0)  # [N, k]
    y_vals = np.concatenate(y_list, axis=0)  # [N]
    dydx_vals = np.concatenate(dydx_list, axis=0)  # [N, k]

    # GS quotient coordinates are ordinary Stage-A proposals. They remain
    # subject to the same training, validation, unit, and CoE gates as every
    # other compound coordinate. Existing compound-token inputs are skipped
    # because the current GS bridge is defined on raw coordinate axes.
    gs_proposals = []
    try:
        import os as _os

        # Retain the environment opt-out as a backwards-compatible manual
        # diagnostic switch. GS proposals no longer trigger provisional
        # whole-run rollback: they must earn ordinary Stage-A acceptance.
        gs_suppressed = _os.environ.get("NNSR_SUPPRESS_GS_STAGEA") == "1"
        gs_active = bool(
            (not gs_suppressed)
            and gs_cfg is not None
            and getattr(gs_cfg, "active", lambda: False)()
        )
        raw_integer_cols = bool(not is_compound and all(isinstance(c, int) for c in cols))
        if gs_active and raw_integer_cols:
            from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals

            gs_proposals, gs_diagnostics = stageA_generalized_symmetry_proposals(
                atom=atom,
                leaf=leaf,
                x_vals=x_vals,
                y_vals=y_vals,
                dydx_vals=dydx_vals,
                cols=tuple(int(c) for c in cols),
                device=device,
                cfg=gs_cfg,
                units_spec=units_spec,
            )
            accepted_count = sum(bool(row.get("accepted", False)) for row in (gs_diagnostics or []))
            if gs_proposals or gs_diagnostics:
                print(
                    f"[Stage A GS] audited={len(gs_diagnostics or [])}, "
                    f"accepted={accepted_count}, proposals={len(gs_proposals or [])}"
                )
            recursive_rejections = [
                row
                for row in (gs_diagnostics or ())
                if str(row.get("kind", "")) == "recursive_composition"
                and not bool(row.get("accepted", False))
                and row.get("reason")
            ]
            for row in recursive_rejections[:3]:
                metric_bits = []
                for key, label in (
                    ("joint_residual_rel", "joint"),
                    ("joint_residual_tol", "tol"),
                    ("baseline_pair_residual", "pair_baseline"),
                    ("reported_max_pair_residual", "reported_pair"),
                ):
                    try:
                        value = float(row.get(key))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        metric_bits.append(f"{label}={value:.3e}")
                suffix = f", {', '.join(metric_bits)}" if metric_bits else ""
                print(
                    "[Stage A GS Recursive] Rejected "
                    f"depth={row.get('depth', '?')}, "
                    f"route={row.get('route', '?')}, "
                    f"reason={row.get('reason')}{suffix}"
                )
            if len(recursive_rejections) > 3:
                print(
                    "[Stage A GS Recursive] "
                    f"{len(recursive_rejections) - 3} additional rejection(s) omitted."
                )
            if gs_proposals:
                try:
                    from nestynet_sr.sr_search.gate_telemetry import record_gate

                    record_gate(
                        "gs_compound_lane",
                        "proposed",
                        float(len(gs_proposals)),
                        float("nan"),
                        accepted=True,
                        context={
                            "audited": len(gs_diagnostics or []),
                            "accepted_witnesses": int(accepted_count),
                            "cols": str(tuple(int(c) for c in cols)),
                        },
                    )
                except Exception:
                    pass
    except Exception as exc:
        print(f"[Stage A GS] Detection failed: {type(exc).__name__}: {exc}")
        gs_proposals = []

    if bool(gs_only):
        return list(gs_proposals or []), []

    # ------------------------------------------------------------------
    # Local oracle probes on this leaf's input space.
    # Gives precise monomial degrees and sin/cos identification.
    # ------------------------------------------------------------------
    local_oracle_specs = []
    local_trig_scale_specs = []
    try:
        def _leaf_datagen():
            dg = datagen_train() if callable(datagen_train) else datagen_train
            for batch in dg:
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device=device)
                x_in, _, _ = eval_inputs(atom, x, need_grad=False, need_hess=False)
                yield x_in

        local_oracle_specs = probe_oracle_scaling(
            leaf, _leaf_datagen, Nxvars=eff_ar, device=device,
        )
        local_trig_scale_specs = probe_trig_scaling(
            leaf, _leaf_datagen, Nxvars=eff_ar, device=device,
            oracle_specs=local_oracle_specs,
        )
        if local_oracle_specs or local_trig_scale_specs:
            print(
                f"[Local Oracle] Fresh path: {len(local_oracle_specs)} scaling, "
                f"{len(local_trig_scale_specs)} trig specs"
            )
            for osp in local_oracle_specs:
                display = osp.compound_name if osp.compound_name else f"x{osp.indices}"
                print(f"  [Local Oracle] {display}: k≈{osp.oracle_k:.3f}, rel_std={osp.oracle_rel_std:.4f}")
            for ts in local_trig_scale_specs:
                # Map local axis to original input expression for diagnostics
                inp_exprs = get_input_exprs(atom)
                if ts.axis < len(inp_exprs):
                    inp = inp_exprs[ts.axis]
                    orig_label = f"x{inp.var_idxs[0]}" if is_trivial_input(inp) else "z"
                else:
                    inp = None
                    orig_label = f"axis{ts.axis}"
                print(
                    f"  [Local Oracle] leaf {orig_label}: {ts.trig_fn}, "
                    f"\u03c9={ts.omega:.3g}, k={ts.k_hat:.2g}, rel_std={ts.rel_std:.3g}"
                )
                if inp is not None:
                    try:
                        transform_kind, shadow_ast = _stageA_trig_shadow_from_spec(ts, inp)
                        unit_status = _stageA_shadow_unit_status(shadow_ast, units_spec, bool(enforce_units))
                        conf = max(0.0, min(1.0, 1.0 - float(getattr(ts, "rel_std", 1.0))))
                        _stageA_record_shadow_coordinate(
                            shadow_registry,
                            atom=atom,
                            base_ast=inp,
                            shadow_ast=shadow_ast,
                            transform_kind=transform_kind,
                            source="local_oracle_trig",
                            confidence=conf,
                            unit_status=unit_status,
                            evidence={
                                "axis": int(ts.axis),
                                "omega": float(ts.omega),
                                "basis_fn": str(getattr(ts, "basis_fn", "") or getattr(ts, "trig_fn", "")),
                                "rel_std": float(ts.rel_std),
                                "k_hat": float(ts.k_hat),
                            },
                        )
                    except Exception:
                        pass
    except Exception as e:
        print(f"[Local Oracle] Fresh path probes failed: {e}")

    # ------------------------------------------------------------------
    # Monomial compound detection
    #
    # We run BOTH detectors:
    #   (A) classic rank-1 test on u_i = x_i df/dx_i
    #   (B) log-derivative test on v_i = (x_i df/dx_i)/f
    #
    # The log-derivative test is robust when the leaf has an outer monomial
    # prefactor m(x) multiplying a function of a monomial z, even with overlaps.
    #
    # IMPORTANT: we always run the log-derivative test (quietly) so that we can
    # propose explicit prefactor-peel candidates even when the classic test passes.
    # This avoids accepting "approximately 1D" compounds that are incomplete up
    # to a simple monomial prefactor (common in AIF-style targets).
    # ------------------------------------------------------------------

    proposals_raw_u, _ = check_monomial_compound(
        var_idxs=tuple(range(len(cols))),  # Use local indices 0, 1, ...
        x_vals=x_vals,
        dydx_vals=dydx_vals,
        max_exponent=max_exponent,
        precision=precision,
    )

    # Quiet by default: this runs on every multivariate NN leaf.
    verbose_logderiv = bool(getattr(model, "verbose_compound", False))
    proposals_raw_ld, sigma_ratio_ld, b_perp = check_monomial_compound_logderiv(
        var_idxs=tuple(range(len(cols))),  # local indices 0,1,...
        x_vals=x_vals,
        y_vals=y_vals,
        dydx_vals=dydx_vals,
        max_exponent=max_exponent,
        precision=precision,
        verbose=verbose_logderiv,
    )

    if (not proposals_raw_u) and proposals_raw_ld:
        print(
            f"[Compound LogDeriv] Recovered {len(proposals_raw_ld)} proposal(s) "
            f"(sigma_ratio={sigma_ratio_ld}) for vars={tuple(cols)}"
        )

    # Helper: infer a small-integer monomial prefactor exponent vector from b_perp.
    def _infer_prefactor_exps(exponents_global, b_perp_vec):
        try:
            import numpy as _np

            a = _np.asarray(exponents_global, dtype=float)
            b = _np.asarray(b_perp_vec, dtype=float)
            if a.shape != b.shape:
                return None

            max_shift = int(max(2, min(6, 2 * int(max_exponent))))
            max_abs_b = int(max(4, 2 * int(max_exponent)))
            round_tol = 0.35

            best = None
            best_score = float("inf")
            for k in range(-max_shift, max_shift + 1):
                b_try = b + float(k) * a
                b_round = _np.round(b_try)
                resid = float(_np.max(_np.abs(b_try - b_round)))
                if not _np.isfinite(resid) or resid > round_tol:
                    continue
                b_int = b_round.astype(int)
                if int(_np.max(_np.abs(b_int))) > max_abs_b:
                    continue
                score = float(_np.sum(_np.abs(b_int)) + 0.25 * resid)
                if score < best_score:
                    best_score = score
                    best = b_int
            if best is None:
                return None
            pref = tuple(int(v) for v in best.tolist())
            if not any(int(v) != 0 for v in pref):
                return None
            return pref
        except Exception:
            return None

    # Build a merged proposal set.
    proposals_tmp = []

    # (A0) Clean-product lane from raw scaling evidence.
    #
    # The rank/log-derivative detectors are intentionally greedy and can be
    # pulled toward an approximate high-arity coordinate.  Keep one maximal
    # clean monomial product alive as a separate proposal lane, leaving dirty
    # axes as extras.  LM validation still decides whether it is accepted.
    clean_product_prop = _clean_monomial_product_proposal_from_scaling(
        scaling_features,
        cols,
        z_ast_existing=z_ast_existing,
        rel_std_threshold=0.05,
        k_int_threshold=0.2,
    )
    if clean_product_prop is not None:
        proposals_tmp.append(clean_product_prop)
        try:
            pat = tuple(int(v) for v in clean_product_prop[0])
            extras = [
                int(col) for col, exp in zip(cols, pat)
                if isinstance(col, int) and int(exp) == 0
            ]
            print(
                "[Compound CleanProduct] Proposing clean monomial product "
                f"z={ast_to_human_readable(clean_product_prop[1])}, extras={extras}"
            )
        except Exception:
            pass

    # (A) Classic detector proposals.
    for exponents_local, confidence in (proposals_raw_u or []):
        exponents_global = tuple(int(v) for v in exponents_local)
        try:
            z_ast = _build_monomial_ast_from_cols(cols, exponents_global, z_ast=z_ast_existing)
        except ValueError:
            continue
        proposals_tmp.append((exponents_global, z_ast, float(confidence), None, None))

    # (B) Log-derivative detector proposals (+ prefactor peel + legacy extras).
    for exponents_local, confidence in (proposals_raw_ld or []):
        exponents_global = tuple(int(v) for v in exponents_local)
        try:
            z_ast = _build_monomial_ast_from_cols(cols, exponents_global, z_ast=z_ast_existing)
        except ValueError:
            continue

        # Base proposal (no prefactor metadata). This lets the classic path compete
        # when the prefactor estimate is noisy/unreliable.
        proposals_tmp.append((exponents_global, z_ast, float(confidence), None, None))

        if b_perp is not None:
            # 1) Prefer an explicit monomial prefactor peel when b_perp rounds well.
            prefactor_exps = _infer_prefactor_exps(exponents_global, b_perp)
            if prefactor_exps is not None:
                meta = {
                    "kind": "monomial",
                    "logderiv": True,
                    "prefactor_exponents": prefactor_exps,
                }
                proposals_tmp.append((exponents_global, z_ast, float(confidence), None, meta))

            # 2) Legacy heuristic: keep some raw axes as extra inputs.
            #    This is useful when the prefactor isn't a clean monomial but the
            #    log-derivative offset suggests a persistent axis dependence.
            extra_local = {i for i, e in enumerate(exponents_global) if int(e) == 0}
            b_thresh = 0.5
            try:
                for i, b in enumerate(b_perp):
                    if abs(float(b)) > b_thresh:
                        extra_local.add(int(i))
            except Exception:
                pass
            extra_override = sorted({int(cols[i]) for i in extra_local if isinstance(cols[i], int)})
            if extra_override:
                proposals_tmp.append((exponents_global, z_ast, float(confidence) * 0.95, extra_override, None))

    # (B2) Clean monomial subproducts.
    #
    # If a high-arity monomial coordinate is detected, any nontrivial integer
    # subproduct is also a valid coordinate-compression proposal.  This keeps a
    # conservative lane alive when the greedy full z bundles clean monomial
    # factors together with a residual homogeneous block.
    subset_props = []
    for pat, _z_ast, conf, _extra_override, meta in list(proposals_tmp):
        if meta and isinstance(meta, dict) and bool(meta.get("retained_axis_wrapper", False)):
            continue
        try:
            pat_t = tuple(int(v) for v in pat)
        except Exception:
            continue
        for sub_pat in clean_subset_patterns(pat_t, max_subsets=4, min_support=2):
            try:
                z_sub_ast = _build_monomial_ast_from_cols(cols, sub_pat, z_ast=z_ast_existing)
            except ValueError:
                continue
            extra_override = sorted(
                int(col)
                for col, exp in zip(cols, sub_pat)
                if isinstance(col, int) and int(exp) == 0
            )
            sub_meta = dict(meta or {})
            sub_meta.setdefault("kind", "monomial")
            sub_meta["compound_subset"] = True
            sub_meta["source"] = "monomial_subset"
            sub_meta["parent_pattern"] = tuple(int(v) for v in pat_t)
            subset_props.append(
                (
                    tuple(int(v) for v in sub_pat),
                    z_sub_ast,
                    float(conf) * 0.985,
                    extra_override if extra_override else None,
                    sub_meta,
                )
            )
    proposals_tmp.extend(subset_props)

    # (C) Extension-only scan for existing compound atoms.
    # Once Stage A has accepted a coordinate q, the remaining raw extras can
    # still contain their own arity-reducing coordinate, e.g.
    #   NN[q=x2*x3, x0, x1] -> NN[p=x0*x1, q].
    # The full local detector often prefers q/p or p/q in that situation; keep
    # this extra-only proposal alive so the normal LM/units gates can decide.
    if is_compound:
        raw_extra_local_idxs = [i for i, c in enumerate(cols) if isinstance(c, int)]
        if len(raw_extra_local_idxs) >= 2:
            try:
                x_vals_extra = x_vals[:, raw_extra_local_idxs]
                dydx_vals_extra = dydx_vals[:, raw_extra_local_idxs]
                extra_props, extra_sigma = check_monomial_compound(
                    var_idxs=tuple(range(len(raw_extra_local_idxs))),
                    x_vals=x_vals_extra,
                    dydx_vals=dydx_vals_extra,
                    max_exponent=max_exponent,
                    precision=precision,
                )
            except Exception:
                extra_props, extra_sigma = [], None

            if extra_props:
                for exp_extra, confidence in extra_props:
                    exp_extra = tuple(int(v) for v in exp_extra)
                    if not any(int(v) != 0 for v in exp_extra):
                        continue
                    subset_cols = [cols[i] for i in raw_extra_local_idxs]
                    try:
                        z_ast = _build_monomial_ast_from_cols(
                            subset_cols,
                            exp_extra,
                            z_ast=z_ast_existing,
                        )
                    except ValueError:
                        continue

                    exponents_global = [0] * len(cols)
                    for local_i, exp_i in zip(raw_extra_local_idxs, exp_extra):
                        exponents_global[int(local_i)] = int(exp_i)

                    extra_override = sorted(
                        int(cols[local_i])
                        for local_i, exp_i in zip(raw_extra_local_idxs, exp_extra)
                        if int(exp_i) == 0 and isinstance(cols[local_i], int)
                    )
                    meta = {
                        "kind": "monomial",
                        "extra_only": True,
                    }
                    try:
                        meta["preserve_z_ast"] = clone_ast(
                            _compound_ast_for_token(z_ast_existing, _COMPOUND_Z_TOKEN)
                        )
                    except Exception:
                        pass

                    proposals_tmp.append(
                        (
                            tuple(exponents_global),
                            z_ast,
                            float(confidence) * 0.99,
                            extra_override if extra_override else None,
                            meta,
                        )
                    )
                    try:
                        sigma_s = (
                            f", sigma={float(extra_sigma):.4g}"
                            if extra_sigma is not None
                            else ""
                        )
                        print(
                            "[Compound ExtraOnly] Found extras-only monomial "
                            f"z={ast_to_human_readable(z_ast)} preserving existing compound, "
                            f"conf={float(confidence):.3f}{sigma_s}"
                        )
                    except Exception:
                        pass

    # (D) Retained-scale wrappers.
    # A high-confidence monomial such as r = q/x0 is often not a pure 1D
    # solution: f(q, x0) may be h(r) * g(x0).  However, exposing x0 as a raw
    # extra while r already contains x0 is a coordinate-gauge move, not a
    # confirmed Stage-A simplification.  The later acceptance gate therefore
    # requires an overlap-aware certificate: the retained raw axis must split
    # off as a simple power/constant factor, not as an arbitrary NN.
    retained_props = []
    if bool(enable_retained_axis_wrappers):
        seen_retained = set()
        for pat, z_ast, conf, extra_override, meta in list(proposals_tmp):
            if extra_override is not None:
                continue
            try:
                pat_t = tuple(int(v) for v in pat)
            except Exception:
                continue
            if len(pat_t) != len(cols):
                continue
            # If the pattern already leaves raw zero-exponent extras, the standard
            # default-extra logic already supplies a mixed coordinate.
            if any(isinstance(col, int) and int(exp) == 0 for col, exp in zip(cols, pat_t)):
                continue
            # Retained-axis wrappers are mainly for scale/ratio coordinates.  Avoid
            # multiplying proposal counts for plain all-positive products.
            has_pos = any(int(exp) > 0 for exp in pat_t)
            has_neg = any(int(exp) < 0 for exp in pat_t)
            if not (has_pos and has_neg):
                continue
            participating_raw = [
                int(col)
                for col, exp in zip(cols, pat_t)
                if isinstance(col, int) and int(exp) != 0
            ]
            if not participating_raw:
                continue
            for raw_idx in participating_raw:
                key = (pat_t, int(raw_idx))
                if key in seen_retained:
                    continue
                seen_retained.add(key)
                meta2 = dict(meta or {})
                meta2.setdefault("kind", "monomial")
                meta2["retained_axis_wrapper"] = True
                meta2["retained_axis"] = int(raw_idx)
                retained_props.append(
                    (
                        pat_t,
                        clone_ast(z_ast),
                        float(conf) * 0.995,
                        [int(raw_idx)],
                        meta2,
                    )
                )
                try:
                    print(
                        "[Compound RetainAxis] Adding wrapper "
                        f"z={ast_to_human_readable(z_ast)} with extra x{int(raw_idx)}"
                    )
                except Exception:
                    pass
    proposals_tmp.extend(retained_props)

    # Deduplicate proposals by (pattern, extras, prefactor_exponents).
    dedup = {}
    for pat, z_ast, conf, extra_override, meta in proposals_tmp:
        pref = None
        if meta and isinstance(meta, dict):
            pref = meta.get("prefactor_exponents", None)
            if pref is not None:
                try:
                    pref = tuple(int(v) for v in pref)
                except Exception:
                    pref = None
        key = (
            tuple(int(v) for v in pat),
            tuple(int(v) for v in extra_override) if extra_override is not None else None,
            tuple(pref) if pref is not None else None,
        )
        prev = dedup.get(key)
        cand_clean = bool(meta and isinstance(meta, dict) and meta.get("clean_monomial_product", False))
        prev_meta = prev[4] if (prev is not None and len(prev) > 4 and isinstance(prev[4], dict)) else {}
        prev_clean = bool(prev_meta.get("clean_monomial_product", False))
        if prev is None:
            dedup[key] = (pat, z_ast, conf, extra_override, meta)
        elif cand_clean and not prev_clean:
            dedup[key] = (pat, z_ast, max(float(conf), float(prev[2])), extra_override, meta)
        elif prev_clean and not cand_clean:
            if float(conf) > float(prev[2]):
                dedup[key] = (prev[0], prev[1], float(conf), prev[3], prev[4])
        elif float(conf) > float(prev[2]):
            dedup[key] = (pat, z_ast, conf, extra_override, meta)

    proposals = []
    for pat, z_ast, conf, extra_override, meta in dedup.values():
        if meta is not None:
            proposals.append((pat, z_ast, conf, extra_override, meta))
        elif extra_override is not None:
            proposals.append((pat, z_ast, conf, extra_override))
        else:
            proposals.append((pat, z_ast, conf))

    # If full compound detection failed but we have 3+ variables, try subsets
    # Uses best-of-subsets: try all, pick minimum sigma_ratio, accept if < 0.1.
    # This handles cases like f(x1,x2,x3,x4) = x1 * g(x1*x2/(x3*x4)) where x1
    # appears both outside AND inside the compound, breaking the rank-1 test.
    if not proposals and len(cols) >= 3:
        # Generous threshold for subset acceptance (LM validates later)
        subset_threshold = precision

        # Collect (sigma_ratio, exclude_local_idx, proposals) for all subsets
        subset_results = []

        for exclude_local_idx in range(len(cols)):
            # Build subset mask excluding this variable
            subset_local_idxs = [i for i in range(len(cols)) if i != exclude_local_idx]
            x_vals_subset = x_vals[:, subset_local_idxs]
            dydx_vals_subset = dydx_vals[:, subset_local_idxs]

            subset_proposals, sigma_ratio = check_monomial_compound(
                var_idxs=tuple(range(len(subset_local_idxs))),
                x_vals=x_vals_subset,
                dydx_vals=dydx_vals_subset,
                max_exponent=max_exponent,
                precision=subset_threshold,
            )

            if sigma_ratio is not None:
                subset_results.append((sigma_ratio, exclude_local_idx, subset_proposals))

        # Pick best subset by sigma_ratio (then by proposal confidence), but skip
        # subsets that don't yield a usable integer exponent proposal.
        if subset_results:
            subset_results.sort(
                key=lambda t: (
                    t[0],
                    -(t[2][0][1] if (t[2] and len(t[2]) > 0) else -1.0),
                )
            )

            best = None
            for sigma_ratio, exclude_local_idx, subset_proposals in subset_results:
                if sigma_ratio < subset_threshold and subset_proposals:
                    best = (sigma_ratio, exclude_local_idx, subset_proposals)
                    break

            if best is not None:
                best_sigma, best_exclude_idx, best_proposals = best
                cols[best_exclude_idx]

                for exp_subset, conf in best_proposals:
                    # Build full exponent tuple with 0 for excluded variable
                    exp_full = list(exp_subset)
                    exp_full.insert(best_exclude_idx, 0)  # Mark excluded as "extra"
                    exp_full = tuple(exp_full)

                    # Build z_ast using only the subset variables
                    subset_cols = [
                        cols[i] for i in range(len(cols)) if i != best_exclude_idx
                    ]
                    try:
                        z_ast = _build_monomial_ast_from_cols(subset_cols, exp_subset, z_ast=z_ast_existing)
                    except ValueError:
                        continue

                    proposals.append((exp_full, z_ast, conf))

    # If single exclusion didn't work and we have 4+ vars, try excluding pairs.
    # This handles cases where 2 spurious variables pollute the rank-1 test.
    if not proposals and len(cols) >= 4:
        from itertools import combinations

        for exclude_pair in combinations(range(len(cols)), 2):
            # Build subset mask excluding these two variables
            subset_local_idxs = [i for i in range(len(cols)) if i not in exclude_pair]
            x_vals_subset = x_vals[:, subset_local_idxs]
            dydx_vals_subset = dydx_vals[:, subset_local_idxs]

            subset_proposals, sigma_ratio = check_monomial_compound(
                var_idxs=tuple(range(len(subset_local_idxs))),
                x_vals=x_vals_subset,
                dydx_vals=dydx_vals_subset,
                max_exponent=max_exponent,
                precision=subset_threshold,
            )

            if sigma_ratio is not None and sigma_ratio < subset_threshold and subset_proposals:
                # Found a valid compound! Build full exponent tuple with 0s for excluded
                for exp_subset, conf in subset_proposals:
                    exp_full = [0] * len(cols)
                    subset_idx = 0
                    for i in range(len(cols)):
                        if i not in exclude_pair:
                            exp_full[i] = exp_subset[subset_idx]
                            subset_idx += 1

                    # Build z_ast using only the subset variables
                    subset_cols = [
                        cols[i] for i in range(len(cols)) if i not in exclude_pair
                    ]
                    try:
                        z_ast = _build_monomial_ast_from_cols(subset_cols, exp_subset, z_ast=z_ast_existing)
                    except ValueError:
                        continue

                    # Mark excluded vars as "extra" (they appear outside the compound)
                    extra_override = sorted([int(cols[i]) for i in exclude_pair if isinstance(cols[i], int)])
                    proposals.append((tuple(exp_full), z_ast, conf, extra_override))

                if proposals:
                    break  # Found valid compound, stop searching

    # For compound atoms: if no monomial proposal found, create a passthrough
    # so trig extension can try z * trig(ω * extra_k).
    if is_compound and not proposals and len(cols) >= 2:
        passthrough_extras = [int(c) for c in cols if isinstance(c, int)]
        if passthrough_extras:
            try:
                z_passthrough = clone_ast(_compound_ast_for_token(z_ast_existing, _COMPOUND_Z_TOKEN))
            except Exception:
                z_passthrough = None
            if z_passthrough is not None:
                proposals.append(((1,), z_passthrough, 0.9, passthrough_extras, {"kind": "passthrough"}))

    # -------------------------------------------------------------------------
    # Trig extension on extras for fresh monomial proposals.
    # After detecting a monomial compound z = x0^a * x1^b with extras [x2],
    # also try trig extension: w = z * sin(ω*x2) or w = z * cos(ω*x2).
    # This generates both options so the system can pick the best one:
    #   1. z = x0*x1 with extras [x2] → can split to NN[z] * NN[x2]
    #   2. w = x0*x1*sin(ω*x2) with no extras → single 1D compound
    # -------------------------------------------------------------------------
    if proposals and (trig_axis_specs or local_trig_scale_specs):
        trig_extended_proposals = []
        oracle_trig_as_axis = _oracle_trig_to_axis_specs(local_trig_scale_specs, cols)
        trig_source = oracle_trig_as_axis  # Only use oracle-verified trig specs
        for prop in proposals:
            # Normalize proposal to 5-tuple format
            if len(prop) == 3:
                pattern, z_ast, conf = prop
                extras = None
                meta = {"kind": "monomial"}
            elif len(prop) == 4:
                pattern, z_ast, conf, extras = prop
                meta = {"kind": "monomial"}
            else:
                pattern, z_ast, conf, extras, meta = prop

            kind = meta.get("kind", "monomial") if meta else "monomial"

            # Only extend monomial-type proposals that have extras
            if kind not in ("monomial", "linear", "radial") or not extras:
                continue
            if is_compound:
                preserves_existing_compound = bool(
                    isinstance(meta, dict) and meta.get("preserve_z_ast") is not None
                )
                if not preserves_existing_compound:
                    try:
                        for z_col, z_tok in enumerate(cols):
                            if (
                                _is_compound_token(z_tok)
                                and int(z_col) < len(pattern)
                                and _compound_pattern_entry_is_zero(pattern[int(z_col)])
                            ):
                                preserves_existing_compound = True
                                break
                    except Exception:
                        preserves_existing_compound = False
                if preserves_existing_compound:
                    try:
                        z_s = ast_to_human_readable(z_ast)
                    except Exception:
                        z_s = "z"
                    print(
                        "[Fresh Trig Extension] Skipping preserved compound lane "
                        f"z={z_s}; trig extension must not drop an existing compound coordinate."
                    )
                    continue

            # Check for trig on each extra variable (oracle-first source)
            for extra_idx in extras:
                trig_spec = next(
                    (s for s in trig_source if int(s.axis) == int(extra_idx)),
                    None,
                )
                if trig_spec is None:
                    continue

                # Oracle-first: map global extra_idx to local axis for oracle lookup
                local_axis_for_extra = cols.index(int(extra_idx)) if int(extra_idx) in cols else -1
                oracle_info = _oracle_trig_for_axis(local_axis_for_extra, local_trig_scale_specs)
                if oracle_info is not None:
                    trig_kind = oracle_info.trig_fn
                    omega = snap_omega(oracle_info.omega)
                else:
                    trig_kind = _trig_kind_from_phase(float(getattr(trig_spec, "phase", 0.0)))
                    omega = snap_omega(float(trig_spec.omega))

                # Build z * sin(ω*xk) or z * cos(ω*xk)
                omega_extra = MulNode(ConstNode(float(omega)), Var(int(extra_idx)))
                trig_ast = SinNode(omega_extra) if trig_kind == "sin" else CosNode(omega_extra)
                z_times_trig = MulNode(z_ast, trig_ast)

                # New extras = old extras minus the absorbed trig variable
                new_extras = tuple(e for e in extras if e != extra_idx)

                trig_extended_proposals.append(
                    (
                        (1,),  # pattern for "already extended"
                        z_times_trig,
                        float(conf) * 0.95,  # slight confidence penalty
                        new_extras if new_extras else None,
                        {
                            "kind": "var_times_trig",
                            "trig_fn": trig_kind,
                            "omega": float(omega),
                            "trig_var_idx": int(extra_idx),
                            "base_z_ast": z_ast,
                            "from_fresh_extension": True,  # mark origin
                        },
                    )
                )

                print(
                    f"[Fresh Trig Extension] Proposing z * {trig_kind}({omega:.3g}*x{int(extra_idx)}) "
                    f"from monomial z={ast_to_human_readable(z_ast)}, new_extras={new_extras}"
                )

        # Add extended proposals (they'll be sorted by confidence with others)
        proposals.extend(trig_extended_proposals)

    # Try difference-product compounds: z = (xi^n - xj^n) * xk
    # Gradient-ratio scan is the primary mechanism (no invariance features needed).
    # Invariance features (if present) serve as fallback for n=1.
    if not is_compound and len(cols) >= 3:
        diff_proposals = _detect_difference_product_compounds(
            x_vals=x_vals,
            dydx_vals=dydx_vals,
            var_idxs=tuple(cols),
            invariance_feats=invariance_features,
            precision=precision,
            f_vals=y_vals,
        )
        proposals.extend(diff_proposals)

    # --- Pure power-difference compounds: z = xi^n - xj^n ---
    # Gradient-ratio scan detects all integer powers n.
    # Invariance features (if present) serve as fallback for n=1.
    if len(cols) >= 2:
        pure_diff_proposals = _detect_pure_difference_compounds(
            x_vals=x_vals,
            dydx_vals=dydx_vals,
            var_idxs=tuple(cols),
            invariance_feats=invariance_features,
            precision=precision,
            z_ast_existing=z_ast_existing,
            units_spec=units_spec,
            enforce_units=bool(enforce_units),
        )
        proposals.extend(pure_diff_proposals)

    # --- Mixed compounds via scaling features + trig detection ---
    # Use scaling features to identify monomial variables (clean integer exponents),
    # then check if remaining variables show trig behavior.
    # This approach is similar to Early Compound but runs in Normal Compound Detection.
    if enable_mixed_compound and scaling_features and (trig_axis_specs or local_trig_scale_specs) and len(cols) >= 2:
        try:
            # Use shared helper to find vars with clean scaling exponents
            clean_exponents = _get_qualifying_scaling_vars(
                scaling_features,
                var_filter=set(c for c in cols if isinstance(c, int)),  # Only consider int vars in this atom
                rel_std_threshold=0.05,
                k_int_threshold=0.2,  # Slightly looser for mixed compounds
            )

            # Remaining vars = those without clean scaling
            monomial_vars = sorted(clean_exponents.keys())
            remaining_vars = [v for v in cols if isinstance(v, int) and v not in clean_exponents]

            # Check if remaining vars are ALL trig (oracle-only)
            if monomial_vars and remaining_vars:
                oracle_trig_as_axis = _oracle_trig_to_axis_specs(local_trig_scale_specs, cols)
                trig_source = oracle_trig_as_axis  # Only use oracle-verified trig specs
                trig_remaining = [s for s in trig_source
                                  if s.axis in remaining_vars]

                if len(trig_remaining) == len(remaining_vars):
                    # All remaining vars are trig! Build mixed compound.
                    # Use oracle trig probe for precise sin/cos + omega when available
                    trig_kinds = []
                    trig_omegas_final = []
                    for spec in trig_remaining:
                        local_idx = cols.index(int(spec.axis)) if int(spec.axis) in cols else -1
                        oracle_info = _oracle_trig_for_axis(local_idx, local_trig_scale_specs)
                        if oracle_info is not None:
                            trig_kinds.append(oracle_info.trig_fn)
                            trig_omegas_final.append(snap_omega(oracle_info.omega))
                        else:
                            trig_kinds.append("sin")  # Fallback
                            trig_omegas_final.append(float(spec.omega))

                    z_ast = build_mixed_compound_ast(
                        linear_var_idxs=tuple(monomial_vars),
                        linear_exponents=tuple(clean_exponents[v] for v in monomial_vars),
                        trig_var_idxs=tuple(s.axis for s in trig_remaining),
                        trig_omegas=tuple(trig_omegas_final),
                        trig_kinds=tuple(trig_kinds),
                        trig_phases=(0.0,) * len(trig_remaining),
                    )

                    # Build pattern for consistent interface
                    pattern = tuple(
                        clean_exponents.get(v, "trig" if v in [s.axis for s in trig_remaining] else 0)
                        for v in cols
                    )

                    # High confidence since oracle-verified
                    conf = 0.9  # oracle-verified

                    meta = {
                        "kind": "mixed_scaling",
                        "monomial_vars": tuple(monomial_vars),
                        "monomial_exponents": tuple(clean_exponents[v] for v in monomial_vars),
                        "trig_vars": tuple(s.axis for s in trig_remaining),
                        "trig_omegas": tuple(trig_omegas_final),
                    }
                    proposals.append((pattern, z_ast, conf, None, meta))
                    print(f"[Mixed Compound] Scaling-based: monomial={monomial_vars} exp={[clean_exponents[v] for v in monomial_vars]}, "
                          f"trig={[s.axis for s in trig_remaining]} ω={trig_omegas_final}, kinds={trig_kinds}")

        except Exception as e:
            print(f"[Mixed Compound] Scaling-based detection failed: {e}")

    # --- Mixed compounds: z = monomial * trig_product (e.g., x0*x1*cos(x2)) ---
    has_oracle_trig = bool(local_trig_scale_specs)
    if enable_mixed_compound and (trig_axis_specs or has_oracle_trig) and len(cols) >= 2:
        try:
            # Run LOCAL trig detection on this leaf to avoid false positives from
            # global trig specs (which were computed on the full model before separability).
            # For example, global specs might flag ALL axes as trig-like if the function
            # has a global trig structure, but after separability the leaf may only have
            # trig behavior on specific axes.
            local_trig_specs_for_mixed = []
            if leaf is not None:
                try:
                    local_trig_raw = discover_trig_axes(
                        model=leaf,
                        datagen=(
                            lambda: (
                                (eval_inputs(atom, (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device=device), need_grad=False, need_hess=False)[0])
                                for batch in (datagen_train() if callable(datagen_train) else datagen_train)
                            )
                        ),
                        Nxvars=int(effective_arity(atom)),
                        device=device,
                        strength_threshold=5.0,
                    )
                    # Map local axis indices (0, 1, ...) back to original variable indices
                    local_trig_specs_for_mixed = [
                        TrigAxisSpec(
                            axis=int(cols[s.axis]),  # Map local to original
                            omega=s.omega,
                            strength=s.strength,
                            n_points=s.n_points,
                            tmin=s.tmin,
                            tmax=s.tmax,
                            phase=s.phase,
                            basis_fn=str(getattr(s, "basis_fn", "")),
                        )
                        for s in local_trig_raw if s.axis < len(cols) and isinstance(cols[s.axis], int)
                    ]
                    if local_trig_specs_for_mixed:
                        print(
                            f"[Mixed Compound] Local trig detection found: "
                            f"{[(s.axis, f'ω={s.omega:.2f}', f'str={s.strength:.1f}') for s in local_trig_specs_for_mixed]}"
                        )
                except Exception as e:
                    print(f"[Mixed Compound] Local trig detection failed: {e}")

            # Oracle-first: derive trig axis specs from oracle probes
            oracle_trig_as_axis = _oracle_trig_to_axis_specs(local_trig_scale_specs, cols)
            if oracle_trig_as_axis:
                relevant_trig_specs = oracle_trig_as_axis
            else:
                # Log what FFT found, but don't act on it — oracle-only policy
                if local_trig_specs_for_mixed:
                    print("[Mixed Compound] FFT found trig specs but oracle disagrees; skipping trig proposals")
                relevant_trig_specs = []

            if relevant_trig_specs:
                # Remap trig specs from global to local axis indices for
                # check_mixed_compound, which operates on x_vals with local columns.
                global_to_local = {g: i for i, g in enumerate(cols) if isinstance(g, int)}
                local_trig_specs = []
                for ts in relevant_trig_specs:
                    g_idx = int(ts.axis)
                    if g_idx in global_to_local:
                        local_trig_specs.append(
                            TrigAxisSpec(
                                axis=global_to_local[g_idx],
                                omega=ts.omega,
                                strength=ts.strength,
                                n_points=ts.n_points,
                                tmin=ts.tmin,
                                tmax=ts.tmax,
                                phase=ts.phase,
                                basis_fn=str(getattr(ts, "basis_fn", "")),
                            )
                        )

                if not local_trig_specs:
                    local_trig_specs = relevant_trig_specs  # fallback (shouldn't happen)

                mixed_proposals = check_mixed_compound(
                    var_idxs=tuple(range(len(cols))),  # Local indices (0, 1, ...)
                    x_vals=x_vals,
                    dydx_vals=dydx_vals,
                    trig_axis_specs=local_trig_specs,
                    max_exponent=max_exponent,
                    precision=precision,
                )
                for mp in mixed_proposals:
                    # Map local indices back to global for AST building and meta
                    mp_linear_var_idxs = tuple(cols[i] for i in mp.linear_var_idxs)
                    mp_trig_var_idxs = tuple(cols[i] for i in mp.trig_var_idxs)

                    # Skip proposals that map to z-token (passthrough + trig extension handles those)
                    if any(_is_compound_token(t) for t in mp_linear_var_idxs) or any(_is_compound_token(t) for t in mp_trig_var_idxs):
                        continue

                    # Post-hoc override of trig_kinds/omegas from oracle probes
                    mp_trig_kinds = list(mp.trig_kinds)
                    mp_trig_omegas = list(mp.trig_omegas)
                    oracle_overridden = False
                    for ti, tv_local_idx in enumerate(mp.trig_var_idxs):
                        oracle_info = _oracle_trig_for_axis(int(tv_local_idx), local_trig_scale_specs)
                        if oracle_info is not None:
                            mp_trig_kinds[ti] = oracle_info.trig_fn
                            mp_trig_omegas[ti] = snap_omega(oracle_info.omega)
                            oracle_overridden = True
                    mp_trig_kinds = tuple(mp_trig_kinds)
                    mp_trig_omegas = tuple(mp_trig_omegas)

                    # Rebuild z_ast with global indices
                    mp_z_ast = mp.z_ast
                    try:
                        mp_z_ast = build_mixed_compound_ast(
                            linear_var_idxs=mp_linear_var_idxs,
                            linear_exponents=mp.linear_exponents,
                            trig_var_idxs=mp_trig_var_idxs,
                            trig_omegas=mp_trig_omegas,
                            trig_kinds=mp_trig_kinds,
                            trig_phases=mp.trig_phases,
                        )
                    except Exception:
                        mp_z_ast = mp.z_ast  # Keep original on failure

                    # Build a pattern tuple for consistent interface
                    # Pattern: exponents for linear vars, special marker for trig vars
                    pattern = []
                    for v in cols:
                        if v in mp_linear_var_idxs:
                            idx = mp_linear_var_idxs.index(v)
                            pattern.append(mp.linear_exponents[idx])
                        elif v in mp_trig_var_idxs:
                            # Mark trig variables with a special "trig" indicator
                            pattern.append("trig")
                        else:
                            pattern.append(0)
                    pattern = tuple(pattern)

                    meta = {
                        "kind": "mixed",
                        "linear_var_idxs": mp_linear_var_idxs,
                        "linear_exponents": mp.linear_exponents,
                        "trig_var_idxs": mp_trig_var_idxs,
                        "trig_omegas": mp_trig_omegas,
                        "trig_kinds": mp_trig_kinds,
                        "trig_phases": mp.trig_phases,
                        "trig_strengths": mp.trig_strengths,
                        "monomial_sigma_ratio": mp.monomial_sigma_ratio,
                    }
                    proposals.append((pattern, mp_z_ast, mp.overall_confidence, None, meta))
                    print(
                        f"[Mixed Compound] Detected: linear={mp_linear_var_idxs} "
                        f"exp={mp.linear_exponents}, trig={mp_trig_var_idxs} "
                        f"ω={mp_trig_omegas}, kinds={mp_trig_kinds}, conf={mp.overall_confidence:.3f}"
                        f"{' (oracle-refined)' if oracle_overridden else ''}"
                    )
        except Exception as e:
            print(f"[Mixed Compound] Detection failed: {e}")

    # --- Linear compounds: z = \sum_i c_i x_i (rank-1 gradient test) ---------
    if not is_compound and enable_linear and len(cols) >= 2:
        try:
            lin_raw, lin_sigma = check_linear_compound(
                var_idxs=tuple(range(len(cols))),
                dydx_vals=dydx_vals,
                max_coeff=int(max_linear_coeff),
                precision=float(precision),
            )
            for coeffs_local, conf in (lin_raw or []):
                # Map to global variable order and build a linear z AST.
                coeffs_global = tuple(int(c) for c in coeffs_local)
                try:
                    z_ast = build_linear_ast(tuple(cols), coeffs_global)
                except Exception:
                    continue
                meta = {
                    "kind": "linear",
                    "sigma_ratio": float(lin_sigma) if lin_sigma is not None else None,
                    "coeffs": tuple(int(c) for c in coeffs_global),
                }
                proposals.append((coeffs_global, z_ast, float(conf), None, meta))
        except Exception as e:
            print(f"[Compound Linear] Detection failed: {e}")

    # --- Metric-distance compounds -----------------------------------------
    # Try law-of-cosines and coordinate-difference metrics before the generic
    # radial detector.  These proposals are visible coordinates only; Stage A
    # still has to train/validate NN[z] and pass the usual unit/Buckingham gates.
    if enable_radial and len(cols) >= 2:
        try:
            metric_props = build_metric_distance_compound_proposals(
                atom,
                units_spec=units_spec if bool(enforce_units) else None,
                include_polar=True,
                include_cartesian=True,
                wrappers=("q",),
                max_cartesian_pairs=int(radial_max_group_size),
                max_proposals=12,
            )
            for mp in metric_props:
                proposals.append(stageA_tuple_from_proposal(mp))
                try:
                    z_desc = ast_to_human_readable(mp.z_ast)
                except Exception:
                    z_desc = str(mp.label)
                print(
                    "[Stage A Metric] Proposed "
                    f"{mp.family}:{mp.wrapper} z={z_desc} "
                    f"(conf={float(mp.confidence):.3f})"
                )
        except Exception as e:
            print(f"[Stage A Metric] Detection failed: {type(e).__name__}: {e}")

    # --- Barycentric / weighted-average compounds --------------------------
    # Visible coordinate proposals such as (w0*v0 + w1*v1)/(w0+w1).  These are
    # only proposal evidence; Stage A still needs an ordinary validated
    # NN[z, extras] or visible analytic closure before accepting anything.
    if enable_linear and len(cols) >= 4:
        try:
            bary_props = build_barycentric_compound_proposals(
                atom,
                units_spec=units_spec if bool(enforce_units) else None,
                wrappers=("z",),
                max_proposals=8,
            )
            for bp in bary_props:
                proposals.append(stageA_tuple_from_proposal(bp))
                try:
                    z_desc = ast_to_human_readable(bp.z_ast)
                except Exception:
                    z_desc = str(bp.label)
                print(
                    "[Stage A Barycentric] Proposed "
                    f"{bp.family}:{bp.wrapper} z={z_desc} "
                    f"(conf={float(bp.confidence):.3f})"
                )
        except Exception as e:
            print(f"[Stage A Barycentric] Detection failed: {type(e).__name__}: {e}")

    # --- Dimensionless log/exp coordinate lifts ----------------------------
    # These catch visible coordinates such as log(x4/x3) and exp(x0/x1).
    # A successful NN[log(z)] fit is only evidence, because an NN can absorb an
    # invertible coordinate relabeling.  Stage A therefore records these as
    # shadows and lets later rules promote them only through visible
    # simplifications.
    if enable_linear and len(cols) >= 1:
        try:
            logexp_props = build_logexp_compound_proposals(
                atom,
                units_spec=units_spec if bool(enforce_units) else None,
                wrappers=("log", "exp"),
                max_proposals=8,
            )
            _stageA_record_logexp_shadows(
                atom=atom,
                proposals=logexp_props,
                shadow_registry=shadow_registry,
                units_spec=units_spec,
                enforce_units=bool(enforce_units),
            )
        except Exception as e:
            print(f"[Stage A LogExp] Detection failed: {type(e).__name__}: {e}")

    # --- Radial compounds: r^2 = sum_i q_i^2, optionally r = sqrt(r^2) ---
    # q_i are the atom's effective inputs, so this also works after a prior
    # coordinate lift such as q0=x2-x3, q1=x0-x1.
    if enable_radial and len(cols) >= 2:
        try:
            import itertools

            import numpy as np

            x_sub = x_vals  # [N, k] local
            g_sub = dydx_vals  # [N, k] local

            k = int(len(cols))
            max_r = int(max(2, min(int(radial_max_group_size), k)))
            eps = 1e-12
            for r in range(2, max_r + 1):
                for S_local in itertools.combinations(range(k), r):
                    S_local = tuple(int(i) for i in S_local)
                    xS = x_sub[:, S_local]
                    gS = g_sub[:, S_local]

                    rx = np.linalg.norm(xS, axis=1)
                    rg = np.linalg.norm(gS, axis=1)
                    m = (rx > eps) & (rg > eps) & np.isfinite(rx) & np.isfinite(rg)
                    if int(np.sum(m)) < 50:
                        continue
                    dot = np.sum(xS[m] * gS[m], axis=1)
                    cos = np.abs(dot / (rx[m] * rg[m] + eps))
                    mean_cos = float(np.mean(cos))
                    if not np.isfinite(mean_cos):
                        continue
                    if mean_cos < float(radial_cos_threshold):
                        continue

                    # Confidence mapped to [0,1]
                    conf = (mean_cos - float(radial_cos_threshold)) / (1.0 - float(radial_cos_threshold) + 1e-12)
                    conf = float(max(0.0, min(1.0, conf)))

                    S_cols = tuple(cols[i] for i in S_local)
                    # Pattern vector aligned with atom.var_idxs: 1 for vars in S, 0 otherwise
                    pattern = tuple(1 if i in S_local else 0 for i in range(k))

                    extra_override = [
                        int(cols[i])
                        for i in range(k)
                        if i not in S_local and not _is_compound_token(cols[i])
                    ]
                    extra_input_asts = []
                    for i in range(k):
                        if i in S_local or not _is_compound_token(cols[i]):
                            continue
                        # The first compound input is auto-preserved elsewhere
                        # when its pattern entry is zero; keep later compound
                        # extras explicitly.
                        if cols[i] == _COMPOUND_Z_TOKEN:
                            continue
                        extra_input_asts.append(
                            clone_ast(_compound_ast_for_token(z_ast_existing, cols[i]))
                        )

                    # r^2 proposal. We treat r = sqrt(r^2) as a *wrapper variant*
                    # later (kind-aware wrapper generation), which avoids doubling
                    # the number of proposals we train.
                    try:
                        if is_compound:
                            z_r2 = _build_radial_r2_ast_from_cols(S_cols, z_ast_existing)
                        else:
                            z_r2 = build_radial_r2_ast(tuple(int(c) for c in S_cols))
                        meta = {
                            "kind": "radial",
                            "form": "r2",
                            "indices": tuple(S_cols),
                            "mean_abs_cos": float(mean_cos),
                            "allow_sqrt": bool(radial_try_sqrt),
                        }
                        if extra_input_asts:
                            meta["extra_input_asts"] = tuple(extra_input_asts)
                        proposals.append((
                            pattern,
                            z_r2,
                            conf,
                            extra_override if extra_override else None,
                            meta,
                        ))
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Compound Radial] Detection failed: {e}")

    # --- Preferred-origin / translation-like compounds: z = x_j - x0 ---------
    # We look for axes where df/dx_j is approximately linear in x_j:
    #   df/dx_j ≈ intercept + slope * x_j
    # which suggests structure around a preferred origin x0 = -intercept/slope.
    # This is primarily useful for centering variables to enable downstream
    # discovery (e.g. radial dependence about a non-zero center, even/odd parity).
    if not is_compound and enable_shift and len(cols) >= 2:
        try:
            import numpy as np

            k = int(len(cols))
            shift_cands = []
            for j_local in range(k):
                xj = x_vals[:, j_local]
                gj = dydx_vals[:, j_local]
                m = np.isfinite(xj) & np.isfinite(gj)
                if int(np.sum(m)) < 50:
                    continue
                xjm = xj[m]
                gjm = gj[m]
                # OLS: gj ≈ intercept + slope * xj
                xm = float(np.mean(xjm))
                ym = float(np.mean(gjm))
                xv = xjm - xm
                yv = gjm - ym
                denom = float(np.dot(xv, xv) + 1e-12)
                slope = float(np.dot(xv, yv) / denom)
                intercept = float(ym - slope * xm)
                if not (math.isfinite(slope) and math.isfinite(intercept)):
                    continue
                if abs(slope) < float(shift_min_abs_slope):
                    continue
                num = float(np.dot(xv, yv))
                r2 = float((num * num) / (float(np.dot(xv, xv) * np.dot(yv, yv)) + 1e-12))
                if not math.isfinite(r2):
                    continue
                if r2 < float(shift_min_r2):
                    continue
                x0 = float(-intercept / (slope + 1e-30))
                x_min = float(np.min(xjm))
                x_max = float(np.max(xjm))
                in_range = (x_min <= x0 <= x_max)
                if bool(shift_require_in_range) and (not in_range):
                    continue

                # Skip near-trivial shifts (x0 ~ 0 relative to data span)
                span = float(x_max - x_min)
                if math.isfinite(span) and span > 0:
                    if abs(x0) < 1e-6 * span:
                        continue

                conf = (r2 - float(shift_min_r2)) / (1.0 - float(shift_min_r2) + 1e-12)
                conf = float(max(0.0, min(1.0, conf)))

                axis_global = int(cols[j_local])
                # Pattern aligned with atom.var_idxs: 1 for the shifted axis, 0 for extras
                pattern = tuple(1 if i == j_local else 0 for i in range(k))
                z_shift = AddNode(AtomNode(kind="var", var_idxs=(axis_global,)), ConstNode(-x0))
                meta = {
                    "kind": "shift",
                    "axis": axis_global,
                    "origin": float(x0),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r2": float(r2),
                    "in_range": bool(in_range),
                }
                shift_cands.append((conf, pattern, z_shift, meta))

            if shift_cands:
                shift_cands.sort(key=lambda t: -float(t[0]))
                max_keep = int(max(0, shift_max_axes_per_atom))
                if max_keep > 0:
                    shift_cands = shift_cands[:max_keep]
                for conf, pattern, z_ast, meta in shift_cands:
                    proposals.append((pattern, z_ast, float(conf), None, meta))
        except Exception as e:
            print(f"[Compound Shift] Detection failed: {e}")

    # --- Shadow-coordinate promotion ---------------------------------------
    # Local shadows are not accepted coordinate rewrites.  They become real
    # proposals only when attached to a confirmed compound lane, reducing the
    # child NN's visible inputs.  Run after all proposal families so log/exp
    # shadows and late metric/linear/radial lanes can also participate.
    shadow_promoted = _stageA_shadow_composite_proposals(
        proposals,
        atom=atom,
        cols=cols,
        z_ast_existing=z_ast_existing,
        shadow_registry=shadow_registry,
        enforce_units=bool(enforce_units),
        x_transform_map=None,
    )
    if shadow_promoted:
        proposals.extend(shadow_promoted)
    shadow_preserved_promoted = _stageA_shadow_preserved_factor_proposals(
        atom=atom,
        cols=cols,
        shadow_registry=shadow_registry,
        enforce_units=bool(enforce_units),
        x_transform_map=None,
    )
    if shadow_preserved_promoted:
        proposals.extend(shadow_preserved_promoted)
    shadow_trig_factor_peels = _stageA_shadow_trig_factor_peel_proposals(
        atom=atom,
        cols=cols,
        shadow_registry=shadow_registry,
        enforce_units=bool(enforce_units),
        x_transform_map=None,
    )
    if shadow_trig_factor_peels:
        proposals.extend(shadow_trig_factor_peels)

    # Append GS proposals after the legacy family-specific construction. Do
    # not collapse a GS carrier into an ordinary proposal with the same AST:
    # they have separate budgets and the GS copy carries the certificate that
    # can earn one protected trial. The carrier bank already canonicalizes and
    # deduplicates GS coordinates internally.
    if gs_proposals:
        proposals.extend(gs_proposals)

    # Normalize proposal format: legacy entries are (exponents, z_ast, conf) or
    # (exponents, z_ast, conf, extra_override). New entries add a meta dict.
    normed = []
    for p in proposals:
        if len(p) == 3:
            normed.append((p[0], p[1], float(p[2]), None, {"kind": "monomial"}))
        elif len(p) == 4:
            normed.append((p[0], p[1], float(p[2]), p[3], {"kind": "monomial"}))
        else:
            normed.append(p)

    # Carry local oracle scaling evidence with every proposal.  This evidence
    # is leaf-local and may not exist in the global Stage-A scale_specs list,
    # but it is exactly what partial visible prefactor peels need to decide
    # which integer powers are clean enough to externalize.
    local_clean_scaling = _get_direct_integer_scaling_evidence(
        local_oracle_specs,
        var_filter=set(range(int(eff_ar))),
        rel_std_threshold=0.08,
        k_int_threshold=0.20,
        require_oracle=True,
    )
    if local_clean_scaling:
        evidence_tuple = tuple(
            (int(i), int(exp), float(rel))
            for i, (exp, rel) in sorted(local_clean_scaling.items())
        )
        normed2 = []
        for pattern, z_ast, conf, extra_override, meta in normed:
            meta2 = dict(meta or {})
            meta2.setdefault("local_clean_scaling_exponents", evidence_tuple)
            normed2.append((pattern, z_ast, conf, extra_override, meta2))
        normed = normed2

    # ---------------------------------------------------------------------
    # Filter out pure trig proposals (no monomial part).
    # The NN can absorb a pure trig transformation, so z=sin(x0) alone is
    # not useful. Only mixed proposals (monomial * trig) are valid.
    # ---------------------------------------------------------------------
    def _is_pure_trig(proposal):
        """Check if proposal is pure trig (no monomial component)."""
        meta = proposal[4] if len(proposal) > 4 else {}
        kind = meta.get("kind", "")

        # Pure trig: kind is "mixed" but no linear (monomial) vars
        if kind == "mixed":
            linear_vars = meta.get("linear_var_idxs", ())
            linear_exps = meta.get("linear_exponents", ())
            # Pure trig if no linear vars or all exponents are zero
            if not linear_vars or all(e == 0 for e in linear_exps):
                return True

        # Mixed scaling: similar check
        if kind == "mixed_scaling":
            monomial_vars = meta.get("monomial_vars", ())
            monomial_exps = meta.get("monomial_exponents", ())
            if not monomial_vars or all(e == 0 for e in monomial_exps):
                return True

        return False

    filtered = [p for p in normed if not _is_pure_trig(p)]
    if len(filtered) < len(normed):
        print(f"[Compound] Filtered out {len(normed) - len(filtered)} pure-trig proposal(s)")
    normed = filtered

    # ---------------------------------------------------------------------
    # Sort proposals by priority:
    # 1. Pure monomial (highest priority) - kind in {monomial, radial, linear, shift}
        # 2. Mixed monomial*trig (lower priority) - kind in {mixed, mixed_scaling}
        # 3. Within each group, sort by confidence descending
    # ---------------------------------------------------------------------
    def _proposal_sort_key(proposal):
        meta = proposal[4] if len(proposal) > 4 else {}
        kind = meta.get("kind", "monomial")
        conf = float(proposal[2]) if len(proposal) > 2 else 0.0
        extra_override = proposal[3] if len(proposal) > 3 else None

        # Priority:
        #   -1 = full oracle mixed composite; try visible analytic closure first
        #    0 = pure compound proposals
        #    1 = partial mixed composites that still leave extras
        is_mixed = kind in (
            "mixed",
            "mixed_scaling",
            "var_times_trig",
            "shadow_composite",
            "shadow_preserved_factor",
            "shadow_trig_factor_peel",
        )
        try:
            has_pattern_extra = any(_compound_pattern_entry_is_zero(v) for v in proposal[0])
        except Exception:
            has_pattern_extra = False
        full_mixed = bool(is_mixed and not extra_override and not has_pattern_extra)
        priority = -1 if full_mixed else (1 if is_mixed else 0)

        # Return (priority, -confidence) so lower priority and higher conf come first
        return (priority, -conf)

    try:
        normed.sort(key=_proposal_sort_key)
    except Exception:
        pass

    return normed, local_trig_scale_specs


def _get_qualifying_scaling_vars(
    scale_specs,
    var_filter=None,
    rel_std_threshold=0.05,
    k_int_threshold=0.15,
    require_oracle=False,
):
    """
    Extract variables with clean integer scaling exponents.

    This is shared logic used by both Early Compound detection and
    Mixed Compound detection to identify variables that follow power-law
    scaling with integer exponents.

    When exponents are fractional but share a common denominator (e.g.
    [0.5, 0.5, -0.5]), we try small multipliers d=2,3 to normalise them
    to minimum integer ratios (e.g. [1, 1, -1]).

    Parameters
    ----------
    scale_specs : list of ScaleSpec
        Output from discover_scaling_features.
    var_filter : set or list, optional
        If provided, only consider these var indices.
    rel_std_threshold : float
        Maximum relative std for "clean" scaling.
    k_int_threshold : float
        Maximum |k - round(k)| for "integer" exponent.

    Returns
    -------
    dict
        var_idx -> int(exponent) for qualifying vars.
    """
    if not scale_specs:
        return {}

    # Extract single-variable scale specs (S=[i] for each i)
    single_var_specs = {}
    for sp in scale_specs:
        if len(sp.indices) != 1:
            continue
        var_idx = sp.indices[0]
        if var_filter is not None and var_idx not in var_filter:
            continue
        # Keep the one with lowest rel_std if duplicates
        if var_idx not in single_var_specs or sp.rel_std < single_var_specs[var_idx].rel_std:
            single_var_specs[var_idx] = sp

    # Collect clean specs (low rel_std) with their k values
    clean_kvs = {}  # var_idx -> k_val
    for var_idx, sp in single_var_specs.items():
        if require_oracle and not getattr(sp, 'oracle_verified', False):
            continue
        k_val = sp.oracle_k if (require_oracle and sp.oracle_k is not None) else sp.k_hat
        rstd = sp.oracle_rel_std if (require_oracle and getattr(sp, 'oracle_rel_std', None) is not None) else sp.rel_std
        if rstd < rel_std_threshold:
            clean_kvs[var_idx] = k_val

    # Try multipliers d=1,2,3 to find integer ratios.
    # d=1 reproduces the original integer-exponent check.
    # d=2,3 catch fractional exponents like 1/2, 1/3, 2/3.
    for d in (1, 2, 3):
        qualifying = {}
        for var_idx, k_val in clean_kvs.items():
            scaled = k_val * d
            n = round(scaled)
            if abs(scaled - n) < k_int_threshold and n != 0:
                qualifying[var_idx] = int(n)
        if len(qualifying) >= 2:
            # Reduce to minimum integer ratios via GCD
            from math import gcd
            from functools import reduce
            g = reduce(gcd, (abs(v) for v in qualifying.values()))
            if g > 1:
                qualifying = {k: v // g for k, v in qualifying.items()}
            return qualifying

    return {}


def _get_direct_integer_scaling_evidence(
    scale_specs,
    var_filter=None,
    rel_std_threshold=0.08,
    k_int_threshold=0.15,
    require_oracle=False,
) -> Dict[int, Tuple[int, float]]:
    """Return single-axis integer scaling evidence without ratio rescaling.

    Unlike ``_get_qualifying_scaling_vars``, this helper does not multiply
    fractional powers by a common denominator.  It is used by partial monomial
    peels, where a visible prefactor power must be directly supported by the
    observed one-axis homogeneity evidence.
    """

    if not scale_specs:
        return {}

    allowed = None if var_filter is None else {int(v) for v in var_filter}
    out: Dict[int, Tuple[int, float]] = {}
    for sp in scale_specs:
        try:
            if len(sp.indices) != 1:
                continue
            var_idx = int(sp.indices[0])
        except Exception:
            continue
        if allowed is not None and var_idx not in allowed:
            continue
        if bool(require_oracle) and not bool(getattr(sp, "oracle_verified", False)):
            continue

        try:
            k_val = (
                float(sp.oracle_k)
                if bool(require_oracle) and getattr(sp, "oracle_k", None) is not None
                else float(sp.k_hat)
            )
            rstd = (
                float(sp.oracle_rel_std)
                if bool(require_oracle) and getattr(sp, "oracle_rel_std", None) is not None
                else float(sp.rel_std)
            )
        except Exception:
            continue
        if not (math.isfinite(k_val) and math.isfinite(rstd)):
            continue
        if rstd > float(rel_std_threshold):
            continue
        n = int(round(k_val))
        if n == 0 or abs(k_val - float(n)) > float(k_int_threshold):
            continue

        prev = out.get(var_idx)
        if prev is None or rstd < float(prev[1]):
            out[var_idx] = (n, rstd)

    return out


def _check_early_compound_from_scaling(
    scale_specs,
    Nxvars,
    rel_std_threshold=0.05,
    k_int_threshold=0.15,
    require_oracle=False,
    soft_noise_floor_raw: float = 0.0,
    search_hp=None,
):
    """
    Check if single-variable scaling exponents suggest compound variables.

    Returns a LIST of candidates sorted by size (largest first, greedy).
    This allows the caller to try the largest compound first, falling back
    to smaller ones if needed.

    Parameters
    ----------
    scale_specs : list of ScaleSpec
        Output from discover_scaling_features.
    Nxvars : int
        Number of input variables.
    rel_std_threshold : float
        Maximum relative std for "clean" scaling (default 0.05).
    k_int_threshold : float
        Maximum |k - round(k)| for "integer" exponent (default 0.15).

    Returns
    -------
    list of tuples
        Each tuple is (z_var_idxs, z_exponents, remaining_var_idxs):
        - z_var_idxs: tuple of var indices that participate in z
        - z_exponents: tuple of integer exponents for those vars
        - remaining_var_idxs: tuple of var indices NOT in z
        Sorted by size of z_var_idxs descending (largest first).
        Returns empty list if no compound candidates found.
    """
    from itertools import combinations

    # Use the shared helper to find qualifying vars
    qualifying_dict = _get_qualifying_scaling_vars(
        scale_specs,
        var_filter=None,  # Consider all vars
        rel_std_threshold=rel_std_threshold,
        k_int_threshold=k_int_threshold,
        require_oracle=require_oracle,
    )

    candidates = []
    seen = set()
    soft_props = _stageA_noisy_soft_monomial_product_proposals_from_scaling(
        scale_specs,
        tuple(range(int(Nxvars))),
        search_hp=search_hp,
        noise_floor_raw=float(soft_noise_floor_raw or 0.0),
    )
    for prop in soft_props:
        try:
            pattern = tuple(int(v) for v in prop[0])
        except Exception:
            continue
        z_var_idxs = tuple(i for i, e in enumerate(pattern) if int(e) != 0)
        if len(z_var_idxs) < 2:
            continue
        z_exponents = tuple(int(pattern[i]) for i in z_var_idxs)
        remaining = tuple(i for i in range(Nxvars) if i not in z_var_idxs)
        key = (z_var_idxs, z_exponents)
        if key in seen:
            continue
        candidates.append((z_var_idxs, z_exponents, remaining))
        seen.add(key)

    # Need at least 2 certificate variables for the deterministic lane.
    if len(qualifying_dict) < 2:
        return candidates

    # Convert to list of (var_idx, exponent) sorted by var_idx
    qualifying = sorted(qualifying_dict.items())

    # Generate all subsets of size >= 2, sorted by size descending (greedy).
    # When oracle filtering is active, also preserve one maximal raw clean
    # product lane.  The oracle can be too strict on individual axes even when
    # the raw scaling evidence clearly says "product of these clean monomial
    # axes, leave the remaining dirty axes as extras"; Stage A validation will
    # still decide whether the move is real.
    if require_oracle:
        clean_prop = _clean_monomial_product_proposal_from_scaling(
            scale_specs,
            tuple(range(int(Nxvars))),
            rel_std_threshold=rel_std_threshold,
            k_int_threshold=k_int_threshold,
        )
        if clean_prop is not None:
            pattern = tuple(int(v) for v in clean_prop[0])
            z_var_idxs = tuple(i for i, e in enumerate(pattern) if int(e) != 0)
            z_exponents = tuple(int(pattern[i]) for i in z_var_idxs)
            remaining = tuple(i for i in range(Nxvars) if i not in z_var_idxs)
            key = (z_var_idxs, z_exponents)
            if key not in seen:
                candidates.append((z_var_idxs, z_exponents, remaining))
                seen.add(key)

    for size in range(len(qualifying), 1, -1):
        for subset in combinations(qualifying, size):
            z_var_idxs = tuple(v[0] for v in subset)
            z_exponents = tuple(v[1] for v in subset)
            remaining = tuple(i for i in range(Nxvars) if i not in z_var_idxs)
            key = (z_var_idxs, z_exponents)
            if key in seen:
                continue
            candidates.append((z_var_idxs, z_exponents, remaining))
            seen.add(key)

    return candidates


def _try_early_compound_candidate(
    *,
    z_var_idxs,
    z_exponents,
    remaining_var_idxs,
    model,
    current_ast,
    atom,
    tag_to_leaf,
    datagen_train_noshuffle,
    datagen_val_noshuffle,
    device,
    dtype,
    leaf_builder,
    dual_layer_used,
    search_hp,
    lm_hp,
    loss_target_eff,
    accept_threshold_eff,
    best_val_loss,
    best_train_loss_initial,
    loss_scale,
    model_sep_output,
    y_op,
    y_op_inv,
    Nxvars,
    x_transform_map,
    z_ast_override=None,
):
    """
    Try an early compound candidate based on scaling exponents.

    If remaining_var_idxs is empty, proposes pure NN[z].
    Otherwise proposes NN[z] * NN[remaining_vars].

    Parameters
    ----------
    z_ast_override : Node | None
        If provided, use this AST for z instead of building from exponents.
        This allows passing trig-extended compounds like z*sin(ω*xk).

    Returns
    -------
    accepted : bool
    new_model : torch.nn.Module | None
    new_ast : Node | None
    new_val_loss : float | None
    """
    # Build z AST
    if z_ast_override is not None:
        z_ast = z_ast_override
    else:
        z_ast = build_monomial_ast(z_var_idxs, z_exponents)

    try:
        z_readable = ast_to_human_readable(z_ast, x_transform_map)
    except Exception:
        z_readable = f"z=x^{z_exponents}"

    # Get parent hyperparameters
    parent_num_segments = atom.kwargs.get("num_segments", search_hp.num_segments_map[dual_layer_used])
    parent_dual_layer = atom.kwargs.get("dual_layer", dual_layer_used)

    # Boost segments for 1D compounds (all vars in z)
    if not remaining_var_idxs:
        use_num_segments = max(parent_num_segments, int(getattr(search_hp, "compound_1d_num_segments", 32)))
        print(f"[Early Compound] Pure 1D compound: z={z_readable}")
        print(f"[Early Compound] Boosting segments to {use_num_segments}")

        # Build pure compound candidate: replace atom with compound version
        # All variables participate in z, so exponents for build_compound_candidate_ast
        # need to be in the same order as atom.var_idxs
        # When z_ast_override is provided, z_exponents may be None - use dummy exponents
        # since the actual z_ast is already built
        full_exponents = []
        for vi in atom.var_idxs:
            if vi in z_var_idxs:
                if z_exponents is not None:
                    idx = z_var_idxs.index(vi)
                    full_exponents.append(z_exponents[idx])
                else:
                    full_exponents.append(1)  # Dummy exponent when using z_ast_override
            else:
                full_exponents.append(0)
        full_exponents = tuple(full_exponents)

        cand_ast = _build_compound_candidate_ast(current_ast, atom, z_ast, full_exponents)

    else:
        use_num_segments = parent_num_segments
        print(f"[Early Compound] Mixed compound: NN[{z_readable}, x{list(remaining_var_idxs)}]")

        # Build general compound: single NN with z + remaining vars as inputs.
        # This avoids assuming multiplicative separability which may be wrong.
        # The NN takes compound z plus the remaining variables as explicit extra inputs.

        # Build exponents array for compound atom (only z vars have non-zero)
        # When z_ast_override is provided, z_exponents may be None - use dummy exponents
        full_exponents = []
        for vi in atom.var_idxs:
            if vi in z_var_idxs:
                if z_exponents is not None:
                    idx = z_var_idxs.index(vi)
                    full_exponents.append(z_exponents[idx])
                else:
                    full_exponents.append(1)  # Dummy exponent when using z_ast_override
            else:
                full_exponents.append(0)
        full_exponents = tuple(full_exponents)

        cand_ast = _build_compound_candidate_ast(
            current_ast, atom, z_ast, full_exponents,
            extra_var_idxs_override=tuple(remaining_var_idxs),
        )

    # Build and train the candidate model
    skip_tag = getattr(atom, "tag", None)
    reuse_map_raw = {t: leaf for t, leaf in (tag_to_leaf or {}).items() if t != skip_tag}
    reuse_leaves = _clone_reuse_leaves(reuse_map_raw, device, dtype)

    temp_model, _, cand_ast_updated = build_composite_ast(
        cand_ast,
        use_num_segments,
        dual_layer=parent_dual_layer,
        leaf_builder=leaf_builder,
        device=device,
        dtype=dtype,
        reuse_leaves=reuse_leaves,
    )
    temp_model = _apply_fit_link_to_model(temp_model, lm_hp)

    # Build acceptance threshold (shared loss-budget policy).
    max_worsening_factor = float(getattr(search_hp, "max_worsening_factor", 100.0))
    worsening_floor = float(getattr(search_hp, "worsening_floor", 1.0e-6)) * loss_scale
    n_params_base = int(model.num_parameters())
    n_params_cand = int(temp_model.num_parameters())
    acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)
    accept_threshold = _compute_accept_threshold(
        base_loss=best_val_loss,
        best_loss=best_val_loss,
        base_ast=current_ast,
        cand_ast=cand_ast_updated,
        base_params=n_params_base,
        cand_params=n_params_cand,
        loss_floor=float(loss_target_eff),
        loss_cap=float(accept_threshold_eff),
        count_weight=float(getattr(lm_hp, "select_count_weight", 1.0)),
        struct_gamma=float(getattr(lm_hp, "select_struct_gamma", 0.05)),
        param_gamma=float(getattr(lm_hp, "select_param_gamma", 0.30)),
        base_bonus_decades=float(getattr(lm_hp, "select_base_bonus_decades", 0.0)),
        sep_bonus_decades=float(getattr(lm_hp, "select_sep_bonus_decades", 0.05)),
        partial_sep_bonus_decades=float(getattr(lm_hp, "select_partial_sep_bonus_decades", 0.02)),
        # Early compound rewrites are separability-like: they often unlock the
        # correct skeleton even if they require a small temporary loss regression.
        is_separability=True,
        max_worsening_factor=max_worsening_factor,
        worsening_floor=worsening_floor,
        noise_floor=float(acceptance_noise_floor_raw),
    )
    accept_threshold, structural_target = _accept_threshold_with_structural_target(
        base_ast=current_ast,
        cand_ast=cand_ast_updated,
        accept_threshold=accept_threshold,
        loss_target_eff=loss_target_eff,
    )

    print(
        f"[Early Compound] Training candidate; accept_threshold={accept_threshold:.4e}"
    )
    if structural_target:
        print(
            "[Early Compound] Structural arity reduction target enabled: "
            f"arity signature {_nn_split_signature(current_ast)}"
            f" → {_nn_split_signature(cand_ast_updated)}, "
            f"target-quality threshold {accept_threshold:.4e}"
        )

    max_train_degradation = float(
        getattr(search_hp, "early_compound_max_train_degradation", 100.0)
    )
    lane_train_loss_cap = (
        float("inf")
        if best_train_loss_initial is None or best_train_loss_initial <= 0
        else max(max_train_degradation * best_train_loss_initial, loss_target_eff)
    )

    accepted, best_val_loss_cand, best_train_loss_cand, best_param_vec, temp_opt = fit_stageA_candidate_with_tournament(
        temp_model,
        datagen_train_noshuffle,
        datagen_val_noshuffle,
        epochs=lm_hp.epochs,
        LM_strategy=lm_hp.strategy,
        nval_patience=lm_hp.nval_patience,
        loss_target=loss_target_eff,
        accept_threshold=accept_threshold,
        epochs_min=lm_hp.epochs_min,
        chisq_tol=lm_hp.chisq_tol,
        device=device,
        epochs_awful_check=lm_hp.epochs_awful_check,
        awful_threshold=lm_hp.awful_threshold,
        log_file=lm_hp.log_file,
        log_to_console=lm_hp.log_to_console,
        log_level=lm_hp.log_level,
        lm_verbose=lm_hp.LM_verbose,
        y_op=y_op,
        y_op_inv=y_op_inv,
        max_lane_train_loss=lane_train_loss_cap,
        lm_hp=lm_hp,
    )

    # --- Training chisq sanity check ---
    # Reject if the compound's training loss is BOTH:
    #   1. Much worse than the initial model's training loss (>100× degradation)
    #   2. Worse than target-level accuracy (loss_target = 1e-7 base)
    # This ensures we don't reject compounds that achieve genuinely good accuracy
    # even if the initial model happened to be absurdly good (e.g., ~0 on easy functions).
    absolute_train_threshold = loss_target_eff  # target-level accuracy

    passes_relative_check = (
        best_train_loss_initial is None
        or best_train_loss_initial <= 0
        or best_train_loss_cand <= max_train_degradation * best_train_loss_initial
    )
    passes_absolute_check = best_train_loss_cand <= absolute_train_threshold

    if accepted and not passes_relative_check and not passes_absolute_check:
        degradation = best_train_loss_cand / best_train_loss_initial if best_train_loss_initial else float('inf')
        print(
            f"{RED}[Early Compound] Rejected:{RESET} training chisq {degradation:.0f}× worse than initial "
            f"(compound={best_train_loss_cand:.4e}, initial={best_train_loss_initial:.4e}, "
            f"max_degradation={max_train_degradation:.0f}×, absolute_threshold={absolute_train_threshold:.4e})"
        )
        return False, None, None, None

    if accepted:
        print(
            f"{GREEN}[Early Compound] ACCEPTED {RESET} with val_loss={best_val_loss_cand:.4e}"
        )
        temp_opt._update_param_groups(best_param_vec)

        # Save checkpoint
        if model_sep_output is not None:
            torch.save(
                dict(
                    y_op=y_op,
                    y_op_inv=y_op_inv,
                    Nxvars=Nxvars,
                    dual_layer=dual_layer_used,
                    x_transform=x_transform_map,
                    model_state_dict=temp_model.state_dict(),
                    ast=cand_ast_updated,
                    val_loss=best_val_loss_cand,
                    fit_y_link=getattr(lm_hp, "fit_y_link", None),
                    fit_y_link_scale=float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                ),
                model_sep_output,
            )

        return True, temp_model, cand_ast_updated, best_val_loss_cand
    else:
        print(
            f"{RED}[Early Compound] Rejected:{RESET} (val_loss={best_val_loss_cand:.4e} > threshold={accept_threshold:.4e})"
        )
        return False, None, None, None


def _build_compound_candidate_ast(
    current_ast,
    atom,
    z_ast,
    exponents=None,
    extra_var_idxs_override=None,
    prefactor_exponents=None,
    prefactor_ast=None,
    extra_input_asts=None,
    unit_handoff_metadata=None,
):
    """
    Build a candidate AST replacing an atom with a compound version.

    The new atom has the same kind/tag but with input_expr=z_ast in kwargs.
    Variables with exponent 0 in the compound are kept as additional independent
    inputs via the extra_var_idxs kwarg.

    Parameters
    ----------
    exponents : tuple, optional
        Exponent tuple from compound detection, e.g. (1, 1, 0) for z = x0*x1.
        Variables with exponent 0 are kept as extra inputs to the leaf.
    """
    new_kwargs = dict(getattr(atom, "kwargs", {}) or {})
    # Remove legacy compound keys — compound info now goes via inputs=
    new_kwargs.pop("input_expr", None)
    new_kwargs.pop("extra_var_idxs", None)
    new_kwargs.pop("compound", None)
    if isinstance(unit_handoff_metadata, dict):
        new_kwargs["_unit_handoff"] = dict(unit_handoff_metadata)

    if has_nontrivial_input(atom):
        local_inputs = tuple(get_input_exprs(atom))
    else:
        local_inputs = tuple(Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ())

    # Variables being peeled as prefactors should not also appear as extra inputs
    peeled_vars = set()
    peeled_local_idxs = set()
    if prefactor_exponents is not None:
        for i, pe in enumerate(prefactor_exponents):
            if int(pe) != 0:
                peeled_local_idxs.add(int(i))
                if i < len(local_inputs) and is_trivial_input(local_inputs[i]):
                    peeled_vars.add(int(local_inputs[i].var_idxs[0]))

    # Build inputs tuple: z_ast as compound expression + extra variables
    inputs_list = [z_ast]
    seen_extra_input_asts = set()
    if extra_var_idxs_override is not None:
        for j in extra_var_idxs_override:
            if int(j) not in peeled_vars:
                inputs_list.append(Var(int(j)))
    elif exponents is not None:
        for i, exp in enumerate(exponents):
            if (not _compound_pattern_entry_is_zero(exp)) or int(i) in peeled_local_idxs:
                continue
            if i >= len(local_inputs):
                continue
            inp = local_inputs[i]
            if is_trivial_input(inp):
                j = int(inp.var_idxs[0])
                if j not in peeled_vars:
                    inputs_list.append(Var(j))
            else:
                _append_compound_extra_input_asts(
                    inputs_list,
                    inp,
                    seen=seen_extra_input_asts,
                )
    # Append compound-expression extras (e.g., preserved z_old from extra-vs-extra PureDiff)
    if extra_input_asts:
        _append_compound_extra_input_asts(
            inputs_list,
            extra_input_asts,
            seen=seen_extra_input_asts,
        )
    inputs_list = [
        inputs_list[0],
        *_compound_extra_input_asts_after_prefactor_peel(
            atom,
            inputs_list[1:],
            prefactor_exponents=prefactor_exponents,
            prefactor_ast=prefactor_ast,
        ),
    ]

    new_atom = AtomNode(
        kind=getattr(atom, "kind", "nn"),
        var_idxs=_collect_var_idxs_from_inputs(tuple(inputs_list)),
        kwargs=new_kwargs,
        tag=getattr(atom, "tag", None),
        inputs=tuple(inputs_list),
    )

    # Optional: explicitly peel a monomial prefactor outside the leaf.
    # This supports log-derivative compound detection for targets of the form
    #   f(x) = m(x) * g(z),  z = \prod x_i^{a_i}
    # where b_perp provides an estimate of m(x) up to z^k factors.
    repl = new_atom
    if prefactor_exponents is not None:
        try:
            prefactor_exponents = tuple(int(v) for v in prefactor_exponents)
        except Exception:
            prefactor_exponents = None

    if prefactor_ast is not None:
        try:
            repl = MulNode(clone_ast(prefactor_ast), repl)
        except Exception:
            repl = new_atom
    elif prefactor_exponents is not None:
        try:
            pref_terms = []
            for inp, exp in zip(local_inputs, prefactor_exponents):
                if int(exp) == 0:
                    continue
                if is_trivial_input(inp):
                    idx = int(inp.var_idxs[0])
                    # Outer-AST Var nodes need inputs=(Var(idx),) so that
                    # _build_leaf_module sees n_in=1 (passes VarLeaf check)
                    # and eval_inputs takes the fast path (is_simple -> x[:, idxs]).
                    outer_var = AtomNode(
                        kind="var", var_idxs=(idx,), inputs=(Var(idx),)
                    )
                else:
                    outer_var = clone_ast(inp)
                if int(exp) == 1:
                    pref_terms.append(outer_var)
                else:
                    pref_terms.append(PowNode(outer_var, int(exp)))
            if pref_terms:
                pref_ast = pref_terms[0]
                for t in pref_terms[1:]:
                    pref_ast = MulNode(pref_ast, t)
                repl = MulNode(pref_ast, new_atom)
        except Exception:
            # If anything goes wrong, fall back to the plain compound leaf.
            repl = new_atom

    cand_ast = replace_atom_in_ast(current_ast, atom, repl)
    return cand_ast

__search_definitions__ = (
    "_scan_gradient_ratio_pairs",
    "_test_power_difference_structure",
    "_test_power_diff_product_structure",
    "_detect_pure_difference_compounds",
    "_detect_difference_product_compounds",
    "_detect_compound_variable_for_atom",
    "_get_qualifying_scaling_vars",
    "_get_direct_integer_scaling_evidence",
    "_check_early_compound_from_scaling",
    "_try_early_compound_candidate",
    "_build_compound_candidate_ast",
)

__search_constants__ = (

)

__search_late_bindings__ = (
    "_COMPOUND_Z_TOKEN",
    "_append_compound_extra_input_asts",
    "_atom_compound_cols",
    "_build_monomial_ast_from_cols",
    "_build_radial_r2_ast_from_cols",
    "_clean_monomial_product_proposal_from_scaling",
    "_compound_ast_for_token",
    "_compound_extra_input_asts_after_prefactor_peel",
    "_compound_pattern_entry_is_zero",
    "_is_compound_token",
    "_stageA_noisy_soft_monomial_product_proposals_from_scaling",
    "_accept_threshold_with_structural_target",
    "_nn_split_signature",
)
