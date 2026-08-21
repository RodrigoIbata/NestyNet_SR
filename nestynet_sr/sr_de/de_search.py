# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Native DE discovery from a frozen surrogate u(x).

This is a deliberately conservative first step: we search for *linear-in-coeff*
implicit DEs of the form

    A(x,u,u_x,u_xx,...) + Σ_k c_k φ_k(x,u,u_x,u_xx,...) = 0

where the feature functions φ_k are nonlinear in (x,u,du,...) but linear in the
unknown coefficients c_k. Coefficients are estimated by least-squares with a
sparse (STLSQ) thresholding loop.

Why this shape?
--------------
* It avoids any DE solving.
* It fits naturally into your SR machinery: once the equation residual is an
  AST, you can run Stage-B simplification and (optionally) refine c_k with LM.

For now we focus on **1D DEs** with an axis index `x_axis`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import torch

from nestynet_sr.adaptors.u_feature_leaf import UFeatureCache, UFeatureLeaf
from nestynet_sr.sr_core.bridges import (
    AcosNode,
    D2U,
    DU,
    Add,
    AddNode,
    AsinNode,
    AtanNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    Mul,
    MulNode,
    Node,
    Pow,
    PowNode,
    SinNode,
    U,
    Var,
    const_full_like,
)
from nestynet_sr.sr_core.ast_simplify import (
    SimplifyOptions,
    simplify_ast,
    stable_ast_key,
)
from nestynet_sr.sr_core.constants import make_unit_aware_scalar_atom

# ──────────────────────────────────────────────────────────────
# Configuration & results
# ──────────────────────────────────────────────────────────────


_EXHAUSTIVE_SUPPORT_PRUNE_MAX_TERMS = 10
_SUPPORT_PRUNE_REL_TOL = 2.0e-2
_SUPPORT_PRUNE_ABS_TOL = 1.0e-10


@dataclass
class DESearchConfig:
    x_axis: int = 0
    order_candidates: Tuple[int, ...] = (1, 2)

    # Term library controls (kept small on purpose)
    max_x_power: int = 1  # include x^p up to this p
    max_u_power: int = 2  # include u^q up to this q
    max_xu_total_degree: int = 0  # if >0, cap p+q for x^p*u^q cross terms
    include_const: bool = True  # include a constant offset term
    include_x: bool = True  # include x itself (and powers)
    include_u: bool = True
    include_du: bool = False  # include u_x in library (if not the anchor)
    include_d2u: bool = False  # include u_xx in library (if not the anchor)
    include_xu: bool = True  # include x*u (and x^p*u^q) cross terms
    include_xdu: bool = True  # include x*u_x (and x^p*u_x) cross terms
    include_inv_xdu: bool = False  # include x^-1*u_x singular cross term
    include_inv_xu: bool = False   # include x^-1*u singular cross term
    include_inv_x2u: bool = False  # include x^-2*u singular cross term (Bessel-ν)
    include_udu: bool = False  # include u*u_x cross term
    de_hard_tail_templates: bool = False
    de_hard_tail_radial_templates: bool = True
    de_hard_tail_velocity_templates: bool = False

    # STLSQ sparsification
    ridge: float = 1e-10
    stlsq_lambda: float = 1e-3
    stlsq_max_iter: int = 10

    # Sampling
    max_batches: int = 32
    max_points: int = 20000

    # Model-selection heuristic
    sparsity_penalty: float = 1e-3  # added as penalty*#terms

    # Optional dimensional analysis context
    units_spec: Any = None
    enforce_units: bool = False

    # Optional conservative AST canonicalisation. Off by default.
    ast_simplify: bool = False
    ast_simplify_level: str = "safe"
    ast_simplify_domain_policy: str = "strict"
    ast_simplify_max_passes: int = 12
    ast_simplify_validate: bool = False
    ast_simplify_trace: bool = False

    # Optional shared expression IR. Defaults preserve the current AST/STLSQ
    # library path; QDAG canonicalization is an opt-in post-merge dedupe layer.
    expr_ir: str = "ast"
    expr_canonicalize: str = "off"
    expr_domain_mode: str = "strict"
    expr_qdag_hash_cons: bool = True
    expr_qdag_flatten_ac: bool = True
    expr_qdag_combine_like_terms: bool = True
    expr_qdag_combine_powers: bool = True
    expr_qdag_constant_fold: bool = True
    expr_qdag_polynomial_islands: bool = False
    expr_qdag_rational_islands: bool = False
    expr_qdag_max_nodes: int = 200_000
    expr_qdag_max_terms_per_add: int = 128
    expr_qdag_max_factors_per_mul: int = 128
    expr_symmetry_signatures: bool = False
    expr_symmetry_prune: bool = False
    expr_invariant_coordinates: bool = False
    expr_invariant_seeds: str = "none"
    expr_deep_enable: bool = False
    expr_deep_max_depth: int | None = None
    expr_qdag_max_cost: float | None = None
    expr_qdag_max_unique: int | None = None
    expr_max_lowered_depth: int | None = None
    expr_max_lowered_size: int | None = None
    expr_gs_fss_score: bool = False
    expr_gs_fss_aux_generator: bool = False
    expr_gs_fss_max_aux_atoms: int = 0
    expr_gs_fss_max_seed_blocks: int = 0
    expr_gs_fss_max_source_fraction: float = 0.0
    expr_egraph_enable: bool = False
    expr_egraph_rules: str = "safe"
    expr_egraph_max_input_size: int = 64
    expr_egraph_max_eclasses: int = 5_000
    expr_egraph_max_enodes: int = 20_000
    expr_egraph_max_iters: int = 8
    expr_egraph_time_ms: int = 50
    expr_report: bool = False
    expr_fallback_on_error: bool = True
    expr_strict_errors: bool = False
    expr_debug_dump_examples: int = 0

    # Generalized-symmetry DE diagnostics and neutral hard-tail structural
    # priors are disabled separately, so the legacy sparse library is unchanged
    # unless an explicit opt-in flag enables one of those paths.
    gs_enable: bool = False
    gs_mode: str = "propose"
    gs_policy: str = "augment"
    gs_known_generators: bool = True
    gs_general_affine: bool = False
    gs_jet_enable: bool = True
    gs_jet_separability: bool = True
    gs_jet_multiplicative: bool = True
    gs_translations: bool = True
    gs_diagonal_translations: bool = True
    gs_scalings: bool = True
    gs_rotations: bool = True
    gs_lorentz_boosts: bool = False
    gs_output_equivariance: bool = True
    gs_residual_tol: float = 0.03
    gs_audit_residual_tol: float = 0.10
    gs_min_confidence: float = 0.65
    gs_affine_max_terms: int = 4
    gs_affine_num_candidates: int = 4
    gs_de_templates: bool = False
    gs_de_radial_templates: bool = True
    gs_de_velocity_templates: bool = False
    gs_de_all_upgrades: bool = False
    gs_de_determining_equations: bool = False
    gs_de_auto_nonlinear: bool = True
    gs_de_auto_fss: bool = True
    gs_de_auto_fss_max_attempts: int = 1
    gs_de_auto_fss_n_iter: int = 1500
    gs_de_auto_fss_n_fit: int = 1024
    gs_de_auto_fss_n_probe: int = 1024
    gs_de_auto_fss_max_depth: int = 4
    gs_de_auto_fss_return_topk: int = 8
    gs_de_contact_templates: bool = False
    gs_de_noether_templates: bool = False
    gs_de_discrete_symmetry_templates: bool = False
    gs_de_weighted_scaling_templates: bool = False
    gs_de_radial_reduction_templates: bool = False
    gs_de_invariant_library: bool = False
    gs_de_invariant_max_terms: int = 64
    gs_de_invariant_seed_generators: Any = ("d_x", "u_d_u", "x_d_x")
    gs_de_upgrade_max_terms: int = 64
    gs_de_determining_max_degree: int = 2
    gs_de_determining_max_generators: int = 4
    gs_de_determining_multiplier_degree: int = 2
    gs_de_determining_bootstraps: int = 8
    gs_de_determining_sparse_rotation: bool = True
    gs_de_determining_bracket_certificate: bool = True
    gs_de_nonlinear_invariants: bool = False
    gs_de_nonlinear_invariant_max_degree: int = 3
    gs_de_nonlinear_invariant_max_candidates: int = 8
    gs_de_nonlinear_invariant_tol: float = 0.03
    gs_de_nonlinear_orbit_coordinate: bool = True
    gs_de_compiled_nonlinear_invariants: Any = None
    gs_de_compiled_orbit_coordinate: Any = None
    gs_de_weighted_max_abs_x_power: int = 2
    gs_de_weighted_max_u_power: int = 5
    gs_de_weighted_max_du_power: int = 4
    gs_de_weighted_tol: float = 1.0e-12
    gs_de_lie_prolongation: bool = False
    gs_de_lie_use_for_selection: bool = False
    gs_de_lie_prolongation_weight: float = 0.05
    gs_de_lie_prolongation_tol: float = 0.05
    gs_de_lie_prolongation_max_samples: int = 2048
    gs_de_lie_prolongation_min_coverage: float = 0.90
    gs_de_determining_certificate: bool = False
    gs_de_certificate_tol: float = 1.0e-6
    gs_de_certificate_coeff_prune_tol: float = 0.0
    gs_de_certificate_max_samples: int = 1024
    gs_unit_torus: bool = False
    gs_pi_invariants: bool = False
    gs_dim_policy: str = "audit"
    gs_dim_both_rule: str = "rref-dominates"
    gs_dim_validator: str = "nullspace"
    gs_dim_keep_local_gates: bool = True
    gs_pi_max_exponent: int = 3
    gs_pi_max_l1: int = 6
    gs_pi_max_proposals: int = 24
    gs_pi_max_basis: int = 8
    gs_pi_rational_denom: int = 1
    gs_pi_include_free_consts: bool = True
    gs_report_dim_disagreements: bool = True


@dataclass
class DESearchResult:
    order: int
    x_axis: int
    term_asts: List[Node]
    coeffs: torch.Tensor  # (K,) for term_asts
    rms_train: float
    rms_val: Optional[float] = None
    residual_ast: Optional[Node] = None
    varpro_metadata: Optional[Dict] = None
    condition_number: Optional[float] = None
    term_sources: Optional[List[str]] = None
    prolongation_metadata: Optional[Dict] = None
    determining_certificate: Optional[Dict] = None

    def canonicalize_coeffs(self, tol: float = 1e-3) -> List[Tuple[float, str]]:
        """Snap near-integer coefficients for human-readable output.

        Returns list of (snapped_coeff, term_repr) tuples.
        """
        results = []
        for c, term in zip(self.coeffs.tolist(), self.term_asts):
            # Snap to nearest integer if close
            c_snapped = c
            for target in [0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, -2.0, -3.0]:
                if abs(c - target) < tol:
                    c_snapped = target
                    break

            if term is None:
                term_str = "1"
            else:
                try:
                    term_str = repr(term)
                except Exception:
                    # Fallback for template ASTs with const exponents
                    term_str = str(type(term).__name__)
            results.append((c_snapped, term_str))
        return results

    def format_equation(self, tol: float = 1e-3, var_name: str = "x0") -> str:
        """Format the discovered DE in human-readable form.

        Snaps near-integer coefficients and formats as equation.
        """
        if self.order == 1:
            lhs = f"u_{var_name}"
        elif self.order == 2:
            lhs = f"u_{var_name}{var_name}"
        else:
            lhs = f"d^{self.order}u/d{var_name}^{self.order}"

        terms_str = []
        canon = self.canonicalize_coeffs(tol=tol)

        for c_snap, term_str in canon:
            if abs(c_snap) < 1e-10:
                continue

            # Format coefficient and term
            if term_str == "1":
                # Constant term
                term_full = f"{c_snap:g}"
            else:
                # Replace term representations for readability
                term_clean = term_str.replace("Var(", "").replace("U(", "u").replace("DU(", "u_")
                # Remove all parentheses for cleaner output
                term_clean = term_clean.replace("(", "").replace(")", "")

                if abs(abs(c_snap) - 1.0) < 1e-10:
                    term_full = f"{term_clean}" if c_snap > 0 else f"-{term_clean}"
                else:
                    term_full = f"{c_snap:g}*{term_clean}"

            terms_str.append(term_full)

        rhs = " + ".join(terms_str).replace("+ -", "- ")
        return f"{lhs} + {rhs} = 0"


@dataclass
class DESearchResultMulti:
    """Result of multi-dataset DE discovery with shared term support.

    The discovered implicit DE has the same *active terms* across all datasets,
    but allows dataset-specific coefficients. Coefficients are returned as a
    matrix with shape (D, K_sel) matching `term_asts`.
    """

    order: int
    x_axis: int
    term_asts: List[Node]
    coeffs: torch.Tensor  # (D, K_sel)
    rms_train: List[float]
    rms_val: Optional[List[float]] = None
    dataset_ids: Optional[List[str]] = None
    residual_asts: Optional[List[Node]] = None
    term_sources: Optional[List[str]] = None
    prolongation_metadata: Optional[Dict] = None
    determining_certificate: Optional[Dict] = None

    def format_equation_for_dataset(self, d: int, tol: float = 1e-3, var_name: str = "x0") -> str:
        tmp = DESearchResult(
            order=self.order,
            x_axis=self.x_axis,
            term_asts=self.term_asts,
            coeffs=self.coeffs[d].detach().cpu(),
            rms_train=self.rms_train[d],
            rms_val=None,
        )
        return tmp.format_equation(tol=tol, var_name=var_name)


# ──────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────


def _flatten_x(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    if x is None:
        raise ValueError("Dataset returned x=None")
    if x.ndim == 1:
        return x.unsqueeze(1)
    if x.ndim == 2:
        return x
    return x.view(x.shape[0], -1)


def _as_N(t: torch.Tensor) -> torch.Tensor:
    """(N,1) -> (N,)"""
    if t.ndim == 2 and t.shape[1] == 1:
        return t[:, 0]
    if t.ndim == 1:
        return t
    raise ValueError(f"Expected (N,) or (N,1), got {tuple(t.shape)}")


def _eval_ast(node: Node, x: torch.Tensor, cache: UFeatureCache) -> torch.Tensor:
    """Evaluate an SR AST numerically using (x, u, du, d2u) from cache."""
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input"):
            if len(node.var_idxs) != 1:
                raise ValueError(f"var atom expects one index; got {node.var_idxs}")
            j = int(node.var_idxs[0])
            return x[:, j : j + 1]
        if kind in ("u", "field", "state"):
            cache.ensure(x, need_grad=False, need_hess=False)
            return cache.u
        if kind in ("du", "d1u", "grad_u"):
            axis = int(getattr(node, "kwargs", {}).get("axis", 0))
            cache.ensure(x, need_grad=True, need_hess=False)
            # cache.g has shape (N, Nout, Nx), we want (N, 1) for single output
            g = cache.g[:, :, axis : axis + 1]  # (N, Nout, 1)
            if g.shape[1] == 1:
                return g[:, 0, :]  # (N, 1)
            return g.squeeze(-1)  # (N, Nout)
        if kind in ("d2u", "ddu", "hess_u"):
            a0 = int(getattr(node, "kwargs", {}).get("axis0", 0))
            a1 = int(getattr(node, "kwargs", {}).get("axis1", 0))
            cache.ensure(x, need_grad=False, need_hess=True)
            # cache.H has shape (N, Nout, Nx, Nx), we want (N, 1) for single output
            h = cache.H[:, :, a0, a1].unsqueeze(-1)  # (N, Nout, 1)
            if h.shape[1] == 1:
                return h[:, 0, :]  # (N, 1)
            return h.squeeze(-1)  # (N, Nout)

        # Constant values (from template parameter substitution)
        if kind in ("const", "constant"):
            val_raw = getattr(node, "kwargs", {}).get("value", 1.0)
            # Reuse ConstNode coercion so NumPy / Torch scalars and complex work too.
            val = ConstNode(val_raw).value
            return const_full_like(x, (x.shape[0], 1), val)

        # Scalar constants that may appear before substitution in exploratory paths.
        if kind in ("free_const", "freeconst", "free_constant", "scale"):
            val_raw = getattr(node, "kwargs", {}).get("init", 1.0)
            val = ConstNode(val_raw).value
            return const_full_like(x, (x.shape[0], 1), val)

        # Fixed (non-trainable) scalar constants.
        if kind in ("fixed_const", "fixedconst", "fixed_constant"):
            val_raw = getattr(node, "kwargs", {}).get("value", 1.0)
            val = ConstNode(val_raw).value
            return const_full_like(x, (x.shape[0], 1), val)

        raise ValueError(f"Unsupported atom kind in DE eval: {kind!r}")

    if isinstance(node, AddNode):
        return _eval_ast(node.left, x, cache) + _eval_ast(node.right, x, cache)
    if isinstance(node, MulNode):
        return _eval_ast(node.left, x, cache) * _eval_ast(node.right, x, cache)
    if isinstance(node, PowNode):
        base_val = _eval_ast(node.base, x, cache)
        # Handle constant exponents (int/float) vs Node exponents
        if isinstance(node.exponent, (int, float)):
            return base_val.pow(node.exponent)
        else:
            exp_val = _eval_ast(node.exponent, x, cache)
            # Handle both scalar and tensor exponents
            if exp_val.numel() == 1:
                return base_val.pow(exp_val.item())
            else:
                return torch.pow(base_val, exp_val)
    if isinstance(node, LogNode):
        return torch.log(_eval_ast(node.arg, x, cache))
    if isinstance(node, ExpNode):
        return torch.exp(_eval_ast(node.arg, x, cache))
    if isinstance(node, SinNode):
        return torch.sin(_eval_ast(node.arg, x, cache))
    if isinstance(node, CosNode):
        return torch.cos(_eval_ast(node.arg, x, cache))
    if isinstance(node, ConstNode):
        return const_full_like(x, (x.shape[0], 1), node.value)

    raise TypeError(f"Unknown node type: {type(node)}")


# Re-export from lightweight numerics module so existing callers keep working.
from nestynet_sr.sr_core.numerics import rank_aware_lstsq, ridge_lstsq, stlsq  # noqa: F401


def group_stlsq(
    Phis: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
    *,
    ridge: float,
    lam: float,
    max_iter: int,
    scale_columns: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-sparse STLSQ across multiple datasets.

    Solves for coefficients C with shape (D, K) where D is number of datasets,
    while enforcing a shared *support* (row sparsity across datasets).

    Parameters
    ----------
    Phis : Sequence[torch.Tensor]
        Design matrices for each dataset, each shape (N_d, K)
    ys : Sequence[torch.Tensor]
        Target vectors for each dataset, each shape (N_d,)
    ridge : float
        Ridge regularization parameter
    lam : float
        Sparsity threshold (in original coefficient scale if scale_columns=True)
    max_iter : int
        Maximum STLSQ iterations
    scale_columns : bool, optional
        If True (default), normalize columns by their RMS across datasets for
        numerical stability. This ensures fair term selection when mixing terms
        of different magnitudes (e.g., polynomials + trig/exp). Recommended for
        Hamiltonian discovery with mixed libraries.

    Returns
    -------
    C : torch.Tensor
        Coefficient matrix, shape (D, K) in original scale
    keep_mask : torch.Tensor
        Boolean mask of selected columns, shape (K,)
    """
    if len(Phis) != len(ys):
        raise ValueError("Phis and ys must have same length")
    D = len(Phis)
    if D == 0:
        raise ValueError("Need at least one dataset")
    K = int(Phis[0].shape[1])
    for i in range(D):
        if int(Phis[i].shape[1]) != K:
            raise ValueError("All Phi matrices must share the same number of columns")

    # Column scaling for numerical stability (analogous to stlsq)
    if scale_columns:
        # Shared column scale across datasets: mean RMS per column
        col_scale = (
            torch.stack([Phi.square().mean(0) for Phi in Phis], dim=0)
            .mean(0)
            .sqrt()
            .clamp_min(1e-12)
        )
        Phis_n = [Phi / col_scale for Phi in Phis]
    else:
        col_scale = torch.ones(K, device=Phis[0].device, dtype=Phis[0].dtype)
        Phis_n = Phis

    # Initialise with per-dataset ridge on scaled matrices
    C = torch.stack([ridge_lstsq(Phis_n[i], ys[i], ridge) for i in range(D)], dim=0)  # (D,K)
    keep = torch.ones(K, dtype=torch.bool, device=Phis[0].device)

    for _ in range(max_iter):
        # Threshold in *original* coefficient scale (like stlsq)
        row_norms = torch.linalg.vector_norm(C, ord=2, dim=0) / col_scale
        keep_new = row_norms >= lam
        if int(keep_new.sum()) == 0:
            # Keep at least the strongest row to avoid degeneracy
            kmax = int(row_norms.argmax().item())
            keep_new = torch.zeros_like(keep_new)
            keep_new[kmax] = True

        if torch.equal(keep_new, keep):
            break
        keep = keep_new
        for i in range(D):
            c_sel = ridge_lstsq(Phis_n[i][:, keep], ys[i], ridge)
            C[i].zero_()
            C[i, keep] = c_sel

    # Unscale coefficients back to original scale
    return C / col_scale, keep


# ──────────────────────────────────────────────────────────────
# Units helpers
# ──────────────────────────────────────────────────────────────


def _as_real_fraction(v: Any) -> Fraction:
    """Convert numeric scalar to exact-ish Fraction, rejecting complex/non-finite."""
    c = ConstNode(v).value
    if isinstance(c, complex):
        if abs(float(c.imag)) > 1e-12:
            raise ValueError(f"Complex value is not supported here: {c!r}")
        c = float(c.real)
    f = float(c)
    if not torch.isfinite(torch.tensor(f)):
        raise ValueError(f"Non-finite scalar value: {c!r}")
    return Fraction.from_float(f).limit_denominator(128)


def _de_anchor_dim(*, order: int, x_axis: int, units_spec: Any):
    """Dimension of DE anchor term (u_x or u_xx) under UnitsSpec."""
    from nestynet_sr.sr_core.units import sub_dim

    y_dim = units_spec.y_phi_dim
    if x_axis < 0 or x_axis >= len(units_spec.x_dims):
        raise ValueError(f"x_axis {x_axis} out of range for x_dims (len={len(units_spec.x_dims)})")
    x_dim = units_spec.x_dims[x_axis]
    if int(order) == 1:
        return sub_dim(y_dim, x_dim)
    if int(order) == 2:
        return sub_dim(sub_dim(y_dim, x_dim), x_dim)
    raise ValueError(f"Unsupported DE order for units: {order}")


def _de_output_dim(units_spec: Any, out_idx: int | None = None):
    """Resolve the physical dimension of a surrogate output component."""
    out_dims = getattr(units_spec, "output_dims", None)
    if out_dims is None or out_idx is None:
        return units_spec.y_phi_dim
    j = int(out_idx)
    if j < 0 or j >= len(out_dims):
        raise ValueError(f"out_idx {j} out of range for output_dims (len={len(out_dims)})")
    local_spec = replace(units_spec, y_dim=tuple(out_dims[j]))
    return local_spec.y_phi_dim


def _infer_de_node_dim(node: Node, units_spec: Any):
    """Infer dimension of a DE AST node using UnitsSpec."""
    from nestynet_sr.sr_core.units import UnitError, add_dim, is_dimless, scale_dim, sub_dim

    us = units_spec.unit_system
    dimless = us.dimless()

    if node is None:
        return dimless

    if isinstance(node, ConstNode):
        return dimless

    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        kwargs = getattr(node, "kwargs", {}) or {}
        tag = getattr(node, "tag", None)

        if kind in ("const", "constant", "scale"):
            return dimless
        if kind in ("var", "x", "input"):
            if len(node.var_idxs) != 1:
                raise UnitError(f"var atom expects 1 index; got {node.var_idxs}")
            j = int(node.var_idxs[0])
            if j < 0 or j >= len(units_spec.x_dims):
                raise UnitError(f"var atom index out of range: x{j} (Nx={len(units_spec.x_dims)})")
            return units_spec.x_dims[j]
        if kind in ("u", "field", "state"):
            return _de_output_dim(units_spec, kwargs.get("out_idx", kwargs.get("out", kwargs.get("component", None))))
        if kind in ("du", "d1u", "grad_u"):
            axis = int(kwargs.get("axis", 0))
            if axis < 0 or axis >= len(units_spec.x_dims):
                raise UnitError(f"du axis out of range: axis={axis} (Nx={len(units_spec.x_dims)})")
            y_dim = _de_output_dim(
                units_spec,
                kwargs.get("out_idx", kwargs.get("out", kwargs.get("component", None))),
            )
            return sub_dim(y_dim, units_spec.x_dims[axis])
        if kind in ("d2u", "ddu", "hess_u"):
            a0 = int(kwargs.get("axis0", 0))
            a1 = int(kwargs.get("axis1", 0))
            if (
                a0 < 0
                or a0 >= len(units_spec.x_dims)
                or a1 < 0
                or a1 >= len(units_spec.x_dims)
            ):
                raise UnitError(
                    f"d2u axes out of range: axis0={a0}, axis1={a1} (Nx={len(units_spec.x_dims)})"
                )
            y_dim = _de_output_dim(
                units_spec,
                kwargs.get("out_idx", kwargs.get("out", kwargs.get("component", None))),
            )
            return sub_dim(sub_dim(y_dim, units_spec.x_dims[a0]), units_spec.x_dims[a1])
        if kind in ("free_const", "freeconst", "free_constant"):
            name = kwargs.get("name", None) or tag
            if name is None:
                raise UnitError("free_const leaf requires kwargs['name'] or a non-None tag")
            name = str(name)
            if name not in units_spec.free_const_dims:
                raise UnitError(
                    f"free_const {name!r} is not declared in UnitsSpec.free_const_dims"
                )
            return units_spec.free_const_dims[name]
        if kind in ("fixed_const", "fixedconst", "fixed_constant"):
            name = kwargs.get("name", None) or tag
            if name is None:
                raise UnitError("fixed_const leaf requires kwargs['name'] or a non-None tag")
            name = str(name)
            if name not in units_spec.fixed_const_dims:
                raise UnitError(
                    f"fixed_const {name!r} is not declared in UnitsSpec.fixed_const_dims"
                )
            return units_spec.fixed_const_dims[name]
        raise UnitError(f"Unsupported atom kind for DE units inference: {kind!r}")

    if isinstance(node, AddNode):
        ld = _infer_de_node_dim(node.left, units_spec)
        rd = _infer_de_node_dim(node.right, units_spec)
        if tuple(ld) != tuple(rd):
            us = units_spec.unit_system
            raise ValueError(
                f"Add dimension mismatch: {us.format_dim(ld)} vs {us.format_dim(rd)}"
            )
        return ld

    if isinstance(node, MulNode):
        ld = _infer_de_node_dim(node.left, units_spec)
        rd = _infer_de_node_dim(node.right, units_spec)
        return add_dim(ld, rd)

    if isinstance(node, PowNode):
        base_dim = _infer_de_node_dim(node.base, units_spec)
        exp = node.exponent
        exp_dim = None
        exp_frac = None

        if isinstance(exp, (int, float, Fraction)):
            exp_frac = _as_real_fraction(exp)
            exp_dim = units_spec.unit_system.dimless()
        elif isinstance(exp, ConstNode):
            exp_frac = _as_real_fraction(exp.value)
            exp_dim = units_spec.unit_system.dimless()
        elif isinstance(exp, AtomNode):
            exp_dim = _infer_de_node_dim(exp, units_spec)
            k = str(getattr(exp, "kind", "")).lower()
            if k in ("const", "constant"):
                exp_frac = _as_real_fraction(getattr(exp, "kwargs", {}).get("value", 1.0))
        else:
            # Non-atom exponent node: infer units but keep exponent symbolic.
            exp_dim = _infer_de_node_dim(exp, units_spec)

        if exp_dim is not None and not is_dimless(exp_dim):
            raise ValueError("Power exponent must be dimensionless")

        if exp_frac is None:
            # Unknown numeric exponent only safe for dimensionless bases.
            if not is_dimless(base_dim):
                raise ValueError("Non-constant exponent on a unitful base is not unit-safe")
            return units_spec.unit_system.dimless()

        return scale_dim(base_dim, exp_frac)

    if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode)):
        from nestynet_sr.sr_core.units import is_dimless

        arg_dim = _infer_de_node_dim(node.arg, units_spec)
        if not is_dimless(arg_dim):
            raise ValueError(
                f"{type(node).__name__} argument must be dimensionless under units constraints"
            )
        return units_spec.unit_system.dimless()

    raise ValueError(f"Unsupported DE node type for units inference: {type(node)}")


def required_coeff_dim_for_term(
    term: Optional[Node],
    *,
    order: int,
    x_axis: int,
    units_spec: Any,
):
    """Required coefficient dimension for anchor + c*term = 0."""
    from nestynet_sr.sr_core.units import sub_dim

    anchor_dim = _de_anchor_dim(order=order, x_axis=x_axis, units_spec=units_spec)
    term_dim = units_spec.unit_system.dimless() if term is None else _infer_de_node_dim(term, units_spec)
    return sub_dim(anchor_dim, term_dim)


def term_units_feasible(
    term: Optional[Node],
    *,
    order: int,
    x_axis: int,
    units_spec: Any,
    enforce_units: bool,
) -> Tuple[bool, str]:
    """Check if a term is dimensionally admissible in the DE residual."""
    if units_spec is None:
        return True, ""

    try:
        req_dim = required_coeff_dim_for_term(
            term,
            order=order,
            x_axis=x_axis,
            units_spec=units_spec,
        )
    except Exception as e:
        if bool(enforce_units):
            return False, str(e)
        return True, ""

    if not bool(enforce_units):
        return True, ""

    from nestynet_sr.sr_core.constants import unit_aware_scalar_choice
    from nestynet_sr.sr_core.units import is_dimless

    if is_dimless(req_dim):
        return True, ""

    choice = unit_aware_scalar_choice(req_dim, units_spec, prefer_scope="experiment")
    if choice is None:
        us = units_spec.unit_system
        return (
            False,
            f"no declared free constant matches required coefficient dim {us.format_dim(req_dim)}",
        )
    return True, ""


def _gs_dim_policy_active(cfg: "DESearchConfig") -> bool:
    policy = str(getattr(cfg, "gs_dim_policy", "audit") or "audit").strip().lower().replace("_", "-")
    return (
        bool(getattr(cfg, "gs_enable", False))
        and str(getattr(cfg, "gs_mode", "propose") or "propose").lower() != "off"
        and bool(getattr(cfg, "gs_unit_torus", False))
        and policy != "baseline"
    )


def _gs_term_units_accept(
    term: Optional[Node],
    *,
    order: int,
    x_axis: int,
    units_spec: Any,
    validator: str,
    include_free_consts: bool,
) -> tuple[bool | None, str, dict[str, Any]]:
    if units_spec is None:
        return None, "units_spec_missing", {}
    try:
        req_dim = required_coeff_dim_for_term(
            term,
            order=order,
            x_axis=x_axis,
            units_spec=units_spec,
        )
    except Exception as exc:
        return False, str(exc), {"error": str(exc)}

    try:
        us = units_spec.unit_system
        req_s = us.format_dim(req_dim)
    except Exception:
        req_s = str(req_dim)
    metadata = {"required_coeff_dim": req_s, "validator": str(validator)}

    try:
        from nestynet_sr.sr_core.units import is_dimless

        if is_dimless(req_dim):
            return True, "dimensionless_coefficient", metadata
    except Exception:
        pass

    validator_l = str(validator or "nullspace").strip().lower().replace("_", "-")
    if validator_l == "local":
        return False, f"local validator requires dimensionless coefficient; required {req_s}", metadata

    try:
        from nestynet_sr.sr_gs.unit_torus import constant_dims_from_units_spec, dim_in_rational_span

        const_dims = constant_dims_from_units_spec(
            units_spec,
            include_free=bool(include_free_consts),
            include_fixed=True,
        )
        metadata["constant_dim_count"] = int(len(const_dims))
        if dim_in_rational_span(req_dim, const_dims):
            return True, "coefficient_dim_in_declared_constant_span", metadata
    except Exception as exc:
        metadata["span_error"] = str(exc)

    return False, f"required coefficient dimension {req_s} is outside declared constant span", metadata


def term_units_feasible_under_gs_policy(
    term: Optional[Node],
    *,
    order: int,
    x_axis: int,
    cfg: "DESearchConfig",
) -> Tuple[bool, str]:
    baseline_accept, baseline_reason = term_units_feasible(
        term,
        order=order,
        x_axis=x_axis,
        units_spec=cfg.units_spec,
        enforce_units=cfg.enforce_units,
    )
    if not _gs_dim_policy_active(cfg):
        return baseline_accept, baseline_reason

    validator = str(getattr(cfg, "gs_dim_validator", "nullspace") or "nullspace")
    gs_accept, gs_reason, metadata = _gs_term_units_accept(
        term,
        order=order,
        x_axis=x_axis,
        units_spec=cfg.units_spec,
        validator=validator,
        include_free_consts=bool(getattr(cfg, "gs_pi_include_free_consts", True)),
    )
    candidate = "1" if term is None else repr(term)
    try:
        from nestynet_sr.sr_gs.dim_policy import combine_dimensional_decision, should_record_decision
        from nestynet_sr.sr_gs.reporting import record_unit_torus_event

        decision = combine_dimensional_decision(
            candidate=candidate,
            baseline_accept=bool(baseline_accept),
            gs_accept=gs_accept,
            policy=getattr(cfg, "gs_dim_policy", "audit"),
            both_rule=getattr(cfg, "gs_dim_both_rule", "rref-dominates"),
            validator=validator,
            reason=gs_reason or baseline_reason,
            metadata=metadata,
        )
        if should_record_decision(
            decision,
            report_disagreements=bool(getattr(cfg, "gs_report_dim_disagreements", True)),
        ):
            record_unit_torus_event(
                event_type="de_dimensional_decision",
                decisions=[decision.to_dict()],
                context={
                    "order": int(order),
                    "x_axis": int(x_axis),
                    "dim_policy": decision.policy,
                    "validator": decision.validator,
                },
            )
        if decision.final_accept:
            return True, ""
        return False, decision.reason or baseline_reason
    except Exception:
        return baseline_accept, baseline_reason


# ──────────────────────────────────────────────────────────────
# Library generation
# ──────────────────────────────────────────────────────────────


def _pow_if(node: Node, p: int) -> Node:
    if p == 1:
        return node
    return Pow(node, p)


def _canonical_gs_policy(policy: str | None) -> str:
    p = str(policy or "augment").strip().lower().replace("_", "-")
    if p in {"replace", "replace-baseline", "replace-shadow"}:
        return "replace-shadowed"
    if p in {"gs-only", "affine-only"}:
        return "gs-only-affine"
    if p not in {"augment", "replace-shadowed", "gs-only-affine"}:
        return "augment"
    return p


def _append_unique_term(rows: list[tuple[Node, str, str]], term: Node, source: str, family: str) -> None:
    rep = repr(term)
    for i, (old, old_source, old_family) in enumerate(rows):
        if repr(old) == rep:
            if str(source).startswith("gs") and not str(old_source).startswith("gs"):
                rows[i] = (term, source, family)
            return
    rows.append((term, source, family))


def _merge_de_label(old: str, new: str) -> str:
    vals: list[str] = []
    for item in (old, new):
        for part in str(item or "").split("|"):
            part = part.strip()
            if part and part not in vals:
                vals.append(part)
    return "|".join(vals)


def _source_prefers_new(old_source: str, new_source: str) -> bool:
    old_is_gs = str(old_source).startswith("gs")
    new_is_gs = str(new_source).startswith("gs")
    return bool(new_is_gs and not old_is_gs)


def _ast_simplify_options_from_cfg(cfg: DESearchConfig, *, context: str) -> SimplifyOptions:
    return SimplifyOptions(
        enabled=bool(getattr(cfg, "ast_simplify", False)),
        level=str(getattr(cfg, "ast_simplify_level", "safe") or "safe"),
        domain_policy=str(getattr(cfg, "ast_simplify_domain_policy", "strict") or "strict"),
        context=str(context),
        max_passes=int(getattr(cfg, "ast_simplify_max_passes", 12)),
        validate_numeric=bool(getattr(cfg, "ast_simplify_validate", False)),
        trace=bool(getattr(cfg, "ast_simplify_trace", False)),
        fail_closed=True,
    )


def _maybe_simplify_de_rows(
    rows: list[tuple[Node, str, str]],
    cfg: DESearchConfig,
    *,
    order: int,
) -> list[tuple[Node, str, str]]:
    if not bool(getattr(cfg, "ast_simplify", False)):
        return rows

    opts = _ast_simplify_options_from_cfg(cfg, context="de_term")
    by_key: dict[tuple, tuple[Node, str, str]] = {}
    changed_rows = 0
    duplicate_hits = 0
    rules_fired: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for term, source, family in rows:
        before_repr = repr(term)
        simplified, stats = simplify_ast(term, opts, units_spec=getattr(cfg, "units_spec", None))
        if stats.changed:
            changed_rows += 1
        for rule, count in stats.rules_fired.items():
            rules_fired[str(rule)] = int(rules_fired.get(str(rule), 0)) + int(count)
        key = stable_ast_key(simplified, ignore_tags=True, context="de_term")
        old = by_key.get(key)
        if old is None:
            by_key[key] = (simplified, str(source), str(family))
            if stats.changed and len(examples) < 8:
                examples.append(
                    {"before": before_repr, "after": repr(simplified), "source": str(source), "family": str(family)}
                )
            continue

        duplicate_hits += 1
        old_term, old_source, old_family = old
        if _source_prefers_new(old_source, str(source)):
            merged_source = _merge_de_label(str(source), old_source)
            merged_family = _merge_de_label(str(family), old_family)
            by_key[key] = (simplified, merged_source, merged_family)
            kept = "new"
        else:
            merged_source = _merge_de_label(old_source, str(source))
            merged_family = _merge_de_label(old_family, str(family))
            by_key[key] = (old_term, merged_source, merged_family)
            kept = "old"
        if len(examples) < 8:
            examples.append(
                {
                    "duplicate": True,
                    "kept": kept,
                    "old": repr(old_term),
                    "new": repr(simplified),
                    "source": merged_source,
                    "family": merged_family,
                }
            )

    simplified_rows = list(by_key.values())
    if bool(getattr(cfg, "gs_enable", False)):
        try:
            from nestynet_sr.sr_gs.reporting import record_policy_event

            record_policy_event(
                policy=str(getattr(cfg, "gs_policy", "augment")),
                action="ast_simplify_de_rows",
                details={
                    "order": int(order),
                    "enabled": True,
                    "level": str(getattr(cfg, "ast_simplify_level", "safe")),
                    "domain_policy": str(getattr(cfg, "ast_simplify_domain_policy", "strict")),
                    "rows_before": len(rows),
                    "rows_after": len(simplified_rows),
                    "changed_rows": int(changed_rows),
                    "duplicate_hits": int(duplicate_hits),
                    "rules_fired": rules_fired,
                    "examples": examples,
                },
            )
        except Exception:
            pass
    return simplified_rows


def _maybe_expr_ir_de_rows(
    rows: list[tuple[Node, str, str]],
    cfg: DESearchConfig,
    *,
    order: int,
) -> list[tuple[Node, str, str]]:
    from nestynet_sr.sr_expr_ir.config import coerce_expr_ir_config, expr_ir_active
    from nestynet_sr.sr_expr_ir.core_bridge import canonical_key_core_ast, maybe_canonicalize_core_ast
    from nestynet_sr.sr_expr_ir.reporting import expression_ir_report
    from nestynet_sr.sr_expr_ir.stats import ExpressionIRStats

    ir_cfg = coerce_expr_ir_config(cfg)
    if not expr_ir_active(ir_cfg):
        return rows

    stats = ExpressionIRStats()
    by_key: dict[tuple, tuple[Node, str, str]] = {}
    duplicate_hits = 0
    changed_rows = 0
    examples: list[dict[str, Any]] = []
    signature_context = {"units_spec": getattr(cfg, "units_spec", None), "order": int(order)}

    for term, source, family in rows:
        before_repr = repr(term)
        canon = maybe_canonicalize_core_ast(
            term,
            ir_cfg,
            stats=stats,
            signature_context=signature_context,
        )
        if repr(canon) != before_repr:
            changed_rows += 1
        key = canonical_key_core_ast(canon, ir_cfg, stats=stats, signature_context=signature_context)
        old = by_key.get(key)
        if old is None:
            by_key[key] = (canon, str(source), str(family))
            if repr(canon) != before_repr and len(examples) < 8:
                examples.append({"before": before_repr, "after": repr(canon), "source": str(source), "family": str(family)})
            continue

        duplicate_hits += 1
        old_term, old_source, old_family = old
        if _source_prefers_new(old_source, str(source)):
            merged_source = _merge_de_label(str(source), old_source)
            merged_family = _merge_de_label(str(family), old_family)
            by_key[key] = (canon, merged_source, merged_family)
            kept = "new"
        else:
            merged_source = _merge_de_label(old_source, str(source))
            merged_family = _merge_de_label(old_family, str(family))
            by_key[key] = (old_term, merged_source, merged_family)
            kept = "old"
        if len(examples) < 8:
            examples.append(
                {
                    "duplicate": True,
                    "kept": kept,
                    "old": repr(old_term),
                    "new": repr(canon),
                    "source": merged_source,
                    "family": merged_family,
                }
            )

    out = list(by_key.values())
    report = expression_ir_report(
        ir_cfg,
        stats,
        extra={
            "order": int(order),
            "rows_before": int(len(rows)),
            "rows_after": int(len(out)),
            "changed_rows": int(changed_rows),
            "duplicate_hits": int(duplicate_hits),
            "examples": examples,
        },
    )
    try:
        setattr(cfg, "_expr_ir_last_de_report", report)
        by_order = getattr(cfg, "_expr_ir_de_report_by_order", None)
        if not isinstance(by_order, dict):
            by_order = {}
            setattr(cfg, "_expr_ir_de_report_by_order", by_order)
        by_order[int(order)] = report
    except Exception:
        pass
    if bool(getattr(cfg, "gs_enable", False)):
        try:
            from nestynet_sr.sr_gs.reporting import record_policy_event

            record_policy_event(
                policy=str(getattr(cfg, "gs_policy", "augment")),
                action="expr_ir_de_rows",
                details=report,
            )
        except Exception:
            pass
    return out


def _baseline_de_library_rows(cfg: DESearchConfig, *, order: int) -> list[tuple[Node, str, str]]:
    """Return baseline library rows with source/family metadata."""
    xj = Var(cfg.x_axis)
    u = U()
    du = DU(cfg.x_axis)
    d2u = D2U(cfg.x_axis, cfg.x_axis)
    rows: list[tuple[Node, str, str]] = []

    if cfg.include_x:
        for p in range(1, max(1, cfg.max_x_power) + 1):
            _append_unique_term(rows, _pow_if(xj, p), "baseline", "x_power")
    if cfg.include_u:
        for q in range(1, max(1, cfg.max_u_power) + 1):
            _append_unique_term(rows, _pow_if(u, q), "baseline", "u_power")
    if cfg.include_du and order != 1:
        _append_unique_term(rows, du, "baseline", "derivative")
    if cfg.include_d2u and order != 2:
        _append_unique_term(rows, d2u, "baseline", "derivative")
    if cfg.include_xu and cfg.include_x and cfg.include_u:
        deg_cap = cfg.max_xu_total_degree
        for p in range(1, max(1, cfg.max_x_power) + 1):
            for q in range(1, max(1, cfg.max_u_power) + 1):
                if deg_cap > 0 and p + q > deg_cap:
                    continue
                _append_unique_term(rows, Mul(_pow_if(xj, p), _pow_if(u, q)), "baseline", "x_u_cross")
    if cfg.include_xdu and cfg.include_x and order != 1:
        for p in range(1, max(1, cfg.max_x_power) + 1):
            _append_unique_term(rows, Mul(_pow_if(xj, p), du), "baseline", "x_du_cross")
    if cfg.include_inv_xdu and order != 1:
        _append_unique_term(rows, Mul(Pow(xj, -1), du), "baseline", "radial_derivative")
    if cfg.include_inv_xu:
        _append_unique_term(rows, Mul(Pow(xj, -1), u), "baseline", "radial_value")
    if cfg.include_inv_x2u:
        _append_unique_term(rows, Mul(Pow(xj, -2), u), "baseline", "radial_value")
    if cfg.include_udu and order != 1:
        _append_unique_term(rows, Mul(u, du), "baseline", "velocity_state")
    return rows


def _gs_de_library_rows(cfg: DESearchConfig, *, order: int) -> list[tuple[Node, str, str]]:
    rows: list[tuple[Node, str, str]] = []
    if bool(getattr(cfg, "gs_enable", False)) and (
        bool(getattr(cfg, "gs_unit_torus", False))
        or bool(getattr(cfg, "gs_de_all_upgrades", False))
        or bool(getattr(cfg, "gs_de_contact_templates", False))
        or bool(getattr(cfg, "gs_de_noether_templates", False))
        or bool(getattr(cfg, "gs_de_discrete_symmetry_templates", False))
        or bool(getattr(cfg, "gs_de_weighted_scaling_templates", False))
        or bool(getattr(cfg, "gs_de_radial_reduction_templates", False))
        or bool(getattr(cfg, "gs_de_invariant_library", False))
        or getattr(cfg, "gs_de_compiled_nonlinear_invariants", None) is not None
        or getattr(cfg, "gs_de_compiled_orbit_coordinate", None) is not None
        or bool(getattr(cfg, "gs_de_reduction_rows", None))
    ):
        try:
            from nestynet_sr.sr_gs.de_bridge import generalized_symmetry_de_term_rows

            for t, source, family in generalized_symmetry_de_term_rows(cfg, order=order):
                _append_unique_term(rows, t, str(source), str(family))
        except Exception as exc:
            print(f"[GS-DE] Failed to build generalized-symmetry templates: {type(exc).__name__}: {exc}")
    return rows


def _structural_prior_de_library_rows(cfg: DESearchConfig, *, order: int) -> list[tuple[Node, str, str]]:
    rows: list[tuple[Node, str, str]] = []
    if bool(getattr(cfg, "de_hard_tail_templates", False)):
        try:
            from nestynet_sr.sr_gs.de_bridge import hard_tail_de_term_rows

            for t, source, family in hard_tail_de_term_rows(cfg, order=order):
                _append_unique_term(rows, t, str(source), str(family))
        except Exception as exc:
            print(f"[DE-Prior] Failed to build hard-tail templates: {type(exc).__name__}: {exc}")
    return rows


def build_de_library_terms_with_sources(cfg: DESearchConfig, *, order: int) -> tuple[List[Node], List[str]]:
    """Return library terms and source labels under the selected GS policy."""
    policy = _canonical_gs_policy(getattr(cfg, "gs_policy", "augment"))
    baseline = _baseline_de_library_rows(cfg, order=order)
    prior_rows = _structural_prior_de_library_rows(cfg, order=order)
    gs_rows = _gs_de_library_rows(cfg, order=order)

    if policy == "gs-only-affine" and bool(getattr(cfg, "gs_enable", False)):
        rows = list(gs_rows)
    elif policy == "replace-shadowed" and gs_rows:
        shadow_families = {"radial_derivative", "radial_value"}
        if bool(getattr(cfg, "gs_de_velocity_templates", False)):
            shadow_families.add("velocity_state")
        rows = [r for r in baseline if r[2] not in shadow_families]
        for term, source, family in prior_rows:
            _append_unique_term(rows, term, source, family)
        for term, source, family in gs_rows:
            _append_unique_term(rows, term, source, family)
        try:
            from nestynet_sr.sr_gs.reporting import record_policy_event

            record_policy_event(
                policy=policy,
                action="de_replace_shadowed_terms",
                details={
                    "order": int(order),
                    "removed_baseline_families": sorted(shadow_families),
                    "baseline_terms_before": len(baseline),
                    "gs_terms": len(gs_rows),
                    "terms_after": len(rows),
                },
            )
        except Exception:
            pass
    else:
        rows = list(baseline)
        for term, source, family in prior_rows:
            _append_unique_term(rows, term, source, family)
        for term, source, family in gs_rows:
            _append_unique_term(rows, term, source, family)

    rows = _maybe_simplify_de_rows(rows, cfg, order=order)
    rows = _maybe_expr_ir_de_rows(rows, cfg, order=order)
    return [r[0] for r in rows], [r[1] for r in rows]


def build_de_library_terms(cfg: DESearchConfig, *, order: int) -> List[Node]:
    """Return a list of term ASTs φ_k (no coefficients included)."""
    return build_de_library_terms_with_sources(cfg, order=order)[0]


# ──────────────────────────────────────────────────────────────
# Main search
# ──────────────────────────────────────────────────────────────


def _gather_x(dataloader, *, max_batches: int, max_points: int, device=None) -> torch.Tensor:
    xs = []
    n = 0
    for bi, batch in enumerate(dataloader):
        if bi >= max_batches or n >= max_points:
            break
        x = _flatten_x(batch)
        if device is not None:
            x = x.to(device)
        xs.append(x)
        n += x.shape[0]
    if not xs:
        raise ValueError("No batches found in dataloader")
    X = torch.cat(xs, dim=0)
    if X.shape[0] > max_points:
        X = X[:max_points]
    return X


def _de_lie_prolongation_enabled(cfg: DESearchConfig) -> bool:
    return bool(getattr(cfg, "gs_enable", False)) and bool(
        getattr(cfg, "gs_de_lie_prolongation", False)
    )


def _prolongation_metric(meta: Optional[Dict]) -> Optional[float]:
    if not isinstance(meta, dict):
        return None
    try:
        metric = float(meta.get("best_metric"))
    except Exception:
        return None
    if not math.isfinite(metric):
        return None
    return metric


def _prolongation_selection_enabled(cfg: DESearchConfig) -> bool:
    return bool(getattr(cfg, "gs_de_lie_use_for_selection", False))


def _prolongation_penalty(cfg: DESearchConfig, metric: Optional[float]) -> float:
    if not _prolongation_selection_enabled(cfg):
        return 0.0
    weight = float(getattr(cfg, "gs_de_lie_prolongation_weight", 0.05))
    if weight <= 0.0:
        return 0.0
    if metric is None:
        return weight * 10.0
    return weight * min(max(float(metric), 0.0), 10.0)


class _DeterminingCertificateReport(dict):
    """JSON-safe report carrying non-serialized automatic-search artifacts."""

    compiled_invariants: Any = None
    selected_symmetry_result: Any = None


def _certify_de_determining_candidate(
    *,
    order: int,
    X: torch.Tensor,
    cache: UFeatureCache,
    term_asts: Sequence[Optional[Node]],
    coeffs: torch.Tensor,
    cfg: DESearchConfig,
) -> Optional[Dict]:
    """Point-symmetry determining certificate for a selected DE candidate.

    The certificate is intrinsic to the candidate equation (the anchor is
    eliminated with the candidate's own right-hand side); the surrogate data
    only set the (x, u) sampling ranges.
    """

    gs_enabled = bool(getattr(cfg, "gs_enable", False))
    auto_nonlinear = gs_enabled and bool(
        getattr(cfg, "gs_de_auto_nonlinear", True)
    )
    if not (
        gs_enabled
        and (
            bool(getattr(cfg, "gs_de_determining_certificate", False))
            or auto_nonlinear
        )
    ):
        return None
    try:
        from nestynet_sr.sr_gs.de_certificates import certify_scalar_ode_candidate

        max_samples = int(getattr(cfg, "gs_de_certificate_max_samples", 1024))
        n = int(X.shape[0])
        Xs = X
        if n > max_samples:
            idx = torch.linspace(0, n - 1, steps=max_samples, device=X.device).round().long()
            Xs = X.index_select(0, torch.unique_consecutive(idx))
        need_hess = int(order) >= 2
        cache.reset()
        cache.ensure(Xs, need_grad=True, need_hess=need_hess)
        if cache.u is None:
            raise RuntimeError("UFeatureCache did not populate u values")
        x_axis = int(cfg.x_axis)
        x_np = Xs[:, x_axis].detach().cpu().numpy()
        u_np = cache.u[:, 0].detach().cpu().numpy()
        u1_np = None
        if int(order) >= 2:
            if cache.g is None:
                raise RuntimeError("UFeatureCache did not populate first derivatives")
            u1_np = cache.g[:, 0, x_axis].detach().cpu().numpy()
        tol = float(getattr(cfg, "gs_de_certificate_tol", 1.0e-6))
        common = dict(
            x=x_np,
            u=u_np,
            u1=u1_np,
            coeffs=[float(v) for v in coeffs.detach().cpu().reshape(-1)],
            term_asts=list(term_asts),
            order=int(order),
            x_axis=x_axis,
            coeff_prune_tol=float(
                getattr(cfg, "gs_de_certificate_coeff_prune_tol", 0.0)
            ),
            on_shell_tol=tol,
            off_shell_tol=tol,
            max_samples=max_samples,
            multiplier_max_degree=int(
                getattr(cfg, "gs_de_determining_multiplier_degree", 2)
            ),
            bootstrap=(
                max(1, int(getattr(cfg, "gs_de_determining_bootstraps", 8)))
                if auto_nonlinear
                else int(getattr(cfg, "gs_de_determining_bootstraps", 8))
            ),
            max_generators=int(getattr(cfg, "gs_de_determining_max_generators", 4)),
            sparse_rotation=bool(
                getattr(cfg, "gs_de_determining_sparse_rotation", True)
            ),
            bracket_certificate=(
                True
                if auto_nonlinear
                else bool(
                    getattr(cfg, "gs_de_determining_bracket_certificate", True)
                )
            ),
        )

        escalation_report: Dict[str, Any] | None = None
        if auto_nonlinear:
            affine = certify_scalar_ode_candidate(
                **common,
                generator_max_degree=1,
                use_coupled_polynomial_solver=True,
            )
            max_degree = int(getattr(cfg, "gs_de_determining_max_degree", 2))
            quadratic = None
            if max_degree >= 2:
                quadratic = certify_scalar_ode_candidate(
                    **common,
                    generator_max_degree=2,
                    use_coupled_polynomial_solver=True,
                )
            affine_nullity = int(getattr(affine, "certified_nullity", 0))
            quadratic_nullity = int(
                getattr(quadratic, "certified_nullity", 0)
                if quadratic is not None
                else 0
            )
            affine_ok = bool(getattr(affine, "promotable_generators", False))
            quadratic_ok = (
                quadratic is not None
                and bool(getattr(quadratic, "promotable_generators", False))
            )
            if quadratic_ok and (not affine_ok or quadratic_nullity > affine_nullity):
                result = quadratic
                reason = (
                    "affine_rejected"
                    if not affine_ok
                    else "quadratic_added_certified_directions"
                )
            else:
                result = affine
                reason = (
                    "degree_bound_one"
                    if quadratic is None
                    else "quadratic_added_no_certified_directions"
                )
            escalation_report = {
                "enabled": True,
                "policy": "affine_first_bounded_quadratic",
                "affine_status": str(getattr(affine, "status", "unknown")),
                "affine_certified_nullity": affine_nullity,
                "quadratic_status": (
                    None
                    if quadratic is None
                    else str(getattr(quadratic, "status", "unknown"))
                ),
                "quadratic_certified_nullity": quadratic_nullity,
                "selected_degree": int(getattr(result.config, "generator_degree", 1)),
                "reason": reason,
            }
        else:
            coupled = bool(getattr(cfg, "gs_de_determining_equations", False))
            degree = int(getattr(cfg, "gs_de_determining_max_degree", 2)) if coupled else 1
            result = certify_scalar_ode_candidate(
                **common,
                generator_max_degree=degree,
                use_coupled_polynomial_solver=coupled,
            )

        report = _DeterminingCertificateReport(result.to_report())
        report.selected_symmetry_result = result
        if escalation_report is not None:
            report["automatic_escalation"] = escalation_report
        if bool(getattr(cfg, "gs_de_nonlinear_invariants", False)) or auto_nonlinear:
            try:
                from nestynet_sr.sr_gs.de_invariant_compiler import (
                    InvariantCompilerConfig,
                    compile_subalgebra_invariants,
                )

                generators = []
                if bool(getattr(result, "promotable_generators", False)):
                    generators = [
                        row.generator
                        for row in list(getattr(result, "candidates", ()) or ())
                        if bool(getattr(row, "accepted", True))
                    ]
                points = torch.as_tensor(
                    np.column_stack((x_np, u_np)), dtype=torch.float64
                )
                cut = max(4, min(int(points.shape[0]) - 4, int(0.75 * points.shape[0])))
                inv_cfg = InvariantCompilerConfig(
                    max_polynomial_degree=int(
                        getattr(cfg, "gs_de_nonlinear_invariant_max_degree", 3)
                    ),
                    max_candidates=int(getattr(cfg, "gs_de_nonlinear_invariant_max_candidates", 8)),
                    max_invariants=min(
                        4,
                        int(getattr(cfg, "gs_de_nonlinear_invariant_max_candidates", 8)),
                    ),
                    action_rtol=float(getattr(cfg, "gs_de_nonlinear_invariant_tol", 0.03)),
                    orbit_rtol=float(getattr(cfg, "gs_de_nonlinear_invariant_tol", 0.03)),
                )
                compiled = compile_subalgebra_invariants(
                    generators,
                    points[:cut],
                    points[cut:],
                    config=inv_cfg,
                    include_full_algebra=bool(
                        getattr(result, "promotable_full_algebra", False)
                    ),
                    include_orbit_coordinates=bool(
                        getattr(cfg, "gs_de_nonlinear_orbit_coordinate", True)
                    ),
                )
                report.compiled_invariants = compiled
                report["nonlinear_carriers"] = compiled.to_report()
            except Exception as carrier_exc:
                report["nonlinear_carriers"] = {
                    "status": "failed",
                    "reason": str(carrier_exc)[:300],
                }
        report["enabled"] = True
        return report
    except Exception as exc:
        return {"enabled": True, "status": "failed", "reason": str(exc)[:300]}


def _score_de_lie_prolongation_candidate(
    *,
    order: int,
    X: torch.Tensor,
    cache: UFeatureCache,
    term_asts: Sequence[Optional[Node]],
    coeffs: torch.Tensor,
    cfg: DESearchConfig,
) -> Tuple[Optional[Dict], float]:
    if not _de_lie_prolongation_enabled(cfg):
        return None, 0.0
    try:
        from nestynet_sr.sr_gs.prolongation import score_de_lie_prolongation

        meta = score_de_lie_prolongation(
            order=int(order),
            x_axis=int(cfg.x_axis),
            X=X,
            cache=cache,
            term_asts=term_asts,
            coeffs=coeffs,
            cfg=cfg,
        )
    except Exception as exc:
        meta = {
            "enabled": True,
            "status": "failed",
            "reason": str(exc)[:300],
            "order": int(order),
            "best_metric": None,
        }
    metric = _prolongation_metric(meta)
    penalty = _prolongation_penalty(cfg, metric)
    if isinstance(meta, dict):
        meta["score_penalty"] = float(penalty)
        meta["used_for_selection"] = bool(_prolongation_selection_enabled(cfg))
        if metric is None and _prolongation_selection_enabled(cfg):
            meta["selection_penalty_reason"] = "missing_or_nonfinite_metric"
    return meta, penalty


def _score_de_lie_prolongation_multi(
    *,
    order: int,
    Xs: Sequence[torch.Tensor],
    caches: Sequence[UFeatureCache],
    term_asts: Sequence[Optional[Node]],
    coeffs: torch.Tensor,
    cfg: DESearchConfig,
    dataset_ids: Optional[Sequence[str]] = None,
) -> Tuple[Optional[Dict], float]:
    if not _de_lie_prolongation_enabled(cfg):
        return None, 0.0
    try:
        from nestynet_sr.sr_gs.prolongation import score_de_lie_prolongation
    except Exception as exc:
        meta = {
            "enabled": True,
            "status": "failed",
            "reason": str(exc)[:300],
            "order": int(order),
            "best_metric": None,
            "datasets": [],
        }
        penalty = _prolongation_penalty(cfg, None)
        meta["score_penalty"] = float(penalty)
        meta["used_for_selection"] = bool(_prolongation_selection_enabled(cfg))
        if _prolongation_selection_enabled(cfg):
            meta["selection_penalty_reason"] = "scorer_import_failed"
        return meta, penalty

    dataset_metas: List[Dict] = []
    generator_metrics: Dict[str, List[float]] = {}
    generator_rows: Dict[str, Dict] = {}
    failed_required_datasets = 0
    for i, (X, cache) in enumerate(zip(Xs, caches)):
        try:
            c_i = coeffs[i] if getattr(coeffs, "ndim", 0) == 2 else coeffs
            meta_i = score_de_lie_prolongation(
                order=int(order),
                x_axis=int(cfg.x_axis),
                X=X,
                cache=cache,
                term_asts=term_asts,
                coeffs=c_i,
                cfg=cfg,
            )
        except Exception as exc:
            meta_i = {
                "enabled": True,
                "status": "failed",
                "reason": str(exc)[:300],
                "order": int(order),
                "best_metric": None,
            }
        meta_i["dataset_index"] = int(i)
        if dataset_ids is not None and i < len(dataset_ids):
            meta_i["dataset_id"] = str(dataset_ids[i])
        metric_i = _prolongation_metric(meta_i)
        if metric_i is None:
            failed_required_datasets += 1
        for row in list(meta_i.get("generators", []) or []):
            if not isinstance(row, dict) or not bool(row.get("metric_eligible", False)):
                continue
            name = str(row.get("name", "") or "")
            if not name:
                continue
            try:
                m = float(row.get("on_shell_metric"))
            except Exception:
                continue
            if not math.isfinite(m):
                continue
            generator_metrics.setdefault(name, []).append(m)
            generator_rows.setdefault(name, row)
        dataset_metas.append(meta_i)

    required = len(dataset_metas)
    aggregate_rows: List[Dict] = []
    for name, values in sorted(generator_metrics.items()):
        complete = len(values) == required
        row = {
            "name": name,
            "dataset_count": int(len(values)),
            "required_dataset_count": int(required),
            "eligible_all_datasets": bool(complete),
            "metric_mean": float(sum(values) / len(values)) if values else None,
            "metric_min": float(min(values)) if values else None,
            "metric_max": float(max(values)) if values else None,
            "representative": generator_rows.get(name, {}),
        }
        aggregate_rows.append(row)
    eligible_aggregate_rows = [row for row in aggregate_rows if bool(row["eligible_all_datasets"])]
    eligible_aggregate_rows.sort(
        key=lambda row: (
            float("inf") if row["metric_mean"] is None else float(row["metric_mean"]),
            str(row["name"]),
        )
    )
    best_row = eligible_aggregate_rows[0] if eligible_aggregate_rows else None
    aggregate_metric = None if best_row is None else float(best_row["metric_mean"])
    tol_v = float(getattr(cfg, "gs_de_lie_prolongation_tol", 0.05))
    accepted_names = [
        str(row["name"])
        for row in eligible_aggregate_rows
        if row["metric_mean"] is not None and float(row["metric_mean"]) <= tol_v
    ]
    penalty = _prolongation_penalty(cfg, aggregate_metric)
    meta = {
        "enabled": True,
        "status": "scored_multi" if aggregate_metric is not None else "skipped_multi",
        "order": int(order),
        "num_datasets": int(len(dataset_metas)),
        "num_scored_datasets": int(required - failed_required_datasets),
        "num_failed_required_datasets": int(failed_required_datasets),
        "best_metric": aggregate_metric,
        "best_generator": best_row,
        "aggregate": {
            "best_metric_mean": aggregate_metric,
            "best_metric_min": None if best_row is None else best_row["metric_min"],
            "best_metric_max": None if best_row is None else best_row["metric_max"],
            "common_generator_required": True,
            "generator_metrics": aggregate_rows,
        },
        "accepted_generator_names": sorted(accepted_names),
        "datasets": dataset_metas,
        "score_penalty": float(penalty),
        "used_for_selection": bool(_prolongation_selection_enabled(cfg)),
    }
    if aggregate_metric is None and _prolongation_selection_enabled(cfg):
        meta["selection_penalty_reason"] = (
            "missing_or_nonfinite_metric_or_no_common_generator"
        )
    return meta, penalty


def _compute_condition_number(matrix: torch.Tensor) -> Optional[float]:
    """Best-effort condition number with an SVD fallback for older Torch paths."""
    if matrix.ndim != 2 or int(matrix.shape[1]) == 0:
        return None
    try:
        return float(torch.linalg.cond(matrix).item())
    except Exception:
        try:
            _u, s, _vh = torch.linalg.svd(matrix, full_matrices=False)
        except Exception:
            return None
        if int(s.numel()) == 0:
            return None
        sigma_max = float(s[0].item())
        sigma_min = float(s[-1].item())
        if sigma_min < 1e-15:
            return float("inf")
        return sigma_max / sigma_min


def _rms_residual(residual: torch.Tensor) -> float:
    return float((residual.square().mean().sqrt()).detach().cpu())


def _scale_normalized_refit_weights(
    Phi: torch.Tensor,
    y: torch.Tensor,
    *,
    dynamic_range_min: float = 1.0e4,
    min_rows: int = 32,
) -> torch.Tensor | None:
    """Return row weights for a scale-normalized final coefficient refit.

    Implicit DE residuals can span many orders of magnitude, especially near
    singular coordinates. In that regime an absolute residual fit may be
    dominated by a few boundary rows. This guarded weight only activates when
    the selected support has a large row-scale dynamic range.
    """
    if Phi.ndim != 2 or y.ndim != 1 or int(Phi.shape[0]) != int(y.shape[0]):
        return None
    if int(Phi.shape[0]) < int(min_rows) or int(Phi.shape[1]) <= 0:
        return None
    row_scale = (y.square() + Phi.square().sum(dim=1)).sqrt()
    finite = torch.isfinite(row_scale) & (row_scale > 0)
    vals = row_scale[finite]
    if int(vals.numel()) < int(min_rows):
        return None
    try:
        q_lo = torch.quantile(vals, 0.10)
        q_hi = vals.max()
    except Exception:
        vals_sorted = torch.sort(vals).values
        lo_idx = max(0, min(int(vals_sorted.numel()) - 1, int(0.10 * (int(vals_sorted.numel()) - 1))))
        q_lo = vals_sorted[lo_idx]
        q_hi = vals_sorted[-1]
    if not bool(torch.isfinite(q_lo).item()) or not bool(torch.isfinite(q_hi).item()):
        return None
    if float(q_lo.detach().cpu()) <= 0.0:
        return None
    dyn = float((q_hi / q_lo.clamp_min(torch.finfo(vals.dtype).eps)).detach().cpu())
    if dyn < float(dynamic_range_min):
        return None
    floor = q_lo.clamp_min(torch.finfo(vals.dtype).eps)
    weights = 1.0 / row_scale.clamp_min(floor)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    if int((weights > 0).sum().item()) < int(min_rows):
        return None
    return weights


def _maybe_scale_normalized_refit_matrix(
    Phi: torch.Tensor,
    y: torch.Tensor,
    coeffs: torch.Tensor | None = None,
    *,
    dynamic_range_min: float = 1.0e4,
    guarded_dynamic_range_min: float = 1.0e2,
) -> tuple[torch.Tensor, bool]:
    """Optionally refit coefficients after normalizing rows by residual scale."""
    if coeffs is None:
        coeffs = ridge_lstsq(Phi, y, ridge=0.0)
    weights = _scale_normalized_refit_weights(
        Phi,
        y,
        dynamic_range_min=float(dynamic_range_min),
    )
    if weights is None:
        weights = _scale_normalized_refit_weights(
            Phi,
            y,
            dynamic_range_min=float(guarded_dynamic_range_min),
        )
        if weights is None:
            return coeffs, False
        guarded_accept = True
    else:
        guarded_accept = False
    try:
        w = weights.reshape(-1, 1)
        coeffs_refit = ridge_lstsq(Phi * w, y * weights, ridge=0.0)
    except Exception:
        return coeffs, False
    if not bool(torch.isfinite(coeffs_refit).all().item()):
        return coeffs, False
    if guarded_accept and not _accept_guarded_scale_normalized_refit(Phi, y, coeffs, coeffs_refit):
        return coeffs, False
    return coeffs_refit, True


def _accept_guarded_scale_normalized_refit(
    Phi: torch.Tensor,
    y: torch.Tensor,
    coeffs: torch.Tensor,
    coeffs_refit: torch.Tensor,
    *,
    trim_quantile: float = 0.995,
    min_rows: int = 32,
    min_bulk_improvement_frac: float = 0.25,
    max_full_rms_worsen: float = 2.0,
) -> bool:
    """Accept a moderate-range normalized refit only if the non-extreme bulk improves.

    Radial/singular DEs can have one or two boundary rows with much larger
    residual scale than the rest of the trajectory. A lower dynamic-range
    trigger is useful there, but too aggressive for ordinary data. This guard
    requires the normalized refit to improve the bulk residual while not
    substantially worsening the full residual.
    """
    if Phi.ndim != 2 or y.ndim != 1 or int(Phi.shape[0]) != int(y.shape[0]):
        return False
    if int(Phi.shape[0]) < int(min_rows) or int(Phi.shape[1]) <= 0:
        return False

    row_scale = (y.square() + Phi.square().sum(dim=1)).sqrt()
    finite = torch.isfinite(row_scale) & torch.isfinite(y) & torch.isfinite(Phi).all(dim=1)
    vals = row_scale[finite]
    if int(vals.numel()) < int(min_rows):
        return False
    try:
        cutoff = torch.quantile(vals, float(trim_quantile))
    except Exception:
        vals_sorted = torch.sort(vals).values
        idx = max(0, min(int(vals_sorted.numel()) - 1, int(float(trim_quantile) * (int(vals_sorted.numel()) - 1))))
        cutoff = vals_sorted[idx]
    if not bool(torch.isfinite(cutoff).item()):
        return False

    bulk = finite & (row_scale <= cutoff)
    if int(bulk.sum().item()) < int(min_rows):
        return False

    r0 = Phi @ coeffs - y
    r1 = Phi @ coeffs_refit - y
    if not bool(torch.isfinite(r0[finite]).all().item()) or not bool(torch.isfinite(r1[finite]).all().item()):
        return False

    bulk0 = r0[bulk].square().mean().sqrt()
    bulk1 = r1[bulk].square().mean().sqrt()
    full0 = r0[finite].square().mean().sqrt()
    full1 = r1[finite].square().mean().sqrt()
    if not all(bool(torch.isfinite(v).item()) for v in (bulk0, bulk1, full0, full1)):
        return False

    eps = torch.finfo(Phi.dtype).eps
    improved_bulk = bulk1 <= bulk0 * (1.0 - float(min_bulk_improvement_frac))
    bounded_full = full1 <= full0 * float(max_full_rms_worsen) + eps
    return bool(improved_bulk.item()) and bool(bounded_full.item())


def _maybe_scale_normalized_refit_multi(
    Phis: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
    keep: torch.Tensor,
    coeffs: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Apply the guarded scale-normalized refit independently per dataset."""
    if int(keep.sum()) <= 0 or len(Phis) != len(ys):
        return coeffs, False
    coeffs_out = coeffs.clone()
    used_any = False
    for i, (Phi, y) in enumerate(zip(Phis, ys)):
        c0 = coeffs_out[i]
        c_refit, used = _maybe_scale_normalized_refit_matrix(Phi[:, keep], y, c0)
        if used:
            coeffs_out[i] = c_refit
            used_any = True
    return coeffs_out, used_any


def _mean_rms_values(values: Sequence[float]) -> float:
    if not values:
        return float("inf")
    vals = []
    for v in values:
        f = float(v)
        if torch.isfinite(torch.tensor(f)):
            vals.append(f)
    if not vals:
        return float("inf")
    return float(sum(vals) / max(1, len(vals)))


def _within_support_prune_tolerance(metric: float, best_metric: float) -> bool:
    if not torch.isfinite(torch.tensor(float(metric))):
        return False
    if not torch.isfinite(torch.tensor(float(best_metric))):
        return True
    tol = _SUPPORT_PRUNE_ABS_TOL + _SUPPORT_PRUNE_REL_TOL * max(float(best_metric), 0.0)
    return float(metric) <= float(best_metric) + float(tol)


def _mask_from_indices_like(keep: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros_like(keep, dtype=torch.bool)
    if indices:
        idx = torch.tensor([int(i) for i in indices], device=keep.device, dtype=torch.long)
        mask[idx] = True
    return mask


def _fit_single_support(
    Phi: torch.Tensor,
    y: torch.Tensor,
    keep: torch.Tensor,
    *,
    Phi_val: torch.Tensor | None,
    y_val: torch.Tensor | None,
) -> tuple[torch.Tensor, float, float | None]:
    Phi_sel = Phi[:, keep]
    coeffs = ridge_lstsq(Phi_sel, y, ridge=0.0)
    rms_train = _rms_residual(Phi_sel @ coeffs - y)
    rms_val = None
    if Phi_val is not None and y_val is not None:
        rms_val = _rms_residual(Phi_val[:, keep] @ coeffs - y_val)
    return coeffs, rms_train, rms_val


def _validation_prune_single_support(
    Phi: torch.Tensor,
    y: torch.Tensor,
    keep: torch.Tensor,
    *,
    Phi_val: torch.Tensor | None,
    y_val: torch.Tensor | None,
    sparsity_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float | None]:
    keep_best = keep.clone()
    coeffs_best, rms_train_best, rms_val_best = _fit_single_support(
        Phi,
        y,
        keep_best,
        Phi_val=Phi_val,
        y_val=y_val,
    )
    if Phi_val is None or y_val is None:
        return keep_best, coeffs_best, rms_train_best, rms_val_best
    metric_best = _mean_rms_values([float(rms_val_best)])
    active_idx = [int(i) for i in torch.where(keep_best)[0].tolist()]
    if 1 < len(active_idx) <= _EXHAUSTIVE_SUPPORT_PRUNE_MAX_TERMS:
        candidates = [
            (
                keep_best,
                coeffs_best,
                rms_train_best,
                rms_val_best,
                int(keep_best.sum()),
                metric_best,
            )
        ]
        for size in range(1, len(active_idx) + 1):
            for subset in combinations(active_idx, size):
                trial = _mask_from_indices_like(keep_best, subset)
                try:
                    coeffs_trial, rms_train_trial, rms_val_trial = _fit_single_support(
                        Phi,
                        y,
                        trial,
                        Phi_val=Phi_val,
                        y_val=y_val,
                    )
                except Exception:
                    continue
                if rms_val_trial is None:
                    continue
                candidates.append(
                    (
                        trial,
                        coeffs_trial,
                        rms_train_trial,
                        rms_val_trial,
                        int(trial.sum()),
                        _mean_rms_values([float(rms_val_trial)]),
                    )
                )
        best_metric = min(metric for *_rest, metric in candidates)
        eligible = [
            cand for cand in candidates if _within_support_prune_tolerance(cand[-1], best_metric)
        ]
        eligible.sort(key=lambda cand: (int(cand[-2]), float(cand[-1])))
        keep_best, coeffs_best, rms_train_best, rms_val_best, _n_terms, _metric = eligible[0]
        return keep_best, coeffs_best, rms_train_best, rms_val_best

    improved = True
    while improved and int(keep_best.sum()) > 1:
        improved = False
        trial_best = None
        trial_metric_best = metric_best
        for idx in torch.where(keep_best)[0].tolist():
            trial = keep_best.clone()
            trial[int(idx)] = False
            if int(trial.sum()) <= 0:
                continue
            coeffs_trial, rms_train_trial, rms_val_trial = _fit_single_support(
                Phi,
                y,
                trial,
                Phi_val=Phi_val,
                y_val=y_val,
            )
            if rms_val_trial is None:
                continue
            metric_trial = _mean_rms_values([float(rms_val_trial)])
            if _within_support_prune_tolerance(metric_trial, metric_best) and (
                trial_best is None or metric_trial < trial_metric_best
            ):
                trial_best = (trial, coeffs_trial, rms_train_trial, rms_val_trial)
                trial_metric_best = metric_trial
        if trial_best is not None:
            keep_best, coeffs_best, rms_train_best, rms_val_best = trial_best
            metric_best = trial_metric_best
            improved = True
    return keep_best, coeffs_best, rms_train_best, rms_val_best


def _fit_multi_support(
    Phis: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
    keep: torch.Tensor,
    *,
    Phi_vals: Sequence[torch.Tensor] | None,
    y_vals: Sequence[torch.Tensor] | None,
) -> tuple[torch.Tensor, list[float], list[float] | None]:
    coeffs = torch.zeros((len(Phis), int(keep.sum())), device=Phis[0].device, dtype=Phis[0].dtype)
    rms_train = []
    rms_val = [] if Phi_vals is not None and y_vals is not None else None
    for i in range(len(Phis)):
        coeffs[i] = ridge_lstsq(Phis[i][:, keep], ys[i], ridge=0.0)
        rms_train.append(_rms_residual(Phis[i][:, keep] @ coeffs[i] - ys[i]))
        if rms_val is not None and Phi_vals is not None and y_vals is not None:
            rms_val.append(_rms_residual(Phi_vals[i][:, keep] @ coeffs[i] - y_vals[i]))
    return coeffs, rms_train, rms_val


def _validation_prune_multi_support(
    Phis: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
    keep: torch.Tensor,
    *,
    Phi_vals: Sequence[torch.Tensor] | None,
    y_vals: Sequence[torch.Tensor] | None,
    sparsity_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float] | None]:
    keep_best = keep.clone()
    coeffs_best, rms_train_best, rms_val_best = _fit_multi_support(
        Phis,
        ys,
        keep_best,
        Phi_vals=Phi_vals,
        y_vals=y_vals,
    )
    if Phi_vals is None or y_vals is None or rms_val_best is None:
        return keep_best, coeffs_best, rms_train_best, rms_val_best
    metric_best = _mean_rms_values(rms_val_best)
    active_idx = [int(i) for i in torch.where(keep_best)[0].tolist()]
    if 1 < len(active_idx) <= _EXHAUSTIVE_SUPPORT_PRUNE_MAX_TERMS:
        candidates = [
            (
                keep_best,
                coeffs_best,
                rms_train_best,
                rms_val_best,
                int(keep_best.sum()),
                metric_best,
            )
        ]
        for size in range(1, len(active_idx) + 1):
            for subset in combinations(active_idx, size):
                trial = _mask_from_indices_like(keep_best, subset)
                try:
                    coeffs_trial, rms_train_trial, rms_val_trial = _fit_multi_support(
                        Phis,
                        ys,
                        trial,
                        Phi_vals=Phi_vals,
                        y_vals=y_vals,
                    )
                except Exception:
                    continue
                if rms_val_trial is None:
                    continue
                candidates.append(
                    (
                        trial,
                        coeffs_trial,
                        rms_train_trial,
                        rms_val_trial,
                        int(trial.sum()),
                        _mean_rms_values(rms_val_trial),
                    )
                )
        best_metric = min(metric for *_rest, metric in candidates)
        eligible = [
            cand for cand in candidates if _within_support_prune_tolerance(cand[-1], best_metric)
        ]
        eligible.sort(key=lambda cand: (int(cand[-2]), float(cand[-1])))
        keep_best, coeffs_best, rms_train_best, rms_val_best, _n_terms, _metric = eligible[0]
        return keep_best, coeffs_best, rms_train_best, rms_val_best

    improved = True
    while improved and int(keep_best.sum()) > 1:
        improved = False
        trial_best = None
        trial_metric_best = metric_best
        for idx in torch.where(keep_best)[0].tolist():
            trial = keep_best.clone()
            trial[int(idx)] = False
            if int(trial.sum()) <= 0:
                continue
            coeffs_trial, rms_train_trial, rms_val_trial = _fit_multi_support(
                Phis,
                ys,
                trial,
                Phi_vals=Phi_vals,
                y_vals=y_vals,
            )
            if rms_val_trial is None:
                continue
            metric_trial = _mean_rms_values(rms_val_trial)
            if _within_support_prune_tolerance(metric_trial, metric_best) and (
                trial_best is None or metric_trial < trial_metric_best
            ):
                trial_best = (trial, coeffs_trial, rms_train_trial, rms_val_trial)
                trial_metric_best = metric_trial
        if trial_best is not None:
            keep_best, coeffs_best, rms_train_best, rms_val_best = trial_best
            metric_best = trial_metric_best
            improved = True
    return keep_best, coeffs_best, rms_train_best, rms_val_best


def discover_de_from_surrogate(
    surrogate,
    train_dataloader,
    val_dataloader=None,
    *,
    cfg: Optional[DESearchConfig] = None,
    device=None,
    dataset=None,
) -> DESearchResult:
    """Discover a sparse DE residual from a frozen surrogate u(x).

    Parameters
    ----------
    surrogate : nn.Module
        Trained neural network surrogate u(x)
    train_dataloader : DataLoader
        Training data loader
    val_dataloader : DataLoader, optional
        Validation data loader
    cfg : DESearchConfig, optional
        DE discovery configuration. If None, uses defaults.
    device : torch.device, optional
        Device for computation
    dataset : PhysDataset, optional
        Dataset object with coordinate metadata. If provided and cfg.x_axis is not
        explicitly set, will auto-detect x_axis from dataset.get_time_coords().

    Returns
    -------
    DESearchResult
        Discovered DE with coefficients and terms
    """
    if cfg is None:
        cfg = DESearchConfig()

    # Auto-detect x_axis from dataset coordinate metadata if available
    if dataset is not None:
        if hasattr(dataset, "has_coord_metadata") and dataset.has_coord_metadata():
            time_coords = dataset.get_time_coords()
            if time_coords and len(time_coords) > 0:
                # Use the first time coordinate if x_axis not explicitly set
                # We check if x_axis is still at default value (0)
                detected_x_axis = time_coords[0]
                if cfg.x_axis == 0 or cfg.x_axis is None:
                    print(
                        f"Auto-detected x_axis={detected_x_axis} from dataset coordinate metadata"
                    )
                    print(f"  Coordinate names: {dataset.get_coord_names()}")
                    print(f"  Time coordinates: {time_coords}")
                    cfg = replace(cfg, x_axis=detected_x_axis)
                elif cfg.x_axis != detected_x_axis:
                    print(
                        f"Warning: x_axis={cfg.x_axis} explicitly set, but dataset metadata suggests {detected_x_axis}"
                    )

    dev = device
    if dev is None:
        try:
            dev = next(surrogate.parameters()).device
        except Exception:
            dev = torch.device("cpu")

    Xtr = _gather_x(
        train_dataloader, max_batches=cfg.max_batches, max_points=cfg.max_points, device=dev
    )
    Xva = None
    if val_dataloader is not None:
        Xva = _gather_x(
            val_dataloader, max_batches=cfg.max_batches, max_points=cfg.max_points, device=dev
        )

    cache = UFeatureCache(surrogate)

    best: Optional[DESearchResult] = None

    for order in cfg.order_candidates:
        if order not in (1, 2):
            continue

        # Anchor is moved to the LHS: anchor + Σ c_k φ_k = 0.
        if order == 1:
            _anchor_ast = DU(cfg.x_axis)
            cache.ensure(Xtr, need_grad=True, need_hess=False)
            anchor = cache.g[:, 0, cfg.x_axis]
        else:
            _anchor_ast = D2U(cfg.x_axis, cfg.x_axis)
            cache.ensure(Xtr, need_grad=False, need_hess=True)
            anchor = cache.H[:, 0, cfg.x_axis, cfg.x_axis]

        # Build library terms.
        terms, term_sources_full = build_de_library_terms_with_sources(cfg, order=order)

        # Design matrix Phi.
        cols = []
        term_asts: List[Node] = []
        term_sources: List[str] = []
        if cfg.include_const:
            ok_const, why_const = term_units_feasible_under_gs_policy(
                None,
                order=order,
                x_axis=cfg.x_axis,
                cfg=cfg,
            )
            if ok_const:
                cols.append(torch.ones(Xtr.shape[0], device=dev, dtype=Xtr.dtype))
                term_asts.append(None)  # sentinel for constant
                term_sources.append("baseline")
            elif cfg.enforce_units:
                print(f"[Units] Dropping constant term 1: {why_const}")

        for t, src in zip(terms, term_sources_full):
            ok_t, why_t = term_units_feasible_under_gs_policy(
                t,
                order=order,
                x_axis=cfg.x_axis,
                cfg=cfg,
            )
            if not ok_t:
                if cfg.enforce_units:
                    try:
                        t_str = repr(t)
                    except Exception:
                        t_str = str(type(t).__name__)
                    print(f"[Units] Dropping library term {t_str}: {why_t}")
                continue
            v = _as_N(_eval_ast(t, Xtr, cache))
            cols.append(v)
            term_asts.append(t)
            term_sources.append(str(src))

        if not cols:
            continue

        Phi = torch.stack(cols, dim=1)  # (N,K)
        y = -anchor  # (N,)

        # Drop non-finite rows (numerical noise near domain boundaries)
        finite_mask = torch.isfinite(y) & torch.isfinite(Phi).all(dim=1)
        if finite_mask.sum() < 10:
            continue  # too few finite points for this order
        Phi, y = Phi[finite_mask], y[finite_mask]

        _c, keep = stlsq(Phi, y, ridge=cfg.ridge, lam=cfg.stlsq_lambda, max_iter=cfg.stlsq_max_iter)

        # Validation matrix over the full library, used both for scoring and
        # for backward pruning of over-complete STLSQ supports.
        Phi_va = None
        y_va = None
        if Xva is not None:
            cache.reset()
            if order == 1:
                cache.ensure(Xva, need_grad=True, need_hess=False)
                anchor_va = cache.g[:, 0, cfg.x_axis]
            else:
                cache.ensure(Xva, need_grad=False, need_hess=True)
                anchor_va = cache.H[:, 0, cfg.x_axis, cfg.x_axis]

            cols_va = []
            for t in term_asts:
                if t is None:
                    cols_va.append(torch.ones(Xva.shape[0], device=dev, dtype=Xva.dtype))
                else:
                    cols_va.append(_as_N(_eval_ast(t, Xva, cache)))
            Phi_va = torch.stack(cols_va, dim=1)
            y_va = -anchor_va
            finite_mask_va = torch.isfinite(y_va) & torch.isfinite(Phi_va).all(dim=1)
            if int(finite_mask_va.sum()) < 10:
                continue
            Phi_va = Phi_va[finite_mask_va]
            y_va = y_va[finite_mask_va]

        keep, c_sel, rms_tr, rms_va = _validation_prune_single_support(
            Phi,
            y,
            keep,
            Phi_val=Phi_va,
            y_val=y_va,
            sparsity_penalty=float(cfg.sparsity_penalty),
        )
        term_sel = [t for t, k in zip(term_asts, keep.tolist()) if k]
        source_sel = [s for s, k in zip(term_sources, keep.tolist()) if k]
        Phi_sel = Phi[:, keep]
        c_refit, used_scale_refit = _maybe_scale_normalized_refit_matrix(Phi_sel, y, c_sel)
        if used_scale_refit:
            c_sel = c_refit
            rms_tr = _rms_residual(Phi_sel @ c_sel - y)
            if Phi_va is not None and y_va is not None:
                rms_va = _rms_residual(Phi_va[:, keep] @ c_sel - y_va)

        # Check condition number for degeneracy detection
        if len(c_sel) > 0:
            cond_num = _compute_condition_number(Phi_sel)
            # Always print condition number
            cond_str = "N/A" if cond_num is None else f"{float(cond_num):.2e}"
            print(f"  Condition number: {cond_str}")

            # Warn if very high
            if cond_num is not None and cond_num > 1e8:
                print("  ⚠ WARNING: High condition number detected!")
                print("  → Multiple equivalent ODE forms likely exist")
                print("  → Consider using --varpro_templates for better disambiguation")
        else:
            cond_num = None

        prolongation_metadata, prolongation_penalty = _score_de_lie_prolongation_candidate(
            order=int(order),
            X=Xtr,
            cache=cache,
            term_asts=term_sel,
            coeffs=c_sel,
            cfg=cfg,
        )
        determining_certificate = _certify_de_determining_candidate(
            order=int(order),
            X=Xtr,
            cache=cache,
            term_asts=term_sel,
            coeffs=c_sel,
            cfg=cfg,
        )

        # Model score: val RMS + sparsity penalty.
        score = (
            (rms_va if rms_va is not None else rms_tr)
            + cfg.sparsity_penalty * len(term_sel)
            + float(prolongation_penalty)
        )
        if best is None:
            best = DESearchResult(
                order=order,
                x_axis=cfg.x_axis,
                term_asts=term_sel,
                coeffs=c_sel.detach().cpu(),
                rms_train=rms_tr,
                rms_val=rms_va,
                condition_number=cond_num,
                term_sources=source_sel,
                prolongation_metadata=prolongation_metadata,
                determining_certificate=determining_certificate,
            )
            best_score = score
        else:
            if score < best_score:
                best = DESearchResult(
                    order=order,
                    x_axis=cfg.x_axis,
                    term_asts=term_sel,
                    coeffs=c_sel.detach().cpu(),
                    rms_train=rms_tr,
                    rms_val=rms_va,
                    condition_number=cond_num,
                    term_sources=source_sel,
                    prolongation_metadata=prolongation_metadata,
                    determining_certificate=determining_certificate,
                )
                best_score = score

    if best is None:
        raise RuntimeError("DE discovery failed to produce any candidate")

    best.residual_ast = build_de_residual_ast(
        best,
        units_spec=cfg.units_spec,
        enforce_units=cfg.enforce_units,
        simplify_options=_ast_simplify_options_from_cfg(cfg, context="de_residual")
        if bool(getattr(cfg, "ast_simplify", False))
        else None,
    )
    try:
        reports_by_order = getattr(cfg, "_expr_ir_de_report_by_order", None)
        if isinstance(reports_by_order, dict):
            setattr(best, "expr_ir_reports_by_order", dict(reports_by_order))
        report = getattr(cfg, "_expr_ir_last_de_report", None)
        setattr(best, "expr_ir_report", report)
    except Exception:
        pass
    return best


def discover_de_from_surrogates(
    surrogates: Sequence[Any],
    train_dataloaders: Sequence[Any],
    val_dataloaders: Optional[Sequence[Any]] = None,
    *,
    cfg: Optional[DESearchConfig] = None,
    device=None,
    datasets: Optional[Sequence[Any]] = None,
    dataset_ids: Optional[Sequence[str]] = None,
) -> DESearchResultMulti:
    """Multi-dataset variant of discover_de_from_surrogate.

    Discovers an implicit DE with a shared term library support across datasets,
    but allows dataset-specific coefficients.
    """
    if cfg is None:
        cfg = DESearchConfig()
    if len(surrogates) != len(train_dataloaders):
        raise ValueError("surrogates and train_dataloaders must have same length")
    D = len(surrogates)
    if D == 0:
        raise ValueError("Need at least one dataset")
    if val_dataloaders is not None and len(val_dataloaders) != D:
        raise ValueError("val_dataloaders must match number of surrogates")

    # Auto-detect x_axis from the first dataset if possible
    if datasets is not None and len(datasets) > 0 and (cfg.x_axis == 0 or cfg.x_axis is None):
        ds0 = datasets[0]
        try:
            if hasattr(ds0, "has_coord_metadata") and ds0.has_coord_metadata():
                time_coords = ds0.get_time_coords()
                if time_coords and len(time_coords) > 0:
                    detected_x_axis = int(time_coords[0])
                    print(
                        f"Auto-detected x_axis={detected_x_axis} from dataset coordinate metadata (dataset 0)"
                    )
                    cfg = replace(cfg, x_axis=detected_x_axis)
        except Exception:
            pass

    dev = device
    if dev is None:
        try:
            dev = next(surrogates[0].parameters()).device
        except Exception:
            dev = torch.device("cpu")

    Xtrs = [
        _gather_x(dl, max_batches=cfg.max_batches, max_points=cfg.max_points, device=dev)
        for dl in train_dataloaders
    ]
    Xvas = None
    if val_dataloaders is not None:
        Xvas = [
            _gather_x(dl, max_batches=cfg.max_batches, max_points=cfg.max_points, device=dev)
            for dl in val_dataloaders
        ]

    caches = [UFeatureCache(s) for s in surrogates]

    best: Optional[DESearchResultMulti] = None
    best_score = float("inf")

    for order in cfg.order_candidates:
        if order not in (1, 2):
            continue

        # Anchor and library
        terms, term_sources_full = build_de_library_terms_with_sources(cfg, order=order)
        term_asts: List[Optional[Node]] = []
        term_sources: List[str] = []
        if cfg.include_const:
            ok_const, why_const = term_units_feasible_under_gs_policy(
                None,
                order=order,
                x_axis=cfg.x_axis,
                cfg=cfg,
            )
            if ok_const:
                term_asts.append(None)
                term_sources.append("baseline")
            elif cfg.enforce_units:
                print(f"[Units] Dropping constant term 1: {why_const}")
        for t, src in zip(terms, term_sources_full):
            ok_t, why_t = term_units_feasible_under_gs_policy(
                t,
                order=order,
                x_axis=cfg.x_axis,
                cfg=cfg,
            )
            if ok_t:
                term_asts.append(t)
                term_sources.append(str(src))
                continue
            if cfg.enforce_units:
                try:
                    t_str = repr(t)
                except Exception:
                    t_str = str(type(t).__name__)
                print(f"[Units] Dropping library term {t_str}: {why_t}")
        if len(term_asts) == 0:
            continue

        Phis = []
        ys = []
        for i in range(D):
            Xtr = Xtrs[i]
            cache = caches[i]
            cache.reset()
            if order == 1:
                cache.ensure(Xtr, need_grad=True, need_hess=False)
                anchor = cache.g[:, 0, cfg.x_axis]
            else:
                cache.ensure(Xtr, need_grad=False, need_hess=True)
                anchor = cache.H[:, 0, cfg.x_axis, cfg.x_axis]

            cols = []
            for t in term_asts:
                if t is None:
                    cols.append(torch.ones(Xtr.shape[0], device=dev, dtype=Xtr.dtype))
                else:
                    cols.append(_as_N(_eval_ast(t, Xtr, cache)))
            Phi = torch.stack(cols, dim=1)
            y = -anchor

            # Drop non-finite rows (domain issues / gradients)
            m = torch.isfinite(y)
            m &= torch.isfinite(Phi).all(dim=1)
            if int(m.sum()) < 10:
                raise RuntimeError(f"Too few finite rows for dataset {i} (order={order})")
            Phis.append(Phi[m])
            ys.append(y[m])

        _C, keep = group_stlsq(
            Phis, ys, ridge=cfg.ridge, lam=cfg.stlsq_lambda, max_iter=cfg.stlsq_max_iter
        )

        if int(keep.sum()) == 0:
            continue

        Phi_vas = None
        y_vas = None
        if Xvas is not None:
            Phi_vas = []
            y_vas = []
            for i in range(D):
                Xva = Xvas[i]
                cache = caches[i]
                cache.reset()
                if order == 1:
                    cache.ensure(Xva, need_grad=True, need_hess=False)
                    anchor_va = cache.g[:, 0, cfg.x_axis]
                else:
                    cache.ensure(Xva, need_grad=False, need_hess=True)
                    anchor_va = cache.H[:, 0, cfg.x_axis, cfg.x_axis]
                cols_va = []
                for t in term_asts:
                    if t is None:
                        cols_va.append(torch.ones(Xva.shape[0], device=dev, dtype=Xva.dtype))
                    else:
                        cols_va.append(_as_N(_eval_ast(t, Xva, cache)))
                Phi_va = torch.stack(cols_va, dim=1)
                y_va = -anchor_va
                finite_mask_va = torch.isfinite(y_va) & torch.isfinite(Phi_va).all(dim=1)
                if int(finite_mask_va.sum()) < 10:
                    raise RuntimeError(f"Too few finite validation rows for dataset {i} (order={order})")
                Phi_vas.append(Phi_va[finite_mask_va])
                y_vas.append(y_va[finite_mask_va])

        keep, Csel, rms_tr, rms_va = _validation_prune_multi_support(
            Phis,
            ys,
            keep,
            Phi_vals=Phi_vas,
            y_vals=y_vas,
            sparsity_penalty=float(cfg.sparsity_penalty),
        )
        if int(keep.sum()) == 0:
            continue
        term_sel = [t for t, k in zip(term_asts, keep.tolist()) if k]
        source_sel = [s for s, k in zip(term_sources, keep.tolist()) if k]
        C_refit, used_scale_refit = _maybe_scale_normalized_refit_multi(Phis, ys, keep, Csel)
        if used_scale_refit:
            Csel = C_refit
            rms_tr = [
                _rms_residual(Phis[i][:, keep] @ Csel[i] - ys[i])
                for i in range(D)
            ]
            if Phi_vas is not None and y_vas is not None:
                rms_va = [
                    _rms_residual(Phi_vas[i][:, keep] @ Csel[i] - y_vas[i])
                    for i in range(D)
                ]

        prolongation_metadata, prolongation_penalty = _score_de_lie_prolongation_multi(
            order=int(order),
            Xs=Xtrs,
            caches=caches,
            term_asts=term_sel,
            coeffs=Csel,
            cfg=cfg,
            dataset_ids=dataset_ids,
        )
        # The certificate is intrinsic to the candidate; dataset 0 only sets
        # the (x, u) sampling ranges.  Per-dataset coefficients can differ, so
        # certify with the first dataset's coefficient row.
        determining_certificate = _certify_de_determining_candidate(
            order=int(order),
            X=Xtrs[0],
            cache=caches[0],
            term_asts=term_sel,
            coeffs=Csel[0] if getattr(Csel, "ndim", 0) == 2 else Csel,
            cfg=cfg,
        )

        # Model score: mean(val RMS) + sparsity penalty
        ref = rms_va if rms_va is not None else rms_tr
        score = (
            float(sum(ref) / max(1, len(ref)))
            + cfg.sparsity_penalty * len(term_sel)
            + float(prolongation_penalty)
        )
        if score < best_score or best is None:
            best_score = score
            best = DESearchResultMulti(
                order=int(order),
                x_axis=int(cfg.x_axis),
                term_asts=term_sel,
                coeffs=Csel.detach().cpu(),
                rms_train=rms_tr,
                rms_val=rms_va,
                dataset_ids=list(dataset_ids) if dataset_ids is not None else None,
                term_sources=source_sel,
                prolongation_metadata=prolongation_metadata,
                determining_certificate=determining_certificate,
            )

    if best is None:
        raise RuntimeError("Multi-dataset DE discovery failed to produce any candidate")

    # Optional residual ASTs (per dataset)
    try:
        best.residual_asts = [
            build_de_residual_ast(
                DESearchResult(
                    order=best.order,
                    x_axis=best.x_axis,
                    term_asts=best.term_asts,
                    coeffs=best.coeffs[d],
                    rms_train=best.rms_train[d],
                ),
                units_spec=cfg.units_spec,
                enforce_units=cfg.enforce_units,
                simplify_options=_ast_simplify_options_from_cfg(cfg, context="de_residual")
                if bool(getattr(cfg, "ast_simplify", False))
                else None,
            )
            for d in range(best.coeffs.shape[0])
        ]
    except Exception:
        best.residual_asts = None
    try:
        reports_by_order = getattr(cfg, "_expr_ir_de_report_by_order", None)
        if isinstance(reports_by_order, dict):
            setattr(best, "expr_ir_reports_by_order", dict(reports_by_order))
        report = getattr(cfg, "_expr_ir_last_de_report", None)
        setattr(best, "expr_ir_report", report)
    except Exception:
        pass
    return best


def build_de_residual_ast(
    result: DESearchResult,
    *,
    coeff_prefix: str = "c",
    units_spec: Any = None,
    enforce_units: bool = False,
    simplify_options: SimplifyOptions | None = None,
) -> Node:
    """Build an AST representing the discovered residual: anchor + Σ c_k term_k."""

    def _coeff_atom(name: str, init: float, term: Optional[Node]) -> AtomNode:
        required_dim = None
        if units_spec is not None:
            try:
                required_dim = required_coeff_dim_for_term(
                    term,
                    order=int(result.order),
                    x_axis=int(result.x_axis),
                    units_spec=units_spec,
                )
            except Exception:
                if bool(enforce_units):
                    raise
                required_dim = None

        # Route through the shared scalar-constant framework.
        return make_unit_aware_scalar_atom(
            required_dim=required_dim,
            units_spec=units_spec,
            base_tag=str(name),
            init=float(init),
            strict=bool(enforce_units),
        )

    x_axis = int(result.x_axis)
    if result.order == 1:
        root: Node = DU(x_axis)
    elif result.order == 2:
        root = D2U(x_axis, x_axis)
    else:
        raise ValueError(f"Unsupported DE order: {result.order}")

    for i, (term, c) in enumerate(zip(result.term_asts, result.coeffs.tolist())):
        name = f"{coeff_prefix}{i}"
        coeff_atom = _coeff_atom(name, float(c), term)
        if term is None:
            # Constant offset term.
            root = Add(root, coeff_atom)
        else:
            root = Add(root, Mul(coeff_atom, term))
    if simplify_options is not None and bool(getattr(simplify_options, "enabled", False)):
        root, _stats = simplify_ast(root, simplify_options, units_spec=units_spec)
    return root


def make_u_feature_atom_factory(surrogate, *, cache: Optional[UFeatureCache] = None):
    """Return an atom_factory suitable for build_composite_from_ast.

    The returned factory builds parameter-free leaves for AtomNode kinds:
      - 'u'   : u(x)
      - 'du'  : ∂u/∂x_axis
      - 'd2u' : ∂²u/∂x_axis0∂x_axis1

    A single shared cache is used so all feature leaves reuse the same surrogate
    evaluations per batch.
    """
    shared_cache = cache if cache is not None else UFeatureCache(surrogate)

    def factory(atom: AtomNode, existing=None):
        kind = str(getattr(atom, "kind", "")).lower()
        # Optional output component selection for vector-valued surrogates.
        kw = getattr(atom, "kwargs", {}) or {}
        out_idx = int(kw.get("out_idx", kw.get("out", kw.get("component", 0))))
        if kind in ("u", "field", "state"):
            if (
                isinstance(existing, UFeatureLeaf)
                and existing.kind in ("u", "field", "state")
                and int(getattr(existing, "out_idx", 0)) == out_idx
            ):
                return existing
            return UFeatureLeaf(shared_cache, "u", out_idx=out_idx)

        if kind in ("du", "d1u", "grad_u"):
            axis = int(kw.get("axis", 0))
            if (
                isinstance(existing, UFeatureLeaf)
                and existing.kind in ("du", "d1u", "grad_u")
                and int(getattr(existing, "out_idx", 0)) == out_idx
                and existing.axis == axis
            ):
                return existing
            return UFeatureLeaf(shared_cache, "du", out_idx=out_idx, axis=axis)

        if kind in ("d2u", "ddu", "hess_u"):
            a0 = int(kw.get("axis0", 0))
            a1 = int(kw.get("axis1", 0))
            if (
                isinstance(existing, UFeatureLeaf)
                and existing.kind in ("d2u", "ddu", "hess_u")
                and int(getattr(existing, "out_idx", 0)) == out_idx
                and existing.axis0 == a0
                and existing.axis1 == a1
            ):
                return existing
            return UFeatureLeaf(shared_cache, "d2u", out_idx=out_idx, axis0=a0, axis1=a1)

        return None

    return factory
