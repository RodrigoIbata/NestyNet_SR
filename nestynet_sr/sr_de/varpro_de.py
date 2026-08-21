# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Variable Projection for DE Discovery.

This module provides functions for refining DE coefficients discovered by STLSQ
using Variable Projection (VarPro) with Levenberg-Marquardt optimization.

Phase 1: Linear coefficient refinement (this file)
Phase 2: Template system with nonlinear parameters (future extension)
"""

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from nestynet_sr.adaptors.de_varpro_adaptor import DEVarProAdaptor
from nestynet_sr.adaptors.u_feature_leaf import UFeatureCache
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    CosNode,
    ExpNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
)
from nestynet_sr.sr_de.de_search import (
    DESearchConfig,
    DESearchResult,
    _as_N,
    _eval_ast,
    ridge_lstsq,
    stlsq,
    build_de_residual_ast,
    term_units_feasible,
)
from nestynet_sr.sr_de.de_templates import TEMPLATE_REGISTRY, TemplateInstance, get_template


def _classify_term_type(node: Node) -> str:
    """Classify DE term as 'state', 'x', or 'mixed'.

    Term types:
    - 'state': depends only on u, u_x, u_xx (autonomous)
    - 'x': depends only on x (forcing)
    - 'mixed': depends on both x and state variables
    - 'const': constant term (1)

    Parameters
    ----------
    node : Node
        AST node to classify

    Returns
    -------
    str
        One of: 'const', 'state', 'x', 'mixed'
    """
    if node is None:
        return "const"

    # Track what this term depends on
    has_x = False
    has_state = False

    def traverse(n):
        nonlocal has_x, has_state

        if isinstance(n, AtomNode):
            kind = str(getattr(n, "kind", "")).lower()
            if kind in ("var", "x", "input"):
                has_x = True
            elif kind in ("u", "field", "state", "du", "d1u", "grad_u", "d2u", "ddu", "hess_u"):
                has_state = True
            elif kind in ("const", "constant"):
                # Constants don't affect classification
                pass
            # scalar params should not appear (we substitute them first)
            return

        # Recursively traverse composite nodes
        if isinstance(n, (AddNode, MulNode)):
            traverse(n.left)
            traverse(n.right)
        elif isinstance(n, PowNode):
            traverse(n.base)
            traverse(n.exponent)
        elif isinstance(n, (ExpNode, SinNode, CosNode)):
            traverse(n.arg)

    traverse(node)

    # Classify based on dependencies
    if has_x and has_state:
        return "mixed"
    elif has_x:
        return "x"
    elif has_state:
        return "state"
    else:
        return "const"


def _gather_all_x(dataloader, device) -> torch.Tensor:
    """Concatenate all x batches from dataloader to a single tensor.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader with (x, y) batches
    device : torch.device
        Device to move tensor to

    Returns
    -------
    X : torch.Tensor
        Concatenated x tensor (N_total, Nx)
    """
    x_batches = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            x = batch[0]
        else:
            x = batch
        x_batches.append(x.to(device))
    return torch.cat(x_batches, dim=0)


def _compute_dataset_weights(X_list: list[torch.Tensor]) -> torch.Tensor:
    """Compute size-based weights for datasets.

    Weights are normalized to sum to 1.0 and proportional to dataset sizes
    (number of data points). This ensures larger datasets get more weight
    in multi-dataset optimization.

    Parameters
    ----------
    X_list : list of torch.Tensor
        List of input tensors, one per dataset. Each has shape (N_d, Nx).

    Returns
    -------
    weights : torch.Tensor
        Dataset weights (D,), normalized to sum to 1.0

    Example
    -------
    >>> X_list = [torch.randn(1000, 2), torch.randn(2000, 2), torch.randn(500, 2)]
    >>> weights = _compute_dataset_weights(X_list)
    >>> print(weights)  # tensor([0.2857, 0.5714, 0.1429])  (proportional to [1000, 2000, 500])
    """
    sizes = torch.tensor([X.shape[0] for X in X_list], dtype=torch.float32)
    weights = sizes / sizes.sum()
    return weights


def _group_stlsq_multi(
    X_list: list[torch.Tensor],
    y_list: list[torch.Tensor],
    term_asts: list[Node],
    surrogates: list[nn.Module],
    *,
    order: int,
    x_axis: int,
    ridge: float,
    lam: float,
    max_iter: int,
    device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Group-sparse STLSQ for multi-dataset VarPro template search.

    Builds design matrices per dataset from template ASTs and applies
    group-sparse thresholding to enforce shared support.

    Parameters
    ----------
    X_list : list of torch.Tensor
        Input data per dataset, each (N_d, Nx)
    y_list : list of torch.Tensor
        Target data per dataset, each (N_d,) - typically anchor residuals
    term_asts : list of Node
        Template term ASTs to evaluate (baseline + template)
    surrogates : list of nn.Module
        Frozen surrogates for u(x), one per dataset
    order : int
        DE order
    x_axis : int
        Derivative axis
    ridge : float
        Ridge regularization
    lam : float
        Sparsity threshold for row-wise L2 norms
    max_iter : int
        Maximum STLSQ iterations
    device : torch.device
        Device for computation

    Returns
    -------
    coeffs_list : list of torch.Tensor
        Coefficients per dataset, each (K_selected,)
    keep_mask : torch.Tensor
        Shared support mask (K,) bool
    """
    from nestynet_sr.sr_de.de_search import group_stlsq

    D = len(X_list)

    # Build design matrices per dataset
    Phi_list = []
    for d in range(D):
        cache = UFeatureCache(surrogates[d])

        # Evaluate all terms
        cols = []
        for term_ast in term_asts:
            if term_ast is None:
                # Constant term
                cols.append(torch.ones(X_list[d].shape[0], device=device))
            else:
                col_val = _eval_ast(term_ast, X_list[d], cache)
                cols.append(_as_N(col_val))

        Phi = torch.stack(cols, dim=1)  # (N_d, K)
        Phi_list.append(Phi)

    # Run group-sparse STLSQ
    C, keep_mask = group_stlsq(Phi_list, y_list, ridge=ridge, lam=lam, max_iter=max_iter)

    # Extract selected coefficients per dataset
    coeffs_list = [C[d, keep_mask] for d in range(D)]

    return coeffs_list, keep_mask


def _support_minimization_multi(
    X_train_list: list[torch.Tensor],
    y_train_list: list[torch.Tensor],
    X_val_list: Optional[list[torch.Tensor]],
    y_val_list: Optional[list[torch.Tensor]],
    term_asts: list[Node],
    surrogates: list[nn.Module],
    *,
    order: int,
    x_axis: int,
    ridge: float,
    rms_baseline_list: list[float],
    rms_tol_factor: float,
    device,
) -> tuple[list[Node], list[torch.Tensor], list[float], Optional[list[float]]]:
    """Greedy term removal with shared support constraint (multi-dataset).

    Iteratively removes the least important term while maintaining
    RMS < rms_baseline * rms_tol_factor on *all* datasets.

    Parameters
    ----------
    X_train_list : list of torch.Tensor
        Training inputs per dataset
    y_train_list : list of torch.Tensor
        Training targets per dataset
    X_val_list : list of torch.Tensor, optional
        Validation inputs per dataset
    y_val_list : list of torch.Tensor, optional
        Validation targets per dataset
    term_asts : list of Node
        Current term library
    surrogates : list of nn.Module
        Frozen surrogates
    order : int
        DE order
    x_axis : int
        Derivative axis
    ridge : float
        Ridge regularization
    rms_baseline_list : list of float
        Baseline RMS per dataset (threshold for stopping)
    rms_tol_factor : float
        Tolerance factor (stop if any RMS > baseline * factor)
    device : torch.device
        Device

    Returns
    -------
    selected_terms : list of Node
        Minimized term library
    coeffs_list : list of torch.Tensor
        Coefficients per dataset
    rms_train_list : list of float
        Training RMS per dataset
    rms_val_list : list of float or None
        Validation RMS per dataset (if validation data provided)
    """

    D = len(X_train_list)
    current_terms = term_asts.copy()

    # Helper to compute RMS for given terms
    def compute_multi_rms(terms):
        Phi_train_list = []
        Phi_val_list = [] if X_val_list is not None else None

        for d in range(D):
            cache_train = UFeatureCache(surrogates[d])
            cols_train = []
            for term in terms:
                if term is None:
                    cols_train.append(torch.ones(X_train_list[d].shape[0], device=device))
                else:
                    col_val = _eval_ast(term, X_train_list[d], cache_train)
                    cols_train.append(_as_N(col_val))
            Phi_train_list.append(torch.stack(cols_train, dim=1))

            if Phi_val_list is not None:
                cache_val = UFeatureCache(surrogates[d])
                cols_val = []
                for term in terms:
                    if term is None:
                        cols_val.append(torch.ones(X_val_list[d].shape[0], device=device))
                    else:
                        col_val = _eval_ast(term, X_val_list[d], cache_val)
                        cols_val.append(_as_N(col_val))
                Phi_val_list.append(torch.stack(cols_val, dim=1))

        # Fit coefficients (no sparsity, just ridge)
        C_train = torch.stack(
            [ridge_lstsq(Phi_train_list[d], y_train_list[d], ridge) for d in range(D)], dim=0
        )

        # Compute RMS
        rms_train = [
            float(
                ((Phi_train_list[d] @ C_train[d] - y_train_list[d]).square().mean().sqrt()).item()
            )
            for d in range(D)
        ]

        rms_val = None
        if Phi_val_list is not None:
            rms_val = [
                float(
                    ((Phi_val_list[d] @ C_train[d] - y_val_list[d]).square().mean().sqrt()).item()
                )
                for d in range(D)
            ]

        coeffs_list = [C_train[d] for d in range(D)]
        return coeffs_list, rms_train, rms_val

    # Greedy removal
    while len(current_terms) > 1:
        # Try removing each term
        best_idx = -1
        best_rms_train = None
        _best_rms_val = None
        _best_coeffs = None

        for i in range(len(current_terms)):
            trial_terms = current_terms[:i] + current_terms[i + 1 :]
            coeffs, rms_train, rms_val = compute_multi_rms(trial_terms)

            # Check if *all* datasets stay below threshold
            all_ok = all(rms_train[d] <= rms_baseline_list[d] * rms_tol_factor for d in range(D))

            if all_ok:
                # Accept: store this as best candidate
                # Prefer removing terms with smallest impact on RMS
                max_rms = max(rms_train)
                if best_idx == -1 or max_rms < max(best_rms_train):
                    best_idx = i
                    best_rms_train = rms_train
                    _best_rms_val = rms_val
                    _best_coeffs = coeffs

        if best_idx == -1:
            # No term can be removed without exceeding threshold
            break

        # Remove the best term
        current_terms = current_terms[:best_idx] + current_terms[best_idx + 1 :]

    # Final fit with selected terms
    final_coeffs, final_rms_train, final_rms_val = compute_multi_rms(current_terms)

    return current_terms, final_coeffs, final_rms_train, final_rms_val


def _check_condition_number(
    X: torch.Tensor, term_asts: list[Node], *, threshold: float = 1e8, verbose: bool = False
) -> tuple[float, bool]:
    """Compute condition number and check for degeneracy.

    Large condition numbers indicate ill-conditioning and potential
    representational degeneracy (multiple equivalent equation forms).

    Parameters
    ----------
    X : torch.Tensor
        Feature matrix (N, K)
    term_asts : list[Node]
        Term ASTs for context in warning messages
    threshold : float
        Condition number threshold for warning (default: 1e8)
    verbose : bool
        Print warning if condition number is large

    Returns
    -------
    cond_num : float
        Condition number of X
    is_degenerate : bool
        True if condition number exceeds threshold

    Notes
    -----
    A large condition number (κ > 1e8) suggests:
    - Multiple terms are nearly linearly dependent
    - Small data perturbations → large coefficient changes
    - Representational degeneracy (e.g., exp(kx) ≈ linear combo of u, x)

    Example: For u_x = exp(2x), we have exp(2x) ≈ 2u - 1 exactly,
    creating perfect linear dependence in the feature matrix.
    """
    # Compute condition number via SVD
    # cond(X) = σ_max / σ_min
    try:
        cond_num = float(torch.linalg.cond(X).item())
    except Exception:
        # Fallback: compute manually via SVD
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        sigma_max = S[0].item()
        sigma_min = S[-1].item()
        if sigma_min < 1e-15:
            cond_num = float("inf")
        else:
            cond_num = sigma_max / sigma_min

    is_degenerate = cond_num > threshold

    if verbose and is_degenerate:
        print("\n  ⚠ WARNING: High condition number detected!")
        print(f"  Condition number: {cond_num:.2e} (threshold: {threshold:.2e})")
        print("  → Feature matrix is ill-conditioned")
        print("  → Multiple equivalent equation forms likely exist")
        print("  → Linear dependence between terms:")
        print("     - e.g., exp(kx) ≈ linear combination of u, x")
        print("  → Consider reporting multiple valid forms")
        print("  → True VarPro (LM over ψ) may help resolve ambiguity\n")

    return cond_num, is_degenerate


def _support_minimization(
    term_asts: list[Node],
    coeffs: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    *,
    ridge: float = 0.0,
    rms_tol_factor: float = 1.1,
    verbose: bool = False,
) -> tuple[list[Node], torch.Tensor]:
    """Greedily remove terms to find minimal support.

    After fitting, try removing each term one at a time. If removing a term
    doesn't significantly hurt the validation RMS (increase < rms_tol_factor),
    permanently remove it. Repeat until no more terms can be removed.

    Parameters
    ----------
    term_asts : list[Node]
        Current list of term ASTs (including None for constant)
    coeffs : torch.Tensor
        Current coefficients (K,)
    X_train : torch.Tensor
        Training feature matrix (N, K)
    y_train : torch.Tensor
        Training target vector (N,)
    X_val : torch.Tensor, optional
        Validation feature matrix (M, K)
    y_val : torch.Tensor, optional
        Validation target vector (M,)
    ridge : float
        Ridge regularization for refitting
    rms_tol_factor : float
        Maximum allowed RMS increase factor (e.g., 1.1 = 10% worse)
    verbose : bool
        Print removal progress

    Returns
    -------
    pruned_terms : list[Node]
        Minimal term list
    pruned_coeffs : torch.Tensor
        Refitted coefficients for minimal support

    Notes
    -----
    This is a greedy algorithm that may not find the global optimum, but
    it's fast and often finds near-optimal sparse solutions.

    The first term (anchor) is never removed.
    """

    # Compute baseline validation RMS
    if X_val is not None and y_val is not None:
        residuals_val = X_val @ coeffs - y_val
        rms_baseline = torch.sqrt((residuals_val**2).mean())
        use_val = True
    else:
        # Fallback to training RMS
        residuals_train = X_train @ coeffs - y_train
        rms_baseline = torch.sqrt((residuals_train**2).mean())
        use_val = False

    rms_threshold = rms_baseline * rms_tol_factor

    if verbose:
        val_str = "val" if use_val else "train"
        print(f"  Support minimization: baseline RMS ({val_str}) = {rms_baseline:.6e}")
        print(f"  Threshold: {rms_threshold:.6e} ({rms_tol_factor:.2f}x baseline)")

    # Current support
    keep_mask = torch.ones(len(term_asts), dtype=torch.bool, device=X_train.device)
    current_terms = term_asts.copy()
    current_coeffs = coeffs.clone()

    # Note: the DE "anchor" derivative (u_x or u_xx) is not part of term_asts.
    # We are therefore free to remove any term here; we only enforce that at least one remains.
    n_removed_total = 0

    # Greedy removal loop
    while True:
        improved = False
        best_idx_to_remove = None
        best_rms = rms_threshold

        # Try removing each term
        for idx in range(0, len(current_terms)):
            if not keep_mask[idx]:
                continue  # Already removed

            # Create trial mask with this term removed
            trial_mask = keep_mask.clone()
            trial_mask[idx] = False

            if trial_mask.sum() == 0:
                continue  # Can't remove all terms

            # Refit on reduced support
            X_train_reduced = X_train[:, trial_mask]
            trial_coeffs_reduced = ridge_lstsq(X_train_reduced, y_train, ridge)

            # Expand back to full size (with zeros for removed terms)
            trial_coeffs_full = torch.zeros_like(coeffs)
            trial_coeffs_full[trial_mask] = trial_coeffs_reduced

            # Compute validation RMS
            if use_val:
                X_val_reduced = X_val[:, trial_mask]
                residuals = X_val_reduced @ trial_coeffs_reduced - y_val
            else:
                residuals = X_train_reduced @ trial_coeffs_reduced - y_train

            rms_trial = torch.sqrt((residuals**2).mean())

            # Check if this removal is acceptable and best so far
            if rms_trial < best_rms:
                best_rms = rms_trial
                best_idx_to_remove = idx
                improved = True

        # If we found a term to remove, do it
        if improved and best_idx_to_remove is not None:
            keep_mask[best_idx_to_remove] = False
            n_removed_total += 1

            # Refit final coefficients with this term removed
            X_train_reduced = X_train[:, keep_mask]
            current_coeffs_reduced = ridge_lstsq(X_train_reduced, y_train, ridge)

            # Expand back to full size
            current_coeffs = torch.zeros_like(coeffs)
            current_coeffs[keep_mask] = current_coeffs_reduced

            if verbose:
                try:
                    term_repr = (
                        repr(current_terms[best_idx_to_remove])
                        if current_terms[best_idx_to_remove]
                        else "1"
                    )
                except Exception:
                    term_repr = str(type(current_terms[best_idx_to_remove]).__name__)
                print(f"    Removed term {best_idx_to_remove}: {term_repr} (RMS: {best_rms:.6e})")
        else:
            # No more terms can be removed
            break

    # Build final pruned lists
    pruned_terms = [t for t, keep in zip(current_terms, keep_mask.tolist()) if keep]
    pruned_coeffs = current_coeffs[keep_mask]

    if verbose:
        print(f"  → Removed {n_removed_total} terms, {len(pruned_terms)} remain")

    return pruned_terms, pruned_coeffs


def _substitute_free_consts(node: Node, params: dict[str, float]) -> Node:
    """Substitute scalar-parameter atoms with constant values.

    Handles both legacy ``free_const`` and newer ``scale`` leaves.

    Parameters
    ----------
    node : Node
        AST node potentially containing scalar-parameter atoms
    params : dict[str, float]
        Parameter values to substitute (key = tag or name)

    Returns
    -------
    Node
        AST with scalar-parameter atoms replaced by constant AtomNodes
    """
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("free_const", "freeconst", "free_constant", "scale"):
            # Extract parameter name/tag
            tag = getattr(node, "tag", None)
            name = getattr(node, "kwargs", {}).get("name", None)

            # Look up value
            if tag and tag in params:
                val = params[tag]
            elif name and name in params:
                val = params[name]
            else:
                # Default: use init value if available
                val = getattr(node, "kwargs", {}).get("init", 1.0)

            # Return a constant atom
            return AtomNode(kind="const", var_idxs=(), kwargs={"value": float(val)}, tag=tag)
        # Other atoms: pass through
        return node

    elif isinstance(node, (AddNode, MulNode)):
        # Binary operations: recursively substitute children
        left = _substitute_free_consts(node.left, params)
        right = _substitute_free_consts(node.right, params)
        return type(node)(left, right)

    elif isinstance(node, (PowNode, ExpNode, SinNode, CosNode)):
        # Unary/binary operations: recursively substitute
        if hasattr(node, "base") and hasattr(node, "exponent"):
            # PowNode
            base = _substitute_free_consts(node.base, params)
            exponent = _substitute_free_consts(node.exponent, params)
            return PowNode(base, exponent)
        elif hasattr(node, "arg"):
            # ExpNode, SinNode, CosNode
            arg = _substitute_free_consts(node.arg, params)
            return type(node)(arg)
        else:
            return node

    else:
        # Unknown node type: pass through
        return node


def varpro_refine_linear(
    ode_result: DESearchResult,
    surrogate: nn.Module,
    train_dataloader,
    val_dataloader=None,
    *,
    cfg: Optional[DESearchConfig] = None,
    device=None,
    epochs: int = 500,
    strategy: str = "direct_solve",
    verbose: bool = True,
) -> DESearchResult:
    """Refine linear DE coefficients using Variable Projection + LM.

    Takes an existing DESearchResult from STLSQ and refines the linear
    coefficients using the Levenberg-Marquardt optimizer with analytical
    elimination of linear parameters (Variable Projection).

    Parameters
    ----------
    ode_result : DESearchResult
        Initial DE discovered by STLSQ
    surrogate : nn.Module
        Frozen neural network surrogate for u(x)
    train_dataloader : DataLoader
        Training data
    val_dataloader : DataLoader, optional
        Validation data
    cfg : DESearchConfig, optional
        DE search configuration
    device : torch.device, optional
        Device for computation
    epochs : int
        Maximum LM iterations
    strategy : str
        LM strategy ('direct_solve', 'explicit', 'matfree')
    verbose : bool
        Print progress

    Returns
    -------
    DESearchResult
        Refined DE with updated coefficients and varpro_metadata

    Notes
    -----
    Phase 1 implementation focuses on improving coefficient accuracy for
    existing linear terms discovered by STLSQ. No new terms are added.

    Example
    -------
    >>> # After STLSQ discovery
    >>> refined_result = varpro_refine_linear(
    ...     ode_result, surrogate, train_dl, val_dl,
    ...     epochs=500, strategy='direct_solve'
    ... )
    >>> print(f"Original coeff: {ode_result.coeffs[0]:.6f}")  # 0.999995
    >>> print(f"Refined coeff: {refined_result.coeffs[0]:.6f}")  # 1.000000
    """
    if verbose:
        print("\n" + "=" * 70)
        print("VARPRO LINEAR REFINEMENT")
        print("=" * 70)
        print(f"Input: {len(ode_result.term_asts)} terms, order {ode_result.order}")
        val_str = f"{ode_result.rms_val:.6e}" if ode_result.rms_val is not None else "N/A"
        print(f"STLSQ RMS: train={ode_result.rms_train:.6e}, val={val_str}")

    # Setup device
    if device is None:
        device = next(surrogate.parameters()).device

    # Create cache for surrogate derivatives
    cache = UFeatureCache(surrogate)

    # Extract ridge parameter from config
    ridge = getattr(cfg, "ridge", 1e-10) if cfg else 1e-10

    # Create VarPro adaptor
    # For Phase 1, we don't have a composite_model yet - we'll work directly
    # with the library terms from STLSQ
    varpro_adaptor = DEVarProAdaptor(
        composite_model=None,  # Phase 1: no composite model needed
        order=ode_result.order,
        x_axis=ode_result.x_axis,
        cache=cache,
        term_asts=ode_result.term_asts,  # Pass discovered terms
        lambda_reg=ridge,
    )
    varpro_adaptor = varpro_adaptor.to(device)

    # For Phase 1, we optimize by iteratively evaluating the library
    # and solving for coefficients. Since there are no structural parameters
    # to optimize, we just compute the best coefficients over the full dataset.

    # Gather all training data
    all_x_train = []
    for batch in train_dataloader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        all_x_train.append(x.to(device))

    X_train = torch.cat(all_x_train, dim=0)  # (N_train, Nx)

    # Compute optimal coefficients on full training set
    with torch.no_grad():
        result_train = varpro_adaptor(X_train)
        beta_train = result_train["beta"].detach()
        refined_coeffs = beta_train.cpu()
        residuals_train = result_train["residuals"]
        rms_train = float(residuals_train.square().mean().sqrt().item())

    # Compute validation RMS using the *training* coefficients (no refit on val)
    rms_val = None
    if val_dataloader is not None:
        all_x_val = []
        for batch in val_dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            all_x_val.append(x.to(device))

        X_val = torch.cat(all_x_val, dim=0)

        with torch.no_grad():
            result_val = varpro_adaptor(X_val, beta_override=beta_train)
            residuals_val = result_val["residuals"]
            rms_val = float(residuals_val.square().mean().sqrt().item())

    if verbose:
        print("\nVarPro refined coefficients:")
        for i, (term, c_old, c_new) in enumerate(
            zip(ode_result.term_asts, ode_result.coeffs.tolist(), refined_coeffs.tolist())
        ):
            term_str = "1" if term is None else repr(term)
            delta = c_new - c_old
            print(f"  [{i}] {c_old:12.6g} → {c_new:12.6g}  (Δ={delta:+.2e})  {term_str}")

        print("\nRMS comparison:")
        stlsq_val_str = f"{ode_result.rms_val:.6e}" if ode_result.rms_val is not None else "N/A"
        varpro_val_str = f"{rms_val:.6e}" if rms_val is not None else "N/A"
        print(f"  STLSQ:  train={ode_result.rms_train:.6e}, val={stlsq_val_str}")
        print(f"  VarPro: train={rms_train:.6e}, val={varpro_val_str}")

        if rms_train < ode_result.rms_train:
            improvement = (ode_result.rms_train - rms_train) / ode_result.rms_train * 100
            print(f"  → Improvement: {improvement:.2f}% reduction in train RMS")

    # Build refined result
    refined_result = DESearchResult(
        order=ode_result.order,
        x_axis=ode_result.x_axis,
        term_asts=ode_result.term_asts,
        coeffs=refined_coeffs,
        rms_train=rms_train,
        rms_val=rms_val,
        condition_number=getattr(ode_result, "condition_number", None),
        term_sources=getattr(ode_result, "term_sources", None),
        prolongation_metadata=getattr(ode_result, "prolongation_metadata", None),
        residual_ast=build_de_residual_ast(
            DESearchResult(
                order=ode_result.order,
                x_axis=ode_result.x_axis,
                term_asts=ode_result.term_asts,
                coeffs=refined_coeffs,
                rms_train=rms_train,
                rms_val=rms_val,
                condition_number=getattr(ode_result, "condition_number", None),
            ),
            units_spec=getattr(cfg, "units_spec", None),
            enforce_units=bool(getattr(cfg, "enforce_units", False)),
        ),
        varpro_metadata={
            "method": "varpro_linear_phase1",
            "strategy": strategy,
            "epochs_requested": epochs,
            "original_rms_train": ode_result.rms_train,
            "original_rms_val": ode_result.rms_val,
            "improvement_train": (ode_result.rms_train - rms_train)
            if rms_train < ode_result.rms_train
            else 0.0,
        },
    )
    for _attr in ("expr_ir_report", "expr_ir_reports_by_order"):
        if hasattr(ode_result, _attr):
            setattr(refined_result, _attr, getattr(ode_result, _attr))

    if verbose:
        print("=" * 70)

    return refined_result


def varpro_refine_linear_multi(
    ode_result,  # Union[DESearchResult, DESearchResultMulti]
    surrogates: list[nn.Module],
    train_dataloaders: list,
    val_dataloaders: Optional[list] = None,
    *,
    cfg: Optional[DESearchConfig] = None,
    device=None,
    epochs: int = 500,
    strategy: str = "direct_solve",
    verbose: bool = True,
) -> list[DESearchResult]:
    """Refine linear DE coefficients for multiple datasets using VarPro.

    Takes a DESearchResult or DESearchResultMulti with shared term support
    and refines the linear coefficients separately for each dataset using
    Variable Projection with analytical coefficient elimination.

    Parameters
    ----------
    ode_result : DESearchResult or DESearchResultMulti
        Initial DE discovered by STLSQ (typically DESearchResultMulti from
        discover_de_from_surrogates with multi-dataset)
    surrogates : list of nn.Module
        Frozen neural network surrogates for u(x), one per dataset
    train_dataloaders : list of DataLoader
        Training data loaders, one per dataset
    val_dataloaders : list of DataLoader, optional
        Validation data loaders, one per dataset
    cfg : DESearchConfig, optional
        DE search configuration
    device : torch.device, optional
        Device for computation
    epochs : int
        Maximum LM iterations (not used in Phase 1, kept for consistency)
    strategy : str
        LM strategy (not used in Phase 1, kept for consistency)
    verbose : bool
        Print progress

    Returns
    -------
    list of DESearchResult
        Refined DE results, one per dataset, with updated coefficients

    Notes
    -----
    This function applies VarPro analytically to each dataset independently,
    using the *shared term support* but allowing different coefficients.
    Unlike template search, no parameters are optimized jointly - this is
    purely linear coefficient refinement.

    The results are returned as a list of DESearchResult for easier processing.
    They can be merged back into DESearchResultMulti via _merge_results_to_multi
    in run_de.py.

    Example
    -------
    >>> # After multi-dataset STLSQ discovery
    >>> refined_results = varpro_refine_linear_multi(
    ...     ode_result_multi, surrogates, train_dls, val_dls,
    ...     epochs=500, strategy='direct_solve'
    ... )
    >>> # refined_results is List[DESearchResult], one per dataset
    """
    # Extract shared term support
    from nestynet_sr.sr_de.de_search import DESearchResultMulti

    if isinstance(ode_result, DESearchResultMulti):
        term_asts = ode_result.term_asts
        order = ode_result.order
        x_axis = ode_result.x_axis
        D = len(surrogates)
    else:
        # Single-dataset result - should not happen in multi-dataset mode
        # but handle gracefully
        term_asts = ode_result.term_asts
        order = ode_result.order
        x_axis = ode_result.x_axis
        D = len(surrogates)

    if verbose:
        print("\n" + "=" * 70)
        print("VARPRO LINEAR REFINEMENT (MULTI-DATASET)")
        print("=" * 70)
        print(f"Input: {len(term_asts)} shared terms, order {order}")
        print(f"Datasets: {D}")

    # Setup device
    if device is None:
        device = next(surrogates[0].parameters()).device

    # Extract ridge parameter
    ridge = getattr(cfg, "ridge", 1e-10) if cfg else 1e-10

    # Refine each dataset independently
    refined_results = []

    for d in range(D):
        if verbose:
            print(f"\n[Dataset {d + 1}/{D}]")

        # Create cache for this surrogate
        cache = UFeatureCache(surrogates[d])

        # Create VarPro adaptor for this dataset
        varpro_adaptor = DEVarProAdaptor(
            composite_model=None,
            order=order,
            x_axis=x_axis,
            cache=cache,
            term_asts=term_asts,
            lambda_reg=ridge,
        )
        varpro_adaptor = varpro_adaptor.to(device)

        # Gather all training data for this dataset
        X_train = _gather_all_x(train_dataloaders[d], device)

        # Compute optimal coefficients on full training set
        with torch.no_grad():
            result_train = varpro_adaptor(X_train)
            beta_train = result_train["beta"].detach()
            refined_coeffs = beta_train.cpu()
            residuals_train = result_train["residuals"]
            rms_train = float(residuals_train.square().mean().sqrt().item())

        # Compute validation RMS using training coefficients (no refit on val)
        rms_val = None
        if val_dataloaders is not None and d < len(val_dataloaders):
            X_val = _gather_all_x(val_dataloaders[d], device)

            with torch.no_grad():
                result_val = varpro_adaptor(X_val, beta_override=beta_train)
                residuals_val = result_val["residuals"]
                rms_val = float(residuals_val.square().mean().sqrt().item())

        if verbose:
            val_str = f"{rms_val:.6e}" if rms_val is not None else "N/A"
            print(f"  VarPro RMS: train={rms_train:.6e}, val={val_str}")

        # Build refined result for this dataset
        refined_result = DESearchResult(
            order=order,
            x_axis=x_axis,
            term_asts=term_asts,
            coeffs=refined_coeffs,
            rms_train=rms_train,
            rms_val=rms_val,
            condition_number=getattr(ode_result, "condition_number", None),
            term_sources=getattr(ode_result, "term_sources", None),
            prolongation_metadata=getattr(ode_result, "prolongation_metadata", None),
            residual_ast=build_de_residual_ast(
                DESearchResult(
                    order=order,
                    x_axis=x_axis,
                    term_asts=term_asts,
                    coeffs=refined_coeffs,
                    rms_train=rms_train,
                    rms_val=rms_val,
                    condition_number=getattr(ode_result, "condition_number", None),
                ),
                units_spec=getattr(cfg, "units_spec", None),
                enforce_units=bool(getattr(cfg, "enforce_units", False)),
            ),
            varpro_metadata={
                "method": "varpro_linear_phase1_multi",
                "dataset_index": d,
                "strategy": strategy,
                "epochs_requested": epochs,
            },
        )
        for _attr in ("expr_ir_report", "expr_ir_reports_by_order"):
            if hasattr(ode_result, _attr):
                setattr(refined_result, _attr, getattr(ode_result, _attr))

        refined_results.append(refined_result)

    if verbose:
        print("=" * 70)

    return refined_results


# -----------------------------------------------------------------------------
# Phase 2 helper: LM over template nonlinear parameters ψ (VarPro)
# -----------------------------------------------------------------------------


def _make_fullbatch_dataloader(
    X: torch.Tensor, y: torch.Tensor, *, batch_size: Optional[int] = None
):
    """Build a deterministic full-batch DataLoader from tensors.

    We keep tensors on CPU here; NestyNet's ResidualsModule moves data to the
    requested device.
    """
    from torch.utils.data import DataLoader, TensorDataset

    Xc = X.detach().cpu()
    yc = y.detach().cpu()
    if yc.ndim == 2 and yc.shape[1] == 1:
        yc = yc.squeeze(1)

    ds = TensorDataset(Xc, yc)
    bs = int(batch_size) if batch_size is not None else len(ds)
    return DataLoader(ds, batch_size=bs, shuffle=False)


def _lm_optimize_template_params_multi(
    *,
    baseline_term_asts: list[Node],
    template: TemplateInstance,
    surrogates: list[nn.Module],
    X_train_list: list[torch.Tensor],
    y_train_list: list[torch.Tensor],
    order: int,
    x_axis: int,
    device,
    ridge: float = 0.0,
    lm_epochs: int = 200,
    lm_epochs_min: int = 20,
    lm_nval_patience: int = 50,
    lm_loss_target: Optional[float] = None,
    lm_strategy: str = "direct_solve",
    lm_chisq_tol: float = 1e-10,
    verbose: bool = False,
) -> tuple[Dict[str, float], float]:
    """Optimize template nonlinear parameters ψ jointly across datasets with weighted LM.

    Creates shared template parameters across all datasets while allowing
    dataset-specific linear coefficients. Uses dataset size weighting to
    prevent bias toward larger datasets.

    Parameters
    ----------
    baseline_term_asts : list of Node
        Baseline term ASTs (shared across datasets)
    template : TemplateInstance
        Template to optimize (shares ψ across datasets)
    surrogates : list of nn.Module
        Frozen surrogates, one per dataset
    X_train_list : list of torch.Tensor
        Training inputs per dataset
    y_train_list : list of torch.Tensor
        Training targets per dataset (anchor residuals)
    order : int
        DE order
    x_axis : int
        Derivative axis
    device : torch.device
        Device
    ridge : float
        Ridge regularization for linear coefficients
    lm_epochs : int
        Maximum LM iterations
    lm_epochs_min : int
        Minimum LM iterations before early stopping
    lm_nval_patience : int
        Patience for early stopping
    lm_loss_target : float, optional
        Target loss for early stopping
    lm_strategy : str
        LM strategy
    lm_chisq_tol : float
        Convergence tolerance
    verbose : bool
        Print progress

    Returns
    -------
    psi : dict
        Optimized template parameters (shared across datasets)
    final_loss : float
        Final weighted loss

    Notes
    -----
    This function creates one TemplateVarProBase per dataset, shares their
    parameters via share_params_from(), and optimizes the shared ψ while
    VarPro analytically eliminates per-dataset linear coefficients β_d.
    """
    if template.params is None or len(template.params) == 0:
        return {}, float("nan")

    D = len(surrogates)

    # Lazy imports
    from adaptors.template_varpro_adaptor import TemplateBaseAdaptor
    from adaptors.template_varpro_base import TemplateVarProBase

    try:
        from nestynet.adaptors.varpro_adaptors import VarProAdaptor as NestyVarProAdaptor
    except Exception:
        try:
            from nestynet.adaptors.varpro_adaptor import VarProAdaptor as NestyVarProAdaptor
        except Exception as e:
            raise RuntimeError(
                "Could not import nestynet VarProAdaptor; LM-over-ψ requires NestyNet."
            ) from e

    import nestynet.optimizer

    # Compute dataset weights (proportional to size)
    weights = _compute_dataset_weights(X_train_list)

    # Build combined feature model per dataset: baseline + template
    baseline_insts: list[TemplateInstance] = []
    for i, t in enumerate(baseline_term_asts):
        baseline_insts.append(
            TemplateInstance(
                template_name="baseline",
                ast=t,
                params={},
                param_bounds={},
                description=f"baseline_{i}",
            )
        )

    tmpl_inst = TemplateInstance(
        template_name=template.template_name,
        ast=template.ast,
        params=dict(template.params),
        param_bounds=dict(getattr(template, "param_bounds", {}) or {}),
        description=template.description,
    )

    # Create TemplateVarProBase per dataset
    bases = []
    for d in range(D):
        cache_d = UFeatureCache(surrogates[d])
        base_d = TemplateVarProBase(
            template_instances=baseline_insts + [tmpl_inst],
            cache=cache_d,
            order=order,
            x_axis=x_axis,
        ).to(device)
        bases.append(base_d)

    # Share parameters from base[0] to all others
    for d in range(1, D):
        bases[d].share_params_from(bases[0])

    # Wrap each in TemplateBaseAdaptor with global_map_offset=0 (shared parameter block)
    base_adaptors = [TemplateBaseAdaptor(bases[d], global_map_offset=0) for d in range(D)]

    # Wrap each in VarProAdaptor (eliminates β_d per dataset)
    varpro_models = []
    for d in range(D):
        varpro_d = NestyVarProAdaptor(
            base_adaptors[d],
            bias=False,
            lambda_reg=ridge,
            expected_n_out=len(baseline_insts) + 1,
            target_dim=1,
            shape_policy="strict",
        )
        varpro_models.append(varpro_d)

    # Build full-batch dataloaders per dataset
    dataloaders = [_make_fullbatch_dataloader(X_train_list[d], y_train_list[d]) for d in range(D)]

    # Create weighted residual module factories
    # Weight by sqrt(weight) since loss is squared residuals
    def weighted_seg_factory(provider, dataloader, weight):
        def factory(_):
            rm = nestynet.optimizer.ResidualsModule(
                providers=[provider], dataloader=dataloader, device=device
            )
            # Scale residuals by sqrt(weight) for correct weighted loss
            # Since loss = sum(residuals^2), we want weight * sum(residuals^2)
            # So we scale residuals by sqrt(weight)
            rm_wrapped = WeightedResidualsWrapper(rm, weight)
            return rm_wrapped

        return factory

    # Simple wrapper class for weighted residuals
    class WeightedResidualsWrapper:
        def __init__(self, base_rm, weight):
            self.base_rm = base_rm
            self.weight_factor = float(torch.sqrt(weight).item())

        def __call__(self, *args, **kwargs):
            residuals = self.base_rm(*args, **kwargs)
            return residuals * self.weight_factor

        def __getattr__(self, name):
            # Delegate all other attributes to base_rm
            return getattr(self.base_rm, name)

    residual_factories = [
        weighted_seg_factory(varpro_models[d], dataloaders[d], weights[d]) for d in range(D)
    ]

    # Create optimizer with shared parameters from base[0]
    params = list(bases[0].parameters())

    from nestynet_sr.sr_search.training import (
        SR_LM_OVERRIDES,
        _sr_latest_joint_loss_metrics,
    )

    cfg = nestynet.optimizer.LMConfig(
        verbose=False,
        LM_strategy=lm_strategy,
        chisq_tol=lm_chisq_tol,
        **SR_LM_OVERRIDES,
    )

    lm_opt = nestynet.optimizer.Predictive_LM_Optimizer(
        params,
        residual_factories,
        residual_module_factories_val=None,
        cfg=cfg,
    )

    # Run LM loop
    best_loss = float("inf")
    best_p = None
    patience_counter = 0

    for epoch in range(lm_epochs):
        step_result = lm_opt.step()
        loss_obj = step_result[0] if isinstance(step_result, tuple) else step_result
        loss_metrics = _sr_latest_joint_loss_metrics(
            lm_opt,
            target_count=D,
            label="[varpro shared psi] ",
        )
        loss = float(loss_metrics.get("train_selection_loss", loss_metrics.get("train_data_mean_loss", loss_obj)))

        if loss < best_loss:
            best_loss = loss
            best_p = lm_opt._flat_params().clone().detach()
            patience_counter = 0
        else:
            patience_counter += 1

        # Check stopping criteria
        if epoch >= lm_epochs_min:
            if lm_loss_target is not None and loss < lm_loss_target:
                break
            if patience_counter >= lm_nval_patience:
                break

        if lm_opt.state.get("halt", False):
            break

    if best_p is not None:
        lm_opt._update_param_groups(best_p)

    # Extract optimized ψ from base[0] (shared across all datasets)
    psi = {k: float(v.detach().cpu().item()) for k, v in bases[0].param_dict.items()}

    return psi, float(best_loss)


def _lm_optimize_template_params(
    *,
    baseline_term_asts: list[Node],
    template: TemplateInstance,
    surrogate: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    order: int,
    x_axis: int,
    device,
    ridge: float = 0.0,
    lm_epochs: int = 200,
    lm_epochs_min: int = 20,
    lm_nval_patience: int = 50,
    lm_loss_target: Optional[float] = None,
    lm_strategy: str = "direct_solve",
    lm_chisq_tol: float = 1e-10,
    verbose: bool = False,
) -> tuple[Dict[str, float], float]:
    """Optimize template nonlinear parameters ψ with LM while eliminating linear β via VarPro.

    Notes
    -----
    * We include the current baseline terms in the feature matrix so ψ is fit
      in-context.
    * We deliberately do *not* use a separate validation set inside this LM
      inner loop to avoid optimistic leakage (re-solving β on validation).
      The outer template search still scores using β fitted on train and
      evaluated on val.
    """
    if template.params is None or len(template.params) == 0:
        return {}, float("nan")

    # Lazy imports: Phase 2 fixed-ψ mode doesn't need NestyNet.
    from adaptors.template_varpro_adaptor import TemplateBaseAdaptor
    from adaptors.template_varpro_base import TemplateVarProBase

    try:
        from nestynet.adaptors.varpro_adaptors import VarProAdaptor as NestyVarProAdaptor
    except Exception:
        try:
            from nestynet.adaptors.varpro_adaptor import VarProAdaptor as NestyVarProAdaptor
        except Exception as e:
            raise RuntimeError(
                "Could not import nestynet VarProAdaptor; LM-over-ψ requires NestyNet."
            ) from e

    # Build combined feature model: baseline terms (no params) + this template (params).
    cache = UFeatureCache(surrogate)

    baseline_insts: list[TemplateInstance] = []
    for i, t in enumerate(baseline_term_asts):
        baseline_insts.append(
            TemplateInstance(
                template_name="baseline",
                ast=t,
                params={},
                param_bounds={},
                description=f"baseline_{i}",
            )
        )

    tmpl_inst = TemplateInstance(
        template_name=template.template_name,
        ast=template.ast,
        params=dict(template.params),
        param_bounds=dict(getattr(template, "param_bounds", {}) or {}),
        description=template.description,
    )

    base = TemplateVarProBase(
        template_instances=baseline_insts + [tmpl_inst],
        cache=cache,
        order=order,
        x_axis=x_axis,
    ).to(device)  # Move the base module to device (it's an nn.Module)

    base_adaptor = TemplateBaseAdaptor(base)

    # Eliminate all linear coefficients (baseline + template) by VarPro.
    # No .to() needed: the wrapped base module was already moved to device above.
    varpro_model = NestyVarProAdaptor(
        base_adaptor,
        bias=False,
        lambda_reg=ridge,
        expected_n_out=len(baseline_insts) + 1,
        target_dim=1,
        shape_policy="strict",
    )

    # Full-batch LM on train only.
    dl_train = _make_fullbatch_dataloader(X_train, y_train)

    # Optimize psi only: take the parameters straight from the base module rather
    # than the VarPro wrapper, whose linear head is eliminated analytically.
    import nestynet.optimizer

    params = list(base.parameters())

    def seg_factory(dataloader):
        def factory(_):
            return nestynet.optimizer.ResidualsModule(
                providers=[varpro_model],  # VarProAdaptor as provider
                dataloader=dataloader,
                device=device,
            )

        return factory

    from nestynet_sr.sr_search.training import (
        SR_LM_OVERRIDES,
        _sr_latest_single_target_loss_metrics,
    )

    cfg = nestynet.optimizer.LMConfig(
        verbose=False,
        LM_strategy=lm_strategy,
        chisq_tol=lm_chisq_tol,
        **SR_LM_OVERRIDES,
    )

    lm_opt = nestynet.optimizer.Predictive_LM_Optimizer(
        params,
        [seg_factory(dl_train)],
        residual_module_factories_val=None,  # No separate validation for inner loop
        cfg=cfg,
    )

    # Run LM loop (train-only, no validation since we validate at outer level)
    best_loss = float("inf")
    best_p = None
    patience_counter = 0

    for epoch in range(lm_epochs):
        step_result = lm_opt.step()
        # step() may return loss or (loss, info_dict)
        loss_obj = step_result[0] if isinstance(step_result, tuple) else step_result
        loss_metrics = _sr_latest_single_target_loss_metrics(
            lm_opt,
            label="[varpro template psi] ",
        )
        loss = float(loss_metrics.get("train_selection_loss", loss_metrics.get("train_data_mean_loss", loss_obj)))

        if loss < best_loss:
            best_loss = loss
            best_p = lm_opt._flat_params().clone().detach()
            patience_counter = 0
        else:
            patience_counter += 1

        # Check stopping criteria
        if epoch >= lm_epochs_min:
            if lm_loss_target is not None and loss < lm_loss_target:
                break
            if patience_counter >= lm_nval_patience:
                break

        if lm_opt.state.get("halt", False):
            break

    if best_p is not None:
        lm_opt._update_param_groups(best_p)

    best_val_loss = best_loss  # Use final training loss as the "validation" loss

    # Read ψ back out of the base module.
    psi = {k: float(v.detach().cpu().item()) for k, v in base.param_dict.items()}

    return psi, float(best_val_loss)


def varpro_template_search(
    ode_result: DESearchResult,
    surrogate: nn.Module,
    train_dataloader,
    val_dataloader=None,
    *,
    template_families: list[str] = None,
    cfg: Optional[DESearchConfig] = None,
    device=None,
    max_templates: int = 3,
    complexity_penalty: float = 1e-3,
    prefer_autonomous: bool = False,
    prefer_forced: bool = False,
    x_penalty: float = 0.0,
    mixed_penalty: float = 0.0,
    enable_support_minimization: bool = False,
    rms_tol_factor: float = 1.05,
    optimize_psi: bool = False,
    psi_lm_epochs: int = 200,
    psi_lm_epochs_min: int = 20,
    psi_lm_nval_patience: int = 50,
    psi_lm_loss_target: Optional[float] = None,
    psi_lm_strategy: str = "direct_solve",
    psi_lm_chisq_tol: float = 1e-10,
    verbose: bool = True,
) -> DESearchResult:
    """Search over template families to discover nonlinear DE forms.

    Phase 2 implementation: extends the linear library from STLSQ with
    parameterized templates (power laws, exponentials, sinusoids, saturation).

    Parameters
    ----------
    ode_result : DESearchResult
        Baseline DE from STLSQ (Phase 1)
    surrogate : nn.Module
        Frozen neural network surrogate for u(x)
    train_dataloader : DataLoader
        Training data
    val_dataloader : DataLoader, optional
        Validation data
    template_families : list[str], optional
        Template families to try (default: ['power', 'exp'])
    cfg : DESearchConfig, optional
        DE search configuration
    device : torch.device, optional
        Device for computation
    max_templates : int
        Maximum number of template terms to add
    complexity_penalty : float
        Penalty per term (encourages parsimony)
    prefer_autonomous : bool
        If True, penalize x-only and mixed terms (prefer state-only)
    prefer_forced : bool
        If True, penalize state-only terms (prefer forcing)
    x_penalty : float
        Additional penalty per x-only term (default: 0.0)
    mixed_penalty : float
        Additional penalty per mixed term (default: 0.0)
    verbose : bool
        Print progress

    Returns
    -------
    DESearchResult
        Extended DE with template terms and varpro_metadata

    Notes
    -----
    Phase 2 MVP strategy:
    1. Generate template instances from specified families
    2. Initialize template parameters heuristically (FFT, log-log, etc.)
    3. Add templates to library alongside existing linear terms
    4. Solve for coefficients with extended library
    5. Select best model based on validation loss + complexity

    If optimize_psi=True, each template trial runs an inner LM loop to optimize
    the nonlinear template parameters ψ (while eliminating linear coefficients by VarPro).

    Example
    -------
    >>> # After STLSQ + Phase 1
    >>> extended_result = varpro_template_search(
    ...     ode_result, surrogate, train_dl, val_dl,
    ...     template_families=['power', 'exp'],
    ...     max_templates=2
    ... )
    >>> # Might discover u_x = k*u^2 instead of linear terms
    """
    # Set default penalties based on preferences
    if prefer_autonomous and x_penalty == 0.0:
        x_penalty = 1e-2  # Penalize forcing terms
        mixed_penalty = max(mixed_penalty, 5e-3)  # Also penalize mixed
    if prefer_forced and x_penalty == 0.0:
        # Penalize state-only terms by not penalizing x-only
        # (effectively making x-only preferred)
        x_penalty = 0.0
        mixed_penalty = max(mixed_penalty, 1e-3)

    if verbose:
        print("\n" + "=" * 70)
        print("VARPRO TEMPLATE SEARCH (Phase 2)")
        print("=" * 70)
        print(f"Baseline: {len(ode_result.term_asts)} linear terms")
        val_str = f"{ode_result.rms_val:.6e}" if ode_result.rms_val is not None else "N/A"
        print(f"Baseline RMS: train={ode_result.rms_train:.6e}, val={val_str}")

        # Show preference settings
        if prefer_autonomous:
            print(
                f"Preference: AUTONOMOUS (x_penalty={x_penalty:.2e}, mixed_penalty={mixed_penalty:.2e})"
            )
        elif prefer_forced:
            print(f"Preference: FORCED (mixed_penalty={mixed_penalty:.2e})")
        elif x_penalty > 0 or mixed_penalty > 0:
            print(f"Custom penalties: x_penalty={x_penalty:.2e}, mixed_penalty={mixed_penalty:.2e}")

    # Setup device
    if device is None:
        device = next(surrogate.parameters()).device

    # Default template families
    if template_families is None:
        template_families = ["power", "exp"]

    # Validate template families
    for fam in template_families:
        if fam not in TEMPLATE_REGISTRY:
            raise ValueError(
                f"Unknown template family '{fam}'. Available: {list(TEMPLATE_REGISTRY.keys())}"
            )

    # Create cache for surrogate derivatives
    cache = UFeatureCache(surrogate)

    # Gather training data
    all_x_train = []
    for batch in train_dataloader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        all_x_train.append(x.to(device))

    X_train = torch.cat(all_x_train, dim=0)  # (N_train, Nx)

    # Compute surrogate values and derivatives for template initialization
    cache.ensure(X_train, need_grad=True, need_hess=False)
    u_train = cache.u  # (N, 1)
    du_train = cache.g[:, 0, ode_result.x_axis : ode_result.x_axis + 1]  # (N, 1)

    # Get anchor and target
    if ode_result.order == 1:
        cache.ensure(X_train, need_grad=True, need_hess=False)
        anchor_train = cache.g[:, 0, ode_result.x_axis]  # (N,)
    else:
        cache.ensure(X_train, need_grad=False, need_hess=True)
        anchor_train = cache.H[:, 0, ode_result.x_axis, ode_result.x_axis]  # (N,)

    target_train = -anchor_train  # (N,)

    # Hyperparams reused across trials
    stlsq_lambda = getattr(cfg, "stlsq_lambda", 1e-3) if cfg else 1e-3
    stlsq_max_iter = getattr(cfg, "stlsq_max_iter", 10) if cfg else 10
    ridge = getattr(cfg, "ridge", 1e-10) if cfg else 1e-10
    units_spec = getattr(cfg, "units_spec", None) if cfg is not None else None
    enforce_units = bool(getattr(cfg, "enforce_units", False)) if cfg is not None else False

    if verbose and optimize_psi:
        print(
            f"  Template ψ LM: epochs={psi_lm_epochs}, strategy={psi_lm_strategy}, ridge={ridge:.1e}"
        )

    # Determine available x variables
    Nx = X_train.shape[1]
    x_vars = list(range(Nx))

    if verbose:
        print(f"\nGenerating templates from families: {template_families}")
        print(f"  Available x variables: {x_vars}")

    # Generate and initialize all template instances
    all_templates = []
    for fam_name in template_families:
        template_gen = get_template(fam_name)

        # Build instances
        instances = template_gen.build_instances(
            x_vars=x_vars, include_u=True, include_du=False, x_axis=ode_result.x_axis
        )

        # Initialize parameters for each instance
        for inst in instances:
            try:
                init_params = template_gen.init_params(
                    inst, X_train, u_train, du_train, target_train
                )
                # Update instance with initialized params
                inst.params = init_params

                # Canonicalize
                inst.params = template_gen.canonicalize(inst.params)

                all_templates.append(inst)

                if verbose:
                    param_str = ", ".join([f"{k}={v:.3f}" for k, v in inst.params.items()])
                    print(f"    [{fam_name}] {inst.description}: {param_str}")

            except Exception as e:
                if verbose:
                    print(f"    [SKIP] {inst.description}: initialization failed ({e})")
                continue

    if not all_templates:
        if verbose:
            print("\nNo valid templates generated. Returning baseline.")
            print("=" * 70)
        return ode_result

    if verbose:
        print(f"\nGenerated {len(all_templates)} template instances")
        print(f"Will try adding up to {max_templates} templates")

    # Simple greedy search: add one template at a time, keep if improves
    best_result = ode_result
    best_score = float("inf")

    # Baseline score
    if ode_result.rms_val is not None:
        baseline_score = ode_result.rms_val + complexity_penalty * len(ode_result.term_asts)
    else:
        baseline_score = ode_result.rms_train + complexity_penalty * len(ode_result.term_asts)

    best_score = baseline_score

    if verbose:
        print(
            f"\nBaseline score: {baseline_score:.6e} (RMS + {complexity_penalty:.0e} * {len(ode_result.term_asts)} terms)"
        )

    # Try adding each template
    for i, tmpl in enumerate(all_templates[: max_templates * 3]):  # Try more than max, keep best
        if verbose:
            print(
                f"\n--- Trial {i + 1}/{min(len(all_templates), max_templates * 3)}: {tmpl.description} ---"
            )

        try:
            # Choose ψ (template nonlinear params): either fixed init, or LM-refined.
            psi_init = dict(tmpl.params)
            psi_used = psi_init
            psi_lm_loss = None

            # Fast unit-feasibility gate before optional ψ-LM.
            if units_spec is not None and enforce_units:
                tmpl_ast_init = _substitute_free_consts(tmpl.ast, psi_init)
                ok_units, why_units = term_units_feasible(
                    tmpl_ast_init,
                    order=ode_result.order,
                    x_axis=ode_result.x_axis,
                    units_spec=units_spec,
                    enforce_units=enforce_units,
                )
                if not ok_units:
                    if verbose:
                        print(f"  [SKIP][Units] {tmpl.description}: {why_units}")
                    continue

            if optimize_psi and len(psi_init) > 0:
                if verbose:
                    param_str = ", ".join([f"{k}={v:.3f}" for k, v in psi_init.items()])
                    print(f"  ψ init: {param_str}")
                    print(f"  → LM over ψ (epochs={psi_lm_epochs}, strategy={psi_lm_strategy}) ...")

                psi_used, psi_lm_loss = _lm_optimize_template_params(
                    baseline_term_asts=ode_result.term_asts,
                    template=tmpl,
                    surrogate=surrogate,
                    X_train=X_train,
                    y_train=target_train,
                    order=ode_result.order,
                    x_axis=ode_result.x_axis,
                    device=device,
                    ridge=ridge,
                    lm_epochs=psi_lm_epochs,
                    lm_epochs_min=psi_lm_epochs_min,
                    lm_nval_patience=psi_lm_nval_patience,
                    lm_loss_target=psi_lm_loss_target,
                    lm_strategy=psi_lm_strategy,
                    lm_chisq_tol=psi_lm_chisq_tol,
                    verbose=verbose,
                )

                # Canonicalize refined ψ
                psi_used = get_template(tmpl.template_name).canonicalize(psi_used)

                if verbose:
                    param_str = ", ".join([f"{k}={v:.6g}" for k, v in psi_used.items()])
                    print(f"  ψ fit:  {param_str} (lm_loss={psi_lm_loss:.3e})")

            # Substitute template parameters (ψ) into the AST to produce a fixed term.
            tmpl_ast_substituted = _substitute_free_consts(tmpl.ast, psi_used)

            if units_spec is not None:
                ok_units, why_units = term_units_feasible(
                    tmpl_ast_substituted,
                    order=ode_result.order,
                    x_axis=ode_result.x_axis,
                    units_spec=units_spec,
                    enforce_units=enforce_units,
                )
                if not ok_units:
                    if verbose:
                        print(f"  [SKIP][Units] {tmpl.description}: {why_units}")
                    continue

            # Build extended library: baseline terms + template term (with params substituted)
            extended_term_asts = ode_result.term_asts + [tmpl_ast_substituted]

            # Create VarPro adaptor with extended library
            varpro_adaptor = DEVarProAdaptor(
                composite_model=None,
                order=ode_result.order,
                x_axis=ode_result.x_axis,
                cache=cache,
                term_asts=extended_term_asts,
                lambda_reg=ridge,
            ).to(device)

            # Solve for coefficients on training data
            cache.reset()
            with torch.no_grad():
                result_train = varpro_adaptor(X_train)
                extended_coeffs_dense = result_train["beta"].detach()
                X_matrix = result_train["X"]  # (N, K)
                y_vector = result_train["y"]  # (N,)

                extended_coeffs_sparse, keep_mask = stlsq(
                    X_matrix, y_vector, ridge=ridge, lam=stlsq_lambda, max_iter=stlsq_max_iter
                )

                # Unbiased refit on selected support
                if keep_mask.sum() > 0:
                    X_selected = X_matrix[:, keep_mask]
                    extended_coeffs_sparse = ridge_lstsq(X_selected, y_vector, ridge=0.0)

                    # Build pruned term list
                    extended_term_asts_pruned = [
                        t for t, k in zip(extended_term_asts, keep_mask.tolist()) if k
                    ]
                    extended_coeffs = extended_coeffs_sparse
                else:
                    # Keep all if nothing selected
                    extended_term_asts_pruned = extended_term_asts
                    extended_coeffs = extended_coeffs_dense

                # Recompute residuals with pruned model
                if keep_mask.sum() > 0:
                    residuals_train = y_vector - X_selected @ extended_coeffs
                else:
                    residuals_train = y_vector - X_matrix @ extended_coeffs
                rms_train = float(residuals_train.square().mean().sqrt().item())

                # Check condition number for degeneracy detection
                X_for_cond = X_selected if keep_mask.sum() > 0 else X_matrix
                cond_num, is_degenerate = _check_condition_number(
                    X_for_cond, extended_term_asts_pruned, threshold=1e8, verbose=verbose
                )

            # Compute validation RMS with pruned model
            rms_val = None
            X_val_matrix = None
            y_val_vector = None
            if val_dataloader is not None:
                all_x_val = []
                for batch in val_dataloader:
                    if isinstance(batch, (tuple, list)):
                        x = batch[0]
                    else:
                        x = batch
                    all_x_val.append(x.to(device))

                X_val = torch.cat(all_x_val, dim=0)

                # Create adaptor with pruned terms
                varpro_adaptor_pruned = DEVarProAdaptor(
                    composite_model=None,
                    order=ode_result.order,
                    x_axis=ode_result.x_axis,
                    cache=cache,
                    term_asts=extended_term_asts_pruned,
                    lambda_reg=ridge,
                ).to(device)

                cache.reset()
                with torch.no_grad():
                    # IMPORTANT: evaluate validation residuals using the *training* coefficients
                    # (no refit on validation).
                    beta_fixed = extended_coeffs.to(device=X_val.device, dtype=X_val.dtype).view(-1)
                    result_val = varpro_adaptor_pruned(X_val, beta_override=beta_fixed)
                    residuals_val = result_val["residuals"]
                    rms_val = float(residuals_val.square().mean().sqrt().item())

                    # Get validation feature matrix + target for optional support minimization
                    X_val_matrix = result_val["X"]  # (M, K)
                    y_val_vector = result_val["y"]  # (M,)

            # Support minimization: greedily remove redundant terms (optional)
            # Use the feature matrices from the STLSQ-pruned model
            if enable_support_minimization and keep_mask.sum() > 0:
                if verbose:
                    print(
                        f"  Running support minimization (rms_tol_factor={rms_tol_factor:.2f})..."
                    )

                # Use the selected features after STLSQ
                X_train_for_min = X_selected if keep_mask.sum() > 0 else X_matrix

                # Run greedy minimization
                extended_term_asts_pruned, extended_coeffs = _support_minimization(
                    extended_term_asts_pruned,
                    extended_coeffs,
                    X_train_for_min,
                    y_vector,
                    X_val_matrix,
                    y_val_vector,
                    ridge=0.0,
                    rms_tol_factor=rms_tol_factor,
                    verbose=verbose,
                )

                # Recompute final RMS values after support minimization
                # Build new adaptor with minimized terms
                varpro_adaptor_final = DEVarProAdaptor(
                    composite_model=None,
                    order=ode_result.order,
                    x_axis=ode_result.x_axis,
                    cache=cache,
                    term_asts=extended_term_asts_pruned,
                    lambda_reg=ridge,
                ).to(device)

                cache.reset()
                with torch.no_grad():
                    # Evaluate with fixed coefficients returned by support minimization
                    beta_fixed_train = extended_coeffs.to(
                        device=X_train.device, dtype=X_train.dtype
                    ).view(-1)
                    result_train_final = varpro_adaptor_final(
                        X_train, beta_override=beta_fixed_train
                    )
                    residuals_train_final = result_train_final["residuals"]
                    rms_train = float(residuals_train_final.square().mean().sqrt().item())

                    if val_dataloader is not None:
                        beta_fixed_val = extended_coeffs.to(
                            device=X_val.device, dtype=X_val.dtype
                        ).view(-1)
                        result_val_final = varpro_adaptor_final(X_val, beta_override=beta_fixed_val)
                        residuals_val_final = result_val_final["residuals"]
                        rms_val = float(residuals_val_final.square().mean().sqrt().item())

            # Classify term types for canonical preference scoring
            term_types = [_classify_term_type(t) for t in extended_term_asts_pruned]
            n_const = term_types.count("const")
            n_state = term_types.count("state")
            n_x = term_types.count("x")
            n_mixed = term_types.count("mixed")
            num_terms_pruned = len(extended_term_asts_pruned)

            # Score: validation RMS + complexity penalties + canonical preference
            base_rms = rms_val if rms_val is not None else rms_train
            score = (
                base_rms
                + complexity_penalty * num_terms_pruned
                + x_penalty * n_x
                + mixed_penalty * n_mixed
            )

            if verbose:
                val_rms_str = f"{rms_val:.6e}" if rms_val is not None else "N/A"
                print(f"  RMS train: {rms_train:.6e}, val: {val_rms_str}")
                print(f"  Pruned to {num_terms_pruned} terms (from {len(extended_term_asts)})")
                if n_const + n_state + n_x + n_mixed > 0:
                    print(f"    Types: const={n_const}, state={n_state}, x={n_x}, mixed={n_mixed}")
                print(f"  Condition number: {cond_num:.2e}")
                print(f"  Score: {score:.6e} (vs best {best_score:.6e})")

            # Keep if better
            if score < best_score:
                if verbose:
                    print(f"  → NEW BEST! (improvement: {best_score - score:.6e})")

                best_score = score
                best_result = DESearchResult(
                    order=ode_result.order,
                    x_axis=ode_result.x_axis,
                    term_asts=extended_term_asts_pruned,
                    coeffs=extended_coeffs,
                    rms_train=rms_train,
                    rms_val=rms_val,
                    condition_number=cond_num,
                    residual_ast=build_de_residual_ast(
                        DESearchResult(
                            order=ode_result.order,
                            x_axis=ode_result.x_axis,
                            term_asts=extended_term_asts_pruned,
                            coeffs=extended_coeffs,
                            rms_train=rms_train,
                            rms_val=rms_val,
                            condition_number=cond_num,
                        ),
                        units_spec=getattr(cfg, "units_spec", None) if cfg is not None else None,
                        enforce_units=bool(getattr(cfg, "enforce_units", False))
                        if cfg is not None
                        else False,
                    ),
                    varpro_metadata={
                        "method": "varpro_template_search_phase2",
                        "template_families": template_families,
                        "max_templates": max_templates,
                        "complexity_penalty": complexity_penalty,
                        "baseline_rms_train": ode_result.rms_train,
                        "baseline_rms_val": ode_result.rms_val,
                        "best_score": best_score,
                        "added_template": tmpl.description,
                        "template_params_init": psi_init,
                        "template_params": psi_used,
                        "template_lm_loss": psi_lm_loss,
                        "num_terms_pruned": num_terms_pruned,
                        "num_terms_before_pruning": len(extended_term_asts),
                        "condition_number": cond_num,
                        "is_degenerate": is_degenerate,
                    },
                )

        except Exception as e:
            if verbose:
                print(f"  [ERROR] {e}")
            continue

    if verbose:
        if best_result.varpro_metadata and "added_template" in best_result.varpro_metadata:
            print(
                f"\n✓ Best model includes template: {best_result.varpro_metadata['added_template']}"
            )
            print(f"  Final score: {best_score:.6e}")
        else:
            print("\n✗ No template improved over baseline")
        print("=" * 70)

    return best_result


def varpro_template_search_multi(
    ode_result,  # Union[DESearchResult, DESearchResultMulti]
    surrogates: list[nn.Module],
    train_dataloaders: list,
    val_dataloaders: Optional[list] = None,
    *,
    template_families: list[str] = None,
    cfg: Optional[DESearchConfig] = None,
    device=None,
    max_templates: int = 3,
    complexity_penalty: float = 1e-3,
    prefer_autonomous: bool = False,
    prefer_forced: bool = False,
    enable_support_minimization: bool = False,
    rms_tol_factor: float = 1.05,
    optimize_psi: bool = False,
    psi_lm_epochs: int = 200,
    psi_lm_epochs_min: int = 20,
    psi_lm_nval_patience: int = 50,
    psi_lm_loss_target: Optional[float] = None,
    psi_lm_strategy: str = "direct_solve",
    verbose: bool = True,
):
    """Search over template families for multi-dataset DE discovery with shared nonlinear parameters.

    This function extends the baseline DE (from STLSQ or VarPro linear refinement) by
    searching over parameterized templates (power laws, exponentials, sinusoids) with:
    - **Shared nonlinear parameters ψ** across all datasets (e.g., power law exponent p)
    - **Dataset-specific linear coefficients β_d** for each term
    - **Dataset weighting** proportional to number of points

    Parameters
    ----------
    ode_result : DESearchResult or DESearchResultMulti
        Baseline DE from STLSQ or VarPro Phase 1
    surrogates : list of nn.Module
        Frozen neural network surrogates for u(x), one per dataset
    train_dataloaders : list of DataLoader
        Training data loaders, one per dataset
    val_dataloaders : list of DataLoader, optional
        Validation data loaders, one per dataset
    template_families : list of str, optional
        Template families to search ('power', 'exp', 'sin', 'rational')
        If None, searches all available families
    cfg : DESearchConfig, optional
        DE search configuration
    device : torch.device, optional
        Device for computation
    max_templates : int
        Maximum number of template terms to try per family
    complexity_penalty : float
        Penalty per additional term in scoring
    prefer_autonomous : bool
        Penalty for non-autonomous (x-dependent) terms
    prefer_forced : bool
        Penalty for autonomous (state-only) terms
    enable_support_minimization : bool
        Apply greedy term removal after template addition
    rms_tol_factor : float
        Tolerance factor for support minimization (keep RMS < baseline * factor)
    optimize_psi : bool
        If True, optimize nonlinear parameters ψ via LM (jointly across datasets)
        If False, use heuristic initialization only
    psi_lm_epochs : int
        Maximum LM epochs for ψ optimization
    psi_lm_epochs_min : int
        Minimum LM epochs before early stopping
    psi_lm_nval_patience : int
        Patience for LM early stopping
    psi_lm_loss_target : float, optional
        Target loss for LM early stopping
    psi_lm_strategy : str
        LM strategy for ψ optimization
    verbose : bool
        Print progress

    Returns
    -------
    DESearchResultMulti
        Best DE with shared terms (baseline + templates), per-dataset coefficients

    Notes
    -----
    **Algorithm**:
    1. Use dataset 0 for template heuristic initialization
    2. For each template candidate:
       a. Initialize ψ heuristically
       b. **If optimize_psi=True**: Run _lm_optimize_template_params_multi() with weighted loss
       c. Substitute ψ into template AST to get fixed term
       d. Build extended library: baseline + template
       e. Build per-dataset design matrices
       f. Run _group_stlsq_multi() for shared support
       g. **If enable_support_minimization**: Run _support_minimization_multi()
       h. Score: weighted_mean(RMS_val) + complexity_penalty + preference_penalties
    3. Select best candidate (including baseline fallback)
    4. Return DESearchResultMulti with shared terms, (D, K) coefficient matrix

    **Dataset Weighting**: Datasets are weighted by size (number of points) to prevent
    bias toward larger datasets. A 2000-point dataset gets 2x weight of a 1000-point dataset.

    Example
    -------
    >>> # After multi-dataset STLSQ discovery
    >>> result_multi = varpro_template_search_multi(
    ...     ode_result_multi, surrogates, train_dls, val_dls,
    ...     template_families=['power', 'exp'],
    ...     optimize_psi=True,
    ...     enable_support_minimization=True
    ... )
    >>> # result_multi.term_asts includes baseline + discovered template(s)
    >>> # result_multi.coeffs is (D, K) with per-dataset coefficients
    """
    from nestynet_sr.sr_de.de_search import DESearchResultMulti

    # Extract baseline terms and metadata
    if isinstance(ode_result, DESearchResultMulti):
        baseline_terms = ode_result.term_asts
        order = ode_result.order
        x_axis = ode_result.x_axis
        baseline_rms_train = ode_result.rms_train
        baseline_rms_val = ode_result.rms_val
        dataset_ids = ode_result.dataset_ids
    else:
        # Single-dataset result - should not happen, but handle gracefully
        baseline_terms = ode_result.term_asts
        order = ode_result.order
        x_axis = ode_result.x_axis
        baseline_rms_train = [ode_result.rms_train]
        baseline_rms_val = [ode_result.rms_val] if ode_result.rms_val is not None else None
        dataset_ids = None

    D = len(surrogates)

    if verbose:
        print("\n" + "=" * 70)
        print("VARPRO TEMPLATE SEARCH (MULTI-DATASET)")
        print("=" * 70)
        print(f"Baseline: {len(baseline_terms)} terms, order {order}")
        print(f"Datasets: {D}")
        print(f"Template families: {template_families or 'all'}")
        print(f"Optimize ψ: {optimize_psi}")
        print(f"Support minimization: {enable_support_minimization}")

    # Setup device
    if device is None:
        device = next(surrogates[0].parameters()).device

    # Extract config parameters
    ridge = getattr(cfg, "ridge", 1e-10) if cfg else 1e-10
    stlsq_lambda = getattr(cfg, "stlsq_lambda", 0.01) if cfg else 0.01
    stlsq_max_iter = getattr(cfg, "stlsq_max_iter", 20) if cfg else 20
    units_spec = getattr(cfg, "units_spec", None) if cfg is not None else None
    enforce_units = bool(getattr(cfg, "enforce_units", False)) if cfg is not None else False

    # Compute dataset weights
    X_train_list = [_gather_all_x(train_dataloaders[d], device) for d in range(D)]
    weights = _compute_dataset_weights(X_train_list)

    # Gather training targets (anchor residuals) per dataset
    y_train_list = []
    for d in range(D):
        cache = UFeatureCache(surrogates[d])
        X_d = X_train_list[d]

        if order == 1:
            cache.ensure(X_d, need_grad=True, need_hess=False)
            anchor = cache.g[:, 0, x_axis]
        else:
            cache.ensure(X_d, need_grad=False, need_hess=True)
            anchor = cache.H[:, 0, x_axis, x_axis]

        y_train_list.append(-anchor)

    # Gather validation data if available
    X_val_list = None
    y_val_list = None
    if val_dataloaders is not None:
        X_val_list = [_gather_all_x(val_dataloaders[d], device) for d in range(D)]
        y_val_list = []
        for d in range(D):
            cache = UFeatureCache(surrogates[d])
            X_d = X_val_list[d]

            if order == 1:
                cache.ensure(X_d, need_grad=True, need_hess=False)
                anchor = cache.g[:, 0, x_axis]
            else:
                cache.ensure(X_d, need_grad=False, need_hess=True)
                anchor = cache.H[:, 0, x_axis, x_axis]

            y_val_list.append(-anchor)

    # Initialize template registry
    if template_families is None:
        template_families = list(TEMPLATE_REGISTRY.keys())

    # Score baseline (no template)
    def compute_baseline_score():
        # Baseline RMS is already computed
        rms_score = (
            sum(baseline_rms_val[d] * weights[d] for d in range(D))
            if baseline_rms_val
            else sum(baseline_rms_train[d] * weights[d] for d in range(D))
        )
        complexity = len(baseline_terms) * complexity_penalty
        return float(rms_score + complexity)

    best_score = compute_baseline_score()
    best_terms = baseline_terms
    best_coeffs_list = None  # Will use baseline coeffs if no template improves
    best_rms_train = baseline_rms_train
    best_rms_val = baseline_rms_val
    best_metadata = {"method": "varpro_template_search_multi", "baseline_only": True}

    if verbose:
        print(f"\nBaseline score: {best_score:.6e}")
        print(
            f"  Weighted RMS (val): {sum(baseline_rms_val[d] * weights[d] for d in range(D)) if baseline_rms_val else 'N/A':.6e}"
        )
        print(f"  Complexity: {len(baseline_terms)} terms")

    # Try each template family
    trial_count = 0
    for family_name in template_families:
        if family_name not in TEMPLATE_REGISTRY:
            if verbose:
                print(f"\n[WARNING] Unknown template family: {family_name}")
            continue

        template_class = TEMPLATE_REGISTRY[family_name]

        # Generate template instances (use dataset 0 for heuristic init)
        try:
            # Determine available x variables
            Nx = X_train_list[0].shape[1]
            x_vars = list(range(Nx))

            # Build template instances
            templates = template_class.build_instances(
                x_vars=x_vars, include_u=True, include_du=(order >= 1), x_axis=x_axis
            )

            # Limit number of templates if specified
            if max_templates is not None and len(templates) > max_templates:
                templates = templates[:max_templates]
        except Exception as e:
            if verbose:
                print(f"\n[ERROR] Failed to generate templates for {family_name}: {e}")
            continue

        for tmpl_idx, template in enumerate(templates):
            trial_count += 1

            if verbose:
                print(f"\n[Trial {trial_count}] {template.description}")

            try:
                # Step 0: Initialize template parameters from data (dataset 0)
                cache_0 = UFeatureCache(surrogates[0])
                X_0 = X_train_list[0]

                if order == 1:
                    cache_0.ensure(X_0, need_grad=True, need_hess=False)
                    du_0 = cache_0.g[:, 0, x_axis]
                else:
                    cache_0.ensure(X_0, need_grad=False, need_hess=True)
                    du_0 = None

                u_0 = cache_0.u[:, 0]  # Fixed: use .u instead of .f
                target_0 = y_train_list[0]

                psi = template_class.init_params(
                    instance=template,
                    x=X_0,
                    u=u_0.unsqueeze(1) if u_0.dim() == 1 else u_0,
                    du=du_0.unsqueeze(1) if du_0 is not None and du_0.dim() == 1 else du_0,
                    target=target_0,
                )

                if verbose:
                    print(f"  Initialized ψ: {psi}")

                # Fast units gate before optional ψ-LM.
                if units_spec is not None and enforce_units:
                    template_ast_init = _substitute_free_consts(template.ast, psi)
                    ok_units, why_units = term_units_feasible(
                        template_ast_init,
                        order=order,
                        x_axis=x_axis,
                        units_spec=units_spec,
                        enforce_units=enforce_units,
                    )
                    if not ok_units:
                        if verbose:
                            print(f"  [SKIP][Units] {template.description}: {why_units}")
                        continue

                # Step 1: Optimize ψ if requested
                if optimize_psi and len(psi) > 0:
                    if verbose:
                        print(f"  Optimizing ψ jointly across {D} datasets...")

                    psi, lm_loss = _lm_optimize_template_params_multi(
                        baseline_term_asts=baseline_terms,
                        template=template,
                        surrogates=surrogates,
                        X_train_list=X_train_list,
                        y_train_list=y_train_list,
                        order=order,
                        x_axis=x_axis,
                        device=device,
                        ridge=ridge,
                        lm_epochs=psi_lm_epochs,
                        lm_epochs_min=psi_lm_epochs_min,
                        lm_nval_patience=psi_lm_nval_patience,
                        lm_loss_target=psi_lm_loss_target,
                        lm_strategy=psi_lm_strategy,
                        verbose=False,
                    )

                    if verbose:
                        print(f"  Optimized ψ: {psi}, loss={lm_loss:.6e}")

                # Step 2: Substitute ψ into template AST
                template_ast = _substitute_free_consts(template.ast, psi)

                if units_spec is not None:
                    ok_units, why_units = term_units_feasible(
                        template_ast,
                        order=order,
                        x_axis=x_axis,
                        units_spec=units_spec,
                        enforce_units=enforce_units,
                    )
                    if not ok_units:
                        if verbose:
                            print(f"  [SKIP][Units] {template.description}: {why_units}")
                        continue

                # Step 3: Build extended library
                extended_terms = baseline_terms + [template_ast]

                # Step 4: Run group-sparse STLSQ
                coeffs_list, keep_mask = _group_stlsq_multi(
                    X_list=X_train_list,
                    y_list=y_train_list,
                    term_asts=extended_terms,
                    surrogates=surrogates,
                    order=order,
                    x_axis=x_axis,
                    ridge=ridge,
                    lam=stlsq_lambda,
                    max_iter=stlsq_max_iter,
                    device=device,
                )

                # Extract selected terms
                selected_terms = [t for t, keep in zip(extended_terms, keep_mask) if keep]
                selected_coeffs_list = coeffs_list

                # Step 5: Support minimization (optional)
                if enable_support_minimization:
                    if verbose:
                        print("  Running support minimization...")

                    selected_terms, selected_coeffs_list, rms_train_ds, rms_val_ds = (
                        _support_minimization_multi(
                            X_train_list=X_train_list,
                            y_train_list=y_train_list,
                            X_val_list=X_val_list,
                            y_val_list=y_val_list,
                            term_asts=selected_terms,
                            surrogates=surrogates,
                            order=order,
                            x_axis=x_axis,
                            ridge=ridge,
                            rms_baseline_list=baseline_rms_train,
                            rms_tol_factor=rms_tol_factor,
                            device=device,
                        )
                    )
                else:
                    # Compute RMS without minimization
                    rms_train_ds = []
                    rms_val_ds = [] if X_val_list is not None else None

                    for d in range(D):
                        cache_train = UFeatureCache(surrogates[d])
                        cols_train = []
                        for term in selected_terms:
                            if term is None:
                                cols_train.append(
                                    torch.ones(X_train_list[d].shape[0], device=device)
                                )
                            else:
                                col_val = _eval_ast(term, X_train_list[d], cache_train)
                                cols_train.append(_as_N(col_val))
                        Phi_train = torch.stack(cols_train, dim=1)

                        residuals_train = Phi_train @ selected_coeffs_list[d] - y_train_list[d]
                        rms_train_ds.append(float(residuals_train.square().mean().sqrt().item()))

                        if rms_val_ds is not None:
                            cache_val = UFeatureCache(surrogates[d])
                            cols_val = []
                            for term in selected_terms:
                                if term is None:
                                    cols_val.append(
                                        torch.ones(X_val_list[d].shape[0], device=device)
                                    )
                                else:
                                    col_val = _eval_ast(term, X_val_list[d], cache_val)
                                    cols_val.append(_as_N(col_val))
                            Phi_val = torch.stack(cols_val, dim=1)

                            residuals_val = Phi_val @ selected_coeffs_list[d] - y_val_list[d]
                            rms_val_ds.append(float(residuals_val.square().mean().sqrt().item()))

                # Step 6: Score with weighted RMS
                rms_tensor = torch.tensor(
                    rms_val_ds if rms_val_ds else rms_train_ds, dtype=torch.float32
                )
                rms_weighted = float((rms_tensor * weights).sum().item())

                complexity = len(selected_terms) * complexity_penalty

                # Preference penalties
                pref_penalty = 0.0
                for term in selected_terms:
                    term_type = _classify_term_type(term)
                    if prefer_autonomous and term_type in ("x", "mixed"):
                        pref_penalty += complexity_penalty
                    if prefer_forced and term_type == "state":
                        pref_penalty += complexity_penalty

                score = rms_weighted + complexity + pref_penalty

                if verbose:
                    print(
                        f"  Score: {score:.6e} (RMS={rms_weighted:.6e}, complexity={complexity:.6e}, pref={pref_penalty:.6e})"
                    )
                    print(
                        f"  Terms: {len(selected_terms)}, RMS per dataset: {[f'{r:.4e}' for r in (rms_val_ds or rms_train_ds)]}"
                    )

                # Update best if improved
                if score < best_score:
                    best_score = score
                    best_terms = selected_terms
                    best_coeffs_list = selected_coeffs_list
                    best_rms_train = rms_train_ds
                    best_rms_val = rms_val_ds
                    best_metadata = {
                        "method": "varpro_template_search_multi",
                        "added_template": template.description,
                        "template_family": family_name,
                        "psi_optimized": psi if optimize_psi else None,
                        "support_minimization": enable_support_minimization,
                        "score": float(score),
                        "baseline_only": False,
                    }

                    if verbose:
                        print("  ✓ New best!")

            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {e}")
                import traceback

                if verbose:
                    traceback.print_exc()
                continue

    # Build final result
    if best_coeffs_list is None:
        # Fallback: use baseline (refit if needed)
        if verbose:
            print("\n[WARNING] No template improved; using baseline")

        # Extract baseline coefficients from original result
        if isinstance(ode_result, DESearchResultMulti):
            best_coeffs_list = [ode_result.coeffs[d] for d in range(D)]
        else:
            best_coeffs_list = [ode_result.coeffs for _ in range(D)]

    # Stack coefficients
    coeffs_matrix = torch.stack([c.clone() for c in best_coeffs_list], dim=0)  # (D, K)

    # Build residual ASTs per dataset
    residual_asts = []
    for d in range(D):
        tmp_result = DESearchResult(
            order=order,
            x_axis=x_axis,
            term_asts=best_terms,
            coeffs=best_coeffs_list[d],
            rms_train=best_rms_train[d],
            rms_val=best_rms_val[d] if best_rms_val else None,
        )
        residual_asts.append(
            build_de_residual_ast(
                tmp_result,
                units_spec=getattr(cfg, "units_spec", None) if cfg is not None else None,
                enforce_units=bool(getattr(cfg, "enforce_units", False))
                if cfg is not None
                else False,
            )
        )

    result_multi = DESearchResultMulti(
        order=order,
        x_axis=x_axis,
        term_asts=best_terms,
        coeffs=coeffs_matrix,
        rms_train=best_rms_train,
        rms_val=best_rms_val,
        dataset_ids=dataset_ids,
        residual_asts=residual_asts,
        term_sources=getattr(ode_result, "term_sources", None),
        prolongation_metadata=getattr(ode_result, "prolongation_metadata", None),
    )
    for _attr in ("expr_ir_report", "expr_ir_reports_by_order"):
        if hasattr(ode_result, _attr):
            setattr(result_multi, _attr, getattr(ode_result, _attr))

    if verbose:
        if not best_metadata.get("baseline_only"):
            print(f"\n✓ Best model includes template: {best_metadata['added_template']}")
            print(f"  Final score: {best_score:.6e}")
        else:
            print("\n✗ No template improved over baseline")
        print("=" * 70)

    return result_multi
