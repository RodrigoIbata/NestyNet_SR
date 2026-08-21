# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Structural, invariance, compound-feature, and affine candidate builders."""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from nestynet_sr.sr_core.atoms import PlanckLeaf
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    CosNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    clone_ast,
    effective_arity,
    get_input_exprs,
    is_trivial_input,
)
from nestynet_sr.sr_core.constants import (
    build_scalar_atom_from_variant as _build_scalar_atom_from_variant,
)

from ._candidate_builders_common import (
    _build_atom_input_tensor,
    _eval_input_expr_value,
    _find_matching_core,
    _gather_atom_teacher_data,
    _replace_node,
)
from .fitting_utils import _fit_planck_tail_discrete_power, _gather_teacher_data_1d

def _build_ratio_invariance_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    xi_local_idx: int,
    xj_local_idx: int,
    degree: int = 2,
    exponent: float = -0.5,
    min_points: int = 400,
    rel_rms_threshold: float = 0.02,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Build a candidate where a bivariate NN leaf is replaced by a polynomial in
    the ratio of two effective inputs, composed with a power: (poly(r))^exponent.

    This is useful for homogeneous degree-0 functions like the Lorentz factor:
        G(x1, x2) = 1/sqrt(1 - (x2/x1)²)

    Here G^(-2) = 1 - r² is a simple polynomial in r, so:
        G = (1 - r²)^(-0.5)

    Parameters
    ----------
    root : Node
        The current AST root.
    target : AtomNode
        The NN atom to replace.
    reuse : dict
        Tag -> module mapping for teacher extraction.
    train_loader : DataLoader
        Training data loader.
    device, dtype : torch.device, torch.dtype
        Device and dtype for computation.
    xi_local_idx : int
        Local effective-input index of the denominator in the ratio r = xj/xi.
    xj_local_idx : int
        Local effective-input index of the numerator in the ratio r = xj/xi.
    degree : int
        Polynomial degree for fitting.
    exponent : float
        Exponent to apply to the polynomial (e.g., -0.5 for 1/sqrt).
    min_points : int
        Minimum data points for fitting.
    rel_rms_threshold : float
        Maximum relative RMS error for accepting the fit.

    Returns
    -------
    cand_root : Node | None
        The candidate AST with the replacement, or None if rejected.
    custom_init : callable | None
        Initialization callback, or None.
    """
    from nestynet_sr.sr_core.atoms import RatioPolyLeaf

    from .stageB import _collect_all_atoms

    if target.kind.lower() != "nn":
        return None, None

    input_exprs = get_input_exprs(target)
    dim = int(effective_arity(target))
    if dim != 2 or len(input_exprs) != 2:
        return None, None

    if not (0 <= int(xi_local_idx) < dim) or not (0 <= int(xj_local_idx) < dim):
        return None, None
    if int(xi_local_idx) == int(xj_local_idx):
        return None, None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None, None
    teacher = reuse[tag]

    # Gather teacher data
    data = _gather_atom_teacher_data(
        train_loader=train_loader,
        atom=target,
        teacher=teacher,
        device=device,
        dtype=dtype,
        max_points=5000,
    )
    if data is None:
        return None, None

    X, F = data  # X: [N, 2], F: [N]
    X = X.to(dtype=torch.float64)
    F = F.to(dtype=torch.float64).view(-1)
    N, dim2 = X.shape
    if N < min_points or dim2 != dim:
        return None, None

    x_i = X[:, int(xi_local_idx)]  # denominator
    x_j = X[:, int(xj_local_idx)]  # numerator

    # Filter out points where denominator is too small
    eps = 1e-8
    mask = x_i.abs() > eps
    if mask.sum() < min_points:
        return None, None
    x_i = x_i[mask]
    x_j = x_j[mask]
    F = F[mask]

    # Compute ratio
    r = x_j / x_i  # [N]

    # We want to fit: F^(1/exponent) = poly(r)
    # For exponent=-0.5: F^(-2) = poly(r)
    inv_exp = 1.0 / exponent
    try:
        # Handle sign: F^(-2) should be positive if F > 0
        F_sign_mask = F.abs() > eps
        if F_sign_mask.sum() < min_points:
            return None, None
        F_valid = F[F_sign_mask].abs()
        r_valid = r[F_sign_mask]

        F_transformed = F_valid.pow(inv_exp)  # [N]
    except Exception:
        return None, None

    # Check for invalid values
    if not torch.isfinite(F_transformed).all():
        return None, None

    # Build Vandermonde matrix for polynomial in r
    # poly(r) = c0 + c1*r + c2*r² + ...
    powers = torch.arange(degree + 1, dtype=torch.float64, device=r_valid.device)
    Phi = r_valid.unsqueeze(1).pow(powers)  # [N, degree+1]

    # Solve least squares: Phi @ coeffs = F_transformed
    try:
        coeffs = torch.linalg.lstsq(Phi, F_transformed.unsqueeze(1)).solution.squeeze()
    except Exception:
        return None, None

    if not torch.isfinite(coeffs).all():
        return None, None

    # Evaluate fit quality
    F_fit_transformed = (Phi @ coeffs)
    resid = F_fit_transformed - F_transformed
    rms = float(torch.sqrt(torch.mean(resid ** 2)).item())
    scale = float(F_transformed.std(unbiased=False).item())
    rel_rms = rms / max(scale, eps)

    if rel_rms > rel_rms_threshold:
        print(
            f"[Stage B] ratio_invariance rejected: rel_rms={rel_rms:.4f} > {rel_rms_threshold}, "
            f"vars={tuple(int(i) for i in target.var_idxs)}, "
            f"xi_local={xi_local_idx}, xj_local={xj_local_idx}, deg={degree}, exp={exponent}"
        )
        return None, None

    # Build candidate AST: PowNode(RatioPolyLeaf(num_expr, den_expr), exponent)
    ratio_inputs = (
        clone_ast(input_exprs[int(xj_local_idx)]),
        clone_ast(input_exprs[int(xi_local_idx)]),
    )
    base_tag = getattr(target, "tag", None)
    ratio_tag = (
        f"{base_tag}_RI_{int(xj_local_idx)}_{int(xi_local_idx)}_{int(degree)}"
        if base_tag
        else f"RI_{id(target)}_{int(xj_local_idx)}_{int(xi_local_idx)}_{int(degree)}"
    )
    ratio_atom = AtomNode(
        kind="ratio_poly",
        var_idxs=target.var_idxs,
        kwargs={"degree": int(degree)},
        tag=ratio_tag,
        inputs=ratio_inputs,
    )
    new_subtree = PowNode(base=ratio_atom, exponent=float(exponent))
    cand_root = _replace_node(root, target, new_subtree)

    # Cache coefficients for custom init
    coeffs_cpu = coeffs.detach().cpu()

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """Initialize the RatioPolyLeaf with fitted coefficients."""
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        ratio_core: Optional[RatioPolyLeaf] = None

        for atom_i, leaf_mod in zip(atoms, leaves):
            core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))
            if isinstance(core, RatioPolyLeaf) and getattr(atom_i, "tag", None) == ratio_tag:
                ratio_core = core
                break

        if ratio_core is None:
            print(f"[Stage B custom_init ratio] No RatioPolyLeaf found for tag {ratio_tag}")
            return

        dev = ratio_core.coeffs.device
        dt = ratio_core.coeffs.dtype
        with torch.no_grad():
            n_model = ratio_core.coeffs.numel()
            n_fit = coeffs_cpu.numel()
            if n_fit == n_model:
                ratio_core.coeffs.copy_(coeffs_cpu.to(device=dev, dtype=dt))
            elif n_fit < n_model:
                # Pad with zeros
                ratio_core.coeffs[:n_fit].copy_(coeffs_cpu.to(device=dev, dtype=dt))
                ratio_core.coeffs[n_fit:].zero_()
            else:
                # Truncate
                ratio_core.coeffs.copy_(coeffs_cpu[:n_model].to(device=dev, dtype=dt))

    return cand_root, _custom_init


def _build_homogeneity_peel_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    degree: float,
    power_var_idx: int,
    ratio_var_idx: int,
    nn_hyperparams: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Node], Optional[callable], Optional[Dict[str, Any]]]:
    """
    Build a candidate that peels off a homogeneous power factor.

    When f(xi, xj) is homogeneous of degree k in (xi, xj), we can write:
        f(xi, xj) = xi^k * h(xj/xi)

    This function builds the candidate AST:
        Mul(Pow(Var(power_var_idx), degree), NN(ratio))

    where the NN takes the ratio xj/xi as its compound input.

    Parameters
    ----------
    root : Node
        The current AST root.
    target : AtomNode
        The bivariate NN atom to replace.
    reuse : dict
        Tag -> module mapping for teacher extraction.
    train_loader : DataLoader
        Training data loader.
    device, dtype : torch.device, torch.dtype
        Device and dtype for computation.
    degree : float
        The homogeneous degree k.
    power_var_idx : int
        Index of the variable to raise to power k (denominator in ratio).
    ratio_var_idx : int
        Index of the numerator variable in the ratio.
    nn_hyperparams : dict, optional
        Hyperparameters for the residual NN (num_segments, dual_layer, etc.)

    Returns
    -------
    cand_root : Node | None
        The candidate AST with the replacement, or None if rejected.
    custom_init : callable | None
        Initialization callback, or None.
    metadata : dict | None
        Metadata about the rewrite, or None.
    """
    from nestynet_sr.sr_core.bridges import AtomNode, MulNode, PowNode
    from nestynet_sr.sr_core.separability_math import build_monomial_ast

    if target.kind.lower() != "nn":
        return None, None, None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) != 2:
        return None, None, None

    # Ensure both indices are in the target's var_idxs
    if power_var_idx not in var_idxs or ratio_var_idx not in var_idxs:
        return None, None, None

    # Build the ratio AST: r = ratio_var / power_var
    # This is ratio_var^1 * power_var^(-1) = ratio_var / power_var
    ratio_ast = build_monomial_ast(
        var_idxs=(ratio_var_idx, power_var_idx),
        exponents=(1, -1),
    )

    # Build the NN atom with compound input (ratio)
    nn_kwargs: Dict[str, Any] = {}
    if nn_hyperparams:
        nn_kwargs.update(nn_hyperparams)
    # Copy relevant hyperparams from original target if present
    if target.kwargs:
        for key in ("num_segments", "dual_layer", "seg_width"):
            if key in target.kwargs and key not in nn_kwargs:
                nn_kwargs[key] = target.kwargs[key]

    # IMPORTANT: Stage B does not auto-tag newly created NN atoms. If we leave
    # this untagged, teacher-based univariate rewrites (sqrt_poly, ratpoly_1d,
    # planck, powexp, ...) cannot see it after the rewrite is accepted because
    # it won't appear in the state.reuse map.
    base_tag = getattr(target, "tag", None)
    new_tag = (
        f"{base_tag}_H_{power_var_idx}_{ratio_var_idx}"
        if base_tag
        else f"H_{power_var_idx}_{ratio_var_idx}"
    )

    compound_nn = AtomNode(
        kind="nn",
        var_idxs=var_idxs,  # Keep both variables for proper data routing
        kwargs=nn_kwargs,
        tag=new_tag,
        inputs=(ratio_ast,),
    )

    # Build the power factor: power_var^degree
    # Use a Var atom (or a simple PowNode on Var)
    from nestynet_sr.sr_core.bridges import Var

    power_factor: Node
    if abs(degree - 1.0) < 1e-9:
        # Degree 1: just use Var directly
        power_factor = Var(power_var_idx)
    else:
        power_factor = PowNode(base=Var(power_var_idx), exponent=float(degree))

    # Build the full candidate: power_var^k * NN(ratio)
    new_subtree = MulNode(left=power_factor, right=compound_nn)

    # Replace target in the root
    cand_root = _replace_node(root, target, new_subtree)

    # Metadata for logging and debugging
    metadata = {
        "pattern": "homogeneity_peel",
        "degree": degree,
        "power_var_idx": power_var_idx,
        "ratio_var_idx": ratio_var_idx,
        "structural": True,
    }

    # No special init needed - the NN will be trained from scratch
    return cand_root, None, metadata


def _build_product_homogeneity_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    degree: float,
    power_var_idx: int,
    product_var_idx: int,
    nn_hyperparams: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Node], Optional[callable], Optional[Dict[str, Any]]]:
    """
    Build a candidate that peels off a power factor with product compound.

    When f(xi, xj) = xi^k * h(xi*xj), we can write the residual as a
    function of the product compound. This is useful for patterns like:
        f(x, z) = x * tanh(x*z)

    This function builds the candidate AST:
        Mul(Pow(Var(power_var_idx), degree), NN(product))

    where the NN takes the product xi*xj as its compound input.

    Parameters
    ----------
    root : Node
        The current AST root.
    target : AtomNode
        The bivariate NN atom to replace.
    reuse : dict
        Tag -> module mapping for teacher extraction.
    train_loader : DataLoader
        Training data loader.
    device, dtype : torch.device, torch.dtype
        Device and dtype for computation.
    degree : float
        The homogeneous degree k for the outer power factor.
    power_var_idx : int
        Index of the variable to raise to power k.
    product_var_idx : int
        Index of the other variable in the product.
    nn_hyperparams : dict, optional
        Hyperparameters for the residual NN.

    Returns
    -------
    cand_root : Node | None
        The candidate AST with the replacement, or None if rejected.
    custom_init : callable | None
        Initialization callback, or None.
    metadata : dict | None
        Metadata about the rewrite, or None.
    """
    from nestynet_sr.sr_core.bridges import AtomNode, MulNode, PowNode
    from nestynet_sr.sr_core.separability_math import build_monomial_ast

    if target.kind.lower() != "nn":
        return None, None, None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    if len(var_idxs) != 2:
        return None, None, None

    # Ensure both indices are in the target's var_idxs
    if power_var_idx not in var_idxs or product_var_idx not in var_idxs:
        return None, None, None

    # Build the product AST: w = power_var * product_var
    # This is power_var^1 * product_var^1
    product_ast = build_monomial_ast(
        var_idxs=(power_var_idx, product_var_idx),
        exponents=(1, 1),
    )

    # Build the NN atom with compound input (product)
    nn_kwargs: Dict[str, Any] = {}
    if nn_hyperparams:
        nn_kwargs.update(nn_hyperparams)
    # Copy relevant hyperparams from original target if present
    if target.kwargs:
        for key in ("num_segments", "dual_layer", "seg_width"):
            if key in target.kwargs and key not in nn_kwargs:
                nn_kwargs[key] = target.kwargs[key]

    # IMPORTANT: Stage B does not auto-tag newly created NN atoms. Keep the new
    # residual NN tagged so it becomes visible to later teacher-based 1D
    # rewrites (sqrt_poly, ratpoly_1d, powexp, ...).
    base_tag = getattr(target, "tag", None)
    new_tag = (
        f"{base_tag}_PH_{power_var_idx}_{product_var_idx}"
        if base_tag
        else f"PH_{power_var_idx}_{product_var_idx}"
    )

    compound_nn = AtomNode(
        kind="nn",
        var_idxs=var_idxs,  # Keep both variables for proper data routing
        kwargs=nn_kwargs,
        tag=new_tag,
        inputs=(product_ast,),
    )

    # Build the power factor: power_var^degree
    from nestynet_sr.sr_core.bridges import Var

    power_factor: Node
    if abs(degree - 1.0) < 1e-9:
        # Degree 1: just use Var directly
        power_factor = Var(power_var_idx)
    else:
        power_factor = PowNode(base=Var(power_var_idx), exponent=float(degree))

    # Build the full candidate: power_var^k * NN(product)
    new_subtree = MulNode(left=power_factor, right=compound_nn)

    # Replace target in the root
    cand_root = _replace_node(root, target, new_subtree)

    # Metadata for logging and debugging
    metadata = {
        "pattern": "product_homogeneity",
        "degree": degree,
        "power_var_idx": power_var_idx,
        "product_var_idx": product_var_idx,
        "structural": True,
    }

    # No special init needed - the NN will be trained from scratch
    return cand_root, None, metadata


def _build_coupled_ratio_candidate(
    root: Node,
    target_F: AtomNode,
    target_G: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    poly_form: str,
    poly_coeffs: Dict[str, float],
    min_points: int = 400,
) -> Tuple[Optional[Node], Optional[callable]]:
    """
    Build a candidate where F is replaced by poly * G when F/G is a simple polynomial.

    This is useful when two NN leaves have a simple ratio relationship, e.g.:
        F(x1, x2) = (-x2/x1²) * G(x1, x2)

    In this case, we can factor out the polynomial and keep only one NN leaf.

    Parameters
    ----------
    root : Node
        The current AST root.
    target_F : AtomNode
        The NN atom F to be replaced by poly * G.
    target_G : AtomNode
        The reference NN atom G (kept as-is, but referenced in the replacement).
    reuse : dict
        Tag -> module mapping for teacher extraction.
    train_loader : DataLoader
        Training data loader.
    device, dtype : torch.device, torch.dtype
        Device and dtype for computation.
    poly_form : str
        Description of the polynomial form, e.g., "a*xj/xi2" for a*xj/xi².
    poly_coeffs : dict
        Fitted polynomial coefficients, e.g., {"a": -1.0} for -x2/x1².
    min_points : int
        Minimum data points required.

    Returns
    -------
    cand_root : Node | None
        The candidate AST with F replaced by poly * G, or None if rejected.
    custom_init : callable | None
        Initialization callback, or None.

    Notes
    -----
    The polynomial multiplier is constructed using existing AST nodes (Mul, Pow, etc.)
    rather than introducing new leaf types. This keeps the AST simpler.
    """
    # Validate inputs
    if target_F.kind.lower() != "nn" or target_G.kind.lower() != "nn":
        return None, None

    F_vars = set(int(i) for i in target_F.var_idxs)
    G_vars = set(int(i) for i in target_G.var_idxs)
    shared_vars = F_vars & G_vars
    if not shared_vars:
        return None, None

    # Build the polynomial multiplier AST based on poly_form
    # Common forms from check_coupled_leaf_ratio_from_derivs:
    # - "const": constant factor
    # - "a*xj/xi2": a * xj / xi²
    # - "a+b*xj/xi2": a + b * xj / xi²
    # - "a*xj/xi": a * xj / xi
    # - "a+b*xj/xi": a + b * xj / xi

    poly_multiplier: Optional[Node] = None

    if poly_form == "const":
        # Constant: just use FreeConst
        coeff = poly_coeffs.get("a", 1.0)
        poly_multiplier = _build_scalar_atom_from_variant(
            {"mode": "scale", "name": "ratio_a", "tag": None, "value": float(coeff)}
        )
    elif "xj/xi2" in poly_form:
        # Form: a * xj / xi² or a + b * xj / xi²
        # Need xi and xj indices - they should be in the shared vars
        xi_idx = poly_coeffs.get("xi_idx")
        xj_idx = poly_coeffs.get("xj_idx")
        if xi_idx is None or xj_idx is None:
            print("[Stage B] coupled_ratio: missing xi_idx or xj_idx in poly_coeffs")
            return None, None

        # Build: xj / xi² = xj * xi^(-2)
        # AtomNode for xj (VarLeaf extracts x[:, xj])
        xj_var = AtomNode(kind="var", var_idxs=(int(xj_idx),), kwargs={}, tag=None)
        # PowNode for xi^(-2)
        xi_var = AtomNode(kind="var", var_idxs=(int(xi_idx),), kwargs={}, tag=None)
        xi_pow_minus2 = PowNode(base=xi_var, exponent=-2.0)
        # xj * xi^(-2)
        ratio_term = MulNode(left=xj_var, right=xi_pow_minus2)

        if poly_form == "a*xj/xi2":
            # Just a * (xj/xi²)
            coeff_a = poly_coeffs.get("a", 1.0)
            coeff_node = _build_scalar_atom_from_variant(
                {"mode": "scale", "name": "ratio_a", "tag": None, "value": float(coeff_a)}
            )
            poly_multiplier = MulNode(left=coeff_node, right=ratio_term)
        elif poly_form == "a+b*xj/xi2":
            # a + b * (xj/xi²)
            coeff_a = poly_coeffs.get("a", 0.0)
            coeff_b = poly_coeffs.get("b", 1.0)
            const_node = _build_scalar_atom_from_variant(
                {"mode": "scale", "name": "ratio_a", "tag": None, "value": float(coeff_a)}
            )
            b_node = _build_scalar_atom_from_variant(
                {"mode": "scale", "name": "ratio_b", "tag": None, "value": float(coeff_b)}
            )
            b_ratio = MulNode(left=b_node, right=ratio_term)
            poly_multiplier = AddNode(left=const_node, right=b_ratio)
    elif "xj/xi" in poly_form and "xi2" not in poly_form:
        # Form: a * xj / xi or a + b * xj / xi
        xi_idx = poly_coeffs.get("xi_idx")
        xj_idx = poly_coeffs.get("xj_idx")
        if xi_idx is None or xj_idx is None:
            return None, None

        # Build: xj / xi = xj * xi^(-1)
        xj_var = AtomNode(kind="var", var_idxs=(int(xj_idx),), kwargs={}, tag=None)
        xi_var = AtomNode(kind="var", var_idxs=(int(xi_idx),), kwargs={}, tag=None)
        xi_pow_minus1 = PowNode(base=xi_var, exponent=-1.0)
        ratio_term = MulNode(left=xj_var, right=xi_pow_minus1)

        if poly_form == "a*xj/xi":
            coeff_a = poly_coeffs.get("a", 1.0)
            coeff_node = _build_scalar_atom_from_variant(
                {"mode": "scale", "name": "ratio_a", "tag": None, "value": float(coeff_a)}
            )
            poly_multiplier = MulNode(left=coeff_node, right=ratio_term)
        elif poly_form == "a+b*xj/xi":
            coeff_a = poly_coeffs.get("a", 0.0)
            coeff_b = poly_coeffs.get("b", 1.0)
            const_node = _build_scalar_atom_from_variant(
                {"mode": "scale", "name": "ratio_a", "tag": None, "value": float(coeff_a)}
            )
            b_node = _build_scalar_atom_from_variant(
                {"mode": "scale", "name": "ratio_b", "tag": None, "value": float(coeff_b)}
            )
            b_ratio = MulNode(left=b_node, right=ratio_term)
            poly_multiplier = AddNode(left=const_node, right=b_ratio)
    else:
        print(f"[Stage B] coupled_ratio: unsupported poly_form '{poly_form}'")
        return None, None

    if poly_multiplier is None:
        return None, None

    # Build the replacement: poly * G (reference to target_G)
    # We need to create a reference to G's leaf. Since G is already in the AST,
    # we can create a new reference to it via its tag.
    G_ref = AtomNode(
        kind=target_G.kind,
        var_idxs=target_G.var_idxs,
        kwargs=dict(target_G.kwargs) if target_G.kwargs else {},
        tag=target_G.tag,  # Same tag to reuse the same leaf
    )

    # new_subtree = poly_multiplier * G
    new_subtree = MulNode(left=poly_multiplier, right=G_ref)

    # Replace F with the new subtree
    cand_root = _replace_node(root, target_F, new_subtree)

    # No special custom_init needed - the FreeConst nodes will be optimized by LM
    return cand_root, None


def _estimate_trig_params_on_compound(
    train_loader,
    reuse: Dict[str, torch.nn.Module],
    target_atom: AtomNode,
    input_expr,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int = 2000,
) -> Optional[Tuple[float, float, float, float]]:
    """Estimate trig parameters (omega, amp, phase, offset) on compound variable z.

    Uses FFT to extract:
    - omega: dominant angular frequency
    - amp: amplitude of the sinusoid
    - phase: phase offset
    - offset: DC offset (mean value)

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader
    reuse : dict
        Map from atom tags to teacher modules
    target_atom : AtomNode
        The target NN atom being replaced
    input_expr : Node
        AST node representing the compound variable expression (e.g., x1*x2)
    device : torch.device
        Device for computation
    dtype : torch.dtype
        Data type for computation
    max_points : int
        Maximum number of points to collect

    Returns
    -------
    tuple or None
        (omega, amp, phase, offset) or None if detection failed
    """
    import numpy as np

    # Get teacher for this atom
    tag = target_atom.tag
    if tag is None or tag not in reuse:
        return None
    teacher = reuse[tag]

    # Gather data (unified for compound atoms)
    data = _gather_teacher_data_1d(
        train_loader, teacher, device, dtype,
        input_expr=input_expr, max_points=max_points
    )
    if data is None:
        return None

    Z, F = data
    z_all = Z.view(-1).numpy()
    f_all = F.view(-1).numpy()

    if len(z_all) < 50:
        return None

    # Sort by z and interpolate to uniform grid for FFT
    order = np.argsort(z_all)
    z_sorted, f_sorted = z_all[order], f_all[order]

    # Remove duplicates by averaging
    z_unique, indices = np.unique(z_sorted, return_inverse=True)
    f_unique = np.zeros_like(z_unique)
    counts = np.zeros_like(z_unique)
    np.add.at(f_unique, indices, f_sorted)
    np.add.at(counts, indices, 1)
    f_unique = f_unique / np.maximum(counts, 1)

    if len(z_unique) < 20:
        return None

    # Interpolate to uniform grid
    n_grid = min(len(z_unique), 512)
    z_uniform = np.linspace(z_unique.min(), z_unique.max(), n_grid)
    f_interp = np.interp(z_uniform, z_unique, f_unique)

    # Extract DC offset (mean)
    offset = float(f_interp.mean())
    f_centered = f_interp - offset

    # FFT to find dominant frequency
    dz = z_uniform[1] - z_uniform[0]
    fft = np.fft.rfft(f_centered)
    freqs = np.fft.rfftfreq(len(f_centered), dz)

    # Skip DC component (index 0) and find peak
    if len(fft) < 2:
        return None
    magnitudes = np.abs(fft[1:])
    peak_idx = np.argmax(magnitudes) + 1  # +1 because we skipped index 0
    omega = 2 * np.pi * freqs[peak_idx]

    # Extract amplitude and phase from the complex FFT coefficient
    coeff = fft[peak_idx]
    amp = 2 * np.abs(coeff) / len(f_centered)  # Factor of 2 for one-sided FFT
    phase = np.angle(coeff)

    # Sanity check: omega should be positive and reasonable
    if omega < 0.1 or omega > 100:
        omega = 1.0  # fallback
    if amp < 1e-8:
        amp = 1.0  # fallback

    return float(omega), float(amp), float(phase), float(offset)


def _estimate_univariate_trig_amplitude(
    train_loader,
    reuse: Dict[str, torch.nn.Module],
    target_atom: AtomNode,
    axis: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int = 2000,
) -> Tuple[float, float]:
    """Estimate amplitude and offset for a simple univariate NN atom.

    For A*sin(wx+phi), std(f) = A/sqrt(2), so A = std(f) * sqrt(2).

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader
    reuse : dict
        Map from atom tags to teacher modules
    target_atom : AtomNode
        The target NN atom being replaced
    axis : int
        The variable axis (column index) this atom operates on
    device : torch.device
        Device for computation
    dtype : torch.dtype
        Data type for computation
    max_points : int
        Maximum number of points to collect

    Returns
    -------
    tuple
        (amplitude, offset). Returns (1.0, 0.0) on failure.
    """
    tag = target_atom.tag
    if tag is None or tag not in reuse:
        return 1.0, 0.0

    teacher = reuse[tag]

    # Gather data (unified for univariate atoms)
    data = _gather_teacher_data_1d(
        train_loader, teacher, device, dtype,
        axis=axis, max_points=max_points
    )
    if data is None:
        return 1.0, 0.0

    _, f_all = data  # Only need F values for amplitude estimation
    f_all = f_all.view(-1)

    if len(f_all) < 50:
        return 1.0, 0.0

    # Estimate offset and amplitude
    offset = float(f_all.mean())
    f_centered = f_all - offset
    std_f = float(f_centered.std())

    # For A*sin(wx+phi), std = A/sqrt(2), so A = std * sqrt(2)
    amp = std_f * math.sqrt(2)

    # Sanity checks
    if not math.isfinite(amp) or amp < 1e-10:
        amp = 1.0
    if not math.isfinite(offset):
        offset = 0.0

    return amp, offset


def _build_planck_derived_feature_candidate(
    root: Node,
    target: AtomNode,
    extra_prod: Node,
    z_expr: Node,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    min_points: int = 400,
    eps: float = 1e-8,
    tail_fraction: float = 0.5,
    max_abs_p: float = 10.0,
    rel_rms_threshold: float = 0.08,
    label: str = "compound_planck",
    signature_extra: Optional[Tuple[int, ...]] = None,
):
    """
    Planck rewrite for compound atoms with derived features.

    For compound atoms like NN[z=(x2*x3), x0, x1], this function proposes:

        f(z, x_extra) ≈ z * A * w^p / (exp(α*w + β) - 1)

    where w = (x0*x1) / z is the derived feature built from the extra vars
    divided by the compound variable z.

    The key insight is that for Planck-like functions, the template
        y = u / (exp(u/c) - 1)
    where u is a product of variables and c is a constant (like 2π), can be
    rewritten using w = u/z as:
        y = z * w / (exp(w/c') - 1)

    Parameters
    ----------
    root : Node
        Current AST root.
    target : AtomNode
        The compound NN atom to replace.
    extra_prod : Node
        AST for the product of extra variables (e.g., x0*x1).
    z_expr : Node
        AST for the compound variable expression (e.g., x2*x3).
    reuse : dict
        Tag-to-module mapping.
    train_loader : DataLoader
        Training data.
    device, dtype : torch device/dtype
    min_points : int
        Minimum number of valid data points required.
    eps : float
        Small value for positivity checks.
    tail_fraction : float
        Fraction of data (high-w region) to use for tail fit.
    max_abs_p : float
        Maximum absolute value of power exponent p.
    rel_rms_threshold : float
        Maximum relative RMS error in log-space fit.

    Returns
    -------
    Candidate or None
        A Stage B candidate if the Planck template fits well.
    """
    from .stageB import _collect_all_atoms
    from .stageB.engine import Candidate, atom_content_hash

    # Get the teacher leaf for this compound atom
    tag = target.tag
    if tag is None or tag not in reuse:
        return None
    teacher = reuse[tag]

    # Gather data: (x_full, y_teacher)
    # We need x_full to evaluate z and extra_prod
    xs: list = []
    ys: list = []
    n_collected = 0
    max_points = 5000

    teacher.eval()
    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            x_full, _ = batch
        else:
            x_full = batch
        x_full = x_full.to(device=device, dtype=dtype)

        # Build input for the compound atom: [z, extras...]
        x_sub = _build_atom_input_tensor(target, x_full)

        with torch.no_grad():
            y = teacher(x_sub)
            if y.dim() == 2:
                y = y[:, 0]
            else:
                y = y.view(-1)

        xs.append(x_full.detach().cpu())
        ys.append(y.detach().cpu())
        n_collected += x_full.size(0)
        if n_collected >= max_points:
            break

    if not xs:
        return None

    X_full = torch.cat(xs, dim=0)[:max_points].to(dtype=torch.float64)
    Y = torch.cat(ys, dim=0)[:max_points].to(dtype=torch.float64).view(-1)

    # Evaluate z = z_expr(x) and extra_prod = extra_prod(x)
    try:
        z_vals = _eval_input_expr_value(z_expr, X_full).view(-1)
        u_vals = _eval_input_expr_value(extra_prod, X_full).view(-1)
    except Exception:
        return None

    # w = u / z (the derived feature)
    w_vals = u_vals / (z_vals + eps)

    # Planck form: Y ≈ z * A * w^p / (exp(α*w) - 1)
    # Normalize by z to get: Y/z ≈ A * w^p / (exp(α*w) - 1)
    Y_norm = Y / (z_vals + eps)

    # Fit Planck tail on (w, Y_norm) using shared helper
    fit_result = _fit_planck_tail_discrete_power(
        w_vals, Y_norm,
        min_points=min_points,
        eps=eps,
        tail_fraction=tail_fraction,
        rel_rms_threshold=rel_rms_threshold,
    )
    if fit_result is None:
        return None

    p_est, a_est, b0, rms_rel = fit_result

    # Build the replacement AST:
    # z * planck(w) where planck has input_expr = extra_prod / z_expr
    #
    # Actually, we need to be careful: the Planck leaf expects a single input
    # that it transforms. We'll create a Planck atom with input_expr = w_expr.

    # w_expr = extra_prod / z_expr = extra_prod * z_expr^(-1)
    # Since there's no DivNode, use MulNode with PowNode(exponent=-1)
    w_expr = MulNode(clone_ast(extra_prod), PowNode(clone_ast(z_expr), -1.0))

    target_var_idxs = tuple(int(v) for v in target.var_idxs)
    planck_inputs = (w_expr,)

    # Create Planck atom with compound input w
    planck_atom = AtomNode(
        kind="planck",
        var_idxs=target_var_idxs,
        kwargs={"p": float(p_est)},
        tag=None,
        inputs=(w_expr,),
    )

    # The AST form is: z * Planck(w) = z * Planck(u/z)
    # We need to create an AST node for z
    new_subtree = MulNode(clone_ast(z_expr), planck_atom)

    cand_root = _replace_node(root, target, new_subtree)

    # Clamp initialization values
    b0_clamped = max(-20.0, min(20.0, b0))
    a_est_clamped = max(1e-4, min(60.0, a_est))
    p_est_clamped = max(-max_abs_p, min(max_abs_p, p_est))

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """Initialize the Planck leaf with fitted parameters."""
        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)
        core_planck = _find_matching_core(
            atoms,
            leaves,
            core_types=PlanckLeaf,
            expected_kind="planck",
            expected_inputs=planck_inputs,
        )

        if core_planck is None:
            return

        with torch.no_grad():
            core_planck.p.copy_(
                torch.as_tensor(
                    p_est_clamped,
                    dtype=core_planck.p.dtype,
                    device=core_planck.p.device,
                )
            )
            core_planck.log_a.copy_(
                torch.log(
                    torch.as_tensor(
                        a_est_clamped,
                        dtype=core_planck.log_a.dtype,
                        device=core_planck.log_a.device,
                    )
                )
            )
            core_planck.log_amp.copy_(
                torch.as_tensor(
                    b0_clamped,
                    dtype=core_planck.log_amp.dtype,
                    device=core_planck.log_amp.device,
                )
            )

        print(
            f"[Stage B custom_init compound_planck] vars={target_var_idxs}, "
            f"p≈{p_est_clamped:.3g}, α≈{a_est_clamped:.3g} (≈1/(2π)={1/(2*math.pi):.3g}), logA≈{b0_clamped:.3g}"
        )

    # Create signature for deduplication.  The Planck rule can try multiple
    # orientations for the same atom, so the prefactor/argument choice must be
    # part of the signature.
    sig = (atom_content_hash(target),)
    if signature_extra is not None:
        sig = sig + tuple(int(v) for v in signature_extra)

    return Candidate(
        label=label,
        root=cand_root,
        init_fn=_custom_init,
        signature=sig,
        meta={
            "structural": True,
            "pattern_family": "compound_planck",
            "min_free_params": 2,
            "log": (
                f"[Stage B]  Trying {label} on NN vars={target_var_idxs} "
                f"p≈{p_est:.3g}, α≈{a_est:.3g}, rms_rel_log≈{rms_rel:.3g}"
            ),
        },
    )


def _build_affine_decomp_candidate(
    root: Node,
    target: AtomNode,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    hit: Dict,
    dataset_hit_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Tuple[Node, Optional[Callable], Dict[str, Any]]]:
    """Build a candidate from an affine decomposition screening hit.

    Given that ``g(f(z, w)) = a(z) + b(z) * h(w)``, replace the 2D NN atom
    with ``g_inv(NN_a(z) + NN_b(z) * h(w))``, reducing one 2D problem to
    two 1D problems.

    Parameters
    ----------
    root : Node
        Current AST root.
    target : AtomNode
        The 2D NN atom to replace.
    reuse : dict
        Tag -> module mapping for teacher extraction.
    train_loader : DataLoader
        Training data loader.
    device, dtype : torch.device, torch.dtype
    hit : dict
        Screening hit from ``_affine_decomposition_screen``.

    Returns
    -------
    ``(new_root, init_fn, meta)`` or ``None`` on failure.
    """
    if target.kind.lower() != "nn":
        return None

    var_idxs = tuple(int(i) for i in target.var_idxs)
    input_exprs = tuple(get_input_exprs(target))
    if len(input_exprs) != 2:
        return None

    tag = target.tag
    if tag is None or tag not in reuse:
        return None

    g_name = hit["g_name"]
    h_name = hit["h_name"]
    omega = float(hit.get("omega", 1.0))
    dataset_hit_map = {
        str(k): dict(v) for k, v in (dataset_hit_map or {}).items()
    }

    col_w = int(hit.get("col_w", 1))
    if col_w < 0 or col_w >= len(input_exprs):
        return None
    col_z = 1 - col_w
    z_expr = input_exprs[col_z]
    w_expr = input_exprs[col_w]

    # Build NN_a and NN_b as functions of the z effective input, which may be
    # a raw variable or a compound coordinate.
    if is_trivial_input(z_expr):
        z_global_idx = int(z_expr.var_idxs[0])
        nn_a_inputs = None
        nn_b_inputs = None
        nn_var_idxs = (z_global_idx,)
    else:
        nn_a_inputs = (clone_ast(z_expr),)
        nn_b_inputs = (clone_ast(z_expr),)
        nn_var_idxs = var_idxs
    nn_a_kwargs: Dict[str, Any] = {}
    nn_b_kwargs: Dict[str, Any] = {}

    # Copy NN hyperparams from original target
    if target.kwargs:
        for key in ("num_segments", "dual_layer", "seg_width"):
            if key in target.kwargs:
                nn_a_kwargs[key] = target.kwargs[key]
                nn_b_kwargs[key] = target.kwargs[key]

    # Tags for the two new NN atoms
    base_tag = tag or ""
    tag_a = f"{base_tag}_AD_a"
    tag_b = f"{base_tag}_AD_b"

    # Build NN atoms for a(z) and b(z)
    nn_a = AtomNode(
        kind="nn",
        var_idxs=nn_var_idxs,
        kwargs=nn_a_kwargs,
        tag=tag_a,
        inputs=nn_a_inputs,
    )
    nn_b = AtomNode(
        kind="nn",
        var_idxs=nn_var_idxs,
        kwargs=nn_b_kwargs,
        tag=tag_b,
        inputs=nn_b_inputs,
    )

    # Build h(w) node
    w_node = clone_ast(w_expr)
    if h_name == "identity":
        h_node = w_node
    elif h_name == "cos":
        if abs(omega - 1.0) > 1e-9:
            h_node = CosNode(arg=MulNode(
                left=ConstNode(value=omega),
                right=w_node,
            ))
        else:
            h_node = CosNode(arg=w_node)
    elif h_name == "sin":
        if abs(omega - 1.0) > 1e-9:
            h_node = SinNode(arg=MulNode(
                left=ConstNode(value=omega),
                right=w_node,
            ))
        else:
            h_node = SinNode(arg=w_node)
    elif h_name == "one_minus_cos":
        if abs(omega - 1.0) > 1e-9:
            cos_arg = MulNode(
                left=ConstNode(value=omega),
                right=w_node,
            )
        else:
            cos_arg = w_node
        h_node = AddNode(
            left=ConstNode(value=1.0),
            right=MulNode(
                left=ConstNode(value=-1.0),
                right=CosNode(arg=cos_arg),
            ),
        )
    else:
        return None

    if bool(hit.get("global_affine", False)):
        alpha_fit = float(hit.get("global_alpha", 0.0))
        z_slope_fit = float(hit.get("global_z_slope", 0.0))
        w_slope_fit = float(hit.get("global_w_slope", 0.0))
        scale_ref = max(1.0, abs(z_slope_fit), abs(w_slope_fit))
        if abs(alpha_fit) <= 1.0e-4 * scale_ref:
            base_tag = tag or "leaf"
            z_node = clone_ast(z_expr)
            z_term = MulNode(
                left=_build_scalar_atom_from_variant(
                    {
                        "mode": "scale",
                        "name": f"{base_tag}_AD_lin_z",
                        "tag": f"{base_tag}_AD_lin_z",
                        "value": z_slope_fit,
                    }
                ),
                right=z_node,
            )
            w_term = MulNode(
                left=_build_scalar_atom_from_variant(
                    {
                        "mode": "scale",
                        "name": f"{base_tag}_AD_lin_w",
                        "tag": f"{base_tag}_AD_lin_w",
                        "value": w_slope_fit,
                    }
                ),
                right=h_node,
            )
            inner = AddNode(left=z_term, right=w_term)
            meta = {
                "structural": True,
                "pattern": "affine_decomp",
                "g_name": g_name,
                "h_name": h_name,
                "omega": omega,
                "median_r2": float(hit.get("median_r2", 0.0)),
                "global_affine": True,
            }
            if g_name == "identity":
                return _replace_node(root, target, inner), None, meta
            if g_name == "reciprocal":
                return _replace_node(root, target, PowNode(base=inner, exponent=-1.0)), None, meta

    # Compose inner = NN_a(z) + NN_b(z) * h(w)
    inner = AddNode(left=nn_a, right=MulNode(left=nn_b, right=h_node))

    # Apply g_inv
    if g_name == "identity":
        new_subtree = inner
    elif g_name == "reciprocal":
        new_subtree = PowNode(base=inner, exponent=-1.0)
    else:
        return None

    cand_root = _replace_node(root, target, new_subtree)

    # Build init_fn that pre-trains NN_a on (z, a_interp) and NN_b on (z, b_interp)
    # using the bin data from screening
    def _custom_init(root_inner, model_inner, *, dataset_idx=None, dataset_id=None):
        """Initialize NN_a and NN_b from screening bin data via short LM fit."""
        from .stageB import _collect_all_atoms

        selected_hit = hit
        if dataset_id is not None:
            selected_hit = dataset_hit_map.get(str(dataset_id), selected_hit)
        z_centers_t = torch.tensor(selected_hit["z_centers"], dtype=torch.float64)
        a_values_t = torch.tensor(selected_hit["a_values"], dtype=torch.float64)
        b_values_t = torch.tensor(selected_hit["b_values"], dtype=torch.float64)

        atoms = _collect_all_atoms(root_inner)
        leaves = list(model_inner.leaf)

        for atom_i, leaf_mod in zip(atoms, leaves):
            atag = getattr(atom_i, "tag", None)
            if atag not in (tag_a, tag_b):
                continue
            teacher_nn = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))

            # Build training data: z_centers -> a_values or b_values
            vals = a_values_t if atag == tag_a else b_values_t
            z_in = z_centers_t.unsqueeze(1).to(
                device=teacher_nn.weight.device if hasattr(teacher_nn, "weight") else device,
                dtype=dtype,
            ) if hasattr(teacher_nn, "weight") else z_centers_t.unsqueeze(1).to(device=device, dtype=dtype)
            vals_in = vals.to(device=z_in.device, dtype=z_in.dtype)

            # Densify: interpolate sparse bin points → dense grid so the
            # NN pre-training is well-conditioned (20 pts → 1000 pts).
            import numpy as _np

            n_dense = 1000
            z_np = z_in.detach().cpu().numpy().ravel()
            v_np = vals_in.detach().cpu().numpy().ravel()
            z_dense_np = _np.linspace(z_np.min(), z_np.max(), n_dense)
            v_dense_np = _np.interp(z_dense_np, z_np, v_np)
            z_in = torch.from_numpy(z_dense_np).unsqueeze(1).to(
                device=z_in.device, dtype=z_in.dtype
            )
            vals_in = torch.from_numpy(v_dense_np).to(
                device=z_in.device, dtype=z_in.dtype
            )

            # Pre-train NN to approximate the interpolated values
            # Phase 1: Adam warm-up (gets into the right local optimum)
            # Phase 2: LBFGS fine-tuning (superlinear convergence to high accuracy)
            try:
                teacher_nn.train()
                # Phase 1: Adam warm-up
                opt = torch.optim.Adam(teacher_nn.parameters(), lr=1e-2)
                for _ in range(300):
                    pred = teacher_nn(z_in).view(-1)
                    loss = ((pred - vals_in) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # Phase 2: LBFGS fine-tuning
                try:
                    opt_lbfgs = torch.optim.LBFGS(
                        teacher_nn.parameters(), lr=1.0, max_iter=20,
                        history_size=20, line_search_fn="strong_wolfe",
                    )

                    def closure():
                        opt_lbfgs.zero_grad()
                        pred = teacher_nn(z_in).view(-1)
                        l = ((pred - vals_in) ** 2).mean()
                        l.backward()
                        return l

                    for _ in range(30):
                        opt_lbfgs.step(closure)
                except Exception:
                    pass  # Adam init is sufficient if LBFGS fails

                teacher_nn.eval()
                with torch.no_grad():
                    final_loss = ((teacher_nn(z_in).view(-1) - vals_in) ** 2).mean().item()
                print(f"[Stage B] AffineDecomp init {atag}: final_mse={final_loss:.4e}")
            except Exception as e:
                print(f"[Stage B] AffineDecomp init {atag} failed: {e}")

    meta = {
        "structural": True,
        "pattern": "affine_decomp",
        "g_name": g_name,
        "h_name": h_name,
        "omega": omega,
        "median_r2": float(hit.get("median_r2", 0.0)),
    }
    return cand_root, _custom_init, meta
