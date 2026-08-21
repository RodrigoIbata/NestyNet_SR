# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Hamiltonian and contact geometry discovery from (z, zdot) data.

This module discovers sparse Hamiltonian systems of the form:

    ż = J∇H(z)

where z = (q, p) is the phase-space state, J is the symplectic matrix, and
H(z) is a sparse linear combination of library terms:

    H(z) = Σ a_k φ_k(z)

Key insight: Even though H is a scalar potential, the design matrix for
regression is *linear in the unknown coefficients a_k*, because:

    J∇H(z) = Σ a_k J∇φ_k(z)

This lets us reuse STLSQ/group-STLSQ machinery from de_search.py while
naturally embedding Hamiltonian structure into the hypothesis class.

Phase-1 data convention
-----------------------
For direct (x, xdot, xddot) data:
    q := x           (generalized positions)
    p := ẋ           (generalized velocities)
    z = (q, p)       shape (N, 2n)
    ż = (q̇, ṗ) = (ẋ, ẍ)   shape (N, 2n)

Extension: Contact geometry with linear damping
------------------------------------------------
For systems with dissipation, we support the contact-inspired form:

    q̇ = ∂_p H_0
    ṗ = -∂_q H_0 - γp

where γ is a damping coefficient. This stays linear in unknowns if γ is
constant (or linear in basis functions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from nestynet_sr.sr_core.bridges import (
    Add,
    AddNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    FreeConst,
    LogNode,
    Mul,
    MulNode,
    Node,
    Pow,
    PowNode,
    SinNode,
    Var,
    const_full_like,
)

# Reuse STLSQ machinery from de_search
from nestynet_sr.sr_de.de_search import group_stlsq, ridge_lstsq, stlsq

# ──────────────────────────────────────────────────────────────
# Configuration & results
# ──────────────────────────────────────────────────────────────


@dataclass
class HamiltonianSearchConfig:
    """Configuration for Hamiltonian discovery.

    Attributes
    ----------
    n_dof : int
        Number of degrees of freedom (n). State dimension is 2n.

    Term library controls
    ---------------------
    max_q_power : int
        Maximum power for q_i monomials (default 2)
    max_p_power : int
        Maximum power for p_i monomials (default 2)
    include_const : bool
        Include constant term (pure gauge; shifts H by constant, default False)
    mechanical_split : bool
        Use H = T(p) + V(q) form (no q-p cross terms). Highly recommended.
        Default True.
    include_cross_dof : bool
        Include cross-DOF terms (q_i*q_j, p_i*p_j for i≠j). Expensive.
        Default False.
    include_qp_cross : bool
        Include within-DOF q-p cross terms (q_i*p_i, q_i^2*p_i, etc).
        Only active if mechanical_split=False. Default False.

    STLSQ sparsification
    --------------------
    ridge : float
        Ridge regularization for numerical stability (default 1e-10)
    stlsq_lambda : float
        STLSQ sparsity threshold (default 1e-3)
    stlsq_max_iter : int
        Maximum STLSQ iterations (default 10)

    Model selection
    ---------------
    sparsity_penalty : float
        Penalty per term for model selection (default 1e-3)
    """

    n_dof: int = 1

    # Library controls
    max_q_power: int = 2
    max_p_power: int = 2
    include_const: bool = False
    mechanical_split: bool = True
    include_cross_dof: bool = False
    include_qp_cross: bool = False

    # STLSQ
    ridge: float = 1e-10
    stlsq_lambda: float = 1e-3
    stlsq_max_iter: int = 10

    # Model selection
    sparsity_penalty: float = 1e-3


@dataclass
class HamiltonianSearchResult:
    """Result of single-dataset Hamiltonian discovery.

    Attributes
    ----------
    n_dof : int
        Number of degrees of freedom
    term_asts : List[Node]
        List of term ASTs φ_k that form H = Σ a_k φ_k(z)
    coeffs : torch.Tensor
        Coefficients a_k, shape (K,)
    rms_train : float
        RMS residual on training data
    rms_val : float, optional
        RMS residual on validation data
    H_ast : Node, optional
        Full AST for H(z) = Σ a_k φ_k(z)
    """

    n_dof: int
    term_asts: List[Node]
    coeffs: torch.Tensor  # (K,)
    rms_train: float
    rms_val: Optional[float] = None
    H_ast: Optional[Node] = None

    def canonicalize_coeffs(self, tol: float = 1e-3) -> List[Tuple[float, str]]:
        """Snap near-integer coefficients for human-readable output."""
        results = []
        for c, term in zip(self.coeffs.tolist(), self.term_asts):
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
                    term_str = str(type(term).__name__)
            results.append((c_snapped, term_str))
        return results

    def format_hamiltonian(self, tol: float = 1e-3) -> str:
        """Format the discovered Hamiltonian in human-readable form."""
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
                # Strip only outer parentheses if entire expression is wrapped
                term_clean = term_str
                if term_clean.startswith("(") and term_clean.endswith(")"):
                    # Check if these are balanced outer parens
                    depth = 0
                    for i, ch in enumerate(term_clean):
                        if ch == "(":
                            depth += 1
                        elif ch == ")":
                            depth -= 1
                        # If depth hits 0 before the end, outer parens aren't wrapping everything
                        if depth == 0 and i < len(term_clean) - 1:
                            break
                    # If we made it to the end with balanced parens, strip them
                    if depth == 0 and i == len(term_clean) - 1:
                        term_clean = term_clean[1:-1]

                if abs(abs(c_snap) - 1.0) < 1e-10:
                    term_full = f"{term_clean}" if c_snap > 0 else f"-{term_clean}"
                else:
                    term_full = f"{c_snap:g}*{term_clean}"

            terms_str.append(term_full)

        if not terms_str:
            return "H(z) = 0"

        rhs = " + ".join(terms_str).replace("+ -", "- ")
        return f"H(z) = {rhs}"


@dataclass
class HamiltonianSearchResultMulti:
    """Result of multi-dataset Hamiltonian discovery with shared term support.

    The discovered Hamiltonian has the same *active terms* across all datasets,
    but allows dataset-specific coefficients. Coefficients are returned as a
    matrix with shape (D, K_sel) matching `term_asts`.

    Attributes
    ----------
    n_dof : int
        Number of degrees of freedom
    term_asts : List[Node]
        List of shared term ASTs φ_k
    coeffs : torch.Tensor
        Coefficient matrix, shape (D, K_sel)
    rms_train : List[float]
        RMS residuals on training data per dataset
    rms_val : List[float], optional
        RMS residuals on validation data per dataset
    dataset_ids : List[str], optional
        Dataset identifiers
    H_asts : List[Node], optional
        Per-dataset Hamiltonian ASTs
    """

    n_dof: int
    term_asts: List[Node]
    coeffs: torch.Tensor  # (D, K_sel)
    rms_train: List[float]
    rms_val: Optional[List[float]] = None
    dataset_ids: Optional[List[str]] = None
    H_asts: Optional[List[Node]] = None

    def format_hamiltonian_for_dataset(self, d: int, tol: float = 1e-3) -> str:
        """Format Hamiltonian for a specific dataset."""
        tmp = HamiltonianSearchResult(
            n_dof=self.n_dof,
            term_asts=self.term_asts,
            coeffs=self.coeffs[d].detach().cpu(),
            rms_train=self.rms_train[d],
            rms_val=None,
        )
        return tmp.format_hamiltonian(tol=tol)


# ──────────────────────────────────────────────────────────────
# AST evaluation with autograd for ∇φ_k
# ──────────────────────────────────────────────────────────────


def _eval_var_ast(node: Optional[Node], z: torch.Tensor) -> torch.Tensor:
    """Evaluate a pure-variable AST (no u/du atoms) on phase-space coords z.

    Parameters
    ----------
    node : Node or None
        AST node to evaluate. If None, returns all-ones (constant 1).
    z : torch.Tensor
        Phase-space coordinates, shape (N, 2n)

    Returns
    -------
    val : torch.Tensor
        Evaluated values, shape (N, 1)
    """
    if node is None:
        return z.new_ones(z.shape[0], 1)

    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()

        if kind in ("var", "x", "input"):
            j = int(node.var_idxs[0])
            return z[:, j : j + 1]

        if kind in ("const", "constant"):
            val = float(getattr(node, "kwargs", {}).get("value", 1.0))
            return z.new_full((z.shape[0], 1), val)

        raise TypeError(f"Unsupported atom kind in Hamiltonian term: {kind!r}")

    if isinstance(node, AddNode):
        return _eval_var_ast(node.left, z) + _eval_var_ast(node.right, z)

    if isinstance(node, MulNode):
        return _eval_var_ast(node.left, z) * _eval_var_ast(node.right, z)

    if isinstance(node, PowNode):
        base_val = _eval_var_ast(node.base, z)
        exp = node.exponent
        if isinstance(exp, (int, float)):
            return base_val.pow(float(exp))
        # Non-constant exponents break gradient correctness
        raise TypeError(
            "Hamiltonian library terms must have constant (int/float) exponents. "
            f"Got exponent as Node: {exp}"
        )

    if isinstance(node, LogNode):
        return torch.log(_eval_var_ast(node.arg, z))

    if isinstance(node, ExpNode):
        return torch.exp(_eval_var_ast(node.arg, z))

    if isinstance(node, SinNode):
        return torch.sin(_eval_var_ast(node.arg, z))

    if isinstance(node, CosNode):
        return torch.cos(_eval_var_ast(node.arg, z))

    if isinstance(node, ConstNode):
        return const_full_like(z, (z.shape[0],), node.value)

    raise TypeError(f"Unknown node type: {type(node)}")


def build_hamiltonian_design_matrix(
    z: torch.Tensor,
    zdot: torch.Tensor,
    term_asts: List[Optional[Node]],
    n_dof: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build design matrix Φ and target y for Hamiltonian regression.

    For each term φ_k(z), compute J∇φ_k(z) where J is the symplectic matrix:
        J = [[0, I_n],
             [-I_n, 0]]

    Then stack into design matrix:
        Φ[:, k] = vec(J∇φ_k(z))
        y = vec(ż)

    Parameters
    ----------
    z : torch.Tensor
        Phase-space state, shape (N, 2n)
    zdot : torch.Tensor
        Phase-space derivative, shape (N, 2n)
    term_asts : List[Node]
        List of term ASTs φ_k. None represents constant term.
    n_dof : int
        Number of degrees of freedom (n)

    Returns
    -------
    Phi : torch.Tensor
        Design matrix, shape (N*2n, K)
    y : torch.Tensor
        Target vector, shape (N*2n,)
    """
    N, M = z.shape
    assert M == 2 * n_dof, f"Expected z.shape = (N, {2 * n_dof}), got {z.shape}"
    assert zdot.shape == z.shape, f"zdot.shape {zdot.shape} != z.shape {z.shape}"

    # Target vector
    y = zdot.reshape(N * M)

    # Design matrix columns
    Phi = z.new_empty((N * M, len(term_asts)))

    # Reuse z_grad for all terms (performance optimization)
    z_grad = z.detach().requires_grad_(True)

    for k, term_ast in enumerate(term_asts):
        # Special case: constant terms have zero gradient → zero contribution to J∇H
        if term_ast is None:
            Phi[:, k].zero_()
            continue

        # Check for explicit constant atoms
        if isinstance(term_ast, AtomNode):
            kind = str(getattr(term_ast, "kind", "")).lower()
            if kind in ("const", "constant"):
                Phi[:, k].zero_()
                continue

        # Evaluate term φ_k(z) with autograd enabled
        phi_val = _eval_var_ast(term_ast, z_grad)  # (N, 1)

        # Compute gradient ∇φ_k(z), shape (N, 2n)
        # allow_unused=True handles edge cases where term doesn't depend on z
        grad_phi = torch.autograd.grad(
            phi_val.sum(), z_grad, create_graph=False, allow_unused=True
        )[0]

        # Handle case where gradient is None (term doesn't depend on z)
        if grad_phi is None:
            Phi[:, k].zero_()
            continue

        # Apply symplectic matrix J:
        # J∇φ = [∂_p φ, -∂_q φ] where ∂_q φ = grad_phi[:, :n], ∂_p φ = grad_phi[:, n:]
        J_grad_phi = torch.cat(
            [
                grad_phi[:, n_dof:],  # ∂_p φ  (q̇ component)
                -grad_phi[:, :n_dof],  # -∂_q φ (ṗ component)
            ],
            dim=1,
        )  # (N, 2n)

        # Flatten to column vector
        Phi[:, k] = J_grad_phi.reshape(-1)

    return Phi, y


# ──────────────────────────────────────────────────────────────
# Library generation
# ──────────────────────────────────────────────────────────────


def _pow_if(node: Node, p: int) -> Node:
    """Return node^p, or just node if p=1."""
    if p == 1:
        return node
    return Pow(node, p)


def build_hamiltonian_library_terms(cfg: HamiltonianSearchConfig) -> List[Optional[Node]]:
    """Build library of candidate terms for H(z).

    Returns list of term ASTs φ_k (no coefficients). None represents constant.

    Default behavior (mechanical_split=True):
        H = T(p) + V(q)
        - Includes p_i^k for k=1..max_p_power (kinetic energy terms)
        - Includes q_i^k for k=1..max_q_power (potential energy terms)
        - No q-p cross terms

    With mechanical_split=False:
        - Allows within-DOF q_i*p_i cross terms if include_qp_cross=True
        - Still excludes cross-DOF q-p terms (q_i*p_j for i≠j) by default

    With include_cross_dof=True:
        - Includes q_i*q_j, p_i*p_j for i≠j
        - Search space grows as O(n^2)

    Parameters
    ----------
    cfg : HamiltonianSearchConfig
        Configuration with library knobs

    Returns
    -------
    terms : List[Optional[Node]]
        List of term ASTs. First element is None if include_const=True.
    """
    n = cfg.n_dof
    terms: List[Optional[Node]] = []

    # Constant term (pure gauge for Hamiltonian)
    if cfg.include_const:
        terms.append(None)

    # Build q_i and p_i variable nodes
    # Convention: z = [q_0, ..., q_{n-1}, p_0, ..., p_{n-1}]
    q_vars = [Var(i) for i in range(n)]
    p_vars = [Var(n + i) for i in range(n)]

    # Potential energy terms: V(q) = Σ powers of q_i
    for i in range(n):
        for power in range(1, cfg.max_q_power + 1):
            terms.append(_pow_if(q_vars[i], power))

    # Kinetic energy terms: T(p) = Σ powers of p_i
    for i in range(n):
        for power in range(1, cfg.max_p_power + 1):
            terms.append(_pow_if(p_vars[i], power))

    # Cross-DOF terms in q (q_i * q_j for i < j)
    if cfg.include_cross_dof:
        for i in range(n):
            for j in range(i + 1, n):
                terms.append(Mul(q_vars[i], q_vars[j]))

    # Cross-DOF terms in p (p_i * p_j for i < j)
    if cfg.include_cross_dof:
        for i in range(n):
            for j in range(i + 1, n):
                terms.append(Mul(p_vars[i], p_vars[j]))

    # Within-DOF q-p cross terms (only if not mechanical_split)
    if not cfg.mechanical_split and cfg.include_qp_cross:
        for i in range(n):
            # q_i * p_i
            terms.append(Mul(q_vars[i], p_vars[i]))
            # q_i^2 * p_i, q_i * p_i^2, etc. (optional, can add more)
            for q_pow in range(1, cfg.max_q_power + 1):
                for p_pow in range(1, cfg.max_p_power + 1):
                    if q_pow == 1 and p_pow == 1:
                        continue  # Already added q_i*p_i
                    terms.append(Mul(_pow_if(q_vars[i], q_pow), _pow_if(p_vars[i], p_pow)))

    # Deduplicate by repr
    uniq: Dict[str, Optional[Node]] = {}
    for t in terms:
        key = "1" if t is None else repr(t)
        uniq[key] = t

    return list(uniq.values())


# ──────────────────────────────────────────────────────────────
# Main search
# ──────────────────────────────────────────────────────────────


def discover_hamiltonian_from_data(
    z_train: torch.Tensor,
    zdot_train: torch.Tensor,
    z_val: Optional[torch.Tensor] = None,
    zdot_val: Optional[torch.Tensor] = None,
    *,
    cfg: Optional[HamiltonianSearchConfig] = None,
) -> HamiltonianSearchResult:
    """Discover a sparse Hamiltonian H(z) from (z, ż) data.

    Solves for sparse coefficients a_k in:
        H(z) = Σ a_k φ_k(z)

    such that ż ≈ J∇H(z), where J is the symplectic matrix.

    Parameters
    ----------
    z_train : torch.Tensor
        Training phase-space states, shape (N, 2n)
    zdot_train : torch.Tensor
        Training phase-space derivatives, shape (N, 2n)
    z_val : torch.Tensor, optional
        Validation phase-space states, shape (M, 2n)
    zdot_val : torch.Tensor, optional
        Validation phase-space derivatives, shape (M, 2n)
    cfg : HamiltonianSearchConfig, optional
        Configuration. If None, uses defaults.

    Returns
    -------
    result : HamiltonianSearchResult
        Discovered Hamiltonian with coefficients and terms

    Examples
    --------
    >>> # Simple harmonic oscillator: H = 0.5*p^2 + 0.5*q^2
    >>> cfg = HamiltonianSearchConfig(n_dof=1, max_q_power=2, max_p_power=2)
    >>> result = discover_hamiltonian_from_data(z_train, zdot_train, cfg=cfg)
    >>> print(result.format_hamiltonian())
    H(z) = 0.5*q0**2 + 0.5*p0**2
    """
    if cfg is None:
        cfg = HamiltonianSearchConfig()

    N, M = z_train.shape
    n_dof = M // 2

    # Auto-detect n_dof if not set
    if cfg.n_dof != n_dof:
        print(f"Auto-detected n_dof={n_dof} from data shape {z_train.shape}")
        cfg = HamiltonianSearchConfig(
            n_dof=n_dof,
            max_q_power=cfg.max_q_power,
            max_p_power=cfg.max_p_power,
            include_const=cfg.include_const,
            mechanical_split=cfg.mechanical_split,
            include_cross_dof=cfg.include_cross_dof,
            include_qp_cross=cfg.include_qp_cross,
            ridge=cfg.ridge,
            stlsq_lambda=cfg.stlsq_lambda,
            stlsq_max_iter=cfg.stlsq_max_iter,
            sparsity_penalty=cfg.sparsity_penalty,
        )

    # Build term library
    term_asts = build_hamiltonian_library_terms(cfg)
    print(f"Built Hamiltonian library with {len(term_asts)} terms")
    if cfg.mechanical_split:
        print("  Using mechanical split: H = T(p) + V(q)")

    # Build design matrix and target
    Phi, y = build_hamiltonian_design_matrix(z_train, zdot_train, term_asts, cfg.n_dof)

    # Drop non-finite rows
    mask = torch.isfinite(y) & torch.isfinite(Phi).all(dim=1)
    if int(mask.sum()) < 10:
        raise RuntimeError("Too few finite rows after filtering")
    Phi = Phi[mask]
    y = y[mask]

    print(f"Design matrix shape: {Phi.shape}, target shape: {y.shape}")

    # Run STLSQ
    c, keep = stlsq(Phi, y, ridge=cfg.ridge, lam=cfg.stlsq_lambda, max_iter=cfg.stlsq_max_iter)

    # Apply mask
    c_sel = c[keep]
    term_sel = [t for t, k in zip(term_asts, keep.tolist()) if k]
    Phi_sel = Phi[:, keep]

    print(f"STLSQ selected {len(term_sel)} / {len(term_asts)} terms")

    # Unbiased refit on selected terms
    if len(c_sel) > 0:
        c_sel = ridge_lstsq(Phi_sel, y, ridge=0.0)

    # Training RMS
    residual = y - Phi_sel @ c_sel
    rms_train = float(residual.square().mean().sqrt().detach().cpu())

    # Validation RMS
    rms_val = None
    if z_val is not None and zdot_val is not None:
        Phi_val, y_val = build_hamiltonian_design_matrix(z_val, zdot_val, term_sel, cfg.n_dof)
        residual_val = y_val - Phi_val @ c_sel
        rms_val = float(residual_val.square().mean().sqrt().detach().cpu())

    # Build Hamiltonian AST
    H_ast = build_hamiltonian_ast(term_sel, c_sel.detach().cpu())

    return HamiltonianSearchResult(
        n_dof=cfg.n_dof,
        term_asts=term_sel,
        coeffs=c_sel.detach().cpu(),
        rms_train=rms_train,
        rms_val=rms_val,
        H_ast=H_ast,
    )


def discover_hamiltonian_from_data_multi(
    z_trains: Sequence[torch.Tensor],
    zdot_trains: Sequence[torch.Tensor],
    z_vals: Optional[Sequence[torch.Tensor]] = None,
    zdot_vals: Optional[Sequence[torch.Tensor]] = None,
    *,
    cfg: Optional[HamiltonianSearchConfig] = None,
    dataset_ids: Optional[Sequence[str]] = None,
    mode: str = "group",
) -> HamiltonianSearchResultMulti:
    """Multi-dataset Hamiltonian discovery with shared term support.

    Discovers a Hamiltonian with shared *active terms* across all datasets.

    Modes:
    ------
    - "shared": Fully shared H across all datasets (Mode A)
        - Concatenate all datasets and run single STLSQ
        - One coefficient vector (replicated across datasets in result)
        - Use when physics is identical across datasets
    - "group": Shared support, dataset-specific coefficients (Mode B, default)
        - Use group-STLSQ for shared term selection
        - Coefficient matrix (D, K) with dataset-specific values
        - Use when same functional form but different parameter values

    Parameters
    ----------
    z_trains : Sequence[torch.Tensor]
        Training states for each dataset, each shape (N_d, 2n)
    zdot_trains : Sequence[torch.Tensor]
        Training derivatives for each dataset, each shape (N_d, 2n)
    z_vals : Sequence[torch.Tensor], optional
        Validation states for each dataset
    zdot_vals : Sequence[torch.Tensor], optional
        Validation derivatives for each dataset
    cfg : HamiltonianSearchConfig, optional
        Configuration
    dataset_ids : Sequence[str], optional
        Dataset identifiers
    mode : str, optional
        Multi-dataset mode: "shared" or "group" (default "group")

    Returns
    -------
    result : HamiltonianSearchResultMulti
        Multi-dataset result with coefficient matrix (D, K)
    """
    if cfg is None:
        cfg = HamiltonianSearchConfig()

    if mode not in ("shared", "group"):
        raise ValueError(f"mode must be 'shared' or 'group', got {mode!r}")

    D = len(z_trains)
    if D == 0:
        raise ValueError("Need at least one dataset")
    if len(zdot_trains) != D:
        raise ValueError("z_trains and zdot_trains must have same length")

    # Auto-detect n_dof from first dataset
    n_dof = z_trains[0].shape[1] // 2
    if cfg.n_dof != n_dof:
        print(f"Auto-detected n_dof={n_dof} from data shape")
        cfg = HamiltonianSearchConfig(
            n_dof=n_dof,
            max_q_power=cfg.max_q_power,
            max_p_power=cfg.max_p_power,
            include_const=cfg.include_const,
            mechanical_split=cfg.mechanical_split,
            include_cross_dof=cfg.include_cross_dof,
            include_qp_cross=cfg.include_qp_cross,
            ridge=cfg.ridge,
            stlsq_lambda=cfg.stlsq_lambda,
            stlsq_max_iter=cfg.stlsq_max_iter,
            sparsity_penalty=cfg.sparsity_penalty,
        )

    # Build shared term library
    term_asts = build_hamiltonian_library_terms(cfg)
    print(f"Built Hamiltonian library with {len(term_asts)} terms")
    print(f"Multi-dataset mode: {mode}")

    # Build design matrices for each dataset
    Phis = []
    ys = []
    for i in range(D):
        Phi, y = build_hamiltonian_design_matrix(z_trains[i], zdot_trains[i], term_asts, cfg.n_dof)

        # Drop non-finite rows
        mask = torch.isfinite(y) & torch.isfinite(Phi).all(dim=1)
        if int(mask.sum()) < 10:
            raise RuntimeError(f"Too few finite rows for dataset {i}")

        Phis.append(Phi[mask])
        ys.append(y[mask])

    # Mode A: Fully shared H (concatenate datasets)
    if mode == "shared":
        print("  Mode A: Fitting fully shared Hamiltonian across all datasets")

        # Concatenate all datasets
        Phi_cat = torch.cat(Phis, dim=0)
        y_cat = torch.cat(ys, dim=0)

        # Run single STLSQ
        c, keep = stlsq(
            Phi_cat, y_cat, ridge=cfg.ridge, lam=cfg.stlsq_lambda, max_iter=cfg.stlsq_max_iter
        )

        # Unbiased refit
        c_sel = c[keep]
        K_sel = int(keep.sum())
        if K_sel == 0:
            raise RuntimeError("STLSQ selected zero terms")

        c_sel = ridge_lstsq(Phi_cat[:, keep], y_cat, ridge=0.0)

        # Replicate coefficients across datasets
        C_sel = c_sel.unsqueeze(0).expand(D, K_sel).clone()

        term_sel = [t for t, k in zip(term_asts, keep.tolist()) if k]
        print(
            f"  STLSQ selected {len(term_sel)} / {len(term_asts)} terms (shared across all datasets)"
        )

    # Mode B: Shared support, dataset-specific coefficients (group-STLSQ)
    else:  # mode == "group"
        print("  Mode B: Fitting shared support with dataset-specific coefficients")

        # Run group-STLSQ
        C, keep = group_stlsq(
            Phis, ys, ridge=cfg.ridge, lam=cfg.stlsq_lambda, max_iter=cfg.stlsq_max_iter
        )

        # Unbiased refit on selected terms
        K_sel = int(keep.sum())
        if K_sel == 0:
            raise RuntimeError("Group-STLSQ selected zero terms")

        C_sel = torch.zeros((D, K_sel), device=Phis[0].device, dtype=Phis[0].dtype)
        for i in range(D):
            C_sel[i] = ridge_lstsq(Phis[i][:, keep], ys[i], ridge=0.0)

        term_sel = [t for t, k in zip(term_asts, keep.tolist()) if k]
        print(f"  Group-STLSQ selected {len(term_sel)} / {len(term_asts)} terms")

    # Training RMS per dataset
    rms_trains = []
    for i in range(D):
        residual = ys[i] - Phis[i][:, keep] @ C_sel[i]
        rms_trains.append(float(residual.square().mean().sqrt().detach().cpu()))

    # Validation RMS per dataset
    rms_vals = None
    if z_vals is not None and zdot_vals is not None:
        rms_vals = []
        for i in range(D):
            Phi_val, y_val = build_hamiltonian_design_matrix(
                z_vals[i], zdot_vals[i], term_sel, cfg.n_dof
            )
            residual_val = y_val - Phi_val @ C_sel[i]
            rms_vals.append(float(residual_val.square().mean().sqrt().detach().cpu()))

    # Build per-dataset Hamiltonian ASTs
    H_asts = []
    for i in range(D):
        H_ast = build_hamiltonian_ast(term_sel, C_sel[i].detach().cpu())
        H_asts.append(H_ast)

    return HamiltonianSearchResultMulti(
        n_dof=cfg.n_dof,
        term_asts=term_sel,
        coeffs=C_sel.detach().cpu(),
        rms_train=rms_trains,
        rms_val=rms_vals,
        dataset_ids=list(dataset_ids) if dataset_ids is not None else None,
        H_asts=H_asts,
    )


def build_hamiltonian_ast(
    term_asts: List[Optional[Node]], coeffs: torch.Tensor, *, coeff_prefix: str = "a"
) -> Node:
    """Build an AST for H(z) = Σ a_k φ_k(z).

    Parameters
    ----------
    term_asts : List[Optional[Node]]
        List of term ASTs. None represents constant.
    coeffs : torch.Tensor
        Coefficients, shape (K,)
    coeff_prefix : str
        Prefix for coefficient names (default "a")

    Returns
    -------
    H_ast : Node
        AST for H(z) = Σ a_k φ_k(z)
    """
    if len(term_asts) == 0:
        # Return constant zero
        return AtomNode(kind="const", var_idxs=(), kwargs={"value": 0.0})

    # Start with first term
    root: Optional[Node] = None

    for i, (term, c) in enumerate(zip(term_asts, coeffs.tolist())):
        name = f"{coeff_prefix}{i}"

        if term is None:
            # Constant term
            term_node = FreeConst(name, init=float(c))
        else:
            # Coefficient * term
            term_node = Mul(FreeConst(name, init=float(c)), term)

        if root is None:
            root = term_node
        else:
            root = Add(root, term_node)

    return root if root is not None else AtomNode(kind="const", var_idxs=(), kwargs={"value": 0.0})
