# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Polynomial split function for SubtreeSeparability with coefficient transfer.
This is the implementation of _build_poly_split_from_subtree_separability.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

from nestynet_sr.sr_core.atoms import PolyLeaf
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    Node,
    ast_equals,
    get_input_exprs,
    replace_atom_in_ast,
)


def _build_poly_split_from_subtree_separability(
    root: Node,
    u_node: AtomNode,  # The polynomial atom being split
    model: torch.nn.Module,  # Full composite model
    op: torch.op,  # torch.add or torch.multiply
    group1_global: List[int],
    group2_global: List[int],
    device: torch.device,
    dtype: torch.dtype,
    rel_coeff_tol: float = 1e-10,
) -> Tuple[Optional[Node], Optional[Callable]]:
    """
    Build polynomial split candidate with coefficient transfer for SubtreeSeparability.

    When SubtreeSeparability detects that a polynomial atom is separable, this function
    creates polynomial atoms (not NN atoms) for each variable group and transfers
    fitted coefficients from the original polynomial to preserve the functional form.

    Parameters
    ----------
    root : Node
        The full AST root
    u_node : AtomNode
        The polynomial atom being split (kind='poly')
    model : torch.nn.Module
        The full composite model containing trained parameters
    op : torch.op
        torch.add or torch.multiply from separability detection
    group1_global : List[int]
        Global variable indices for first factor/term
    group2_global : List[int]
        Global variable indices for second factor/term
    device : torch.device
    dtype : torch.dtype
    rel_coeff_tol : float
        Relative tolerance for "non-zero" coefficients

    Returns
    -------
    cand_root : Optional[Node]
        New AST with u_node replaced by split, or None if split not applicable
    custom_init_fn : Optional[Callable]
        Function to initialize split polynomials from original coefficients,
        or None if split not applicable
    """
    from .stageB import _collect_all_atoms  # local import to avoid circular dependency

    # Verify this is a polynomial atom
    if u_node.kind.lower() != "poly":
        return None, None

    # Only handle binary additive splits for now
    # Multiplicative polynomial splits require factorization which is complex
    if op is not torch.add:
        return None, None

    if len(group1_global) == 0 or len(group2_global) == 0:
        return None, None

    # Find the polynomial core in the composite model
    atoms = _collect_all_atoms(root)
    leaves = list(model.leaf)
    poly_core: Optional[PolyLeaf] = None

    for atom_i, leaf_mod in zip(atoms, leaves):
        core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))
        if atom_i is u_node and isinstance(core, PolyLeaf):
            poly_core = core
            break

    if poly_core is None:
        return None, None

    # Extract fitted coefficients and exponents
    exps_full = poly_core.exps.detach().cpu()  # Shape: [n_terms, n_dim]
    coeffs_full = poly_core.coeffs.detach().cpu()  # Shape: [n_terms]
    n_terms, n_dim_original = exps_full.shape

    if n_terms == 0:
        return None, None

    # Get degree from original polynomial
    degree = int(u_node.kwargs.get("degree", u_node.kwargs.get("deg", poly_core.degree)))

    # Build mapping: local dimension index -> global variable index
    # u_node.var_idxs maps local coords (0, 1, ...) to global coords (e.g., 2, 3)
    local_to_global = {i: int(u_node.var_idxs[i]) for i in range(n_dim_original)}

    # Build mapping: global variable index -> which group it belongs to
    global_to_group = {}
    for g_idx in group1_global:
        global_to_group[g_idx] = 0  # Group 0
    for g_idx in group2_global:
        global_to_group[g_idx] = 1  # Group 1

    # Build mapping: global index -> local index within each group
    group1_local_map = {g: i for i, g in enumerate(group1_global)}
    group2_local_map = {g: i for i, g in enumerate(group2_global)}
    group_local_maps = [group1_local_map, group2_local_map]

    # Determine tolerance for "zero" coefficients
    scale = float(coeffs_full.abs().max().item())
    tol = scale * rel_coeff_tol if scale > 0 else rel_coeff_tol

    # Pre-check for cross-terms (terms that involve variables from both groups)
    # For additive split, ANY cross-term makes the function non-separable
    for k in range(n_terms):
        if float(abs(coeffs_full[k]).item()) <= tol:
            continue
        e_full = exps_full[k]  # Local coordinates

        # Check which groups this term involves
        groups_involved = set()
        for local_idx in range(n_dim_original):
            exp_val = int(e_full[local_idx].item())
            if exp_val == 0:
                continue
            global_idx = local_to_global[local_idx]
            group = global_to_group.get(global_idx, None)
            if group is not None:
                groups_involved.add(group)

        # If term involves both groups, it's a cross-term
        if len(groups_involved) > 1:
            # Cross-term detected - function is not additively separable
            # Reject this candidate
            return None, None

    # Create polynomial atoms for each group
    groups_global = [tuple(group1_global), tuple(group2_global)]
    group_inputs = [
        get_input_exprs(AtomNode(kind="poly", var_idxs=g, kwargs={}))
        for g in groups_global
    ]
    atoms_new = [
        AtomNode(kind="poly", var_idxs=g, kwargs={"degree": degree, "min_total": 0}, tag=None)
        for g in groups_global
    ]

    # Build AddNode for additive split
    new_subtree = AddNode(atoms_new[0], atoms_new[1])

    # Replace u_node in the full AST
    cand_root = replace_atom_in_ast(root, u_node, new_subtree)

    # Capture data for custom_init (create closure)
    exps_full_captured = exps_full
    coeffs_full_captured = coeffs_full
    local_to_global_captured = local_to_global
    global_to_group_captured = global_to_group
    group_local_maps_captured = group_local_maps
    tol_captured = tol

    def _custom_init(root_inner: Node, model_inner: torch.nn.Module):
        """
        Initialize the split polynomial atoms with coefficients from the original.

        This runs AFTER the new model is built but BEFORE training starts.
        """
        from .stageB import _collect_all_atoms  # re-import inside closure

        # Find the new polynomial cores in the rebuilt model
        atoms2 = _collect_all_atoms(root_inner)
        leaves2 = list(model_inner.leaf)

        cores = [None, None]  # [core_group0, core_group1]
        for atom_i, leaf_mod in zip(atoms2, leaves2):
            core = getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))
            if not isinstance(core, PolyLeaf):
                continue
            atom_inputs = get_input_exprs(atom_i)
            for gi, expected_inputs in enumerate(group_inputs):
                if len(atom_inputs) != len(expected_inputs):
                    continue
                if all(ast_equals(a, b) for a, b in zip(atom_inputs, expected_inputs)):
                    cores[gi] = core
                    break

        if any(c is None for c in cores):
            # Something went wrong; bail out
            return

        # Build exponent -> index maps for each new poly
        exp_maps: List[Dict[Tuple[int, ...], int]] = []
        for gi, c in enumerate(cores):
            exps_g = c.exps.detach().cpu()
            m: Dict[Tuple[int, ...], int] = {}
            for k, e in enumerate(exps_g):
                key = tuple(int(x) for x in e.tolist())
                m[key] = k
            exp_maps.append(m)

        # Prepare accumulators for new coefficients
        new_coeffs_cpu = [torch.zeros_like(c.coeffs.detach().cpu()) for c in cores]

        # Map each term from original polynomial to appropriate group
        for k in range(exps_full_captured.shape[0]):
            c_full = coeffs_full_captured[k]
            if float(abs(c_full).item()) <= tol_captured:
                continue

            e_full = exps_full_captured[k]  # Exponents in local coords
            total_deg = int(e_full.sum().item())

            # Handle constant term (attribute to first group)
            if total_deg == 0:
                gi = 0
                key = tuple([0] * len(groups_global[gi]))
                if key in exp_maps[gi]:
                    idx_new = exp_maps[gi][key]
                    new_coeffs_cpu[gi][idx_new] += c_full
                continue

            # Determine which group this term belongs to
            # and build the exponent in that group's local coordinates
            term_group = None
            term_exp_in_group = None

            for local_idx in range(n_dim_original):
                exp_val = int(e_full[local_idx].item())
                if exp_val == 0:
                    continue

                global_idx = local_to_global_captured[local_idx]
                group = global_to_group_captured.get(global_idx, None)

                if group is None:
                    # Variable not in either group - shouldn't happen after pre-check
                    return

                if term_group is None:
                    term_group = group
                    # Initialize exponent vector for this group
                    term_exp_in_group = [0] * len(groups_global[group])

                if term_group != group:
                    # Cross-term - shouldn't happen after pre-check
                    return

                # Map to local coordinate within the group
                group_local_idx = group_local_maps_captured[group][global_idx]
                term_exp_in_group[group_local_idx] = exp_val

            if term_group is not None and term_exp_in_group is not None:
                key = tuple(term_exp_in_group)
                if key in exp_maps[term_group]:
                    idx_new = exp_maps[term_group][key]
                    new_coeffs_cpu[term_group][idx_new] += c_full

        # Copy coefficients into the new cores
        for gi, c in enumerate(cores):
            with torch.no_grad():
                if c.coeffs.shape == new_coeffs_cpu[gi].shape:
                    c.coeffs.copy_(
                        new_coeffs_cpu[gi].to(device=c.coeffs.device, dtype=c.coeffs.dtype)
                    )

    return cand_root, _custom_init
