# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Subtree separability analysis utilities for Stage B.

This module provides functions for analyzing subtrees for separability
and building candidates based on separability detection.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from nestynet_sr.adaptors.ast_composite import ASTCompositeAdaptor
from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AddNode,
    AsinNode,
    AtanNode,
    AtomNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    collect_nn_atoms,
    replace_atom_in_ast,
    separability_proposal_to_ast,
)
from nestynet_sr.sr_core.separability_math import (
    check_generalized_additivity_ops,
    trapped_variable_ops,
)
from nestynet_sr.sr_search.poly_split_subtree_separability import (
    _build_poly_split_from_subtree_separability,
)
from nestynet_sr.sr_search.subtree_separability_helpers import run_subtree_separability

from .atom_mapping import _collect_all_atoms, _vars_in_subtree, build_atom_to_leaf_map
from .models import _SubtreeModel


def _probe_genadd_for_nn_leaf(
    root: Node,
    model: nn.Module,
    target: AtomNode,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    n_points: int = 2048,
    poly_deg: int = 2,
):
    if target.kind.lower() != "nn":
        return None
    var_idxs = [int(j) for j in target.var_idxs]
    if len(var_idxs) != 2:
        return None
    try:
        atom_to_leaf = build_atom_to_leaf_map(root, model)
    except Exception:
        return None
    subtree = _SubtreeModel(root=target, atom_to_leaf=atom_to_leaf)
    res = check_generalized_additivity_ops(
        model=subtree,
        datagen=train_loader,
        X_group=[var_idxs[0]],
        Y_group=[var_idxs[1]],
        device=device,
        dtype=dtype,
        n_points=n_points,
        poly_deg=poly_deg,
    )
    return res


def _probe_trapped_for_nn_leaf(
    root: Node,
    model: nn.Module,
    target: AtomNode,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    n_points: int = 2048,
    kind: str = "multiplicative",
    candidate_P: str = "product",
):
    if target.kind.lower() != "nn":
        return None
    var_idxs = [int(j) for j in target.var_idxs]
    if len(var_idxs) != 2:
        return None
    try:
        atom_to_leaf = build_atom_to_leaf_map(root, model)
    except Exception:
        return None
    subtree = _SubtreeModel(root=target, atom_to_leaf=atom_to_leaf)
    best = None
    for trapped_local, leaky_local in ((0, 1), (1, 0)):
        res = trapped_variable_ops(
            model=subtree,
            datagen=train_loader,
            trapped_idx=var_idxs[trapped_local],
            leaky_idx=var_idxs[leaky_local],
            device=device,
            dtype=dtype,
            n_points=n_points,
            kind=kind,
            candidate_P=candidate_P,
        )
        if res.ok and (best is None or res.rel_res < best.rel_res):
            best = res
    return best


def _collect_subtree_separability_targets(root: Node) -> List[Node]:
    """
    Collect inner subtrees that are promising candidates for SubtreeSeparability
    analysis. For now we target:

      - bases of PowNode with exponent ±1/2 (sqrt or 1/sqrt),
      - arguments of LogNode.
    """
    targets: List[Node] = []

    def _walk(n: Node):
        if isinstance(n, PowNode):
            e = abs(float(n.exponent))
            if e in (0.5, 2.0):
                targets.append(n.base)

        if isinstance(n, LogNode):
            targets.append(n.arg)

        if isinstance(n, (SinNode, CosNode, AsinNode, AcosNode, AtanNode, ExpNode)):
            targets.append(n.arg)

        if isinstance(n, (AddNode, MulNode)):
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, PowNode):
            _walk(n.base)
        elif isinstance(n, (LogNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ExpNode)):
            _walk(n.arg)

    _walk(root)
    return targets


def _infer_nn_hyperparams_from_root(
    root: Node,
    default_num_segments: int = 32,
    default_dual_layer: bool = False,
) -> Tuple[int, bool]:
    """
    Infer (num_segments, dual_layer) for 'nn' atoms from the current AST,
    so that any new 'nn' atoms we introduce are consistent with Stage A.
    """
    atoms = collect_nn_atoms(root)
    for a in atoms:
        kw = a.kwargs or {}
        return int(kw.get("num_segments", default_num_segments)), bool(
            kw.get("dual_layer", default_dual_layer)
        )
    return default_num_segments, default_dual_layer


def _build_subtree_separability_candidate(
    root: Node,
    u_node: Node,
    model: nn.Module,
    reuse: Dict[str, torch.nn.Module],
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    very_verbose: bool = False,
) -> Tuple[Optional[Node], Optional[Callable]]:
    """
    Use SubtreeSeparability to propose a separability rewrite for a subtree
    u_node. If the hook reports a split between variable groups A,B, we rewrite
    that subtree using separability_proposal_to_ast and return a new root.

    This now applies to *any* subtree involving ≥2 variables, irrespective
    of whether it currently contains NN or analytic leaves. New separated
    factors are represented as fresh 'nn' atoms; Stage B can then simplify
    those NN leaves with its usual analytic rewrite patterns.
    """
    if run_subtree_separability is None:
        return None, None

    var_indices = _vars_in_subtree(u_node)
    if len(var_indices) < 2:
        return None, None

    # Heuristic guard: if this subtree already mixes NN and analytic leaves,
    # we *don't* want to replace it by fresh NN factors, because that would
    # throw away Stage‑B’s previous simplifications (e.g. 1D poly rewrites).
    #
    # Example: u_node = poly(x1) + nn(x2)
    #   kinds_here = {"poly", "nn"}  -> skip local Stage‑A here.
    #
    # We still allow local Stage‑A on:
    #   - pure NN blobs (kinds == {"nn"})
    #   - pure analytic blobs (kinds has no "nn")
    kinds_here = _subtree_leaf_kinds(u_node)
    if kinds_here and ("nn" in kinds_here) and (len(kinds_here) > 1):
        return None, None

    try:
        atom_to_leaf = build_atom_to_leaf_map(root, model)
    except Exception:
        return None, None

    # Wrap the subtree in a _SubtreeModel that reuses the *current* Stage‑B
    # leaf modules and exposes analytic input derivatives via the AST chain
    # rules. This keeps SubtreeSeparability consistent with the fitted model.
    model_u = _SubtreeModel(
        root=u_node,
        atom_to_leaf=atom_to_leaf,
    )

    res = run_subtree_separability(
        model_u=model_u,
        datagen=train_loader,
        var_indices=var_indices,
        device=device,
        dtype=dtype,
        very_verbose=very_verbose,
    )
    if res is None:
        return None, None

    op, group1_local, group2_local = res
    if not group1_local or not group2_local:
        return None, None

    group1_global = [var_indices[i] for i in group1_local]
    group2_global = [var_indices[i] for i in group2_local]

    # Check if u_node is a polynomial atom and use specialized split with coefficient transfer
    if isinstance(u_node, AtomNode) and str(u_node.kind).lower() in ("poly", "polynomial", "rpoly", "rpolynomial", "r_polynomial"):
        poly_split = _build_poly_split_from_subtree_separability(
            root=root,
            u_node=u_node,
            model=model,
            op=op,
            group1_global=group1_global,
            group2_global=group2_global,
            device=device,
            dtype=dtype,
        )
        if poly_split[0] is not None:  # If split succeeded
            return poly_split
        # Fall through to NN-based split if polynomial split fails

    num_segments, dual_layer = _infer_nn_hyperparams_from_root(root)

    # Pass parent tag if u_node is an AtomNode with a tag (for Stage A reuse)
    parent_tag = u_node.tag if isinstance(u_node, AtomNode) else None

    new_subtree = separability_proposal_to_ast(
        op,
        group1_global,
        group2_global,
        num_segments=num_segments,
        dual_layer=dual_layer,
        parent_tag=parent_tag,
        parent_atom=u_node if isinstance(u_node, AtomNode) else None,
    )

    cand_root = replace_atom_in_ast(root, u_node, new_subtree)
    return cand_root, None


# -----------------------------------------------------------------------------
# Counterterm-based multiplicative splits (u(x) = P(x_A) + g(x_A) h(x_B))
# -----------------------------------------------------------------------------


def _subtree_leaf_kinds(node: Node) -> Set[str]:
    """
    Return the set of .kind strings for AtomNodes under 'node'.
    """
    atoms = _collect_all_atoms(node)
    return {str(a.kind).lower() for a in atoms}


def _collect_pure_analytic_subtree_separability_subtrees(
    root: Node,
    model: nn.Module,
    max_subtrees: int = 4,
) -> List[Tuple[Node, List[int], ASTCompositeAdaptor]]:
    """
    Find *pure analytic* subtrees (no 'nn' atoms) to feed to
    run_subtree_separability.

    Returns a list of (subroot, var_indices, model_u), where:
      - subroot     : Node (root of the analytic subtree)
      - var_indices : sorted list of global x-indices used by that subtree
      - model_u     : ASTCompositeAdaptor sharing the fitted leaf modules
                      from the current Stage-B model.

    We deliberately:
      - skip the global root node,
      - require the subtree to involve at least 2 distinct variables,
      - require that all AtomNodes under the subtree are non-'nn'.
    """
    atom_to_leaf = build_atom_to_leaf_map(root, model)
    out: List[Tuple[Node, List[int], ASTCompositeAdaptor]] = []

    def visit(node: Node, is_root: bool) -> Set[int]:
        nonlocal out

        if isinstance(node, AtomNode):
            return {int(j) for j in node.var_idxs}

        child_vars: Set[int] = set()
        if isinstance(node, AddNode) or isinstance(node, MulNode):
            child_vars |= visit(node.left, False)
            child_vars |= visit(node.right, False)
        elif isinstance(node, PowNode):
            child_vars |= visit(node.base, False)
        elif isinstance(node, (LogNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ExpNode)):
            child_vars |= visit(node.arg, False)
        else:
            return set()

        # --- patched block: allow the root as a candidate once ---
        is_candidate = len(child_vars) >= 2 and len(out) < max_subtrees
        if is_candidate and ((not is_root) or not out):
            kinds = _subtree_leaf_kinds(node)
            if kinds and ("nn" not in kinds):
                var_indices = sorted(child_vars)

                atoms_sub = _collect_all_atoms(node)
                leaves_sub: List[nn.Module] = []
                missing = False
                for a in atoms_sub:
                    leaf = atom_to_leaf.get(id(a), None)
                    if leaf is None:
                        missing = True
                        break
                    leaves_sub.append(leaf)

                if not missing:
                    try:
                        model_u = ASTCompositeAdaptor(node, leaves_sub)
                    except Exception as e:
                        print(
                            "[SubtreeSeparability analytic] Failed to build ASTCompositeAdaptor:", e
                        )
                    else:
                        out.append((node, var_indices, model_u))

        return child_vars

    visit(root, is_root=True)
    return out


def _collect_composite_subtree_separability_subtrees(
    root: Node,
    model: nn.Module,
    max_subtrees: int = 4,
) -> List[Tuple[Node, List[int], ASTCompositeAdaptor]]:
    """
    Find composite subtrees (NN + analytic mixtures) to feed to
    run_subtree_separability.

    Returns a list of (subroot, var_indices, model_u), where:
      - subroot     : Node (root of the subtree in the AST)
      - var_indices : sorted list of global x-indices used by that subtree
      - model_u     : ASTCompositeAdaptor implementing u(x) for that subtree,
                      sharing the fitted leaf modules from the current Stage‑B
                      model and exposing analytic grad/grad_grad w.r.t. inputs.

    We deliberately:
      - skip the global root node,
      - require the subtree to contain at least one 'nn' leaf AND at
        least one non-'nn' leaf (analytic mixture),
      - require at least 2 distinct variables.
    """
    atom_to_leaf = build_atom_to_leaf_map(root, model)
    out: List[Tuple[Node, List[int], ASTCompositeAdaptor]] = []

    def visit(node: Node, is_root: bool) -> Set[int]:
        nonlocal out

        if isinstance(node, AtomNode):
            return {int(j) for j in node.var_idxs}

        child_vars: Set[int] = set()
        if isinstance(node, (AddNode, MulNode)):
            child_vars |= visit(node.left, False)
            child_vars |= visit(node.right, False)
        elif isinstance(node, PowNode):
            child_vars |= visit(node.base, False)
        elif isinstance(node, (LogNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ExpNode)):
            child_vars |= visit(node.arg, False)
        else:
            return set()

        if (not is_root) and len(child_vars) >= 2 and len(out) < max_subtrees:
            kinds = _subtree_leaf_kinds(node)
            if "nn" in kinds and len(kinds) > 1:
                var_indices = sorted(child_vars)
                atoms_sub = _collect_all_atoms(node)
                leaves_sub: List[nn.Module] = []
                missing = False
                for a in atoms_sub:
                    leaf = atom_to_leaf.get(id(a), None)
                    if leaf is None:
                        missing = True
                        break
                    leaves_sub.append(leaf)
                if not missing:
                    try:
                        model_u = ASTCompositeAdaptor(node, leaves_sub)
                    except Exception as e:
                        print("[local StageA] Failed to build ASTCompositeAdaptor:", e)
                    else:
                        out.append((node, var_indices, model_u))

        return child_vars

    visit(root, is_root=True)
    return out


def _build_gauge_split_candidates(
    root: Node,
    target: AtomNode,
    unique_vars: Set[int],
    shared_vars: Set[int],
    context: str = "additive",
    num_segments: int = 16,
    dual_layer: bool = False,
) -> List[Tuple[str, Node, dict]]:
    """
    Build split candidates for gauge context.

    When an NN atom has unique variables (not shared with siblings) and shared
    variables, we directly propose splits without running separability detection,
    because gauge freedom between siblings masks the true structure.

    The operation order depends on the context:
    - Additive context (NN + NN): gauge masks multiplicative separability
      → propose multiplicative first, then additive
    - Multiplicative context (NN * NN): gauge masks additive separability
      → propose additive first, then multiplicative

    Args:
        root: AST root node
        target: The multivariate NN atom to split
        unique_vars: Variables unique to target (not in any sibling)
        shared_vars: Variables shared with at least one sibling
        context: Either "additive" or "multiplicative" - describes the sibling relationship
        num_segments: Number of segments for new NN atoms
        dual_layer: Whether to use dual-layer architecture

    Returns:
        List of (label, new_root, meta) tuples. Operation order depends on context.
    """
    candidates = []
    unique_list = sorted(unique_vars)
    shared_list = sorted(shared_vars)

    # Get parent tag for deterministic child tags
    parent_tag = target.tag

    # Determine operation order based on context
    if context == "additive":
        # Additive gauge masks multiplicative separability
        # Try multiplicative first, then additive
        ops = [
            (torch.multiply, "gauge_mul_split", "*"),
            (torch.add, "gauge_add_split", "+"),
        ]
    else:  # multiplicative context
        # Multiplicative gauge masks additive separability
        # Try additive first, then multiplicative
        ops = [
            (torch.add, "gauge_add_split", "+"),
            (torch.multiply, "gauge_mul_split", "*"),
        ]

    for op_fn, label, op_sym in ops:
        subtree = separability_proposal_to_ast(
            op_fn,
            unique_list,
            shared_list,
            num_segments=num_segments,
            dual_layer=dual_layer,
            parent_tag=parent_tag,
            parent_atom=target,
        )
        new_root = replace_atom_in_ast(root, target, subtree)
        candidates.append(
            (
                label,
                new_root,
                {
                    "log": f"[Stage B]  Trying gauge {op_sym}: NN{unique_list} {op_sym} NN{shared_list}",
                    "structural": True,
                },
            )
        )

    return candidates
