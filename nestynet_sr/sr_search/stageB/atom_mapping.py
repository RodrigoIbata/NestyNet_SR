# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Atom-to-leaf mapping and atom collection utilities for Stage B.

This module provides functions for traversing ASTs and building mappings
between AtomNodes and their corresponding leaf modules.
"""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn

from nestynet_sr.sr_core.bridges import (
    UNARY_NODE_TYPES,
    AddNode,
    AtomNode,
    ConstNode,
    MulNode,
    Node,
    PowNode,
    effective_arity,
    is_problem_atom,
)


def _debug_atom_labels(atoms: List[AtomNode], *, limit: int = 12) -> List[str]:
    """Compact atom labels for mismatch diagnostics."""
    out = [f"{a.kind}({getattr(a, 'tag', None)})" for a in atoms[:max(0, int(limit))]]
    if len(atoms) > int(limit):
        out.append(f"...+{len(atoms) - int(limit)} more")
    return out


def build_atom_to_leaf_map(
    root: Node,
    model: nn.Module,  # typically an ASTCompositeAdaptor
) -> Dict[int, nn.Module]:
    """
    Build a mapping from AtomNode object identity (via id) to its
    leaf module in the current Stage-B model.

    This function uses a robust approach that collects atoms in the same
    DFS order that build_composite_from_ast used, eliminating fragile
    order assumptions.

    Note: For new code, prefer using build_composite_from_ast(..., return_atom_map=True)
    which returns the mapping directly during construction. This function is
    kept for compatibility with existing code.

    Parameters
    ----------
    root : Node
        The AST used to build the model.
    model : nn.Module
        The model (typically ASTCompositeAdaptor) with a .leaf attribute.

    Returns
    -------
    atom_to_leaf : dict[int, nn.Module]
        Mapping from id(atom) -> leaf module.
    """
    atoms = _collect_all_atoms(root)
    leaves = getattr(model, "leaf", None)
    if leaves is None:
        raise TypeError("build_atom_to_leaf_map expects model with a .leaf attribute")

    leaves = list(leaves)
    if len(atoms) != len(leaves):
        print(
            f"[build_atom_to_leaf_map] Warning: len(atoms)={len(atoms)} != len(model.leaf)={len(leaves)}; mapping may be incorrect."
        )
        try:
            model_root = getattr(model, "ast_root", None)
            root_is_model_root = model_root is root
            print(
                f"[build_atom_to_leaf_map] root_is_model_root={root_is_model_root}; "
                f"requested_root_atoms={_debug_atom_labels(atoms)}"
            )
            if model_root is not None:
                try:
                    model_root_atoms = _collect_all_atoms(model_root)
                    print(
                        f"[build_atom_to_leaf_map] model.ast_root stageB_atom_count={len(model_root_atoms)}; "
                        f"atoms={_debug_atom_labels(model_root_atoms)}"
                    )
                except Exception as exc:
                    print(f"[build_atom_to_leaf_map] model.ast_root stageB atom collection failed: {exc}")
                try:
                    collect_atoms = getattr(model, "_collect_atoms", None)
                    if callable(collect_atoms):
                        model_root_atoms_adaptor = collect_atoms(model_root)
                        print(
                            f"[build_atom_to_leaf_map] model._collect_atoms(ast_root)={len(model_root_atoms_adaptor)}"
                        )
                except Exception as exc:
                    print(f"[build_atom_to_leaf_map] adaptor atom collection failed: {exc}")
        except Exception as exc:
            print(f"[build_atom_to_leaf_map] mismatch diagnostics failed: {exc}")

    return {id(atom): leaf for atom, leaf in zip(atoms, leaves)}


def _collect_univariate_nn_atoms(root: Node) -> List[AtomNode]:
    """Collect all NN atoms with effective arity 1.

    This includes both true univariate atoms (single var_idx) and compound
    atoms with input_expr (which operate on a scalar compound variable).
    """
    atoms: List[AtomNode] = []

    def _walk(node: Node):
        if isinstance(node, AtomNode):
            if (
                node.kind.lower() == "nn"
                and effective_arity(node) == 1
                and not is_problem_atom(node)
            ):
                atoms.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, PowNode):
            _walk(node.base)
        elif isinstance(node, UNARY_NODE_TYPES):
            _walk(node.arg)
        elif isinstance(node, ConstNode):
            pass  # Constants have no atoms
        else:
            raise TypeError(f"Unexpected node type in AST: {type(node)}")

    _walk(root)
    return atoms


def _refresh_reuse_from_state(root: Node, model: nn.Module) -> Dict[str, nn.Module]:
    """
    Refresh the reuse-map from the currently fitted Stage-B model.
    Keys are AtomNode.tag, values are the corresponding fitted leaf modules.
    """
    try:
        atom_to_leaf = build_atom_to_leaf_map(root, model)
    except Exception:
        return {}
    reuse_new: Dict[str, nn.Module] = {}
    for a in _collect_all_atoms(root):
        if isinstance(a, AtomNode) and getattr(a, "tag", None):
            leaf = atom_to_leaf.get(id(a), None)
            if leaf is not None:
                reuse_new[str(a.tag)] = leaf
    return reuse_new


def _collect_multivariate_nn_atoms(root: Node) -> List[AtomNode]:
    """
    Collect NN atoms with effective arity >= 2. These are
    candidates for multi-dimensional exp-rational rewrites.

    Note: Compound atoms (with input_expr) have effective_arity=1 even if
    var_idxs has multiple elements, so they won't be collected here.
    """
    atoms: List[AtomNode] = []

    def _walk(node: Node):
        if isinstance(node, AtomNode):
            if (
                node.kind.lower() == "nn"
                and effective_arity(node) >= 2
                and not is_problem_atom(node)
            ):
                atoms.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, PowNode):
            _walk(node.base)
        elif isinstance(node, UNARY_NODE_TYPES):
            _walk(node.arg)

    _walk(root)
    return atoms


def _collect_multivariate_poly_atoms(root: Node) -> List[AtomNode]:
    """
    Collect analytic polynomial atoms that depend on 2 or more variables.
    These typically arise after a NN leaf has been rewritten to a PolyLeaf
    (possibly under a PowNode).
    """
    atoms: List[AtomNode] = []

    def _walk(node: Node):
        if isinstance(node, AtomNode):
            # Use effective_arity to properly handle compound atoms
            if node.kind.lower() in ("poly", "polynomial", "rpoly", "rpolynomial", "r_polynomial") and effective_arity(node) >= 2:
                atoms.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, PowNode):
            _walk(node.base)
        elif isinstance(node, UNARY_NODE_TYPES):
            _walk(node.arg)

    _walk(root)
    return atoms


def _collect_all_atoms(root: Node) -> List[AtomNode]:
    """Collect all atom nodes from an AST in DFS order."""
    atoms: List[AtomNode] = []

    def _walk(node: Node):
        if isinstance(node, AtomNode):
            atoms.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, PowNode):
            _walk(node.base)
        elif isinstance(node, UNARY_NODE_TYPES):
            _walk(node.arg)
        elif isinstance(node, ConstNode):
            pass  # Constants are not atoms
        else:
            raise TypeError(f"Unexpected node type in AST: {type(node)}")

    _walk(root)
    return atoms


def _vars_in_subtree(node: Node) -> List[int]:
    """
    Collect the set of global variable indices used anywhere inside `node`.
    Returns a sorted list of unique variable indices.
    """
    s: set[int] = set()

    def _walk(n: Node):
        if isinstance(n, AtomNode):
            for j in n.var_idxs:
                s.add(int(j))
        elif isinstance(n, (AddNode, MulNode)):
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, PowNode):
            _walk(n.base)
        elif isinstance(n, UNARY_NODE_TYPES):
            _walk(n.arg)
        elif isinstance(n, ConstNode):
            pass  # Constants have no variables
        else:
            raise TypeError(f"Unexpected node type in AST: {type(n)}")

    _walk(node)
    return sorted(s)


def _collect_nn_atoms_under_add_tree(node: Node) -> List[AtomNode]:
    """
    Collect all NN atoms under a connected AddNode tree.

    Traverses through AddNodes but stops at MulNodes and other non-Add nodes.
    This finds all NN atoms that are "additive siblings" - connected by addition only.

    Example: Add(Add(NN1, NN2), Mul(NN3, NN4)) -> [NN1, NN2] (not NN3, NN4)
    """
    nns: List[AtomNode] = []

    def _walk_add_tree(n: Node):
        if isinstance(n, AddNode):
            # Continue traversing through AddNodes
            _walk_add_tree(n.left)
            _walk_add_tree(n.right)
        elif isinstance(n, AtomNode):
            # Leaf: check if it's an NN
            if str(n.kind).lower() == "nn":
                nns.append(n)
        # For MulNode, PowNode, etc. - don't traverse further
        # (they break the additive chain)

    _walk_add_tree(node)
    return nns


def _find_nns_in_add_chain(root: Node, target: AtomNode) -> List[AtomNode]:
    """
    Find all NN atoms in the same additive chain as target.

    An NN is in the same additive chain if it's connected to target through
    AddNodes only. This handles nested Add structures correctly:
    Add(Add(NN1, NN2), NN3) -> all three are in the same chain.

    The key insight is that gauge freedom exists between NNs connected by
    addition: any function of shared variables can drift between them.

    Args:
        root: AST root node
        target: The NN atom we're querying about

    Returns:
        List of NN atoms in the same additive chain (excludes target itself).
        Empty list if target is not under an AddNode.
    """
    # Strategy: Find the highest AddNode that contains target,
    # then collect all NN atoms under that AddNode.

    def _find_containing_add_and_collect(node: Node, parent_add: Node = None):
        """
        Walk AST to find target. Track the highest AddNode ancestor.
        When target is found, collect NNs from that AddNode.
        Returns (found, siblings) tuple.
        """
        if node is target:
            # Found target - if we have a parent AddNode, collect siblings
            if parent_add is not None:
                all_nns = _collect_nn_atoms_under_add_tree(parent_add)
                # Exclude target itself
                siblings = [nn for nn in all_nns if nn is not target]
                return True, siblings
            return True, []

        if isinstance(node, AddNode):
            # If we're in an AddNode, this becomes the new "highest Add" for children
            # But we need to track the HIGHEST Add, so only update if parent_add is None
            # or if we're continuing an Add chain
            new_parent = parent_add if parent_add is not None else node
            # Actually, we want the ROOT of the connected Add tree
            # So we pass down the current Add if parent_add was already an Add's child,
            # otherwise start a new Add tree
            if parent_add is None or not isinstance(parent_add, AddNode):
                new_parent = node

            found, sibs = _find_containing_add_and_collect(node.left, new_parent)
            if found:
                return True, sibs
            found, sibs = _find_containing_add_and_collect(node.right, new_parent)
            if found:
                return True, sibs
            return False, []

        elif isinstance(node, MulNode):
            # MulNode breaks the Add chain - start fresh for each subtree
            found, sibs = _find_containing_add_and_collect(node.left, None)
            if found:
                return True, sibs
            found, sibs = _find_containing_add_and_collect(node.right, None)
            if found:
                return True, sibs
            return False, []

        elif isinstance(node, PowNode):
            return _find_containing_add_and_collect(node.base, None)

        elif isinstance(node, UNARY_NODE_TYPES):
            return _find_containing_add_and_collect(node.arg, None)

        elif isinstance(node, AtomNode):
            return False, []

        return False, []

    found, siblings = _find_containing_add_and_collect(root, None)
    return siblings if found else []


def _collect_nn_atoms_under_mul_tree(node: Node) -> List[AtomNode]:
    """
    Collect all NN atoms under a connected MulNode tree.

    Traverses through MulNodes but stops at AddNodes and other non-Mul nodes.
    This finds all NN atoms that are "multiplicative siblings" - connected by multiplication only.

    Example: Mul(Mul(NN1, NN2), Add(NN3, NN4)) -> [NN1, NN2] (not NN3, NN4)
    """
    nns: List[AtomNode] = []

    def _walk_mul_tree(n: Node):
        if isinstance(n, MulNode):
            # Continue traversing through MulNodes
            _walk_mul_tree(n.left)
            _walk_mul_tree(n.right)
        elif isinstance(n, AtomNode):
            # Leaf: check if it's an NN
            if str(n.kind).lower() == "nn":
                nns.append(n)
        # For AddNode, PowNode, etc. - don't traverse further
        # (they break the multiplicative chain)

    _walk_mul_tree(node)
    return nns


def _find_nns_in_mul_chain(root: Node, target: AtomNode) -> List[AtomNode]:
    """
    Find all NN atoms in the same multiplicative chain as target.

    An NN is in the same multiplicative chain if it's connected to target through
    MulNodes only. This handles nested Mul structures correctly:
    Mul(Mul(NN1, NN2), NN3) -> all three are in the same chain.

    The key insight is that gauge freedom exists between NNs connected by
    multiplication: any function of shared variables can drift multiplicatively.

    Args:
        root: AST root node
        target: The NN atom we're querying about

    Returns:
        List of NN atoms in the same multiplicative chain (excludes target itself).
        Empty list if target is not under a MulNode.
    """
    # Strategy: Find the highest MulNode that contains target,
    # then collect all NN atoms under that MulNode.

    def _find_containing_mul_and_collect(node: Node, parent_mul: Node = None):
        """
        Walk AST to find target. Track the highest MulNode ancestor.
        When target is found, collect NNs from that MulNode.
        Returns (found, siblings) tuple.
        """
        if node is target:
            # Found target - if we have a parent MulNode, collect siblings
            if parent_mul is not None:
                all_nns = _collect_nn_atoms_under_mul_tree(parent_mul)
                # Exclude target itself
                siblings = [nn for nn in all_nns if nn is not target]
                return True, siblings
            return True, []

        if isinstance(node, MulNode):
            # If we're in a MulNode, this becomes the new "highest Mul" for children
            # But we need to track the HIGHEST Mul, so only update if parent_mul is None
            # or if we're continuing a Mul chain
            new_parent = parent_mul if parent_mul is not None else node
            # Actually, we want the ROOT of the connected Mul tree
            # So we pass down the current Mul if parent_mul was already a Mul's child,
            # otherwise start a new Mul tree
            if parent_mul is None or not isinstance(parent_mul, MulNode):
                new_parent = node

            found, sibs = _find_containing_mul_and_collect(node.left, new_parent)
            if found:
                return True, sibs
            found, sibs = _find_containing_mul_and_collect(node.right, new_parent)
            if found:
                return True, sibs
            return False, []

        elif isinstance(node, AddNode):
            # AddNode breaks the Mul chain - start fresh for each subtree
            found, sibs = _find_containing_mul_and_collect(node.left, None)
            if found:
                return True, sibs
            found, sibs = _find_containing_mul_and_collect(node.right, None)
            if found:
                return True, sibs
            return False, []

        elif isinstance(node, PowNode):
            return _find_containing_mul_and_collect(node.base, None)

        elif isinstance(node, UNARY_NODE_TYPES):
            return _find_containing_mul_and_collect(node.arg, None)

        elif isinstance(node, AtomNode):
            return False, []

        return False, []

    found, siblings = _find_containing_mul_and_collect(root, None)
    return siblings if found else []
