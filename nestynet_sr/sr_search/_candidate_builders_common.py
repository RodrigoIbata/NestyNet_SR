# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Shared AST and teacher-data helpers for Stage-B candidate builders."""

from typing import Callable, List, Optional, Tuple

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    SinNode,
    ast_equals,
    clone_ast,
    const_full_like,
    eval_inputs,
    get_input_exprs,
)

def _unwrap_leaf_core(leaf_mod: torch.nn.Module) -> torch.nn.Module:
    """Return the analytic core for a leaf wrapper."""
    return getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))


def _atom_inputs_match(atom: AtomNode, expected_inputs: Tuple[Node, ...]) -> bool:
    """Return True when an atom sees the same effective inputs."""
    atom_inputs = get_input_exprs(atom)
    if len(atom_inputs) != len(expected_inputs):
        return False
    for atom_inp, expected_inp in zip(atom_inputs, expected_inputs):
        try:
            if not ast_equals(atom_inp, expected_inp):
                return False
        except Exception:
            return False
    return True


def _find_matching_core(
    atoms: List[Node],
    leaves: List[torch.nn.Module],
    *,
    core_types,
    expected_kind: Optional[str] = None,
    expected_tag: Optional[str] = None,
    expected_inputs: Optional[Tuple[Node, ...]] = None,
    predicate: Optional[Callable[[AtomNode, torch.nn.Module], bool]] = None,
):
    """Locate a leaf core by tag or effective-input signature."""
    if not isinstance(core_types, tuple):
        core_types = (core_types,)

    kind_norm = str(expected_kind).lower() if expected_kind is not None else None
    for atom_i, leaf_mod in zip(atoms, leaves):
        if not isinstance(atom_i, AtomNode):
            continue
        core = _unwrap_leaf_core(leaf_mod)
        if not isinstance(core, core_types):
            continue
        if kind_norm is not None and str(getattr(atom_i, "kind", "")).lower() != kind_norm:
            continue
        if expected_tag is not None and getattr(atom_i, "tag", None) != expected_tag:
            continue
        if expected_inputs is not None and not _atom_inputs_match(atom_i, expected_inputs):
            continue
        if predicate is not None and not predicate(atom_i, core):
            continue
        return core
    return None


def _single_power_coordinate_inputs(atom: AtomNode, exponent: float) -> Optional[Tuple[Node, ...]]:
    """Return a single explicit coordinate input ``z**exponent`` for a 1D atom."""
    try:
        inputs = get_input_exprs(atom)
    except Exception:
        return None
    if len(inputs) != 1:
        return None
    return (PowNode(clone_ast(inputs[0]), float(exponent)),)


def _support_is_valid(support: Optional[torch.Tensor], exps_dense: torch.Tensor) -> bool:
    if support is None or int(support.numel()) <= 0:
        return False
    if exps_dense.ndim != 2 or int(exps_dense.shape[0]) <= 0:
        return False
    if int(support.min().item()) < 0:
        return False
    if int(support.max().item()) >= int(exps_dense.shape[0]):
        return False
    return True


def _max_total_degree_from_exps(exps: Optional[torch.Tensor], fallback: int = 0) -> int:
    if exps is None or exps.ndim != 2 or int(exps.shape[0]) <= 0:
        return int(fallback)
    return int(exps.sum(dim=1).max().item())


def _exps_override_from_tensor(exps: Optional[torch.Tensor]) -> Optional[List[List[int]]]:
    if exps is None or exps.ndim != 2 or int(exps.shape[0]) <= 0:
        return None
    return [[int(v) for v in row] for row in exps.detach().cpu().tolist()]


def _exps_key(exps: Optional[torch.Tensor]) -> Optional[Tuple[Tuple[int, ...], ...]]:
    if exps is None or exps.ndim != 2 or int(exps.shape[0]) <= 0:
        return None
    return tuple(tuple(int(v) for v in row) for row in exps.detach().cpu().tolist())


def _select_clear_rratpoly_pivot(
    exps_num: Optional[torch.Tensor],
    coeffs_num: Optional[torch.Tensor],
    *,
    dominance_ratio: float = 5.0,
    rel_floor: float = 0.1,
) -> Tuple[Optional[int], Optional[str]]:
    if (
        exps_num is None
        or coeffs_num is None
        or exps_num.ndim != 2
        or coeffs_num.ndim != 1
        or int(exps_num.shape[0]) <= 0
        or int(exps_num.shape[0]) != int(coeffs_num.numel())
    ):
        return None, None

    degs = exps_num.sum(dim=1)
    max_deg = int(degs.max().item())
    highest = torch.nonzero(degs == max_deg, as_tuple=False).view(-1)
    if int(highest.numel()) <= 0:
        return None, None
    if int(highest.numel()) == 1:
        return int(highest.item()), "unique-highest-degree"

    mags = coeffs_num.abs().to(dtype=torch.float64)
    mags_hi = mags[highest]
    order = torch.argsort(mags_hi, descending=True)
    best_local = int(order[0].item())
    best_idx = int(highest[best_local].item())
    best_mag = float(mags_hi[best_local].item())
    second_mag = float(mags_hi[int(order[1].item())].item()) if int(order.numel()) > 1 else 0.0
    max_mag = float(mags.max().item()) if int(mags.numel()) > 0 else 0.0

    if best_mag <= 1e-12:
        return None, None
    if best_mag >= float(dominance_ratio) * max(second_mag, 1e-12) and best_mag >= float(rel_floor) * max(max_mag, 1e-12):
        return best_idx, "dominant-highest-degree"
    return None, None


def _move_sparse_pivot_to_end(
    exps_num: torch.Tensor,
    coeffs_num: torch.Tensor,
    pivot_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(pivot_idx) < 0 or int(pivot_idx) >= int(exps_num.shape[0]):
        return exps_num, coeffs_num
    if int(pivot_idx) == int(exps_num.shape[0]) - 1:
        return exps_num, coeffs_num
    keep = [i for i in range(int(exps_num.shape[0])) if i != int(pivot_idx)] + [int(pivot_idx)]
    perm = torch.tensor(keep, dtype=torch.int64, device=exps_num.device)
    return exps_num[perm].clone(), coeffs_num[perm].clone()


def _select_sign_region(
    F: torch.Tensor,
    min_points: int,
    eps: float,
):
    """
    Choose a sign-consistent subset of samples for sqrt/exp-style analysis.

    Returns
    -------
    mask : BoolTensor | None
        Boolean mask selecting samples to use.
    sign : float | None
        +1.0 or -1.0 such that sign * F[mask] > 0.
    """
    if F.numel() == 0:
        return None, None

    m_pos = F > eps
    m_neg = F < -eps
    n_pos = int(m_pos.sum().item())
    n_neg = int(m_neg.sum().item())

    # Prefer the dominant sign if it has enough support.
    if n_pos >= min_points and n_pos >= n_neg:
        return m_pos, 1.0
    if n_neg >= min_points and n_neg > n_pos:
        return m_neg, -1.0

    # Not clearly sign-definite (or too few points of either sign)
    return None, None


def _parse_pure_difference_expr(expr: Node) -> Optional[Tuple[int, int]]:
    """Check if expr is a pure difference: z = Var(i) - Var(j).

    Returns (i, j) if expr represents x_i - x_j, else None.

    The expected AST forms are:
      - Add(Var(i), Mul(ConstNode(-1.0), Var(j)))  [from build_linear_ast]
      - Add(Mul(ConstNode(-1.0), Var(j)), Var(i))  [symmetric]
    """
    if not isinstance(expr, AddNode):
        return None

    left, right = expr.left, expr.right

    # Check pattern: Add(Var(i), Mul(ConstNode(-1), Var(j)))
    def _is_var(node: Node) -> Optional[int]:
        if isinstance(node, AtomNode):
            kind = str(getattr(node, 'kind', '')).lower()
            if kind in ('var', 'x', 'input') and len(node.var_idxs) == 1:
                return int(node.var_idxs[0])
        return None

    def _is_neg_var(node: Node) -> Optional[int]:
        if isinstance(node, MulNode):
            # Mul(ConstNode(-1), Var(j)) or Mul(Var(j), ConstNode(-1))
            if isinstance(node.left, ConstNode) and abs(node.left.value + 1.0) < 1e-12:
                return _is_var(node.right)
            if isinstance(node.right, ConstNode) and abs(node.right.value + 1.0) < 1e-12:
                return _is_var(node.left)
        return None

    # Pattern 1: Add(Var(i), Mul(ConstNode(-1), Var(j)))
    i = _is_var(left)
    j = _is_neg_var(right)
    if i is not None and j is not None:
        return (i, j)

    # Pattern 2: Add(Mul(ConstNode(-1), Var(j)), Var(i))
    j = _is_neg_var(left)
    i = _is_var(right)
    if i is not None and j is not None:
        return (i, j)

    return None


def _eval_input_expr_value(expr: Node, x: "torch.Tensor") -> "torch.Tensor":
    """Evaluate a structural AST (input_expr) on x.

    Returns a 1D tensor of shape [B]. Supports Var/Add/Mul/Pow and common unary ops.
    """
    import torch

    def rec(node: Node) -> torch.Tensor:
        if isinstance(node, AtomNode):
            kind = str(getattr(node, 'kind', '')).lower()
            if kind in ('var', 'x', 'input'):
                if len(node.var_idxs) != 1:
                    raise ValueError('Var node in input_expr must have exactly 1 var_idx')
                j = int(node.var_idxs[0])
                return x[:, j]
            raise ValueError(f"Unsupported atom kind '{kind}' in input_expr")
        if isinstance(node, AddNode):
            return rec(node.left) + rec(node.right)
        if isinstance(node, MulNode):
            return rec(node.left) * rec(node.right)
        if isinstance(node, PowNode):
            return rec(node.base) ** float(node.exponent)
        if isinstance(node, SinNode):
            return torch.sin(rec(node.arg))
        if isinstance(node, CosNode):
            return torch.cos(rec(node.arg))
        if isinstance(node, ExpNode):
            return torch.exp(rec(node.arg))
        if isinstance(node, LogNode):
            return torch.log(rec(node.arg))
        if isinstance(node, ConstNode):
            return const_full_like(x, (x.shape[0],), node.value)
        raise ValueError(f'Unsupported node type in input_expr: {type(node)}')

    return rec(expr)


def _build_atom_input_tensor(atom: AtomNode, x_full: "torch.Tensor") -> "torch.Tensor":
    """Build the input tensor for an AtomNode leaf.

    Uses the unified eval_inputs() for both simple and compound atoms.
    For compound atoms, returns shape [B, 1+len(extra_var_idxs)] = [z, extras...].
    For simple atoms, returns shape [B, len(var_idxs)] = x[:, var_idxs].
    """
    if atom.n_in > 0:
        x_in, _, _ = eval_inputs(atom, x_full, need_grad=False, need_hess=False)
        return x_in
    else:
        # Fallback for atoms with no inputs
        cols = [int(j) for j in atom.var_idxs]
        return x_full[:, cols] if cols else x_full


def _gather_atom_teacher_data(
    train_loader,
    atom: AtomNode,
    teacher: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    max_points: int = 5000,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Gather (X_atom, f_teacher) pairs for a given atom.

    For standard atoms, X_atom is x[:, atom.var_idxs]. For compound atoms
    (kwargs['input_expr']), X_atom is the *leaf input* [z, extras...] with
    z=eval(input_expr).
    """
    xs: List[torch.Tensor] = []
    fs: List[torch.Tensor] = []
    n_collected = 0

    teacher.eval()

    for batch in train_loader:
        if isinstance(batch, (list, tuple)):
            x, _ = batch
        else:
            x = batch
        x = x.to(device=device, dtype=dtype)
        x_sub = _build_atom_input_tensor(atom, x)
        with torch.no_grad():
            f = teacher(x_sub)
            if f.dim() == 2:
                f = f[:, 0]
            else:
                f = f.view(-1)
        xs.append(x_sub.detach().cpu())
        fs.append(f.detach().cpu())
        n_collected += x_sub.size(0)
        if n_collected >= max_points:
            break
    if not xs:
        return None
    X = torch.cat(xs, dim=0)[:max_points]
    F = torch.cat(fs, dim=0)[:max_points]
    return X, F


def _replace_node(root: Node, target: AtomNode, new_subtree: Node) -> Node:
    """
    Pure functional replacement: return a *new* AST where 'target' has
    been replaced by 'new_atom'.
    """
    if root is target:
        return new_subtree
    if isinstance(root, AtomNode):
        return root
    if isinstance(root, AddNode):
        return AddNode(
            left=_replace_node(root.left, target, new_subtree),
            right=_replace_node(root.right, target, new_subtree),
        )
    if isinstance(root, MulNode):
        return MulNode(
            left=_replace_node(root.left, target, new_subtree),
            right=_replace_node(root.right, target, new_subtree),
        )
    if isinstance(root, PowNode):
        return PowNode(
            base=_replace_node(root.base, target, new_subtree),
            exponent=root.exponent,
        )
    if isinstance(root, LogNode):
        return LogNode(
            arg=_replace_node(root.arg, target, new_subtree),
        )
    if isinstance(root, ExpNode):
        return ExpNode(
            arg=_replace_node(root.arg, target, new_subtree),
        )
    if isinstance(root, SinNode):
        return SinNode(
            arg=_replace_node(root.arg, target, new_subtree),
        )
    if isinstance(root, CosNode):
        return CosNode(
            arg=_replace_node(root.arg, target, new_subtree),
        )
    if isinstance(root, ConstNode):
        return root  # Constants are unchanged
    raise TypeError(f"Unsupported node type in _replace_node: {type(root)}")
