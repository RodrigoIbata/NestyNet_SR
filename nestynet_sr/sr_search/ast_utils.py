# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
AST utilities for display and manipulation.

This module provides shared utilities for working with AST nodes,
including compact string representation for logging.
"""

from __future__ import annotations

from typing import Dict, Optional

from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AbsNode,
    AddNode,
    AsinNode,
    ArgNode,
    AtanNode,
    AtomNode,
    ConjNode,
    ConstNode,
    CosNode,
    ExpNode,
    ImagNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    RealNode,
    SinNode,
    atom_problem_label,
    compound_input_expr,
    extra_input_nodes,
    format_const_value,
    get_input_exprs,
    has_nontrivial_input,
    is_trivial_input,
)

# ANSI color codes
PURPLE = "\033[35m"
RESET = "\033[0m"


def check_ast_is_tree(root: Node) -> tuple[bool, str | None]:
    """Check whether an AST is a strict tree (no shared Node objects).

    Some candidate builders may accidentally reuse the *same* Node instance in
    multiple places (creating a DAG). This repo's leaf ordering / chain
    evaluation assumes a strict tree; shared nodes can lead to subtle failures
    (e.g. leaf-index mismatches) and downstream crashes.

    Returns
    -------
    ok : bool
        True if each Node object appears exactly once in the traversal.
    detail : str | None
        If not ok, a short description of the first shared node encountered.
    """

    first_path: dict[int, str] = {}
    dup_detail: list[str] = []

    def walk(node: Node, path: str) -> None:
        if dup_detail:
            return
        nid = id(node)
        if nid in first_path:
            p0 = first_path[nid]
            dup_detail.append(
                f"shared {type(node).__name__} at {p0} and {path}"
            )
            return
        first_path[nid] = path

        if isinstance(node, AtomNode):
            if has_nontrivial_input(node):
                for inp in get_input_exprs(node):
                    walk(inp, f"{path}.compound")
            return

        if isinstance(node, (AddNode, MulNode)):
            walk(node.left, f"{path}.left")
            walk(node.right, f"{path}.right")
            return

        if isinstance(node, PowNode):
            walk(node.base, f"{path}.base")
            return

        if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
            walk(node.arg, f"{path}.arg")
            return

        # Defensive fallback for future node types.
        for attr in ("left", "right", "base", "arg"):
            if hasattr(node, attr):
                try:
                    child = getattr(node, attr)
                except Exception:
                    continue
                if child is not None:
                    walk(child, f"{path}.{attr}")

    walk(root, "root")
    if dup_detail:
        return False, dup_detail[0]
    return True, None


def compact_expression_repr(
    root: Node, *, max_length: int = 120, use_color: bool = True, y_op_inv=None,
    x_labels: Optional[Dict[int, str]] = None,
) -> str:
    """
    Create a compact, pretty representation of an AST with numbered atoms.

    Example output: "nn0(x0, x1) * poly1(x2)" or "sqrt(nn0(x0, x1) * poly1(x2))"

    Args:
        root: AST root node
        max_length: Maximum string length before truncation
        use_color: Whether to apply purple coloring
        y_op_inv: Optional inverse y-transform for display (e.g., sqrt)

    Returns:
        Compact string representation
    """
    # First pass: number all atoms by type
    atom_counters = {}
    atom_names = {}

    def assign_names(node):
        if isinstance(node, AtomNode):
            kind = str(node.kind).lower()
            if id(node) not in atom_names:
                count = atom_counters.get(kind, 0)
                atom_counters[kind] = count + 1
                atom_names[id(node)] = f"{kind}{count}"
        elif hasattr(node, "left") and hasattr(node, "right"):
            assign_names(node.left)
            assign_names(node.right)
        elif hasattr(node, "arg"):
            assign_names(node.arg)
        elif hasattr(node, "base"):
            assign_names(node.base)

    assign_names(root)

    # Second pass: build compact string
    def to_str(node, parent_op=None):
        if isinstance(node, AtomNode):
            kind = str(node.kind).lower()
            # Variable atoms (used in input_expr): just show x{idx}
            if kind in ("var", "x", "input"):
                if node.var_idxs and len(node.var_idxs) == 1:
                    idx = int(node.var_idxs[0])
                    if x_labels and idx in x_labels:
                        return x_labels[idx]
                    return f"x{idx}"
                # Fallback for multi-var
                return ", ".join(
                    (x_labels[int(v)] if (x_labels and int(v) in x_labels) else f"x{int(v)}")
                    for v in node.var_idxs
                )

            problem_label = atom_problem_label(node)
            name = (
                str(problem_label)
                if problem_label is not None
                else atom_names.get(id(node), f"{kind}?")
            )
            if has_nontrivial_input(node):
                z_expr = compound_input_expr(node)
                z_str = to_str(z_expr)
                if " " in z_str:
                    z_str = f"({z_str})"
                extra_nodes = extra_input_nodes(node)
                if extra_nodes:
                    has_compound_extra = any(
                        not is_trivial_input(extra) for extra in extra_nodes
                    )
                    primary_label = "z0" if has_compound_extra else "z"
                    args = [f"{primary_label}={z_str}"]
                    z_count = 1
                    for extra in extra_nodes:
                        extra_str = to_str(extra)
                        if " " in extra_str:
                            extra_str = f"({extra_str})"
                        if is_trivial_input(extra):
                            args.append(extra_str)
                        else:
                            args.append(f"z{z_count}={extra_str}")
                            z_count += 1
                    return f"{name}({', '.join(args)})"
                return f"{name}(z={z_str})"
            # Regular atom: show var_idxs
            vars_str = ", ".join(f"x{int(v)}" for v in node.var_idxs) if node.var_idxs else ""
            return f"{name}({vars_str})"

        elif isinstance(node, MulNode):
            left_str = to_str(node.left, "mul")
            right_str = to_str(node.right, "mul")
            if isinstance(node.left, AddNode):
                left_str = f"({left_str})"
            if isinstance(node.right, AddNode):
                right_str = f"({right_str})"
            result = f"{left_str} * {right_str}"
            # Add parens if we're inside an Add or Pow
            if parent_op in ("add", "pow"):
                result = f"({result})"
            return result

        elif isinstance(node, AddNode):
            left_str = to_str(node.left, "add")
            right_str = to_str(node.right, "add")
            result = f"{left_str} + {right_str}"
            if parent_op in ("pow", "mul"):
                result = f"({result})"
            return result

        elif isinstance(node, PowNode):
            base_str = to_str(node.base, "pow")
            exp_val = node.exponent
            # Format exponent nicely
            if isinstance(exp_val, (int, float)):
                if exp_val == int(exp_val):
                    exp_str = str(int(exp_val))
                else:
                    exp_str = f"{exp_val:.2g}"
            else:
                exp_str = str(exp_val)
            return f"{base_str}^{exp_str}"

        elif isinstance(node, LogNode):
            arg_str = to_str(node.arg, "log")
            return f"log({arg_str})"

        elif isinstance(node, ExpNode):
            arg_str = to_str(node.arg, "exp")
            return f"exp({arg_str})"

        elif isinstance(node, SinNode):
            arg_str = to_str(node.arg, "sin")
            return f"sin({arg_str})"

        elif isinstance(node, CosNode):
            arg_str = to_str(node.arg, "cos")
            return f"cos({arg_str})"

        elif isinstance(node, AsinNode):
            arg_str = to_str(node.arg, "asin")
            return f"arcsin({arg_str})"

        elif isinstance(node, AcosNode):
            arg_str = to_str(node.arg, "acos")
            return f"arccos({arg_str})"

        elif isinstance(node, AtanNode):
            arg_str = to_str(node.arg, "atan")
            return f"arctan({arg_str})"

        elif isinstance(node, ConjNode):
            arg_str = to_str(node.arg, "conj")
            return f"conj({arg_str})"

        elif isinstance(node, RealNode):
            arg_str = to_str(node.arg, "real")
            return f"Re({arg_str})"

        elif isinstance(node, ImagNode):
            arg_str = to_str(node.arg, "imag")
            return f"Im({arg_str})"

        elif isinstance(node, AbsNode):
            arg_str = to_str(node.arg, "abs")
            return f"|{arg_str}|"

        elif isinstance(node, ArgNode):
            arg_str = to_str(node.arg, "arg")
            return f"arg({arg_str})"

        elif isinstance(node, ConstNode):
            return format_const_value(node.value)

        else:
            return f"<{type(node).__name__}>"

    expr = to_str(root)

    # Wrap with outer y-transform if provided
    if y_op_inv is not None:
        # Extract function name from callable
        if hasattr(y_op_inv, "__name__"):
            transform_name = y_op_inv.__name__
        else:
            transform_name = str(y_op_inv)
        expr = f"{transform_name}({expr})"

    # Truncate if too long
    if len(expr) > max_length:
        expr = expr[: max_length - 3] + "..."

    # Apply color
    if use_color:
        expr = f"{PURPLE}{expr}{RESET}"

    return expr
