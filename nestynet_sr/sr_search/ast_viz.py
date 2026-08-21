# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
AST visualization utilities for Streamlit GUI.

Provides tree rendering, metadata extraction, and visual representations of AST structures.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from nestynet_sr.sr_core.bridges import AddNode, AtomNode, LogNode, MulNode, Node, PowNode


@dataclass
class ASTMetadata:
    """Metadata extracted from an AST."""

    num_atoms: int
    num_nn_atoms: int
    num_analytic_atoms: int
    total_variables: set
    max_depth: int
    atom_types: Dict[str, int]  # kind -> count
    atoms_by_vars: Dict[Tuple[int, ...], List[Dict[str, Any]]]  # var_idxs -> [atom_info]


def collect_nn_atoms(root: Node) -> List[AtomNode]:
    """Collect all NN atoms from AST."""
    result = []

    def visit(node):
        if isinstance(node, AtomNode):
            if node.kind == "nn":
                result.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, (PowNode, LogNode)):
            visit(node.base if isinstance(node, PowNode) else node.arg)

    visit(root)
    return result


def collect_all_atoms(root: Node) -> List[AtomNode]:
    """Collect all atoms (NN and analytic) from AST."""
    result = []

    def visit(node):
        if isinstance(node, AtomNode):
            result.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, (PowNode, LogNode)):
            visit(node.base if isinstance(node, PowNode) else node.arg)

    visit(root)
    return result


def extract_ast_metadata(root: Optional[Node]) -> Optional[ASTMetadata]:
    """Extract metadata from AST for visualization."""
    if root is None:
        return None

    atoms = collect_all_atoms(root)
    total_vars = set()
    atom_types = {}
    atoms_by_vars = {}
    nn_count = 0
    analytic_count = 0

    for atom in atoms:
        # Count by kind
        if atom.kind == "nn":
            nn_count += 1
        else:
            analytic_count += 1

        atom_types[atom.kind] = atom_types.get(atom.kind, 0) + 1

        # Track variables
        for v in atom.var_idxs:
            total_vars.add(v)

        # Group by variables
        if atom.var_idxs not in atoms_by_vars:
            atoms_by_vars[atom.var_idxs] = []
        atoms_by_vars[atom.var_idxs].append(
            {"kind": atom.kind, "kwargs": atom.kwargs, "tag": atom.tag}
        )

    # Calculate max depth
    def get_depth(node):
        if isinstance(node, AtomNode):
            return 1
        elif isinstance(node, (AddNode, MulNode)):
            return 1 + max(get_depth(node.left), get_depth(node.right))
        elif isinstance(node, PowNode):
            return 1 + get_depth(node.base)
        elif isinstance(node, LogNode):
            return 1 + get_depth(node.arg)
        return 0

    max_depth = get_depth(root)

    return ASTMetadata(
        num_atoms=len(atoms),
        num_nn_atoms=nn_count,
        num_analytic_atoms=analytic_count,
        total_variables=total_vars,
        max_depth=max_depth,
        atom_types=atom_types,
        atoms_by_vars=atoms_by_vars,
    )


def ast_to_tree_str(root: Optional[Node], indent: int = 0, prefix: str = "") -> str:
    """Convert AST to indented tree string for display."""
    if root is None:
        return "(empty)"

    lines = []

    def visit(node, depth, is_last, parent_prefix):
        connector = "└─ " if is_last else "├─ "
        extension = "   " if is_last else "│  "

        if isinstance(node, AtomNode):
            kind_str = node.kind
            vars_str = f"[x{',x'.join(map(str, node.var_idxs))}]"

            # Add key kwargs
            extra = []
            if "num_segments" in node.kwargs:
                extra.append(f"seg={node.kwargs['num_segments']}")
            if "degree" in node.kwargs:
                extra.append(f"deg={node.kwargs['degree']}")
            if "dual_layer" in node.kwargs and node.kwargs["dual_layer"]:
                extra.append("dual")

            extra_str = f" ({', '.join(extra)})" if extra else ""

            lines.append(f"{parent_prefix}{connector}{kind_str}{vars_str}{extra_str}")

        elif isinstance(node, AddNode):
            lines.append(f"{parent_prefix}{connector}ADD")
            new_prefix = parent_prefix + extension
            visit(node.left, depth + 1, False, new_prefix)
            visit(node.right, depth + 1, True, new_prefix)

        elif isinstance(node, MulNode):
            lines.append(f"{parent_prefix}{connector}MUL")
            new_prefix = parent_prefix + extension
            visit(node.left, depth + 1, False, new_prefix)
            visit(node.right, depth + 1, True, new_prefix)

        elif isinstance(node, PowNode):
            exp_str = f"^{node.exponent:.3g}"
            lines.append(f"{parent_prefix}{connector}POW{exp_str}")
            new_prefix = parent_prefix + extension
            visit(node.base, depth + 1, True, new_prefix)

        elif isinstance(node, LogNode):
            lines.append(f"{parent_prefix}{connector}LOG")
            new_prefix = parent_prefix + extension
            visit(node.arg, depth + 1, True, new_prefix)

    visit(root, 0, True, "")
    return "\n".join(lines)


def ast_to_compact_str(root: Optional[Node]) -> str:
    """Convert AST to compact single-line string."""
    if root is None:
        return "(empty)"

    def visit(node):
        if isinstance(node, AtomNode):
            kind = node.kind
            vars_str = ",".join(f"x{v}" for v in node.var_idxs)
            return f"{kind}[{vars_str}]"
        elif isinstance(node, AddNode):
            return f"({visit(node.left)} + {visit(node.right)})"
        elif isinstance(node, MulNode):
            return f"{visit(node.left)} * {visit(node.right)}"
        elif isinstance(node, PowNode):
            return f"({visit(node.base)})^{node.exponent:.3g}"
        elif isinstance(node, LogNode):
            return f"log({visit(node.arg)})"
        return "?"

    return visit(root)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model."""
    if model is None:
        return 0

    # Try multiple approaches
    if hasattr(model, "num_parameters") and callable(model.num_parameters):
        try:
            return int(model.num_parameters())
        except Exception:
            pass

    # Fall back to summing parameters
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


def format_loss_value(loss: float) -> str:
    """Format loss value for display."""
    if loss == float("inf"):
        return "∞"
    elif loss == float("-inf"):
        return "-∞"
    elif abs(loss) < 1e-10:
        return f"{loss:.2e}"
    elif abs(loss) < 1e-3:
        return f"{loss:.2e}"
    elif abs(loss) < 1000:
        return f"{loss:.6f}"
    else:
        return f"{loss:.2e}"


def get_ytransform_emoji(ytransform_name: str) -> str:
    """Get emoji for y-transform name."""
    emoji_map = {
        "identity": "=",
        "square": "²",
        "log": "㏒",
        "reciprocal": "⁻¹",
        "sqrt": "√",
        "exp": "ℯ",
        "arctan": "∡",
        "arcsin": "∿",
        "arccos": "∿",
        "sin": "∿",
        "cos": "∿",
        "tan": "∿",
    }
    return emoji_map.get(ytransform_name, "?")


def get_stage_emoji(stage: str) -> str:
    """Get emoji for stage name."""
    emoji_map = {
        "init": "🔧",
        "stageA": "🔍",
        "stageB": "🎯",
        "complete": "✅",
    }
    return emoji_map.get(stage, "❓")


def format_time(seconds: float) -> str:
    """Format elapsed time for display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def create_progress_summary(state) -> Dict[str, Any]:
    """Create a summary dictionary for display."""
    return {
        "stage": state.stage,
        "ytransform": state.current_ytransform or "N/A",
        "ytransform_progress": f"{state.ytransform_index + 1}/{state.total_ytransforms}"
        if state.total_ytransforms > 0
        else "0/0",
        "iteration": state.stageA_iteration if state.stage == "stageA" else state.stageB_outer_iter,
        "num_leaves": state.num_leaves,
        "num_params": state.num_params,
        "current_loss": format_loss_value(state.current_val_loss),
        "best_loss": format_loss_value(state.best_val_loss),
        "elapsed_time": format_time(state.elapsed_time),
        "lm_progress": f"{state.lm_epoch}/{state.lm_max_epochs}"
        if state.lm_max_epochs > 0
        else "0/0",
    }
