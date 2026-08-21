# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
AST (Abstract Syntax Tree) infrastructure for symbolic regression.

AST = Abstract Syntax Tree - a tree representation of the syntactic structure
of expressions.

- **Abstract**: Captures the meaning/structure, not literal syntax details
- **Syntax**: The compositional structure (what multiplies/adds with what)
- **Tree**: Hierarchical node structure

Example
-------
The expression `NN[x0] * (NN[x1] * NN[x2])` becomes::

           MulNode
          /       \\
      AtomNode   MulNode
      ('nn',(0)) /     \\
             AtomNode  AtomNode
             ('nn',(1)) ('nn',(2))

Note this gives:
1. Single source of truth - no synchronization issues
2. Natural tree operations - traverse, replace, clone
3. Type safety - structure is explicit in the type system
4. Easier to extend - add new node types or operations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .atoms import (
    Expm1Leaf,
    ExpPolyLeaf,
    ExpRationalPolyLeaf,
    FreeConstLeaf,
    FixedConstLeaf,
    InverseMonomialLeaf,
    RInverseMonomialLeaf,
    LinLeaf,
    PolyLogLeaf,
    PlanckFullLeaf,
    PlanckLeaf,
    PolyLeaf,
    PowerLeaf,
    RationalPolyLeaf,
    RatioPolyLeaf,
    RExpPolyLeaf,
    RPolyLogLeaf,
    RPolyLeaf,
    RRationalPolyLeaf,
    RRatioPolyLeaf,
    SinLinearLeaf,
    TanhLinearLeaf,
    VarLeaf,
)

# Optional dependency: nestynet provides AutogradAdaptor.
#
# The AST / units / rewrite logic in this repo does not *require* nestynet to be
# importable; only the compilation step (build_composite_from_ast) needs it.
try:
    from nestynet.adaptors import AutogradAdaptor  # type: ignore
except ImportError:  # pragma: no cover
    try:
        # Fallback for when running from within a larger mono-repo layout.
        import os
        import sys

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from nestynet.adaptors import AutogradAdaptor  # type: ignore
    except ImportError:
        AutogradAdaptor = None  # type: ignore

# Note: ASTCompositeAdaptor is imported lazily inside build_composite_from_ast()
# to avoid circular import (ast_composite.py imports from this module)


# ──────────────────────────────────────────────────────────────
# AST node types
# ──────────────────────────────────────────────────────────────

Node = Union[
    "AtomNode",
    "AddNode",
    "MulNode",
    "PowNode",
    "LogNode",
    "ExpNode",
    "SinNode",
    "CosNode",
    "AsinNode",
    "AcosNode",
    "AtanNode",
    "ConjNode",
    "RealNode",
    "ImagNode",
    "AbsNode",
    "ArgNode",
    "UNARY_NODE_TYPES",
    "ConstNode",
]




# ──────────────────────────────────────────────────────────────
# Unified z-space helpers
# ──────────────────────────────────────────────────────────────
#
# These helpers support treating ALL variables as z-coordinates with either
# trivial (z = x) or compound (z = f(x)) mappings. A "normal" variable is
# just a trivial compound: z_i = x_i.


def is_trivial_input(expr: "Node") -> bool:
    """Return True if expr is a trivial Var(i) (trivial z = x_i mapping).

    This distinguishes between:
    - Trivial inputs: Var(i) nodes representing z_i = x_i
    - Compound inputs: expressions like Mul(Var(0), Var(1)) representing z = x0*x1

    Parameters
    ----------
    expr : Node
        An AST node representing an input expression.

    Returns
    -------
    bool
        True if the expression is a plain Var(i) node.

    Examples
    --------
    >>> is_trivial_input(Var(0))  # Trivial: z = x0
    True
    >>> is_trivial_input(MulNode(Var(0), Var(1)))  # Compound: z = x0*x1
    False
    """
    if not isinstance(expr, AtomNode):
        return False
    kind = str(getattr(expr, 'kind', '')).lower()
    return kind in ('var', 'x', 'input') and len(expr.var_idxs) == 1


def get_input_exprs(atom: "AtomNode") -> Tuple["Node", ...]:
    """Return the input expressions for this atom (always defined).

    For atoms with trivial inputs [Var(0), Var(1), ...], this returns those
    Var nodes. For compound atoms, this returns the compound expression plus
    any extra variables.

    This provides a unified interface for accessing what an atom sees.

    Parameters
    ----------
    atom : AtomNode
        The atom to query.

    Returns
    -------
    Tuple[Node, ...]
        The input expressions. For normal atoms, each element is Var(i).
        For compound atoms, first element is the compound expression,
        remaining elements are Var(extra_i).

    Examples
    --------
    >>> # Normal atom seeing x0, x1
    >>> atom = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={})
    >>> get_input_exprs(atom)
    (Var(0), Var(1))

    >>> # Compound atom: z = x0*x1 with extra x2
    >>> atom_compound = ...  # compound with input_expr=Mul(Var(0),Var(1)), extra_var_idxs=[2]
    >>> get_input_exprs(atom_compound)
    (Mul(Var(0), Var(1)), Var(2))
    """
    if atom.inputs is not None and len(atom.inputs) > 0:
        return atom.inputs
    # Fallback: build from var_idxs
    return tuple(_make_var_node(int(i)) for i in atom.var_idxs)


def has_nontrivial_input(atom: "AtomNode") -> bool:
    """Return True if any input is a nontrivial expression (not a plain Var).

    Use this when you genuinely need to distinguish between simple atoms
    (all inputs are Var(i)) and atoms with compound input expressions
    (e.g. Mul(Var(0), Var(1))).

    Parameters
    ----------
    atom : AtomNode
        The atom to check.

    Returns
    -------
    bool
        True if any input expression is compound (not a plain Var(i)).
    """
    inputs = get_input_exprs(atom)
    return any(not is_trivial_input(inp) for inp in inputs)



def compound_input_expr(atom: "AtomNode") -> Optional["Node"]:
    """Return the first input expression for this atom.

    For simple atoms this returns Var(i); for compound atoms it returns the
    compound expression.  Returns None for constant atoms that have no
    input variables (e.g. FreeConstLeaf with var_idxs=()).

    Use ``has_nontrivial_input(atom)`` when you need to know whether the
    input is a genuinely compound expression vs. a trivial Var(i).
    """
    inputs = get_input_exprs(atom)
    if not inputs:
        return None
    return inputs[0]


def extra_input_var_idxs(atom: "AtomNode") -> Tuple[int, ...]:
    """Return raw var indices of inputs[1:] extras.  Replaces ``cv.extra_var_idxs``."""
    inputs = get_input_exprs(atom)
    if len(inputs) <= 1:
        return ()
    result: list[int] = []
    for inp in inputs[1:]:
        result.extend(_collect_var_idxs_from_node(inp))
    return tuple(result)


def extra_input_nodes(atom: "AtomNode") -> Tuple["Node", ...]:
    """Return inputs[1:] for compound atoms (empty for simple atoms)."""
    inputs = get_input_exprs(atom)
    if len(inputs) <= 1:
        return ()
    return inputs[1:]


def trivial_input_position(atom: "AtomNode", var_idx: int) -> Optional[int]:
    """Return the input position whose expression is exactly ``Var(var_idx)``.

    For compound atoms, rules that reason about a specific raw axis (e.g. a
    trig hint on x6) must map that axis to an INPUT POSITION of the atom.  A
    raw axis that appears only inside a nontrivial compound input (e.g. x0
    inside z = x0/x1) is not isolable, and callers must skip the variant
    rather than fall back to raw ``var_idxs``.

    Returns the first matching position, or None when the axis has no trivial
    input slot.
    """

    target = int(var_idx)
    for position, expr in enumerate(get_input_exprs(atom)):
        if is_trivial_input(expr) and int(expr.var_idxs[0]) == target:
            return position
    return None


def clone_inputs(atom: "AtomNode") -> Optional[Tuple["Node", ...]]:
    """Deep-copy atom.inputs for use in new atoms (prevents shared DAGs).

    Returns None for simple atoms (let __post_init__ build from var_idxs).
    """
    if not has_nontrivial_input(atom):
        return None
    return tuple(clone_ast(inp) for inp in get_input_exprs(atom))


def atom_problem_label(atom: "AtomNode") -> Optional[str]:
    """Return a user-facing problem label for an atom, if one is attached."""
    try:
        kwargs = getattr(atom, "kwargs", None) or {}
        label = kwargs.get("_problem_label", kwargs.get("_problem_code", None))
    except Exception:
        label = None
    if label is None:
        return None
    try:
        label_s = str(label).strip()
    except Exception:
        return None
    return label_s or None


def atom_problem_message(atom: "AtomNode") -> Optional[str]:
    """Return the stored problem detail message for an atom, if present."""
    try:
        kwargs = getattr(atom, "kwargs", None) or {}
        msg = kwargs.get("_problem_msg", kwargs.get("_problem_reason", None))
    except Exception:
        msg = None
    if msg is None:
        return None
    try:
        msg_s = str(msg).strip()
    except Exception:
        return None
    return msg_s or None


def is_problem_atom(atom: "AtomNode", label: str | None = None) -> bool:
    """Return True when an atom carries internal problem-leaf metadata."""
    problem = atom_problem_label(atom)
    if problem is None:
        return False
    if label is None:
        return True
    return str(problem) == str(label)


def _select_inputs_for_var_group(
    parent_atom: "AtomNode", raw_var_group
) -> Optional[Tuple["Node", ...]]:
    """Select parent input expressions whose variables fall within a raw-var group.

    When Stage A detects compound variables (e.g. u=x0-x1), these are stored as
    ``inputs`` on the AtomNode.  When Stage B splits an atom by separability or
    pruning, child atoms must inherit the relevant subset of those inputs.

    Returns a tuple of cloned input Nodes, or None if parent has only trivial inputs.
    """
    if not has_nontrivial_input(parent_atom):
        return None
    inputs = get_input_exprs(parent_atom)
    group_set = set(int(v) for v in raw_var_group)
    selected = []
    for inp in inputs:
        inp_vars = set(_collect_var_idxs_from_node(inp))
        if inp_vars and inp_vars <= group_set:
            selected.append(clone_ast(inp))
    return tuple(selected) if selected else None


# ──────────────────────────────────────────────────────────────
# Unified atom input handling (inputs field)
# ──────────────────────────────────────────────────────────────

def _collect_var_idxs_from_node(node: "Node") -> Tuple[int, ...]:
    """Recursively collect all raw variable indices from an AST node."""
    if isinstance(node, AtomNode):
        kind = str(getattr(node, 'kind', '')).lower()
        if kind in ('var', 'x', 'input'):
            return node.var_idxs
        # For other atom types (should not appear in input expressions), return empty
        return ()
    elif isinstance(node, (AddNode, MulNode)):
        left = _collect_var_idxs_from_node(node.left)
        right = _collect_var_idxs_from_node(node.right)
        return tuple(sorted(set(left) | set(right)))
    elif isinstance(node, PowNode):
        return _collect_var_idxs_from_node(node.base)
    elif isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        return _collect_var_idxs_from_node(node.arg)
    elif isinstance(node, ConstNode):
        return ()
    else:
        return ()


def _collect_var_idxs_from_inputs(inputs: Tuple["Node", ...]) -> Tuple[int, ...]:
    """Collect all raw variable indices referenced by input expressions."""
    all_idxs = set()
    for inp in inputs:
        all_idxs.update(_collect_var_idxs_from_node(inp))
    return tuple(sorted(all_idxs))


def _is_plain_var(node: "Node") -> bool:
    """Check if a node is a plain Var(i) node."""
    if not isinstance(node, AtomNode):
        return False
    kind = str(getattr(node, 'kind', '')).lower()
    return kind in ('var', 'x', 'input') and len(node.var_idxs) == 1


def _make_var_node(idx: int) -> "AtomNode":
    """Create a Var-like AtomNode without populating inputs (to avoid recursion)."""
    # Create directly without calling Var() to avoid __post_init__ recursion
    node = object.__new__(AtomNode)
    node.kind = "var"
    node.var_idxs = (int(idx),)
    node.kwargs = {}
    node.tag = None
    node.inputs = ()  # Var nodes have empty inputs
    return node



@dataclass
class AtomNode:
    """
    A symbolic 'leaf' in the SR grammar, which will be compiled into
    a small torch.nn.Module (SinLinearLeaf / PolyLeaf / RationalPolyLeaf / NN).

    Attributes
    ----------
    kind      : 'sin_linear' | 'poly' | 'ratpoly' | 'nn' | (future: 'log', 'exp', ...)
    var_idxs  : indices of x-columns this atom sees (global input coords)
    kwargs    : extra hyperparameters, e.g. degree for polynomials
    tag       : optional identifier to link this node to a specific module instance
    inputs    : tuple of input expressions (each evaluates to [B,1])
                The unified interface for specifying what the atom sees.
                If None, auto-populated from var_idxs in __post_init__.
    """

    kind: str
    var_idxs: Tuple[int, ...]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    tag: str | None = None
    inputs: Tuple["Node", ...] | None = None
    scope: str = "experiment"  # "class" or "experiment" (for multi-dataset class SR)

    def __post_init__(self):
        self.var_idxs = tuple(int(i) for i in self.var_idxs)

        if self.inputs is None:
            if self.kind.lower() in ("var", "x", "input"):
                self.inputs = ()
            else:
                self.inputs = tuple(_make_var_node(int(i)) for i in self.var_idxs)

    @property
    def n_in(self) -> int:
        """Number of input dimensions this atom expects."""
        return len(self.inputs) if self.inputs else 0

    @property
    def raw_var_idxs(self) -> Tuple[int, ...]:
        """All raw variable indices referenced by any input expression."""
        if self.inputs is None:
            return self.var_idxs
        return _collect_var_idxs_from_inputs(self.inputs)

    def is_simple(self) -> bool:
        """True if all inputs are plain Var nodes (fast path eligible)."""
        if self.inputs is None:
            return True  # Empty inputs is trivially simple
        return all(_is_plain_var(inp) for inp in self.inputs)

    def simple_var_idxs(self) -> Optional[Tuple[int, ...]]:
        """If is_simple(), return var indices in input order. Else None."""
        if not self.is_simple():
            return None
        if self.inputs is None:
            return ()
        result = []
        for inp in self.inputs:
            if isinstance(inp, AtomNode) and len(inp.var_idxs) == 1:
                result.append(inp.var_idxs[0])
        return tuple(result)

    def __repr__(self):
        kind = str(self.kind).lower()
        tag_str = f"#{self.tag}" if self.tag is not None else ""

        # Pretty forms for common leaves
        if kind in ("var", "x", "input"):
            if len(self.var_idxs) == 1:
                return f"x{int(self.var_idxs[0])}{tag_str}"

        if kind in ("free_const", "freeconst", "free_constant"):
            name = None
            try:
                name = self.kwargs.get("name", None)
            except Exception:
                name = None
            if name is None:
                name = self.tag
            name = str(name) if name is not None else "c"
            # Avoid duplicate name#name when tag==name
            if self.tag is not None and str(self.tag) == name:
                tag_str = ""
            return f"{name}{tag_str}"

        if kind in ("fixed_const", "fixedconst", "fixed_constant"):
            name = None
            try:
                name = self.kwargs.get("name", None)
            except Exception:
                name = None
            if name is None:
                name = self.tag
            name = str(name) if name is not None else "k"
            # Avoid duplicate name#name when tag==name
            if self.tag is not None and str(self.tag) == name:
                tag_str = ""
            return f"{name}{tag_str}"

        # Pretty forms for fitted-field feature atoms (native DE search)
        if kind in ("u", "field", "state"):
            # Optional pretty naming for named (vector) fields: Field("E","x"), etc.
            field_name = None
            comp_name = None
            try:
                field_name = self.kwargs.get("field", None)
                comp_name = self.kwargs.get("comp_name", None)
            except Exception:
                field_name = None
                comp_name = None

            if field_name is not None:
                base = str(field_name)
                if comp_name is None:
                    # Fallback: try numeric component
                    try:
                        comp = self.kwargs.get("comp", None)
                        if comp is not None:
                            comp_name = str(comp)
                    except Exception:
                        comp_name = None
                if comp_name is not None:
                    comp_name_str = str(comp_name)
                    if comp_name_str != "" and comp_name_str.lower() != "none":
                        base = f"{base}_{comp_name_str}"
                return f"{base}{tag_str}"

            out_idx = 0
            try:
                out_idx = int(self.kwargs.get("out_idx", self.kwargs.get("out", self.kwargs.get("component", 0))))
            except Exception:
                out_idx = 0
            base = f"u{out_idx}" if out_idx != 0 else "u"
            return f"{base}{tag_str}"

        if kind in ("du", "d1u", "grad_u"):
            axis = None
            try:
                axis = int(self.kwargs.get("axis", 0))
            except Exception:
                axis = 0

            field_name = None
            comp_name = None
            try:
                field_name = self.kwargs.get("field", None)
                comp_name = self.kwargs.get("comp_name", None)
            except Exception:
                field_name = None
                comp_name = None

            if field_name is not None:
                base = str(field_name)
                if comp_name is None:
                    try:
                        comp = self.kwargs.get("comp", None)
                        if comp is not None:
                            comp_name = str(comp)
                    except Exception:
                        comp_name = None
                if comp_name is not None:
                    comp_name_str = str(comp_name)
                    if comp_name_str != "" and comp_name_str.lower() != "none":
                        base = f"{base}_{comp_name_str}"
                return f"{base}_x{axis}{tag_str}"

            out_idx = 0
            try:
                out_idx = int(self.kwargs.get("out_idx", self.kwargs.get("out", self.kwargs.get("component", 0))))
            except Exception:
                out_idx = 0
            base = f"u{out_idx}" if out_idx != 0 else "u"
            return f"{base}_x{axis}{tag_str}"

        if kind in ("d2u", "ddu", "hess_u"):
            a0 = a1 = 0
            try:
                a0 = int(self.kwargs.get("axis0", 0))
                a1 = int(self.kwargs.get("axis1", 0))
            except Exception:
                a0 = a1 = 0

            field_name = None
            comp_name = None
            try:
                field_name = self.kwargs.get("field", None)
                comp_name = self.kwargs.get("comp_name", None)
            except Exception:
                field_name = None
                comp_name = None

            if field_name is not None:
                base = str(field_name)
                if comp_name is None:
                    try:
                        comp = self.kwargs.get("comp", None)
                        if comp is not None:
                            comp_name = str(comp)
                    except Exception:
                        comp_name = None
                if comp_name is not None:
                    comp_name_str = str(comp_name)
                    if comp_name_str != "" and comp_name_str.lower() != "none":
                        base = f"{base}_{comp_name_str}"
                return f"{base}_x{a0}x{a1}{tag_str}"

            out_idx = 0
            try:
                out_idx = int(self.kwargs.get("out_idx", self.kwargs.get("out", self.kwargs.get("component", 0))))
            except Exception:
                out_idx = 0
            base = f"u{out_idx}" if out_idx != 0 else "u"
            return f"{base}_x{a0}x{a1}{tag_str}"

        if has_nontrivial_input(self):
            from nestynet_sr.sr_search.representation import _input_expr_to_str
            inputs = get_input_exprs(self)
            args = ", ".join(_input_expr_to_str(inp) for inp in inputs)
        else:
            args = ", ".join(f"x{int(j)}" for j in self.var_idxs)
        return f"{self.kind}({args}){tag_str}"


@dataclass
class AddNode:
    """Binary addition node: left + right."""

    left: Node
    right: Node

    def __repr__(self):
        return f"({self.left} + {self.right})"


@dataclass
class MulNode:
    """Binary multiplication node: left * right."""

    left: Node
    right: Node

    def __repr__(self):
        return f"({self.left} * {self.right})"


@dataclass
class PowNode:
    """Unary power node: (base)**exponent."""

    base: Node
    exponent: float

    def __repr__(self):
        # Handle both numeric and Node exponents
        if isinstance(self.exponent, (int, float)):
            return f"({self.base} ** {self.exponent:g})"
        else:
            return f"({self.base} ** {self.exponent})"


@dataclass
class LogNode:
    """Unary log node: log(arg)."""

    arg: Node

    def __repr__(self):
        return f"log({self.arg})"


@dataclass
class ExpNode:
    """Unary exp node: exp(arg)."""

    arg: Node

    def __repr__(self):
        return f"exp({self.arg})"


@dataclass
class SinNode:
    """Unary sin node: sin(arg)."""

    arg: Node

    def __repr__(self):
        return f"sin({self.arg})"


@dataclass
class CosNode:
    """Unary cos node: cos(arg)."""

    arg: Node

    def __repr__(self):
        return f"cos({self.arg})"


@dataclass
class AsinNode:
    """Unary arcsin node: arcsin(arg)."""

    arg: Node

    def __repr__(self):
        return f"arcsin({self.arg})"


@dataclass
class AcosNode:
    """Unary arccos node: arccos(arg)."""

    arg: Node

    def __repr__(self):
        return f"arccos({self.arg})"


@dataclass
class AtanNode:
    """Unary arctan node: arctan(arg)."""

    arg: Node

    def __repr__(self):
        return f"arctan({self.arg})"


@dataclass
class ConjNode:
    """Complex conjugate node: conj(arg)."""

    arg: Node

    def __repr__(self):
        return f"conj({self.arg})"


@dataclass
class RealNode:
    """Real part node: real(arg)."""

    arg: Node

    def __repr__(self):
        return f"real({self.arg})"


@dataclass
class ImagNode:
    """Imaginary part node: imag(arg)."""

    arg: Node

    def __repr__(self):
        return f"imag({self.arg})"


@dataclass
class AbsNode:
    """Absolute value node: abs(arg)."""

    arg: Node

    def __repr__(self):
        return f"abs({self.arg})"


@dataclass
class ArgNode:
    """Complex argument (phase) node: arg(arg)."""

    arg: Node

    def __repr__(self):
        return f"arg({self.arg})"


# Canonical tuple of every unary AST node (single ``.arg`` child).  AST
# walkers must use this rather than hand-maintained subsets: sympy
# round-trips can legitimately introduce e.g. AbsNode (sqrt(x**2) -> |x|)
# into accepted Stage-B states, and a walker with a stale subset either
# crashes ("Unexpected node type") or silently skips the subtree.
UNARY_NODE_TYPES = (
    LogNode,
    ExpNode,
    SinNode,
    CosNode,
    AsinNode,
    AcosNode,
    AtanNode,
    ConjNode,
    RealNode,
    ImagNode,
    AbsNode,
    ArgNode,
)



def _coerce_const_value(value: Any) -> float | complex:
    """Coerce a scalar value to a Python float or complex.

    Notes
    -----
    This is intentionally permissive: callers may pass Python scalars,
    NumPy scalars, or 0-dim/size-1 Torch tensors.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(f"ConstNode expects a scalar, got tensor with shape {tuple(value.shape)}")
        value = value.item()

    # Handle NumPy scalars without taking a hard dependency.
    try:  # pragma: no cover
        import numpy as _np  # type: ignore

        if isinstance(_np, type) or isinstance(value, _np.generic):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, complex):
        return complex(value)

    try:
        return float(value)
    except Exception:
        try:
            return complex(value)
        except Exception as e:
            raise TypeError(f"ConstNode value must be a scalar number, got {type(value)}") from e


def format_const_value(value: float | complex) -> str:
    """Format a numeric constant for compact expression printing.

    For complex constants we preserve the complex marker even when the
    imaginary part is 0, because dtype promotion (real vs complex) can
    change the semantics of functions like log().
    """
    if isinstance(value, complex):
        a = float(value.real)
        b = float(value.imag)
        if b == 0.0:
            if a == 0.0:
                return "0j"
            return f"({a:g}+0j)"
        if a == 0.0:
            return f"{b:g}j"
        return f"({a:g}{b:+g}j)"
    return f"{float(value):g}"


def _complex_dtype_from_real(dtype: torch.dtype) -> torch.dtype:
    """Best-effort mapping from a real dtype to a complex dtype."""
    if dtype.is_complex:
        return dtype
    if dtype == torch.float64:
        return torch.complex128
    # float16/bfloat16/float32 all map to complex64
    return torch.complex64


def const_full_like(ref: torch.Tensor, shape: Tuple[int, ...], value: float | complex) -> torch.Tensor:
    """torch.full(...), but promotes dtype to complex when needed."""
    dtype = ref.dtype
    if isinstance(value, complex) and (not dtype.is_complex):
        dtype = _complex_dtype_from_real(dtype)
    return torch.full(shape, value, device=ref.device, dtype=dtype)


class ConstNode:
    """Fixed scalar constant node (not trainable).

    Used for including fixed numeric constants in expressions.

    Notes
    -----
    ConstNode supports both real and complex scalars. Complex constants are
    first-class because they can force complex dtype promotion, which is
    required to make operations like torch.log produce complex outputs
    (e.g. log(-1) -> i*pi rather than NaN).
    """

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = _coerce_const_value(value)

    def __repr__(self):
        return format_const_value(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConstNode):
            return False
        # Treat real vs complex as semantically different (dtype promotion).
        if isinstance(self.value, complex) != isinstance(other.value, complex):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        v = self.value
        if isinstance(v, complex):
            return hash(("ConstNode", float(v.real), float(v.imag)))
        return hash(("ConstNode", float(v)))


# ──────────────────────────────────────────────────────────────
# Unified eval_inputs() for AtomNode
# ──────────────────────────────────────────────────────────────

def _eval_single_input(
    node: "Node",
    x: torch.Tensor,
    *,
    need_grad: bool = False,
    need_hess: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Evaluate a single input expression node to get value, gradient, and optionally Hessian.

    Parameters
    ----------
    node : Node
        The input expression node (typically a Var, or a compound expression).
    x : torch.Tensor, shape (B, Nx)
        Full input data.
    need_grad : bool
        Whether to compute gradient w.r.t. x.
    need_hess : bool
        Whether to compute Hessian w.r.t. x.

    Returns
    -------
    tuple : (value, grad, hess)
        - value: (B, 1) - scalar value
        - grad: (B, 1, Nx) or None - gradient w.r.t. x
        - hess: (B, 1, Nx, Nx) or None - Hessian w.r.t. x
    """
    B, Nx = x.shape
    device, dtype = x.device, x.dtype

    def rec(n: "Node"):
        if isinstance(n, AtomNode):
            kind = str(getattr(n, 'kind', '')).lower()
            if kind in ('var', 'x', 'input'):
                if len(n.var_idxs) != 1:
                    raise ValueError("Var node in input_expr must have exactly 1 var_idx")
                idx = n.var_idxs[0]
                v = x[:, idx:idx+1]  # [B, 1]
                g = None
                gg = None
                if need_grad:
                    g = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                    g[:, 0, idx] = 1.0
                if need_hess:
                    gg = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                return v, g, gg
            if kind in ('fixed_const', 'fixedconst', 'fixed_constant'):
                kwargs = dict(getattr(n, "kwargs", {}) or {})
                value = float(
                    kwargs.get(
                        "value",
                        kwargs.get("val", kwargs.get("init", kwargs.get("value_init", 1.0))),
                    )
                )
                v = torch.full((B, 1), value, device=device, dtype=dtype)
                g = (
                    torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                    if need_grad
                    else None
                )
                gg = (
                    torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                    if need_hess
                    else None
                )
                return v, g, gg
            if kind in ('free_const', 'freeconst', 'free_constant', 'scale', 'mul_scale'):
                raise ValueError(
                    "input_expr contains a trainable free constant, but input-expression "
                    "atoms are not compiled as fitted leaves; refusing to substitute its "
                    "initial value"
                )
            else:
                raise ValueError(
                    f"Unsupported atom kind '{kind}' in input_expr; "
                    "only variables and fixed constants are allowed"
                )

        elif isinstance(n, ConstNode):
            v = const_full_like(x, (B, 1), n.value)
            g = torch.zeros(B, 1, Nx, device=device, dtype=v.dtype) if need_grad else None
            gg = torch.zeros(B, 1, Nx, Nx, device=device, dtype=v.dtype) if need_hess else None
            return v, g, gg

        elif isinstance(n, AddNode):
            v1, g1, gg1 = rec(n.left)
            v2, g2, gg2 = rec(n.right)
            v = v1 + v2
            g = None
            gg = None
            if need_grad:
                g = g1 + g2
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if gg2 is None:
                    gg2 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                gg = gg1 + gg2
            return v, g, gg

        elif isinstance(n, MulNode):
            v1, g1, gg1 = rec(n.left)
            v2, g2, gg2 = rec(n.right)
            v = v1 * v2
            g = None
            gg = None
            if need_grad:
                # Product rule: (fg)' = f'g + fg'
                g = v2[..., None] * g1 + v1[..., None] * g2
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if gg2 is None:
                    gg2 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                if g2 is None:
                    g2 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                # (fg)'' = f''g + 2f'g' + fg''
                outer = g1.unsqueeze(-1) * g2.unsqueeze(-2)
                outer = outer + g2.unsqueeze(-1) * g1.unsqueeze(-2)
                gg = v2[..., None, None] * gg1 + v1[..., None, None] * gg2 + outer
            return v, g, gg

        elif isinstance(n, PowNode):
            v1, g1, gg1 = rec(n.base)
            c = float(n.exponent)
            v = v1.pow(c)
            g = None
            gg = None
            if need_grad:
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                # d/dx[f^c] = c*f^(c-1) * f'
                g = c * v1.pow(c - 1)[..., None] * g1
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                # d²/dx²[f^c] = c*f^(c-1)*f'' + c*(c-1)*f^(c-2)*(f')²
                term1 = c * v1.pow(c - 1)[..., None, None] * gg1
                outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                term2 = c * (c - 1) * v1.pow(c - 2)[..., None, None] * outer
                gg = term1 + term2
            return v, g, gg

        elif isinstance(n, LogNode):
            v1, g1, gg1 = rec(n.arg)
            v = torch.log(v1)
            g = None
            gg = None
            if need_grad:
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                g = (1.0 / v1)[..., None] * g1
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                term1 = (1.0 / v1)[..., None, None] * gg1
                outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                term2 = -(1.0 / v1.pow(2))[..., None, None] * outer
                gg = term1 + term2
            return v, g, gg

        elif isinstance(n, ExpNode):
            v1, g1, gg1 = rec(n.arg)
            v = torch.exp(v1)
            g = None
            gg = None
            if need_grad:
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                g = v[..., None] * g1
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                term1 = v[..., None, None] * gg1
                outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                term2 = v[..., None, None] * outer
                gg = term1 + term2
            return v, g, gg

        elif isinstance(n, SinNode):
            v1, g1, gg1 = rec(n.arg)
            v = torch.sin(v1)
            cos_v = torch.cos(v1)
            g = None
            gg = None
            if need_grad:
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                g = cos_v[..., None] * g1
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                term1 = cos_v[..., None, None] * gg1
                outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                term2 = -v[..., None, None] * outer
                gg = term1 + term2
            return v, g, gg

        elif isinstance(n, CosNode):
            v1, g1, gg1 = rec(n.arg)
            v = torch.cos(v1)
            sin_v = torch.sin(v1)
            g = None
            gg = None
            if need_grad:
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                g = -sin_v[..., None] * g1
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                term1 = -sin_v[..., None, None] * gg1
                outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                term2 = -v[..., None, None] * outer
                gg = term1 + term2
            return v, g, gg

        elif isinstance(n, AsinNode):
            v1, g1, gg1 = rec(n.arg)
            v = torch.asin(torch.clamp(v1, -1.0 + 1.0e-12, 1.0 - 1.0e-12))
            denom = torch.clamp(1.0 - v1.pow(2), min=1.0e-24)
            d1 = denom.rsqrt()
            d2 = v1 * denom.pow(-1.5)
            g = None
            gg = None
            if need_grad:
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                g = d1[..., None] * g1
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                gg = d1[..., None, None] * gg1 + d2[..., None, None] * outer
            return v, g, gg

        elif isinstance(n, AcosNode):
            v1, g1, gg1 = rec(n.arg)
            v = torch.acos(torch.clamp(v1, -1.0 + 1.0e-12, 1.0 - 1.0e-12))
            denom = torch.clamp(1.0 - v1.pow(2), min=1.0e-24)
            d1 = -denom.rsqrt()
            d2 = -v1 * denom.pow(-1.5)
            g = None
            gg = None
            if need_grad:
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                g = d1[..., None] * g1
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                gg = d1[..., None, None] * gg1 + d2[..., None, None] * outer
            return v, g, gg

        elif isinstance(n, AtanNode):
            v1, g1, gg1 = rec(n.arg)
            v = torch.atan(v1)
            denom = 1.0 + v1.pow(2)
            d1 = 1.0 / denom
            d2 = -2.0 * v1 / denom.pow(2)
            g = None
            gg = None
            if need_grad:
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                g = d1[..., None] * g1
            if need_hess:
                if gg1 is None:
                    gg1 = torch.zeros(B, 1, Nx, Nx, device=device, dtype=dtype)
                if g1 is None:
                    g1 = torch.zeros(B, 1, Nx, device=device, dtype=dtype)
                outer = g1.unsqueeze(-1) * g1.unsqueeze(-2)
                gg = d1[..., None, None] * gg1 + d2[..., None, None] * outer
            return v, g, gg

        else:
            raise TypeError(f"Unsupported node type in input_expr: {type(n)}")

    return rec(node)


def eval_inputs(
    atom: AtomNode,
    x: torch.Tensor,
    *,
    need_grad: bool = False,
    need_hess: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Evaluate atom's input expressions to produce leaf input tensor.

    This is the unified entry point for computing an atom's effective inputs.
    It handles both simple atoms (just selecting columns from x) and compound
    atoms (where inputs are arbitrary expressions of x).

    Parameters
    ----------
    atom : AtomNode
        The atom whose inputs to evaluate.
    x : torch.Tensor, shape (B, Nx)
        Full input data.
    need_grad : bool
        Whether to compute Jacobian of inputs w.r.t. x.
    need_hess : bool
        Whether to compute Hessian of inputs w.r.t. x.

    Returns
    -------
    tuple : (x_in, grad, hess)
        - x_in: (B, n_in) - the effective inputs to the leaf
        - grad: (B, n_in, Nx) or None - Jacobian d(x_in)/dx
        - hess: (B, n_in, Nx, Nx) or None - Hessian d²(x_in)/dx²
    """
    B, Nx = x.shape
    device, dtype = x.device, x.dtype

    if atom.inputs is None or len(atom.inputs) == 0:
        # Standalone Var atom (e.g., bare Var(4) from tier-3 combos):
        # select columns from x and provide identity Jacobian.
        kind = str(getattr(atom, 'kind', '')).lower()
        if kind in ('var', 'x', 'input') and atom.var_idxs:
            idxs = list(atom.var_idxs)
            n_in = len(idxs)
            x_in = x[:, idxs]
            grad = None
            hess = None
            if need_grad:
                grad = torch.zeros(B, n_in, Nx, device=device, dtype=dtype)
                for i, j in enumerate(idxs):
                    grad[:, i, j] = 1.0
            if need_hess:
                hess = torch.zeros(B, n_in, Nx, Nx, device=device, dtype=dtype)
            return x_in, grad, hess
        # No inputs (e.g., DE feature atoms U(), DU(), etc.)
        x_in = torch.empty(B, 0, device=device, dtype=dtype)
        grad = torch.empty(B, 0, Nx, device=device, dtype=dtype) if need_grad else None
        hess = torch.empty(B, 0, Nx, Nx, device=device, dtype=dtype) if need_hess else None
        return x_in, grad, hess

    # Fast path: simple atoms (all inputs are plain Var nodes)
    if atom.is_simple():
        idxs = atom.simple_var_idxs()
        n_in = len(idxs)
        x_in = x[:, idxs]  # (B, n_in)

        grad = None
        hess = None
        if need_grad:
            grad = torch.zeros(B, n_in, Nx, device=device, dtype=dtype)
            for i, j in enumerate(idxs):
                grad[:, i, j] = 1.0
        if need_hess:
            hess = torch.zeros(B, n_in, Nx, Nx, device=device, dtype=dtype)

        return x_in, grad, hess

    # Compound path: evaluate each input expression
    vals = []
    grads = []
    hesses = []

    for inp in atom.inputs:
        v, g, gg = _eval_single_input(inp, x, need_grad=need_grad, need_hess=need_hess)
        vals.append(v)  # (B, 1)
        if need_grad:
            grads.append(g)  # (B, 1, Nx)
        if need_hess:
            hesses.append(gg)  # (B, 1, Nx, Nx)

    x_in = torch.cat(vals, dim=1)  # (B, n_in)

    grad = None
    hess = None
    if need_grad:
        grad = torch.cat(grads, dim=1)  # (B, n_in, Nx)
    if need_hess:
        hess = torch.cat(hesses, dim=1)  # (B, n_in, Nx, Nx)

    return x_in, grad, hess


# Convenience constructors for writing Add(a,b) / Mul(a,b) more concisely
def Add(a: Node, b: Node) -> AddNode:
    return AddNode(a, b)


def Mul(a: Node, b: Node) -> MulNode:
    return MulNode(a, b)


def Pow(base: Node, exponent: float) -> PowNode:
    return PowNode(base, exponent)


def Log(arg: Node) -> LogNode:
    return LogNode(arg)


def Exp(arg: Node) -> ExpNode:
    return ExpNode(arg)


def Div(a: Node, b: Node) -> MulNode:
    """Division as multiplication by inverse: a / b = a * b^(-1).

    This avoids introducing a separate DivNode, reusing existing Mul and Pow nodes
    for simpler traversal and evaluation code.
    """
    return Mul(a, Pow(b, -1))


def Sin(arg: Node) -> SinNode:
    return SinNode(arg)


def Cos(arg: Node) -> CosNode:
    return CosNode(arg)


def Asin(arg: Node) -> AsinNode:
    return AsinNode(arg)


def Acos(arg: Node) -> AcosNode:
    return AcosNode(arg)


def Atan(arg: Node) -> AtanNode:
    return AtanNode(arg)


def Conj(arg: Node) -> ConjNode:
    return ConjNode(arg)


def Real(arg: Node) -> RealNode:
    return RealNode(arg)


def Imag(arg: Node) -> ImagNode:
    return ImagNode(arg)


def Abs(arg: Node) -> AbsNode:
    return AbsNode(arg)


def Arg(arg: Node) -> ArgNode:
    return ArgNode(arg)


def Var(i: int, *, tag: str | None = None) -> AtomNode:
    """Convenience constructor for a raw input variable leaf."""
    return AtomNode(kind="var", var_idxs=(int(i),), tag=tag)


# ──────────────────────────────────────────────────────────────
# Named (vector) field helpers
# ──────────────────────────────────────────────────────────────

Comp = Union[int, str]


@dataclass(frozen=True)
class VectorFieldSpec:
    """Mapping from a named field (e.g. 'E') to surrogate output indices.

    This is purely a *convenience layer* for building scalar AtomNodes that
    carry an `out_idx` and some metadata (`field`, `comp_name`) for nicer
    printing / debugging.

    Parameters
    ----------
    name : str
        Field name used for pretty printing (e.g. "E", "B", "phi").
    base_out_idx : int
        Index of component 0 in the surrogate output u(x) vector.
    n_comp : int
        Number of components in the field.
    comp_names : tuple[str, ...]
        Pretty component labels. For n_comp=3 defaults to ('x','y','z').
        For n_comp=1 defaults to ('',) meaning "no suffix" (scalar field).
    """

    name: str
    base_out_idx: int
    n_comp: int = 3
    comp_names: Tuple[str, ...] = ("x", "y", "z")

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "base_out_idx", int(self.base_out_idx))
        object.__setattr__(self, "n_comp", int(self.n_comp))
        cn = tuple("" if (c is None) else str(c) for c in (self.comp_names or ()))
        object.__setattr__(self, "comp_names", cn)


_VECTOR_FIELD_REGISTRY: Dict[str, VectorFieldSpec] = {}


def register_vector_field(
    name: str,
    base_out_idx: int,
    *,
    n_comp: int = 3,
    comp_names: Tuple[str, ...] | None = None,
    overwrite: bool = True,
) -> VectorFieldSpec:
    """Register a named (vector) field so Field()/DField()/D2Field() can resolve out_idx.

    Examples
    --------
    >>> register_vector_field('E', base_out_idx=0)
    >>> register_vector_field('B', base_out_idx=3)
    >>> Ex = Field('E', 'x')
    >>> By = Field('B', 'y')
    """

    key = str(name)
    if (not overwrite) and (key in _VECTOR_FIELD_REGISTRY):
        return _VECTOR_FIELD_REGISTRY[key]

    n_comp_i = int(n_comp)
    if comp_names is None:
        if n_comp_i == 1:
            comp_names = ("",)
        elif n_comp_i == 2:
            comp_names = ("x", "y")
        elif n_comp_i == 3:
            comp_names = ("x", "y", "z")
        else:
            comp_names = tuple(f"c{i}" for i in range(n_comp_i))

    comp_names_tup = tuple(str(c) for c in comp_names)
    if len(comp_names_tup) != n_comp_i:
        raise ValueError(
            f"comp_names length ({len(comp_names_tup)}) must match n_comp ({n_comp_i}) for field '{key}'"
        )

    spec = VectorFieldSpec(
        name=key,
        base_out_idx=int(base_out_idx),
        n_comp=n_comp_i,
        comp_names=comp_names_tup,
    )
    _VECTOR_FIELD_REGISTRY[key] = spec
    return spec


def unregister_vector_field(name: str) -> None:
    _VECTOR_FIELD_REGISTRY.pop(str(name), None)


def clear_vector_fields() -> None:
    _VECTOR_FIELD_REGISTRY.clear()


def get_vector_field_spec(name: str) -> Optional[VectorFieldSpec]:
    return _VECTOR_FIELD_REGISTRY.get(str(name), None)


def list_vector_fields() -> List[VectorFieldSpec]:
    return list(_VECTOR_FIELD_REGISTRY.values())


def _resolve_vector_field_spec(
    name: str,
    *,
    base_out_idx: int | None = None,
    n_comp: int = 3,
    comp_names: Tuple[str, ...] | None = None,
) -> VectorFieldSpec:
    spec = _VECTOR_FIELD_REGISTRY.get(str(name), None)
    if spec is not None:
        if base_out_idx is not None and int(base_out_idx) != int(spec.base_out_idx):
            raise ValueError(
                f"Field '{name}' is registered with base_out_idx={spec.base_out_idx}, but base_out_idx={base_out_idx} was provided."
            )
        return spec

    if base_out_idx is None:
        raise ValueError(
            f"Unknown field '{name}'. Register it with register_vector_field(...) or pass base_out_idx=..."
        )

    # Ephemeral spec (not registered)
    n_comp_i = int(n_comp)
    if comp_names is None:
        if n_comp_i == 1:
            comp_names = ("",)
        elif n_comp_i == 2:
            comp_names = ("x", "y")
        elif n_comp_i == 3:
            comp_names = ("x", "y", "z")
        else:
            comp_names = tuple(f"c{i}" for i in range(n_comp_i))

    comp_names_tup = tuple(str(c) for c in comp_names)
    if len(comp_names_tup) != n_comp_i:
        raise ValueError(
            f"comp_names length ({len(comp_names_tup)}) must match n_comp ({n_comp_i}) for field '{name}'"
        )

    return VectorFieldSpec(
        name=str(name),
        base_out_idx=int(base_out_idx),
        n_comp=n_comp_i,
        comp_names=comp_names_tup,
    )


def _resolve_field_component(spec: VectorFieldSpec, comp: Comp) -> Tuple[int, str]:
    if isinstance(comp, str):
        comp_s = comp
        if comp_s in spec.comp_names:
            i = spec.comp_names.index(comp_s)
            return int(i), str(comp_s)
        try:
            i = int(comp_s)
        except Exception:
            raise ValueError(
                f"Invalid component '{comp}' for field '{spec.name}'. Valid names: {spec.comp_names} or integer index."
            )
        if i < 0 or i >= int(spec.n_comp):
            raise ValueError(
                f"Component index {i} out of range for field '{spec.name}' (n_comp={spec.n_comp})."
            )
        nm = spec.comp_names[i] if i < len(spec.comp_names) else str(i)
        return int(i), str(nm)

    i = int(comp)
    if i < 0 or i >= int(spec.n_comp):
        raise ValueError(
            f"Component index {i} out of range for field '{spec.name}' (n_comp={spec.n_comp})."
        )
    nm = spec.comp_names[i] if i < len(spec.comp_names) else str(i)
    return int(i), str(nm)


def Field(
    name: str,
    comp: Comp = 0,
    *,
    base_out_idx: int | None = None,
    n_comp: int = 3,
    comp_names: Tuple[str, ...] | None = None,
    tag: str | None = None,
) -> AtomNode:
    """Convenience constructor for a *named* field component.

    This returns an AtomNode(kind='u') with kwargs containing at least:
      - out_idx (unless it is 0)
      - field, comp, comp_name (for pretty printing)

    Typical usage is to register fields once:

    >>> register_vector_field('E', base_out_idx=0)
    >>> register_vector_field('B', base_out_idx=3)
    >>> Ex = Field('E', 'x')
    >>> By = Field('B', 'y')

    For quick scripts you can also bypass the registry:

    >>> Ex = Field('E', 'x', base_out_idx=0)
    """

    spec = _resolve_vector_field_spec(
        name,
        base_out_idx=base_out_idx,
        n_comp=n_comp,
        comp_names=comp_names,
    )
    comp_i, comp_nm = _resolve_field_component(spec, comp)
    out_idx = int(spec.base_out_idx) + int(comp_i)

    kw: Dict[str, Any] = {
        "field": str(spec.name),
        "comp": int(comp_i),
        "comp_name": str(comp_nm),
    }
    kw["out_idx"] = int(out_idx)

    return AtomNode(kind="u", var_idxs=(), kwargs=kw, tag=tag)


def DField(
    name: str,
    axis: int,
    comp: Comp = 0,
    *,
    base_out_idx: int | None = None,
    n_comp: int = 3,
    comp_names: Tuple[str, ...] | None = None,
    tag: str | None = None,
) -> AtomNode:
    """First derivative of a named field component: ∂Field/∂x_axis."""

    spec = _resolve_vector_field_spec(
        name,
        base_out_idx=base_out_idx,
        n_comp=n_comp,
        comp_names=comp_names,
    )
    comp_i, comp_nm = _resolve_field_component(spec, comp)
    out_idx = int(spec.base_out_idx) + int(comp_i)

    kw: Dict[str, Any] = {
        "axis": int(axis),
        "field": str(spec.name),
        "comp": int(comp_i),
        "comp_name": str(comp_nm),
    }
    kw["out_idx"] = int(out_idx)

    return AtomNode(kind="du", var_idxs=(), kwargs=kw, tag=tag)


def D2Field(
    name: str,
    axis0: int,
    axis1: int,
    comp: Comp = 0,
    *,
    base_out_idx: int | None = None,
    n_comp: int = 3,
    comp_names: Tuple[str, ...] | None = None,
    tag: str | None = None,
) -> AtomNode:
    """Second derivative of a named field component: ∂²Field/∂x_axis0∂x_axis1."""

    spec = _resolve_vector_field_spec(
        name,
        base_out_idx=base_out_idx,
        n_comp=n_comp,
        comp_names=comp_names,
    )
    comp_i, comp_nm = _resolve_field_component(spec, comp)
    out_idx = int(spec.base_out_idx) + int(comp_i)

    kw: Dict[str, Any] = {
        "axis0": int(axis0),
        "axis1": int(axis1),
        "field": str(spec.name),
        "comp": int(comp_i),
        "comp_name": str(comp_nm),
    }
    kw["out_idx"] = int(out_idx)

    return AtomNode(kind="d2u", var_idxs=(), kwargs=kw, tag=tag)


@dataclass(frozen=True)
class VField:
    """Syntactic sugar wrapper for named (vector) fields.

    This is a thin wrapper around Field / DField / D2Field so you can write:

        register_vector_field('E', base_out_idx=0)
        E = VField('E')
        Ex = E('x')
        dEy_dx0 = E.d(0, 'y')
        d2Ez_dx0dx1 = E.d2(0, 1, 'z')

    Notes
    -----
    * If the field is not registered, you can pass base_out_idx=... to VField
      for quick one-off scripts.
    * The returned nodes are still scalar AtomNodes, carrying `out_idx` in kwargs.
    """

    name: str
    base_out_idx: int | None = None
    n_comp: int = 3
    comp_names: Tuple[str, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, 'name', str(self.name))
        if self.base_out_idx is not None:
            object.__setattr__(self, 'base_out_idx', int(self.base_out_idx))
        object.__setattr__(self, 'n_comp', int(self.n_comp))
        if self.comp_names is not None:
            object.__setattr__(self, 'comp_names', tuple(str(c) for c in self.comp_names))

    def spec(self) -> VectorFieldSpec:
        return _resolve_vector_field_spec(
            self.name,
            base_out_idx=self.base_out_idx,
            n_comp=self.n_comp,
            comp_names=self.comp_names,
        )

    def __call__(self, comp: Comp = 0, *, tag: str | None = None) -> AtomNode:
        return Field(
            self.name,
            comp,
            base_out_idx=self.base_out_idx,
            n_comp=self.n_comp,
            comp_names=self.comp_names,
            tag=tag,
        )

    def d(self, axis: int, comp: Comp = 0, *, tag: str | None = None) -> AtomNode:
        return DField(
            self.name,
            axis,
            comp,
            base_out_idx=self.base_out_idx,
            n_comp=self.n_comp,
            comp_names=self.comp_names,
            tag=tag,
        )

    def d2(self, axis0: int, axis1: int, comp: Comp = 0, *, tag: str | None = None) -> AtomNode:
        return D2Field(
            self.name,
            axis0,
            axis1,
            comp,
            base_out_idx=self.base_out_idx,
            n_comp=self.n_comp,
            comp_names=self.comp_names,
            tag=tag,
        )

    def __repr__(self):
        return f"VField({self.name!r})"


def U(*, out_idx: int = 0, tag: str | None = None) -> AtomNode:
    """Convenience constructor for the fitted field component u[out_idx](x)."""
    kw: Dict[str, Any] = {}
    kw["out_idx"] = int(out_idx)
    return AtomNode(kind="u", var_idxs=(), kwargs=kw, tag=tag)


def DU(axis: int, *, out_idx: int = 0, tag: str | None = None) -> AtomNode:
    """Convenience constructor for a first derivative of u[out_idx]: ∂u[out_idx]/∂x_axis."""
    kw: Dict[str, Any] = {"axis": int(axis)}
    kw["out_idx"] = int(out_idx)
    return AtomNode(kind="du", var_idxs=(), kwargs=kw, tag=tag)


def D2U(axis0: int, axis1: int, *, out_idx: int = 0, tag: str | None = None) -> AtomNode:
    """Convenience constructor for a second derivative of u[out_idx]: ∂²u[out_idx]/∂x_axis0∂x_axis1."""
    kw: Dict[str, Any] = {"axis0": int(axis0), "axis1": int(axis1)}
    kw["out_idx"] = int(out_idx)
    return AtomNode(kind="d2u", var_idxs=(), kwargs=kw, tag=tag)


def FreeConst(name: str, *, tag: str | None = None, init: float = 1.0, scope: str = "experiment") -> AtomNode:
    """Convenience constructor for a trainable scalar free constant leaf."""
    if tag is None:
        tag = str(name)
    atom = AtomNode(
        kind="free_const", var_idxs=(), kwargs={"name": str(name), "init": float(init)}, tag=tag
    )
    atom.scope = scope
    return atom

def Scale(name: str = "s", *, tag: str | None = None, init: float = 1.0) -> AtomNode:
    """Convenience constructor for a trainable *dimensionless* scalar parameter."""
    if tag is None:
        tag = str(name)
    return AtomNode(kind="scale", var_idxs=(), kwargs={"name": str(name), "init": float(init)}, tag=tag)


def FixedConst(name: str, *, value: float, tag: str | None = None) -> AtomNode:
    """Convenience constructor for a fixed (non-trainable) scalar constant leaf."""
    if tag is None:
        tag = str(name)
    return AtomNode(kind="fixed_const", var_idxs=(), kwargs={"name": str(name), "value": float(value)}, tag=tag)



# ──────────────────────────────────────────────────────────────
# AST Tagging utilities (Phase 2 Task 1)
# ──────────────────────────────────────────────────────────────


def ensure_atom_tag(atom: AtomNode, context: str = "") -> AtomNode:
    """
    Ensure an AtomNode has a deterministic tag.

    If the atom already has a tag, return it unchanged.
    If tag is None, generate a deterministic tag based on:
    - kind (nn, poly, sin, exp_poly, etc.)
    - var_idxs (which variables it depends on)
    - context (path in AST or role)

    Parameters
    ----------
    atom : AtomNode
        The atom to ensure has a tag.
    context : str
        Additional context for tag generation (e.g., "L" for left branch,
        "R" for right branch, or semantic role like "tsr_num").

    Returns
    -------
    tagged_atom : AtomNode
        The atom with a guaranteed non-None tag.

    Examples
    --------
    >>> atom = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag=None)
    >>> tagged = ensure_atom_tag(atom, context="L")
    >>> tagged.tag  # "poly_x0_L"

    >>> atom_with_tag = AtomNode(kind="nn", var_idxs=(1,2), tag="my_tag")
    >>> ensure_atom_tag(atom_with_tag)  # Returns unchanged
    AtomNode(kind='nn', var_idxs=(1, 2), tag='my_tag')
    """
    if atom.tag is not None:
        return atom

    # Generate deterministic tag: kind_varstr_context
    var_str = "_".join(f"x{int(v)}" for v in atom.var_idxs)

    # Sanitize context (remove special chars, limit length)
    context_clean = "".join(c for c in context if c.isalnum() or c in "_-")[:20]

    if context_clean:
        tag = f"{atom.kind}_{var_str}_{context_clean}"
    else:
        tag = f"{atom.kind}_{var_str}"

    return AtomNode(
        kind=atom.kind,
        var_idxs=atom.var_idxs,
        kwargs=dict(atom.kwargs),  # Copy kwargs
        tag=tag,
        inputs=(
            None
            if atom.inputs is None
            else tuple(clone_ast(inp) for inp in atom.inputs)
        ),
        scope=atom.scope,
    )


def auto_tag_ast(root: Node, prefix: str = "", _counter: Dict[str, int] = None) -> Node:
    """
    Recursively ensure all AtomNodes in an AST have tags.

    Generates deterministic tags based on AST path (L/R for left/right branches)
    and position. If an atom already has a tag, it is preserved.

    Parameters
    ----------
    root : Node
        The root of the AST to tag.
    prefix : str
        Current path prefix (e.g., "L", "R", "LL", "LR", etc.)
    _counter : dict
        Internal counter for ensuring unique tags. Do not pass explicitly.

    Returns
    -------
    tagged_root : Node
        AST with all atoms guaranteed to have tags.

    Examples
    --------
    >>> atom1 = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag=None)
    >>> atom2 = AtomNode(kind="poly", var_idxs=(1,), kwargs={"degree": 1}, tag=None)
    >>> root = AddNode(atom1, atom2)
    >>> tagged = auto_tag_ast(root)
    >>> # atom1 gets tag like "poly_x0_L"
    >>> # atom2 gets tag like "poly_x1_R"
    """
    if _counter is None:
        _counter = {}

    if isinstance(root, AtomNode):
        # Ensure this atom has a tag
        if root.tag is None:
            # Generate context from prefix and counter
            kind_key = f"{root.kind}_{'_'.join(str(v) for v in root.var_idxs)}"
            count = _counter.get(kind_key, 0)
            _counter[kind_key] = count + 1

            if count > 0:
                context = f"{prefix}_{count}" if prefix else f"{count}"
            else:
                context = prefix

            return ensure_atom_tag(root, context=context)
        return root

    elif isinstance(root, AddNode):
        left = auto_tag_ast(root.left, f"{prefix}L", _counter)
        right = auto_tag_ast(root.right, f"{prefix}R", _counter)
        return AddNode(left, right)

    elif isinstance(root, MulNode):
        left = auto_tag_ast(root.left, f"{prefix}L", _counter)
        right = auto_tag_ast(root.right, f"{prefix}R", _counter)
        return MulNode(left, right)

    elif isinstance(root, PowNode):
        base = auto_tag_ast(root.base, f"{prefix}P", _counter)
        return PowNode(base=base, exponent=root.exponent)

    elif isinstance(root, LogNode):
        arg = auto_tag_ast(root.arg, f"{prefix}log", _counter)
        return LogNode(arg=arg)

    elif isinstance(root, SinNode):
        arg = auto_tag_ast(root.arg, f"{prefix}sin", _counter)
        return SinNode(arg=arg)

    elif isinstance(root, CosNode):
        arg = auto_tag_ast(root.arg, f"{prefix}cos", _counter)
        return CosNode(arg=arg)

    elif isinstance(root, AsinNode):
        arg = auto_tag_ast(root.arg, f"{prefix}asin", _counter)
        return AsinNode(arg=arg)

    elif isinstance(root, AcosNode):
        arg = auto_tag_ast(root.arg, f"{prefix}acos", _counter)
        return AcosNode(arg=arg)

    elif isinstance(root, AtanNode):
        arg = auto_tag_ast(root.arg, f"{prefix}atan", _counter)
        return AtanNode(arg=arg)

    elif isinstance(root, ExpNode):
        arg = auto_tag_ast(root.arg, f"{prefix}exp", _counter)
        return ExpNode(arg=arg)

    elif isinstance(root, (ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        arg = auto_tag_ast(root.arg, f"{prefix}u", _counter)
        return type(root)(arg=arg)

    elif isinstance(root, ConstNode):
        return root  # ConstNodes don't need tagging

    else:
        raise TypeError(f"Unknown node type: {type(root)}")


def check_fully_tagged(root: Node) -> Tuple[bool, List[str]]:
    """
    Check if all atoms in an AST have tags.

    Parameters
    ----------
    root : Node
        The AST root to check.

    Returns
    -------
    is_fully_tagged : bool
        True if all atoms have non-None tags.
    untagged_paths : list[str]
        List of paths to untagged atoms (empty if fully tagged).

    Examples
    --------
    >>> atom1 = AtomNode(kind="poly", var_idxs=(0,), tag="p0")
    >>> atom2 = AtomNode(kind="poly", var_idxs=(1,), tag=None)
    >>> root = AddNode(atom1, atom2)
    >>> is_tagged, paths = check_fully_tagged(root)
    >>> is_tagged  # False
    >>> paths  # ['right: poly(x1)']
    """
    untagged = []

    def traverse(node: Node, path: str = "root"):
        if isinstance(node, AtomNode):
            if node.tag is None:
                untagged.append(f"{path}: {node}")
            return

        if isinstance(node, (AddNode, MulNode)):
            traverse(node.left, f"{path}.left")
            traverse(node.right, f"{path}.right")
        elif isinstance(node, PowNode):
            traverse(node.base, f"{path}.base")
        elif isinstance(node, (LogNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ExpNode)):
            traverse(node.arg, f"{path}.arg")

    traverse(root)
    return len(untagged) == 0, untagged


# ──────────────────────────────────────────────────────────────
# Atom builders (SR-atom → torch.nn.Module)
# ──────────────────────────────────────────────────────────────


def _safe_init_poly_like_core(core):
    """
    Conservative initialisation for polynomial / rational / exp-of-rational
    cores, to avoid immediate NaNs / infs when they sit under sqrt or 1/sqrt.

    Strategy:
      - For PolyLeaf: P(x) ≈ 1 + small noise
      - For RationalPolyLeaf: P/Q with P≈1, Q≈1 + tiny noise (so P/Q>0)
      - For ExpPolyLeaf: exponent ≈ 0 + small noise  (exp ~ 1)
      - For ExpRationalPolyLeaf: exponent ≈ 0 via P≈0, Q≈1
    """
    with torch.no_grad():
        # ---- Plain polynomial ----
        if isinstance(core, (PolyLeaf, PolyLogLeaf, RPolyLeaf, RPolyLogLeaf)):
            # PolyLeaf and PolyLogLeaf share the same coeff / exps layout.
            exps = core.exps.detach()
            coeffs = core.coeffs
            coeffs.zero_()

            idx_const = None
            for k, e in enumerate(exps):
                if int(e.sum().item()) == 0:
                    idx_const = k
                    break

            if idx_const is not None:
                # 1 + 0.01 * N(0,1)
                coeffs[idx_const] = 1.0 + 0.01 * torch.randn(
                    (), device=coeffs.device, dtype=coeffs.dtype
                )

            if coeffs.numel() > 1:
                noise = 0.01 * torch.randn_like(coeffs)
                if idx_const is not None:
                    noise[idx_const] = 0.0
                coeffs.add_(noise)

        # ---- Rational polynomial P/Q ----
        elif isinstance(core, (RationalPolyLeaf, RRationalPolyLeaf)):
            exps_num = core.exps_num.detach()
            exps_den = core.exps_den.detach()
            cn = core.coeffs_num
            cd = core.coeffs_den
            cn.zero_()
            cd.zero_()

            idx_num_const = None
            for k, e in enumerate(exps_num):
                if int(e.sum().item()) == 0:
                    idx_num_const = k
                    break
            idx_den_const = None
            for k, e in enumerate(exps_den):
                if int(e.sum().item()) == 0:
                    idx_den_const = k
                    break

            # Denominator ~ 1 (positive, to keep P/Q well-defined)
            if idx_den_const is not None and cd.numel() > 0:
                cd[idx_den_const] = 1.0 + 0.01 * torch.randn((), device=cd.device, dtype=cd.dtype)
            elif cd.numel() > 0:
                cd[0] = 1.0

            # Numerator ~ 1 with very small noise so that P/Q > 0 initially
            if idx_num_const is not None and cn.numel() > 0:
                cn[idx_num_const] = 1.0
            elif cn.numel() > 0:
                cn[0] = 1.0

            if cn.numel() > 1:
                cn_noise = 1e-4 * torch.randn_like(cn)
                if idx_num_const is not None:
                    cn_noise[idx_num_const] = 0.0
                cn.add_(cn_noise)

            if cd.numel() > 1:
                cd_noise = 1e-4 * torch.randn_like(cd)
                if idx_den_const is not None:
                    cd_noise[idx_den_const] = 0.0
                cd.add_(cd_noise)

        # ---- exp(poly) ----
        elif isinstance(core, (ExpPolyLeaf, RExpPolyLeaf)):
            # Exponent small ⇒ exp ≈ 1
            coeffs = core.coeffs
            coeffs.zero_()
            if coeffs.numel() > 0:
                coeffs.add_(1e-3 * torch.randn_like(coeffs))

        # ---- exp(rational) ----
        elif isinstance(core, ExpRationalPolyLeaf):
            exps_num = core.exps_num.detach()
            exps_den = core.exps_den.detach()
            cn = core.coeffs_num
            cd = core.coeffs_den
            cn.zero_()
            cd.zero_()

            idx_den_const = None
            for k, e in enumerate(exps_den):
                if int(e.sum().item()) == 0:
                    idx_den_const = k
                    break

            # Q(x) ≈ 1, P(x) ≈ 0 ⇒ exponent ≈ 0
            if idx_den_const is not None and cd.numel() > 0:
                cd[idx_den_const] = 1.0
            elif cd.numel() > 0:
                cd[0] = 1.0

            if cn.numel() > 0:
                cn.add_(1e-3 * torch.randn_like(cn))


def _build_leaf_module(atom: AtomNode, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    """
    Map an AtomNode(kind, var_idxs, kwargs) to a *core* nn.Module.

    The caller will wrap this with AutogradAdaptor so it plugs into
    ASTCompositeAdaptor naturally.
    """
    kind = atom.kind.lower()
    kw = dict(atom.kwargs)
    # Internal metadata keys (used by rewrite passes) must not reach leaf constructors.
    for _k in list(kw.keys()):
        if str(_k).startswith('_'):
            kw.pop(_k, None)
    # Compound-variable atoms carry a mapping z(x) (and optional extra raw axes).
    # Leaf modules must not receive the mapping itself, only the effective input dimension.
    kw.pop("compound", None)
    kw.pop("input_expr", None)
    kw.pop("extra_var_idxs", None)
    n_in = atom.n_in

    # Raw variable leaf (identity).
    if kind in ("var", "x", "input"):
        return VarLeaf()

    # Trainable free constant leaf.
    # The constant's *units* are tracked at the AST level by sr_core.units.
    if kind in ("free_const", "freeconst", "free_constant"):
        if n_in != 0:
            raise ValueError(f"FreeConstLeaf expects 0 inputs; got {n_in} for {atom}")
        init = float(kw.pop("init", kw.pop("value_init", 1.0)))
        return FreeConstLeaf(init=init, dtype=dtype, device=device)
    # Gauge-fixing scale leaf (dimensionless 0-input constant).
    if kind in ("scale", "mul_scale"):
        if n_in != 0:
            raise ValueError(f"Scale leaf expects 0 inputs; got {n_in} for {atom}")
        init = float(kw.pop("init", kw.pop("value_init", 1.0)))
        return FreeConstLeaf(init=init, dtype=dtype, device=device)



    # Fixed (non-trainable) physical constant leaf.
    # The constant's units are tracked at the AST level by sr_core.units.
    if kind in ("fixed_const", "fixedconst", "fixed_constant"):
        if n_in != 0:
            raise ValueError(f"FixedConstLeaf expects 0 inputs; got {n_in} for {atom}")
        kw.pop("name", kw.pop("const_name", ""))
        # Value is stored in kwargs as a plain float (and is not trainable).
        val = float(kw.pop("value", kw.pop("val", kw.pop("init", kw.pop("value_init", 1.0)))))
        return FixedConstLeaf(value=val, dtype=dtype, device=device)

    if kind in ("lin", "linear", "linpoly"):
        # Linear combination (dimensionless weights, no bias)
        core = LinLeaf(n_in=n_in, dtype=dtype, device=device, **kw)
        return core

    if kind in ("sin", "sin_linear", "sinlin"):
        if n_in != 1:
            raise ValueError(f"SinLinearLeaf currently expects 1 input; got {n_in} for {atom}")
        return SinLinearLeaf(n_in=1, dtype=dtype, device=device, **kw)

    if kind in ("tanh", "tanh_linear", "tanhlin"):
        if n_in != 1:
            raise ValueError(f"TanhLinearLeaf currently expects 1 input; got {n_in} for {atom}")
        return TanhLinearLeaf(n_in=1, dtype=dtype, device=device, **kw)

    if kind in ("poly", "polynomial"):
        deg = int(kw.pop("degree", kw.pop("deg", 2)))
        init_coeffs = kw.pop("init_coeffs", None)
        core = PolyLeaf(n_in=n_in, degree=deg, dtype=dtype, device=device, **kw)
        if init_coeffs is not None:
            # Use provided initialization coefficients
            with torch.no_grad():
                coeffs_t = torch.tensor(init_coeffs, dtype=dtype, device=device)
                if coeffs_t.numel() == core.coeffs.numel():
                    core.coeffs.copy_(coeffs_t)
                elif coeffs_t.numel() < core.coeffs.numel():
                    # Partial init: set first N coeffs, zero the rest
                    core.coeffs.zero_()
                    core.coeffs.view(-1)[:coeffs_t.numel()].copy_(coeffs_t.view(-1))
                # else: coeffs_t has more elements than core - just copy what fits
                else:
                    core.coeffs.copy_(coeffs_t.view(-1)[:core.coeffs.numel()])
        else:
            _safe_init_poly_like_core(core)
        return core

    if kind in ("rpoly", "rpolynomial", "r_polynomial"):
        deg = int(kw.pop("degree", kw.pop("deg", 2)))
        init_coeffs = kw.pop("init_coeffs", None)
        core = RPolyLeaf(n_in=n_in, degree=deg, dtype=dtype, device=device, **kw)
        if init_coeffs is not None:
            with torch.no_grad():
                c = torch.tensor(init_coeffs, dtype=dtype, device=device).view(-1)
                # Accept either free-coeff vector length or full-coeff length.
                if c.numel() == core.coeffs.numel():
                    core.coeffs.copy_(c)
                elif c.numel() == core.exps_full.shape[0]:
                    lead = float(c[int(core.lead_pos)].item())
                    if abs(lead) > 1e-16 and core.coeffs.numel() > 0:
                        free = c[core.free_pos] / lead
                        core.coeffs.copy_(free)
                elif c.numel() > core.coeffs.numel() and core.coeffs.numel() > 0:
                    core.coeffs.copy_(c[: core.coeffs.numel()])
        else:
            _safe_init_poly_like_core(core)
        return core

    if kind in ("polylog", "polylogarithmic", "logpoly"):
        # Polynomial in log(x) implemented by PolyLogLeaf
        deg = int(kw.pop("degree", kw.pop("deg", 2)))
        core = PolyLogLeaf(n_in=n_in, degree=deg, dtype=dtype, device=device, **kw)
        _safe_init_poly_like_core(core)
        return core

    if kind in ("rpolylog", "rlogpoly"):
        deg = int(kw.pop("degree", kw.pop("deg", 2)))
        core = RPolyLogLeaf(n_in=n_in, degree=deg, dtype=dtype, device=device, **kw)
        _safe_init_poly_like_core(core)
        return core

    if kind in ("logshifted", "log_shifted", "logshift"):
        # Shifted logarithm: a*log(x - b) + c
        from nestynet_sr.sr_core.atoms import LogShiftedLeaf
        if n_in != 1:
            raise ValueError(f"LogShiftedLeaf expects 1 input; got {n_in} for {atom}")
        return LogShiftedLeaf(n_in=1, dtype=dtype, device=device, **kw)

    if kind in ("ratpoly", "rational_poly", "rationalpolynomial"):
        deg_num = int(kw.pop("deg_num", kw.pop("deg_n", 2)))
        deg_den = int(kw.pop("deg_den", kw.pop("deg_d", 1)))

        # ASTCompositeAdaptor will already slice x down to shape [B, n_in]
        # based on expr_tokens. Inside this leaf we want to treat those
        # n_in coordinates as local variables 0..n_in-1.
        local_indices = tuple(range(n_in))

        core = RationalPolyLeaf(
            indices=local_indices,
            deg_num=deg_num,
            deg_den=deg_den,
            dtype=dtype,
            device=device,
            **kw,
        )
        _safe_init_poly_like_core(core)
        return core

    if kind in ("rratpoly", "rrational_poly", "rrationalpolynomial"):
        deg_num = int(kw.pop("deg_num", kw.pop("deg_n", 2)))
        deg_den = int(kw.pop("deg_den", kw.pop("deg_d", 1)))
        local_indices = tuple(range(n_in))
        core = RRationalPolyLeaf(
            indices=local_indices,
            deg_num=deg_num,
            deg_den=deg_den,
            dtype=dtype,
            device=device,
            **kw,
        )
        _safe_init_poly_like_core(core)
        return core

    # power-law leaf
    if kind in ("pow", "power", "powerleaf", "power_law"):
        if n_in != 1:
            raise ValueError(f"PowerLeaf currently expects 1 input; got {n_in} for {atom}")
        exp_init = float(kw.pop("exponent_init", 1.0))
        return PowerLeaf(
            n_in=1,
            exponent_init=exp_init,
            dtype=dtype,
            device=device,
            **kw,
        )

    # inverse monomial leaf: a/x^degree with fixed degree
    if kind in ("inv_monomial", "inverse_monomial", "inv_mono"):
        if n_in != 1:
            raise ValueError(f"InverseMonomialLeaf expects 1 input; got {n_in} for {atom}")
        degree = int(kw.pop("degree", 1))
        return InverseMonomialLeaf(
            n_in=1,
            degree=degree,
            dtype=dtype,
            device=device,
            **kw,
        )

    # reduced inverse monomial leaf: 1/x^degree with NO learnable params
    if kind in ("rinv_monomial", "r_inv_monomial", "rinverse_monomial"):
        if n_in != 1:
            raise ValueError(f"RInverseMonomialLeaf expects 1 input; got {n_in} for {atom}")
        degree = int(kw.pop("degree", 1))
        return RInverseMonomialLeaf(
            n_in=1,
            degree=degree,
            dtype=dtype,
            device=device,
            **kw,
        )

    # exp-of-polynomial leaf
    if kind in ("exp", "exp_poly", "expquad", "exp_poly_leaf"):
        deg = int(kw.pop("degree", kw.pop("deg", 2)))
        core = ExpPolyLeaf(
            n_in=n_in,
            degree=deg,
            dtype=dtype,
            device=device,
            **kw,
        )
        _safe_init_poly_like_core(core)
        return core

    # reduced exp-of-polynomial leaf (pinned exponent constant term)
    if kind in ("rexp", "rexp_poly", "r_exp_poly", "rexpquad", "rexp_poly_leaf"):
        deg = int(kw.pop("degree", kw.pop("deg", 2)))
        core = RExpPolyLeaf(
            n_in=n_in,
            degree=deg,
            dtype=dtype,
            device=device,
            **kw,
        )
        _safe_init_poly_like_core(core)
        return core

    # exp-of-rational-polynomial leaf
    if kind in ("exp_ratpoly", "exp_rational", "exprat"):
        deg_num = int(kw.pop("deg_num", 2))
        deg_den = int(kw.pop("deg_den", 1))
        core = ExpRationalPolyLeaf(
            n_in=n_in,
            deg_num=deg_num,
            deg_den=deg_den,
            dtype=dtype,
            device=device,
            **kw,
        )
        _safe_init_poly_like_core(core)
        return core

    # Polynomial in a ratio of two variables: poly(x_num / x_den)
    # Used for ratio-invariance (homogeneous degree-0 functions)
    if kind in ("ratio_poly", "ratiopoly", "ratio_polynomial"):
        if n_in != 2:
            raise ValueError(
                f"RatioPolyLeaf expects exactly 2 inputs (numerator, denominator); got {n_in} for {atom}"
            )
        deg = int(kw.pop("degree", kw.pop("deg", 2)))
        core = RatioPolyLeaf(
            degree=deg,
            dtype=dtype,
            device=device,
            **kw,
        )
        # Initialize with constant 1.0 for stable starting point
        with torch.no_grad():
            if core.coeffs.numel() > 0:
                core.coeffs[0].fill_(1.0)
        return core

    if kind in ("rratio_poly", "rratiopoly", "rratio_polynomial"):
        if n_in != 2:
            raise ValueError(
                f"RRatioPolyLeaf expects exactly 2 inputs (numerator, denominator); got {n_in} for {atom}"
            )
        deg = int(kw.pop("degree", kw.pop("deg", 2)))
        core = RRatioPolyLeaf(degree=deg, dtype=dtype, device=device, **kw)
        # Monic: leading coeff fixed to 1; start with remaining coeffs at 0.
        return core

    # Planck / Bose–Einstein–like leaf
    if kind in ("planck", "bose", "bose_einstein", "planckleaf"):
        if n_in != 1:
            raise ValueError(f"PlanckLeaf currently expects 1 input; got {n_in} for {atom}")
        return PlanckLeaf(n_in=1, dtype=dtype, device=device, **kw)

    if kind in ("planck_full", "full_planck", "planckfull", "planck_full_leaf"):
        if n_in != 1:
            raise ValueError(f"PlanckFullLeaf currently expects 1 input; got {n_in} for {atom}")
        return PlanckFullLeaf(n_in=1, dtype=dtype, device=device, **kw)

    if kind in ("expm1", "expm1leaf", "exp_minus_one"):
        if n_in != 1:
            raise ValueError(f"Expm1Leaf currently expects 1 input; got {n_in} for {atom}")
        return Expm1Leaf(n_in=1, dtype=dtype, device=device, **kw)

    raise ValueError(f"Unknown atom kind: {atom.kind!r}")


# ──────────────────────────────────────────────────────────────
# AST → (leaves, expr_tokens) → ASTCompositeAdaptor
# ──────────────────────────────────────────────────────────────


def _module_input_arity(module: torch.nn.Module) -> int | None:
    """Best-effort input arity for reused leaves and adaptor wrappers."""

    seen: set[int] = set()

    def _walk(obj) -> int | None:
        if obj is None:
            return None
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        for attr in ("n_in", "Nx_size"):
            val = getattr(obj, attr, None)
            if val is not None:
                try:
                    return int(val)
                except Exception:
                    pass

        for attr in ("core", "base_model", "model", "stage0", "_stage0"):
            child = getattr(obj, attr, None)
            got = _walk(child)
            if got is not None:
                return got
        return None

    return _walk(module)


def _is_module_compatible_with_atom(module: torch.nn.Module, atom: AtomNode) -> bool:
    """
    Check if a reused module is compatible with an atom's kind.

    This prevents accidentally reusing a NN module when an analytic atom is expected,
    or vice versa.

    Parameters
    ----------
    module : torch.nn.Module
        The module from the reuse dict.
    atom : AtomNode
        The atom node requesting reuse.

    Returns
    -------
    compatible : bool
        True if the module type matches the atom kind.
    """
    kind = str(atom.kind).lower()

    # Extract the core from wrapped modules
    core = getattr(module, "core", getattr(module, "model", module))

    # Check n_in compatibility.  Some NN leaves are wrapped in adaptors where
    # the arity is exposed as base_model.Nx_size rather than core.n_in.
    expected_n_in = atom.n_in
    core_n_in = _module_input_arity(module)
    if core_n_in is not None and core_n_in != expected_n_in:
        return False

    # Check compatibility based on atom kind
    # Check compatibility based on atom kind
    if kind in ("poly", "polynomial"):
        return isinstance(core, PolyLeaf)
    elif kind in ("rpoly", "rpolynomial", "r_polynomial"):
        return isinstance(core, RPolyLeaf)
    elif kind in ("polylog", "polylogarithmic", "logpoly"):
        return isinstance(core, PolyLogLeaf)
    elif kind in ("rpolylog", "rlogpoly"):
        return isinstance(core, RPolyLogLeaf)
    elif kind in ("logshifted",):
        from nestynet_sr.sr_core.atoms import LogShiftedLeaf
        return isinstance(core, LogShiftedLeaf)
    elif kind in ("ratpoly", "rational_poly", "rationalpolynomial"):
        return isinstance(core, RationalPolyLeaf)
    elif kind in ("rratpoly", "rrational_poly", "rrationalpolynomial"):
        return isinstance(core, RRationalPolyLeaf)
    elif kind in ("ratio_poly", "ratiopoly", "ratio_polynomial"):
        return isinstance(core, RatioPolyLeaf)
    elif kind in ("rratio_poly", "rratiopoly", "rratio_polynomial"):
        return isinstance(core, RRatioPolyLeaf)
    elif kind in ("scale", "mul_scale"):
        return isinstance(core, FreeConstLeaf)
    elif kind in ("sin", "sin_linear"):
        return isinstance(core, SinLinearLeaf)
    elif kind in ("tanh", "tanh_linear"):
        return isinstance(core, TanhLinearLeaf)
    elif kind in ("pow", "power", "powerleaf", "power_law", "power"):
        return isinstance(core, PowerLeaf)
    elif kind in ("inv_monomial", "inverse_monomial", "inv_mono", "inv_monomial"):
        return isinstance(core, InverseMonomialLeaf)
    elif kind in ("rinv_monomial", "r_inv_monomial", "rinverse_monomial"):
        return isinstance(core, RInverseMonomialLeaf)
    elif kind in ("exp", "exp_poly", "expquad", "exp_poly_leaf", "exp_poly"):
        return isinstance(core, ExpPolyLeaf)
    elif kind in ("rexp", "rexp_poly", "r_exp_poly", "rexpquad", "rexp_poly_leaf"):
        return isinstance(core, RExpPolyLeaf)
    elif kind in ("exp_ratpoly", "exp_rational", "exprat", "exp_ratpoly"):
        return isinstance(core, ExpRationalPolyLeaf)
    elif kind in ("planck", "bose", "bose_einstein", "planckleaf", "planck"):
        return isinstance(core, PlanckLeaf)
    elif kind in ("planck_full", "full_planck", "planckfull", "planck_full_leaf"):
        return isinstance(core, PlanckFullLeaf)
    elif kind in ("expm1", "expm1leaf", "exp_minus_one", "expm1"):
        return isinstance(core, Expm1Leaf)
    elif kind in ("lin", "linear", "linpoly"):
        return isinstance(core, LinLeaf)
    elif kind in ("var", "x", "input"):
        return isinstance(core, VarLeaf)
    elif kind in ("free_const", "freeconst", "free_constant"):
        return isinstance(core, FreeConstLeaf)
    elif kind in ("fixed_const", "fixedconst", "fixed_constant"):
        return isinstance(core, FixedConstLeaf)
    elif kind == "nn":
        analytic_types = (
            PolyLeaf,
            RPolyLeaf,
            PolyLogLeaf,
            RPolyLogLeaf,
            RationalPolyLeaf,
            RRationalPolyLeaf,
            RatioPolyLeaf,
            RRatioPolyLeaf,
            SinLinearLeaf,
            TanhLinearLeaf,
            PowerLeaf,
            ExpPolyLeaf,
            RExpPolyLeaf,
            ExpRationalPolyLeaf,
            PlanckLeaf,
            PlanckFullLeaf,
        )
        return not isinstance(core, analytic_types)

    else:
        # Unknown kind - be conservative and reject reuse
        return False


def _build_leaves_from_ast(
    node: Node,
    *,
    dtype: torch.dtype,
    device: torch.device,
    nn_factory=None,
    atom_factory=None,
    reuse: Dict[str, torch.nn.Module] | None = None,
    return_atom_map: bool = False,
) -> List[torch.nn.Module] | tuple[List[torch.nn.Module], Dict[int, torch.nn.Module]]:
    """
    Build leaf modules in left-to-right depth-first order from AST.

    Parameters
    ----------
    node        : Node
        The AST node to compile.
    dtype       : torch.dtype
    device      : torch.device
    nn_factory  : callable | None
        Factory function for kind='nn' atoms: nn_factory(atom, existing) -> module.
        If None, kind='nn' atoms will raise an error.
    reuse       : dict[str, torch.nn.Module] | None
        Map from tag to existing module instances for reuse.
    return_atom_map : bool
        If True, return (leaves, atom_to_leaf) where atom_to_leaf maps
        id(atom) -> leaf module. Default False.

    Returns
    -------
    leaves : list[torch.nn.Module]
        Leaf modules in left-to-right depth-first order.
    atom_to_leaf : dict[int, torch.nn.Module] (if return_atom_map=True)
        Mapping from atom object id to its corresponding leaf module.
    """
    leaves: List[torch.nn.Module] = []
    atom_to_leaf: Dict[int, torch.nn.Module] = {}

    # Local tag->leaf map for this build (ASTCompositeAdaptor stores leaves in a list,
    # and does not provide a built-in tag->leaf dict).
    tag_to_leaf: Dict[str, torch.nn.Module] = {}

    # If we need to push a coefficient into a scale leaf before that scale leaf is
    # visited in DFS order, stash it here and apply once the build is complete.
    pending_scale_mul: Dict[str, float] = {}

    def _unwrap_leaf_core(mod: torch.nn.Module) -> torch.nn.Module:
        """Best-effort unwrapping of thin adaptors to reach the analytic core."""
        m = mod
        for _ in range(8):
            nxt = None
            if hasattr(m, "core"):
                try:
                    nxt = getattr(m, "core")
                except Exception:
                    nxt = None
            if nxt is None and hasattr(m, "model"):
                try:
                    nxt = getattr(m, "model")
                except Exception:
                    nxt = None
            if nxt is None and hasattr(m, "base_model"):
                try:
                    nxt = getattr(m, "base_model")
                except Exception:
                    nxt = None
            if nxt is None or nxt is m:
                break
            m = nxt
        return m

    def _mul_scale_by_tag(scale_tag, factor) -> None:
        """Multiply a tagged scale leaf's value by `factor` (best-effort)."""
        if scale_tag is None:
            return
        st = str(scale_tag)
        try:
            fac = float(factor)
        except Exception:
            return
        # NaN guard
        if not (fac == fac):
            return

        if st in tag_to_leaf:
            s_leaf = tag_to_leaf[st]
            s_core = _unwrap_leaf_core(s_leaf)
            if hasattr(s_core, "value"):
                try:
                    with torch.no_grad():
                        s_core.value.mul_(fac)
                except Exception:
                    pass
        else:
            pending_scale_mul[st] = float(pending_scale_mul.get(st, 1.0)) * fac

    def traverse(node: Node):
        if isinstance(node, AtomNode):
            # Local tag-sharing: reuse already-built leaf if same tag appears
            # multiple times in this AST (e.g. cloned FreeConst scale nodes).
            t = getattr(node, "tag", None)
            if t is not None:
                tt = str(t)
                if tt in tag_to_leaf:
                    existing_leaf = tag_to_leaf[tt]
                    try:
                        if _is_module_compatible_with_atom(existing_leaf, node):
                            leaves.append(existing_leaf)
                            atom_to_leaf[id(node)] = existing_leaf
                            return
                    except Exception:
                        pass

            # Allow callers to override leaf construction for custom atom kinds
            # (e.g. native DE/PDE feature atoms like u, du, d2u).
            if atom_factory is not None and node.kind.lower() != "nn":
                existing = None
                if node.tag is not None and reuse is not None:
                    existing = reuse.get(node.tag, None)
                leaf = atom_factory(node, existing)
                if leaf is not None:
                    leaves.append(leaf)
                    atom_to_leaf[id(node)] = leaf
                    if node.tag is not None:
                        tag_to_leaf[str(node.tag)] = leaf
                    return

            # Handle kind='nn' atoms via nn_factory
            if node.kind.lower() == "nn":
                if nn_factory is None:
                    raise ValueError("Atom kind 'nn' requires nn_factory")
                existing = None
                if node.tag is not None and reuse is not None:
                    existing = reuse.get(node.tag, None)
                leaf = nn_factory(node, existing)
                leaves.append(leaf)
                atom_to_leaf[id(node)] = leaf
                if node.tag is not None:
                    tag_to_leaf[str(node.tag)] = leaf
                return

            # Handle analytic atoms (poly, sin, ratpoly, etc.)
            # Check reuse first - but only if type-compatible
            can_reuse = False
            if node.tag is not None and reuse is not None and node.tag in reuse:
                existing_leaf = reuse[node.tag]
                # Check if the existing module is compatible with the desired atom kind
                can_reuse = _is_module_compatible_with_atom(existing_leaf, node)

            if can_reuse:
                leaf = existing_leaf
            else:
                # Create fresh analytic module
                # Note: if node.tag exists in reuse with incompatible type,
                # we keep it there for analytic initialization to use as teacher
                core = _build_leaf_module(node, dtype=dtype, device=device)
                if AutogradAdaptor is None:
                    raise ImportError(
                        "AutogradAdaptor is unavailable (nestynet not importable). "
                        "Compiling analytic leaves requires nestynet.adaptors.AutogradAdaptor."
                    )
                # Warm-start reduced polynomial-like leaves (option-B gauge fixing).
                # If the same tag previously referred to the non-reduced leaf,
                # initialise the reduced leaf and push the removed leading coefficient
                # into the shared multiplicative 'scale' leaf (if present).
                try:
                    teacher_leaf = None
                    if node.tag is not None and reuse is not None:
                        teacher_leaf = reuse.get(node.tag, None)
                    if teacher_leaf is not None:
                        teacher_core = _unwrap_leaf_core(teacher_leaf)
                        kw = (getattr(node, "kwargs", None) or {})
                        scale_tag = kw.get("_mul_scale_tag", None)

                        if isinstance(core, RPolyLeaf) and isinstance(teacher_core, PolyLeaf):
                            ok = True
                            for _attr in ("n_in", "degree", "min_total"):
                                a = getattr(teacher_core, _attr, None)
                                b = getattr(core, _attr, None)
                                if a is not None and b is not None and int(a) != int(b):
                                    ok = False
                                    break
                            if ok:
                                tc = teacher_core.coeffs.detach().to(
                                    device=core.coeffs.device, dtype=core.coeffs.dtype
                                ).view(-1)
                                if hasattr(core, "exps_full") and int(tc.numel()) == int(core.exps_full.shape[0]):
                                    lead = tc[int(core.lead_pos)]
                                    lead_f = float(lead.item())
                                    if abs(lead_f) > 1e-16:
                                        with torch.no_grad():
                                            if core.coeffs.numel() > 0:
                                                idx = core.free_pos.to(device=tc.device)
                                                core.coeffs.copy_(tc[idx] / lead)
                                        if scale_tag is not None:
                                            _mul_scale_by_tag(scale_tag, lead_f)

                        elif isinstance(core, RPolyLogLeaf) and isinstance(teacher_core, PolyLogLeaf):
                            ok = True
                            for _attr in ("n_in", "degree"):
                                a = getattr(teacher_core, _attr, None)
                                b = getattr(core, _attr, None)
                                if a is not None and b is not None and int(a) != int(b):
                                    ok = False
                                    break
                            if ok:
                                tc = teacher_core.coeffs.detach().to(
                                    device=core.coeffs.device, dtype=core.coeffs.dtype
                                ).view(-1)
                                if hasattr(core, "exps_full") and int(tc.numel()) == int(core.exps_full.shape[0]):
                                    lead = tc[int(core.lead_pos)]
                                    lead_f = float(lead.item())
                                    if abs(lead_f) > 1e-16:
                                        with torch.no_grad():
                                            if core.coeffs.numel() > 0:
                                                idx = core.free_pos.to(device=tc.device)
                                                core.coeffs.copy_(tc[idx] / lead)
                                        if scale_tag is not None:
                                            _mul_scale_by_tag(scale_tag, lead_f)



                        elif isinstance(core, RExpPolyLeaf) and isinstance(teacher_core, ExpPolyLeaf):
                            ok = True
                            for _attr in ("n_in", "degree"):
                                a = getattr(teacher_core, _attr, None)
                                b = getattr(core, _attr, None)
                                if a is not None and b is not None and int(a) != int(b):
                                    ok = False
                                    break
                            if ok:
                                tc = teacher_core.coeffs.detach().to(
                                    device=core.coeffs.device, dtype=core.coeffs.dtype
                                ).view(-1)
                                if hasattr(core, "exps_full") and int(tc.numel()) == int(core.exps_full.shape[0]):
                                    # Move the constant coefficient out of the exponent:
                                    #   exp(c0 + Q(x)) = exp(c0) * exp(Q(x)).
                                    c0 = tc[int(getattr(core, "const_pos", 0))]
                                    c0_f = float(c0.item())
                                    with torch.no_grad():
                                        if core.coeffs.numel() > 0:
                                            idx = core.free_pos.to(device=tc.device)
                                            core.coeffs.copy_(tc[idx])
                                    if scale_tag is not None:
                                        # Use the same clamp as the exp leaf to avoid overflow.
                                        try:
                                            clamp = float(getattr(teacher_core, "clamp", 60.0))
                                        except Exception:
                                            clamp = 60.0
                                        if clamp is not None:
                                            if c0_f > clamp:
                                                c0_f = clamp
                                            elif c0_f < -clamp:
                                                c0_f = -clamp
                                        try:
                                            fac = float(torch.exp(torch.tensor(c0_f)).item())
                                        except Exception:
                                            fac = None
                                        if fac is not None and fac == fac and fac != float("inf"):
                                            _mul_scale_by_tag(scale_tag, fac)
                        elif isinstance(core, RRationalPolyLeaf) and isinstance(teacher_core, RationalPolyLeaf):
                            ok = True
                            for _attr in ("deg_num", "deg_den"):
                                a = getattr(teacher_core, _attr, None)
                                b = getattr(core, _attr, None)
                                if a is not None and b is not None and int(a) != int(b):
                                    ok = False
                                    break
                            if ok and hasattr(core, "exps_num_full") and hasattr(core, "exps_den"):
                                tc_num = teacher_core.coeffs_num.detach().to(
                                    device=core.coeffs_num.device, dtype=core.coeffs_num.dtype
                                ).view(-1)
                                tc_den = teacher_core.coeffs_den.detach().to(
                                    device=core.coeffs_den.device, dtype=core.coeffs_den.dtype
                                ).view(-1)
                                if int(tc_num.numel()) == int(core.exps_num_full.shape[0]) and int(tc_den.numel()) == int(core.exps_den.shape[0]):
                                    lead = tc_num[int(core.lead_pos_num)]
                                    lead_f = float(lead.item())
                                    if abs(lead_f) > 1e-16:
                                        with torch.no_grad():
                                            if core.coeffs_num.numel() > 0:
                                                idx = core.free_pos_num.to(device=tc_num.device)
                                                core.coeffs_num.copy_(tc_num[idx] / lead)
                                            if core.coeffs_den.numel() == tc_den.numel():
                                                core.coeffs_den.copy_(tc_den)
                                            elif core.coeffs_den.numel() > 0:
                                                core.coeffs_den.copy_(tc_den[: core.coeffs_den.numel()])
                                        if scale_tag is not None:
                                            _mul_scale_by_tag(scale_tag, lead_f)

                        elif isinstance(core, RRatioPolyLeaf) and isinstance(teacher_core, RatioPolyLeaf):
                            ok = True
                            a = getattr(teacher_core, "degree", None)
                            b = getattr(core, "degree", None)
                            if a is not None and b is not None and int(a) != int(b):
                                ok = False
                            if ok:
                                tc = teacher_core.coeffs.detach().to(
                                    device=core.coeffs.device, dtype=core.coeffs.dtype
                                ).view(-1)
                                if int(tc.numel()) == int(core.degree + 1):
                                    lead = tc[-1]
                                    lead_f = float(lead.item())
                                    if abs(lead_f) > 1e-16:
                                        with torch.no_grad():
                                            if core.coeffs.numel() > 0:
                                                core.coeffs.copy_(tc[:-1] / lead)
                                        if scale_tag is not None:
                                            _mul_scale_by_tag(scale_tag, lead_f)
                except Exception:
                    pass
                leaf = AutogradAdaptor(core)

            leaves.append(leaf)
            atom_to_leaf[id(node)] = leaf
            if node.tag is not None:
                tag_to_leaf[str(node.tag)] = leaf
            return

        if isinstance(node, (AddNode, MulNode)):
            traverse(node.left)
            traverse(node.right)
            return
        if isinstance(node, PowNode):
            traverse(node.base)
            return
        if isinstance(node, LogNode):
            traverse(node.arg)
            return
        if isinstance(node, (SinNode, CosNode, AsinNode, AcosNode, AtanNode, ExpNode, AbsNode)):
            traverse(node.arg)
            return

        if isinstance(node, ConstNode):
            return  # ConstNodes are not leaves

        raise TypeError(f"Unsupported node type in AST: {type(node).__name__}")

    traverse(node)

    # Apply any deferred scale multipliers (in case the 'scale' atom appears
    # after reduced polynomial factors in DFS order).
    if pending_scale_mul:
        for _st, _fac in pending_scale_mul.items():
            if _st in tag_to_leaf:
                _s_leaf = tag_to_leaf[_st]
                _s_core = _unwrap_leaf_core(_s_leaf)
                if hasattr(_s_core, "value"):
                    try:
                        with torch.no_grad():
                            _s_core.value.mul_(float(_fac))
                    except Exception:
                        pass

    if return_atom_map:
        return leaves, atom_to_leaf
    return leaves


def build_composite_from_ast(
    root: Node,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    nn_factory=None,
    atom_factory=None,
    reuse: Dict[str, torch.nn.Module] | None = None,
    return_atom_map: bool = False,
):
    """
    Compile a symbolic AST into an ASTCompositeAdaptor.

    This is the unified entry point for both Stage A (NN leaves) and Stage B
    (analytic leaves). Atoms with kind='nn' are compiled via nn_factory,
    while analytic atoms (poly, sin, ratpoly) are wrapped in AutogradAdaptor.

    Parameters
    ----------
    root       : Node
        The root of the AST to compile.
    dtype      : torch.dtype
        Data type for leaf modules.
    device     : torch.device
        Device for leaf modules.
    nn_factory : callable | None
        Factory for kind='nn' atoms: nn_factory(atom, existing) -> module.
        If None, kind='nn' atoms will raise an error.
    atom_factory : callable | None
        Optional factory hook for non-'nn' atoms. If provided, it is called as
        atom_factory(atom, existing) and may return a leaf module to use.
        Returning None falls back to the standard analytic leaf construction.
    reuse      : dict[str, torch.nn.Module] | None
        Map from tag to existing module instances to reuse.
    return_atom_map : bool
        If True, return (model, atom_to_leaf) where atom_to_leaf maps
        id(atom) -> leaf module. This eliminates fragile DFS-order assumptions.
        Default False.

    Returns
    -------
    model      : ASTCompositeAdaptor
        The compiled composite model.
    atom_to_leaf : dict[int, torch.nn.Module] (if return_atom_map=True)
        Mapping from atom object id to its corresponding leaf module.

    Examples
    --------
    >>> # Old way (fragile, relies on DFS order):
    >>> model = build_composite_from_ast(root, ...)
    >>> atoms = _collect_all_atoms(root)
    >>> atom_to_leaf = {id(a): leaf for a, leaf in zip(atoms, model.leaf)}

    >>> # New way (robust, tag-based):
    >>> model, atom_to_leaf = build_composite_from_ast(root, ..., return_atom_map=True)
    """
    # Lazy import to avoid circular dependency
    try:
        from adaptors.ast_composite import ASTCompositeAdaptor
    except ImportError:
        import os
        import sys

        sr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if sr_dir not in sys.path:
            sys.path.insert(0, sr_dir)
        from adaptors.ast_composite import ASTCompositeAdaptor

    # Build leaves in left-to-right depth-first order
    if return_atom_map:
        leaves, atom_to_leaf = _build_leaves_from_ast(
            root,
            dtype=dtype,
            device=device,
            nn_factory=nn_factory,
            atom_factory=atom_factory,
            reuse=reuse,
            return_atom_map=True,
        )
        model = ASTCompositeAdaptor(root, leaves)
        model = model.to(device=device).to(dtype=dtype)
        return model, atom_to_leaf
    else:
        leaves = _build_leaves_from_ast(
            root,
            dtype=dtype,
            device=device,
            nn_factory=nn_factory,
            atom_factory=atom_factory,
            reuse=reuse,
            return_atom_map=False,
        )
        model = ASTCompositeAdaptor(root, leaves)
        return model.to(device=device).to(dtype=dtype)


def ast_from_composite(model) -> tuple[Node, Dict[str, torch.nn.Module]]:
    """
    Extract AST and reuse map from an ASTCompositeAdaptor.

    For ASTCompositeAdaptor, this is trivial since the model already stores
    the AST directly. We just tag the atoms and build the reuse map.

    Parameters
    ----------
    model : ASTCompositeAdaptor
        The model to extract from.

    Returns
    -------
    root : Node
        AST root with tagged atoms.
    reuse : dict[str, torch.nn.Module]
        Map from tag to leaf module for reuse.

    Example
    -------
    >>> model = ASTCompositeAdaptor(ast, leaves)
    >>> root, reuse = ast_from_composite(model)
    >>> # root is a clone of the original AST with tags added
    >>> # reuse maps 'leaf0', 'leaf1', ... to the leaf modules
    """
    # Check if it's an ASTCompositeAdaptor
    if hasattr(model, "ast_root"):
        # Clone the AST and add tags
        root = _tag_atoms_in_ast(model.ast_root, 0)[0]

        # Build reuse map
        reuse = {}
        for k, leaf in enumerate(model.leaf):
            tag = f"leaf{k}"
            reuse[tag] = leaf

        return root, reuse

    raise TypeError(
        f"ast_from_composite() requires an ASTCompositeAdaptor (with .ast_root), "
        f"got {type(model).__name__}."
    )


def _tag_atoms_in_ast(node: Node, start_idx: int) -> tuple[Node, int]:
    """
    Clone AST and add sequential tags to atoms.

    Returns
    -------
    tagged_node : Node
        Cloned AST with tags.
    next_idx : int
        Next available tag index.
    """
    if isinstance(node, AtomNode):
        new_node = AtomNode(
            kind=node.kind,
            var_idxs=node.var_idxs,
            kwargs=dict(node.kwargs),
            tag=f"leaf{start_idx}",
            inputs=(
                None
                if node.inputs is None
                else tuple(clone_ast(inp) for inp in node.inputs)
            ),
            scope=node.scope,
        )
        return new_node, start_idx + 1
    elif isinstance(node, AddNode):
        left, idx = _tag_atoms_in_ast(node.left, start_idx)
        right, idx = _tag_atoms_in_ast(node.right, idx)
        return AddNode(left, right), idx
    elif isinstance(node, MulNode):
        left, idx = _tag_atoms_in_ast(node.left, start_idx)
        right, idx = _tag_atoms_in_ast(node.right, idx)
        return MulNode(left, right), idx
    elif isinstance(node, PowNode):
        base, idx = _tag_atoms_in_ast(node.base, start_idx)
        return PowNode(base=base, exponent=node.exponent), idx
    elif isinstance(node, LogNode):
        arg, idx = _tag_atoms_in_ast(node.arg, start_idx)
        return LogNode(arg=arg), idx
    elif isinstance(node, SinNode):
        arg, idx = _tag_atoms_in_ast(node.arg, start_idx)
        return SinNode(arg=arg), idx
    elif isinstance(node, CosNode):
        arg, idx = _tag_atoms_in_ast(node.arg, start_idx)
        return CosNode(arg=arg), idx
    elif isinstance(node, AsinNode):
        arg, idx = _tag_atoms_in_ast(node.arg, start_idx)
        return AsinNode(arg=arg), idx
    elif isinstance(node, AcosNode):
        arg, idx = _tag_atoms_in_ast(node.arg, start_idx)
        return AcosNode(arg=arg), idx
    elif isinstance(node, AtanNode):
        arg, idx = _tag_atoms_in_ast(node.arg, start_idx)
        return AtanNode(arg=arg), idx
    elif isinstance(node, ExpNode):
        arg, idx = _tag_atoms_in_ast(node.arg, start_idx)
        return ExpNode(arg=arg), idx
    elif isinstance(node, (ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        arg, idx = _tag_atoms_in_ast(node.arg, start_idx)
        return type(node)(arg=arg), idx
    elif isinstance(node, ConstNode):
        return node, start_idx  # ConstNodes don't need tagging
    else:
        raise TypeError(f"Unknown node type: {type(node)}")


# ──────────────────────────────────────────────────────────────
# NN factory helpers for Stage A and Stage B
# ──────────────────────────────────────────────────────────────


def make_reuse_only_nn_factory(device=None, dtype=None, fresh_nn_factory=None):
    """
    Stage‑B NN factory.

    Behaviour
    ---------
    - If `existing` (from the `reuse` dict) is not None, reuse that module
      (optionally moved to `device` / `dtype`).
    - If `existing` is None and `atom.tag is None` and `fresh_nn_factory`
      is provided, delegate to `fresh_nn_factory(atom, None)` to build a
      brand‑new NN leaf (e.g. via a Stage‑A style builder).
    - Otherwise, raise as before (strict reuse‑only semantics).

    Parameters
    ----------
    device : torch.device | None
        Optional device override for reused modules.
    dtype : torch.dtype | None
        Optional dtype override for reused modules.
    fresh_nn_factory : callable | None
        Optional secondary factory with the same
        `(atom: AtomNode, existing: nn.Module | None) -> nn.Module`
        API as `make_stage_a_nn_factory(...)`. Used only when
        `existing is None` and `atom.tag is None`.
    """

    def nn_factory(atom, existing):
        # 1) Reuse existing module when present (standard Stage‑B path).
        if existing is not None:
            target_device = device
            if target_device is None:
                # infer from first parameter if available
                try:
                    target_device = next(existing.parameters()).device
                except StopIteration:
                    target_device = torch.device("cpu")
            if target_device is not None or dtype is not None:
                existing = existing.to(
                    device=target_device if target_device is not None else existing.device,
                    dtype=dtype if dtype is not None else None,
                )
            return existing

        # 2) No reused module. If we were given a fresh‑leaf factory, fall back
        #    to building a new NN leaf. This handles both untagged atoms (rare in
        #    Stage B) and newly-created tagged atoms from splits (e.g., "leaf0_L"
        #    from splitting "leaf0").
        if fresh_nn_factory is not None:
            return fresh_nn_factory(atom, None)

        # 3) Legacy behaviour: strict reuse‑only semantics.
        raise ValueError(
            f"Stage B: No reused module found for atom with tag='{atom.tag}'. "
            f"This usually means the AST doesn't have proper tags, or you need "
            f"to supply a `fresh_nn_factory` to make_reuse_only_nn_factory(). "
            f"Use ast_from_composite() to extract the AST with tags from the Stage A model."
        )

    return nn_factory


def make_stage_a_nn_factory(leaf_builder):
    """
    Create an nn_factory for Stage A that builds new NN leaves.

    This factory creates new SegmentedAdaptor or DualSegmentedAdaptor leaves
    using the provided LeafBuilder.

    Parameters
    ----------
    leaf_builder : LeafBuilder
        Factory for building segmented adaptors.

    Returns
    -------
    nn_factory : callable
        Factory function: nn_factory(atom, existing) -> module

    Notes
    -----
    - If 'existing' is provided, it is reused
    - Otherwise, a new leaf is created from atom.kwargs
    - Required kwargs: 'num_segments', 'dual_layer'

    Example
    -------
    >>> from nestynet_sr.sr_search.model_builders import LeafBuilder
    >>> from nestynet_sr.sr_search.config import ModelHyperparams
    >>> hp = ModelHyperparams()
    >>> builder = LeafBuilder(hp, device, dtype)
    >>> nn_factory = make_stage_a_nn_factory(builder)
    >>> atom = AtomNode('nn', (0, 1), kwargs={'num_segments': 16, 'dual_layer': False})
    >>> leaf = nn_factory(atom, existing=None)
    """

    def nn_factory(atom: AtomNode, existing):
        if existing is not None:
            return existing

        # Extract parameters from atom.kwargs
        num_segments = atom.kwargs.get("num_segments", 16)
        dual_layer = atom.kwargs.get("dual_layer", False)

        # Token length determines n_in (works for both compound and simple atoms).
        token = list(range(atom.n_in))

        # Build new leaf
        leaf, _ = leaf_builder.build_leaf(
            token=token, num_segments=num_segments, dual_layer=dual_layer
        )
        return leaf

    return nn_factory


# ──────────────────────────────────────────────────────────────
# AST construction and manipulation helpers for Stage A
# ──────────────────────────────────────────────────────────────


def build_initial_ast(
    Nxvars: int, num_segments: int = 32, dual_layer: bool = False, tag: str | None = "A0"
) -> AtomNode:
    """
    Build a single NN atom covering all input variables.

    This is the typical starting point for Stage A symbolic regression.

    Parameters
    ----------
    Nxvars : int
        Number of input variables.
    num_segments : int
        Number of segments for the NN.
    dual_layer : bool
        Whether to use dual-layer architecture.
    tag : str | None
        Tag for this atom. Defaults to "A0" for Stage A reuse.

    Returns
    -------
    AtomNode
        NN atom covering variables 0..Nxvars-1.

    Example
    -------
    >>> ast = build_initial_ast(Nxvars=3, num_segments=16, dual_layer=False)
    >>> # Creates NN[x0, x1, x2] with 16 segments
    """
    return AtomNode(
        kind="nn",
        var_idxs=tuple(range(Nxvars)),
        kwargs={"num_segments": num_segments, "dual_layer": dual_layer},
        tag=tag,
    )


def build_product_ast(
    var_groups: List[List[int]], num_segments: int = 32, dual_layer: bool = False
) -> Node:
    """
    Build a product of NN atoms, one per variable group.

    This is used when proposing multiplicative separability.

    Parameters
    ----------
    var_groups : list of list of int
        Each inner list contains variable indices for one NN atom.
    num_segments : int
        Number of segments for each NN.
    dual_layer : bool
        Whether to use dual-layer architecture.

    Returns
    -------
    Node
        Product tree: NN[group0] * NN[group1] * ...
        If only one group, returns a single AtomNode.

    Example
    -------
    >>> ast = build_product_ast([[0], [1, 2]], num_segments=16)
    >>> # Creates NN[x0] * NN[x1, x2]
    """
    if len(var_groups) == 0:
        raise ValueError("var_groups cannot be empty")

    if len(var_groups) == 1:
        return AtomNode(
            kind="nn",
            var_idxs=tuple(var_groups[0]),
            kwargs={"num_segments": num_segments, "dual_layer": dual_layer},
        )

    # Build nested multiplication tree (left-associative)
    atoms = [
        AtomNode(
            kind="nn",
            var_idxs=tuple(group),
            kwargs={"num_segments": num_segments, "dual_layer": dual_layer},
        )
        for group in var_groups
    ]

    result = atoms[0]
    for atom in atoms[1:]:
        result = MulNode(result, atom)

    return result


def build_sum_ast(
    var_groups: List[List[int]], num_segments: int = 32, dual_layer: bool = False
) -> Node:
    """
    Build a sum of NN atoms, one per variable group.

    This is used when proposing additive separability.

    Parameters
    ----------
    var_groups : list of list of int
        Each inner list contains variable indices for one NN atom.
    num_segments : int
        Number of segments for each NN.
    dual_layer : bool
        Whether to use dual-layer architecture.

    Returns
    -------
    Node
        Sum tree: NN[group0] + NN[group1] + ...
        If only one group, returns a single AtomNode.

    Example
    -------
    >>> ast = build_sum_ast([[0], [1, 2]], num_segments=16)
    >>> # Creates NN[x0] + NN[x1, x2]
    """
    if len(var_groups) == 0:
        raise ValueError("var_groups cannot be empty")

    if len(var_groups) == 1:
        return AtomNode(
            kind="nn",
            var_idxs=tuple(var_groups[0]),
            kwargs={"num_segments": num_segments, "dual_layer": dual_layer},
        )

    # Build nested addition tree (left-associative)
    atoms = [
        AtomNode(
            kind="nn",
            var_idxs=tuple(group),
            kwargs={"num_segments": num_segments, "dual_layer": dual_layer},
        )
        for group in var_groups
    ]

    result = atoms[0]
    for atom in atoms[1:]:
        result = AddNode(result, atom)

    return result


def clone_ast(node: Node) -> Node:
    """
    Deep copy an AST node tree.

    This is useful when you need to modify an AST without affecting the original.

    Parameters
    ----------
    node : Node
        AST node to clone.

    Returns
    -------
    Node
        Deep copy of the input tree.
    """
    if isinstance(node, AtomNode):
        new_kwargs = dict(node.kwargs)

        # Clone any embedded structural ASTs that live in kwargs.
        for expr_key in ("input_ast", "arg_expr", "z_expr"):
            if expr_key in new_kwargs and new_kwargs[expr_key] is not None:
                new_kwargs[expr_key] = clone_ast(new_kwargs[expr_key])

        # Clone inputs field (the unified input representation)
        new_inputs = None
        if node.inputs is not None:
            new_inputs = tuple(clone_ast(inp) for inp in node.inputs)

        cloned = AtomNode(
            kind=node.kind,
            var_idxs=node.var_idxs,
            kwargs=new_kwargs,
            tag=node.tag,
            inputs=new_inputs,
        )
        cloned.scope = node.scope
        return cloned
    elif isinstance(node, AddNode):
        return AddNode(left=clone_ast(node.left), right=clone_ast(node.right))
    elif isinstance(node, MulNode):
        return MulNode(left=clone_ast(node.left), right=clone_ast(node.right))
    elif isinstance(node, PowNode):
        return PowNode(
            base=clone_ast(node.base),
            exponent=float(node.exponent),
        )
    elif isinstance(node, LogNode):
        return LogNode(
            arg=clone_ast(node.arg),
        )
    elif isinstance(node, ExpNode):
        return ExpNode(
            arg=clone_ast(node.arg),
        )
    elif isinstance(node, SinNode):
        return SinNode(
            arg=clone_ast(node.arg),
        )
    elif isinstance(node, CosNode):
        return CosNode(arg=clone_ast(node.arg))
    elif isinstance(node, AsinNode):
        return AsinNode(arg=clone_ast(node.arg))
    elif isinstance(node, AcosNode):
        return AcosNode(arg=clone_ast(node.arg))
    elif isinstance(node, AtanNode):
        return AtanNode(arg=clone_ast(node.arg))
    elif isinstance(node, ConjNode):
        return ConjNode(arg=clone_ast(node.arg))
    elif isinstance(node, RealNode):
        return RealNode(arg=clone_ast(node.arg))
    elif isinstance(node, ImagNode):
        return ImagNode(arg=clone_ast(node.arg))
    elif isinstance(node, AbsNode):
        return AbsNode(arg=clone_ast(node.arg))
    elif isinstance(node, ArgNode):
        return ArgNode(arg=clone_ast(node.arg))
    elif isinstance(node, ConstNode):
        return ConstNode(node.value)
    else:
        raise TypeError(f"Unknown node type: {type(node)}")


def update_ast_nn_kwargs(
    node: Node,
    num_segments: int | None = None,
    dual_layer: bool | None = None,
    skip_tags: set | None = None,
) -> Node:
    """
    Recursively update num_segments and/or dual_layer in all NN atoms.

    This creates a new AST with updated kwargs, leaving the original unchanged.

    Parameters
    ----------
    node : Node
        Root of AST to update.
    num_segments : int | None
        If provided, update num_segments in all NN atoms.
    dual_layer : bool | None
        If provided, update dual_layer in all NN atoms.
    skip_tags : set | None
        If provided, atoms with tags in this set will NOT have their kwargs
        updated. This is useful when reusing leaves that should preserve
        their original num_segments.

    Returns
    -------
    Node
        New AST with updated kwargs.
    """
    if isinstance(node, AtomNode):
        if node.kind.lower() == "nn":
            # Skip updating if this atom's tag is in skip_tags
            if skip_tags is not None and node.tag in skip_tags:
                return node  # Preserve original kwargs
            new_kwargs = dict(node.kwargs)
            if num_segments is not None:
                new_kwargs["num_segments"] = num_segments
            if dual_layer is not None:
                new_kwargs["dual_layer"] = dual_layer
            return AtomNode(kind=node.kind, var_idxs=node.var_idxs, kwargs=new_kwargs, tag=node.tag, inputs=node.inputs)
        else:
            return node  # Non-NN atoms unchanged
    elif isinstance(node, AddNode):
        return AddNode(
            left=update_ast_nn_kwargs(node.left, num_segments, dual_layer, skip_tags),
            right=update_ast_nn_kwargs(node.right, num_segments, dual_layer, skip_tags),
        )
    elif isinstance(node, MulNode):
        return MulNode(
            left=update_ast_nn_kwargs(node.left, num_segments, dual_layer, skip_tags),
            right=update_ast_nn_kwargs(node.right, num_segments, dual_layer, skip_tags),
        )
    elif isinstance(node, PowNode):
        return PowNode(
            base=update_ast_nn_kwargs(node.base, num_segments, dual_layer, skip_tags),
            exponent=node.exponent,
        )
    elif isinstance(node, LogNode):
        return LogNode(
            arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags),
        )
    elif isinstance(node, ExpNode):
        return ExpNode(
            arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags),
        )
    elif isinstance(node, SinNode):
        return SinNode(
            arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags),
        )
    elif isinstance(node, CosNode):
        return CosNode(arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags))
    elif isinstance(node, AsinNode):
        return AsinNode(arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags))
    elif isinstance(node, AcosNode):
        return AcosNode(arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags))
    elif isinstance(node, AtanNode):
        return AtanNode(arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags))
    elif isinstance(node, ConjNode):
        return ConjNode(arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags))
    elif isinstance(node, RealNode):
        return RealNode(arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags))
    elif isinstance(node, ImagNode):
        return ImagNode(arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags))
    elif isinstance(node, AbsNode):
        return AbsNode(arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags))
    elif isinstance(node, ArgNode):
        return ArgNode(arg=update_ast_nn_kwargs(node.arg, num_segments, dual_layer, skip_tags))
    elif isinstance(node, ConstNode):
        return node  # ConstNodes have no NN kwargs
    else:
        raise TypeError(f"Unknown node type: {type(node)}")


def ast_to_human_readable(node: Node, x_transform_map: dict = None) -> str:
    """
    Convert AST to human-readable string representation.

    Parameters
    ----------
    node : Node
        AST node to convert.
    x_transform_map : dict, optional
        Mapping from variable index to transformation info, e.g.,
        {2: {"pipeline": [{"kind": "cos"}]}} means x2 -> cos(x2).

    Returns
    -------
    str
        Human-readable expression string.

    Example
    -------
    >>> ast = Mul(AtomNode('nn', (0,)), AtomNode('nn', (1,)))
    >>> print(ast_to_human_readable(ast))
    (NN[x0] * NN[x1])
    """

    def _format_var(idx: int) -> str:
        """Format a single variable, applying x-transform if present."""
        base = f"x{idx}"
        if x_transform_map and idx in x_transform_map:
            info = x_transform_map[idx]
            fn, omega, shift = "", 1.0, 0.0
            for step in info.get("pipeline", []):
                kind = step.get("kind", "")
                if kind in ("cos", "sin", "recip", "log", "sqrt", "square"):
                    fn = kind
                elif kind == "scale":
                    omega = step.get("scale", step.get("omega", 1.0))
                elif kind == "shift":
                    shift = step.get("shift", 0.0)
            if fn:
                arg = base if omega == 1.0 else f"{omega:g}*{base}"
                if shift != 0.0:
                    arg = f"({arg} + {shift:g})"
                return f"{fn}({arg})"
        return base

    if isinstance(node, AtomNode):
        kind = str(node.kind).lower()
        # Raw variable leaf
        if kind in ("var", "x", "input"):
            if len(node.var_idxs) != 1:
                return f"var({', '.join(_format_var(int(i)) for i in node.var_idxs)})"
            return _format_var(int(node.var_idxs[0]))

        # Named scalar constants retain their symbolic identity.  Their fitted
        # or declared numeric values are carried separately as coefficient
        # metadata for numerical evaluation.
        if kind in (
            "free_const",
            "freeconst",
            "free_constant",
            "fixed_const",
            "fixedconst",
            "fixed_constant",
        ):
            nm = None
            try:
                nm = node.kwargs.get("name", None)
            except Exception:
                nm = None
            if nm is None:
                nm = node.tag
            nm = str(nm) if nm is not None else "c"
            # Import lazily to avoid the bridges <-> metadata import cycle.
            from .coefficient_metadata import coefficient_symbol_for_name

            return coefficient_symbol_for_name(nm)

        if kind == "nn":
            # Unified display: always format via input expressions.
            # For simple atoms inputs=(Var(0), Var(1)) → "NN[x0, x1]"
            # For compound inputs=(Mul(Var(0),Var(1)), Var(2)) → "NN[x0*x1, x2]"
            inputs = get_input_exprs(node)
            parts = [ast_to_human_readable(inp, x_transform_map) for inp in inputs]
            problem_label = atom_problem_label(node)
            if problem_label is not None:
                return f"{problem_label}[{', '.join(parts)}]"
            return f"NN[{', '.join(parts)}]"
        # For analytical atoms (ratpoly, poly, etc.), also use input expressions
        # so compound/wrapped inputs (e.g. cos(x2)) render correctly.
        inputs = get_input_exprs(node)
        if inputs and has_nontrivial_input(node):
            parts = [ast_to_human_readable(inp, x_transform_map) for inp in inputs]
            return f"{node.kind}({', '.join(parts)})"
        vars_str = ", ".join(_format_var(i) for i in node.var_idxs)
        return f"{node.kind}({vars_str})"
    elif isinstance(node, AddNode):
        left_str = ast_to_human_readable(node.left, x_transform_map)
        right_str = ast_to_human_readable(node.right, x_transform_map)
        return f"({left_str} + {right_str})"
    elif isinstance(node, MulNode):
        left_str = ast_to_human_readable(node.left, x_transform_map)
        right_str = ast_to_human_readable(node.right, x_transform_map)
        return f"({left_str} * {right_str})"
    elif isinstance(node, PowNode):
        # Canonical display for common inverse forms:
        #   (u^-1)^-1            -> u
        #   (a * b^-1)^-1        -> (b / a)
        if abs(node.exponent + 1.0) < 1e-8:
            base = node.base
            if isinstance(base, PowNode) and abs(base.exponent + 1.0) < 1e-8:
                return ast_to_human_readable(base.base, x_transform_map)
            if isinstance(base, MulNode):
                left, right = base.left, base.right
                if isinstance(right, PowNode) and abs(right.exponent + 1.0) < 1e-8:
                    num_str = ast_to_human_readable(right.base, x_transform_map)
                    den_str = ast_to_human_readable(left, x_transform_map)
                    return f"({num_str} / {den_str})"
                if isinstance(left, PowNode) and abs(left.exponent + 1.0) < 1e-8:
                    num_str = ast_to_human_readable(left.base, x_transform_map)
                    den_str = ast_to_human_readable(right, x_transform_map)
                    return f"({num_str} / {den_str})"

        base_str = ast_to_human_readable(node.base, x_transform_map)
        if abs(node.exponent - 0.5) < 1e-8:
            return f"sqrt({base_str})"
        if abs(node.exponent + 0.5) < 1e-8:
            return f"1/sqrt({base_str})"
        return f"({base_str})**{node.exponent:g}"
    elif isinstance(node, LogNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"log({arg_str})"
    elif isinstance(node, ExpNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"exp({arg_str})"
    elif isinstance(node, SinNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"sin({arg_str})"
    elif isinstance(node, CosNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"cos({arg_str})"
    elif isinstance(node, AsinNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"arcsin({arg_str})"
    elif isinstance(node, AcosNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"arccos({arg_str})"
    elif isinstance(node, AtanNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"arctan({arg_str})"
    elif isinstance(node, ConjNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"conj({arg_str})"
    elif isinstance(node, RealNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"real({arg_str})"
    elif isinstance(node, ImagNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"imag({arg_str})"
    elif isinstance(node, AbsNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"abs({arg_str})"
    elif isinstance(node, ArgNode):
        arg_str = ast_to_human_readable(node.arg, x_transform_map)
        return f"arg({arg_str})"
    elif isinstance(node, ConstNode):
        return format_const_value(node.value)
    else:
        raise TypeError(f"Unknown node type: {type(node)}")


def collect_nn_atoms(root: Node) -> List[AtomNode]:
    """
    Collect all NN atoms from an AST in depth-first order.

    Parameters
    ----------
    root : Node
        AST root to traverse.

    Returns
    -------
    list of AtomNode
        All NN atoms found in the tree.

    Example
    -------
    >>> ast = Mul(AtomNode('nn', (0,)), AtomNode('nn', (1, 2)))
    >>> atoms = collect_nn_atoms(ast)
    >>> # Returns [AtomNode('nn', (0,)), AtomNode('nn', (1, 2))]
    """
    atoms = []

    def traverse(node):
        if isinstance(node, AtomNode):
            if node.kind.lower() == "nn":
                atoms.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            traverse(node.left)
            traverse(node.right)
        elif isinstance(node, PowNode):
            traverse(node.base)
        elif isinstance(
            node,
            (
                LogNode,
                ExpNode,
                SinNode,
                CosNode,
                AsinNode,
                AcosNode,
                AtanNode,
                ConjNode,
                RealNode,
                ImagNode,
                AbsNode,
                ArgNode,
            ),
        ):
            traverse(node.arg)

    traverse(root)
    return atoms


def collect_all_atoms(root: Node) -> List[AtomNode]:
    """
    Collect ALL atoms from an AST in depth-first order.

    This includes both NN atoms and non-NN atoms (FreeConst, etc.)
    in the same order that _build_leaves_from_ast uses for leaf indexing.

    Parameters
    ----------
    root : Node
        AST root to traverse.

    Returns
    -------
    list of AtomNode
        All atoms found in the tree, in DFS order.

    Example
    -------
    >>> ast = Mul(AtomNode('nn', (0,)), AtomNode('freeconst', ()))
    >>> atoms = collect_all_atoms(ast)
    >>> # Returns [AtomNode('nn', (0,)), AtomNode('freeconst', ())]
    """
    atoms = []

    def traverse(node):
        if isinstance(node, AtomNode):
            atoms.append(node)
        elif isinstance(node, (AddNode, MulNode)):
            traverse(node.left)
            traverse(node.right)
        elif isinstance(node, PowNode):
            traverse(node.base)
        elif isinstance(
            node,
            (
                LogNode,
                ExpNode,
                SinNode,
                CosNode,
                AsinNode,
                AcosNode,
                AtanNode,
                ConjNode,
                RealNode,
                ImagNode,
                AbsNode,
                ArgNode,
            ),
        ):
            traverse(node.arg)

    traverse(root)
    return atoms


def count_atom_params(atom: AtomNode) -> int:
    """Count trainable parameters for a single atom without fitting.

    Instantiates the leaf module on CPU and counts parameters.
    Returns a large sentinel (999) for NN atoms (not leaf-backed).
    """
    kind = atom.kind.lower()
    if kind in ("nn", "nn_atom"):
        return 999  # NN atoms shouldn't appear in exhaustive candidates
    try:
        module = _build_leaf_module(atom, dtype=torch.float64, device=torch.device("cpu"))
        return sum(p.numel() for p in module.parameters())
    except Exception:
        return 999  # unknown / broken → sort last


def effective_arity(atom: AtomNode) -> int:
    """
    Return the effective input dimensionality of an atom.

    For compound atoms (those with input_expr), returns
    1 + len(extra_var_idxs) where extra_var_idxs are additional raw
    variables passed to the leaf alongside the compound scalar z.
    For regular atoms, returns len(var_idxs).

    Parameters
    ----------
    atom : AtomNode
        The atom to check.

    Returns
    -------
    int
        1 + n_extra for compound atoms, len(var_idxs) otherwise.

    Notes
    -----
    This is useful for Stage B rules that need to treat compound atoms
    as univariate for pattern matching (e.g., Planck, trig composition).
    """
    return atom.n_in


def is_pure_1d_full_compound_ast(root: Optional[Node], Nxvars: int) -> bool:
    """Return True iff AST is exactly y ~= f(z(x0..xN)) with arity-1 NN leaf.

    Conditions:
    - exactly one NN atom
    - that NN has effective arity 1 and a nontrivial input expression
    - raw variable coverage includes all original inputs
    - no non-constant atoms outside that NN (prevents prefactor/extra terms)
    """
    if root is None:
        return False

    nn_atoms = collect_nn_atoms(root)
    if len(nn_atoms) != 1:
        return False

    at = nn_atoms[0]
    if int(effective_arity(at)) != 1:
        return False
    if not has_nontrivial_input(at):
        return False

    const_kinds = {
        "free_const",
        "freeconst",
        "free_constant",
        "fixed_const",
        "fixedconst",
        "fixed_constant",
        "scale",
    }
    for a in collect_all_atoms(root):
        k = str(getattr(a, "kind", "")).lower()
        if k == "nn":
            continue
        if k in const_kinds:
            continue
        return False

    raw = getattr(at, "raw_var_idxs", at.var_idxs)
    return set(int(v) for v in raw) >= set(range(int(Nxvars)))


def eval_input_expr(expr: Node, x: "torch.Tensor") -> "torch.Tensor":
    """
    Evaluate a pure structural AST (input_expr) to get compound variable values.

    Used for compound variable expressions like z = x0 * x1 or z = x0 / x1.
    The AST should only contain Var, Mul, Add, Pow nodes - no trainable leaves.

    Parameters
    ----------
    expr : Node
        AST representing the compound variable expression.
    x : torch.Tensor, shape (B, Nx)
        Input data.

    Returns
    -------
    torch.Tensor, shape (B, 1)
        Compound variable values.
    """
    import torch

    def rec(node: Node):
        if isinstance(node, AtomNode):
            kind = str(getattr(node, "kind", "")).lower()
            if kind in ("var", "x", "input"):
                if len(node.var_idxs) != 1:
                    raise ValueError("Var node in input_expr must have exactly 1 var_idx")
                idx = node.var_idxs[0]
                return x[:, idx : idx + 1]  # [B, 1]
            else:
                raise ValueError(f"Unsupported atom kind '{kind}' in input_expr")
        elif isinstance(node, AddNode):
            return rec(node.left) + rec(node.right)
        elif isinstance(node, MulNode):
            return rec(node.left) * rec(node.right)
        elif isinstance(node, PowNode):
            base_val = rec(node.base)
            exp_val = node.exponent  # PowNode.exponent is a float
            return base_val**exp_val
        elif isinstance(node, LogNode):
            return torch.log(rec(node.arg))
        elif isinstance(node, ExpNode):
            return torch.exp(rec(node.arg))
        elif isinstance(node, SinNode):
            return torch.sin(rec(node.arg))
        elif isinstance(node, CosNode):
            return torch.cos(rec(node.arg))
        elif isinstance(node, AsinNode):
            return torch.asin(torch.clamp(rec(node.arg), -1.0 + 1.0e-12, 1.0 - 1.0e-12))
        elif isinstance(node, AcosNode):
            return torch.acos(torch.clamp(rec(node.arg), -1.0 + 1.0e-12, 1.0 - 1.0e-12))
        elif isinstance(node, AtanNode):
            return torch.atan(rec(node.arg))
        elif isinstance(node, ConjNode):
            return torch.conj(rec(node.arg))
        elif isinstance(node, RealNode):
            return torch.real(rec(node.arg))
        elif isinstance(node, ImagNode):
            return torch.imag(rec(node.arg))
        elif isinstance(node, AbsNode):
            return torch.abs(rec(node.arg))
        elif isinstance(node, ArgNode):
            return torch.angle(rec(node.arg))
        elif isinstance(node, ConstNode):
            return const_full_like(x, (x.shape[0], 1), node.value)
        else:
            raise ValueError(f"Unsupported node type in input_expr: {type(node)}")

    return rec(expr)


def sync_ast_num_segments_from_state_dict(ast: Node, state_dict: dict) -> dict:
    """
    Infer per-leaf NN shape kwargs from a state_dict and update AST atoms in-place.

    This fixes the common issue where the AST stores a num_segments value that
    doesn't match the actual trained model (e.g., after separability splits where
    the model is rebuilt with different num_segments).  It also syncs the
    dual_layer flag: checkpoint restores must rebuild the same leaf architecture
    before loading weights.

    Parameters
    ----------
    ast : Node
        AST root whose atoms will be updated in-place.
    state_dict : dict
        Model state_dict from torch.load().

    Returns
    -------
    dict
        Mapping from leaf index to inferred num_segments (for debugging).

    Notes
    -----
    Uses the _ever_active tensor which has shape [num_segments] directly.
    This is more reliable than a_fit (which depends on num_inputs) or
    other tensors.
    """
    # Infer num_segments from _ever_active tensors.
    # Keys look like:
    #   "leaf.0.model._ever_active"                  (single-layer NN)
    #   "leaf.0._stage0.model._ever_active"          (dual-layer NN)
    #   "leaf.0._stage1.model._ever_active"          (dual-layer NN)
    leaf_num_segments = {}
    leaf_dual_layer = {}
    for key, tensor in state_dict.items():
        if not isinstance(key, str):
            continue
        parts = key.split(".")
        if len(parts) < 3 or parts[0] != "leaf":
            continue
        try:
            leaf_idx = int(parts[1])
        except Exception:
            continue
        if len(parts) > 2 and parts[2] in {"_stage0", "_stage1", "stage0", "stage1"}:
            leaf_dual_layer[leaf_idx] = True
        elif leaf_idx not in leaf_dual_layer:
            leaf_dual_layer[leaf_idx] = False
        if (
            (".model._ever_active" in key or ".base_model._ever_active" in key)
            and torch.is_tensor(tensor)
            and tensor.dim() == 1
        ):
            # _ever_active has shape [num_segments]
            leaf_num_segments[leaf_idx] = tensor.shape[0]

    # Update AST atoms in-place
    # Use collect_all_atoms to get correct leaf index mapping.
    # The leaf indices in state_dict correspond to ALL atoms in DFS order,
    # not just NN atoms. Non-NN atoms (FreeConst, etc.) also get leaf indices.
    all_atoms = collect_all_atoms(ast)
    for i, atom in enumerate(all_atoms):
        if atom.kind.lower() == "nn" and i in leaf_num_segments:
            new_kwargs = dict(atom.kwargs)
            new_kwargs["num_segments"] = leaf_num_segments[i]
            if i in leaf_dual_layer:
                new_kwargs["dual_layer"] = bool(leaf_dual_layer[i])
            atom.kwargs = new_kwargs
        elif atom.kind.lower() == "nn" and i in leaf_dual_layer:
            new_kwargs = dict(atom.kwargs)
            new_kwargs["dual_layer"] = bool(leaf_dual_layer[i])
            atom.kwargs = new_kwargs

    return leaf_num_segments


def ast_equals(node1: Node, node2: Node) -> bool:
    """
    Check if two AST nodes are structurally equal.

    Compares node types, atom properties (kind, var_idxs, kwargs, tag),
    and recursively compares children for composite nodes.

    Parameters
    ----------
    node1 : Node
        First AST node.
    node2 : Node
        Second AST node.

    Returns
    -------
    bool
        True if the ASTs are structurally equal, False otherwise.

    Examples
    --------
    >>> ast1 = MulNode(AtomNode('nn', (0,)), AtomNode('nn', (1,)))
    >>> ast2 = MulNode(AtomNode('nn', (0,)), AtomNode('nn', (1,)))
    >>> ast_equals(ast1, ast2)
    True

    >>> ast3 = AddNode(AtomNode('nn', (0,)), AtomNode('nn', (1,)))
    >>> ast_equals(ast1, ast3)
    False
    """
    # Different types -> not equal
    if type(node1) is not type(node2):
        return False

    # AtomNode: compare kind, var_idxs, kwargs, tag, scope, and explicit inputs.
    if isinstance(node1, AtomNode):
        node2 = node2  # type: AtomNode
        if node1.kind != node2.kind:
            return False
        if node1.var_idxs != node2.var_idxs:
            return False
        if node1.tag != node2.tag:
            return False
        if node1.scope != node2.scope:
            return False
        # Compare kwargs (dict comparison)
        if node1.kwargs != node2.kwargs:
            return False
        inputs1 = tuple(node1.inputs or ())
        inputs2 = tuple(node2.inputs or ())
        if len(inputs1) != len(inputs2):
            return False
        if not all(ast_equals(a, b) for a, b in zip(inputs1, inputs2)):
            return False
        return True

    # Binary nodes: AddNode, MulNode
    if isinstance(node1, (AddNode, MulNode)):
        return ast_equals(node1.left, node2.left) and ast_equals(node1.right, node2.right)

    # PowNode: compare base and exponent
    if isinstance(node1, PowNode):
        return ast_equals(node1.base, node2.base) and node1.exponent == node2.exponent

    # Unary nodes
    if isinstance(node1, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        return ast_equals(node1.arg, node2.arg)

    # ConstNode: compare values
    if isinstance(node1, ConstNode):
        if isinstance(node1.value, complex) != isinstance(node2.value, complex):
            return False
        return node1.value == node2.value

    # Unknown node type
    raise TypeError(f"ast_equals does not support node type: {type(node1)}")


def replace_atom_in_ast(root: Node, old_atom: AtomNode, new_subtree: Node) -> Node:
    """
    Replace a specific atom in the AST with a new subtree.

    Uses object identity to find the exact atom to replace.

    Parameters
    ----------
    root : Node
        AST root.
    old_atom : AtomNode
        The specific atom instance to replace.
    new_subtree : Node
        The new subtree to insert.

    Returns
    -------
    Node
        New AST with replacement made.

    Example
    -------
    >>> ast = Mul(AtomNode('nn', (0,)), AtomNode('nn', (1, 2)))
    >>> atoms = collect_nn_atoms(ast)
    >>> # Replace second atom with two atoms multiplied
    >>> new_ast = replace_atom_in_ast(ast, atoms[1],
    ...     Mul(AtomNode('nn', (1,)), AtomNode('nn', (2,))))
    """
    if root is old_atom:
        return new_subtree

    if isinstance(root, AtomNode):
        return root

    if isinstance(root, AddNode):
        return AddNode(
            left=replace_atom_in_ast(root.left, old_atom, new_subtree),
            right=replace_atom_in_ast(root.right, old_atom, new_subtree),
        )

    if isinstance(root, MulNode):
        return MulNode(
            left=replace_atom_in_ast(root.left, old_atom, new_subtree),
            right=replace_atom_in_ast(root.right, old_atom, new_subtree),
        )

    if isinstance(root, PowNode):
        return PowNode(
            base=replace_atom_in_ast(root.base, old_atom, new_subtree),
            exponent=root.exponent,
        )

    if isinstance(root, LogNode):
        return LogNode(
            arg=replace_atom_in_ast(root.arg, old_atom, new_subtree),
        )
    if isinstance(root, ExpNode):
        return ExpNode(
            arg=replace_atom_in_ast(root.arg, old_atom, new_subtree),
        )

    if isinstance(root, SinNode):
        return SinNode(
            arg=replace_atom_in_ast(root.arg, old_atom, new_subtree),
        )

    if isinstance(root, CosNode):
        return CosNode(arg=replace_atom_in_ast(root.arg, old_atom, new_subtree))

    if isinstance(root, AsinNode):
        return AsinNode(arg=replace_atom_in_ast(root.arg, old_atom, new_subtree))

    if isinstance(root, AcosNode):
        return AcosNode(arg=replace_atom_in_ast(root.arg, old_atom, new_subtree))

    if isinstance(root, AtanNode):
        return AtanNode(arg=replace_atom_in_ast(root.arg, old_atom, new_subtree))

    if isinstance(root, (ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        return type(root)(arg=replace_atom_in_ast(root.arg, old_atom, new_subtree))

    if isinstance(root, ConstNode):
        return root  # ConstNodes are not replaceable atoms

    raise TypeError(f"Unknown node type: {type(root)}")


def separability_proposal_to_ast(
    op,
    group1: List[int],
    group2: List[int],
    num_segments: int,
    dual_layer: bool,
    parent_tag: str | None = None,
    parent_atom: "AtomNode | None" = None,
) -> Node:
    """
    Convert a separability proposal from check_separability() to AST.

    Parameters
    ----------
    op : torch.add or torch.multiply
        Operation type from separability detection.
    group1, group2 : list of int
        Variable groups for each factor.
    num_segments : int
        Number of segments for NN atoms.
    dual_layer : bool
        Whether to use dual-layer architecture.
    parent_tag : str | None
        Tag of the parent atom being split. Child atoms will be tagged
        as "{parent_tag}_L" and "{parent_tag}_R" for reuse.
    parent_atom : AtomNode | None
        If provided, compound input expressions from this atom are
        propagated to the child atoms so they inherit the correct
        compound-variable structure.

    Returns
    -------
    Node
        AST representing the proposed separation.

    Example
    -------
    >>> import torch
    >>> ast = separability_proposal_to_ast(torch.multiply, [0], [1, 2], 16, False, parent_tag="A0")
    >>> # Returns Mul(AtomNode('nn', (0,), tag="A0_L"), AtomNode('nn', (1, 2), tag="A0_R"))
    """
    kwargs = {"num_segments": num_segments, "dual_layer": dual_layer}

    # Create deterministic child tags from parent tag
    tag_left = f"{parent_tag}_L" if parent_tag is not None else None
    tag_right = f"{parent_tag}_R" if parent_tag is not None else None

    # Propagate compound inputs from parent atom to children
    left_inputs = _select_inputs_for_var_group(parent_atom, group1) if parent_atom else None
    right_inputs = _select_inputs_for_var_group(parent_atom, group2) if parent_atom else None

    left = AtomNode("nn", tuple(group1), kwargs=kwargs, tag=tag_left, inputs=left_inputs)
    right = AtomNode("nn", tuple(group2), kwargs=kwargs, tag=tag_right, inputs=right_inputs)

    if op is torch.add:
        return AddNode(left, right)
    elif op is torch.multiply:
        return MulNode(left, right)
    else:
        raise ValueError(f"Unknown separability operation: {op}")


def separability_proposal_to_ast_with_offset(
    group1: List[int],
    group2: List[int],
    num_segments: int,
    dual_layer: bool,
    parent_tag: str | None,
    offset_info,
    y_mad: float,
    b_atom: float | None = None,
    *,
    b_kind: str = "free_const",
    b_name: str | None = None,
    b_scope: str | None = None,
    parent_atom: "AtomNode | None" = None,
) -> Node:
    """
    Build Add(FreeConst(b), Mul(NN_L, NN_R)) for offset-multiplicative split.

    For expressions like y = c + f(x_A) * g(x_B), this creates an AST with
    an explicit additive offset constant.

    Parameters
    ----------
    group1, group2 : list of int
        Variable groups for each factor.
    num_segments : int
        Number of segments for NN atoms.
    dual_layer : bool
        Whether to use dual-layer architecture.
    parent_tag : str | None
        Tag of the parent atom being split.
    offset_info : MultiplicativityOffset
        Offset detection result containing b_hat (in normalized scale).
    y_mad : float
        Median absolute deviation of y, used to denormalize b_hat.
    b_atom : float | None
        Pre-computed offset in atom units. If provided, this is used directly
        instead of computing from offset_info.b_hat * y_mad. This is essential
        for nested atoms where the surrounding AST introduces a scaling factor
        between the submodel output and the atom's own output.

    Returns
    -------
    Node
        AST: Add(FreeConst(b), Mul(NN_L, NN_R))
    """
    tag_left = f"{parent_tag}_L" if parent_tag is not None else None
    tag_right = f"{parent_tag}_R" if parent_tag is not None else None
    kwargs = {"num_segments": num_segments, "dual_layer": dual_layer}

    # Propagate compound inputs from parent atom to children
    left_inputs = _select_inputs_for_var_group(parent_atom, group1) if parent_atom else None
    right_inputs = _select_inputs_for_var_group(parent_atom, group2) if parent_atom else None

    left = AtomNode("nn", tuple(group1), kwargs=kwargs, tag=tag_left, inputs=left_inputs)
    right = AtomNode("nn", tuple(group2), kwargs=kwargs, tag=tag_right, inputs=right_inputs)
    product = MulNode(left, right)

    # Use b_atom if provided, otherwise fall back to old behavior
    if b_atom is not None:
        b_init = b_atom
    else:
        # Legacy fallback (may be incorrect for nested atoms)
        b_init = offset_info.b_hat * y_mad
    b_tag = f"{parent_tag}_b" if parent_tag is not None else "offset"

    # Choose constant leaf kind for the offset.
    # - b_kind="scale": a dimensionless trainable scalar (only valid if output is dimless).
    # - b_kind="free_const": a trainable unitful constant; b_name may be provided to select
    #   a user-declared free constant name.
    kind = str(b_kind or "free_const").lower()
    if kind in ("scale", "mul_scale"):
        b_node = Scale(name=b_tag, tag=b_tag, init=b_init)
    else:
        nm = str(b_name) if b_name is not None else b_tag
        b_node = FreeConst(nm, tag=str(nm), init=b_init,
                           scope=str(b_scope) if b_scope is not None else "experiment")

    return AddNode(b_node, product)
