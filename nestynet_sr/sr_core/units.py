# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Lightweight dimensional-analysis utilities.

This module is intentionally *static* and cheap: it operates only on the AST
structure (Add/Mul/Pow/Exp/Log/Sin/Cos) and treats AtomNodes as (by default)
having unknown output dimensions.

The immediate use-case is a **precheck hook** for Stage B candidate selection:
we can reject candidates that are *dimensionally impossible* (e.g. trying to
explain a dimensionful target with a pure exp/log/trig tree that is forced to
be dimensionless).

It is also meant as a foundation for a more PhySO-like “units straightjacket”,
that progressively pins down which subtrees must be dimensionless and/or
which constants must carry which units.

Key design choices
------------------
* Dimensions are exponent vectors over a chosen basis (default: SI base dims).
* Exponents are exact rationals (`fractions.Fraction`) to support sqrt, etc.
* Each AtomNode output gets its own **unknown dimension vector** (unless
  overridden by kind or tag). Unknowns are solved via linear consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .bridges import (
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
    compound_input_expr,
    get_input_exprs,
    has_nontrivial_input,
)

# Optional: only required if explicit division nodes exist in bridges.py
try:
    from .bridges import DivNode  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    DivNode = None  # type: ignore


# ──────────────────────────────────────────────────────────────
# Core types
# ──────────────────────────────────────────────────────────────

Dim = Tuple[Fraction, ...]


def _as_fraction(x: Any, *, max_den: int = 128) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        s = x.strip()
        if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
            s = s[1:-1].strip()
        try:
            return Fraction(s)
        except Exception:
            return Fraction.from_float(float(s)).limit_denominator(max_den)
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, float):
        # Keep common SR exponents tidy (1/2, 3/2, 2, ...)
        return Fraction.from_float(float(x)).limit_denominator(max_den)
    raise TypeError(f"Unsupported exponent type: {type(x)}")


@dataclass(frozen=True)
class UnitSystem:
    """A basis for dimensional exponent vectors.

    Default is the 7 SI base dimensions.
    """

    base: Tuple[str, ...] = ("L", "M", "T", "I", "Θ", "N", "J")

    def dimless(self) -> Dim:
        return tuple(Fraction(0) for _ in self.base)

    def dim(self, exps: Mapping[str, Any] | Sequence[Any]) -> Dim:
        """Construct a Dim from either a mapping {base: exponent} or a sequence.

        Examples
        --------
        >>> us = UnitSystem(("L","T"))
        >>> us.dim({"L": 1, "T": -2})   # L T^-2
        >>> us.dim([1, -2])
        """
        if isinstance(exps, Mapping):
            out = [Fraction(0) for _ in self.base]
            for k, v in exps.items():
                if k not in self.base:
                    raise ValueError(f"Unknown base dimension {k!r}; basis={self.base}")
                out[self.base.index(k)] = _as_fraction(v)
            return tuple(out)
        if isinstance(exps, Sequence):
            if len(exps) != len(self.base):
                raise ValueError(f"Expected {len(self.base)} exponents, got {len(exps)}")
            return tuple(_as_fraction(v) for v in exps)
        raise TypeError("dim() expects a mapping or a sequence")

    def format_dim(self, d: Dim) -> str:
        """Human-friendly representation of a Dim exponent vector."""
        parts = []
        for base, e in zip(self.base, d):
            if e == 0:
                continue
            # Pretty-print Fractions
            if isinstance(e, Fraction):
                if e.denominator == 1:
                    exp_s = str(e.numerator)
                else:
                    exp_s = f"{e.numerator}/{e.denominator}"
            else:
                exp_s = str(e)
            if exp_s == "1":
                parts.append(f"{base}")
            else:
                parts.append(f"{base}^{exp_s}")
        return "1" if not parts else "*".join(parts)


def is_dimless(d: Dim) -> bool:
    return all(e == 0 for e in d)


class UnitError(ValueError):
    pass


def normalize_free_const_scope(scope: Any, *, default: str = "experiment") -> str:
    """Normalize free-constant scope to internal vocabulary.

    Internal scope names are:
      - ``"experiment"``: dataset-local parameter
      - ``"class"``: shared across datasets

    Accepted legacy aliases:
      - ``"local"``  -> ``"experiment"``
      - ``"global"`` -> ``"class"``
    """

    def _norm_default(v: Any) -> str:
        s = str(v).strip().lower()
        if s in ("experiment", "local"):
            return "experiment"
        if s in ("class", "global"):
            return "class"
        raise ValueError(f"Invalid default free-constant scope {v!r}; expected experiment/class or local/global")

    default_norm = _norm_default(default)

    if scope is None:
        return default_norm

    s = str(scope).strip().lower()
    if s == "":
        return default_norm
    if s in ("experiment", "local"):
        return "experiment"
    if s in ("class", "global"):
        return "class"
    raise ValueError(
        f"Invalid free-constant scope {scope!r}; expected experiment/class (or legacy local/global)"
    )


# ──────────────────────────────────────────────────────────────
# DimSubspace — feasible dimension region for a node
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DimSubspace:
    """Feasible dimension subspace for a single AST node.

    Represents the affine subspace  ``offset + span(basis)``  of possible
    Dim vectors that the node can take, given global constraints.

    * ``offset`` is the particular solution.
    * ``basis`` is a tuple of direction vectors spanning the free part.

    Infeasibility is conveyed by returning ``None`` from
    :func:`compute_node_domains`, not by an empty DimSubspace.
    """

    offset: Dim
    basis: Tuple[Dim, ...] = ()

    # -- convenience predicates --

    def n_base(self) -> int:
        """Number of SI base dimensions."""
        return len(self.offset)

    def is_unconstrained(self) -> bool:
        """True when every Dim is feasible (rank == n_base)."""
        return len(self.basis) == self.n_base()

    def is_pinned(self) -> bool:
        """True when exactly one Dim is feasible (rank == 0)."""
        return len(self.basis) == 0

    def rank(self) -> int:
        """Dimension of the feasible subspace (0 = pinned)."""
        return len(self.basis)


def parse_dim_string(s: str, us: UnitSystem) -> Dim:
    """Parse a compact dimension string into a Dim.

    Supported syntax (examples):
      - "1"                 -> dimensionless
      - "L"                 -> length
      - "L*T^-2"            -> acceleration
      - "L/(T^2)"           -> acceleration
      - "M*L^2/T"           -> action-like

    Notes
    -----
    - Exponents may be integers, fractions (e.g. 1/2), or decimals.
    - Division flips the sign of exponents for the subsequent factor.
    - Parentheses are ignored around exponents.
    """
    import re

    s = str(s).strip()
    if s == "" or s == "1":
        return us.dimless()

    # Normalize whitespace to multiplication.
    s = re.sub(r"\s+", "*", s)
    # Split on * and / while retaining separators.
    toks = re.split(r"(\*|/)", s)

    exps: Dict[str, Fraction] = {b: Fraction(0) for b in us.base}
    sign = 1  # +1 for multiplication, -1 for division

    for tok in toks:
        tok = tok.strip()
        if tok == "":
            continue
        if tok == "*":
            sign = 1
            continue
        if tok == "/":
            sign = -1
            continue

        # Allow parentheses around a single factor, e.g. "(T^2)".
        if len(tok) >= 2 and tok[0] == "(" and tok[-1] == ")":
            tok = tok[1:-1].strip()

        base = tok
        exp = Fraction(1)
        if "^" in tok:
            base, exp_s = tok.split("^", 1)
            base = base.strip()
            exp_s = exp_s.strip().strip("()")
            exp = _as_fraction(exp_s)

        if base not in us.base:
            raise UnitError(f"Unknown base dimension {base!r} in dim string {s!r}; basis={us.base}")
        exps[base] += Fraction(sign) * exp

    return us.dim(exps)


def add_dim(a: Dim, b: Dim) -> Dim:
    return tuple(x + y for x, y in zip(a, b))


def sub_dim(a: Dim, b: Dim) -> Dim:
    return tuple(x - y for x, y in zip(a, b))


def scale_dim(a: Dim, s: Fraction) -> Dim:
    return tuple(s * x for x in a)


# ──────────────────────────────────────────────────────────────
# Y-transform dimensional effects
# ──────────────────────────────────────────────────────────────


def y_transform_dim(y_dim: Dim, name: str, us: UnitSystem) -> Dim:
    """Return the dimension of φ(y) given dim(y) and the transform name.

    This is used for filtering which y-transforms are physically admissible.
    """
    n = str(name).lower()
    if n in ("identity", "none"):
        return y_dim
    if n in ("reciprocal", "inv"):
        return scale_dim(y_dim, Fraction(-1))
    if n == "sqrt":
        return scale_dim(y_dim, Fraction(1, 2))
    if n == "square":
        return scale_dim(y_dim, Fraction(2))

    # Dimensionless-argument transforms
    if n in (
        "sin", "arcsin", "cos", "arccos", "tan", "arctan",
        "exp", "log",
        "logneg",   # log(-y)
        "expneg",   # exp(-y)
        "sqrt1p",   # y^2 - 1  (requires y dimless because of "- 1")
    ):
        if not is_dimless(y_dim):
            raise UnitError(f"y-transform '{name}' requires dimensionless y")
        return us.dimless()

    raise UnitError(f"Unknown y-transform name: {name!r}")


# ──────────────────────────────────────────────────────────────
# Linear dimension expressions and constraints
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DimExpr:
    """A linear expression over unknown dimension vectors.

    dim(node) = const + Σ_i coeffs[i] * U_i
    where each U_i is an unknown Dim vector.
    """

    us: UnitSystem
    const: Dim
    coeffs: Mapping[int, Fraction] = field(default_factory=dict)

    @staticmethod
    def dimless(us: UnitSystem) -> "DimExpr":
        return DimExpr(us=us, const=us.dimless(), coeffs={})

    @staticmethod
    def var(us: UnitSystem, var_id: int) -> "DimExpr":
        return DimExpr(us=us, const=us.dimless(), coeffs={var_id: Fraction(1)})

    def add(self, other: "DimExpr") -> "DimExpr":
        if self.us != other.us:
            raise ValueError("UnitSystem mismatch")
        coeffs: Dict[int, Fraction] = dict(self.coeffs)
        for k, v in other.coeffs.items():
            coeffs[k] = coeffs.get(k, Fraction(0)) + v
            if coeffs[k] == 0:
                coeffs.pop(k, None)
        return DimExpr(self.us, add_dim(self.const, other.const), coeffs)

    def sub(self, other: "DimExpr") -> "DimExpr":
        if self.us != other.us:
            raise ValueError("UnitSystem mismatch")
        coeffs: Dict[int, Fraction] = dict(self.coeffs)
        for k, v in other.coeffs.items():
            coeffs[k] = coeffs.get(k, Fraction(0)) - v
            if coeffs[k] == 0:
                coeffs.pop(k, None)
        return DimExpr(self.us, sub_dim(self.const, other.const), coeffs)

    def scale(self, s: Fraction) -> "DimExpr":
        coeffs = {k: s * v for k, v in self.coeffs.items() if v != 0}
        return DimExpr(self.us, scale_dim(self.const, s), coeffs)


@dataclass(frozen=True)
class UnitsSpec:
    """Container describing the units context for an SR run."""

    unit_system: UnitSystem
    x_dims: Tuple[Dim, ...]
    y_dim: Dim
    # Optional per-surrogate-output dimensions for multi-output / vector systems.
    # When omitted, all outputs are assumed to share ``y_dim``.
    output_dims: Tuple[Dim, ...] | None = None
    y_transform_name: str = "identity"  # φ(y) used for fitting

    # Optional: pin certain AtomNode outputs by kind or by tag.
    # Keys are compared in lower-case.
    atom_kind_fixed: Mapping[str, Dim] = field(default_factory=dict)
    atom_tag_fixed: Mapping[str, Dim] = field(default_factory=dict)

    # Pre-declared trainable free constants (a.k.a. "free parameters")
    # that are allowed to carry physical units. Keys are constant names.
    # These are referenced by AtomNode(kind='free_const', kwargs={'name': ...}).
    free_const_dims: Mapping[str, Dim] = field(default_factory=dict)

    # Scope of each free constant: "experiment" (per-dataset) or "class" (shared).
    # Legacy values "local"/"global" are accepted and normalized.
    # Constants not listed default to "experiment".
    free_const_scope: Mapping[str, str] = field(default_factory=dict)

    # Pre-declared *fixed* physical constants with known values and units.
    # These are referenced by AtomNode(kind='fixed_const', kwargs={'name': ...}).
    fixed_const_dims: Mapping[str, Dim] = field(default_factory=dict)
    fixed_const_values: Mapping[str, float] = field(default_factory=dict)
    fixed_const_mode: str = "strict"  # "strict" | "minimal" | "off"

    # Dimensional policy controlling how AtomNodes are interpreted.
    #
    #   - "free_const_only": Only input variables (var leaves) and pre-declared
    #     free constants may be unitful. Certain analytic leaves are treated as
    #     dimensionless functions and rejected if fed unitful x-variables.
    policy: str = "free_const_only"

    # Under policy="free_const_only", decide how to treat NN atoms:
    #   - "unknown": treat NN as an uninterpreted function with unknown output units
    #                (units inferred from surrounding constraints).
    #   - "dimless": treat NN as a dimensionless function; require dimensionless inputs
    #                and output.
    nn_semantics: str = "unknown"

    # Atom kinds that are interpreted as dimensionless functions under
    # policy="free_const_only".
    # Atoms in this list are only permitted to consume *dimensionless* inputs
    # (and return dimensionless outputs) under the default unit policy. This
    # prevents hidden unitful parameters (e.g. trig frequencies, offsets, etc.)
    # from silently absorbing dimensions.
    dimless_atom_kinds: Tuple[str, ...] = (
        # Affine / rational forms: would need unitful offsets or scales unless x is dimless.
        "scale",
        "inv",
        # Nonlinear analytic functions: require dimensionless arguments.
        "polylog",
        "rpolylog",
        "logshifted",
        "exp_poly",
        "rexp_poly",
        "exp_ratpoly",
        "planck",
        "planck_full",
        "sin",
        "cos",
        "sin_linear",
        "cos_linear",
        "tanh",
        "tanh_linear",
        "expm1",
    )

    # If True, AtomNodes sharing a non-None tag are treated as the *same* unknown.
    share_atom_units_by_tag: bool = True

    @property
    def y_phi_dim(self) -> Dim:
        return y_transform_dim(self.y_dim, self.y_transform_name, self.unit_system)

    def __post_init__(self):
        out_dims = getattr(self, "output_dims", None)
        if out_dims is not None:
            out_dims_t = tuple(tuple(d) for d in out_dims)
            for d in out_dims_t:
                if len(d) != len(self.unit_system.base):
                    raise ValueError(
                        f"output_dims entry {d!r} does not match unit-system rank {len(self.unit_system.base)}"
                    )
            object.__setattr__(self, "output_dims", out_dims_t)

        # Normalize free-constant scope vocabulary once at construction so all
        # downstream code can rely on "experiment"/"class".
        norm_scope: Dict[str, str] = {}
        for name, scope in (self.free_const_scope or {}).items():
            norm_scope[str(name)] = normalize_free_const_scope(scope, default="experiment")
        object.__setattr__(self, "free_const_scope", norm_scope)

        mode = str(getattr(self, "fixed_const_mode", "strict")).strip().lower()
        if mode in ("enabled", "on", "true"):
            mode = "strict"
        elif mode in ("disabled", "none"):
            mode = "off"
        if mode not in ("strict", "minimal", "off"):
            raise ValueError(
                f"Invalid fixed_const_mode {self.fixed_const_mode!r}; expected strict|minimal|off"
            )
        object.__setattr__(self, "fixed_const_mode", mode)


@dataclass(frozen=True)
class UnitsCheckResult:
    ok: bool
    reason: str = ""


def _reduce_to_independent(dims: List[Dim]) -> List[Dim]:
    """Keep only linearly independent Dim vectors (incremental rank check).

    Redundant (linearly dependent) basis vectors reduce inference power in
    ``infer_atom_output_dim`` because the underdetermined coefficients cause
    ``_rref_unique_var_value`` to return None.  Filtering to a maximal
    independent subset avoids this.
    """
    if not dims:
        return []
    n = len(dims[0])  # number of SI base dimensions
    # Maintain RREF of accepted vectors
    rows: List[List[Fraction]] = []
    pivot_cols: List[int] = []
    kept: List[Dim] = []

    for d in dims:
        # Try to reduce d against current RREF
        r = [Fraction(x) for x in d]
        for i, pc in enumerate(pivot_cols):
            if r[pc] != 0:
                factor = r[pc] / rows[i][pc]
                for j in range(n):
                    r[j] -= factor * rows[i][j]
        # Find pivot in reduced row
        piv = None
        for j in range(n):
            if r[j] != 0:
                piv = j
                break
        if piv is None:
            continue  # linearly dependent — skip
        # Normalize and add
        scale = r[piv]
        r = [x / scale for x in r]
        rows.append(r)
        pivot_cols.append(piv)
        kept.append(d)
    return kept


def _dim_matrix_rank(dims: List[Dim]) -> int:
    """Return the rank of the dimension matrix (number of independent dims)."""
    return len(_reduce_to_independent(dims))


def _count_dimensionless_combos(dims: List[Dim]) -> int:
    """Buckingham π: number of independent dimensionless combinations.

    Returns max(n_vars - rank(dim_matrix), 0).
    """
    if not dims:
        return 0
    return max(len(dims) - _dim_matrix_rank(dims), 0)


def check_compound_buckingham(
    atom_var_idxs: List[int],
    extra_var_idxs: List[int],
    z_dim: Dim | None,
    x_dims: Tuple[Dim, ...],
    min_freedom: int = 1,
    y_dim: Dim | None = None,
    extra_preserved_dims: List[Dim] | None = None,
) -> Tuple[bool, str]:
    """Check if a compound proposal preserves enough dimensionless freedom
    and dimensional rank.

    Parameters
    ----------
    atom_var_idxs : list[int]
        All original variable indices of the atom (before compound).
    extra_var_idxs : list[int]
        Variable indices kept as extras alongside z.
    z_dim : Dim or None
        Dimension of compound variable z. If None, skip the check.
    x_dims : tuple[Dim, ...]
        Dimensions of all input variables.
    min_freedom : int
        Minimum required dimensionless combos after compound (default 1).
    y_dim : Dim or None
        Dimension of the model target (after any y-transform).  A rank
        drop is only rejected when y_dim is NOT in the span of the
        post-compound dimensions — i.e. when a monomial prefix with the
        right units can no longer be constructed.  If None, the check
        falls back to the conservative original-rank comparison.
    extra_preserved_dims : list[Dim] or None
        Dimensions of preserved compound inputs (e.g. an existing z_first
        that is kept alongside the new compound z).  Included in the
        post-compound dimensionality calculation.

    Returns
    -------
    (ok, reason) : (bool, str)
        ok=True means the compound preserves enough freedom.
    """
    if z_dim is None:
        return (True, "")

    # Before: dims of original atom inputs
    before_dims = [x_dims[i] for i in atom_var_idxs if i < len(x_dims)]
    freedom_before = _count_dimensionless_combos(before_dims)

    # After: dims = [z_dim] + [x_dims[i] for extras] + preserved compound dims
    after_dims = [z_dim] + [x_dims[i] for i in extra_var_idxs if i < len(x_dims)]
    if extra_preserved_dims:
        after_dims.extend(extra_preserved_dims)
    freedom_after = _count_dimensionless_combos(after_dims)

    if freedom_after < min_freedom and freedom_before >= min_freedom:
        return (
            False,
            f"compound destroys dimensionless freedom: "
            f"{freedom_before} -> {freedom_after} "
            f"(need >= {min_freedom}); forces monomial",
        )

    # Dimensional rank check: when a dimensionless compound (e.g. x1/x3)
    # replaces all carriers of some base dimension, the atom can no longer
    # form quantities along that dimension.  By Buckingham π the true
    # function factors as monomial(remaining_vars) · g(π-groups), but
    # if remaining_vars don't span the original dimension space the
    # monomial prefix is unreachable.
    #
    # However, the rank drop only matters if y_dim is NOT in the span of
    # the post-compound dimensions.  If y_dim ∈ span(after_dims) then a
    # monomial prefix with the right dimensions is still constructible.
    # (The dimensionless case y=[0,...,0] is just the trivial instance
    # where the zero vector belongs to every subspace.)
    rank_before = _dim_matrix_rank(before_dims)
    rank_after = _dim_matrix_rank(after_dims)
    if rank_after < rank_before:
        # Check whether y_dim is still reachable from after_dims.
        # If augmenting after_dims with y_dim doesn't increase the rank,
        # then y_dim is already in the span and the prefix is constructible.
        if y_dim is not None:
            rank_augmented = _dim_matrix_rank(after_dims + [y_dim])
            if rank_augmented == rank_after:
                # y_dim is in the span — monomial prefix is still reachable.
                pass
            else:
                return (
                    False,
                    f"compound reduces dimensional rank: "
                    f"{rank_before} -> {rank_after}; "
                    f"remaining variables cannot span output dimension",
                )
        else:
            # No y_dim info — be conservative.
            return (
                False,
                f"compound reduces dimensional rank: "
                f"{rank_before} -> {rank_after}; "
                f"remaining variables cannot span original dimension space",
            )

    return (True, "")


def _atom_key(atom: AtomNode, *, share_by_tag: bool) -> Tuple[str, Any]:
    # Prefer user-provided tags because reuse maps also key by tag.
    tag = getattr(atom, "tag", None)
    if share_by_tag and tag is not None:
        return ("tag", str(tag))
    return ("id", id(atom))


def _infer_dimexpr_and_constraints(
    root: Node,
    spec: UnitsSpec,
    out_node_exprs: Dict[int, DimExpr] | None = None,
) -> Tuple[DimExpr, List[DimExpr], Dict[Tuple[str, Any], int], Dict[int, List[Dim]]]:
    """Return (expr(root), constraints, atom_key->var_id, span_bases).

    *span_bases* maps var_id → list of basis Dim vectors for atoms under
    ``nn_semantics="span"``.  Empty when no span atoms are present.

    If *out_node_exprs* is not None, it will be populated with
    ``{id(node): DimExpr}`` for every node visited during traversal.
    """
    us = spec.unit_system

    constraints: List[DimExpr] = []
    key_to_var: Dict[Tuple[str, Any], int] = {}
    span_bases: Dict[int, List[Dim]] = {}

    def _atom_input_expr(atom: AtomNode) -> Any | None:
        """Return an AST node representing the *effective* scalar input to this atom, if any.

        This supports compound-variable preconditioning by allowing an AtomNode to carry
        an explicit argument expression (e.g. z=x0*x1/x3) in kwargs, while its var_idxs
        can still list the raw x-indices referenced by that expression.
        """
        # Zero-input atoms (e.g. scale(), free_const with no vars) have no
        # input expression to validate.
        if len(getattr(atom, "var_idxs", ())) == 0:
            return None
        # compound_input_expr always returns a Node (Var(i) for simple atoms).
        # For nontrivial inputs, return the expression; for trivial ones, return None
        # to signal that no special argument expression is needed.
        from nestynet_sr.sr_core.bridges import is_trivial_input
        expr = compound_input_expr(atom)
        if not is_trivial_input(expr):
            return expr
        # Legacy fallback for kwargs-based expressions
        kw = getattr(atom, "kwargs", None)
        if not isinstance(kw, Mapping):
            return None
        for k in ("input_ast", "arg_expr", "z_expr"):
            if k in kw and kw[k] is not None:
                return kw[k]
        return None

    def atom_expr(atom: AtomNode) -> DimExpr:
        kind = str(getattr(atom, "kind", "")).lower()
        tag = getattr(atom, "tag", None)

        # If this atom carries an explicit argument expression (compound-variable
        # preconditioning, explicit input rewrites, etc.), we still want to
        # validate that expression for internal dimensional consistency.  This is
        # especially important for NN atoms under nn_semantics="unknown": the NN
        # itself may absorb units, but the *analytic* input_expr must still obey
        # dimensional rules (e.g. x0+x1 requires commensurate units; sin/log args
        # must be dimensionless).
        input_node = _atom_input_expr(atom)

        # Native DE/PDE feature atoms: treat units as *known* from (x_dims, y_dim).
        # These atoms represent the fitted field u(x) and its derivatives.
        if kind in ("u", "field", "state"):
            return DimExpr(us, spec.y_phi_dim, {})

        if kind in ("du", "d1u", "grad_u"):
            try:
                axis = int(getattr(atom, "kwargs", {}).get("axis", 0))
            except Exception:
                axis = 0
            if axis < 0 or axis >= len(spec.x_dims):
                raise UnitError(f"du atom axis out of range: axis={axis} (Nx={len(spec.x_dims)})")
            return DimExpr(us, sub_dim(spec.y_phi_dim, spec.x_dims[axis]), {})

        if kind in ("d2u", "ddu", "hess_u"):
            try:
                a0 = int(getattr(atom, "kwargs", {}).get("axis0", 0))
                a1 = int(getattr(atom, "kwargs", {}).get("axis1", 0))
            except Exception:
                a0, a1 = 0, 0
            if a0 < 0 or a0 >= len(spec.x_dims) or a1 < 0 or a1 >= len(spec.x_dims):
                raise UnitError(
                    f"d2u atom axis out of range: axis0={a0}, axis1={a1} (Nx={len(spec.x_dims)})"
                )
            return DimExpr(
                us, sub_dim(sub_dim(spec.y_phi_dim, spec.x_dims[a0]), spec.x_dims[a1]), {}
            )
        for k, fixed in spec.atom_kind_fixed.items():
            if str(k).lower() == kind:
                return DimExpr(us, fixed, {})
        if (tag is not None) and (str(tag) in spec.atom_tag_fixed):
            return DimExpr(us, spec.atom_tag_fixed[str(tag)], {})

        # Raw variable identity leaf: always use the provided x_dims, regardless
        # of policy.
        if kind in ("var", "x", "input"):
            if len(getattr(atom, "var_idxs", ())) != 1:
                raise UnitError(
                    f"var leaf expects exactly 1 var_idx; got {getattr(atom, 'var_idxs', None)}"
                )
            idx = int(atom.var_idxs[0])
            if idx < 0 or idx >= len(spec.x_dims):
                raise UnitError(f"var leaf index out of range: x{idx} (Nx={len(spec.x_dims)})")
            return DimExpr(us, spec.x_dims[idx], {})

        # Trainable free constant leaf. We *require* its units to be declared.
        if kind in ("free_const", "freeconst", "free_constant"):
            name = None
            try:
                name = atom.kwargs.get("name", None)
            except Exception:
                name = None
            if name is None and tag is not None:
                name = str(tag)
            if name is None:
                raise UnitError("free_const leaf requires kwargs['name'] or a non-None tag")
            name = str(name)
            if name in spec.free_const_dims:
                return DimExpr(us, spec.free_const_dims[name], {})
            raise UnitError(
                f"free_const {name!r} is not declared in UnitsSpec.free_const_dims"
            )

        # Fixed (non-trainable) physical constant leaf. We require its units to be declared.
        if kind in ("fixed_const", "fixedconst", "fixed_constant"):
            name = None
            try:
                name = atom.kwargs.get("name", None)
            except Exception:
                name = None
            if name is None and tag is not None:
                name = str(tag)
            if name is None:
                raise UnitError("fixed_const leaf requires kwargs['name'] or a non-None tag")
            name = str(name)
            if name in spec.fixed_const_dims:
                return DimExpr(us, spec.fixed_const_dims[name], {})
            raise UnitError(f"fixed_const {name!r} is not declared in UnitsSpec.fixed_const_dims")


        # Only variables and *pre-declared* free constants may be unitful.
        if spec.policy == "free_const_only":
            # Linear combination leaf with dimensionless weights.
            if kind in ("lin", "linear", "linpoly"):
                if input_node is not None:
                    return rec(input_node)

                idxs = tuple(getattr(atom, "var_idxs", ()))
                if len(idxs) == 0:
                    raise UnitError("lin leaf requires at least one input index")
                dims = []
                for j in idxs:
                    idx = int(j)
                    if idx < 0 or idx >= len(spec.x_dims):
                        raise UnitError(
                            f"lin leaf references x{idx} out of range (Nx={len(spec.x_dims)})"
                        )
                    dims.append(spec.x_dims[idx])
                d0 = dims[0]
                for d in dims[1:]:
                    if d != d0:
                        raise UnitError(
                            f"lin leaf requires commensurate inputs; got {spec.unit_system.format_dim(d0)} and {spec.unit_system.format_dim(d)}"
                        )
                return DimExpr(us, d0, {})

            # Polynomials use *dimensionless* coefficients. Therefore their output
            # units are completely determined by the units of their inputs.
            #
            # With unitful inputs, dimensional homogeneity forbids mixing different
            # total degrees inside a single polynomial (unless unit-carrying "free
            # constants" are introduced). We enforce this by using homogeneous
            # polynomial bases in `PolyLeaf` and by requiring commensurate input
            # variables here.
            if kind in ("poly", "polynomial", "rpoly", "rpolynomial", "r_polynomial"):
                deg = int(atom.kwargs.get("degree", atom.kwargs.get("deg", 1)))
                if deg == 0:
                    return DimExpr.dimless(us)

                min_total = atom.kwargs.get("min_total", deg)  # default: homogeneous
                if min_total is None:
                    min_total = deg
                min_total = int(min_total)

                if input_node is not None:
                    # Validate EVERY effective input, not just inputs[0]: a
                    # multi-input compound poly mixes its arguments in shared
                    # monomials, so all of them must be commensurate (the same
                    # rule the raw-index path below enforces).
                    input_exprs = (
                        get_input_exprs(atom)
                        if has_nontrivial_input(atom)
                        else (input_node,)
                    )
                    input_dim_exprs = [rec(e) for e in input_exprs]
                    inp = input_dim_exprs[0]
                    for dj in input_dim_exprs[1:]:
                        if not inp.coeffs and not dj.coeffs:
                            if dj.const != inp.const:
                                raise UnitError(
                                    "poly leaf requires commensurate effective "
                                    "inputs; got "
                                    f"{spec.unit_system.format_dim(inp.const)} and "
                                    f"{spec.unit_system.format_dim(dj.const)}"
                                )
                        else:
                            constraints.append(dj.sub(inp))
                    # Mixed-degree polynomials are only dimensionally consistent if the
                    # (effective) input is dimensionless.
                    if min_total < deg:
                        constraints.append(inp)  # enforce inp == 0
                    return inp.scale(Fraction(deg, 1))

                if len(getattr(atom, "var_idxs", ())) == 0:
                    raise UnitError(
                        "poly leaf with degree>0 must reference at least one input variable"
                    )
                dims = []
                for j in getattr(atom, "var_idxs", ()):  # raw x indices
                    idx = int(j)
                    if idx < 0 or idx >= len(spec.x_dims):
                        raise UnitError(
                            f"poly leaf references x{idx} out of range (Nx={len(spec.x_dims)})"
                        )
                    dims.append(spec.x_dims[idx])
                d0 = dims[0]
                for d in dims[1:]:
                    if d != d0:
                        raise UnitError(
                            f"poly leaf requires commensurate inputs; got {spec.unit_system.format_dim(d0)} and {spec.unit_system.format_dim(d)}"
                        )
                # Check min_total: if inputs have non-trivial units, polynomial must
                # be homogeneous (min_total=degree) to avoid mixing dimensions.
                dimless = us.dimless()
                if d0 != dimless and min_total < deg:
                    raise UnitError(
                        f"poly leaf with unitful inputs requires homogeneous basis "
                        f"(min_total=degree={deg}), but got min_total={min_total}"
                    )
                out_dim = tuple(Fraction(deg, 1) * u for u in d0)
                return DimExpr(us, out_dim, {})

            if kind in ("power",):
                # power leaf represents x^p with a dimensionless exponent parameter.
                # Under the 'free_const_only' policy we treat the exponent as fixed (from
                # exponent_init/exponent) so the output units are derived from the input.

                exp_val = atom.kwargs.get("exponent_init", atom.kwargs.get("exponent", 1.0))
                exp = _as_fraction(float(exp_val))

                if input_node is not None:
                    base = rec(input_node)
                    return base.scale(exp)

                if len(getattr(atom, "var_idxs", ())) != 1:
                    raise UnitError("power leaf must reference exactly one input variable")
                j = int(getattr(atom, "var_idxs", ())[0])
                if j < 0 or j >= len(spec.x_dims):
                    raise UnitError(
                        f"power leaf references x{j} out of range (Nx={len(spec.x_dims)})"
                    )
                base_dim = spec.x_dims[j]
                out_dim = tuple(exp * u for u in base_dim)
                return DimExpr(us, out_dim, {})

            # Inverse monomial leaves: a/x^degree (inv_monomial) or 1/x^degree (rinv_monomial).
            # Output dimension is -degree * dim(input).
            if kind in (
                "inv_monomial", "inverse_monomial", "inv_mono",
                "rinv_monomial", "r_inv_monomial", "rinverse_monomial",
            ):
                deg = int(atom.kwargs.get("degree", 1))
                exp = _as_fraction(float(-deg))

                if input_node is not None:
                    base = rec(input_node)
                    return base.scale(exp)

                if len(getattr(atom, "var_idxs", ())) != 1:
                    raise UnitError("inv_monomial leaf must reference exactly one input variable")
                j = int(getattr(atom, "var_idxs", ())[0])
                if j < 0 or j >= len(spec.x_dims):
                    raise UnitError(
                        f"inv_monomial leaf references x{j} out of range (Nx={len(spec.x_dims)})"
                    )
                base_dim = spec.x_dims[j]
                out_dim = tuple(exp * u for u in base_dim)
                return DimExpr(us, out_dim, {})

            # Rational polynomial leaves (P(x)/Q(x)).  Every producer must
            # expose its active numerator and denominator monomials so the
            # shared coefficient solver can type each additive term exactly.
            if kind in ("ratpoly", "rratpoly", "rational_poly", "r_rational_poly", "rrational_poly"):
                exps_num_override = atom.kwargs.get("exps_num_override")
                exps_den_override = atom.kwargs.get("exps_den_override")
                if (exps_num_override is None) != (exps_den_override is None):
                    raise UnitError(
                        "ratpoly exact support requires both numerator and denominator overrides"
                    )
                if exps_num_override is not None:
                    # Resolve every effective input dimension and use the shared
                    # per-coefficient solver; no span variable or post-hoc guess
                    # is involved.
                    effective_dims: List[Dim] = []
                    try:
                        for input_expr in get_input_exprs(atom):
                            input_dim_expr = rec(input_expr)
                            if input_dim_expr.coeffs:
                                raise UnitError(
                                    "ratpoly exact support input dimension is underdetermined"
                                )
                            effective_dims.append(input_dim_expr.const)
                    except UnitError:
                        raise
                    except Exception as exc:
                        raise UnitError(
                            f"ratpoly exact support inputs are invalid: {exc}"
                        ) from exc
                    if not effective_dims:
                        raise UnitError(
                            "ratpoly exact support requires at least one effective input"
                        )

                    try:
                        from .coefficient_units import (
                            monomial_dimension,
                            solve_rational_coefficient_gauge,
                        )

                        numerator_rows = tuple(tuple(row) for row in exps_num_override)
                        denominator_rows = tuple(tuple(row) for row in exps_den_override)
                        if not numerator_rows or not denominator_rows:
                            raise UnitError(
                                "ratpoly exact support cannot be empty"
                            )
                        numerator_block_dim = monomial_dimension(
                            numerator_rows[0], effective_dims
                        )
                        denominator_block_dim = monomial_dimension(
                            denominator_rows[0], effective_dims
                        )
                        rational_target_dim = sub_dim(
                            numerator_block_dim, denominator_block_dim
                        )
                        solution = solve_rational_coefficient_gauge(
                            target_dim=rational_target_dim,
                            input_dims=effective_dims,
                            numerator_exponents=numerator_rows,
                            denominator_exponents=denominator_rows,
                            numerator_pivot=(
                                len(numerator_rows) - 1
                                if kind in (
                                    "rratpoly",
                                    "r_rational_poly",
                                    "rrational_poly",
                                )
                                else None
                            ),
                            coefficient_policy="free_const_only",
                        )
                    except UnitError:
                        raise
                    except Exception as exc:
                        raise UnitError(
                            f"ratpoly exact support is malformed: {exc}"
                        ) from exc
                    if not solution.ok or solution.target_dim is None:
                        raise UnitError(
                            "ratpoly exact coefficient support is inconsistent: "
                            f"{solution.code}: {solution.reason}"
                        )
                    return DimExpr(us, solution.target_dim, {})

                raise UnitError(
                    "ratpoly unit inference requires explicit numerator and denominator "
                    "supports; the legacy rational-span fallback has been removed"
                )

            # Ratio-polynomial leaves: P(x_num/x_den).  With free coefficients
            # the output can carry any dimension in the rational span of the
            # input dims.  Allocate ONE span variable.
            if kind in ("ratio_poly", "ratiopoly", "ratio_polynomial", "rratio_poly", "rratiopoly", "rratio_polynomial"):
                idxs = tuple(getattr(atom, "var_idxs", ()))
                if len(idxs) != 2:
                    raise UnitError("ratio_poly leaf requires exactly two input variables")
                rpo_basis: List[Dim] = []
                rpo_seen: set[Dim] = set()
                if input_node is not None:
                    inp_expr = rec(input_node)
                    if not inp_expr.coeffs:
                        d = inp_expr.const
                        if not is_dimless(d) and d not in rpo_seen:
                            rpo_basis.append(d)
                            rpo_seen.add(d)
                else:
                    for j in idxs:
                        idx = int(j)
                        if idx < 0 or idx >= len(spec.x_dims):
                            raise UnitError(
                                f"ratio_poly leaf references x{idx} out of range "
                                f"(Nx={len(spec.x_dims)})"
                            )
                        d = spec.x_dims[idx]
                        if not is_dimless(d) and d not in rpo_seen:
                            rpo_basis.append(d)
                            rpo_seen.add(d)
                rpo_basis = _reduce_to_independent(rpo_basis)
                if not rpo_basis:
                    return DimExpr.dimless(us)  # all-dimless → dimless output
                key = _atom_key(atom, share_by_tag=spec.share_atom_units_by_tag)
                if key not in key_to_var:
                    key_to_var[key] = len(key_to_var)
                var_id = key_to_var[key]
                if var_id not in span_bases:
                    span_bases[var_id] = list(rpo_basis)
                return DimExpr.var(us, var_id)

            # Analytic leaf families whose parameters are assumed dimensionless.
            # These are treated as dimensionless functions; to prevent implicit
            # unit-carrying coefficients, we reject them when they ingest *unitful*
            # effective inputs.
            #
            # If the atom provides an explicit input_expr, we enforce that *expression*
            # is dimensionless. This allows e.g. sin(x0/x1) even when x0 and x1 are
            # individually unitful but commensurate.
            if kind in set(getattr(spec, "dimless_atom_kinds", ())):
                if input_node is not None:
                    # Every effective input must be dimensionless, not just
                    # inputs[0].
                    input_exprs = (
                        get_input_exprs(atom)
                        if has_nontrivial_input(atom)
                        else (input_node,)
                    )
                    for input_expr in input_exprs:
                        constraints.append(rec(input_expr))  # enforce == 0
                    return DimExpr.dimless(us)

                for j in getattr(atom, "var_idxs", ()):  # raw x indices
                    idx = int(j)
                    if idx < 0 or idx >= len(spec.x_dims):
                        raise UnitError(
                            f"Atom references x{idx} out of range (Nx={len(spec.x_dims)})"
                        )
                    if not is_dimless(spec.x_dims[idx]):
                        raise UnitError(
                            f"Atom kind '{kind}' expects dimensionless inputs; x{idx} has units {spec.unit_system.format_dim(spec.x_dims[idx])}"
                        )
                return DimExpr.dimless(us)

            # Optional NN semantics.
            if kind == "nn" and str(getattr(spec, "nn_semantics", "unknown")).lower() in (
                "dimless",
                "dimensionless",
                "dl",
            ):
                if input_node is not None:
                    # Every effective input must be dimensionless, not just
                    # inputs[0].
                    input_exprs = (
                        get_input_exprs(atom)
                        if has_nontrivial_input(atom)
                        else (input_node,)
                    )
                    for input_expr in input_exprs:
                        constraints.append(rec(input_expr))  # enforce == 0
                    return DimExpr.dimless(us)

                for j in getattr(atom, "var_idxs", ()):  # raw x indices
                    idx = int(j)
                    if idx < 0 or idx >= len(spec.x_dims):
                        raise UnitError(
                            f"Atom references x{idx} out of range (Nx={len(spec.x_dims)})"
                        )
                    if not is_dimless(spec.x_dims[idx]):
                        raise UnitError(
                            f"NN atom expects dimensionless inputs; x{idx} has units {spec.unit_system.format_dim(spec.x_dims[idx])}"
                        )
                return DimExpr.dimless(us)

            # nn_semantics="span": dim(NN) ∈ rational_span(input_dims ∪ const_dims).
            if kind == "nn" and str(getattr(spec, "nn_semantics", "unknown")).lower() == "span":
                # Compute span basis from effective inputs (atom.inputs),
                # NOT raw var_idxs.  This correctly handles compound atoms
                # like z = x0/x1 where the NN sees a dimensionless channel.
                basis: List[Dim] = []
                seen: set[Dim] = set()
                fall_back_to_unknown = False
                for inp in (atom.inputs or ()):
                    inp_expr = rec(inp)
                    if inp_expr.coeffs:
                        # Input dim depends on unknowns — can't determine span
                        fall_back_to_unknown = True
                        break
                    d = inp_expr.const
                    if not is_dimless(d) and d not in seen:
                        basis.append(d)
                        seen.add(d)
                if fall_back_to_unknown:
                    # Drop through to the normal "unknown" fallback below
                    pass
                else:
                    # Also add declared constant dims
                    for d in list(spec.free_const_dims.values()) + list(spec.fixed_const_dims.values()):
                        if not is_dimless(d) and d not in seen:
                            basis.append(d)
                            seen.add(d)
                    # Remove linearly dependent vectors so that
                    # _rref_unique_var_value can pin every coefficient.
                    basis = _reduce_to_independent(basis)
                    if not basis:
                        return DimExpr.dimless(us)  # all-dimless inputs → dimless output
                    # Allocate var_id only after confirming non-empty basis
                    key = _atom_key(atom, share_by_tag=spec.share_atom_units_by_tag)
                    if key not in key_to_var:
                        key_to_var[key] = len(key_to_var)
                    var_id = key_to_var[key]
                    if var_id not in span_bases:
                        span_bases[var_id] = list(basis)
                    else:
                        # Same tag, different inputs: union of bases
                        existing = set(span_bases[var_id])
                        for d in basis:
                            if d not in existing:
                                span_bases[var_id].append(d)
                                existing.add(d)
                    return DimExpr.var(us, var_id)

        key = _atom_key(atom, share_by_tag=spec.share_atom_units_by_tag)
        if key not in key_to_var:
            key_to_var[key] = len(key_to_var)

        # Even when an atom's *output* units are treated as unknown, any explicit
        # input expression we attach to it (compound-variable preconditioning) is
        # analytic and must still be dimensionally consistent.
        if input_node is not None:
            _ = rec(input_node)
        return DimExpr.var(us, key_to_var[key])

    def _store(node: Node, expr: DimExpr) -> DimExpr:
        """Optionally record node → DimExpr mapping."""
        if out_node_exprs is not None:
            out_node_exprs[id(node)] = expr
        return expr

    def rec(node: Node) -> DimExpr:
        if isinstance(node, ConstNode):
            # Fixed numeric constants are treated as pure numbers.
            return _store(node, DimExpr.dimless(us))
        if isinstance(node, AtomNode):
            return _store(node, atom_expr(node))

        if isinstance(node, AddNode):
            L = rec(node.left)
            R = rec(node.right)
            # L - R == 0
            constraints.append(L.sub(R))
            return _store(node, L)

        if isinstance(node, MulNode):
            L = rec(node.left)
            R = rec(node.right)
            return _store(node, L.add(R))

        # Optional division node support (if DivNode is defined in bridges.py).
        if DivNode is not None and isinstance(node, DivNode):  # type: ignore[arg-type]
            num = getattr(node, "numerator", None)
            den = getattr(node, "denominator", None)
            if num is None:
                num = getattr(node, "left", None)
            if den is None:
                den = getattr(node, "right", None)
            if num is None or den is None:
                raise TypeError(f"DivNode missing numerator/denominator fields: {node!r}")
            return _store(node, rec(num).sub(rec(den)))

        if isinstance(node, PowNode):
            base = rec(node.base)
            s = _as_fraction(node.exponent)
            return _store(node, base.scale(s))

        if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode)):
            arg = rec(node.arg)
            # arg must be dimensionless: arg == 0
            constraints.append(arg)  # arg - 0 == 0
            return _store(node, DimExpr.dimless(us))

        # ---- Complex unary ops ------------------------------------------------
        if isinstance(node, ConjNode):
            # Conjugate preserves dimension
            return _store(node, rec(node.arg))

        if isinstance(node, (RealNode, ImagNode)):
            # Real/Imag extract preserve dimension
            return _store(node, rec(node.arg))

        if isinstance(node, (AbsNode, ArgNode)):
            # |z| preserves dimension; arg(z) is dimensionless (angle)
            if isinstance(node, ArgNode):
                # Phase angle should be dimensionless
                arg_dim = rec(node.arg)
                constraints.append(arg_dim)  # arg must be dimensionless
                return _store(node, DimExpr.dimless(us))
            # AbsNode preserves dimension
            return _store(node, rec(node.arg))

        raise TypeError(f"Unhandled AST node type for units: {type(node)}")

    root_expr = rec(root)
    # root == y_phi
    constraints.append(root_expr.sub(DimExpr(us, spec.y_phi_dim, {})))
    return root_expr, constraints, key_to_var, span_bases


def _gauss_consistent(A: List[List[Fraction]], b: List[Fraction]) -> bool:
    """Check consistency of A x = b over rationals (Fractions)."""
    m = len(A)
    if m == 0:
        return True
    n = len(A[0]) if A else 0

    # Build augmented matrix
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]

    r = 0
    for c in range(n):
        # Find pivot
        pivot = None
        for i in range(r, m):
            if M[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            M[r], M[pivot] = M[pivot], M[r]

        piv = M[r][c]
        # Normalize pivot row
        for j in range(c, n + 1):
            M[r][j] /= piv

        # Eliminate below
        for i in range(r + 1, m):
            factor = M[i][c]
            if factor == 0:
                continue
            for j in range(c, n + 1):
                M[i][j] -= factor * M[r][j]

        r += 1
        if r >= m:
            break

    # Check for 0 == nonzero rows
    for i in range(m):
        all0 = True
        for c in range(n):
            if M[i][c] != 0:
                all0 = False
                break
        if all0 and M[i][n] != 0:
            return False
    return True


def _solve_affine_system(
    A: List[List[Fraction]], b: List[Fraction], n_cols: int,
) -> Tuple[List[Fraction] | None, List[List[Fraction]]]:
    """Solve A x = b over exact rationals via RREF.

    Returns ``(particular, null_basis)`` where *particular* is a solution
    vector of length *n_cols* (or ``None`` if inconsistent), and *null_basis*
    is a list of vectors spanning ker(A).
    """
    m = len(A)
    if n_cols == 0:
        # No variables — check that b == 0.
        for bi in b:
            if bi != 0:
                return (None, [])
        return ([], [])

    # Build augmented matrix [A | b]
    M: List[List[Fraction]] = [
        [Fraction(A[i][j]) for j in range(n_cols)] + [Fraction(b[i])]
        for i in range(m)
    ]

    pivot_col: List[int] = []  # pivot column for each pivot row
    row = 0
    for col in range(n_cols):
        # Find pivot
        piv = None
        for r in range(row, m):
            if M[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        if piv != row:
            M[row], M[piv] = M[piv], M[row]
        pv = M[row][col]
        for c in range(col, n_cols + 1):
            M[row][c] /= pv
        # Eliminate in all other rows
        for r in range(m):
            if r == row:
                continue
            f = M[r][col]
            if f == 0:
                continue
            for c in range(col, n_cols + 1):
                M[r][c] -= f * M[row][c]
        pivot_col.append(col)
        row += 1
        if row >= m:
            break

    # Inconsistency check: 0 == nonzero
    for r in range(m):
        if all(M[r][c] == 0 for c in range(n_cols)) and M[r][n_cols] != 0:
            return (None, [])

    pivot_set = set(pivot_col)
    free_cols = [c for c in range(n_cols) if c not in pivot_set]

    # Particular solution (set free vars to 0)
    particular = [Fraction(0)] * n_cols
    pivot_row_by_col = {pc: r for r, pc in enumerate(pivot_col)}
    for pc in pivot_col:
        particular[pc] = M[pivot_row_by_col[pc]][n_cols]

    # Null-space basis: one vector per free variable
    null_basis: List[List[Fraction]] = []
    for fc in free_cols:
        vec = [Fraction(0)] * n_cols
        vec[fc] = Fraction(1)
        for pc in pivot_col:
            vec[pc] = -M[pivot_row_by_col[pc]][fc]
        null_basis.append(vec)

    return (particular, null_basis)


def _assign_columns(
    n_vars: int,
    span_bases: Dict[int, List[Dim]],
    us: UnitSystem,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], int]:
    """Compute column layout for the combined linear system.

    Returns ``(free_cols, span_cols, total_cols)`` where:
    - ``free_cols[var_id]`` is a list of B column indices (one per SI dim)
    - ``span_cols[var_id]`` is a list of m column indices (one per span basis vector)
    """
    B = len(us.base)
    free_cols: Dict[int, List[int]] = {}
    span_cols: Dict[int, List[int]] = {}
    col = 0
    for vid in range(n_vars):
        if vid in span_bases:
            m = len(span_bases[vid])
            span_cols[vid] = list(range(col, col + m))
            col += m
        else:
            free_cols[vid] = list(range(col, col + B))
            col += B
    return free_cols, span_cols, col


def _build_combined_system(
    constraints: List[DimExpr],
    n_vars: int,
    span_bases: Dict[int, List[Dim]],
    us: UnitSystem,
) -> Tuple[List[List[Fraction]], List[Fraction]]:
    """Build a single linear system that couples all SI dimensions.

    Free (non-span) unknowns get B independent columns (one per SI dim).
    Span unknowns get m columns (one per basis vector), shared across SI dims.

    Returns (A_combined, b_combined) ready for _gauss_consistent or
    _rref_unique_var_value.
    """
    B = len(us.base)
    free_cols, span_cols, total_cols = _assign_columns(n_vars, span_bases, us)

    A: List[List[Fraction]] = []
    b_vec: List[Fraction] = []

    for eq in constraints:
        for k in range(B):
            row = [Fraction(0)] * total_cols
            for var_id, coef in eq.coeffs.items():
                var_id = int(var_id)
                if var_id in span_cols:
                    basis = span_bases[var_id]
                    for j, cj in enumerate(span_cols[var_id]):
                        row[cj] += coef * basis[j][k]
                else:
                    row[free_cols[var_id][k]] += coef
            A.append(row)
            b_vec.append(-eq.const[k])

    return A, b_vec


def check_units_ast(root: Node, spec: UnitsSpec) -> UnitsCheckResult:
    """Return whether an AST is dimensionally consistent under this UnitsSpec."""
    us = spec.unit_system
    if len(spec.y_dim) != len(us.base):
        return UnitsCheckResult(False, "y_dim does not match unit basis")
    for d in spec.x_dims:
        if len(d) != len(us.base):
            return UnitsCheckResult(False, "x_dims entry does not match unit basis")

    try:
        _, constraints, key_to_var, span_bases = _infer_dimexpr_and_constraints(root, spec)
    except UnitError as e:
        return UnitsCheckResult(False, str(e))
    except Exception as e:
        return UnitsCheckResult(False, f"units-inference-error: {e}")

    n_vars = len(key_to_var)
    if n_vars == 0:
        # Fully fixed expression: check constraints are all zero.
        for eq in constraints:
            if not is_dimless(eq.const):
                return UnitsCheckResult(False, "fixed-units-mismatch")
        return UnitsCheckResult(True, "")

    if span_bases:
        # Combined system: span unknowns couple SI dimensions.
        A_comb, b_comb = _build_combined_system(constraints, n_vars, span_bases, us)
        if not _gauss_consistent(A_comb, b_comb):
            return UnitsCheckResult(False, "units-inconsistent (span)")
        return UnitsCheckResult(True, "")

    # No span atoms: per-component solve (original path).
    A: List[List[Fraction]] = []
    C: List[Dim] = []
    for eq in constraints:
        row = [Fraction(0) for _ in range(n_vars)]
        for var_id, coef in eq.coeffs.items():
            row[int(var_id)] = Fraction(coef)
        A.append(row)
        C.append(eq.const)

    # Check consistency per base dimension component.
    for k in range(len(us.base)):
        b = [-(c[k]) for c in C]
        if not _gauss_consistent(A, b):
            return UnitsCheckResult(False, f"units-inconsistent ({us.base[k]})")

    return UnitsCheckResult(True, "")



def _rref_unique_var_value(A: List[List[Fraction]], b: List[Fraction], var_id: int) -> Fraction | None:
    """Return the unique value for variable var_id in A x = b if it is uniquely determined.

    Uses exact rational row-reduction on the augmented matrix.
    Returns None when the variable is free/underdetermined.
    Raises UnitError on inconsistency.
    """
    if not A:
        return None
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    if n == 0:
        return None
    # Build augmented matrix
    M: List[List[Fraction]] = [
        [Fraction(A[i][j]) for j in range(n)] + [Fraction(b[i])] for i in range(m)
    ]
    pivot_col_for_row: List[int] = []
    row = 0
    for col in range(n):
        # Find pivot row
        piv = None
        for r in range(row, m):
            if M[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        if piv != row:
            M[row], M[piv] = M[piv], M[row]
        pv = M[row][col]
        # Normalize pivot row
        for c in range(col, n + 1):
            M[row][c] = M[row][c] / pv
        # Eliminate this column in all other rows
        for r in range(m):
            if r == row:
                continue
            f = M[r][col]
            if f == 0:
                continue
            for c in range(col, n + 1):
                M[r][c] = M[r][c] - f * M[row][c]
        pivot_col_for_row.append(col)
        row += 1
        if row >= m:
            break

    # Check inconsistency: 0 == nonzero
    for r in range(m):
        if all(M[r][c] == 0 for c in range(n)) and M[r][n] != 0:
            raise UnitError('units-inconsistent')

    pivot_row_by_col = {c: i for i, c in enumerate(pivot_col_for_row)}
    if var_id not in pivot_row_by_col:
        return None
    pr = pivot_row_by_col[var_id]

    # If the pivot row depends on any free variable, var_id is not uniquely determined.
    for c in range(n):
        if c == var_id:
            continue
        if c not in pivot_row_by_col:  # free column
            if M[pr][c] != 0:
                return None
    return M[pr][n]


def infer_atom_output_dim(root: Node, atom: AtomNode, spec: UnitsSpec) -> Dim | None:
    """Best-effort inference of an AtomNode's output dimension within a full AST.

    This is primarily intended as a *cheap pruning hook* for Stage A: when an NN atom
    sits inside a larger expression, its output units may be uniquely determined by
    the surrounding operations and the known units of y and x.

    Returns
    -------
    Dim | None
        The inferred dimension vector if uniquely determined; otherwise None.

    Notes
    -----
    - Only returns a dimension when **all** base-dimension components are uniquely
      pinned (i.e. the variable is not expressed in terms of any free variables).
    - Raises UnitError if the overall AST is inconsistent under the provided spec.
    """
    us = spec.unit_system

    # Fast-path: known leaves
    kind = str(getattr(atom, 'kind', '')).lower()
    tag = getattr(atom, 'tag', None)
    if kind in ('var', 'x', 'input'):
        idxs = tuple(getattr(atom, 'var_idxs', ()) or ())
        if len(idxs) == 1:
            i = int(idxs[0])
            if 0 <= i < len(spec.x_dims):
                return spec.x_dims[i]
        return None
    if kind in ('free_const', 'freeconst', 'free_constant'):
        nm = None
        try:
            nm = atom.kwargs.get('name', None)
        except Exception:
            nm = None
        if nm is None and tag is not None:
            nm = str(tag)
        if nm is not None and str(nm) in spec.free_const_dims:
            return spec.free_const_dims[str(nm)]
        return None
    if kind in ('fixed_const', 'fixedconst', 'fixed_constant'):
        nm = None
        try:
            nm = atom.kwargs.get('name', None)
        except Exception:
            nm = None
        if nm is None and tag is not None:
            nm = str(tag)
        if nm is not None and str(nm) in spec.fixed_const_dims:
            return spec.fixed_const_dims[str(nm)]
        return None

    # General case: build the constraint system and see whether this atom's
    # unknown is uniquely determined.
    _root_expr, constraints, key_to_var, span_bases = _infer_dimexpr_and_constraints(root, spec)
    key = _atom_key(atom, share_by_tag=spec.share_atom_units_by_tag)
    if key not in key_to_var:
        # Either fixed by kind/tag, or computed analytically; if so, the unit is
        # already baked into the AST constraints and there is no dedicated unknown.
        return None
    var_id = int(key_to_var[key])
    n_vars = len(key_to_var)
    if n_vars <= 0:
        return None

    if span_bases:
        # Combined system: span atoms couple SI dimensions.
        A_comb, b_comb = _build_combined_system(constraints, n_vars, span_bases, us)
        B = len(us.base)

        free_cols, span_cols, _total = _assign_columns(n_vars, span_bases, us)

        if var_id in span_cols:
            # Solve for basis coefficients c_j, then reconstruct dim = Σ c_j * d_j.
            basis = span_bases[var_id]
            coeffs: List[Fraction] = []
            for cj in span_cols[var_id]:
                v = _rref_unique_var_value(A_comb, b_comb, cj)
                if v is None:
                    return None
                coeffs.append(v)
            out_dim: List[Fraction] = [Fraction(0)] * B
            for j, c in enumerate(coeffs):
                for k in range(B):
                    out_dim[k] += c * basis[j][k]
            return tuple(out_dim)
        else:
            # Free (non-span) atom in a mixed system: solve per-component columns.
            out_dim_free: List[Fraction] = []
            for cj in free_cols[var_id]:
                v = _rref_unique_var_value(A_comb, b_comb, cj)
                if v is None:
                    return None
                out_dim_free.append(v)
            return tuple(out_dim_free)

    # No span atoms: per-component solve (original path).
    A: List[List[Fraction]] = []
    C: List[Dim] = []
    for eq in constraints:
        row = [Fraction(0) for _ in range(n_vars)]
        for k, v in eq.coeffs.items():
            if 0 <= int(k) < n_vars:
                row[int(k)] = Fraction(v)
        A.append(row)
        C.append(eq.const)

    out_dim_list: List[Fraction] = []
    for k in range(len(us.base)):
        b = [-(c[k]) for c in C]
        v = _rref_unique_var_value(A, b, var_id)
        if v is None:
            return None
        out_dim_list.append(Fraction(v))
    return tuple(out_dim_list)

# ──────────────────────────────────────────────────────────────
# Dimensional feasibility gate for Stage A separability splits
# ──────────────────────────────────────────────────────────────


def _dim_in_rational_span(target: Dim, basis_dims: List[Dim]) -> bool:
    """Check if *target* is in the rational span of *basis_dims*.

    Returns True when there exist rational coefficients c_i such that
    ``sum(c_i * basis_dims[i]) == target`` (element-wise over the base
    dimensions).  Uses :func:`_gauss_consistent` for exact rational
    arithmetic.

    Trivially True when *target* is dimensionless (zero vector).
    """
    if is_dimless(target):
        return True
    if not basis_dims:
        return False
    B = len(target)  # number of SI base dimensions
    # Build one linear system per base dimension component:
    #   For each base k: sum_j c_j * basis_dims[j][k] == target[k]
    # This is A c = b with A[k][j] = basis_dims[j][k], b[k] = target[k].
    n_cols = len(basis_dims)
    A: List[List[Fraction]] = []
    b: List[Fraction] = []
    for k in range(B):
        row = [Fraction(basis_dims[j][k]) for j in range(n_cols)]
        A.append(row)
        b.append(Fraction(target[k]))
    return _gauss_consistent(A, b)


def eval_analytic_expr_dim(
    node: Node,
    x_dims: Tuple[Dim, ...],
    *,
    free_const_dims: Mapping[str, Dim] | None = None,
    fixed_const_dims: Mapping[str, Dim] | None = None,
) -> Dim | None:
    """Compute the dimension of a pure-analytic AST expression.

    Handles variables, declared free/fixed constants, arithmetic, dimensionless
    unary functions, and numeric ``ConstNode`` values.  Constant mappings are
    optional for backwards compatibility; a named constant fails closed when
    its corresponding mapping is absent or does not contain the name.

    Returns ``None`` if the expression contains NN atoms, an undeclared named
    constant, or another unsupported node.
    """
    free_dims = free_const_dims or {}
    fixed_dims = fixed_const_dims or {}

    def _declared_constant_dim(atom: AtomNode, table: Mapping[str, Dim]) -> Dim | None:
        kwargs = getattr(atom, "kwargs", {}) or {}
        name = kwargs.get("name", kwargs.get("const_name"))
        if name is None:
            name = getattr(atom, "tag", None)
        if name is None:
            return None
        dim = table.get(str(name))
        return None if dim is None else tuple(dim)

    def _rec(child: Node) -> Dim | None:
        return eval_analytic_expr_dim(
            child,
            x_dims,
            free_const_dims=free_dims,
            fixed_const_dims=fixed_dims,
        )

    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input") and len(node.var_idxs) == 1:
            idx = node.var_idxs[0]
            if 0 <= idx < len(x_dims):
                return x_dims[idx]
        if kind in ("free_const", "freeconst", "free_constant"):
            return _declared_constant_dim(node, free_dims)
        if kind in ("fixed_const", "fixedconst", "fixed_constant"):
            return _declared_constant_dim(node, fixed_dims)
        return None

    if isinstance(node, ConstNode):
        # Numeric constants are dimensionless
        if x_dims:
            return tuple(Fraction(0) for _ in x_dims[0])
        return None

    if isinstance(node, MulNode):
        ld = _rec(node.left)
        rd = _rec(node.right)
        if ld is None or rd is None:
            return None
        return tuple(a + b for a, b in zip(ld, rd))

    if isinstance(node, PowNode):
        bd = _rec(node.base)
        if bd is None:
            return None
        exp = Fraction(node.exponent).limit_denominator(128)
        return tuple(e * exp for e in bd)

    if isinstance(node, AddNode):
        ld = _rec(node.left)
        rd = _rec(node.right)
        if ld is None or rd is None or ld != rd:
            return None
        return ld

    if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode)):
        ad = _rec(node.arg)
        if ad is None or not is_dimless(ad):
            return None
        if x_dims:
            return tuple(Fraction(0) for _ in x_dims[0])
        return tuple()

    if isinstance(node, (ConjNode, RealNode, ImagNode, AbsNode)):
        return _rec(node.arg)

    if isinstance(node, ArgNode):
        ad = _rec(node.arg)
        if ad is None:
            return None
        if x_dims:
            return tuple(Fraction(0) for _ in x_dims[0])
        return tuple()

    # Unsupported node types (NN, etc.)
    return None


def check_split_feasibility(
    op: str,
    group1: List[int],
    group2: List[int],
    y_phi_dim: Dim,
    x_dims: Tuple[Dim, ...],
    us: UnitSystem,
    free_const_dims: Mapping[str, Dim] | None = None,
    fixed_const_dims: Mapping[str, Dim] | None = None,
    has_offset: bool = False,
    context_dims: List[Dim] | None = None,
    compound_dims: Mapping[str, Dim] | None = None,
) -> Tuple[bool, str]:
    """Check whether a separability split is dimensionally feasible.

    Parameters
    ----------
    op : str
        ``"add"`` for additive splits, ``"mul"`` for multiplicative.
    group1, group2 : list[int]
        Variable indices assigned to each child.  May contain non-int tokens
        (e.g. ``"z"``) for compound variables.
    y_phi_dim : Dim
        Target output dimension (y after y-transform).
    x_dims : tuple[Dim, ...]
        Dimensions of all input variables.
    us : UnitSystem
        Unit system for formatting messages.
    free_const_dims : mapping, optional
        Pre-declared trainable free constants with known dimensions.
    fixed_const_dims : mapping, optional
        Pre-declared fixed physical constants with known dimensions.
    has_offset : bool
        If True, the split includes an additive offset constant.
    context_dims : list[Dim], optional
        Dimensions from multiplicative siblings in the parent AST.
        When an atom is nested inside a MulNode, its siblings provide
        extra dimensional degrees of freedom that relax the target.
    compound_dims : mapping, optional
        Dimensions of compound variable tokens (e.g. ``{"z": dim}``).

    Returns
    -------
    (feasible, reason) : (bool, str)
        ``feasible=True`` means the split is not provably impossible.
        ``reason`` explains why the split was rejected.
    """
    if is_dimless(y_phi_dim):
        return (True, "")

    fc_dims = list((free_const_dims or {}).values())
    fx_dims = list((fixed_const_dims or {}).values())
    ctx = list(context_dims or [])
    _cd = compound_dims or {}

    def _compound_dims_from_group(grp):
        """Collect known dims for non-int tokens (e.g. 'z') in a group."""
        return [_cd[tok] for tok in grp if not isinstance(tok, int) and tok in _cd]

    if bool(has_offset) and (not is_dimless(y_phi_dim)):
        # An explicit additive offset requires a standalone constant with units of y_phi.
        # Under the default policy, this is only possible if the user has declared
        # a trainable free constant with exactly that dimension.
        if not any(d == y_phi_dim for d in fc_dims):
            return (
                False,
                f"offset requires a declared free constant with dim {us.format_dim(y_phi_dim)}",
            )

    if op == "add":
        # Additive: each child must independently produce y_phi_dim,
        # but context_dims from multiplicative ancestors relax the target.
        for label, grp in [("child1", group1), ("child2", group2)]:
            child_basis = (
                [x_dims[i] for i in grp if isinstance(i, int) and i < len(x_dims)]
                + _compound_dims_from_group(grp)
                + fc_dims + fx_dims + ctx
            )
            if not _dim_in_rational_span(y_phi_dim, child_basis):
                child_dim_strs = [us.format_dim(d) for d in child_basis] if child_basis else ["(none)"]
                return (
                    False,
                    f"additive {label} inputs {grp} with dims [{', '.join(child_dim_strs)}] "
                    f"cannot reach target dim {us.format_dim(y_phi_dim)}",
                )
        return (True, "")

    if op == "mul":
        # Multiplicative: combined inputs must span y_phi_dim.
        all_members = set(group1) | set(group2)
        all_grp = sorted(i for i in all_members if isinstance(i, int))
        combined_basis = (
            [x_dims[i] for i in all_grp if i < len(x_dims)]
            + _compound_dims_from_group(list(all_members))
            + fc_dims + fx_dims + ctx
        )
        if not _dim_in_rational_span(y_phi_dim, combined_basis):
            combined_strs = [us.format_dim(d) for d in combined_basis] if combined_basis else ["(none)"]
            return (
                False,
                f"multiplicative inputs {all_grp} with dims [{', '.join(combined_strs)}] "
                f"cannot reach target dim {us.format_dim(y_phi_dim)}",
            )
        return (True, "")

    # Unknown op — don't block.
    return (True, "")


def filter_y_transform_names_by_units(
    names: Sequence[str],
    *,
    y_dim: Dim,
    us: UnitSystem,
) -> List[str]:
    """Filter y-transform names to those that are dimensionally admissible."""
    out: List[str] = []
    for n in names:
        try:
            y_transform_dim(y_dim, n, us)
        except UnitError:
            continue
        out.append(n)
    return out


# ──────────────────────────────────────────────────────────────
# Buckingham-Sudoku: global constraint propagation (Level 2)
# ──────────────────────────────────────────────────────────────


def _eval_dimexpr_at(
    expr: DimExpr,
    sol: List[Fraction],
    free_cols: Dict[int, List[int]],
    span_cols: Dict[int, List[int]],
    span_bases: Dict[int, List[Dim]],
    B: int,
) -> Dim:
    """Evaluate a DimExpr at a given solution vector → concrete Dim."""
    out = [Fraction(0)] * B
    for k in range(B):
        out[k] = expr.const[k]
        for var_id, coef in expr.coeffs.items():
            var_id = int(var_id)
            if var_id in span_cols:
                basis = span_bases[var_id]
                for j, cj in enumerate(span_cols[var_id]):
                    out[k] += coef * sol[cj] * basis[j][k]
            else:
                out[k] += coef * sol[free_cols[var_id][k]]
    return tuple(out)


def _eval_dimexpr_direction(
    expr: DimExpr,
    direction: List[Fraction],
    free_cols: Dict[int, List[int]],
    span_cols: Dict[int, List[int]],
    span_bases: Dict[int, List[Dim]],
    B: int,
) -> Dim:
    """Evaluate a DimExpr on a null-space direction → Dim displacement."""
    out = [Fraction(0)] * B
    for k in range(B):
        for var_id, coef in expr.coeffs.items():
            var_id = int(var_id)
            if var_id in span_cols:
                basis = span_bases[var_id]
                for j, cj in enumerate(span_cols[var_id]):
                    out[k] += coef * direction[cj] * basis[j][k]
            else:
                out[k] += coef * direction[free_cols[var_id][k]]
    return tuple(out)


def compute_node_domains(
    root: Node,
    spec: UnitsSpec,
) -> Dict[int, DimSubspace] | None:
    """Compute feasible dimension subspaces for every node in *root*.

    Returns ``{id(node): DimSubspace}`` mapping, or ``None`` if the global
    system is inconsistent (the entire AST is dimensionally infeasible).

    This is the core of the *Buckingham-Sudoku* Level 2 check: it solves the
    full constraint system once and projects the solution space onto each node.
    """
    us = spec.unit_system
    B = len(us.base)

    node_exprs: Dict[int, DimExpr] = {}
    try:
        _root_expr, constraints, key_to_var, span_bases = _infer_dimexpr_and_constraints(
            root, spec, out_node_exprs=node_exprs,
        )
    except (UnitError, TypeError):
        return None

    n_vars = len(key_to_var)

    # ---- Fully fixed (no unknowns): all nodes are pinned ----
    if n_vars == 0:
        for eq in constraints:
            if not is_dimless(eq.const):
                return None  # inconsistent
        domains: Dict[int, DimSubspace] = {}
        for nid, expr in node_exprs.items():
            domains[nid] = DimSubspace(offset=expr.const)
        return domains

    # ---- Build and solve the global system ----
    A_comb, b_comb = _build_combined_system(constraints, n_vars, span_bases, us)
    free_cols, span_cols, total_cols = _assign_columns(n_vars, span_bases, us)

    particular, null_basis = _solve_affine_system(A_comb, b_comb, total_cols)
    if particular is None:
        return None  # inconsistent

    # ---- Project onto each node ----
    domains = {}
    for nid, expr in node_exprs.items():
        offset = _eval_dimexpr_at(
            expr, particular, free_cols, span_cols, span_bases, B,
        )
        raw_basis: List[Dim] = []
        for nvec in null_basis:
            d = _eval_dimexpr_direction(
                expr, nvec, free_cols, span_cols, span_bases, B,
            )
            if not is_dimless(d):
                raw_basis.append(d)
        ind_basis = _reduce_to_independent(raw_basis)
        domains[nid] = DimSubspace(offset=offset, basis=tuple(ind_basis))

    return domains


def propose_split(
    root: Node,
    spec: UnitsSpec,
    atom: AtomNode,
    op: str,
    group1: List[int],
    group2: List[int],
) -> Dict[int, DimSubspace] | None:
    """Build a candidate split AST and return its node domains, or None if infeasible.

    Constructs a lightweight AST where *atom* is replaced by
    ``op(NN[group1], NN[group2])`` and runs :func:`compute_node_domains`
    on the result.

    Parameters
    ----------
    op : ``"add"`` or ``"mul"``
    group1, group2 : variable index lists for each child
    """
    from .bridges import replace_atom_in_ast

    child1 = AtomNode(kind="nn", var_idxs=tuple(int(v) for v in group1), tag=None)
    child2 = AtomNode(kind="nn", var_idxs=tuple(int(v) for v in group2), tag=None)

    if op == "add":
        replacement = AddNode(child1, child2)
    elif op == "mul":
        replacement = MulNode(child1, child2)
    else:
        return None

    # replace_atom_in_ast uses identity matching and reconstructs a new tree
    # without mutating the original — no need to clone first.
    candidate = replace_atom_in_ast(root, atom, replacement)
    return compute_node_domains(candidate, spec)
