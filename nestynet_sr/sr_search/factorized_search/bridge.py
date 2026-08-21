# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Bridge between factorized_search explorer tuple-AST and NestyNet_SR dataclass-AST.

Provides:
    factorized_search_to_nestynet(node)   – tuple AST → NestyNet_SR Node
    nestynet_to_factorized_search(node)   – NestyNet_SR Node → tuple AST
    dims_to_fraction(dims)      – float-tuple dims → Fraction-tuple Dim
    fraction_to_dims(dim)       – Fraction-tuple Dim → float-tuple
    dims_to_units_spec(...)     – build UnitsSpec from float dim info
    run_explorer(...)           – programmatic API returning list[dict]
"""

from __future__ import annotations

import logging
import math
import time
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# NestyNet_SR AST imports
# ---------------------------------------------------------------------------
from nestynet_sr.sr_core.bridges import (
    Add, Mul, Pow, Div, Sin, Cos, Exp, Log, Var,
    ConstNode, AtomNode, Node, clone_ast,
    AddNode, MulNode, PowNode, SinNode, CosNode, ExpNode, LogNode,
    AsinNode, AcosNode, AtanNode, AbsNode, ConjNode, RealNode, ImagNode, ArgNode,
    FixedConst, FreeConst, Scale,
)
from nestynet_sr.sr_core.sympy_bridge import sympy_to_nestynet
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, Dim

# ---------------------------------------------------------------------------
# factorized symbolic search explorer imports (relative — same package)
# ---------------------------------------------------------------------------
from .engine.search import run_explorer_core
from .explorer import node_str, node_size, fit_best, make_engine_runtime_hooks


# ===================================================================
# Atom-local AST embedding helpers
# ===================================================================

def remap_var_to_exprs(node: Node, expr_list: list) -> Node:
    """Replace Var(i) in an atom-local AST with clone_ast(expr_list[i]).

    The factorized symbolic search explorer produces ASTs with local variable indices 0..nvars-1.
    Each entry in *expr_list* is the corresponding input expression for that
    local variable. For compound atoms, expr_list entries can be arbitrary
    NestyNet_SR AST subtrees.
    """
    from nestynet_sr.sr_core.bridges import (
        AbsNode, ConjNode, RealNode, ImagNode, ArgNode,
    )

    if isinstance(node, AtomNode):
        if node.kind == "var":
            idx = node.var_idxs[0]
            return clone_ast(expr_list[idx])
        return node  # non-var atoms pass through

    if isinstance(node, ConstNode):
        return node
    if isinstance(node, AddNode):
        return AddNode(remap_var_to_exprs(node.left, expr_list),
                       remap_var_to_exprs(node.right, expr_list))
    if isinstance(node, MulNode):
        return MulNode(remap_var_to_exprs(node.left, expr_list),
                       remap_var_to_exprs(node.right, expr_list))
    if isinstance(node, PowNode):
        return PowNode(remap_var_to_exprs(node.base, expr_list), node.exponent)
    if isinstance(node, SinNode):
        return SinNode(remap_var_to_exprs(node.arg, expr_list))
    if isinstance(node, CosNode):
        return CosNode(remap_var_to_exprs(node.arg, expr_list))
    if isinstance(node, ExpNode):
        return ExpNode(remap_var_to_exprs(node.arg, expr_list))
    if isinstance(node, LogNode):
        return LogNode(remap_var_to_exprs(node.arg, expr_list))
    if isinstance(node, AbsNode):
        return AbsNode(remap_var_to_exprs(node.arg, expr_list))
    if isinstance(node, ConjNode):
        return ConjNode(remap_var_to_exprs(node.arg, expr_list))
    if isinstance(node, RealNode):
        return RealNode(remap_var_to_exprs(node.arg, expr_list))
    if isinstance(node, ImagNode):
        return ImagNode(remap_var_to_exprs(node.arg, expr_list))
    if isinstance(node, ArgNode):
        return ArgNode(remap_var_to_exprs(node.arg, expr_list))

    raise ValueError(f"remap_var_to_exprs: unsupported node type {type(node).__name__}")


def promote_argument_const_scales(
    root: Node,
    *,
    tag_prefix: str = "factorized_search",
    use_free_const: bool = False,
) -> Node:
    """Promote immediate const multipliers in trig/log/exp arguments to trainable leaves.

    Rewrites only patterns of the form:
      ``sin(const * u)``, ``sin(u * const)``, and likewise for ``cos``/``log``/``exp``.
    The promoted leaf is a ``Scale`` by default, or a ``FreeConst`` when
    ``use_free_const=True``.
    """
    counter = 0

    def _next_name() -> str:
        nonlocal counter
        name = f"{tag_prefix}__arg_scale_{counter}"
        counter += 1
        return name

    def _as_finite_float(v) -> Optional[float]:
        if isinstance(v, complex):
            return None
        try:
            out = float(v)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return out

    def _mk_leaf(init_val: float) -> Node:
        nm = _next_name()
        if use_free_const:
            return FreeConst(nm, tag=nm, init=float(init_val), scope="experiment")
        return Scale(nm, tag=nm, init=float(init_val))

    def _maybe_promote_arg(arg: Node) -> Node:
        if not isinstance(arg, MulNode):
            return arg

        l_is_const = isinstance(arg.left, ConstNode)
        r_is_const = isinstance(arg.right, ConstNode)
        if l_is_const == r_is_const:
            return arg

        if l_is_const:
            c_val = _as_finite_float(arg.left.value)
            if c_val is None:
                return arg
            return MulNode(_mk_leaf(c_val), arg.right)

        c_val = _as_finite_float(arg.right.value)
        if c_val is None:
            return arg
        return MulNode(arg.left, _mk_leaf(c_val))

    def _walk(node: Node) -> Node:
        if isinstance(node, (AtomNode, ConstNode)):
            return node
        if isinstance(node, AddNode):
            return AddNode(_walk(node.left), _walk(node.right))
        if isinstance(node, MulNode):
            return MulNode(_walk(node.left), _walk(node.right))
        if isinstance(node, PowNode):
            return PowNode(_walk(node.base), node.exponent)
        if isinstance(node, SinNode):
            return SinNode(_maybe_promote_arg(_walk(node.arg)))
        if isinstance(node, CosNode):
            return CosNode(_maybe_promote_arg(_walk(node.arg)))
        if isinstance(node, LogNode):
            return LogNode(_maybe_promote_arg(_walk(node.arg)))
        if isinstance(node, ExpNode):
            return ExpNode(_maybe_promote_arg(_walk(node.arg)))
        if isinstance(node, AbsNode):
            return AbsNode(_walk(node.arg))
        if isinstance(node, ConjNode):
            return ConjNode(_walk(node.arg))
        if isinstance(node, RealNode):
            return RealNode(_walk(node.arg))
        if isinstance(node, ImagNode):
            return ImagNode(_walk(node.arg))
        if isinstance(node, ArgNode):
            return ArgNode(_walk(node.arg))
        return node

    return _walk(root)


def embed_mapping_in_ast(
    f_ast: Node,
    mapping: dict,
    input_exprs: list,
    *,
    units_mode: str = "raw",
    scale_name: Optional[str] = None,
    scale_kind: str = "fixed",
    scale_floor_rel: float = 0.0,
    trainable_dimless: bool = False,
    tag_prefix: Optional[str] = None,
    z_affine: bool = False,
    z_alpha_init: float = 1.0,
    z_beta_init: Optional[float] = None,
    z_train_alpha: bool = True,
    z_train_beta: bool = True,
    sin_arg_mode: str = "omega_z",
) -> Optional[Node]:
    """Wrap a factorized symbolic search symbolic AST in its learned mapping coefficients.

    The explorer returns an inner expression f(x) plus a mapping from f(x) to
    the target y. This function builds a NestyNet_SR AST for the full mapped
    expression and remaps local variables to the atom's input expressions.

    Parameters
    ----------
    units_mode : str
        ``"raw"`` (default): embed coefficients as dimensionless ConstNodes.
        ``"scaled"``: factor out a single unit-carrying scale *S* so that
        the inner mapping is entirely dimensionless: ``y = S · g(z)``.
    scale_name : str, optional
        Name for the scale leaf (required when ``units_mode="scaled"``).
    scale_kind : str
        ``"fixed"`` (default) or ``"free"`` — whether the unit-carrying scale
        is a FixedConst or FreeConst.
    scale_floor_rel : float
        If > 0, build a smooth floor guard ``S_safe = sqrt(S² + floor²)``
        to prevent division blow-up if LM drives S→0.
    trainable_dimless : bool
        If True, dimensionless mapping coefficients become ``Scale(...)`` leaves
        (trainable by LM) instead of frozen ``ConstNode`` values.
    tag_prefix : str, optional
        Prefix for generating Scale/const tags.  Defaults to *scale_name* or
        ``"factorized_search"``.
    z_affine : bool
        If True, the standardised input z is reparameterised as
        ``z = α·u + β`` where α, β are trainable Scale leaves.  This lets LM
        slide and stretch the mapping input without rediscovering the skeleton.
    z_alpha_init : float
        Initial value for the affine scale α (default 1.0).
    z_beta_init : float or None
        Initial value for the affine shift β.  None (default) uses ``-μ/σ``
        from the mapping's own standardisation.
    z_train_alpha : bool
        Whether α is trainable (default True).
    z_train_beta : bool
        Whether β is trainable (default True).
    sin_arg_mode : str
        Sine argument parameterisation (only affects ``kind="sine"`` in
        scaled mode).

        ``"wu"`` (recommended): ``arg = w·u`` with phase absorbed into
        rotated A', B' coefficients at init.  Best conditioned — removes
        the ω↔α and φ↔(A,B) degeneracies.

        ``"wu_phi"``: ``arg = w·u + φ`` with explicit phase Scale leaf.

        ``"omega_z"`` (default/legacy): ``arg = ω·z`` where z includes
        the μ/σ shift.

    Returns None if the mapping kind is unsupported or numerically degenerate.
    """
    kind = mapping.get("kind")
    if kind is None:
        return None

    units_mode = str(units_mode or "raw").lower().strip()
    scale_kind = str(scale_kind or "fixed").lower().strip()
    if units_mode not in ("raw", "scaled"):
        raise ValueError(f"embed_mapping_in_ast: unsupported units_mode={units_mode!r}")
    if scale_kind not in ("fixed", "free"):
        raise ValueError(f"embed_mapping_in_ast: unsupported scale_kind={scale_kind!r}")

    # Remap the explorer AST's local var indices to the atom's input expressions
    input_exprs_list = list(input_exprs)
    remapped = remap_var_to_exprs(f_ast, input_exprs_list)

    if tag_prefix is None:
        tag_prefix = scale_name if scale_name is not None else "factorized_search"

    # ----- helpers -----

    def _tag(suf: str) -> str:
        return f"{tag_prefix}__{suf}" if tag_prefix else suf

    def _D(val: float, suf: str, *, train: bool = True) -> Node:
        """Dimensionless coefficient: Scale leaf (trainable) or ConstNode."""
        v = float(val)
        if trainable_dimless and train:
            nm = _tag(suf)
            return Scale(nm, tag=nm, init=v)
        return ConstNode(v)

    def _S(init_val: float) -> Optional[Node]:
        """Build the unit-carrying scale node (only in scaled mode)."""
        if units_mode != "scaled":
            return None
        if scale_name is None or str(scale_name).strip() == "":
            return None
        init_val = float(init_val)
        if scale_kind == "free":
            S0: Node = FreeConst(scale_name, tag=scale_name, init=init_val)
        else:
            S0 = FixedConst(scale_name, value=init_val, tag=scale_name)
        if scale_floor_rel and float(scale_floor_rel) > 0.0:
            # Smooth positivity: S_safe = sqrt(S0² + floor²)
            floor_name = f"{scale_name}__floor"
            floor_val = abs(init_val) * float(scale_floor_rel)
            F0 = FixedConst(floor_name, value=floor_val, tag=floor_name)
            return PowNode(
                AddNode(PowNode(clone_ast(S0), 2.0), PowNode(clone_ast(F0), 2.0)),
                0.5,
            )
        return S0

    def _build_z(mu: float, std: float):
        """Build (S_node_or_None, z_node) for mapping families with μ/σ.

        In raw mode:  z = (f − μ) / σ,  optionally affine-reparameterised.
        In scaled mode: u = f/S,  z = α·u + β  (dimensionless).
        Returns (None, z) for raw, (S, z) for scaled.
        Raises ValueError on degenerate std.
        """
        if not math.isfinite(std) or abs(std) < 1e-30:
            raise ValueError("degenerate std")

        if units_mode == "raw":
            z0 = MulNode(ConstNode(1.0 / std), AddNode(remapped, ConstNode(-mu)))
            if not z_affine:
                return None, z0
            a = _D(z_alpha_init, "z_alpha", train=z_train_alpha)
            b0 = 0.0 if z_beta_init is None else float(z_beta_init)
            b = _D(b0, "z_beta", train=z_train_beta)
            return None, AddNode(MulNode(a, z0), b)

        # scaled mode: u = f/S,  z = α·u + β  (dimensionless)
        S = _S(std)
        if S is None:
            raise ValueError("scaled mode requires scale_name")
        u = MulNode(remapped, PowNode(clone_ast(S), -1.0))  # f/S
        beta0 = (-mu / std) if z_beta_init is None else float(z_beta_init)
        if not z_affine:
            # Even without full affine, keep shift trainable
            b = _D(beta0, "z_beta", train=True)
            return S, AddNode(u, b)
        a = _D(z_alpha_init, "z_alpha", train=z_train_alpha)
        b = _D(beta0, "z_beta", train=z_train_beta)
        return S, AddNode(MulNode(a, u), b)

    def _maybe_add_linear_head(out: Node, *, S: Optional[Node], std_ref: float) -> Node:
        """Attach an optional additive linear head stored in mapping['_lin_head'].

        Head format:
            mapping['_lin_head'] = {
                'terms': [tuple-AST nodes in factorized symbolic search format],
                'coeffs': [b0, a0, ..., ak],  # bias then per-term coefficients
                ...
            }

        In scaled units_mode, the bias is represented as (b0/std_ref) * S to keep it unit-consistent.
        """
        head = mapping.get("_lin_head", None)
        if not isinstance(head, dict):
            return out
        terms = head.get("terms", None)
        coeffs = head.get("coeffs", None)
        if (not isinstance(terms, (list, tuple))) or (not isinstance(coeffs, (list, tuple))):
            return out
        if len(coeffs) != len(terms) + 1:
            return out

        std_ref = float(std_ref) if (std_ref is not None and math.isfinite(float(std_ref))) else 1.0

        # Bias term (first coefficient).
        try:
            b0 = float(coeffs[0])
        except Exception:
            b0 = 0.0

        head_node: Optional[Node] = None
        if abs(b0) > 1e-30:
            if units_mode == "scaled":
                S_use = S if S is not None else _S(std_ref)
                if S_use is None:
                    return out
                head_node = MulNode(clone_ast(S_use), _D(b0 / std_ref, "head_b0"))
            else:
                head_node = _D(b0, "head_b0")

        # Per-term coefficients.
        for i, t in enumerate(terms):
            try:
                a = float(coeffs[i + 1])
            except Exception:
                continue
            if abs(a) < 1e-30:
                continue
            t_ast = factorized_search_to_nestynet(t)
            if t_ast is None:
                continue
            t_remap = remap_var_to_exprs(t_ast, input_exprs_list)
            term_node = MulNode(_D(a, f"head_a{i}"), t_remap)
            head_node = term_node if head_node is None else AddNode(head_node, term_node)

        if head_node is None:
            return out
        return AddNode(out, head_node)

    if kind == "basis_state_native":
        return _maybe_add_linear_head(remapped, S=None, std_ref=1.0)

    # ----- poly -----
    if kind == "poly":
        coeffs = mapping.get("coeffs")
        if coeffs is None:
            return None
        mu = float(mapping.get("mu", 0.0))
        std = float(mapping.get("std", 1.0))
        try:
            S, z = _build_z(mu, std)
        except ValueError:
            return None

        if units_mode == "scaled":
            inner: Node = _D(float(coeffs[0]) / std, "poly_a0")
            for k in range(1, len(coeffs)):
                ck = float(coeffs[k])
                if abs(ck) < 1e-30:
                    continue
                zc = clone_ast(z)
                zp = zc if k == 1 else PowNode(zc, float(k))
                inner = AddNode(inner, MulNode(_D(ck / std, f"poly_a{k}"), zp))
            out = MulNode(clone_ast(S), inner)
            return _maybe_add_linear_head(out, S=S, std_ref=std)

        out: Node = _D(float(coeffs[0]), "poly_c0")
        for k in range(1, len(coeffs)):
            ck = float(coeffs[k])
            if abs(ck) < 1e-30:
                continue
            zc = clone_ast(z)
            zp = zc if k == 1 else PowNode(zc, float(k))
            out = AddNode(out, MulNode(_D(ck, f"poly_c{k}"), zp))
        return _maybe_add_linear_head(out, S=S, std_ref=std)

    # ----- pade -----
    if kind == "pade":
        numer = mapping.get("numer")
        denom = mapping.get("denom")
        if numer is None or denom is None:
            return None
        mu = float(mapping.get("mu", 0.0))
        std = float(mapping.get("std", 1.0))
        try:
            S, z = _build_z(mu, std)
        except ValueError:
            return None

        if units_mode == "scaled":
            num_node: Node = _D(float(numer[0]) / std, "pade_p0")
            for k in range(1, len(numer)):
                pk = float(numer[k])
                if abs(pk) < 1e-30:
                    continue
                zc = clone_ast(z)
                zp = zc if k == 1 else PowNode(zc, float(k))
                num_node = AddNode(num_node, MulNode(_D(pk / std, f"pade_p{k}"), zp))
            # denominator q0 kept fixed (gauge)
            den_node: Node = ConstNode(float(denom[0]))
            for k in range(1, len(denom)):
                dk = float(denom[k])
                if abs(dk) < 1e-30:
                    continue
                zc = clone_ast(z)
                zp = zc if k == 1 else PowNode(zc, float(k))
                den_node = AddNode(den_node, MulNode(_D(dk, f"pade_q{k}"), zp))
            inner = MulNode(num_node, PowNode(den_node, -1.0))
            out = MulNode(clone_ast(S), inner)
            return _maybe_add_linear_head(out, S=S, std_ref=std)

        num_node: Node = _D(float(numer[0]), "pade_p0")
        for k in range(1, len(numer)):
            pk = float(numer[k])
            if abs(pk) < 1e-30:
                continue
            zc = clone_ast(z)
            zp = zc if k == 1 else PowNode(zc, float(k))
            num_node = AddNode(num_node, MulNode(_D(pk, f"pade_p{k}"), zp))
        den_node: Node = ConstNode(float(denom[0]))
        for k in range(1, len(denom)):
            dk = float(denom[k])
            if abs(dk) < 1e-30:
                continue
            zc = clone_ast(z)
            zp = zc if k == 1 else PowNode(zc, float(k))
            den_node = AddNode(den_node, MulNode(_D(dk, f"pade_q{k}"), zp))
        out = MulNode(num_node, PowNode(den_node, -1.0))
        return _maybe_add_linear_head(out, S=S, std_ref=std)

    # ----- sine -----
    if kind == "sine":
        A_val = mapping.get("A")
        B_val = mapping.get("B")
        c_val = mapping.get("c")
        omega_val = mapping.get("omega")
        if any(v is None for v in (A_val, B_val, c_val, omega_val)):
            return None
        mu = float(mapping.get("mu", 0.0))
        std = float(mapping.get("std", 1.0))
        if not math.isfinite(std) or abs(std) < 1e-30:
            return None
        A_val = float(A_val)
        B_val = float(B_val)
        c_val = float(c_val)
        omega_val = float(omega_val)

        if units_mode == "scaled":
            S = _S(std)
            if S is None:
                return None
            u = MulNode(remapped, PowNode(clone_ast(S), -1.0))  # f/S (dimensionless)

            mode = str(sin_arg_mode or "wu").lower().strip()

            if mode in ("wu", "w*u", "w_u"):
                # Best conditioned: absorb phase into rotated A'/B'.
                # ω·(u + β₀) with β₀ = -μ/σ  →  A'sin(ωu) + B'cos(ωu)
                phi0 = omega_val * (-mu / std)
                cphi = math.cos(phi0)
                sphi = math.sin(phi0)
                Arot = A_val * cphi - B_val * sphi
                Brot = A_val * sphi + B_val * cphi
                w = _D(omega_val, "sin_w")
                arg = MulNode(clone_ast(w), u)
                inner: Node = _D(c_val / std, "sin_c")
                if abs(Arot) > 1e-30:
                    inner = AddNode(inner, MulNode(_D(Arot / std, "sin_A"), SinNode(clone_ast(arg))))
                if abs(Brot) > 1e-30:
                    inner = AddNode(inner, MulNode(_D(Brot / std, "sin_B"), CosNode(clone_ast(arg))))
                out = MulNode(clone_ast(S), inner)
                return _maybe_add_linear_head(out, S=S, std_ref=std)

            if mode in ("wu_phi", "w*u+phi", "w_u_phi"):
                # Explicit phase: arg = w·u + φ
                w = _D(omega_val, "sin_w")
                phi = _D(omega_val * (-mu / std), "sin_phi")
                arg = AddNode(MulNode(clone_ast(w), u), phi)
                inner: Node = _D(c_val / std, "sin_c")
                if abs(A_val) > 1e-30:
                    inner = AddNode(inner, MulNode(_D(A_val / std, "sin_A"), SinNode(clone_ast(arg))))
                if abs(B_val) > 1e-30:
                    inner = AddNode(inner, MulNode(_D(B_val / std, "sin_B"), CosNode(clone_ast(arg))))
                out = MulNode(clone_ast(S), inner)
                return _maybe_add_linear_head(out, S=S, std_ref=std)

            # fallback "omega_z": legacy ω·z with z = u - μ/σ
            z = AddNode(u, ConstNode(-mu / std))
            arg = MulNode(_D(omega_val, "sin_w"), z)
            inner: Node = _D(c_val / std, "sin_c")
            if abs(A_val) > 1e-30:
                inner = AddNode(inner, MulNode(_D(A_val / std, "sin_A"), SinNode(clone_ast(arg))))
            if abs(B_val) > 1e-30:
                inner = AddNode(inner, MulNode(_D(B_val / std, "sin_B"), CosNode(clone_ast(arg))))
            out = MulNode(clone_ast(S), inner)
            return _maybe_add_linear_head(out, S=S, std_ref=std)

        # raw mode (legacy): z = (f - μ)/σ,  arg = ω·z
        z_ast = MulNode(ConstNode(1.0 / std), AddNode(remapped, ConstNode(-mu)))
        wz = MulNode(_D(omega_val, "sin_w"), z_ast)
        out: Node = _D(c_val, "sin_c")
        if abs(A_val) > 1e-30:
            out = AddNode(out, MulNode(_D(A_val, "sin_A"), SinNode(clone_ast(wz))))
        if abs(B_val) > 1e-30:
            out = AddNode(out, MulNode(_D(B_val, "sin_B"), CosNode(clone_ast(wz))))
        return _maybe_add_linear_head(out, S=None, std_ref=std)

    # ----- exp -----
    if kind == "exp":
        a_val = mapping.get("a")
        b_val = mapping.get("b")
        c_val = mapping.get("c")
        if any(v is None for v in (a_val, b_val, c_val)):
            return None
        mu = float(mapping.get("mu", 0.0))
        std = float(mapping.get("std", 1.0))
        if not math.isfinite(std) or abs(std) < 1e-30:
            return None
        a_val = float(a_val)
        b_val = float(b_val)
        c_val = float(c_val)

        if units_mode == "scaled":
            S = _S(std)
            if S is None:
                return None
            u = MulNode(remapped, PowNode(clone_ast(S), -1.0))  # f/S
            # Absorb shift into amplitude:
            #   a·exp(b·(u + β₀)) = (a·exp(b·β₀))·exp(b·u)
            beta0 = -mu / std
            a_rot = a_val * math.exp(b_val * beta0)
            bb = _D(b_val, "exp_b")
            inner = AddNode(
                MulNode(_D(a_rot / std, "exp_a"), ExpNode(MulNode(bb, u))),
                _D(c_val / std, "exp_c"),
            )
            out = MulNode(clone_ast(S), inner)
            return _maybe_add_linear_head(out, S=S, std_ref=std)

        # raw mode: use _build_z() for z_affine support
        try:
            S, z = _build_z(mu, std)
        except ValueError:
            return None
        bb = _D(b_val, "exp_b")
        out = AddNode(
            MulNode(_D(a_val, "exp_a"), ExpNode(MulNode(bb, z))),
            _D(c_val, "exp_c"),
        )
        return _maybe_add_linear_head(out, S=S, std_ref=std)

    # ----- power (direct f, or scaled f/S) -----
    if kind == "power":
        log_a = float(mapping.get("log_a", 0.0))
        b = float(mapping.get("b", 1.0))
        if not math.isfinite(log_a) or not math.isfinite(b):
            return None
        a = math.exp(log_a)

        # Raw mode: y = a * f^b (dimensionless a).
        if units_mode == "raw":
            out = MulNode(_D(a, "pow_a"), PowNode(remapped, b))
            return _maybe_add_linear_head(out, S=None, std_ref=float(mapping.get("std", 1.0)))

        # Scaled mode: y = S * a' * (f/S)^b, with a' dimensionless.
        std = float(mapping.get("std", 1.0))
        if (not math.isfinite(std)) or (std <= 0.0):
            std = 1.0
        S = _S(std)
        if S is None:
            return None
        u = MulNode(remapped, PowNode(clone_ast(S), -1.0))  # f/S (dimensionless)

        # Convert a → a' so the expression matches numerically at init:
        #   a f^b  ==  S * a' * (f/S)^b  =>  a' = a * S^(b-1)
        a_scaled = a * (std ** (b - 1.0))
        inner = MulNode(_D(a_scaled, "pow_a"), PowNode(u, b))
        out = MulNode(clone_ast(S), inner)
        return _maybe_add_linear_head(out, S=S, std_ref=std)

    return None


def eval_embedded_mapping_ast(root: Node, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a structural embedded NestyNet_SR AST on ``x``.

    This intentionally supports only explicit structural nodes and constant-like
    leaves. It is a lightweight invariant checker for ``embed_mapping_in_ast``;
    trainable leaves are evaluated at their initial values.
    """

    def _const_like(value: float) -> torch.Tensor:
        return torch.full((int(x.shape[0]), 1), float(value), dtype=x.dtype, device=x.device)

    def _rec(node: Node) -> torch.Tensor:
        if isinstance(node, AtomNode):
            kind = str(getattr(node, "kind", "") or "").lower()
            if kind in ("var", "x", "input"):
                if len(tuple(getattr(node, "var_idxs", ()) or ())) != 1:
                    raise ValueError("embedded AST Var node must have exactly one index")
                idx = int(tuple(node.var_idxs)[0])
                return x[:, idx : idx + 1]
            if kind == "fixed_const":
                return _const_like(float(dict(getattr(node, "kwargs", {}) or {}).get("value", 0.0)))
            if kind in ("free_const", "scale"):
                return _const_like(float(dict(getattr(node, "kwargs", {}) or {}).get("init", 0.0)))
            raise ValueError(f"unsupported embedded AST atom kind {kind!r}")
        if isinstance(node, ConstNode):
            return _const_like(float(node.value))
        if isinstance(node, AddNode):
            return _rec(node.left) + _rec(node.right)
        if isinstance(node, MulNode):
            return _rec(node.left) * _rec(node.right)
        if isinstance(node, PowNode):
            return _rec(node.base) ** float(node.exponent)
        if isinstance(node, SinNode):
            return torch.sin(_rec(node.arg))
        if isinstance(node, CosNode):
            return torch.cos(_rec(node.arg))
        if isinstance(node, ExpNode):
            return torch.exp(_rec(node.arg))
        if isinstance(node, LogNode):
            return torch.log(_rec(node.arg))
        if isinstance(node, AsinNode):
            return torch.asin(torch.clamp(_rec(node.arg), -1.0 + 1.0e-12, 1.0 - 1.0e-12))
        if isinstance(node, AcosNode):
            return torch.acos(torch.clamp(_rec(node.arg), -1.0 + 1.0e-12, 1.0 - 1.0e-12))
        if isinstance(node, AtanNode):
            return torch.atan(_rec(node.arg))
        if isinstance(node, AbsNode):
            return torch.abs(_rec(node.arg))
        if isinstance(node, ConjNode):
            return torch.conj(_rec(node.arg))
        if isinstance(node, RealNode):
            return torch.real(_rec(node.arg))
        if isinstance(node, ImagNode):
            return torch.imag(_rec(node.arg))
        if isinstance(node, ArgNode):
            return torch.angle(_rec(node.arg))
        raise ValueError(f"unsupported embedded AST node type {type(node).__name__}")

    return _rec(root)


def nestynet_mapping_embedding_roundtrip(
    inner_ast: Node,
    mapping: dict,
    input_exprs: Sequence[Node],
    x: torch.Tensor,
    *,
    units_mode: str = "raw",
    scale_name: Optional[str] = None,
    scale_kind: str = "fixed",
    sin_arg_mode: str = "omega_z",
) -> dict:
    """Compare scorer prediction against a mapped embedded NestyNet_SR AST.

    Returns a JSON-friendly diagnostic dictionary.  This is meant for debug and
    benchmark invariants; it does not mutate the search result.
    """

    out = {
        "ok": False,
        "max_abs_err": None,
        "max_rel_err": None,
        "error": None,
    }
    try:
        from .inverse_core import eval_mapping_total

        embedded = embed_mapping_in_ast(
            inner_ast,
            mapping,
            list(input_exprs),
            units_mode=units_mode,
            scale_name=scale_name,
            scale_kind=scale_kind,
            sin_arg_mode=sin_arg_mode,
        )
        if embedded is None:
            out["error"] = "embedding_returned_none"
            return out
        inner_values = eval_embedded_mapping_ast(inner_ast, x)
        expected = eval_mapping_total(inner_values, mapping, x=x).reshape(int(x.shape[0]), -1)
        actual = eval_embedded_mapping_ast(embedded, x).reshape(int(x.shape[0]), -1)
        if expected.shape[1] != 1:
            expected = expected[:, :1]
        if actual.shape[1] != 1:
            actual = actual[:, :1]
        mask = torch.isfinite(expected) & torch.isfinite(actual)
        if not bool(mask.any()):
            out["error"] = "no_finite_overlap"
            return out
        diff = torch.abs(actual[mask] - expected[mask])
        denom = torch.clamp(torch.abs(expected[mask]), min=1.0)
        max_abs = float(torch.max(diff).detach().cpu().item())
        max_rel = float(torch.max(diff / denom).detach().cpu().item())
        out["max_abs_err"] = max_abs
        out["max_rel_err"] = max_rel
        out["ok"] = bool(max_abs <= 1.0e-8 or max_rel <= 1.0e-8)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def mapping_embedding_roundtrip(
    toy_ast: tuple,
    mapping: dict,
    input_exprs: Sequence[Node],
    x: torch.Tensor,
    *,
    units_mode: str = "raw",
    scale_name: Optional[str] = None,
    scale_kind: str = "fixed",
    sin_arg_mode: str = "omega_z",
) -> dict:
    """Compare a factorized-search scorer prediction against its embedded AST."""

    try:
        inner_nn = factorized_search_to_nestynet(toy_ast)
    except Exception as exc:
        return {
            "ok": False,
            "max_abs_err": None,
            "max_rel_err": None,
            "error": str(exc),
        }
    return nestynet_mapping_embedding_roundtrip(
        inner_nn,
        mapping,
        input_exprs,
        x,
        units_mode=units_mode,
        scale_name=scale_name,
        scale_kind=scale_kind,
        sin_arg_mode=sin_arg_mode,
    )


def bounds_from_data(
    x: torch.Tensor,
    *,
    fallback_lo: float = 1.0,
    fallback_hi: float = 5.0,
    min_span: float = 1e-12,
) -> tuple:
    """Compute per-dimension sampling bounds from data.

    Returns (lo, hi) where each is either:
      - a torch.Tensor of shape (D,) in float64, or
      - fallback scalars (fallback_lo, fallback_hi) if the data is degenerate.
    """
    _x = x.detach().cpu().to(torch.float64)
    if _x.ndim != 2:
        return float(fallback_lo), float(fallback_hi)
    lo = _x.min(dim=0).values
    hi = _x.max(dim=0).values
    if (not torch.isfinite(lo).all()) or (not torch.isfinite(hi).all()):
        return float(fallback_lo), float(fallback_hi)
    if float((hi - lo).abs().max()) < float(min_span):
        return float(fallback_lo), float(fallback_hi)
    return lo, hi


def promote_const_to_scale(
    root: Node,
    *,
    tag_prefix: str = "factorized_search",
) -> Node:
    """Replace every ``ConstNode`` in *root* with a trainable ``Scale(init=v)``.

    Walks the NestyNet_SR AST and converts frozen ``ConstNode(v)`` leaves into
    ``Scale(name, init=v)`` atoms, giving LM per-coefficient freedom during
    refitting.  Use this on an already-built AST (e.g. after sympy
    simplification) when you want all numeric constants to be trainable.
    """
    counter = 0

    def _next_tag() -> str:
        nonlocal counter
        name = f"{tag_prefix}__const_{counter}"
        counter += 1
        return name

    def _walk(node: Node) -> Node:
        if isinstance(node, ConstNode):
            v = float(node.value)
            if math.isfinite(v):
                nm = _next_tag()
                return Scale(nm, tag=nm, init=v)
            return node  # non-finite: keep frozen
        if isinstance(node, AtomNode):
            return node
        if isinstance(node, AddNode):
            return AddNode(_walk(node.left), _walk(node.right))
        if isinstance(node, MulNode):
            return MulNode(_walk(node.left), _walk(node.right))
        if isinstance(node, PowNode):
            return PowNode(_walk(node.base), node.exponent)
        if isinstance(node, SinNode):
            return SinNode(_walk(node.arg))
        if isinstance(node, CosNode):
            return CosNode(_walk(node.arg))
        if isinstance(node, LogNode):
            return LogNode(_walk(node.arg))
        if isinstance(node, ExpNode):
            return ExpNode(_walk(node.arg))
        if isinstance(node, AbsNode):
            return AbsNode(_walk(node.arg))
        if isinstance(node, ConjNode):
            return ConjNode(_walk(node.arg))
        if isinstance(node, RealNode):
            return RealNode(_walk(node.arg))
        if isinstance(node, ImagNode):
            return ImagNode(_walk(node.arg))
        if isinstance(node, ArgNode):
            return ArgNode(_walk(node.arg))
        return node

    return _walk(root)


# ===================================================================
# AST conversion: factorized_search tuple → NestyNet_SR Node
# ===================================================================

def factorized_search_to_nestynet(node) -> "Node":
    """Convert a factorized_search tuple-AST to a NestyNet_SR Node tree."""
    op = node[0]
    if op == "var":
        return Var(node[1])
    if op == "const":
        return ConstNode(node[1])
    if op == "sin":
        return Sin(factorized_search_to_nestynet(node[1]))
    if op == "cos":
        return Cos(factorized_search_to_nestynet(node[1]))
    if op == "exp":
        return Exp(factorized_search_to_nestynet(node[1]))
    if op == "log":
        return Log(factorized_search_to_nestynet(node[1]))
    if op == "asin":
        return AsinNode(factorized_search_to_nestynet(node[1]))
    if op == "acos":
        return AcosNode(factorized_search_to_nestynet(node[1]))
    if op == "sqrt":
        return Pow(factorized_search_to_nestynet(node[1]), 0.5)
    if op == "sqr":
        return Pow(factorized_search_to_nestynet(node[1]), 2.0)
    if op == "neg":
        return Mul(ConstNode(-1), factorized_search_to_nestynet(node[1]))
    if op == "add":
        return Add(factorized_search_to_nestynet(node[1]), factorized_search_to_nestynet(node[2]))
    if op == "sub":
        return Add(factorized_search_to_nestynet(node[1]), Mul(ConstNode(-1), factorized_search_to_nestynet(node[2])))
    if op == "mul":
        return Mul(factorized_search_to_nestynet(node[1]), factorized_search_to_nestynet(node[2]))
    if op == "div":
        return Div(factorized_search_to_nestynet(node[1]), factorized_search_to_nestynet(node[2]))
    raise ValueError(f"Unknown op: {op!r}")


# ===================================================================
# AST conversion: NestyNet_SR Node → factorized_search tuple
# ===================================================================

def _is_const(node, value) -> bool:
    """Check if node is a ConstNode with the given value."""
    return isinstance(node, ConstNode) and node.value == value


def nestynet_to_factorized_search(node) -> tuple:
    """Convert a NestyNet_SR Node tree to a factorized_search tuple-AST.

    Recognises canonical patterns:
        Mul(ConstNode(-1), x)       → ("neg", x)
        Mul(a, Pow(b, -1))          → ("div", a, b)
        Add(a, Mul(ConstNode(-1),b))→ ("sub", a, b)
        Pow(x, 0.5)                 → ("sqrt", x)

    Raises ValueError for node types without a factorized_search equivalent (e.g.
    AtomNode with kind='nn').
    """
    if isinstance(node, AtomNode):
        if node.kind == "var":
            return ("var", node.var_idxs[0])
        raise ValueError(f"Cannot convert AtomNode kind={node.kind!r} to factorized_search AST")

    if isinstance(node, ConstNode):
        return ("const", node.value)

    if isinstance(node, SinNode):
        return ("sin", nestynet_to_factorized_search(node.arg))
    if isinstance(node, CosNode):
        return ("cos", nestynet_to_factorized_search(node.arg))
    if isinstance(node, ExpNode):
        return ("exp", nestynet_to_factorized_search(node.arg))
    if isinstance(node, LogNode):
        return ("log", nestynet_to_factorized_search(node.arg))

    if isinstance(node, PowNode):
        exp = float(node.exponent)
        base = nestynet_to_factorized_search(node.base)
        if exp == 0.5:
            return ("sqrt", base)
        if exp == 2.0:
            return ("sqr", base)
        # Support the small negative-power subset that maps cleanly onto the
        # existing tuple AST primitives. This is enough to unblock truth-AST
        # compilation for reciprocal and inverse-root oracle targets.
        if exp == -1.0:
            return ("div", ("const", 1.0), base)
        if exp == -0.5:
            return ("div", ("const", 1.0), ("sqrt", base))
        if exp == -2.0:
            return ("div", ("const", 1.0), ("sqr", base))
        # General integer powers (|n| up to 8): the tuple AST has no general
        # power primitive, so expand into repeated squaring/multiplication
        # (exponentiation by squaring). This unblocks warp-discovered coordinates
        # such as x0^2 + x1^3 + x2^2.
        if float(exp).is_integer() and 1 <= abs(int(exp)) <= 8:
            n = abs(int(exp))

            def _pow_int(b: tuple, k: int) -> tuple:
                result = None
                sq = b
                while k > 0:
                    if k & 1:
                        result = sq if result is None else ("mul", result, sq)
                    k >>= 1
                    if k > 0:
                        sq = ("sqr", sq)
                return result

            powered = _pow_int(base, n)
            return powered if int(exp) > 0 else ("div", ("const", 1.0), powered)
        raise ValueError(f"PowNode with exponent={node.exponent} has no direct factorized_search equivalent")

    if isinstance(node, MulNode):
        def _is_unit_reciprocal(ast: tuple) -> bool:
            return (
                isinstance(ast, tuple)
                and len(ast) == 3
                and ast[0] == "div"
                and ast[1] == ("const", 1.0)
            )

        # Mul(ConstNode(-1), x) → ("neg", x)
        if _is_const(node.left, -1):
            return ("neg", nestynet_to_factorized_search(node.right))
        # Mul(a, Pow(b, -1)) → ("div", a, b)
        if isinstance(node.right, PowNode) and node.right.exponent == -1:
            return ("div", nestynet_to_factorized_search(node.left), nestynet_to_factorized_search(node.right.base))
        left = nestynet_to_factorized_search(node.left)
        right = nestynet_to_factorized_search(node.right)
        # Canonicalize products with reciprocal factors back into an explicit
        # quotient form. This keeps truth ASTs for rational / inverse-root
        # targets in the cleaner shape expected by downstream evaluators.
        if _is_unit_reciprocal(right):
            return ("div", left, right[2])
        if _is_unit_reciprocal(left):
            return ("div", right, left[2])
        return ("mul", left, right)

    if isinstance(node, AddNode):
        # Add(a, Mul(ConstNode(-1), b)) → ("sub", a, b)
        if isinstance(node.right, MulNode) and _is_const(node.right.left, -1):
            return ("sub", nestynet_to_factorized_search(node.left), nestynet_to_factorized_search(node.right.right))
        return ("add", nestynet_to_factorized_search(node.left), nestynet_to_factorized_search(node.right))

    raise ValueError(f"Unsupported NestyNet_SR node type: {type(node).__name__}")


# ===================================================================
# Dimensional-analysis interop
# ===================================================================

def dims_to_fraction(dims: Sequence[float]) -> Dim:
    """Convert a float-exponent tuple to a Fraction-exponent Dim."""
    return tuple(Fraction(x).limit_denominator(128) for x in dims)


def fraction_to_dims(dim: Dim) -> Tuple[float, ...]:
    """Convert a Fraction-exponent Dim to a float tuple."""
    return tuple(float(x) for x in dim)


def dims_to_units_spec(
    var_dims: Sequence[Sequence[float]],
    y_dims: Sequence[float],
) -> UnitsSpec:
    """Build a UnitsSpec from float-format dimensions.

    The unit system basis length is inferred from the dimension vectors.
    """
    n_base = len(y_dims)
    # Build a minimal basis with generic names
    base_names = ("L", "M", "T", "I", "Θ", "N", "J")[:n_base]
    if n_base > 7:
        base_names = tuple(f"D{i}" for i in range(n_base))
    us = UnitSystem(base=base_names)
    x_dims_frac = tuple(dims_to_fraction(d) for d in var_dims)
    y_dim_frac = dims_to_fraction(y_dims)
    return UnitsSpec(unit_system=us, x_dims=x_dims_frac, y_dim=y_dim_frac)


# ===================================================================
# Sympy simplification helpers
# ===================================================================

_log = logging.getLogger(__name__)


def _tuple_ast_to_sympy(node, syms):
    """Convert a factorized symbolic search tuple-AST to a sympy expression.

    Parameters
    ----------
    node : tuple
        factorized symbolic search tuple-AST node.
    syms : list of sympy.Symbol
        Sympy symbols for each variable index.
    """
    import sympy as sp

    op = node[0]
    if op == "var":
        return syms[node[1]]
    if op == "const":
        # Use Rational for exact integer/half-integer constants
        v = node[1]
        if v == int(v):
            return sp.Integer(int(v))
        return sp.Float(v)
    if op == "sin":
        return sp.sin(_tuple_ast_to_sympy(node[1], syms))
    if op == "cos":
        return sp.cos(_tuple_ast_to_sympy(node[1], syms))
    if op == "exp":
        return sp.exp(_tuple_ast_to_sympy(node[1], syms))
    if op == "log":
        return sp.log(_tuple_ast_to_sympy(node[1], syms))
    if op == "asin":
        return sp.asin(_tuple_ast_to_sympy(node[1], syms))
    if op == "acos":
        return sp.acos(_tuple_ast_to_sympy(node[1], syms))
    if op == "sqrt":
        return sp.sqrt(_tuple_ast_to_sympy(node[1], syms))
    if op == "sqr":
        inner = _tuple_ast_to_sympy(node[1], syms)
        return inner ** 2
    if op == "neg":
        return -_tuple_ast_to_sympy(node[1], syms)
    if op == "add":
        return _tuple_ast_to_sympy(node[1], syms) + _tuple_ast_to_sympy(node[2], syms)
    if op == "sub":
        return _tuple_ast_to_sympy(node[1], syms) - _tuple_ast_to_sympy(node[2], syms)
    if op == "mul":
        return _tuple_ast_to_sympy(node[1], syms) * _tuple_ast_to_sympy(node[2], syms)
    if op == "div":
        return _tuple_ast_to_sympy(node[1], syms) / _tuple_ast_to_sympy(node[2], syms)
    raise ValueError(f"Unknown op: {op}")


def _prune_poly_degree(pred, y, max_degree, mse_tol_factor=2.0):
    """Find the lowest poly degree whose MSE is within *mse_tol_factor* of the best.

    Fits poly at degrees 1 through *max_degree* (each is one lstsq call) and
    returns (best_degree, best_mse, best_mapping) where *best_degree* is the
    lowest degree whose MSE ≤ mse_tol_factor * global_best_mse.

    Returns None if no finite fit is found.
    """
    from .expr_mapping import eval_poly, fit_poly, mean_squared_error_same_shape

    fits = []  # (degree, mse, mapping_dict)
    for deg in range(1, max_degree + 1):
        pf = fit_poly(pred, y, deg)
        if pf is None:
            continue
        coeffs, mu, std = pf
        y_hat = eval_poly(pred, coeffs, mu, std)
        mse = mean_squared_error_same_shape(y, y_hat)
        if not math.isfinite(mse):
            continue
        fits.append((deg, mse, {"kind": "poly", "coeffs": coeffs, "mu": mu, "std": std}))

    if not fits:
        return None

    global_best_mse = min(f[1] for f in fits)
    threshold = mse_tol_factor * global_best_mse if global_best_mse > 0 else 1e-30

    # Return lowest degree within tolerance
    for deg, mse, mapping in fits:
        if mse <= threshold:
            _log.info(
                "poly pruning: degree %d → %d  (mse %.4g vs best %.4g, tol factor %.1f)",
                max_degree, deg, mse, global_best_mse, mse_tol_factor,
            )
            return deg, mse, mapping

    # Fallback: return the best MSE fit
    best = min(fits, key=lambda f: f[1])
    return best


def _simplify_and_refit(result, x_probe, y_probe, nvars, poly_degree):
    """Sympy-simplify a factorized symbolic search skeleton and refit the mapping.

    Returns an updated result dict on success, or None to keep the original.
    """
    import sympy as sp

    toy_ast = result["toy_ast"]

    # Build sympy symbols with positivity hints from the data
    x_min = x_probe.min(dim=0).values
    syms = []
    for i in range(nvars):
        positive = bool(x_min[i].item() > 0)
        syms.append(sp.Symbol(f"x{i}", positive=positive))

    # Convert tuple-AST to sympy (this already simplifies via sympy arithmetic)
    # then apply sp.simplify() for trig identities and other non-trivial cases.
    try:
        sp_expr = _tuple_ast_to_sympy(toy_ast, syms)
        simplified = sp.simplify(sp_expr, ratio=1.5)
    except Exception:
        return None

    # Quick check: if the sympy string matches the original, no gain
    simplified_str = str(simplified)
    if simplified_str == result["expr"]:
        return None

    # Lambdify and evaluate on probe data
    import warnings
    try:
        f_np = sp.lambdify(syms, simplified, modules="numpy")
        x_np = x_probe.cpu().numpy()
        x_cols = [x_np[:, i] for i in range(nvars)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
            pred_np = f_np(*x_cols)
            # Handle scalar result (expression simplified to a constant)
            pred_np = np.broadcast_to(
                np.atleast_1d(np.asarray(pred_np, dtype=np.float64)),
                (x_np.shape[0],),
            )
        pred = torch.tensor(pred_np, dtype=x_probe.dtype, device=x_probe.device).unsqueeze(-1)
    except Exception:
        return None

    if not torch.isfinite(pred).all():
        return None

    # Refit mapping on probe data
    fb = fit_best(pred, y_probe, poly_degree)
    if fb is None:
        return None
    new_mse, new_mapping = fb

    # Try to prune poly degree to the lowest adequate level
    if new_mapping.get("kind") == "poly" and poly_degree > 1:
        pruned = _prune_poly_degree(pred, y_probe, poly_degree)
        if pruned is not None:
            p_deg, p_mse, p_mapping = pruned
            if p_deg < poly_degree:
                new_mse, new_mapping = p_mse, p_mapping

    # Accept only if not much worse than original (within 10x), in raw-MSE terms.
    orig_mse = float(result.get("mse_raw", result.get("mse", float("inf"))))
    if orig_mse > 0 and new_mse > orig_mse * 10:
        return None

    # Convert simplified sympy → NestyNet_SR Node
    try:
        nn_ast = sympy_to_nestynet(simplified, nvars)
    except Exception:
        return None

    # Build updated result dict
    updated = dict(result)
    updated["expr"] = str(simplified)
    updated["nestynet_ast"] = nn_ast
    updated["mse"] = float(new_mse)
    updated["mse_raw"] = float(new_mse)
    updated["mse_eff"] = float(new_mse)
    updated["mapping"] = new_mapping
    updated["simplified"] = True
    if new_mapping["kind"] == "poly":
        updated["coeffs"] = new_mapping["coeffs"]
        updated["mu"] = new_mapping["mu"]
        updated["std"] = new_mapping["std"]
    else:
        updated["coeffs"] = None
        updated["mu"] = None
        updated["std"] = None

    _log.info(
        "simplified skeleton: %s  →  %s  (mse %.4g → %.4g)",
        result["expr"], updated["expr"], orig_mse, new_mse,
    )
    return updated


# ===================================================================
# Programmatic explorer API
# ===================================================================

def run_explorer(
    x_data=None,
    y_data=None,
    target_fn=None,
    nvars=None,
    n_iter=20000,
    max_depth=6,
    poly_degree=4,
    lo=1.0,
    hi=5.0,
    seed=0,
    var_dims=None,
    y_dims=None,
    return_topk=10,
    dtype=torch.float64,
    x_fit_data=None,
    y_fit_data=None,
    x_probe_data=None,
    y_probe_data=None,
    simplify_skeletons=True,
    validate_embedded_mapping=False,
    embedding_roundtrip_raise=False,
    **kwargs,
) -> List[dict]:
    """Run the factorized symbolic search explorer and return top results with NestyNet_SR ASTs.

    Either supply ``target_fn`` + ``nvars``, or ``x_data`` + ``y_data``,
    or pre-split ``x_fit_data``/``y_fit_data``/``x_probe_data``/``y_probe_data``.

    Parameters
    ----------
    x_data, y_data : torch.Tensor, optional
        Explicit data tensors.  If provided, ``target_fn`` is built from them
        via nearest-neighbour interpolation on the sample grid.
    target_fn : callable, optional
        Maps (N, nvars) tensor → (N, 1) tensor.
    nvars : int
        Number of input variables (required if target_fn is used without
        pre-built data).
    x_fit_data, y_fit_data, x_probe_data, y_probe_data : torch.Tensor, optional
        Pre-split fit/probe data.  When all four are provided, the explorer
        uses them directly (no random sampling, no target_fn calls).
    var_dims : list of tuples, optional
        Dimensional exponents per variable (float tuples).
    y_dims : tuple, optional
        Dimensional exponents for the target.
    return_topk : int
        Number of top results to return.
    simplify_skeletons : bool
        If True (default), sympy-simplify each skeleton and refit the mapping.
    validate_embedded_mapping : bool
        If True, attach a numerical scorer-vs-embedded-AST roundtrip diagnostic
        to each returned candidate.
    embedding_roundtrip_raise : bool
        If True with ``validate_embedded_mapping``, raise if any diagnostic
        fails.
    kwargs
        Forwarded to ``run_explorer_core``.

    Returns
    -------
    list of dict
        Each dict has keys: expr (str), toy_ast (tuple), mse (float),
        nestynet_ast (Node), size (int).
    """
    has_prebuilt = (x_fit_data is not None and y_fit_data is not None
                    and x_probe_data is not None and y_probe_data is not None)

    # Joint multi-dataset scoring uses per-dataset affine maps; the sympy simplify+refit pass
    # currently only supports the pooled single-dataset scoring path, so disable it automatically.
    if simplify_skeletons and bool(kwargs.get("refine_joint_score_enable", False)):
        simplify_skeletons = False

    if has_prebuilt:
        # Pre-built data path: infer nvars, target_fn not needed
        if nvars is None:
            nvars = x_fit_data.shape[1]
    elif target_fn is None:
        # Legacy data path: build target_fn via nearest-neighbour
        if x_data is None or y_data is None:
            raise ValueError("Provide either target_fn+nvars, x_data+y_data, "
                             "or x_fit_data/y_fit_data/x_probe_data/y_probe_data")
        nvars = x_data.shape[1]
        _x = x_data.clone()
        _y = y_data.clone()
        def target_fn(x, _x=_x, _y=_y):
            # Nearest-neighbour lookup
            dists = torch.cdist(x, _x)
            idx = dists.argmin(dim=1)
            return _y[idx] if _y.dim() == 2 else _y[idx].unsqueeze(-1)
    elif nvars is None:
        raise ValueError("nvars is required when using target_fn")

    run_started = time.perf_counter()
    arch = run_explorer_core(
        target_fn, nvars,
        n_iter=n_iter, max_depth=max_depth, poly_degree=poly_degree,
        lo=lo, hi=hi, seed=seed,
        var_dims=var_dims, y_dims=y_dims,
        dtype=dtype,
        x_fit_data=x_fit_data, y_fit_data=y_fit_data,
        x_probe_data=x_probe_data, y_probe_data=y_probe_data,
        _runtime_hooks=make_engine_runtime_hooks(),
        **kwargs,
    )
    run_wall_s = float(time.perf_counter() - run_started)

    explorer_diagnostics = getattr(arch, "refine_diagnostics", None)
    if isinstance(explorer_diagnostics, dict):
        explorer_diagnostics = dict(explorer_diagnostics)
        explorer_diagnostics["run_explorer_wall_s"] = float(run_wall_s)
        explorer_diagnostics["wall_seconds"] = float(run_wall_s)
        explorer_diagnostics["search_stop_reason"] = str(getattr(arch, "search_stop_reason", "") or "")
        explorer_diagnostics["search_wall_time_elapsed_s"] = float(
            getattr(arch, "search_wall_time_elapsed_s", run_wall_s) or run_wall_s
        )
        if "fit_poly_s" not in explorer_diagnostics and "fit_poly_wall_seconds" in explorer_diagnostics:
            explorer_diagnostics["fit_poly_s"] = float(explorer_diagnostics.get("fit_poly_wall_seconds", 0.0) or 0.0)
        if "base_score_s" in explorer_diagnostics:
            explorer_diagnostics["base_score_s"] = float(explorer_diagnostics.get("base_score_s", 0.0) or 0.0)
        phase_diag = getattr(arch, "search_phase_diagnostics", None)
        if isinstance(phase_diag, dict):
            explorer_diagnostics["phase_diagnostics"] = dict(phase_diag)
        expr_ir_report = getattr(arch, "expr_ir_report", None)
        if isinstance(expr_ir_report, dict):
            explorer_diagnostics["expr_ir"] = dict(expr_ir_report)
        gs_fss_report = getattr(arch, "gs_fss_report", None)
        if isinstance(gs_fss_report, dict):
            explorer_diagnostics["gs_fss"] = dict(gs_fss_report)
    else:
        explorer_diagnostics = None

    results = []
    for rec in arch.best(return_topk, strategy="mse_decade_size"):
        toy_ast = rec.best_expr
        try:
            nn_ast = factorized_search_to_nestynet(toy_ast)
        except ValueError:
            nn_ast = None
        m = rec.mapping
        # Backward compat: extract coeffs/mu/std when poly, None otherwise
        if m["kind"] == "poly":
            coeffs, mu, std = m["coeffs"], m["mu"], m["std"]
        else:
            coeffs, mu, std = None, None, None
        mse_raw = float(getattr(rec, "best_raw_mse", rec.best_mse))
        mse_eff = float(rec.best_mse)
        row = {
            "expr": node_str(toy_ast),
            "toy_ast": toy_ast,
            "mse": mse_raw,  # keep legacy key as raw MSE for solved checks
            "mse_raw": mse_raw,
            "mse_eff": mse_eff,
            "nestynet_ast": nn_ast,
            "size": node_size(toy_ast),
            "mapping": m,
            "coeffs": coeffs,
            "mu": mu,
            "std": std,
        }
        if explorer_diagnostics is not None:
            row["explorer_diagnostics"] = dict(explorer_diagnostics)
        results.append(row)

    # --- Sympy simplification pass ---
    if simplify_skeletons and results:
        x_probe = arch.x_probe
        y_probe = arch.y_probe
        for i, res in enumerate(results):
            try:
                updated = _simplify_and_refit(
                    res, x_probe, y_probe, nvars, poly_degree,
                )
            except Exception:
                updated = None
            if updated is not None:
                results[i] = updated

    if validate_embedded_mapping and results:
        x_check = getattr(arch, "x_probe", None)
        if torch.is_tensor(x_check):
            input_exprs = [Var(i) for i in range(int(nvars))]
            for res in results:
                inner_ast = res.get("nestynet_ast")
                if inner_ast is None:
                    diag = {
                        "ok": False,
                        "max_abs_err": None,
                        "max_rel_err": None,
                        "error": "missing_nestynet_ast",
                    }
                else:
                    diag = nestynet_mapping_embedding_roundtrip(
                        inner_ast,
                        res.get("mapping", {}),
                        input_exprs,
                        x_check,
                        units_mode="raw",
                    )
                res["embedding_roundtrip"] = diag
                if embedding_roundtrip_raise and not bool(diag.get("ok", False)):
                    raise AssertionError(
                        f"embedded mapping roundtrip failed for {res.get('expr', '<unknown>')}: {diag}"
                    )

    # --- Post-hoc re-ranking: prefer simpler expressions at similar MSE ---
    # Sort by (log-binned MSE, node_size) so that within the same MSE decade,
    # simpler expressions come first.
    if results:
        def _rank_key(r):
            mse = max(float(r.get("mse_raw", r.get("mse", 1e100))), 1e-100)
            log_bin = int(math.floor(math.log10(mse)))  # decade bin
            return (log_bin, r["size"])
        results.sort(key=_rank_key)

    return results
