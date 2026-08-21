# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Centralised wrapper policy logic.

This module centralises *policy* decisions about when to try simple wrappers
(rational / trig / power/shape) around a base analytic expression.

Historically, Stage A (compound-variable proposals) embedded kind-aware wrapper
logic directly inside sr_search.search. Stage B macro/template logic introduced
its own wrapper-related gates, which risked semantic drift over time.

The intent here is to keep the *enumeration* (feature_grammar.build_*_wrappers)
and the *policy* (what to enable, how to prefer raw vs wrapped) in shared,
testable helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nestynet_sr.sr_core.bridges import CosNode, Node, SinNode

from .feature_grammar import OMEGA_SNAP_CANDS, build_wrapper_variants, snap_to_scales


def _compound_exponents_ratio_like(exponents: Sequence[int]) -> bool:
    """Heuristic: treat a compound monomial as ratio-like if it mixes +/- exponents."""
    try:
        exps = [float(e) for e in (exponents or ())]
    except Exception:
        return False
    if not exps:
        return False
    has_pos = any(e > 0 for e in exps)
    has_neg = any(e < 0 for e in exps)
    return bool(has_pos and has_neg)


def snap_omega(omega: Optional[float]) -> float:
    """Snap omega to a simple nearby value when it is clearly close."""
    if omega is None:
        return 1.0
    try:
        w = float(omega)
    except Exception:
        return float(omega) if omega is not None else 1.0
    try:
        return snap_to_scales(w, OMEGA_SNAP_CANDS, rel_tol=0.25, abs_tol=0.25)
    except Exception:
        return w

def _ast_contains_trig(expr: Node) -> bool:
    """Return True if the AST contains a SinNode/CosNode anywhere."""
    if expr is None:
        return False
    seen = set()
    stack = [expr]
    while stack:
        n = stack.pop()
        if n is None:
            continue
        nid = id(n)
        if nid in seen:
            continue
        seen.add(nid)
        if isinstance(n, (SinNode, CosNode)):
            return True

        # Common child fields across bridge nodes
        for attr in ("left", "right", "arg", "base", "expr"):
            c = getattr(n, attr, None)
            if c is None:
                continue
            if isinstance(c, (list, tuple)):
                for cc in c:
                    if cc is not None:
                        stack.append(cc)
            else:
                stack.append(c)

        # Some nodes embed expressions inside kwargs (e.g. AtomNode input_expr)
        kw = getattr(n, "kwargs", None)
        if isinstance(kw, dict):
            for v in kw.values():
                if isinstance(v, Node):
                    stack.append(v)
                elif isinstance(v, (list, tuple)):
                    for vv in v:
                        if isinstance(vv, Node):
                            stack.append(vv)
    return False



@dataclass(frozen=True)
class CompoundZWrapperPolicy:
    # Wrapper family toggles
    square: bool = False
    abs_: bool = False
    sqrt: bool = False
    rational: bool = False
    rational_flavor: str = "default"
    trig: bool = False
    trig_omega: float = 1.0
    trig_fn: Optional[str] = None  # None = try both sin+cos, "sin"/"cos" = only that one

    # Selection preferences (used when multiple wrapper variants are accepted)
    wrapper_prefer_factor: float = 0.1
    sqrt_prefer_factor: float = 0.5
    square_prefer_factor: float = 0.3
    abs_prefer_factor: float = 0.3
    trig_prefer_factor: float = 0.01


def compound_z_wrapper_policy(
    *,
    kind: str,
    pattern: Sequence[int],
    meta: Optional[Dict[str, Any]],
    search_hp: Any,
    trig_spec: Any = None,
    atom_var_idxs: Optional[Sequence[int]] = None,
    leaf_features: Any = None,
    oracle_trig_specs: Optional[List[Any]] = None,
) -> CompoundZWrapperPolicy:
    """Decide which wrappers to try around a proposed compound coordinate z.

    This encapsulates Stage-A's kind-aware wrapper choices.

    Parameters
    ----------
    leaf_features : LeafFeatures, optional
        Feature detection results from the compound leaf's input space.
        If provided, trig_by_axis[0] refers to the compound variable z itself.
        This allows detecting trig structure on z = x*y even if neither x nor y
        alone shows trig behaviour.
    oracle_trig_specs : list of TrigScaleSpec, optional
        Oracle trig probe results (local axis indices). When available,
        narrows sin/cos selection to the oracle-identified function.
    """

    meta = meta or {}
    kind = str(kind or "compound").lower()

    # GS promoted reductions carry their coordinate family in meta; remap so
    # they receive the same kind-aware wrapper variants as the legacy
    # detector proposals they subsume (sqrt for radial, rational for
    # ratio-like monomials, ...). Only ever active for GS-emitted proposals.
    if kind == "gs_promoted_reduction":
        _gs_kind_remap = {
            "monomial": "monomial",
            "monomial_of_linear": "monomial",
            "monomial_of_virtual": "monomial",
            "radial": "radial",
            "quadratic_radius": "radial",
            "linear_projection": "linear",
            "translation_invariant_linear": "linear",
        }
        _gs_coordinate_kind = str(meta.get("gs_coordinate_kind", "")).lower()
        kind = _gs_kind_remap.get(_gs_coordinate_kind, kind)
        if _gs_coordinate_kind == "quadratic_form" and bool(meta.get("allow_sqrt", False)):
            # Definite composed quadratic forms (e.g. Euclidean distance
            # carriers over virtual axes) take the radial wrapper family.
            kind = "radial"
        if kind == "monomial" and (meta.get("gs_monomial_exponents") or meta.get("gs_virtual_exponents")):
            # GS patterns are 0/1 support masks; the ratio-like check below
            # needs the true (coordinate-level) exponents.
            pattern = tuple(meta.get("gs_monomial_exponents") or meta.get("gs_virtual_exponents"))

    # Global wrapper toggles
    enable_trig_wrappers = bool(getattr(search_hp, "compound_try_trig_wrappers", True))
    enable_rational_wrappers = bool(getattr(search_hp, "compound_try_rational_wrappers", True))
    enable_square_wrappers = bool(getattr(search_hp, "compound_try_square_wrappers", True))
    enable_abs_wrappers = bool(getattr(search_hp, "compound_try_abs_wrappers", True))
    rational_only_if_ratio_like = bool(getattr(search_hp, "compound_rational_only_if_ratio_like", True))
    ratio_like = bool(kind == "monomial" and _compound_exponents_ratio_like(pattern))

    # Kind-aware wrapper family enables
    square = False
    abs_ = False
    sqrt = False
    rational = False
    rational_flavor = "default"
    trig = False
    trig_omega = 1.0

    if kind == "radial":
        # For radial compounds we propose r^2 as the base z; optionally also
        # try r = sqrt(r^2).
        if str(meta.get("form", "")) == "r2" and bool(meta.get("allow_sqrt", False)):
            sqrt = True
        if enable_rational_wrappers and bool(getattr(search_hp, "compound_rational_allow_for_radial", True)):
            rational = True
            rational_flavor = "radial"

    elif kind in ("linear", "shift"):
        square = bool(enable_square_wrappers)
        abs_ = bool(enable_abs_wrappers)

    elif kind == "monomial":
        if enable_rational_wrappers and ((not rational_only_if_ratio_like) or ratio_like):
            rational = True
            rational_flavor = "default"

    # Trig wrappers: check both axis-level and leaf-level features.
    # 1) Axis-level: a detected trig axis participates in z.
    if enable_trig_wrappers and (trig_spec is not None) and atom_var_idxs is not None:
        try:
            trig_axis = int(getattr(trig_spec, "axis", -1))
            if trig_axis in [int(j) for j in atom_var_idxs]:
                pos = list(atom_var_idxs).index(trig_axis)
                if pos < len(pattern) and int(pattern[pos]) != 0:
                    omega_raw = float(getattr(trig_spec, "omega", 1.0))
                    trig_omega = snap_omega(omega_raw)
                    trig = True
        except Exception:
            pass

    # 2) Leaf-level / compound-z: oracle trig probe detected trig on z (axis 0).
    # This catches cases like f(x,y) = sin(x*y) where neither x nor y is trig,
    # but z = x*y is.
    if enable_trig_wrappers and (not trig) and oracle_trig_specs:
        try:
            for ospec in oracle_trig_specs:
                if int(getattr(ospec, "axis", -1)) == 0:
                    omega_raw = float(getattr(ospec, "omega", 1.0))
                    trig_omega = snap_omega(omega_raw)
                    trig = True
                    break
        except Exception:
            pass

    # Preference factors (selection policy)
    wrapper_prefer_factor = float(getattr(search_hp, "compound_wrapper_prefer_factor", 0.1))
    sqrt_prefer_factor = float(getattr(search_hp, "compound_sqrt_wrapper_prefer_factor", wrapper_prefer_factor))
    square_prefer_factor = float(getattr(search_hp, "compound_square_wrapper_prefer_factor", wrapper_prefer_factor))
    abs_prefer_factor = float(getattr(search_hp, "compound_abs_wrapper_prefer_factor", wrapper_prefer_factor))
    trig_prefer_factor = float(getattr(search_hp, "compound_trig_wrapper_prefer_factor", 0.01))

    # Oracle trig narrowing: if we know sin vs cos from oracle, set trig_fn
    trig_fn: Optional[str] = None
    if trig and oracle_trig_specs:
        # Look for any oracle spec on the compound variable's axes
        # axis 0 in leaf-local space = compound z itself
        for ospec in oracle_trig_specs:
            if int(getattr(ospec, "axis", -1)) == 0:
                trig_fn = str(getattr(ospec, "trig_fn", ""))
                if trig_fn not in ("sin", "cos"):
                    trig_fn = None
                break

    return CompoundZWrapperPolicy(
        square=square,
        abs_=abs_,
        sqrt=sqrt,
        rational=rational,
        rational_flavor=str(rational_flavor),
        trig=trig,
        trig_omega=float(trig_omega),
        trig_fn=trig_fn,
        wrapper_prefer_factor=float(wrapper_prefer_factor),
        sqrt_prefer_factor=float(sqrt_prefer_factor),
        square_prefer_factor=float(square_prefer_factor),
        abs_prefer_factor=float(abs_prefer_factor),
        trig_prefer_factor=float(trig_prefer_factor),
    )


def build_compound_z_variants(
    z_ast: Node,
    *,
    kind: str,
    pattern: Sequence[int],
    meta: Optional[Dict[str, Any]],
    search_hp: Any,
    trig_spec: Any = None,
    atom_var_idxs: Optional[Sequence[int]] = None,
    leaf_features: Any = None,
    oracle_trig_specs: Optional[List[Any]] = None,
) -> List[Tuple[str, Node]]:
    """Return ordered (name, expr) variants for a base compound coordinate z."""

    policy = compound_z_wrapper_policy(
        kind=str(kind),
        pattern=pattern,
        meta=meta,
        search_hp=search_hp,
        trig_spec=trig_spec,
        atom_var_idxs=atom_var_idxs,
        leaf_features=leaf_features,
        oracle_trig_specs=oracle_trig_specs,
    )

    variants: List[Tuple[str, Node]] = [("z", z_ast)]

    # Wrapper families are generated in a fixed order (shape -> rational -> trig)
    # to match historical behaviour.
    try:
        trig_allowed = bool(policy.trig)
        # If z already contains trig internally (e.g. z = x*sin(y)), wrapping again
        # as sin(z) / cos(z) is almost always wasted work in Stage A.
        if trig_allowed and _ast_contains_trig(z_ast):
            trig_allowed = False

        # Narrow sin/cos from oracle when available
        include_sin = True
        include_cos = True
        if policy.trig_fn == "sin":
            include_cos = False
        elif policy.trig_fn == "cos":
            include_sin = False

        wrapped = build_wrapper_variants(
            z_ast,
            square=policy.square,
            abs_=policy.abs_,
            sqrt=policy.sqrt,
            rational=policy.rational,
            rational_flavor=str(policy.rational_flavor),
            trig=trig_allowed,
            omega=float(policy.trig_omega),
            include_sin=include_sin,
            include_cos=include_cos,
        )
    except Exception:
        wrapped = []

    seen_names = {"z"}
    for nm, ex in wrapped:
        if nm in seen_names:
            continue
        seen_names.add(nm)
        variants.append((nm, ex))
    return variants


def should_select_compound_variant(
    best_variant: Optional[Dict[str, Any]],
    *,
    z_name: str,
    val_loss: float,
    enables_sep: bool,
    policy: CompoundZWrapperPolicy,
) -> bool:
    """Selection policy for the best accepted wrapper variant.

    Mirrors the historical Stage-A logic:
      1) Prefer the first separability-enabling variant over non-sep.
      2) Prefer raw z over wrapped variants when both enable separability.
      3) Otherwise, apply wrapper-specific improvement thresholds.
    """

    if best_variant is None:
        return True

    best_enables_sep = bool(best_variant.get("enables_sep", False))
    best_loss = float(best_variant.get("val_loss", float("inf")))
    best_name = str(best_variant.get("z_name", ""))

    if enables_sep:
        if not best_enables_sep:
            return True
        # Both enable sep
        if z_name == "z" and best_name != "z":
            return True
        if best_name == "z" and z_name != "z":
            return False
        return float(val_loss) < best_loss

    # Candidate doesn't enable sep
    if best_enables_sep:
        return False

    effective_threshold = best_loss

    # Any wrapper must improve significantly before replacing raw z.
    if best_name == "z" and z_name != "z":
        factor = float(policy.wrapper_prefer_factor)
        if z_name == "sqrt":
            factor = float(policy.sqrt_prefer_factor)
        elif z_name == "sq":
            factor = float(policy.square_prefer_factor)
        elif z_name == "abs":
            factor = float(policy.abs_prefer_factor)
        effective_threshold = factor * best_loss

    # Trig wrappers must beat the best non-trig variant by a stronger factor.
    if (best_name not in ("sin", "cos")) and (z_name in ("sin", "cos")):
        effective_threshold = min(effective_threshold, float(policy.trig_prefer_factor) * best_loss)

    return float(val_loss) < float(effective_threshold)


@dataclass(frozen=True)
class MacroArgWrapperPolicy:
    """Policy controlling wrapper enrichment for Stage-B macro argument pools.

    Stage B macro proposals (compound_functions.py) build a small pool of
    candidate argument expressions. We allow a small amount of wrapper
    enrichment (e.g. sin/cos of cheap phases), but want to keep this bounded
    and consistent with Stage-A wrapper policy.

    The most important gate is typically whether to include *squared* trig
    wrappers (sin^2, cos^2) in the argument pool. These can be very useful
    when trig structure is already hinted, but they otherwise increase the
    search branching factor.
    """

    trig: bool = True
    trig_squares: bool = True
    trig_max_bases: int = 16


def macro_arg_wrapper_policy(ctx: Any, macro_hp: Any, target: Any, *, leaf_features: Any = None) -> MacroArgWrapperPolicy:
    """Decide Stage-B arg-pool wrapper policy for a target leaf.

    By default we keep sin/cos wrappers enabled, but only include sin^2/cos^2
    wrappers when a trig hint exists on at least one participating axis.

    The policy can be controlled by LMHyperparams fields:
      - macro_arg_trig_enable
      - macro_arg_trig_squares_enable
      - macro_arg_trig_sq_requires_hint
      - macro_arg_trig_max_bases

    Parameters
    ----------
    leaf_features : LeafFeatures, optional
        Feature detection results from the leaf's input space. If provided,
        trig_by_axis[0] for compound atoms refers to the compound variable z.
    """

    trig_enable = bool(getattr(macro_hp, 'macro_arg_trig_enable', True))
    trig_sq_enable = bool(getattr(macro_hp, 'macro_arg_trig_squares_enable', True))
    trig_sq_requires_hint = bool(getattr(macro_hp, 'macro_arg_trig_sq_requires_hint', True))
    try:
        trig_max_bases = int(getattr(macro_hp, 'macro_arg_trig_max_bases', 16))
    except Exception:
        trig_max_bases = 16

    has_trig_hint = False

    # Check axis-level trig hints from ctx
    try:
        trig_by_axis = getattr(ctx, 'trig_by_axis', {}) or {}
        for j in getattr(target, 'var_idxs', ()) or ():
            if trig_by_axis.get(int(j)) is not None:
                has_trig_hint = True
                break
    except Exception:
        pass

    # Check leaf-level trig features (axis 0 = compound variable z)
    if (not has_trig_hint) and (leaf_features is not None):
        try:
            leaf_trig_by_axis = getattr(leaf_features, 'trig_by_axis', None) or {}
            if leaf_trig_by_axis.get(0) is not None:
                has_trig_hint = True
        except Exception:
            pass

    trig_squares = bool(trig_sq_enable and ((not trig_sq_requires_hint) or has_trig_hint))
    if not trig_enable:
        trig_squares = False

    return MacroArgWrapperPolicy(trig=bool(trig_enable), trig_squares=bool(trig_squares), trig_max_bases=int(trig_max_bases))
