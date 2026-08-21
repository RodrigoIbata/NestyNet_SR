# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Stage-A shadow coordinates, common utilities, and fit-link helpers."""

from typing import TYPE_CHECKING
import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from nestynet_sr.sr_core import ast_to_human_readable, collect_nn_atoms
from nestynet_sr.sr_core.bridges import AcosNode, AddNode, AsinNode, AtanNode, AtomNode, ConstNode, CosNode, ExpNode, LogNode, MulNode, Node, PowNode, SinNode, _collect_var_idxs_from_node, ast_equals, clone_ast, effective_arity, get_input_exprs, has_nontrivial_input, is_trivial_input
from .features import TrigAxisSpec
from .shadow_coordinates import ShadowCoordinate, ShadowRegistry, shadow_parent_key
from .wrapper_policy import snap_omega


if TYPE_CHECKING:
    from ._search_proposals import (
        _atom_compound_cols,
        _compound_candidate_default_extra_var_idxs,
        _compound_pattern_entry_is_zero,
        _is_compound_token,
    )



_STAGEA_UNARY_AST_NODES = (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode)


def _stageA_classify_iso_z_result(
    *,
    ratio: float,
    y_scale: float,
    noise_floor_screen: float,
    clean_threshold: float = 0.03,
    noise_mult: float = 2.0,
    noise_cap: float = 0.25,
    struct_margin: float = 0.01,
    confidence: Optional[float] = None,
    min_confidence: float = 0.75,
) -> Dict[str, Any]:
    """Classify a monomial iso-z certificate in a noise-aware way.

    ``ratio`` and ``y_scale`` must be measured in the same screen space used by
    the iso-z statistic.  Clean passes keep the old certificate semantics.  In
    noisy runs, an above-threshold observed ratio can become an uncertified
    proposal only when the excess structural residual is compatible with the
    expected label/model noise.
    """

    def _finite_nonneg(v, default: float = 0.0) -> float:
        try:
            out = float(v)
        except Exception:
            return float(default)
        if not math.isfinite(out) or out < 0.0:
            return float(default)
        return float(out)

    r = _finite_nonneg(ratio)
    theta0 = max(0.0, _finite_nonneg(clean_threshold, 0.03))
    if r <= theta0:
        return {
            "status": "certified",
            "decision": "allow",
            "iso_z_clean_certified": True,
            "iso_z_noise_compatible": True,
            "iso_z_uncertified": False,
            "iso_z_ratio": float(r),
            "iso_z_threshold": float(theta0),
            "iso_z_reject_reason": None,
        }

    scale = _finite_nonneg(y_scale)
    nf = _finite_nonneg(noise_floor_screen)
    if scale <= 1e-30 or nf <= 0.0:
        return {
            "status": "reject",
            "decision": "reject",
            "iso_z_clean_certified": False,
            "iso_z_noise_compatible": False,
            "iso_z_uncertified": False,
            "iso_z_ratio": float(r),
            "iso_z_threshold": float(theta0),
            "iso_z_reject_reason": "clean_threshold_failed_without_noise_floor",
        }

    rho = math.sqrt(max(0.0, nf)) / max(scale, 1.0e-30)
    r_struct = math.sqrt(max(0.0, r * r - rho * rho))
    mult = max(0.0, _finite_nonneg(noise_mult, 2.0))
    cap = max(theta0, _finite_nonneg(noise_cap, 0.25))
    theta_eff = min(cap, math.sqrt(theta0 * theta0 + (mult * rho) * (mult * rho)))
    margin = max(0.0, _finite_nonneg(struct_margin, 0.01))

    conf_ok = True
    conf_val = None
    try:
        conf_val = float(confidence) if confidence is not None else None
        if conf_val is not None and math.isfinite(conf_val):
            conf_ok = conf_val >= float(min_confidence)
    except Exception:
        conf_ok = True
        conf_val = None

    base = {
        "iso_z_clean_certified": False,
        "iso_z_ratio": float(r),
        "iso_z_noise_ratio": float(rho),
        "iso_z_struct_ratio": float(r_struct),
        "iso_z_threshold": float(theta0),
        "iso_z_threshold_eff": float(theta_eff),
        "iso_z_noise_floor_screen": float(nf),
        "iso_z_y_scale": float(scale),
        "iso_z_confidence": conf_val,
    }

    if r <= theta_eff and r_struct <= theta0 + margin and conf_ok:
        return {
            **base,
            "status": "provisional",
            "decision": "allow",
            "iso_z_noise_compatible": True,
            "iso_z_uncertified": True,
            "provisional": True,
            "proposal_lane_protected": True,
            "structural_protected_acceptance": True,
            "iso_z_reject_reason": None,
        }

    reason = "noise_adjusted_threshold_failed"
    if not conf_ok:
        reason = "confidence_below_noisy_iso_z_minimum"
    return {
        **base,
        "status": "reject",
        "decision": "reject",
        "iso_z_noise_compatible": False,
        "iso_z_uncertified": False,
        "iso_z_reject_reason": reason,
    }


def _stageA_shadow_registry(search_hp) -> ShadowRegistry:
    """Return the mutable Stage-A shadow-coordinate registry for this run."""
    reg = getattr(search_hp, "_stageA_shadow_registry", None)
    if isinstance(reg, ShadowRegistry):
        return reg
    reg = ShadowRegistry()
    try:
        setattr(search_hp, "_stageA_shadow_registry", reg)
    except Exception:
        pass
    return reg


def _stageA_reset_shadow_registry(search_hp, *, reason: str = "") -> ShadowRegistry:
    """Start a fresh shadow-coordinate scope for an independent Stage-A run."""
    reg = ShadowRegistry()
    if search_hp is None:
        return reg
    old = getattr(search_hp, "_stageA_shadow_registry", None)
    try:
        setattr(search_hp, "_stageA_shadow_registry", reg)
    except Exception:
        return reg
    try:
        old_count = int(old.count()) if isinstance(old, ShadowRegistry) else 0
    except Exception:
        old_count = 0
    if old_count > 0:
        suffix = f" for {reason}" if reason else ""
        print(f"[Shadow] reset coordinate evidence{suffix}: dropped {old_count} stale shadow(s)")
    return reg


def _stageA_sync_shadow_registry(search_hp, current_ast: Node, *, reason: str = "") -> None:
    """Drop stale Stage-A shadows after an accepted AST rewrite."""
    reg = getattr(search_hp, "_stageA_shadow_registry", None)
    if not isinstance(reg, ShadowRegistry):
        return
    try:
        live_parent_vars = {}
        live_atoms = collect_nn_atoms(current_ast)
        for atom in live_atoms:
            live_parent_vars[shadow_parent_key(atom)] = tuple(int(v) for v in getattr(atom, "var_idxs", ()) or ())
    except Exception:
        return
    removed_parents, removed_shadows = reg.prune_for_live_parent_vars(live_parent_vars)
    consumed = {}
    try:
        for atom in live_atoms:
            key = shadow_parent_key(atom)
            for shadow in reg.local_for(key):
                if _stageA_shadow_ast_present_in_inputs(atom, shadow.shadow_ast):
                    consumed.setdefault(key, []).append(tuple(shadow.shadow_key))
    except Exception:
        consumed = {}
    removed_consumed = reg.prune_for_shadow_keys(consumed) if consumed else 0
    if removed_parents or removed_shadows or removed_consumed:
        suffix = f" after {reason}" if reason else ""
        print(
            "[Shadow] pruned stale coordinate evidence"
            f"{suffix}: parents={removed_parents}, shadows={removed_shadows}, "
            f"consumed={removed_consumed}"
        )


def _stageA_shadow_unit_status(expr: Node, units_spec, enforce_units: bool) -> str:
    if (not bool(enforce_units)) or units_spec is None:
        return "unchecked"
    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim

        return "unit_valid" if eval_analytic_expr_dim(expr, units_spec.x_dims) is not None else "unit_invalid"
    except Exception:
        return "unit_unknown"


def _stageA_record_shadow_coordinate(
    shadow_registry: ShadowRegistry | None,
    *,
    atom: AtomNode,
    base_ast: Node,
    shadow_ast: Node,
    transform_kind: str,
    source: str,
    confidence: float,
    unit_status: str = "unchecked",
    domain_ok_frac: float | None = None,
    evidence: Optional[Dict[str, Any]] = None,
    x_transform_map=None,
) -> None:
    """Record an uncommitted coordinate hint without changing the AST."""
    if shadow_registry is None:
        return
    try:
        shadow_readable = ast_to_human_readable(shadow_ast, x_transform_map)
    except Exception:
        shadow_readable = str(transform_kind)
    shadow = ShadowCoordinate(
        parent_key=shadow_parent_key(atom),
        parent_atom_tag=None if getattr(atom, "tag", None) is None else str(getattr(atom, "tag")),
        base_ast=clone_ast(base_ast),
        shadow_ast=clone_ast(shadow_ast),
        transform_kind=str(transform_kind),
        source=str(source),
        confidence=float(confidence),
        unit_status=str(unit_status),
        domain_ok_frac=domain_ok_frac,
        evidence=dict(evidence or {}),
    )
    stored, created = shadow_registry.add(shadow)
    verb = "recorded" if created else "merged"
    print(
        f"[Shadow] {verb} {shadow_readable} for parent={stored.parent_key} "
        f"source={stored.source} conf={float(stored.confidence):.3f} "
        f"unit={stored.unit_status}"
    )


def _stageA_record_logexp_shadows(
    *,
    atom: AtomNode,
    proposals,
    shadow_registry: ShadowRegistry | None,
    units_spec=None,
    enforce_units: bool = False,
    x_transform_map=None,
) -> None:
    """Record Stage-A log/exp coordinate lifts as shadows only."""
    for lp in proposals or ():
        try:
            z_desc = ast_to_human_readable(lp.z_ast, x_transform_map)
        except Exception:
            z_desc = str(getattr(lp, "label", "logexp"))
        print(
            "[Stage A LogExp] Shadowed "
            f"{lp.family}:{lp.wrapper} z={z_desc} "
            f"(conf={float(lp.confidence):.3f}); coordinate lift is not committed in Stage A"
        )
        _stageA_record_shadow_coordinate(
            shadow_registry,
                atom=atom,
                base_ast=getattr(lp, "base_ast", lp.z_ast),
                shadow_ast=lp.z_ast,
                transform_kind=str(getattr(lp, "wrapper", "logexp")),
                source="stageA_logexp",
                confidence=float(getattr(lp, "confidence", 0.0)),
                unit_status=_stageA_shadow_unit_status(lp.z_ast, units_spec, bool(enforce_units)),
                evidence=dict(getattr(lp, "meta", {}) or {}),
                x_transform_map=x_transform_map,
            )


def _stageA_compound_variant_shadow_only(z_name: str) -> bool:
    """Return True for wrapper variants that are evidence, not Stage-A structure."""
    return str(z_name).lower() in {"sin", "cos", "one_minus_cos"}


def _stageA_one_minus_cos_ast(arg: Node) -> Node:
    return AddNode(ConstNode(1.0), MulNode(ConstNode(-1.0), CosNode(clone_ast(arg))))


def _stageA_trig_shadow_from_spec(ts, inp: Node) -> Tuple[str, Node]:
    basis = str(getattr(ts, "basis_fn", "") or getattr(ts, "trig_fn", "")).lower()
    trig_fn = str(getattr(ts, "trig_fn", "")).lower()
    omega = snap_omega(float(getattr(ts, "omega", 1.0)))
    arg = clone_ast(inp) if abs(float(omega) - 1.0) <= 1.0e-12 else MulNode(ConstNode(float(omega)), clone_ast(inp))
    if basis == "one_minus_cos":
        return "one_minus_cos", _stageA_one_minus_cos_ast(arg)
    if trig_fn == "sin":
        return "sin", SinNode(arg)
    return "cos", CosNode(arg)


def _stageA_ast_contains_subexpr(root: Node, needle: Node) -> bool:
    """Return True if ``needle`` is already visible inside ``root``."""

    seen: set[int] = set()

    def _walk(node: Node) -> bool:
        obj_id = id(node)
        if obj_id in seen:
            return False
        seen.add(obj_id)
        try:
            if ast_equals(node, needle):
                return True
        except Exception:
            pass

        if isinstance(node, (AddNode, MulNode)):
            return _walk(node.left) or _walk(node.right)
        if isinstance(node, PowNode):
            return _walk(node.base)
        if isinstance(node, _STAGEA_UNARY_AST_NODES):
            return _walk(node.arg)
        if isinstance(node, AtomNode):
            # Only descend through explicitly stored compound inputs.  Plain
            # Var atoms synthesize themselves from var_idxs in get_input_exprs.
            inputs = getattr(node, "inputs", None)
            if inputs:
                return any(_walk(inp) for inp in inputs)
        return False

    return _walk(root)


def _stageA_shadow_ast_present_in_inputs(atom: AtomNode, shadow_ast: Node) -> bool:
    """Return True when a shadow coordinate has already been consumed by a leaf."""
    try:
        inputs = get_input_exprs(atom)
    except Exception:
        inputs = ()
    for expr in inputs:
        if _stageA_ast_contains_subexpr(expr, shadow_ast):
            return True
    return False


def _stageA_shadow_composite_proposals(
    proposals,
    *,
    atom: AtomNode,
    cols,
    z_ast_existing=None,
    shadow_registry: ShadowRegistry | None = None,
    enforce_units: bool = False,
    x_transform_map=None,
):
    """Promote shadows only when they combine with a visible base lane.

    A shadow such as ``sin(x4)`` or ``log(x4/x3)`` is not a committed coordinate
    by itself: an NN can absorb an invertible or many-to-one relabeling without
    any structural payoff.  It becomes a legitimate Stage-A proposal when it is
    multiplied into an existing monomial/radial/linear coordinate and thereby
    removes at least one raw input from the child NN, e.g.
    ``NN[x1,x2,x3,x4] -> NN[x2*x3*sin(x4), x1]``.

    A singleton raw-factor lane is intentionally allowed for cases such as
    ``x0*sin(x1)``; it is tagged separately in metadata and still needs visible
    payoff before commit.
    """
    if shadow_registry is None or not proposals:
        return []

    try:
        shadows = shadow_registry.local_for(shadow_parent_key(atom))
    except Exception:
        shadows = []
    if not shadows:
        return []

    usable_shadows = []
    promoted_kinds = {"sin", "cos", "one_minus_cos", "log", "exp"}
    for shadow in shadows:
        kind = str(getattr(shadow, "transform_kind", "")).lower()
        if kind not in promoted_kinds:
            continue
        unit_status_l = str(getattr(shadow, "unit_status", "unchecked")).lower()
        if unit_status_l == "unit_invalid":
            continue
        if bool(enforce_units) and unit_status_l != "unit_valid":
            continue
        if _stageA_shadow_ast_present_in_inputs(atom, shadow.shadow_ast):
            continue
        try:
            shadow_vars = tuple(int(v) for v in _collect_var_idxs_from_node(shadow.shadow_ast))
        except Exception:
            shadow_vars = ()
        if not shadow_vars:
            continue
        usable_shadows.append((shadow, tuple(sorted(set(shadow_vars)))))
    if not usable_shadows:
        return []

    def _normalise(prop):
        if len(prop) == 3:
            pattern, z_ast, conf = prop
            return pattern, z_ast, float(conf), None, {"kind": "monomial"}
        if len(prop) == 4:
            pattern, z_ast, conf, extra_override = prop
            return pattern, z_ast, float(conf), extra_override, {"kind": "monomial"}
        pattern, z_ast, conf, extra_override, meta = prop
        return pattern, z_ast, float(conf), extra_override, dict(meta or {})

    promoted = []
    seen = set()
    existing = set()

    def _key(pattern, z_ast, extras):
        try:
            z_key = ast_to_human_readable(z_ast, x_transform_map)
        except Exception:
            z_key = repr(z_ast)
        extras_key = tuple(sorted(int(v) for v in (extras or ())))
        pat_key = []
        for value in pattern:
            try:
                fv = float(value)
                pat_key.append(int(fv) if fv.is_integer() else round(fv, 12))
            except Exception:
                pat_key.append(str(value))
        return tuple(pat_key), extras_key, z_key

    for prop in proposals:
        try:
            pattern, z_ast, _conf, extra_override, _meta = _normalise(prop)
            extras = (
                list(extra_override)
                if extra_override is not None
                else _compound_candidate_default_extra_var_idxs(atom, pattern)
            )
            existing.add(_key(pattern, z_ast, extras))
        except Exception:
            continue

    for prop in proposals:
        try:
            pattern, z_ast, conf, extra_override, meta = _normalise(prop)
        except Exception:
            continue

        kind = str(meta.get("kind", "monomial")).lower()
        if kind not in {"monomial", "linear", "radial", "passthrough"}:
            continue
        try:
            pattern_list = list(pattern)
        except Exception:
            continue
        if len(pattern_list) != len(cols):
            continue

        extras = (
            list(extra_override)
            if extra_override is not None
            else _compound_candidate_default_extra_var_idxs(atom, tuple(pattern_list))
        )
        extras_set = {int(v) for v in extras}
        if not extras_set:
            continue

        # Require at least one non-shadow visible base factor.  A pure shadow like
        # sin(x4) is still coordinate churn, while x0*sin(x1) is an explicitly
        # allowed raw-factor lane that must pass the normal payoff audit.
        try:
            support_positions = [
                i for i, value in enumerate(pattern_list)
                if not _compound_pattern_entry_is_zero(value)
            ]
            base_support = int(
                len(support_positions)
            )
        except Exception:
            support_positions = []
            base_support = 0
        if base_support < 1:
            continue
        shadow_base_lane_class = "compound_lane"
        if base_support == 1:
            try:
                only_col = cols[int(support_positions[0])]
            except Exception:
                only_col = None
            shadow_base_lane_class = "compound_lane" if _is_compound_token(only_col) else "raw_factor"

        for shadow, shadow_vars in usable_shadows:
            shadow_var_set = set(int(v) for v in shadow_vars)
            if not shadow_var_set.issubset(extras_set):
                continue
            try:
                shadow_positions = [
                    i
                    for i, col in enumerate(cols)
                    if isinstance(col, int) and int(col) in shadow_var_set
                ]
            except Exception:
                shadow_positions = []
            if not shadow_positions:
                continue

            new_pattern = list(pattern_list)
            for pos in shadow_positions:
                new_pattern[pos] = "shadow"

            new_extras = sorted(int(v) for v in extras_set if int(v) not in shadow_var_set)
            z_shadow = MulNode(clone_ast(z_ast), clone_ast(shadow.shadow_ast))
            key = _key(new_pattern, z_shadow, new_extras)
            if key in seen or key in existing:
                continue
            seen.add(key)

            meta2 = dict(meta)
            meta2["kind"] = "shadow_composite"
            meta2["source"] = "shadow_coordinate_promotion"
            meta2["shadow_transform"] = str(getattr(shadow, "transform_kind", ""))
            meta2["shadow_source"] = str(getattr(shadow, "source", ""))
            meta2["shadow_unit_status"] = str(getattr(shadow, "unit_status", "unchecked"))
            meta2["shadow_vars"] = tuple(int(v) for v in shadow_vars)
            meta2["shadow_visible_ast"] = True
            meta2["shadow_requires_payoff"] = True
            meta2["hidden_shadow_only"] = False
            meta2["base_kind"] = kind
            meta2["base_pattern"] = tuple(pattern_list)
            meta2["shadow_base_lane_class"] = shadow_base_lane_class

            try:
                z_base_s = ast_to_human_readable(z_ast, x_transform_map)
                z_shadow_s = ast_to_human_readable(z_shadow, x_transform_map)
            except Exception:
                z_base_s = "z"
                z_shadow_s = "shadow_composite"
            meta2["shadow_base_readable"] = str(z_base_s)
            meta2["shadow_candidate_readable"] = str(z_shadow_s)
            print(
                "[Shadow] Promoting shadow with compound: "
                f"base={z_base_s}, base_lane={shadow_base_lane_class}, "
                f"shadow={meta2['shadow_transform']}({shadow_vars}), "
                f"z={z_shadow_s}, extras={new_extras}"
            )

            try:
                shadow_conf = float(getattr(shadow, "confidence", 0.0))
            except Exception:
                shadow_conf = 0.0
            conf_new = max(0.0, min(0.999, min(float(conf), shadow_conf) * 0.97))
            promoted.append(
                (
                    tuple(new_pattern),
                    z_shadow,
                    float(conf_new),
                    new_extras if new_extras else None,
                    meta2,
                )
            )

    return promoted


def _stageA_shadow_preserved_factor_proposals(
    *,
    atom: AtomNode,
    cols,
    shadow_registry: ShadowRegistry | None = None,
    enforce_units: bool = False,
    x_transform_map=None,
):
    """Promote shadows by multiplying them into already-visible factors.

    This is the visible-factor complement to
    ``_stageA_shadow_composite_proposals``.  The existing promoter handles
    ``NN[base, raw_extra] -> NN[base*sin(raw_extra)]``.  Here we handle leaves
    that already contain a clean effective factor, e.g. ``NN[P, x3, x4]`` or
    ``NN[R, P]`` with a recorded ``log(R)`` shadow:
    ``NN[P, x3, x4] -> NN[P*log(x4/x3)]``.

    The proposal is still only visible evidence.  It must pass the normal Stage-A
    validation, units/Buckingham checks, and shadow payoff guard before commit.
    """
    if shadow_registry is None:
        return []

    try:
        inputs = tuple(get_input_exprs(atom))
    except Exception:
        return []
    if not inputs:
        return []

    try:
        shadows = shadow_registry.local_for(shadow_parent_key(atom))
    except Exception:
        shadows = []
    if not shadows:
        return []

    cols = list(cols or ())
    if len(cols) != len(inputs):
        cols = list(range(len(inputs)))

    def _support(expr) -> set[int]:
        try:
            return set(int(v) for v in _collect_var_idxs_from_node(expr))
        except Exception:
            return set()

    def _matches(a, b) -> bool:
        try:
            if ast_equals(a, b):
                return True
        except Exception:
            pass
        try:
            return ast_to_human_readable(a, x_transform_map) == ast_to_human_readable(b, x_transform_map)
        except Exception:
            return False

    def _mul_all(factors):
        out = clone_ast(factors[0])
        for factor in factors[1:]:
            out = MulNode(out, clone_ast(factor))
        return out

    promoted = []
    seen = set()
    promoted_kinds = {"sin", "cos", "one_minus_cos", "log", "exp"}
    input_supports = [_support(inp) for inp in inputs]

    for shadow in shadows:
        kind = str(getattr(shadow, "transform_kind", "")).lower()
        if kind not in promoted_kinds:
            continue
        unit_status_l = str(getattr(shadow, "unit_status", "unchecked")).lower()
        if unit_status_l == "unit_invalid":
            continue
        if bool(enforce_units) and unit_status_l != "unit_valid":
            continue
        if _stageA_shadow_ast_present_in_inputs(atom, shadow.shadow_ast):
            continue

        shadow_base = getattr(shadow, "base_ast", None)
        shadow_vars = _support(shadow_base) if shadow_base is not None else set()
        if not shadow_vars:
            shadow_vars = _support(shadow.shadow_ast)
        if not shadow_vars:
            continue

        consumed_positions: set[int] = set()
        covered_vars: set[int] = set()
        partial_overlap = False
        for idx, inp in enumerate(inputs):
            supp = set(input_supports[idx])
            exact_base = shadow_base is not None and _matches(inp, shadow_base)
            if exact_base or (supp and supp.issubset(shadow_vars)):
                consumed_positions.add(int(idx))
                covered_vars.update(supp)
                continue
            if supp & shadow_vars:
                partial_overlap = True
                break
        if partial_overlap:
            continue
        if not consumed_positions or not shadow_vars.issubset(covered_vars):
            continue

        preserved_positions = [i for i in range(len(inputs)) if i not in consumed_positions]
        if not preserved_positions:
            continue

        # Keep this promoter narrowly scoped to *preserved factor* situations.
        # All-raw complements are handled by the normal compound-lane promoter.
        if not any(not is_trivial_input(inputs[i]) for i in preserved_positions):
            continue

        try:
            preserved_factor = _mul_all([inputs[i] for i in preserved_positions])
            z_shadow = MulNode(preserved_factor, clone_ast(shadow.shadow_ast))
        except Exception:
            continue

        pattern = tuple("shadow" if i in consumed_positions else "factor" for i in range(len(cols)))
        try:
            z_key = ast_to_human_readable(z_shadow, x_transform_map)
        except Exception:
            z_key = repr(z_shadow)
        key = (pattern, z_key)
        if key in seen:
            continue
        seen.add(key)

        meta = {
            "kind": "shadow_preserved_factor",
            "source": "shadow_coordinate_preserved_factor",
            "shadow_transform": str(getattr(shadow, "transform_kind", "")),
            "shadow_source": str(getattr(shadow, "source", "")),
            "shadow_unit_status": str(getattr(shadow, "unit_status", "unchecked")),
            "shadow_vars": tuple(int(v) for v in sorted(shadow_vars)),
            "shadow_consumed_positions": tuple(int(i) for i in sorted(consumed_positions)),
            "shadow_preserved_positions": tuple(int(i) for i in preserved_positions),
            "shadow_visible_ast": True,
            "shadow_requires_payoff": True,
            "hidden_shadow_only": False,
            "base_kind": "preserved_factor",
            "shadow_base_lane_class": "preserved_factor",
        }

        try:
            shadow_s = ast_to_human_readable(shadow.shadow_ast, x_transform_map)
            factor_s = ast_to_human_readable(preserved_factor, x_transform_map)
        except Exception:
            shadow_s = str(getattr(shadow, "transform_kind", "shadow"))
            factor_s = "preserved_factor"
        meta["shadow_base_readable"] = str(factor_s)
        meta["shadow_candidate_readable"] = str(z_key)
        print(
            "[Shadow] Promoting shadow with preserved factor: "
            f"factor={factor_s}, shadow={shadow_s}, z={z_key}, extras=[]"
        )

        try:
            shadow_conf = float(getattr(shadow, "confidence", 0.0))
        except Exception:
            shadow_conf = 0.0
        conf_new = max(0.0, min(0.999, shadow_conf * 0.95))
        promoted.append((pattern, z_shadow, float(conf_new), None, meta))

    return promoted


def _stageA_shadow_trig_factor_peel_proposals(
    *,
    atom: AtomNode,
    cols,
    shadow_registry: ShadowRegistry | None = None,
    enforce_units: bool = False,
    x_transform_map=None,
):
    """Materialise trig shadows as visible multiplicative peels.

    This is the pre-separability counterpart to shadow-composite promotion:
    ``NN[theta, extras] -> trig(theta) * NN[extras]``.  A pure hidden lift
    ``NN[theta] -> NN[trig(theta)]`` remains forbidden; the trig expression
    must appear explicitly in the AST and reduce the residual NN burden.
    """
    if shadow_registry is None:
        return []
    try:
        inputs = tuple(get_input_exprs(atom))
    except Exception:
        return []
    if len(inputs) < 2:
        return []
    try:
        shadows = shadow_registry.local_for(shadow_parent_key(atom))
    except Exception:
        shadows = []
    if not shadows:
        return []

    cols = list(cols or ())
    if len(cols) != len(inputs):
        cols = list(range(len(inputs)))

    def _support(expr) -> set[int]:
        try:
            return set(int(v) for v in _collect_var_idxs_from_node(expr))
        except Exception:
            return set()

    def _matches(a, b) -> bool:
        try:
            if ast_equals(a, b):
                return True
        except Exception:
            pass
        try:
            return ast_to_human_readable(a, x_transform_map) == ast_to_human_readable(b, x_transform_map)
        except Exception:
            return False

    promoted = []
    seen = set()
    input_supports = [_support(inp) for inp in inputs]
    trig_kinds = {"sin", "cos", "one_minus_cos"}

    for shadow in shadows:
        kind = str(getattr(shadow, "transform_kind", "")).lower()
        if kind not in trig_kinds:
            continue
        unit_status_l = str(getattr(shadow, "unit_status", "unchecked")).lower()
        if unit_status_l == "unit_invalid":
            continue
        if bool(enforce_units) and unit_status_l != "unit_valid":
            continue
        if _stageA_shadow_ast_present_in_inputs(atom, shadow.shadow_ast):
            continue

        shadow_base = getattr(shadow, "base_ast", None)
        shadow_vars = _support(shadow_base) if shadow_base is not None else set()
        if not shadow_vars:
            shadow_vars = _support(shadow.shadow_ast)
        if not shadow_vars:
            continue

        consumed_positions: set[int] = set()
        covered_vars: set[int] = set()
        partial_overlap = False
        for idx, inp in enumerate(inputs):
            supp = set(input_supports[idx])
            exact_base = shadow_base is not None and _matches(inp, shadow_base)
            if exact_base or (supp and supp.issubset(shadow_vars)):
                consumed_positions.add(int(idx))
                covered_vars.update(supp)
                continue
            if supp & shadow_vars:
                partial_overlap = True
                break
        if partial_overlap:
            continue
        if not consumed_positions or not shadow_vars.issubset(covered_vars):
            continue

        preserved_positions = [i for i in range(len(inputs)) if i not in consumed_positions]
        if not preserved_positions:
            # The terminal NN[theta] -> scale*trig(theta) case belongs to Stage B.
            continue

        # Choose one preserved input as the residual leaf's primary z, and keep
        # the others as raw/compound extras.  Prefer a nontrivial existing
        # compound so already-discovered structure is not flattened.
        primary_pos = None
        for pos in preserved_positions:
            if not is_trivial_input(inputs[pos]):
                primary_pos = int(pos)
                break
        if primary_pos is None:
            primary_pos = int(preserved_positions[0])

        extra_raw: List[int] = []
        extra_asts: List[Node] = []
        for pos in preserved_positions:
            if int(pos) == int(primary_pos):
                continue
            inp = inputs[int(pos)]
            if is_trivial_input(inp):
                try:
                    extra_raw.append(int(inp.var_idxs[0]))
                except Exception:
                    pass
            else:
                extra_asts.append(clone_ast(inp))

        pattern = []
        for pos in range(len(inputs)):
            if int(pos) in consumed_positions:
                pattern.append("shadow")
            elif int(pos) == int(primary_pos):
                pattern.append(1)
            else:
                pattern.append(0)

        z_ast = clone_ast(inputs[int(primary_pos)])
        try:
            z_key = ast_to_human_readable(z_ast, x_transform_map)
            shadow_key = ast_to_human_readable(shadow.shadow_ast, x_transform_map)
        except Exception:
            z_key = "residual"
            shadow_key = kind
        extra_keys = []
        for expr in extra_asts:
            try:
                extra_keys.append(ast_to_human_readable(expr, x_transform_map))
            except Exception:
                extra_keys.append(repr(expr))
        key = (tuple(pattern), tuple(sorted(extra_raw)), tuple(extra_keys), z_key, shadow_key)
        if key in seen:
            continue
        seen.add(key)

        meta = {
            "kind": "shadow_trig_factor_peel",
            "source": "shadow_coordinate_trig_factor_peel",
            "prefactor_ast": clone_ast(shadow.shadow_ast),
            "shadow_transform": kind,
            "shadow_source": str(getattr(shadow, "source", "")),
            "shadow_unit_status": str(getattr(shadow, "unit_status", "unchecked")),
            "shadow_vars": tuple(int(v) for v in sorted(shadow_vars)),
            "shadow_consumed_positions": tuple(int(i) for i in sorted(consumed_positions)),
            "shadow_preserved_positions": tuple(int(i) for i in preserved_positions),
            "shadow_visible_ast": True,
            "shadow_requires_payoff": True,
            "hidden_shadow_only": False,
            "base_kind": "trig_factor_peel",
            "shadow_base_lane_class": "factor_peel",
        }
        if extra_asts:
            meta["extra_input_asts"] = tuple(extra_asts)
        meta["shadow_base_readable"] = str(z_key)
        meta["shadow_candidate_readable"] = f"{shadow_key} * NN[residual]"
        print(
            "[Shadow] Promoting shadow as visible trig factor peel: "
            f"factor={shadow_key}, residual_z={z_key}, raw_extras={sorted(extra_raw)}, "
            f"compound_extras={len(extra_asts)}"
        )

        try:
            shadow_conf = float(getattr(shadow, "confidence", 0.0))
        except Exception:
            shadow_conf = 0.0
        conf_new = max(0.0, min(0.999, shadow_conf * 0.96))
        promoted.append(
            (
                tuple(pattern),
                z_ast,
                float(conf_new),
                sorted(set(extra_raw)) if extra_raw else None,
                meta,
            )
        )

    return promoted


def _stageA_shadow_trig_composite_proposals(*args, **kwargs):
    """Back-compat wrapper for tests/imports from the initial trig-only PR."""
    return _stageA_shadow_composite_proposals(*args, **kwargs)

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
RESET = "\033[0m"


def _loss_str(loss: float, lm_hp) -> str:
    """Format loss value with asinh indicator if fit-link is active."""
    if getattr(lm_hp, "fit_y_link", None) == "asinh":
        return f"{loss:.4e} [asinh]"
    return f"{loss:.4e}"


def _compound_candidate_new_arity(
    *,
    extra_var_count: int,
    extra_input_asts: Optional[List[Node]] = None,
) -> int:
    """Return the effective arity of a proposed compound leaf."""
    n_ast_extras = len(extra_input_asts) if extra_input_asts else 0
    return int(1 + int(extra_var_count) + int(n_ast_extras))


def _should_skip_compound_extension_after_sep(
    *,
    already_sep: bool,
    extra_var_idxs: Optional[List[int]] = None,
    extra_input_asts: Optional[List[Node]] = None,
) -> bool:
    """Return whether an already-separable compound atom can skip extension."""
    if not bool(already_sep):
        return False
    n_effective_extras = len(extra_var_idxs or ()) + len(extra_input_asts or ())
    # If extras remain, a candidate may still consume the separated coordinate
    # into a lower-arity expression, e.g. NN[q, x2] -> NN[q*x2].  Extra-only
    # refinements that would preserve q as an NN input are rejected later by
    # _compound_candidate_preserves_separated_coordinate; after q separates it
    # belongs to the split transaction, not to the residual branch.
    return int(n_effective_extras) <= 0


def _compound_candidate_preserves_separated_coordinate(
    *,
    already_sep: bool,
    atom: Optional[AtomNode] = None,
    pattern=None,
    preserve_z_ast: Optional[Node] = None,
    extra_input_asts: Optional[Iterable[Node]] = None,
) -> bool:
    """Return whether a proposal keeps an already-separated compound as an NN input.

    Once the current compound coordinate has been certified separable from the
    remaining extras, extra-only refinements must be applied inside that split
    transaction.  Treating the old coordinate as a synthetic extra makes
    NN[q, x, y] -> NN[z(x,y), q] look like an arity reduction even though the
    correct baseline is q * NN[x, y] (or the additive analogue).
    """
    if not bool(already_sep):
        return False
    if preserve_z_ast is not None:
        return True
    if atom is None or not has_nontrivial_input(atom):
        return False
    if extra_input_asts:
        try:
            existing_inputs = [
                inp for inp in get_input_exprs(atom)
                if not is_trivial_input(inp)
            ]
        except Exception:
            existing_inputs = []
        if existing_inputs:
            existing_keys = set()
            for inp in existing_inputs:
                try:
                    existing_keys.add(ast_to_human_readable(inp))
                except Exception:
                    pass
            for extra_ast in extra_input_asts:
                for existing in existing_inputs:
                    try:
                        if ast_equals(extra_ast, existing):
                            return True
                    except Exception:
                        pass
                if existing_keys:
                    try:
                        if ast_to_human_readable(extra_ast) in existing_keys:
                            return True
                    except Exception:
                        pass
    if pattern is None:
        return False
    try:
        cols, z_existing = _atom_compound_cols(atom)
    except Exception:
        return False
    if z_existing is None:
        return False
    for z_col, z_tok in enumerate(cols):
        if not _is_compound_token(z_tok) or int(z_col) >= len(pattern):
            continue
        try:
            if _compound_pattern_entry_is_zero(pattern[int(z_col)]):
                return True
        except Exception:
            continue
    return False


def _select_compound_z_variant_shortlist(
    scored_variants,
    *,
    kind: str,
    screen_gate: Optional[float] = None,
    max_variants_to_try: int = 0,
):
    """Select bounded z-wrapper variants without dropping radial essentials."""
    variants = list(scored_variants or ())
    if not variants:
        return []

    def _score(item) -> float:
        try:
            value = item[2]
            return 0.0 if value is None else float(value)
        except Exception:
            return 0.0

    if screen_gate is None:
        ordered = variants
        kept = list(variants)
    else:
        ordered = sorted(variants, key=_score, reverse=True)
        kept = [t for t in ordered if _score(t) >= float(screen_gate)]
        if not kept and ordered:
            kept = [ordered[0]]

    out = []
    seen = set()

    def _add(item) -> None:
        name = str(item[0])
        if name in seen:
            return
        seen.add(name)
        out.append(item)

    # For radial proposals the base coordinate is r^2 and sqrt(r^2) is the
    # physically natural radius.  Scatter screening can rank rational wrappers
    # above both, so retain these two after unit filtering and before the cap.
    if str(kind).lower() == "radial":
        by_name = {}
        for item in ordered:
            by_name.setdefault(str(item[0]), item)
        for required in ("z", "sqrt"):
            if required in by_name:
                _add(by_name[required])

    for item in kept:
        _add(item)

    if int(max_variants_to_try) > 0:
        out = out[: int(max_variants_to_try)]
    return out


def _stageA_coordinate_collapse_screen(coord_vals, y_vals: torch.Tensor, n_bins: int = 64) -> float:
    """Cheap grouped-mean score for ``y`` as a function of one or more coordinates."""
    try:
        coords = []
        for cv in list(coord_vals or ()):
            if cv is None:
                continue
            coords.append(cv.reshape(-1))
        if not coords:
            return 0.0

        y = y_vals.reshape(-1)
        n = int(y.numel())
        if n < 128:
            return 0.0
        coords = [c[:n] for c in coords]
        y = y[:n]
        mask = torch.isfinite(y)
        for c in coords:
            mask = mask & torch.isfinite(c)
        if int(mask.sum().item()) < 128:
            return 0.0
        y = y[mask]
        coords = [c[mask] for c in coords]
        y_mean = y.mean()
        sst = torch.sum((y - y_mean) ** 2)
        if float(sst) <= 1e-20:
            return 0.0

        d = len(coords)
        if d == 1:
            idx = torch.argsort(coords[0])
            y_sorted = y[idx]
            nb = int(max(4, min(int(n_bins), int(y_sorted.numel()))))
            step = max(1, int(y_sorted.numel()) // nb)
            sse = torch.tensor(0.0, device=y.device, dtype=y.dtype)
            for b in range(nb):
                a = b * step
                c = int(y_sorted.numel()) if b == nb - 1 else min(int(y_sorted.numel()), (b + 1) * step)
                if c <= a:
                    break
                yb = y_sorted[a:c]
                mu = yb.mean()
                sse = sse + torch.sum((yb - mu) ** 2)
        else:
            if d > 4:
                return 0.0
            bins_per_dim = int(max(2, min(8, round(float(n_bins) ** (1.0 / float(d))))))
            code = torch.zeros_like(y, dtype=torch.long)
            for cvals in coords:
                try:
                    lo = torch.quantile(cvals, 0.01)
                    hi = torch.quantile(cvals, 0.99)
                except Exception:
                    lo = torch.min(cvals)
                    hi = torch.max(cvals)
                span = hi - lo
                if (not torch.isfinite(span)) or float(torch.abs(span).detach().cpu()) <= 1.0e-30:
                    b = torch.zeros_like(code)
                else:
                    t = torch.clamp((cvals - lo) / (span + 1.0e-30), 0.0, 0.999999)
                    b = torch.floor(t * bins_per_dim).to(dtype=torch.long)
                code = code * int(bins_per_dim) + b

            order = torch.argsort(code)
            code_s = code[order]
            y_s = y[order]
            sse = torch.tensor(0.0, device=y.device, dtype=y.dtype)
            start = 0
            try:
                _, counts = torch.unique_consecutive(code_s, return_counts=True)
                group_counts = [int(v) for v in counts.detach().cpu().tolist()]
            except Exception:
                group_counts = [int(y_s.numel())]
            for cnt in group_counts:
                end = start + int(cnt)
                yb = y_s[start:end]
                if int(yb.numel()) > 0:
                    mu = yb.mean()
                    sse = sse + torch.sum((yb - mu) ** 2)
                start = end

        r2 = 1.0 - float((sse / (sst + 1e-30)).detach().cpu())
        if not math.isfinite(r2):
            return 0.0
        return float(max(0.0, min(1.0, r2)))
    except Exception:
        return 0.0


def _compound_buckingham_min_freedom(kind: str) -> int:
    """Return the Buckingham π freedom floor for a Stage-A compound kind."""
    kind = str(kind or "").lower()
    if kind in {
        "linear",
        "shift",
        "radial",
        "power_difference",
        "power_difference_bundle",
    }:
        return 0
    return 1


def _stageA_compound_buckingham_target_dim(current_ast: Node, atom: AtomNode, units_spec):
    """Return the local output dimension for a Stage-A compound target.

    Compound Buckingham checks are local to the NN atom being rewritten.  If
    that atom sits inside an analytic multiplicative context, judging it
    against the global y dimension incorrectly rejects dimension-carrying
    compounds such as NN[x1,x2,x3,x4] -> NN[x1,x2*x3,x4] in
    x0*NN[...].
    """
    if units_spec is None:
        return None
    try:
        from nestynet_sr.sr_core.units import infer_atom_output_dim

        target_dim = infer_atom_output_dim(current_ast, atom, units_spec)
        if target_dim is not None:
            return tuple(target_dim)
    except Exception:
        pass
    try:
        return units_spec.y_phi_dim
    except Exception:
        return None


def _analytic_units_rejection(expr: Node, units_spec, *, enforce_units: bool):
    """Return a units rejection reason for the actual analytic candidate AST."""
    if (not bool(enforce_units)) or (units_spec is None):
        return None
    try:
        from nestynet_sr.sr_core.units import check_units_ast

        ures = check_units_ast(expr, units_spec)
        if not bool(getattr(ures, "ok", False)):
            return getattr(ures, "reason", "unit check failed")
    except Exception as e:
        return f"units error: {e}"
    return None


def _compound_absorbed_effective_inputs(atom: AtomNode, new_arity: int) -> int:
    """Return how many effective inputs a compound proposal absorbs."""
    return max(0, int(effective_arity(atom)) - int(new_arity))


def _aggregate_losses(losses: List[float], mode: str = "mean", weights: Optional[List[float]] = None) -> float:
    """Aggregate per-dataset losses with D=1 identity behaviour."""
    if not losses:
        return float("inf")
    if len(losses) == 1:
        return float(losses[0])
    m = str(mode or "mean").lower().strip()
    if m in ("mean", "avg", "average"):
        return float(sum(float(x) for x in losses) / len(losses))
    if m in ("sum",):
        return float(sum(float(x) for x in losses))
    if m in ("median",):
        xs = sorted(float(x) for x in losses)
        k = len(xs) // 2
        return float(xs[k]) if (len(xs) % 2 == 1) else float(0.5 * (xs[k - 1] + xs[k]))
    if m in ("weighted", "weighted_mean", "wmean"):
        if not weights or len(weights) != len(losses):
            return float(sum(float(x) for x in losses) / len(losses))
        wsum = float(sum(float(w) for w in weights))
        if wsum <= 0:
            return float(sum(float(x) for x in losses) / len(losses))
        return float(sum(float(w) * float(x) for w, x in zip(weights, losses)) / wsum)
    return float(sum(float(x) for x in losses) / len(losses))


def _trig_kind_from_phase(phase: float) -> str:
    """Determine sin vs cos from FFT phase.

    Phase near 0 or π → cos (even function)
    Phase near ±π/2 → sin (odd function)

    Returns "sin" or "cos".
    """
    # Normalize phase to [-π, π)
    phase = phase % (2 * math.pi)
    if phase > math.pi:
        phase -= 2 * math.pi

    # Distance to cos-like phases (0, π, -π)
    cos_dist = min(abs(phase), abs(phase - math.pi), abs(phase + math.pi))
    # Distance to sin-like phases (π/2, -π/2)
    sin_dist = min(abs(phase - math.pi / 2), abs(phase + math.pi / 2))

    return "cos" if cos_dist < sin_dist else "sin"


def _oracle_trig_for_axis(axis, oracle_trig_specs):
    """Look up a TrigScaleSpec for a given axis index from oracle trig probes.

    Returns the matching TrigScaleSpec or None.
    """
    if not oracle_trig_specs:
        return None
    for spec in oracle_trig_specs:
        if int(spec.axis) == int(axis):
            return spec
    return None


def _oracle_trig_to_axis_specs(oracle_trig_specs, cols):
    """Convert oracle TrigScaleSpecs (local axes) to TrigAxisSpecs (global axes)."""
    result = []
    for ts in (oracle_trig_specs or []):
        local_ax = int(ts.axis)
        if local_ax < len(cols) and isinstance(cols[local_ax], int):
            result.append(TrigAxisSpec(
                axis=int(cols[local_ax]),
                omega=ts.omega,
                strength=100.0,  # synthetic strength; oracle-verified
                n_points=ts.n_points,
                tmin=0.0, tmax=0.0,
                phase=0.0 if str(getattr(ts, "trig_fn", "cos")) == "cos" else math.pi / 2,
                basis_fn=str(getattr(ts, "basis_fn", "") or getattr(ts, "trig_fn", "")),
            ))
    return result


@torch.no_grad()
def _sanitize_tensor(t):
    """Replace a TensorWrapper / FunctionalTensor with a plain tensor."""
    if torch.is_tensor(t) and type(t) is not torch.Tensor:
        return t.detach().clone()
    return t


def _sanitize_any(obj, visited=None):
    """Recursively sanitize wrapped tensors in containers and plain objects."""
    if visited is None:
        visited = set()
    if torch.is_tensor(obj):
        return _sanitize_tensor(obj)
    if isinstance(obj, dict):
        return _sanitize_container(obj, visited)
    if isinstance(obj, list):
        return _sanitize_container(obj, visited)
    if isinstance(obj, tuple):
        return _sanitize_container(obj, visited)
    if isinstance(obj, set):
        return _sanitize_container(obj, visited)

    # Last-resort: walk plain object attributes (custom helper/cache objects).
    # Skip nn.Module here - modules are traversed explicitly in _sanitize_func_tensors.
    if (
        hasattr(obj, "__dict__")
        and not isinstance(obj, torch.nn.Module)
        and not isinstance(obj, type)
    ):
        obj_id = id(obj)
        if obj_id in visited:
            return obj
        visited.add(obj_id)
        for attr_name, val in list(obj.__dict__.items()):
            if callable(val):
                continue
            new_val = _sanitize_any(val, visited)
            if new_val is not val:
                try:
                    setattr(obj, attr_name, new_val)
                except Exception:
                    # Best-effort cleanup only.
                    pass
    return obj


def _sanitize_container(obj, visited=None):
    """Recursively sanitize wrapped tensors in Python containers."""
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _sanitize_any(v, visited)
        return obj

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _sanitize_any(v, visited)
        return obj

    if isinstance(obj, tuple):
        vals = [_sanitize_any(v, visited) for v in obj]
        if any(v_new is not v_old for v_new, v_old in zip(vals, obj)):
            return tuple(vals)
        return obj

    if isinstance(obj, set):
        vals = [_sanitize_any(v, visited) for v in obj]
        if any(v_new is not v_old for v_new, v_old in zip(vals, obj)):
            obj.clear()
            obj.update(vals)
        return obj

    return obj


def _sanitize_func_tensors(model):
    """Replace any TensorWrapper / FunctionalTensor objects with plain tensors.

    ``torch.func`` transforms (jvp, vjp, jacrev) used by AutogradAdaptor can
    leave FunctionalTensor wrappers in the model's state.  These cannot be
    ``copy.deepcopy``'d.  Call this after gauge-fix training to clean up.
    """
    visited = set()

    # nn.Parameter .data
    for param in model.parameters():
        if type(param.data) is not torch.Tensor:
            param.data = param.data.detach().clone()

    # Registered buffers
    module_attr_skip = {
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_tensor_hooks",
    }
    for mod in model.modules():
        for key, buf in list(mod._buffers.items()):
            if buf is not None and type(buf) is not torch.Tensor:
                mod._buffers[key] = buf.detach().clone()
        # Walk all remaining attributes on each module's __dict__ to catch
        # TensorWrappers stored as plain/private attributes, in tuples, or in nested containers.
        for attr_name, val in list(mod.__dict__.items()):
            if attr_name in module_attr_skip:
                continue
            new_val = _sanitize_any(val, visited)
            if new_val is not val:
                try:
                    setattr(mod, attr_name, new_val)
                except Exception:
                    # Best-effort cleanup only.
                    pass


def _clone_reuse_leaves(reuse, device, dtype):
    """
    Deep-clone a reuse_leaves dictionary so that candidate models don't share
    module instances with the current model.

    This is critical for correctness: if a candidate is rejected after LM training,
    we don't want those parameter updates to "haunt" the current accepted model.
    """
    if not reuse:
        return {}
    out = {}
    for k, m in reuse.items():
        cloned = None
        last_exc = None
        try:
            cloned = copy.deepcopy(m)
        except Exception as exc:
            # TensorWrapper from torch.func transforms can't be deepcopy'd.
            # Sanitize in-place and retry.
            last_exc = exc
            _sanitize_func_tensors(m)
            try:
                cloned = copy.deepcopy(m)
            except Exception as exc2:
                last_exc = exc2
        if cloned is None:
            # Fail-safe: avoid crashing the whole run if one leaf carries
            # non-copyable runtime wrappers. This leaf simply won't be warm-started.
            print(
                f"[Warm-start] Skipping reuse for leaf '{k}': "
                f"deepcopy failed after sanitization ({type(last_exc).__name__}: {last_exc})"
            )
            continue
        cloned.to(device=device, dtype=dtype)
        # Move any known auxiliary tensor lists that are not registered buffers/params.
        # (Module.to(...) will not move raw tensors inside python lists/tuples.)
        for obj in (cloned, getattr(cloned, "base_model", None)):
            if obj is None:
                continue
            for attr in ("a_pieces_fixed", "b_pieces_fixed"):
                v = getattr(obj, attr, None)
                if isinstance(v, (list, tuple)):
                    setattr(
                        obj,
                        attr,
                        [t.to(device=device, dtype=dtype) if torch.is_tensor(t) else t for t in v],
                    )
        out[k] = cloned
    return out


def _ast_matches_arch(ast, *, num_segments: int, dual_layer: bool) -> bool:
    # Only reuse/warm-start if we are not changing the leaf architectures.
    for a in collect_nn_atoms(ast):
        kw = getattr(a, "kwargs", None) or {}
        if int(kw.get("num_segments", num_segments)) != int(num_segments):
            return False
        if bool(kw.get("dual_layer", dual_layer)) != bool(dual_layer):
            return False
    return True


def _apply_fit_link_to_model(model, lm_hp):
    """Propagate fit-link settings onto a freshly built composite model."""
    if model is None:
        return None
    # Store the raw name (may be None). The adaptor will canonicalize on use.
    setattr(model, "fit_y_link", getattr(lm_hp, "fit_y_link", None))
    setattr(model, "fit_y_link_scale", float(getattr(lm_hp, "fit_y_link_scale", 1.0)))
    return model


# ──────────────────────────────────────────────────────────────
# Residual re-fit helpers: remove additive gauge contamination
# ──────────────────────────────────────────────────────────────

__search_definitions__ = (
    "_stageA_classify_iso_z_result",
    "_stageA_shadow_registry",
    "_stageA_reset_shadow_registry",
    "_stageA_sync_shadow_registry",
    "_stageA_shadow_unit_status",
    "_stageA_record_shadow_coordinate",
    "_stageA_record_logexp_shadows",
    "_stageA_compound_variant_shadow_only",
    "_stageA_one_minus_cos_ast",
    "_stageA_trig_shadow_from_spec",
    "_stageA_ast_contains_subexpr",
    "_stageA_shadow_ast_present_in_inputs",
    "_stageA_shadow_composite_proposals",
    "_stageA_shadow_preserved_factor_proposals",
    "_stageA_shadow_trig_factor_peel_proposals",
    "_stageA_shadow_trig_composite_proposals",
    "_loss_str",
    "_compound_candidate_new_arity",
    "_should_skip_compound_extension_after_sep",
    "_compound_candidate_preserves_separated_coordinate",
    "_select_compound_z_variant_shortlist",
    "_stageA_coordinate_collapse_screen",
    "_compound_buckingham_min_freedom",
    "_stageA_compound_buckingham_target_dim",
    "_analytic_units_rejection",
    "_compound_absorbed_effective_inputs",
    "_aggregate_losses",
    "_trig_kind_from_phase",
    "_oracle_trig_for_axis",
    "_oracle_trig_to_axis_specs",
    "_sanitize_tensor",
    "_sanitize_any",
    "_sanitize_container",
    "_sanitize_func_tensors",
    "_clone_reuse_leaves",
    "_ast_matches_arch",
    "_apply_fit_link_to_model",
)

__search_constants__ = (
    "_STAGEA_UNARY_AST_NODES",
    "RED",
    "GREEN",
    "YELLOW",
    "BLUE",
    "PURPLE",
    "RESET",
)

__search_late_bindings__ = (
    "_atom_compound_cols",
    "_compound_candidate_default_extra_var_idxs",
    "_compound_pattern_entry_is_zero",
    "_is_compound_token",
)
