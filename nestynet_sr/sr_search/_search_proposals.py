# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compound proposal bookkeeping, replay, separability, and closure scoring."""

from typing import TYPE_CHECKING
import json
import math
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple
import torch
from nestynet_sr.sr_core import Var, ast_to_human_readable, collect_nn_atoms, replace_atom_in_ast
from nestynet_sr.sr_core.bridges import AcosNode, AddNode, AsinNode, AtanNode, AtomNode, ConstNode, CosNode, ExpNode, LogNode, MulNode, Node, PowNode, Scale, SinNode, _collect_var_idxs_from_inputs, _collect_var_idxs_from_node, clone_ast, compound_input_expr, effective_arity, eval_inputs, extra_input_var_idxs, get_input_exprs, has_nontrivial_input, is_pure_1d_full_compound_ast as _shared_is_pure_1d_full_compound_ast, is_trivial_input
from nestynet_sr.sr_core.fit_links import canonical_fit_link_name
from nestynet_sr.sr_core.problem_identity import canonical_problem_id
from .candidate_builders import _gather_atom_teacher_data
from .monomial_screen import candidate_monomial_exponent, candidate_priority_from_screen, fit_univariate_monomial_screen, half_power_domain_ok, monomial_power_label, snap_to_half_integer_monomial_power
from .model_builders import build_composite_ast
from .model_selection import compute_accept_threshold as _compute_accept_threshold, loss_excess_above_floor as _loss_excess_above_floor, resolve_acceptance_noise_floor_raw as _resolve_acceptance_noise_floor_raw
from .training import train_candidate_model

from ._search_shadow import (
    GREEN,
    RED,
    RESET,
    _STAGEA_UNARY_AST_NODES,
    _analytic_units_rejection,
    _apply_fit_link_to_model,
    _clone_reuse_leaves,
    _loss_str,
    _stageA_compound_buckingham_target_dim,
)
from ._search_detection import (
    _build_compound_candidate_ast,
    _get_qualifying_scaling_vars,
)
from ._search_structure import (
    _build_stageA_composite_closure_ast,
    _loader_n_eff,
    _stageA_cap_terminal_analytic_threshold,
    _stageA_composite_reduces_nn_burden,
    _stageA_noisy_terminal_yspace_accept,
)
from ._search_training import (
    _build_tag_to_leaf_map,
    _eval_yspace_mse,
)

if TYPE_CHECKING:
    from ._search_policy import (
        _accept_threshold_with_structural_target,
        _nn_split_signature,
        _stageA_terminal_closure_committee_gate,
        _stageA_under_protest_threshold_cap,
    )

_COMPOUND_Z_TOKEN = "z"


def _is_compound_token(tok):
    """Return True if tok is a compound variable token: 'z', 'z0', 'z1', etc."""
    if tok == _COMPOUND_Z_TOKEN:
        return True
    return isinstance(tok, str) and tok.startswith("z") and tok[1:].isdigit()


def _stageA_split_group_record_payload(group):
    """Return a JSON-safe split group while preserving compound z tokens."""

    out = []
    for tok in list(group or ()):
        if isinstance(tok, str):
            try:
                out.append(int(tok))
            except (TypeError, ValueError):
                out.append(tok)
            continue
        try:
            out.append(int(tok))
        except Exception:
            out.append(str(tok))
    return out


def _compound_token_index(tok):
    """Return the ordinal index among compound inputs. 'z'->0, 'z0'->0, 'z1'->1."""
    if tok is _COMPOUND_Z_TOKEN or tok == "z":
        return 0
    return int(tok[1:])


def _atom_compound_cols(atom):
    """Derive compound-aware column list and existing z AST from an atom."""
    inputs = get_input_exprs(atom)
    if any(not is_trivial_input(inp) for inp in inputs):
        cols = []
        z_map = {}
        z_count = 0
        for inp in inputs:
            if is_trivial_input(inp):
                cols.append(int(inp.var_idxs[0]))
                continue
            tok = _COMPOUND_Z_TOKEN if z_count == 0 else f"z{z_count}"
            cols.append(tok)
            z_map[tok] = inp
            z_count += 1
        z_ast = z_map[_COMPOUND_Z_TOKEN] if z_count == 1 else z_map
    else:
        z_ast = None
        cols = [int(j) for j in atom.var_idxs]
    return cols, z_ast


def _compound_ast_for_token(z_ast, token):
    """Resolve a compound token against a single AST or token->AST mapping."""
    if z_ast is None:
        raise ValueError("z_ast required when cols contains z-token")
    if isinstance(z_ast, dict):
        if token in z_ast:
            return z_ast[token]
        if token == "z0" and _COMPOUND_Z_TOKEN in z_ast:
            return z_ast[_COMPOUND_Z_TOKEN]
        raise ValueError(f"No compound AST for token {token!r}")
    if isinstance(z_ast, (list, tuple)):
        idx = _compound_token_index(token)
        if 0 <= idx < len(z_ast):
            return z_ast[idx]
        raise ValueError(f"No compound AST for token {token!r}")
    if _compound_token_index(token) != 0:
        raise ValueError(f"No compound AST for token {token!r}")
    return z_ast


def _compound_extra_sort_key(token):
    try:
        if _is_compound_token(token):
            return (0, _compound_token_index(token), str(token))
    except Exception:
        pass
    return (1, 0, str(token))


def _compound_extra_ast_key(ast_node, x_transform_map=None):
    try:
        rendered = ast_to_human_readable(ast_node, x_transform_map)
        if rendered is not None:
            return rendered
    except Exception:
        pass
    try:
        return repr(ast_node)
    except Exception:
        return str(id(ast_node))


def _append_compound_extra_input_asts(out, value, *, seen=None, x_transform_map=None):
    """Append cloneable compound extra ASTs from a node/list/token map.

    Multi-input compound atoms expose their existing compound coordinates as a
    token->AST mapping.  Callers that preserve unconsumed coordinates must
    append the mapped AST nodes, not the mapping object itself.
    """
    if value is None:
        return seen if seen is not None else set()
    if seen is None:
        seen = set()

    if isinstance(value, dict):
        values = [
            value[k]
            for k in sorted(value.keys(), key=_compound_extra_sort_key)
        ]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]

    for ast_node in values:
        if ast_node is None:
            continue
        try:
            cloned = clone_ast(ast_node)
        except Exception:
            continue
        key = _compound_extra_ast_key(cloned, x_transform_map)
        if key in seen:
            continue
        seen.add(key)
        out.append(cloned)
    return seen


def _compound_extra_input_asts_after_prefactor_peel(
    atom,
    extra_input_asts,
    *,
    prefactor_exponents=None,
    prefactor_ast=None,
    x_transform_map=None,
):
    """Drop compound extras that are already exposed as visible prefactors.

    A zero exponent preserves an existing compound coordinate as an NN input.
    If a later Buckingham transaction peels that same coordinate outside the
    leaf, preserving it would produce ``P * NN[z, P]`` instead of the intended
    ``P * NN[z]``.  This is a structural rule and deliberately does not depend
    on whether the coordinate carries physical units.
    """
    extras = []
    _append_compound_extra_input_asts(
        extras,
        extra_input_asts,
        x_transform_map=x_transform_map,
    )
    if not extras:
        return []

    peeled_keys = set()
    if prefactor_ast is not None:
        peeled_keys.add(_compound_extra_ast_key(prefactor_ast, x_transform_map))

    try:
        pref = tuple(int(v) for v in prefactor_exponents)
    except Exception:
        pref = ()
    if pref:
        try:
            local_inputs = tuple(get_input_exprs(atom))
        except Exception:
            local_inputs = ()
        for idx, power in enumerate(pref):
            if int(power) == 0 or idx >= len(local_inputs):
                continue
            peeled_keys.add(
                _compound_extra_ast_key(local_inputs[idx], x_transform_map)
            )

    return [
        expr
        for expr in extras
        if _compound_extra_ast_key(expr, x_transform_map) not in peeled_keys
    ]


def _ast_repr_safe(ast, x_transform_map=None):
    """Best-effort stable text form for AST structural comparisons."""
    try:
        return ast_to_human_readable(ast, x_transform_map)
    except Exception:
        return None


def _is_ast_noop_candidate(current_ast, candidate_ast, x_transform_map=None):
    """True when candidate preserves the exact current AST structure."""
    cur_repr = _ast_repr_safe(current_ast, x_transform_map)
    cand_repr = _ast_repr_safe(candidate_ast, x_transform_map)
    return (cur_repr is not None) and (cand_repr == cur_repr)


def _is_passthrough_noop_candidate(atom, z_expr, extra_var_idxs, x_transform_map=None):
    """Cheap check for passthrough proposals that re-propose the current compound."""
    if not has_nontrivial_input(atom):
        return False
    cur_z = compound_input_expr(atom)
    if cur_z is None:
        return False
    cur_z_repr = _ast_repr_safe(cur_z, x_transform_map)
    new_z_repr = _ast_repr_safe(z_expr, x_transform_map)
    if (cur_z_repr is None) or (new_z_repr is None) or (cur_z_repr != new_z_repr):
        return False
    try:
        cur_extras = tuple(sorted(int(v) for v in (extra_input_var_idxs(atom) or ())))
        new_extras = tuple(sorted(int(v) for v in (extra_var_idxs or ())))
    except Exception:
        return False
    return cur_extras == new_extras


def _compound_candidate_default_extra_var_idxs(atom, pattern) -> List[int]:
    """Return raw extra variables represented by zero entries in a local pattern."""
    if has_nontrivial_input(atom):
        try:
            cols, _ = _atom_compound_cols(atom)
        except Exception:
            cols = list(getattr(atom, "var_idxs", ()) or ())
        out: List[int] = []
        for i, v in enumerate(pattern):
            if not _compound_pattern_entry_is_zero(v):
                continue
            if i >= len(cols):
                continue
            col = cols[i]
            if isinstance(col, int):
                out.append(int(col))
        return sorted(set(out))

    out: List[int] = []
    for i, v in enumerate(pattern):
        try:
            if not _compound_pattern_entry_is_zero(v):
                continue
            out.append(int(atom.var_idxs[i]))
        except Exception:
            continue
    return sorted(set(out))


def _compound_pattern_entry_is_zero(value) -> bool:
    """Return True for pattern entries that denote a preserved extra input."""
    try:
        return int(value) == 0
    except Exception:
        return False


def _compound_proposal_support_arity(pattern, z_ast=None, meta=None) -> int:
    """Count effective input columns participating in a compound proposal.

    This is deliberately based on the proposal pattern, not on raw variables in
    the AST. For an already-compressed atom, a proposal like z0*z1 should count
    as arity 2 even if z0 and z1 each contain multiple raw variables.
    """
    try:
        return int(sum(1 for v in pattern if not _compound_pattern_entry_is_zero(v)))
    except Exception:
        pass

    # Defensive fallback for unusual proposal kinds without a usable pattern.
    vars_seen = set()

    def rec(n):
        try:
            if isinstance(n, AtomNode):
                k = str(getattr(n, "kind", "")).lower()
                if k in ("var", "x", "input") and getattr(n, "var_idxs", None):
                    vars_seen.add(int(n.var_idxs[0]))
                if has_nontrivial_input(n):
                    for inp in get_input_exprs(n):
                        rec(inp)
                return
            if isinstance(n, ConstNode):
                return
            if isinstance(n, (AddNode, MulNode)):
                rec(n.left)
                rec(n.right)
                return
            if isinstance(n, PowNode):
                rec(n.base)
                return
            arg = getattr(n, "arg", None)
            if arg is not None:
                rec(arg)
                return
        except Exception:
            return

    rec(z_ast)
    return int(len(vars_seen))


def _compound_proposal_extra_ast_count(meta) -> int:
    if not isinstance(meta, dict):
        return 0
    try:
        return int(len(tuple(meta.get("extra_input_asts", ()) or ())))
    except Exception:
        return 0


def _compound_proposal_new_arity_for_sort(pattern, extra_override=None, meta=None) -> int:
    """Estimate candidate NN arity for proposal ordering/logging.

    For ordinary raw-coordinate compounds, this agrees with the structural
    arity reduction implied by the proposal pattern.  For visible-prefactor
    partial peels, it keeps the sorter focused on the residual NN burden rather
    than the placeholder residual-z support, which is always one.
    """
    try:
        if extra_override is not None:
            raw_extra_count = int(len(tuple(extra_override)))
        else:
            raw_extra_count = int(
                sum(1 for value in pattern if _compound_pattern_entry_is_zero(value))
            )
    except Exception:
        raw_extra_count = 0
    return int(1 + raw_extra_count + _compound_proposal_extra_ast_count(meta))


def _compound_proposal_prefactor_support(meta) -> int:
    if not isinstance(meta, dict):
        return 0
    pref = meta.get("prefactor_exponents")
    if pref is None and meta.get("prefactor_ast") is not None:
        try:
            return int(len(set(int(v) for v in _collect_var_idxs_from_node(meta.get("prefactor_ast")))))
        except Exception:
            return 1
    if pref is None:
        return 0
    try:
        return int(sum(1 for value in tuple(pref) if int(value) != 0))
    except Exception:
        return 0


def _compound_proposal_sort_key(prop):
    pattern, z_ast, confidence, extra_override, meta = prop
    meta = meta if isinstance(meta, dict) else {}
    has_visible_prefactor = 1 if (
        meta.get("prefactor_exponents") is not None or meta.get("prefactor_ast") is not None
    ) else 0
    is_partial_peel = 1 if bool(meta.get("partial_forced_monomial_peel", False)) else 0
    support = _compound_proposal_support_arity(pattern, z_ast, meta)
    new_arity = _compound_proposal_new_arity_for_sort(pattern, extra_override, meta)
    pref_support = _compound_proposal_prefactor_support(meta)
    structural_gain = pref_support if is_partial_peel else support
    return (
        has_visible_prefactor,
        structural_gain,
        -int(new_arity),
        is_partial_peel,
        float(confidence),
    )


def _compound_proposal_brief(
    pattern,
    z_ast,
    confidence,
    meta,
    *,
    extra_override=None,
    x_transform_map=None,
) -> str:
    """Concise compound proposal description for Stage-A audit logs."""
    meta = meta or {}
    kind = str(meta.get("kind", "compound"))
    family = meta.get("metric_family") or meta.get("family") or meta.get("form")
    wrapper = meta.get("metric_wrapper") or meta.get("wrapper")
    try:
        support = _compound_proposal_support_arity(pattern, z_ast, meta)
    except Exception:
        support = "?"
    try:
        z_desc = ast_to_human_readable(z_ast, x_transform_map)
    except Exception:
        z_desc = str(z_ast)
    details = f"kind={kind}"
    if family is not None:
        details += f", family={family}"
    if wrapper is not None:
        details += f", wrapper={wrapper}"
    if bool(meta.get("partial_forced_monomial_peel", False)):
        details += (
            f", new_arity={_compound_proposal_new_arity_for_sort(pattern, extra_override, meta)}"
            f", peeled={_compound_proposal_prefactor_support(meta)}"
            f", residual_local={meta.get('residual_local_indices')}"
        )
    return (
        f"{details}, conf={float(confidence):.3f}, "
        f"support={support}, z={z_desc}"
    )


def _log_compound_proposal_shortlist(normed_proposals, *, x_transform_map=None) -> None:
    """Print the ordered Stage-A compound shortlist after sorting/capping."""
    if not normed_proposals:
        return
    print("[Compound] Normalized proposal shortlist:")
    for i, (pattern, z_ast, confidence, _extra_override, meta) in enumerate(normed_proposals, start=1):
        print(
            f"[Compound]   {i}. "
            + _compound_proposal_brief(
                pattern,
                z_ast,
                confidence,
                meta,
                extra_override=_extra_override,
                x_transform_map=x_transform_map,
            )
        )


def _compound_best_proposal_confidence(proposals) -> float:
    """Return the best confidence in a mixed list of compound proposals."""
    best = 0.0
    for proposal in proposals or ():
        try:
            if len(proposal) >= 3:
                best = max(best, float(proposal[2]))
        except Exception:
            continue
    return float(best)


def _phase_hint_compound_proposals_for_atom(phase_hints, atom, *, min_confidence: float):
    """Convert Stage-0 phase hints into ordinary Stage-A monomial proposals.

    Phase hints are proposal evidence only.  They are injected as normal
    monomial compound candidates so the existing LM validation, units checks,
    iso-z sanity gate, and gauge-aware acceptance policy remain in control.
    """

    if not phase_hints:
        return []
    if has_nontrivial_input(atom):
        return []

    try:
        atom_vars = tuple(int(v) for v in getattr(atom, "var_idxs", ()) or ())
    except Exception:
        return []
    if not atom_vars:
        return []
    atom_var_set = set(atom_vars)
    atom_pos = {int(v): i for i, v in enumerate(atom_vars)}

    out = []
    seen = set()
    for hint in list(phase_hints or [])[:16]:
        details = getattr(hint, "details", {}) or {}
        exp_rows = details.get("exponents", ())
        parsed = []
        for row in exp_rows:
            try:
                idx = int(row[0])
                exp = Fraction(str(row[1]))
            except Exception:
                parsed = []
                break
            # Stage-A compound patterns currently encode integer monomials.
            # Half-power phase carriers remain Stage-B frequency hints for now.
            if exp.denominator != 1:
                parsed = []
                break
            parsed.append((idx, int(exp)))
        if not parsed:
            continue
        support = {idx for idx, exp in parsed if int(exp) != 0}
        if not support or not support.issubset(atom_var_set):
            continue

        pattern = [0 for _ in atom_vars]
        for idx, exp in parsed:
            if idx in atom_pos:
                pattern[atom_pos[idx]] = int(exp)
        if sum(1 for v in pattern if int(v) != 0) < 1:
            continue

        try:
            z_ast = clone_ast(getattr(hint, "carrier_ast"))
        except Exception:
            z_ast = getattr(hint, "carrier_ast", None)
        if z_ast is None:
            continue

        key = (tuple(pattern), ast_to_human_readable(z_ast))
        if key in seen:
            continue
        seen.add(key)

        try:
            phase_score = float(getattr(hint, "score", 0.0))
        except Exception:
            phase_score = 0.0
        try:
            phase_conf = float(getattr(hint, "confidence", phase_score))
        except Exception:
            phase_conf = phase_score
        phase_strength = max(0.0, min(1.0, max(phase_score, phase_conf)))
        conf = min(0.999, 0.50 + 0.499 * math.sqrt(phase_strength))
        if conf < float(min_confidence):
            continue
        meta = {
            "kind": "monomial",
            "family": "phase_hint",
            "phase_hint": True,
            "source": "phase_prescan",
            "phase_score": float(phase_score),
            "phase_confidence": float(phase_conf),
            "phase_observed_omega": getattr(hint, "observed_omega", None),
            "phase_unit_status": getattr(hint, "unit_status", "unchecked"),
        }
        out.append((tuple(pattern), z_ast, float(conf), None, meta))

    return out


def _clean_monomial_product_proposal_from_scaling(
    scale_specs,
    cols,
    *,
    z_ast_existing=None,
    rel_std_threshold=0.05,
    k_int_threshold=0.15,
):
    """Build the maximal clean monomial-product proposal from scaling hints.

    This is a proposal-lane helper, not an acceptance shortcut.  It keeps the
    clean product of axes whose single-variable scaling exponents look like a
    common small-integer monomial, and leaves all other axes as NN extras.  That
    is the conservative move needed for cases such as
    ``NN[x0..x6] -> NN[x0*x1*x2*x3*x4/x5, x6]``.
    """
    if not scale_specs or not cols:
        return None

    raw_cols = []
    for c in cols:
        if isinstance(c, int):
            raw_cols.append(int(c))
    if len(raw_cols) < 2:
        return None

    qualifying = _get_qualifying_scaling_vars(
        scale_specs,
        var_filter=set(raw_cols),
        rel_std_threshold=float(rel_std_threshold),
        k_int_threshold=float(k_int_threshold),
        require_oracle=False,
    )
    if len(qualifying) < 2:
        return None

    pattern = []
    for col in cols:
        if _is_compound_token(col):
            pattern.append(0)
        else:
            pattern.append(int(qualifying.get(int(col), 0)))

    if sum(1 for v in pattern if int(v) != 0) < 2:
        return None

    try:
        z_ast = _build_monomial_ast_from_cols(cols, tuple(pattern), z_ast=z_ast_existing)
    except ValueError:
        return None

    rels = []
    for sp in scale_specs:
        try:
            if len(sp.indices) != 1:
                continue
            idx = int(sp.indices[0])
            if idx not in qualifying:
                continue
            rels.append(float(sp.rel_std))
        except Exception:
            continue
    rel_max = max(rels) if rels else float(rel_std_threshold)
    conf = max(0.0, min(0.999, 1.0 - float(rel_max)))

    extras = tuple(
        int(col) for col, exp in zip(cols, pattern)
        if isinstance(col, int) and int(exp) == 0
    )
    meta = {
        "kind": "monomial",
        "clean_monomial_product": True,
        "source": "scaling_clean_product",
        "clean_rel_std_max": float(rel_max),
        "clean_support": tuple(int(k) for k in sorted(qualifying)),
    }
    if extras:
        meta["clean_extras"] = extras
    return (tuple(pattern), z_ast, float(conf), None, meta)


def _stageA_scaling_spec_k_rel(sp) -> Optional[Tuple[float, float]]:
    """Best-effort homogeneity degree/scatter from a ScaleSpec-like object."""
    try:
        if bool(getattr(sp, "oracle_verified", False)) and getattr(sp, "oracle_k", None) is not None:
            k_val = float(getattr(sp, "oracle_k"))
        else:
            k_val = float(getattr(sp, "k_hat"))
        if bool(getattr(sp, "oracle_verified", False)) and getattr(sp, "oracle_rel_std", None) is not None:
            rel = float(getattr(sp, "oracle_rel_std"))
        else:
            rel = float(getattr(sp, "rel_std"))
    except Exception:
        return None
    if not (math.isfinite(k_val) and math.isfinite(rel)):
        return None
    return k_val, max(0.0, rel)


def _stageA_noisy_soft_monomial_product_proposals_from_scaling(
    scale_specs,
    cols,
    *,
    z_ast_existing=None,
    search_hp=None,
    noise_floor_raw: float = 0.0,
):
    """Build bounded noisy-prior monomial proposals from near-integer scaling hints.

    The deterministic clean-product detector is a certificate lane.  This helper
    is a proposal lane: it is active only when an explicit noise floor exists,
    keeps low-complexity primitive integer monomials suggested by noisy
    homogeneity diagnostics, and relies on normal Stage-A/CoE validation for
    acceptance.
    """
    try:
        nf = float(noise_floor_raw)
    except Exception:
        nf = 0.0
    if not (math.isfinite(nf) and nf > 0.0):
        return []
    if not scale_specs or not cols:
        return []

    raw_cols = [int(c) for c in cols if isinstance(c, int)]
    if len(raw_cols) < 2:
        return []
    raw_set = set(raw_cols)
    pos = {int(c): i for i, c in enumerate(cols) if isinstance(c, int)}
    if len(pos) < 2:
        return []

    max_abs = int(getattr(search_hp, "noisy_soft_monomial_max_abs_power", 6))
    max_l1 = int(getattr(search_hp, "noisy_soft_monomial_max_l1", 10))
    max_support = int(getattr(search_hp, "noisy_soft_monomial_max_support", 5))
    max_candidates = int(getattr(search_hp, "noisy_soft_monomial_max_candidates_per_atom", 2))
    rel_single_max = float(getattr(search_hp, "noisy_soft_monomial_rel_std", 0.12))
    rel_group_max = float(getattr(search_hp, "noisy_soft_monomial_group_rel_std", rel_single_max))
    k_int_tol = float(getattr(search_hp, "noisy_soft_monomial_k_int", 0.25))
    group_resid_tol = float(getattr(search_hp, "noisy_soft_monomial_group_resid", 0.35))
    if max_candidates <= 0:
        return []

    singles: Dict[int, Tuple[int, float, float, float]] = {}
    group_specs: list[Tuple[Tuple[int, ...], float, float]] = []
    for sp in scale_specs:
        try:
            idxs = tuple(int(v) for v in getattr(sp, "indices", ()) or ())
        except Exception:
            continue
        if not idxs or not set(idxs).issubset(raw_set):
            continue
        vals = _stageA_scaling_spec_k_rel(sp)
        if vals is None:
            continue
        k_val, rel = vals
        n = int(round(k_val))
        resid = abs(float(k_val) - float(n))
        if len(idxs) == 1:
            if rel > rel_single_max or resid > k_int_tol or n == 0 or abs(n) > max_abs:
                continue
            idx = int(idxs[0])
            prev = singles.get(idx)
            score = float(rel + 0.25 * resid)
            if prev is None or score < float(prev[3]):
                singles[idx] = (int(n), float(rel), float(resid), float(score))
        elif len(idxs) <= max_support + 1:
            if rel <= rel_group_max and resid <= group_resid_tol:
                group_specs.append((tuple(sorted(idxs)), float(k_val), float(rel)))

    if len(singles) < 2 and not group_specs:
        return []

    def _candidate_from_known(known: Dict[int, Tuple[int, float, float, str]]):
        pattern = []
        for col in cols:
            if isinstance(col, int):
                pattern.append(int(known.get(int(col), (0, 0.0, 0.0, ""))[0]))
            else:
                pattern.append(0)
        support = sum(1 for v in pattern if int(v) != 0)
        if support < 2 or support > max_support:
            return None
        l1 = sum(abs(int(v)) for v in pattern)
        if l1 > max_l1:
            return None
        if max(abs(int(v)) for v in pattern) > max_abs:
            return None
        nz = [abs(int(v)) for v in pattern if int(v) != 0]
        try:
            from functools import reduce
            from math import gcd

            g = reduce(gcd, nz) if nz else 1
        except Exception:
            g = 1
        if g > 1:
            pattern = [int(v) // int(g) for v in pattern]
            l1 = sum(abs(int(v)) for v in pattern)
        try:
            z_ast = _build_monomial_ast_from_cols(cols, tuple(pattern), z_ast=z_ast_existing)
        except ValueError:
            return None
        evidence = []
        rels = []
        residuals = []
        inferred_count = 0
        for idx, (exp, rel, resid, source) in sorted(known.items()):
            if idx not in pos:
                continue
            evidence.append(
                {
                    "var": int(idx),
                    "exp": int(exp),
                    "rel_std": float(rel),
                    "integer_residual": float(resid),
                    "source": str(source),
                }
            )
            rels.append(float(rel))
            residuals.append(float(resid))
            if str(source).startswith("group"):
                inferred_count += 1
        rel_max = max(rels) if rels else rel_single_max
        resid_max = max(residuals) if residuals else k_int_tol
        score = float(0.5 * rel_max + resid_max + 0.03 * l1 + 0.08 * support - 0.05 * inferred_count)
        conf = float(max(0.86, min(0.985, 1.0 - 0.5 * rel_max - 0.25 * resid_max + 0.02 * inferred_count)))
        extras = tuple(
            int(col) for col, exp in zip(cols, pattern)
            if isinstance(col, int) and int(exp) == 0
        )
        meta = {
            "kind": "monomial",
            "family": "noisy_soft_monomial_compound",
            "source": "noisy_scaling_prior",
            "soft_monomial_compound": True,
            "structural_protected": True,
            "q_exponents": tuple(int(v) for v in pattern),
            "q_l1": int(l1),
            "q_support": int(support),
            "soft_scaling_rel_std_max": float(rel_max),
            "soft_scaling_integer_resid_max": float(resid_max),
            "soft_scaling_inferred_count": int(inferred_count),
            "proposal_score": float(score),
            "soft_scaling_evidence": tuple(evidence),
        }
        if extras:
            meta["soft_extras"] = extras
        return (tuple(int(v) for v in pattern), z_ast, conf, None, meta, score)

    base_known: Dict[int, Tuple[int, float, float, str]] = {
        idx: (int(row[0]), float(row[1]), float(row[2]), "single")
        for idx, row in singles.items()
    }
    candidates = []
    base = _candidate_from_known(base_known)
    if base is not None:
        candidates.append(base)

    expanded = dict(base_known)
    changed = True
    while changed:
        changed = False
        for idxs, k_val, rel in group_specs:
            known_sum = 0
            missing = []
            for idx in idxs:
                if idx in expanded:
                    known_sum += int(expanded[idx][0])
                else:
                    missing.append(int(idx))
            if len(missing) != 1:
                continue
            n_missing = int(round(float(k_val) - float(known_sum)))
            if n_missing == 0 or abs(n_missing) > max_abs:
                continue
            resid = abs((float(known_sum) + float(n_missing)) - float(k_val))
            if resid > group_resid_tol:
                continue
            expanded[int(missing[0])] = (
                int(n_missing),
                float(rel),
                float(resid),
                f"group:{','.join(str(i) for i in idxs)}",
            )
            changed = True

    expanded_cand = _candidate_from_known(expanded)
    if expanded_cand is not None:
        candidates.append(expanded_cand)

    out = []
    seen = set()
    for cand in sorted(candidates, key=lambda row: (-sum(1 for v in row[0] if int(v) != 0), row[5])):
        key = tuple(int(v) for v in cand[0])
        if key in seen:
            continue
        seen.add(key)
        out.append(cand[:5])
        if len(out) >= max_candidates:
            break
    return out


def _stageA_append_noisy_soft_monomial_compound_proposals(
    proposals,
    *,
    atom: AtomNode,
    scaling_features,
    search_hp=None,
    lm_hp=None,
    loss_scale: float = 1.0,
) -> list:
    """Append bounded noisy-prior monomial proposals for the current atom."""
    try:
        noise_floor_raw = float(_resolve_acceptance_noise_floor_raw(lm_hp, float(loss_scale)))
    except Exception:
        noise_floor_raw = 0.0
    if not (math.isfinite(noise_floor_raw) and noise_floor_raw > 0.0):
        return list(proposals or [])
    try:
        cols, z_ast_existing = _atom_compound_cols(atom)
    except Exception:
        return list(proposals or [])

    extra = _stageA_noisy_soft_monomial_product_proposals_from_scaling(
        scaling_features,
        cols,
        z_ast_existing=z_ast_existing,
        search_hp=search_hp,
        noise_floor_raw=noise_floor_raw,
    )
    if not extra:
        return list(proposals or [])
    out = list(proposals or [])
    seen = set()
    for prop in out:
        try:
            seen.add((tuple(int(v) for v in prop[0]), _stageA_ast_fingerprint(prop[1])))
        except Exception:
            continue
    added = 0
    for prop in extra:
        try:
            key = (tuple(int(v) for v in prop[0]), _stageA_ast_fingerprint(prop[1]))
        except Exception:
            key = None
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        out.append(prop)
        added += 1
    if added:
        print(
            f"[Compound SoftMonomial] Added {added} noisy scaling-prior "
            "monomial proposal(s)."
        )
    return out


def _shortlist_compound_proposals_with_pair_backup(normed_proposals, max_proposals_to_try: int):
    """Apply the greedy proposal cap while preserving clean and arity-2 lanes.

    The primary slice is exactly the current ranked list capped at
    ``compound_max_proposals_to_try``.  We then append a tiny clean-product lane
    and a same-ranked backup slice restricted to effective-input arity 2.  This
    keeps Stage A greedy while preventing a family of high-arity variants from
    starving clean monomial products or conservative pair compounds such as
    x2*x3 or z0*z1.
    """
    try:
        cap = int(max_proposals_to_try)
    except Exception:
        cap = 0
    if cap <= 0 or len(normed_proposals) <= cap:
        return list(normed_proposals)

    primary = list(normed_proposals[:cap])
    clean_backup = []
    soft_backup = []
    visible_prefactor_backup = []
    partial_backup = []
    for prop in normed_proposals:
        meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
        if bool(meta.get("retained_axis_wrapper", False)):
            continue
        if not bool(meta.get("clean_monomial_product", False)):
            continue
        clean_backup.append(prop)
        if len(clean_backup) >= cap:
            break

    soft_lane_k = min(2, cap)
    for prop in normed_proposals:
        meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
        if bool(meta.get("retained_axis_wrapper", False)):
            continue
        if not bool(meta.get("soft_monomial_compound", False)):
            continue
        soft_backup.append(prop)
        if len(soft_backup) >= soft_lane_k:
            break

    visible_prefactor_lane_k = min(2, cap)
    for prop in normed_proposals:
        meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
        if bool(meta.get("retained_axis_wrapper", False)):
            continue
        if not (
            meta.get("prefactor_exponents") is not None
            or meta.get("prefactor_ast") is not None
            or bool(meta.get("visible_prefactor_transaction", False))
        ):
            continue
        visible_prefactor_backup.append(prop)
        if len(visible_prefactor_backup) >= visible_prefactor_lane_k:
            break

    for prop in normed_proposals:
        meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
        if bool(meta.get("retained_axis_wrapper", False)):
            continue
        if not bool(meta.get("partial_forced_monomial_peel", False)):
            continue
        partial_backup.append(prop)
        if len(partial_backup) >= cap:
            break

    pair_backup = []
    for prop in normed_proposals:
        meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
        if bool(meta.get("retained_axis_wrapper", False)):
            continue
        if _compound_proposal_support_arity(prop[0], prop[1], meta) != 2:
            continue
        pair_backup.append(prop)
        if len(pair_backup) >= cap:
            break

    # Metric-distance backup lane. Law-of-cosines / Euclidean-distance forms
    # (family lawcos, lawcos_sq_*, cartesian) are arity-3+ high-value structural
    # carriers whose sqrt-closure only fits in the untransformed y-space.  They
    # can be pushed below the primary cap by high-confidence dimensionless
    # prefactor probes, so preserve a few as appended backups the same way the
    # clean-monomial and arity-2 lanes do.
    metric_lane_k = min(6, cap)
    metric_backup = []
    for prop in normed_proposals:
        meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
        if bool(meta.get("retained_axis_wrapper", False)):
            continue
        if str(meta.get("kind", "")) != "metric_distance":
            continue
        metric_backup.append(prop)
        if len(metric_backup) >= metric_lane_k:
            break

    shortlist = []
    seen = set()

    def _stable_pattern_key(pattern):
        out = []
        for value in pattern:
            try:
                fv = float(value)
                if fv.is_integer():
                    out.append(int(fv))
                else:
                    out.append(round(fv, 12))
            except Exception:
                out.append(str(value))
        return tuple(out)

    def _proposal_key(prop):
        meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
        extra = prop[3] if len(prop) > 3 else None
        pref = meta.get("prefactor_exponents") if isinstance(meta, dict) else None
        try:
            pref_key = None if pref is None else tuple(int(v) for v in pref)
        except Exception:
            pref_key = repr(pref)
        try:
            z_key = ast_to_human_readable(prop[1])
        except Exception:
            z_key = repr(prop[1])
        return (
            _stable_pattern_key(prop[0]),
            tuple(int(v) for v in extra) if extra is not None else None,
            pref_key,
            bool(meta.get("retained_axis_wrapper", False)) if isinstance(meta, dict) else False,
            meta.get("retained_axis") if isinstance(meta, dict) else None,
            z_key,
        )

    added_clean = 0
    added_soft = 0
    added_visible_prefactor = 0
    added_partial = 0
    added_pair = 0
    added_metric = 0
    for prop in primary:
        key = _proposal_key(prop)
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(prop)

    for prop in clean_backup:
        key = _proposal_key(prop)
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(prop)
        added_clean += 1

    for prop in soft_backup:
        key = _proposal_key(prop)
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(prop)
        added_soft += 1

    for prop in visible_prefactor_backup:
        key = _proposal_key(prop)
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(prop)
        added_visible_prefactor += 1

    for prop in partial_backup:
        key = _proposal_key(prop)
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(prop)
        added_partial += 1

    for prop in pair_backup:
        key = _proposal_key(prop)
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(prop)
        added_pair += 1

    for prop in metric_backup:
        key = _proposal_key(prop)
        if key in seen:
            continue
        seen.add(key)
        shortlist.append(prop)
        added_metric += 1

    if added_clean:
        print(
            f"[Compound] Preserving {added_clean} clean monomial-product proposal(s) "
            "beyond primary compound shortlist."
        )
    if added_soft:
        print(
            f"[Compound] Preserving {added_soft} noisy soft-monomial "
            "proposal(s) beyond primary compound shortlist."
        )
    if added_visible_prefactor:
        print(
            f"[Compound] Preserving {added_visible_prefactor} visible-prefactor "
            "proposal(s) beyond primary compound shortlist."
        )
    if added_partial:
        print(
            f"[Compound] Preserving {added_partial} partial monomial-peel "
            "proposal(s) beyond primary compound shortlist."
        )
    if added_pair:
        print(
            f"[Compound] Preserving {added_pair} arity-2 backup "
            "proposal(s) beyond primary compound shortlist."
        )
    if added_metric:
        print(
            f"[Compound] Preserving {added_metric} metric-distance backup "
            "proposal(s) beyond primary compound shortlist."
        )
    return shortlist


def _stageA_schedule_gs_compound_lanes(
    normed_proposals,
    *,
    max_ordinary_proposals: int,
    max_gs_proposals: int,
    decisive_min_confidence: float,
    decisive_max_trials: int = 1,
):
    """Build protected decisive/ordinary/fallback compound trial lanes.

    GS evidence controls only scheduling. Every returned proposal still passes
    through the ordinary wrapper, LM, validation, and structural-payoff gates.
    The ordinary shortlist is constructed exactly as it is with GS absent, so
    GS cannot consume its budget or change its ordering.
    """

    ordinary = []
    gs_candidates = []
    for proposal in list(normed_proposals or ()):
        meta = proposal[4] if len(proposal) >= 5 and isinstance(proposal[4], dict) else {}
        if str(meta.get("source", "")) == "generalized_symmetry":
            gs_candidates.append(proposal)
        else:
            ordinary.append(proposal)

    try:
        ordinary.sort(key=_compound_proposal_sort_key, reverse=True)
    except Exception:
        ordinary.sort(key=lambda row: float(row[2]), reverse=True)
    ordinary_shortlist = _shortlist_compound_proposals_with_pair_backup(
        ordinary,
        int(max_ordinary_proposals),
    )

    try:
        gs_candidates.sort(key=_compound_proposal_sort_key, reverse=True)
    except Exception:
        gs_candidates.sort(key=lambda row: float(row[2]), reverse=True)

    def _decisive(prop) -> bool:
        try:
            pattern, _ast, confidence, extra_override, meta = prop
            pattern = tuple(int(v) for v in pattern)
        except Exception:
            return False
        if len(pattern) <= 1 or not all(value != 0 for value in pattern):
            return False
        if extra_override:
            return False
        if float(confidence) < float(decisive_min_confidence):
            return False
        if not bool(meta.get("carrier_certified", False)):
            return False
        if str(meta.get("candidate_role", "")) != "inner_coordinate":
            return False
        if meta.get("extra_input_asts"):
            return False
        if meta.get("prefactor_exponents") is not None or meta.get("prefactor_ast") is not None:
            return False
        if abs(float(meta.get("gs_output_alpha", 0.0) or 0.0)) > 1.0e-10:
            return False
        if abs(float(meta.get("gs_output_beta", 0.0) or 0.0)) > 1.0e-10:
            return False
        return True

    def _with_lane(prop, lane: str):
        meta = dict(prop[4]) if len(prop) >= 5 and isinstance(prop[4], dict) else {}
        meta["gs_stagea_lane"] = str(lane)
        return (prop[0], prop[1], prop[2], prop[3], meta)

    gs_budget = max(0, int(max_gs_proposals))
    decisive = []
    remaining = []
    decisive_limit = min(gs_budget, max(0, int(decisive_max_trials)))
    for proposal in gs_candidates:
        if len(decisive) < decisive_limit and _decisive(proposal):
            decisive.append(_with_lane(proposal, "decisive"))
        else:
            remaining.append(proposal)
    # Unlike the ordinary lane, GS has a strict proposal budget: do not invoke
    # the legacy backup-preservation helper, which may deliberately exceed its
    # nominal cap to protect ordinary proposal families.
    fallback = remaining[: max(0, gs_budget - len(decisive))]
    fallback = [_with_lane(proposal, "fallback") for proposal in fallback]
    return decisive, ordinary_shortlist, fallback


def _stageA_json_key(value: Any) -> str:
    """Stable JSON key for structural fingerprints used by CoE replay."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return repr(value)


def _stageA_const_fingerprint(value: Any) -> Any:
    try:
        f = float(value)
        if math.isfinite(f):
            if abs(f - round(f)) <= 1e-12:
                return int(round(f))
            return round(f, 12)
    except Exception:
        pass
    return str(value)


def _stageA_ast_fp_obj(node: Any, *, target_atom: Optional[AtomNode] = None) -> Any:
    """Canonical AST fingerprint.

    This intentionally ignores fitted leaf tags while preserving ordered NN input
    expressions.  Add/Mul children are sorted because those operations are
    commutative; parent input lists remain ordered separately.
    """
    if target_atom is not None and node is target_atom:
        try:
            in_fps = [_stageA_ast_fp_obj(inp) for inp in get_input_exprs(node)]
        except Exception:
            in_fps = []
        return ["hole", str(getattr(node, "kind", "atom")).lower(), in_fps]
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input"):
            try:
                return ["var", int(getattr(node, "var_idxs", ())[0])]
            except Exception:
                return ["var", repr(getattr(node, "var_idxs", ()))]
        try:
            in_fps = [_stageA_ast_fp_obj(inp) for inp in get_input_exprs(node)]
        except Exception:
            in_fps = []
        kwargs = getattr(node, "kwargs", {}) or {}
        stable_kwargs = {}
        for key, val in sorted(kwargs.items(), key=lambda kv: str(kv[0])):
            if str(key) in {"tag", "name", "init"} or str(key).startswith("_"):
                continue
            try:
                json.dumps(val, sort_keys=True, default=str)
                stable_kwargs[str(key)] = val
            except Exception:
                stable_kwargs[str(key)] = str(val)
        return [
            "atom",
            kind,
            str(getattr(node, "scope", "experiment")),
            in_fps,
            stable_kwargs,
        ]
    if isinstance(node, ConstNode):
        return ["const", _stageA_const_fingerprint(getattr(node, "value", None))]
    if isinstance(node, AddNode):
        children = [_stageA_ast_fp_obj(node.left), _stageA_ast_fp_obj(node.right)]
        return ["add", sorted(children, key=_stageA_json_key)]
    if isinstance(node, MulNode):
        children = [_stageA_ast_fp_obj(node.left), _stageA_ast_fp_obj(node.right)]
        return ["mul", sorted(children, key=_stageA_json_key)]
    if isinstance(node, PowNode):
        return [
            "pow",
            _stageA_ast_fp_obj(node.base),
            _stageA_const_fingerprint(getattr(node, "exponent", None)),
        ]
    for cls, name in (
        (LogNode, "log"),
        (ExpNode, "exp"),
        (SinNode, "sin"),
        (CosNode, "cos"),
        (AsinNode, "asin"),
        (AcosNode, "acos"),
        (AtanNode, "atan"),
    ):
        if isinstance(node, cls):
            return [name, _stageA_ast_fp_obj(node.arg)]
    return ["unknown", type(node).__name__, repr(node)]


def _stageA_ast_fingerprint(node: Any, *, target_atom: Optional[AtomNode] = None) -> str:
    return _stageA_json_key(_stageA_ast_fp_obj(node, target_atom=target_atom))


def _stageA_ast_to_payload(node: Any) -> Optional[Any]:
    """Serialize a safe analytic input-expression AST for CoE replay."""
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input"):
            try:
                return ["var", int(getattr(node, "var_idxs", ())[0])]
            except Exception:
                return None
        return None
    if isinstance(node, ConstNode):
        return ["const", _stageA_const_fingerprint(getattr(node, "value", None))]
    if isinstance(node, AddNode):
        left = _stageA_ast_to_payload(node.left)
        right = _stageA_ast_to_payload(node.right)
        return None if left is None or right is None else ["add", left, right]
    if isinstance(node, MulNode):
        left = _stageA_ast_to_payload(node.left)
        right = _stageA_ast_to_payload(node.right)
        return None if left is None or right is None else ["mul", left, right]
    if isinstance(node, PowNode):
        base = _stageA_ast_to_payload(node.base)
        if base is None:
            return None
        return ["pow", base, _stageA_const_fingerprint(getattr(node, "exponent", None))]
    for cls, name in (
        (LogNode, "log"),
        (ExpNode, "exp"),
        (SinNode, "sin"),
        (CosNode, "cos"),
        (AsinNode, "asin"),
        (AcosNode, "acos"),
        (AtanNode, "atan"),
    ):
        if isinstance(node, cls):
            arg = _stageA_ast_to_payload(node.arg)
            return None if arg is None else [name, arg]
    return None


def _stageA_ast_from_payload(payload: Any) -> Optional[Node]:
    if not isinstance(payload, (list, tuple)) or not payload:
        return None
    tag = str(payload[0])
    try:
        if tag == "var":
            return Var(int(payload[1]))
        if tag == "const":
            return ConstNode(payload[1])
        if tag in {"add", "mul"}:
            left = _stageA_ast_from_payload(payload[1])
            right = _stageA_ast_from_payload(payload[2])
            if left is None or right is None:
                return None
            return AddNode(left, right) if tag == "add" else MulNode(left, right)
        if tag == "pow":
            base = _stageA_ast_from_payload(payload[1])
            if base is None:
                return None
            return PowNode(base, float(payload[2]))
        unary = {
            "log": LogNode,
            "exp": ExpNode,
            "sin": SinNode,
            "cos": CosNode,
            "asin": AsinNode,
            "acos": AcosNode,
            "atan": AtanNode,
        }
        if tag in unary:
            arg = _stageA_ast_from_payload(payload[1])
            return None if arg is None else unary[tag](arg)
    except Exception:
        return None
    return None


def _stageA_dim_key(dim: Any) -> Any:
    if dim is None:
        return None
    try:
        vals = list(dim)
        return [_stageA_const_fingerprint(v) for v in vals]
    except Exception:
        return str(dim)


def _stageA_x_transform_fingerprint(x_transform_map: Any) -> str:
    if not x_transform_map:
        return "none"
    try:
        return _stageA_json_key(x_transform_map)
    except Exception:
        return repr(x_transform_map)


def _stageA_current_y_transform_name(lm_hp: Any, search_hp: Any = None) -> str:
    for obj in (lm_hp, search_hp):
        if obj is None:
            continue
        for attr in ("coe_current_y_transform_name", "current_y_transform_name", "y_transform_name"):
            value = getattr(obj, attr, None)
            if value:
                return str(value)
    return "identity"


def _stageA_fit_link_context(lm_hp: Any) -> dict[str, Any]:
    name = canonical_fit_link_name(getattr(lm_hp, "fit_y_link", None))
    scale = getattr(lm_hp, "fit_y_link_scale", None)
    try:
        scale = None if scale is None else round(float(scale), 12)
    except Exception:
        scale = str(scale)
    return {"fit_y_link": name, "fit_y_link_scale": scale}


def _stageA_parent_context_descriptor(
    current_ast: Any,
    atom: AtomNode,
    *,
    units_spec=None,
    x_transform_map=None,
) -> dict[str, Any]:
    inputs = tuple(get_input_exprs(atom))
    output_dim = None
    if units_spec is not None:
        try:
            output_dim = _stageA_dim_key(
                _stageA_compound_buckingham_target_dim(current_ast, atom, units_spec)
            )
        except Exception:
            output_dim = None
    return {
        "parent_atom_kind": str(getattr(atom, "kind", "")).lower(),
        "parent_scope": str(getattr(atom, "scope", "experiment")),
        "parent_effective_arity": int(effective_arity(atom)),
        "parent_raw_support": sorted(int(v) for v in _collect_var_idxs_from_inputs(inputs)),
        "parent_effective_input_fps": [_stageA_ast_fingerprint(inp) for inp in inputs],
        "parent_effective_input_readable": [
            ast_to_human_readable(inp, x_transform_map) for inp in inputs
        ],
        "parent_hole_context_fp": _stageA_ast_fingerprint(current_ast, target_atom=atom),
        "parent_output_dim": output_dim,
    }


def _stageA_compound_replay_disallowed_reason(meta: dict, *, extra_input_asts=None, prefactor_exps=None, prefactor_ast=None) -> Optional[str]:
    meta = meta if isinstance(meta, dict) else {}
    kind = str(meta.get("kind", "") or "").lower()
    if bool(meta.get("hidden_shadow_only", False)) or "shadow" in kind:
        return "hidden_or_shadow_lineage"
    for key in meta:
        if str(key).startswith("shadow"):
            return "shadow_lineage"
    if bool(meta.get("retained_axis_wrapper", False)):
        return "retained_axis_replay_not_pr1"
    if meta.get("preserve_z_ast") is not None:
        return "preserved_separable_coordinate"
    if extra_input_asts:
        return "extra_compound_inputs"
    if prefactor_exps is not None or prefactor_ast is not None:
        return "prefactor_or_buckingham_transaction"
    if meta.get("prefactor_exponents") is not None or meta.get("prefactor_ast") is not None:
        return "prefactor_or_buckingham_transaction"
    if bool(meta.get("partial_forced_monomial_peel", False)):
        return "partial_prefactor_peel"
    if kind in {"shadow_composite", "shadow_preserved_factor", "shadow_trig_factor_peel"}:
        return "shadow_lineage"
    if bool(meta.get("has_pending_split_context", False)):
        return "pending_split_context"
    return None


def _stageA_build_compound_replay_descriptor(
    *,
    current_ast: Any,
    atom: AtomNode,
    pattern,
    z_expr: Node,
    extra_var_idxs,
    extra_input_asts=None,
    meta: Optional[dict] = None,
    old_arity: int,
    new_arity: int,
    confidence: float,
    z_name: str,
    search_hp=None,
    lm_hp=None,
    Nxvars: Optional[int] = None,
    x_transform_map=None,
    units_spec=None,
    prefactor_exps=None,
    prefactor_ast=None,
) -> Optional[dict[str, Any]]:
    """Build a portable strict compound-coordinate replay descriptor.

    The descriptor is proposal fuel only.  It never stores a candidate subtree
    for direct splicing; replay rebuilds through the current reference parent.
    """
    if int(new_arity) >= int(old_arity):
        return None
    if _stageA_x_transform_fingerprint(x_transform_map) != "none":
        return None
    meta = meta if isinstance(meta, dict) else {}
    disallowed = _stageA_compound_replay_disallowed_reason(
        meta,
        extra_input_asts=extra_input_asts,
        prefactor_exps=prefactor_exps,
        prefactor_ast=prefactor_ast,
    )
    if disallowed is not None:
        return None
    z_payload = _stageA_ast_to_payload(z_expr)
    if z_payload is None:
        return None
    inputs = tuple(get_input_exprs(atom))
    input_index_by_raw_var: dict[int, int] = {}
    for idx, inp in enumerate(inputs):
        if not is_trivial_input(inp):
            continue
        try:
            input_index_by_raw_var[int(inp.var_idxs[0])] = int(idx)
        except Exception:
            pass
    extra_selectors = []
    for raw_idx in list(extra_var_idxs or ()):
        raw_i = int(raw_idx)
        if raw_i not in input_index_by_raw_var:
            return None
        extra_selectors.append({"kind": "parent_input", "index": int(input_index_by_raw_var[raw_i])})
    parent_ctx = _stageA_parent_context_descriptor(
        current_ast,
        atom,
        units_spec=units_spec,
        x_transform_map=x_transform_map,
    )
    transform_ctx = {
        "y_transform": _stageA_current_y_transform_name(lm_hp, search_hp),
        "x_transform_fingerprint": _stageA_x_transform_fingerprint(x_transform_map),
        **_stageA_fit_link_context(lm_hp),
    }
    problem_id = getattr(search_hp, "coe_problem_id", None) if search_hp is not None else None
    problem_id_s = None if problem_id is None else canonical_problem_id(problem_id)
    try:
        pattern_payload = [int(v) for v in pattern]
    except Exception:
        return None
    candidate_descriptor = {
        "pattern": pattern_payload,
        "z_expr_payload": z_payload,
        "z_expr_fp": _stageA_ast_fingerprint(z_expr),
        "z_expr_readable": ast_to_human_readable(z_expr, x_transform_map),
        "z_name": str(z_name),
        "extra_input_selectors": extra_selectors,
        "old_arity": int(old_arity),
        "new_arity": int(new_arity),
        "full_compound": int(new_arity) == 1,
        "proposal_kind": str(meta.get("kind", "monomial") or "monomial"),
    }
    replay_key = {
        "schema": "stageA_replay_v1",
        "proposal_class": "strict_visible_arity_reducing_compound",
        "problem_id": problem_id_s,
        "Nxvars": None if Nxvars is None else int(Nxvars),
        "transform_context": transform_ctx,
        "parent_key": {
            "parent_atom_kind": parent_ctx.get("parent_atom_kind"),
            "parent_scope": parent_ctx.get("parent_scope"),
            "parent_effective_arity": parent_ctx.get("parent_effective_arity"),
            "parent_raw_support": parent_ctx.get("parent_raw_support"),
            "parent_effective_input_fps": parent_ctx.get("parent_effective_input_fps"),
            "parent_hole_context_fp": parent_ctx.get("parent_hole_context_fp"),
            "parent_output_dim": parent_ctx.get("parent_output_dim"),
        },
        "candidate_key": {
            "pattern": pattern_payload,
            "z_expr_payload": z_payload,
            "extra_input_selectors": extra_selectors,
            "old_arity": int(old_arity),
            "new_arity": int(new_arity),
            "proposal_kind": str(meta.get("kind", "monomial") or "monomial"),
        },
    }
    return {
        "schema": "stageA_replay_v1",
        "kind": "compound_coordinate",
        "proposal_class": "strict_visible_arity_reducing_compound",
        "replay_eligible": True,
        "problem_id": problem_id_s,
        "Nxvars": None if Nxvars is None else int(Nxvars),
        "transform_context": transform_ctx,
        "parent_context": parent_ctx,
        "candidate_descriptor": candidate_descriptor,
        "replay_key": replay_key,
        "guards": {
            "hidden_shadow_only": False,
            "has_prefactor_peel": False,
            "has_preserved_separable_coordinate": False,
            "has_pending_split_context": False,
            "has_extra_compound_inputs": False,
        },
        "source_evidence": {
            "confidence": float(confidence),
        },
    }


def _stageA_log_replay_status(search_hp: Any, row: dict[str, Any]) -> None:
    if search_hp is None:
        return
    try:
        log = getattr(search_hp, "coe_stageA_replay_log", None)
        if log is None:
            log = []
            setattr(search_hp, "coe_stageA_replay_log", log)
        if isinstance(log, list):
            log.append(row)
    except Exception:
        pass


def _stageA_replay_problem_identity_telemetry(descriptor: dict[str, Any], search_hp=None) -> dict[str, str]:
    desc_raw = descriptor.get("problem_id")
    cur_raw = getattr(search_hp, "coe_problem_id", None) if search_hp is not None else None
    if desc_raw is None and cur_raw is None:
        return {}
    desc_text = None if desc_raw is None else str(desc_raw)
    cur_text = None if cur_raw is None else str(cur_raw)
    desc_canonical = None if desc_raw is None else canonical_problem_id(desc_raw)
    cur_canonical = None if cur_raw is None else canonical_problem_id(cur_raw)
    if desc_text == desc_canonical and cur_text == cur_canonical:
        return {}
    return {
        "descriptor_problem_id_raw": desc_text,
        "descriptor_problem_id_canonical": desc_canonical,
        "current_problem_id_raw": cur_text,
        "current_problem_id_canonical": cur_canonical,
    }


def _stageA_compound_replay_context_skip_reason(
    descriptor: dict[str, Any],
    *,
    current_ast: Any,
    atom: AtomNode,
    search_hp=None,
    lm_hp=None,
    Nxvars: Optional[int] = None,
    x_transform_map=None,
    units_spec=None,
) -> Optional[str]:
    if not isinstance(descriptor, dict):
        return "descriptor_not_dict"
    if not bool(descriptor.get("replay_eligible", False)):
        return "not_replay_eligible"
    if str(descriptor.get("proposal_class", "")) != "strict_visible_arity_reducing_compound":
        return "unsupported_proposal_class"
    if Nxvars is not None and descriptor.get("Nxvars") is not None:
        try:
            if int(descriptor.get("Nxvars")) != int(Nxvars):
                return "Nxvars_mismatch"
        except Exception:
            return "Nxvars_mismatch"
    desc_problem = descriptor.get("problem_id")
    cur_problem = getattr(search_hp, "coe_problem_id", None) if search_hp is not None else None
    if desc_problem and cur_problem and canonical_problem_id(desc_problem) != canonical_problem_id(cur_problem):
        return "problem_id_mismatch"
    transform = descriptor.get("transform_context") if isinstance(descriptor.get("transform_context"), dict) else {}
    if str(transform.get("y_transform", "identity")) != _stageA_current_y_transform_name(lm_hp, search_hp):
        return "y_transform_mismatch"
    cur_x_fp = _stageA_x_transform_fingerprint(x_transform_map)
    if cur_x_fp != "none":
        return "x_transform_replay_unsupported"
    if str(transform.get("x_transform_fingerprint", "none")) != cur_x_fp:
        return "x_transform_mismatch"
    cur_fit = _stageA_fit_link_context(lm_hp)
    if str(transform.get("fit_y_link", cur_fit["fit_y_link"])) != str(cur_fit["fit_y_link"]):
        return "fit_link_mismatch"
    if transform.get("fit_y_link_scale") != cur_fit.get("fit_y_link_scale"):
        return "fit_link_scale_mismatch"
    current_ctx = _stageA_parent_context_descriptor(
        current_ast,
        atom,
        units_spec=units_spec,
        x_transform_map=x_transform_map,
    )
    parent_ctx = descriptor.get("parent_context") if isinstance(descriptor.get("parent_context"), dict) else {}
    for key in (
        "parent_atom_kind",
        "parent_scope",
        "parent_effective_arity",
        "parent_raw_support",
        "parent_effective_input_fps",
        "parent_hole_context_fp",
        "parent_output_dim",
    ):
        if parent_ctx.get(key) != current_ctx.get(key):
            return f"{key}_mismatch"

    # Exact context should identify one current NN parent.  If the same descriptor
    # matches two sibling leaves, keep the scout out rather than choosing a side.
    matches = 0
    try:
        for cand_atom in collect_nn_atoms(current_ast):
            try:
                cand_ctx = _stageA_parent_context_descriptor(
                    current_ast,
                    cand_atom,
                    units_spec=units_spec,
                    x_transform_map=x_transform_map,
                )
                if all(parent_ctx.get(k) == cand_ctx.get(k) for k in (
                    "parent_atom_kind",
                    "parent_scope",
                    "parent_effective_arity",
                    "parent_raw_support",
                    "parent_effective_input_fps",
                    "parent_hole_context_fp",
                    "parent_output_dim",
                )):
                    matches += 1
            except Exception:
                continue
    except Exception:
        matches = 1
    if matches != 1:
        return "ambiguous_parent_match" if matches > 1 else "no_parent_match"
    return None


def _stageA_append_compound_replay_proposals(
    proposals,
    *,
    search_hp,
    lm_hp,
    current_ast,
    atom,
    Nxvars,
    x_transform_map,
    units_spec=None,
) -> list:
    """Inject matching CoE scout compound descriptors as ordinary proposals."""
    out = list(proposals or [])
    payload = getattr(search_hp, "coe_stageA_replay_reservoir", None)
    if not isinstance(payload, dict):
        return out
    candidates = list(payload.get("candidates") or [])
    if not candidates:
        return out
    existing = set()
    for prop in out:
        try:
            extra = prop[3] if len(prop) > 3 else None
            existing.add((
                _stageA_ast_fingerprint(prop[1]),
                tuple(int(v) for v in (extra or ())),
            ))
        except Exception:
            continue
    scout_lane_k = max(0, int(getattr(search_hp, "coe_stageA_replay_scout_lane_k", 2) or 2))
    added = 0
    for rec in candidates:
        if added >= scout_lane_k:
            break
        if not isinstance(rec, dict) or str(rec.get("kind", "")) != "compound_coordinate_replay":
            continue
        descriptor = rec.get("payload") if isinstance(rec.get("payload"), dict) else None
        identity_telemetry = _stageA_replay_problem_identity_telemetry(descriptor or {}, search_hp)
        reason = _stageA_compound_replay_context_skip_reason(
            descriptor or {},
            current_ast=current_ast,
            atom=atom,
            search_hp=search_hp,
            lm_hp=lm_hp,
            Nxvars=Nxvars,
            x_transform_map=x_transform_map,
            units_spec=units_spec,
        )
        if reason is not None:
            _stageA_log_replay_status(
                search_hp,
                {
                    "status": "skipped",
                    "reason": reason,
                    "reservoir_id": rec.get("reservoir_id"),
                    "kind": rec.get("kind"),
                    "support_count": rec.get("support_count"),
                    **identity_telemetry,
                },
            )
            continue
        cand_desc = descriptor.get("candidate_descriptor", {})
        z_expr = _stageA_ast_from_payload(cand_desc.get("z_expr_payload"))
        if z_expr is None:
            _stageA_log_replay_status(
                search_hp,
                {
                    "status": "skipped",
                    "reason": "z_payload_unreadable",
                    "reservoir_id": rec.get("reservoir_id"),
                    **identity_telemetry,
                },
            )
            continue
        inputs = tuple(get_input_exprs(atom))
        extras = []
        selectors_ok = True
        for selector in list(cand_desc.get("extra_input_selectors") or []):
            if not isinstance(selector, dict) or selector.get("kind") != "parent_input":
                selectors_ok = False
                break
            idx = int(selector.get("index"))
            if idx < 0 or idx >= len(inputs) or not is_trivial_input(inputs[idx]):
                selectors_ok = False
                break
            extras.append(int(inputs[idx].var_idxs[0]))
        if not selectors_ok:
            _stageA_log_replay_status(
                search_hp,
                {
                    "status": "skipped",
                    "reason": "extra_selector_unavailable",
                    "reservoir_id": rec.get("reservoir_id"),
                    **identity_telemetry,
                },
            )
            continue
        try:
            pattern = tuple(int(v) for v in cand_desc.get("pattern") or ())
        except Exception:
            _stageA_log_replay_status(
                search_hp,
                {
                    "status": "skipped",
                    "reason": "pattern_unreadable",
                    "reservoir_id": rec.get("reservoir_id"),
                    **identity_telemetry,
                },
            )
            continue
        kind = str(cand_desc.get("proposal_kind") or "monomial")
        key = (_stageA_ast_fingerprint(z_expr), tuple(extras))
        if key in existing:
            _stageA_log_replay_status(
                search_hp,
                {
                    "status": "skipped",
                    "reason": "duplicate_candidate",
                    "reservoir_id": rec.get("reservoir_id"),
                    **identity_telemetry,
                },
            )
            continue
        meta = {
            "kind": kind,
            "coe_scout_replay": True,
            "source": "coe_stageA_scout_replay",
            "source_reservoir_id": rec.get("reservoir_id"),
            "source_support_count": rec.get("support_count", 1),
            "replay_descriptor_schema": descriptor.get("schema"),
            "replay_parent_key": descriptor.get("parent_context", {}).get("parent_hole_context_fp"),
        }
        try:
            conf = float(rec.get("score") if rec.get("score") is not None else cand_desc.get("confidence", 0.995))
        except Exception:
            conf = 0.995
        if not math.isfinite(conf) or conf <= 0.0:
            conf = 0.995
        prop = (pattern, z_expr, min(0.999, max(0.85, conf)), extras, meta)
        out.append(prop)
        existing.add(key)
        added += 1
        try:
            z_readable = ast_to_human_readable(z_expr, x_transform_map)
        except Exception:
            z_readable = str(z_expr)
        print(
            "[CoE StageA replay] Injected scout compound proposal "
            f"z={z_readable}, extras={extras}, reservoir_id={rec.get('reservoir_id')}"
        )
        _stageA_log_replay_status(
            search_hp,
            {
                "status": "matched_and_injected",
                "reservoir_id": rec.get("reservoir_id"),
                "kind": rec.get("kind"),
                "support_count": rec.get("support_count"),
                "z": z_readable,
                "extras": list(extras),
                **identity_telemetry,
            },
        )
    return out


def _compound_overlapping_raw_extras(z_expr, extra_var_idxs) -> Tuple[int, ...]:
    """Return raw extras that are already referenced inside a compound input."""
    if not extra_var_idxs:
        return ()
    try:
        z_vars = {int(v) for v in _collect_var_idxs_from_node(z_expr)}
        extra_vars = {int(v) for v in extra_var_idxs}
    except Exception:
        return ()
    return tuple(sorted(z_vars & extra_vars))


def _is_pure_1d_full_compound_ast(ast, Nxvars: int) -> bool:
    """Back-compat shim: delegate to shared sr_core predicate."""
    return bool(_shared_is_pure_1d_full_compound_ast(ast, int(Nxvars)))


def _build_monomial_ast_from_cols(cols, exponents, z_ast=None):
    """Build monomial AST from cols (which may contain _COMPOUND_Z_TOKEN) and exponents."""
    terms = []
    for col, exp in zip(cols, exponents):
        if exp == 0:
            continue
        if _is_compound_token(col):
            base = clone_ast(_compound_ast_for_token(z_ast, col))
        else:
            base = Var(int(col))
        terms.append(base if exp == 1 else PowNode(base, int(exp)))
    if not terms:
        raise ValueError("All exponents are zero")
    result = terms[0]
    for t in terms[1:]:
        result = MulNode(result, t)
    return result


def _build_radial_r2_ast_from_cols(cols, z_ast=None):
    """Build r^2 from raw columns or compound-input tokens."""
    terms = []
    for col in cols:
        if _is_compound_token(col):
            base = clone_ast(_compound_ast_for_token(z_ast, col))
        else:
            base = Var(int(col))
        terms.append(PowNode(base, 2))
    if len(terms) < 2:
        raise ValueError("Need at least two coordinates for a radial compound")
    result = terms[0]
    for t in terms[1:]:
        result = AddNode(result, t)
    return result


def _collect_var_idxs_in_input_expr(expr):
    """Collect global variable indices referenced inside an input_expr AST."""
    out = set()

    def walk(node):
        if node is None:
            return
        # AtomNode('var') is used for variables inside input_expr
        if isinstance(node, AtomNode):
            try:
                kind = str(getattr(node, 'kind', '')).lower()
            except Exception:
                kind = ''
            if kind in ('var', 'x', 'input'):
                try:
                    for vi in getattr(node, 'var_idxs', ()) or ():
                        out.add(int(vi))
                except Exception:
                    pass
            return

        # Generic recursion over common AST attributes
        for attr in ('left', 'right', 'arg', 'base'):
            if hasattr(node, attr):
                try:
                    child = getattr(node, attr)
                except Exception:
                    child = None
                walk(child)

    walk(expr)
    return out


def _separability_proposal_to_ast_unified(
    op,
    group1,
    group2,
    parent_atom,
    num_segments: int,
    dual_layer: bool,
    parent_tag: str | None = None,
):
    """Build an AST split for a parent atom using input-space (z-coordinate) groups.

    This unified builder handles separability proposals from input-space checks,
    where groups contain:
    - `_COMPOUND_Z_TOKEN`: indicates the compound expression z(x) should be preserved
    - Integer indices: global variable indices (either extras or direct vars)

    For atoms with compound inputs, any child group containing _COMPOUND_Z_TOKEN
    becomes a compound NN atom that retains the parent's input_expr and uses only
    the extras present in that group.

    For atoms with trivial inputs, groups contain only integers (global var indices),
    and children are built as standard NN atoms on those variables.
    """
    from nestynet_sr.sr_core.bridges import AddNode, MulNode

    def build_child(group, tag):
        group = list(group) if group is not None else []
        z_tokens = [tok for tok in group if _is_compound_token(tok)]
        extras = [int(v) for v in group if isinstance(v, int)]

        child_kwargs = {"num_segments": num_segments, "dual_layer": dual_layer}
        child_inputs = None

        if z_tokens:
            # Build compound child: resolve each z-token to correct input expression
            parent_inputs = get_input_exprs(parent_atom)
            parent_compound_exprs = [
                inp for inp in parent_inputs if not is_trivial_input(inp)
            ]
            child_inputs_list = []
            all_expr_vars = set()
            for ztok in z_tokens:
                idx = _compound_token_index(ztok)
                z_expr = parent_compound_exprs[idx]
                child_inputs_list.append(clone_ast(z_expr))
                all_expr_vars |= _collect_var_idxs_in_input_expr(z_expr)
            for v in extras:
                child_inputs_list.append(Var(int(v)))
            child_inputs = tuple(child_inputs_list) if child_inputs_list else None

            # Choose var_idxs for dependency tracking: vars in z exprs + kept extras
            keep = all_expr_vars | set(int(v) for v in extras)
            parent_order = list(getattr(parent_atom, 'var_idxs', ()) or ())
            child_var_idxs = [int(v) for v in parent_order if int(v) in keep]
            # Fallback: if parent_order didn't contain expr vars (shouldn't), append them
            for v in sorted(keep):
                if v not in child_var_idxs:
                    child_var_idxs.append(int(v))
        else:
            # Pure-extra child: standard NN on those global axes.
            parent_order = list(getattr(parent_atom, 'var_idxs', ()) or ())
            keep = set(int(v) for v in extras)
            child_var_idxs = [int(v) for v in parent_order if int(v) in keep]
            for v in extras:
                if int(v) not in child_var_idxs:
                    child_var_idxs.append(int(v))

        return AtomNode("nn", tuple(child_var_idxs), kwargs=child_kwargs, tag=tag, inputs=child_inputs)

    tag_left = f"{parent_tag}_L" if parent_tag is not None else None
    tag_right = f"{parent_tag}_R" if parent_tag is not None else None

    left = build_child(group1, tag_left)
    right = build_child(group2, tag_right)

    if op is torch.add:
        return AddNode(left, right)
    elif op is torch.multiply:
        return MulNode(left, right)
    else:
        raise ValueError(f"Unknown separability operation: {op}")


def _check_separability_in_input_space(
    *,
    model,
    atom,
    leaf,
    datagen_train,
    device,
    dtype,
    precision_sum: float,
    precision_mult: float,
    very_verbose: bool = False,
):
    """Check add/mult separability of an atom in its input (z-coordinate) space.

    This is the unified separability checker that works in the atom's input space,
    handling both:
    - **Trivial inputs**: atom sees [Var(0), Var(1), ...] (z_i = x_i)
    - **Compound inputs**: atom sees [z(x), x_j, ...] where z is a compound expression

    For compound atoms with input_expr=z(x_raw) and extras [x_j, ...], the leaf
    operates on inputs [z, x_j, ...]. This routine computes the leaf's gradients
    and Hessians w.r.t those inputs and runs the standard separability tests in
    that transformed coordinate system.

    Returns
    -------
    cand_list : list
        Like sr_core.check_separability(), but groups contain the special token
        _COMPOUND_Z_TOKEN for z plus global indices (ints) for extra vars.
    rest_add, rest_mult : list[int] | None
        Overlap variables in the original axis space (ints only). Any overlap
        involving z is dropped (no global index exists for z).
    y_mad_scalar : float
        MAD scale of the leaf output in raw units.
    """
    if leaf is None:
        return [], None, None, 1.0

    if effective_arity(atom) <= 1:
        # Effective arity is 1 -> no separability to check.
        return [], None, None, 1.0

    if not (hasattr(leaf, 'grad') and hasattr(leaf, 'grad_grad')):
        return [], None, None, 1.0

    # datagen may be a loader or a thunk returning a loader
    dl = datagen_train() if callable(datagen_train) else datagen_train
    if dl is None:
        return [], None, None, 1.0

    y_list = []
    g_list = []
    gg_list = []

    with torch.no_grad():
        for batch in dl:
            x_full = batch[0] if isinstance(batch, (tuple, list)) else batch
            if x_full is None:
                continue
            x_full = x_full.to(device=device, dtype=dtype)
            x_full = x_full.view(x_full.shape[0], -1)

            # Build leaf input via unified eval_inputs
            x_in, _, _ = eval_inputs(atom, x_full, need_grad=False, need_hess=False)

            # Leaf forward + analytic derivatives in leaf-input space
            f = leaf(x_in)
            if torch.is_tensor(f) and f.dim() == 1:
                f = f.view(-1, 1)

            cache = {"x": x_in}
            g = leaf.grad(cache)
            gg = leaf.grad_grad(cache)
            if g is None or gg is None:
                return [], None, None, 1.0

            # Squeeze output dimension if present
            if torch.is_tensor(g) and g.dim() == 3:
                g = g[:, 0, :]
            elif torch.is_tensor(g) and g.dim() == 2:
                pass
            else:
                g = g.view(g.shape[0], -1)

            if torch.is_tensor(gg) and gg.dim() == 4:
                gg = gg[:, 0, :, :]
            elif torch.is_tensor(gg) and gg.dim() == 3:
                pass
            else:
                # Fallback attempt
                k = g.shape[1]
                gg = gg.view(gg.shape[0], k, k)

            y_list.append(f.detach())
            g_list.append(g.detach())
            gg_list.append(gg.detach())

    if not y_list:
        return [], None, None, 1.0

    y_vals = torch.cat(y_list, dim=0).squeeze(-1)
    dydx_vals = torch.cat(g_list, dim=0)
    d2ydx2_vals = torch.cat(gg_list, dim=0)

    # Robust scale for normalization
    eps = 1e-10
    y_med = torch.median(y_vals)
    y_mad = torch.median(torch.abs(y_vals - y_med))
    if (not torch.isfinite(y_mad)) or (float(y_mad.item()) == 0.0):
        y_mad = y_mad + eps
    y_mad_scalar = float(y_mad.item())

    y_norm = y_vals / y_mad
    dydx_norm = dydx_vals / y_mad
    d2ydx2_norm = d2ydx2_vals / y_mad

    # Symbol list for mapping local indices -> global var indices or compound token
    all_inputs = get_input_exprs(atom)
    _n_compounds = sum(1 for inp in all_inputs if not is_trivial_input(inp))
    symb = []
    _z_counter = 0
    for inp in all_inputs:
        if is_trivial_input(inp):
            symb.append(int(inp.var_idxs[0]))
        else:
            if _n_compounds == 1:
                symb.append(_COMPOUND_Z_TOKEN)       # backward compat
            else:
                symb.append(f"z{_z_counter}")         # indexed: "z0", "z1"
            _z_counter += 1

    from nestynet_sr.sr_core.separability_math import (
        COMPLETE_TOL_FACTOR,
        check_additivity,
        check_multiplicativity,
    )

    proposed = []
    rest_add = None
    rest_mult = None

    # Strict checks
    add_ok, g1_add, g2_add, complete_add, resta, add_metric, add_overlapping = check_additivity(
        symb, d2ydx2_norm, precision=precision_sum, very_verbose=very_verbose
    )
    if add_ok:
        g1_tok = [symb[i] for i in g1_add]
        g2_tok = [symb[i] for i in g2_add]
        proposed.append([torch.add, g1_tok, g2_tok, None, add_metric])
        # Promote overlapping additive candidates as fallback proposals
        if add_overlapping:
            primary_key = (frozenset(g1_tok), frozenset(g2_tok))
            for g1o_local, g2o_local, mo in add_overlapping:
                g1o_tok = [symb[i] for i in g1o_local]
                g2o_tok = [symb[i] for i in g2o_local]
                ovlp_key = (frozenset(g1o_tok), frozenset(g2o_tok))
                if ovlp_key == primary_key or ovlp_key == (primary_key[1], primary_key[0]):
                    continue
                proposed.append([torch.add, g1o_tok, g2o_tok, None, mo])
        if resta:
            resta_tok = [symb[i] for i in resta]
            resta_int = [t for t in resta_tok if isinstance(t, int)]
            if resta_int:
                rest_add = resta_int if rest_add is None else rest_add + resta_int

    mult_ok, g1_mult, g2_mult, complete_mult, restm, offset_info, mult_metric = check_multiplicativity(
        symb, d2ydx2_norm, dydx_norm, y_norm, precision=precision_mult, very_verbose=very_verbose
    )
    if mult_ok:
        g1_tok = [symb[i] for i in g1_mult]
        g2_tok = [symb[i] for i in g2_mult]
        proposed.append([torch.multiply, g1_tok, g2_tok, offset_info, mult_metric])
        if restm:
            restm_tok = [symb[i] for i in restm]
            restm_int = [t for t in restm_tok if isinstance(t, int)]
            if restm_int:
                rest_mult = restm_int if rest_mult is None else rest_mult + restm_int

    # Loose pass: allow a complete split with slightly relaxed tolerances.
    if (not complete_add) or (not complete_mult):
        prec_sum_loose = precision_sum * COMPLETE_TOL_FACTOR
        prec_mult_loose = precision_mult * COMPLETE_TOL_FACTOR

        if not complete_add:
            add_ok2, g1_add2, g2_add2, complete_add2, _resta2, add_metric2, add_overlapping2 = check_additivity(
                symb, d2ydx2_norm, precision=prec_sum_loose, very_verbose=very_verbose
            )
            if add_ok2 and complete_add2:
                g1_tok = [symb[i] for i in g1_add2]
                g2_tok = [symb[i] for i in g2_add2]
                proposed.append([torch.add, g1_tok, g2_tok, None, add_metric2])
            # Promote overlapping additive candidates from loose pass
            if add_ok2 and add_overlapping2:
                existing_keys = {
                    (frozenset(c[1]), frozenset(c[2]))
                    for c in proposed if c[0] is torch.add
                }
                for g1o_local, g2o_local, mo in add_overlapping2:
                    g1o_tok = [symb[i] for i in g1o_local]
                    g2o_tok = [symb[i] for i in g2o_local]
                    ovlp_key = (frozenset(g1o_tok), frozenset(g2o_tok))
                    if ovlp_key in existing_keys or (ovlp_key[1], ovlp_key[0]) in existing_keys:
                        continue
                    proposed.append([torch.add, g1o_tok, g2o_tok, None, mo])

        if not complete_mult:
            mult_ok2, g1_mult2, g2_mult2, complete_mult2, _restm2, offset_info2, mult_metric2 = check_multiplicativity(
                symb, d2ydx2_norm, dydx_norm, y_norm, precision=prec_mult_loose, very_verbose=very_verbose
            )
            if mult_ok2 and complete_mult2:
                g1_tok = [symb[i] for i in g1_mult2]
                g2_tok = [symb[i] for i in g2_mult2]
                proposed.append([torch.multiply, g1_tok, g2_tok, offset_info2, mult_metric2])

    # Reorder: disjoint (complete) splits before overlapping ones.
    # This matches the priority in check_separability() (separability_math.py)
    # which correctly tries clean disjoint splits before falling back to
    # overlapping ones.  Without this, a strict-pass overlapping split can
    # shadow a loose-pass disjoint split simply because it was appended first,
    # injecting spurious shared variables into downstream atoms.
    if len(proposed) > 1:
        proposed.sort(key=lambda c: 0 if not (set(c[1]) & set(c[2])) else 1)

    return proposed, rest_add, rest_mult, y_mad_scalar


def _quick_separability_check(
    *,
    model,
    leaf,
    z_expr,
    extra_var_idxs,
    extra_input_asts=None,
    datagen_train,
    device,
    dtype,
) -> bool:
    """Quick check if compound wrapper enables z to separate from extras.

    Used during wrapper selection to prefer wrappers that enable separability
    over those that don't, even if the loss improvement is modest.

    Parameters
    ----------
    model : nn.Module
        The trained compound model.
    leaf : nn.Module
        The compound leaf (with grad/grad_grad methods).
    z_expr : AST node
        The input expression (z wrapper).
    extra_var_idxs : list[int]
        Global indices of extra variables passed alongside z.
    extra_input_asts : list[AST node] | None
        Existing compound-expression extras passed alongside z.
    datagen_train : DataLoader or callable
        Training data generator.
    device, dtype : torch device and dtype.

    Returns
    -------
    bool
        True if the wrapper enables at least one separability (add or mult).
    """
    return bool(_quick_separability_candidates(
        model=model,
        leaf=leaf,
        z_expr=z_expr,
        extra_var_idxs=extra_var_idxs,
        extra_input_asts=extra_input_asts,
        datagen_train=datagen_train,
        device=device,
        dtype=dtype,
    ))


def _quick_separability_candidates(
    *,
    model,
    leaf,
    z_expr,
    extra_var_idxs,
    extra_input_asts=None,
    datagen_train,
    device,
    dtype,
):
    """Return cheap separability candidates for a compound wrapper."""
    extra_exprs = list(extra_input_asts or ())
    if not extra_var_idxs and not extra_exprs:
        return []  # Nothing to separate from
    if leaf is None:
        return []

    try:
        # Build a lightweight AtomNode for the separability check
        z_var_idxs = _collect_var_idxs_from_node(z_expr)
        extra_expr_var_idxs = []
        for expr in extra_exprs:
            extra_expr_var_idxs.extend(_collect_var_idxs_from_node(expr))
        all_var_idxs = tuple(dict.fromkeys(
            list(z_var_idxs)
            + extra_expr_var_idxs
            + [int(v) for v in extra_var_idxs]
        ))
        mock_inputs = (
            (z_expr,)
            + tuple(extra_exprs)
            + tuple(Var(int(v)) for v in extra_var_idxs)
        )
        mock_atom = AtomNode(
            kind="nn",
            var_idxs=all_var_idxs,
            kwargs={},
            tag="compound_sep_check",
            inputs=mock_inputs,
        )

        # Reuse existing separability check with looser precision for speed
        sep_cands, _, _, _ = _check_separability_in_input_space(
            model=model,
            atom=mock_atom,
            leaf=leaf,
            datagen_train=datagen_train,
            device=device,
            dtype=dtype,
            precision_sum=0.01,   # looser than normal (0.001) for speed
            precision_mult=0.01,
        )
        return list(sep_cands or [])
    except Exception:
        return []


def _retained_axis_power_factor_certificate(
    *,
    leaf,
    z_expr,
    extra_var_idxs,
    extra_input_asts=None,
    retained_axis: int,
    datagen_train,
    device,
    dtype,
    rel_std_tol: float,
    min_abs_power: float,
    max_abs_power: float,
    min_valid: int,
    max_points: int,
):
    """Check whether the retained raw-axis factor is a stable power/constant.

    For a multiplicative split ``F(z, xk) = A(z) * B(xk)``, the retained
    factor is power-like when ``xk * dF/dxk / F`` is essentially constant.
    This rejects gauge splits where ``B`` is an arbitrary 1D function.
    """
    if leaf is None or not hasattr(leaf, "grad"):
        return False, "retained-axis leaf has no gradient"

    extra_exprs = list(extra_input_asts or ())
    extra_vars = [int(v) for v in (extra_var_idxs or [])]
    try:
        retained_axis = int(retained_axis)
        retained_local_axis = 1 + len(extra_exprs) + extra_vars.index(retained_axis)
    except Exception:
        return False, "retained axis is not a raw extra input"

    try:
        z_var_idxs = _collect_var_idxs_from_node(z_expr)
        extra_expr_var_idxs = []
        for expr in extra_exprs:
            extra_expr_var_idxs.extend(_collect_var_idxs_from_node(expr))
        all_var_idxs = tuple(dict.fromkeys(
            list(z_var_idxs)
            + extra_expr_var_idxs
            + extra_vars
        ))
        mock_inputs = (
            (z_expr,)
            + tuple(extra_exprs)
            + tuple(Var(int(v)) for v in extra_vars)
        )
        mock_atom = AtomNode(
            kind="nn",
            var_idxs=all_var_idxs,
            kwargs={},
            tag="retained_axis_power_check",
            inputs=mock_inputs,
        )
    except Exception as exc:
        return False, f"retained-axis mock input build failed: {exc}"

    k_chunks = []
    n_seen = 0
    try:
        dl = datagen_train() if callable(datagen_train) else datagen_train
        for batch in dl:
            x_full = batch[0] if isinstance(batch, (tuple, list)) else batch
            if x_full is None:
                continue
            x_full = x_full.to(device=device, dtype=dtype).view(x_full.shape[0], -1)
            x_in, _, _ = eval_inputs(mock_atom, x_full, need_grad=False, need_hess=False)
            if x_in.shape[1] <= retained_local_axis:
                continue
            with torch.no_grad():
                y = leaf(x_in)
                if torch.is_tensor(y) and y.dim() > 1:
                    y = y[:, 0]
                else:
                    y = y.view(-1)
                g = leaf.grad({"x": x_in})
                if g is None:
                    return False, "retained-axis gradient unavailable"
                if torch.is_tensor(g) and g.dim() == 3:
                    g_axis = g[:, 0, retained_local_axis]
                elif torch.is_tensor(g) and g.dim() == 2:
                    g_axis = g[:, retained_local_axis]
                else:
                    g_axis = g.view(g.shape[0], -1)[:, retained_local_axis]
                x_axis = x_in[:, retained_local_axis]

                y_abs = torch.abs(y)
                finite = torch.isfinite(y) & torch.isfinite(g_axis) & torch.isfinite(x_axis)
                finite = finite & (torch.abs(x_axis) > 1.0e-12)
                if bool(torch.isfinite(y_abs).any()):
                    floor = max(float(torch.quantile(y_abs[torch.isfinite(y_abs)], 0.10).item()), 1.0e-12)
                else:
                    floor = 1.0e-12
                finite = finite & (y_abs > floor)
                if finite.any():
                    k_chunks.append((x_axis[finite] * g_axis[finite] / y[finite]).detach().cpu())
                    n_seen += int(finite.sum().item())
            if n_seen >= int(max_points):
                break
    except Exception as exc:
        return False, f"retained-axis power check failed: {exc}"

    if not k_chunks:
        return False, "retained-axis power check has no valid points"
    k_vals = torch.cat(k_chunks, dim=0)
    if k_vals.numel() > int(max_points):
        k_vals = k_vals[: int(max_points)]
    if int(k_vals.numel()) < int(min_valid):
        return False, f"retained-axis power check has too few valid points ({int(k_vals.numel())})"

    k_med = float(torch.median(k_vals).item())
    k_std = float(torch.std(k_vals, unbiased=False).item())
    rel = k_std / max(abs(k_med), 1.0)
    if abs(k_med) < float(min_abs_power):
        return True, (
            f"retained-axis effectively constant "
            f"(k={k_med:.4g}, rel_std={rel:.4g}, n={int(k_vals.numel())}); "
            "prefer leaf-projection cleanup over treating this as a real factor"
        )
    if abs(k_med) > float(max_abs_power):
        return False, f"retained-axis power {k_med:.4g} exceeds cap {float(max_abs_power):.4g}"
    if rel > float(rel_std_tol):
        return False, f"retained-axis factor is not power-like (k={k_med:.4g}, rel_std={rel:.4g})"
    return True, f"retained-axis power-like factor k={k_med:.4g}, rel_std={rel:.4g}, n={int(k_vals.numel())}"


def _retained_axis_overlap_split_confirmed(
    *,
    sep_cands,
    leaf,
    z_expr,
    extra_var_idxs,
    extra_input_asts=None,
    retained_axis: int,
    datagen_train,
    device,
    dtype,
    search_hp,
):
    """Return whether an overlapping retained-axis split is a real simplification."""
    retained_axis = int(retained_axis)
    has_retained_axis_mul_split = False
    for cand in sep_cands or ():
        try:
            op, group1, group2 = cand[0], list(cand[1]), list(cand[2])
        except Exception:
            continue
        if op is not torch.multiply:
            continue
        g1 = set(group1)
        g2 = set(group2)
        retained = {retained_axis}
        if g1 == retained and any(_is_compound_token(tok) for tok in g2):
            has_retained_axis_mul_split = True
            break
        if g2 == retained and any(_is_compound_token(tok) for tok in g1):
            has_retained_axis_mul_split = True
            break
    if not has_retained_axis_mul_split:
        return False, "no multiplicative split isolates the retained raw axis"

    return _retained_axis_power_factor_certificate(
        leaf=leaf,
        z_expr=z_expr,
        extra_var_idxs=extra_var_idxs,
        extra_input_asts=extra_input_asts,
        retained_axis=retained_axis,
        datagen_train=datagen_train,
        device=device,
        dtype=dtype,
        rel_std_tol=float(getattr(search_hp, "early_compound_rel_std", 0.05)),
        min_abs_power=float(getattr(search_hp, "early_compound_k_int", 0.15)),
        max_abs_power=float(getattr(search_hp, "compound_max_exponent", 5)),
        min_valid=int(getattr(search_hp, "compound_iso_z_min_valid", 64)),
        max_points=int(getattr(search_hp, "compound_pretrain_max_points", 5000) or 5000),
    )


def _stageA_ast_structural_cost(node: Node | None) -> int:
    """Small local AST cost used only to compare Stage-A split coordinates."""
    if node is None:
        return 0
    try:
        if isinstance(node, AtomNode):
            kind = str(getattr(node, "kind", "")).lower()
            if kind in ("var", "x", "input"):
                return 1
            if kind in ("const", "fixed_const", "fixedconst", "free_const", "freeconst"):
                return 1
            return 4 + int(max(0, effective_arity(node)))
        if isinstance(node, (AddNode, MulNode)):
            return 1 + _stageA_ast_structural_cost(node.left) + _stageA_ast_structural_cost(node.right)
        if isinstance(node, PowNode):
            return 2 + _stageA_ast_structural_cost(node.base)
        arg = getattr(node, "arg", None)
        if arg is not None:
            return 2 + _stageA_ast_structural_cost(arg)
    except Exception:
        pass
    try:
        return max(1, len(ast_to_human_readable(node)))
    except Exception:
        return 100


def _stageA_split_simplicity_score(
    *,
    sep_cands,
    z_expr: Node,
    extra_var_idxs,
    extra_input_asts=None,
    retained_axis_wrapper: bool = False,
    same_arity_coordinate: bool = False,
) -> Optional[tuple[int, int, int, int, int, int]]:
    """Return a lexicographic local complexity score for a visible Stage-A split.

    This is intentionally narrower than final expression complexity.  It only
    adjudicates loss-equivalent ways to split the same Stage-A NN leaf.  The
    first terms encode the important invariant: a clean split in the current
    coordinates should beat a retained-axis coordinate rewrite such as
    ``NN[q, x] -> NN[q/x] * NN[x]`` unless the rewrite gives a stricter visible
    payoff elsewhere.
    """
    if not sep_cands:
        return None

    try:
        z_raw = set(int(v) for v in _collect_var_idxs_from_node(z_expr))
    except Exception:
        z_raw = set()
    extra_raw = set(int(v) for v in (extra_var_idxs or ()))
    for expr in extra_input_asts or ():
        try:
            extra_raw.update(int(v) for v in _collect_var_idxs_from_node(expr))
        except Exception:
            pass
    raw_overlap = len(z_raw & extra_raw)
    z_cost = _stageA_ast_structural_cost(z_expr)
    extra_cost = sum(_stageA_ast_structural_cost(expr) for expr in (extra_input_asts or ()))
    n_extras = len(extra_var_idxs or ()) + len(extra_input_asts or ())

    best: Optional[tuple[int, int, int, int, int, int]] = None
    for cand in sep_cands or ():
        try:
            group1 = set(cand[1])
            group2 = set(cand[2])
        except Exception:
            group1 = set()
            group2 = set()
        score = (
            int(raw_overlap),
            1 if bool(retained_axis_wrapper) else 0,
            1 if bool(same_arity_coordinate) else 0,
            int(len(group1 & group2)),
            int(z_cost + extra_cost),
            int(n_extras),
        )
        if best is None or score < best:
            best = score
    return best


def _stageA_split_score_str(score) -> str:
    if score is None:
        return "none"
    try:
        names = ("raw_overlap", "retained", "same_arity", "split_overlap", "ast_cost", "extras")
        return ", ".join(f"{name}={int(value)}" for name, value in zip(names, score))
    except Exception:
        return str(score)


def _stageA_has_meaningful_loss_improvement(
    *,
    cand_loss: Optional[float],
    reference_loss: Optional[float],
    loss_floor: float,
    noise_floor: float = 0.0,
) -> bool:
    """Return whether a candidate wins on loss under existing floor semantics."""
    try:
        cand = float(cand_loss)
        ref = float(reference_loss)
    except (TypeError, ValueError):
        return False
    if (not math.isfinite(cand)) or (not math.isfinite(ref)):
        return False

    try:
        floor = float(loss_floor)
    except (TypeError, ValueError):
        floor = 0.0
    try:
        nfloor = float(noise_floor)
    except (TypeError, ValueError):
        nfloor = 0.0
    if not math.isfinite(floor) or floor < 0.0:
        floor = 0.0
    if not math.isfinite(nfloor) or nfloor < 0.0:
        nfloor = 0.0

    meaningful_floor = max(float(floor), float(nfloor))
    if cand <= meaningful_floor and ref <= meaningful_floor:
        return False

    cand_cmp = _loss_excess_above_floor(cand, nfloor)
    ref_cmp = _loss_excess_above_floor(ref, nfloor)
    if cand_cmp is None or ref_cmp is None:
        return False
    if cand_cmp <= 0.0 and ref_cmp <= 0.0:
        return False
    return float(cand_cmp) < float(ref_cmp)


def _compound_candidate_payoff_policy(old_arity: int, new_arity: int) -> str:
    """Classify the structural payoff required before adopting a compound."""
    old_arity = int(old_arity)
    new_arity = int(new_arity)
    if new_arity > old_arity:
        return "reject"
    if new_arity == old_arity:
        return "require_sep"
    return "arity_reduction"


def _compound_candidate_has_confirmed_payoff(
    *,
    old_arity: int,
    new_arity: int,
    enables_sep: bool,
) -> bool:
    """Return whether a trained compound candidate earned adoption."""
    policy = _compound_candidate_payoff_policy(old_arity, new_arity)
    if policy == "reject":
        return False
    if policy == "require_sep":
        return bool(enables_sep)
    return True


def _monomial_candidate_input_for_atom(atom: AtomNode, *, reciprocal: bool) -> tuple[Node, ...]:
    """Return the effective 1D input for a Stage-A monomial leaf replacement."""
    inputs = get_input_exprs(atom)
    if inputs:
        base = clone_ast(inputs[0])
    else:
        base = Var(int(atom.var_idxs[0]))
    if bool(reciprocal):
        base = PowNode(base, -1.0)
    return (base,)


def _fit_stageA_fixed_power_amplitude(x: torch.Tensor, y: torch.Tensor, exponent: float) -> float:
    """Least-squares scalar initialiser for a fixed Stage-A half-power peel."""

    try:
        xv = x.detach().reshape(-1).to(dtype=torch.float64, device=torch.device("cpu"))
        yv = y.detach().reshape(-1).to(dtype=torch.float64, device=torch.device("cpu"))
        n = min(int(xv.numel()), int(yv.numel()))
        if n <= 0:
            return 1.0
        xv = xv[:n]
        yv = yv[:n]
        mask = torch.isfinite(xv) & torch.isfinite(yv) & (xv > 0)
        if int(mask.sum().item()) <= 0:
            return 1.0
        basis = torch.pow(xv[mask], float(exponent))
        den = torch.sum(basis * basis)
        if not torch.isfinite(den) or float(den.item()) <= 1.0e-30:
            return 1.0
        amp = torch.sum(basis * yv[mask]) / den
        amp_f = float(amp.item())
        return amp_f if math.isfinite(amp_f) else 1.0
    except Exception:
        return 1.0


def _make_stageA_fixed_power_monomial_ast(
    current_ast: Node,
    atom: AtomNode,
    *,
    power,
    reciprocal: bool,
    use_reduced: bool,
    amp_init: float,
) -> Optional[Node]:
    """Build a visible fixed half-power monomial replacement for Stage A."""

    inputs = _monomial_candidate_input_for_atom(atom, reciprocal=bool(reciprocal))
    if len(inputs) != 1:
        return None
    core = PowNode(inputs[0], float(power))
    if bool(use_reduced):
        replacement = core
    else:
        tag_base = str(getattr(atom, "tag", None) or "stageA_half_power")
        label = monomial_power_label(power).replace("[", "_").replace("]", "")
        replacement = MulNode(
            Scale(
                name=f"{tag_base}_{label}_scale",
                tag=f"{tag_base}_{label}_scale",
                init=float(amp_init),
            ),
            core,
        )
    return replace_atom_in_ast(current_ast, atom, replacement)


def _stageA_effective_var_support(node: Node) -> set[int]:
    """Raw variables that an AST node depends on through its effective inputs."""
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input"):
            return {int(v) for v in getattr(node, "var_idxs", ())}
        if has_nontrivial_input(node):
            return set(int(v) for v in _collect_var_idxs_from_inputs(tuple(get_input_exprs(node))))
        return {int(v) for v in getattr(node, "var_idxs", ())}
    if isinstance(node, (AddNode, MulNode)):
        return _stageA_effective_var_support(node.left) | _stageA_effective_var_support(node.right)
    if isinstance(node, PowNode):
        return _stageA_effective_var_support(node.base)
    if isinstance(node, _STAGEA_UNARY_AST_NODES):
        return _stageA_effective_var_support(node.arg)
    return set()


def _stageA_node_contains(root: Node, target: Node) -> bool:
    if root is target:
        return True
    if isinstance(root, (AddNode, MulNode)):
        return _stageA_node_contains(root.left, target) or _stageA_node_contains(root.right, target)
    if isinstance(root, PowNode):
        return _stageA_node_contains(root.base, target)
    if isinstance(root, _STAGEA_UNARY_AST_NODES):
        return _stageA_node_contains(root.arg, target)
    return False


def _stageA_flatten_mul(node: Node) -> list[Node]:
    if isinstance(node, MulNode):
        return _stageA_flatten_mul(node.left) + _stageA_flatten_mul(node.right)
    return [node]


def _stageA_nn_overlap_with_support(node: Node, support: set[int]) -> bool:
    if not support:
        return False
    if isinstance(node, AtomNode):
        if str(getattr(node, "kind", "")).lower() == "nn":
            return bool(_stageA_effective_var_support(node) & support)
        return False
    if isinstance(node, (AddNode, MulNode)):
        return (
            _stageA_nn_overlap_with_support(node.left, support)
            or _stageA_nn_overlap_with_support(node.right, support)
        )
    if isinstance(node, PowNode):
        return _stageA_nn_overlap_with_support(node.base, support)
    if isinstance(node, _STAGEA_UNARY_AST_NODES):
        return _stageA_nn_overlap_with_support(node.arg, support)
    return False


def _stageA_monomial_has_shared_multiplicative_nn_support(root: Node, atom: AtomNode) -> bool:
    """Whether a Stage-A monomial peel would choose a representative in a shared product gauge.

    A disjoint product like ``NN[x0] * NN[x1,x2]`` only has a scalar scale gauge,
    so the monomial peel is safe and useful.  A product like
    ``NN[x0] * NN[x0,x1]`` has an unresolved functional multiplicative gauge:
    powers of the shared coordinate can move between the two NN factors.  Stage A
    should leave that case to the gauge-aware Stage B machinery unless a later
    rewrite gives whole-scope confirmation.
    """
    support = _stageA_effective_var_support(atom)
    if not support:
        return False

    def _walk(node: Node) -> bool:
        if isinstance(node, MulNode):
            factors = _stageA_flatten_mul(node)
            if any(_stageA_node_contains(f, atom) for f in factors):
                for factor in factors:
                    if _stageA_node_contains(factor, atom):
                        continue
                    if _stageA_nn_overlap_with_support(factor, support):
                        return True
            return _walk(node.left) or _walk(node.right)
        if isinstance(node, AddNode):
            return _walk(node.left) or _walk(node.right)
        if isinstance(node, PowNode):
            return _walk(node.base)
        if isinstance(node, _STAGEA_UNARY_AST_NODES):
            return _walk(node.arg)
        return False

    return _walk(root)


def _stageA_node_contains_trainable_nn(node: Node) -> bool:
    if isinstance(node, AtomNode):
        return str(getattr(node, "kind", "")).lower() == "nn"
    if isinstance(node, (AddNode, MulNode)):
        return _stageA_node_contains_trainable_nn(node.left) or _stageA_node_contains_trainable_nn(node.right)
    if isinstance(node, PowNode):
        return _stageA_node_contains_trainable_nn(node.base)
    if isinstance(node, _STAGEA_UNARY_AST_NODES):
        return _stageA_node_contains_trainable_nn(node.arg)
    return False


def _stageA_monomial_should_use_reduced_form(root: Node, atom: AtomNode) -> bool:
    """Use rpoly for Stage-A monomial peels when a product sibling can absorb scale.

    This mirrors the Stage-B monomial-only policy without making isolated
    monomial closures underpowered: in a disjoint product such as
    ``NN[x0] * NN[x1,x2]``, replacing the first factor by monic ``rpoly(x0)``
    removes a needless scalar gauge and lets the remaining NN carry the scale.
    If no trainable NN sibling exists, keep ``poly`` so the monomial factor can
    fit its own coefficient.
    """

    def _walk(node: Node) -> bool:
        if isinstance(node, MulNode):
            factors = _stageA_flatten_mul(node)
            if any(_stageA_node_contains(f, atom) for f in factors):
                return any(
                    (not _stageA_node_contains(f, atom)) and _stageA_node_contains_trainable_nn(f)
                    for f in factors
                )
            return _walk(node.left) or _walk(node.right)
        if isinstance(node, AddNode):
            return _walk(node.left) or _walk(node.right)
        if isinstance(node, PowNode):
            return _walk(node.base)
        if isinstance(node, _STAGEA_UNARY_AST_NODES):
            return _walk(node.arg)
        return False

    return _walk(root)


def _try_stageA_univariate_monomial_for_atom(
    *,
    model,
    current_ast,
    atom,
    tag_to_leaf,
    datagen_train_noshuffle,
    datagen_val_noshuffle,
    device,
    dtype,
    leaf_builder,
    dual_layer_used,
    search_hp,
    lm_hp,
    loss_target_eff,
    accept_threshold_eff_cand,
    best_val_loss,
    current_val_loss: Optional[float] = None,
    stageA_under_protest: bool = False,
    best_train_loss=None,
    loss_scale=1.0,
    units_spec=None,
    enforce_units: bool = False,
    units_reject_cb=None,
    y_op=None,
    y_op_inv=None,
):
    """Try a confirmed Stage-A terminal monomial peel for a 1D NN atom."""
    if (
        not isinstance(atom, AtomNode)
        or str(getattr(atom, "kind", "")).lower() != "nn"
        or int(effective_arity(atom)) != 1
    ):
        return False, None, None, None

    original_leaf = tag_to_leaf.get(atom.tag) if isinstance(tag_to_leaf, dict) else None
    if original_leaf is None:
        return False, None, None, None

    if _stageA_monomial_has_shared_multiplicative_nn_support(current_ast, atom):
        print(
            f"[Stage A Monomial] Skipping NN{list(atom.var_idxs)}: "
            "shared multiplicative NN support; leaving gauge-sensitive peel to Stage B."
        )
        return False, None, None, None

    data = _gather_atom_teacher_data(
        train_loader=datagen_train_noshuffle,
        atom=atom,
        teacher=original_leaf,
        device=device,
        dtype=dtype,
        max_points=int(getattr(search_hp, "stageA_monomial_screen_max_points", 5000) or 5000),
    )
    if data is None:
        return False, None, None, None
    X, F = data
    if X.ndim != 2 or X.shape[1] < 1:
        return False, None, None, None
    screen = fit_univariate_monomial_screen(X[:, 0], F)
    if not screen.ok:
        return False, None, None, None

    proposals = []
    half_power_ok = False
    half_power = None
    if screen.rel_rms <= 1.0e-3:
        half_power = snap_to_half_integer_monomial_power(screen.k_hat)
        if half_power is not None:
            half_power_ok, half_power_reason = half_power_domain_ok(X[:, 0], F)
            if not half_power_ok:
                print(
                    f"[Stage A Monomial] Half-power snap rejected for NN{list(atom.var_idxs)}: "
                    f"{half_power_reason}"
                )
    for degree in (1, 2, 3):
        for reciprocal in (False, True):
            label = f"monomial_deg{degree}" + ("[z_inv]" if reciprocal else "")
            key = candidate_priority_from_screen(
                label=label,
                screen=screen,
                is_raw_variable=not has_nontrivial_input(atom),
                scale_hint=None,
            )
            proposals.append((key, label, int(degree), bool(reciprocal), False))
    if half_power is not None and half_power_ok:
        for reciprocal in (False, True):
            label = monomial_power_label(half_power) + ("[z_inv]" if reciprocal else "")
            key = candidate_priority_from_screen(
                label=label,
                screen=screen,
                is_raw_variable=not has_nontrivial_input(atom),
                scale_hint=None,
            )
            proposals.append((key, label, half_power, bool(reciprocal), True))
    proposals.sort(key=lambda t: t[0])

    parent_num_segments = atom.kwargs.get("num_segments", search_hp.num_segments_map[dual_layer_used])
    parent_dual_layer = atom.kwargs.get("dual_layer", dual_layer_used)
    max_worsening_factor = float(getattr(search_hp, "max_worsening_factor", 100.0))
    worsening_floor = float(getattr(search_hp, "worsening_floor", 1.0e-6)) * float(loss_scale)
    acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)

    for _key, label, degree, reciprocal, is_half_power in proposals:
        # Avoid spending LM on degrees that are visibly far from the monomial certificate.
        eff_power = candidate_monomial_exponent(label)
        if eff_power is None:
            continue
        snap_tol = 8.0e-2 if bool(is_half_power) else 1.25
        if abs(float(eff_power) - float(screen.k_hat)) > snap_tol:
            continue

        use_reduced = _stageA_monomial_should_use_reduced_form(current_ast, atom)
        if bool(is_half_power):
            x_eff = X[:, 0]
            if reciprocal:
                x_eff = torch.reciprocal(torch.clamp(x_eff, min=1.0e-30))
            amp_init = (
                1.0
                if use_reduced
                else _fit_stageA_fixed_power_amplitude(x_eff, F, float(degree))
            )
            cand_ast = _make_stageA_fixed_power_monomial_ast(
                current_ast,
                atom,
                power=degree,
                reciprocal=bool(reciprocal),
                use_reduced=bool(use_reduced),
                amp_init=float(amp_init),
            )
        else:
            mono_atom = AtomNode(
                kind="rpoly" if use_reduced else "poly",
                var_idxs=tuple(int(v) for v in getattr(atom, "var_idxs", ()) or ()),
                kwargs={"degree": int(degree), "min_total": int(degree)},
                tag=getattr(atom, "tag", None),
                inputs=_monomial_candidate_input_for_atom(atom, reciprocal=reciprocal),
            )
            cand_ast = replace_atom_in_ast(current_ast, atom, mono_atom)
        if cand_ast is None:
            continue

        units_reason = _analytic_units_rejection(cand_ast, units_spec, enforce_units=bool(enforce_units))
        if units_reason is not None:
            print(f"[Stage A Monomial] Skipping {label} due to units: {units_reason}")
            if units_reject_cb is not None:
                units_reject_cb("stageA_monomial", units_reason)
            continue

        skip_tag = getattr(atom, "tag", None)
        reuse_map_raw = {t: leaf for t, leaf in (tag_to_leaf or {}).items() if t != skip_tag}
        reuse_leaves = _clone_reuse_leaves(reuse_map_raw, device, dtype)

        try:
            temp_model, _, cand_ast_updated = build_composite_ast(
                cand_ast,
                parent_num_segments,
                dual_layer=parent_dual_layer,
                leaf_builder=leaf_builder,
                device=device,
                dtype=dtype,
                reuse_leaves=reuse_leaves,
            )
            temp_model = _apply_fit_link_to_model(temp_model, lm_hp)
        except Exception as exc:
            print(f"[Stage A Monomial] Build failed for {label}: {exc}")
            continue

        n_params_base = int(model.num_parameters())
        n_params_cand = int(temp_model.num_parameters())
        accept_threshold = _compute_accept_threshold(
            base_loss=best_val_loss,
            best_loss=best_val_loss,
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            base_params=n_params_base,
            cand_params=n_params_cand,
            loss_floor=float(loss_target_eff),
            loss_cap=float(accept_threshold_eff_cand),
            count_weight=float(getattr(lm_hp, "select_count_weight", 1.0)),
            struct_gamma=float(getattr(lm_hp, "select_struct_gamma", 0.05)),
            param_gamma=float(getattr(lm_hp, "select_param_gamma", 0.30)),
            base_bonus_decades=float(getattr(lm_hp, "select_base_bonus_decades", 0.0)),
            sep_bonus_decades=float(getattr(lm_hp, "select_sep_bonus_decades", 0.05)),
            partial_sep_bonus_decades=float(getattr(lm_hp, "select_partial_sep_bonus_decades", 0.02)),
            is_separability=True,
            max_worsening_factor=max_worsening_factor,
            worsening_floor=worsening_floor,
            noise_floor=float(acceptance_noise_floor_raw),
        )
        accept_threshold, structural_target = _accept_threshold_with_structural_target(
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            accept_threshold=accept_threshold,
            loss_target_eff=loss_target_eff,
        )
        accept_threshold, terminal_analytic_cap = _stageA_cap_terminal_analytic_threshold(
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            accept_threshold=accept_threshold,
            absolute_cap=accept_threshold_eff_cand,
        )
        accept_threshold, under_protest_cap = _stageA_under_protest_threshold_cap(
            accept_threshold=accept_threshold,
            current_val_loss=current_val_loss if current_val_loss is not None else best_val_loss,
            loss_floor=loss_target_eff,
            noise_floor=acceptance_noise_floor_raw,
            under_protest=bool(stageA_under_protest),
            label=f"monomial {label}",
        )
        print(
            f"[Stage A Monomial] Trying {label} on NN{list(atom.var_idxs)} "
            f"as {'fixed-power' if is_half_power else ('rpoly' if use_reduced else 'poly')} "
            f"(k≈{screen.k_hat:.3g}, rel={screen.rel_rms:.3g}); "
            f"accept_threshold={accept_threshold:.4e}"
        )
        if structural_target:
            print(
                "[Stage A Monomial] Structural NN simplification target enabled: "
                f"{_nn_split_signature(current_ast)} → {_nn_split_signature(cand_ast_updated)}"
            )
        if terminal_analytic_cap:
            print(
                "[Stage A Monomial] Terminal analytic closure: "
                f"using absolute candidate cap {float(accept_threshold):.4e}"
            )
        if under_protest_cap:
            print("[Stage A Monomial] Under-protest branch: requiring non-regressing validation loss.")

        accepted, best_val_loss_cand, best_train_loss_cand, best_param_vec, temp_opt = train_candidate_model(
            temp_model,
            datagen_train_noshuffle,
            datagen_val_noshuffle,
            epochs=lm_hp.epochs,
            LM_strategy=lm_hp.strategy,
            nval_patience=lm_hp.nval_patience,
            loss_target=loss_target_eff,
            accept_threshold=accept_threshold,
            epochs_min=lm_hp.epochs_min,
            chisq_tol=lm_hp.chisq_tol,
            device=device,
            epochs_awful_check=lm_hp.epochs_awful_check,
            awful_threshold=lm_hp.awful_threshold,
            log_file=lm_hp.log_file,
            log_to_console=lm_hp.log_to_console,
            log_level=lm_hp.log_level,
            lm_verbose=lm_hp.LM_verbose,
            lm_hp=lm_hp,
        )
        if not accepted:
            continue

        max_train_degradation = float(getattr(search_hp, "max_train_degradation", 100.0))
        passes_relative = (
            best_train_loss is None
            or best_train_loss <= 0
            or best_train_loss_cand <= max_train_degradation * best_train_loss
        )
        passes_absolute = best_train_loss_cand <= loss_target_eff
        if not passes_relative and not passes_absolute:
            degradation = best_train_loss_cand / best_train_loss if best_train_loss else float("inf")
            print(
                f"{RED}[Stage A Monomial] Rejected{RESET} ({label}): "
                f"training loss {degradation:.0f}× worse than current model"
            )
            continue

        temp_opt._update_param_groups(best_param_vec)
        coe_ok, coe_reason, coe_summary = _stageA_terminal_closure_committee_gate(
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            base_model=model,
            cand_model=temp_model,
            label=f"monomial:{label}",
            gate_kind="stageA_univariate_monomial_closure",
            lm_hp=lm_hp,
            loss_floor=float(loss_target_eff),
            y_op=y_op,
            y_op_inv=y_op_inv,
            dtype=dtype,
            device=device,
        )
        if bool(coe_summary.get("enabled", False)):
            print(f"[CoE StageA terminal gate] {coe_reason}")
        if not coe_ok:
            print(
                f"{RED}[Stage A Monomial] Rejected by CoE terminal gate{RESET} "
                f"({label}): {coe_reason}"
            )
            continue
        print(
            f"{GREEN}[Stage A Monomial] Accepted{RESET} {label} on NN{list(atom.var_idxs)}, "
            f"val-loss {_loss_str(float(best_val_loss_cand), lm_hp)}"
        )
        return True, temp_model, cand_ast_updated, float(best_val_loss_cand)

    return False, None, None, None


def _try_nontrig_for_var_quick(
    *,
    base_z_ast,
    trig_var_idx: int,
    extra_override,
    atom,
    original_leaf,
    tag_to_leaf,
    current_ast,
    datagen_train_noshuffle,
    datagen_val_noshuffle,
    device,
    dtype,
    leaf_builder,
    parent_dual_layer,
    parent_num_segments,
    search_hp,
    lm_hp,
    loss_target_eff,
    accept_threshold_eff_cand,
    best_val_loss,
    best_train_loss,
    loss_scale,
    x_train,
    y_teacher,
) -> bool:
    """Try z = base_z * xk (non-trig) and return whether it enables separability.

    This is a quick check used before trying trig proposals. If the simpler
    z*xk version enables separability, we skip the trig proposal to avoid
    false positives from unnecessary trig wrappers.

    Parameters
    ----------
    base_z_ast : AST node
        The base z expression (without trig wrapper).
    trig_var_idx : int
        The variable index to multiply by.
    ... (other params same as _try_compound_candidates_for_atom)

    Returns
    -------
    bool
        True if z*xk enables separability.
    """

    # Build z_nontrig = base_z * Var(trig_var_idx)
    z_nontrig = MulNode(clone_ast(base_z_ast), Var(trig_var_idx))

    # Compute new extras: remove trig_var_idx since it's absorbed into z
    if extra_override is not None:
        new_extras = tuple(v for v in extra_override if int(v) != trig_var_idx)
    else:
        if has_nontrivial_input(atom):
            new_extras = tuple(
                int(v)
                for v in (extra_input_var_idxs(atom) or ())
                if int(v) != trig_var_idx
            )
        else:
            new_extras = tuple(int(v) for v in atom.var_idxs if int(v) != trig_var_idx)

    extra_var_idxs = list(new_extras)

    # If no extras remain, can't check separability
    if not extra_var_idxs:
        return False

    # Build the compound candidate AST
    try:
        cand_ast_compound = _build_compound_candidate_ast(
            current_ast,
            atom,
            z_nontrig,
            (1,),  # pattern for extension proposals
            extra_var_idxs_override=new_extras,
            prefactor_exponents=None,
        )
    except Exception as e:
        print(f"[Compound] Non-trig check failed to build AST: {e}")
        return False

    # Build and train a minimal model to check separability
    try:
        skip_tag = getattr(atom, "tag", None)
        reuse_map_raw = {t: leaf for t, leaf in (tag_to_leaf or {}).items() if t != skip_tag}
        reuse_leaves = _clone_reuse_leaves(reuse_map_raw, device, dtype)

        temp_model, _, cand_ast_updated = build_composite_ast(
            cand_ast_compound,
            parent_num_segments,
            dual_layer=parent_dual_layer,
            leaf_builder=leaf_builder,
            device=device,
            dtype=dtype,
            reuse_leaves=reuse_leaves,
        )
        temp_model = _apply_fit_link_to_model(temp_model, lm_hp)

        # Get the compound leaf
        tag_to_leaf_compound = _build_tag_to_leaf_map(cand_ast_updated, temp_model)
        compound_leaf = tag_to_leaf_compound.get(atom.tag)
        if compound_leaf is None:
            return False

        # Teacher distillation pretrain (short)
        if (original_leaf is not None) and (x_train is not None):
            try:
                from .training import pretrain_compound_leaf_from_teacher

                temp_model = pretrain_compound_leaf_from_teacher(
                    compound_model=temp_model,
                    original_leaf=original_leaf,
                    compound_leaf=compound_leaf,
                    z_ast=z_nontrig,
                    x_data=x_train,
                    original_var_idxs=list(atom.var_idxs),
                    device=device,
                    dtype=dtype,
                    extra_var_idxs=extra_var_idxs,
                    prefactor_ast=None,
                    original_input_asts=get_input_exprs(atom),
                    epochs=min(500, int(getattr(search_hp, "compound_pretrain_epochs", 2000))),
                    verbose=False,
                )
            except Exception:
                pass

        # Short training to get a reasonable model
        max_worsening_factor = float(getattr(search_hp, "max_worsening_factor", 100.0))
        worsening_floor = float(getattr(search_hp, "worsening_floor", 1.0e-6)) * loss_scale

        n_params_base = 100  # rough estimate
        n_params_cand = int(temp_model.num_parameters())
        acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)
        accept_threshold = _compute_accept_threshold(
            base_loss=best_val_loss,
            best_loss=best_val_loss,
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            base_params=n_params_base,
            cand_params=n_params_cand,
            loss_floor=float(loss_target_eff),
            loss_cap=float(accept_threshold_eff_cand),
            count_weight=float(getattr(lm_hp, "select_count_weight", 1.0)),
            struct_gamma=float(getattr(lm_hp, "select_struct_gamma", 0.05)),
            param_gamma=float(getattr(lm_hp, "select_param_gamma", 0.30)),
            base_bonus_decades=float(getattr(lm_hp, "select_base_bonus_decades", 0.0)),
            sep_bonus_decades=float(getattr(lm_hp, "select_sep_bonus_decades", 0.05)),
            partial_sep_bonus_decades=float(getattr(lm_hp, "select_partial_sep_bonus_decades", 0.02)),
            is_separability=True,
            max_worsening_factor=max_worsening_factor,
            worsening_floor=worsening_floor,
            noise_floor=float(acceptance_noise_floor_raw),
        )
        accept_threshold, _ = _accept_threshold_with_structural_target(
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            accept_threshold=accept_threshold,
            loss_target_eff=loss_target_eff,
        )

        # Short diagnostic training (reduced epochs for speed). This model is
        # never adopted: it only screens whether a non-trig coordinate could
        # unlock separability before proposing trig variants. Deliberately do
        # not invoke the CoE fit tournament for this throw-away probe.
        accepted, val_loss, train_loss, param_vec, opt = train_candidate_model(
            temp_model,
            datagen_train_noshuffle,
            datagen_val_noshuffle,
            epochs=min(1000, lm_hp.epochs),  # reduced for speed
            LM_strategy=lm_hp.strategy,
            nval_patience=max(10, lm_hp.nval_patience // 2),
            loss_target=loss_target_eff,
            accept_threshold=accept_threshold,
            epochs_min=min(50, lm_hp.epochs_min),
            chisq_tol=lm_hp.chisq_tol,
            device=device,
            epochs_awful_check=lm_hp.epochs_awful_check,
            awful_threshold=lm_hp.awful_threshold,
            log_file=lm_hp.log_file,
            log_to_console=False,  # quiet
            log_level=lm_hp.log_level,
            lm_verbose=lm_hp.LM_verbose,
            lm_hp=lm_hp,
        )

        if not accepted:
            print(f"[Compound] Non-trig z*x{trig_var_idx} did not train well enough")
            return False

        opt._update_param_groups(param_vec)

        # Check separability
        enables_sep = _quick_separability_check(
            model=temp_model,
            leaf=compound_leaf,
            z_expr=z_nontrig,
            extra_var_idxs=extra_var_idxs,
            datagen_train=datagen_train_noshuffle,
            device=device,
            dtype=dtype,
        )

        if enables_sep:
            print(f"[Compound] Non-trig z*x{trig_var_idx} enables separability")
        else:
            print(f"[Compound] Non-trig z*x{trig_var_idx} does not enable separability")

        return enables_sep

    except Exception as e:
        print(f"[Compound] Non-trig check failed: {e}")
        return False


def _try_stageA_composite_closure_candidate(
    *,
    model,
    current_ast,
    atom,
    z_expr,
    z_readable: str,
    kind: str,
    confidence: float,
    tag_to_leaf,
    datagen_train_noshuffle,
    datagen_val_noshuffle,
    device,
    dtype,
    leaf_builder,
    parent_num_segments: int,
    parent_dual_layer: bool,
    search_hp,
    lm_hp,
    loss_target_eff: float,
    accept_threshold_eff_cand: float,
    best_val_loss: float,
    current_val_loss: Optional[float] = None,
    stageA_under_protest: bool = False,
    best_train_loss=None,
    loss_scale: float = 1.0,
    model_sep_output=None,
    y_op=None,
    y_op_inv=None,
    Nxvars=None,
    x_transform_map=None,
    units_spec=None,
    enforce_units: bool = False,
    units_reject_cb=None,
    x_train=None,
    y_teacher=None,
    buckingham_reason: Optional[str] = None,
    yspace_noise_floor_raw: Optional[float] = None,
    acceptance_noise_n_eff: Optional[float] = None,
):
    """Try replacing a full mixed compound NN leaf by a visible analytic scalar*z."""
    if buckingham_reason:
        print(
            "[Stage A Composite] Buckingham would block NN[z] compression; "
            f"trying visible analytic closure first: {buckingham_reason}"
        )

    cand_ast, build_reason = _build_stageA_composite_closure_ast(
        current_ast,
        atom,
        z_expr,
        x_train=x_train,
        y_teacher=y_teacher,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
    )
    if cand_ast is None:
        print(f"[Stage A Composite] Rejected ({kind}) z={z_readable}: {build_reason}")
        if str(kind or "").lower() == "metric_distance":
            print(
                "[Stage A Metric] Visible analytic closure could not be built; "
                "falling back to NN[z] compression if policy permits."
            )
        if build_reason and callable(units_reject_cb):
            units_reject_cb("stageA_composite_closure", build_reason)
        return False, None, None, None

    if not _stageA_composite_reduces_nn_burden(current_ast, cand_ast):
        print(
            f"[Stage A Composite] Rejected ({kind}) z={z_readable}: "
            "visible AST does not reduce NN burden"
        )
        return False, None, None, None

    skip_tag = getattr(atom, "tag", None)
    reuse_map_raw = {t: leaf for t, leaf in (tag_to_leaf or {}).items() if t != skip_tag}
    reuse_leaves = _clone_reuse_leaves(reuse_map_raw, device, dtype)

    try:
        temp_model, _, cand_ast_updated = build_composite_ast(
            cand_ast,
            parent_num_segments,
            dual_layer=parent_dual_layer,
            leaf_builder=leaf_builder,
            device=device,
            dtype=dtype,
            reuse_leaves=reuse_leaves,
        )
        temp_model = _apply_fit_link_to_model(temp_model, lm_hp)
    except Exception as exc:
        print(f"[Stage A Composite] Rejected ({kind}) z={z_readable}: build failed: {exc}")
        return False, None, None, None

    n_params_base = int(model.num_parameters())
    n_params_cand = int(temp_model.num_parameters())
    max_worsening_factor = float(getattr(search_hp, "max_worsening_factor", 100.0))
    worsening_floor = float(getattr(search_hp, "worsening_floor", 1.0e-6)) * float(loss_scale)
    acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)
    accept_threshold = _compute_accept_threshold(
        base_loss=best_val_loss,
        best_loss=best_val_loss,
        base_ast=current_ast,
        cand_ast=cand_ast_updated,
        base_params=n_params_base,
        cand_params=n_params_cand,
        loss_floor=float(loss_target_eff),
        loss_cap=float(accept_threshold_eff_cand),
        count_weight=float(getattr(lm_hp, "select_count_weight", 1.0)),
        struct_gamma=float(getattr(lm_hp, "select_struct_gamma", 0.05)),
        param_gamma=float(getattr(lm_hp, "select_param_gamma", 0.30)),
        base_bonus_decades=float(getattr(lm_hp, "select_base_bonus_decades", 0.0)),
        sep_bonus_decades=float(getattr(lm_hp, "select_sep_bonus_decades", 0.05)),
        partial_sep_bonus_decades=float(getattr(lm_hp, "select_partial_sep_bonus_decades", 0.02)),
        is_separability=True,
        max_worsening_factor=max_worsening_factor,
        worsening_floor=worsening_floor,
        noise_floor=float(acceptance_noise_floor_raw),
    )
    accept_threshold, structural_target = _accept_threshold_with_structural_target(
        base_ast=current_ast,
        cand_ast=cand_ast_updated,
        accept_threshold=accept_threshold,
        loss_target_eff=loss_target_eff,
    )
    accept_threshold, terminal_analytic_cap = _stageA_cap_terminal_analytic_threshold(
        base_ast=current_ast,
        cand_ast=cand_ast_updated,
        accept_threshold=accept_threshold,
        absolute_cap=accept_threshold_eff_cand,
    )
    accept_threshold, under_protest_cap = _stageA_under_protest_threshold_cap(
        accept_threshold=accept_threshold,
        current_val_loss=current_val_loss if current_val_loss is not None else best_val_loss,
        loss_floor=loss_target_eff,
        noise_floor=acceptance_noise_floor_raw,
        under_protest=bool(stageA_under_protest),
        label=f"composite {kind}",
    )

    print(
        f"[Stage A Composite] Trying analytic closure ({kind}) conf {float(confidence):.3f}: "
        f"z={z_readable}; accept_threshold={accept_threshold:.4e}"
    )
    if str(kind or "").lower() == "metric_distance":
        print("[Stage A Metric] Trying visible analytic closure scale*z before NN[z] compression.")
    if structural_target:
        print(
            "[Stage A Composite] Structural NN simplification target enabled: "
            f"{_nn_split_signature(current_ast)} → {_nn_split_signature(cand_ast_updated)}"
        )
    if terminal_analytic_cap:
        print(
            "[Stage A Composite] Terminal analytic closure: "
            f"using absolute candidate cap {float(accept_threshold):.4e}"
        )
    if under_protest_cap:
        print("[Stage A Composite] Under-protest branch: requiring non-regressing validation loss.")

    accepted, best_val_loss_cand, best_train_loss_cand, best_param_vec, temp_opt = train_candidate_model(
        temp_model,
        datagen_train_noshuffle,
        datagen_val_noshuffle,
        epochs=lm_hp.epochs,
        LM_strategy=lm_hp.strategy,
        nval_patience=lm_hp.nval_patience,
        loss_target=loss_target_eff,
        accept_threshold=accept_threshold,
        epochs_min=lm_hp.epochs_min,
        chisq_tol=lm_hp.chisq_tol,
        device=device,
        epochs_awful_check=lm_hp.epochs_awful_check,
        awful_threshold=lm_hp.awful_threshold,
        log_file=lm_hp.log_file,
        log_to_console=lm_hp.log_to_console,
        log_level=lm_hp.log_level,
        lm_verbose=lm_hp.LM_verbose,
        lm_hp=lm_hp,
    )
    if not accepted:
        noisy_terminal_ok = False
        noisy_terminal_reason = ""
        if bool(terminal_analytic_cap):
            y_nf = yspace_noise_floor_raw
            if y_nf is None:
                y_nf = getattr(lm_hp, "stageA_yspace_noise_floor_raw", None)
            try:
                y_nf_f = float(y_nf)
            except Exception:
                y_nf_f = 0.0
            n_eff = acceptance_noise_n_eff
            if n_eff is None:
                n_eff = getattr(lm_hp, "acceptance_noise_n_eff", None)
            if n_eff is None:
                n_eff = _loader_n_eff(datagen_val_noshuffle)
            try:
                n_eff_f = float(n_eff) if n_eff is not None else None
                if n_eff_f is not None and ((not math.isfinite(n_eff_f)) or n_eff_f <= 0.0):
                    n_eff_f = None
            except Exception:
                n_eff_f = None

            if math.isfinite(y_nf_f) and y_nf_f > 0.0 and best_param_vec is not None:
                try:
                    temp_opt._update_param_groups(best_param_vec)
                    cand_y_mse = float(_eval_yspace_mse(temp_model, datagen_val_noshuffle, device))
                    base_y_mse = float(_eval_yspace_mse(model, datagen_val_noshuffle, device))
                    noisy_terminal_ok, noisy_terminal_reason = _stageA_noisy_terminal_yspace_accept(
                        base_ast=current_ast,
                        cand_ast=cand_ast_updated,
                        base_y_mse=base_y_mse,
                        cand_y_mse=cand_y_mse,
                        noise_floor_raw=y_nf_f,
                        n_eff=n_eff_f,
                    )
                except Exception as exc:
                    noisy_terminal_ok = False
                    noisy_terminal_reason = f"y-space fallback check failed: {type(exc).__name__}: {exc}"

        if noisy_terminal_ok:
            best_val_loss_cand = float(best_val_loss_cand)
            if "forced_monomial" not in str(kind or "").lower():
                coe_ok, coe_reason, coe_summary = _stageA_terminal_closure_committee_gate(
                    base_ast=current_ast,
                    cand_ast=cand_ast_updated,
                    base_model=model,
                    cand_model=temp_model,
                    label=f"{kind}:{z_readable}",
                    gate_kind="stageA_composite_closure",
                    lm_hp=lm_hp,
                    loss_floor=float(loss_target_eff),
                    y_op=y_op,
                    y_op_inv=y_op_inv,
                    dtype=dtype,
                    device=device,
                )
                if bool(coe_summary.get("enabled", False)):
                    print(f"[CoE StageA terminal gate] {coe_reason}")
                if not coe_ok:
                    print(
                        f"{RED}[Stage A Composite] Rejected by CoE terminal gate{RESET} "
                        f"({kind}) z={z_readable}: {coe_reason}"
                    )
                    return False, None, None, None
            print(
                f"{GREEN}[Stage A Composite] Accepted{RESET} ({kind}) z={z_readable}, "
                f"val-loss {_loss_str(best_val_loss_cand, lm_hp)} "
                f"(noisy original-y terminal closure: {noisy_terminal_reason})"
            )
            if str(kind or "").lower() == "metric_distance":
                print(
                    f"{GREEN}[Stage A Metric] Accepted visible analytic metric closure{RESET} "
                    "by noisy original-y terminal gate."
                )
            if model_sep_output is not None:
                torch.save(
                    dict(
                        y_op=y_op,
                        y_op_inv=y_op_inv,
                        Nxvars=Nxvars,
                        dual_layer=parent_dual_layer,
                        x_transform=x_transform_map,
                        model_state_dict=temp_model.state_dict(),
                        ast=cand_ast_updated,
                        val_loss=best_val_loss_cand,
                        fit_y_link=getattr(lm_hp, "fit_y_link", None),
                        fit_y_link_scale=float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                    ),
                    model_sep_output,
                )
            return True, temp_model, cand_ast_updated, best_val_loss_cand

        print(
            f"[Stage A Composite] Rejected ({kind}) z={z_readable}, "
            f"val-loss {float(best_val_loss_cand):.4e}"
            + (f"; noisy original-y fallback: {noisy_terminal_reason}" if noisy_terminal_reason else "")
        )
        if str(kind or "").lower() == "metric_distance":
            print(
                "[Stage A Metric] Visible analytic closure rejected; "
                "falling back to NN[z] compression if policy permits."
            )
        return False, None, None, None

    max_train_degradation = float(getattr(search_hp, "max_train_degradation", 100.0))
    passes_relative = (
        best_train_loss is None
        or best_train_loss <= 0
        or best_train_loss_cand <= max_train_degradation * best_train_loss
    )
    passes_absolute = best_train_loss_cand <= loss_target_eff
    if not passes_relative and not passes_absolute:
        degradation = best_train_loss_cand / best_train_loss if best_train_loss else float("inf")
        print(
            f"{RED}[Stage A Composite] Rejected{RESET} ({kind}) z={z_readable}: "
            f"training loss {degradation:.0f}× worse than current model"
        )
        return False, None, None, None

    temp_opt._update_param_groups(best_param_vec)
    best_val_loss_cand = float(best_val_loss_cand)
    if "forced_monomial" not in str(kind or "").lower():
        coe_ok, coe_reason, coe_summary = _stageA_terminal_closure_committee_gate(
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            base_model=model,
            cand_model=temp_model,
            label=f"{kind}:{z_readable}",
            gate_kind="stageA_composite_closure",
            lm_hp=lm_hp,
            loss_floor=float(loss_target_eff),
            y_op=y_op,
            y_op_inv=y_op_inv,
            dtype=dtype,
            device=device,
        )
        if bool(coe_summary.get("enabled", False)):
            print(f"[CoE StageA terminal gate] {coe_reason}")
        if not coe_ok:
            print(
                f"{RED}[Stage A Composite] Rejected by CoE terminal gate{RESET} "
                f"({kind}) z={z_readable}: {coe_reason}"
            )
            return False, None, None, None
    print(
        f"{GREEN}[Stage A Composite] Accepted{RESET} ({kind}) z={z_readable}, "
        f"val-loss {_loss_str(best_val_loss_cand, lm_hp)}"
    )
    if str(kind or "").lower() == "metric_distance":
        print(
            f"{GREEN}[Stage A Metric] Accepted visible analytic metric closure{RESET}, "
            f"val-loss {_loss_str(best_val_loss_cand, lm_hp)}"
        )

    if model_sep_output is not None:
        torch.save(
            dict(
                y_op=y_op,
                y_op_inv=y_op_inv,
                Nxvars=Nxvars,
                dual_layer=parent_dual_layer,
                x_transform=x_transform_map,
                model_state_dict=temp_model.state_dict(),
                ast=cand_ast_updated,
                val_loss=best_val_loss_cand,
                fit_y_link=getattr(lm_hp, "fit_y_link", None),
                fit_y_link_scale=float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
            ),
            model_sep_output,
        )

    return True, temp_model, cand_ast_updated, best_val_loss_cand

__search_definitions__ = (
    "_is_compound_token",
    "_stageA_split_group_record_payload",
    "_compound_token_index",
    "_atom_compound_cols",
    "_compound_ast_for_token",
    "_compound_extra_sort_key",
    "_compound_extra_ast_key",
    "_append_compound_extra_input_asts",
    "_compound_extra_input_asts_after_prefactor_peel",
    "_ast_repr_safe",
    "_is_ast_noop_candidate",
    "_is_passthrough_noop_candidate",
    "_compound_candidate_default_extra_var_idxs",
    "_compound_pattern_entry_is_zero",
    "_compound_proposal_support_arity",
    "_compound_proposal_extra_ast_count",
    "_compound_proposal_new_arity_for_sort",
    "_compound_proposal_prefactor_support",
    "_compound_proposal_sort_key",
    "_compound_proposal_brief",
    "_log_compound_proposal_shortlist",
    "_compound_best_proposal_confidence",
    "_phase_hint_compound_proposals_for_atom",
    "_clean_monomial_product_proposal_from_scaling",
    "_stageA_scaling_spec_k_rel",
    "_stageA_noisy_soft_monomial_product_proposals_from_scaling",
    "_stageA_append_noisy_soft_monomial_compound_proposals",
    "_shortlist_compound_proposals_with_pair_backup",
    "_stageA_schedule_gs_compound_lanes",
    "_stageA_json_key",
    "_stageA_const_fingerprint",
    "_stageA_ast_fp_obj",
    "_stageA_ast_fingerprint",
    "_stageA_ast_to_payload",
    "_stageA_ast_from_payload",
    "_stageA_dim_key",
    "_stageA_x_transform_fingerprint",
    "_stageA_current_y_transform_name",
    "_stageA_fit_link_context",
    "_stageA_parent_context_descriptor",
    "_stageA_compound_replay_disallowed_reason",
    "_stageA_build_compound_replay_descriptor",
    "_stageA_log_replay_status",
    "_stageA_compound_replay_context_skip_reason",
    "_stageA_append_compound_replay_proposals",
    "_compound_overlapping_raw_extras",
    "_is_pure_1d_full_compound_ast",
    "_build_monomial_ast_from_cols",
    "_build_radial_r2_ast_from_cols",
    "_collect_var_idxs_in_input_expr",
    "_separability_proposal_to_ast_unified",
    "_check_separability_in_input_space",
    "_quick_separability_check",
    "_quick_separability_candidates",
    "_retained_axis_power_factor_certificate",
    "_retained_axis_overlap_split_confirmed",
    "_stageA_ast_structural_cost",
    "_stageA_split_simplicity_score",
    "_stageA_split_score_str",
    "_stageA_has_meaningful_loss_improvement",
    "_compound_candidate_payoff_policy",
    "_compound_candidate_has_confirmed_payoff",
    "_monomial_candidate_input_for_atom",
    "_fit_stageA_fixed_power_amplitude",
    "_make_stageA_fixed_power_monomial_ast",
    "_stageA_effective_var_support",
    "_stageA_node_contains",
    "_stageA_flatten_mul",
    "_stageA_nn_overlap_with_support",
    "_stageA_monomial_has_shared_multiplicative_nn_support",
    "_stageA_node_contains_trainable_nn",
    "_stageA_monomial_should_use_reduced_form",
    "_try_stageA_univariate_monomial_for_atom",
    "_try_nontrig_for_var_quick",
    "_try_stageA_composite_closure_candidate",
)

__search_constants__ = (
    "_COMPOUND_Z_TOKEN",
)

__search_late_bindings__ = (
    "_accept_threshold_with_structural_target",
    "_nn_split_signature",
    "_stageA_terminal_closure_committee_gate",
    "_stageA_under_protest_threshold_cap",
)
