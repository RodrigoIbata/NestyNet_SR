# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Stage-B portfolio and direct-closure helpers for ``run_SR``."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import nestynet
import numpy as np
import torch

from nestynet_sr.sr_core import ast_to_human_readable
from nestynet_sr.sr_core.bridges import is_pure_1d_full_compound_ast
from nestynet_sr.run_sr_reports import _stageB_unresolved_symbolic_info


GREEN = "\033[32m"
RESET = "\033[0m"
_STAGEB_DECISIVELY_EXACT_RATIO = 1.0e-12


def _compute_compound_is_1d(ast, Nxvars):
    """Back-compat wrapper around shared pure-1D full-compound predicate."""
    return bool(is_pure_1d_full_compound_ast(ast, int(Nxvars)))


def _has_stageA_split(rest_add, rest_mult):
    """True iff Stage A actually produced an add/mult separability split."""
    return (rest_add is not None) or (rest_mult is not None)


def _split_success(success, rest_add, rest_mult):
    """Gate legacy Stage-A success by true separability-split evidence."""
    return bool(success) and _has_stageA_split(rest_add, rest_mult)


def _payload_confirmation_status(payload) -> str:
    if not isinstance(payload, dict):
        return "unresolved"
    status = str(
        payload.get(
            "branch_confirmation",
            payload.get("branch_confirmation_status", ""),
        )
        or ""
    )
    confirmed = {
        "outer_affine_confirmed",
        "split_confirmed",
        "stageB_confirmed",
        "analytic_rewrite_confirmed",
    }
    if status in confirmed:
        return status
    try:
        if bool(payload.get("split_success", False)):
            return "split_confirmed"
    except Exception:
        pass
    sig = payload.get("stagea_signals", {})
    if isinstance(sig, dict):
        for key, name in (
            ("outer_affine_confirmed", "outer_affine_confirmed"),
            ("stageB_confirmed", "stageB_confirmed"),
            ("analytic_rewrite_confirmed", "analytic_rewrite_confirmed"),
            ("split_success", "split_confirmed"),
        ):
            try:
                if float(sig.get(key, 0.0)) >= 0.5:
                    return name
            except Exception:
                continue
        for key in ("full_compound_compressed", "full_compound_solved"):
            try:
                if float(sig.get(key, 0.0)) >= 0.5:
                    return "provisional"
            except Exception:
                continue
    if status in {"provisional", "unresolved"}:
        return status
    return "unresolved"


def _payload_is_confirmed(payload) -> bool:
    return _payload_confirmation_status(payload) in {
        "outer_affine_confirmed",
        "split_confirmed",
        "stageB_confirmed",
        "analytic_rewrite_confirmed",
    }


def _payload_confirmation_rank(payload) -> int:
    status = _payload_confirmation_status(payload)
    if status in {"outer_affine_confirmed", "stageB_confirmed", "analytic_rewrite_confirmed"}:
        return 0
    if status == "split_confirmed":
        return 1
    if status == "provisional":
        return 2
    return 3


def _trial_adjudication_key(trial):
    payload = trial.payload if hasattr(trial, "payload") and isinstance(trial.payload, dict) else {}
    inv_ok = 1.0
    try:
        inv_ok = float(payload.get("inv_branch_ok", 1.0))
    except Exception:
        inv_ok = 0.0
    out = payload.get("out", None)
    ast_obj = getattr(out, "current_ast", None) if out is not None else None
    nn_count = 10**9
    expr_len = 10**9
    if ast_obj is not None:
        try:
            from nestynet_sr.sr_core import collect_nn_atoms
            nn_count = len(collect_nn_atoms(ast_obj))
        except Exception:
            pass
        try:
            expr_len = len(ast_to_human_readable(ast_obj))
        except Exception:
            pass
    return (
        _payload_confirmation_rank(payload),
        0 if inv_ok >= 0.5 else 1,
        int(nn_count),
        int(expr_len),
        float(getattr(trial, "val_loss_base", float("inf"))),
        str(getattr(getattr(trial, "state", None), "y_stack", getattr(trial, "name", ""))),
    )


def _stageB_shortlist_names(
    *,
    final_y_op_name,
    outer_peel_ranked,
    available_y_names,
    virtual_top_names=None,
    virtual_reserved_names=None,
    top_k=3,
    include_identity=True,
):
    """Build the Stage-B y-space adjudication shortlist.

    The shortlist is a union of independent nomination channels:
    the existing virtual y-search top-k, its single structural-reserve entry,
    the best transformed-scatter candidate, the best confirmed compound-z
    outer-affine candidate, and identity as an explicit baseline.

    Execution order is deliberately conservative: only branch-safe confirmed
    outer-affine evidence can run before identity.  Proxy/scatter candidates
    are kept in the portfolio, but they should not delay the plain identity
    route, which is the overwhelmingly common correct y-space.
    """
    available = set(str(n) for n in (available_y_names or []))
    if include_identity:
        available.add("identity")

    def _name_from_entry(entry):
        if isinstance(entry, dict):
            return entry.get("name") or entry.get("y_op_name") or entry.get("transform")
        return getattr(entry, "name", None)

    def _entry_confirmed(entry) -> bool:
        if isinstance(entry, dict):
            for key in ("confirmed", "outer_affine_confirmed"):
                try:
                    if float(entry.get(key, 0.0)) >= 0.5:
                        return True
                except Exception:
                    pass
        else:
            for key in ("confirmed", "outer_affine_confirmed"):
                try:
                    if float(getattr(entry, key, 0.0)) >= 0.5:
                        return True
                except Exception:
                    pass
        return False

    nominations = []

    for nm in list(virtual_top_names or [])[: int(max(1, top_k))]:
        nominations.append((str(nm), "virtual"))
    for nm in list(virtual_reserved_names or [])[:1]:
        nominations.append((str(nm), "virtual_structural_reserve"))

    if isinstance(outer_peel_ranked, dict):
        vals = outer_peel_ranked.get("ranked", None)
        if isinstance(vals, (list, tuple)):
            for entry in vals:
                nm = _name_from_entry(entry)
                if nm is not None:
                    nominations.append((str(nm), "scatter"))
                    break

        vals = outer_peel_ranked.get("compound_z_affine", None)
        if isinstance(vals, (list, tuple)):
            for entry in vals:
                if not _entry_confirmed(entry):
                    continue
                nm = _name_from_entry(entry)
                if nm is not None:
                    nominations.append((str(nm), "outer_affine_confirmed"))
                    break

    if final_y_op_name is not None and final_y_op_name not in {"identity", "None", "none", ""}:
        nominations.append((str(final_y_op_name), "stageA_selected"))

    if include_identity:
        nominations.append(("identity", "baseline"))

    out = []
    seen = set()
    for nm, _source in nominations:
        nm = "identity" if nm in {"", "None", "none", "null"} else str(nm)
        if nm not in available:
            continue
        if nm in seen:
            continue
        out.append(nm)
        seen.add(nm)

    if not out and include_identity and "identity" in available:
        out = ["identity"]

    confirmed_first = []
    if isinstance(outer_peel_ranked, dict):
        vals = outer_peel_ranked.get("compound_z_affine", None)
        if isinstance(vals, (list, tuple)):
            for entry in vals:
                if not _entry_confirmed(entry):
                    continue
                nm = _name_from_entry(entry)
                nm = "identity" if nm in {None, "", "None", "none", "null"} else str(nm)
                if nm in available and nm in out and nm not in confirmed_first:
                    confirmed_first.append(nm)

    ordered = []
    seen = set()

    def _append(nm):
        if nm not in out or nm in seen:
            return
        ordered.append(nm)
        seen.add(nm)

    for nm in confirmed_first:
        _append(nm)
    if include_identity:
        _append("identity")
    for nm in out:
        _append(nm)

    return ordered


def _stageB_shortlist_source_map(
    *,
    names,
    final_y_op_name,
    outer_peel_ranked,
    virtual_top_names=None,
    virtual_reserved_names=None,
):
    """Return {y_name: [nomination_source, ...]} for logging/reporting."""
    allowed = set(str(n) for n in (names or []))

    def _add(nm, source, out):
        nm = "identity" if nm in {None, "", "None", "none", "null"} else str(nm)
        if nm not in allowed:
            return
        out.setdefault(nm, [])
        if source not in out[nm]:
            out[nm].append(source)

    def _name_from_entry(entry):
        if isinstance(entry, dict):
            return entry.get("name") or entry.get("y_op_name") or entry.get("transform")
        return getattr(entry, "name", None)

    def _entry_confirmed(entry) -> bool:
        if isinstance(entry, dict):
            for key in ("confirmed", "outer_affine_confirmed"):
                try:
                    if float(entry.get(key, 0.0)) >= 0.5:
                        return True
                except Exception:
                    pass
        else:
            for key in ("confirmed", "outer_affine_confirmed"):
                try:
                    if float(getattr(entry, key, 0.0)) >= 0.5:
                        return True
                except Exception:
                    pass
        return False

    out = {}
    for nm in virtual_top_names or []:
        _add(nm, "virtual", out)
    for nm in list(virtual_reserved_names or [])[:1]:
        _add(nm, "virtual_structural_reserve", out)

    if isinstance(outer_peel_ranked, dict):
        vals = outer_peel_ranked.get("ranked", None)
        if isinstance(vals, (list, tuple)):
            for entry in vals[:1]:
                _add(_name_from_entry(entry), "scatter", out)

        vals = outer_peel_ranked.get("compound_z_affine", None)
        if isinstance(vals, (list, tuple)):
            for entry in vals:
                if _entry_confirmed(entry):
                    _add(_name_from_entry(entry), "outer_affine_confirmed", out)
                    break

    if final_y_op_name is not None and final_y_op_name not in {"identity", "None", "none", ""}:
        _add(final_y_op_name, "stageA_selected", out)
    _add("identity", "baseline", out)
    for nm in names or []:
        out.setdefault(str(nm), [])
    return out


def _stageB_candidate_metrics(state, *, y_name=None):
    num_nn = getattr(state, "num_nn_atoms", None)
    num_mv = getattr(state, "num_multivar_nn_atoms", None)
    max_ar = getattr(state, "max_nn_arity", None)
    if num_nn is None or num_mv is None or max_ar is None:
        try:
            from nestynet_sr.sr_core import collect_nn_atoms
            from nestynet_sr.sr_core.bridges import effective_arity

            atoms = collect_nn_atoms(getattr(state, "root", None))
            num_nn = len(atoms) if num_nn is None else num_nn
            if num_mv is None:
                num_mv = sum(1 for a in atoms if effective_arity(a) > 1)
            if max_ar is None:
                max_ar = max((effective_arity(a) for a in atoms), default=0)
        except Exception:
            num_nn = 10**9 if num_nn is None else num_nn
            num_mv = 10**9 if num_mv is None else num_mv
            max_ar = 10**9 if max_ar is None else max_ar

    try:
        n_params = int(getattr(state, "model").num_parameters())
    except Exception:
        n_params = 10**9

    try:
        val_loss = float(getattr(state, "val_loss", float("inf")))
    except Exception:
        val_loss = float("inf")
    try:
        noise_floor_raw = float(getattr(state, "acceptance_noise_floor_raw", 0.0) or 0.0)
    except Exception:
        noise_floor_raw = 0.0
    if not math.isfinite(noise_floor_raw) or noise_floor_raw < 0.0:
        noise_floor_raw = 0.0
    try:
        noise_n_eff = float(getattr(state, "acceptance_noise_n_eff", None))
        if (not math.isfinite(noise_n_eff)) or noise_n_eff <= 0.0:
            noise_n_eff = None
    except Exception:
        noise_n_eff = None

    accepted = getattr(state, "enabled_patterns", None) or []
    try:
        accepted_labels = tuple(str(x) for x in accepted)
    except Exception:
        accepted_labels = ()
    try:
        n_accept = len(accepted)
    except Exception:
        n_accept = 0

    loss_acceptable_eff = getattr(state, "loss_acceptable_eff", None)
    try:
        loss_acceptable_eff = float(loss_acceptable_eff)
    except Exception:
        loss_acceptable_eff = float("inf")

    loss_good_enough_eff = getattr(
        state,
        "loss_good_enough_eff",
        getattr(state, "loss_target_eff", None),
    )
    try:
        loss_good_enough_eff = float(loss_good_enough_eff)
    except Exception:
        loss_good_enough_eff = float("inf")

    try:
        from nestynet_sr.sr_search.model_selection import (
            loss_excess_above_floor,
            loss_within_floor_or_noise_equivalent,
            noise_equivalent,
            noise_equivalence_tolerance,
        )
    except Exception:
        loss_excess_above_floor = None
        loss_within_floor_or_noise_equivalent = None
        noise_equivalent = None
        noise_equivalence_tolerance = None

    if loss_excess_above_floor is not None:
        val_loss_cmp = loss_excess_above_floor(val_loss, noise_floor_raw)
        acceptable_cmp = loss_excess_above_floor(loss_acceptable_eff, noise_floor_raw)
        good_cmp = loss_excess_above_floor(loss_good_enough_eff, noise_floor_raw)
        val_loss_cmp = val_loss if val_loss_cmp is None else float(val_loss_cmp)
        acceptable_cmp = loss_acceptable_eff if acceptable_cmp is None else float(acceptable_cmp)
        good_cmp = loss_good_enough_eff if good_cmp is None else float(good_cmp)
    else:
        val_loss_cmp = val_loss
        acceptable_cmp = loss_acceptable_eff
        good_cmp = loss_good_enough_eff

    if loss_within_floor_or_noise_equivalent is not None:
        transformed_acceptable = loss_within_floor_or_noise_equivalent(
            val_loss,
            loss_acceptable_eff,
            noise_floor=noise_floor_raw,
            n_eff=noise_n_eff,
        )
    else:
        transformed_acceptable = bool(
            math.isfinite(acceptable_cmp) and val_loss_cmp <= acceptable_cmp
        )
    transformed_bad_loss = (not math.isfinite(val_loss)) or (not transformed_acceptable)
    transformed_exact_loss = bool(
        math.isfinite(val_loss)
        and math.isfinite(loss_good_enough_eff)
        and loss_good_enough_eff >= 0.0
        and (
            val_loss_cmp <= good_cmp
            or (
                loss_within_floor_or_noise_equivalent is not None
                and loss_within_floor_or_noise_equivalent(
                    val_loss,
                    loss_good_enough_eff,
                    noise_floor=noise_floor_raw,
                    n_eff=noise_n_eff,
                )
            )
            or (
                noise_equivalence_tolerance is not None
                and math.isfinite(noise_floor_raw)
                and noise_floor_raw > 0.0
                and val_loss_cmp
                <= noise_equivalence_tolerance(
                    val_loss,
                    noise_floor_raw,
                    noise_floor=noise_floor_raw,
                    n_eff=noise_n_eff,
                )
            )
            or (
                noise_equivalent is not None
                and noise_equivalent(
                    val_loss,
                    loss_good_enough_eff,
                    noise_floor=noise_floor_raw,
                    n_eff=noise_n_eff,
                )
            )
        )
    )
    bad_loss = bool(transformed_bad_loss)
    exact_loss = bool(transformed_exact_loss)

    original_y_val_loss = getattr(state, "original_y_val_loss", None)
    try:
        original_y_val_loss = float(original_y_val_loss)
    except Exception:
        original_y_val_loss = float("nan")
    original_y_loss_acceptable_eff = getattr(state, "original_y_loss_acceptable_eff", None)
    try:
        original_y_loss_acceptable_eff = float(original_y_loss_acceptable_eff)
    except Exception:
        original_y_loss_acceptable_eff = float("nan")
    original_y_loss_good_enough_eff = getattr(state, "original_y_loss_good_enough_eff", None)
    try:
        original_y_loss_good_enough_eff = float(original_y_loss_good_enough_eff)
    except Exception:
        original_y_loss_good_enough_eff = float("nan")
    original_y_noise_floor_raw = getattr(state, "original_y_noise_floor_raw", None)
    try:
        original_y_noise_floor_raw = float(original_y_noise_floor_raw)
    except Exception:
        original_y_noise_floor_raw = float("nan")
    if (
        (not math.isfinite(original_y_noise_floor_raw))
        or original_y_noise_floor_raw < 0.0
    ):
        original_y_noise_floor_raw = noise_floor_raw
    has_original_y_validation = bool(
        math.isfinite(original_y_val_loss)
        and math.isfinite(original_y_loss_acceptable_eff)
        and original_y_loss_acceptable_eff >= 0.0
    )
    if has_original_y_validation and loss_within_floor_or_noise_equivalent is not None:
        original_y_acceptable = bool(
            loss_within_floor_or_noise_equivalent(
                original_y_val_loss,
                original_y_loss_acceptable_eff,
                noise_floor=original_y_noise_floor_raw,
                n_eff=noise_n_eff,
            )
        )
    else:
        original_y_acceptable = bool(
            has_original_y_validation
            and original_y_val_loss <= original_y_loss_acceptable_eff
        )
    original_y_bad_loss = bool(has_original_y_validation and not original_y_acceptable)
    original_y_exact_ok = bool(
        (not math.isfinite(original_y_val_loss))
        or (
            math.isfinite(original_y_loss_good_enough_eff)
            and original_y_loss_good_enough_eff >= 0.0
            and (
                original_y_val_loss <= original_y_loss_good_enough_eff
                or (
                    loss_within_floor_or_noise_equivalent is not None
                    and loss_within_floor_or_noise_equivalent(
                        original_y_val_loss,
                        original_y_loss_good_enough_eff,
                        noise_floor=original_y_noise_floor_raw,
                        n_eff=noise_n_eff,
                    )
                )
            )
        )
    )
    use_original_y_for_noisy_portfolio = bool(
        has_original_y_validation
        and math.isfinite(original_y_noise_floor_raw)
        and original_y_noise_floor_raw > 0.0
    )
    if original_y_bad_loss:
        bad_loss = True
    elif use_original_y_for_noisy_portfolio and original_y_acceptable:
        # In noisy y-transform / fit-link searches, transformed-space proxy
        # losses can make a clean original-y solution look unacceptable even
        # though the final formula is within the original-y acceptance budget.
        # Do not apply this rescue without an active noise floor; noiseless
        # model ordering should remain governed by the native branch loss.
        bad_loss = False
    if exact_loss and not original_y_exact_ok:
        exact_loss = False

    if bad_loss:
        accuracy_bucket = 2
    elif exact_loss:
        accuracy_bucket = 0
    else:
        accuracy_bucket = 1

    def _loss_ratio(loss, ref):
        if math.isfinite(loss) and math.isfinite(ref) and ref > 0.0:
            return float(loss / ref)
        return float(loss) if math.isfinite(loss) else float("inf")

    portfolio_val_loss = val_loss
    portfolio_val_loss_cmp = val_loss_cmp
    portfolio_good_cmp = good_cmp
    portfolio_acceptable_cmp = acceptable_cmp
    portfolio_noise_floor_raw = noise_floor_raw
    if use_original_y_for_noisy_portfolio:
        portfolio_val_loss = original_y_val_loss
        portfolio_noise_floor_raw = original_y_noise_floor_raw
        if loss_excess_above_floor is not None:
            _v_cmp = loss_excess_above_floor(
                original_y_val_loss,
                original_y_noise_floor_raw,
            )
            _g_cmp = loss_excess_above_floor(
                original_y_loss_good_enough_eff,
                original_y_noise_floor_raw,
            )
            _a_cmp = loss_excess_above_floor(
                original_y_loss_acceptable_eff,
                original_y_noise_floor_raw,
            )
            portfolio_val_loss_cmp = (
                original_y_val_loss if _v_cmp is None else float(_v_cmp)
            )
            portfolio_good_cmp = (
                original_y_loss_good_enough_eff if _g_cmp is None else float(_g_cmp)
            )
            portfolio_acceptable_cmp = (
                original_y_loss_acceptable_eff if _a_cmp is None else float(_a_cmp)
            )
        else:
            portfolio_val_loss_cmp = original_y_val_loss
            portfolio_good_cmp = original_y_loss_good_enough_eff
            portfolio_acceptable_cmp = original_y_loss_acceptable_eff

    loss_target_ratio = _loss_ratio(portfolio_val_loss_cmp, portfolio_good_cmp)
    loss_acceptable_ratio = _loss_ratio(portfolio_val_loss_cmp, portfolio_acceptable_cmp)

    ast_cost = float("inf")
    try:
        from nestynet_sr.sr_search.model_selection import ast_cost_physics_prior
        ast_cost = float(ast_cost_physics_prior(getattr(state, "root", None)))
    except Exception:
        pass
    sympy_meta = getattr(state, "sympy_meta", None)
    complexity_score = float("inf")
    if isinstance(sympy_meta, dict):
        try:
            complexity_score = float(sympy_meta.get("complexity_score", float("inf")))
        except Exception:
            complexity_score = float("inf")
    if not math.isfinite(complexity_score):
        complexity_score = ast_cost
    original_y_complexity_score = float("nan")
    original_y_expr = (
        getattr(state, "y_expr_raw_str", None)
        or getattr(state, "y_expr_str", None)
    )
    if original_y_expr:
        original_y_complexity_score = _stageB_expression_complexity_score(original_y_expr)
        if math.isfinite(original_y_complexity_score):
            # Portfolio selection compares final formulas in original y-space,
            # not just the cheaper representative in phi(y)-space.  A branch
            # such as arctan(y)=poly(z)^2 must pay for y=tan(poly(z)^2).
            complexity_score = max(float(complexity_score), float(original_y_complexity_score))

    explicit_simplified_expr = bool(
        isinstance(sympy_meta, dict)
        and bool(sympy_meta.get("accepted", False))
        and (
            getattr(state, "y_expr_raw_str", None)
            or getattr(state, "y_expr_str", None)
            or getattr(state, "phi_expr_raw_str", None)
            or getattr(state, "phi_expr_str", None)
        )
    )
    if explicit_simplified_expr:
        stage_data_for_unresolved_check = {
            "ast": getattr(state, "root", None),
            "num_nn_atoms": num_nn,
            "sympy_meta": sympy_meta,
            "y_expr_raw_str": getattr(state, "y_expr_raw_str", None),
            "y_expr_str": getattr(state, "y_expr_str", None),
            "phi_expr_raw_str": getattr(state, "phi_expr_raw_str", None),
            "phi_expr_str": getattr(state, "phi_expr_str", None),
        }
        if _stageB_unresolved_symbolic_info(stage_data_for_unresolved_check).get("unresolved"):
            explicit_simplified_expr = False
    simplified_display_expr = (
        getattr(state, "y_expr_str", None)
        or getattr(state, "phi_expr_str", None)
    )
    simple_integer_rational_expr = bool(
        explicit_simplified_expr
        and _stageB_small_integer_rational_expression(simplified_display_expr)
    )
    generic_approx, generic_reason = _stageB_generic_approximant_signature(
        getattr(state, "root", None),
        accepted_labels=accepted_labels,
        explicit_simplified_expr=explicit_simplified_expr,
        simple_integer_rational_expr=simple_integer_rational_expr,
    )
    raw_family = _stageB_raw_y_branch_family_signature(
        original_y_expr,
        accepted_labels=accepted_labels,
        y_name=(
            y_name
            or getattr(state, "y_transform_name", None)
            or getattr(state, "_stageB_portfolio_y_name", None)
        ),
        explicit_simplified_expr=explicit_simplified_expr,
    )
    if bool(raw_family.get("raw_protected_family", False)):
        generic_approx = False
        generic_reason = ""
    elif bool(raw_family.get("raw_generic_approximant", False)):
        generic_approx = True
        generic_reason = str(raw_family.get("raw_generic_reason") or "raw-y generic approximant")

    decisive_exact_ratio = float(loss_target_ratio)
    decisive_noise_floor = float(portfolio_noise_floor_raw)
    if has_original_y_validation:
        decisive_exact_ratio = _loss_ratio(
            original_y_val_loss,
            original_y_loss_good_enough_eff,
        )
        decisive_noise_floor = float(original_y_noise_floor_raw)

    full_rewrite = bool(int(num_nn) == 0 and int(n_accept) > 0)
    decisively_exact = bool(
        full_rewrite
        and exact_loss
        and not bad_loss
        and math.isfinite(decisive_noise_floor)
        and decisive_noise_floor <= 0.0
        and math.isfinite(decisive_exact_ratio)
        and decisive_exact_ratio <= _STAGEB_DECISIVELY_EXACT_RATIO
    )
    decisively_exact_structural = bool(decisively_exact and not generic_approx)

    return {
        "num_nn": int(num_nn),
        "num_multivar_nn": int(num_mv),
        "max_nn_arity": int(max_ar),
        "n_params": int(n_params),
        "val_loss": val_loss,
        "noise_floor_raw": float(noise_floor_raw),
        "noise_n_eff": None if noise_n_eff is None else float(noise_n_eff),
        "val_loss_cmp": float(val_loss_cmp),
        "portfolio_val_loss": float(portfolio_val_loss),
        "portfolio_val_loss_cmp": float(portfolio_val_loss_cmp),
        "portfolio_noise_floor_raw": float(portfolio_noise_floor_raw),
        "loss_good_enough_eff": loss_good_enough_eff,
        "loss_acceptable_eff": loss_acceptable_eff,
        "loss_target_ratio": float(loss_target_ratio),
        "loss_acceptable_ratio": float(loss_acceptable_ratio),
        "original_y_val_loss": float(original_y_val_loss),
        "original_y_loss_good_enough_eff": float(original_y_loss_good_enough_eff),
        "original_y_loss_acceptable_eff": float(original_y_loss_acceptable_eff),
        "original_y_noise_floor_raw": float(original_y_noise_floor_raw),
        "has_original_y_validation": bool(has_original_y_validation),
        "original_y_bad_loss": bool(original_y_bad_loss),
        "accepted_patterns": int(n_accept),
        "bad_loss": bool(bad_loss),
        "exact_loss": bool(exact_loss),
        "accuracy_bucket": int(accuracy_bucket),
        "full_rewrite": bool(full_rewrite),
        "decisive_exact_ratio": float(decisive_exact_ratio),
        "decisively_exact": bool(decisively_exact),
        "decisively_exact_structural": bool(decisively_exact_structural),
        "complexity_score": float(complexity_score),
        "original_y_complexity_score": float(original_y_complexity_score),
        "ast_cost": float(ast_cost),
        "accepted_labels": accepted_labels,
        "generic_approximant": bool(generic_approx),
        "generic_approximant_reason": str(generic_reason),
        "simple_integer_rational_expr": bool(simple_integer_rational_expr),
        "raw_family": str(raw_family.get("raw_family") or ""),
        "raw_protected_family": bool(raw_family.get("raw_protected_family", False)),
        "raw_generic_approximant": bool(raw_family.get("raw_generic_approximant", False)),
        "raw_generic_reason": str(raw_family.get("raw_generic_reason") or ""),
        "inverse_y_transform_wrapped": bool(raw_family.get("inverse_y_transform_wrapped", False)),
        "raw_sparse_rational": bool(raw_family.get("raw_sparse_rational", False)),
    }


def _stageB_expression_complexity_score(expr_str) -> float:
    """Return the display complexity of a SymPy-readable expression string."""
    if expr_str is None:
        return float("inf")
    try:
        text = str(expr_str).strip()
    except Exception:
        return float("inf")
    if not text:
        return float("inf")
    try:
        import sympy as sp
        from nestynet_sr.sr_search.polish_utils import (
            constant_code_cost,
            final_polish_snap_targets,
        )
    except Exception:
        return float("inf")
    try:
        locals_map = {
            "pi": sp.pi,
            "E": sp.E,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "log": sp.log,
            "exp": sp.exp,
        }
        expr = sp.sympify(text, locals=locals_map)
    except Exception:
        return float("inf")
    try:
        ops = float(sp.count_ops(expr, visual=False))
    except Exception:
        try:
            ops = float(len(sp.sstr(expr)))
        except Exception:
            ops = float(len(text))
    try:
        const_cost, n_long = constant_code_cost(
            expr,
            snap_targets=final_polish_snap_targets(),
            snap_rel_tol=1.0e-4,
        )
    except Exception:
        const_cost, n_long = 0.0, 0
    return float(ops + float(const_cost) + 4.0 * float(n_long))


def _stageB_small_integer_rational_expression(expr_str) -> bool:
    """Return True for compact polynomial/rational expressions with integer coefficients."""
    if expr_str is None:
        return False
    try:
        text = str(expr_str).strip()
    except Exception:
        return False
    if not text:
        return False
    try:
        import sympy as sp
    except Exception:
        return False
    try:
        locals_map = {
            "pi": sp.pi,
            "E": sp.E,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "log": sp.log,
            "exp": sp.exp,
        }
        expr = sp.sympify(text, locals=locals_map)
    except Exception:
        return False
    try:
        if expr.atoms(sp.Float):
            return False
        if expr.atoms(sp.Function):
            return False
        if not expr.free_symbols:
            return False
        if float(sp.count_ops(expr, visual=False)) > 24.0:
            return False
        num, den = sp.fraction(sp.together(expr))
        symbols = tuple(sorted(expr.free_symbols, key=lambda s: str(s)))
        total_terms = 0
        for part in (num, den):
            poly = sp.Poly(part, *symbols)
            if int(poly.total_degree()) > 4:
                return False
            terms = poly.terms()
            total_terms += len(terms)
            for _, coeff in terms:
                if coeff.atoms(sp.Float):
                    return False
                if not bool(coeff.is_Integer):
                    return False
                if abs(int(coeff)) > 128:
                    return False
        if total_terms > 8:
            return False
    except Exception:
        return False
    return True


def _stageB_sparse_rational_raw_y_expression(expr_str) -> bool:
    """Return True for compact raw-y rational formulas, allowing exact constants."""
    if expr_str is None:
        return False
    try:
        text = str(expr_str).strip()
    except Exception:
        return False
    if not text:
        return False
    try:
        import sympy as sp
    except Exception:
        return False
    try:
        locals_map = {
            "pi": sp.pi,
            "E": sp.E,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "log": sp.log,
            "exp": sp.exp,
        }
        expr = sp.sympify(text, locals=locals_map)
    except Exception:
        return False
    try:
        if not expr.free_symbols:
            return False
        # Transcendental dependence on variables is not a sparse rational law.
        for fn in expr.atoms(sp.Function):
            if getattr(fn, "free_symbols", None):
                return False
        if float(sp.count_ops(expr, visual=False)) > 48.0:
            return False
        num, den = sp.fraction(sp.together(expr))
        symbols = tuple(sorted(expr.free_symbols, key=lambda s: str(s)))
        total_terms = 0
        for part in (num, den):
            poly = sp.Poly(part, *symbols)
            if int(poly.total_degree()) > 4:
                return False
            terms = poly.terms()
            total_terms += len(terms)
            if len(terms) > 5:
                return False
            for _, coeff in terms:
                if coeff.atoms(sp.Function):
                    return False
        if total_terms > 8:
            return False
    except Exception:
        return False
    return True


def _stageB_raw_y_branch_family_signature(
    expr_str,
    *,
    accepted_labels=(),
    y_name=None,
    explicit_simplified_expr: bool = False,
) -> dict:
    """Classify a y-portfolio winner by the expression it asserts in raw y-space."""
    try:
        text = str(expr_str or "").strip()
    except Exception:
        text = ""
    labels = tuple(str(label).strip().lower() for label in (accepted_labels or ()))
    y_name_s = str(y_name or "").strip().lower()
    lower = text.lower().replace(" ", "")
    has_inverse_wrapper = bool(
        y_name_s
        and y_name_s not in {"identity", "none"}
        and any(tok in lower for tok in ("tan(", "exp(", "asin(", "acos(", "sqrt("))
    )
    raw_sparse_rational = bool(
        _stageB_sparse_rational_raw_y_expression(text)
        and any(
            _stageB_is_plain_rational_approximant_label(label)
            for label in labels
        )
    )
    clean_macro = any(
        token in label
        for label in labels
        for token in (
            "inv_sqrt1m",
            "sqrt1m",
            "inv1m",
            "inv1p",
            "planck",
            "exp_linear",
        )
    )
    raw_protected = bool(
        raw_sparse_rational
        or clean_macro
        or (explicit_simplified_expr and _stageB_small_integer_rational_expression(text))
    )

    flexible_closure = any(
        _stageB_is_plain_rational_approximant_label(label)
        or any(
            token in label
            for token in (
                "leaftr",
                "tanh",
                "poly3",
                "poly4",
                "exp_rat",
                "sqrt_rat",
                "log_rat",
            )
        )
        for label in labels
    )
    raw_generic = False
    generic_reasons = []
    if has_inverse_wrapper and not raw_protected and flexible_closure:
        raw_generic = True
        generic_reasons.append("inverse_y_transform_wrapped")
        generic_reasons.append("transformed_space_flexible_closure")

    family = "raw_sparse_rational" if raw_sparse_rational else ""
    if clean_macro and not family:
        family = "raw_clean_macro"
    if raw_generic and not family:
        family = "raw_inverse_transform_flexible"
    return {
        "raw_family": family,
        "raw_protected_family": bool(raw_protected),
        "raw_generic_approximant": bool(raw_generic),
        "raw_generic_reason": ",".join(dict.fromkeys(generic_reasons)),
        "inverse_y_transform_wrapped": bool(has_inverse_wrapper),
        "raw_sparse_rational": bool(raw_sparse_rational),
    }


def _stageB_is_plain_rational_approximant_label(label) -> bool:
    s = str(label).strip().lower()
    if not s:
        return False
    if any(token in s for token in ("exp_rat", "sqrt_rat", "log_rat")):
        return False
    return any(
        token in s
        for token in (
            "ratpoly",
            "rratpoly",
            "rationalpoly",
            "rational_poly",
            "rrationalpoly",
            "rrational_poly",
            "ratio_poly",
            "ratiopoly",
            "rratio_poly",
        )
    )


def _stageB_generic_approximant_signature(
    root,
    *,
    accepted_labels=(),
    explicit_simplified_expr: bool = False,
    simple_integer_rational_expr: bool = False,
):
    """Return whether a Stage-B branch is an opaque approximation family.

    Local Stage B may accept rational-polynomial approximants when they improve
    the current branch.  Portfolio selection is stricter: these families are
    useful fallbacks, but they should not preempt other y-spaces that may expose
    a simple analytic closure.
    """
    label_hits = []
    for label in accepted_labels or ():
        s = str(label).strip().lower()
        if not s:
            continue
        if any(
            token in s
            for token in (
                "ratpoly",
                "rratpoly",
                "rationalpoly",
                "rational_poly",
                "exp_rat",
                "sqrt_rat",
                "log_rat",
            )
        ):
            label_hits.append(str(label))

    atom_hits = []
    try:
        from nestynet_sr.sr_core.bridges import collect_all_atoms, effective_arity

        generic_kinds = {
            "ratpoly",
            "rationalpoly",
            "rational_poly",
            "rratpoly",
            "rrationalpoly",
            "rrational_poly",
            "ratio_poly",
            "ratiopoly",
            "rratio_poly",
            "exp_ratpoly",
            "expratpoly",
            "sqrt_ratpoly",
            "log_ratpoly",
        }
        for atom in collect_all_atoms(root):
            kind = str(getattr(atom, "kind", "")).strip().lower()
            if kind not in generic_kinds:
                continue
            try:
                ar = int(effective_arity(atom))
            except Exception:
                ar = int(len(getattr(atom, "var_idxs", ()) or ()))
            atom_hits.append(f"{kind}/arity{ar}")
    except Exception:
        pass

    if explicit_simplified_expr and simple_integer_rational_expr:
        # A compact accepted integer polynomial/rational expression is no
        # longer an opaque approximant: the ratpoly family was just the fitting
        # route.  Keep decorated rational families generic unless separately
        # justified.
        label_hits = [
            label
            for label in label_hits
            if not _stageB_is_plain_rational_approximant_label(label)
        ]
        atom_hits = [
            hit
            for hit in atom_hits
            if not _stageB_is_plain_rational_approximant_label(str(hit).split("/arity", 1)[0])
        ]
    elif explicit_simplified_expr and not atom_hits:
        # A verified Stage-C expression is no longer an opaque 1D rational
        # atom; the ratpoly_1d label is only provenance.  More expressive
        # families such as exp_rat/log_rat and non-1D ratpoly labels remain
        # generic unless their own policy says otherwise.
        label_hits = [
            label
            for label in label_hits
            if "ratpoly_1d" not in str(label).strip().lower()
        ]

    if label_hits or atom_hits:
        parts = []
        if label_hits:
            parts.append("labels=" + ",".join(label_hits[:3]))
        if atom_hits:
            parts.append("atoms=" + ",".join(atom_hits[:3]))
        return True, "; ".join(parts)
    return False, ""


def _stageB_adjudication_key(state, *, y_name, rank, y_sources=None):
    """Prefer Stage-B-confirmed simplification over proxy rank/loss noise."""
    m = _stageB_candidate_metrics(state, y_name=y_name)
    sources = set(str(s) for s in (y_sources or []))
    confirmed_penalty = (
        0
        if sources.intersection({"outer_affine_confirmed", "stageA_selected"})
        else 1
    )
    identity_penalty = 0 if str(y_name) == "identity" else 1
    generic_penalty = 1 if m["generic_approximant"] else 0
    raw_protected_penalty = 0 if m.get("raw_protected_family", False) else 1
    inverse_transform_penalty = 1 if m.get("inverse_y_transform_wrapped", False) else 0

    def _portfolio_loss_bucket() -> float:
        """Coarsen noisy full-rewrite losses before applying simplicity priors."""
        if bool(m.get("exact_loss", False)) and not bool(m.get("bad_loss", False)):
            return 0.0
        try:
            val_cmp = float(m.get("portfolio_val_loss_cmp", m.get("val_loss_cmp", float("inf"))))
        except Exception:
            val_cmp = float("inf")
        if not math.isfinite(val_cmp):
            return float("inf")
        try:
            noise_floor = float(
                m.get("portfolio_noise_floor_raw", m.get("noise_floor_raw", 0.0))
                or 0.0
            )
        except Exception:
            noise_floor = 0.0
        if math.isfinite(noise_floor) and noise_floor > 0.0:
            try:
                from nestynet_sr.sr_search.model_selection import noise_equivalence_tolerance

                tol = float(
                    noise_equivalence_tolerance(
                        m.get("portfolio_val_loss", m.get("val_loss", val_cmp)),
                        noise_floor,
                        noise_floor=noise_floor,
                        n_eff=m.get("noise_n_eff", None),
                    )
                )
            except Exception:
                tol = 0.0
            if math.isfinite(tol) and val_cmp <= tol:
                return 0.0
        return float(val_cmp)

    if m["bad_loss"]:
        return (
            2,
            m["loss_acceptable_ratio"],
            m["num_multivar_nn"],
            m["num_nn"],
            m["max_nn_arity"],
            raw_protected_penalty,
            generic_penalty,
            inverse_transform_penalty,
            m["complexity_score"],
            m["n_params"],
            confirmed_penalty,
            identity_penalty,
            int(rank),
            str(y_name),
        )
    if m["exact_loss"]:
        return (
            0,
            _portfolio_loss_bucket(),
            0 if m.get("decisively_exact_structural", False) else 1,
            m["num_multivar_nn"],
            m["num_nn"],
            m["max_nn_arity"],
            raw_protected_penalty,
            generic_penalty,
            inverse_transform_penalty,
            m["complexity_score"],
            m["n_params"],
            0 if m["accepted_patterns"] > 0 else 1,
            confirmed_penalty,
            identity_penalty,
            int(rank),
            str(y_name),
        )
    if not m["full_rewrite"]:
        return (
            1,
            1,
            m["num_multivar_nn"],
            m["num_nn"],
            m["max_nn_arity"],
            confirmed_penalty,
            identity_penalty,
            int(rank),
            m["loss_target_ratio"],
            m["n_params"],
            str(y_name),
        )
    return (
        1,
        0,
        _portfolio_loss_bucket(),
        m["num_multivar_nn"],
        m["num_nn"],
        m["max_nn_arity"],
        raw_protected_penalty,
        generic_penalty,
        inverse_transform_penalty,
        m["complexity_score"],
        m["n_params"],
        0 if m["accepted_patterns"] > 0 else 1,
        m["loss_target_ratio"],
        confirmed_penalty,
        identity_penalty,
        int(rank),
        str(y_name),
    )


def _stageB_portfolio_early_stop_decision(state, *, y_sources=None, y_name=None):
    """Return (can_stop, reason) for Stage-B y-transform portfolio stopping."""
    sources = set(str(s) for s in (y_sources or []))
    if not sources.intersection({"outer_affine_confirmed", "stageA_selected", "baseline"}):
        return False, ""
    m = _stageB_candidate_metrics(state, y_name=y_name)
    core_ok = bool(
        m["exact_loss"]
        and (not m["bad_loss"])
        and m["num_multivar_nn"] == 0
        and m["num_nn"] == 0
        and m["max_nn_arity"] == 0
        and m["accepted_patterns"] > 0
        and math.isfinite(m["val_loss"])
    )
    if not core_ok:
        return False, ""
    try:
        pf_noise = float(m.get("portfolio_noise_floor_raw", 0.0) or 0.0)
    except Exception:
        pf_noise = 0.0
    if math.isfinite(pf_noise) and pf_noise > 0.0:
        return (
            False,
            "noisy raw-y branch ballot keeps remaining y-branches for original-y adjudication",
        )
    if m["generic_approximant"]:
        detail = m["generic_approximant_reason"] or "generic approximation family"
        return (
            False,
            "validation-good full rewrite is a generic approximant "
            f"({detail}); preserving remaining y-branches",
        )
    return True, "branch-safe validation-good full rewrite"


def _stageB_portfolio_can_stop_early(state, *, y_sources=None, y_name=None):
    """Stop once a branch-safe route has a target-good non-generic full rewrite."""
    ok, _reason = _stageB_portfolio_early_stop_decision(
        state,
        y_sources=y_sources,
        y_name=y_name,
    )
    return bool(ok)


def _stageB_portfolio_continue_reason(state, *, y_sources=None, y_name=None):
    """Human-readable reason for a validation-good branch that should not stop."""
    ok, reason = _stageB_portfolio_early_stop_decision(
        state,
        y_sources=y_sources,
        y_name=y_name,
    )
    if ok:
        return ""
    return reason


def _stageB_y_branch_artifact(state, *, y_name, rank, y_sources=None):
    """Return a portable raw-y terminal branch candidate for final CoE scoring."""
    try:
        m = _stageB_candidate_metrics(state, y_name=y_name)
    except Exception:
        return None
    try:
        num_nn = int(m.get("num_nn", 1))
    except Exception:
        num_nn = 1
    if num_nn != 0:
        return None
    expr = getattr(state, "y_expr_raw_str", None) or getattr(state, "y_expr_str", None)
    if expr is None:
        return None
    try:
        text = str(expr).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        raw_loss = float(m.get("original_y_val_loss", float("nan")))
    except Exception:
        raw_loss = float("nan")
    if not math.isfinite(raw_loss):
        raw_loss = float(m.get("val_loss", float("nan")))
    return {
        "expr": text,
        "source": "stageB_y_branch",
        "label": f"y_branch:{y_name}",
        "complexity": m.get("complexity_score"),
        "n_free_params": 0,
        "loss": raw_loss if math.isfinite(raw_loss) else None,
        "metadata": {
            "branch_id": str(y_name),
            "branch_rank": int(rank),
            "branch_sources": list(y_sources or []),
            "raw_family": m.get("raw_family"),
            "raw_protected_family": bool(m.get("raw_protected_family", False)),
            "raw_generic_approximant": bool(m.get("raw_generic_approximant", False)),
            "inverse_y_transform_wrapped": bool(m.get("inverse_y_transform_wrapped", False)),
            "original_y_val_loss": m.get("original_y_val_loss"),
            "phi_val_loss": m.get("val_loss"),
            "accepted_labels": list(m.get("accepted_labels") or []),
            "coefficient_metadata": copy.deepcopy(
                getattr(state, "coefficient_metadata", None)
            ),
            "coefficient_metadata_by_dataset": copy.deepcopy(
                getattr(state, "coefficient_metadata_by_dataset", None)
            ),
            "dataset_ids": copy.deepcopy(getattr(state, "dataset_ids", None)),
        },
    }


def _stageB_shadow_rescue_reason(state) -> str:
    """Return why a retained Stage-A fit-link branch should be activated.

    Shadow branches are intentionally rare and expensive.  They are only used
    when the active Stage-B result has not earned a branch-safe confirmation.
    """
    m = _stageB_candidate_metrics(state)
    if m["bad_loss"]:
        return "active branch ended above acceptable loss"
    if m["num_multivar_nn"] > 0:
        return "active branch still has multivariate NN atoms"
    if m["generic_approximant"]:
        detail = m["generic_approximant_reason"] or "generic approximation family"
        return f"active branch ended in generic approximant ({detail})"
    if m["num_nn"] > 0 and m["accepted_patterns"] <= 0:
        return "active branch made no confirmed analytic rewrite"
    return ""


def _identity_outer_affine_units_ok(
    z_expr,
    units_payload,
    *,
    intercept,
    y_scale,
    intercept_rel_tol=1.0e-8,
):
    """Return whether an identity φ(y)≈a*z+b shortcut is unit-safe.

    With units enabled this shortcut is deliberately narrower than a general
    affine certificate: it only means "stay in identity y-space" when z has the
    same dimension as y and the offset is negligible.  Stage B still proves the
    actual analytic rewrite under the normal UnitsSpec.
    """
    if not isinstance(units_payload, dict):
        return True, "units-not-enforced"

    try:
        y_scale = abs(float(y_scale))
    except Exception:
        y_scale = 1.0
    y_scale = max(y_scale, 1.0e-12)
    try:
        intercept_rel = abs(float(intercept)) / y_scale
    except Exception:
        return False, "identity-affine intercept is non-finite"
    if (not math.isfinite(intercept_rel)) or intercept_rel > float(intercept_rel_tol):
        return (
            False,
            f"identity-affine offset too large for unit-safe shortcut "
            f"(rel={intercept_rel:.3g})",
        )

    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim

        us = units_payload.get("unit_system")
        x_dims = tuple(units_payload.get("x_dims", ()))
        y_dim = units_payload.get("y_dim", None)
        z_dim = eval_analytic_expr_dim(z_expr, x_dims)
    except Exception as exc:
        return False, f"could not infer compound units ({type(exc).__name__}: {exc})"

    if z_dim is None or y_dim is None:
        return False, "could not infer compound or target units"
    if z_dim != y_dim:
        try:
            z_s = us.format_dim(z_dim) if us is not None else str(z_dim)
            y_s = us.format_dim(y_dim) if us is not None else str(y_dim)
        except Exception:
            z_s = str(z_dim)
            y_s = str(y_dim)
        return False, f"compound dim {z_s} != y dim {y_s}"
    return True, "compound dim matches y and offset is negligible"


@dataclass(frozen=True)
class CompoundCoordVariant:
    """A coordinate choice for probing an outer affine map on a 1D compound."""

    name: str
    display: str
    expr: object


def _compound_coordinate_variants(z_expr):
    """Return the branch-safe coordinate variants tried for a discovered z(x)."""
    from nestynet_sr.sr_core.bridges import PowNode

    return (
        CompoundCoordVariant("z", "z", z_expr),
        CompoundCoordVariant("z_inv", "1/z", PowNode(z_expr, -1.0)),
    )


def _compound_coordinate_variant_values(z_expr, x_values, Nxvars):
    """Evaluate z and 1/z on the supplied points without adding new policies."""
    from nestynet_sr.sr_core.bridges import eval_input_expr

    z_values = eval_input_expr(z_expr, x_values[:, :Nxvars]).view(-1)
    variants = _compound_coordinate_variants(z_expr)
    out = [(variants[0], z_values)]

    z_inv = torch.full_like(z_values, float("nan"))
    finite_nonzero = torch.isfinite(z_values) & (z_values != 0)
    if int(finite_nonzero.sum().item()) > 0:
        z_inv[finite_nonzero] = 1.0 / z_values[finite_nonzero]
        out.append((variants[1], z_inv))
    return out


def _probe_compound_outer_affine_variants(
    *,
    y_values,
    z_expr,
    x_values,
    Nxvars,
    transform_names,
    units_payload=None,
    rms_thr=1.0e-6,
    dom_thr=0.995,
    min_points=256,
    min_domain_frac=0.20,
):
    """Probe φ(y)≈a*z+b over z and 1/z, returning one best entry per transform.

    The caller supplies thresholds from SearchHyperparams; this helper only
    centralizes coordinate variants, affineness/unit checks, and de-duplication.
    """
    from nestynet_sr.sr_search.outer_peel import probe_affine_outer_peels_on_z

    y_values = y_values.view(-1)
    coord_values = _compound_coordinate_variant_values(z_expr, x_values, Nxvars)
    if not coord_values:
        return [], {}, False

    y_aff_scale = float(torch.median(y_values.abs()).item())
    y_aff_scale = max(abs(y_aff_scale), 1.0e-12)

    best_entries_by_name = {}
    for variant, z_coord in coord_values:
        ranked_i = probe_affine_outer_peels_on_z(
            y=y_values,
            z=z_coord,
            transform_names=transform_names,
            min_points=int(min_points),
            min_domain_frac=float(min_domain_frac),
        )

        finite = torch.isfinite(z_coord)
        if int(finite.sum().item()) > 0:
            z_scale = float(torch.median(z_coord[finite].abs()).item())
        else:
            z_scale = 1.0e-12
        z_scale = max(abs(z_scale), 1.0e-12)

        for r in ranked_i:
            details = dict(getattr(r, "details", {}) or {})
            details["coordinate"] = variant.name
            details["coordinate_display"] = variant.display
            fit_kind = str(details.get("fit_kind", "affine"))
            q2 = float(details.get("q2", 0.0) or 0.0)
            lin_scale = max(
                abs(float(getattr(r, "a", 0.0))) * z_scale
                + abs(float(getattr(r, "b", 0.0))),
                1.0e-12,
            )
            quad_rel = abs(q2) * z_scale * z_scale / lin_scale
            simple_affine = (fit_kind == "affine") or (quad_rel <= 1.0e-6)
            confirmed = bool(
                math.isfinite(float(r.rms_rel))
                and float(r.rms_rel) <= float(rms_thr)
                and float(r.domain_ok_frac) >= float(dom_thr)
                and simple_affine
            )
            unit_safe = True
            unit_reason = "not checked"
            if str(r.name) == "identity":
                unit_safe, unit_reason = _identity_outer_affine_units_ok(
                    variant.expr,
                    units_payload,
                    intercept=float(getattr(r, "b", 0.0)),
                    y_scale=y_aff_scale,
                    intercept_rel_tol=max(1.0e-8, 10.0 * float(rms_thr)),
                )
                confirmed = bool(confirmed and unit_safe)

            entry = {
                **r.__dict__,
                "details": details,
                "confirmed": bool(confirmed),
                "rms_rel": float(r.rms_rel),
                "domain_ok_frac": float(r.domain_ok_frac),
                "a": float(r.a),
                "b": float(r.b),
                "coordinate": variant.name,
                "coordinate_display": variant.display,
                "fit_kind": fit_kind,
                "q2": q2,
                "quad_rel": float(quad_rel),
                "unit_safe": bool(unit_safe),
                "unit_reason": str(unit_reason),
            }
            old = best_entries_by_name.get(str(r.name))
            old_key = (
                0 if bool(old and old.get("confirmed", False)) else 1,
                float(old.get("rms_rel", float("inf"))) if old else float("inf"),
            )
            new_key = (
                0 if bool(confirmed) else 1,
                float(r.rms_rel),
            )
            if old is None or new_key < old_key:
                best_entries_by_name[str(r.name)] = entry

    entries = sorted(
        best_entries_by_name.values(),
        key=lambda e: (
            0 if bool(e.get("confirmed", False)) else 1,
            float(e.get("rms_rel", float("inf"))),
        ),
    )
    by_name = {
        str(e.get("name")): {
            k: v
            for k, v in e.items()
            if k not in {"details", "name", "n_points"}
        }
        for e in entries
    }
    identity_confirmed = bool(
        by_name.get("identity", {}).get("confirmed", False)
    )
    return entries, by_name, identity_confirmed


def _loss_scale_from_loader_raw_y(loader, device, *, loss_in_mad_units: bool) -> float:
    """Return the raw-y loss scale used by the identity branch."""
    if not bool(loss_in_mad_units):
        return 1.0
    ys = []
    try:
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                y = batch[1].to(device=device, dtype=torch.float64).reshape(-1)
                ys.append(y.detach())
    except Exception:
        return 1.0
    if not ys:
        return 1.0
    y_all = torch.cat(ys, dim=0)
    y_all = y_all[torch.isfinite(y_all)]
    if int(y_all.numel()) == 0:
        return 1.0
    med = torch.median(y_all)
    mad = torch.median(torch.abs(y_all - med))
    mad_f = float(mad.item())
    if not math.isfinite(mad_f) or mad_f <= 0.0:
        return 1.0
    return float(mad_f * mad_f)


def _strong_phase_hints_for_direct_closure(phase_hints):
    """Select PhaseScan hints worth trying before training a surrogate."""
    strong = []
    for hint in list(phase_hints or []):
        try:
            r2_phase = float(getattr(hint, "r2_phase", float("-inf")))
            r2_trend = float(getattr(hint, "r2_trend", 0.0))
            support_fraction = float(getattr(hint, "support_fraction", 0.0))
            unit_status = str(getattr(hint, "unit_status", "unchecked"))
        except Exception:
            continue
        trend_ref = max(0.0, r2_trend if math.isfinite(r2_trend) else 0.0)
        if r2_phase < 0.98:
            continue
        if (r2_phase - trend_ref) < 0.50:
            continue
        if support_fraction < 0.98:
            continue
        if unit_status not in {"dimensionless", "unchecked"}:
            continue
        strong.append(hint)
    return strong[:4]


def _strong_phase_context_hints_for_direct_closure(phase_context_hints):
    """Select contextual PhaseScan hints worth trying before Stage A."""
    strong = []
    for hint in list(phase_context_hints or []):
        try:
            r2_phase = float(getattr(hint, "r2_context_phase", float("-inf")))
            delta = float(getattr(hint, "delta_r2_phase", float("-inf")))
            unit_status = str(getattr(hint, "unit_status", "unchecked"))
            details = getattr(hint, "details", {}) if isinstance(getattr(hint, "details", None), dict) else {}
            support_fraction = float(details.get("support_fraction", 0.0))
            family = str(getattr(hint, "waveform_family", ""))
        except Exception:
            continue
        if family not in {"one_minus_cos", "contextual_fourier"}:
            continue
        if r2_phase < 0.98:
            continue
        if delta < 0.50:
            continue
        if support_fraction < 0.98:
            continue
        if unit_status not in {"dimensionless", "unchecked"}:
            continue
        strong.append(hint)
    return strong[:4]


def _strong_outer_link_hints_for_direct_closure(outer_link_hints):
    """Select inside-out inverse-link hints worth trying before Stage A."""
    strong = []
    for hint in list(outer_link_hints or []):
        try:
            r2 = float(getattr(hint, "r2", float("-inf")))
            rms_rel = float(getattr(hint, "rms_rel", float("inf")))
            domain_ok = float(getattr(hint, "domain_ok_frac", 0.0))
            branch_ok = float(getattr(hint, "branch_ok_frac", 0.0))
            unit_status = str(getattr(hint, "unit_status", "unchecked"))
        except Exception:
            continue
        if r2 < 0.995:
            continue
        if rms_rel > 1.0e-3:
            continue
        if domain_ok < 0.995 or branch_ok < 0.995:
            continue
        if unit_status not in {"dimensionless", "unchecked"}:
            continue
        strong.append(hint)
    return strong[:4]


def _phase_prescan_training_arrays(filepaths, *, Nxvars, np_dtype, data_hp):
    """Return raw identity training-split arrays for PhaseScan hint generation."""

    paths = list(filepaths or [])
    if not paths:
        return None, None
    xs = []
    ys = []
    for fp in paths:
        try:
            ds = nestynet.dataloader.PhysDataset(
                str(fp),
                mode="train",
                # Source-order split, matching the rest of the SR pipeline.
                split_policy="contiguous",
                data_loader=nestynet.dataloader.get_csv_data_as_pandas,
                ndata_select=data_hp.ndata_select,
                ndata_select_val=data_hp.ndata_select_val,
                Nxvars=Nxvars,
                np_dtype=np_dtype,
                y_operator=None,
            )
        except Exception:
            return None, None
        if getattr(ds, "y_op_error", False):
            return None, None
        try:
            x_arr = np.asarray(ds.x_arr, dtype=np.float64)
            y_arr = np.asarray(ds.y_ref_arr, dtype=np.float64).reshape(-1)
        except Exception:
            return None, None
        if x_arr.ndim != 2 or x_arr.shape[0] == 0 or y_arr.shape[0] == 0:
            return None, None
        n = min(int(x_arr.shape[0]), int(y_arr.shape[0]))
        xs.append(x_arr[:n])
        ys.append(y_arr[:n])
    if not xs or not ys:
        return None, None
    try:
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
    except Exception:
        return None, None


def _try_phase_prescan_direct_closure(
    *,
    phase_hints,
    phase_context_hints,
    outer_link_hints,
    initial_ast,
    filepath,
    filepaths,
    Nxvars,
    np_dtype,
    data_hp,
    lm_hp,
    device,
    dtype,
    fresh_nn_factory,
    units_payload,
    enforce_units,
    units_policy,
    nn_units_semantics,
    model_output,
    model_sep_output,
):
    """Try exact visible phase closures before Stage A trains a surrogate."""
    strong_hints = _strong_phase_hints_for_direct_closure(phase_hints)
    strong_context_hints = _strong_phase_context_hints_for_direct_closure(phase_context_hints)
    strong_outer_link_hints = _strong_outer_link_hints_for_direct_closure(outer_link_hints)
    if not strong_hints and not strong_context_hints and not strong_outer_link_hints:
        return None
    if len(filepaths) != 1:
        print("[PhaseScan Direct] Strong phase hint found; skipping pre-Stage-A closure in multi-dataset mode.")
        return None

    try:
        from nestynet_sr.sr_core import collect_nn_atoms
        from nestynet_sr.sr_core.bridges import AtomNode
    except Exception:
        return None

    try:
        nn_atoms = collect_nn_atoms(initial_ast)
    except Exception:
        return None
    if len(nn_atoms) != 1 or initial_ast is not nn_atoms[0] or not isinstance(nn_atoms[0], AtomNode):
        return None

    print(
        "[PhaseScan Direct] Strong phase/outer-link hint detected; "
        "trying direct analytic closure before Stage A surrogate training."
    )
    for hint in strong_hints:
        try:
            print(f"[PhaseScan Direct]   {hint.carrier_label} (R2_phase={float(hint.r2_phase):.4g})")
        except Exception:
            pass
    for hint in strong_context_hints:
        try:
            print(
                f"[PhaseScan Direct]   {hint.carrier_label} "
                f"(context ΔR2={float(hint.delta_r2_phase):.4g}, "
                f"R2={float(hint.r2_context_phase):.4g})"
            )
        except Exception:
            pass
    for hint in strong_outer_link_hints:
        try:
            print(
                f"[PhaseScan Direct]   {hint.link_name}({hint.carrier_label}) "
                f"(R2={float(hint.r2):.4g}, rel={float(hint.rms_rel):.2e})"
            )
        except Exception:
            pass

    try:
        from nestynet_sr.sr_search.data_utils import build_datasets

        _, _, train_loader, val_loader = build_datasets(
            filepath=str(filepath),
            Nxvars=Nxvars,
            np_dtype=np_dtype,
            data_hp=data_hp,
            y_op=None,
        )
    except Exception as e:
        print(f"[PhaseScan Direct] Dataset build failed; falling back to Stage A. ({e})")
        return None
    if train_loader is None or val_loader is None:
        print("[PhaseScan Direct] Dataset build returned no loader; falling back to Stage A.")
        return None

    try:
        from nestynet_sr.sr_search.stageB.engine import StageBContext, StageBState
        from nestynet_sr.sr_search.stageB.rules import (
            RuleInverseTrigOuterClosure,
            RulePhaseContextTrigClosure,
            RulePhaseHintTrigClosure,
            RulePhaseHintReciprocalTrigPower,
        )
    except Exception as e:
        print(f"[PhaseScan Direct] Stage-B closure machinery unavailable; falling back to Stage A. ({e})")
        return None

    units_spec = None
    if units_payload is not None:
        try:
            from nestynet_sr.sr_core.units import UnitsSpec

            units_spec = UnitsSpec(
                unit_system=units_payload["unit_system"],
                x_dims=units_payload["x_dims"],
                y_dim=units_payload["y_dim"],
                y_transform_name="identity",
                policy=units_policy,
                nn_semantics=nn_units_semantics,
                free_const_dims=units_payload.get("free_const_dims", {}),
                free_const_scope=units_payload.get("free_const_scope", {}),
                fixed_const_dims=units_payload.get("fixed_const_dims", {}),
                fixed_const_values=units_payload.get("fixed_const_values", {}),
                fixed_const_mode=units_payload.get("fixed_const_mode", "strict"),
            )
        except Exception as e:
            if enforce_units:
                print(f"[PhaseScan Direct] Units setup failed; falling back to Stage A. ({e})")
                return None

    loss_scale = _loss_scale_from_loader_raw_y(
        train_loader,
        device,
        loss_in_mad_units=bool(getattr(lm_hp, "loss_in_MAD_units", False)),
    )
    loss_target_raw = float(getattr(lm_hp, "loss_target", 1.0e-7)) * float(loss_scale)
    try:
        noise_floor = getattr(lm_hp, "acceptance_noise_floor_raw", None)
        if noise_floor is not None and math.isfinite(float(noise_floor)):
            loss_target_raw = max(loss_target_raw, float(noise_floor))
    except Exception:
        pass

    state0 = StageBState(
        root=initial_ast,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=float("inf"),
    )
    ctx = StageBContext(
        state=state0,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device,
        dtype=dtype,
        epochs_stageB=min(300, int(getattr(lm_hp, "epochs", 300) or 300)),
        loss_scale=float(loss_scale),
        loss_good_enough_raw=float(loss_target_raw),
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        phase_hints=list(strong_hints),
        phase_context_hints=list(strong_context_hints),
        outer_link_hints=list(strong_outer_link_hints),
        verbose=True,
        fresh_nn_factory=fresh_nn_factory,
        atom_factory=None,
        disabled_patterns=set(),
        enabled_patterns=[],
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
    )

    rules = [
        RuleInverseTrigOuterClosure(),
        RulePhaseHintTrigClosure(),
        RulePhaseHintReciprocalTrigPower(),
        RulePhaseContextTrigClosure(),
    ]
    targets = []
    for rule in rules:
        targets.extend((rule, target) for target in list(rule.iter_targets(ctx)))
    if not targets:
        print("[PhaseScan Direct] No direct-closure target; falling back to Stage A.")
        return None
    candidates = []
    for rule, target in targets:
        candidates.extend(rule.propose(ctx, target))
    if not candidates:
        print("[PhaseScan Direct] No direct analytic phase closure validated by the screen; falling back to Stage A.")
        return None

    best_state = None
    for cand in candidates:
        label = str(getattr(cand, "label", "phase_hint"))
        meta = getattr(cand, "meta", {}) if isinstance(getattr(cand, "meta", None), dict) else {}
        if meta.get("log"):
            print(
                str(meta["log"])
                .replace("[Stage B PhaseHint]", "[PhaseScan Direct]")
                .replace("[Stage B PhaseContext]", "[PhaseScan Direct]")
                .replace("[Stage B OuterLink]", "[PhaseScan Direct]")
            )
        try:
            cand_state = ctx.fit_candidate(cand, epochs_override=ctx.epochs_stageB)
        except Exception as e:
            print(f"[PhaseScan Direct] Candidate {label} fit failed: {e}")
            continue
        val_loss = float(getattr(cand_state, "val_loss", float("inf")))
        print(
            f"[PhaseScan Direct] Candidate {label}: val-loss={val_loss:.4e}, "
            f"target={float(loss_target_raw):.4e}"
        )
        if best_state is None or val_loss < float(best_state.val_loss):
            best_state = cand_state
        if math.isfinite(val_loss) and val_loss <= float(loss_target_raw):
            cand_state.enabled_patterns = [str(meta.get("pattern", "phase_hint_trig_closure"))]
            cand_state.loss_scale = float(loss_scale)
            cand_state.loss_good_enough_eff = float(loss_target_raw)
            try:
                cand_state.num_nn_atoms = 0
                cand_state.num_multivar_nn_atoms = 0
                cand_state.max_nn_arity = 0
            except Exception:
                pass
            model = cand_state.model
            try:
                model._stageA_initial_val_loss = float(val_loss)
                model._stageA_initial_n_params = int(model.num_parameters()) if hasattr(model, "num_parameters") else None
                model._stageA_val_loss_agg = float(val_loss)
                model._best_val_loss_base = float(val_loss) / max(float(loss_scale), 1.0e-30)
                model._phase_prescan_direct_closure = label
            except Exception:
                pass

            save_dict = {
                "y_op": None,
                "y_op_inv": None,
                "Nxvars": Nxvars,
                "num_segments": None,
                "dual_layer": False,
                "model_state_dict": model.state_dict(),
                "ast": cand_state.root,
                "val_loss": float(val_loss),
                "fit_y_link": getattr(lm_hp, "fit_y_link", None),
                "fit_y_link_scale": float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                "phase_prescan_direct_closure": label,
            }
            for out_path in (model_output, model_sep_output):
                try:
                    torch.save(save_dict, out_path)
                except Exception as e:
                    print(f"[PhaseScan Direct] Warning: failed to save {out_path}: {e}")
            print(f"{GREEN}[PhaseScan Direct] Accepted {label}; skipping Stage A surrogate training.{RESET}")
            return {
                "model": model,
                "ast": cand_state.root,
                "val_loss": float(val_loss),
                "state": cand_state,
                "label": label,
                "loss_scale": float(loss_scale),
            }

    if best_state is not None:
        print(
            f"[PhaseScan Direct] Best direct closure missed target "
            f"({float(best_state.val_loss):.4e} > {float(loss_target_raw):.4e}); "
            "falling back to Stage A."
        )
    return None


def _print_compound_outer_affine_entries(title, entries, *, limit=5, domain_digits=3):
    print(title)
    for j, meta in enumerate(list(entries or [])[: int(limit)]):
        mark = " CONFIRMED" if bool(meta.get("confirmed", False)) else " provisional"
        unit_note = (
            f", units={meta.get('unit_reason')}"
            if str(meta.get("name")) == "identity" and meta.get("unit_reason")
            else ""
        )
        coord_note = f", coord={meta.get('coordinate', 'z')}"
        print(
            f"  {j + 1:2d}) {str(meta.get('name')):10s} "
            f"rms_rel≈{float(meta.get('rms_rel', float('inf'))):.3g} "
            f"domain={float(meta.get('domain_ok_frac', 0.0)):.{int(domain_digits)}f} "
            f"a≈{float(meta.get('a', float('nan'))):.4g} "
            f"b≈{float(meta.get('b', float('nan'))):.4g} "
            f"[{meta.get('fit_kind', 'affine')}, "
            f"q2rel≈{float(meta.get('quad_rel', float('inf'))):.3g}"
            f"{coord_note}{unit_note}]{mark}"
        )
