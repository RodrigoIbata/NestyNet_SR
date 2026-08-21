# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Stage-A closure, dimensional prefactors, and shared-response proposals."""

from typing import TYPE_CHECKING
import itertools
import math
import os
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from nestynet_sr.sr_core import Var, ast_to_human_readable, build_monomial_ast, collect_nn_atoms, replace_atom_in_ast
from nestynet_sr.sr_core.bridges import AtomNode, MulNode, Node, PowNode, Scale, _collect_var_idxs_from_node, ast_equals, clone_ast, effective_arity, eval_input_expr, get_input_exprs, has_nontrivial_input, is_trivial_input
from nestynet_sr.sr_core.carrier_units import mark_stagea_buckingham_deferred
from .ast_utils import compact_expression_repr as _compact_expression_repr
from .monomial_peel_plan import expand_forced_power_vector
from .model_builders import build_composite_ast
from .model_selection import compute_accept_threshold as _compute_accept_threshold, noise_equivalent as _noise_equivalent, resolve_acceptance_noise_floor_raw as _resolve_acceptance_noise_floor_raw
from .stagea_fit_tournament import fit_stageA_candidate_with_tournament

from ._search_shadow import (
    GREEN,
    RED,
    RESET,
    YELLOW,
    _analytic_units_rejection,
    _apply_fit_link_to_model,
    _clone_reuse_leaves,
    _compound_buckingham_min_freedom,
    _loss_str,
    _stageA_compound_buckingham_target_dim,
    _stageA_coordinate_collapse_screen,
)
from ._search_detection import (
    _build_compound_candidate_ast,
    _get_direct_integer_scaling_evidence,
)
from ._search_training import (
    _build_tag_to_leaf_map,
    _eval_ast_subtree_on_data,
    _flatten_additive_terms,
    _rebuild_additive_chain,
)

if TYPE_CHECKING:
    from ._search_policy import (
        _accept_threshold_with_structural_target,
        _format_stageA_compound_shortlist_committee_report,
        _nn_split_signature,
        _stageA_compound_shortlist_committee_rank,
        _stageA_terminal_closure_committee_gate,
        _stageA_under_protest_threshold_cap,
    )
    from ._search_proposals import (
        _compound_candidate_payoff_policy,
        _stageA_ast_fingerprint,
        _stageA_parent_context_descriptor,
        _stageA_x_transform_fingerprint,
        _try_stageA_composite_closure_candidate,
    )

def _stageA_composite_closure_applicable(
    *,
    kind: str,
    extra_var_idxs: Optional[List[int]],
    extra_input_asts: Optional[List[Node]],
    prefactor_exps,
) -> bool:
    """Return whether a compound proposal is eligible for visible analytic closure.

    This is intentionally narrow: the local oracle may rank many coordinate
    rewrites, but Stage A should only close a leaf when the proposed composite
    consumes the full leaf and is already analytic in the visible AST.
    """
    return _stageA_composite_closure_skip_reason(
        kind=kind,
        extra_var_idxs=extra_var_idxs,
        extra_input_asts=extra_input_asts,
        prefactor_exps=prefactor_exps,
    ) is None


def _stageA_composite_closure_skip_reason(
    *,
    kind: str,
    extra_var_idxs: Optional[List[int]],
    extra_input_asts: Optional[List[Node]],
    prefactor_exps,
) -> Optional[str]:
    """Return why a compound proposal cannot be a visible Stage-A closure."""
    kind_l = str(kind or "").lower()
    if kind_l not in {
        "mixed",
        "mixed_scaling",
        "var_times_trig",
        "shadow_composite",
        "shadow_preserved_factor",
        "metric_distance",
    }:
        return f"kind={kind_l or 'unknown'} is coordinate-compression only"
    if extra_var_idxs or extra_input_asts:
        parts = []
        if extra_var_idxs:
            parts.append(f"raw extras={list(extra_var_idxs)}")
        if extra_input_asts:
            parts.append(f"compound extras={len(extra_input_asts)}")
        return "proposal leaves preserved extras (" + ", ".join(parts) + ")"
    if prefactor_exps is not None:
        return "proposal already includes a prefactor peel"
    return None


def _stageA_composite_scale_init(z_expr: Node, x_train, y_teacher) -> float:
    """Least-squares initial value for the scalar in ``scale * z_expr``."""
    if x_train is None or y_teacher is None:
        return 1.0
    try:
        z = eval_input_expr(z_expr, x_train).reshape(-1).to(dtype=torch.float64)
        y = y_teacher.reshape(-1).to(dtype=torch.float64)
        n = min(int(z.numel()), int(y.numel()))
        if n <= 0:
            return 1.0
        z = z[:n]
        y = y[:n]
        mask = torch.isfinite(z) & torch.isfinite(y)
        if int(mask.sum().item()) < 8:
            return 1.0
        z = z[mask]
        y = y[mask]
        den = torch.sum(z * z)
        if (not torch.isfinite(den)) or float(den.detach().cpu()) <= 1.0e-30:
            return 1.0
        val = torch.sum(z * y) / den
        val_f = float(val.detach().cpu())
        if not math.isfinite(val_f):
            return 1.0
        return val_f
    except Exception:
        return 1.0


def _stageA_composite_scalar_atom(
    *,
    current_ast: Node,
    atom: AtomNode,
    z_expr: Node,
    units_spec,
    enforce_units: bool,
    init: float,
) -> AtomNode:
    """Build the scalar coefficient for a Stage-A analytic composite closure."""
    base_tag = f"{getattr(atom, 'tag', None) or 'leaf'}_stageA_comp_scale"
    if bool(enforce_units) and units_spec is not None:
        try:
            from nestynet_sr.sr_core.constants import make_unit_aware_scalar_atom
            from nestynet_sr.sr_core.units import (
                eval_analytic_expr_dim,
                infer_atom_output_dim,
                sub_dim,
            )

            target_dim = infer_atom_output_dim(current_ast, atom, units_spec)
            z_dim = eval_analytic_expr_dim(z_expr, units_spec.x_dims)
            required_dim = sub_dim(target_dim, z_dim) if (target_dim is not None and z_dim is not None) else None
            return make_unit_aware_scalar_atom(
                required_dim,
                units_spec,
                base_tag=base_tag,
                init=float(init),
                strict=True,
            )
        except Exception as exc:
            raise ValueError(f"could not build unit-aware scalar: {exc}") from exc

    return AtomNode(
        kind="scale",
        var_idxs=(),
        kwargs={"name": base_tag, "init": float(init)},
        tag=base_tag,
    )


_FORCED_MONOMIAL_POWERS = (
    Fraction(-3, 1),
    Fraction(-5, 2),
    Fraction(-2, 1),
    Fraction(-3, 2),
    Fraction(-1, 1),
    Fraction(-1, 2),
    Fraction(1, 2),
    Fraction(1, 1),
    Fraction(3, 2),
    Fraction(2, 1),
    Fraction(5, 2),
    Fraction(3, 1),
)
_FORCED_MONOMIAL_DENSE_MAX_TERMS = 3_000_000
_FORCED_MONOMIAL_SPARSE_MAX_SUPPORT = 4
_FORCED_MONOMIAL_SPARSE_MAX_TERMS = 2_000_000


def _stageA_forced_monomial_reason(reason: Optional[str]) -> bool:
    """Return True when a Buckingham rejection says only a monomial is possible."""
    return "forces monomial" in str(reason or "").lower()


def _stageA_dim_power_sum(dims, powers):
    if not dims or not powers or len(dims) != len(powers):
        return None
    try:
        n = len(tuple(dims[0]))
        out = [Fraction(0) for _ in range(n)]
        for dim, power in zip(dims, powers):
            if dim is None:
                return None
            for i, exp in enumerate(tuple(dim)):
                out[i] += Fraction(power) * Fraction(exp)
        return tuple(out)
    except Exception:
        return None


def _stageA_forced_monomial_expr_from_units(
    *,
    current_ast: Node,
    atom: AtomNode,
    z_expr: Node,
    extra_var_idxs: Optional[List[int]],
    extra_input_asts: Optional[List[Node]],
    units_spec,
    enforce_units: bool,
    max_dense_terms: int = _FORCED_MONOMIAL_DENSE_MAX_TERMS,
    max_sparse_support: int = _FORCED_MONOMIAL_SPARSE_MAX_SUPPORT,
    max_sparse_terms: int = _FORCED_MONOMIAL_SPARSE_MAX_TERMS,
) -> Tuple[Optional[Node], Optional[Tuple[Fraction, ...]], str]:
    """Build the visible monomial forced by dimensional analysis, if small.

    When the Buckingham check says a compound has destroyed all dimensionless
    freedom, accepting ``NN[z]`` would only postpone an already-determined
    monomial closure.  This helper constructs that explicit monomial over the
    proposed coordinate and any preserved extras, then the normal Stage-A
    validation path decides whether it is actually correct.
    """
    if (not bool(enforce_units)) or units_spec is None:
        return None, None, "units disabled"

    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim, infer_atom_output_dim

        target_dim = infer_atom_output_dim(current_ast, atom, units_spec)
        if target_dim is None:
            return None, None, "could not infer target dimension"

        bases: List[Node] = [clone_ast(z_expr)]
        dims = [eval_analytic_expr_dim(z_expr, units_spec.x_dims)]
        for idx in list(extra_var_idxs or ()):
            try:
                i = int(idx)
            except Exception:
                continue
            bases.append(Var(i))
            dims.append(units_spec.x_dims[i])
        for expr in list(extra_input_asts or ()):
            bases.append(clone_ast(expr))
            dims.append(eval_analytic_expr_dim(expr, units_spec.x_dims))

        if not bases or any(d is None for d in dims):
            return None, None, "unknown monomial basis dimension"

        n_basis = len(bases)
        matches: List[Tuple[Tuple[Fraction, ...], int, Fraction]] = []
        dense_terms = len(_FORCED_MONOMIAL_POWERS) ** int(n_basis)
        use_dense = int(dense_terms) <= int(max_dense_terms)
        sparse_limit = max(0, min(int(max_sparse_support), int(n_basis)))

        def _record_if_match(powers) -> None:
            powers_t = tuple(Fraction(p) for p in powers)
            if all(p == 0 for p in powers_t):
                return
            dim_sum = _stageA_dim_power_sum(dims, powers_t)
            if dim_sum is None or tuple(dim_sum) != tuple(target_dim):
                return
            complexity = sum(1 for p in powers_t if p != 0)
            abs_sum = sum(abs(p) for p in powers_t)
            matches.append((powers_t, complexity, abs_sum))

        if use_dense:
            # Preserve the historical behavior for small bases: every basis
            # term receives one of the allowed nonzero powers.
            for powers in itertools.product(_FORCED_MONOMIAL_POWERS, repeat=n_basis):
                _record_if_match(powers)
        elif sparse_limit > 0:
            # High-dimensional early compound probes can otherwise spend hours
            # in Fraction arithmetic before any candidate is logged.  Search
            # sparse visible monomials instead, increasing support size; once a
            # given support has a match, larger supports cannot win the
            # complexity ordering below.
            zero_powers = [Fraction(0) for _ in range(n_basis)]
            for support in range(1, sparse_limit + 1):
                level_terms = math.comb(int(n_basis), int(support)) * (
                    len(_FORCED_MONOMIAL_POWERS) ** int(support)
                )
                if int(level_terms) > int(max_sparse_terms):
                    break
                before = len(matches)
                for idxs in itertools.combinations(range(n_basis), support):
                    for vals in itertools.product(_FORCED_MONOMIAL_POWERS, repeat=support):
                        powers = list(zero_powers)
                        for idx, val in zip(idxs, vals):
                            powers[int(idx)] = Fraction(val)
                        _record_if_match(powers)
                if len(matches) > before:
                    break

        if not matches:
            if not use_dense:
                return (
                    None,
                    None,
                    "no sparse unit-forced monomial power matches target dimension "
                    f"(basis={n_basis}, dense_terms={dense_terms}, "
                    f"sparse_support<={sparse_limit}, sparse_terms<={int(max_sparse_terms)})",
                )
            return None, None, "no small unit-forced monomial power matches target dimension"

        matches.sort(key=lambda item: (item[1], item[2], tuple(float(p) for p in item[0])))
        powers = matches[0][0]
        factors = []
        for base, power in zip(bases, powers):
            if power == 0:
                continue
            if power == 1:
                factors.append(clone_ast(base))
            else:
                factors.append(PowNode(clone_ast(base), float(power)))
        if not factors:
            return None, None, "forced monomial reduced to a constant"
        expr = factors[0]
        for factor in factors[1:]:
            expr = MulNode(expr, factor)
        return expr, powers, ""
    except Exception as exc:
        return None, None, f"forced monomial construction failed: {exc}"


def _stageA_partial_forced_monomial_peel_proposal(
    *,
    current_ast: Node,
    atom: AtomNode,
    z_expr: Node,
    pattern,
    extra_var_idxs: Optional[List[int]],
    extra_input_asts: Optional[List[Node]],
    confidence: float,
    meta: Optional[Dict[str, Any]],
    units_spec,
    enforce_units: bool,
    scaling_features=None,
    scaling_rel_std_threshold: float = 0.08,
    scaling_k_int_threshold: float = 0.20,
) -> Optional[Tuple[Tuple[int, ...], Node, float, Optional[List[int]], Dict[str, Any]]]:
    """Build ``P*NN[residual]`` from evidence-backed integer factors.

    This is the partial homogeneity transaction used when dimensional evidence
    suggests a monomial.  We only peel integer powers whose one-axis scaling
    evidence independently supports that same exponent; every unsupported,
    fractional, or otherwise ambiguous input is left inside the residual NN.
    """

    if meta and bool(meta.get("partial_forced_monomial_peel", False)):
        return None
    if extra_input_asts:
        # v1 keeps the transaction raw/effective-input local.  Compound extras
        # need a separate residual-dimension audit.
        return None

    forced_expr, basis_powers, reason = _stageA_forced_monomial_expr_from_units(
        current_ast=current_ast,
        atom=atom,
        z_expr=z_expr,
        extra_var_idxs=extra_var_idxs,
        extra_input_asts=extra_input_asts,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
    )
    if forced_expr is None or not basis_powers:
        return None

    try:
        pat = tuple(int(v) for v in pattern)
    except Exception:
        return None

    try:
        local_inputs = tuple(get_input_exprs(atom)) if has_nontrivial_input(atom) else tuple(
            Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ()
        )
    except Exception:
        local_inputs = tuple(Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ())
    if len(local_inputs) != len(pat):
        return None

    raw_to_local: Dict[int, int] = {}
    for i, inp in enumerate(local_inputs):
        if is_trivial_input(inp):
            try:
                raw_to_local[int(inp.var_idxs[0])] = int(i)
            except Exception:
                pass

    extra_local_indices = []
    for idx in list(extra_var_idxs or ()):
        try:
            extra_local_indices.append(raw_to_local[int(idx)])
        except Exception:
            return None

    full_powers = expand_forced_power_vector(
        pattern=pat,
        basis_powers=basis_powers,
        extra_local_indices=tuple(extra_local_indices),
    )
    if full_powers is None:
        return None

    clean_scaling_local: Dict[int, Tuple[int, float]] = {}
    if isinstance(meta, dict):
        for row in tuple(meta.get("local_clean_scaling_exponents", ()) or ()):
            try:
                li = int(row[0])
                exp_i = int(row[1])
                rel_i = float(row[2]) if len(row) > 2 else 0.0
            except Exception:
                continue
            if 0 <= li < len(local_inputs):
                clean_scaling_local[li] = (exp_i, rel_i)

    if not clean_scaling_local and scaling_features:
        raw_evidence = _get_direct_integer_scaling_evidence(
            scaling_features,
            var_filter=set(raw_to_local.keys()),
            rel_std_threshold=float(scaling_rel_std_threshold),
            k_int_threshold=float(scaling_k_int_threshold),
            require_oracle=True,
        )
        for raw_idx, (exp_i, rel_i) in raw_evidence.items():
            try:
                clean_scaling_local[int(raw_to_local[int(raw_idx)])] = (int(exp_i), float(rel_i))
            except Exception:
                continue

    if not clean_scaling_local:
        return None

    full = tuple(Fraction(p) for p in full_powers)
    clean_powers: List[int] = []
    residual_indices: List[int] = []
    evidence_used: List[Tuple[int, int, float]] = []
    for i, power in enumerate(full):
        clean_exp = 0
        if power != 0 and power.denominator == 1 and abs(int(power)) <= 8:
            ev = clean_scaling_local.get(int(i))
            if ev is not None and int(ev[0]) == int(power):
                clean_exp = int(power)
                evidence_used.append((int(i), int(ev[0]), float(ev[1])))
        clean_powers.append(int(clean_exp))
        if int(clean_exp) == 0:
            residual_indices.append(int(i))

    if sum(1 for p in clean_powers if int(p) != 0) < 1:
        return None
    if len(residual_indices) < 1:
        return None
    if len(residual_indices) >= len(local_inputs):
        return None

    # The first residual input is represented as the candidate's z input;
    # remaining residual inputs are kept as raw/compound extras.
    first_residual = int(residual_indices[0])
    residual_z = clone_ast(local_inputs[first_residual])
    residual_pattern = [0 for _ in local_inputs]
    residual_pattern[first_residual] = 1

    residual_raw_extras: List[int] = []
    residual_compound_extras: List[Node] = []
    for li in residual_indices[1:]:
        inp = local_inputs[int(li)]
        if is_trivial_input(inp):
            try:
                residual_raw_extras.append(int(inp.var_idxs[0]))
            except Exception:
                return None
        else:
            residual_compound_extras.append(clone_ast(inp))

    peel_meta = dict(meta or {})
    peel_meta["kind"] = "partial_monomial_peel"
    peel_meta["partial_forced_monomial_peel"] = True
    peel_meta["prefactor_exponents"] = tuple(int(v) for v in clean_powers)
    peel_meta["forced_basis_powers"] = tuple(str(p) for p in basis_powers)
    peel_meta["forced_full_powers"] = tuple(str(p) for p in full)
    peel_meta["residual_local_indices"] = tuple(int(i) for i in residual_indices)
    peel_meta["residual_powers"] = tuple(str(full[int(i)]) for i in residual_indices)
    peel_meta["clean_scaling_evidence"] = tuple(evidence_used)
    if residual_compound_extras:
        peel_meta["extra_input_asts"] = tuple(residual_compound_extras)

    return (
        tuple(int(v) for v in residual_pattern),
        residual_z,
        float(confidence) * 0.992,
        list(residual_raw_extras),
        peel_meta,
    )


def _stageA_forced_monomial_loss_equivalent(
    *,
    forced_loss: float,
    reference_loss: Optional[float],
    lm_hp,
    loss_scale: float,
    search_hp=None,
) -> Tuple[bool, str]:
    """Classify whether a forced monomial terminal closure is non-regressing.

    This deliberately uses numerical/noise tolerances, not the broader Stage-A
    target-quality threshold.  The latter is an admission budget for proposals;
    it is not evidence that a worse terminal monomial is equivalent to the
    unresolved branch that produced it.
    """
    try:
        cand = float(forced_loss)
    except Exception:
        return False, "invalid forced monomial loss"
    try:
        ref = float(reference_loss)
    except Exception:
        ref = float("nan")
    if not (math.isfinite(cand) and cand >= 0.0):
        return False, "invalid forced monomial loss"
    if not (math.isfinite(ref) and ref >= 0.0):
        return True, "no finite reference loss"

    try:
        noise = float(_resolve_acceptance_noise_floor_raw(lm_hp, float(loss_scale)))
    except Exception:
        noise = 0.0
    if not math.isfinite(noise) or noise < 0.0:
        noise = 0.0
    try:
        chisq_tol = float(getattr(lm_hp, "chisq_tol", 0.0)) * float(loss_scale)
    except Exception:
        chisq_tol = 0.0
    if math.isfinite(chisq_tol) and chisq_tol > noise:
        noise = chisq_tol

    rel = float(getattr(search_hp, "forced_monomial_equiv_rel", 1.0e-2))
    noise_mult = float(getattr(search_hp, "forced_monomial_equiv_noise_mult", 100.0))
    if not math.isfinite(rel) or rel < 0.0:
        rel = 1.0e-2
    if not math.isfinite(noise_mult) or noise_mult < 0.0:
        noise_mult = 100.0

    tol = max(
        noise_mult * noise,
        rel * max(ref, noise, 1.0e-30),
        1.0e-14 * max(1.0, ref),
    )
    if cand <= ref + tol:
        if cand < ref - tol:
            return True, f"improves reference by {ref - cand:.3e}"
        return True, f"within forced-monomial equivalence tolerance {tol:.3e}"
    return False, f"material regression {ref:.4e}->{cand:.4e} (tol={tol:.3e})"


def _stageA_forced_monomial_leftover_candidates(
    *,
    atom: AtomNode,
    forced_expr: Node,
    z_expr: Node,
    units_spec,
    enforce_units: bool,
    x_train,
    y_teacher,
    search_hp,
    x_transform_map=None,
) -> List[Tuple[Node, float, str]]:
    """Return ranked one-coordinate dimensionless leftovers for ``P*NN[pi]``.

    This is intentionally tiny: it is a post-probe rescue for a forced
    monomial prefactor that was good enough but regressed.  It searches only
    visible, support-1/2 dimensionless coordinates over the current atom's
    local raw variables and uses a y/P collapse screen as a hard gate.
    """
    if (not bool(enforce_units)) or units_spec is None:
        return []
    if x_train is None or y_teacher is None:
        return []

    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim, is_dimless

        local_inputs = tuple(get_input_exprs(atom)) if has_nontrivial_input(atom) else tuple(
            Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ()
        )
        raw_inputs: List[Tuple[int, Any]] = []
        for inp in local_inputs:
            if not is_trivial_input(inp):
                continue
            try:
                idx = int(inp.var_idxs[0])
                raw_inputs.append((idx, units_spec.x_dims[idx]))
            except Exception:
                continue
        if not raw_inputs:
            return []

        with torch.no_grad():
            p_vals = eval_input_expr(forced_expr, x_train)
            y_resid = y_teacher / (p_vals + 1.0e-30)

        screen_bins = int(getattr(search_hp, "compound_screen_bins", 64))
        base_gate = float(getattr(search_hp, "compound_variant_screen_gate", 0.15))
        gate = float(getattr(search_hp, "forced_monomial_prefactor_fallback_screen_gate", max(0.50, base_gate)))
        max_candidates = int(getattr(search_hp, "forced_monomial_prefactor_fallback_max_candidates", 3))
        if max_candidates <= 0:
            return []

        full_raws = tuple(idx for idx, _ in raw_inputs)
        seen: set[Tuple[int, ...]] = set()
        scored: List[Tuple[float, int, int, Tuple[int, ...], Node, str]] = []

        def _canonical_power_key(powers: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
            vals = [int(v) for v in powers]
            if not any(vals):
                return None
            g = 0
            for v in vals:
                if v:
                    g = math.gcd(g, abs(int(v)))
            if g <= 0:
                return None
            vals = [int(v // g) for v in vals]
            first = next((v for v in vals if v != 0), 0)
            if first < 0:
                vals = [-v for v in vals]
            return tuple(vals)

        raw_dim = {int(idx): dim for idx, dim in raw_inputs}
        exponent_values = (-2, -1, 1, 2)
        proposals: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

        # Support-1 dimensionless raw axes.
        for idx, dim in raw_inputs:
            if is_dimless(dim):
                powers = tuple(1 if int(r) == int(idx) else 0 for r in full_raws)
                proposals.append((full_raws, powers))

        # Support-2 sparse monomial ratios/products.
        for (idx0, _d0), (idx1, _d1) in itertools.combinations(raw_inputs, 2):
            for e0, e1 in itertools.product(exponent_values, repeat=2):
                powers = tuple(
                    int(e0) if int(r) == int(idx0) else int(e1) if int(r) == int(idx1) else 0
                    for r in full_raws
                )
                key = _canonical_power_key(powers)
                if key is None or key in seen:
                    continue
                dim_sum = _stageA_dim_power_sum(
                    [raw_dim[int(r)] for r in full_raws],
                    powers,
                )
                if dim_sum is None or not is_dimless(dim_sum):
                    continue
                seen.add(key)
                proposals.append((full_raws, key))

        for raws, powers in proposals:
            if not any(int(v) != 0 for v in powers):
                continue
            try:
                pi_ast = build_monomial_ast(tuple(int(r) for r in raws), tuple(int(p) for p in powers))
                if ast_equals(pi_ast, z_expr):
                    continue
                if eval_analytic_expr_dim(pi_ast, units_spec.x_dims) is None:
                    continue
                pi_vals = eval_input_expr(pi_ast, x_train)
                score = _stageA_coordinate_collapse_screen([pi_vals], y_resid, n_bins=screen_bins)
            except Exception:
                continue
            if float(score) < float(gate):
                try:
                    pi_readable = ast_to_human_readable(pi_ast, x_transform_map)
                except Exception:
                    pi_readable = "pi"
                print(
                    "[Stage A ForcedMonomial] Rejecting prefactor residual candidate "
                    f"pi={pi_readable}: y/P collapse score {float(score):.3f} < {float(gate):.3f}"
                )
                continue
            support = sum(1 for p in powers if int(p) != 0)
            l1 = sum(abs(int(p)) for p in powers)
            try:
                pi_readable = ast_to_human_readable(pi_ast, x_transform_map)
            except Exception:
                pi_readable = "pi"
            scored.append((float(score), int(support), int(l1), tuple(int(p) for p in powers), pi_ast, pi_readable))

        scored.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
        return [(clone_ast(row[4]), float(row[0]), str(row[5])) for row in scored[:max_candidates]]
    except Exception as exc:
        print(f"[Stage A ForcedMonomial] Prefactor residual scan failed: {exc}")
        return []


def _try_stageA_forced_monomial_prefactor_fallback_candidate(
    *,
    model,
    current_ast: Node,
    atom: AtomNode,
    forced_expr: Node,
    pi_expr: Node,
    pi_readable: str,
    score: float,
    kind: str,
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
    current_val_loss: Optional[float],
    stageA_under_protest: bool,
    best_train_loss: Optional[float],
    loss_scale: float,
    model_sep_output,
    y_op,
    y_op_inv,
    Nxvars: int,
    x_transform_map,
    units_spec,
    enforce_units: bool,
    units_reject_cb=None,
    x_train=None,
) -> Tuple[bool, Optional[Any], Optional[Node], Optional[float]]:
    """Try ``forced_expr * NN[pi_expr]`` after a regressing terminal P probe."""
    try:
        cand_ast = _build_compound_candidate_ast(
            current_ast,
            atom,
            pi_expr,
            (1,),
            extra_var_idxs_override=[],
            prefactor_ast=forced_expr,
            extra_input_asts=None,
        )
    except Exception as exc:
        print(f"[Stage A ForcedMonomial] Could not build P*NN[pi] fallback: {exc}")
        return False, None, None, None

    if bool(enforce_units) and units_spec is not None:
        try:
            from nestynet_sr.sr_core.units import check_units_ast, eval_analytic_expr_dim, is_dimless

            pi_dim = eval_analytic_expr_dim(pi_expr, units_spec.x_dims)
            if pi_dim is None or not is_dimless(pi_dim):
                print(
                    "[Stage A ForcedMonomial] Rejecting P*NN[pi] fallback: "
                    f"pi={pi_readable} is not dimensionless"
                )
                return False, None, None, None
            ures = check_units_ast(cand_ast, units_spec)
            if not bool(getattr(ures, "ok", False)):
                reason = getattr(ures, "reason", "unit check failed")
                print(f"[Stage A ForcedMonomial] Rejecting P*NN[pi] fallback: {reason}")
                if callable(units_reject_cb):
                    units_reject_cb("forced_monomial_prefactor_fallback", str(reason))
                return False, None, None, None
        except Exception as exc:
            print(f"[Stage A ForcedMonomial] Rejecting P*NN[pi] fallback: unit check failed: {exc}")
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
        print(f"[Stage A ForcedMonomial] P*NN[pi] fallback build failed: {exc}")
        return False, None, None, None

    try:
        original_leaf = (tag_to_leaf or {}).get(getattr(atom, "tag", None))
        tag_to_leaf_compound = _build_tag_to_leaf_map(cand_ast_updated, temp_model)
        compound_leaf = tag_to_leaf_compound.get(getattr(atom, "tag", None))
    except Exception:
        original_leaf = None
        compound_leaf = None
    if (original_leaf is not None) and (compound_leaf is not None) and (x_train is not None):
        try:
            from .training import pretrain_compound_leaf_from_teacher

            temp_model = pretrain_compound_leaf_from_teacher(
                compound_model=temp_model,
                original_leaf=original_leaf,
                compound_leaf=compound_leaf,
                z_ast=pi_expr,
                x_data=x_train,
                original_var_idxs=list(atom.var_idxs),
                device=device,
                dtype=dtype,
                extra_var_idxs=[],
                extra_input_asts=None,
                prefactor_ast=forced_expr,
                original_input_asts=get_input_exprs(atom),
                epochs=int(getattr(search_hp, "compound_pretrain_epochs", 2000)),
                verbose=bool(getattr(search_hp, "compound_pretrain_verbose", True)),
            )
        except Exception as exc:
            print(f"[Stage A ForcedMonomial] P*NN[pi] pretrain failed: {exc}")

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
    accept_threshold, under_protest_cap = _stageA_under_protest_threshold_cap(
        accept_threshold=accept_threshold,
        current_val_loss=current_val_loss if current_val_loss is not None else best_val_loss,
        loss_floor=loss_target_eff,
        noise_floor=acceptance_noise_floor_raw,
        under_protest=bool(stageA_under_protest),
        label=f"forced monomial prefactor fallback {kind}",
    )

    print(
        "[Stage A ForcedMonomial] Trying prefactor residual fallback "
        f"P*NN[pi] with pi={pi_readable} (collapse={float(score):.3f}); "
        f"accept_threshold={accept_threshold:.4e}"
    )
    if structural_target:
        print(
            "[Stage A ForcedMonomial] P*NN[pi] structural target enabled: "
            f"{_nn_split_signature(current_ast)} → {_nn_split_signature(cand_ast_updated)}"
        )
    if under_protest_cap:
        print("[Stage A ForcedMonomial] Under-protest fallback: requiring non-regressing validation loss.")

    max_train_degradation = float(getattr(search_hp, "max_train_degradation", 100.0))
    lane_train_loss_cap = (
        float("inf")
        if best_train_loss is None or best_train_loss <= 0
        else max(max_train_degradation * best_train_loss, loss_target_eff)
    )

    accepted, best_val_loss_cand, best_train_loss_cand, best_param_vec, temp_opt = fit_stageA_candidate_with_tournament(
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
        y_op=y_op,
        y_op_inv=y_op_inv,
        max_lane_train_loss=lane_train_loss_cap,
        lm_hp=lm_hp,
    )
    if not accepted:
        print(
            "[Stage A ForcedMonomial] Rejected P*NN[pi] fallback "
            f"pi={pi_readable}, val-loss {float(best_val_loss_cand):.4e}"
        )
        return False, None, None, None

    passes_relative = (
        best_train_loss is None
        or best_train_loss <= 0
        or best_train_loss_cand <= max_train_degradation * best_train_loss
    )
    passes_absolute = best_train_loss_cand <= loss_target_eff
    if not passes_relative and not passes_absolute:
        degradation = best_train_loss_cand / best_train_loss if best_train_loss else float("inf")
        print(
            f"{RED}[Stage A ForcedMonomial] Rejected P*NN[pi] fallback{RESET} "
            f"pi={pi_readable}: training loss {degradation:.0f}× worse than current model"
        )
        return False, None, None, None

    temp_opt._update_param_groups(best_param_vec)
    best_val_loss_cand = float(best_val_loss_cand)
    print(
        f"{GREEN}[Stage A ForcedMonomial] Accepted prefactor residual fallback{RESET} "
        f"pi={pi_readable}, val-loss {_loss_str(best_val_loss_cand, lm_hp)}"
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


def _try_stageA_forced_monomial_closure_candidate(
    *,
    model,
    current_ast: Node,
    atom: AtomNode,
    z_expr: Node,
    extra_var_idxs: Optional[List[int]],
    extra_input_asts: Optional[List[Node]],
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
    best_train_loss: Optional[float],
    loss_scale: float,
    model_sep_output,
    y_op,
    y_op_inv,
    Nxvars: int,
    x_transform_map,
    units_spec,
    enforce_units: bool,
    current_val_loss: Optional[float] = None,
    stageA_under_protest: bool = False,
    units_reject_cb=None,
    x_train=None,
    y_teacher=None,
    skip_power_one: bool = False,
) -> Tuple[bool, Optional[Any], Optional[Node], Optional[float], str]:
    """Try the explicit monomial implied by a Buckingham forced-monomial reject."""
    forced_expr, powers, reason = _stageA_forced_monomial_expr_from_units(
        current_ast=current_ast,
        atom=atom,
        z_expr=z_expr,
        extra_var_idxs=extra_var_idxs,
        extra_input_asts=extra_input_asts,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
    )
    if forced_expr is None:
        return False, None, None, None, reason
    if bool(skip_power_one) and powers == (Fraction(1, 1),):
        return False, None, None, None, "power-one closure already tested"

    try:
        forced_readable = ast_to_human_readable(forced_expr, x_transform_map)
    except Exception:
        forced_readable = "forced_monomial(z)"
    print(
        "[Stage A ForcedMonomial] Buckingham forces monomial; "
        f"trying visible closure z_powers={tuple(str(p) for p in powers or ())}: "
        f"{forced_readable}"
    )

    acc, new_model, new_ast, new_loss = _try_stageA_composite_closure_candidate(
        model=model,
        current_ast=current_ast,
        atom=atom,
        z_expr=forced_expr,
        z_readable=forced_readable,
        kind=f"{kind}:forced_monomial",
        confidence=float(confidence),
        tag_to_leaf=tag_to_leaf,
        datagen_train_noshuffle=datagen_train_noshuffle,
        datagen_val_noshuffle=datagen_val_noshuffle,
        device=device,
        dtype=dtype,
        leaf_builder=leaf_builder,
        parent_num_segments=parent_num_segments,
        parent_dual_layer=parent_dual_layer,
        search_hp=search_hp,
        lm_hp=lm_hp,
        loss_target_eff=loss_target_eff,
        accept_threshold_eff_cand=accept_threshold_eff_cand,
        best_val_loss=best_val_loss,
        current_val_loss=current_val_loss,
        stageA_under_protest=bool(stageA_under_protest),
        best_train_loss=best_train_loss,
        loss_scale=loss_scale,
        model_sep_output=model_sep_output,
        y_op=y_op,
        y_op_inv=y_op_inv,
        Nxvars=Nxvars,
        x_transform_map=x_transform_map,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
        units_reject_cb=units_reject_cb,
        x_train=x_train,
        y_teacher=y_teacher,
        buckingham_reason=None,
    )
    if acc:
        reference_loss = current_val_loss if current_val_loss is not None else best_val_loss
        nonregressing, loss_reason = _stageA_forced_monomial_loss_equivalent(
            forced_loss=float(new_loss) if new_loss is not None else float("inf"),
            reference_loss=reference_loss,
            lm_hp=lm_hp,
            loss_scale=loss_scale,
            search_hp=search_hp,
        )
        coe_terminal_veto_reason = None
        if nonregressing:
            coe_ok, coe_reason, coe_summary = _stageA_terminal_closure_committee_gate(
                base_ast=current_ast,
                cand_ast=new_ast,
                base_model=model,
                cand_model=new_model,
                label=f"{kind}:forced_monomial",
                gate_kind="stageA_forced_monomial_closure",
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
                coe_terminal_veto_reason = str(coe_reason)
            else:
                print(
                    f"{GREEN}[Stage A ForcedMonomial] Accepted forced monomial closure{RESET}, "
                    f"val-loss {_loss_str(float(new_loss), lm_hp) if new_loss is not None else 'unknown'} "
                    f"({loss_reason})"
                )
                return True, new_model, new_ast, new_loss, ""

        # The terminal monomial was either a material regression locally or a
        # witness-slice regression under CoE. Keep the dimensionally legal
        # prefactor, but let a bounded residual NN carry any missing
        # dimensionless dependence.
        hold_reason = coe_terminal_veto_reason if coe_terminal_veto_reason else str(loss_reason)
        print(
            "[Stage A ForcedMonomial] Holding forced monomial as visible prefactor: "
            f"{hold_reason}; trying bounded P*NN[pi] rescue before rejecting P."
        )
        leftovers = _stageA_forced_monomial_leftover_candidates(
            atom=atom,
            forced_expr=forced_expr,
            z_expr=z_expr,
            units_spec=units_spec,
            enforce_units=bool(enforce_units),
            x_train=x_train,
            y_teacher=y_teacher,
            search_hp=search_hp,
            x_transform_map=x_transform_map,
        )
        if not leftovers:
            reason = "forced monomial regressed and no dimensionless leftover survived y/P screen"
            print(f"[Stage A ForcedMonomial] Rejected: {reason}")
            return False, None, None, None, reason
        for pi_expr, score, pi_readable in leftovers:
            acc_fb, model_fb, ast_fb, loss_fb = _try_stageA_forced_monomial_prefactor_fallback_candidate(
                model=model,
                current_ast=current_ast,
                atom=atom,
                forced_expr=forced_expr,
                pi_expr=pi_expr,
                pi_readable=pi_readable,
                score=float(score),
                kind=kind,
                tag_to_leaf=tag_to_leaf,
                datagen_train_noshuffle=datagen_train_noshuffle,
                datagen_val_noshuffle=datagen_val_noshuffle,
                device=device,
                dtype=dtype,
                leaf_builder=leaf_builder,
                parent_num_segments=parent_num_segments,
                parent_dual_layer=parent_dual_layer,
                search_hp=search_hp,
                lm_hp=lm_hp,
                loss_target_eff=loss_target_eff,
                accept_threshold_eff_cand=accept_threshold_eff_cand,
                best_val_loss=best_val_loss,
                current_val_loss=current_val_loss,
                stageA_under_protest=bool(stageA_under_protest),
                best_train_loss=best_train_loss,
                loss_scale=loss_scale,
                model_sep_output=model_sep_output,
                y_op=y_op,
                y_op_inv=y_op_inv,
                Nxvars=Nxvars,
                x_transform_map=x_transform_map,
                units_spec=units_spec,
                enforce_units=bool(enforce_units),
                units_reject_cb=units_reject_cb,
                x_train=x_train,
            )
            if acc_fb:
                return True, model_fb, ast_fb, loss_fb, ""

        reason = "forced monomial regressed and P*NN[pi] fallback did not accept"
        print(f"[Stage A ForcedMonomial] Rejected: {reason}")
        return False, None, None, None, reason

    reason = "forced monomial did not produce acceptable fit"
    print(f"[Stage A ForcedMonomial] Rejected: {reason}")
    return False, None, None, None, reason


def _build_stageA_composite_closure_ast(
    current_ast: Node,
    atom: AtomNode,
    z_expr: Node,
    *,
    x_train=None,
    y_teacher=None,
    units_spec=None,
    enforce_units: bool = False,
) -> Tuple[Optional[Node], Optional[str]]:
    """Build ``current_ast`` with ``atom`` visibly replaced by ``scale * z_expr``."""
    try:
        scale_init = _stageA_composite_scale_init(z_expr, x_train, y_teacher)
        scale_atom = _stageA_composite_scalar_atom(
            current_ast=current_ast,
            atom=atom,
            z_expr=z_expr,
            units_spec=units_spec,
            enforce_units=bool(enforce_units),
            init=scale_init,
        )
        replacement = MulNode(scale_atom, clone_ast(z_expr))
        cand_ast = replace_atom_in_ast(current_ast, atom, replacement)
        if cand_ast is None:
            return None, "replace failed"
        units_reason = _analytic_units_rejection(
            cand_ast,
            units_spec,
            enforce_units=bool(enforce_units),
        )
        if units_reason is not None:
            return None, str(units_reason)
        return cand_ast, None
    except Exception as exc:
        return None, str(exc)


def _stageA_composite_reduces_nn_burden(base_ast: Node, cand_ast: Node) -> bool:
    """Visible analytic closure must reduce the unresolved NN burden."""
    try:
        base_atoms = collect_nn_atoms(base_ast)
        cand_atoms = collect_nn_atoms(cand_ast)
        if len(cand_atoms) < len(base_atoms):
            return True
        base_sig = sorted(int(effective_arity(a)) for a in base_atoms)
        cand_sig = sorted(int(effective_arity(a)) for a in cand_atoms)
        return tuple(cand_sig) < tuple(base_sig)
    except Exception:
        return False


def _stageA_nn_burden_signature(ast: Node) -> tuple[int, tuple[int, ...]]:
    """Compact unresolved-NN burden used for shadow-promotion audits."""
    try:
        atoms = collect_nn_atoms(ast)
        return (len(atoms), tuple(sorted(int(effective_arity(a)) for a in atoms)))
    except Exception:
        return (10**9, (10**9,))


def _stageA_shadow_promotion_payoff_reason(
    *,
    base_ast: Node,
    cand_ast: Node,
    old_arity: int,
    new_arity: int,
    enables_sep: bool,
    meta: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Candidate-aware guard for committed shadow-coordinate promotions."""
    meta = dict(meta or {})
    if bool(meta.get("hidden_shadow_only", False)):
        return False, "hidden shadow-only state is not an accepted Stage-A model"
    if not bool(meta.get("shadow_visible_ast", False)):
        return False, "shadow promotion did not produce a visible AST coordinate"

    base_sig = _stageA_nn_burden_signature(base_ast)
    cand_sig = _stageA_nn_burden_signature(cand_ast)
    if cand_sig < base_sig:
        return True, f"visible NN burden {base_sig} → {cand_sig}"

    if _compound_candidate_payoff_policy(int(old_arity), int(new_arity)) == "require_sep" and bool(enables_sep):
        return True, "same-arity shadow promotion has confirmed separability payoff"

    return False, f"no visible NN-burden payoff ({base_sig} → {cand_sig})"


def _stageA_shadow_promotion_audit(
    *,
    base_ast: Node,
    cand_ast: Node,
    old_arity: int,
    new_arity: int,
    enables_sep: bool,
    meta: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Run the shadow-promotion audit when candidate metadata requires it."""
    meta = dict(meta or {})
    kind_l = str(meta.get("kind", "")).lower()
    if not (
        bool(meta.get("shadow_requires_payoff", False))
        or kind_l in {"shadow_composite", "shadow_preserved_factor", "shadow_trig_factor_peel"}
    ):
        return True, "not a shadow promotion"
    return _stageA_shadow_promotion_payoff_reason(
        base_ast=base_ast,
        cand_ast=cand_ast,
        old_arity=int(old_arity),
        new_arity=int(new_arity),
        enables_sep=bool(enables_sep),
        meta=meta,
    )


def _stageA_cap_terminal_analytic_threshold(
    *,
    base_ast: Node,
    cand_ast: Node,
    accept_threshold: float,
    absolute_cap: float,
) -> tuple[float, bool]:
    """Keep terminal analytic closures on the existing absolute quality budget.

    Stage-A structural moves may get a generous temporary loss budget because an
    NN[z, extras] rewrite can unlock later separability.  A terminal analytic
    closure has no remaining NN to refine, so accepting a poor fit just because
    the baseline NN was also poor destroys the search state.  The screen is
    still only proposal evidence; the accepted no-NN AST must clear the normal
    absolute candidate cap.
    """
    try:
        if len(collect_nn_atoms(base_ast)) <= 0 or len(collect_nn_atoms(cand_ast)) != 0:
            return float(accept_threshold), False
    except Exception:
        return float(accept_threshold), False

    try:
        cap = float(absolute_cap)
    except Exception:
        cap = float("nan")
    if not (math.isfinite(cap) and cap > 0.0):
        return float(accept_threshold), True
    return float(min(float(accept_threshold), cap)), True


def _loader_n_eff(dl) -> Optional[float]:
    try:
        ds = getattr(dl, "dataset", None)
        if ds is not None:
            n = int(len(ds))
            if n > 0:
                return float(n)
    except Exception:
        pass
    return None


def _stageA_noisy_terminal_yspace_accept(
    *,
    base_ast: Node,
    cand_ast: Node,
    base_y_mse: float,
    cand_y_mse: float,
    noise_floor_raw: float,
    n_eff: Optional[float],
) -> tuple[bool, str]:
    """Accept a visible terminal closure when it is statistically noise-limited.

    This is intentionally disabled for noiseless data.  It is used only after
    the normal fit-space threshold has rejected a terminal analytic candidate,
    so it cannot weaken exact/noiseless model selection.
    """

    try:
        nf = float(noise_floor_raw)
        base = float(base_y_mse)
        cand = float(cand_y_mse)
    except Exception:
        return False, "non-finite noisy terminal metrics"
    if (not math.isfinite(nf)) or nf <= 0.0:
        return False, "no explicit positive y-space noise floor"
    if not math.isfinite(cand):
        return False, "candidate y-space MSE is non-finite"
    try:
        if len(collect_nn_atoms(base_ast)) <= 0 or len(collect_nn_atoms(cand_ast)) != 0:
            return False, "not an NN-removing terminal analytic closure"
    except Exception:
        return False, "could not confirm terminal analytic closure"

    noise_mult = 3.0
    cand_at_floor = bool(
        cand <= nf
        or _noise_equivalent(
            cand,
            nf,
            noise_floor=nf,
            n_eff=n_eff,
            noise_mult=noise_mult,
        )
    )
    if not cand_at_floor:
        return (
            False,
            f"candidate y-MSE={cand:.3e} is not noise-limited "
            f"(floor={nf:.3e}, n_eff={n_eff if n_eff is not None else 'none'})",
        )

    base_unknown = not math.isfinite(base)
    tied_or_better_than_base = bool(
        base_unknown
        or cand <= base
        or _noise_equivalent(
            cand,
            base,
            noise_floor=nf,
            n_eff=n_eff,
            noise_mult=noise_mult,
        )
    )
    if not tied_or_better_than_base:
        return (
            False,
            f"candidate y-MSE={cand:.3e} materially regresses current y-MSE={base:.3e} "
            f"(floor={nf:.3e}, n_eff={n_eff if n_eff is not None else 'none'})",
        )

    return (
        True,
        f"candidate y-MSE={cand:.3e}, current y-MSE={base:.3e}, "
        f"floor={nf:.3e}, n_eff={n_eff if n_eff is not None else 'none'}",
    )


def _stageA_compound_buckingham_reason(
    *,
    current_ast: Node,
    atom: AtomNode,
    z_expr: Node,
    kind: str,
    extra_var_idxs: Optional[List[int]],
    extra_input_asts: Optional[List[Node]],
    units_spec,
    enforce_units: bool,
    candidate_meta=None,
) -> Optional[str]:
    """Return the Buckingham rejection reason for an ``NN[z, extras]`` form."""
    if (not bool(enforce_units)) or units_spec is None:
        return None
    if mark_stagea_buckingham_deferred(candidate_meta):
        return None
    try:
        from nestynet_sr.sr_core.units import check_compound_buckingham, eval_analytic_expr_dim

        z_dim_computed = eval_analytic_expr_dim(z_expr, units_spec.x_dims)
        preserved_dims = None
        if extra_input_asts:
            plist = []
            for expr in extra_input_asts:
                try:
                    ed = eval_analytic_expr_dim(expr, units_spec.x_dims)
                    if ed is not None:
                        plist.append(ed)
                except Exception:
                    pass
            if plist:
                preserved_dims = plist
        ok, reason = check_compound_buckingham(
            atom_var_idxs=[int(v) for v in getattr(atom, "var_idxs", ())],
            extra_var_idxs=list(extra_var_idxs or ()),
            z_dim=z_dim_computed,
            x_dims=units_spec.x_dims,
            min_freedom=_compound_buckingham_min_freedom(kind),
            y_dim=_stageA_compound_buckingham_target_dim(current_ast, atom, units_spec),
            extra_preserved_dims=preserved_dims,
        )
        return None if ok else str(reason)
    except Exception:
        return None


def _stageA_normalize_nonzero_prefactor_exponents(
    atom: AtomNode,
    prefactor_exponents,
) -> Optional[Tuple[int, ...]]:
    """Return integer prefactor powers when they visibly peel a nontrivial factor."""
    if prefactor_exponents is None:
        return None
    try:
        pref = tuple(int(v) for v in prefactor_exponents)
    except Exception:
        return None
    if not any(int(v) != 0 for v in pref):
        return None
    try:
        if has_nontrivial_input(atom):
            local_arity = len(tuple(get_input_exprs(atom)))
        else:
            local_arity = len(tuple(getattr(atom, "var_idxs", ()) or ()))
    except Exception:
        local_arity = len(tuple(getattr(atom, "var_idxs", ()) or ()))
    if int(local_arity) != len(pref):
        return None
    return pref


def _stageA_prefactor_peeled_raw_vars(
    atom: AtomNode,
    prefactor_exponents,
) -> set[int]:
    """Return raw input variables visibly removed by a monomial prefactor."""
    pref = _stageA_normalize_nonzero_prefactor_exponents(atom, prefactor_exponents)
    if pref is None:
        return set()
    try:
        local_inputs = tuple(get_input_exprs(atom)) if has_nontrivial_input(atom) else tuple(
            Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ()
        )
    except Exception:
        local_inputs = tuple(Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ())
    peeled: set[int] = set()
    for i, pe in enumerate(pref):
        if int(pe) == 0 or int(i) >= len(local_inputs):
            continue
        inp = local_inputs[int(i)]
        if is_trivial_input(inp):
            try:
                peeled.add(int(inp.var_idxs[0]))
            except Exception:
                pass
    return peeled


def _stageA_visible_prefactor_buckingham_transaction_reason(
    *,
    current_ast: Node,
    atom: AtomNode,
    z_expr: Node,
    pattern,
    extra_var_idxs: Optional[List[int]],
    extra_input_asts: Optional[List[Node]],
    prefactor_exponents,
    units_spec,
    enforce_units: bool,
) -> Optional[str]:
    """Return why a visible ``P*NN[z,...]`` Buckingham transaction is invalid.

    The bare Buckingham gate should continue to reject illegal ``NN[z]`` forms.
    This helper is only the Stage-A transaction rescue for already-detected
    monomial prefactors: the prefactor must appear in the candidate AST, carry
    the local atom output dimension, and leave a dimensionless residual child.
    Raw extras consumed by the visible prefactor are not residual NN inputs.
    """
    if (not bool(enforce_units)) or units_spec is None:
        return None

    pref = _stageA_normalize_nonzero_prefactor_exponents(atom, prefactor_exponents)
    if pref is None:
        return "no nonzero visible prefactor transaction"

    try:
        from nestynet_sr.sr_core.units import check_units_ast, eval_analytic_expr_dim, is_dimless

        target_dim = _stageA_compound_buckingham_target_dim(current_ast, atom, units_spec)
        if target_dim is None:
            return "could not infer local atom output dimension"

        local_inputs = tuple(get_input_exprs(atom)) if has_nontrivial_input(atom) else tuple(
            Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ()
        )
        if len(local_inputs) != len(pref):
            return "prefactor exponent arity does not match atom inputs"

        input_dims = []
        for inp in local_inputs:
            d = eval_analytic_expr_dim(inp, units_spec.x_dims)
            if d is None:
                return "unknown prefactor input dimension"
            input_dims.append(d)
        pref_dim = _stageA_dim_power_sum(input_dims, pref)
        if pref_dim is None:
            return "could not infer visible prefactor dimension"
        if tuple(pref_dim) != tuple(target_dim):
            fmt = getattr(getattr(units_spec, "unit_system", None), "format_dim", None)
            got = fmt(pref_dim) if callable(fmt) else str(pref_dim)
            need = fmt(target_dim) if callable(fmt) else str(target_dim)
            return f"visible prefactor dimension {got} does not match local target {need}"

        z_dim = eval_analytic_expr_dim(z_expr, units_spec.x_dims)
        if z_dim is None:
            return "unknown residual compound dimension"
        if not is_dimless(z_dim):
            return "visible-prefactor transaction currently requires dimensionless z"

        peeled_raw_vars = _stageA_prefactor_peeled_raw_vars(atom, pref)
        for idx in list(extra_var_idxs or ()):
            if int(idx) in peeled_raw_vars:
                continue
            try:
                d = units_spec.x_dims[int(idx)]
            except Exception:
                return "unknown residual raw-extra dimension"
            if not is_dimless(d):
                return "visible-prefactor transaction currently requires dimensionless residual extras"

        for expr in list(extra_input_asts or ()):
            d = eval_analytic_expr_dim(expr, units_spec.x_dims)
            if d is None:
                return "unknown residual compound-extra dimension"
            if not is_dimless(d):
                return "visible-prefactor transaction currently requires dimensionless residual extras"

        cand_ast_probe = _build_compound_candidate_ast(
            current_ast,
            atom,
            z_expr,
            pattern,
            extra_var_idxs_override=extra_var_idxs,
            prefactor_exponents=pref,
            extra_input_asts=extra_input_asts,
        )
        ures = check_units_ast(cand_ast_probe, units_spec)
        if not bool(getattr(ures, "ok", False)):
            reason = getattr(ures, "reason", "unit check failed")
            return f"visible prefactor transaction unit check failed: {reason}"
        return None
    except Exception as exc:
        return f"visible prefactor transaction failed: {exc}"


def _stageA_generate_unit_prefactor_exponents(
    *,
    current_ast: Node,
    atom: AtomNode,
    z_expr: Node,
    pattern,
    extra_var_idxs: Optional[List[int]],
    extra_input_asts: Optional[List[Node]],
    units_spec,
    enforce_units: bool,
    max_abs_power: int = 3,
    max_support: int = 4,
) -> Tuple[Optional[Tuple[int, ...]], str]:
    """Generate one sparse integer complement ``P`` for ``P*NN[z,...]``.

    This is deliberately narrower than the Buckingham theorem: it only handles
    the clean Stage-A rescue where the residual coordinate set is dimensionless
    after visibly peeled raw extras are removed from the residual NN, and a
    visible integer monomial over the current atom inputs can carry the local
    atom output dimension.
    """
    if (not bool(enforce_units)) or units_spec is None:
        return None, "units disabled"
    try:
        from nestynet_sr.sr_core.units import eval_analytic_expr_dim, is_dimless

        target_dim = _stageA_compound_buckingham_target_dim(current_ast, atom, units_spec)
        if target_dim is None:
            return None, "could not infer local atom output dimension"
        if is_dimless(target_dim):
            return None, "local target dimensionless; generated complement disabled"

        z_dim = eval_analytic_expr_dim(z_expr, units_spec.x_dims)
        if z_dim is None:
            return None, "unknown residual compound dimension"
        if not is_dimless(z_dim):
            return None, "generated complement requires dimensionless z"

        for expr in list(extra_input_asts or ()):
            d = eval_analytic_expr_dim(expr, units_spec.x_dims)
            if d is None:
                return None, "unknown residual compound-extra dimension"
            if not is_dimless(d):
                return None, "generated complement requires dimensionless residual extras"

        local_inputs = tuple(get_input_exprs(atom)) if has_nontrivial_input(atom) else tuple(
            Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ()
        )
        if not local_inputs:
            return None, "atom has no local inputs"
        if len(local_inputs) > 8:
            return None, "too many atom inputs for sparse generated complement"

        input_dims = []
        for inp in local_inputs:
            d = eval_analytic_expr_dim(inp, units_spec.x_dims)
            if d is None:
                return None, "unknown prefactor input dimension"
            input_dims.append(d)

        try:
            pat = tuple(int(v) for v in pattern)
            if len(pat) != len(local_inputs):
                pat = None
        except Exception:
            pat = None

        candidates: List[Tuple[Any, Tuple[int, ...]]] = []
        n_inputs = len(local_inputs)
        nonzero_powers = tuple(
            p for p in range(-int(max_abs_power), int(max_abs_power) + 1)
            if int(p) != 0
        )
        for support in range(1, min(int(max_support), int(n_inputs)) + 1):
            for idxs in itertools.combinations(range(int(n_inputs)), support):
                for vals in itertools.product(nonzero_powers, repeat=int(support)):
                    powers_list = [0 for _ in range(int(n_inputs))]
                    for idx, val in zip(idxs, vals):
                        powers_list[int(idx)] = int(val)
                    powers = tuple(powers_list)
                    dim_sum = _stageA_dim_power_sum(input_dims, powers)
                    if dim_sum is None or tuple(dim_sum) != tuple(target_dim):
                        continue

                    peeled_raw_vars = set()
                    for i, pe in enumerate(powers):
                        if int(pe) == 0 or int(i) >= len(local_inputs):
                            continue
                        inp = local_inputs[int(i)]
                        if is_trivial_input(inp):
                            try:
                                peeled_raw_vars.add(int(inp.var_idxs[0]))
                            except Exception:
                                pass
                    residual_extra_ok = True
                    for idx in list(extra_var_idxs or ()):
                        if int(idx) in peeled_raw_vars:
                            continue
                        try:
                            d = units_spec.x_dims[int(idx)]
                        except Exception:
                            return None, "unknown residual raw-extra dimension"
                        if not is_dimless(d):
                            residual_extra_ok = False
                            break
                    if not residual_extra_ok:
                        continue

                    l1 = sum(abs(int(v)) for v in powers)
                    max_abs = max(abs(int(v)) for v in powers)
                    n_negative = sum(1 for v in powers if int(v) < 0)
                    if pat is not None:
                        # z is dimensionless, so multiplying P by z^k is a gauge.
                        # Prefer the representative closest to perpendicular to z's
                        # log-exponent direction; this avoids arbitrary powers of z
                        # inside the prefactor when a balanced complement exists.
                        gauge_dot = abs(sum(int(a) * int(b) for a, b in zip(pat, powers)))
                    else:
                        gauge_dot = 0
                    key = (gauge_dot, l1, support, max_abs, n_negative, tuple(int(v) for v in powers))
                    candidates.append((key, tuple(int(v) for v in powers)))

        if not candidates:
            return None, "no sparse integer complement matches local target dimension"

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1], ""
    except Exception as exc:
        return None, f"generated complement failed: {exc}"


def _stageA_local_monomial_ast_from_inputs(inputs, powers) -> Optional[Node]:
    """Build ``prod input_i**power_i`` over an atom's current effective inputs."""
    try:
        terms = []
        for inp, power in zip(tuple(inputs or ()), tuple(powers or ())):
            p = int(power)
            if p == 0:
                continue
            base = clone_ast(inp)
            terms.append(base if p == 1 else PowNode(base, int(p)))
        if not terms:
            return None
        out = terms[0]
        for term in terms[1:]:
            out = MulNode(out, term)
        return out
    except Exception:
        return None


def _stageA_sparse_integer_power_vectors(
    n_inputs: int,
    *,
    max_abs_power: int,
    max_support: int,
):
    nonzero = tuple(
        p for p in range(-int(max_abs_power), int(max_abs_power) + 1)
        if int(p) != 0
    )
    for support in range(1, min(int(max_support), int(n_inputs)) + 1):
        for idxs in itertools.combinations(range(int(n_inputs)), int(support)):
            for vals in itertools.product(nonzero, repeat=int(support)):
                powers = [0 for _ in range(int(n_inputs))]
                for idx, val in zip(idxs, vals):
                    powers[int(idx)] = int(val)
                yield tuple(int(v) for v in powers)


def _stageA_canonical_dimensionless_power_vector(powers) -> Optional[Tuple[int, ...]]:
    try:
        vals = [int(v) for v in tuple(powers or ())]
    except Exception:
        return None
    if not vals or not any(vals):
        return None
    g = 0
    for v in vals:
        if v:
            g = math.gcd(g, abs(int(v)))
    if g <= 0:
        return None
    vals = [int(v // g) for v in vals]
    first = next((v for v in vals if int(v) != 0), 0)
    if first < 0:
        vals = [-int(v) for v in vals]
    return tuple(int(v) for v in vals)


def _stageA_power_vector_complexity_key(powers) -> Tuple[int, int, int, int, Tuple[int, ...]]:
    vals = tuple(int(v) for v in tuple(powers or ()))
    support = sum(1 for v in vals if int(v) != 0)
    l1 = sum(abs(int(v)) for v in vals)
    max_abs = max((abs(int(v)) for v in vals), default=0)
    n_negative = sum(1 for v in vals if int(v) < 0)
    return (int(support), int(l1), int(max_abs), int(n_negative), vals)


def _stageA_prefactor_pi_gauge_info(
    prefactor_powers,
    pi_powers,
) -> Tuple[int, int, Tuple[int, ...], Tuple[int, int, int, int, Tuple[int, ...]]]:
    """Return the simplest representative of ``P * pi^k`` prefactor gauges.

    If ``pi=x^q`` is dimensionless, then ``P=x^p`` and ``P*pi^k`` carry the
    same units.  For Stage-A ballot construction we prefer the canonical gauge
    with the smallest visible prefactor, leaving dimensionless powers inside the
    NN response.  The returned tuple is ``(abs(k), k, canonical_p, key)`` where
    ``canonical_p = p - k*q`` has minimal sparse-integer complexity.
    """

    p = tuple(int(v) for v in tuple(prefactor_powers or ()))
    q = tuple(int(v) for v in tuple(pi_powers or ()))
    if len(p) != len(q) or not p or not any(q):
        key = _stageA_power_vector_complexity_key(p)
        return 0, 0, p, key
    max_power = max([abs(int(v)) for v in p + q] or [0])
    shift_limit = max(6, int(max_power) + 6)
    best = None
    for k in range(-shift_limit, shift_limit + 1):
        reduced = tuple(int(pi - k * qi) for pi, qi in zip(p, q))
        key = _stageA_power_vector_complexity_key(reduced)
        ranked = (key, abs(int(k)), int(k))
        if best is None or ranked < best[0]:
            best = (ranked, int(k), reduced, key)
    if best is None:
        key = _stageA_power_vector_complexity_key(p)
        return 0, 0, p, key
    _ranked, k_best, reduced_best, key_best = best
    return abs(int(k_best)), int(k_best), tuple(int(v) for v in reduced_best), key_best


def _stageA_visible_buckingham_1d_prefactor_proposals_for_atom(
    *,
    current_ast: Node,
    atom: AtomNode,
    units_spec,
    enforce_units: bool,
    search_hp=None,
    x_transform_map=None,
) -> List[tuple]:
    """Return unit-certified visible ``P * NN[pi]`` proposals for one atom.

    This is deliberately a proposal lane, not a Planck-specific rescue.  It asks
    whether the current local NN target admits a sparse monomial prefactor
    carrying the atom's local output dimension and a single dimensionless
    monomial group.  Normal Stage-A screening, LM validation, and CoE ranking
    still decide whether the candidate is accepted.
    """
    # LOAD-BEARING, do not remove: the provisional-acceptance rollback in
    # run_SR.py (_maybe_retry_without_visible_buckingham) re-executes a run
    # with this variable set when the lane's accepted peel led to an
    # uncertified/unresolved final expression. Without this check the
    # rollback is a no-op.
    if os.environ.get("NNSR_SUPPRESS_VISIBLE_BUCKINGHAM") == "1":
        return []
    if (not bool(enforce_units)) or units_spec is None:
        return []
    try:
        if _stageA_x_transform_fingerprint(x_transform_map) != "none":
            return []
    except Exception:
        return []
    try:
        old_arity = int(effective_arity(atom))
    except Exception:
        old_arity = 0
    if old_arity <= 1:
        return []

    try:
        from nestynet_sr.sr_core.units import check_units_ast, eval_analytic_expr_dim, is_dimless

        target_dim = _stageA_compound_buckingham_target_dim(current_ast, atom, units_spec)
        if target_dim is None or is_dimless(target_dim):
            return []

        local_inputs = tuple(get_input_exprs(atom)) if has_nontrivial_input(atom) else tuple(
            Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ()
        )
        if not local_inputs:
            return []
        max_inputs = int(getattr(search_hp, "visible_buckingham_1d_max_inputs", 6))
        if len(local_inputs) > max_inputs:
            return []

        input_dims = []
        for inp in local_inputs:
            d = eval_analytic_expr_dim(inp, units_spec.x_dims)
            if d is None:
                return []
            input_dims.append(d)

        max_abs_q = int(getattr(search_hp, "visible_buckingham_1d_max_abs_pi_power", 3))
        max_abs_p = int(getattr(search_hp, "visible_buckingham_1d_max_abs_prefactor_power", 3))
        max_support_q = int(getattr(search_hp, "visible_buckingham_1d_max_pi_support", 4))
        max_support_p = int(getattr(search_hp, "visible_buckingham_1d_max_prefactor_support", 4))
        max_groups = int(getattr(search_hp, "visible_buckingham_1d_max_groups", 8))
        max_prefactors = int(getattr(search_hp, "visible_buckingham_1d_max_prefactors", 8))
        max_candidates = int(getattr(search_hp, "visible_buckingham_1d_max_candidates", 4))
        if max_candidates <= 0:
            return []

        q_seen: set[Tuple[int, ...]] = set()
        q_candidates: List[Tuple[Tuple[int, int, int, Tuple[int, ...]], Tuple[int, ...], Node]] = []
        for powers_raw in _stageA_sparse_integer_power_vectors(
            len(local_inputs),
            max_abs_power=max_abs_q,
            max_support=max_support_q,
        ):
            dim_sum = _stageA_dim_power_sum(input_dims, powers_raw)
            if dim_sum is None or not is_dimless(dim_sum):
                continue
            powers = _stageA_canonical_dimensionless_power_vector(powers_raw)
            if powers is None or powers in q_seen:
                continue
            q_seen.add(powers)
            pi_ast = _stageA_local_monomial_ast_from_inputs(local_inputs, powers)
            if pi_ast is None:
                continue
            pi_dim = eval_analytic_expr_dim(pi_ast, units_spec.x_dims)
            if pi_dim is None or not is_dimless(pi_dim):
                continue
            support = sum(1 for v in powers if int(v) != 0)
            l1 = sum(abs(int(v)) for v in powers)
            max_abs = max(abs(int(v)) for v in powers)
            q_candidates.append(((support, l1, max_abs, tuple(int(v) for v in powers)), powers, pi_ast))

        p_candidates: List[Tuple[Tuple[int, int, int, int, Tuple[int, ...]], Tuple[int, ...], Node]] = []
        for powers in _stageA_sparse_integer_power_vectors(
            len(local_inputs),
            max_abs_power=max_abs_p,
            max_support=max_support_p,
        ):
            dim_sum = _stageA_dim_power_sum(input_dims, powers)
            if dim_sum is None or tuple(dim_sum) != tuple(target_dim):
                continue
            pref_ast = _stageA_local_monomial_ast_from_inputs(local_inputs, powers)
            if pref_ast is None:
                continue
            pref_dim = eval_analytic_expr_dim(pref_ast, units_spec.x_dims)
            if pref_dim is None or tuple(pref_dim) != tuple(target_dim):
                continue
            support = sum(1 for v in powers if int(v) != 0)
            l1 = sum(abs(int(v)) for v in powers)
            max_abs = max(abs(int(v)) for v in powers)
            n_negative = sum(1 for v in powers if int(v) < 0)
            p_candidates.append(((support, l1, max_abs, n_negative, tuple(int(v) for v in powers)), powers, pref_ast))

        if not q_candidates or not p_candidates:
            return []
        q_candidates.sort(key=lambda row: row[0])
        p_candidates.sort(key=lambda row: row[0])

        parent_ctx = _stageA_parent_context_descriptor(
            current_ast,
            atom,
            units_spec=units_spec,
            x_transform_map=x_transform_map,
        )
        proposal_entries: dict[Tuple[Tuple[int, ...], Tuple[int, ...]], tuple] = {}
        for _q_key, q_powers, pi_ast in q_candidates[:max(1, max_groups)]:
            for _p_key, p_powers, pref_ast in p_candidates[:max(1, max_prefactors)]:
                try:
                    gauge_abs, gauge_shift, canonical_p, canonical_key = _stageA_prefactor_pi_gauge_info(
                        p_powers,
                        q_powers,
                    )
                    gauge_class_key = (
                        tuple(int(v) for v in q_powers),
                        tuple(int(v) for v in canonical_p),
                    )
                    entry_rank = (
                        int(gauge_abs),
                        _stageA_power_vector_complexity_key(p_powers),
                        _q_key,
                    )
                    existing = proposal_entries.get(gauge_class_key)
                    if existing is not None and entry_rank >= existing[0]:
                        continue
                    cand_ast = _build_compound_candidate_ast(
                        current_ast,
                        atom,
                        pi_ast,
                        q_powers,
                        extra_var_idxs_override=[],
                        prefactor_exponents=p_powers,
                        prefactor_ast=pref_ast,
                        extra_input_asts=None,
                    )
                    ures = check_units_ast(cand_ast, units_spec)
                    if not bool(getattr(ures, "ok", False)):
                        continue
                except Exception:
                    continue
                try:
                    pi_readable = ast_to_human_readable(pi_ast, x_transform_map)
                except Exception:
                    pi_readable = "pi"
                try:
                    pref_readable = ast_to_human_readable(pref_ast, x_transform_map)
                except Exception:
                    pref_readable = "P"
                meta = {
                    "kind": "monomial",
                    "family": "visible_buckingham_1d_prefactor",
                    "source": "unit_buckingham_1d_prefactor",
                    "visible_buckingham_1d_prefactor": True,
                    "buckingham_1d": True,
                    "structural_protected": True,
                    "unit_verified": True,
                    "visible_prefactor_transaction": True,
                    "prefactor_ast": clone_ast(pref_ast),
                    "prefactor_exponents": tuple(int(v) for v in p_powers),
                    "prefactor_pi_gauge_abs": int(gauge_abs),
                    "prefactor_pi_gauge_shift": int(gauge_shift),
                    "prefactor_pi_gauge_canonical_exponents": tuple(int(v) for v in canonical_p),
                    "prefactor_pi_gauge_canonical_key": tuple(canonical_key),
                    "pi_ast": clone_ast(pi_ast),
                    "pi_exponents": tuple(int(v) for v in q_powers),
                    "old_arity": int(old_arity),
                    "new_arity": 1,
                    "arity_drop": int(old_arity) - 1,
                    "collapse_score": None,
                    "hole_context_fp": parent_ctx.get("parent_hole_context_fp"),
                    "prefactor_readable": pref_readable,
                    "pi_readable": pi_readable,
                }
                proposal = (
                    tuple(int(v) for v in q_powers),
                    clone_ast(pi_ast),
                    float(getattr(search_hp, "visible_buckingham_1d_confidence", 0.995)),
                    [],
                    meta,
                )
                proposal_entries[gauge_class_key] = (entry_rank, proposal)
        proposals = [row[1] for row in sorted(proposal_entries.values(), key=lambda row: row[0])]
        if len(proposals) > max_candidates:
            proposals = proposals[:max_candidates]
        if proposals:
            # Proposal-level marker: acceptance can happen on several Stage-A
            # paths (CoE committee, plain screening), so the rollback trigger
            # in run_SR.py falls back to "the lane proposed at all" when no
            # committee accept-record exists.
            try:
                from nestynet_sr.sr_search.gate_telemetry import record_gate

                record_gate(
                    "visible_buckingham_lane",
                    "proposed",
                    float(len(proposals)),
                    float("nan"),
                    accepted=True,
                    context={"atom_tag": str(getattr(atom, "tag", None))},
                )
            except Exception:
                pass
        return proposals
    except Exception as exc:
        if bool(getattr(search_hp, "verbose_compound", False)):
            print(f"[Units/Buckingham] Visible 1D prefactor proposal scan failed: {exc}")
        return []


def _stageA_append_visible_buckingham_1d_prefactor_proposals(
    proposals,
    *,
    current_ast: Node,
    atom: AtomNode,
    units_spec,
    enforce_units: bool,
    search_hp=None,
    x_transform_map=None,
) -> list:
    """Append a tiny protected ``P*NN[pi]`` proposal lane, deduping candidates."""
    out = list(proposals or [])
    extra = _stageA_visible_buckingham_1d_prefactor_proposals_for_atom(
        current_ast=current_ast,
        atom=atom,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
        search_hp=search_hp,
        x_transform_map=x_transform_map,
    )
    if not extra:
        return out

    seen = set()
    for prop in out:
        try:
            meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
            pref_ast = meta.get("prefactor_ast")
            pref_key = _stageA_ast_fingerprint(pref_ast) if pref_ast is not None else None
            seen.add((
                _stageA_ast_fingerprint(prop[1]),
                tuple(int(v) for v in (prop[3] or ())),
                pref_key,
            ))
        except Exception:
            continue

    added = 0
    for prop in extra:
        try:
            meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
            pref_ast = meta.get("prefactor_ast")
            key = (
                _stageA_ast_fingerprint(prop[1]),
                tuple(int(v) for v in (prop[3] or ())),
                _stageA_ast_fingerprint(pref_ast) if pref_ast is not None else None,
            )
        except Exception:
            key = None
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        out.append(prop)
        added += 1
        try:
            meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
            print(
                "[Units/Buckingham] Adding visible 1D prefactor proposal: "
                f"P={meta.get('prefactor_readable', 'P')}, "
                f"pi={meta.get('pi_readable', 'pi')}, "
                f"gauge={meta.get('prefactor_pi_gauge_abs', 0)}"
            )
        except Exception:
            pass
    if added:
        print(
            f"[Units/Buckingham] Added {added} unit-certified visible "
            "Buckingham 1D prefactor proposal(s)."
        )
    return out


def _stageA_direct_additive_nn_terms(root: Node) -> list[tuple[int, AtomNode, Node]]:
    """Return direct top-level additive terms that are bare NN atoms."""
    terms = _flatten_additive_terms(root)
    out: list[tuple[int, AtomNode, Node]] = []
    for idx, term in enumerate(terms):
        if isinstance(term, AtomNode) and str(getattr(term, "kind", "")).lower() == "nn":
            out.append((int(idx), term, term))
    return out


def _stageA_replace_direct_additive_subset(
    root: Node,
    selected_indices: Iterable[int],
    replacement: Node,
) -> Node:
    selected = {int(i) for i in selected_indices}
    terms = _flatten_additive_terms(root)
    if not selected:
        return clone_ast(root)
    new_terms: list[Node] = []
    inserted = False
    for idx, term in enumerate(terms):
        if int(idx) in selected:
            if not inserted:
                new_terms.append(clone_ast(replacement))
                inserted = True
            continue
        new_terms.append(clone_ast(term))
    return _rebuild_additive_chain(new_terms)


def _stageA_eval_additive_block_target(
    *,
    current_ast: Node,
    selected_indices: Iterable[int],
    tag_to_leaf: dict,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Evaluate y minus additive terms outside the selected direct block."""
    terms = _flatten_additive_terms(current_ast)
    selected = {int(i) for i in selected_indices}
    target = y.reshape(-1).clone()
    if len(selected) >= len(terms):
        return target
    with torch.no_grad():
        outside = torch.zeros_like(target)
        for idx, term in enumerate(terms):
            if int(idx) in selected:
                continue
            outside = outside + _eval_ast_subtree_on_data(term, tag_to_leaf, x)
    return target - outside


def _stageA_collect_loader_xy(
    loader,
    *,
    device,
    dtype,
    max_points: Optional[int] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    xs = []
    ys = []
    n = 0
    dl = loader() if callable(loader) else loader
    try:
        for batch in dl:
            if isinstance(batch, (list, tuple)):
                xb, yb = batch[0], batch[1]
            else:
                continue
            if max_points is not None:
                remaining = int(max_points) - n
                if remaining <= 0:
                    break
                if int(xb.shape[0]) > remaining:
                    xb = xb[:remaining]
                    yb = yb[:remaining]
            xs.append(xb.to(device=device, dtype=dtype))
            ys.append(yb.to(device=device, dtype=dtype))
            n += int(xb.shape[0])
    except Exception:
        return None, None
    if not xs or not ys:
        return None, None
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _stageA_prefactor_gauge_variants_for_asr(
    *,
    atom: AtomNode,
    proposal: tuple,
    search_hp,
) -> list[dict]:
    """Expand a local P,pi Buckingham seed by P*pi^k gauge shifts."""
    try:
        _pattern, _pi_ast, conf, _extra, meta = proposal
    except Exception:
        return []
    if not isinstance(meta, dict):
        return []
    try:
        p0 = tuple(int(v) for v in meta.get("prefactor_exponents"))
        q = tuple(int(v) for v in meta.get("pi_exponents"))
    except Exception:
        return []
    if len(p0) != len(q):
        return []
    local_inputs = tuple(get_input_exprs(atom)) if has_nontrivial_input(atom) else tuple(
        Var(int(v)) for v in getattr(atom, "var_idxs", ()) or ()
    )
    if len(local_inputs) != len(q):
        return []
    max_shift = max(0, int(getattr(search_hp, "additive_shared_response_max_gauge_shift", 2) or 2))
    max_abs = max(1, int(getattr(search_hp, "additive_shared_response_max_abs_power", 3) or 3))
    max_support = max(1, int(getattr(search_hp, "additive_shared_response_max_prefactor_support", 4) or 4))
    variants = []
    seen = set()
    for k in range(-max_shift, max_shift + 1):
        powers = tuple(int(p0_i) + int(k) * int(q_i) for p0_i, q_i in zip(p0, q))
        if max((abs(int(v)) for v in powers), default=0) > max_abs:
            continue
        support = sum(1 for v in powers if int(v) != 0)
        if support <= 0 or support > max_support:
            continue
        if powers in seen:
            continue
        seen.add(powers)
        pref_ast = _stageA_local_monomial_ast_from_inputs(local_inputs, powers)
        if pref_ast is None:
            continue
        try:
            pref_readable = ast_to_human_readable(pref_ast)
        except Exception:
            pref_readable = "P"
        variants.append(
            {
                "atom_tag": getattr(atom, "tag", None),
                "atom_var_idxs": tuple(int(v) for v in getattr(atom, "var_idxs", ()) or ()),
                "prefactor_ast": clone_ast(pref_ast),
                "prefactor_powers": tuple(int(v) for v in powers),
                "gauge_shift": int(k),
                "support": int(support),
                "l1": int(sum(abs(int(v)) for v in powers)),
                "confidence": float(conf),
                "prefactor_readable": pref_readable,
            }
        )
    variants.sort(key=lambda row: (int(row["support"]), int(row["l1"]), abs(int(row["gauge_shift"]))))
    return variants


def _stageA_rank_one_prefactor_span_screen(
    *,
    x: torch.Tensor,
    target: torch.Tensor,
    pi_ast: Node,
    prefactor_rows: list[dict],
    n_bins: int,
    min_ok_frac: float,
) -> Optional[dict]:
    """Screen T(x) ~= (sum_j a_j P_j(x)) * H(pi(x)) via a rank-one binned fit."""
    try:
        z = eval_input_expr(pi_ast, x).reshape(-1)
        y = target.reshape(-1)
        p_vals = [eval_input_expr(row["prefactor_ast"], x).reshape(-1) for row in prefactor_rows]
        n = int(min([int(y.numel()), int(z.numel())] + [int(p.numel()) for p in p_vals]))
        if n < 128:
            return None
        z = z[:n]
        y = y[:n]
        p_vals = [p[:n] for p in p_vals]
        mask = torch.isfinite(z) & torch.isfinite(y)
        for p in p_vals:
            mask = mask & torch.isfinite(p)
        ok_frac = float(mask.to(dtype=torch.float64).mean().detach().cpu())
        if ok_frac < float(min_ok_frac):
            return None
        z = z[mask]
        y = y[mask]
        p_vals = [p[mask] for p in p_vals]
        n_ok = int(y.numel())
        if n_ok < 128:
            return None
        y_mean = y.mean()
        var_y = float(torch.mean((y - y_mean) ** 2).detach().cpu())
        if not math.isfinite(var_y) or var_y <= 1.0e-30:
            return None

        nb = int(max(4, min(int(n_bins), n_ok // 8 if n_ok >= 256 else n_ok // 4)))
        nb = max(4, min(nb, n_ok))
        order = torch.argsort(z)
        bin_ids_sorted = torch.div(
            torch.arange(n_ok, device=x.device, dtype=torch.long) * nb,
            max(1, n_ok),
            rounding_mode="floor",
        )
        bin_ids = torch.empty_like(bin_ids_sorted)
        bin_ids[order] = torch.clamp(bin_ids_sorted, 0, nb - 1)

        j_count = len(p_vals)
        cols = j_count * nb
        design = torch.zeros((n_ok, cols), device=x.device, dtype=x.dtype)
        row_idx = torch.arange(n_ok, device=x.device)
        for j, p in enumerate(p_vals):
            design[row_idx, int(j) * nb + bin_ids] = p
        y_col = y.reshape(-1, 1)
        ridge = float(var_y) * 1.0e-8 + 1.0e-12
        eye = torch.eye(cols, device=x.device, dtype=x.dtype)
        ata = design.T @ design + ridge * eye
        aty = design.T @ y_col
        beta = torch.linalg.solve(ata, aty).reshape(j_count, nb)
        pred_full = (design @ beta.reshape(-1, 1)).reshape(-1)
        mse_full = float(torch.mean((pred_full - y) ** 2).detach().cpu())

        try:
            u, s, vh = torch.linalg.svd(beta, full_matrices=False)
        except Exception:
            return None
        if int(s.numel()) == 0:
            return None
        s2 = s * s
        denom = float(torch.sum(s2).detach().cpu())
        if denom <= 1.0e-30:
            return None
        rank_energy = float((s2[0] / torch.sum(s2)).detach().cpu())
        beta_rank1 = s[0] * torch.outer(u[:, 0], vh[0, :])
        pred_rank1 = (design @ beta_rank1.reshape(-1, 1)).reshape(-1)
        mse_rank1 = float(torch.mean((pred_rank1 - y) ** 2).detach().cpu())
        r2_rank1 = float(1.0 - mse_rank1 / (var_y + 1.0e-30))

        coeff = (u[:, 0] * s[0]).detach()
        coeff_cpu = [float(v) for v in coeff.cpu().tolist()]
        max_abs_coeff = max((abs(v) for v in coeff_cpu), default=0.0)
        if max_abs_coeff <= 1.0e-30:
            coeff_init = [1.0 for _ in coeff_cpu]
        else:
            coeff_init = [float(v / max_abs_coeff) for v in coeff_cpu]
        row_norms = [float(torch.linalg.vector_norm(beta_rank1[j]).detach().cpu()) for j in range(j_count)]
        max_row_norm = max(row_norms, default=0.0)
        active = sum(1 for v in row_norms if max_row_norm > 0.0 and v >= 0.05 * max_row_norm)
        return {
            "ok_frac": float(ok_frac),
            "n": int(n_ok),
            "rank_energy": float(rank_energy),
            "mse_full": float(mse_full),
            "mse_rank1": float(mse_rank1),
            "r2_rank1": float(r2_rank1),
            "var_target": float(var_y),
            "coeff_init": coeff_init,
            "active_prefactors": int(active),
        }
    except Exception:
        return None


def _stageA_additive_shared_response_replacement(
    *,
    pi_ast: Node,
    prefactor_rows: list[dict],
    coeff_init: Iterable[float],
    num_segments: int,
    dual_layer: bool,
    tag_prefix: str,
) -> Node:
    q_terms: list[Node] = []
    coeff_vals = list(coeff_init)
    for j, row in enumerate(prefactor_rows):
        init = float(coeff_vals[j]) if j < len(coeff_vals) else 1.0
        scale = Scale(name=f"{tag_prefix}_c{j}", tag=f"{tag_prefix}_c{j}", init=init)
        q_terms.append(MulNode(scale, clone_ast(row["prefactor_ast"])))
    q_ast = _rebuild_additive_chain(q_terms)
    nn_tag = f"{tag_prefix}_h"
    nn_leaf = AtomNode(
        kind="nn",
        var_idxs=tuple(sorted(int(v) for v in _collect_var_idxs_from_node(pi_ast))),
        kwargs={"num_segments": int(num_segments), "dual_layer": bool(dual_layer)},
        tag=nn_tag,
        inputs=(clone_ast(pi_ast),),
    )
    return MulNode(q_ast, nn_leaf)


def _try_stageA_additive_shared_response_block(
    *,
    model,
    current_ast: Node,
    tag_to_leaf: dict,
    datagen_train_noshuffle,
    datagen_val_noshuffle,
    device,
    dtype,
    leaf_builder,
    dual_layer_used: bool,
    search_hp,
    lm_hp,
    loss_target_eff: float,
    accept_threshold_eff_cand: float,
    best_val_loss: float,
    current_val_loss: Optional[float],
    stageA_under_protest: bool,
    best_train_loss: Optional[float],
    loss_scale: float,
    y_op,
    y_op_inv,
    units_spec,
    enforce_units: bool,
    x_transform_map=None,
    data_hp=None,
) -> tuple[bool, Any, Optional[Node], Optional[float], dict]:
    """Try a direct-additive block transaction T -> Q(x) * NN[pi]."""
    if not bool(getattr(search_hp, "additive_shared_response_enable", True)):
        return False, None, None, None, {}
    if (not bool(enforce_units)) or units_spec is None:
        return False, None, None, None, {}
    try:
        if _stageA_x_transform_fingerprint(x_transform_map) != "none":
            return False, None, None, None, {}
    except Exception:
        return False, None, None, None, {}
    direct_terms = _stageA_direct_additive_nn_terms(current_ast)
    if len(direct_terms) < 2:
        return False, None, None, None, {}

    max_siblings = max(2, int(getattr(search_hp, "additive_shared_response_max_siblings", 2) or 2))
    max_pi_groups = max(1, int(getattr(search_hp, "additive_shared_response_max_pi_groups", 2) or 2))
    max_pref_per = max(1, int(getattr(search_hp, "additive_shared_response_max_prefactors_per_sibling", 3) or 3))
    max_candidates = max(1, int(getattr(search_hp, "additive_shared_response_max_candidates", 4) or 4))

    pi_groups: dict[str, dict] = {}
    for term_idx, atom, _term in direct_terms:
        if int(effective_arity(atom)) <= 1:
            continue
        props = _stageA_visible_buckingham_1d_prefactor_proposals_for_atom(
            current_ast=current_ast,
            atom=atom,
            units_spec=units_spec,
            enforce_units=bool(enforce_units),
            search_hp=search_hp,
            x_transform_map=x_transform_map,
        )
        for prop in props:
            try:
                meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
                pi_ast = meta.get("pi_ast") or prop[1]
                pi_key = _stageA_ast_fingerprint(pi_ast)
            except Exception:
                continue
            pref_variants = _stageA_prefactor_gauge_variants_for_asr(
                atom=atom,
                proposal=prop,
                search_hp=search_hp,
            )
            if not pref_variants:
                continue
            group = pi_groups.setdefault(
                pi_key,
                {
                    "pi_ast": clone_ast(pi_ast),
                    "pi_readable": meta.get("pi_readable", "pi"),
                    "members": {},
                },
            )
            member = group["members"].setdefault(
                getattr(atom, "tag", None),
                {"term_idx": int(term_idx), "atom": atom, "prefactors": []},
            )
            member["prefactors"].extend(pref_variants)

    groups = []
    for key, group in pi_groups.items():
        members = {
            tag: row
            for tag, row in dict(group.get("members") or {}).items()
            if row.get("prefactors")
        }
        if len(members) < 2:
            continue
        for row in members.values():
            seen_pref = set()
            deduped = []
            for pref in list(row.get("prefactors") or []):
                pkey = tuple(pref.get("prefactor_powers") or ())
                if pkey in seen_pref:
                    continue
                seen_pref.add(pkey)
                deduped.append(pref)
            deduped.sort(key=lambda r: (int(r.get("support", 99)), int(r.get("l1", 99)), abs(int(r.get("gauge_shift", 99)))))
            row["prefactors"] = deduped[:max_pref_per]
        group["members"] = members
        groups.append((key, group))
    if not groups:
        return False, None, None, None, {}
    groups = groups[:max_pi_groups]

    max_points = getattr(search_hp, "compound_pretrain_max_points", 5000)
    try:
        max_points_i = None if max_points is None else int(max_points)
    except Exception:
        max_points_i = 5000
    x_screen, y_screen = _stageA_collect_loader_xy(
        datagen_train_noshuffle,
        device=device,
        dtype=dtype,
        max_points=max_points_i,
    )
    if x_screen is None or y_screen is None:
        return False, None, None, None, {}

    screen_bins = int(getattr(search_hp, "additive_shared_response_screen_bins", 32) or 32)
    min_ok_frac = float(getattr(search_hp, "additive_shared_response_min_ok_frac", 0.98) or 0.98)
    min_rank_energy = float(getattr(search_hp, "additive_shared_response_min_rank_energy", 0.80) or 0.80)
    min_r2 = float(getattr(search_hp, "additive_shared_response_min_r2_rank1", 0.60) or 0.60)

    proposals = []
    for _pi_key, group in groups:
        members_items = list(dict(group.get("members") or {}).items())
        for subset_size in range(2, min(max_siblings, len(members_items)) + 1):
            for subset in itertools.combinations(members_items, subset_size):
                selected_indices = [int(row["term_idx"]) for _tag, row in subset]
                try:
                    target = _stageA_eval_additive_block_target(
                        current_ast=current_ast,
                        selected_indices=selected_indices,
                        tag_to_leaf=tag_to_leaf,
                        x=x_screen,
                        y=y_screen,
                    )
                except Exception:
                    continue
                pref_lists = [list(row.get("prefactors") or []) for _tag, row in subset]
                for pref_combo in itertools.product(*pref_lists):
                    pref_rows = [dict(p) for p in pref_combo]
                    screen = _stageA_rank_one_prefactor_span_screen(
                        x=x_screen,
                        target=target,
                        pi_ast=group["pi_ast"],
                        prefactor_rows=pref_rows,
                        n_bins=screen_bins,
                        min_ok_frac=min_ok_frac,
                    )
                    if not screen:
                        continue
                    if int(screen.get("active_prefactors", 0)) < 2:
                        continue
                    if float(screen.get("rank_energy", 0.0)) < min_rank_energy:
                        continue
                    if float(screen.get("r2_rank1", 0.0)) < min_r2:
                        continue
                    selected_tags = [str(tag) for tag, _row in subset]
                    tag_seed = "_".join(str(t).replace(" ", "_") for t in selected_tags[:3])
                    tag_prefix = f"asr_{abs(sum(ord(c) for c in str(group.get('pi_readable', 'pi')) + tag_seed))}"
                    parent_segments = max(
                        int(getattr(search_hp, "compound_1d_num_segments", 32) or 32),
                        max(
                            int((row["atom"].kwargs or {}).get("num_segments", 0) or 0)
                            for _tag, row in subset
                        ),
                    )
                    parent_dual = any(bool((row["atom"].kwargs or {}).get("dual_layer", dual_layer_used)) for _tag, row in subset)
                    replacement = _stageA_additive_shared_response_replacement(
                        pi_ast=group["pi_ast"],
                        prefactor_rows=pref_rows,
                        coeff_init=screen.get("coeff_init", []),
                        num_segments=parent_segments,
                        dual_layer=parent_dual,
                        tag_prefix=tag_prefix,
                    )
                    cand_ast = _stageA_replace_direct_additive_subset(
                        current_ast,
                        selected_indices,
                        replacement,
                    )
                    try:
                        from nestynet_sr.sr_core.units import check_units_ast

                        ures = check_units_ast(cand_ast, units_spec)
                        if not bool(getattr(ures, "ok", False)):
                            continue
                    except Exception:
                        continue
                    proposals.append(
                        {
                            "candidate_ast": cand_ast,
                            "selected_indices": selected_indices,
                            "selected_tags": selected_tags,
                            "pi_ast": clone_ast(group["pi_ast"]),
                            "pi_readable": str(group.get("pi_readable", "pi")),
                            "prefactor_rows": pref_rows,
                            "screen": screen,
                            "old_arity": int(sum(int(effective_arity(row["atom"])) for _tag, row in subset)),
                            "new_arity": 1,
                            "parent_num_segments": int(parent_segments),
                            "parent_dual_layer": bool(parent_dual),
                        }
                    )
                    if len(proposals) >= max_candidates * 4:
                        break
                if len(proposals) >= max_candidates * 4:
                    break
            if len(proposals) >= max_candidates * 4:
                break

    if not proposals:
        return False, None, None, None, {}
    proposals.sort(
        key=lambda row: (
            -float(row["screen"].get("rank_energy", 0.0)),
            -float(row["screen"].get("r2_rank1", 0.0)),
            int(sum(int(p.get("l1", 99)) for p in row.get("prefactor_rows", []))),
        )
    )
    proposals = proposals[:max_candidates]

    accepted_rows = []
    tag_to_leaf_current = _build_tag_to_leaf_map(current_ast, model)
    for idx, prop in enumerate(proposals):
        cand_ast = prop["candidate_ast"]
        try:
            desc = _compact_expression_repr(cand_ast, max_length=240, y_op_inv=y_op_inv)
        except Exception:
            desc = ast_to_human_readable(cand_ast)
        print(
            "[Stage A AdditiveSharedResponse] Trying "
            f"pi={prop.get('pi_readable')} siblings={prop.get('selected_tags')} "
            f"rank_energy={float(prop['screen'].get('rank_energy', 0.0)):.3f} "
            f"r2={float(prop['screen'].get('r2_rank1', 0.0)):.3f}; "
            f"proposed: {desc}"
        )
        reuse_raw = {}
        selected_tags = set(str(t) for t in prop.get("selected_tags") or [])
        for tag, leaf in (tag_to_leaf_current or {}).items():
            if str(tag) in selected_tags:
                continue
            reuse_raw[tag] = leaf
        reuse_leaves = _clone_reuse_leaves(reuse_raw, device, dtype)
        try:
            temp_model, _, cand_ast_updated = build_composite_ast(
                cand_ast,
                int(prop.get("parent_num_segments", getattr(search_hp, "compound_1d_num_segments", 32))),
                dual_layer=bool(prop.get("parent_dual_layer", dual_layer_used)),
                leaf_builder=leaf_builder,
                device=device,
                dtype=dtype,
                reuse_leaves=reuse_leaves,
            )
            temp_model = _apply_fit_link_to_model(temp_model, lm_hp)
        except Exception as exc:
            print(f"[Stage A AdditiveSharedResponse] Candidate build failed: {type(exc).__name__}: {exc}")
            continue

        acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)
        accept_threshold = _compute_accept_threshold(
            base_loss=best_val_loss,
            best_loss=best_val_loss,
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            base_params=int(model.num_parameters()),
            cand_params=int(temp_model.num_parameters()),
            loss_floor=float(loss_target_eff),
            loss_cap=float(accept_threshold_eff_cand),
            count_weight=float(getattr(lm_hp, "select_count_weight", 1.0)),
            struct_gamma=float(getattr(lm_hp, "select_struct_gamma", 0.05)),
            param_gamma=float(getattr(lm_hp, "select_param_gamma", 0.30)),
            base_bonus_decades=float(getattr(lm_hp, "select_base_bonus_decades", 0.0)),
            sep_bonus_decades=float(getattr(lm_hp, "select_sep_bonus_decades", 0.05)),
            partial_sep_bonus_decades=float(getattr(lm_hp, "select_partial_sep_bonus_decades", 0.02)),
            is_separability=False,
            max_worsening_factor=float(getattr(search_hp, "max_worsening_factor", 100.0)),
            worsening_floor=float(getattr(search_hp, "worsening_floor", 1.0e-6)) * float(loss_scale),
            hard_ceiling=float("inf"),
            noise_floor=float(acceptance_noise_floor_raw),
        )
        accept_threshold, _structural_target = _accept_threshold_with_structural_target(
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            accept_threshold=accept_threshold,
            loss_target_eff=loss_target_eff,
        )
        accept_threshold, under_protest_cap = _stageA_under_protest_threshold_cap(
            accept_threshold=accept_threshold,
            current_val_loss=current_val_loss if current_val_loss is not None else best_val_loss,
            loss_floor=loss_target_eff,
            noise_floor=acceptance_noise_floor_raw,
            under_protest=bool(stageA_under_protest),
            label="additive shared response",
        )
        if under_protest_cap:
            print("[Stage A AdditiveSharedResponse] Under-protest branch: requiring non-regressing validation loss.")
        max_train_degradation = float(getattr(search_hp, "max_train_degradation", 100.0))
        lane_train_loss_cap = (
            float("inf")
            if best_train_loss is None or best_train_loss <= 0
            else max(max_train_degradation * best_train_loss, loss_target_eff)
        )
        try:
            accepted, cand_val, cand_train, cand_params, cand_opt = fit_stageA_candidate_with_tournament(
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
                y_op=y_op,
                y_op_inv=y_op_inv,
                max_lane_train_loss=lane_train_loss_cap,
                lm_hp=lm_hp,
            )
        except RuntimeError as exc:
            print(f"[Stage A AdditiveSharedResponse] Candidate crashed; rejecting ({type(exc).__name__}: {exc})")
            accepted, cand_val, cand_train, cand_params, cand_opt = False, float("inf"), float("inf"), None, None
        if not accepted:
            print(
                "[Stage A AdditiveSharedResponse] Rejected local fit, "
                f"val-loss {float(cand_val):.4e}"
            )
            continue
        passes_relative = (
            best_train_loss is None
            or best_train_loss <= 0
            or float(cand_train) <= max_train_degradation * float(best_train_loss)
        )
        passes_absolute = float(cand_train) <= float(loss_target_eff)
        if not passes_relative and not passes_absolute:
            print(
                "[Stage A AdditiveSharedResponse] Rejected: training loss "
                f"{float(cand_train):.4e} too degraded."
            )
            continue
        if cand_params is not None and cand_opt is not None:
            cand_opt._update_param_groups(cand_params)
        details = {
            "additive_shared_response": True,
            "unit_certified_additive_shared_response": True,
            "selected_sibling_tags": list(prop.get("selected_tags") or []),
            "shared_pi_readable": str(prop.get("pi_readable", "pi")),
            "screen_rank_energy": float(prop["screen"].get("rank_energy", 0.0)),
            "screen_r2_rank1": float(prop["screen"].get("r2_rank1", 0.0)),
            "screen_mse_rank1": float(prop["screen"].get("mse_rank1", float("inf"))),
            "old_arity": int(prop.get("old_arity", 0)),
            "new_arity": int(prop.get("new_arity", 1)),
        }
        accepted_rows.append(
            {
                "z_name": "additive_shared_response",
                "kind": "additive_shared_response",
                "family": "unit_certified_additive_shared_response",
                "z_readable": str(prop.get("pi_readable", "pi")),
                "model": temp_model,
                "ast": cand_ast_updated,
                "val_loss": float(cand_val),
                "pattern": tuple(str(v) for v in (prop.get("prefactor_rows") or [])),
                "old_arity": int(prop.get("old_arity", 0)),
                "new_arity": int(prop.get("new_arity", 1)),
                "confidence": float(min(0.999, max(0.0, prop["screen"].get("rank_energy", 0.0)))),
                "screen": float(prop["screen"].get("r2_rank1", 0.0)),
                "structural_protected": True,
                "proposal_lane_protected": True,
                "visible_prefactor_transaction": True,
                "additive_shared_response": True,
                "unit_verified": True,
                "details": details,
            }
        )
        print(
            f"{GREEN}[Stage A AdditiveSharedResponse] LM fit passed{RESET}, "
            f"val-loss {float(cand_val):.4e}"
        )

    if not accepted_rows:
        return False, None, None, None, {}
    selected = accepted_rows[0]
    reason = "legacy-stageA-additive-shared-response"
    summary = None
    if len(accepted_rows) > 1 or str(getattr(lm_hp, "coe_mode", "off") or "off") in {"committee_gated", "reservoir_discovery"}:
        selected, reason, summary = _stageA_compound_shortlist_committee_rank(
            base_model=model,
            candidates=accepted_rows,
            lm_hp=lm_hp,
            y_op=y_op,
            y_op_inv=y_op_inv,
            dtype=dtype,
            device=device,
            data_hp=data_hp,
        )
        print("\n" + _format_stageA_compound_shortlist_committee_report(summary))
    if selected is None:
        print(f"{YELLOW}[Stage A AdditiveSharedResponse] CoE rejected all candidates: {reason}{RESET}")
        return False, None, None, None, {"coe_stageA_compound_shortlist": summary}
    try:
        setattr(selected["model"], "_stageA_coe_compound_shortlist", dict(summary or {}))
    except Exception:
        pass
    details = dict(selected.get("details") or {})
    details["coe_stageA_compound_shortlist"] = summary
    try:
        expr_str = _compact_expression_repr(selected["ast"], max_length=240, y_op_inv=y_op_inv)
    except Exception:
        expr_str = ast_to_human_readable(selected["ast"])
    print(
        f"{GREEN}[Stage A AdditiveSharedResponse] Selected{RESET} "
        f"pi={selected.get('z_readable')}, val-loss {float(selected.get('val_loss')):.4e}"
    )
    print(f"[Stage A]   Current: {expr_str}")
    return True, selected["model"], selected["ast"], float(selected["val_loss"]), details


def _stageA_buckingham_reason_after_visible_prefactor_transaction(
    *,
    bare_reason: Optional[str],
    current_ast: Node,
    atom: AtomNode,
    z_expr: Node,
    pattern,
    extra_var_idxs: Optional[List[int]],
    extra_input_asts: Optional[List[Node]],
    prefactor_exponents,
    units_spec,
    enforce_units: bool,
) -> Optional[str]:
    """Keep a bare Buckingham reject unless a visible prefactor transaction is valid."""
    if bare_reason is None:
        return None
    tx_reason = _stageA_visible_prefactor_buckingham_transaction_reason(
        current_ast=current_ast,
        atom=atom,
        z_expr=z_expr,
        pattern=pattern,
        extra_var_idxs=extra_var_idxs,
        extra_input_asts=extra_input_asts,
        prefactor_exponents=prefactor_exponents,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
    )
    if tx_reason is None:
        return None
    return str(bare_reason)


# -----------------------------------------------------------------------------
# Compound-leaf separability helpers (operate in transformed (z, extra) space)
# -----------------------------------------------------------------------------

__search_definitions__ = (
    "_stageA_composite_closure_applicable",
    "_stageA_composite_closure_skip_reason",
    "_stageA_composite_scale_init",
    "_stageA_composite_scalar_atom",
    "_stageA_forced_monomial_reason",
    "_stageA_dim_power_sum",
    "_stageA_forced_monomial_expr_from_units",
    "_stageA_partial_forced_monomial_peel_proposal",
    "_stageA_forced_monomial_loss_equivalent",
    "_stageA_forced_monomial_leftover_candidates",
    "_try_stageA_forced_monomial_prefactor_fallback_candidate",
    "_try_stageA_forced_monomial_closure_candidate",
    "_build_stageA_composite_closure_ast",
    "_stageA_composite_reduces_nn_burden",
    "_stageA_nn_burden_signature",
    "_stageA_shadow_promotion_payoff_reason",
    "_stageA_shadow_promotion_audit",
    "_stageA_cap_terminal_analytic_threshold",
    "_loader_n_eff",
    "_stageA_noisy_terminal_yspace_accept",
    "_stageA_compound_buckingham_reason",
    "_stageA_normalize_nonzero_prefactor_exponents",
    "_stageA_prefactor_peeled_raw_vars",
    "_stageA_visible_prefactor_buckingham_transaction_reason",
    "_stageA_generate_unit_prefactor_exponents",
    "_stageA_local_monomial_ast_from_inputs",
    "_stageA_sparse_integer_power_vectors",
    "_stageA_canonical_dimensionless_power_vector",
    "_stageA_power_vector_complexity_key",
    "_stageA_prefactor_pi_gauge_info",
    "_stageA_visible_buckingham_1d_prefactor_proposals_for_atom",
    "_stageA_append_visible_buckingham_1d_prefactor_proposals",
    "_stageA_direct_additive_nn_terms",
    "_stageA_replace_direct_additive_subset",
    "_stageA_eval_additive_block_target",
    "_stageA_collect_loader_xy",
    "_stageA_prefactor_gauge_variants_for_asr",
    "_stageA_rank_one_prefactor_span_screen",
    "_stageA_additive_shared_response_replacement",
    "_try_stageA_additive_shared_response_block",
    "_stageA_buckingham_reason_after_visible_prefactor_transaction",
)

__search_constants__ = (
    "_FORCED_MONOMIAL_POWERS",
    "_FORCED_MONOMIAL_DENSE_MAX_TERMS",
    "_FORCED_MONOMIAL_SPARSE_MAX_SUPPORT",
    "_FORCED_MONOMIAL_SPARSE_MAX_TERMS",
)

__search_late_bindings__ = (
    "_accept_threshold_with_structural_target",
    "_format_stageA_compound_shortlist_committee_report",
    "_nn_split_signature",
    "_stageA_compound_shortlist_committee_rank",
    "_stageA_terminal_closure_committee_gate",
    "_stageA_under_protest_threshold_cap",
    "_compound_candidate_payoff_policy",
    "_stageA_ast_fingerprint",
    "_stageA_parent_context_descriptor",
    "_stageA_x_transform_fingerprint",
    "_try_stageA_composite_closure_candidate",
)
