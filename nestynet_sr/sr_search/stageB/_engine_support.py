# ruff: noqa: F401, F821
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Policy, diagnostics, and numerical helpers for the Stage-B engine."""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import groupby
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode, AtomNode, MulNode, PowNode,
    LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode,
    ConjNode, RealNode, ImagNode, AbsNode, ArgNode, Node, collect_all_atoms, collect_nn_atoms,
    _collect_var_idxs_from_node,
    atom_problem_label, count_atom_params, effective_arity, eval_inputs, get_input_exprs,
    clone_ast, clone_inputs,
    ast_to_human_readable,
)

# Optional units precheck (PhySO-like dimensional straightjacket).
# This is imported defensively so Stage B remains usable even if the
# units module is not present in some minimal deployments.
try:
    from nestynet_sr.sr_core.units import (
        _dim_in_rational_span,
        UnitsSpec,
        check_units_ast,
        compute_node_domains,
        eval_analytic_expr_dim,
        is_dimless,
        scale_dim,
        infer_atom_output_dim,
    )
except Exception:  # pragma: no cover
    _dim_in_rational_span = None  # type: ignore
    UnitsSpec = None  # type: ignore
    check_units_ast = None  # type: ignore
    compute_node_domains = None  # type: ignore
    eval_analytic_expr_dim = None  # type: ignore
    is_dimless = None  # type: ignore
    scale_dim = None  # type: ignore
    infer_atom_output_dim = None  # type: ignore

# Import hyperparameters and feature specs from sibling modules
# These will be resolved at runtime
if False:  # TYPE_CHECKING
    pass

# Import shared AST utilities from parent module
from ..ast_utils import (
    check_ast_is_tree as _check_ast_is_tree,
)
from ..ast_utils import (
    compact_expression_repr as _compact_expression_repr,
)
from ..coe_witness import (
    CoEWitnessExecutor,
    coe_stageB_refit_ast_to_payload,
    coe_witness_execution_metadata,
    coe_witness_jobs_from_specs,
    run_fixed_expression_pair_witnesses,
    run_stageB_refit_pair_witnesses,
    run_stageB_refit_pair_witness_preflight,
    summarize_witness_errors,
)
from ..model_selection import (
    ast_cost_physics_prior as _ast_cost_physics_prior,
    complexity_key as _complexity_key,
)
from ..model_selection import (
    mapping_cost as _mapping_cost,
)
from ..model_selection import (
    pareto_front_indices_2d as _pareto_front_indices_2d,
)
from ..model_selection import (
    compute_accept_threshold as _compute_accept_threshold,
    loss_within_floor_or_noise_equivalent as _loss_within_floor_or_noise_equivalent,
    noise_equivalent as _noise_equivalent,
    resolve_acceptance_noise_floor_raw as _resolve_acceptance_noise_floor_raw,
)

# Shared model-selection policy (used by both Stage A & Stage B).
from ..model_selection import (
    nn_multivar_complexity as _shared_nn_multivar_complexity,
)
from ..model_selection import (
    nn_structural_score as _nn_structural_score,
)
from ..model_selection import (
    simplification_budget_decades as _simplification_budget_decades,
)
from ..monomial_screen import candidate_monomial_exponent
from .additive_gauge_scope import AdditiveGaugeGlobalScore, AdditiveGaugeScopeIndex, additive_gauge_global_score
from .homogeneous_gauge_scope import (
    HomogeneousGaugeGlobalScore,
    HomogeneousGaugeScopeIndex,
    homogeneous_gauge_global_score,
)

# Terminal colors for logging
GREEN = "\033[32m"
PURPLE = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"


def _snapshot_rng_state() -> dict[str, Any]:
    """Capture process RNG state so disposable CoE refits cannot perturb search."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    try:
        import numpy as _np

        state["numpy"] = _np.random.get_state()
    except Exception:
        state["numpy"] = None
    try:
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        else:
            state["torch_cuda"] = None
    except Exception:
        state["torch_cuda"] = None
    return state


def _restore_rng_state(state: Optional[dict[str, Any]]) -> None:
    if not isinstance(state, dict):
        return
    try:
        random.setstate(state["python"])
    except Exception:
        pass
    try:
        torch.random.set_rng_state(state["torch_cpu"])
    except Exception:
        pass
    try:
        np_state = state.get("numpy")
        if np_state is not None:
            import numpy as _np

            _np.random.set_state(np_state)
    except Exception:
        pass
    try:
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    except Exception:
        pass


GAUGE_SCOPE_RULES = {
    "additive_gauge_transfer",
    "multiplicative_homogeneity_transfer",
    "common_prefactor",
    "overlap_prefactor_peel",
    "overlap_counterterm_peel",
    "counterfactor_add_split",
    "counterterm_mul_split",
    "coupled_leaf_ratio",
}

GAUGE_TERMINALISH_RULES = {
    "compound_fn_macros",
    "compound_planck",
    "monomial_prefactor_compound",
    "joint_product_monomial_closure",
    "last_hard_trig_square",
    "last_hard_trig_power",
    "phase_hint_trig_closure",
    "phase_hint_reciprocal_trig_power",
    "phase_context_trig_closure",
    "inverse_trig_outer_closure",
    "inverse_trig_outer_rational_closure",
    "univariate_nn",
    "log_ratio",
    "univariate_oracle_invariants",
}

GAUGE_SENSITIVE_RULES = {
    "homogeneity_peel",
    "ratio_invariance",
    "product_homogeneity",
    "nonlinear_substitution",
    "affine_decomp",
    "poly_split",
    "multid_nn",
    "factorized_search",
    "preconditioner_fallback_nn",
    "nn_leaf_separability",
}


def _safe_ast_cost(root) -> Optional[float]:
    """Compute ast_cost_physics_prior, returning None on failure."""
    try:
        return float(_ast_cost_physics_prior(root))
    except Exception:
        return None


def _clamp_nonnegative_finite(value: Any, default: float = 0.0) -> float:
    """Parse float; return default if non-finite or negative."""
    try:
        out = float(value)
    except Exception:
        return float(default)
    if (not math.isfinite(out)) or out < 0.0:
        return float(default)
    return float(out)


def _loss_excess_above_floor(value: Any, noise_floor: float) -> float:
    """Return max(value - noise_floor, 0), preserving +inf when present."""
    try:
        out = float(value)
    except Exception:
        return float("inf")
    if math.isnan(out):
        return float("inf")
    if math.isinf(out):
        return float("inf") if out > 0 else 0.0
    nf = _clamp_nonnegative_finite(noise_floor, default=0.0)
    return float(max(0.0, out - nf))


def _effective_loss_floor(loss_floor: float, base_loss: float, guard_decades: float) -> float:
    """Adaptive floor used when deciding whether losses are effectively tied."""
    floor = _clamp_nonnegative_finite(loss_floor, default=0.0)
    base = _clamp_nonnegative_finite(base_loss, default=0.0)
    guard = _clamp_nonnegative_finite(guard_decades, default=0.0)
    if base <= 0.0:
        return float(floor)
    try:
        guarded_floor = base * (10.0 ** guard)
    except Exception:
        guarded_floor = float("inf")
    if not math.isfinite(guarded_floor):
        return float(floor)
    return float(min(floor, max(0.0, guarded_floor)))


def _best_seen_restore_decision(
    *,
    cur_loss: float,
    best_loss: float,
    cur_root: Node,
    best_root: Node,
    n_params_cur: int,
    n_params_best: int,
    loss_floor: float,
    loss_floor_eff: float,
    count_weight: float,
    cur_mapping_cost: float = 0.0,
    best_mapping_cost: float = 0.0,
    losses_noise_equivalent: bool = False,
) -> Tuple[bool, str]:
    """Decide whether to restore ``best_seen`` over the current final state.

    When both states are already below the nominal loss floor, a strict loss-only
    comparison can incorrectly reintroduce NN leaves after later passes have
    found a fully analytical form. In that regime, if the NN structural scores
    differ, we allow the nominal floor to unlock the usual complexity tie-break.
    Noisy runs need the same treatment when the two raw losses are statistically
    equivalent, even if their excesses above the noise floor differ slightly.
    """
    restore = True
    reason = f"loss={best_loss:.3e} < current={cur_loss:.3e}"

    cmp_floor = _clamp_nonnegative_finite(loss_floor_eff, default=0.0)
    nominal_floor = _clamp_nonnegative_finite(loss_floor, default=0.0)

    if losses_noise_equivalent:
        cmp_floor = max(cmp_floor, cur_loss, best_loss)

    if nominal_floor > 0.0 and cur_loss <= nominal_floor and best_loss <= nominal_floor:
        try:
            cur_nn_score = float(
                _nn_structural_score(cur_root, count_weight=float(count_weight))
            )
            best_nn_score = float(
                _nn_structural_score(best_root, count_weight=float(count_weight))
            )
        except Exception:
            cur_nn_score = None
            best_nn_score = None
        if (
            cur_nn_score is not None
            and best_nn_score is not None
            and cur_nn_score != best_nn_score
        ):
            cmp_floor = max(cmp_floor, nominal_floor)

    if (cur_loss <= cmp_floor) and (best_loss <= cmp_floor):
        cur_key = _complexity_key(cur_root, n_params_cur, count_weight=count_weight)
        best_key = _complexity_key(best_root, n_params_best, count_weight=count_weight)
        cur_map_cost = _clamp_nonnegative_finite(cur_mapping_cost, default=0.0)
        best_map_cost = _clamp_nonnegative_finite(best_mapping_cost, default=0.0)
        cur_key = (float(cur_key[0]) + float(cur_map_cost), int(cur_key[1]))
        best_key = (float(best_key[0]) + float(best_map_cost), int(best_key[1]))

        if best_key < cur_key:
            restore = True
            reason = (
                f"both<=floor_cmp({cmp_floor:.3e}); "
                f"best complexity {best_key} < current {cur_key}"
            )
        elif best_key == cur_key:
            restore = True
            reason = (
                f"both<=floor_cmp({cmp_floor:.3e}); "
                f"equal complexity {best_key}, better loss"
            )
        else:
            restore = False
            reason = (
                f"both<=floor_cmp({cmp_floor:.3e}); "
                f"keeping current complexity {cur_key} <= best {best_key}"
            )

    return bool(restore), str(reason)


def _below_floor_regression_cap(base_loss: float, max_regress_decades: float) -> float:
    """Hard cap on allowed loss regression in the below-floor regime."""
    base = _clamp_nonnegative_finite(base_loss, default=0.0)
    if base <= 0.0:
        return 0.0
    max_dec = _clamp_nonnegative_finite(max_regress_decades, default=0.0)
    try:
        cap = base * (10.0 ** max_dec)
    except Exception:
        cap = base
    if not math.isfinite(cap):
        return float("inf")
    return float(max(base, cap))


def _below_floor_regression_rejected(
    *,
    cand_loss: float,
    below_floor_regress_cap: float,
    is_separability_rewrite: bool,
    relaxed_below_floor: bool = False,
) -> bool:
    """Return True when below-floor regression should be rejected."""
    if bool(relaxed_below_floor):
        return False
    if bool(is_separability_rewrite):
        return False
    try:
        return float(cand_loss) > float(below_floor_regress_cap)
    except Exception:
        return False


def _candidate_mapping_cost(cand: "Candidate") -> float:
    """Extract explicit mapping complexity from candidate metadata."""
    meta = getattr(cand, "meta", None)
    if not isinstance(meta, dict):
        return 0.0
    # Prefer explicit precomputed cost if present.
    try:
        c = float(meta.get("mapping_cost", meta.get("factorized_mapping_cost", 0.0)))
        if math.isfinite(c) and c >= 0.0:
            return c
    except Exception:
        pass
    # Fallback: derive from raw mapping dict when available.
    try:
        m = meta.get("factorized_mapping")
        c = float(_mapping_cost(m))
        if math.isfinite(c) and c >= 0.0:
            return c
    except Exception:
        pass
    return 0.0


def _candidate_is_unpromoted_generic(cand: "Candidate") -> bool:
    """Best-effort Stage-B generic-approximant flag before Stage C promotion.

    Some ratpoly-like routes are later promoted to clean non-generic formulas
    after simplification/snapping.  During the engine pass, though, an accepted
    generic family should not floor-lock the search while later cleaner
    candidates remain in the menu.
    """

    meta = getattr(cand, "meta", None)
    if isinstance(meta, dict):
        if bool(
            meta.get("exact_non_generic", False)
            or meta.get("promoted_non_generic", False)
            or meta.get("simple_integer_rational_expr", False)
            or meta.get("stageB_non_generic", False)
        ):
            return False
        if meta.get("generic_approximant") is False:
            return False
        if meta.get("generic_approximant") is True:
            return True

    label = str(getattr(cand, "label", "") or "").strip().lower()
    if not label:
        return False
    return any(
        token in label
        for token in (
            "ratpoly",
            "rratpoly",
            "rationalpoly",
            "rational_poly",
            "exp_rat",
            "sqrt_rat",
            "log_rat",
        )
    )


def _mapping_descriptor(mapping: Any) -> Dict[str, Any]:
    """Summarise mapping family for logging/auditing."""
    m = mapping if isinstance(mapping, dict) else {}
    kind = str(m.get("kind", "")).strip().lower()
    degree: Optional[int] = None
    mclass = "none"
    is_struct = False

    if kind == "poly":
        coeffs = m.get("coeffs", [])
        degree = max(0, len(coeffs) - 1) if isinstance(coeffs, (list, tuple)) else None
        is_struct = degree is not None and int(degree) <= 1
        mclass = "structural" if is_struct else "approximative"
    elif kind in ("monomial", "mono"):
        is_struct = True
        mclass = "structural"
    elif kind in ("power", "sine", "exp"):
        is_struct = True
        mclass = "structural"
    elif kind == "pade":
        numer = m.get("numer", [])
        denom = m.get("denom", [])
        p = max(0, len(numer) - 1) if isinstance(numer, (list, tuple)) else 0
        q = max(0, len(denom) - 1) if isinstance(denom, (list, tuple)) else 0
        degree = int(p + q)
        mclass = "approximative"
    elif kind in ("", "identity"):
        mclass = "none"
    elif kind:
        mclass = "other"

    return {
        "kind": kind if kind else None,
        "degree": degree,
        "class": mclass,
        "is_structural": bool(is_struct),
    }


def _candidate_mapping_descriptor(cand: "Candidate") -> Dict[str, Any]:
    """Mapping descriptor derived from candidate metadata."""
    meta = getattr(cand, "meta", None)
    if not isinstance(meta, dict):
        desc = _mapping_descriptor(None)
    else:
        desc = _mapping_descriptor(meta.get("factorized_mapping"))
    desc["cost"] = _candidate_mapping_cost(cand)
    return desc


def _candidate_has_mapping(cand: "Candidate") -> bool:
    """Return True when candidate carries explicit mapping metadata."""
    try:
        desc = _candidate_mapping_descriptor(cand)
        if str(desc.get("class", "none")) != "none":
            return True
    except Exception:
        pass
    try:
        return _candidate_mapping_cost(cand) > 0.0
    except Exception:
        return False


def _candidate_is_structural_accept(cand: "Candidate") -> bool:
    """Classify whether an accepted rewrite should count as structural."""
    try:
        if _is_structural_candidate(cand):
            return True
    except Exception:
        pass
    try:
        desc = _candidate_mapping_descriptor(cand)
        if bool(desc.get("is_structural", False)):
            return True
    except Exception:
        pass
    return False


def _phase2_trigger_flags(
    *,
    improved: bool,
    phase1_accept_count: int,
    phase1_structural_accept_count: int,
    phase1_mapping_accept_count: int,
    phase1_mapping_structural_accept_count: int,
) -> Tuple[bool, bool, bool]:
    """Return (run_phase2, only_nonstruct_accepts, only_nonstruct_mapping_accepts)."""
    only_nonstruct_accepts = (
        int(phase1_accept_count) > 0 and int(phase1_structural_accept_count) == 0
    )
    only_nonstruct_mapping_accepts = (
        int(phase1_mapping_accept_count) > 0
        and int(phase1_mapping_structural_accept_count) == 0
    )
    run_phase2 = (
        (not bool(improved))
        or bool(only_nonstruct_accepts)
        or bool(only_nonstruct_mapping_accepts)
    )
    return bool(run_phase2), bool(only_nonstruct_accepts), bool(
        only_nonstruct_mapping_accepts
    )


def _target_uid(root: Node, target: Node) -> str:
    """Path-from-root UID for a target node inside an AST.

    Priority: (1) AtomNode with tag -> "Atom#<tag>",
    (2) walk root to find target by identity -> "TypeName@L.R.A",
    (3) fallback -> "TypeName@id=<id>".
    """
    if isinstance(target, AtomNode) and getattr(target, "tag", None) is not None:
        return f"Atom#{target.tag}"

    def _walk(node, parts):
        if node is target:
            return True
        if isinstance(node, (AddNode, MulNode)):
            for label, child in (("L", node.left), ("R", node.right)):
                parts.append(label)
                if _walk(child, parts):
                    return True
                parts.pop()
        elif isinstance(node, PowNode):
            parts.append("B")
            if _walk(node.base, parts):
                return True
            parts.pop()
        elif isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode,
                               ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
            parts.append("A")
            if _walk(node.arg, parts):
                return True
            parts.pop()
        elif isinstance(node, AtomNode):
            for i, inp in enumerate(getattr(node, "inputs", None) or ()):
                parts.append(f"i{i}")
                if _walk(inp, parts):
                    return True
                parts.pop()
        return False

    parts: list = []
    if _walk(root, parts):
        return f"{type(target).__name__}@{'.' .join(parts)}" if parts else f"{type(target).__name__}@root"
    return f"{type(target).__name__}@id={id(target)}"


def _eval_yspace_mse(model: torch.nn.Module, val_loader, device: torch.device) -> float:
    """Evaluate MSE in original y-space (not asinh-transformed).

    This is used for sanity checking when asinh fit-link is active.
    Returns the raw MSE of model predictions vs targets.
    """
    model.eval()
    se_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                continue
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            if y_pred.dim() == 2:
                y_pred = y_pred[:, 0]
            else:
                y_pred = y_pred.view(-1)
            if y.dim() == 2:
                y_true = y[:, 0]
            else:
                y_true = y.view(-1)
            diff = y_pred - y_true
            se_sum += float((diff * diff).sum().cpu())
            n_total += diff.numel()
    if n_total == 0:
        return float("inf")
    return se_sum / float(n_total)


def _asinh_yspace_scale_from_loader(
    val_loader, device: torch.device, s: float, q: float = 0.9, max_points: int = 20000
) -> float:
    """Compute D_ref = quantile_q(s² + y²) for correct asinh-to-yspace scaling.

    The asinh Jacobian is d(asinh(y/s))/dy = 1/sqrt(s² + y²), so the squared
    Jacobian is 1/(s² + y²). To convert asinh-space loss to y-space units,
    multiply by D = s² + y². We use a quantile to get a robust reference.

    Args:
        val_loader: Validation data loader
        device: torch device
        s: asinh scale parameter
        q: quantile (0.9 = 90th percentile)
        max_points: max points to sample for efficiency

    Returns:
        D_ref: Reference scale factor for converting asinh loss to y-space units
    """
    ys = []
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                _, y = batch
            else:
                continue
            y = y.to(device)
            if y.dim() == 2:
                y = y[:, 0]
            ys.append(y.detach().flatten())
            if sum(t.numel() for t in ys) >= max_points:
                break
    if not ys:
        return float("nan")
    y = torch.cat(ys, dim=0)
    # D = s² + y²
    D = (float(s) ** 2) + (y * y)
    return float(torch.quantile(D, q).item())


def _loss_str(loss: float, lm_hp) -> str:
    """Format loss value with asinh indicator if fit-link is active."""
    if getattr(lm_hp, "fit_y_link", None) == "asinh":
        return f"{loss:.4e} [asinh]"
    return f"{loss:.4e}"


def _format_dim_for_problem(units_spec: Any, dim: Any) -> str:
    """Format a dimension vector for log/display messages."""
    try:
        us = getattr(units_spec, "unit_system", None)
        if us is not None and hasattr(us, "format_dim"):
            return str(us.format_dim(dim))
    except Exception:
        pass
    return str(dim)


def _target_dim_for_root(root: Node, atom: AtomNode, units_spec: Any):
    """Infer the required output dimension for *atom* within *root*."""
    result = None
    if infer_atom_output_dim is not None:
        try:
            result = infer_atom_output_dim(root, atom, units_spec)
        except Exception:
            result = None
    if result is None and compute_node_domains is not None:
        try:
            span_spec = replace(units_spec, nn_semantics="span")
            domains = compute_node_domains(root, span_spec)
            if domains is not None:
                dom = domains.get(id(atom))
                if dom is not None and dom.is_pinned():
                    result = dom.offset
        except Exception:
            result = None
    return result


def _input_basis_dims_for_atom(atom: AtomNode, units_spec: Any) -> List[Any]:
    """Return non-dimless basis dims available to an atom from inputs/constants."""
    basis: List[Any] = []
    seen: Set[Any] = set()

    def _append_dim(d: Any) -> None:
        if d is None:
            return
        try:
            if is_dimless is not None and is_dimless(d):
                return
        except Exception:
            pass
        if d in seen:
            return
        seen.add(d)
        basis.append(d)

    try:
        if eval_analytic_expr_dim is not None:
            for inp in get_input_exprs(atom):
                _append_dim(
                    eval_analytic_expr_dim(
                        inp,
                        units_spec.x_dims,
                        free_const_dims=(
                            getattr(units_spec, "free_const_dims", {}) or {}
                        ),
                        fixed_const_dims=(
                            getattr(units_spec, "fixed_const_dims", {}) or {}
                        ),
                    )
                )
        else:
            for idx in atom.var_idxs:
                j = int(idx)
                if 0 <= j < len(units_spec.x_dims):
                    _append_dim(units_spec.x_dims[j])
        for d in list((getattr(units_spec, "free_const_dims", {}) or {}).values()):
            _append_dim(d)
        for d in list((getattr(units_spec, "fixed_const_dims", {}) or {}).values()):
            _append_dim(d)
    except Exception:
        pass
    return basis


def _find_nonsense_units_leaves(
    root: Node,
    *,
    units_spec: Any,
    enforce_units: bool,
    log_fn: Optional[Callable[[str], None]] = None,
    mutate: bool = True,
) -> List[dict]:
    """Find NN leaves whose required units are unreachable from their own inputs."""
    problems: List[dict] = []
    if (
        not enforce_units
        or units_spec is None
        or _dim_in_rational_span is None
    ):
        return problems

    for atom in collect_all_atoms(root):
        if not isinstance(atom, AtomNode) or str(atom.kind).lower() != "nn":
            continue

        target_dim = _target_dim_for_root(root, atom, units_spec)
        if target_dim is None:
            continue

        basis_dims = _input_basis_dims_for_atom(atom, units_spec)
        try:
            reachable = bool(_dim_in_rational_span(target_dim, basis_dims))
        except Exception:
            reachable = True
        if reachable:
            if mutate and atom_problem_label(atom) == "nonsense_units":
                kw = dict(getattr(atom, "kwargs", {}) or {})
                for key in (
                    "_problem_code",
                    "_problem_label",
                    "_problem_reason",
                    "_problem_msg",
                ):
                    kw.pop(key, None)
                atom.kwargs = kw
            continue

        target_dim_str = _format_dim_for_problem(units_spec, target_dim)
        basis_dim_strs = [_format_dim_for_problem(units_spec, d) for d in basis_dims]
        input_parts = [
            ast_to_human_readable(inp)
            for inp in get_input_exprs(atom)
        ]
        was_flagged = atom_problem_label(atom) == "nonsense_units"
        problem_msg = (
            f"target={target_dim_str}; basis="
            f"{basis_dim_strs if basis_dim_strs else ['dimless-only']}"
        )
        if mutate:
            kw = dict(getattr(atom, "kwargs", {}) or {})
            kw["_problem_code"] = "nonsense_units"
            kw["_problem_label"] = "nonsense_units"
            kw["_problem_reason"] = "target-units-unreachable-from-inputs"
            kw["_problem_msg"] = problem_msg
            atom.kwargs = kw

        rec = {
            "label": "nonsense_units",
            "tag": getattr(atom, "tag", None),
            "var_idxs": tuple(int(v) for v in getattr(atom, "var_idxs", ())),
            "inputs": input_parts,
            "target_dim": target_dim_str,
            "basis_dims": list(basis_dim_strs),
            "message": problem_msg,
        }
        problems.append(rec)

        if (not was_flagged) and log_fn is not None:
            tag_s = f"#{atom.tag}" if getattr(atom, "tag", None) else ""
            inputs_s = ", ".join(input_parts) if input_parts else "no inputs"
            basis_s = ", ".join(basis_dim_strs) if basis_dim_strs else "dimless-only"
            log_fn(
                f"{RED}[Stage B] nonsense_units{RESET} leaf nn{tag_s} "
                f"on ({inputs_s}): target {target_dim_str} unreachable from basis [{basis_s}]"
            )

    return problems


def _annotate_nonsense_units_leaves(
    state: "StageBState",
    *,
    units_spec: Any,
    enforce_units: bool,
    log_fn: Optional[Callable[[str], None]] = None,
    mutate: bool = True,
) -> List[dict]:
    """Tag NN leaves whose required units are unreachable from their own inputs."""
    state.problem_leaves = None
    problems = _find_nonsense_units_leaves(
        state.root,
        units_spec=units_spec,
        enforce_units=enforce_units,
        log_fn=log_fn,
        mutate=mutate,
    )
    if problems:
        state.problem_leaves = problems
    return problems


def _problem_candidate_desc(cand: "Candidate") -> str:
    """Human-readable description for problem-leaf repair candidates."""
    meta = getattr(cand, "meta", None)
    if not isinstance(meta, dict):
        return "flagged leaf"
    label = str(meta.get("problem_label", "problem_leaf") or "problem_leaf")
    tag = meta.get("problem_tag")
    inputs = meta.get("problem_inputs", [])
    tag_s = f"#{tag}" if tag else ""
    if isinstance(inputs, (list, tuple)):
        inputs_s = ", ".join(str(x) for x in inputs if x is not None)
    else:
        inputs_s = str(inputs)
    if not inputs_s:
        inputs_s = "no inputs"
    return f"{label}{tag_s} inputs=({inputs_s})"


# Structural candidate classification
STRUCTURAL_LABEL_PREFIXES = ("outer_",)
STRUCTURAL_LABELS = {
    "monomial_peel_priority",
    "counterterm_mul_split",
    "counterfactor_add_split",
    "overlap_counterterm_peel",
    "overlap_prefactor_peel",
    "subtree_separability",
    "poly_split",
    "nn_leaf_separability",
    "nn_variable_prune",
    "peel_known_factor",
    "homogeneity_peel",  # peel x^k factor from homogeneous degree-k functions
    "ratio_invariance",
    "coupled_leaf_ratio",
    "trig_diff",
    "trig_diff_affine_env",
    "trig_comp",
    "trig_affine_env",
    "sqrt_ratpoly",  # replaces multivariate NN with sqrt(rational_poly)
    "sqrt_ratpoly_1d",  # effective-1D counterpart for compound NN[z] leaves
    "affine_decomp",  # g(f(z,w)) = a(z) + b(z)*h(w) decomposition
    "common_prefactor",  # factor shared multiplicative structure from AddNode siblings
    "joint_product_monomial_closure",  # terminal closure of coupled NN products
    "nonsense_units_zero_prune",  # replace dimensionally impossible NN leaf with zero
}


def candidate_pattern_name(cand_or_label: Any) -> str:
    """Normalise indexed candidate labels to their family name.

    Examples:
      - ``ratpoly[2]`` -> ``ratpoly``
      - ``factorized_search[0]`` -> ``factorized_search``

    Candidates may also provide an explicit ``meta["pattern_family"]``.
    """
    meta = getattr(cand_or_label, "meta", None)
    if isinstance(meta, dict):
        family = meta.get("pattern_family")
        if isinstance(family, str) and family:
            return family

    label = getattr(cand_or_label, "label", cand_or_label)
    if label is None:
        return ""
    label = str(label)
    if label.endswith("]"):
        prefix, sep, suffix = label.rpartition("[")
        if prefix and sep and suffix[:-1].isdigit():
            return prefix
    return label


# Separability rewrites: candidates that decompose an NN into sub-problems,
# unlocking further Stage B iterations.  This is a strict subset of
# STRUCTURAL_LABELS — it excludes terminal fits like sqrt_ratpoly that
# produce a closed-form expression without creating new sub-trees.
SEPARABILITY_LABELS = {
    "monomial_peel_priority",
    "nn_leaf_separability",
    "nn_leaf_separability_sq",
    "nn_leaf_separability_sqrt",
    "nn_variable_prune",
    "counterterm_mul_split",
    "counterfactor_add_split",
    "overlap_counterterm_peel",
    "overlap_prefactor_peel",
    "peel_known_factor",
    "gauge_mul_split",
    "gauge_add_split",
    "homogeneity_peel",
    "multiplicative_homogeneity_transfer",
    "ratio_invariance",
    "coupled_leaf_ratio",
    "affine_split",
    "trig_diff_affine_env",
    "affine_decomp",
    "common_prefactor",
}

def _count_ast_params(root: Node) -> int:
    """Total trainable parameter count for a proposed AST (pre-fitting)."""
    return sum(count_atom_params(a) for a in collect_all_atoms(root))


def _candidate_min_free_params(cand: "Candidate") -> int:
    """Lower bound on the fitted degrees of freedom for a candidate family.

    Several proposal families have tied, fixed, or hidden coefficients, so the
    naive AST parameter count can overstate or understate the minimum fitted
    complexity.  Candidate builders can expose the correct lower bound through
    ``meta["min_free_params"]`` or the older ``meta["n_free_params"]``.
    """
    meta = getattr(cand, "meta", None)
    if isinstance(meta, dict):
        for key in ("min_free_params", "n_free_params"):
            override = meta.get(key)
            if override is not None:
                try:
                    return max(0, int(override))
                except Exception:
                    pass
    if getattr(cand, "root", None) is None:
        try:
            if not cand.materialise():
                return int(1e18)
        except Exception:
            return int(1e18)
    return _count_ast_params(cand.root)


def _cand_sort_key(cand: "Candidate") -> int:
    """Sort key for exhaustive candidate evaluation."""

    return _candidate_min_free_params(cand)


def _candidate_can_beat_floor_locked_state(
    ctx: "StageBContext",
    cand: "Candidate",
    candidate_min_free_params_fn: Optional[Callable[["Candidate"], int]] = None,
) -> bool:
    """True if a candidate can still beat a below-floor incumbent.

    Below the meaningful loss floor, more accuracy is not evidence.  A later
    candidate should only bypass the floor lock if its visible structure or
    minimum free-parameter lower bound can beat the current state by the same
    complexity ordering used by selection.
    """

    if _is_separability_candidate(cand):
        return True
    if getattr(cand, "root", None) is None:
        try:
            if not cand.materialise():
                return False
        except Exception:
            return False

    try:
        cand_nn = len(collect_nn_atoms(cand.root))
    except Exception:
        cand_nn = int(1e18)
    cur_nn = _stageB_state_num_nn_atoms(ctx.state)
    if cand_nn < cur_nn:
        return True

    try:
        count_weight = float(getattr(ctx.lm_hp, "select_count_weight", 1.0))
    except Exception:
        count_weight = 1.0
    try:
        cur_params = _stageB_state_num_params(ctx.state)
        cand_params = (
            int(candidate_min_free_params_fn(cand))
            if candidate_min_free_params_fn is not None
            else _candidate_min_free_params(cand)
        )
        cur_key = _complexity_key(
            ctx.state.root,
            cur_params,
            count_weight=count_weight,
        )
        cand_key = _complexity_key(
            cand.root,
            cand_params,
            count_weight=count_weight,
        )
        cur_map_cost = _clamp_nonnegative_finite(
            getattr(ctx.state, "complexity_mapping_cost", 0.0),
            default=0.0,
        )
        cand_map_cost = _candidate_mapping_cost(cand)
        cur_key = (float(cur_key[0]) + float(cur_map_cost), int(cur_key[1]))
        cand_key = (float(cand_key[0]) + float(cand_map_cost), int(cand_key[1]))
        return bool(cand_key < cur_key)
    except Exception:
        return False


def _is_exact_final_leaf_monomial_accept(
    ctx: "StageBContext",
    cand: "Candidate",
    cand_state: "StageBState",
) -> bool:
    """Return True when a confirmed candidate has solved the final leaf as a monomial."""

    if candidate_monomial_exponent(getattr(cand, "label", "")) is None:
        return False
    if ctx.loss_floor is None:
        return False
    try:
        if len(collect_nn_atoms(ctx.state.root)) != 1:
            return False
        if len(collect_nn_atoms(cand_state.root)) != 0:
            return False
    except Exception:
        return False
    try:
        cand_loss = float(cand_state.val_loss)
        loss_floor = float(ctx.loss_floor)
    except Exception:
        return False
    return math.isfinite(cand_loss) and math.isfinite(loss_floor) and cand_loss <= loss_floor


def _stageB_state_num_params(state: "StageBState") -> int:
    """Best-effort trainable parameter count for a Stage-B state."""

    model = getattr(state, "model", None)
    if model is None:
        return int(1e18)
    try:
        return int(model.num_parameters())
    except Exception:
        pass
    try:
        return int(
            sum(
                int(p.numel())
                for p in model.parameters()
                if getattr(p, "requires_grad", False)
            )
        )
    except Exception:
        return int(1e18)


def _stageB_state_num_nn_atoms(state: "StageBState") -> int:
    """Best-effort unresolved NN count for a Stage-B state."""

    n = getattr(state, "num_nn_atoms", None)
    if n is not None:
        try:
            return int(n)
        except Exception:
            pass
    try:
        return int(len(collect_nn_atoms(state.root)))
    except Exception:
        return int(1e18)


def _stageB_completion_loss_floor(ctx: "StageBContext") -> float:
    """Loss-equivalence floor used by the terminal completion predicate."""

    floor = _clamp_nonnegative_finite(getattr(ctx, "loss_floor", None), default=0.0)
    if floor <= 0.0:
        floor = _clamp_nonnegative_finite(
            getattr(ctx, "loss_good_enough_raw", None),
            default=0.0,
        )
    return float(floor)


def _min_following_candidate_free_params(
    following_candidates: Optional[List["Candidate"]],
    candidate_min_free_params_fn: Optional[Callable[["Candidate"], int]] = None,
) -> Optional[int]:
    """Minimum published free-parameter lower bound in a remaining menu."""

    if not following_candidates:
        return None
    best: Optional[int] = None
    for cand in following_candidates:
        if cand is None:
            continue
        try:
            n = (
                int(candidate_min_free_params_fn(cand))
                if candidate_min_free_params_fn is not None
                else _candidate_min_free_params(cand)
            )
        except Exception:
            n = _candidate_min_free_params(cand)
        if best is None or n < best:
            best = int(n)
    return best


def _are_we_done_yet(
    ctx: "StageBContext",
    state: Optional["StageBState"] = None,
    *,
    following_candidates: Optional[List["Candidate"]] = None,
    candidate_min_free_params_fn: Optional[Callable[["Candidate"], int]] = None,
) -> bool:
    """True when Stage B has reached a terminal, non-improvable state.

    This is intentionally stricter than "no NN atoms remain" but does *not*
    require zero fitted parameters.  A fully analytic expression with fitted
    constants can be final once the normal post-accept polish has had its
    chance, the validation loss is already equivalent under the same
    floor/noise-floor semantics used by Stage-B selection, and the following
    candidate menu cannot lower the published free-parameter bound.
    """

    state = state if state is not None else ctx.state
    if state is None:
        return False

    if getattr(state, "problem_leaves", None):
        return False
    if _stageB_state_num_nn_atoms(state) != 0:
        return False

    try:
        loss = float(state.val_loss)
    except Exception:
        return False
    if not math.isfinite(loss):
        return False

    try:
        noise_floor = _resolve_acceptance_noise_floor_raw(ctx.lm_hp, ctx.loss_scale)
    except Exception:
        noise_floor = 0.0
    try:
        noise_n_eff = float(getattr(state, "acceptance_noise_n_eff", None))
        if (not math.isfinite(noise_n_eff)) or noise_n_eff <= 0.0:
            noise_n_eff = None
    except Exception:
        noise_n_eff = None
    try:
        noise_n_eff = float(getattr(state, "acceptance_noise_n_eff", None))
        if (not math.isfinite(noise_n_eff)) or noise_n_eff <= 0.0:
            noise_n_eff = None
    except Exception:
        try:
            noise_n_eff = float(getattr(ctx, "acceptance_noise_n_eff", None))
            if (not math.isfinite(noise_n_eff)) or noise_n_eff <= 0.0:
                noise_n_eff = None
        except Exception:
            noise_n_eff = None
    floor = _stageB_completion_loss_floor(ctx)
    if not _loss_within_floor_or_noise_equivalent(
        loss,
        floor,
        noise_floor=noise_floor,
        n_eff=noise_n_eff,
    ):
        return False

    # If a y-transform branch tracked original-y metrics, require that branch
    # to be good enough in original y-space too before declaring the search done.
    original_y_loss = getattr(state, "original_y_val_loss", None)
    if original_y_loss is not None:
        try:
            original_y_loss = float(original_y_loss)
        except Exception:
            return False
        if not math.isfinite(original_y_loss):
            return False
        original_floor = _clamp_nonnegative_finite(
            getattr(state, "original_y_loss_good_enough_eff", None),
            default=floor,
        )
        original_noise_floor = _clamp_nonnegative_finite(
            getattr(state, "original_y_noise_floor_raw", None),
            default=noise_floor,
        )
        if not _loss_within_floor_or_noise_equivalent(
            original_y_loss,
            original_floor,
            noise_floor=original_noise_floor,
            n_eff=noise_n_eff,
        ):
            return False

    following_min = _min_following_candidate_free_params(
        following_candidates,
        candidate_min_free_params_fn,
    )
    if (
        bool(getattr(state, "generic_approximant_unpromoted", False))
        and following_candidates
    ):
        return False
    if following_min is not None and following_min < _stageB_state_num_params(state):
        return False

    return True


def _are_we_done_yet_reason(
    ctx: "StageBContext",
    state: Optional["StageBState"] = None,
    *,
    following_candidates: Optional[List["Candidate"]] = None,
    candidate_min_free_params_fn: Optional[Callable[["Candidate"], int]] = None,
) -> str:
    """Compact log reason for the terminal completion predicate."""

    state = state if state is not None else ctx.state
    try:
        loss = float(state.val_loss)
    except Exception:
        loss = float("nan")
    try:
        noise_floor = _resolve_acceptance_noise_floor_raw(ctx.lm_hp, ctx.loss_scale)
    except Exception:
        noise_floor = 0.0
    try:
        noise_n_eff = float(getattr(state, "acceptance_noise_n_eff", None))
        if (not math.isfinite(noise_n_eff)) or noise_n_eff <= 0.0:
            noise_n_eff = None
    except Exception:
        try:
            noise_n_eff = float(getattr(ctx, "acceptance_noise_n_eff", None))
            if (not math.isfinite(noise_n_eff)) or noise_n_eff <= 0.0:
                noise_n_eff = None
        except Exception:
            noise_n_eff = None
    floor = _stageB_completion_loss_floor(ctx)
    following_min = _min_following_candidate_free_params(
        following_candidates,
        candidate_min_free_params_fn,
    )
    following_s = "none" if following_min is None else str(int(following_min))
    generic_s = (
        ", generic=unpromoted-with-following"
        if bool(getattr(state, "generic_approximant_unpromoted", False))
        and following_candidates
        else ""
    )
    return (
        f"NN={_stageB_state_num_nn_atoms(state)}, "
        f"params={_stageB_state_num_params(state)}, "
        f"following_min_params={following_s}, "
        f"loss={loss:.3e}, floor={floor:.3e}, noise_floor={float(noise_floor):.3e}, "
        f"noise_n_eff={noise_n_eff if noise_n_eff is not None else 'none'}"
        f"{generic_s}"
    )


def _skip_post_accept_polish_for_terminal_state(ctx: "StageBContext") -> bool:
    """Avoid opportunistic SymPy polish when Stage B is already terminal.

    Fully analytic Stage-B polish is useful cleanup, but it runs before the
    outer loop can observe the no-NN terminal state.  For trig/rational forms
    this can hand a large expression to SymPy and block completion even though
    no Stage-B NN targets remain and the accepted expression is already below
    the meaningful loss floor.  Final Pareto polish/reporting can still do the
    slower global cleanup after Stage B has returned.
    """

    try:
        if _are_we_done_yet(ctx):
            ctx.log(
                "[Stage B polish] skipped: accepted state is already terminal; "
                f"deferring cleanup ({_are_we_done_yet_reason(ctx)})."
            )
            return True
    except Exception:
        pass
    return False


def _count_effective_params(
    model: torch.nn.Module,
    root: "Node",
    x_data: torch.Tensor,
    rel_tol: float = 1e-6,
) -> int:
    """Count trainable parameters that contribute meaningfully after fitting.

    For polynomial / rational-polynomial leaves, each raw coefficient is
    weighted by the median scale of its monomial evaluated on *x_data*,
    producing the "normalised-data" contribution.  A coefficient is counted
    as active when its normalised contribution exceeds *rel_tol* times the
    largest contribution in that leaf.  This avoids over-counting parameters
    that are nominally non-zero but multiply a monomial whose scale is tiny
    (or vice-versa).

    For non-polynomial leaves the nominal parameter count is kept.
    """
    from nestynet_sr.sr_core.atoms import (
        PolyLeaf, RPolyLeaf,
        RationalPolyLeaf, RRationalPolyLeaf,
        _eval_monomials,
    )

    atoms = list(collect_all_atoms(root))
    count = 0

    for atom, leaf in zip(atoms, model.leaf):
        core = getattr(leaf, "model", leaf)

        # --- rational polynomial leaves --------------------------------
        if isinstance(core, (RRationalPolyLeaf, RationalPolyLeaf)):
            count += _effective_ratpoly_params(
                core, atom, x_data, rel_tol, _eval_monomials,
            )
            continue

        # --- plain polynomial leaves -----------------------------------
        if isinstance(core, (PolyLeaf, RPolyLeaf)):
            count += _effective_poly_params(
                core, atom, x_data, rel_tol, _eval_monomials,
            )
            continue

        # --- everything else: count all trainable params ---------------
        for p in leaf.parameters():
            if p.requires_grad:
                count += p.numel()

    return max(count, 1)


def _leaf_z_data(
    atom: "AtomNode", x_data: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a leaf atom's inputs on *x_data*, returning (B, n_in)."""
    try:
        z, _, _ = eval_inputs(atom, x_data, need_grad=False, need_hess=False)
    except Exception:
        # Fallback: use raw columns from var_idxs
        idx = list(getattr(atom, "var_idxs", ()) or ())
        z = x_data[:, idx] if idx else x_data[:, :0]
    return z



def _effective_ratpoly_params(core, atom, x_data, rel_tol, _eval_monomials):
    """Count effective parameters in a rational-polynomial leaf."""
    z = _leaf_z_data(atom, x_data)

    # Numerator -------------------------------------------------------
    # RRationalPolyLeaf has full_coeffs_num() + exps_num_full;
    # plain RationalPolyLeaf has coeffs_num + exps_num.
    full_fn = getattr(core, "full_coeffs_num", None)
    if full_fn is not None:
        full_num = full_fn().detach()
        exps_num = core.exps_num_full
    else:
        full_num = core.coeffs_num.detach()
        exps_num = core.exps_num

    # First pass: compute contributions for BOTH num and den to get a
    # single reference scale (they share the same output space).
    Phi_num = _eval_monomials(z, exps_num)
    contrib_num = (full_num.unsqueeze(0) * Phi_num).abs().median(dim=0).values

    coeffs_den = core.coeffs_den.detach()
    Phi_den = _eval_monomials(z, core.exps_den)
    contrib_den = (coeffs_den.unsqueeze(0) * Phi_den).abs().median(dim=0).values

    peak = max(contrib_num.max().item(), contrib_den.max().item())
    if peak < 1e-30:
        return 0

    threshold = rel_tol * peak

    # Count only *free* numerator coefficients (the leading monomial
    # is fixed to 1.0 and is not a trainable parameter).
    free_pos = getattr(core, "free_pos_num", None)
    if free_pos is not None:
        free_contrib = contrib_num[free_pos.to(device=contrib_num.device)]
        n_num = int((free_contrib > threshold).sum().item())
    else:
        n_num = int((contrib_num > threshold).sum().item())

    n_den = int((contrib_den > threshold).sum().item())
    return n_num + n_den


def _effective_poly_params(core, atom, x_data, rel_tol, _eval_monomials):
    """Count effective parameters in a polynomial leaf."""
    z = _leaf_z_data(atom, x_data)
    # RPolyLeaf has full_coeffs() + exps_full; plain PolyLeaf has coeffs + exps
    full_c_fn = getattr(core, "full_coeffs", None)
    if full_c_fn is not None:
        full_c = full_c_fn().detach()
        exps = core.exps_full
    else:
        full_c = core.coeffs.detach()
        exps = core.exps

    Phi = _eval_monomials(z, exps)
    contrib = (full_c.unsqueeze(0) * Phi).abs().median(dim=0).values
    peak = contrib.max().item()
    if peak < 1e-30:
        return 0

    # Count only free (trainable) coefficients
    free_pos = getattr(core, "free_pos", None)
    if free_pos is not None:
        free_contrib = contrib[free_pos.to(device=contrib.device)]
        return int((free_contrib > rel_tol * peak).sum().item())
    return int((contrib > rel_tol * peak).sum().item())


def _unwrap_leaf_core(leaf_mod: torch.nn.Module) -> torch.nn.Module:
    """Return the analytic core for a leaf wrapper."""
    return getattr(leaf_mod, "core", getattr(leaf_mod, "model", leaf_mod))


def _filter_reuse_map(
    reuse: Optional[Dict[str, torch.nn.Module]],
    blocked_tags: Set[str],
) -> Optional[Dict[str, torch.nn.Module]]:
    """Drop blocked tags from a reuse map before rebuilding a candidate."""
    if reuse is None or not blocked_tags:
        return reuse
    return {k: v for k, v in reuse.items() if str(k) not in blocked_tags}


def _find_ratpoly_scale_pair(root: Node, scale_tag: str) -> Optional[Tuple[MulNode, AtomNode, AtomNode]]:
    """Locate the ``scale * rratpoly`` subtree for a ratpoly_1d candidate."""
    if isinstance(root, MulNode):
        for scale_atom, rat_atom in ((root.left, root.right), (root.right, root.left)):
            if not isinstance(scale_atom, AtomNode) or not isinstance(rat_atom, AtomNode):
                continue
            if str(getattr(scale_atom, "kind", "")).lower() != "scale":
                continue
            if str(getattr(rat_atom, "kind", "")).lower() != "rratpoly":
                continue
            if str(getattr(scale_atom, "tag", "")) != str(scale_tag):
                continue
            if str((getattr(rat_atom, "kwargs", {}) or {}).get("_mul_scale_tag", "")) != str(scale_tag):
                continue
            return root, scale_atom, rat_atom
        found = _find_ratpoly_scale_pair(root.left, scale_tag)
        if found is not None:
            return found
        return _find_ratpoly_scale_pair(root.right, scale_tag)
    if isinstance(root, AddNode):
        found = _find_ratpoly_scale_pair(root.left, scale_tag)
        if found is not None:
            return found
        return _find_ratpoly_scale_pair(root.right, scale_tag)
    if isinstance(root, PowNode):
        return _find_ratpoly_scale_pair(root.base, scale_tag)
    if isinstance(root, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        inner = getattr(root, "arg", getattr(root, "child", None))
        if inner is not None:
            return _find_ratpoly_scale_pair(inner, scale_tag)
    return None


def _ratpoly_degree_bands(
    exps: torch.Tensor,
    *,
    exclude_degree: Optional[int] = None,
) -> List[int]:
    """Return removable exact-degree bands, preserving the highest degree."""
    if exps.ndim != 2 or int(exps.shape[0]) <= 0:
        return []
    degs = exps.sum(dim=1).to(dtype=torch.int64)
    max_deg = int(degs.max().item())
    bands = sorted({int(d.item()) for d in degs if int(d.item()) < max_deg})
    if exclude_degree is not None:
        bands = [d for d in bands if int(d) != int(exclude_degree)]
    return bands


def _ratpoly_support_degrees(exps: torch.Tensor) -> List[int]:
    """Return sorted total-degree support for a 1D rational polynomial tensor."""
    if exps.ndim != 2 or int(exps.shape[0]) <= 0:
        return []
    degs = exps.sum(dim=1).to(dtype=torch.int64)
    return sorted({int(d.item()) for d in degs})


def _format_ratpoly_support(exps_num: torch.Tensor, exps_den: torch.Tensor) -> str:
    """Format numerator/denominator degree support for trim diagnostics."""
    num_deg = _ratpoly_support_degrees(exps_num)
    den_deg = _ratpoly_support_degrees(exps_den)
    return f"num={num_deg}, den={den_deg}"


def _ratpoly_den_pivot_degree(
    exps_den: torch.Tensor,
    coeffs_den: torch.Tensor,
) -> Optional[int]:
    """Choose the denominator pivot degree that should remain fixed."""
    if exps_den.ndim != 2 or coeffs_den.ndim != 1 or int(exps_den.shape[0]) != int(coeffs_den.numel()):
        return None
    try:
        from ..rational_sparsify import RationalSparsifyConfig, _choose_denominator_pivot

        pivot_idx = _choose_denominator_pivot(
            coeffs_den.detach().to(dtype=torch.float64, device=torch.device("cpu")),
            RationalSparsifyConfig(),
        )
    except Exception:
        pivot_idx = None
    if pivot_idx is None or int(pivot_idx) < 0 or int(pivot_idx) >= int(exps_den.shape[0]):
        return None
    return int(exps_den[int(pivot_idx)].sum().item())


def _is_ratpoly_candidate(cand: "Candidate") -> bool:
    """Return True for trim-capable ratpoly candidates."""
    family = candidate_pattern_name(cand)
    if family in {"ratpoly", "ratpoly_1d"}:
        return True
    meta = cand.meta if isinstance(getattr(cand, "meta", None), dict) else {}
    return str(meta.get("pattern_family", "")) in {"ratpoly", "ratpoly_1d"}


def _ratpoly_exps_key(exps: Optional[torch.Tensor]) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """Canonical hashable key for an exponent tensor."""
    if exps is None:
        return None
    try:
        exps_cpu = exps.detach().cpu().to(dtype=torch.int64)
    except Exception:
        return None
    if exps_cpu.ndim != 2:
        return None
    return tuple(tuple(int(v) for v in row.tolist()) for row in exps_cpu)


def _ratpoly_support_signature_exact(
    *,
    target_sig: int,
    leaf_kind: str,
    exps_num: torch.Tensor,
    exps_den: torch.Tensor,
) -> Tuple[int, ...]:
    """Canonical exact-support signature for arbitrary-dimensional ratpolys."""
    def _flat(exps_t: torch.Tensor) -> Tuple[int, ...]:
        if exps_t.ndim != 2:
            return (0, 0)
        rows = int(exps_t.shape[0])
        dim_local = int(exps_t.shape[1])
        return (rows, dim_local, *(int(v) for v in exps_t.reshape(-1).tolist()))

    kind_code = 1 if str(leaf_kind).lower() == "rratpoly" else 0
    return (int(target_sig), int(kind_code), *_flat(exps_num), -1, *_flat(exps_den))


def _refresh_ratpoly_trim_unit_certificate(
    meta: Dict[str, Any],
    *,
    leaf_kind: str,
    exps_num: torch.Tensor,
    exps_den: torch.Tensor,
) -> None:
    """Re-certify exact supports after degree trimming, never retaining a stale certificate."""
    certificate = meta.get("coefficient_unit_certificate")
    if not bool(meta.get("unit_support_planned", False)) and certificate is None:
        return

    def _remove_stale(*, code: str, reason: str) -> None:
        meta.pop("unit_support_planned", None)
        meta.pop("coefficient_unit_certificate", None)
        meta.pop("unit_support_complexity", None)
        meta["unit_support_certificate_refresh"] = {
            "status": "removed",
            "code": str(code),
            "reason": str(reason),
        }

    if not isinstance(certificate, dict):
        _remove_stale(
            code="missing_certificate_payload",
            reason="degree trimming could not re-certify a missing unit certificate",
        )
        return

    target_dim = certificate.get("target_dim")
    input_dims = certificate.get("input_dims")
    if target_dim is None or not isinstance(input_dims, (list, tuple)) or not input_dims:
        _remove_stale(
            code="missing_certificate_dimensions",
            reason=(
                "degree trimming removed a legacy unit certificate that did not "
                "record the target and input dimensions needed for re-certification"
            ),
        )
        return

    try:
        from nestynet_sr.sr_core.coefficient_units import (
            solve_rational_coefficient_gauge,
        )

        numerator_pivot = None
        if str(leaf_kind).lower() == "rratpoly":
            totals = exps_num.sum(dim=1)
            max_degree = int(totals.max().item())
            pivot_positions = (totals == max_degree).nonzero(as_tuple=False).view(-1)
            numerator_pivot = int(pivot_positions[-1].item())
        solution = solve_rational_coefficient_gauge(
            target_dim=target_dim,
            input_dims=input_dims,
            numerator_exponents=exps_num.tolist(),
            denominator_exponents=exps_den.tolist(),
            numerator_pivot=numerator_pivot,
            coefficient_policy=str(
                certificate.get("coefficient_policy", "free_const_only")
            ),
        )
    except Exception as exc:
        _remove_stale(
            code="certificate_refresh_error",
            reason=f"degree-trim unit re-certification raised {type(exc).__name__}: {exc}",
        )
        return

    if not solution.ok:
        _remove_stale(code=solution.code, reason=solution.reason)
        return

    meta["unit_support_planned"] = True
    meta["coefficient_unit_certificate"] = solution.to_dict()
    if "unit_support_complexity" in meta:
        meta["unit_support_complexity"] = int(exps_num.shape[0] + exps_den.shape[0])
    meta["unit_support_certificate_refresh"] = {
        "status": "recomputed",
        "code": solution.code,
        "reason": "exact support changed during degree trimming",
    }


def _ratpoly_num_pivot_degree(exps_num: torch.Tensor, lead_pos: Optional[int]) -> Optional[int]:
    """Return the numerator degree band containing the fixed reduced pivot."""
    if exps_num.ndim != 2 or int(exps_num.shape[0]) <= 0:
        return None
    if lead_pos is None:
        return None
    idx = int(lead_pos)
    if idx < 0 or idx >= int(exps_num.shape[0]):
        return None
    return int(exps_num[idx].sum().item())


def _lookup_rratpoly_trim_target(
    state: "StageBState",
    scale_tag: str,
) -> Optional[Tuple[AtomNode, AtomNode, torch.nn.Module, Optional[torch.nn.Module]]]:
    """Return the current ratpoly atom/core pair and its tagged scale core."""
    atoms = list(collect_all_atoms(state.root))
    leaves = list(state.model.leaf)
    rat_atom = None
    scale_atom = None
    rat_core = None
    scale_core = None

    for atom, leaf_mod in zip(atoms, leaves):
        if not isinstance(atom, AtomNode):
            continue
        core = _unwrap_leaf_core(leaf_mod)
        kind = str(getattr(atom, "kind", "")).lower()
        if kind == "scale" and str(getattr(atom, "tag", "")) == str(scale_tag):
            scale_atom = atom
            scale_core = core
        elif (
            kind == "rratpoly"
            and str((getattr(atom, "kwargs", {}) or {}).get("_mul_scale_tag", "")) == str(scale_tag)
        ):
            rat_atom = atom
            rat_core = core

    if rat_atom is None or rat_core is None or scale_atom is None:
        return None
    return rat_atom, scale_atom, rat_core, scale_core


def _lookup_ratpoly_trim_target(
    state: "StageBState",
    cand: "Candidate",
) -> Optional[Tuple[AtomNode, Optional[AtomNode], torch.nn.Module, Optional[torch.nn.Module], str]]:
    """Locate the live ratpoly leaf for a fitted ratpoly/rratpoly candidate."""
    meta = cand.meta if isinstance(getattr(cand, "meta", None), dict) else {}
    leaf_kind = str(meta.get("leaf_kind", "rratpoly" if meta.get("ratpoly_scale_tag") else "ratpoly")).lower()

    if leaf_kind == "rratpoly":
        scale_tag = meta.get("ratpoly_scale_tag")
        if not scale_tag:
            return None
        found = _lookup_rratpoly_trim_target(state, str(scale_tag))
        if found is None:
            return None
        rat_atom, scale_atom, rat_core, scale_core = found
        return rat_atom, scale_atom, rat_core, scale_core, leaf_kind

    target_tag = meta.get("ratpoly_target_tag")
    target_var_idxs = tuple(int(i) for i in meta.get("ratpoly_var_idxs", ()) or ())
    exps_num_key = meta.get("ratpoly_exps_num_key")
    exps_den_key = meta.get("ratpoly_exps_den_key")

    atoms = list(collect_all_atoms(state.root))
    leaves = list(state.model.leaf)
    for atom, leaf_mod in zip(atoms, leaves):
        if not isinstance(atom, AtomNode):
            continue
        if str(getattr(atom, "kind", "")).lower() != "ratpoly":
            continue
        if target_tag is not None and str(getattr(atom, "tag", "")) != str(target_tag):
            continue
        if target_var_idxs and tuple(int(i) for i in atom.var_idxs) != target_var_idxs:
            continue
        core = _unwrap_leaf_core(leaf_mod)
        try:
            num_key = _ratpoly_exps_key(core.exps_num.detach().cpu())
            den_key = _ratpoly_exps_key(core.exps_den.detach().cpu())
        except Exception:
            continue
        if exps_num_key is not None and num_key != exps_num_key:
            continue
        if exps_den_key is not None and den_key != exps_den_key:
            continue
        return atom, None, core, None, leaf_kind

    return None


def _build_rratpoly_degree_trim_candidate(
    state: "StageBState",
    cand: "Candidate",
    *,
    branch: str,
    degree: int,
) -> Optional["Candidate"]:
    """Build a masked exact-degree ratpoly-family candidate from a fitted state."""
    if not _is_ratpoly_candidate(cand):
        return None
    meta = cand.meta if isinstance(getattr(cand, "meta", None), dict) else {}
    lookup = _lookup_ratpoly_trim_target(state, cand)
    if lookup is None:
        return None
    rat_atom, scale_atom, rat_core, scale_core, leaf_kind = lookup

    from nestynet_sr.sr_core.atoms import RRationalPolyLeaf, RationalPolyLeaf
    from ..candidate_builders import _replace_node

    if not isinstance(rat_core, (RRationalPolyLeaf, RationalPolyLeaf)):
        return None

    if isinstance(rat_core, RRationalPolyLeaf):
        exps_num_old = rat_core.exps_num_full.detach().cpu().to(dtype=torch.int64)
        full_num_old = rat_core.full_coeffs_num().detach().cpu().to(dtype=torch.float64)
    else:
        exps_num_old = rat_core.exps_num.detach().cpu().to(dtype=torch.int64)
        full_num_old = rat_core.coeffs_num.detach().cpu().to(dtype=torch.float64)
    exps_den_old = rat_core.exps_den.detach().cpu().to(dtype=torch.int64)
    coeffs_den_old = rat_core.coeffs_den.detach().cpu().to(dtype=torch.float64)

    if branch == "num":
        degs = exps_num_old.sum(dim=1).to(dtype=torch.int64)
        keep_mask = degs != int(degree)
        if int(keep_mask.sum().item()) <= 0:
            return None
        exps_num_new = exps_num_old[keep_mask].clone()
        exps_den_new = exps_den_old.clone()
    elif branch == "den":
        degs = exps_den_old.sum(dim=1).to(dtype=torch.int64)
        keep_mask = degs != int(degree)
        if int(keep_mask.sum().item()) <= 0:
            return None
        exps_num_new = exps_num_old.clone()
        exps_den_new = exps_den_old[keep_mask].clone()
    else:
        return None

    if int(exps_num_new.shape[0]) <= 0 or int(exps_den_new.shape[0]) <= 0:
        return None

    deg_num_new = int(exps_num_new.sum(dim=1).max().item())
    deg_den_new = int(exps_den_new.sum(dim=1).max().item())

    rat_kwargs = dict(getattr(rat_atom, "kwargs", {}) or {})
    rat_kwargs["deg_num"] = deg_num_new
    rat_kwargs["deg_den"] = deg_den_new
    rat_kwargs["exps_num_override"] = [
        [int(v) for v in row]
        for row in exps_num_new.tolist()
    ]
    rat_kwargs["exps_den_override"] = [
        [int(v) for v in row]
        for row in exps_den_new.tolist()
    ]
    if scale_atom is not None:
        rat_kwargs["_mul_scale_tag"] = str(getattr(scale_atom, "tag", ""))
    else:
        rat_kwargs.pop("_mul_scale_tag", None)

    new_rat_atom = AtomNode(
        kind=str(getattr(rat_atom, "kind", leaf_kind)),
        var_idxs=rat_atom.var_idxs,
        kwargs=rat_kwargs,
        tag=rat_atom.tag,
        inputs=clone_inputs(rat_atom),
    )
    new_root = _replace_node(state.root, rat_atom, new_rat_atom)

    old_num_map = {
        tuple(int(v) for v in row.tolist()): float(coeff.item())
        for row, coeff in zip(exps_num_old, full_num_old)
    }
    old_den_map = {
        tuple(int(v) for v in row.tolist()): float(coeff.item())
        for row, coeff in zip(exps_den_old, coeffs_den_old)
    }
    scale_value_old = None
    if scale_core is not None and hasattr(scale_core, "value"):
        scale_value_old = float(scale_core.value.detach().cpu().item())

    target_tag = getattr(rat_atom, "tag", None)

    def _make_init_fn(
        *,
        _leaf_kind: str = str(leaf_kind),
        _scale_tag: str = str(getattr(scale_atom, "tag", "")) if scale_atom is not None else "",
        _target_tag: Optional[str] = str(target_tag) if target_tag is not None else None,
        _exps_num_new: torch.Tensor = exps_num_new.clone(),
        _exps_den_new: torch.Tensor = exps_den_new.clone(),
        _old_num_map: Dict[Tuple[int, ...], float] = dict(old_num_map),
        _old_den_map: Dict[Tuple[int, ...], float] = dict(old_den_map),
        _scale_value_old: Optional[float] = scale_value_old,
        _branch: str = str(branch),
        _degree: int = int(degree),
    ):
        def _init_fn(root_inner: Node, model_inner: torch.nn.Module):
            from nestynet_sr.sr_core.atoms import RRationalPolyLeaf, RationalPolyLeaf

            rat_core_new = None
            scale_core_new = None
            atoms_inner = list(collect_all_atoms(root_inner))
            leaves_inner = list(model_inner.leaf)
            for atom_i, leaf_mod in zip(atoms_inner, leaves_inner):
                if not isinstance(atom_i, AtomNode):
                    continue
                core_i = _unwrap_leaf_core(leaf_mod)
                kind_i = str(getattr(atom_i, "kind", "")).lower()
                if (
                    kind_i == _leaf_kind
                    and isinstance(core_i, RRationalPolyLeaf if _leaf_kind == "rratpoly" else RationalPolyLeaf)
                    and (_leaf_kind != "rratpoly" or str((getattr(atom_i, "kwargs", {}) or {}).get("_mul_scale_tag", "")) == _scale_tag)
                    and (_target_tag is None or str(getattr(atom_i, "tag", "")) == _target_tag)
                    and torch.equal(
                        (core_i.exps_num_full if _leaf_kind == "rratpoly" else core_i.exps_num).detach().cpu(),
                        _exps_num_new,
                    )
                    and torch.equal(core_i.exps_den.detach().cpu(), _exps_den_new)
                ):
                    rat_core_new = core_i
                elif _leaf_kind == "rratpoly" and kind_i == "scale" and str(getattr(atom_i, "tag", "")) == _scale_tag:
                    scale_core_new = core_i

            if rat_core_new is None:
                print(
                    f"[Stage B ratpoly_trim] No {_leaf_kind} leaf found for {_scale_tag or _target_tag} "
                    f"({_branch} deg={_degree})"
                )
                return

            new_full_coeffs = []
            for row in _exps_num_new:
                key = tuple(int(v) for v in row.tolist())
                new_full_coeffs.append(float(_old_num_map.get(key, 0.0)))
            new_full_coeffs_t = torch.tensor(
                new_full_coeffs,
                dtype=rat_core_new.coeffs_num.dtype,
                device=rat_core_new.coeffs_num.device,
            )
            lead_coeff = 1.0
            if _leaf_kind == "rratpoly":
                if int(new_full_coeffs_t.numel()) <= int(rat_core_new.lead_pos_num):
                    return
                lead_coeff = float(new_full_coeffs_t[int(rat_core_new.lead_pos_num)].item())
                if abs(lead_coeff) < 1e-30:
                    print(
                        f"[Stage B ratpoly_trim] Lead vanished for {_scale_tag} "
                        f"({_branch} deg={_degree})"
                    )
                    return

            new_den_coeffs = []
            for row in _exps_den_new:
                key = tuple(int(v) for v in row.tolist())
                new_den_coeffs.append(float(_old_den_map.get(key, 0.0)))
            new_den_coeffs_t = torch.tensor(
                new_den_coeffs,
                dtype=rat_core_new.coeffs_den.dtype,
                device=rat_core_new.coeffs_den.device,
            )

            with torch.no_grad():
                if _leaf_kind == "rratpoly":
                    if rat_core_new.free_pos_num.numel() > 0:
                        idx = rat_core_new.free_pos_num.to(device=new_full_coeffs_t.device)
                        free_vals = new_full_coeffs_t[idx] / lead_coeff
                        rat_core_new.coeffs_num.copy_(free_vals)
                else:
                    if int(new_full_coeffs_t.numel()) == int(rat_core_new.coeffs_num.numel()):
                        rat_core_new.coeffs_num.copy_(new_full_coeffs_t)
                if int(new_den_coeffs_t.numel()) == int(rat_core_new.coeffs_den.numel()):
                    rat_core_new.coeffs_den.copy_(new_den_coeffs_t)
                if (
                    _leaf_kind == "rratpoly"
                    and scale_core_new is not None
                    and hasattr(scale_core_new, "value")
                    and _scale_value_old is not None
                ):
                    scale_core_new.value.copy_(
                        torch.as_tensor(
                            float(_scale_value_old * lead_coeff),
                            dtype=scale_core_new.value.dtype,
                            device=scale_core_new.value.device,
                        )
                    )

        return _init_fn

    new_meta = dict(meta)
    new_meta["ratpoly_trim_branch"] = str(branch)
    new_meta["ratpoly_trim_degree"] = int(degree)
    target_sig = int(new_meta.get("ratpoly_target_sig", 0) or 0)
    trim_signature = _ratpoly_support_signature_exact(
        target_sig=target_sig,
        leaf_kind=leaf_kind,
        exps_num=exps_num_new,
        exps_den=exps_den_new,
    )
    new_meta["signature"] = trim_signature
    new_meta["leaf_kind"] = str(leaf_kind)
    new_meta["ratpoly_exps_num_key"] = _ratpoly_exps_key(exps_num_new)
    new_meta["ratpoly_exps_den_key"] = _ratpoly_exps_key(exps_den_new)
    _refresh_ratpoly_trim_unit_certificate(
        new_meta,
        leaf_kind=leaf_kind,
        exps_num=exps_num_new,
        exps_den=exps_den_new,
    )
    blocked = set(str(t) for t in new_meta.get("reuse_blacklist_tags", []) if t is not None)
    if target_tag is not None:
        blocked.add(str(target_tag))
    new_meta["reuse_blacklist_tags"] = sorted(blocked)

    return Candidate(
        label=f"{cand.label}/trim_{branch}{int(degree)}",
        root=new_root,
        init_fn=_make_init_fn(),
        meta=new_meta,
        signature=trim_signature,
    )


def _ast_node_to_tuple(node) -> tuple:
    """Convert AST node to canonical hashable tuple representation.

    Handles the recursive tree structure of AST nodes (MulNode, AddNode, etc.)
    by converting each node to a tuple of (type_name, *children_or_values).
    """
    cls_name = node.__class__.__name__
    if cls_name == "AtomNode":
        return ("AtomNode", str(node.kind), tuple(int(i) for i in node.var_idxs))
    elif cls_name in ("MulNode", "AddNode"):
        return (cls_name, _ast_node_to_tuple(node.left), _ast_node_to_tuple(node.right))
    elif cls_name == "PowNode":
        return ("PowNode", _ast_node_to_tuple(node.base), float(node.exponent))
    elif cls_name in ("LogNode", "ExpNode", "SinNode", "CosNode", "AsinNode", "AcosNode", "AtanNode"):
        return (cls_name, _ast_node_to_tuple(node.arg))
    else:
        # Fallback: use string repr for unknown node types
        return ("Unknown", repr(node))


def _target_arity(node: Node) -> int:
    """Like effective_arity but works for any Node (AddNode, MulNode, etc.)."""
    if isinstance(node, AtomNode):
        return effective_arity(node)
    return len(set(_collect_var_idxs_from_node(node)))


def atom_content_hash(atom: AtomNode) -> int:
    """
    Compute content-based hash for an atom node.

    Based on atom kind, var_idxs, and kwargs (NOT object identity).
    This allows detecting logically identical atoms even after AST rewrites
    create new node objects.

    Args:
        atom: AtomNode to hash

    Returns:
        Hash value based on content
    """

    def _stable_sort_key(v):
        return (type(v).__qualname__, repr(v))

    def _make_hashable(v):
        """Canonicalize the JSON-like metadata admitted by AtomNode kwargs."""
        # Handle AST nodes (MulNode, AddNode, AtomNode, etc.)
        # These appear in compound atom kwargs like 'input_expr'
        if hasattr(v, "__class__") and v.__class__.__name__ in (
            "MulNode",
            "AddNode",
            "PowNode",
            "LogNode",
            "ExpNode",
            "SinNode",
            "CosNode",
            "AsinNode",
            "AcosNode",
            "AtanNode",
            "AtomNode",
        ):
            return _ast_node_to_tuple(v)
        if isinstance(v, Mapping):
            items = [
                (_make_hashable(key), _make_hashable(value))
                for key, value in v.items()
            ]
            return ("mapping", tuple(sorted(items, key=lambda item: _stable_sort_key(item[0]))))
        if isinstance(v, list):
            return ("list", tuple(_make_hashable(x) for x in v))
        if isinstance(v, tuple):
            return ("tuple", tuple(_make_hashable(x) for x in v))
        if isinstance(v, (set, frozenset)):
            values = [_make_hashable(x) for x in v]
            return ("set", tuple(sorted(values, key=_stable_sort_key)))
        if v is None or isinstance(v, (str, bytes, bool, Number)):
            return v
        raise TypeError(
            "unsupported AtomNode kwarg value for content hashing: "
            f"{type(v).__qualname__}"
        )
    kwargs_items = ()
    if atom.kwargs:
        kw = dict(atom.kwargs)
        # Canonicalize: remove legacy compound keys, always hash via inputs.
        kw.pop('input_expr', None)
        kw.pop('extra_var_idxs', None)
        kw.pop('compound', None)
        kw['_inputs_hash'] = tuple(_ast_node_to_tuple(inp) for inp in get_input_exprs(atom))
        kwargs_items = tuple(sorted((k, _make_hashable(v)) for k, v in kw.items()))

    return hash(
        (
            str(atom.kind),
            tuple(int(v) for v in atom.var_idxs),
            kwargs_items,
        )
    )


def _is_structural_candidate(cand: Candidate) -> bool:
    """
    Determine if a candidate represents a structural rewrite.

    Structural rewrites introduce or change the decomposition skeleton:
    - New Add/Mul splits (counterterm_mul_split)
    - Partitions and separability (subtree_separability, nn_leaf_separability)
    - Outer transforms (outer_log_add, outer_sqrt_mul, etc.)
    - Polynomial splits (poly_split)

    Cosmetic rewrites just compress leaves without changing structure:
    - Trig compression (trig_comp, planck)
    - Exponential/rational forms (exp_rat, pure_exp_branch)
    - Polynomial fits (quad_poly, sqrt_poly)

    Args:
        cand: Candidate to check

    Returns:
        True if structural, False if cosmetic
    """
    # First check the explicit meta flag
    if cand.meta and cand.meta.get("structural", False):
        return True
    # Fallback: check label against known structural patterns
    if cand.label in STRUCTURAL_LABELS:
        return True
    return any(cand.label.startswith(p) for p in STRUCTURAL_LABEL_PREFIXES)


def _is_separability_candidate(cand: Candidate) -> bool:
    """True if this candidate decomposes the NN into sub-problems
    (separability rewrites that unlock further iterations)."""
    if cand.meta and cand.meta.get("separability_like", False):
        return True
    return cand.label in SEPARABILITY_LABELS


def _nn_multivar_complexity(root: Node) -> Tuple[int, int, int]:
    """Backward-compatible wrapper for the shared NN complexity proxy."""
    return _shared_nn_multivar_complexity(root)


def _compute_nn_metrics(root: Node) -> Tuple[int, int, int]:
    """
    Compute NN atom metrics for reporting and stopping conditions.

    Args:
        root: AST root node

    Returns:
        Tuple of (num_nn_atoms, num_multivar_nn_atoms, max_nn_arity):
        - num_nn_atoms: Total number of NN atoms
        - num_multivar_nn_atoms: Number of multivariate NN atoms (len(var_idxs) > 1)
        - max_nn_arity: Maximum arity across all NN atoms (0 if no NNs)

    Example:
        NN(x0) + NN(x1, x2) + poly(x3) -> (2, 1, 2)
        NN(x0) + NN(x1) + poly(x2) -> (2, 0, 1)
        poly(x0) + poly(x1) -> (0, 0, 0)
    """
    nns = [a for a in collect_nn_atoms(root) if str(a.kind).lower() == "nn"]
    num_nn_atoms = len(nns)
    # Use effective_arity for correct compound atom handling
    num_multivar = sum(1 for a in nns if effective_arity(a) > 1)
    max_arity = max((effective_arity(a) for a in nns), default=0)
    return (num_nn_atoms, num_multivar, max_arity)
