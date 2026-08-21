# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Stage-A result types, acceptance policy, and committee gates."""

from typing import TYPE_CHECKING
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset
from nestynet_sr.sr_core import collect_nn_atoms
from nestynet_sr.sr_core.bridges import AtomNode, clone_ast, effective_arity, has_nontrivial_input
from .coe_witness import CoEWitnessExecutor, coe_witness_execution_metadata, coe_witness_jobs_from_specs, run_threaded_witnesses
from .model_builders import build_composite_ast
from .model_selection import noise_equivalence_tolerance as _noise_equivalence_tolerance
from .training import train_candidate_model
from .y_transforms import precision_for_transform

from ._search_shadow import (
    _apply_fit_link_to_model,
)
from ._search_training import (
    _build_tag_to_leaf_map,
    _eval_yspace_mse,
)
from ._search_proposals import (
    _check_separability_in_input_space,
)

if TYPE_CHECKING:
    from ._search_runtime import (
        run_separability_for_transform,
    )

def _iter_limited_batches(datagen, max_batches: int):
    """Yield up to `max_batches` batches from a DataLoader or datagen callable."""
    iterator = datagen() if callable(datagen) else datagen
    for bi, batch in enumerate(iterator):
        yield batch
        if max_batches is not None and (bi + 1) >= int(max_batches):
            break


def _candidate_metric(candidate_sep):
    """Best-effort extraction of the optional per-candidate metric."""
    if candidate_sep is None:
        return None
    if len(candidate_sep) > 4:
        m = candidate_sep[4]
        if m is None:
            return None
        try:
            return float(m)
        except Exception:
            return None
    return None


def _sep_metric_to_score(metric, precision_hint: float) -> float:
    """Map separability metric (lower-better) to a [0, 1] confidence-like score."""
    try:
        m = float(metric)
    except Exception:
        return 0.0
    if not math.isfinite(m):
        return 0.0
    m = max(0.0, m)
    ref = max(float(precision_hint), 1.0e-12)
    # m == 0 -> 1.0, m == ref -> 0.5, m >> ref -> ~0.0
    return float(1.0 / (1.0 + (m / ref)))


def _is_clean_disjoint_cover(g1, g2, symb_set):
    """True if {g1,g2} is a disjoint cover of symb_set."""
    if not g1 or not g2:
        return False
    s1, s2 = set(g1), set(g2)
    if s1 & s2:
        return False
    return (s1 | s2) == set(symb_set)


def _is_singleton_split(g1, g2):
    """True if either side of the split is a singleton."""
    if not g1 or not g2:
        return False
    s1, s2 = set(g1), set(g2)
    return (len(s1) == 1) or (len(s2) == 1)


def _is_singleton_disjoint_cover(g1, g2, symb_set):
    """True if {g1,g2} is a disjoint cover of symb_set with a singleton side."""
    return _is_clean_disjoint_cover(g1, g2, symb_set) and _is_singleton_split(g1, g2)


def _stageA_best_disjoint_separability_metric(
    model,
    current_ast,
    nn_atoms,
    datagen_train_noshuffle,
    device,
    dtype,
    search_hp,
    lm_hp,
    y_op,
    y_med,
    y_mad,
    y_log_dynamic_range,
    require_singleton: bool = False,
):
    """Cheap quickscan: look for a *very clean* disjoint separability split.

    If `require_singleton` is True, restrict to singleton-vs-rest disjoint covers.

    Returns
    -------
    best_metric : float | None
    best_atom : Node | None
    precision_used : float
    best_is_singleton : bool | None
    """
    # Precision used by separability checks (same for sum/mult in Stage A)
    precision = precision_for_transform(
        y_op=y_op,
        y_med=y_med,
        y_mad=y_mad,
        base_precision=search_hp.precision_derivs_d2y,
    )

    # Mirror the asinh precision adjustment used in Phase B.
    if getattr(lm_hp, "fit_y_link", None) == "asinh":
        import math

        asinh_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)
        base_factor = math.sqrt(1.0 + (asinh_scale / (y_mad + 1e-30)) ** 2)
        dynamic_range_factor = y_log_dynamic_range if y_log_dynamic_range is not None else 2.0
        precision *= base_factor * dynamic_range_factor

    quickscan_batches = int(getattr(search_hp, "stageA_disjoint_sep_quickscan_batches", 1))
    if quickscan_batches <= 0:
        return None, None, precision, None

    best_metric = None
    best_atom = None
    best_is_singleton = None

    tag_to_leaf = _build_tag_to_leaf_map(current_ast, model)

    for atom in nn_atoms:
        if effective_arity(atom) <= 1:
            continue

        leaf = tag_to_leaf.get(getattr(atom, "tag", None))
        if leaf is None:
            continue

        try:
            cand_list, _, _, _ = _check_separability_in_input_space(
                model=model,
                atom=atom,
                leaf=leaf,
                datagen_train=_iter_limited_batches(datagen_train_noshuffle, quickscan_batches),
                device=device,
                dtype=dtype,
                precision_sum=precision,
                precision_mult=precision,
                very_verbose=bool(getattr(search_hp, 'verbose_separabilities', False)),
            )
        except Exception:
            continue

        # Build the set of symbols for disjoint-cover checks.
        # For compound atoms, groups may contain _COMPOUND_Z_TOKEN; these
        # candidates cannot be tested for raw-variable disjointness, so
        # they pass through the filter untouched.
        atom_is_compound = has_nontrivial_input(atom)
        symb_set = set(getattr(atom, "var_idxs", ()))

        for cand in cand_list:
            g1 = cand[1] if len(cand) > 1 else []
            g2 = cand[2] if len(cand) > 2 else []

            # Compound-atom candidates operate in z-space; skip the
            # raw-variable disjoint filter (not meaningful for z-tokens).
            if not atom_is_compound:
                if require_singleton:
                    if not _is_singleton_disjoint_cover(g1, g2, symb_set):
                        continue
                else:
                    if not _is_clean_disjoint_cover(g1, g2, symb_set):
                        continue

            m = _candidate_metric(cand)
            if m is None:
                continue
            try:
                m = float(m)
            except Exception:
                continue

            is_single = _is_singleton_split(g1, g2)
            if best_metric is None or m < best_metric:
                best_metric = m
                best_atom = atom
                best_is_singleton = bool(is_single)

    return best_metric, best_atom, precision, best_is_singleton


@dataclass
class StageAOut:
    """Stage A analysis output for one transform/state evaluation."""

    success: bool
    model: Any
    rest_add: Any
    rest_mult: Any
    candidate_sep_ops: List[bool]
    current_ast: Any
    last_resort_suggested: bool
    full_compound_solved: bool
    y_transform_name: str
    val_loss_base: float
    signals: Dict[str, float]
    split_plans: List[Any]
    move_records: List[Dict[str, Any]]


def _stageA_provisional_move_reason(
    move_kind: str,
    risk_tags: Optional[Iterable[str]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return why an accepted Stage-A move needs downstream confirmation.

    PR-A8 deliberately treats only exploratory moves as provisional.  Destructive
    simplifications (axis deletion/pruning) and terminal closures are handled by
    their own hard gates before commit; marking them provisional after the fact
    would be too late to protect the search tree.
    """

    details = details if isinstance(details, dict) else {}
    tags = {str(t).lower() for t in list(risk_tags or []) if str(t)}
    kind_l = str(move_kind or "").lower()

    if any(tok in kind_l for tok in ("prune", "delete", "projection")) or "destructive_prune" in tags:
        return None
    if "terminal_closure" in tags or "terminal" in kind_l or "closure" in kind_l:
        return None

    full_refit_confirmed = bool(
        details.get("full_refit_confirmed", False)
        or details.get("full_validation_confirmed", False)
    )

    if bool(details.get("coe_provisional_budget_admission", False)):
        return "coe_budgeted_structural_admission_requires_stageB_confirmation"

    if bool(details.get("full_compound", False)):
        return "full_compound_compression_requires_stageB_confirmation"

    old_arity = details.get("old_arity", None)
    new_arity = details.get("new_arity", None)
    desc = details.get("compound_replay_descriptor", None)
    if isinstance(desc, dict):
        cand_desc = desc.get("candidate_descriptor", {})
        if isinstance(cand_desc, dict):
            old_arity = cand_desc.get("old_arity", old_arity)
            new_arity = cand_desc.get("new_arity", new_arity)
    try:
        old_a = int(old_arity)
        new_a = int(new_arity)
        if old_a > 0 and new_a >= old_a and "compound_coordinate" in tags:
            return "same_arity_compound_requires_downstream_confirmation"
    except Exception:
        pass

    if bool(details.get("shadow_requires_payoff", False)) and "compound_coordinate" in tags:
        return "shadow_promoted_compound_requires_downstream_confirmation"

    if not full_refit_confirmed and (
        bool(details.get("soft_monomial_compound", False))
        or "soft_monomial_compound" in tags
    ):
        return "soft_monomial_compound_requires_full_refit_confirmation"

    if not full_refit_confirmed and (
        bool(details.get("coe_scout_replay", False))
        or "coe_scout_replay" in tags
    ):
        return "scout_replay_compound_requires_full_refit_confirmation"

    if not full_refit_confirmed and (
        bool(details.get("visible_buckingham_1d_prefactor", False))
        or "visible_buckingham_1d_prefactor" in tags
    ):
        return "visible_buckingham_prefactor_requires_full_refit_confirmation"

    if not full_refit_confirmed and (
        bool(details.get("additive_shared_response", False))
        or "additive_shared_response" in tags
    ):
        return "additive_shared_response_requires_full_refit_confirmation"

    if not full_refit_confirmed and (
        details.get("null_verified", None) is False
        or bool(details.get("accepted_under_noisy_tie", False))
    ):
        return "noisy_provisional_compound_requires_full_refit_confirmation"

    if (
        "x_transform_active" in tags
        or kind_l.startswith("x_preconditioning_")
        or bool(details.get("x_transform_map"))
    ):
        return "x_transform_branch_requires_downstream_confirmation"

    if "ambiguous_y_transform" in tags or bool(details.get("ambiguous_y_transform", False)):
        return "ambiguous_y_transform_requires_downstream_confirmation"

    return None


def _stageA_provisional_full_refit_failure_status(
    *,
    candidate_loss,
    parent_loss,
    acceptable_loss,
    noise_floor_raw: float = 0.0,
    n_eff: Optional[float] = None,
    k_bad: float = 5.0,
    bad_mult: float = 2.0,
    rel_tol: float = 1.0e-3,
    abs_tol: float = 1.0e-14,
) -> Dict[str, Any]:
    """Classify a failed provisional full refit without accepting the scaffold."""

    def _finite_float(value, default=float("inf")) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return out if math.isfinite(out) else float(default)

    cand = _finite_float(candidate_loss)
    parent = _finite_float(parent_loss)
    acceptable = _finite_float(acceptable_loss)
    noise = max(0.0, _finite_float(noise_floor_raw, default=0.0))
    try:
        n_val = float(n_eff) if n_eff is not None else 0.0
    except Exception:
        n_val = 0.0
    if not math.isfinite(n_val) or n_val <= 0.0:
        n_val = 1.0

    scale = max(parent, cand, acceptable, noise, float(abs_tol))
    tau = max(
        float(abs_tol),
        float(rel_tol) * scale,
        float(noise) * math.sqrt(2.0 / max(1.0, n_val)),
    )
    severe = False
    reasons: List[str] = []
    if cand > acceptable + float(k_bad) * tau:
        severe = True
        reasons.append("candidate_exceeds_acceptable_by_severe_margin")
    if cand > parent + float(k_bad) * tau:
        severe = True
        reasons.append("candidate_worse_than_parent_by_severe_margin")
    if cand > float(bad_mult) * max(parent, acceptable, noise, float(abs_tol)):
        severe = True
        reasons.append("candidate_exceeds_bad_loss_multiplier")
    status = "severe" if severe else "ambiguous"
    return {
        "status": status,
        "decision": "rollback",
        "candidate_loss": cand,
        "parent_loss": parent,
        "acceptable_loss": acceptable,
        "noise_floor_raw": noise,
        "n_eff": n_val,
        "tolerance": tau,
        "k_bad": float(k_bad),
        "bad_mult": float(bad_mult),
        "reasons": reasons or ["candidate_failed_full_refit_confirmation"],
    }


@dataclass(frozen=True)
class SplitPlan:
    """Lightweight split proposal emitted by Stage A analysis."""

    kind: str
    partition: tuple
    score: float
    details: Dict[str, Any]


def _nn_split_signature(ast: Any) -> tuple[int, int, int]:
    """Return a compact complexity signature for unresolved NN atom arities."""
    try:
        atoms = collect_nn_atoms(ast)
    except Exception:
        atoms = []
    arities: list[int] = []
    for atom in atoms:
        try:
            arities.append(max(0, int(effective_arity(atom))))
        except Exception:
            try:
                arities.append(max(0, len(tuple(getattr(atom, "var_idxs", ()) or ()))))
            except Exception:
                arities.append(0)
    if not arities:
        return (0, 0, 0)
    multivar = sum(1 for a in arities if a > 1)
    max_arity = max(arities)
    arity_excess = sum(max(0, a - 1) for a in arities)
    return (multivar, max_arity, arity_excess)


def _is_confirmed_nn_split_simplification(base_ast: Any, cand_ast: Any) -> bool:
    """True when a candidate strictly reduces unresolved multivariate NN arity."""
    return _nn_split_signature(cand_ast) < _nn_split_signature(base_ast)


def _accept_threshold_with_structural_target(
    *,
    base_ast: Any,
    cand_ast: Any,
    accept_threshold: float,
    loss_target_eff: float,
) -> tuple[float, bool]:
    """Raise an accept threshold to target-quality for true NN arity reductions."""
    is_structural = bool(_is_confirmed_nn_split_simplification(base_ast, cand_ast))
    threshold = float(accept_threshold)
    if is_structural:
        threshold = max(threshold, float(loss_target_eff))
    return float(threshold), bool(is_structural)


def _stageA_under_protest_threshold_cap(
    *,
    accept_threshold: float,
    current_val_loss: Optional[float],
    loss_floor: float,
    noise_floor: float = 0.0,
    under_protest: bool = False,
    label: str = "Stage A",
) -> tuple[float, bool]:
    """Do not let under-protest Stage-A moves accept validation regressions.

    A model kept under protest is useful as proposal evidence, but its
    derivatives/scatter are not trustworthy enough to justify the usual
    structure-first loss budget.  In that state candidates must improve the
    current validation loss, except that losses already below the existing
    floor are treated as equivalent.
    """
    threshold = float(accept_threshold)
    if not under_protest:
        return threshold, False

    try:
        current = float(current_val_loss)
    except (TypeError, ValueError):
        return threshold, False
    if not math.isfinite(current) or current < 0.0:
        return threshold, False

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
    if math.isfinite(nfloor) and nfloor > floor:
        floor = nfloor

    cap = max(current, floor) if current <= floor else current
    capped = min(threshold, cap)
    if capped < threshold:
        print(
            f"[Stage A] Under-protest non-regression cap for {label}: "
            f"accept_threshold {threshold:.4e} → {capped:.4e} "
            f"(current val-loss {current:.4e}, floor {floor:.4e})"
        )
        return float(capped), True
    return threshold, False


def _stageA_loss_budget_multiplier(
    *,
    base_loss: Any,
    allowed_loss: Any,
    noise_floor: Any = 0.0,
) -> float:
    """Translate a local structure-first loss cap into an excess-risk ratio."""

    try:
        base = float(base_loss)
        allowed = float(allowed_loss)
        floor = max(0.0, float(noise_floor))
    except Exception:
        return 1.0
    if not (math.isfinite(base) and math.isfinite(allowed) and math.isfinite(floor)):
        return 1.0
    base_excess = max(0.0, base - floor)
    if base_excess <= 0.0:
        return 1.0
    return float(max(1.0, max(0.0, allowed - floor) / base_excess))


def _stageA_budgeted_witness_admission(
    rows: Iterable[dict],
    *,
    base_loss_key: str,
    candidate_loss_key: str,
    budget_multiplier: Any,
    noise_floor: Any = 0.0,
) -> dict:
    """Check the structural excess-risk budget on every witness slice."""

    try:
        budget = float(budget_multiplier)
        floor = max(0.0, float(noise_floor))
    except Exception:
        budget, floor = 1.0, 0.0
    if not math.isfinite(budget) or budget < 1.0:
        budget = 1.0
    if not math.isfinite(floor):
        floor = 0.0

    checked = []
    invalid = 0
    for source in list(rows or []):
        row = dict(source) if isinstance(source, dict) else {"status": "error"}
        if row.get("status") != "success":
            invalid += 1
            continue
        try:
            base = float(row[base_loss_key])
            cand = float(row[candidate_loss_key])
            tol = max(0.0, float(row.get("tolerance", 0.0) or 0.0))
        except Exception:
            invalid += 1
            continue
        if not (
            math.isfinite(base)
            and math.isfinite(cand)
            and math.isfinite(tol)
            and base >= 0.0
            and cand >= 0.0
        ):
            invalid += 1
            continue
        numeric = float(torch.finfo(torch.float64).eps) * max(
            abs(base), abs(cand), abs(floor), 1.0e-300
        )
        scale = max(base - floor, tol, numeric, 1.0e-300)
        cand_excess = max(0.0, cand - floor)
        ceiling = floor + budget * scale
        checked.append(
            {
                "slice_id": row.get("slice_id"),
                "incumbent_excess_scale": float(scale),
                "candidate_excess": float(cand_excess),
                "excess_ratio": float(cand_excess / scale),
                "allowed_loss": float(ceiling),
                "within_budget": bool(cand <= ceiling),
            }
        )
    within = bool(checked and invalid == 0 and all(row["within_budget"] for row in checked))
    ratios = sorted(float(row["excess_ratio"]) for row in checked)
    median_ratio = None
    if ratios:
        mid = len(ratios) // 2
        median_ratio = ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])
    return {
        "enabled": True,
        "budget_multiplier": float(budget),
        "noise_floor": float(floor),
        "evaluated_slices": int(len(checked)),
        "invalid_slices": int(invalid),
        "all_slices_within_budget": bool(within),
        "median_excess_ratio": median_ratio,
        "max_excess_ratio": max(ratios) if ratios else None,
        "rows": checked,
    }


def _stageA_leaf_projection_nonregression_override(
    *,
    base_ast: Any,
    cand_ast: Any,
    base_val_loss: Optional[float],
    cand_val_loss: Optional[float],
    loss_floor: float,
    noise_floor: float,
    base_train_loss: Optional[float],
    cand_train_loss: Optional[float],
    max_train_degradation: float,
    axes_to_drop: Any,
) -> tuple[bool, str]:
    """Allow strict NN input projections when they do not regress validation.

    This is intentionally narrower than the ordinary Stage-A simplification
    budget.  A leaf non-dependency prune is not a new separability hypothesis;
    it is a projection of an existing NN representative onto a subset of its
    current inputs.  If that projection lowers unresolved NN burden and does
    not worsen validation loss, rejecting it in favor of an overlapping gauge
    split is path-dependent and counterproductive.
    """
    if not axes_to_drop:
        return False, "no axes dropped"

    try:
        base_atoms = collect_nn_atoms(base_ast)
        cand_atoms = collect_nn_atoms(cand_ast)
    except Exception as exc:
        return False, f"could not inspect NN atoms ({type(exc).__name__})"
    if len(cand_atoms) > len(base_atoms):
        return False, "candidate increases NN atom count"

    base_sig = _nn_split_signature(base_ast)
    cand_sig = _nn_split_signature(cand_ast)
    if not cand_sig < base_sig:
        return False, f"candidate is not a strict NN projection ({base_sig} -> {cand_sig})"

    try:
        base_v = float(base_val_loss)
        cand_v = float(cand_val_loss)
    except (TypeError, ValueError):
        return False, "missing validation loss"
    if not math.isfinite(base_v) or not math.isfinite(cand_v) or cand_v < 0.0:
        return False, "non-finite validation loss"

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
    if math.isfinite(nfloor) and nfloor > floor:
        floor = nfloor

    val_ok = cand_v <= base_v or (base_v <= floor and cand_v <= floor)
    if not val_ok:
        return False, f"validation regression ({cand_v:.4e} > {base_v:.4e})"

    try:
        cand_t = float(cand_train_loss)
    except (TypeError, ValueError):
        return False, "missing training loss"
    if not math.isfinite(cand_t) or cand_t < 0.0:
        return False, "non-finite training loss"

    train_ok = cand_t <= floor if floor > 0.0 else False
    try:
        base_t = float(base_train_loss)
        if math.isfinite(base_t) and base_t > 0.0:
            train_ok = train_ok or cand_t <= float(max_train_degradation) * base_t
    except (TypeError, ValueError):
        pass
    if not train_ok:
        return False, f"training sanity failed ({cand_t:.4e})"

    return True, (
        f"strict NN input projection non-regression "
        f"val {base_v:.4e}->{cand_v:.4e}, train={cand_t:.4e}, "
        f"signature {base_sig}->{cand_sig}"
    )


def _stageA_ast_raw_var_set(node: Any) -> set[int]:
    """Collect raw variable indices from a full Stage-A AST.

    ``_collect_var_idxs_from_node`` is intentionally scoped to analytic input
    expressions and ignores NN atoms.  Leaf-prune safety needs full-model
    support, including unresolved NN leaves and analytic atom leaves.
    """
    if node is None:
        return set()
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        if kind in ("var", "x", "input"):
            raw = getattr(node, "var_idxs", ()) or ()
        else:
            raw = getattr(node, "raw_var_idxs", None)
            if raw is None:
                raw = getattr(node, "var_idxs", ()) or ()
            if callable(raw):
                raw = raw()
            if not raw:
                raw = getattr(node, "var_idxs", ()) or ()
        try:
            return {int(v) for v in raw}
        except Exception:
            return set()
    out: set[int] = set()
    for attr in ("left", "right", "base", "arg"):
        child = getattr(node, attr, None)
        if child is not None:
            out.update(_stageA_ast_raw_var_set(child))
    return out


def _stageA_leaf_prune_acceptance_gate(
    *,
    base_ast: Any,
    cand_ast: Any,
    axes_to_drop: Any,
    base_val_loss: Optional[float],
    cand_val_loss: Optional[float],
    loss_floor: float,
    noise_floor: float,
    n_eff: Optional[float] = None,
) -> tuple[bool, str]:
    """Extra safety gate for leaf non-dependency pruning.

    Dropping an NN input axis is information-destroying.  Ordinary Stage-A
    simplification budgets can tolerate temporary loss regression for visible
    decompositions, but they are too permissive for deleting a raw variable.

    Policy:
      * If the deleted raw variable still appears elsewhere, require
        non-regression up to statistical/noise equivalence.
      * If the deleted raw variable disappears from the whole AST, require a
        real validation improvement and target-quality loss.  For noiseless
        exact/below-floor cases, allow equivalent losses so exact historical
        behavior is preserved.
    """
    dropped = tuple(sorted({int(a) for a in (axes_to_drop or ())}))
    if not dropped:
        return False, "no axes dropped"

    try:
        base_v = float(base_val_loss)
        cand_v = float(cand_val_loss)
    except (TypeError, ValueError):
        return False, "missing validation loss"
    if (not math.isfinite(base_v)) or (not math.isfinite(cand_v)) or cand_v < 0.0:
        return False, "non-finite validation loss"

    try:
        floor = float(loss_floor)
    except (TypeError, ValueError):
        floor = 0.0
    try:
        nf = float(noise_floor)
    except (TypeError, ValueError):
        nf = 0.0
    if (not math.isfinite(floor)) or floor < 0.0:
        floor = 0.0
    if (not math.isfinite(nf)) or nf < 0.0:
        nf = 0.0

    try:
        n_eff_f = float(n_eff) if n_eff is not None else None
        if n_eff_f is not None and ((not math.isfinite(n_eff_f)) or n_eff_f <= 0.0):
            n_eff_f = None
    except Exception:
        n_eff_f = None

    tol = _noise_equivalence_tolerance(
        cand_v,
        base_v,
        noise_floor=nf,
        n_eff=n_eff_f,
        noise_mult=3.0,
        rel_tol=1.0e-3,
        abs_floor=1.0e-14 * max(1.0, abs(base_v), abs(cand_v)),
    )
    tied_with_base = abs(cand_v - base_v) <= tol
    nonregressing = cand_v <= base_v or tied_with_base

    try:
        base_vars = _stageA_ast_raw_var_set(base_ast)
        cand_vars = _stageA_ast_raw_var_set(cand_ast)
    except Exception as exc:
        return False, f"could not inspect raw variable support ({type(exc).__name__})"

    erased = tuple(v for v in dropped if v in base_vars and v not in cand_vars)
    if not erased:
        if nonregressing:
            return True, (
                f"local axis prune non-regressing within tol={tol:.3e}: "
                f"val {base_v:.4e}->{cand_v:.4e}"
            )
        return False, (
            f"local axis prune regresses validation beyond tol={tol:.3e}: "
            f"val {base_v:.4e}->{cand_v:.4e}"
        )

    good_tol = _noise_equivalence_tolerance(
        cand_v,
        floor,
        noise_floor=nf,
        n_eff=n_eff_f,
        noise_mult=3.0,
        rel_tol=1.0e-3,
        abs_floor=1.0e-14 * max(1.0, abs(cand_v), abs(floor)),
    )
    target_quality = cand_v <= floor or abs(cand_v - floor) <= good_tol
    if not target_quality:
        return False, (
            f"global raw-axis erasure {list(erased)} rejected: candidate loss "
            f"{cand_v:.4e} is not in target-quality regime "
            f"(floor={floor:.4e}, tol={good_tol:.3e}, "
            f"n_eff={n_eff_f if n_eff_f is not None else 'none'})"
        )

    materially_improves = cand_v < base_v and not tied_with_base
    exact_noiseless_equivalent = (
        nf <= 0.0
        and base_v <= floor
        and cand_v <= floor
        and tied_with_base
    )
    if not (materially_improves or exact_noiseless_equivalent):
        return False, (
            f"global raw-axis erasure {list(erased)} rejected: requires validation "
            f"improvement beyond tol={tol:.3e}; val {base_v:.4e}->{cand_v:.4e}"
        )

    if materially_improves:
        return True, (
            f"global raw-axis erasure {list(erased)} accepted: validation improves "
            f"{base_v:.4e}->{cand_v:.4e} and candidate is target-quality "
            f"(floor={floor:.4e})"
        )
    return True, (
        f"global raw-axis erasure {list(erased)} accepted as exact below-floor "
        f"noiseless equivalent: val {base_v:.4e}->{cand_v:.4e}"
    )


def _stageA_snapshot_rng_state() -> dict[str, Any]:
    """Capture RNG state so disposable CoE Stage-A refits do not perturb search."""
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


def _stageA_restore_rng_state(state: Optional[dict[str, Any]]) -> None:
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


def _stageA_destructive_prune_committee_gate(
    *,
    base_ast: Any,
    cand_ast: Any,
    axes_to_drop: Any,
    filepath: Optional[str],
    np_dtype,
    dtype,
    device,
    data_hp,
    model_hp,
    lm_hp,
    leaf_builder,
    y_op,
    y_op_inv,
    dual_layer_used: bool,
    num_segments: int,
) -> tuple[bool, str, dict]:
    """CoE hard gate for destructive Stage-A leaf/axis pruning.

    Committee slices briefly refit the same parent/candidate ASTs and vote on
    original-y loss deltas. They do not run independent Stage-A search.
    """

    mode = str(getattr(lm_hp, "coe_mode", "off") or "off")
    if mode not in {"committee_gated", "reservoir_discovery"}:
        return True, "coe-stageA-prune-gate-disabled", {"enabled": False}
    if not bool(getattr(lm_hp, "coe_stageB_refit_gate", True)):
        return True, "coe-stageA-prune-gate-unsupported: refit gate disabled", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": "refit gate disabled",
        }
    if not filepath:
        return True, "coe-stageA-prune-gate-unsupported: no filepath", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": "no single-dataset filepath configured",
        }
    n_slices = min(
        int(getattr(lm_hp, "coe_num_slices", 0) or 0),
        int(getattr(lm_hp, "coe_stageB_gate_slices", 0) or 0),
    )
    if n_slices <= 0:
        return True, "coe-stageA-prune-gate-unsupported: no slices configured", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": "no CoE Stage-A prune gate slices configured",
        }

    try:
        import numpy as _np

        from nestynet_sr.sr_search.coe_committee import (
            _committee_tolerance,
            _load_dataset_arrays,
            build_slice_specs,
        )
    except Exception as exc:
        return True, f"coe-stageA-prune-gate-unsupported: helpers unavailable ({type(exc).__name__})", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": f"committee helpers unavailable: {type(exc).__name__}: {exc}",
        }

    try:
        X_all, y_all, _cols = _load_dataset_arrays(str(filepath))
    except Exception as exc:
        return True, f"coe-stageA-prune-gate-unsupported: data load failed ({type(exc).__name__})", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": f"slice data load failed: {type(exc).__name__}: {exc}",
        }

    specs = build_slice_specs(
        n_slices=n_slices,
        ndata_train=int(getattr(lm_hp, "coe_ndata_train", 2000) or 2000),
        ndata_val=int(getattr(lm_hp, "coe_ndata_val", 2000) or 2000),
        start_slice=int(getattr(lm_hp, "coe_start_slice", 0) or 0),
        skip_slice_ids=(
            ()
            if getattr(lm_hp, "coe_reference_slice", None) is None
            else (int(getattr(lm_hp, "coe_reference_slice")),)
        ),
        max_rows=int(y_all.shape[0]),
    )
    if not specs:
        return True, "coe-stageA-prune-gate-unsupported: no valid witness slices", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": "no independent CoE witness slices fit inside the dataset",
        }

    try:
        base_vars = _stageA_ast_raw_var_set(base_ast)
        cand_vars = _stageA_ast_raw_var_set(cand_ast)
        dropped = tuple(sorted({int(a) for a in (axes_to_drop or ())}))
        erased = tuple(v for v in dropped if v in base_vars and v not in cand_vars)
    except Exception:
        dropped = tuple(sorted({int(a) for a in (axes_to_drop or ())}))
        erased = ()

    def _slice_loader(start: int, stop: int):
        if start < 0 or stop <= start or stop > int(y_all.shape[0]):
            raise ValueError(
                f"slice rows [{start}, {stop}) outside dataset with {int(y_all.shape[0])} rows"
            )
        x_slice = _np.array(X_all[start:stop], dtype=np_dtype, copy=True)
        y_slice = _np.array(y_all[start:stop], dtype=np_dtype, copy=True).reshape(-1, 1)
        if y_op is not None:
            y_slice = _np.array(y_op(y_slice), dtype=np_dtype, copy=True).reshape(-1, 1)
            if not _np.all(_np.isfinite(y_slice)):
                raise ValueError(f"non-finite y-transform target values on rows [{start}, {stop})")
        xb = torch.as_tensor(x_slice, dtype=dtype)
        yb = torch.as_tensor(y_slice, dtype=dtype)
        batch_size = int(getattr(data_hp, "batch_size", 0) or xb.shape[0])
        batch_size = max(1, min(batch_size, int(xb.shape[0])))
        return DataLoader(
            TensorDataset(xb, yb),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    def _build_model(ast_node):
        model_i, _, ast_i = build_composite_ast(
            clone_ast(ast_node),
            int(num_segments),
            dual_layer=bool(dual_layer_used),
            leaf_builder=leaf_builder,
            device=device,
            dtype=dtype,
            freeze_non_nn=False,
        )
        model_i = _apply_fit_link_to_model(model_i, lm_hp)
        if model_hp is not None and getattr(model_hp, "nparam_max", None) is not None:
            if int(model_i.num_parameters()) > int(model_hp.nparam_max):
                raise RuntimeError(
                    f"model has {int(model_i.num_parameters())} parameters "
                    f"> nparam_max {int(model_hp.nparam_max)}"
                )
        return model_i, ast_i

    def _compare_loss(model_i, val_loader):
        if y_op_inv is not None:
            from nestynet_sr.sr_search.stageB.evaluation import _eval_original_y_mse_with_inverse

            return float(_eval_original_y_mse_with_inverse(model_i, val_loader, device, y_op_inv))
        return float(_eval_yspace_mse(model_i, val_loader, device))

    reference_rng_state = _stageA_snapshot_rng_state()

    def _run_pass(pass_epochs: int, refit_tier: str) -> dict:
        executor = CoEWitnessExecutor.from_config(lm_hp)

        def _worker(job) -> dict:
            spec = job.payload
            row = {
                "method": "refit_compare",
                "refit_tier": str(refit_tier),
                "slice_id": int(spec.slice_id),
                "train_rows": [int(spec.train_start), int(spec.train_stop)],
                "val_rows": [int(spec.val_start), int(spec.val_stop)],
                "epochs": int(pass_epochs),
                "status": "error",
            }
            try:
                train_loader = _slice_loader(int(spec.train_start), int(spec.train_stop))
                val_loader = _slice_loader(int(spec.val_start), int(spec.val_stop))
                _stageA_restore_rng_state(reference_rng_state)
                base_rng_state = _stageA_snapshot_rng_state()
                base_model, _ = _build_model(base_ast)
                _, base_fit_loss, base_train_loss, base_p, base_opt = train_candidate_model(
                    base_model,
                    train_loader,
                    val_loader,
                    epochs=int(pass_epochs),
                    LM_strategy=lm_hp.strategy,
                    nval_patience=max(int(pass_epochs) + 1, int(getattr(lm_hp, "nval_patience", 1) or 1)),
                    loss_target=None,
                    accept_threshold=None,
                    epochs_min=min(int(getattr(lm_hp, "epochs_min", 0) or 0), int(pass_epochs)),
                    chisq_tol=lm_hp.chisq_tol,
                    device=device,
                    epochs_awful_check=None,
                    awful_threshold=None,
                    log_file=lm_hp.log_file,
                    log_to_console=False,
                    log_level=lm_hp.log_level,
                    lm_verbose=False,
                    lm_hp=lm_hp,
                )
                if base_p is not None and base_opt is not None:
                    base_opt._update_param_groups(base_p)

                _stageA_restore_rng_state(base_rng_state)
                cand_model, _ = _build_model(cand_ast)
                _, cand_fit_loss, cand_train_loss, cand_p, cand_opt = train_candidate_model(
                    cand_model,
                    train_loader,
                    val_loader,
                    epochs=int(pass_epochs),
                    LM_strategy=lm_hp.strategy,
                    nval_patience=max(int(pass_epochs) + 1, int(getattr(lm_hp, "nval_patience", 1) or 1)),
                    loss_target=None,
                    accept_threshold=None,
                    epochs_min=min(int(getattr(lm_hp, "epochs_min", 0) or 0), int(pass_epochs)),
                    chisq_tol=lm_hp.chisq_tol,
                    device=device,
                    epochs_awful_check=None,
                    awful_threshold=None,
                    log_file=lm_hp.log_file,
                    log_to_console=False,
                    log_level=lm_hp.log_level,
                    lm_verbose=False,
                    lm_hp=lm_hp,
                )
                if cand_p is not None and cand_opt is not None:
                    cand_opt._update_param_groups(cand_p)

                base_compare = _compare_loss(base_model, val_loader)
                cand_compare = _compare_loss(cand_model, val_loader)
                if not (
                    math.isfinite(float(base_compare))
                    and math.isfinite(float(cand_compare))
                ):
                    raise RuntimeError(
                        f"non-finite compare loss base={base_compare}, candidate={cand_compare}"
                    )
                tol = _committee_tolerance(
                    loss_a=float(base_compare),
                    loss_b=float(cand_compare),
                    noise_floor_raw=float(getattr(lm_hp, "coe_noise_floor_raw", 0.0) or 0.0),
                    n_eff=max(1, int(spec.val_stop - spec.val_start)),
                    noise_mult=float(getattr(lm_hp, "coe_noise_mult", 3.0) or 3.0),
                    rel_tol=float(getattr(lm_hp, "coe_rel_tol", 1.0e-3) or 1.0e-3),
                )
                delta = float(cand_compare) - float(base_compare)
                if delta < -tol:
                    vote = "win"
                elif delta > tol:
                    vote = "loss"
                else:
                    vote = "tie"
                row.update(
                    {
                        "status": "success",
                        "incumbent_val_loss": float(base_fit_loss),
                        "candidate_val_loss": float(cand_fit_loss),
                        "incumbent_train_loss": float(base_train_loss),
                        "candidate_train_loss": float(cand_train_loss),
                        "incumbent_compare_loss": float(base_compare),
                        "candidate_compare_loss": float(cand_compare),
                        "comparison_space": "raw_y" if y_op_inv is not None else "model_output_space",
                        "delta": float(delta),
                        "tolerance": float(tol),
                        "vote": vote,
                        "n_val": int(spec.val_stop - spec.val_start),
                    }
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            return row

        rows = executor.run(
            coe_witness_jobs_from_specs(specs, prefix=f"stageA_prune_{refit_tier}"),
            _worker,
        )
        paired = [row for row in rows if row.get("status") == "success"]

        wins = sum(1 for r in paired if r.get("vote") == "win")
        ties = sum(1 for r in paired if r.get("vote") == "tie")
        losses = sum(1 for r in paired if r.get("vote") == "loss")
        inc_losses = [float(r["incumbent_compare_loss"]) for r in paired]
        cand_losses = [float(r["candidate_compare_loss"]) for r in paired]
        inc_med = float(_np.median(_np.asarray(inc_losses, dtype=float))) if inc_losses else float("inf")
        cand_med = float(_np.median(_np.asarray(cand_losses, dtype=float))) if cand_losses else float("inf")
        med_tol = _committee_tolerance(
            loss_a=inc_med,
            loss_b=cand_med,
            noise_floor_raw=float(getattr(lm_hp, "coe_noise_floor_raw", 0.0) or 0.0),
            n_eff=max(1, len(paired) * int(getattr(lm_hp, "coe_ndata_val", 2000) or 2000)),
            noise_mult=float(getattr(lm_hp, "coe_noise_mult", 3.0) or 3.0),
            rel_tol=float(getattr(lm_hp, "coe_rel_tol", 1.0e-3) or 1.0e-3),
        )
        return {
            "enabled": True,
            "gate_kind": "stageA_destructive_prune_refit",
            "gate_status": "evaluated",
            "mode": mode,
            "n_slices": int(len(specs)),
            "reference_slice": getattr(lm_hp, "coe_reference_slice", None),
            "excluded_slice_ids": (
                []
                if getattr(lm_hp, "coe_reference_slice", None) is None
                else [int(getattr(lm_hp, "coe_reference_slice"))]
            ),
            "evaluated_slices": int(len(rows)),
            "n_paired_success": int(len(paired)),
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "incumbent_median_mse": float(inc_med),
            "candidate_median_mse": float(cand_med),
            "median_delta": float(cand_med - inc_med),
            "median_tolerance": float(med_tol),
            "comparison_space": "raw_y" if y_op_inv is not None else "model_output_space",
            "global_raw_axis_erasure": bool(erased),
            "erased_raw_axes": [int(v) for v in erased],
            "axes_to_drop": [int(v) for v in dropped],
            "epochs": int(pass_epochs),
            "refit_tier": str(refit_tier),
            "witness_executor": coe_witness_execution_metadata(
                executor,
                rows,
                parallel_disabled_reason="stageA_destructive_prune_refit_mutates_live_context",
            ),
            "results": rows,
            "rng_restored": True,
        }

    def _allow_from(summary_i: dict) -> bool:
        paired_i = int(summary_i.get("n_paired_success", 0) or 0)
        if paired_i <= 0:
            return True
        if int(summary_i.get("losses", 0) or 0) > 0:
            return False
        delta_i = float(summary_i.get("median_delta", float("inf")))
        tol_i = float(summary_i.get("median_tolerance", 0.0) or 0.0)
        return delta_i <= tol_i

    epochs = max(1, int(getattr(lm_hp, "coe_stageB_refit_epochs", 200) or 200))
    escalate_epochs = max(0, int(getattr(lm_hp, "coe_stageB_refit_escalate_epochs", 0) or 0))
    try:
        summary = _run_pass(epochs, "tier0")
        if int(summary.get("n_paired_success", 0) or 0) <= 0:
            summary["gate_status"] = "legacy_allow"
            summary["decision"] = "allow"
            return True, "coe-stageA-prune-gate-unsupported: no paired witness refits", summary
        if _allow_from(summary):
            summary["gate_status"] = "accepted"
            summary["decision"] = "allow"
            return True, (
                "coe-stageA-prune-gate-accepted "
                f"(wins/ties/losses={summary['wins']}/{summary['ties']}/{summary['losses']}, "
                f"median_delta={summary['median_delta']:.3e}, tol={summary['median_tolerance']:.3e})"
            ), summary
        if escalate_epochs > epochs:
            summary2 = _run_pass(escalate_epochs, "tier1")
            summary["escalated"] = summary2
            if _allow_from(summary2):
                summary2["gate_status"] = "accepted_after_escalation"
                summary2["decision"] = "allow"
                return True, (
                    "coe-stageA-prune-gate-accepted-after-escalation "
                    f"(wins/ties/losses={summary2['wins']}/{summary2['ties']}/{summary2['losses']}, "
                    f"median_delta={summary2['median_delta']:.3e}, tol={summary2['median_tolerance']:.3e})"
                ), summary2
            summary2["gate_status"] = "veto"
            summary2["decision"] = "veto"
            return False, (
                "reject-coe-stageA-prune-gate "
                f"(wins/ties/losses={summary2['wins']}/{summary2['ties']}/{summary2['losses']}, "
                f"median_delta={summary2['median_delta']:.3e}, tol={summary2['median_tolerance']:.3e})"
            ), summary2
        summary["gate_status"] = "veto"
        summary["decision"] = "veto"
        return False, (
            "reject-coe-stageA-prune-gate "
            f"(wins/ties/losses={summary['wins']}/{summary['ties']}/{summary['losses']}, "
            f"median_delta={summary['median_delta']:.3e}, tol={summary['median_tolerance']:.3e})"
        ), summary
    finally:
        _stageA_restore_rng_state(reference_rng_state)


def _stageA_terminal_closure_committee_gate(
    *,
    base_ast: Any,
    cand_ast: Any,
    base_model,
    cand_model,
    label: str,
    gate_kind: str,
    lm_hp,
    loss_floor: float,
    y_op,
    y_op_inv,
    dtype,
    device,
    data_hp=None,
) -> tuple[bool, str, dict]:
    """CoE gate for visible Stage-A terminal closures.

    This is intentionally a narrow PR-A3 gate.  It only acts on candidates that
    have already removed all NN leaves.  Committee slices evaluate the same
    reference-trained incumbent/candidate models on independent validation
    slices; they do not run independent Stage-A search.
    """

    mode = str(getattr(lm_hp, "coe_mode", "off") or "off")
    if mode not in {"committee_gated", "reservoir_discovery"}:
        return True, "coe-stageA-terminal-gate-disabled", {"enabled": False}
    try:
        if collect_nn_atoms(cand_ast):
            return True, "coe-stageA-terminal-gate-unsupported: candidate still contains NN", {
                "enabled": True,
                "gate_status": "legacy_allow",
                "reason": "candidate still contains NN leaves",
                "gate_kind": str(gate_kind),
            }
    except Exception:
        return True, "coe-stageA-terminal-gate-unsupported: NN scan failed", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": "candidate NN scan failed",
            "gate_kind": str(gate_kind),
        }
    xmap = getattr(base_model, "_x_transform", None) or getattr(cand_model, "_x_transform", None) or {}
    if xmap:
        return True, "coe-stageA-terminal-gate-unsupported: active x-transform", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": "active x-transform replay is not wired for Stage-A terminal gate",
            "gate_kind": str(gate_kind),
        }
    filepath = getattr(lm_hp, "coe_filepath", None)
    if not filepath:
        return True, "coe-stageA-terminal-gate-unsupported: no filepath", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": "no single-dataset filepath configured",
            "gate_kind": str(gate_kind),
        }
    n_slices = min(
        int(getattr(lm_hp, "coe_num_slices", 0) or 0),
        int(getattr(lm_hp, "coe_stageB_gate_slices", 0) or 0),
    )
    if n_slices <= 0:
        return True, "coe-stageA-terminal-gate-unsupported: no slices configured", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": "no CoE Stage-A terminal gate slices configured",
            "gate_kind": str(gate_kind),
        }

    try:
        import numpy as _np

        from nestynet_sr.sr_search.coe_committee import (
            _committee_tolerance,
            _load_dataset_arrays,
            build_slice_specs,
        )
    except Exception as exc:
        return True, f"coe-stageA-terminal-gate-unsupported: helpers unavailable ({type(exc).__name__})", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": f"committee helpers unavailable: {type(exc).__name__}: {exc}",
            "gate_kind": str(gate_kind),
        }

    try:
        X_all, y_all, _cols = _load_dataset_arrays(str(filepath))
    except Exception as exc:
        return True, f"coe-stageA-terminal-gate-unsupported: data load failed ({type(exc).__name__})", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": f"slice data load failed: {type(exc).__name__}: {exc}",
            "gate_kind": str(gate_kind),
        }

    specs = build_slice_specs(
        n_slices=n_slices,
        ndata_train=int(getattr(lm_hp, "coe_ndata_train", 2000) or 2000),
        ndata_val=int(getattr(lm_hp, "coe_ndata_val", 2000) or 2000),
        start_slice=int(getattr(lm_hp, "coe_start_slice", 0) or 0),
        skip_slice_ids=(
            ()
            if getattr(lm_hp, "coe_reference_slice", None) is None
            else (int(getattr(lm_hp, "coe_reference_slice")),)
        ),
        max_rows=int(y_all.shape[0]),
    )
    if not specs:
        return True, "coe-stageA-terminal-gate-unsupported: no valid witness slices", {
            "enabled": True,
            "gate_status": "legacy_allow",
            "reason": "no independent CoE witness slices fit inside the dataset",
            "gate_kind": str(gate_kind),
        }

    def _slice_loader(start: int, stop: int):
        if start < 0 or stop <= start or stop > int(y_all.shape[0]):
            raise ValueError(
                f"slice rows [{start}, {stop}) outside dataset with {int(y_all.shape[0])} rows"
            )
        x_slice = _np.array(X_all[start:stop], dtype=_np.float64, copy=True)
        y_slice = _np.array(y_all[start:stop], dtype=_np.float64, copy=True).reshape(-1, 1)
        if y_op is not None:
            y_slice = _np.array(y_op(y_slice), dtype=_np.float64, copy=True).reshape(-1, 1)
            if not _np.all(_np.isfinite(y_slice)):
                raise ValueError(f"non-finite y-transform target values on rows [{start}, {stop})")
        xb = torch.as_tensor(x_slice, dtype=dtype)
        yb = torch.as_tensor(y_slice, dtype=dtype)
        batch_size = int(getattr(data_hp, "batch_size", 0) or xb.shape[0])
        batch_size = max(1, min(batch_size, int(xb.shape[0])))
        return DataLoader(
            TensorDataset(xb, yb),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    def _compare_loss(model_i, val_loader):
        if model_i is None:
            return float("inf")
        if y_op_inv is not None:
            from nestynet_sr.sr_search.stageB.evaluation import _eval_original_y_mse_with_inverse

            return float(_eval_original_y_mse_with_inverse(model_i, val_loader, device, y_op_inv))
        return float(_eval_yspace_mse(model_i, val_loader, device))

    executor = CoEWitnessExecutor.from_config(lm_hp)

    def _worker(job) -> dict:
        spec = job.payload
        row = {
            "method": "transfer_compare",
            "slice_id": int(spec.slice_id),
            "train_rows": [int(spec.train_start), int(spec.train_stop)],
            "val_rows": [int(spec.val_start), int(spec.val_stop)],
            "status": "error",
        }
        try:
            val_loader = _slice_loader(int(spec.val_start), int(spec.val_stop))
            base_loss = _compare_loss(base_model, val_loader)
            cand_loss = _compare_loss(cand_model, val_loader)
            if not (
                math.isfinite(float(base_loss))
                and math.isfinite(float(cand_loss))
                and float(base_loss) >= 0.0
                and float(cand_loss) >= 0.0
            ):
                raise RuntimeError(
                    f"non-finite compare loss base={base_loss}, candidate={cand_loss}"
                )
            tol = _committee_tolerance(
                loss_a=float(base_loss),
                loss_b=float(cand_loss),
                noise_floor_raw=float(getattr(lm_hp, "coe_noise_floor_raw", 0.0) or 0.0),
                n_eff=max(1, int(spec.val_stop - spec.val_start)),
                noise_mult=float(getattr(lm_hp, "coe_noise_mult", 3.0) or 3.0),
                rel_tol=float(getattr(lm_hp, "coe_rel_tol", 1.0e-3) or 1.0e-3),
            )
            delta = float(cand_loss) - float(base_loss)
            if delta < -tol:
                vote = "win"
            elif delta > tol:
                vote = "loss"
            else:
                vote = "tie"
            row.update(
                {
                    "status": "success",
                    "incumbent_compare_loss": float(base_loss),
                    "candidate_compare_loss": float(cand_loss),
                    "comparison_space": "raw_y" if y_op_inv is not None else "model_output_space",
                    "delta": float(delta),
                    "tolerance": float(tol),
                    "vote": vote,
                    "n_val": int(spec.val_stop - spec.val_start),
                }
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    rows = run_threaded_witnesses(
        coe_witness_jobs_from_specs(specs, prefix=f"stageA_terminal_{gate_kind}"),
        _worker,
        executor=executor,
    )
    paired = [row for row in rows if row.get("status") == "success"]

    wins = sum(1 for r in paired if r.get("vote") == "win")
    ties = sum(1 for r in paired if r.get("vote") == "tie")
    losses = sum(1 for r in paired if r.get("vote") == "loss")
    inc_losses = [float(r["incumbent_compare_loss"]) for r in paired]
    cand_losses = [float(r["candidate_compare_loss"]) for r in paired]
    inc_med = float(_np.median(_np.asarray(inc_losses, dtype=float))) if inc_losses else float("inf")
    cand_med = float(_np.median(_np.asarray(cand_losses, dtype=float))) if cand_losses else float("inf")
    med_tol = _committee_tolerance(
        loss_a=inc_med,
        loss_b=cand_med,
        noise_floor_raw=float(getattr(lm_hp, "coe_noise_floor_raw", 0.0) or 0.0),
        n_eff=max(1, len(paired) * int(getattr(lm_hp, "coe_ndata_val", 2000) or 2000)),
        noise_mult=float(getattr(lm_hp, "coe_noise_mult", 3.0) or 3.0),
        rel_tol=float(getattr(lm_hp, "coe_rel_tol", 1.0e-3) or 1.0e-3),
    )
    summary = {
        "enabled": True,
        "gate_kind": str(gate_kind),
        "gate_status": "evaluated",
        "mode": mode,
        "label": str(label),
        "n_slices": int(len(specs)),
        "reference_slice": getattr(lm_hp, "coe_reference_slice", None),
        "excluded_slice_ids": (
            []
            if getattr(lm_hp, "coe_reference_slice", None) is None
            else [int(getattr(lm_hp, "coe_reference_slice"))]
        ),
        "evaluated_slices": int(len(rows)),
        "n_paired_success": int(len(paired)),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "incumbent_median_mse": float(inc_med),
        "candidate_median_mse": float(cand_med),
        "median_delta": float(cand_med - inc_med),
        "median_tolerance": float(med_tol),
        "loss_floor": float(loss_floor),
        "comparison_space": "raw_y" if y_op_inv is not None else "model_output_space",
        "witness_executor": coe_witness_execution_metadata(executor, rows),
        "results": rows,
    }
    if int(summary["n_paired_success"]) <= 0:
        summary["gate_status"] = "legacy_allow"
        summary["decision"] = "allow"
        return True, "coe-stageA-terminal-gate-unsupported: no paired witness evaluations", summary
    if losses == 0 and float(summary["median_delta"]) <= float(summary["median_tolerance"]):
        summary["gate_status"] = "accepted"
        summary["decision"] = "allow"
        return True, (
            "coe-stageA-terminal-gate-accepted "
            f"(wins/ties/losses={wins}/{ties}/{losses}, "
            f"median_delta={summary['median_delta']:.3e}, tol={summary['median_tolerance']:.3e})"
        ), summary

    summary["gate_status"] = "veto"
    summary["decision"] = "veto"
    return False, (
        "reject-coe-stageA-terminal-gate "
        f"(wins/ties/losses={wins}/{ties}/{losses}, "
        f"median_delta={summary['median_delta']:.3e}, tol={summary['median_tolerance']:.3e})"
    ), summary


def _stageA_compound_pattern_support_size(pattern: Any) -> int:
    try:
        vals = tuple(pattern or ())
    except Exception:
        vals = ()
    support = 0
    for val in vals:
        try:
            if abs(float(val)) > 1.0e-12:
                support += 1
                continue
        except Exception:
            pass
        try:
            if str(val).strip() not in {"", "0", "0.0"}:
                support += 1
        except Exception:
            pass
    return int(support)


def _stageA_compound_structural_priority(cand: dict) -> tuple[int, int, int, int, int]:
    """Lower is better for noisy CoE compound tie-breaking."""

    try:
        old_arity = int(cand.get("old_arity", 0))
        new_arity = int(cand.get("new_arity", old_arity))
    except Exception:
        old_arity = 0
        new_arity = 0
    arity_drop = max(0, old_arity - new_arity)
    support_size = _stageA_compound_pattern_support_size(cand.get("pattern"))
    kind = str(cand.get("kind", "") or "").lower()
    visible_prefactor = bool(
        cand.get("visible_prefactor_transaction")
        or cand.get("prefactor_ast_present")
        or cand.get("prefactor_exponents") is not None
    )
    norm_like = kind in {"radial", "metric_distance", "power_pair_sumdiff"}
    full_compound = bool(new_arity <= 1 and arity_drop >= 2)
    full_support_radial = bool(norm_like and support_size >= old_arity and new_arity <= 1)
    protected_norm_completion = bool(norm_like and support_size >= 2 and arity_drop >= 1)
    full_visible_prefactor = bool(visible_prefactor and full_compound)
    try:
        prefactor_pi_gauge_abs = int(cand.get("prefactor_pi_gauge_abs", 0) or 0)
    except Exception:
        prefactor_pi_gauge_abs = 0
    if full_visible_prefactor:
        bucket = 0
    elif visible_prefactor:
        bucket = 1
    elif full_support_radial or protected_norm_completion:
        bucket = 2
    elif full_compound:
        bucket = 3
    elif arity_drop >= 2:
        bucket = 4
    else:
        bucket = 5
    return (
        int(bucket),
        int(prefactor_pi_gauge_abs),
        -int(arity_drop),
        int(new_arity),
        -int(support_size),
    )


def _stageA_compound_is_structurally_protected(cand: dict) -> bool:
    if bool(cand.get("structural_protected", False)):
        try:
            old_a = int(cand.get("old_arity", 0))
            new_a = int(cand.get("new_arity", old_a))
        except Exception:
            old_a = 0
            new_a = 0
        if old_a > new_a:
            return True
    priority = _stageA_compound_structural_priority(cand)
    try:
        arity_drop = int(cand.get("old_arity", 0)) - int(cand.get("new_arity", 0))
    except Exception:
        arity_drop = 0
    return bool(priority[0] <= 2 or arity_drop >= 2)


def _stageA_compound_shortlist_committee_rank(
    *,
    base_model,
    candidates,
    lm_hp,
    y_op,
    y_op_inv,
    dtype,
    device,
    data_hp=None,
) -> tuple[Optional[dict], str, dict]:
    """CoE ranker for a bounded shortlist of accepted Stage-A compounds."""

    mode = str(getattr(lm_hp, "coe_mode", "off") or "off")
    cand_list = [c for c in list(candidates or []) if isinstance(c, dict)]
    legacy = cand_list[0] if cand_list else None

    def _reduces_arity(cand: dict) -> bool:
        try:
            old_arity = int(cand.get("old_arity", -1) or -1)
            new_arity = int(cand.get("new_arity", old_arity) or old_arity)
            return old_arity > new_arity
        except Exception:
            return False

    budgeted_witnesses_required = bool(
        mode in {"committee_gated", "reservoir_discovery"}
        and any(_reduces_arity(cand) for cand in cand_list)
    )
    summary: dict[str, Any] = {
        "enabled": mode in {"committee_gated", "reservoir_discovery"},
        "gate_kind": "stageA_compound_shortlist_rank",
        "mode": mode,
        "gate_status": "skipped",
        "decision": "legacy",
        "legacy_selected": None if legacy is None else str(legacy.get("z_name", "compound")),
        "selected": None if legacy is None else str(legacy.get("z_name", "compound")),
        "candidate_count": int(len(cand_list)),
        "budgeted_witnesses_required": bool(budgeted_witnesses_required),
        "excluded_slice_ids": [],
        "results": [],
    }

    def _witness_unavailable(reason: str, legacy_code: str):
        if budgeted_witnesses_required:
            summary.update(
                {
                    "gate_status": "veto",
                    "decision": "veto_all",
                    "selected": None,
                    "reason": f"Required budgeted structural witnesses unavailable: {reason}",
                }
            )
            return None, "reject-coe-stageA-compound-budget-witness-unavailable", summary
        summary.update({"gate_status": "legacy_allow", "reason": reason})
        return legacy, legacy_code, summary

    if mode not in {"committee_gated", "reservoir_discovery"}:
        summary["reason"] = "CoE compound shortlist ranker is inactive in normal mode."
        return legacy, "legacy-coe-stageA-compound-shortlist-disabled", summary
    if not cand_list:
        summary.update({"decision": "none", "reason": "No compound candidates were available."})
        return None, "coe-stageA-compound-shortlist-empty", summary
    filepath = getattr(lm_hp, "coe_filepath", None)
    if not filepath:
        return _witness_unavailable(
            "No single raw data filepath is available for CoE compound witnesses.",
            "legacy-coe-stageA-compound-shortlist-no-filepath",
        )
    if base_model is None:
        return _witness_unavailable(
            "Base model is unavailable for paired compound comparison.",
            "legacy-coe-stageA-compound-shortlist-no-base",
        )
    try:
        xmap = getattr(base_model, "_x_transform", None) or {}
        for cand in cand_list:
            xmap = xmap or getattr(cand.get("model"), "_x_transform", None) or {}
        if xmap:
            return _witness_unavailable(
                "Active x-transform replay is not wired for compound shortlist ranking.",
                "legacy-coe-stageA-compound-shortlist-x-transform",
            )
    except Exception:
        pass

    n_slices = min(
        int(getattr(lm_hp, "coe_num_slices", 0) or 0),
        int(getattr(lm_hp, "coe_stageB_gate_slices", 0) or 0),
    )
    if n_slices <= 0:
        return _witness_unavailable(
            "No CoE compound shortlist witness slices are configured.",
            "legacy-coe-stageA-compound-shortlist-no-slices",
        )

    try:
        import numpy as _np

        from nestynet_sr.sr_search.coe_committee import (
            _committee_tolerance,
            _load_dataset_arrays,
            build_slice_specs,
        )
    except Exception as exc:
        return _witness_unavailable(
            f"committee helpers unavailable: {type(exc).__name__}: {exc}",
            "legacy-coe-stageA-compound-shortlist-import-error",
        )

    try:
        X_all, y_all, _cols = _load_dataset_arrays(str(filepath))
    except Exception as exc:
        return _witness_unavailable(
            f"slice data load failed: {type(exc).__name__}: {exc}",
            "legacy-coe-stageA-compound-shortlist-data-error",
        )

    specs = build_slice_specs(
        n_slices=n_slices,
        ndata_train=int(getattr(lm_hp, "coe_ndata_train", 2000) or 2000),
        ndata_val=int(getattr(lm_hp, "coe_ndata_val", 2000) or 2000),
        start_slice=int(getattr(lm_hp, "coe_start_slice", 0) or 0),
        skip_slice_ids=(
            ()
            if getattr(lm_hp, "coe_reference_slice", None) is None
            else (int(getattr(lm_hp, "coe_reference_slice")),)
        ),
        max_rows=int(y_all.shape[0]),
    )
    summary["slice_specs"] = [s.to_dict() for s in specs]
    summary["excluded_slice_ids"] = (
        []
        if getattr(lm_hp, "coe_reference_slice", None) is None
        else [int(getattr(lm_hp, "coe_reference_slice"))]
    )
    if not specs:
        return _witness_unavailable(
            "No independent CoE witness slices fit inside the dataset.",
            "legacy-coe-stageA-compound-shortlist-no-valid-slices",
        )

    def _slice_loader(start: int, stop: int):
        if start < 0 or stop <= start or stop > int(y_all.shape[0]):
            raise ValueError(
                f"slice rows [{start}, {stop}) outside dataset with {int(y_all.shape[0])} rows"
            )
        x_slice = _np.array(X_all[start:stop], dtype=_np.float64, copy=True)
        y_slice = _np.array(y_all[start:stop], dtype=_np.float64, copy=True).reshape(-1, 1)
        if y_op is not None:
            y_slice = _np.array(y_op(y_slice), dtype=_np.float64, copy=True).reshape(-1, 1)
            if not _np.all(_np.isfinite(y_slice)):
                raise ValueError(f"non-finite y-transform target values on rows [{start}, {stop})")
        xb = torch.as_tensor(x_slice, dtype=dtype)
        yb = torch.as_tensor(y_slice, dtype=dtype)
        batch_size = int(
            getattr(data_hp, "batch_size", 0)
            or getattr(lm_hp, "coe_ndata_val", 0)
            or xb.shape[0]
        )
        batch_size = max(1, min(batch_size, int(xb.shape[0])))
        return DataLoader(TensorDataset(xb, yb), batch_size=batch_size, shuffle=False, drop_last=False)

    def _compare_loss(model_i, val_loader):
        if model_i is None:
            return float("inf")
        if y_op_inv is not None:
            from nestynet_sr.sr_search.stageB.evaluation import _eval_original_y_mse_with_inverse

            return float(_eval_original_y_mse_with_inverse(model_i, val_loader, device, y_op_inv))
        return float(_eval_yspace_mse(model_i, val_loader, device))

    executor = CoEWitnessExecutor.from_config(lm_hp)
    witness_jobs = coe_witness_jobs_from_specs(specs, prefix="stageA_compound_base")

    def _base_worker(job) -> dict:
        spec = job.payload
        row = {"slice_id": int(spec.slice_id), "val_rows": [int(spec.val_start), int(spec.val_stop)], "status": "error"}
        try:
            val_loader = _slice_loader(int(spec.val_start), int(spec.val_stop))
            base_loss = float(_compare_loss(base_model, val_loader))
            if not math.isfinite(base_loss):
                raise RuntimeError(f"non-finite base loss {base_loss}")
            row.update({"status": "success", "loss": base_loss})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    base_rows = run_threaded_witnesses(witness_jobs, _base_worker, executor=executor)
    base_losses: dict[int, float] = {
        int(row["slice_id"]): float(row["loss"])
        for row in base_rows
        if row.get("status") == "success"
    }
    summary["incumbent_rows"] = base_rows
    if not base_losses:
        return _witness_unavailable(
            "Incumbent had no successful independent witness evaluations.",
            "legacy-coe-stageA-compound-shortlist-no-base-witness",
        )

    def _median(vals) -> float:
        arr = _np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=float)
        if arr.size == 0:
            return float("inf")
        return float(_np.median(arr))

    allowed = []
    cand_summaries: list[dict] = []
    noise_floor_raw = float(getattr(lm_hp, "coe_noise_floor_raw", 0.0) or 0.0)
    noise_mult = float(getattr(lm_hp, "coe_noise_mult", 3.0) or 3.0)
    rel_tol = float(getattr(lm_hp, "coe_rel_tol", 1.0e-3) or 1.0e-3)
    protected_loss_mult = float(getattr(lm_hp, "coe_stageA_protected_tie_max_loss_mult", 5.0) or 5.0)

    for idx, cand in enumerate(cand_list):
        def _cand_worker(job) -> dict:
            spec = job.payload
            sid = int(spec.slice_id)
            row = {
                "slice_id": sid,
                "val_rows": [int(spec.val_start), int(spec.val_stop)],
                "status": "error",
            }
            try:
                if sid not in base_losses:
                    raise RuntimeError("incumbent witness loss unavailable")
                val_loader = _slice_loader(int(spec.val_start), int(spec.val_stop))
                cand_loss = float(_compare_loss(cand.get("model"), val_loader))
                if not math.isfinite(cand_loss):
                    raise RuntimeError(f"non-finite candidate loss {cand_loss}")
                base_loss = float(base_losses[sid])
                tol = _committee_tolerance(
                    loss_a=base_loss,
                    loss_b=cand_loss,
                    noise_floor_raw=noise_floor_raw,
                    n_eff=max(1, int(spec.val_stop - spec.val_start)),
                    noise_mult=noise_mult,
                    rel_tol=rel_tol,
                )
                delta = cand_loss - base_loss
                if delta < -tol:
                    vote = "win"
                elif delta > tol:
                    vote = "loss"
                else:
                    vote = "tie"
                row.update(
                    {
                        "status": "success",
                        "incumbent_loss": base_loss,
                        "candidate_loss": cand_loss,
                        "delta": float(delta),
                        "tolerance": float(tol),
                        "vote": vote,
                        "n_val": int(spec.val_stop - spec.val_start),
                    }
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            return row

        rows = run_threaded_witnesses(
            coe_witness_jobs_from_specs(specs, prefix=f"stageA_compound_c{idx}"),
            _cand_worker,
            executor=executor,
        )
        wins = sum(1 for row in rows if row.get("status") == "success" and row.get("vote") == "win")
        ties = sum(1 for row in rows if row.get("status") == "success" and row.get("vote") == "tie")
        losses = sum(1 for row in rows if row.get("status") == "success" and row.get("vote") == "loss")
        invalid = sum(1 for row in rows if row.get("status") != "success")
        cand_losses = [
            float(row["candidate_loss"])
            for row in rows
            if row.get("status") == "success" and row.get("candidate_loss") is not None
        ]
        deltas = [
            float(row["delta"])
            for row in rows
            if row.get("status") == "success" and row.get("delta") is not None
        ]

        med_loss = _median(cand_losses)
        med_delta = _median(deltas)
        med_tol = _median(
            [
                row.get("tolerance", 0.0)
                for row in rows
                if isinstance(row, dict) and row.get("tolerance") is not None
            ]
        )
        n_paired = int(wins + ties + losses)
        structural_priority = _stageA_compound_structural_priority(cand)
        arity_drop = max(
            0,
            int(cand.get("old_arity", -1) or -1) - int(cand.get("new_arity", -1) or -1),
        )
        protected = _stageA_compound_is_structurally_protected(cand)
        minority_losses = bool(losses <= max(0, (n_paired - 1) // 2))
        protected_loss_rows_ok = True
        for row in rows:
            if not isinstance(row, dict) or row.get("status") != "success" or row.get("vote") != "loss":
                continue
            try:
                delta_i = float(row.get("delta", float("inf")))
                tol_i = float(row.get("tolerance", 0.0) or 0.0)
                if not math.isfinite(delta_i) or delta_i > protected_loss_mult * max(tol_i, 1.0e-30):
                    protected_loss_rows_ok = False
                    break
            except Exception:
                protected_loss_rows_ok = False
                break
        strict_allowed = bool(
            n_paired > 0
            and invalid == 0
            and losses == 0
            and math.isfinite(float(med_delta))
            and float(med_delta) <= max(0.0, float(med_tol))
        )
        protected_tie_allowed = bool(
            noise_floor_raw > 0.0
            and protected
            and n_paired > 0
            and invalid == 0
            and minority_losses
            and protected_loss_rows_ok
            and math.isfinite(float(med_delta))
            and float(med_delta) <= max(0.0, float(med_tol))
        )
        structural_budget_multiplier = float(
            cand.get("structural_budget_multiplier", 1.0) or 1.0
        )
        budgeted_admission = _stageA_budgeted_witness_admission(
            rows,
            base_loss_key="incumbent_loss",
            candidate_loss_key="candidate_loss",
            budget_multiplier=structural_budget_multiplier,
            noise_floor=noise_floor_raw,
        )
        old_arity_for_budget = int(cand.get("old_arity", -1) or -1)
        new_arity_for_budget = int(
            cand.get("new_arity", old_arity_for_budget) or old_arity_for_budget
        )
        provisional_allowed = bool(
            old_arity_for_budget > new_arity_for_budget
            and not strict_allowed
            and not protected_tie_allowed
            and budgeted_admission["all_slices_within_budget"]
        )
        allowed_i = bool(strict_allowed or protected_tie_allowed or provisional_allowed)
        cand_summary = {
            "index": int(idx),
            "z_name": str(cand.get("z_name", "compound")),
            "kind": str(cand.get("kind", "compound")),
            "z_readable": str(cand.get("z_readable", "")),
            "old_arity": int(cand.get("old_arity", -1)),
            "new_arity": int(cand.get("new_arity", -1)),
            "arity_drop": int(arity_drop),
            "pattern_support_size": int(_stageA_compound_pattern_support_size(cand.get("pattern"))),
            "structural_priority": list(structural_priority),
            "structurally_protected": bool(protected),
            "minority_losses": bool(minority_losses),
            "protected_loss_rows_ok": bool(protected_loss_rows_ok),
            "protected_tie_max_loss_mult": float(protected_loss_mult),
            "protected_tie_allowed": bool(protected_tie_allowed and not strict_allowed),
            "strict_allowed": bool(strict_allowed),
            "provisional_budget_allowed": bool(provisional_allowed),
            "admission_tier": "provisional_budget" if provisional_allowed else "strict",
            "structural_budget_multiplier": float(structural_budget_multiplier),
            "budgeted_witness_admission": budgeted_admission,
            "reference_val_loss": float(cand.get("val_loss", float("inf"))),
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "invalid": int(invalid),
            "n_paired_success": int(n_paired),
            "median_mse": float(med_loss),
            "median_delta": float(med_delta),
            "median_tolerance": float(med_tol),
            "allowed": bool(allowed_i),
            "rows": rows,
        }
        try:
            from nestynet_sr.sr_search.gate_telemetry import record_gate

            record_gate(
                "stageA_coe_accept",
                "committee",
                float(med_delta),
                float(med_tol),
                accepted=bool(allowed_i),
                context={
                    "kind": str(cand.get("kind", "compound")),
                    "z_name": str(cand.get("z_name", "")),
                    "family": str(cand.get("family", "")),
                    "visible_buckingham": bool(
                        cand.get("visible_buckingham_1d_prefactor")
                        or cand.get("family") == "visible_buckingham_1d_prefactor"
                    ),
                    "structural_protected_flag": bool(cand.get("structural_protected", False)),
                    "protected_effective": bool(protected),
                    "protected_tie_allowed": bool(protected_tie_allowed and not strict_allowed),
                    "provisional_budget_allowed": bool(provisional_allowed),
                    "structural_budget_multiplier": float(
                        structural_budget_multiplier
                    ),
                    "wins": int(wins),
                    "ties": int(ties),
                    "losses": int(losses),
                    "old_arity": int(cand.get("old_arity", -1)),
                    "new_arity": int(cand.get("new_arity", -1)),
                },
            )
        except Exception:
            pass
        cand_summaries.append(cand_summary)
        if allowed_i:
            allowed.append(
                (
                    tuple(structural_priority),
                    2 if provisional_allowed else (0 if wins > 0 else 1),
                    int(cand_summary["new_arity"]),
                    float(cand_summary["median_mse"]),
                    float(cand_summary["reference_val_loss"]),
                    int(idx),
                    cand,
                    cand_summary,
                )
            )

    summary.update(
        {
            "gate_status": "evaluated",
            "comparison_space": "raw_y" if y_op_inv is not None else "model_output_space",
            "incumbent_median_mse": _median(base_losses.values()),
            "witness_executor": coe_witness_execution_metadata(executor, base_rows),
            "results": cand_summaries,
        }
    )
    if not allowed:
        summary.update(
            {
                "decision": "veto_all",
                "selected": None,
                "reason": "All strict arity-reducing compound candidates lost or failed on witnesses.",
            }
        )
        return None, "reject-coe-stageA-compound-shortlist", summary

    best_allowed_median = min(float(item[7].get("median_mse", float("inf"))) for item in allowed)
    tied_allowed = []
    for item in allowed:
        cand_summary = item[7]
        med = float(cand_summary.get("median_mse", float("inf")))
        n_eff = 0
        for row in cand_summary.get("rows") or []:
            if isinstance(row, dict) and row.get("status") == "success":
                try:
                    n_eff += int(row.get("n_val", 0) or 0)
                except Exception:
                    pass
        tol = _committee_tolerance(
            loss_a=float(best_allowed_median),
            loss_b=med,
            noise_floor_raw=noise_floor_raw,
            n_eff=max(1, int(n_eff)),
            noise_mult=noise_mult,
            rel_tol=rel_tol,
        )
        cand_summary["noise_tied_with_best_allowed"] = bool(med <= best_allowed_median + float(tol))
        cand_summary["noise_tie_tolerance_vs_best_allowed"] = float(tol)
        if cand_summary["noise_tied_with_best_allowed"]:
            tied_allowed.append(item)

    selection_pool = tied_allowed or sorted(allowed, key=lambda item: item[3])[:1]
    selection_pool.sort(key=lambda item: item[:6])
    selected = selection_pool[0][6]
    selected_summary = selection_pool[0][7]
    selected_is_provisional = bool(selected_summary.get("provisional_budget_allowed", False))
    summary.update(
        {
            "gate_status": "accepted_provisional" if selected_is_provisional else "accepted",
            "decision": (
                "select_provisional_candidate" if selected_is_provisional else "select_candidate"
            ),
            "selected": str(selected_summary.get("z_name")),
            "selected_kind": str(selected_summary.get("kind")),
            "best_allowed_median_mse": float(best_allowed_median),
            "selection_pool_size": int(len(selection_pool)),
            "selected_median_mse": float(selected_summary.get("median_mse", float("inf"))),
            "selected_median_delta": float(selected_summary.get("median_delta", float("inf"))),
            "wins": int(selected_summary.get("wins", 0)),
            "ties": int(selected_summary.get("ties", 0)),
            "losses": int(selected_summary.get("losses", 0)),
            "provisional_budget_admission": bool(selected_is_provisional),
            "structural_budget_multiplier": float(
                selected_summary.get("structural_budget_multiplier", 1.0)
            ),
            "budgeted_witness_admission": selected_summary.get(
                "budgeted_witness_admission"
            ),
            "reason": (
                "Provisionally admitted a budgeted structural compound "
                if selected_is_provisional
                else "Selected compound candidate by independent witness ranking "
            ) + (
                f"({selected_summary.get('wins', 0)} wins, "
                f"{selected_summary.get('ties', 0)} ties, "
                f"{selected_summary.get('losses', 0)} losses)."
            ),
        }
    )
    reason = (
        "provisional-budget-coe-stageA-compound-shortlist"
        if selected_is_provisional
        else "accepted-coe-stageA-compound-shortlist"
    )
    return selected, reason, summary


def _format_stageA_compound_shortlist_committee_report(summary: Optional[dict]) -> str:
    if not isinstance(summary, dict):
        return "=== CoE Stage A Compound Shortlist ===\nenabled=False"
    lines = ["=== CoE Stage A Compound Shortlist ==="]
    lines.append(
        f"enabled={bool(summary.get('enabled', False))} "
        f"status={summary.get('gate_status', 'unknown')} "
        f"decision={summary.get('decision', 'unknown')} "
        f"selected={summary.get('selected')}"
    )
    try:
        lines.append(f"incumbent_median_mse={float(summary.get('incumbent_median_mse')):.6e}")
    except Exception:
        pass
    for row in list(summary.get("results") or [])[:6]:
        if not isinstance(row, dict):
            continue
        try:
            lines.append(
                f"candidate {row.get('index')} {row.get('z_name')} ({row.get('kind')}): "
                f"arity {row.get('old_arity')}->{row.get('new_arity')}, "
                f"prio={row.get('structural_priority')}, "
                f"median={float(row.get('median_mse')):.6e}, "
                f"delta={float(row.get('median_delta')):.6e}, "
                f"W/T/L/I={int(row.get('wins', 0))}/{int(row.get('ties', 0))}/"
                f"{int(row.get('losses', 0))}/{int(row.get('invalid', 0))}, "
                f"allowed={bool(row.get('allowed', False))}, "
                f"tier={row.get('admission_tier', 'strict')}, "
                f"budget={float(row.get('structural_budget_multiplier', 1.0)):.3g}x"
            )
        except Exception:
            lines.append(f"candidate {row.get('index')}: {row}")
    if summary.get("reason"):
        lines.append(str(summary.get("reason")))
    return "\n".join(lines)


def _stageA_overlap_split_committee_gate(
    *,
    base_model,
    cand_model,
    split_kind: str,
    has_overlap: bool,
    base_val_loss: Optional[float],
    cand_val_loss: Optional[float],
    noise_floor: float,
    under_protest: bool,
    lm_hp,
    y_op,
    y_op_inv,
    dtype,
    device,
    data_hp=None,
    split_diagnostic: Optional[dict] = None,
    structural_simplification: bool = False,
    structural_budget_multiplier: float = 1.0,
) -> tuple[bool, str, dict]:
    """CoE witness gate for high-risk or budgeted structural Stage-A splits.

    Witness slices compare the already-materialized parent and candidate split
    models. They do not search for their own split or mutate the reference tree.
    """

    mode = str(getattr(lm_hp, "coe_mode", "off") or "off")
    coe_enabled = mode in {"committee_gated", "reservoir_discovery"}
    budgeted_witnesses_required = bool(coe_enabled and structural_simplification)
    split_kind_s = str(split_kind or "split")
    is_mul = split_kind_s in {"mul", "multiply", "multiplicative"}
    risk_tags: list[str] = []
    if bool(has_overlap) and is_mul:
        risk_tags.append("overlap_multiplicative_split")

    try:
        nf = float(noise_floor)
    except Exception:
        nf = 0.0
    if (not math.isfinite(nf)) or nf <= 0.0:
        try:
            nf = float(getattr(lm_hp, "coe_noise_floor_raw", 0.0) or 0.0)
        except Exception:
            nf = 0.0
    near_mult = float(getattr(lm_hp, "coe_stageA_split_near_floor_mult", 25.0) or 25.0)
    if nf > 0.0 and math.isfinite(nf):
        finite_losses = []
        for v in (base_val_loss, cand_val_loss):
            try:
                fv = float(v)
                if math.isfinite(fv) and fv >= 0.0:
                    finite_losses.append(fv)
            except Exception:
                pass
        if finite_losses and min(finite_losses) <= near_mult * nf:
            risk_tags.append("near_noise_floor_split")
    if bool(under_protest):
        risk_tags.append("under_protest_split")

    summary: dict[str, Any] = {
        "enabled": coe_enabled,
        "gate_kind": "stageA_overlap_split_gate",
        "mode": mode,
        "structural_simplification": bool(structural_simplification),
        "structural_budget_multiplier": float(structural_budget_multiplier),
        "budgeted_witnesses_required": bool(budgeted_witnesses_required),
        "gate_status": "skipped",
        "decision": "legacy",
        "split_kind": split_kind_s,
        "has_overlap": bool(has_overlap),
        "risk_tags": list(risk_tags),
        "base_val_loss": None if base_val_loss is None else float(base_val_loss),
        "candidate_val_loss": None if cand_val_loss is None else float(cand_val_loss),
        "noise_floor": float(nf) if math.isfinite(nf) else 0.0,
        "under_protest": bool(under_protest),
        "excluded_slice_ids": [],
        "results": [],
    }
    if isinstance(split_diagnostic, dict):
        summary["split_diagnostic"] = dict(split_diagnostic)

    def _witness_unavailable(reason: str, legacy_code: str) -> tuple[bool, str, dict]:
        if budgeted_witnesses_required:
            summary.update(
                {
                    "gate_status": "veto",
                    "decision": "veto",
                    "reason": f"Required budgeted structural witnesses unavailable: {reason}",
                }
            )
            return (
                False,
                "reject-coe-stageA-overlap-split-budget-witness-unavailable",
                summary,
            )
        summary.update(
            {
                "gate_status": "legacy_allow",
                "reason": reason,
            }
        )
        return True, legacy_code, summary

    if mode not in {"committee_gated", "reservoir_discovery"}:
        summary["reason"] = "CoE Stage-A overlap split gate is inactive in normal mode."
        return True, "legacy-coe-stageA-overlap-split-disabled", summary
    if not risk_tags and not budgeted_witnesses_required:
        summary.update(
            {
                "decision": "allow",
                "reason": "Split is not in the current high-risk CoE Stage-A class.",
            }
        )
        return True, "coe-stageA-overlap-split-not-high-risk", summary
    filepath = getattr(lm_hp, "coe_filepath", None)
    if not filepath:
        return _witness_unavailable(
            "No single raw data filepath is available for CoE split witnesses.",
            "legacy-coe-stageA-overlap-split-no-filepath",
        )
    if base_model is None or cand_model is None:
        return _witness_unavailable(
            "Parent or candidate split model is unavailable.",
            "legacy-coe-stageA-overlap-split-no-model",
        )
    try:
        xmap = getattr(base_model, "_x_transform", None) or getattr(cand_model, "_x_transform", None) or {}
        if xmap:
            return _witness_unavailable(
                "Active x-transform replay is not wired for Stage-A split gating.",
                "legacy-coe-stageA-overlap-split-x-transform",
            )
    except Exception:
        pass

    n_slices = min(
        int(getattr(lm_hp, "coe_num_slices", 0) or 0),
        int(getattr(lm_hp, "coe_stageB_gate_slices", 0) or 0),
    )
    if n_slices <= 0:
        return _witness_unavailable(
            "No CoE Stage-A split witness slices are configured.",
            "legacy-coe-stageA-overlap-split-no-slices",
        )

    try:
        import numpy as _np

        from nestynet_sr.sr_search.coe_committee import (
            _committee_tolerance,
            _load_dataset_arrays,
            build_slice_specs,
        )
    except Exception as exc:
        return _witness_unavailable(
            f"committee helpers unavailable: {type(exc).__name__}: {exc}",
            "legacy-coe-stageA-overlap-split-import-error",
        )

    try:
        X_all, y_all, _cols = _load_dataset_arrays(str(filepath))
    except Exception as exc:
        return _witness_unavailable(
            f"slice data load failed: {type(exc).__name__}: {exc}",
            "legacy-coe-stageA-overlap-split-data-error",
        )

    specs = build_slice_specs(
        n_slices=n_slices,
        ndata_train=int(getattr(lm_hp, "coe_ndata_train", 2000) or 2000),
        ndata_val=int(getattr(lm_hp, "coe_ndata_val", 2000) or 2000),
        start_slice=int(getattr(lm_hp, "coe_start_slice", 0) or 0),
        skip_slice_ids=(
            ()
            if getattr(lm_hp, "coe_reference_slice", None) is None
            else (int(getattr(lm_hp, "coe_reference_slice")),)
        ),
        max_rows=int(y_all.shape[0]),
    )
    summary["slice_specs"] = [s.to_dict() for s in specs]
    summary["excluded_slice_ids"] = (
        []
        if getattr(lm_hp, "coe_reference_slice", None) is None
        else [int(getattr(lm_hp, "coe_reference_slice"))]
    )
    if not specs:
        return _witness_unavailable(
            "No independent CoE witness slices fit inside the dataset.",
            "legacy-coe-stageA-overlap-split-no-valid-slices",
        )

    def _slice_loader(start: int, stop: int):
        if start < 0 or stop <= start or stop > int(y_all.shape[0]):
            raise ValueError(
                f"slice rows [{start}, {stop}) outside dataset with {int(y_all.shape[0])} rows"
            )
        x_slice = _np.array(X_all[start:stop], dtype=_np.float64, copy=True)
        y_slice = _np.array(y_all[start:stop], dtype=_np.float64, copy=True).reshape(-1, 1)
        if y_op is not None:
            y_slice = _np.array(y_op(y_slice), dtype=_np.float64, copy=True).reshape(-1, 1)
            if not _np.all(_np.isfinite(y_slice)):
                raise ValueError(f"non-finite y-transform target values on rows [{start}, {stop})")
        xb = torch.as_tensor(x_slice, dtype=dtype)
        yb = torch.as_tensor(y_slice, dtype=dtype)
        batch_size = int(
            getattr(data_hp, "batch_size", 0)
            or getattr(lm_hp, "coe_ndata_val", 0)
            or xb.shape[0]
        )
        batch_size = max(1, min(batch_size, int(xb.shape[0])))
        return DataLoader(TensorDataset(xb, yb), batch_size=batch_size, shuffle=False, drop_last=False)

    def _compare_loss(model_i, val_loader):
        if model_i is None:
            return float("inf")
        if y_op_inv is not None:
            from nestynet_sr.sr_search.stageB.evaluation import _eval_original_y_mse_with_inverse

            return float(_eval_original_y_mse_with_inverse(model_i, val_loader, device, y_op_inv))
        return float(_eval_yspace_mse(model_i, val_loader, device))

    noise_floor_raw = float(getattr(lm_hp, "coe_noise_floor_raw", nf) or nf or 0.0)
    noise_mult = float(getattr(lm_hp, "coe_noise_mult", 3.0) or 3.0)
    rel_tol = float(getattr(lm_hp, "coe_rel_tol", 1.0e-3) or 1.0e-3)
    executor = CoEWitnessExecutor.from_config(lm_hp)

    def _worker(job) -> dict:
        spec = job.payload
        row = {
            "method": "transfer_compare",
            "slice_id": int(spec.slice_id),
            "train_rows": [int(spec.train_start), int(spec.train_stop)],
            "val_rows": [int(spec.val_start), int(spec.val_stop)],
            "status": "error",
        }
        try:
            val_loader = _slice_loader(int(spec.val_start), int(spec.val_stop))
            base_loss = float(_compare_loss(base_model, val_loader))
            cand_loss = float(_compare_loss(cand_model, val_loader))
            if not (
                math.isfinite(base_loss)
                and math.isfinite(cand_loss)
                and base_loss >= 0.0
                and cand_loss >= 0.0
            ):
                raise RuntimeError(
                    f"non-finite compare loss base={base_loss}, candidate={cand_loss}"
                )
            tol = _committee_tolerance(
                loss_a=base_loss,
                loss_b=cand_loss,
                noise_floor_raw=noise_floor_raw,
                n_eff=max(1, int(spec.val_stop - spec.val_start)),
                noise_mult=noise_mult,
                rel_tol=rel_tol,
            )
            delta = cand_loss - base_loss
            if delta < -tol:
                vote = "win"
            elif delta > tol:
                vote = "loss"
            else:
                vote = "tie"
            row.update(
                {
                    "status": "success",
                    "incumbent_compare_loss": base_loss,
                    "candidate_compare_loss": cand_loss,
                    "comparison_space": "raw_y" if y_op_inv is not None else "model_output_space",
                    "delta": float(delta),
                    "tolerance": float(tol),
                    "vote": vote,
                    "n_val": int(spec.val_stop - spec.val_start),
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    rows = run_threaded_witnesses(
        coe_witness_jobs_from_specs(specs, prefix=f"stageA_overlap_{split_kind_s}"),
        _worker,
        executor=executor,
    )
    paired = [row for row in rows if row.get("status") == "success"]
    invalid = int(len(rows) - len(paired))

    wins = sum(1 for r in paired if r.get("vote") == "win")
    ties = sum(1 for r in paired if r.get("vote") == "tie")
    losses = sum(1 for r in paired if r.get("vote") == "loss")
    inc_losses = [float(r["incumbent_compare_loss"]) for r in paired]
    cand_losses = [float(r["candidate_compare_loss"]) for r in paired]
    inc_med = float(_np.median(_np.asarray(inc_losses, dtype=float))) if inc_losses else float("inf")
    cand_med = float(_np.median(_np.asarray(cand_losses, dtype=float))) if cand_losses else float("inf")
    med_tol = _committee_tolerance(
        loss_a=inc_med,
        loss_b=cand_med,
        noise_floor_raw=noise_floor_raw,
        n_eff=max(1, len(paired) * int(getattr(lm_hp, "coe_ndata_val", 2000) or 2000)),
        noise_mult=noise_mult,
        rel_tol=rel_tol,
    )
    summary.update(
        {
            "gate_status": "evaluated",
            "decision": "evaluate",
            "n_slices": int(len(specs)),
            "reference_slice": getattr(lm_hp, "coe_reference_slice", None),
            "evaluated_slices": int(len(rows)),
            "n_paired_success": int(len(paired)),
            "invalid": int(invalid),
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "incumbent_median_mse": float(inc_med),
            "candidate_median_mse": float(cand_med),
            "median_delta": float(cand_med - inc_med),
            "median_tolerance": float(med_tol),
            "comparison_space": "raw_y" if y_op_inv is not None else "model_output_space",
            "witness_executor": coe_witness_execution_metadata(executor, rows),
            "results": rows,
        }
    )
    if int(summary["n_paired_success"]) <= 0:
        if budgeted_witnesses_required:
            summary["gate_status"] = "veto"
            summary["decision"] = "veto"
            summary["reason"] = "No valid witness comparison was available for budgeted admission."
            return False, "reject-coe-stageA-overlap-split-budget-witness-unavailable", summary
        summary["gate_status"] = "legacy_allow"
        summary["decision"] = "allow"
        return True, "coe-stageA-overlap-split-gate-unsupported: no paired witness evaluations", summary
    if (
        (invalid == 0 or not budgeted_witnesses_required)
        and losses == 0
        and float(summary["median_delta"]) <= float(summary["median_tolerance"])
    ):
        summary["gate_status"] = "accepted"
        summary["decision"] = "allow"
        return True, (
            "coe-stageA-overlap-split-gate-accepted "
            f"(wins/ties/losses={wins}/{ties}/{losses}, "
            f"median_delta={summary['median_delta']:.3e}, tol={summary['median_tolerance']:.3e})"
        ), summary

    budgeted_admission = _stageA_budgeted_witness_admission(
        rows,
        base_loss_key="incumbent_compare_loss",
        candidate_loss_key="candidate_compare_loss",
        budget_multiplier=structural_budget_multiplier,
        noise_floor=noise_floor_raw,
    )
    summary["budgeted_witness_admission"] = budgeted_admission
    if (
        budgeted_witnesses_required
        and budgeted_admission["all_slices_within_budget"]
    ):
        summary["gate_status"] = "accepted_provisional"
        summary["decision"] = "allow_provisional"
        summary["provisional_budget_admission"] = True
        return True, (
            "provisional-budget-coe-stageA-overlap-split-gate "
            f"(wins/ties/losses={wins}/{ties}/{losses}, "
            f"budget={float(structural_budget_multiplier):.3g}x, "
            f"max_ratio={float(budgeted_admission['max_excess_ratio']):.3g}x)"
        ), summary

    summary["gate_status"] = "veto"
    summary["decision"] = "veto"
    return False, (
        "reject-coe-stageA-overlap-split-gate "
        f"(wins/ties/losses={wins}/{ties}/{losses}, "
        f"median_delta={summary['median_delta']:.3e}, tol={summary['median_tolerance']:.3e})"
    ), summary


def _format_stageA_overlap_split_committee_report(summary: Optional[dict]) -> str:
    if not isinstance(summary, dict):
        return "=== CoE Stage A Overlap Split Gate ===\nenabled=False"
    lines = ["=== CoE Stage A Overlap Split Gate ==="]
    lines.append(
        f"enabled={bool(summary.get('enabled', False))} "
        f"status={summary.get('gate_status', 'unknown')} "
        f"decision={summary.get('decision', 'unknown')} "
        f"split={summary.get('split_kind')} "
        f"risks={','.join(str(x) for x in summary.get('risk_tags', []) or [])}"
    )
    try:
        lines.append(
            f"median_delta={float(summary.get('median_delta')):.6e}, "
            f"tol={float(summary.get('median_tolerance')):.6e}, "
            f"W/T/L={int(summary.get('wins', 0))}/{int(summary.get('ties', 0))}/"
            f"{int(summary.get('losses', 0))}"
        )
    except Exception:
        pass
    if summary.get("reason"):
        lines.append(str(summary.get("reason")))
    return "\n".join(lines)


def _stageA_noisy_overlap_split_gate(
    *,
    split_kind: str,
    has_overlap: bool,
    base_ast: Any,
    cand_ast: Any,
    base_val_loss: Optional[float],
    cand_val_loss: Optional[float],
    noise_floor: float,
    n_eff: Optional[float] = None,
) -> tuple[bool, str]:
    """Require a genuine Pareto gain before accepting a noisy overlap split.

    Overlapping decompositions can fit the same noisy target while increasing
    the unresolved NN burden.  Such sideways moves make later analytical
    rewriting harder (and multiplicative overlaps additionally have a gauge
    ambiguity), so require either a statistically material validation gain or,
    for additive splits, a strict NN-arity simplification at equivalent loss.
    """
    kind = str(split_kind).strip().lower()
    kind_label = "multiplicative" if kind == "mul" else "additive"
    if not has_overlap:
        return True, "not an overlapping split"
    if kind not in {"add", "mul"}:
        return False, f"unknown overlapping split kind {split_kind!r}"
    try:
        nf = float(noise_floor)
    except (TypeError, ValueError):
        nf = 0.0
    if (not math.isfinite(nf)) or nf <= 0.0:
        return True, "no positive noise floor"

    try:
        base_v = float(base_val_loss)
        cand_v = float(cand_val_loss)
    except (TypeError, ValueError):
        return False, "missing validation loss"
    if (
        (not math.isfinite(base_v))
        or (not math.isfinite(cand_v))
        or base_v < 0.0
        or cand_v < 0.0
    ):
        return False, "non-finite validation loss"

    try:
        n_eff_f = float(n_eff) if n_eff is not None else None
        if n_eff_f is not None and ((not math.isfinite(n_eff_f)) or n_eff_f <= 0.0):
            n_eff_f = None
    except Exception:
        n_eff_f = None

    tol = _noise_equivalence_tolerance(
        cand_v,
        base_v,
        noise_floor=nf,
        n_eff=n_eff_f,
        noise_mult=3.0,
        rel_tol=1.0e-3,
        abs_floor=1.0e-14 * max(1.0, abs(base_v), abs(cand_v)),
    )
    if cand_v < base_v - tol:
        return True, (
            f"overlapping {kind_label} split materially improves validation "
            f"{base_v:.4e}->{cand_v:.4e} beyond tol={tol:.3e}"
        )

    structurally_simpler = _is_confirmed_nn_split_simplification(base_ast, cand_ast)
    if kind == "add" and structurally_simpler and cand_v <= base_v + tol:
        return True, (
            "overlapping additive split strictly simplifies unresolved NN arity "
            f"{_nn_split_signature(base_ast)}->{_nn_split_signature(cand_ast)} "
            f"at noise-equivalent validation loss (tol={tol:.3e})"
        )

    structural_reason = (
        "multiplicative overlap requires a material loss improvement"
        if kind == "mul"
        else (
            "validation loss is materially worse"
            if structurally_simpler
            else (
                "unresolved NN arity is not strictly simpler "
                f"({_nn_split_signature(base_ast)}->{_nn_split_signature(cand_ast)})"
            )
        )
    )
    return False, (
        f"noisy overlapping {kind_label} split rejected: {structural_reason}; "
        f"val {base_v:.4e}->{cand_v:.4e}, tol={tol:.3e}"
    )


def _stageA_noisy_overlap_mul_split_gate(
    *,
    is_multiplicative: bool,
    has_overlap: bool,
    base_val_loss: Optional[float],
    cand_val_loss: Optional[float],
    noise_floor: float,
    n_eff: Optional[float] = None,
) -> tuple[bool, str]:
    """Compatibility wrapper for the original multiplicative-only guard."""
    if not is_multiplicative:
        return True, "not an overlapping multiplicative split"
    return _stageA_noisy_overlap_split_gate(
        split_kind="mul",
        has_overlap=has_overlap,
        base_ast=None,
        cand_ast=None,
        base_val_loss=base_val_loss,
        cand_val_loss=cand_val_loss,
        noise_floor=noise_floor,
        n_eff=n_eff,
    )


def stageA_analyze(
    *,
    i_op,
    y_op,
    y_op_inv,
    candidate_sep_ops,
    y_transform_names=None,
    initial_ast=None,
    filepath=None,
    Nxvars=None,
    y_med=None,
    y_mad=None,
    np_dtype=None,
    dtype=None,
    device=None,
    data_hp=None,
    model_hp=None,
    lm_hp=None,
    search_hp=None,
    leaf_builder=None,
    model_output=None,
    model_sep_output=None,
    mode="full",
    units_payload=None,
    enforce_units: bool = False,
    units_policy: str = "free_const_only",
    nn_units_semantics: str = "unknown",
    y_log_dynamic_range: float = None,
    y_abs_median: float = None,
    global_best_val_loss_base: Optional[float] = None,
    reuse_leaves_init: dict = None,
    freeze_non_nn: bool = False,
    skip_initial_fit: bool = False,
    y_raw_full=None,
    noise_sigma_y: Optional[float] = None,
    noise_floor_mc_samples: int = 8,
    fast: bool = False,
) -> StageAOut:
    """Callable Stage A entrypoint for one y-transform/state.

    This is a lightweight extraction over ``run_separability_for_transform`` to
    make y-search/controller orchestration explicit and reusable.
    """
    run_mode = "quick" if bool(fast) else str(mode)
    (
        success,
        model,
        rest_add,
        rest_mult,
        candidate_sep_ops_out,
        current_ast,
        last_resort_suggested,
        full_compound_solved,
    ) = run_separability_for_transform(
        i_op=i_op,
        y_op=y_op,
        y_op_inv=y_op_inv,
        candidate_sep_ops=candidate_sep_ops,
        y_transform_names=y_transform_names,
        initial_ast=initial_ast,
        filepath=filepath,
        Nxvars=Nxvars,
        y_med=y_med,
        y_mad=y_mad,
        np_dtype=np_dtype,
        dtype=dtype,
        device=device,
        data_hp=data_hp,
        model_hp=model_hp,
        lm_hp=lm_hp,
        search_hp=search_hp,
        leaf_builder=leaf_builder,
        model_output=model_output,
        model_sep_output=model_sep_output,
        mode=run_mode,
        units_payload=units_payload,
        enforce_units=enforce_units,
        units_policy=units_policy,
        nn_units_semantics=nn_units_semantics,
        y_log_dynamic_range=y_log_dynamic_range,
        y_abs_median=y_abs_median,
        global_best_val_loss_base=global_best_val_loss_base,
        reuse_leaves_init=reuse_leaves_init,
        freeze_non_nn=freeze_non_nn,
        skip_initial_fit=skip_initial_fit,
        y_raw_full=y_raw_full,
        noise_sigma_y=noise_sigma_y,
        noise_floor_mc_samples=noise_floor_mc_samples,
    )

    y_name = None
    try:
        if y_transform_names is not None and i_op is not None:
            idx = int(i_op)
            if 0 <= idx < len(y_transform_names):
                y_name = str(y_transform_names[idx])
    except Exception:
        y_name = None
    if not y_name:
        y_name = "identity" if (y_op is None) else str(getattr(y_op, "__name__", y_op))

    val_loss_base = float("inf")
    if model is not None:
        try:
            bvlb = getattr(model, "_best_val_loss_base", None)
            if bvlb is not None and math.isfinite(float(bvlb)):
                val_loss_base = float(bvlb)
        except Exception:
            val_loss_base = float("inf")

    signals: Dict[str, float] = {}
    if model is not None:
        try:
            raw_signals = getattr(model, "_stageA_signals", None)
            if isinstance(raw_signals, dict):
                for k, v in raw_signals.items():
                    try:
                        signals[str(k)] = float(v)
                    except Exception:
                        continue
        except Exception:
            signals = {}
    split_success = bool(success) and ((rest_add is not None) or (rest_mult is not None))
    if not signals:
        sep_score = 1.0 if split_success else 0.0
        try:
            idx = int(i_op)
            if 0 <= idx < len(candidate_sep_ops_out) and bool(candidate_sep_ops_out[idx]):
                sep_score = max(sep_score, 0.55)
        except Exception:
            pass
        signals = {
            "trig_affine_conf": 0.0,
            "sep_score": float(sep_score),
            "best_split_score": float(1.0 if split_success else 0.0),
            "split_success": float(1.0 if split_success else 0.0),
            "full_compound_compressed": float(1.0 if bool(full_compound_solved) else 0.0),
            "full_compound_solved": 0.0,
            "sep_candidates_seen": 0.0,
            "split_accept_count": float(1.0 if split_success else 0.0),
        }
    if "full_compound_compressed" not in signals:
        signals["full_compound_compressed"] = float(1.0 if bool(full_compound_solved) else 0.0)
    # Keep the legacy key present, but do not let NN[z] compression masquerade
    # as a solved symbolic relation.
    signals["full_compound_solved"] = 0.0

    def _freeze_partition(obj):
        if isinstance(obj, tuple):
            return tuple(_freeze_partition(x) for x in obj)
        if isinstance(obj, list):
            return tuple(_freeze_partition(x) for x in obj)
        if isinstance(obj, set):
            return tuple(sorted(_freeze_partition(x) for x in obj))
        try:
            return int(obj)
        except Exception:
            pass
        if isinstance(obj, (str, float, bool)) or obj is None:
            return obj
        return str(obj)

    split_score_signal = 1.0 if split_success else 0.0
    try:
        s = signals.get("best_split_score", split_score_signal)
        split_score_signal = float(s) if math.isfinite(float(s)) else float(split_score_signal)
    except Exception:
        split_score_signal = float(split_score_signal)

    split_plans: List[SplitPlan] = []
    if rest_add is not None:
        add_part = _freeze_partition(rest_add)
        if not isinstance(add_part, tuple):
            add_part = (add_part,)
        split_plans.append(
            SplitPlan(
                kind="add",
                partition=add_part,
                score=float(split_score_signal),
                details={"source": "stageA", "y_transform": str(y_name)},
            )
        )
    if rest_mult is not None:
        mult_part = _freeze_partition(rest_mult)
        if not isinstance(mult_part, tuple):
            mult_part = (mult_part,)
        split_plans.append(
            SplitPlan(
                kind="mul",
                partition=mult_part,
                score=float(split_score_signal),
                details={"source": "stageA", "y_transform": str(y_name)},
            )
        )

    move_records: List[Dict[str, Any]] = []
    if model is not None:
        try:
            raw_records = getattr(model, "_stageA_move_records", None)
            if isinstance(raw_records, list):
                move_records = [dict(r) for r in raw_records if isinstance(r, dict)]
        except Exception:
            move_records = []

    return StageAOut(
        success=bool(success),
        model=model,
        rest_add=rest_add,
        rest_mult=rest_mult,
        candidate_sep_ops=list(candidate_sep_ops_out),
        current_ast=current_ast,
        last_resort_suggested=bool(last_resort_suggested),
        full_compound_solved=bool(full_compound_solved),
        y_transform_name=str(y_name),
        val_loss_base=float(val_loss_base),
        signals=signals,
        split_plans=split_plans,
        move_records=move_records,
    )

__search_definitions__ = (
    "_iter_limited_batches",
    "_candidate_metric",
    "_sep_metric_to_score",
    "_is_clean_disjoint_cover",
    "_is_singleton_split",
    "_is_singleton_disjoint_cover",
    "_stageA_best_disjoint_separability_metric",
    "StageAOut",
    "_stageA_provisional_move_reason",
    "_stageA_provisional_full_refit_failure_status",
    "SplitPlan",
    "_nn_split_signature",
    "_is_confirmed_nn_split_simplification",
    "_accept_threshold_with_structural_target",
    "_stageA_under_protest_threshold_cap",
    "_stageA_loss_budget_multiplier",
    "_stageA_budgeted_witness_admission",
    "_stageA_leaf_projection_nonregression_override",
    "_stageA_ast_raw_var_set",
    "_stageA_leaf_prune_acceptance_gate",
    "_stageA_snapshot_rng_state",
    "_stageA_restore_rng_state",
    "_stageA_destructive_prune_committee_gate",
    "_stageA_terminal_closure_committee_gate",
    "_stageA_compound_pattern_support_size",
    "_stageA_compound_structural_priority",
    "_stageA_compound_is_structurally_protected",
    "_stageA_compound_shortlist_committee_rank",
    "_format_stageA_compound_shortlist_committee_report",
    "_stageA_overlap_split_committee_gate",
    "_format_stageA_overlap_split_committee_report",
    "_stageA_noisy_overlap_split_gate",
    "_stageA_noisy_overlap_mul_split_gate",
    "stageA_analyze",
)

__search_constants__ = (

)

__search_late_bindings__ = (
    "run_separability_for_transform",
)
